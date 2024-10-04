import Mathlib
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Algebra.PrimeNums
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Module
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Odd
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.ArcTan
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.NumberTheory.Primes
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.Notation
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Topology.Circles
import Mathlib.Topology.Instances.Real

namespace absent_minded_scientist_mistake_l731_731734

theorem absent_minded_scientist_mistake (ξ η : ℝ) (h₁ : E ξ = 3) (h₂ : E η = 5) (h₃ : E (min ξ η) = 3 + 2/3) : false :=
by
  sorry

end absent_minded_scientist_mistake_l731_731734


namespace coins_after_five_hours_l731_731781

-- Definitions of the conditions
def first_hour : ℕ := 20
def next_two_hours : ℕ := 2 * 30
def fourth_hour : ℕ := 40
def fifth_hour : ℕ := -20

-- The total number of coins calculation
def total_coins : ℕ := first_hour + next_two_hours + fourth_hour + fifth_hour

-- The theorem to be proved
theorem coins_after_five_hours : total_coins = 100 :=
by
  sorry

end coins_after_five_hours_l731_731781


namespace sector_to_circle_ratio_l731_731267

noncomputable def ratio_sector_inscribed_circle (r : ℝ) : ℝ :=
  let θ := 22.5 * (Real.pi / 180) in
  let s := π * r^2 / 8 in
  let c := π * (r * Real.sin θ)^2 in
  s / c

theorem sector_to_circle_ratio :
  ratio_sector_inscribed_circle 2 = 1 / (8 * (Real.sin (22.5 * (Real.pi / 180)))^2) :=
by 
  sorry

end sector_to_circle_ratio_l731_731267


namespace solve_inequality_l731_731381

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731381


namespace odd_product_less_than_20000_l731_731863

theorem odd_product_less_than_20000 :
  (∏ i in finset.filter (λ x, x % 2 = 1) (finset.range 20000), i) =
  (20000! / (2 ^ 10000 * 10000!)) :=
sorry

end odd_product_less_than_20000_l731_731863


namespace sum_of_valid_n_l731_731866

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l731_731866


namespace number_of_shirts_is_20_l731_731638

/-- Given the conditions:
1. The total price for some shirts is 360,
2. The total price for 45 sweaters is 900,
3. The average price of a sweater exceeds that of a shirt by 2,
prove that the number of shirts is 20. -/

theorem number_of_shirts_is_20
  (S : ℕ) (P_shirt P_sweater : ℝ)
  (h1 : S * P_shirt = 360)
  (h2 : 45 * P_sweater = 900)
  (h3 : P_sweater = P_shirt + 2) :
  S = 20 :=
by
  sorry

end number_of_shirts_is_20_l731_731638


namespace bush_height_at_2_years_l731_731468

theorem bush_height_at_2_years (H: ℕ → ℕ) 
  (quadruple_height: ∀ (n: ℕ), H (n+1) = 4 * H n)
  (H_4: H 4 = 64) : H 2 = 4 :=
by
  sorry

end bush_height_at_2_years_l731_731468


namespace transformed_triangle_area_l731_731281

structure Point where
  x : ℝ
  y : ℝ

def transform_point (M : Matrix (Fin 2) (Fin 2) ℝ) (p : Point) : Point :=
  ⟨M 0 0 * p.x + M 0 1 * p.y, M 1 0 * p.x + M 1 1 * p.y⟩

def area_of_triangle (p1 p2 p3 : Point) : ℝ :=
  (1 / 2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨3, 0⟩
def C : Point := ⟨2, 2⟩

def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 2]]

def N : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0], ![0, 1]]

def T : Matrix (Fin 2) (Fin 2) ℝ := N.mul M

def A' : Point := transform_point T A
def B' : Point := transform_point T B
def C' : Point := transform_point T C

theorem transformed_triangle_area : area_of_triangle A' B' C' = 12 := by
  sorry

end transformed_triangle_area_l731_731281


namespace sum_of_integer_n_l731_731897

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l731_731897


namespace measure_4_minutes_with_hourglasses_l731_731055

/-- Prove that it is possible to measure exactly 4 minutes using hourglasses of 9 minutes and 7 minutes and the minimum total time required is 18 minutes -/
theorem measure_4_minutes_with_hourglasses : 
  ∃ (a b : ℕ), (9 * a - 7 * b = 4) ∧ (a + b) * 1 ≤ 2 ∧ (a * 9 ≤ 18 ∧ b * 7 <= 18) :=
by {
  sorry
}

end measure_4_minutes_with_hourglasses_l731_731055


namespace recycling_weight_l731_731318

theorem recycling_weight :
  let marcus_milk_bottles := 25
  let john_milk_bottles := 20
  let sophia_milk_bottles := 15
  let marcus_cans := 30
  let john_cans := 25
  let sophia_cans := 35
  let milk_bottle_weight := 0.5
  let can_weight := 0.025

  let total_milk_bottles_weight := (marcus_milk_bottles + john_milk_bottles + sophia_milk_bottles) * milk_bottle_weight
  let total_cans_weight := (marcus_cans + john_cans + sophia_cans) * can_weight
  let combined_weight := total_milk_bottles_weight + total_cans_weight

  combined_weight = 32.25 :=
by
  sorry

end recycling_weight_l731_731318


namespace min_ab_is_2sqrt6_l731_731186

noncomputable def min_ab (a b : ℝ) : ℝ :=
  if h : (a > 0) ∧ (b > 0) ∧ ((2 / a) + (3 / b) = Real.sqrt (a * b)) then
      2 * Real.sqrt 6
  else
      0 -- or any other value, since this case should not occur in the context

theorem min_ab_is_2sqrt6 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : (2 / a) + (3 / b) = Real.sqrt (a * b)) :
  min_ab a b = 2 * Real.sqrt 6 := 
by
  sorry

end min_ab_is_2sqrt6_l731_731186


namespace david_marks_in_english_l731_731141

-- Definitions from the problem conditions:
def marks_mathematics : ℕ := 65
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def number_of_subjects : ℕ := 5
def average_marks : ℚ := 75

-- Define David's marks in English as a variable:
variable (marks_english : ℕ)

-- The math proof statement to show:
theorem david_marks_in_english
  (h_avg : average_marks = (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / number_of_subjects)
  : marks_english = 76 :=
begin
  sorry -- Proof goes here
end

end david_marks_in_english_l731_731141


namespace last_digit_of_power_tower_l731_731514

theorem last_digit_of_power_tower : 
  let n : ℕ := (3 ^ 3) ^ (3 ^ (3 ^ 3)) ^ (3 ^ (3 ^ (3 ^ 3))) ^ (3 ^ (3 ^ (3 ^ (3 ^ 3))))
  in n % 10 = 7 :=
by sorry

end last_digit_of_power_tower_l731_731514


namespace alok_paid_rs_811_l731_731101

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l731_731101


namespace discount_difference_l731_731399

theorem discount_difference 
  (original_price : ℝ) (fixed_discount : ℝ) (percent_discount : ℝ) :
  original_price = 30 → fixed_discount = 5 → percent_discount = 0.25 →
  let first_scenario := (original_price - fixed_discount) * (1 - percent_discount) in
  let second_scenario := original_price * (1 - percent_discount) - fixed_discount in
  (first_scenario - second_scenario) * 100 = 125 :=
by
  intros
  -- The proof is skipped as we are only interested in the statement
  sorry

end discount_difference_l731_731399


namespace solve_inequality_l731_731368

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731368


namespace sqrt_fraction_subtraction_l731_731520

theorem sqrt_fraction_subtraction :
  (Real.sqrt (9 / 2) - Real.sqrt (2 / 9)) = (7 * Real.sqrt 2 / 6) :=
by sorry

end sqrt_fraction_subtraction_l731_731520


namespace geometric_sequence_proof_l731_731223

theorem geometric_sequence_proof :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ),
    (∀ n, a (n + 1) = 4 * a n) ∧
    (a 1 + a 3 = 17 ∧ a 1 * a 3 = 16) →
    (∀ n, a n = 4 ^ (n - 1)) ∧
    (∀ n, b n = log (1 / 2) (a n) + 11) →
    (∀ n, T n =
      if 1 ≤ n ∧ n ≤ 6 then 12 * n - n ^ 2
      else if n ≥ 7 then n ^ 2 - 12 * n + 72
      else 0) :=
by
  sorry

end geometric_sequence_proof_l731_731223


namespace incorrect_calculation_l731_731715

noncomputable def ξ : ℝ := 3
noncomputable def η : ℝ := 5

def T (ξ η : ℝ) : ℝ := min ξ η

theorem incorrect_calculation : E[T(ξ, η)] ≤ 3 := by
  sorry

end incorrect_calculation_l731_731715


namespace f_prime_at_1_l731_731305

def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

theorem f_prime_at_1 : (deriv f 1) = 11 :=
by
  sorry

end f_prime_at_1_l731_731305


namespace smaller_group_men_l731_731620

theorem smaller_group_men (work_same : ∀ (a b c d : ℕ), a * b = c * d → c = (a * b) / d) :
  ∃ x : ℕ, 22 * 55 = x * 121 ∧ x = 10 :=
by
  let x := 10
  have h : 22 * 55 = x * 121 := 
    calc
      22 * 55 = 22 * (5 * 11) * 5 : by sorry
      ...     = 110 * 5 * 5       : by sorry  
      ...     = 110 * 25           : by sorry  
      ...     = 1210 * 2            : by sorry  
      ...     = 121 * (10)         : by sorry  
  use x
  exact ⟨h, rfl⟩

end smaller_group_men_l731_731620


namespace total_money_correct_l731_731120

def total_money_in_cents : ℕ :=
  let Cindy := 5 * 10 + 3 * 50
  let Eric := 3 * 25 + 2 * 100 + 1 * 50
  let Garrick := 8 * 5 + 7 * 1
  let Ivy := 60 * 1 + 5 * 25
  let TotalBeforeRemoval := Cindy + Eric + Garrick + Ivy
  let BeaumontRemoval := 2 * 10 + 3 * 5 + 10 * 1
  let EricRemoval := 1 * 25 + 1 * 50
  TotalBeforeRemoval - BeaumontRemoval - EricRemoval

theorem total_money_correct : total_money_in_cents = 637 := by
  sorry

end total_money_correct_l731_731120


namespace percent_of_percent_l731_731906

theorem percent_of_percent (a b : ℝ) (h₁ : a = 0.20) (h₂ : b = 0.25) : ((a * b) * 100) = 5 :=
by 
  rw [h₁, h₂]
  norm_num
  sorry

end percent_of_percent_l731_731906


namespace probability_sum_less_than_product_l731_731812

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731812


namespace sum_of_all_n_l731_731891

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l731_731891


namespace expand_binom_l731_731531

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l731_731531


namespace min_set_covers_handshakes_l731_731465

def CircularHandshakes := {i : Fin 36}

def Neighbors (x : CircularHandshakes) : Set CircularHandshakes :=
  {y | y = (x + 1) % 36 ∨ y = (x - 1) % 36}

theorem min_set_covers_handshakes :
  ∃ S : Set CircularHandshakes, S.card = 2 ∧ ∀ x ∈ {i : Fin 36}, ∃ y ∈ S, y ∈ Neighbors x :=
sorry

end min_set_covers_handshakes_l731_731465


namespace salary_increase_l731_731398

theorem salary_increase (x : ℝ) (y : ℝ) :
  (1000 : ℝ) * 80 + 50 = y → y - (50 + 80 * x) = 80 :=
by
  intros h
  sorry

end salary_increase_l731_731398


namespace problem1_problem2_problem3_l731_731593

-- Definitions
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Math.log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) : ℝ := f x m + g x m

-- Problem 1: Interval of monotonic increase for f(x)
theorem problem1 (x : ℝ) : f x (1 / 2) > 0 → 0 < x ∧ x < 1 :=
sorry

-- Problem 2: Minimum integer m such that F(x) <= mx - 1
theorem problem2 : ∀ x : ℝ, F x 2 ≤ 2 * x - 1 :=
sorry

-- Problem 3: Prove x₁ + x₂ ≥ (√5 - 1) / 2 when m = -2 
theorem problem3 (x₁ x₂ : ℝ) (h₁ : F x₁ (-2) + F x₂ (-2) + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end problem1_problem2_problem3_l731_731593


namespace _l731_731279

noncomputable def right_triangle_BD_length : ℝ :=
  let AB := 1
  let BC := 3
  let AC := Real.sqrt (AB ^ 2 + BC ^ 2) -- Implementing Pythagorean theorem
  let x := (1 / 4) * 3 -- Angle bisector theorem application result
  x

example (BD : ℝ) (ABC : Type) [IsRightTriangle ABC] 
  (hB : Angle B = 90) 
  (hAB : length AB = 1) 
  (hBC : length BC = 3) 
  (hAD : Bisection (Angle A) D BC) : 
  BD = right_triangle_BD_length :=
  sorry

end _l731_731279


namespace sum_S100_l731_731240

def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then n - 1 else n

def S (N : ℕ) : ℕ := ∑ n in Finset.range N.succ, a_n n

theorem sum_S100 : S 100 = 5000 := by
  sorry

end sum_S100_l731_731240


namespace average_speed_is_70_l731_731765

theorem average_speed_is_70 
  (distance1 distance2 : ℕ) (time1 time2 : ℕ)
  (h1 : distance1 = 80) (h2 : distance2 = 60)
  (h3 : time1 = 1) (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 70 := 
by 
  sorry

end average_speed_is_70_l731_731765


namespace area_of_triangle_PQR_l731_731108

/-!
# Problem Statement
An equilateral triangle PQR has an inscribed circle with center O. The area of the circle is 16π square centimeters. Prove that the area of the triangle PQR is 32√3 square centimeters.
-/

def inscribed_circle_area : ℝ := 16 * Real.pi

def radius (r : ℝ) : Prop := r ^ 2 = 16

def equilateral_triangle_area (s : ℝ) : ℝ := (s ^ 2 * Mathlib.sqrt 3) / 4

theorem area_of_triangle_PQR (A : ℝ) (r : ℝ) (s : ℝ) (h₁ : A = inscribed_circle_area) (h₂ : radius r) (h₃ : s = 2 * r * Mathlib.sqrt 3) :
  equilateral_triangle_area s = 32 * Mathlib.sqrt 3 :=
by
  sorry

end area_of_triangle_PQR_l731_731108


namespace angle_sum_property_l731_731081

variables (A B C D O : Type) 
variables [inner_product_space ℝ (Euclidean3)] 
variables (a b c d o : (Euclidean3) A)

-- Assuming points A, B, C, D are on the circle centered at O
def quadrilateral_circumscribed_around_circle (A B C D O : Euclidean3) : Prop :=
  ∃ r, dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r 

-- The angles subtended by circle arcs
def angle_subtended (A B O : Euclidean3) : ℝ := sorry

theorem angle_sum_property
  (circumscribed : quadrilateral_circumscribed_around_circle A B C D O)
  : angle_subtended B O C + angle_subtended D O A = angle_subtended A O B + angle_subtended C O D :=
sorry

end angle_sum_property_l731_731081


namespace closest_integer_to_cube_root_of_200_l731_731032

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), n = 6 ∧ (n^3 = 216 ∨ n^3 > 125 ∧ n^3 < 216) := 
by
  existsi 6
  split
  · refl
  · right
    split
    · norm_num
    · norm_num

end closest_integer_to_cube_root_of_200_l731_731032


namespace average_of_five_digits_l731_731393

theorem average_of_five_digits 
  (S : ℝ)
  (S3 : ℝ)
  (h_avg8 : S / 8 = 20)
  (h_avg3 : S3 / 3 = 33.333333333333336) :
  (S - S3) / 5 = 12 := 
by
  sorry

end average_of_five_digits_l731_731393


namespace log_eight_of_five_twelve_l731_731995

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731995


namespace total_profit_l731_731092

theorem total_profit 
  (A B C : ℝ)                           -- Subscriptions by A, B, and C
  (total_subscription : ℝ := 50000)      -- Total subscription amount
  (A_subs_more_B : ℝ := 4000)            -- A subscribes Rs. 4000 more than B
  (B_subs_more_C : ℝ := 5000)            -- B subscribes Rs. 5000 more than C
  (A_profit : ℝ := 15120)                -- A's profit received
  (profit_ratio_A : ℝ := 21)             -- A's share ratio
  (total_ratio : ℝ := 50)               -- Total ratio
  (h1 : A = B + A_subs_more_B)           -- Relationship between A and B
  (h2 : B = C + B_subs_more_C)           -- Relationship between B and C
  (h3 : A + B + C = total_subscription)  -- Total subscription constraint
  (h4 : A_profit / 36_000 = profit_ratio_A / total_ratio)  -- A's profit ratio
  : total_profit := 36_000 :=
begin
  sorry,
end

end total_profit_l731_731092


namespace sum_of_valid_n_l731_731879

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l731_731879


namespace find_volume_V_l731_731058

def satisfies_condition (x y z : ℝ) (t : ℝ) :=
  0 ≤ x * t^2 + y * t + z ∧ x * t^2 + y * t + z ≤ 1

def V (x y z : ℝ) :=
  ∀ t ∈ set.Icc (0 : ℝ) (1 : ℝ), satisfies_condition x y z t

noncomputable def volume_V :=
  ∫ x in set.Icc (0 : ℝ) (1 : ℝ), ∫ y in set.Icc (-2 * x) (0 : ℝ), ∫ z in set.Icc 0 (1 : ℝ), if V x y z then 1 else 0

theorem find_volume_V : volume_V = 17 / 18 :=
  sorry

end find_volume_V_l731_731058


namespace part1_part2_l731_731405

-- Definitions and conditions
def S_n (a_n : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a_n n - 3 * n

def a_n (n : ℕ) : ℕ := 3 * 2^n - 3

def b_n (a_n : ℕ → ℕ) (n : ℕ) : ℕ := a_n n + 3

-- Problem 1: Prove b_n is a geometric sequence and find general formula for a_n
theorem part1 
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (h1 : ∀ n, S_n n = 2 * a_n n - 3 * n)
  (b_n := λ n, a_n n + 3) :
  (∀ n, b_n (n+1) = 2 * b_n n) ∧ (a_n = λ n, 3 * 2^n - 3) := 
sorry

-- Problem 2: Find the sum of the first n terms of the sequence {na_n}
theorem part2
  (a_n : ℕ → ℕ)
  {S_na : ℕ → ℕ}
  (h1 : ∀ n, a_n  n = 3 * 2^n - 3)
  (sum_na : ℕ → ℕ → ℕ)
  (sum_na := λ n a_n, (6*n-6) * 2^n + 6 - (3*n*(n+1))/2) :
  sum_na n (a_n n) = (6*n-6) * 2^n + 6 - (3*n*(n+1))/2 :=
sorry

end part1_part2_l731_731405


namespace sqrt_simplest_l731_731438

def is_simplest (sq_root : ℝ) : Prop :=
  (sq_root == real.sqrt 6)

theorem sqrt_simplest : is_simplest (real.sqrt 6) :=
by
  sorry

end sqrt_simplest_l731_731438


namespace count_perf_squares_less_than_20000_l731_731525

theorem count_perf_squares_less_than_20000 :
  ∃ (n : ℕ), (n = 35) ∧
  ∀ (a : ℕ), (∃ (b : ℕ), a^2 = (2 * b + 2)^2 - (2 * b)^2) → a^2 < 20000 :=
begin
  use 35,
  split,
  { refl },
  intros a h,
  obtain ⟨b, h1⟩ := h,
  sorry
end

end count_perf_squares_less_than_20000_l731_731525


namespace perpendicular_lines_planes_l731_731952

variables (m n : Line) (alpha beta : Plane)

-- Conditions
axiom m_perp_beta : Perpendicular m beta
axiom n_perp_beta : Perpendicular n beta
axiom n_perp_alpha : Perpendicular n alpha

-- Proof Problem
theorem perpendicular_lines_planes :
  Perpendicular m alpha :=
sorry

end perpendicular_lines_planes_l731_731952


namespace members_play_both_eq_21_l731_731051

-- Given definitions
def TotalMembers := 80
def MembersPlayBadminton := 48
def MembersPlayTennis := 46
def MembersPlayNeither := 7

-- Inclusion-Exclusion Principle application to solve the problem
def MembersPlayBoth : ℕ := MembersPlayBadminton + MembersPlayTennis - (TotalMembers - MembersPlayNeither)

-- The theorem we want to prove
theorem members_play_both_eq_21 : MembersPlayBoth = 21 :=
by
  -- skipping the proof
  sorry

end members_play_both_eq_21_l731_731051


namespace find_f_log_l731_731210

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom f_def : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2

-- Theorem to be proved
theorem find_f_log : f (Real.log 6 / Real.log (1/2)) = 1 / 2 :=
by
  sorry

end find_f_log_l731_731210


namespace solution_set_of_inequality_l731_731226

theorem solution_set_of_inequality (a b x : ℝ) :
  (2 < x ∧ x < 3) → (x^2 - a * x - b < 0) →
  (a = 5 ∧ b = -6) →
  (bx^2 - a * x - 1 > 0) → ( - (1:ℝ)/2 < x ∧ x < - (1:ℝ)/3 ) :=
begin
  sorry
end

end solution_set_of_inequality_l731_731226


namespace buildings_floors_l731_731114

theorem buildings_floors :
  let A := 4
  let B := A + 9
  let C := 5 * B - 6
  let D := 4.5 * A + (1 / 3) * C
  let E := 0.75 * (A + B + C + D)
  let F := 3 * max (D - E) 0
  in A = 4 ∧ 
     B = 13 ∧ 
     C = 59 ∧ 
     D = 37 ∧ -- Assuming D is rounded off to nearest integer
     E ≤ 100 ∧ 
     E = 84 ∧ 
     F = 0 :=
by
  unfold A B C D E F
  sorry

end buildings_floors_l731_731114


namespace cos_equation_solution_l731_731623

open Real

theorem cos_equation_solution (m : ℝ) :
  (∀ x : ℝ, 4 * cos x - cos x^2 + m - 3 = 0) ↔ (0 ≤ m ∧ m ≤ 8) := by
  sorry

end cos_equation_solution_l731_731623


namespace value_of_polynomial_l731_731188

theorem value_of_polynomial (x y : ℝ) (h : x - y = 5) : (x - y)^2 + 2 * (x - y) - 10 = 25 :=
by sorry

end value_of_polynomial_l731_731188


namespace molecular_weight_of_NH4Br_is_correct_l731_731861

-- Define atomic weights
constant atomic_weight_N : ℝ := 14.01
constant atomic_weight_H : ℝ := 1.01
constant atomic_weight_Br : ℝ := 79.90

-- Define the molecular weight calculations
def molecular_weight_NH4 : ℝ := atomic_weight_N + 4 * atomic_weight_H
def molecular_weight_Br : ℝ := atomic_weight_Br

-- Define the combined molecular weight of NH4Br
def molecular_weight_NH4Br : ℝ := molecular_weight_NH4 + molecular_weight_Br

-- theorem to prove that the molecular weight is approximately 97.95 g/mol
theorem molecular_weight_of_NH4Br_is_correct : molecular_weight_NH4Br = 97.95 := by
  sorry

end molecular_weight_of_NH4Br_is_correct_l731_731861


namespace min_value_of_geometric_sequence_l731_731192

variable {a_n : ℕ → ℝ}

noncomputable def a (n : ℕ) : ℝ := a_n n

theorem min_value_of_geometric_sequence 
  (h_pos : ∀ n : ℕ, a n > 0)
  (h : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ n : ℝ, (∀ b, (2 * a 8 + a 7) ≥ b) ∧ n = 54 :=
sorry

end min_value_of_geometric_sequence_l731_731192


namespace perimeter_ratio_area_ratio_l731_731419

variable (A a : ℝ)

-- Define the properties for Triangle I (equilateral)
def P : ℝ := 3 * A
def K : ℝ := (A^2 * Real.sqrt 3) / 4

-- Define the properties for Triangle II (isosceles right)
def p : ℝ := 2 * a * (1 + Real.sqrt 2 / 2)
def k : ℝ := (1 / 2) * a^2

-- Prove the statement for perimeter ratio
theorem perimeter_ratio : P / p = 3 * A / (2 * a * (1 + Real.sqrt 2 / 2)) := sorry

-- Prove the statement for area ratio
theorem area_ratio : K / k = (A^2 * Real.sqrt 3) / (2 * a^2) := sorry

end perimeter_ratio_area_ratio_l731_731419


namespace total_number_of_athletes_l731_731507

theorem total_number_of_athletes (M F x : ℕ) (r1 r2 r3 : ℕ×ℕ) (H1 : r1 = (19, 12)) (H2 : r2 = (20, 13)) (H3 : r3 = (30, 19))
  (initial_males : M = 380 * x) (initial_females : F = 240 * x)
  (males_after_gym : M' = 390 * x) (females_after_gym : F' = 247 * x)
  (conditions : (M' - M) - (F' - F) = 30) : M' + F' = 6370 :=
by
  sorry

end total_number_of_athletes_l731_731507


namespace probability_sum_less_than_product_l731_731828

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731828


namespace sum_of_all_n_l731_731889

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l731_731889


namespace fifth_and_sixth_different_l731_731083

-- We define the possible colors
inductive Color
| Blue
| Red

open Color

-- We are given a list of 10 marbles
constant marbles : List Color
noncomputable def marbles_length : marbles.length = 10 := by sorry

-- No 3 consecutive marbles can be the same color
def no_three_consecutive (lst : List Color) : Prop :=
  ∀ i, i + 2 < lst.length → (lst.get? i ≠ lst.get? (i + 1) ∨ lst.get? (i + 1) ≠ lst.get? (i + 2))

-- There are equal numbers of blue and red marbles in the list
def equal_blue_red (lst : List Color) : Prop :=
  lst.countp (· = Blue) = lst.countp (· = Red)

-- Final theorem
theorem fifth_and_sixth_different :
  marbles.length = 10 ∧ no_three_consecutive marbles ∧ equal_blue_red marbles →
  marbles.get? 4 ≠ marbles.get? 5 :=
by sorry

end fifth_and_sixth_different_l731_731083


namespace solve_inequality_l731_731357

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l731_731357


namespace angle_CBD_l731_731668

-- Define the triangle with the given properties
structure Triangle :=
  (A B C : ℝ × ℝ)
  (is_right_angle : C.angle B A = 90)
  (is_isosceles : dist A C = dist B C)

-- Define the conditions for D and the parallel line
structure PointOnLineThroughCParallelToAB :=
  (D : ℝ × ℝ)
  (line_parallel_to_AB : ∃ m, line_through C D = y = -x)
  (equal_distances : dist A B = dist B D)
  (D_closer_to_B_than_A: dist B D < dist A D)

-- The theorem to be proved
theorem angle_CBD {ABC : Triangle} {D : PointOnLineThroughCParallelToAB} :
  ∠ CBD = 105 :=
sorry

end angle_CBD_l731_731668


namespace sum_lt_prod_probability_l731_731824

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731824


namespace multiples_of_3_never_reach_1_l731_731197

theorem multiples_of_3_never_reach_1 (n : ℕ) (hn : n > 0) :
  (∀ k : ℕ, nat.gcd n 3 = 3 → n ≠ 1) :=
by sorry

end multiples_of_3_never_reach_1_l731_731197


namespace gcd_fraction_is_integer_l731_731545

variables {k : ℕ} {a : ℕ → ℕ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ k → a i > 0)
                      (h2 : ∃ n, n = ∑ i in finset.range k, a i)
                      (h3 : ∃ d, d = nat.gcd (list.of_fn (λ i, a i)).foldr nat.gcd 0)

theorem gcd_fraction_is_integer {d n : ℕ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ k → a i > 0)
    (h2 : n = ∑ i in finset.range k, a i)
    (h3 : d = nat.gcd (list.of_fn (λ i, a i)).foldr nat.gcd 0) :
  nat.gcd (list.of_fn (λ i, a i)).foldr nat.gcd 0 ∣ 
  (∑ i in finset.range k, a i) * (∏ i in finset.range k, nat.factorial (a i) / 
  nat.factorial (∑ i in finset.range k, a i)) := 
sorry

end gcd_fraction_is_integer_l731_731545


namespace probability_sum_less_than_product_l731_731834

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731834


namespace sum_of_valid_n_l731_731868

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l731_731868


namespace approx_linear_by_odd_poly_approx_odd_cont_by_odd_poly_l731_731047

-- Part (a)
theorem approx_linear_by_odd_poly (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (λ : Fin n → ℝ), ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  abs (x - ∑ k in Finset.range n, λ ⟨k, sorry⟩ * x^(2*k+1)) < ε := sorry

-- Part (b)
theorem approx_odd_cont_by_odd_poly (f : ℝ → ℝ) (h_f : ∀ x, f (-x) = -f x) -- f is odd
  (h_cont : ContinuousOn f (Set.Icc (-1 : ℝ) 1)) (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (μ : Fin n → ℝ), ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  abs (f x - ∑ k in Finset.range n, μ ⟨k, sorry⟩ * x^(2*k+1)) < ε := sorry

end approx_linear_by_odd_poly_approx_odd_cont_by_odd_poly_l731_731047


namespace sum_of_valid_n_l731_731880

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l731_731880


namespace sum_of_positive_integers_lcm_eq_180_l731_731902

theorem sum_of_positive_integers_lcm_eq_180 :
  (∑ ν in {ν | Nat.lcm ν 45 = 180}, ν) = 260 :=
by
  sorry

end sum_of_positive_integers_lcm_eq_180_l731_731902


namespace incorrect_lifetime_calculation_l731_731725

-- Define expectation function
noncomputable def expectation (X : ℝ) : ℝ := sorry

-- We define the lifespans
variables (xi eta : ℝ)
-- Expected lifespan of the sensor and transmitter
axiom exp_xi : expectation xi = 3
axiom exp_eta : expectation eta = 5

-- Define the lifetime of the device
noncomputable def T := min xi eta

-- Given conditions
theorem incorrect_lifetime_calculation :
  expectation T ≤ 3 → 3 + (2 / 3) > 3 → false := 
sorry

end incorrect_lifetime_calculation_l731_731725


namespace quotient_when_divided_by_44_l731_731077

theorem quotient_when_divided_by_44 (N Q P : ℕ) (h1 : N = 44 * Q) (h2 : N = 35 * P + 3) : Q = 12 :=
by {
  -- Proof
  sorry
}

end quotient_when_divided_by_44_l731_731077


namespace find_f_2_l731_731553

def f : ℝ → ℝ := λ x, match x with
  | y + 1    => y^2 - 2*y
  | _        => sorry -- we'll provide specific definition only as used in the condition

theorem find_f_2 : f 2 = -1 :=
by { sorry }

end find_f_2_l731_731553


namespace triangle_base_length_l731_731391

theorem triangle_base_length :
  ∀ (base height area : ℕ), height = 4 → area = 16 → area = (base * height) / 2 → base = 8 :=
by
  intros base height area h_height h_area h_formula
  sorry

end triangle_base_length_l731_731391


namespace emily_sixth_quiz_score_l731_731161

-- Define the scores Emily has received
def scores : List ℕ := [92, 96, 87, 89, 100]

-- Define the number of quizzes
def num_quizzes : ℕ := 6

-- Define the desired average score
def desired_average : ℕ := 94

-- The theorem to prove the score Emily needs on her sixth quiz to achieve the desired average
theorem emily_sixth_quiz_score : ∃ (x : ℕ), List.sum scores + x = desired_average * num_quizzes := by
  sorry

end emily_sixth_quiz_score_l731_731161


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l731_731131

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l731_731131


namespace telescope_visual_range_increase_l731_731070

theorem telescope_visual_range_increase (original_range : ℝ) (increase_percent : ℝ) 
(h1 : original_range = 100) (h2 : increase_percent = 0.50) : 
original_range + (increase_percent * original_range) = 150 := 
sorry

end telescope_visual_range_increase_l731_731070


namespace smallest_positive_floor_l731_731672

def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x + 3 * Real.tan x

theorem smallest_positive_floor {r : ℝ} (h1 : f r = 0) (h2 : ∀ x, 0 < x → f x = 0 → x ≥ r) : 
  ∀ r, f r = 0 ∧ ∀ x, 0 < x → f x = 0 → x ≥ r → ⌊ r ⌋ = 3 :=
by
  sorry

end smallest_positive_floor_l731_731672


namespace closest_integer_to_cube_root_of_200_l731_731042

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), 
    (n = 6) ∧ (n^3 < 200) ∧ (200 < (n + 1)^3) ∧ 
    (∀ m : ℤ, (m^3 < 200) → (200 < (m + 1)^3) → (Int.abs (n - Int.ofNat (200 ^ (1/3 : ℝ)).round) < Int.abs (m - Int.ofNat (200 ^ (1/3 : ℝ)).round))) :=
begin
  sorry
end

end closest_integer_to_cube_root_of_200_l731_731042


namespace abs_fraction_inequality_solution_l731_731352

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l731_731352


namespace probability_sum_less_than_product_l731_731815

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731815


namespace sum_of_n_values_such_that_fraction_is_integer_l731_731886

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l731_731886


namespace times_older_l731_731159

-- Conditions
variables (H S : ℕ)
axiom hold_age : H = 36
axiom hold_son_relation : H = 3 * S

-- Statement of the problem
theorem times_older (H S : ℕ) (h1 : H = 36) (h2 : H = 3 * S) : (H - 8) / (S - 8) = 7 :=
by
  -- Proof will be provided here
  sorry

end times_older_l731_731159


namespace length_of_bridge_l731_731494

variable (train_length bridge_length train_speed time_to_cross : ℕ)

def train_length := 90
def time_to_cross := 36
def train_speed := 29

theorem length_of_bridge :
  train_length + bridge_length = train_speed * time_to_cross → bridge_length = 954 :=
by
  sorry

end length_of_bridge_l731_731494


namespace total_area_between_largest_smallest_circles_l731_731779

noncomputable def total_area_regions_between_circles (r1 r2 r3 : ℝ) : ℝ :=
  π * r1^2 - π * r3^2

theorem total_area_between_largest_smallest_circles :
  total_area_regions_between_circles 12 8 4 = 128 * π :=
by
  sorry

end total_area_between_largest_smallest_circles_l731_731779


namespace count_participants_with_wins_gte_two_l731_731659

noncomputable def num_participants_with_net_wins (total_players: ℕ) (winner_games: ℕ) : ℕ :=
  if total_players = 2 ^ winner_games then
    8
  else
    0

theorem count_participants_with_wins_gte_two (total_players: ℕ) (winner_games: ℕ) :
  total_players = 64 → winner_games = 6 → num_participants_with_net_wins total_players winner_games = 8 :=
by
  intros h1 h2
  simp [num_participants_with_net_wins, h1, h2]
  have n_pos : 64 = 2 ^ 6 := by norm_num
  rw [n_pos]
  norm_num

#eval count_participants_with_wins_gte_two 64 6

end count_participants_with_wins_gte_two_l731_731659


namespace C₂_symmetry_l731_731786

-- Define the initial curve C1
def C₁ (x : ℝ) : ℝ := 2 * Real.sin (x - (Real.pi / 6))

-- Define the translated and scaled curve C2
def C₂ (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 3))

-- Statement of the theorem
theorem C₂_symmetry : Symmetric_about_line (C₂, Real.pi / 2, (5 * Real.pi) / 12) := sorry

end C₂_symmetry_l731_731786


namespace another_divisor_l731_731764

theorem another_divisor (n : ℕ) (h1 : n = 44402) (h2 : ∀ d ∈ [12, 48, 74, 100], (n + 2) % d = 0) : 
  199 ∣ (n + 2) := 
by 
  sorry

end another_divisor_l731_731764


namespace total_cost_after_discounts_l731_731154

theorem total_cost_after_discounts 
    (price_iphone : ℝ)
    (discount_iphone : ℝ)
    (price_iwatch : ℝ)
    (discount_iwatch : ℝ)
    (cashback_percentage : ℝ) :
    (price_iphone = 800) →
    (discount_iphone = 0.15) →
    (price_iwatch = 300) →
    (discount_iwatch = 0.10) →
    (cashback_percentage = 0.02) →
    let discounted_iphone := price_iphone * (1 - discount_iphone),
        discounted_iwatch := price_iwatch * (1 - discount_iwatch),
        total_discounted := discounted_iphone + discounted_iwatch,
        cashback := total_discounted * cashback_percentage 
    in total_discounted - cashback = 931 :=
by {
  intros,
  sorry
}

end total_cost_after_discounts_l731_731154


namespace probability_sum_less_than_product_l731_731788

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731788


namespace closest_integer_to_cube_root_of_200_l731_731029

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), n = 6 ∧ (n^3 = 216 ∨ n^3 > 125 ∧ n^3 < 216) := 
by
  existsi 6
  split
  · refl
  · right
    split
    · norm_num
    · norm_num

end closest_integer_to_cube_root_of_200_l731_731029


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l731_731130

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l731_731130


namespace perfect_squares_count_l731_731253

theorem perfect_squares_count : 
  let n_min := Nat.ceil (sqrt 50)
  let n_max := Nat.floor (sqrt 1000)
  n_min = 8 ∧ n_max = 31 → (n_max - n_min + 1 = 24) :=
begin
  intros,
  -- step a, nonnegative integer sqrt
  have h_n_min := Nat.ceil_spec (sqrt 50),
  have h_n_max := Nat.floor_spec (sqrt 1000),
  -- step b, we can prove the floors by direct calculation
  -- n_min = 8 and n_max = 31 must be true
  have : n_min = 8 := by linarith only [Nat.ceil (sqrt 50)],
  have : n_max = 31 := by linarith only [Nat.floor (sqrt 1000)],
  exact sorry -- Proof of main statement, assuming correct bounds give 24
end

end perfect_squares_count_l731_731253


namespace incorrect_calculation_l731_731710

noncomputable def ξ : ℝ := 3 -- Expected lifetime of the sensor
noncomputable def η : ℝ := 5 -- Expected lifetime of the transmitter
noncomputable def T (ξ η : ℝ) : ℝ := min ξ η -- Lifetime of the entire device

theorem incorrect_calculation (h1 : E ξ = 3) (h2 : E η = 5) (h3 : E (min ξ η ) = 3.67) : False :=
by
  have h4 : E (min ξ η ) ≤ 3 := sorry -- Based on properties of expectation and min
  have h5 : 3.67 > 3 := by linarith -- Known inequality
  sorry

end incorrect_calculation_l731_731710


namespace prob_three_students_exactly_two_absent_l731_731630

def prob_absent : ℚ := 1 / 30
def prob_present : ℚ := 29 / 30

theorem prob_three_students_exactly_two_absent :
  (prob_absent * prob_absent * prob_present) * 3 = 29 / 9000 := by
  sorry

end prob_three_students_exactly_two_absent_l731_731630


namespace find_height_l731_731646

namespace RightTriangleProblem

variables {x h : ℝ}

-- Given the conditions described in the problem
def right_triangle_proportional (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), a = 3 * x ∧ b = 4 * x ∧ c = 5 * x

def hypotenuse (c : ℝ) : Prop := 
  c = 25

def leg (b : ℝ) : Prop :=
  b = 20

-- The theorem stating that the height h of the triangle is 12
theorem find_height (a b c : ℝ) (h : ℝ)
  (H1 : right_triangle_proportional a b c)
  (H2 : hypotenuse c)
  (H3 : leg b) :
  h = 12 :=
by
  sorry

end RightTriangleProblem

end find_height_l731_731646


namespace alok_total_payment_l731_731096

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l731_731096


namespace solve_inequality_l731_731358

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l731_731358


namespace no_such_function_exists_l731_731986

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (∀ x : ℝ, f(x^2) - (f(x))^2 ≥ 1/4) ∧ (∀ x y : ℝ, x ≠ y → f(x) ≠ f(y)) :=
by
  sorry

end no_such_function_exists_l731_731986


namespace find_polar_equation_of_circle_l731_731287

-- Define the conditions and the problem statement.
def polar_equation_of_circle (P : ℝ × ℝ) (PolarCenter : ℝ × ℝ) (C : ℝ → ℝ) : Prop :=
  P = (2 * Real.sqrt 2, Real.pi / 4) ∧ PolarCenter = (2, 0) ∧ C = (λ θ, 4 * Real.cos θ)

theorem find_polar_equation_of_circle :
  ∀ (P : ℝ × ℝ) (PolarCenter : ℝ × ℝ),
    P = (2 * Real.sqrt 2, Real.pi / 4) →
    PolarCenter = (2, 0) →
    ∃ (C : ℝ → ℝ), polar_equation_of_circle P PolarCenter (λ θ, 4 * Real.cos θ) :=
by
  intros P PolarCenter hP hC
  use (λ θ, 4 * Real.cos θ)
  exact ⟨hP, hC, rfl⟩

end find_polar_equation_of_circle_l731_731287


namespace transform_spiral_to_squares_l731_731510

theorem transform_spiral_to_squares (initial_configuration : Set ℕ) (matchsticks_to_move : Set ℕ) (new_positions : Set ℕ) :
  (∃ m1 m2 m3 m4 : ℕ, m1 ∈ matchsticks_to_move ∧ m2 ∈ matchsticks_to_move ∧ m3 ∈ matchsticks_to_move ∧ m4 ∈ matchsticks_to_move 
    ∧ m1 ≠ m2 ∧ m1 ≠ m3 ∧ m1 ≠ m4 ∧ m2 ≠ m3 ∧ m2 ≠ m4 ∧ m3 ≠ m4
    ∧ transform_configuration initial_configuration matchsticks_to_move new_positions) → 
    ∃ squares : Set (Set ℕ), three_distinct_squares squares :=
sorry

end transform_spiral_to_squares_l731_731510


namespace domain_of_sqrt_sum_l731_731168

theorem domain_of_sqrt_sum (x : ℝ) (h1 : 3 + x ≥ 0) (h2 : 1 - x ≥ 0) : -3 ≤ x ∧ x ≤ 1 := by
  sorry

end domain_of_sqrt_sum_l731_731168


namespace arcsin_sqrt_three_over_two_l731_731127

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l731_731127


namespace find_natural_n_l731_731166

theorem find_natural_n (n x y k : ℕ) (h_rel_prime : Nat.gcd x y = 1) (h_k_gt_one : k > 1) (h_eq : 3^n = x^k + y^k) :
  n = 2 := by
  sorry

end find_natural_n_l731_731166


namespace solution_set_of_inequality_l731_731225

theorem solution_set_of_inequality (a b x : ℝ) :
  (2 < x ∧ x < 3) → (x^2 - a * x - b < 0) →
  (a = 5 ∧ b = -6) →
  (bx^2 - a * x - 1 > 0) → ( - (1:ℝ)/2 < x ∧ x < - (1:ℝ)/3 ) :=
begin
  sorry
end

end solution_set_of_inequality_l731_731225


namespace tina_coins_after_five_hours_l731_731782

theorem tina_coins_after_five_hours :
  let coins_in_first_hour := 20
  let coins_in_second_hour := 30
  let coins_in_third_hour := 30
  let coins_in_fourth_hour := 40
  let coins_taken_out_in_fifth_hour := 20
  let total_coins_after_five_hours := coins_in_first_hour + coins_in_second_hour + coins_in_third_hour + coins_in_fourth_hour - coins_taken_out_in_fifth_hour
  total_coins_after_five_hours = 100 :=
by {
  sorry
}

end tina_coins_after_five_hours_l731_731782


namespace tina_coins_after_five_hours_l731_731783

theorem tina_coins_after_five_hours :
  let coins_in_first_hour := 20
  let coins_in_second_hour := 30
  let coins_in_third_hour := 30
  let coins_in_fourth_hour := 40
  let coins_taken_out_in_fifth_hour := 20
  let total_coins_after_five_hours := coins_in_first_hour + coins_in_second_hour + coins_in_third_hour + coins_in_fourth_hour - coins_taken_out_in_fifth_hour
  total_coins_after_five_hours = 100 :=
by {
  sorry
}

end tina_coins_after_five_hours_l731_731783


namespace unit_digit_sum_1_to_100_factorials_l731_731971

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def unit_digit_of_sum_of_factorials (n : ℕ) : ℕ :=
let sum := (List.range (n+1)).map factorial |>.sum
in sum % 10

theorem unit_digit_sum_1_to_100_factorials : unit_digit_of_sum_of_factorials 100 = 4 := by
  sorry

end unit_digit_sum_1_to_100_factorials_l731_731971


namespace sugar_percentage_correct_l731_731444

noncomputable def original_volume : ℝ := 340
noncomputable def original_water_percentage : ℝ := 0.64
noncomputable def original_kola_percentage : ℝ := 0.09
noncomputable def additional_sugar : ℝ := 3.2
noncomputable def additional_water : ℝ := 8
noncomputable def additional_kola : ℝ := 6.8

noncomputable def original_water := original_volume * original_water_percentage
noncomputable def original_kola := original_volume * original_kola_percentage
noncomputable def original_sugar := original_volume - original_water - original_kola

noncomputable def new_sugar := original_sugar + additional_sugar
noncomputable def new_water := original_water + additional_water
noncomputable def new_kola := original_kola + additional_kola
noncomputable def new_volume := new_sugar + new_water + new_kola

noncomputable def sugar_percentage := 100 * (new_sugar / new_volume)

theorem sugar_percentage_correct :
  sugar_percentage ≈ 26.54 :=
by
  sorry

end sugar_percentage_correct_l731_731444


namespace verify_statements_l731_731044

-- Definitions of the conditions

def direction_vector : ℝ × ℝ := (3, real.sqrt 3)
def line (x : ℝ) : ℝ := x + 1
def point : ℝ × ℝ := (0, 2)

-- Lean theorem statement verifying the correctness of the given statements
theorem verify_statements :
  (¬ (∀ (l : ℝ → ℝ), ∃ (m : ℝ), ∃ θ : ℝ, ∀ x y, (x, y) ∈ {p : ℝ × ℝ | p.2 = l p.1} → 
    ((θ ≠ real.pi / 2) → (∃ t : ℝ, (direction_vector.2 = t * direction_vector.1) → y = t*x)) ∧ 
    (θ = real.pi / 2 → (∀ s : ℝ, ((direction_vector).1 ≠ 0 → x = s )) )) ∧
  (¬ (∀ θ₁ θ₂ : ℝ, θ₁ < θ₂ → real.tan θ₁ < real.tan θ₂)) ∧
  (real.atan (direction_vector.2 / direction_vector.1) = real.pi / 6) ∧
  (¬ (∃ p' : ℝ × ℝ, p' = (1, 1) ∧ 
    (point.1 = (line point.1).1 / 2 + 1 / 2 ∧ point.2 = (point.2 + line point.1.2) / 2) ∧
    ∀ x y, (x, y) ∈ {p : ℝ × ℝ | p.2 = line p.1} → 
    ∃ m n, x^2 + y^2 = (m - n)^2 / 2 ∧ p' = (m, n)))) ∧
  (¬ (line 1 ≠ 3)) ∧
  (∀ x : ℝ, ∃ y : ℝ, y ≠ x): sorry

end verify_statements_l731_731044


namespace percentage_difference_length_shoes_l731_731076

theorem percentage_difference_length_shoes (size_min size_max : ℤ) (length_15 : ℝ) (increment : ℝ) :
  size_min = 8 → size_max = 17 → length_15 = 9.25 → increment = 0.25 →
  let length_min := length_15 - (15 - size_min) * increment in
  let length_max := length_min + (size_max - size_min) * increment in
  ((length_max - length_min) / length_min) * 100 = 30 :=
by
  intros h_min h_max h_length15 h_increment;
  simp [h_min, h_max, h_length15, h_increment];
  have length_min : ℝ := 9.25 - (15 - 8) * 0.25;
  have length_max : ℝ := length_min + (17 - 8) * 0.25;
  simp [length_min, length_max];
  sorry

end percentage_difference_length_shoes_l731_731076


namespace club_leadership_team_selection_l731_731928

theorem club_leadership_team_selection :
  let n := 20 in let k := 2 in let m := 1 in 
  (nat.choose n k) * (nat.choose (n - k) m) = 3420 :=
  by sorry

end club_leadership_team_selection_l731_731928


namespace closest_integer_to_cuberoot_of_200_l731_731018

theorem closest_integer_to_cuberoot_of_200 :
  ∃ n : ℤ, (n = 6) ∧ ( |200 - n^3| ≤ |200 - m^3|  ) ∀ m : ℤ := sorry

end closest_integer_to_cuberoot_of_200_l731_731018


namespace arcsin_sqrt3_div_2_l731_731137

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l731_731137


namespace log_eight_of_five_twelve_l731_731994

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731994


namespace solve_inequality_l731_731378

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l731_731378


namespace ordered_pairs_count_l731_731546

theorem ordered_pairs_count :
  let pairs := { (b, c) : ℕ × ℕ | b > 0 ∧ c > 0 ∧ Int.gcd b c = 4 ∧ b ^ 2 ≤ 4 * c ∧ c ^ 2 ≤ 4 * b } in
  pairs.card = 1 :=
by
  let pairs : Finset (ℕ × ℕ) := Finset.filter (λ (bc : ℕ × ℕ), 
      let b := bc.1;
      let c := bc.2;
      b > 0 ∧ c > 0 ∧ Int.gcd b c = 4 ∧ b ^ 2 ≤ 4 * c ∧ c ^ 2 ≤ 4 * b) (Finset.univ : Finset (ℕ × ℕ))
  exact Finset.card_singleton (4, 4)

end ordered_pairs_count_l731_731546


namespace largest_trifecta_sum_l731_731466

def trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a ∣ b ∧ b ∣ c ∧ c ∣ (a * b) ∧ (100 ≤ a) ∧ (a < 1000) ∧ (100 ≤ b) ∧ (b < 1000) ∧ (100 ≤ c) ∧ (c < 1000)

theorem largest_trifecta_sum : ∃ (a b c : ℕ), trifecta a b c ∧ a + b + c = 700 :=
sorry

end largest_trifecta_sum_l731_731466


namespace coprime_solution_l731_731179

theorem coprime_solution (n : ℕ) (hn : 0 < n) (p q : ℕ) (hpq_coprime : Nat.coprime p q) 
    (h_eq : p + q^2 = (n^2 + 1) * p^2 + q) : 
    (p = n + 1 ∧ q = n^2 + n + 1) :=
sorry

end coprime_solution_l731_731179


namespace calc_expr_eq_l731_731116

-- Define the polynomial and expression
def expr (x : ℝ) : ℝ := x * (x * (x * (3 - 2 * x) - 4) + 8) + 3 * x^2

theorem calc_expr_eq (x : ℝ) : expr x = -2 * x^4 + 3 * x^3 - x^2 + 8 * x := 
by
  sorry

end calc_expr_eq_l731_731116


namespace area_of_common_region_l731_731400

noncomputable def polar_circle_common_region_area (θ ρ : ℝ) : Prop :=
  ρ = -2 * Real.cos θ ∧ ρ = 2 * Real.sin θ → 
  ∃ S : ℝ, S = (π / 2 - 1)

theorem area_of_common_region (θ ρ : ℝ) :
  polar_circle_common_region_area θ ρ :=
by
  sorry

end area_of_common_region_l731_731400


namespace tan_arccos_eq_2y_l731_731775

noncomputable def y_squared : ℝ :=
  (-1 + Real.sqrt 17) / 8

theorem tan_arccos_eq_2y (y : ℝ) (hy : 0 < y) (htan : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = y_squared := sorry

end tan_arccos_eq_2y_l731_731775


namespace find_doubling_time_l731_731425

noncomputable def doubling_time (N N0 T : ℕ) : ℕ :=
let n := Nat.log 2 N in
let T_min := T * 60 in
T_min / n

theorem find_doubling_time :
  doubling_time 4096 1 4 = 20 := by
  sorry

end find_doubling_time_l731_731425


namespace absolute_sum_value_l731_731388

theorem absolute_sum_value (x1 x2 x3 x4 x5 : ℝ) 
(h : x1 + 1 = x2 + 2 ∧ x2 + 2 = x3 + 3 ∧ x3 + 3 = x4 + 4 ∧ x4 + 4 = x5 + 5 ∧ x5 + 5 = x1 + x2 + x3 + x4 + x5 + 6) :
  |(x1 + x2 + x3 + x4 + x5)| = 3.75 := 
by
  sorry

end absolute_sum_value_l731_731388


namespace solution_set_of_quadratic_l731_731227

theorem solution_set_of_quadratic (a b x : ℝ) (h1 : a = 5) (h2 : b = -6) :
  (2 ≤ x ∧ x ≤ 3) → (bx^2 - ax - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end solution_set_of_quadratic_l731_731227


namespace principal_amount_l731_731954

theorem principal_amount 
  (R : ℝ) (hR : R = 0.0375)
  (T : ℝ) (hT : T = 5)
  (total_amount : ℝ) (htotal_amount : total_amount = 950) : 
  ∃ P : ℝ, P = 800 ∧ total_amount = P + P * R * T :=
by {
  use 800,
  split,
  { refl },
  { rw [hR, hT],
    norm_num }
}

end principal_amount_l731_731954


namespace number_of_inverses_modulo_12_l731_731249

def count_inverses_modulo_12 : Nat :=
  List.length (List.filter (λ n, Nat.gcd n 12 = 1) (List.range 12))

theorem number_of_inverses_modulo_12 :
  count_inverses_modulo_12 = 4 := by sorry

end number_of_inverses_modulo_12_l731_731249


namespace work_done_by_gas_l731_731690

def gas_constant : ℝ := 8.31 -- J/(mol·K)
def temperature_change : ℝ := 100 -- K (since 100°C increase is equivalent to 100 K in Kelvin)
def moles_of_gas : ℝ := 1 -- one mole of gas

theorem work_done_by_gas :
  (1/2) * gas_constant * temperature_change = 415.5 :=
by sorry

end work_done_by_gas_l731_731690


namespace a_100_l731_731759

  noncomputable def a : ℕ → ℕ
  | 1     := 1
  | (n+1) := a n + (2 * a n) / n

  theorem a_100 : a 100 = 5151 := by
  sorry
  
end a_100_l731_731759


namespace max_value_x_minus_y_l731_731615

theorem max_value_x_minus_y (x y : ℝ) (h₀ : 0 < y) (h₁ : y ≤ x) (h₂ : x < π / 2) (h₃ : tan x = 3 * tan y) : x - y = π / 6 :=
by 
  sorry

end max_value_x_minus_y_l731_731615


namespace sum_of_integer_n_l731_731899

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l731_731899


namespace find_initial_winnings_l731_731326

-- Define conditions
variable (W : ℝ)
variable (savings : ℝ)
variable (bet : ℝ)
variable (profit : ℝ)
variable (totalSavings : ℝ)

-- Assume the initial conditions based on the problem
-- Opal's initial actions with her winnings
def initial_winnings := W / 2
def bet_initial := W / 2

-- Profit from the second bet (60% profit)
def profit_from_bet := 0.60 * (bet_initial)

-- Total amount after second bet including profit
def total_after_bet := bet_initial + profit_from_bet

-- Savings from the second bet
def savings_from_bet := total_after_bet / 2

-- Total savings including both savings from initial and second bet
def total_savings := initial_winnings + savings_from_bet

-- Problem states the total amount in savings is $90
def given_total_savings : Prop := total_savings = 90

-- The amount Opal initially won
theorem find_initial_winnings (h : given_total_savings) : W = 100 := sorry

end find_initial_winnings_l731_731326


namespace count_integer_length_chords_l731_731927

-- Define the circle with radius 12 and center O.
structure Circle where
  radius : ℝ
  center : (ℝ × ℝ)
  radius_pos : 0 < radius

-- Define a point P inside the circle such that the distance from the center to P is 5 units.
structure PointInCircle (c : Circle) where
  point : (ℝ × ℝ)
  dist_to_center : ℝ
  distance_spec : dist_to_center = Real.sqrt ((point.1 - c.center.1)^2 + (point.2 - c.center.2)^2)
  within_radius : dist_to_center < c.radius

-- Define the specific circle and point P from the problem
def specific_circle : Circle := {radius := 12, center := (0, 0), radius_pos := by norm_num}

def point_P : PointInCircle specific_circle :=
{ point := ((5, 0)),
  dist_to_center := 5,
  distance_spec := by norm_num,
  within_radius := by norm_num }

-- Define the theorem to prove the count of different chords containing point P and having integer lengths.
theorem count_integer_length_chords : ∃ (n : ℕ), n = 3 :=
sorry

end count_integer_length_chords_l731_731927


namespace log_eq_three_l731_731214

theorem log_eq_three (x y z : ℝ) 
  (hx : real.log2 (xyz - 3 + real.log x / real.log 5) = 5) 
  (hy : real.log3 (xyz - 3 + real.log y / real.log 5) = 4) 
  (hz : real.log4 (xyz - 3 + real.log z / real.log 5) = 4) 
  (hxyz : xyz = x * y * z) : 
  real.log (x * y * z) / real.log 5 = 3 :=
sorry

end log_eq_three_l731_731214


namespace cost_to_paint_cube_l731_731913

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) 
  (h1 : cost_per_kg = 40) 
  (h2 : coverage_per_kg = 20) 
  (h3 : side_length = 10) 
  : (6 * side_length^2 / coverage_per_kg) * cost_per_kg = 1200 :=
by
  sorry

end cost_to_paint_cube_l731_731913


namespace problem1_problem2_l731_731570

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem problem1 : f (5 * Real.pi / 6) = 0 := sorry

theorem problem2 : set.range (f ∘ (coe : set.Icc 0 (Real.pi / 4) → ℝ)) = set.Icc 1 2 := sorry

end problem1_problem2_l731_731570


namespace rebecca_gemstones_needed_l731_731701

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l731_731701


namespace octahedron_vertex_probability_l731_731951

/-- An octahedron consists of two square-based pyramids glued together along their square bases. 
    This forms a polyhedron with eight faces.
    An ant starts walking from the bottom vertex and randomly picks one of the four adjacent vertices 
    (middle ring) and calls it vertex A. 
    From vertex A, the ant then randomly selects one of its four adjacent vertices and calls it vertex B. 
    Prove that the probability that vertex B is the top vertex of the octahedron is 1/4. -/
theorem octahedron_vertex_probability : 
  let bottom_vertex := "initial vertex", 
      mid_ring := Set.of_list ["v1", "v2", "v3", "v4"], 
      top_vertex := "top vertex" in 
  ∀ A ∈ mid_ring, (cond_prob (λ v, v = top_vertex) (λ v, v ∈ {bottom_vertex} ∪ mid_ring ∪ {top_vertex})) = 1/4 :=
sorry

end octahedron_vertex_probability_l731_731951


namespace shortest_distance_is_1_l731_731755

-- Define the curve C using its polar equation
def curveC (θ : ℝ) : Prop := ∃ (ρ : ℝ), ρ = 2 * Real.sin θ ∧ ρ^2 = (2 * Real.sin θ)^2

-- Define the line l using its parametric form
def lineL (t : ℝ) : Prop :=
  ∃ (x y : ℝ), x = (√3) * t + √3 ∧ y = -3 * t + 2

-- Define the Cartesian equation form of line l
def lineL_Cartesian (x y : ℝ) : Prop :=
  (√3) * x + y - 5 = 0

-- Define the Cartesian form of the curve C, a circle with center (0, 1) and radius 1
def circleC (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

-- Prove that the minimum distance from any point on curve C to the line l is 1
theorem shortest_distance_is_1 : ∀ (x y : ℝ),
  circleC x y → (∃ t, lineL t → lineL_Cartesian x y) → 
  | (0 : ℝ) + 1 - 5 | / Real.sqrt (3+1) - 1 = 1 :=
by
  intros x y hC t hl
  sorry

end shortest_distance_is_1_l731_731755


namespace total_cost_is_correct_l731_731155

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end total_cost_is_correct_l731_731155


namespace find_f_neg_two_l731_731234

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg_two (h : ∀ x : ℝ, x ≠ 0 → f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
by
  sorry

end find_f_neg_two_l731_731234


namespace parallelepiped_analog_is_parallelogram_l731_731501

def is_parallelepiped_analog (f : Type) : Prop :=
  (∀ s1 s2 : f, opposite_sides_parallel s1 s2)

def opposite_sides_parallel (s1 s2 : Type) : Prop :=
  sorry -- Implementation details of the condition

theorem parallelepiped_analog_is_parallelogram (f : Type)
  (h : is_parallelepiped_analog f) : f = Parallelogram :=
sorry

end parallelepiped_analog_is_parallelogram_l731_731501


namespace charges_equal_at_x_4_cost_effectiveness_l731_731756

-- Defining the conditions
def full_price : ℕ := 240

def yA (x : ℕ) : ℕ := 120 * x + 240
def yB (x : ℕ) : ℕ := 144 * x + 144

-- (Ⅰ) Establishing the expressions for the charges is already encapsulated in the definitions.

-- (Ⅱ) Proving the equivalence of the two charges for a specific number of students x.
theorem charges_equal_at_x_4 : ∀ x : ℕ, yA x = yB x ↔ x = 4 := 
by {
  sorry
}

-- (Ⅲ) Discussing which travel agency is more cost-effective based on the number of students x.
theorem cost_effectiveness (x : ℕ) :
  (x < 4 → yA x > yB x) ∧ (x > 4 → yA x < yB x) :=
by {
  sorry
}

end charges_equal_at_x_4_cost_effectiveness_l731_731756


namespace unique_solution_integer_equation_l731_731983

theorem unique_solution_integer_equation : 
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end unique_solution_integer_equation_l731_731983


namespace square_area_with_circles_l731_731087

theorem square_area_with_circles 
  (r : ℝ)
  (nrows : ℕ)
  (ncols : ℕ)
  (circle_radius : r = 3)
  (rows : nrows = 2)
  (columns : ncols = 3)
  (num_circles : nrows * ncols = 6)
  : ∃ (side_length area : ℝ), side_length = ncols * 2 * r ∧ area = side_length ^ 2 ∧ area = 324 := 
by sorry

end square_area_with_circles_l731_731087


namespace chromium_percentage_in_second_alloy_l731_731277

theorem chromium_percentage_in_second_alloy (x : ℝ) :
  (15 * 0.12) + (35 * (x / 100)) = 50 * 0.106 → x = 10 :=
by
  sorry

end chromium_percentage_in_second_alloy_l731_731277


namespace equal_diagonals_implies_quad_or_pent_l731_731176

-- Define a convex polygon with n edges and equal diagonals
structure ConvexPolygon (n : ℕ) :=
(edges : ℕ)
(convex : Prop)
(diagonalsEqualLength : Prop)

-- State the theorem to prove
theorem equal_diagonals_implies_quad_or_pent (n : ℕ) (poly : ConvexPolygon n) 
    (h1 : poly.convex) 
    (h2 : poly.diagonalsEqualLength) :
    (n = 4) ∨ (n = 5) :=
sorry

end equal_diagonals_implies_quad_or_pent_l731_731176


namespace probability_sum_less_than_product_l731_731842

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731842


namespace length_of_ST_l731_731503

-- Define the isosceles triangle PQR
structure Triangle :=
  (base : ℝ)
  (height : ℝ)
  (area : ℝ)

-- Conditions given in the problem
def PQR : Triangle := {
  base := QR,
  height := 24,
  area := 144
}

-- Function to compute area of a triangle
def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

-- Horizontal line ST cuts PQR into an isosceles trapezoid and a smaller isosceles triangle
def isosceles_trapezoid_area : ℝ := 108

-- Prove that the length of ST is 6 inches
theorem length_of_ST : ∃ (ST : ℝ), ST = 6 :=
by
  -- Use sorry to skip the proof
  sorry

end length_of_ST_l731_731503


namespace minimum_people_correct_answer_l731_731464

theorem minimum_people_correct_answer (people questions : ℕ) (common_correct : ℕ) (h_people : people = 21) (h_questions : questions = 15) (h_common_correct : ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ people → 1 ≤ b ∧ b ≤ people → a ≠ b → common_correct ≥ 1) :
  ∃ (min_correct : ℕ), min_correct = 7 := 
sorry

end minimum_people_correct_answer_l731_731464


namespace alok_paid_rs_811_l731_731102

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l731_731102


namespace average_marks_for_class_l731_731748

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l731_731748


namespace second_group_size_approx_l731_731922

-- The condition that 16 men can complete the work in 25 days
def total_man_days_1 (num_men_1 : ℕ) (days_1 : ℕ) : ℕ := num_men_1 * days_1

-- The condition that another group can complete the work in 26.666666666666668 days
def total_man_days_2 (num_men_2 : ℝ) (days_2 : ℝ) : ℝ := num_men_2 * days_2

-- The initial known value of man-days
def man_days_constant : ℕ := total_man_days_1 16 25

noncomputable def men_in_second_group : ℝ :=
  man_days_constant / 26.666666666666668

theorem second_group_size_approx :
  men_in_second_group ≈ 15 :=
by
  sorry

end second_group_size_approx_l731_731922


namespace larger_number_is_322_l731_731914

/-- Define the highest common factor (HCF) and least common multiple (LCM) properties. --/
def hcf (a b : Nat) : Nat := Nat.gcd a b
def lcm (a b : Nat) : Nat := Nat.lcm a b

/-- Given conditions --/
variables (A B : Nat)
variable (H : hcf A B = 23)
variable (H1 : ∃ C D, lcm A B = 23 * 13 * 14 ∧ C * D = 23 * (23 * 13 * 14) ∧ (A = 23 * 14 ∨ A = 23 * 13))

/-- Proving the largest number is 322 --/
theorem larger_number_is_322 : A = 23 * 14 := by
  sorry

end larger_number_is_322_l731_731914


namespace length_of_inscribed_triangle_l731_731289

noncomputable def Ln (n : ℕ) : ℝ := 2 * n / 3

theorem length_of_inscribed_triangle (n : ℕ) :
  ∀ (L : ℕ → ℝ), 
    (∀ m, m ≥ 1 → (L m = 2 * m / 3)) →
    L n = Ln n :=
by
  intro L h
  exact h n (Nat.le_refl n)

end length_of_inscribed_triangle_l731_731289


namespace mappings_A_to_B_correct_mappings_B_to_A_correct_l731_731243

-- Define sets A and B
def A : Set := {1, 2, 3, 4, 5}
def B : Set := {1, 2, 3, 4}

-- Define number of elements in sets A and B
def nA : Nat := 5
def nB : Nat := 4

-- Define number of mappings from A to B
def num_mappings_A_to_B : Nat := 4^5

-- Define number of mappings from B to A
def num_mappings_B_to_A : Nat := 5^4

-- Statements to prove:
-- 1. The number of different mappings from set A to set B is 1024
theorem mappings_A_to_B_correct : num_mappings_A_to_B = 1024 := by
  sorry

-- 2. The number of different mappings from set B to set A is 625
theorem mappings_B_to_A_correct : num_mappings_B_to_A = 625 := by
  sorry

end mappings_A_to_B_correct_mappings_B_to_A_correct_l731_731243


namespace sum_of_digits_N_l731_731977

open Int

def N : Int := (List.range' 1 501).sum (λ k : Int, 10^k - 1)

theorem sum_of_digits_N : 
  ∑ digit in (N.digits 10), digit = 501 := 
  sorry

end sum_of_digits_N_l731_731977


namespace karan_initial_borrowed_amount_l731_731320

theorem karan_initial_borrowed_amount (P : ℝ) :
  let I1_3 := P * (4 / 100) * 3 in
  let A4_7 := P * (1 + 5 / 100)^4 in
  let I8_9 := A4_7 * (6 / 100) * 2 in
  P + I1_3 + A4_7 - P + I8_9 = 8210 → P = 5541.68 :=
by
  intros I1_3 A4_7 I8_9 h
  sorry

end karan_initial_borrowed_amount_l731_731320


namespace incorrect_calculation_l731_731716

noncomputable def ξ : ℝ := 3
noncomputable def η : ℝ := 5

def T (ξ η : ℝ) : ℝ := min ξ η

theorem incorrect_calculation : E[T(ξ, η)] ≤ 3 := by
  sorry

end incorrect_calculation_l731_731716


namespace sum_of_all_n_l731_731892

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l731_731892


namespace real_root_interval_l731_731536

noncomputable def P (b : ℝ) : polynomial ℝ := polynomial.C 1 * polynomial.X ^ 4 + polynomial.C b * polynomial.X ^ 3 - polynomial.C 2 * polynomial.X ^ 2 + polynomial.C b * polynomial.X + polynomial.C 4 

theorem real_root_interval (b : ℝ) : (∃ x : ℝ, polynomial.eval x (P b) = 0) →
  b ∈ set.Iic (-3/2) ∪ set.Ici (3/2) :=
by
  sorry

end real_root_interval_l731_731536


namespace probability_sum_less_than_product_l731_731839

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731839


namespace probability_sum_less_than_product_is_5_div_9_l731_731808

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731808


namespace probability_ab_gt_a_add_b_l731_731801

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731801


namespace abs_inequality_l731_731361

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l731_731361


namespace sum_of_integer_n_l731_731900

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l731_731900


namespace transformation_correct_l731_731341

def original_function (x : ℝ) : ℝ := Real.sin x

def shifted_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 10)

def transformed_function (x : ℝ) : ℝ := Real.sin ((1 / 2) * x - Real.pi / 10)

theorem transformation_correct : 
  ∀ x : ℝ, transformed_function x = Real.sin ((1 / 2) * x - Real.pi / 10) := by
  sorry

end transformation_correct_l731_731341


namespace binomial_coeff_pairwise_coprime_l731_731332

theorem binomial_coeff_pairwise_coprime (n k : ℕ) :
  ∃ (a : ℕ) (coprime_factors : Fin n → ℕ), (∀ i : Fin n, k + (i : ℕ) + 1 ∣ coprime_factors i) ∧ (∀ i j : Fin n, i ≠ j → Nat.coprime (coprime_factors i) (coprime_factors j)) ∧ (a = ∏ i in Finset.fin_range n, coprime_factors i) ∧ (a = Nat.choose (n + k) n) :=
by
  sorry

end binomial_coeff_pairwise_coprime_l731_731332


namespace problem_expansion_l731_731258

theorem problem_expansion (a : ℝ) (a_1 a_2 ... a_2013 : ℝ) :
  ( (λ x : ℝ, (1 - 2 * x) ^ 2013) = λ (x : ℝ), a + a_1 * x + a_2 * x^2 + ... + a_2013 * x^2013) →
  ((a + a_1) + (a + a_2) + ... + (a + a_2013) = 2011 :=
by
  sorry

end problem_expansion_l731_731258


namespace bogan_maggots_l731_731508

theorem bogan_maggots (x : ℕ) (total_maggots : ℕ) (eaten_first : ℕ) (eaten_second : ℕ) (thrown_out : ℕ) 
  (h1 : eaten_first = 1) (h2 : eaten_second = 3) (h3 : total_maggots = 20) (h4 : thrown_out = total_maggots - eaten_first - eaten_second) 
  (h5 : x + eaten_first = thrown_out) : x = 15 :=
by
  -- Use the given conditions
  sorry

end bogan_maggots_l731_731508


namespace geometric_sequence_sum_l731_731568

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h₀ : q > 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₂ : ∀ x : ℝ, 4 * x^2 - 8 * x + 3 = 0 → (x = a 2005 ∨ x = a 2006)) : 
  a 2007 + a 2008 = 18 := 
sorry

end geometric_sequence_sum_l731_731568


namespace probability_fx_positive_l731_731233

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem probability_fx_positive :
  let interval := λ a b x, a ≤ x ∧ x ≤ b in
  let prob := (λ I, if I then f (a) > 0 else f (b) < 0) in
  (4 + 3) / 10 = 0.7 :=
by
  sorry

end probability_fx_positive_l731_731233


namespace closest_integer_to_cube_root_of_200_l731_731038

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), 
    (n = 6) ∧ (n^3 < 200) ∧ (200 < (n + 1)^3) ∧ 
    (∀ m : ℤ, (m^3 < 200) → (200 < (m + 1)^3) → (Int.abs (n - Int.ofNat (200 ^ (1/3 : ℝ)).round) < Int.abs (m - Int.ofNat (200 ^ (1/3 : ℝ)).round))) :=
begin
  sorry
end

end closest_integer_to_cube_root_of_200_l731_731038


namespace probability_ab_gt_a_add_b_l731_731797

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731797


namespace jarry_prob_secretary_or_treasurer_l731_731067

/- The club has 10 members, including Jarry, who is represented as "J". -/
def Club := Fin 10
def J : Club := ⟨0, by decide⟩

/- Define events as selecting president, secretary, and treasurer, 
   which does not choose the same member for multiple positions. -/
noncomputable def select_president : Club := arbitrary
noncomputable def select_secretary (p : Club) : Club := by apply arbitrary; decide
noncomputable def select_treasurer (p s : Club) : Club := by repeat {apply arbitrary; decide}

/- Define the probability calculation for Jarry being either a secretary or a treasurer. -/
def prob_J_secretary := (1 : ℚ) / 9
def prob_J_treasurer := (1 : ℚ) / 10
def prob_J_secretary_or_treasurer := prob_J_secretary + prob_J_treasurer

/- Main theorem statement -/
theorem jarry_prob_secretary_or_treasurer : prob_J_secretary_or_treasurer = 19 / 90 := 
sorry

end jarry_prob_secretary_or_treasurer_l731_731067


namespace probability_sum_less_than_product_l731_731790

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731790


namespace batch_total_parts_l731_731451

theorem batch_total_parts
  (planned_rate : ℕ) (actual_rate : ℕ) (extra_days : ℕ) (remaining_parts : ℕ)
  (total_parts : ℕ) (planned_days : ℕ) :
  planned_rate = 50 →
  actual_rate = 44 →
  extra_days = 2 →
  remaining_parts = 32 →
  total_parts = planned_rate * planned_days →
  total_parts = actual_rate * (planned_days + extra_days) + remaining_parts →
  total_parts = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6,
  have h_total_eq : planned_rate * planned_days = actual_rate * (planned_days + extra_days) + remaining_parts :=
    by rw [h5, h6],
  sorry

end batch_total_parts_l731_731451


namespace bridge_length_l731_731942

theorem bridge_length (train_length : ℝ) (train_time : ℝ) (train_speed_kmph : ℝ)
  (convert : train_speed_kmph = 36) 
  (time_eq_25997920166386688 : train_time = 25.997920166386688)
  (train_length_eq_110 : train_length = 110) :
  let train_speed := train_speed_kmph * 1000 / 3600 in
  train_length + train_speed * train_time - train_length = 149.97920166386688 :=
by
  have speed_in_ms : train_speed = 10  :=
    by calc train_speed
      = train_speed_kmph * 1000 / 3600 : rfl
      ... = 36 * 1000 / 3600         : by rw [convert]
      ... = 10                       : by norm_num
  have total_distance : train_length + train_speed * train_time = 259.97920166386688 :=
    by calc train_length + train_speed * train_time
      = 110 + 10 * 25.997920166386688 : by rw [train_length_eq_110, speed_in_ms, time_eq_25997920166386688]
      ... = 259.97920166386688        : by norm_num
  calc train_length + train_speed * train_time - train_length
    = 259.97920166386688 - 110 : by rw [total_distance]
    ... = 149.97920166386688  : by norm_num

end bridge_length_l731_731942


namespace single_colony_reaches_limit_l731_731469

theorem single_colony_reaches_limit :
  ∀ (x : ℕ), (2^x = 2 * 2^19) → (x = 20) :=
by
  assume x hx,
  sorry

end single_colony_reaches_limit_l731_731469


namespace cost_of_3600_pencils_l731_731089

theorem cost_of_3600_pencils (cost_per_120_pencils : ℝ) (num_pencils_120 : ℕ) (num_pencils_3600 : ℕ)
  (unit_cost : cost_per_120_pencils = 40) : num_pencils_120 = 120 → 
  num_pencils_3600 = 3600 → 
  ∃ cost : ℝ, cost = 1200 :=
by
  intros h120 h3600
  let unit_price := cost_per_120_pencils / num_pencils_120
  have h_unit_price : unit_price = (40 : ℝ) / 120 := by
    rw [unit_cost, h120]
    simp
  let total_cost := unit_price * num_pencils_3600
  have h_total_cost : total_cost = 1200 := by
    rw [h_unit_price, h3600]
    simp
  exact ⟨total_cost, h_total_cost⟩
  sorry

end cost_of_3600_pencils_l731_731089


namespace smallest_positive_period_of_f_l731_731588

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x ^ 2 + sin x * cos x - (sqrt 3 / 2)

theorem smallest_positive_period_of_f : ∀ (T > 0), (∀ x, f (x + T) = f x) → T = π := 
sorry

end smallest_positive_period_of_f_l731_731588


namespace triangle_area_less_than_50_l731_731063

theorem triangle_area_less_than_50 (points : Fin 500 → ℝ × ℝ) :
  (∀ i, 0 ≤ points i.1 ∧ points i.1 ≤ 2) ∧ 
  (∀ i, 0 ≤ points i.2 ∧ points i.2 ≤ 1) → ∃ (i j k: Fin 500), 
  ( i ≠ j ∧ j ≠ k ∧ i ≠ k ) ∧ 
  (triangle_area (points i) (points j) (points k) < 50) :=
sorry

-- Helper definition for calculating triangle area
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1, y1 := A.2
  let x2 := B.1, y2 := B.2
  let x3 := C.1, y3 := C.2
  abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2

end triangle_area_less_than_50_l731_731063


namespace sin_cos_product_l731_731184

theorem sin_cos_product (ϕ : ℝ) (h : Real.tan (ϕ + Real.pi / 4) = 5) : 
  1 / (Real.sin ϕ * Real.cos ϕ) = 13 / 6 :=
by
  sorry

end sin_cos_product_l731_731184


namespace probability_sum_less_than_product_l731_731844

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731844


namespace solve_inequality_l731_731383

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731383


namespace height_of_right_triangle_on_parabola_equals_one_l731_731947

theorem height_of_right_triangle_on_parabola_equals_one 
    (x0 x1 x2 : ℝ) 
    (h0 : x0 ≠ x1)
    (h1 : x0 ≠ x2) 
    (h2 : x1 ≠ x2) 
    (h3 : x0^2 = x1^2) 
    (h4 : x0^2 < x2^2):
    x2^2 - x0^2 = 1 := by
  sorry

end height_of_right_triangle_on_parabola_equals_one_l731_731947


namespace largest_integral_solution_l731_731171

noncomputable def largest_integral_value : ℤ :=
  let a : ℚ := 1 / 4
  let b : ℚ := 7 / 11 
  let lower_bound : ℚ := 7 * a
  let upper_bound : ℚ := 7 * b
  let x := 3  -- The largest integral value within the bounds
  x

-- A theorem to prove that x = 3 satisfies the inequality conditions and is the largest integer.
theorem largest_integral_solution (x : ℤ) (h₁ : 1 / 4 < x / 7) (h₂ : x / 7 < 7 / 11) : x = 3 := by
  sorry

end largest_integral_solution_l731_731171


namespace triangle_min_product_sum_l731_731268

noncomputable section

open Classical

theorem triangle_min_product_sum (A B C : Point) (AB : ℝ) (h : height A B C = 3) (h_AB : dist A B = 10) :
  ∃ (AC BC : ℝ), (min (AC * BC) = 30) ∧ (AC + BC = 4 * real.sqrt 10) := sorry

end triangle_min_product_sum_l731_731268


namespace a_n_formula_b_n_formula_T_n_formula_l731_731560

noncomputable def S (n : ℕ) : ℕ := n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 3 else 2 * n + 1
noncomputable def b (n : ℕ) : ℕ := (4 * n - 1)/(3^(n-1))

theorem a_n_formula : ∀ n : ℕ, n ≠ 0 → a n = if n = 1 then 3 else 2 * n + 1 := by
  sorry

theorem b_n_formula : ∀ n : ℕ, b n = (4 * n - 1)/(3^(n-1)) := by
  sorry
  
noncomputable def T_n (n : ℕ) : ℕ := ∑ i in Finset.range(n), b (i + 1)

theorem T_n_formula : ∀ n : ℕ, T_n n = (15/2) - (4 * n + 5)/(2 * 3^(n - 1)) := by
  sorry 

end a_n_formula_b_n_formula_T_n_formula_l731_731560


namespace find_length_of_side_AB_l731_731292

noncomputable def side_AB 
(triangle : Type*) 
[metric_space triangle]
[inner_product_space ℝ triangle]
(a b c : triangle)
(I : triangle)
(radius_incircle : ℝ) 
(h_incircle : radius_incircle = 10) 
(bc_length : ℝ) 
(h_bc : bc_length = 35) 
(AL_ratio : ℝ) 
(h_AL_ratio : AL_ratio = 5 / 2)
: ℝ :=
let AB_length := (5 / 12) * (105 + 2 * real.sqrt 105) in
AB_length

theorem find_length_of_side_AB (triangle : Type*) [metric_space triangle] [inner_product_space ℝ triangle] (a b c : triangle) (I : triangle) (radius_incircle bc_length AL_ratio: ℝ) 
    (h_incircle : radius_incircle = 10) 
    (h_bc : bc_length = 35) 
    (h_AL_ratio : AL_ratio = 5 / 2) :
    side_AB triangle a b c I radius_incircle h_incircle bc_length h_bc AL_ratio h_AL_ratio = (5 / 12) * (105 + 2 * real.sqrt 105) 
:= by
    sorry

end find_length_of_side_AB_l731_731292


namespace min_adventurers_l731_731480

theorem min_adventurers (R E S D : Finset ℕ) 
  (hR : R.card = 5) 
  (hE : E.card = 11) 
  (hS : S.card = 10) 
  (hD : D.card = 6)
  (hDiamonds : ∀ x, x ∈ D → (x ∈ E ∧ x ∉ S) ∨ (x ∉ E ∧ x ∈ S))
  (hEmeralds : ∀ x, x ∈ E → (x ∈ R ∧ x ∉ D) ∨ (x ∉ R ∧ x ∈ D)) :
  ∃ n, n = 16 := 
begin
  use 16,
  sorry
end

end min_adventurers_l731_731480


namespace length_segmentPQ_is_sqrt2_div2_l731_731639

noncomputable def tetrahedron_length_segmentPQ : ℝ :=
  let A := (0 : ℝ, 0 : ℝ, real.sqrt (2 / 3))
  let B := (1 : ℝ, 0 : ℝ, real.sqrt (2 / 3))
  let C := (1 / 2 : ℝ, real.sqrt 3 / 2, 0)
  let D := (1 / 2 : ℝ, -real.sqrt 3 / 2, 0)
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
  let Q := ((C.1 + D.1) / 2, (C.2 + D.2) / 2, (C.3 + D.3) / 2)
  real.sqrt (((P.1 - Q.1)^2) + ((P.2 - Q.2)^2) + ((P.3 - Q.3)^2))

theorem length_segmentPQ_is_sqrt2_div2 :
  tetrahedron_length_segmentPQ = real.sqrt 2 / 2 :=
sorry

end length_segmentPQ_is_sqrt2_div2_l731_731639


namespace tangent_line_parallel_to_x_axis_l731_731740

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def f_derivative (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

theorem tangent_line_parallel_to_x_axis :
  ∀ x₀ : ℝ, 
  f_derivative x₀ = 0 → 
  f x₀ = 1 / Real.exp 1 :=
by
  intro x₀ h_deriv_zero
  sorry

end tangent_line_parallel_to_x_axis_l731_731740


namespace perpendicular_lines_l731_731626

-- Define the conditions of the problem
def line1 (m : ℝ) : (x y : ℝ) → Prop :=
  λ x y, m * x + 2 * y + 1 = 0

def line2 (m : ℝ) : (x y : ℝ) → Prop :=
  λ x y, x - m^2 * y + (1 / 2) = 0

-- Define the proof statement that the lines are perpendicular if these conditions are met
theorem perpendicular_lines (m : ℝ) :
  ∀ x y : ℝ, 
  line1 m x y → line2 m x y → 
  m = 0 ∨ m = (1 / 2) :=
by
  intros x y h1 h2
  sorry

end perpendicular_lines_l731_731626


namespace tom_needs_44000_pounds_salt_l731_731785

theorem tom_needs_44000_pounds_salt 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℕ)
  (salt_cost_per_pound : ℝ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_revenue : ℕ) 
  (expected_salt_cost : ℝ) 
  (S : ℝ) : 
  flour_needed = 500 → 
  flour_bag_weight = 50 → 
  flour_bag_cost = 20 → 
  salt_cost_per_pound = 0.2 → 
  promotion_cost = 1000 → 
  ticket_price = 20 → 
  tickets_sold = 500 → 
  total_revenue = 8798 → 
  0.2 * S = (500 * 20) - (500 / 50) * 20 - 1000 →
  S = 44000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end tom_needs_44000_pounds_salt_l731_731785


namespace find_a_squared_plus_b_squared_l731_731230

noncomputable def circle (a b : ℝ) := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 2 * a * p.1 - 2 * b * p.2 + a ^ 2 + b ^ 2 - 1 = 0 }

theorem find_a_squared_plus_b_squared (a b : ℝ) 
  (h1 : a < 0)
  (h2 : b = sqrt 3 * (a + 1))
  (h3 : ∀ (p : ℝ × ℝ), 
       p ∈ circle a b → dist (p, line (sqrt 3)  0) = 1 + sqrt 3) :
  a^2 + b^2 = 3 :=
sorry

end find_a_squared_plus_b_squared_l731_731230


namespace solve_inequality_l731_731374

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l731_731374


namespace one_side_half_length_of_P1_l731_731140

variables {P1 P2 P3 : Type} [Parallelogram P1] [Parallelogram P2] [Parallelogram P3]

-- Conditions
variables (inside_P2_P1 : Π {x : P2}, ∀ y, y ∈ (vertices x) → y ∈ (edges P1))
variables (inside_P3_P2 : Π {x : P3}, ∀ y, y ∈ (vertices x) → y ∈ (edges P2))
variables (parallel_sides_P3_P1 : Π {s1 : sides P3} {s2 : sides P1}, parallel s1 s2)

-- Theorem statement: There exists a side of P3 whose length is at least half the length of the parallel side of P1
theorem one_side_half_length_of_P1 (side_P1_long : ℝ) (side_P3_long : ℝ) :
  ∃ (s : sides P3) (t : sides P1), parallel s t ∧ length s ≥ (1/2) * length t := 
sorry

end one_side_half_length_of_P1_l731_731140


namespace stratified_sampling_selection_probability_one_female_from_A_probability_two_male_from_total_l731_731641

-- Definitions of the group sizes and female counts
def groupA_size : Nat := 10
def groupB_size : Nat := 10
def groupA_female : Nat := 4
def groupB_female : Nat := 6

-- Stratified sampling definitions
def total_sampled : Nat := 4
def each_group_sampled : Nat := 2

-- Probability calculation placeholders
def P_A : Real := sorry -- formula for P(A)
def P_B : Real := sorry -- formula for P(B)

-- The statement we want to prove
theorem stratified_sampling_selection : 
  each_group_sampled = 2 :=
begin
  sorry
end

theorem probability_one_female_from_A : 
  P_A = sorry := 
begin
  sorry
end

theorem probability_two_male_from_total : 
  P_B = sorry := 
begin
  sorry
end

end stratified_sampling_selection_probability_one_female_from_A_probability_two_male_from_total_l731_731641


namespace total_test_subjects_l731_731084

-- Defining the conditions as mathematical entities
def number_of_colors : ℕ := 5
def unique_two_color_codes : ℕ := number_of_colors * number_of_colors
def excess_subjects : ℕ := 6

-- Theorem stating the question and correct answer
theorem total_test_subjects :
  unique_two_color_codes + excess_subjects = 31 :=
by
  -- Leaving the proof as sorry, since the task only requires statement creation
  sorry

end total_test_subjects_l731_731084


namespace exists_real_numbers_a_in_range_l731_731181

theorem exists_real_numbers_a_in_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    x ^ 3 + a = -3 * (y + z) ∧
    y ^ 3 + a = -3 * (x + z) ∧
    z ^ 3 + a = -3 * (x + y)) ↔ a ∈ set.Ioo (-2 : ℝ) (2 : ℝ) \ {0} :=
sorry

end exists_real_numbers_a_in_range_l731_731181


namespace min_value_of_expression_l731_731617

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end min_value_of_expression_l731_731617


namespace variance_N_l731_731666

-- Define the independent random variables ε_i
noncomputable def epsilon (i : ℕ) : ℕ → ℤ := sorry

-- Define S_k as the sum of the ε_i's up to k
noncomputable def S (k : ℕ) : ℕ → ℤ :=
  λ n, (1 to k).sum_fun (εpsilon n)

-- Define N_{2n} as the count of integers k in [2, 2n] such that S_k > 0 or S_k = 0 and S_{k-1} > 0
noncomputable def N (n : ℕ) : ℕ → ℤ :=
  λ n, (2 to 2 * n).count_fun (λ k, (S k n > 0) ∨ ((S k n = 0) ∧ (S (k - 1) n > 0)))

-- Prove the variance of N_{2n} is 3(2n-1)/16
theorem variance_N (n : ℕ) : var (N (2 * n)) = 3 * (2 * n - 1) / 16 := by
  sorry

end variance_N_l731_731666


namespace acute_angle_sufficient_not_necessary_l731_731982

noncomputable def angle := ℝ

def acute_angle (α : angle) : Prop :=
  0 < α ∧ α < π / 2

def sin_pos (α : angle) : Prop :=
  Real.sin α > 0

theorem acute_angle_sufficient_not_necessary (α : angle) :
  (acute_angle α → sin_pos α) ∧ (¬(sin_pos α → acute_angle α)) :=
by
  sorry

end acute_angle_sufficient_not_necessary_l731_731982


namespace factorization_correct_l731_731434

theorem factorization_correct (m : ℤ) : m^2 - 1 = (m - 1) * (m + 1) :=
by {
  -- sorry, this is a place-holder for the proof.
  sorry
}

end factorization_correct_l731_731434


namespace sum_lt_prod_probability_l731_731823

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731823


namespace perimeter_last_triangle_l731_731304

theorem perimeter_last_triangle :
  let a₁ := 403
  let b₁ := 405
  let c₁ := 401
  let a : ℕ → ℝ := λ n, if n = 0 then a₁ else (b (n-1) + c (n-1) - a (n-1)) / 2
  let b : ℕ → ℝ := λ n, if n = 0 then b₁ else (a (n-1) + c (n-1) - b (n-1)) / 2
  let c : ℕ → ℝ := λ n, if n = 0 then c₁ else (a (n-1) + b (n-1) - c (n-1)) / 2
  in 
  lim (λ n, a n + b n + c n) = 1209 / 512 :=
sorry

end perimeter_last_triangle_l731_731304


namespace incorrect_lifetime_calculation_l731_731729

-- Define expectation function
noncomputable def expectation (X : ℝ) : ℝ := sorry

-- We define the lifespans
variables (xi eta : ℝ)
-- Expected lifespan of the sensor and transmitter
axiom exp_xi : expectation xi = 3
axiom exp_eta : expectation eta = 5

-- Define the lifetime of the device
noncomputable def T := min xi eta

-- Given conditions
theorem incorrect_lifetime_calculation :
  expectation T ≤ 3 → 3 + (2 / 3) > 3 → false := 
sorry

end incorrect_lifetime_calculation_l731_731729


namespace price_per_kilo_of_bananas_l731_731328

def initial_money : ℕ := 500
def potatoes_cost : ℕ := 6 * 2
def tomatoes_cost : ℕ := 9 * 3
def cucumbers_cost : ℕ := 5 * 4
def bananas_weight : ℕ := 3
def remaining_money : ℕ := 426

-- Defining total cost of all items
def total_item_cost : ℕ := initial_money - remaining_money

-- Defining the total cost of bananas
def cost_bananas : ℕ := total_item_cost - (potatoes_cost + tomatoes_cost + cucumbers_cost)

-- Final question: Prove that the price per kilo of bananas is $5
theorem price_per_kilo_of_bananas : cost_bananas / bananas_weight = 5 :=
by
  sorry

end price_per_kilo_of_bananas_l731_731328


namespace no_multiple_among_given_numbers_l731_731771

-- Definitions of the conditions
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def uses_digits_3_4_6_8 (n : ℕ) : Prop :=
  let digits := [3, 4, 6, 8] in
  let n_digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  ∀ d ∈ n_digits, d ∈ digits ∧ n_digits.count d = digits.count d

def is_multiple (a b : ℕ) : Prop :=
  a = b * 2 ∨ a = b * 3

-- The proof problem statement
theorem no_multiple_among_given_numbers :
  ¬ ∃ (a b : ℕ),
    a ∈ {3486, 4638, 6384, 6843, 8643} ∧ 
    b ∈ {3486, 4638, 6384, 6843, 8643} ∧ 
    uses_digits_3_4_6_8 a ∧
    uses_digits_3_4_6_8 b ∧ 
    is_multiple a b :=
begin
  sorry
end

end no_multiple_among_given_numbers_l731_731771


namespace distinct_pizzas_l731_731935

theorem distinct_pizzas (n : ℕ) (h : n = 8) : (n + (n * (n - 1)) / 2) = 36 :=
by
  rw h
  sorry

end distinct_pizzas_l731_731935


namespace problem_l731_731618

theorem problem (x : ℝ) (h : x = 2 / (2 - real.cbrt 3)) : 
  x = 2 * (2 + real.cbrt 3) / (4 - real.cbrt 9) :=
by 
  sorry

end problem_l731_731618


namespace find_angle_l731_731284

theorem find_angle {x : ℝ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 → ∃ x, x > 0) (h2 : 9 * x = 900) : x = 100 :=
  sorry

end find_angle_l731_731284


namespace probability_sum_less_than_product_is_5_div_9_l731_731809

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731809


namespace equality_of_t_l731_731343

variables 
  (ρ ρ₁ ρ₂ ρ₃ : ℝ)
  (α β γ : ℝ)

noncomputable def t_equiv_ctg : Prop :=
  let lhs := ρ * ρ₁ * (Real.cot (α / 2)) in
  let rhs₁ := ρ * ρ₂ * (Real.cot (β / 2)) in
  let rhs₂ := ρ * ρ₃ * (Real.cot (γ / 2)) in
  lhs = rhs₁ ∧ rhs₁ = rhs₂

noncomputable def t_equiv_tg : Prop :=
  let lhs₂ := ρ₂ * ρ₃ * (Real.tan (α / 2)) in
  let rhs₃ := ρ₁ * ρ₃ * (Real.tan (β / 2)) in
  let rhs₄ := ρ₁ * ρ₂ * (Real.tan (γ / 2)) in
  lhs₂ = rhs₃ ∧ rhs₃ = rhs₄

theorem equality_of_t 
  (h1: t_equiv_ctg ρ ρ₁ ρ₂ ρ₃ α β γ)
  (h2: t_equiv_tg ρ ρ₁ ρ₂ ρ₃ α β γ)
: 
  ρ * ρ₁ * (Real.cot (α / 2)) = ρ * ρ₂ * (Real.cot (β / 2)) ∧
  ρ * ρ₂ * (Real.cot (β / 2)) = ρ * ρ₃ * (Real.cot (γ / 2)) ∧
  ρ₂ * ρ₃ * (Real.tan (α / 2)) = ρ₁ * ρ₃ * (Real.tan (β / 2)) ∧
  ρ₁ * ρ₃ * (Real.tan (β / 2)) = ρ₁ * ρ₂ * (Real.tan (γ / 2)) :=
by 
  exact ⟨h1, h2⟩;
  repeat { sorry }


end equality_of_t_l731_731343


namespace third_repair_cost_calculation_l731_731504

noncomputable def problem (C : ℝ) (third_repair_cost : ℝ) : Prop :=
  0.25 * C = 1800 ∧ third_repair_cost = 0.05 * C

theorem third_repair_cost_calculation : ∃ C third_repair_cost, problem C third_repair_cost ∧ third_repair_cost = 360 :=
by
  use 7200
  use 360
  split
  {
    split
    {
      norm_num
    }
    {
      norm_num
    }
  }
  norm_num
  sorry

end third_repair_cost_calculation_l731_731504


namespace sum_of_b_for_one_solution_l731_731984

theorem sum_of_b_for_one_solution :
  (∃ b : ℝ, (3:ℝ) * x^2 + (b : ℝ) * x + 6 * x + 7 = 0 ∧ ((b+6)^2 - 4 * 3 * 7 = 0 → b = -6 + 2 * real.sqrt 21 ∨ b = -6 - 2 * real.sqrt 21)) →
  (-6 + 2 * real.sqrt 21) + (-6 - 2 * real.sqrt 21) = -12 :=
sorry

end sum_of_b_for_one_solution_l731_731984


namespace number_of_inverses_modulo_12_l731_731250

def count_inverses_modulo_12 : Nat :=
  List.length (List.filter (λ n, Nat.gcd n 12 = 1) (List.range 12))

theorem number_of_inverses_modulo_12 :
  count_inverses_modulo_12 = 4 := by sorry

end number_of_inverses_modulo_12_l731_731250


namespace find_first_term_l731_731224

open Int

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_first_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 2 = 1)
  (h_a4_a10 : a 3 + a 9 = 18) :
  a 0 = -3 :=
by
  sorry

end find_first_term_l731_731224


namespace determine_counterfeit_bag_l731_731467

-- Conditions
variables (n : ℕ) [h_n : fact (n ≥ 3)] 
          (num_coins : ℕ := (n * (n + 1)) / 2 + 1) 
          (bags : fin n → fin num_coins → ℝ) -- weight of each coin in each bag (real differentiable function ensuring distinct weights)

-- Definitions
def real_bags (i : fin n) : Prop :=
i ∈ finset.range (n - 1)

def counterfeit_bag (i : fin n) : Prop :=
i = n - 1

-- Hypotheses
hypothesis h_real_weights : ∀ i, real_bags i → ∀ j, ∀ k, bags i j = bags i k
hypothesis h_counterfeit_weight : ∀ i j, counterfeit_bag i → bags i j ≠ ∃ k, real_bags i k 

-- Statement - Proof not required, only the statement
theorem determine_counterfeit_bag :
  ∃ method : (fin n) × (fin num_coins → ℝ) → Prop, -- a method describing the strategy
  (∀ c : fin n, ∀ p : fin num_coins → ℝ, 
  method (c, p) → counterfeit_bag c) :=
begin
  -- The proof is required here
  sorry
end

end determine_counterfeit_bag_l731_731467


namespace sum_of_integer_n_l731_731898

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l731_731898


namespace hyperbola_eq_l731_731216

theorem hyperbola_eq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (hyp_eq : ∀ x y, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1)
  (asymptote : b / a = Real.sqrt 3)
  (focus_parabola : c = 4) : 
  a^2 = 4 ∧ b^2 = 12 := by
sorry

end hyperbola_eq_l731_731216


namespace second_candidate_votes_l731_731777

-- Define the conditions
def total_votes (V : ℕ) : ℕ :=
  let T := 21000 in
    T

def winning_percentage : ℝ := 0.55371428571428574
def winning_votes : ℕ := 11628
def first_votes : ℕ := 1136

theorem second_candidate_votes :
  ∃ V : ℕ, total_votes V - first_votes - winning_votes = 8236 :=
by
  use 8236
  have T : ℕ := total_votes 8236
  have H1 : 11628 = (winning_percentage * T).to_nat := by sorry
  have H2 : T ≈ 21000 := by sorry
  simp [total_votes, H1, H2]
  sorry

end second_candidate_votes_l731_731777


namespace savings_percentage_l731_731910

theorem savings_percentage (I S : ℝ) (h1 : I > 0) (h2 : S > 0) (h3 : S ≤ I) 
  (h4 : 1.25 * I - 2 * S + I - S = 2 * (I - S)) :
  (S / I) * 100 = 25 :=
by
  sorry

end savings_percentage_l731_731910


namespace graph_transformation_l731_731418

theorem graph_transformation :
  ∀ (x : ℝ), 
  (sin 2x - (sqrt 3) * cos 2x) = 2 * sin (2x - π / 3) :=
by
  sorry

end graph_transformation_l731_731418


namespace Trapezoid_APQC_Circumscribed_l731_731652

def Triangle (A B C : Type) := ∃ PQ RS BM : Type, 
  (PQ ∥ AC) ∧ (RS ∥ AC) ∧ BM = BM

def TrapezoidCircumscribed (P Q R S : Type) : Prop :=
  ∃ r : Type, ∀ point, True -- Simplified circumscription definition, adjust as needed

variables {A B C P Q R S K L M : Type}
variables (α β φ : ℝ)

theorem Trapezoid_APQC_Circumscribed
  (hABC : Triangle A B C)
  (hPQ_parallel : PQ ∥ AC)
  (hRS_parallel : RS ∥ AC)
  (hRPKL_circumscribed : TrapezoidCircumscribed R P K L)
  (hMLSC_circumscribed : TrapezoidCircumscribed M L S C)
  (hPK_RL_ratio : ∀ tanα tanφ, ∃ tanα tanφ, (tanα * tanφ) = (PK / RL))
  (hLS_MC_ratio : ∀ cotφ tanβ, ∃ cotφ tanβ, (cotφ * tanβ) = (LS / MC)) :
  TrapezoidCircumscribed A P Q C := 
sorry

end Trapezoid_APQC_Circumscribed_l731_731652


namespace cos_eq_neg_half_range_f_l731_731606

-- Problem 1
theorem cos_eq_neg_half 
  (x : ℝ) 
  (m : ℝ×ℝ) 
  (n : ℝ×ℝ) 
  (h1 : m = (sqrt(3) * sin(x / 4), 1)) 
  (h2 : n = (cos(x / 4), cos(x / 4) ^ 2)) 
  (h3 : m.1 * n.1 + m.2 * n.2 = 1) : 
  cos (2 * pi / 3 - x) = -1 / 2 := 
by 
  sorry

-- Problem 2
theorem range_f 
  (A B C a b c : ℝ) 
  (m : ℝ×ℝ) 
  (n : ℝ×ℝ) 
  (h1 : m = (sqrt(3) * sin(A / 4), 1)) 
  (h2 : n = (cos(A / 4), cos(A / 4) ^ 2)) 
  (h3 : (2 * a - c) * cos B = b * cos C) : 
  1 < (sin (A / 2 + pi / 6) + 1 / 2) ∧ (sin (A / 2 + pi / 6) + 1 / 2) < 3 / 2 := 
by 
  sorry

end cos_eq_neg_half_range_f_l731_731606


namespace least_period_l731_731979

-- We define the condition that f satisfies for all real x.
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 6) + f(x - 6) = f(x)

-- We state the theorem that if f satisfies the condition, then the least positive period of f is 36.
theorem least_period (f : ℝ → ℝ) (hf : condition f) : ∀ p > 0, (∀ x : ℝ, f(x + p) = f(x)) ↔ p = 36 :=
sorry

end least_period_l731_731979


namespace lauras_european_stamps_cost_l731_731057

def stamp_cost (count : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  count * cost_per_stamp

def total_stamps_cost (stamps80 : ℕ) (stamps90 : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  stamp_cost stamps80 cost_per_stamp + stamp_cost stamps90 cost_per_stamp

def european_stamps_cost_80_90 :=
  total_stamps_cost 10 12 0.09 + total_stamps_cost 18 16 0.07

theorem lauras_european_stamps_cost : european_stamps_cost_80_90 = 4.36 :=
by
  sorry

end lauras_european_stamps_cost_l731_731057


namespace rewinding_time_l731_731478

theorem rewinding_time (a L S ω T : ℝ) (h_pos_ω : ω > 0) (h_pos_S : S > 0) :
  (T = (π / (S * ω)) * (sqrt (a^2 + (4 * S * L) / π) - a)) :=
begin
  sorry
end

end rewinding_time_l731_731478


namespace positional_relationships_non_coincident_lines_ranges_of_angles_theorem_completion1_theorem_completion2_theorem_completion3_theorem_completion4_l731_731920

-- Definitions of lines, planes, and their relationships
variables 
  {A P : Type} -- representing points
  {a b l : Type} -- representing lines
  {α β : Type} -- representing planes
  [LinearSpace A] 

-- 1. Prove positional relationships
theorem positional_relationships_non_coincident_lines 
  (h_non_coincident : ∀ (a b : A), a ≠ b) :
  {r : Type} (h1: a || b → r = "parallel") 
  (h2: ∃ P, a ∩ b = P → r = "intersecting") 
  (h3: ⊥, a || b → r = "skew") := sorry

-- 2. Prove the ranges of angles
theorem ranges_of_angles 
  (h_skew_lines : (0, π / 2]) 
  (h_line_plane : [0, π / 2])
  (h_dihedral : [0, π]) := 
  (h_skew_lines, h_line_plane, h_dihedral)

-- 3. Theorem completions
theorem theorem_completion1 
  {a b : Type} {α : Type} 
  (h1_1 : a \not\subset α) 
  (h1_2 : b ⊂ α) 
  (h1_3 : a || b) :
  a || α := sorry

theorem theorem_completion2
  {a b : Type} {α β : Type} 
  (h2_1 : a ⊂ β) 
  (h2_2 : b ⊂ β)
  (h2_3 : a ∩ b = P) 
  (h2_4 : a \not\parallel α) 
  (h2_5 : b \not\parallel α) :
  α ∥ β := sorry

theorem theorem_completion3
  {a b l : Type} {α : Type}
  (h3_1 : a ⊂ α)
  (h3_2 : b ⊂ α) 
  (h3_3 : a ∩ b = A)
  (h3_4 : l ⊥ a)
  (h3_5 : l ⊥ b) :
  l ⊥ α := sorry

theorem theorem_completion4
  {l : Type} {α β : Type}
  (h4_1 : l ⊥ α)
  (h4_2 : l ⊂ β) :
  α ⊥ β := sorry

end positional_relationships_non_coincident_lines_ranges_of_angles_theorem_completion1_theorem_completion2_theorem_completion3_theorem_completion4_l731_731920


namespace probability_ab_gt_a_add_b_l731_731798

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731798


namespace cut_square_into_two_squares_l731_731524

theorem cut_square_into_two_squares (a : ℝ) (h : a > 0) :
  ∃ (parts : List (set (ℝ × ℝ))), parts.length = 4 ∧
  (∀ p ∈ parts, ∃ r s, is_rectangle p r s) ∧
  ∃ (s1 s2 : set (ℝ × ℝ)), 
    is_square s1 (a / 2) ∧ is_square s2 (a / 2) ∧ 
    (⋃ p ∈ parts, p) = s1 ∪ s2 :=
sorry

end cut_square_into_two_squares_l731_731524


namespace repeating_decimal_value_l731_731622

def repeating_decimal : ℝ := 0.0000253253325333 -- Using repeating decimal as given in the conditions

theorem repeating_decimal_value :
  (10^7 - 10^5) * repeating_decimal = 253 / 990 :=
sorry

end repeating_decimal_value_l731_731622


namespace circle_equation_and_trajectory_l731_731294

variable (x y x0 y0 m n : ℝ)

-- Definitions based on conditions
def circle_center := (0, 0)
def tangent_line (x y : ℝ) : Prop := x - y = 2 * Real.sqrt 2
def point_A_on_circle (x0 y0 : ℝ) : Prop := x0^2 + y0^2 = 4
def N_perpendicular_to_x_axis (x0 : ℝ) : (x0, 0)
def Q_moving_condition (x y x0 y0 m n : ℝ) : Prop := 
  x = (m + n) * x0 ∧ y = m * y0 ∧ (m + n = 1) ∧ (m ≠ 0) ∧ (n ≠ 0)

-- Proof problem
theorem circle_equation_and_trajectory :
  (point_A_on_circle x0 y0) →
  (Q_moving_condition x y x0 y0 m n) →
  (x0 = x ∧ y0 = y / m) →
  (∃ x y, x^2 + y^2 = 4 ∧ (x^2 / 4 + y^2 / (4 * m^2) = 1)) :=
by
  intros h1 h2 h3
  sorry

end circle_equation_and_trajectory_l731_731294


namespace rod_total_length_l731_731487

theorem rod_total_length
  (n : ℕ) (l : ℝ)
  (h₁ : n = 50)
  (h₂ : l = 0.85) :
  n * l = 42.5 := by
  sorry

end rod_total_length_l731_731487


namespace solve_inequality_l731_731359

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l731_731359


namespace ellipse_equation_line_equation_of_area_l731_731577

open Real

-- Defining constants and variables
variables (a b c x y m : ℝ)
variables (ellipse_eq line_eq : ℝ → ℝ → Prop)

-- Conditions
def conditions : Prop :=
  (a > b ∧ b > 0) ∧
  (2 * a * 1 / 2 = 2 * b) ∧
  (a - c = 2 - sqrt 3) ∧
  (a^2 = b^2 + c^2)

-- The equation of the ellipse
def ellipse_eq := (x : ℝ) (y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- The equation of the line
def line_eq := (x : ℝ) (y : ℝ) : Prop := (y = x + m)

-- Proof of the ellipse equation
theorem ellipse_equation (h : conditions a b c) : ellipse_eq x y :=
by
  sorry

-- Proof of the line equation given the area of the triangle is 1
theorem line_equation_of_area (h : conditions a b c)
  (ha : ellipse_eq x y)
  (area_eq : ∃ A B, (A ≠ B ∧ ∀ m, y = x + m ∧ 
  abs ((1 / 2) * (A.1 * B.2 - A.2 * B.1)) = 1)) : 
  ∃ m, (y = x + m ∨ y = x - m) :=
by
  sorry

end ellipse_equation_line_equation_of_area_l731_731577


namespace max_contestants_l731_731483

/- 
Given:
- Each contestant scores between 0 and 7 inclusive on three problems.
- For any two contestants, there exists at most one problem in which they have obtained the same score.

To Prove:
- The maximum number of contestants is 64.
-/

theorem max_contestants :
  ∀ (contestants : Set (Fin 8 × Fin 8 × Fin 8)),
    (∀ c1 c2 ∈ contestants, c1 ≠ c2 → (c1.1 ≠ c2.1 ∨ c1.2 ≠ c2.2 ∨ c1.3 ≠ c2.3)) →
    contestants.card ≤ 64 :=
by
  sorry

end max_contestants_l731_731483


namespace intersection_M_N_l731_731059

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then some_function(x)  -- Assuming some_function is strictly increasing for positive x
else if x < 0 then -some_function(-x) else 0

def g (q : ℝ) (m : ℝ) : ℝ := sin (2 * q) + m * cos q - 2 * m

def M (q : ℝ) : set ℝ := {m | g q m < 0}

def N (q : ℝ) : set ℝ := {m | f (g q m) < 0}

theorem intersection_M_N (q : ℝ) : M q ∩ N q = {m | m > 2} :=
sorry

end intersection_M_N_l731_731059


namespace smallest_positive_perfect_cube_has_divisor_l731_731308

theorem smallest_positive_perfect_cube_has_divisor (p q r s : ℕ) (hp : Prime p) (hq : Prime q)
  (hr : Prime r) (hs : Prime s) (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  ∃ n : ℕ, n = (p * q * r * s^2)^3 ∧ ∀ m : ℕ, (m = p^2 * q^3 * r^4 * s^5 → m ∣ n) :=
by
  sorry

end smallest_positive_perfect_cube_has_divisor_l731_731308


namespace alok_total_payment_l731_731094

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l731_731094


namespace abs_fraction_inequality_solution_l731_731350

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l731_731350


namespace probability_sum_less_than_product_is_5_div_9_l731_731807

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731807


namespace first_grade_sample_count_l731_731481

-- Defining the total number of students and their ratio in grades 1, 2, and 3.
def total_students : ℕ := 2400
def ratio_grade1 : ℕ := 5
def ratio_grade2 : ℕ := 4
def ratio_grade3 : ℕ := 3
def total_ratio := ratio_grade1 + ratio_grade2 + ratio_grade3

-- Defining the sample size
def sample_size : ℕ := 120

-- Proving that the number of first-grade students sampled should be 50.
theorem first_grade_sample_count : 
  (sample_size * ratio_grade1) / total_ratio = 50 :=
by
  -- sorry is added here to skip the proof
  sorry

end first_grade_sample_count_l731_731481


namespace ratio_of_inradii_l731_731001

-- Given triangle XYZ with sides XZ=5, YZ=12, XY=13
-- Let W be on XY such that ZW bisects ∠ YZX
-- The inscribed circles of triangles ZWX and ZWY have radii r_x and r_y respectively
-- Prove the ratio r_x / r_y = 1/6

theorem ratio_of_inradii
  (XZ YZ XY : ℝ)
  (W : ℝ)
  (r_x r_y : ℝ)
  (h1 : XZ = 5)
  (h2 : YZ = 12)
  (h3 : XY = 13)
  (h4 : r_x / r_y = 1/6) :
  r_x / r_y = 1/6 :=
by sorry

end ratio_of_inradii_l731_731001


namespace probability_sum_less_than_product_l731_731817

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731817


namespace sqrt_expression_l731_731970

open Real

theorem sqrt_expression :
  3 * sqrt 12 / (3 * sqrt (1 / 3)) - 2 * sqrt 3 = 6 - 2 * sqrt 3 :=
by
  sorry

end sqrt_expression_l731_731970


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l731_731124

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l731_731124


namespace number_of_members_greater_than_median_l731_731634

theorem number_of_members_greater_than_median (n : ℕ) (median : ℕ) (avg_age : ℕ) (youngest : ℕ) (oldest : ℕ) :
  n = 100 ∧ avg_age = 21 ∧ youngest = 1 ∧ oldest = 70 →
  ∃ k, k = 50 :=
by
  sorry

end number_of_members_greater_than_median_l731_731634


namespace solve_inequality_l731_731377

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l731_731377


namespace find_x_of_total_area_l731_731647

theorem find_x_of_total_area 
  (x : Real)
  (h_triangle : (1/2) * (4 * x) * (3 * x) = 6 * x^2)
  (h_square1 : (3 * x)^2 = 9 * x^2)
  (h_square2 : (6 * x)^2 = 36 * x^2)
  (h_total : 6 * x^2 + 9 * x^2 + 36 * x^2 = 700) :
  x = Real.sqrt (700 / 51) :=
by {
  sorry
}

end find_x_of_total_area_l731_731647


namespace sufficient_but_not_necessary_condition_l731_731917

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 4 → a^2 > 16) ∧ (∃ a, (a < -4) ∧ (a^2 > 16)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l731_731917


namespace rewinding_time_l731_731475

noncomputable def time_required_for_rewinding (a L S ω : ℝ) : ℝ :=
  (π / (S * ω)) * (sqrt (a^2 + (4 * S * L) / π) - a)

theorem rewinding_time (a L S ω : ℝ) (hS : S > 0) (hω : ω > 0) : 
  ∃ T : ℝ, T = time_required_for_rewinding a L S ω :=
by
  use time_required_for_rewinding a L S ω
  sorry

end rewinding_time_l731_731475


namespace incorrect_calculation_l731_731723

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l731_731723


namespace abs_fraction_inequality_solution_l731_731353

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l731_731353


namespace minimal_natives_needed_l731_731183

theorem minimal_natives_needed (k : ℕ) : ∃ N, (∀ i, i < N → (N = 2 ^ k)) ∧ (∀ j < N, (know_jokes j ≥ k)) :=
begin
  sorry -- Proof goes here
end

end minimal_natives_needed_l731_731183


namespace sum_of_all_n_l731_731893

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l731_731893


namespace total_climbing_time_l731_731518

-- Definitions for the conditions given.
def first_flight_time : ℕ := 30
def common_difference : ℕ := 8
def num_flights : ℕ := 6

-- The main theorem
theorem total_climbing_time : 
  let a := first_flight_time in
  let d := common_difference in
  let n := num_flights in
  let s := n * (2 * a + (n - 1) * d) / 2 in
  s = 300 := 
by
  sorry

end total_climbing_time_l731_731518


namespace probability_sum_less_than_product_is_5_div_9_l731_731804

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731804


namespace min_value_of_f_l731_731592

-- Define the function f with the given properties
variable {f : ℝ → ℝ}
variable {f1 f2 : ℝ}

-- Assume the functional equation for f
axiom f_property : ∀ x : ℝ, 0 ≤ x → f(x) = f1 * x + f2 / x - 1

-- Define the minimum value property
def min_value (f : ℝ → ℝ) (a b : ℝ) : ℝ := 
  Inf (set.range (λ x : {x // a < x ∧ x < b}, f x.val))

-- Prove that minimum value is √3 - 1
theorem min_value_of_f : min_value f 0 0+∞ = sqrt(3) - 1 :=
sorry

end min_value_of_f_l731_731592


namespace total_number_of_coins_l731_731295

variable (nickels dimes total_value : ℝ)
variable (total_nickels : ℕ)

def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10

theorem total_number_of_coins :
  total_value = 3.50 → total_nickels = 30 → total_value = total_nickels * value_of_nickel + dimes * value_of_dime → 
  total_nickels + dimes = 50 :=
by
  intros h_total_value h_total_nickels h_value_equation
  sorry

end total_number_of_coins_l731_731295


namespace eccentricity_is_correct_l731_731218

open Real

noncomputable def eccentricity_of_ellipse 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (angle_cond : ∀ (F1 F2 P : ℝ × ℝ), (∠ P F1 F2) = π / 3)
  (distance_cond : ∀ (P F1 F2 : ℝ × ℝ), dist P F1 = 2 * dist P F2)
  (major_axis_def : ∀ (P F1 F2 : ℝ × ℝ), 2 * a = dist P F1 + dist P F2) : 
  ℝ :=
  sorry

theorem eccentricity_is_correct
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (angle_cond : ∀ (F1 F2 P : ℝ × ℝ), (∠ P F1 F2) = π / 3)
  (distance_cond : ∀ (P F1 F2 : ℝ × ℝ), dist P F1 = 2 * dist P F2)
  (major_axis_def : ∀ (P F1 F2 : ℝ × ℝ), 2 * a = dist P F1 + dist P F2) :
  eccentricity_of_ellipse a b h1 h2 angle_cond distance_cond major_axis_def = sqrt 3 / 3 :=
sorry

end eccentricity_is_correct_l731_731218


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l731_731133

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l731_731133


namespace sum_of_first_21_terms_l731_731403

noncomputable def a_seq (n : ℕ) : ℚ :=
if even n then 2 else -3/2

noncomputable def S (n : ℕ) : ℚ :=
∑ i in finset.range (n + 1), a_seq i

theorem sum_of_first_21_terms :
  S 21 = 7 / 2 :=
sorry

end sum_of_first_21_terms_l731_731403


namespace find_rate_l731_731956

variable (P A T : ℕ)
hypothesis hP : P = 12500
hypothesis hA : A = 18500
hypothesis hT : T = 8

theorem find_rate (R : ℕ) 
  (hR : (A - P) = P * R * T / 100) : 
  R = 6 :=
by 
  rw [hP, hA, hT] at hR
  sorry

end find_rate_l731_731956


namespace calculate_expression_l731_731119

theorem calculate_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end calculate_expression_l731_731119


namespace probability_sum_less_than_product_l731_731846

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731846


namespace abs_fraction_inequality_solution_l731_731351

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l731_731351


namespace area_of_annulus_l731_731105

-- Definitions for the problem conditions
variables (b c a : ℝ)
variable (hb : b > c)
variables (O X Y Z : Type)
variable [metric_space O]
variables [inner_product_space ℝ O]

-- Defining the lengths and geometric conditions
axiom OX_radius : dist O X = b
axiom OY_radius_contains_Z : dist O Y = b
axiom XZ_tangent_to_smaller_circle_at_Z : dist O Z = c ∧ dist X Z = a

-- Theorem statement: Proof of the area of the annulus
theorem area_of_annulus (hb : b > c)
    (OX_radius : dist O X = b)
    (OY_radius_contains_Z : dist O Y = b)
    (XZ_tangent_to_smaller_circle_at_Z : dist O Z = c ∧ dist X Z = a) :
    ∃ (π : ℝ), π * a^2 = π * (b^2 - c^2) :=
by sorry

end area_of_annulus_l731_731105


namespace sum_of_valid_n_l731_731878

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l731_731878


namespace train_speed_l731_731852

theorem train_speed (V_p : ℝ) : 
  (∀ (dist_pq : ℝ) (start_time_p start_time_q : ℝ) (meet_time : ℝ)
    (speed_q : ℝ), 
    dist_pq = 200 ∧ 
    start_time_p = 7 ∧ 
    start_time_q = 8 ∧ 
    meet_time = 12 ∧ 
    speed_q = 25 → 
      let time_p := meet_time - start_time_p in
      let time_q := meet_time - start_time_q in
      let dist_q_covered := speed_q * time_q in
      let dist_p_covered := dist_pq - dist_q_covered in
      V_p = dist_p_covered / time_p
  ) ↔ V_p = 20 :=
begin
  sorry
end

end train_speed_l731_731852


namespace sum_valid_qs_l731_731901

noncomputable def valid_qs : List ℚ :=
  List.filter (λ q : ℚ,
    let a := q.nat_num; b := q.denominator in b ≤ 10 ∧ ∃ n : ℤ, (nat_succ (nat_fl (sqrt q)) ≤ n) ∧ (n < q.num)
  )
  [(m / n)| m, n ∈ Finset.range 101, n ≠ 0]

theorem sum_valid_qs : (List.sum valid_qs) = 777.5 := sorry

end sum_valid_qs_l731_731901


namespace gemstones_needed_l731_731697

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l731_731697


namespace sin_A_perimeter_triangle_l731_731459

-- Problem 1: Prove that sin A = sqrt(5) / 5
theorem sin_A {a b c A B C : ℝ} (hac : a = c) (hbc : b = c)
  (h_sin : sqrt(5) * a * Real.sin B = b) :
  Real.sin A = sqrt(5) / 5 := 
sorry

-- Problem 2: Prove that the perimeter of ΔABC = √26 + √5 + 3
theorem perimeter_triangle {a b c A B C : ℝ} (hA_obtuse : Real.pi / 2 < A)
  (hb : b = sqrt(5)) (hc : c = 3) 
  (h_sin_A : Real.sin A = sqrt(5) / 5) 
  (h_cos_A : Real.cos A = -2 * sqrt(5) / 5) :
  a = sqrt(26) → a + b + c = sqrt(26) + sqrt(5) + 3 := 
sorry

end sin_A_perimeter_triangle_l731_731459


namespace find_integer_n_l731_731754

theorem find_integer_n (a b : ℕ) (n : ℕ)
  (h1 : n = 2^a * 3^b)
  (h2 : (2^(a+1) - 1) * ((3^(b+1) - 1) / (3 - 1)) = 1815) : n = 648 :=
  sorry

end find_integer_n_l731_731754


namespace map_scale_l731_731323

theorem map_scale (cm : ℝ) (km : ℝ) (cm_to_km_conversion : cm = 10) (km_representation : km = 50): (15 : ℝ) * (50 : ℝ) / (10 : ℝ) = 75 :=
by
  calc
    (15 : ℝ) * (50 : ℝ) / (10 : ℝ) = 15 * (50 / 10) : by rw [← mul_div_assoc]
    ... = 15 * 5 : by norm_num
    ... = 75 : by norm_num

end map_scale_l731_731323


namespace incorrect_calculation_l731_731719

noncomputable def ξ : ℝ := 3
noncomputable def η : ℝ := 5

def T (ξ η : ℝ) : ℝ := min ξ η

theorem incorrect_calculation : E[T(ξ, η)] ≤ 3 := by
  sorry

end incorrect_calculation_l731_731719


namespace circle_tangent_line_l731_731389

theorem circle_tangent_line (m : ℝ) (h : m > 0) :
  ∃ (x y : ℝ), (x^2 + y^2 = 4 * m) ∧ (x + y = 2 * real.sqrt m) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 = 4 * m → x + y = 2 * real.sqrt m → (y + 2 * sqrt m) * (y - 2 * sqrt m) ≤ 0) :=
sorry

end circle_tangent_line_l731_731389


namespace katie_travel_distance_l731_731298

theorem katie_travel_distance (d_train d_bus d_bike d_car d_total d1 d2 d3 : ℕ)
  (h1 : d_train = 162)
  (h2 : d_bus = 124)
  (h3 : d_bike = 88)
  (h4 : d_car = 224)
  (h_total : d_total = d_train + d_bus + d_bike + d_car)
  (h1_distance : d1 = 96)
  (h2_distance : d2 = 108)
  (h3_distance : d3 = 130)
  (h1_prob : 30 = 30)
  (h2_prob : 50 = 50)
  (h3_prob : 20 = 20) :
  (d_total + d1 = 694) ∧
  (d_total + d2 = 706) ∧
  (d_total + d3 = 728) :=
sorry

end katie_travel_distance_l731_731298


namespace closest_integer_to_cube_root_of_200_l731_731036

theorem closest_integer_to_cube_root_of_200 : 
  let n := 200 in
  let a := 5 in 
  let b := 6 in 
  abs (b - real.cbrt n) < abs (a - real.cbrt n) := 
by sorry

end closest_integer_to_cube_root_of_200_l731_731036


namespace probability_sum_less_than_product_l731_731845

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731845


namespace attendees_not_from_A_B_C_D_l731_731472

theorem attendees_not_from_A_B_C_D
  (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (num_D : ℕ) (total_attendees : ℕ)
  (hA : num_A = 30)
  (hB : num_B = 2 * num_A)
  (hC : num_C = num_A + 10)
  (hD : num_D = num_C - 5)
  (hTotal : total_attendees = 185)
  : total_attendees - (num_A + num_B + num_C + num_D) = 20 := by
  sorry

end attendees_not_from_A_B_C_D_l731_731472


namespace geometric_series_sum_l731_731513

def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  sum_geometric_series (1/4) (1/4) 7 = 4/3 :=
by
  -- Proof is omitted
  sorry

end geometric_series_sum_l731_731513


namespace integral_result_eq_l731_731162

noncomputable def integral_expression : ℝ :=
  ∫ x in -1..1, (x^2 + real.sqrt(1 - x^2))

theorem integral_result_eq :
  integral_expression = (2 / 3) + (real.pi / 2) :=
by
  sorry

end integral_result_eq_l731_731162


namespace installation_time_l731_731931

theorem installation_time (total_windows : Nat) (installed_windows : Nat) (time_per_window : Nat) 
  (h_total : total_windows = 9) (h_installed : installed_windows = 6) (h_time : time_per_window = 6) : 
  (total_windows - installed_windows) * time_per_window = 18 := 
by
  rw [h_total, h_installed, h_time]
  norm_num

end installation_time_l731_731931


namespace circle_square_area_ratio_l731_731074

theorem circle_square_area_ratio (R r : ℝ) (c d : ℝ)
  (h1 : R > 0)
  (h2 : r > 0)
  (h3 : R > r)
  (h4 : ∀ s: ℝ, s = r * real.sqrt 2)
  (eq_area : π * R^2 = (c / d) * (π * R^2 - π * r^2 + 4 * r^2))
  : R / r = real.sqrt ((c * (4 - real.pi)) / (d * real.pi - c * real.pi)) :=
begin
  sorry
end

end circle_square_area_ratio_l731_731074


namespace closest_integer_to_cube_root_of_200_l731_731031

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), n = 6 ∧ (n^3 = 216 ∨ n^3 > 125 ∧ n^3 < 216) := 
by
  existsi 6
  split
  · refl
  · right
    split
    · norm_num
    · norm_num

end closest_integer_to_cube_root_of_200_l731_731031


namespace smallest_next_divisor_l731_731296

theorem smallest_next_divisor (m : ℕ) (h_digit : 10000 ≤ m ∧ m < 100000) (h_odd : m % 2 = 1) (h_div : 437 ∣ m) :
  ∃ d : ℕ, 437 < d ∧ d ∣ m ∧ (∀ e : ℕ, 437 < e ∧ e < d → ¬ e ∣ m) ∧ d = 475 := 
sorry

end smallest_next_divisor_l731_731296


namespace find_principal_l731_731046

-- Define the conditions
def total_simple_interest : ℝ := 6016.75
def rate_per_annum : ℝ := 8
def time_years : ℝ := 5

-- Define the formula for simple interest and the simplified form
def principal (SI R T : ℝ) : ℝ := (SI * 100) / (R * T)

-- State the theorem we need to prove
theorem find_principal : principal total_simple_interest rate_per_annum time_years = 15041.875 :=
by sorry

end find_principal_l731_731046


namespace students_appeared_l731_731643

theorem students_appeared (T : ℕ) (h_passed : 35% of students passed)
    (h_failed : 481 students failed) :
    (0.65 * T = 481) → T = 740 :=
by
  sorry

end students_appeared_l731_731643


namespace solve_inequality_l731_731382

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731382


namespace sum_of_integer_n_l731_731896

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l731_731896


namespace quadratic_root_value_m_l731_731211

theorem quadratic_root_value_m (m : ℝ) : ∃ x, x = 1 ∧ x^2 + x - m = 0 → m = 2 := by
  sorry

end quadratic_root_value_m_l731_731211


namespace team_configurations_l731_731946

theorem team_configurations :
  let members : List String := ["Alice", "Bob", "Clara", "David", "Eve", "Felipe", "Grace", "Holly"]
  ∃ teams : List (List String), (∀ team ∈ teams, team.length ≠ 1 ∧ (1 < team.length < 8)) ∧ (∃ pairs : ℕ, pairs = 105) ∧ (∃ quads : ℕ, quads = 70) ∧ (pairs + quads = 175) := 
  by
    trivial -- this represents the proof steps we skip
    exact sorry

end team_configurations_l731_731946


namespace int_solutions_l731_731535

noncomputable def valid_solutions : List (ℤ × ℤ × ℤ) :=
[(0, 1, 1), (0, 1, -2), (1, 1, 2), (1, 1, -3), (1, 2, 3), (1, 2, -4), (2, 3, 8), (2, 3, -9)]

theorem int_solutions (a b k : ℤ) : 
  2 ^ b * 3 ^ a = k * (k + 1) ↔ (a, b, k) ∈ valid_solutions := 
by
  sorry

end int_solutions_l731_731535


namespace product_of_binomials_l731_731117

theorem product_of_binomials :
  (2*x^2 + 3*x - 4) * (x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 :=
by {
  sorry
}

end product_of_binomials_l731_731117


namespace solve_inequality_l731_731384

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731384


namespace a_100_l731_731758

  noncomputable def a : ℕ → ℕ
  | 1     := 1
  | (n+1) := a n + (2 * a n) / n

  theorem a_100 : a 100 = 5151 := by
  sorry
  
end a_100_l731_731758


namespace total_cost_is_correct_l731_731156

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end total_cost_is_correct_l731_731156


namespace probability_floor_sqrt_100x_l731_731310

noncomputable def probability_floor_sqrt_100x_eq_150_given_floor_sqrt_x_eq_15 
  (x : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (h_floor_sqrt_x : ⌊√x⌋ = 15) : ℝ :=
  if h : 100 ≤ x ∧ x < 300 then ∃ length_valid_interval : ℝ, 
    length_valid_interval = 228.01 - 225
    length_total_interval = 256 - 225
  by
    exact (length_valid_interval / length_total_interval)
  else absurd (hx) (by linarith)

theorem probability_floor_sqrt_100x 
  (x : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (h_floor_sqrt_x : ⌊√x⌋ = 15) : 
  probability_floor_sqrt_100x_eq_150_given_floor_sqrt_x_eq_15 x hx h_floor_sqrt_x = 301 / 3100 := by
  sorry

end probability_floor_sqrt_100x_l731_731310


namespace evaluate_expression_l731_731976

def g (x : ℝ) : ℝ := x^3 - 3 * Real.sqrt x

theorem evaluate_expression : 3 * g 2 - g 8 = -488 - 3 * Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l731_731976


namespace B_and_C_mutually_exclusive_but_not_complementary_l731_731495

-- Define the sample space of the cube
def faces : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events based on conditions
def event_A (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_B (n : ℕ) : Prop := n = 1 ∨ n = 2
def event_C (n : ℕ) : Prop := n = 4 ∨ n = 5 ∨ n = 6

-- Define mutually exclusive events
def mutually_exclusive (A B : ℕ → Prop) : Prop := ∀ n, A n → ¬ B n

-- Define complementary events (for events over finite sample spaces like faces)
-- Events A and B are complementary if they partition the sample space faces
def complementary (A B : ℕ → Prop) : Prop := (∀ n, n ∈ faces → A n ∨ B n) ∧ (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n)

theorem B_and_C_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_B event_C ∧ ¬ complementary event_B event_C := 
by
  sorry

end B_and_C_mutually_exclusive_but_not_complementary_l731_731495


namespace probability_sum_less_than_product_l731_731789

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731789


namespace largest_tan_C_l731_731293

-- Define points and the sides of the triangle
variables {A B C : Type}
variables (AB BC : ℝ) (AC : ℝ) (angleA : ℝ)

-- Conditions: AB = 30, BC = 18, and ∠A = 90°
def AB_length : AB = 30 := 
  by 
  sorry

def BC_length : BC = 18 := 
  by 
  sorry

def angleA_is_right : angleA = 90 := 
  by 
  sorry

-- According to the given conditions, we prove that tan(C) = 5/4
theorem largest_tan_C (hAB : AB_length) (hBC : BC_length) (hAngleA : angleA_is_right): 
  (\tan (∠ C)) = 5 / 4 := 
  by
  sorry

end largest_tan_C_l731_731293


namespace unique_fixed_point_of_rotation_invariant_function_l731_731219

theorem unique_fixed_point_of_rotation_invariant_function
  (f : ℝ → ℝ)
  (h : ∀ x, f(-f x) = x) :
  ∃! x, f x = x :=
sorry

end unique_fixed_point_of_rotation_invariant_function_l731_731219


namespace principal_is_correct_l731_731492

-- Definitions of given conditions
def SI : ℝ := 4016.25
def R : ℝ := 11 -- Rate in percentage
def T : ℝ := 5 -- Time in years

-- Computed Principal
def P : ℝ := SI / (R * T / 100)

-- The theorem we want to prove
theorem principal_is_correct : P = 7302.27 := 
by
  -- The proof goes here, which computes the principal and confirms it matches 7302.27
  sorry

end principal_is_correct_l731_731492


namespace ladder_distance_slides_outwards_l731_731064

noncomputable def ladder_slide_distance : Real :=
  let ladder_length : Real := 30
  let initial_distance : Real := 11
  let slip_down_distance : Real := 6
  let initial_height := Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2)
  let new_height := initial_height - slip_down_distance
  let equation := (new_height)^2 + ((initial_distance + ladder_slide_distance)^2) = ladder_length^2
  ladder_slide_distance

theorem ladder_distance_slides_outwards : 
  ladder_slide_distance = 0.9 := 
by
  -- Proof required here, the value is derived from solving the quadratic equation
  sorry

end ladder_distance_slides_outwards_l731_731064


namespace rebecca_gemstones_needed_l731_731700

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l731_731700


namespace closest_integer_to_cube_root_of_200_l731_731034

theorem closest_integer_to_cube_root_of_200 : 
  let n := 200 in
  let a := 5 in 
  let b := 6 in 
  abs (b - real.cbrt n) < abs (a - real.cbrt n) := 
by sorry

end closest_integer_to_cube_root_of_200_l731_731034


namespace smallest_pos_int_ends_in_6_div_by_5_l731_731864

theorem smallest_pos_int_ends_in_6_div_by_5 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 6 ∧ n % 5 = 0 ∧ (∀ m : ℕ, m > 0 ∧ m % 10 = 6 ∧ m % 5 = 0 → n ≤ m) :=
begin
  use 46,
  split, by norm_num,
  split, by norm_num,
  split, by norm_num,
  intros m hm,
  obtain ⟨_, ha, hb⟩ := hm,
  suffices : m ≥ 46,
  { exact this.le },
  sorry -- Proof omitted
end

end smallest_pos_int_ends_in_6_div_by_5_l731_731864


namespace rewinding_time_l731_731476

noncomputable def time_required_for_rewinding (a L S ω : ℝ) : ℝ :=
  (π / (S * ω)) * (sqrt (a^2 + (4 * S * L) / π) - a)

theorem rewinding_time (a L S ω : ℝ) (hS : S > 0) (hω : ω > 0) : 
  ∃ T : ℝ, T = time_required_for_rewinding a L S ω :=
by
  use time_required_for_rewinding a L S ω
  sorry

end rewinding_time_l731_731476


namespace teacher_allocation_l731_731544

theorem teacher_allocation :
  ∃ n : ℕ, n = 150 ∧ 
  (∀ t1 t2 t3 t4 t5 : Prop, -- represent the five teachers
    ∃ s1 s2 s3 : Prop, -- represent the three schools
      s1 ∧ s2 ∧ s3 ∧ -- each school receives at least one teacher
        ((t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧ -- allocation condition
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5))) := sorry

end teacher_allocation_l731_731544


namespace benzene_carbon_mass_percentage_l731_731541

noncomputable def carbon_mass_percentage_in_benzene 
  (carbon_atomic_mass : ℝ) (hydrogen_atomic_mass : ℝ) 
  (benzene_formula_ratio : (ℕ × ℕ)) : ℝ := 
    let (num_carbon_atoms, num_hydrogen_atoms) := benzene_formula_ratio
    let total_carbon_mass := num_carbon_atoms * carbon_atomic_mass
    let total_hydrogen_mass := num_hydrogen_atoms * hydrogen_atomic_mass
    let total_mass := total_carbon_mass + total_hydrogen_mass
    100 * (total_carbon_mass / total_mass)

theorem benzene_carbon_mass_percentage 
  (carbon_atomic_mass : ℝ := 12.01) 
  (hydrogen_atomic_mass : ℝ := 1.008) 
  (benzene_formula_ratio : (ℕ × ℕ) := (6, 6)) : 
    carbon_mass_percentage_in_benzene carbon_atomic_mass hydrogen_atomic_mass benzene_formula_ratio = 92.23 :=
by 
  unfold carbon_mass_percentage_in_benzene
  sorry

end benzene_carbon_mass_percentage_l731_731541


namespace sin_x_value_l731_731567

theorem sin_x_value (x : ℝ) (h : Real.sec x - Real.tan x = 5 / 2) : Real.sin x = -21 / 29 :=
  sorry

end sin_x_value_l731_731567


namespace max_imaginary_part_of_root_l731_731500

theorem max_imaginary_part_of_root :
  ∀ (z : ℂ),
    (z ^ 12 - z ^ 9 + z ^ 6 - z ^ 3 + 1 = 0 →
    ∃ (θ : ℝ), -π / 2 ≤ θ ∧ θ ≤ π / 2 ∧ θ = 84 * (π / 180) ∧ z.im = real.sin θ) :=
sorry

end max_imaginary_part_of_root_l731_731500


namespace cost_equations_l731_731069

-- Define the conditions
def price_racket : ℕ := 100
def price_shuttlecock : ℕ := 20
def racket_quantity : ℕ := 10
variable {x : ℕ}
def cond_x : Prop := x > 10

-- Define the payment options as functions
def option1_cost (x : ℕ) : ℕ := 20 * x + 800
def option2_cost (x : ℕ) : ℕ := 18 * x + 900

-- Prove the costs in terms of x
theorem cost_equations (h : cond_x) : 
  option1_cost x = 20 * x + 800 ∧ option2_cost x = 18 * x + 900 := 
by
  sorry

end cost_equations_l731_731069


namespace derivative_at_one_l731_731624

def f (x: ℝ): ℝ := x^2

theorem derivative_at_one : (derivative f 1) = 2 :=
by
  sorry

end derivative_at_one_l731_731624


namespace sum_of_valid_n_l731_731870

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l731_731870


namespace problem_1_problem_2_l731_731414

-- Condition: Numbers 1,2,3,4 in the bag
def numbers : List ℕ := [1, 2, 3, 4]

-- Problem (1): Probability that the sum of the numbers on the balls drawn is no greater than 4.
def prob_sum_no_greater_than_4 : ℚ :=
  let outcomes := [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
  let favorable := List.filter (λ (x : ℕ × ℕ), x.fst + x.snd ≤ 4) outcomes
  favorable.length / outcomes.length

theorem problem_1 : prob_sum_no_greater_than_4 = 1 / 3 := by
  sorry

-- Problem (2): Probability that the second number is less than the first number plus 2
def prob_n_less_than_m_plus_2 : ℚ :=
  let outcomes := [ (m, n) | m <- numbers, n <- numbers ]
  let unfavorable := List.filter (λ (x : ℕ × ℕ), x.snd >= x.fst + 2) outcomes
  (outcomes.length - unfavorable.length) / outcomes.length

theorem problem_2 : prob_n_less_than_m_plus_2 = 13 / 16 := by
  sorry

end problem_1_problem_2_l731_731414


namespace exists_integers_satisfying_conditions_l731_731706

theorem exists_integers_satisfying_conditions :
  ∃ (a b c d : ℤ), d ≥ 1 ∧ (b ≡ c [MOD d]) ∧ a ∣ b ∧ a ∣ c ∧ ¬ ((b / a) ≡ (c / a) [MOD d]) :=
by
  use 2, 4, 0, 4
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }

end exists_integers_satisfying_conditions_l731_731706


namespace closest_integer_to_cuberoot_of_200_l731_731023

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l731_731023


namespace parabola_intersection_l731_731851

theorem parabola_intersection :
  let y1 := 2 * ( (-5 - sqrt 69) / 2 )^2 + 5 * ( (-5 - sqrt 69) / 2 ) - 3
  let y2 := 2 * ( (-5 + sqrt 69) / 2 )^2 + 5 * ( (-5 + sqrt 69) / 2 ) - 3
  (∃ y, y = 2 * ( (-5 - sqrt 69) / 2 )^2 + 5 * ( (-5 - sqrt 69) / 2 ) - 3 /\ y = x^2 + 8 ) ↔
  (2 * x ^ 2 + 5 * x - 3 = x^2 + 8) →
  set_eq {(((-5 - sqrt 69) / 2), y1), (((-5 + sqrt 69) / 2), y2)}
    {p : ℝ × ℝ | ∃ x, (p = (x, 2 * x^2 + 5 * x - 3) ∧ p = (x, x^2 + 8))} := sorry

end parabola_intersection_l731_731851


namespace solve_equation1_solve_equation2_l731_731348

-- Statement of the first problem.
theorem solve_equation1 (x : ℝ) : 
  (2 - x) / (x - 3) = 3 / (3 - x) ↔ x = 5 :=
begin
  sorry
end

-- Statement of the second problem.
theorem solve_equation2 (x : ℝ) :
  4 / (x^2 - 1) + 1 = (x - 1) / (x + 1) ↔ x = -1 :=
begin
  sorry
end

end solve_equation1_solve_equation2_l731_731348


namespace percent_of_students_in_range_l731_731479

theorem percent_of_students_in_range (n_90_to_100 n_85_to_89 n_75_to_84 n_65_to_74 n_below_65 : ℕ) 
  (h_90_to_100 : n_90_to_100 = 6) 
  (h_85_to_89 : n_85_to_89 = 4) 
  (h_75_to_84 : n_75_to_84 = 7) 
  (h_65_to_74 : n_65_to_74 = 10) 
  (h_below_65 : n_below_65 = 3) :
  (n_85_to_89).to_rat / (n_90_to_100 + n_85_to_89 + n_75_to_84 + n_65_to_74 + n_below_65).to_rat * 100 = 40 / 3 :=
by {
  sorry
}

end percent_of_students_in_range_l731_731479


namespace sum_of_n_l731_731874

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l731_731874


namespace fourth_term_expansion_l731_731958

theorem fourth_term_expansion (a x : ℝ) : 
  ((∑ k in Finset.range 8, (Nat.choose 7 k) * (a / x) ^ (7 - k) * (x / a ^ 3) ^ k).nth 3).get_or_else 0 = 35 / (a ^ 5 * x) := 
sorry

end fourth_term_expansion_l731_731958


namespace problem1_l731_731457

theorem problem1 (a : ℝ) (m n : ℕ) (h1 : a^m = 10) (h2 : a^n = 2) : a^(m - 2 * n) = 2.5 := by
  sorry

end problem1_l731_731457


namespace angle_BAC_36_iff_condition_l731_731767

-- Define the points on a plane
variables {A B C D : Type}

-- Define the lengths of the line segments and the angle
variables [InnerProductSpace ℝ A]
variables (a b c d : A)

-- Conditions: collinearity of B, C, D and location of A
-- AB = AC = CD, and D is collinear with B and C, C is between B and D
variables (h_collinear : ∃ l : A, ∃ s t : ℝ, ∃ w : A, b = s • l ∧ d = t • l ∧ c = w)
variables (h_c_between_bd : ∀ x : ℝ, x • b + (1-x) • d = c)
variables (h_AB_eq_AC : dist a b = dist a c)
variables (h_AC_eq_CD : dist a c = dist c d)

-- The proof problem statement
theorem angle_BAC_36_iff_condition :
  angle a b c = real.pi / 5 ↔ 1 / dist c d - 1 / dist b d = 1 / (dist c d + dist b d) :=
sorry

end angle_BAC_36_iff_condition_l731_731767


namespace probability_sum_less_than_product_l731_731811

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731811


namespace general_term_c_range_l731_731595

-- Condition: The sequence and its properties
def seq (a : ℕ → ℕ) := (a 1 = 3) ∧ ∀ m n : ℕ, a (m + n) = a m + a n + 2 * m * n

-- First proof problem: Prove the general term of the sequence
theorem general_term (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, a n = n^2 + 2 * n :=
sorry

-- Second proof problem: Prove the range for c
theorem c_range (c : ℝ) : (∀ k : ℕ, ∑ i in Finset.range k, 1 / (i^2 + 2 * i) < c) → c ≥ 3/4 :=
sorry

end general_term_c_range_l731_731595


namespace part_a_part_b_l731_731741

def condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) + (x + 1/2) * f(1 - x) = 1

theorem part_a (f : ℝ → ℝ) (hf : condition f) : f(0) = 2 ∧ f(1) = -2 :=
sorry

theorem part_b (f : ℝ → ℝ) (hf : condition f) :
  ∀ x : ℝ, (x ≠ 1/2 → f(x) = 2 / (1 - 2 * x)) ∧ (f(1/2) = 1/2) :=
sorry

end part_a_part_b_l731_731741


namespace bugs_eat_plants_l731_731509

theorem bugs_eat_plants (initial_plants : ℕ) (remaining_plants : ℕ) (plants_after_third_day: ℕ) (plants_eaten_third_day : ℕ) 
  (plants_remaining_second_day : ℕ) (half_remaining_after_first_day : ℕ) :
  initial_plants = 30 →
  plants_after_third_day = 4 →
  plants_eaten_third_day = 1 →
  plants_remaining_second_day = plants_after_third_day + plants_eaten_third_day →
  half_remaining_after_first_day = 2 * plants_remaining_second_day →
  remaining_plants = initial_plants - half_remaining_after_first_day →
  remaining_plants = 10 →
  (initial_plants - remaining_plants = 20) :=
by
  intros h_initial h_third h_eaten h_plants_second_day h_half_remaining h_remaining h_10
  rw [h_10] at h_remaining
  have h_total_eaten : initial_plants - remaining_plants = 20 := by
    calc
      initial_plants - remaining_plants = 30 - 10 := congrArg (λ x => initial_plants - x) h_remaining
                    ... = 20 := by rfl
  exact h_total_eaten


end bugs_eat_plants_l731_731509


namespace prob1_solution_prob2_solution_l731_731316

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Proof Problem 1: Prove the solution to the inequality
theorem prob1_solution (x : ℝ) : f(2 * x) ≤ f(x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

-- Proof Problem 2: Prove the minimum value of f(a^2) + f(b^2)
theorem prob2_solution (a b : ℝ) (h : a + b = 2) : f(a^2) + f(b^2) = 2 := by sorry

end prob1_solution_prob2_solution_l731_731316


namespace planted_fraction_correct_l731_731533

-- Define the given conditions
def right_triangle (a b : ℕ) (hypotenuse : ℝ) : Prop := hypotenuse = real.sqrt (a^2 + b^2)

def point (x y : ℝ) : Prop := true

def rectangle (a : ℝ) (x y : ℝ) (d : ℝ) : Prop :=
  12 * x + 5 * y = 21 ∨ 12 * x + 5 * y = 99 ∧ d = 3

def field_area (a b : ℕ) : ℝ := (a * b) / 2

def rectangle_area (x y : ℝ) : ℝ := x * y

def planted_fraction (triangle_area rectangle_area : ℝ) : ℝ := (triangle_area - rectangle_area) / triangle_area

-- The main theorem to prove
theorem planted_fraction_correct :
  let triangle_area := field_area 5 12 in
  let rectangle_x := 7 / 4 in
  let rectangle_y := 21 / 5 in
  let rectangle_area := rectangle_area rectangle_x rectangle_y in
  let planted_fraction := planted_fraction triangle_area rectangle_area in
  planted_fraction = 151 / 200 :=
sorry

end planted_fraction_correct_l731_731533


namespace sum_of_solutions_is_2009_l731_731572

theorem sum_of_solutions_is_2009
  (f : ℝ → ℝ)
  (H1 : ∀ x : ℝ, f (1 - x) = f (1 + x))
  (H2 : (f x = 0).Finset.card = 2009) :
  Finset.sum (Finset.filter (λ x, f x = 0) (Finset.range 2009)) id = 2009
:= sorry

end sum_of_solutions_is_2009_l731_731572


namespace max_profit_analytical_expression_l731_731336
noncomputable def T (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 25 then (x^2 / 20) + 50
else if 25 < x ∧ x ≤ 50 then (91 * x) / (3 + x)
else 0

noncomputable def cost (x : ℝ) : ℝ := 6 * x

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 25 then 30 * ((x^2 / 20) + 50) - cost(x)
else if 25 < x ∧ x ≤ 50 then 30 * ((91 * x) / (3 + x)) - cost(x)
else 0

theorem max_profit (x : ℝ) : ∃ x : ℝ, 0 < x ∧ x ≤ 50 ∧ f x = 11520 / 5 :=
begin
  sorry
end

theorem analytical_expression (x : ℝ) (hx : 0 < x ∧ x ≤ 25 ∨ 25 < x ∧ x ≤ 50) :
  f x = if 0 < x ∧ x ≤ 25 then (3 / 2) * x^2 - 6 * x + 1500
        else 2730 * x / (3 + x) - 6 * x :=
begin
  sorry
end

end max_profit_analytical_expression_l731_731336


namespace exists_1997_digit_composite_number_l731_731334

theorem exists_1997_digit_composite_number : ∃ N : ℕ, (1996 < (nat_num_digits N)) ∧ (∀ digits : fin 1995 → ℕ, (∀ pos : ℕ, pos < 1995 → digits pos < 10) → (nat_val_replace_triple N digits).is_composite) :=
sorry

end exists_1997_digit_composite_number_l731_731334


namespace fraction_spent_toy_store_l731_731609

noncomputable def student_weekly_allowance : ℝ := 3.75
noncomputable def fraction_spent_arcade : ℝ := 3/5
noncomputable def amount_spent_candy_store : ℝ := 1.00

theorem fraction_spent_toy_store :
  let remaining_after_arcade := student_weekly_allowance * (1 - fraction_spent_arcade)
  let amount_spent_toy_store := remaining_after_arcade - amount_spent_candy_store
  (amount_spent_toy_store / remaining_after_arcade) = 1/3 :=
by
  let remaining_after_arcade := student_weekly_allowance * (1 - fraction_spent_arcade)
  let amount_spent_toy_store := remaining_after_arcade - amount_spent_candy_store
  show (amount_spent_toy_store / remaining_after_arcade) = 1/3 from sorry

end fraction_spent_toy_store_l731_731609


namespace ones_digit_of_three_to_fourth_power_l731_731543

theorem ones_digit_of_three_to_fourth_power :
  Nat.ones_digit ((3 ^ (Nat.factors 9 3).sum) = 1 := by
    sorry

end ones_digit_of_three_to_fourth_power_l731_731543


namespace exists_infinite_a_no_solution_tau_eqn_l731_731300

noncomputable def tau (n : ℕ) : ℕ :=
  if n = 0 then 0 else
    (@Multiset.card ℕ _ ∘ @Multiset.toFinset ℕ _) (List.filter (λ d, n % d = 0) (List.range (n + 1)))

theorem exists_infinite_a_no_solution_tau_eqn :
  ∃ᶠ a : ℕ, ¬∃ n : ℕ, tau (a * n) = n := sorry

end exists_infinite_a_no_solution_tau_eqn_l731_731300


namespace find_a_100_l731_731760

-- Define the sequence a_n recursively
def a : ℕ → ℕ
| 0       := 1  -- Note: Lean sequence typically starts from 0, for compatibility
| (n + 1) := a n + (2 * a n) / n

-- Define the statement we want to prove: a_100 = 5151
theorem find_a_100 : a 100 = 5151 :=
sorry

end find_a_100_l731_731760


namespace average_rainfall_virginia_l731_731766

noncomputable def average_rainfall : ℝ :=
  (3.79 + 4.5 + 3.95 + 3.09 + 4.67) / 5

theorem average_rainfall_virginia : average_rainfall = 4 :=
by
  sorry

end average_rainfall_virginia_l731_731766


namespace eventually_reach_five_l731_731065

def is_prime (p : ℕ) : Prop := Nat.Prime p

def slip (n p : ℕ) : ℕ := (n + p * p) / p

theorem eventually_reach_five (n : ℕ) (h : n ≥ 5) :
  ∃ k, (∀ p, is_prime p → p ∣ (slip^[k] n) → slip^[k] n = 5) ∨ (∃ m, slip^[m] n = 5) :=
sorry

end eventually_reach_five_l731_731065


namespace find_angle_CEM_l731_731691

-- Define the points A, B, C, D, E, and M as well as the circles ω1 and ω2
variables (A B C M D E : Type*)
          [Point A] [Point B] [Point C] [Point M] [Point D] [Point E]
          (ω1 ω2 : Circle)

-- Definitions of geometric properties given in the problem
variable [is_midpoint A B M]
variable [tangent_to_line ω1 A C]
variable [circle_passes_through ω1 A M]
variable [tangent_to_line ω2 B C]
variable [circle_passes_through ω2 B M]
variable [intersects_again_at ω1 ω2 D]
variable [is_symmetric E D A B]

-- Theorem statement to prove the angle CEM is 180 degrees
theorem find_angle_CEM :
  angle C E M = 180 :=
sorry

end find_angle_CEM_l731_731691


namespace max_volume_of_hollow_cube_l731_731008

/-- 
We have 1000 solid cubes with edge lengths of 1 unit each. 
The small cubes can be glued together but not cut. 
The cube to be created is hollow with a wall thickness of 1 unit.
Prove that the maximum external volume of the cube we can create is 2197 cubic units.
--/

theorem max_volume_of_hollow_cube :
  ∃ x : ℕ, 6 * x^2 - 12 * x + 8 ≤ 1000 ∧ x^3 = 2197 :=
sorry

end max_volume_of_hollow_cube_l731_731008


namespace sum_of_n_l731_731875

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l731_731875


namespace hyperbola_sufficient_condition_l731_731231

theorem hyperbola_sufficient_condition (m : ℝ)
  (h : m > 4) : (5 < m) :=
begin
  exact lt_of_lt_of_le (by linarith) h,
  sorry
end

end hyperbola_sufficient_condition_l731_731231


namespace simplest_sqrt_l731_731440

-- Define the given square roots
def sqrt6 := Real.sqrt 6
def sqrt8 := Real.sqrt 8
def sqrt_one_over_three := Real.sqrt (1/3)
def sqrt4 := Real.sqrt 4

-- State that sqrt6 is the simplest among the given square roots
theorem simplest_sqrt : sqrt6 = Real.sqrt 6 ∧
  ¬ (∃ (x : ℝ), x^2 = 6 ∧ x < sqrt6) ∧
  (sqrt8 ≠ Real.sqrt 8 ∨ sqrt8 = 2*Real.sqrt 2) ∧
  (sqrt_one_over_three ≠ Real.sqrt (1/3) ∨ House.ksqrt (1/Real.sqrt 3) * Real.sqrt 3 ≠ Real.sqrt (1/3)) ∧
  (sqrt4 ≠ Real.sqrt 4 ∨ sqrt4 = 2) :=
sorry

end simplest_sqrt_l731_731440


namespace probability_sum_less_than_product_l731_731832

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731832


namespace piglet_straws_l731_731420

noncomputable def straws_per_piglet (total_straws : ℕ) (adult_pigs_fraction piglets_fraction : ℚ)
(piglets_count goats_count : ℕ): ℕ :=
let adult_pigs_straws := adult_pigs_fraction * total_straws in
let piglets_straws := piglets_fraction * total_straws in
let remaining_straws := total_straws - (adult_pigs_straws + piglets_straws) in
(piglets_straws / piglets_count)

theorem piglet_straws : straws_per_piglet 300 (7/15 : ℚ) (2/5 : ℚ) 20 10 = 6 :=
by
  sorry

end piglet_straws_l731_731420


namespace no_points_equidistant_circle_tangents_l731_731972

theorem no_points_equidistant_circle_tangents {r s : ℝ} (hs : s > 0): 
  let O : Point := (0, 0)
  let C : Circle := { center := O, radius := r }
  let tangent1 : Line := { slope := 0, intercept := r + s }
  let tangent2 : Line := { slope := 0, intercept := r + 2s }
  number_of_points_equidistant_from_circle_and_tangents C tangent1 tangent2 = 0 :=
by
  sorry

end no_points_equidistant_circle_tangents_l731_731972


namespace geometric_series_sum_positive_l731_731193

theorem geometric_series_sum_positive (a1 t : ℝ) (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - t ^ n) / (1 - t))
  (h_pos : ∀ n, S n > 0) :
  t ∈ Ioo (-1 : ℝ) (0 : ℝ) ∨ t ∈ Ioi (0 : ℝ) := 
sorry

end geometric_series_sum_positive_l731_731193


namespace count_numbers_with_property_l731_731007

open Nat

theorem count_numbers_with_property : 
  let N := { n : ℕ | n < 10^6 ∧ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 43 ∧ 2012 ∣ n^k - 1 }
  N.card = 1988 :=
by
  sorry

end count_numbers_with_property_l731_731007


namespace number_of_terms_in_sequence_l731_731981

theorem number_of_terms_in_sequence (a d l : ℤ) (h_a : a = -5) (h_d : d = 5) (h_l : l = 55) :
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 13 :=
by
  existsi 13
  split
  sorry

end number_of_terms_in_sequence_l731_731981


namespace incorrect_calculation_l731_731711

noncomputable def ξ : ℝ := 3 -- Expected lifetime of the sensor
noncomputable def η : ℝ := 5 -- Expected lifetime of the transmitter
noncomputable def T (ξ η : ℝ) : ℝ := min ξ η -- Lifetime of the entire device

theorem incorrect_calculation (h1 : E ξ = 3) (h2 : E η = 5) (h3 : E (min ξ η ) = 3.67) : False :=
by
  have h4 : E (min ξ η ) ≤ 3 := sorry -- Based on properties of expectation and min
  have h5 : 3.67 > 3 := by linarith -- Known inequality
  sorry

end incorrect_calculation_l731_731711


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l731_731122

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l731_731122


namespace total_cost_is_correct_l731_731157

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end total_cost_is_correct_l731_731157


namespace find_x_l731_731049

theorem find_x (x : ℤ) (h : (1 + 2 + 4 + 5 + 6 + 9 + 9 + 10 + 12 + x) / 10 = 7) : x = 12 :=
by
  sorry

end find_x_l731_731049


namespace sequence_eventually_periodic_modulo_l731_731142

noncomputable def a_n (n : ℕ) : ℕ :=
  n ^ n + (n - 1) ^ (n + 1)

theorem sequence_eventually_periodic_modulo (m : ℕ) (hm : m > 0) : ∃ K s : ℕ, ∀ k : ℕ, (K ≤ k → a_n (k) % m = a_n (k + s) % m) :=
sorry

end sequence_eventually_periodic_modulo_l731_731142


namespace burgers_ordered_l731_731945

theorem burgers_ordered (H : ℕ) (Ht : H + 2 * H = 45) : 2 * H = 30 := by
  sorry

end burgers_ordered_l731_731945


namespace interval_increase_evaluate_at_alpha_l731_731605

noncomputable def f (x : ℝ) : ℝ :=
  let a := (Real.sin x, Real.cos x)
  let b := (Real.cos (x + π/6) + Real.sin x, Real.cos x)
  a.1 * b.1 + a.2 * b.2

theorem interval_increase (k : ℤ) : 
  ∃ I : Set ℝ, (I = Set.Icc (k * π - π / 3) (k * π + π / 6)) ∧
  ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y :=
sorry

theorem evaluate_at_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 12) = 1 / 3) : 
  f α = (2 * Real.sqrt 2) / 9 + 3 / 4 :=
sorry

end interval_increase_evaluate_at_alpha_l731_731605


namespace probability_sum_less_than_product_l731_731836

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731836


namespace minimum_area_of_square_with_four_interior_lattice_points_l731_731088

-- Definitions of the conditions
def square_contains_only_four_interior_lattice_points (s : set (ℤ × ℤ)) : Prop :=
  ∃ p q r t : ℤ × ℤ, p ≠ q ∧ q ≠ r ∧ r ≠ t ∧ t ≠ p ∧ p ≠ r ∧ q ≠ t ∧ 
  (0, 0) ∈ s ∧ p ∈ s ∧ q ∈ s ∧ r ∈ s ∧ t ∈ s ∧ 
  (∀ x : ℤ × ℤ, (x ∈ s → x = (0, 0) ∨ x = p ∨ x = q ∨ x = r ∨ x = t)) ∧
  (∀ x : ℤ × ℤ, (x ∈ s → x.1 ≠ 0 ∨ x.2 ≠ 0))

def square_has_no_border_lattice_points (s : set (ℤ × ℤ)) : Prop :=
  ∀ x : ℤ × ℤ, x ∈ s → (x.1 ≠ 0 ∨ x.2 ≠ 0)

-- Statement of the theorem/proof problem
theorem minimum_area_of_square_with_four_interior_lattice_points : 
  ∀ s : set (ℤ × ℤ), square_contains_only_four_interior_lattice_points s ∧ square_has_no_border_lattice_points s → 
  ∃ a : ℕ, a = 16 := 
by 
  sorry

end minimum_area_of_square_with_four_interior_lattice_points_l731_731088


namespace remaining_fruits_correct_l731_731106

-- The definitions for the number of fruits in terms of the number of plums
def apples := 180
def plums := apples / 3
def pears := 2 * plums
def cherries := 4 * apples

-- Damien's portion of each type of fruit picked
def apples_picked := (3/5) * apples
def plums_picked := (2/3) * plums
def pears_picked := (3/4) * pears
def cherries_picked := (7/10) * cherries

-- The remaining number of fruits
def apples_remaining := apples - apples_picked
def plums_remaining := plums - plums_picked
def pears_remaining := pears - pears_picked
def cherries_remaining := cherries - cherries_picked

-- The total remaining number of fruits
def total_remaining_fruits := apples_remaining + plums_remaining + pears_remaining + cherries_remaining

theorem remaining_fruits_correct :
  total_remaining_fruits = 338 :=
by {
  -- The conditions ensure that the imported libraries are broad
  sorry
}

end remaining_fruits_correct_l731_731106


namespace complex_modulus_z_l731_731580

-- Define the complex number z with given conditions
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I + Complex.I

-- State the theorem to be proven
theorem complex_modulus_z : Complex.abs z = Real.sqrt 2 := 
sorry

end complex_modulus_z_l731_731580


namespace number_of_subsets_with_property_l731_731146

theorem number_of_subsets_with_property :
  let S := {1,2,3,4,5,6,7,8,9,10}
  (∀ a b c ∈ S, a < b < c → (a ∈ S) ∧ (b ∉ S) ∧ (c ∈ S)) →
  (∃ (n : ℕ), n = 968) :=
sorry

end number_of_subsets_with_property_l731_731146


namespace apples_sold_fresh_l731_731736

-- Definitions per problem conditions
def total_production : Float := 8.0
def initial_percentage_mixed : Float := 0.30
def percentage_increase_per_million : Float := 0.05
def percentage_for_apple_juice : Float := 0.60
def percentage_sold_fresh : Float := 0.40

-- We need to prove that given the conditions, the amount of apples sold fresh is 2.24 million tons
theorem apples_sold_fresh :
  ( (total_production - (initial_percentage_mixed * total_production)) * percentage_sold_fresh = 2.24 ) :=
by
  sorry

end apples_sold_fresh_l731_731736


namespace probability_ab_gt_a_add_b_l731_731796

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731796


namespace probability_sum_less_than_product_l731_731835

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731835


namespace Tn_lt_1_over_4_l731_731086

noncomputable def a (n : ℕ) : ℕ :=
  2 * n + 1

def b (n : ℕ) : ℝ :=
  1 / ((a n) ^ 2 - 1)

def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_1_over_4 (n : ℕ) : T n < 1 / 4 :=
by
  sorry

end Tn_lt_1_over_4_l731_731086


namespace four_digit_numbers_count_l731_731248

theorem four_digit_numbers_count : (card (set.Icc 1000 2000) = 1001) :=
sorry

end four_digit_numbers_count_l731_731248


namespace problem_I_problem_II_l731_731589

-- First proof problem
theorem problem_I (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a⁻¹ = 3) : 
  f 1 (1 / 2)=(sqrt 5) 

-- Second proof problem
theorem problem_II (a : ℝ) (k : ℤ) (h1 : a > 1) (h2 : is_odd_function (λ x, a^x + k * a^(-x))) 
  : ∀ x ∈ Icc 0 (2 * Real.pi / 3), ∀ λ, λ ≥ 3 → (a^x + k * a^(-x)) +
  (a^(2*λ*sin x - 5) + k * a^(-(2*λ*sin x-5))) < 0 :=
begin
  assume a k h1 h2 x h4 λ h5,
  sorry,
end

end problem_I_problem_II_l731_731589


namespace intersection_A_B_l731_731262

def A : set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : set ℝ := {x | (x - 2)*(x - 5) < 0}
def C : set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_A_B :
  (A ∩ B) = C :=
sorry

end intersection_A_B_l731_731262


namespace cos_330_l731_731904

def cos_330_eq : Prop :=
  cos (330 * (Real.pi / 180)) = Real.sqrt 3 / 2

theorem cos_330 : cos_330_eq := by
  sorry

end cos_330_l731_731904


namespace eccentricity_of_ellipse_with_given_conditions_l731_731458

variables {a c : ℝ}
-- Foci F1 and F2, center of the ellipse, and the point M on the ellipse
def is_right_angle (P Q R : ℝ × ℝ) : Prop := 
  let d1 := dist P Q in
  let d2 := dist Q R in
  let d3 := dist P R in
  d1 ^ 2 + d2 ^ 2 = d3 ^ 2

def ellipse_eccentricity (e : ℝ) (a c : ℝ) : Prop :=
  0 < c ∧ 0 < a ∧ (2 * a - c)^2 + c^2 = 4 * c^2 ∧ e = c / a

theorem eccentricity_of_ellipse_with_given_conditions :
  ellipse_eccentricity (sqrt 3 - 1) a c :=
by
  -- proof here
  sorry

end eccentricity_of_ellipse_with_given_conditions_l731_731458


namespace sum_of_valid_n_l731_731881

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l731_731881


namespace great_circles_divide_sphere_l731_731062

theorem great_circles_divide_sphere (n : ℕ) (hn : n ≥ 1) :
  ∃ F : ℕ, F = n^2 - n + 2 := 
  by
    use n^2 - n + 2
    sorry

end great_circles_divide_sphere_l731_731062


namespace segment_shadow_proportion_l731_731085

-- Defining the segment lengths
def segment_length (l : Type*) [field l] (AB A'B' CD C'D' : l) : Prop :=
  (AB = 3 * A'B') → (CD = 3 * C'D')

-- Theorem statement
theorem segment_shadow_proportion
  (l : Type*) [field l] (AB A'B' CD C'D' : l) (h : AB = 3 * A'B') : CD = 3 * C'D' :=
sorry

end segment_shadow_proportion_l731_731085


namespace average_marks_l731_731747

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l731_731747


namespace largest_number_of_subsets_l731_731858

theorem largest_number_of_subsets (n : ℕ) :
  let S := finset.range (2 * n + 2) in 
  (∃ T : finset (finset ℕ), (∀ A B ∈ T, A ≠ B → ∃ k : ℕ, (A ∩ B) = finset.range k ∧ 2 ≤ k) 
  ∧ T.card = (2 * n + 1) * (2 * n + 1) / 4)
:= sorry

end largest_number_of_subsets_l731_731858


namespace log_eight_of_five_twelve_l731_731989

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731989


namespace common_point_eq_l731_731397

theorem common_point_eq (a b c d : ℝ) (h₀ : a ≠ b) 
  (h₁ : ∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) : 
  d = c :=
by
  sorry

end common_point_eq_l731_731397


namespace molecular_weight_of_7_moles_AlOH3_l731_731431

theorem molecular_weight_of_7_moles_AlOH3 (
  atomic_weight_Al : ℝ := 26.98,
  atomic_weight_O : ℝ := 16.00,
  atomic_weight_H : ℝ := 1.01) :
  7 * (atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H) = 546.07 :=
by
  -- We assume the periodic table values for the atomic weights are correct 
  have molecular_weight_AlOH3 : ℝ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H
  -- Now we calculate 7 times the molecular weight
  have weight_7_moles : ℝ := 7 * molecular_weight_AlOH3
  -- Finally, we compare it to the given correct answer
  simp only [molecular_weight_AlOH3, weight_7_moles]
  sorry -- Proof will be completed here

end molecular_weight_of_7_moles_AlOH3_l731_731431


namespace ray_KA_angle_bisector_l731_731272

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry  -- Define circumcenter
noncomputable def reflection (P : Point) (l : Line) : Point := sorry  -- Define reflection
noncomputable def line_of_points (A B : Point) : Line := sorry  -- Line through two points
noncomputable def intersection (l1 l2 : Line) : Point := sorry  -- Intersection of two lines
noncomputable def angle_bisector (A B C : Point) : Line := sorry  -- Angle bisector

theorem ray_KA_angle_bisector (A B C K O B1: Point) (h : Triangle A B C) :
  O = circumcenter A B C →
  B1 = reflection B (line_of_points A C) →
  K = intersection (line_of_points A O) (line_of_points B1 C) →
  ∃ l : Line, l = angle_bisector K A B1 ∧ on_ray K A (point_on_bisector K A B B1) :=
sorry

end ray_KA_angle_bisector_l731_731272


namespace circle_area_with_diameter_10_l731_731862

noncomputable def circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  in π * (r * r)

theorem circle_area_with_diameter_10 : circle_area 10 = 25 * π := 
by
  -- sorry is added to skip the actual proof, the statement is our goal
  sorry

end circle_area_with_diameter_10_l731_731862


namespace solve_inequality_l731_731356

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l731_731356


namespace probability_sum_less_than_product_l731_731791

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731791


namespace number_with_20_multiples_l731_731410

theorem number_with_20_multiples : ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k) → (k ≤ 100) → (n ∣ k) → (k / n ≤ 20) ) ∧ n = 5 := 
  sorry

end number_with_20_multiples_l731_731410


namespace total_cost_after_discounts_and_cashback_l731_731149

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l731_731149


namespace students_only_one_activity_l731_731417

theorem students_only_one_activity 
  (total : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 317) 
  (h_both : both = 30) 
  (h_neither : neither = 20) : 
  (total - both - neither) = 267 :=
by 
  sorry

end students_only_one_activity_l731_731417


namespace abs_inequality_l731_731364

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l731_731364


namespace sin_comparison_l731_731968

theorem sin_comparison : 
  (-π / 18 ∈ Icc (-π / 2) 0) → 
  (-π / 10 ∈ Icc (-π / 2) 0) → 
  (-π / 18 > -π / 10) → 
  (sin (-π / 18) > sin (-π / 10)) :=
by
  intros h1 h2 h3
  sorry

end sin_comparison_l731_731968


namespace solution_set_of_quadratic_l731_731228

theorem solution_set_of_quadratic (a b x : ℝ) (h1 : a = 5) (h2 : b = -6) :
  (2 ≤ x ∧ x ≤ 3) → (bx^2 - ax - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end solution_set_of_quadratic_l731_731228


namespace sqrt_simplest_l731_731437

def is_simplest (sq_root : ℝ) : Prop :=
  (sq_root == real.sqrt 6)

theorem sqrt_simplest : is_simplest (real.sqrt 6) :=
by
  sorry

end sqrt_simplest_l731_731437


namespace unit_vector_subtraction_l731_731244

noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let norm := (v.1 ^ 2 + v.2 ^ 2).sqrt
  (v.1 / norm, v.2 / norm)

theorem unit_vector_subtraction :
  let a := (3, 1 : ℝ)
  let b := (7, -2 : ℝ)
  unit_vector (a.1 - b.1, a.2 - b.2) = (-4 / 5, 3 / 5) :=
  sorry

end unit_vector_subtraction_l731_731244


namespace negation_of_proposition_l731_731206

theorem negation_of_proposition (a b : ℝ) : 
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) :=
by sorry

end negation_of_proposition_l731_731206


namespace g_of_f_of_3_is_1852_l731_731306

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 - x + 2

theorem g_of_f_of_3_is_1852 : g (f 3) = 1852 := by
  sorry

end g_of_f_of_3_is_1852_l731_731306


namespace min_value_expression_l731_731203

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  ∃ c, c = (1/(a+1) + 4/(b+1)) ∧ c ≥ 9/4 :=
by
  sorry

end min_value_expression_l731_731203


namespace arithmetic_sequence_properties_l731_731949

-- Defining the arithmetic sequence and the conditions
variable {a : ℕ → ℤ}
variable {d : ℤ}
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = d

-- Given conditions
variable (h1 : a 5 = 10)
variable (h2 : a 1 + a 2 + a 3 = 3)

-- The theorem to prove
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d → a 1 = -2 ∧ d = 3 :=
sorry

end arithmetic_sequence_properties_l731_731949


namespace value_of_c_l731_731973

-- The conditions
variables (a b c : ℝ)

-- The equation of the parabola
def parabola := ∀ x : ℝ, y = a * x^2 + b * x + c

-- The vertex of the parabola
def vertex := y = a * (3 : ℝ)^2 + b * 3 + (-5 : ℝ)

-- The point (4, -3) is on the parabola
def point := parabola 4 = (4: ℝ) * a^2 + b * 4 + c = -3

-- The statement to prove
theorem value_of_c (h : vertex) (p : point) :
  c = 13 :=
sorry

end value_of_c_l731_731973


namespace probability_sum_less_than_product_is_5_div_9_l731_731806

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731806


namespace hyperbola_eccentricity_l731_731303

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (FP FQ : ℝ) (h_FP : FP = 3 * FQ) : 
  let e := sqrt (1 + b^2 / a^2) in e = sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l731_731303


namespace solve_for_x_l731_731909

theorem solve_for_x (x : ℝ) (h₁: 0.45 * x = 0.15 * (1 + x)) : x = 0.5 :=
by sorry

end solve_for_x_l731_731909


namespace total_cost_after_discounts_and_cashback_l731_731151

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l731_731151


namespace negation_of_proposition_l731_731594

theorem negation_of_proposition (p : ∀ x : ℝ, Real.sin x ≤ 1) :
  ¬ (∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
begin
  sorry
end

end negation_of_proposition_l731_731594


namespace charity_meaning_l731_731048

theorem charity_meaning (noun_charity : String) (h : noun_charity = "charity") : 
  (noun_charity = "charity" → "charity" = "charitable organization") :=
by
  sorry

end charity_meaning_l731_731048


namespace expand_binom_l731_731529

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l731_731529


namespace number_of_inverses_modulo_12_l731_731251

def has_inverse_mod_12 (n : ℤ) : Prop :=
  ∃ m : ℤ, (n * m) % 12 = 1

def count_inverses_mod_12 (a b : ℤ) : ℕ :=
  Set.card {n | a ≤ n ∧ n ≤ b ∧ Nat.gcd n 12 = 1}

theorem number_of_inverses_modulo_12 : count_inverses_mod_12 0 11 = 4 := sorry

end number_of_inverses_modulo_12_l731_731251


namespace original_avg_is_40_l731_731663

noncomputable def original_average (A : ℝ) := (15 : ℝ) * A

noncomputable def new_sum (A : ℝ) := (15 : ℝ) * A + 15 * (15 : ℝ)

theorem original_avg_is_40 (A : ℝ) (h : new_sum A / 15 = 55) :
  A = 40 :=
by sorry

end original_avg_is_40_l731_731663


namespace candy_packs_l731_731515

theorem candy_packs : (let trays := 4 in
                       let cookies_per_tray := 24 in
                       let total_cookies := trays * cookies_per_tray in
                       let cookies_per_pack := 12 in
                       total_cookies / cookies_per_pack) = 8 :=
by
  sorry

end candy_packs_l731_731515


namespace mika_saucer_surface_area_l731_731678

noncomputable def surface_area_saucer (r h rim_thickness : ℝ) : ℝ :=
  let A_cap := 2 * Real.pi * r * h  -- Surface area of the spherical cap
  let R_outer := r
  let R_inner := r - rim_thickness
  let A_rim := Real.pi * (R_outer^2 - R_inner^2)  -- Area of the rim
  A_cap + A_rim

theorem mika_saucer_surface_area :
  surface_area_saucer 3 1.5 1 = 14 * Real.pi :=
sorry

end mika_saucer_surface_area_l731_731678


namespace sum_of_n_l731_731876

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l731_731876


namespace chicken_coop_problem_l731_731470

-- Definitions of conditions
def available_area : ℝ := 240
def area_per_chicken : ℝ := 4
def area_per_chick : ℝ := 2
def max_daily_feed : ℝ := 8000
def feed_per_chicken : ℝ := 160
def feed_per_chick : ℝ := 40

-- Variables representing the number of chickens and chicks
variables (x y : ℕ)

-- Condition expressions
def space_condition (x y : ℕ) : Prop := 
  (2 * x + y = (available_area / area_per_chick))

def feed_condition (x y : ℕ) : Prop := 
  ((4 * x + y) * feed_per_chick <= max_daily_feed / feed_per_chick)

-- Given conditions and queries proof problem
theorem chicken_coop_problem : 
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 20 ∧ y = 80)) 
  ∧
  (¬ ∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 30 ∧ y = 100))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 40 ∧ y = 40))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 0 ∧ y = 120)) :=
by
  sorry  -- The proof will be provided here.


end chicken_coop_problem_l731_731470


namespace four_digit_number_multiple_of_5_probability_l731_731066

theorem four_digit_number_multiple_of_5_probability : 
  let digits := {1, 3, 5, 7, 9}
  let total_ways := 5 * 4 * 3 * 2
  let favorable_ways := 3 * 2 * 1
  (favorable_ways / total_ways: ℚ) = 1 / 20 :=
by
  let digits := {1, 3, 5, 7, 9}
  let total_ways := 5 * 4 * 3 * 2
  let favorable_ways := 3 * 2 * 1
  have h : (favorable_ways / total_ways: ℚ) = 1 / 20 := by norm_num
  exact h

end four_digit_number_multiple_of_5_probability_l731_731066


namespace sum_of_ages_l731_731739

theorem sum_of_ages (y : ℕ) 
  (h_diff : 38 - y = 2) : y + 38 = 74 := 
by {
  sorry
}

end sum_of_ages_l731_731739


namespace probability_sum_less_than_product_l731_731843

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731843


namespace gemstones_needed_l731_731699

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l731_731699


namespace log_eight_of_five_twelve_l731_731993

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731993


namespace fifth_coin_touching_four_exists_l731_731773

theorem fifth_coin_touching_four_exists :
  ∃ fifth_coin : ℝ × ℝ,
    (∀ i j, i < j → (EuclideanDist (coin_pos i) (coin_pos j) = coin_diameter)) →
    (∀ i < 4, EuclideanDist (fifth_coin) (coin_pos i) = coin_diameter / 2) := 
by
  -- Define positions for the first four coins.
  let coin_pos : ℕ → ℝ × ℝ := λ i,
    if i = 0 then (0, 0)
    else if i = 1 then (coin_diameter, 0)
    else if i = 2 then (coin_diameter / 2, (coin_diameter * sqrt 3 / 2))
    else (coin_diameter / 2, (coin_diameter * sqrt 3 / 6))
  -- Diameter of the coins
  let coin_diameter : ℝ := 1 
  -- Placeholder for the actual proof
  sorry

end fifth_coin_touching_four_exists_l731_731773


namespace hclo4_needed_moles_l731_731611

-- Start the Lean 4 statement
theorem hclo4_needed_moles (moles_NaOH : ℝ) 
  (partial_dissociation_ratio : ℝ) 
  (effective_moles_HClO4 : ℝ) :
  moles_NaOH = 3 →
  partial_dissociation_ratio = 0.80 →
  effective_moles_HClO4 = moles_NaOH →
  (initial_moles_HClO4 : ℝ), 
    initial_moles_HClO4 = effective_moles_HClO4 / partial_dissociation_ratio → initial_moles_HClO4 = 3.75 :=
begin
  intros h1 h2 h3,
  -- The theorem statement ensures that the math problem
  -- will execute correctly with the given conditions.
  sorry, -- Proof is not required as per instructions
end

end hclo4_needed_moles_l731_731611


namespace find_triples_l731_731537

theorem find_triples (x n p : ℕ) (hp : Nat.Prime p) 
  (hx_pos : x > 0) (hn_pos : n > 0) : 
  x^3 + 3 * x + 14 = 2 * p^n → (x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5) :=
by 
  sorry

end find_triples_l731_731537


namespace gemstones_needed_l731_731695

-- Define the initial quantities and relationships
def magnets_per_earring := 2
def buttons_per_magnet := 1 / 2
def gemstones_per_button := 3
def earrings_per_set := 2
def sets_of_earrings := 4

-- Define the total gemstones needed
theorem gemstones_needed : 
    let earrings := sets_of_earrings * earrings_per_set in
    let total_magnets := earrings * magnets_per_earring in
    let total_buttons := total_magnets * buttons_per_magnet in
    let total_gemstones := total_buttons * gemstones_per_button in
    total_gemstones = 24 :=
by
    have earrings := 2 * 4
    have total_magnets := earrings * 2
    have total_buttons := total_magnets / 2
    have total_gemstones := total_buttons * 3
    exact eq.refl 24

end gemstones_needed_l731_731695


namespace traveler_drank_32_ounces_l731_731493

-- Definition of the given condition
def total_gallons : ℕ := 2
def ounces_per_gallon : ℕ := 128
def total_ounces := total_gallons * ounces_per_gallon
def camel_multiple : ℕ := 7
def traveler_ounces (T : ℕ) := T
def camel_ounces (T : ℕ) := camel_multiple * T
def total_drunk (T : ℕ) := traveler_ounces T + camel_ounces T

-- Theorem to prove
theorem traveler_drank_32_ounces :
  ∃ T : ℕ, total_drunk T = total_ounces ∧ T = 32 :=
by 
  sorry

end traveler_drank_32_ounces_l731_731493


namespace necessary_but_not_sufficient_condition_for_ellipse_l731_731521

def constant_sum_of_distances (M F1 F2 : Point) (k : Real) : Prop :=
  distance M F1 + distance M F2 = k

def is_ellipse (M F1 F2 : Point) (k c : Real) : Prop :=
  c < k ∧ constant_sum_of_distances M F1 F2 k

theorem necessary_but_not_sufficient_condition_for_ellipse (M F1 F2 : Point) (k c : Real) :
  (constant_sum_of_distances M F1 F2 k) → (is_ellipse M F1 F2 k c → constant_sum_of_distances M F1 F2 k) := 
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l731_731521


namespace min_value_xy_k_l731_731430

theorem min_value_xy_k (x y k : ℝ) : ∃ x y : ℝ, (xy - k)^2 + (x + y - 1)^2 = 1 := by
  sorry

end min_value_xy_k_l731_731430


namespace S_4_6_eq_640_l731_731569

noncomputable def f (i k : Nat) : ℕ := i * 2^(k - 1)

noncomputable def g (i n : Nat) : ℕ := 
  (List.range (i + 1)).sum (λ j => f (2^j) n)

noncomputable def S (m n : Nat) : ℕ := 
  (List.range m).sum (λ i => (-1)^(i+1) * g (i+1) n)

theorem S_4_6_eq_640 : S 4 6 = 640 := 
by
  sorry

end S_4_6_eq_640_l731_731569


namespace nancy_shoes_l731_731685

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l731_731685


namespace probability_sum_less_than_product_l731_731848

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731848


namespace major_axis_of_ellipse_l731_731936

noncomputable def major_axis_length (r : ℝ) : ℝ :=
let minor_axis := 2 * r in
2 * minor_axis

theorem major_axis_of_ellipse (r : ℝ) (h : r = 2) :
  major_axis_length r = 8 :=
by
  rw [←h]
  unfold major_axis_length
  norm_num
  sorry

end major_axis_of_ellipse_l731_731936


namespace ABDC_is_parallelogram_l731_731658

variables {A B C M D : Point} (h_midpoint : M = midpoint B C) (h_AM_MD : dist A M = dist M D)

theorem ABDC_is_parallelogram 
  (h_dist_AD : dist A D = 2 * dist A M) 
  (h_midline : D = A + 2 * (M - A)) :
  parallelogram A B D C :=
begin
  sorry
end

end ABDC_is_parallelogram_l731_731658


namespace minimal_length_of_PQ_l731_731937

theorem minimal_length_of_PQ (A B C M P Q : ℝ) (hABC_acute : acute_triangle A B C)
(hM_on_AB : M ∈ (AB_segment A B)) (h_perp_MP_BC : is_perpendicular M P (BC_segment B C))
(h_perp_MQ_AC : is_perpendicular M Q (AC_segment A C)) (hM_foot : is_foot_of_perpendicular M C (AB_segment A B)) :
is_minimal_distance PQ (distance P Q) :=
by
  sorry

end minimal_length_of_PQ_l731_731937


namespace probability_ab_gt_a_add_b_l731_731800

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731800


namespace solution_set_eq_l731_731557

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x ∈ set.univ
axiom f_zero : f 0 = 2
axiom f_ineq : ∀ x : ℝ, f x + deriv f x < 1

theorem solution_set_eq :
  {x : ℝ | exp x * f x < exp x + 1} = set.Ioi (0 : ℝ) :=
sorry

end solution_set_eq_l731_731557


namespace incorrect_calculation_l731_731721

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l731_731721


namespace total_cost_after_discounts_l731_731153

theorem total_cost_after_discounts 
    (price_iphone : ℝ)
    (discount_iphone : ℝ)
    (price_iwatch : ℝ)
    (discount_iwatch : ℝ)
    (cashback_percentage : ℝ) :
    (price_iphone = 800) →
    (discount_iphone = 0.15) →
    (price_iwatch = 300) →
    (discount_iwatch = 0.10) →
    (cashback_percentage = 0.02) →
    let discounted_iphone := price_iphone * (1 - discount_iphone),
        discounted_iwatch := price_iwatch * (1 - discount_iwatch),
        total_discounted := discounted_iphone + discounted_iwatch,
        cashback := total_discounted * cashback_percentage 
    in total_discounted - cashback = 931 :=
by {
  intros,
  sorry
}

end total_cost_after_discounts_l731_731153


namespace shauna_min_test_score_l731_731637

theorem shauna_min_test_score (score1 score2 score3 : ℕ) (h1 : score1 = 82) (h2 : score2 = 88) (h3 : score3 = 95) 
  (max_score : ℕ) (h4 : max_score = 100) (desired_avg : ℕ) (h5 : desired_avg = 85) :
  ∃ (score4 score5 : ℕ), score4 ≥ 75 ∧ score5 ≥ 75 ∧ (score1 + score2 + score3 + score4 + score5) / 5 = desired_avg :=
by
  -- proof here
  sorry

end shauna_min_test_score_l731_731637


namespace benny_initial_comics_l731_731113

variable (x : ℕ)

def initial_comics (x : ℕ) : ℕ := x

def comics_after_selling (x : ℕ) : ℕ := (2 * x) / 5

def comics_after_buying (x : ℕ) : ℕ := (comics_after_selling x) + 12

def traded_comics (x : ℕ) : ℕ := (comics_after_buying x) / 4

def comics_after_trading (x : ℕ) : ℕ := (3 * (comics_after_buying x)) / 4 + 18

theorem benny_initial_comics : comics_after_trading x = 72 → x = 150 := by
  intro h
  sorry

end benny_initial_comics_l731_731113


namespace probability_sum_less_than_product_l731_731794

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731794


namespace tan_neg_585_eq_neg_1_l731_731519

theorem tan_neg_585_eq_neg_1 : Real.tan (-585 * Real.pi / 180) = -1 := by
  sorry

end tan_neg_585_eq_neg_1_l731_731519


namespace oliver_fresh_mango_dishes_l731_731082

def total_dishes : ℕ := 36
def mango_salsa_dishes : ℕ := 3
def mango_jelly_dishes : ℕ := 1
def fresh_mango_fraction : ℚ := 1 / 6
def dishes_left_for_oliver : ℕ := 28

def fresh_mango_dishes : ℕ := total_dishes * fresh_mango_fraction.natAbs
def dishes_oliver_wont_eat : ℕ := mango_salsa_dishes + mango_jelly_dishes
def potential_dishes_for_oliver : ℕ := dishes_left_for_oliver + dishes_oliver_wont_eat
def dishes_without_salsa_or_jelly : ℕ := total_dishes - potential_dishes_for_oliver

theorem oliver_fresh_mango_dishes : fresh_mango_dishes - dishes_without_salsa_or_jelly = 4 :=
by
  sorry

end oliver_fresh_mango_dishes_l731_731082


namespace last_digit_in_hundreds_position_is_zero_l731_731735

noncomputable def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

def appears_last (digit: ℕ) : Prop :=
    ∀ n, fib n % 100 = digit

theorem last_digit_in_hundreds_position_is_zero :
  appears_last 0 :=
sorry

end last_digit_in_hundreds_position_is_zero_l731_731735


namespace total_games_seventh_grade_score_l731_731916

theorem total_games (k : ℕ) : (k * (k - 1)) / 2 = (k * (k - 1)) / 2 :=
by sorry

theorem seventh_grade_score (n m : ℕ)
  (h1 : ∀ (n : ℕ), 10 * n = 10 * n)
  (h2 : ∀ (m : ℕ), 4.5 * m = 4.5 * m)
  (hn : n = 1) :
  m = n * (11 * n - 1) → m = 10 :=
by sorry

end total_games_seventh_grade_score_l731_731916


namespace solve_quadratic_l731_731347

noncomputable def quadratic_roots (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic : ∀ x : ℝ, quadratic_roots 1 (-4) (-5) x ↔ (x = -1 ∨ x = 5) :=
by
  intro x
  rw [quadratic_roots]
  sorry

end solve_quadratic_l731_731347


namespace acute_angle_MN_XY_l731_731628

theorem acute_angle_MN_XY
  (X Y Z : Point)
  (R S M N : Point)
  (XY XZ RS : Line)
  (a X36 Y80: Angle)
  (length_XY10 M_midpoint N_midpoint: Prop)
  (R_on_XZ S_on_XY : Prop) :
  ∠X.val = 36 ∧  
  ∠Y.val = 80 ∧  
  length XY = 10 ∧  
  R ∈ XZ ∧ 
  S ∈ XY ∧ 
  distance X R = 2 ∧ 
  distance S Y = 2 ∧ 
  mid_point M XY ∧ 
  mid_point N RS →
  acute_angle (angle_of MN XY) = 50 :=
by sorry

end acute_angle_MN_XY_l731_731628


namespace sum_of_n_l731_731871

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l731_731871


namespace total_shoes_l731_731686

variable boots : ℕ
variable slippers : ℕ
variable heels : ℕ

-- Condition: Nancy has six pairs of boots
def boots_pairs : boots = 6 := rfl

-- Condition: Nancy has nine more pairs of slippers than boots
def slippers_pairs : slippers = boots + 9 := rfl

-- Condition: Nancy has a number of pairs of heels equal to three times the combined number of slippers and boots
def heels_pairs : heels = 3 * (boots + slippers) := by
  rw [boots_pairs, slippers_pairs]
  sorry  -- assuming the correctness of the consequent computation as rfl

-- Goal: Total number of individual shoes is 168
theorem total_shoes : (boots * 2) + (slippers * 2) + (heels * 2) = 168 := by
  rw [boots_pairs, slippers_pairs, heels_pairs]
  sorry  -- verifying the summing up to 168 as a proof

end total_shoes_l731_731686


namespace nancy_shoes_l731_731680

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l731_731680


namespace sum_lt_prod_probability_l731_731826

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731826


namespace book_pairs_count_l731_731613

theorem book_pairs_count :
  let mystery_count := 3
  let fantasy_count := 4
  let biography_count := 3
  mystery_count * fantasy_count + mystery_count * biography_count + fantasy_count * biography_count = 33 :=
by 
  sorry

end book_pairs_count_l731_731613


namespace line_intersects_circle_l731_731288

noncomputable def equation_of_circle_polar (θ : ℝ) : ℝ := 2 * sqrt 2 * sin (θ + π / 4)

def parametric_equations_of_line_l (t : ℝ) : ℝ × ℝ := (t, 1 + 2 * t)

def cartesian_equation_of_line_l (x y : ℝ) : Prop := y = 2 * x + 1

def cartesian_equation_of_circle_c (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

def distance_between_center_and_line (x0 y0 : ℝ) : ℝ :=
  abs (2 * x0 - y0 + 1) / sqrt (2^2 + 1^2)

theorem line_intersects_circle : 
  ∃ (x y : ℝ), cartesian_equation_of_line_l x y ∧ cartesian_equation_of_circle_c x y :=
sorry

end line_intersects_circle_l731_731288


namespace incorrect_lifetime_calculation_l731_731728

-- Define expectation function
noncomputable def expectation (X : ℝ) : ℝ := sorry

-- We define the lifespans
variables (xi eta : ℝ)
-- Expected lifespan of the sensor and transmitter
axiom exp_xi : expectation xi = 3
axiom exp_eta : expectation eta = 5

-- Define the lifetime of the device
noncomputable def T := min xi eta

-- Given conditions
theorem incorrect_lifetime_calculation :
  expectation T ≤ 3 → 3 + (2 / 3) > 3 → false := 
sorry

end incorrect_lifetime_calculation_l731_731728


namespace bianca_initial_cupcakes_l731_731547

theorem bianca_initial_cupcakes (X : ℕ) (h : X - 6 + 17 = 25) : X = 14 := by
  sorry

end bianca_initial_cupcakes_l731_731547


namespace tan_product_l731_731526

theorem tan_product :
  (∏ x in (finset.range 89).filter (λ x, x > 0 && x < 90 + 1), (1 + real.tan (x : ℝ))) = 2^45 := 
sorry

end tan_product_l731_731526


namespace integer_inequality_l731_731692

theorem integer_inequality (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := 
  sorry

end integer_inequality_l731_731692


namespace closest_integer_to_cuberoot_of_200_l731_731022

theorem closest_integer_to_cuberoot_of_200 :
  ∃ n : ℤ, (n = 6) ∧ ( |200 - n^3| ≤ |200 - m^3|  ) ∀ m : ℤ := sorry

end closest_integer_to_cuberoot_of_200_l731_731022


namespace probability_not_within_B_l731_731386

-- Definition representing the problem context
structure Squares where
  areaA : ℝ
  areaA_pos : areaA = 65
  perimeterB : ℝ
  perimeterB_pos : perimeterB = 16

-- The theorem to be proved
theorem probability_not_within_B (s : Squares) : 
  let sideA := Real.sqrt s.areaA
  let sideB := s.perimeterB / 4
  let areaB := sideB^2
  let area_not_covered := s.areaA - areaB
  let probability := area_not_covered / s.areaA
  probability = 49 / 65 := 
by
  sorry

end probability_not_within_B_l731_731386


namespace derivative_of_odd_function_is_even_l731_731322

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the main theorem
theorem derivative_of_odd_function_is_even (f g : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, g x = deriv f x) :
  ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_function_is_even_l731_731322


namespace spoons_in_set_l731_731941

def number_of_spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) : ℕ :=
  let c := cost_five_spoons / 5
  let s := total_cost_set / c
  s

theorem spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) (h1 : total_cost_set = 21) (h2 : cost_five_spoons = 15) : 
  number_of_spoons_in_set total_cost_set cost_five_spoons = 7 :=
by
  sorry

end spoons_in_set_l731_731941


namespace probability_sum_less_than_product_l731_731829

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731829


namespace probability_sum_less_than_product_l731_731830

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731830


namespace alok_paid_rs_811_l731_731100

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l731_731100


namespace square_logarithm_a_l731_731385

theorem square_logarithm_a (A B C D : ℝ) (a : ℝ) (x y : ℝ) :
  let side_length := 8 in
  let square_area := side_length * side_length in
  square_area = 64 ∧
  A = (x, y) ∧ B = (x + side_length, y) ∧ C = (x + 2 * side_length, y) ∧
  y = log a x ∧ y = log a (x - side_length) ∧ y = log a (x + side_length) →
  a = 2 :=
by
  -- This part is left as an exercise for the user.
  sorry

end square_logarithm_a_l731_731385


namespace cosine_angle_l731_731608

-- Assume vectors in some inner product space
variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b c : V)
variables (k : ℤ) (h_k : k > 0)
variables (h1 : ∥a∥ = 1)
variables (h2 : ∥b∥ = k)
variables (h3 : ∥c∥ = 3)
variables (h4 : b - a = 2 * (c - b))

-- The problem to prove
theorem cosine_angle (α : ℝ) 
  (hα : real.angle a c = α) :
  real.cos α = -1 / 12 :=
sorry

end cosine_angle_l731_731608


namespace polygonal_chain_covered_by_circle_l731_731195

theorem polygonal_chain_covered_by_circle (chain : Set Point) 
  (perimeter : ℝ) (h_perimeter : perimeter = 1)
  (closed : ∀ (x y : Point), x ∈ chain ∧ y ∈ chain → dist x y ≤ 1) :
  ∃ (O : Point) (r : ℝ), r = 1/4 ∧ (∀ P ∈ chain, dist O P ≤ r) :=
begin
  sorry
end

end polygonal_chain_covered_by_circle_l731_731195


namespace total_pupils_correct_l731_731653

-- Definitions of the number of girls and boys in each school
def girlsA := 542
def boysA := 387
def girlsB := 713
def boysB := 489
def girlsC := 628
def boysC := 361

-- Total pupils in each school
def pupilsA := girlsA + boysA
def pupilsB := girlsB + boysB
def pupilsC := girlsC + boysC

-- Total pupils across all schools
def total_pupils := pupilsA + pupilsB + pupilsC

-- The proof statement (no proof provided, hence sorry)
theorem total_pupils_correct : total_pupils = 3120 := by sorry

end total_pupils_correct_l731_731653


namespace sum_of_valid_n_l731_731867

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l731_731867


namespace divide_rectangle_into_trominoes_l731_731644

theorem divide_rectangle_into_trominoes :
  let rectangle := (3, 8)
  let tromino_area := 3
  number_of_ways rectangle = 16 :=
by
  sorry

end divide_rectangle_into_trominoes_l731_731644


namespace seating_arrangements_count_l731_731160

structure person (id : ℕ) (is_man : Bool) (is_single : Bool) (spouse_id : Option ℕ)

def valid_seating (positions : Array ℕ) (persons : Array (person)) : Prop :=
  ∀ i, 
    (i < positions.size) → 
    -- Alternate pattern of men and women
    (persons[positions[i]].is_man ≠ persons[positions[(i + 1) % 11]].is_man) ∧
    (persons[positions[i]].is_man ≠ persons[positions[(i + 10) % 11]].is_man) ∧
    -- No one sits next to or across from their spouse
    (persons[positions[i]].spouse_id ≠ some positions[(i + 1) % 11]) ∧
    (persons[positions[i]].spouse_id ≠ some positions[(i + 10) % 11])

def count_valid_arrangements : ℕ :=
  let persons := [ 
    person.mk 0 false false (some  1), person.mk 1 true  false (some  0), -- Couple 1
    person.mk 2 false false (some  3), person.mk 3 true  false (some  2), -- Couple 2
    person.mk 4 false false (some  5), person.mk 5 true  false (some  4), -- Couple 3
    person.mk 6 false false (some  7), person.mk 7 true  false (some  6), -- Couple 4
    person.mk 8 false false (some  9), person.mk 9 true  false (some  8), -- Couple 5
    person.mk 10 true true none -- Single individual
  ]
  in 
  -- Count the number of valid seating arrangements
  (count_if (λ (seating : Array ℕ), valid_seating seating persons) (permutations (Array.of_list (List.range 11))))

theorem seating_arrangements_count : count_valid_arrangements = 6 :=
  sorry

end seating_arrangements_count_l731_731160


namespace horner_first_calculation_l731_731426

def f (x : ℕ) : ℕ := 7 * x^6 + 6 * x^5 + 3 * x^2 + 2

theorem horner_first_calculation (x : ℕ) (h : x = 4) :
  7 * x = 28 :=
by
  rw [h]
  rfl

end horner_first_calculation_l731_731426


namespace abs_inequality_l731_731365

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l731_731365


namespace coupon1_best_discount_l731_731473

noncomputable def listed_prices : List ℝ := [159.95, 179.95, 199.95, 219.95, 239.95]

theorem coupon1_best_discount (x : ℝ) (h₁ : x ∈ listed_prices) (h₂ : x > 120) :
  0.15 * x > 25 ∧ 0.15 * x > 0.20 * (x - 120) ↔ 
  x = 179.95 ∨ x = 199.95 ∨ x = 219.95 ∨ x = 239.95 :=
sorry

end coupon1_best_discount_l731_731473


namespace probability_sum_less_than_product_is_5_div_9_l731_731805

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731805


namespace circle_center_l731_731071

theorem circle_center 
  (a b c d : ℝ)
  (hA : ∃ k : ℝ, (k, 0) = (a, 0))
  (hB : ∃ k : ℝ, (k, 0) = (b, 0))
  (hC : ∃ k : ℝ, (0, k) = (0, c))
  (hD : ∃ k : ℝ, (0, k) = (0, d)) :
  ∃ x y : ℝ, (x, y) = (a + b) / 2 ∧ y = (c + d) / 2 :=
by
  sorry

end circle_center_l731_731071


namespace melissa_solves_equation_l731_731319

theorem melissa_solves_equation : 
  ∃ b c : ℤ, (∀ x : ℝ, x^2 - 6 * x + 9 = 0 ↔ (x + b)^2 = c) ∧ b + c = -3 :=
by
  sorry

end melissa_solves_equation_l731_731319


namespace find_m_value_l731_731265

def quadratic_inequality_solution_set (a b c : ℝ) (m : ℝ) := {x : ℝ | 0 < x ∧ x < 2}

theorem find_m_value (a b c : ℝ) (m : ℝ) 
  (h1 : a = -1/2) 
  (h2 : b = 2) 
  (h3 : c = m) 
  (h4 : quadratic_inequality_solution_set a b c m = {x : ℝ | 0 < x ∧ x < 2}) : 
  m = 1 := 
sorry

end find_m_value_l731_731265


namespace principal_amount_correct_l731_731053

noncomputable def principal_amount (r : ℝ) : ℝ := 8800 / (1 + r)^2

theorem principal_amount_correct (r : ℝ) (P : ℝ) 
  (h1 : 8800 = P * (1 + r)^2) 
  (h2 : 9261 = P * (1 + r)^3) 
  (hr : r = 461 / 8800) : 
  P ≈ 7945.67 :=
by
  have h : (1 + r)^3 / (1 + r)^2 = 1 + r := by sorry
  have hr_correct : 1 + r = 9261 / 8800 := by
    calc
    1 + r = (1 + r)^3 / (1 + r)^2 : by sorry
    ... = 9261 / 8800 : by sorry
  have r_val : r = 461 / 8800 := by sorry
  have P_val : P = 8800 / (1 + r)^2 := by
    calc
    P = 8800 / (1 + r)^2 : by sorry
  show P ≈ 7945.67 := by sorry

end principal_amount_correct_l731_731053


namespace scientific_notation_l731_731966

theorem scientific_notation :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_l731_731966


namespace average_marks_l731_731745

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l731_731745


namespace probability_of_mixing_purple_l731_731337

/-- A painter Rita rolls a fair 6-sided die with 3 red, 2 yellow, and 1 blue side. If she rolls the die twice, 
the probability that she mixes the color purple (red and blue) is 1/6. -/
theorem probability_of_mixing_purple : 
  let P_R : ℚ := 3 / 6,
      P_B : ℚ := 1 / 6 in
  let P_red_first_blue_second : ℚ := P_R * P_B in
  let P_blue_first_red_second : ℚ := P_B * P_R in
  P_red_first_blue_second + P_blue_first_red_second = 1 / 6 := 
begin
  sorry
end

end probability_of_mixing_purple_l731_731337


namespace rewinding_time_l731_731477

theorem rewinding_time (a L S ω T : ℝ) (h_pos_ω : ω > 0) (h_pos_S : S > 0) :
  (T = (π / (S * ω)) * (sqrt (a^2 + (4 * S * L) / π) - a)) :=
begin
  sorry
end

end rewinding_time_l731_731477


namespace common_ratio_l731_731649

variable {G : Type} [LinearOrderedField G]

-- Definitions based on conditions
def geometric_seq (a₁ q : G) (n : ℕ) : G := a₁ * q^(n-1)
def sum_geometric_seq (a₁ q : G) (n : ℕ) : G :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from conditions
variable {a₁ q : G}
variable (h1 : sum_geometric_seq a₁ q 3 = 7)
variable (h2 : sum_geometric_seq a₁ q 6 = 63)

theorem common_ratio (a₁ q : G) (h1 : sum_geometric_seq a₁ q 3 = 7)
  (h2 : sum_geometric_seq a₁ q 6 = 63) : q = 2 :=
by
  -- Proof to be completed
  sorry

end common_ratio_l731_731649


namespace smallest_of_2_and_4_smallest_combined_of_2_and_4_l731_731011

def smallest_number_of_two (a b : ℕ) : ℕ :=
  if a <= b then a else b

theorem smallest_of_2_and_4 : smallest_number_of_two 2 4 = 2 :=
  by simp [smallest_number_of_two]

def form_combined_number (a b : ℕ) : ℕ :=
  10 * a + b

theorem smallest_combined_of_2_and_4 : form_combined_number 2 4 = 24 :=
  by simp [form_combined_number]

end smallest_of_2_and_4_smallest_combined_of_2_and_4_l731_731011


namespace intersect_on_AB_l731_731497

open EuclideanGeometry

variables {A B C P P1 P2 Q1 Q2 : Point} -- Define points
variables (h_triangle : Triangle A B C) -- Define that they form a triangle
variables (h_interior : InteriorPoint P A B C) -- P is an interior point
variables (h_perp_P_AC : Perpendicular P P1 A C) -- P1 is foot of perpendicular from P to AC
variables (h_perp_P_BC : Perpendicular P P2 B C) -- P2 is foot of perpendicular from P to BC
variables (h_perp_C_AP : Perpendicular C Q1 A P) -- Q1 is foot of perpendicular from C to AP
variables (h_perp_C_BP : Perpendicular C Q2 B P) -- Q2 is foot of perpendicular from C to BP

theorem intersect_on_AB : 
  Collinear A B (LineIntersection (line_through P1 Q2) (line_through P2 Q1)) :=
by
  sorry

end intersect_on_AB_l731_731497


namespace machine_production_time_l731_731017

theorem machine_production_time (x : ℝ) 
  (h1 : 60 / x + 2 = 12) : 
  x = 6 :=
sorry

end machine_production_time_l731_731017


namespace average_marks_l731_731753

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l731_731753


namespace log_eight_of_five_twelve_l731_731987

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731987


namespace closest_integer_to_cube_root_of_200_l731_731028

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), n = 6 ∧ (n^3 = 216 ∨ n^3 > 125 ∧ n^3 < 216) := 
by
  existsi 6
  split
  · refl
  · right
    split
    · norm_num
    · norm_num

end closest_integer_to_cube_root_of_200_l731_731028


namespace closest_integer_to_cube_root_of_200_l731_731039

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), 
    (n = 6) ∧ (n^3 < 200) ∧ (200 < (n + 1)^3) ∧ 
    (∀ m : ℤ, (m^3 < 200) → (200 < (m + 1)^3) → (Int.abs (n - Int.ofNat (200 ^ (1/3 : ℝ)).round) < Int.abs (m - Int.ofNat (200 ^ (1/3 : ℝ)).round))) :=
begin
  sorry
end

end closest_integer_to_cube_root_of_200_l731_731039


namespace sum_gcd_lcm_l731_731012

theorem sum_gcd_lcm (a b : ℕ) (ha : a = 45) (hb : b = 4095) :
    Nat.gcd a b + Nat.lcm a b = 4140 :=
by
  sorry

end sum_gcd_lcm_l731_731012


namespace probability_sum_less_than_product_l731_731827

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731827


namespace sasha_skated_distance_l731_731452

theorem sasha_skated_distance (d total_distance v : ℝ)
  (h1 : total_distance = 3300)
  (h2 : v > 0)
  (h3 : d = 3 * v * (total_distance / (3 * v + 2 * v))) :
  d = 1100 :=
by
  sorry

end sasha_skated_distance_l731_731452


namespace range_of_a_l731_731238

def piecewise_function (x a : ℝ) : ℝ :=
  if x > a then -x + 10 else x^2 + 2 * x

theorem range_of_a (a : ℝ) :
  (∀ b : ℝ, ∃ x0 : ℝ, piecewise_function x0 a = b) ↔ -5 ≤ a ∧ a ≤ 11 :=
by
  sorry

end range_of_a_l731_731238


namespace tan_alpha_equiv_l731_731552

theorem tan_alpha_equiv (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end tan_alpha_equiv_l731_731552


namespace problem_solution_l731_731614

-- Define the polynomial expansion coefficients
variables {a : ℕ → ℝ}

-- Assume the initial polynomial equation holds
def polynomial_expansion (x : ℝ) : ℝ :=
  (1 - 2 * x) ^ 2017

def coefficients (x : ℝ) : ℝ :=
  a 0 + ∑ i in finset.range 2018, a (i + 1) * x ^ (i + 1)

lemma given_polynomial:
  ∀ x : ℝ, polynomial_expansion x = coefficients x := sorry

-- The problem is to prove the value of the series
theorem problem_solution :
  ∑ i in finset.range 2017, a (i + 1) * (1 / 2) ^ (i + 1) = -1 := sorry

end problem_solution_l731_731614


namespace incorrect_calculation_l731_731720

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l731_731720


namespace probability_ab_gt_a_add_b_l731_731799

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731799


namespace john_travelled_distance_l731_731664

theorem john_travelled_distance :
  ∃ (t₁ t₂ t₃ t₄ t₅ : ℕ) (x : ℕ),
  -- Condition 1: Travel hours each day
  t₁ = 1 ∧ t₂ = 2 ∧ t₃ = 3 ∧ t₄ = 4 ∧ t₅ = 5 ∧
  -- Condition 2: Time to travel one mile on the first day and subsequent halved until a minimum reached
  let time_seq := [x, x / 2, x / 4, x / 4, x / 4] in
  -- Condition 3: Total time to travel one mile for all days is 30 minutes
  x + x / 2 + 3 * (x / 4) = 30 ∧
  -- Condition 4: Each day's distance is an integer number of miles
  (∀ day in time_seq, 60 % day = 0) ∧
  -- Condition 5: Calculate the total distance
  let distances := [t₁ * (60 / time_seq.nth 0).getD 0, 
                    t₂ * (60 / time_seq.nth 1).getD 0,
                    t₃ * (60 / time_seq.nth 2).getD 0,
                    t₄ * (60 / time_seq.nth 3).getD 0,
                    t₅ * (60 / time_seq.nth 4).getD 0] in
  distances.sum = 265 :=
begin
  sorry,
end

end john_travelled_distance_l731_731664


namespace solve_q_l731_731346

theorem solve_q (n m q : ℤ) 
  (h₁ : 5/6 = n/72) 
  (h₂ : 5/6 = (m + n)/90) 
  (h₃ : 5/6 = (q - m)/150) : 
  q = 140 := by
  sorry

end solve_q_l731_731346


namespace smallest_angle_in_scalene_triangle_l731_731488

theorem smallest_angle_in_scalene_triangle :
  ∃ (triangle : Type) (a b c : ℝ),
    ∀ (A B C : triangle),
      a = 162 ∧
      b / c = 3 / 4 ∧
      a + b + c = 180 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c ->
        min b c = 7.7 :=
sorry

end smallest_angle_in_scalene_triangle_l731_731488


namespace pages_needed_l731_731456

def new_cards : ℕ := 2
def old_cards : ℕ := 10
def cards_per_page : ℕ := 3
def total_cards : ℕ := new_cards + old_cards

theorem pages_needed : total_cards / cards_per_page = 4 := by
  sorry

end pages_needed_l731_731456


namespace arrive_together_possible_l731_731517

-- Definitions for the constants and conditions
def constant_speeds (v_A v_B v_V : ℝ) (h1 : v_A < v_B) (h2 : v_V < v_A) : Prop :=
  ∀ (d1 d2 : ℝ),
    -- Day 1: Anya leaves first, Borya leaves second, Vasya leaves third
    ((∃ t : ℝ, (d1 - t * v_A = d2 - t * v_B) ∧ (d2 - t * v_V)) ∨
    -- Day 2: Vasya leaves first, Borya leaves second, Anya leaves third
    (∃ t : ℝ, (d1 - t * v_V = d2 - t * v_B) ∧ (d2 - t * v_A))) →
    -- Conclusion: Can they all arrive together?
    ((∃ t : ℝ, d1 - t * v_A = d2 - t * v_B) ∧ (d2 - t * v_V))

-- Main theorem stating they can all arrive together
theorem arrive_together_possible : ∃ (v_A v_B v_V : ℝ) (h1 : v_A < v_B) (h2 : v_V < v_A), 
  constant_speeds v_A v_B v_V h1 h2 :=
begin
  sorry -- Placeholder for the proof.
end

end arrive_together_possible_l731_731517


namespace complement_of_A_in_U_l731_731601

noncomputable def U : Set ℤ := {x : ℤ | x^2 ≤ 2*x + 3}
def A : Set ℤ := {0, 1, 2}

theorem complement_of_A_in_U : (U \ A) = {-1, 3} :=
by
  sorry

end complement_of_A_in_U_l731_731601


namespace eval_expr1_eval_expr2_l731_731163

theorem eval_expr1 : (0.25^(-1 / 2) - 2 * (7 / 3)^0 * (-2)^((3) / (3)) + 10*(2- Real.sqrt 3)^(-1) - Real.sqrt 300) = 30 :=
by
  sorry

theorem eval_expr2 : (2 * (Real.log 3) - Real.log (32 / 9) + Real.log 8 - (5^2)^((Real.log 5) / (3))) = -7 :=
by
  sorry

end eval_expr1_eval_expr2_l731_731163


namespace sum_of_all_n_l731_731890

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l731_731890


namespace find_power_of_7_l731_731175

theorem find_power_of_7 (x : ℕ) :
  ∀ (total_prime_factors : ℕ),
    total_prime_factors = 29 →
    x = total_prime_factors - (22 + 2) →
    x = 5 :=
by
  intros total_prime_factors total_pf_eq x_eq
  have pf_4_11 := 22  -- Number of prime factors from (4)^{11} = (2^2)^{11} = 2^{22}
  have pf_11_2 := 2  -- Number of prime factors from (11)^2 = 11^2
  have pf_total := pf_4_11 + pf_11_2 -- Total prime factors of (4)^{11} and (11)^2
  rw total_pf_eq at x_eq
  calc
    29 - pf_total = 29 - 24 := by sorry
    29 - 24 = 5 := by sorry

end find_power_of_7_l731_731175


namespace discount_percentage_l731_731107

theorem discount_percentage (cost_price marked_price : ℝ) (profit_percentage : ℝ) 
  (h_cost_price : cost_price = 47.50)
  (h_marked_price : marked_price = 65)
  (h_profit_percentage : profit_percentage = 0.30) :
  ((marked_price - (cost_price + (profit_percentage * cost_price))) / marked_price) * 100 = 5 :=
by
  sorry

end discount_percentage_l731_731107


namespace cos2theta_sin2theta_l731_731261

theorem cos2theta_sin2theta (θ : ℝ) (h : 2 * Real.cos θ + Real.sin θ = 0) :
  Real.cos (2 * θ) + (1 / 2) * Real.sin (2 * θ) = -1 :=
sorry

end cos2theta_sin2theta_l731_731261


namespace problem_solution_l731_731009

theorem problem_solution : 12 * ((1/3) + (1/4) + (1/6) + (1/12))⁻¹ = 72 / 5 := 
by
  sorry

end problem_solution_l731_731009


namespace ice_cubes_per_chest_l731_731502

theorem ice_cubes_per_chest (total_ice_cubes : ℕ) (num_chests : ℕ) (h1 : total_ice_cubes = 294) (h2 : num_chests = 7) : total_ice_cubes / num_chests = 42 :=
by
  rw [h1, h2]
  norm_num
  sorry

end ice_cubes_per_chest_l731_731502


namespace product_xyz_l731_731266

theorem product_xyz (x y z : ℝ) (h1 : x = y) (h2 : x = 2 * z) (h3 : x = 7.999999999999999) :
    x * y * z = 255.9999999999998 := by
  sorry

end product_xyz_l731_731266


namespace solve_inequality_l731_731355

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l731_731355


namespace possible_box_dimensions_l731_731925

-- Define the initial conditions
def edge_length_original_box := 4
def edge_length_dice := 1
def total_cubes := (edge_length_original_box * edge_length_original_box * edge_length_original_box)

-- Prove that these are the possible dimensions of boxes with square bases that fit all the dice
theorem possible_box_dimensions :
  ∃ (len1 len2 len3 : ℕ), 
  total_cubes = (len1 * len2 * len3) ∧ 
  (len1 = len2) ∧ 
  ((len1, len2, len3) = (1, 1, 64) ∨ (len1, len2, len3) = (2, 2, 16) ∨ (len1, len2, len3) = (4, 4, 4) ∨ (len1, len2, len3) = (8, 8, 1)) :=
by {
  sorry -- The proof would be placed here
}

end possible_box_dimensions_l731_731925


namespace number_of_monkeys_that_ate_bird_l731_731964

-- Definitions based on the conditions
def initial_monkeys : ℕ := 6
def initial_birds : ℕ := 6
def final_monkey_percentage : ℝ := 0.60

-- Let M denote the number of monkeys that ate a bird.
def monkeys_ate_bird (M : ℕ) : Proposition :=
  let remaining_birds := initial_birds - M
  let total_animals := initial_monkeys + remaining_birds
  let monkey_fraction := initial_monkeys / total_animals.to_float
  monkey_fraction = final_monkey_percentage

-- We need to prove that M = 2
theorem number_of_monkeys_that_ate_bird : ferm_all_actor.equal hope_monkey_bird == 2카오s.coronavirus_use_bash_synomoks_boy_funny_traits_retional_solution.finocres :
  ∃ (M : ℕ), monkeys_ate_bird M ∧ M = 2 := sorry

end number_of_monkeys_that_ate_bird_l731_731964


namespace closest_integer_to_cuberoot_of_200_l731_731020

theorem closest_integer_to_cuberoot_of_200 :
  ∃ n : ℤ, (n = 6) ∧ ( |200 - n^3| ≤ |200 - m^3|  ) ∀ m : ℤ := sorry

end closest_integer_to_cuberoot_of_200_l731_731020


namespace problem_part1_problem_part2_problem_part3_l731_731574

noncomputable def f (x : ℝ) : ℝ := -x + (2 / x)

def passes_through (p1 p2 : ℝ × ℝ) : Prop :=
  f p1.1 = p1.2 ∧ f p2.1 = p2.2

theorem problem_part1 :
  (∃ a b : ℝ, passes_through (1, 1) (2, -1)) → f = λ x, -x + (2 / x) := 
sorry

theorem problem_part2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  x1 < x2 → f x1 > f x2 := 
sorry

theorem problem_part3 : 
  set.range (f ∘ (λ x, [0.25, 1])) = set.range (λ y, [1, 31 / 4]) := 
sorry

end problem_part1_problem_part2_problem_part3_l731_731574


namespace max_value_of_a_l731_731571

theorem max_value_of_a : ∀ a : ℝ, (∀ x : ℝ, x ∈ Icc (1/2) 2 → x * log x - (1 + a) * x + 1 ≥ 0) → a ≤ 0 :=
by
  sorry

end max_value_of_a_l731_731571


namespace kittens_total_number_l731_731000

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end kittens_total_number_l731_731000


namespace nancy_shoes_l731_731682

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l731_731682


namespace log_base_8_of_512_l731_731997

theorem log_base_8_of_512 : log 8 512 = 3 := by
  have h₁ : 8 = 2^3 := by rfl
  have h₂ : 512 = 2^9 := by rfl
  rw [h₂, h₁]
  sorry

end log_base_8_of_512_l731_731997


namespace paul_diner_total_cost_l731_731111

/-- At Paul's Diner, sandwiches cost $5 each and sodas cost $3 each. If a customer buys
more than 4 sandwiches, they receive a $10 discount on the total bill. Calculate the total
cost if a customer purchases 6 sandwiches and 3 sodas. -/
def totalCost (num_sandwiches num_sodas : ℕ) : ℕ :=
  let sandwich_cost := 5
  let soda_cost := 3
  let discount := if num_sandwiches > 4 then 10 else 0
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) - discount

theorem paul_diner_total_cost : totalCost 6 3 = 29 :=
by
  sorry

end paul_diner_total_cost_l731_731111


namespace complex_conjugate_in_first_quadrant_l731_731538

-- Definitions and conditions
def z : ℂ := 2 / (1 + complex.I * real.sqrt 3)

-- Statement of the proof problem
theorem complex_conjugate_in_first_quadrant :
  let z_conj := conj z in
  z_conj.re > 0 ∧ z_conj.im > 0 :=
by
  sorry

end complex_conjugate_in_first_quadrant_l731_731538


namespace circumcircles_intersection_or_tangent_l731_731669

open Point

variables {Γ : Circle} {A B X Y X' Y' : Point} {l : Line}

-- Hypotheses
variable (h1 : Diameter Γ A B)
variable (h2 : LineOutsidePerpendicular l Γ A B)
variable (h3 : OnLine X l)
variable (h4 : OnLine Y l)
variable (h5 : OnLine X' l)
variable (h6 : OnLine Y' l)
variable (h7 : IntersectsOn Γ (LineThrough A X) (LineThrough B X'))
variable (h8 : IntersectsOn Γ (LineThrough A Y) (LineThrough B Y'))

-- Goal
theorem circumcircles_intersection_or_tangent 
  (h1 : Diameter Γ A B)
  (h2 : LineOutsidePerpendicular l Γ A B)
  (h3 : OnLine X l)
  (h4 : OnLine Y l)
  (h5 : OnLine X' l)
  (h6 : OnLine Y' l)
  (h7 : IntersectsOn Γ (LineThrough A X) (LineThrough B X')) 
  (h8 : IntersectsOn Γ (LineThrough A Y) (LineThrough B Y')) :
  (∃ P, OnCircle P (Circumcircle (Triangle A X Y)) ∧ OnCircle P (Circumcircle (Triangle A X' Y')) ∧ OnCircle P Γ ∧ P ≠ A)
  ∨ Tangent (Circumcircle (Triangle A X Y)) (Circumcircle (Triangle A X' Y')) A :=
sorry

end circumcircles_intersection_or_tangent_l731_731669


namespace find_zero_of_function_l731_731221

def log_base_14 (n : ℝ) : ℝ := Real.log n / Real.log 14

def point := (log_base_14 7, log_base_14 56)

def f (x k : ℝ) : ℝ := k * x + 3

theorem find_zero_of_function : ∃ x : ℝ, f x (-2) = 0 := by
  sorry

end find_zero_of_function_l731_731221


namespace total_amount_is_correct_l731_731484

-- Definitions based on the conditions
def share_a (x : ℕ) : ℕ := 2 * x
def share_b (x : ℕ) : ℕ := 4 * x
def share_c (x : ℕ) : ℕ := 5 * x
def share_d (x : ℕ) : ℕ := 4 * x

-- Condition: combined share of a and b is 1800
def combined_share_of_ab (x : ℕ) : Prop := share_a x + share_b x = 1800

-- Theorem we want to prove: Total amount given to all children is $4500
theorem total_amount_is_correct (x : ℕ) (h : combined_share_of_ab x) : 
  share_a x + share_b x + share_c x + share_d x = 4500 := sorry

end total_amount_is_correct_l731_731484


namespace solve_inequality_l731_731379

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731379


namespace store_b_discount_l731_731422

theorem store_b_discount 
  (full_price_A : ℝ := 125) 
  (discount_A_percent : ℝ := 0.08)
  (full_price_B : ℝ := 130)
  (final_price_diff : ℝ := 2) :
  ∃ (x : ℝ), 
    let final_price_A := full_price_A * (1 - discount_A_percent),
        final_price_B := full_price_B * (1 - x / 100) 
    in final_price_A = final_price_B + final_price_diff :=
begin
  use 10,
  let final_price_A := full_price_A * (1 - discount_A_percent),
  let final_price_B := full_price_B * (1 - 10 / 100),
  sorry,
end

end store_b_discount_l731_731422


namespace lattice_intersections_l731_731965

theorem lattice_intersections (squares : ℕ) (circles : ℕ) 
        (line_segment : ℤ × ℤ → ℤ × ℤ) 
        (radius : ℚ) (side_length : ℚ) : 
        line_segment (0, 0) = (1009, 437) → 
        radius = 1/8 → side_length = 1/4 → 
        (squares + circles = 430) :=
by
  sorry

end lattice_intersections_l731_731965


namespace problem_l731_731311

theorem problem (O O1 O2 : Circle) (A B C D P Q E F : Point) 
  [Inscribed_ABCD : Inscribed_Convex_Quadrilateral O A B C D]
  [Inter_AC_BD_P : Intersect AC BD P]
  [Circle_O1_through_PB : Circle_through O1 P B]
  [Circle_O2_through_PA : Circle_through O2 P A]
  [Inter_O1_O2_PQ : Intersect O1 O2 P Q]
  [Inter_O1_O_E_other_than_B : Intersect_other O1 O E B]
  [Inter_O2_O_F_other_than_A : Intersect_other O2 O F A] :
  Concurrent_or_Parallel PQ CE DF :=
sorry

end problem_l731_731311


namespace gemstones_needed_l731_731696

-- Define the initial quantities and relationships
def magnets_per_earring := 2
def buttons_per_magnet := 1 / 2
def gemstones_per_button := 3
def earrings_per_set := 2
def sets_of_earrings := 4

-- Define the total gemstones needed
theorem gemstones_needed : 
    let earrings := sets_of_earrings * earrings_per_set in
    let total_magnets := earrings * magnets_per_earring in
    let total_buttons := total_magnets * buttons_per_magnet in
    let total_gemstones := total_buttons * gemstones_per_button in
    total_gemstones = 24 :=
by
    have earrings := 2 * 4
    have total_magnets := earrings * 2
    have total_buttons := total_magnets / 2
    have total_gemstones := total_buttons * 3
    exact eq.refl 24

end gemstones_needed_l731_731696


namespace triangle_fraction_of_grid_covered_l731_731090

theorem triangle_fraction_of_grid_covered :
  let A := (2, 4)
  let B := (6, 2)
  let C := (5, 5)
  let grid_area := 8 * 6
  let triangle_area := 1 / 2 * |2 * (2 - 5) + 6 * (5 - 4) + 5 * (4 - 2)|
  triangle_area / grid_area = 5 / 48 := by
sorry

end triangle_fraction_of_grid_covered_l731_731090


namespace closest_integer_to_cube_root_of_200_l731_731041

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), 
    (n = 6) ∧ (n^3 < 200) ∧ (200 < (n + 1)^3) ∧ 
    (∀ m : ℤ, (m^3 < 200) → (200 < (m + 1)^3) → (Int.abs (n - Int.ofNat (200 ^ (1/3 : ℝ)).round) < Int.abs (m - Int.ofNat (200 ^ (1/3 : ℝ)).round))) :=
begin
  sorry
end

end closest_integer_to_cube_root_of_200_l731_731041


namespace symmetry_of_angles_l731_731190

-- Define the condition alpha = π / 6
def alpha : ℝ := (real.pi / 6)

-- Define the predicate for the symmetry condition
def symmetric_with_respect_to_y_eq_x (beta alpha : ℝ) : Prop :=
  ∃ k : ℤ, beta = 2 * k * real.pi + (real.pi / 3)

-- The proof statement to be proven
theorem symmetry_of_angles (alpha_eq : alpha = real.pi / 6) :
  ∀ β : ℝ, symmetric_with_respect_to_y_eq_x β alpha ↔ ∃ k : ℤ, β = 2 * k * real.pi + (real.pi / 3) :=
by
  sorry

end symmetry_of_angles_l731_731190


namespace Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l731_731679

theorem Mishas_fathers_speed (d : ℝ) (t : ℝ) (V : ℝ) 
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) :
  V = 1 :=
by
  sorry

theorem Mishas_fathers_speed_in_kmh (d : ℝ) (t : ℝ) (V : ℝ) (V_kmh : ℝ)
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) 
  (h4 : V_kmh = V * 60):
  V_kmh = 60 :=
by
  sorry

end Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l731_731679


namespace coins_after_five_hours_l731_731780

-- Definitions of the conditions
def first_hour : ℕ := 20
def next_two_hours : ℕ := 2 * 30
def fourth_hour : ℕ := 40
def fifth_hour : ℕ := -20

-- The total number of coins calculation
def total_coins : ℕ := first_hour + next_two_hours + fourth_hour + fifth_hour

-- The theorem to be proved
theorem coins_after_five_hours : total_coins = 100 :=
by
  sorry

end coins_after_five_hours_l731_731780


namespace y_intercept_of_line_l731_731491

theorem y_intercept_of_line (m x y : ℝ) (b : ℝ) (h_slope : m = 3.8666666666666667) (h_point : x = 150 ∧ y = 600) : 
    y = m * x + b → 
    b ≈ 20.0000000000001 :=
by
  sorry

end y_intercept_of_line_l731_731491


namespace average_marks_for_class_l731_731749

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l731_731749


namespace alok_total_payment_l731_731099

def cost_of_chapatis : Nat := 16 * 6
def cost_of_rice_plates : Nat := 5 * 45
def cost_of_mixed_vegetable_plates : Nat := 7 * 70
def total_cost : Nat := cost_of_chapatis + cost_of_rice_plates + cost_of_mixed_vegetable_plates

theorem alok_total_payment :
  total_cost = 811 := by
  unfold total_cost
  unfold cost_of_chapatis
  unfold cost_of_rice_plates
  unfold cost_of_mixed_vegetable_plates
  calc
    16 * 6 + 5 * 45 + 7 * 70 = 96 + 5 * 45 + 7 * 70 := by rfl
                      ... = 96 + 225 + 7 * 70 := by rfl
                      ... = 96 + 225 + 490 := by rfl
                      ... = 96 + (225 + 490) := by rw Nat.add_assoc
                      ... = (96 + 225) + 490 := by rw Nat.add_assoc
                      ... = 321 + 490 := by rfl
                      ... = 811 := by rfl

end alok_total_payment_l731_731099


namespace remaining_pieces_leq_n_plus_one_l731_731054

theorem remaining_pieces_leq_n_plus_one
  (n : ℕ)
  (rectangles_non_overlapping : ∀ i j : Fin n, i ≠ j → 
    ∀ (x y : ℤ) (w h : ℤ), 
      ¬(x, y, w, h) ∈ rectangles[i]
      → ¬(x, y, w, h) ∈ rectangles[j]
  ) :
  (remaining_pieces(rectangles, sheet) <= n + 1) := 
  sorry

end remaining_pieces_leq_n_plus_one_l731_731054


namespace assignment_statements_count_l731_731104

theorem assignment_statements_count :
  (is_assignment_statement (m = x^3 - x^2) ∧
   is_assignment_statement (T = T * I) ∧
   ¬ is_assignment_statement (32 = A) ∧
   is_assignment_statement (A = A + 2) ∧
   ¬ is_assignment_statement (A = 2 * (B + 1) = 2B + 2) ∧
   is_assignment_statement (P = [(7x + 3) - 5]x + 1)) →
  count_assignment_statements [m = x^3 - x^2, T = T * I, 32 = A, A = A + 2, A = 2 * (B + 1) = 2B + 2, P = [(7x + 3) - 5]x + 1] = 4 := 
by
  intro h
  sorry

end assignment_statements_count_l731_731104


namespace min_value_f_at_a_eq_1_range_of_a_for_one_zero_l731_731586

-- Part (1) Minimum value of f(x) when a = 1
theorem min_value_f_at_a_eq_1 : 
  let f (x : ℝ) := Math.log x + 1/x + 1 in 
  ∃ x : ℝ, x > 0 ∧ f(1) = 2 := 
begin
  let f (x : ℝ) := Math.log x + 1/x + 1,
  use 1,
  split,
  { exact one_pos },
  { simp [f, Math.log_one] }
end

-- Part (2) Range of a for which f(x) has exactly one zero in (1/e^3, +∞)
theorem range_of_a_for_one_zero :
  let f (x : ℝ) (a : ℝ) := Math.log x + a/x + 1 in
  let interval := Ioo (1/Real.exp 3) (Real.infinity) in
  ∃ a : ℝ, ∀ x ∈ interval, f(x, a) = 0 ↔ a ∈ Set.Iic (2/Real.exp 3) ∪ {1/Real.exp 2} :=
begin
  let f (x : ℝ) (a : ℝ) := Math.log x + a/x + 1,
  let interval := Ioo (1/Real.exp 3) (Real.infinity),
  sorry
end

end min_value_f_at_a_eq_1_range_of_a_for_one_zero_l731_731586


namespace betting_game_final_amount_l731_731933

theorem betting_game_final_amount :
  ∀ (initial_amount : ℝ) (bets : ℕ) (wins : ℕ) (losses : ℕ),
  initial_amount = 100 ∧ bets = 4 ∧ wins = 2 ∧ losses = 2 →
  (∀ (first_bet_win : bool), first_bet_win = true →
   let amount_after_first_win := initial_amount * 3 / 2,
       amount_after_second_bet := amount_after_first_win - (amount_after_first_win * 1 / 4),
       amount_after_third_bet := amount_after_second_bet - (amount_after_second_bet * 1 / 2),
       final_amount := amount_after_third_bet - (amount_after_third_bet * 1 / 2)
   in final_amount = 28.125) :=
sorry

end betting_game_final_amount_l731_731933


namespace find_n_l731_731172

theorem find_n :
  (∃ n : ℕ, arctan (1 / 3 : ℝ) + arctan (1 / 4) + arctan (1 / 6) + arctan (1 / n) = π / 4) →
  ∃ (n : ℕ), n = 113 :=
by
  sorry

end find_n_l731_731172


namespace value_range_f_l731_731407

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem value_range_f : 
  set.range (λ x, f x) ⊆ set.Icc (-1 : ℝ) (3 : ℝ) := 
by 
  sorry

end value_range_f_l731_731407


namespace grid_product_sum_nonzero_l731_731275

theorem grid_product_sum_nonzero :
  ∀ (grid : Fin 25 → Fin 25 → ℤ),
    (∀ i j, grid i j = 1 ∨ grid i j = -1) →
    let row_product (i : Fin 25) := ∏ j, grid i j in
    let col_product (j : Fin 25) := ∏ i, grid i j in
    (∑ i, row_product i) + (∑ j, col_product j) ≠ 0 :=
begin
  intros grid h_grid,
  let row_product := λ i, ∏ j, grid i j,
  let col_product := λ j, ∏ i, grid i j,
  sorry
end

end grid_product_sum_nonzero_l731_731275


namespace perfect_squares_between_50_and_1000_l731_731255

theorem perfect_squares_between_50_and_1000 :
  ∃ (count : ℕ), count = 24 ∧ ∀ (n : ℕ), 50 < n * n ∧ n * n < 1000 ↔ 8 ≤ n ∧ n ≤ 31 :=
by {
  -- proof goes here
  sorry
}

end perfect_squares_between_50_and_1000_l731_731255


namespace find_C_find_a_b_l731_731602

open Real

noncomputable def problem1 (C : ℝ) : Prop :=
  sqrt 3 * sin C * cos C - cos C ^ 2 = 1 / 2

-- to use degrees in Lean, we often have to convert to radians.
noncomputable def degrees_to_radians (deg : ℝ) := (deg * π) / 180

theorem find_C (C : ℝ) (h : problem1 C) : C = degrees_to_radians 60 :=
  sorry

noncomputable def problem2 (A B : ℝ) : Prop :=
  degree A * π / 180 + degree B * π / 180 = (120 * π) / 180 ∧ (sin (degree B * π / 180) - 2 * sin (degree A * π / 180) = 0)

theorem find_a_b (A B a b : ℝ) (c : ℝ) (hc : c = 3)
  (hC : find_C (degrees_to_radians 60 * π / 180)) (hAB : problem2 A B)  :
  a = sqrt 3 ∧ b = 2 * sqrt 3 :=
  sorry

end find_C_find_a_b_l731_731602


namespace power_inequality_l731_731312

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := 
by 
  sorry

end power_inequality_l731_731312


namespace MR_plus_MS_eq_2AF_l731_731056

variables {Point : Type} [EuclideanGeometry Point]
variables (A B C D M S R F Q : Point)

-- Assuming ABCD is a parallelogram
axiom parallelogram_ABCD : is_parallelogram A B C D

-- M is a point on diagonal AC
axiom M_on_AC : lies_on_line M A C

-- MS ⊥ BD
axiom MS_perp_BD : perpendicular M S B D

-- MR ⊥ AC
axiom MR_perp_AC : perpendicular M R A C

-- AF ⊥ BD, where F is the midpoint of BD
axiom AF_perp_BD : perpendicular A F B D
axiom F_midpoint_BD : is_midpoint F B D

-- MQ ⊥ AF
axiom MQ_perp_AF : perpendicular M Q A F

-- We want to prove MR + MS = 2AF
theorem MR_plus_MS_eq_2AF :
  segment_length M R + segment_length M S = 2 * segment_length A F := 
sorry

end MR_plus_MS_eq_2AF_l731_731056


namespace remainder_when_divided_by_9_l731_731052

theorem remainder_when_divided_by_9 (z : ℤ) (k : ℤ) (h : z + 3 = 9 * k) :
  z % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l731_731052


namespace sum_of_n_values_such_that_fraction_is_integer_l731_731887

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l731_731887


namespace triangle_proof_1_triangle_proof_2_l731_731656

noncomputable def triangle_answer_1 (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = b * cos C + c * cos A)
  (h2 : b = a * cos A + c * cos B)
  (h3 : c = a * cos B + b * cos A) 
  (h4 : (a + c)^2 = b^2 + 2 * real.sqrt 3 * a * c * sin C) : Prop :=
  B = real.pi / 3

noncomputable def triangle_answer_2 (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = b * cos C + c * cos A)
  (h2 : b = a * cos A + c * cos B)
  (h3 : c = a * cos B + b * cos A) 
  (h4 : (a + c)^2 = b^2 + 2 * real.sqrt 3 * a * c * sin C)
  (h5 : b = 8)
  (h6 : a > c)
  (h7 : 0.5 * a * c * sin B = 3 * real.sqrt 3) : Prop :=
  a = 5 + real.sqrt 13

theorem triangle_proof_1 (a b c A B C : ℝ)
  (h1 : a = b * cos C + c * cos A)
  (h2 : b = a * cos A + c * cos B)
  (h3 : c = a * cos B + b * cos A) 
  (h4 : (a + c)^2 = b^2 + 2 * real.sqrt 3 * a * c * sin C) : triangle_answer_1 a b c A B C h1 h2 h3 h4 :=
sorry

theorem triangle_proof_2 (a b c A B C : ℝ)
  (h1 : a = b * cos C + c * cos A)
  (h2 : b = a * cos A + c * cos B)
  (h3 : c = a * cos B + b * cos A) 
  (h4 : (a + c)^2 = b^2 + 2 * real.sqrt 3 * a * c * sin C)
  (h5 : b = 8)
  (h6 : a > c)
  (h7 : 0.5 * a * c * sin B = 3 * real.sqrt 3) : triangle_answer_2 a b c A B C h1 h2 h3 h4 h5 h6 h7 :=
sorry

end triangle_proof_1_triangle_proof_2_l731_731656


namespace closest_integer_to_cuberoot_of_200_l731_731021

theorem closest_integer_to_cuberoot_of_200 :
  ∃ n : ℤ, (n = 6) ∧ ( |200 - n^3| ≤ |200 - m^3|  ) ∀ m : ℤ := sorry

end closest_integer_to_cuberoot_of_200_l731_731021


namespace sum_b_first_10_terms_l731_731222

noncomputable def a_n (n : ℕ) : ℚ := if n = 0 then 0 else (2 * n - 1)

noncomputable def b (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

theorem sum_b_first_10_terms :
  (∑ n in Finset.range 10, b n) = 10 / 21 :=
by
  sorry

end sum_b_first_10_terms_l731_731222


namespace sum_lt_prod_probability_l731_731821

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731821


namespace solve_inequality_l731_731380

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731380


namespace students_with_two_talents_l731_731633

-- Definitions based on problem conditions
def total_students : ℕ := 120
def cannot_sing : ℕ := 50
def cannot_dance : ℕ := 75
def cannot_act : ℕ := 35

-- Definitions derived from the conditions
def can_sing : ℕ := total_students - cannot_sing
def can_dance : ℕ := total_students - cannot_dance
def can_act : ℕ := total_students - cannot_act

theorem students_with_two_talents (h1 : total_students = 120) 
                                  (h2 : cannot_sing = 50) 
                                  (h3 : cannot_dance = 75) 
                                  (h4 : cannot_act = 35) 
                                  (h5 : (total_students - cannot_sing) + (total_students - cannot_dance) + (total_students - cannot_act) - total_students = 80) 
                                  : 
                                  can_sing + can_dance + can_act - total_students = 80 :=
by {
  rw [h1, h2, h3, h4],
  simp,
  sorry
}

end students_with_two_talents_l731_731633


namespace area_aoh_eq_area_boh_plus_coh_l731_731198

-- Given definitions and conditions
variables {A B C O H : Type} [acute_triangle A B C]
variables [circumcenter O A B C] [orthocenter H A B C]

-- Mathematical problem statement
theorem area_aoh_eq_area_boh_plus_coh 
  (h_triangle : acute_triangle A B C)
  (h_circumcenter : circumcenter O A B C)
  (h_orthocenter : orthocenter H A B C) :
  (area_of_triangle A O H) = (area_of_triangle B O H) + (area_of_triangle C O H) :=
sorry

end area_aoh_eq_area_boh_plus_coh_l731_731198


namespace simplest_sqrt_l731_731439

-- Define the given square roots
def sqrt6 := Real.sqrt 6
def sqrt8 := Real.sqrt 8
def sqrt_one_over_three := Real.sqrt (1/3)
def sqrt4 := Real.sqrt 4

-- State that sqrt6 is the simplest among the given square roots
theorem simplest_sqrt : sqrt6 = Real.sqrt 6 ∧
  ¬ (∃ (x : ℝ), x^2 = 6 ∧ x < sqrt6) ∧
  (sqrt8 ≠ Real.sqrt 8 ∨ sqrt8 = 2*Real.sqrt 2) ∧
  (sqrt_one_over_three ≠ Real.sqrt (1/3) ∨ House.ksqrt (1/Real.sqrt 3) * Real.sqrt 3 ≠ Real.sqrt (1/3)) ∧
  (sqrt4 ≠ Real.sqrt 4 ∨ sqrt4 = 2) :=
sorry

end simplest_sqrt_l731_731439


namespace solve_inequality_l731_731376

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l731_731376


namespace smallest_positive_n_l731_731708

theorem smallest_positive_n
  (a x y : ℤ)
  (h1 : x ≡ a [ZMOD 9])
  (h2 : y ≡ -a [ZMOD 9]) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 9 = 0 ∧ n = 6 :=
by
  sorry

end smallest_positive_n_l731_731708


namespace girls_with_tablets_l731_731629

def total_boys (b : ℕ) : ℕ := b
def total_students_with_tablets (t : ℕ) : ℕ := t
def boys_with_tablets (bt : ℕ) : ℕ := bt

theorem girls_with_tablets
  (b : ℕ) (t : ℕ) (bt : ℕ)
  (hb : total_boys b = 20)
  (ht : total_students_with_tablets t = 24)
  (hbt : boys_with_tablets bt = 11) :
  t - bt = 13 :=
by
  rw [ht, hbt]
  norm_num

end girls_with_tablets_l731_731629


namespace sum_of_squares_l731_731331

theorem sum_of_squares (n : Nat) (h : n = 2005^2) : 
  ∃ a1 b1 a2 b2 a3 b3 a4 b4 : Int, 
    (n = a1^2 + b1^2 ∧ a1 ≠ 0 ∧ b1 ≠ 0) ∧ 
    (n = a2^2 + b2^2 ∧ a2 ≠ 0 ∧ b2 ≠ 0) ∧ 
    (n = a3^2 + b3^2 ∧ a3 ≠ 0 ∧ b3 ≠ 0) ∧ 
    (n = a4^2 + b4^2 ∧ a4 ≠ 0 ∧ b4 ≠ 0) ∧ 
    (a1, b1) ≠ (a2, b2) ∧ 
    (a1, b1) ≠ (a3, b3) ∧ 
    (a1, b1) ≠ (a4, b4) ∧ 
    (a2, b2) ≠ (a3, b3) ∧ 
    (a2, b2) ≠ (a4, b4) ∧ 
    (a3, b3) ≠ (a4, b4) :=
by
  sorry

end sum_of_squares_l731_731331


namespace seq_2008_is_2017036_l731_731940

def seq : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+1) := seq n + (n + 1)

theorem seq_2008_is_2017036 : seq 2008 = 2017036 :=
sorry

end seq_2008_is_2017036_l731_731940


namespace probability_sum_less_than_product_l731_731814

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731814


namespace probability_of_being_selected_l731_731549

theorem probability_of_being_selected (total_students : ℕ) (eliminated_students : ℕ) (selected_students : ℕ) :
  total_students = 2013 →
  eliminated_students = 13 →
  selected_students = 50 →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ total_students → probability_of_selection i = 1 / total_students) →
  probability_of_selection 1 = 50 / 2013 :=
by
  intros h1 h2 h3 h4
  -- The proof is omitted
  sorry

def probability_of_selection (i : ℕ) : ℝ :=
  if i ≤ 50
  then 1 / 2013
  else 0

end probability_of_being_selected_l731_731549


namespace convex_polygon_isosceles_triangles_equal_sides_l731_731929

theorem convex_polygon_isosceles_triangles_equal_sides {P : Type*} [polygon P] (h_convex : convex P)
  (h_divided : divided_by_non_intersecting_diagonals P isosceles_triangle) :
  ∃ (a b : side P), a = b := sorry

end convex_polygon_isosceles_triangles_equal_sides_l731_731929


namespace incorrect_calculation_l731_731712

noncomputable def ξ : ℝ := 3 -- Expected lifetime of the sensor
noncomputable def η : ℝ := 5 -- Expected lifetime of the transmitter
noncomputable def T (ξ η : ℝ) : ℝ := min ξ η -- Lifetime of the entire device

theorem incorrect_calculation (h1 : E ξ = 3) (h2 : E η = 5) (h3 : E (min ξ η ) = 3.67) : False :=
by
  have h4 : E (min ξ η ) ≤ 3 := sorry -- Based on properties of expectation and min
  have h5 : 3.67 > 3 := by linarith -- Known inequality
  sorry

end incorrect_calculation_l731_731712


namespace quadratic_function_range_l731_731173

theorem quadratic_function_range (x : ℝ) (y : ℝ) (h1 : y = x^2 - 2*x - 3) (h2 : -2 ≤ x ∧ x ≤ 2) :
  -4 ≤ y ∧ y ≤ 5 :=
sorry

end quadratic_function_range_l731_731173


namespace smallest_number_is_neg1_l731_731948

def eval_neg3_sq : ℚ := (-3)^2
def eval_abs_neg9 : ℚ := |(-9:ℚ)|
def eval_neg1_to_4 : ℚ := -(1^4 : ℚ)
def nums : List ℚ := [0, eval_neg3_sq, eval_abs_neg9, eval_neg1_to_4]

theorem smallest_number_is_neg1 :
  eval_neg3_sq = 9 ∧ eval_abs_neg9 = 9 ∧ eval_neg1_to_4 = -1 ∧ nums = [0, 9, 9, -1] → 
  ∀ n ∈ nums, n ≥ -1 :=
begin
  sorry
end

end smallest_number_is_neg1_l731_731948


namespace math_proof_problem_l731_731191

noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * b * x^2 + c * x
noncomputable def f' (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem math_proof_problem (a b c : ℝ)
  (h1: f' a b c (-1) = 0)
  (h2: f' a b c (1) = 1)
  (h3: ∀ x : ℝ, x ≤ f' a b c x) :
  (f a b c x = (1/12) * x^3 + (1/4) * x^2 + (1/4) * x) ∧
  (∀ x ∈ set.Icc (1/2 : ℝ) 2, x^3 - x^2 - 3 ≤ 1) ∧
  (∀ x ∈ set.Icc (1/2 : ℝ) 2, ∃ m : ℝ, m ≥ 1 → (m / x + x * real.log x) ≥ (x^3 - x^2 - 3)) :=
sorry

end math_proof_problem_l731_731191


namespace pupils_in_program_l731_731778

theorem pupils_in_program {total_people parents : ℕ} (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end pupils_in_program_l731_731778


namespace log_base_8_of_512_l731_731998

theorem log_base_8_of_512 : log 8 512 = 3 := by
  have h₁ : 8 = 2^3 := by rfl
  have h₂ : 512 = 2^9 := by rfl
  rw [h₂, h₁]
  sorry

end log_base_8_of_512_l731_731998


namespace february_sales_increase_l731_731490

theorem february_sales_increase (Slast : ℝ) (r : ℝ) (Sthis : ℝ) 
  (h_last_year_sales : Slast = 320) 
  (h_percent_increase : r = 0.25) : 
  Sthis = 400 :=
by
  have h1 : Sthis = Slast * (1 + r) := sorry
  sorry

end february_sales_increase_l731_731490


namespace parallel_lines_iff_a_eq_2_l731_731603

-- Define line equations
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - a + 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 2 = 0

-- Prove that a = 2 is necessary and sufficient for the lines to be parallel.
theorem parallel_lines_iff_a_eq_2 (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → ∃ u v : ℝ, l2 a u v → x = u ∧ y = v) ↔ (a = 2) :=
by {
  sorry
}

end parallel_lines_iff_a_eq_2_l731_731603


namespace negation_1_negation_2_negation_3_negation_4_l731_731442

-- (1) Negation statement and falsehood for arbitrary x in ℝ
theorem negation_1 (x : ℝ) : (x^2 - 4 ≠ 0) = False := 
    sorry

-- (2) Negation statement and falsehood for T ≠ 2kπ where k ∈ ℤ
theorem negation_2 (x : ℝ) : (∃ T : ℝ, (T ≠ 2 * k * Real.pi) ∧ (k : ℤ) ∧ (Real.sin (x + T) ≠ Real.sin x)) = False := 
    sorry

-- (3) Negation statement and falsehood for A, a subset of A ∪ B and A ∩ B
theorem negation_3 {α : Type*} (A B : Set α) : (∃ A, ¬ (A ⊆ A ∪ B ∧ A ⊆ A ∩ B)) = False := 
    sorry

-- (4) Negation statement and falsehood for skew lines a and b, with ∀A ∈ a, ∀B ∈ b, AB ⊥ a and AB ⊥ b
theorem negation_4 (a b : Set ℝ) (A : ℝ) (B : ℝ) : (∀ A ∈ a, ∀ B ∈ b, ¬ (AB ∥ a ∧ AB ∥ b)) = False := 
    sorry

end negation_1_negation_2_negation_3_negation_4_l731_731442


namespace set_equality_theorem_l731_731185

noncomputable theory

open Classical

theorem set_equality_theorem {a b : ℝ} (h : {1, a, b / a} = {0, a^2, a + b}) : a^2015 + b^2015 = -1 := by
  sorry

end set_equality_theorem_l731_731185


namespace calc1_calc2_calc3_l731_731962

theorem calc1 : 1 - 2 + 3 - 4 + 5 = 3 := by sorry
theorem calc2 : - (4 / 7) / (8 / 49) = - (7 / 2) := by sorry
theorem calc3 : ((1 / 2) - (3 / 5) + (2 / 3)) * (-15) = - (17 / 2) := by sorry

end calc1_calc2_calc3_l731_731962


namespace scheduling_methods_l731_731271

-- Definitions for the problem
def subjects : List String := ["Chinese", "Mathematics", "Physical Education", "Foreign Language"]
def periods : List Nat := [1, 2, 3, 4]

-- The condition that the Physical Education teacher is unavailable for the first and second periods
def pe_unavailable_periods : List Nat := [1, 2]

-- The condition that Physical Education can only be arranged in the 3rd or 4th period
def pe_available_periods : List Nat := [3, 4]

-- The theorem to prove the number of different scheduling methods
theorem scheduling_methods : (Finset.card({p // p ∉ pe_unavailable_periods && p ∈ pe_available_periods}).purge (λ p, p ∉ periods)) * (Finset.card({q // q ∉ pe_available_periods}).purge (λ q, q ∉ periods)) = 12 := by
  -- This is where the proof steps would go, but we'll just put sorry here to indicate the target result
  sorry

end scheduling_methods_l731_731271


namespace absent_minded_scientist_mistake_l731_731732

theorem absent_minded_scientist_mistake (ξ η : ℝ) (h₁ : E ξ = 3) (h₂ : E η = 5) (h₃ : E (min ξ η) = 3 + 2/3) : false :=
by
  sorry

end absent_minded_scientist_mistake_l731_731732


namespace abs_c_eq_9_l731_731387

theorem abs_c_eq_9 {a b c : ℤ} 
  (h : a * (1 + complex.I * real.sqrt 3)^3 + b * (1 + complex.I * real.sqrt 3)^2 + c * (1 + complex.I * real.sqrt 3) + b + a = 0)
  (gcd_cond : int.gcd (int.gcd a b) c = 1) :
  |c| = 9 :=
sorry

end abs_c_eq_9_l731_731387


namespace probability_sum_less_than_product_l731_731838

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731838


namespace sum_of_n_values_such_that_fraction_is_integer_l731_731884

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l731_731884


namespace exists_a_b_not_multiple_p_l731_731559

theorem exists_a_b_not_multiple_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬ (m^3 + 2017 * a * m + b) ∣ (p : ℤ) :=
sorry

end exists_a_b_not_multiple_p_l731_731559


namespace equal_variance_properties_l731_731290

def eq_variance_seq (a : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ^ 2 - a (n+1) ^ 2 = p

theorem equal_variance_properties (a : ℕ → ℝ) (p : ℝ) (k : ℕ) :
  (eq_variance_seq a p) →
  (∀ n : ℕ, n ≥ 1 → a (n+1) ^ 2 - a n ^ 2 = -p) ∧
  (eq_variance_seq (λ n, (-1 : ℝ) ^ n) 0) ∧
  (k ≥ 1 → eq_variance_seq (λ n, a (k * n)) (k * p)) :=
begin
  sorry
end

end equal_variance_properties_l731_731290


namespace interval_of_monotonic_increase_solve_triangle_ABC_l731_731607

noncomputable def f (x : ℝ) : ℝ :=
    let m := (sin x - real.sqrt 3 * cos x, 1)
    let n := (sin (real.pi / 2 + x), real.sqrt 3 / 2)
    m.1 * n.1 + m.2 * n.2

theorem interval_of_monotonic_increase :
  ∃ k : ℤ, ∀ x : ℝ, (f x).deriv > 0 ↔ x ∈ set.Icc (-real.pi / 12 + k * real.pi) (5 * real.pi / 12 + k * real.pi) :=
sorry

theorem solve_triangle_ABC (a b c : ℝ) (A B C : ℝ) :
  a = 3 → f (A / 2 + real.pi / 12) = 1/2 → sin C = 2 * sin B →
  A = real.pi / 3 ∧ b = real.sqrt 3 ∧ c = 2 * real.sqrt 3 :=
sorry

end interval_of_monotonic_increase_solve_triangle_ABC_l731_731607


namespace total_number_of_students_l731_731675

theorem total_number_of_students (b h p s : ℕ) 
  (h1 : b = 30)
  (h2 : b = 2 * h)
  (h3 : p = h + 5)
  (h4 : s = 3 * p) :
  b + h + p + s = 125 :=
by sorry

end total_number_of_students_l731_731675


namespace seven_points_with_distance_one_l731_731963

noncomputable def exists_seven_points : Prop :=
  ∃ (P : Fin 7 → ℝ × ℝ), ∀ s : Finset (Fin 7), s.card = 3 → ∃ (i j ∈ s), i ≠ j ∧ dist (P i) (P j) = 1

theorem seven_points_with_distance_one : exists_seven_points := 
sorry 


end seven_points_with_distance_one_l731_731963


namespace person_B_correct_probability_l731_731932

-- Define probabilities
def P_A_correct : ℝ := 0.4
def P_A_incorrect : ℝ := 1 - P_A_correct
def P_B_correct_if_A_incorrect : ℝ := 0.5
def P_B_correct : ℝ := P_A_incorrect * P_B_correct_if_A_incorrect

-- Theorem statement
theorem person_B_correct_probability : P_B_correct = 0.3 :=
by
  -- Problem conditions implicitly used in definitions
  sorry

end person_B_correct_probability_l731_731932


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l731_731125

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l731_731125


namespace rebecca_gemstones_needed_l731_731702

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l731_731702


namespace net_percent_change_over_five_years_l731_731408

def yearly_changes := [1.2, 0.9, 1.15, 0.7, 1.2]

def expected_result := 422

theorem net_percent_change_over_five_years (fc : yearly_changes) : 
  let prod := list.prod fc in
  (prod - 1) * 100 = expected_result :=
by
  sorry

end net_percent_change_over_five_years_l731_731408


namespace t_shirt_coloring_l731_731955

theorem t_shirt_coloring 
  (G : Type) [graph G]
  (V : Type) [vertices V] (v : V → Prop)
  (E : Type) [edges E] (e : E → Prop)
  (deg : V → ℕ)
  (h1 : ∀ v, 50 ≤ deg v ∧ deg v ≤ 100)
  (k : ℕ)  : 
  ∃ (coloring : V → ℕ), (∀ v, finset.card (finset.image coloring (neighbors v)) ≥ 20) ∧ (∀ v, coloring v ≤ 1331) :=
sorry

end t_shirt_coloring_l731_731955


namespace graph_of_equation_is_two_lines_l731_731907

-- Defining the condition
variables {a : ℝ} (x y : ℝ)

-- Define the equation
def equation := (x + a * y)^2 = (x^2 + y^2)

-- State the theorem
theorem graph_of_equation_is_two_lines (h : equation x y) : 
  (y = 0) ∨ (2 * a * x + (a^2 - 1) * y = 0) :=
by {
  sorry
}

end graph_of_equation_is_two_lines_l731_731907


namespace sum_of_series_l731_731138

-- Define the imaginary unit i.
def i : ℂ := Complex.I

-- Define the geometric series sum formula.
def geom_series_sum (a r : ℂ) (n : ℕ) : ℂ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_series : geom_series_sum i i 2018 = -1 + i := by
  sorry

end sum_of_series_l731_731138


namespace possible_values_of_a₁_l731_731045

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end possible_values_of_a₁_l731_731045


namespace can_be_rotated_180_degrees_cannot_be_rotated_90_degrees_l731_731415

-- Definitions (conditions)
variable {Cube : Type} 
variable (initial_position : Cube) -- initial position of the cube
variable (roll_cube : Cube → Cube) -- function to roll the cube
variable (returns_to_original : Cube → Prop) -- predicate for returning to original position

-- Hypotheses based on the conditions
axiom (h_initial : initial_position ≠ initial_position)
axiom (h_rolls : ∀ (c : Cube), (∃ n : ℕ, roll_cube^[n] c = initial_position))
axiom (h_returns : ∀ (c : Cube), returns_to_original c → c = initial_position)

-- Part (a) proof statement
theorem can_be_rotated_180_degrees :
  ∃ n : ℕ, (∃ k : ℕ, roll_cube^[k * 2] initial_position = initial_position ∧ k = n ∧ returns_to_original (roll_cube^[k * 2] initial_position)) :=
sorry

-- Part (b) proof statement
theorem cannot_be_rotated_90_degrees :
  ∀ n : ℕ, ¬(∃ m : ℕ, (m mod 4 = 1 ∨ m mod 4 = 3) ∧ roll_cube^[m] initial_position = roll_cube^[n] initial_position ∧ returns_to_original (roll_cube^[m] initial_position)) :=
sorry

end can_be_rotated_180_degrees_cannot_be_rotated_90_degrees_l731_731415


namespace probability_sum_less_than_product_is_5_div_9_l731_731803

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731803


namespace jose_bottle_caps_proof_l731_731665

def jose_bottle_caps_initial : Nat := 7
def rebecca_bottle_caps : Nat := 2
def jose_bottle_caps_final : Nat := 9

theorem jose_bottle_caps_proof : jose_bottle_caps_initial + rebecca_bottle_caps = jose_bottle_caps_final := by
  sorry

end jose_bottle_caps_proof_l731_731665


namespace original_bananas_l731_731143

theorem original_bananas (removed left : ℕ) (h_removed : removed = 5) (h_left : left = 41) :
  (removed + left = 46) :=
by { rw [h_removed, h_left], exact rfl }

end original_bananas_l731_731143


namespace nancy_shoes_l731_731683

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l731_731683


namespace closest_integer_to_cube_root_of_200_l731_731030

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), n = 6 ∧ (n^3 = 216 ∨ n^3 > 125 ∧ n^3 < 216) := 
by
  existsi 6
  split
  · refl
  · right
    split
    · norm_num
    · norm_num

end closest_integer_to_cube_root_of_200_l731_731030


namespace probability_x_plus_2y_lt_6_l731_731079

noncomputable def prob_x_plus_2y_lt_6 : ℚ :=
  let rect_area : ℚ := (4 : ℚ) * 3
  let quad_area : ℚ := (4 : ℚ) * 1 + (1 / 2 : ℚ) * 4 * 2
  quad_area / rect_area

theorem probability_x_plus_2y_lt_6 :
  prob_x_plus_2y_lt_6 = 2 / 3 :=
by
  sorry

end probability_x_plus_2y_lt_6_l731_731079


namespace wise_men_can_take_gold_l731_731743

-- Defining initial conditions
def initial_coins (n : ℕ) : ℕ :=
if n ≥ 1 ∧ n ≤ 10 then n else 0

-- Action definition: Every minute, give 1 coin to any 9 wise men and take 1 coin from another
def action (coins : ℕ) (k : ℕ) : ℕ :=
if k > 0 then coins + 9 - 1 else coins

-- Prophecy that there exists a sequence of actions so that all wise men have 55 coins
theorem wise_men_can_take_gold :
  ∃ (n : ℕ) (actions : List (ℕ → ℕ)), 
    (∀ i, initial_coins i + (actions.foldr (λ act acc, act acc) 0) = 55) :=
sorry

end wise_men_can_take_gold_l731_731743


namespace jimmy_climb_time_l731_731121

theorem jimmy_climb_time (a d n : ℕ) (h_a : a = 25) (h_d : d = 7) (h_n : n = 6) :
  let l := a + (n - 1) * d in
  let S := n * (a + l) / 2 in
  S = 255 :=
by
  rw [h_a, h_d, h_n]
  let l := 25 + (6 - 1) * 7
  let S := 6 * (25 + l) / 2
  have h_l : l = 60 := by norm_num
  rw h_l
  have h_S : S = 255 := by norm_num
  exact h_S
  done

end jimmy_climb_time_l731_731121


namespace sqrt_74_product_l731_731406

theorem sqrt_74_product :
  ∃ (a b : ℕ), a < b ∧ a^2 ≤ 74 ∧ 74 < b^2 ∧ a * b = 72 :=
by {
  use [8, 9],
  split,
  { exact nat.lt_succ_self 8 },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
}

--sorry

end sqrt_74_product_l731_731406


namespace line_through_A_and_B_l731_731566

variables (x y x₁ y₁ x₂ y₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * x₁ - 4 * y₁ - 2 = 0
def condition2 : Prop := 3 * x₂ - 4 * y₂ - 2 = 0

-- Proof that the line passing through A(x₁, y₁) and B(x₂, y₂) is 3x - 4y - 2 = 0
theorem line_through_A_and_B (h1 : condition1 x₁ y₁) (h2 : condition2 x₂ y₂) :
    ∀ (x y : ℝ), (∃ k : ℝ, x = x₁ + k * (x₂ - x₁) ∧ y = y₁ + k * (y₂ - y₁)) → 3 * x - 4 * y - 2 = 0 :=
sorry

end line_through_A_and_B_l731_731566


namespace ellipse_general_eqn_length_of_chord_AB_l731_731650

-- Definition for parametric equations of line l
def line_parametric_eqns (t α : ℝ) (hα : 0 < α ∧ α < π / 2) :=
  ∃ x y, x = 2 + t * Real.cos α ∧ y = sqrt 3 + t * Real.sin α

-- Definition for parametric equations of ellipse M
def ellipse_parametric_eqns (β : ℝ) :=
  ∃ x y, x = 2 * Real.cos β ∧ y = Real.sin β

-- Definition for the standard equation of circle C
def circle_eqn (x y : ℝ) := (x - 1)^2 + y^2 = 1

-- The proof problem part 1: General equation of ellipse M
theorem ellipse_general_eqn (x y β : ℝ) :
  (2 * Real.cos β = x) ∧ (Real.sin β = y) → (x^2 / 4 + y^2 = 1) :=
sorry

-- The proof problem part 2: Length of chord AB
theorem length_of_chord_AB (α β t1 t2 : ℝ) (hα : 0 < α ∧ α < π / 2) :
  (∀ t : ℝ, line_parametric_eqns t α hα ∧ circle_eqn (2 + t * Real.cos α) (sqrt 3 + t * Real.sin α)) →
  (∀ β : ℝ, ellipse_parametric_eqns β) →
  (|t1 + t2| = 24 * sqrt 3 / 7 ∧ t1 * t2 = 48 / 7) →
  (|t1 - t2| = 8 * sqrt 6 / 7) :=
sorry

end ellipse_general_eqn_length_of_chord_AB_l731_731650


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l731_731014

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (finset.sum (finset.range 9) (λ k, (binomial 8 k) * (x ^ k) * (y ^ (8 - k)))) = 56 :=
by sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l731_731014


namespace tangent_line_circle_l731_731579

open Real

theorem tangent_line_circle (a : ℝ)
    (h_circle : ∀ (x y : ℝ), (x - a)^2 + y^2 = 4)
    (h_line : ∀ (x y : ℝ), x - y + sqrt 2 = 0)
    (h_tangent : ∀ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x - y + sqrt 2 = 0 → distance (a, 0) (x, y) = 2) :
    a = sqrt 2 ∨ a = -3 * sqrt 2 :=
sorry

end tangent_line_circle_l731_731579


namespace complete_the_square_l731_731433

theorem complete_the_square (x : ℝ) : 
  ∃ c : ℝ, (x^2 - 4 * x + c) = (5 + c) ∧ c = 4 :=
by
  use 4
  split
  case left =>
    sorry
  case right =>
    rfl

end complete_the_square_l731_731433


namespace x_minus_y_eq_3_l731_731245

variables {α : Type*} [add_comm_group α] [module ℝ α]

theorem x_minus_y_eq_3 (a b : α) (h : ¬ collinear ℝ ({a, b} : set α)) (x y : ℝ)
  (h_eq : (3*x - 4*y) • a + (2*x - 3*y) • b = 6 • a + 3 • b) :
  x - y = 3 :=
sorry

end x_minus_y_eq_3_l731_731245


namespace combinatorial_problem_correct_l731_731939

def combinatorial_problem : Prop :=
  let boys := 4
  let girls := 3
  let chosen_boys := 3
  let chosen_girls := 2
  let num_ways_select := Nat.choose boys chosen_boys * Nat.choose girls chosen_girls
  let arrangements_no_consecutive_girls := 6 * Nat.factorial 4 / Nat.factorial 2
  num_ways_select * arrangements_no_consecutive_girls = 864

theorem combinatorial_problem_correct : combinatorial_problem := 
  by 
  -- proof to be provided
  sorry

end combinatorial_problem_correct_l731_731939


namespace average_correct_l731_731857

theorem average_correct :
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1252140 + 2345) / 10 = 125831.9 := 
sorry

end average_correct_l731_731857


namespace value_of_x_l731_731619

theorem value_of_x (x : ℤ) (h : x + 3 = 4 ∨ x + 3 = -4) : x = 1 ∨ x = -7 := sorry

end value_of_x_l731_731619


namespace point_in_cross_hatched_region_l731_731324

def circle (center : ℝ × ℝ) (radius : ℝ) := 
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2}

def valid_point (C1 C2 : set (ℝ × ℝ)) (M : ℝ × ℝ) : Prop :=
  ∀ l : set (ℝ × ℝ), (l = {p | ∃ k b, p.2 = k * p.1 + b}) → (l ∩ {M} ≠ ∅) → ((l ∩ C1 ≠ ∅) ∨ (l ∩ C2 ≠ ∅))

def cross_hatched_region (tangents : set (set (ℝ × ℝ))) : set (ℝ × ℝ) := 
  -- Assume this function defines the cross-hatched region correctly

theorem point_in_cross_hatched_region {center1 center2 : ℝ × ℝ} {radius1 radius2 : ℝ} 
  (C1 := circle center1 radius1) 
  (C2 := circle center2 radius2)
  (tangents : set (set (ℝ × ℝ))) -- Assuming tangents is correctly defined
  (M : ℝ × ℝ) :
  valid_point C1 C2 M ↔ M ∈ cross_hatched_region tangents :=
sorry

end point_in_cross_hatched_region_l731_731324


namespace segment_length_of_points_A_l731_731486

-- Define the basic setup
variable (d BA CA : ℝ)
variable {A B C : Point} -- Assume we have a type Point for the geometric points

-- Establish some conditions: A right triangle with given lengths
def is_right_triangle (A B C : Point) : Prop := sorry -- Placeholder for definition

def distance (P Q : Point) : ℝ := sorry -- Placeholder for the distance function

-- Conditions
variables (h_right_triangle : is_right_triangle A B C)
variables (h_hypotenuse : distance B C = d)
variables (h_smallest_leg : min (distance B A) (distance C A) = min BA CA)

-- The theorem statement
theorem segment_length_of_points_A (h_right_triangle : is_right_triangle A B C)
                                    (h_hypotenuse : distance B C = d)
                                    (h_smallest_leg : min (distance B A) (distance C A) = min BA CA) :
  ∃ A, (∀ t : ℝ, distance O A = d - min BA CA) :=
sorry -- Proof to be provided

end segment_length_of_points_A_l731_731486


namespace probability_sum_less_than_product_l731_731813

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731813


namespace sum_of_all_n_l731_731894

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l731_731894


namespace sine_A_in_triangle_ABC_l731_731270

theorem sine_A_in_triangle_ABC (a c : ℝ) (C : Real.Angle) (h1 : c = 4) (h2 : a = 2) (h3 : C = Real.Angle.ofDegrees 45) : 
  Real.sinA = (Real.sin (Real.Angle.ofDegrees 45) / 2) :=
sorry

end sine_A_in_triangle_ABC_l731_731270


namespace fraction_ordering_l731_731707

noncomputable def t1 : ℝ := (100^100 + 1) / (100^90 + 1)
noncomputable def t2 : ℝ := (100^99 + 1) / (100^89 + 1)
noncomputable def t3 : ℝ := (100^101 + 1) / (100^91 + 1)
noncomputable def t4 : ℝ := (101^101 + 1) / (101^91 + 1)
noncomputable def t5 : ℝ := (101^100 + 1) / (101^90 + 1)
noncomputable def t6 : ℝ := (99^99 + 1) / (99^89 + 1)
noncomputable def t7 : ℝ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering : t6 < t7 ∧ t7 < t2 ∧ t2 < t1 ∧ t1 < t3 ∧ t3 < t5 ∧ t5 < t4 := by
  sorry

end fraction_ordering_l731_731707


namespace directrix_of_parabola_l731_731539

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := -4 * x^2 + 8 * x - 1

-- Statement to prove the equation of the directrix
theorem directrix_of_parabola : (∀ x : ℝ, parabola x = -4 * x^2 + 8 * x - 1) → 
  (∃ y : ℝ, y = 49 / 16) :=
by
  intro h
  use 49 / 16
  sorry

end directrix_of_parabola_l731_731539


namespace arcsin_sqrt3_div_2_l731_731134

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l731_731134


namespace odd_lattice_points_on_BC_l731_731280

theorem odd_lattice_points_on_BC
  (A B C : ℤ × ℤ)
  (odd_lattice_points_AB : Odd ((B.1 - A.1) * (B.2 - A.2)))
  (odd_lattice_points_AC : Odd ((C.1 - A.1) * (C.2 - A.2))) :
  Odd ((C.1 - B.1) * (C.2 - B.2)) :=
sorry

end odd_lattice_points_on_BC_l731_731280


namespace solve_inequality_l731_731375

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l731_731375


namespace abs_fraction_inequality_solution_l731_731354

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l731_731354


namespace calculate_distance_between_students_l731_731853

def distance_between_students_after_4_hours (speedsA speedsB : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range (n), (speedsA i * 4/3 + speedsB i * 4/3)
  
theorem calculate_distance_between_students 
  (speedsA speedsB: Π i: Fin 3, ℕ)
  (hA: speedsA 0 = 4 ∧ speedsA 1 = 6 ∧ speedsA 2 = 8)
  (hB: speedsB 0 = 7 ∧ speedsB 1 = 9 ∧ speedsB 2 = 11) :
  distance_between_students_after_4_hours speedsA speedsB 3 = 60 :=
by
  sorry

end calculate_distance_between_students_l731_731853


namespace gcd_50421_35343_l731_731170

theorem gcd_50421_35343 : Int.gcd 50421 35343 = 23 := by
  sorry

end gcd_50421_35343_l731_731170


namespace max_area_of_garden_l731_731338

theorem max_area_of_garden (l w : ℝ) 
  (h : 2 * l + w = 400) : 
  l * w ≤ 20000 :=
sorry

end max_area_of_garden_l731_731338


namespace usual_time_to_school_l731_731855

theorem usual_time_to_school (R T : ℝ) (h : (R * T = (6/5) * R * (T - 4))) : T = 24 :=
by 
  sorry

end usual_time_to_school_l731_731855


namespace find_k_l731_731604

-- Define vector a and vector b
def vec_a : (ℝ × ℝ) := (1, 1)
def vec_b : (ℝ × ℝ) := (-3, 1)

-- Define the expression for k * vec_a - vec_b
def k_vec_a_minus_vec_b (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - vec_b.1, k * vec_a.2 - vec_b.2)

-- Define the dot product condition for perpendicular vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved: k = -1 is the value that makes the dot product zero
theorem find_k : ∃ k : ℝ, dot_product (k_vec_a_minus_vec_b k) vec_a = 0 :=
by
  use -1
  sorry

end find_k_l731_731604


namespace total_shoes_l731_731687

variable boots : ℕ
variable slippers : ℕ
variable heels : ℕ

-- Condition: Nancy has six pairs of boots
def boots_pairs : boots = 6 := rfl

-- Condition: Nancy has nine more pairs of slippers than boots
def slippers_pairs : slippers = boots + 9 := rfl

-- Condition: Nancy has a number of pairs of heels equal to three times the combined number of slippers and boots
def heels_pairs : heels = 3 * (boots + slippers) := by
  rw [boots_pairs, slippers_pairs]
  sorry  -- assuming the correctness of the consequent computation as rfl

-- Goal: Total number of individual shoes is 168
theorem total_shoes : (boots * 2) + (slippers * 2) + (heels * 2) = 168 := by
  rw [boots_pairs, slippers_pairs, heels_pairs]
  sorry  -- verifying the summing up to 168 as a proof

end total_shoes_l731_731687


namespace negation_of_proposition_l731_731744

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l731_731744


namespace solve_inequality_l731_731367

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731367


namespace card_hands_leading_digit_A_l731_731640

theorem card_hands_leading_digit_A (n k : ℕ) (h1 : n = 60) (h2 : k = 12)
  (h3 : compute_digits (binom n k) = 'AF30B00A12C0') : get_digit 'A' 'AF30B00A12C0' = 5 :=
sorry

end card_hands_leading_digit_A_l731_731640


namespace range_of_k_specific_k_l731_731453

noncomputable def line_through_point_with_slope (P : ℝ × ℝ) (k : ℝ) : ℝ → ℝ :=
λ x, k * x + P.2

noncomputable def hyperbola (x y : ℝ) : Prop :=
x^2 - (y^2) / 3 = 1

def intersection_points (P : ℝ × ℝ) (k : ℝ) : Prop :=
∃ x1 x2 y1 y2,
line_through_point_with_slope P k x1 = y1 ∧
line_through_point_with_slope P k x2 = y2 ∧
hyperbola x1 y1 ∧
hyperbola x2 y2 ∧
x1 ≠ x2

def valid_k_range (k : ℝ) : Prop :=
k ∈ set.Ioo (-2) (-Real.sqrt 3) ∪
k ∈ set.Ioo (-Real.sqrt 3) (Real.sqrt 3) ∪
k ∈ set.Ioo (Real.sqrt 3) 2

def right_focus (F2 : ℝ × ℝ) : Prop :=
F2 = (2, 0)

def distance (A B : ℝ × ℝ) : ℝ :=
Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def distance_condition (A B F2 : ℝ × ℝ) : Prop :=
distance A F2 + distance B F2 = 6

def find_k (P : ℝ × ℝ) (F2 : ℝ × ℝ) (k : ℝ) : Prop :=
∃ x1 x2 y1 y2,
line_through_point_with_slope P k x1 = y1 ∧
line_through_point_with_slope P k x2 = y2 ∧
hyperbola x1 y1 ∧
hyperbola x2 y2 ∧
x1 ≠ x2 ∧
right_focus F2 ∧
distance_condition (x1, y1) (x2, y2) F2 ∧
(k = 1 ∨ k = -1)

theorem range_of_k : ∀ (k : ℝ), (∃ P : ℝ × ℝ, intersection_points P k) → valid_k_range k :=
sorry

theorem specific_k : ∀ (P : ℝ × ℝ) (F2 : ℝ × ℝ) (k : ℝ), find_k P F2 k :=
sorry

end range_of_k_specific_k_l731_731453


namespace reaction2_follows_markovnikov_l731_731908

-- Define Markovnikov's rule - applying to case with protic acid (HX) to an alkene.
def follows_markovnikov_rule (HX : String) (initial_molecule final_product : String) : Prop :=
  initial_molecule = "CH3-CH=CH2 + HBr" ∧ final_product = "CH3-CHBr-CH3"

-- Example reaction data
def reaction1_initial : String := "CH2=CH2 + Br2"
def reaction1_final : String := "CH2Br-CH2Br"

def reaction2_initial : String := "CH3-CH=CH2 + HBr"
def reaction2_final : String := "CH3-CHBr-CH3"

def reaction3_initial : String := "CH4 + Cl2"
def reaction3_final : String := "CH3Cl + HCl"

def reaction4_initial : String := "CH ≡ CH + HOH"
def reaction4_final : String := "CH3''-C-H"

-- Proof statement
theorem reaction2_follows_markovnikov : follows_markovnikov_rule "HBr" reaction2_initial reaction2_final := by
  sorry

end reaction2_follows_markovnikov_l731_731908


namespace probability_even_sum_l731_731709

def p_even_first_wheel : ℚ := 1 / 3
def p_odd_first_wheel : ℚ := 2 / 3
def p_even_second_wheel : ℚ := 3 / 5
def p_odd_second_wheel : ℚ := 2 / 5

theorem probability_even_sum : 
  (p_even_first_wheel * p_even_second_wheel) + (p_odd_first_wheel * p_odd_second_wheel) = 7 / 15 :=
by
  sorry

end probability_even_sum_l731_731709


namespace probability_sum_less_than_product_l731_731831

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731831


namespace range_of_m_l731_731209

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ [-2, 2] then
    if 0 < x ∧ x ≤ 2 then 2 ^ x - 1
    else if x < 0 ∧ -2 ≤ x then -(2 ^ (-x) - 1)
    else 0
  else 0

def g (x m : ℝ) : ℝ :=
  x ^ 2 - 2 * x + m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Icc (-2 : ℝ) (2 : ℝ), ∃ x2 ∈ Icc (-2 : ℝ) (2 : ℝ), f x1 ≤ g x2 m) ↔ m ≥ -5 :=
by
  sorry

end range_of_m_l731_731209


namespace max_popsicles_l731_731327

/-- 
Given the costs of different popsicle packages and a total budget, 
prove the maximum number of popsicles that can be purchased. 
--/
theorem max_popsicles (budget : ℝ) (cost_single : ℝ) (cost_box_3 : ℝ) (cost_box_7 : ℝ) 
  (single_popsicles : ℕ) (box_3_popsicles : ℕ) (box_7_popsicles : ℕ) :
  budget = 12 →
  cost_single = 1.50 →
  cost_box_3 = 3 →
  cost_box_7 = 5 →
  single_popsicles = 1 →
  box_3_popsicles = 3 →
  box_7_popsicles = 7 →
  ∃ num_popsicles : ℕ, num_popsicles = 15 :=
begin
  sorry
end

end max_popsicles_l731_731327


namespace values_of_m_and_magnitude_l731_731548

variable (m : ℝ)

def z (m : ℝ) : Complex := Complex.mk (m^2 - 2 * m) m

theorem values_of_m_and_magnitude :
  (m = 1 ∨ m = 2) ∧ (z m).re < 0 ∧ (z m).im > 0 ∧ (∀ m, (m = 1 ∨ m = 2) → (|z m| = if m = 1 then Real.sqrt 2 else 2)) := by
  sorry

end values_of_m_and_magnitude_l731_731548


namespace scheduling_with_constraints_is_halved_l731_731496

theorem scheduling_with_constraints_is_halved :
  let professors := ["Professor White", "Professor Black", "Professor C", "Professor D", "Professor E", "Professor F", "Professor G", "Professor H"]
  let total_arrangements := factorial 8
  let white_before_black_arrangements := total_arrangements / 2
  white_before_black_arrangements = 20160 :=
sorry

end scheduling_with_constraints_is_halved_l731_731496


namespace cost_of_one_dozen_pens_l731_731395

variable (x : ℝ)

-- Conditions 1 and 2 as assumptions
def pen_cost := 5 * x
def pencil_cost := x

axiom cost_equation  : 3 * pen_cost + 5 * pencil_cost = 200
axiom cost_ratio     : pen_cost / pencil_cost = 5 / 1 -- ratio is given

-- Question and target statement
theorem cost_of_one_dozen_pens : 12 * pen_cost = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l731_731395


namespace max_non_colored_cubes_l731_731004

open Nat

-- Define the conditions
def isRectangularPrism (length width height volume : ℕ) := length * width * height = volume

-- The theorem stating the equivalent math proof problem
theorem max_non_colored_cubes (length width height : ℕ) (h₁ : isRectangularPrism length width height 1024) :
(length > 2 ∧ width > 2 ∧ height > 2) → (length - 2) * (width - 2) * (height - 2) = 504 := by
  sorry

end max_non_colored_cubes_l731_731004


namespace quadratic_solution_range_l731_731264
noncomputable theory

theorem quadratic_solution_range (a : ℝ) :
  (∃ x ∈ set.Icc 1 5, x^2 + a * x - 2 = 0) → (-23 / 5 ≤ a ∧ a ≤ 1) :=
by
  intro h
  sorry

end quadratic_solution_range_l731_731264


namespace larger_group_men_l731_731921

theorem larger_group_men (m1 : ℕ) (d1 : ℕ) (m2 : ℕ) (d2 : ℕ) : m1 = 12 → d1 = 25 → d2 = 15 → m2 = (m1 * d1) / d2 → m2 = 20 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  exact h4

end larger_group_men_l731_731921


namespace prime_leq_n_add_one_l731_731938

theorem prime_leq_n_add_one (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn_pos : 0 < n)
  (hdiv : p^3 ∣ ∏ k in Finset.range (n+1), (k^3 + 1)) :
  p ≤ n + 1 :=
sorry

end prime_leq_n_add_one_l731_731938


namespace closest_integer_to_cube_root_of_200_l731_731035

theorem closest_integer_to_cube_root_of_200 : 
  let n := 200 in
  let a := 5 in 
  let b := 6 in 
  abs (b - real.cbrt n) < abs (a - real.cbrt n) := 
by sorry

end closest_integer_to_cube_root_of_200_l731_731035


namespace smallest_k_l731_731923

-- Definitions based on given problem
def non_neg_numbers (n : ℕ) := {m : ℕ // m ≤ 100}
def circle (n : ℕ) := fin n → non_neg_numbers n
def allowed_operation := ((x y : non_neg_numbers 2009) → (x.val + 1 ≤ 100) ∧ (y.val + 1 ≤ 100))

noncomputable def min_k (circle : circle 2009) : ℕ := 100400

-- Problem statement in Lean
theorem smallest_k (circle : circle 2009) (k : ℕ)
  (h1 : ∀ i : fin 2009, circle i ≤ 100)
  (h2 : ∀ i : fin 2009, h3 : allowed_operation (circle i) (circle ((i + 1) % 2009.val))):
  min_k circle = 100400 :=
sorry

end smallest_k_l731_731923


namespace students_taking_neither_l731_731412

theorem students_taking_neither (total_students math_students physics_students both_students : ℕ) 
  (h_total : total_students = 80) 
  (h_math : math_students = 50) 
  (h_physics : physics_students = 32) 
  (h_both : both_students = 15) : total_students - (math_students - both_students + physics_students - both_students + both_students) = 13 := 
by
  -- Primary calculations:
  have h_only_math := math_students - both_students,
  have h_only_physics := physics_students - both_students,
  have h_at_least_one := h_only_math + h_only_physics + both_students,
  -- Final calculation:
  have h_neither := total_students - h_at_least_one,
  -- Proof of the goal:
  rw [h_total, h_math, h_physics, h_both] at *,
  unfold h_only_math h_only_physics h_at_least_one h_neither,
  norm_num,
  rfl,

end students_taking_neither_l731_731412


namespace triangle_side_lengths_l731_731330

-- Defining the conditions
variables (a c : ℝ)

-- Main theorem statement
theorem triangle_side_lengths (H T : Point) (ABC : Triangle)
  (AH_altitude : IsAltitude A H BC)
  (BT_bisector : IsAngleBisector B T AC)
  (G_centroid : IsCentroid G ABC)
  (G_on_HT : LiesOnLine G H T)
  (BC_eq_a : SideLength BC = a)
  (AB_eq_c : SideLength AB = c) :
  ∃ b : ℝ, b = sqrt ((c * (a^2 + 2*a*c - c^2)) / (2*a - c)) :=
sorry

end triangle_side_lengths_l731_731330


namespace closest_integer_to_cube_root_of_200_l731_731037

theorem closest_integer_to_cube_root_of_200 : 
  let n := 200 in
  let a := 5 in 
  let b := 6 in 
  abs (b - real.cbrt n) < abs (a - real.cbrt n) := 
by sorry

end closest_integer_to_cube_root_of_200_l731_731037


namespace solve_inequality_l731_731360

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l731_731360


namespace probability_sum_less_than_product_l731_731841

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731841


namespace therapy_hours_l731_731068

variables (F A : ℕ) -- define F and A as natural numbers
variables (h1 : F = A + 40) -- condition 1
variables (h2 : F + A = 174) -- condition 2
variables (total_charge : ℕ) -- total charge
noncomputable def total_hours (total_charge : ℕ) : ℕ :=
let first_hour_cost := F in -- cost of the first hour
let additional_hours_cost := total_charge - first_hour_cost in -- cost for additional hours
  additional_hours_cost / A + 1 -- total hours including the first hour

open_locale classical -- to use the noncomputable feature

theorem therapy_hours (h3 : total_charge = 375) : 
  total_hours F A total_charge h1 h2 h3 = 5 := 
by {
  sorry
}

end therapy_hours_l731_731068


namespace total_pay_l731_731002
noncomputable def employee_pay (Y_pay : ℝ) (X_factor : ℝ) : ℝ :=
  Y_pay * (1 + X_factor)

theorem total_pay (Y_pay : ℝ) (X_factor : ℝ) (total : ℝ) 
  (hY : Y_pay = 263.64) (hX : X_factor = 1.2) (htotal : total = 580.008) :
  employee_pay Y_pay X_factor = total := 
by
  unfold employee_pay
  rw [hY, hX]
  norm_num
  exact htotal
  sorry

end total_pay_l731_731002


namespace polynomial_congruent_mod_p_square_l731_731462

theorem polynomial_congruent_mod_p_square 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (odd_p : p % 2 = 1) :
  ∃ Q : Polynomial ℤ, ∀ x : ℤ, 
  (2 * (1 + x^( (p + 1)/ 2 ) + (1 - x)^((p + 1) / 2 ) )) % p = (Q x)^2 % p := 
sorry

end polynomial_congruent_mod_p_square_l731_731462


namespace gatorade_price_correct_l731_731738

noncomputable def cupcake_sales_before_discount : ℝ := 120 * 2
noncomputable def cookie_sales : ℝ := 100 * 0.5
noncomputable def candy_bar_sales : ℝ := 25 * 1.5
noncomputable def granola_bar_sales_before_discount : ℝ := 15 * 1

noncomputable def cupcake_discount : ℝ := 0.1 * cupcake_sales_before_discount
noncomputable def cupcake_sales_after_discount : ℝ := cupcake_sales_before_discount - cupcake_discount

noncomputable def granola_bar_discount : ℝ := 0.05 * granola_bar_sales_before_discount
noncomputable def granola_bar_sales_after_discount : ℝ := granola_bar_sales_before_discount - granola_bar_discount

noncomputable def total_sales_euros_after_discounts : ℝ :=
  cupcake_sales_after_discount + cookie_sales + candy_bar_sales + granola_bar_sales_after_discount

noncomputable def exchange_rate_to_usd : ℝ := 1.1
noncomputable def total_sales_usd : ℝ := total_sales_euros_after_discounts * exchange_rate_to_usd

noncomputable def sales_tax : ℝ := 0.05 * total_sales_usd
noncomputable def amount_after_sales_tax : ℝ := total_sales_usd - sales_tax

noncomputable def soccer_team_share : ℝ := 0.1 * amount_after_sales_tax
noncomputable def remaining_amount : ℝ := amount_after_sales_tax - soccer_team_share

noncomputable def basketball_price : ℝ := 60
noncomputable def basketball_discount : ℝ := 0.15 * basketball_price
noncomputable def discounted_basketball_price : ℝ := basketball_price - basketball_discount
noncomputable def total_basketball_cost : ℝ := 4 * discounted_basketball_price

noncomputable def amount_left_for_gatorade : ℝ := remaining_amount - total_basketball_cost

noncomputable def bottles_of_gatorade : ℝ := 35
noncomputable def cost_per_bottle_of_gatorade : ℝ := amount_left_for_gatorade / bottles_of_gatorade

theorem gatorade_price_correct :
  cost_per_bottle_of_gatorade ≈ 2.71 := 
sorry

end gatorade_price_correct_l731_731738


namespace point_below_and_left_l731_731704

noncomputable def probability_parallel : ℝ := 1 / 18
noncomputable def probability_intersect : ℝ := 11 / 12

theorem point_below_and_left :
  (λ P Q : ℝ × ℝ, P.1 + 2 * P.2 < 2 ∧ P.1 < Q.1) (probability_parallel, probability_intersect) (2, 0) :=
by
  sorry

end point_below_and_left_l731_731704


namespace statement_B_statement_C_l731_731441

-- Definition for statement B
def vec_e1 : ℝ × ℝ := (2, -3)
def vec_e2 : ℝ × ℝ := (1 / 2, -3 / 4)

theorem statement_B : ∃ k : ℝ, vec_e1 = (k * vec_e2.1, k * vec_e2.2) :=
by
  use 4
  simp [vec_e1, vec_e2]
  norm_num

-- Definition for statement C
def vector_parallel (a b : ℝ → ℝ) : Prop := ∃ k : ℝ, ∀ x, a x = k * b x

def magnitude_projection (a b : ℝ → ℝ) (h : vector_parallel a b) : ℝ :=
if h1: ∃ k, k ≠ 0 ∧ ∀ x y, a x = k * b y then
  norm_num (a 0) * abs (1 / (b 0) * (a 0 / norm_num b 0))
else
  | a 0 |

theorem statement_C (a b : ℝ → ℝ) (h : vector_parallel a b) : magnitude_projection a b h = |a 0| :=
by
  sorry

end statement_B_statement_C_l731_731441


namespace red_wins_in_four_moves_l731_731455

/-- 
  In the game of $\mathbb{R}^2$-tic-tac-toe, two players (Red and Blue) take turns placing points on the $xy$ plane.
  - Red plays first.
  - The objective is to have 3 of their own points in a line without any of the opponent's points in between.
  Prove that Red can guarantee a win in exactly 4 moves.
-/
theorem red_wins_in_four_moves : 
  ∀ (red_moves blue_moves : ℕ),
  (red_moves + blue_moves) % 2 = 0 → (red_moves > blue_moves) → red_moves = 4 → 
  ∃ (sequence : list (ℝ × ℝ)), 
  (∀ (r : ℝ × ℝ), r ∈ sequence → r ∉ sequence.tail.nil) ∧
  ∃ (line : ℝ × ℝ × ℝ), red_wins_in_four_moves → length sequence = 4 := 
  sorry

end red_wins_in_four_moves_l731_731455


namespace night_crew_fraction_of_day_l731_731506

variable (D : ℕ) -- Number of workers in the day crew
variable (N : ℕ) -- Number of workers in the night crew
variable (total_boxes : ℕ) -- Total number of boxes loaded by both crews

-- Given conditions
axiom day_fraction : D > 0 ∧ N > 0 ∧ total_boxes > 0
axiom night_workers_fraction : N = (4 * D) / 5
axiom day_crew_boxes_fraction : (5 * total_boxes) / 7 = (5 * D)
axiom night_crew_boxes_fraction : (2 * total_boxes) / 7 = (2 * N)

-- To prove
theorem night_crew_fraction_of_day : 
  let F_d := (5 : ℚ) / (7 * D)
  let F_n := (2 : ℚ) / (7 * N)
  F_n = (5 / 14) * F_d :=
by
  sorry

end night_crew_fraction_of_day_l731_731506


namespace sum_lt_prod_probability_l731_731820

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731820


namespace zero_in_interval_l731_731556

noncomputable def f (x : ℝ) : ℝ := if x ≥ -2 then 3^x - 4 else 3^(-4 - x) - 4

theorem zero_in_interval :
  (∃ k : ℝ, (3^k - 4) = 0 ∧ -6 < k ∧ k < -5) ∧
  (∃ x : ℝ, (x ≥ -2 ∧ f x = 0) ↔ (-2 - f(-4 - x) = 0)) :=
by
  -- Proof will be added here
  sorry

end zero_in_interval_l731_731556


namespace num_proper_subsets_P_odot_Q_l731_731242

def P : Set ℕ := {4, 5, 6}
def Q : Set ℕ := {1, 2, 3}

def P_odot_Q : Set ℕ := {x | ∃ p q : ℕ, p ∈ P ∧ q ∈ Q ∧ x = p - q}

theorem num_proper_subsets_P_odot_Q : (2 ^ P_odot_Q.to_finset.card - 1) = 31 :=
by {
  sorry
}

end num_proper_subsets_P_odot_Q_l731_731242


namespace ratio_alcohol_to_water_l731_731447

-- Definitions of volume fractions for alcohol and water
def alcohol_volume_fraction : ℚ := 1 / 7
def water_volume_fraction : ℚ := 2 / 7

-- The theorem stating the ratio of alcohol to water volumes
theorem ratio_alcohol_to_water : (alcohol_volume_fraction / water_volume_fraction) = 1 / 2 :=
by sorry

end ratio_alcohol_to_water_l731_731447


namespace plane_parallel_l731_731632

-- Definitions for planes and lines within a plane
variable (Plane : Type) (Line : Type)
variables (lines_in_plane1 : Set Line)
variables (parallel_to_plane2 : Line → Prop)
variables (Plane1 Plane2 : Plane)

-- Conditions
axiom infinite_lines_in_plane1_parallel_to_plane2 : ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l
axiom planes_are_parallel : ∀ (P1 P2 : Plane), (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) → P1 = Plane1 → P2 = Plane2 → (Plane1 ≠ Plane2 ∧ (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l))

-- The proof that Plane 1 and Plane 2 are parallel based on the conditions
theorem plane_parallel : Plane1 ≠ Plane2 → ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l → (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) := 
by
  sorry

end plane_parallel_l731_731632


namespace board_train_immediately_probability_l731_731215

-- Define conditions
def total_time : ℝ := 10
def favorable_time : ℝ := 1

-- Define the probability P(A) as favorable_time / total_time
noncomputable def probability_A : ℝ := favorable_time / total_time

-- State the proposition to prove that the probability is 1/10
theorem board_train_immediately_probability : probability_A = 1 / 10 :=
by sorry

end board_train_immediately_probability_l731_731215


namespace power_calculation_l731_731960

theorem power_calculation : (16 ^ (1 / 12 : ℝ)) * (8 ^ (1 / 2 : ℝ)) / (2 ^ (-1 / 6 : ℝ)) = 4 := by
  sorry

end power_calculation_l731_731960


namespace probability_sum_less_than_product_l731_731818

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731818


namespace log_eight_of_five_twelve_l731_731996

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731996


namespace triangle_perimeter_l731_731512

-- Define the points
def point1 := (2 : ℝ, -2 : ℝ)
def point2 := (8 : ℝ, 4 : ℝ)
def point3 := (2 : ℝ, 4 : ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the perimeter function
def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p1 p3 + distance p2 p3

-- State the theorem
theorem triangle_perimeter :
  perimeter point1 point2 point3 = 12 + 6 * real.sqrt 2 :=
by
  sorry

end triangle_perimeter_l731_731512


namespace john_bought_slurpees_l731_731297

noncomputable def slurpees_bought (total_money paid change slurpee_cost : ℕ) : ℕ :=
  (paid - change) / slurpee_cost

theorem john_bought_slurpees :
  let total_money := 20
  let slurpee_cost := 2
  let change := 8
  slurpees_bought total_money total_money change slurpee_cost = 6 :=
by
  sorry

end john_bought_slurpees_l731_731297


namespace count_valid_5x5_arrays_l731_731247

def is_valid_array (A : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  (∀ i : Fin 5, (∑ j, A i j) = 1) ∧ (∀ j : Fin 5, (∑ i, A i j) = 1) ∧
  (∀ i j, A i j = 1 ∨ A i j = -1)

theorem count_valid_5x5_arrays : Finset.card {A : Matrix (Fin 5) (Fin 5) ℤ | is_valid_array A} = 60 := 
by
  sorry

end count_valid_5x5_arrays_l731_731247


namespace tenth_term_geom_seq_l731_731522

theorem tenth_term_geom_seq :
  let a := (5 : ℚ)
  let r := (4 / 3 : ℚ)
  let n := 10
  (a * r^(n - 1)) = (1310720 / 19683 : ℚ) :=
by
  sorry

end tenth_term_geom_seq_l731_731522


namespace minimum_matches_to_determine_two_strongest_l731_731061

theorem minimum_matches_to_determine_two_strongest (n : ℕ) (h : n = 25) : 
  ∃ m : ℕ, m = 28 ∧ 
    (∀ (players : fin n → ℕ), 
      (∀ i j : fin n, i ≠ j → players i ≠ players j) ∧ 
      (∀ (i j : fin n) (hij : players i < players j), m = 28)) :=
by sorry

end minimum_matches_to_determine_two_strongest_l731_731061


namespace incorrect_calculation_l731_731724

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l731_731724


namespace alok_total_payment_l731_731097

def cost_of_chapatis : Nat := 16 * 6
def cost_of_rice_plates : Nat := 5 * 45
def cost_of_mixed_vegetable_plates : Nat := 7 * 70
def total_cost : Nat := cost_of_chapatis + cost_of_rice_plates + cost_of_mixed_vegetable_plates

theorem alok_total_payment :
  total_cost = 811 := by
  unfold total_cost
  unfold cost_of_chapatis
  unfold cost_of_rice_plates
  unfold cost_of_mixed_vegetable_plates
  calc
    16 * 6 + 5 * 45 + 7 * 70 = 96 + 5 * 45 + 7 * 70 := by rfl
                      ... = 96 + 225 + 7 * 70 := by rfl
                      ... = 96 + 225 + 490 := by rfl
                      ... = 96 + (225 + 490) := by rw Nat.add_assoc
                      ... = (96 + 225) + 490 := by rw Nat.add_assoc
                      ... = 321 + 490 := by rfl
                      ... = 811 := by rfl

end alok_total_payment_l731_731097


namespace find_length_CM_l731_731200

-- Define the vertices of the equilateral triangle
variables {A B C K L M : Type}

-- Define lengths based on given conditions
axiom AK_eq_3 : ∀ {A K : ℝ}, AK = 3
axiom BL_eq_2 : ∀ {B L : ℝ}, BL = 2
axiom KL_eq_KM : ∀ {K L M : ℝ}, KL = KM

-- Statement to prove the length of CM
theorem find_length_CM {A B C K L M : ℝ}
  (AK_eq_3 : AK = 3)
  (BL_eq_2 : BL = 2)
  (KL_eq_KM : KL = KM)
  (equilateral : equilateral_triangle A B C)
  : CM = 5 :=
by
  sorry

end find_length_CM_l731_731200


namespace sum_lt_prod_probability_l731_731819

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731819


namespace probability_sum_less_than_product_l731_731787

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731787


namespace equation_of_line_and_distance_value_of_intercept_l731_731558

-- Part (1) proof problem
theorem equation_of_line_and_distance (l1 : Point → Point → Line) (l : Point → Point → Line) (p : Point) (q : Point) 
  (h1 : l1 = Line.mkPointSlope 0 2) (h2 : l = Line.mkPointSlope 0 2) (hpq : p = ⟨1, 4⟩) :
  (l.equation = Line.mkEquation 2 2) ∧ (Line.distance l1 l = (2 * Real.sqrt 5) / 5) :=
sorry 

-- Part (2) proof problem
theorem value_of_intercept (l : Point → Point → Line) (p : Point) (hp : p = ⟨1, 4⟩) (a : ℝ) (ha: a ≠ 0)
  (hl : l = Line.mkInterceptForm a a) : 
  a = 5 :=
sorry

end equation_of_line_and_distance_value_of_intercept_l731_731558


namespace find_a_l731_731217

theorem find_a (a : ℝ) :
  (1 = a ∨ -3 = a) →
  (let d := abs (a - 2 + 3) / real.sqrt (1^2 + (-1)^2)
  in d = real.sqrt 2) :=
by
  assume h : (1 = a ∨ -3 = a)
  have H_dist_formula : forall x0 y0 A B C, abs (A*x0 + B*y0 + C) / real.sqrt (A^2 + B^2) = real.sqrt 2 := sorry
  sorry

end find_a_l731_731217


namespace x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l731_731461

variable {x : ℝ}
variable {y : ℝ}

theorem x_gt_y_necessary_not_sufficient_for_x_gt_abs_y
  (hx : x > 0) :
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|) := by
  sorry

end x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l731_731461


namespace at_least_one_triangle_l731_731674

theorem at_least_one_triangle {n : ℕ} (h1 : n ≥ 2) (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : 
(points.card = 2 * n) ∧ (segments.card = n^2 + 1) → 
∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ((a, b) ∈ segments ∨ (b, a) ∈ segments) ∧ ((b, c) ∈ segments ∨ (c, b) ∈ segments) ∧ ((c, a) ∈ segments ∨ (a, c) ∈ segments) := 
by 
  sorry

end at_least_one_triangle_l731_731674


namespace part1_part2_l731_731600

variable {a x y : ℝ} 

-- Conditions
def condition_1 (a x y : ℝ) := x - y = 1 + 3 * a
def condition_2 (a x y : ℝ) := x + y = -7 - a
def condition_3 (x : ℝ) := x ≤ 0
def condition_4 (y : ℝ) := y < 0

-- Part 1: Range for a
theorem part1 (a : ℝ) : 
  (∀ x y, condition_1 a x y ∧ condition_2 a x y ∧ condition_3 x ∧ condition_4 y → (-2 < a ∧ a ≤ 3)) :=
sorry

-- Part 2: Specific integer value for a
theorem part2 (a : ℝ) :
  (-2 < a ∧ a ≤ 3 → (∃ (x : ℝ), (2 * a + 1) * x > 2 * a + 1 ∧ x < 1) → a = -1) :=
sorry

end part1_part2_l731_731600


namespace average_marks_l731_731751

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l731_731751


namespace value_of_a_l731_731460

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_a (a : ℝ) (f_symmetric : ∀ x y : ℝ, y = f x ↔ -y = 2^(-x + a)) (sum_f_condition : f (-2) + f (-4) = 1) :
  a = 2 :=
sorry

end value_of_a_l731_731460


namespace probability_sum_less_than_product_l731_731792

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731792


namespace ring_stack_distance_l731_731489

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end ring_stack_distance_l731_731489


namespace determine_log_base_3_l731_731205

-- Defining conditions
variables (a b : ℝ)
variable h1 : log 10 (4 * a) + log 10 b = 2 * log 10 (a - 3 * b)
variable h2 : a > 3 * b
noncomputable def log_base_3 (x : ℝ) := (log 10 x) / (log 10 3)

-- The theorem to be proven
theorem determine_log_base_3 (h1 : log 10 (4 * a) + log 10 b = 2 * log 10 (a - 3 * b))
  (h2 : a > 3 * b) : log_base_3 (a / b) = 2 := 
sorry

end determine_log_base_3_l731_731205


namespace trigonometric_expression_identity_l731_731961

theorem trigonometric_expression_identity :
    3 * tan (Real.pi / 4) * cot (Real.pi / 3) + 
    2 * abs (sin (Real.pi / 6) - 1) - 
    (cot (Real.pi / 4)) / (tan (Real.pi / 3) + 2 * cos (Real.pi / 4)) = 
    1 + Real.sqrt 2 := 
by 
  sorry

end trigonometric_expression_identity_l731_731961


namespace find_k_l731_731563

-- We have an ellipse with the given equation and one of its foci.
def ellipse_focus_k (a : ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  (x ^ 2 / a ^ 2 + y ^ 2 / k = 1) ∧ (x = 0 ∧ y = sqrt 2)

-- Prove that for the given ellipse and focus, the value of k is 2.
theorem find_k (a : ℝ) (k : ℝ) (x y : ℝ) (h : ellipse_focus_k a k x y) : k = 2 :=
  sorry

end find_k_l731_731563


namespace closest_integer_to_cube_root_of_200_l731_731033

theorem closest_integer_to_cube_root_of_200 : 
  let n := 200 in
  let a := 5 in 
  let b := 6 in 
  abs (b - real.cbrt n) < abs (a - real.cbrt n) := 
by sorry

end closest_integer_to_cube_root_of_200_l731_731033


namespace incorrect_lifetime_calculation_l731_731727

-- Define expectation function
noncomputable def expectation (X : ℝ) : ℝ := sorry

-- We define the lifespans
variables (xi eta : ℝ)
-- Expected lifespan of the sensor and transmitter
axiom exp_xi : expectation xi = 3
axiom exp_eta : expectation eta = 5

-- Define the lifetime of the device
noncomputable def T := min xi eta

-- Given conditions
theorem incorrect_lifetime_calculation :
  expectation T ≤ 3 → 3 + (2 / 3) > 3 → false := 
sorry

end incorrect_lifetime_calculation_l731_731727


namespace sum_expr_le_e4_l731_731673

theorem sum_expr_le_e4
  (α β γ δ ε : ℝ) :
  (1 - α) * Real.exp α +
  (1 - β) * Real.exp (α + β) +
  (1 - γ) * Real.exp (α + β + γ) +
  (1 - δ) * Real.exp (α + β + γ + δ) +
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 :=
sorry

end sum_expr_le_e4_l731_731673


namespace incorrect_calculation_l731_731714

noncomputable def ξ : ℝ := 3 -- Expected lifetime of the sensor
noncomputable def η : ℝ := 5 -- Expected lifetime of the transmitter
noncomputable def T (ξ η : ℝ) : ℝ := min ξ η -- Lifetime of the entire device

theorem incorrect_calculation (h1 : E ξ = 3) (h2 : E η = 5) (h3 : E (min ξ η ) = 3.67) : False :=
by
  have h4 : E (min ξ η ) ≤ 3 := sorry -- Based on properties of expectation and min
  have h5 : 3.67 > 3 := by linarith -- Known inequality
  sorry

end incorrect_calculation_l731_731714


namespace who_hit_7_l731_731345

noncomputable def player_scores : Type :=
{ player_points : String → ℕ // 
    player_points "Alex" = 15 ∧
    player_points "Bobby" = 9 ∧
    player_points "Carla" = 13 ∧
    player_points "Diana" = 12 ∧
    player_points "Eric" = 14 ∧
    player_points "Fiona" = 17 }

theorem who_hit_7 (ps : player_scores) :
  ∃ (p : String), p = "Diana" ∧
  ∃ (n m : ℕ), n ≠ m ∧ n + m = ps.player_points p ∧ (n = 7 ∨ m = 7) := sorry

end who_hit_7_l731_731345


namespace length_of_MN_in_triangle_l731_731269

theorem length_of_MN_in_triangle (
  ABC : Type*
  (A B C M N : ABC)
  (distance_AB : ℝ)
  (distance_AC : ℝ)
  (is_midpoint_M : midpoint B C M)
  (angle_bisector_AN : angle_bisector A B C N)
  (perpendicular_BN_AN : ∠ B A N = 90)
  (AB_eq : distance_AB = 15)
  (AC_eq : distance_AC = 17)
  ) : distance M N = 1 := 
by sorry

end length_of_MN_in_triangle_l731_731269


namespace arcsin_sqrt3_div_2_l731_731135

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l731_731135


namespace increasing_sequence_solution_l731_731194

-- Given the conditions of the problem
def strictlyIncreasingSeq (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀(i j : ℕ), (i < j ∧ j ≤ n) → a i < a j

def gcdCondition (a1 a2 : ℕ) : Prop :=
  Nat.gcd a1 a2 = 1

def positiveIntegerCond (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ (m k : ℕ), (n ≥ m ∧ m ≥ 3 ∧ m ≥ k) → (a 1 + a 2 + a 3+ ... + a m) % a k = 0

-- Define the sequence a_i
def sequence (i : ℕ) : ℕ :=
  if i = 1 then 1
  else if i = 2 then 2
  else if i = 3 then 3
  else 3 * 2^(i-3)

-- Proof problem statement
theorem increasing_sequence_solution (n : ℕ) (hn : n ≥ 3) :
  strictlyIncreasingSeq n sequence ∧
  gcdCondition (sequence 1) (sequence 2) ∧
  positiveIntegerCond n sequence := by
  sorry

end increasing_sequence_solution_l731_731194


namespace staircase_markings_199_cells_l731_731856

-- Defining the recurrence relation for L_n
def L : ℕ → ℕ
| 0     := 2
| (n+1) := L n + 1

-- Prove the number of markings for a staircase with 199 cells
theorem staircase_markings_199_cells : L 199 = 200 :=
by
  sorry

end staircase_markings_199_cells_l731_731856


namespace problem_solution_l731_731555

noncomputable def vector_magnitudes_and_angle 
  (a b : ℝ) (angle_ab : ℝ) (norma normb : ℝ) (k : ℝ) : Prop :=
(a = 4 ∧ b = 8 ∧ angle_ab = 2 * Real.pi / 3 ∧ norma = 4 ∧ normb = 8) →
((norma^2 + normb^2 + 2 * norma * normb * Real.cos angle_ab = 48) ∧
  (16 * k - 32 * k + 16 - 128 = 0))

theorem problem_solution : vector_magnitudes_and_angle 4 8 (2 * Real.pi / 3) 4 8 (-7) := 
by 
  sorry

end problem_solution_l731_731555


namespace log_eight_of_five_twelve_l731_731990

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731990


namespace gemstones_needed_l731_731698

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l731_731698


namespace customer_count_l731_731091

theorem customer_count :
  let initial_customers := 13
  let customers_after_first_leave := initial_customers - 5
  let customers_after_new_arrival := customers_after_first_leave + 4
  let customers_after_group_join := customers_after_new_arrival + 8
  let final_customers := customers_after_group_join - 6
  final_customers = 14 :=
by
  sorry

end customer_count_l731_731091


namespace max_n_for_factorable_poly_l731_731145

/-- 
  Let p(x) = 6x^2 + n * x + 48 be a quadratic polynomial.
  We want to find the maximum value of n such that p(x) can be factored into
  the product of two linear factors with integer coefficients.
-/
theorem max_n_for_factorable_poly :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * B + A = n → A * B = 48) ∧ n = 289 := 
by
  sorry

end max_n_for_factorable_poly_l731_731145


namespace sum_of_valid_n_l731_731877

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l731_731877


namespace ab_le_one_l731_731208

theorem ab_le_one {a b : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 2) : ab ≤ 1 :=
by
  sorry

end ab_le_one_l731_731208


namespace tangent_line_at_2_tangent_line_passing_through_A_l731_731582

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem tangent_line_at_2 :
  tangent_line (f x) 2 = "x - y - 4 = 0" :=
sorry

theorem tangent_line_passing_through_A :
  tangent_line (f x) (x0 => f x0) 2 (-2) = "x - y - 4 = 0" ∨ "y + 2 = 0" :=
sorry

end tangent_line_at_2_tangent_line_passing_through_A_l731_731582


namespace problem_statement_l731_731299

def a := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5
def b := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5
def c := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5
def d := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5

theorem problem_statement :
  (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 = 80 / 361 :=
by
  sorry

end problem_statement_l731_731299


namespace find_m_l731_731676

/-
We define the vectors and the equation as conditions, and state the proof goal.
-/

theorem find_m (m : ℝ) (a b : ℝ × ℝ)
  (h_a : a = (m, 1))
  (h_b : b = (1, 2))
  (h_eq : ∥a + b∥ ^ 2 = ∥a∥ ^ 2 + ∥b∥ ^ 2) :
  m = -2 :=
sorry

end find_m_l731_731676


namespace area_R_correct_l731_731427

noncomputable def area_of_region_R : ℝ := sorry

theorem area_R_correct :
  let ABCD : rectangle := { length := 2, width := 2 },
      ABE : equilateral_triangle := { side := 2, vertex_interior := true },
      AD := side_ABCD(ABCD, A, D),
      strip := region_within_distance_interval(ABCD, AD, 2/3, 1) 
  in  area_of_region_R = 4 - Math.sqrt 3 / 6 :=
begin
  sorry
end

end area_R_correct_l731_731427


namespace minimum_parallelepipeds_for_cube_l731_731317

structure typical_parallelepiped :=
(length : ℕ) (width : ℕ) (height : ℕ)
(h_diff : length ≠ width ∧ width ≠ height ∧ height ≠ length)

def cube (s : ℕ) := (s, s, s)

noncomputable def minimum_typical_parallelepipeds (s : ℕ) :=
  if s = 0 then 0 else 4  -- Because for nonzero s, 4 is stated minimum

theorem minimum_parallelepipeds_for_cube (s : ℕ) (h : s > 0) : 
  minimum_typical_parallelepipeds(s) = 4 :=
by {
  sorry
}

end minimum_parallelepipeds_for_cube_l731_731317


namespace arcsin_sqrt_three_over_two_l731_731128

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l731_731128


namespace max_sum_of_factors_of_1764_l731_731616

theorem max_sum_of_factors_of_1764 :
  ∃ (a b : ℕ), a * b = 1764 ∧ a + b = 884 :=
by
  sorry

end max_sum_of_factors_of_1764_l731_731616


namespace sum_of_valid_n_l731_731869

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l731_731869


namespace min_value_g_l731_731576

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 
  sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ :=
  let φ₀ := φ + Real.pi / 6
  2 * Real.cos (2 * x + φ₀)

theorem min_value_g :
  ∀ φ : ℝ, (0 < φ ∧ φ < Real.pi) →
  (∀ x, (x, 0) = (-Real.pi/6, 0) → f x φ = 2 * Real.sin (2 * x + Real.pi/6)) →
  let g₀ := fun x => g (x - Real.pi/4) φ in
  ∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/6), 
    g₀ x = -1 :=
by
  sorry

end min_value_g_l731_731576


namespace distinct_prime_factors_l731_731610

def condition (M: ℕ) : Prop := log 2 (log 5 (log 7 (log 11 M))) = 7

theorem distinct_prime_factors (M : ℕ) (h : condition M) : ∃ p, p.prime ∧ (M = p ^ k for some k: ℕ) :=
by 
  sorry

end distinct_prime_factors_l731_731610


namespace average_age_decrease_l731_731392

theorem average_age_decrease (A : ℝ) :
  let original_total_age := 10 * A in
  let new_total_age := original_total_age - 45 + 15 in
  let original_average_age := A in
  let new_average_age := new_total_age / 10 in
  original_average_age - new_average_age = 3 :=
by
  let original_total_age := 10 * A
  let new_total_age := original_total_age - 45 + 15
  let original_average_age := A
  let new_average_age := new_total_age / 10
  calc
    original_average_age - new_average_age
        = A - ((10 * A - 45 + 15) / 10) : sorry
    ... = 3 : sorry

end average_age_decrease_l731_731392


namespace sum_of_numbers_l731_731413

theorem sum_of_numbers : 
  let S := {0.8, (1 / 2), 0.5} in 
  ∀ (x ∈ S), x ≤ 2 → ∑ x in S, x = 1.8 :=
by
  let S : set ℝ := {0.8, 0.5, 0.5}
  assume x hx
  have h1 : 0.8 ≤ 2 := by norm_num
  have h2 : 0.5 ≤ 2 := by norm_num
  have h3 : 0.5 ≤ 2 := by norm_num
  have : ∑ x in S, x = 1.8 := by norm_num
  exact this

end sum_of_numbers_l731_731413


namespace total_candidates_l731_731274

theorem total_candidates (T B G : ℕ) (H1 : G = 900)
                         (H2 : B = T - G)
                         (H3 : 0.34 * B + 0.32 * G = 0.331 * T)
                         (H4 : 0.66 * B + 0.68 * G = 0.669 * T) : T = 2000 :=
sorry

end total_candidates_l731_731274


namespace total_cost_after_discounts_and_cashback_l731_731150

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l731_731150


namespace g_correct_l731_731207

noncomputable def g (a : ℝ) : ℝ :=
  if 0 ≤ a ∧ a ≤ 2 then 3 else 2^a - 1

theorem g_correct (a : ℝ) (h1 : a ≥ 0) (h2 : {y | ∃ x, y = 2^|x| ∧ -2 ≤ x ∧ x ≤ a} = set.Icc (1 : ℝ) (2^a)) :
  g(a) = if 0 ≤ a ∧ a ≤ 2 then 3 else 2^a - 1 :=
by
  sorry

end g_correct_l731_731207


namespace sum_of_n_values_such_that_fraction_is_integer_l731_731888

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l731_731888


namespace classroom_scores_analysis_l731_731278

-- Define the conditions
def scores_first_grade_C : list ℕ := [85, 81, 88]
def scores_second_grade : list ℕ := [71, 76, 81, 82, 83, 86, 86, 88, 89, 90, 93, 95, 100, 100, 100]

-- Lean statement to prove the correct values of a, b, and c
theorem classroom_scores_analysis :
  let a := 85  -- median of first grade's scores in C
  let b := 100 -- mode of second grade's scores
  let c := 29  -- range of second grade's scores
  (a, b, c) = (85, 100, 29) :=
by {
  -- Proof would go here, replaced by sorry for now
  sorry
}

end classroom_scores_analysis_l731_731278


namespace charles_cleaning_time_l731_731499

theorem charles_cleaning_time :
  let Alice_time := 20
  let Bob_time := (3/4) * Alice_time
  let Charles_time := (2/3) * Bob_time
  Charles_time = 10 :=
by
  sorry

end charles_cleaning_time_l731_731499


namespace determine_k_values_l731_731974

noncomputable def quadratic_discriminant (a b c : ℂ) : ℂ :=
  b^2 - 4 * a * c

theorem determine_k_values :
  ∃ k : ℂ, 
  (∀ x : ℂ, (x ≠ 0 → (kx² + (5 * k - 2) * x + (6 * k - 5)) = 0) → k = 2 * Complex.I ∨ k = -2 * Complex.I) :=
begin
  sorry
end

end determine_k_values_l731_731974


namespace auditorium_shared_days_l731_731396

theorem auditorium_shared_days :
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  Nat.lcm (Nat.lcm drama_club_days choir_days) debate_team_days = 105 :=
by
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  sorry

end auditorium_shared_days_l731_731396


namespace charlie_has_32_cards_l731_731967

variable (Chris_cards Charlie_cards : ℕ)

def chris_has_18_cards : Chris_cards = 18 := sorry
def chris_has_14_fewer_cards_than_charlie : Chris_cards + 14 = Charlie_cards := sorry

theorem charlie_has_32_cards (h18 : Chris_cards = 18) (h14 : Chris_cards + 14 = Charlie_cards) : Charlie_cards = 32 := 
sorry

end charlie_has_32_cards_l731_731967


namespace arcsin_sqrt3_div_2_l731_731136

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l731_731136


namespace find_a_100_l731_731761

-- Define the sequence a_n recursively
def a : ℕ → ℕ
| 0       := 1  -- Note: Lean sequence typically starts from 0, for compatibility
| (n + 1) := a n + (2 * a n) / n

-- Define the statement we want to prove: a_100 = 5151
theorem find_a_100 : a 100 = 5151 :=
sorry

end find_a_100_l731_731761


namespace total_shoes_l731_731688

variable boots : ℕ
variable slippers : ℕ
variable heels : ℕ

-- Condition: Nancy has six pairs of boots
def boots_pairs : boots = 6 := rfl

-- Condition: Nancy has nine more pairs of slippers than boots
def slippers_pairs : slippers = boots + 9 := rfl

-- Condition: Nancy has a number of pairs of heels equal to three times the combined number of slippers and boots
def heels_pairs : heels = 3 * (boots + slippers) := by
  rw [boots_pairs, slippers_pairs]
  sorry  -- assuming the correctness of the consequent computation as rfl

-- Goal: Total number of individual shoes is 168
theorem total_shoes : (boots * 2) + (slippers * 2) + (heels * 2) = 168 := by
  rw [boots_pairs, slippers_pairs, heels_pairs]
  sorry  -- verifying the summing up to 168 as a proof

end total_shoes_l731_731688


namespace remainder_of_products_mod_7_l731_731174

theorem remainder_of_products_mod_7 :
  ((list.product [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]) *
   (list.product [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]))
   % 7 = 0 :=
by
  sorry

end remainder_of_products_mod_7_l731_731174


namespace range_of_m_l731_731587

noncomputable def f (x m : ℝ) : ℝ :=
  if m > 0 ∧ m ≠ 1 then log m ((4 * x ^ 2 + m) / x) else 0

theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : m ≠ 1) 
  (h_mono : ∀ x1 x2 : ℝ, 2 ≤ x1 → x1 ≤ x2 → x2 ≤ 3 → f x1 m ≤ f x2 m) :
  1 < m ∧ m ≤ 16 :=
  sorry

end range_of_m_l731_731587


namespace usual_time_to_cover_distance_l731_731006

theorem usual_time_to_cover_distance (S T : ℝ) (h1 : 0.75 * S = S / (T + 24)) (h2 : S * T = 0.75 * S * (T + 24)) : T = 72 :=
by
  sorry

end usual_time_to_cover_distance_l731_731006


namespace find_d_plus_f_l731_731411

-- Define complex numbers and conditions
variable (a b c d e f : ℝ)
variable (i : ℂ) [ComplexField i]

-- Given conditions
axiom h1 : b = 1
axiom h2 : e = -a - c
axiom h3 : (a + b * i) + (c + d * i) + (e + f * i) = -i

-- The theorem we want to prove
theorem find_d_plus_f : d + f = -2 :=
by
  sorry

end find_d_plus_f_l731_731411


namespace solve_inequality_l731_731373

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l731_731373


namespace goods_train_speed_l731_731075

-- Definitions based on given conditions
def V_man : ℝ := 50 -- kmph
def t_pass : ℝ := 9 / 3600 -- converted to hours
def d_goods : ℝ := 280 / 1000 -- converted to kilometers

-- The statement to prove the speed of the goods train
theorem goods_train_speed (V_g : ℝ) :
  V_g + V_man = d_goods / t_pass → V_g = 62 :=
by
  intro h
  sorry

end goods_train_speed_l731_731075


namespace sum_of_possible_values_of_x_l731_731762

theorem sum_of_possible_values_of_x : 
  (let x_values := [21 / 4 + Real.sqrt 145 / 4, 21 / 4 - Real.sqrt 145 / 4]
   ∑ x in x_values, x = 10.5) := by
  -- Conditions
  let side_square := (x - 3)
  let area_square := side_square ^ 2
  let length_rectangle := (x - 2)
  let width_rectangle := (x + 5)
  let area_rectangle := length_rectangle * width_rectangle
  have h : area_rectangle = 3 * area_square := by sorry

  -- We will use h to proceed with verifying the necessary solutions
  sorry

end sum_of_possible_values_of_x_l731_731762


namespace student_correct_answers_l731_731273

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 140) : C = 40 :=
by
  sorry

end student_correct_answers_l731_731273


namespace decreasing_interval_l731_731742

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem decreasing_interval : ∀ x : ℝ, x < -2 → (f'(x) < 0) := by
  sorry

end decreasing_interval_l731_731742


namespace middle_term_of_binomial_expansion_l731_731578

theorem middle_term_of_binomial_expansion (x : ℝ) :
  (∃ n : ℕ, (1 - 2 * x)^n = 2^n ∧ 2^n = 64) →
  (n = 6 → middle_term (1-2*x)^n = -160*x^3) :=
begin
  sorry
end

end middle_term_of_binomial_expansion_l731_731578


namespace gauss_lines_intersect_or_parallel_l731_731635

theorem gauss_lines_intersect_or_parallel
  (l₁ l₂ l₃ l₄ l₅ : Line)
  (G : set (set Line) × Line) : 
  (∀ l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 : Line, 
  ∃ P : Point, l1 ≠ l2 ∧ G = ({{l1, l2, l3, l4}, P}, {{l5, l6, l7, l8}, P}, {{l9, l10}, P})) →
  (∃ P : Point, ∀ Gi : Line, Gi ∈ G → P ∈ Gi) ∨ 
  (∀ Gi Gj : Line, Gi ∈ G → Gj ∈ G → Gi ≠ Gj → Gi ∥ Gj) :=
sorry

end gauss_lines_intersect_or_parallel_l731_731635


namespace non_union_women_percent_is_90_l731_731449

noncomputable def employees_men_percent : ℝ := 0.46
noncomputable def employees_union_percent : ℝ := 0.60
noncomputable def union_men_percent : ℝ := 0.70

theorem non_union_women_percent_is_90 :
  let total_employees := 100 in
  let men := employees_men_percent * total_employees in
  let union := employees_union_percent * total_employees in
  let union_men := union_men_percent * union in
  let non_union_men := men - union_men in
  let total_non_union := total_employees - union in
  let non_union_women := total_non_union - non_union_men in
  (non_union_women / total_non_union) * 100 = 90 :=
by sorry

end non_union_women_percent_is_90_l731_731449


namespace power_function_through_point_l731_731575

theorem power_function_through_point (k α : ℝ) (h : (λ x : ℝ, k * x^α) (1 / 2) = sqrt 2 / 2) : k + α = 3 / 2 :=
by
  sorry

end power_function_through_point_l731_731575


namespace sum_mod_9_l731_731959

theorem sum_mod_9 (n : ℕ) : 
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999 in
  s % 9 = n ∧ n < 9 ∧ 0 ≤ n ∧ (n % 2 = 0 ↔ (even n)) :=
by
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999
  suffices h0 : 24 % 9 = 6, from h0.symm
  have p : 2 % 9 + 33 % 9 + 444 % 9 + 5555 % 9 + 66666 % 9 + 777777 % 9 + 8888888 % 9 + 99999999 % 9 = 24 := 
    sorry
  have q : 6 < 9 := by norm_num
  have r : 0 ≤ 6 := by norm_num
  have m : 6 % 2 = 0 ↔ True := by norm_num
  exact ⟨24, q, r, m⟩

end sum_mod_9_l731_731959


namespace accounting_major_students_count_l731_731050

theorem accounting_major_students_count (p q r s: ℕ) (h1: p * q * r * s = 1365) (h2: 1 < p) (h3: p < q) (h4: q < r) (h5: r < s):
  p = 3 :=
sorry

end accounting_major_students_count_l731_731050


namespace alberts_total_earnings_l731_731498

theorem alberts_total_earnings :
  ∃ x1 y1 z1 x2 y2 z2 : ℝ,
  (1.45 * x1 = 550) ∧
  (1.38 * y1 = 400) ∧
  (1.30 * z1 = 350) ∧
  (x2 = x1 * 1.40) ∧
  (y2 = y1 * 1.33) ∧
  (z2 = z1 * 1.25) ∧
  (x2 + y2 + z2 = 1253.08) :=
by {
  exists (550 / 1.45),
  exists (400 / 1.38),
  exists (350 / 1.30),
  exists (550 / 1.45 * 1.40),
  exists (400 / 1.38 * 1.33),
  exists (350 / 1.30 * 1.25),
  simp [div_mul_cancel, div_eq_mul_inv],
  sorry,
}

end alberts_total_earnings_l731_731498


namespace sin_minus_cos_l731_731564

variable (A : ℝ) (h : sin A + cos A = sqrt 5 / 5)

theorem sin_minus_cos (A : ℝ) (h : sin A + cos A = sqrt 5 / 5) : sin A - cos A = 3 * sqrt 5 / 5 := by
  sorry

end sin_minus_cos_l731_731564


namespace shaded_areas_equal_l731_731528

/-- 
Given three squares I, II, and III:
- Square I is divided by its diagonals and lines connecting the midpoints of opposite sides into 4 triangles (\(\frac{1}{8}\) each) and a central square (\(\frac{1}{4}\)), with the central square shaded.
- Square II is divided by connecting midpoints of its sides into 4 smaller squares (\(\frac{1}{4}\) each), with two adjacent squares shaded, making the shaded area \(\frac{1}{2}\).
- Square III is divided by two perpendicular sets of parallel lines connecting midpoints of opposite sides into 16 smaller squares (\(\frac{1}{16}\) each), with the four corner squares shaded, making the shaded area \(\frac{1}{4}\).

Prove: Only the shaded areas of squares I and III are equal.
-/
theorem shaded_areas_equal :
  let I_shaded := 1 / 4,
      II_shaded := 1 / 2,
      III_shaded := 1 / 4 in
  I_shaded = III_shaded ∧ I_shaded ≠ II_shaded ∧ III_shaded ≠ II_shaded :=
by {
  sorry
}

end shaded_areas_equal_l731_731528


namespace no_solution_in_A_l731_731667

def A : Set ℕ := 
  {n | ∃ k : ℤ, abs (n * Real.sqrt 2022 - 1 / 3 - k) ≤ 1 / 2022}

theorem no_solution_in_A (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) : 
  20 * x + 21 * y ≠ 22 * z := 
sorry

end no_solution_in_A_l731_731667


namespace measure_angle_BAM_l731_731202

def isosceles_triangle (A B C : Type*) [DecidableEq (angle A B C)] (AB AC : ℝ) (ABC_angle : ℝ) :=
  ∃ (B C : pt), AB = AC ∧ ∠ABC = ABC_angle

def midpoint (A B C : Type*) [DecidableEq (seg A B C)] :=
  ∃ (A K C : pt), AC = CK

def angle_MAK_maximized (A K M : Type*) [DecidableEq (angle A K M)] :=
  ∃ (K M : pt), ∠MAK = 30°

noncomputable def problem_solution : ℝ :=
  44

theorem measure_angle_BAM (A B C K M : Type*) [DecidableEq (angle A K M)] :
  isosceles_triangle A B C 1 1 53 ∧
  midpoint A B C ∧
  angle_MAK_maximized A K M →
  measure_angle_BAM A B M = 44 :=
by
  sorry

end measure_angle_BAM_l731_731202


namespace abs_fraction_inequality_solution_l731_731349

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l731_731349


namespace closest_integer_to_cuberoot_of_200_l731_731026

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l731_731026


namespace trajectory_of_P_l731_731585

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 1)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1 - m

theorem trajectory_of_P (m P : ℝ × ℝ) (A B: ℝ × ℝ)
  (h1 : f A.1 = A.2) 
  (h2 : g m A.1 = A.2)
  (h3 : f B.1 = B.2)
  (h4 : g m B.1 = B.2)
  (h5 : |(A.1 - P.1, A.2 - P.2) + (B.1 - P.1, B.2 - P.2)| = 2) :
  (P.1 - 1)^2 + (P.2 - 1)^2 = 4 :=
sorry

end trajectory_of_P_l731_731585


namespace m_n_sum_l731_731934

noncomputable def find_m_n : ℚ × ℚ :=
let m := 1.4375
let n := 8.9375
(m, n)

theorem m_n_sum : Σ (m n : ℚ), m + n = 10.375 :=
⟨1.4375, 8.9375, by norm_num⟩

end m_n_sum_l731_731934


namespace ensure_three_buttons_of_same_color_l731_731930

theorem ensure_three_buttons_of_same_color (red white blue : ℕ) (total : ℕ) : 
  red = 203 → white = 117 → blue = 28 → total = red + white + blue → 
  ∃ buttons_taken : ℕ, buttons_taken = 7 ∧ (∀ red_buttons white_buttons blue_buttons, 
    red_buttons + white_buttons + blue_buttons = buttons_taken → 
    red_buttons ≥ 3 ∨ white_buttons ≥ 3 ∨ blue_buttons ≥ 3) :=
begin
  sorry
end

end ensure_three_buttons_of_same_color_l731_731930


namespace tangent_line_equation_at_P_l731_731144

noncomputable def f (x : ℝ) : ℝ := x^3 + 1
def P : ℝ × ℝ := (1, 2)

theorem tangent_line_equation_at_P : 
  let m := 3 * (1 : ℝ)^2 in 
  ∃ (y : ℝ → ℝ), (∀ x : ℝ, y x = m * (x - 1) + 2) ∧ (∀ x y, y - 2 = 3 * (x - 1) → 3 * x - y - 1 = 0) :=
begin
  let m := 3 * (1 : ℝ)^2,
  have slope_at_1 := m,
  use (λ x, slope_at_1 * (x - 1) + 2),
  split,
  { intros x,
    refl },
  { intros x y h,
    linarith only [h] }
end

end tangent_line_equation_at_P_l731_731144


namespace omega_value_l731_731583

theorem omega_value (ω : ℝ) (h1 : 0 < ω) (h2 : ω < π)
  (h3 : ∀ x : ℝ, 2 * sin (ω * (2 + x) - π / 3) = 2 * sin (ω * (2 - x) - π / 3)) :
  ω = 5 * π / 12 :=
begin
  sorry
end

end omega_value_l731_731583


namespace probability_sum_less_than_product_l731_731849

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731849


namespace parking_lot_wheels_l731_731527

-- Define the total number of wheels for each type of vehicle
def car_wheels (n : ℕ) : ℕ := n * 4
def motorcycle_wheels (n : ℕ) : ℕ := n * 2
def truck_wheels (n : ℕ) : ℕ := n * 6
def van_wheels (n : ℕ) : ℕ := n * 4

-- Number of each type of guests' vehicles
def num_cars : ℕ := 5
def num_motorcycles : ℕ := 4
def num_trucks : ℕ := 3
def num_vans : ℕ := 2

-- Number of parents' vehicles and their wheels
def parents_car_wheels : ℕ := 4
def parents_jeep_wheels : ℕ := 4

-- Summing up all the wheels
def total_wheels : ℕ :=
  car_wheels num_cars +
  motorcycle_wheels num_motorcycles +
  truck_wheels num_trucks +
  van_wheels num_vans +
  parents_car_wheels +
  parents_jeep_wheels

theorem parking_lot_wheels : total_wheels = 62 := by
  sorry

end parking_lot_wheels_l731_731527


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l731_731013

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (finset.sum (finset.range 9) (λ k, (binomial 8 k) * (x ^ k) * (y ^ (8 - k)))) = 56 :=
by sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l731_731013


namespace find_P_7_eq_8640_l731_731314

noncomputable def P (x : ℝ) : ℝ :=
  (x^5 - 15*x^4 + a*x^3 + b*x^2 + c*x + d) * (2*x^6 - 42*x^5 + e*x^4 + f*x^3 + g*x^2 + h*x + i)

theorem find_P_7_eq_8640 (a b c d e f g h i : ℝ) :
    (set_of_complex_roots(P) = {1, 2, 3, 4, 5, 6}) → 
    P(7) = 8640 :=
by
    sorry

end find_P_7_eq_8640_l731_731314


namespace round_robin_tournament_l731_731409

theorem round_robin_tournament (n k : ℕ) (h : (n-2) * (n-3) = 2 * 3^k): n = 5 :=
sorry

end round_robin_tournament_l731_731409


namespace invitational_tournament_l731_731784

theorem invitational_tournament (x : ℕ) (h : 2 * (x * (x - 1) / 2) = 56) : x = 8 :=
by
  sorry

end invitational_tournament_l731_731784


namespace probability_of_even_sum_is_two_thirds_l731_731182

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def choose_4_without_2 : ℕ := (Nat.factorial 11) / ((Nat.factorial 4) * (Nat.factorial 7))

noncomputable def choose_4_from_12 : ℕ := (Nat.factorial 12) / ((Nat.factorial 4) * (Nat.factorial 8))

noncomputable def probability_even_sum : ℚ := (choose_4_without_2 : ℚ) / (choose_4_from_12 : ℚ)

theorem probability_of_even_sum_is_two_thirds :
  probability_even_sum = (2 / 3 : ℚ) :=
sorry

end probability_of_even_sum_is_two_thirds_l731_731182


namespace no_factor_among_given_options_l731_731325

def polynomial := (y : ℝ) → y^4 - 4*y^2 + 16

theorem no_factor_among_given_options :
  ¬ ∃ (factor : ℝ → ℝ), 
     (factor = (λ y, y^2 + 4) ∨
      factor = (λ y, y + 2) ∨
      factor = (λ y, y^2 - 4) ∨
      factor = (λ y, y^2 + 2*y + 4)) ∧ 
     ∀ (y : ℝ), polynomial y = 0 → factor y = 0 :=
by
  sorry

end no_factor_among_given_options_l731_731325


namespace sum_of_n_l731_731873

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l731_731873


namespace expression_value_l731_731905

theorem expression_value : (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end expression_value_l731_731905


namespace closest_integer_to_cuberoot_of_200_l731_731027

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l731_731027


namespace domain_range_right_hand_limit_l731_731110

open Real

noncomputable def integral_eq (x y : ℝ) : Prop := 
  ∫ t in x..1, 1 / t ^ 2 = ∫ t in 1..y, 1 / t ^ 2

theorem domain_range_right_hand_limit :
  (∀ x y : ℝ, x < 1 → y > 1 → integral_eq x y → 
    (∃ a b c : set ℝ, (a = {x | x < 1 ∧ x ≠ 1 / 2} ∧ b = {y | y > 1} 
    ∧ (x ∈ a ∧ y ∈ b) → 
    (a =
    (λ x, ∀ x_n : ℝ, x_n < 1 → x_n ≠ 1 / 2 → (∃ y_n : ℝ, y_n > 1 ∧ integral_eq x_n y_n)) 
    ∧ b =
    (λ y, ∀ x_n : ℝ, x_n < 1 → x_n ≠ 1 / 2 → integral_eq x_n y → y > 1) 
    ∧ (∀ x : ℝ, lim (y, y → 1^-) = 1))) :=
sorry

end domain_range_right_hand_limit_l731_731110


namespace solution_days_l731_731421

variables (x y : ℝ)

def evaporated_portion_first (x : ℝ) : ℝ := 48 / x
def evaporated_portion_second (y : ℝ) : ℝ := 27 / y

def first_container_salt := 48
def second_container_salt := 27

def daily_evaporation_relation (x y : ℝ) := (48 / x) * y = (27 / y) * x

theorem solution_days (h1 : y = x - 6) (h2 : daily_evaporation_relation x y) : x = 24 ∧ y = 18 :=
by
  -- proof to be filled
  sorry

end solution_days_l731_731421


namespace dealer_gross_profit_l731_731445

theorem dealer_gross_profit (P S G : ℝ) (hP : P = 150) (markup : S = P + 0.5 * S) :
  G = S - P → G = 150 :=
by
  sorry

end dealer_gross_profit_l731_731445


namespace irreducible_f_when_n_geq_4_l731_731335

noncomputable def f (x : ℤ) (n : ℕ) : ℤ := x^n + x^3 + x^2 + x + 5

-- The main theorem statement
theorem irreducible_f_when_n_geq_4 (n : ℕ) (h : n ≥ 4) : 
  ¬ ∃ g h : ℤ[X], g.degree ≥ 1 ∧ h.degree ≥ 1 ∧ (f g * f h = f x n) := sorry

end irreducible_f_when_n_geq_4_l731_731335


namespace sum_not_arp_l731_731693

theorem sum_not_arp (a d b q : ℝ) (q_ne_one : q ≠ 1) :
  ¬ (∀ c_n : ℕ → ℝ, ∃ (a d : ℝ), ∀ (n : ℕ), (c n = a + (n - 1) * d)
  ∧ c n = (a + (n - 1) * d) + b * q^(n - 1)) :=
sorry

end sum_not_arp_l731_731693


namespace maximize_S_n_proof_l731_731229

-- Define the values a_1 and d as given in the problem
def a_1 : ℝ := 5
def d : ℝ := -(5 / 7)

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) : ℝ :=
  a_1 * n + (n * (n - 1) / 2) * d

-- Define the maximization condition
def maximize_S_n : Prop :=
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ ∀ m : ℕ, S_n m ≤ S_n n

-- Statement asserting that 7 or 8 maximizes S_n
theorem maximize_S_n_proof : maximize_S_n :=
  sorry

end maximize_S_n_proof_l731_731229


namespace intersection_M_S_l731_731598

def M := {x : ℕ | 0 < x ∧ x < 4 }

def S : Set ℕ := {2, 3, 5}

theorem intersection_M_S : (M ∩ S) = {2, 3} := by
  sorry

end intersection_M_S_l731_731598


namespace smallest_positive_period_of_f_range_of_f_on_interval_l731_731581

def f (x : ℝ) : ℝ := sqrt 3 * (sin x ^ 2 - cos x ^ 2) + 2 * sin x * cos x

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ p) := sorry

theorem range_of_f_on_interval :
  ∀ x ∈ Icc(0, π / 3), -sqrt 3 ≤ f x ∧ f x ≤ sqrt 3 := sorry

end smallest_positive_period_of_f_range_of_f_on_interval_l731_731581


namespace log_base_8_of_512_l731_731999

theorem log_base_8_of_512 : log 8 512 = 3 := by
  have h₁ : 8 = 2^3 := by rfl
  have h₂ : 512 = 2^9 := by rfl
  rw [h₂, h₁]
  sorry

end log_base_8_of_512_l731_731999


namespace sum_of_valid_n_l731_731882

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l731_731882


namespace sum_of_integer_n_l731_731895

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l731_731895


namespace div_a2_plus_2_congr_mod8_l731_731301

variable (a d : ℤ)
variable (h_odd : a % 2 = 1)
variable (h_pos : a > 0)

theorem div_a2_plus_2_congr_mod8 :
  (d ∣ (a ^ 2 + 2)) → (d % 8 = 1 ∨ d % 8 = 3) :=
by
  sorry

end div_a2_plus_2_congr_mod8_l731_731301


namespace units_digit_7_pow_5_eq_7_l731_731432

theorem units_digit_7_pow_5_eq_7 : (7^5) % 10 = 7 := by
  -- Definitions based on problem conditions
  have h1 : (7^1) % 10 = 7 := by norm_num
  have h2 : (7^2) % 10 = 9 := by norm_num
  have h3 : (7^3) % 10 = 3 := by norm_num
  have h4 : (7^4) % 10 = 1 := by norm_num

  -- Leverage the cyclical pattern given the reminder of 5 mod 4
  have cycle : ∀ n, (7^(4*n + 1)) % 10 = 7 := by
    intro n
    sorry -- Proof of cyclical pattern

  -- Use the cycle pattern to conclude the units digit of 7^5
  show (7^5) % 10 = 7 from cycle 1 

end units_digit_7_pow_5_eq_7_l731_731432


namespace log_sum_l731_731260

theorem log_sum (x y z : ℕ) 
  (h1 : log 4 (log 3 (log 2 x)) = 0)
  (h2 : log 2 (log 4 (log 3 y)) = 0)
  (h3 : log 3 (log 2 (log 4 z)) = 0) :
  x + y + z = 105 :=
sorry

end log_sum_l731_731260


namespace fraction_of_sophomores_attending_fair_l731_731953

theorem fraction_of_sophomores_attending_fair
  (s j n : ℕ)
  (h1 : s = j)
  (h2 : j = n)
  (soph_attend : ℚ)
  (junior_attend : ℚ)
  (senior_attend : ℚ)
  (fraction_s : soph_attend = 4/5 * s)
  (fraction_j : junior_attend = 3/4 * j)
  (fraction_n : senior_attend = 1/3 * n) :
  soph_attend / (soph_attend + junior_attend + senior_attend) = 240 / 565 :=
by
  sorry

end fraction_of_sophomores_attending_fair_l731_731953


namespace gemstones_needed_l731_731694

-- Define the initial quantities and relationships
def magnets_per_earring := 2
def buttons_per_magnet := 1 / 2
def gemstones_per_button := 3
def earrings_per_set := 2
def sets_of_earrings := 4

-- Define the total gemstones needed
theorem gemstones_needed : 
    let earrings := sets_of_earrings * earrings_per_set in
    let total_magnets := earrings * magnets_per_earring in
    let total_buttons := total_magnets * buttons_per_magnet in
    let total_gemstones := total_buttons * gemstones_per_button in
    total_gemstones = 24 :=
by
    have earrings := 2 * 4
    have total_magnets := earrings * 2
    have total_buttons := total_magnets / 2
    have total_gemstones := total_buttons * 3
    exact eq.refl 24

end gemstones_needed_l731_731694


namespace f_has_infinitely_many_extreme_points_l731_731236

def f (x : ℝ) : ℝ := x^2 - 2 * x * Real.cos x

theorem f_has_infinitely_many_extreme_points :
  ∃ (ext_points : ℕ → ℝ), ∀ n : ℕ, is_extreme_point f (ext_points n) := sorry

end f_has_infinitely_many_extreme_points_l731_731236


namespace problem_intersection_point_l731_731915

open Real
open EuclideanGeometry

noncomputable def proofProblem :=
  let ABC := triangle
  let A  := point
  let B  := point
  let C  := point
  
  let C1 := foot_of_perpendicular A B C
  let B1 := foot_of_perpendicular A C B
  let A0 := midpoint B C
  let A1 := foot_of_perpendicular A B C
  
  let PQ  := line_through A (parallel_line B C)
  let P   := intersection PQ C1
  let Q   := intersection PQ B1
  
  let K := intersection (line_through A0 C1) PQ
  let L := intersection (line_through A0 B1) PQ
  
  -- Circumcircles of triangles PQA1, KLA0, A1B1C1, and the circle with diameter AA1 intersect at T
  let omega1 := circumscribed_circle P Q A1
  let omega2 := circumscribed_circle K L A0
  let omega3 := circumscribed_circle A1 B1 C1
  let omega4 := circumscribed_circle_on_diameter A A1

  ∃ T : point, T ∈ omega1.circle ∧ T ∈ omega2.circle ∧ T ∈ omega3.circle ∧ T ∈ omega4.circle

theorem problem_intersection_point : ∃ (T : point), ∃ ω1 ω2 ω3 ω4 (circle T), 
    (T ∈ ω1 ∧ T ∈ ω2 ∧ T ∈ ω3 ∧ T ∈ ω4) :=
by
  sorry

end problem_intersection_point_l731_731915


namespace training_days_l731_731662

theorem training_days (d : ℕ) (h : 5 * d + 10 * d + 20 * d = 1050) : d = 30 :=
by
  calc
    35 * d = 1050 : by simp [*, add_mul]
    d = 30 : by sorry

end training_days_l731_731662


namespace number_of_inverses_modulo_12_l731_731252

def has_inverse_mod_12 (n : ℤ) : Prop :=
  ∃ m : ℤ, (n * m) % 12 = 1

def count_inverses_mod_12 (a b : ℤ) : ℕ :=
  Set.card {n | a ≤ n ∧ n ≤ b ∧ Nat.gcd n 12 = 1}

theorem number_of_inverses_modulo_12 : count_inverses_mod_12 0 11 = 4 := sorry

end number_of_inverses_modulo_12_l731_731252


namespace solve_inequality_l731_731371

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731371


namespace part1_part2_l731_731291

def a (n : ℕ) : ℕ :=
  if h : n > 0 then (2^(n-1)) * n else 0

def b (n : ℕ) : ℕ :=
  if h : n > 0 then 2^(n-1) else 0

def c (k : ℕ) : ℕ :=
  if h : k > 0 then (3 / 2) * k * 2^(k-1) else 0

def T (n : ℕ) : ℕ :=
  ∑ i in (range n).filter (λ x, x > 0), c i

theorem part1 (n : ℕ) (h : n > 0) : 
  a 1 = 1 ∧ (∀ n > 0, a (n+1) = 2 * (n + 1) / n * a n) → a n = n * 2^(n-1) := by
  sorry

theorem part2 (n : ℕ) : 
  T n = (3 / 2) * ((n-1) * 2^n + 1) := by
  sorry

end part1_part2_l731_731291


namespace abs_inequality_l731_731363

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l731_731363


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l731_731132

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l731_731132


namespace max_is_x2_l731_731043

noncomputable def x1 := 7.45678
noncomputable def x2 := Real.ofRat (745677 / 100000)
noncomputable def x3 := Real.ofRat 745 / 100 + (Real.ofRat 67 / 10000) / (1 - Real.ofRat 1 / 100)
noncomputable def x4 := Real.ofRat 74 / 10 + (Real.ofRat 567 / 10000) / (1 - Real.ofRat 1 / 1000)
noncomputable def x5 := (Real.ofRat 4567 / 10000) / (1 - Real.ofRat 1 / 10000)

theorem max_is_x2 :
  max x1 (max x2 (max x3 (max x4 x5))) = x2 := by sorry

end max_is_x2_l731_731043


namespace volume_of_solid_rotation_l731_731118

noncomputable def volume_of_solid : ℝ :=
  π * (∫ y in 0..1, (4 - (1 + sqrt y) ^ 2) - (1 - sqrt y) ^ 2) y

theorem volume_of_solid_rotation :
  volume_of_solid = 4 * π / 3 := by
  sorry

end volume_of_solid_rotation_l731_731118


namespace inequality_proof_l731_731342

variable {R : Type} [LinearOrderedField R]

theorem inequality_proof (x y z : R) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x^4 + y^4 + z^2 ≥ x * y * z * (Real.sqrt 8) :=
by
  sorry

end inequality_proof_l731_731342


namespace sum_lt_prod_probability_l731_731822

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731822


namespace max_axes_of_symmetry_of_three_lines_l731_731860

-- Define the conditions and the main problem
theorem max_axes_of_symmetry_of_three_lines :
  ∀ (l₁ l₂ l₃ : ℝ × ℝ × ℝ → Prop),
    (¬ ∃ v₁ v₂ : ℝ × ℝ × ℝ, l₁ v₁ ∧ l₂ v₂ ∧ (v₁ = v₂ ∨ l₁ = l₂)) → -- No two lines are coincident
    (¬ ∃ d₁ d₂ : ℝ × ℝ × ℝ, l₁ d₁ ∧ l₂ d₂ ∧ ∃ k : ℝ, d₁ = k • d₂) → -- No two lines are parallel
    (¬ ∃ v₃ : ℝ × ℝ × ℝ, l₃ v₃) →                                     -- The third line is also non-coincident with the first two
    (∀ (p : ℝ × ℝ × ℝ), (∃ v₁ v₂ v₃ : ℝ × ℝ × ℝ, l₁ v₁ ∧ l₂ v₂ ∧ l₃ v₃ ∧ p = v₁ ∨ p = v₂ ∨ p = v₃) → 
    number_of_axes_of_symmetry ({l₁, l₂, l₃} : set (ℝ × ℝ × ℝ → Prop)) ≤ 9) := 
sorry

end max_axes_of_symmetry_of_three_lines_l731_731860


namespace root_in_interval_l731_731390

noncomputable def f (x : ℝ) : ℝ := log x / log 4 + x - 7

theorem root_in_interval : ∃ x ∈ Ioo 5 6, f x = 0 := by
  have : f 5 * f 6 < 0 := by sorry
  apply IntermediateValueTheorem
  exact this
  sorry

end root_in_interval_l731_731390


namespace annulus_area_l731_731980

theorem annulus_area (B C RW : ℝ) (h1 : B > C)
  (h2 : B^2 - (C + 5)^2 = RW^2) : 
  π * RW^2 = π * (B^2 - (C + 5)^2) :=
by
  sorry

end annulus_area_l731_731980


namespace intersection_A_B_l731_731315

def setA : Set ℝ := {x | x^2 - 1 > 0}
def setB : Set ℝ := {x | Real.log x / Real.log 2 < 1}

theorem intersection_A_B :
  {x | x ∈ setA ∧ x ∈ setB} = {x | 1 < x ∧ x < 2} :=
sorry

end intersection_A_B_l731_731315


namespace ZacharysBusRideLength_l731_731005

theorem ZacharysBusRideLength (vince_ride zach_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : vince_ride = zach_ride + 0.125) : 
  zach_ride = 0.500 := 
by
  sorry

end ZacharysBusRideLength_l731_731005


namespace socks_purchase_l731_731329

theorem socks_purchase :
  ∃ (x y z : ℕ), (x + y + z = 15) ∧ (2 * x + 3 * y + 5 * z = 45) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ (z ≥ 1) ∧ (x = 6) :=
by {
  sorry,
}

end socks_purchase_l731_731329


namespace impossible_to_cover_checkerboard_with_T_tetrominoes_l731_731276

theorem impossible_to_cover_checkerboard_with_T_tetrominoes :
  ∀ (checkerboard : (ℕ × ℕ) → Prop)
  (tetromino : (ℕ × ℕ) → Prop),
  (∀ i j, i < 10 ∧ j < 10 → checkerboard (i, j)) →
  (∀ i j, tetromino (i, j) ↔ ((i + j) % 2 = 0)) →
   (∀ t : ℕ, t = 25 → length (filter (λ t, tetromino t) (all_checkerboard_cells checkerboard)) = 100) →
   ∃ n b, 3 * n + b = 50 ∧ n + 3 * b = 50 → False := 
sorry

end impossible_to_cover_checkerboard_with_T_tetrominoes_l731_731276


namespace discount_double_time_l731_731621

theorem discount_double_time (TD FV : ℝ) (h1 : TD = 10) (h2 : FV = 110) : 
  2 * TD = 20 :=
by
  sorry

end discount_double_time_l731_731621


namespace six_people_arrangement_l731_731772

theorem six_people_arrangement (A B C D E F : Type) :
  let arrangements := {l : List (A ⊕ B ⊕ C ⊕ D ⊕ E ⊕ F) // l.length = 6 ∧ l.head ≠ A ∧ l.getLast ≠ A},
  arrangements.card = 480 :=
by
  sorry

end six_people_arrangement_l731_731772


namespace time_spent_on_type_a_problems_l731_731448

-- Define the conditions
def total_questions := 200
def examination_duration_hours := 3
def type_a_problems := 100
def type_b_problems := total_questions - type_a_problems
def type_a_time_coeff := 2

-- Convert examination duration to minutes
def examination_duration_minutes := examination_duration_hours * 60

-- Variables for time per problem
variable (x : ℝ)

-- The total time spent
def total_time_spent : ℝ := type_a_problems * (type_a_time_coeff * x) + type_b_problems * x

-- Statement we need to prove
theorem time_spent_on_type_a_problems :
  total_time_spent x = examination_duration_minutes → type_a_problems * (type_a_time_coeff * x) = 120 :=
by
  sorry

end time_spent_on_type_a_problems_l731_731448


namespace nancy_shoes_l731_731681

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l731_731681


namespace arcsin_sqrt_three_over_two_l731_731129

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l731_731129


namespace total_cost_paper_plates_and_cups_l731_731768

theorem total_cost_paper_plates_and_cups :
  ∀ (P C : ℝ), (20 * P + 40 * C = 1.20) → (100 * P + 200 * C = 6.00) := by
  intros P C h
  sorry

end total_cost_paper_plates_and_cups_l731_731768


namespace collinear_intersection_points_l731_731158

-- Define points and cyclic hexagon
variables {A B C A' B' C' Q X Y Z : Type}

-- Assumptions regarding intersections and cyclic nature
variables [cyclic : CyclicHexagon C B A A' Q C']
variables [intersect_X : IntersectingPoint QA' BC X]
variables [intersect_Y : IntersectingPoint QB' CA Y]
variables [intersect_Z : IntersectingPoint QC' AB Z]

-- Statement of collinearity using Pascal's Theorem
theorem collinear_intersection_points :
  collinear X Y Z :=
by sorry

end collinear_intersection_points_l731_731158


namespace solve_inequality_l731_731372

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731372


namespace largest_valid_number_l731_731010

/-
Problem: 
What is the largest number, all of whose digits are 3, 2, or 4 whose digits add up to 16?

We prove that 4432 is the largest such number.
-/

def digits := [3, 2, 4]

def sum_of_digits (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 2 ∨ d = 4

def generate_number (l : List ℕ) : ℕ :=
  l.foldl (λ acc d => acc * 10 + d) 0

theorem largest_valid_number : 
  ∃ l : List ℕ, (∀ d ∈ l, is_valid_digit d) ∧ sum_of_digits l = 16 ∧ generate_number l = 4432 :=
  sorry

end largest_valid_number_l731_731010


namespace find_Q_l731_731093

theorem find_Q
  (roots : List ℕ)  -- roots of the polynomial, given as a list of positive integers
  (h_roots : ∀ r ∈ roots, 0 < r)  -- all roots are positive integers
  (h_sum_roots : roots.sum = 15)  -- the sum of roots is 15
  (h_product_roots : roots.prod = 64)  -- the product of roots is 64
  : (∑ i in Finset.univ.filter (λ i, ∃ r1 r2 r3 : ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r1 ∧ r1 * r2 * r3 = i), i) = 45 :=
by
  sorry

end find_Q_l731_731093


namespace fraction_simplification_l731_731436

theorem fraction_simplification :
    1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end fraction_simplification_l731_731436


namespace alok_total_payment_l731_731095

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l731_731095


namespace part_1_part_2_part_3_l731_731307

variable {m : ℝ}
variable {x y : ℝ}

def ellipse_Γ (m : ℝ) := ∀ (x y : ℝ), (m > 0) → (x^2 / (3 * m) + y^2 / m = 1)
def hyperbola_C (m : ℝ) := ∀ (x y : ℝ), (m > 0) → (m^2 * x^2 - y^2 = m^2)
def line_l (b : ℝ) (x y : ℝ) := (y = x + b)
def through_vertex (x y : ℝ) := (x = 1) ∧ (y = 0)
def symmetric_about (A B : ℝ × ℝ) (l : ℝ → ℝ) := ∀ (p q : ℝ), (p, q ∈ l) → true
-- Note: These functions can be expanded as necessary to fit more precise criteria given the problem

noncomputable def lean_problem_part_1 : Prop :=
  ∀ (x y : ℝ), (m = 1) → ellipse_Γ m ∧ hyperbola_C m

noncomputable def lean_problem_part_2 : Prop :=
  ∀ (k1 k2 : ℝ), (k1 * k2 = -1) → (inclination_angle PQ = 0)

noncomputable def lean_problem_part_3 : Prop :=
  ∀ (b : ℝ), symmetric_about A B line_l ∧ (9 < 4 * ⟦TA⟧ * ⟦TB⟧ < 10) → (-1/2 < b < -1/4 ∨ -1/4 < b < 0)

theorem part_1 : lean_problem_part_1 := sorry

theorem part_2 : lean_problem_part_2 := sorry

theorem part_3 : lean_problem_part_3 := sorry

end part_1_part_2_part_3_l731_731307


namespace proportional_division_middle_part_l731_731259

theorem proportional_division_middle_part : 
  ∃ x : ℕ, x = 8 ∧ 5 * x = 40 ∧ 3 * x + 5 * x + 7 * x = 120 := 
by
  sorry

end proportional_division_middle_part_l731_731259


namespace sequence_general_term_l731_731239

/-- Given the sequence {a_n} defined by a_n = 2^n * a_{n-1} for n > 1 and a_1 = 1,
    prove that the general term a_n = 2^((n^2 + n - 2) / 2) -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n > 1, a n = 2^n * a (n-1)) :
  ∀ n, a n = 2^((n^2 + n - 2) / 2) :=
sorry

end sequence_general_term_l731_731239


namespace tiles_needed_to_cover_floor_l731_731661

-- Definitions of the conditions
def room_length : ℕ := 2
def room_width : ℕ := 12
def tile_area : ℕ := 4

-- The proof statement: calculate the number of tiles needed to cover the entire floor
theorem tiles_needed_to_cover_floor : 
  (room_length * room_width) / tile_area = 6 := 
by 
  sorry

end tiles_needed_to_cover_floor_l731_731661


namespace equal_lengths_imply_equal_segments_l731_731454

theorem equal_lengths_imply_equal_segments 
  (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h₁ : a₁ = a₂) 
  (h₂ : b₁ = b₂) : 
  x = y := 
sorry

end equal_lengths_imply_equal_segments_l731_731454


namespace repeating_decimal_sum_1144_l731_731147

def sum_numerator_denominator_repeating_decimal (x : ℚ) : ℕ := 
  let n := x.num
  let d := x.denom
  n + d

theorem repeating_decimal_sum_1144 : sum_numerator_denominator_repeating_decimal (145 / 999) = 1144 := by 
  sorry

end repeating_decimal_sum_1144_l731_731147


namespace p_sufficient_not_necessary_l731_731554

theorem p_sufficient_not_necessary:
  (∀ a b : ℝ, a > b ∧ b > 0 → (1 / a^2 < 1 / b^2)) ∧ 
  (∃ a b : ℝ, (1 / a^2 < 1 / b^2) ∧ ¬ (a > b ∧ b > 0)) :=
sorry

end p_sufficient_not_necessary_l731_731554


namespace perpendicular_lines_slope_l731_731625

theorem perpendicular_lines_slope (a : ℝ) :
  let k1 := -a / 2,
      k2 := -1
  in (k1 * k2 = -1) → a = -2 :=
by
  intros k1 k2 hp
  sorry

end perpendicular_lines_slope_l731_731625


namespace conditional_probability_l731_731016

-- Define the events and the probability space
def Ω := {TT, TH, HT, HH}

def P : Ω → ℝ 
| TT := 1 / 4
| TH := 1 / 4
| HT := 1 / 4
| HH := 1 / 4

def A : set Ω := {HT, HH} -- First appearance of heads
def B : set Ω := {TH, HT} -- Second appearance of tails
def AB : set Ω := {HT}    -- Both events A and B occurring

-- Probability of event A
def P_A := ∑ ω in A, P ω

-- Probability of events A and B
def P_AB := ∑ ω in AB, P ω

-- Conditional probability
def P_B_given_A := P_AB / P_A

theorem conditional_probability :
  P_B_given_A = 1 / 2 :=
by
  unfold P_B_given_A P_AB P_A A B AB P
  simp
  sorry

end conditional_probability_l731_731016


namespace average_marks_l731_731746

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l731_731746


namespace hamburger_combinations_l731_731246

theorem hamburger_combinations : 
  let condiments := 10  -- Number of available condiments
  let patty_choices := 4 -- Number of meat patty options
  2^condiments * patty_choices = 4096 :=
by sorry

end hamburger_combinations_l731_731246


namespace closest_integer_to_cuberoot_of_200_l731_731019

theorem closest_integer_to_cuberoot_of_200 :
  ∃ n : ℤ, (n = 6) ∧ ( |200 - n^3| ≤ |200 - m^3|  ) ∀ m : ℤ := sorry

end closest_integer_to_cuberoot_of_200_l731_731019


namespace max_cross_section_of_regular_tetrahedron_l731_731523

theorem max_cross_section_of_regular_tetrahedron (a : ℝ) (h_a : a > 0) :
  ∃ t : ℝ, t = (a^2) / 4 :=
begin
  sorry
end

end max_cross_section_of_regular_tetrahedron_l731_731523


namespace area_triangle_DEF_l731_731429

-- Define points D, E, and F
def D := (-5, 3 : ℤ × ℤ)
def E := (9, 3 : ℤ × ℤ)
def F := (6, -6 : ℤ × ℤ)

-- Function to calculate the area of a triangle given three points
def triangle_area (A B C : ℤ × ℤ) : ℤ := 
  (B.fst - A.fst) * (C.snd - A.snd) - (B.snd - A.snd) * (C.fst - A.fst)

-- Note that the area calculation here is two times the actual area
-- because we do not divide by 2 as it's often done in triangle area formula
-- for verifying the result, we need to check if the absolute value divided by 2
-- gives us the correct result. As such, we need to prove that abs(area) divided by 2 equals 63.
theorem area_triangle_DEF : (abs (triangle_area D E F) / 2) = 63 := by
  sorry

end area_triangle_DEF_l731_731429


namespace odd_sum_probability_l731_731534

-- Initialize the problem conditions
def fifteen_tiles : List ℕ := [1, 2, 3, ..., 15] -- list of 15 tiles
def odd_tiles : List ℕ := [1, 3, 5, 7, 9, 11, 13, 15] -- list of odd tiles
def even_tiles : List ℕ := [2, 4, 6, 8, 10, 12, 14] -- list of even tiles
def players : ℕ := 5
def tiles_per_player : ℕ := 3

theorem odd_sum_probability :
  let total_ways := (binom 15 3) * (binom 12 3) * (binom 9 3) * (binom 6 3) * (binom 3 3),
      odd_ways := (binom 8 5) * 5! * (binom 7 2) * (binom 5 2) * (binom 3 2) * (binom 1 2),
      prob := odd_ways / total_ways in
  let frac := prob.to_rational in
  frac.num + frac.denom = 19499 := 
by sorry


end odd_sum_probability_l731_731534


namespace similarity_coefficient_interval_l731_731540

-- Definitions
def similarTriangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

-- Theorem statement
theorem similarity_coefficient_interval (x y z p k : ℝ) (h_sim : similarTriangles x y z p) :
  0 ≤ k ∧ k ≤ 2 :=
sorry

end similarity_coefficient_interval_l731_731540


namespace average_of_remaining_numbers_l731_731737

theorem average_of_remaining_numbers (T : ℕ) (a b : ℕ) (n m : ℕ) (h1 : n = 12) (h2 : T = 90 * n) (h3 : a = 80) (h4 : b = 85) : 
  (T - a - b) / (n - 2) = 91.5 :=
by
  sorry

end average_of_remaining_numbers_l731_731737


namespace probability_sum_less_than_product_l731_731793

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l731_731793


namespace A_neq_R_l731_731313

open Nat

noncomputable def distinct_primes_prod (k : ℕ) (hk : k ≥ 2) (p : fin k → ℕ) 
  (hp : ∀ i, prime (p i) ∧ ∀ j, i ≠ j → p i ≠ p j) : ℕ :=
∏ i, p i

noncomputable def set_A (N : ℕ) : set ℕ :=
{a | a < N ∧ gcd a N ≠ 1}

noncomputable def set_R (N : ℕ) (A : set ℕ) : set ℕ :=
{r | r < N ∧ ∃ a b ∈ A, b ∈ permutation A ∧ r ≡ a * b [MOD N]}

theorem A_neq_R
  (k : ℕ) (hk : k ≥ 2) 
  (p : fin k → ℕ) 
  (hp : ∀ i, prime (p i) ∧ ∀ j, i ≠ j → p i ≠ p j) 
  (N : ℕ := distinct_primes_prod k hk p hp)
  (A : set ℕ := set_A N)
  (R : set ℕ := set_R N A) : 
  A ≠ R :=
sorry

end A_neq_R_l731_731313


namespace log_eight_of_five_twelve_l731_731988

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731988


namespace triangles_acute_obtuse_l731_731263

variables {A1 B1 C1 A2 B2 C2 : ℝ}

theorem triangles_acute_obtuse
  (h1 : cos A1 = sin A2)
  (h2 : cos B1 = sin B2)
  (h3 : cos C1 = sin C2)
  (h_A1 : 0 < A1) (h_A1_lt_pi : A1 < π)
  (h_B1 : 0 < B1) (h_B1_lt_pi : B1 < π)
  (h_C1 : 0 < C1) (h_C1_lt_pi : C1 < π)
  (h_A2 : 0 < A2) (h_A2_lt_pi : A2 < π)
  (h_B2 : 0 < B2) (h_B2_lt_pi : B2 < π)
  (h_C2 : 0 < C2) (h_C2_lt_pi : C2 < π)
  (sum_A1B1C1 : A1 + B1 + C1 = π)
  (sum_A2B2C2 : A2 + B2 + C2 = π) :
  (A1 < π / 2) ∧ (B1 < π / 2) ∧ (C1 < π / 2) ∧ ((A2 > π / 2) ∨ (B2 > π / 2) ∨ (C2 > π / 2)) ∧ (A2 < π) ∧ (B2 < π) ∧ (C2 < π) ∧ (A2 > 0) ∧ (B2 > 0) ∧ (C2 > 0) :=
by
  sorry

end triangles_acute_obtuse_l731_731263


namespace simplify_expression_l731_731164

variable (y : ℝ)
variable (h : y ≠ 0)

theorem simplify_expression : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry

end simplify_expression_l731_731164


namespace fraction_power_multiplication_l731_731165

theorem fraction_power_multiplication :
  (1/3)^9 * (2/5)^(-4) = 625 / 314928 := by
  sorry

end fraction_power_multiplication_l731_731165


namespace find_positive_number_a_l731_731201

theorem find_positive_number_a 
  (f : ℝ → ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_domain : ∀ x, abs (x + 2 - a) < a) 
  (h_positive : a > 0) : 
  a = 2 :=
sorry

end find_positive_number_a_l731_731201


namespace sum_first_n_reciprocal_to_T_l731_731670

noncomputable def a1 := 2

def S (n : ℕ) (d : ℤ) : ℤ := (n * (n + 1)) / 2 * d + n * a1

theorem sum_first_n_reciprocal_to_T (d : ℤ) (h : S 5 d / 5 - S 3 d / 3 = 2) :
  let T := (1 - (1 / 2)) + ((1 / 2) - (1 / 3)) + ((1 / 3) - (1 / 4)) +
            ((1 / 4) - (1 / 5)) + ((1 / 5) - (1 / 6)) + ((1 / 6) - (1 / 7)) +
            ((1 / 7) - (1 / 8)) + ((1 / 8) - (1 / 9)) + ((1 / 9) - (1 / 10)) +
            ((1 / 10) - (1 / 11))
  in T = 10 / 11 := by
{
  sorry
}

end sum_first_n_reciprocal_to_T_l731_731670


namespace Simone_plays_squash_on_Friday_l731_731344

variable (day_of_week : Type)
variable [Enum day_of_week] -- Enum day_of_week or similar definition to manage week days

-- Definitions of days
variable (Monday Tuesday Wednesday Thursday Friday Saturday Sunday : day_of_week)
-- Note that below order and interval constraints are implied by being Enum

-- Define the sports that Simone plays
inductive Sport
| Jogging
| Karate
| Volleyball
| Squash
| Cricket

open Sport

-- Define a function that indicates the sport Simone plays on each day
variable (sport_of_day : day_of_week → Sport)

-- Given conditions
variable (H1 : ¬ (sport_of_day Tuesday = Jogging)) 
variable (H2 : sport_of_day Tuesday = Karate)
variable (H3 : sport_of_day Thursday = Volleyball)
variable (H4 : ∀ d, sport_of_day d = Jogging → sport_of_day (d.succ) ≠ Jogging)
variable (H5 : ∀ d, sport_of_day d = Cricket → (sport_of_day (d.pred) ≠ Jogging ∧ sport_of_day (d.pred) ≠ Squash))
variable (H6 : ∃ j1 j2 j3, sport_of_day j1 = Jogging ∧ 
                              sport_of_day j2 = Jogging ∧ 
                              sport_of_day j3 = Jogging ∧ 
                              j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3)

-- The statement: "Prove that Simone plays squash on Friday"
theorem Simone_plays_squash_on_Friday : sport_of_day Friday = Squash := 
sorry

end Simone_plays_squash_on_Friday_l731_731344


namespace probability_sum_less_than_product_l731_731833

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731833


namespace maximize_x4_y3_l731_731309

theorem maximize_x4_y3 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 50) :
  ∃ x y, x = 200 / 7 ∧ y = 150 / 7 ∧ (∀ a b, a > 0 → b > 0 → a + b = 50 → (a^4 * b^3) ≤ ((200 / 7)^4 * (150 / 7)^3)) :=
by {
  use [200 / 7, 150 / 7],
  split,
  {
    refl,
  },
  split,
  {
    refl,
  },
  {
    intro a,
    intro b,
    intro ha,
    intro hb,
    intro hab,
    sorry
  }
}

end maximize_x4_y3_l731_731309


namespace perimeter_of_similar_triangle_l731_731763

theorem perimeter_of_similar_triangle (a b c d : ℕ) (h_iso : (a = 12) ∧ (b = 24) ∧ (c = 24)) (h_sim : d = 30) 
  : (d + 2 * b) = 150 := by
  sorry

end perimeter_of_similar_triangle_l731_731763


namespace find_eighth_number_l731_731394

theorem find_eighth_number (x : ℝ) (h1 : x = 212) (h2 : (201 + 202 + 204 + 205 + 206 + 209 + 209 + 212 + x) / 9 = 207) : 
  201 + 202 + 204 + 205 + 206 + 209 + 209 + x = 215 := 
by 
  have sum_eq : 201 + 202 + 204 + 205 + 206 + 209 + 209 + 212 + 212 = 1863 :=
    sorry
  have sum_others_eq : 201 + 202 + 204 + 205 + 206 + 209 + 209 + 212 = 1648 :=
    sorry
  rw [<-sum_eq, <-sum_others_eq, h1]
  exact (1863 - 1648)

end find_eighth_number_l731_731394


namespace solve_for_x_l731_731015

theorem solve_for_x (c : ℝ) (h1 : c ≠ 0) (h2 : c ≠ 3) :
  (∃ x : ℝ, (4 + x) / (5 + x) = c / (3 * c)) →
  x = -7 / 2 :=
by
  -- introduce variables and assumptions
  intro h
  -- provide existence proof that x = -7/2
  use -7 / 2
  sorry

end solve_for_x_l731_731015


namespace range_of_expression_l731_731212

theorem range_of_expression (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 ≤ β ∧ β ≤ π / 2) :
    -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
by
  sorry

end range_of_expression_l731_731212


namespace angle_DAC_in_trapezoid_l731_731651

theorem angle_DAC_in_trapezoid 
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AD AB BC DC : ℝ) 
  (angle_DAC : ℝ)
  (h1 : AD = 1)
  (h2 : AB = 1)
  (h3 : BC = 1)
  (h4 : DC = 2)
  (h_par : parallel AB DC)
  (h_midpoint_P : ∃ P, midpoint P DC ∧ DP = 1 ∧ PC = 1) :
  angle_DAC = 90 :=
by
  sorry

end angle_DAC_in_trapezoid_l731_731651


namespace union_of_sets_l731_731241

variable {V : Type}
variable (a b : ℕ)
variable M : Set ℕ := {3, 2 ^ a}
variable N : Set ℕ := {a, b}

theorem union_of_sets (h : M ∩ N = {2}) : M ∪ N = {1, 2, 3} :=
by sorry

end union_of_sets_l731_731241


namespace sin_alpha_is_correct_l731_731204

noncomputable def sin_alpha_plus_pi_over_2 (α : ℝ) (x : ℝ) (sqrt_5 : ℝ) : ℝ :=
  if 1 < sqrt_5 then -- This condition ensures sqrt_5 = sqrt(5), more elaborate constraints can be added.
    -ℂ.sqrt (x^2 + sqrt_5^2) / 4
  else
    0

theorem sin_alpha_is_correct (α x : ℝ) (sqrt_5 : ℝ) 
  (hα_q2 : π/2 < (α % (2*π)) ∧ (α % (2*π)) < π) -- α is in the second quadrant
  (hP : (x^2 + 5 = sqrt_5^2)) -- Point P(x, sqrt(5)) on the terminal side
  (hcos : cos α = (ℂ.sqrt 2 * x) / 4) -- cos α = (ℂ.sqrt 2 * x) / 4
  : sin_alpha_plus_pi_over_2 α x sqrt_5 = -√(6)/4 :=
sorry

end sin_alpha_is_correct_l731_731204


namespace equalize_apricots_in_boxes_l731_731060

theorem equalize_apricots_in_boxes : 
  ∃ k : ℕ, k > 0 ∧ 
    (∀ (b : ℕ) (boxes : ℕ → ℕ), b = 15 ∧ (∀ i < b, boxes i = 0) → 
      (∃ (moves : ℕ → (ℕ → ℕ)), moves 0 = boxes ∧
      (∀ n > 0, ∃ (distinct_powers : list ℕ),
        (∀ x ∈ distinct_powers, ∃ m : ℕ, x = 2^m) ∧ 
        distinct_powers.nodup ∧
        (∀ i < b, moves n (i) = moves (n - 1) (i) + if i ∈ (range b).filter (λ j, j < distinct_powers.length) then distinct_powers.nth_le i sorry else 0) ∧
      (∀ i j < b, moves k i = moves k j))) ∧ k = 8) :=
sorry

end equalize_apricots_in_boxes_l731_731060


namespace max_min_f_interval_l731_731235

def f (x : ℝ) : ℝ := 3 * x / (x + 1)

theorem max_min_f_interval : 
  (∀ x ∈ Icc (2 : ℝ) 5, f x ≥ f 2) ∧ (∀ x ∈ Icc (2 : ℝ) 5, f x ≤ f 5) :=
by 
  sorry

end max_min_f_interval_l731_731235


namespace chord_length_of_circle_and_line_l731_731471

noncomputable def chord_length (x y : ℝ) : ℝ := 2 * real.sqrt (7 - 1)

theorem chord_length_of_circle_and_line :
  ∀ (x y : ℝ), (x^2 + (y - real.sqrt 3)^2 = 7) →
  (∀ (l : ℝ), l = √6 * x - √3 * y → l = 0) →
  chord_length x y = 2 * real.sqrt 6 := 
begin
  sorry
end

end chord_length_of_circle_and_line_l731_731471


namespace acute_triangle_ratio_le_0_7_l731_731189

theorem acute_triangle_ratio_le_0_7 (M : Finset (Finₓ 100 → ℝ × ℝ)) (hM : ∀ (p1 p2 p3 : ℝ × ℝ) (hp1 : p1 ∈ M) (hp2 : p2 ∈ M) (hp3 : p3 ∈ M), p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1 → ¬collinear p1 p2 p3) :
    (∃ (S G : ℕ), ∑ (t : Finset (Finₓ 100 → ℝ × ℝ)), acute_triangle t = S ∧ ∑ (t : Finset (Finₓ 100 → ℝ × ℝ)), triangle t = G ∧ (S : ℚ) / (G : ℚ) ≤ 0.7) := sorry

end acute_triangle_ratio_le_0_7_l731_731189


namespace new_ratio_l731_731482

-- Define variables and assumptions
variables (b s : ℝ)
assume h1: b = 3 * s -- Given condition: b is three times the speed of stream s

theorem new_ratio (b s : ℝ) (h1 : b = 3 * s) : b / (1.2 * s) = 5 / 2 :=
by
  -- Here we would provide the proof, but it is not required for this task
  sorry

end new_ratio_l731_731482


namespace add_in_base14_l731_731944

-- Define symbols A, B, C, D in base 10 as they are used in the base 14 representation
def base14_A : ℕ := 10
def base14_B : ℕ := 11
def base14_C : ℕ := 12
def base14_D : ℕ := 13

-- Define the numbers given in base 14
def num1_base14 : ℕ := 9 * 14^2 + base14_C * 14 + 7
def num2_base14 : ℕ := 4 * 14^2 + base14_B * 14 + 3

-- Define the expected result in base 14
def result_base14 : ℕ := 1 * 14^2 + 0 * 14 + base14_A

-- The theorem statement that needs to be proven
theorem add_in_base14 : num1_base14 + num2_base14 = result_base14 := by
  sorry

end add_in_base14_l731_731944


namespace least_number_of_roots_l731_731073

variable (g : ℝ → ℝ) -- Declare the function g with domain ℝ and codomain ℝ

-- Define the conditions as assumptions.
variable (h1 : ∀ x : ℝ, g (3 + x) = g (3 - x))
variable (h2 : ∀ x : ℝ, g (8 + x) = g (8 - x))
variable (h3 : g 0 = 0)

-- State the theorem to prove the necessary number of roots.
theorem least_number_of_roots : ∀ a b : ℝ, a ≤ -2000 ∧ b ≥ 2000 → ∃ n ≥ 668, ∃ x : ℝ, g x = 0 ∧ a ≤ x ∧ x ≤ b :=
by
  -- To be filled in with the logic to prove the theorem.
  sorry

end least_number_of_roots_l731_731073


namespace average_marks_l731_731752

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l731_731752


namespace log_eight_of_five_twelve_l731_731992

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731992


namespace cone_volume_l731_731769

theorem cone_volume (r h : ℝ) (h_cylinder_vol : π * r^2 * h = 72 * π) : 
  (1 / 3) * π * r^2 * (h / 2) = 12 * π := by
  sorry

end cone_volume_l731_731769


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l731_731597

open Set

noncomputable def U : Set ℤ := {x | -2 < x ∧ x < 2}
def A : Set ℤ := {x | x^2 - 5 * x - 6 = 0}
def B : Set ℤ := {x | x^2 = 1}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 6} :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
by
  sorry

theorem complement_of_intersection_in_U : U \ (A ∩ B) = {0, 1} :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l731_731597


namespace tangent_line_eq_monotonicity_intervals_range_of_m_l731_731584

-- (1) Prove the equation of the tangent line
theorem tangent_line_eq (a : ℝ) (h : a = 1) :
  let f := λ x : ℝ, (a * x) / (1 + x^2) + 1 in
  let df := λ x : ℝ, (1 - x^2) / ((x^2 + 1)^2) in
  f 0 = 1 ∧ df 0 = 1 →
  ∀ x y, (y - 1) = x - 0 → x - y + 1 = 0 :=
by sorry

-- (2) Prove the intervals of monotonicity
theorem monotonicity_intervals (a : ℝ) (f := λ x : ℝ, (a * x) / (1 + x^2) + 1) :
  let df := λ x : ℝ, (a * (1 - x^2)) / ((x^2 + 1)^2) in
  (∀ x ∈ (-1 : ℝ)..1, df x > 0) ∧ (∀ x ∈ {x : ℝ | x ∈ (-∞..-1) ∪ (1..∞)}, df x < 0) ↔ a > 0 ∧
  (∀ x ∈ (-1 : ℝ)..1, df x < 0) ∧ (∀ x ∈ {x : ℝ | x ∈ (-∞..-1) ∪ (1..∞)}, df x > 0) ↔ a < 0 :=
by sorry

-- (3) Prove the range of m
theorem range_of_m (a : ℝ) (m : ℝ) (h : a > 0) (f := λ x : ℝ, (a * x) / (1 + x^2) + 1) (g := λ x : ℝ, x^2 * exp (m * x)) :
  (∀ x1 x2, x1 ∈ [0, 2] → x2 ∈ [0, 2] → f x1 ≥ g x2) ↔ m ∈ Icc (-∞) (-log 2) :=
by sorry

end tangent_line_eq_monotonicity_intervals_range_of_m_l731_731584


namespace proof_solution_l731_731551

noncomputable def proof_problem (n : ℕ) (hn : n ≥ 3) 
                                (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) 
                                (b : Fin n → ℝ) (hb : ∀ i, 0 ≤ b i) : Prop :=
∀ k : Fin n, ∑ i in Finset.univ.erase k, ∑ j in Finset.univ.erase k, if i ≠ j then a i * b j else 0 = 0 → 
(b = 0)

theorem proof_solution (n : ℕ) (hn : n ≥ 3) 
                                (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) 
                                (b : Fin n → ℝ) (hb : ∀ i, 0 ≤ b i) :
  proof_problem n hn a ha b hb :=
by { sorry }

end proof_solution_l731_731551


namespace sum_of_primes_between_20_and_40_l731_731903

def prime_numbers_between_20_and_40 : set ℕ := {23, 29, 31, 37}

def primes_greater_than_25 (s : set ℕ) : set ℕ := { x ∈ s | x > 25 }

theorem sum_of_primes_between_20_and_40 :
  let s := primes_greater_than_25 prime_numbers_between_20_and_40 in
  ∑ x in s, x = 97 :=
sorry

end sum_of_primes_between_20_and_40_l731_731903


namespace FH_minus_GH_eq_BC_l731_731654

theorem FH_minus_GH_eq_BC {A B C D E F G H : Point} {a b c : ℝ} 
    (ABC_isosceles : AB = AC)
    (D_midpoint : D == (B + C) / 2)
    (DE_square : E == opposite_vertex_square D B)
    (circle_intersections : circle D (sqrt 2 * BD) ∩ BC = {F, G})
    (H_measured : parallel A BC ∧ H == A + AB) :
    FH - GH = BC :=
sorry

end FH_minus_GH_eq_BC_l731_731654


namespace probability_sum_less_than_product_l731_731840

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731840


namespace closest_integer_to_cube_root_of_200_l731_731040

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), 
    (n = 6) ∧ (n^3 < 200) ∧ (200 < (n + 1)^3) ∧ 
    (∀ m : ℤ, (m^3 < 200) → (200 < (m + 1)^3) → (Int.abs (n - Int.ofNat (200 ^ (1/3 : ℝ)).round) < Int.abs (m - Int.ofNat (200 ^ (1/3 : ℝ)).round))) :=
begin
  sorry
end

end closest_integer_to_cube_root_of_200_l731_731040


namespace probability_sum_less_than_product_l731_731837

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l731_731837


namespace probability_ab_gt_a_add_b_l731_731802

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731802


namespace common_ratio_of_geometric_series_l731_731950

noncomputable def geometric_series_common_ratio (a S : ℝ) : ℝ := 1 - (a / S)

theorem common_ratio_of_geometric_series :
  geometric_series_common_ratio 520 3250 = 273 / 325 :=
by
  sorry

end common_ratio_of_geometric_series_l731_731950


namespace Peter_age_approx_l731_731516

def age_of_Cindy : ℕ := 5
def age_of_Jan := age_of_Cindy + 2
def age_of_Marcia := 2 * age_of_Jan
def age_of_Greg := age_of_Marcia + 2
def age_of_Bobby := 1.5 * age_of_Greg
def age_of_Peter := (Real.sqrt age_of_Bobby) * 2

theorem Peter_age_approx : abs (age_of_Peter - 10) < 1 :=
by
  sorry

end Peter_age_approx_l731_731516


namespace log_prop_example_l731_731178

def custom_operation (a b : ℝ) : ℝ :=
  if a < b then (b - 1) / a else (a + 1) / b

theorem log_prop_example :
  custom_operation (Real.log 10000) ((1 / 2) ^ -2) = 5 / 4 :=
by
  have h1 : Real.log 10000 = 4 := sorry
  have h2 : (1 / 2) ^ -2 = 4 := sorry
  show custom_operation 4 4 = 5 / 4 from sorry

end log_prop_example_l731_731178


namespace cylinder_surface_area_calc_l731_731474

def cylinder_total_surface_area (h r : ℕ) := 
  2 * π * r^2 + 2 * h * π * r 

theorem cylinder_surface_area_calc : cylinder_total_surface_area 12 5 = 170 * π := 
  sorry

end cylinder_surface_area_calc_l731_731474


namespace segments_form_triangle_l731_731428

-- Define the types for Points, Lines, and the Plane
noncomputable theory
open Set Classical

universe u

variable {P : Type u} [h : Inhabited P] [decP : DecidableEq P]

-- Definition of a line in a plane
def Line := {l : Set P // ∃ A B : P, A ≠ B ∧ ∀ (p : P), p ∈ l ↔ (p = A ∨ p = B)}

-- The set of all lines in the plane
axiom L : Set (Line)

-- We create segments as subsets of lines
def segment (l : Line) : Set P := {p : P | ∃ A B : P, A ≠ B ∧ (p = A ∨ p = B)}

-- Set X is the set of all points of the chosen segments
def X : Set P := {x : P | ∃ (l : Line), x ∈ segment l}

-- The statement to be proved:
theorem segments_form_triangle (h_infinite : Infinite L) : ∃ A B C : P, 
  (A ∈ X) ∧ (B ∈ X) ∧ (C ∈ X) ∧ 
  ∃ l₁ l₂ l₃ : Line, 
    A ∈ segment l₁ ∧ B ∈ segment l₁ ∧ 
    B ∈ segment l₂ ∧ C ∈ segment l₂ ∧ 
    C ∈ segment l₃ ∧ A ∈ segment l₃ :=
sorry

end segments_form_triangle_l731_731428


namespace calculate_fraction_l731_731115

theorem calculate_fraction : (5 / (8 / 13) / (10 / 7) = 91 / 16) :=
by
  sorry

end calculate_fraction_l731_731115


namespace abs_inequality_l731_731362

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l731_731362


namespace part_a_part_b_part_c_part_d_part_e_part_f_l731_731911

-- Part (a)
theorem part_a (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^2 = 5 * k + 1 ∨ n^2 = 5 * k - 1 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^4 - 1 = 5 * k := 
sorry

-- Part (c)
theorem part_c (n : ℤ) : n^5 % 10 = n % 10 := 
sorry

-- Part (d)
theorem part_d (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := 
sorry

-- Part (e)
theorem part_e (k n : ℤ) (h1 : ¬ ∃ j : ℤ, k = 5 * j) (h2 : ¬ ∃ j : ℤ, n = 5 * j) : ∃ j : ℤ, k^4 - n^4 = 5 * j := 
sorry

-- Part (f)
theorem part_f (k m n : ℤ) (h : k^2 + m^2 = n^2) : ∃ j : ℤ, k = 5 * j ∨ ∃ r : ℤ, m = 5 * r ∨ ∃ s : ℤ, n = 5 * s := 
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_l731_731911


namespace sum_first_13_terms_l731_731283

variable {a_n : ℕ → ℝ}

-- Define the arithmetic sequence property
variable h1 : a_n 6 + a_n 7 + a_n 8 = 12

-- Prove the sum of the first 13 terms is 52
theorem sum_first_13_terms (h : a_n 6 + a_n 7 + a_n 8 = 12) :
  ∑ i in finset.range 13, a_n i = 52 := by
  sorry

end sum_first_13_terms_l731_731283


namespace addition_and_rounding_l731_731943

def round_to_hundredth (x : ℝ) : ℝ :=
  let factor := 100
  let temp := (x * factor).round
  temp / factor

theorem addition_and_rounding:
  let A := 132.478
  let B := 56.925
  round_to_hundredth (A + B) = 189.40 :=
by
  sorry

end addition_and_rounding_l731_731943


namespace absent_minded_scientist_mistake_l731_731733

theorem absent_minded_scientist_mistake (ξ η : ℝ) (h₁ : E ξ = 3) (h₂ : E η = 5) (h₃ : E (min ξ η) = 3 + 2/3) : false :=
by
  sorry

end absent_minded_scientist_mistake_l731_731733


namespace library_visits_l731_731340

theorem library_visits (n : ℕ) (a b : Fin n → ℝ) :
  (∀ i : Fin n, a i ≤ b i) → 
  (∀ (i j : Fin n), i ≠ j → ∃ k : Fin n, (i ≠ k ∧ j ≠ k) ∧ (max (a i) (a j)) * (max (a i) (a k)) ≤ min (b i) (b k)) →
  ∃ (T T' : ℝ), ∀ i : Fin n, (T ∈ Icc (a i) (b i)) ∨ (T' ∈ Icc (a i) (b i)) :=
begin
  sorry
end

end library_visits_l731_731340


namespace absent_minded_scientist_mistake_l731_731731

theorem absent_minded_scientist_mistake (ξ η : ℝ) (h₁ : E ξ = 3) (h₂ : E η = 5) (h₃ : E (min ξ η) = 3 + 2/3) : false :=
by
  sorry

end absent_minded_scientist_mistake_l731_731731


namespace original_faculty_approx_l731_731446

noncomputable def original_faculty_number (reduced_amount : ℝ) (percentage_left : ℝ) : ℝ :=
reduced_amount / percentage_left

theorem original_faculty_approx (reduced_amount : ℝ) (percentage_left : ℝ)
  (h : reduced_amount = 195) (p : percentage_left = 0.77) :
  original_faculty_number reduced_amount percentage_left ≈ 253 :=
sorry

end original_faculty_approx_l731_731446


namespace trains_clear_time_l731_731424

noncomputable def total_distance (length1 length2 : ℤ) : ℤ :=
  length1 + length2

noncomputable def relative_speed (speed1 speed2 : ℤ) : ℤ :=
  speed1 + speed2

noncomputable def relative_speed_m_s (relative_speed_kmph : ℤ) : ℝ :=
  (relative_speed_kmph * 1000) / 3600

noncomputable def time_to_clear (distance : ℤ) (speed_m_s : ℝ) : ℝ :=
  distance / speed_m_s

theorem trains_clear_time :
  time_to_clear 
    (total_distance 141 165)
    (relative_speed_m_s (relative_speed 80 65)) ≈ 7.6 := 
sorry

end trains_clear_time_l731_731424


namespace coffee_blend_solution_l731_731103

def coffee_blend_condition (A B C : ℝ) :=
  (B = 2 * A) ∧
  (C = 6 * A) ∧
  (4.60 * A + 5.95 * B + 6.80 * C = 1924.80) ∧
  (A + B + C ≥ 150)

theorem coffee_blend_solution : ∃ (A B C : ℝ), coffee_blend_condition A B C ∧ A ≈ 33.59 ∧ B ≈ 67.18 ∧ C ≈ 201.54 :=
by
  sorry

end coffee_blend_solution_l731_731103


namespace part1_part2_part3_l731_731627

variable {a b c : ℝ}

-- Part (1)
theorem part1 (a b c : ℝ) : a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

-- Part (2)
theorem part2 (a b c : ℝ) : 2 * a ^ 2 * b ^ 2 + 2 * b ^ 2 * c ^ 2 + 2 * c ^ 2 * a ^ 2 > a ^ 4 + b ^ 4 + c ^ 4 :=
sorry

-- Part (3)
theorem part3 (a b c : ℝ) : 2 * a * b + 2 * b * c + 2 * c * a > a ^ 2 + b ^ 2 + c ^ 2 :=
sorry

end part1_part2_part3_l731_731627


namespace sum_of_valid_n_l731_731865

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l731_731865


namespace abs_inequality_l731_731366

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l731_731366


namespace cube_face_diagonal_edge_angle_l731_731631

-- defining a cube
structure Cube :=
(vertex : Type)
(edges : vertex → vertex → Prop)

-- face diagonal and which should be incident to the same vertex
def face_diagonal_incident_edge_angle_60 (C : Cube) (v : C.vertex) (e1 e2 : C.vertex → C.vertex → Prop) : Prop :=
  -- there are three edges extending from any vertex
  (e1 v : C.vertex) ∧ (e2 v : C.vertex) ∧
  -- angle between a face diagonal and an edge which meet at the same vertex is 60 degrees
  ∃ angle, angle = 60

theorem cube_face_diagonal_edge_angle (C : Cube) (v : C.vertex) (e1 e2 : C.vertex → C.vertex → Prop) :
  face_diagonal_incident_edge_angle_60 C v e1 e2 :=
by
  sorry

end cube_face_diagonal_edge_angle_l731_731631


namespace max_a_b_c_l731_731671

noncomputable def matrix_B (a b c : ℤ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (1 / 7 : ℚ) • Matrix.of ![![-5, a], ![b, c]]

theorem max_a_b_c (a b c : ℤ)
  (hB2 : (matrix_B a b c) ⬝ (matrix_B a b c) = 1) : 
  a + b + c ≤ 32 :=
  sorry

end max_a_b_c_l731_731671


namespace requires_conditional_l731_731435

def f_A (x : ℝ) : ℝ := x^2 - 1

def f_B (x : ℝ) : ℝ := Real.log x / Real.log 2

def f_C (x : ℝ) : ℝ :=
  if x ≥ -1 then
    x + 1
  else
    -x^2 - 2 * x

def f_D (x : ℝ) : ℝ := 3^x

theorem requires_conditional (requires_conditional : (ℝ → ℝ) → Prop) :
  requires_conditional f_C :=
sorry

end requires_conditional_l731_731435


namespace quadratic_through_points_l731_731573

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_through_points :
  f (1 / 2) = 3 / 4 ∧ f (-1) = 3 ∧ f (2) = 3 :=
by
  split
  show f (1/2) = 3/4
  . unfold f
    norm_num
  split
  show f (-1) = 3
  . unfold f
    norm_num
  show f (2) = 3
  . unfold f
    norm_num

end quadratic_through_points_l731_731573


namespace trigonometric_identity_l731_731562

noncomputable def triangle_constants (PQ PR QR : ℝ) :=
  PQ = 7 ∧ PR = 8 ∧ QR = 6

theorem trigonometric_identity (PQ PR QR P Q R : ℝ) 
  (h : triangle_constants PQ PR QR) :
  (cos ((P - Q) / 2) / sin (R / 2)) - (sin ((P - Q) / 2) / cos (R / 2)) = 2 :=
sorry

end trigonometric_identity_l731_731562


namespace probability_ab_gt_a_add_b_l731_731795

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l731_731795


namespace lunar_phase_intervals_l731_731599

noncomputable def Earth_orbit_radius : ℝ := 150 * 10 ^ 6
noncomputable def Moon_orbit_radius : ℝ := 384 * 10 ^ 3
noncomputable def lunar_cycle : ℝ × ℕ × ℕ := (29, 12, 44) -- 29 days, 12 hours, 44 minutes

theorem lunar_phase_intervals :
  let gamma := Real.arccos ((Moon_orbit_radius) / (Earth_orbit_radius)),
      quarter_phase := (90 * (Math.pi / 180)) - gamma,
      quarter_duration := (lunar_cycle.1 * 24 * 3600 + lunar_cycle.2 * 3600 + lunar_cycle.3 * 60) / 4,
      deviation := (9 / 60) * (quarter_duration / 90),
      adjusted_duration_short := quarter_duration - deviation,
      adjusted_duration_long := quarter_duration + deviation
  in
  (adjusted_duration_short, adjusted_duration_long) = 
  (7 * 24 * 3600 + 8 * 3600 + 53 * 60, 7 * 24 * 3600 + 9 * 3600 + 29 * 60)
:=
sorry

end lunar_phase_intervals_l731_731599


namespace eggs_from_Martha_is_2_l731_731660

def eggs_from_Gertrude : ℕ := 4
def eggs_from_Blanche : ℕ := 3
def eggs_from_Nancy : ℕ := 2
def total_eggs_left : ℕ := 9
def eggs_dropped : ℕ := 2

def total_eggs_before_dropping (eggs_from_Martha : ℕ) :=
  eggs_from_Gertrude + eggs_from_Blanche + eggs_from_Nancy + eggs_from_Martha - eggs_dropped = total_eggs_left

-- The theorem stating the eggs collected from Martha.
theorem eggs_from_Martha_is_2 : ∃ (m : ℕ), total_eggs_before_dropping m ∧ m = 2 :=
by
  use 2
  sorry

end eggs_from_Martha_is_2_l731_731660


namespace product_of_integers_odd_then_factors_and_sum_parity_l731_731402

theorem product_of_integers_odd_then_factors_and_sum_parity (n : ℕ) (a : Fin n → ℤ) 
  (h : (∏ i, a i) % 2 = 1) : (∀ i, a i % 2 = 1) ∧ ((∑ i, a i) % 2 = n % 2) := 
  by sorry

end product_of_integers_odd_then_factors_and_sum_parity_l731_731402


namespace probability_sum_less_than_product_l731_731850

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731850


namespace living_room_size_l731_731677

theorem living_room_size :
  let length := 16
  let width := 10
  let total_rooms := 6
  let total_area := length * width
  let unit_size := total_area / total_rooms
  let living_room_size := 3 * unit_size
  living_room_size = 80 := by
    sorry

end living_room_size_l731_731677


namespace sum_divisibility_l731_731333

theorem sum_divisibility (a b : ℤ) (h : 6 * a + 11 * b ≡ 0 [ZMOD 31]) : a + 7 * b ≡ 0 [ZMOD 31] :=
sorry

end sum_divisibility_l731_731333


namespace sum_of_n_l731_731872

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l731_731872


namespace perfect_squares_between_50_and_1000_l731_731256

theorem perfect_squares_between_50_and_1000 :
  ∃ (count : ℕ), count = 24 ∧ ∀ (n : ℕ), 50 < n * n ∧ n * n < 1000 ↔ 8 ≤ n ∧ n ≤ 31 :=
by {
  -- proof goes here
  sorry
}

end perfect_squares_between_50_and_1000_l731_731256


namespace incorrect_calculation_l731_731713

noncomputable def ξ : ℝ := 3 -- Expected lifetime of the sensor
noncomputable def η : ℝ := 5 -- Expected lifetime of the transmitter
noncomputable def T (ξ η : ℝ) : ℝ := min ξ η -- Lifetime of the entire device

theorem incorrect_calculation (h1 : E ξ = 3) (h2 : E η = 5) (h3 : E (min ξ η ) = 3.67) : False :=
by
  have h4 : E (min ξ η ) ≤ 3 := sorry -- Based on properties of expectation and min
  have h5 : 3.67 > 3 := by linarith -- Known inequality
  sorry

end incorrect_calculation_l731_731713


namespace area_bounded_by_curves_l731_731957

noncomputable def bounded_area : ℝ :=
  ∫ (x : ℝ) in 1..Real.exp 3, 1 / (x * Real.sqrt(1 + Real.log x))

theorem area_bounded_by_curves : bounded_area = 2 := by
  sorry

end area_bounded_by_curves_l731_731957


namespace solve_inequality_l731_731370

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731370


namespace correct_propositions_l731_731232

-- Define the conditions (propositions)
def prop1 (a x : ℝ) : Prop := a < x ∧ x < 3a
def prop2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(x + 1)) = f (x + 1)
def prop3 (a : ℝ) : Prop := a >= 1 → ∃ x, |x - 4| + |x - 3| < a
def prop4 (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ y, f y = a → y = a
def prop5 (α β : ℝ) : Prop := cos α * cos β = 1 → sin (α + β) = 0

-- The statement to prove
theorem correct_propositions (a x : ℝ) (f : ℝ → ℝ) (α β : ℝ) :
  ¬prop1 a x ∧ prop2 f ∧ ¬prop3 a ∧ prop4 f a ∧ ¬prop5 α β :=
by
  sorry

end correct_propositions_l731_731232


namespace diameter_of_circle_l731_731919

theorem diameter_of_circle
  (A B C D : Point)
  (AB CD AD BE : Line)
  (x y z : ℝ)
  (Circle : ∀ (P : Point), P ∈ Circle → (distance P A = distance P B))
  (Hdiam : AB = diameter Circle)
  (Htangent_AD : Tangent Circle A AD)
  (Htangent_BE : Tangent Circle B BE)
  (Hlength_AD : length(AD) = x)
  (Hlength_BE : length(BE) = y)
  (Hmidpoint_C : midpoint C AB)
  (Htangent_CD : Tangent Circle C CD)
  (Hperpendicular_CD_AB : perpendicular CD AB)
  (Hlength_CD : length(CD) = z) :
  diameter Circle = 2 * z := 
  sorry

end diameter_of_circle_l731_731919


namespace probability_sum_less_than_product_l731_731847

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l731_731847


namespace cube_root_floor_product_l731_731969

theorem cube_root_floor_product :
  (∏ i in (range 500).filter (λ k, k % 3 = 1), ⌊(k : ℝ)^(1/3)⌋) / 
  (∏ i in (range 501).filter (λ k, k % 3 = 2), ⌊(k : ℝ)^(1/3)⌋) = 1 / 8 := 
sorry

end cube_root_floor_product_l731_731969


namespace solve_inequality_l731_731369

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l731_731369


namespace cos_value_l731_731257

theorem cos_value (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 :=
sorry

end cos_value_l731_731257


namespace sum_of_consecutive_integers_20_l731_731612

theorem sum_of_consecutive_integers_20 : 
  ∃! (s : Set ℕ), (∃ (a n : ℕ), 1 ≤ a ∧ 2 ≤ n ∧ s = {a + i | i < n}) ∧ (s.sum id = 20) := 
sorry

end sum_of_consecutive_integers_20_l731_731612


namespace p2_div_q2_eq_4_l731_731139

theorem p2_div_q2_eq_4 
  (p q : ℝ → ℝ)
  (h1 : ∀ x, p x = 12 * x)
  (h2 : ∀ x, q x = (x + 4) * (x - 1))
  (h3 : p 0 = 0)
  (h4 : p (-1) / q (-1) = -2) :
  (p 2 / q 2 = 4) :=
by {
  sorry
}

end p2_div_q2_eq_4_l731_731139


namespace perfect_squares_count_l731_731254

theorem perfect_squares_count : 
  let n_min := Nat.ceil (sqrt 50)
  let n_max := Nat.floor (sqrt 1000)
  n_min = 8 ∧ n_max = 31 → (n_max - n_min + 1 = 24) :=
begin
  intros,
  -- step a, nonnegative integer sqrt
  have h_n_min := Nat.ceil_spec (sqrt 50),
  have h_n_max := Nat.floor_spec (sqrt 1000),
  -- step b, we can prove the floors by direct calculation
  -- n_min = 8 and n_max = 31 must be true
  have : n_min = 8 := by linarith only [Nat.ceil (sqrt 50)],
  have : n_max = 31 := by linarith only [Nat.floor (sqrt 1000)],
  exact sorry -- Proof of main statement, assuming correct bounds give 24
end

end perfect_squares_count_l731_731254


namespace area_inequality_l731_731657

-- Definitions of the given conditions
variables {A B C K M R : Type} -- Points
variables (E E1 E2 : ℝ) -- Areas
variables [hABC : (triangle ABC)] [hK : (point_on_segment K BC)]
variables [hM : (point_on_segment M AB)] [hR : (point_on_segment R MK)]

-- Areas of triangles
def area_ABC : ℝ := E
def area_AMR : ℝ := E1
def area_CKR : ℝ := E2

-- Lean 4 statement: Prove the inequality
theorem area_inequality
  (h₁ : E > 0) (h₂ : E1 > 0) (h₃ : E2 > 0) : 
  (E1 / E) ^ (1/3) + (E2 / E) ^ (1/3) ≤ 1 :=
  sorry

end area_inequality_l731_731657


namespace closest_integer_to_cuberoot_of_200_l731_731025

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l731_731025


namespace angle_PQR_either_30_or_150_l731_731213

theorem angle_PQR_either_30_or_150
  {A B C P Q R : Type}
  (h1 : parallel AB PQ)
  (h2 : parallel BC QR)
  (angle_ABC_eq_30 : ∠ABC = 30) :
  ∠PQR = 30 ∨ ∠PQR = 150 := by
  sorry

end angle_PQR_either_30_or_150_l731_731213


namespace complement_of_A_in_U_l731_731596

def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}
def CU_A : Set ℤ := {x ∈ U | x ∉ A}

theorem complement_of_A_in_U :
  CU_A = {-2, 1, 5} :=
by
  -- Proof goes here
  sorry

end complement_of_A_in_U_l731_731596


namespace crabapple_recipients_sequences_l731_731321

-- Define the number of students in Mrs. Crabapple's class
def num_students : ℕ := 12

-- Define the number of class meetings per week
def num_meetings : ℕ := 5

-- Define the total number of different sequences
def total_sequences : ℕ := num_students ^ num_meetings

-- The target theorem to prove
theorem crabapple_recipients_sequences :
  total_sequences = 248832 := by
  sorry

end crabapple_recipients_sequences_l731_731321


namespace min_value_ineq_l731_731196

noncomputable def a_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a 2018 = a 2017 + 2 * a 2016

theorem min_value_ineq (a : ℕ → ℝ) (m n : ℕ) 
  (h : a_sequence a) 
  (h2 : a m * a n = 16 * (a 1) ^ 2) :
  (4 / m) + (1 / n) ≥ 5 / 3 :=
sorry

end min_value_ineq_l731_731196


namespace sarah_vs_ryan_l731_731339

theorem sarah_vs_ryan (sarah_apples ryan_apples : ℕ) (lila_cherries : ℕ)
  (sarah_apples_eq : sarah_apples = 45)
  (ryan_apples_eq : ryan_apples = 9)
  (lila_apples : ℝ)
  (lila_apples_eq : lila_apples = 2.5 * ryan_apples)
  (sarah_cherries ryan_cherries : ℕ)
  (lila_cherries_eq : lila_cherries = 60)
  (sarah_cherries_eq : sarah_cherries = 0.75 * lila_cherries)
  (ryan_cherries_eq : ryan_cherries = 2 * lila_cherries)
  (sarah_total ryan_total : ℝ)
  (sarah_total_eq : sarah_total = sarah_apples + sarah_cherries)
  (ryan_total_eq : ryan_total = ryan_apples + ryan_cherries):
  sarah_total / ryan_total ≈ 0.7 :=
by
  sorry

end sarah_vs_ryan_l731_731339


namespace domain_of_f_2x_minus_1_l731_731591

variable (f : ℝ → ℝ)
variable h₁ : ∀ x, x ∈ Set.Icc (-2 : ℝ) (3 : ℝ) → x ∈ dom f

theorem domain_of_f_2x_minus_1 : 
  (∀ x, x ∈ Set.Icc (-2 : ℝ) (3 : ℝ) ↔ (2 * x - 1) ∈ dom f) → 
  (∀ x, x ∈ Set.Icc (-1/2: ℝ) (2 : ℝ)) :=
by
  sorry

end domain_of_f_2x_minus_1_l731_731591


namespace domain_of_f_symmetry_of_f_l731_731703

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 * x^2 - x^4)) / (abs (x - 2) - 2)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

theorem symmetry_of_f :
  ∀ x : ℝ, f (x + 1) + 1 = f (-(x + 1)) + 1 :=
by
  sorry

end domain_of_f_symmetry_of_f_l731_731703


namespace closest_integer_to_cuberoot_of_200_l731_731024

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l731_731024


namespace incorrect_calculation_l731_731717

noncomputable def ξ : ℝ := 3
noncomputable def η : ℝ := 5

def T (ξ η : ℝ) : ℝ := min ξ η

theorem incorrect_calculation : E[T(ξ, η)] ≤ 3 := by
  sorry

end incorrect_calculation_l731_731717


namespace find_angle_CDE_l731_731648

-- Definition of the angles and their properties
variables {A B C D E : Type}

-- Hypotheses
def angleA_is_right (angleA: ℝ) : Prop := angleA = 90
def angleB_is_right (angleB: ℝ) : Prop := angleB = 90
def angleC_is_right (angleC: ℝ) : Prop := angleC = 90
def angleAEB_value (angleAEB : ℝ) : Prop := angleAEB = 40
def angleBED_eq_angleBDE (angleBED angleBDE : ℝ) : Prop := angleBED = angleBDE

-- The theorem to be proved
theorem find_angle_CDE 
  (angleA : ℝ) (angleB : ℝ) (angleC : ℝ) (angleAEB : ℝ) (angleBED angleBDE : ℝ) (angleCDE : ℝ) :
  angleA_is_right angleA → 
  angleB_is_right angleB → 
  angleC_is_right angleC → 
  angleAEB_value angleAEB → 
  angleBED_eq_angleBDE angleBED angleBDE →
  angleBED = 45 →
  angleCDE = 95 :=
by
  intros
  sorry


end find_angle_CDE_l731_731648


namespace radius_of_bicycle_wheel_is_13_l731_731859

-- Define the problem conditions
def diameter_cm : ℕ := 26

-- Define the function to calculate radius from diameter
def radius (d : ℕ) : ℕ := d / 2

-- Prove that the radius is 13 cm when diameter is 26 cm
theorem radius_of_bicycle_wheel_is_13 :
  radius diameter_cm = 13 := 
sorry

end radius_of_bicycle_wheel_is_13_l731_731859


namespace quadratic_eq_solution_trig_expression_calc_l731_731918

-- Part 1: Proof for the quadratic equation solution
theorem quadratic_eq_solution : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by
  sorry

-- Part 2: Proof for trigonometric expression calculation
theorem trig_expression_calc : (-1 : ℝ) ^ 2 + 2 * Real.sin (Real.pi / 3) - Real.tan (Real.pi / 4) = Real.sqrt 3 :=
by
  sorry

end quadratic_eq_solution_trig_expression_calc_l731_731918


namespace sum_lt_prod_probability_l731_731825

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l731_731825


namespace sin_A_add_B_correct_l731_731642

noncomputable def sin_A_add_B (A B : ℝ) : ℝ :=
  if h1 : B > π / 6 ∧ sin (A + π / 6) = 3 / 5 ∧ cos (B - π / 6) = 4 / 5 then
    sin (A + B)
  else 0

theorem sin_A_add_B_correct (A B : ℝ) (hB : B > π / 6)
  (hA_plus_pi_6 : sin (A + π / 6) = 3 / 5)
  (hB_minus_pi_6 : cos (B - π / 6) = 4 / 5) : 
  sin (A + B) = 24 / 25 := 
begin
  sorry
end

end sin_A_add_B_correct_l731_731642


namespace geometric_similarity_possible_l731_731505

-- Assume edge lengths of the box are a, b, and c
variables (a b c : ℝ)

-- Define the volumes
def V_box := a * b * c
def V_package := 10 * V_box

-- Define the similarity ratio
def k := real.cbrt 10
def k_cubed_eq_ten : k^3 = 10 := real.cbrt_cubed 10

-- Define the conditions for similar geometric shapes
def is_geometrically_similar := 
  (a = k * a ∨ b = k * b ∨ c = k * c)

theorem geometric_similarity_possible 
  (h₁ : V_box = a * b * c)
  (h₂ : V_package = 10 * V_box)
  (h₃ : k^3 = 10) : 
  is_geometrically_similar a b c := 
sorry

end geometric_similarity_possible_l731_731505


namespace sculpture_cost_in_INR_l731_731689

def USD_per_NAD := 1 / 5
def INR_per_USD := 8
def cost_in_NAD := 200
noncomputable def cost_in_INR := (cost_in_NAD * USD_per_NAD) * INR_per_USD

theorem sculpture_cost_in_INR :
  cost_in_INR = 320 := by
  sorry

end sculpture_cost_in_INR_l731_731689


namespace sum_of_digits_of_N_l731_731978

-- Define N based on given conditions
def N : ℕ := (List.range 1 65).sum (λ k => 100^k - 1)

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Main theorem to prove
theorem sum_of_digits_of_N : sum_of_digits N = 1152 := by
  sorry

end sum_of_digits_of_N_l731_731978


namespace probability_in_interval_l731_731286

open Probability

noncomputable def probability_event_in_interval (a : ℝ) : Prop :=
  2 + a - a^2 > 0

theorem probability_in_interval : 
  (1/2 : ℝ) = P (λ a, a ∈ set.Icc (-1) 2) (uniform (set.Icc (-3) 3)) :=
begin
  simp only [uniform, interval_integral, set_integral, Icc, Ioc, Pi.smul_apply],
  sorry
-- the details of the integral and probability calculation would go here, omitted by "sorry"
end

end probability_in_interval_l731_731286


namespace find_probabilities_l731_731072

theorem find_probabilities (p_1 p_3 : ℝ)
  (h1 : p_1 + 0.15 + p_3 + 0.25 + 0.35 = 1)
  (h2 : p_3 = 4 * p_1) :
  p_1 = 0.05 ∧ p_3 = 0.20 :=
by
  sorry

end find_probabilities_l731_731072


namespace chord_properties_l731_731926

-- Define the conditions: parabola, point P, and chord bisected by P
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

def on_line (P : ℝ × ℝ) (m c : ℝ) : Prop :=
  P.2 = m * P.1 + c

-- Define the problem statement
theorem chord_properties (A B P : ℝ × ℝ) (hP : P = (2, 1)) 
  (hA_on_parabola : parabola A.1 A.2) (hB_on_parabola : parabola B.1 B.2) (hMid : is_midpoint P A B) :
  (∃ m c : ℝ, on_line P m c ∧ ∀ Q ∈ {A, B}, on_line Q m c → 2*P.1 - P.2 - 3 = 0)
  ∧ real.sqrt (5 * ((A.1 + B.1)^2 - 4 * A.1 * B.1)) = real.sqrt 35 := 
  sorry

end chord_properties_l731_731926


namespace min_distance_to_directrix_l731_731199

noncomputable def ellipse_min_distance (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 1/a^2 + 4/b^2 = 1) : ℝ :=
  if a * b > 0 then sqrt 5 + 2 else 0

theorem min_distance_to_directrix (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 1/a^2 + 4/b^2 = 1) :
  ellipse_min_distance a b h1 h2 h3 = sqrt 5 + 2 :=
sorry

end min_distance_to_directrix_l731_731199


namespace log_eight_of_five_twelve_l731_731991

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l731_731991


namespace equal_angles_in_projection_l731_731302

-- Defining the problem in Lean
theorem equal_angles_in_projection (ABC : Triangle) (X : Point)
  (h1 : X ∈ interior ABC)
  (D E Y Z : Point)
  (h2 : lies_on D (line_through B X) ∧ lies_on D (line_through A C))
  (h3 : lies_on E (line_through C X) ∧ lies_on E (line_through A B))
  (h4 : Y = intersection (line_through A D) (line_through X E))
  (h5 : Z = orthogonal_projection X (line_through B C)) :
  angle D Z X = angle E Z A :=
begin
  sorry
end

end equal_angles_in_projection_l731_731302


namespace find_angle_BAC_l731_731423

-- Define the circle and points
structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*} [euclidean_space α]

def Circle_incircle {α : Type*} [euclidean_space α] (A B C : α) : Prop :=
∃ (Ω : Circle α), Ω.touches A /\ Ω.touches B /\ Ω.touches C

-- The condition given in the problem
def condition (A B C : α) := Circle_incircle A B C ∧ (∃ (x : ℝ), 3 * x + 5 * x = 360)

-- The Lean theorem statement equivalent to the problem
theorem find_angle_BAC (A B C : α) (h : condition A B C) : 
  ∃ (θ : ℝ), θ = 67.5 :=
by
  sorry

end find_angle_BAC_l731_731423


namespace rational_solution_square_of_rational_l731_731565

theorem rational_solution_square_of_rational (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : 
    ∃ q : ℚ, 1 - x * y = q^2 :=
begin
    sorry
end

end rational_solution_square_of_rational_l731_731565


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l731_731123

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l731_731123


namespace number_of_balls_is_fifty_l731_731924

variable (x : ℝ)
variable (h : x - 40 = 60 - x)

theorem number_of_balls_is_fifty : x = 50 :=
by
  have : 2 * x = 100 := by
    linarith
  linarith

end number_of_balls_is_fifty_l731_731924


namespace fraction_calculation_correct_l731_731511

noncomputable def calculate_fraction : ℚ :=
  let numerator := (1 / 2) - (1 / 3)
  let denominator := (3 / 4) + (1 / 8)
  numerator / denominator

theorem fraction_calculation_correct : calculate_fraction = 4 / 21 := 
  by
    sorry

end fraction_calculation_correct_l731_731511


namespace paving_cost_l731_731450

variable (L : ℝ) (W : ℝ) (R : ℝ)

def area (L W : ℝ) := L * W
def cost (A R : ℝ) := A * R

theorem paving_cost (hL : L = 5) (hW : W = 4.75) (hR : R = 900) : cost (area L W) R = 21375 :=
by
  sorry

end paving_cost_l731_731450


namespace probability_sum_less_than_product_l731_731816

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l731_731816


namespace total_selling_price_l731_731401

theorem total_selling_price (original_price discount tax : ℝ) (h₀ : original_price = 120) (h₁ : discount = 0.30) (h₂ : tax = 0.08) :
  let discount_amount := discount * original_price,
      sale_price := original_price - discount_amount,
      tax_amount := tax * sale_price,
      total_selling_price := sale_price + tax_amount
  in total_selling_price = 90.72 :=
by
  sorry

end total_selling_price_l731_731401


namespace intersection_points_count_l731_731975

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0
def line3 (x y : ℝ) : Prop := x = 3
def line4 (x y : ℝ) : Prop := y = 1

theorem intersection_points_count : 
  (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1) ∧ 
  (∃ x2 y2, line3 x2 y2 ∧ line4 x2 y2) → 
  2 :=
sorry

end intersection_points_count_l731_731975


namespace fold_triangle_crease_length_l731_731078

theorem fold_triangle_crease_length :
  ∀ (A B C : ℝ) (a b c : ℝ), a = 5 ∧ b = 12 ∧ c = 13 →
  ∃ crease_length : ℝ, crease_length = 7 / 2 :=
by
  intros A B C a b c h
  obtain ⟨ha, hb, hc⟩ := h
  have H : a = 5 ∧ b = 12 ∧ c = 13 := ⟨ha, hb, hc⟩
  let A := (0 : ℝ)
  let B := (13 : ℝ)
  let C := sqrt (a^2 + (b / 2)^2)
  use (7 / 2)
  sorry

end fold_triangle_crease_length_l731_731078


namespace trisect_AB_intersection_point_l731_731003

noncomputable def exp_func : ℝ → ℝ := λ x, Real.exp x

theorem trisect_AB_intersection_point :
  ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 ∧ x1 = 1 ∧ x2 = 3 → 
  let y1 := exp_func x1,
      y2 := exp_func x2,
      xC := (2 / 3) * x1 + (1 / 3) * x2,
      yC := (2 / 3) * y1 + (1 / 3) * y2,
      y_horizontal := yC,
      x3 := Real.log y_horizontal
  in x3 = Real.log ((2 / 3) * Real.exp 1 + (1 / 3) * Real.exp 3) := 
sorry

end trisect_AB_intersection_point_l731_731003


namespace expand_binom_l731_731530

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l731_731530


namespace arcsin_sqrt_three_over_two_l731_731126

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l731_731126


namespace prob_diff_colors_with_replacement_expectation_variance_white_balls_without_replacement_l731_731770

-- (I) Probability of drawing two balls of different colors with replacement
theorem prob_diff_colors_with_replacement 
  (white_balls black_balls : ℕ) 
  (h_white : white_balls = 2) 
  (h_black : black_balls = 3) :
  let total_balls := white_balls + black_balls in
  let prob_white := (white_balls : ℝ) / total_balls in
  let prob_black := (black_balls : ℝ) / total_balls in
  (prob_white * prob_black + prob_black * prob_white = 12 / 25) :=
begin
  sorry
end

-- (II) Expectation and variance of the number of white balls drawn without replacement
theorem expectation_variance_white_balls_without_replacement
  (white_balls black_balls : ℕ) 
  (h_white : white_balls = 2) 
  (h_black : black_balls = 3) :
  let total_balls := white_balls + black_balls in
  let prob_0_white := (black_balls : ℝ) / total_balls * (black_balls - 1) / (total_balls - 1) in
  let prob_1_white := (black_balls : ℝ) / total_balls * white_balls / (total_balls - 1) + 
                      (white_balls : ℝ) / total_balls * black_balls / (total_balls - 1) in
  let prob_2_white := (white_balls : ℝ) / total_balls * (white_balls - 1) / (total_balls - 1) in
  let E_xi := 0 * prob_0_white + 1 * prob_1_white + 2 * prob_2_white in
  let D_xi := (0 - E_xi)^2 * prob_0_white + (1 - E_xi)^2 * prob_1_white + (2 - E_xi)^2 * prob_2_white in
  (E_xi = 4 / 5) ∧ (D_xi = 9 / 25) :=
begin
  sorry
end

end prob_diff_colors_with_replacement_expectation_variance_white_balls_without_replacement_l731_731770


namespace general_term_formula_l731_731561

def S (n : ℕ) : ℝ := sorry  -- Define the sequence of sum terms

def a (n : ℕ) : ℝ := 
  if n = 1 then -1 
  else 1 / (n * (n - 1))

theorem general_term_formula : ∀ n : ℕ, n ≠ 0 → 
  (a(n) = if n = 1 then -1 else 1 / (n * (n - 1))) :=
by
  intros n hn
  sorry

end general_term_formula_l731_731561


namespace mn_equals_2pq_l731_731109

open EuclideanGeometry

-- Definitions and conditions
variables {A B C M N O Q P : Point}

def equilateral_triangle (A B C : Point) : Prop := ∀ a ∈ {A, B, C}, ∀ b ∈ {A, B, C}, ∀ c ∈ {A, B, C}, 
  a ≠ b → b ≠ c → c ≠ a → dist a b = dist b c ∧ dist b c = dist c a

def semicircle_center_on_side (A B O : Point) : Prop := 
  is_center_of_semicircle O ∧ is_on_segment A B O

def tangent_intersects_sides_at (ABC : triangle) (O M N : Point) : Prop := 
  is_tangent_to_semicircle O M ∧ is_tangent_to_semicircle O N ∧ 
  is_on_segment C M ∧ is_on_segment C N

def line_joins_points_of_tangency (AB AC : side) (Q P : Point) : Prop := 
  is_on_segment AB Q ∧ is_on_segment AC P ∧ 
  line_intersects_segments Q AB P AC

-- Problem statement
theorem mn_equals_2pq 
  (h_equilateral_triangle : equilateral_triangle A B C)
  (h_semicircle_center_on_side : semicircle_center_on_side A B O)
  (h_tangent_intersects_sides_at : tangent_intersects_sides_at (triangle.mk A B C) O M N)
  (h_line_joins_points_of_tangency : line_joins_points_of_tangency (side.mk A B) (side.mk A C) Q P) : 
  dist M N = 2 * dist P Q :=
sorry

end mn_equals_2pq_l731_731109


namespace sum_of_n_values_such_that_fraction_is_integer_l731_731883

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l731_731883


namespace sheilas_hours_mwf_is_24_l731_731705

-- Define Sheila's earning conditions and working hours
def sheilas_hours_mwf (H : ℕ) : Prop :=
  let hours_tu_th := 6 * 2
  let earnings_tu_th := hours_tu_th * 14
  let earnings_mwf := 504 - earnings_tu_th
  H = earnings_mwf / 14

-- The theorem to state that Sheila works 24 hours on Monday, Wednesday, and Friday
theorem sheilas_hours_mwf_is_24 : sheilas_hours_mwf 24 :=
by
  -- Proof is omitted
  sorry

end sheilas_hours_mwf_is_24_l731_731705


namespace mixing_ratio_to_get_2x_zinc_l731_731774

variables (x y : ℝ) -- mass of zinc in the first and second alloy respectively
variables (copper1 copper2 : ℝ) -- mass of copper in the first and second alloy respectively
variable (k : ℝ) -- mixing ratio

-- Conditions
def condition1 := copper1 = 2 * x
def condition2 := copper2 = y / 5

-- Goal: The new alloy should have twice as much zinc as copper.
def new_alloy := (k * x + y) = 2 * (k * 2 * x + y / 5)

theorem mixing_ratio_to_get_2x_zinc : 
  condition1 → condition2 → new_alloy → k = 1 / 2 :=
by 
  intros h1 h2 h3
  sorry

end mixing_ratio_to_get_2x_zinc_l731_731774


namespace probability_perfect_square_l731_731080

theorem probability_perfect_square (p : ℝ) (n : ℕ) 
  (h1 : n ≤ 120) 
  (h2 : ∑ k in finset.Icc 1 60, p = 60 * p ∧ ∑ k in finset.Icc 61 120, 2 * p = 60 * 2 * p) 
  (h3 : 60 * p + 60 * 2 * p = 1) :
  (∑ k in finset.filter (λ k, nat.sqrt k * nat.sqrt k = k) (finset.Icc 1 60), p + 
   ∑ k in finset.filter (λ k, nat.sqrt k * nat.sqrt k = k) (finset.Icc 61 120), 2 * p) = 13 / 180 := 
by
  sorry

end probability_perfect_square_l731_731080


namespace minimum_value_of_A_times_x1_x2_distance_l731_731220

noncomputable def f (x : ℝ) : ℝ := Real.sin (2019 * x + π / 6) + Real.cos (2019 * x - π / 3)

def max_value_of_f : ℝ := 2

def period_of_function : ℝ := 2 * π / 2019

def x1_x2_min_distance : ℝ := π / 2019

def A : ℝ := max_value_of_f

theorem minimum_value_of_A_times_x1_x2_distance (x1 x2 : ℝ) (hx : ∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) : A * |x1 - x2| = 2 * π / 2019 :=
by {
  sorry
}

end minimum_value_of_A_times_x1_x2_distance_l731_731220


namespace math_problem_l731_731180

variable (a b c m : ℝ)

-- Quadratic equation: y = ax^2 + bx + c
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Opens downward
axiom a_neg : a < 0
-- Passes through A(1, 0)
axiom passes_A : quadratic a b c 1 = 0
-- Passes through B(m, 0) with -2 < m < -1
axiom passes_B : quadratic a b c m = 0
axiom m_range : -2 < m ∧ m < -1

-- Prove the conclusions
theorem math_problem : b < 0 ∧ (a + b + c = 0) ∧ (a * (m+1) - b + c > 0) ∧ ¬(4 * a * c - b^2 > 4 * a) :=
by
  sorry

end math_problem_l731_731180


namespace problem1_solution_l731_731463

theorem problem1_solution (n : ℕ) (h : n ≥ 2) :
  ∃ (a : Fin n → ℝ), { |a i - a j| | i j : Fin n, i < j } = {1, 2, ..., (n * (n - 1)) / 2} →
  n = 2 ∨ n = 3 ∨ n = 4 :=
by
  intros a ha hn
  sorry

end problem1_solution_l731_731463


namespace even_function_sum_eval_l731_731112

variable (v : ℝ → ℝ)

theorem even_function_sum_eval (h_even : ∀ x : ℝ, v x = v (-x)) :
    v (-2.33) + v (-0.81) + v (0.81) + v (2.33) = 2 * (v 2.33 + v 0.81) :=
by
  sorry

end even_function_sum_eval_l731_731112


namespace incorrect_calculation_l731_731722

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l731_731722


namespace domain_of_tan_l731_731169

theorem domain_of_tan :
    ∀ k : ℤ, ∀ x : ℝ,
    (x > (k * π / 2 - π / 8) ∧ x < (k * π / 2 + 3 * π / 8)) ↔
    2 * x - π / 4 ≠ k * π + π / 2 :=
by
  intro k x
  sorry

end domain_of_tan_l731_731169


namespace nancy_shoes_l731_731684

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l731_731684


namespace solution_set_inequalities_l731_731404

theorem solution_set_inequalities (x : ℝ) :
  (2 * x + 3 ≥ -1) ∧ (7 - 3 * x > 1) ↔ (-2 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_inequalities_l731_731404


namespace greatest_possible_median_l731_731912

theorem greatest_possible_median : 
  ∀ (k m r s t : ℕ),
    k < m → m < r → r < s → s < t →
    (k + m + r + s + t = 90) →
    (t = 40) →
    (r = 23) :=
by
  intros k m r s t h1 h2 h3 h4 h_sum h_t
  sorry

end greatest_possible_median_l731_731912


namespace diana_age_is_8_l731_731985

noncomputable def age_of_grace_last_year : ℕ := 3
noncomputable def age_of_grace_today : ℕ := age_of_grace_last_year + 1
noncomputable def age_of_diana_today : ℕ := 2 * age_of_grace_today

theorem diana_age_is_8 : age_of_diana_today = 8 :=
by
  -- The proof would go here
  sorry

end diana_age_is_8_l731_731985


namespace probability_sum_less_than_product_is_5_div_9_l731_731810

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l731_731810


namespace count_true_forms_l731_731177

-- Definitions of lines and parallelism
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

-- Proposition p
def p (l1 l2 : Line) : Prop :=
  parallel l1 l2 → l1.a * l2.b - l2.a * l1.b = 0

-- Converse of Proposition p
def converse_p (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0 → parallel l1 l2

-- Negation of Proposition p
def not_p (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2 → l1.a * l2.b - l2.a * l1.b = 0)

-- Contrapositive of Proposition p
def contrapositive_p (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b ≠ 0 → ¬parallel l1 l2

-- The function f(p) to count the number of true propositions
def f (l1 l2 : Line) : ℕ :=
  [p l1 l2, converse_p l1 l2, not_p l1 l2, contrapositive_p l1 l2].count (· = true)

-- The main theorem to prove
theorem count_true_forms (l1 l2 : Line) : f l1 l2 = 2 :=
  sorry

end count_true_forms_l731_731177


namespace sum_of_n_values_such_that_fraction_is_integer_l731_731885

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l731_731885


namespace quadratic_sum_l731_731148

theorem quadratic_sum (b c : ℤ) : 
  (∃ b c : ℤ, (x^2 - 10*x + 15 = 0) ↔ ((x + b)^2 = c)) → b + c = 5 :=
by
  intros h
  sorry

end quadratic_sum_l731_731148


namespace solutions_to_equation_l731_731167

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 10*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 12*x - 8)) = 0

theorem solutions_to_equation :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2) :=
sorry

end solutions_to_equation_l731_731167


namespace average_marks_for_class_l731_731750

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l731_731750


namespace area_of_fountain_l731_731485

/--
  Assume we have a circular fountain with center \( C \).
  We place a 20-foot plank from \( A \) to \( B \) and
  a 12-foot plank from \( D \) to \( C \), where \( D \) is
  the midpoint of \( \overline{AB} \). Prove that the area
  of the circular base of the fountain is \( 244\pi \) square feet.
-/
theorem area_of_fountain (A B C D : ℝ → ℝ → ℝ)
  (hAB : dist A B = 20)
  (hD : D = (A + B) / 2)
  (hCD : dist D C = 12) :
  area_of_circle (dist C A) = 244 * π := 
  sorry

end area_of_fountain_l731_731485


namespace alok_total_payment_l731_731098

def cost_of_chapatis : Nat := 16 * 6
def cost_of_rice_plates : Nat := 5 * 45
def cost_of_mixed_vegetable_plates : Nat := 7 * 70
def total_cost : Nat := cost_of_chapatis + cost_of_rice_plates + cost_of_mixed_vegetable_plates

theorem alok_total_payment :
  total_cost = 811 := by
  unfold total_cost
  unfold cost_of_chapatis
  unfold cost_of_rice_plates
  unfold cost_of_mixed_vegetable_plates
  calc
    16 * 6 + 5 * 45 + 7 * 70 = 96 + 5 * 45 + 7 * 70 := by rfl
                      ... = 96 + 225 + 7 * 70 := by rfl
                      ... = 96 + 225 + 490 := by rfl
                      ... = 96 + (225 + 490) := by rw Nat.add_assoc
                      ... = (96 + 225) + 490 := by rw Nat.add_assoc
                      ... = 321 + 490 := by rfl
                      ... = 811 := by rfl

end alok_total_payment_l731_731098


namespace locus_of_points_l731_731636

-- Given points A (-m, 0), B (n, 0) and origin O (0, 0) in Cartesian coordinate system

variables {m n : ℝ} -- m and n are real numbers

-- Define the equation of the locus of point C as described
theorem locus_of_points
  (C : ℝ × ℝ) -- C(x, y) is a point in the 2D plane
  (angle_eq : ∀ O A B : ℝ × ℝ, ∠ A O B = ∠ B O A) -- Angle at point C of the lines AO and OB being equal
  (A : ℝ × ℝ := (-m, 0))
  (B : ℝ × ℝ := (n, 0))
  (O : ℝ × ℝ := (0, 0)) :
  (C.fst - (m*n / (m - n)))^2 + C.snd^2 = (m*n / (m - n))^2 :=
begin
  sorry
end

end locus_of_points_l731_731636


namespace length_of_segment_DB_l731_731285

-- Definitions for the problem
variables (A B C D : Type) [EuclideanGeometry A]
variables (AC AD AB : ℝ)
variable (angle_ABC : A → B → C → ∠ ABC = π / 2)
variable (angle_ADB : A → D → B → ∠ ADB = π / 2)
variable (AC_val : AC = 20)
variable (AD_val : AD = 6)
variable (AB_val : AB = 10)

-- The Lean statement of the proof problem
theorem length_of_segment_DB :
  ∃ (DB : ℝ), ∠ ABC = π / 2 ∧ ∠ ADB = π / 2 ∧ AC = 20 ∧ AD = 6 ∧ AB = 10 ∧ DB = 8 :=
by
  sorry

end length_of_segment_DB_l731_731285


namespace incorrect_calculation_l731_731718

noncomputable def ξ : ℝ := 3
noncomputable def η : ℝ := 5

def T (ξ η : ℝ) : ℝ := min ξ η

theorem incorrect_calculation : E[T(ξ, η)] ≤ 3 := by
  sorry

end incorrect_calculation_l731_731718


namespace range_of_a_l731_731237

noncomputable def f (x : ℝ) : ℝ := Real.log (2 + 3 * x) - (3 / 2) * x^2
noncomputable def f' (x : ℝ) : ℝ := (3 / (2 + 3 * x)) - 3 * x
noncomputable def valid_range (a : ℝ) : Prop := 
∀ x : ℝ, (1 / 6) ≤ x ∧ x ≤ (1 / 3) → |a - Real.log x| + Real.log (f' x + 3 * x) > 0

theorem range_of_a : { a : ℝ | valid_range a } = { a : ℝ | a ≠ Real.log (1 / 3) } := 
sorry

end range_of_a_l731_731237


namespace total_cost_after_discounts_l731_731152

theorem total_cost_after_discounts 
    (price_iphone : ℝ)
    (discount_iphone : ℝ)
    (price_iwatch : ℝ)
    (discount_iwatch : ℝ)
    (cashback_percentage : ℝ) :
    (price_iphone = 800) →
    (discount_iphone = 0.15) →
    (price_iwatch = 300) →
    (discount_iwatch = 0.10) →
    (cashback_percentage = 0.02) →
    let discounted_iphone := price_iphone * (1 - discount_iphone),
        discounted_iwatch := price_iwatch * (1 - discount_iwatch),
        total_discounted := discounted_iphone + discounted_iwatch,
        cashback := total_discounted * cashback_percentage 
    in total_discounted - cashback = 931 :=
by {
  intros,
  sorry
}

end total_cost_after_discounts_l731_731152


namespace incorrect_lifetime_calculation_l731_731726

-- Define expectation function
noncomputable def expectation (X : ℝ) : ℝ := sorry

-- We define the lifespans
variables (xi eta : ℝ)
-- Expected lifespan of the sensor and transmitter
axiom exp_xi : expectation xi = 3
axiom exp_eta : expectation eta = 5

-- Define the lifetime of the device
noncomputable def T := min xi eta

-- Given conditions
theorem incorrect_lifetime_calculation :
  expectation T ≤ 3 → 3 + (2 / 3) > 3 → false := 
sorry

end incorrect_lifetime_calculation_l731_731726


namespace absent_minded_scientist_mistake_l731_731730

theorem absent_minded_scientist_mistake (ξ η : ℝ) (h₁ : E ξ = 3) (h₂ : E η = 5) (h₃ : E (min ξ η) = 3 + 2/3) : false :=
by
  sorry

end absent_minded_scientist_mistake_l731_731730


namespace Lisa_mean_score_l731_731416

theorem Lisa_mean_score {a b c d e f : ℕ} {mean_john : ℕ}
  (h_scores : {a, b, c, d, e, f} = {82, 88, 91, 95, 96, 97})
  (h_cnt : ({a, b, c} : set ℕ).card = 3 ∧ ({d, e, f} : set ℕ).card = 3)
  (h_mean_john : mean_john = 91)
  (h_sum_john : a + b + c = 3 * mean_john) :
  (d + e + f) / 3 = 92 :=
by
  sorry

end Lisa_mean_score_l731_731416


namespace rectangle_diagonal_length_l731_731645

theorem rectangle_diagonal_length
    (PQ QR : ℝ) (RT RU ST : ℝ) (Area_RST : ℝ)
    (hPQ : PQ = 8) (hQR : QR = 10)
    (hRT_RU : RT = RU)
    (hArea_RST: Area_RST = (1/5) * (PQ * QR)) :
    ST = 8 :=
by
  sorry

end rectangle_diagonal_length_l731_731645


namespace complex_solutions_count_l731_731542

theorem complex_solutions_count : 
  let numerator := (λ z : ℂ, z^4 - 1)
  let denominator := (λ z : ℂ, z^2 + z - 2)
  (∃ (z : ℂ), numerator z = 0 ∧ denominator z ≠ 0) → 3 :=
by
  let numerator := (λ z : ℂ, z^4 - 1)
  let denominator := (λ z : ℂ, z^2 + z - 2)
  -- We need to prove the statement here
  sorry

end complex_solutions_count_l731_731542


namespace area_of_triangle_PQR_l731_731655

-- Define the conditions required for the problem
variables (P Q R S T : Type) [triangle P Q R] (PS : altitude P S)
  (PQ_eq_QR : PQ = QR) (ST_eq_12 : ST = 12)
  (tan_RTS_Geom_Prog : ∃ a b c : ℝ, tan (angle R S T) = a ∧ tan (angle P S T) = b ∧ tan (angle Q S T) = c ∧ is_geom_prog [a, b, c])
  (cot_Arithmetic_Prog : ∃ p q r : ℝ, cot (angle P S T) = p ∧ cot (angle R S T) = q ∧ cot (angle P S R) = r ∧ is_arith_prog [p, q, r])

-- Define the final statement asserting the area of triangle P Q R equals 36
theorem area_of_triangle_PQR :
  area P Q R = 36 :=
sorry

end area_of_triangle_PQR_l731_731655


namespace equilateral_triangle_area_l731_731550

-- Definition of the problem conditions
def perpendiculars (a b c : ℕ) := (a = 2 ∧ b = 3 ∧ c = 4)

-- The main theorem statement
theorem equilateral_triangle_area (a b c : ℕ) (h : ℕ) :
  perpendiculars a b c →
  (h = 9) →
  let side := 6 * Real.sqrt 3 in
  let area := (side * h) / 2 in
  area = 27 * Real.sqrt 3 :=
sorry

end equilateral_triangle_area_l731_731550


namespace train_passing_times_l731_731854

noncomputable def time_to_pass (length : ℕ) (speed_kmph : ℤ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  length / speed_mps

theorem train_passing_times :
  let time_A := time_to_pass 150 50
  let time_B := time_to_pass 200 40
  time_A ≈ 10.80 ∧ time_B ≈ 18.00 ∧ (time_B - time_A) ≈ 7.20 :=
by
  sorry

end train_passing_times_l731_731854


namespace general_formula_l731_731757

noncomputable theory

open Nat

def a : ℕ → ℕ
| 0     => 0           -- Note that we're using 0 for a1 since Lean uses 0-based indexing
| (n+1) => a n + 2 * n

theorem general_formula (n : ℕ) : a n = n * (n - 1) := by
  induction n with
  | zero      => simp
  | succ n ih => simp [a, ih]

end general_formula_l731_731757


namespace range_f_neg2_l731_731187

variable {a b : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x

theorem range_f_neg2 
  (h1 : -1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 2 ≤ f (1) ∧ f (1) ≤ 4) :
  -1 ≤ f (-2) ∧ f (-2) ≤ 10 := by
  sorry

end range_f_neg2_l731_731187


namespace max_binom_coeff_800_l731_731532

theorem max_binom_coeff_800 :
  ∃ k, (k = 200 ∧ ∀ n, 0 ≤ n ∧ n ≤ 800 → (∃ m, (m = n → ∀ l, 0 ≤ l ∧ l < m → B_l < B_m) ∧ B_k ≤ B_m)) :=
by
  let B := λ k : ℕ, if k ≤ 800 then nat.choose 800 k * (0.3)^k else 0
  let k := 200
  existsi k
  split
  . exact rfl
  . intro n
    intro hn
    existsi n
    split
    . intro heq
      intros l hl
      sorry
    . sorry

end max_binom_coeff_800_l731_731532


namespace train_speed_l731_731443

theorem train_speed
  (length_of_train : ℝ) 
  (time_to_cross : ℝ) 
  (train_length_is_140 : length_of_train = 140)
  (time_is_6 : time_to_cross = 6) :
  (length_of_train / time_to_cross) = 23.33 :=
sorry

end train_speed_l731_731443


namespace arithmetic_sequence_term_number_l731_731282

theorem arithmetic_sequence_term_number
  (a : ℕ → ℤ)
  (ha1 : a 1 = 1)
  (ha2 : a 2 = 3)
  (n : ℕ)
  (hn : a n = 217) :
  n = 109 :=
sorry

end arithmetic_sequence_term_number_l731_731282


namespace solution_set_l731_731590

noncomputable def f (x : ℝ) : ℝ := sorry

axiom A1 : ∀ x : ℝ, 0 < x → x * (f' x) > 1
axiom A2 : f 1 = 0

theorem solution_set : {x : ℝ | 0 < x ∧ f x ≤ Real.log x} = Set.Ioc 0 1 := 
by sorry

end solution_set_l731_731590


namespace one_planet_unwatched_l731_731776

theorem one_planet_unwatched {P : Type} [Fintype P] (n : ℕ) (h_odd : Fintype.card P = 2 * n + 1) 
  (d : P → P → ℝ) (h_diff : ∀ p1 p2 : P, p1 ≠ p2 → d p1 p2 ≠ d p1 p2)
  (h_closest : ∀ (p : P), ∃ (q : P), (q ≠ p) ∧ (∀ r : P, r ≠ p → d p q < d p r))
  (h_ast : ∀ (p : P), ∃ (q : P), q ≠ p ∧ (∀ r : P, r ≠ p → d p q < d p r)) : 
  ∃ (p : P), ¬ ∃ (q : P), h_closest q p :=
by
  sorry

end one_planet_unwatched_l731_731776
