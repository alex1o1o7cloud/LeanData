import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Linear
import Mathlib.Algebra.Order
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Square
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Bool.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Init.Data.Int.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace part_I_part_II_part_III_l692_692540

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem part_I (a : ℝ) : (∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f a x ≥ f a 1) ↔ a ≥ -1/2 :=
by
  sorry

theorem part_II : ∀ x : ℝ, f (-Real.exp 1) x + 2 ≤ 0 :=
by
  sorry

theorem part_III : ¬ ∃ x : ℝ, |f (-Real.exp 1) x| = Real.log x / x + 3 / 2 :=
by
  sorry

end part_I_part_II_part_III_l692_692540


namespace comics_in_box_l692_692017

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l692_692017


namespace problem_C_and_l_intersection_l692_692304
open Real

theorem problem_C_and_l_intersection
  (C : ℝ → ℝ → Prop)
  (polar_eq : ∀ (r θ : ℝ), r^2 = 12 / (5 - cos (2 * θ)) ↔ C (r * cos θ) (r * sin θ))
  (l : ℝ → ℝ × ℝ)
  (parametric_eq : ∀ t, l t = (1 + sqrt 2 / 2 * t, sqrt 2 / 2 * t))
  (M : ℝ × ℝ)
  (M_coord : M = (0, -1))
  (intersection_pts : ∃ (A B : ℝ × ℝ), (∃ tA tB, l tA = A ∧ l tB = B ∧ C A.1 A.2 ∧ C B.1 B.2))
  :
  let MA (A : ℝ × ℝ) := sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  in ∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 →
  ∃ t1 t2, l t1 = A ∧ l t2 = B → 
  (|MA A| + |MA B|) / (|MA A| * |MA B|) = 4 * sqrt 3 / 3 :=
sorry

end problem_C_and_l_intersection_l692_692304


namespace axis_center_symmetry_sine_shifted_l692_692047
  noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 3 * Real.pi / 4 + k * Real.pi

  noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ := (Real.pi / 4 + k * Real.pi, 0)

  theorem axis_center_symmetry_sine_shifted :
    ∀ (k : ℤ),
    ∃ x y : ℝ,
      (x = axis_of_symmetry k) ∧ (y = 0) ∧ (y, 0) = center_of_symmetry k := 
  sorry
  
end axis_center_symmetry_sine_shifted_l692_692047


namespace calculation_result_l692_692028

theorem calculation_result : (18 * 23 - 24 * 17) / 3 + 5 = 7 :=
by
  sorry

end calculation_result_l692_692028


namespace proof_problem_l692_692556

variable {A B C a b c : ℝ}
variable {m n : ℝ × ℝ}

-- Given conditions
def vec_m := (Real.cos A, Real.sin B)
def vec_n := (Real.cos B, -Real.sin A)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Mathematical proof problem
theorem proof_problem :
  vec_m = (Real.cos A, Real.sin B) ∧
  vec_n = (Real.cos B, -Real.sin A) ∧
  dot_product vec_m vec_n = Real.cos (2 * C) ∧
  A + B + C = Real.pi ∧
  2 * c = a + b ∧
  a * b * Real.cos C = 18 →
  C = Real.pi / 3 ∧ c = 6 := by
  sorry

end proof_problem_l692_692556


namespace opposite_pairs_l692_692452

theorem opposite_pairs :
  ∃ (x y : ℤ), (x = -5 ∧ y = -(-5)) ∧ (x = -y) ∧ (
    (¬ (∃ (a b : ℤ), (a = -2 ∧ b = 1/2) ∧ (a = -b))) ∧ 
    (¬ (∃ (c d : ℤ), (c = | -1 | ∧ d = 1) ∧ (c = -d))) ∧
    (¬ (∃ (e f : ℤ), (e = (-3)^2 ∧ f = 3^2) ∧ (e = -f)))
  ) :=
by
  sorry

end opposite_pairs_l692_692452


namespace measure_angle_EDZ_is_30_l692_692603

-- Definitions based on the given conditions
variables {X Y Z D E : Type} -- Points representing vertices of the triangle and additional points
variable (P : Triangle X Y Z) -- Define P as the triangle formed by points X, Y, and Z
variable (XZ_YZ_equal : Side P XZ = Side P YZ) -- Side XZ is equal to side YZ
variable (angle_DYZ_30 : Angle P D YZ = 30) -- The measure of angle DYZ is 30 degrees
variable (DY_parallel_XZ : Parallel DY XZ) -- Line DY is parallel to line XZ
variable (E_midpoint_XY : Midpoint E XY) -- Point E is the midpoint of segment XY

-- Proof: measure of angle EDZ is 30 degrees
theorem measure_angle_EDZ_is_30 : Angle E D Z = 30 := sorry

end measure_angle_EDZ_is_30_l692_692603


namespace factor_sum_abs_eq_twelve_l692_692390

variable {x h b c d : Int}

def polynomial := 6 * x ^ 2 + x - 12

theorem factor_sum_abs_eq_twelve :
  (polynomial = (h * x + b) * (c * x + d)) →
  |h| + |b| + |c| + |d| = 12 := by
sorry

end factor_sum_abs_eq_twelve_l692_692390


namespace car_travel_distance_l692_692921

-- Define the conditions.
def train_speed : ℝ := 90  -- Train speed in miles per hour
def car_ratio : ℝ := 2 / 3  -- Car's speed as a fraction of train's speed
def time_minutes : ℝ := 40  -- Time in minutes

-- Convert 40 minutes to hours.
def time_hours : ℝ := time_minutes / 60

-- Calculate the car's speed in miles per hour.
def car_speed : ℝ := car_ratio * train_speed

-- Calculate the distance the car travels.
def car_distance : ℝ := car_speed * time_hours

-- The theorem stating the distance calculation.
theorem car_travel_distance : car_distance = 40 := by
  sorry

end car_travel_distance_l692_692921


namespace boat_speed_still_water_l692_692970

theorem boat_speed_still_water : 
  ∀ (B S : ℝ), (B + S = 32 ∧ B - S = 12) → B = 22 :=
by
  intros B S h
  cases h with h1 h2
  sorry

end boat_speed_still_water_l692_692970


namespace polynomial_even_coeff_sum_zero_l692_692223

noncomputable def polynomial_coefficients {α : Type*} [Field α] : (α → α) → List α 
| f := (List.pmap (λ n H, f (n : α)).val) (multiset.range 7)

theorem polynomial_even_coeff_sum_zero 
  {R : Type*} [CommRing R] (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : R)
  (h : ∀ (x : R), (x - 1)^6 = a_0 * x^6 + a_1 * x^5 + a_2 * x^4 + a_3 * x^3 + a_4 * x^2 + a_5 * x + a_6) :
  a_0 + a_2 + a_4 + a_6 = 0 := 
by 
  sorry

end polynomial_even_coeff_sum_zero_l692_692223


namespace range_of_a_l692_692562

theorem range_of_a {x a : ℝ} (h1 : abs (2*x - 5) ≤ 4) : x < a → a > 9 / 2 :=
by 
  -- Step 1: From h1, derive the inequality 1/2 ≤ x ≤ 9/2
  have h2 : 1 / 2 ≤ x ∧ x ≤ 9 / 2 := abs_le.1 h1
  -- Step 2: Use h2 to prove the necessity condition for a
  sorry

end range_of_a_l692_692562


namespace problem_correct_answer_l692_692764

theorem problem_correct_answer :
  (∀ (P L : Type) (passes_through_point : P → L → Prop) (parallel_to : L → L → Prop),
    (∀ (l₁ l₂ : L) (p : P), passes_through_point p l₁ ∧ ¬ passes_through_point p l₂ → (∃! l : L, passes_through_point p l ∧ parallel_to l l₂)) ->
  (∃ (l₁ l₂ : L) (A : P), passes_through_point A l₁ ∧ ¬ passes_through_point A l₂ ∧ ∃ l : L, passes_through_point A l ∧ parallel_to l l₂) ) :=
sorry

end problem_correct_answer_l692_692764


namespace integral_evaluation_l692_692097

theorem integral_evaluation : ∫ x in 0..5, (2 * x - 4) = 5 :=
by
  sorry

end integral_evaluation_l692_692097


namespace circumcircle_center_lies_on_circle_l692_692731

section circumcircle_center

variables {A B C D E F O : Type*}
variables [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O]

-- Conditions
-- 1. The trapezoid ABCD (AB || CD) is inscribed in a circle ω
axiom inscribed_trapezoid (A B C D : Type*) (ω : Type*) [is_circle ω] [is_trapezoid ABCD] : 
  inscribed_in ω ABCD

-- 2. Point E such that BC = BE and E is on the ray beyond C along DC
axiom E_on_ray (B C D E : Type*) : on_ray D C E ∧ BC = BE

-- 3. The line BE intersects the circle ω again at F, which lies outside the segment BE
axiom BE_intersects_again (B E ω F : Type*) [is_circle ω] [is_line B E] : intersects_again_in_circle B E ω F ∧ outside_segment B E F

-- Assertion to be proved
theorem circumcircle_center_lies_on_circle (A B C D E F O ω : Type*) [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O] [is_circle ω] : 
  center_of_circumcircle_lies_on_circle (triangle C E F) ω :=
by {
  assume inscribed_trapezoid ABCD ω,
  assume E_on_ray B C D E,
  assume BE_intersects_again B E ω F,
  exact sorry
}

end circumcircle_center

end circumcircle_center_lies_on_circle_l692_692731


namespace loader_max_weight_l692_692420

/--
A loader has a waggon and a little cart. The waggon can carry up to 1000 kg,
and the cart can carry only up to 1 kg. A finite number of sacks with sand 
lie in a storehouse. It is known that their total weight is more than 1001 kg, 
while each sack weighs not more than 1 kg. Prove that the maximum weight of sand 
that the loader can carry in the waggon and the cart is 1001 kg.
-/
theorem loader_max_weight (W : ℝ) (n : ℕ) (w : ℕ → ℝ)
  (hW : W > 1001) (hw : ∀ i, w i ≤ 1) (hsum : ∑ i in finset.range n, w i = W) :
  ∃ (sacks_waggon sacks_cart : ℕ), sacks_waggon ≤ 1000 ∧ sacks_cart ≤ 1 ∧
  (∑ i in finset.range sacks_waggon, w i) + (∑ i in finset.range sacks_cart, w i) = 1001 :=
by sorry

end loader_max_weight_l692_692420


namespace division_equivalent_l692_692757

def division_to_fraction (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 ∧ 0 ≤ a ∧ 0 ≤ b → a / b = (a * 1000) / (b * 1000) :=
by
  intros h
  field_simp
  
theorem division_equivalent (h : 0 ≤ 0.08 ∧ 0 ≤ 0.002 ∧ 0.08 ≠ 0 ∧ 0.002 ≠ 0) :
  0.08 / 0.002 = 40 :=
by
  have := division_to_fraction 0.08 0.002 h
  norm_num at this
  exact this

end division_equivalent_l692_692757


namespace probability_no_3x3_red_square_l692_692066

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l692_692066


namespace total_collection_l692_692418

theorem total_collection (n : ℕ) (c : ℕ) (h1 : n = 93) (h2 : c = 93) : (n * c) / 100 = 86.49 :=
by
  sorry

end total_collection_l692_692418


namespace aston_comics_l692_692015

theorem aston_comics (total_pages_on_floor : ℕ) (pages_per_comic : ℕ) (untorn_comics_in_box : ℕ) :
  total_pages_on_floor = 150 →
  pages_per_comic = 25 →
  untorn_comics_in_box = 5 →
  (total_pages_on_floor / pages_per_comic + untorn_comics_in_box) = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end aston_comics_l692_692015


namespace minimum_real_roots_of_polynomial_l692_692230

open Real

noncomputable def g (x : ℝ) : ℝ := sorry -- since exact polynomial is not given, left as a placeholder

theorem minimum_real_roots_of_polynomial :
  ∃ (s : list ℂ), 
    list.length s = 2020 ∧ 
    (∀ i, i < 2020 → s.nth i ≠ none) ∧
    ∃ (distinct_mags : finset ℝ), distinct_mags.card = 1010 ∧ 
    (∀ i, i < 2020 → abs (s.nth_le i sorry) ∈ distinct_mags) ∧ 
    (list.sum s = 0) →
    (g ∈ set.Polynomial) ∧ g.degree = 2020 ∧ 
    (∃ r : list ℝ, r.length = 10 ∧ 
    (∀ i, i < 10 → is_real_root g (r.nth_le i sorry))) :=
begin
  sorry
end

end minimum_real_roots_of_polynomial_l692_692230


namespace Martha_needs_54_cakes_l692_692252

theorem Martha_needs_54_cakes :
  let n_children := 3
  let n_cakes_per_child := 18
  let n_cakes_total := 54
  n_cakes_total = n_children * n_cakes_per_child :=
by
  sorry

end Martha_needs_54_cakes_l692_692252


namespace min_L_Trominos_l692_692656

theorem min_L_Trominos (x y : ℕ) :
  (2020 * 2021 % 4 = 0) →
  (4 * x + 4 * y = 2020 * 2021) →
  (2020 * 1010 ≥ 2 * x + y) →
  y = 1010 :=
by
  sorry

end min_L_Trominos_l692_692656


namespace find_multiplier_l692_692379

theorem find_multiplier (x : ℕ) (h₁ : 3 * x = (26 - x) + 26) (h₂ : x = 13) : 3 = 3 := 
by 
  sorry

end find_multiplier_l692_692379


namespace labor_day_to_national_day_l692_692561

theorem labor_day_to_national_day :
  let labor_day := 1 -- Monday is represented as 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  (labor_day + total_days % 7) % 7 = 0 := -- Since 0 corresponds to Sunday modulo 7
by
  let labor_day := 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  have h1 : (labor_day + total_days % 7) % 7 = ((1 + (31 * 3 + 30 * 2) % 7) % 7) := by rfl
  sorry

end labor_day_to_national_day_l692_692561


namespace intersection_of_P_and_Q_l692_692547

def P : Set ℕ := {0, 1, 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 3^x}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := 
by {
  sorry
}

end intersection_of_P_and_Q_l692_692547


namespace isosceles_triangle_base_length_l692_692897

theorem isosceles_triangle_base_length
  (a b c : ℕ)
  (h_iso : a = b)
  (h_perimeter : a + b + c = 62)
  (h_leg_length : a = 25) :
  c = 12 :=
by
  sorry

end isosceles_triangle_base_length_l692_692897


namespace division_of_decimals_l692_692752

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l692_692752


namespace evaluation_l692_692869
-- Import the entire Mathlib library

-- Define the operations triangle and nabla
def triangle (a b : ℕ) : ℕ := 3 * a + 2 * b
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

-- The proof statement
theorem evaluation : triangle 2 (nabla 3 4) = 42 :=
by
  -- Provide a placeholder for the proof
  sorry

end evaluation_l692_692869


namespace opposite_of_neg2023_l692_692702

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l692_692702


namespace domain_g_k_value_is_1_intersection_range_l692_692541

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def g (a : ℝ) (x : ℝ) : ℝ :=
  log (a * 2^x - (4 / 3) * a) / log 2

def f (k : ℝ) (x : ℝ) : ℝ :=
  log (4^x + 1) / log 2 - k * x

theorem domain_g (a : ℝ) (h : a > 0) : ∀ x : ℝ, x > log (4 / 3) / log 2 → g a x ≥ 0 :=
sorry

theorem k_value_is_1 (k : ℝ) : is_even (λ x, log (4^x + 1) / log 2 - k * x) → k = 1 :=
sorry

theorem intersection_range (a : ℝ) (h : a > 0) : 
  (∃! x : ℝ, x > log (4 / 3) / log 2 ∧ f 1 x = g a x) ↔ a > 1 :=
sorry

end domain_g_k_value_is_1_intersection_range_l692_692541


namespace cotton_candy_ratio_l692_692704

-- Define the constants
def caramel_price : ℝ := 3
def candy_bar_price : ℝ := 2 * caramel_price

-- Define the total cost of the items
def total_cost (CC : ℝ) : ℝ := 6 * candy_bar_price + 3 * caramel_price + CC

-- Prove the required ratio of cotton candy price to 4 candy bars price
theorem cotton_candy_ratio (CC : ℝ) (h : total_cost CC = 57) :
  CC / (4 * candy_bar_price) = 1 / 2 :=
by {
  -- Definitions based on the conditions
  have C_def := caramel_price,
  have B_def := candy_bar_price,
  -- Use the given total cost equation to solve for CC
  calc
  CC = 12 := by sorry
}


end cotton_candy_ratio_l692_692704


namespace aurelia_percentage_l692_692001

variables (P : ℝ)

theorem aurelia_percentage (h1 : 2000 + (P / 100) * 2000 = 3400) : 
  P = 70 :=
by
  sorry

end aurelia_percentage_l692_692001


namespace jeremy_money_left_l692_692216

theorem jeremy_money_left (computer_cost : ℕ) (accessories_percentage : ℕ) (factor : ℕ)
  (h1 : computer_cost = 3000)
  (h2 : accessories_percentage = 10)
  (h3 : factor = 2) :
  let accessories_cost := (accessories_percentage * computer_cost) / 100 in
  let total_money_before := factor * computer_cost in
  let total_spent := computer_cost + accessories_cost in
  let money_left := total_money_before - total_spent in
  money_left = 2700 :=
by
  sorry

end jeremy_money_left_l692_692216


namespace smallest_positive_period_of_f_range_of_f_sin_2x0_eq_l692_692539

noncomputable def f (x : ℝ) : ℝ :=
  1 - cos x ^ 2 + 2 * sqrt 3 * sin x * cos x - 1 / 2 * cos (2 * x)

theorem smallest_positive_period_of_f :
  is_periodic f π :=
sorry

theorem range_of_f :
  ∀ (x : ℝ), -3/2 ≤ f x ∧ f x ≤ 5/2 :=
sorry

theorem sin_2x0_eq :
  ∀ (x0 : ℝ), 0 ≤ x0 ∧ x0 ≤ π / 2 ∧ f x0 = 0 → sin (2 * x0) = (sqrt 15 - sqrt 3) / 8 :=
sorry

end smallest_positive_period_of_f_range_of_f_sin_2x0_eq_l692_692539


namespace final_cards_l692_692220

def initial_cards : ℝ := 47.0
def lost_cards : ℝ := 7.0

theorem final_cards : (initial_cards - lost_cards) = 40.0 :=
by
  sorry

end final_cards_l692_692220


namespace remainder_of_division_l692_692113

noncomputable def f (x : ℚ) := x^2007 + 1
noncomputable def g (x : ℚ) := x^6 - x^4 + x^2 - 1

theorem remainder_of_division :
  ∀ x : ℚ, euclidean_domain.mod (f x) (g x) = -x + 1 :=
by
  sorry

end remainder_of_division_l692_692113


namespace shared_property_l692_692005

-- Definitions of the shapes
structure Parallelogram where
  sides_equal    : Bool -- Parallelograms have opposite sides equal but not necessarily all four.

structure Rectangle where
  sides_equal    : Bool -- Rectangles have opposite sides equal.
  diagonals_equal: Bool

structure Rhombus where
  sides_equal: Bool -- Rhombuses have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a rhombus are perpendicular.

structure Square where
  sides_equal: Bool -- Squares have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a square are perpendicular.
  diagonals_equal: Bool -- Diagonals of a square are equal in length.

-- Definitions of properties
def all_sides_equal (p1 p2 p3 p4 : Parallelogram) := p1.sides_equal ∧ p2.sides_equal ∧ p3.sides_equal ∧ p4.sides_equal
def diagonals_equal (r1 r2 r3 : Rectangle) (s1 s2 : Square) := r1.diagonals_equal ∧ r2.diagonals_equal ∧ s1.diagonals_equal ∧ s2.diagonals_equal
def diagonals_perpendicular (r1 : Rhombus) (s1 s2 : Square) := r1.diagonals_perpendicular ∧ s1.diagonals_perpendicular ∧ s2.diagonals_perpendicular
def diagonals_bisect_each_other (p1 p2 p3 p4 : Parallelogram) (r1 : Rectangle) (r2 : Rhombus) (s1 s2 : Square) := True -- All these shapes have diagonals that bisect each other.

-- The statement we need to prove
theorem shared_property (p1 p2 p3 p4 : Parallelogram) (r1 r2 : Rectangle) (r3 : Rhombus) (s1 s2 : Square) : 
  (diagonals_bisect_each_other p1 p2 p3 p4 r1 r3 s1 s2) :=
by
  sorry

end shared_property_l692_692005


namespace division_equivalent_l692_692755

def division_to_fraction (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 ∧ 0 ≤ a ∧ 0 ≤ b → a / b = (a * 1000) / (b * 1000) :=
by
  intros h
  field_simp
  
theorem division_equivalent (h : 0 ≤ 0.08 ∧ 0 ≤ 0.002 ∧ 0.08 ≠ 0 ∧ 0.002 ≠ 0) :
  0.08 / 0.002 = 40 :=
by
  have := division_to_fraction 0.08 0.002 h
  norm_num at this
  exact this

end division_equivalent_l692_692755


namespace part_I_part_II_l692_692229

noncomputable def f (x b c : ℝ) := x^2 + b*x + c

theorem part_I (x_1 x_2 b c : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) :
  b^2 > 2 * (b + 2 * c) :=
sorry

theorem part_II (x_1 x_2 b c t : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) (h5 : 0 < t ∧ t < x_1) :
  f t b c > x_1 :=
sorry

end part_I_part_II_l692_692229


namespace solve_inequalities_l692_692499

theorem solve_inequalities (x : ℝ) (h1 : |4 - x| < 5) (h2 : x^2 < 36) : (-1 < x) ∧ (x < 6) :=
by
  sorry

end solve_inequalities_l692_692499


namespace A_independent_of_beta_l692_692661

noncomputable def A (alpha beta : ℝ) : ℝ :=
  (Real.sin (alpha + beta) ^ 2) + (Real.sin (beta - alpha) ^ 2) - 
  2 * (Real.sin (alpha + beta)) * (Real.sin (beta - alpha)) * (Real.cos (2 * alpha))

theorem A_independent_of_beta (alpha beta : ℝ) : 
  ∃ (c : ℝ), ∀ beta : ℝ, A alpha beta = c :=
by
  sorry

end A_independent_of_beta_l692_692661


namespace solution_set_of_inequality_l692_692726

theorem solution_set_of_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_inequality_l692_692726


namespace find_k_l692_692604

def vector (α : Type) := α × α 

def dot_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def is_right_triangle (AB AC BC : vector ℝ) : Prop :=
  dot_product AB AC = 0 ∨ dot_product AB BC = 0 ∨ dot_product AC BC = 0

theorem find_k (k : ℝ) :
  let AB := (2, 3)
  let AC := (1, k)
  let BC := (1 - 2, k - 3)
  is_right_triangle AB AC BC →
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + real.sqrt 13) / 2 ∨ k = (3 - real.sqrt 13) / 2 :=
begin
  intros h,
  sorry
end

end find_k_l692_692604


namespace min_distance_sum_l692_692142

open Real

-- Conditions definitions
def parabola (P : ℝ × ℝ) : Prop :=
  P.2^2 = 4 * P.1

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 2)

-- Lean 4 statement
theorem min_distance_sum (P : ℝ × ℝ) (hP : parabola P) :
  |((P.1 - A.1)^2 + (P.2 - A.2)^2).sqrt|
  + |((P.1 - B.1)^2 + (P.2 - B.2)^2).sqrt| = 5 :=
begin
  sorry
end

end min_distance_sum_l692_692142


namespace problem_1_problem_2a_problem_2b_problem_2c_l692_692132

section
variable (x m : ℝ)

def f (x m : ℝ) : ℝ := x^2 - (m + (1 / m)) * x + 1

theorem problem_1 (h : m = 2) : { x | f x m <= 0 } = { x | (1 / 2) ≤ x ∧ x ≤ 2 } :=
by {
  sorry
}

theorem problem_2a (h : 0 < m ∧ m < 1) : { x | f x m >= 0 } = { x | x <= m ∨ x ≥ (1 / m) } :=
by {
  sorry
}

theorem problem_2b (h : m = 1) : { x | f x m >= 0 } = { x | true } :=
by {
  sorry
}

theorem problem_2c (h : 1 < m) : { x | f x m >= 0 } = { x | x >= m ∨ x <= (1 / m) } :=
by {
  sorry
}

end

end problem_1_problem_2a_problem_2b_problem_2c_l692_692132


namespace driver_net_rate_of_pay_l692_692413

def hours := 3
def speed := 45 -- miles per hour
def fuel_efficiency := 36 -- miles per gallon
def pay_per_mile := 0.60 -- dollars per mile
def cost_per_gallon := 2.50 -- dollars per gallon

def net_rate_of_pay_per_hour (hours : ℕ) (speed fuel_efficiency : ℕ) (pay_per_mile cost_per_gallon : ℝ) : ℝ :=
  let distance := speed * hours -- total distance driven
  let fuel_used := distance / fuel_efficiency -- gasoline used
  let earnings := pay_per_mile * distance -- total earnings
  let fuel_cost := cost_per_gallon * fuel_used -- cost of gasoline
  let net_earnings := earnings - fuel_cost -- net earnings
  net_earnings / hours -- net rate of pay per hour

theorem driver_net_rate_of_pay : net_rate_of_pay_per_hour hours speed fuel_efficiency pay_per_mile cost_per_gallon = 23.875 := by
  sorry

end driver_net_rate_of_pay_l692_692413


namespace inequality_transformation_l692_692511

theorem inequality_transformation (x y : ℝ) (h : x > y) : 3 * x > 3 * y :=
by sorry

end inequality_transformation_l692_692511


namespace roots_square_sum_l692_692179

theorem roots_square_sum (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) : 
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by 
  -- proof skipped
  sorry

end roots_square_sum_l692_692179


namespace original_problem_l692_692024

/-- We define the sums of the relevant expressions. -/
def sum_1 : ℚ := 3 + 6 + 9
def sum_2 : ℚ := 2 + 5 + 8

/-- The main theorem states the original problem's proof. -/
theorem original_problem :
  (sum_1 / sum_2) + (sum_2 / sum_1) = 61 / 30 :=
by
  /-- Provide definitions for sums to aid readability. -/
  have h1 : sum_1 = 18 := by norm_num
  have h2 : sum_2 = 15 := by norm_num
  rw [h1, h2]
  -- Reduce the fractions and simplify.
  norm_num
  /-- end goal is the final result -/
  sorry

end original_problem_l692_692024


namespace unique_positive_b_solution_exists_l692_692480

theorem unique_positive_b_solution_exists (c : ℝ) (k : ℝ) :
  (∃b : ℝ, b > 0 ∧ ∀x : ℝ, x^2 + (b + 1/b) * x + c = 0 → x = 0) ∧
  (∀b : ℝ, b^4 + (2 - 4 * c) * b^2 + k = 0) → c = 1 :=
by
  sorry

end unique_positive_b_solution_exists_l692_692480


namespace dawn_saves_each_month_l692_692474

def annual_income : ℝ := 48000
def monthly_income : ℝ := annual_income / 12
def tax_rate : ℝ := 0.2
def variable_expense_rate : ℝ := 0.3
def investment_rate : ℝ := 0.05
def retirement_rate : ℝ := 0.15
def savings_rate : ℝ := 0.1

def tax_deductions (income : ℝ) : ℝ := tax_rate * income
def after_tax_income (income : ℝ) : ℝ := income - tax_deductions(income)
def variable_expenses (income : ℝ) : ℝ := variable_expense_rate * income
def investments (income : ℝ) : ℝ := investment_rate * income
def retirement_contributions (income : ℝ) : ℝ := retirement_rate * income

def total_spent (income : ℝ) : ℝ := variable_expenses(income) + investments(income) + retirement_contributions(income)
def remaining_income (income : ℝ) : ℝ := after_tax_income(income) - total_spent(income)
def savings (income : ℝ) : ℝ := savings_rate * remaining_income(income)

theorem dawn_saves_each_month : savings monthly_income = 160 := by
  sorry

end dawn_saves_each_month_l692_692474


namespace weight_of_replaced_person_l692_692290

-- Define the conditions
variables (W : ℝ) (new_person_weight : ℝ) (avg_weight_increase : ℝ)
#check ℝ

def initial_group_size := 10

-- Define the conditions as hypothesis statements
axiom weight_increase_eq : avg_weight_increase = 3.5
axiom new_person_weight_eq : new_person_weight = 100

-- Define the result to be proved
theorem weight_of_replaced_person (W : ℝ) : 
  ∀ (avg_weight_increase : ℝ) (new_person_weight : ℝ),
    avg_weight_increase = 3.5 ∧ new_person_weight = 100 → 
    (new_person_weight - (avg_weight_increase * initial_group_size)) = 65 := 
by
  sorry

end weight_of_replaced_person_l692_692290


namespace books_sold_on_Wednesday_l692_692614

theorem books_sold_on_Wednesday :
  ∃ (books_sold : ℕ), books_sold = 60 ∧ 
  (∀ (total_stock books_sold_mon books_sold_tue books_sold_thu books_sold_fri books_unsold : ℕ),
    total_stock = 1400 →
    books_sold_mon = 62 →
    books_sold_tue = 62 →
    books_sold_thu = 48 →
    books_sold_fri = 40 →
    books_unsold = nat.floor (1400 * 0.8057142857142857) →
    total_stock - (books_unsold + books_sold_mon + books_sold_tue + books_sold_thu + books_sold_fri) = books_sold) := 
  sorry

end books_sold_on_Wednesday_l692_692614


namespace find_d_for_single_point_l692_692284

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end find_d_for_single_point_l692_692284


namespace axis_of_symmetry_l692_692952

noncomputable def g : ℝ → ℝ := sorry

theorem axis_of_symmetry (h : ∀ x, g(x) = g(3 - x)) : ∀ y, g(1.5 + y) = g(1.5 - y) :=
by
  assume y
  sorry

end axis_of_symmetry_l692_692952


namespace matrix_row_col_sum_int_l692_692430

open Matrix 

theorem matrix_row_col_sum_int {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
  (H_row : ∀ i : Fin m, ∃ k : ℤ, (∑ j, A i j) = k)
  (H_col : ∀ j : Fin n, ∃ k : ℤ, (∑ i, A i j) = k) :
  ∃ B : Matrix (Fin m) (Fin n) ℤ,
    (∀ i j, B i j = ⌊A i j⌋ ∨ B i j = ⌈A i j⌉) ∧
    (∀ i, ∑ j, B i j = ∑ j, A i j) ∧
    (∀ j, ∑ i, B i j = ∑ i, A i j) :=
by
  -- Proof is omitted
  sorry

end matrix_row_col_sum_int_l692_692430


namespace seating_arrangements_l692_692745

/-- There are two rows of seats with three side-by-side seats in each row.
Two little boys, two little girls, and two adults sit in the six seats such that neither little boy sits next to either little girl.
Prove that the number of different seating arrangements is 216. -/
theorem seating_arrangements : 
  let rows := 2
  let seats_per_row := 3
  let people := ["boy1", "boy2", "girl1", "girl2", "adult1", "adult2"]
  in ((rows * (nat.choose seats_per_row 2) * 2 * 2 * seats_per_row * 2) + 
      ((seats_per_row * seats_per_row * 2 * 2 * 2)) = 216) :=
by
  sorry

end seating_arrangements_l692_692745


namespace shortest_route_l692_692750

/--
Given two villages, A and B, on opposite sides of a straight river.
B will host a market, and residents of A wish to attend.
A bridge is to be built perpendicular to the river such that
the total route from A to B via the bridge is minimized.

Prove that the shortest route is 6 km.
-/
theorem shortest_route (A B: Type) (river: A → B → Prop) (d_AB: ℝ)
  (d_AE: ℝ) (d_EB: ℝ) (d_XC: ℝ): 
  d_AE = 5 → d_EB = 1 → d_XC = 0.75 → 
  (∀ C, route A C B river = d_AE + d_EB) → 
  route_min A B river = 6 := 
by
  sorry

end shortest_route_l692_692750


namespace two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692929

theorem two_pairs_satisfy_x_squared_minus_y_squared_is_77 :
  {p : ℕ × ℕ // p.1 ^ 2 - p.2 ^ 2 = 77}.card = 2 :=
sorry

end two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692929


namespace range_of_values_a_l692_692148

-- Define logarithmic inequality
def log_inequality (a : ℝ) : Prop :=
  log a (2 / 3) < 1

-- Define the main theorem
theorem range_of_values_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  log_inequality a ↔ (0 < a ∧ a < 2 / 3) ∨ (1 < a) :=
by
  sorry

end range_of_values_a_l692_692148


namespace proof_evaluate_expression_l692_692094

-- The cyclical properties of 'i'
def i_cyclical_properties : Prop :=
  ∀ n : ℕ, i ^ (4 * n) = 1 ∧ i ^ (4 * n + 1) = i ∧ i ^ (4 * n + 2) = -1 ∧ i ^ (4 * n + 3) = -i

noncomputable def evaluate_expression : Prop :=
  2 * (i ^ 13 + i ^ 18 + i ^ 23 + i ^ 28 + i ^ 33) = 2 * i

theorem proof_evaluate_expression (h : i_cyclical_properties) : evaluate_expression :=
sorry

end proof_evaluate_expression_l692_692094


namespace length_of_segment_correct_l692_692170

noncomputable def length_of_segment (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem length_of_segment_correct :
  length_of_segment 5 (-1) 13 11 = 4 * Real.sqrt 13 := by
  sorry

end length_of_segment_correct_l692_692170


namespace find_N_l692_692008

theorem find_N : 
  ∃ N : ℕ, 
  let p_same_color : ℚ := 0.58 in
  let favorable_green := 4 * 16 in
  let favorable_blue := 6 * N in
  let total_ways := (4 + 6) * (16 + N) in
  (favorable_green + favorable_blue) / total_ways = p_same_color :=
begin
  -- The proof will go here, but as per the instruction, we will just use sorry.
  use 144,
  sorry
end

end find_N_l692_692008


namespace books_combination_l692_692943

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l692_692943


namespace find_p_q_sum_equals_six_l692_692320

/-- Triangle DEF has side lengths DE = 13, EF = 30, and FD = 37.
    Parallelogram XYZW has vertex X on DE, vertex Y on DF, 
    and vertices Z and W on EF. The area of XYZW can be expressed 
    as the polynomial Area(XYZW) = γ * φ - δ * φ^2. 
    Given that δ for Area(XYZW) = γ * φ - δ * φ^2 as sides length DE, 
    Ef and FD is δ = 1/5, then p (numerator of reduced fraction) = 1, 
    q (denominator of reduced fraction) = 5. Thus, p + q should be 6. -/
theorem find_p_q_sum_equals_six : 
  let DE := 13
  let EF := 30
  let FD := 37
  let φ := 15
  let γ := 30 * (1/5)
  let Area := γ * φ - (1/5) * φ^2
  let δ := 1/5
  let p := 1
  let q := 5
  in p + q = 6 :=
by
  sorry

end find_p_q_sum_equals_six_l692_692320


namespace complement_U_A_l692_692162

noncomputable def U := {-2, -1, 0, 1, 2}

noncomputable def A := {x : ℕ | -2 < x ∧ x < 3}

theorem complement_U_A : (U \ A) = {-2, -1} :=
by
  sorry

end complement_U_A_l692_692162


namespace hyperbola_equation_l692_692879

noncomputable def hyperbola_eqn : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (b = (1/2) * a) ∧ (a^2 + b^2 = 25) ∧ 
    (∀ x y, (x^2 / (a^2)) - (y^2 / (b^2)) = 1 ↔ (x^2 / 20) - (y^2 / 5) = 1)

theorem hyperbola_equation : hyperbola_eqn := 
  sorry

end hyperbola_equation_l692_692879


namespace fraction_equality_l692_692172

theorem fraction_equality (a b : ℝ) (h : a / 4 = b / 3) : b / (a - b) = 3 :=
sorry

end fraction_equality_l692_692172


namespace log3_20_approx_l692_692522

theorem log3_20_approx :
  (∀ (log10_2 log10_3 : ℝ), log10_2 ≈ 0.30103 ∧ log10_3 ≈ 0.47712 →
    (log10_2 + 1) / log10_3 ≈ 2.25) :=
by
  intros log10_2 log10_3 h,
  cases h with h2 h3,
  sorry

end log3_20_approx_l692_692522


namespace acute_angle_between_hands_l692_692956

-- The given conditions
def initial_time : ℕ := 3 -- 3 PM
def final_hour_position : ℕ := 4 -- 4 PM
def initial_hour_angle : ℕ := 90 -- Degrees at 3 PM (3 * 30 degrees)
def degrees_per_minute_hour_moving_minute_hand : ℝ := 6
def degrees_per_minute_minute_moving_hour_hand : ℝ := 0.5

-- The final angle calculated (this is the correct answer according to solution)
def correct_acute_angle : ℝ := 117.5

-- The proof problem statement
theorem acute_angle_between_hands : 
  ∀ t : ℝ, (t = 5) → 
  (correct_acute_angle = 
    let hour_position := initial_hour_angle + t * degrees_per_minute_hour_moving_minute_hand in
    let minute_position := t * degrees_per_minute_minute_moving_hour_hand in
    let angle_between := abs (hour_position - minute_position) in
    if angle_between > 180 then 360 - angle_between else angle_between
  ) := 
sorry

end acute_angle_between_hands_l692_692956


namespace bob_remaining_corns_l692_692459

theorem bob_remaining_corns (total_bushels : ℕ) (terry_bushels : ℕ) (jerry_bushels : ℕ)
                            (linda_bushels: ℕ) (stacy_ears: ℕ) (ears_per_bushel: ℕ):
                            total_bushels = 50 → terry_bushels = 8 → jerry_bushels = 3 →
                            linda_bushels = 12 → stacy_ears = 21 → ears_per_bushel = 14 →
                            (total_bushels - (terry_bushels + jerry_bushels + linda_bushels + stacy_ears / ears_per_bushel)) * ears_per_bushel = 357 :=
by intros total_cond terry_cond jerry_cond linda_cond stacy_cond ears_cond
   rw [total_cond, terry_cond, jerry_cond, linda_cond, stacy_cond, ears_cond]
   norm_cast
   have : 21 / 14 = (3 / 2 : ℕ) := sorry
   rw this
   linarith
   sorry

end bob_remaining_corns_l692_692459


namespace tangent_secant_distance_relation_l692_692507

-- Define the geometric setup and the conditions
variables {O : Type*} [metric_space O] [inner_product_space ℝ O]
variables (P A B E F C : O)

-- Auxiliary definitions for tangency points and distances
def is_tangent (X P : O) (O : set O) : Prop := ... -- Tangency definition here, skipped for brevity
def is_secant (A B P : O) (O : set O) : Prop := ... -- Secant definition here, skipped for brevity

-- Key geometric conditions given in the problem
axiom P_outside_circle : ∃ (r : ℝ), dist P O > r
axiom PE_tangent : is_tangent E P (emetric.ball O r)
axiom PF_tangent : is_tangent F P (emetric.ball O r)
axiom secant_line : is_secant A B P (emetric.ball O r)
axiom intersection_on_line : ∃ (t : ℝ), C = t • (F - E) + E

-- The theorem to prove
theorem tangent_secant_distance_relation :
  (2 / dist P C) = (1 / dist P A) + (1 / dist P B) :=
by
  sorry

end tangent_secant_distance_relation_l692_692507


namespace min_value_CP_PA1_l692_692012

variables (A B C A1 B1 C1 P : Type) 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]
variables [MetricSpace P]

-- Assumptions
variables (angle_ACB_right : ∠ ACB = 90)
variables (BC_eq_CC1 : BC = 2 ∧ CC1 = 2)
variables (AC_length : AC = 4 * sqrt 2)
variables (P_on_BC1 : P ∈ lineSegment BC1)

-- Proof the minimum value
theorem min_value_CP_PA1 : minimumDistance (CP + PA1) = 2 * sqrt 13 :=
by
  sorry

end min_value_CP_PA1_l692_692012


namespace arithmetic_sequence_sum_l692_692535

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ), (a 11 = 3) → (a 2 = 39) →
  (∃ S₁₁ : ℤ, S₁₁ = 253) :=
begin
  sorry
end

end arithmetic_sequence_sum_l692_692535


namespace student_l692_692438

-- Definition of the conditions
def mistaken_calculation (x : ℤ) : ℤ :=
  x + 10

def correct_calculation (x : ℤ) : ℤ :=
  x + 5

-- Theorem statement: Prove that the student's result is 10 more than the correct result
theorem student's_error {x : ℤ} : mistaken_calculation x = correct_calculation x + 5 :=
by
  sorry

end student_l692_692438


namespace all_arrive_same_time_l692_692448

-- Define conditions based on the problem statement
def car_speed := 30 -- mph
def bob_bike_speed := 10 -- mph
def clara_bike_speed := 15 -- mph
def total_distance := 150 -- miles

-- Define the distance Clara travels by car
variable (d1 : ℝ)

-- Total time for Clara's journey
def clara_journey_time (d1 : ℝ) : ℝ :=
  d1 / car_speed + (total_distance - d1) / clara_bike_speed

-- Total time for Bob's journey
def bob_journey_time (d1 : ℝ) : ℝ :=
  d1 / bob_bike_speed + (total_distance - d1) / car_speed

-- The total time required for all three to reach the conference
def total_time (d1 : ℝ) : ℝ :=
  if clara_journey_time d1 = bob_journey_time d1 then
    clara_journey_time d1
  else
    0 -- Placeholder since the two times must be equal for synchronization

-- Prove that the total time T is 7.5 hours for an optimal d1
theorem all_arrive_same_time :
  ∃ d1, d1 = 75 ∧ total_time d1 = 7.5 := by
  -- Proof omitted
  sorry

end all_arrive_same_time_l692_692448


namespace integer_roots_no_integer_roots_l692_692100

def polynomial (x : ℤ) : ℤ := x^3 - 5 * x^2 - 8 * x + 24

theorem integer_roots : ∃ (x ∈ { -3, 2, 4} ), polynomial x = 0 :=
by
  exists -3
  rw polynomial
  sorry

-- If no roots are to be found at any evaluations:
theorem no_integer_roots : ∀ x : ℤ, polynomial x ≠ 0 :=
by
  intros x
  rw polynomial
  sorry

end integer_roots_no_integer_roots_l692_692100


namespace angle_D_is_72_degrees_l692_692276

structure GeometryProblem :=
  (A B C D E : Point)
  (BD_AE_intersect_C : Segment BD ∩ Segment AE = {C})
  (AB_eq_BC : dist A B = dist B C)
  (BC_eq_CD : dist B C = dist C D)
  (CD_eq_CE : dist C D = dist C E)
  (angle_A_eq_3B : angle A = 3 * angle B)

theorem angle_D_is_72_degrees (P : GeometryProblem) : angle D = 72 := 
  sorry

end angle_D_is_72_degrees_l692_692276


namespace geometric_sequence_root_equation_l692_692978

noncomputable def geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ): Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_root_equation (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence_condition a q) (h_root: ∀ x, x^2 + 6 * x + 2 = 0 → x = a 2 ∨ x = a 16) : 
  a 2 * a 16 / a 9 = sqrt 2 ∨ a 2 * a 16 / a 9 = -sqrt 2 :=
sorry

end geometric_sequence_root_equation_l692_692978


namespace swim_speed_l692_692421

noncomputable def speed_man_in_still_water (v_m v_s : ℝ) (downstream_time upstream_time downstream_distance upstream_distance : ℝ) : ℝ :=
  let downstream_eq : Prop := downstream_distance = (v_m + v_s) * downstream_time
  let upstream_eq : Prop := upstream_distance = (v_m - v_s) * upstream_time
  if downstream_eq ∧ upstream_eq then v_m else 0

theorem swim_speed :
  ∀ (v_m v_s : ℝ),
    let downstream_time := 10
    let upstream_time := 10
    let downstream_distance := 60
    let upstream_distance := 100
    speed_man_in_still_water v_m v_s downstream_time upstream_time downstream_distance upstream_distance = 8 :=
begin
  intros v_m v_s,
  sorry
end

end swim_speed_l692_692421


namespace geometric_sum_S6_l692_692621

variable {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Conditions: S_n represents the sum of the first n terms of the geometric sequence {a_n}
-- and we have S_2 = 4 and S_4 = 6
theorem geometric_sum_S6 (S : ℕ → ℝ) (h1 : S 2 = 4) (h2 : S 4 = 6) : S 6 = 7 :=
sorry

end geometric_sum_S6_l692_692621


namespace converse_of_prop1_true_l692_692155

theorem converse_of_prop1_true
  (h1 : ∀ {x : ℝ}, x^2 - 3 * x + 2 = 0 → x = 1 ∨ x = 2)
  (h2 : ∀ {x : ℝ}, -2 ≤ x ∧ x < 3 → (x - 2) * (x - 3) ≤ 0)
  (h3 : ∀ {x y : ℝ}, x = 0 ∧ y = 0 → x^2 + y^2 = 0)
  (h4 : ∀ {x y : ℕ}, x > 0 ∧ y > 0 ∧ (x + y) % 2 = 1 → (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)) :
  (∀ {x : ℝ}, x = 1 ∨ x = 2 → x^2 - 3 * x + 2 = 0) :=
by
  sorry

end converse_of_prop1_true_l692_692155


namespace center_of_circle_tangent_to_lines_l692_692791

theorem center_of_circle_tangent_to_lines :
  ∃ (x y : ℚ), 
  (3 * x - 5 * y = 15 ∨ 3 * x - 5 * y = -45) ∧
  (x - 3 * y = 0) ∧
  (x = -45 / 4 ∧ y = -15 / 4) :=
begin
  sorry
end

end center_of_circle_tangent_to_lines_l692_692791


namespace equilateral_projection_side_l692_692115

theorem equilateral_projection_side
  (A B C : Type)
  (AB BC AC : ℝ)
  (h1 : AB = sqrt 14)
  (h2 : BC = sqrt 6)
  (h3 : AC = 3)
  (B1 C1 : Type)
  (BB1 CC1 : ℝ)
  (h4 : B ≠ C)
  (proj_eq_triangle : AB = AC ∧ AB = B1C1) :
  (AB = sqrt 5 ∧ AC = sqrt 5 ∧ B1C1 = sqrt 5) := sorry

end equilateral_projection_side_l692_692115


namespace reflections_translation_l692_692790

/-- Given a circle with four distinct points A, B, C, D, 
    the image obtained after successive reflections 
    across the lines AB, BC, CD, and DA is a translation of the original figure. -/
theorem reflections_translation (A B C D : Point) (figure : PlanarFigure) :
  distinct_points A B C D → 
  reflections_successive figure [AB, BC, CD, DA] → 
  is_translation_of (reflect_in_order figure [AB, BC, CD, DA]) figure :=
sorry

end reflections_translation_l692_692790


namespace cube_root_of_sum_of_expressions_l692_692761

theorem cube_root_of_sum_of_expressions :
  (∛ ((2^4 : ℝ) + (4^3 : ℝ) + (8^2 : ℝ)) = 2^(4/3) * 3^(2/3)) :=
by
  -- Conditions derived directly from the problem
  have h1 : (2^4 : ℝ) = 16 := by norm_num
  have h2 : (4^3 : ℝ) = 64 := by norm_num
  have h3 : (8^2 : ℝ) = 64 := by norm_num
  -- Sum these consistent values
  have h_sum : (∛ (16 + 64 + 64) = ∛144) := by norm_num
  -- Verification and conclusion
  rw [h_sum]
  rw [←real.eq_inv_cbrt_of_nonneg] -- Ensure non-negativity is handled appropriately
  sorry

#eval cube_root_of_sum_of_expressions

end cube_root_of_sum_of_expressions_l692_692761


namespace probability_of_irrational_card_l692_692312

-- Define the set of cards
def cards : List ℝ := [0, Real.pi, Real.sqrt 2, 1/9, 1.333]

-- Define the property of a number being irrational
def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

-- Count the number of irrational numbers in the list of cards
def count_irrationals (l : List ℝ) : ℕ := l.countp is_irrational

-- Define the total number of cards
def num_cards : ℕ := card.length

-- Define the probability as the ratio of irrational cards to total cards
def irrational_probability : ℚ := count_irrationals cards / num_cards

-- Theorem stating the desired proposition
theorem probability_of_irrational_card : 
  irrational_probability = 2 / 5 :=
by
  sorry

end probability_of_irrational_card_l692_692312


namespace quadruple_pieces_count_l692_692648

theorem quadruple_pieces_count (earned_amount_per_person_in_dollars : ℕ) 
    (total_single_pieces : ℕ) (total_double_pieces : ℕ)
    (total_triple_pieces : ℕ) (single_piece_circles : ℕ) 
    (double_piece_circles : ℕ) (triple_piece_circles : ℕ)
    (quadruple_piece_circles : ℕ) (cents_per_dollar : ℕ) :
    earned_amount_per_person_in_dollars * 2 * cents_per_dollar -
    (total_single_pieces * single_piece_circles + 
    total_double_pieces * double_piece_circles + 
    total_triple_pieces * triple_piece_circles) = 
    165 * quadruple_piece_circles :=
        sorry

#eval quadruple_pieces_count 5 100 45 50 1 2 3 4 100

end quadruple_pieces_count_l692_692648


namespace eulerian_path_exists_for_connected_graph_with_at_most_two_odd_vertices_l692_692267

-- Definition of a connected graph
def is_connected (G : SimpleGraph V) : Prop := ∀ (u v : V), G.reachable u v

-- Definition of vertex degree
def degree (G : SimpleGraph V) (v : V) : ℕ := G.degree v

-- Definition of odd degree vertex
def is_odd_degree (G : SimpleGraph V) (v : V) : Prop := (degree G v) % 2 = 1

-- Definition of Eulerian path
def has_eulerian_path (G : SimpleGraph V) : Prop :=
  ∃ (path : List (V × V)), 
    (∀ (e : V × V), e ∈ G.edgeSet ↔ e ∈ path) ∧ 
    -- Traversing each edge exactly once
    (path.Nodup) ∧
    -- No edge traversal twice
    (∀ (u : V), 
       (is_odd_degree G u → (∃ (v : V), (u, v) ∈ path)) 
        ∨ (!is_odd_degree G u))

theorem eulerian_path_exists_for_connected_graph_with_at_most_two_odd_vertices 
    (G : SimpleGraph V) 
    (h_connected : is_connected G)
    (h_odd_vertices : ∑ v in G.vertices, (if is_odd_degree G v then 1 else 0) ≤ 2) 
    : has_eulerian_path G := 
  sorry

end eulerian_path_exists_for_connected_graph_with_at_most_two_odd_vertices_l692_692267


namespace product_common_divisors_l692_692109

theorem product_common_divisors (h120 : ∀ d ∈ {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10, 12, -12, 15, -15, 20, -20, 24, -24, 30, -30, 40, -40, 60, -60, 120, -120},
                                 d ∣ 120) 
                                (h30 : ∀ d ∈ {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30},
                                 d ∣ 30) :
  (∏ d in {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30}.to_finset, d) = 729000000 := by
  -- mathematical proof here
  sorry

end product_common_divisors_l692_692109


namespace find_m_find_max_value_l692_692876

noncomputable def f (x m : ℝ) : ℝ := -sin x ^ 2 + m * (2 * cos x - 1)

theorem find_m (m : ℝ) :
  (∃ x ∈ set.Icc (-π/3) (2*π/3), f x m = -1) → m = 0 ∨ m = -1 :=
sorry

theorem find_max_value (x : ℝ) (m : ℝ) (h : m = 0 ∨ m = -1) (hx : x ∈ set.Icc (-π/3) (2*π/3)) :
  ∃ y ∈ set.Icc (-π/3) (2*π/3), (∀ z ∈ set.Icc (-π/3) (2*π/3), f z m ≤ f y m) :=
sorry

end find_m_find_max_value_l692_692876


namespace find_min_max_w_l692_692147

variable {R : Type*} [Real R]

-- Defining the main condition of the problem
def main_condition (x y : R) : Prop :=
  x^2 + y^2 = 16 * x + 8 * y + 20

-- Defining the function to be maximized/minimized
def w (x y : R) : R :=
  4 * x + 3 * y

-- Main theorem statement
theorem find_min_max_w :
  (∃ (x y : R), main_condition x y ∧ w x y = -64) ∧
  (∃ (x y : R), main_condition x y ∧ w x y = 116) :=
by sorry

end find_min_max_w_l692_692147


namespace probability_no_3by3_red_grid_correct_l692_692063

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l692_692063


namespace triangle_common_area_fraction_l692_692208

theorem triangle_common_area_fraction (A1 A2 A3 B1 B2 B3 C1 C2 C3 : Type)
  (area_A123 : ℝ)
  (quarter_points : ∀ (i : ℕ), 
    ((B1 A1 A2) ∧ (B2 A2 A3) ∧ (B3 A3 A1) ∧ 
     (C1 A1 A2) ∧ (C2 A2 A3) ∧ (C3 A3 A1))) :
  (∃ common_area : ℝ, common_area = (49 / 160) * area_A123) :=
sorry

end triangle_common_area_fraction_l692_692208


namespace john_income_l692_692994

theorem john_income 
  (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) (ingrid_income : ℝ) (combined_tax_rate : ℝ)
  (jt_30 : john_tax_rate = 0.30) (it_40 : ingrid_tax_rate = 0.40) (ii_72000 : ingrid_income = 72000) 
  (ctr_35625 : combined_tax_rate = 0.35625) :
  ∃ J : ℝ, (0.30 * J + ingrid_tax_rate * ingrid_income = combined_tax_rate * (J + ingrid_income)) ∧ (J = 56000) :=
by
  sorry

end john_income_l692_692994


namespace double_summation_value_l692_692865

theorem double_summation_value :
  ∑ i in Finset.range 50 + 1, ∑ j in Finset.range 150 + 1, (i + 2 * j + 5) = 1311250 :=
by
  sorry

end double_summation_value_l692_692865


namespace find_r_l692_692299

noncomputable def geometric_series_partial_sum (a r n : ℕ) :=
  ∑ k in range (n+1), a * r^k

noncomputable def sum_odd_powers (a r n : ℕ) :=
  ∑ k in range (n+1), if k % 2 == 1 then a * r^k else 0

theorem find_r (a r : ℚ) (h_series_sum : geometric_series_partial_sum a r ∞ = 24)
    (h_odd_powers_sum : sum_odd_powers a r ∞ = 10) :
    r = 5 / 7 := by
  sorry

end find_r_l692_692299


namespace flour_in_cupboard_l692_692613

theorem flour_in_cupboard :
  let flour_on_counter := 100
  let flour_in_pantry := 100
  let flour_per_loaf := 200
  let loaves := 2
  let total_flour_needed := loaves * flour_per_loaf
  let flour_outside_cupboard := flour_on_counter + flour_in_pantry
  let flour_in_cupboard := total_flour_needed - flour_outside_cupboard
  flour_in_cupboard = 200 :=
by
  sorry

end flour_in_cupboard_l692_692613


namespace sum_m_n_l692_692055

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l692_692055


namespace largest_prime_factor_1001_l692_692359

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692359


namespace area_ratio_triangle_l692_692209

theorem area_ratio_triangle (XY XZ YZ : ℝ) (hXY : XY = 18) (hXZ : XZ = 27) (hYZ : YZ = 22)
  (angle_bisector : ∀ YD ZD : ℝ, YD / ZD = XY / XZ) :
  ∃ YD ZD : ℝ, (YD / ZD = 2/3) ∧ (area_ratio := YD / ZD) :=
begin
  use [12, 18], -- YD and ZD that give the required ratio.
  split,
  -- Show that this choice of YD and ZD gives the ratio 2/3 as required
  { exact angle_bisector 12 18 },
  -- Directly using the area ratio observed from YD and ZD
  { exact (2 : ℝ) / 3 },
end

end area_ratio_triangle_l692_692209


namespace median_to_longest_side_l692_692884

theorem median_to_longest_side
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26)
  (h4 : a^2 + b^2 = c^2) :
  ∃ m : ℕ, m = c / 2 ∧ m = 13 := 
by {
  sorry
}

end median_to_longest_side_l692_692884


namespace Jerry_stickers_l692_692991

theorem Jerry_stickers (Fred George Jerry : ℕ) 
    (hFred : Fred = 18)
    (hGeorge : George = Fred - 6)
    (hJerry : Jerry = 3 * George) :
    Jerry = 36 :=
by
    rw [hFred, hGeorge] at hJerry
    change Jerry = 3 * (18 - 6) at hJerry
    norm_num at hJerry
    exact hJerry

end Jerry_stickers_l692_692991


namespace expected_length_first_group_19_49_l692_692717

noncomputable def expected_first_group_length (ones zeros : ℕ) : ℕ :=
  let total := ones + zeros;
  let pI := 1 / 50;
  let pJ := 1 / 20;
  let expected_I := ones * pI;
  let expected_J := zeros * pJ;
  expected_I + expected_J

theorem expected_length_first_group_19_49 : expected_first_group_length 19 49 = 2.83 :=
by
  sorry

end expected_length_first_group_19_49_l692_692717


namespace vector_parallel_exists_l692_692039

noncomputable def find_vector : ℝ × ℝ :=
let t := -7 / 6 in
(3 * t + 1, 2 * t + 3)

theorem vector_parallel_exists :
  ∃ (a b : ℝ), (a, b) = (-3, -2) ∧ (∃ k : ℝ, (a, b) = (3 * k, 2 * k)) :=
by
  use -3
  use -2
  split
  . refl
  . use -1
    simp

end vector_parallel_exists_l692_692039


namespace quadruple_pieces_sold_l692_692647

theorem quadruple_pieces_sold (split_earnings : (2 : ℝ) * 5 = 10) 
  (single_pieces_sold : 100 * (0.01 : ℝ) = 1) 
  (double_pieces_sold : 45 * (0.02 : ℝ) = 0.9) 
  (triple_pieces_sold : 50 * (0.03 : ℝ) = 1.5) : 
  let total_earnings := 10
  let earnings_from_others := 3.4
  let quadruple_piece_price := 0.04
  total_earnings - earnings_from_others = 6.6 → 
  6.6 / quadruple_piece_price = 165 :=
by 
  intros 
  sorry

end quadruple_pieces_sold_l692_692647


namespace number_of_four_digit_numbers_divisible_by_15_l692_692107

theorem number_of_four_digit_numbers_divisible_by_15 :
  ∃ (S : Finset (Fin 10000)), S.card = 36 ∧
  ∀ n ∈ S, (∃ d1 d2 d3 d4 : Nat, 
  d1 ∈ {1, 2, 3, 4, 5, 6, 7} ∧ 
  d2 ∈ {1, 2, 3, 4, 5, 6, 7} ∧ 
  d3 ∈ {1, 2, 3, 4, 5, 6, 7} ∧ 
  d4 ∈ {1, 2, 3, 4, 5, 6, 7} ∧ 
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
  n = d1*1000 + d2*100 + d3*10 + d4 ∧ 
  d4 = 5 ∧ (d1 + d2 + d3 + d4) % 3 = 0 ∧ n % 15 = 0) := sorry

end number_of_four_digit_numbers_divisible_by_15_l692_692107


namespace two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692928

theorem two_pairs_satisfy_x_squared_minus_y_squared_is_77 :
  {p : ℕ × ℕ // p.1 ^ 2 - p.2 ^ 2 = 77}.card = 2 :=
sorry

end two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692928


namespace radius_is_sqrt_10_l692_692949

noncomputable def tangent_circle_line (r : ℝ) : Prop :=
  r > 0 ∧ (∀ (x y : ℝ), (x + 2 * y = r → x^2 + y^2 = 2 * r^2))

theorem radius_is_sqrt_10 (r : ℝ) : tangent_circle_line r → r = sqrt 10 :=
by
  sorry

end radius_is_sqrt_10_l692_692949


namespace probability_no_3x3_red_square_l692_692065

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l692_692065


namespace wyatt_headmaster_duration_l692_692187

def Wyatt_start_month : Nat := 3 -- March
def Wyatt_break_start_month : Nat := 7 -- July
def Wyatt_break_end_month : Nat := 12 -- December
def Wyatt_end_year : Nat := 2011

def months_worked_before_break : Nat := Wyatt_break_start_month - Wyatt_start_month -- March to June (inclusive, hence -1)
def break_duration : Nat := 6
def months_worked_after_break : Nat := 12 -- January to December 2011

def total_months_worked : Nat := months_worked_before_break + months_worked_after_break
theorem wyatt_headmaster_duration : total_months_worked = 16 :=
by
  sorry

end wyatt_headmaster_duration_l692_692187


namespace employees_both_fraction_l692_692022

-- Define the total number of employees as a variable
variable {x : ℚ}

-- Define the fractions representing employees with cell phones, pagers, and neither
def cell_phone_fraction : ℚ := 2/3
def pager_fraction : ℚ := 2/5
def neither_fraction : ℚ := 1/3

-- Define a fraction representing the employees with both devices
def both_fraction (x : ℚ) := cell_phone_fraction + pager_fraction - 1 + neither_fraction

-- Statement to prove
theorem employees_both_fraction : both_fraction x = 2/5 :=
by
  sorry

end employees_both_fraction_l692_692022


namespace max_two_digit_sequence_length_l692_692758

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_not_coprime (a b : ℕ) : Prop := Nat.gcd a b > 1
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def sequence_satisfies_conditions (seq : List ℕ) : Prop :=
  (∀ i, i < seq.length → is_two_digit (seq.nth_le i sorry)) ∧
  (∀ i, i + 1 < seq.length → is_not_coprime (seq.nth_le i sorry) (seq.nth_le (i + 1) sorry)) ∧
  (∀ i j, i < j ∧ j < seq.length ∧ j ≠ i + 1 → is_coprime (seq.nth_le i sorry) (seq.nth_le j sorry))

theorem max_two_digit_sequence_length : 
  ∃ seq : List ℕ, sequence_satisfies_conditions seq ∧ seq.length = 10 :=
sorry

end max_two_digit_sequence_length_l692_692758


namespace min_xy_positive_real_l692_692529

theorem min_xy_positive_real (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 / (2 + x) + 3 / (2 + y) = 1) :
  ∃ m : ℝ, m = 16 ∧ ∀ xy : ℝ, (xy = x * y) → xy ≥ m :=
by
  sorry

end min_xy_positive_real_l692_692529


namespace rate_of_markup_on_cost_l692_692612

noncomputable def selling_price : ℝ := 10.0
noncomputable def profit_rate : ℝ := 0.20
noncomputable def expense_rate : ℝ := 0.10
noncomputable def tax_rate : ℝ := 0.05

theorem rate_of_markup_on_cost : 
  let C := selling_price * (1 - (profit_rate + expense_rate + tax_rate)) in
  (selling_price - C) / C * 100 = 53.85 :=
by
  sorry

end rate_of_markup_on_cost_l692_692612


namespace find_multiple_l692_692423

theorem find_multiple:
  let number := 220025
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := number / sum
  let remainder := number % sum
  (remainder = 25) → (quotient = 220) → (quotient / diff = 2) :=
by
  intros number sum diff quotient remainder h1 h2
  sorry

end find_multiple_l692_692423


namespace sum_of_satisfying_primes_is_zero_l692_692500

def is_prime (n : ℕ) : Prop := Nat.Prime n

def satisfies_conditions (p : ℕ) : Prop :=
  p % 6 = 1 ∧ p % 5 = 3 ∧ is_prime p ∧ p ≥ 1 ∧ p ≤ 150

theorem sum_of_satisfying_primes_is_zero :
  ∑ p in (finset.filter satisfies_conditions (finset.range 151)), p = 0 :=
by
  sorry

end sum_of_satisfying_primes_is_zero_l692_692500


namespace gray_triangle_area_correct_l692_692800

noncomputable def gray_triangle_area : ℕ :=
  let side_1: ℕ := 12
  let side_2: ℕ := 15
  let hypotenuse: ℕ := 15
  let larger_triangle_leg: ℕ := 9
  let smaller_triangle_leg1: ℕ := 6
  let smaller_triangle_leg2: ℚ := 9 / 2
  let base: ℚ := 9 / 2
  let height: ℕ := 15
  (1 / 2 * base * height)

theorem gray_triangle_area_correct :
  gray_triangle_area = (135 / 4) :=
by
  sorry

end gray_triangle_area_correct_l692_692800


namespace range_of_f_l692_692863

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - x + 2)

theorem range_of_f : set.range f = set.Icc (-(1 / 7 : ℚ)) 1 := by
  sorry

end range_of_f_l692_692863


namespace total_comics_in_box_l692_692021

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l692_692021


namespace coefficient_x2_expansion_l692_692103

theorem coefficient_x2_expansion : 
  (∃ c:int, (1-x)^3 * (1-√x)^4 = c * x^2 + ...) ∧  c = -14 :=
by
  sorry

end coefficient_x2_expansion_l692_692103


namespace basement_pump_time_l692_692783

theorem basement_pump_time :
  ∀ (floor_length floor_width water_depth_inches pumps pumping_rate gallons_per_cubic_foot : ℕ)
    (n_pumps : ℕ),
  (floor_length = 30) →
  (floor_width = 40) →
  (water_depth_inches = 24) →
  (pumps = 4) →
  (pumping_rate = 10) →
  (gallons_per_cubic_foot = 7.5) →
  (n_pumps = 4) →
  ((24 / 12) * 30 * 40 * 7.5 / (10 * 4) = 450) := by
sorry

end basement_pump_time_l692_692783


namespace probability_no_3x3_red_square_l692_692074

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l692_692074


namespace min_period_sin_squared_l692_692694

theorem min_period_sin_squared (ω : ℝ) (hω : 0 < ω) :
  (∃ T > 0, ∀ x, sin^2((ω/2) * x - (π/4)) = sin^2((ω/2) * (x + T) - (π/4))) ↔ ω = 2 :=
by
  sorry

end min_period_sin_squared_l692_692694


namespace parallelogram_area_correct_l692_692588

noncomputable def parallelogram_area_proof : ℝ :=
  let AB := 4
  let AD := 3
  let AE_EB_ratio := 1 / 3
  let DF_FC_ratio := 2
  let BF_CE_dot_product := -6

  -- Calculate the cosine of the angle
  let cos_BAD := 1 / 5
  -- Calculate the sine of the angle using Pythagorean identity
  let sin_BAD := Real.sqrt(1 - cos_BAD ^ 2)
  -- Calculate the area of the parallelogram using the side lengths and sine of the angle
  let area := AB * AD * sin_BAD

  -- Given conditions imply that sin_BAD should be 2 * Real.sqrt(6) / 5
  let expected_area := 24 * Real.sqrt(6) / 5

  -- Proof that area == expected_area
  area

theorem parallelogram_area_correct : parallelogram_area_proof = (24 * Real.sqrt(6) / 5) := sorry

end parallelogram_area_correct_l692_692588


namespace prove_ff_ff_l692_692123

def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2
  else if z.re > 0 then -z^2
  else z^3

theorem prove_ff_ff (z : ℂ) (hz : z = 1 + complex.I) :
  f(f(f(f(z)))) = -262144 :=
by
  rw [hz]
  sorry

end prove_ff_ff_l692_692123


namespace equation_infinite_solutions_when_m_is_2_l692_692153

theorem equation_infinite_solutions_when_m_is_2 (x : ℝ) :
  ∀ m : ℝ, m = 2 → (m^2 * x + m * (1 - x) - 2 * (1 + x) = 0) :=
by
  intro m h
  rw h
  sorry

end equation_infinite_solutions_when_m_is_2_l692_692153


namespace find_constant_l692_692572

theorem find_constant (n : ℤ) (c : ℝ) (h1 : ∀ n ≤ 10, c * (n : ℝ)^2 ≤ 12100) : c ≤ 121 :=
sorry

end find_constant_l692_692572


namespace proof_expr_28_times_35_1003_l692_692096

theorem proof_expr_28_times_35_1003 :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 :=
by
  sorry

end proof_expr_28_times_35_1003_l692_692096


namespace probability_of_different_suits_l692_692254

theorem probability_of_different_suits (cards : Finset (Fin 104)) 
  (h1 : cards.card = 104)
  (h2 : ∀ x y ∈ cards, x ≠ y → ∃ suit1 suit2, suit1 ≠ suit2 ∧ suit_of x = suit1 ∧ suit_of y = suit2) :
  let probability := 78 / 103 in
  probability = 78 / 103 :=
by
  sorry

end probability_of_different_suits_l692_692254


namespace renusuma_work_together_l692_692663

theorem renusuma_work_together (r_R r_S : ℝ) (hR : r_R = 1/8) (hS : r_S = 1/8) : 
  let combined_rate := r_R + r_S in
  let time_to_complete := 1 / combined_rate in
  time_to_complete = 4 :=
by
  sorry

end renusuma_work_together_l692_692663


namespace average_cost_is_2_l692_692642

noncomputable def total_amount_spent (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℕ :=
  apples_quantity * apples_cost + bananas_quantity * bananas_cost + oranges_quantity * oranges_cost

noncomputable def total_number_of_fruits (apples_quantity bananas_quantity oranges_quantity : ℕ) : ℕ :=
  apples_quantity + bananas_quantity + oranges_quantity

noncomputable def average_cost (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℚ :=
  (total_amount_spent apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℚ) /
  (total_number_of_fruits apples_quantity bananas_quantity oranges_quantity : ℚ)

theorem average_cost_is_2 :
  average_cost 12 4 4 2 1 3 = 2 := 
by
  sorry

end average_cost_is_2_l692_692642


namespace approx_cone_volume_pi_l692_692590

theorem approx_cone_volume_pi (L h : ℝ) (V : ℝ) (hpos : h > 0) (Lpos : L > 0)
  (approx_formula : V = (1/75) * L^2 * h) :
  4 * V = (L^2 * h) := 
begin
  let actual_formula := (1 / (12 * π)) * L^2 * h,
  have actual_volume : V = actual_formula,
  by linarith,
  have pi_value : π = 25 / 4,
  by sorry,
end

end approx_cone_volume_pi_l692_692590


namespace problem_1_problem_2_problem_3_l692_692477

-- Definitions from conditions
def seq_c (c d : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → c n > 0 ∧ d n > 0 ∧ c (n + 1) = (c n + d n) / real.sqrt (c n ^ 2 + d n ^ 2)

def companion_seq (c d : ℕ → ℝ) : Prop :=
seq_c c d

variables {a b : ℕ → ℝ}

-- Conditions
axiom a_pos : ∀ n, a n > 0
axiom b_pos : ∀ n, b n > 0
axiom a_b_comp_seq : companion_seq a b

-- Statement for question (1)
theorem problem_1 (h1 : ∀ n, b n = a n) (h2 : b 1 = real.sqrt 2) : ∀ n, a n = real.sqrt 2 :=
sorry

-- Statement for question (2)
theorem problem_2 (h1 : ∀ n, b (n + 1) = 1 + b n / a n) (h2 : (b 1 / a 1) = (b 1 / a 1)) :
    ∀ n, (b n / a n) ^ 2 = (b 1 / a 1) ^ 2 + n :=
sorry

-- Statement for question (3)
theorem problem_3 (h1 : ∀ n, b (n + 1) = real.sqrt 2 * (b n / a n))
    (h2 : ∀ n, a n = a 1 * (real.pow ((a 1 / a 1), (n - 1)))) : a 1 = real.sqrt 2 ∧ b 1 = real.sqrt 2 :=
sorry

end problem_1_problem_2_problem_3_l692_692477


namespace triangle_height_eq_sum_radii_l692_692915

open Classical

noncomputable theory

-- Define the structure of the problem
structure isosceles_triangle_with_tangents (O₁ O₂ : Type) [MetricSpace O₁] [MetricSpace O₂] :=
  (R r : ℝ)
  (h : ℝ)
  (ABC : Triangle)
  (Cir₁ Cir₂ : Circle)
  (R_gt_r : R > r)
  (radius_of_Cir₁ : Cir₁.radius = R)
  (radius_of_Cir₂ : Cir₂.radius = r)
  (tangent_AC_Cir₁_Cir₂ : Tangent Cir₁ Cir₂ ABC.base)
  (vertex_B_on_other_tangent : ∃ tangent', Tangent Cir₁ Cir₂ tangent' ∧ (ABC.vertex.opposite ∈ tangent'))
  (side_AB_touches_Cir₁ : Touches Cir₁ ABC.side₁)
  (side_BC_touches_Cir₂ : Touches Cir₂ ABC.side₂)
  (height_B_perpendicular_to_AC : ABC.height_from_opposite_vertex_perpendicular_to_base = h)

-- Define the theorem to be proven
theorem triangle_height_eq_sum_radii
  (O₁ O₂ : Type) [MetricSpace O₁] [MetricSpace O₂]
  (tri : isosceles_triangle_with_tangents O₁ O₂)
  : tri.h = tri.R + tri.r := by
  sorry

end triangle_height_eq_sum_radii_l692_692915


namespace find_angle_and_dot_products_l692_692164

open Real EuclideanGeometry

variables {a b c : ℝ^3}
variables {α : ℝ}

noncomputable def vector_conditions
  (a b c : ℝ^3) (ha : ∥a∥ = 2) (hb : ∥b∥ = 4) (hc : c = a - b) (hperp : c ⬝ a = 0) :=
  ∥a∥ = 2 ∧ ∥b∥ = 4 ∧ c = a - b ∧ c ⬝ a = 0

theorem find_angle_and_dot_products
  {a b c : ℝ^3}
  (ha : ∥a∥ = 2) (hb : ∥b∥ = 4) (hc : c = a - b) (hperp : c ⬝ a = 0) :
  let α := Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥))
  in α = π / 3 ∧ a ⬝ (a + 3 • b) = 16 ∧ ∥3 • a + b∥ = 2 * Real.sqrt 19 := by
  sorry

end find_angle_and_dot_products_l692_692164


namespace p_and_q_together_complete_in_10_days_l692_692772

noncomputable def p_time := 50 / 3
noncomputable def q_time := 25
noncomputable def r_time := 50

theorem p_and_q_together_complete_in_10_days 
  (h1 : 1 / p_time = 1 / q_time + 1 / r_time)
  (h2 : r_time = 50)
  (h3 : q_time = 25) :
  (p_time * q_time) / (p_time + q_time) = 10 :=
by
  sorry

end p_and_q_together_complete_in_10_days_l692_692772


namespace initial_volume_kola_solution_l692_692787

-- Initial composition of the kola solution
def initial_composition_sugar (V : ℝ) : ℝ := 0.20 * V

-- Final volume after additions
def final_volume (V : ℝ) : ℝ := V + 3.2 + 12 + 6.8

-- Final amount of sugar after additions
def final_amount_sugar (V : ℝ) : ℝ := initial_composition_sugar V + 3.2

-- Final percentage of sugar in the solution
def final_percentage_sugar (total_sol : ℝ) : ℝ := 0.1966850828729282 * total_sol

theorem initial_volume_kola_solution : 
  ∃ V : ℝ, final_amount_sugar V = final_percentage_sugar (final_volume V) :=
sorry

end initial_volume_kola_solution_l692_692787


namespace average_speed_correct_l692_692391

-- Define the conditions as constants
def distance (D : ℝ) := D
def first_segment_speed := 60 -- km/h
def second_segment_speed := 24 -- km/h
def third_segment_speed := 48 -- km/h

-- Define the function that calculates average speed
noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / first_segment_speed
  let t2 := (D / 3) / second_segment_speed
  let t3 := (D / 3) / third_segment_speed
  let total_time := t1 + t2 + t3
  let total_distance := D
  total_distance / total_time

-- Prove that the average speed is 720 / 19 km/h
theorem average_speed_correct (D : ℝ) (hD : D > 0) : 
  average_speed D = 720 / 19 :=
by
  sorry

end average_speed_correct_l692_692391


namespace combination_15_choose_3_l692_692946

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l692_692946


namespace distance_Q_to_EH_l692_692583

noncomputable def N : ℝ × ℝ := (3, 0)
noncomputable def E : ℝ × ℝ := (0, 6)
noncomputable def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 9
noncomputable def EH_line (y : ℝ) : Prop := y = 6

theorem distance_Q_to_EH :
  ∃ (Q : ℝ × ℝ), circle1 Q.1 Q.2 ∧ circle2 Q.1 Q.2 ∧ Q ≠ (0, 0) ∧ abs (Q.2 - 6) = 19 / 3 := sorry

end distance_Q_to_EH_l692_692583


namespace find_m_plus_n_l692_692083

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l692_692083


namespace find_AD_length_l692_692260

-- We define a structure for the problem that includes all the given conditions
structure TrapezoidProblem :=
(AD BC CD : ℝ)
(M : ℝ)
(H : point)
(AD_parallel_BC : AD ∥ BC)
(AD_eq_HD : AD = HD)
(BC_val : BC = 16)
(CM_val : CM = 8)
(MD_val : MD = 9)

-- The statement defining the proof problem
theorem find_AD_length (p : TrapezoidProblem) : p.AD = 18 :=
by
  sorry -- Proof goes here

end find_AD_length_l692_692260


namespace largest_prime_factor_of_1001_l692_692328

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692328


namespace pencil_count_l692_692306

/-- The number of pencils given the ratio of pens to pencils and the difference in their counts. -/
theorem pencil_count (x : ℕ) (h1 : ∃ x, pens = 5 * x) (h2 : ∃ x, pencils = 6 * x) 
                      (h3 : pencils = pens + 7) : pencils = 42 :=
by
  -- Assume the definitions for pens and pencils based on the given ratio
  obtain ⟨x, hpens⟩ := h1
  obtain ⟨x, hpencils⟩ := h2
  
  -- Substitute given conditions into Lean statements
  rw [hpens] at h3
  rw [hpencils] at h3
  
  -- Calculation of number of pencils
  have h : 6 * x = 5 * x + 7, from h3
  have hx : x = 7, by linarith
  
  -- Number of pencils is 6 * 7 = 42
  rw [hx] at hpencils
  have pencils_42 : pencils = 6 * 7, from hpencils.symm
  rw [hpencils]

  exact pencils_42

end pencil_count_l692_692306


namespace rational_expr_evaluation_l692_692145

theorem rational_expr_evaluation (a b c : ℚ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) (h2 : a + b + c = a * b * c) :
  (a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a) = -3 :=
by
  sorry

end rational_expr_evaluation_l692_692145


namespace heads_at_least_once_in_three_tosses_l692_692568

theorem heads_at_least_once_in_three_tosses :
  let total_outcomes := 8
  let all_tails_outcome := 1
  (1 - (all_tails_outcome / total_outcomes) = (7 / 8)) :=
by
  let total_outcomes := 8
  let all_tails_outcome := 1
  sorry

end heads_at_least_once_in_three_tosses_l692_692568


namespace price_increase_l692_692197

def initial_price := 1.0

theorem price_increase :
  let original_total := initial_price + initial_price in
  let new_price_eggs := initial_price * (1 - 0.02) in
  let new_price_apples := initial_price * (1 + 0.10) in
  let new_total := new_price_eggs + new_price_apples in
  let increase := new_total - original_total in
  (increase / original_total) * 100 = 4 :=
sorry

end price_increase_l692_692197


namespace train_length_is_correct_l692_692441

noncomputable def length_of_train (time_to_cross_bridge : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let
    train_speed_mps := train_speed_kmph * (1000 / 3600)
    distance_crossed := train_speed_mps * time_to_cross_bridge
  in distance_crossed - bridge_length

theorem train_length_is_correct :
  length_of_train 25.997920166386688 150 36 = 109.97920166386688 :=
by
  sorry

end train_length_is_correct_l692_692441


namespace find_m_plus_n_l692_692085

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l692_692085


namespace determine_values_l692_692854

theorem determine_values (A B : ℚ) :
  (A + B = 4) ∧ (2 * A - 7 * B = 3) →
  A = 31 / 9 ∧ B = 5 / 9 :=
by
  sorry

end determine_values_l692_692854


namespace length_of_first_train_l692_692749

theorem length_of_first_train
  (speed_train_1 : ℝ) (speed_train_2 : ℝ)
  (clear_time : ℝ) (length_train_2 : ℝ)
  (h1 : speed_train_1 = 80) (h2 : speed_train_2 = 65)
  (h3 : clear_time = 7.844889650207294)
  (h4 : length_train_2 = 165) :
  let relative_speed := (speed_train_1 + speed_train_2) * (1000 / 3600) in
  let total_distance := relative_speed * clear_time in
  let length_train_1 := total_distance - length_train_2 in
  length_train_1 = 151.019 :=
by
  sorry

end length_of_first_train_l692_692749


namespace primes_correct_arithmetic_mean_of_primes_l692_692102

-- Definitions for the conditions
def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

-- The given list
def numbers := [18, 20, 22, 23, 29, 31, 33]

-- Function to filter prime numbers greater than 20
def primes_gt_20 := numbers.filter (λ n, n > 20 ∧ is_prime n)

-- Checking the specific set of filtered primes
def expected_primes := [23, 29, 31]
theorem primes_correct : primes_gt_20 = expected_primes := by
  sorry

-- The mean of the primes
def mean (l : List Nat) : Rat := (l.sum : Rat) / l.length

-- Expected mean value
def expected_mean := (83 : Rat) / 3

-- The theorem to prove
theorem arithmetic_mean_of_primes :
  mean primes_gt_20 = expected_mean := by
  sorry

end primes_correct_arithmetic_mean_of_primes_l692_692102


namespace largest_prime_factor_1001_l692_692346

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692346


namespace realGDPChange_l692_692686

-- Define all the quantities given in the problem
def Vegetables2014 := 1200
def Fruits2014 := 750
def PriceVegetables2014 := 90000
def PriceFruits2014 := 75000
def Vegetables2015 := 900
def Fruits2015 := 900
def PriceVegetables2015 := 100000
def PriceFruits2015 := 70000

-- Define Nominal GDP for 2014
def NominalGDP2014 := (Vegetables2014 * PriceVegetables2014) + (Fruits2014 * PriceFruits2014)

-- Define Real GDP for 2015 using 2014 prices
def RealGDP2015 := (Vegetables2015 * PriceVegetables2014) + (Fruits2015 * PriceFruits2014)

-- Define the percentage change formula
def PercentageChange (initial final : ℝ) := 100 * ((final - initial) / initial)

-- The main theorem to prove
theorem realGDPChange : (PercentageChange NominalGDP2014 RealGDP2015) = -9.59 :=
by
  have h1 : NominalGDP2014 = 164250 := by sorry
  have h2 : RealGDP2015 = 148500 := by sorry
  have h3 : (148500 - 164250) / 164250 ≈ -0.0959 := by sorry
  have h4 : PercentageChange 164250 148500 = -9.59 := by sorry
  exact h4

end realGDPChange_l692_692686


namespace evaluate_expr_l692_692485

theorem evaluate_expr : 
  (Int.floor ((Real.ceil ((13 / 7 : ℚ) ^ 2 : ℚ) : ℚ) + 17 / 4 : ℚ) : ℤ) = 8 := 
by
  sorry

end evaluate_expr_l692_692485


namespace binomial_coefficient_plus_ten_l692_692033

theorem binomial_coefficient_plus_ten :
  Nat.choose 9 5 + 10 = 136 := 
by
  sorry

end binomial_coefficient_plus_ten_l692_692033


namespace find_b_for_intersection_l692_692455

theorem find_b_for_intersection (b : ℝ) :
  (∀ x : ℝ, bx^2 + 2 * x + 3 = 3 * x + 4 → bx^2 - x - 1 = 0) →
  (∀ x : ℝ, x^2 * b - x - 1 = 0 → (1 + 4 * b = 0) → b = -1/4) :=
by
  intros h_eq h_discriminant h_solution
  sorry

end find_b_for_intersection_l692_692455


namespace book_page_count_l692_692007

theorem book_page_count (d : ℕ) : d = 636 → 635 < 636 → ∀ n ≤ 636, 
(∃ (n : ℕ), (n ≥ 1 ∧ n < 10) ∨ (n ≥ 10 ∧ n < 100) ∨ (n ≥ 100 ∧ (3 * (n - 99) ≤ 635))) → 
∃ (p : ℕ), p = 248 :=
begin
  sorry
end

end book_page_count_l692_692007


namespace B_pow_99_identity_l692_692616

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_99_identity : (B ^ 99) = 1 := by
  sorry

end B_pow_99_identity_l692_692616


namespace quadratic_positive_difference_l692_692281
open Real

theorem quadratic_positive_difference :
  ∀ (x : ℝ), (2*x^2 - 7*x + 1 = x + 31) →
    (abs ((2 + sqrt 19) - (2 - sqrt 19)) = 2 * sqrt 19) :=
by intros x h
   sorry

end quadratic_positive_difference_l692_692281


namespace number_of_pairs_satisfying_equation_l692_692924

theorem number_of_pairs_satisfying_equation : 
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77) = 2 :=
by 
  sorry

end number_of_pairs_satisfying_equation_l692_692924


namespace circumcircle_radius_l692_692188

-- Definition of given values in problem
def AB : ℝ := Real.sqrt 2
def AC : ℝ := 4
def angle_BAC : ℝ := Real.pi / 4 -- 45 degrees in radians

-- Prove the radius of the circumcircle 
theorem circumcircle_radius :
  let BC := Real.sqrt (AB^2 + AC^2 - 2 * AB * AC * Real.cos angle_BAC) in
  let radius := (1 / 2) * (BC / (Real.sin angle_BAC)) in
  radius = Real.sqrt 5 :=
by
  -- proof would go here
  sorry

end circumcircle_radius_l692_692188


namespace center_circumcircle_CEF_on_circle_omega_l692_692740

open Set
open Finset
open Int

/-- Formalize the given geometric problem in Lean -/
theorem center_circumcircle_CEF_on_circle_omega
  (A B C D E F : Type)
  [IncidenceGeometry A B C D E F]  -- assuming an incidence geometry instance
  (trapezoid_ABCD : trapezoid A B C D)
  (inscribed_circle_omega : inscribed_circle (trapezoid A B C D))
  (ray_DC : ray D C beyond C)
  (point_E : point_on_ray E ray_DC)
  (BC_BE_eq : length B C = length B E)
  (circ_intersection_point_F : circle_intersection_point B E omega F outside_BE)
  : lies_on_circle (circumcenter (triangle C E F)) omega := 
sorry

end center_circumcircle_CEF_on_circle_omega_l692_692740


namespace ratio_A_B_l692_692820

variables {a : ℕ → ℕ → ℝ} (S : ℕ → ℝ) (T : ℕ → ℝ)

def A : ℝ := (∑ i in finset.range 40, S i) / 40
def B : ℝ := (∑ j in finset.range 75, T j) / 75

hypothesis S_def : ∀ i, S i = ∑ j in finset.range 75, a i j
hypothesis T_def : ∀ j, T j = ∑ i in finset.range 40, a i j
hypothesis total_sum : ∑ i in finset.range 40, S i = ∑ j in finset.range 75, T j

theorem ratio_A_B : A / B = 15 / 8 :=
by
  sorry

end ratio_A_B_l692_692820


namespace circle_area_approx_error_exceeds_one_l692_692417

theorem circle_area_approx_error_exceeds_one (r : ℝ) : 
  (3.14159 < Real.pi ∧ Real.pi < 3.14160) → 
  2 * r > 25 →  
  |(r * r * Real.pi - r * r * 3.14)| > 1 → 
  2 * r = 51 := 
by 
  sorry

end circle_area_approx_error_exceeds_one_l692_692417


namespace sixty_percent_is_240_l692_692770

variable (x : ℝ)

-- Conditions
def forty_percent_eq_160 : Prop := 0.40 * x = 160

-- Proof problem
theorem sixty_percent_is_240 (h : forty_percent_eq_160 x) : 0.60 * x = 240 :=
sorry

end sixty_percent_is_240_l692_692770


namespace cos_alpha_in_second_quadrant_l692_692953

variable (α : Real) -- Define the variable α as a Real number (angle in radians)
variable (h1 : α > π / 2 ∧ α < π) -- Condition that α is in the second quadrant
variable (h2 : Real.sin α = 2 / 3) -- Condition that sin(α) = 2/3

theorem cos_alpha_in_second_quadrant (α : Real) (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.sin α = 2 / 3) : Real.cos α = - Real.sqrt (1 - (2 / 3) ^ 2) :=
by
  sorry

end cos_alpha_in_second_quadrant_l692_692953


namespace find_c_l692_692692

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_c (c : ℝ) :
  let p1 := (1, 4)
  let p2 := (5, 12)
  let m := midpoint p1 p2
  (2 * m.1 + m.2 = c) → c = 14 :=
by
  let p1 := (1, 4)
  let p2 := (5, 12)
  let m := midpoint p1 p2
  have h : m = (3, 8) := sorry
  have h₁ : 2 * m.1 + m.2 = 14 := sorry
  have h₂ : 2 * m.1 + m.2 = c := by exact sorry
  rw [h₂, h₁]
  exact sorry

end find_c_l692_692692


namespace average_cost_is_2_l692_692641

noncomputable def total_amount_spent (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℕ :=
  apples_quantity * apples_cost + bananas_quantity * bananas_cost + oranges_quantity * oranges_cost

noncomputable def total_number_of_fruits (apples_quantity bananas_quantity oranges_quantity : ℕ) : ℕ :=
  apples_quantity + bananas_quantity + oranges_quantity

noncomputable def average_cost (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℚ :=
  (total_amount_spent apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℚ) /
  (total_number_of_fruits apples_quantity bananas_quantity oranges_quantity : ℚ)

theorem average_cost_is_2 :
  average_cost 12 4 4 2 1 3 = 2 := 
by
  sorry

end average_cost_is_2_l692_692641


namespace triangle_inequality_inequality_l692_692606

theorem triangle_inequality_inequality {a b c : ℝ}
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) :=
sorry

end triangle_inequality_inequality_l692_692606


namespace find_average_l692_692680

theorem find_average (x y : ℝ) (h : x + y = 17) : 
  let m := ((x + 3) + (x + 5) + (y + 2) + 8 + (y + 18)) / 5 in
  m = 14 :=
by
  let m := ((x + 3) + (x + 5) + (y + 2) + 8 + (y + 18)) / 5
  have h1 : (x + 3) + (x + 5) + (y + 2) + 8 + (y + 18) = 2*x + 2*y + 36 := by
    sorry -- Simplification step
  have h2 : 2*x + 2*y + 36 = 2*(x + y) + 36 := by
    sorry -- Algebraic manipulation step
  have h3 : 2*(x + y) + 36 = 2*17 + 36 := by
    rw h -- Substitution step
  have h4 : 2*17 + 36 = 70 := by
    linarith -- Simplification step
  have h5 : m = 70 / 5 := by
    rw [h1, h2, h3, h4] -- Combining all previous steps
  have h6 : 70 / 5 = 14 := by
    norm_num -- Final calculation step
  exact h6 -- Conclusion

end find_average_l692_692680


namespace sufficient_condition_l692_692623

noncomputable section

variables {V : Type*} [normed_group V] [normed_space ℝ V]
variables (a b : V)

def are_nonzero_vectors (a b : V) : Prop := ∥a∥ ≠ 0 ∧ ∥b∥ ≠ 0

def are_collinear (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem sufficient_condition (h : are_nonzero_vectors a b) (h_collinear : a = 2 • b) :
  (a / ∥a∥) = (b / ∥b∥) :=
by 
  sorry

end sufficient_condition_l692_692623


namespace minimum_value_function_l692_692528

variable {m n x y : ℝ}
variable (h1 : 0 < m) (h2 : 0 < n) (h3 : m ≠ n) (h4 : 0 < x) (h5 : 0 < y) (h6 : x + y ≠ 0)

theorem minimum_value_function (hx : x ∈ Ioo 0 1) :
  ∃ x₀, x₀ ∈ Ioo (0 : ℝ) 1 ∧ (∀ x ∈ Ioo (0 : ℝ) 1, f x₀ ≤ f x) ∧ f x₀ = 25 / 3 :=
by
  let f := λ x:ℝ, 4 / (3 * x) + 3 / (1 - x)
  have h7 : ∀ (x y : ℝ), 0 < x → 0 < y → ℝ, m ≠ n → 
              ( (m^2 / x) + (n^2 / y) ≥ (m+n)^2 / (x+y) ) ∧ ((m^2 / x) + (n^2 / y) = (m+n)^2 / (x+y) ↔ (m / x) = (n / y)),
  sorry

  have h8 : f(x) ≥ 25 / 3,
  sorry

  have h9 : ∃ x₀, x₀ ∈ Ioo 0 1 ∧ (∀ x ∈ Ioo 0 1, f x₀ ≤ f x),
  use 2 / 5
  simp [f]
  -- Here you can show that f(2/5) = 25 / 3 using straightforward calculations.
  sorry

  exact ⟨ 2 / 5, ⟨by norm_num, by norm_num⟩, h8, by norm_num⟩

end minimum_value_function_l692_692528


namespace triangle_median_exists_m_l692_692042

-- Definitions for the coordinates of the vertices and midpoints of the triangle's legs.
variables {a b c d : ℝ}  -- coordinates and variables for the triangle and midpoints

-- The right triangle conditions and lines where medians lie
def is_right_triangle (x1 y1 x2 y2) :=
  x1 = a ∧ y1 = b ∧ x2 = a ∧ y2 = b + 2 * c ∧ a - 2 * d = a ∧ b = b ∧ c ≠ 0 ∧ d ≠ 0

-- Slopes of the medians corresponding to the given lines
def slope_conditions (x1 y1 x2 y2 m : ℝ) :=
  (y2 - y1) / (x2 - x1) = 2 * d / c ∧ (y2 - y1) / (x2 - x1) = m * d / (2 * c)

-- Proving the number of possible values for m
theorem triangle_median_exists_m :
  ∃! m : ℝ, ∀ a b c d : ℝ,
  is_right_triangle a b c d → slope_conditions (a, b + c) (a - d, b) m :=
begin
  sorry
end

end triangle_median_exists_m_l692_692042


namespace expected_length_first_group_19_49_l692_692718

noncomputable def expected_first_group_length (ones zeros : ℕ) : ℕ :=
  let total := ones + zeros;
  let pI := 1 / 50;
  let pJ := 1 / 20;
  let expected_I := ones * pI;
  let expected_J := zeros * pJ;
  expected_I + expected_J

theorem expected_length_first_group_19_49 : expected_first_group_length 19 49 = 2.83 :=
by
  sorry

end expected_length_first_group_19_49_l692_692718


namespace total_divisors_of_24_factorial_odd_divisors_of_24_factorial_probability_odd_divisor_of_24_factorial_l692_692463

-- Definitions for the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := sorry -- Assume a list of pairs (prime, exponent)
  in prime_factorization.foldl (λ acc pair => acc * (pair.snd + 1)) 1

def num_odd_divisors (n : ℕ) : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := sorry -- Assume a list of pairs (prime, exponent)
  let odd_factors := prime_factorization.filter (λ pair => pair.fst ≠ 2)
  in odd_factors.foldl (λ acc pair => acc * (pair.snd + 1)) 1

-- Theorems to prove the steps
theorem total_divisors_of_24_factorial : num_divisors (factorial 24) = 23040 := sorry
theorem odd_divisors_of_24_factorial : num_odd_divisors (factorial 24) = 1000 := sorry
theorem probability_odd_divisor_of_24_factorial : 
  (num_odd_divisors (factorial 24) : ℚ) / (num_divisors (factorial 24) : ℚ) = 1 / 23 :=
  by
    sorry

end total_divisors_of_24_factorial_odd_divisors_of_24_factorial_probability_odd_divisor_of_24_factorial_l692_692463


namespace least_years_to_double_l692_692106

theorem least_years_to_double (r : ℝ) (lt_half : r = 0.5) (n : ℕ) (one : n = 1) :
  ∃ (t : ℕ), 2 < (1 + r/n)^(t * n) ∧ (∀ t', t' < t → ¬ (2 < (1 + r/n)^(t' * n))) :=
by
  use 2
  split
  { calc 2 < (1 + r/1)^2 : by sorry }
  { intro t' h
    calc ¬ (2 < (1 + r/1)^(t' * 1)) : by sorry }

end least_years_to_double_l692_692106


namespace jessica_quarters_l692_692993

theorem jessica_quarters (quarters_initial quarters_given : Nat) (h_initial : quarters_initial = 8) (h_given : quarters_given = 3) :
  quarters_initial + quarters_given = 11 := by
  sorry

end jessica_quarters_l692_692993


namespace min_questions_for_five_digit_number_l692_692213

theorem min_questions_for_five_digit_number: ∃ n: ℕ, (∀ N: ℕ, N = 100000 → 2^n ≥ N) ∧ n = 17 :=
by
  have : ∀ N: ℕ, N = 100000 → ∃ n: ℕ, 2^n ≥ N :=
    by
      intro N hN
      use 17
      rw hN
      exact sorry    -- Skip the proof calculation here
  use 17
  constructor
  · exact this 100000 rfl
  · exact rfl

end min_questions_for_five_digit_number_l692_692213


namespace washer_price_l692_692443

def dryer_cost (D : ℕ) : ℕ := D
def washer_cost (D : ℕ) : ℕ := D + 220

theorem washer_price (D : ℕ) (total_cost : ℕ) (w_cost : ℕ) 
  (h_total : total_cost = dryer_cost D + washer_cost D)
  (h_220 : washer_cost D = dryer_cost D + 220)
  (h_total_eq : total_cost = 1200) : 
  w_cost = 710 :=
by {
  have h_dryer_eq : dryer_cost D = 490 := sorry,
  have h_washer_eq : washer_cost D = 710 := sorry,
  exact h_washer_eq,
}

end washer_price_l692_692443


namespace shorter_trisector_l692_692981

noncomputable def shorter_trisector_length (BC AC : ℝ) : ℝ :=
  ├ If (BC = 3 ∧ AC = 4), return (32*real.sqrt 3 - 24) / 13 
  ├ else return 0

theorem shorter_trisector (BC AC : ℝ) (hBC : BC = 3) (hAC : AC = 4) : 
shorter_trisector_length BC AC == (32*real.sqrt 3 - 24) / 13 :=
begin
  sorry
end

end shorter_trisector_l692_692981


namespace evaluate_complex_product_magnitude_l692_692093

noncomputable def complex_magnitude_product : ℂ := (5 * real.sqrt 2 - 3 * complex.I) * (2 * real.sqrt 3 + 4 * complex.I)

theorem evaluate_complex_product_magnitude :
  complex.norm complex_magnitude_product = 2 * real.sqrt 413 := by
  sorry

end evaluate_complex_product_magnitude_l692_692093


namespace probability_minimal_S_l692_692654

open Real

-- Define the sum of absolute differences function
-- S represents the sum of absolute difference of adjacent pairs
def S (pos : Fin 9 → ℕ) : ℕ :=
  let cyclic_pos := List.append (List.ofFn pos) (List.ofFn pos)
  (List.range 9).sum (λ i, abs (cyclic_pos[i] - cyclic_pos[i + 1]))

-- Define the conditions as an array of numbers 1 to 9 
def nine_balls : Set (Fin 9 → ℕ) :=
  {pos | (List.ofFn pos).perm (List.range 1 10)}

-- The main theorem statement
theorem probability_minimal_S :
  (∃ pos ∈ nine_balls, S pos = 16) → 
  (∃ n, n = 1 / 315) :=
sorry

end probability_minimal_S_l692_692654


namespace probability_no_3by3_red_grid_correct_l692_692059

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l692_692059


namespace M_minus_N_equals_sqrt_15_minus_cbrt_4_l692_692521

theorem M_minus_N_equals_sqrt_15_minus_cbrt_4 
  (m n : ℝ) 
  (h1 : ∀ M, M = real.sqrt (m + 3) ↔ M = (m + 3)^(1 / (n - 4)))
  (h2 : ∀ N, N = real.cbrt (n - 2) ↔ N = (n - 2)^(1 / (2 * m - 4 * n + 3))) : 
  (real.sqrt (m + 3) - real.cbrt (n - 2)) = real.sqrt 15 - real.cbrt 4 := 
sorry

end M_minus_N_equals_sqrt_15_minus_cbrt_4_l692_692521


namespace simplify_and_evaluate_l692_692666

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4 * x - 4 = 0) :
  3 * (x - 2) ^ 2 - 6 * (x + 1) * (x - 1) = 6 :=
by
  sorry

end simplify_and_evaluate_l692_692666


namespace rank_commutator_condition_B_diagonalizable_condition_l692_692221

noncomputable def A : Matrix (Fin 3) (Fin 3) ℂ := 
  ![![1, 1, 0], ![0, 1, 0], ![0, 0, 2]]

noncomputable def B (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ := 
  ![![a, 1, 0], ![b, 2, c], ![0, 0, a + 1]]

theorem rank_commutator_condition (a b c : ℂ) : 
  (rank (A * (B a b c) - (B a b c) * A) ≤ 1) ↔ (b = 0 ∧ (a = 2 ∨ c = 0)) :=
sorry

theorem B_diagonalizable_condition (a b c : ℂ) (h : rank (A * (B a b c) - (B a b c) * A) ≤ 1) : 
  (is_diagonalizable (B a b c)) ↔ (b = 0 ∧ c = 0 ∧ a ≠ 2) :=
sorry

end rank_commutator_condition_B_diagonalizable_condition_l692_692221


namespace triangle_geometry_problem_l692_692821

theorem triangle_geometry_problem
  (A B C M D E F G N : Type*)
  [Point A] [Point B] [Point C]
  [Midpoint B C M] [MidpointOfArc BAC D]
  [MidpointOfArc BC E] [IncenterTangencyPoint A B F]
  [IntersectionPoint AE BC G] [OnLineSegment E F N]
  [Perpendicular N B A B]
  (hBN_eq_EM : BN = EM) :
  Perpendicular DF FG := by
  sorry

end triangle_geometry_problem_l692_692821


namespace equivalence_BD_parallel_EF_midpoint_G_l692_692580

variable {Point Line : Type}
variable [Geometry Line Point]

variables (A B C D E F G : Point)

-- Definitions of convex quadrilateral and intersection
def ConvexQuadrilateral (A B C D : Point) : Prop := 
  -- Definition of convex quadrilateral can be given here

def Intersection (P Q : Point) (l₁ l₂ : Line) : Prop := 
  -- Definition of intersection can be given here

-- Conditions
axiom h1 : ConvexQuadrilateral A B C D
axiom h2 : Intersection E (line_through A B) (line_through C D)
axiom h3 : Intersection F (line_through A D) (line_through B C)
axiom h4 : Intersection G (line_through A C) (line_through E F)

-- Equivalent statements to prove
theorem equivalence_BD_parallel_EF_midpoint_G :
  (Parallel (line_through B D) (line_through E F)) ↔ (Midpoint G E F) :=
by
  sorry

end equivalence_BD_parallel_EF_midpoint_G_l692_692580


namespace ratio_radius_diameter_circle_y_l692_692467

theorem ratio_radius_diameter_circle_y 
    (area_x area_y : ℝ) (circumference_x : ℝ)
    (h1 : area_x = area_y) 
    (h2 : circumference_x = 20 * Real.pi) : 
    (∀ (r_y d_y : ℝ), d_y = 2 * r_y → r_y / d_y = (1 : ℝ) / (2 : ℝ)) :=
by
  intro r_y d_y h_d_y
  rw h_d_y
  simp
  linarith

end ratio_radius_diameter_circle_y_l692_692467


namespace expected_length_of_first_group_l692_692720

-- Define the conditions of the problem
def sequence : Finset ℕ := {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

-- Expected length of the first group
def expected_length_first_group : ℝ := 2.83

-- The formal statement of the proof problem
theorem expected_length_of_first_group (seq : Finset ℕ) (h1 : seq.card = 68) (h2 : seq.filter (λ x, x = 1) = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
(h3 : seq.filter (λ x, x = 0) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) :
  let X := 
  (Finset.sum (Finset.range 19) (λ k, if k = 1 then (1/50 : ℝ) else 0)) +
  (Finset.sum (Finset.range 49) (λ m, if m = 1 then (1/20 : ℝ) else 0)) in 
  ∃ x : ℝ, ∑ x = expected_length_first_group := 
by sorry
 
end expected_length_of_first_group_l692_692720


namespace platform_length_is_520_l692_692782

noncomputable def speed_km_per_hr := 55 -- Speed of train in km/hr
noncomputable def length_train := 470 -- Length of train in meters
noncomputable def time_to_cross := 64.79481641468682 -- Time to cross the platform in seconds

-- Convert speed from km/hr to m/s
noncomputable def speed_m_per_s := speed_km_per_hr * 1000 / 3600

-- Define the total distance covered by the train to cross the platform
noncomputable def total_distance_covered := speed_m_per_s * time_to_cross

-- Calculate the length of the platform
noncomputable def length_platform := total_distance_covered - length_train

theorem platform_length_is_520 :
  length_platform ≈ 520 :=
by
  sorry

end platform_length_is_520_l692_692782


namespace average_coins_collected_per_day_l692_692998

theorem average_coins_collected_per_day :
  ∀ (n : ℕ), (n = 4) → 
    (∀ (init : ℕ), (init = 12) → 
      ∀ (s : list ℕ), 
        (s = [12, 24, 48, 96]) → 
          (list.sum s / 4 = 45)) :=
begin
  intros n hn init hinit s hs,
  sorry,
end

end average_coins_collected_per_day_l692_692998


namespace cooking_time_per_side_l692_692837

-- Defining the problem conditions
def total_guests : ℕ := 30
def guests_wanting_2_burgers : ℕ := total_guests / 2
def guests_wanting_1_burger : ℕ := total_guests / 2
def burgers_per_guest_2 : ℕ := 2
def burgers_per_guest_1 : ℕ := 1
def total_burgers : ℕ := guests_wanting_2_burgers * burgers_per_guest_2 + guests_wanting_1_burger * burgers_per_guest_1
def burgers_per_batch : ℕ := 5
def total_batches : ℕ := total_burgers / burgers_per_batch
def total_cooking_time : ℕ := 72
def time_per_batch : ℕ := total_cooking_time / total_batches
def sides_per_burger : ℕ := 2

-- the theorem to prove the desired cooking time per side
theorem cooking_time_per_side : (time_per_batch / sides_per_burger) = 4 := by {
    -- Here we would enter the proof steps, but this is omitted as per the instructions.
    sorry
}

end cooking_time_per_side_l692_692837


namespace september_has_five_thursdays_l692_692673

theorem september_has_five_thursdays (N : ℕ) 
  (july_31_days : true) 
  (september_31_days : true) 
  (july_five_fridays: true) : 
  Exists (λ d: ℕ, d = nat.days_in_week ∧ d = 4 ∧ ∀ m : ℕ, m = 9 -> (number_of_days_in_m_thursday m = 5)) 
:= 
  sorry

end september_has_five_thursdays_l692_692673


namespace number_of_pairs_satisfying_equation_l692_692922

theorem number_of_pairs_satisfying_equation : 
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77) = 2 :=
by 
  sorry

end number_of_pairs_satisfying_equation_l692_692922


namespace sum_even_1_to_200_l692_692759

open Nat

/-- The sum of all even numbers from 1 to 200 is 10100. --/
theorem sum_even_1_to_200 :
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  sum = 10100 :=
by
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  show sum = 10100
  sorry

end sum_even_1_to_200_l692_692759


namespace conjugate_product_real_not_conjugate_example_l692_692395

open Complex

theorem conjugate_product_real (a b : ℝ) (z1 z2 : ℂ) (h1: z1 = a + b * I) (h2: z2 = a - b * I): z1 * z2 ∈ ℝ :=
by
  have h3 : z1 * z2 = (a + b * I) * (a - b * I) := by rw [h1, h2]
  sorry

-- Partially proving that the reverse implication does not hold
theorem not_conjugate_example : ∃ (z1 z2 : ℂ), z1 * z2 ∈ ℝ ∧ ¬(conj z1 = z2) :=
by
  use [I, 2 * I]
  sorry

end conjugate_product_real_not_conjugate_example_l692_692395


namespace largest_prime_factor_of_1001_l692_692333

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692333


namespace find_length_QR_l692_692917

noncomputable def triangle_similarity (XYZ PQR : ℝ) (a b c d e : ℝ) (shared_angle : ℝ := 120) : Prop :=
  ∀ (XY YZ XZ PQ PR QR : ℝ), 
  (XY = a) ∧ (YZ = b) ∧ (XZ = c) ∧ (PQ = d) ∧ (PR = e) ∧ 
  (shared_angle = 120) ∧ 
  (QR = b / (a / d))

theorem find_length_QR (XY YZ XZ PQ PR : ℝ)
  (hXY : XY = 8) (hYZ : YZ = 18) (hXZ : XZ = 12) 
  (hPQ : PQ = 4) (hPR : PR = 9) 
  (shared_angle : ℝ := 120) :
  ∃ QR, QR = 9 :=
by 
  use 9
  rw [←hYZ, ←(hXY / hPQ)]
  exact sorry
 
end find_length_QR_l692_692917


namespace circles_intersect_l692_692852

def center_radius (a b c : ℝ) : (ℝ × ℝ) × ℝ :=
  let h := (a^2 + b^2 - 4*c) / 4
  ((-a/2, -b/2), Real.sqrt h)

noncomputable def circle1_center_radius := center_radius 2 8 (-8)
noncomputable def circle2_center_radius := center_radius (-4) (-4) (-1)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circles_intersect :
  let ((c1x, c1y), r1) := circle1_center_radius
  let ((c2x, c2y), r2) := circle2_center_radius
  let d := distance c1x c1y c2x c2y
  |r1 - r2| < d ∧ d < r1 + r2 :=
  by
    sorry

end circles_intersect_l692_692852


namespace vowel_sequences_count_l692_692796

theorem vowel_sequences_count : 
  let vowels := ['A', 'E', 'I', 'O', 'U'] in
  (∀ v ∈ vowels, list.count v ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'E', 'E', 'E', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'U', 'U', 'U', 'U', 'U'] > 0) →
  nat.factorial 5 = 120 :=
by {
  intro vowels h,
  unfold list.count at h,
  exact nat.factorial_five,
}

end vowel_sequences_count_l692_692796


namespace trajectory_equation_slope_range_l692_692598

def point (α : Type*) := prod α α

variables {α : Type*} [linear_ordered_field α]

def A : point α := (2, 0)
def B : point α := (-2, 0)

noncomputable def slope (P Q : point α) : α := 
  (Q.2 - P.2) / (Q.1 - P.1)

def trajectory (P : point α) := 
  slope P A * slope P B = -3 / 4

theorem trajectory_equation {P : point α} 
  (h : trajectory P) : 
  ∀ P : point α, P.1^2 / 4 + P.2^2 / 3 = 1 :=
sorry

theorem slope_range (m : α) (M : point α) 
  (hM : M.1 = A.1 ∨ M.2 = A.2) :
  -1/8 ≤ slope M A ∧ slope M A ≤ 1/8 ∧ slope M A ≠ 0 :=
sorry

end trajectory_equation_slope_range_l692_692598


namespace total_surface_area_of_right_pyramid_l692_692432

-- Define the conditions of the problem
def square_base_side_length : ℝ := 6
def pyramid_height : ℝ := 15

-- The main theorem to be proved
theorem total_surface_area_of_right_pyramid 
  (a : ℝ) (b : ℝ) 
  (side_length_eq : a = square_base_side_length) 
  (height_eq : b = pyramid_height) : 
  let base_area := a * a,
      slant_height := Real.sqrt (b^2 + (a / 2)^2),
      triangle_area := (1 / 2) * a * slant_height,
      total_surface_area := base_area + 4 * triangle_area
  in total_surface_area = 219.564 := sorry

end total_surface_area_of_right_pyramid_l692_692432


namespace correct_description_of_sperm_l692_692815

def sperm_carries_almost_no_cytoplasm (sperm : Type) : Prop := sorry

theorem correct_description_of_sperm : sperm_carries_almost_no_cytoplasm sperm := 
sorry

end correct_description_of_sperm_l692_692815


namespace sum_h_k_a_b_l692_692294

-- Defining h, k, a, and b with their respective given values
def h : Int := -4
def k : Int := 2
def a : Int := 5
def b : Int := 3

-- Stating the theorem to prove \( h + k + a + b = 6 \)
theorem sum_h_k_a_b : h + k + a + b = 6 := by
  /- Proof omitted as per instructions -/
  sorry

end sum_h_k_a_b_l692_692294


namespace initial_orchid_bushes_l692_692313

def final_orchid_bushes : ℕ := 35
def orchid_bushes_to_be_planted : ℕ := 13

theorem initial_orchid_bushes :
  final_orchid_bushes - orchid_bushes_to_be_planted = 22 :=
by
  sorry

end initial_orchid_bushes_l692_692313


namespace symmetric_point_y_axis_square_root_of_ab_l692_692895

theorem symmetric_point_y_axis_square_root_of_ab 
  (a b : ℝ)
  (h1 : a + b = -3)
  (h2 : 1 - b = -1)
  (P : ℝ × ℝ := (3, -1))
  (Q : ℝ × ℝ := (a + b, 1 - b)) :
  real.sqrt (-a * b) = real.sqrt 10 :=
sorry

end symmetric_point_y_axis_square_root_of_ab_l692_692895


namespace player1_winning_strategy_l692_692846

/--
Player 1 has a winning strategy if and only if N is not an odd power of 2,
under the game rules where players alternately subtract proper divisors
and a player loses when given a prime number or 1.
-/
theorem player1_winning_strategy (N: ℕ) : 
  ¬ (∃ k: ℕ, k % 2 = 1 ∧ N = 2^k) ↔ (∃ strategy: ℕ → ℕ, ∀ n ≠ 1, n ≠ prime → n - strategy n = m) :=
sorry

end player1_winning_strategy_l692_692846


namespace combined_effective_tax_rate_l692_692651

theorem combined_effective_tax_rate 
  (income_mork income_mindy : ℝ) 
  (mindy_earns_four_times : income_mindy = 4 * income_mork)
  (flambo_brackets : (0 <= income_mork) → 
                      if income_mork <= 20000 then income_mork * 0.1
                      else if income_mork ≤ 50000 then (20000 * 0.1 + (income_mork - 20000) * 0.15)
                      else (20000 * 0.1 + 30000 * 0.15 + (income_mork - 50000) * 0.2))
  (flambo_deduction : (income_mork >= 0) → income_mork - 5000)
  (flambo_percent_deduction : (income_mork > 25000) → (income_mork - 25000) * 0.05)
  (zingo_brackets : (0 <= income_mindy) → 
                     if income_mindy <= 10000 then income_mindy * 0.1
                      else if income_mindy ≤ 20000 then (10000 * 0.1 + (income_mindy - 10000) * 0.15)
                      else if income_mindy ≤ 40000 then (10000 * 0.1 + 10000 * 0.15 + (income_mindy - 20000) * 0.2)
                      else if income_mindy ≤ 60000 then (10000 * 0.1 + 10000 * 0.15 + 20000 * 0.2 + (income_mindy - 40000) * 0.3)
                      else (10000 * 0.1 + 10000 * 0.15 + 20000 * 0.2 + 20000 * 0.3 + (income_mindy - 60000) * 0.35))
  (zingo_deduction : (income_mindy >= 0) → income_mindy - 7000)
  (zingo_percent_deduction : (income_mindy > 30000) → (income_mindy - 30000) * 0.1)
  (combined_effective_tax_rate : ((income_mork + income_mindy) / 
                      (income_mork + income_mindy)): ℝ  = (13.4 / 100)): 
  sorry 

end combined_effective_tax_rate_l692_692651


namespace probability_laurent_greater_than_chloe_l692_692466

def chloe_distribution : MeasureTheory.ProbMeasure ℝ :=
  MeasureTheory.Measure.uniform [0, 1000]

def laurent_distribution : MeasureTheory.ProbMeasure ℝ :=
  let dist1 := MeasureTheory.Measure.uniform [0, 2000]
  let dist2 := MeasureTheory.Measure.uniform [0, 3000]
  (2/3) • dist1 + (1/3) • dist2

theorem probability_laurent_greater_than_chloe :
  MeasureTheory.MeasureTheory.Measure.probability (λ (x y : ℝ), y > x)
    (MeasureTheory.Measure.prod chloe_distribution laurent_distribution) = 2 / 3 :=
sorry

end probability_laurent_greater_than_chloe_l692_692466


namespace work_hours_goal_l692_692560

theorem work_hours_goal (H₀ : ℝ) (W₀ : ℝ) (E : ℝ) (missed_weeks : ℝ) (remaining_weeks : ℝ) :
  H₀ = 25 → W₀ = 15 → missed_weeks = 3 → remaining_weeks = W₀ - missed_weeks → E = 4000 →
  remaining_weeks = 12 →
  (∃ H : ℝ, H = 333.33 ∧ H * remaining_weeks = E) :=
by
  intros H₀_def W₀_def missed_weeks_def remaining_weeks_def E_def remaining_weeks_cond
  use 333.33
  split
  . rfl
  . rw [remaining_weeks_def, H₀_def, W₀_def, missed_weeks_def, remaining_weeks_cond, E_def]
    norm_num
    sorry

end work_hours_goal_l692_692560


namespace books_combination_l692_692941

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l692_692941


namespace zach_saved_money_l692_692386

-- Definitions of known quantities
def cost_of_bike : ℝ := 100
def weekly_allowance : ℝ := 5
def mowing_earnings : ℝ := 10
def babysitting_rate : ℝ := 7
def babysitting_hours : ℝ := 2
def additional_earnings_needed : ℝ := 6

-- Calculate total earnings for this week
def total_earnings_this_week : ℝ := weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)

-- Prove that Zach has already saved $65
theorem zach_saved_money : (cost_of_bike - total_earnings_this_week - additional_earnings_needed) = 65 :=
by
  -- Sorry used as placeholder to skip the proof
  sorry

end zach_saved_money_l692_692386


namespace find_correct_average_of_numbers_l692_692679

variable (nums : List ℝ)
variable (n : ℕ) (avg_wrong avg_correct : ℝ) (wrong_val correct_val : ℝ)

noncomputable def correct_average (nums : List ℝ) (wrong_val correct_val : ℝ) : ℝ :=
  let correct_sum := nums.sum - wrong_val + correct_val
  correct_sum / nums.length

theorem find_correct_average_of_numbers
  (h₀ : n = 10)
  (h₁ : avg_wrong = 15)
  (h₂ : wrong_val = 26)
  (h₃ : correct_val = 36)
  (h₄ : avg_correct = 16)
  (nums : List ℝ) :
  avg_wrong * n - wrong_val + correct_val = avg_correct * n := 
sorry

end find_correct_average_of_numbers_l692_692679


namespace product_of_ratios_eq_102_l692_692601

-- Definitions of the geometric setup
variables {A B C A' B' C' O : Type*}
variables (BC AC AB : set Type*) (intersects_AT : A -> BC -> A' -> Prop)
variables (intersects_BT : B -> AC -> B' -> Prop) (intersects_CT : C -> AB -> C' -> Prop)
variable (intersects_AT_BT_CT : O -> AA' -> BB' -> CC' -> Prop)

-- Given conditions definitions
def AO_OA' (A O A' : Type*) (r : nnreal) : Prop := r → A + O = O + A' ∧ r_nat = 100
def BO_OB' (B O B' : Type*) (r : nnreal) : Prop := r → B + O = O + B' ∧ r_nat = 100
def CO_OC' (C O C' : Type*) (r : nnreal) : Prop := r → C + O = O + C' ∧ r_nat = 100


-- The theorem to state and prove
theorem product_of_ratios_eq_102 : 
  (∀ (rAO rBO rCO : nnreal), (intersects_AT_BT_CT O AA' BB' CC' = 100) → rAO * rBO * rCO = 102) :=
begin
  sorry,
end

end product_of_ratios_eq_102_l692_692601


namespace probability_no_3x3_red_square_l692_692072

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l692_692072


namespace parts_of_milk_in_drink_A_l692_692789

-- Define the parts of drink A and B
def parts_of_drink_A (m : ℕ) : ℕ := m + 3
def parts_of_drink_B : ℕ := 28

-- Define the conversion process from A to B
def convert_drink_A_to_B (vol_A vol_B added_juice : ℕ) : Prop :=
  vol_A + added_juice = vol_B

-- Define the proportion of fruit juice and milk in drink B
def proportions_in_drink_B (b : ℕ) : (ℕ × ℕ) :=
  (4 * b / 7, 3 * b / 7)

-- The proof statement
theorem parts_of_milk_in_drink_A (m : ℕ) (added_juice : ℕ) :
  convert_drink_A_to_B 21 parts_of_drink_B 7 →
  proportions_in_drink_B parts_of_drink_B = (16, 12) →
  m = 12 :=
  begin
    -- Here the proof will use the conditions and show that m equals 12
    sorry,
  end

end parts_of_milk_in_drink_A_l692_692789


namespace largest_prime_factor_1001_l692_692341

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692341


namespace circumcenter_CEF_lies_on_ω_l692_692733

-- Definitions for given conditions
variables {A B C D E F O : Point}
variables {ω : Circle}
variables (H1: Trapezoid ABCD) (H2: Inscribed ABCD ω)
variables (H3: PointOnRayBeyond E C D) (H4: BC = BE)
variables (H5: IntersectsAt BE ω F) (H6: Isosceles BEC)

-- The theorem statement
theorem circumcenter_CEF_lies_on_ω :
  CenterOfCircumcircle C E F = O ∧ PointOnCircle O ω :=
sorry

end circumcenter_CEF_lies_on_ω_l692_692733


namespace largest_prime_factor_1001_l692_692348

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692348


namespace quadratic_completing_square_l692_692707

theorem quadratic_completing_square :
  ∃ (a b c : ℚ), a = 12 ∧ b = 6 ∧ c = 1296 ∧ 12 + 6 + 1296 = 1314 ∧
  (12 * (x + b)^2 + c = 12 * x^2 + 144 * x + 1728) :=
by
  sorry

end quadratic_completing_square_l692_692707


namespace unit_price_ratio_correct_l692_692832

variable (v p : Real)

def volume_y : Real := v
def price_y : Real := p

def volume_x : Real := 1.25 * volume_y v
def price_x : Real := 0.88 * price_y p

noncomputable def unit_price_ratio : Real :=
  (price_x v p / volume_x v) / (price_y p / volume_y v)

theorem unit_price_ratio_correct : unit_price_ratio v p = 88 / 125 := by
  sorry

end unit_price_ratio_correct_l692_692832


namespace noncongruent_triangle_count_l692_692558

theorem noncongruent_triangle_count (a b : ℝ) (θ : ℝ) (h₀ : a = 20) (h₁ : b = 17) (h₂ : θ = real.pi / 3) :
  ∃! (t : Triangle), (t.has_side a ∧ t.has_side b ∧ t.has_angle θ) :=
sorry

end noncongruent_triangle_count_l692_692558


namespace find_k_l692_692160

noncomputable def inequality (k x : ℝ) : Prop :=
  ((k^2 + 6*k + 14) * x - 9) * ((k^2 + 28) * x - 2*k^2 - 12*k) < 0

theorem find_k (k : ℝ) :
  (∀ x : ℝ, inequality k x → (M ∩ Z = {1})) →
  (k < -14 ∨ (2 < k ∧ k ≤ 14 / 3)) :=
by
  sorry

end find_k_l692_692160


namespace line_through_points_a_minus_b_l692_692185

theorem line_through_points_a_minus_b :
  ∃ a b : ℝ, 
  (∀ x, (x = 3 → 7 = a * 3 + b) ∧ (x = 6 → 19 = a * 6 + b)) → 
  a - b = 9 :=
by
  sorry

end line_through_points_a_minus_b_l692_692185


namespace books_combination_l692_692942

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l692_692942


namespace largest_prime_factor_of_1001_l692_692374

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692374


namespace pure_imaginary_solution_l692_692862

theorem pure_imaginary_solution (k : ℝ) : 
  let x := k * Complex.I in
  (x^5 - 4 * x^4 + 6 * x^3 - 50 * x^2 - 100 * x - 120 = 0) ↔ (x = Complex.I * Real.sqrt 14 ∨ x = -Complex.I * Real.sqrt 14) :=
sorry

end pure_imaginary_solution_l692_692862


namespace trapezoid_extension_l692_692681

theorem trapezoid_extension (AD BC AB CD MB MC : ℝ) (hAD : AD = 1.8) (hBC : BC = 1.2) 
  (hAB : AB = 1.5) (hCD : CD = 1.2) (h_sim : MB = 2 * AB) (h_sim2 : MC = 2 * CD) : 
  MB = 3 ∧ MC = 2.4 :=
by {
  subst hAD,
  subst hBC,
  subst hAB,
  subst hCD,
  subst h_sim,
  subst h_sim2,
  sorry
}

end trapezoid_extension_l692_692681


namespace biology_marks_l692_692045

theorem biology_marks (E M P C: ℝ) (A: ℝ) (N: ℕ) 
  (hE: E = 96) (hM: M = 98) (hP: P = 99) (hC: C = 100) (hA: A = 98.2) (hN: N = 5):
  (E + M + P + C + B) / N = A → B = 98 :=
by
  intro h
  sorry

end biology_marks_l692_692045


namespace collin_puts_in_savings_l692_692032

noncomputable def total_savings : ℚ :=
let lightweight_home := 12 * 0.15 in
let medium_grandparents := (3 * 12) * 0.25 in
let heavyweight_neighbor := 46 * 0.35 in
let mixed_weight_office := 
  let total_office := 250 in
  let lightweight_office := (50 / 100) * total_office * 0.15 in
  let medium_office := (30 / 100) * total_office * 0.25 in
  let heavyweight_office := (20 / 100) * total_office * 0.35 in
  lightweight_office + medium_office + heavyweight_office in
(lightweight_home + medium_grandparents + heavyweight_neighbor + mixed_weight_office) / 2

theorem collin_puts_in_savings : total_savings = 41.45 := by
  -- Include the details of how this theorem is proven.
  sorry

end collin_puts_in_savings_l692_692032


namespace a_le_one_l692_692910

variable {a : ℝ}

def p : Prop := ∃ x ∈ (Set.Icc 1 2), x^2 - a < 0

def neg_p : Prop := ∀ x ∈ (Set.Icc 1 2), x^2 - a ≥ 0

theorem a_le_one (h : neg_p) : a ≤ 1 := by
  sorry

end a_le_one_l692_692910


namespace largest_prime_factor_of_1001_l692_692371

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692371


namespace triangle_angle_A_triangle_length_b_l692_692605

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (m n : ℝ × ℝ)
variable (S : ℝ)

theorem triangle_angle_A (h1 : a = 7) (h2 : c = 8) (h3 : m = (1, 7 * a)) (h4 : n = (-4 * a, Real.sin C))
  (h5 : m.1 * n.1 + m.2 * n.2 = 0) : 
  A = Real.pi / 6 := 
  sorry

theorem triangle_length_b (h1 : a = 7) (h2 : c = 8) (h3 : (7 * 8 * Real.sin B) / 2 = 16 * Real.sqrt 3) :
  b = Real.sqrt 97 :=
  sorry

end triangle_angle_A_triangle_length_b_l692_692605


namespace measure_of_angle_F_l692_692984

theorem measure_of_angle_F (D E F : ℝ) (hD : D = E) 
  (hF : F = D + 40) (h_sum : D + E + F = 180) : F = 140 / 3 + 40 :=
by
  sorry

end measure_of_angle_F_l692_692984


namespace sum_m_n_l692_692054

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l692_692054


namespace opposite_of_neg_2023_l692_692698

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l692_692698


namespace find_omega_l692_692537

variable (A : ℝ) (ω : ℝ) (φ : ℝ)

theorem find_omega (h1 : A ≠ 0) (h2 : ω > 0)
  (h3 : 0 ≤ φ ∧ φ ≤ π)
  (h4 : ∀ x, A * sin (ω * x + φ) = A * sin (ω * -x + φ)) -- f is even
  (h5 : ∀ x, A * cos (ω * (x - π / 4)) = -A * cos (ω * (x + π / 4))) : ω = 2 :=
sorry

end find_omega_l692_692537


namespace average_cost_of_fruit_l692_692639

theorem average_cost_of_fruit : 
  (12 * 2 + 4 * 1 + 4 * 3) / (12 + 4 + 4) = 2 := 
by
  -- Given conditions as definitions
  let cost_apple := 2     -- cost per apple
  let cost_banana := 1    -- cost per banana
  let cost_orange := 3    -- cost per orange
  let qty_apples := 12    -- number of apples bought
  let qty_bananas := 4    -- number of bananas bought
  let qty_oranges := 4    -- number of oranges bought
  
  -- Average cost calculation
  have total_cost := qty_apples * cost_apple + qty_bananas * cost_banana + qty_oranges * cost_orange
  have total_qty := qty_apples + qty_bananas + qty_oranges
  have average_cost := total_cost.toRat / total_qty.toRat
  
  show average_cost = 2 by sorry

end average_cost_of_fruit_l692_692639


namespace proof_task_l692_692911

variable (a : ℕ → ℚ)
variable (b : ℕ → ℚ)
variable (T : ℕ → ℚ)
variable (λ : ℚ)

def sequence_a := ∀ n : ℕ, a 0 = 1 ∧ (∀ n > 0, a (n+1) = a n / (a n + 3))
def is_geometric_sequence (a : ℕ → ℚ) := 
  ∀ n : ℕ, ((1 / a n) + 1/2) = (3 / 2) * 3 ^ (n - 1)
def find_an_general_formula : Prop :=
  ∀ n : ℕ, a n = 2 / (3 ^ n - 1)

def sequence_b := ∀ n : ℕ, b n = (3^n - 1) * (n / 2^n) * (2 / (3^n - 1))
def sum_T := ∀ n : ℕ, T n = 4 - (n + 2) / (2^(n-1))

def range_of_λ := 
  ∀ n : ℕ, ((-1)^n * λ < T n + n / 2^(n-1)) ↔ (-2 < λ ∧ λ < 3)

theorem proof_task :
  sequence_a a →
  is_geometric_sequence a →
  find_an_general_formula →
  sequence_b b →
  sum_T T →
  range_of_λ λ T :=
  sorry

end proof_task_l692_692911


namespace sum_first_1998_no_sum_eq_2001_l692_692775

def sequence (n : ℕ) : ℕ :=
if n = 0 then 0 else
let k := (n - 1).bits.idx 1 + 1 - 1 in
if (n - 1).bits.default! k = tt then 2 else 1

noncomputable def S (n : ℕ) : ℕ :=
(nat.sum (finset.range n) (λ k, sequence (k + 1)))

theorem sum_first_1998 :
  S 1998 = 3985 := sorry

theorem no_sum_eq_2001 :
  ¬ ∃ n : ℕ, S n = 2001 := sorry

end sum_first_1998_no_sum_eq_2001_l692_692775


namespace number_of_possible_values_of_k_l692_692831

-- Define the primary conditions and question
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def quadratic_roots_prime (p q k : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 72 ∧ p * q = k

theorem number_of_possible_values_of_k :
  ¬ ∃ k : ℕ, ∃ p q : ℕ, quadratic_roots_prime p q k :=
by
  sorry

end number_of_possible_values_of_k_l692_692831


namespace find_segment_length_l692_692602

theorem find_segment_length (DE DF EF DR DS ER : ℝ) (R S : Point ℝ) (U V : Point ℝ) (D E F : Point ℝ) 
  (h1 : triangle D E F)
  (h2 : D.dist E = 130)
  (h3 : D.dist F = 150)
  (h4 : E.dist F = 140)
  (h5 : angle_bisector D R EF)
  (h6 : angle_bisector E S DF)
  (h7 : perp_from F U ER)
  (h8 : perp_from F V DS) :
  D.dist U V = 80 :=
sorry

end find_segment_length_l692_692602


namespace find_n_eq_5_l692_692710

variable {a_n b_n : ℕ → ℤ}

def a (n : ℕ) : ℤ := 2 + 3 * (n - 1)
def b (n : ℕ) : ℤ := -2 + 4 * (n - 1)

theorem find_n_eq_5 :
  ∃ n : ℕ, a n = b n ∧ n = 5 :=
by
  sorry

end find_n_eq_5_l692_692710


namespace complement_union_l692_692913

open Set

namespace ProofFormalization

/-- Declaration of the universal set U, and sets A and B -/
def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

def complement {α : Type*} (s t : Set α) : Set α := t \ s

/-- Theorem statement that proves the complement of A ∪ B with respect to U is {5} -/
theorem complement_union :
  complement (A ∪ B) U = {5} :=
by
  sorry

end ProofFormalization

end complement_union_l692_692913


namespace problem_1_problem_2_l692_692036

noncomputable def complex_z (θ : ℝ) : ℂ :=
  -3 * Complex.cos θ + 2 * Complex.sin θ * Complex.i

-- Problem 1: Given θ = 4/3 π, prove |z| = sqrt(21)/2
theorem problem_1 (θ : ℝ) (hθ : θ = (4 / 3) * Real.pi) :
  Complex.abs (complex_z θ) = Real.sqrt 21 / 2 :=
by
  rw [complex_z, hθ]
  sorry

-- Problem 2: Given 2 * sin θ = cos θ, prove the expression equals 2/3
theorem problem_2 (θ : ℝ) (hθ : 2 * Real.sin θ = Real.cos θ) :
  (2 * Real.cos (θ / 2) ^ 2 - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 2 / 3 :=
by
  rw [hθ]
  sorry

end problem_1_problem_2_l692_692036


namespace total_money_made_l692_692829

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l692_692829


namespace twin_functions_9_l692_692569

def f (x : ℝ) : ℝ := x^2 + 1

def twin_functions_count : ℕ := 9

theorem twin_functions_9 (f : ℝ → ℝ)
  (h_expr : ∀ x, f x = x^2 + 1)
  (h_range : ∃ x, f x = 5 ∧ ∃ x, f x = 10):
  (∃ s : set ℝ, f '' s = {5, 10}) → 
  (∃ D : finset (set ℝ), D.card = 9 ∧ ∀ s ∈ D, f '' s = {5, 10}) :=
sorry

end twin_functions_9_l692_692569


namespace multiply_decimals_l692_692462

noncomputable def real_num_0_7 : ℝ := 7 * 10⁻¹
noncomputable def real_num_0_3 : ℝ := 3 * 10⁻¹
noncomputable def real_num_0_21 : ℝ := 0.21

theorem multiply_decimals :
  real_num_0_7 * real_num_0_3 = real_num_0_21 :=
sorry

end multiply_decimals_l692_692462


namespace parabola_equation_trajectory_midpoint_l692_692908

-- Given data and conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola_x3 (p : ℝ) : Prop := ∃ y, parabola p 3 y
def distance_point_to_line (x d : ℝ) : Prop := x + d = 5

-- Prove that given these conditions, the parabola equation is y^2 = 8x
theorem parabola_equation (p : ℝ) (h1 : point_on_parabola_x3 p) (h2 : distance_point_to_line (3 + p / 2) 2) : p = 4 :=
sorry

-- Prove the equation of the trajectory for the midpoint of the line segment FP
def point_on_parabola (p x y : ℝ) : Prop := y^2 = 8 * x
theorem trajectory_midpoint (p x y : ℝ) (h1 : parabola 4 x y) : y^2 = 4 * (x - 1) :=
sorry

end parabola_equation_trajectory_midpoint_l692_692908


namespace P_neg1_equals_n_plus_1_l692_692236

variable {R : Type*} [CommRing R]

noncomputable def P (n : ℕ) : R[X] :=
  sorry  -- The polynomial P defined as per the condition

theorem P_neg1_equals_n_plus_1 {n : ℕ} 
  (h_deg : ∀ k ∈ finset.range (n+1) + 1, eval k (P n) = (1 / (k : R))) :
  eval (-1 : ℝ) (P n) = ↑(n + 1) :=
sorry

end P_neg1_equals_n_plus_1_l692_692236


namespace zero_inside_convex_pentagon_l692_692501

theorem zero_inside_convex_pentagon 
  (c₁ c₂ c₃ c₄ c₅ : ℂ) 
  (h : (1 / c₁) + (1 / c₂) + (1 / c₃) + (1 / c₄) + (1 / c₅) = 0)
  (convex_pentagon : convex_hull ℂ ({c₁, c₂, c₃, c₄, c₅} : set ℂ).finite.compl) 
  :
  0 ∈ convex_hull ℂ ({c₁, c₂, c₃, c₄, c₅} : set ℂ) :=
sorry

end zero_inside_convex_pentagon_l692_692501


namespace range_of_R_l692_692781

def sphere_surface (x y z : ℝ) : Prop :=
  x^2 + y^2 + (z - 1)^2 = 1

def line_through_Q_P (Q P : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (qx, qy, qz) := Q in
  let (px, py, pz) := P in
  (px + t * (qx - px), py + t * (qy - py), pz + t * (qz - pz))

def plane_z_eq_0 (x y z : ℝ) : Prop :=
  z = 0

theorem range_of_R (Q : ℝ × ℝ × ℝ)
  (hq : sphere_surface Q.1 Q.2 Q.3)
  (hne : Q ≠ (0, 0, 2))
  (P : ℝ × ℝ × ℝ) (hp : P = (1, 0, 2)) :
  ∃ R : ℝ × ℝ, let l := line_through_Q_P Q P in
  plane_z_eq_0 (l (2 / (2 - Q.3))).1 (l (2 / (2 - Q.3))).2 (l (2 / (2 - Q.3))).3 ∧
  let (xr, yr, zr) := l (2 / (2 - Q.3)) in
  (yr^2 + 4 * xr - 4 ≤ 0) :=
begin
  sorry,
end

end range_of_R_l692_692781


namespace find_angle_B_l692_692190

noncomputable def B : ℝ :=
  sorry 

theorem find_angle_B (A B : ℝ) (a b : ℝ)
  (hA : A = 120) (ha : a = 2) (hb : b = (2 * real.sqrt 3) / 3) :
  B = 30 :=
by
  sorry

end find_angle_B_l692_692190


namespace brians_gas_usage_l692_692833

theorem brians_gas_usage (miles_per_gallon : ℕ) (miles_traveled : ℕ) (gallons_used : ℕ) 
  (h1 : miles_per_gallon = 20) 
  (h2 : miles_traveled = 60) 
  (h3 : gallons_used = miles_traveled / miles_per_gallon) : 
  gallons_used = 3 := 
by 
  rw [h1, h2] at h3 
  exact h3

end brians_gas_usage_l692_692833


namespace smallest_n_for_Tn_integer_l692_692117

def sum_reciprocals : ℚ := (∑ i in finset.range (10), if i ≠ 0 then 1 / (i : ℚ) else 0)

def T_n (n : ℕ) : ℚ :=
  sum_reciprocals * ((2^(n+1) - 1) / 9)

theorem smallest_n_for_Tn_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, m < n → T_n m ∉ ℤ :=
begin
  use 6,
  split,
  { -- Proof that T_6 is an integer
    sorry
  },
  { -- Proof that no smaller n works
    intros m hm,
    sorry
  }
end

end smallest_n_for_Tn_integer_l692_692117


namespace percentage_decrease_l692_692830

-- Define conditions and variables
def original_selling_price : ℝ := 659.9999999999994
def profit_rate1 : ℝ := 0.10
def increase_in_selling_price : ℝ := 42
def profit_rate2 : ℝ := 0.30

-- Define the actual proof problem
theorem percentage_decrease (C C_prime : ℝ) 
    (h1 : 1.10 * C = original_selling_price) 
    (h2 : 1.30 * C_prime = original_selling_price + increase_in_selling_price) : 
    ((C - C_prime) / C) * 100 = 10 := 
sorry

end percentage_decrease_l692_692830


namespace prop_true_l692_692948

variables {a b l : Type} {α β : Type}

-- Conditions:
def skew_lines (a b : Type) : Prop :=
  ¬ (∃ p : Type, (p ∈ a) ∧ (p ∈ b)) ∧ ¬ (∃ q : Type, (q ∈ a) ∧ (q ∈ b))

def lines_do_not_intersect (a b : Type) : Prop :=
  ¬ (∃ p : Type, (p ∈ a) ∧ (p ∈ b))

def perp (l : Type) (α : Type) : Prop :=
  ∃! m : Type, (m ∈ α) ∧ perp (m, l)

def parallel (l : Type) (α : Type) : Prop :=
  l ⊥ α

def parallel_planes (α β : Type) : Prop :=
  ∀ p1 p2 : Type, (p1 ∈ α) ∧ (p2 ∈ β)

-- The theorem states that Proposition ④ is the true proposition:
theorem prop_true :
  (skew_lines a b → lines_do_not_intersect a b) →
  (lines_do_not_intersect a b → skew_lines a b) →
  (∀ m : Type, (m ∈ α) → perp (l, α)) →
  (∀ p1 p2 : Type, (p1 ∈ l) ∧ (p2 ∈ l) → p1 = p2 ∧ p1 ≠ p2) →
  (α ≠ β) →
  (∃ l ⊃ α, ∃ m ⊃ α, parallel_planes α β) :=
sorry

end prop_true_l692_692948


namespace no_solutions_exist_l692_692040

def sequence (u : ℕ → ℤ) : Prop :=
  u 0 = 0 ∧ u 1 = 1 ∧ ∀ n ≥ 2, u n = 6 * u (n - 1) + 7 * u (n - 2)

theorem no_solutions_exist (u : ℕ → ℤ) (h_seq : sequence u) :
  ¬ ∃ a b c n : ℕ, ab(a + b)(a^2 + ab + b^2) = c^{2022} + 42 ∧ c^{2022} + 42 = u n :=
by
  sorry

end no_solutions_exist_l692_692040


namespace part1_part2_l692_692905
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1| - (a - 1) * |x|

theorem part1 :
  let a := 2
  in { x : ℝ | f a x > 2 } = { x : ℝ | x < -1 ∨ x > 3 } :=
by
  sorry

theorem part2 (x : ℝ) (a : ℝ) :
  (1 < x ∧ x < 2) ∧ f a x < a + 1 → a ≥ 2/5 :=
by
  sorry

end part1_part2_l692_692905


namespace problem1_problem2_l692_692280

theorem problem1 (x : ℝ) : (x + 3) * (x - 1) ≤ 0 ↔ -3 ≤ x ∧ x ≤ 1 :=
sorry

theorem problem2 (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 :=
sorry

end problem1_problem2_l692_692280


namespace absolute_value_sum_10_terms_l692_692545

def sequence_sum (n : ℕ) : ℤ := (n^2 - 4 * n + 2)

def term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n - 1)

-- Prove that the sum of the absolute values of the first 10 terms is 66.
theorem absolute_value_sum_10_terms : 
  (|term 1| + |term 2| + |term 3| + |term 4| + |term 5| + 
   |term 6| + |term 7| + |term 8| + |term 9| + |term 10| = 66) := 
by 
  -- Skip the proof
  sorry

end absolute_value_sum_10_terms_l692_692545


namespace second_largest_minus_smallest_is_two_l692_692743

theorem second_largest_minus_smallest_is_two :
  let numbers := [10, 11, 12, 13]
  secondLargest numbers - smallest numbers = 2 :=
by
  sorry

def secondLargest (l : List ℕ) : ℕ :=
  l.sort.reverse.tail.headD 0

def smallest (l : List ℕ) : ℕ :=
  l.minimum.getD 0

end second_largest_minus_smallest_is_two_l692_692743


namespace max_elements_in_set_A_l692_692536

theorem max_elements_in_set_A (A : Set ℕ) (hA : ∀ x y ∈ A, x ≠ y → |x - y| ≥ (1 / 25 : ℝ) * x * y) : ∃ n : ℕ, (∀ B : Set ℕ, ∀ hB : (∀ x y ∈ B, x ≠ y → |x - y| ≥ (1 / 25 : ℝ) * x * y), B ⊆ A → B.finite → B.card ≤ n) ∧ n = 9 :=
by sorry

end max_elements_in_set_A_l692_692536


namespace minimum_positive_period_correct_minimum_value_correct_intervals_monotonic_increasing_correct_l692_692498

noncomputable def y : ℝ → ℝ := λ x, (sin x) ^ 4 + 2 * (sqrt 3) * (sin x) * (cos x) - (cos x) ^ 4

def minimum_positive_period : ℝ := π

def minimum_value : ℝ := -2

def intervals_monotonic_increasing : set (set ℝ) := {Icc 0 (π / 3), Icc (5 * π / 6) π}

theorem minimum_positive_period_correct :
  (∃ T > 0, ∀ x, y (x + T) = y x) ∧ (∀ T > 0, (∀ x, y (x + T) = y x) → T ≥ π) :=
by
  sorry

theorem minimum_value_correct :
  ∃ x, y x = -2 ∧ ∀ z, y z ≥ -2 :=
by
  sorry

theorem intervals_monotonic_increasing_correct :
  ∀ x, (x ∈ Icc 0 π) → ((∃ a b, (Icc a b) ∈ intervals_monotonic_increasing ∧ x ∈ Icc a b ∧ (∀ u v, (u ∈ Icc a b ∧ v ∈ Icc a b ∧ u < v) → y u < y v)) ∨
    (∃ a b, (Icc a b) ∈ intervals_monotonic_increasing ∧ x ∈ Icc a b ∧ (∀ u v, (u ∈ Icc a b ∧ v ∈ Icc a b ∧ u < v) → y u > y v))) :=
by
  sorry

end minimum_positive_period_correct_minimum_value_correct_intervals_monotonic_increasing_correct_l692_692498


namespace train_length_calculation_l692_692404

theorem train_length_calculation (len1 : ℝ) (speed1_kmph : ℝ) (speed2_kmph : ℝ) (crossing_time : ℝ) (len2 : ℝ) :
  len1 = 120.00001 → 
  speed1_kmph = 120 → 
  speed2_kmph = 80 → 
  crossing_time = 9 → 
  (len1 + len2) = ((speed1_kmph * 1000 / 3600 + speed2_kmph * 1000 / 3600) * crossing_time) → 
  len2 = 379.99949 :=
by
  intros hlen1 hspeed1 hspeed2 htime hdistance
  sorry

end train_length_calculation_l692_692404


namespace expected_length_first_group_l692_692711

noncomputable def indicator_prob (n : ℕ) : ℚ :=
if n = 1 then 1/50 else 1/20

theorem expected_length_first_group (ones zeros : ℕ) (h : ones = 19) (h2 : zeros = 49) : 
  let X := ∑ i in (finset.range ones ∪ finset.range zeros), (indicator_prob (i + 1)) in
  (X : ℝ) = 2.83 :=
sorry

end expected_length_first_group_l692_692711


namespace cost_of_600_pages_l692_692609

def cost_per_5_pages := 10 -- 10 cents for 5 pages
def pages_to_copy := 600
def expected_cost := 12 * 100 -- 12 dollars in cents

theorem cost_of_600_pages : pages_to_copy * (cost_per_5_pages / 5) = expected_cost := by
  sorry

end cost_of_600_pages_l692_692609


namespace last_three_digits_of_7_to_50_l692_692497

theorem last_three_digits_of_7_to_50 : (7^50) % 1000 = 991 := 
by 
  sorry

end last_three_digits_of_7_to_50_l692_692497


namespace arrange_2010_rays_l692_692608

def ray (P Q : Point) : Prop := sorry -- Definition of a ray passing through two points P and Q

variables {Point : Type} [fintype Point]

-- Representation of the given conditions
def plane_configuration (rays : fin 2010 → set (set Point)) : Prop :=
  ∀ (r : fin 2010), 
  -- No more than two rays pass through any point
  (∀ p : Point, (finset.filter (λ s, p ∈ s) (finset.univ)).card ≤ 2) ∧ 
  -- Each ray intersects exactly two others
  (∀ (r1 r2 : fin 2010), r ≠ r1 → nonempty (rays r ∩ rays r1) ∧ r ≠ r2 → r1 ≠ r2 → nonempty (rays r ∩ rays r2)) ∧
  -- Any two points on any two rays can be connected by a broken line entirely contained in the union of these rays
  (∀ (p1 p2 : Point) (r1 r2 : fin 2010), ∃ (s : finset (set Point)), p1 ∈ rays r1 ∧ p2 ∈ rays r2 ∧ (∀ r ∈ s, r ∈ rays) ∧ connected_within_union p1 p2 s)

theorem arrange_2010_rays : ∃ (rays : fin 2010 → set (set Point)), plane_configuration rays :=
sorry

end arrange_2010_rays_l692_692608


namespace max_real_roots_l692_692112

noncomputable def P (n : ℕ) : ℝ → ℝ :=
  λ x => ∑ i in finset.range (n + 1), (-1 : ℝ)^i * x^(n - i)

theorem max_real_roots (n : ℕ) (hn : 0 < n) :
  (∀ x : ℝ, P n x = 0 → x = -1 ∧ n % 2 = 1) ∨ (∀ x : ℝ, P n x = 0 → (x = -1 ∨ x = 1) ∧ n % 2 = 0) :=
by
  sorry

end max_real_roots_l692_692112


namespace proof_problem_l692_692631

variable (n : ℕ) (x : Fin n → ℝ)

theorem proof_problem
  (h₁ : ∀ i, x i > 0)
  (h₂ : (∑ i, 1 / (x i + 1998)) = 1 / 1998) :
  (Real.sqrt (∏ i, x i) / (n - 1)) ≥ 1998 := sorry

end proof_problem_l692_692631


namespace total_flowers_eaten_l692_692257

theorem total_flowers_eaten :
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = 16.5 :=
by
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  sorry

end total_flowers_eaten_l692_692257


namespace cube_light_path_l692_692225

-- Define the problem and its conditions
def cube_side_length := 10
def point_A := (0, 0, 0)
def point_P := (3, 1, 10)

-- Given conditions: distances from P to lines EH and EF
def dist_P_to_EH := 3
def dist_P_to_EF := 1

-- Correct answer:
def path_length := 10 * Real.sqrt 110
def result_m := 10
def result_n := 110
def result_m_plus_n := result_m + result_n

-- State the theorem to be proved
theorem cube_light_path : result_m + result_n = 120 := by
  sorry

end cube_light_path_l692_692225


namespace number_of_groups_of_three_books_l692_692940

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l692_692940


namespace largest_prime_factor_1001_l692_692340

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692340


namespace f_17_l692_692388

def f : ℕ → ℤ := sorry

axiom f_prop1 : f 1 = 0
axiom f_prop2 : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) = f m + f n + 4 * (9 * m * n - 1)

theorem f_17 : f 17 = 1052 := by
  sorry

end f_17_l692_692388


namespace combination_15_choose_3_l692_692944

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l692_692944


namespace slope_of_line_l692_692482

theorem slope_of_line (x y : ℝ) (h : x / 4 + y / 3 = 1) : let m := - (3 / 4) in m = -3 / 4 :=
by 
  sorry

end slope_of_line_l692_692482


namespace value_of_a3_a6_a9_l692_692246

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The common difference is 2
axiom common_difference : d = 2

-- Condition: a_1 + a_4 + a_7 = -50
axiom sum_a1_a4_a7 : a 1 + a 4 + a 7 = -50

-- The goal: a_3 + a_6 + a_9 = -38
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = -38 := 
by 
  sorry

end value_of_a3_a6_a9_l692_692246


namespace biased_coin_probability_l692_692762

theorem biased_coin_probability (h : ℚ) (H : 21 * (1 - h) = 35 * h) :
  let p : ℚ := 35 * (3/8)^4 * (5/8)^3 in
  let num_denom_sum := p.num + p.denom in
  num_denom_sum = 2451527 :=
by
  sorry

end biased_coin_probability_l692_692762


namespace frequency_of_jumps_in_range_90_to_110_l692_692052

def jumps : List ℕ := [50, 63, 77, 83, 87, 88, 89, 91, 93, 100, 102, 111, 117, 121, 130, 133, 146, 158, 177, 188]

def frequency_in_range (data : List ℕ) (low high : ℕ) : ℕ :=
  data.count (λ x => low ≤ x ∧ x ≤ high)

def total_students : ℕ := 20

theorem frequency_of_jumps_in_range_90_to_110 : frequency_in_range jumps 90 110 / total_students = 0.20 := by
  sorry

end frequency_of_jumps_in_range_90_to_110_l692_692052


namespace min_value_of_f_l692_692695

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x ^ 2

theorem min_value_of_f : ∀ x > 0, f x ≥ 9 ∧ (f x = 9 ↔ x = 2) :=
by
  sorry

end min_value_of_f_l692_692695


namespace rational_sqrts_l692_692662

def is_rational (n : ℝ) : Prop := ∃ (q : ℚ), n = q

theorem rational_sqrts 
  (x y z : ℝ) 
  (hxr : is_rational x) 
  (hyr : is_rational y) 
  (hzr : is_rational z)
  (hw : is_rational (Real.sqrt x + Real.sqrt y + Real.sqrt z)) :
  is_rational (Real.sqrt x) ∧ is_rational (Real.sqrt y) ∧ is_rational (Real.sqrt z) :=
sorry

end rational_sqrts_l692_692662


namespace part1_m_n_part2_k_l692_692553

-- Definitions of vectors a, b, and c
def veca : ℝ × ℝ := (3, 2)
def vecb : ℝ × ℝ := (-1, 2)
def vecc : ℝ × ℝ := (4, 1)

-- Part (1)
theorem part1_m_n : 
  ∃ (m n : ℝ), (-m + 4 * n = 3) ∧ (2 * m + n = 2) :=
sorry

-- Part (2)
theorem part2_k : 
  ∃ (k : ℝ), (3 + 4 * k) * 2 - (-5) * (2 + k) = 0 :=
sorry

end part1_m_n_part2_k_l692_692553


namespace parallelogram_properties_l692_692150

-- Define the points A, B, and C
def A := (2: ℝ, 1: ℝ)
def B := (3: ℝ, 2: ℝ)
def C := (6: ℝ, 3: ℝ)

-- Define the line equation function
def line_equation (p1 p2 : ℝ × ℝ) : String :=
  let k := (p2.2 - p1.2) / (p2.1 - p1.1)
  let c := p1.2 - k * p1.1
  "y - " ++ toString p1.2 ++ " = " ++ toString k ++ "(x - " ++ toString p1.1 ++ ")"
  
-- Define the distance function from a point to a line
def distance_from_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / real.sqrt (a^2 + b^2)

-- Define the magnitude of a vector
def magnitude (p1 p2: ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The Lean statement for the proof
theorem parallelogram_properties :
  (line_equation A B = "y - 1 = 1(x - 2)") ∧
  (let AB := magnitude A B
   let d := distance_from_point_to_line C 1 (-1) (-1)
   (AB * d = 2)) :=
by
  sorry

end parallelogram_properties_l692_692150


namespace marbles_given_l692_692034

theorem marbles_given (initial remaining given : ℕ) (h_initial : initial = 143) (h_remaining : remaining = 70) :
    given = initial - remaining → given = 73 :=
by
  intros
  sorry

end marbles_given_l692_692034


namespace abby_bridget_chris_probability_l692_692278

noncomputable def seatingProbability : ℚ :=
  let totalArrangements := 720
  let favorableArrangements := 114
  favorableArrangements / totalArrangements

theorem abby_bridget_chris_probability :
  seatingProbability = 19 / 120 :=
by
  simp [seatingProbability]
  sorry

end abby_bridget_chris_probability_l692_692278


namespace dihedral_angle_probability_l692_692982

-- Definition of the problem conditions
def triangular_pyramid (A B C D : ℝ) := ∀ (s : ℝ), s > 0 → 
  let tri_ABC := ∀ (x y z : ℝ), x + y + z = s in 
  triangle_ABC A B C

def equilateral_triangle (A B C : ℝ) := ∀ (s : ℝ), s > 0 → 
  let eq_tri := ∀ (x y z : ℝ), x = y := z := s in 
  triangle_ABC A B C = eq_tri

def plane_angle_greater (E C D A : ℝ) (theta : ℝ) := 
  (theta > (π / 4))

def valid_labels (f : ℝ → ℕ) := ∀ (eta : ℝ), 
  (f eta) ∈ {1, 2, 3, 4, 5, 6, 7, 8}

def point_on_line (E A B : ℝ) (f : ℝ → ℕ) := 
  (|B - E| / |E - A| = f(B) / f(A))

-- Probability Calculation
def probability_dihedral_angle := 
  (let num_combinations := 56 in 
   let valid_combinations := 9 in 
   valid_combinations / num_combinations)

theorem dihedral_angle_probability (A B C D E : ℝ) (f : ℝ → ℕ) (theta : ℝ) :
  triangular_pyramid A B C D → 
  equilateral_triangle A C D → 
  equilateral_triangle B C D → 
  plane_angle_greater E C D A theta →
  valid_labels f →
  point_on_line E A B f →
  probability_dihedral_angle = (9/56) := 
sorry

end dihedral_angle_probability_l692_692982


namespace coffee_shop_sales_l692_692824

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l692_692824


namespace roots_polynomial_sum_products_l692_692627

theorem roots_polynomial_sum_products (p q r : ℂ)
  (h : 6 * p^3 - 5 * p^2 + 13 * p - 10 = 0)
  (h' : 6 * q^3 - 5 * q^2 + 13 * q - 10 = 0)
  (h'' : 6 * r^3 - 5 * r^2 + 13 * r - 10 = 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) :
  p * q + q * r + r * p = 13 / 6 := 
sorry

end roots_polynomial_sum_products_l692_692627


namespace locus_equation_dot_product_no_point_Q_l692_692533

-- 1. 
theorem locus_equation (t : ℝ) (ht_pos : t > 0) (ht_ne_one : t ≠ 1) :
  (1 - t^2) * x^2 + y^2 + 4 * x + 3 = 0 :=
sorry

-- 2.
theorem dot_product (Q P₁ P₂ : ℝ × ℝ) (t : ℝ) (ht : t = real.sqrt 3)
  (QP1 : Q.1 = P₁.1 + 1) (QP2 : Q.2 = P₂.2 + 1) :
  vector_dot (Q - P₁) (Q - P₂) = 5 / 9 :=
sorry

-- 3.
theorem no_point_Q (t : ℝ) (θ : ℝ) (h0 : 0 < θ) (hπ : θ < π) :
  0 < t ∧ t < real.sqrt ((1 - real.cos θ) / 2) :=
sorry

end locus_equation_dot_product_no_point_Q_l692_692533


namespace four_genuine_coin_probability_l692_692484

noncomputable def probability_all_genuine_given_equal_weight : ℚ :=
  let total_coins := 20
  let genuine_coins := 12
  let counterfeit_coins := 8

  -- Calculate the probability of selecting two genuine coins from total coins
  let prob_first_pair_genuine := (genuine_coins / total_coins) * 
                                    ((genuine_coins - 1) / (total_coins - 1))

  -- Updating remaining counts after selecting the first pair
  let remaining_genuine_coins := genuine_coins - 2
  let remaining_total_coins := total_coins - 2

  -- Calculate the probability of selecting another two genuine coins
  let prob_second_pair_genuine := (remaining_genuine_coins / remaining_total_coins) * 
                                    ((remaining_genuine_coins - 1) / (remaining_total_coins - 1))

  -- Probability of A ∩ B
  let prob_A_inter_B := prob_first_pair_genuine * prob_second_pair_genuine

  -- Assuming prob_B represents the weighted probabilities including complexities
  let prob_B := (110 / 1077) -- This is an estimated combined probability for the purpose of this definition

  -- Conditional probability P(A | B)
  prob_A_inter_B / prob_B

theorem four_genuine_coin_probability :
  probability_all_genuine_given_equal_weight = 110 / 1077 := sorry

end four_genuine_coin_probability_l692_692484


namespace probability_no_3x3_red_square_l692_692078

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l692_692078


namespace only_natural_number_dividing_power_diff_l692_692492

theorem only_natural_number_dividing_power_diff (n : ℕ) (h : n ∣ (2^n - 1)) : n = 1 :=
by
  sorry

end only_natural_number_dividing_power_diff_l692_692492


namespace quadruple_pieces_sold_l692_692646

theorem quadruple_pieces_sold (split_earnings : (2 : ℝ) * 5 = 10) 
  (single_pieces_sold : 100 * (0.01 : ℝ) = 1) 
  (double_pieces_sold : 45 * (0.02 : ℝ) = 0.9) 
  (triple_pieces_sold : 50 * (0.03 : ℝ) = 1.5) : 
  let total_earnings := 10
  let earnings_from_others := 3.4
  let quadruple_piece_price := 0.04
  total_earnings - earnings_from_others = 6.6 → 
  6.6 / quadruple_piece_price = 165 :=
by 
  intros 
  sorry

end quadruple_pieces_sold_l692_692646


namespace polynomial_value_at_2_l692_692751

def f (x : ℕ) : ℕ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem polynomial_value_at_2 : f 2 = 1397 := by
  sorry

end polynomial_value_at_2_l692_692751


namespace chess_tournament_results_l692_692964

-- Introduction of Players
inductive Player
| first
| second
| third
| fourth
| fifth
deriving DecidableEq

open Player

-- Definitions given as conditions
def played_once (games : List (Player × Player)) : Prop :=
  ∀ p1 p2 : Player, p1 ≠ p2 → (p1, p2) ∈ games ∨ (p2, p1) ∈ games

def no_draws (scores : Player → ℝ) (games : List (Player × Player)) : Prop :=
  (scores first - 1) ∈ (λ p, scores first + 1 - scores p) ∘ Pair.snd <$> games

def no_losses (scores : Player → ℝ) (games : List (Player × Player)) : Prop :=
  (scores second) ∈ scores second :: (λ p, scores p + 1 - scores second) ∘ Pair.snd <$> games 

def no_wins (scores : Player → ℝ) (games : List (Player × Player)) : Prop :=
  (scores fourth) ∈ scores fourth :: (λ p, scores fourth - 1 - scores p) ∘ Pair.snd <$> games 

noncomputable def distinct_scores (scores : Player → ℝ) : Prop :=
  list.pairwise (λ a b, a ≠ b) [scores first, scores second, scores third, scores fourth, scores fifth]

-- Points conditions
def total_points (scores : Player → ℝ) : Prop :=
  scores first + scores second + scores third + scores fourth + scores fifth = 10

def specific_points (scores : Player → ℝ) : Prop :=
  scores first = 3 ∧ scores second = 2.5 ∧ scores third = 2 ∧ scores fourth = 1.5 ∧ scores fifth = 1

theorem chess_tournament_results :
  ∃ scores : Player → ℝ, played_once && no_draws && no_losses && no_wins && distinct_scores && total_points scores && specific_points scores := sorry

end chess_tournament_results_l692_692964


namespace simplify_expression_l692_692665

theorem simplify_expression :
  sqrt (sqrt (sqrt (1 / 15625))) = 1 / (5 ^ (3 / 8)) :=
sorry

end simplify_expression_l692_692665


namespace max_distance_from_circle_to_line_l692_692634

open Real

def circle (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0
def line (x y : ℝ) : Prop := x - sqrt 3 * y + 3 * sqrt 3 = 0

theorem max_distance_from_circle_to_line :
  ∀ M : ℝ × ℝ, circle M.1 M.2 →
    ∃ d : ℝ, d = (1 + sqrt 3 / 2) :=
sorry

end max_distance_from_circle_to_line_l692_692634


namespace largest_prime_factor_1001_l692_692338

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692338


namespace prob_angie_opposite_carlos_l692_692668

section HexagonalTable

-- Let's define the set of friends.
inductive Friend
| Angie | Bridget | Carlos | Diego | Elena | Frank

open Friend

-- Define a function that counts the number of ways to arrange friends around a table
def arrangements_around_table (n : Nat) : Nat :=
  if h : n > 0 then (n - 1)! else 0

-- Condition: Assume there are six friends around a hexagonal table.
def six_friends : List Friend := [Angie, Bridget, Carlos, Diego, Elena, Frank]

-- Function to check if two friends are opposite in a given arrangement
def are_opposite (arrangement : List Friend) (a b : Friend) : Prop :=
  match arrangement.indexOf? a, arrangement.indexOf? b with
  | some i, some j => (i + (arrangement.length / 2)) % arrangement.length = j
  | _, _ => false

-- Theorem stating the probability calculation
theorem prob_angie_opposite_carlos :
  (24 : ℚ) / (120 : ℚ) = 1 / 5 :=
by
  -- The proof can be formalized here
  sorry

end HexagonalTable

end prob_angie_opposite_carlos_l692_692668


namespace ella_model_dome_height_l692_692091

noncomputable def height_of_model_dome (observatory_height : ℝ) (observatory_volume : ℝ)
    (model_volume : ℝ) : ℝ :=
  observatory_height / (observatory_volume / model_volume)^(1/3)

theorem ella_model_dome_height :
  height_of_model_dome 70 500000 0.05 ≈ 0.325 := sorry

end ella_model_dome_height_l692_692091


namespace min_value_expression_l692_692133

theorem min_value_expression (x y z : ℝ) (h : x - 2 * y + 2 * z = 5) : (x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2 ≥ 36 :=
by
  sorry

end min_value_expression_l692_692133


namespace sum_of_three_numbers_l692_692308

theorem sum_of_three_numbers :
  ∃ (S1 S2 S3 : ℕ), 
    S2 = 72 ∧
    S1 = 2 * S2 ∧
    S3 = S1 / 3 ∧
    S1 + S2 + S3 = 264 := 
by
  sorry

end sum_of_three_numbers_l692_692308


namespace rectangle_not_sum_110_l692_692674

noncomputable def not_sum_110 : Prop :=
  ∀ (w : ℕ), (w > 0) → (2 * w^2 + 6 * w ≠ 110)

theorem rectangle_not_sum_110 : not_sum_110 := 
  sorry

end rectangle_not_sum_110_l692_692674


namespace min_pairs_of_acquaintances_l692_692198

-- Define the condition of acquaintances within a group of 6 children
def three_non_overlapping_pairs (children : Set (Fin 225)) : Prop :=
  ∀ (s : Finset (Fin 225)), s.card = 6 → ∃ (p : List (Fin 225 × Fin 225)),
    p.length = 3 ∧ pairwise (λ (x y : Fin 225 × Fin 225), disjoint x.fst y.fst ∧ disjoint x.snd y.snd) p ∧
    ∀ (q : Fin 225 × Fin 225), q ∈ p → (q.fst, q.snd) ⊆ s ∧ (q.snd, q.fst) ⊆ s

-- Define the problem as a Lean theorem statement
theorem min_pairs_of_acquaintances (children : Finset (Fin 225)) (h : three_non_overlapping_pairs children) :
  ∃ (min_pairs : Nat), min_pairs = 24750 := 
sorry

end min_pairs_of_acquaintances_l692_692198


namespace number_of_groups_of_three_books_l692_692938

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l692_692938


namespace intersection_M_N_l692_692635

def M := { x : ℝ | |x| ≤ 1 }
def N := { x : ℝ | x^2 - x < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l692_692635


namespace circumcircle_center_lies_on_circle_l692_692729

section circumcircle_center

variables {A B C D E F O : Type*}
variables [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O]

-- Conditions
-- 1. The trapezoid ABCD (AB || CD) is inscribed in a circle ω
axiom inscribed_trapezoid (A B C D : Type*) (ω : Type*) [is_circle ω] [is_trapezoid ABCD] : 
  inscribed_in ω ABCD

-- 2. Point E such that BC = BE and E is on the ray beyond C along DC
axiom E_on_ray (B C D E : Type*) : on_ray D C E ∧ BC = BE

-- 3. The line BE intersects the circle ω again at F, which lies outside the segment BE
axiom BE_intersects_again (B E ω F : Type*) [is_circle ω] [is_line B E] : intersects_again_in_circle B E ω F ∧ outside_segment B E F

-- Assertion to be proved
theorem circumcircle_center_lies_on_circle (A B C D E F O ω : Type*) [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O] [is_circle ω] : 
  center_of_circumcircle_lies_on_circle (triangle C E F) ω :=
by {
  assume inscribed_trapezoid ABCD ω,
  assume E_on_ray B C D E,
  assume BE_intersects_again B E ω F,
  exact sorry
}

end circumcircle_center

end circumcircle_center_lies_on_circle_l692_692729


namespace largest_prime_factor_1001_l692_692343

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692343


namespace spots_doghouse_area_l692_692671

/-- Spot's Doghouse Problem theorem --/
theorem spots_doghouse_area:
  ∀ (side_length rope_length : ℝ),
  side_length = 1 ∧ rope_length = 3 →
  let sector_area := fun (r : ℝ) (θ : ℝ) => θ / 360 * π * r^2 in
  let large_sector_area := sector_area rope_length 240 in
  let small_sector_area := sector_area 2 60 * 2 in
  large_sector_area + small_sector_area = 22 * π / 3 :=
by
  intros side_length rope_length h,
  simp [*, function.comp],
  sorry

end spots_doghouse_area_l692_692671


namespace rational_elements_of_S_l692_692222

theorem rational_elements_of_S (S : set ℝ) (hS_sub : S ⊆ set.Icc 0 1)
  (hS_fin : set.finite S)
  (x0_in_S : (0 : ℝ) ∈ S)
  (x1_in_S : (1 : ℝ) ∈ S)
  (h_distances : ∀ d ∈ {d | ∃ x y ∈ S, d = abs (x - y) ∧ d ≠ 1}, ∃ x₁ x₂ y₁ y₂ ∈ S, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ d = abs (x₁ - x₂) ∧ d = abs (y₁ - y₂)) :
  ∀ x ∈ S, ∃ q : ℚ, x = ↑q :=
by
  sorry

end rational_elements_of_S_l692_692222


namespace isosceles_triangle_has_min_hypotenuse_l692_692383

theorem isosceles_triangle_has_min_hypotenuse 
  (a b c : ℝ) (α β γ : ℝ) (k : ℝ)
  (h_perimeter : a + b + c = k)
  (h_law_of_sines : a / sin α = b / sin β = c / sin γ)
  (h_angle_sum : α + β + γ = real.pi)
  (h_equal_perimeter : ∀ a' b' c', a' + b' + c' = k → True)   -- We assume there are other triangles with the same perimeter
  : (∀ a b c, a = b → a + b + c = k → c ≤ c ↔ α = β) :=  -- Given that isosceles triangle has minimized hypotenuse among those with equal perimeter and the same angle γ
begin
  sorry
end

end isosceles_triangle_has_min_hypotenuse_l692_692383


namespace mittens_per_box_l692_692839

theorem mittens_per_box (total_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) 
  (h_total_boxes : total_boxes = 4) 
  (h_scarves_per_box : scarves_per_box = 2) 
  (h_total_clothing : total_clothing = 32) : 
  (total_clothing - total_boxes * scarves_per_box) / total_boxes = 6 := 
by
  -- Sorry, proof is omitted
  sorry

end mittens_per_box_l692_692839


namespace sum_of_squares_of_roots_l692_692471

theorem sum_of_squares_of_roots 
  (a b c : ℝ) (h : a ≠ 0)
  (h_eq : 5 * a * a + 15 * b - 25 * c = 0) :
  let x1 := (-b + sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - sqrt (b^2 - 4*a*c)) / (2*a)
  (x1^2 + x2^2 = 19) :=
by
  have hsum : x1 + x2 = -b / a := sorry,
  have hprod : x1 * x2 = c / a := sorry,
  have hsqrt : x1^2 + x2^2 = (x1 + x2)^2 - 2 * (x1 * x2) := sorry,
  rw [hsum, hprod] at hsqrt,
  assumption -- or sorry if assumption does not hold correctly

end sum_of_squares_of_roots_l692_692471


namespace center_circumcircle_CEF_on_circle_omega_l692_692738

open Set
open Finset
open Int

/-- Formalize the given geometric problem in Lean -/
theorem center_circumcircle_CEF_on_circle_omega
  (A B C D E F : Type)
  [IncidenceGeometry A B C D E F]  -- assuming an incidence geometry instance
  (trapezoid_ABCD : trapezoid A B C D)
  (inscribed_circle_omega : inscribed_circle (trapezoid A B C D))
  (ray_DC : ray D C beyond C)
  (point_E : point_on_ray E ray_DC)
  (BC_BE_eq : length B C = length B E)
  (circ_intersection_point_F : circle_intersection_point B E omega F outside_BE)
  : lies_on_circle (circumcenter (triangle C E F)) omega := 
sorry

end center_circumcircle_CEF_on_circle_omega_l692_692738


namespace probability_no_3x3_red_square_l692_692082

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l692_692082


namespace triangle_equilateral_l692_692983

variables {A B C : ℝ} -- angles of the triangle
variables {a b c : ℝ} -- sides opposite to the angles

-- Given conditions
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C = c * Real.cos A ∧ (b * b = a * c)

-- The proof goal
theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c → a = b ∧ b = c :=
sorry

end triangle_equilateral_l692_692983


namespace jake_weight_loss_l692_692180

variable {J K L : Nat}

theorem jake_weight_loss
  (h1 : J + K = 290)
  (h2 : J = 196)
  (h3 : J - L = 2 * K) : L = 8 :=
by
  sorry

end jake_weight_loss_l692_692180


namespace probability_of_passing_test_l692_692191

noncomputable def probability_of_successful_shots (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
nat.choose n k * (p^k) * ((1-p)^(n-k))

theorem probability_of_passing_test :
  let p := 0.6 in
  probability_of_successful_shots 3 p 2 + probability_of_successful_shots 3 p 3 = 0.648 :=
by
  sorry

end probability_of_passing_test_l692_692191


namespace probability_no_3x3_red_square_l692_692067

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l692_692067


namespace graph_single_point_l692_692286

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end graph_single_point_l692_692286


namespace probability_1_lt_a_lt_2_in_0_3_l692_692947

open Set

noncomputable def probability_interval_subset {a : ℝ} (interval full_interval : Set ℝ) : ℚ :=
  (measure_theory.measure.count (interval ∩ full_interval)) / (measure_theory.measure.count full_interval)

theorem probability_1_lt_a_lt_2_in_0_3 :
  probability_interval_subset {a | 1 < a ∧ a < 2} {a | 0 ≤ a ∧ a ≤ 3} = 1 / 3 :=
  sorry

end probability_1_lt_a_lt_2_in_0_3_l692_692947


namespace alec_extra_games_l692_692002

theorem alec_extra_games (total_games won_games : ℕ) (H1 : total_games = 200) (H2 : won_games = 98) : 
  ∃ x : ℕ, (won_games + x) / (total_games + x) = 0.50 ∧ x = 4 :=
by
  sorry

end alec_extra_games_l692_692002


namespace a_10_eq_19_div_2_l692_692525

-- Arithmetic sequence with common difference 1
def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ :=
  a_1 + d * (n - 1)

-- Sum of the first n terms in an arithmetic sequence
def sum_arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ :=
  n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom a1 : ℕ
axiom common_difference : ℕ := 1
axiom S8_eq_4S4 : sum_arithmetic_sequence a1 common_difference 8 = 4 * sum_arithmetic_sequence a1 common_difference 4

-- Prove a₁₀ = 19 / 2
theorem a_10_eq_19_div_2 : arithmetic_sequence a1 common_difference 10 = 19 / 2 :=
by
  sorry

end a_10_eq_19_div_2_l692_692525


namespace largest_prime_factor_1001_l692_692368

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692368


namespace f_odd_and_inequality_l692_692636

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem f_odd_and_inequality (x c : ℝ) : ∀ x c, 
  f x < c^2 - 3 * c + 3 := by 
  sorry

end f_odd_and_inequality_l692_692636


namespace small_ball_rubber_bands_l692_692256

theorem small_ball_rubber_bands (S : ℕ) 
    (large_ball : ℕ := 300) 
    (initial_rubber_bands : ℕ := 5000) 
    (small_balls : ℕ := 22) 
    (large_balls : ℕ := 13) :
  (small_balls * S + large_balls * large_ball = initial_rubber_bands) → S = 50 := by
    sorry

end small_ball_rubber_bands_l692_692256


namespace no_difference_of_squares_equals_222_l692_692030

theorem no_difference_of_squares_equals_222 (a b : ℤ) : a^2 - b^2 ≠ 222 := 
  sorry

end no_difference_of_squares_equals_222_l692_692030


namespace angle_BAC_is_45_degrees_l692_692622

noncomputable def angle_BAC (A B C : Point) : ℝ := sorry

theorem angle_BAC_is_45_degrees (A B C : Point)
    (AD BE CF : Line)
    (H1 : is_median A D B C)
    (H2 : is_median B E A C)
    (H3 : is_median C F A B)
    (H4 : 5 * vector AD + 7 * vector BE + 3 * vector CF = vector_zero) : 
    angle_BAC A B C = 45 :=
sorry

end angle_BAC_is_45_degrees_l692_692622


namespace probability_no_3x3_red_square_l692_692077

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l692_692077


namespace minimum_value_of_f_range_of_t_l692_692901

def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem minimum_value_of_f : (∀ x : ℝ, x > 3 → f x ≥ 9) ∧ (∃ x : ℝ, x > 3 ∧ f x = 9) :=
by
  sorry

theorem range_of_t (t : ℝ) : (∀ x : ℝ, x > 3 → f x ≥ (t / (t + 1) + 7)) → t ≤ -2 ∨ t > -1 :=
by
  sorry

end minimum_value_of_f_range_of_t_l692_692901


namespace number_of_shapes_after_4_folds_sum_of_areas_n_folds_l692_692380

def dimensions := (20, 12) -- 20dm by 12dm paper
def S1 := 240           -- Sum of areas after first fold
def S2 := 180           -- Sum of areas after second fold

/--
Prove that the number of different shapes obtained by folding the paper 4 times is 5.
-/
theorem number_of_shapes_after_4_folds : 
  number_of_shapes 4 dimensions S1 S2 = 5 := sorry

/--
Prove that the sum of the areas for n folds is 240 * (3 - (n + 3) / (2 ^ n)).
-/
theorem sum_of_areas_n_folds (n : ℕ) :
  ∑ k in finset.range n, S k dimensions = 240 * (3 - (n + 3) / (2 ^ n)) := sorry

end number_of_shapes_after_4_folds_sum_of_areas_n_folds_l692_692380


namespace locus_A_max_OP_OQ_l692_692822

-- Problem 1: Locus of point A
theorem locus_A (x_0 y_0 r x y : ℝ) (h1 : 0 < r) (h2 : r < 1)
  (hM : (x - x_0)^2 + (y - y_0)^2 = r^2)
  (h_ellipse : x_0^2 / 4 + y_0^2 = 1)
  (h_AF1_BF2 : forall A B F1 F2 : ℝ, abs(A - F1) - abs(B - F2) = 2*r) :
  ((x + real.sqrt 3)^2 + y^2 = 4) → x > 0 :=
by sorry

-- Problem 2: Maximum value of |OP| * |OQ|
theorem max_OP_OQ (x_0 y_0 r k1 k2 const : ℝ)
  (h1 : 0 < r) (h2 : r < 1) (h3 : k1 * k2 = const)
  (h_ellipse : x_0^2 / 4 + y_0^2 = 1) :
  |OP| * |OQ| ≤ 5/2 :=
by sorry

end locus_A_max_OP_OQ_l692_692822


namespace value_of_f_5_l692_692130

variable (a b c m : ℝ)

-- Conditions: definition of f and given value of f(-5)
def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2
axiom H1 : f a b c (-5) = m

-- Question: Prove that f(5) = -m + 4
theorem value_of_f_5 : f a b c 5 = -m + 4 :=
by
  sorry

end value_of_f_5_l692_692130


namespace pairs_satisfying_equation_l692_692932

theorem pairs_satisfying_equation : 
  {p : Nat × Nat // let x := p.1; let y := p.2 in x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77}.toList.length = 2 :=
by
  sorry

end pairs_satisfying_equation_l692_692932


namespace units_digit_difference_l692_692502

theorem units_digit_difference (a b c d e : ℕ) 
  (h1 : 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9)
  (h2 : (finset.singleton a ∪ {b, c, d, e}).card = 5) :
  (100 * e + 10 * d + c - (100 * a + 10 * b + c)) % 10 = 0 :=
by 
  sorry

end units_digit_difference_l692_692502


namespace equation_has_real_roots_for_all_K_l692_692870

open Real

noncomputable def original_equation (K x : ℝ) : ℝ :=
  x - K^3 * (x - 1) * (x - 3)

theorem equation_has_real_roots_for_all_K :
  ∀ K : ℝ, ∃ x : ℝ, original_equation K x = 0 :=
sorry

end equation_has_real_roots_for_all_K_l692_692870


namespace sector_central_angle_and_chord_length_l692_692803

-- Definitions based on the conditions
def sector (O A B : Type) := ∃ R α, 2 * R + R * α = 4 ∧ 0.5 * R^2 * α = 1

-- The Lean theorem statement assuming above conditions and definitions
theorem sector_central_angle_and_chord_length (O A B : Type) (R α : ℝ) 
  (h₁ : 2 * R + R * α = 4) (h₂ : 0.5 * R^2 * α = 1) : α = 2 ∧ (∃ chord_length : ℝ, chord_length = 2 * real.sin 1) :=
by {
  sorry
}

end sector_central_angle_and_chord_length_l692_692803


namespace sum_odd_indexed_eq_l692_692544

-- Define the sequence aₙ
def a (n : ℕ) : ℤ :=
  if n = 0 then 0 else if n = 1 then -1 else 
  (-a (n - 1) - 4 * (n - 1) - 2)

-- Define the sum of the first n odd-indexed terms of the sequence aₙ
def sum_odd_indexed (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a (2 * i + 1)

-- Theorem statement
theorem sum_odd_indexed_eq (n : ℕ) : sum_odd_indexed n = -2 * n * n + n :=
by
  sorry

end sum_odd_indexed_eq_l692_692544


namespace projection_composition_l692_692227

def projection_matrix (u : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let ⟨u1, u2⟩ := u
  let v := u1 ^ 2 + u2 ^ 2
  Matrix.ofBlocks
    ![(u1 * u1 / v), (u1 * u2 / v)]
    ![(u2 * u1 / v), (u2 * u2 / v)]

theorem projection_composition :
  projection_matrix (2, 2) ⬝ projection_matrix (4, 2) =
  ![![3/5, 3/10], ![3/5, 3/10]] :=
by
  sorry

end projection_composition_l692_692227


namespace cover_tiles_count_l692_692805

-- Definitions corresponding to the conditions
def tile_side : ℕ := 6 -- in inches
def tile_area : ℕ := tile_side * tile_side -- area of one tile in square inches

def region_length : ℕ := 3 * 12 -- 3 feet in inches
def region_width : ℕ := 6 * 12 -- 6 feet in inches
def region_area : ℕ := region_length * region_width -- area of the region in square inches

-- The statement of the proof problem
theorem cover_tiles_count : (region_area / tile_area) = 72 :=
by
   -- Proof would be filled in here
   sorry

end cover_tiles_count_l692_692805


namespace accuracy_percentage_correct_l692_692960

-- Definitions of the conditions.
def correctAnswers : ℕ := 58
def totalQuestions : ℕ := 84

-- The statement to be proved.
theorem accuracy_percentage_correct :
  (correctAnswers / totalQuestions) * 100 = 69.05 :=
sorry

end accuracy_percentage_correct_l692_692960


namespace largest_prime_factor_1001_l692_692344

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692344


namespace squares_placement_l692_692141

noncomputable def squares_no_overlap (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ A : ℝ, (∀ n : ℕ, S n = ∑ i in finset.range (n + 1), (a i)^2) ∧ (∃ k : ℕ, S k = A^2) ∧
  (∀ ε > 0, ∃ n, ∀ m ≥ n, |S m - A^2| < ε)

theorem squares_placement (a : ℕ → ℝ) (S : ℕ → ℝ) (h : squares_no_overlap a S) :
  ∃ A : ℝ, (∀ n : ℕ, S n = ∑ i in finset.range (n + 1), (a i)^2) ∧ (∃ k : ℕ, S k = A^2) ∧
  (∀ ε > 0, ∃ n, ∀ m ≥ n, |S m - A^2| < ε) →
  ∃ b : ℝ, b = 2 * A ∧ (∀ i j : ℕ, i ≠ j → ¬(a i = a j)) :=
by
  sorry

end squares_placement_l692_692141


namespace goose_eggs_laid_l692_692392

theorem goose_eggs_laid (E : ℕ)
  (hatch_rate : E / 4)
  (survival_first_month_rate : 4 / 5)
  (survival_first_year_rate : 2 / 5)
  (geese_survived_first_year : E * (2 / 5) * (4 / 5) * (1 / 4) = 120) :
  E = 1500 :=
begin
  sorry
end

end goose_eggs_laid_l692_692392


namespace systematic_sampling_8000_50_7894_l692_692126

def systematic_sampling (total: Nat) (sampling_size : Nat) (last_sample: Nat) : 
  (last_segment_range: Nat × Nat) × List Nat :=
  let segment_size :=
    total / sampling_size
  let last_segment_start := 
    total - segment_size + 1
  let first_5_sampled_numbers :=
    List.range 5 |>.map (λ n => last_sample - last_segment_start + n * segment_size)
   ((last_segment_start, total), first_5_sampled_numbers)

theorem systematic_sampling_8000_50_7894 : 
  systematic_sampling 8000 50 7894 = ((7840, 7999), [54, 214, 374, 534, 694]) := 
sorry

end systematic_sampling_8000_50_7894_l692_692126


namespace no_two_primes_sum_to_51_l692_692202

theorem no_two_primes_sum_to_51 :
  ¬ ∃ p1 p2 : ℕ, p1.prime ∧ p2.prime ∧ p1 + p2 = 51 :=
sorry

end no_two_primes_sum_to_51_l692_692202


namespace balance_difference_is_3259_l692_692447

noncomputable def alice_initial_deposit : ℕ := 10000
noncomputable def alice_interest_rate : ℚ := 6 / 100
noncomputable def bob_initial_deposit : ℕ := 10000
noncomputable def bob_interest_rate : ℚ := 4 / 100
noncomputable def years : ℕ := 10

noncomputable def compounded_value (principal : ℚ) (rate : ℚ) (n : ℕ) (t : ℕ) :=
  principal * (1 + rate / n) ^ (n * t)

noncomputable def alice_final_balance : ℚ :=
  compounded_value alice_initial_deposit (alice_interest_rate / 2) 2 years

noncomputable def bob_final_balance : ℚ :=
  compounded_value bob_initial_deposit bob_interest_rate 1 years

noncomputable def balance_difference : ℚ :=
  alice_final_balance - bob_final_balance

theorem balance_difference_is_3259 :
  balance_difference ≈ 3259 :=
by
  sorry

end balance_difference_is_3259_l692_692447


namespace alex_additional_coins_l692_692003

theorem alex_additional_coins (friends : ℕ) (coins : ℕ) (needed_coins : ℕ) :
  friends = 15 ∧ coins = 85 ∧ needed_coins = (∑ i in finset.range 16, i) - coins → needed_coins = 35 :=
begin
  intros h,
  obtain ⟨hf, hc, hn⟩ := h,
  have hn_sum : (∑ i in finset.range 16, i) = 120,
  { simp [finset.sum_range_succ], norm_num },
  rw [hf, hc, hn_sum] at hn,
  exact hn,
end

end alex_additional_coins_l692_692003


namespace average_height_of_five_people_l692_692611

noncomputable def cm_to_inch : ℝ := 2.54

theorem average_height_of_five_people :
  let itzayana_height := (zora_height : ℝ) + 4
  let zora_height := (brixton_height : ℝ) - 8
  let brixton_height := 64
  let zara_height := brixton_height
  let jaxon_height := 170 / cm_to_inch
  let heights := [itzayana_height, zora_height, brixton_height, zara_height, jaxon_height]
  (∑ h in heights, h) / (heights.length) = 62.2 :=
by
  sorry

end average_height_of_five_people_l692_692611


namespace probability_four_or_more_same_value_dice_l692_692120

theorem probability_four_or_more_same_value_dice :
  let p := (1 / 1296 : ℚ) + (25 / 1296 : ℚ)
  in p = (1 / 54 : ℚ) :=
by
  let p := (1 / 1296 : ℚ) + (25 / 1296 : ℚ)
  have h : p = (1 / 54 : ℚ) := sorry
  exact h

end probability_four_or_more_same_value_dice_l692_692120


namespace orbit_time_l692_692282

-- Define the given conditions
def radius : ℝ := 3500
def speed : ℝ := 550

-- The formula for circumference
def circumference : ℝ := 2 * Real.pi * radius

-- The formula to calculate the time required for one complete orbit
def time_required : ℝ := circumference / speed

-- The goal is to prove the calculated time is approximately 40.04 hours
theorem orbit_time (h_circumference : circumference = 2 * Real.pi * radius)
                   (h_speed : speed = 550) :
  time_required ≈ 40.04 :=
by
  -- Using the given conditions
  have h_radius : radius = 3500 := rfl
  sorry -- Proof to be completed

end orbit_time_l692_692282


namespace John_works_5_days_a_week_l692_692995

theorem John_works_5_days_a_week
  (widgets_per_hour : ℕ)
  (hours_per_day : ℕ)
  (widgets_per_week : ℕ)
  (H1 : widgets_per_hour = 20)
  (H2 : hours_per_day = 8)
  (H3 : widgets_per_week = 800) :
  widgets_per_week / (widgets_per_hour * hours_per_day) = 5 :=
by
  sorry

end John_works_5_days_a_week_l692_692995


namespace comics_in_box_l692_692018

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l692_692018


namespace add_fractions_l692_692489

theorem add_fractions : (1 / 4 : ℚ) + (3 / 5) = 17 / 20 := 
by
  sorry

end add_fractions_l692_692489


namespace find_f_of_7_l692_692527

-- Defining the conditions in the problem.
variables (f : ℝ → ℝ)
variables (odd_f : ∀ x : ℝ, f (-x) = -f x)
variables (periodic_f : ∀ x : ℝ, f (x + 4) = f x)
variables (f_eqn : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = x + 2)

-- The statement of the problem, to prove f(7) = -3.
theorem find_f_of_7 : f 7 = -3 :=
by
  sorry

end find_f_of_7_l692_692527


namespace ratio_x_y_l692_692051

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 7) : x / y = 29 / 64 :=
by
  sorry

end ratio_x_y_l692_692051


namespace angle_bisector_perpendicular_l692_692976

-- Given conditions and definitions
variables {A B C D : Type} [InnerProductSpace ℝ D]
variable [ConvexQuadrilateral A B C D]
variable (angle_eq : ∠B = ∠C)
variable (right_angle_D : ∠D = 90)
variable (length_relation : |AB| = 2 * |CD|)

-- Statement to prove: The angle bisector of ∠ACB is perpendicular to CD.
theorem angle_bisector_perpendicular (h : ConvexQuadrilateral A B C D) :
  Bisector (∠ A C B) ⊥ Segment C D :=
  sorry

end angle_bisector_perpendicular_l692_692976


namespace geom_sequence_solution_l692_692896

noncomputable def a_n (n : ℕ) : ℝ :=
  3 * 2 ^ (n - 1)

noncomputable def b_n (n : ℕ) : ℝ :=
  a_n n ^ 2

noncomputable def T_n (n : ℕ) : ℝ :=
  (1 - 2 ^ n) ^ 2

theorem geom_sequence_solution
    (S_n : ℕ → ℝ)
    (a : ℝ) (b : ℝ)
    (h_sum : ∀ n, S_n n = a * 2 ^ n + b)
    (h_a1 : a_n 1 = 3) :
  a = 3 ∧ b = -3 ∧ 
  (∀ n, a_n n = 3 * 2 ^ (n - 1)) ∧ 
  (∀ n, ∑ i in Finset.range n, b_n i = T_n n) :=
by {
  sorry
}

end geom_sequence_solution_l692_692896


namespace number_of_ways_to_buy_fruits_l692_692050

-- Define the conditions for selling the fruits
def fruits_sold_in_packs : Prop := (∀ n : ℕ, n > 0 → ∃ (a b o p : ℕ), 2 * a + 5 * b + o + p = n ∧ o ≤ 4 ∧ p ≤ 1)

-- Prove the number of ways to buy n fruits.
theorem number_of_ways_to_buy_fruits (n : ℕ) (h : fruits_sold_in_packs) : ℕ :=
  n + 1

-- Sorry for skipping the proof
sorry

end number_of_ways_to_buy_fruits_l692_692050


namespace expected_length_first_group_l692_692714

noncomputable def indicator_prob (n : ℕ) : ℚ :=
if n = 1 then 1/50 else 1/20

theorem expected_length_first_group (ones zeros : ℕ) (h : ones = 19) (h2 : zeros = 49) : 
  let X := ∑ i in (finset.range ones ∪ finset.range zeros), (indicator_prob (i + 1)) in
  (X : ℝ) = 2.83 :=
sorry

end expected_length_first_group_l692_692714


namespace sin_double_angle_l692_692874

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l692_692874


namespace cos_seven_theta_l692_692567

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (7 * θ) = -83728 / 390625 := 
sorry

end cos_seven_theta_l692_692567


namespace people_owning_all_three_pets_l692_692771

theorem people_owning_all_three_pets 
  (total_people : ℕ) (cats_owners : ℕ) (dogs_owners : ℕ) (rabbits_owners : ℕ) 
  (exactly_two_pets_owners : ℕ) (total_pets_owners : ℕ) : 
  total_people = 60 → 
  cats_owners = 30 → 
  dogs_owners = 40 → 
  rabbits_owners = 16 → 
  exactly_two_pets_owners = 12 → 
  total_pets_owners = total_people → 
  ∃ x : ℕ, x = 14 ∧ total_people = cats_owners + dogs_owners + rabbits_owners - exactly_two_pets_owners - x :=
begin
  intros,
  simp,
  use 14,
  sorry
end

end people_owning_all_three_pets_l692_692771


namespace tubs_from_usual_vendor_l692_692426

def total_tubs_needed : Nat := 100
def tubs_in_storage : Nat := 20
def fraction_from_new_vendor : Rat := 1 / 4

theorem tubs_from_usual_vendor :
  let remaining_tubs := total_tubs_needed - tubs_in_storage
  let tubs_from_new_vendor := remaining_tubs * fraction_from_new_vendor
  let tubs_from_usual_vendor := remaining_tubs - tubs_from_new_vendor
  tubs_from_usual_vendor = 60 :=
by
  intro remaining_tubs tubs_from_new_vendor
  exact sorry

end tubs_from_usual_vendor_l692_692426


namespace tiling_configuration_exists_l692_692784

theorem tiling_configuration_exists (black_square : set (ℝ × ℝ)) (tiles : fin 7 → set (ℝ × ℝ)) (size : ℝ) :
  (∀ i, tiles i = set.prod (set.Icc (i.1 : ℝ) (i.1 + size)) (set.Icc (i.2 : ℝ) (i.2 + size))) →
  (∀ i, ∃ x ∈ black_square, x ∈ tiles i) ∧ (∀ i ≠ j, tiles i ∩ tiles j = ∅) :=
sorry

end tiling_configuration_exists_l692_692784


namespace sum_h_eq_2008_l692_692025

def h (t : ℝ) : ℝ := 5 / (5 + 25^t)

theorem sum_h_eq_2008 : 
  2 * (∑ k in Finset.range 2008 | 1 ≤ k ∧ k < 2009, h (k / 2009)) = 2008 :=
by
  sorry

end sum_h_eq_2008_l692_692025


namespace minimum_value_of_c_l692_692961

theorem minimum_value_of_c 
  (a b c A B C : ℝ)
  (h1 : 2 * sin (C / 2) * (sqrt(3) * cos (C / 2) - sin (C / 2)) = 1)
  (h2 : 1 / 2 * a * b * sin (C) = 2 * sqrt 3)
  (hA : A + B + C = π) -- sum of angles in a triangle
  (hC_nonneg : 0 < C) -- C is a positive angle
  (hC_le_pi : C < π) -- C is less than pi in a triangle
  : c ≥ 2 * sqrt 2 :=
by
  sorry

end minimum_value_of_c_l692_692961


namespace eval_f_l692_692624

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem eval_f : f (f (1 / 2)) = 1 :=
by
  sorry

end eval_f_l692_692624


namespace citizen_wealth_ratio_l692_692473

theorem citizen_wealth_ratio (W P : ℝ) (hW : 0 < W) (hP : 0 < P) :
  let pX := 0.25 * P,
      wX := 0.30 * W,
      pY := 0.35 * P,
      wY := 0.40 * W * 0.90,
      wx := wX / pX,
      wy := wY / pY
  in wx / wy = 7 / 6 := 
by sorry

end citizen_wealth_ratio_l692_692473


namespace friend_area_multiple_l692_692748

theorem friend_area_multiple (tommy_north : ℕ) (tommy_east : ℕ) (tommy_west : ℕ) (tommy_south : ℕ) 
    (friend_area : ℕ) (tommy_area : ℕ) :
  tommy_north = 2 → tommy_east = 3 → tommy_west = 2 → tommy_south = 2 →
  tommy_area = (tommy_east - tommy_west) * (tommy_north - tommy_south) →
  friend_area = 80 →
  friend_area / tommy_area = 80 :=
by
  intros h_north h_east h_west h_south h_tommy_area h_friend_area
  rw [h_friend_area, h_tommy_area, h_east, h_west, h_north, h_south]
  norm_num

end friend_area_multiple_l692_692748


namespace area_of_DEF_l692_692429

-- Define the conditions
variables (Q : Point) (DEF : Triangle) (t1 t2 t3 : Triangle)
variables (area_t1 area_t2 area_t3 : ℝ)
variables (area_DEF : ℝ)

-- Assume the areas of the smaller triangles
axiom h_t1 : area_t1 = 16
axiom h_t2 : area_t2 = 25
axiom h_t3 : area_t3 = 36

-- Assume t1, t2, and t3 are the smaller triangles formed
-- by lines through Q parallel to the sides of DEF

-- Define the goal
theorem area_of_DEF (h_similar : ∀ (t : Triangle), similar t DEF) : 
  area_DEF = 225 :=
sorry

end area_of_DEF_l692_692429


namespace number_of_people_in_tour_group_l692_692808

theorem number_of_people_in_tour_group 
(total_cost_with_tax : ℝ)
(sales_tax_rate : ℝ)
(face_value_without_tax : ℝ)
(h_total_cost_with_tax : total_cost_with_tax = 945)
(h_sales_tax_rate : sales_tax_rate = 0.05)
(h_face_value_without_tax : face_value_without_tax = 35.91) :
  let total_cost_without_tax := total_cost_with_tax / (1 + sales_tax_rate)
      number_of_people := total_cost_without_tax / face_value_without_tax
  in number_of_people ≈ 25 :=
by
  -- Skipping the proof
  sorry

end number_of_people_in_tour_group_l692_692808


namespace total_surface_area_of_tower_l692_692090

theorem total_surface_area_of_tower :
  let volumes := [1, 27, 125, 64, 343, 216, 512, 729]
  let side_lengths := volumes.map (λ v, Int.ofNat (Real.toNat (Real.sqrt (v : ℝ))))
  let surface_areas := side_lengths.map (λ s, 6 * s^2)
  let adjusted_areas := [
    surface_areas[0],
    surface_areas[1] - side_lengths[0] ^ 2,
    surface_areas[2] - side_lengths[1] ^ 2,
    surface_areas[3] - side_lengths[1] ^ 2 - side_lengths[2] ^ 2,
    surface_areas[4] - side_lengths[2] ^ 2 - side_lengths[3] ^ 2,
    surface_areas[5] - side_lengths[3] ^ 2 - side_lengths[4] ^ 2,
    surface_areas[6] - side_lengths[4] ^ 2,
    surface_areas[7] - side_lengths[5] ^ 2
  ]
  adjusted_areas.sum = 1305 := by
    sorry

end total_surface_area_of_tower_l692_692090


namespace similar_triangles_l692_692137

structure Parabola (p : ℝ) where
  p_pos : p > 0
  parabola_eqn : ∀ (x y : ℝ), y ^ 2 = 2 * p * x

structure Point (ℝ) where
  x : ℝ
  y : ℝ

def Focus (p : ℝ) : Point ℝ :=
  { x := p / 2, y := 0 }

structure PointOnParabola (p : ℝ) extends Point ℝ where
  on_parabola : parabola_eqn p x y

variable (p : ℝ) [hp : Parabola p]

def inDifferentQuadrants (P Q : PointOnParabola p) : Prop :=
  (P.y > 0 ∧ Q.y < 0) ∨ (P.y < 0 ∧ Q.y > 0)

def isRPoint (P Q R : PointOnParabola p) : Prop :=
  2 * (angle_PR_x P R) = (angle_PF_x P) ∧
  2 * (angle_QR_x Q R) = (angle_QF_x Q) 

theorem similar_triangles (P Q R : PointOnParabola p) 
  (hDifferentQuadrants : inDifferentQuadrants P Q)
  (hRPoint : isRPoint P Q R) :
  △ (P.toPoint ℝ) (Focus p) R ~ △ (R) (Focus p) (Q.toPoint ℝ) :=
sorry

end similar_triangles_l692_692137


namespace probability_no_3by3_red_grid_correct_l692_692062

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l692_692062


namespace real_gdp_change_2015_l692_692688

noncomputable def gdp_2014 := 
  (1200 * 90) + (750 * 75)

noncomputable def real_gdp_2015 := 
  (900 * 90) + (900 * 75)

noncomputable def percentage_change := 
  (real_gdp_2015 - gdp_2014) * 100 / gdp_2014

theorem real_gdp_change_2015 : 
  percentage_change = -9.59 := 
sorry

end real_gdp_change_2015_l692_692688


namespace solve_for_c_l692_692182

theorem solve_for_c (a b c : ℝ) (h : 1/a - 1/b = 2/c) : c = (a * b * (b - a)) / 2 := by
  sorry

end solve_for_c_l692_692182


namespace area_triangle_ABC_l692_692408

-- Define the central regular hexagon with side length 1
def central_hexagon_side_length : ℝ := 1

-- Define the centers of the hexagons A, B, and C such that they are every second hexagon
-- surrounding the central hexagon
def circumradius (s : ℝ) : ℝ := s

-- Given that triangle ABC forms an equilateral triangle with a specific side length
def side_length_ABC (s : ℝ) : ℝ := 3 * s

-- Define the area of an equilateral triangle given its side length
def equilateral_triangle_area (a : ℝ) : ℝ := (sqrt 3 / 4) * a^2

theorem area_triangle_ABC :
  equilateral_triangle_area (side_length_ABC central_hexagon_side_length) = (9 * sqrt 3 / 4) :=
by
  sorry

end area_triangle_ABC_l692_692408


namespace exists_distinct_nats_with_square_sums_l692_692483

theorem exists_distinct_nats_with_square_sums :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (x y : ℕ), a^2 + 2 * c * d + b^2 = x^2 ∧ c^2 + 2 * a * b + d^2 = y^2 :=
sorry

end exists_distinct_nats_with_square_sums_l692_692483


namespace congruent_faces_of_tetrahedron_l692_692400

theorem congruent_faces_of_tetrahedron 
  (V : Type*)
  [triangle V] 
  {A B C D : V}
  (α β γ δ ε ζ η θ ι κ λ μ : ℝ)
  (hA : α + β + γ = 180)
  (hB : δ + ε + ζ = 180)
  (hC : η + θ + ι = 180)
  (hD : κ + λ + μ = 180) :
  congruent (triangle A B C) (triangle A B D) ∧
  congruent (triangle A B D) (triangle A C D) ∧
  congruent (triangle A C D) (triangle B C D) :=
sorry

end congruent_faces_of_tetrahedron_l692_692400


namespace probability_no_3x3_red_square_l692_692079

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l692_692079


namespace average_value_of_T_l692_692037

variables {T : Finset ℕ}
variable [DecidableEq ℕ]
hypotheses 
  (h1 : ∑ i in T.erase (T.max' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T))))))))) = 45 * (T.card - 1))
  (h2 : ∑ i in (T.erase (T.min' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.min'_mem T (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.min'_mem T (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.min'_mem T (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T))))))))) (T.erase (T.max' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T))))))) )) = 50 * (T.card - 2))
  (h3 : ∑ i in (insert (T.max' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T)))) (T.erase (T.min' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.min'_mem T))))) = 55 * (T.card - 1))
  (h4 : T.max' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.max'_mem T))) = T.min' (Finset.nonempty_of_ne_empty (Finset.ne_empty_of_mem (Finset.min'_mem T))) + 80)

theorem average_value_of_T (h1 h2 h3 h4) : T.sum.fintype.to_average = 50 :=
  sorry

end average_value_of_T_l692_692037


namespace construct_triangle_l692_692043

/-- Prove the existence of a triangle ABC given the side AB, angle BAC, and altitude from C to AB. -/
theorem construct_triangle (AB : ℝ) (angle_BAC : ℝ) (altitude_CC1 : ℝ) :
  ∃ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  ∃ (a b c : ℝ),
  a = AB ∧
  angle_BAC = ∠(/) a ∧
  altitude_CC1 = dist (/) b :=
sorry

end construct_triangle_l692_692043


namespace area_of_BEC_is_correct_l692_692885

-- Definitions of the variables and conditions
variable (AB CD EM : ℝ) (E : Point) (A B C D : Point)

-- Trapezoid properties and given conditions
axiom h1 : AB = 5
axiom h2 : CD = 9
axiom h3 : EM = 4
axiom h4 : ∠AEB = 90
axiom h5 : AB ∥ CD
axiom h6 : is_intersection E A C B D

-- Define a point struct for E
structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def area_triangle (A B C E : Point) : ℝ :=
  let x : ℝ := some_expression_satisfying_conditions -- This specific calculation can be based on the geometric properties in the full proof,
  0.5 * x * EM -- The definition area of triangle BEC

theorem area_of_BEC_is_correct : area_triangle A E B C = 4.5 := sorry


end area_of_BEC_is_correct_l692_692885


namespace min_value_expression_l692_692247

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/a + 1/b = 1 ∧ 
  (∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 9/y >= 6) := 
begin
  sorry
end

end min_value_expression_l692_692247


namespace largest_prime_factor_of_1001_l692_692372

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692372


namespace additional_amount_per_10_cents_l692_692788

-- Definitions of the given conditions
def expected_earnings_per_share : ℝ := 0.80
def dividend_ratio : ℝ := 0.5
def actual_earnings_per_share : ℝ := 1.10
def shares_owned : ℕ := 600
def total_dividend_paid : ℝ := 312

-- Proof statement
theorem additional_amount_per_10_cents (additional_amount : ℝ) :
  (total_dividend_paid - (shares_owned * (expected_earnings_per_share * dividend_ratio))) / shares_owned / 
  ((actual_earnings_per_share - expected_earnings_per_share) / 0.10) = additional_amount :=
sorry

end additional_amount_per_10_cents_l692_692788


namespace num_pairs_of_integers_satisfying_equation_l692_692935

theorem num_pairs_of_integers_satisfying_equation : 
  {p : ℕ × ℕ // p.1 > 0 ∧ p.2 > 0 ∧ (p.1^2 - p.2^2 = 77)}.to_list.length = 2 := 
sorry

end num_pairs_of_integers_satisfying_equation_l692_692935


namespace odd_function_log_a_l692_692177

theorem odd_function_log_a (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2^x + 2^(-x) * log10 a) :
  (∀ x, f (-x) = -f x) → a = 1 / 10 :=
by
  sorry

end odd_function_log_a_l692_692177


namespace whiteboards_per_class_is_10_l692_692587

-- Definitions from conditions
def classes : ℕ := 5
def ink_per_whiteboard_ml : ℕ := 20
def cost_per_ml_cents : ℕ := 50
def total_cost_cents : ℕ := 100 * 100  -- converting $100 to cents

-- Following the solution, define other useful constants
def cost_per_whiteboard_cents : ℕ := ink_per_whiteboard_ml * cost_per_ml_cents
def total_cost_all_classes_cents : ℕ := classes * total_cost_cents
def total_whiteboards : ℕ := total_cost_all_classes_cents / cost_per_whiteboard_cents
def whiteboards_per_class : ℕ := total_whiteboards / classes

-- We want to prove that each class uses 10 whiteboards.
theorem whiteboards_per_class_is_10 : whiteboards_per_class = 10 :=
  sorry

end whiteboards_per_class_is_10_l692_692587


namespace division_of_decimals_l692_692753

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l692_692753


namespace larger_number_is_seventy_two_l692_692570

def five_times_larger_is_six_times_smaller (x y : ℕ) : Prop := 5 * y = 6 * x
def difference_is_twelve (x y : ℕ) : Prop := y - x = 12

theorem larger_number_is_seventy_two (x y : ℕ) 
  (h1 : five_times_larger_is_six_times_smaller x y)
  (h2 : difference_is_twelve x y) : y = 72 :=
sorry

end larger_number_is_seventy_two_l692_692570


namespace vector_at_t_neg2_is_target_l692_692419

-- Defining the initial conditions
def t₅_vector : ℝ × ℝ := (0, 5)
def t₈_vector : ℝ × ℝ := (9, 1)

-- The target vector to prove
def target_vector : ℝ × ℝ := (21, -23 / 3)

-- The parameterized line 
def line (a d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := (a.1 + t * d.1, a.2 + t * d.2)

theorem vector_at_t_neg2_is_target :
  ∃ (a d : ℝ × ℝ), 
    line a d 5 = t₅_vector ∧ 
    line a d 8 = t₈_vector ∧ 
    line a d (-2) = target_vector :=
sorry

end vector_at_t_neg2_is_target_l692_692419


namespace number_of_valid_triplets_l692_692851

theorem number_of_valid_triplets : 
  (∃ S : Finset ℕ, S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧ 
  ((S.card = 12) ∧ (∃ R : Finset (Finset ℕ), 
  (R.card = 4) ∧ 
  (∀ T ∈ R, T.card = 3 ∧ (78 - T.sum = 63))))) :=
sorry

end number_of_valid_triplets_l692_692851


namespace largest_prime_factor_of_1001_l692_692375

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692375


namespace sum_S5_is_31_l692_692534

variable (a : ℕ → ℚ) 

noncomputable def q : ℚ := 
  if a 1 = 0 then 1 else (a 2 / a 1)

def S : ℕ → ℚ := 
  λ n => a 1 * (1 - q a ^ n) / (1 - q a)

axiom geom_seq (n : ℕ) : 
  a (n + 1) = a 1 * q a ^ n

axiom a2a3_eq_2a1 : 
  a 2 * a 3 = 2 * a 1

axiom arith_mean_a4_2a7 : 
  (a 4 + 2 * a 7) / 2 = 5 / 4

theorem sum_S5_is_31 : S a 5 = 31 := 
sorry

end sum_S5_is_31_l692_692534


namespace spadesuit_calculation_l692_692124

def spadesuit (x y : ℝ) : ℝ := x - 1 / y

theorem spadesuit_calculation :
  spadesuit 3 (spadesuit 3 (spadesuit 3 3)) = 55 / 21 := by
  sorry

end spadesuit_calculation_l692_692124


namespace arithmetic_sequence_iff_sum_formula_l692_692149

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Defining the condition that Sn is the sum of the first n terms of the sequence a
def is_sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ k in finset.range n, a (k + 1)

-- The main theorem statement
theorem arithmetic_sequence_iff_sum_formula (hS : is_sum_of_first_n_terms S a):
  (∀ n, S n = n * a n) ↔ (∃ d, ∀ n, a (n + 1) = a n + d) :=
sorry

end arithmetic_sequence_iff_sum_formula_l692_692149


namespace count_true_props_l692_692183

-- Original proposition
def original_prop (a α : Prop) : Prop :=
  ∀ l1 l2 : Prop, (a → l1 ∧ a → l2) → (a → α)

-- Converse
def converse (a α : Prop) : Prop :=
  ∀ l1 l2 : Prop, (a → α) → (a → l1 ∧ a → l2)

-- Negation
def negation (a α : Prop) : Prop :=
  ¬(∀ l1 l2 : Prop, (a → l1 ∧ a → l2)) ∧ ¬(a → α)

-- Contrapositive
def contrapositive (a α : Prop) : Prop :=
  ¬(a → α) → ¬(∀ l1 l2 : Prop, (a → l1 ∧ a → l2))

theorem count_true_props (a α : Prop) :
  (original_prop a α ∨ converse a α ∨ negation a α ∨ contrapositive a α) →
  (original_prop a α →
   converse a α →
   contrapositive a α →
   3 = 3) :=
by
  intros h h1 h2 h3
  exact eq.refl 3

end count_true_props_l692_692183


namespace son_distance_from_father_is_correct_l692_692387

noncomputable def distance_between_son_and_father 
  (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1) 
  (incident_point_condition : F / d = L / (d + x) ∧ S / x = F / (d + x)) : ℝ :=
  4.9

theorem son_distance_from_father_is_correct (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1)
  (incident_point_condition : F / d = L / (d + 4.9) ∧ S / 4.9 = F / (d + 4.9)) : 
  distance_between_son_and_father L F S d h_L h_F h_S h_d incident_point_condition = 4.9 :=
sorry

end son_distance_from_father_is_correct_l692_692387


namespace angle_BED_l692_692576

theorem angle_BED (A B C D E : Type) [geometry A B C D E]
  (hA : ∠A = 45)
  (hC : ∠C = 85)
  (hD : midpoint D A B)
  (hE : midpoint E B C)
  (hDB_BE : DB = BE) : ∠BED = 65 := sorry

end angle_BED_l692_692576


namespace blue_line_length_l692_692858

theorem blue_line_length (w b : ℝ) (h1 : w = 7.666666666666667) (h2 : w = b + 4.333333333333333) :
  b = 3.333333333333334 :=
by sorry

end blue_line_length_l692_692858


namespace pharmacy_tubs_needed_l692_692428

theorem pharmacy_tubs_needed 
  (total_tubs_needed : ℕ) 
  (tubs_in_storage : ℕ) 
  (fraction_bought_new_vendor : ℚ) 
  (total_tubs_needed = 100) 
  (tubs_in_storage = 20)
  (fraction_bought_new_vendor = 1 / 4) :
  let tubs_needed_to_buy := total_tubs_needed - tubs_in_storage in
  let tubs_from_new_vendor := (tubs_needed_to_buy / 4 : ℕ) in
  let total_tubs_now := tubs_in_storage + tubs_from_new_vendor in
  let tubs_from_usual_vendor := total_tubs_needed - total_tubs_now in
  tubs_from_usual_vendor = 60 := 
by sorry

end pharmacy_tubs_needed_l692_692428


namespace division_equivalent_l692_692756

def division_to_fraction (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 ∧ 0 ≤ a ∧ 0 ≤ b → a / b = (a * 1000) / (b * 1000) :=
by
  intros h
  field_simp
  
theorem division_equivalent (h : 0 ≤ 0.08 ∧ 0 ≤ 0.002 ∧ 0.08 ≠ 0 ∧ 0.002 ≠ 0) :
  0.08 / 0.002 = 40 :=
by
  have := division_to_fraction 0.08 0.002 h
  norm_num at this
  exact this

end division_equivalent_l692_692756


namespace probability_no_3x3_red_square_l692_692070

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l692_692070


namespace triangle_area_doubled_side_angle_l692_692571

-- Definitions of the initial condition
def triangle_sides : Type := ℤ
def triangle_angles : Type := ℝ

def triangle (a b c : triangle_sides) (α β γ : triangle_angles) : Prop := 
  ∃ a b c : triangle_sides, 
    ∃ α β γ : triangle_angles, 
      α + β + γ = π -- Angle sum property of triangle

def area (a b c : triangle_sides) (α β γ : triangle_angles) : ℝ := 
  (a * b * sin α) / 2 -- Area formula using side a and angle α.

-- Mathematical proof problem
theorem triangle_area_doubled_side_angle (a b c : triangle_sides) (α β γ : triangle_angles) 
  (h : triangle a b c α β γ) : area (2 * a) b c (2 * α) β γ = 4 * area a b c α β γ :=
sorry

end triangle_area_doubled_side_angle_l692_692571


namespace part1_part2_l692_692973

noncomputable def inverse_function_constant (k : ℝ) : Prop :=
  (∀ x : ℝ, 0 < x → (x, 3) ∈ {p : ℝ × ℝ | p.snd = k / p.fst})

noncomputable def range_m (m : ℝ) : Prop :=
  0 < m → m < 3

theorem part1 (k : ℝ) (hk : k ≠ 0) (h : (1, 3).snd = k / (1, 3).fst) :
  k = 3 := by
  sorry

theorem part2 (m : ℝ) (hm : m ≠ 0) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) := by
  sorry

end part1_part2_l692_692973


namespace cartesian_eq_curve_C_trajectory_eq_midpoint_P_l692_692909

-- Defining the parametric equations of curve C
def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, 2 * Real.sin θ)

-- Defining the coordinate transformation
def coordinate_transformation (x y : ℝ) : ℝ × ℝ :=
  (x / 3, y / 2)

-- Cartesian equation of curve C'
theorem cartesian_eq_curve_C' :
  ∀ (θ : ℝ), (coordinate_transformation (3 * Real.cos θ) (2 * Real.sin θ)) = (Real.cos θ, Real.sin θ) ∧ 
   (Real.cos θ)^2 + (Real.sin θ)^2 = 1 :=
by
  intro θ
  sorry

-- Trajectory equation of midpoint P of AB with A on curve C' and B(3,0)
theorem trajectory_eq_midpoint_P (x y : ℝ) :
  (2 * x - 3)^2 + (2 * y)^2 = 1 → (x - 3 / 2)^2 + y^2 = 1 / 4 :=
by 
  intro h
  sorry

end cartesian_eq_curve_C_trajectory_eq_midpoint_P_l692_692909


namespace question1_question2_l692_692029

-- For question (1)
theorem question1 : 
  (Real.sqrt 25 - Real.cbrt 27 - abs (Real.sqrt 3 - 2) = Real.sqrt 3) :=
by
  sorry

-- For question (2)
theorem question2 :
  (Real.sqrt 3 * (Real.sqrt 3 - 1) + Real.sqrt ((-2)^2) - Real.cbrt ((7 / 8) - 1) = 11 / 2 - Real.sqrt 3) :=
by
  sorry

end question1_question2_l692_692029


namespace reflection_symmetry_y_axis_l692_692591

theorem reflection_symmetry_y_axis (P : ℝ × ℝ) (h : P = (2, 1)) : 
  let P' := (-(P.1), P.2) in P' = (-2, 1) :=
by
  sorry

end reflection_symmetry_y_axis_l692_692591


namespace find_minimum_a_plus_b_l692_692174

open Real

-- Define the problem conditions and the objective statement
theorem find_minimum_a_plus_b {a b : ℝ} (h : log 4 (3 * a + 4 * b) = log 2 (sqrt (a * b))) (a_pos : 0 < a) (b_pos : 0 < b) : 
  a + b ≥ 7 + 4 * sqrt 3 :=
sorry

end find_minimum_a_plus_b_l692_692174


namespace Marie_speed_l692_692250

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end Marie_speed_l692_692250


namespace largest_prime_factor_1001_l692_692365

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692365


namespace watermelon_not_necessarily_divided_l692_692444

noncomputable def radius := 10

def cut_depth_1 := 17
def cut_depth_2 := 18

theorem watermelon_not_necessarily_divided (R : ℝ) (h : ℝ) (h ∈ {17, 18}) : ¬(sphere_divided R h) := sorry

end watermelon_not_necessarily_divided_l692_692444


namespace largest_prime_factor_of_1001_l692_692334

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692334


namespace option_a_is_correct_l692_692450

-- Define what it means for two monomials to be of the same type.
def same_type (m1 m2 : ℕ → ℕ) : Prop := 
  ∀ n, m1 n = m2 n

-- Define the monomials in question as functions from variable indices to their exponents.
def monomial1 (n : ℕ) : ℕ :=
  match n with 
  | 0 => 2  -- exponent of a
  | 1 => 3  -- exponent of b
  | _ => 0  -- no other variables
  end

def monomialA (n : ℕ) : ℕ :=
  match n with 
  | 0 => 2  -- exponent of a
  | 1 => 3  -- exponent of b
  | _ => 0  -- no other variables
  end

-- The proof statement
theorem option_a_is_correct : same_type monomial1 monomialA := 
  by
  intros n
  cases n 
  sorry -- Proof not required

end option_a_is_correct_l692_692450


namespace hyperbola_eccentricity_l692_692519

theorem hyperbola_eccentricity (m n : ℝ) (hcond1 : n > m) (hcond2 : m > 0) 
  (h_ecc_ellipse : (sqrt 2) / 2 = sqrt (1 - (1/n) / (1/m))) : 
  let e_hyperbola := sqrt (1 + (1/n) / (1/m)) in 
  e_hyperbola = (sqrt 6) / 2 :=
sorry

end hyperbola_eccentricity_l692_692519


namespace find_absolute_value_l692_692959

theorem find_absolute_value 
(h : ℝ) (k : ℝ)
(h_eq1 : k = 2 * h + 2)
(h_eq2 : k = 18 * h - 258) :
  |3 * h + 2 * k| = 117.75 :=
by
sory

end find_absolute_value_l692_692959


namespace correct_option_l692_692449

def optionA (x : ℝ) : Prop := 6 + x = 10 → x = 10 + 6
def optionB (x : ℕ) : Prop := 3 * x + 5 = 4 * x → 3 * x - 4 * x = -5
def optionC (x : ℝ) : Prop := 8 * x = 4 - 3 * x → 8 * x - 3 * x = 4
def optionD (x : ℝ) : Prop := 2 * (x - 1) = 3 → 2 * x - 1 = 3

theorem correct_option :
  (∃ x : ℕ, optionB x) ∧ (¬ ∃ x : ℝ, optionA x ∧ optionC x ∧ optionD x) :=
by {
  sorry
}

end correct_option_l692_692449


namespace largest_prime_factor_1001_l692_692358

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692358


namespace opposite_of_neg2023_l692_692700

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l692_692700


namespace commute_days_l692_692786

theorem commute_days (a b d e x : ℕ) 
  (h1 : b + e = 12)
  (h2 : a + d = 20)
  (h3 : a + b = 15)
  (h4 : x = a + b + d + e) :
  x = 32 :=
by {
  sorry
}

end commute_days_l692_692786


namespace arithmetic_geometric_mean_eq_or_ge_l692_692552

theorem arithmetic_geometric_mean_eq_or_ge (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : c = (a + b) / 2) :
  (a + b + c) / 3 ≥ (↑3√(a * b * c)) :=
by
  sorry

end arithmetic_geometric_mean_eq_or_ge_l692_692552


namespace min_value_of_a1_b2_l692_692526

theorem min_value_of_a1_b2 (a b : ℝ) (h1 : a > 1) (h2 : ab = 2a + b) : (a + 1) * (b + 2) ≥ 18 :=
by
  sorry

end min_value_of_a1_b2_l692_692526


namespace largest_prime_factor_1001_l692_692337

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692337


namespace measure_15_minutes_with_hourglasses_l692_692167

theorem measure_15_minutes_with_hourglasses (h7 h11 : ℕ) (h7_eq : h7 = 7) (h11_eq : h11 = 11) : ∃ t : ℕ, t = 15 :=
by
  let t := 15
  have h7 : ℕ := 7
  have h11 : ℕ := 11
  exact ⟨t, by norm_num⟩

end measure_15_minutes_with_hourglasses_l692_692167


namespace one_number_greater_than_one_l692_692271

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1)
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  ((1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ 1 < b ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ 1 < c)) 
  ∧ (¬ ((1 < a ∧ 1 < b) ∨ (1 < b ∧ 1 < c) ∨ (1 < a ∧ 1 < c))) :=
sorry

end one_number_greater_than_one_l692_692271


namespace product_of_last_two_digits_of_divisible_by_6_l692_692954

-- Definitions
def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def sum_of_last_two_digits (n : ℤ) (a b : ℤ) : Prop := (n % 100) = 10 * a + b

-- Theorem statement
theorem product_of_last_two_digits_of_divisible_by_6 (x a b : ℤ)
  (h1 : is_divisible_by_6 x)
  (h2 : sum_of_last_two_digits x a b)
  (h3 : a + b = 15) :
  (a * b = 54 ∨ a * b = 56) := 
sorry

end product_of_last_two_digits_of_divisible_by_6_l692_692954


namespace quadratic_has_one_solution_k_l692_692283

theorem quadratic_has_one_solution_k (
  k : ℚ
) :
  (∃ (a b c : ℚ), a = 2 ∧ b = -5 ∧ c = k ∧ (b^2 - 4 * a * c = 0)) → k = 25 / 8 :=
begin
  intro h,
  cases h with a ha,
  cases ha with b hb,
  cases hb with c hc,
  cases hc with ha2 ha3,
  cases ha3 with hb5 hc_k,
  cases hc_k with h_disc zero_discrim,
  have ha_eq : a = 2 := ha2,
  have hb_eq : b = -5 := hb5,
  have hc_eq : c = k := hc_k,
  rw [ha_eq, hb_eq] at zero_discrim,
  simp at zero_discrim,
  sorry -- Proof to be filled in
end

end quadratic_has_one_solution_k_l692_692283


namespace exercise_sum_of_squares_l692_692857

theorem exercise_sum_of_squares :
  (∑ k in finset.range 50, (2 * k + 2)^2 - (2 * k + 1)^2) = 5050 :=
by
  sorry

end exercise_sum_of_squares_l692_692857


namespace jesse_total_carpet_l692_692992

theorem jesse_total_carpet : 
  let length_rect := 12
  let width_rect := 8
  let base_tri := 10
  let height_tri := 6
  let area_rect := length_rect * width_rect
  let area_tri := (base_tri * height_tri) / 2
  area_rect + area_tri = 126 :=
by
  sorry

end jesse_total_carpet_l692_692992


namespace eccentricity_of_ellipse_l692_692898

theorem eccentricity_of_ellipse {a b c : ℝ} (h1 : a > b > 0) (h2 : c = sqrt (a^2 - b^2)) 
(x₀ y₀ : ℝ) (hP : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
(hG : ∃ G : ℝ × ℝ, 3 * (x₀ - G.1, y₀ - G.2) = ((x₀ + c), y₀) - ((x₀ - c), y₀)) 
(I : ℝ × ℝ) (λ : ℝ) (hI : I = (0, λ * (2 * c))) 
: ∃ (e : ℝ), e = 1 / 2 :=
sorry

end eccentricity_of_ellipse_l692_692898


namespace comics_in_box_l692_692016

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l692_692016


namespace correct_calculation_l692_692382

theorem correct_calculation (a : ℝ) : a^2 * a^3 = a^5 := by
  calc
    a^2 * a^3 = a^(2 + 3) : by ring_exp
          ... = a^5      : by norm_num

end correct_calculation_l692_692382


namespace solve_for_y_in_terms_of_x_l692_692565

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3 * x) : y = -2 * x - 2 :=
sorry

end solve_for_y_in_terms_of_x_l692_692565


namespace smallest_abs_z_add_i_l692_692231

noncomputable def smallest_possible_value (z : ℂ) (h : abs (z^2 + 9) = abs (z * (z + 3 * complex.I))) : ℝ :=
  Inf { abs (z + complex.I) | abs (z^2 + 9) = abs (z * (z + 3 * complex.I)) }

theorem smallest_abs_z_add_i :
  smallest_possible_value z (abs (z^2 + 9) = abs (z * (z + 3 * complex.I))) = 2 :=
sorry

end smallest_abs_z_add_i_l692_692231


namespace bumper_car_line_total_in_both_lines_l692_692454

theorem bumper_car_line (x y Z : ℕ) (hZ : Z = 25 - x + y) : Z = 25 - x + y :=
by
  sorry

theorem total_in_both_lines (x y Z : ℕ) (hZ : Z = 25 - x + y) : 40 - x + y = Z + 15 :=
by
  sorry

end bumper_car_line_total_in_both_lines_l692_692454


namespace largest_prime_factor_of_1001_l692_692349

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692349


namespace boat_speed_is_13_l692_692406

noncomputable def boatSpeedStillWater : ℝ := 
  let Vs := 6 -- Speed of the stream in km/hr
  let time := 3.6315789473684212 -- Time taken in hours to travel 69 km downstream
  let distance := 69 -- Distance traveled in km
  (distance - Vs * time) / time

theorem boat_speed_is_13 : boatSpeedStillWater = 13 := by
  sorry

end boat_speed_is_13_l692_692406


namespace inequality_problem_l692_692564

variable (a b c d : ℝ)

theorem inequality_problem (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := sorry

end inequality_problem_l692_692564


namespace expand_product_l692_692487

theorem expand_product :
  ∀ (x y : ℝ), 4 * (x + 3) * (x + 2 + y) = 4 * x^2 + 4 * x * y + 20 * x + 12 * y + 24 :=
by
  intros x y
  sorry

end expand_product_l692_692487


namespace problem_solution_l692_692516

-- Define the sequence {a_n} satisfying the given conditions
theorem problem_solution (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (hT : ∀ n, (finset.range (n+1)).prod a = T n)
  (h1 : ∀ n, a n + T n = 1) :
  (∀ n, a n = n / (n + 1)) ∧ 
  (∀ n S_n, S_n = (finset.range n).sum (λ k, (a (k + 1)) / (a k) - 1) → (1 / 3 ≤ S_n) ∧ (S_n < 3 / 4)) := 
by 
  sorry

end problem_solution_l692_692516


namespace _l692_692834

def count_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log10 n).toNat + 1

example : count_digits (12_457_789_102_345_679 * 273_456_981_234_567) = 33 := 
by
  -- Use the theorem that product of the digit counts or approximation
  sorry

end _l692_692834


namespace distance_range_l692_692689

variable {A B S : Type} -- Locations A, B, and the school S
variable [metric_space A] [metric_space B] [metric_space S]

-- Defining distances using the metric_space
def distance (x y : Type) [metric_space x] [metric_space y] : ℝ

-- Conditions
axiom dist_A_S : distance A S = 5
axiom dist_B_S : distance B S = 2
variable (d : ℝ)

theorem distance_range (dist_A_S : distance A S = 5) (dist_B_S : distance B S = 2) :
  3 ≤ d ∧ d ≤ 7 :=
sorry

end distance_range_l692_692689


namespace min_formula_l692_692682

theorem min_formula (a b : ℝ) : 
  min a b = (a + b - Real.sqrt((a - b)^2)) / 2 :=
sorry

end min_formula_l692_692682


namespace sequence_converges_to_limit_l692_692298

theorem sequence_converges_to_limit :
  ∃ L, (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (a n - L) < ε) ∧ L = 1 / 2
:= sorry
where
  a : ℕ → ℝ
  | 0     := 0
  | (n+1) := (1 / 5) * (2 * (a n) ^ 3 - (a n) ^ 2 + 3 * (a n) + 1)

end sequence_converges_to_limit_l692_692298


namespace roots_real_and_distinct_l692_692479

theorem roots_real_and_distinct (x : ℝ) :
    ∃ (a b c : ℝ), a = 1 ∧ b = 6 ∧ c = 8 ∧ a * x^2 + b * x + c = 0 ∧ (b^2 - 4 * a * c > 0) :=
by
  let a := 1
  let b := 6
  let c := 8
  have h_eq : a * x^2 + b * x + c = x^2 + 6 * x + 8, by sorry
  have h_disc : b^2 - 4 * a * c = 4, by sorry
  have h_positive : b^2 - 4 * a * c > 0, by sorry
  exact ⟨a, b, c, rfl, rfl, rfl, h_eq, h_positive⟩

end roots_real_and_distinct_l692_692479


namespace largest_prime_factor_1001_l692_692363

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692363


namespace percentage_raise_l692_692855

theorem percentage_raise (original_salary new_salary : ℕ) (video_game_spending_percent : ℝ) (video_game_spending last_year_video_game_spending new_salary_amount : ℕ) :
  video_game_spending_percent = 0.40 →
  last_year_video_game_spending = 100 →
  new_salary_amount = 275 →
  video_game_spending = video_game_spending_percent * original_salary →
  original_salary = last_year_video_game_spending / video_game_spending_percent →
  new_salary - original_salary = 25 →
  ((new_salary - original_salary) / original_salary) * 100 = 10 :=
begin
  -- sorry to skip the proof
  sorry
end

end percentage_raise_l692_692855


namespace Marie_speed_l692_692251

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end Marie_speed_l692_692251


namespace hammerhead_teeth_fraction_l692_692807

theorem hammerhead_teeth_fraction (f : ℚ) : 
  let t := 180 
  let h := f * t
  let w := 2 * (t + h)
  w = 420 → f = (1 : ℚ) / 6 := by
  intros _ 
  sorry

end hammerhead_teeth_fraction_l692_692807


namespace complex_conjugate_sum_l692_692134

theorem complex_conjugate_sum (z : ℂ) (h : z = (2 + 1 * Complex.I) / (1 + 1 * Complex.I)) : 
    z + Complex.conj z = 3 :=
sorry

end complex_conjugate_sum_l692_692134


namespace probability_same_grade_l692_692309

open Finset

def products : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def firstGrade : Finset ℕ := {1, 2, 3, 4}
def secondGrade : Finset ℕ := {5, 6, 7}
def thirdGrade : Finset ℕ := {8, 9, 10}

noncomputable def count_combinations (s : Finset ℕ) (k : ℕ) : ℕ :=
  Fintype.card {t : Finset ℕ // t.card = k ∧ t ⊆ s}

theorem probability_same_grade :
  let same_grade := count_combinations firstGrade 2 + count_combinations secondGrade 2 + count_combinations thirdGrade 2
  let total := count_combinations products 2
  (same_grade : ℚ) / total = 4 / 15 :=
by
  sorry

end probability_same_grade_l692_692309


namespace nature_of_roots_l692_692049

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 + 3 * x^2 - 8 * x + 16

theorem nature_of_roots : (∀ x : ℝ, x < 0 → P x > 0) ∧ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ P x = 0 := 
by
  sorry

end nature_of_roots_l692_692049


namespace sum_of_f_greater_than_zero_l692_692157

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem sum_of_f_greater_than_zero 
  (a b c : ℝ) 
  (h1 : a + b > 0) 
  (h2 : b + c > 0) 
  (h3 : c + a > 0) : 
  f a + f b + f c > 0 := 
by 
  sorry

end sum_of_f_greater_than_zero_l692_692157


namespace three_boxes_incur_fee_l692_692798

def box (length height : ℕ) := length / height

def shouldIncurFee (length height : ℕ) : Bool :=
  (box length height < 1.5) ∨ (box length height > 3.0)

def boxX_length := 8
def boxX_height := 5

def boxY_length := 10
def boxY_height := 2

def boxZ_length := 7
def boxZ_height := 7

def boxW_length := 14
def boxW_height := 4

theorem three_boxes_incur_fee :
  (shouldIncurFee boxX_length boxX_height) +
  (shouldIncurFee boxY_length boxY_height) +
  (shouldIncurFee boxZ_length boxZ_height) +
  (shouldIncurFee boxW_length boxW_height) = 3 :=
by
  sorry

end three_boxes_incur_fee_l692_692798


namespace probability_no_3x3_red_square_l692_692071

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l692_692071


namespace average_attendance_l692_692401

def monday_attendance := 10
def tuesday_attendance := 15
def wednesday_attendance := 10
def thursday_attendance := 10
def friday_attendance := 10
def total_days := 5

theorem average_attendance :
  (monday_attendance + tuesday_attendance + wednesday_attendance + thursday_attendance + friday_attendance) / total_days = 11 :=
by
  sorry

end average_attendance_l692_692401


namespace macaroon_count_l692_692125

def baked_red_macaroons : ℕ := 50
def baked_green_macaroons : ℕ := 40
def ate_green_macaroons : ℕ := 15
def ate_red_macaroons := 2 * ate_green_macaroons

def remaining_macaroons : ℕ := (baked_red_macaroons - ate_red_macaroons) + (baked_green_macaroons - ate_green_macaroons)

theorem macaroon_count : remaining_macaroons = 45 := by
  sorry

end macaroon_count_l692_692125


namespace number_of_ways_to_sum_10003_as_two_primes_l692_692848

theorem number_of_ways_to_sum_10003_as_two_primes : 
  (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003) ∧ 
  (∀ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 → (p = 2 ∨ q = 2)) → 1 := 
sorry

end number_of_ways_to_sum_10003_as_two_primes_l692_692848


namespace necessary_and_sufficient_condition_l692_692518

variables {A B C P D E F : Type*}
variables [has_dist A] [has_dist B] [has_dist C] [has_dist P]
variables [∀ t, has_projection t D] [∀ t, has_projection t E] [∀ t, has_projection t F]

def AF (A P : Type*) [has_dist A] [has_dist P] : ℝ := sorry
def BD (B P : Type*) [has_dist B] [has_projection P D] : ℝ := sorry
def CE (C P : Type*) [has_dist C] [has_projection P E] : ℝ := sorry

noncomputable def perimeter {A B C : Type*} [has_dist A] [has_dist B] [has_dist C] : ℝ := sorry

theorem necessary_and_sufficient_condition (L : ℝ) (P_incenter : P) (P_circumcenter : P) 
  (hAF : ∀ (A P : Type*) [has_dist A] [has_dist P], AF A P = sorry)
  (hBD : ∀ (B P : Type*) [has_dist B] [has_projection P D], BD B P = sorry)
  (hCE : ∀ (C P : Type*) [has_dist C] [has_projection P E], CE C P = sorry)
  (h_perimeter : perimeter = L) :
  2 * (AF A P + BD B P + CE C P) = L ↔ P = P_incenter ∨ P = P_circumcenter :=
sorry


end necessary_and_sufficient_condition_l692_692518


namespace common_ratio_eq_l692_692228

variables {a : ℕ → ℝ} {S : ℕ → ℝ} (q : ℝ)

-- Definition for geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a n = a 0 * q ^ n

-- Definition for the sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in range(n+1), a i

-- Conditions from the problem
variables [geometric_sequence a q] [sum_of_first_n_terms a S]

-- Definition for arithmetic sequence
def arithmetic_sequence (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Given condition that -S₁, S₂, and a₃ form an arithmetic sequence
axiom arithmetic_seq_condition : arithmetic_sequence (-S 1) (S 2) (a 3)

-- Main goal to prove
theorem common_ratio_eq : q = -1 ∨ q = 3 :=
sorry

end common_ratio_eq_l692_692228


namespace odd_expression_divisible_by_48_l692_692269

theorem odd_expression_divisible_by_48 (x : ℤ) (h : Odd x) : 48 ∣ (x^3 + 3*x^2 - x - 3) :=
  sorry

end odd_expression_divisible_by_48_l692_692269


namespace exists_subset_X_l692_692836

theorem exists_subset_X :
  ∃ (X : Set ℕ), X ⊆ {n | n < 2 ^ 1996}
    ∧ 1 ∈ X
    ∧ (2 ^ 1996 - 1) ∈ X
    ∧ (∀ x ∈ X, x ≠ 1 → (∃ (a b ∈ X), a ≠ b ∧ x = a + b) ∨ (∃ y ∈ X, x = 2 * y))
    ∧ X.Finite
    ∧ X.toFinset.card ≤ 2012 := 
sorry

end exists_subset_X_l692_692836


namespace distance_planes_l692_692494

theorem distance_planes :
  let plane1 := {p : ℝ × ℝ × ℝ | 3 * p.1 - p.2 + 2 * p.3 = 6}
  let plane2 := {p : ℝ × ℝ × ℝ | 6 * p.1 - 2 * p.2 + 4 * p.3 = -4}
  ∃ d : ℝ, d = (abs (3 * 0 - 1 * 0 + 2 * (-1) - 6)) / sqrt (3^2 + (-1)^2 + 2^2) ∧ d = 4 * sqrt 14 / 7 :=
begin
  -- Insert proof here
  sorry
end

end distance_planes_l692_692494


namespace max_area_of_triangle_l692_692600

noncomputable def triangle_conditions (a b c : ℝ) : Prop :=
  ∀ (A B C : Angle) (u v w d : ℝ^2),
  (side u v = give_length a) ∧ (side v w = give_length b) ∧ (side w u = give_length c) ∧
  (2 * b * (cos C.toReal) = 2 * a - (sqrt 3) * c) ∧
  ((CA ⬝ CB) = 2 * CM) ∧ 
  (norm CM = 1)

theorem max_area_of_triangle {a b c : ℝ} :
  triangle_conditions a b c →
  ∃ (B : ℝ), B = π / 6 ∧ ∃ A C : ℝ, ∀ (S : Triangle), area S ≤ (sqrt 3) / 2 :=
by
  sorry

end max_area_of_triangle_l692_692600


namespace largest_prime_factor_1001_l692_692360

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692360


namespace QT_value_l692_692200

noncomputable def find_QT (PQ RS PT : ℝ) : ℝ :=
  let tan_gamma := (RS / PQ)
  let QT := (RS / tan_gamma) - PT
  QT

theorem QT_value :
  let PQ := 45
  let RS := 75
  let PT := 15
  find_QT PQ RS PT = 210 := by
  sorry

end QT_value_l692_692200


namespace alpha_in_third_quadrant_l692_692175

theorem alpha_in_third_quadrant (α : ℝ)
 (h₁ : Real.tan (α - 3 * Real.pi) > 0)
 (h₂ : Real.sin (-α + Real.pi) < 0) :
 (0 < α % (2 * Real.pi) ∧ α % (2 * Real.pi) < Real.pi) := 
sorry

end alpha_in_third_quadrant_l692_692175


namespace track_circumference_l692_692396

theorem track_circumference (A B : ℝ) (start_diam: ℝ) (meet_b: ℝ) (meet_a: ℝ):
  start_diam / 2 = A → 
  start_diam / 2 = B → 
  meet_b = 100 →
  meet_a := 2 - 60 →
  (start_diam = 480) :=
begin
  sorry
end

end track_circumference_l692_692396


namespace problem_1_problem_2_problem_3_l692_692633

-- Definition of the set of functions F
def F (n : ℕ) : Set (ℕ → ℕ) := {f | ∀ x, x ∈ Finset.range n → f x ∈ Finset.range n}

-- Problem 1: Prove |F| = n^n
theorem problem_1 (n : ℕ) : F n.card = n^n := 
sorry

-- Problem 2: For n = 2k, prove n^n < e * (4k)^k
theorem problem_2 (k : ℕ) (h : n = 2 * k) : n^n < Real.exp 1 * (4 * k)^k := 
sorry

-- Problem 3: Prove that there is no n such that n = 2k and n^n = 540
theorem problem_3 : ¬ ∃ k : ℕ, let n := 2 * k in n^n = 540 := 
sorry

end problem_1_problem_2_problem_3_l692_692633


namespace different_participation_methods_l692_692667

theorem different_participation_methods :
  ∃ (A B C D : set (fin 5)), (A ∪ B ∪ C ∪ D = ⊤ ∧ 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ D ≠ ∅ ∧
  disjoint A B ∧ disjoint A C ∧ disjoint A D ∧ disjoint B C ∧ disjoint B D ∧ disjoint C D ∧
  fin.succ 0 ∉ D) → 
  (card {p : { A B C D ∥ A ∪ B ∪ C ∪ D = ⊤ ∧ 
                     A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ D ≠ ∅ ∧
                     disjoint A B ∧ disjoint A C ∧ disjoint A D ∧ disjoint B C ∧ disjoint B D ∧ disjoint C D ∧
                     fin.succ 0 ∉ D | true }} = 180) := 
by
  sorry

end different_participation_methods_l692_692667


namespace opposite_of_sqrt7_minus_3_l692_692303

theorem opposite_of_sqrt7_minus_3 : -(sqrt 7 - 3) = 3 - sqrt 7 :=
by 
  sorry

end opposite_of_sqrt7_minus_3_l692_692303


namespace problem2_problem3_l692_692415

noncomputable def problem1 (A ω φ : ℝ) (h1 : A > 0) (h2 : ω > 0) 
  (h3 : ∀ x ∈ Set.Ioo 0 (7 * Real.pi), A * Real.sin (ω * x + φ) ≤ 3)
  (h4 : A * Real.sin (ω * Real.pi + φ) = 3)
  (h5 : A * Real.sin (ω * (6 * Real.pi) + φ) = -3) :
  A = 3 ∧ ω = 1 / 5 ∧ φ = 3 * Real.pi / 10 :=
by
  sorry

theorem problem2 :
  ∀ k : ℤ,
  ∀ x ∈ Set.Ioo (10 * k * Real.pi - 4 * Real.pi) (10 * k * Real.pi + Real.pi),
  Real.derivative (λ x, 3 * Real.sin (1 / 5 * x + 3 * Real.pi / 10)) x > 0 :=
by
  sorry

theorem problem3 : 
  ∃ (m : ℝ), 1 / 2 < m ∧ m ≤ 2 ∧
  3 * Real.sin (1 / 5 * Real.sqrt (-m^2 + 2 * m + 3) + 3 * Real.pi / 10) >
  3 * Real.sin (1 / 5 * Real.sqrt (-m^2 + 4) + 3 * Real.pi / 10) :=
by
  sorry

end problem2_problem3_l692_692415


namespace equal_circle_radii_l692_692210

theorem equal_circle_radii (a r : ℝ) (h_a_pos : 0 < a) (h_r_pos : 0 < r) :
  ∃ x > 0, (x = ar / (a + 2r)) := 
sorry

end equal_circle_radii_l692_692210


namespace probability_ξ_eta_leq_z_probability_ξ_div_eta_leq_z_l692_692241

variable {ξ η : ℝ}

-- Assuming ξ and η are independent random variables
axiom independent_ξ_η : ¬(ξ = η)

def Fξ (x : ℝ) : ℝ := ∫ t in -∞..x, fξ t
def Fη (y : ℝ) : ℝ := ∫ t in -∞..y, fη t

theorem probability_ξ_eta_leq_z (z : ℝ) (x : List ℝ) (p n : Nat)
  (hx_neg : ∀ k, k < p → x[k] < 0)  -- x[k] < 0 for k = 1,...,p
  (hx_pos : ∀ k, p ≤ k ∧ k ≤ n → x[k] > 0) -- x[k] > 0 for k = p+1,...,n
  (fξ : ℝ → ℝ) (fη : ℝ → ℝ) :
  ∑ k in (Finset.range p), (Fη (x[k]) - Fη (x[k] - 0)) * (1 - Fξ (z / x[k] - 0)) +
  ∑ k in (Finset.Icc p n), (Fη (x[k]) - Fη (x[k] - 0)) * Fξ (z / x[k]) = 
  ∫ t in 0..z, (fξ (t / η)) * (fη η) :=
sorry

theorem probability_ξ_div_eta_leq_z (z : ℝ) (x : List ℝ) (p n : Nat)
  (hx_neg : ∀ k, k < p → x[k] < 0)  -- x[k] < 0 for k = 1,...,p
  (hx_pos : ∀ k, p ≤ k ∧ k ≤ n → x[k] > 0) -- x[k] > 0 for k = p+1,...,n
  (fξ : ℝ → ℝ) (fη : ℝ → ℝ) :
  ∑ k in (Finset.range p), (Fη (x[k]) - Fη (x[k] - 0)) * 
      (1 - Fξ (z * x[k] - 0)) +
  ∑ k in (Finset.Icc p n), (Fη (x[k]) - Fη (x[k] - 0)) * 
      Fξ (z * x[k]) = 
  ∫ t in -∞..z, (fξ (t * η)) * (fη η) :=
sorry

end probability_ξ_eta_leq_z_probability_ξ_div_eta_leq_z_l692_692241


namespace population_proof_l692_692838

noncomputable def Springfield_population : ℕ := 482653
noncomputable def Greenville_population : ℕ := Springfield_population - 119666
noncomputable def Oakville_population : ℕ := 
  let diff := Springfield_population - Greenville_population
  round ((diff: ℝ) * 1.25)

noncomputable def total_population : ℕ := Springfield_population + Greenville_population + Oakville_population

noncomputable def Greenville_households : ℕ := round ((Greenville_population: ℝ) / 3.5)
noncomputable def Oakville_households : ℕ := round ((Oakville_population: ℝ) / 4)

noncomputable def difference_households : ℕ := Greenville_households - Oakville_households

theorem population_proof :
  total_population = 995223 ∧ difference_households = 66314 := by
  sorry

end population_proof_l692_692838


namespace largest_prime_factor_of_1001_l692_692373

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692373


namespace centroid_traces_circle_area_l692_692638

noncomputable def point := ℝ × ℝ

def is_diameter (A B : point) (d : ℝ) : Prop :=
  dist A B = d

def lies_on_circle (C O : point) (r : ℝ) : Prop :=
  dist C O = r

theorem centroid_traces_circle_area
  (A B C O : point)
  (d r : ℝ)
  (hAB : is_diameter A B d)
  (h_diameter : d = 30)
  (hC_not_A : C ≠ A)
  (hC_not_B : C ≠ B)
  (hC_circle : lies_on_circle C O r)
  (h_radius : r = 15)
  (hO_centroid : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  : ∃ area : ℝ, area = 25 * Real.pi :=
sorry

end centroid_traces_circle_area_l692_692638


namespace binomial_expansion_problem_l692_692146

theorem binomial_expansion_problem :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ),
    (1 + 2 * x) ^ 11 =
      a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
      a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 +
      a_9 * x^9 + a_10 * x^10 + a_11 * x^11 →
    a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 + 5 * a_5 - 6 * a_6 +
    7 * a_7 - 8 * a_8 + 9 * a_9 - 10 * a_10 + 11 * a_11 = 22 :=
by
  -- The proof is omitted for this exercise
  sorry

end binomial_expansion_problem_l692_692146


namespace circumcircle_center_lies_on_circle_l692_692732

section circumcircle_center

variables {A B C D E F O : Type*}
variables [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O]

-- Conditions
-- 1. The trapezoid ABCD (AB || CD) is inscribed in a circle ω
axiom inscribed_trapezoid (A B C D : Type*) (ω : Type*) [is_circle ω] [is_trapezoid ABCD] : 
  inscribed_in ω ABCD

-- 2. Point E such that BC = BE and E is on the ray beyond C along DC
axiom E_on_ray (B C D E : Type*) : on_ray D C E ∧ BC = BE

-- 3. The line BE intersects the circle ω again at F, which lies outside the segment BE
axiom BE_intersects_again (B E ω F : Type*) [is_circle ω] [is_line B E] : intersects_again_in_circle B E ω F ∧ outside_segment B E F

-- Assertion to be proved
theorem circumcircle_center_lies_on_circle (A B C D E F O ω : Type*) [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O] [is_circle ω] : 
  center_of_circumcircle_lies_on_circle (triangle C E F) ω :=
by {
  assume inscribed_trapezoid ABCD ω,
  assume E_on_ray B C D E,
  assume BE_intersects_again B E ω F,
  exact sorry
}

end circumcircle_center

end circumcircle_center_lies_on_circle_l692_692732


namespace vector_c_coordinates_l692_692165

open Real EuclideanSpace

noncomputable def vector_a : ℝ × ℝ := (2, -3)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)
noncomputable def vector_diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)
noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ := (v.1 / magnitude v, v.2 / magnitude v)
noncomputable def vector_c : ℝ × ℝ := unit_vector vector_diff

theorem vector_c_coordinates :
  vector_c = (3/5 : ℝ, -4/5 : ℝ) :=
by {
  simp [vector_a, vector_b, vector_diff, magnitude],
  sorry
}

end vector_c_coordinates_l692_692165


namespace mirka_more_pears_l692_692990

noncomputable def pears_shared (n: ℕ): Prop :=
  ∃ (M I: ℕ), 
  I + M = n ∧
  I = 6 ∧
  M = 8 ∧
  (∀ σ: Type, ∃ ivan mirka : ℕ, 
    ivan = 2 * (fin.find (λ _, ivan)/6 + fin.find (λ _, ivan)/6 + 2) + 2 ∧
    mirka = 2 * (2 + 4) + 6
  )

theorem mirka_more_pears : ∀ (n: ℕ), pears_shared n → ∃ (diff: ℕ), diff = 2 :=
by
  intros n h
  cases h with M hM
  cases hM with I hI
  existsi 2
  rw [hI.left, hI.right.left, hI.right.right.left, hI.right.right.right]
  sorry

end mirka_more_pears_l692_692990


namespace correct_answer_l692_692705

-- Propositions conditions
def is_proposition_1 : Prop := ¬ (∃ x : ℝ, x^2 - 3 = 0)
def is_proposition_2 : Prop := ∀ l₁ l₂ : ℝ, False  -- questioning lines' parallelism is not a proposition
def is_proposition_3 : Prop := 3 + 1 = 5
def is_proposition_4 : Prop := ∀ x : ℝ, 5 * x - 3 > 6

theorem correct_answer:
  (is_proposition_3 ∧ is_proposition_4) → ("D") :=
by
  sorry

end correct_answer_l692_692705


namespace x_minus_y_values_l692_692892

theorem x_minus_y_values (x y : ℝ) 
  (h1 : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) : x - y = -1 ∨ x - y = -7 := 
  sorry

end x_minus_y_values_l692_692892


namespace rahul_share_l692_692272

theorem rahul_share (payment : ℚ) (rahul_days : ℚ) (rajesh_days : ℚ) (total_payment : ℚ) :
  payment = 355 → rahul_days = 3 → rajesh_days = 2 → total_payment = 355 → 
  (payment * (1/rahul_days) / ((1/rahul_days) + (1/rajesh_days))) = 142 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end rahul_share_l692_692272


namespace ratio_of_ages_l692_692402

theorem ratio_of_ages (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : a + b = 35) : a / gcd a b = 3 ∧ b / gcd a b = 4 :=
by
  sorry

end ratio_of_ages_l692_692402


namespace median_of_removed_extremes_l692_692804

theorem median_of_removed_extremes (x : Fin 10 → ℝ) (h : ∀ i j, i < j → x i < x j) :
  median ({x i | i ≠ 0 ∧ i ≠ 9} : Set ℝ) = median (Finset.erase (Finset.erase (Finset.univ : Finset (Fin 10)) 0) 9).val :=
by sorry

end median_of_removed_extremes_l692_692804


namespace average_height_students_count_l692_692966

-- Definitions based on the conditions
def total_students : ℕ := 400
def short_students : ℕ := (2 * total_students) / 5
def extremely_tall_students : ℕ := total_students / 10
def tall_students : ℕ := 90
def average_height_students : ℕ := total_students - (short_students + tall_students + extremely_tall_students)

-- Theorem to prove
theorem average_height_students_count : average_height_students = 110 :=
by
  -- This proof is omitted, we are only stating the theorem.
  sorry

end average_height_students_count_l692_692966


namespace expected_length_of_first_group_l692_692719

-- Define the conditions of the problem
def sequence : Finset ℕ := {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

-- Expected length of the first group
def expected_length_first_group : ℝ := 2.83

-- The formal statement of the proof problem
theorem expected_length_of_first_group (seq : Finset ℕ) (h1 : seq.card = 68) (h2 : seq.filter (λ x, x = 1) = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
(h3 : seq.filter (λ x, x = 0) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) :
  let X := 
  (Finset.sum (Finset.range 19) (λ k, if k = 1 then (1/50 : ℝ) else 0)) +
  (Finset.sum (Finset.range 49) (λ m, if m = 1 then (1/20 : ℝ) else 0)) in 
  ∃ x : ℝ, ∑ x = expected_length_first_group := 
by sorry
 
end expected_length_of_first_group_l692_692719


namespace find_m_in_geometric_seq_l692_692596

theorem find_m_in_geometric_seq (a : ℕ → ℝ) (m : ℕ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) * a (n - 1) = 2 * a n) →
  (∀ n, S n = ∑ i in finset.range n, real.logb 2 (a i)) →
  S (2 * m - 1) = 9 →
  (a m = 2) →
  m = 5 :=
by
  intros h1 h2 h3 h4
  -- Here is where the proof will go.
  sorry

end find_m_in_geometric_seq_l692_692596


namespace division_of_decimals_l692_692754

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l692_692754


namespace multiply_power_of_five_l692_692397

theorem multiply_power_of_five :
  112 * (5 ^ 4) = 70000 := by
  have h_power : 5 ^ 4 = 625 := by norm_num
  rw h_power
  norm_num

end multiply_power_of_five_l692_692397


namespace length_of_PS_l692_692971

theorem length_of_PS (QT TS PT TR PQ : ℝ)
  (hQT : QT = 5)
  (hTS : TS = 7)
  (hPT : PT = 9)
  (hTR : TR = 4)
  (hPQ : PQ = 7) :
  (19.8 < sqrt (PT^2 + TS^2 + 2 * PT * TS * (19 / 30)) ∧ sqrt (PT^2 + TS^2 + 2 * PT * TS * (19 / 30)) < 20) :=
by {
  sorry
}

end length_of_PS_l692_692971


namespace product_of_divisors_120_30_l692_692111

theorem product_of_divisors_120_30 :
  let divisors_120 : List ℤ := [±1, ±2, ±3, ±4, ±5, ±6, ±8, ±10, ±12, ±15, ±20, ±24, ±30, ±40, ±60, ±120]
  let divisors_30 : List ℤ := [±1, ±2, ±3, ±5, ±6, ±10, ±15, ±30]
  let common_divisors := [±1, ±2, ±3, ±5, ±6, ±10, ±15, ±30]
  List.prod common_divisors = 2^2 * 3^2 * 5^2 * 6^2 * 10^2 * 15^2 * 30^2 := 
by
  sorry

end product_of_divisors_120_30_l692_692111


namespace angle_bisector_perpendicular_l692_692977

-- Given conditions and definitions
variables {A B C D : Type} [InnerProductSpace ℝ D]
variable [ConvexQuadrilateral A B C D]
variable (angle_eq : ∠B = ∠C)
variable (right_angle_D : ∠D = 90)
variable (length_relation : |AB| = 2 * |CD|)

-- Statement to prove: The angle bisector of ∠ACB is perpendicular to CD.
theorem angle_bisector_perpendicular (h : ConvexQuadrilateral A B C D) :
  Bisector (∠ A C B) ⊥ Segment C D :=
  sorry

end angle_bisector_perpendicular_l692_692977


namespace elegant_numbers_equal_l692_692248

def is_elegant (n : ℕ) : Prop :=
  let digits : List ℕ := (List.unfoldr (λ n, if n = 0 then none else some (n % 10, n / 10)) n).reverse
  digits.Nodup ∧ (List.sorted (≤) digits)

def count_elegant_numbers (k : ℕ) : ℕ :=
  ((Finset.range 9).powerset.filter (λ s, s.card = k)).card

theorem elegant_numbers_equal :
  count_elegant_numbers 4 = count_elegant_numbers 5 := by
  sorry

end elegant_numbers_equal_l692_692248


namespace determine_ordered_pair_l692_692469

theorem determine_ordered_pair (s n : ℤ)
    (h1 : ∀ t : ℤ, ∃ x y : ℤ,
        (x, y) = (s + 2 * t, -3 + n * t)) 
    (h2 : ∀ x y : ℤ, y = 2 * x - 7) :
    (s, n) = (2, 4) :=
by
  sorry

end determine_ordered_pair_l692_692469


namespace problem1_problem2_l692_692027

-- Problem 1 statement
theorem problem1 : 
  ((-1 : ℝ) ^ 2023 + 2 * real.cos (real.pi / 4) - abs (real.sqrt 2 - 2) - (1 / 2)⁻¹ = 2 * real.sqrt 2 - 5) := 
  by 
  -- Placeholder for proof
  sorry

-- Problem 2 statement
theorem problem2 (x : ℝ) (h : x ≠ -1) : 
  ((1 - 1 / (x + 1)) / (x^2 / (x^2 + 2 * x + 1)) = (x + 1) / x) := 
  by 
  -- Placeholder for proof
  sorry

end problem1_problem2_l692_692027


namespace cosine_A_side_c_l692_692189

noncomputable section

variables {A B C : ℝ} -- angles
variables {a b c : ℝ} -- side lengths

-- Conditions
def is_triangle_ABC (a b : ℝ) (A B : ℝ) : Prop :=
  a = 3 ∧ b = 2 * Real.sqrt 6 ∧ B = 2 * A

-- Question I: Cosine of A
theorem cosine_A (A B : ℝ) (a b : ℝ) (h : is_triangle_ABC a b A) : 
  Real.cos A = Real.sqrt 6 / 3 :=
sorry

-- Question II: length of side c
theorem side_c (A B C : ℝ) (a b c : ℝ) (h : is_triangle_ABC a b A) (h_cos : Real.cos A = Real.sqrt 6 / 3) : 
  c = 5 :=
sorry

end cosine_A_side_c_l692_692189


namespace triangle_area_is_correct_l692_692841

noncomputable def radius_ω : ℝ := 5

def externally_tangent (ω1 ω2 : set (ℝ × ℝ)) : Prop := sorry
def ω1 : set (ℝ × ℝ) := sorry
def ω2 : set (ℝ × ℝ) := sorry
def ω3 : set (ℝ × ℝ) := sorry
def Q1 : ℝ × ℝ := sorry
def Q2 : ℝ × ℝ := sorry
def Q3 : ℝ × ℝ := sorry

axiom ext_tangent_ω : ∀ i j : {1, 2, 3}, i ≠ j → externally_tangent (ω_i) (ω_j)
axiom points_on_ω : ∀ i : {1, 2, 3}, Qi ∈ ωi
axiom equilateral_Q : Q1Q2 = Q2Q3 ∧ Q2Q3 = Q3Q1
axiom tangent_lines : ∀ i : {1, 2, 3}, tangent_line Qi Qi₊1

theorem triangle_area_is_correct : 
  area (triangle Q1 Q2 Q3) = (155 * Real.sqrt 3 + 150 * Real.sqrt 22) / 12 := 
sorry

end triangle_area_is_correct_l692_692841


namespace numGlobalCompletelySymmetricalDays_l692_692168

def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isGlobalCompletelySymmetricalDay (date : ℕ) : Prop :=
  let year := date / 10000
  let month_day := date % 10000
  isPalindrome date ∧
  year ≥ 2001 ∧ year ≤ 2099 ∧
  let month := month_day / 100
  month ≥ 1 ∧ month ≤ 12

theorem numGlobalCompletelySymmetricalDays : 
  ∃ n, n = 99 ∧ ∀ d, isGlobalCompletelySymmetricalDay d → n = (Finset.range 100).card (λ d, isGlobalCompletelySymmetricalDay (20010000 + d)) :=
sorry

end numGlobalCompletelySymmetricalDays_l692_692168


namespace vec_same_direction_l692_692543

theorem vec_same_direction (k : ℝ) : (k = 2) ↔ ∃ m : ℝ, m > 0 ∧ (k, 2) = (m * 1, m * 1) :=
by
  sorry

end vec_same_direction_l692_692543


namespace mutually_exclusive_and_not_contradictory_l692_692506

-- Definitions based on conditions provided in the problem
def bag : Type := fin 4
def red_balls : fin 4 → Prop := λ b, b.val < 2
def white_balls : fin 4 → Prop := λ b, b.val ≥ 2
def drawn_balls : finset (fin 4) := {0, 1, 2, 3} -- all balls

-- Definitions of the events as per the conditions
def event_A_1 : finset (finset (fin 4)) := { {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3} } -- At least 1 white ball
def event_A_2 : finset (finset (fin 4)) := { {2, 3} }                                           -- Both are white balls
def event_A : finset (finset (fin 4)) := event_A_1 ∩ event_A_2 -- Intersection, since it must satisfy both

def event_B_1 : finset (finset (fin 4)) := event_A_1
def event_B_2 : finset (finset (fin 4)) := { {0, 1, 2, 3} \ {2, 3} }
def event_B : finset (finset (fin 4)) := event_B_1 ∩ event_B_2

def event_C_1 : finset (finset (fin 4)) := event_A_1
def event_C_2 : finset (finset (fin 4)) := { {0, 1} }
def event_C : finset (finset (fin 4)) := event_C_1 ∩ event_C_2

def event_D_1 : finset (finset (fin 4)) := { {0, 2}, {0, 3}, {1, 2}, {1, 3} } -- Exactly 1 white ball
def event_D_2 : finset (finset (fin 4)) := { {2, 3} } -- Exactly 2 white balls
def event_D : finset (finset (fin 4)) := event_D_1 ∪ event_D_2

-- The statement we aim to prove
theorem mutually_exclusive_and_not_contradictory :
  (∀ (x ∈ event_D_1) (y ∈ event_D_2), x ≠ y) ∧ (∃ (z ∈ event_D_1 ∪ event_D_2), z ≠ ∅) :=
by
  sorry

end mutually_exclusive_and_not_contradictory_l692_692506


namespace aston_comics_l692_692013

theorem aston_comics (total_pages_on_floor : ℕ) (pages_per_comic : ℕ) (untorn_comics_in_box : ℕ) :
  total_pages_on_floor = 150 →
  pages_per_comic = 25 →
  untorn_comics_in_box = 5 →
  (total_pages_on_floor / pages_per_comic + untorn_comics_in_box) = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end aston_comics_l692_692013


namespace sin_B_in_arithmetic_sequence_triangle_l692_692520

theorem sin_B_in_arithmetic_sequence_triangle (A B C : ℝ) 
(h1 : A + B + C = 180) 
(h2 : A + C = 2 * B) :
  sin (B * (π / 180)) = sqrt 3 / 2 :=
sorry

end sin_B_in_arithmetic_sequence_triangle_l692_692520


namespace opposite_of_neg_2023_l692_692699

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l692_692699


namespace angle_HCF_l692_692619

-- Using necessary imports to bring in geometric constructs from Mathlib

open EuclideanGeometry

axiom exists_circle_inter (A B : Point) : ∃ (w : Circle), A ∈ w ∧ B ∈ w
axiom cuts_circle (D E : Point) (w : Circle) : ∃ (w' : Circle), D ∈ w' ∧ E ∈ w' ∧ w ∩ w' = {D, E}
axiom tangent_to_circle (C : Point) (w : Circle) : ∃ (w' : Circle), C ∈ w ∧ is_tangent(C, w')
axiom symmetric_point (F G : Point) : ∃ (H : Point), is_symmetric(F, G, H)

theorem angle_HCF (A B C D E F G H : Point) (w1 w2 w3 : Circle)
    (h1 : A ∈ w1 ∧ B ∈ w1)
    (h2 : C ∈ w2 ∧ is_tangent(C, w3))
    (h3 : D ∈ w3 ∧ E ∈ w3 ∧ D ≠ E)
    (h4 : F ∈ line_through A B ∧ is_tangent(F, line_through A B))
    (h5 : G ∈ line_through D E ∧ G ∈ line_through A B)
    (h6 : H = symmetric_point(F, G)) :
    inner_angle H C F = 90° :=
sorry

end angle_HCF_l692_692619


namespace reflection_symmetry_y_axis_l692_692592

theorem reflection_symmetry_y_axis (P : ℝ × ℝ) (h : P = (2, 1)) : 
  let P' := (-(P.1), P.2) in P' = (-2, 1) :=
by
  sorry

end reflection_symmetry_y_axis_l692_692592


namespace find_a5_l692_692515

theorem find_a5 (a : ℕ → ℤ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1) 
  (h2 : a 2 + a 4 + a 6 = 18) : 
  a 5 = 5 :=
sorry

end find_a5_l692_692515


namespace purpose_of_LB_full_nutrient_medium_l692_692706

/--
Given the experiment "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source",
which involves both experimental and control groups with the following conditions:
- The variable in the experiment is the difference in the medium used.
- The experimental group uses a medium with urea as the only nitrogen source (selective medium).
- The control group uses a full-nutrient medium.

Prove that the purpose of preparing LB full-nutrient medium is to observe the types and numbers
of soil microorganisms that can grow under full-nutrient conditions.
-/
theorem purpose_of_LB_full_nutrient_medium
  (experiment: String) (experimental_variable: String) (experimental_group: String) (control_group: String)
  (H1: experiment = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (H2: experimental_variable = "medium")
  (H3: experimental_group = "medium with urea as the only nitrogen source (selective medium)")
  (H4: control_group = "full-nutrient medium") :
  purpose_of_preparing_LB_full_nutrient_medium = "observe the types and numbers of soil microorganisms that can grow under full-nutrient conditions" :=
sorry

end purpose_of_LB_full_nutrient_medium_l692_692706


namespace solve_system_l692_692670

noncomputable def system_solutions := 
(x^(3:ℕ) + y^(3:ℕ) = 3 * y + 3 * z + 4) ∧ 
(y^(3:ℕ) + z^(3:ℕ) = 3 * z + 3 * x + 4) ∧ 
(z^(3:ℕ) + x^(3:ℕ) = 3 * x + 3 * y + 4)

theorem solve_system (x y z : ℝ) : 
  system_solutions x y z → 
  (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 2 ∧ y = 2 ∧ z = 2) :=
by
  sorry

end solve_system_l692_692670


namespace max_f_is_sqrt_five_l692_692048

def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin x

theorem max_f_is_sqrt_five : 
  ∃ x : ℝ, f x = sqrt 5 ∧ (∀ y : ℝ, f y ≤ sqrt 5) :=
begin
  sorry
end

end max_f_is_sqrt_five_l692_692048


namespace hexagon_area_l692_692226

def regular_hexagon (ABCDEF : Type) := sorry

structure Midpoints (
  G H I : Type
) where
  G_AB : G
  H_CD : H
  I_EF : I

def triangle_GHI_area (G H I : Type) := 100

theorem hexagon_area
  (ABCDEF : Type) 
  (h1 : regular_hexagon ABCDEF)
  (G H I : Type) 
  (h2 : Midpoints G H I)
  (h3 : triangle_GHI_area G H I = 100) :
  ∃ (area : ℝ), area = 2400 / 9 := 
sorry

end hexagon_area_l692_692226


namespace max_volume_of_inscribed_right_cylinder_l692_692006

theorem max_volume_of_inscribed_right_cylinder (R : ℝ) : 
  ∃ V, V = (8 * R^3 * real.sqrt 3) / 27 :=
by
  sorry

end max_volume_of_inscribed_right_cylinder_l692_692006


namespace jennie_total_rental_cost_l692_692683

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end jennie_total_rental_cost_l692_692683


namespace largest_prime_factor_1001_l692_692342

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692342


namespace xy_sum_square_l692_692490

theorem xy_sum_square (x y : ℕ) (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end xy_sum_square_l692_692490


namespace circumcircle_center_lies_on_circle_l692_692730

section circumcircle_center

variables {A B C D E F O : Type*}
variables [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O]

-- Conditions
-- 1. The trapezoid ABCD (AB || CD) is inscribed in a circle ω
axiom inscribed_trapezoid (A B C D : Type*) (ω : Type*) [is_circle ω] [is_trapezoid ABCD] : 
  inscribed_in ω ABCD

-- 2. Point E such that BC = BE and E is on the ray beyond C along DC
axiom E_on_ray (B C D E : Type*) : on_ray D C E ∧ BC = BE

-- 3. The line BE intersects the circle ω again at F, which lies outside the segment BE
axiom BE_intersects_again (B E ω F : Type*) [is_circle ω] [is_line B E] : intersects_again_in_circle B E ω F ∧ outside_segment B E F

-- Assertion to be proved
theorem circumcircle_center_lies_on_circle (A B C D E F O ω : Type*) [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O] [is_circle ω] : 
  center_of_circumcircle_lies_on_circle (triangle C E F) ω :=
by {
  assume inscribed_trapezoid ABCD ω,
  assume E_on_ray B C D E,
  assume BE_intersects_again B E ω F,
  exact sorry
}

end circumcircle_center

end circumcircle_center_lies_on_circle_l692_692730


namespace largest_prime_factor_of_1001_l692_692353

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692353


namespace complex_power_is_one_l692_692468

noncomputable def compute_complex_power : ℂ :=
  (complex.cos (210 * real.pi / 180) + complex.sin (210 * real.pi / 180) * complex.I)^60

-- Using De Moivre's Theorem and trigonometric identities, we state:
theorem complex_power_is_one :
  compute_complex_power = 1 :=
by
  sorry

end complex_power_is_one_l692_692468


namespace jeremy_remaining_money_l692_692218

-- Conditions as definitions
def computer_cost : ℝ := 3000
def accessories_cost : ℝ := 0.1 * computer_cost
def initial_money : ℝ := 2 * computer_cost

-- Theorem statement for the proof problem
theorem jeremy_remaining_money : initial_money - computer_cost - accessories_cost = 2700 := by
  -- Proof will be added here
  sorry

end jeremy_remaining_money_l692_692218


namespace pairs_satisfying_equation_l692_692933

theorem pairs_satisfying_equation : 
  {p : Nat × Nat // let x := p.1; let y := p.2 in x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77}.toList.length = 2 :=
by
  sorry

end pairs_satisfying_equation_l692_692933


namespace range_of_a_minus_b_l692_692573

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) : -1 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l692_692573


namespace bucket_volume_correct_l692_692291

-- conditions
def diameter := 4 -- in decimeters
def height_of_water := 4 -- in decimeters
def water_percentage := 0.4

-- derived values
def radius := diameter / 2
def volume_of_water := Float.pi * radius^2 * height_of_water

-- target volume of the bucket
def target_volume := 125.6 -- in liters

-- statement to prove
theorem bucket_volume_correct : volume_of_water / water_percentage = target_volume := 
sorry

end bucket_volume_correct_l692_692291


namespace cory_fruit_arrangements_l692_692472

theorem cory_fruit_arrangements : 
  let apples := 4 
  let oranges := 2 
  let banana := 1 
  let total_fruits := apples + oranges + banana
  total_fruits = 7 ∧
  (λ arrangements, (arrangements = (total_fruits - 1)! / (apples! * oranges!) * 6)) 90
: sorry

end cory_fruit_arrangements_l692_692472


namespace complement_is_correct_l692_692551

def U := {1, 2, 3, 4, 5}
def A := {2, 4}
def complement (U : Set ℕ) (A : Set ℕ) := U \ A

theorem complement_is_correct :
  complement U A = {1, 3, 5} :=
sorry

end complement_is_correct_l692_692551


namespace train_speed_l692_692442

theorem train_speed (length time : ℝ) (h_length : length = 300) (h_time : time = 27) : 
    (length / time) * 3.6 = 40 :=
by
  -- Conditions given
  have h1 : length = 300 := h_length
  have h2 : time = 27 := h_time
  -- Formula for speed
  have speed_m_s := length / time -- in meters per second
  have speed_km_h := speed_m_s * 3.6 -- conversion to km/h
  -- Conclusion
  show (300 / 27) * 3.6 = 40 from sorry

end train_speed_l692_692442


namespace area_of_triangle_AOB_is_three_l692_692980

-- Define the polar coordinates for points A and B
def polar_coord_A : ℝ × ℝ := (3, Real.pi / 3)
def polar_coord_B : ℝ × ℝ := (-4, 7 * Real.pi / 6)

-- Define the function for the area of a triangle formed by origin O and points A, B in polar coordinates
noncomputable def area_of_triangle_polar (A B : ℝ × ℝ) : ℝ :=
  let r1 := A.1
  let θ1 := A.2
  let r2 := B.1
  let θ2 := B.2
  1 / 2 * r1 * r2 * Real.sin (θ2 - θ1)

-- Prove that the area of triangle AOB is 3
theorem area_of_triangle_AOB_is_three : area_of_triangle_polar polar_coord_A polar_coord_B = 3 := by
  sorry

end area_of_triangle_AOB_is_three_l692_692980


namespace dad_strawberries_weight_l692_692643

-- Definitions for the problem
def weight_marco := 15
def total_weight := 37

-- Theorem statement
theorem dad_strawberries_weight :
  (total_weight - weight_marco = 22) :=
by
  sorry

end dad_strawberries_weight_l692_692643


namespace num_pairs_of_integers_satisfying_equation_l692_692937

theorem num_pairs_of_integers_satisfying_equation : 
  {p : ℕ × ℕ // p.1 > 0 ∧ p.2 > 0 ∧ (p.1^2 - p.2^2 = 77)}.to_list.length = 2 := 
sorry

end num_pairs_of_integers_satisfying_equation_l692_692937


namespace angle_EFG_is_60_degrees_l692_692747

theorem angle_EFG_is_60_degrees :
  ∀ (D E F G O : Type) (DE DF DG : D → O) (OE OF OG : O → E) 
  (arc_EF arc_FG arc_GE : ℝ),
  (OE = ⟂ (DE)) →
  (OF = ⟂ (DF)) →
  (OG = ⟂ (DG)) →
  (arc_EF = 3 * 30) →
  (arc_FG = 4 * 30) →
  (arc_GE = 5 * 30) →
  (arc_EF + arc_FG + arc_GE = 360) →
  angle (E F G) = 60 :=
by
  sorry

end angle_EFG_is_60_degrees_l692_692747


namespace triangle_area_is_six_l692_692323

noncomputable theory
open_locale classical

def point := ℝ × ℝ

def line_slope (slope : ℝ) (pt : point) : ℝ → ℝ := λ x, slope * (x - pt.1) + pt.2

-- Line definitions from conditions
def line1 := line_slope (1/2) (2, 2)
def line2 := line_slope 2 (2, 2)
def line3 (x: ℝ) := 10 - x

-- Intersection points
def intersection (f g : ℝ → ℝ) (x: ℝ) : x ∈ ℝ ∧ f x = g x

def pointA : point := (2, 2)
def pointB : point := (4, 6)
def pointC : point := (6, 4)

-- Triangle area using vertices (A = (2, 2), B = (4, 6), C = (6, 4))
def triangle_area (A B C: point) :=
  (1/2 : ℝ) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_six : 
  triangle_area pointA pointB pointC = 6 :=
by sorry

end triangle_area_is_six_l692_692323


namespace largest_prime_factor_1001_l692_692356

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692356


namespace cubic_equation_solution_bound_l692_692871

theorem cubic_equation_solution_bound (a : ℝ) :
  a ∈ Set.Ici (-15) → ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ → x₂ ≠ x₃ → x₁ ≠ x₃ →
  (x₁^3 + 6 * x₁^2 + a * x₁ + 8 = 0) →
  (x₂^3 + 6 * x₂^2 + a * x₂ + 8 = 0) →
  (x₃^3 + 6 * x₃^2 + a * x₃ + 8 = 0) →
  False := 
sorry

end cubic_equation_solution_bound_l692_692871


namespace product_common_divisors_l692_692108

theorem product_common_divisors (h120 : ∀ d ∈ {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10, 12, -12, 15, -15, 20, -20, 24, -24, 30, -30, 40, -40, 60, -60, 120, -120},
                                 d ∣ 120) 
                                (h30 : ∀ d ∈ {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30},
                                 d ∣ 30) :
  (∏ d in {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30}.to_finset, d) = 729000000 := by
  -- mathematical proof here
  sorry

end product_common_divisors_l692_692108


namespace range_of_a_l692_692574

theorem range_of_a 
  (a x y : ℝ)
  (h1 : 2 * x + y = 3 - a)
  (h2 : x + 2 * y = 4 + 2 * a)
  (h3 : x + y < 1) :
  a < -4 := sorry

end range_of_a_l692_692574


namespace fraction_div_addition_l692_692095

theorem fraction_div_addition : ( (3 / 7 : ℚ) / 4) + (1 / 28) = (1 / 7) :=
  sorry

end fraction_div_addition_l692_692095


namespace complex_number_quadrant_l692_692205

noncomputable def question : ℂ := (2 / (1 - complex.I))

theorem complex_number_quadrant :
  let z := question in 
  z = 1 + complex.I ∧ (1 : ℝ) > 0 ∧ (1 : ℝ) > 0 → ∃ (q : ℕ), q = 1 :=
by
  intro z
  have h1 : z = 1 + complex.I := sorry
  have h2 : (1 : ℝ) > 0 := by norm_num
  have h3 : (1 : ℝ) > 0 := by norm_num
  use 1
  exact sorry

end complex_number_quadrant_l692_692205


namespace find_vector_BC_l692_692143

theorem find_vector_BC (A B : ℝ × ℝ) (AC : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (3, 2)) (hAC : AC = (-4, -3)) :
  let C := (fst AC, snd AC + 1)
  ∃ C : ℝ × ℝ, C = (-4, -2) ∧ (fst (B - C), snd (B - C)) = (-7, -4) :=
by
  sorry

end find_vector_BC_l692_692143


namespace angle_sum_proof_l692_692594

theorem angle_sum_proof (x α β : ℝ) (h1 : 3 * x + 4 * x + α = 180)
 (h2 : α + 5 * x + β = 180)
 (h3 : 2 * x + 2 * x + 6 * x = 180) :
  x = 18 := by
  sorry

end angle_sum_proof_l692_692594


namespace coffee_shop_sales_l692_692826

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l692_692826


namespace servings_per_day_l692_692098

-- Conditions
def week_servings := 21
def days_per_week := 7

-- Question and Answer
theorem servings_per_day : week_servings / days_per_week = 3 := 
by
  sorry

end servings_per_day_l692_692098


namespace range_of_alpha_plus_beta_l692_692127

theorem range_of_alpha_plus_beta (α β : ℝ) (h1 : 0 < α - β) (h2 : α - β < π) (h3 : 0 < α + 2 * β) (h4 : α + 2 * β < π) :
  0 < α + β ∧ α + β < π :=
sorry

end range_of_alpha_plus_beta_l692_692127


namespace coaches_together_next_l692_692092

theorem coaches_together_next (a b c d : ℕ) (h_a : a = 5) (h_b : b = 9) (h_c : c = 8) (h_d : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 3960 :=
by 
  rw [h_a, h_b, h_c, h_d]
  sorry

end coaches_together_next_l692_692092


namespace petals_digits_l692_692615

noncomputable def unique_digits := {d : ℕ | d ∈ (finset.range 10)}

def sum_of_petals (a : finset ℕ) : ℕ :=
  ∑ i in a, i

def condition1 (a : finset ℕ) : Prop :=
  ∃ A B : finset ℕ, A ∪ B = a ∧ A ∩ B = ∅ ∧ A.card = 4 ∧ B.card = 4 ∧ sum_of_petals A = sum_of_petals B

def condition2 (a : finset ℕ) : Prop :=
  ∃ A B : finset ℕ, A ∪ B = a ∧ A ∩ B = ∅ ∧ A.card = 4 ∧ B.card = 4 ∧ sum_of_petals A = 2 * sum_of_petals B

def condition3 (a : finset ℕ) : Prop :=
  ∃ A B : finset ℕ, A ∪ B = a ∧ A ∩ B = ∅ ∧ A.card = 4 ∧ B.card = 4 ∧ sum_of_petals A = 4 * sum_of_petals B

theorem petals_digits : (unique_digits.to_finset = {0, 1, 2, 3, 4, 5, 6, 9}.to_finset ∨ 
                        unique_digits.to_finset = {0, 1, 2, 3, 4, 5, 7, 8}.to_finset) →
                        condition1 unique_digits.to_finset →
                        condition2 unique_digits.to_finset →
                        condition3 unique_digits.to_finset :=
begin
  sorry
end

end petals_digits_l692_692615


namespace f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l692_692891

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x / (1 + x) else 1 / (1 + x)

theorem f_property (x : ℝ) (hx : 0 < x) : 
  f x = f (1 / x) :=
by
  sorry

theorem f_equals_when_x_lt_1 (x : ℝ) (hx0 : 0 < x) (hx1 : x < 1) : 
  f x = 1 / (1 + x) :=
by
  sorry

theorem f_equals_when_x_gt_1 (x : ℝ) (hx : 1 < x) : 
  f x = x / (1 + x) :=
by
  sorry

end f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l692_692891


namespace term_containing_x_squared_is_third_l692_692128

variable {n : ℕ}

-- Given condition
axiom binomial_condition : choose n 6 = choose n 4

-- Statement to be proved
theorem term_containing_x_squared_is_third :
  (sqrt x - (1 / (3 * x)))^n = list.to_finset ((iterate (r → (- 1 / 3)^r * choose n r * x^(5 - (3 * r / 2))) 10).nth (2)).get :=
by
  sorry

end term_containing_x_squared_is_third_l692_692128


namespace sum_of_complex_numbers_l692_692563

variable B Q R T : ℂ

theorem sum_of_complex_numbers :
  B = 3 - 2 * complex.I →
  Q = 1 + 3 * complex.I →
  R = -2 + 4 * complex.I →
  T = 5 - 3 * complex.I →
  B + Q + R + T = 7 + 2 * complex.I :=
by
  intros hB hQ hR hT
  rw [hB, hQ, hR, hT]
  sorry

end sum_of_complex_numbers_l692_692563


namespace quadruple_pieces_count_l692_692649

theorem quadruple_pieces_count (earned_amount_per_person_in_dollars : ℕ) 
    (total_single_pieces : ℕ) (total_double_pieces : ℕ)
    (total_triple_pieces : ℕ) (single_piece_circles : ℕ) 
    (double_piece_circles : ℕ) (triple_piece_circles : ℕ)
    (quadruple_piece_circles : ℕ) (cents_per_dollar : ℕ) :
    earned_amount_per_person_in_dollars * 2 * cents_per_dollar -
    (total_single_pieces * single_piece_circles + 
    total_double_pieces * double_piece_circles + 
    total_triple_pieces * triple_piece_circles) = 
    165 * quadruple_piece_circles :=
        sorry

#eval quadruple_pieces_count 5 100 45 50 1 2 3 4 100

end quadruple_pieces_count_l692_692649


namespace complement_union_l692_692550

open Set

variable (U : Set ℕ := {0, 1, 2, 3, 4}) (A : Set ℕ := {1, 2, 3}) (B : Set ℕ := {2, 4})

theorem complement_union (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) : 
  (U \ A ∪ B) = {0, 2, 4} :=
by
  sorry

end complement_union_l692_692550


namespace functional_eq_one_l692_692496

theorem functional_eq_one (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
    (h2 : ∀ x > 0, ∀ y > 0, f x * f (y * f x) = f (x + y)) :
    ∀ x, 0 < x → f x = 1 := 
by
  sorry

end functional_eq_one_l692_692496


namespace problem_I_problem_II_l692_692906

def f (x a : ℝ) := |x + 2| - |2 * x - a|

theorem problem_I (x : ℝ) : f x 3 > 0 ↔ (1 / 3 < x ∧ x < 5) := 
sorry

theorem problem_II (a : ℝ) : (∀ x, x ∈ set.Ici 0 → f x a < 3) ↔ a < 2 :=
sorry

end problem_I_problem_II_l692_692906


namespace pairs_satisfying_equation_l692_692930

theorem pairs_satisfying_equation : 
  {p : Nat × Nat // let x := p.1; let y := p.2 in x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77}.toList.length = 2 :=
by
  sorry

end pairs_satisfying_equation_l692_692930


namespace combined_cost_price_approx_l692_692258

noncomputable def cost_price (sp : ℝ) (profit_percent : ℝ) : ℝ := sp / (1 + profit_percent / 100)

theorem combined_cost_price_approx :
  let sp1 := 100
  let sp2 := 150
  let sp3 := 200
  let profit1 := 40
  let profit2 := 30
  let profit3 := 20
  let cp1 := cost_price sp1 profit1
  let cp2 := cost_price sp2 profit2
  let cp3 := cost_price sp3 profit3
  combined_cp := cp1 + cp2 + cp3
  abs (combined_cp - 353.48) < 0.1 :=
by
  sorry

end combined_cost_price_approx_l692_692258


namespace ten_digit_of_factorial_expression_l692_692760

theorem ten_digit_of_factorial_expression : 
  let k := 6840 in
  let expr := 5! * (5! - 3!) / k in
  expr = 2 →
  (10 ^ 1 % 13680) / 10 % 10 = 8 := 
by
  sorry

end ten_digit_of_factorial_expression_l692_692760


namespace relation_among_a_b_c_l692_692875

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relation_among_a_b_c : c > a ∧ a > b :=
by
  -- Prove that c > a and a > b
  sorry

end relation_among_a_b_c_l692_692875


namespace max_distance_ellipse_to_line_l692_692888

theorem max_distance_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
  let line (x y : ℝ) := x + y - 7 = 0
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
    (∀ (Q : ℝ × ℝ), ellipse Q.1 Q.2 → dist_to_line Q.1 Q.2 ≤ 6 * Real.sqrt 2)
    ∧ dist_to_line P.1 P.2 = 6 * Real.sqrt 2 
:= sorry

noncomputable def dist_to_line (x y : ℝ) : ℝ :=
  (|x + y - 7|) / Real.sqrt (1^2 + 1^2)

end max_distance_ellipse_to_line_l692_692888


namespace circumcenter_CEF_lies_on_ω_l692_692734

-- Definitions for given conditions
variables {A B C D E F O : Point}
variables {ω : Circle}
variables (H1: Trapezoid ABCD) (H2: Inscribed ABCD ω)
variables (H3: PointOnRayBeyond E C D) (H4: BC = BE)
variables (H5: IntersectsAt BE ω F) (H6: Isosceles BEC)

-- The theorem statement
theorem circumcenter_CEF_lies_on_ω :
  CenterOfCircumcircle C E F = O ∧ PointOnCircle O ω :=
sorry

end circumcenter_CEF_lies_on_ω_l692_692734


namespace range_of_function_l692_692481

theorem range_of_function :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -2 ≤ (Real.sin x - 1) / (Real.sin x + 2) ∧ (Real.sin x - 1) / (Real.sin x + 2) ≤ 0) :=
by
  intro x hx
  have h1 := hx.1
  have h2 := hx.2
  have : 1 ≤ Real.sin x + 2 ∧ Real.sin x + 2 ≤ 3 := by
    split;
    linarith
  sorry

end range_of_function_l692_692481


namespace integer_solution_interval_l692_692907

theorem integer_solution_interval {f : ℝ → ℝ} (m : ℝ) :
  (∀ x : ℤ, (-x^2 + x + m + 2 ≥ |x| ↔ (x : ℝ) = n)) ↔ (-2 ≤ m ∧ m < -1) := 
sorry

end integer_solution_interval_l692_692907


namespace largest_prime_factor_1001_l692_692369

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692369


namespace total_amount_shared_l692_692766

theorem total_amount_shared (a b c : ℝ)
  (h1 : a = 1/3 * (b + c))
  (h2 : b = 2/7 * (a + c))
  (h3 : a = b + 20) : 
  a + b + c = 720 :=
by
  sorry

end total_amount_shared_l692_692766


namespace trigonometric_identity_proof_l692_692277

noncomputable theory

open Real

theorem trigonometric_identity_proof : 
  sin (15 * (π / 180)) * cos (75 * (π / 180)) + cos (15 * (π / 180)) * sin (105 * (π / 180)) = 1 :=
by
  sorry

end trigonometric_identity_proof_l692_692277


namespace saltwater_concentration_l692_692412

def initial_concentration := 0.16

def volume_ratio_large := 10
def volume_ratio_medium := 4
def volume_ratio_small := 3

def overflow_ratio := 0.1

def final_concentration := 0.107

theorem saltwater_concentration
  (initial_concentration = 0.16)
  (volume_ratio_large = 10)
  (volume_ratio_medium = 4)
  (volume_ratio_small = 3)
  (overflow_ratio = 0.1) :
  final_concentration = 0.107 :=
sorry

end saltwater_concentration_l692_692412


namespace calculate_exponent_product_l692_692026

theorem calculate_exponent_product : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end calculate_exponent_product_l692_692026


namespace prove_range_l692_692708

noncomputable def valid_range (f : ℝ → ℝ) :=
  {x : ℝ | f x ≠ 0}

def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem prove_range : valid_range f = {x : ℝ | x ≠ -2} :=
by
  sorry

end prove_range_l692_692708


namespace term_3001_in_sequence_l692_692845

theorem term_3001_in_sequence :
  ∃ (xs : Finset ℝ), 
  (∀ (x ∈ xs), ∃ (a : ℕ → ℝ), 
   a 0 = x ∧ a 1 = 3000 ∧ 
   (∀ n, a (n+2) = (a (n+1) + 1) / a n) ∧ 
   ∃ m, a m = 3001) ∧ xs.card = 4 :=
by {
  sorry
}

end term_3001_in_sequence_l692_692845


namespace find_pre_tax_remuneration_l692_692813

def pre_tax_remuneration (x : ℝ) : Prop :=
  let taxable_amount := if x <= 4000 then x - 800 else x * 0.8
  let tax_due := taxable_amount * 0.2
  let final_tax := tax_due * 0.7
  final_tax = 280

theorem find_pre_tax_remuneration : ∃ x : ℝ, pre_tax_remuneration x ∧ x = 2800 := by
  sorry

end find_pre_tax_remuneration_l692_692813


namespace range_of_real_a_l692_692853

theorem range_of_real_a (a : ℝ) :
  (∃ x : ℝ, exp(2 * x) + a * exp(x) + 1 = 0) ↔ a ∈ set.Iic (-2) := by
  sorry

end range_of_real_a_l692_692853


namespace find_m_plus_n_l692_692087

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l692_692087


namespace number_of_plans_correct_l692_692031

noncomputable def number_of_assignment_plans (females_males_to_positions : ℕ) (females : ℕ) (males : ℕ) 
(conditions_c10 : ℕ → ℕ) (conditions_c40 : ℕ → ℕ) (conditions_a3 : ℕ → ℕ) 
(conditions_a4 : ℕ → ℕ)  :=
  conditions_c10 2 * conditions_a3 1 * conditions_a4 4 * conditions_c40 3

theorem number_of_plans_correct (n : ℕ) : 
  (number_of_assignment_plans n 10 40 
  (λ k, nat.choose 10 k) 
  (λ k, nat.choose 40 k) 
  (λ k, nat.perm 3 k) 
  (λ k, nat.perm 4 k)) 
  = nat.choose 10 2 * nat.perm 3 1 * nat.perm 4 4 * nat.choose 40 3 :=
sorry

end number_of_plans_correct_l692_692031


namespace probability_age_21_to_30_l692_692194

theorem probability_age_21_to_30 : 
  let total_people := 160 
  let people_10_to_20 := 40
  let people_21_to_30 := 70
  let people_31_to_40 := 30
  let people_41_to_50 := 20
  (people_21_to_30 / total_people : ℚ) = 7 / 16 := by
  sorry

end probability_age_21_to_30_l692_692194


namespace find_alpha_l692_692902

def f (x : ℝ) : ℝ := cos x * (sin x + sqrt 3 * cos x) - sqrt 3 / 2

def g (x α : ℝ) : ℝ := f (x + α)

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h x = -h (-x)

noncomputable def k_val (k : ℕ) : ℝ := k * π / 2 - π / 6

noncomputable def α (k : ℕ) : ℝ := k_val k

theorem find_alpha (k : ℕ) (hk : 0 < k) : 
  is_odd (g α) → α = k_val k :=
sorry

end find_alpha_l692_692902


namespace opposite_pairs_l692_692451

theorem opposite_pairs :
  ∃ (x y : ℤ), (x = -5 ∧ y = -(-5)) ∧ (x = -y) ∧ (
    (¬ (∃ (a b : ℤ), (a = -2 ∧ b = 1/2) ∧ (a = -b))) ∧ 
    (¬ (∃ (c d : ℤ), (c = | -1 | ∧ d = 1) ∧ (c = -d))) ∧
    (¬ (∃ (e f : ℤ), (e = (-3)^2 ∧ f = 3^2) ∧ (e = -f)))
  ) :=
by
  sorry

end opposite_pairs_l692_692451


namespace diff_set_Q_minus_P_l692_692235

def P (x : ℝ) : Prop := 1 - (2 / x) < 0
def Q (x : ℝ) : Prop := |x - 2| < 1
def diff_set (P Q : ℝ → Prop) (x : ℝ) : Prop := Q x ∧ ¬ P x

theorem diff_set_Q_minus_P :
  ∀ x : ℝ, diff_set Q P x ↔ (2 ≤ x ∧ x < 3) :=
by
  sorry

end diff_set_Q_minus_P_l692_692235


namespace chess_tournament_compatriots_l692_692965

theorem chess_tournament_compatriots 
    (participants : Fin 10 → Type) 
    (game : ∀ p q : Fin 10, p ≠ q → bool)
    (compatriots : ∀ p q : Fin 10, Prop)
    (H1 : ∀ p q, p ≠ q → game p q = true ∨ game p q = false)
    (H2 : ∀ p, ∃ q1 q2 q3 q4 q5 q6 q7 q8 q9, 
        p ≠ q1 ∧ p ≠ q2 ∧ p ≠ q3 ∧ p ≠ q4 ∧ p ≠ q5 ∧ p ≠ q6 ∧ p ≠ q7 ∧ p ≠ q8 ∧ p ≠ q9 ∧
        game p q1 = true ∧ game p q2 = true ∧ game p q3 = true ∧ game p q4 = true ∧ game p q5 = true ∧ 
        game p q6 = true ∧ game p q7 = true ∧ game p q8 = true ∧ game p q9 = true)
    (H3 : ∀ p, ∃ c1 c2 c3 c4 c5, 
        compatriots p c1 ∧ compatriots p c2 ∧ compatriots p c3 ∧ compatriots p c4 ∧ compatriots p c5)
    (H4 : ∀ r : Fin 10, ∀ p q, r ≠ p ∧ r ≠ q → ∃ r1 r2 r3 r4 r5,
        ∀ prq, (prq = (r, r1) ∨ prq = (r, r2) ∨ prq = (r, r3) ∨ prq = (r, r4) ∨ prq = (r, r5)) → 
                ∃ s1, game prq.1 s1 = true):
    ∀ round : Fin 10, ∃ (pr : Fin 10 × Fin 10), game pr.1 pr.2 = true ∧ compatriots pr.1 pr.2 := 
sorry

end chess_tournament_compatriots_l692_692965


namespace seq_contains_1_or_3_seq_contains_3_if_divisible_by_3_seq_contains_1_if_not_divisible_by_3_l692_692776

noncomputable def f (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 else n + 3

noncomputable def a_seq (m : ℕ) : ℕ → ℕ
| 0       => m
| (n + 1) => f (a_seq n)

theorem seq_contains_1_or_3 (m : ℕ) : ∃ n, a_seq m n = 1 ∨ a_seq m n = 3 :=
sorry

theorem seq_contains_3_if_divisible_by_3 (m : ℕ) (h : m % 3 = 0) : ∃ n, a_seq m n = 3 :=
sorry

theorem seq_contains_1_if_not_divisible_by_3 (m : ℕ) (h : m % 3 ≠ 0) : ∃ n, a_seq m n = 1 :=
sorry

end seq_contains_1_or_3_seq_contains_3_if_divisible_by_3_seq_contains_1_if_not_divisible_by_3_l692_692776


namespace relation_correct_l692_692245

def M := {x : ℝ | x < 2}
def N := {x : ℝ | 0 < x ∧ x < 1}
def CR (S : Set ℝ) := {x : ℝ | x ∈ (Set.univ : Set ℝ) \ S}

theorem relation_correct : M ∪ CR N = (Set.univ : Set ℝ) :=
by sorry

end relation_correct_l692_692245


namespace remainder_div_x_y_l692_692763

theorem remainder_div_x_y :
  ∀ (x y : ℝ),
  x / y = 96.16 →
  y = 50.000000000001066 →
  x - y * 96 ≈ 8.00000000041 :=
by {
  intros x y h1 h2,
  sorry
}

end remainder_div_x_y_l692_692763


namespace imaginary_part_of_conjugate_l692_692301

theorem imaginary_part_of_conjugate (z : ℂ) (h : z = (4 - I) / (1 + I)) : z.conj.im = 5 / 2 := by
  sorry

end imaginary_part_of_conjugate_l692_692301


namespace cosine_interior_angle_cube_l692_692011

open Real

-- Definitions based on the conditions
def midpoint (A B : (ℝ × ℝ × ℝ)) : (ℝ × ℝ × ℝ) := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def cube_midpoints (A1 B1 B : (ℝ × ℝ × ℝ)) :=
  let M := midpoint A1 B1 in
  let N := midpoint B B1 in
  (M, N)

noncomputable def cosine_interior_angle_at_D (D M N : (ℝ × ℝ × ℝ)) : ℝ :=
  let DE := dist D M
  let DF := dist D N
  let EF := dist M N
  (DE^2 + DF^2 - EF^2) / (2 * DE * DF)

-- The theorem to be proven based on the equivalent proof problem
theorem cosine_interior_angle_cube :
  ∀ (A B C D A1 B1 C1 D1 : (ℝ × ℝ × ℝ)),
  let (M, N) := cube_midpoints A1 B1 B in
  dist A B = 3 → dist B C = 3 → dist C D = 3 → dist D A = 3 →
  dist A1 B1 = 3 → dist B1 C1 = 3 → dist C1 D1 = 3 → dist D1 A1 = 3 →
  dist A A1 = 3 → dist B B1 = 3 → dist C C1 = 3 → dist D D1 = 3 →
  cosine_interior_angle_at_D D M N = 4 / 13 := sorry

end cosine_interior_angle_cube_l692_692011


namespace count_blocks_differ_3_l692_692435

-- Definitions from the problem conditions
def materials := ["plastic", "wood", "metal"]
def sizes := ["small", "medium", "large"]
def colors := ["blue", "green", "red", "yellow"]
def shapes := ["circle", "hexagon", "square", "triangle"]

-- Target block
def target_block := ("plastic", "medium", "red", "circle")

-- Total number of distinct blocks
def total_blocks : ℕ := 144

-- Function to count the number of ways two blocks can differ by n attributes
def count_differ_by_n (target_block : (String, String, String, String)) (n : ℕ) : ℕ :=
  if target_block = ("plastic", "medium", "red", "circle") ∧ n = 3 then 68 else 0

-- Theorem statement
theorem count_blocks_differ_3 (target_block = ("plastic", "medium", "red", "circle")) :
  count_differ_by_n target_block 3 = 68 :=
begin
  sorry,
end

end count_blocks_differ_3_l692_692435


namespace probability_at_least_half_correct_l692_692672

theorem probability_at_least_half_correct :
  (∃ n : ℕ, 15 > n ∧ 0 ≤ n ∧ n + n = 15) →
  (∀ i : ℕ, i ∈ (finset.range 15) → ∃ p, p = 1/2 ∧ 0 ≤ p ∧ p ≤ 1) →
  ∑ k in (finset.range (15 + 1)), if (k ≥ 8) then (nat.choose 15 k) * (1/2)^k * (1/2)^(15 - k) else 0 = 1/2 :=
  by
    intro h1 h2
    sorry

end probability_at_least_half_correct_l692_692672


namespace smallest_odd_m_satisfying_inequality_l692_692116

theorem smallest_odd_m_satisfying_inequality : ∃ m : ℤ, m^2 - 11 * m + 24 ≥ 0 ∧ (m % 2 = 1) ∧ ∀ n : ℤ, n^2 - 11 * n + 24 ≥ 0 ∧ (n % 2 = 1) → m ≤ n → m = 3 :=
by
  sorry

end smallest_odd_m_satisfying_inequality_l692_692116


namespace coeff_a2_in_expansion_of_sixth_power_l692_692655

theorem coeff_a2_in_expansion_of_sixth_power :
  let expansion_1 := (1 + x + x^2)
  let expansion_2 := (1 + x + x^2)^2
  let expansion_3 := (1 + x + x^2)^3
  let expansion_4 := (1 + x + x^2)^4
  let expansion_6 := (1 + x + x^2)^6
  coefficient expansion_6 2 = 21 :=
by sorry

end coeff_a2_in_expansion_of_sixth_power_l692_692655


namespace find_m_plus_n_l692_692084

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l692_692084


namespace find_m_l692_692532

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ :=
  sin (x + π / 2) + cos (x - π / 2) + m

-- The main theorem stating the problem.
theorem find_m (m : ℝ) (h : ∃ x : ℝ, ∀ y : ℝ, f y m ≤ f x m ∧ f x m = 2 * real.sqrt 2) : 
  m = real.sqrt 2 :=
sorry

end find_m_l692_692532


namespace circumcenter_CEF_lies_on_ω_l692_692735

-- Definitions for given conditions
variables {A B C D E F O : Point}
variables {ω : Circle}
variables (H1: Trapezoid ABCD) (H2: Inscribed ABCD ω)
variables (H3: PointOnRayBeyond E C D) (H4: BC = BE)
variables (H5: IntersectsAt BE ω F) (H6: Isosceles BEC)

-- The theorem statement
theorem circumcenter_CEF_lies_on_ω :
  CenterOfCircumcircle C E F = O ∧ PointOnCircle O ω :=
sorry

end circumcenter_CEF_lies_on_ω_l692_692735


namespace function_property_l692_692920

variable (g : ℝ × ℝ → ℝ)
variable (cond : ∀ x y : ℝ, g (x, y) = - g (y, x))

theorem function_property (x : ℝ) : g (x, x) = 0 :=
by
  sorry

end function_property_l692_692920


namespace largest_prime_factor_of_1001_l692_692350

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692350


namespace largest_prime_factor_of_1001_l692_692354

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692354


namespace minimize_polygon_area_l692_692122

theorem minimize_polygon_area (n : ℕ) (h : n ≥ 3) :
  ∃ λ ∈ Icc (0:ℝ) 1, λ = 1/2 ∧ ∀ μ ∈ Icc (0:ℝ) 1, 
  -λ^2 + λ ≤ -μ^2 + μ :=
by
  sorry

end minimize_polygon_area_l692_692122


namespace neznaika_mistake_l692_692259

-- Let's define the conditions
variables {X A Y M E O U : ℕ} -- Represents distinct digits

-- Ascending order of the numbers
variables (XA AY AX OY EM EY MU : ℕ)
  (h1 : XA < AY)
  (h2 : AY < AX)
  (h3 : AX < OY)
  (h4 : OY < EM)
  (h5 : EM < EY)
  (h6 : EY < MU)

-- Identical digits replaced with the same letters
variables (h7 : XA = 10 * X + A)
  (h8 : AY = 10 * A + Y)
  (h9 : AX = 10 * A + X)
  (h10 : OY = 10 * O + Y)
  (h11 : EM = 10 * E + M)
  (h12 : EY = 10 * E + Y)
  (h13 : MU = 10 * M + U)

-- Each letter represents a different digit
variables (h_distinct : X ≠ A ∧ X ≠ Y ∧ X ≠ M ∧ X ≠ E ∧ X ≠ O ∧ X ≠ U ∧
                       A ≠ Y ∧ A ≠ M ∧ A ≠ E ∧ A ≠ O ∧ A ≠ U ∧
                       Y ≠ M ∧ Y ≠ E ∧ Y ≠ O ∧ Y ≠ U ∧
                       M ≠ E ∧ M ≠ O ∧ M ≠ U ∧
                       E ≠ O ∧ E ≠ U ∧
                       O ≠ U)

-- Prove Neznaika made a mistake
theorem neznaika_mistake : false :=
by
  -- Here we'll reach a contradiction, proving false.
  sorry

end neznaika_mistake_l692_692259


namespace factorize_poly1_factorize_poly2_l692_692488

variable (a b m n : ℝ)

theorem factorize_poly1 : 3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2 :=
sorry

theorem factorize_poly2 : 4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n) :=
sorry

end factorize_poly1_factorize_poly2_l692_692488


namespace triangle_intersection_area_gt_half_l692_692199

-- Definitions and theorem
variable {ABC : Type} [acute_triangle ABC]
variable {AM : median ABC}
variable {BK : angle_bisector ABC}
variable {CH : altitude ABC}

theorem triangle_intersection_area_gt_half {Area_ABC : ℝ} (h1 : Area_ABC > 0) :
  ∃ Area : ℝ, Area > 0.499 * Area_ABC := 
sorry

end triangle_intersection_area_gt_half_l692_692199


namespace function_zero_probability_l692_692889

-- Definitions for the parameters and the function
def f (x a b : ℝ) : ℝ := x^3 + a * x - b
def a_values : List ℝ := [1, 2, 3, 4]
def b_values : List ℝ := [2, 4, 8, 12]

-- The main theorem to prove
theorem function_zero_probability :
  let possible_combinations := (a_values.product b_values).filter (λ (p : ℝ × ℝ), (f 1 p.1 p.2) * (f 2 p.1 p.2) ≤ 0)
  (possible_combinations.length.toℚ / (a_values.length * b_values.length).toℚ) = 11 / 16 :=
by
  -- Definitions to be used in the proof
  let possible_combinations := (a_values.product b_values).filter (λ (p : ℝ × ℝ), (f 1 p.1 p.2) * (f 2 p.1 p.2) ≤ 0)
  have : possible_combinations.length = 11 := sorry       -- This is the actual count calculation
  have : a_values.length * b_values.length = 16 := by simp -- Simplifying the lengths
  suffices : (11 : ℚ) / 16 = 11 / 16 by sorry             -- Proving the fraction is indeed 11/16
  sorry

end function_zero_probability_l692_692889


namespace isosceles_triangle_area_l692_692101

-- Define the sides of the isosceles triangle.
def a : ℝ := 9
def b : ℝ := 9
def c : ℝ := 14

-- Define the height using the Pythagorean theorem.
def h : ℝ := 4 * Real.sqrt 2

-- Define the area calculation function for a triangle.
def triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height

-- Define the specific problem statement.
theorem isosceles_triangle_area :
  triangle_area c h = 28 * Real.sqrt 2 := 
by 
  -- Proof omitted.
  sorry

end isosceles_triangle_area_l692_692101


namespace even_positive_factors_count_l692_692178

theorem even_positive_factors_count (n : ℕ) (h : n = 2^4 * 3^3 * 7) : 
  ∃ k : ℕ, k = 32 := 
by
  sorry

end even_positive_factors_count_l692_692178


namespace equal_share_is_168_l692_692999

namespace StrawberryProblem

def brother_baskets : ℕ := 3
def strawberries_per_basket : ℕ := 15
def brother_strawberries : ℕ := brother_baskets * strawberries_per_basket

def kimberly_multiplier : ℕ := 8
def kimberly_strawberries : ℕ := kimberly_multiplier * brother_strawberries

def parents_difference : ℕ := 93
def parents_strawberries : ℕ := kimberly_strawberries - parents_difference

def total_strawberries : ℕ := kimberly_strawberries + brother_strawberries + parents_strawberries
def total_people : ℕ := 4

def equal_share : ℕ := total_strawberries / total_people

theorem equal_share_is_168 :
  equal_share = 168 := by
  -- We state that for the given problem conditions,
  -- the total number of strawberries divided equally among the family members results in 168 strawberries per person.
  sorry

end StrawberryProblem

end equal_share_is_168_l692_692999


namespace range_of_a_for_solution_set_l692_692868

theorem range_of_a_for_solution_set (a : ℝ) :
  ((∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1)) :=
sorry

end range_of_a_for_solution_set_l692_692868


namespace jeremy_remaining_money_l692_692219

-- Conditions as definitions
def computer_cost : ℝ := 3000
def accessories_cost : ℝ := 0.1 * computer_cost
def initial_money : ℝ := 2 * computer_cost

-- Theorem statement for the proof problem
theorem jeremy_remaining_money : initial_money - computer_cost - accessories_cost = 2700 := by
  -- Proof will be added here
  sorry

end jeremy_remaining_money_l692_692219


namespace iter_f_eq_l692_692239

namespace IteratedFunction

def f (n : ℕ) (x : ℕ) : ℕ :=
  if 2 * x <= n then
    2 * x
  else
    2 * n - 2 * x + 1

def iter_f (n m : ℕ) (x : ℕ) : ℕ :=
  (Nat.iterate (f n) m) x

variables (n m : ℕ) (S : Fin n.succ → Fin n.succ)

theorem iter_f_eq (h : iter_f n m 1 = 1) (k : Fin n.succ) :
  iter_f n m k = k := by
  sorry

end IteratedFunction

end iter_f_eq_l692_692239


namespace sum_of_roots_l692_692243

theorem sum_of_roots 
  (a b c : ℝ)
  (h1 : 1^2 + a * 1 + 2 = 0)
  (h2 : (∀ x : ℝ, x^2 + 5 * x + c = 0 → (x = a ∨ x = b))) :
  a + b + c = 1 :=
by
  sorry

end sum_of_roots_l692_692243


namespace masha_can_win_l692_692326

def balloon_game_winner (vika_first : Bool) (packs : List Nat) : String :=
  let total_packs := packs.length
  let odd_count := packs.count (λ n => n % 2 = 1)
  let even_count := total_packs - odd_count
  if odd_count > even_count then "Masha" else "Vika"

theorem masha_can_win : 
  (balloon_game_winner true [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) = "Masha" := 
by 
  -- the provable statement that Masha can guarantee a win here
  sorry

end masha_can_win_l692_692326


namespace grade_7a_students_l692_692585

theorem grade_7a_students :
  ∃ d m : ℕ, (d / (d + m) = 0.6 ∧ ((d - 1) / (d + m - 3) = 0.625)) ∧ d = 21 ∧ m = 14 :=
by
  sorry

end grade_7a_students_l692_692585


namespace find_n_l692_692728

noncomputable theory

def S : ℕ → ℝ := sorry -- Assume an arbitrary sequence S

axiom S7_gt_S8 : S 7 > S 8
axiom S8_gt_S6 : S 8 > S 6

theorem find_n : ∃ n > 0, S n * S (n + 1) < 0 :=
by {
  use 14,
  split,
  { norm_num },
  { sorry } -- Here would go the proof that S 14 * S 15 < 0 based on the provided conditions
}

end find_n_l692_692728


namespace woman_born_1892_l692_692812

theorem woman_born_1892 (y : ℕ) (hy : 1850 ≤ y^2 - y ∧ y^2 - y < 1900) : y = 44 :=
by
  sorry

end woman_born_1892_l692_692812


namespace intersecting_lines_l692_692475

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem intersecting_lines (x y : ℝ) : x ≠ 0 → y ≠ 0 → 
  (diamond x y = diamond y x) ↔ (y = x ∨ y = -x) := 
by
  sorry

end intersecting_lines_l692_692475


namespace num_pairs_of_integers_satisfying_equation_l692_692936

theorem num_pairs_of_integers_satisfying_equation : 
  {p : ℕ × ℕ // p.1 > 0 ∧ p.2 > 0 ∧ (p.1^2 - p.2^2 = 77)}.to_list.length = 2 := 
sorry

end num_pairs_of_integers_satisfying_equation_l692_692936


namespace triangle_isosceles_l692_692872

theorem triangle_isosceles (α β γ : ℝ) (A B C D E : Type)
  [triangle ABC] [angle A B C γ] [angle A C B β] :
  (triangle ADE) → (isosceles ADE) :=
by 
  -- Defining Conditions and Angle Properties
  have h1 : ∠BAC = α := sorry,
  have h2 : ∠ABC = β := sorry,
  have h3 : ∠ACB = γ := sorry,
  -- Angle at DAB due to given condition
  have h4 : ∠DAB = γ := sorry,
  -- Angle at CAE due to given condition
  have h5 : ∠CAE = β := sorry,
  -- Using exterior angle theorem and properties
  have h6 : ∠ADE = α + γ := sorry,
  have h7 : ∠AED = α + β := sorry,
  -- Using equality of angles to establish isosceles triangle
  have : ∠ADE = ∠AED := sorry,
  -- Concluding the isosceles property
  show isosceles ADE := sorry

end triangle_isosceles_l692_692872


namespace total_number_of_marbles_l692_692195

-- Define the variables o, p, and y as real numbers
variables (o p y : ℝ)

-- Define the conditions as Lean propositions
def condition1 := p + y = 7
def condition2 := o + y = 5
def condition3 := o + p = 9

-- The theorem statement that proves the total number of marbles in the jar equals 10.5
theorem total_number_of_marbles 
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3) : 
    o + p + y = 10.5 :=
by 
  sorry

end total_number_of_marbles_l692_692195


namespace train_length_l692_692810

theorem train_length (L : ℝ) 
    (cross_bridge : ∀ (t_bridge : ℝ), t_bridge = 10 → L + 200 = t_bridge * (L / 5))
    (cross_lamp_post : ∀ (t_lamp_post : ℝ), t_lamp_post = 5 → L = t_lamp_post * (L / 5)) :
  L = 200 := 
by 
  -- sorry is used to skip the proof part
  sorry

end train_length_l692_692810


namespace minimize_max_value_of_f_l692_692691

noncomputable def f (x A B : ℝ) : ℝ :=
  (Real.cos x) ^ 2 + 2 * Real.sin x * Real.cos x - (Real.sin x) ^ 2 + A * x + B

theorem minimize_max_value_of_f :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 / 2 * Real.pi →
    (argmin (λ x, max (f x 0 0)) (Icc 0 (3 / 2 * π))) ( f x 0 0 ) = sqrt 2 := sorry

end minimize_max_value_of_f_l692_692691


namespace triangle_area_is_correct_l692_692840

noncomputable def radius_ω : ℝ := 5

def externally_tangent (ω1 ω2 : set (ℝ × ℝ)) : Prop := sorry
def ω1 : set (ℝ × ℝ) := sorry
def ω2 : set (ℝ × ℝ) := sorry
def ω3 : set (ℝ × ℝ) := sorry
def Q1 : ℝ × ℝ := sorry
def Q2 : ℝ × ℝ := sorry
def Q3 : ℝ × ℝ := sorry

axiom ext_tangent_ω : ∀ i j : {1, 2, 3}, i ≠ j → externally_tangent (ω_i) (ω_j)
axiom points_on_ω : ∀ i : {1, 2, 3}, Qi ∈ ωi
axiom equilateral_Q : Q1Q2 = Q2Q3 ∧ Q2Q3 = Q3Q1
axiom tangent_lines : ∀ i : {1, 2, 3}, tangent_line Qi Qi₊1

theorem triangle_area_is_correct : 
  area (triangle Q1 Q2 Q3) = (155 * Real.sqrt 3 + 150 * Real.sqrt 22) / 12 := 
sorry

end triangle_area_is_correct_l692_692840


namespace sum_m_n_l692_692056

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l692_692056


namespace perimeter_original_rectangle_l692_692678

variable {L W : ℕ}

axiom area_original : L * W = 360
axiom area_changed : (L + 10) * (W - 6) = 360

theorem perimeter_original_rectangle : 2 * (L + W) = 76 :=
by
  sorry

end perimeter_original_rectangle_l692_692678


namespace find_hyperbola_l692_692878

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem find_hyperbola :
  let a₁ := 4
  let b₁ := 3
  let c := 5
  let P := ( -√5 / 2, -√6 )
  let a := 1
  let b := 24
  hyperbola a b P.1 P.2 :=
  hyperbola a b x y → x^2 - y^2 / 24 = 1 :=
by
  sorry

end find_hyperbola_l692_692878


namespace largest_prime_factor_of_1001_l692_692352

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692352


namespace number_of_pairs_satisfying_equation_l692_692923

theorem number_of_pairs_satisfying_equation : 
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77) = 2 :=
by 
  sorry

end number_of_pairs_satisfying_equation_l692_692923


namespace binomial_sum_of_coefficients_l692_692173

-- Definition and conditions
def integral_condition (m : ℝ) : Prop :=
  ∫ x in set.Icc 1 m, (2*x - 1) = 6

-- Theorem
theorem binomial_sum_of_coefficients (m : ℝ) (h : integral_condition m):
  ∑ k in finset.range (3 * m + 1), (3 * m).choose k * (-2)^k * (1)^((3 * m) - k) = -1 := by
  sorry

end binomial_sum_of_coefficients_l692_692173


namespace sum_m_n_l692_692058

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l692_692058


namespace ticket_savings_percentage_l692_692023

theorem ticket_savings_percentage:
  ∀ (P : ℝ), 9 * P - 6 * P = (1 / 3) * (9 * P) ∧ (33 + 1/3) = 100 * (3 * P / (9 * P)) := 
by
  intros P
  sorry

end ticket_savings_percentage_l692_692023


namespace maximum_value_conditions_l692_692131

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem maximum_value_conditions (x_0 : ℝ) (h_max : ∀ x : ℝ, f x ≤ f x_0) :
    f x_0 = x_0 ∧ f x_0 < 1 / 2 :=
by
  sorry

end maximum_value_conditions_l692_692131


namespace green_beads_in_pattern_l692_692288

noncomputable def G : ℕ := 3
def P : ℕ := 5
def R (G : ℕ) : ℕ := 2 * G
def total_beads (G : ℕ) (P : ℕ) (R : ℕ) : ℕ := 3 * (G + P + R) + 10 * 5 * (G + P + R)

theorem green_beads_in_pattern :
  total_beads 3 5 (R 3) = 742 :=
by
  sorry

end green_beads_in_pattern_l692_692288


namespace percentage_of_students_70_79_l692_692414

def tally_90_100 := 6
def tally_80_89 := 9
def tally_70_79 := 8
def tally_60_69 := 6
def tally_50_59 := 3
def tally_below_50 := 1

def total_students := tally_90_100 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50

theorem percentage_of_students_70_79 : (tally_70_79 : ℚ) / total_students = 8 / 33 :=
by
  sorry

end percentage_of_students_70_79_l692_692414


namespace scientific_notation_16000000_l692_692044

theorem scientific_notation_16000000 : (16_000_000 : ℝ) = 1.6 * 10^7 :=
by
  sorry

end scientific_notation_16000000_l692_692044


namespace sum_cos_eq_l692_692119

open Real

noncomputable def sum_cos (n : ℕ) (θ : ℝ) : ℝ :=
∑ k in Finset.range(n + 1), k * cos (k * θ)

theorem sum_cos_eq (n : ℕ) (θ : ℝ) :
  sum_cos n θ =  (n + 1) * cos (n * θ) - n * cos ((n + 1) * θ) - 1) / (4 * sin (θ / 2) ^ 2) := by
  sorry

end sum_cos_eq_l692_692119


namespace pharmacy_tubs_needed_l692_692427

theorem pharmacy_tubs_needed 
  (total_tubs_needed : ℕ) 
  (tubs_in_storage : ℕ) 
  (fraction_bought_new_vendor : ℚ) 
  (total_tubs_needed = 100) 
  (tubs_in_storage = 20)
  (fraction_bought_new_vendor = 1 / 4) :
  let tubs_needed_to_buy := total_tubs_needed - tubs_in_storage in
  let tubs_from_new_vendor := (tubs_needed_to_buy / 4 : ℕ) in
  let total_tubs_now := tubs_in_storage + tubs_from_new_vendor in
  let tubs_from_usual_vendor := total_tubs_needed - total_tubs_now in
  tubs_from_usual_vendor = 60 := 
by sorry

end pharmacy_tubs_needed_l692_692427


namespace mrs_li_actual_birthdays_l692_692652
   
   def is_leap_year (year : ℕ) : Prop :=
     (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
   
   def num_leap_years (start end_ : ℕ) : ℕ :=
     (start / 4 - start / 100 + start / 400) -
     (end_ / 4 - end_ / 100 + end_ / 400)
   
   theorem mrs_li_actual_birthdays : num_leap_years 1944 2011 = 16 :=
   by
     -- Calculation logic for the proof
     sorry
   
end mrs_li_actual_birthdays_l692_692652


namespace qualified_products_correct_l692_692610

def defect_rate : ℝ := 0.005
def total_produced : ℝ := 18000

theorem qualified_products_correct :
  total_produced * (1 - defect_rate) = 17910 := by
  sorry

end qualified_products_correct_l692_692610


namespace triangle_problems_l692_692207

variables {a b c : ℝ}
variables {A B : ℝ}

-- Given conditions
def condition1 : a = 2 * Real.sqrt 6 := sorry
def condition2 : Real.sin A = (2 * Real.sqrt 2) / 3 := sorry
def condition3 : (b * c * Real.cos A) = 9 := sorry
def condition4 : (b * b + c * c + (2 * b * c) / 3) = 24 := sorry

-- To prove
theorem triangle_problems :
  (b = 3) ∧
  (c = 3) ∧
  (Real.sin (A - B) = (5 * Real.sqrt 3) / 9) :=
by
  have h1 := condition1
  have h2 := condition2
  have h3 := condition3
  have h4 := condition4
  sorry

end triangle_problems_l692_692207


namespace max_intersection_points_l692_692099

theorem max_intersection_points : 
  let x_points := 15
  let y_points := 10
  ∃ I : ℕ, I = (x_points * (x_points - 1) / 2) * (y_points * (y_points - 1) / 2) ∧ I = 4725 := 
by
  let x_points := 15
  let y_points := 10
  exists.intro ((x_points * (x_points - 1) / 2) * (y_points * (y_points - 1) / 2)) (and.intro rfl sorry)

end max_intersection_points_l692_692099


namespace inequality_solution_set_l692_692864

theorem inequality_solution_set :
  {x : ℝ | (3 - x) * (1 + x) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end inequality_solution_set_l692_692864


namespace log_exp_symmetry_l692_692300

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5
noncomputable def exp_base_5 (x : ℝ) := 5 ^ x

theorem log_exp_symmetry : 
  ∀ x y : ℝ, (y = log_base_5 x) ↔ (x = exp_base_5 y) :=
by
  sorry

end log_exp_symmetry_l692_692300


namespace average_daily_attendance_percentage_l692_692193

theorem average_daily_attendance_percentage :
  let workers : ℕ := 86
  let attendance : ℕ × ℕ × ℕ × ℕ × ℕ := (72, 78, 69, 80, 75)
  let total_attendance := attendance.1 + attendance.2 + attendance.3 + attendance.4 + attendance.5
  let total_possible_attendance := workers * 5
  let average_percentage := (total_attendance : ℚ) / (total_possible_attendance : ℚ) * 100
  Float.roundHalfUp (average_percentage) = 87.0 := by
  sorry

end average_daily_attendance_percentage_l692_692193


namespace total_marbles_in_all_jars_l692_692214

theorem total_marbles_in_all_jars :
  let jarA := 28
  let jarB := jarA + 12
  let jarC := 2 * jarB
  jarA + jarB + jarC = 148 :=
by
  let jarA := 28
  let jarB := jarA + 12
  let jarC := 2 * jarB
  calc
    jarA + jarB + jarC = 28 + (28 + 12) + (2 * (28 + 12)) : by rfl
                   ... = 28 + 40 + 80 : by rfl
                   ... = 148 : by rfl

end total_marbles_in_all_jars_l692_692214


namespace number_of_elements_in_M_l692_692513

def f (x : ℝ) : ℝ := 10 / (x + 1) - real.sqrt x / 3

def M : set ℤ := { n | f ((n : ℝ) ^ 2 - 1) ≥ 0 }

theorem number_of_elements_in_M : fintype.card M = 6 :=
by sorry

end number_of_elements_in_M_l692_692513


namespace bob_corn_calc_l692_692457

noncomputable def bob_corn_left (initial_bushels : ℕ) (ears_per_bushel : ℕ) (bushels_taken_by_terry : ℕ) (bushels_taken_by_jerry : ℕ) (bushels_taken_by_linda : ℕ) (ears_taken_by_stacy : ℕ) : ℕ :=
  let initial_ears := initial_bushels * ears_per_bushel
  let ears_given_away := (bushels_taken_by_terry + bushels_taken_by_jerry + bushels_taken_by_linda) * ears_per_bushel + ears_taken_by_stacy
  initial_ears - ears_given_away

theorem bob_corn_calc :
  bob_corn_left 50 14 8 3 12 21 = 357 :=
by
  sorry

end bob_corn_calc_l692_692457


namespace warren_total_distance_approx_l692_692653

def warmup_speed : ℝ := 3
def warmup_time : ℝ := 15 / 60
def flat_run_speed : ℝ := 6
def flat_run_time : ℝ := 20 / 60
def uphill_speed : ℝ := 4
def uphill_time : ℝ := 15 / 60
def downhill_speed : ℝ := 7
def downhill_time : ℝ := 10 / 60
def cooldown_speed : ℝ := 2
def cooldown_time : ℝ := 30 / 60

def warmup_distance : ℝ := warmup_speed * warmup_time
def flat_run_distance : ℝ := flat_run_speed * flat_run_time
def uphill_distance : ℝ := uphill_speed * uphill_time
def downhill_distance : ℝ := downhill_speed * downhill_time
def cooldown_distance : ℝ := cooldown_speed * cooldown_time

def total_distance : ℝ :=
  warmup_distance + flat_run_distance + uphill_distance + downhill_distance + cooldown_distance

theorem warren_total_distance_approx : total_distance ≈ 5.92 := by
  sorry

end warren_total_distance_approx_l692_692653


namespace cosine_angle_between_a_and_b_l692_692523

noncomputable def vector_a := (3 : ℝ, 4 : ℝ)
noncomputable def vector_b := (5 : ℝ, 12 : ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cosine_of_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cosine_angle_between_a_and_b:
  cosine_of_angle vector_a vector_b = 63 / 65 := 
by
  sorry -- Proof is not required.

end cosine_angle_between_a_and_b_l692_692523


namespace third_shiny_penny_prob_l692_692785

open Nat

def num_shiny : Nat := 4
def num_dull : Nat := 5
def total_pennies : Nat := num_shiny + num_dull

theorem third_shiny_penny_prob :
  let a := 5
  let b := 9
  a + b = 14 := 
by
  sorry

end third_shiny_penny_prob_l692_692785


namespace find_distinct_prime_triples_l692_692860

noncomputable def areDistinctPrimes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r

def satisfiesConditions (p q r : ℕ) : Prop :=
  p ∣ (q + r) ∧ q ∣ (r + 2 * p) ∧ r ∣ (p + 3 * q)

theorem find_distinct_prime_triples :
  { (p, q, r) : ℕ × ℕ × ℕ | areDistinctPrimes p q r ∧ satisfiesConditions p q r } =
  { (5, 3, 2), (2, 11, 7), (2, 3, 11) } :=
by
  sorry

end find_distinct_prime_triples_l692_692860


namespace general_formula_for_a_integer_part_T_2017_l692_692546

section Problem1

variable (a : ℕ → ℝ)

-- Conditions
axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, 0 < n → a (n + 1) > a n
axiom a3 : ∀ n : ℕ, 0 < n → (a n + a (n + 1) - 1)^2 = 4 * a n * a (n + 1)

-- Prove the general formula for the sequence {a_n}
theorem general_formula_for_a (n : ℕ) : a n = n^2 := sorry

end Problem1

section Problem2

-- Define b_n and T_n
def b (n : ℕ) (a_n : ℝ) := 1 / (a_n)^(1/4)
def T_n (n : ℕ) (a : ℕ → ℝ) := ∑ i in Finset.range n, b i (a i)

-- Estimate the integer part of T_2017
theorem integer_part_T_2017 (a : ℕ → ℝ) : 88 < T_n 2017 a ∧ T_n 2017 a < 89 := sorry

end Problem2

end general_formula_for_a_integer_part_T_2017_l692_692546


namespace largest_prime_factor_1001_l692_692366

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692366


namespace find_x_l692_692780

theorem find_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  -- proof is not required, so we insert sorry
  sorry

end find_x_l692_692780


namespace parabola_ratio_l692_692887

theorem parabola_ratio (p : ℝ) (hp : p > 0) 
  (A B : ℝ × ℝ)
  (hA : C (A.snd * A.snd = 2 * p * A.fst)) 
  (hB : C (B.snd * B.snd = 2 * p * B.fst))
  (hF : F = (p / 2, 0))
  (h_lineAB : ∀ x : ℝ, (∃ y : ℝ, y = sqrt 3 * (x - p / 2)) → (C (y * y = 2 * p * x))) 
  (hA_quad : A.snd = - (sqrt 3 / 3) * p) 
  (hB_quad : B.snd = sqrt 3 * p) 
  (M : ℝ × ℝ)
  (hM : M = (-p / 2, A.snd)) :
  dist (0,0) (B.fst, B.snd) = 3 * dist (0,0) (M.fst, M.snd) := 
sorry

end parabola_ratio_l692_692887


namespace sum_m_n_l692_692053

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l692_692053


namespace degree_sum_of_star_l692_692035

theorem degree_sum_of_star (m : ℕ) (h : m ≥ 7) : degree_sum m = 180 * (m - 4) := 
sorry

-- Definitions to support theorem statement
noncomputable def degree_sum (m : ℕ) : ℕ :=
  let n := m + 3 in
  m * (180 - 360 / n)

end degree_sum_of_star_l692_692035


namespace rose_bushes_after_work_l692_692597

def initial_rose_bushes := 2
def planned_rose_bushes := 4
def planting_rate := 3
def removed_rose_bushes := 5

theorem rose_bushes_after_work :
  initial_rose_bushes + (planned_rose_bushes * planting_rate) - removed_rose_bushes = 9 :=
by
  sorry

end rose_bushes_after_work_l692_692597


namespace move_3m_left_is_neg_3m_l692_692819

-- Define the notation for movements
def move_right (distance : Int) : Int := distance
def move_left (distance : Int) : Int := -distance

-- Define the specific condition
def move_1m_right : Int := move_right 1

-- Define the assertion for moving 3m to the left
def move_3m_left : Int := move_left 3

-- State the proof problem
theorem move_3m_left_is_neg_3m : move_3m_left = -3 := by
  unfold move_3m_left
  unfold move_left
  rfl

end move_3m_left_is_neg_3m_l692_692819


namespace hershey_kisses_to_kitkats_ratio_l692_692461

-- Definitions based on the conditions
def kitkats : ℕ := 5
def nerds : ℕ := 8
def lollipops : ℕ := 11
def baby_ruths : ℕ := 10
def reeses : ℕ := baby_ruths / 2
def candy_total_before : ℕ := kitkats + nerds + lollipops + baby_ruths + reeses
def candy_remaining : ℕ := 49
def lollipops_given : ℕ := 5
def total_candy_before : ℕ := candy_remaining + lollipops_given
def hershey_kisses : ℕ := total_candy_before - candy_total_before

-- Theorem to prove the desired ratio
theorem hershey_kisses_to_kitkats_ratio : hershey_kisses / kitkats = 3 := by
  sorry

end hershey_kisses_to_kitkats_ratio_l692_692461


namespace subset_N_M_l692_692912

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | x^2 - x < 0 }

-- The proof goal
theorem subset_N_M : N ⊆ M := by
  sorry

end subset_N_M_l692_692912


namespace sum_common_divisors_l692_692105

theorem sum_common_divisors :
  let divisors_48 := {n : ℕ | n ∣ 48}
  let divisors_60 := {n : ℕ | n ∣ 60}
  let divisors_m30 := {n : ℕ | n ∣ 30}
  let divisors_180 := {n : ℕ | n ∣ 180}
  let divisors_90 := {n : ℕ | n ∣ 90}
  let common_divisors := divisors_48 ∩ divisors_60 ∩ divisors_m30 ∩ divisors_180 ∩ divisors_90 
  let common_divisors_set := {1, 2, 3, 6}
  ∑ x in common_divisors_set, x = 12 :=
by sorry

end sum_common_divisors_l692_692105


namespace find_m_l692_692184

theorem find_m (m : ℝ) (h : binomial_expansion_coefficient (mx + y)^6 3 3 = -160) : m = -2 := by
  sorry

end find_m_l692_692184


namespace smallest_e_l692_692470

theorem smallest_e (a b c d e : ℤ) (h_poly : a * (x:ℝ)^4 + b * x^3 + c * x^2 + d * x + e = 0)
  (h_root1 : x = -3)
  (h_root2 : x = 6)
  (h_root3 : x = 10)
  (h_root4 : x = -2 / 5)
  (h_int_coeffs : a ∈ ℤ ∧ b ∈ ℤ ∧ c ∈ ℤ ∧ d ∈ ℤ ∧ e ∈ ℤ) :
  e = 360 :=
by
  sorry

end smallest_e_l692_692470


namespace find_m_value_l692_692273

theorem find_m_value (P : set.Icc (-1) 3 → Prop) (P_prob : ∀ s, P s ↔ ∃ m, s = {x | |x| < m} ∧ intervalIntegrable s volume ∧ volume s / volume (Icc (-1 : ℝ) 3) = 0.75) :
  ∃ m : ℝ, m = 2 :=
by 
  sorry

end find_m_value_l692_692273


namespace probability_condition_l692_692632

open Real

def probability_log_floor (y : ℝ) : ℝ :=
  if 0 < y ∧ y < 1 then
    if (⌊log 10 (5 * y)⌋) = (⌊log 10 y⌋)
    then 1
    else 0
  else 0

theorem probability_condition :
  ∫ y in 0..1, probability_log_floor y = 1 / 9 :=
sorry

end probability_condition_l692_692632


namespace largest_prime_factor_1001_l692_692345

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692345


namespace pq_combined_work_rate_10_days_l692_692773

/-- Conditions: 
1. wr_p = wr_qr, where wr_qr is the combined work rate of q and r
2. wr_r allows completing the work in 30 days
3. wr_q allows completing the work in 30 days

We need to prove that the combined work rate of p and q allows them to complete the work in 10 days.
-/
theorem pq_combined_work_rate_10_days
  (wr_p wr_q wr_r wr_qr : ℝ)
  (h1 : wr_p = wr_qr)
  (h2 : wr_r = 1/30)
  (h3 : wr_q = 1/30) :
  wr_p + wr_q = 1/10 := by
  sorry

end pq_combined_work_rate_10_days_l692_692773


namespace intersection_of_spheres_has_13_integer_points_l692_692041

theorem intersection_of_spheres_has_13_integer_points :
  { p : ℤ × ℤ × ℤ | let ⟨x, y, z⟩ := p in 
    x^2 + y^2 + (z - 6)^2 ≤ 49 ∧ x^2 + y^2 + (z - 2)^2 ≤ 16 }.to_finset.card = 13 :=
by 
  sorry

end intersection_of_spheres_has_13_integer_points_l692_692041


namespace problem_1_and_2_problem_1_infinite_solutions_l692_692778

open Nat

theorem problem_1_and_2 (k : ℕ) (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2 = k * a * b * c) →
  (k = 1 ∨ k = 3) :=
sorry

theorem problem_1_infinite_solutions (k : ℕ) (h_k : k = 1 ∨ k = 3) :
  ∃ (a_n b_n c_n : ℕ) (n : ℕ), 
  a_n > 0 ∧ b_n > 0 ∧ c_n > 0 ∧
  (a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n) ∧
  ∀ x y : ℕ, (x = a_n ∧ y = b_n) ∨ (x = a_n ∧ y = c_n) ∨ (x = b_n ∧ y = c_n) →
    ∃ p q : ℕ, x * y = p^2 + q^2 :=
sorry

end problem_1_and_2_problem_1_infinite_solutions_l692_692778


namespace segment_longer_than_incircle_diameter_l692_692657

open Triangle

/--
Let \(A, B, C\) be points such that they form a triangle.
Let \(C'\) be a point on \(AB\), \(A'\) be a point on \(BC\), and \(B'\) a point on \(AC\).
If \(\angle A'C'B'\) is a right angle, prove that \(A'B'\) is longer than the diameter of the inscribed circle of \(\triangle ABC\).
-/
theorem segment_longer_than_incircle_diameter (A B C C' A' B' : Point)
    (hABC : Triangle A B C)
    (hC' : OnLineSegment A B C')
    (hA' : OnLineSegment B C A')
    (hB' : OnLineSegment A C B')
    (hAngleRight : ∠A'C'B' = 90) :
    SegmentLength A' B' > 2 * (incircle_radius A B C) :=
sorry

end segment_longer_than_incircle_diameter_l692_692657


namespace fare_range_l692_692192

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 8 else 8 + 1.5 * (x - 3)

theorem fare_range (x : ℝ) (hx : fare x = 16) : 8 ≤ x ∧ x < 9 :=
by
  sorry

end fare_range_l692_692192


namespace dot_product_value_l692_692166

variables (a b : ℝ × ℝ)

theorem dot_product_value
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a.1 * b.1 + a.2 * b.2 = -12 :=
sorry

end dot_product_value_l692_692166


namespace male_female_ratio_l692_692305

def ratio_of_male_to_female_in_class : ℕ → ℕ → Prop :=
  λ M F, 72 * 2 = 36 * (M + F) → M = F

theorem male_female_ratio (M F : ℕ) (h1 : 72 * 2 = 36 * (M + F)) : M = F :=
by
  sorry

end male_female_ratio_l692_692305


namespace patio_total_tiles_l692_692437

theorem patio_total_tiles (s : ℕ) (red_tiles : ℕ) (h1 : s % 2 = 1) (h2 : red_tiles = 2 * s - 1) (h3 : red_tiles = 61) :
  s * s = 961 :=
by
  sorry

end patio_total_tiles_l692_692437


namespace largest_prime_factor_1001_l692_692339

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692339


namespace lottery_at_least_one_match_l692_692582

noncomputable def lottery_probability : ℚ :=
  1 - (Nat.choose 30 6 : ℚ) / (Nat.choose 36 6 : ℚ)

theorem lottery_at_least_one_match :
  Real.round (lottery_probability * 1000) / 1000 = 0.695 := by
  sorry

end lottery_at_least_one_match_l692_692582


namespace bob_remaining_corns_l692_692458

theorem bob_remaining_corns (total_bushels : ℕ) (terry_bushels : ℕ) (jerry_bushels : ℕ)
                            (linda_bushels: ℕ) (stacy_ears: ℕ) (ears_per_bushel: ℕ):
                            total_bushels = 50 → terry_bushels = 8 → jerry_bushels = 3 →
                            linda_bushels = 12 → stacy_ears = 21 → ears_per_bushel = 14 →
                            (total_bushels - (terry_bushels + jerry_bushels + linda_bushels + stacy_ears / ears_per_bushel)) * ears_per_bushel = 357 :=
by intros total_cond terry_cond jerry_cond linda_cond stacy_cond ears_cond
   rw [total_cond, terry_cond, jerry_cond, linda_cond, stacy_cond, ears_cond]
   norm_cast
   have : 21 / 14 = (3 / 2 : ℕ) := sorry
   rw this
   linarith
   sorry

end bob_remaining_corns_l692_692458


namespace who_is_who_l692_692384

-- Define the types for inhabitants
inductive Inhabitant
| A : Inhabitant
| B : Inhabitant

-- Define the property of being a liar
def is_liar (x : Inhabitant) : Prop := 
  match x with
  | Inhabitant.A  => false -- Initial assumption, to be refined
  | Inhabitant.B  => false -- Initial assumption, to be refined

-- Define the statement made by A
def statement_by_A : Prop :=
  (is_liar Inhabitant.A ∧ ¬ is_liar Inhabitant.B)

-- The main theorem to prove
theorem who_is_who (h : ¬statement_by_A) :
  is_liar Inhabitant.A ∧ is_liar Inhabitant.B :=
by
  -- Proof goes here
  sorry

end who_is_who_l692_692384


namespace num_arrangements_l692_692318

def volunteers : Finset ℕ := {1, 2, 3, 4}
def counties : Finset ℕ := {1, 2, 3}

noncomputable def count_arrangements : ℕ :=
  (volunteers.card.choose 2) * (volunteers.card - 2).choose 1 * 1 / 2 * county_permutations.count

theorem num_arrangements :
  count_arrangements = 36 := 
by 
  sorry

end num_arrangements_l692_692318


namespace greatest_xy_value_l692_692950

theorem greatest_xy_value :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 5 * y = 200 ∧ x * y = 285 :=
by 
  sorry

end greatest_xy_value_l692_692950


namespace Mary_bought_stickers_initially_l692_692253

variable (S A M : ℕ) -- Define S, A, and M as natural numbers

-- Given conditions in the problem
def condition1 : Prop := S = A
def condition2 : Prop := M = 3 * A
def condition3 : Prop := A + (2 / 3) * M = 900

-- The theorem we need to prove
theorem Mary_bought_stickers_initially
  (h1 : condition1 S A)
  (h2 : condition2 A M)
  (h3 : condition3 A M)
  : S + A + M = 1500 :=
sorry -- Proof

end Mary_bought_stickers_initially_l692_692253


namespace tim_morning_running_hours_l692_692316

theorem tim_morning_running_hours 
  (runs_per_week : ℕ) 
  (total_hours_per_week : ℕ) 
  (runs_per_day : ℕ → ℕ) 
  (hrs_per_day_morning_evening_equal : ∀ (d : ℕ), runs_per_day d = runs_per_week * total_hours_per_week / runs_per_week) 
  (hrs_per_day : ℕ) 
  (hrs_per_morning : ℕ) 
  (hrs_per_evening : ℕ) 
  : hrs_per_morning = 1 :=
by 
  -- Given conditions
  have hrs_per_day := total_hours_per_week / runs_per_week
  have hrs_per_morning_evening := hrs_per_day / 2
  -- Conclusion
  sorry

end tim_morning_running_hours_l692_692316


namespace log_inequality_solution_l692_692118

open Real

theorem log_inequality_solution (x : ℝ) (h1 : log 2 (x - 1) / x > 0) (h2 : x ≠ 0) : 
  log 2 ((x - 1) / x) ≥ 1 ↔ (-1 ≤ x ∧ x < 0) :=
by
  simp [log]
  sorry

end log_inequality_solution_l692_692118


namespace find_age_l692_692675

open Nat

-- Definition of ages
def Teacher_Zhang_age (z : Nat) := z
def Wang_Bing_age (w : Nat) := w

-- Conditions
axiom teacher_zhang_condition (z w : Nat) : z = 3 * w + 4
axiom age_comparison_condition (z w : Nat) : z - 10 = w + 10

-- Proposition to prove
theorem find_age (z w : Nat) (hz : z = 3 * w + 4) (hw : z - 10 = w + 10) : z = 28 ∧ w = 8 := by
  sorry

end find_age_l692_692675


namespace total_money_divided_l692_692409

theorem total_money_divided (x y : ℕ) (hx : x = 1000) (ratioxy : 2 * y = 8 * x) : x + y = 5000 := 
by
  sorry

end total_money_divided_l692_692409


namespace coffee_shop_sales_l692_692825

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l692_692825


namespace ratio_fired_to_total_businesses_l692_692460

theorem ratio_fired_to_total_businesses (total_businesses : ℕ) (can_apply : ℕ) (fired_quit_ratio : ℚ) 
  (h1 : total_businesses = 72) 
  (h2 : can_apply = 12) 
  (h3 : fired_quit_ratio = 1 / 3) : 
  let F := total_businesses - can_apply - (fired_quit_ratio * (total_businesses - can_apply)) in
  let Q := fired_quit_ratio * (total_businesses - can_apply) in
  (F / total_businesses) = (5 / 9) := 
by 
  sorry

end ratio_fired_to_total_businesses_l692_692460


namespace number_of_flute_players_l692_692307

theorem number_of_flute_players (F T B D C H : ℕ)
  (hT : T = 3 * F)
  (hB : B = T - 8)
  (hD : D = B + 11)
  (hC : C = 2 * F)
  (hH : H = B + 3)
  (h_total : F + T + B + D + C + H = 65) :
  F = 6 :=
by
  sorry

end number_of_flute_players_l692_692307


namespace ant_rest_position_l692_692818

noncomputable def percent_way_B_to_C (s : ℕ) : ℕ :=
  let perimeter := 3 * s
  let distance_traveled := (42 * perimeter) / 100
  let distance_AB := s
  let remaining_distance := distance_traveled - distance_AB
  (remaining_distance * 100) / s

theorem ant_rest_position :
  ∀ (s : ℕ), percent_way_B_to_C s = 26 :=
by
  intros
  unfold percent_way_B_to_C
  sorry

end ant_rest_position_l692_692818


namespace sum_b_n_first_32_terms_l692_692139
  
  variable (a : ℕ → ℝ)
  variable (b : ℕ → ℝ)
  
  def a_n (n : ℕ) : ℝ := 3 * n + 1
  
  def b_n (n : ℕ) : ℝ := 1 / (a n * Real.sqrt (a (n + 1)) + a (n + 1) * Real.sqrt (a n))
  
  theorem sum_b_n_first_32_terms :
    (Σ k in Finset.range 32, b k) = 2 / 15 :=
  sorry
  
end sum_b_n_first_32_terms_l692_692139


namespace right_triangle_third_side_l692_692881

theorem right_triangle_third_side (c a b : ℝ) (h : a^2 + b^2 = c^2)
  (h_a : a = 4 ∨ b = 4) (h_b : a = 5 ∨ b = 5) :
  c = 3 ∨ c = real.sqrt 41 :=
by sorry

end right_triangle_third_side_l692_692881


namespace min_abs_z_plus_i_l692_692233

theorem min_abs_z_plus_i (z : ℂ) (h: |z^2 + 9| = |z * (z + 3 * complex.I)|) : ∃ w : ℂ, z = -3 * complex.I → |w + complex.I| = 2 :=
by
  sorry

end min_abs_z_plus_i_l692_692233


namespace coefficient_x3_term_in_expansion_l692_692104

theorem coefficient_x3_term_in_expansion :
  (expand (2 - x)^5).coeff 3 = -40 := by sorry

end coefficient_x3_term_in_expansion_l692_692104


namespace find_cost_of_book_sold_at_loss_l692_692389

-- Definitions from the conditions
def total_cost (C1 C2 : ℝ) : Prop := C1 + C2 = 540
def selling_price_loss (C1 : ℝ) : ℝ := 0.85 * C1
def selling_price_gain (C2 : ℝ) : ℝ := 1.19 * C2
def same_selling_price (SP1 SP2 : ℝ) : Prop := SP1 = SP2

theorem find_cost_of_book_sold_at_loss (C1 C2 : ℝ) 
  (h1 : total_cost C1 C2) 
  (h2 : same_selling_price (selling_price_loss C1) (selling_price_gain C2)) :
  C1 = 315 :=
by {
   sorry
}

end find_cost_of_book_sold_at_loss_l692_692389


namespace largest_prime_factor_of_1001_l692_692370

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692370


namespace chord_length_circle_line_l692_692411

noncomputable def circle_equation := ∀ (x y : ℝ), x^2 + y^2 - 2 * x = 0

theorem chord_length_circle_line
    (Cx Cy : ℝ)
    (r : ℝ)
    (h_circle : ∀ (x y : ℝ), circle_equation x y ↔ (x - Cx)^2 + (y - Cy)^2 = r^2)
    (h_line_intercepts : ∀ (P : ℝ × ℝ), (∃ (x y : ℝ), P = (x, y) ∧ y = x)) :
    (∃ (chord_length : ℝ), chord_length = (2 * sqrt (r^2 - (sqrt 2 / 2)^2)) ∧ chord_length = sqrt 2) := sorry

end chord_length_circle_line_l692_692411


namespace arithmetic_expression_equality_l692_692835

theorem arithmetic_expression_equality : 18 * 36 - 27 * 18 = 162 := by
  sorry

end arithmetic_expression_equality_l692_692835


namespace problem_l692_692156

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := x^2 + a * x + b
noncomputable def g (x : ℝ) := 2 * x^2 + 4 * x - 30
noncomputable def a_seq : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := (a_seq n)^2 / 2 + a_seq n

noncomputable def b_seq (n : ℕ) : ℝ := 1 / (2 + a_seq n)
noncomputable def S (n : ℕ) : ℝ := (finset.range n).sum (λ i, b_seq i)
noncomputable def T (n : ℕ) : ℝ := (finset.range n).prod (λ i, b_seq i)

theorem problem(
  (H1 : (∀ x : ℝ, f x 2 (-15) = g x)) :
  ∃ (a b : ℝ), (a = 2) ∧ (b = -15) ∧
  ∀ n : ℕ, 2 ^ (n + 1) * T n + S n = 2 ∧ 
  (2 * (1 - (4/5)^n) ≤ S n) ∧ (S n < 2) :=
begin
  -- Proof steps are not required
  sorry
end

end problem_l692_692156


namespace expression_equals_eight_thirds_l692_692842

noncomputable def compute_expression : ℚ :=
  (27 / 8) ^ (-1 / 3) + Real.log2 (Real.log2 16)

theorem expression_equals_eight_thirds :
  compute_expression = 8 / 3 :=
  sorry

end expression_equals_eight_thirds_l692_692842


namespace fibonacci_150_mod_9_eq_8_l692_692676

def fibonacci_mod (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 1
  else
    let f : ℕ → ℕ := λ n, fibonacci_mod (n - 1) + fibonacci_mod (n - 2)
    in f n % 9

theorem fibonacci_150_mod_9_eq_8 :
  fibonacci_mod 150 = 8 :=
sorry

end fibonacci_150_mod_9_eq_8_l692_692676


namespace a_lt_b_lt_c_l692_692873

noncomputable def a := 12 * e * (real.log 4 / 4)
noncomputable def b := 12 * e * (real.log 3 / 3)
def c : ℝ := 12

theorem a_lt_b_lt_c : a < b ∧ b < c :=
by {
    -- Placeholder for the actual proof.
    sorry
}

end a_lt_b_lt_c_l692_692873


namespace opposite_of_neg2023_l692_692701

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l692_692701


namespace translated_line_expression_l692_692186

theorem translated_line_expression (x y : ℝ) (b : ℝ) :
  (∀ x y, y = 2 * x + 3 ∧ (5, 1).2 = 2 * (5, 1).1 + b) → y = 2 * x - 9 :=
by
  sorry

end translated_line_expression_l692_692186


namespace time_to_pass_tree_l692_692440

noncomputable def length_of_train : ℝ := 275
noncomputable def speed_in_kmh : ℝ := 90
noncomputable def speed_in_m_per_s : ℝ := speed_in_kmh * (5 / 18)

theorem time_to_pass_tree : (length_of_train / speed_in_m_per_s) = 11 :=
by {
  sorry
}

end time_to_pass_tree_l692_692440


namespace largest_prime_factor_of_1001_l692_692332

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692332


namespace arrow_symmetry_l692_692416

structure ArrowShape (P: Type _) [OrderedCommGroup P] :=
  (shaft : linear_order.C (OrderedCommGroup P))
  (head : linear_order.Triangular C (OrderedCommGroup P))
  (aligned : centered C (OrderedCommGroup P))

def arrow_has_vertical_symmetry {P : Type _} [OrderedCommGroup P] (arrow : ArrowShape P) : Prop :=
  ∃ line, is_vertical_symmetry line arrow

theorem arrow_symmetry {P : Type _} [OrderedCommGroup P] (arrow : ArrowShape P) :
  arrow_has_vertical_symmetry arrow :=
sorry

end arrow_symmetry_l692_692416


namespace geom_prog_common_ratio_l692_692491

-- Definition of a geometric progression
def geom_prog (u : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)

-- Statement of the problem
theorem geom_prog_common_ratio (u : ℕ → ℝ) (q : ℝ) (hq : ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)) :
  (q = (1 + Real.sqrt 5) / 2) ∨ (q = (1 - Real.sqrt 5) / 2) :=
sorry

end geom_prog_common_ratio_l692_692491


namespace largest_prime_factor_of_1001_l692_692329

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692329


namespace max_guaranteed_amount_one_blue_cube_l692_692201

-- Conditions: There is only one blue cube and 100 boxes
-- Question: What is the maximum amount Alexei can guarantee himself?

theorem max_guaranteed_amount_one_blue_cube
  (n : Nat) (h_n : n = 100) :
  let amount := (2^n) / n in
  amount = 2^100 / 100 := 
by 
  -- Placeholder for the actual proof
  sorry

end max_guaranteed_amount_one_blue_cube_l692_692201


namespace distance_to_mall_l692_692289

-- Given conditions
def average_speed : ℝ := 60 -- km/h
def stops_every : ℝ := 5 / 60 -- hours (5 minutes converted to hours)
def stops_away : ℝ := 5

-- The main theorem
theorem distance_to_mall : 
  average_speed * (stops_away * stops_every) = 25 := 
by 
  -- The proof goes here.
  sorry

end distance_to_mall_l692_692289


namespace sum_of_interior_edges_of_frame_l692_692431

noncomputable def frame_sum_of_interior_edges
  (frame_width : ℝ)
  (frame_area : ℝ)
  (outer_edge : ℝ)
  (interior_length : ℝ) : ℝ :=
  let interior_width := (outer_edge - 2 * frame_width) in
  let y := (frame_area + 3 * (interior_length - frame_width)) / (outer_edge - interior_width) in
  2 * (interior_length + (y - 2 * frame_width))

theorem sum_of_interior_edges_of_frame : 
  frame_sum_of_interior_edges 1.5 27 6 3 = 12 :=
sorry

end sum_of_interior_edges_of_frame_l692_692431


namespace cube_has_8_perpendicular_edges_l692_692224

-- Define a cube and an edge of the cube
structure Cube :=
  (edges : Fin 12)  -- Assume every cube has exactly 12 edges.

-- Define the condition that $AA_1$ is an edge of the cube
variable (AA_1 : Cube)

-- The theorem to prove the number of edges perpendicular to $AA_1$ is 8
theorem cube_has_8_perpendicular_edges (c : Cube) (h : AA_1 ∈ c.edges) : 
  ∃ n, n = 8 ∧ (∀ e ∈ c.edges, e ≠ AA_1 → e ⊥ AA_1) := sorry

end cube_has_8_perpendicular_edges_l692_692224


namespace geometric_series_sum_eq_base_case_n1_l692_692266

noncomputable section

open Real

theorem geometric_series_sum_eq (a : ℝ) (h : a ≠ 1) (n : ℕ) (hn : 0 < n) : 
  (Finset.range (n + 2)).sum (λ i, a^i) = (1 - a^(n + 2)) / (1 - a) := sorry

theorem base_case_n1 (a : ℝ) (h : a ≠ 1) : 
  (Finset.range 3).sum (λ i, a^i) = 1 + a + a^2 := by
  simp [h]
  sorry

end geometric_series_sum_eq_base_case_n1_l692_692266


namespace monotonic_increasing_interval_log_quadratic_l692_692302

noncomputable def function_monotonic_increasing_interval (f : ℝ → ℝ) (interval : set ℝ) :=
  ∀ x y ∈ interval, x < y → f x ≤ f y

noncomputable def log_base (b : ℝ) (x : ℝ) := Real.log x / Real.log b

theorem monotonic_increasing_interval_log_quadratic :
  function_monotonic_increasing_interval
    (λ x, log_base 0.5 (x^2 - 4*x + 3))
    { x : ℝ | x < 1 } :=
sorry

end monotonic_increasing_interval_log_quadratic_l692_692302


namespace green_blue_tile_difference_l692_692211

theorem green_blue_tile_difference (initial_blue initial_green add_green : ℕ) 
  (h_initial_blue : initial_blue = 20)
  (h_initial_green : initial_green = 8)
  (h_add_green : add_green = 36) : 
  initial_green + add_green - initial_blue = 24 := 
by 
  rw [h_initial_blue, h_initial_green, h_add_green]
  exact rfl

end green_blue_tile_difference_l692_692211


namespace lincoln_county_houses_l692_692315

theorem lincoln_county_houses (original_houses : ℕ) (built_houses : ℕ) (total_houses : ℕ) 
(h1 : original_houses = 20817) 
(h2 : built_houses = 97741) 
(h3 : total_houses = original_houses + built_houses) : 
total_houses = 118558 :=
by
  -- proof omitted
  sorry

end lincoln_county_houses_l692_692315


namespace find_perimeter_l692_692774

-- Defining the conditions
def width : ℝ := 70
def rate : ℝ := 6.5
def cost : ℝ := 1950
def length := width + 10
def perimeter := 2 * (length + width)

-- Stating the theorem to be proved
theorem find_perimeter (h1 : cost = perimeter * rate) : perimeter = 300 :=
by
  have h2 : length = width + 10 := rfl
  have h3 : perimeter = 2 * (width + length) := rfl
  -- Proof omitted for brevity
  sorry

end find_perimeter_l692_692774


namespace num_pairs_of_integers_satisfying_equation_l692_692934

theorem num_pairs_of_integers_satisfying_equation : 
  {p : ℕ × ℕ // p.1 > 0 ∧ p.2 > 0 ∧ (p.1^2 - p.2^2 = 77)}.to_list.length = 2 := 
sorry

end num_pairs_of_integers_satisfying_equation_l692_692934


namespace calculate_expression_l692_692464

theorem calculate_expression : 2 * (-2) + (-3) = -7 := 
  sorry

end calculate_expression_l692_692464


namespace power_function_fourth_quadrant_l692_692817

theorem power_function_fourth_quadrant (α : ℚ): 
  ∀ x : ℝ, x > 0 → x^α > 0 :=
by
  sorry

end power_function_fourth_quadrant_l692_692817


namespace scientific_notation_of_1206_million_l692_692972

theorem scientific_notation_of_1206_million :
  (1206 * 10^6 : ℝ) = 1.206 * 10^7 :=
by
  sorry

end scientific_notation_of_1206_million_l692_692972


namespace remainder_when_divided_l692_692114

open Polynomial

noncomputable def poly : Polynomial ℚ := X^6 + X^5 + 2*X^3 - X^2 + 3
noncomputable def divisor : Polynomial ℚ := (X + 2) * (X - 1)
noncomputable def remainder : Polynomial ℚ := -X + 5

theorem remainder_when_divided :
  ∃ q : Polynomial ℚ, poly = divisor * q + remainder :=
sorry

end remainder_when_divided_l692_692114


namespace consistent_2_configurations_of_order_2_with_exactly_1_cell_l692_692883

-- Define the set A with 10 elements
def A : Finset (Fin 10) := Finset.univ

-- Define what it means to be a consistent 2-configuration of order 2 with exactly 1 cell
def isConsistent2Config (A : Finset (Fin 10)) (config : Finset (Finset (Fin 10))) : Prop :=
  ∃ (pairs : List (Finset (Fin 10))), 
    (pairs.length = 10) ∧ 
    (∀ pair ∈ pairs, pair.card = 2) ∧
    (config = pairs.toFinset) ∧ 
    (∀ i ∈ A, ∃! pair ∈ config, i ∈ pair) ∧
    (pairs.head = {0, 1}) ∧ 
    (pairs.tail.head = {1, 2}) ∧ 
    -- repeat similar conditions until the pairs form a cycle

-- Define the number of configurations
def countConsistent2Configs (A : Finset (Fin 10)) : Nat :=
  if isConsistent2Config A ?config then 181440 else 0

-- The theorem to prove the correct answer
theorem consistent_2_configurations_of_order_2_with_exactly_1_cell :
  countConsistent2Configs A = 181440 :=
by
  sorry

end consistent_2_configurations_of_order_2_with_exactly_1_cell_l692_692883


namespace rate_of_second_batch_l692_692009

-- Define the problem statement
theorem rate_of_second_batch
  (rate_first : ℝ)
  (weight_first weight_second weight_total : ℝ)
  (rate_mixture : ℝ)
  (profit_multiplier : ℝ) 
  (total_selling_price : ℝ) :
  rate_first = 11.5 →
  weight_first = 30 →
  weight_second = 20 →
  weight_total = weight_first + weight_second →
  rate_mixture = 15.12 →
  profit_multiplier = 1.20 →
  total_selling_price = weight_total * rate_mixture →
  (rate_first * weight_first + (weight_second * x) * profit_multiplier = total_selling_price) →
  x = 14.25 :=
by
  intros
  sorry

end rate_of_second_batch_l692_692009


namespace infinite_series_eval_l692_692486

open Filter
open Real
open Topology
open BigOperators

-- Define the relevant expression for the infinite sum
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (n / (n^4 - 4 * n^2 + 8))

-- The theorem statement
theorem infinite_series_eval : infinite_series_sum = 5 / 24 :=
by sorry

end infinite_series_eval_l692_692486


namespace div_of_abs_values_l692_692135

theorem div_of_abs_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x < y) : x / y = -2 := 
by
  sorry

end div_of_abs_values_l692_692135


namespace guess_probability_greater_than_two_thirds_l692_692262

theorem guess_probability_greater_than_two_thirds :
  (1335 : ℝ) / 2002 > 2 / 3 :=
by {
  -- Placeholder for proof
  sorry
}

end guess_probability_greater_than_two_thirds_l692_692262


namespace initial_shells_count_l692_692089

theorem initial_shells_count : 
  let ed_limpet := 7
  let ed_oyster := 2
  let ed_conch := 4
  let ed_total := ed_limpet + ed_oyster + ed_conch
  let jacob_total := ed_total + 2
  let total_shells := 30
  let found_shells := ed_total + jacob_total
  total_shells - found_shells = 2 :=
by
  -- Ed's and Jacob's shells data
  let ed_limpet := 7
  let ed_oyster := 2
  let ed_conch := 4
  let ed_total := ed_limpet + ed_oyster + ed_conch
  let jacob_total := ed_total + 2
  let total_shells := 30
  let found_shells := ed_total + jacob_total
  
  -- Prove the initial shells count
  show total_shells - found_shells = 2, by sorry

end initial_shells_count_l692_692089


namespace number_of_arrangements_2x50_l692_692969

theorem number_of_arrangements_2x50 :
  let arrangements := {f : ℕ → (ℕ × ℕ) | ∀ n, n ∈ {1..99} → adj (f n) (f (n+1))} 
  let adj (a b : ℕ × ℕ) : Prop := (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨ ((a.1 + 1 = b.1 ∨ a.1 = b.1 + 1) ∧ a.2 = b.2)
  in arrangements.count = 4904 := sorry

end number_of_arrangements_2x50_l692_692969


namespace graph_single_point_l692_692287

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end graph_single_point_l692_692287


namespace expansion_terms_l692_692154

theorem expansion_terms (x : ℝ) :
  let n := 5 in
  (2^10 - 2^5 = 992) →
  ((2x - (1 / x))^10 = 
    ∑ r in Finset.range 11, (Nat.choose 10 r * 2^(10 - r) * (-1)^r * x^(10 - 2 * r)) →
  ((Nat.choose 10 9 * 2^(10 - 9) * (-1)^9 * x^(10 - 2 * 9) = -20 * x^(-8)) ∧
   (Nat.choose 10 5 * 2^(10 - 5) * (-1)^5 * x^(10 - 2 * 5) = -8064) ∧
   (Nat.choose 10 3 * 2^(10 - 3) * (-1)^3 * x^(10 - 2 * 3) = -15360 * x^4)) :=
begin
  sorry
end

end expansion_terms_l692_692154


namespace isabella_hair_length_l692_692212

theorem isabella_hair_length (original : ℝ) (increase_percent : ℝ) (new_length : ℝ) 
    (h1 : original = 18) (h2 : increase_percent = 0.75) 
    (h3 : new_length = original + increase_percent * original) : 
    new_length = 31.5 := by sorry

end isabella_hair_length_l692_692212


namespace prove_p_l692_692723

variable (P : Set ℤ)
variable (h1 : ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0)
variable (h2 : ∃ x ∈ P, (x % 2 = 0) ∧ ∃ y ∈ P, (y % 2 = 1))
variable (h3 : -1 ∉ P)
variable (h4 : ∀ x y ∈ P, x + y ∈ P)

theorem prove_p (P : Set ℤ) :
  (∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0) ∧
  (∃ x ∈ P, (x % 2 = 0) ∧ ∃ y ∈ P, (y % 2 = 1)) ∧
  (-1 ∉ P) ∧
  (∀ x y ∈ P, x + y ∈ P) →
  (0 ∈ P ∧ 2 ∉ P) :=
by
  sorry

end prove_p_l692_692723


namespace sum_of_three_squares_l692_692958

theorem sum_of_three_squares (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
  sorry

end sum_of_three_squares_l692_692958


namespace find_d_for_single_point_l692_692285

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end find_d_for_single_point_l692_692285


namespace Connie_blue_markers_l692_692843

theorem Connie_blue_markers (total_markers red_markers : ℕ) (h_total_markers : total_markers = 3343) (h_red_markers : red_markers = 2315) : 
total_markers - red_markers = 1028 := 
by {
  rw [h_total_markers, h_red_markers],
  norm_num,
}

end Connie_blue_markers_l692_692843


namespace expected_length_first_group_19_49_l692_692716

noncomputable def expected_first_group_length (ones zeros : ℕ) : ℕ :=
  let total := ones + zeros;
  let pI := 1 / 50;
  let pJ := 1 / 20;
  let expected_I := ones * pI;
  let expected_J := zeros * pJ;
  expected_I + expected_J

theorem expected_length_first_group_19_49 : expected_first_group_length 19 49 = 2.83 :=
by
  sorry

end expected_length_first_group_19_49_l692_692716


namespace largest_prime_factor_1001_l692_692357

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692357


namespace shaded_area_l692_692593

structure Square where
  side_length : ℝ

structure Position where
  -- Coordinates of point (could be adjusted as needed)
  x : ℝ
  y : ℝ

def area_of_square (s : Square) : ℝ :=
  s.side_length ^ 2

def right_triangle_area (base height : ℝ) : ℝ :=
  0.5 * base * height

theorem shaded_area 
  (large_square : Square) (large_square.side_length = 12) 
  (small_square : Square) (small_square.side_length = 4)
  (D : Position) (E : Position) (H : Position) (F : Position) 
  (D.x = 0) (D.y = 6) (H.x = 8) (H.y = 0)
  (right_angle_DGF : ¬(D = E) ∧ ¬(E = F) ∧ ¬(F = D))
  : 
  let DG := (3:ℝ) -- Using the similarity of triangles to establish DG = 3
  let GF := (4:ℝ) -- Given in the problem side of DF
  let triangle_area := right_triangle_area 3 4
  let small_square_area := area_of_square small_square
  small_square_area - triangle_area = 10 := sorry

end shaded_area_l692_692593


namespace f_240_is_388_l692_692724

open Set

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem f_240_is_388 :
    (∀ n : ℕ, n > 0 → f(n) ∈ {f(1), f(2), f(3), ...}) ∧ 
    (∀ n : ℕ, n > 0 → g(n) = f(f(n)) + 1) ∧ 
    (∀ n m : ℕ, n > 0 ∧ m > 0 → f(n) ≠ g(m))  ∧ 
    (∀ n : ℕ, n > 0 → f(n) < f(n+1) ∧ g(n) < g(n+1)) ∧ 
    (∀ n m : ℕ, n ≠ m → f(n) ≠ f(m) ∧ g(n) ≠ g(m) ) 
    → f(240) = 388 :=
begin
  sorry
end

end f_240_is_388_l692_692724


namespace money_total_l692_692445

theorem money_total (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 350) (h3 : C = 100) : A + B + C = 450 :=
by {
  sorry
}

end money_total_l692_692445


namespace log_addition_property_l692_692903

noncomputable def logFunction (x : ℝ) : ℝ := Real.log x

theorem log_addition_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : logFunction (a * b) = 1) :
  logFunction (a^2) + logFunction (b^2) = 2 :=
by
  sorry

end log_addition_property_l692_692903


namespace candle_lighting_time_l692_692321

noncomputable def burning_time := λ (l : ℕ) (burnt_rate : ℕ), l - (burnt_rate * (60 * (l / burnt_rate)))

theorem candle_lighting_time :
  ∀ (l : ℕ), (l > 0) → (burning_time l (l / 4) (60 * 5)) = 3 * (burning_time l (l / 6) (60 * 5)) →
    (5 = 5) :=
by
  intros l hl h_eq
  sorry

end candle_lighting_time_l692_692321


namespace lcm_of_coprime_product_eq_self_l692_692393

theorem lcm_of_coprime_product_eq_self (a b : ℕ) (h_coprime : Nat.coprime a b) (h_product : a * b = 117) : Nat.lcm a b = 117 := 
sorry

end lcm_of_coprime_product_eq_self_l692_692393


namespace max_excellent_videos_l692_692962

-- Definition of a video with likes and expert score
structure Video :=
  (likes : ℕ)
  (score : ℕ)

-- Definition of not inferior relationship
def not_inferior (A B : Video) : Prop :=
  A.likes > B.likes ∨ A.score > B.score

-- Definition of being an excellent video
def is_excellent (A : Video) (others : List Video) : Prop :=
  ∀ B ∈ others, not_inferior A B

-- The main theorem we need to prove
theorem max_excellent_videos {A1 A2 A3 A4 A5 : Video} :
  let videos := [A1, A2, A3, A4, A5]
  (∀ v ∈ videos, is_excellent v (videos.erase v)) →
  ∃ n, n = 5 :=
by
  sorry

end max_excellent_videos_l692_692962


namespace intersection_P_Q_l692_692548

open Set

noncomputable def P : Set ℝ := {1, 2, 3, 4}

noncomputable def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := 
by {
  sorry
}

end intersection_P_Q_l692_692548


namespace sequence_logarithms_none_of_above_l692_692951

theorem sequence_logarithms_none_of_above (a b c n : ℝ) (s : ℝ) (ha : a < b) (hb : b < c) (hs : s > 1) (hn : n > 1) (hb_eq_as : b = a * s) (hc_eq_as3 : c = a * s^3) :
  (∃ d : ℝ, ∀ k, k = [(log n) / (log a), (log n) / (log b), (log n) / (log c)] -> k ≠ (arithmetic_progression d)) ∧
  (∃ r : ℝ, ∀ k, k = [(log n) / (log a), (log n) / (log b), (log n) / (log c)] -> k ≠ (geometric_progression r)) ∧
  (∃ t : ℝ, ∀ k, k = [1 / (log n / log a), 1 / (log n / log b), 1 / (log n / log c)] -> k ≠ (arithmetic_progression t)) :=
  sorry

end sequence_logarithms_none_of_above_l692_692951


namespace average_cost_of_fruit_l692_692640

theorem average_cost_of_fruit : 
  (12 * 2 + 4 * 1 + 4 * 3) / (12 + 4 + 4) = 2 := 
by
  -- Given conditions as definitions
  let cost_apple := 2     -- cost per apple
  let cost_banana := 1    -- cost per banana
  let cost_orange := 3    -- cost per orange
  let qty_apples := 12    -- number of apples bought
  let qty_bananas := 4    -- number of bananas bought
  let qty_oranges := 4    -- number of oranges bought
  
  -- Average cost calculation
  have total_cost := qty_apples * cost_apple + qty_bananas * cost_banana + qty_oranges * cost_orange
  have total_qty := qty_apples + qty_bananas + qty_oranges
  have average_cost := total_cost.toRat / total_qty.toRat
  
  show average_cost = 2 by sorry

end average_cost_of_fruit_l692_692640


namespace find_length_of_EF_l692_692152

-- Definitions based on conditions
noncomputable def AB : ℝ := 300
noncomputable def DC : ℝ := 180
noncomputable def BC : ℝ := 200
noncomputable def E_as_fraction_of_BC : ℝ := (3 / 5)

-- Derived definition based on given conditions
noncomputable def EB : ℝ := E_as_fraction_of_BC * BC
noncomputable def EC : ℝ := BC - EB
noncomputable def EF : ℝ := (EC / BC) * DC

-- The theorem we need to prove
theorem find_length_of_EF : EF = 72 := by
  sorry

end find_length_of_EF_l692_692152


namespace opposite_of_neg_2023_l692_692697

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l692_692697


namespace probability_no_3x3_red_square_l692_692080

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l692_692080


namespace sector_angle_l692_692531

theorem sector_angle (r θ : ℝ) 
  (h1 : r * θ + 2 * r = 6) 
  (h2 : 1/2 * r^2 * θ = 2) : 
  θ = 1 ∨ θ = 4 :=
by 
  sorry

end sector_angle_l692_692531


namespace number_of_true_propositions_l692_692816

theorem number_of_true_propositions :
  let P1 := (∀ T : Triangle, (T.equilateral → (∀ A : Angle, A.internal ∧ A.measure = 60 → T.has_angle A)))
  let P2 := (∀ T1 T2 : Triangle, (T1.congruent T2 → T1.area = T2.area))
  let P3 := (k > 0 → ∃ x : ℝ, x^2 + 2*x - k = 0)
  let P4 := (∀ a b : ℝ, (ab ≠ 0 → a ≠ 0))
  let P1c := (∀ T : Triangle, (∀ A : Angle, A.internal ∧ A.measure = 60 → T.has_angle A) → T.equilateral) -- Converse of P1
  let P2n := (∀ T1 T2 : Triangle, (T1.area = T2.area → ¬ T1.congruent T2)) -- Negation of P2
  let P3c := (∃ x : ℝ, x^2 + 2*x - k = 0 → k > 0) -- Contrapositive of P3
  let P4n := (∃ a b : ℝ, ab = 0 ∧ ¬(a = 0)) -- Negation of P4
  (P1c ∧ P2n → false) ∧ (P3c ∧ P4n → false)
  → 2 := sorry

end number_of_true_propositions_l692_692816


namespace trajectory_of_moving_point_l692_692129

noncomputable def locus_of_point_P (F1 F2 : ℝ × ℝ) (a : ℝ) (P : ℝ × ℝ) : Prop :=
  let d1 := (dist P F1: ℝ)
  let d2 := (dist P F2: ℝ)
  d1 - d2 = a

theorem trajectory_of_moving_point (F1 F2 : ℝ × ℝ) (a : ℝ) (h_a_pos : a > 0) :
  ∃ P : ℝ × ℝ, locus_of_point_P F1 F2 a P →
  (P ∈ { p : ℝ × ℝ | let d := dist p F1 - dist p F2 in d < dist F1 F2 } ∨
  ∃ l : ℝ × ℝ ∃ r : ℝ, P ∈ { p : ℝ × ℝ | p = l + r • (F2 - F1) }) :=
by
  sorry

end trajectory_of_moving_point_l692_692129


namespace largest_prime_factor_of_1001_l692_692331

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692331


namespace exists_zero_in_interval_l692_692850

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + logBase 2 x

theorem exists_zero_in_interval : ∃ x ∈ (Ioo 0 1), f x = 0 := sorry

end exists_zero_in_interval_l692_692850


namespace positive_solution_for_y_l692_692549

theorem positive_solution_for_y (x y z : ℝ) 
  (h1 : x * y = 4 - x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z)
  (h3 : x * z = 40 - 5 * x - 2 * z) : y = 2 := 
sorry

end positive_solution_for_y_l692_692549


namespace a_is_arithmetic_sequence_general_term_an_l692_692882

-- Definitions for Part 1
variable (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Condition that bm is the sum of the first am terms of the sequence a
def bm_is_sum_of_am_terms (m : ℕ) : Prop :=
  b m = ∑ i in finset.range (a m), (a i)

-- Condition that b is an arithmetic sequence with common difference d
def b_is_arithmetic_sequence (d : ℕ) : Prop :=
  ∀ m, b (m + 1) = b m + d

-- Lean statement for Part 1
theorem a_is_arithmetic_sequence (h1 : ∀ m, b m = ∑ i in finset.range (a m), (a i))
                                 (h2 : ∃ d, ∀ m, b (m + 1) = b m + d) : 
  ∃ c, ∀ n, a (n + 1) = a n + c := 
sorry

-- Definitions for Part 2
-- Condition that bm = (2m - 1)^2
def bm_is_squared_term (m : ℕ) : Prop :=
  b m = (2 * m - 1) ^ 2

-- Lean statement for Part 2
theorem general_term_an (h1 : ∀ m, b m = ∑ i in finset.range (a m), (a i))
                        (h2 : ∀ m, b m = (2 * m - 1) ^ 2) :
  ∀ n, a n = 2 * n - 1 := 
sorry

end a_is_arithmetic_sequence_general_term_an_l692_692882


namespace solve_some_number_l692_692900

theorem solve_some_number (n : ℝ) (h : (n * 10) / 100 = 0.032420000000000004) : n = 0.32420000000000004 :=
by
  -- The proof steps are omitted with 'sorry' here.
  sorry

end solve_some_number_l692_692900


namespace students_arrangement_count_l692_692121

-- Mathematical Definitions Based on the Conditions
def students : Type := ℕ -- Represent students by natural numbers (0,1,2,3,4 stands for students A, B, C, D, E)

def is_together (a b : students) (seq : list students) : Prop :=
  ∃ i, (seq.nth i = some a ∧ seq.nth (i+1) = some b) ∨ (seq.nth i = some b ∧ seq.nth (i+1) = some a)

def not_together (a b : students) (seq : list students) : Prop :=
  ¬ ∃ i, seq.nth i = some a ∧ seq.nth (i+1) = some b

-- Proof Problem Statement
theorem students_arrangement_count : 
  let a := 0
      b := 1
      c := 2
      d := 3
      e := 4 in
  ∀ (seq : list students),
  seq ~ [a, b, c, d, e] ∧ is_together a b seq ∧ not_together c d seq → 
  seq.count = 24 :=
sorry

end students_arrangement_count_l692_692121


namespace handshake_problem_l692_692581

theorem handshake_problem (n : ℕ) (H : (n * (n - 1)) / 2 = 28) : n = 8 := 
sorry

end handshake_problem_l692_692581


namespace unique_fibonacci_representation_l692_692268

theorem unique_fibonacci_representation (n : ℕ) :
  ∃! (k_sequence : List ℕ) (m : ℕ) (k : ℕ → ℕ), 
  (∀ i, i < m → (k_sequence.length = m) ∧ i < k_sequence.length ∧ k i = k_sequence.nth_le i (sorry : i < k_sequence.length)) ∧ 
  (∑ i in List.range m, fib (k i)) = n ∧ 
  (∀ i < m - 1, k i > (k (i+1)) + 1) ∧ 
  (k (m-1) > 1) := 
sorry

end unique_fibonacci_representation_l692_692268


namespace train_speed_l692_692809

theorem train_speed (length_of_train : ℝ) (time_to_cross_pole : ℝ) (calculated_speed : ℝ) :
  length_of_train = 50 ∧ 
  time_to_cross_pole = 0.49996000319974404 ∧ 
  calculated_speed = 100.0080012800512 →
  (length_of_train / time_to_cross_pole) ≈ calculated_speed :=
by
  sorry

end train_speed_l692_692809


namespace fraction_of_pigs_l692_692264

variable (animals cows ducks pigs : ℕ)
variable (fraction_pigs total_ducks_cows : ℚ)

-- Conditions
def conditions := 
  cows = 20 ∧ 
  ducks = cows + cows / 2 ∧
  total_ducks_cows = cows + ducks ∧
  pigs = 60 - total_ducks_cows ∧ 
  fraction_pigs = pigs / total_ducks_cows

-- The theorem to be proved
theorem fraction_of_pigs (h : conditions cows ducks total_ducks_cows pigs) : 
  fraction_pigs = 1 / 5 := 
sorry

end fraction_of_pigs_l692_692264


namespace greatest_monthly_drop_l692_692693

theorem greatest_monthly_drop :
  let price_changes := [("January", -1.00), ("February", 1.50), ("March", -3.00), ("April", 2.00),
                        ("May", -0.75), ("June", 1.00), ("July", -2.50), ("August", -2.00)] in
  (∀ month price, month ∈ ["January", "March", "May", "July", "August"] →
    (month, price) ∈ price_changes →
      price ≤ -3.00) →
  (∀ month price, (month, price) ∈ price_changes →
    month = "March" →
    price = -3.00) := sorry

end greatest_monthly_drop_l692_692693


namespace sample_size_120_l692_692405

theorem sample_size_120
  (x y : ℕ)
  (h_ratio : x / 2 = y / 3 ∧ y / 3 = 60 / 5)
  (h_max : max x (max y 60) = 60) :
  x + y + 60 = 120 := by
  sorry

end sample_size_120_l692_692405


namespace filled_fraction_correct_l692_692436

variable (C : ℝ) (hC : C > 0)

def small_beaker_salt_water : ℝ := C / 2

def large_beaker_capacity : ℝ := 5 * C

def large_beaker_fresh_water : ℝ := large_beaker_capacity / 5

def total_liquid_in_large_beaker : ℝ :=
  large_beaker_fresh_water + small_beaker_salt_water

def filled_fraction : ℝ := total_liquid_in_large_beaker / large_beaker_capacity

theorem filled_fraction_correct :
  filled_fraction C hC = 3 / 10 :=
by
  sorry

end filled_fraction_correct_l692_692436


namespace probability_no_3x3_red_square_l692_692073

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l692_692073


namespace probability_90_100_l692_692963

open MeasureTheory ProbabilityTheory

-- Define the problem conditions and target probability
noncomputable def normal_distribution : ProbabilityDistribution ℝ := 
  ProbabilityDistribution.normal 80 (σ : ℝ)  -- defining the given normal distribution with mean 80 and variance σ² where σ > 0. 

axiom interval_prob_70_90 : ∀ (σ : ℝ) (h : σ > 0), 
  prob {ξ : ℝ | 70 < ξ ∧ ξ < 90} = 0.8   -- the probability that ξ falls in the interval (70, 90) is 0.8.

theorem probability_90_100 : ∀ (σ : ℝ) (h : σ > 0),
  prob {ξ : ℝ | 90 ≤ ξ ∧ ξ ≤ 100} = 0.1 := sorry

end probability_90_100_l692_692963


namespace largest_prime_factor_of_1001_l692_692376

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l692_692376


namespace a_seq_is_M_class_seq_b_seq_is_M_class_seq_a_plus_a_next_is_M_class_a_times_a_next_not_necessarily_M_class_a_seq_custom_not_M_class_S_n_sum_l692_692866

-- Definition of an M-class sequence
def isMClassSequence {α : Type*} [linear_ordered_field α] (c : ℕ → α) (p q : α) : Prop :=
  ∀ n : ℕ, c (n + 1) = p * c n + q

-- Sequences definitions
def a_seq (n : ℕ) : ℕ := 2 * (n + 1)
def b_seq (n : ℕ) : ℕ := 3 * 2^(n + 1)

-- Given conditions: a_n and b_n "M-class" sequences
theorem a_seq_is_M_class_seq : ∃ p q : ℕ, p ≠ 0 ∧ isMClassSequence a_seq p q :=
sorry

theorem b_seq_is_M_class_seq : ∃ p q : ℕ, p ≠ 0 ∧ isMClassSequence b_seq p q :=
sorry

-- Condition: examining sequences derived from a_n
theorem a_plus_a_next_is_M_class (h : ∃ p q : ℕ, p ≠ 0 ∧ isMClassSequence a_seq p q) : ∃ p q : ℕ, p ≠ 0 ∧ isMClassSequence (λ n, a_seq n + a_seq (n + 1)) p q :=
sorry

theorem a_times_a_next_not_necessarily_M_class (h : ∃ p q : ℕ, p ≠ 0 ∧ isMClassSequence a_seq p q) : ¬ ∀ p q : ℕ, p ≠ 0 → isMClassSequence (λ n, a_seq n * a_seq (n + 1)) p q :=
sorry

-- Conditions for sequences defined with given values
def a_seq_custom (n : ℕ) : ℕ :=
if n = 0 then 1 else
  if even n then 2^(n+1) + 1 else 2^(n+1) - 1

def S_n (n : ℕ) : ℕ :=
if even n then 2^(n+1) - 2 else 2^(n+1) - 3

theorem a_seq_custom_not_M_class : ¬ ∃ p q : ℕ, p ≠ 0 ∧ isMClassSequence a_seq_custom p q := 
sorry

theorem S_n_sum : ∀ n, S_n n = ∑ k in range (n + 1), a_seq_custom k :=
sorry

end a_seq_is_M_class_seq_b_seq_is_M_class_seq_a_plus_a_next_is_M_class_a_times_a_next_not_necessarily_M_class_a_seq_custom_not_M_class_S_n_sum_l692_692866


namespace range_of_omega_l692_692151

theorem range_of_omega (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_shift : ∀ x, g(x) = cos (ω * (x + π / (3 * ω)))) 
  (h_monotonic_incr : ∀ x ∈ (2 * π / 3, 4 * π / 3), ∀ y, x < y → y ∈ (2 * π / 3, 4 * π / 3) → g(x) < g(y)) : 
  (1 ≤ ω ∧ ω ≤ 5 / 4) :=
sorry

end range_of_omega_l692_692151


namespace scientific_notation_eight_million_l692_692595

theorem scientific_notation_eight_million :
  ∃ a n, 8000000 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = 6 :=
by
  use 8
  use 6
  sorry

end scientific_notation_eight_million_l692_692595


namespace fraction_of_crop_to_longest_side_is_one_third_l692_692792

noncomputable def trapezoid_fraction_to_longest_side 
  (AB CD BC AD : ℝ) (angle_ABC angle_BCD : ℝ) 
  (h_AB : AB = 150) (h_CD : CD = 250) 
  (h_BC_AD : BC = 200 ∧ AD = 200) 
  (h_angle_ABC : angle_ABC = 120) 
  (h_angle_BCD : angle_BCD = 120) 
  : ℝ :=
  let fraction := 1 / 3 in -- Given from the solution
  fraction

theorem fraction_of_crop_to_longest_side_is_one_third : 
  ∀ (AB CD BC AD angle_ABC angle_BCD : ℝ), 
  AB = 150 → CD = 250 → 
  (BC = 200 ∧ AD = 200) → 
  angle_ABC = 120 → 
  angle_BCD = 120 → 
  trapezoid_fraction_to_longest_side AB CD BC AD angle_ABC angle_BCD 
  = 1 / 3 :=
by
  intros
  unfold trapezoid_fraction_to_longest_side
  sorry

end fraction_of_crop_to_longest_side_is_one_third_l692_692792


namespace range_of_m_l692_692886

theorem range_of_m (x m : ℝ) (h₁ : x^2 - 3 * x + 2 > 0) (h₂ : ¬(x^2 - 3 * x + 2 > 0) → x < m) : 2 < m :=
by
  sorry

end range_of_m_l692_692886


namespace sin_365_1_eq_m_l692_692509

noncomputable def sin_value (θ : ℝ) : ℝ := Real.sin (Real.pi * θ / 180)
variables (m : ℝ) (h : sin_value 5.1 = m)

theorem sin_365_1_eq_m : sin_value 365.1 = m :=
by sorry

end sin_365_1_eq_m_l692_692509


namespace tangent_slope_at_minus_one_l692_692725

-- Conditions:
-- 1. y = ax^3 - 2
-- 2. Slope of the tangent line at x = -1 is 45 degrees
-- Prove: a = 1 / 3

theorem tangent_slope_at_minus_one (a : ℝ) (y : ℝ → ℝ) 
  (h_curve : ∀ x, y x = a * x^3 - 2)
  (h_slope : ∀ x, x = -1 → (deriv y x) = real.tan (real.pi / 4)) : 
  a = 1 / 3 := 
sorry

end tangent_slope_at_minus_one_l692_692725


namespace range_of_a_l692_692877

open Set Real

theorem range_of_a :
  let p := ∀ x : ℝ, |4 * x - 3| ≤ 1
  let q := ∀ x : ℝ, x^2 - (2 * a + 1) * x + (a * (a + 1)) ≤ 0
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q)
  → (∀ x : Icc (0 : ℝ) (1 / 2 : ℝ), a = x) :=
by
  intros
  sorry

end range_of_a_l692_692877


namespace travel_time_l692_692799

theorem travel_time (distance speed : ℕ) (h₀ : distance = 50) (h₁ : speed = 20) : 
  distance / speed = 2.5 := 
by
  sorry

end travel_time_l692_692799


namespace total_calories_consumed_l692_692997

-- Definitions for conditions
def calories_per_chip : ℕ := 60 / 10
def extra_calories_per_cheezit := calories_per_chip / 3
def calories_per_cheezit: ℕ := calories_per_chip + extra_calories_per_cheezit
def total_calories_chips : ℕ := 60
def total_calories_cheezits : ℕ := 6 * calories_per_cheezit

-- Main statement to be proved
theorem total_calories_consumed : total_calories_chips + total_calories_cheezits = 108 := by 
  sorry

end total_calories_consumed_l692_692997


namespace basis_group1_basis_group2_basis_group3_basis_l692_692004

def vector (α : Type*) := α × α

def is_collinear (v1 v2: vector ℝ) : Prop :=
  v1.1 * v2.2 - v2.1 * v1.2 = 0

def group1_v1 : vector ℝ := (-1, 2)
def group1_v2 : vector ℝ := (5, 7)

def group2_v1 : vector ℝ := (3, 5)
def group2_v2 : vector ℝ := (6, 10)

def group3_v1 : vector ℝ := (2, -3)
def group3_v2 : vector ℝ := (0.5, 0.75)

theorem basis_group1 : ¬ is_collinear group1_v1 group1_v2 :=
by sorry

theorem basis_group2 : is_collinear group2_v1 group2_v2 :=
by sorry

theorem basis_group3 : ¬ is_collinear group3_v1 group3_v2 :=
by sorry

theorem basis : (¬ is_collinear group1_v1 group1_v2) ∧ (is_collinear group2_v1 group2_v2) ∧ (¬ is_collinear group3_v1 group3_v2) :=
by sorry

end basis_group1_basis_group2_basis_group3_basis_l692_692004


namespace modulus_of_z_l692_692626

theorem modulus_of_z
  (z : ℂ)
  (h : z * (1 + complex.i) = 1 - complex.i) :
  complex.abs z = 1 :=
by sorry

end modulus_of_z_l692_692626


namespace range_of_a_l692_692542

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a + 3

theorem range_of_a :
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ g x2 a = 0 ∧ |x1 - x2| ≤ 1) ↔ (a ∈ Set.Icc 2 3) := sorry

end range_of_a_l692_692542


namespace arithmetic_sequence_count_l692_692559

theorem arithmetic_sequence_count :
  ∃ n : ℕ, 2 + (n-1) * 5 = 2507 ∧ n = 502 :=
by
  sorry

end arithmetic_sequence_count_l692_692559


namespace annual_percentage_increase_l692_692295

theorem annual_percentage_increase (present_value future_value : ℝ) (years: ℝ) (r : ℝ) 
  (h1 : present_value = 20000)
  (h2 : future_value = 24200)
  (h3 : years = 2) : 
  future_value = present_value * (1 + r)^years → r = 0.1 :=
sorry

end annual_percentage_increase_l692_692295


namespace max_a_in_inequality_l692_692136

theorem max_a_in_inequality (x : ℝ) : 
  (∃ a, ∀ x : ℝ, 2 * x^2 - a * real.sqrt (x^2 + 1) + 3 ≥ 0) → (a ≤ 3) :=
sorry

end max_a_in_inequality_l692_692136


namespace exists_m_lt_0_001_l692_692847

noncomputable def a (n : ℕ) (b : ℕ → ℝ) : ℝ := 
  if n < 101 then 1 else real.sqrt ((finset.range 100).sum (λ j, (b (n-1)) ^ 2) / 100)

noncomputable def b (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  if n < 101 then 1 else real.sqrt ((finset.range 100).sum (λ j, (a (n-1)) ^ 2) / 100)

theorem exists_m_lt_0_001 :
  ∃ m : ℕ, 0 < m ∧ abs (a m b - b m a) < 0.001 :=
begin
  sorry
end

end exists_m_lt_0_001_l692_692847


namespace function_even_l692_692985

def f (x : ℝ) : ℝ := 3^(x^2 - 3) - |x|

theorem function_even : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  -- Definition of f(-x)
  calc
    f (-x) = 3^((-x)^2 - 3) - |-x| : by rfl
    ...     = 3^(x^2 - 3) - |x|     : by sorry    -- The actual equality needs to be shown
  sorry

end function_even_l692_692985


namespace existsNonConvexPolygonWithParallelEqualOppositeSides_l692_692987

noncomputable def isPolygon (vertices : List (ℝ × ℝ)) : Prop :=
  vertices.length ≥ 3 ∧ vertices.head? = vertices.last?

noncomputable def isNonConvex (vertices : List (ℝ × ℝ)) : Prop :=
  ¬(∃ p : ℝ × ℝ, convexHull (set.of_list vertices) = ball p 1)

noncomputable def sidesAreParallelAndEqual (vertices : List (ℝ × ℝ)) : Prop :=
  ∃ (n : ℕ), (list.nth vertices 0).is_some ∧ ∀ i ∈ list.range (vertices.length/2),
    (let (x1, y1) := list.nth_le vertices i (list.nth_le_range i ?m_1)
         (x2, y2) := list.nth_le vertices (i + 1) (list.nth_le_range (i + 1) ?m_1),
         (x3, y3) := list.nth_le vertices (i + (vertices.length / 2)) (list.nth_le_range ((i + (vertices.length / 2))) ?m_1),
         (x4, y4) := list.nth_le vertices (i + 1 + (vertices.length / 2)) (list.nth_le_range ((i + 1 + (vertices.length / 2))) ?m_1)) in
      (x2 - x1) = (x4 - x3) ∧ (y2 - y1) = (y4 - y3) ∧
      sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt ((x4 - x3)^2 + (y4 - y3)^2))

theorem existsNonConvexPolygonWithParallelEqualOppositeSides :
  ∃ vertices : List (ℝ × ℝ), isPolygon vertices ∧ isNonConvex vertices ∧ sidesAreParallelAndEqual vertices :=
by
  -- intentionally skipping the proof
  sorry

end existsNonConvexPolygonWithParallelEqualOppositeSides_l692_692987


namespace range_of_y_eq_frac_3_sin_x_add_1_div_sin_x_add_2_l692_692159

noncomputable def range_of_function : set ℝ :=
  {y : ℝ | ∃ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 ∧ y = (3 * Real.sin x + 1) / (Real.sin x + 2)}

theorem range_of_y_eq_frac_3_sin_x_add_1_div_sin_x_add_2 :
  range_of_function = set.Icc (-2 : ℝ) (4 / 3) :=
begin
  sorry
end

end range_of_y_eq_frac_3_sin_x_add_1_div_sin_x_add_2_l692_692159


namespace school_allocation_methods_l692_692779

-- Define the conditions
def doctors : ℕ := 3
def nurses : ℕ := 6
def schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- The combinatorial function for binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Verify the number of allocation methods
theorem school_allocation_methods : 
  C doctors doctors_per_school * C nurses nurses_per_school *
  C (doctors - 1) doctors_per_school * C (nurses - 2) nurses_per_school *
  C (doctors - 2) doctors_per_school * C (nurses - 4) nurses_per_school = 540 := 
sorry

end school_allocation_methods_l692_692779


namespace polynomial_value_at_n_plus_3_l692_692629

-- Define the polynomial P
noncomputable def P (x : ℕ) : ℕ → ℝ := λ n, if x = n then (2^x) else 0

-- Define the theorem to be proved
theorem polynomial_value_at_n_plus_3 (n : ℕ) : 
  (P (n + 3) n) = 2 * (2^(n + 2) - n - 3) :=
by sorry

end polynomial_value_at_n_plus_3_l692_692629


namespace product_of_divisors_120_30_l692_692110

theorem product_of_divisors_120_30 :
  let divisors_120 : List ℤ := [±1, ±2, ±3, ±4, ±5, ±6, ±8, ±10, ±12, ±15, ±20, ±24, ±30, ±40, ±60, ±120]
  let divisors_30 : List ℤ := [±1, ±2, ±3, ±5, ±6, ±10, ±15, ±30]
  let common_divisors := [±1, ±2, ±3, ±5, ±6, ±10, ±15, ±30]
  List.prod common_divisors = 2^2 * 3^2 * 5^2 * 6^2 * 10^2 * 15^2 * 30^2 := 
by
  sorry

end product_of_divisors_120_30_l692_692110


namespace probability_no_3x3_red_square_l692_692075

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l692_692075


namespace probability_no_3x3_red_square_l692_692076

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l692_692076


namespace john_shots_l692_692446

theorem john_shots :
  let initial_shots := 30
  let initial_percentage := 0.60
  let additional_shots := 10
  let final_percentage := 0.58
  let made_initial := initial_percentage * initial_shots
  let total_shots := initial_shots + additional_shots
  let made_total := final_percentage * total_shots
  let made_additional := made_total - made_initial
  made_additional = 5 :=
by
  sorry

end john_shots_l692_692446


namespace combination_15_choose_3_l692_692945

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l692_692945


namespace sum_m_n_l692_692057

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l692_692057


namespace real_gdp_change_2015_l692_692687

noncomputable def gdp_2014 := 
  (1200 * 90) + (750 * 75)

noncomputable def real_gdp_2015 := 
  (900 * 90) + (900 * 75)

noncomputable def percentage_change := 
  (real_gdp_2015 - gdp_2014) * 100 / gdp_2014

theorem real_gdp_change_2015 : 
  percentage_change = -9.59 := 
sorry

end real_gdp_change_2015_l692_692687


namespace probability_no_3by3_red_grid_correct_l692_692061

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l692_692061


namespace DE_parallel_AC_l692_692261

open Point
open LineSeg
open Triangle
open Parallel

variable (A B C D E : Point)
variable (AD DB CE EB : ℝ)

-- Definitions from the conditions
def triangle_ABC : Triangle := ⟨A, B, C⟩
def point_D_on_AB : Prop := D ∈ LineSeg A B ∧ AD / DB = 2 / 1
def point_E_on_BC : Prop := E ∈ LineSeg B C ∧ CE / EB = 2 / 1

-- The theorem to prove
theorem DE_parallel_AC 
  (h1 : point_D_on_AB A B D AD DB)
  (h2 : point_E_on_BC B C E CE EB)
  : Parallel (LineSeg D E) (LineSeg A C) :=
sorry

end DE_parallel_AC_l692_692261


namespace percent_less_than_R_l692_692292

variables (G R P : ℝ)

-- Conditions
def condition1 : Prop := P = 0.90 * G
def condition2 : Prop := R = 1.20 * G

-- Statement to Prove
theorem percent_less_than_R : condition1 G R P ∧ condition2 G R P → ((R - P) / R * 100 = 25) :=
by
  intros
  sorry

end percent_less_than_R_l692_692292


namespace max_time_to_store_l692_692255

noncomputable def yards_to_feet : ℤ → ℤ := λ y, y * 3
noncomputable def walking_rate (distance : ℤ) (time : ℤ) : ℤ := distance / time
noncomputable def time_for_distance (distance : ℤ) (rate : ℤ) : ℤ := distance / rate

def max_distance_yards := 36
def max_time_minutes := 18

def max_distance_feet := yards_to_feet max_distance_yards
def max_walking_rate := walking_rate max_distance_feet max_time_minutes

def remaining_distance := 120

theorem max_time_to_store : 
  time_for_distance remaining_distance max_walking_rate = 20 := 
by
  -- Proof omitted
  sorry

end max_time_to_store_l692_692255


namespace range_of_k_l692_692046

noncomputable def f : ℝ → ℝ :=
sorry

theorem range_of_k 
  (h1 : ∀ x : ℝ, f (f x + x^3) = 2)
  (h2 : ∀ x : ℝ, f'' x ≠ 0)
  (h3 : ∀ k ∈ Icc 0 +∞, ∀ x ∈ Icc -1 1, (f x - k * x)' = f' x) : true → false :=
  sorry

end range_of_k_l692_692046


namespace expected_length_first_group_19_49_l692_692715

noncomputable def expected_first_group_length (ones zeros : ℕ) : ℕ :=
  let total := ones + zeros;
  let pI := 1 / 50;
  let pJ := 1 / 20;
  let expected_I := ones * pI;
  let expected_J := zeros * pJ;
  expected_I + expected_J

theorem expected_length_first_group_19_49 : expected_first_group_length 19 49 = 2.83 :=
by
  sorry

end expected_length_first_group_19_49_l692_692715


namespace expected_length_first_group_l692_692713

noncomputable def indicator_prob (n : ℕ) : ℚ :=
if n = 1 then 1/50 else 1/20

theorem expected_length_first_group (ones zeros : ℕ) (h : ones = 19) (h2 : zeros = 49) : 
  let X := ∑ i in (finset.range ones ∪ finset.range zeros), (indicator_prob (i + 1)) in
  (X : ℝ) = 2.83 :=
sorry

end expected_length_first_group_l692_692713


namespace robert_cash_spent_as_percentage_l692_692664

theorem robert_cash_spent_as_percentage 
  (raw_material_cost : ℤ) (machinery_cost : ℤ) (total_amount : ℤ) 
  (h_raw : raw_material_cost = 100) 
  (h_machinery : machinery_cost = 125) 
  (h_total : total_amount = 250) :
  ((total_amount - (raw_material_cost + machinery_cost)) * 100 / total_amount) = 10 := 
by 
  -- Proof will be filled here
  sorry

end robert_cash_spent_as_percentage_l692_692664


namespace guesthouse_assignment_count_l692_692433

theorem guesthouse_assignment_count :
  let rooms := 6
  let friends := 6
  let maxPerRoom := 3
  let roomsWithThreeGuests := 2
  ∃ assignments : Nat, assignments = 1800 :=
by
  let binomial (n k : Nat) : Nat := Nat.choose n k
  let factorial (n : Nat) : Nat := Nat.factorial n

  have totalAssignments : Nat :=
    binomial 6 2 * binomial 6 3 * binomial 3 3 * factorial 3

  exists totalAssignments
  have h : totalAssignments = 1800 := by
    simp [binomial, factorial]
    rfl

  exact h

end guesthouse_assignment_count_l692_692433


namespace triangle_not_necessarily_isosceles_l692_692986

theorem triangle_not_necessarily_isosceles (A B C O M N : Point)
    (h₁ : is_triangle A B C) 
    (h₂ : is_incenter O A B C) 
    (h₃ : is_midpoint M (segment A B)) 
    (h₄ : is_midpoint N (segment A C))
    (h₅ : dist O M = dist O N) : 
    ¬ is_isosceles A B C := 
sorry

end triangle_not_necessarily_isosceles_l692_692986


namespace total_comics_in_box_l692_692020

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l692_692020


namespace probability_at_least_one_black_ball_l692_692196

theorem probability_at_least_one_black_ball
  (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (selected_balls : ℕ)
  (white_balls = 2) (black_balls = 2) (total_balls = 4) (selected_balls = 2)
  : (∃ (prob : ℚ), prob = 5 / 6) :=
sorry

end probability_at_least_one_black_ball_l692_692196


namespace total_number_of_sweets_l692_692741

theorem total_number_of_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) (total_sweets : ℕ) 
  (h1 : num_crates = 4) (h2 : sweets_per_crate = 16) : total_sweets = 64 := by
  sorry

end total_number_of_sweets_l692_692741


namespace vector_magnitude_proof_l692_692916

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_proof (m : ℝ) (h : magnitude (vector_a.1 + vector_b(m).1, vector_a.2 + vector_b(m).2) = magnitude (vector_a.1 - vector_b(m).1, vector_a.2 - vector_b(m).2)) :
  magnitude (1 + 2 * -2, 2 + 2 * m) = 5 :=
by
  sorry

end vector_magnitude_proof_l692_692916


namespace initial_overs_played_l692_692968

-- Define the conditions
def initial_run_rate : ℝ := 6.2
def remaining_overs : ℝ := 40
def remaining_run_rate : ℝ := 5.5
def target_runs : ℝ := 282

-- Define what we seek to prove
theorem initial_overs_played :
  ∃ x : ℝ, (6.2 * x) + (5.5 * 40) = 282 ∧ x = 10 :=
by
  sorry

end initial_overs_played_l692_692968


namespace circumcenter_CEF_lies_on_ω_l692_692736

-- Definitions for given conditions
variables {A B C D E F O : Point}
variables {ω : Circle}
variables (H1: Trapezoid ABCD) (H2: Inscribed ABCD ω)
variables (H3: PointOnRayBeyond E C D) (H4: BC = BE)
variables (H5: IntersectsAt BE ω F) (H6: Isosceles BEC)

-- The theorem statement
theorem circumcenter_CEF_lies_on_ω :
  CenterOfCircumcircle C E F = O ∧ PointOnCircle O ω :=
sorry

end circumcenter_CEF_lies_on_ω_l692_692736


namespace tan_C_l692_692577

variable (A B C : ℝ)

def tan_condition_1 : Prop := Real.tan A = 1
def tan_condition_2 : Prop := Real.tan B = 2
def triangle_angles : Prop := A + B + C = Real.pi

theorem tan_C (h1 : tan_condition_1) (h2 : tan_condition_2) (h3 : triangle_angles) : Real.tan C = 3 :=
sorry

end tan_C_l692_692577


namespace determinant_of_non_right_triangle_l692_692240

theorem determinant_of_non_right_triangle (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
(h_sum_ABC : A + B + C = π) :
  Matrix.det ![
    ![2 * Real.sin A, 1, 1],
    ![1, 2 * Real.sin B, 1],
    ![1, 1, 2 * Real.sin C]
  ] = 2 := by
  sorry

end determinant_of_non_right_triangle_l692_692240


namespace center_circumcircle_CEF_on_circle_omega_l692_692737

open Set
open Finset
open Int

/-- Formalize the given geometric problem in Lean -/
theorem center_circumcircle_CEF_on_circle_omega
  (A B C D E F : Type)
  [IncidenceGeometry A B C D E F]  -- assuming an incidence geometry instance
  (trapezoid_ABCD : trapezoid A B C D)
  (inscribed_circle_omega : inscribed_circle (trapezoid A B C D))
  (ray_DC : ray D C beyond C)
  (point_E : point_on_ray E ray_DC)
  (BC_BE_eq : length B C = length B E)
  (circ_intersection_point_F : circle_intersection_point B E omega F outside_BE)
  : lies_on_circle (circumcenter (triangle C E F)) omega := 
sorry

end center_circumcircle_CEF_on_circle_omega_l692_692737


namespace expected_length_of_first_group_l692_692722

-- Define the conditions of the problem
def sequence : Finset ℕ := {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

-- Expected length of the first group
def expected_length_first_group : ℝ := 2.83

-- The formal statement of the proof problem
theorem expected_length_of_first_group (seq : Finset ℕ) (h1 : seq.card = 68) (h2 : seq.filter (λ x, x = 1) = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
(h3 : seq.filter (λ x, x = 0) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) :
  let X := 
  (Finset.sum (Finset.range 19) (λ k, if k = 1 then (1/50 : ℝ) else 0)) +
  (Finset.sum (Finset.range 49) (λ m, if m = 1 then (1/20 : ℝ) else 0)) in 
  ∃ x : ℝ, ∑ x = expected_length_first_group := 
by sorry
 
end expected_length_of_first_group_l692_692722


namespace largest_prime_factor_of_1001_l692_692330

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l692_692330


namespace average_visitors_l692_692767

theorem average_visitors (visitors_sunday visitors_otherdays : ℕ) (days_month sundays otherdays : ℕ) :
  visitors_sunday = 510 → visitors_otherdays = 240 →
  days_month = 30 → sundays = 5 → otherdays = 25 →
  (sundays * visitors_sunday + otherdays * visitors_otherdays) / days_month = 285 :=
by
  intros
  rw [←h, ←h_1, ←h_2, ←h_3, ←h_4]
  sorry

end average_visitors_l692_692767


namespace coordinates_at_5PM_l692_692424

noncomputable def particle_coords_at_5PM : ℝ × ℝ :=
  let t1 : ℝ := 7  -- 7 AM
  let t2 : ℝ := 9  -- 9 AM
  let t3 : ℝ := 17  -- 5 PM in 24-hour format
  let coord1 : ℝ × ℝ := (1, 2)
  let coord2 : ℝ × ℝ := (3, -2)
  let dx : ℝ := (coord2.1 - coord1.1) / (t2 - t1)
  let dy : ℝ := (coord2.2 - coord1.2) / (t2 - t1)
  (coord2.1 + dx * (t3 - t2), coord2.2 + dy * (t3 - t2))

theorem coordinates_at_5PM
  (t1 t2 t3 : ℝ)
  (coord1 coord2 : ℝ × ℝ)
  (h_t1 : t1 = 7)
  (h_t2 : t2 = 9)
  (h_t3 : t3 = 17)
  (h_coord1 : coord1 = (1, 2))
  (h_coord2 : coord2 = (3, -2))
  (h_dx : (coord2.1 - coord1.1) / (t2 - t1) = 1)
  (h_dy : (coord2.2 - coord1.2) / (t2 - t1) = -2)
  : particle_coords_at_5PM = (11, -18) :=
by
  sorry

end coordinates_at_5PM_l692_692424


namespace recycling_points_l692_692465

-- Define the statement
theorem recycling_points : 
  ∀ (C H L I : ℝ) (points_per_six_pounds : ℝ), 
  C = 28 → H = 4.5 → L = 3.25 → I = 8.75 → points_per_six_pounds = 1 / 6 →
  (⌊ C * points_per_six_pounds ⌋ + ⌊ I * points_per_six_pounds ⌋  + ⌊ H * points_per_six_pounds ⌋ + ⌊ L * points_per_six_pounds ⌋ = 5) :=
by
  intros C H L I pps hC hH hL hI hpps
  rw [hC, hH, hL, hI, hpps]
  simp
  sorry

end recycling_points_l692_692465


namespace polynomial_proof_l692_692880

theorem polynomial_proof (x : ℝ) : 
  (2 * x^2 + 5 * x + 4) = (2 * x^2 + 5 * x - 2) + (10 * x + 6) :=
by sorry

end polynomial_proof_l692_692880


namespace product_of_roots_l692_692171

theorem product_of_roots :
  ∀ x : ℝ, (x + 3) * (x - 5) = 22 → let roots := {r : ℝ | ∃ y : ℝ, (r = y) ∧ ((y + 3) * (y - 5) = 22)} in
    roots = {-37} :=
begin
  sorry
end

end product_of_roots_l692_692171


namespace estimate_47_times_20_estimate_744_div_6_l692_692856

-- Definitions for conditions in a)
noncomputable def estimated_multiplication (a b : ℕ) : ℕ :=
  let nearest_ten (n : ℕ) := (n + 5) / 10 * 10 in
  nearest_ten a * nearest_ten b

noncomputable def estimated_division (dividend divisor : ℕ) : ℕ :=
  let nearest_hundred_multiple (n m : ℕ) := (n / m * m) in
  nearest_hundred_multiple dividend divisor / divisor

-- Statements rewriting the math proof problems in Lean 4
theorem estimate_47_times_20 : estimated_multiplication 47 20 = 1000 := by
  sorry

theorem estimate_744_div_6 : estimated_division 744 6 = 120 := by
  sorry

end estimate_47_times_20_estimate_744_div_6_l692_692856


namespace cost_of_banana_l692_692249

theorem cost_of_banana (B : ℝ) (apples bananas oranges total_pieces total_cost : ℝ) 
  (h1 : apples = 12) (h2 : bananas = 4) (h3 : oranges = 4) 
  (h4 : total_pieces = 20) (h5 : total_cost = 40)
  (h6 : 2 * apples + 3 * oranges + bananas * B = total_cost)
  : B = 1 :=
by
  sorry

end cost_of_banana_l692_692249


namespace sinA_value_triangle_area_l692_692578

-- Definitions of the given variables
variables (A B C : ℝ)
variables (a b c : ℝ)
variables (sinA sinC cosC : ℝ)

-- Given conditions
axiom h_c : c = Real.sqrt 2
axiom h_a : a = 1
axiom h_cosC : cosC = 3 / 4
axiom h_sinC : sinC = Real.sqrt 7 / 4
axiom h_b : b = 2

-- Question 1: Prove sin A = sqrt 14 / 8
theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
sorry

-- Question 2: Prove the area of triangle ABC is sqrt 7 / 4
theorem triangle_area : 1/2 * a * b * sinC = Real.sqrt 7 / 4 :=
sorry

end sinA_value_triangle_area_l692_692578


namespace line_through_point_satisfies_eq_line_y_intercept_satisfies_eq_l692_692495

open Real

noncomputable theory

def line1 (x y : ℝ) : Prop := sqrt 3 * x - y + 4 * (sqrt 3) + 1 = 0
def line2 (x y : ℝ) : Prop := y = sqrt 3 * x - 10

theorem line_through_point_satisfies_eq : 
  ∀ (x y : ℝ), 
  (x, y) = (-4, 1) → line1 x y :=
by 
  intros x y h
  rw [prod.mk.inj_iff] at h
  cases h with h1 h2
  rw [h1, h2]
  sorry

theorem line_y_intercept_satisfies_eq : 
  ∀ (x y : ℝ), 
  x = 0 → y = -10 → line2 x y :=
by 
  intros x y hx hy
  rw [hx, hy]
  sorry

end line_through_point_satisfies_eq_line_y_intercept_satisfies_eq_l692_692495


namespace sequence_not_expressible_as_3_alpha_5_beta_l692_692530

-- Define the sequence {v_n}
def v : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 1) := 8 * v n - v (n - 1)

-- Define the theorem statement
theorem sequence_not_expressible_as_3_alpha_5_beta : 
  ∀ n : ℕ, ∀ α : ℕ, ∀ β : ℕ, α > 0 ∧ β > 0 → v n ≠ 3 ^ α * 5 ^ β := 
by
  intro n α β h
  sorry

end sequence_not_expressible_as_3_alpha_5_beta_l692_692530


namespace min_abs_z_plus_i_l692_692234

theorem min_abs_z_plus_i (z : ℂ) (h: |z^2 + 9| = |z * (z + 3 * complex.I)|) : ∃ w : ℂ, z = -3 * complex.I → |w + complex.I| = 2 :=
by
  sorry

end min_abs_z_plus_i_l692_692234


namespace price_of_205_tickets_l692_692677

open_locale classical

theorem price_of_205_tickets (P1 P2 : ℝ) (h1 : 205 * P1 + 175 * P2 = 1972.50) : 
  ∃ P1 P2, 205 * P1 + 175 * P2 = 1972.50 :=
begin
  use [P1, P2],
  exact h1,
end

end price_of_205_tickets_l692_692677


namespace expected_length_first_group_l692_692712

noncomputable def indicator_prob (n : ℕ) : ℚ :=
if n = 1 then 1/50 else 1/20

theorem expected_length_first_group (ones zeros : ℕ) (h : ones = 19) (h2 : zeros = 49) : 
  let X := ∑ i in (finset.range ones ∪ finset.range zeros), (indicator_prob (i + 1)) in
  (X : ℝ) = 2.83 :=
sorry

end expected_length_first_group_l692_692712


namespace remainder_when_divided_by_11_l692_692742

theorem remainder_when_divided_by_11 :
  ∃ (xs : List ℕ), (∀ x ∈ xs, 0 ≤ x ∧ x ≤ 50) ∧ (xs.length = 5) 
  ∧ (∀ x ∈ xs, x % 11 = 1) :=
by
  let xs := [1, 12, 23, 34, 45]
  existsi xs
  split
  { intros x hx
    simp at hx
    cases hx <|> cases hx <|> cases hx <|> cases hx <|> cases hx
    all_goals { norm_num [hx] } }
  split
  { norm_num [length] }
  { intros x hx
    simp at hx
    cases hx <|> cases hx <|> cases hx <|> cases hx <|> cases hx
    all_goals { norm_num [hx] } }

end remainder_when_divided_by_11_l692_692742


namespace largest_at_least_l692_692914

theorem largest_at_least (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 8) : 
  ∃ x ∈ {a, b, c}, x ≥ 2 * real.cbrt 4 := sorry

end largest_at_least_l692_692914


namespace probability_no_3x3_red_square_l692_692068

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l692_692068


namespace func_eq_condition_l692_692181

variable (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := c * x + d

theorem func_eq_condition (h : a = 2 * c) : (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ c = 1 / 2) :=
by sorry

end func_eq_condition_l692_692181


namespace orthocenter_identity_l692_692517

variable {ABC : Type*} [Triangle ABC]
variable {H : Point}
variable {A B C : Point}
variable {a b c x y z : ℝ}

-- Conditions for the acute triangle and the orthocenter
axiom acute_triangle : ∀ {A B C : Point}, IsAcuteTriangle A B C
axiom orthocenter : IsOrthocenter H A B C

-- Given side lengths
axiom side_lengths : ∀ {A B C : Point}, 
  (segment_length A B = c) ∧ 
  (segment_length B C = a) ∧ 
  (segment_length C A = b)

-- Given segment lengths from vertices to the orthocenter
axiom orthocenter_segment_lengths : ∀ {A B C H : Point}, 
  (segment_length A H = x) ∧ 
  (segment_length B H = y) ∧ 
  (segment_length C H = z)

-- The theorem statement
theorem orthocenter_identity :
  ∀ {A B C H : Point}
  {a b c x y z : ℝ},
  IsAcuteTriangle A B C →
  IsOrthocenter H A B C →
  segment_length A B = c →
  segment_length B C = a →
  segment_length C A = b →
  segment_length A H = x →
  segment_length B H = y →
  segment_length C H = z →
  a * y * z + b * z * x + c * x * y = a * b * c := sorry

end orthocenter_identity_l692_692517


namespace largest_prime_factor_1001_l692_692347

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692347


namespace number_of_students_paper_C_l692_692802

-- Define the arithmetic sequence with first term 12 and common difference 15.
def arithmetic_seq (n : ℕ) : ℕ := 15 * n - 3

-- Define the range of students tested for paper C (those with numbers ≥ 331 and ≤ 600)
def in_paper_C_range (k : ℕ) : Prop := 331 ≤ k ∧ k ≤ 600

-- Define the total number of students
def total_students : ℕ := 600

-- Define the sample size
def sample_size : ℕ := 40

-- Define the condition for students taking paper C to be within given n values
def within_n_range (n : ℕ) : Prop :=
  23 ≤ n ∧ n ≤ 40

-- The main theorem stating that the number of students taking test paper C is 18
theorem number_of_students_paper_C : 
  (finset.card (finset.filter (λ n, within_n_range n) (finset.range sample_size))) = 18 := 
sorry

end number_of_students_paper_C_l692_692802


namespace bob_corn_calc_l692_692456

noncomputable def bob_corn_left (initial_bushels : ℕ) (ears_per_bushel : ℕ) (bushels_taken_by_terry : ℕ) (bushels_taken_by_jerry : ℕ) (bushels_taken_by_linda : ℕ) (ears_taken_by_stacy : ℕ) : ℕ :=
  let initial_ears := initial_bushels * ears_per_bushel
  let ears_given_away := (bushels_taken_by_terry + bushels_taken_by_jerry + bushels_taken_by_linda) * ears_per_bushel + ears_taken_by_stacy
  initial_ears - ears_given_away

theorem bob_corn_calc :
  bob_corn_left 50 14 8 3 12 21 = 357 :=
by
  sorry

end bob_corn_calc_l692_692456


namespace projection_of_a_on_c_l692_692918

noncomputable def vector_proj (a b c : Vector ℝ 2) (lambda : ℝ) : ℝ :=
  (a.1 * c.1 + lambda * c.2) / (c.1^2 + c.2^2)

theorem projection_of_a_on_c 
  (λ : ℝ)
  (a b c : Vector ℝ 2)
  (h1 : a = (1, λ))
  (h2 : b = (3, 1))
  (h3 : c = (1, 2))
  (h4 : ∃ k : ℝ, (2 * (1, λ) - (3, 1)) = k * (1, 2)) :
  vector_proj (1, λ) (3, 1) (1, 2) λ = 0 := sorry

end projection_of_a_on_c_l692_692918


namespace phoebe_dog_peanut_butter_l692_692659

-- Definitions based on the conditions
def servings_per_jar : ℕ := 15
def jars_needed : ℕ := 4
def days : ℕ := 30

-- Problem statement
theorem phoebe_dog_peanut_butter :
  (jars_needed * servings_per_jar) / days / 2 = 1 :=
by sorry

end phoebe_dog_peanut_butter_l692_692659


namespace camphor_decay_l692_692407

theorem camphor_decay (a k : ℝ) (h₁ : a > 0) (h₂ : k > 0)
  (h₃ : a * real.exp (-50 * k) = (4 / 9) * a) : 
  ∃ t : ℝ, a * real.exp (-k * t) = (8 / 27) * a ∧ t = 75 :=
begin
  sorry
end

end camphor_decay_l692_692407


namespace jennie_total_rental_cost_l692_692684

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end jennie_total_rental_cost_l692_692684


namespace brown_gumdrops_after_replacement_l692_692794

theorem brown_gumdrops_after_replacement
  (total_gumdrops : ℕ)
  (percent_blue : ℚ)
  (percent_brown : ℚ)
  (percent_red : ℚ)
  (percent_yellow : ℚ)
  (num_green : ℕ)
  (replace_half_blue_with_brown : ℕ) :
  total_gumdrops = 120 →
  percent_blue = 0.30 →
  percent_brown = 0.20 →
  percent_red = 0.15 →
  percent_yellow = 0.10 →
  num_green = 30 →
  replace_half_blue_with_brown = 18 →
  ((percent_brown * ↑total_gumdrops) + replace_half_blue_with_brown) = 42 :=
by sorry

end brown_gumdrops_after_replacement_l692_692794


namespace coverable_hook_l692_692476

def is_coverable (m n : ℕ) : Prop :=
  ∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5)

theorem coverable_hook (m n : ℕ) : (∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5))
  ↔ is_coverable m n :=
by
  sorry

end coverable_hook_l692_692476


namespace third_step_eq_945_l692_692297

theorem third_step_eq_945 :
  let step1 := 3 * 5,
      step2 := step1 * 7,
      step3 := step2 * 9
  in step3 = 945 :=
by
  let step1 := 3 * 5
  let step2 := step1 * 7
  let step3 := step2 * 9
  show step3 = 945
  sorry

end third_step_eq_945_l692_692297


namespace angle_ECF_45_l692_692453

variable {A B C E F : Type}
variables [Angle ACB ABC ECF : ℝ]

axiom angle_ACB_90 : angle ACB = 90
axiom AC_eq_AE : AC = AE
axiom BC_eq_BF : BC = BF

theorem angle_ECF_45 : angle ECF = 45 :=
by
  sorry

end angle_ECF_45_l692_692453


namespace hyperbola_eccentricity_is_2_l692_692894

variables (a b c : ℝ) (x y : ℝ)
variable (P : EuclideanSpace ℝ (Fin 2)) -- point P in R^2 space

-- Define the hyperbola
def is_on_hyperbola (P : EuclideanSpace ℝ (Fin 2)) : Prop :=
    let ⟨x, y⟩ := P in
    (x / a)^2 - (y / b)^2 = 1

-- Define the condition for point M
def is_on_line (M : EuclideanSpace ℝ (Fin 2)) : Prop :=
    let ⟨x, _⟩ := M in
    x = -a^2 / c

-- Define the focus F
def is_focus (F : EuclideanSpace ℝ (Fin 2)) : Prop := 
    let ⟨x, y⟩ := F in
    x = c ∧ y = 0

-- Define the vector conditions
variable (O : EuclideanSpace ℝ (Fin 2)) -- Origin
def vector_condition1 (P F M : EuclideanSpace ℝ (Fin 2)) : Prop :=
    P - O = F - O + M - O 

def vector_condition2 (P F M : EuclideanSpace ℝ (Fin 2)) : Prop :=
    inner (P - O) (M - F) = 0

-- Prove that the eccentricity is 2
theorem hyperbola_eccentricity_is_2 
    (hp : is_on_hyperbola P) 
    (hm : ∃ M, is_on_line M) 
    (hf : ∃ F, is_focus F) 
    (vc1 : ∃ F M, vector_condition1 P F M) 
    (vc2 : ∃ F M, vector_condition2 P F M) : 
    let c := 2 * a in
    c / a = 2 := 
sorry

end hyperbola_eccentricity_is_2_l692_692894


namespace number_of_groups_of_three_books_l692_692939

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l692_692939


namespace license_plate_combinations_l692_692169

theorem license_plate_combinations : 
  let letters := 26 * 25 * 24 * 23,
      digits := 8 * 7 
  in letters * digits = 8128320 := 
by
  let letters := 26 * 25 * 24 * 23
  let digits := 8 * 7
  show letters * digits = 8128320
  sorry

end license_plate_combinations_l692_692169


namespace payment_difference_l692_692410

theorem payment_difference (average_payment : ℝ) (num_payments : ℕ) (first_payment : ℝ) (num_first_payments : ℕ) (num_second_payments : ℕ) :
    average_payment = 442.5 ∧ num_payments = 40 ∧ first_payment = 410 ∧ num_first_payments = 20 ∧ num_second_payments = 20 → 
    let total_first_payments := num_first_payments * first_payment in
    let total_second_payments := num_second_payments * (first_payment + x) in
    let total_payments := num_payments * average_payment in
    20 * 410 + 20 * (410 + x) = 40 * 442.5 → 
    x = 65 :=
begin
  sorry
end

end payment_difference_l692_692410


namespace coeff_of_term_equal_three_l692_692566

theorem coeff_of_term_equal_three (x : ℕ) (h : x = 13) : 
    2^x - 2^(x - 2) = 3 * 2^(11) :=
by
    rw [h]
    sorry

end coeff_of_term_equal_three_l692_692566


namespace medal_awarding_ways_l692_692310

def num_sprinters := 10
def num_americans := 4
def num_kenyans := 2
def medal_positions := 3 -- gold, silver, bronze

-- The main statement to be proven
theorem medal_awarding_ways :
  let ways_case1 := 2 * 3 * 5 * 4
  let ways_case2 := 4 * 3 * 2 * 2 * 5
  ways_case1 + ways_case2 = 360 :=
by
  sorry

end medal_awarding_ways_l692_692310


namespace center_circumcircle_CEF_on_circle_omega_l692_692739

open Set
open Finset
open Int

/-- Formalize the given geometric problem in Lean -/
theorem center_circumcircle_CEF_on_circle_omega
  (A B C D E F : Type)
  [IncidenceGeometry A B C D E F]  -- assuming an incidence geometry instance
  (trapezoid_ABCD : trapezoid A B C D)
  (inscribed_circle_omega : inscribed_circle (trapezoid A B C D))
  (ray_DC : ray D C beyond C)
  (point_E : point_on_ray E ray_DC)
  (BC_BE_eq : length B C = length B E)
  (circ_intersection_point_F : circle_intersection_point B E omega F outside_BE)
  : lies_on_circle (circumcenter (triangle C E F)) omega := 
sorry

end center_circumcircle_CEF_on_circle_omega_l692_692739


namespace Mary_seashells_l692_692644

variable {S M J : ℕ}

-- Condition: Jessica found 41 seashells
def foundJessica (J : ℕ) : Prop := J = 41

-- Condition: Together Mary and Jessica found 59 seashells
def foundTogether (M J : ℕ) : Prop := M + J = 59

-- Statement: Prove the number of seashells Mary found
theorem Mary_seashells (M J : ℕ) (hJ : foundJessica J) (hT : foundTogether M J) : M = 18 := by
  rw [foundJessica, foundTogether] at *
  rw [hJ]
  simp at hT
  exact hT

end Mary_seashells_l692_692644


namespace fraction_of_beans_remaining_l692_692394

variables (J B R : ℝ)

-- Given conditions
def condition1 : Prop := J = 0.10 * (J + B)
def condition2 : Prop := J + R = 0.60 * (J + B)

theorem fraction_of_beans_remaining (h1 : condition1 J B) (h2 : condition2 J B R) :
  R / B = 5 / 9 :=
  sorry

end fraction_of_beans_remaining_l692_692394


namespace jeremy_money_left_l692_692217

theorem jeremy_money_left (computer_cost : ℕ) (accessories_percentage : ℕ) (factor : ℕ)
  (h1 : computer_cost = 3000)
  (h2 : accessories_percentage = 10)
  (h3 : factor = 2) :
  let accessories_cost := (accessories_percentage * computer_cost) / 100 in
  let total_money_before := factor * computer_cost in
  let total_spent := computer_cost + accessories_cost in
  let money_left := total_money_before - total_spent in
  money_left = 2700 :=
by
  sorry

end jeremy_money_left_l692_692217


namespace rhombus_area_l692_692844

theorem rhombus_area (A B C D M N : ℝ)
  (h₁ : ∃ (AB : ℝ), AB = 4)
  (h₂ : ∃ (BC : ℝ), BC = 8)
  (h₃ : ∃ (x : ℝ), 4^2 + (8 - x)^2 = x^2)
  (h₄ : ∀ {B M}, B ≠ M)
  (h₅ : ∀ {D N}, D ≠ N)
  (h₆ : ∀ (x : ℝ), 80 - 16 * x = 0 → x = 5) :
  (4 * 5 = 20) :=
by {
  sorry
}

end rhombus_area_l692_692844


namespace geom_seq_identity_l692_692979

theorem geom_seq_identity {α : Type*} [CommSemiring α] 
  (a : ℕ → α) (n : ℕ) (h_geom : ∀ i j, a i * a j = a (i + j) * a (n - (i + j))) :
  (a 1 * a n) ^ 2 - a 2 * a 4 * a (n - 1) * a (n - 3) = 0 :=
by
  sorry

end geom_seq_identity_l692_692979


namespace percentage_deficit_is_correct_l692_692586

noncomputable def percentage_deficit (L W : ℝ) : ℝ :=
  let A := L * W
  let L' := L * 1.06
  let W' := W * (1 - x / 100)
  let A' := L' * W'
  let epsilon := 0.007 
  have error_eq : A' = A * (1 + epsilon), from sorry,
  x / 100 = 1 - (1 + epsilon) / 1.06

theorem percentage_deficit_is_correct
   (L W : ℝ)
   (h6 : L' = L * 1.06)
   (hA' : A' = A * 1.007) :
   percentage_deficit L W = 4.95 := 
by 
  sorry

end percentage_deficit_is_correct_l692_692586


namespace part_a_part_b_l692_692669

def fibonacci : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem part_a (n : ℤ) : 
∃ x y : ℤ, x = fibonacci (2 * n + 1) ∧ y = fibonacci (2 * n) ∧ x^2 - x * y - y^2 = 1 :=
sorry

theorem part_b (n : ℤ) : 
∃ x y : ℤ, x = fibonacci (2 * n) ∧ y = fibonacci (2 * n - 1) ∧ x^2 - x * y - y^2 = -1 :=
sorry

end part_a_part_b_l692_692669


namespace smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l692_692434

/-- Define what it means for a number to be a prime greater than 3 -/
def is_prime_gt_3 (n : ℕ) : Prop :=
  Prime n ∧ 3 < n

/-- Define a scalene triangle with side lengths that are distinct primes greater than 3 -/
def is_scalene_triangle_with_distinct_primes (a b c : ℕ) : Prop :=
  is_prime_gt_3 a ∧ is_prime_gt_3 b ∧ is_prime_gt_3 c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The proof problem statement -/
theorem smallest_possible_perimeter_of_scalene_triangle_with_prime_sides :
  ∃ (a b c : ℕ), is_scalene_triangle_with_distinct_primes a b c ∧ Prime (a + b + c) ∧ (a + b + c = 23) :=
sorry

end smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l692_692434


namespace ratio_AC_AB_l692_692010

-- Definitions for the geometric entities involved
def circle (center : Point) (radius : ℝ) := { p : Point | dist center p = radius }

variable (O1 O2 : Point)
variable (P A B C : Point)
variable (R1 R2 : ℝ)

-- Conditions from the problem
def condition_tangent_at_P := P ∈ circle O1 R1 ∧ P ∈ circle O2 R2
def condition_point_A_on_circle_O1 := A ∈ circle O1 R1
def condition_tangent_AB := is_tangent A B O2
def condition_point_C := C ∈ circle O2 R2 ∧ lies_on_line AP C
def condition_radii := R1 = 2 ∧ R2 = 1

-- The mathematical goal
theorem ratio_AC_AB 
    (h1 : condition_tangent_at_P O1 O2 R1 R2 P)
    (h2 : condition_point_A_on_circle_O1 O1 R1 A)
    (h3 : condition_tangent_AB A B O2)
    (h4 : condition_point_C AP C)
    (h5 : condition_radii R1 R2) :
    dist A C / dist A B = Real.sqrt 6 / 2 :=
sorry

end ratio_AC_AB_l692_692010


namespace slope_of_line_slope_sum_eq_52_l692_692038

/-
Define the vertices of the parallelogram
-/
def vertex1 := (2, 5)
def vertex2 := (2, 23)
def vertex3 := (7, 38)
def vertex4 := (7, 20)

/-
Define the requirement that a line through the origin cuts the parallelogram into two congruent polygons,
and that the slope of this line is expressed as m/n where m and n are relatively prime positive integers.
-/
theorem slope_of_line (m n : ℕ) (h_rel_prime : Nat.coprime m n) : 
    ∃ k : ℚ, k = m / n ∧ (2 * k = (5 + 41 / 9) / 2 ) ∧ (7 * k = (38 - 41 / 9) / 7 ) := sorry


/-
Prove that m + n = 52 for the line described above
-/
theorem slope_sum_eq_52 : 
  ∃ (m n : ℕ), Nat.coprime m n ∧ (m + n = 52) ∧ (∃ k : ℚ, k = m / n 
  ∧ (2 * k = (5 + 41 / 9) / 2 ) ∧ (7 * k = (38 - 41 / 9) / 7 )) := sorry

end slope_of_line_slope_sum_eq_52_l692_692038


namespace cannot_be_perfect_square_l692_692957

theorem cannot_be_perfect_square (n : ℕ) (h1 : n.digits 10.sum = 2006) : ¬ ∃ m : ℕ, n = m * m := 
sorry

end cannot_be_perfect_square_l692_692957


namespace units_digit_S_54321_l692_692238

/-- 
  Define the sequence S_n based on the given recurrence relation
  and initial conditions 
-/
def S : ℕ → ℤ
| 0     := 1
| 1     := 4
| (n+2) := 8 * S (n+1) - 4 * S n

/-- 
  The goal is to prove the units digit of S_54321.
-/
theorem units_digit_S_54321 : (S 54321) % 10 = 4 :=
sorry

end units_digit_S_54321_l692_692238


namespace polynomial_real_root_exists_l692_692859

theorem polynomial_real_root_exists (a : ℝ) :
    (∃ x : ℝ, x^4 + a * x^3 - 2 * x^2 + a * x + 2 = 0) ↔ a ∈ Set.Iic 0 :=
begin
  -- The proof will go here
  sorry
end

end polynomial_real_root_exists_l692_692859


namespace total_comics_in_box_l692_692019

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l692_692019


namespace largest_prime_factor_1001_l692_692335

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692335


namespace coloring_theorem_l692_692630

open Real BigOperators

def adjacent (P Q : ℤ × ℤ) : Prop :=
  (↑P.fst - ↑Q.fst)^2 + (↑P.snd - ↑Q.snd)^2 = 2 ∨
  P.fst = Q.fst ∨ P.snd = Q.snd

def colorings (n : ℕ) : ℝ :=
  (6 / Real.sqrt 33) * ((7 + Real.sqrt 33) / 2)^n - 
  (6 / Real.sqrt 33) * ((7 - Real.sqrt 33) / 2)^n

theorem coloring_theorem (n : ℕ) (hn : 0 < n) :
  ∃ f : ℤ × ℤ → ℕ, 
    (∀ P Q, adjacent P Q → f P ≠ f Q) ∧
    ∑ (x, y) in (finset.Icc (-n, -n) (n, n)), colorings n 
    = (6 / Real.sqrt 33) * ((7 + Real.sqrt 33) / 2)^n - 
      (6 / Real.sqrt 33) * ((7 - Real.sqrt 33) / 2)^n := 
sorry

end coloring_theorem_l692_692630


namespace arithmetic_sequence_identification_l692_692176

variable (a : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_identification (h : is_arithmetic a d) :
  (is_arithmetic (fun n => a n + 3) d) ∧
  ¬ (is_arithmetic (fun n => a n ^ 2) d) ∧
  (is_arithmetic (fun n => a (n + 1) - a n) d) ∧
  (is_arithmetic (fun n => 2 * a n) (2 * d)) ∧
  (is_arithmetic (fun n => 2 * a n + n) (2 * d + 1)) :=
by
  sorry

end arithmetic_sequence_identification_l692_692176


namespace incorrect_statements_l692_692625

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- We transform the conditions into Lean 4 statements
theorem incorrect_statements : ¬ 
  ( (∀ x : ℝ, f (x) = f (Real.pi / 3 - x)) ∧ 
    (∀ x : ℝ, f (x + Real.pi / 12) = f (-x + Real.pi / 12)) ∧ 
    (∃ ⦃g : ℝ → ℝ⦄, ∀ x : ℝ, g (x) = Real.cos (2 * x) ∧ g x = g (-x)) ∧ 
    (Real.sin_periodic f Real.pi ∧  (∀ x ∈ (set.Icc 0 (Real.pi / 6)), @MonotoneOn ℝ ℝ _ _ _ (f) (set.Icc 0 (Real.pi / 6))))
  ) := 
  sorry

end incorrect_statements_l692_692625


namespace tangent_curves_k_value_l692_692512

theorem tangent_curves_k_value (x k : ℝ) (h1 : x + 4 = k / x) (h2 : ∃ x0 , y = x + 4 ∧ y = k / x) : k = -4 := by
  sorry

end tangent_curves_k_value_l692_692512


namespace tops_count_l692_692703

def price_eq (C T : ℝ) : Prop := 3 * C + 6 * T = 1500 ∧ C + 12 * T = 1500

def tops_to_buy (C T : ℝ) (num_tops : ℝ) : Prop := 500 = 100 * num_tops

theorem tops_count (C T num_tops : ℝ) (h1 : price_eq C T) (h2 : tops_to_buy C T num_tops) : num_tops = 5 :=
by
  sorry

end tops_count_l692_692703


namespace equation_of_perpendicular_line_l692_692955

theorem equation_of_perpendicular_line :
  ∃ l : LinearEquation, 
    (passes_through_point l (-1, 3)) ∧ 
    (is_perpendicular l (LinearEquation.mk 1 (-2) 3)) ∧ 
    to_standard_form l = (2, 1, -1) :=
by
  sorry

end equation_of_perpendicular_line_l692_692955


namespace loci_of_P_l692_692765

noncomputable theory

open_locale real

-- Definitions for points and sequences involved
variables 
  (P1 P3 : point)
  (P2 : point)
  (Pn : ℕ → point)
  (foot_of_perpendicular : Π (P A B : point), point)
  (line_through : point → point → set point)
  (circle_with_diameter : point → point → set point)
  (locus_of_convergence : set point)

-- Conditions
def is_perpendicular (A B C : point) : Prop := ∃ (l : set point), line_through B C = l ∧ A ∈ l ∧ ⟂ A (line_through B C)

axiom P2_condition : P2 ∈ (line_through P3 P2) ∧ (is_perpendicular P3 P1 P3)

axiom sequence_definition : ∀ n, Pn n = foot_of_perpendicular (Pn (n-1)) (Pn (n-2)) (line_through (Pn (n-1)) (Pn (n-2)))

-- The theorem to prove
theorem loci_of_P :
  locus_of_convergence = {p : point | p ∈ circle_with_diameter P1 P3 ∧ angle P1 p P3 = 106 * π / 180} :=
sorry

end loci_of_P_l692_692765


namespace number_of_odd_binomial_coeff_l692_692620

theorem number_of_odd_binomial_coeff (x : ℕ) (h : x = 8) :
  (finset.filter (λ k, nat.choose 8 k % 2 = 1) (finset.range 9)).card = 2 :=
by sorry

end number_of_odd_binomial_coeff_l692_692620


namespace option_C_is_incorrect_l692_692505

variables {Point Line Plane : Type}
variables {A B : Point} {a b l : Line} {α β : Plane}

-- Conditions based on the problem statement
variable (A_in_l : A ∈ l)
variable (l_not_in_α : ¬ l ⊆ α)

-- Proving the equivalece statement
theorem option_C_is_incorrect : ¬ (A_in_l ∧ l_not_in_α → A ∉ α) :=
sorry

end option_C_is_incorrect_l692_692505


namespace number_of_solutions_l692_692618

theorem number_of_solutions (N : ℕ) :
  (∑ n in finset.range (N), 2 * n) + 1 = N^2 - N + 1 :=
sorry

end number_of_solutions_l692_692618


namespace total_money_made_l692_692827

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l692_692827


namespace intersection_sum_coords_l692_692967

structure Point :=
  (x : ℤ)
  (y : ℤ)

def A : Point := ⟨0, 8⟩
def B : Point := ⟨0, 0⟩
def C : Point := ⟨10, 0⟩

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def D : Point := midpoint A B -- mid_AB
def E : Point := midpoint B C -- mid_BC

-- Equations of the lines AE and CD
def line_AE (x : ℤ) : ℤ := -8 * x / 5 + 8
def line_CD (x : ℤ) : ℤ := -4 * x / 5 + 4

-- Intersection point F of AE and CD
def F : Point :=
  let x := 5 in
  ⟨x, line_AE x⟩

def sum_coordinates (P : Point) : ℤ := P.x + P.y

theorem intersection_sum_coords : sum_coordinates F = 5 :=
  sorry

end intersection_sum_coords_l692_692967


namespace locus_of_C_l692_692811

-- Define given properties of the triangle
def base (A B : Point) : ℝ := 6
def median (A D : Point) : ℝ := 4
def altitude (A E : Point) : ℝ := 3
def midpoint (B C D : Point) : Prop := dist B D = dist D C
def loci (C : Point → Prop) : Prop :=
  ∃ O : Point, dist A O = 4 ∧ ∀ P, dist O P = 3 ↔ loci P

-- State the theorem
theorem locus_of_C (A B C D E : Point)
  (AB_length : base A B = 6)
  (AD_length : median A D = 4)
  (AE_length : altitude A E = 3)
  (D_midpoint : midpoint B C D)
  : loci C :=
by
  sorry

end locus_of_C_l692_692811


namespace even_digit_three_digit_numbers_count_l692_692861

theorem even_digit_three_digit_numbers_count : 
  (card { n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∀ d ∈ ([n / 100 % 10, n / 10 % 10, n % 10] : List ℕ), d ∈ {0, 2, 4, 6, 8} }) = 100 :=
by
  sorry

end even_digit_three_digit_numbers_count_l692_692861


namespace correct_calculation_result_l692_692385

-- Define the conditions in Lean
variable (num : ℤ) (mistake_mult : ℤ) (result : ℤ)
variable (h_mistake : mistake_mult = num * 10) (h_result : result = 50)

-- The statement we want to prove
theorem correct_calculation_result 
  (h_mistake : mistake_mult = num * 10) 
  (h_result : result = 50) 
  (h_num_correct : num = result / 10) :
  (20 / num = 4) := sorry

end correct_calculation_result_l692_692385


namespace sin_x_plus_pi_l692_692510

theorem sin_x_plus_pi {x : ℝ} (hx : Real.sin x = -4 / 5) : Real.sin (x + Real.pi) = 4 / 5 :=
by
  -- Proof steps go here
  sorry

end sin_x_plus_pi_l692_692510


namespace probability_no_3x3_red_square_l692_692069

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l692_692069


namespace board_lighting_problem_l692_692579

theorem board_lighting_problem :
  let row := 6
  let col := 6
  let initial_state := λ (r c : ℕ), false
  let toggle := λ (state: ℕ × ℕ → bool) (pos: ℕ × ℕ), 
                λ p, if p = pos ∨ p = (pos.1 + 1, pos.2) ∨ p = (pos.1 - 1, pos.2) ∨ 
                        p = (pos.1, pos.2 + 1) ∨ p = (pos.1, pos.2 - 1)
                then ¬ state p else state p
  let final_state := 
    let positions := (List.range row).bind (λ r, (List.range col).map (λ c, (r, c)))
    positions.foldl toggle initial_state

  (List.range row).bind (λ r, (List.range col).map (λ c, (r, c)))
    .count (λ pos, final_state pos = true) = 20 :=
by
  sorry

end board_lighting_problem_l692_692579


namespace largest_prime_factor_1001_l692_692367

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692367


namespace AM_DN_XY_concurrent_l692_692628

noncomputable def areConcurrent (A B C D X Y Z P : Point) (M N Q : Point) : Prop :=
  collinear A M Q ∧ collinear D N Q ∧ collinear X Y Q

theorem AM_DN_XY_concurrent
  (A B C D X Y Z P M N : Point)
  (h_distinct: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)
  (h_order: collinear A B C ∧ collinear B C D)
  (h_circles_inter: circle_diameter A C ∩ circle_diameter B D = {X, Y})
  (h_lineXY: collinear X Y)
  (h_lineInt: line XY ∩ line BC = {Z})
  (h_pointP: P ≠ Z ∧ P ∈ line XY)
  (h_intCircleAC : line CP ∩ circle_diameter A C = {C, M})
  (h_intCircleBD : line BP ∩ circle_diameter B D = {B, N}) :
  ∃ Q, areConcurrent A B C D X Y Z P M N Q :=
by
  sorry

end AM_DN_XY_concurrent_l692_692628


namespace tubs_from_usual_vendor_l692_692425

def total_tubs_needed : Nat := 100
def tubs_in_storage : Nat := 20
def fraction_from_new_vendor : Rat := 1 / 4

theorem tubs_from_usual_vendor :
  let remaining_tubs := total_tubs_needed - tubs_in_storage
  let tubs_from_new_vendor := remaining_tubs * fraction_from_new_vendor
  let tubs_from_usual_vendor := remaining_tubs - tubs_from_new_vendor
  tubs_from_usual_vendor = 60 :=
by
  intro remaining_tubs tubs_from_new_vendor
  exact sorry

end tubs_from_usual_vendor_l692_692425


namespace two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692927

theorem two_pairs_satisfy_x_squared_minus_y_squared_is_77 :
  {p : ℕ × ℕ // p.1 ^ 2 - p.2 ^ 2 = 77}.card = 2 :=
sorry

end two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692927


namespace train_crossing_time_l692_692325

theorem train_crossing_time
  (L1 L2 : ℝ) (v : ℝ) 
  (t1 t2 t : ℝ) 
  (h_t1 : t1 = 27)
  (h_t2 : t2 = 17)
  (hv_ratio : v = v)
  (h_L1 : L1 = v * t1)
  (h_L2 : L2 = v * t2)
  (h_t12 : t = (L1 + L2) / (v + v)) :
  t = 22 :=
by
  sorry

end train_crossing_time_l692_692325


namespace other_root_of_quadratic_l692_692263

theorem other_root_of_quadratic (z : ℂ) (h : z ^ 2 = -75 + 40 * Complex.i) (root : z = 5 + 7 * Complex.i) : 
  ∃ z' : ℂ, z' ^ 2 = -75 + 40 * Complex.i ∧ root ≠ z' ∧ z' = -5 - 7 * Complex.i :=
by 
  use -5 - 7 * Complex.i
  split
  sorry
  split
  sorry
  refl

end other_root_of_quadratic_l692_692263


namespace find_m_plus_n_l692_692086

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l692_692086


namespace solve_y_condition_solve_for_y_l692_692279

theorem solve_y_condition (y : ℝ) (h : y ≠ 2) : 7 * y - 5 = 2 → y = 1 := 
by
  intro h_eq
  have x : 7 * y = 7 := by
    linarith
  exact eq_of_mul_eq_mul_left (by norm_num) x

-- The main proof theorem
theorem solve_for_y : ∃ y : ℝ, (y ≠ 2) ∧ (7 * y - 5 = 2) ∧ (y = 1) := 
by
  use 1
  constructor
  { norm_num, }
  constructor
  { norm_num, }
  { refl }

end solve_y_condition_solve_for_y_l692_692279


namespace minimum_value_of_reciprocals_l692_692538

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1) + x + sin x

theorem minimum_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_cond : f (4 * a) + f (b - 9) = 0) : (1 / a) + (1 / b) = 1 := by
  sorry

end minimum_value_of_reciprocals_l692_692538


namespace trapezoid_area_l692_692000

theorem trapezoid_area (h : ℝ) : 
  let b1 : ℝ := 4 * h + 2
  let b2 : ℝ := 5 * h
  (b1 + b2) / 2 * h = (9 * h ^ 2 + 2 * h) / 2 :=
by 
  let b1 := 4 * h + 2
  let b2 := 5 * h
  sorry

end trapezoid_area_l692_692000


namespace aston_comics_l692_692014

theorem aston_comics (total_pages_on_floor : ℕ) (pages_per_comic : ℕ) (untorn_comics_in_box : ℕ) :
  total_pages_on_floor = 150 →
  pages_per_comic = 25 →
  untorn_comics_in_box = 5 →
  (total_pages_on_floor / pages_per_comic + untorn_comics_in_box) = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end aston_comics_l692_692014


namespace probability_opposite_rooms_l692_692439

theorem probability_opposite_rooms :
  let rooms := {301, 302, 303, 304, 305, 306}
  ∃ A B : ∀ x ∈ rooms, bool,
  let roomPairs := [(301, 302), (303, 304), (305, 306)],
      totalWays := fintype.card (equiv.perm (fin 6)),
      favorableWays := 3 * (fintype.card (equiv.perm (fin 4)) * 2),
      prob := favorableWays / totalWays
  in prob = 1/5 :=
begin
  sorry
end

end probability_opposite_rooms_l692_692439


namespace least_blue_cells_l692_692746

theorem least_blue_cells (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, 
    k ≥ 1 ∧ 
    (n = 2 * k - 1 ∨ n = 2 * k) ∧ 
    (n^2 = 2 * (⌊(n^2 - 1) / 2⌋ + 1 - k) + (n^2 % 2)) :=
by sorry

end least_blue_cells_l692_692746


namespace log_simplify_trig_expression_evaluation_l692_692399

-- Problem (I)
theorem log_simplify:
  logBase (1/3) (sqrt 27) + log 25 + log 4 + 7^(-logBase 7 2) + (-0.98)^0 = 2 := by
  sorry

-- Problem (II)
theorem trig_expression_evaluation 
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (sqrt 2, -sqrt 6)) :
  let cos := real.cos
  let sin := real.sin in
  (cos (π / 2 + α) * cos (2 * π - α) + sin (-α - π / 2) * cos (π - α)) /
  (sin (π + α) * cos (π / 2 - α)) = -((sqrt 3 + 1) / 3) := by
  sorry

end log_simplify_trig_expression_evaluation_l692_692399


namespace domain_of_function_l692_692293

theorem domain_of_function : 
  (∀ x : ℝ, (x + 1 > 0) ∧ (x - 2 ≠ 0) → (x ∈ (-1, 2) ∪ (2, +∞))) :=
by
  sorry

end domain_of_function_l692_692293


namespace range_of_a_decreasing_l692_692904

-- Define the function f(x)
def f (a x : ℝ) : ℝ := a * x ^ 2 + (2 * (a - 3)) * x + 1

-- Predicate indicating that f(x) is decreasing on [-2, +∞)
def isDecreasingOn (a : ℝ) : Prop :=
  ∀ x y, x ∈ set.Ici ( -2 : ℝ ) → y ∈ set.Ici ( -2 : ℝ ) → x < y → f a x ≥ f a y

-- The theorem statement
theorem range_of_a_decreasing :
  {a | isDecreasingOn a} = set.Icc (-3) 0 :=
sorry

end range_of_a_decreasing_l692_692904


namespace correct_f_l692_692637

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom functional_equation (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem correct_f (x : ℝ) : f x = x + 1 := sorry

end correct_f_l692_692637


namespace expected_length_of_first_group_l692_692721

-- Define the conditions of the problem
def sequence : Finset ℕ := {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

-- Expected length of the first group
def expected_length_first_group : ℝ := 2.83

-- The formal statement of the proof problem
theorem expected_length_of_first_group (seq : Finset ℕ) (h1 : seq.card = 68) (h2 : seq.filter (λ x, x = 1) = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
(h3 : seq.filter (λ x, x = 0) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) :
  let X := 
  (Finset.sum (Finset.range 19) (λ k, if k = 1 then (1/50 : ℝ) else 0)) +
  (Finset.sum (Finset.range 49) (λ m, if m = 1 then (1/20 : ℝ) else 0)) in 
  ∃ x : ℝ, ∑ x = expected_length_first_group := 
by sorry
 
end expected_length_of_first_group_l692_692721


namespace problem_solution_l692_692378

structure ArithmeticSequence (a₀ : ℕ) (d : ℕ) :=
  (sequence : ℕ → ℕ)
  (is_arithmetic : ∀ n, sequence (n + 1) = sequence n + d)

def sequence : ArithmeticSequence 3 4 :=
{ sequence := λ n, 3 + n * 4,
  is_arithmetic := by {
    intro n,
    simp [(λ n, 3 + n * 4)],
    ring } }

lemma x_y_sum_in_sequence : ∃ x y, x + y = 42 ∧
  sequence.sequence 5 = x ∧ sequence.sequence 6 = y ∧ sequence.sequence 7 = 27 :=
begin
  use [19, 23],
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { refl }
end

theorem problem_solution : (∃ x y, x + y = 42 ∧
  ∃ sequence : ℕ → ℕ, sequence 0 = 3 ∧ sequence 1 = 7 ∧ sequence 2 = 11 ∧ sequence 5 = x ∧ sequence 6 = y ∧ sequence 7 = 27) :=
begin
  use [19, 23],
  exact ⟨42, 3, 7, 11, 19, 23, 27⟩
end

end problem_solution_l692_692378


namespace probability_no_3by3_red_grid_correct_l692_692064

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l692_692064


namespace max_value_of_f_range_of_a_l692_692158

variables {x a : ℝ}

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x + 2) - (3 / 2) * x^2

theorem max_value_of_f : 
  f (1 / 3) = Real.log 3 - 1 / 6 :=
by sorry

theorem range_of_a (h : ∀ x ∈ Icc 1 2, abs (a - Real.log x) + Real.log (abs (f' x + 3 * x)) > 0) :
  a < Real.log (3 / 5) ∨ a > Real.log (16 / 3) :=
by sorry

end max_value_of_f_range_of_a_l692_692158


namespace find_line_equation_l692_692514

theorem find_line_equation :
  ∃ (a b c : ℝ), (a * -5 + b * -1 = c) ∧ (a * 1 + b * 1 = c + 2) ∧ (b ≠ 0) ∧ (a * 2 + b = 0) → (∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = -5) :=
by
  sorry

end find_line_equation_l692_692514


namespace new_students_admitted_l692_692314

-- Definitions of the conditions
def original_students := 35
def increase_in_expenses := 42
def decrease_in_average_expense := 1
def original_expenditure := 420

-- Main statement: proving the number of new students admitted
theorem new_students_admitted : ∃ x : ℕ, 
  (original_expenditure + increase_in_expenses = 11 * (original_students + x)) ∧ 
  (x = 7) := 
sorry

end new_students_admitted_l692_692314


namespace largest_prime_factor_1001_l692_692361

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692361


namespace find_sv_l692_692555

variables {V : Type*} [inner_product_space ℝ V]
variables (a b p : V)

theorem find_sv
  (h : ∥p - b∥ = 3 * ∥p - a∥) :
  ∃ (s v : ℝ), s = 9 / 8 ∧ v = -1 / 8 ∧ ∃ c : ℝ, ∥p - (s • a + v • b)∥ = c :=
by
  sorry

end find_sv_l692_692555


namespace infinitely_many_ks_no_prime_divisor_l692_692270

theorem infinitely_many_ks_no_prime_divisor (p : ℕ) (k : ℕ → ℕ) (h1 : ∀ (t : ℕ), p = 8 * t + 5 → ¬ ∃ x, (x ^ 2 ≡ 2 [MOD p]) ∧ (x ^ 2 ≡ -2 [MOD p]))
  (h2 : ∀ (x : ℕ), k x = 8 * x ^ 4 - 2):
  ∃ (infinitely_many k : ℕ), ∀ (t : ℕ) (p = 8 * t + 5), ¬(p ∣ (k * (k + 1) * (k + 2) * (k + 3))) :=
begin
  sorry -- proof is not required
end

end infinitely_many_ks_no_prime_divisor_l692_692270


namespace eleventh_term_arithmetic_seq_l692_692690

variable (α : Type) [LinearOrderedField α]

def arithmetic_sequence (a : α) (d : α) (n : ℕ) : α := 
  a + (n - 1) * d

theorem eleventh_term_arithmetic_seq
  (a d : α)
  (h5 : arithmetic_sequence a d 5 = (3 : α) / 8)
  (h17 : arithmetic_sequence a d 17 = (7 : α) / 12) :
  arithmetic_sequence a d 11 = (23 : α) / 48 :=
by
  sorry

end eleventh_term_arithmetic_seq_l692_692690


namespace find_lambda_l692_692575

variables {V : Type*} [inner_product_space ℝ V]

-- Conditions
variables (a b : V)
variable (λ : ℝ)
hypothesis (h1 : ¬ collinear ℝ (set.range ![a, b])) -- ¬ collinear(a, b)
hypothesis (ha_norm : ∥a∥ = 2)
hypothesis (hb_norm : ∥b∥ = 3)
hypothesis (orthogonal1 : ⟪3 • a + 2 • b, λ • a - b⟫ = 0)
hypothesis (orthogonal2 : ⟪a, b⟫ = 0)

-- The theorem statement
theorem find_lambda : λ = 3 / 2 :=
begin
  sorry,
end

end find_lambda_l692_692575


namespace intersection_union_complement_union_l692_692508

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable [Inhabited (Set ℝ)]

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 2) > 1 }
noncomputable def setB : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection (U : Set ℝ) : 
  (setA ∩ setB) = { x : ℝ | (0 < x ∧ x < 1) ∨ x > 3 } := 
  sorry

theorem union (U : Set ℝ) : 
  (setA ∪ setB) = univ := 
  sorry

theorem complement_union (U : Set ℝ) : 
  ((U \ setA) ∪ setB) = { x : ℝ | x ≥ 0 } := 
  sorry

end intersection_union_complement_union_l692_692508


namespace onions_left_on_scale_l692_692403

-- Define the given weights and conditions
def total_weight_of_40_onions : ℝ := 7680 -- in grams
def avg_weight_remaining_onions : ℝ := 190 -- grams
def avg_weight_removed_onions : ℝ := 206 -- grams

-- Converting original weight from kg to grams
def original_weight_kg_to_g (w_kg : ℝ) : ℝ := w_kg * 1000

-- Proof problem
theorem onions_left_on_scale (w_kg : ℝ) (n_total : ℕ) (n_removed : ℕ) 
    (total_weight : ℝ) (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ)
    (h1 : original_weight_kg_to_g w_kg = total_weight)
    (h2 : n_total = 40)
    (h3 : n_removed = 5)
    (h4 : avg_weight_remaining = avg_weight_remaining_onions)
    (h5 : avg_weight_removed = avg_weight_removed_onions) : 
    n_total - n_removed = 35 :=
sorry

end onions_left_on_scale_l692_692403


namespace sean_div_julie_l692_692275

def sean_sum : ℕ := (range 250).sum (λ n, 1 + 2 * n)

def julie_sum : ℕ := (range 300).sum (λ n, n + 1)

theorem sean_div_julie : sean_sum / julie_sum = 625 / 451.5 :=
by
  sorry

end sean_div_julie_l692_692275


namespace distance_marta_walks_in_15_minutes_l692_692989

theorem distance_marta_walks_in_15_minutes 
  (t1 t2 : ℕ) (d1 d2 : ℝ) 
  (h1 : t1 = 36) 
  (h2 : d1 = 1.5) 
  (h3 : t2 = 15) :
  Real.round ((d1 / t1) * t2 * 10) = 6 := 
sorry

end distance_marta_walks_in_15_minutes_l692_692989


namespace john_buys_1000_balloons_l692_692996

-- Define conditions
def balloon_volume : ℕ := 10
def tank_volume : ℕ := 500
def num_tanks : ℕ := 20

-- Define the total volume of gas
def total_gas_volume : ℕ := num_tanks * tank_volume

-- Define the number of balloons
def num_balloons : ℕ := total_gas_volume / balloon_volume

-- Prove that the number of balloons is 1,000
theorem john_buys_1000_balloons : num_balloons = 1000 := by
  sorry

end john_buys_1000_balloons_l692_692996


namespace speed_of_boat_is_15_l692_692727

noncomputable def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 / 5 ∧ (x + 3) * t = 3.6 ∧ x = 15

theorem speed_of_boat_is_15 (x : ℝ) (t : ℝ) (rate_of_current : ℝ) (distance_downstream : ℝ) :
  rate_of_current = 3 →
  distance_downstream = 3.6 →
  t = 1 / 5 →
  (x + rate_of_current) * t = distance_downstream →
  x = 15 :=
by
  intros h1 h2 h3 h4
  -- proof goes here
  sorry

end speed_of_boat_is_15_l692_692727


namespace only_eigenvalue_is_3_l692_692493

-- Define our specific matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, 3]]

-- The proof problem statement
theorem only_eigenvalue_is_3 : ∀ k : ℝ,
  (∃ (v : Fin 2 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v) ↔ k = 3 := by
  sorry

end only_eigenvalue_is_3_l692_692493


namespace problem1_problem2_problem3_l692_692140

-- Define the given conditions
def ellipse_eqn(a b : ℝ) (M : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
    M x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

def focus_point(F : ℝ → ℝ → Prop) : Prop :=
  F (-1) 0

def inclination_45_deg : Prop := 
  ∀ (x y : ℝ), y = x + 1
  
def line_segment(C D : ℝ → ℝ → Prop) (CD_length : ℝ) : Prop :=
  ∀ x1 x2 y1 y2 : ℝ, C x1 y1 ∧ D x2 y2 → 
  CD_length = 24 / 7

def max_value_of_triangle_diff(k : ℝ) (max_diff : ℝ) : Prop :=
  max_diff = sqrt 3

-- Proof statements
theorem problem1 : 
  ∃ a b (M : ℝ → ℝ → Prop), 0 < a ∧ 3 = b^2 ∧ focus_point(F) ∧ ellipse_eqn(a, b, M) :=
sorry

theorem problem2 : 
  ∃ (F : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) (CD_length : ℝ), 
  focus_point(F) ∧ inclination_45_deg ∧ line_segment(F, l, CD_length) :=
sorry

theorem problem3 : 
  ∃ (ABD ABC : ℝ → ℝ → ℝ → Prop) (k max_diff : ℝ), 
  max_value_of_triangle_diff(k, max_diff) :=
sorry

end problem1_problem2_problem3_l692_692140


namespace reflection_vector_l692_692801

theorem reflection_vector (v1 v2 v3 reflected_v : ℝ × ℝ) (h1 : v1 = (2, -3)) (h2 : v2 = (-2, 9)) (h3 : v3 = (3, 1)) (h_reflection : reflected_v = (-3, 1)) :
  ∃ line_reflection_vector : ℝ × ℝ, (∃ k : ℝ, v2 = (-k * v1.1, k * v1.2)) → ((2 * line_reflection_vector - v3) = reflected_v) :=
begin
  sorry
end

end reflection_vector_l692_692801


namespace amazing_squares_exist_l692_692504

structure Quadrilateral :=
(A B C D : Point)

def diagonals_not_perpendicular (quad : Quadrilateral) : Prop := sorry -- The precise definition will abstractly represent the non-perpendicularity of diagonals.

def amazing_square (quad : Quadrilateral) (square : Square) : Prop :=
  -- Definition stating that the sides of the square (extended if necessary) pass through distinct vertices of the quadrilateral
  sorry

theorem amazing_squares_exist (quad : Quadrilateral) (h : diagonals_not_perpendicular quad) :
  ∃ squares : Finset Square, squares.card ≥ 6 ∧ ∀ square ∈ squares, amazing_square quad square :=
by sorry

end amazing_squares_exist_l692_692504


namespace chess_amateurs_l692_692744

theorem chess_amateurs (n : ℕ) (h1 : ∀ x, x ∈ finset.range n → finset.card (finset.filter (λ y, y ≠ x) (finset.range n)) = 4)
                       (h2 : (n * 4) / 2 = 12) : n = 6 := 
by sorry

end chess_amateurs_l692_692744


namespace problem1_problem2_l692_692398

-- Definitions for permutation and combination
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problems statements
theorem problem1 : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by 
  sorry

theorem problem2 :
  C 200 198 + C 200 196 + 2 * C 200 197 = C 202 4 := by 
  sorry

end problem1_problem2_l692_692398


namespace limit_of_sequence_equals_e_cubed_l692_692777

theorem limit_of_sequence_equals_e_cubed :
  (Real.lim (λ n : ℕ, (↑(2 * n^2 + 2 * n + 3) / ↑(2 * n^2 + 2 * n + 1)) ^ (3 * n^2 - 7)) = Real.exp 3) :=
begin
  sorry
end

end limit_of_sequence_equals_e_cubed_l692_692777


namespace prove_formula_general_prove_sum_seq_l692_692988

-- Define the conditions given in the problem
def geo_seq (a : ℕ → ℝ) := ∀ n : ℕ, a n > 0 ∧ (n ≥ 1 → a (n + 1) = r * a n)
def a1 (a : ℕ → ℝ) := a 1 = 1
def arith_seq (a : ℕ → ℝ) := a 4 - 2 * (3 * a 3) + a 5 = 0

-- Define the sequence a_n for the solution and match it with given conditions
def a_n (n : ℕ) : ℝ := 2^(n-1)

theorem prove_formula_general (a : ℕ → ℝ) (r : ℝ) :
  geo_seq a ∧ a1 a ∧ arith_seq a → a = a_n :=
sorry

-- Definitions for the conditions given in the second part of the problem
def S_n (s : ℕ → ℝ) := ∀ n : ℕ, s (n+1) - s n = 2^n - 1
def seq_diff (a : ℕ → ℝ) (λ : ℝ) := λ = 1

-- Sum of the first n terms condition
def sum_n (a : ℕ → ℝ) (sum_seq : ℕ → ℝ) := ∑ i in range n, (a (i + 1) - λ * a i) = sum_seq n

-- Prove the sequence condition and the value of lambda
theorem prove_sum_seq (a : ℕ → ℝ) (s : ℕ → ℝ) (λ : ℝ) :
  geo_seq a ∧ a1 a ∧ arith_seq a ∧ (sum_n a s) ∧ (S_n s) → seq_diff a λ :=
sorry

end prove_formula_general_prove_sum_seq_l692_692988


namespace petya_pencils_l692_692658

theorem petya_pencils (x : ℕ) (promotion : x + 12 = 61) :
  x = 49 :=
by
  sorry

end petya_pencils_l692_692658


namespace triangle_area_is_six_l692_692324

noncomputable theory
open_locale classical

def point := ℝ × ℝ

def line_slope (slope : ℝ) (pt : point) : ℝ → ℝ := λ x, slope * (x - pt.1) + pt.2

-- Line definitions from conditions
def line1 := line_slope (1/2) (2, 2)
def line2 := line_slope 2 (2, 2)
def line3 (x: ℝ) := 10 - x

-- Intersection points
def intersection (f g : ℝ → ℝ) (x: ℝ) : x ∈ ℝ ∧ f x = g x

def pointA : point := (2, 2)
def pointB : point := (4, 6)
def pointC : point := (6, 4)

-- Triangle area using vertices (A = (2, 2), B = (4, 6), C = (6, 4))
def triangle_area (A B C: point) :=
  (1/2 : ℝ) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_six : 
  triangle_area pointA pointB pointC = 6 :=
by sorry

end triangle_area_is_six_l692_692324


namespace part_a_l692_692769

theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : 
  (a % 10 = b % 10) := 
sorry

end part_a_l692_692769


namespace daughter_weight_l692_692422

variable (M D C : ℝ)

def condition1 := M + D + C = 150
def condition2 := D + C = 60
def condition3 := C = M / 5

theorem daughter_weight :
  condition1 → condition2 → condition3 → D = 42 := by
  intros h1 h2 h3
  sorry

end daughter_weight_l692_692422


namespace megan_bottles_left_l692_692645

-- Defining the initial conditions
def initial_bottles : Nat := 17
def bottles_drank : Nat := 3

-- Theorem stating that Megan has 14 bottles left
theorem megan_bottles_left : initial_bottles - bottles_drank = 14 := by
  sorry

end megan_bottles_left_l692_692645


namespace F_sum_l692_692890

noncomputable def f : ℝ → ℝ := sorry -- even function f(x)
noncomputable def F (x a c : ℝ) : ℝ := 
  let b := (a + c) / 2
  (x - b) * f (x - b) + 2016

theorem F_sum (a c : ℝ) : F a a c + F c a c = 4032 := 
by {
  sorry
}

end F_sum_l692_692890


namespace n_independent_polynomials_exist_l692_692503

noncomputable def n_independent_poly_exists (n : ℤ) : Prop :=
  ∃ (P : Polynomial ℝ), P.degree = 2000 ∧
  (∀ Q ∈ set.perm (coeffs P),
    ∃ Q₁ ∈ set.from_list (perm Q.coeffs),
      Q₁ n = 0 ∧ (∃ a b, Q₁ = Q.swap a b))

theorem n_independent_polynomials_exist :
  {n | n_independent_poly_exists n} = {0, 1} :=
sorry

end n_independent_polynomials_exist_l692_692503


namespace three_digit_number_ending_in_4_divisible_by_3_l692_692377

def three_digit_n_ends_in_4_divisible_by_3_probability : ℚ :=
  let count_valid_pairs := 3 * 3 + 3 * 4 + 3 * 3 in
  let total_pairs := 9 * 10 in
  count_valid_pairs / total_pairs

theorem three_digit_number_ending_in_4_divisible_by_3 :
  three_digit_n_ends_in_4_divisible_by_3_probability = 11 / 30 :=
by
  sorry

end three_digit_number_ending_in_4_divisible_by_3_l692_692377


namespace polynomial_evaluation_l692_692237

theorem polynomial_evaluation (P : ℕ → ℕ) (n : ℕ) 
  (h_degree : ∀ m, P m = 2^m ∧ m ∈ (finset.range (n+2)).erase (n+2)) :
  P (n+2) = 2^(n+2) - 2 :=
sorry

end polynomial_evaluation_l692_692237


namespace sufficient_but_not_necessary_l692_692244

variable (x : ℝ)

theorem sufficient_but_not_necessary : (x = 1) → (x^3 = x) ∧ (∀ y, y^3 = y → y = 1 → x ≠ y) :=
by
  sorry

end sufficient_but_not_necessary_l692_692244


namespace cos_expression_sum_to_abc_cos_given_condition_l692_692296

theorem cos_expression_sum_to_abc_cos_given_condition :
  ∃ (a b c d : ℕ+), (a + b + c + d = 13 ∧
                     (∀ x : ℝ, cos x + cos (3 * x) + cos (7 * x) + cos (9 * x)
                     = a * cos (b * x) * cos (c * x) * cos (d * x))) :=
sorry

end cos_expression_sum_to_abc_cos_given_condition_l692_692296


namespace num_non_positive_int_greater_than_neg_3_05_l692_692696

/-
  Definitions and Conditions:
  1. Non-positive integers are less than or equal to zero.
  2. Consider integers greater than -3.05.
-/

def is_non_positive_int (n : ℤ) : Prop := n ≤ 0

def greater_than_neg_3_05 (n : ℤ) : Prop := n > -3.05

-- Theorem to prove:
theorem num_non_positive_int_greater_than_neg_3_05 : ∃ (count : ℕ), count = 4 ∧ 
  count = ((finset.range 1).filter (λ n, is_non_positive_int n ∧ greater_than_neg_3_05 (n - (4 : ℤ)))).card :=
sorry

end num_non_positive_int_greater_than_neg_3_05_l692_692696


namespace corrected_sum_l692_692584

theorem corrected_sum : 37541 + 43839 ≠ 80280 → 37541 + 43839 = 81380 :=
by
  sorry

end corrected_sum_l692_692584


namespace two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692926

theorem two_pairs_satisfy_x_squared_minus_y_squared_is_77 :
  {p : ℕ × ℕ // p.1 ^ 2 - p.2 ^ 2 = 77}.card = 2 :=
sorry

end two_pairs_satisfy_x_squared_minus_y_squared_is_77_l692_692926


namespace seq_a_formula_b_seq_first_term_infinite_geom_seq_l692_692138

-- 1. Definition of the sequence a_n
def seq_a (n : ℕ) : ℝ := if n = 0 then 1 else 1 / sqrt (4 * n - 3)

-- Hypothesis: a sequence {a_n} such that a_1 = 1 and 1/a_(n+1) = sqrt(1/a_n^2 + 4)
axiom a_seq_prop (n : ℕ) (hn : n > 0) : n' = n - 1 → 1 / seq_a (n + 1) = sqrt (1 / (seq_a n')^2 + 4)

-- To prove: a_n = 1 / sqrt(4n - 3)
theorem seq_a_formula (n : ℕ) (hn : n > 0) : seq_a n = 1 / sqrt (4 * n - 3) := 
by
  sorry

-- 2. Definition of the sequence b_n and its sum S_n
def sum_S (n : ℕ) : ℝ := -- definition of the sum of b_n up to n term
sorry

-- Hypothesis: sum sequence {S_n} such that S_(n+1)/a_n^2 = S_n/a_(n+1)^2 + 16n^2 - 8n - 3
axiom b_seq_prop (n : ℕ) (hn : n > 0) : sum_S (n + 1) / (seq_a n)^2 = sum_S n / (seq_a (n + 1))^2 + 16 * n^2 - 8 * n - 3

-- To prove: b_1 = 1 if {b_n} is arithmetic.
theorem b_seq_first_term (b_seq : ℕ → ℝ) (arithmetic : ∀ n > 0, b_seq (n + 1) - b_seq n = b_seq 2 - b_seq 1) : b_seq 1 = 1 :=
by
  sorry

-- 3. Definition of the sequence c_n from terms of {1/a_n^2} with c_1 = 5
def seq_c (n : ℕ) (a : ℕ → ℝ) : ℝ := if n = 0 then 5 else 1 / (a n)^2

-- To prove: there exist infinitely many geometric sequences {c_n}
theorem infinite_geom_seq (c_seq : ℕ → ℝ) (a_seq : ℕ → ℝ) (c_1_eq : c_seq 0 = 5)
  (c_from_a : ∀ n, c_seq (n + 1) = 1 / (a_seq n)^2) : ∃q ∈ ℝ, ∀ m > 0, ∃inf_geom_seq, ∀ k > 0, c_seq k = c_seq 0 * q^(k - 1) :=
by
  sorry

end seq_a_formula_b_seq_first_term_infinite_geom_seq_l692_692138


namespace remainder_numGreenRedModal_l692_692919

def numGreenMarbles := 7
def numRedMarbles (n : ℕ) := 7 + n
def validArrangement (g r : ℕ) := (g + r = numGreenMarbles + numRedMarbles r) ∧ 
  (g = r)

theorem remainder_numGreenRedModal (N' : ℕ) :
  N' % 1000 = 432 :=
sorry

end remainder_numGreenRedModal_l692_692919


namespace total_money_made_l692_692828

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l692_692828


namespace calculate_t_u_l692_692795

variable (A B Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q]
variable [Module ℝ A] [Module ℝ B] [Module ℝ Q]

-- Given conditions
variable (a b q : A)
variable (k m n : ℝ)
variable (AQ_QB_ratio : k = 7 ∧ m = 2)
variable (section_formula : q = (m * b + n * a) / (m + n))

-- Mathematically equivalent statement
theorem calculate_t_u :
  ∃ t u : ℝ, (q = t • a + u • b) ∧ t = 2 / 9 ∧ u = 7 / 9 :=
by
  use [2 / 9, 7 / 9]
  sorry

end calculate_t_u_l692_692795


namespace binomial_divisor_l692_692478

theorem binomial_divisor (n k : ℕ) (h : k > 1) :
  (∀ m, 1 ≤ m ∧ m < n → k ∣ (nat.choose n m)) →
  ∃ (p : ℕ) (t : ℕ), nat.prime p ∧ t > 0 ∧ n = p ^ t ∧ k = p :=
sorry

end binomial_divisor_l692_692478


namespace max_sin_a_l692_692242

theorem max_sin_a (a b c : ℝ) (h1 : Real.cos a = Real.tan b) 
                                  (h2 : Real.cos b = Real.tan c) 
                                  (h3 : Real.cos c = Real.tan a) : 
  Real.sin a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := 
by
  sorry

end max_sin_a_l692_692242


namespace impossible_to_make_all_black_l692_692327

/-!
We have a 4x4 board where all 1x1 squares are initially white.
A move involves changing the colors of all squares in a 1x3 rectangle
(from black to white and from white to black).

In this statement, we aim to prove that it is impossible to make all the 1x1 squares black
after several moves.
-/

def initial_board := (fun (i j : Fin 4) => true)  -- all squares initially white

def move (board : Fin 4 → Fin 4 → Bool) (r : Fin 4) (c : Fin 4) : Fin 4 → Fin 4 → Bool :=
  fun i j =>
    if (i = r ∧ (j = c ∨ j = (c + 1) % 4 ∨ j = (c + 2) % 4))
    then ¬ board i j
    else board i j

theorem impossible_to_make_all_black :
  ¬ ∃ (sequence : List (Fin 4 × Fin 4)),
    let final_board := List.foldl (λ b mv, move b (mv.fst) (mv.snd)) initial_board sequence in
    ∀ i j : Fin 4, final_board i j = false :=
by
  sorry

end impossible_to_make_all_black_l692_692327


namespace test_point_selection_0618_method_l692_692206

theorem test_point_selection_0618_method :
  ∀ (x1 x2 x3 : ℝ),
    1000 + 0.618 * (2000 - 1000) = x1 →
    1000 + (2000 - x1) = x2 →
    x2 < x1 →
    (∀ (f : ℝ → ℝ), f x2 < f x1) →
    x1 + (1000 - x2) = x3 →
    x3 = 1236 :=
by
  intros x1 x2 x3 h1 h2 h3 h4 h5
  sorry

end test_point_selection_0618_method_l692_692206


namespace count_valid_numbers_correct_l692_692557

-- Conditions definitions
def is_even_digit (d : ℕ) : Prop := d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def has_only_even_digits (n : ℕ) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, is_even_digit d

-- Main statement: Count 4-digit numbers with only even digits divisible by 4
def count_valid_numbers : ℕ :=
  (List.range 9000).count (λ x => let n := x + 1000 in is_four_digit_number n ∧ has_only_even_digits n ∧ is_divisible_by_4 n)

-- Theorem statement: The count is equal to 500
theorem count_valid_numbers_correct : count_valid_numbers = 500 :=
by
  sorry

end count_valid_numbers_correct_l692_692557


namespace separate_curves_l692_692163

variable {A : Type} [CommRing A]

def crossing_characteristic (ε : A → ℤ) (A1 A2 A3 A4 : A) : Prop :=
  ε A1 + ε A2 + ε A3 + ε A4 = 0

theorem separate_curves {A : Type} [CommRing A]
  {ε : A → ℤ} {A1 A2 A3 A4 : A} 
  (h : ε A1 + ε A2 + ε A3 + ε A4 = 0)
  (h1 : ε A1 = 1 ∨ ε A1 = -1)
  (h2 : ε A2 = 1 ∨ ε A2 = -1)
  (h3 : ε A3 = 1 ∨ ε A3 = -1)
  (h4 : ε A4 = 1 ∨ ε A4 = -1) :
  (∃ B1 B2 : A, B1 ≠ B2 ∧  ∀ (A : A), ((ε A = 1) → (A = B1)) ∨ ((ε A = -1) → (A = B2))) :=
  sorry

end separate_curves_l692_692163


namespace number_of_pairs_satisfying_equation_l692_692925

theorem number_of_pairs_satisfying_equation : 
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77) = 2 :=
by 
  sorry

end number_of_pairs_satisfying_equation_l692_692925


namespace arith_seq_sum_l692_692204

-- We start by defining what it means for a sequence to be arithmetic
def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- We are given that a_2 = 5 and a_6 = 33 for an arithmetic sequence
variable (a : ℕ → ℤ)
variable (h_arith : is_arith_seq a)
variable (h1 : a 2 = 5)
variable (h2 : a 6 = 33)

-- The statement we want to prove
theorem arith_seq_sum (a : ℕ → ℤ) (h_arith : is_arith_seq a) (h1 : a 2 = 5) (h2 : a 6 = 33) :
  (a 3 + a 5) = 38 :=
  sorry

end arith_seq_sum_l692_692204


namespace project_completion_days_l692_692317

theorem project_completion_days (A B C : ℝ) (h1 : 1/A + 1/B = 1/2) (h2 : 1/B + 1/C = 1/4) (h3 : 1/C + 1/A = 1/2.4) : A = 3 :=
by
sorry

end project_completion_days_l692_692317


namespace graph_shift_sine_function_l692_692319

theorem graph_shift_sine_function :
  ∀ x, (sin (4 * x - π / 3) = sin (4 * (x - π / 12))) := 
by
  sorry

end graph_shift_sine_function_l692_692319


namespace largest_prime_factor_1001_l692_692362

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l692_692362


namespace transistors_1995_l692_692650

theorem transistors_1995 
    (initial_transistors : ℕ)
    (doubling_period_months : ℕ)
    (current_year : ℕ)
    (start_year : ℕ)
    (initial_year_transistors : ℕ) 
    (year_difference : ℕ)
    (months_in_a_year : ℕ)
    (doublings_count : ℕ) : 
    initial_transistors = 500000 → 
    doubling_period_months = 18 → 
    start_year = 1985 → 
    current_year = 1995 → 
    initial_year_transistors = 500000 → 
    year_difference = current_year - start_year → 
    months_in_a_year = 12 → 
    doublings_count = (year_difference * months_in_a_year) / doubling_period_months → 
    (initial_transistors * 2 ^ doublings_count) = 32000000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  calc
    initial_transistors * 2 ^ ((((current_year - start_year) * months_in_a_year) / doubling_period_months)) 
        = 500000 * 2 ^ 6 : by rw [h1, h2, h3, h4, h5, h6, h7, h8]
    ... = 32000000 : by norm_num

end transistors_1995_l692_692650


namespace smallest_abs_z_add_i_l692_692232

noncomputable def smallest_possible_value (z : ℂ) (h : abs (z^2 + 9) = abs (z * (z + 3 * complex.I))) : ℝ :=
  Inf { abs (z + complex.I) | abs (z^2 + 9) = abs (z * (z + 3 * complex.I)) }

theorem smallest_abs_z_add_i :
  smallest_possible_value z (abs (z^2 + 9) = abs (z * (z + 3 * complex.I))) = 2 :=
sorry

end smallest_abs_z_add_i_l692_692232


namespace largest_prime_factor_1001_l692_692336

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l692_692336


namespace sam_paid_amount_l692_692274

theorem sam_paid_amount (F : ℝ) (Joe Peter Sam : ℝ) 
  (h1 : Joe = (1/4)*F + 7) 
  (h2 : Peter = (1/3)*F - 7) 
  (h3 : Sam = (1/2)*F - 12)
  (h4 : Joe + Peter + Sam = F) : 
  Sam = 60 := 
by 
  sorry

end sam_paid_amount_l692_692274


namespace sin_alpha_value_l692_692524

theorem sin_alpha_value (α : ℝ) (h1 : Real.sin (α + π / 4) = 4 / 5) (h2 : α ∈ Set.Ioo (π / 4) (3 * π / 4)) :
  Real.sin α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end sin_alpha_value_l692_692524


namespace probability_of_covering_black_region_l692_692806

noncomputable def side_length_square : ℝ := 10
noncomputable def leg_length_triangles : ℝ := 3
noncomputable def side_length_diamond : ℝ := 3 * Real.sqrt 2
noncomputable def diameter_coin : ℝ := 2

-- Placeholder value for a and b
noncomputable def a : ℕ := 900
noncomputable def b : ℕ := 300

theorem probability_of_covering_black_region : 
  ∃ (a b : ℕ), a + b = 1200 ∧ 
  ∀ (square : ℝ) (triangles : ℝ) (diamond : ℝ) (coin : ℝ),
  square = 10 → triangles = 3 → diamond = 3 * Real.sqrt 2 → 
  coin = 2 → 
  let prob := (1 / 225) * (↑a + ↑b * Real.sqrt 2 + Real.pi) 
  in prob ≠ 0 :=
by
  use [a, b]
  use [side_length_square, leg_length_triangles, side_length_diamond, diameter_coin]
  split
  . exact rfl
  . intros
    unfold side_length_square leg_length_triangles side_length_diamond diameter_coin
    intro prob
    exact sorry  -- Proof of probability calculation

end probability_of_covering_black_region_l692_692806


namespace quadrilateral_area_l692_692589

open Real

/--Let \(A, B, C, D\) be points of a rectangle \(ABCD\) with \(AB = 2\) and \(BC = 3\). 
Point \(E\) is the midpoint of \(\overline{BC}\). 
Point \(F\) is at the quarter-point on \(\overline{CD}\) (i.e., nearer to \(C\)). 
Point \(G\) is at the three-quarter point on \(\overline{AD}\) (i.e., nearer to \(D\)). 
Point \(H\) is the midpoint of the segment connecting \(G\) and \(E\). 
What is the area of the quadrilateral formed by connecting \(A\), \(F\), \(H\), and \(D\) sequentially? -/
theorem quadrilateral_area :
  let A := (0, 0 : ℝ)
  let B := (2, 0 : ℝ)
  let C := (2, 3 : ℝ)
  let D := (0, 3 : ℝ)
  let E := (2, (1.5 : ℝ))
  let F := (2, (0.75 : ℝ))
  let G := (0, (2.25 : ℝ))
  let H := (1, ((1.5 + 2.25) / 2 : ℝ))
  area_of_quadrilateral A F H D = 3 / 4 :=
by
  /- skipping the proof -/
  sorry

#check @quadrilateral_area

end quadrilateral_area_l692_692589


namespace largest_prime_factor_of_1001_l692_692355

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692355


namespace b_is_26_l692_692867

noncomputable def find_b (a b c d : ℂ) : Prop :=
∀ (α β : ℂ), (α.conj = α) ∧ (β.conj = β) ∧ (α * α.conj = 13 + complex.I) ∧ (β * β.conj = 13 - complex.I) ∧
(α + α.conj = 3 + 4 * complex.I) ∧ (β + β.conj = 3 - 4 * complex.I) → b = 26

-- The statement to prove:
theorem b_is_26 (a c d : ℂ) : find_b a 26 c d :=
sorry -- The proof is omitted

end b_is_26_l692_692867


namespace triangle_area_correct_l692_692599

noncomputable def area_triangle_c2mn : ℝ :=
let l := λ x, sqrt 3 * x in
let C1_param := λ φ, (cos φ, 1 + sin φ) in
let C2 := λ θ, 4 * cos θ in
let intersection_M := (sqrt 3, π / 3) in
let intersection_N := (2, π / 3) in
1/2 * 4 * 2 * sin (π / 3) - 1/2 * 4 * sqrt 3 * sin (π / 3)

theorem triangle_area_correct :
  area_triangle_c2mn = sqrt 3 - 3 / 2 :=
begin
  sorry
end

end triangle_area_correct_l692_692599


namespace pairs_satisfying_equation_l692_692931

theorem pairs_satisfying_equation : 
  {p : Nat × Nat // let x := p.1; let y := p.2 in x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77}.toList.length = 2 :=
by
  sorry

end pairs_satisfying_equation_l692_692931


namespace number_of_groups_of_four_l692_692215

/-
Jenna is at a fair with six friends. They all want to ride the Ferris wheel, but only four people can fit in a cabin at one time.
How many different groups of four can the seven of them make?
-/

theorem number_of_groups_of_four (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  nat.choose n k = 35 :=
by
  rw [h1, h2]
  /- Now it simply remains to show nat.choose 7 4 = 35 -/
  sorry

end number_of_groups_of_four_l692_692215


namespace realGDPChange_l692_692685

-- Define all the quantities given in the problem
def Vegetables2014 := 1200
def Fruits2014 := 750
def PriceVegetables2014 := 90000
def PriceFruits2014 := 75000
def Vegetables2015 := 900
def Fruits2015 := 900
def PriceVegetables2015 := 100000
def PriceFruits2015 := 70000

-- Define Nominal GDP for 2014
def NominalGDP2014 := (Vegetables2014 * PriceVegetables2014) + (Fruits2014 * PriceFruits2014)

-- Define Real GDP for 2015 using 2014 prices
def RealGDP2015 := (Vegetables2015 * PriceVegetables2014) + (Fruits2015 * PriceFruits2014)

-- Define the percentage change formula
def PercentageChange (initial final : ℝ) := 100 * ((final - initial) / initial)

-- The main theorem to prove
theorem realGDPChange : (PercentageChange NominalGDP2014 RealGDP2015) = -9.59 :=
by
  have h1 : NominalGDP2014 = 164250 := by sorry
  have h2 : RealGDP2015 = 148500 := by sorry
  have h3 : (148500 - 164250) / 164250 ≈ -0.0959 := by sorry
  have h4 : PercentageChange 164250 148500 = -9.59 := by sorry
  exact h4

end realGDPChange_l692_692685


namespace nonagon_diagonals_l692_692797

theorem nonagon_diagonals : 
  ∀ (n : ℕ), n = 9 → ∃ (d : ℕ), d = (n * (n - 3)) / 2 := 
by
  intros n h
  have h' : n = 9 := h
  use 27
  rw h'
  simp
  exact rfl

end nonagon_diagonals_l692_692797


namespace largest_prime_factor_of_1001_l692_692351

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l692_692351


namespace part1_part2_l692_692823

open EuclideanGeometry

/-- Given the geometric setup with points M, N, P, I, T, and certain parallel lines and collinearity conditions,
prove that MP * MT = NP * NT. -/
theorem part1 (A B C M N P I T : Point)
  (h_triangle : triangle A B C)
  (h_ac_strict_angle : A.angle < B.angle)
  (h_midpoints_arcs : (is_midpoint_arc M (B,C)) ∧ (is_midpoint_arc N (A,C)))
  (h_PC_parallel_MN: parallel_lines P C M N)
  (h_P_on_circum : on_circumcircle P A B C)
  (h_I_incenter_ABC : incenter I (triangle A B C))
  (h_PI_intersects_T: PI ∩ circumcircle (A, B, C) = {T}) :
  MP * MT = NP * NT := sorry

/-- Given the geometric setup with points Q, I1, I2, and T,
prove that Q, I1, I2, and T are concyclic. -/
theorem part2 (A B C Q I1 I2 T : Point)
  (h_triangle : triangle A B C)
  (h_ac_strict_angle : A.angle < B.angle)
  (h_Q_on_arc_AB : on_arc Q A B C)
  (h_not_on_C : Q ≠ A ∧ Q ≠ T ∧ Q ≠ B)
  (h_incenter_AQC : incenter I1 (triangle A Q C))
  (h_incenter_QCB : incenter I2 (triangle Q C B))
  (h_T_on_circum : on_circumcircle T A B C) :
  cyclic Q I1 I2 T := sorry

end part1_part2_l692_692823


namespace common_difference_l692_692974

namespace ArithmeticSequence

variables {a₁ d : ℤ}
def a_n (n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem common_difference (h1 : a_n 3 + a_n 6 = 11) (h2 : a_n 5 + a_n 8 = 39) : d = 7 :=
by
  sorry

end ArithmeticSequence

end common_difference_l692_692974


namespace quadratic_prime_solutions_l692_692899

theorem quadratic_prime_solutions (m : ℕ) : 
  (∃ x₁ x₂ : ℕ, prime x₁ ∧ prime x₂ ∧ (x^2 - 1999 * x + m = 0) ∧ x₁ + x₂ = 1999) → m = 3994 :=
begin
  sorry
end

end quadratic_prime_solutions_l692_692899


namespace points_on_same_circle_l692_692265

variables (A B C O P Q : Type*) [euclidean_space A]
variables [metric_space A] [normed_group A] [normed_space ℝ A]

-- Definitions of the points being intermediate points of segments
variables (M N R S : A)
variables (midpoint_AP : midpoint R S A) (midpoint_CQ : midpoint Q S A )
variables (center_circumcircle: center_circumcircle O A)

-- Definitions for perpendicular line and intersection points
variables (perp_line : ∀ a b c, line_perpendicular_to c a c)
variables (intersects_BC : BC P intersects_B) (intersects_AB: AB Q intersects_A)

-- Formal statement to prove that the points lie on the same circle
theorem points_on_same_circle 
(O_center : circumcenter ABC O) 
(line_perpendicular : perp_line A C A P)
(intersection_BC : intersects_BC B C Q)
(intersection_AB : intersects_AB O B P)
(midpoint_AP : midpoint AP S A)
(midpoint_CQ : midpoint CQ Q A) :
cyclic_quadrilateral B O midpoint_AP midpoint_CQ := sorry

end points_on_same_circle_l692_692265


namespace KM_is_inscribed_square_side_l692_692607

-- Definitions
variable {A B C D E T K M : Type*}

-- Assume A, B, C, D form a square, and E, T, K, M are defined accordingly.
def is_square (sq : (A × B × C × D)) : Prop :=
  ∃ (AB BC CD DA : ℝ), 
    AB = BC ∧ BC = CD ∧ CD = DA ∧
    collinear [A, B, C, D] ∧
    right_angle ∠ABC ∧ right_angle ∠BCD ∧ right_angle ∠CDA ∧ right_angle ∠DAB

def inside_square (sq : (A × B × C × D)) (point : E) : Prop :=
  ∃ (a b c d e : ℝ),
    e > a ∧ e < c ∧ b < a ∧ b > d ∧
    collinear [AB, E] ∧ collinear [BC, E] ∧ collinear [CD, E] ∧ collinear [DA, E]

def altitude (ET A B E : Type*) : Prop :=
  ∃ (h : ℝ), 
    is_perpendicular ET {line(A, B)}

def intersection (a b : (A × B)) : Type := sorry

-- Given conditions
variables (sq : A × B × C × D)
variables (point : E)
variables (ET : altitude h A B E)
variables (K : intersection D T A E)
variables (M : intersection C T B E)

-- Main theorem
theorem KM_is_inscribed_square_side : ∀ {sq : (A × B × C × D)} {point : E} {ET : altitude h A B E} {K M},
  is_square sq → inside_square sq point → (K = intersection (D T) (A E)) → (M = intersection (C T) (B E)) →
  ∃ s, is_side_of_inscribed_square K M :=
by
  sorry

end KM_is_inscribed_square_side_l692_692607


namespace equivalence_of_r_i_and_inequality_l692_692144

variable {n : ℕ} (a : Fin n → ℝ) (m : ℝ) (h_pos : ∀ i, 0 < a i) (h_m_gt_1 : 1 < m)

theorem equivalence_of_r_i_and_inequality (r : Fin n → ℝ) (x : Fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ x i) :
  (∑ i, r i * (x i - a i)) ≤ ((∑ i, (x i)^m)^(1/m) - (∑ i, (a i)^m)^(1/m)) ↔
  (∀ i, r i = (a i)^((m-1)/m) / (∑ i, (a i)^m)^((m-1)/m)) :=
sorry

end equivalence_of_r_i_and_inequality_l692_692144


namespace ratio_of_autobiographies_to_fiction_l692_692311

theorem ratio_of_autobiographies_to_fiction (total_books fiction_books non_fiction_books picture_books autobiographies: ℕ) 
  (h1 : total_books = 35) 
  (h2 : fiction_books = 5) 
  (h3 : non_fiction_books = fiction_books + 4) 
  (h4 : picture_books = 11) 
  (h5 : autobiographies = total_books - (fiction_books + non_fiction_books + picture_books)) :
  autobiographies / fiction_books = 2 :=
by sorry

end ratio_of_autobiographies_to_fiction_l692_692311


namespace add_to_divisible_l692_692381

theorem add_to_divisible (n d x : ℕ) (h : n = 987654) (h1 : d = 456) (h2 : x = 222) : 
  (n + x) % d = 0 := 
by {
  sorry
}

end add_to_divisible_l692_692381


namespace probability_no_3by3_red_grid_correct_l692_692060

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l692_692060


namespace probability_no_3x3_red_square_l692_692081

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l692_692081


namespace duration_of_conflict_l692_692793

-- Definitions based on conditions in (a)
def start_date := (1812, 3, 10)
def end_date := (1816, 2, 29)
def start_day_of_week := "Tuesday"

-- Proof statements based on answers in (b)
theorem duration_of_conflict : 
  ∃ days: ℕ, days = 1452 ∧
  (end_date.day_of_week = "Friday") :=
by
  -- We need to compute and prove the exact number of days between the two dates
  sorry

end duration_of_conflict_l692_692793


namespace shortest_distance_A_B_l692_692161

theorem shortest_distance_A_B (a : ℝ) (h : 0 < a) :
  let A := (a, ln(a) - 1)
  let B := (a, a^2 + 1)
  let d := sqrt ((B.2 - A.2)^2 + (B.1 - A.1)^2)
  d = (5 + log 2) / 2 :=
begin
  sorry
end

end shortest_distance_A_B_l692_692161


namespace find_k_l692_692709

variables (l w : ℝ) (p A k : ℝ)

def rectangle_conditions : Prop :=
  (l / w = 5 / 2) ∧ (p = 2 * (l + w))

theorem find_k (h : rectangle_conditions l w p) :
  A = (5 / 98) * p^2 :=
sorry

end find_k_l692_692709


namespace midpoints_form_circle_l692_692203

noncomputable def set_of_midpoints (a d : ℝ) : set (ℝ × ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), p = (x / 2, y / 2, a / 2) ∧ x^2 + y^2 = d^2 - a^2 }

theorem midpoints_form_circle (a d : ℝ) (h : a > 0 ∧ d > a) :
  ∃ (r : ℝ) (c : ℝ × ℝ × ℝ), 
  r = sqrt ((d^2 - a^2) / 4) ∧ 
  c = (0, 0, a / 2) ∧ 
  ∀ p ∈ set_of_midpoints a d, 
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ∧ p.3 = c.3 :=
by
  sorry

end midpoints_form_circle_l692_692203


namespace g_is_even_l692_692849

def g (x : ℝ) : ℝ := log (x^2 + sqrt (1 + x^4))

theorem g_is_even : ∀ x : ℝ, g(-x) = g(x) := by
  -- Proof omitted
  sorry

end g_is_even_l692_692849


namespace combined_work_rate_l692_692768

theorem combined_work_rate (h1 : (1 / 6 : ℝ)) (h2 : (1 / 8 : ℝ)) :
  1 / ((1 / 6) + (1 / 8)) = 24 / 7 :=
begin
  sorry
end

end combined_work_rate_l692_692768


namespace largest_prime_factor_1001_l692_692364

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l692_692364


namespace meeting_lamppost_l692_692814

theorem meeting_lamppost (lamps : ℕ) (start_P : ℕ) (start_V : ℕ) (meet_P : ℕ) (meet_V : ℕ) : ℕ :=
  have intervals : ℕ := lamps - 1
  have covered_P : ℕ := meet_P - start_P
  have covered_V : ℕ := start_V - meet_V
  let ratio := (covered_P : ℝ) / (covered_V : ℝ)
  let d := (ratio * intervals.toRat) / (1 + ratio) -- Ratio of distance covered converted to Rational
  vary intervals d ℕ sorry

def main : IO Unit :=
  let lamps := 100
  let start_P := 1
  let start_V := 100
  let meet_P := 22
  let meet_V := 88
  IO.println $ meeting_lamppost lamps start_P start_V meet_P meet_V  -- Prints the 64th lamppost number.

end meeting_lamppost_l692_692814


namespace dot_product_ab_eq_3_norm_a_add_b_eq_sqrt_19_cos_angle_a_add_b_a_sub_b_eq_neg_5_sqrt_133_div_133_l692_692893

open Real

variables {a b : EuclideanSpace ℝ ℝ}

/-- The conditions for the given vectors and their properties -/
variables 
  (norm_a : ∥a∥ = 2)
  (norm_b : ∥b∥ = 3)
  (angle_ab : real.angleBetween a b = real.pi / 3)

/-- Proof that the dot product of a and b is 3 -/
theorem dot_product_ab_eq_3 : (a ⬝ b) = 3 :=
by sorry

/-- Proof that the magnitude of a + b is sqrt 19 -/
theorem norm_a_add_b_eq_sqrt_19 : ∥a + b∥ = sqrt 19 :=
by sorry

/-- Proof that the cosine of the angle between (a + b) and (a - b) is (-5 * sqrt (133)) / 133 -/
theorem cos_angle_a_add_b_a_sub_b_eq_neg_5_sqrt_133_div_133 : 
  (real.angleBetween (a + b) (a - b)).cos = (-5 * sqrt 133) / 133 :=
by sorry

end dot_product_ab_eq_3_norm_a_add_b_eq_sqrt_19_cos_angle_a_add_b_a_sub_b_eq_neg_5_sqrt_133_div_133_l692_692893


namespace prove_planes_parallel_l692_692554

-- Define planes M, N, and Q
variables (M N Q : Plane)

-- Define lines l and m
variables (l m : Line)

-- Define conditions
axiom condition_2 : Parallel M Q ∧ Parallel N Q
axiom condition_5 : (Skew l m) ∧ (Parallel l M) ∧ (Parallel m M) ∧ (Parallel l N) ∧ (Parallel m N)

-- Statement to prove
theorem prove_planes_parallel : Parallel M N :=
by
  -- proof would go here
  sorry

end prove_planes_parallel_l692_692554


namespace find_m_plus_n_l692_692088

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l692_692088


namespace exists_close_set_l692_692322

-- Definitions of conditions
def is_close (a b : ℕ) : Prop :=
  Nat.gcd a b = Nat.abs (a - b)

-- Theorem statement: for any n, there exists a set S of n elements such that any two elements of S are close.
theorem exists_close_set (n : ℕ) : ∃ S : Finset ℕ, S.card = n ∧ ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → is_close a b :=
sorry

end exists_close_set_l692_692322


namespace smallest_value_2a_minus_ab_l692_692660

theorem smallest_value_2a_minus_ab (a b : ℕ) (ha : 1 ≤ a ∧ a < 10) (hb : 1 ≤ b ∧ b < 10) : 2 * a - a * b = -63 :=
sorry

end smallest_value_2a_minus_ab_l692_692660


namespace parabola_focus_vertex_ratio_l692_692617

theorem parabola_focus_vertex_ratio :
  let P := fun (x : ℝ) => x^2
  let V1 := (0, 0)
  let F1 := (0, 1/4)
  let angle_cond (A B : ℝ × ℝ) := ∃ A B : ℝ × ℝ, A.2 = A.1^2 ∧ B.2 = B.1^2 ∧ angle A V1 B = 90
  let midpoint_locus := fun t : ℝ => (t, 2*t^2 + 1)
  let Q := fun (x : ℝ) => 2*x^2 + 1
  let V2 := (0, 1)
  let F2 := (0, 9/8) in
  dist V1 V2 ≠ 0 →
  dist F1 F2 ≠ 0 →
  dist F1 F2 / dist V1 V2 = 7 / 8 :=
by
  sorry

end parabola_focus_vertex_ratio_l692_692617


namespace boats_equation_correct_l692_692975

theorem boats_equation_correct (x : ℕ) (h1 : x ≤ 8) (h2 : 4 * x + 6 * (8 - x) = 38) : 
    4 * x + 6 * (8 - x) = 38 :=
by
  sorry

end boats_equation_correct_l692_692975
