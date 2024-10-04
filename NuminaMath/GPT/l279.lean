import Complex
import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.GCD
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.Combinatorial.Probability
import Mathlib.Combinatorics.Graph
import Mathlib.Combinatorics.Turan
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMeasure
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Init.Data.Nat.Lemmas
import Mathlib.PlanarAlgebra.Parabola
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Real

namespace derivative_at_3_l279_279369

def f (x : ℝ) : ℝ := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l279_279369


namespace largest_c_c_eq_47_l279_279229

theorem largest_c (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) : c ≤ 47 := 
begin 
  sorry 
end 

theorem c_eq_47 (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) (h_max : c ≤ 47) : c = 47 :=
begin 
  sorry 
end

end largest_c_c_eq_47_l279_279229


namespace determine_x_l279_279702

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem determine_x :
  (∀ x > 0, x ≠ 1, (1 / log_base 5 x + 1 / log_base 7 x + 1 / log_base 8 x = 1) → x = 280) :=
by
  sorry

end determine_x_l279_279702


namespace length_of_BC_l279_279446

-- Declare the function scope with conditions as assumptions
theorem length_of_BC {A B C : Type*} [metric_space A] [metric_space B] [metric_space C] 
    (cosA : real.cos A = 1/3) 
    (length_AC : AC = real.sqrt 3) 
    (area_ABC : real.triangle_area A B C = real.sqrt 2) : 
    BC = 2 := 
sorry

end length_of_BC_l279_279446


namespace number_of_divisors_of_400_l279_279569

def star (a b : ℤ) : ℤ := a * a / b

theorem number_of_divisors_of_400 :
  {x : ℕ | 20 * 20 / x > 0}.toFinset.card = 15 :=
by
  sorry

end number_of_divisors_of_400_l279_279569


namespace at_least_one_negative_l279_279496

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a^2 + 1 / b = b^2 + 1 / a) : a < 0 ∨ b < 0 :=
by
  sorry

end at_least_one_negative_l279_279496


namespace problem_l279_279014

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end problem_l279_279014


namespace sum_p_q_eq_seven_l279_279056

-- Given definitions for sets M and N
def M : set ℝ := { x | x^2 - 5 * x < 0 }
def N (p : ℝ) : set ℝ := { x | p < x ∧ x < 6 }

-- Given condition
def M_inter_N (p q : ℝ) : set ℝ := { x | 2 < x ∧ x < q }

theorem sum_p_q_eq_seven (p q : ℝ) (hM : M = { x | 0 < x ∧ x < 5 }) 
                         (hN : N p = { x | p < x ∧ x < 6 })
                         (h_inter : M ∩ N p = M_inter_N p q) : p + q = 7 :=
by
  sorry

end sum_p_q_eq_seven_l279_279056


namespace first_day_of_month_l279_279179

theorem first_day_of_month (h: ∀ n, (n % 7 = 2) → n_day_of_week n = "Wednesday"): 
  n_day_of_week 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279179


namespace sufficiency_not_necessity_l279_279844

variable (A B C : Type)
variable [metric_space A] [metric_space B] [metric_space C]

-- Define the obtuse triangle property
def is_obtuse_triangle (a b c : ℝ) : Prop := 
∃ x y z : A, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ a = dist x y ∧ b = dist y z ∧ c = dist z x ∧ 
  (∃ θ : ℝ, θ > 90 ∧ cos θ < 0)

-- Define the given condition
def condition_lt (a b c : ℝ) : Prop :=
  a^2 + b^2 < c^2 

-- The statement to prove
theorem sufficiency_not_necessity (a b c : ℝ) (h : condition_lt a b c) : is_obtuse_triangle a b c ∧ ¬ ∀ x y t : ℝ, is_obtuse_triangle x y t → condition_lt x y t :=
sorry

end sufficiency_not_necessity_l279_279844


namespace simplify_fraction_l279_279904

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l279_279904


namespace no_common_integers_l279_279967

theorem no_common_integers (a b : ℕ) (h1 : b = a + 9) : (set.range (λ n, a + n) ∩ set.range (λ n, b + n) = ∅) :=
sorry

end no_common_integers_l279_279967


namespace least_positive_integer_n_l279_279712

theorem least_positive_integer_n :
  let lhs := (∑ k in (finset.range 60).filter (λ k, k ≥ 30), (1 : ℝ) / (Real.sin (k : ℝ)) / (Real.sin (k + 1))) in
  let rhs := (1 : ℝ) / (Real.sin 60) in
  lhs = rhs := 
sorry

end least_positive_integer_n_l279_279712


namespace tangent_circles_l279_279456

noncomputable def circle_C1_in_polar (O A B : ℝ × ℝ) : ℝ := 
  let (ρ, θ) := A in 2 * √2 * cos (θ - π / 4)

noncomputable def circle_C2_in_cartesian (a : ℝ) : ℝ × ℝ × ℝ :=
  let f (θ : ℝ) := ((-1 + a * cos θ), (-1 + a * sin θ), a) in f

theorem tangent_circles (a : ℝ) (O A B : ℝ × ℝ) :
  let (x1, y1) := O,
      (x2, y2) := A,
      (x3, y3) := B in
  let r1 := √2,  -- radius of circle C1
      r2 := a,  -- radius of circle C2
      d := 2 * √2 in  -- distance between centers of C1 and C2
  sqrt(1^2 + 1^2) = r1 →   -- mid-point of OB is (1,1) which is center and radius is sqrt(2)
  r1 + r2 = d →   -- Tangent condition for two circles
  a = √2 \/ a = -√2 := sorry

end tangent_circles_l279_279456


namespace calc_m_n_l279_279933

def m (x : ℝ) : ℝ := -3 * (x + 4) * (x + 4)
def n (x : ℝ) : ℝ := (x - 3) * (x + 4)

theorem calc_m_n : (m (-1)) / (n (-1)) = 9 / 4 := 
by
  -- Given the conditions described, the proof follows from calculating m(-1) and n(-1)
  let m_val := -3 * (-1 + 4) * (-1 + 4)
  let n_val := ((-1) - 3) * ((-1) + 4)
  have h1 : m_val = -27 := by sorry
  have h2 : n_val = -12 := by sorry
  have div_val := ( -27:ℝ / -12:ℝ )
  linarith
  sorry

end calc_m_n_l279_279933


namespace sara_pumpkins_l279_279891

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end sara_pumpkins_l279_279891


namespace min_omega_l279_279875

noncomputable def f (omega : ℝ) (x : ℝ) : ℝ := real.cos (omega * x)

theorem min_omega (omega : ℝ) (h₀ : omega > 0) 
  (h₁ : ∀ x, f omega (x + π / 3) = f omega x) :
  omega = 6 :=
sorry

end min_omega_l279_279875


namespace points_per_bag_l279_279225

theorem points_per_bag
  (total_bags : ℕ)
  (bags_not_recycled : ℕ)
  (total_points : ℕ)
  (h1 : total_bags = 11)
  (h2 : bags_not_recycled = 2)
  (h3 : total_points = 45) :
  let bags_recycled := total_bags - bags_not_recycled in
  (total_points / bags_recycled) = 5 :=
by
  -- The proof goes here
  sorry

end points_per_bag_l279_279225


namespace necessary_but_not_sufficient_l279_279371

variable (p q : Prop)

theorem necessary_but_not_sufficient (hp : p) : p ∧ q ↔ p ∧ (p ∧ q → q) :=
  sorry

end necessary_but_not_sufficient_l279_279371


namespace first_day_of_month_l279_279174

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l279_279174


namespace triangle_area_l279_279948

variables {a b c : ℕ}
noncomputable def s := 5 * a
noncomputable def t := 12 * a
noncomputable def u := 13 * a

theorem triangle_area (h1 : s + t + u = 60)
                      (h2 : s^2 + t^2 = u^2) : 
  1 / 2 * s * t = 120 := 
sorry

end triangle_area_l279_279948


namespace nonnegative_interval_l279_279722

theorem nonnegative_interval (x : ℝ) : 
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3) ≥ 0 ↔ (x ≥ 0 ∧ x < 3) :=
by sorry

end nonnegative_interval_l279_279722


namespace vector_projection_theorem_l279_279763

noncomputable def vector_projection_on : ℝ := sorry

theorem vector_projection_theorem (a b : ℝ) (ha : ∥a∥ = 2) (hb : ∥b∥ = 1) (θ : ℝ) (hθ : θ = real.pi / 4) :
  vector_projection_on = real.sqrt 2 := sorry

end vector_projection_theorem_l279_279763


namespace solve_system_l279_279537

theorem solve_system : ∃ x y : ℝ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_l279_279537


namespace hyperbola_equation_l279_279047

theorem hyperbola_equation (a b : ℝ) (h1 : 2 * a - 2 * b = 2) (h2 : a^2 + b^2 = 25) (h3 : a > 0) (h4 : b > 0) (h5 : a > b) :
  \(\frac{x^2}{16} - \frac{y^2}{9} = 1\) :=
begin
  -- Required condition to prove: \frac{x^2}{16} - \frac{y^2}{9} = 1
  sorry
end

end hyperbola_equation_l279_279047


namespace area_quadrilateral_O1ADO2_l279_279963

theorem area_quadrilateral_O1ADO2
    (R : ℝ) -- radius of the circles
    (O1 O2 A D : EuclideanGeometry.Point) -- center and points
    (O1O2_touch : dist O1 O2 = 2 * R) -- condition that centers touch
    (A B C D L : EuclideanGeometry.Point) -- points of intersection
    (is_intersection : EuclideanGeometry.affineSpan ℝ [A, B, C, D] = set.range L) -- intersection condition
    (AB_eq_BC_eq_CD : dist A B = dist B C ∧ dist B C = dist C D) -- segment equality condition
    : EuclideanGeometry.area ([O1, A, D, O2]) = (5 * R^2 * Real.sqrt 3) / 4 := 
sorry

end area_quadrilateral_O1ADO2_l279_279963


namespace card_combinations_l279_279067

def choose_suits : ℕ := Nat.choose 4 1 * Nat.choose 3 2

def choose_cards (suit_cards : ℕ) (other_suits_cards : ℕ) : ℕ := (Nat.choose suit_cards 2) * (Nat.choose other_suits_cards 1) * (Nat.choose other_suits_cards 1)

theorem card_combinations:
  let suit_cards := 13 in
  let other_suits_cards := 13 in
  choose_suits * (choose_cards suit_cards other_suits_cards) = 158184 :=
by
  sorry

end card_combinations_l279_279067


namespace Eleanor_books_l279_279704

theorem Eleanor_books (h p : ℕ) : 
    h + p = 12 ∧ 28 * h + 18 * p = 276 → h = 6 :=
by
  intro hp
  sorry

end Eleanor_books_l279_279704


namespace find_f_neg_one_l279_279486

variable {R : Type} [LinearOrderedField R]

noncomputable def f (x : R) (m : R) : R :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : R) : (2^0 + 2 * 0 + m = 0) → f (-1) (-1) = -3 :=
by
  intro h1
  have h2 : m = -1 := by linarith
  rw [f, if_neg (show -1 >= 0, by linarith)]
  simp only [f, h2]
  norm_num
  sorry

end find_f_neg_one_l279_279486


namespace area_before_halving_l279_279076

theorem area_before_halving (A : ℝ) (h : A / 2 = 7) : A = 14 :=
sorry

end area_before_halving_l279_279076


namespace fraction_absent_l279_279959

theorem fraction_absent (total_students present_students : ℕ) (h1 : total_students = 28) (h2 : present_students = 20) : 
  (total_students - present_students) / total_students = 2 / 7 :=
by
  sorry

end fraction_absent_l279_279959


namespace unique_n_in_eq_l279_279915

theorem unique_n_in_eq (k x n : ℕ) (h1 : k ≥ 2) (h2 : 2^{2 * n + 1} + 2^n + 1 = x^k) : 
  n = 4 :=
by
  sorry

end unique_n_in_eq_l279_279915


namespace problem_distance_from_point_to_line_PAB_l279_279800

noncomputable def dist_point_to_line (P A B : ℝ × ℝ × ℝ) : ℝ := 
let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
let AB_dot_AP := AB.1 * AP.1 + AB.2 * AP.2 + AB.3 * AP.3 in
let AB_mag := Real.sqrt (AB.1 * AB.1 + AB.2 * AB.2 + AB.3 * AB.3) in
let AP_mag := Real.sqrt (AP.1 * AP.1 + AP.2 * AP.2 + AP.3 * AP.3) in
let cos_theta := AB_dot_AP / (AB_mag * AP_mag) in
AP_mag * Real.sqrt (1 - cos_theta * cos_theta)

theorem problem_distance_from_point_to_line_PAB :
  dist_point_to_line (1, 1, 1) (1, 0, 1) (0, 1, 0) = Real.sqrt 6 / 3 :=
by sorry

end problem_distance_from_point_to_line_PAB_l279_279800


namespace hyperbola_eccentricity_range_l279_279048

-- Definitions for the hyperbola and circle
structure Hyperbola (a b : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)

structure Circle (a : ℝ) :=
  (eqn : ∀ x y : ℝ, x^2 + y^2 - 2*a*x + (3/4)*a^2 = 0)

-- Asymptote intersection condition
def asymptote_intersects_circle (h : Hyperbola a b) (c : Circle a) : Prop :=
  ∃ x y : ℝ, (x^2 + y^2 - 2*a*x + (3/4)*a^2 = 0) ∧ (b*x - a*y = 0 ∨ b*x + a*y = 0)

-- Proof problem: Prove that the range of the eccentricity of the hyperbola is (1, 2sqrt(3)/3)
theorem hyperbola_eccentricity_range {a b : ℝ} (h : Hyperbola a b) (c : Circle a) 
  (h_intersection : asymptote_intersects_circle h c) : 1 < h.eccentricity ∧ h.eccentricity < 2 * sqrt 3 / 3 := 
sorry

end hyperbola_eccentricity_range_l279_279048


namespace find_x_l279_279421

theorem find_x (x y : ℝ) (hx : x ≠ 0) (h1 : x / 2 = y^2) (h2 : x / 4 = 4 * y) : x = 128 :=
by
  sorry

end find_x_l279_279421


namespace exponentiation_power_rule_l279_279995

theorem exponentiation_power_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_power_rule_l279_279995


namespace percentage_ginger_is_43_l279_279151

noncomputable def calculate_percentage_ginger 
  (ginger_tbsp : ℝ) (cardamom_tsp : ℝ) (mustard_tsp : ℝ) (garlic_tbsp : ℝ) (conversion_rate : ℝ) (chile_factor : ℝ) 
  : ℝ :=
  let ginger_tsp := ginger_tbsp * conversion_rate in
  let garlic_tsp := garlic_tbsp * conversion_rate in
  let chile_tsp := mustard_tsp * chile_factor in
  let total_tsp := ginger_tsp + garlic_tsp + chile_tsp + cardamom_tsp + mustard_tsp in
  (ginger_tsp / total_tsp) * 100

theorem percentage_ginger_is_43 :
  calculate_percentage_ginger 3 1 1 2 3 4 = 43 :=
by
  sorry

end percentage_ginger_is_43_l279_279151


namespace rent_percentage_increase_l279_279473

theorem rent_percentage_increase 
  (E : ℝ) 
  (h1 : ∀ (E : ℝ), rent_last_year = 0.25 * E)
  (h2 : ∀ (E : ℝ), earnings_this_year = 1.45 * E)
  (h3 : ∀ (E : ℝ), rent_this_year = 0.35 * earnings_this_year) :
  (rent_this_year / rent_last_year) * 100 = 203 := 
by 
  sorry

end rent_percentage_increase_l279_279473


namespace Sam_spent_3_dimes_on_each_candy_bar_l279_279527

theorem Sam_spent_3_dimes_on_each_candy_bar
  (initial_dimes : ℕ)
  (initial_quarters : ℕ)
  (candy_bars : ℕ)
  (lollipop_cost : ℕ)
  (amount_left : ℕ)
  (dime_value : ℕ := 10)
  (quarter_value : ℕ := 25)
  (initial_dimes = 19)
  (initial_quarters = 6)
  (candy_bars = 4)
  (lollipop_cost = 1 * quarter_value)
  (amount_left = 195) :
  candy_bars * 3 * dime_value = ((initial_dimes * dime_value) + (initial_quarters * quarter_value) - lollipop_cost - amount_left) :=
by sorry

end Sam_spent_3_dimes_on_each_candy_bar_l279_279527


namespace original_pumpkins_count_l279_279890

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end original_pumpkins_count_l279_279890


namespace integral_cos_neg_one_l279_279705

theorem integral_cos_neg_one: 
  ∫ x in (Set.Icc (Real.pi / 2) Real.pi), Real.cos x = -1 :=
by
  sorry

end integral_cos_neg_one_l279_279705


namespace min_transport_cost_l279_279605

/- Definitions for the problem conditions -/
def villageA_vegetables : ℕ := 80
def villageB_vegetables : ℕ := 60
def destinationX_requirement : ℕ := 65
def destinationY_requirement : ℕ := 75

def cost_A_to_X : ℕ := 50
def cost_A_to_Y : ℕ := 30
def cost_B_to_X : ℕ := 60
def cost_B_to_Y : ℕ := 45

def W (x : ℕ) : ℕ :=
  cost_A_to_X * x +
  cost_A_to_Y * (villageA_vegetables - x) +
  cost_B_to_X * (destinationX_requirement - x) +
  cost_B_to_Y * (x - 5) + 6075 - 225

/- Prove that the minimum total cost W is 6100 -/
theorem min_transport_cost : ∃ (x : ℕ), 5 ≤ x ∧ x ≤ 65 ∧ W x = 6100 :=
by sorry

end min_transport_cost_l279_279605


namespace number_removed_is_2_l279_279990

-- Define the list of numbers
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

-- Define the function for checking if the average of the remaining numbers is 6.5
def check_average (removed_num : ℕ) : Prop :=
  let remaining_numbers = numbers.erase removed_num
  let remaining_sum = remaining_numbers.sum
  remaining_sum = 10 * 6.5

-- The main theorem stating that removing 2 makes the average of the remaining numbers 6.5
theorem number_removed_is_2 : check_average 2 :=
by sorry

end number_removed_is_2_l279_279990


namespace nigella_commission_rate_l279_279508

-- Define the conditions
def base_salary : ℝ := 3000
def total_earnings : ℝ := 8000
def house_A_cost : ℝ := 60000
def house_B_cost : ℝ := 3 * house_A_cost
def house_C_cost : ℝ := 2 * house_A_cost - 110000
def total_house_cost : ℝ := house_A_cost + house_B_cost + house_C_cost
def commission : ℝ := total_earnings - base_salary
def commission_rate : ℝ := commission / total_house_cost

-- Prove that Nigella's commission rate is 2%
theorem nigella_commission_rate : commission_rate = 0.02 := by
  sorry

end nigella_commission_rate_l279_279508


namespace gcd_7392_15015_l279_279355

-- Define the two numbers
def num1 : ℕ := 7392
def num2 : ℕ := 15015

-- State the theorem and use sorry to omit the proof
theorem gcd_7392_15015 : Nat.gcd num1 num2 = 1 := 
  by sorry

end gcd_7392_15015_l279_279355


namespace angle_between_vectors_is_60_degrees_l279_279413

open_locale real_inner_product_space

variables {a b : ℝ}

-- Definitions extracted from conditions:
def vector_a (a b : ℝ) := ⟨a, b⟩ : ℝ²
def vector_b (a b : ℝ) := ⟨2 * a, 2 * b⟩ : ℝ²

-- Prove the angle between the two vectors is 60 degrees
theorem angle_between_vectors_is_60_degrees (a b : ℝ) :
  (vector_a a b) ⬝ ((vector_a a b) - (vector_b a b)) = 0 ∧
  2 * ∥(vector_a a b)∥ = ∥(vector_b a b)∥ →
  real.angle (vector_a a b) (vector_b a b) = real.pi / 3 := 
sorry

end angle_between_vectors_is_60_degrees_l279_279413


namespace intersect_set_A_complement_B_when_a_3_possible_values_of_a_when_intersection_is_empty_l279_279133

-- Problem 1
theorem intersect_set_A_complement_B_when_a_3 :
  let A := {x : ℝ | 3 - 3 ≤ x ∧ x ≤ 2 + 3}  -- A when a = 3
  let B := {x : ℝ | x < 1 ∨ x > 6}  -- B 
  let not_B := {x : ℝ | ¬ (x < 1 ∨ x > 6)}  -- complement of B
  A ∩ not_B = {x : ℝ | 1 ≤ x ∧ x ≤ 5} :=
by {
  let A := {x : ℝ | 3 - 3 ≤ x ∧ x ≤ 2 + 3},
  let B := {x : ℝ | x < 1 ∨ x > 6},
  let not_B := {x : ℝ | ¬ (x < 1 ∨ x > 6)},
  sorry
}

-- Problem 2
theorem possible_values_of_a_when_intersection_is_empty :
  let A := {x : ℝ | 3 - a ≤ x ∧ x ≤ 2 + a} -- A
  let B := {x : ℝ | x < 1 ∨ x > 6}  -- B
  (∀ a > 0, A ∩ B = ∅ → 0 < a ∧ a ≤ 2) :=
by {
  let A := {x : ℝ | 3 - a ≤ x ∧ x ≤ 2 + a},
  let B := {x : ℝ | x < 1 ∨ x > 6},
  sorry
}

end intersect_set_A_complement_B_when_a_3_possible_values_of_a_when_intersection_is_empty_l279_279133


namespace equation_of_line_l279_279762

theorem equation_of_line 
  (M : ℝ × ℝ) (hM : M = (1, -1))
  (A B : ℝ × ℝ)
  (hA : ∃ k l, A = (k, l) ∧ (k^2 / 4 + l^2 / 3 = 1))
  (hB : ∃ m n, B = (m, n) ∧ (m^2 / 4 + n^2 / 3 = 1))
  (h_midpoint : (1 / 2) * (A.1 + B.1) = 1 ∧ (1 / 2) * (A.2 + B.2) = -1) :
  ∃ k, ∀ x y : ℝ, (3 * x - 4 * y - 7 = 0) := 
sorry

end equation_of_line_l279_279762


namespace arithmetic_mean_exists_l279_279295

theorem arithmetic_mean_exists (A B : set ℕ) (hA_B : ∀ n, n ∈ A ∨ n ∈ B) (disjoint_A_B : ∀ a b, a ∈ A ∧ b ∈ B → a ≠ b) : 
  ∃ (G : set ℕ), (∀ a b c : ℕ, a ∈ G ∧ b ∈ G ∧ c ∈ G → (a + c = 2 * b)) :=
sorry

end arithmetic_mean_exists_l279_279295


namespace rationalize_denominator_l279_279162

theorem rationalize_denominator (h : ∀ x: ℝ, x = 1 / (Real.sqrt 3 - 2)) : 
    1 / (Real.sqrt 3 - 2) = - Real.sqrt 3 - 2 :=
by
  sorry

end rationalize_denominator_l279_279162


namespace ratio_of_dinner_to_lunch_l279_279318

theorem ratio_of_dinner_to_lunch
  (dinner: ℕ) (lunch: ℕ) (breakfast: ℕ) (k: ℕ)
  (h1: dinner = 240)
  (h2: dinner = k * lunch)
  (h3: dinner = 6 * breakfast)
  (h4: breakfast + lunch + dinner = 310) :
  dinner / lunch = 8 :=
by
  -- Proof to be completed
  sorry

end ratio_of_dinner_to_lunch_l279_279318


namespace sum_of_midpoint_coords_l279_279193

theorem sum_of_midpoint_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 1) (hy1 : y1 = 4) (hx2 : x2 = 9) (hy2 : y2 = 18) : 
  let mx := (x1 + x2) / 2 
  let my := (y1 + y2) / 2 
in mx + my = 16 :=
by
  obtain ⟨hx1, hy1, hx2, hy2⟩ := hx1, hy1, hx2, hy2
  let mx := (x1 + x2) / 2 
  let my := (y1 + y2) / 2 
  have h1 : mx = 5 := by sorry
  have h2 : my = 11 := by sorry
  calc
    mx + my = _ := by sorry

end sum_of_midpoint_coords_l279_279193


namespace first_day_of_month_l279_279186

noncomputable def day_of_week := ℕ → ℕ

def is_wednesday (n : ℕ) : Prop := day_of_week n = 3

theorem first_day_of_month (day_of_week : day_of_week) (h : is_wednesday 30) : day_of_week 1 = 2 :=
by
  sorry

end first_day_of_month_l279_279186


namespace equation_of_ellipse_maximum_AB_and_line_equation_l279_279502

-- Given Conditions
variable (a b : ℝ) (h1 : a > b) (h2 : 0 < b)
variable (F : ℝ × ℝ) (eccentricity : ℝ) (hf : eccentricity = (Real.sqrt 2) / 2)
variable (P : ℝ × ℝ)
variable (hP : P = (0, 2))
variable (line_len : ℝ) (hline_len : line_len = Real.sqrt 2)

-- Proof Statements
theorem equation_of_ellipse (h_ellipse : ∀ x y : ℝ, x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) :
  (a ^ 2 = 2 ∧ b ^ 2 = 1) → ∀ x y : ℝ, (x ^ 2) / 2 + y ^ 2 = 1 :=
sorry

theorem maximum_AB_and_line_equation (O A B : ℝ × ℝ) 
  (S_max : ∀ O A B : ℝ × ℝ, 0 < (1 / 2) * (dist A B) * (2 / (Real.sqrt (1 + (14 / 4)))) → |AB| = 3 / 2 ∧ (exists k : ℝ, k = Real.sqrt 14 / 2 ∨ k = - Real.sqrt 14 / 2)):
  true :=
sorry

end equation_of_ellipse_maximum_AB_and_line_equation_l279_279502


namespace points_distance_le_sqrt5_l279_279607

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_distance_le_sqrt5_l279_279607


namespace max_coefficient_seventh_term_l279_279453

theorem max_coefficient_seventh_term (n : ℕ) (x y : ℝ) :
  ∃ m ∈ {11, 12, 13}, (binom (n + 1) 6 = binom (n + 1) 7 ∨
  binom (n + 1) 7 = binom (n + 1) 8 ∨
  (n = 12 ∧ binom (n + 1) 7 > binom_lt_neigbors (n + 1) 6)) :=
sorry

end max_coefficient_seventh_term_l279_279453


namespace average_eq_median_mode_is_three_l279_279247

/-
  Problem Statement A: Prove that the average and median of the data 1, 2, 3, 3, 4, 5 are the same.
-/
theorem average_eq_median : 
  let data := [1, 2, 3, 3, 4, 5]
  let average := (data.sum / data.length)
  let median := ((data.nth_le (data.length / 2 - 1) sorry + data.nth_le (data.length / 2) sorry) / 2)
  average = median :=
by
  sorry

/-
  Problem Statement B: Prove that the mode of the data 6, 5, 4, 3, 3, 3, 2, 2, 1 is 3.
-/
theorem mode_is_three :
  let data := [6, 5, 4, 3, 3, 3, 2, 2, 1]
  let mode := data.mode
  mode = 3 :=
by 
  sorry

end average_eq_median_mode_is_three_l279_279247


namespace probability_of_selecting_one_defective_l279_279019

theorem probability_of_selecting_one_defective (total_products defective_products qualified_products : ℕ)
    (total_eq : total_products = 6) (defective_eq : defective_products = 2) (qualified_eq : qualified_products = 4) :
    (nat.choose defective_products 1 * nat.choose qualified_products 1) / (nat.choose total_products 2) = 8 / 15 :=
by
  rw [total_eq, defective_eq, qualified_eq]
  exact congr_arg (/15) (congr_arg2 (*) (nat.choose_eq 2 1) (nat.choose_eq 4 1)) sorry

end probability_of_selecting_one_defective_l279_279019


namespace marathon_problem_l279_279829

-- Defining the given conditions in the problem.
def john_position_right := 28
def john_position_left := 42
def mike_ahead := 10

-- Define total participants.
def total_participants := john_position_right + john_position_left - 1

-- Define Mike's positions based on the given conditions.
def mike_position_left := john_position_left - mike_ahead
def mike_position_right := john_position_right - mike_ahead

-- Proposition combining all the facts.
theorem marathon_problem :
  total_participants = 69 ∧ mike_position_left = 32 ∧ mike_position_right = 18 := by 
     sorry

end marathon_problem_l279_279829


namespace exists_a_half_l279_279869

variables {n : ℕ} (a : fin n → ℝ)

noncomputable def f (S : finset (fin n)) : ℝ :=
(S.prod (λ i, a i)) * ((finset.univ \ S).prod (λ j, 1 - a j))

theorem exists_a_half (hpos : 0 < n) (ha : ∀ i, 0 < a i ∧ a i < 1)
  (hsum : (finset.powerset (finset.univ : finset (fin n))).sum 
            (λ S, if S.card % 2 = 1 then f a S else 0) = 1 / 2) :
  ∃ k, a k = 1 / 2 :=
sorry

end exists_a_half_l279_279869


namespace number_of_members_in_club_l279_279270

theorem number_of_members_in_club :
  (∃ (x : ℕ),
    (∀ (a b : Fin 4), a ≠ b → (∃! (m : Fin x), ∈_committees m a b))
    ∧ (∀ (m : Fin x), ∃! (a b : Fin 4), a ≠ b ∧ ∈_committees m a b)) → x = 6 :=
by
  sorry

end number_of_members_in_club_l279_279270


namespace goods_train_pass_time_l279_279278

noncomputable def relative_speed (v1 v2 : ℕ) : ℝ :=
  (v1 + v2 : ℝ) * 1000 / 3600

noncomputable def time_to_pass (length : ℕ) (speed : ℝ) : ℝ :=
  length / speed

-- Given conditions as definitions in Lean
def speed_passenger_train := 50
def speed_goods_train := 62
def length_goods_train := 280

-- Lean statement for the problem
theorem goods_train_pass_time : 
  time_to_pass length_goods_train (relative_speed speed_passenger_train speed_goods_train) ≈ 8.99 :=
by sorry

end goods_train_pass_time_l279_279278


namespace possible_to_divide_square_into_convex_pentagons_l279_279113

-- Definitions based on conditions
structure ConvexPentagon :=
(vertices : List Point)
(length_eq_5 : vertices.length = 5)
(all_internal_angles_lt_180 : ∀ p₁ p₂ p₃ p₄ p₅, internal_angle p₁ p₂ p₃ < 180 ∧ internal_angle p₂ p₃ p₄ < 180 ∧ internal_angle p₃ p₄ p₅ < 180 ∧ internal_angle p₄ p₅ p₁ < 180 ∧ internal_angle p₅ p₁ p₂ < 180)

structure Square :=
(vertices : List Point)
(length_eq_4 : vertices.length = 4)
(all_right_angles : ∀ p₁ p₂ p₃ p₄, angle p₁ p₂ p₃ = 90 ∧ angle p₂ p₃ p₄ = 90 ∧ angle p₃ p₄ p₁ = 90 ∧ angle p₄ p₁ p₂ = 90)

-- Desired theorem statement
theorem possible_to_divide_square_into_convex_pentagons (s : Square) :
  ∃ (pentagons : List ConvexPentagon), (∀ p ∈ pentagons, is_convex p) ∧ covers_square pentagons s :=
sorry

end possible_to_divide_square_into_convex_pentagons_l279_279113


namespace slope_of_intersection_of_lines_l279_279365

theorem slope_of_intersection_of_lines (t : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y, 3*x - 2*y = 8*t - 5
  let l₂ : ℝ → ℝ → Prop := λ x y, 2*x + 3*y = 6*t + 9
  let l₃ : ℝ → ℝ → Prop := λ x y, x + y = 2*t + 1
  let intersection : set (ℝ × ℝ) := {p | ∃ x y, (l₁ x y ∧ l₂ x y ∧ l₃ x y) ∧ p = (x, y)}
  ∃ m : ℝ, ∀ (p1 p2 : ℝ × ℝ), p1 ∈ intersection → p2 ∈ intersection → 
    (p1.snd - p2.snd) = m * (p1.fst - p2.fst) :=
begin
  sorry
end

end slope_of_intersection_of_lines_l279_279365


namespace coloring_existence_l279_279860

-- Definitions for the problem
open Finset

def is_3_element_subset (A : Finset ℕ) : Prop := A.card = 3

def valid_coloring (n : ℕ) (A : Finset (Finset ℕ)) (colored : Finset ℕ) : Prop :=
  ∀ (a : Finset ℕ), a ∈ A → ∃ (x : ℕ), x ∈ a ∧ x ∉ colored

noncomputable def max_colored_elements (n : ℕ) : ℕ :=
  int.to_nat ⌊(2 * n / 5 : ℝ)⌋

theorem coloring_existence (n : ℕ)
  (A : Finset (Finset ℕ)) (hA : ∀ a ∈ A, is_3_element_subset a) :
  ∃ (colored : Finset ℕ), colored.card ≤ max_colored_elements n ∧ valid_coloring n A colored :=
sorry

end coloring_existence_l279_279860


namespace mary_potatoes_l279_279152

theorem mary_potatoes (original new_except : ℕ) (h₁ : original = 25) (h₂ : new_except = 7) :
  original + new_except = 32 := by
  sorry

end mary_potatoes_l279_279152


namespace final_answer_is_d_l279_279384

-- Definitions of the propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x > 1
def q : Prop := false  -- since the distance between focus and directrix is not 1/6 but 3/2

-- The statement to be proven
theorem final_answer_is_d : p ∧ ¬ q := by sorry

end final_answer_is_d_l279_279384


namespace five_digit_numbers_divisible_by_3_with_digit_6_l279_279063

theorem five_digit_numbers_divisible_by_3_with_digit_6 : 
  ∃ n : ℕ, 
    (n = 12504) ∧
    (∀ x : ℕ, 10000 ≤ x ∧ x ≤ 99999 → 
      (x % 3 = 0 → x.contains_digit(6) → x ∈ finset.range n)) := 
sorry

end five_digit_numbers_divisible_by_3_with_digit_6_l279_279063


namespace simplify_expression_l279_279908

theorem simplify_expression (x y : ℝ) (n : ℕ) (hx : x ≠ y) (hxy : 0 < x ∧ 0 < y) (hn : n = 2 ∨ n = 3 ∨ n = 4) :
    let r := (x^2 + y^2) / (2 * x * y)
    (sqrt (r + 1) - sqrt (r - 1))^n - (sqrt (r + 1) + sqrt (r - 1))^n = (y^n - x^n) * (sqrt (2 * x * y)^(-n)) ∧
    (sqrt (r + 1) - sqrt (r - 1))^n + (sqrt (r + 1) + sqrt (r - 1))^n = (y^n + x^n) * (sqrt (2 * x * y)^(-n)) →
    (sqrt (r + 1) - sqrt (r - 1))^n - (sqrt (r + 1) + sqrt (r - 1))^n / (sqrt (r + 1) - sqrt (r - 1))^n + (sqrt (r + 1) + sqrt (r - 1))^n  = 
    (y^n - x^n) / (y^n + x^n)  :=
begin
  sorry
end

end simplify_expression_l279_279908


namespace find_incorrect_statement_l279_279802

theorem find_incorrect_statement :
  let data := [2, 3, 5, 3, 7]
  let mean := (2 + 3 + 5 + 3 + 7) / 5
  let mode := 3
  let median := 3
  let variance := ((2 - 4)^2 + (3 - 4)^2 + (5 - 4)^2 + (3 - 4)^2 + (7 - 4)^2) / 4
  mean = 4 → mode = 3 → median = 3 → variance = 4 → ¬ (median = 5) :=
by
  intros data mean mode median variance hmean hmode hmedian hvariance
  rw [hmedian] -- lean proof assistant auto-refines the proof steps
  exact hmedian.symm 
  sorry

end find_incorrect_statement_l279_279802


namespace find_number_l279_279350

noncomputable def S (x : ℝ) : ℝ :=
  -- Assuming S(x) is a non-trivial function that sums the digits
  sorry

theorem find_number (x : ℝ) (hx_nonzero : x ≠ 0) (h_cond : x = (S x) / 5) : x = 1.8 :=
by
  sorry

end find_number_l279_279350


namespace percentage_ginger_is_43_l279_279150

noncomputable def calculate_percentage_ginger 
  (ginger_tbsp : ℝ) (cardamom_tsp : ℝ) (mustard_tsp : ℝ) (garlic_tbsp : ℝ) (conversion_rate : ℝ) (chile_factor : ℝ) 
  : ℝ :=
  let ginger_tsp := ginger_tbsp * conversion_rate in
  let garlic_tsp := garlic_tbsp * conversion_rate in
  let chile_tsp := mustard_tsp * chile_factor in
  let total_tsp := ginger_tsp + garlic_tsp + chile_tsp + cardamom_tsp + mustard_tsp in
  (ginger_tsp / total_tsp) * 100

theorem percentage_ginger_is_43 :
  calculate_percentage_ginger 3 1 1 2 3 4 = 43 :=
by
  sorry

end percentage_ginger_is_43_l279_279150


namespace smallest_positive_period_range_of_f_l279_279036

noncomputable def f (x : ℝ) := Real.sin x * Real.sin (x + π / 6)

theorem smallest_positive_period (T : ℝ) : (∀ x, f (x + T) = f x) ∧ (T > 0) ∧ (∀ ε, ε > 0 → ∃ x, f (x + (T - ε)) ≠ f x) :=
sorry

theorem range_of_f : Set.Icc 0 (1 / 2 + Real.sqrt 3 / 4) = (Set.image f (Set.Icc 0 (π / 2))) :=
sorry

end smallest_positive_period_range_of_f_l279_279036


namespace remainder_of_alpha_1995_mod_9_l279_279750

def b1 : ℕ := 2
def b2 : ℕ := 1
def b3 : ℕ := 2
def b4 : ℕ := 3

def g (z : ℂ) : ℂ := (1 - z)^(b1) * (1 - z^2)^(b2) * (1 - z^3)^(b3) * (1 - z^4)^(b4)

def f (x : ℂ) : ℂ := x^3 - b4 * x^2 + b2

noncomputable def alpha : ℂ := sorry -- alpha is the largest root of f(x)

noncomputable def power_result (n : ℕ) : ℤ := int.floor (complex.abs (alpha ^ n))

theorem remainder_of_alpha_1995_mod_9 : power_result 1995 % 9 = 5 := 
by
  sorry

end remainder_of_alpha_1995_mod_9_l279_279750


namespace find_angle_of_inclination_l279_279049

-- Defining the parabola y^2 = 8x
def is_on_parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Defining the line y = k(x - 2)
def is_on_line (x y k : ℝ) : Prop := y = k * (x - 2)

-- Defining the focus of the parabola
def focus_parabola : (ℝ × ℝ) := (2, 0)

-- Distance condition |AF| = 3|BF|
def distance_condition (A B F : ℝ × ℝ) : Prop :=
  let dist (P Q : ℝ × ℝ) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  in dist A F = 3 * dist B F

-- Main theorem to prove the angle of inclination of the line
theorem find_angle_of_inclination (k : ℝ) :
  (∀ x y, is_on_line x y k → is_on_parabola x y) →
  (∃ A B, is_on_parabola A.1 A.2 ∧ is_on_parabola B.1 B.2 ∧
           is_on_line A.1 A.2 k ∧ is_on_line B.1 B.2 k ∧
           distance_condition A B focus_parabola) →
  (∃ θ : ℝ, θ = real.arctan k ∧ (θ = real.pi / 3 ∨ θ = 2 * real.pi / 3)) :=
sorry

end find_angle_of_inclination_l279_279049


namespace first_day_of_month_l279_279172

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l279_279172


namespace shaded_area_floor_l279_279630

noncomputable def area_of_white_quarter_circle : ℝ := Real.pi / 4

noncomputable def area_of_white_per_tile : ℝ := 4 * area_of_white_quarter_circle

noncomputable def area_of_tile : ℝ := 4

noncomputable def shaded_area_per_tile : ℝ := area_of_tile - area_of_white_per_tile

noncomputable def number_of_tiles : ℕ := by
  have floor_area : ℝ := 12 * 15
  have tile_area : ℝ := 2 * 2
  exact Nat.floor (floor_area / tile_area)

noncomputable def total_shaded_area (num_tiles : ℕ) : ℝ := num_tiles * shaded_area_per_tile

theorem shaded_area_floor : total_shaded_area number_of_tiles = 180 - 45 * Real.pi := by
  sorry

end shaded_area_floor_l279_279630


namespace numerical_value_expression_l279_279359

theorem numerical_value_expression (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (ab + 1)) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (ab + 1) = 2 := 
by 
  -- Proof outline provided in the solution section, but actual proof is omitted
  sorry

end numerical_value_expression_l279_279359


namespace sequences_equal_l279_279804

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2018 / (n + 1)) * a (n + 1) + a n

noncomputable def b : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2020 / (n + 1)) * b (n + 1) + b n

theorem sequences_equal :
  (a 1010) / 1010 = (b 1009) / 1009 :=
sorry

end sequences_equal_l279_279804


namespace find_principal_sum_l279_279253

theorem find_principal_sum (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) 
  (hSI : SI = 4016.25) (hR : R = 9) (hT : T = 5) : P = 8925 := 
by
  sorry

end find_principal_sum_l279_279253


namespace simplify_fraction_l279_279905

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l279_279905


namespace two_digit_numbers_l279_279871

theorem two_digit_numbers :
  ∃ (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ x < y ∧ 2000 + x + y = x * y := 
sorry

end two_digit_numbers_l279_279871


namespace solve_system_eqns_l279_279535

theorem solve_system_eqns (x y z : ℝ) (h1 : x^3 + y^3 + z^3 = 8)
  (h2 : x^2 + y^2 + z^2 = 22)
  (h3 : 1/x + 1/y + 1/z + z/(x * y) = 0) :
  (x = 3 ∧ y = 2 ∧ z = -3) ∨ (x = -3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = -3) ∨ (x = 2 ∧ y = -3 ∧ z = 3) :=
by
  sorry

end solve_system_eqns_l279_279535


namespace proof_solution_l279_279683

noncomputable def proof_problem : ℝ := 
  (Real.pi - 3.14)^0 + (-1/2)^(-1) + abs (3 - Real.sqrt 8) - 4 * Real.cos (Real.pi / 4)

theorem proof_solution : proof_problem = 2 - 4 * Real.sqrt 2 :=
by
  sorry

end proof_solution_l279_279683


namespace problem_solution_l279_279834

theorem problem_solution : 
  let men := 4
  let women := 3
  let group_sizes := [2, 2, 3]
  number_of_ways_to_form_groups (men, women) group_sizes (h : ∀ g ∈ group_sizes, one_man_one_woman g) = 18 :=
sorry

def number_of_ways_to_form_groups (men_women : ℕ × ℕ) (group_sizes : list ℕ) 
(h : ∀ g ∈ group_sizes, one_man_one_woman g) : ℕ :=
3 * 6 * 1  -- Computed based on combinatorial choices

def one_man_one_woman (group_size : ℕ) : Prop :=
group_size ≥ 2

end problem_solution_l279_279834


namespace sum_of_coefficients_l279_279577

theorem sum_of_coefficients (x : ℝ) : 
  (1 - 2 * x) ^ 10 = 1 :=
sorry

end sum_of_coefficients_l279_279577


namespace tan_theta_neg2_eqn_l279_279388

theorem tan_theta_neg2_eqn : ∀ (θ : ℝ), tan θ = -2 → (7 * sin θ - 3 * cos θ) / (4 * sin θ + 5 * cos θ) = 17 / 3 := 
by
  sorry

end tan_theta_neg2_eqn_l279_279388


namespace super_cool_triangle_area_sum_l279_279652

theorem super_cool_triangle_area_sum :
  (∑ ab in { (a, b) : ℕ × ℕ | a * b / 2 = 3 * (a + b) }, a * b / 2) = 471 :=
sorry

end super_cool_triangle_area_sum_l279_279652


namespace probability_of_multiples_l279_279883

theorem probability_of_multiples (n : ℕ) (h : n = 100) : 
  (∑ k in (Finset.filter 
              (λ x, (x % 4 = 0 ∨ x % 6 = 0)) 
              (Finset.range (n + 1))
          ), 1) / n = 33 / 100 := 
by
  sorry

end probability_of_multiples_l279_279883


namespace cards_A_correct_l279_279526

def cards_A : Set ℕ := {1, 8, 9}

def cards_B (cards : Set ℕ) : Prop := 
  ∀ n ∈ cards, Nat.Prime n

def common_prime_factor_set (cards : Set ℕ) : Prop := 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n ∈ cards, p ∣ n ∧ ¬ Nat.Prime n

def cards_D (remaining_primes : Set ℕ) (C_cards : Set ℕ) (A_cards : Set ℕ) : Prop :=
  ∃ D_cards, 
    (D_cards ∩ remaining_primes).card = 2 ∧ -- D has exactly 2 primes
    ∃ m ∈ (C_cards ∪ A_cards), m ∈ D_cards -- D has 1 card from the rest

theorem cards_A_correct (cards_B cards_D common_prime_factor_set : Prop) 
  (hA : 8 ∈ cards_A)
  (hB : cards_B {b | b < 20 ∧ Nat.Prime b})
  (hC : common_prime_factor_set {x | x ∈ {4, 6, 8, 9, 10, 12}})
  (hD : cards_D {2, 3, 5, 7, 11} T) :
  cards_A = {1, 8, 9} :=
  sorry

end cards_A_correct_l279_279526


namespace problem_solution_l279_279868

noncomputable def sum_reciprocal_divisors (n : ℕ) (d : fin n → ℕ) (factorial_val: ℝ) : ℝ :=
  ∑ j, 1 / (d j + real.sqrt factorial_val)

theorem problem_solution :
  let n := 270 in
  let factorial_val := 10.factorial in
  let divisors := λ (i : fin n), nat.divisors 10.factorial i in
  sum_reciprocal_divisors n divisors factorial_val = n / 2 :=
sorry

end problem_solution_l279_279868


namespace john_spent_15_dollars_on_soap_l279_279463

-- Define the number of soap bars John bought
def num_bars : ℕ := 20

-- Define the weight of each bar of soap in pounds
def weight_per_bar : ℝ := 1.5

-- Define the cost per pound of soap in dollars
def cost_per_pound : ℝ := 0.5

-- Total weight of the soap in pounds
def total_weight : ℝ := num_bars * weight_per_bar

-- Total cost of the soap in dollars
def total_cost : ℝ := total_weight * cost_per_pound

-- Statement to prove
theorem john_spent_15_dollars_on_soap : total_cost = 15 :=
by sorry

end john_spent_15_dollars_on_soap_l279_279463


namespace percent_reduction_is_40_l279_279945

-- Definitions of conditions
def original_price : ℝ := 500
def reduction_amount : ℝ := 200

-- Definition of the percent reduction calculation
def percent_reduction (original_price reduction_amount : ℝ) : ℝ :=
  (reduction_amount / original_price) * 100

-- Statement to prove
theorem percent_reduction_is_40 :
  percent_reduction original_price reduction_amount = 40 :=
by
  -- proof will be provided here
  sorry

end percent_reduction_is_40_l279_279945


namespace sum_sin6_180_l279_279688

noncomputable def sum_sin6_deg (start : ℕ) (end_ : ℕ) : ℝ :=
  ∑ i in finset.range (end_ - start + 1), (Real.sin (Real.pi * ((start + i : ℝ)/180))) ^ 6

theorem sum_sin6_180 : sum_sin6_deg 0 180 = 229 / 4 := 
  sorry

end sum_sin6_180_l279_279688


namespace sum_of_possible_x_values_l279_279418

theorem sum_of_possible_x_values :
  ∑ x in finset.Ico 290 325, x = 10745 :=
by {
  sorry
}

end sum_of_possible_x_values_l279_279418


namespace sum_of_every_fourth_term_l279_279655

-- Define the arithmetic sequence
def arith_seq (a d n : ℕ) := a + (n - 1) * d

-- Define the conditions as given in the problem
def seq_length := 2023
def common_diff := 2
def total_sum := 6070

-- Define the sum of the whole sequence
def sum_arith_seq (a d n : ℕ) := n * (a + arith_seq a d n) / 2

-- Define the sum of every fourth term
def sum_every_fourth_term (a d n : ℕ) := (n / 4) * (a + arith_seq a d (n - 2))

-- Prove the problem statement
theorem sum_of_every_fourth_term :
  ∑ (k : ℕ) in (range 506).map (λ i, arith_seq x1 common_diff (4 * i + 1)) = 1521 :=
sorry

end sum_of_every_fourth_term_l279_279655


namespace no_rectangle_from_six_different_squares_l279_279160

theorem no_rectangle_from_six_different_squares (a1 a2 a3 a4 a5 a6 : ℝ) (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) :
  ¬ (∃ (L W : ℝ), a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = L * W) :=
sorry

end no_rectangle_from_six_different_squares_l279_279160


namespace first_day_of_month_l279_279175

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l279_279175


namespace first_day_of_month_l279_279183

theorem first_day_of_month (h : weekday 30 = "Wednesday") : weekday 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279183


namespace sum_of_products_l279_279718

def product_sum (n : ℕ) : ℕ :=
  (∑ 1 <= i < j <= n, i * j) 

theorem sum_of_products (n : ℕ) : 
  product_sum n = n * (n + 1) * (n - 1) * (3 * n + 2) / 24 := 
sorry

end sum_of_products_l279_279718


namespace boulder_weight_in_pounds_l279_279585

theorem boulder_weight_in_pounds (kg_in_pound : ℝ) (boulder_weight_kg : ℝ) (pound_conversion_factor : kg_in_pound = 0.4536) (boulder_weight : boulder_weight_kg = 350) :
  Int.nearest (boulder_weight_kg / kg_in_pound) = 772 := 
by 
  sorry

end boulder_weight_in_pounds_l279_279585


namespace biased_coin_heads_probability_l279_279237

-- Define the conditions and problem statement
theorem biased_coin_heads_probability :
  (∀ p : ℝ, 0 < p ∧ p < 1 → 
    5 * p * (1 - p)^4 = 10 * p^2 * (1 - p)^3) →
  ∃ i j : ℕ, (Nat.gcd i j = 1) → 
  let q := 10 * (p^3) * ((1 - p)^2)
  in (q = (i : ℚ) / (j : ℚ)) → 
  i + j = 283 :=
by
  sorry

end biased_coin_heads_probability_l279_279237


namespace incorrect_proposition_l279_279297

theorem incorrect_proposition
  (P1 P2 : Plane) (L : Line)
  (h1 : P1 ∥ L) -- P1 is parallel to L
  (h2 : P2 ∥ L) -- P2 is parallel to L
  : ¬ (P1 ∥ P2) -- P1 is not necessarily parallel to P2
:= sorry

end incorrect_proposition_l279_279297


namespace pick_two_different_cards_order_matters_l279_279656

theorem pick_two_different_cards_order_matters (deck_size : ℕ) (cards_chosen : ℕ) 
  (h_deck_size : deck_size = 52) (h_cards_chosen : cards_chosen = 2) :
  (deck_size * (deck_size - 1) = 2652) :=
by
  rw [h_deck_size]
  sorry

end pick_two_different_cards_order_matters_l279_279656


namespace center_of_hyperbola_is_3_4_l279_279710

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y + 72 = 0

-- Define the center of hyperbola
def center_hyperbola := (3, 4)

-- The theorem to prove the center of the hyperbola
theorem center_of_hyperbola_is_3_4 :
  ∀ x y : ℝ, hyperbola_eq x y → (x, y) = center_hyperbola :=
by
  sorry

end center_of_hyperbola_is_3_4_l279_279710


namespace probability_correct_l279_279962

-- Definitions corresponding to the problem's conditions
def boxA_tiles : Finset ℕ := Finset.range 31  -- tiles numbered 1 through 30
def boxB_tiles : Finset ℕ := Finset.range' 21 50  -- tiles numbered 21 through 50

-- Definition of the events in the problem
def eventA (n : ℕ) : Prop := n < 20

def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def eventB (n : ℕ) : Prop := is_prime n ∨ is_odd n

-- Definitions of probabilities for each event
def probability_eventA : ℚ :=
  (boxA_tiles.filter eventA).card /. boxA_tiles.card

def probability_eventB : ℚ :=
  (boxB_tiles.filter eventB).card /. boxB_tiles.card

-- Combined probability
def combined_probability : ℚ :=
  probability_eventA * probability_eventB

-- The proof problem statement
theorem probability_correct : combined_probability = 19 / 60 :=
by
  -- Sorry placeholder for proof
  sorry

end probability_correct_l279_279962


namespace simplest_quadratic_surd_is_c_l279_279242

def is_quadratic_surd (x : ℚ) : Prop := 
  ∃ (y : ℚ), y^2 = x ∧ ¬(∃ (z : ℚ), z^2 = x ∧ (z ≠ y))

def A : ℚ := real.sqrt (1 / 2)
def B : ℚ := real.cbrt (8)
def C : ℚ := real.sqrt (3)
def D : ℚ := real.sqrt (16)

theorem simplest_quadratic_surd_is_c : is_quadratic_surd C ∧ ¬is_quadratic_surd A ∧ ¬is_quadratic_surd B ∧ ¬is_quadratic_surd D :=
by
  sorry

end simplest_quadratic_surd_is_c_l279_279242


namespace intersection_A_B_l279_279408

-- Definition of sets A and B
def A : Set ℤ := {0, 1, 2, 3}
def B : Set ℤ := { x | -1 ≤ x ∧ x < 3 }

-- Statement to prove
theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := 
sorry

end intersection_A_B_l279_279408


namespace reading_time_per_week_l279_279591

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end reading_time_per_week_l279_279591


namespace central_angle_is_2_l279_279922

-- Define the conditions
def area_of_sector (r θ : ℝ) : ℝ := (1/2) * r^2 * θ

-- Given conditions in Lean:
def area : ℝ := 1  -- The area of sector OAB is 1 cm^2.
def radius : ℝ := 1  -- The radius of the sector is 1 cm.

-- Theorem stating the question and answer
theorem central_angle_is_2 : ∃ θ : ℝ, area_of_sector radius θ = area ∧ θ = 2 :=
by {
  sorry
}

end central_angle_is_2_l279_279922


namespace rotation_exists_l279_279918

-- There are 67 trainees.
def num_trainees := 67

-- Assuming the circular table's initial configuration where no one is in front of their correct plate.
def initial_configuration (f : Fin num_trainees → Fin num_trainees) : Prop :=
  ∀ (i : Fin num_trainees), f i ≠ i

-- Define the function for rotation.
def rotate (f : Fin num_trainees → Fin num_trainees) (n : Fin num_trainees) : Fin num_trainees → Fin num_trainees :=
  λ i, f ((i + n) % num_trainees)

-- Main theorem: there exists a rotation such that at least 2 trainees are correctly seated.
theorem rotation_exists (f : Fin num_trainees → Fin num_trainees) (h : initial_configuration f) :
  ∃ n : Fin num_trainees, 2 ≤ (Finset.univ.filter (λ i, rotate f n i = i)).card :=
by
  sorry

end rotation_exists_l279_279918


namespace trajectory_of_C_l279_279764

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the perimeter of triangle ABC
def perimeter (C : ℝ × ℝ) : ℝ :=
  (dist A C) + (dist B C) + (dist A B)

theorem trajectory_of_C :
  ∀ C : ℝ × ℝ,
  perimeter C = 10 →
  C ≠ (x, 0) →
  (x^2 / 9 + y^2 / 5 = 1) :=
begin
  sorry,
end

end trajectory_of_C_l279_279764


namespace mother_time_l279_279265

-- Define the times taken by each family member
def time_father : ℕ := 1
def time_son : ℕ := 4
def time_daughter : ℕ := 5
def cave_explosion_time : ℕ := 12

-- Define the conditions
def max_people_in_tunnel : ℕ := 2
def afraid_of_going_alone : bool := true

-- The theorem to prove: mother takes 2 minutes to go through the tunnel
theorem mother_time (time_mother : ℕ) (total_time : ℕ) :
  (total_time = time_father + time_daughter + time_father + time_son + time_mother) ∧
  (total_time ≤ cave_explosion_time) → time_mother = 2 :=
by
  sorry

end mother_time_l279_279265


namespace samantha_routes_eq_800_l279_279888

theorem samantha_routes_eq_800 :
  let north_west_ways := Nat.choose (3 + 3) 3,
      east_south_ways := Nat.choose (3 + 3) 3,
      shortcut_paths := 2
  in north_west_ways * east_south_ways * shortcut_paths = 800 :=
by
  let north_west_ways := Nat.choose (3 + 3) 3
  let east_south_ways := Nat.choose (3 + 3) 3
  let shortcut_paths := 2
  show north_west_ways * east_south_ways * shortcut_paths = 800
  sorry

end samantha_routes_eq_800_l279_279888


namespace average_age_new_students_l279_279548

theorem average_age_new_students (O A_old A_new_avg A_new : ℕ) 
  (hO : O = 8) 
  (hA_old : A_old = 40) 
  (hA_new_avg : A_new_avg = 36)
  (h_total_age_before : O * A_old = 8 * 40)
  (h_total_age_after : (O + 8) * A_new_avg = 16 * 36)
  (h_age_new_students : (16 * 36) - (8 * 40) = A_new * 8) :
  A_new = 32 := 
by 
  sorry

end average_age_new_students_l279_279548


namespace max_people_in_company_l279_279086

-- Define the finite type for people in the company
variable {α : Type*} [Fintype α]

-- Define the mutual friendship relation
variable (friends : α → α → Prop)
variable [symmetric : ∀ x y, friends x y → friends y x]

-- Define the predicate to count pairs of friends in a subset 
noncomputable def odd_friends_in_subset [DecidableEq α] (s : Finset α) : Prop :=
  (s.card.choose 2) % 2 = 1

-- Define the main theorem statement
theorem max_people_in_company (h : ∀ (s : Finset α), s.card = 101 → odd_friends_in_subset friends s) :
  Fintype.card α ≤ 102 :=
sorry

end max_people_in_company_l279_279086


namespace number_of_primes_in_sequence_l279_279998

-- Define the sequence generation
def sequence : Nat → Nat
| 0       => 2
| (n + 1) => Nat.ofDigits [6, 1, 0, 2] (sequence n)

-- Define a predicate to check if a number is prime
def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Find the number of primes in the first n terms of the sequence
def number_of_primes (n : Nat) : Nat :=
  List.length (List.filter is_prime (List.ofFn sequence n))

theorem number_of_primes_in_sequence : number_of_primes 10 = 1 := 
sorry

end number_of_primes_in_sequence_l279_279998


namespace fish_population_estimate_l279_279084

-- Define the conditions from each round
variable (C1 C2 R2 R3 : ℕ)

-- The given conditions translated to Lean definitions
def condition_round_2 : C1 = 80 := rfl
def condition_round_3_c2 : C2 = 100 := rfl
def condition_round_3_r2 : R3 = 10 := rfl

theorem fish_population_estimate (hc1 : C1 = 80) (hc2 : C2 = 100) (hr2 : R3 = 10) : 
    (C1 * C2) / R3 = 800 :=
by 
  rw [hc1, hc2, hr2]
  norm_num
  rfl

end fish_population_estimate_l279_279084


namespace hypotenuse_length_l279_279566

-- Definitions based on conditions
def longer_leg (x : ℝ) := 3 * x - 1
def area (x : ℝ) (longer_leg : ℝ) := (1 / 2) * x * longer_leg

-- Hypothesis based on conditions
theorem hypotenuse_length (x : ℝ) (hx : area x (longer_leg x) = 24) : 
  let shorter_leg := x
  let longer_leg := longer_leg x
  let hypotenuse := Math.sqrt (shorter_leg^2 + longer_leg^2)
  hypotenuse = Math.sqrt 137 := 
by 
  sorry

end hypotenuse_length_l279_279566


namespace nancy_games_this_month_l279_279880

-- Define the variables and conditions from the problem
def went_games_last_month : ℕ := 8
def plans_games_next_month : ℕ := 7
def total_games : ℕ := 24

-- Let's calculate the games this month and state the theorem
def games_last_and_next : ℕ := went_games_last_month + plans_games_next_month
def games_this_month : ℕ := total_games - games_last_and_next

-- The theorem statement
theorem nancy_games_this_month : games_this_month = 9 := by
  -- Proof is omitted for the sake of brevity
  sorry

end nancy_games_this_month_l279_279880


namespace bottle_caps_bought_l279_279466

theorem bottle_caps_bought (original_caps new_caps bought_caps : ℕ) 
    (h_original : original_caps = 40) 
    (h_new : new_caps = 47) : 
    bought_caps = new_caps - original_caps :=
by {
    rw [h_original, h_new],
    exact eq.refl 7,
}

end bottle_caps_bought_l279_279466


namespace zero_not_in_range_of_g_l279_279695

def g (x : ℝ) : ℤ :=
  if x > -1 then ⌈2 / ((x+2) * (x+1))⌉ 
  else if x < -2 then ⌊2 / ((x+2) * (x+1))⌋ 
  else 0 -- function undefined in [-2, -1] is represented by a dummy value

theorem zero_not_in_range_of_g : ¬ (∃ x : ℝ, g x = 0) :=
by
  sorry

end zero_not_in_range_of_g_l279_279695


namespace quadratic_value_at_three_l279_279565

theorem quadratic_value_at_three :
  ∃ (a b c : ℝ), (∀ x, a * x ^ 2 + b * x + c ≥ -4) ∧
                 (a * (-2) ^ 2 + b * (-2) + c = -4) ∧
                 (a * 0 ^ 2 + b * 0 + c = 8) ∧
                 (a * 3 ^ 2 + b * 3 + c = 71) :=
begin
  -- Proof is omitted
  sorry
end

end quadratic_value_at_three_l279_279565


namespace arithmetic_sequence_a15_l279_279100

variable {α : Type*} [LinearOrderedField α]

-- Conditions for the arithmetic sequence
variable (a : ℕ → α)
variable (d : α)
variable (a1 : α)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 5)
variable (h_a10 : a 10 = 15)

-- To prove that a15 = 25
theorem arithmetic_sequence_a15 : a 15 = 25 := by
  sorry

end arithmetic_sequence_a15_l279_279100


namespace probability_at_least_one_woman_selected_l279_279430

theorem probability_at_least_one_woman_selected:
  let men := 10
  let women := 5
  let totalPeople := men + women
  let totalSelections := Nat.choose totalPeople 4
  let menSelections := Nat.choose men 4
  let noWomenProbability := (menSelections : ℚ) / (totalSelections : ℚ)
  let atLeastOneWomanProbability := 1 - noWomenProbability
  atLeastOneWomanProbability = 11 / 13 :=
by
  sorry

end probability_at_least_one_woman_selected_l279_279430


namespace trees_died_due_to_typhoon_l279_279663

theorem trees_died_due_to_typhoon (initial_trees trees_left trees_died : ℕ) (h_initial : initial_trees = 12) (h_left : trees_left = 10) : 
  trees_died = initial_trees - trees_left → trees_died = 2 :=
by
  intros h_trees_died
  rw [h_initial, h_left] at h_trees_died
  rw [h_trees_died]
  simp
  sorry

end trees_died_due_to_typhoon_l279_279663


namespace tetrahedron_solution_l279_279375

noncomputable def num_triangles (a : ℝ) (E F G : ℝ → ℝ → ℝ) : ℝ :=
  if a > 3 then 3 else 0

theorem tetrahedron_solution (a : ℝ) (E F G : ℝ → ℝ → ℝ) :
  a > 3 → num_triangles a E F G = 3 := by
  sorry

end tetrahedron_solution_l279_279375


namespace weight_differences_correct_l279_279495

-- Define the weights of Heather, Emily, Elizabeth, and Emma
def H : ℕ := 87
def E1 : ℕ := 58
def E2 : ℕ := 56
def E3 : ℕ := 64

-- Proof problem statement
theorem weight_differences_correct :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) :=
by
  -- Note: 'sorry' is used to skip the proof itself
  sorry

end weight_differences_correct_l279_279495


namespace more_girls_than_boys_l279_279946

theorem more_girls_than_boys (num_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ) (total_students : ℕ) (total_students_eq : num_students = 42) (ratio_eq : boys_ratio = 3 ∧ girls_ratio = 4) : (4 * 6) - (3 * 6) = 6 := by
  sorry

end more_girls_than_boys_l279_279946


namespace find_b_value_l279_279301

open Real

noncomputable def isEllipseWithFociPassingThrough (h k a b : ℝ) (pt1 pt2 : ℝ × ℝ) (pt : ℝ × ℝ) : Prop :=
  (distance (pt.1, pt.2) pt1) + (distance (pt.1, pt.2) pt2) = 2 * a ∧
  (h, k) = ((pt1.1 + pt2.1) / 2, (pt1.2 + pt2.2) / 2) ∧
  pt1.1 = pt2.1 -- same x-coordinate implies major axis is vertical

theorem find_b_value :
  ∃ b : ℝ,
  isEllipseWithFociPassingThrough 1 0 (sqrt 37) b (1, 1) (1, -1) (7, 0) ∧
  b = 6 :=
sorry

end find_b_value_l279_279301


namespace paula_candies_l279_279511

theorem paula_candies (initial_candies : ℕ) (additional_candies : ℕ) (total_friends : ℕ) :
  initial_candies = 35 → additional_candies = 15 → total_friends = 10 → 
  (initial_candies + additional_candies) / total_friends = 5 :=
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end paula_candies_l279_279511


namespace factorization_correct_l279_279709

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by
  sorry

end factorization_correct_l279_279709


namespace part1_part2_l279_279794

-- Define the original function
def f (x : ℝ) (ω : ℝ) : ℝ := sin (ω * x) * cos (ω * x) + sqrt 3 * cos (ω * x) ^ 2 - sqrt 3 / 2

-- Define conditions
axiom ω_gt_zero (ω : ℝ) : ω > 0
def symm_axes_property (x1 x2 : ℝ) (f : ℝ → ℝ) : Prop := x1 ≠ x2 ∧ f (x1) = f (x2)
def min_dist (x1 x2 : ℝ) : Prop := abs (x1 - x2) = π / 4

-- Theorem for Part (1)
theorem part1 (ω : ℝ) (hω : ω_gt_zero ω) :
  (∀ x : ℝ, f(x) (2) = sin (4 * x + π / 3)) :=
sorry

-- Transformations for Part (2)
def g (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Defining the condition for exactly one solution in the interval [0, π/2]
def one_solution (g : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ g x + k = 0

-- Theorem for Part (2)
theorem part2 : ∀ k : ℝ, one_solution g k ↔ k ∈ Ioc (-1/2) 1/2 ∨ k = -1 :=
sorry

end part1_part2_l279_279794


namespace find_S30_l279_279741

-- Definitions for conditions (arithmetic sequence and sum conditions)
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ (n : ℕ), a (n+1) - a n = a 1 - a 0

def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
S 0 = 0 ∧ (∀ (n : ℕ), S (n+1) = S n + a n)

-- Given conditions
def S10 : ℕ := 12
def S20 : ℕ := 17

-- The statement we are proving
theorem find_S30 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_arith : arithmetic_sequence a) (h_sum : sum_of_first_n_terms S a) :
  S 10 = 12 → S 20 = 17 → S 30 = 22 :=
by
  intros hS10 hS20
  have h1 : S10 = S 10 := rfl
  have h2 : S20 = S 20 := rfl
  sorry

end find_S30_l279_279741


namespace smallest_x_satisfies_eq_l279_279716

theorem smallest_x_satisfies_eq : ∃ x : ℝ, (1 / (x - 5) + 1 / (x - 7) = 5 / (2 * (x - 6))) ∧ x = 7 - Real.sqrt 6 :=
by
  -- The proof steps would go here, but we're skipping them with sorry for now.
  sorry

end smallest_x_satisfies_eq_l279_279716


namespace number_of_triangles_l279_279135

theorem number_of_triangles (n : ℕ) (h₁ : n = 30) (h₂ : ∀ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ c ≠ a → 
  a < b ∧ b < c → 
  ∃ d e f, d ≠ e ∧ e ≠ f ∧ f ≠ d ∧ (b - a) ≥ 3 ∧ (c - b) ≥ 3 ∧ (d - c) ≥ 3) :
  ∃ t : ℕ, t = 2530 :=
by
  sorry

end number_of_triangles_l279_279135


namespace tyler_cucumbers_for_apples_l279_279817

variable (A B C : Type)
variables (cost : A → ℝ) (apples : A) (bananas : B) (cucumbers : C)

-- Definitions from the conditions.
def price_equiv (x y : Type) (cost_x : x → ℝ) (cost_y : y → ℝ) (equiv_units : ℝ) : Prop :=
  cost_x ≃ cost_y

variable (price_equiv_apples_bananas : price_equiv _ _ cost apples bananas 0.5)
variable (price_equiv_bananas_cucumbers : price_equiv _ _ cost bananas cucumbers 0.8)

-- Proving the main question
theorem tyler_cucumbers_for_apples :
  price_equiv _ _ cost apples cucumbers 0.4 :=
sorry

end tyler_cucumbers_for_apples_l279_279817


namespace daily_B_mask_production_maximize_profit_l279_279218

def total_masks := 2_000_000
def total_days := 30
def ratio_A_to_B := 2
def profit_A := 0.8
def profit_B := 1.2
def delta_days := 6
def time_for_50M_B (B_per_day : ℝ) := 50_000_000 / B_per_day
def time_for_40M_A (A_per_day : ℝ) := 40_000_000 / A_per_day

theorem daily_B_mask_production (B_per_day : ℝ) :
  (time_for_50M_B B_per_day) - (time_for_40M_A (ratio_A_to_B * B_per_day)) = delta_days →
  B_per_day = 5 :=
  sorry

theorem maximize_profit (A_per_day B_per_day : ℝ) :
  A_per_day = 2 * B_per_day →
  0 < A_per_day ∧ 0 < B_per_day →
  A_per_day * total_days + B_per_day * total_days = total_masks →
  ∃ (a: ℝ), a = 100_000_000 ∧ 200_000_000 - a = 100_000_000 :=
  sorry

end daily_B_mask_production_maximize_profit_l279_279218


namespace rice_mixed_grain_amount_l279_279919

theorem rice_mixed_grain_amount (total_rice : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) (proportion : ℚ) 
    (h1 : total_rice = 1536) 
    (h2 : sample_size = 256)
    (h3 : mixed_in_sample = 18)
    (h4 : proportion = mixed_in_sample / sample_size) : 
    total_rice * proportion = 108 :=
  sorry

end rice_mixed_grain_amount_l279_279919


namespace length_ST_l279_279102

theorem length_ST (PQ QR RS SP SQ PT RT : ℝ) 
  (h1 : PQ = 6) (h2 : QR = 6)
  (h3 : RS = 6) (h4 : SP = 6)
  (h5 : SQ = 6) (h6 : PT = 14)
  (h7 : RT = 14) : 
  ∃ ST : ℝ, ST = 10 := 
by
  -- sorry is used to complete the theorem without a proof
  sorry

end length_ST_l279_279102


namespace trigonometric_identity_l279_279520

-- The theorem to prove
theorem trigonometric_identity (α β γ : ℝ) :
  sin α + sin β + sin γ - sin (α + β) * cos γ - cos (α + β) * sin γ = 
  4 * sin ((α + β) / 2) * sin ((β + γ) / 2) * sin ((γ + α) / 2) := 
sorry

end trigonometric_identity_l279_279520


namespace circle_intersect_length_l279_279126

theorem circle_intersect_length (A B C D : Point) (P Q X Y: Point)
  (hA : A ∈ circle)
  (hB : B ∈ circle)
  (hC : C ∈ circle)
  (hD : D ∈ circle)
  (hAB : distance A B = 15)
  (hCD : distance C D = 23)
  (hP : P ∈ segment A B)
  (hQ : Q ∈ segment C D)
  (hAP : distance A P = 9)
  (hCQ : distance C Q = 11)
  (hPQ : distance P Q = 35):
  distance X Y = 68 := 
by
  sorry

end circle_intersect_length_l279_279126


namespace time_reading_per_week_l279_279593

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l279_279593


namespace slope_of_line_l279_279977

theorem slope_of_line : ∀ x y : ℝ, 3 * y + 2 * x = 6 * x - 9 → ∃ m b : ℝ, y = m * x + b ∧ m = -4 / 3 :=
by
  -- Sorry to skip proof
  sorry

end slope_of_line_l279_279977


namespace find_integer_k_l279_279005

theorem find_integer_k {k : ℤ} :
  (∀ x : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 →
    (∃ m n : ℝ, m ≠ n ∧ m * n = 1 / (k^2 + 1) ∧ m + n = (4 - k) / (k^2 + 1) ∧
      ((1 < m ∧ n < 1) ∨ (1 < n ∧ m < 1)))) →
  k = -1 ∨ k = 0 :=
by
  sorry

end find_integer_k_l279_279005


namespace hyperbola_eccentricity_sqrt2_l279_279797

theorem hyperbola_eccentricity_sqrt2
  (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (h_asymptote : ∀ {P : ℝ × ℝ}, (P.1 / a = P.2 / b) → 
    (∃ (M : ℝ × ℝ), (M.1 / a = -M.2 / b ∧ (M.1 = (P.1 + (sqrt (a^2 + b^2))) / 2) ∧ 
    (M.2 = (P.2 + 0) / 2))))
  (h_angle : ∀ {P : ℝ × ℝ}, (P.1 / a = P.2 / b) → (∃ (F1 F2 : ℝ × ℝ), 
    (∠ P F2 F1 = 45))) :
  eccentricity a b = real.sqrt 2 :=
begin
  sorry
end

end hyperbola_eccentricity_sqrt2_l279_279797


namespace f_of_f_half_l279_279789

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then (2^x - x^3) else Real.log x / Real.log 2

theorem f_of_f_half : f (f (1 / 2)) = 3 / 2 :=
by
  sorry

end f_of_f_half_l279_279789


namespace find_f_neg_one_l279_279488

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end find_f_neg_one_l279_279488


namespace probability_two_more_sons_or_daughters_l279_279879

theorem probability_two_more_sons_or_daughters (n : ℕ) (p : ℚ) (k : ℕ) :
  n = 8 → p = 1 / 2 → k = 2 →
  (prob_at_least_two_more_sons_or_daughters n p k) = 37 / 128 :=
by
  sorry

-- Definitions used
def prob_at_least_two_more_sons_or_daughters (n : ℕ) (p : ℚ) (k : ℕ) : rat :=
  let total_combinations := (bit0 256 : rat)
  let equal_sons_daughters := nat.choose n (n / 2)
  let one_more_or_less_son_daughter := nat.choose n (n / 2 + 1)
  let non_favorable_cases := equal_sons_daughters + 2 * one_more_or_less_son_daughter
  (total_combinations - non_favorable_cases) / total_combinations

end probability_two_more_sons_or_daughters_l279_279879


namespace smallest_base10_integer_l279_279986

def is_valid_digit_base_6 (C : ℕ) : Prop := C ≤ 5
def is_valid_digit_base_8 (D : ℕ) : Prop := D ≤ 7

def CC_6_to_base10 (C : ℕ) : ℕ := 7 * C
def DD_8_to_base10 (D : ℕ) : ℕ := 9 * D

theorem smallest_base10_integer : ∃ C D : ℕ, 
  is_valid_digit_base_6 C ∧ 
  is_valid_digit_base_8 D ∧ 
  CC_6_to_base10 C = DD_8_to_base10 D ∧
  CC_6_to_base10 C = 63 := 
begin
  sorry
end

end smallest_base10_integer_l279_279986


namespace Trevor_future_age_when_brother_is_three_times_now_l279_279220

def Trevor_current_age := 11
def Brother_current_age := 20

theorem Trevor_future_age_when_brother_is_three_times_now :
  ∃ (X : ℕ), Brother_current_age + (X - Trevor_current_age) = 3 * Trevor_current_age :=
by
  use 24
  sorry

end Trevor_future_age_when_brother_is_three_times_now_l279_279220


namespace total_eggs_collected_by_all_four_l279_279310

def benjamin_eggs := 6
def carla_eggs := 3 * benjamin_eggs
def trisha_eggs := benjamin_eggs - 4
def david_eggs := 2 * trisha_eggs

theorem total_eggs_collected_by_all_four :
  benjamin_eggs + carla_eggs + trisha_eggs + david_eggs = 30 := by
  sorry

end total_eggs_collected_by_all_four_l279_279310


namespace minimum_value_of_f_l279_279356

noncomputable def f (x : ℝ) : ℝ := x^2 + 10*x + (100 / x^3)

theorem minimum_value_of_f :
  ∃ x > 0, f x = 40 ∧ ∀ y > 0, f y ≥ 40 :=
sorry

end minimum_value_of_f_l279_279356


namespace find_f_neg_one_l279_279487

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end find_f_neg_one_l279_279487


namespace angle_sum_eq_angle_l279_279858

-- Definitions for the triangle ABC and its centroid G.
variables {A B C G R S : Type}
-- Assume A, B, and C are distinct points
variables [distinct_points A B C]
-- G is the centroid of triangle ABC.
def is_centroid (A B C G : Type) : Prop :=
  -- Define centroid condition for G if necessary
  sorry

-- Points R and S are on rays GB and GC respectively
def on_ray (G X Y : Type) : Prop :=
  -- Define that Y is on ray GX if necessary
  sorry

-- Given the angle condition
def angle_condition (A B C G R S : Type) : Prop :=
  sorry

axioms
  (angle_ABC_G : angle_condition A B C G R S)
  (centroid_G : is_centroid A B C G)
  (on_ray_GB : on_ray G B R)
  (on_ray_GC : on_ray G C S)

-- We must prove that ∠ RAS + ∠ BAC = ∠ BGC.
theorem angle_sum_eq_angle (A B C G R S : Type)
  (angle_ABC_G : angle_condition A B C G R S)
  (centroid_G : is_centroid A B C G)
  (on_ray_GB : on_ray G B R)
  (on_ray_GC : on_ray G C S) :
  sorry := 
  sorry

end angle_sum_eq_angle_l279_279858


namespace residents_rent_contribution_l279_279572

theorem residents_rent_contribution (x R : ℝ) (hx1 : 10 * x + 88 = R) (hx2 : 10.80 * x = 1.025 * R) :
  R / x = 10.54 :=
by sorry

end residents_rent_contribution_l279_279572


namespace cost_price_proof_l279_279647

noncomputable def cost_price_per_bowl : ℚ := 1400 / 103

theorem cost_price_proof
  (total_bowls: ℕ) (sold_bowls: ℕ) (selling_price_per_bowl: ℚ)
  (percentage_gain: ℚ) 
  (total_bowls_eq: total_bowls = 110)
  (sold_bowls_eq: sold_bowls = 100)
  (selling_price_per_bowl_eq: selling_price_per_bowl = 14)
  (percentage_gain_eq: percentage_gain = 300 / 11) :
  (selling_price_per_bowl * sold_bowls - (sold_bowls + 3) * (selling_price_per_bowl / (3 * percentage_gain / 100))) = cost_price_per_bowl :=
by
  sorry

end cost_price_proof_l279_279647


namespace length_CD_l279_279050

theorem length_CD (m : ℝ) (A B C D : ℝ × ℝ) (l intersects_circle : Prop)
  (line_eqn : ∀ (x y : ℝ), l ↔ m * x + y + 3 * m - real.sqrt 3 = 0)
  (circle_eqn : ∀ (x y : ℝ), intersects_circle ↔ x^2 + y^2 = 12)
  (AB_eq : dist A B = 2 * real.sqrt 3)
  (perpendiculars_intersect_x_axis : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → ∃ x, (x, 0) = P ∧ ¬ l) :
  dist C D = 4 :=
sorry

end length_CD_l279_279050


namespace ads_on_first_web_page_l279_279972

theorem ads_on_first_web_page 
  (A : ℕ)
  (second_page_ads : ℕ := 2 * A)
  (third_page_ads : ℕ := 2 * A + 24)
  (fourth_page_ads : ℕ := 3 * A / 2)
  (total_ads : ℕ := 68 * 3 / 2)
  (sum_of_ads : A + 2 * A + (2 * A + 24) + 3 * A / 2 = total_ads) :
  A = 12 := 
by
  sorry

end ads_on_first_web_page_l279_279972


namespace prob_eq_l279_279958

noncomputable def prob_defective_components_found_at_r (p q r : ℕ) (hpq : q < p) (hrp : p < r) (hrpq : r < p + q) : ℝ :=
  (choose q 1 * -- binom{q}{1}
    (factorial (r - 1) / factorial (r - 1 - (q - 1))) * -- ℎ_{r-1}^{(q-1)}
    (factorial p / factorial (p - (r - q))) + -- ℎ_{p}^{(r-q)}
    choose p 1 * -- binom{p}{1}
    (factorial (r - 1) / factorial (r - 1 - (p - 1))) * -- ℎ_{r-1}^{(p-1)}
    (factorial q / factorial (q - (r - p)))) / -- ℎ_{q}^{(r-p)}
    (factorial (p + q) / factorial ((p + q) - r)) -- ℎ_{(p+q)}^{r}

theorem prob_eq (p q r : ℕ) (hpq : q < p) (hrp : p < r) (hrpq : r < p + q) :
  prob_defective_components_found_at_r p q r hpq hrp hrpq =
  ((choose q 1 * (factorial (r-1) / factorial (r-1 - (q-1))) * 
  (factorial p / factorial (p - (r - q)))) + 
  (choose p 1 * 
  (factorial (r-1) / factorial (r-1 - (p-1))) * 
  (factorial q / factorial (q - (r - p))))) / 
  (factorial (p + q) / factorial ((p + q) - r)) :=
sorry

end prob_eq_l279_279958


namespace distance_between_axes_of_symmetry_l279_279407

theorem distance_between_axes_of_symmetry (m : ℝ) (h : 0 < m ∧ 0 < m^2 ∧ 
  ∀ x, (x = -m ∨ x = 0 ∨ x = m ∨ x = m^2) → 
        (y = -x^2 + m^2 * x ∨ y = x^2 - m^2) → 
          (y = 0 ∧ y.has_root x)) : 
  abs (m^2 / 2 - 0) = 2 := 
by
  sorry

end distance_between_axes_of_symmetry_l279_279407


namespace adjusted_mean_is_89_l279_279146

-- Defining the six test scores
def scores : List ℕ := [84, 76, 90, 97, 85, 88]

-- Function to drop the lowest score from the list
def drop_lowest (scores : List ℕ) : List ℕ :=
  List.erase scores (List.minimum scores).get_or_else 0

-- Function to calculate the arithmetic mean of a list of scores
def arithmetic_mean (scores : List ℕ) : ℚ :=
  (List.sum scores : ℚ) / (List.length scores : ℚ)

-- Problem statement as a Lean theorem
theorem adjusted_mean_is_89 :
  arithmetic_mean (drop_lowest scores) = 89 := sorry

end adjusted_mean_is_89_l279_279146


namespace brick_height_l279_279267

-- Define the known values
def wall_length := 850 -- in cm
def wall_width := 600 -- in cm
def wall_height := 22.5 -- in cm
def brick_length := 25 -- in cm
def brick_width := 11.25 -- in cm
def num_bricks := 6800 -- total bricks
def wall_volume := wall_length * wall_width * wall_height -- total volume of wall in cubic centimeters
def brick_base_area := brick_length * brick_width -- base area of one brick in square centimeters

-- Prove that the height of each brick is 5.98 cm given the conditions
theorem brick_height : (wall_volume / (num_bricks * brick_base_area) = 5.98) :=
by
  -- This remains unproven, but the statement itself is specified.
  sorry

end brick_height_l279_279267


namespace minimum_perimeter_l279_279145

def fractional_part (x : ℝ) : ℝ := x - Real.floor x

def meets_conditions (ℓ m n : ℕ) : Prop :=
  (ℓ > m ∧ m > n) ∧ 
  ((fractional_part ((3:ℝ)^ℓ / 10^4) = fractional_part ((3:ℝ)^m / 10^4)) ∧ 
   (fractional_part ((3:ℝ)^m / 10^4) = fractional_part ((3:ℝ)^n / 10^4)))

theorem minimum_perimeter (ℓ m n : ℕ) (h : meets_conditions ℓ m n) : ℓ + m + n = 3003 :=
sorry

end minimum_perimeter_l279_279145


namespace room_area_in_square_meters_l279_279283

theorem room_area_in_square_meters :
  ∀ (length_ft width_ft : ℝ), 
  (length_ft = 15) → 
  (width_ft = 8) → 
  (1 / 9 * 0.836127 = 0.092903) → 
  (length_ft * width_ft * 0.092903 = 11.14836) :=
by
  intros length_ft width_ft h_length h_width h_conversion
  -- sorry to skip the proof steps.
  sorry

end room_area_in_square_meters_l279_279283


namespace debate_team_girls_l279_279950

theorem debate_team_girls (boys: ℕ) (groups: ℕ) (members_per_group: ℕ) 
  (members_total: ℕ) (girls: ℕ) 
  (h1: boys = 26) 
  (h2: groups = 8) 
  (h3: members_per_group = 9) 
  (h4: members_total = groups * members_per_group): 
  members_total - boys = girls → girls = 46 :=
by 
  intros h
  rw [h1, h2, h3] at h4
  have ht : members_total = 72 := h4
  rw ht at h
  exact h

end debate_team_girls_l279_279950


namespace value_of_a_minus_b_l279_279812

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) : a - b = 7 ∨ a - b = 3 :=
sorry

end value_of_a_minus_b_l279_279812


namespace greatest_product_l279_279228

theorem greatest_product (x : ℤ) (h : x + (1998 - x) = 1998) : 
  x * (1998 - x) ≤ 998001 :=
  sorry

end greatest_product_l279_279228


namespace exists_m_for_triangle_area_l279_279022

theorem exists_m_for_triangle_area :
  ∃ (m : ℝ), let f : ℝ → ℝ := λ y, (y - 1)^2 + y^2 - 4 in
             let line := λ x y, x - m * y + 1 = 0 in
             let intersect_points := {a : ℝ × ℝ | f a.2 = 0 ∧ line a.1 a.2} in
             triangle_area intersect_points  = 8 / 5 :=
begin
  -- Triangle area and other definitions would be done here
  -- Proof will be added here
  sorry
end

end exists_m_for_triangle_area_l279_279022


namespace product_of_distinct_solutions_l279_279754

theorem product_of_distinct_solutions (x y : ℝ) (h₁ : x ≠ y) (h₂ : x ≠ 0) (h₃ : y ≠ 0) (h₄ : x - 2 / x = y - 2 / y) :
  x * y = -2 :=
sorry

end product_of_distinct_solutions_l279_279754


namespace simplest_quadratic_surd_is_c_l279_279243

def is_quadratic_surd (x : ℚ) : Prop := 
  ∃ (y : ℚ), y^2 = x ∧ ¬(∃ (z : ℚ), z^2 = x ∧ (z ≠ y))

def A : ℚ := real.sqrt (1 / 2)
def B : ℚ := real.cbrt (8)
def C : ℚ := real.sqrt (3)
def D : ℚ := real.sqrt (16)

theorem simplest_quadratic_surd_is_c : is_quadratic_surd C ∧ ¬is_quadratic_surd A ∧ ¬is_quadratic_surd B ∧ ¬is_quadratic_surd D :=
by
  sorry

end simplest_quadratic_surd_is_c_l279_279243


namespace wheel_center_distance_l279_279969

-- Conditions as definitions
def radius := 1
def circumference (r : ℝ) := 2 * Real.pi * r

-- Main theorem statement
theorem wheel_center_distance :
  let r := (radius : ℝ) in
  circumference r = 2 * Real.pi :=
by
-- Skipping the proof steps
  sorry

end wheel_center_distance_l279_279969


namespace sufficient_but_not_necessary_condition_l279_279873

def P (x : ℝ) : Prop := 0 < x ∧ x < 5
def Q (x : ℝ) : Prop := |x - 2| < 3

theorem sufficient_but_not_necessary_condition
  (x : ℝ) : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end sufficient_but_not_necessary_condition_l279_279873


namespace simplest_quadratic_surd_is_sqrt3_l279_279244

def isQuadraticSurd (x : ℝ) : Prop := ∃ n : ℕ, (n > 0) ∧ (¬ (∃ m : ℕ, m^2 = n)) ∧ (x = Real.sqrt n)

def simplestQuadraticSurd : ℝ :=
  if isQuadraticSurd (Real.sqrt (1/2)) then Real.sqrt (1/2)
  else if isQuadraticSurd (Real.cbrt 8) then Real.cbrt 8
  else if isQuadraticSurd (Real.sqrt 3) then Real.sqrt 3
  else if isQuadraticSurd (Real.sqrt 16) then Real.sqrt 16
  else 0

theorem simplest_quadratic_surd_is_sqrt3 :
  simplestQuadraticSurd = Real.sqrt 3 :=
by
  sorry

end simplest_quadratic_surd_is_sqrt3_l279_279244


namespace smallest_n_for_f_n_4_l279_279482

def f (n : ℕ) : ℕ :=
  (Finset.univ.filter (λ p : ℕ × ℕ, p.1 ≠ p.2 ∧ p.1^2 + p.2^2 = n)).card

theorem smallest_n_for_f_n_4 : ∃ n : ℕ, f(n) = 4 ∧ ∀ m : ℕ, m < n → f(m) ≠ 4 :=
  by
    existsi 65
    split
    sorry -- proof that f(65) = 4 goes here
    intros m hm
    sorry -- proof that f(m) ≠ 4 for all m < 65 goes here

end smallest_n_for_f_n_4_l279_279482


namespace simplify_and_evaluate_expression_l279_279531

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 / (x + 2) + x - 2) / ((x^2 - 2*x + 1) / (x + 2))

theorem simplify_and_evaluate_expression (x : ℝ) (hx : |x| = 2) (h_ne : x ≠ -2) :
  given_expression x = 3 :=
by
  sorry

end simplify_and_evaluate_expression_l279_279531


namespace rhombus_area_l279_279393

theorem rhombus_area (side_length : ℝ) (d1_diff_d2 : ℝ) 
  (h_side_length : side_length = Real.sqrt 104) 
  (h_d1_diff_d2 : d1_diff_d2 = 10) : 
  (1 / 2) * (2 * Real.sqrt 104 - d1_diff_d2) * (d1_diff_d2 + 2 * Real.sqrt 104) = 79.17 :=
by
  sorry

end rhombus_area_l279_279393


namespace batsman_average_after_15th_inning_l279_279251

theorem batsman_average_after_15th_inning (x : ℕ) (hx : (14 * x + 75) / 15 = x + 3) : 
  (x + 3 = 33) :=
by
  have : x = 30 := by sorry
  rw this
  exact Eq.refl 33

end batsman_average_after_15th_inning_l279_279251


namespace at_least_26_equal_differences_l279_279124

theorem at_least_26_equal_differences (x : Fin 102 → ℕ) (h : ∀ i j, i < j → x i < x j) (h' : ∀ i, x i < 255) :
  (∃ d : Fin 101 → ℕ, ∃ s : Finset ℕ, s.card ≥ 26 ∧ (∀ i, d i = x i.succ - x i) ∧ ∃ i j, i ≠ j ∧ (d i = d j)) :=
by {
  sorry
}

end at_least_26_equal_differences_l279_279124


namespace ratio_of_cubes_l279_279274

/-- A cubical block of metal weighs 7 pounds. Another cube of the same metal, with sides of a certain ratio longer, weighs 56 pounds. Prove that the ratio of the side length of the second cube to the first cube is 2:1. --/
theorem ratio_of_cubes (s r : ℝ) (weight1 weight2 : ℝ)
  (h1 : weight1 = 7) (h2 : weight2 = 56)
  (h_vol1 : weight1 = s^3)
  (h_vol2 : weight2 = (r * s)^3) :
  r = 2 := 
sorry

end ratio_of_cubes_l279_279274


namespace magnitude_of_expression_l279_279731

def complex_z : ℂ := 3 + complex.i

theorem magnitude_of_expression :
  abs ((complex_z ^ 2) - 3 * complex_z) = real.sqrt 10 := by
  sorry

end magnitude_of_expression_l279_279731


namespace find_expression_l279_279015

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end find_expression_l279_279015


namespace mn_ab_squared_l279_279601

-- Given data
variables {R : ℝ} {M N A B : EuclideanGeometry.Point}
variables (S₁ S₂ : EuclideanGeometry.Circle R) -- Circles with radius R
variables (O₁ O₂ : EuclideanGeometry.Point) -- Centers of the circles S₁ and S₂
variables (h₁ : S₁.contains M) (h₂ : S₁.contains N) (h₃ : S₂.contains M) (h₄ : S₂.contains N)
variables (h₅ : EuclideanGeometry.PerpendicularBisector A B M N)
variables (h₆ : EuclideanGeometry.SameSide MN A B)

-- To be proved
theorem mn_ab_squared (h₁ : S₁.contains M) (h₂ : S₁.contains N) (h₃ : S₂.contains M) (h₄ : S₂.contains N)
  (h₅ : EuclideanGeometry.PerpendicularBisector A B M N) (h₆ : EuclideanGeometry.SameSide MN A B) :
  EuclideanGeometry.distance M N ^ 2 + EuclideanGeometry.distance A B ^ 2 = 4 * R ^ 2 :=
sorry

end mn_ab_squared_l279_279601


namespace geom_sequence_sum_l279_279454

theorem geom_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, a n > 0)
  (h_geom : ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q)
  (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 :=
sorry

end geom_sequence_sum_l279_279454


namespace power_of_four_l279_279814

theorem power_of_four (x : ℕ) (h : 5^29 * 4^x = 2 * 10^29) : x = 15 := by
  sorry

end power_of_four_l279_279814


namespace trigonometric_identity_l279_279031

theorem trigonometric_identity (m : ℝ) (h : m < 0) :
  let x := -4 * m,
      y := 3 * m,
      r := -5 * m in
  2 * (y / r) + (x / r) = -2 / 5 := by
  sorry

end trigonometric_identity_l279_279031


namespace solve_inequality_l279_279910

theorem solve_inequality (x : ℝ) :
  3 * 7^(2 * x) + 37 * 140^x < 26 * 20^(2 * x) ↔ x ≥ Real.log (2 / 3) / Real.log (7 / 20) :=
sorry

end solve_inequality_l279_279910


namespace total_pictures_painted_l279_279667

def pictures_painted_in_june : ℕ := 2
def pictures_painted_in_july : ℕ := 2
def pictures_painted_in_august : ℕ := 9

theorem total_pictures_painted : 
  pictures_painted_in_june + pictures_painted_in_july + pictures_painted_in_august = 13 :=
by
  sorry

end total_pictures_painted_l279_279667


namespace point_on_segment_ratio_l279_279479

theorem point_on_segment_ratio (A B P : Type) 
  [AddCommGroup P] [Module ℝ P] [AffineSpace P A B] 
  (t u : ℝ) (AP_ : (4 : ℝ), PB_ : (1 : ℝ)) (h : (AP_ / PB_ = 4 / 1)) :
  \(\overrightarrow{P}\) = t * \(\overrightarrow{A}\) + u * \(\overrightarrow{B}\) → 
  (t, u) = (4/5, 1/5) := by
sorry

end point_on_segment_ratio_l279_279479


namespace decomposition_l279_279616

-- Define the vectors x, p, q, r
def x : ℝ × ℝ × ℝ := (3, 3, -1)
def p : ℝ × ℝ × ℝ := (3, 1, 0)
def q : ℝ × ℝ × ℝ := (-1, 2, 1)
def r : ℝ × ℝ × ℝ := (-1, 0, 2)

-- Define the decomposition theorem
theorem decomposition : x = (p.1 + q.1 - r.1, p.2 + q.2 - r.2, p.3 + q.3 - r.3) :=
by
  -- Proof will be constructed here
  sorry

end decomposition_l279_279616


namespace intersects_l279_279007

noncomputable theory

variables {Point : Type} [AffineSpace ℝ Point] 

-- Given points on a rectangular parallelepiped
variable (A B C D A1 B1 C1 D1 K P H E : Point)

-- Midpoints and ratios
def midpoint (x y : Point) : Point := (x + y) / 2

def B_midpoint : Prop := K = midpoint B B1
def D1_midpoint : Prop := P = midpoint A1 D1
def C1_midpoint : Prop := H = midpoint C C1
def B1_C1_ratio : Prop := ∃ λ : ℝ, E = B1 + λ • (C1 - B1) ∧ 1 / λ = 3

-- Line intersection
def line (p1 p2 : Point) : set Point := { r | ∃ t : ℝ, r = p1 + t • (p2 - p1) }
def K_AE_intersect : Prop := ∃ t s : ℝ, K + t • (P - K) = E + s • (A - E)
def K_A1H_intersect : Prop := ∃ t u : ℝ, K + t • (P - K) = H + u • (A1 - H)

-- Statement to prove
theorem intersects : B_midpoint ∧ D1_midpoint ∧ C1_midpoint ∧ B1_C1_ratio → K_AE_intersect ∧ K_A1H_intersect := by
  sorry

end intersects_l279_279007


namespace find_f_neg_one_l279_279490

def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m)
def m : ℝ := -1

theorem find_f_neg_one (m : ℝ) (h_m : m = -1) (h_odd : ∀ x : ℝ, f (-x) = -f x) : f (-1) = -3 := 
by
  have h_def : f x = if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m),
  from sorry,
  sorry

end find_f_neg_one_l279_279490


namespace binomial_square_given_expression_value_l279_279235

theorem binomial_square (a b : ℕ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 :=
by sorry

theorem given_expression_value : 17 ^ 2 + 2 * 17 * 5 + 5 ^ 2 = 484 :=
by {
  have h : (17 + 5) ^ 2 = 17 ^ 2 + 2 * 17 * 5 + 5 ^ 2,
  { exact binomial_square 17 5 },
  rw [←nat.add_mul_self_eq],
  rw [h],
  norm_num,
  rfl,
  sorry
}

end binomial_square_given_expression_value_l279_279235


namespace solution_positive_iff_k_range_l279_279953

theorem solution_positive_iff_k_range (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (k / (2 * x - 4) - 1 = x / (x - 2))) ↔ (k > -4 ∧ k ≠ 4) := 
sorry

end solution_positive_iff_k_range_l279_279953


namespace andy_late_duration_l279_279669

theorem andy_late_duration :
  let start_time := 7 * 60 + 15 in -- 7:15 AM converted to minutes
  let school_start := 8 * 60 in -- 8:00 AM converted to minutes
  let normal_travel_time := 30 in
  let red_light_stops := 3 * 4 in
  let construction_delay := 10 in
  let total_travel_time := normal_travel_time + red_light_stops + construction_delay in
  total_travel_time - (school_start - start_time) = 7 :=
by
  sorry

end andy_late_duration_l279_279669


namespace problem1_problem2_l279_279534

-- Define conditions for Problem 1
def problem1_cond (x : ℝ) : Prop :=
  x ≠ 0 ∧ 2 * x ≠ 1

-- Statement for Problem 1
theorem problem1 (x : ℝ) (h : problem1_cond x) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by
  sorry

-- Define conditions for Problem 2
def problem2_cond (x : ℝ) : Prop :=
  x ≠ 2 

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h : problem2_cond x) :
  ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ↔ x = 1 := by
  sorry

end problem1_problem2_l279_279534


namespace smallest_integer_repr_CCCD8_l279_279978

theorem smallest_integer_repr_CCCD8 (C D : ℕ) (hC : C < 6) (hD : D < 8)
    (h_eq : 7 * C = 9 * D) : ∃ n : ℕ, (n = 7 * C) ∧ (n = 9 * D) ∧ (7 * C = 63) :=
by {
  existsi 63,
  split,
  { simp [←h_eq, mul_comm, mul_assoc, Nat.mul_div_cancel_left, Nat.gcd_eq, Nat.lcm_eq_gcd_mul] },
  { exact h_eq }
}

end smallest_integer_repr_CCCD8_l279_279978


namespace prisoner_release_possible_l279_279625

-- Define the problem scenario
def prisonerScenario : Prop :=
∀ (num_prisoners : ℕ) (counter : ℕ) (initial_lamp_state : bool),
  num_prisoners = 100 ∧ counter < num_prisoners ∧ initial_lamp_state = false →
  ∃ (strategy : (ℕ × bool) → ℕ × bool), 
    (∀ (steps : ℕ), steps ≥ num_prisoners - 1 → 
      (let final_state := (strategy^[steps] (0, initial_lamp_state)) in 
      fst final_state = num_prisoners - 1))

-- The theorem statement: the strategy exists for the prisoner problem
theorem prisoner_release_possible : prisonerScenario :=
sorry

end prisoner_release_possible_l279_279625


namespace max_area_central_angle_l279_279024

theorem max_area_central_angle (r l : ℝ) (S α : ℝ) (h1 : 2 * r + l = 4)
  (h2 : S = (1 / 2) * l * r) : (∀ x y : ℝ, (1 / 2) * x * y ≤ (1 / 4) * ((x + y) / 2) ^ 2) → α = l / r → α = 2 :=
by
  sorry

end max_area_central_angle_l279_279024


namespace min_dist_sum_l279_279475

-- Definitions of points and distances
structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Given conditions
def A : Point := ⟨0, 0⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨12, 0⟩

-- Line m is perpendicular to line ℓ at point A, so we consider point P on the y-axis
variable (d : ℝ) -- distance along the y-axis

def P : Point := ⟨0, d⟩

-- Distances from P to B and C
noncomputable def PB : ℝ := distance P B
noncomputable def PC : ℝ := distance P C

-- Theorem statement
theorem min_dist_sum : ∀ d : ℝ, (PB d + PC d) ≥ 19 :=
by
  sorry

end min_dist_sum_l279_279475


namespace correct_statements_in_problem_l279_279298

theorem correct_statements_in_problem :
  let correct1 := false in
  let correct2 := true in
  let correct3 := false in
  let correct4 := false in
  let correct5 := true in
  [correct1, correct2, correct3, correct4, correct5] = [false, true, false, false, true] :=
by
  simp only [correct1, correct2, correct3, correct4, correct5]
  trivial

end correct_statements_in_problem_l279_279298


namespace number_of_typists_needed_l279_279072

theorem number_of_typists_needed :
  (∃ t : ℕ, (20 * 40) / 20 * 60 * t = 180) ↔ t = 30 :=
by sorry

end number_of_typists_needed_l279_279072


namespace elizabeth_needs_more_cents_l279_279343

theorem elizabeth_needs_more_cents 
  (pencil_cost : ℤ) 
  (elizabeth_has : ℤ) 
  (borrowed : ℤ) 
  (additional_needed : ℤ) : 
  (pencil_cost = 600) → (elizabeth_has = 500) → (borrowed = 53) → (additional_needed = 47) → 
  (additional_needed = pencil_cost - (elizabeth_has + borrowed)) :=
begin
  intros,
  sorry
end

end elizabeth_needs_more_cents_l279_279343


namespace solve_abs_inequality_l279_279536

theorem solve_abs_inequality (x : ℝ) : x + |2 * x + 3| ≥ 2 ↔ (x ≤ -5 ∨ x ≥ -1/3) :=
by {
  sorry
}

end solve_abs_inequality_l279_279536


namespace part1_part2_l279_279745

-- Definitions based on conditions in part (a)
def O := (0 : ℝ, 0 : ℝ)
def A := (1 : ℝ, 1 : ℝ)
def B := (-1 : ℝ, 0 : ℝ)

def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def OA := vec O A
def OB := vec O B
def AB := vec A B

def OC (λ : ℝ) : ℝ × ℝ := 
    (OA.1 + λ * OB.1, OA.2 + λ * OB.2)

-- \perp and \cdot definitions
def perpendicular (v w : ℝ × ℝ) : Prop :=
    v.1 * w.1 + v.2 * w.2 = 0

-- Proof of part (1)
theorem part1 (λ : ℝ) (h : perpendicular (OC λ) AB) : λ = 3 / 2 :=
sorry

-- Cosine of the angle between vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
    v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
    real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_angle (v w : ℝ × ℝ) : ℝ :=
    dot_product v w / (magnitude v * magnitude w)

-- Proof of part (2)
theorem part2 : cos_angle OA OB = - (real.sqrt 2) / 2 :=
sorry

end part1_part2_l279_279745


namespace average_rate_of_change_is_4_l279_279041

def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change_is_4 : 
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_is_4_l279_279041


namespace sqrt_simplification_l279_279313

variable (q : ℝ)

theorem sqrt_simplification : 
  sqrt (15 * q) * sqrt (8 * q^2) * sqrt (14 * q^3) = 4 * q^3 * sqrt 105 :=
by
  sorry

end sqrt_simplification_l279_279313


namespace sulfuric_acid_percentage_l279_279115

theorem sulfuric_acid_percentage :
  ∀ (x y : ℕ), y = 18 → x + y = 60 → 
  ((0.02 * x + 0.12 * y) / 60 * 100) = 5 := by
  intros x y hy hxy
  have hx : x = 42 := by linarith
  sorry

end sulfuric_acid_percentage_l279_279115


namespace speed_of_man_l279_279288

theorem speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ)
  (relative_speed_km_h : ℝ)
  (h_train_length : train_length = 440)
  (h_train_speed : train_speed_kmph = 60)
  (h_time : time_seconds = 24)
  (h_relative_speed : relative_speed_km_h = (train_length / time_seconds) * 3.6):
  (relative_speed_km_h - train_speed_kmph) = 6 :=
by sorry

end speed_of_man_l279_279288


namespace line_AM_bisects_DPrimeX_common_radical_point_exists_l279_279757

variables 
  (A B C H O S M D D' E F X Y Z G L : Type) 
  (triangle_non_isosceles_acute : ∀ (A B C : Type), Triangle A B C)
  (circumcircle : ∀ (A B C O : Type), Circle A B C O)
  (orthocenter : ∀ (A B C H : Type), Orthocenter A B C H)
  (midpoint : ∀ (B C M : Type), Midpoint B C M)
  (circle_with_diameter_bc : ∀ (B C D D' : Type), CircleWithDiameter B C D D')
  (line_intersection : ∀ (A H D D' : Type), LineIntersection A H D D')
  (line_intersection_ao_md : ∀ (A O M D X : Type), LineIntersectionAO_MD A O M D X)
  (tangents_intersection : ∀ (O B C S : Type), TangentsIntersection O B C S)
  (projection : ∀ (A S G : Type), Projection A S G)

-- Part 1: Line AM bisects segment D'X
theorem line_AM_bisects_DPrimeX :
  ∀ (A M D' X : Type), LineBisectsSegment A M D' X := sorry

-- Part 2: Common radical point exists for four circles
theorem common_radical_point_exists :
  ∃ (L : Type), EqualPowerWithRespectToFourCircles L O S G B Y E C Z F := sorry

end line_AM_bisects_DPrimeX_common_radical_point_exists_l279_279757


namespace common_chord_length_l279_279600

theorem common_chord_length (r : ℝ) (d : ℝ) (h_r : r = 12) (h_d : d = 12) :
  ∃ l : ℝ, l = 12 * Real.sqrt 3 :=
by
  -- Conditions given in the problem
  have h_radius : r = 12 := h_r
  have h_distance : d = 12 := h_d
  -- The correct answer to be proved
  use 12 * Real.sqrt 3
  sorry

end common_chord_length_l279_279600


namespace quadrilateral_circumscribed_l279_279550

open EuclideanGeometry

-- Define the conditions of the problem
variables {A B C P K D : Point}
variable [Triangle (A B C)]
variable (hAB_gt_AC : length B A > length C A)

-- Define the points and their properties
variable (h_angle_bisector : AngleBisector (A) B C P)
variable (h_perpendicular : Perpendicular (AC) C (A K))
variable (h_circle_with_center_P : Circle P (distance P K) (arc_minor P A D))

-- The theorem to be proven
theorem quadrilateral_circumscribed :
  length A B + length C D = length A C + length B D :=
by
  sorry

end quadrilateral_circumscribed_l279_279550


namespace lowest_n_for_K6_l279_279882

theorem lowest_n_for_K6 (G : SimpleGraph (Fin 1991)) (h : ∀ v : Fin 1991, G.degree v ≥ 1593) :
  ∃ (S : Finset (Fin 1991)), S.card = 6 ∧ (∀ u v ∈ S, G.Adj u v) :=
by
  sorry

end lowest_n_for_K6_l279_279882


namespace iodine_initial_amount_l279_279850

theorem iodine_initial_amount (half_life : ℕ) (days_elapsed : ℕ) (final_amount : ℕ) (initial_amount : ℕ) :
  half_life = 8 → days_elapsed = 24 → final_amount = 2 → initial_amount = final_amount * 2 ^ (days_elapsed / half_life) → initial_amount = 16 :=
by
  intros h_half_life h_days_elapsed h_final_amount h_initial_exp
  rw [h_half_life, h_days_elapsed, h_final_amount] at h_initial_exp
  norm_num at h_initial_exp
  exact h_initial_exp

end iodine_initial_amount_l279_279850


namespace angle_bisector_proportion_l279_279824

theorem angle_bisector_proportion (P Q R E : Type) 
  (p q r x y : ℝ) 
  (triangle_PQR : Triangle P Q R)
  (PE_bisects_P : BisectsAngle P E)
  (E_on_QR : OnSegment E Q R)
  (x_eq_QE : x = distance Q E)
  (y_eq_RE : y = distance R E) 
  (hx : x + y = p) 
  (h_eq_ratio : ∀ (a b c d : ℝ), a b = c d ↔ a / b = c / d) :
  (y / q = p / (r + q)) := 
sorry

end angle_bisector_proportion_l279_279824


namespace sin_cos_15_sin_cos_18_l279_279259

theorem sin_cos_15 (h45sin : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h45cos : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h30sin : Real.sin (30 * Real.pi / 180) = 1 / 2)
                  (h30cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2) :
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 ∧
  Real.cos (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

theorem sin_cos_18 (h18sin : Real.sin (18 * Real.pi / 180) = (-1 + Real.sqrt 5) / 4)
                   (h36cos : Real.cos (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 4) :
  Real.cos (18 * Real.pi / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4 := by
  sorry

end sin_cos_15_sin_cos_18_l279_279259


namespace total_time_of_four_sets_of_stairs_l279_279320

def time_first : ℕ := 15
def time_increment : ℕ := 10
def num_sets : ℕ := 4

theorem total_time_of_four_sets_of_stairs :
  let a := time_first
  let d := time_increment
  let n := num_sets
  let l := a + (n - 1) * d
  let S := n / 2 * (a + l)
  S = 120 :=
by
  sorry

end total_time_of_four_sets_of_stairs_l279_279320


namespace arithmetic_sequence_range_of_t_l279_279354

theorem arithmetic_sequence_range_of_t (d : ℝ) (a n : ℕ) (t : ℝ) (h1 : a = -1)
  (h2 : ∀ n : ℕ, n > 0 → ∑ i in finset.range n, (a + i * d) > t) : t ∈ set.Iio (-6) :=
begin
  sorry

end arithmetic_sequence_range_of_t_l279_279354


namespace sine_2010_deg_equal_neg_one_half_l279_279211

/-- The sine function has a period of 360 degrees, meaning ∀ k ∈ ℤ, sin θ = sin (θ + k * 360). 
    Prove that sin 2010° = -1/2. -/
theorem sine_2010_deg_equal_neg_one_half : real.sin (2010 * real.pi / 180) = -1/2 :=
  sorry

end sine_2010_deg_equal_neg_one_half_l279_279211


namespace jefferson_bananas_l279_279118

variable (J W : ℕ)

theorem jefferson_bananas:
  W = J - (1/4 : ℚ) * J →
  2 * W = 49 * 2 →
  J = 56 :=
by
-- We assume W = 49 because Walter gets 49 bananas after sharing equally
  assume h1 : W = J - (1/4 : ℚ) * J,
  assume h2 : 2 * W = 49 * 2,
  sorry

end jefferson_bananas_l279_279118


namespace series_inequality_l279_279154

theorem series_inequality (n : ℕ) (hn : n ≥ 2) :
  1 + ∑ k in Finset.range n \ {0, 1}, (1 : ℝ) / (k + 1) ^ 2 < (2 * n - 1 : ℝ) / n :=
by
  sorry

end series_inequality_l279_279154


namespace part_a_l279_279171

theorem part_a (x : ℝ) (hx : x > 0) :
  ∃ color : ℕ, ∃ p1 p2 : ℝ × ℝ, (p1 = p2 ∨ x = dist p1 p2) :=
sorry

end part_a_l279_279171


namespace missing_number_is_4_l279_279347

theorem missing_number_is_4 (x : ℝ) : 11 + sqrt(-4 + 6 * x / 3) = 13 → x = 4 :=
by
  sorry

end missing_number_is_4_l279_279347


namespace problem1_value_of_m_problem2_value_of_sinA_plus_cosB_l279_279400

-- Define the function f
def f (x m : ℝ) : ℝ := (cos x * (sqrt 3 * sin x - cos x)) + m

-- Define the transformation to get g
def g (x m : ℝ) : ℝ := f (x + π/6) m

-- Problem (1): Prove the value of m 
theorem problem1_value_of_m (m : ℝ) :
  (∀ x ∈ set.Icc (π/4) (π/3), g x m ≥ f (x + π/6) m) →
  m = sqrt 3 / 2 := 
sorry

-- Problem (2): Prove the range of values for sin A + cos B
theorem problem2_value_of_sinA_plus_cosB (A B C : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (hC : 0 < C ∧ C < π/2) :
  g (C/2) (sqrt 3 / 2) = - 1/2 + sqrt 3 →
  (set.Icc (π/3) (π/2) ⊆ { sin A + cos B | A B : ℝ } ∧ ∀ x ∈ set.Icc (π/6) (π/3), sqrt 3 / 2 < sqrt 3 * sin (x - π/6)) →
  √3 * sin (A - π/6) ∈ set.Icc (√3 / 2) (3 / 2) :=
sorry

end problem1_value_of_m_problem2_value_of_sinA_plus_cosB_l279_279400


namespace wizard_collection_value_l279_279291

theorem wizard_collection_value :
  let crystal_ball := nat.of_digits 7 [3, 4, 2, 6]
  let wand := nat.of_digits 7 [0, 5, 6, 1]
  let book_of_spells := nat.of_digits 7 [2, 0, 2]
  crystal_ball + wand + book_of_spells = 2959 :=
by
  let crystal_ball := nat.of_digits 7 [3, 4, 2, 6]
  let wand := nat.of_digits 7 [0, 5, 6, 1]
  let book_of_spells := nat.of_digits 7 [2, 0, 2]
  sorry

end wizard_collection_value_l279_279291


namespace mail_sent_in_august_is_2000_l279_279223

-- Define the conditions
def mail_sent_july : ℕ := 40
def mail_sent_business_day_august : ℕ := 2 * mail_sent_july
def mail_sent_holiday_august : ℕ := mail_sent_july / 2
def number_business_days_august : ℕ := 23
def number_holidays_august : ℕ := 8

-- Define the computed total mail in August
def total_mail_august : ℕ :=
  (number_business_days_august * mail_sent_business_day_august) +
  (number_holidays_august * mail_sent_holiday_august)

-- The proof goal
theorem mail_sent_in_august_is_2000 : total_mail_august = 2000 :=
by {
  -- Total mail sent on business days
  have total_mail_business_days := number_business_days_august * mail_sent_business_day_august,
  -- Total mail sent on holidays
  have total_mail_holidays := number_holidays_august * mail_sent_holiday_august,
  -- Summing both to get the total mail in August
  have total_mail := total_mail_business_days + total_mail_holidays,
  -- Exact calculation to equate with 2000
  exact calc
  total_mail_august = total_mail : by rfl
                   ... = 2000 : sorry  -- Provide proof here
}

end mail_sent_in_august_is_2000_l279_279223


namespace stratified_sampling_l279_279633

variable (M F Fs : ℕ)
variable (Ms : ℕ := M * Fs / F)

theorem stratified_sampling (M F Fs : ℕ) (hM : M = 1200) (hF : F = 1000) (hFs : Fs = 80) :
  M * Fs / F + Fs = 176 :=
by 
  rw [hM, hF, hFs]
  sorry

end stratified_sampling_l279_279633


namespace leading_digits_sum_l279_279128

def M : ℕ := 10^499 * 10 - 1  -- M is a 500-digit number of all 9s

def leading_digit (x : ℝ) : ℕ :=
    let d := x / 10^((Mathlib.Real.floor (Mathlib.Real.log x)))
    Mathlib.Nat.floor d

def g (r : ℕ) : ℕ := leading_digit (M ^ (1 / r : ℝ))

theorem leading_digits_sum : g 2 + g 4 + g 5 + g 7 + g 9 = 8 := by
    sorry

end leading_digits_sum_l279_279128


namespace james_total_points_l279_279116

-- Defining the conditions given in the problem
def field_goals := 13
def field_goal_points := 3
def shots := 20
def shot_points := 2
def free_throws := 5
def free_throw_points := 1

-- Calculating the total points
def total_points : ℕ := (field_goals * field_goal_points) + (shots * shot_points) + (free_throws * free_throw_points)

-- Theorem stating the total points James scored
theorem james_total_points : total_points = 84 := by
  unfold total_points
  norm_num

end james_total_points_l279_279116


namespace opposite_blue_face_is_white_l279_279896

-- Define colors
inductive Color
| Red
| Blue
| Orange
| Purple
| Green
| Yellow
| White

-- Define the positions of colors on the cube
structure CubeConfig :=
(top : Color)
(front : Color)
(bottom : Color)
(back : Color)
(left : Color)
(right : Color)

-- The given conditions
def cube_conditions (c : CubeConfig) : Prop :=
  c.top = Color.Purple ∧
  c.front = Color.Green ∧
  c.bottom = Color.Yellow ∧
  c.back = Color.Orange ∧
  c.left = Color.Blue ∧
  c.right = Color.White

-- The statement we need to prove
theorem opposite_blue_face_is_white (c : CubeConfig) (h : cube_conditions c) :
  c.right = Color.White :=
by
  -- Proof placeholder
  sorry

end opposite_blue_face_is_white_l279_279896


namespace find_d_l279_279079

theorem find_d (c d : ℝ) (h1 : c / d = 5) (h2 : c = 18 - 7 * d) : d = 3 / 2 := by
  sorry

end find_d_l279_279079


namespace first_day_of_month_l279_279176

theorem first_day_of_month (h: ∀ n, (n % 7 = 2) → n_day_of_week n = "Wednesday"): 
  n_day_of_week 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279176


namespace smallest_integer_repr_CCCD8_l279_279981

theorem smallest_integer_repr_CCCD8 (C D : ℕ) (hC : C < 6) (hD : D < 8)
    (h_eq : 7 * C = 9 * D) : ∃ n : ℕ, (n = 7 * C) ∧ (n = 9 * D) ∧ (7 * C = 63) :=
by {
  existsi 63,
  split,
  { simp [←h_eq, mul_comm, mul_assoc, Nat.mul_div_cancel_left, Nat.gcd_eq, Nat.lcm_eq_gcd_mul] },
  { exact h_eq }
}

end smallest_integer_repr_CCCD8_l279_279981


namespace solution_eq_three_l279_279025

theorem solution_eq_three (a : ℝ) : (∃ x, (∀ x, (sqrt (3 * x + 1) + sqrt (3 * x + 6) = sqrt (4 * x - 2) + sqrt (4 * x + 3)) → x = a) → a = 3) :=
by
  sorry

end solution_eq_three_l279_279025


namespace series_sum_eq_five_l279_279687

open Nat Real

noncomputable def sum_series : ℝ := ∑' (n : ℕ), (2 * n ^ 2 - n) / (n * (n + 1) * (n + 2))

theorem series_sum_eq_five : sum_series = 5 :=
sorry

end series_sum_eq_five_l279_279687


namespace radius_of_inscribed_circle_of_PF1Q_l279_279367

-- Conditions and definitions from part a)
def condition_ellipse (x y : ℝ) : Prop :=
  (x^2 / 16 + y^2 / 7 = 1)

def symmetric_wrt_origin (P Q : ℝ × ℝ) : Prop :=
  (P.1 = -Q.1) ∧ (P.2 = -Q.2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Given the above conditions and definitions, we need to prove the following theorem
theorem radius_of_inscribed_circle_of_PF1Q {F1 F2 P Q : ℝ × ℝ} 
  (h1 : condition_ellipse P.1 P.2) 
  (h2 : condition_ellipse Q.1 Q.2) 
  (h3 : symmetric_wrt_origin P Q) 
  (h4 : distance P Q = distance F1 F2) :
  ∃ r : ℝ, r = 1 :=
by
  -- Skipping the proof, hence using sorry
  sorry

end radius_of_inscribed_circle_of_PF1Q_l279_279367


namespace trigonometric_identity_l279_279724

-- Definitions for the conditions provided
def condition_x (x : ℝ) := (5 * Real.pi / 2 < x) ∧ (x < 3 * Real.pi)

-- Mathematical statement to prove the question equals the correct answer given the conditions
theorem trigonometric_identity (x : ℝ) (h : condition_x x) :
  sqrt ((1 - Real.sin (3 * Real.pi / 2 - x)) / 2) = - Real.cos (x / 2) :=
sorry

end trigonometric_identity_l279_279724


namespace a1_eq_3_general_form_an_Tn_value_l279_279129

noncomputable def S (n: ℕ) (a: ℕ → ℕ) : ℕ := (1 / 4) * (a n) ^ 2 + (1 / 2) * (a n) - (3 / 4)

-- Part 1: Prove a₁ = 3
theorem a1_eq_3 (a: ℕ → ℕ) (S : ℕ → ℕ):
  S 1 = (1 / 4) * (a 1) ^ 2 + (1 / 2) * (a 1) - (3 / 4) → 
  (a 1 = 3) := sorry

-- Part 2: Prove general form aₙ = 2n + 1
theorem general_form_an (a: ℕ → ℕ) (S : ℕ → ℕ):
  (∀ n, S n = (1 / 4) * (a n) ^ 2 + (1 / 2) * (a n) - (3 / 4)) →
  (∀ n, a n = 2 * n + 1) := sorry

-- Part 3: Prove Tₙ = (2n-1) * 2^(n+1) + 2
theorem Tn_value (a: ℕ → ℕ) (b: ℕ → ℕ) (T: ℕ → ℕ):
  (∀ n, b n = 2^n) →
  (∀ n, T n = ∑ i in range (n + 1), a i * b i) →
  (∀ n, a n = 2 * n + 1) →
  (∀ n, T n = (2 * n - 1) * 2^(n + 1) + 2) := sorry

end a1_eq_3_general_form_an_Tn_value_l279_279129


namespace simplify_fraction_l279_279906

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l279_279906


namespace ricky_pennies_l279_279887

open Nat

theorem ricky_pennies (n : ℕ) (h1 : ∀ k : ℕ, k ≥ 4 → (fibonacci k = 48))
    (h2 : fibonacci 0 = 3) :
    fibonacci 0 = n :=
  sorry

end ricky_pennies_l279_279887


namespace sum_sequence_property_l279_279106

open BigOperators
open Set

noncomputable def a_seq (n : ℕ) (h_pos : n > 0) : ℝ := real.sqrt n - real.sqrt (n - 1)

theorem sum_sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_sum : ∀ n, S n = (n * (a n + 1/a n) / 2)) :
  ∀ n, a n = real.sqrt n - real.sqrt (n - 1) := 
begin
  sorry
end

end sum_sequence_property_l279_279106


namespace tunnel_length_l279_279956

theorem tunnel_length (train_length : ℝ) (time_front_to_tail_exit : ℝ) (train_speed : ℝ)
  (h_train_length : train_length = 1.5) 
  (h_time : time_front_to_tail_exit = 4 / 60)  -- 4 minutes is 4/60 hours
  (h_speed : train_speed = 45) :
  ∃ (tunnel_length : ℝ), tunnel_length = 1.5 :=
by
  use 1.5
  sorry

end tunnel_length_l279_279956


namespace tips_fraction_l279_279660

theorem tips_fraction {S T I : ℚ} (h1 : T = (7/4) * S) (h2 : I = S + T) : (T / I) = 7 / 11 :=
by
  sorry

end tips_fraction_l279_279660


namespace angle_ABC_tangent_circle_l279_279574

theorem angle_ABC_tangent_circle 
  (BAC ACB : ℝ)
  (h1 : BAC = 70)
  (h2 : ACB = 45)
  (D : Type)
  (incenter : ∀ D : Type, Prop)  -- Represent the condition that D is the incenter
  : ∃ ABC : ℝ, ABC = 65 :=
by
  sorry

end angle_ABC_tangent_circle_l279_279574


namespace circle_radius_squared_l279_279268

open Real

/-- Prove that the square of the radius of a circle is 200 given the conditions provided. -/

theorem circle_radius_squared {r : ℝ}
  (AB CD : ℝ)
  (BP : ℝ) 
  (APD : ℝ) 
  (hAB : AB = 12)
  (hCD : CD = 9)
  (hBP : BP = 10)
  (hAPD : APD = 45) :
  r^2 = 200 := 
sorry

end circle_radius_squared_l279_279268


namespace number_of_laceable_acute_angles_l279_279300

-- Define the laceable angle problem:
def laceable_angle_exists (θ : ℝ) (n : ℕ) : Prop :=
  ∃ (X : Fin (2 * n) → ℝ × ℝ),
    ∀ k, (X (⟨2 * k - 1, sorry⟩)).1 = (X (⟨2 * k, sorry⟩)).1 ∧
         (X (⟨2 * k, sorry⟩)).2 = (X (⟨2 * k - 1, sorry⟩)).2 ∧
         dist (⟨0, 0⟩, X ⟨1, sorry⟩) = dist (X ⟨1, sorry⟩, X ⟨2, sorry⟩) ∧
         dist (X ⟨2, sorry⟩, X ⟨1, sorry⟩) = dist (X ⟨3, sorry⟩, X ⟨2, sorry⟩) ∧
         dist (X ⟨2 * n - 1, sorry⟩, X ⟨2 * n, sorry⟩) = dist (X ⟨2 * n, sorry⟩, ⟨0, 0⟩)

-- The main theorem to prove:
theorem number_of_laceable_acute_angles : ∃ θ_values : Fin 5 → ℝ, 
  ∀ i, laceable_angle_exists (θ_values i) (nat_of_fin i + 1) := sorry

end number_of_laceable_acute_angles_l279_279300


namespace simplify_expression_l279_279533

theorem simplify_expression(x : ℝ) : 2 * x * (4 * x^2 - 3 * x + 1) - 7 * (2 * x^2 - 3 * x + 4) = 8 * x^3 - 20 * x^2 + 23 * x - 28 :=
by
  sorry

end simplify_expression_l279_279533


namespace rectangle_length_l279_279620

theorem rectangle_length {width length : ℝ} (h1 : (3 : ℝ) * 3 = 9) (h2 : width = 3) (h3 : width * length = 9) : 
  length = 3 :=
by
  sorry

end rectangle_length_l279_279620


namespace sin_double_angle_l279_279758

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end sin_double_angle_l279_279758


namespace existence_of_B_l279_279820

theorem existence_of_B (a b : ℝ) (A : ℝ) (H_a : a = 1) (H_b : b = Real.sqrt 3) (H_A : A = Real.pi / 6) :
  ∃ B : ℝ, (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) ∧ a / Real.sin A = b / Real.sin B :=
by
  sorry

end existence_of_B_l279_279820


namespace mans_speed_against_current_l279_279643

theorem mans_speed_against_current (V_with_current V_current V_against : ℝ) (h1 : V_with_current = 21) (h2 : V_current = 4.3) : 
  V_against = V_with_current - 2 * V_current := 
sorry

end mans_speed_against_current_l279_279643


namespace parallel_to_plane_l279_279909

-- Define Tetrahedron vertices and points on altitudes
structure Point := (x y z : ℝ)
structure Tetrahedron := (A B C D : Point)

def isAltitude (A : Point) (X Y : Point) : Prop :=
  ∃ (M : Point), M ∈ (lineSegment X Y) ∧ distance A M = distance (proj_on_line A X Y) M

def circumcenter (A B C : Point) : Point := sorry -- Define the circumcenter function

def feet_of_altitudes (T : Tetrahedron) : (Point × Point × Point) :=
  let (A, B, C, D) := (T.A, T.B, T.C, T.D) in
  let M := some (isAltitude A B D) in
  let N := some (isAltitude B A D) in
  let P := some (isAltitude A C D) in
  (M, N, P)

def perpendicular_to_plane (P Q R : Point) (D : Point) : Prop :=
  ∃ (N : Point), N ∈ plane P Q R ∧ ∀ M ∈ plane_line P Q, ∠ N M D = 90

theorem parallel_to_plane (T : Tetrahedron) :
  let (M, N, P) := feet_of_altitudes T in
  ∀ (Q : Point), Q = circumcenter T.A T.B T.D →
  let plane_perp_to_DO := plane T.D Q (Q + vector_of_direction (T.D, Q)) in
  parallel (line_through_points M N) plane_perp_to_DO ∧
  parallel (line_through_points P (some (isAltitude C T.A T.D))) plane_perp_to_DO :=
begin
  intros,
  sorry -- Proof goes here
end

end parallel_to_plane_l279_279909


namespace time_in_vancouver_l279_279214

theorem time_in_vancouver (toronto_time vancouver_time : ℕ) (h : toronto_time = 18 + 30 / 60) (h_diff : vancouver_time = toronto_time - 3) :
  vancouver_time = 15 + 30 / 60 :=
by
  sorry

end time_in_vancouver_l279_279214


namespace correct_conclusions_l279_279930

-- Definitions for conditions
def conclusion1 (a : ℝ) : Prop := (1 : ℝ) / ((a^2) + 1) ≠ 0
def conclusion2 (a : ℝ) : Prop := a = -1 → (a + 1) / ((a^2) - 1) = 0
def conclusion3 (x : ℝ) : Prop := (x^2 + 1) / (x - 1) < 0 → x < 1
def conclusion4 (x : ℝ) : Prop := (x + 1) / (x + 2) / ((x + 1) / x) ≠ 0 → x ≠ -2 ∧ x ≠ 0

-- Main theorem
theorem correct_conclusions : (∀ a, conclusion1 a) ∧ (∀ a, ¬ conclusion2 a) ∧ (∀ x, conclusion3 x) ∧ (∀ x, ¬ conclusion4 x) → 
  2 = 2 :=
by
  intros h,
  sorry

end correct_conclusions_l279_279930


namespace correct_scenarios_l279_279545

universe u

open Nat

def sampling_scenarios : List (List ℕ) :=
  [[5, 10, 17, 36, 47, 53, 65, 76, 90, 95],
   [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
   [8, 17, 42, 48, 52, 56, 61, 64, 74, 88],
   [8, 15, 22, 29, 48, 55, 62, 78, 85, 92]]

def total_students : ℕ := 100
def first_grade_students : ℕ := 40
def second_grade_students : ℕ := 30
def third_grade_students : ℕ := 30

def is_stratified_sampling (lst : List ℕ) : Prop :=
  let in_first_grade := lst.countP (· ≤ 40) 
  let in_second_grade := lst.countP (λ n, 41 ≤ n ∧ n ≤ 70)
  let in_third_grade := lst.countP (λ n, 71 ≤ n)
  in_first_grade = 4 ∧ in_second_grade = 3 ∧ in_third_grade = 3

theorem correct_scenarios : is_stratified_sampling (sampling_scenarios.get! 0) ∧
                            is_stratified_sampling (sampling_scenarios.get! 3) :=
by
  sorry

end correct_scenarios_l279_279545


namespace odd_function_analytic_expression_value_of_f_f_neg_2_l279_279021

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x - 1 else if x < 0 then 2 * x + 1 else 0

theorem odd_function (x : ℝ) : f (-x) = -f x :=
by
  unfold f
  split_ifs with h1 h2 h3 h4
  · rw [neg_gt_zero] at h1
    rw [neg_sub]
    rw [neg_mul, neg_one_mul, add_neg_cancel_right]
  · rw [neg_lt_zero] at h2
    rw [neg_add]
    rw [neg_mul, neg_one_mul, add_neg_cancel_right]
  · rw [neg_zero] at h3
    simp
  · by_contradiction h5
    push_neg at h5
    linarith

theorem analytic_expression (x : ℝ) : 
  f x = if x > 0 then 2 * x - 1 else if x < 0 then 2 * x + 1 else 0 := 
by
  unfold f
  split_ifs with h1 h2 h3 h4
  · rw [if_pos h1]
  · rw [if_neg h1, if_pos h2]
  · rw [if_neg h1, if_neg h2, if_pos h3]
  · rw [if_neg h1, if_neg h2, if_neg h3, if_neg h4]

theorem value_of_f_f_neg_2 : f (f (-2)) = -5 :=
by
  have h1 : f (-2) = -3 := 
  by
    unfold f
    split_ifs
    · rw [neg_lt_zero]
      fragment 
      
    have h2 : f (-3) = -5 := 
  by
    unfold f
    split_ifs
    · rw [neg_lt_zero]
      fragment
    exact Eq.trans h1 h2

end odd_function_analytic_expression_value_of_f_f_neg_2_l279_279021


namespace lcm_of_12_and_16_is_48_l279_279965

-- Define numbers and conditions
def n : ℕ := 12
def m : ℕ := 16
def gcf (a b: ℕ) : ℕ := Nat.gcd a b
def lcm (a b: ℕ) : ℕ := Nat.lcm a b

-- Theorem statement
theorem lcm_of_12_and_16_is_48 : lcm n m = 48 := 
by
  have h1 : n = 12 := rfl
  have h2 : m = 16 := rfl
  have gcf_12_16 : gcf n m = 4 := by sorry
  show lcm n m = 48 sorry

end lcm_of_12_and_16_is_48_l279_279965


namespace intersection_sum_l279_279555

noncomputable def f (x : ℝ) : ℝ :=
if x > 1 then 2*x^2 - 12*x + 16
else 
let y := -(2*(x-1)^2 - 12*(x-1) + 16) in if x < 1 then y else 0

theorem intersection_sum :
  let intersection_points := (set_of (λ x => f x = 2)) in
  is_sum (intersection_points) 5 :=
by
  sorry

end intersection_sum_l279_279555


namespace sin_sum_of_equilateral_triangle_angles_l279_279920

theorem sin_sum_of_equilateral_triangle_angles (α β γ : ℝ) (h_sum : α + β + γ = π) :
  (sin α = sin β + sin γ) ∨ (sin β = sin α + sin γ) ∨ (sin γ = sin α + sin β) :=
by
  sorry -- Proof is omitted

end sin_sum_of_equilateral_triangle_angles_l279_279920


namespace sum_of_powers_of_z_equals_given_value_l279_279000

noncomputable theory

-- Define the given complex number z.
def z : ℂ := -1/2 + (real.sqrt 3)/2 * complex.I

-- Define the sequence sum we need to prove.
def sum_of_powers_of_z : ℂ := ∑ i in finset.range 2023, z^(i + 1)

-- State the theorem that we need to prove.
theorem sum_of_powers_of_z_equals_given_value :
  sum_of_powers_of_z = -1/2 + (real.sqrt 3)/2 * complex.I :=
by sorry

end sum_of_powers_of_z_equals_given_value_l279_279000


namespace tim_reading_hours_per_week_l279_279597

theorem tim_reading_hours_per_week :
  (meditation_hours_per_day = 1) →
  (reading_hours_per_day = 2 * meditation_hours_per_day) →
  (reading_hours_per_week = reading_hours_per_day * 7) →
  reading_hours_per_week = 14 :=
by
  intros h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  exact h3

end tim_reading_hours_per_week_l279_279597


namespace derivative_f_at_alpha_l279_279071

-- Define the function f
def f (x α : ℝ) : ℝ :=
  sin α - cos x

-- State the theorem
theorem derivative_f_at_alpha (α : ℝ) : 
  ∂ (λ x, f x α) / ∂ x (α) = sin α :=
by
  sorry

end derivative_f_at_alpha_l279_279071


namespace time_reading_per_week_l279_279592

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l279_279592


namespace five_digit_palindrome_count_l279_279415

theorem five_digit_palindrome_count : 
  (∃ a b c, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) → 
  ∃ n, n = 9 * 10 * 10 :=
by 
  have h : 900 = 9 * 10 * 10 := by norm_num
  use 900
  exact h

end five_digit_palindrome_count_l279_279415


namespace find_x_intersection_l279_279023

-- Given conditions
variables (k b : ℝ)
def line_eq (k b x : ℝ) := k * x + b

-- Conditions from the problem
axiom parallel_to_line (hp : k = -3)
axiom passes_through_point (hpt : b = -2)

-- The goal is to find the intersection point with the x-axis, i.e., y = 0
theorem find_x_intersection :
  ∃ x : ℝ, line_eq k b x = 0 ∧ (x = -2 / 3) :=
by {
  sorry
}

end find_x_intersection_l279_279023


namespace product_of_distinct_divisors_l279_279130

theorem product_of_distinct_divisors (T : Set ℕ) (hT : T = { n | n ∣ 72000 ∧ 0 < n }) :
  (finset.card { n | ∃ a b ∈ T, a ≠ b ∧ n = a * b }) = 447 := 
sorry

end product_of_distinct_divisors_l279_279130


namespace regular_decagon_triangle_count_l279_279092

def is_regular_decagon (vertices : Finset ℕ) : Prop :=
  vertices.card = 10 ∧ (∀ v1 v2, v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → ((v2 - v1) % 10) ∈ {0, 1, 9})

def is_triangle_formed (A B C : ℕ) : Prop :=
  (A + B + C = 180) ∧ (A % 18 = 0) ∧ (B % 18 = 0) ∧ (C % 18 = 0)

noncomputable def distinct_triangles_count (vertices : Finset ℕ) (triangle_set : Finset (Finset ℕ)) : ℕ :=
  if is_regular_decagon vertices then triangle_set.filter (λ t, t.card = 3 ∧
    ∃ (angles : Finset ℕ), t ⊆ angles ∧ is_triangle_formed (angles.sum)).card
  else 0

theorem regular_decagon_triangle_count (vertices : Finset ℕ) (triangle_set : Finset (Finset ℕ)) :
  is_regular_decagon vertices → distinct_triangles_count vertices triangle_set = 8 := 
by
  sorry

end regular_decagon_triangle_count_l279_279092


namespace simplify_expression_l279_279532

theorem simplify_expression (y : ℝ) (h : y ≠ 0) : 
    y^(-2) - 2 * y + 1 = (1 - 2*y^3 + y^2) / y^2 :=
by sorry

end simplify_expression_l279_279532


namespace fractional_parts_sum_neq_one_l279_279159

theorem fractional_parts_sum_neq_one (x : ℚ) : (frac x + frac (x ^ 2) ≠ 1) := sorry

end fractional_parts_sum_neq_one_l279_279159


namespace octagon_side_length_from_square_l279_279461

theorem octagon_side_length_from_square (s : ℝ) (h_s_square : s = 1) :
  let x := 1 - 2 * (1 / (2 + Real.sqrt 2)) in
  x = 1 - Real.sqrt 2 / 2 :=
by
  sorry

end octagon_side_length_from_square_l279_279461


namespace orthocenter_on_circumcircle_l279_279450

-- Defining the conditions and the theorem to be proven
variables {A B C M N D E : Type}
  [field A] [field B] [field C] [field M] [field N] [field D] [field E]

-- Assume that points form an acute triangle ABC
variable (acute_triangle_ABC : ∀ (A B C : Type), Type)

-- Assume M is a point on the interior of the segment AC
variable (M_on_AC : ∀ (M : Type) (AC : Type), Type)

-- Assume N is a point on the extension of segment AC such that MN = AC
variable (N_on_AC_ext : ∀ (N : Type) (AC : Type) (MN_eq_AC : Type), Type)

-- Assume D is the foot of the perpendicular from M onto BC
variable (D_foot : ∀ (M : Type) (BC : Type), Type)

-- Assume E is the foot of the perpendicular from N onto AB
variable (E_foot : ∀ (N : Type) (AB : Type), Type)

-- The theorem statement
theorem orthocenter_on_circumcircle 
  (acute_triangle_ABC : ∀ A B C : Type, Prop)
  (M_on_AC : ∀ M : Type, Prop)
  (N_on_AC_ext : ∀ N : Type, Prop)
  (D_foot : ∀ D : Type, Prop)
  (E_foot : ∀ E : Type, Prop) :
  ∀ (H : Type), orthocenter H (triangle ABC) ∈ circumcircle (triangle BED) :=
begin
  sorry
end

end orthocenter_on_circumcircle_l279_279450


namespace sqrt_expression_eq_neg_one_l279_279315

theorem sqrt_expression_eq_neg_one : 
  Real.sqrt ((-2)^2) + (Real.sqrt 3)^2 - (Real.sqrt 12 * Real.sqrt 3) = -1 :=
sorry

end sqrt_expression_eq_neg_one_l279_279315


namespace simplify_fraction_l279_279899

def gcd (a b : ℕ) : ℕ := nat.gcd a b

theorem simplify_fraction : (180 = 2^2 * 3^2 * 5) ∧ (270 = 2 * 3^3 * 5) ∧ (gcd 180 270 = 90) →
  180 / 270 = 2 / 3 :=
by
  intro h
  cases h with h1 h2h3
  cases h2h3 with h2 h3
  sorry -- Proof is omitted

end simplify_fraction_l279_279899


namespace minimum_height_l279_279147

theorem minimum_height (x : ℝ) (h : ℝ) (A : ℝ) :
  (h = x + 4) →
  (A = 6*x^2 + 16*x) →
  (A ≥ 120) →
  (x ≥ 2) →
  h = 6 :=
by
  intros h_def A_def A_geq min_x
  sorry

end minimum_height_l279_279147


namespace number_of_solutions_l279_279859

def greatestInt (x : ℝ) : ℤ := int.floor x

def equation (x : ℝ) : ℝ := 2^x - 2 * (greatestInt x) - 1

theorem number_of_solutions : ∃! n, n = 3 ∧ ∀ x, equation x = 0 → 1 ≤ x ∧ x < x + 1 :=
by
  sorry

end number_of_solutions_l279_279859


namespace calculation_result_l279_279573

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by
  sorry

end calculation_result_l279_279573


namespace least_possible_beta_l279_279542

def is_odd_prime (n : ℕ) : Prop := nat.prime n ∧ odd n

theorem least_possible_beta :
  ∃ (α β : ℕ), is_odd_prime α ∧ is_odd_prime β ∧ α > β ∧ α + β = 100 ∧ β = 3 :=
by
  sorry

end least_possible_beta_l279_279542


namespace andy_late_duration_l279_279668

theorem andy_late_duration :
  let start_time := 7 * 60 + 15 in -- 7:15 AM converted to minutes
  let school_start := 8 * 60 in -- 8:00 AM converted to minutes
  let normal_travel_time := 30 in
  let red_light_stops := 3 * 4 in
  let construction_delay := 10 in
  let total_travel_time := normal_travel_time + red_light_stops + construction_delay in
  total_travel_time - (school_start - start_time) = 7 :=
by
  sorry

end andy_late_duration_l279_279668


namespace sum_of_angles_of_roots_l279_279954

theorem sum_of_angles_of_roots (z1 z2 z3 z4 : Complex) (r1 r2 r3 r4 : ℝ) (θ1 θ2 θ3 θ4 : ℝ) 
  (hz1 : z1 = Complex.mkPolar r1 θ1)
  (hz2 : z2 = Complex.mkPolar r2 θ2)
  (hz3 : z3 = Complex.mkPolar r3 θ3)
  (hz4 : z4 = Complex.mkPolar r4 θ4)
  (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) (hr4 : r4 > 0)
  (hθ1 : 0 ≤ θ1 ∧ θ1 < 360)
  (hθ2 : 0 ≤ θ2 ∧ θ2 < 360)
  (hθ3 : 0 ≤ θ3 ∧ θ3 < 360)
  (hθ4 : 0 ≤ θ4 ∧ θ4 < 360)
  (hz_eq : z1 ^ 4 = -16 * Complex.i ∧ z2 ^ 4 = -16 * Complex.i ∧ z3 ^ 4 = -16 * Complex.i ∧ z4 ^ 4 = -16 * Complex.i) :
  θ1 + θ2 + θ3 + θ4 = 810 :=
sorry

end sum_of_angles_of_roots_l279_279954


namespace joe_time_to_store_l279_279852

theorem joe_time_to_store :
  ∀ (r_w : ℝ) (r_r : ℝ) (t_w t_r t_total : ℝ), 
   (r_r = 2 * r_w) → (t_w = 10) → (t_r = t_w / 2) → (t_total = t_w + t_r) → (t_total = 15) := 
by
  intros r_w r_r t_w t_r t_total hrw hrw_eq hr_tw hr_t_total
  sorry

end joe_time_to_store_l279_279852


namespace percentage_of_ginger_is_correct_l279_279149

noncomputable def teaspoons_per_tablespoon : ℕ := 3
noncomputable def ginger_tablespoons : ℕ := 3
noncomputable def cardamom_teaspoons : ℕ := 1
noncomputable def mustard_teaspoons : ℕ := 1
noncomputable def garlic_tablespoons : ℕ := 2
noncomputable def chile_powder_factor : ℕ := 4

theorem percentage_of_ginger_is_correct :
  let ginger_teaspoons := ginger_tablespoons * teaspoons_per_tablespoon
  let garlic_teaspoons := garlic_tablespoons * teaspoons_per_tablespoon
  let chile_teaspoons := chile_powder_factor * mustard_teaspoons
  let total_teaspoons := ginger_teaspoons + cardamom_teaspoons + mustard_teaspoons + garlic_teaspoons + chile_teaspoons
  let percentage_ginger := (ginger_teaspoons * 100) / total_teaspoons
  percentage_ginger = 43 :=
by
  sorry

end percentage_of_ginger_is_correct_l279_279149


namespace average_speed_l279_279289

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem average_speed (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) = (2 * b * a) / (a + b) :=
by
  sorry

end average_speed_l279_279289


namespace correct_option_a_l279_279241

theorem correct_option_a :
  (sqrt 12 = 2 * sqrt 3) ∧ (¬ (3 * sqrt 3 - sqrt 3 = 3)) ∧ (¬ (2 + sqrt 3 = 2 * sqrt 3)) ∧ (¬ (sqrt ((-2)^2) = -2)) :=
by {
  sorry
}

end correct_option_a_l279_279241


namespace simplify_fraction_l279_279901

def gcd (a b : ℕ) : ℕ := nat.gcd a b

theorem simplify_fraction : (180 = 2^2 * 3^2 * 5) ∧ (270 = 2 * 3^3 * 5) ∧ (gcd 180 270 = 90) →
  180 / 270 = 2 / 3 :=
by
  intro h
  cases h with h1 h2h3
  cases h2h3 with h2 h3
  sorry -- Proof is omitted

end simplify_fraction_l279_279901


namespace matrix_diagonal_solution_l279_279579

theorem matrix_diagonal_solution :
  ∃ x : ℕ, 2 * x + 1 * 6 = 16 :=
by {
  use 5,
  sorry
}

end matrix_diagonal_solution_l279_279579


namespace right_triangle_area_l279_279204

theorem right_triangle_area (h : Real) (a : Real) (b : Real) (c : Real) (h_is_hypotenuse : h = 13) (a_is_leg : a = 5) (pythagorean_theorem : a^2 + b^2 = h^2) : (1 / 2) * a * b = 30 := 
by 
  sorry

end right_triangle_area_l279_279204


namespace expand_polynomial_l279_279236

theorem expand_polynomial (N : ℕ) :
  (∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a + b + c + d + 1)^N = 715) ↔ N = 13 := by
  sorry -- Replace with the actual proof when ready

end expand_polynomial_l279_279236


namespace square_roots_proof_l279_279677

theorem square_roots_proof : (sqrt 2 + 1) ^ 2 - sqrt (9 / 2) = 3 + sqrt 2 / 2 := 
by sorry

end square_roots_proof_l279_279677


namespace largest_divisor_of_product_of_four_consecutive_even_numbers_l279_279338

theorem largest_divisor_of_product_of_four_consecutive_even_numbers :
  ∀ (k : ℕ), ∃ (n : ℕ), (∀ m ∈ finset.range 4, (2 * k + 2 * m).gcd n = 1) → n = 96 :=
by
  sorry

end largest_divisor_of_product_of_four_consecutive_even_numbers_l279_279338


namespace distance_from_P_to_focus_l279_279736

-- Definition of a parabola y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Definition of distance from P to y-axis
def distance_to_y_axis (x : ℝ) : ℝ := abs x

-- Definition of the focus of the parabola y^2 = 8x
def focus : (ℝ × ℝ) := (2, 0)

-- Definition of Euclidean distance
def euclidean_distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  (P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 

theorem distance_from_P_to_focus (x y : ℝ) (h₁ : parabola x y) (h₂ : distance_to_y_axis x = 4) :
  abs (euclidean_distance (x, y) focus) = 6 :=
sorry

end distance_from_P_to_focus_l279_279736


namespace relationship_y1_y2_y3_l279_279516

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end relationship_y1_y2_y3_l279_279516


namespace standing_arrangements_24_l279_279827

-- Definitions
def num_male_students : Nat := 3
def num_female_students : Nat := 3
def adjacent_diff_gender (students : List Nat) : Prop := ∀ i, i < students.length - 1 → students.get i % 2 ≠ students.get (i + 1) % 2
def a_adj_b (students : List Nat) (a b : Nat) : Prop := ∃ i, i < students.length - 1 ∧ students.get i = a ∧ students.get (i + 1) = b
def not_at_ends (students : List Nat) (a b : Nat) : Prop :=
  ∀ i, (i = 0 ∨ i = students.length - 1) → students.get i ≠ a ∧ students.get i ≠ b

-- Theorem statement
theorem standing_arrangements_24 :
  ∃ (students : List Nat), 
    students.length = num_male_students + num_female_students ∧ 
    adjacent_diff_gender students ∧ 
    a_adj_b students 0 1 ∧
    not_at_ends students 0 1 ∧
    true -- placeholder for conditions guaranteeing 24 arrangements
    :=
sorry

end standing_arrangements_24_l279_279827


namespace dot_product_condition_l279_279020

-- Definitions of the conditions
variable {a b : ℝ} -- Assuming vectors in ℝ for simplicity
variable θ : ℝ -- Angle between vectors

-- Theorem statement
theorem dot_product_condition (h1 : θ = real.pi) (h2 : a * b < 0) :
  (∃ θ, θ > real.pi / 2 ∧ θ < real.pi ∧ a * b < 0) → 
  (a * b < 0) ∧ ¬(θ > real.pi / 2 ∧ θ < real.pi) :=
by
  sorry
  
end dot_product_condition_l279_279020


namespace smallest_integer_CC6_DD8_l279_279985

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l279_279985


namespace distance_from_P_to_line_AB_l279_279798

structure Point3D :=
  (x y z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def distance_point_to_line (P A B : Point3D) : ℝ :=
  let AB := vector_sub B A
  let AP := vector_sub P A
  let cos_theta := dot_product AB AP / (magnitude AB * magnitude AP)
  magnitude AP * Real.sqrt (1 - cos_theta^2)

noncomputable def correct_distance : ℝ := Real.sqrt 6 / 3

theorem distance_from_P_to_line_AB : ∀ (P A B : Point3D), 
  P = ⟨1, 1, 1⟩ → A = ⟨1, 0, 1⟩ → B = ⟨0, 1, 0⟩ → 
  distance_point_to_line P A B = correct_distance :=
by
  intros P A B hP hA hB
  rw [hP, hA, hB]
  sorry

end distance_from_P_to_line_AB_l279_279798


namespace volume_of_cone_is_correct_l279_279651

noncomputable def volume_of_cone (V_pyramid : ℝ) (alpha : ℝ) : ℝ :=
  let alpha_rad := alpha * (Real.pi / 180)
  let tangent_half_alpha := Real.tan (alpha_rad / 2)
  let cot_alpha := Real.cot alpha_rad
  V_pyramid * Real.pi * tangent_half_alpha^2 * cot_alpha

theorem volume_of_cone_is_correct :
  volume_of_cone 671.6 58 = 405.1 :=
by
  sorry

end volume_of_cone_is_correct_l279_279651


namespace ratio_BZ_ZC_eq_one_l279_279845

noncomputable def triangle := ℝ × ℝ × ℝ
noncomputable def point := ℝ × ℝ
def AC_gt_AB (A B C : point) : Prop := dist C A > dist B A
def is_bisect_perpendicular_intersection (A B C P : point) : Prop :=
  ∃ Q1 Q2 : point, (Q1 = midpoint B C) ∧ (Q2 = line_through P A ∩ line_bisector B C)
noncomputable def PX_perp_AB (A B C P X : point) : Prop :=
  line_perpendicular_through P A B
noncomputable def PY_perp_AC (A B C P Y : point) : Prop :=
  line_perpendicular_through P A C
noncomputable def intersection_XY_BC (X Y B C Z : point) : Prop :=
  collinear X Y Z ∧ collinear B C Z

theorem ratio_BZ_ZC_eq_one
  (A B C P X Y Z : point)
  (h1 : AC_gt_AB A B C)
  (h2 : is_bisect_perpendicular_intersection A B C P)
  (h3 : PX_perp_AB A B C P X)
  (h4 : PY_perp_AC A B C P Y)
  (h5 : intersection_XY_BC X Y B C Z) :
  dist B Z = dist Z C :=
sorry

end ratio_BZ_ZC_eq_one_l279_279845


namespace f_is_odd_period_2pi_l279_279562

noncomputable def f (x : ℝ) : ℝ := cos (x + π / 4) - cos (x - π / 4)

theorem f_is_odd_period_2pi : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2 * π) = f x) :=
by
  sorry

end f_is_odd_period_2pi_l279_279562


namespace ratio_of_radii_is_rational_product_of_half_angle_sins_is_rational_l279_279376

theorem ratio_of_radii_is_rational 
    (a b c : ℚ) 
    (p : ℚ) (h_p : p = (a + b + c) / 2) 
    (R r : ℚ)
    (S : ℚ) (h_S1 : S = p * r)
    (h_S2 : S = a * b * c / (4 * R)) : 
    (R / r : ℚ) :=
sorry

theorem product_of_half_angle_sins_is_rational 
    (a b c : ℚ) 
    (α β γ : ℚ)
    (h_αβγ : α + β + γ = π) : 
    (sin(α / 2) * sin(β / 2) * sin(γ / 2) : ℚ) :=
sorry

end ratio_of_radii_is_rational_product_of_half_angle_sins_is_rational_l279_279376


namespace sqrt_arithmetic_sequence_l279_279158

theorem sqrt_arithmetic_sequence : ¬ (∃ a d : ℝ, a + d = sqrt 3 ∧ sqrt 2 = a ∧ sqrt 5 = a + 2 * d) := by
  sorry

end sqrt_arithmetic_sequence_l279_279158


namespace tony_bought_10_play_doughs_l279_279123

noncomputable def num_play_doughs 
    (lego_cost : ℕ) 
    (sword_cost : ℕ) 
    (play_dough_cost : ℕ) 
    (bought_legos : ℕ) 
    (bought_swords : ℕ) 
    (total_paid : ℕ) : ℕ :=
  let lego_total := lego_cost * bought_legos
  let sword_total := sword_cost * bought_swords
  let total_play_dough_cost := total_paid - (lego_total + sword_total)
  total_play_dough_cost / play_dough_cost

theorem tony_bought_10_play_doughs : 
  num_play_doughs 250 120 35 3 7 1940 = 10 := 
sorry

end tony_bought_10_play_doughs_l279_279123


namespace dihedral_angle_of_regular_quadrangular_pyramid_l279_279832

theorem dihedral_angle_of_regular_quadrangular_pyramid (
    (a l : ℝ) 
    (h : \(\sqrt{l^2 - \frac{a^2}{2}} = h\)) 
    (α : ℝ) 
    (hα1 : 2*θ = α)
    (hα2 : 1) \)
:
  (φ : ℝ) 
  = arccos((2 - sqrt(5))) :=
by 
  sorry

end dihedral_angle_of_regular_quadrangular_pyramid_l279_279832


namespace omega_range_l279_279401

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (ω * x / 2) ^ 2 + (1 / 2) * sin (ω * x) - (1 / 2)

theorem omega_range (ω : ℝ) (x : ℝ) (hω : ω > 0):
  (∃ x, π < x ∧ x < 2 * π ∧ f ω x = 0) →
  (ω ∈ set.Ioo (1/8) (1/4) ∪ set.Ioi (5/8)) :=
by
  sorry

end omega_range_l279_279401


namespace number_of_true_propositions_is_two_l279_279666

/--
Among the following propositions:
1. In a triangle ABC, if cos A < cos B, then A > B.
2. If the derivative of the function f(x) is f'(x), a necessary and sufficient condition for f(x_0) to be an extremum of f(x) is f'(x_0) = 0.
3. The smallest positive period of the function y = |tan(2x + π/3)| is π/2.
4. In the same Cartesian coordinate system, the graph of the function f(x) = sin x and the graph of the function f(x) = x have only three common points.
-/
def number_of_true_propositions : ℕ :=
if (1_is_true ∘ prop1) && (2_is_true ∘ prop2) && (3_is_true ∘ prop3) && (4_is_true ∘ prop4) then 2 else 0

theorem number_of_true_propositions_is_two :
  number_of_true_propositions = 2 := 
sorry

end number_of_true_propositions_is_two_l279_279666


namespace number_of_proper_subsets_l279_279339

noncomputable def proper_subsets_count (M : Set ℤ) := 2 ^ (Set.card M) - 1

theorem number_of_proper_subsets : 
  proper_subsets_count {x : ℤ | 0 < abs (x - 1) ∧ abs (x - 1) < 3} = 15 := 
by
  sorry

end number_of_proper_subsets_l279_279339


namespace polynomial_satisfies_functional_equation_l279_279497

theorem polynomial_satisfies_functional_equation
  (a : ℝ) (ha : a ≠ 0) (g : ℝ → ℝ)
  (H : ∀ x, g (a * x + 1) = a * g x + 1) :
  (a = 1 → ∃ c : ℝ, ∀ x, g x = x + c) ∧
  (a ≠ 1 → ∃ b : ℝ, ∀ x, g x = b * x + (1 - b) / (a - 1)) ∧
  (a = -1 → ∃ (h : ℝ → ℝ), (∀ x, g x = 0.5 + ∑ k in finset.range 1000, if k % 2 = 1 then h k * (x - 0.5)^k else 0)) :=
by {
  -- proof goes here
  sorry
}

end polynomial_satisfies_functional_equation_l279_279497


namespace domain_log_sqrt_l279_279556

noncomputable def domain (f : ℝ → ℝ) : Set ℝ := { x : ℝ | ∃ y : ℝ, f x = y }

theorem domain_log_sqrt : domain (λ x : ℝ, (Real.log (x + 1)) / (Real.sqrt x)) = { x : ℝ | x > 0 } := by
sorry

end domain_log_sqrt_l279_279556


namespace jose_bottle_caps_l279_279120

theorem jose_bottle_caps (start_caps : ℝ) (given_away : ℝ) (remaining_caps : ℝ) 
  (h1 : start_caps = 7.0) 
  (h2 : given_away = 2.0) : 
  remaining_caps = 5.0 :=
by 
  have h : remaining_caps = start_caps - given_away, 
    sorry
  exact sorry

end jose_bottle_caps_l279_279120


namespace angle_rotation_l279_279081

theorem angle_rotation (α : ℝ) (β : ℝ) (k : ℤ) :
  (∃ k' : ℤ, α + 30 = 120 + 360 * k') →
  (β = 360 * k + 90) ↔ (∃ k'' : ℤ, β = 360 * k'' + α) :=
by
  sorry

end angle_rotation_l279_279081


namespace remaining_shoes_to_sell_l279_279878

def shoes_goal : Nat := 80
def shoes_sold_last_week : Nat := 27
def shoes_sold_this_week : Nat := 12

theorem remaining_shoes_to_sell : shoes_goal - (shoes_sold_last_week + shoes_sold_this_week) = 41 :=
by
  sorry

end remaining_shoes_to_sell_l279_279878


namespace savannah_wraps_4_with_third_roll_l279_279166

variable (gifts total_rolls : ℕ)
variable (wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ)
variable (no_leftover : Prop)

def savannah_wrapping_presents (gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ) (no_leftover : Prop) : Prop :=
  gifts = 12 ∧
  total_rolls = 3 ∧
  wrap_with_roll1 = 3 ∧
  wrap_with_roll2 = 5 ∧
  remaining_wrap_with_roll3 = gifts - (wrap_with_roll1 + wrap_with_roll2) ∧
  no_leftover = (total_rolls = 3) ∧ (wrap_with_roll1 + wrap_with_roll2 + remaining_wrap_with_roll3 = gifts)

theorem savannah_wraps_4_with_third_roll
  (h : savannah_wrapping_presents gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 no_leftover) :
  remaining_wrap_with_roll3 = 4 :=
by
  sorry

end savannah_wraps_4_with_third_roll_l279_279166


namespace perspective_square_not_square_l279_279296

namespace ObliqueProjection

variables (L : Type) [LinearOrderedField L]

/-- Oblique projection properties -/
structure ObliqueProjection :=
  (unchanged_parallel : ∀ (a b : L), a || b → proj(a) || proj(b))
  (length_x_unchanged : ∀ (a : L), parallel_x_axis(a) → length(proj(a)) = length(a))
  (length_y_halved : ∀ (a : L), parallel_y_axis(a) → length(proj(a)) = length(a) / 2)

/-- Given the properties of oblique projection, prove that the perspective drawing of a square cannot remain a square. -/
theorem perspective_square_not_square {a b c d : L}
  (h1 : ObliqueProjection L)
  (h2 : a = b)
  (h3 : c = d)
  (h_ax : parallel_x_axis(a))
  (h_ay : parallel_y_axis(c)) :
  ¬(proj(a) = proj(b) ∧ proj(c) = proj(d)) :=
begin
  sorry
end

end ObliqueProjection

end perspective_square_not_square_l279_279296


namespace problem_integer_coord_intersections_l279_279518

def integer_coord_intersections (A B : (ℝ × ℝ)) (C : Fin 999 → (ℝ × ℝ)) : ℕ :=
  -- Function definition stating the integer-coordinate intersections of lines AC_i and BC_i
  sorry

theorem problem_integer_coord_intersections : 
  integer_coord_intersections (0,0) (1000,0) (λ i, (i.1 + 1, 1)) = 2326 := 
  sorry

end problem_integer_coord_intersections_l279_279518


namespace range_of_omega_l279_279769

theorem range_of_omega :
  ∀ (ω : ℝ), 
  (0 < ω) → 
  (∀ x, x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → cos (ω * x) - 1 = 0 → x ∈ {0, 2 * Real.pi, 4 * Real.pi}) →
  (2 ≤ ω ∧ ω < 3) :=
by
  intros ω hω_pos hzeros
  sorry

end range_of_omega_l279_279769


namespace contrapositive_example_l279_279926

theorem contrapositive_example (a b : ℝ) : (a > b → a - 8 > b - 8) → (a - 8 ≤ b - 8 → a ≤ b) :=
by 
  intro h
  intro h1
  apply not_lt.1
  intro h2
  apply not_le.2 h1
  exact h h2

end contrapositive_example_l279_279926


namespace simplify_fraction_l279_279903

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l279_279903


namespace inequality_proof_l279_279866

/-- Given a and b are positive and satisfy the inequality ab > 2007a + 2008b,
    prove that a + b > (sqrt 2007 + sqrt 2008)^2 -/
theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 :=
by
  sorry

end inequality_proof_l279_279866


namespace increasing_f_implies_m_le_2_condition_holds_implies_m_le_e_minus_1_l279_279143

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := x - 1/x - m * Real.log x

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := x - Real.log x - 1/Real.exp 1

-- Problem (1): f(x) is increasing on (0, ∞) implies m ≤ 2
theorem increasing_f_implies_m_le_2 (m : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (λ x, f x m) x ≥ 0) → m ≤ 2 := sorry

-- Problem (2): Given f(x) increasing and f(x₁) ≥ h(x₂) for x₁, x₂ ∈ [1, e], then m ≤ e - 1
theorem condition_holds_implies_m_le_e_minus_1 (m : ℝ) :
  (∀ x₁ x₂ : ℝ, (1 ≤ x₁ ∧ x₁ ≤ Real.exp 1) ∧ (1 ≤ x₂ ∧ x₂ ≤ Real.exp 1) → f x₁ m ≥ h x₂) →
  (∀ x : ℝ, 0 < x → deriv (λ x, f x m) x ≥ 0) → m ≤ Real.exp 1 - 1 := sorry

end increasing_f_implies_m_le_2_condition_holds_implies_m_le_e_minus_1_l279_279143


namespace largest_crate_dimension_l279_279273

def largest_dimension_of_crate : ℝ := 10

theorem largest_crate_dimension (length width : ℝ) (r : ℝ) (h : ℝ) 
  (h_length : length = 5) (h_width : width = 8) (h_radius : r = 5) (h_height : h >= 10) :
  h = largest_dimension_of_crate :=
by 
  sorry

end largest_crate_dimension_l279_279273


namespace geometric_progression_third_term_l279_279560

theorem geometric_progression_third_term (a₁ a₂ a₃ : ℝ) 
    (h₁ : a₁ = Real.sqrt 5)
    (h₂ : a₂ = Real.root 5 5) 
    (h₃ : a₃ = a₂ * (a₂ / a₁)) :
    a₃ = Real.root 10 5 :=
by
  sorry

end geometric_progression_third_term_l279_279560


namespace red_triangles_l279_279001

variables (k : ℕ)
variables (P : Finset (Fin (3 * k + 2)))
variables (R : Fin (3 * k + 2) → Finset (Fin (3 * k + 2)))

-- Conditions
def condition1 (P : Fin (3 * k + 2)) : Prop := (R P).card ≥ k + 2

def condition2 (P Q : Fin (3 * k + 2)) (hPQ : ¬ (P = Q) ∧ red P Q = ff): Prop := 
  (R P ∪ R Q).card ≥ 2 * k + 2

-- Theorem
theorem red_triangles (h1 : ∀ P, condition1 P) (h2 : ∀ P Q, condition2 P Q (by sorry)) : 
  ∃ T : Finset (Fin (3 * k + 2)), T.card = 3 ∧ (∀ (P Q R : Fin (3 * k + 2)), 
  P ∈ T ∧ Q ∈ T ∧ R ∈ T ∧ (red P Q = tt) ∧ (red P R = tt) ∧ (red Q R = tt)) :=
sorry

end red_triangles_l279_279001


namespace points_on_line_l279_279075

theorem points_on_line (b m n : ℝ) (hA : m = -(-5) + b) (hB : n = -(4) + b) :
  m > n :=
by
  sorry

end points_on_line_l279_279075


namespace find_f_neg_one_l279_279485

variable {R : Type} [LinearOrderedField R]

noncomputable def f (x : R) (m : R) : R :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : R) : (2^0 + 2 * 0 + m = 0) → f (-1) (-1) = -3 :=
by
  intro h1
  have h2 : m = -1 := by linarith
  rw [f, if_neg (show -1 >= 0, by linarith)]
  simp only [f, h2]
  norm_num
  sorry

end find_f_neg_one_l279_279485


namespace ratio_of_a_to_b_l279_279813

theorem ratio_of_a_to_b (a b : ℝ) (h1 : 0.5 / 100 * a = 85) (h2 : 0.75 / 100 * b = 150) : a / b = 17 / 20 :=
by {
  -- Proof will go here
  sorry
}

end ratio_of_a_to_b_l279_279813


namespace cubic_sum_of_roots_l279_279144

variable (r s : ℝ)

theorem cubic_sum_of_roots : (r^3 + s^3 = 35) ∧ (Polynomial.root_multiplicity (r, s) (Polynomial.X^2 - Polynomial.C 5 * Polynomial.X + Polynomial.C 6) = 2) := by
suffices h₁ : (r + s) = 5
suffices h₂ : (r * s) = 6
  have h₃ : r^3 + s^3 = (r + s) * ((r + s)^2 - 3 * (r * s)) by sorry
  rw [h₁, h₂] at h₃
  simp at h₃
  exact h₃
sorry

end cubic_sum_of_roots_l279_279144


namespace probability_two_boxes_three_same_color_l279_279828

def students : ℕ := 3
def block_colors : ℕ := 6
def boxes : ℕ := 6

axiom independent_placement : ∀ (s : ℕ), s = students → True  -- Placeholder for independence assumption

theorem probability_two_boxes_three_same_color :
  (students = 3) →
  (block_colors = 6) →
  (boxes = 6) →
  True → -- Placeholder for more detailed conditions about block placements
  (probability (exactly_two_boxes_receive_three_blocks_same_color students block_colors boxes) = 625 / 729) :=
by
  intros h_students h_colors h_boxes h_independent
  sorry

end probability_two_boxes_three_same_color_l279_279828


namespace fortieth_term_is_237_l279_279951

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def contains_digit_2 (n : ℕ) : Prop :=
  ('2').toNat ∈ n.digits 10

def special_sequence : List ℕ :=
  List.filter (λ n, is_multiple_of_3 n ∧ contains_digit_2 n) (List.range' 1 1000)  -- We use a range big enough to find the first 40 valid terms.

def nth_in_special_sequence (n : ℕ) : ℕ :=
  special_sequence.get! (n - 1)  -- Adjust index for Lean's 0-based indexing

theorem fortieth_term_is_237 : nth_in_special_sequence 40 = 237 := 
sorry

end fortieth_term_is_237_l279_279951


namespace family_groups_correct_l279_279952

structure Child where
  name : String
  eyeColor : String
  hairColor : String

def Liam   := Child.mk "Liam" "Green" "Black"
def Mia    := Child.mk "Mia" "Brown" "Red"
def Noah   := Child.mk "Noah" "Green" "Red"
def Eva    := Child.mk "Eva" "Brown" "Black"
def Oliver := Child.mk "Oliver" "Green" "Black"
def Lucy   := Child.mk "Lucy" "Brown" "Red"
def Jack   := Child.mk "Jack" "Green" "Red"

def family1 : Set Child := {Liam, Oliver}
def family2 : Set Child := {Mia, Lucy}
def family3 : Set Child := {Noah, Jack}

theorem family_groups_correct :
  (∀ f ∈ {family1, family2, family3}, ∃ c1 ∈ f, ∃ c2 ∈ f, 
      (c1.eyeColor = c2.eyeColor) ∨ (c1.hairColor = c2.hairColor)) ∧
  (∀ c ∈ {Liam, Mia, Noah, Eva, Oliver, Lucy, Jack},
      c ∈ family1 ∨ c ∈ family2 ∨ c ∈ family3) :=
by
  -- Proof skipped
  sorry

end family_groups_correct_l279_279952


namespace oranges_count_l279_279617

theorem oranges_count (N : ℕ) (k : ℕ) (m : ℕ) (j : ℕ) :
  (N ≡ 2 [MOD 10]) ∧ (N ≡ 0 [MOD 12]) → N = 72 :=
by
  sorry

end oranges_count_l279_279617


namespace general_formula_for_a_seq_l279_279037

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (x - 1) * (x - 2)
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := a * (2 * x - 3)
noncomputable def x_seq (a : ℝ) (x₀ : ℝ) (n : ℕ) : ℝ :=
  Nat.rec x₀ (fun n xₙ => xₙ - (f xₙ a) / (f' xₙ a)) n
noncomputable def a_seq (a : ℝ) (x₀ : ℝ) (n : ℕ) : ℝ :=
  Real.log ((x_seq a x₀ n - 2) / (x_seq a x₀ n - 1))

theorem general_formula_for_a_seq (a : ℝ) (h₀ : a > 0) (x₀ : ℝ) (h₀' : x₀ > 2) (h : a_seq a x₀ 0 = 1/2) :
  ∀ n : ℕ, a_seq a x₀ n = 2 ^ (n - 2) :=
begin
  sorry
end

end general_formula_for_a_seq_l279_279037


namespace find_n_l279_279481

noncomputable def a : ℝ := Real.pi / 2016

theorem find_n :
  ∃ n : ℕ, n > 0 ∧ 2 * (∑ k in Finset.range n + 1, Real.cos (↑(k^(2:ℕ)) * a) * Real.sin (↑k * a)) ∈ Int ∧ n = 72 :=
sorry

end find_n_l279_279481


namespace simplify_fraction_l279_279907

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l279_279907


namespace percentage_outside_circle_correct_l279_279452

noncomputable def percentage_outside_circle (s : ℝ) : ℝ :=
  let xy := s * (Real.sqrt 2) in
  let diameter := s in
  let length_outside := xy - diameter in
  (length_outside / xy) * 100

theorem percentage_outside_circle_correct :
  percentage_outside_circle 2 ≈ 29.3 :=
by
  sorry

end percentage_outside_circle_correct_l279_279452


namespace general_term_sequence_l279_279201

/--
Given the sequence a : ℕ → ℝ such that a 0 = 1/2,
a 1 = 1/4,
a 2 = -1/8,
a 3 = 1/16,
and we observe that
a n = (-(1/2))^n,
prove that this formula holds for all n : ℕ.
-/
theorem general_term_sequence (a : ℕ → ℝ) :
  (∀ n, a n = (-(1/2))^n) :=
sorry

end general_term_sequence_l279_279201


namespace promotional_sale_price_l279_279552

variable (a : ℝ) -- Defining the cost price as a real number

-- Lean statement to prove the selling price during the promotional sale
theorem promotional_sale_price (h : 0 ≤ a) : 
  let marked_price := 1.5 * a in
  let selling_price := 0.7 * marked_price in
  selling_price = 1.05 * a :=
by
  sorry

end promotional_sale_price_l279_279552


namespace probability_at_least_one_woman_selected_l279_279429

theorem probability_at_least_one_woman_selected:
  let men := 10
  let women := 5
  let totalPeople := men + women
  let totalSelections := Nat.choose totalPeople 4
  let menSelections := Nat.choose men 4
  let noWomenProbability := (menSelections : ℚ) / (totalSelections : ℚ)
  let atLeastOneWomanProbability := 1 - noWomenProbability
  atLeastOneWomanProbability = 11 / 13 :=
by
  sorry

end probability_at_least_one_woman_selected_l279_279429


namespace find_M_l279_279290

theorem find_M (M : ℤ) (h1 : 9.5 < M / 4) (h2 : M / 4 < 10) : M = 39 :=
sorry

end find_M_l279_279290


namespace perp_A1B1_A2B2_l279_279603

-- Definitions and assumptions based on the problem conditions
variables {k1 k2 : Type*} -- Circles
variables {p : Type*} -- Line
variables (O1 O2 A1 A2 B1 B2 : Type*)

-- Assuming conditions as given in Lean
def conditions :=
  k1 ≠ k2 ∧ -- Disjoint circles
  ¬(∃ x, x ∈ k1 ∧ x ∈ k2) ∧ -- Disjointness condition
  same_side p k1 k2 ∧ -- Both circles on the same side of p
  tangent k1 p A1 ∧ -- k1 touches line p at A1
  tangent k2 p A2 ∧ -- k2 touches line p at A2
  intersects (segment O1 O2) k1 B1 ∧ -- Segment O1O2 intersects k1 at B1
  intersects (segment O1 O2) k2 B2  -- Segment O1O2 intersects k2 at B2

-- Statement to be proved
theorem perp_A1B1_A2B2 (h : conditions O1 O2 A1 A2 B1 B2) : perp A1 B1 A2 B2 :=
sorry

end perp_A1B1_A2B2_l279_279603


namespace first_day_of_month_l279_279178

theorem first_day_of_month (h: ∀ n, (n % 7 = 2) → n_day_of_week n = "Wednesday"): 
  n_day_of_week 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279178


namespace lemon_ratio_l279_279503

variable (Levi Jayden Eli Ian : ℕ)

theorem lemon_ratio (h1: Levi = 5)
    (h2: Jayden = Levi + 6)
    (h3: Jayden = Eli / 3)
    (h4: Levi + Jayden + Eli + Ian = 115) :
    Eli = Ian / 2 :=
by
  sorry

end lemon_ratio_l279_279503


namespace angle_equality_QXA_QKP_l279_279319

-- Let W1 and W2 be two circles intersecting at points P and K.
variables {W1 W2 : Circle} (P K X Y C B A Q : Point)

-- X is a point on W1, Y is a point on W2. XY is a common tangent nearer to P.
variable (XY : Line)
axiom tangent_XY : is_tangent XY W1 ∧ is_tangent XY W2
axiom point_X : on_circle X W1
axiom point_Y : on_circle Y W2

-- XP intersects W2 again at C.
axiom intersection_XP_W2 : second_intersection (line_through X P) W2 C

-- YP intersects W1 again at B.
axiom intersection_YP_W1 : second_intersection (line_through Y P) W1 B

-- A is the intersection of lines BX and CY.
axiom intersection_BX_CY : intersection (line_through B X) (line_through C Y) A

-- Q is the second intersection point of circumcircles of triangle ABC and triangle AXY.
axiom second_intersection_circumcircle_ABC_AXY : on_circle Q (circumcircle (triangle A B C))
axiom second_intersection_circumcircle_ABC_AXY' : on_circle Q (circumcircle (triangle A X Y))

-- Proposition: ∠QXA = ∠QKP
theorem angle_equality_QXA_QKP : ∠ Q X A = ∠ Q K P :=
by sorry

end angle_equality_QXA_QKP_l279_279319


namespace average_age_increases_by_3_l279_279547

noncomputable def avg_age_increase 
  (n : ℕ) 
  (original_count : ℕ) 
  (new_count : ℕ) 
  (ages_men_replaced : list ℕ) 
  (average_age_women : ℚ) : ℚ :=
  (new_count * average_age_women - list.sum ages_men_replaced) / n

theorem average_age_increases_by_3 :
  avg_age_increase 7 2 2 [18, 22] 30.5 = 3 := 
by
  sorry

end average_age_increases_by_3_l279_279547


namespace num_terms_arithmetic_seq_l279_279066

noncomputable def nth_term (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem num_terms_arithmetic_seq :
  let a := -48
  let d := 7
  let l := 119
  ∃ n : ℕ, nth_term a d n = l ∧ n = 24 :=
begin
  sorry
end

end num_terms_arithmetic_seq_l279_279066


namespace molecular_weight_AlF3_total_weight_of_7_moles_AlF3_l279_279976

theorem molecular_weight_AlF3 (atomic_weight_Al : ℝ) (atomic_weight_F : ℝ) :
  let molecular_weight_AlF3 := atomic_weight_Al + 3 * atomic_weight_F
  in atomic_weight_Al = 26.98 ∧ atomic_weight_F = 19.00 →
     molecular_weight_AlF3 = 83.98 :=
by 
  intros h
  cases h
  let molecular_weight_AlF3 := h_left + 3 * h_right
  show molecular_weight_AlF3 = 83.98, from sorry

theorem total_weight_of_7_moles_AlF3 (atomic_weight_Al : ℝ) (atomic_weight_F : ℝ) :
  let molecular_weight_AlF3 := atomic_weight_Al + 3 * atomic_weight_F
      total_weight := molecular_weight_AlF3 * 7
  in atomic_weight_Al = 26.98 ∧ atomic_weight_F = 19.00 →
     total_weight = 587.86 :=
by 
  intros h
  cases h
  let molecular_weight_AlF3 := h_left + 3 * h_right
  let total_weight := molecular_weight_AlF3 * 7
  have molecular_weight_hyp := molecular_weight_AlF3 (h_left) (h_right)
  show total_weight = 587.86, from sorry

end molecular_weight_AlF3_total_weight_of_7_moles_AlF3_l279_279976


namespace invertible_product_l279_279691

def is_invertible (f : ℝ → ℝ) (domain : set ℝ) : Prop :=
  ∀ ⦃x₁ x₂⦄, x₁ ∈ domain → x₂ ∈ domain → f x₁ = f x₂ → x₁ = x₂

noncomputable def f6 : ℝ → ℝ := λ x, x^3 - 3 * x
noncomputable def f7 : ℤ → ℤ := 
  λ x, if x = -6 then 3 else if x = -5 then -5 else if x = -4 then 1 else if x = -3 then 0 else if x = -2 then -2 else if x = -1 then 4 else if x = 0 then -4 else 2

def domain7 : set ℤ := {-6, -5, -4, -3, -2, -1, 0, 1}

noncomputable def f8 : ℝ → ℝ := λ x, Real.tan x
def domain8 : set ℝ := {x : ℝ | -Real.pi / 2 < x ∧ x < Real.pi / 2}

noncomputable def f9 : ℝ → ℝ := λ x, 5 / x
def domain9 : set ℝ := {x : ℝ | x ≠ 0}

theorem invertible_product : 
  (is_invertible f7 domain7) ∧ 
  (is_invertible f8 domain8) ∧ 
  (is_invertible f9 domain9) → 
  7 * 8 * 9 = 504 :=
by
  sorry

end invertible_product_l279_279691


namespace shift_parabola_l279_279198

theorem shift_parabola (x : ℝ) : 
  (let y := x^2 in let y' := y - 3 in (x - 1)^2 + 3 = y' + 6) := 
by
  { sorry }

end shift_parabola_l279_279198


namespace letters_with_line_not_dot_l279_279825

-- Defining the conditions
def num_letters_with_dot_and_line : ℕ := 9
def num_letters_with_dot_only : ℕ := 7
def total_letters : ℕ := 40

-- Proving the number of letters with a straight line but not a dot
theorem letters_with_line_not_dot :
  (num_letters_with_dot_and_line + num_letters_with_dot_only + x = total_letters) → x = 24 :=
by
  intros h
  sorry

end letters_with_line_not_dot_l279_279825


namespace number_of_cages_l279_279648

/-- Given the following conditions:
1. Each cage has 6 parrots.
2. Each cage has 2 parakeets.
3. The pet store has a total of 48 birds.
Prove that the number of bird cages in the pet store is 6. -/
theorem number_of_cages (parrots per_cage : ℕ) (parakeets per_cage : ℕ) (total_birds : ℕ)
  (h_parrots : parrots per_cage = 6) (h_parakeets : parakeets per_cage = 2) (h_total_birds : total_birds = 48) :
  ∃ x : ℕ, x * (parrots per_cage + parakeets per_cage) = total_birds ∧ x = 6 :=
by
  use 6
  rw [h_parrots, h_parakeets, h_total_birds]
  sorry

end number_of_cages_l279_279648


namespace bar_weight_calc_l279_279309

variable (blue_weight green_weight num_blue_weights num_green_weights bar_weight total_weight : ℕ)

theorem bar_weight_calc
  (h1 : blue_weight = 2)
  (h2 : green_weight = 3)
  (h3 : num_blue_weights = 4)
  (h4 : num_green_weights = 5)
  (h5 : total_weight = 25)
  (weights_total := num_blue_weights * blue_weight + num_green_weights * green_weight)
  : bar_weight = total_weight - weights_total :=
by
  sorry

end bar_weight_calc_l279_279309


namespace find_f_neg_one_l279_279484

variable {R : Type} [LinearOrderedField R]

noncomputable def f (x : R) (m : R) : R :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : R) : (2^0 + 2 * 0 + m = 0) → f (-1) (-1) = -3 :=
by
  intro h1
  have h2 : m = -1 := by linarith
  rw [f, if_neg (show -1 >= 0, by linarith)]
  simp only [f, h2]
  norm_num
  sorry

end find_f_neg_one_l279_279484


namespace laptop_final_price_l279_279641

theorem laptop_final_price (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  initial_price = 500 → first_discount = 10 → second_discount = 20 →
  (initial_price * (1 - first_discount / 100) * (1 - second_discount / 100)) = initial_price * 0.72 :=
by
  sorry

end laptop_final_price_l279_279641


namespace original_numbers_l279_279586

theorem original_numbers (a b c d : ℝ) (h1 : a + b + c + d = 45)
    (h2 : ∃ x : ℝ, a + 2 = x ∧ b - 2 = x ∧ 2 * c = x ∧ d / 2 = x) : 
    a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 :=
by
  sorry

end original_numbers_l279_279586


namespace odd_function_behavior_l279_279391

variable {f : ℝ → ℝ}

theorem odd_function_behavior (h1 : ∀ x : ℝ, f (-x) = -f x) 
                             (h2 : ∀ x : ℝ, 0 < x → f x = x * (1 + x)) 
                             (x : ℝ)
                             (hx : x < 0) : 
  f x = x * (1 - x) :=
by
  -- Insert proof here
  sorry

end odd_function_behavior_l279_279391


namespace probability_point_between_lines_l279_279082

theorem probability_point_between_lines :
  let p := λ x : ℝ, -2 * x + 8
  let q := λ x : ℝ, -3 * x + 8
  let triangle_area (f : ℝ → ℝ) (x1 x2 y1 : ℝ) := (1 / 2) * (x2 - x1) * y1 
  let area_under_p := triangle_area p 0 4 8 
  let area_under_q := triangle_area q 0 (8 / 3) 8
  ∃ (prob : ℝ), prob = (area_under_p - area_under_q) / area_under_p 
  ∧ prob = 0.33 :=
by
  sorry

end probability_point_between_lines_l279_279082


namespace total_shoes_sum_is_154_l279_279678

noncomputable def total_shoes := 
  let Bn := 13
  let B := (Bn + 5) / 2
  let Bb := 3.5 * B
  let Ch := (Bn + B) + 4.5
  let Dn := (3 * Ch) - 5
  Bn + B + Bb + Ch + Dn

theorem total_shoes_sum_is_154 : ⌊total_shoes⌋ = 154 :=
by
  sorry

end total_shoes_sum_is_154_l279_279678


namespace distance_MN_l279_279760

open Real

-- Given conditions
def line_eq (x y : ℝ) : Prop := x + y - 2 = 0
def curve_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Polar coordinate system transformation
def to_polar_x (ρ θ : ℝ) : ℝ := ρ * cos θ
def to_polar_y (ρ θ : ℝ) : ℝ := ρ * sin θ

-- Transformed polar forms of the given equations
def polar_line_eq (ρ θ : ℝ) : Prop := ρ * (sin θ + cos θ) = 2
def polar_curve_eq (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

-- Given the specific angle
def theta : ℝ := π / 4

-- Points on the line and curve in polar coordinates at θ = π/4
def ρ_M : ℝ := 2 / (sin theta + cos theta)
def ρ_N : ℝ := 4 * cos theta

-- Proof statement
theorem distance_MN : abs (ρ_N - ρ_M) = sqrt 2 := by
  sorry

end distance_MN_l279_279760


namespace maximum_omega_range_of_m_l279_279793

-- Condition: monotonicity of f(x)
def f (x : ℝ) (ω : ℝ) := 4 * Real.sin (ω * x + Real.pi / 3)
def monotonicDecreasing (ω : ℝ) : Prop := 
  ∀ x1 x2, (π / 6) ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ π → f x1 ω ≥ f x2 ω

-- Condition: symmetry about the point (3π/2, 0)
def symmetricGraph (ω : ℝ) : Prop := 
  ∃ (k : ℤ), ω = (2 * k - 1) * (2 / (3 * π))

-- Condition: the range of f(x) on [-9π/20, m] is [-2, 4]
def rangeCondition (ω : ℝ) (m : ℝ) : Prop := 
  ∀ x, -(9 * π / 20) ≤ x ∧ x ≤ m → -2 ≤ f x ω ∧ f x ω ≤ 4

-- Correct answer for maximum ω
def maxOmega := 7 / 6

-- Correct range for m
def rangeM := set.Icc (3 * π / 20) (3 * π / 4)

-- Proof of maximum ω
theorem maximum_omega (ω : ℝ) :
  monotonicDecreasing ω → ω ≤ maxOmega :=
sorry

-- Proof of range of m
theorem range_of_m (ω : ℝ) :
  symmetricGraph ω → rangeCondition ω (3 * π / 20) ∧ rangeCondition ω (3 * π / 4) :=
sorry

end maximum_omega_range_of_m_l279_279793


namespace sqrt_9800_minus_53_in_form_l279_279916

theorem sqrt_9800_minus_53_in_form (a b : ℕ) (h₁: (a + 3 * b^2) * √a = √9800) (h₂: -3 * a * b - b^3 = -53) :
  a + b = 18 :=
  sorry

end sqrt_9800_minus_53_in_form_l279_279916


namespace unique_value_not_in_range_l279_279563

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_value_not_in_range (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  g p q r s 11 = 11 ∧ g p q r s 41 = 41 ∧ (∀ x, x ≠ -s / r → g p q r s (g p q r s x) = x) → ∃! y, ¬ ∃ x, g p q r s x = y :=
begin
  intros h,
  use 30,
  split,
  {
    intro hy,
    sorry -- Proof omitted
  },
  {
    intros z hz,
    have hneq : z ≠ 30 := sorry, -- Proof omitted
    assumption
  }
end

end unique_value_not_in_range_l279_279563


namespace external_triangles_concurrent_internal_triangles_concurrent_l279_279254

variables {α β γ : ℝ}
variables {A B C A1 B1 C1 : Type*}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited A1] [inhabited B1] [inhabited C1]

-- Conditions
axiom hαβ : α + β < 180
axiom hβγ : β + γ < 180
axiom hγα : γ + α < 180

axiom h_triangle1 : ∀ (A B C A1 : Type*), ∃ (AA1 : Type*), AA1 -- Line AA1 in triangle A_1BC
axiom h_triangle2 : ∀ (A B C B1 : Type*), ∃ (BB1 : Type*), BB1 -- Line BB1 in triangle AB_1C
axiom h_triangle3 : ∀ (A B C C1 : Type*), ∃ (CC1 : Type*), CC1 -- Line CC1 in triangle ABC_1

theorem external_triangles_concurrent 
  (h1 : h_triangle1 A B C A1)
  (h2 : h_triangle2 A B C B1)
  (h3 : h_triangle3 A B C C1) :
  let AA1 := classical.some h1,
      BB1 := classical.some h2,
      CC1 := classical.some h3 in
  concurrent AA1 BB1 CC1 :=
sorry

-- Setting up the internal triangles scenario:
axiom h_internal_triangle1 : ∀ (A B C A1 : Type*), ∃ (AA1' : Type*), AA1' -- Line AA1 in internally constructed triangle
axiom h_internal_triangle2 : ∀ (A B C B1 : Type*), ∃ (BB1' : Type*), BB1' -- Line BB1 in internally constructed triangle
axiom h_internal_triangle3 : ∀ (A B C C1 : Type*), ∃ (CC1' : Type*), CC1' -- Line CC1 in internally constructed triangle

theorem internal_triangles_concurrent 
  (h4 : h_internal_triangle1 A B C A1)
  (h5 : h_internal_triangle2 A B C B1)
  (h6 : h_internal_triangle3 A B C C1) :
  let AA1' := classical.some h4,
      BB1' := classical.some h5,
      CC1' := classical.some h6 in
  concurrent AA1' BB1' CC1' :=
sorry

end external_triangles_concurrent_internal_triangles_concurrent_l279_279254


namespace smallest_integer_repr_CCCD8_l279_279980

theorem smallest_integer_repr_CCCD8 (C D : ℕ) (hC : C < 6) (hD : D < 8)
    (h_eq : 7 * C = 9 * D) : ∃ n : ℕ, (n = 7 * C) ∧ (n = 9 * D) ∧ (7 * C = 63) :=
by {
  existsi 63,
  split,
  { simp [←h_eq, mul_comm, mul_assoc, Nat.mul_div_cancel_left, Nat.gcd_eq, Nat.lcm_eq_gcd_mul] },
  { exact h_eq }
}

end smallest_integer_repr_CCCD8_l279_279980


namespace probability_of_intersection_l279_279054

noncomputable def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
noncomputable def B : set ℝ := {x | (x - 2) * (3 - x) ≥ 0}

theorem probability_of_intersection :
  (∃ (measure : ennreal), measure = (1 : ℝ)/(6 : ℝ)) ↔ 
    (∃ lenA lenIntersection : ennreal, 
     lenA = (5 - (-1)) ∧ 
     lenIntersection = (3 - 2) ∧ 
     measure = lenIntersection / lenA) := 
by
  sorry

end probability_of_intersection_l279_279054


namespace factorization_correct_l279_279197

theorem factorization_correct : ∃ (a b : ℕ), (a > b) ∧ (3 * b - a = 12) ∧ (x^2 - 16 * x + 63 = (x - a) * (x - b)) :=
by
  sorry

end factorization_correct_l279_279197


namespace find_p_q_r_l279_279493

theorem find_p_q_r :
  ∃ (p q r : ℕ), 
    let n := p + Real.sqrt (q + Real.sqrt r) in
    n = (fun x => if 0 < (λ x, (x^2 - 7*x - 6 - ((4 / (x-2)) + (6 / (x-6)) + (13 / (x-13)) + (15 / (x-15))))) x then x else 0) x
   ∧ p + q + r = 103 :=
by
  sorry

end find_p_q_r_l279_279493


namespace percentage_is_correct_l279_279632

noncomputable def find_percentage (P : ℝ) : ℝ :=
  if 30% * 50% * 5600 = 126 then P * 5600 else 0

theorem percentage_is_correct (P : ℝ) : 
  (30% * 50% * 5600 = 126 → P = 126 / 5600) → P * 100 = 2.25 :=
by
  sorry

end percentage_is_correct_l279_279632


namespace rectangular_to_cylindrical_l279_279330

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (hx : x = 6) (hy : y = -6 * Real.sqrt 3) (hz : z = 4) 
  (hr1 : r = Real.sqrt (x^2 + y^2)) (hr2 : r > 0) (hθ1 : tan θ = y / x)
  (hθ2 : 0 ≤ θ) (hθ3 : θ < 2 * Real.pi) :
  (r = 12) ∧ (θ = 4 * Real.pi / 3) ∧ (z = 4) :=
by
  sorry

end rectangular_to_cylindrical_l279_279330


namespace omega_range_l279_279775

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l279_279775


namespace Kendra_weekly_words_not_determined_without_weeks_l279_279470

def Kendra_goal : Nat := 60
def Kendra_already_learned : Nat := 36
def Kendra_needs_to_learn : Nat := 24

theorem Kendra_weekly_words_not_determined_without_weeks (weeks : Option Nat) : weeks = none → Kendra_needs_to_learn / weeks.getD 1 = 24 -> False := by
  sorry

end Kendra_weekly_words_not_determined_without_weeks_l279_279470


namespace first_day_of_month_l279_279180

theorem first_day_of_month (h : weekday 30 = "Wednesday") : weekday 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279180


namespace difference_of_cubes_divisible_by_8_l279_279167

theorem difference_of_cubes_divisible_by_8 (a b : ℤ) : 
  8 ∣ ((2 * a - 1) ^ 3 - (2 * b - 1) ^ 3) := 
by
  sorry

end difference_of_cubes_divisible_by_8_l279_279167


namespace product_eval_at_3_l279_279706

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end product_eval_at_3_l279_279706


namespace centroid_coincidence_l279_279501

structure Tetrahedron where
  A B C D : Point

structure Heights (T : Tetrahedron) where
  h_a h_b h_c h_d : ℝ

structure PointsAlongHeights (T : Tetrahedron) (H : Heights T) where
  k : ℝ
  A1 B1 C1 D1 : Point
  AA1_eq : T.A.distance_to A1 = k / H.h_a
  BB1_eq : T.B.distance_to B1 = k / H.h_b
  CC1_eq : T.C.distance_to C1 = k / H.h_c
  DD1_eq : T.D.distance_to D1 = k / H.h_d

theorem centroid_coincidence
    (T : Tetrahedron)
    (H : Heights T)
    (PH : PointsAlongHeights T H) :
    centroid T.A T.B T.C T.D = centroid PH.A1 PH.B1 PH.C1 PH.D1 := 
sorry

end centroid_coincidence_l279_279501


namespace sum_of_coefficients_is_60_l279_279558

theorem sum_of_coefficients_is_60 :
  ∀ (a b c d e : ℤ), (∀ x : ℤ, 512 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) →
  a + b + c + d + e = 60 :=
by
  intros a b c d e h
  sorry

end sum_of_coefficients_is_60_l279_279558


namespace number_of_intersection_points_eq_10_l279_279324

theorem number_of_intersection_points_eq_10 :
  let f (x : ℝ) := (x - floor x)
  let graph1 (x y : ℝ) := (f x)^2 + y^2 = f x
  let graph2 (x y : ℝ) := y = (1/4) * x
  ∃ (points : set (ℝ × ℝ)), (∀ p ∈ points, graph1 p.1 p.2 ∧ graph2 p.1 p.2) 
  ∧ set.finite points ∧ set.card points = 10 := 
sorry

end number_of_intersection_points_eq_10_l279_279324


namespace smallest_integer_CC6_DD8_l279_279982

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l279_279982


namespace arithmetic_sequence_sum_positive_l279_279740

noncomputable def a_2015 : ℕ → ℝ
noncomputable def a_2016 : ℕ → ℝ

theorem arithmetic_sequence_sum_positive {a : ℕ → ℝ}
  (h₁ : ∀ n, a n = a 1 + (n - 1) * d) 
  (h₂ : a 1 > 0)
  (h₃ : a_2015 * a_2016 < 0)
  (h₄ : a_2015 + a_2016 > 0) : 
  ∃ n : ℕ, is_largest_n n 4029 :=
sorry

-- Helper definition for largest natural number
def is_largest_n (n : ℕ) (max_n : ℕ) : Prop :=
  (sum (range n) a > 0) ∧ (sum (range (max_n + 1)) a = 0)

end arithmetic_sequence_sum_positive_l279_279740


namespace radius_of_circle_l279_279921

-- Define the problem and its conditions
variables (r : ℝ) (π : ℝ := Real.pi)

-- Given conditions
def condition1 : Prop := π * r^2 = 64 * π

-- Statement
theorem radius_of_circle (h : condition1) : r = 8 :=
sorry

end radius_of_circle_l279_279921


namespace inclination_angle_x_minus_1_l279_279202

-- Definition of inclination angle of a line parallel to the y-axis.
def inclination_angle_parallel_y_axis (x : ℝ) (hx : x = -1) : ℝ :=
  if hx then π / 2 else 0  -- Since we know this function should only take hx = true case as given by problem condition.

-- Main proof statement
theorem inclination_angle_x_minus_1 : inclination_angle_parallel_y_axis (-1) (eq.refl (-1)) = π / 2 :=
by sorry

end inclination_angle_x_minus_1_l279_279202


namespace circumscribed_circle_radius_l279_279543

theorem circumscribed_circle_radius (A : ℝ) (R : ℝ) (h : A = 81) : R = 6 * real.sqrt(real.sqrt(3)) :=
by
  sorry

end circumscribed_circle_radius_l279_279543


namespace heart_sum_equals_ten_l279_279363

def heart (x : ℝ) : ℝ :=
  (x + x^2) / 2

theorem heart_sum_equals_ten :
  heart 1 + heart 2 + heart 3 = 10 :=
by
  sorry

end heart_sum_equals_ten_l279_279363


namespace point_B_coordinates_l279_279513

variable (A : ℝ × ℝ)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

theorem point_B_coordinates : 
  (move_left (move_up (-3, -5) 4) 3) = (-6, -1) :=
by
  sorry

end point_B_coordinates_l279_279513


namespace distance_max_min_point_on_ellipse_l279_279885

def on_ellipse (x y : ℝ) : Prop :=
  (x ^ 2) / 16 + (y ^ 2) / 9 = 1

def line_eq (x y : ℝ) : Prop :=
  3 * x - 4 * y = 24

noncomputable def distance_from_point_to_line (x y : ℝ) : ℝ :=
  abs (12 * x / 4 - 12 * y / 3 - 24) / (real.sqrt (3 ^ 2 + (-4) ^ 2))

theorem distance_max_min_point_on_ellipse :
  ∀ (x y : ℝ), on_ellipse x y →
  max (distance_from_point_to_line x y) (distance_from_point_to_line x y) = 12 * (2 + real.sqrt 2) / 5 ∧
  min (distance_from_point_to_line x y) (distance_from_point_to_line x y) = 12 * (2 - real.sqrt 2) / 5 :=
sorry

end distance_max_min_point_on_ellipse_l279_279885


namespace common_volume_tetrahedron_theorem_l279_279444

noncomputable def common_volume_fraction (original_tetrahedron : ℕ) : ℚ :=
  1 / 10

theorem common_volume_tetrahedron_theorem (original_tetrahedron : Type) 
  [volume : ∀ t : original_tetrahedron, ℚ] :
  ∃ common_volume : ℚ,
    common_volume = common_volume_fraction (volume original_tetrahedron) :=
by
  sorry

end common_volume_tetrahedron_theorem_l279_279444


namespace T_n_correct_l279_279861

-- Definitions corresponding to conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a(n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a(0) + (n * (n + 1) / 2) * (a(1) - a(0))

variable (a : ℕ → ℝ)
variable (S_n T_n : ℕ → ℝ)

-- Given conditions
axiom S_7_eq_7 : S_n 7 = 7
axiom S_15_eq_75 : S_n 15 = 75
axiom S_n_def : ∀ n, S_n n = sum_of_first_n_terms a n

-- First question proof: find a_1 and d
noncomputable def a_1 : ℝ := -2
noncomputable def d : ℝ := 1

-- Assuming arithmetic sequence with first term a_1 and common difference d
axiom a_sequence : is_arithmetic_sequence a

-- Second question proof: find T_n
noncomputable def T_n_formula (n : ℕ) : ℝ :=
  (1/4) * n ^ 2 - (9/4) * n

-- Required theorem to prove
theorem T_n_correct : ∀ n, T_n n = T_n_formula n :=
sorry

end T_n_correct_l279_279861


namespace pyramid_section_area_l279_279192

theorem pyramid_section_area:
  let side_length := 8
  let height := 9
  let angle := Real.arctan (3 / 4)
  (area_section : ℝ) :=
  area_section = 45 := 
sorry

end pyramid_section_area_l279_279192


namespace arc_length_proof_l279_279679
noncomputable def arc_length_of_curve : Real :=
  let rho (ϕ : Real) := 7 * (1 - Real.sin ϕ)
  let drho_dϕ (ϕ : Real) := -7 * Real.cos ϕ
  7 * ∫ ϕ in (-(π / 6))..(π / 6), Real.sqrt (2 - 2 * Real.sin ϕ)

theorem arc_length_proof :
  arc_length_of_curve = 14 * (Real.sqrt 3 - 1) :=
by sorry

end arc_length_proof_l279_279679


namespace quadratic_value_l279_279153

theorem quadratic_value (a b c : ℤ) (a_pos : a > 0) (h_eq : ∀ x : ℝ, (a * x + b)^2 = 49 * x^2 + 70 * x + c) : a + b + c = -134 :=
by
  -- Proof starts here
  sorry

end quadratic_value_l279_279153


namespace reciprocal_of_2_l279_279949

theorem reciprocal_of_2 : 1 / 2 = 1 / (2 : ℝ) := by
  sorry

end reciprocal_of_2_l279_279949


namespace sundae_price_l279_279264

def price_per_sundae (total_items : Nat) (total_price : Float) (ice_cream_bars : Nat) (price_ice_cream_bar : Float) (num_sundaes : Nat) : Float :=
  let cost_ice_cream_bars := ice_cream_bars * price_ice_cream_bar
  let cost_sundaes := total_price - cost_ice_cream_bars
  cost_sundaes / num_sundaes

theorem sundae_price :
  price_per_sundae 250 250.0 125 0.60 125 = 1.4 :=
by
  sorry

end sundae_price_l279_279264


namespace formation_of_NH4OH_l279_279357

-- Define necessary chemical substances
def NH4Cl : Type := ℕ  -- Representation for ammonium chloride
def NaOH : Type := ℕ    -- Representation for sodium hydroxide
def NH4OH : Type := ℕ   -- Representation for ammonium hydroxide
def NaCl : Type := ℕ    -- Representation for sodium chloride

-- Define the initial amounts of reactants
def initial_moles_NH4Cl : NH4Cl := 1
def initial_moles_NaOH : NaOH := 1

-- Define the reaction function based on stoichiometry
def reaction (NH4Cl : ℕ) (NaOH : ℕ) : NH4OH :=
  if NH4Cl = NaOH then NH4OH else 0

-- State the theorem
theorem formation_of_NH4OH :
  reaction initial_moles_NH4Cl initial_moles_NaOH = 1 :=
by sorry

end formation_of_NH4OH_l279_279357


namespace spider_moves_away_from_bee_l279_279261

noncomputable def bee : ℝ × ℝ := (14, 5)
noncomputable def spider_line (x : ℝ) : ℝ := -3 * x + 25
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1 / 3) * x + 14 / 3

theorem spider_moves_away_from_bee : ∃ (c d : ℝ), 
  (d = spider_line c) ∧ (d = perpendicular_line c) ∧ c + d = 13.37 := 
sorry

end spider_moves_away_from_bee_l279_279261


namespace initial_milk_in_container_A_l279_279255

theorem initial_milk_in_container_A (A B C D : ℝ) 
  (h1 : B = A - 0.625 * A) 
  (h2 : C - 158 = B) 
  (h3 : D = 0.45 * (C - 58)) 
  (h4 : D = 58) 
  : A = 231 := 
sorry

end initial_milk_in_container_A_l279_279255


namespace smallest_value_for_x_eq_9_l279_279362

theorem smallest_value_for_x_eq_9 :
  ∀ (x : ℕ), x = 9 →
  (min
    (6 / x : ℚ)
    (min
      (6 / (x + 1) : ℚ)
      (min
        (6 / (x - 1) : ℚ)
        (min
          (x / 6 : ℚ)
          (min
            ((x + 1) / 6 : ℚ)
            (x / (x - 1) : ℚ))))) = (6 / (x + 1) : ℚ) :=
by 
  sorry

end smallest_value_for_x_eq_9_l279_279362


namespace license_plates_count_divided_by_10_l279_279279

def characters := {'A', 'I', 'M', 'E', '2', '0', '7'}
def max_occurrences : ∀ c, c ∈ characters → ℕ
| 'A' h := 1
| 'I' h := 1
| 'M' h := 1
| 'E' h := 1
| '2' h := 1
| '0' h := 2
| '7' h := 1

theorem license_plates_count_divided_by_10 :
  let N := (∑ (s : Finset (Fin 5 → characters)), 
            (∀ c ∈ characters, s.count c ≤ max_occurrences c)) in
  N / 10 = 372 :=
by sorry

end license_plates_count_divided_by_10_l279_279279


namespace coefficient_of_x2021_l279_279994

/-- Prove that the coefficient of x^2021 in the expanded product of 
  (2021 * x^2021 + 2020 * x^2020 + ... + 3 * x^3 + 2 * x^2 + x)
  and (x^2021 - x^2020 + ... + x^3 - x^2 + x - 1) is -1011. -/
theorem coefficient_of_x2021 :
  let P := (λ x : ℝ, finset.sum (finset.range 2022) (λ k, (2021 - k) * x ^ (2021 - k)))
  let Q := (λ x : ℝ, finset.sum (finset.range 2022) (λ k, (-1)^(k+1) * x ^ (2021 - k)))
  polynomial.coeff (P * Q) 2021 = -1011 :=
by 
  -- We need to prove this is equal to -1011.
  sorry

end coefficient_of_x2021_l279_279994


namespace value_of_a_for_pure_imaginary_l279_279427

def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem value_of_a_for_pure_imaginary (a : ℝ) : 
  pure_imaginary ((a + complex.I) * (2 + complex.I)) → a = 1/2 :=
by
  sorry

end value_of_a_for_pure_imaginary_l279_279427


namespace greatest_number_of_candies_l279_279658

theorem greatest_number_of_candies(taken : ℕ) (students : ℕ) (mean : ℝ) (h_student_count : students = 30) (h_mean : mean = 5) (h_everyone_takes_at_least_one : ∀ i, i < students → taken ≥ 1) :
  taken = 121 :=
by
  -- Definitions based on the conditions
  let total_students := 30
  let mean_taken := 5
  let total_candies := total_students * mean_taken
  let min_candies_for_29 := (total_students - 1) * 1
  let max_candies_for_last := total_candies - min_candies_for_29
  
  have total_candies_hyp : total_candies = 150 := by norm_num
  have min_candies_for_29_hyp : min_candies_for_29 = 29 := by norm_num
  have max_candies_for_last_hyp : max_candies_for_last = 121 := by norm_num
  
  exact max_candies_for_last_hyp

end greatest_number_of_candies_l279_279658


namespace ten_crates_probability_l279_279917

theorem ten_crates_probability (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  let num_crates := 10
  let crate_dimensions := [3, 4, 6]
  let target_height := 41

  -- Definition of the generating function coefficients and constraints will be complex,
  -- so stating the specific problem directly.
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ m = 190 ∧ n = 2187 →
  let probability := (m : ℚ) / (n : ℚ)
  probability = (190 : ℚ) / 2187 := 
by
  sorry

end ten_crates_probability_l279_279917


namespace sum_tan_not_equal_l279_279317

-- Define angles and their properties for triangles
variables {α β γ : ℝ} (a_angle b_angle c_angle : ℝ)

-- Conditions for acute and obtuse angles
def acute_triangle := (α < π / 2) ∧ (β < π / 2) ∧ (γ < π / 2) ∧ (α + β + γ = π)
def obtuse_triangle := (∃ (delta : ℝ), delta > π / 2 ∧ (alpha_angle + beta_angle + delta = π))

-- Problem statement in Lean
theorem sum_tan_not_equal (α β γ : ℝ) (delta ε ζ : ℝ) :
  acute_triangle α β γ
  → obtuse_triangle δ ε ζ
  → ¬ (tan α + tan β + tan γ = tan δ + tan ε + tan ζ)
:= sorry

end sum_tan_not_equal_l279_279317


namespace subtracting_seven_percent_l279_279257

theorem subtracting_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a :=
by 
  sorry

end subtracting_seven_percent_l279_279257


namespace number_of_safe_integers_l279_279713

def is_p_safe (n : ℕ) (p : ℕ) : Prop :=
  ∀ k : ℤ, abs (n - k * p) > 2

def count_simultaneously_safe (n : ℕ) : ℕ :=
  (filter (λ x, is_p_safe x 5 ∧ is_p_safe x 7 ∧ is_p_safe x 11) (list.range (n + 1))).length

theorem number_of_safe_integers : count_simultaneously_safe 10000 = 0 :=
by
  sorry

end number_of_safe_integers_l279_279713


namespace range_of_quadratic_l279_279571

noncomputable def f (x : ℝ) := x^2 - 4 * x + 1

theorem range_of_quadratic :
  set.range (λ x : ℝ, f x) ∩ set.Icc 3 5 = set.Icc (-2 : ℝ) 6 :=
sorry

end range_of_quadratic_l279_279571


namespace cricket_team_number_of_players_l279_279960

theorem cricket_team_number_of_players 
  (throwers : ℕ)
  (total_right_handed : ℕ)
  (frac_left_handed_non_thrower : ℚ)
  (right_handed_throwers : throwers = 37)
  (all_throwers_right_handed : throwers * 1 = throwers)
  (total_right_handed_players : total_right_handed = 55)
  (left_handed_non_thrower_fraction : frac_left_handed_non_thrower = 1 / 3)
  (right_handed_non_throwers : total_right_handed - throwers = 18) :
  let non_throwers := (total_right_handed - throwers) * (1 / frac_left_handed_non_thrower - 1) in
  let total_players := throwers + non_throwers.to_nat in
  total_players = 64 :=
by 
  let non_throwers := 18 * (3 / 2) in 
  let total_players := 37 + non_throwers in
  have : total_players = 64 := rfl
  sorry

end cricket_team_number_of_players_l279_279960


namespace certain_event_drawing_triangle_interior_angles_equal_180_deg_l279_279240

-- Define a triangle in the Euclidean space
structure Triangle (α : Type) [plane : TopologicalSpace α] :=
(a b c : α)

-- Define the sum of the interior angles of a triangle
noncomputable def sum_of_interior_angles {α : Type} [TopologicalSpace α] (T : Triangle α) : ℝ :=
180

-- The proof statement
theorem certain_event_drawing_triangle_interior_angles_equal_180_deg {α : Type} [TopologicalSpace α]
(T : Triangle α) : 
(sum_of_interior_angles T = 180) :=
sorry

end certain_event_drawing_triangle_interior_angles_equal_180_deg_l279_279240


namespace tangent_line_hyperbola_l279_279557

variable {a b x x₀ y y₀ : ℝ}
variable (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (he : x₀^2 / a^2 + y₀^2 / b^2 = 1)
variable (hh : x₀^2 / a^2 - y₀^2 / b^2 = 1)

theorem tangent_line_hyperbola
  (h_tangent_ellipse : (x₀ * x / a^2 + y₀ * y / b^2 = 1)) :
  (x₀ * x / a^2 - y₀ * y / b^2 = 1) :=
sorry

end tangent_line_hyperbola_l279_279557


namespace scrambled_eggs_count_l279_279471

-- Definitions based on the given conditions
def num_sausages := 3
def time_per_sausage := 5
def time_per_egg := 4
def total_time := 39

-- Prove that Kira scrambled 6 eggs
theorem scrambled_eggs_count : (total_time - num_sausages * time_per_sausage) / time_per_egg = 6 := by
  sorry

end scrambled_eggs_count_l279_279471


namespace number_of_solutions_l279_279809

theorem number_of_solutions :
  {N : ℕ | N < 500 ∧ (∃ x : ℝ, x > 0 ∧ (floor x) ≥ 0 ∧ x^((floor x) : ℝ) = N) ∧ even N}.to_finset.card = 143 :=
by sorry

end number_of_solutions_l279_279809


namespace shirt_price_is_150_l279_279346

def price_of_shirt (X C : ℝ) : Prop :=
  (X + C = 600) ∧ (X = C / 3)

theorem shirt_price_is_150 :
  ∃ X C : ℝ, price_of_shirt X C ∧ X = 150 :=
by {
  use [150, 450],
  dsimp [price_of_shirt],
  split,
  { norm_num, },
  { norm_num, },
}

end shirt_price_is_150_l279_279346


namespace medians_equal_segments_l279_279849

variable {A B C A₁ B₁ : Type}
variables [MetricSpace α] [Metric α]
variables (triangle : Triangle α)
variables (A B C A₁ B₁ : α)

-- Given: Medians
variable (median_A : Segment A C A₁)
variable (median_B : Segment B C B₁)

-- Condition: Equal angles
variable (equal_angles : ∠ A A₁ C = ∠ B B₁ C)

-- Goal: Prove AC = BC
theorem medians_equal_segments (hA : median_A.is_median) (hB : median_B.is_median) (h_angle : equal_angles) : dist A C = dist B C := 
sorry

end medians_equal_segments_l279_279849


namespace integral_even_function_integral_odd_function_l279_279521

variables {α : Type*} [integrable_space α] {a : ℤ} (f : α → ℝ)

-- First part: Even function
theorem integral_even_function (hf : ∀ x, f (-x) = f x) :
  ∫ x in (-a)..a, f x = 2 * ∫ x in 0..a, f x :=
sorry

-- Second part: Odd function
theorem integral_odd_function (hf : ∀ x, f (-x) = -f x) :
  ∫ x in (-a)..a, f x = 0 :=
sorry

end integral_even_function_integral_odd_function_l279_279521


namespace find_y_coordinate_of_P_l279_279125

def coordinates_A : ℝ × ℝ := (-4, 0)
def coordinates_B : ℝ × ℝ := (-3, 2)
def coordinates_C : ℝ × ℝ := (3, 2)
def coordinates_D : ℝ × ℝ := (4, 0)

def PA (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 + 4)^2 + (P.2 - 0)^2)
def PB (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 + 3)^2 + (P.2 - 2)^2)
def PC (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - 3)^2 + (P.2 - 2)^2)
def PD (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - 4)^2 + (P.2 - 0)^2)

def satisfies_conditions (P : ℝ × ℝ) : Prop := PA P + PD P = 10 ∧ PB P + PC P = 10

def y_coordinate_of_P (P : ℝ × ℝ) : ℝ := P.2

theorem find_y_coordinate_of_P (P : ℝ × ℝ) (h : satisfies_conditions P) :
  y_coordinate_of_P P = 6 / 7 :=
sorry

end find_y_coordinate_of_P_l279_279125


namespace first_day_of_month_l279_279181

theorem first_day_of_month (h : weekday 30 = "Wednesday") : weekday 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279181


namespace students_on_field_trip_l279_279881

theorem students_on_field_trip (vans: ℕ) (capacity_per_van: ℕ) (adults: ℕ) 
  (H_vans: vans = 3) 
  (H_capacity_per_van: capacity_per_van = 5) 
  (H_adults: adults = 3) : 
  (vans * capacity_per_van - adults = 12) :=
by
  sorry

end students_on_field_trip_l279_279881


namespace difference_between_smallest_and_second_smallest_3_digit_numbers_l279_279711

theorem difference_between_smallest_and_second_smallest_3_digit_numbers : 
  ∀ (a b c : ℕ), (a = 1) → (b = 6) → (c = 8) →
  let smallest := min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))) in
  let second_smallest := if smallest = 100 * a + 10 * b + c then min (min (100 * a + 10 * c + b) (100 * b + 10 * a + c)) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * a + 10 * c + b then min (min (100 * a + 10 * b + c) (100 * b + 10 * a + c)) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * b + 10 * a + c then min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * b + 10 * c + a then min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * c + 10 * a + b then min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * b + 10 * c + a) (100 * c + 10 * b + a)))
    else min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * b + 10 * c + a) (100 * c + 10 * a + b))) in
  (second_smallest - smallest = 18) :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  let smallest := min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a))))
  let second_smallest := if smallest = 100 * a + 10 * b + c then min (min (100 * a + 10 * c + b) (100 * b + 10 * a + c)) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * a + 10 * c + b then min (min (100 * a + 10 * b + c) (100 * b + 10 * a + c)) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * b + 10 * a + c then min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * c + a) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * b + 10 * c + a then min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * c + 10 * a + b) (100 * c + 10 * b + a)))
    else if smallest = 100 * c + 10 * a + b then min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * b + 10 * c + a) (100 * c + 10 * b + a)))
    else min (min (100 * a + 10 * b + c) (100 * a + 10 * c + b)) (min (100 * b + 10 * a + c) (min (100 * b + 10 * c + a) (100 * c + 10 * a + b)))
  sorry

end difference_between_smallest_and_second_smallest_3_digit_numbers_l279_279711


namespace determine_c_l279_279341

noncomputable def c_floor : ℤ := -3
noncomputable def c_frac : ℝ := (25 - Real.sqrt 481) / 8

theorem determine_c : c_floor + c_frac = -2.72 := by
  have h1 : 3 * (c_floor : ℝ)^2 + 19 * (c_floor : ℝ) - 63 = 0 := by
    sorry
  have h2 : 4 * c_frac^2 - 25 * c_frac + 9 = 0 := by
    sorry
  sorry

end determine_c_l279_279341


namespace find_p_q_l279_279422

theorem find_p_q (p q : ℚ) : 
    (∀ x, x^5 - x^4 + x^3 - p*x^2 + q*x + 9 = 0 → (x = -3 ∨ x = 2)) →
    (p, q) = (-19.5, -55.5) :=
by {
  sorry
}

end find_p_q_l279_279422


namespace commodity_y_annual_increase_l279_279207

-- Given conditions
def price_x_initial := 4.20
def price_y_initial := 4.40
def annual_increase_x := 0.30
def year_start := 2001
def year_end := 2012
def price_difference := 0.90

-- Derived condition to find the annual increase of y
theorem commodity_y_annual_increase:
  ∃ d : ℝ, (price_x_initial + (year_end - year_start) * annual_increase_x) 
            = price_y_initial + (year_end - year_start) * d + price_difference ∧ 
          d = 0.20 :=
by
  -- Statement of the theorem
  sorry

end commodity_y_annual_increase_l279_279207


namespace anya_initial_seat_l279_279830

theorem anya_initial_seat (V G D E A : ℕ) (A' : ℕ) 
  (h1 : V + G + D + E + A = 15)
  (h2 : V + 1 ≠ A')
  (h3 : G - 3 ≠ A')
  (h4 : (D = A' → E ≠ A') ∧ (E = A' → D ≠ A'))
  (h5 : A = 3 + 2)
  : A = 3 := by
  sorry

end anya_initial_seat_l279_279830


namespace hyperbola_eccentricity_l279_279003

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (c : ℝ) (hc : c = b^2 / a)
  (e : ℝ) (he : a * e = c)
  (hyp : ∃ F B, ∃ asymptote : ℝ → ℝ, 
          F = (c, 0) ∧ B = (0, b) ∧ 
          asymptote = λ x, (b / a) * x ∧ 
          ((b - 0) / (0 - c)) * (b / a) = -1) : 
  e = (1 + Real.sqrt 5) / 2 := sorry

end hyperbola_eccentricity_l279_279003


namespace car_start_time_at_10_10_l279_279263

variable (speed_car1 speed_car2 distance_gap time_to_catch_up total_time_caught_up : ℕ)
-- Define the speeds of the cars
def speed_car1 : ℕ := 30 -- miles per hour
def speed_car2 : ℕ := 60 -- miles per hour
-- Convert speeds to miles per minute
def speed_car1_mins : ℕ := speed_car1 / 60 -- in miles per minute
def speed_car2_mins : ℕ := speed_car2 / 60 -- in miles per minute
-- Car 1 starts 10 minutes ahead
def distance_gap : ℕ := speed_car1_mins * 10 -- initial distance gap in miles
-- Relative speed between the two cars
def relative_speed_mins : ℕ := speed_car2_mins - speed_car1_mins
-- Time taken for car 2 to catch up with car 1 
def time_to_catch_up : ℕ := distance_gap / relative_speed_mins
-- Time when car 2 catches up
def caught_up_time : ℕ := 10 * 60 + 30 -- in minutes after midnight (630 minutes, corresponding to 10:30 a.m.)
-- The time when car 1 starts
def time_car1_start := caught_up_time - (10 + time_to_catch_up)

theorem car_start_time_at_10_10 : time_car1_start = 10 * 60 + 10 :=
  by
  sorry

end car_start_time_at_10_10_l279_279263


namespace smallest_integer_CC6_DD8_l279_279983

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l279_279983


namespace range_of_omega_l279_279773

theorem range_of_omega :
  ∀ (ω : ℝ), 
  (0 < ω) → 
  (∀ x, x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → cos (ω * x) - 1 = 0 → x ∈ {0, 2 * Real.pi, 4 * Real.pi}) →
  (2 ≤ ω ∧ ω < 3) :=
by
  intros ω hω_pos hzeros
  sorry

end range_of_omega_l279_279773


namespace problem_proof_l279_279843

-- Definitions for the trapezoid, points of intersection, and parallel lines
variables {Point : Type} [Geometry Point]
variables A B C D O M N : Point
variables {AB CD AD BC : Line Point}

-- Conditions
axiom trapezoid_cond (hTrapezoid : Trapezoid ABCD AB CD)
axiom intersect_cond (hIntersect : Intersect AC BD O)
axiom parallel_cond1 (hParallel1 : Parallel AB CD)
axiom parallel_cond2 (hParallel2 : Parallel MON AB)
axiom parallel_cond3 (hParallel3 : Parallel MON CD)
axiom on_lines (hOnM : OnLine M AD) (hOnN : OnLine N BC)

-- Statement to prove
theorem problem_proof :
  (1 / (length AB) + 1 / (length CD)) = 2 / (length MON) := by
  sorry

end problem_proof_l279_279843


namespace sum_distances_and_pq_l279_279344

-- Definitions for point and distances
structure Point where
  x : ℝ
  y : ℝ

def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Given points D, E, F
def D : Point := ⟨0, 0⟩
def E : Point := ⟨7, 2⟩
def F : Point := ⟨4, 5⟩

-- Given point Q
def Q : Point := ⟨3, 3⟩

-- Prove the sum of distances and simplified p + q
theorem sum_distances_and_pq :
  let DQ := dist D Q 
  let EQ := dist E Q 
  let FQ := dist F Q 
  (DQ + EQ + FQ = 3 * real.sqrt 2 + real.sqrt 17 + real.sqrt 5) ∧ 
  (3 + 1 + 1 = 5) :=
by
  sorry

end sum_distances_and_pq_l279_279344


namespace probability_of_both_contracts_l279_279621

variable (A B : Prop)
variable (P : Prop → ℝ)
axiom P_additivity : ∀ (X Y : Prop), P (X ∨ Y) = P X + P Y - P (X ∧ Y)
axiom P_complement : ∀ (X : Prop), P (¬ X) = 1 - P X

theorem probability_of_both_contracts (hA : P A = 4 / 5) (hB' : P (¬ B) = 3 / 5) (hAuB : P (A ∨ B) = 9 / 10) :
  P (A ∧ B) = 3 / 10 :=
by
  have hB : P B = 1 - P (¬ B) := P_complement B
  have hB_eq : P B = 2 / 5 := by rw [hB, hB']; ring
  have hAuB_eq : P (A ∨ B) = P A + P B - P (A ∧ B) := P_additivity A B
  have hP : P (A ∧ B) = P A + P B - P (A ∨ B) := by rw [hAuB_eq]; linarith
  rw [hP, hA, hB_eq, hAuB]
  ring

end probability_of_both_contracts_l279_279621


namespace sum_d_f_eq_neg4_l279_279961

variable (a b c d e f : ℂ)

constant h1 : b = 2
constant h2 : e = -2 * a - 2 * c
constant h3 : a + b * complex.I + c + d * complex.I + e + f * complex.I = 2 - 2 * complex.I

theorem sum_d_f_eq_neg4 : d + f = -4 :=
by
  -- We need the proof for this, hence we use sorry
  sorry

end sum_d_f_eq_neg4_l279_279961


namespace polyhedron_face_not_divisible_by_n_l279_279224

theorem polyhedron_face_not_divisible_by_n {n : ℕ} (hn : n ≥ 3)
  (polyhedron : Type) [has_vertices polyhedron] [has_edges polyhedron] [has_faces polyhedron]
  (odd_vertices_condition : ∃ (v₁ v₂ : vertex polyhedron), v₁ ≠ v₂ ∧ adjacent v₁ v₂ ∧ 
    odd (vertex_degree v₁) ∧ odd (vertex_degree v₂)) :
  ∃ (f : face polyhedron), ¬(face_sides f % n = 0) :=
sorry

end polyhedron_face_not_divisible_by_n_l279_279224


namespace zenon_minor_axis_distance_l279_279206

theorem zenon_minor_axis_distance :
  ∀ (a c : ℝ), a = 9 ∧ c = 6 →
  ∃ b : ℝ, b = 3 * Real.sqrt 5 ∧ 
  Real.sqrt (b^2 + c^2) = 9 :=
by
  intros a c h
  rcases h with ⟨ha, hc⟩
  use 3 * Real.sqrt 5
  split
  · refl
  · dsimp
    rw [ha, hc]
    sorry

end zenon_minor_axis_distance_l279_279206


namespace handshake_remainder_l279_279088

theorem handshake_remainder :
  let n := 12
  let H := 3
  let total_handshakes := 
    let one_ring_handshakes := factorial (n - 1) / 2
    let two_rings_handshakes := (choose n 6) * (factorial (6 - 1) / 2) * (factorial (6 - 1) / 2)
    one_ring_handshakes + two_rings_handshakes
  total_handshakes % 1000 = 960 := by
  sorry

end handshake_remainder_l279_279088


namespace sum_of_signed_segments_is_zero_l279_279103

-- Define the conditions: a circle, a point A, and a closed polygonal line with tangential segments.
variables (circle : Type) (A : Point)

-- Define a concept of signed segments where each segment can be positive or negative.
def tangent_segment (circle : Type) (A : Point) : signed_segment := sorry

-- Define the proof statement: the sum of the signed tangent segments is zero.
theorem sum_of_signed_segments_is_zero (circle : Type) (A : Point) 
  (segments : list (tangent_segment circle A)) : 
  list.sum (segments.map (λ s, s.signed_length)) = 0 :=
sorry

end sum_of_signed_segments_is_zero_l279_279103


namespace hexagon_area_l279_279108

-- Definition of an equilateral triangle with a given perimeter.
def is_equilateral_triangle (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] :=
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = 42 ∧ ∀ (angle : ℝ), angle = 60

-- Statement of the problem
theorem hexagon_area (P Q R P' Q' R' : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace P'] [MetricSpace Q'] [MetricSpace R']
  (h1 : is_equilateral_triangle P Q R) :
  ∃ (area : ℝ), area = 49 * Real.sqrt 3 := 
sorry

end hexagon_area_l279_279108


namespace no_real_solution_for_pairs_l279_279808

theorem no_real_solution_for_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬ (1 / a + 1 / b = 1 / (a + b)) :=
by
  sorry

end no_real_solution_for_pairs_l279_279808


namespace positive_difference_is_127_div_8_l279_279233

-- Defining the basic expressions
def eight_squared : ℕ := 8 ^ 2 -- 64

noncomputable def expr1 : ℝ := (eight_squared + eight_squared) / 8
noncomputable def expr2 : ℝ := (eight_squared / eight_squared) / 8

-- Problem statement
theorem positive_difference_is_127_div_8 :
  (expr1 - expr2) = 127 / 8 :=
by
  sorry

end positive_difference_is_127_div_8_l279_279233


namespace hundredth_power_remainders_l279_279991

theorem hundredth_power_remainders (a : ℤ) : 
  (a % 5 = 0 → a^100 % 125 = 0) ∧ (a % 5 ≠ 0 → a^100 % 125 = 1) :=
by
  sorry

end hundredth_power_remainders_l279_279991


namespace equilateral_triangle_area_ratio_l279_279302

theorem equilateral_triangle_area_ratio :
  let s := 12
      t := 4
      area_equilateral (side : ℝ) := (real.sqrt 3 / 4) * side^2
  in (area_equilateral t) / (area_equilateral s - 2 * area_equilateral t) = 1 / 7 :=
by
  let s := 12
  let t := 4
  let area_equilateral (side : ℝ) := (real.sqrt 3 / 4) * side^2
  have h1 : area_equilateral s = 36 * real.sqrt 3 := sorry
  have h2 : area_equilateral t = 4 * real.sqrt 3 := sorry
  have h3 : 2 * area_equilateral t = 8 * real.sqrt 3 := sorry
  have h4 : area_equilateral s - 2 * area_equilateral t = 28 * real.sqrt 3 := sorry
  show (area_equilateral t) / (area_equilateral s - 2 * area_equilateral t) = 1 / 7,
  from sorry

end equilateral_triangle_area_ratio_l279_279302


namespace jakes_weight_l279_279815

theorem jakes_weight (J S B : ℝ) 
  (h1 : 0.8 * J = 2 * S)
  (h2 : J + S = 168)
  (h3 : B = 1.25 * (J + S))
  (h4 : J + S + B = 221) : 
  J = 120 :=
by
  sorry

end jakes_weight_l279_279815


namespace average_rate_of_change_is_4_l279_279042

def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change_is_4 : 
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_is_4_l279_279042


namespace solve_logarithmic_equation_l279_279726

theorem solve_logarithmic_equation (x : ℝ) (h : log 10 (x * (x - 3)) = 1) : x = 5 :=
sorry

end solve_logarithmic_equation_l279_279726


namespace count_incorrect_propositions_l279_279665

def is_proper_subset {α : Type*} (A B : set α) : Prop :=
A ⊂ B

def is_subset {α : Type*} (A B : set α) : Prop :=
A ⊆ B

def count_elements {α : Type*} (s : set α) : ℕ :=
s.to_finset.card

theorem count_incorrect_propositions {α : Type*} (A B : set α) :
  (is_proper_subset A B → B.nonempty) ∧
  ¬ (is_subset A B → count_elements A < count_elements B) ∧
  (is_subset A B → count_elements A ≤ count_elements B) ∧
  ¬ (is_subset A B → A ≠ B) →
  2 = 2 :=
by
  sorry

end count_incorrect_propositions_l279_279665


namespace Trevor_future_age_when_brother_is_three_times_now_l279_279219

def Trevor_current_age := 11
def Brother_current_age := 20

theorem Trevor_future_age_when_brother_is_three_times_now :
  ∃ (X : ℕ), Brother_current_age + (X - Trevor_current_age) = 3 * Trevor_current_age :=
by
  use 24
  sorry

end Trevor_future_age_when_brother_is_three_times_now_l279_279219


namespace prove_inequality_l279_279932

noncomputable def f : ℝ → ℝ 
| x => if x ≤ 0 then 2^(-x) - 1 else sqrt x

theorem prove_inequality (x : ℝ) : f x > 1 → (x < -1 ∨ x > 1) :=
by
  sorry

end prove_inequality_l279_279932


namespace octagon_area_in_square_l279_279304

theorem octagon_area_in_square (perimeter_s : ℝ) (h1 : perimeter_s = 80) :
  let side_s := perimeter_s / 4
  let bisection_length := side_s / 2
  let area_square := side_s * side_s
  let area_triangle := (1/2) * bisection_length * bisection_length
  let total_area_removed := 4 * area_triangle
  let area_octagon := area_square - total_area_removed
  area_octagon = 200 :=
by
  have side_s := perimeter_s / 4
  have bisection_length := side_s / 2
  have area_square := side_s * side_s
  have area_triangle := (1/2) * bisection_length * bisection_length
  have total_area_removed := 4 * area_triangle
  have area_octagon := area_square - total_area_removed
  calc 
    area_octagon
      = area_square - total_area_removed : by sorry
    ... = 200 : by sorry

end octagon_area_in_square_l279_279304


namespace limit_derivative_at_3_l279_279372

def f (x : ℝ) := -x^2

theorem limit_derivative_at_3 : 
  (tendsto (λ Δx : ℝ, (f (3 + Δx) - f 3) / Δx) (𝓝 0) (𝓝 (-6))) :=
by
  sorry

end limit_derivative_at_3_l279_279372


namespace vector_MN_l279_279382

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

theorem vector_MN :
  vector_sub N M = (-2, -4) :=
by
  sorry

end vector_MN_l279_279382


namespace minimum_value_am_and_ratio_l279_279737

variable (m : ℕ) (a b : ℝ)
variable (h_m : m ≥ 3)
variable (h_a : a > 0) (h_b : b > 0)
variable (h_a1_eq_r : a = b)
variable (h_b1_eq_d : b = a)
variable (h_am_eq_bm : a + (m-1)*b = b * a^(m-1))

theorem minimum_value_am_and_ratio (m : ℕ) (h_m : m ≥ 3) (a b : ℝ)
  (h_a : a > 0) (h_b : b > 0) (h_a1_eq_r : a = b) (h_b1_eq_d : b = a)
  (h_am_eq_bm : a + (m-1)*b = b * a^(m-1)) :
  (a + (m-1) * b = (a / (a^(m-1) - (m-1))) * a^(m-1) ->
    (a + (m-1) * b = (m^(m) / (m-1)^(m-2))^(1/(m-1)) ∧ a / b = (m-1)^(2)) := 
by
  sorry

end minimum_value_am_and_ratio_l279_279737


namespace biased_coin_probability_l279_279993

theorem biased_coin_probability (h : ℚ) 
  (h_cond : 15 * h * (1 - h) = 20 * h^2) 
  (p := (46104 / 281365)) : 
  h = 3 / 7 ∧ ∀ h = 3/7, 4 out of 6 flips is (240 / 1453) and 240 + 1453 = 1693 := 
sorry

end biased_coin_probability_l279_279993


namespace sales_price_calculation_l279_279209

variables (C S : ℝ)
def gross_profit := 1.25 * C
def gross_profit_value := 30

theorem sales_price_calculation 
  (h1: gross_profit C = 30) :
  S = 54 :=
sorry

end sales_price_calculation_l279_279209


namespace frequency_tends_to_probability_l279_279996

/-- Given mathematical definitions and properties of probability, 
    prove that the first statement is incorrect. -/
theorem frequency_tends_to_probability :
  ¬(∀ n : ℕ, ∀ freq : ℝ,
       random freq →
       (as_num_experiments_tends_to_infinity freq n → frequency_tends_to_probability freq n)) := 
sorry

end frequency_tends_to_probability_l279_279996


namespace probability_prime_or_perfect_square_sum_l279_279968

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def outcomes := { (a, b) | a ∈ finset.range 1 7 ∧ b ∈ finset.range 1 7 }

def favorable_outcomes := { (a, b) | (a + b) ∈ finset.range 2 13 ∧ (is_prime (a + b) ∨ is_perfect_square (a + b)) }

theorem probability_prime_or_perfect_square_sum :
  (favorable_outcomes.to_finset.card : ℚ) / (outcomes.to_finset.card : ℚ) = 11 / 18 :=
sorry

end probability_prime_or_perfect_square_sum_l279_279968


namespace max_value_expression_l279_279494

theorem max_value_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x + y + z = 2) :
  (frac x^2 y / (x + y) + frac y^2 z / (y + z) + frac z^2 x / (z + x)) ≤ 1 :=
sorry

end max_value_expression_l279_279494


namespace value_divided_by_is_three_l279_279645

theorem value_divided_by_is_three (x : ℝ) (h : 72 / x = 24) : x = 3 := 
by
  sorry

end value_divided_by_is_three_l279_279645


namespace impossible_to_arrange_seven_non_neg_integers_in_circle_l279_279112

theorem impossible_to_arrange_seven_non_neg_integers_in_circle : ¬ ∃ (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) (f : ℕ) (g : ℕ),
  let x := [a, b, c, d, e, f, g] in
  (cardinality x = 7) ∧
  (∃ (s : FinSeqOfFin 7 (Fin 8)), 
    s 0 = a + b + c ∧ 
    s 1 = b + c + d ∧ 
    s 2 = c + d + e ∧ 
    s 3 = d + e + f ∧ 
    s 4 = e + f + g ∧ 
    s 5 = f + g + a ∧ 
    s 6 = g + a + b ∧ 
    ∀ i, s i = i + 1) := sorry

end impossible_to_arrange_seven_non_neg_integers_in_circle_l279_279112


namespace compound_interest_equivalence_l279_279575

-- Definitions from conditions:
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100
def compound_interest (P R n : ℝ) : ℝ := P * ((1 + R / 100)^n - 1)

-- Problem statement
theorem compound_interest_equivalence :
  simple_interest 1750 8 3 = 420 ∧
  ∀ n : ℝ, compound_interest 4000 10 n = 2 * 420 → n = 2 :=
by
  sorry

end compound_interest_equivalence_l279_279575


namespace evaluate_expression_l279_279703

theorem evaluate_expression :
  (3 ^ (1 ^ (0 ^ 8)) + ( (3 ^ 1) ^ 0 ) ^ 8) = 4 :=
by
  sorry

end evaluate_expression_l279_279703


namespace dividend_calculation_l279_279973

theorem dividend_calculation :
  ∀ (divisor quotient remainder : ℝ), 
  divisor = 37.2 → 
  quotient = 14.61 → 
  remainder = 0.67 → 
  (divisor * quotient + remainder) = 544.042 :=
by
  intros divisor quotient remainder h_div h_qt h_rm
  sorry

end dividend_calculation_l279_279973


namespace compare_sine_values_1_compare_sine_values_2_l279_279321

theorem compare_sine_values_1 (h1 : Real.sin(-Real.pi / 10) = -Real.sin(Real.pi / 10))
                           (h2 : Real.sin(-Real.pi / 8) = -Real.sin(Real.pi / 8))
                           (h3 : 0 < Real.pi / 10 ∧ Real.pi / 10 < Real.pi / 8 ∧ Real.pi / 8 < Real.pi / 2) :
                           Real.sin(-Real.pi / 10) > Real.sin(-Real.pi / 8) :=
sorry

theorem compare_sine_values_2 (h4 : Real.sin(7 * Real.pi / 8) = Real.sin(Real.pi - Real.pi / 8) ∧ Real.sin(Real.pi - Real.pi / 8) = Real.sin(Real.pi / 8))
                           (h5 : Real.sin(5 * Real.pi / 8) = Real.sin(Real.pi - 3 * Real.pi / 8) ∧ Real.sin(Real.pi - 3 * Real.pi / 8) = Real.sin(3 * Real.pi / 8))
                           (h6 : 0 < Real.pi / 8 ∧ Real.pi / 8 < 3 * Real.pi / 8 ∧ 3 * Real.pi / 8 < Real.pi / 2) :
                           Real.sin(7 * Real.pi / 8) < Real.sin(5 * Real.pi / 8) :=
sorry

end compare_sine_values_1_compare_sine_values_2_l279_279321


namespace range_of_omega_l279_279784

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ zeros : ℝ, (f(x) = cos (ω * x) - 1) and (count_zeros (f(x),  [0, 2 * π]) = 3)) ↔ (2 ≤ ω ∧ ω < 3) := 
sorry

end range_of_omega_l279_279784


namespace extend_line_segment_l279_279348

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B Q : V) (x y : ℝ)

theorem extend_line_segment
  (h_ratio : (Q - A) = 7 • (Q - B) / 2) :
  Q = - (2 / 5) • A + B :=
begin
  sorry
end

end extend_line_segment_l279_279348


namespace simplify_fraction_l279_279900

def gcd (a b : ℕ) : ℕ := nat.gcd a b

theorem simplify_fraction : (180 = 2^2 * 3^2 * 5) ∧ (270 = 2 * 3^3 * 5) ∧ (gcd 180 270 = 90) →
  180 / 270 = 2 / 3 :=
by
  intro h
  cases h with h1 h2h3
  cases h2h3 with h2 h3
  sorry -- Proof is omitted

end simplify_fraction_l279_279900


namespace triangle_is_isosceles_l279_279821

-- lean statement
theorem triangle_is_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) : 
  ∃ k : ℝ, a = k ∧ b = k := 
sorry

end triangle_is_isosceles_l279_279821


namespace john_acclimation_time_l279_279464

variable (A B R D : ℝ)

theorem john_acclimation_time :
  B = 2 ∧ R = 1.75 * B ∧ D = 0.5 * A ∧ A + B + R + D = 7 → A = 1 :=
by
  intro h
  cases h with hB h1
  cases h1 with hR h2
  cases h2 with hD hsum
  rw [hB, hR, hD] at hsum
  have hA : 1.5 * A + 5.5 = 7 := by linarith
  have A_eq : 1.5 * A = 1.5 := by linarith
  have A_val : A = 1 := by linarith
  exact A_val

-- this theorem now checks the correctness of the given conditions.

end john_acclimation_time_l279_279464


namespace number_of_new_students_l279_279549

variable (O N : ℕ)
variable (H1 : 48 * O + 32 * N = 44 * 160)
variable (H2 : O + N = 160)

theorem number_of_new_students : N = 40 := sorry

end number_of_new_students_l279_279549


namespace tangent_acute_angle_l279_279381

noncomputable def prob_tangent_acute_angle (b : ℝ) (h : b ∈ set.Icc (-1 : ℝ) 5) : ℝ :=
  if b > 1 then (5 - 1) / (5 - (-1)) else 0

theorem tangent_acute_angle (b : ℝ) (h : b ∈ set.Icc (-1 : ℝ) 5) :
  prob_tangent_acute_angle b h = 2 / 3 :=
sorry

end tangent_acute_angle_l279_279381


namespace min_spend_on_boxes_l279_279613

theorem min_spend_on_boxes : 
  let box_vol := 18 * 22 * 15
  let usable_vol := box_vol * 0.80
  let big_item1_vol := 72 * 66 * 45
  let big_item2_vol := 54 * 48 * 40
  let big_item3_vol := 36 * 77 * 60
  let total_big_items_vol := big_item1_vol + big_item2_vol + big_item3_vol
  let num_boxes := total_big_items_vol / usable_vol
  let rounded_num_boxes := nat.ceil num_boxes
  let cost := 100 * 0.60 + (rounded_num_boxes - 100) * 0.55
  cost = 61.10 := by
  have h1 : rounded_num_boxes = 102 := sorry
  have h2 : cost = 60 + 1.10 := sorry
  show cost = 61.10, from h2
  sorry

end min_spend_on_boxes_l279_279613


namespace volume_parallelepiped_l279_279032

variables (u v w : ℝ^3)
variable (volume₀ : ℝ)
variable (h : |u ⋅ (v × w)| = 6)

def volume (a b c : ℝ^3) : ℝ := |a ⋅ (b × c)|

theorem volume_parallelepiped : volume (u + 2 • v) (3 • v + 2 • w) (w - 4 • u) = 30 :=
by
  sorry

end volume_parallelepiped_l279_279032


namespace final_movie_ends_at_7_55_pm_l279_279912

variable (start_time : Nat) (first_movie_duration : Nat) (first_break_duration : Nat)
          (second_movie_duration : Nat) (second_break_duration : Nat) (third_movie_duration : Nat)

def time_after_minutes (start time elapsed : Nat) : Nat :=
  let total_minutes := start + time
  total_minutes

def time_after_merge_hours_and_minutes (time_in_hours hours : Nat) (minutes : Nat) : Nat :=
  let total_min := time_in_hours + (hours * 60) + minutes
  total_min

theorem final_movie_ends_at_7_55_pm :
  time_after_merge_hours_and_minutes (time_after_minutes 60 (start_time, first_movie_duration))
  	first_break_duration second_movie_duration second_break_duration third_movie_duration = 715
        := by
  sorry

end final_movie_ends_at_7_55_pm_l279_279912


namespace number_of_valid_subsets_l279_279568

-- Define the universal set and subset conditions
def universal_set := {1, 2, 3}

def is_valid_subset (B : Set ℕ) : Prop :=
  {1, 2} ⊆ B ∧ B ⊆ universal_set

-- Define the problem statement to prove the number of such sets
theorem number_of_valid_subsets : 
  (Finset.filter (is_valid_subset) 
    (Finset.powerset universal_set.toFinset)).card = 2 := 
by 
  sorry

end number_of_valid_subsets_l279_279568


namespace train_speed_is_45_kmph_l279_279287

noncomputable def speed_of_train_kmph (train_length bridge_length total_time : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / total_time
  let speed_kmph := speed_mps * 36 / 10
  speed_kmph

theorem train_speed_is_45_kmph :
  speed_of_train_kmph 150 225 30 = 45 :=
  sorry

end train_speed_is_45_kmph_l279_279287


namespace Iain_pennies_left_l279_279069

theorem Iain_pennies_left (initial_pennies older_pennies : ℕ) (percentage : ℝ)
  (h_initial : initial_pennies = 200)
  (h_older : older_pennies = 30)
  (h_percentage : percentage = 0.20) :
  initial_pennies - older_pennies - (percentage * (initial_pennies - older_pennies)) = 136 :=
by
  sorry

end Iain_pennies_left_l279_279069


namespace rational_ratios_l279_279474

theorem rational_ratios (S : Set ℝ) (hS : S.card = 5) 
  (h : ∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → c ≠ a → (a * b + b * c + c * a) ∈ ℚ) :
  ∀ (a b : ℝ), a ∈ S → b ∈ S → (a / b) ∈ ℚ :=
by
  intro a b ha hb
  sorry

end rational_ratios_l279_279474


namespace total_candidates_l279_279447

theorem total_candidates (C B : ℕ) 
  (h1 : B = C - 900)
  (h2 : 0.62 * B + 0.68 * 900 = 0.647 * C) : C = 2000 :=
by
  sorry

end total_candidates_l279_279447


namespace min_value_inequality_l279_279132

theorem min_value_inequality (b : Fin 8 → ℝ) (h1 : ∀ i, b i > 0) (h2 : (∑ i, b i) = 2) :
  (∑ i, 1 / (b i)) ≥ 32 := 
sorry

end min_value_inequality_l279_279132


namespace constant_term_expansion_l279_279336

theorem constant_term_expansion : 
  let expr := (λ x : ℝ, (x - 1 / x) * (x + 3 / x)^3) in
  ∀ x : ℝ, x ≠ 0 → constant_term (expr x) = -18 :=
by
  intro expr x h 
  let term := constant_term (expr x)
  have h1 : term = -18
  sorry

end constant_term_expansion_l279_279336


namespace solution_l279_279139

theorem solution (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000 * x = y^2 - 2000 * y) : 
  x + y = 2000 := 
by 
  sorry

end solution_l279_279139


namespace Sara_has_3194_quarters_in_the_end_l279_279528

theorem Sara_has_3194_quarters_in_the_end
  (initial_quarters : ℕ)
  (borrowed_quarters : ℕ)
  (initial_quarters_eq : initial_quarters = 4937)
  (borrowed_quarters_eq : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 := by
  sorry

end Sara_has_3194_quarters_in_the_end_l279_279528


namespace parallel_line_through_intersection_point_perpendicular_line_through_intersection_point_l279_279051

theorem parallel_line_through_intersection_point :
  let P := (1, 3)
  let is_solution := ∀ (x y : ℝ), y = -x + 4 → y = x + 2 → x = 1 ∧ y = 3
  let line_parallel := ∀ (x y : ℝ), 2*x - y + 1 = 0 → (x, y) = P
  is_solution 1 3 ∧ line_parallel 1 3 :=
by
  sorry

theorem perpendicular_line_through_intersection_point :
  let P := (1, 3)
  let is_solution := ∀ (x y : ℝ), y = -x + 4 → y = x + 2 → x = 1 ∧ y = 3
  let line_perpendicular := ∀ (x y : ℝ), x + 2*y - 7 = 0 → (x, y) = P
  is_solution 1 3 ∧ line_perpendicular 1 3 :=
by
  sorry

end parallel_line_through_intersection_point_perpendicular_line_through_intersection_point_l279_279051


namespace a_2_value_l279_279368

theorem a_2_value :
  let f := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8 + (1 + x)^9 + (1 + x)^10,
  f = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9 + a₁₀ * x^10
  by let a₂ := ∑ i in FinSet.range (11), if i ≥ 2 then (i - 1).choose 2 else 0,
  a₂ = 165 :=
by 
  sorry

end a_2_value_l279_279368


namespace gasohol_problem_l279_279637

noncomputable def initial_gasohol_volume (x : ℝ) : Prop :=
  let ethanol_in_initial_mix := 0.05 * x
  let ethanol_to_add := 2
  let total_ethanol := ethanol_in_initial_mix + ethanol_to_add
  let total_volume := x + 2
  0.1 * total_volume = total_ethanol

theorem gasohol_problem (x : ℝ) : initial_gasohol_volume x → x = 36 := by
  intro h
  sorry

end gasohol_problem_l279_279637


namespace arithmetic_progression_sum_l279_279394

theorem arithmetic_progression_sum (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : a 2 = 3) (h3 : d = 2) :
  (∑ n in Finset.range 9, 1 / ((a n) * (a (n + 1)))) = 9 / 19 :=
by
  sorry

end arithmetic_progression_sum_l279_279394


namespace number_chosen_div_8_sub_100_eq_6_l279_279281

variable (n : ℤ)

theorem number_chosen_div_8_sub_100_eq_6 (h : (n / 8) - 100 = 6) : n = 848 := 
by
  sorry

end number_chosen_div_8_sub_100_eq_6_l279_279281


namespace largest_possible_last_digit_l279_279559

theorem largest_possible_last_digit 
  (s : String) 
  (h_len : s.length = 3003) 
  (h_first : s.get 0 = '2') 
  (h_div : ∀ i, 0 ≤ i ∧ i < 3002 → (s.get i).to_nat * 10 + (s.get (i + 1)).to_nat ∣ 23 ∨ (s.get i).to_nat * 10 + (s.get (i + 1)).to_nat ∣ 29) : 
  s.get 3002 = '6' :=
sorry

end largest_possible_last_digit_l279_279559


namespace determinant_of_sum_of_columns_l279_279538

variables (a b c : ℝ × ℝ × ℝ)

def D : ℝ :=
(a.1 * (b.2 * c.3 - b.3 * c.2)) -
(a.2 * (b.1 * c.3 - b.3 * c.1)) +
(a.3 * (b.1 * c.2 - b.2 * c.1))

theorem determinant_of_sum_of_columns :
  let ab := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
  let bc := (b.1 + c.1, b.2 + c.2, b.3 + c.3)
  let ca := (c.1 + a.1, c.2 + a.2, c.3 + a.3)
  in ((ab.1 * (bc.2 * ca.3 - bc.3 * ca.2)) -
      (ab.2 * (bc.1 * ca.3 - bc.3 * ca.1)) +
      (ab.3 * (bc.1 * ca.2 - bc.2 * ca.1))
     ) = 2 * D a b c :=
by
  sorry

end determinant_of_sum_of_columns_l279_279538


namespace equivalent_statements_l279_279623

-- Definitions based on the problem
def is_not_negative (x : ℝ) : Prop := x >= 0
def is_not_positive (x : ℝ) : Prop := x <= 0
def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0

-- The main theorem statement
theorem equivalent_statements (x : ℝ) : 
  (is_not_negative x → is_not_positive (x^2)) ↔ (is_positive (x^2) → is_negative x) :=
by
  sorry

end equivalent_statements_l279_279623


namespace card_arrangements_with_removal_l279_279895

theorem card_arrangements_with_removal :
  let cards := {1, 2, 3, 4, 5, 6, 7} in
  let count_ways := λ n k : ℕ, (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k)) in
  let factorial := λ n : ℕ, Nat.factorial n in
  count_ways 7 2 * factorial 5 = 42 :=
by sorry

end card_arrangements_with_removal_l279_279895


namespace find_pointA_coordinates_l279_279836

-- Define point B
def pointB : ℝ × ℝ := (4, -1)

-- Define the symmetry condition with respect to the x-axis
def symmetricWithRespectToXAxis (p₁ p₂ : ℝ × ℝ) : Prop :=
  p₁.1 = p₂.1 ∧ p₁.2 = -p₂.2

-- Theorem statement: Prove the coordinates of point A given the conditions
theorem find_pointA_coordinates :
  ∃ A : ℝ × ℝ, symmetricWithRespectToXAxis pointB A ∧ A = (4, 1) :=
by
  sorry

end find_pointA_coordinates_l279_279836


namespace oakwood_team_combinations_l279_279195

theorem oakwood_team_combinations :
  let total_girls := 5
      total_boys := 7
      required_girls := 2
      required_boys := 2 in
  (∃ Judy (girls : Finset ℕ) (boys : Finset ℕ),
    Judy ∈ girls ∧
    Fintype.card girls = total_girls ∧
    Fintype.card boys = total_boys ∧
    ∃ teams : Finset (Finset ℕ),
    (∀ team ∈ teams, Judy ∈ team ∧
    Fintype.card team.filter (λ member, member ∈ girls) = required_girls ∧
    Fintype.card team.filter (λ member, member ∈ boys) = required_boys) ∧
    Fintype.card teams = 84) := sorry

end oakwood_team_combinations_l279_279195


namespace angle_D_in_pentagon_l279_279448

theorem angle_D_in_pentagon (A B C D E : ℝ) 
  (h1 : A = B) (h2 : B = C) (h3 : D = E) (h4 : A + 40 = D) 
  (h5 : A + B + C + D + E = 540) : D = 132 :=
by
  -- Add proof here if needed
  sorry

end angle_D_in_pentagon_l279_279448


namespace kindergarten_students_present_l279_279509

theorem kindergarten_students_present
  (morning_registered : ℕ) (morning_absent : ℕ)
  (early_afternoon_registered : ℕ) (early_afternoon_absent : ℕ)
  (late_afternoon_registered : ℕ) (late_afternoon_absent : ℕ)
  (early_evening_registered : ℕ) (early_evening_absent : ℕ)
  (late_evening_registered : ℕ) (late_evening_absent : ℕ)
  (transferred_out : ℕ) (new_registrations_attending : ℕ) :
  morning_registered = 75 ∧ morning_absent = 9 ∧
  early_afternoon_registered = 72 ∧ early_afternoon_absent = 12 ∧
  late_afternoon_registered = 90 ∧ late_afternoon_absent = 15 ∧
  early_evening_registered = 50 ∧ early_evening_absent = 6 ∧
  late_evening_registered = 60 ∧ late_evening_absent = 10 ∧
  transferred_out = 3 ∧ new_registrations_attending = 1 →
  let total_present :=
        (morning_registered - morning_absent) +
        (early_afternoon_registered - early_afternoon_absent) +
        (late_afternoon_registered - late_afternoon_absent) +
        (early_evening_registered - early_evening_absent) +
        (late_evening_registered - late_evening_absent)
  in total_present - transferred_out + new_registrations_attending = 293 :=
by
  sorry

end kindergarten_students_present_l279_279509


namespace proportion_equality_l279_279838

section GeometricProblem

variables {E F G H A B C D : Type*}
variables [ConvexQuadrilateral E F G H A B C D]
variables [ConvexQuadrilateral H1_E1 E1_F1 F1_G1 G1_H1 E1 F1 G1 H1]

variables (AE EB BF FC CG GD DH HA : Real)
variables (h1 : AE / EB * BF / FC * CG / GD * DH / HA = 1)

variables (E1A AH1 F1C CG1 : Real)
variables (h2 : Parallel E1F1 EF)
variables (h3 : Parallel F1G1 FG)
variables (h4 : Parallel G1H1 GH)
variables (h5 : Parallel H1E1 HE)

theorem proportion_equality : (F1C / CG1) = (E1A / AH1) :=
sorry

end GeometricProblem

end proportion_equality_l279_279838


namespace gerry_money_l279_279640

theorem gerry_money (apples_cost : ℕ) (emmy_money : ℕ) (total_apples : ℕ) (emmy_apples : ℕ) (gerry_apples : ℕ) : 
  apples_cost = 2 → 
  emmy_money = 200 → 
  total_apples = 150 → 
  emmy_apples = emmy_money / apples_cost →
  gerry_apples = total_apples - emmy_apples → 
  gerry_apples * apples_cost = 100 := 
by
  intros hc hm ht he hg
  rw hc at *
  rw hm at *
  rw ht at *
  rw he at *
  rw hg at *
  linarith

end gerry_money_l279_279640


namespace flowerbed_seeds_l279_279884

theorem flowerbed_seeds (n_fbeds n_seeds_per_fbed total_seeds : ℕ)
    (h1 : n_fbeds = 8)
    (h2 : n_seeds_per_fbed = 4) :
    total_seeds = n_fbeds * n_seeds_per_fbed := by
  sorry

end flowerbed_seeds_l279_279884


namespace betty_needs_more_flies_l279_279361

def flies_per_day := 2
def days_per_week := 7
def flies_needed_per_week := flies_per_day * days_per_week

def flies_caught_morning := 5
def flies_caught_afternoon := 6
def fly_escaped := 1

def flies_caught_total := flies_caught_morning + flies_caught_afternoon - fly_escaped

theorem betty_needs_more_flies : 
  flies_needed_per_week - flies_caught_total = 4 := by
  sorry

end betty_needs_more_flies_l279_279361


namespace find_f2_l279_279561

-- Define the function f and the condition it satisfies
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def condition : Prop := ∀ x, x ≠ 1 / 3 → f x + f ((x + 1) / (1 - 3 * x)) = x

-- State the theorem to prove the value of f(2)
theorem find_f2 (h : condition f) : f 2 = 48 / 35 := 
by
  sorry

end find_f2_l279_279561


namespace find_AB_approximation_l279_279823

theorem find_AB_approximation 
  (A B C D E F : Type) 
  [Triangle A B C] 
  [Triangle D E F] 
  (h_angle_A : ∠ A = 90)
  (h_tan_C : tan(∠ C) = 8) 
  (h_BC : BC = 150) 
  (h_similar : Similar_Triangles (Triangle ABC) (Triangle DEF))
  (h_angle_D : ∠ D = 90) 
  (h_EF : EF = 300) : 
  AB = 149 :=
by
  sorry

end find_AB_approximation_l279_279823


namespace area_ratio_of_extended_equilateral_triangle_l279_279127

theorem area_ratio_of_extended_equilateral_triangle
  (A B C A' B' C' : Type*)
  [equilateral_triangle ABC]
  [line_extension AB B' 2]
  [line_extension BC C' 4]
  [line_extension CA A' 3] :
  area A'B'C' / area ABC = 16 :=
sorry

end area_ratio_of_extended_equilateral_triangle_l279_279127


namespace repeating_decimal_to_fraction_l279_279331

-- Define the repeating decimal
def x := 0.137137137 -- repeating decimal

-- Define the equivalent fraction form
def frac := 137 / 999

-- State the theorem
theorem repeating_decimal_to_fraction : x = frac :=
sorry

end repeating_decimal_to_fraction_l279_279331


namespace part1_part2_l279_279865

variables {x y : ℝ}
def a := (2, x)
def b := (y, 1)
def c := (3, -3)
def a_dot_b_eq_zero := (2 * y + x = 0)
def b_parallel_c := (∃ k : ℝ, b = (k * 3, k * -3))

-- Define vector addition
def vector_add (u v : ℝ × ℝ) := (u.1 + v.1, u.2 + v.2)

-- Define the squared magnitude of a vector
def vector_magnitude_sq (u : ℝ × ℝ) := u.1^2 + u.2^2

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- Problem (1): Prove that |a + b| = sqrt 10
theorem part1 (hx : a_dot_b_eq_zero) (hy : b_parallel_c) : 
    real.sqrt (vector_magnitude_sq (vector_add a b)) = real.sqrt 10 := sorry

-- Problem (2): Prove that cos(theta) between (a + b) and (a + 2 * b + c) equals 3/5
theorem part2 (hx : a_dot_b_eq_zero) (hy : b_parallel_c) :
    dot_product (vector_add a b) (vector_add (vector_add a (vector_add b b)) c) / 
    (real.sqrt (vector_magnitude_sq (vector_add a b)) * 
     real.sqrt (vector_magnitude_sq (vector_add (vector_add a (vector_add b b)) c))) = 3/5 := sorry

end part1_part2_l279_279865


namespace balance_difference_l279_279306

variables (P_A P_B n : ℕ) (r_A r_B : ℚ)

noncomputable def angela_balance (P_A : ℕ) (r_A : ℚ) (n : ℕ) : ℚ :=
P_A * (1 + r_A) ^ n

noncomputable def bob_balance (P_B : ℕ) (r_B : ℚ) (n : ℕ) : ℚ :=
P_B * (1 + r_B) ^ n

theorem balance_difference :
    angela_balance 9000 (5 / 100) 25 - bob_balance 10000 (45 / 1000) 25 ≈ 852 := by
  sorry 

end balance_difference_l279_279306


namespace infinite_T_with_two_distinct_d_values_l279_279364

-- Define d_i(T) function
def d_i (i : ℕ) (T : ℕ) : ℕ :=
  -- counts the number of times digit i appears in multiples of 1829 up to T
  sorry

-- The main statement to be proven
theorem infinite_T_with_two_distinct_d_values :
  ∃ᶠ T in at_top, set.card (set.image (λ i, d_i i T) (set.Icc 1 9)) = 2 :=
sorry

end infinite_T_with_two_distinct_d_values_l279_279364


namespace mistake_in_counts_l279_279662

theorem mistake_in_counts (boys_reports : List ℕ) (girls_reports : List ℕ) (total_participants : List ℕ)
  (h_boys : boys_reports = [3, 3, 3, 3, 3, 5])
  (h_girls : girls_reports = [6, 6, 6, 6, 6, 6, 6, 6, 6])
  (h_total : total_participants = [3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]) :
  ∑ b in boys_reports, b + ∑ g in girls_reports, g ≠ 2 * 37 :=
by sorry

end mistake_in_counts_l279_279662


namespace rainfall_third_day_city_A_rainfall_third_day_city_B_combined_third_day_rainfall_total_rainfall_three_days_l279_279221

variable {B1 : ℚ}

def city_A_day1 : ℚ := 4
def city_A_day2 : ℚ := 5 * city_A_day1
def city_B_day2 : ℚ := 3 * city_A_day1

def city_A_day3 : ℚ := 1 / 2 * (city_A_day1 + city_A_day2)
def city_B_day3 : ℚ := B1 + 6

def X : ℚ := city_A_day3 + city_B_day3
def Y : ℚ := (city_A_day1 + city_A_day2 + city_A_day3) + (B1 + city_B_day2 + city_B_day3)

theorem rainfall_third_day_city_A : city_A_day3 = 12 := by
  sorry

theorem rainfall_third_day_city_B : city_B_day3 = B1 + 6 := by
  sorry

theorem combined_third_day_rainfall : X = 18 + B1 := by
  sorry

theorem total_rainfall_three_days : Y = 54 + 2 * B1 := by
  sorry

end rainfall_third_day_city_A_rainfall_third_day_city_B_combined_third_day_rainfall_total_rainfall_three_days_l279_279221


namespace surface_area_increase_l279_279327

structure RectangularSolid (length : ℝ) (width : ℝ) (height : ℝ) where
  surface_area : ℝ := 2 * (length * width + length * height + width * height)

def cube_surface_contributions (side : ℝ) : ℝ := side ^ 2 * 3

theorem surface_area_increase
  (original : RectangularSolid 4 3 5)
  (cube_side : ℝ := 1) :
  let new_cube_contribution := cube_surface_contributions cube_side
  let removed_face : ℝ := cube_side ^ 2
  let original_surface_area := original.surface_area
  original_surface_area + new_cube_contribution - removed_face = original_surface_area + 2 :=
by
  sorry

end surface_area_increase_l279_279327


namespace surface_area_of_circumscribed_sphere_l279_279402

theorem surface_area_of_circumscribed_sphere (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : 
  ∃ S : ℝ, S = 29 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l279_279402


namespace exist_non_quadratic_residues_sum_l279_279499

noncomputable section

def is_quadratic_residue_mod (p a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 ≡ a [ZMOD p]

theorem exist_non_quadratic_residues_sum {p : ℤ} (hp : p > 5) (hp_modeq : p ≡ 1 [ZMOD 4]) (a : ℤ) : 
  ∃ b c : ℤ, a = b + c ∧ ¬is_quadratic_residue_mod p b ∧ ¬is_quadratic_residue_mod p c :=
sorry

end exist_non_quadratic_residues_sum_l279_279499


namespace computation_l279_279864

def g (x : ℕ) : ℕ := 7 * x - 3

theorem computation : g (g (g (g 1))) = 1201 := by
  sorry

end computation_l279_279864


namespace uncolored_area_of_rectangle_l279_279121

theorem uncolored_area_of_rectangle :
  let width := 30
  let length := 50
  let radius := width / 4
  let rectangle_area := width * length
  let circle_area := π * (radius ^ 2)
  let total_circles_area := 4 * circle_area
  rectangle_area - total_circles_area = 1500 - 225 * π := by
  sorry

end uncolored_area_of_rectangle_l279_279121


namespace students_not_in_biology_l279_279619

theorem students_not_in_biology (total_students : ℕ) (enrolled_in_biology_percent : ℝ) (non_enrolled_students_expected : ℕ) :
  total_students = 840 ∧ enrolled_in_biology_percent = 0.35 ∧ non_enrolled_students_expected = 546 →
  ∃ not_enrolled_students, not_enrolled_students = ((1 - enrolled_in_biology_percent) * total_students) ∧ not_enrolled_students = non_enrolled_students_expected :=
begin
  sorry
end

end students_not_in_biology_l279_279619


namespace billy_distance_l279_279311

-- Definitions
def distance_billy_spit (b : ℝ) : ℝ := b
def distance_madison_spit (m : ℝ) (b : ℝ) : Prop := m = 1.20 * b
def distance_ryan_spit (r : ℝ) (m : ℝ) : Prop := r = 0.50 * m

-- Conditions
variables (m : ℝ) (b : ℝ) (r : ℝ)
axiom madison_farther: distance_madison_spit m b
axiom ryan_shorter: distance_ryan_spit r m
axiom ryan_distance: r = 18

-- Proof problem
theorem billy_distance : b = 30 := by
  sorry

end billy_distance_l279_279311


namespace not_monotonic_on_interval_and_range_of_a_l279_279043

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2

theorem not_monotonic_on_interval_and_range_of_a :
  (∀ x ∈ set.Icc 2 4, ∃ a, f x a ∉ set.monotone f) ↔ (3 < a ∧ a < 6) :=
sorry

end not_monotonic_on_interval_and_range_of_a_l279_279043


namespace bisection_interval_next_l279_279971

def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_interval_next (f : ℝ → ℝ) (a b x₀ : ℝ) 
  (h₁ : a = 2) 
  (h₂ : b = 3) 
  (h₃ : x₀ = (a + b) / 2) 
  (h₄ : f a < 0) 
  (h₅ : f b > 0) 
  (h₆ : f x₀ > 0) : 
  (2 : ℝ) < (2.5 : ℝ) ∧ (∀ x, x ∈ set.Icc (2 : ℝ) (2.5 : ℝ) → f x = 0) :=
by
  sorry

end bisection_interval_next_l279_279971


namespace gymnastics_positions_l279_279089

def positions :=
  { n : ℕ // n = 1 } ∧  -- Nina's position is 1
  { g : ℕ // g = n + 1 } ∧  -- Galya’s position is immediately after Nina
  { z : ℕ // n < z } ∧ -- Zina performed worse than Nina
  { v : ℕ // v ≠ 1 ∧ v ≠ 4 } -- Valya did not perform neither the best nor the worst

theorem gymnastics_positions (n z g v : ℕ) (n_cond : n = 1) (g_cond : g = n + 1) (z_cond : n < z) (v_cond : v ≠ 1 ∧ v ≠ 4) :
  n = 1 ∧ g = 2 ∧ v = 3 ∧ z = 4 :=
by {
  sorry
}

end gymnastics_positions_l279_279089


namespace kate_jenna_sticker_ratio_l279_279467

theorem kate_jenna_sticker_ratio :
  let k := 21
  let j := 12
  Nat.gcd k j = 3 ∧ k / Nat.gcd k j = 7 ∧ j / Nat.gcd k j = 4 :=
by
  let k := 21
  let j := 12
  have g : Nat.gcd k j = 3 := by sorry
  have hr : k / Nat.gcd k j = 7 := by sorry
  have jr : j / Nat.gcd k j = 4 := by sorry
  exact ⟨g, hr, jr⟩

end kate_jenna_sticker_ratio_l279_279467


namespace inequality_solution_set_l279_279169

theorem inequality_solution_set :
  {x : ℝ | (x^2 - 4) / (x^2 - 9) > 0} = {x : ℝ | x < -3 ∨ x > 3} :=
sorry

end inequality_solution_set_l279_279169


namespace function_inequality_l279_279696

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative_inequality : ∀ x : ℝ, (deriv f (2 * x)) > (real.log 2 / 2) * (f (2 * x))

theorem function_inequality :
  f 2 > 2 * f 0 ∧ 2 * f 0 > 4 * f (-2) :=
sorry

end function_inequality_l279_279696


namespace kendalls_total_distance_l279_279469

-- Definitions of the conditions
def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5

-- The theorem to prove the total distance
theorem kendalls_total_distance : distance_with_mother + distance_with_father = 0.67 :=
by
  sorry

end kendalls_total_distance_l279_279469


namespace circumscribed_circle_radius_l279_279544

theorem circumscribed_circle_radius (A : ℝ) (R : ℝ) (h : A = 81) : R = 6 * real.sqrt(real.sqrt(3)) :=
by
  sorry

end circumscribed_circle_radius_l279_279544


namespace average_age_of_5_students_l279_279189

theorem average_age_of_5_students
  (avg_age_20_students : ℕ → ℕ → ℕ → ℕ)
  (total_age_20 : avg_age_20_students 20 20 0 = 400)
  (total_age_9 : 9 * 16 = 144)
  (age_20th_student : ℕ := 186) :
  avg_age_20_students 5 ((400 - 144 - 186) / 5) 5 = 14 :=
by
  sorry

end average_age_of_5_students_l279_279189


namespace perimeter_triangle_ABF1_is_8_distance_origin_to_l_is_1_length_AB_is_8_div_3_all_statements_true_l279_279395

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

noncomputable def focus_F2 : ℝ × ℝ :=
  (real.sqrt 2, 0)

noncomputable def line_l (x y : ℝ) : Prop :=
  y = x - real.sqrt 2

-- Proving the three statements as individual theorems
theorem perimeter_triangle_ABF1_is_8 (x₁ x₂ y₁ y₂ : ℝ) (h₁ : ellipse_eq x₁ y₁) (h₂ : ellipse_eq x₂ y₂) (hx_ne : x₁ ≠ x₂) :
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (8 / 3)^2 → sorry

theorem distance_origin_to_l_is_1 : 
  (0 : ℝ) ∈ set.range (λ x : ℝ, line_l 0 0) → sorry

theorem length_AB_is_8_div_3 (x₁ x₂ y₁ y₂ : ℝ) (h₁ : ellipse_eq x₁ y₁) (h₂ : ellipse_eq x₂ y₂) :
  dist (x₁, y₁) (x₂, y₂) = 8 / 3 → sorry

-- Combined theorem to assert all three statements are true
theorem all_statements_true :
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
  ellipse_eq x₁ y₁ ∧ ellipse_eq x₂ y₂ ∧ 
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (8 / 3)^2 ∧
  dist (x₁, y₁) (x₂, y₂) = 8 / 3 ∧
  (0 : ℝ) ∈ set.range (λ x : ℝ, line_l 0 0)) → sorry

end perimeter_triangle_ABF1_is_8_distance_origin_to_l_is_1_length_AB_is_8_div_3_all_statements_true_l279_279395


namespace ladder_height_l279_279608

def cube_edge_length := 2 -- edge length of cube in meters
def ladder_length := 6 -- length of ladder in meters
def high1 := 4.98 -- one possible height of the ladder end in meters
def high2 := 3.34 -- another possible height of the ladder end in meters

theorem ladder_height :
  ∃ h : ℝ, (h = high1 ∨ h = high2) ∧
  (cube_edge_length = 2) ∧ (ladder_length = 6) ∧
  (Exists (λ x y : ℝ, x^2 + 4*x + 4 + (4 / x + 2)^2 = 36)) :=
by sorry

end ladder_height_l279_279608


namespace part1_part2_l279_279385

noncomputable def problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x * y = x + y) : ℝ :=
  2 * x + 4 * y

theorem part1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x * y = x + y) :
  ∃ x y, (x = (1 + Real.sqrt 2) / 2) ∧ (y = (2 + Real.sqrt 2) / 4) ∧ 
  problem1 x y hx hy (by norm_num : 2 * x * y = x + y) = 3 + 2 * Real.sqrt 2 := sorry

theorem part2 (a : ℝ) (ha: a = 1 / 2) (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x * y = x + y + a * (x^2 + y^2)) :
  ∃ s, s ∈ set.Ici (4 : ℝ) ∧ s = x + y := sorry

end part1_part2_l279_279385


namespace min_lambda_half_l279_279730

theorem min_lambda_half (a : ℕ → ℚ) (x : ℕ → ℚ) (n : ℕ) :
  (∀ i j, i ≤ j → a i ≥ a j) →
  a n ≠ a 1 →
  (∑ i in finset.range n, x i) = 0 →
  (∑ i in finset.range n, |x i|) = 1 →
  |∑ i in finset.range n, a i * x i| ≤ (1 / 2) * (a 1 - a n) :=
by
  sorry

end min_lambda_half_l279_279730


namespace max_value_on_interval_l279_279756

   variable (f : ℝ → ℝ)

   -- Conditions
   def odd_function := ∀ x, f (-x) = -f x
   def f1 := f 1 = 2
   def f_increasing := ∀ x y, 0 < x → x < y → f x < f y
   def functional_equation := ∀ x y, f (x + y) = f x + f y

   theorem max_value_on_interval :
     odd_function f ∧
     f1 f ∧
     f_increasing f ∧
     functional_equation f →
     ∃ c ∈ Icc (-3 : ℝ) (-2), f c = -4 :=
   by
     intro h
     sorry
   
end max_value_on_interval_l279_279756


namespace range_of_m_l279_279045

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
    (∀ x1 : ℝ, x1 ∈ set.Icc (-1 : ℝ) (1 : ℝ) → ∃ x2 : ℝ, x2 ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ f x1 = g x2) 
    → (m ≥ (1 / 2) ∧ m ≤ 1) :=
by
  let f := (λ x : ℝ, x^2)
  let g := (λ x : ℝ, (1 / 2) ^ x - m)
  sorry

end range_of_m_l279_279045


namespace probability_of_forming_coldness_l279_279853

theorem probability_of_forming_coldness :
  let cart := "CART".toList
  let blend := "BLEND".toList
  let show := "SHOW".toList
  let prob_cart := 1 / (cart.length.choose 2)
  let prob_blend := 1 / (blend.length.choose 4)
  let prob_show := 2 / (show.length.choose 3)
  prob_cart * prob_blend * prob_show = 1 / 60 :=
by {
  sorry
}

end probability_of_forming_coldness_l279_279853


namespace money_given_to_last_set_l279_279507

theorem money_given_to_last_set (total first second third fourth last : ℝ) 
  (h_total : total = 4500) 
  (h_first : first = 725) 
  (h_second : second = 1100) 
  (h_third : third = 950) 
  (h_fourth : fourth = 815) 
  (h_sum: total = first + second + third + fourth + last) : 
  last = 910 :=
sorry

end money_given_to_last_set_l279_279507


namespace connie_ticket_distribution_l279_279325

noncomputable def connie_tickets := (500 : ℕ)
noncomputable def koala_bear_tickets := (0.20 * connie_tickets).toNat
noncomputable def earbuds_tickets := (30 : ℕ)
noncomputable def car_tickets := (2 * earbuds_tickets)
noncomputable def bracelets_tickets := (0.15 * connie_tickets).toNat
noncomputable def remaining_tickets := connie_tickets - (koala_bear_tickets + earbuds_tickets + car_tickets + bracelets_tickets)
noncomputable def poster_tickets : ℕ := (135 : ℕ)
noncomputable def keychain_tickets : ℕ := (remaining_tickets - poster_tickets)

theorem connie_ticket_distribution :
  koala_bear_tickets = 100 ∧
  earbuds_tickets = 30 ∧
  car_tickets = 60 ∧
  bracelets_tickets = 75 ∧
  poster_tickets = 135 ∧
  keychain_tickets = 100 :=
by
  sorry

end connie_ticket_distribution_l279_279325


namespace lines_disjoint_planes_parallel_l279_279011

theorem lines_disjoint_planes_parallel (l m : Set Point) (α β : Set Point)
  (h₁ : l ∩ m = ∅) (h₂ : l ⊆ α) (h₃ : m ⊆ β) : 
  (∃ (α β : Set Point), α ∥ β) ↔ (l ∩ m = ∅) ∧ (l ⊆ α) ∧ (m ⊆ β) := 
sorry

end lines_disjoint_planes_parallel_l279_279011


namespace trajectory_equation_max_area_and_line_eqn_l279_279411

-- Define the circles F1 and F2
def F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 9
def F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line equation
def line (k x y : ℝ) : Prop := y = k * x - 2

-- Problem I: Prove the equation of the trajectory
theorem trajectory_equation : ∀ (x y : ℝ), 
  (x ≠ 2) ∧ -- additional condition given in solution
  (∀ r : ℝ, -- for a moving circle tangent to F1 and F2
    (∀ (px py : ℝ), ((px + 1)^2 + py^2 = (3 - r)^2) → ((px - 1)^2 + py^2 = (1 + r)^2) → trajectory px py)) :=
sorry

-- Problem II: Maximum area of triangle OAB
theorem max_area_and_line_eqn : 
  ∃ (k : ℝ), (∀ (x1 x2 y1 y2 : ℝ), trajectory x1 y1 → trajectory x2 y2 → line k x1 y1 → line k x2 y2 →
  ∃ (A B : ℝ × ℝ), A = (x1, y1) ∧ B = (x2, y2) ∧ 
  let d := 2 / (Real.sqrt (1 + k^2)) in
  let AB := Real.sqrt (1 + k^2) * (Real.sqrt ((x1 + x2)^2 - 4 * (x1 * x2)) * sqrt (4 * k^2 - 1)) / (1 + 4 * k^2) in
  let area := (AB * d) / 2 in
  area = sqrt 3 ∧ (k = sqrt 5 / 2 ∨ k = -sqrt 5 / 2))) :=
sorry

end trajectory_equation_max_area_and_line_eqn_l279_279411


namespace relationship_between_y_values_l279_279515

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_values_l279_279515


namespace compute_expression_l279_279248

theorem compute_expression : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end compute_expression_l279_279248


namespace train_passing_time_l279_279222

theorem train_passing_time 
  (length_A : ℝ) (length_B : ℝ) (speed_A_kmph : ℝ) (speed_B_kmph : ℝ)
  (h1 : length_A = 450) (h2 : length_B = 300)
  (h3 : speed_A_kmph = 40) (h4 : speed_B_kmph = 30)
  (h_speed_conversion : ∀ kmph, kmph * (1000 / 3600) = kmph * (5 / 18)) :
  let speed_A := speed_A_kmph * (5 / 18),
      speed_B := speed_B_kmph * (5 / 18),
      relative_speed := speed_A + speed_B,
      total_distance := length_A + length_B,
      time := total_distance / relative_speed
  in time ≈ 38.58 := sorry

end train_passing_time_l279_279222


namespace smallest_integer_repr_CCCD8_l279_279979

theorem smallest_integer_repr_CCCD8 (C D : ℕ) (hC : C < 6) (hD : D < 8)
    (h_eq : 7 * C = 9 * D) : ∃ n : ℕ, (n = 7 * C) ∧ (n = 9 * D) ∧ (7 * C = 63) :=
by {
  existsi 63,
  split,
  { simp [←h_eq, mul_comm, mul_assoc, Nat.mul_div_cancel_left, Nat.gcd_eq, Nat.lcm_eq_gcd_mul] },
  { exact h_eq }
}

end smallest_integer_repr_CCCD8_l279_279979


namespace min_value_u_l279_279018

theorem min_value_u (x y : ℝ) (h1 : x ∈ Ioo (-2 : ℝ) 2) (h2 : y ∈ Ioo (-2 : ℝ) 2) (h3 : x * y = -1) : 
  ∃ u : ℝ, (u = (4 / (4 - x^2)) + (9 / (9 - y^2))) ∧ u ≥ 12 / 5 :=
sorry

end min_value_u_l279_279018


namespace find_x_l279_279587

def side_of_square_eq_twice_radius_of_larger_circle (s: ℝ) (r_l: ℝ) : Prop :=
  s = 2 * r_l

def radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle (r_l: ℝ) (x: ℝ) (r_s: ℝ) : Prop :=
  r_l = x - (1 / 3) * r_s

def circumference_of_smaller_circle_eq (r_s: ℝ) (circumference: ℝ) : Prop :=
  2 * Real.pi * r_s = circumference

def side_squared_eq_area (s: ℝ) (area: ℝ) : Prop :=
  s^2 = area

noncomputable def value_of_x (r_s r_l: ℝ) : ℝ :=
  14 + 4 / (3 * Real.pi)

theorem find_x 
  (s r_l r_s x: ℝ)
  (h1: side_squared_eq_area s 784)
  (h2: side_of_square_eq_twice_radius_of_larger_circle s r_l)
  (h3: radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle r_l x r_s)
  (h4: circumference_of_smaller_circle_eq r_s 8) :
  x = value_of_x r_s r_l :=
sorry

end find_x_l279_279587


namespace problem_1_problem_2_l279_279759

noncomputable def line_through_intersection (l_eq : ℝ → ℝ → Prop) : Prop :=
  ∃ λ : ℝ, ∀ x y : ℝ,
    (2 + λ) * x + (1 - 2 * λ) * y - 5 = 0 ∧
    (2 * x + y + 5 = 0 ∧ x - 2 * y = 0)

noncomputable def distance_point_to_line (P_x P_y : ℝ) (l_eq : ℝ → ℝ → Prop) (d : ℝ) : Prop :=
  ∃ λ : ℝ, ∀ x y : ℝ,
    (|2 * P_x + P_y - 5 + λ * (P_x - 2 * P_y)| / Real.sqrt ((2 + λ)^2 + (1 - 2 * λ)^2) = d) ∧
    l_eq x y

noncomputable def circles_centers_parallel (A_x A_y B_x B_y : ℝ) (l_eq : ℝ → ℝ → Prop) : Prop :=
  ∃ λ : ℝ, ∀ x y : ℝ,
    (2 * x + y - 5 + λ * (x - 2 * y) = 0) ∧
    (x - A_x)^2 + (y - A_y)^2 = 9 ∧
    (x - B_x)^2 + (y - B_y)^2 = 16 ∧
    (A_x - B_x) / (B_y - A_y) = 3 / 4 ∧
    l_eq x y

theorem problem_1 (l_eq : ℝ → ℝ → Prop) :
  line_through_intersection l_eq →
  distance_point_to_line 5 0 l_eq 4 →
  l_eq = (fun x y => x = 2 ∨ 4 * x - 3 * y - 5 = 0) :=
by sorry

theorem problem_2 (l_eq : ℝ → ℝ → Prop) :
  line_through_intersection l_eq →
  circles_centers_parallel 1 2 -3 -1 l_eq →
  l_eq = (fun x y => 3 * x - 4 * y - 2 = 0) :=
by sorry

end problem_1_problem_2_l279_279759


namespace interval_of_monotonic_increase_l279_279934

noncomputable def powerFunction (k n x : ℝ) : ℝ := k * x ^ n

variable {k n : ℝ}

theorem interval_of_monotonic_increase
    (h : ∃ k n : ℝ, powerFunction k n 4 = 2) :
    (∀ x y : ℝ, 0 < x ∧ x < y → powerFunction k n x < powerFunction k n y) ∨
    (∀ x y : ℝ, 0 ≤ x ∧ x < y → powerFunction k n x ≤ powerFunction k n y) := sorry

end interval_of_monotonic_increase_l279_279934


namespace green_marbles_removal_l279_279443

theorem green_marbles_removal:
  let total_marbles := 700 in
  let initial_green := total_marbles * 65 / 100 in
  let remaining_green (x : ℕ) := initial_green - x in
  let remaining_total (x : ℕ) := total_marbles - x in
  let desired_ratio := 60 / 100 in
  (remaining_green 88 / remaining_total 88 = desired_ratio) :=
by
  sorry

end green_marbles_removal_l279_279443


namespace proposition_is_false_l279_279699

noncomputable def false_proposition : Prop :=
¬(∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), Real.sin x + Real.cos x ≥ 2)

theorem proposition_is_false : false_proposition :=
by
  sorry

end proposition_is_false_l279_279699


namespace neg_of_p_l279_279053

variable (x : ℝ)

def p : Prop := ∀ x ≥ 0, 2^x = 3

theorem neg_of_p : ¬p ↔ ∃ x ≥ 0, 2^x ≠ 3 :=
by
  sorry

end neg_of_p_l279_279053


namespace translation_coordinates_l279_279099

theorem translation_coordinates :
  ∀ (x y : ℝ), ∃ (x1 y1 : ℝ),
    (∀ A A₁ D D₁ : ℝ × ℝ,
    (A = (1, -2) → A₁ = (-1, 3) → D = (x, y) → D₁ = (x1, y1)) →
    D₁ = (x - 2, y + 5)) :=
by
  intros x y
  use (x - 2, y + 5)
  intros A A₁ D D₁ h1 h2 h3
  sorry

end translation_coordinates_l279_279099


namespace additional_grazing_area_l279_279631

open Real

theorem additional_grazing_area :
  let initial_rope_length := 12
  let extended_rope_length := 21
  let angle_fraction := 3 / 4
  let area (r : ℝ) := angle_fraction * π * r^2
  let A1 := area initial_rope_length
  let A2 := area extended_rope_length
  let ΔA := A2 - A1
  ΔA ≈ 699.9 := 
  by 
    sorry

end additional_grazing_area_l279_279631


namespace numbers_sum_is_correct_l279_279208

noncomputable def numbers_sum_equals_seventy_two : Prop :=
  ∃ (x : ℝ), (x > 0) ∧ (x^2 + (2 * x)^2 + (3 * x)^2 = 2016) ∧ (x + 2 * x + 3 * x = 72)

theorem numbers_sum_is_correct : numbers_sum_equals_seventy_two :=
begin
  use 12,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end numbers_sum_is_correct_l279_279208


namespace xy_le_half_x2_plus_y2_l279_279898

theorem xy_le_half_x2_plus_y2 (x y : ℝ) : 
  x * y ≤ (x^2 + y^2) / 2 := by
  have h1 : 0 ≤ (x - y)^2 := by
    apply pow_two_nonneg
  have h2 : (x - y)^2 = x^2 - 2 * x * y + y^2 := by
    ring
  have h3 : x^2 - 2 * x * y + y^2 ≥ 0 := by
    linarith [h1, h2]
  have h4 : x^2 + y^2 ≥ 2 * x * y := by
    linarith
  have h5 : (x^2 + y^2) / 2 ≥ x * y := by
    apply (div_le_iff' _).mpr h4
    exact zero_lt_two
  exact h5

end xy_le_half_x2_plus_y2_l279_279898


namespace homothety_centers_collinear_homothety_centers_three_circles_concurrent_lines_to_internal_centers_concurrent_internal_external_l279_279057

noncomputable def circles : Type :=
{C1 C2 C3 : Type}

-- Conditions
def no_two_circles_equal (C1 C2 C3 : Type) : Prop :=
C1 ≠ C2 ∧ C2 ≠ C3 ∧ C1 ≠ C3

def centers_not_collinear (O1 O2 O3: Type) : Prop :=
¬ collinear {O1, O2, O3}

-- Definitions for homothety centers
def external_homothety_center (C1 C2 : Type) : Type := sorry
def internal_homothety_center (C1 C2 : Type) : Type := sorry

-- Proof statements
theorem homothety_centers_collinear (C1 C2 C3: Type) (O1 O2 O3: Type): 
  no_two_circles_equal C1 C2 C3 → centers_not_collinear O1 O2 O3 →
  collinear {external_homothety_center C1 C2, external_homothety_center C2 C3, external_homothety_center C1 C3} :=
sorry

theorem homothety_centers_three_circles (C1 C2 C3: Type) (O1 O2 O3: Type): 
  no_two_circles_equal C1 C2 C3 → centers_not_collinear O1 O2 O3 →
  ∀ (C1 C2 C3: Type), 
  collinear {internal_homothety_center C1 C2, external_homothety_center C1 C3, internal_homothety_center C2 C3} :=
sorry

theorem concurrent_lines_to_internal_centers (C1 C2 C3: Type) (O1 O2 O3: Type):
  no_two_circles_equal C1 C2 C3 → centers_not_collinear O1 O2 O3 →
  concurrent {O1, internal_homothety_center C2 C3; O2, internal_homothety_center C3 C1; O3, internal_homothety_center C1 C2} :=
sorry

theorem concurrent_internal_external (C1 C2 C3: Type) (O1 O2 O3: Type):
  no_two_circles_equal C1 C2 C3 → centers_not_collinear O1 O2 O3 →
  ∀ (C1 C2 C3: Type), 
  concurrent {O1, internal_homothety_center C2 C3; O2, internal_homothety_center C3 C1; O3, external_homothety_center C1 C2; 
  O2, external_homothety_center C3 C1; O3, external_homothety_center C1 C2} :=
sorry

end homothety_centers_collinear_homothety_centers_three_circles_concurrent_lines_to_internal_centers_concurrent_internal_external_l279_279057


namespace find_plane_equation_l279_279931

def is_normal_vector (v : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  v = p

def equation_of_plane (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def gcd_condition (A B C D : ℤ) : Prop :=
  Int.gcd (Int.gcd (Int.gcd A B) C) D = 1

noncomputable def point := (12, -4, 3 : ℝ)

noncomputable def normal_vector := (12, -4, 3 : ℝ)

theorem find_plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ gcd_condition A B C D ∧ 
  is_normal_vector normal_vector point ∧ 
  equation_of_plane A B C D 12 (-4) 3 ∧ 
  equation_of_plane A B C D = (λ x y z, 12 * x - 4 * y + 3 * z - 169) := 
sorry

end find_plane_equation_l279_279931


namespace prob_div_3_is_six_sevenths_l279_279314

/-- 
Prove that the probability of a randomly chosen divisor of 15! being divisible by 3 is 6/7, 
given that the prime factorization of 15! is 2^11 * 3^6 * 5^3 * 7^2 * 11 * 13.
-/
def probability_divisible_by_3 (n : ℕ) (p_factors : n = 2^11 * 3^6 * 5^3 * 7^2 * 11 * 13) : 
  ℚ := 
if n = 15! then 6/7 else 0

theorem prob_div_3_is_six_sevenths (n : ℕ) (h : n = 15!) : 
  probability_divisible_by_3 n (by rw h; norm_num) = 6 / 7 :=
sorry

end prob_div_3_is_six_sevenths_l279_279314


namespace certain_number_l279_279073

theorem certain_number (x y : ℤ) (h1 : x - y = 9) (h2 : x = 9) : 3^x * 4^y = 19683 := by
  sorry

end certain_number_l279_279073


namespace part_I_part_II_part_III_l279_279398

noncomputable def f (x m : ℝ) : ℝ := (x + m) * Real.log x - (m + 1 + 1 / Real.exp 1) * x

theorem part_I (h : deriv (λ x, f x m) e = 0) : m = 1 := sorry

theorem part_II (x : ℝ) (h1 : x > 1) (m : ℝ) (h2 : m = 1) : 
  f x m + (2 + 1 / Real.exp 1) * x > 2 * x - 2 := sorry

theorem part_III (x a : ℝ) (h1 : a ≥ 2) (h2 : x ≥ 1) :
  let px := e / x
  let qx := Real.exp (x - 1) + a
  |f x 1 - e / x| ≤ |f x 1 - (Real.exp (x - 1) + a)| := sorry

end part_I_part_II_part_III_l279_279398


namespace morning_trip_fare_afternoon_trip_fare_afternoon_trip_fare_when_x_is_8_l279_279570

theorem morning_trip_fare : 
  let distance := 6
  let time := 10
  let mileage_rate := 2.8
  let time_rate := 0.38
  in mileage_rate * distance + time_rate * time = 20.6 :=
by
  let distance := 6
  let time := 10
  let mileage_rate := 2.8
  let time_rate := 0.38
  have fare := mileage_rate * distance + time_rate * time
  have expected_fare := 20.6
  exact calc
    fare = 16.8 + 3.8 : by rw [fare]
    ... = 20.6 : by sorry

theorem afternoon_trip_fare (x : ℕ) (h : x ≤ 30) : 
  let mileage_rate := 2.75
  let time_rate := 0.47
  let speed := 30
  let time := x / speed * 60
  in mileage_rate * x + time_rate * time = 3.69 * x :=
by
  let mileage_rate := 2.75
  let time_rate := 0.47
  let speed := 30
  let time := x / speed * 60
  have fare := mileage_rate * x + time_rate * time
  have expected_fare := 3.69 * x
  exact calc
    fare = (mileage_rate * x + 0.47 * (x/30 * 60)) : by rw [fare]
    ... = 3.69 * x : by sorry

theorem afternoon_trip_fare_when_x_is_8 : 
  let x := 8
  let result := 3.69 * x
  (3.69 * 8 = 29.52) :=
by
  let x := 8
  let result := 3.69 * x
  have expected_result := 29.52
  exact calc
    result = 29.52 : by sorry

end morning_trip_fare_afternoon_trip_fare_afternoon_trip_fare_when_x_is_8_l279_279570


namespace constant_term_expansion_l279_279226

open Classical

noncomputable def constant_term : ℕ :=
  (choose 11 4) * 5 ^ 4

theorem constant_term_expansion :
  constant_term = 206250 :=
by
  unfold constant_term
  calc
    (choose 11 4) * 5 ^ 4
        = 330 * 625 : by norm_num [choose]
    ... = 206250   : by norm_num

end constant_term_expansion_l279_279226


namespace calculation_proof_l279_279682
  
  open Real

  theorem calculation_proof :
    sqrt 27 - 2 * cos (30 * (π / 180)) + (1 / 2)^(-2) - abs (1 - sqrt 3) = sqrt 3 + 5 :=
  by
    sorry
  
end calculation_proof_l279_279682


namespace math_proof_problem_l279_279459

theorem math_proof_problem (a b c : ℝ) (h1 : a + b + c = 2006) 
  (h2 : ∃ (σ : Perm (Fin 3)), 
    (σ 0 = a - 2 ∧ σ 1 = b + 2 ∧ σ 2 = c ^ 2) ∨ 
    (σ 1 = a - 2 ∧ σ 0 = b + 2 ∧ σ 2 = c ^ 2) ∨ 
    (σ 2 = a - 2 ∧ σ 1 = b + 2 ∧ σ 0 = c ^ 2)) : 
  a ∈ {671.33} ∨ sorry :=
sorry

end math_proof_problem_l279_279459


namespace range_of_omega_for_zeros_in_interval_l279_279783

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - 1

theorem range_of_omega_for_zeros_in_interval (ω : ℝ) (hω_positve : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f ω x = 0 → 2 ≤ ω ∧ ω < 3) :=
sorry

end range_of_omega_for_zeros_in_interval_l279_279783


namespace first_day_of_month_l279_279184

noncomputable def day_of_week := ℕ → ℕ

def is_wednesday (n : ℕ) : Prop := day_of_week n = 3

theorem first_day_of_month (day_of_week : day_of_week) (h : is_wednesday 30) : day_of_week 1 = 2 :=
by
  sorry

end first_day_of_month_l279_279184


namespace triangle_identity_l279_279134

noncomputable def circumcenter (A B C : Type) : Type := sorry
noncomputable def orthocenter (A B C : Type) : Type := sorry
noncomputable def circumradius (ABC O : Type) : Type := sorry
noncomputable def distance (A B : Type) : ℝ := sorry

variables {A B C O H : Type}

theorem triangle_identity
  (O_is_circumcenter : O = circumcenter A B C)
  (H_is_orthocenter : H = orthocenter A B C)
  (r : ℝ) (R_is_circumradius : circumradius (triangle A B C) O = r)
  (a b c : ℝ) (a_squared_b_squared_c_squared : a^2 + b^2 + c^2 = 9 * r^2 - (distance O H)^2) :
  a^2 + b^2 + c^2 = 9 * r^2 - (distance O H) ^ 2 :=
sorry

end triangle_identity_l279_279134


namespace number_of_true_statements_l279_279748

-- Given non-zero vectors a and b
variables {a b : Vector} (ha : a ≠ 0) (hb : b ≠ 0)

-- Define the statements
def statement_1 : Prop := 
  (2 * a).direction = a.direction ∧ (2 * a).magnitude = 2 * a.magnitude

def statement_2 : Prop := 
  (-2 * a).direction ≠ (5 * a).direction ∧ (-2 * a).magnitude = (5 * a).magnitude

def statement_3 : Prop := 
  -2 * a = - (2 * a)

def statement_4 : Prop := 
  a - b = -(b - a)

-- Proof that the number of true statements is 3
theorem number_of_true_statements : 
  (statement_1 ∧ statement_2 ∧ statement_3 ∧ ¬statement_4) → (3 = 3) :=
by
  intro H
  sorry

end number_of_true_statements_l279_279748


namespace smallest_number_of_students_l279_279095

theorem smallest_number_of_students (n : ℕ) (x : ℕ) 
  (h_total : n = 5 * x + 3) 
  (h_more_than_50 : n > 50) : 
  n = 53 :=
by {
  sorry
}

end smallest_number_of_students_l279_279095


namespace int_satisfying_inequality_l279_279064

theorem int_satisfying_inequality:
  {x : ℤ // (x - 2) ^ 2 ≤ 4}.card = 5 :=
sorry

end int_satisfying_inequality_l279_279064


namespace terminal_side_of_angle_y_eq_neg_one_l279_279767
/-
Given that the terminal side of angle θ lies on the line y = -x,
prove that y = -1 where y = sin θ / |sin θ| + |cos θ| / cos θ + tan θ / |tan θ|.
-/


noncomputable def y (θ : ℝ) : ℝ :=
  (Real.sin θ / |Real.sin θ|) + (|Real.cos θ| / Real.cos θ) + (Real.tan θ / |Real.tan θ|)

theorem terminal_side_of_angle_y_eq_neg_one (θ : ℝ) (k : ℤ) (h : θ = k * Real.pi - (Real.pi / 4)) :
  y θ = -1 :=
by
  sorry

end terminal_side_of_angle_y_eq_neg_one_l279_279767


namespace range_of_omega_l279_279785

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ zeros : ℝ, (f(x) = cos (ω * x) - 1) and (count_zeros (f(x),  [0, 2 * π]) = 3)) ↔ (2 ≤ ω ∧ ω < 3) := 
sorry

end range_of_omega_l279_279785


namespace candy_bar_cost_l279_279854

-- Define the conditions
def cost_gum_over_candy_bar (C G : ℝ) : Prop :=
  G = (1/2) * C

def total_cost (C G : ℝ) : Prop :=
  2 * G + 3 * C = 6

-- Define the proof problem
theorem candy_bar_cost (C G : ℝ) (h1 : cost_gum_over_candy_bar C G) (h2 : total_cost C G) : C = 1.5 :=
by
  sorry

end candy_bar_cost_l279_279854


namespace j_99_is_36_l279_279697

noncomputable def j (x : ℕ) : ℕ :=
  if ∃ n : ℕ, x = 2 ^ n then
    Nat.log 2 x
  else
    1 + j (Nat.lcm x (x + 1))

theorem j_99_is_36 : j 99 = 36 := by
  sorry

end j_99_is_36_l279_279697


namespace problem_distance_is_sqrt3_l279_279553

noncomputable def distance_center_circle_to_line : ℝ :=
  let x₀ := -1 in
  let y₀ := 0 in
  let A := -real.sqrt 3 in
  let B := 1 in
  let C := real.sqrt 3 in
  abs (A * x₀ + B * y₀ + C) / (real.sqrt (A^2 + B^2))

theorem problem_distance_is_sqrt3 :
  distance_center_circle_to_line = real.sqrt 3 :=
sorry

end problem_distance_is_sqrt3_l279_279553


namespace chocolate_bars_in_large_box_l279_279252

theorem chocolate_bars_in_large_box :
  ∀ (number_of_small_boxes : ℕ) (number_of_chocolate_bars_per_small_box : ℕ), 
  number_of_small_boxes = 19 → 
  number_of_chocolate_bars_per_small_box = 25 → 
  number_of_small_boxes * number_of_chocolate_bars_per_small_box = 475 :=
by
  intros number_of_small_boxes number_of_chocolate_bars_per_small_box h1 h2
  rw [h1, h2]
  norm_num
  sorry

end chocolate_bars_in_large_box_l279_279252


namespace max_surface_area_of_30_cubes_l279_279312

theorem max_surface_area_of_30_cubes : ∃ (S : ℕ), S = 122 ∧ 
  (∀ (cubes : ℕ), cubes = 30 →
   ∃ (surface_area : ℕ), 
     (∀ (solid : Type), 
        (connected solid) → 
        (∃ (exposed_faces : ℕ), exposed_faces = surface_area) →
        surface_area ≤ S)) :=
begin
  sorry
end

end max_surface_area_of_30_cubes_l279_279312


namespace geom_seq_sum_l279_279735

variable {a : ℕ → ℝ}

theorem geom_seq_sum (h : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 ∨ a 5 + a 7 = -6 := by
  sorry

end geom_seq_sum_l279_279735


namespace P_n_has_unique_root_limit_of_c_n_l279_279720

def P_n (n : ℕ) (x : ℝ) : ℝ := x^(n+2) - 2*x + 1

theorem P_n_has_unique_root (n : ℕ) (n_pos : n > 0) :
  ∃! c ∈ Ioo 0 1, P_n n c = 0 := sorry

theorem limit_of_c_n (c_n : ℕ → ℝ) (h: ∀ n, (P_n n (c_n n) = 0) ∧ (c_n n) ∈ Ioo 0 1) :
  filter.tendsto c_n filter.at_top (nhds (1 / 2)) := sorry

end P_n_has_unique_root_limit_of_c_n_l279_279720


namespace concur_AA_l279_279541

-- Definitions for given conditions

def isTangentIntersectAt (circumcircle : Type) (A B D : Type) : Prop := sorry

def projectionsCircleIntersects (circle projections : Type) (AB C' : Type) : Prop := sorry

def pointsConstructedSimilarly (A' B' C' projections : Type) : Prop := sorry

-- The main theorem to be proven

theorem concur_AA'_BB'_CC' 
  (circumcircle : Type) 
  (A B C D A' B' C' : Type) 
  (h1 : isTangentIntersectAt circumcircle A B D) 
  (h2 : projectionsCircleIntersects (circle projections) AB C') 
  (h3 : pointsConstructedSimilarly A' B' C' (projections)) :
  ∃ (P : Type), (lineThrough A A' P) ∧ (lineThrough B B' P) ∧ (lineThrough C C' P) :=
sorry

end concur_AA_l279_279541


namespace cans_collected_l279_279269

theorem cans_collected (cans_per_bag : ℕ) (bags_needed : ℕ) : cans_per_bag = 57 → bags_needed = 2 → cans_per_bag * bags_needed = 114 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end cans_collected_l279_279269


namespace total_failing_grades_sum_l279_279212

variables {k : ℕ} (a : ℕ → ℕ)

def total_failing_grades : ℕ := (Finset.range k).sum (λ i, a (i + 1))

theorem total_failing_grades_sum (h : ∀ i, 1 ≤ i ∧ i ≤ k → a i ≥ a (i + 1)) : 
  total_failing_grades a = a 1 + a 2 + ... + a k :=
by sorry

end total_failing_grades_sum_l279_279212


namespace ninth_term_of_sequence_is_4_l279_279692

-- Definition of the first term and common ratio
def a1 : ℚ := 4
def r : ℚ := 1

-- Definition of the nth term of a geometric sequence
def a (n : ℕ) : ℚ := a1 * r^(n-1)

-- Proof that the ninth term of the sequence is 4
theorem ninth_term_of_sequence_is_4 : a 9 = 4 := by
  sorry

end ninth_term_of_sequence_is_4_l279_279692


namespace average_multiplier_l279_279191

theorem average_multiplier (a : ℕ := 7) (avg1 avg2 : ℕ := 15) : ∃ x : ℝ, (∑ i in finset.range a, avg1) * x = (∑ i in finset.range a, avg2) ∧ x = 5 :=
by
  use x
  sorry

end average_multiplier_l279_279191


namespace triangle_medians_equal_legs_l279_279847

variables {A B C A1 B1 : Type} [EuclideanGeometry A B C A1 B1]

theorem triangle_medians_equal_legs
  (h_median_AA1 : is_median A A1)
  (h_median_BB1 : is_median B B1)
  (h_angles_equal : angle C A A1 = angle C B B1) :
  distance A C = distance B C := 
sorry

end triangle_medians_equal_legs_l279_279847


namespace both_go_hiking_l279_279155

-- Define the probabilities of student A and student B going hiking
def probA : ℝ := 1 / 3
def probB : ℝ := 1 / 4

-- Define that these events are independent
def independent (p1 p2 : ℝ) : Prop := p1 * p2

-- State the theorem to be proved
theorem both_go_hiking : independent probA probB = 1 / 12 := by
  sorry

end both_go_hiking_l279_279155


namespace manny_gave_2_marbles_l279_279215

-- Define the total number of marbles
def total_marbles : ℕ := 36

-- Define the ratio parts for Mario and Manny
def mario_ratio : ℕ := 4
def manny_ratio : ℕ := 5

-- Define the total ratio parts
def total_ratio : ℕ := mario_ratio + manny_ratio

-- Define the number of marbles Manny has after giving some away
def manny_marbles_now : ℕ := 18

-- Calculate the marbles per part based on the ratio and total marbles
def marbles_per_part : ℕ := total_marbles / total_ratio

-- Calculate the number of marbles Manny originally had
def manny_marbles_original : ℕ := manny_ratio * marbles_per_part

-- Formulate the theorem
theorem manny_gave_2_marbles : manny_marbles_original - manny_marbles_now = 2 := by
  sorry

end manny_gave_2_marbles_l279_279215


namespace relationship_between_y_values_l279_279514

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_values_l279_279514


namespace unchanged_median_l279_279445

def remove_highest_and_lowest {α : Type*} [LinearOrder α] (scores : List α) : List α :=
  (scores.erase (scores.maximum' sorry)).erase (scores.minimum' sorry)

noncomputable def median {α : Type*} [LinearOrder α] (l : List α) : α := 
  l.nthLe (l.length / 2) sorry

theorem unchanged_median (scores : List ℝ) (h_len : scores.length = 9) :
  median (remove_highest_and_lowest scores) = median (scores : List ℝ) := sorry

end unchanged_median_l279_279445


namespace locus_of_perpendiculars_is_circle_l279_279522

-- Define the problem statements and required conditions.
variables {A B C M : Type} [EuclideanGeometry.point A B C] [EuclideanGeometry.point M]

def perpendiculars_intersect_at_point (A B C : Type) [EuclideanGeometry.point A B C] [EuclideanGeometry.point M] : Prop :=
  ∃ P : (A → B → C → M), is_perpendicular M P A ∧ is_perpendicular M P B ∧ is_perpendicular M P C

theorem locus_of_perpendiculars_is_circle (A B C M : Type) [EuclideanGeometry.point A B C] [EuclideanGeometry.point M]
  (h : perpendiculars_intersect_at_point A B C) :
  is_circumcircle A B C M :=
sorry

end locus_of_perpendiculars_is_circle_l279_279522


namespace greatest_visible_cubes_l279_279629

theorem greatest_visible_cubes (n : ℕ) (h : n = 9) : 
  let face_cubes := n * n,
      edge_cubes_removed := (n - 1) * 3,
      corner_cube := 1 in
  (3 * face_cubes) - edge_cubes_removed + corner_cube = 220 :=
by sorry

end greatest_visible_cubes_l279_279629


namespace find_XY_sq_l279_279480

-- Definitions and Conditions
def triangle (A B C : Type) := true -- Placeholder for triangle definition
def circumcircle (A B C : Type) := true -- Placeholder for circumcircle definition
def tangent (ω : Type) (P : Type) := true -- Placeholder for tangent definition
def projection (T : Type) (l : Type) := true -- Placeholder for projection definition

variables (A B C T X Y : Type)
variables (BT CT BC : ℝ) (TX TY XY : ℝ)
variables (triangle_ABC : triangle A B C)
variables (circumcircle_ABC : circumcircle A B C)
variables (tangent_B : tangent circumcircle_ABC B)
variables (tangent_C : tangent circumcircle_ABC C)
variables (projection_X : projection T (λ AB, true))
variables (projection_Y : projection T (λ AC, true))
variables (BT_length : BT = 15)
variables (CT_length : CT = 15)
variables (BC_length : BC = 20)
variables (eqn_1080 : TX^2 + TY^2 + XY^2 = 1080)

-- Problem Statement in Lean
theorem find_XY_sq
  (triangle_ABC : triangle A B C)
  (circumcircle_ABC : circumcircle A B C)
  (tangent_B : tangent circumcircle_ABC B)
  (tangent_C : tangent circumcircle_ABC C)
  (projection_X : projection T (λ AB, true))
  (projection_Y : projection T (λ AC, true))
  (BT_length : BT = 15)
  (CT_length : CT = 15)
  (BC_length : BC = 20)
  (eqn_1080 : TX^2 + TY^2 + XY^2 = 1080) :
  XY^2 = 697 :=
sorry

end find_XY_sq_l279_279480


namespace complement_union_l279_279409

open Set

namespace proof

def U := {1, 2, 3, 4}
def A := {1, 2}
def B := {2, 3}

theorem complement_union:
  compl (U∪ A ∪ B) = {4} :=
by {
  sorry
}

end proof

end complement_union_l279_279409


namespace tangent_vertical_y_axis_iff_a_gt_0_l279_279077

theorem tangent_vertical_y_axis_iff_a_gt_0 {a : ℝ} (f : ℝ → ℝ) 
    (hf : ∀ x > 0, f x = a * x^2 - Real.log x)
    (h_tangent_vertical : ∃ x > 0, (deriv f x) = 0) :
    a > 0 := 
sorry

end tangent_vertical_y_axis_iff_a_gt_0_l279_279077


namespace find_fourth_intersection_point_l279_279839

-- Conditions: the equation xy = 1 and known intersection points
def hyperbola_eq (x y : ℝ) : Prop := x * y = 1

def known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/5, 5)]

-- The Fourth Point we need to prove
def fourth_point (p : ℝ × ℝ) : Prop := p = (-5/12, -12/5)

-- Main theorem: Given the known intersection points, prove the fourth intersection point
theorem find_fourth_intersection_point :
  ∃ (x y : ℝ), (hyperbola_eq x y) ∧ (∀ (a b : ℝ) (h : (a, b) ∈ known_points), hyperbola_eq a b) ∧ (fourth_point (x, y)) :=
by
  exists (-5/12)
  exists (-12/5)
  split
  · -- show that x * y = 1 for the fourth point
    sorry
  split
  · -- verify that the known points satisfy x * y = 1
    intros a b h
    rcases h with ⟨hx, hy⟩
    cases h_1
    · exact (by norm_num : hyperbola_eq 3 (1/3))
    · exact (by norm_num : hyperbola_eq (-4) (-1/4))
    · exact (by norm_num : hyperbola_eq (1/5) 5)
  · -- show that the fourth point matches our proposed value
    exact rfl
  sorry


end find_fourth_intersection_point_l279_279839


namespace simplify_and_evaluate_l279_279168

noncomputable def simplified_eval_expr (x : ℝ) : ℝ :=
  ((x + 3) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 + x))

theorem simplify_and_evaluate :
  let x := Real.sqrt 2 - 2
  in simplified_eval_expr x = Real.sqrt 2 - 1 :=
by
  sorry

end simplify_and_evaluate_l279_279168


namespace area_of_quadrilateral_l279_279094

-- Given the areas of the three triangles within a main triangle
variables {EFA FAB FBD CEDF : ℝ}
-- Conditions
def condition_1 := EFA = 4
def condition_2 := FAB = 8
def condition_3 := FBD = 12

-- The theorem stating the area of quadrilateral CEDF
theorem area_of_quadrilateral :
  condition_1 → condition_2 → condition_3 → CEDF = 30 :=
by
  intros
  sorry

end area_of_quadrilateral_l279_279094


namespace jessica_alex_difference_l279_279462

def J : ℕ := 12 - (3 * 4)
def A : ℕ := (12 - 3) * 4

theorem jessica_alex_difference : J - A = -36 := by
  sorry

end jessica_alex_difference_l279_279462


namespace sum_of_coefficients_expansion_l279_279578

theorem sum_of_coefficients_expansion (x y : ℝ) : 
  (∑ k in Finset.range (14 + 1), Nat.choose 14 k) = 16384 :=
by
  sorry

end sum_of_coefficients_expansion_l279_279578


namespace first_day_of_month_l279_279177

theorem first_day_of_month (h: ∀ n, (n % 7 = 2) → n_day_of_week n = "Wednesday"): 
  n_day_of_week 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279177


namespace expression_positive_intervals_l279_279698

theorem expression_positive_intervals :
  {x : ℝ | (x + 2) * (x - 3) > 0} = {x | x < -2} ∪ {x | x > 3} :=
by
  sorry

end expression_positive_intervals_l279_279698


namespace polygon_interior_exterior_relation_l279_279027

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end polygon_interior_exterior_relation_l279_279027


namespace cos_2theta_l279_279419

theorem cos_2theta (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) : 
  Real.cos (2 * θ) = 3 / 4 := 
sorry

end cos_2theta_l279_279419


namespace truth_probability_l279_279424

theorem truth_probability (P_A : ℝ) (P_A_and_B : ℝ) (P_B : ℝ) 
  (hA : P_A = 0.70) (hA_and_B : P_A_and_B = 0.42) : 
  P_A * P_B = P_A_and_B → P_B = 0.6 :=
by
  sorry

end truth_probability_l279_279424


namespace find_coefficient_x5y2_expansion_eq_60_l279_279352

noncomputable def coefficient_of_x5y2_in_expansion : ℕ :=
  polynomial.coeff ((x^2 + x + y)^6) (5, 2)

theorem find_coefficient_x5y2_expansion_eq_60 : coefficient_of_x5y2_in_expansion = 60 := 
by 
  -- Placeholder for the actual proof.
  sorry

end find_coefficient_x5y2_expansion_eq_60_l279_279352


namespace probability_of_two_pairs_l279_279440

theorem probability_of_two_pairs (pairs : ℕ) (drawn : ℕ) : 
    pairs = 5 → drawn = 4 → ∃ n, (1 / n : ℚ) = 1 / 21 := 
by
  intros h_pairs h_drawn
  use 21
  norm_num
  assumption

end probability_of_two_pairs_l279_279440


namespace probability_friends_same_group_l279_279308

theorem probability_friends_same_group :
  let total_students := 1204
  let group_sizes := [300, 301, 300, 303]
  let specific_friends := ["Dustin", "Erica", "Lucas"]
  (number_of_groups : Nat) = group_sizes.length →
  (∀ group ∈ group_sizes, total_students = group + (total_students - group) - 4 * (group.count specific_friends.head + group.count specific_friends.head + group.count specific_friends.head)) →
  ∃ probs, probs = [0.062, 0.0624, 0.062, 0.0628] ∧
  (prob_friends_in_same_group : Float) = (probs.sum / number_of_groups : Float) →
  prob_friends_in_same_group ≈ 0.0623 :=
by
  sorry

end probability_friends_same_group_l279_279308


namespace intersection_complement_A_l279_279142

def A : Set ℝ := {x | abs (x - 1) < 1}

def B : Set ℝ := {x | x < 1}

def CRB : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_A :
  (CRB ∩ A) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_l279_279142


namespace exists_parallelepiped_with_congruent_faces_and_symmetry_axes_l279_279114

theorem exists_parallelepiped_with_congruent_faces_and_symmetry_axes :
  ∃ (P : Parallelepiped),
    (∀ (F₁ F₂ : Face P), Face.Congruent F₁ F₂) ∧
    ¬ (is_cube P) ∧
    (∃ (A B : Point P), is_symmetry_axis P A B) :=
by
  sorry

end exists_parallelepiped_with_congruent_faces_and_symmetry_axes_l279_279114


namespace last_three_digits_of_2_pow_10000_l279_279250

theorem last_three_digits_of_2_pow_10000 (h : 2^500 ≡ 1 [MOD 1250]) : (2^10000) % 1000 = 1 :=
by
  sorry

end last_three_digits_of_2_pow_10000_l279_279250


namespace find_p_q_r_s_l279_279872
noncomputable theory

def point := ℚ × ℚ

def A : point := (0, 0)
def B : point := (2, 3)
def C : point := (5, 4)
def D : point := (6, 0)

def line_eq (m b x : ℚ) := m * x + b

def divides_area_equally (p1 p2 p3 p4 : point) (intercept : (ℚ × ℚ)) :=
  let area (a b c d : point) := 
    (1/2) * abs ((a.1 * b.2 - b.1 *a.2) + 
                 (b.1 * c.2 - c.1 * b.2) + 
                 (c.1 * d.2 - d.1 * c.2) + 
                 (d.1 * a.2 - a.1 * d.2))
  in
  let total_area := area p1 p2 p3 p4 in
  let area1 := area p1 p2 intercept p1 in
  (2 * area1) = total_area

def fractions_in_lowest_terms (p q r s : ℤ) : Prop := 
  (∃ k : ℤ, q = s * k) ∧ (∀ n : ℤ, gcd n q = 1) ∧ (∀ m : ℤ, gcd m s = 1)

theorem find_p_q_r_s :
  ∃ p q r s : ℤ, divides_area_equally A B C D (p/q, r/s) ∧ fractions_in_lowest_terms p q r s ∧ p + q + r + s = 116 :=
sorry

end find_p_q_r_s_l279_279872


namespace omega_range_l279_279778

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l279_279778


namespace sqrt_of_16_is_4_l279_279580

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 :=
sorry

end sqrt_of_16_is_4_l279_279580


namespace flag_design_count_l279_279275

-- Define the colors as a finite type
inductive Color
| red
| white
| blue
| green

open Color

-- Define a valid flag strips color assignment based on the given conditions
def validFlag (left middle right : Color) : Prop :=
  left ≠ middle ∧ middle ≠ right

-- Define the main problem to prove
theorem flag_design_count : ∃ n : ℕ, n = 36 ∧ 
  (∃ (left middle right : Color), validFlag left middle right) := 
by
  let colors := [red, white, blue, green]

  -- Total number of options for each choice
  have total_choices : 4 * 3 * 3 = 36 := 
  by norm_num

  existsi 36
  split
  exact total_choices
  
  existsi (red : Color), (white : Color), (blue : Color)
  simp [validFlag]
  split; norm_num

end flag_design_count_l279_279275


namespace max_product_of_roots_l279_279396

theorem max_product_of_roots 
  (m : ℝ) 
  (h : ∀ (x : ℝ), 5 * x^2 - 10 * x + m = 0 → (100 - 20 * m) ≥ 0) :
  (m ≤ 5) → (m = 5 → (m / 5) = 1) :=
begin
  sorry
end

end max_product_of_roots_l279_279396


namespace correct_statements_l279_279038

def f (x : ℝ) : ℝ := x + sin (2 * x)

-- Statement 1: ∀ x > 0, f(x) < 2x
def statement1 : Prop := ∀ x > 0, f x < 2 * x

-- Statement 2: ∃ k ∈ ℝ, such that the equation f(x) = k has four distinct real roots
def statement2 : Prop := ∃ k : ℝ, ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    f x1 = k ∧ f x2 = k ∧ f x3 = k ∧ f x4 = k

-- Statement 3: The graph of f(x) has infinitely many centers of symmetry
def statement3 : Prop := ∀ n : ℤ, ∃ x : ℝ, x = (n + 1/2) * π ∧ 
    ∀ y : ℝ, f (x + y) = f (x - y)

-- Statement 4: If {a_n} is an arithmetic sequence and f(a_1) + f(a_2) + f(a_3) = 3π, then a_2 = π
def arithmetic_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
def statement4 : Prop := ∀ (a : ℕ → ℝ), 
    arithmetic_seq a → f (a 0) + f (a 1) + f (a 2) = 3 * π → a 1 = π

-- The proof goal: determining the correct statements
theorem correct_statements : ¬statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4 := by
  sorry

end correct_statements_l279_279038


namespace quadratic_factorization_m_l279_279423

theorem quadratic_factorization_m (m : ℤ) :
  (∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (x : ℤ) → (x + a) * (x + b) = x^2 + (3 - m) * x + 25) ↔ m = -7 ∨ m = 13 :=
sorry

end quadratic_factorization_m_l279_279423


namespace polynomial_diff_l279_279914

noncomputable def is_degree_42_poly (f : ℤ → ℤ) : Prop :=
  ∃ (a : ℕ → ℤ), (∀ k > 42, a k = 0) ∧ (f = λ x, ∑ i in finset.range 43, a i * x^i)

theorem polynomial_diff
  (f : ℤ → ℤ)
  (H_deg : is_degree_42_poly f)
  (H_cond : ∀ i : ℤ, 0 ≤ i ∧ i ≤ 42 → f i + f (43 + i) + f (2 * 43 + i) + ⋯ + f (46 * 43 + i) = (-2) ^ i) :
  f 2021 - f 0 = 2^43 - 2 :=
begin
  sorry
end

end polynomial_diff_l279_279914


namespace power_of_two_contains_k_as_substring_l279_279733

theorem power_of_two_contains_k_as_substring (k : ℕ) (h1 : 1000 ≤ k) (h2 : k < 10000) : 
  ∃ n < 20000, ∀ m, 10^m * k ≤ 2^n ∧ 2^n < 10^(m+4) * (k+1) :=
sorry

end power_of_two_contains_k_as_substring_l279_279733


namespace andy_solved_problems_up_to_l279_279672

theorem andy_solved_problems_up_to :
  ∀ (start solved count : ℕ), start = 80 → solved = 46 → count = start + solved - 1 → count = 125 :=
by
  intros start solved count hstart hsolved hcount
  rw [hstart, hsolved] at hcount
  dsimp at hcount
  rw [Nat.add_sub_cancel] at hcount
  exact hcount

end andy_solved_problems_up_to_l279_279672


namespace sum_of_elements_in_T_l279_279131

def T := {x : ℕ | x >= 16 ∧ x <= 31}

theorem sum_of_elements_in_T : (finset.sum (finset.filter (λ x, x ∈ T) (finset.range 32))) = 248 :=
by
  sorry

end sum_of_elements_in_T_l279_279131


namespace max_norm_two_a_plus_b_l279_279807

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem max_norm_two_a_plus_b
  (h : 4 * ∥a∥^2 + ⟪a, b⟫ + ∥b∥^2 = 1) :
  ∥2 • a + b∥ ≤ 2 * real.sqrt 10 / 5 :=
sorry

end max_norm_two_a_plus_b_l279_279807


namespace sum_of_factorials_is_perfect_square_l279_279335

theorem sum_of_factorials_is_perfect_square (n : ℕ) : 
  (∃ k : ℕ, (∑ i in Finset.range (n + 1), Nat.factorial i) = k * k) ↔ (n = 1 ∨ n = 3) :=
by sorry

end sum_of_factorials_is_perfect_square_l279_279335


namespace parallelogram_area_correct_l279_279410

/-- Points in space A, B, C -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Vector from point A to point B -/
def vector (A B : Point3D) : Point3D :=
  ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

/-- Dot product of two vectors -/
def dot_product (v w : Point3D) : ℝ :=
  v.x * w.x + v.y * w.y + v.z * w.z

/-- Magnitude of a vector -/
def magnitude (v : Point3D) : ℝ :=
  real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

/-- Given points A, B, and C, the area of the parallelogram formed by vectors AB and AC -/
def parallelogram_area (A B C : Point3D) : ℝ :=
  let AB := vector A B
  let AC := vector A C
  let AB_mag := magnitude AB
  let AC_mag := magnitude AC
  let dot_AB_AC := dot_product AB AC
  let cos_angle := dot_AB_AC / (AB_mag * AC_mag)
  let sin_angle := real.sqrt (1 - cos_angle * cos_angle)
  AB_mag * AC_mag * sin_angle

theorem parallelogram_area_correct : parallelogram_area ⟨0, 2, 3⟩ ⟨2, 5, 2⟩ ⟨-2, 3, 6⟩ = 6 * real.sqrt 5 :=
by sorry

end parallelogram_area_correct_l279_279410


namespace fruit_salad_total_l279_279441

def fruit_salad_problem (R_red G R_rasp total_fruit : ℕ) : Prop :=
  R_red = 67 ∧ (3 * G + 7 = 67) ∧ (R_rasp = G - 5) ∧ (total_fruit = R_red + G + R_rasp)

theorem fruit_salad_total (R_red G R_rasp : ℕ) (total_fruit : ℕ) :
  fruit_salad_problem R_red G R_rasp total_fruit → total_fruit = 102 :=
by
  intro h
  sorry

end fruit_salad_total_l279_279441


namespace find_lines_AB_BC_AC_l279_279110

noncomputable theory
open_locale classical

-- Definitions of known points and lines
def Point := ℝ × ℝ
def line_eq (a b c : ℝ) (p : Point) : Prop := a * p.1 + b * p.2 + c = 0

def A : Point := (0, 1)
def altitude_AB : line_eq 1 2 (-4)
def median_AC : line_eq 2 1 (-3)
def point_on_line (p : Point) (a b c : ℝ) : Prop := line_eq a b c p

-- The additional assumptions to use in the hypothesis
axiom H1 : point_on_line A 1 2 (-4)  -- altitude from A to AB
axiom H2 : point_on_line A 2 1 (-3)  -- median from A to AC

-- The goals to be proved
def side_AB_equation := ∀ B: Point, point_on_line B 2 (-1) 1 = altitude_AB A
def side_BC_equation := ∀ B C: Point, point_on_line B 2 (-1) 1 ∧ point_on_line C 2 1 (-3) → point_on_line C 2 3 (-7)
def side_AC_equation := ∀ C: Point, point_on_line C 2 1 (-3) → point_on_line C 0 1 (-1)

-- Combine all goals into a single theorem statement
theorem find_lines_AB_BC_AC :
side_AB_equation ∧
side_BC_equation ∧
side_AC_equation := by
  sorry  -- Proof goes here

end find_lines_AB_BC_AC_l279_279110


namespace base12_to_base10_mod_l279_279238

theorem base12_to_base10_mod :
  let n := 2 * 12^3 + 7 * 12^2 + 4 * 12 + 3
  in n % 10 = 5 :=
by
  let n := 2 * 12^3 + 7 * 12^2 + 4 * 12 + 3
  have : n = 4515 := by decide -- This simplifies the given base-12 number
  rw [this]
  show 4515 % 10 = 5
  rfl

end base12_to_base10_mod_l279_279238


namespace triangle_inequality_l279_279739

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (h1 : angle A + angle C = 2 * angle B)
  (h2 : sin B = sqrt 3 / 2) : a^4 + c^4 ≤ 2 * b^4 := 
sorry

end triangle_inequality_l279_279739


namespace vector_computation_l279_279870

noncomputable def u : ℝ^3 := ![2, -3, 4]
noncomputable def v : ℝ^3 := ![1, 5, -1]
noncomputable def w : ℝ^3 := ![0, 2, 3]

theorem vector_computation :
  ((u + 2 • v) ⬝ ((v + 3 • w) × (w - 2 • u))) = -183 :=
by
  sorry

end vector_computation_l279_279870


namespace problem_31_36_l279_279498

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem problem_31_36 (p k : ℕ) (hp : is_prime (4 * k + 1)) :
  (∃ x y m : ℕ, x^2 + y^2 = m * p) ∧ (∀ m > 1, ∃ x y m1 : ℕ, x^2 + y^2 = m * p ∧ 0 < m1 ∧ m1 < m) :=
by sorry

end problem_31_36_l279_279498


namespace min_distance_proof_l279_279052

-- Defining the polar equation of the line
def polar_line_eq (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 4) = sqrt 2 / 2

-- Defining the parametric equations of the circle
def circle_param (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, -2 + 2 * sin θ)

-- Defining the Cartesian equation of the line
def cartesian_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define minimum distance computation
def minimum_distance_from_circle_to_line : ℝ :=
  (3 * sqrt 2 / 2) - 2

-- Main proof statement
theorem min_distance_proof :
  ∀ (θ : ℝ) (ρ : ℝ),
    polar_line_eq ρ θ →
    ∃ x y, circle_param θ = (x, y) ∧ cartesian_line x y → 
    |x + (y + 2)| / sqrt 2 = (3 * sqrt 2 / 2) - 2 :=
by
  sorry

end min_distance_proof_l279_279052


namespace f_2021_2022_2023_l279_279305

noncomputable def f (x : ℝ) : ℝ := sorry -- The explicit form of f(x) will be defined through axioms.

-- Assume f is odd: ∀ x, f(x) = -f(-x)
def odd (f : ℝ → ℝ) : Prop := ∀ x, f(x) = -f(-x)

-- Assume f(2-x) = f(x)
def periodic (f : ℝ → ℝ) : Prop := ∀ x, f(2 - x) = f(x)

-- Definition of f(x) over the interval [0, 1]
def f_specified (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) = real.log x + 1 / real.log 2

-- Prove the main thesis
theorem f_2021_2022_2023 (f : ℝ → ℝ) (h1 : odd f) (h2 : periodic f) (h3 : f_specified f) :
  f 2021 + f 2022 + f 2023 = 0 :=
sorry

end f_2021_2022_2023_l279_279305


namespace tangency_and_range_of_a_l279_279523

noncomputable def f (a : ℝ) (x : ℝ) := a * (x - 1)
noncomputable def g (a : ℝ) (x : ℝ) := (a * x - 1) * Real.exp x

theorem tangency_and_range_of_a :
  (∃! a : ℝ, ∃ x₀ : ℝ,
    f a x₀ = g a x₀ ∧
    ∀ x₀: ℝ, f a x₀ = g a x₀ → (Real.exp x₀ + x₀ - 2 = 0))
  ∧
  (∃ x₀ : ℕ, f a x₀ > g a x₀ ∧ ∃ x₁ : ℕ, x₀ ≠ x₁ ∧ f a x₁ > g a x₁)
  → a ∈ set.Ico (Real.exp 2 / (2 * Real.exp 2 - 1)) 1 :=
begin
  sorry,
end

end tangency_and_range_of_a_l279_279523


namespace bottles_more_than_apples_l279_279277

-- Definitions given in the conditions
def apples : ℕ := 36
def regular_soda_bottles : ℕ := 80
def diet_soda_bottles : ℕ := 54

-- Theorem statement representing the question
theorem bottles_more_than_apples : (regular_soda_bottles + diet_soda_bottles) - apples = 98 :=
by
  sorry

end bottles_more_than_apples_l279_279277


namespace neighbor_to_johnson_yield_ratio_l279_279465

-- Definitions
def johnsons_yield (months : ℕ) : ℕ := 80 * (months / 2)
def neighbors_yield_per_hectare (x : ℕ) (months : ℕ) : ℕ := 80 * x * (months / 2)
def total_neighor_yield (x : ℕ) (months : ℕ) : ℕ := 2 * neighbors_yield_per_hectare x months

-- Theorem statement
theorem neighbor_to_johnson_yield_ratio
  (x : ℕ)
  (h1 : johnsons_yield 6 = 240)
  (h2 : total_neighor_yield x 6 = 480 * x)
  (h3 : johnsons_yield 6 + total_neighor_yield x 6 = 1200)
  : x = 2 := by
sorry

end neighbor_to_johnson_yield_ratio_l279_279465


namespace distance_between_axes_of_symmetry_l279_279406

theorem distance_between_axes_of_symmetry (m : ℝ) (h : 0 < m ∧ 0 < m^2 ∧ 
  ∀ x, (x = -m ∨ x = 0 ∨ x = m ∨ x = m^2) → 
        (y = -x^2 + m^2 * x ∨ y = x^2 - m^2) → 
          (y = 0 ∧ y.has_root x)) : 
  abs (m^2 / 2 - 0) = 2 := 
by
  sorry

end distance_between_axes_of_symmetry_l279_279406


namespace lines_through_centroid_l279_279743

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

-- The centroid of a triangle
def centroid (ABC : Triangle) : Point :=
{ x := (ABC.A.x + ABC.B.x + ABC.C.x) / 3,
  y := (ABC.A.y + ABC.B.y + ABC.C.y) / 3 }

-- The distance from a point to a line ax + by + c = 0
def distance (P : Point) (a b c : ℝ) : ℝ :=
  (abs (a * P.x + b * P.y + c)) / (sqrt (a^2 + b^2))

-- The condition that one distance is equal to the sum of two other distances
def distance_condition (ABC : Triangle) (a b c : ℝ) : Prop :=
  distance ABC.A a b c = distance ABC.B a b c + distance ABC.C a b c

-- Proving that any such line passes through the centroid
theorem lines_through_centroid {ABC : Triangle} (a b c : ℝ) :
  distance_condition ABC a b c → (∃ k : ℝ, a * k = centroid ABC.x ∧ b * k = centroid ABC.y) :=
sorry

end lines_through_centroid_l279_279743


namespace hyperbola_sufficient_but_not_necessary_asymptote_l279_279194

-- Define the equation of the hyperbola and the related asymptotes
def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptote_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

-- Stating the theorem that expresses the sufficiency but not necessity
theorem hyperbola_sufficient_but_not_necessary_asymptote (a b : ℝ) :
  (∃ x y, hyperbola_eq a b x y) → (∀ x y, asymptote_eq a b x y) ∧ ¬ (∀ x y, (asymptote_eq a b x y) → (hyperbola_eq a b x y)) := 
sorry

end hyperbola_sufficient_but_not_necessary_asymptote_l279_279194


namespace exists_regular_icosahedron_l279_279161

theorem exists_regular_icosahedron :
  ∃ P : Polyhedron, is_regular P ∧ each_face_is_triangular P ∧ vertices_have_five_edges_meeting P :=
sorry

end exists_regular_icosahedron_l279_279161


namespace least_multiple_of_21_gt_380_l279_279974

theorem least_multiple_of_21_gt_380 : ∃ n : ℕ, (21 * n > 380) ∧ (21 * n = 399) :=
sorry

end least_multiple_of_21_gt_380_l279_279974


namespace limit_calculation_l279_279680

/-- Calculate the limit of the given function -/
theorem limit_calculation (a : ℝ) (h : 0 < a) : 
  (filter.tendsto (λ x : ℝ, 
    (a^(x^2 - a^2) - 1) / (real.tan (real.log (x / a)))) (nhds a) (nhds (2 * a^2 * real.log a))) :=
by
  sorry

end limit_calculation_l279_279680


namespace solve_problem_l279_279451

noncomputable def problem_statement : Prop :=
  ∃ (A B C D : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    ∀ (sol_BD : Bool), D = if sol_BD then C + A else C + A - 10 ∧ ABAC + BCBA = DBCDD ∧ 
    ∃ (dist_values : ℕ), dist_values = 9

theorem solve_problem : problem_statement :=
  sorry

end solve_problem_l279_279451


namespace find_angle_B1C1C_l279_279457

noncomputable def angle_B1C1C (AC AA1 A1C AB A1B BC : ℝ) : ℝ :=
  if AC = sqrt 2 ∧ AA1 = 1 ∧ A1C = 1 ∧ AB = sqrt 6 / 3 ∧ A1B = sqrt 3 / 3 ∧ BC = 2 * sqrt 3 / 3
  then arccos (sqrt 3 / 6)
  else 0
  
theorem find_angle_B1C1C :
  let AC := sqrt 2
  let AA1 := 1
  let A1C := 1
  let AB := sqrt 6 / 3
  let A1B := sqrt 3 / 3
  let BC := 2 * sqrt 3 / 3
  angle_B1C1C AC AA1 A1C AB A1B BC = arccos (sqrt 3 / 6) :=
by
  sorry

end find_angle_B1C1C_l279_279457


namespace math_expression_identity_l279_279685

theorem math_expression_identity :
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 :=
by
  sorry

end math_expression_identity_l279_279685


namespace omega_range_l279_279776

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l279_279776


namespace range_of_omega_for_zeros_in_interval_l279_279780

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - 1

theorem range_of_omega_for_zeros_in_interval (ω : ℝ) (hω_positve : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f ω x = 0 → 2 ≤ ω ∧ ω < 3) :=
sorry

end range_of_omega_for_zeros_in_interval_l279_279780


namespace product_of_numbers_in_given_ratio_l279_279604

theorem product_of_numbers_in_given_ratio :
  ∃ (x y : ℝ), (x - y) ≠ 0 ∧ (x + y) / (x - y) = 9 ∧ (x * y) / (x - y) = 40 ∧ (x * y) = 80 :=
by {
  sorry
}

end product_of_numbers_in_given_ratio_l279_279604


namespace reading_time_per_week_l279_279590

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end reading_time_per_week_l279_279590


namespace gauge_block_diameter_l279_279657

-- Definition of the conditions
def diameter_peg_2 : ℝ := 2
def diameter_peg_1 : ℝ := 1
def diameter_hole : ℝ := 3
def accuracy : ℝ := 1 / 100

-- Main theorem statement
theorem gauge_block_diameter (r : ℝ) 
  (h1 : diameter_hole = diameter_peg_2 + diameter_peg_1 + 2 * r) 
  (h2 : 0 < r ∧ r < diameter_hole / 2) : 
  2 * r = 0.86 := 
sorry

end gauge_block_diameter_l279_279657


namespace distance_from_P_to_line_AB_l279_279799

structure Point3D :=
  (x y z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def distance_point_to_line (P A B : Point3D) : ℝ :=
  let AB := vector_sub B A
  let AP := vector_sub P A
  let cos_theta := dot_product AB AP / (magnitude AB * magnitude AP)
  magnitude AP * Real.sqrt (1 - cos_theta^2)

noncomputable def correct_distance : ℝ := Real.sqrt 6 / 3

theorem distance_from_P_to_line_AB : ∀ (P A B : Point3D), 
  P = ⟨1, 1, 1⟩ → A = ⟨1, 0, 1⟩ → B = ⟨0, 1, 0⟩ → 
  distance_point_to_line P A B = correct_distance :=
by
  intros P A B hP hA hB
  rw [hP, hA, hB]
  sorry

end distance_from_P_to_line_AB_l279_279799


namespace sum_of_integers_l279_279213

theorem sum_of_integers (a b : ℕ) (h1 : a * a + b * b = 585) (h2 : Nat.gcd a b + Nat.lcm a b = 87) : a + b = 33 := 
sorry

end sum_of_integers_l279_279213


namespace at_least_half_girls_prob_l279_279165

-- Define the conditions
def total_children : ℕ := 6
def prob_girl : ℚ := 3 / 5
def prob_boy : ℚ := 1 - prob_girl

-- Define the binomial probability function
noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the probability that at least 3 out of 6 children are girls
noncomputable def prob_at_least_3_girls : ℚ :=
  binomial_prob total_children 3 prob_girl +
  binomial_prob total_children 4 prob_girl +
  binomial_prob total_children 5 prob_girl +
  binomial_prob total_children 6 prob_girl

-- The proof statement
theorem at_least_half_girls_prob :
  prob_at_least_3_girls = 513 / 625 :=
by
  sorry

end at_least_half_girls_prob_l279_279165


namespace find_a_and_b_arithmetic_square_root_l279_279766

theorem find_a_and_b (a b : ℤ) 
    (h₁ : 2 * a - 7 = a + 4 ∨ a + 4 = 2 * a - 7)
    (h₂ : (b - 12)^3 = (-2)^3) : 
    a = 1 ∧ b = 4 := 
begin
  sorry
end

theorem arithmetic_square_root (a b : ℤ)
    (h₁ : 2 * a - 7 = a + 4 ∨ a + 4 = 2 * a - 7)
    (h₂ : (b - 12)^3 = (-2)^3)
    (h₃ : a = 1)
    (h₄ : b = 4) :
    sqrt (5 * a + b) = 3 := 
begin
  sorry
end

end find_a_and_b_arithmetic_square_root_l279_279766


namespace distance_focus_asymptote_hyperbola_l279_279353

/-- The distance from the focus to the asymptote of the hyperbola y^2 - (x^2)/2 = 1 is √2. -/
theorem distance_focus_asymptote_hyperbola : 
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let focus := (0, c)
  let asymptote := (λ x y, x - b * y = 0)
  let distance := (λ p l, abs (fst p * 1 + snd p * (-b) + 0) / Real.sqrt (1^2 + b^2)) 
  distance focus asymptote = Real.sqrt 2 :=
by sorry

end distance_focus_asymptote_hyperbola_l279_279353


namespace triangle_area_proof_l279_279458

-- Define the sides a and b, and the angle C in triangle ABC
def a := 1
def b := 2
def C := 150 * Real.pi / 180  -- Conversion from degrees to radians

-- Define the sine of angle C
def sin_C := Real.sin C

-- Define the area of the triangle using the given formula
def triangle_area (a b sin_C : ℝ) : ℝ := 0.5 * a * b * sin_C

-- The theorem to be proved
theorem triangle_area_proof : triangle_area a b sin_C = 1 / 2 := by
  sorry

end triangle_area_proof_l279_279458


namespace expansion_terms_count_l279_279196

theorem expansion_terms_count (n : ℕ) : 
  (∑ k in Finset.powerset (Finset.range (n.succ + 1)), 
  1 = (n.succ) ^ (3 - 1)) →
  (∃ t : ℕ, t = 78) :=
by
  intros h
  use 78
  rwa Finset.card_powerset_len_eq.symm at h

end expansion_terms_count_l279_279196


namespace sin_beta_value_l279_279747

theorem sin_beta_value
  (α β : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < real.pi / 2)
  (h₃ : -real.pi / 2 < β)
  (h₄ : β < 0)
  (h₅ : real.cos (α - β) = -5 / 13)
  (h₆ : real.sin α = 4 / 5) :
  real.sin β = -56 / 65 := 
sorry

end sin_beta_value_l279_279747


namespace distance_to_other_focus_l279_279433

variable (P : { x // x ∈ ({ p : ℝ × ℝ | (p.1^2 / 144) + (p.2^2 / 36) = 1 }) })

theorem distance_to_other_focus :
  ∀ (F1 F2 : ℝ × ℝ),
  (dist P.val F1 = 10) →
  let a := 12 in
  let total_distance := 2 * a in
  dist P.val F2 = total_distance - 10 := 
by
  intros,
  let a := 12,
  let total_distance := 2 * a,
  let known_distance := 10,
  have h : dist P.val F2 = total_distance - known_distance, sorry,
  exact h

end distance_to_other_focus_l279_279433


namespace rate_of_change_l279_279634

noncomputable def radius : ℝ := 12
noncomputable def θ (t : ℝ) : ℝ := (38 + 5 * t) * (Real.pi / 180)
noncomputable def area (t : ℝ) : ℝ := (1/2) * radius^2 * θ t

theorem rate_of_change (t : ℝ) : deriv area t = 2 * Real.pi :=
by
  sorry

end rate_of_change_l279_279634


namespace triangle_projection_relation_l279_279719

variables (a b c q : ℝ)

-- Define the conditions
def condition1 := a > b ∧ b > c
def condition2 := a = 2 * (b - c)
def condition3 (q : ℝ) := q = (projection_of_side c on_side a)

-- The theorem to prove
theorem triangle_projection_relation (a b c q : ℝ) 
  (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 q a c) : 
  c + 2 * q = (3 * a) / 4 :=
sorry

end triangle_projection_relation_l279_279719


namespace sum_max_min_values_l279_279955

noncomputable def f (x : ℝ) : ℝ := 
    1 - (Real.sin x / (x^4 + 2*x^2 + 1))

theorem sum_max_min_values : 
    let max_val := Real.sup (Set.range f)
    let min_val := Real.inf (Set.range f)
    max_val + min_val = 2 :=
by 
    sorry

end sum_max_min_values_l279_279955


namespace fermats_little_theorem_l279_279260

theorem fermats_little_theorem 
  (a n : ℕ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < n) 
  (h₃ : Nat.gcd a n = 1) 
  (phi : ℕ := (Nat.totient n)) 
  : n ∣ (a ^ phi - 1) := sorry

end fermats_little_theorem_l279_279260


namespace sqrt_multiplication_division_l279_279316

theorem sqrt_multiplication_division :
  Real.sqrt 27 * Real.sqrt (8 / 3) / Real.sqrt (1 / 2) = 18 :=
by
  sorry

end sqrt_multiplication_division_l279_279316


namespace even_function_expression_l279_279992

theorem even_function_expression (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≥ 0 → f x = x^2 - 3 * x + 4)
  (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = if x < 0 then x^2 + 3 * x + 4 else x^2 - 3 * x + 4 :=
by {
  sorry
}

end even_function_expression_l279_279992


namespace necessary_but_not_sufficient_condition_l279_279752

theorem necessary_but_not_sufficient_condition (a b : ℝ) (h : (a + b * Complex.i)^2 = 2 * Complex.i) : 
  (a = 1 ∧ b = 1) ↔ False :=
by
  sorry

end necessary_but_not_sufficient_condition_l279_279752


namespace triangle_medians_equal_legs_l279_279846

variables {A B C A1 B1 : Type} [EuclideanGeometry A B C A1 B1]

theorem triangle_medians_equal_legs
  (h_median_AA1 : is_median A A1)
  (h_median_BB1 : is_median B B1)
  (h_angles_equal : angle C A A1 = angle C B B1) :
  distance A C = distance B C := 
sorry

end triangle_medians_equal_legs_l279_279846


namespace turtle_marathon_time_l279_279929

/-- Given a marathon distance of 42 kilometers and 195 meters and a turtle's speed of 15 meters per minute,
prove that the turtle will reach the finish line in 1 day, 22 hours, and 53 minutes. -/
theorem turtle_marathon_time :
  let speed := 15 -- meters per minute
  let distance_km := 42 -- kilometers
  let distance_m := 195 -- meters
  let total_distance := distance_km * 1000 + distance_m -- total distance in meters
  let time_min := total_distance / speed -- time to complete the marathon in minutes
  let hours := time_min / 60 -- time to complete the marathon in hours (division and modulus)
  let minutes := time_min % 60 -- remaining minutes after converting total minutes to hours
  let days := hours / 24 -- time to complete the marathon in days (division and modulus)
  let remaining_hours := hours % 24 -- remaining hours after converting total hours to days
  (days, remaining_hours, minutes) = (1, 22, 53) -- expected result
:= 
sorry

end turtle_marathon_time_l279_279929


namespace probability_area_less_perimeter_l279_279646

def roll_dice_sum (d1 d2 : ℕ) : ℕ := d1 + d2
def is_valid_sum (s : ℕ) : Prop := 2 ≤ s ∧ s ≤ 12
def s_satisfies_condition (s : ℕ) : Prop := s < 4
def count_ways (n : ℕ) : ℕ := 
  -- count ways to roll a sum n with two 6-sided dice
  list_length [ (d1, d2) | d1 ∈ range 1 7, d2 ∈ range 1 7, roll_dice_sum d1 d2 = n ]

theorem probability_area_less_perimeter : 
  (∑ s in (range 2 13), if s_satisfies_condition s then count_ways s else 0) / 36 = 1 / 12 := 
by sorry

end probability_area_less_perimeter_l279_279646


namespace moles_of_Br2_combined_l279_279358

-- Definition of the reaction relation
def chemical_reaction (CH4 Br2 CH3Br HBr : ℕ) : Prop :=
  CH4 = 1 ∧ HBr = 1

-- Statement of the proof problem
theorem moles_of_Br2_combined (CH4 Br2 CH3Br HBr : ℕ) (h : chemical_reaction CH4 Br2 CH3Br HBr) : Br2 = 1 :=
by
  sorry

end moles_of_Br2_combined_l279_279358


namespace range_of_omega_for_zeros_in_interval_l279_279782

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - 1

theorem range_of_omega_for_zeros_in_interval (ω : ℝ) (hω_positve : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f ω x = 0 → 2 ≤ ω ∧ ω < 3) :=
sorry

end range_of_omega_for_zeros_in_interval_l279_279782


namespace problem1_problem2_l279_279862

variable {a b c : ℝ}

-- Problem 1: Prove that a^2 + b^2 + c^2 >= 1/3 given a, b, c are positive reals and a + b + c = 1.
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry 

-- Problem 2: Prove that sqrt((a^2 + b^2 + c^2) / 3) >= (a + b + c) / 3 given a, b, c are positive reals and a + b + c > 0.
theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c > 0) : sqrt((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := 
sorry 

end problem1_problem2_l279_279862


namespace round_nearest_l279_279524

theorem round_nearest (x : ℝ) (h : x = 7564.49999997) : Real.floor (x + 0.5) = 7564 := by
  sorry

end round_nearest_l279_279524


namespace option_D_correct_l279_279239

theorem option_D_correct (a b : ℝ) : -a * b + 3 * b * a = 2 * a * b :=
by sorry

end option_D_correct_l279_279239


namespace symmetry_distance_l279_279405

theorem symmetry_distance (m : ℝ) 
  (h1 : ∃ a b, a ≠ b ∧ (-a^2 + m^2 * a = 0 ∧ -b^2 + m^2 * b = 0))
  (h2 : ∃ c d, c ≠ d ∧ (c^2 - m^2 = 0 ∧ d^2 - m^2 = 0))
  (h3 : ∀ x1 x2 x3 x4, (x1, x2, x3, x4 ∈ {a, b, c, d}) ∧ (|x2 - x1| = |x3 - x2| ∧ |x4 - x3| = |x3 - x2|)) 
  : |(m^2 / 2) - 0| = 2 := 
by
  sorry

end symmetry_distance_l279_279405


namespace four_digit_numbers_l279_279276

open Finset

theorem four_digit_numbers :
  let digits := {1, 2, 3, 5}
  let digit_list := (digits : Finset ℕ).to_list
  let permutations := digit_list.permutations
  ∃ total_numbers odd_numbers even_numbers numbers_gt_2000,
  (total_numbers = permutations.length) ∧
  (odd_numbers = permutations.count (λ l, l[3] ∈ {1, 3, 5})) ∧
  (even_numbers = permutations.count (λ l, l[3] = 2)) ∧
  (numbers_gt_2000 = permutations.count (λ l, l[0] ∈ {2, 3, 5})) ∧
  (total_numbers = 24) ∧
  (odd_numbers = 18) ∧
  (even_numbers = 6) ∧
  (numbers_gt_2000 = 18) :=
by
  sorry

end four_digit_numbers_l279_279276


namespace problem_statement_l279_279874

-- Definitions based on given conditions
def A : set ℤ := {x | -5 ≤ x ∧ x ≤ 3}
def B : set ℤ := {x | x < -2 ∨ x > 4}
def N : set ℤ := {n | n ∈ ℕ}

-- Required to define C_N(B), the complement of B in N
def C_N_B : set ℤ := N \ B

-- The theorem to prove
theorem problem_statement : A ∩ C_N_B = {0, 1, 2, 3} := 
by
  sorry

end problem_statement_l279_279874


namespace circles_tangent_internally_l279_279944

theorem circles_tangent_internally :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x - 16 = 0) ∧ (x^2 + (y + 1)^2 = 5) →
  ∃ (d : ℝ), d = sqrt ((2-0)^2 + (0-(-1))^2) ∧ d = sqrt 5 ∧ sqrt 20 > sqrt 5 ∧ relationship = "tangent_internally" := 
begin
  sorry
end

end circles_tangent_internally_l279_279944


namespace smallest_number_is_neg2_l279_279664

theorem smallest_number_is_neg2 :
  ∀ (a b c d : ℝ), a = 5 → b = -1/3 → c = 0 → d = -2 →
  ∃ x, x = d ∧ ∀ y, (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≤ y :=
by
  intros a b c d ha hb hc hd
  use d
  split
  · exact hd
  intros y hy
  cases hy
  · rw [hy, ha]
    linarith
  cases hy
  · rw [hy, hb]
    linarith
  cases hy
  · rw [hy, hc]
    linarith
  · rw [hy]
    exact le_refl d

end smallest_number_is_neg2_l279_279664


namespace students_scores_between_120_and_130_l279_279738

noncomputable def normal_distribution_example (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∀ x, (x = 100 → X x = 0.6826) ∧ (x = 90 → X x = 0.9544)

theorem students_scores_between_120_and_130 :
  ∀ (μ σ : ℝ) (X : ℝ → ℝ), 
  (normal_distribution_example μ σ X)
   ∧ μ = 110
   ∧ σ^2 = 100
   → 
   let prob := 1 / 2 * (0.9544 - 0.6826) in
   let num_students := 60 * prob in
   abs (num_students - 8) < 1 := 
sorry

end students_scores_between_120_and_130_l279_279738


namespace P_has_real_root_l279_279744

def P : ℝ → ℝ := sorry
variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom a1_nonzero : a1 ≠ 0
axiom a2_nonzero : a2 ≠ 0
axiom a3_nonzero : a3 ≠ 0

axiom functional_eq (x : ℝ) :
  P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem P_has_real_root :
  ∃ x : ℝ, P x = 0 :=
sorry

end P_has_real_root_l279_279744


namespace polynomial_expansion_proof_l279_279746

theorem polynomial_expansion_proof :
  let a : Fin 2017 → ℝ := λ i, (1 - 2 ^ i) ^ 2016 →  
  a 0 = 1 →
  (a 0 + ∑ i in Finset.range 2016, (a (i + 1)) / (2 ^ (i + 1))) = -1 :=
by 
  intros a h0,
  have : let s := (1 - 2 * (1/2))^2016 in
        s = 0,
  sorry

end polynomial_expansion_proof_l279_279746


namespace range_of_a_l279_279576

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x + 3| + |x - 1| < a^2 - 3a)) ↔ (-1 ≤ a ∧ a ≤ 4) := 
by
  sorry

end range_of_a_l279_279576


namespace proof_problem_l279_279373

-- Given sequence properties
def is_non_constant (a : ℕ → ℝ) : Prop := ∀ n, a n ≠ a (n + 1)
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (1 / 4) * (a n ^ 2 + 4 * n - 1)

-- Given definitions from the problem
def a_property (a : ℕ → ℝ) : Prop := ∀ n, a n > 0
def S_property (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := sum_of_first_n_terms a S

-- Define sequences b_n and their sum T_n
def b_seq (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 / (a n * a (n - 1))
def T_seq (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  (∀ n (h : n > 0), T n = (finset.range n).sum (λ k, b (k + 1))) ∧ (∀ n, T 0 = 0)

-- Main statement to be proved
theorem proof_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) (h_a : a_property a) (h_S : S_property a S) :
  (∀ n, a n = 2 * n + 1 ∨ a n = 2 * n - 1) ∧
  (∀ b (n: ℕ),
    (a n = 2 * n - 1 → b n = 1 / (2 * n - 3) - 1 / (2 * n - 1) ∧ T n = 1 - 1 / (2 * n - 1)) ∧
    (a n = 2 * n + 1 → b n = 1 / (2 * n - 1) - 1 / (2 * n + 1) ∧ T n = 1 - 1 / (2 * n + 1))
  ) :=
begin
  sorry
end

end proof_problem_l279_279373


namespace men_in_first_group_l279_279911

theorem men_in_first_group (M : ℕ) (h1 : M * 35 = 7 * 50) : M = 10 := by
  sorry

end men_in_first_group_l279_279911


namespace part1_part2_l279_279044

open Real

noncomputable def f (a x : ℝ) : ℝ := log (x^2 - 2 * (2 * a - 1) * x + 8) / log (1/2)

theorem part1 (a : ℝ) : f a x is_decreasing_on Ici a ↔ -3/4 < a ∧ a ≤ 1 := 
by sorry

noncomputable def h (a x : ℝ) : ℝ := x^2 - 2 * (2 * a - 1) * x + 8

theorem part2 (a : ℝ) (x : ℝ) : f a x = log (x + 3) / log (1/2) - 1 ∧ x ∈ Ioo 1 3 ↔
  (3/4 ≤ a ∧ a < 11/12) ∨ (a = sqrt 2 / 2) := 
by sorry

end part1_part2_l279_279044


namespace entrance_ticket_cost_is_five_l279_279101

noncomputable def entrance_ticket_cost : ℕ :=
let
  total_paid := 55,
  kids_count := 4,
  parents_count := 2,
  grandmother_count := 1,
  attraction_cost_kid := 2,
  attraction_cost_adult := 4,
  kids_attraction_cost := kids_count * attraction_cost_kid,
  adults_attraction_cost := (parents_count + grandmother_count) * attraction_cost_adult,
  total_attraction_cost := kids_attraction_cost + adults_attraction_cost,
  total_entrance_cost := total_paid - total_attraction_cost,
  total_people := kids_count + parents_count + grandmother_count in
total_entrance_cost / total_people

theorem entrance_ticket_cost_is_five : entrance_ticket_cost = 5 := by
  sorry

end entrance_ticket_cost_is_five_l279_279101


namespace find_a_l279_279806

def perpendicular_lines (a : ℝ) : Prop :=
  let l1 := λ (x y : ℝ), a * x + 2 * y = 0
  let l2 := λ (x y : ℝ), (a - 1) * x - y = 0
  ∃ (x1 y1 x2 y2 : ℝ), l1 x1 y1 ∧ l2 x2 y2 ∧ (x1 - x2) * (y1 - y2) = 0

theorem find_a (a : ℝ) (h : perpendicular_lines a) : a = 2 ∨ a = -1 :=
sorry

end find_a_l279_279806


namespace even_function_a_zero_l279_279078

theorem even_function_a_zero (f : ℝ → ℝ) (a : ℝ) (h : ∀ x, f(-x) = f(x)) :
  f = (λ x : ℝ, x^2 - |x + a|) → a = 0 :=
begin
  sorry
end

end even_function_a_zero_l279_279078


namespace ratio_AB_CD_lengths_AB_CD_l279_279002

-- Definitions of points and conditions
variables (A B C D P Q M N : Type)
variables [ConvexQuadrilateral A B C D]
variables [InscribedCircleCenter P A B D]
variables [InscribedCircleCenter Q C B D]
variables [IntersectsBPAtM M B P A D (8/3)]
variables [IntersectsDQAtN N D Q B C (9/5)]
variables (AM DM BN CN : ℝ) 
variable (valid_M : AM = 8/3 ∧ DM = 4/3)
variable (valid_N : BN = 6/5 ∧ CN = 9/5)

-- Required proof of ratio AB:CD
theorem ratio_AB_CD : (AB : CD) = (4 : 3) :=
sorry

-- Assuming inscribed circles are tangent and proving side lengths
variables [TangentCircles (Circle P) (Circle Q)]
theorem lengths_AB_CD : AB = 4 ∧ CD = 3 :=
sorry

end ratio_AB_CD_lengths_AB_CD_l279_279002


namespace magnitude_of_a_minus_2b_l279_279058

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (theta : ℝ)

-- Given conditions
def angle_between_vectors := θ = π / 3
def magnitude_of_a := ∥a∥ = 1
def magnitude_of_b := ∥b∥ = 1 / 2

-- The proof problem
theorem magnitude_of_a_minus_2b
  (h1 : angle_between_vectors theta)
  (h2 : magnitude_of_a a)
  (h3 : magnitude_of_b b) :
  ∥a - (2: ℝ) • b∥ = 1 := 
sorry

end magnitude_of_a_minus_2b_l279_279058


namespace fraction_of_area_above_line_l279_279203

theorem fraction_of_area_above_line : 
  let square_vertices := [(0,0), (0,4), (4,0), (4,4)],
      p1 := (1, 3),
      p2 := (5, 1) in
  let slope := (p2.2 - p1.2) / (p2.1 - p1.1),
      intercept := p1.2 - slope * p1.1,
      line := (x: ℝ) → slope * x + intercept,
      square_area := 16,
      area_above_line := (square_area - 6) / 16 in
  area_above_line = (5 / 8) := sorry

end fraction_of_area_above_line_l279_279203


namespace heads_matching_probability_l279_279468

-- Define the experiment and outcomes
def keiko_outcomes := {tt, ff} -- Keiko can get heads (tt) or tails (ff)
def ephraim_outcomes := { -- Ephraim's three coin outcomes
  (tt, tt, tt), (tt, tt, ff), (tt, ff, tt), (tt, ff, ff),
  (ff, tt, tt), (ff, tt, ff), (ff, ff, tt), (ff, ff, ff)
}
def linda_outcomes := {tt, ff} -- Linda can get heads (tt) or tails (ff)

-- Calculate the combined outcomes
def combined_outcomes : Set (Bool × (Bool × Bool × Bool) × Bool) :=
  { (k, (e1, e2, e3), l) | k ∈ keiko_outcomes ∧ (e1, e2, e3) ∈ ephraim_outcomes ∧ l ∈ linda_outcomes }

-- Calculate the number of matching outcomes
def count_matching_outcomes : ℕ :=
  combined_outcomes.count (λ (k, (e1, e2, e3), l) => (k.b2n = (e1.b2n + e2.b2n + e3.b2n + l.b2n)))

-- Total number of outcomes
def total_outcomes : ℕ := keiko_outcomes.card * ephraim_outcomes.card * linda_outcomes.card

-- Calculate the probability
def probability := (count_matching_outcomes.to_rat / total_outcomes.to_rat)

-- The proof statement
theorem heads_matching_probability : probability = 5 / 32 :=
by
  sorry

end heads_matching_probability_l279_279468


namespace original_pumpkins_count_l279_279889

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end original_pumpkins_count_l279_279889


namespace inequality_solution_l279_279392

theorem inequality_solution (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1 / x + 4 / y) ≥ 9 / 4 := 
sorry

end inequality_solution_l279_279392


namespace prob_third_red_off_rightmost_blue_on_l279_279525

-- Define the given scenario conditions
def total_lamps := 8
def red_lamps := 4
def blue_lamps := 4
def lamps_on := 4

-- Define the binomial coefficient function (as needed for calculation)
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Calculate the probability
def prob_scenario : ℚ :=
  (binom 6 3 * binom 6 3 : ℚ) / (binom total_lamps red_lamps * binom total_lamps lamps_on : ℚ)

-- Define the theorem to check equivalence
theorem prob_third_red_off_rightmost_blue_on :
  prob_scenario = (8 / 98 : ℚ) :=
begin
  sorry
end

end prob_third_red_off_rightmost_blue_on_l279_279525


namespace daniela_total_spent_l279_279694

-- Step d) Rewrite the math proof problem
theorem daniela_total_spent
    (shoe_price : ℤ) (dress_price : ℤ) (shoe_discount : ℤ) (dress_discount : ℤ)
    (shoe_count : ℤ)
    (shoe_original_price : shoe_price = 50)
    (dress_original_price : dress_price = 100)
    (shoe_discount_rate : shoe_discount = 40)
    (dress_discount_rate : dress_discount = 20)
    (shoe_total_count : shoe_count = 2)
    : shoe_count * (shoe_price - (shoe_price * shoe_discount / 100)) + (dress_price - (dress_price * dress_discount / 100)) = 140 := by 
    sorry

end daniela_total_spent_l279_279694


namespace sufficient_but_not_necessary_l279_279420

theorem sufficient_but_not_necessary {a b : ℝ} (h₁ : a < b) (h₂ : b < 0) : 
  (a^2 > b^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by
  sorry

end sufficient_but_not_necessary_l279_279420


namespace find_sum_l279_279729

variable (a b c d : ℝ)

theorem find_sum :
  (ab + bc + cd + da = 20) →
  (b + d = 4) →
  (a + c = 5) := by
  sorry

end find_sum_l279_279729


namespace binomial_coefficient_sum_l279_279841

theorem binomial_coefficient_sum :
  let f : ℕ × ℕ → ℕ := λ (m n : ℕ), choose 6 (m-4+n) * 2^(6-(m-4+n)) * choose 4 n in
  f (3, 4) + f (5, 3) = 400 :=
by
  sorry

end binomial_coefficient_sum_l279_279841


namespace central_angle_star_in_polygon_l279_279093

theorem central_angle_star_in_polygon (n : ℕ) (h : 2 < n) : 
  ∃ C, C = 720 / n :=
by sorry

end central_angle_star_in_polygon_l279_279093


namespace quadratic_roots_determine_c_l279_279080

theorem quadratic_roots_determine_c (c : ℝ) :
  (∀ x, x = (-20 + real.sqrt 16) / 8 ∨ x = (-20 - real.sqrt 16) / 8 ↔ x^2 + 5 * x + c / 4 = 0) → c = 24 :=
by
  sorry

end quadratic_roots_determine_c_l279_279080


namespace relationship_y1_y2_y3_l279_279517

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end relationship_y1_y2_y3_l279_279517


namespace expected_points_correct_l279_279293

noncomputable def expected_points : ℕ :=
  let n := 13 in
  let p := 5 / 12 in
  (n - 1) * p

theorem expected_points_correct : expected_points = 5 := by
  -- Define the number of rolls
  let n := 13
  -- Define the probability of gaining a point on a single roll after the first
  let p := 5 / 12
  -- Calculate the expected number of points
  have : expected_points = (n - 1) * p := rfl
  -- Simplify and verify the result
  have : (n - 1) * p = 12 * (5 / 12) := by
    simp [n, p]
  show expected_points = 5 from by
    rw this
    norm_num
    sorry

end expected_points_correct_l279_279293


namespace find_expression_l279_279016

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end find_expression_l279_279016


namespace students_in_neither_l279_279085

def total_students := 60
def students_in_art := 40
def students_in_music := 30
def students_in_both := 15

theorem students_in_neither : total_students - (students_in_art - students_in_both + students_in_music - students_in_both + students_in_both) = 5 :=
by
  sorry

end students_in_neither_l279_279085


namespace segment_ratio_l279_279111

open EuclideanGeometry

variables {A B C H T M: Point} (β : Real) (circ : Circle) (ac : Line)

def right_triangle_median (H B C M : Point) :=
  right_angle B C H ∧
  collinear H M C ∧ collinear B M C ∧
  dist H M = dist B M ∧ dist H M = dist M C

def angle_conditions (A O C: Point) (β : Real) : Prop :=
  angle A O C = 2 * β

def circle_properties (H T A C: Point) (circ : Circle) : Prop :=
  on_circle H circ ∧ on_circle T circ ∧ diameter A C circ
  
theorem segment_ratio
  (β : Real)
  (h_angle : angle_condition A O C β)
  (h_circle: circle_properties H T A C circ)
  (h_median: right_triangle_median H B C M) :
  ratio (segment B M) (segment M C) = 1 :=
sorry

end segment_ratio_l279_279111


namespace articles_listed_percent_above_cost_price_l279_279618

theorem articles_listed_percent_above_cost_price {cp sp lp : ℝ} (h1 : cp = 1)
  (h2 : sp = (20 * cp) * (25 / 20))
  (h3 : lp = sp / 0.85)
  (h4 : 36% = ((sp - (25 * cp)) / (25 * cp)))
  (h5 : 15% = ((lp - sp) / lp)) :
  (lp / 25 - cp) / cp * 100 = 60 := 
sorry

end articles_listed_percent_above_cost_price_l279_279618


namespace ray_perpendicular_exists_l279_279723

theorem ray_perpendicular_exists 
  {O : Point} {A B : Plane} {r : Ray} 
  (h_dihedral : O ∈ A ∩ B) 
  (h_ray_in_plane_A : r ∈ A ∧ O ∈ r) 
  : ∃ (s : Ray), (s ∈ B ∧ O ∈ s ∧ s ⊥ r) :=
sorry

end ray_perpendicular_exists_l279_279723


namespace ellipse_equation_max_area_triangle_l279_279009

-- Given:
-- Ellipse foci F₁(-1, 0) and F₂(1, 0)
-- Point P(-1, √2 / 2) is on the ellipse
-- Parabola y² = 2px intersects ellipse at M and N
-- O is the origin

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Defining such that the left-hand side becomes true by conditions.
def ellipse_eqn (a b : ℝ) (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

def ellipse_foci (a : ℝ) : Prop := dist (-1, 0) (1, 0) = 2 * a 

-- Point P(-1, sqrt(2)/2) lies on the ellipse
def point_P_on_ellipse (a b : ℝ) : Prop := ellipse_eqn a b (-1) (sqrt 2/2)

-- Parabola y² = 2px intersects ellipse at M and N
def parabola_eqn (p x y : ℝ) := y^2 = 2 * p * x

-- Prove the equation of ellipse C
theorem ellipse_equation :
  ∃ a b : ℝ, 
    ellipse_foci a ∧ 
    point_P_on_ellipse a b ∧ 
    (∀ x y, ellipse_eqn a b x y) :=
by 
  use sqrt 2
  use 1
  split
  { sorry }
  split
  { sorry }
  { intros x y
    exact sorry }

-- Maximum area of triangle OMN when parabola intersects the ellipse
theorem max_area_triangle (p : ℝ) (x₀ y₀ : ℝ) : 
  (p > 0) ∧ 
  ellipse_eqn (sqrt 2) 1 x₀ y₀ ∧ parabola_eqn p x₀ y₀ ∧ 
  ∀ x y, ellipse_eqn (sqrt 2) 1 x y → parabola_eqn p x y → 
  (x * y ≤ x₀ * y₀) :=
by 
  use 1/4
  use 1
  use sqrt 2 / 2
  split
  { sorry }
  split
  { exact sorry }
  split
  { exact sorry }
  { intros
    exact sorry }

end ellipse_equation_max_area_triangle_l279_279009


namespace triangle_problem_l279_279109

-- We need basic trigonometry and geometry principles
-- First, we set up the conditions as definitions

variable {a b c : ℝ}
variable {A B C : ℝ}

-- conditions in the problem
def is_triangle (A B C : ℝ) : Prop := A + B + C = π
def law_of_sines (a b c A B C : ℝ) : Prop := 
  a / sin A = b / sin B ∧ 
  a / sin A = c / sin C

def given_equation (a b c A B C : ℝ) : Prop := 
  2 * a * sin A = (2 * b + c) * sin B + (2 * c + b) * sin C

def distance_condition (a : ℝ) : ℝ := 
  7

def distance_from_A_to_BC (d : ℝ) (a b c A B C : ℝ) : Prop :=
  b * c * sin A = d

-- The proof problem statement
theorem triangle_problem :
  is_triangle A B C →
  law_of_sines a b c A B C →
  given_equation a b c A B C →
  (A = 2 * π / 3) ∧ 
  (distance_from_A_to_BC (15 * sqrt 3 / 14) a 7 b c A B C → 
    (b = 3 ∧ c = 5) ∨ (b = 5 ∧ c = 3)) :=
by
  intros
  sorry

end triangle_problem_l279_279109


namespace gcd_876543_765432_l279_279227

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 1 :=
by {
  sorry
}

end gcd_876543_765432_l279_279227


namespace sister_age_0_l279_279925

variables (x y t1 t2 : ℕ)

def age_relation1 (x y t1 : ℕ) : Prop := t1 + x = x + y ∧ t1 = y
def age_relation2 (x y t2 : ℕ) : Prop := y = y - t2 ∧ t2 = x

theorem sister_age_0 (x y t1 t2 : ℕ) 
  (h1 : age_relation1 x y t1) 
  (h2 : age_relation2 x y t2) : 
  (y - t2 = 0) :=
begin
  -- This is where the proof would go.
  sorry -- Placeholder for the actual proof
end

end sister_age_0_l279_279925


namespace value_of_a_l279_279765

noncomputable def f (x φ : ℝ) : ℝ := sin ((x - 2 * φ) * π) + cos ((x - 2 * φ) * π)

theorem value_of_a {a φ : ℝ} (h1 : f x φ = sin ((x - 2 * φ) * π) + cos ((x - 2 * φ) * π))
  (h2 : ∀ x, f (-x) φ = -f x φ)
  (h3 : | log a φ | < 1) :
  (8 / 13 < a ∧ a < 5 / 8) ∨ (8 / 5 < a ∧ a < 13 / 8) := 
sorry

end value_of_a_l279_279765


namespace area_swept_by_minute_hand_l279_279939

-- Define the inputs of the problem
def minuteHandLength : ℝ := 6
def arcLength : ℝ := 25.12

-- Define the expected output
def expectedArea : ℝ := 75.36

-- The theorem stating the expected area swept by the minute hand
theorem area_swept_by_minute_hand :
  let radius := minuteHandLength
  let length := arcLength
  -- Convert the known arc length into the central angle in degrees
  let centralAngle := (180 * length) / (π * radius)
  -- Compute the area of the sector with the given central angle
  let area := (centralAngle / 360) * π * (radius ^ 2) in
  area = expectedArea :=
by
  sorry

end area_swept_by_minute_hand_l279_279939


namespace sticks_difference_l279_279333

def sticks_picked_up : ℕ := 14
def sticks_left : ℕ := 4

theorem sticks_difference : (sticks_picked_up - sticks_left) = 10 := by
  sorry

end sticks_difference_l279_279333


namespace smallest_prime_factor_of_setC_l279_279529

def setC : Set ℕ := {51, 53, 54, 56, 57}

def prime_factors (n : ℕ) : Set ℕ :=
  { p | p.Prime ∧ p ∣ n }

theorem smallest_prime_factor_of_setC :
  (∃ n ∈ setC, ∀ m ∈ setC, ∀ p ∈ prime_factors n, ∀ q ∈ prime_factors m, p ≤ q) ∧
  (∃ m ∈ setC, ∀ p ∈ prime_factors 54, ∀ q ∈ prime_factors m, p = q) := 
sorry

end smallest_prime_factor_of_setC_l279_279529


namespace part1_part2_l279_279390

theorem part1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (2 / (1 + a)) + (2 / (1 + b)) + (2 / (1 + c)) :=
sorry

end part1_part2_l279_279390


namespace value_of_m_l279_279689

theorem value_of_m :
  ∃ (m : ℕ) (b : fin m → ℕ), 
  (strict_mono b) ∧ 
  (∀ i, b i ≥ 0) ∧ 
  (2 ^ 171 + 1) / (2 ^ 19 + 1) = (finset.range m).sum (λ i, 2 ^ (b i)) ∧ 
  m = 101 := 
by sorry

end value_of_m_l279_279689


namespace shift_sine_graph_l279_279897

theorem shift_sine_graph (x : ℝ) : 
  (∃ θ : ℝ, θ = (5 * Real.pi) / 4 ∧ 
  y = Real.sin (x - Real.pi / 4) → y = Real.sin (x + θ) 
  ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := sorry

end shift_sine_graph_l279_279897


namespace smallest_base10_integer_l279_279989

def is_valid_digit_base_6 (C : ℕ) : Prop := C ≤ 5
def is_valid_digit_base_8 (D : ℕ) : Prop := D ≤ 7

def CC_6_to_base10 (C : ℕ) : ℕ := 7 * C
def DD_8_to_base10 (D : ℕ) : ℕ := 9 * D

theorem smallest_base10_integer : ∃ C D : ℕ, 
  is_valid_digit_base_6 C ∧ 
  is_valid_digit_base_8 D ∧ 
  CC_6_to_base10 C = DD_8_to_base10 D ∧
  CC_6_to_base10 C = 63 := 
begin
  sorry
end

end smallest_base10_integer_l279_279989


namespace min_value_of_a_l279_279434

theorem min_value_of_a (a : ℝ) :
  (∃ x : ℝ, log ((1:ℝ) / 3) (a - 3^x) = x - 2) → a ≥ 6 := sorry

end min_value_of_a_l279_279434


namespace at_least_three_same_mistakes_l279_279584

structure Student :=
  (mistakes : Fin 13)

theorem at_least_three_same_mistakes (students : List Student)
  (h1 : students.length = 30)
  (h2 : ∃ s : Student, (s.mistakes = 12) ∧ (students.count s = 1)) 
  (h3 : ∀ s : Student, s.mistakes < 13 → (students.filter (λ t, t.mistakes = s.mistakes)).length ≤ 2) :
  ∃ n : Fin 13, 3 ≤ (students.filter (λ s, s.mistakes = n)).length :=
by
  sorry

end at_least_three_same_mistakes_l279_279584


namespace geometric_sequence_a4_l279_279442

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the geometric sequence

axiom a_2 : a 2 = -2
axiom a_6 : a 6 = -32
axiom geom_seq (n : ℕ) : a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geometric_sequence_a4 : a 4 = -8 := 
by
  sorry

end geometric_sequence_a4_l279_279442


namespace sequence_periodic_l279_279622

noncomputable def exists_N (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n+2) = abs (a (n+1)) - a n

theorem sequence_periodic (a : ℕ → ℝ) (h : exists_N a) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a (n+9) = a n :=
sorry

end sequence_periodic_l279_279622


namespace distance_squared_eq_circumradius_minus_2Rr_l279_279674

-- Define the context of the problem with given conditions
variables (R r d : ℝ)
variables (O I : Type) [HasDistance O I] [HasDistance I O]
variable (triangle : Triangle)
variable h1 : triangle.circumradius = R
variable h2 : triangle.inradius = r
variable h3 : triangle.circumcenter = O
variable h4 : triangle.incenter = I
variable h5 : dist O I = d

-- Define the theorem to prove
theorem distance_squared_eq_circumradius_minus_2Rr :
  d^2 = R^2 - 2 * R * r :=
sorry

end distance_squared_eq_circumradius_minus_2Rr_l279_279674


namespace talia_father_age_l279_279835

theorem talia_father_age 
  (t tf tm ta : ℕ) 
  (h1 : t + 7 = 20)
  (h2 : tm = 3 * t)
  (h3 : tf + 3 = tm)
  (h4 : ta = (tm - t) / 2)
  (h5 : ta + 2 = tf + 5) : 
  tf = 36 :=
by
  sorry

end talia_father_age_l279_279835


namespace A_and_C_work_rate_l279_279262

-- Definitions based on the conditions
def work := 1  -- Let’s normalize the work W to 1 unit for simplicity.

def A_work_rate : ℝ := work / 4  -- A's work rate is W/4 per hour.
def B_work_rate : ℝ := work / 4  -- B's work rate is W/4 per hour.
def B_and_C_work_rate : ℝ := work / 2  -- B and C's combined work rate is W/2 per hour.

-- The main theorem we want to prove:
theorem A_and_C_work_rate
  (A_rate : A_work_rate = work / 4)
  (B_and_C_rate : B_and_C_work_rate = work / 2)
  (B_rate : B_work_rate = work / 4) 
  : (A_work_rate + (B_and_C_work_rate - B_work_rate)) = work / 2
:= by
  sorry

end A_and_C_work_rate_l279_279262


namespace integral_value_l279_279581

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end integral_value_l279_279581


namespace cosine_AKB_l279_279837

-- Variables and assumptions according to the given conditions
variables (A B C D K : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited K]

-- Definition of the lengths and the given ratios
variables (AB BC AD DC BK KD : ℝ)
variables (ratio_AD_DC : AD / DC = 4 / 3)
variables (ratio_BK_KD : BK / KD = 1 / 3)

-- Definitions to indicate the relationships in the quadrilateral
variables (is_convex : convex A B C D)
variables (AB_eq_BC : AB = BC)
variables (DB_angle_bisector_ADC : is_angle_bisector D B A C)
variables (K_intersection_AC_BD : intersection A C B D K)

-- Define the goal statement
theorem cosine_AKB :
  ∃ (K : Type) [inhabited K], K_intersection_AC_BD →
  cos_angle_AKB = 1 / 4 :=
sorry

end cosine_AKB_l279_279837


namespace sum_of_positive_x_and_y_is_ten_l279_279477

theorem sum_of_positive_x_and_y_is_ten (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^3 + y^3 + (x + y)^3 + 30 * x * y = 2000) : 
  x + y = 10 :=
sorry

end sum_of_positive_x_and_y_is_ten_l279_279477


namespace find_phi_max_min_f_l279_279399

-- Define the function f
def f (x ϕ : ℝ) : ℝ := 2 * cos x * cos (x + ϕ)

-- Condition 1: f(π/3) = 1
def condition1 (ϕ : ℝ) : Prop := f (π / 3) ϕ = 1

-- Condition 2: The function f is increasing on the interval [0, π/4]
def condition2 (ϕ : ℝ) : Prop := ∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π / 4 → f x₁ ϕ < f x₂ ϕ

-- Condition 3: For all x in ℝ, f(x) ≥ f(2π/3)
def condition3 (ϕ : ℝ) : Prop := ∀ x, f x ϕ ≥ f (2 * π / 3) ϕ

-- Problem statement part (I): Find ϕ given Condition 1 or 3
theorem find_phi (ϕ : ℝ) : condition1 ϕ ∨ condition3 ϕ → ϕ = -π / 3 := sorry

-- Problem statement part (II): Find max and min values of f(x) on the interval [-π/2, 0], given ϕ = -π/3
theorem max_min_f (x : ℝ) (h : -π / 2 ≤ x ∧ x ≤ 0) : 
  let ϕ := -π / 3 in
  f x ϕ ≤ 1 ∧ f x ϕ ≥ -1 / 2 := sorry

end find_phi_max_min_f_l279_279399


namespace extreme_values_and_minimum_c_l279_279792

-- Given function definition
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * a * x - b / x + Real.log x

-- Main theorem statement
theorem extreme_values_and_minimum_c  (a b : ℝ) :
  (∀ x : ℝ, x = 1 → f x a b = 1)
  ∧ (∀ x : ℝ, x = 1/2 → f x a b = 1/2) →
  (a = -1/3 ∧ b = -1/3) ∧ (∃ x0 ∈ Icc (1/4 : ℝ) 2, ∀ c : ℝ, f x0 - c ≤ 0 → c ≥ -7/6 + Real.log 2) :=
by
  sorry

end extreme_values_and_minimum_c_l279_279792


namespace sin_double_angle_l279_279030

theorem sin_double_angle (α : ℝ) (h : ∃ (P : ℝ × ℝ), P.2 = -2 * P.1 ∧ ∃ θ : ℝ, tan θ = -2 ∧ θ = α) : sin (2 * α) = - 4 / 5 :=
by
  cases h with P hP
  cases hP with hP1 hP2
  cases hP2 with θ hθ
  cases hθ with hθ hα
  sorry

end sin_double_angle_l279_279030


namespace percentage_passed_two_topics_only_l279_279091

theorem percentage_passed_two_topics_only :
  let T := 2500 in
  let passed_all := 0.10 * T in
  let passed_none := 0.10 * T in
  let passed_one := 0.20 * (T - passed_all - passed_none) in
  let passed_four := 0.24 * T in
  let passed_three := 500 in
  let total_remaining := T - passed_all - passed_none - passed_one - passed_four - passed_three in
  let P := 100 * passed_two / total_remaining in
  abs (P - 50.02) < 0.01 := 
sorry

end percentage_passed_two_topics_only_l279_279091


namespace range_of_omega_l279_279770

theorem range_of_omega :
  ∀ (ω : ℝ), 
  (0 < ω) → 
  (∀ x, x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → cos (ω * x) - 1 = 0 → x ∈ {0, 2 * Real.pi, 4 * Real.pi}) →
  (2 ≤ ω ∧ ω < 3) :=
by
  intros ω hω_pos hzeros
  sorry

end range_of_omega_l279_279770


namespace base4_has_2_more_digits_than_base7_l279_279065

noncomputable def base_digits (n : ℕ) (b : ℕ) : ℕ :=
  if b > 1 then nat.log b n + 1 else 0

theorem base4_has_2_more_digits_than_base7 (n : ℕ) (h : n = 4563) :
  base_digits n 4 - base_digits n 7 = 2 := 
by
  rw h
  sorry

end base4_has_2_more_digits_than_base7_l279_279065


namespace bubble_sort_probability_r10_r25_l279_279540

theorem bubble_sort_probability_r10_r25 (n : ℕ) (r : ℕ → ℕ) :
  n = 50 ∧ (∀ i, 1 ≤ i ∧ i ≤ 50 → r i ≠ r (i + 1)) ∧ (∀ i j, i ≠ j → r i ≠ r j) →
  let p := 1
  let q := 650
  p + q = 651 :=
by
  intros h
  sorry

end bubble_sort_probability_r10_r25_l279_279540


namespace integer_roots_l279_279476

noncomputable def is_quadratic_root (p q x : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem integer_roots (p q x1 x2 : ℝ)
  (hq1 : is_quadratic_root p q x1)
  (hq2 : is_quadratic_root p q x2)
  (hd : x1 ≠ x2)
  (hx : |x1 - x2| = 1)
  (hpq : |p - q| = 1) :
  (∃ (p_int q_int x1_int x2_int : ℤ), 
      p = p_int ∧ q = q_int ∧ x1 = x1_int ∧ x2 = x2_int) :=
sorry

end integer_roots_l279_279476


namespace ratio_concyclic_points_l279_279478

variable {Point : Type*} [euclidean_geometry Point]
variables {A B C D E : Point}

-- Given conditions
axiom concyclic_points : concyclic {A, B, C, D}
axiom lines_intersect : ∃ E, collinear {A, B, E} ∧ collinear {C, D, E}

-- Statement to prove
theorem ratio_concyclic_points :
  (distance A C / distance B C) * (distance A D / distance B D) = distance A E / distance B E :=
sorry

end ratio_concyclic_points_l279_279478


namespace ac_bd_ge_one_l279_279628

theorem ac_bd_ge_one 
  (A B C D O : Type) 
  [inner_product_space ℝ A] 
  [inner_product_space ℝ B] 
  [inner_product_space ℝ C] 
  [inner_product_space ℝ D] 
  (hAB : dist A B = 1) 
  (hCD : dist C D = 1) 
  (hAOCOeq60deg : ∠ A O C = real.pi / 3) :
  dist A C + dist B D ≥ 1 :=
sorry

end ac_bd_ge_one_l279_279628


namespace exist_primitive_roots_l279_279138

open_locale big_operators

def is_primitive_root (k p : ℕ) : Prop :=
  (gcd (k, p - 1) = 1) ∧
  ∀ r ∈ (nat.divisors (p - 1)).erase (p - 1), ¬ (k ^ r ≡ 1 [MOD p])

theorem exist_primitive_roots 
  (p : ℕ) 
  (h_prime : nat.prime p) 
  (h_form : ∃ k : ℕ, p = 4 * k + 1) 
  (h_phi_ineq : p - 1 < 3 * nat.totient (p - 1)) :
  ∃ k_set : finset ℕ, k_set.card ≥ (3 * nat.totient (p - 1) - (p - 1)) / 2 ∧
                     ∀ k ∈ k_set, is_primitive_root k p 
:= sorry

end exist_primitive_roots_l279_279138


namespace statement_A_statement_C_statement_D_statement_B_l279_279012

variable (a b : ℝ)

theorem statement_A :
  4 * a^2 - a * b + b^2 = 1 → |a| ≤ 2 * Real.sqrt 15 / 15 :=
sorry

theorem statement_C :
  (4 * a^2 - a * b + b^2 = 1) → 4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3 :=
sorry

theorem statement_D :
  4 * a^2 - a * b + b^2 = 1 → |2 * a - b| ≤ 2 * Real.sqrt 10 / 5 :=
sorry

theorem statement_B :
  4 * a^2 - a * b + b^2 = 1 → ¬(|a + b| < 1) :=
sorry

end statement_A_statement_C_statement_D_statement_B_l279_279012


namespace general_formula_for_a_n_l279_279070

-- Condition: f(x) + f(1-x) = 4
variable (f : ℝ → ℝ)
variable f_symmetric : ∀ x : ℝ, f(x) + f(1 - x) = 4

-- Definition of the sequence a_n
def a_n (n : ℕ+) : ℝ :=
  ∑ i in Finset.range (n + 1), f (i / n.to_real)

-- The proof goal
theorem general_formula_for_a_n (n : ℕ+) : a_n f n = 2 * (n + 1) := by
  sorry

end general_formula_for_a_n_l279_279070


namespace prob_a_sq_plus_b_sq_eq_25_prob_isosceles_triangle_l279_279164

-- Define the sample space of rolling a die twice
def sample_space := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]]

-- Part I: Prove the probability that a^2 + b^2 = 25
def event_a_sq_plus_b_sq_eq_25 (a b : ℕ) : Bool :=
  a ^ 2 + b ^ 2 = 25

theorem prob_a_sq_plus_b_sq_eq_25 :
  (∑ (s in sample_space) if event_a_sq_plus_b_sq_eq_25 s.1 s.2 then 1 else 0) / 36 = 1 / 18 := sorry

-- Part II: Prove the probability of forming an isosceles triangle with sides a, b, and 5
def is_isosceles_triangle (a b : ℕ) : Bool :=
  a = b ∨ b = 5 ∨ a = 5

theorem prob_isosceles_triangle :
  (∑ (s in sample_space) if is_isosceles_triangle s.1 s.2 then 1 else 0) / 36 = 7 / 18 := sorry

end prob_a_sq_plus_b_sq_eq_25_prob_isosceles_triangle_l279_279164


namespace ratio_of_incomes_l279_279947

theorem ratio_of_incomes 
  (E1 E2 I1 I2 : ℕ)
  (h1 : E1 / E2 = 3 / 2)
  (h2 : E1 = I1 - 1200)
  (h3 : E2 = I2 - 1200)
  (h4 : I1 = 3000) :
  I1 / I2 = 5 / 4 :=
sorry

end ratio_of_incomes_l279_279947


namespace negation_proposition_l279_279941

theorem negation_proposition (a b : ℝ) : 
  (¬ (a > b → 2^a > 2^b - 1)) = (a ≤ b → 2^a ≤ 2^b - 1) :=
by sorry

end negation_proposition_l279_279941


namespace find_max_n_l279_279428

-- Define condition for n to be a three-digit positive integer
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the condition that sum of first n integers is not a factor of the product of first n integers
def not_factorial_divisible_sum (n : ℕ) : Prop :=
  ¬ (n! % sum_first_n n = 0)

-- Lean statement asserting the maximum value of n satisfying the conditions
theorem find_max_n : ∃ (n : ℕ), is_three_digit n ∧ not_factorial_divisible_sum n ∧ ∀ m, (is_three_digit m ∧ not_factorial_divisible_sum m) → m ≤ n :=
  sorry

end find_max_n_l279_279428


namespace relationship_abc_l279_279751

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom increasing_f_on_negative : ∀ x y : ℝ, x < y → y ≤ 0 → f x < f y

-- Definitions of a, b, c
def a : ℝ := f (Real.log 7 / Real.log 4)
def b : ℝ := f (Real.log 3 / Real.log 2)
def c : ℝ := f (Real.exp (0.6 * Real.log 0.2))

-- Theorem stating the desired relationship
theorem relationship_abc : b < a ∧ a < c :=
by
  sorry

end relationship_abc_l279_279751


namespace beads_currently_have_l279_279676

-- Definitions of the conditions
def friends : Nat := 6
def beads_per_bracelet : Nat := 8
def additional_beads_needed : Nat := 12

-- Theorem statement
theorem beads_currently_have : (beads_per_bracelet * friends - additional_beads_needed) = 36 := by
  sorry

end beads_currently_have_l279_279676


namespace sin_theta_formula_l279_279105

variables (a b c : ℝ)
noncomputable def θ := dihedral_angle_between_planes a b c

theorem sin_theta_formula :
  sin θ = (a ^ 2 * b * c) / (sqrt (b ^ 2 * c ^ 2 + c ^ 2 * a ^ 2 + a ^ 2 * b ^ 2) * sqrt (a ^ 2 * b ^ 2 + 4 * b ^ 2 * c ^ 2 + 4 * c ^ 2 * a ^ 2)) :=
sorry

noncomputable def dihedral_angle_between_planes (a b c : ℝ) : ℝ := -- Placeholder, represent θ
sorry

end sin_theta_formula_l279_279105


namespace equilateral_triangle_third_vertex_y_coord_l279_279303

/-- 
   Theorem: Prove that if two vertices of an equilateral triangle are at (1,3) and (9,3)
   and the third vertex is in the first quadrant, then the y-coordinate of the third vertex is 3 + 4 * sqrt(3).
-/
theorem equilateral_triangle_third_vertex_y_coord : 
  ∃ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 64 ∧ (x - 9)^2 + (y - 3)^2 = 64 ∧ y = 3 + 4 * Real.sqrt 3 :=
begin
  sorry
end

end equilateral_triangle_third_vertex_y_coord_l279_279303


namespace domain_of_log_function_l279_279927

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - x) / Real.log 3

theorem domain_of_log_function :
  (∀ x : ℝ, f x ∈ ℝ ↔ x < 2) :=
by
  sorry

end domain_of_log_function_l279_279927


namespace cartesian_line_l_equiv_max_distance_P_l279_279098

-- Define the curve C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the polar equation of the line l
def polar_line_l (ρ θ : ℝ) : Prop := ρ * (2 * Real.cos θ - Real.sin θ) = 6

-- Define the Cartesian equation of the line l
def cartesian_line_l (x y : ℝ) : Prop := 2 * x - y = 6

-- Define the parametric equations of the curve C2 obtained by stretching C1
def parametric_C2 (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos θ, 2 * Real.sin θ)

-- Prove that the Cartesian equation of the line l is 2x - y - 6 = 0
theorem cartesian_line_l_equiv : ∀ ρ θ : ℝ, polar_line_l ρ θ ↔ cartesian_line_l (ρ * Real.cos θ) (ρ * Real.sin θ) :=
  by
  intros ρ θ
  rw [polar_line_l, cartesian_line_l]
  sorry

-- Prove that there exists a point P on C2 such that the distance to the line l is maximized
-- and calculate the maximum distance
theorem max_distance_P :
  (∃ θ : ℝ, parametric_C2 θ = (- sqrt 3 / 2, 1)) ∧
  (∀ θ : ℝ, (let (x, y) := parametric_C2 θ in ((2 * x - y - 6).abs / sqrt 5) ≤ 2 * sqrt 5) ∧
  ((2 * (- sqrt 3 / 2) - 1 - 6).abs / sqrt 5 = 2 * sqrt 5)) :=
  by
  sorry

end cartesian_line_l_equiv_max_distance_P_l279_279098


namespace problem_distance_from_point_to_line_PAB_l279_279801

noncomputable def dist_point_to_line (P A B : ℝ × ℝ × ℝ) : ℝ := 
let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
let AB_dot_AP := AB.1 * AP.1 + AB.2 * AP.2 + AB.3 * AP.3 in
let AB_mag := Real.sqrt (AB.1 * AB.1 + AB.2 * AB.2 + AB.3 * AB.3) in
let AP_mag := Real.sqrt (AP.1 * AP.1 + AP.2 * AP.2 + AP.3 * AP.3) in
let cos_theta := AB_dot_AP / (AB_mag * AP_mag) in
AP_mag * Real.sqrt (1 - cos_theta * cos_theta)

theorem problem_distance_from_point_to_line_PAB :
  dist_point_to_line (1, 1, 1) (1, 0, 1) (0, 1, 0) = Real.sqrt 6 / 3 :=
by sorry

end problem_distance_from_point_to_line_PAB_l279_279801


namespace log_expression_value_l279_279612

theorem log_expression_value :
  (log 2 64 / log 32 2) - (log 2 256 / log 16 2) = -2 :=
by
  have log_2_64 : log 2 64 = 6 := sorry
  have log_2_256 : log 2 256 = 8 := sorry
  have log_2_32 : log 2 32 = 5 := sorry
  have log_2_16 : log 2 16 = 4 := sorry
  have change_log_32 : log (32) 2 = 1 / log 2 32 := by simp [log, log_base_change]
  have change_log_16 : log (16) 2 = 1 / log 2 16 := by simp [log, log_base_change]
  sorry

end log_expression_value_l279_279612


namespace average_rate_of_change_l279_279039

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l279_279039


namespace basin_leak_rate_proof_l279_279626

def leak_rate (flow_rate : ℕ) (capacity : ℕ) (fill_time : ℕ) : ℕ :=
  (flow_rate * fill_time - capacity) / fill_time

theorem basin_leak_rate_proof :
  leak_rate 24 260 13 = 4 :=
by
  unfold leak_rate
  norm_num
  sorry

end basin_leak_rate_proof_l279_279626


namespace find_f_neg_15_over_2_l279_279742

def f (x : ℝ) : ℝ := sorry  -- Define the function f, but leave the implementation as sorry for now

axiom even_f : ∀ x, f x = f (-x)
axiom periodic_f : ∀ x, f x = f (x + 2)
axiom f_interval : ∀ x, x ≥ -1 ∧ x ≤ 0 → f x = 3^x

theorem find_f_neg_15_over_2 : f (-15 / 2) = sqrt(3) / 3 :=
by sorry  -- Proof goes here

end find_f_neg_15_over_2_l279_279742


namespace simplify_fraction_l279_279902

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l279_279902


namespace complex_quadrant_l279_279753

theorem complex_quadrant (i : ℂ) (z : ℂ) (h1 : i^2 = -1) (h2 : z * (1 + i) = 3 - i) : Re z > 0 ∧ Im z < 0 :=
by
  sorry

end complex_quadrant_l279_279753


namespace find_quadratic_function_find_g_minimum_l279_279374

theorem find_quadratic_function (a b c : ℝ) (h₁ : a ≠ 0) 
  (h₂ : ∀ x, f x < 0 ↔ 1 < x ∧ x < 2)
  (h₃ : f 3 = 2) :
  f x = x^2 - 3x + 2 := by
sorry

theorem find_g_minimum (m : ℝ) 
  (h₁ : ∀ x, f x < 0 ↔ 1 < x ∧ x < 2)
  (h₂ : f 3 = 2) :
  h m = if m ≤ -1 then -m 
        else if -1 < m ∧ m < 1 THEN -((3+m)^2)/4 +2
        else if m ≥ 1 THEN -2m := by
sorry

end find_quadratic_function_find_g_minimum_l279_279374


namespace eccentricity_of_hyperbola_l279_279943

-- Define the problem in terms of hyperbola, focus, tangent line and distances
variable {a b c : ℝ} (h_a_pos : a > 0) (h_b_pos : b > 0)
  (F : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ)
  
-- Given conditions
def is_focus : Prop := F = (c, 0)
def is_on_hyperbola : Prop := (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1
def is_tangent_to_circle : Prop := Q = ((P.1 + F.1) / 2, (P.2 + F.2) / 2) ∧ (Q.1)^2 + (Q.2)^2 = b^2 / 4
def midpoint_theorem_applied : Prop := (Q.1 - (F.1 / 2))^2 + (Q.2 - (F.2 / 2)) ≠ 0

-- Definition of eccentricity
def eccentricity : ℝ := c / a

-- The final proof problem statement: Prove the eccentricity is sqrt(5)
theorem eccentricity_of_hyperbola : 
  is_focus F → 
  is_on_hyperbola P → 
  is_tangent_to_circle P Q F → 
  midpoint_theorem_applied P Q → 
  eccentricity F = sqrt(5) :=
by
  sorry -- No proof required, just the statement

end eccentricity_of_hyperbola_l279_279943


namespace equal_sides_CA_CB_l279_279857

variable {A B C D E F G : Type} [euclidean_geometry.triangle A B C]

-- Angle bisectors meeting points
variable (AD_angle_bisector : euclidean_geometry.angle_bisector A B C = D)
variable (BF_angle_bisector : euclidean_geometry.angle_bisector B C A = F)

-- Lines meeting at E and G
variable (AD_meets_parallel : euclidean_geometry.line (euclidean_geometry.point A D) ∩ euclidean_geometry.parallel_to (euclidean_geometry.line A B) C = E)
variable (BF_meets_parallel : euclidean_geometry.line (euclidean_geometry.point B F) ∩ euclidean_geometry.parallel_to (euclidean_geometry.line A B) C = G)

-- Given condition
variable FG_equals_DE : euclidean_geometry.distance F G = euclidean_geometry.distance D E

theorem equal_sides_CA_CB (h : FG_equals_DE)
    (h_AD : AD_angle_bisector) (h_BF: BF_angle_bisector)
    (h_E : AD_meets_parallel) (h_G : BF_meets_parallel) :
    euclidean_geometry.distance C A = euclidean_geometry.distance C B := sorry

end equal_sides_CA_CB_l279_279857


namespace infinite_series_converges_l279_279322

open BigOperators

noncomputable def problem : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0

theorem infinite_series_converges : problem = 61 / 24 :=
sorry

end infinite_series_converges_l279_279322


namespace least_integer_with_remainders_l279_279230

theorem least_integer_with_remainders :
  ∃ M : ℕ, 
    M % 6 = 5 ∧
    M % 7 = 6 ∧
    M % 9 = 8 ∧
    M % 10 = 9 ∧
    M % 11 = 10 ∧
    M = 6929 :=
by
  sorry

end least_integer_with_remainders_l279_279230


namespace min_value_sequence_l279_279803

def sequence (a : ℕ → ℕ) : Prop := a 1 = 12 ∧ ∀ n, a (n + 1) - a n = 2 * n

theorem min_value_sequence (a : ℕ → ℕ) (h : sequence a) : ∃ n, n > 0 ∧ (a n) / n = 6 := 
sorry

end min_value_sequence_l279_279803


namespace tim_reading_hours_per_week_l279_279596

theorem tim_reading_hours_per_week :
  (meditation_hours_per_day = 1) →
  (reading_hours_per_day = 2 * meditation_hours_per_day) →
  (reading_hours_per_week = reading_hours_per_day * 7) →
  reading_hours_per_week = 14 :=
by
  intros h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  exact h3

end tim_reading_hours_per_week_l279_279596


namespace henry_total_cost_l279_279062

def henry_initial_figures : ℕ := 3
def henry_total_needed_figures : ℕ := 15
def cost_per_figure : ℕ := 12

theorem henry_total_cost :
  (henry_total_needed_figures - henry_initial_figures) * cost_per_figure = 144 :=
by
  sorry

end henry_total_cost_l279_279062


namespace find_a_plus_b_l279_279055

/-- Given the sets M = {x | |x-4| + |x-1| < 5} and N = {x | a < x < 6}, and M ∩ N = {2, b}, 
prove that a + b = 7. -/
theorem find_a_plus_b 
  (M : Set ℝ := { x | |x - 4| + |x - 1| < 5 }) 
  (N : Set ℝ := { x | a < x ∧ x < 6 }) 
  (a b : ℝ)
  (h_inter : M ∩ N = {2, b}) :
  a + b = 7 :=
sorry

end find_a_plus_b_l279_279055


namespace polygon_with_given_angle_sum_l279_279028

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end polygon_with_given_angle_sum_l279_279028


namespace segments_equal_iff_lemoine_point_l279_279217

theorem segments_equal_iff_lemoine_point (A B C X B₁ C₁ A₂ C₂ A₃ B₃ : Type*)
  [planeGeometry A B C X B₁ C₁ A₂ C₂ A₃ B₃]
  (h_antisym_parallel_1 : isAntiparallel B₁ C₁ A C)
  (h_antisym_parallel_2 : isAntiparallel C₂ A₂ A B)
  (h_antisym_parallel_3 : isAntiparallel A₃ B₃ B C)
  (h_abc_triangle : isTriangle A B C)
  (h_X_inside : isInInterior X A B C)
  (h_isosceles_1 : isIsosceles A₂ X A₃)
  (h_isosceles_2 : isIsosceles B₁ X B₃)
  (h_isosceles_3 : isIsosceles C₁ X C₂) :
  (length B₁ C₁ = length C₂ A₂ ∧ length C₂ A₂ = length A₃ B₃) ↔ isLemoinePoint X A B C :=
sorry

end segments_equal_iff_lemoine_point_l279_279217


namespace ratio_of_boys_to_girls_simplified_l279_279090

def number_of_boys : ℕ := 12
def number_of_girls : ℕ := 18

theorem ratio_of_boys_to_girls_simplified :
  let gcd_value := Nat.gcd number_of_boys number_of_girls in
  (number_of_boys / gcd_value) = 2 ∧ (number_of_girls / gcd_value) = 3 := by
  sorry

end ratio_of_boys_to_girls_simplified_l279_279090


namespace first_day_of_month_l279_279185

noncomputable def day_of_week := ℕ → ℕ

def is_wednesday (n : ℕ) : Prop := day_of_week n = 3

theorem first_day_of_month (day_of_week : day_of_week) (h : is_wednesday 30) : day_of_week 1 = 2 :=
by
  sorry

end first_day_of_month_l279_279185


namespace melissa_points_per_game_l279_279505

theorem melissa_points_per_game (total_points : ℕ) (games : ℕ) (h1 : total_points = 81) 
(h2 : games = 3) : total_points / games = 27 :=
by
  sorry

end melissa_points_per_game_l279_279505


namespace part_a_part_b_part_c_l279_279506

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def digits_greater_than_4 (n : ℕ) : ℕ :=
  n.digits.filter (λ d, d > 4).length

variable {M K : ℕ}

/-- Assumptions:
 1. M and K are natural numbers
 2. M and K differ by a permutation of digits
 -/
axiom perm_digits (h : List.permutations M.digits K.digits)

theorem part_a (hperm : List.permutations M.digits K.digits) : 
  sum_of_digits (2 * M) = sum_of_digits (2 * K) :=
sorry

theorem part_b (hevenM : Even M) (hevenK : Even K) (hperm : List.permutations M.digits K.digits) : 
  sum_of_digits (M / 2) = sum_of_digits (K / 2) :=
sorry

theorem part_c (hperm : List.permutations M.digits K.digits) : 
  sum_of_digits (5 * M) = sum_of_digits (5 * K) :=
sorry

end part_a_part_b_part_c_l279_279506


namespace row_swap_matrix_2x2_l279_279329

theorem row_swap_matrix_2x2 (a b c d : ℝ) :
  let N := Matrix.of ![![0, 1], ![1, 0]]
  let M := Matrix.of ![![a, b], ![c, d]]
  N ⬝ M = Matrix.of ![![c, d], ![a, b]] := 
by
  sorry

end row_swap_matrix_2x2_l279_279329


namespace range_of_omega_l279_279788

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ zeros : ℝ, (f(x) = cos (ω * x) - 1) and (count_zeros (f(x),  [0, 2 * π]) = 3)) ↔ (2 ≤ ω ∧ ω < 3) := 
sorry

end range_of_omega_l279_279788


namespace hyperbola_eccentricity_l279_279546

-- Definitions of conditions
def asymptotes_of_hyperbola (a b x y : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  (b * x + a * y = 0) ∨ (b * x - a * y = 0)

def circle_tangent_to_asymptotes (x y a b : ℝ) : Prop :=
  ∀ x1 y1 : ℝ, 
  (x1, y1) = (0, 4) → 
  (Real.sqrt (b^2 + a^2) = 2 * a)

-- Main statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_asymptotes : ∀ (x y : ℝ), asymptotes_of_hyperbola a b x y h_a h_b) 
  (h_tangent : circle_tangent_to_asymptotes 0 4 a b) : 
  ∃ e : ℝ, e = 2 := 
sorry

end hyperbola_eccentricity_l279_279546


namespace hyperbola_real_axis_length_correct_l279_279551

noncomputable def hyperbola_real_axis_length : ℝ :=
  let λ := 1
  let real_axis_length := 2 * λ
  real_axis_length

theorem hyperbola_real_axis_length_correct :
  ∀ (C : Hyperbola),
    C.center = (0, 0) →
    C.foci_on_x_axis →
    C.intersects_parabola_directrix (y^2 = 8 * x) (A, B) →
    dist A B = 2 * sqrt 3 →
    hyperbola_real_axis_length = 2 :=
by
  intros
  sorry

end hyperbola_real_axis_length_correct_l279_279551


namespace symmetry_axis_l279_279938

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + π / 6)

theorem symmetry_axis (ω : ℝ) (ω_pos : 0 < ω) (h : (∃ T > 0, ∀ x, f ω (x + T) = f ω x) ∧ (∀ T > 0, f ω (x + T) = f ω x → T ≥ π)) :
  x = π / 6 := 
sorry

end symmetry_axis_l279_279938


namespace distance_AB_correct_l279_279033

noncomputable theory

open Real

def parabola_eq (x y : ℝ) := y^2 = 4 * x

def line_eq (m y : ℝ) := (λ x, m * y + 1 : ℝ)

def area_ratio (yA yB : ℝ) := (-3) * yB = yA

def distance_AB (yA yB : ℝ) := sqrt (1 + 1/3) * sqrt((yA + yB)^2 - 4 * yA * yB)

theorem distance_AB_correct (yA yB : ℝ) (m : ℝ)
  (hyA : parabola_eq ((line_eq m yA) yA) yA)
  (hyB : parabola_eq ((line_eq m yB) yB) yB)
  (hratio : area_ratio yA yB)
  (hm : m^2 = 1/3):
  |sqrt (1 + 1/3) * sqrt((yA + yB)^2 - 4 * yA * yB)| = 16/3 :=
by sorry

end distance_AB_correct_l279_279033


namespace problem_statement_l279_279060

theorem problem_statement
  (a b c : ℝ) 
  (X : ℝ) 
  (hX : X = a + b + c + 2 * Real.sqrt (a^2 + b^2 + c^2 - a * b - b * c - c * a)) :
  X ≥ max (max (3 * a) (3 * b)) (3 * c) ∧ 
  ∃ (u v w : ℝ), 
    (u = Real.sqrt (X - 3 * a) ∧ v = Real.sqrt (X - 3 * b) ∧ w = Real.sqrt (X - 3 * c) ∧ 
     ((u + v = w) ∨ (v + w = u) ∨ (w + u = v))) :=
by
  sorry

end problem_statement_l279_279060


namespace quadratic_function_distinct_zeros_l279_279819

theorem quadratic_function_distinct_zeros (a : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) ↔ (a ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 0) := 
by
  sorry

end quadratic_function_distinct_zeros_l279_279819


namespace range_of_omega_l279_279771

theorem range_of_omega :
  ∀ (ω : ℝ), 
  (0 < ω) → 
  (∀ x, x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → cos (ω * x) - 1 = 0 → x ∈ {0, 2 * Real.pi, 4 * Real.pi}) →
  (2 ≤ ω ∧ ω < 3) :=
by
  intros ω hω_pos hzeros
  sorry

end range_of_omega_l279_279771


namespace Atlantic_Call_additional_charge_is_0_20_l279_279970

def United_Telephone_base_rate : ℝ := 7.00
def United_Telephone_rate_per_minute : ℝ := 0.25
def Atlantic_Call_base_rate : ℝ := 12.00
def United_Telephone_total_charge_100_minutes : ℝ := United_Telephone_base_rate + 100 * United_Telephone_rate_per_minute
def Atlantic_Call_total_charge_100_minutes (x : ℝ) : ℝ := Atlantic_Call_base_rate + 100 * x

theorem Atlantic_Call_additional_charge_is_0_20 :
  ∃ x : ℝ, United_Telephone_total_charge_100_minutes = Atlantic_Call_total_charge_100_minutes x ∧ x = 0.20 :=
by {
  -- Since United_Telephone_total_charge_100_minutes = 32.00, we need to prove:
  -- Atlantic_Call_total_charge_100_minutes 0.20 = 32.00
  sorry
}

end Atlantic_Call_additional_charge_is_0_20_l279_279970


namespace reading_time_per_week_l279_279589

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end reading_time_per_week_l279_279589


namespace angle_A_in_triangle_l279_279437

noncomputable def is_angle_A (a b : ℝ) (B A: ℝ) : Prop :=
  a = 2 * Real.sqrt 3 ∧ b = 2 * Real.sqrt 2 ∧ B = Real.pi / 4 ∧
  (A = Real.pi / 3 ∨ A = 2 * Real.pi / 3)

theorem angle_A_in_triangle (a b A B : ℝ) (h : is_angle_A a b B A) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end angle_A_in_triangle_l279_279437


namespace inequality_m_2n_l279_279796

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

lemma max_f : ∃ x : ℝ, f x = 2 :=
sorry

theorem inequality_m_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 1/(2*n) = 2) : m + 2*n ≥ 2 :=
sorry

end inequality_m_2n_l279_279796


namespace p_eval_l279_279140

def p : ℤ → ℤ → ℤ 
| x, y := if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
          else if x < 0 ∧ y < 0 then 2 * x - y
          else if x ≥ 0 ∧ y < 0 then 4 * x + 2 * y
          else x ^ 2 - y ^ 2

theorem p_eval : p (p 2 (-3)) (p (-4) 1) = 32 :=
by sorry

end p_eval_l279_279140


namespace range_of_omega_for_zeros_in_interval_l279_279779

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - 1

theorem range_of_omega_for_zeros_in_interval (ω : ℝ) (hω_positve : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f ω x = 0 → 2 ≤ ω ∧ ω < 3) :=
sorry

end range_of_omega_for_zeros_in_interval_l279_279779


namespace line_equation_passing_intersection_and_distance_3_l279_279761

theorem line_equation_passing_intersection_and_distance_3 :
  (∃ l1 l2 l : ℝ × ℝ, 
    (l1.1 = 0 ∧ l1.2 = 10) ∧ 
    (l2.1 = 3 ∧ l2.2 = 9) ∧ 
    l ∈ {(3, 0), (4, -3, 15)} ∧ 
    ∀ p : ℝ × ℝ, p ∈ {(0, 0)} → Real.sqrt (l.1 * l.1 + l.2 * l.2) = 3) :=
sorry

end line_equation_passing_intersection_and_distance_3_l279_279761


namespace cylindrical_to_cartesian_l279_279768

theorem cylindrical_to_cartesian :
  ∀ (r θ z : ℝ), r = 2 ∧ θ = π / 6 ∧ z = 7 →
  (r * cos θ, r * sin θ, z) = (√3, 1, 7) :=
by
  rintro _ _ _ ⟨rfl, rfl, rfl⟩
  sorry

end cylindrical_to_cartesian_l279_279768


namespace smallest_base10_integer_l279_279988

def is_valid_digit_base_6 (C : ℕ) : Prop := C ≤ 5
def is_valid_digit_base_8 (D : ℕ) : Prop := D ≤ 7

def CC_6_to_base10 (C : ℕ) : ℕ := 7 * C
def DD_8_to_base10 (D : ℕ) : ℕ := 9 * D

theorem smallest_base10_integer : ∃ C D : ℕ, 
  is_valid_digit_base_6 C ∧ 
  is_valid_digit_base_8 D ∧ 
  CC_6_to_base10 C = DD_8_to_base10 D ∧
  CC_6_to_base10 C = 63 := 
begin
  sorry
end

end smallest_base10_integer_l279_279988


namespace xy_exponential_340_l279_279519

theorem xy_exponential_340 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : sqrt(log x) + sqrt(log y) + log (sqrt x) + log (sqrt y) + log (sqrt (sqrt x)) + log (sqrt (sqrt y)) = 150)
  (h_int1 : ∃ p : ℤ, sqrt(log x) = p ∧ p > 0)
  (h_int2 : ∃ q : ℤ, sqrt(log y) = q ∧ q > 0)
  (h_int3 : ∃ r : ℤ, log (sqrt x) = r ∧ r > 0)
  (h_int4 : ∃ s : ℤ, log (sqrt y) = s ∧ s > 0)
  (h_int5 : ∃ t : ℤ, log (sqrt (sqrt x)) = t ∧ t > 0)
  (h_int6 : ∃ u : ℤ, log (sqrt (sqrt y)) = u ∧ u > 0) :
  x * y = Real.exp 340 :=
sorry

end xy_exponential_340_l279_279519


namespace part_1_part_2_part_3_l279_279732

noncomputable def complex_m (m : ℝ) : Prop :=
  let z := m - complex.i in
  let conj_z := m + complex.i in
  let product := conj_z * (1 + 3 * complex.i) in
  ((product.re = 0) → (m = 3))

noncomputable def magnitude_z1 (m : ℝ) : ℝ :=
  let z1 := ((m + 4 * complex.i) / (1 - complex.i)) in
  complex.abs z1

noncomputable def range_a (a : ℝ) : Prop :=
  let z := 3 - complex.i in
  let i2023 := -(I : ℂ) in
  let z2 := (a + complex.i) / z in
  ((0 < z2.re) ∧ (0 < z2.im) → (a > 1 / 3))

theorem part_1 (m : ℝ) : complex_m m :=
by sorry

theorem part_2 (m : ℝ) (h : m = 3) : magnitude_z1 m = 5 * real.sqrt 2 / 2 :=
by sorry

theorem part_3 (a : ℝ) : range_a a :=
by sorry

end part_1_part_2_part_3_l279_279732


namespace triangle_inequality_problem_statement_l279_279997

-- Definition of the triangle inequality theorem
theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b):=
begin
  split,
  assumption,
  split,
  assumption,
  assumption,
end

-- Main theorem statement
theorem problem_statement (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
begin
  -- Apply triangle inequality theorem and conditions
  apply triangle_inequality,
  sorry, -- Proof to be completed
end

end triangle_inequality_problem_statement_l279_279997


namespace find_g_3_16_l279_279136

theorem find_g_3_16 (g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x → x ≤ 1 → g x = g x) 
(h2 : g 0 = 0) 
(h3 : ∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) 
(h4 : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x) 
(h5 : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) : 
  g (3 / 16) = 8 / 27 :=
sorry

end find_g_3_16_l279_279136


namespace asymptotes_N_are_correct_l279_279046

-- Given the conditions of the hyperbola M
def hyperbola_M (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m - y^2 / 6 = 1

-- Eccentricity condition
def eccentricity (m : ℝ) (e : ℝ) : Prop :=
  e = 2 ∧ (m > 0)

-- Given hyperbola N
def hyperbola_N (x y : ℝ) (m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

-- The theorem to be proved
theorem asymptotes_N_are_correct (m : ℝ) (x y : ℝ) :
  hyperbola_M x y 2 → eccentricity 2 2 → hyperbola_N x y m →
  (y = x * Real.sqrt 2 ∨ y = -x * Real.sqrt 2) :=
by
  sorry

end asymptotes_N_are_correct_l279_279046


namespace smallest_m_exists_l279_279234

theorem smallest_m_exists :
  ∃ (m : ℕ), 0 < m ∧ (∃ k : ℕ, 5 * m = k^2) ∧ (∃ l : ℕ, 3 * m = l^3) ∧ m = 243 :=
by
  sorry

end smallest_m_exists_l279_279234


namespace simplify_and_evaluate_expression_l279_279530

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 3) : 
  ((x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4))) = (5 : ℝ) / 3 := 
by
  sorry

end simplify_and_evaluate_expression_l279_279530


namespace parabola_FV_sum_l279_279326

theorem parabola_FV_sum (BF BV : ℝ) (hBF : BF = 25) (hBV : BV = 26) :
  let FV := sqrt (BF^2 - (BV - FV)^2)
  in FV = 50 / 3 :=
by
  sorry

end parabola_FV_sum_l279_279326


namespace jimin_has_most_candy_left_l279_279851

-- Definitions based on conditions
def fraction_jimin_ate := 1 / 9
def fraction_taehyung_ate := 1 / 3
def fraction_hoseok_ate := 1 / 6

-- The goal to prove
theorem jimin_has_most_candy_left : 
  (1 - fraction_jimin_ate) > (1 - fraction_taehyung_ate) ∧ (1 - fraction_jimin_ate) > (1 - fraction_hoseok_ate) :=
by
  -- The actual proof steps are omitted here.
  sorry

end jimin_has_most_candy_left_l279_279851


namespace area_formed_by_points_in_plane_l279_279004

noncomputable def area_of_figure : ℝ := 
  let a := 2
  let b := 1
  a * b * π

theorem area_formed_by_points_in_plane :
  (∃ (P : ℝ × ℝ) (θ : ℝ), 
      (P.1 - cos θ)^2 / 4 + (P.2 - 0.5 * sin θ)^2 = 1) 
  → area_of_figure = 4 * π :=
by
  sorry

end area_formed_by_points_in_plane_l279_279004


namespace find_x0_l279_279876

variables {a b x x0 : ℝ}

theorem find_x0 (h₀ : a ≠ 0) 
  (h₁ : ∫ x in 0..3, a * x^2 + b = 3 * (a * x0^2 + b)) 
  (h₂ : x0 > 0) :
  x0 = real.sqrt 3 :=
sorry

end find_x0_l279_279876


namespace walking_distance_l279_279282

theorem walking_distance (a b : ℝ) (h1 : 10 * a + 45 * b = a * 15)
(h2 : x * (a + 9 * b) = 10 * a + 45 * b) : x = 13.5 :=
by
  sorry

end walking_distance_l279_279282


namespace andy_late_l279_279670

theorem andy_late
  (school_start : ℕ := 480) -- 8:00 AM in minutes since midnight
  (normal_travel_time : ℕ := 30)
  (red_lights : ℕ := 4)
  (red_light_wait_time : ℕ := 3)
  (construction_wait_time : ℕ := 10)
  (departure_time : ℕ := 435) -- 7:15 AM in minutes since midnight
  : ((school_start - departure_time) < (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time)) →
    school_start + (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time - (school_start - departure_time)) = school_start + 7 :=
by
  -- This skips the proof part
  sorry

end andy_late_l279_279670


namespace cos_half_pi_minus_2alpha_tan_2alpha_plus_quarter_pi_l279_279387

variable (α : Real)
variable (h1 : cos α = -4 / 5) (h2 : π / 2 < α ∧ α < π)

theorem cos_half_pi_minus_2alpha :
  cos (π / 2 - 2 * α) = -24 / 25 :=
by
  sorry

theorem tan_2alpha_plus_quarter_pi :
  tan (2 * α + π / 4) = -17 / 31 :=
by
  sorry

end cos_half_pi_minus_2alpha_tan_2alpha_plus_quarter_pi_l279_279387


namespace jasmine_gives_lola_marbles_l279_279117

theorem jasmine_gives_lola_marbles :
  ∃ (y : ℕ), ∀ (j l : ℕ), 
    j = 120 ∧ l = 15 ∧ 120 - y = 3 * (15 + y) → y = 19 := 
sorry

end jasmine_gives_lola_marbles_l279_279117


namespace Iain_pennies_left_l279_279068

theorem Iain_pennies_left (initial_pennies older_pennies : ℕ) (percentage : ℝ)
  (h_initial : initial_pennies = 200)
  (h_older : older_pennies = 30)
  (h_percentage : percentage = 0.20) :
  initial_pennies - older_pennies - (percentage * (initial_pennies - older_pennies)) = 136 :=
by
  sorry

end Iain_pennies_left_l279_279068


namespace probability_sum_of_two_dice_is_4_l279_279964

noncomputable def fair_dice_probability_sum_4 : ℚ :=
  let total_outcomes := 6 * 6 -- Total outcomes for two dice
  let favorable_outcomes := 3 -- Outcomes that sum to 4: (1, 3), (3, 1), (2, 2)
  favorable_outcomes / total_outcomes

theorem probability_sum_of_two_dice_is_4 : fair_dice_probability_sum_4 = 1 / 12 := 
by
  sorry

end probability_sum_of_two_dice_is_4_l279_279964


namespace percentage_of_students_chose_spring_is_10_l279_279170

-- Define the constants given in the problem
def total_students : ℕ := 10
def students_spring : ℕ := 1

-- Define the percentage calculation formula
def percentage (part total : ℕ) : ℕ := (part * 100) / total

-- State the theorem
theorem percentage_of_students_chose_spring_is_10 :
  percentage students_spring total_students = 10 :=
by
  -- We don't need to provide a proof here, just state it.
  sorry

end percentage_of_students_chose_spring_is_10_l279_279170


namespace expression_divisible_by_24_l279_279886

theorem expression_divisible_by_24 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end expression_divisible_by_24_l279_279886


namespace distance_from_F_to_midpoint_DE_l279_279097

variable (D E F : Type) [metric_space D] [metric_space E] [metric_space F]

def midpoint (a b : E) : E := (1/2 : ℝ) • (a + b)

theorem distance_from_F_to_midpoint_DE
  (D E F : Type) [metric_space F]
  (DE DF EF : ℝ)
  (h_DE : DE = 15)
  (h_DF : DF = 20)
  (h_EF : EF = real.sqrt (DF ^ 2 - DE ^ 2)) :
  dist F (midpoint D E) = 10 :=
by 
  -- The proof would be completed here
  sorry

end distance_from_F_to_midpoint_DE_l279_279097


namespace least_prime_factor_five_power_difference_l279_279975

theorem least_prime_factor_five_power_difference : 
  ∃ p : ℕ, (Nat.Prime p ∧ p ∣ (5^4 - 5^3)) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (5^4 - 5^3) → p ≤ q) := 
sorry

end least_prime_factor_five_power_difference_l279_279975


namespace circumradius_of_right_triangle_l279_279635

theorem circumradius_of_right_triangle
  (a b c : ℕ)
  (h : a^2 + b^2 = c^2)
  (ha : a = 8)
  (hb : b = 15)
  (hc : c = 17) :
  (c / 2 : ℝ) = 17 / 2 :=
by
  have ha_square : a^2 = 64 := by rw [ha]; norm_num
  have hb_square : b^2 = 225 := by rw [hb]; norm_num
  have hc_square : c^2 = 289 := by rw [hc]; norm_num
  have hypotenuse_calc : 64 + 225 = 289 := by norm_num
  replace h : 289 = 289 := by rw [ha_square, hb_square, hc_square, h]
  have same_hyp : c = 17 := hc
  rw same_hyp
  norm_num
  sorry

end circumradius_of_right_triangle_l279_279635


namespace sequence_finite_l279_279380

def sequence_terminates (a_0 : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (a 0 = a_0) ∧ 
                  (∀ n, ((a n > 5) ∧ (a n % 10 ≤ 5) → a (n + 1) = a n / 10)) ∧
                  (∀ n, ((a n > 5) ∧ (a n % 10 > 5) → a (n + 1) = 9 * a n)) → 
                  ∃ n, a n ≤ 5 

theorem sequence_finite (a_0 : ℕ) : sequence_terminates a_0 :=
sorry

end sequence_finite_l279_279380


namespace max_profit_l279_279638

noncomputable def maximum_profit : ℤ := 
  21000

theorem max_profit (x y : ℕ) 
  (h1 : 4 * x + 8 * y ≤ 8000)
  (h2 : 2 * x + y ≤ 1300)
  (h3 : 15 * x + 20 * y ≤ maximum_profit) : 
  15 * x + 20 * y = maximum_profit := 
sorry

end max_profit_l279_279638


namespace probability_product_zero_l279_279602

open Finset

theorem probability_product_zero :
  let s := \{ -3, -1, 0, 2, 4, 6, 7\} : Finset ℤ,
  n := s.card,
  total := nat.choose n 2,
  favorable := 6
  in  favorable / total = (2:ℚ) / 7 :=
by
  let s := \{ -3, -1, 0, 2, 4, 6, 7\} : Finset ℤ
  let n := s.card
  let total := nat.choose n 2
  let favorable := 6
  have h : favorable / total = (2: ℚ) / 7 := sorry
  exact h

end probability_product_zero_l279_279602


namespace ellipse_standard_form_line_standard_form_parallel_line_through_focus_l279_279104

-- Definitions for the parametric equations of ellipse (C) and line (L)
def ellipse (φ : ℝ) : ℝ × ℝ := (5 * Real.cos φ, 3 * Real.sin φ)
def line (t : ℝ) : ℝ × ℝ := (4 - 2 * t, 3 - t)

-- Theorem statements to be proved
theorem ellipse_standard_form :
  ∀ x y φ: ℝ, (x, y) = ellipse φ → (x^2) / 25 + (y^2) / 9 = 1 :=
by
  sorry

theorem line_standard_form :
  ∀ x y t: ℝ, (x, y) = line t → (x - 2 * y + 2 = 0) :=
by
  sorry

theorem parallel_line_through_focus :
  ∃ l : ℝ → ℝ × ℝ,
  (∀ t, l t = line t) ∧
  ∃ (x y: ℝ), (x, y) = (2, 0) → (x - 2 * y - 2 = 0) :=
by
  sorry

end ellipse_standard_form_line_standard_form_parallel_line_through_focus_l279_279104


namespace remainder_of_n_div_4_is_1_l279_279256

noncomputable def n : ℕ := sorry  -- We declare n as a noncomputable natural number to proceed with the proof complexity

theorem remainder_of_n_div_4_is_1 (n : ℕ) (h : (2 * n) % 4 = 2) : n % 4 = 1 :=
by
  sorry  -- skip the proof

end remainder_of_n_div_4_is_1_l279_279256


namespace exists_long_segment_between_parabolas_l279_279328

def parabola1 (x : ℝ) : ℝ :=
  x ^ 2

def parabola2 (x : ℝ) : ℝ :=
  x ^ 2 - 1

def in_between_parabolas (x y : ℝ) : Prop :=
  (parabola2 x) ≤ y ∧ y ≤ (parabola1 x)

theorem exists_long_segment_between_parabolas :
  ∃ (M1 M2: ℝ × ℝ), in_between_parabolas M1.1 M1.2 ∧ in_between_parabolas M2.1 M2.2 ∧ dist M1 M2 > 10^6 :=
sorry

end exists_long_segment_between_parabolas_l279_279328


namespace polygon_with_given_angle_sum_l279_279029

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end polygon_with_given_angle_sum_l279_279029


namespace range_of_omega_l279_279786

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ zeros : ℝ, (f(x) = cos (ω * x) - 1) and (count_zeros (f(x),  [0, 2 * π]) = 3)) ↔ (2 ≤ ω ∧ ω < 3) := 
sorry

end range_of_omega_l279_279786


namespace angle_SR_XY_is_70_l279_279438

-- Define the problem conditions
variables (X Y Z V H S R : Type) 
variables (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ)

-- Set the conditions
def triangleXYZ (X Y Z V H S R : Type) (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ) : Prop :=
  angleX = 40 ∧ angleY = 70 ∧ XY = 12 ∧ XV = 2 ∧ YH = 2 ∧
  ∃ S R, S = (XY / 2) ∧ R = ((XV + YH) / 2)

-- Construct the theorem to be proven
theorem angle_SR_XY_is_70 {X Y Z V H S R : Type} 
  {angleX angleY angleZ angleSRXY : ℝ} 
  {XY XV YH : ℝ} : 
  triangleXYZ X Y Z V H S R angleX angleY angleZ angleSRXY XY XV YH →
  angleSRXY = 70 :=
by
  -- Placeholder proof steps
  sorry

end angle_SR_XY_is_70_l279_279438


namespace num_of_selection_methods_l279_279893

theorem num_of_selection_methods : 
  let volunteers := 4
  let bok_choys := 3
  let n := volunteers + bok_choys in
  let k := 4 in
  nat.choose n k - nat.choose volunteers k = 34 :=
by
  sorry

end num_of_selection_methods_l279_279893


namespace exists_convex_and_nonconvex_polyhedra_with_congruent_faces_l279_279650

-- Definitions and conditions
def is_convex_polyhedron (P : Polyhedron) : Prop :=
  ∀ (p₁ p₂ : P.points), segment(p₁, p₂) ⊆ P ∧ (∀ (angle : P.inter_vals), angle < 180)

def face_is_convex_polygon (F : Face) : Prop :=
  is_convex_polygon F

def polyhedra_with_30_faces (P Q : Polyhedron) : Prop :=
  P.faces.card = 30 ∧ Q.faces.card = 30

def faces_are_congruent (P Q : Polyhedron) : Prop :=
  ∀ (fP ∈ P.faces) (fQ ∈ Q.faces), congruent_faces fP fQ

-- Theorem statement (proof that such polyhedra exist)
theorem exists_convex_and_nonconvex_polyhedra_with_congruent_faces :
  ∃ (P Q : Polyhedron), is_convex_polyhedron P ∧ ¬ is_convex_polyhedron Q ∧ 
    polyhedra_with_30_faces P Q ∧ faces_are_congruent P Q :=
begin
  sorry
end

end exists_convex_and_nonconvex_polyhedra_with_congruent_faces_l279_279650


namespace interest_earned_l279_279877

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : 
  P = 2000 → r = 0.05 → n = 5 → 
  A = P * (1 + r)^n → 
  A - P = 552.56 :=
by
  intro hP hr hn hA
  rw [hP, hr, hn] at hA
  sorry

end interest_earned_l279_279877


namespace savings_fraction_l279_279661

theorem savings_fraction 
(P : ℝ) 
(f : ℝ) 
(h1 : P > 0) 
(h2 : 12 * f * P = 5 * (1 - f) * P) : 
    f = 5 / 17 :=
by
  sorry

end savings_fraction_l279_279661


namespace determine_a_value_l279_279435

-- Define the initial equation and conditions
def fractional_equation (x a : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- Define the existence of a positive root
def has_positive_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ fractional_equation x a

-- The main theorem stating the correct value of 'a' for the given condition
theorem determine_a_value (x : ℝ) : has_positive_root 1 :=
sorry

end determine_a_value_l279_279435


namespace weight_of_triangular_piece_l279_279659

noncomputable def density_factor (weight : ℝ) (area : ℝ) : ℝ :=
  weight / area

noncomputable def square_weight (side_length : ℝ) (weight : ℝ) : ℝ := weight

noncomputable def triangle_area (side_length : ℝ) : ℝ :=
  (side_length ^ 2 * Real.sqrt 3) / 4

theorem weight_of_triangular_piece :
  let side_square := 4
  let weight_square := 16
  let side_triangle := 6
  let area_square := side_square ^ 2
  let area_triangle := triangle_area side_triangle
  let density_square := density_factor weight_square area_square
  let weight_triangle := area_triangle * density_square
  abs weight_triangle - 15.59 < 0.01 :=
by
  sorry

end weight_of_triangular_piece_l279_279659


namespace min_value_proof_l279_279403

noncomputable def minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
1 / a + 2 / b

theorem min_value_proof :
  (∀ x y : ℝ, ax + 2 * by - 2 = 0) →
  (a b > 0) →
  (∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 8 = 0) →
  (∀ x y : ℝ, ax + 2 * by - 2 = 0) →
  (a + b = 1) →
∃ a b, minimum_value a b ha hb = 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_proof_l279_279403


namespace blue_fractions_denominators_large_l279_279360

theorem blue_fractions_denominators_large
  (a1 a2 a3 a4 a5 : ℚ)
  (h1 : a1.denominator > 10^10 ∧ a1.denominator % 2 = 1)
  (h2 : a2.denominator > 10^10 ∧ a2.denominator % 2 = 1)
  (h3 : a3.denominator > 10^10 ∧ a3.denominator % 2 = 1)
  (h4 : a4.denominator > 10^10 ∧ a4.denominator % 2 = 1)
  (h5 : a5.denominator > 10^10 ∧ a5.denominator % 2 = 1)
  (hblue : ∀ (b : ℚ), ((b = a1 + a2 ∨ b = a2 + a3 ∨ b = a3 + a4 ∨ b = a4 + a5 ∨ b = a5 + a1) → b.denominator < 100)) : false :=
sorry

end blue_fractions_denominators_large_l279_279360


namespace first_day_of_month_l279_279173

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l279_279173


namespace probability_less_than_8000_miles_l279_279199

open ProbabilityMeasure

def distances : List (ℕ × ℕ) := [
  (5900, 1),
  (4800, 1),
  (6200, 1),
  (8700, 0),
  (2133, 1),
  (10400, 0)
]

def favorablePairs (dists : List (ℕ × ℕ)) : ℕ :=
  dists.filter (λ pair, pair.2 = 1).length

def totalPairs (dists : List (ℕ × ℕ)) : ℕ :=
  dists.length

theorem probability_less_than_8000_miles :
  (favorablePairs distances : ℚ) / (totalPairs distances : ℚ) = 2 / 3 := by
sorry

end probability_less_than_8000_miles_l279_279199


namespace least_k_l279_279894

noncomputable def u : ℕ → ℝ
| 0 => 1 / 8
| (n + 1) => 3 * u n - 3 * (u n) ^ 2

theorem least_k :
  ∃ k : ℕ, |u k - (1 / 3)| ≤ 1 / 2 ^ 500 ∧ ∀ m < k, |u m - (1 / 3)| > 1 / 2 ^ 500 :=
by
  sorry

end least_k_l279_279894


namespace total_gym_cost_l279_279119

def cheap_monthly_fee : ℕ := 10
def cheap_signup_fee : ℕ := 50
def expensive_monthly_fee : ℕ := 3 * cheap_monthly_fee
def expensive_signup_fee : ℕ := 4 * expensive_monthly_fee

def yearly_cost_cheap : ℕ := 12 * cheap_monthly_fee + cheap_signup_fee
def yearly_cost_expensive : ℕ := 12 * expensive_monthly_fee + expensive_signup_fee

theorem total_gym_cost : yearly_cost_cheap + yearly_cost_expensive = 650 := by
  -- Proof goes here
  sorry

end total_gym_cost_l279_279119


namespace domain_tan_x_plus_pi_over_3_l279_279928

open Real Set

theorem domain_tan_x_plus_pi_over_3 :
  ∀ x : ℝ, ¬ (∃ k : ℤ, x = k * π + π / 6) ↔ x ∈ {x : ℝ | ¬ ∃ k : ℤ, x = k * π + π / 6} :=
by {
  sorry
}

end domain_tan_x_plus_pi_over_3_l279_279928


namespace negation_of_cube_of_every_odd_number_is_odd_l279_279567

theorem negation_of_cube_of_every_odd_number_is_odd:
  ¬ (∀ n : ℤ, (n % 2 = 1 → (n^3 % 2 = 1))) ↔ ∃ n : ℤ, (n % 2 = 1 ∧ ¬ (n^3 % 2 = 1)) := 
by
  sorry

end negation_of_cube_of_every_odd_number_is_odd_l279_279567


namespace how_many_fewer_runs_did_E_score_l279_279087

-- Define the conditions
variables (a b c d e : ℕ)
variable (h1 : 5 * 36 = 180)
variable (h2 : d = e + 5)
variable (h3 : e = 20)
variable (h4 : b = d + e)
variable (h5 : b + c = 107)
variable (h6 : a + b + c + d + e = 180)

-- Specification to be proved
theorem how_many_fewer_runs_did_E_score :
  a - e = 8 :=
by {
  sorry
}

end how_many_fewer_runs_did_E_score_l279_279087


namespace verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l279_279156

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Definition for conversions from base 60 to base 10
def from_base_60 (d1 d0 : ℕ) : ℕ :=
  d1 * 60 + d0

-- Proof statements
theorem verify_21_base_60 : from_base_60 2 1 = 121 ∧ is_perfect_square 121 :=
by {
  sorry
}

theorem verify_1_base_60 : from_base_60 0 1 = 1 ∧ is_perfect_square 1 :=
by {
  sorry
}

theorem verify_2_base_60_not_square : from_base_60 0 2 = 2 ∧ ¬ is_perfect_square 2 :=
by {
  sorry
}

end verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l279_279156


namespace shirt_price_is_150_l279_279345

def price_of_shirt (X C : ℝ) : Prop :=
  (X + C = 600) ∧ (X = C / 3)

theorem shirt_price_is_150 :
  ∃ X C : ℝ, price_of_shirt X C ∧ X = 150 :=
by {
  use [150, 450],
  dsimp [price_of_shirt],
  split,
  { norm_num, },
  { norm_num, },
}

end shirt_price_is_150_l279_279345


namespace linda_age_l279_279504

theorem linda_age 
  (J : ℕ)  -- Jane's current age
  (H1 : ∃ J, 2 * J + 3 = 13) -- Linda is 3 more than 2 times the age of Jane
  (H2 : (J + 5) + ((2 * J + 3) + 5) = 28) -- In 5 years, the sum of their ages will be 28
  : 2 * J + 3 = 13 :=
by {
  sorry
}

end linda_age_l279_279504


namespace propositions_are_true_l279_279299

variables {α : Type*} [fintype α]
variables (X : α → ℝ) (Y : α → ℝ)

-- Define the variance function
def variance (X : α → ℝ) : ℝ :=
  let μ := (finset.univ.sum X) / fintype.card α in
  (finset.univ.sum (λ a, (X a - μ)^2)) / fintype.card α

-- Conditions
axiom var_scaling : variance X = 1 → variance (λ a, 2 * X a) = 4
axiom correlation_coefficient (X Y : α → ℝ) : (∃ r, r ≥ 0 ∧ r ≤ 1 ∧
  (∀ a, X a = r * Y a + (1 - r) * (finset.univ.sum X) / fintype.card α))

theorem propositions_are_true :
  (variance X = 1 → variance (λ a, 2 * X a) = 4) ∧
  (∀ (k : ℝ), k ≥ 0 → ¬ (small k → confident_related)) ∧
  (∃ r, r ≥ 0 ∧ r ≤ 1 ∧ (∀ a, Y a = r * X a ∨ r = (finset.univ.sum ℝ) / fintype.card α)) :=
by
  sorry

end propositions_are_true_l279_279299


namespace total_mappings_from_S_to_T_l279_279414

-- Define the sets S and T
def S := {a1, a2 : Type}
def T := {b1, b2 : Type}

-- State the theorem with the necessary conditions and expected result
theorem total_mappings_from_S_to_T : ∀ (S T : Type), (S → T).card = 4 :=
by
  sorry

end total_mappings_from_S_to_T_l279_279414


namespace no_x_satisfies_inequality_l279_279795

def f (x : ℝ) : ℝ := x^2 + x

theorem no_x_satisfies_inequality : ¬ ∃ x : ℝ, f (x - 2) + f x < 0 :=
by 
  unfold f 
  sorry

end no_x_satisfies_inequality_l279_279795


namespace percent_enclosed_by_hexagons_l279_279942

variable (b : ℝ) -- side length of smaller squares

def area_of_small_square : ℝ := b^2
def area_of_large_square : ℝ := 16 * area_of_small_square b
def area_of_hexagon : ℝ := 3 * area_of_small_square b
def total_area_of_hexagons : ℝ := 2 * area_of_hexagon b

theorem percent_enclosed_by_hexagons :
  (total_area_of_hexagons b / area_of_large_square b) * 100 = 37.5 :=
by
  -- Proof omitted
  sorry

end percent_enclosed_by_hexagons_l279_279942


namespace total_pieces_in_10_row_triangle_l279_279681

open Nat

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem total_pieces_in_10_row_triangle : 
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  unit_rods + connectors = 231 :=
by
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  show unit_rods + connectors = 231
  sorry

end total_pieces_in_10_row_triangle_l279_279681


namespace z_is_1_2_decades_younger_than_x_l279_279258

variable (x y z w : ℕ) -- Assume ages as natural numbers

def age_equivalence_1 : Prop := x + y = y + z + 12
def age_equivalence_2 : Prop := x + y + w = y + z + w + 12

theorem z_is_1_2_decades_younger_than_x (h1 : age_equivalence_1 x y z) (h2 : age_equivalence_2 x y z w) :
  z = x - 12 := by
  sorry

end z_is_1_2_decades_younger_than_x_l279_279258


namespace theorem_perimeter_shaded_region_theorem_area_shaded_region_l279_279107

noncomputable section

-- Definitions based on the conditions
def r : ℝ := Real.sqrt (1 / Real.pi)  -- radius of the unit circle

-- Define the perimeter and area functions for the shaded region
def perimeter_shaded_region (r : ℝ) : ℝ :=
  2 * Real.sqrt Real.pi

def area_shaded_region (r : ℝ) : ℝ :=
  1 / 5

-- Main theorem statements to prove
theorem theorem_perimeter_shaded_region
  (h : Real.pi * r^2 = 1) : perimeter_shaded_region r = 2 * Real.sqrt Real.pi :=
by
  sorry

theorem theorem_area_shaded_region
  (h : Real.pi * r^2 = 1) : area_shaded_region r = 1 / 5 :=
by
  sorry

end theorem_perimeter_shaded_region_theorem_area_shaded_region_l279_279107


namespace omega_range_l279_279777

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l279_279777


namespace entrepreneurs_not_attending_any_session_l279_279675

theorem entrepreneurs_not_attending_any_session 
  (total_entrepreneurs : ℕ) 
  (digital_marketing_attendees : ℕ) 
  (e_commerce_attendees : ℕ) 
  (both_sessions_attendees : ℕ)
  (h1 : total_entrepreneurs = 40)
  (h2 : digital_marketing_attendees = 22) 
  (h3 : e_commerce_attendees = 18) 
  (h4 : both_sessions_attendees = 8) : 
  total_entrepreneurs - (digital_marketing_attendees + e_commerce_attendees - both_sessions_attendees) = 8 :=
by sorry

end entrepreneurs_not_attending_any_session_l279_279675


namespace time_for_40_workers_to_complete_wall_l279_279271

noncomputable def work_rate (workers : ℕ) (days : ℕ) : ℝ := 1 / (workers * days)

theorem time_for_40_workers_to_complete_wall :
  let r := work_rate 100 5 in
  ∃t : ℝ, (40 * r) * t = 1 ∧ t = 12.5 :=
by
  let r := work_rate 100 5
  use 12.5
  split
  sorry
  norm_num

end time_for_40_workers_to_complete_wall_l279_279271


namespace molecular_weight_one_mole_of_AlOH3_l279_279610

variable (MW_7_moles : ℕ) (MW : ℕ)

theorem molecular_weight_one_mole_of_AlOH3 (h : MW_7_moles = 546) : MW = 78 :=
by
  sorry

end molecular_weight_one_mole_of_AlOH3_l279_279610


namespace find_x_l279_279059

-- Definitions of the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Inner product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Perpendicular condition
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l279_279059


namespace range_of_a_l279_279034

noncomputable def f (x : ℝ) : ℝ :=
  if (1 / 2) < x ∧ x ≤ 1 then (2 * x^2) / (x + 1)
  else if 0 ≤ x ∧ x ≤ (1 / 2) then - (1 / 3) * x + (1 / 6)
  else 0 -- undefined outside [0, 1]

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - 2 * a + 2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) : a > 0 ∧ x1 ∈ Icc (0:ℝ) 1 ∧ x2 ∈ Icc (0:ℝ) 1
  ∧ f x1 = g x2 a
  → (1 / 2) ≤ a ∧ a ≤ (4 / 3) :=
by
  sorry

end range_of_a_l279_279034


namespace expected_volunteers_2006_l279_279842

noncomputable def volunteers_2002 : ℕ := 500

noncomputable def volunteers_2003 : ℝ := volunteers_2002 * 1.4

noncomputable def volunteers_2004 : ℝ := volunteers_2003 * 1.4

noncomputable def volunteers_2005 : ℝ := volunteers_2004 * 1.4

noncomputable def volunteers_2006 : ℝ := volunteers_2005 * 0.8

theorem expected_volunteers_2006 : Int.round volunteers_2006 = 1098 :=
by
  have h_volunteers_2003 : volunteers_2003 = 500 * 1.4 := rfl
  have h_volunteers_2004 : volunteers_2004 = (500 * 1.4) * 1.4 := rfl
  have h_volunteers_2005 : volunteers_2005 = ((500 * 1.4) * 1.4) * 1.4 := rfl
  have h_volunteers_2006 : volunteers_2006 = (((500 * 1.4) * 1.4) * 1.4) * 0.8 := rfl
  sorry

end expected_volunteers_2006_l279_279842


namespace tan_theta_value_l279_279725

theorem tan_theta_value 
  (θ : ℝ)
  (h1 : 0 < θ)
  (h2 : θ < π)
  (h3 : 2 * sin θ + cos θ = sqrt 2 / 3) : 
  tan θ = - (90 + 5 * sqrt 86) / 168 := 
by 
  sorry

end tan_theta_value_l279_279725


namespace tim_reading_hours_per_week_l279_279595

theorem tim_reading_hours_per_week :
  (meditation_hours_per_day = 1) →
  (reading_hours_per_day = 2 * meditation_hours_per_day) →
  (reading_hours_per_week = reading_hours_per_day * 7) →
  reading_hours_per_week = 14 :=
by
  intros h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  exact h3

end tim_reading_hours_per_week_l279_279595


namespace max_value_of_xy_l279_279539

theorem max_value_of_xy (x y : ℝ) (h₁ : x + y = 40) (h₂ : x > 0) (h₃ : y > 0) : xy ≤ 400 :=
sorry

end max_value_of_xy_l279_279539


namespace R_fifteen_l279_279349

def R : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+3) := R (n+2) + R (n+1)
| _     := sorry

theorem R_fifteen : R 15 = 987 :=
sorry

end R_fifteen_l279_279349


namespace sin_eq_sqrt3_div_2_l279_279727

open Real

theorem sin_eq_sqrt3_div_2 (theta : ℝ) :
  sin theta = (sqrt 3) / 2 ↔ (∃ k : ℤ, theta = π/3 + 2*k*π ∨ theta = 2*π/3 + 2*k*π) :=
by
  sorry

end sin_eq_sqrt3_div_2_l279_279727


namespace kasha_pasha_truth_difference_l279_279856

-- Define the conditions for Katya and Pasha
def katya_binomial_distribution := Binomial(4, 2/3)
def pasha_binomial_distribution := Binomial(4, 3/5)

theorem kasha_pasha_truth_difference :
  (∑ x in Finset.range 5,
    (katya_binomial_distribution.prob x) * (pasha_binomial_distribution.prob (x + 2))) = 48 / 625 :=
by
  sorry

end kasha_pasha_truth_difference_l279_279856


namespace first_day_of_month_l279_279187

noncomputable def day_of_week := ℕ → ℕ

def is_wednesday (n : ℕ) : Prop := day_of_week n = 3

theorem first_day_of_month (day_of_week : day_of_week) (h : is_wednesday 30) : day_of_week 1 = 2 :=
by
  sorry

end first_day_of_month_l279_279187


namespace percentage_of_ginger_is_correct_l279_279148

noncomputable def teaspoons_per_tablespoon : ℕ := 3
noncomputable def ginger_tablespoons : ℕ := 3
noncomputable def cardamom_teaspoons : ℕ := 1
noncomputable def mustard_teaspoons : ℕ := 1
noncomputable def garlic_tablespoons : ℕ := 2
noncomputable def chile_powder_factor : ℕ := 4

theorem percentage_of_ginger_is_correct :
  let ginger_teaspoons := ginger_tablespoons * teaspoons_per_tablespoon
  let garlic_teaspoons := garlic_tablespoons * teaspoons_per_tablespoon
  let chile_teaspoons := chile_powder_factor * mustard_teaspoons
  let total_teaspoons := ginger_teaspoons + cardamom_teaspoons + mustard_teaspoons + garlic_teaspoons + chile_teaspoons
  let percentage_ginger := (ginger_teaspoons * 100) / total_teaspoons
  percentage_ginger = 43 :=
by
  sorry

end percentage_of_ginger_is_correct_l279_279148


namespace medians_equal_segments_l279_279848

variable {A B C A₁ B₁ : Type}
variables [MetricSpace α] [Metric α]
variables (triangle : Triangle α)
variables (A B C A₁ B₁ : α)

-- Given: Medians
variable (median_A : Segment A C A₁)
variable (median_B : Segment B C B₁)

-- Condition: Equal angles
variable (equal_angles : ∠ A A₁ C = ∠ B B₁ C)

-- Goal: Prove AC = BC
theorem medians_equal_segments (hA : median_A.is_median) (hB : median_B.is_median) (h_angle : equal_angles) : dist A C = dist B C := 
sorry

end medians_equal_segments_l279_279848


namespace find_pos_integers_A_B_l279_279714

noncomputable def concat (A B : ℕ) : ℕ :=
  let b := Nat.log 10 B + 1
  A * 10 ^ b + B

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def satisfiesConditions (A B : ℕ) : Prop :=
  isPerfectSquare (concat A B) ∧ concat A B = 2 * A * B

theorem find_pos_integers_A_B :
  ∃ (A B : ℕ), A = (5 ^ b + 1) / 2 ∧ B = 2 ^ b * A * 100 ^ m ∧ b % 2 = 1 ∧ ∀ m : ℕ, satisfiesConditions A B :=
sorry

end find_pos_integers_A_B_l279_279714


namespace sin_minus_cos_l279_279379

theorem sin_minus_cos {A : ℝ} (hA : A ∈ set.Icc 0 π)
  (h1 : sin A + cos A = 3/5) : sin A - cos A = sqrt 41 / 5 := by
  sorry

end sin_minus_cos_l279_279379


namespace invariant_cond_l279_279940

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * Real.log x - b * Real.sin x + 3

theorem invariant_cond {a b : ℝ} : 
  (∃ g, ∀ y, f (g y) = y ∧ ∀ x, g (f x) = x) ↔ (b = 0 ∧ a ≠ 0) := 
sorry

end invariant_cond_l279_279940


namespace remitted_amount_is_correct_l279_279653

-- Define the constants and conditions of the problem
def total_sales : ℝ := 32500
def commission_rate1 : ℝ := 0.05
def commission_limit : ℝ := 10000
def commission_rate2 : ℝ := 0.04

-- Define the function to calculate the remitted amount
def remitted_amount (total_sales commission_rate1 commission_limit commission_rate2 : ℝ) : ℝ :=
  let commission1 := commission_rate1 * commission_limit
  let remaining_sales := total_sales - commission_limit
  let commission2 := commission_rate2 * remaining_sales
  total_sales - (commission1 + commission2)

-- Lean statement to prove the remitted amount
theorem remitted_amount_is_correct :
  remitted_amount total_sales commission_rate1 commission_limit commission_rate2 = 31100 :=
by
  sorry

end remitted_amount_is_correct_l279_279653


namespace ratio_sum_l279_279599

theorem ratio_sum {P Q R S T : Type*} 
    [euclidean_geometry P Q R S T] 
    (h1 : right_triangle P Q R)
    (h2 : ∠PQR = π / 2)
    (h3 : PR = 5)
    (h4 : QR = 12)
    (h5 : right_triangle P R S)
    (h6 : ∠PRS = π / 2)
    (h7 : RS = 30)
    (h8 : line_parallel ST PR)
    (h9 : opposite_side P S QR) :
    ∃ (m n : ℕ), (relatively_prime m n) ∧ (ST/SR = m/n) ∧ (m + n = 25) :=
by
  -- Proof would go here
  sorry

end ratio_sum_l279_279599


namespace min_sum_among_positive_numbers_l279_279378

variables {real : Type*}

open_locale classical

theorem min_sum_among_positive_numbers
  {n : ℕ} 
  (a b : fin n → ℝ) 
  (h₁: ∀ i, 0 < a i) 
  (h₂: ∀ i, 0 < b i)
  (h₃: ∑ i in finset.fin_range n, a i = ∑ i in finset.fin_range n, b i)
  (h₄: ∀ i j, i < j → a i * a j ≥ b i + b j) : 
  ∑ i in finset.fin_range n, a i ≥ 2 * n := 
sorry

end min_sum_among_positive_numbers_l279_279378


namespace monotonic_increasing_interval_l279_279205

variable (x : ℝ)

def func : ℝ → ℝ := λ x, 3*x^2 - 2*log x

noncomputable def derivative_of_func : ℝ → ℝ := λ x, 6*x - 2/x

theorem monotonic_increasing_interval :
  {x : ℝ | 0 < x} ∩ {x | 6*x - 2/x > 0} = {x : ℝ | x > real.sqrt(3)/3} :=
by sorry

end monotonic_increasing_interval_l279_279205


namespace find_polynomial_g_l279_279863

-- Define the condition that f(x) and g(x) are quadratic polynomials
variable (f g : ℝ → ℝ)

-- Define the conditions given in the problem
axiom quadratic_f : ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c
axiom quadratic_g : ∃ d e h : ℝ, ∀ x : ℝ, g x = d * x^2 + e * x + h
axiom functional_eq : ∀ x : ℝ, f (g x) = f x * g x
axiom g_at_3 : g 3 = 40

-- Define the statement we want to prove
theorem find_polynomial_g : ∃ e f : ℝ, g = (λ x : ℝ, x^2 + e * x + f) ∧ e = 31 / 2 ∧ f = -(31 / 2) := 
by
  sorry

end find_polynomial_g_l279_279863


namespace average_rate_of_change_l279_279040

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l279_279040


namespace calculate_expression_l279_279684

noncomputable def abs : ℝ → ℝ
| x := if x < 0 then -x else x

noncomputable def tan_deg (deg : ℝ) : ℝ :=
  Real.tan (deg * Real.pi / 180)

theorem calculate_expression :
  abs (-Real.sqrt 3) + (1 / 2)⁻¹ + (Real.pi + 1)^0 - tan_deg 60 = 3 :=
by sorry

end calculate_expression_l279_279684


namespace largest_divisor_of_expression_l279_279936

theorem largest_divisor_of_expression (n : ℤ) : ∃ k : ℤ, k = 6 ∧ (n^3 - n + 15) % k = 0 := 
by
  use 6
  sorry

end largest_divisor_of_expression_l279_279936


namespace derivative_at_one_l279_279370

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem derivative_at_one :
  deriv f 1 = -1 / 4 :=
by
  sorry

end derivative_at_one_l279_279370


namespace vector_combination_of_AD_l279_279307

variables {A B C D : Type} [add_comm_group D] [module ℝ D]
          (AB AC AD BC BD DC : D) (λ : ℝ)
          (h1 : BD = λ • DC) (hλ : λ > 0)

theorem vector_combination_of_AD :
  AD = (1 / (1 + λ)) • AB + (λ / (1 + λ)) • AC :=
sorry

end vector_combination_of_AD_l279_279307


namespace base_of_parallelogram_l279_279351

-- Define parameters for base, area, and height
variables (Base Area Height : ℕ)

-- Conditions
def height_condition : Height = 24 := sorry
def area_condition : Area = 336 := sorry

-- Theorem stating that given the conditions, the base is 14 cm
theorem base_of_parallelogram : 
  (336 : ℕ) = Area → (24 : ℕ) = Height → ((Area / Height) = 14 : ℕ) :=
by
  intros h1 h2
  rw [← h1, ← h2]
  sorry

end base_of_parallelogram_l279_279351


namespace three_letter_permutations_l279_279416

theorem three_letter_permutations (word : Finset Char) (h_card : word.card = 3) : 
  word.perm.toFinset.card = 6 ↔ (∀ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ word ∧ b ∈ word ∧ c ∈ word) :=
by
  sorry

end three_letter_permutations_l279_279416


namespace proportion_check_option_B_l279_279246

theorem proportion_check_option_B (a b c d : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 2) (hd : d = 4) :
  (a / b) = (c / d) :=
by {
  sorry
}

end proportion_check_option_B_l279_279246


namespace triangle_AB_length_l279_279818

theorem triangle_AB_length (area : ℝ) (BC : ℝ) (angleC : ℝ)
  (h_area : area = sqrt 3) (h_BC : BC = 2) (h_angleC : angleC = 60 * (π / 180)) : 
  ∃ AB : ℝ, AB = 2 :=
by 
  have sineC : ℝ := real.sin (angleC),
  have AB := 2,
  use AB,
  sorry

end triangle_AB_length_l279_279818


namespace hyperbola_distance_l279_279554

theorem hyperbola_distance :
  let a : ℝ := 2
  let b : ℝ := 2 * sqrt 2
  let vertex : ℝ × ℝ := (0, 2)
  let asymptote := λ x y : ℝ, x + sqrt 2 * y = 0
  sqrt (2 : ℝ) = sqrt 2 →
  (abs (1 * 0 + sqrt 2 * 2) / sqrt (1^2 + (sqrt 2)^2) = 2 * sqrt 6 / 3) := 
by
  sorry

end hyperbola_distance_l279_279554


namespace max_even_integers_with_odd_product_l279_279294

theorem max_even_integers_with_odd_product (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) (h_odd_product : (a * b * c * d * e * f) % 2 = 1) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) := 
sorry

end max_even_integers_with_odd_product_l279_279294


namespace cos_angle_between_slant_heights_and_k_values_l279_279649

noncomputable def cosine_between_slant_heights (k : ℝ) (h₀ : 0 < k) (h₁ : k < 0.25 * Real.sqrt 2) : ℝ :=
  16 * k^2 - 1

theorem cos_angle_between_slant_heights_and_k_values (k : ℝ) (h₀ : 0 < k) (h₁ : k < 0.25 * Real.sqrt 2) :
  let cos_theta := cosine_between_slant_heights k h₀ h₁ in
  cos_theta = 16 * k^2 - 1 ∧ 0 < k ∧ k < 0.25 * Real.sqrt 2 :=
sorry

end cos_angle_between_slant_heights_and_k_values_l279_279649


namespace intersecting_lines_a_plus_b_l279_279937

theorem intersecting_lines_a_plus_b :
  ∃ (a b : ℝ), (∀ x y : ℝ, (x = 1 / 3 * y + a) ∧ (y = 1 / 3 * x + b) → (x = 3 ∧ y = 4)) ∧ a + b = 14 / 3 :=
sorry

end intersecting_lines_a_plus_b_l279_279937


namespace count_ordered_quintuples_l279_279701

theorem count_ordered_quintuples:
  ( ∃ (Q R : polynomial ℤ), ∀ (a b c d e : ℕ),
    0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 30 →
    (x^a + x^b + x^c + x^d + x^e) = Q * (x^5 + x^4 + x^3 + x + 1) + 2 * R) →
  (finset.card {q : fin 31 × fin 31 × fin 31 × fin 31 × fin 31 | 
    let (a, b, c, d, e) := q in 0 ≤ a.val ∧ a.val < b.val ∧ b.val < c.val ∧ c.val < d.val ∧ d.val < e.val ∧ e.val ≤ 30 ∧
    (∃ (Q R : polynomial ℤ), 
      (x^a.val + x^b.val + x^c.val + x^d.val + x^e.val) = Q * (x^5 + x^4 + x^2 + x + 1) + 2 * R)) = 5208
sorry

end count_ordered_quintuples_l279_279701


namespace find_b_find_sin_2B_minus_pi_over_3_l279_279083

variable {A B C a b c : ℝ}
variable [b_nonzero : b ≠ 0]
variable [B_nonzero : B ≠ 0]

-- Given conditions
axiom cond1 : a = 3
axiom cond2 : b * sin A = 3 * c * sin B
axiom cond3 : cos B = 2/3

-- Goals to prove
theorem find_b :
  (c = 1) ∧ (b = sqrt 6) :=
sorry

theorem find_sin_2B_minus_pi_over_3 :
  sin (2 * B - π / 3) = (4 * sqrt 5 + sqrt 3) / 18 :=
sorry

end find_b_find_sin_2B_minus_pi_over_3_l279_279083


namespace every_term_is_square_l279_279935

noncomputable def sequence (m n : ℤ) : ℕ → ℤ
| 0     := m
| 1     := n
| 2     := 2 * n - m + 2
| (i+3) := 3 * (sequence (m n) (i+2) - sequence (m n) (i+1)) + sequence (m n) i

theorem every_term_is_square (m n : ℤ) :
  (∀ i : ℕ, is_square (sequence m n i)) :=
begin
  sorry
end

end every_term_is_square_l279_279935


namespace decreasing_interval_and_minimum_value_range_of_k_l279_279734

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x + k / x

theorem decreasing_interval_and_minimum_value (k : ℝ) (hk : k = Real.exp 1) :
  (∀ x > 0, x < Real.exp 1 → (f x k).deriv < 0) ∧ (f (Real.exp 1) k = 2) :=
by
  sorry

theorem range_of_k (k : ℝ) :
  (∀ x1 x2 > 0, x1 > x2 → f x1 k - f x2 k < x1 - x2) ↔ k ∈ Set.Ici (1 / 4) :=
by
  sorry

end decreasing_interval_and_minimum_value_range_of_k_l279_279734


namespace find_sin_C_find_area_of_triangle_l279_279822

-- Definitions and conditions
def A : ℝ := 45 * (Real.pi / 180) -- Angle A in radians
def cos_B : ℝ := 3 / 5 -- cosine of angle B
def sin_B : ℝ := Real.sqrt (1 - cos_B ^ 2) -- sine of angle B using Pythagorean identity
def a : ℝ := 5 -- Side a

-- Statements to be proved
theorem find_sin_C : sin (A + Real.acos cos_B) = 7 * Real.sqrt 2 / 10 := sorry

theorem find_area_of_triangle :
  let b := a * sin_B / (Real.sin A) in
  let sin_C := Real.sin (A + Real.acos cos_B) in
  (1 / 2) * a * b * sin_C = 14 := sorry

end find_sin_C_find_area_of_triangle_l279_279822


namespace transformed_sum_l279_279284

variable {ι : Type*} -- Type variable for the index
variable (x : ι → ℝ) -- The original set of numbers x_i
variable [Fintype ι] -- The finite type assumption for the indexing set
variable (s : ℝ) (n : ℕ) -- The sum of the numbers and size of the set

-- Conditions
def sum_eq_s (x : ι → ℝ) : Prop := (∑ i, x i) = s

def cardinality_eq_n : Prop := Fintype.card ι = n

-- The transformation applied to each element x_i
def transformed (x : ι → ℝ) (i : ι) : ℝ := 3 * (x i + 10) - 5

-- The theorem to prove
theorem transformed_sum (h₁ : sum_eq_s x) (h₂ : cardinality_eq_n) : (∑ i, transformed x i) = 3 * s + 25 * n :=
by
  sorry

end transformed_sum_l279_279284


namespace hiring_manager_acceptance_l279_279923

theorem hiring_manager_acceptance:
  ∀ (avg_age std_dev max_diff_ages : ℤ), 
    avg_age = 31 ∧ 
    std_dev = 5 ∧ 
    max_diff_ages = 11 → 
    ∃ k : ℤ, 10 * k + 1 = max_diff_ages ∧ k = 1 :=
by
  intro avg_age std_dev max_diff_ages
  intro h
  use 1
  simp at h
  cases h with h_avg_age h_rest
  cases h_rest with h_std_dev h_max_diff_ages
  rw [h_avg_age, h_std_dev, h_max_diff_ages]
  split; sorry

end hiring_manager_acceptance_l279_279923


namespace slope_after_rotation_l279_279805

theorem slope_after_rotation (slope_PQ : ℝ) (rotation_angle : ℝ) (new_slope : ℝ) : 
  slope_PQ = - real.sqrt 3 ∧ rotation_angle = real.pi / 3 → new_slope = real.sqrt 3 :=
begin
  sorry
end

end slope_after_rotation_l279_279805


namespace range_of_omega_l279_279787

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ zeros : ℝ, (f(x) = cos (ω * x) - 1) and (count_zeros (f(x),  [0, 2 * π]) = 3)) ↔ (2 ≤ ω ∧ ω < 3) := 
sorry

end range_of_omega_l279_279787


namespace find_p_l279_279510

theorem find_p (P : ℝ → ℝ) (p : ℝ) 
  (h1 : ∀ x y, P (x, y) ↔ (x^2 / 4) + y^2 = 1)
  (h2 : ∀ x, P (2, 0))
  (h3 : p > 0)
  : (∀ A B, chord_through_focus A B (2, 0) → (angle_equal A B P (2,0))) → p = 1.2 :=
begin
  sorry,
end

-- Definitions for theorems
def chord_through_focus (A B F : ℝ × ℝ) : Prop := sorry

def angle_equal (A B P F : ℝ × ℝ) : Prop := sorry

end find_p_l279_279510


namespace robert_saves_5_dollars_l279_279636

theorem robert_saves_5_dollars :
  let original_price := 50
  let promotion_c_discount (price : ℕ) := price * 20 / 100
  let promotion_d_discount (price : ℕ) := 15
  let cost_promotion_c := original_price + (original_price - promotion_c_discount original_price)
  let cost_promotion_d := original_price + (original_price - promotion_d_discount original_price)
  (cost_promotion_c - cost_promotion_d) = 5 :=
by
  sorry

end robert_saves_5_dollars_l279_279636


namespace fraction_more_than_three_children_l279_279627

-- Define the variables and conditions as given
variables (total_couples : ℕ)
variables (more_than_one_child: total_couples → Prop)
variables (two_or_three_children: total_couples → Prop)

-- Fractions provided in the problem
def fraction_more_than_one_child : ℚ := 3 / 5
def fraction_two_or_three_children : ℚ := 0.2

-- Theorem statement
theorem fraction_more_than_three_children : 
  fraction_more_than_one_child = 3 / 5 ∧ fraction_two_or_three_children = 0.2 → 
  (fraction_more_than_one_child - fraction_two_or_three_children) = 2 / 5 :=
by
  sorry

end fraction_more_than_three_children_l279_279627


namespace adam_school_percentage_l279_279292

theorem adam_school_percentage :
  ∀ (A B C total : ℕ), 
  (B = A - 21) →
  (C = 37) →
  (total = 80) →
  (A + B + C = total) →
  (A * 100 / total = 40 : ℕ) :=
by
  intros A B C total hB hC hTotal hSum
  sorry

end adam_school_percentage_l279_279292


namespace range_of_a_l279_279006

def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

noncomputable def roots (a : ℝ) : (ℝ × ℝ) :=
  (1, 3)

noncomputable def f_max (a : ℝ) :=
  -a

theorem range_of_a (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x < 0 ↔ (x < 1 ∨ 3 < x))
  (h2 : f_max a < 2) : 
  -2 < a ∧ a < 0 :=
sorry

end range_of_a_l279_279006


namespace BE_parallel_DF_l279_279500

open_locale euclidean_geometry

variables {A B C D E F : Point}

-- Angle condition: 2 * ∠CBA = 3 * ∠ACB
axiom angle_condition : 2 * ∠ C B A = 3 * ∠ A C B

-- Points D and E trisect ∠CBA
axiom D_E_trisect_CBA : ∠ A B D = ∠ D B E ∧ ∠ D B E = ∠ E B C

-- Point F is the intersection of AB and the bisector of ∠ACB
axiom F_bisector_ACB : ∠ A C F = ∠ F C B

-- We need to prove: BE ∥ DF
theorem BE_parallel_DF : BE ∥ DF :=
by sorry

end BE_parallel_DF_l279_279500


namespace abs_pow_eq_one_l279_279749

theorem abs_pow_eq_one {a b : ℤ} (h : (2 * a + b - 1) ^ 2 + |a - b + 4| = 0) : |a^b| = 1 :=
sorry

end abs_pow_eq_one_l279_279749


namespace cream_needed_l279_279366

variable (t h b : ℝ)

theorem cream_needed (ht : t = 300) (hh : h = 149) (hb : b = t - h) : b = 151 := by
  rw [ht, hh] at hb
  exact hb

end cream_needed_l279_279366


namespace baguettes_left_at_end_of_day_l279_279924

-- Conditions summarized as constants
def batches_per_day := 3
def baguettes_per_batch := 48
def first_batch_sold := 37
def second_batch_sold := 52
def third_batch_sold := 49

-- Proof problem: Prove that the number of baguettes left at the end of the day is 6
theorem baguettes_left_at_end_of_day :
  let total_baguettes := batches_per_day * baguettes_per_batch,
      baguettes_after_first_batch := baguettes_per_batch - first_batch_sold,
      baguettes_after_second_batch := (baguettes_after_first_batch + baguettes_per_batch) - second_batch_sold,
      baguettes_after_third_batch := (baguettes_after_second_batch + baguettes_per_batch) - third_batch_sold
  in baguettes_after_third_batch = 6 :=
by
  -- proof will go here
  sorry

end baguettes_left_at_end_of_day_l279_279924


namespace max_base_seven_digit_sum_l279_279609

theorem max_base_seven_digit_sum (n : ℕ) (h1 : 0 < n) (h2 : n < 2019) : 
  let digits := Nat.digits 7 n in 
  digits.sum ≤ 22 := 
sorry

end max_base_seven_digit_sum_l279_279609


namespace harry_100th_term_is_6_l279_279061

def apply_rule (n: ℕ) : ℕ :=
  if n < 10 then
    n * 7
  else if n % 2 = 0 then
    n / 3
  else
    n - 3

def harry_sequence (start: ℕ) : ℕ → ℕ
| 0 => start
| (n + 1) => apply_rule (harry_sequence n)

theorem harry_100th_term_is_6 : harry_sequence 120 99 = 6 := 
  sorry

end harry_100th_term_is_6_l279_279061


namespace range_of_omega_l279_279772

theorem range_of_omega :
  ∀ (ω : ℝ), 
  (0 < ω) → 
  (∀ x, x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → cos (ω * x) - 1 = 0 → x ∈ {0, 2 * Real.pi, 4 * Real.pi}) →
  (2 ≤ ω ∧ ω < 3) :=
by
  intros ω hω_pos hzeros
  sorry

end range_of_omega_l279_279772


namespace remainder_b94_mod_55_eq_29_l279_279867

theorem remainder_b94_mod_55_eq_29 :
  (5^94 + 7^94) % 55 = 29 := 
by
  -- conditions: local definitions for bn, modulo, etc.
  sorry

end remainder_b94_mod_55_eq_29_l279_279867


namespace problem1_problem2_problem3_problem4_l279_279588

-- Question 1
theorem problem1 (a b : ℝ) (h : 5 * a + 3 * b = -4) : 2 * (a + b) + 4 * (2 * a + b) = -8 :=
by
  sorry

-- Question 2
theorem problem2 (a : ℝ) (h : a^2 + a = 3) : 2 * a^2 + 2 * a + 2023 = 2029 :=
by
  sorry

-- Question 3
theorem problem3 (a b : ℝ) (h : a - 2 * b = -3) : 3 * (a - b) - 7 * a + 11 * b + 2 = 14 :=
by
  sorry

-- Question 4
theorem problem4 (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) 
  (h2 : a * b - 2 * b^2 = -3) : a^2 + a * b + 2 * b^2 = -2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l279_279588


namespace robins_count_l279_279157

theorem robins_count (total_birds : ℕ) (sparrows_fraction : ℚ) (total_birds_eq : total_birds = 120) (sparrows_fraction_eq : sparrows_fraction = 1/3) : 
  ∃ (robins : ℕ), robins = (2/3 : ℚ) * total_birds := 
by
  use (2/3 : ℚ) * 120
  -- sorry to skip the proof
  sorry

end robins_count_l279_279157


namespace volume_hemisphere_from_sphere_l279_279582

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end volume_hemisphere_from_sphere_l279_279582


namespace find_f_neg_one_l279_279489

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end find_f_neg_one_l279_279489


namespace initial_total_cash_in_usd_l279_279122

def exchange_rate : Real := 1.2
def sales_tax : Real := 0.05
def labor_cost_rate : Real := 0.10
def inflation_rate : Real := 0.02
def years : Nat := 2
def spent_on_raw_materials_euros : Real := 500
def spent_on_machinery_euros : Real := 400
def remaining_amount_after_2_years : Real := 900

theorem initial_total_cash_in_usd :
  let raw_materials_total_euros := spent_on_raw_materials_euros * (1 + sales_tax)
  let machinery_total_euros := spent_on_machinery_euros * (1 + sales_tax)
  let total_spent_usd := (raw_materials_total_euros + machinery_total_euros) * exchange_rate
  let X := (remaining_amount_after_2_years * (1 + inflation_rate)^years + total_spent_usd) / (1 - labor_cost_rate)
  X ≈ 2300.40 :=
by sorry

end initial_total_cash_in_usd_l279_279122


namespace smallest_number_starts_with_four_and_decreases_four_times_l279_279715

theorem smallest_number_starts_with_four_and_decreases_four_times :
  ∃ (X : ℕ), ∃ (A n : ℕ), (X = 4 * 10^n + A ∧ X = 4 * (10 * A + 4)) ∧ X = 410256 := 
by
  sorry

end smallest_number_starts_with_four_and_decreases_four_times_l279_279715


namespace smallest_possible_elements_in_set_l279_279010

theorem smallest_possible_elements_in_set (n : ℕ) (h_n : 2 ≤ n)
  (a : ℕ → ℕ) (h_a0 : a 0 = 0) (h_an : a n = 2 * n - 1)
  (h_ascending : ∀ i j, i < j → a i < a j) :
  (∃ (S : Set ℕ), (∀ i j, 0 ≤ i → i ≤ n → 0 ≤ j → j ≤ n → a i + a j ∈ S) ∧ (∀ T : Set ℕ, (∀ i j, 0 ≤ i → i ≤ n → 0 ≤ j → j ≤ n → a i + a j ∈ T) → (S ⊆ T)) ∧ (∀ U : Set ℕ, (∀ i j, U = (set.range (λ p : Fin (n + 1) × Fin (n + 1), a p.fst + a p.snd)) → S.card ≤ U.card) ∧ S.card = 3 * n) :=
  sorry

end smallest_possible_elements_in_set_l279_279010


namespace problem_statement_l279_279483

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (f_odd: ∀ x: ℝ, f(-x) = -f(x))
                         (f_condition: ∀ x: ℝ, f(3/2 + x) = -f(3/2 - x))
                         (f_1_eq_2: f 1 = 2):
  f 2 + f 3 = -2 := 
by sorry

end problem_statement_l279_279483


namespace slope_of_parallel_line_l279_279611

theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) : 
  (5 * x - 3 * y = 12) → m = 5 / 3 → (∃ b : ℝ, y = (5 / 3) * x + b) :=
by
  intro h_eqn h_slope
  use -4 / 3
  sorry

end slope_of_parallel_line_l279_279611


namespace lowest_average_processing_cost_required_subsidy_for_no_loss_l279_279598

open Real

/-- Define the monthly processing cost as a function of the processing volume. -/
def monthly_processing_cost (x : ℝ) : ℝ :=
  (1/2) * x ^ 2 - 200 * x + 80000

/-- Define the average processing cost per ton. -/
def average_processing_cost_per_ton (x : ℝ) : ℝ :=
  monthly_processing_cost x / x

/-- Define the monthly profit as a function of the processing volume. -/
def monthly_profit (x : ℝ) : ℝ :=
  100 * x - monthly_processing_cost x

/-- The lowest average processing cost per ton is 200 yuan
    when the monthly processing volume is 400 tons. -/
theorem lowest_average_processing_cost :
  ∃ x : ℝ, 400 ≤ x ∧ x ≤ 600 ∧ average_processing_cost_per_ton x = 200 := 
by
  -- This is the starting point of our proof
  use 400
  split
  . -- Prove 400 ≤ 400
    linarith
  split
  . -- Prove 400 ≤ 600
    linarith
  . -- Prove the average processing cost per ton at 400 tons is 200
    have h : average_processing_cost_per_ton 400 = (1/2) * 400 + 80000 / 400 - 200 := by sorry
    rw h
    norm_num

/-- The unit needs a subsidy of 40000 yuan to avoid losses, given the conditions. -/
theorem required_subsidy_for_no_loss :
  ∀ x, 400 ≤ x ∧ x ≤ 600 → monthly_profit x ≤ -40000 :=
by
  -- This is the starting point of our proof
  intro x
  intro h
  -- Prove that the monthly profit is less than or equal to -40000
  have h1 : monthly_profit 400 = -40000 := by sorry
  have h2 : monthly_profit x ≤ monthly_profit 400 := by sorry
  linarith

end lowest_average_processing_cost_required_subsidy_for_no_loss_l279_279598


namespace polynomial_injective_over_Q_not_R_l279_279693

theorem polynomial_injective_over_Q_not_R (f : ℚ → ℚ) (g : ℝ → ℝ) :
  f = (λ x, x^3 - 2 * x) ∧ g = (λ x, x^3 - 2 * x) →
  (∀ (x y : ℚ), f x = f y → x = y) ∧ ¬ (∀ (x y : ℝ), g x = g y → x = y) :=
by {
  sorry
}

end polynomial_injective_over_Q_not_R_l279_279693


namespace age_ratio_l279_279472

def Kul : ℕ := 22
def Saras : ℕ := 33

theorem age_ratio : (Saras / Kul : ℚ) = 3 / 2 := by
  sorry

end age_ratio_l279_279472


namespace frequency_within_10_40_l279_279654

def sample_intervals : List (Set ℝ) := [
  Set.Ioc 0 10,
  Set.Ioc 10 20,
  Set.Ioc 20 30,
  Set.Ioc 30 40,
  Set.Ioc 40 50,
  Set.Ioc 50 60,
  Set.Ioc 60 70 ]

def sample_frequencies : List ℕ := [
  12, 13, 15, 24, 16, 13, 7 ]

def total_sample_size := 100

theorem frequency_within_10_40 :
  (13 + 15 + 24) / total_sample_size = 0.52 := by
  sorry

end frequency_within_10_40_l279_279654


namespace find_a_l279_279606

-- Define the necessary variables
variables (a b : ℝ) (t : ℝ)

-- Given conditions
def b_val : ℝ := 2120
def t_val : ℝ := 0.5

-- The statement we need to prove
theorem find_a (h: b = b_val) (h2: t = t_val) (h3: t = a / b) : a = 1060 := by
  -- Placeholder for proof
  sorry

end find_a_l279_279606


namespace triangle_side_ineq_l279_279426

theorem triangle_side_ineq (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 :=
  sorry

end triangle_side_ineq_l279_279426


namespace find_f_neg_one_l279_279492

def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m)
def m : ℝ := -1

theorem find_f_neg_one (m : ℝ) (h_m : m = -1) (h_odd : ∀ x : ℝ, f (-x) = -f x) : f (-1) = -3 := 
by
  have h_def : f x = if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m),
  from sorry,
  sorry

end find_f_neg_one_l279_279492


namespace invalid_votes_percentage_l279_279096

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_candidate2 : ℕ) (valid_votes_percentage_candidate1 : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_valid_votes_candidate2 : valid_votes_candidate2 = 2700)
  (h_valid_votes_percentage_candidate1 : valid_votes_percentage_candidate1 = 55) :
  ((total_votes - (valid_votes_candidate2 * 100 / (100 - valid_votes_percentage_candidate1))) * 100 / total_votes) = 20 :=
by sorry

end invalid_votes_percentage_l279_279096


namespace train_speed_excluding_stoppages_l279_279708

theorem train_speed_excluding_stoppages
  (S : ℝ) -- speed excluding stoppages
  (effective_speed : ℝ := 40) -- effective speed including stoppages in kmph
  (stoppage_time_minutes : ℝ := 15.56) -- stoppage time in minutes
  (stoppage_time_hours : ℝ := stoppage_time_minutes / 60) -- convert minutes to hours
  (effective_time_moving : ℝ := 1 - stoppage_time_hours) -- effective time moving in hours
  (distance_covered : ℝ := effective_speed * effective_time_moving) -- distance covered in km
  (S_definition : S = distance_covered) : S = 29.63 :=
by
  sorry

end train_speed_excluding_stoppages_l279_279708


namespace trader_profit_l279_279286

theorem trader_profit (P : ℝ) :
  let purchase_price := 0.80 * P in
  let selling_price := purchase_price * 1.55 in
  (selling_price - P) / P * 100 = 24 := 
by
  sorry

end trader_profit_l279_279286


namespace binom_2057_1_l279_279323

theorem binom_2057_1 : Nat.binomial 2057 1 = 2057 := by
  sorry

end binom_2057_1_l279_279323


namespace right_triangle_points_on_ellipse_l279_279377

theorem right_triangle_points_on_ellipse : 
  let ellipse := {p : ℝ × ℝ | (p.1^2)/4 + (p.2^2)/2 = 1}
  let f1 : ℝ × ℝ := (-√2, 0)
  let f2 : ℝ × ℝ := (√2, 0)
  {P : ℝ × ℝ | P ∈ ellipse ∧ 
    (∃ θ, (f1 - P).angle θ (f2 - P) = 90°)}.card = 6 := 
by sorry

end right_triangle_points_on_ellipse_l279_279377


namespace find_original_cost_price_l279_279642

noncomputable def original_cost_price (C : ℝ) : Prop :=
  let friend_buy_price := 0.87 * C in
  let maintenance_cost := 0.10 * friend_buy_price in
  let total_cost := friend_buy_price + maintenance_cost in
  let resale_price_no_tax := 54000 / 1.05 in
  let desired_sale_price := total_cost * 1.20 in
  resale_price_no_tax = desired_sale_price

theorem find_original_cost_price : ∃ C : ℝ, original_cost_price C ∧ C = 44792.86 :=
  sorry

end find_original_cost_price_l279_279642


namespace find_lambda_l279_279436

theorem find_lambda (λ : ℝ) :
  let C := (-1, 2)
  let r := Real.sqrt 5
  let line := λ x y => x + 2 * y + 5 + λ
  let circle := λ x y => x^2 + y^2 + 2 * x - 4 * y
  (∀ x y, circle x y = 0 → line x y = 0) ↔ (λ = -3 ∨ λ = -13) := by sorry

end find_lambda_l279_279436


namespace quadrilateral_ratio_eq_lambda_l279_279272

theorem quadrilateral_ratio_eq_lambda
  (A B C D E F G H E1 F1 G1 H1 : Type)
  (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ)
  (h1 : AE / EB * BF / FC * CG / GD * DH / HA = 1)
  (h2 : E1A / AH1 = λ) : 
  F1C / CG1 = λ :=
by
  sorry

end quadrilateral_ratio_eq_lambda_l279_279272


namespace omega_range_l279_279774

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l279_279774


namespace gwen_walked_time_l279_279342

-- Definition of given conditions
def time_jogged : ℕ := 15
def ratio_jogged_to_walked (j w : ℕ) : Prop := j * 3 = w * 5

-- Definition to state the exact time walked with given ratio
theorem gwen_walked_time (j w : ℕ) (h1 : j = time_jogged) (h2 : ratio_jogged_to_walked j w) : w = 9 :=
by
  sorry

end gwen_walked_time_l279_279342


namespace function_properties_l279_279790

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (2^(x + 1))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l279_279790


namespace tan_alpha_l279_279017

variable (α : Real)
-- Condition 1: α is an angle in the second quadrant
-- This implies that π/2 < α < π and sin α = 4 / 5
variable (h1 : π / 2 < α ∧ α < π) 
variable (h2 : Real.sin α = 4 / 5)

theorem tan_alpha : Real.tan α = -4 / 3 :=
by
  sorry

end tan_alpha_l279_279017


namespace gray_region_area_l279_279840

-- Definitions based on given conditions
def radius_inner (r : ℝ) := r
def radius_outer (r : ℝ) := r + 3

-- Statement to prove: the area of the gray region
theorem gray_region_area (r : ℝ) : 
  (π * (radius_outer r)^2 - π * (radius_inner r)^2) = 6 * π * r + 9 * π := by
  sorry

end gray_region_area_l279_279840


namespace avg_weight_of_22_boys_l279_279439

theorem avg_weight_of_22_boys:
  let total_boys := 30
  let avg_weight_8 := 45.15
  let avg_weight_total := 48.89
  let total_weight_8 := 8 * avg_weight_8
  let total_weight_all := total_boys * avg_weight_total
  ∃ A : ℝ, A = 50.25 ∧ 22 * A + total_weight_8 = total_weight_all :=
by {
  sorry 
}

end avg_weight_of_22_boys_l279_279439


namespace correct_arrangements_adjacent_f_incorrect_arrangements_alternate_fm_correct_arrangements_alternate_fm_correct_arrangements_non_adjacent_f_correct_arrangements_no_a_ends_l279_279216

-- Definition to represent students
inductive Student
| male1 
| male2 
| male3 
| female1 
| female2 
| female3

open Student

-- Auxiliary function to calculate factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Problem A
def arrangements_adjacent_f : ℕ := factorial 4 * factorial 3 -- 4! * 3!

theorem correct_arrangements_adjacent_f : arrangements_adjacent_f = 144 :=
by simp [arrangements_adjacent_f, factorial]; sorry

-- Problem B
def arrangements_alternate_fm : ℕ := 2 * (factorial 3 * factorial 3) -- 2 * 3! * 3!

theorem incorrect_arrangements_alternate_fm : arrangements_alternate_fm ≠ 96 :=
by simp [arrangements_alternate_fm, factorial]; sorry

theorem correct_arrangements_alternate_fm : arrangements_alternate_fm = 72 :=
by simp [arrangements_alternate_fm, factorial]; sorry

-- Problem C
def arrangements_non_adjacent_f : ℕ := factorial 3 * (factorial 4 / factorial 1) -- 3! * (4! / (4-3)!)

theorem correct_arrangements_non_adjacent_f : arrangements_non_adjacent_f = 144 :=
by simp [arrangements_non_adjacent_f, factorial]; sorry

-- Problem D
def arrangements_no_a_ends : ℕ := 4 * factorial 5 -- 4 * 5!

theorem correct_arrangements_no_a_ends : arrangements_no_a_ends = 480 :=
by simp [arrangements_no_a_ends, factorial]; sorry

end correct_arrangements_adjacent_f_incorrect_arrangements_alternate_fm_correct_arrangements_alternate_fm_correct_arrangements_non_adjacent_f_correct_arrangements_no_a_ends_l279_279216


namespace probability_three_red_cards_l279_279690

theorem probability_three_red_cards : 
  let total_cards := 60
      total_red_cards := 30
      total_ways_draw_any_three := (total_cards - 0) * (total_cards - 1) * (total_cards - 2)
      total_ways_draw_three_red := (total_red_cards - 0) * (total_red_cards - 1) * (total_red_cards - 2) in
  (total_ways_draw_three_red / total_ways_draw_any_three : ℚ) = 29 / 247 :=
by sorry

end probability_three_red_cards_l279_279690


namespace value_of_a_l279_279816

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := (2 : ℝ) * x - y - 1 = 0

def line2 (x y a : ℝ) : Prop := (2 : ℝ) * x + (a + 1) * y + 2 = 0

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 x y) → (line2 x y a)

-- The theorem to be proved
theorem value_of_a (a : ℝ) : parallel_lines a → a = -2 :=
sorry

end value_of_a_l279_279816


namespace f_has_center_of_symmetry_l279_279163

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.sin (4 * x + 7 * Real.pi / 3)) / (Real.sin (2 * x + 2 * Real.pi / 3))

-- Define the statement that f(x) has a center of symmetry at the given points
def has_center_of_symmetry (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (2 * c - x) = f x

-- Define the specific center of symmetry as per the problem
def center_of_symmetry_spec (c : ℝ) : Prop :=
  ∃ k : ℤ, c = k * Real.pi / 2 - Real.pi / 12

-- Lean 4 statement: Prove that f(x) has a center of symmetry at the given specified points
theorem f_has_center_of_symmetry :
  ∃ c : ℝ, center_of_symmetry_spec c ∧ has_center_of_symmetry f c :=
by sorry

end f_has_center_of_symmetry_l279_279163


namespace line_equation_curve_equation_distance_AB_l279_279449

variable {t : ℝ} {x y : ℝ} {θ : ℝ} {ρ : ℝ}

theorem line_equation (t : ℝ) :
  (∃ t, x = - (sqrt 2) / 2 * t ∧ y = -4 + (sqrt 2) / 2 * t) ↔ (x + y + 4 = 0) :=
sorry

theorem curve_equation (θ : ℝ) :
  (ρ = 4 * cos θ) ↔ (x^2 + y^2 - 4 * x = 0) :=
sorry

theorem distance_AB :
  (∃ t, x = 1 - (sqrt 2) / 2 * t ∧ y = (sqrt 2) / 2 * t) ∧ (x^2 + y^2 - 4 * x = 0) →
  (|t1 - t2| = sqrt 14) :=
sorry

end line_equation_curve_equation_distance_AB_l279_279449


namespace increasing_order_l279_279728

noncomputable def a := real.logb 0.6 0.5
noncomputable def b := real.log 0.5
noncomputable def c := real.rpow 0.6 0.5

theorem increasing_order : b < c ∧ c < a :=
by
  sorry

end increasing_order_l279_279728


namespace carter_reading_pages_l279_279686

theorem carter_reading_pages (c l o : ℕ)
  (h1: c = l / 2)
  (h2: l = o + 20)
  (h3: o = 40) : c = 30 := by
  sorry

end carter_reading_pages_l279_279686


namespace determine_K_value_l279_279340

theorem determine_K_value (K : ℕ) : 32^4 * 4^5 = 2^K → K = 30 := 
by 
  intro h
  rw [pow_four, pow_five] at h
  rw [pow_mul, pow_add] at h
  have h₁ : (32 : ℕ) = 2^5 := rfl
  have h₂ : (4 : ℕ) = 2^2 := rfl
  rw [h₁, h₂] at h
  rw [←pow_mul, ←pow_mul] at h
  exact Or.resolve_right (pow_eq_pow) h

end determine_K_value_l279_279340


namespace flowers_are_55_percent_daisies_l279_279639

noncomputable def percent_daisies (F : ℝ) (yellow : ℝ) (white_daisies : ℝ) (yellow_daisies : ℝ) : ℝ :=
  (yellow_daisies + white_daisies) / F * 100

theorem flowers_are_55_percent_daisies (F : ℝ) (yellow_t : ℝ) (yellow_d : ℝ) (white : ℝ) (white_d : ℝ) :
    yellow_t = 0.5 * yellow →
    yellow_d = yellow - yellow_t →
    white_d = (2 / 3) * white →
    yellow = (7 / 10) * F →
    white = F - yellow →
    percent_daisies F yellow white_d yellow_d = 55 :=
by
  sorry

end flowers_are_55_percent_daisies_l279_279639


namespace oldest_child_age_l279_279190

theorem oldest_child_age 
  (ages: Fin 7 → ℕ)
  (h_diff: ∀ i : Fin 6, ages i + 3 = ages (⟨i.val + 1, Nat.lt_of_lt_pred i.isLt⟩ : Fin 7))
  (h_avg: (∑ i, ages i) = 56)
  (h_distinct: ∀ i j, i ≠ j → ages i ≠ ages j) :
  ages ⟨6, by decide⟩ = 17 :=
by
  sorry

end oldest_child_age_l279_279190


namespace range_of_omega_for_zeros_in_interval_l279_279781

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - 1

theorem range_of_omega_for_zeros_in_interval (ω : ℝ) (hω_positve : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f ω x = 0 → 2 ≤ ω ∧ ω < 3) :=
sorry

end range_of_omega_for_zeros_in_interval_l279_279781


namespace original_radius_of_spherical_balloon_l279_279285

noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4.0 / 3.0) * Real.pi * R^3
noncomputable def volume_of_hemisphere (r : ℝ) : ℝ := (2.0 / 3.0) * Real.pi * r^3

theorem original_radius_of_spherical_balloon :
  ∃ (R : ℝ), volume_of_hemisphere 6 = volume_of_sphere R ∧ R ≈ 4.76 :=
sorry

end original_radius_of_spherical_balloon_l279_279285


namespace tan_identity_l279_279425

theorem tan_identity (A B : ℝ) (hA : A = 30) (hB : B = 30) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = (4 + 2 * Real.sqrt 3)/3 := by
  sorry

end tan_identity_l279_279425


namespace problem_l279_279389

def a : ℝ := 3 * (3 - Real.pi) ^ 3
def b : ℝ := 4 * (2 - Real.pi) ^ 4

theorem problem (h1 : a = 3 * (3 - Real.pi) ^ 3) (h2 : b = 4 * (2 - Real.pi) ^ 4) :
  a + b = 1 :=
sorry

end problem_l279_279389


namespace andy_late_l279_279671

theorem andy_late
  (school_start : ℕ := 480) -- 8:00 AM in minutes since midnight
  (normal_travel_time : ℕ := 30)
  (red_lights : ℕ := 4)
  (red_light_wait_time : ℕ := 3)
  (construction_wait_time : ℕ := 10)
  (departure_time : ℕ := 435) -- 7:15 AM in minutes since midnight
  : ((school_start - departure_time) < (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time)) →
    school_start + (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time - (school_start - departure_time)) = school_start + 7 :=
by
  -- This skips the proof part
  sorry

end andy_late_l279_279671


namespace period_sine_minus_cosine_is_2pi_l279_279232

noncomputable def period_of_sine_minus_cosine := 
∀ (x : ℝ), sin (x + 2 * Real.pi) - cos (x + 2 * Real.pi) = sin x - cos x

theorem period_sine_minus_cosine_is_2pi : period_of_sine_minus_cosine := 
by
  intro x
  sorry

end period_sine_minus_cosine_is_2pi_l279_279232


namespace points_on_parabola_relationship_l279_279432

theorem points_on_parabola_relationship (h y1 y2 y3 : ℝ) :
    (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
    (hA : A = (-3, y1))
    (hB : B = (2, y2))
    (hC : C = (3, y3))
    (hA_on_parabola : A.2 = -(A.1+2)^2 + h)
    (hB_on_parabola : B.2 = -(B.1+2)^2 + h)
    (hC_on_parabola : C.2 = -(C.1+2)^2 + h) :
    y3 < y2 ∧ y2 < y1 := by
  sorry

end points_on_parabola_relationship_l279_279432


namespace no_integer_right_triangle_with_median_7point5_l279_279460

theorem no_integer_right_triangle_with_median_7point5 
  (a b c : ℕ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
  (h_median : (∃ m, m = 7.5 ∧ 
    (m ^ 2 = a ^ 2 / 4 + b ^ 2 
     ∨ m ^ 2 = b ^ 2 / 4 + a ^ 2 
     ∨ m ^ 2 = (2 * a ^ 2 + 2 * b ^ 2 - c ^ 2) / 4))
  ) 
  : False := 
sorry

end no_integer_right_triangle_with_median_7point5_l279_279460


namespace greatest_area_difference_l279_279966

theorem greatest_area_difference (a b : ℕ) (h₁ : a = 36) (h₂ : 2*a + 2*b = 144) : 
  greatest_possible_area_difference a b = 1296 := by
  sorry

end greatest_area_difference_l279_279966


namespace minimum_cookies_left_l279_279583

theorem minimum_cookies_left (x : ℕ) (h : 0 < x) : 
  let y := 23 * x + 3 in
  (10 * y) % 23 = 7 :=
by
  let y := 23 * x + 3
  have h1 : y % 23 = 3 := sorry
  have h2 : (10 * y) % 23 = (10 * 3) % 23 := sorry
  have h3 : (10 * 3) % 23 = 7 := sorry
  exact h3

end minimum_cookies_left_l279_279583


namespace product_of_8_dice_is_divisible_by_8_l279_279673

open ProbabilityMeasure
open Classical

-- Define a standard 6-sided die
inductive Die
| face1 | face2 | face3 | face4 | face5 | face6

namespace Die

instance : Inhabited Die := ⟨Die.face1⟩

def eq_classes : Finset (Fin 6) := { ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨5, by decide⟩, ⟨6, by decide⟩ }

def Roll : Die → ℕ
| Die.face1 => 1
| Die.face2 => 2
| Die.face3 => 3
| Die.face4 => 4
| Die.face5 => 5
| Die.face6 => 6

def probability_space : ProbabilityMeasure (Finset Die) := sorry -- Construed probability space for the 8 rolls

-- Event indicating the product of the rolls
def event_production_divisible_by_8 (outcome: Fin 8 → Die) : Prop :=
  (List.prod (List.ofFn (λ i => (Roll (outcome i : Die)))) % 8 = 0)

-- Define measure for this event
def production_divisible_by_8_measure : ℚ := 
  probability_space.measure { outcome | event_production_divisible_by_8 outcome }

-- Statement of the main theorem
theorem product_of_8_dice_is_divisible_by_8 : production_divisible_by_8_measure = 35 / 36 := 
  sorry -- Proof is omitted

end product_of_8_dice_is_divisible_by_8_l279_279673


namespace vote_proportion_inequality_l279_279826

theorem vote_proportion_inequality
  (a b k : ℕ)
  (hb_odd : b % 2 = 1)
  (hb_min : 3 ≤ b)
  (vote_same : ∀ (i j : ℕ) (hi hj : i ≠ j) (votes : ℕ → ℕ), ∃ (k_max : ℕ), ∀ (cont : ℕ), votes cont ≤ k_max) :
  (k : ℚ) / a ≥ (b - 1) / (2 * b) := sorry

end vote_proportion_inequality_l279_279826


namespace exact_value_of_expression_l279_279700

theorem exact_value_of_expression :
  √((2 - cos^2 (Real.pi / 8)) * (2 - cos^2 (2 * Real.pi / 8)) * (2 - cos^2 (3 * Real.pi / 8))) = sqrt(3) / 2 :=
by
  sorry

end exact_value_of_expression_l279_279700


namespace problem_l279_279013

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end problem_l279_279013


namespace trajectory_hyperbola_l279_279210

-- Definitions: 
variables {z : ℂ} {x y : ℝ}

-- Condition given in problem
def condition (z : ℂ) : Prop := 
  ∥ ∥z - 1∥ - ∥z + complex.I∥ ∥ = 1

-- Proving that the trajectory is a hyperbola
theorem trajectory_hyperbola (h : condition (z)) : true := sorry

end trajectory_hyperbola_l279_279210


namespace A_divides_99_l279_279280

def is_divisible_by (N A : ℕ) : Prop := ∃ k, N = k * A

def reverse_number (N : ℕ) : ℕ := 
  let digits := N.digits 10
  digits.reverse.foldl (λ x y, x * 10 + y) 0

theorem A_divides_99 (A : ℕ) (h1 : ∀ N : ℕ, is_divisible_by N A → is_divisible_by (reverse_number N) A) : is_divisible_by 99 A :=
sorry

end A_divides_99_l279_279280


namespace statement_B_statement_D_l279_279615

variable {a b c d : ℝ}

theorem statement_B (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : (c / a) > (c / b) := 
by sorry

theorem statement_D (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : (a * c) < (b * d) := 
by sorry

end statement_B_statement_D_l279_279615


namespace intersection_eq_l279_279624

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_eq : A ∩ B = {-1, 3} := 
by
  sorry

end intersection_eq_l279_279624


namespace range_of_a_l279_279141

-- Definitions of propositions p and q
def p (a : ℝ) : Prop := 0 < a ∧ a < 1 ∧ ∀ x, a^x > 1 ↔ x ∈ Iio 0
def q (a : ℝ) : Prop := a > 0 ∧ (∀ x, ax^2 - x + 2 > 0 ↔ true) ∧ (1 - 8 * a < 0)

-- The final theorem in Lean
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) :
  a ∈ Ioo 0 (1/8 : ℝ) ∨ a ≥ 1 :=
sorry

end range_of_a_l279_279141


namespace monotonic_criteria_l279_279791

noncomputable def monotonic_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 4 → 
  (-2 * x₁^2 + m * x₁ + 1) ≤ (-2 * x₂^2 + m * x₂ + 1)

theorem monotonic_criteria (m : ℝ) : 
  (m ≤ -4 ∨ m ≥ 16) ↔ monotonic_interval m := 
sorry

end monotonic_criteria_l279_279791


namespace john_drove_total_distance_l279_279855

-- Define different rates and times for John's trip
def rate1 := 45 -- mph
def rate2 := 55 -- mph
def time1 := 2 -- hours
def time2 := 3 -- hours

-- Define the distances for each segment of the trip
def distance1 := rate1 * time1
def distance2 := rate2 * time2

-- Define the total distance
def total_distance := distance1 + distance2

-- The theorem to prove that John drove 255 miles in total
theorem john_drove_total_distance : total_distance = 255 :=
by
  sorry

end john_drove_total_distance_l279_279855


namespace geometric_sequence_property_l279_279455

theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a 1 / a 0) (h₁ : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_property_l279_279455


namespace product_eval_at_3_l279_279707

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end product_eval_at_3_l279_279707


namespace valid_range_for_b_l279_279035

noncomputable def f (x b : ℝ) : ℝ := -x^2 + 2 * x + b^2 - b + 1

theorem valid_range_for_b (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x b > 0) → (b < -1 ∨ b > 2) :=
by
  sorry

end valid_range_for_b_l279_279035


namespace distinct_remainders_l279_279334

def f : ℕ → ℕ
| 1     := 1
| (n+1) := f n + 2^(f n)

theorem distinct_remainders (n : ℕ) : 
  ∀ m k : ℕ, m < k → k ≤ 3^n → (f m % 3^n) ≠ (f k % 3^n) := 
sorry

end distinct_remainders_l279_279334


namespace orthocenter_locus_theorem_l279_279957

-- Define points A, B, and varying point C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 0 }

variables (m k : ℝ)
def C : Point := { x := m, y := k }

-- Define the orthocenter locus condition
def orthocenter_locus (x : ℝ) : ℝ := - (x^2 - 1) / k

-- The theorem statement 
theorem orthocenter_locus_theorem : 
  ∀ x, ∃ y, y = orthocenter_locus k x :=
  by
  sorry

end orthocenter_locus_theorem_l279_279957


namespace flowers_total_l279_279999

theorem flowers_total (yoojung_flowers : ℕ) (namjoon_flowers : ℕ)
 (h1 : yoojung_flowers = 32)
 (h2 : yoojung_flowers = 4 * namjoon_flowers) :
  yoojung_flowers + namjoon_flowers = 40 := by
  sorry

end flowers_total_l279_279999


namespace sara_pumpkins_l279_279892

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end sara_pumpkins_l279_279892


namespace darcy_folded_shorts_l279_279332

-- Define the conditions
def total_shirts : Nat := 20
def total_shorts : Nat := 8
def folded_shirts : Nat := 12
def remaining_pieces : Nat := 11

-- Expected result to prove
def folded_shorts : Nat := 5

-- The statement to prove
theorem darcy_folded_shorts : total_shorts - (remaining_pieces - (total_shirts - folded_shirts)) = folded_shorts :=
by
  sorry

end darcy_folded_shorts_l279_279332


namespace range_of_m_l279_279383

theorem range_of_m (m : ℝ) (p q : Prop) 
  (hp : (∃ x y : ℝ, x^2 / (2 * m) - y^2 / (1 - m) = 1 ∧  0 < 2*m ∧ 2*m < 1 - m)) 
  (hq : (∃ y x : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 1 < (5 + m) / 5 < 4)):
  p ∨ q → 0 < m ∧ m < 15 :=
by sorry

end range_of_m_l279_279383


namespace happy_valley_zoo_animal_arrangement_l279_279188

theorem happy_valley_zoo_animal_arrangement :
  let parrots := 5
  let dogs := 3
  let cats := 4
  let total_animals := parrots + dogs + cats
  (total_animals = 12) →
    (∃ no_of_ways_to_arrange,
      no_of_ways_to_arrange = 2 * (parrots.factorial) * (dogs.factorial) * (cats.factorial) ∧
      no_of_ways_to_arrange = 34560) :=
by
  sorry

end happy_valley_zoo_animal_arrangement_l279_279188


namespace problem1_problem2_l279_279397

noncomputable def f (x a : ℝ) : ℝ := 1 + x + a * x^2

theorem problem1 (x : ℝ) (h : x < 0) : ¬ (∃ M > 0, ∀ y ∈ set.Ioi (-∞), abs (f y (-1)) ≤ M) :=
by sorry

theorem problem2 (a : ℝ) 
  (h : ∀ x ∈ set.Icc 1 4, abs (f x a) ≤ 3) : a ∈ set.Icc (-(1/2)) (-(1/8)) :=
by sorry

end problem1_problem2_l279_279397


namespace number_of_divisors_of_18n_cubed_l279_279137

theorem number_of_divisors_of_18n_cubed 
  (n : ℕ) 
  (h1 : ∃ p : ℕ, nat.prime p ∧ n = p ^ 12) 
  (h2 : n % 2 = 1) 
  (h3 : (∃ k : ℕ, nat.totient n = k ∧ k = 13)) : 
  nat.totient (18 * n ^ 3) = 222 := 
by
  sorry

end number_of_divisors_of_18n_cubed_l279_279137


namespace francis_hours_left_l279_279074

theorem francis_hours_left :
  let total_hours : ℕ := 24 in
  let sleeping_fraction : ℚ := 1/3 in
  let studying_fraction : ℚ := 1/4 in
  let eating_fraction : ℚ := 1/8 in
  let hours_spent : ℚ := total_hours * (sleeping_fraction + studying_fraction + eating_fraction) in
  let hours_left : ℚ := total_hours - hours_spent in
  hours_left = 7 := 
by
  sorry

end francis_hours_left_l279_279074


namespace vectors_are_perpendicular_l279_279417

noncomputable def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem vectors_are_perpendicular {u v : ℝ × ℝ} :
  ∥(u.1 + v.1, u.2 + v.2)∥ = ∥(u.1 - v.1, u.2 - v.2)∥ → is_perpendicular u v :=
by
  sorry

end vectors_are_perpendicular_l279_279417


namespace polygon_interior_exterior_relation_l279_279026

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end polygon_interior_exterior_relation_l279_279026


namespace inequality_relationship_l279_279811

theorem inequality_relationship (x : ℝ) (h : 0 < x ∧ x < 1) :
  2^x > sqrt x ∧ sqrt x > log x :=
sorry

end inequality_relationship_l279_279811


namespace larger_angle_at_3_30_l279_279231

def hour_hand_angle_3_30 : ℝ := 105.0
def minute_hand_angle_3_30 : ℝ := 180.0
def smaller_angle_between_hands : ℝ := abs (minute_hand_angle_3_30 - hour_hand_angle_3_30)
def larger_angle_between_hands : ℝ := 360.0 - smaller_angle_between_hands

theorem larger_angle_at_3_30 :
  larger_angle_between_hands = 285.0 := 
  sorry

end larger_angle_at_3_30_l279_279231


namespace arithmetic_geometric_sequence_l279_279008

theorem arithmetic_geometric_sequence
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 2 = 3)
  (h3 : (a 1), (a 3), (a 7) form a geometric sequence)
  (h4 : ∀ n, b n = if n % 2 = 1 then 2^(a n) else (2 / 3) * a n)
  (h5 : S n = ∑ i in (range (n+1)), b i) :
  (∀ n, a n = n + 1) ∧ (S 16 = (1 / 3) * 4^9 + 52) :=
by
  sorry

end arithmetic_geometric_sequence_l279_279008


namespace find_f_neg_one_l279_279491

def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m)
def m : ℝ := -1

theorem find_f_neg_one (m : ℝ) (h_m : m = -1) (h_odd : ∀ x : ℝ, f (-x) = -f x) : f (-1) = -3 := 
by
  have h_def : f x = if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m),
  from sorry,
  sorry

end find_f_neg_one_l279_279491


namespace quadratic_function_value_2_l279_279755

variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^2 + a * x + b

theorem quadratic_function_value_2 :
  f a b 2 = 3 :=
by
  -- Definitions and assumptions to be used
  sorry

end quadratic_function_value_2_l279_279755


namespace symmetry_distance_l279_279404

theorem symmetry_distance (m : ℝ) 
  (h1 : ∃ a b, a ≠ b ∧ (-a^2 + m^2 * a = 0 ∧ -b^2 + m^2 * b = 0))
  (h2 : ∃ c d, c ≠ d ∧ (c^2 - m^2 = 0 ∧ d^2 - m^2 = 0))
  (h3 : ∀ x1 x2 x3 x4, (x1, x2, x3, x4 ∈ {a, b, c, d}) ∧ (|x2 - x1| = |x3 - x2| ∧ |x4 - x3| = |x3 - x2|)) 
  : |(m^2 / 2) - 0| = 2 := 
by
  sorry

end symmetry_distance_l279_279404


namespace total_packs_sold_l279_279913

theorem total_packs_sold (Robyn_sells : ℕ) (Lucy_sells : ℕ) (hR : Robyn_sells = 55) (hL : Lucy_sells = 43) : 
  Robyn_sells + Lucy_sells = 98 := 
by 
  rw [hR, hL]
  simp
  exact eq.refl 98

end total_packs_sold_l279_279913


namespace linear_function_does_not_pass_third_quadrant_l279_279431

/-
Given an inverse proportion function \( y = \frac{a^2 + 1}{x} \), where \( a \) is a constant, and given two points \( (x_1, y_1) \) and \( (x_2, y_2) \) on the same branch of this function, 
with \( b = (x_1 - x_2)(y_1 - y_2) \), prove that the graph of the linear function \( y = bx - b \) does not pass through the third quadrant.
-/

theorem linear_function_does_not_pass_third_quadrant 
  (a x1 x2 : ℝ) 
  (y1 y2 : ℝ)
  (h1 : y1 = (a^2 + 1) / x1) 
  (h2 : y2 = (a^2 + 1) / x2) 
  (h3 : b = (x1 - x2) * (y1 - y2)) : 
  ∃ b, ∀ x y : ℝ, (y = b * x - b) → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) :=
by 
  sorry

end linear_function_does_not_pass_third_quadrant_l279_279431


namespace undefined_expr_iff_l279_279721

theorem undefined_expr_iff (a : ℝ) : (∃ x, x = (a^2 - 9) ∧ x = 0) ↔ (a = -3 ∨ a = 3) :=
by
  sorry

end undefined_expr_iff_l279_279721


namespace lines_intersect_at_same_points_l279_279412

-- Definitions of linear equations in system 1 and system 2
def line1 (a1 b1 c1 x y : ℝ) := a1 * x + b1 * y = c1
def line2 (a2 b2 c2 x y : ℝ) := a2 * x + b2 * y = c2
def line3 (a3 b3 c3 x y : ℝ) := a3 * x + b3 * y = c3
def line4 (a4 b4 c4 x y : ℝ) := a4 * x + b4 * y = c4

-- Equivalence condition of the systems
def systems_equivalent (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :=
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y)

-- Proof statement that the four lines intersect at the same set of points
theorem lines_intersect_at_same_points (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :
  systems_equivalent a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 →
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y) :=
by
  intros h_equiv x y
  exact h_equiv x y

end lines_intersect_at_same_points_l279_279412


namespace first_day_of_month_l279_279182

theorem first_day_of_month (h : weekday 30 = "Wednesday") : weekday 1 = "Tuesday" :=
sorry

end first_day_of_month_l279_279182


namespace discriminant_of_quadratic_eq_l279_279337

-- Define the coefficients of the quadratic equation
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- State the theorem that we want to prove
theorem discriminant_of_quadratic_eq : discriminant a b c = 61 := by
  sorry

end discriminant_of_quadratic_eq_l279_279337


namespace volleyball_min_wins_l279_279833

theorem volleyball_min_wins (teams : ℕ) (total_games : ℕ) (total_wins : ℕ) (no_draws : Prop) : teams = 6 → total_games = 15 → total_wins = 15 → (∃ t_wins : ℕ, t_wins ≥ 3 ∧ ∃ team : ℕ, team < teams ∧ team_wins teams total_games team = t_wins) :=
by
  intros h_teams h_total_games h_total_wins
  sorry

end volleyball_min_wins_l279_279833


namespace yellow_tint_percentage_l279_279644

theorem yellow_tint_percentage (V₀ : ℝ) (P₀Y : ℝ) (V_additional : ℝ) 
  (hV₀ : V₀ = 40) (hP₀Y : P₀Y = 0.35) (hV_additional : V_additional = 8) : 
  (100 * ((V₀ * P₀Y + V_additional) / (V₀ + V_additional)) = 45.83) :=
by
  sorry

end yellow_tint_percentage_l279_279644


namespace linear_function_difference_l279_279564

variables {R : Type*} [LinearOrderedField R]

noncomputable def g (x : R) : R := sorry

theorem linear_function_difference (g : R → R) (h_linear : ∀ x y, g (x + y) = g x + g y) (h_diff : ∀ d : R, g (d + 1) - g d = 5) :
  g 0 - g 10 = -50 :=
by {
  sorry
}

end linear_function_difference_l279_279564


namespace simplest_quadratic_surd_is_sqrt3_l279_279245

def isQuadraticSurd (x : ℝ) : Prop := ∃ n : ℕ, (n > 0) ∧ (¬ (∃ m : ℕ, m^2 = n)) ∧ (x = Real.sqrt n)

def simplestQuadraticSurd : ℝ :=
  if isQuadraticSurd (Real.sqrt (1/2)) then Real.sqrt (1/2)
  else if isQuadraticSurd (Real.cbrt 8) then Real.cbrt 8
  else if isQuadraticSurd (Real.sqrt 3) then Real.sqrt 3
  else if isQuadraticSurd (Real.sqrt 16) then Real.sqrt 16
  else 0

theorem simplest_quadratic_surd_is_sqrt3 :
  simplestQuadraticSurd = Real.sqrt 3 :=
by
  sorry

end simplest_quadratic_surd_is_sqrt3_l279_279245


namespace ellipse_standard_equation_hyperbola_standard_equation_l279_279717

-- Ellipse passing through P(-5, 0) and Q(0, 3)
theorem ellipse_standard_equation (a b : ℝ) :
  (-5 : ℝ, 0) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 } →
  (0, 3) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 } →
  a = 5 → b = 3 →
  { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }
  = { p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1 } :=
sorry

-- Hyperbola passing through M(-5, 3) with eccentricity e = √2
theorem hyperbola_standard_equation (a b e : ℝ) :
  e = real.sqrt 2 →
  a = b →
  (-5, 3) ∈ { p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1 } →
  { p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1 }
  = { p : ℝ × ℝ | p.1^2 / 16 - p.2^2 / 16 = 1 } :=
sorry

end ellipse_standard_equation_hyperbola_standard_equation_l279_279717


namespace time_reading_per_week_l279_279594

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l279_279594


namespace smallest_base10_integer_l279_279987

def is_valid_digit_base_6 (C : ℕ) : Prop := C ≤ 5
def is_valid_digit_base_8 (D : ℕ) : Prop := D ≤ 7

def CC_6_to_base10 (C : ℕ) : ℕ := 7 * C
def DD_8_to_base10 (D : ℕ) : ℕ := 9 * D

theorem smallest_base10_integer : ∃ C D : ℕ, 
  is_valid_digit_base_6 C ∧ 
  is_valid_digit_base_8 D ∧ 
  CC_6_to_base10 C = DD_8_to_base10 D ∧
  CC_6_to_base10 C = 63 := 
begin
  sorry
end

end smallest_base10_integer_l279_279987


namespace indicator_improvement_l279_279266

noncomputable def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def old_mean := 10
noncomputable def new_mean := 10.3

noncomputable def old_variance := 0.036
noncomputable def new_variance := 0.04

theorem indicator_improvement :
  let mean_diff := new_mean - old_mean
  let threshold := 2 * Real.sqrt ((old_variance + new_variance) / 10)
  mean_diff ≥ threshold :=
by {
  let mean_diff := new_mean - old_mean
  let threshold := 2 * Real.sqrt ((old_variance + new_variance) / 10)
  sorry
}

end indicator_improvement_l279_279266


namespace smallest_integer_CC6_DD8_l279_279984

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l279_279984


namespace correct_exponent_operation_l279_279614

theorem correct_exponent_operation (a b : ℝ) : 
  a^2 * a^3 = a^5 := 
by sorry

end correct_exponent_operation_l279_279614


namespace sine_function_zero_points_l279_279200

theorem sine_function_zero_points (ω : ℝ) (h_ω : ω > 0) 
    (h_zero_points : ∃ n : ℤ, n = 11 ∧ ∀ x ∈ (set.Icc (-π/2) (π/2)), sin (ω * x) = 0)
    : ω ∈ set.Ico 10 12 := 
sorry

end sine_function_zero_points_l279_279200


namespace imaginary_fraction_l279_279810

theorem imaginary_fraction (a : ℝ) (h : (2 + Complex.i) / (a - Complex.i) = 0 + (2 + a) * Complex.i / (a^2 + 1)) : 
  a = 1 / 2 :=
by
  sorry

end imaginary_fraction_l279_279810


namespace race_distance_l279_279831

theorem race_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 30 → D / 30 = D / t)
                      (h2 : ∀ t : ℝ, t = 45 → D / 45 = D / t)
                      (h3 : ∀ d : ℝ, d = 33.333333333333336 → D - (D / 45) * 30 = d) :
  D = 100 :=
sorry

end race_distance_l279_279831


namespace problem_a_problem_b_problem_c_l279_279249

theorem problem_a : (7 * (2 / 3) + 16 * (5 / 12)) = 11.3333 := by
  sorry

theorem problem_b : (5 - (2 / (5 / 3))) = 3.8 := by
  sorry

theorem problem_c : (1 + 2 / (1 + 3 / (1 + 4))) = 2.25 := by
  sorry

end problem_a_problem_b_problem_c_l279_279249


namespace paula_remaining_money_l279_279512

-- Definitions based on the conditions
def initialMoney : ℕ := 1000
def shirtCost : ℕ := 45
def pantsCost : ℕ := 85
def jacketCost : ℕ := 120
def shoeCost : ℕ := 95
def jeansOriginalPrice : ℕ := 140
def jeansDiscount : ℕ := 30 / 100  -- 30%

-- Using definitions to compute the spending and remaining money
def totalShirtCost : ℕ := 6 * shirtCost
def totalPantsCost : ℕ := 2 * pantsCost
def totalShoeCost : ℕ := 3 * shoeCost
def jeansDiscountValue : ℕ := jeansDiscount * jeansOriginalPrice
def jeansDiscountedPrice : ℕ := jeansOriginalPrice - jeansDiscountValue
def totalSpent : ℕ := totalShirtCost + totalPantsCost + jacketCost + totalShoeCost
def remainingMoney : ℕ := initialMoney - totalSpent - jeansDiscountedPrice

-- Proof problem statement
theorem paula_remaining_money : remainingMoney = 57 := by
  sorry

end paula_remaining_money_l279_279512


namespace intersection_A_B_eq_12_l279_279386

-- Conditions
def setA : Set ℝ := { y | y > 0 }
def setB : Set ℕ := { x | abs (2 * x - 3) <= 1 }

-- Theorem (the problem's question and answer formatted as a Lean theorem)
theorem intersection_A_B_eq_12 : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_A_B_eq_12_l279_279386
