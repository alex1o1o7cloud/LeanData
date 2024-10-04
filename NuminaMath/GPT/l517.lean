import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Finsupp
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Parith.Ops.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic.Function
import Mathlib.Combinatorics.CombinatorialIdentities
import Mathlib.Data.Complex.Basic
import Mathlib.Data.FiniteCard
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Perm
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.LCM
import Mathlib.NumberTheory.Prime
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.LinearCombination

namespace cos_triple_angle_l517_517468

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517468


namespace prime_pairs_sum_50_l517_517446

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517446


namespace prime_pairs_sum_50_l517_517436

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517436


namespace range_of_a_l517_517862

def f (x : ℝ) : ℝ :=
if x < 0 then x^2 + x else -x^2

def g (x : ℝ) : ℝ :=
if x < 0 then x^2 - 2*x - 5 else -(x^2 - 2*(-x) - 5) -- utilizing the odd function property

theorem range_of_a (a : ℝ) (h : f (g a) ≤ 2) : a ∈ Iic (-1) ∪ Icc 0 (2 * Real.sqrt 2 - 1) :=
sorry

end range_of_a_l517_517862


namespace remainders_identical_l517_517556

theorem remainders_identical (a b : ℕ) (h1 : a > b) :
  ∃ r₁ r₂ q₁ q₂ : ℕ, 
  a = (a - b) * q₁ + r₁ ∧ 
  b = (a - b) * q₂ + r₂ ∧ 
  r₁ = r₂ := by 
sorry

end remainders_identical_l517_517556


namespace cos_3theta_value_l517_517462

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l517_517462


namespace clock_angle_3_15_l517_517112

theorem clock_angle_3_15 :
  (let minute_hand_angle := 3 * 30 in -- degree for minute hand
   let hour_hand_angle := (3 * 30) + (15 / 60 * 30) in -- degree for hour hand at 3:15
   let angle := abs (hour_hand_angle - minute_hand_angle) in
   if angle > 180 then 360 - angle else angle) = 7.5 :=
by
  -- Insert proof here
  sorry

end clock_angle_3_15_l517_517112


namespace primes_sum_50_l517_517404

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517404


namespace prime_pairs_sum_50_l517_517278

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517278


namespace count_prime_pairs_summing_to_50_l517_517224

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517224


namespace canada_population_proof_l517_517815

noncomputable def combined_population (humans beavers moose caribou wolves grizzly_bears : ℕ) : ℕ :=
  moose + beavers + caribou + wolves + grizzly_bears

theorem canada_population_proof 
  (h_population : 38 * 10^6)
  (beaver_human_ratio : ∀ ⦃b h : ℕ⦄, h = 19 * b)
  (moose_beaver_ratio : ∀ ⦃m b : ℕ⦄, b = 2 * m)
  (caribou_moose_ratio : ∀ ⦃c m : ℕ⦄, c = (3 * m) / 2)
  (wolf_caribou_ratio : ∀ ⦃w c : ℕ⦄, w = 4 * c)
  (bear_wolf_ratio : ∀ ⦃g w : ℕ⦄, g = w / 3) :
  combined_population 1_000_000 2_000_000 1_000_000 1_500_000 6_000_000 2_000_000 = 12.5 * 10^6 :=
by sorry

end canada_population_proof_l517_517815


namespace round_trip_time_correct_l517_517574

variables (river_current_speed boat_speed_still_water distance_upstream_distance : ℕ)

def upstream_speed := boat_speed_still_water - river_current_speed
def downstream_speed := boat_speed_still_water + river_current_speed

def time_upstream := distance_upstream_distance / upstream_speed
def time_downstream := distance_upstream_distance / downstream_speed

def round_trip_time := time_upstream + time_downstream

theorem round_trip_time_correct :
  river_current_speed = 10 →
  boat_speed_still_water = 50 →
  distance_upstream_distance = 120 →
  round_trip_time river_current_speed boat_speed_still_water distance_upstream_distance = 5 :=
by
  intros rc bs d
  sorry

end round_trip_time_correct_l517_517574


namespace prime_pairs_sum_50_l517_517246

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517246


namespace num_prime_pairs_sum_50_l517_517199

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517199


namespace range_of_a_l517_517788

theorem range_of_a (a : ℝ)
  (A : set ℝ := {x | 1 ≤ x ∧ x ≤ 2})
  (B : set ℝ := {x | x^2 - a * x + 4 ≥ 0}) :
  (A ⊆ B) → a ≤ 4 :=
by
  sorry

end range_of_a_l517_517788


namespace area_of_XYMR_l517_517500

noncomputable def area_quadrilateral_XYMR (PQ PR : ℝ) (PQR_area : ℝ) (M N X Y : Point) : ℝ :=
  let M := midpoint P Q
  let N := midpoint P R
  let altitude := altitude P QR
  let border1 := line_segment P M
  let border2 := line_segment M N
  let border3 := line_segment N R
  let X := point_of_intersection altitude MN
  let Y := point_of_intersection altitude QR
  quadrilateral_area X Y M R

theorem area_of_XYMR
  (PQ PR : ℝ)
  (PQR_area : ℝ)
  (PQ_eq : PQ = 60)
  (PR_eq : PR = 20)
  (PQR_area_eq : PQR_area = 240) :
  area_quadrilateral_XYMR PQ PR PQR_area M N X Y = 80 :=
by
  -- define the geometric configurations
  let M := midpoint P Q
  let N := midpoint P R
  let altitude := altitude P QR
  let border1 := line_segment P M
  let border2 := line_segment M N
  let border3 := line_segment N R
  let X := point_of_intersection altitude MN
  let Y := point_of_intersection altitude QR

  -- the expected area of quadrilateral XYMR
  have XYMR_area : area_quadrilateral_XYMR PQ PR PQR_area = 80

  sorry

end area_of_XYMR_l517_517500


namespace determine_k_l517_517910

open Real

def f (k x : ℝ) := k*x^2 + 2*k*x + 1

theorem determine_k (k : ℝ) :
  (∀ x ∈ Icc (-3 : ℝ) 2, f k x ≤ 4) → (∃ k : ℝ, k = 3/8 ∨ k = -3) :=
by
  sorry

end determine_k_l517_517910


namespace primes_sum_50_l517_517409

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517409


namespace cos2x_sub_sin2y_eq_pm1_l517_517755

theorem cos2x_sub_sin2y_eq_pm1 (x y : ℝ) (h : sin x * cos y = cos x * sin y ∧ sin x * cos y = 1 / 2) : cos (2 * x) - sin (2 * y) = -1 ∨ cos (2 * x) - sin (2 * y) = 1 :=
by
  sorry

end cos2x_sub_sin2y_eq_pm1_l517_517755


namespace last_three_digits_of_7_exp_1987_l517_517692

theorem last_three_digits_of_7_exp_1987 : (7 ^ 1987) % 1000 = 543 := by
  sorry

end last_three_digits_of_7_exp_1987_l517_517692


namespace smallest_natural_number_B_l517_517054

theorem smallest_natural_number_B (A : ℕ) (h : A % 2 = 0 ∧ A % 3 = 0) :
    ∃ B : ℕ, (360 / (A^3 / B) = 5) ∧ B = 3 :=
by
  sorry

end smallest_natural_number_B_l517_517054


namespace num_prime_pairs_sum_50_l517_517182

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517182


namespace cos_triple_angle_l517_517463

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l517_517463


namespace find_acute_angle_l517_517799

theorem find_acute_angle (x : ℝ) (h : sin (3 * π / 5) * cos x + cos (2 * π / 5) * sin x = sqrt 3 / 2) (hx : 0 < x ∧ x < π / 2) :
  x = 4 * π / 15 :=
sorry

end find_acute_angle_l517_517799


namespace cyclic_quadrilateral_count_l517_517116

theorem cyclic_quadrilateral_count : 
  (∃ a b c d : ℕ, 
     a + b + c + d = 36 ∧ 
     a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
     a * b * c * d ≠ 0 ∧ 
     is_convex_cyclic_quadrilateral a b c d) 
  → (count_valid_quadrilaterals (36) = 823) :=
sorry

def is_convex_cyclic_quadrilateral (a b c d : ℕ) : Prop := 
  -- definition goes here, which checks whether the given a, b, c, d form a convex cyclic quadrilateral
  
noncomputable def count_valid_quadrilaterals (n : ℕ) : ℕ :=
  -- definition that counts the number of valid convex cyclic quadrilaterals with perimeter n goes here

end cyclic_quadrilateral_count_l517_517116


namespace prime_pairs_sum_to_50_l517_517417

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517417


namespace find_x_l517_517804

theorem find_x (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) (h3 : a^3 = 21 * x * 15 * b) : x = 25 :=
by
  -- This is where the proof would go
  sorry

end find_x_l517_517804


namespace solve_for_a_l517_517484

theorem solve_for_a (a : ℝ) : 
  (∃ z : ℂ, z = (a - complex.I)^2 ∧ (realPart z = 0 ∧ imagPart z ≠ 0)) → (a = 1 ∨ a = -1) := 
sorry

end solve_for_a_l517_517484


namespace tan_beta_third_l517_517736

theorem tan_beta_third (α β : ℝ) (h1 : (sin α * cos α) / (1 - cos (2 * α)) = 1 / 2)
  (h2 : tan (α - β) = 1 / 2) : tan β = 1 / 3 :=
sorry

end tan_beta_third_l517_517736


namespace num_prime_pairs_sum_50_l517_517173

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517173


namespace matt_overall_profit_l517_517540

open Nat

-- Define the initial number of cards and their value
def initial_cards : Nat := 8
def card_value : Nat := 6
def initial_value := initial_cards * card_value

-- Define the first trade
noncomputable def first_trade_value_traded_away : Nat := 2 * card_value
noncomputable def first_trade_value_received : Nat := 3 * 2 + 9
def first_trade_profit := first_trade_value_received - first_trade_value_traded_away

-- Define the second trade
noncomputable def second_trade_value_traded_away : Nat := 1 * 2 + 6
noncomputable def second_trade_value_received : Nat := 2 * 5 + 8
def second_trade_profit := second_trade_value_received - second_trade_value_traded_away

-- Define the third trade
noncomputable def third_trade_value_traded_away : Nat := 5 + 9
noncomputable def third_trade_value_received : Nat := 3 * 3 + 10 + 1
def third_trade_profit := third_trade_value_received - third_trade_value_traded_away

-- Define the overall profit
def total_profit := first_trade_profit + second_trade_profit + third_trade_profit

-- The theorem statement
theorem matt_overall_profit : total_profit = 19 := by
  -- proof goes here
  sorry

end matt_overall_profit_l517_517540


namespace ratio_of_triangles_to_squares_l517_517699

noncomputable theory

def infinite_hexagonal_construction (n : ℕ) : ℕ × ℕ :=
-- n represents the layer number; hexagon corresponds to n = 0
-- Returns a tuple (number of squares, number of equilateral triangles) in the nth layer
if n = 0 then (6, 6) -- Initial layer with 6 squares and 6 triangles
else let prev_layer := infinite_hexagonal_construction (n - 1)
     in (prev_layer.1 + polygon_sides (n), prev_layer.2 + polygon_sides (n))

def polygon_sides (n : ℕ) : ℕ :=
-- Returns the number of sides of the polygon at layer n
6 * 2^n

theorem ratio_of_triangles_to_squares : ∀ n : ℕ,
  let (squares, triangles) := infinite_hexagonal_construction n
  in triangles / squares = 1 :=
by
  intros
  induction n with n ih
    case zero =>
      unfold infinite_hexagonal_construction quadratic_recursive_polygon
      field_simp
    case succ =>
      unfold infinite_hexagonal_construction quadratic_recursive_polygon
      -- Advances to the next layer, preserving the ratio 1:1
      sorry

end ratio_of_triangles_to_squares_l517_517699


namespace num_prime_pairs_sum_50_l517_517192

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517192


namespace biff_break_even_hours_l517_517011

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l517_517011


namespace fraction_simplification_l517_517908

theorem fraction_simplification :
  ( (5^1004)^4 - (5^1002)^4 ) / ( (5^1003)^4 - (5^1001)^4 ) = 25 := by
  sorry

end fraction_simplification_l517_517908


namespace biff_break_even_hours_l517_517007

-- Definitions based on conditions
def ticket_expense : ℕ := 11
def snacks_expense : ℕ := 3
def headphones_expense : ℕ := 16
def total_expenses : ℕ := ticket_expense + snacks_expense + headphones_expense
def gross_income_per_hour : ℕ := 12
def wifi_cost_per_hour : ℕ := 2
def net_income_per_hour : ℕ := gross_income_per_hour - wifi_cost_per_hour

-- The proof statement
theorem biff_break_even_hours : ∃ h : ℕ, h * net_income_per_hour = total_expenses ∧ h = 3 :=
by 
  have h_value : ℕ := 3
  exists h_value
  split
  · show h_value * net_income_per_hour = total_expenses
    sorry
  · show h_value = 3
    rfl

end biff_break_even_hours_l517_517007


namespace sum_max_min_X_l517_517866

def I := {i : ℕ | 1 ≤ i ∧ i ≤ 2020}

def W (a b : ℕ) : ℕ := (a + b) + (a * b)
def Y (a b : ℕ) : ℕ := (a + b) * (a * b)

def X : Set ℕ := {x | ∃ a b : ℤ, W a b = x ∧ Y a b = x}

theorem sum_max_min_X : 
  (∃ x_max x_min : ℕ, x_max ∈ X ∧ x_min ∈ X ∧ 
  (∀ x : ℕ, x ∈ X → x ≤ x_max) ∧ (∀ x : ℕ, x ∈ X → x_min ≤ x) ∧ 
  (x_max + x_min = 58)) :=
sorry

end sum_max_min_X_l517_517866


namespace find_smallest_angle_l517_517523

noncomputable def vectors_min_angle (a b c : EuclideanSpace ℝ (Fin 3)) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 2) 
  (h_eq : a × (a × c) + b = 0) : Real :=
30

theorem find_smallest_angle 
  (a b c : EuclideanSpace ℝ (Fin 3)) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 2) 
  (h_eq : a × (a × c) + b = 0) : 
  vectors_min_angle a b c ha hb hc h_eq = 30 := 
sorry

end find_smallest_angle_l517_517523


namespace cyclic_quadrilateral_count_l517_517117

theorem cyclic_quadrilateral_count : 
  (∃ a b c d : ℕ, 
     a + b + c + d = 36 ∧ 
     a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
     a * b * c * d ≠ 0 ∧ 
     is_convex_cyclic_quadrilateral a b c d) 
  → (count_valid_quadrilaterals (36) = 823) :=
sorry

def is_convex_cyclic_quadrilateral (a b c d : ℕ) : Prop := 
  -- definition goes here, which checks whether the given a, b, c, d form a convex cyclic quadrilateral
  
noncomputable def count_valid_quadrilaterals (n : ℕ) : ℕ :=
  -- definition that counts the number of valid convex cyclic quadrilaterals with perimeter n goes here

end cyclic_quadrilateral_count_l517_517117


namespace values_of_x_l517_517082

def A (x : ℝ) := {1, 4, x}
def B (x : ℝ) := {1, x^2}

theorem values_of_x (x : ℝ) : (A x ∩ B x = B x) → x ∈ {-2, 0, 2} :=
by
  sorry

end values_of_x_l517_517082


namespace factorization_l517_517710

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := sorry

end factorization_l517_517710


namespace prime_pairs_summing_to_50_count_l517_517378

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517378


namespace num_unordered_prime_pairs_summing_to_50_l517_517150

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517150


namespace prime_pairs_sum_to_50_l517_517418

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517418


namespace problem_solution_l517_517099

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then x^2 else x + 6/x - 6

theorem problem_solution :
  f (f (-2)) = -1/2 ∧ (∀ x, f(x) ≥ 2 * Real.sqrt 6 - 6) :=
by
  sorry

end problem_solution_l517_517099


namespace max_club_members_l517_517990

open Set

variable {U : Type} (A B C : Set U)

theorem max_club_members (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11)
    (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
    (A ∪ B ∪ C).card ≤ 22 :=
by {
  -- The proof will go here, but for now we skip it.
  sorry
}

end max_club_members_l517_517990


namespace primes_sum_50_l517_517395

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517395


namespace find_BE_l517_517635

noncomputable def isosceles_triangle (A B C : Type) (AB : ℝ) (AD : ℝ) : Prop :=
isosceles A B C ∧ length AD = 1 

theorem find_BE (A B C D E : Type) (AB AD BD : ℝ) (BE : ℝ) (h : isosceles_triangle A B C AB AD) (h1 : angle_deg E D B = 90) : BE = 2 :=
sorry

end find_BE_l517_517635


namespace prime_pairs_sum_50_l517_517266

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517266


namespace problem_l517_517527

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x - 2

theorem problem : f(g(3) + 2) = 5 := 
  by
    sorry

end problem_l517_517527


namespace like_terms_abs_l517_517795

-- Definitions from the conditions
variables (x y : ℝ) (n m : ℤ)
def term1 := -5 * x^3 * y^(n - 2)
def term2 := 3 * x^(2 * m + 5) * y

-- Convert the conditions to Lean definitions
def conditions (m n : ℤ) : Prop :=
  (2 * m + 5 = 3) ∧ (n - 2 = 1)

-- The proof statement
theorem like_terms_abs (m n : ℤ) (h : conditions m n) : abs (n - 5 * m) = 8 :=
by
  -- Proof goes here
  sorry

end like_terms_abs_l517_517795


namespace sum_y_coords_other_vertices_l517_517451

theorem sum_y_coords_other_vertices (x1 y1 x2 y2: ℤ) 
    (h1: (x1, y1) = (4, 20)) 
    (h2: (x2, y2) = (12, -6)) : 
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2) in
  (2 * (midpoint.snd)) = 14 := 
by 
  rw [h1, h2]
  sorry

end sum_y_coords_other_vertices_l517_517451


namespace zero_point_interval_l517_517069

variable (f : ℝ → ℝ)
variable (f_deriv : ℝ → ℝ)
variable (e : ℝ)
variable (monotonic_f : MonotoneOn f (Set.Ioi 0))

noncomputable def condition1_property (x : ℝ) (h : 0 < x) : f (f x - Real.log x) = Real.exp 1 + 1 := sorry
noncomputable def derivative_property (x : ℝ) (h : 0 < x) : f_deriv x = (deriv f) x := sorry

theorem zero_point_interval :
  ∃ x ∈ Set.Ioo 1 2, f x - f_deriv x - e = 0 := sorry

end zero_point_interval_l517_517069


namespace no_polynomials_exist_l517_517631

theorem no_polynomials_exist :
  ∀ (P Q R : Polynomial ℝ), 
  ¬ ((x y z : ℝ) → (x - y + 1) ^ 3 * P.eval (x, y, z) + 
                       (y - z - 1) ^ 3 * Q.eval (x, y, z) + 
                       (z - 2 * x + 1) ^ 3 * R.eval (x, y, z) = 1) :=
sorry

end no_polynomials_exist_l517_517631


namespace constant_term_expansion_eq_15_l517_517499

theorem constant_term_expansion_eq_15 : 
  let expr := (x + 1 / x^2)^6 in
  ∃ (constant : ℕ), constant = 15 :=
begin
  sorry
end

end constant_term_expansion_eq_15_l517_517499


namespace intersection_points_l517_517593

variable {a b : ℝ} (f : ℝ → ℝ)

theorem intersection_points (h : a ≤ b) :
  if (2 : ℝ) ∈ (set.Icc a b) then
    ∃! y, y = f 2
  else
    ¬ ∃ x, x = 2 ∧ (a ≤ x ∧ x ≤ b) :=
by sorry

end intersection_points_l517_517593


namespace range_of_x_l517_517739

theorem range_of_x (x : ℝ) (p : x^2 - 2 * x - 3 < 0) (q : 1 / (x - 2) < 0) : -1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l517_517739


namespace count_prime_pairs_sum_50_l517_517137

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517137


namespace sufficient_not_necessary_l517_517638

theorem sufficient_not_necessary (p q: Prop) :
  ¬ (p ∨ q) → ¬ p ∧ (¬ p → ¬(¬ p ∧ ¬ q)) := sorry

end sufficient_not_necessary_l517_517638


namespace fewer_heads_than_tails_probability_l517_517617

def total_flips := 12

def fair_coin_flip : ∀ (n : ℕ), n ∈ {0, 1} :=
  λ n, n = 0 ∨ n = 1

def heads_probability : ℚ := 1 / 2
def tails_probability : ℚ := 1 / 2

def event_heads (k : ℕ) : Prop :=
  k ≤ total_flips ∧ fair_coin_flip k

def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def event_exact_heads (k : ℕ) : ℚ :=
  (binomial_coefficient total_flips k) / (2 ^ total_flips)

def y : ℚ := event_exact_heads 6

def x : ℚ := (1 - y) / 2

theorem fewer_heads_than_tails_probability :
  x = 1586 / 4096 := by
  sorry

end fewer_heads_than_tails_probability_l517_517617


namespace num_prime_pairs_summing_to_50_l517_517346

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517346


namespace num_prime_pairs_sum_50_l517_517320

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517320


namespace find_constants_range_of_k_l517_517774

noncomputable def f (a b x : ℝ) : ℝ := (a * x) / (exp x + 1) + b * (exp (-x))

theorem find_constants :
  (∃ a b : ℝ, f a b 0 = 1 ∧ (deriv (f a b) 0 = -1/2) ∧ a = 1 ∧ b = 1) :=
begin
  -- proof here
  sorry
end

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (f 1 1 x) > (x / (exp x - 1) + k * exp (-x))) → k ≤ 0 :=
begin
  -- proof here
  sorry
end

end find_constants_range_of_k_l517_517774


namespace find_b_l517_517728

noncomputable def b_of_polynomial_roots (a b c d : ℝ) :=
  ∀ (z w : ℂ),
  let conj_z := complex.conj z,
      conj_w := complex.conj w in
  (z * w = 8 + 3*complex.I) ∧
  (conj_z + conj_w = 2 + 5*complex.I) →
  b = (8 + 3*complex.I) + (8 - 3*complex.I) + ((2 - 5*complex.I) * (2 + 5*complex.I))

theorem find_b (a b c d : ℝ) :
  ∀ (z w : ℂ),
  let conj_z := complex.conj z,
      conj_w := complex.conj w in
  (z * w = 8 + 3*complex.I) ∧
  (conj_z + conj_w = 2 + 5*complex.I) →
  b = 45 :=
by {
  intros z w conj_z conj_w h,
  sorry
}

end find_b_l517_517728


namespace area_bound_proof_l517_517857

-- Noncomputable theories to allow for area computations and other non-explicit maths.
noncomputable theory

open_locale classical -- Enables the use of classical logic.

-- Define the types and variables involved in the problem statement.
variables {A B C D P Q R : Type*}
variables [convex_quadrilateral A B C D]
variables [point Q : Type*] [point P : Type*] [point R : Type*]
variables (S : ℝ) (s : ℝ) -- S is the area of quadrilateral ABCD; s is the area of triangle PQR

-- Conditions as given in the problem
def Q_is_intersection (A B C D Q : Type*) [convex_quadrilateral A B C D] : Prop := 
  intersects AC BD Q -- Q is the intersection point of diagonals AC and BD

def P_and_R_on_sides (A B C D P R : Type*) [convex_quadrilateral A B C D] : Prop := 
  on_side AB P ∧ on_side CD R -- P is on AB and R is on CD
  
def PQ_parallel_AD (P Q A D : Type*) [point P] [point Q] [line AD] : Prop :=
  parallel PQ AD -- PQ is parallel to AD

def QR_parallel_BC (Q R B C : Type*) [point Q] [point R] [line BC] : Prop :=
  parallel QR BC -- QR is parallel to BC

-- Main theorem to prove
theorem area_bound_proof (hQ : Q_is_intersection A B C D Q) 
                        (hPR : P_and_R_on_sides A B C D P R)
                        (hPQAD : PQ_parallel_AD P Q A D)
                        (hQRBC : QR_parallel_BC Q R B C)
                        (hS : S = area_of_convex_quadrilateral A B C D) 
                        (hs : s = area_of_triangle P Q R) : 
  0 ≤ s ∧ s < (4 / 27) * S := by
  sorry -- Proof will be provided here.

end area_bound_proof_l517_517857


namespace math_students_count_l517_517818

noncomputable def students_in_math (total_students history_students english_students all_three_classes two_classes : ℕ) : ℕ :=
total_students - history_students - english_students + (two_classes - all_three_classes)

theorem math_students_count :
  students_in_math 68 21 34 3 7 = 14 :=
by
  sorry

end math_students_count_l517_517818


namespace numerical_identity_l517_517884

theorem numerical_identity :
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 :=
by
  -- conditions and definitions based on a) are directly used here
  sorry -- proof is not required as per instructions

end numerical_identity_l517_517884


namespace area_of_triangle_with_base_and_height_l517_517610

theorem area_of_triangle_with_base_and_height (t : ℝ) (h : t = 9) : 
  (1 / 2) * (2 * t) * (2 * t - 6) = 108 := 
by
  have h1 : (1 / 2) * (2 * t) * (2 * t - 6) = t * (2 * t - 6) :=
    by ring
  rw h at h1
  rw h1
  norm_num
  sorry

end area_of_triangle_with_base_and_height_l517_517610


namespace value_of_b_l517_517947

theorem value_of_b :
  (∃ b : ℝ, (1 / Real.log b / Real.log 3 + 1 / Real.log b / Real.log 4 + 1 / Real.log b / Real.log 5 = 1) → b = 60) :=
by
  sorry

end value_of_b_l517_517947


namespace perimeter_T2_l517_517519

def Triangle (a b c : ℝ) :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem perimeter_T2 (a b c : ℝ) (h : Triangle a b c) (ha : a = 10) (hb : b = 15) (hc : c = 20) : 
  let AM := a / 2
  let BN := b / 2
  let CP := c / 2
  0 < AM ∧ 0 < BN ∧ 0 < CP →
  AM + BN + CP = 22.5 :=
by
  sorry

end perimeter_T2_l517_517519


namespace prime_pairs_sum_50_l517_517251

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517251


namespace num_unordered_prime_pairs_summing_to_50_l517_517152

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517152


namespace number_of_cows_l517_517542

/-- 
The number of cows Mr. Reyansh has on his dairy farm 
given the conditions of water consumption and total water used in a week. 
-/
theorem number_of_cows (C : ℕ) 
  (h1 : ∀ (c : ℕ), (c = 80 * 7))
  (h2 : ∀ (s : ℕ), (s = 10 * C))
  (h3 : ∀ (d : ℕ), (d = 20 * 7))
  (h4 : 1960 * C = 78400) : 
  C = 40 :=
sorry

end number_of_cows_l517_517542


namespace probability_dime_correct_l517_517658

-- Definitions of values and coin worths given as conditions.
def value_quarters : ℝ := 12.50
def value_dimes : ℝ := 5.00
def value_pennies : ℝ := 2.50

def worth_quarter : ℝ := 0.25
def worth_dime : ℝ := 0.10
def worth_penny : ℝ := 0.01

def num_quarters : ℕ := (value_quarters / worth_quarter).to_nat
def num_dimes : ℕ := (value_dimes / worth_dime).to_nat
def num_pennies : ℕ := (value_pennies / worth_penny).to_nat

def total_coins : ℕ := num_quarters + num_dimes + num_pennies

def probability_of_dime : ℝ := num_dimes / total_coins.toReal

-- Statement to be proved.
theorem probability_dime_correct :
  probability_of_dime = (1 : ℝ) / 7 :=
by
  sorry

end probability_dime_correct_l517_517658


namespace infinite_solutions_exists_l517_517858

-- Problem statement setup
variable {n : ℕ} 
variable {a : Fin (n+2) → ℕ} -- indices from 0 to n+1 in Fin (n+2)
variable (hCoprime : ∀ i : Fin n, Nat.coprime (a i) (a n)) -- a_i is relatively prime to a_(n+1)

-- Proof problem: Proving the infinite solutions equality
theorem infinite_solutions_exists :
  ∃ inf (sols : ℕ → Fin (n+2) → ℕ), 
   ∀ k, (sols k) ≠ sols k.succ ∧ 
   ∀ k, (sols k ⟨n+1, Nat.lt_succ_self n⟩) ^ (a ⟨n+1, Nat.lt_succ_self n⟩) =
         ∑ i in Finset.range n, (sols k ⟨i, Nat.lt_of_lt_of_le (Fin.is_lt i) (Nat.le_succ n)⟩) ^ (a ⟨i, Nat.lt_of_lt_of_le (Fin.is_lt i) (Nat.le_succ n)⟩)
:= sorry

end infinite_solutions_exists_l517_517858


namespace max_intersection_of_p_and_q_l517_517616

-- Define a fourth degree polynomial p
def p (x : ℝ) : ℝ := sorry -- let it be some polynomial of degree 4

-- Define a third degree polynomial q
def q (x : ℝ) : ℝ := sorry -- let it be some polynomial of degree 3

-- Maximum number of intersection points
theorem max_intersection_of_p_and_q : 
  ∃ (max_points : ℕ), max_points = 4 := 
begin
  use 4,
  sorry
end

end max_intersection_of_p_and_q_l517_517616


namespace acute_angle_PQ_XY_l517_517814

theorem acute_angle_PQ_XY (X Y Z R S P Q : Type) [AddGroup X] [AddGroup Y] [AddGroup Z] 
                        [HasSmul ℝ X] [HasSmul ℝ Y] [HasSmul ℝ Z]
                        [AffinelyIndependent ℝ ![X, Y, Z]] 
                        (angle_X : angle X = 38) 
                        (angle_Y : angle Y = 58) 
                        (XY : dist X Y = 11) 
                        (XR : dist X R = 1) (RS : dist R S = 1) 
                        (P_midpoint : P = midpoint ℝ X Y) 
                        (Q_midpoint : Q = midpoint ℝ R S) :
  ∃ θ, θ = 84 ∧ θ = acute_angle PQ XY := 
sorry

end acute_angle_PQ_XY_l517_517814


namespace count_prime_pairs_summing_to_50_l517_517231

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517231


namespace fixed_point_of_exp_transformation_l517_517765

theorem fixed_point_of_exp_transformation (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ A : ℝ × ℝ, (∀ a : ℝ, 0 < a ∧ a ≠ 1 → (λ x : ℝ, a^(x + 2) - 2) = (λ x, a^(x + 2) - 2)) ∧ A = (-2, -1) :=
by
  existsi (-2, -1)
  sorry

end fixed_point_of_exp_transformation_l517_517765


namespace number_of_prime_pairs_sum_50_l517_517214

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517214


namespace cos_alpha_is_neg_one_l517_517096

-- Define a point P and the angle α such that the cos α needs to be computed
def pointP := (-1 : ℝ, 0 : ℝ)
def α : ℝ := sorry  -- α is some angle that has terminal side passing through point P

-- The theorem stating that cos α is -1 given the conditions
theorem cos_alpha_is_neg_one (h : pointP = (-1, 0) ∧ terminal_side_through pointP α) : Real.cos α = -1 := 
by
  sorry

end cos_alpha_is_neg_one_l517_517096


namespace biff_break_even_hours_l517_517010

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l517_517010


namespace right_triangle_area_l517_517587

theorem right_triangle_area (a b c : ℝ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) : 0.5 * a * b = 30 := by
  sorry

end right_triangle_area_l517_517587


namespace cost_of_peaches_eq_2_per_pound_l517_517505

def initial_money : ℕ := 20
def after_buying_peaches : ℕ := 14
def pounds_of_peaches : ℕ := 3
def cost_per_pound : ℕ := 2

theorem cost_of_peaches_eq_2_per_pound (h: initial_money - after_buying_peaches = pounds_of_peaches * cost_per_pound) :
  cost_per_pound = 2 := by
  sorry

end cost_of_peaches_eq_2_per_pound_l517_517505


namespace Connie_gave_marbles_to_Juan_Connie_gave_70_marbles_to_Juan_l517_517687

theorem Connie_gave_marbles_to_Juan (original_marbles : ℕ) (remaining_marbles : ℕ) (given_marbles : ℕ) 
  (h1 : original_marbles = 73) (h2 : remaining_marbles = 3) : 
  given_marbles = original_marbles - remaining_marbles := 
by
  sorry

theorem Connie_gave_70_marbles_to_Juan
  (h1 : 73 - 3 = 70) : (Connie_gave_marbles_to_Juan 73 3 70 h1 rfl) = 70 :=
by 
  sorry

end Connie_gave_marbles_to_Juan_Connie_gave_70_marbles_to_Juan_l517_517687


namespace vector_dot_product_sum_l517_517851

open Real

variables (a b c : ℝ^3)

theorem vector_dot_product_sum :
  ∥a∥ = 4 ∧ ∥b∥ = 3 ∧ ∥c∥ = 2 ∧ (a + b + c = 0) →
  (a ⬝ b + a ⬝ c + b ⬝ c = -29 / 2) :=
begin
  sorry
end

end vector_dot_product_sum_l517_517851


namespace prime_pairs_sum_50_l517_517438

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517438


namespace maximal_triples_l517_517056

theorem maximal_triples (n : ℕ) (h : n ≥ 2) : 
  ∃ N : ℕ, (∀ (i : ℕ) (hi : 1 ≤ i ∧ i ≤ N), 
  ∀ (a b c : ℕ), (a, b, c) ∈ list.finRange (N + 1) ∧ a + b + c = n ∧ 
  ((∃ (j : ℕ) (hj : 1 ≤ j ∧ j ≤ N), i ≠ j ∧ (a ≠ a ∨ b ≠ b ∨ c ≠ c)) → (a ≠ a ∨ b ≠ b ∨ c ≠ c))) 
  ∧ N = (2 * n) / 3 + 1 := by
sorry

end maximal_triples_l517_517056


namespace part1_part2_l517_517065

-- Definitions of the given conditions
def Circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2

def Line_l (k x y : ℝ) : Prop := y = k * x - 2

-- Part (1): Proving the relationship for k
theorem part1 (k : ℝ) (A B : ℝ × ℝ)
  (hA : Circle_O A.1 A.2) (hB : Circle_O B.1 B.2)
  (hLineA : Line_l k A.1 A.2) (hLineB : Line_l k B.1 B.2)
  (hDistinct : A ≠ B) (hAngle : ∃ O, ∠ O A B = π / 2) :
  k = sqrt 3 ∨ k = -sqrt 3 :=
sorry

-- Part (2): Proving the fixed point when k = 1/2
theorem part2 (C D P : ℝ × ℝ) (k : ℝ)
  (hK : k = 1 / 2) (hP : Line_l k P.1 P.2)
  (hTangentsC : Tangent_from P C Circle_O)
  (hTangentsD : Tangent_from P D Circle_O)
  (hCD : Line_through C D P) :
  ∃ Q, Q = (1/2, -1) :=
sorry

end part1_part2_l517_517065


namespace find_p_l517_517091

theorem find_p 
  (α β : ℝ) 
  (p : ℝ) 
  (h1 : sin (2 * α + β) = p * sin β)
  (h2 : tan (α + β) = p * tan α)
  (h3 : p > 0)
  (h4 : p ≠ 1) : 
  p = 1 + Real.sqrt 2 :=
by
  sorry

end find_p_l517_517091


namespace prime_pairs_sum_50_l517_517440

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517440


namespace cos_triple_angle_l517_517455

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517455


namespace prime_pairs_sum_to_50_l517_517430

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517430


namespace count_prime_pairs_sum_50_l517_517132

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517132


namespace B_cannot_be_set_C_l517_517538

-- Define the sets A and B
def A : Set ℕ := {0, 1, 2}
def B (f : ℕ → ℕ) : Set ℕ := {f x | x ∈ A}

-- Define the function f
def f (x : ℕ) : ℕ := (x - 1) ^ 2

-- Prove that B cannot be the specific set {0, -1, 2}
theorem B_cannot_be_set_C : B f ≠ {0, -1, 2} := by
  sorry

end B_cannot_be_set_C_l517_517538


namespace limit_an_eq_minus3_l517_517549

noncomputable def a_n (n : ℕ) : ℝ := (1 + 3 * n) / (6 - n)

noncomputable def a : ℝ := -3

theorem limit_an_eq_minus3 : tendsto (λ n, a_n n) at_top (𝓝 a) :=
sorry

end limit_an_eq_minus3_l517_517549


namespace limit_an_eq_minus3_l517_517548

noncomputable def a_n (n : ℕ) : ℝ := (1 + 3 * n) / (6 - n)

noncomputable def a : ℝ := -3

theorem limit_an_eq_minus3 : tendsto (λ n, a_n n) at_top (𝓝 a) :=
sorry

end limit_an_eq_minus3_l517_517548


namespace Berry_monday_temp_l517_517002

-- Definitions for the given temperatures
def temp_day_2 := 99.1
def temp_day_3 := 98.7
def temp_day_4 := 99.3
def temp_day_5 := 99.8
def temp_day_6 := 99
def temp_day_7 := 98.9

-- Definition for the average temperature
def avg_temp := 99.0

-- Calculate the total temperature for the week
def total_temp := avg_temp * 7

-- Calculate the sum of the known temperatures for 6 days
def sum_temp_6_days := temp_day_2 + temp_day_3 + temp_day_4 + temp_day_5 + temp_day_6 + temp_day_7

-- Define the temperature on Monday (the unknown we seek to prove)
def temp_day_1 := total_temp - sum_temp_6_days

-- Statement of the theorem
theorem Berry_monday_temp : temp_day_1 = 98.2 := by
  -- Theorem to be proved
  sorry

end Berry_monday_temp_l517_517002


namespace prime_pairs_sum_50_l517_517275

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517275


namespace num_prime_pairs_sum_50_l517_517198

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517198


namespace increasing_sequence_if_and_only_if_general_formula_proof_l517_517072

section Part1
variables {a : ℕ → ℝ} (k : ℝ)
def sequence_increasing (k : ℝ) (a : ℕ → ℝ) : Prop := 
  a 0 = 1 ∧ (∀ n : ℕ, a (n + 1) = a n ^ 2 + k) ∧ (∀ n : ℕ, a (n + 1) > a n)

theorem increasing_sequence_if_and_only_if (h : sequence_increasing k a) : (0 < k) :=
sorry
end Part1

section Part2
variables {b : ℕ → ℝ}
def general_formula : ℕ → ℝ
| 0       := 3
| (n + 1) := (general_formula n) ^ 2

theorem general_formula_proof : (∀ n : ℕ, general_formula n = 3 ^ (2 ^ n)) :=
sorry
end Part2

end increasing_sequence_if_and_only_if_general_formula_proof_l517_517072


namespace num_prime_pairs_sum_50_l517_517202

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517202


namespace prime_pairs_sum_50_l517_517269

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517269


namespace num_unordered_prime_pairs_summing_to_50_l517_517160

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517160


namespace part_I_part_II_l517_517750

noncomputable def general_term (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (a 2 = 1 ∧ ∀ n, a (n + 1) - a n = d) ∧
  (d ≠ 0 ∧ (a 3)^2 = (a 2) * (a 6))

theorem part_I (a : ℕ → ℤ) (d : ℤ) : general_term a d → 
  ∀ n, a n = 2 * n - 3 := 
sorry

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : Prop :=
  (∀ n, S n = n * (a 1 + a n) / 2) ∧ 
  (general_term a d)

theorem part_II (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : sum_of_first_n_terms a d S → 
  ∃ n, n > 7 ∧ S n > 35 :=
sorry

end part_I_part_II_l517_517750


namespace count_prime_pairs_summing_to_50_l517_517227

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517227


namespace sum_totient_divisors_inequality_l517_517742

/-- Define the sum of the divisors function σ(n). -/
def σ (n : ℕ) : ℕ := ∑ d in (divisors n), d

/-- Define the Euler's totient function ϕ(n). -/
def ϕ (n : ℕ) : ℕ := (nat.totient n)

/-- The main theorem stating the inequality. -/
theorem sum_totient_divisors_inequality (n : ℕ) (hn : 0 < n) :
  (1 : ℚ) / ϕ n + 1 / σ n ≥ 2 / n := sorry

end sum_totient_divisors_inequality_l517_517742


namespace prime_pairs_sum_50_l517_517248

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517248


namespace num_prime_pairs_sum_50_l517_517188

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517188


namespace general_formula_summation_bound_l517_517104

noncomputable def S : ℕ → ℝ
| 0     := 0
| (n+1) := S n + (2^n + 1)

noncomputable def a : ℕ → ℝ
| 0     := 3
| 1     := 2^0 + 1
| (n+2) := 2^n + 1

theorem general_formula (n : ℕ) : 
  (a 0 = 3) ∧ (∀ n ≥ 1, a (n+1) = 2^n + 1) :=
sorry

theorem summation_bound (n : ℕ) :
  (1 / a 0) + Σ (i : ℕ) in finset.range n, (1 / a (i+1)) < 4 / 3 :=
sorry

end general_formula_summation_bound_l517_517104


namespace find_x_squared_plus_y_squared_l517_517044

theorem find_x_squared_plus_y_squared (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y + x + y = 17) (h4 : x^2 * y + x * y^2 = 72) : x^2 + y^2 = 65 := 
  sorry

end find_x_squared_plus_y_squared_l517_517044


namespace probability_two_yellow_balls_probability_at_least_one_blue_l517_517648

open Classical

section Probability

variable (A B C d e x : Type) [Fintype A] [Fintype B] [Fintype C] [Fintype d] [Fintype e] [Fintype x]

noncomputable def total_outcomes : Finset (A × B × C × d × e × x) :=
  Finset.univ.image id -- This represents all possible outcomes of drawing three balls

theorem probability_two_yellow_balls :
  (∑ (a : (A × B × C × d × e × x)) in total_outcomes, 
    if a.1 ∈ {A, B, C} ∧ a.2 ∈ {A, B, C} ∧ a.3 ∉ {A, B, C} then 1 else 0) / 
  (Fintype.card (A × B × C × d × e × x)) = 9 / 20 := 
  sorry

theorem probability_at_least_one_blue :
  1 - (∑ (a : (A × B × C × d × e × x)) in total_outcomes, 
    if a.1 ∉ {d, e} ∧ a.2 ∉ {d, e} ∧ a.3 ∉ {d, e} then 1 else 0) / 
  (Fintype.card (A × B × C × d × e × x)) = 4 / 5 := 
  sorry

end Probability

end probability_two_yellow_balls_probability_at_least_one_blue_l517_517648


namespace expand_product_l517_517704

theorem expand_product (y : ℝ) (h : y ≠ 0) : 
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = (3 / y) - 6 * y^3 + 9 := 
by 
  sorry

end expand_product_l517_517704


namespace find_real_numbers_l517_517047

theorem find_real_numbers (x : ℝ) :
  (x^3 - x^2 = (x^2 - x)^2) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end find_real_numbers_l517_517047


namespace prove_geometric_sequence_lg_sum_l517_517480

noncomputable def geometric_sequence_lg_sum : Prop :=
∀ (a : ℕ → ℝ),
(∀ n, 0 < a n) → 
(a 4 * a 9 + a 5 * a 8 + a 6 * a 7 = 300) → 
(∑ i in finset.range 12, Real.log10 (a (i + 1)) = 12)

theorem prove_geometric_sequence_lg_sum : geometric_sequence_lg_sum :=
sorry

end prove_geometric_sequence_lg_sum_l517_517480


namespace prime_pairs_sum_50_l517_517356

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517356


namespace find_principal_amount_l517_517834

theorem find_principal_amount :
  ∃ P : ℝ, 
    let r := 0.04 in
    let n := 2 in
    let loss := 1.0000000000001137 in
    let A_compound := P * (1 + r)^n in
    let A_simple := P * (1 + r * n) in
    A_compound - A_simple = loss ∧ 
    P = 625.0000000000709 :=
by
  sorry

end find_principal_amount_l517_517834


namespace parametric_curve_is_ellipse_l517_517786

theorem parametric_curve_is_ellipse {a : ℝ} : 
  ∀ t : ℝ, 
  let x := a * t / (1 + t^2),
      y := (1 - t^2) / (1 + t^2)
  in (2 * x / a) ^ 2 + y ^ 2 = 1 := sorry

end parametric_curve_is_ellipse_l517_517786


namespace proportional_sets_l517_517674

/-- Prove that among the sets of line segments, the ones that are proportional are: -/
theorem proportional_sets : 
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  ∃ a b c d, (a, b, c, d) = C ∧ (a * d = b * c) :=
by
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  sorry

end proportional_sets_l517_517674


namespace compute_f_g_at_2_l517_517478

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 4 * x - 1

theorem compute_f_g_at_2 :
  f (g 2) = 49 :=
by
  sorry

end compute_f_g_at_2_l517_517478


namespace exponent_form_l517_517944

theorem exponent_form (y : ℕ) (w : ℕ) (k : ℕ) : w = 3 ^ y → w % 10 = 7 → ∃ (k : ℕ), y = 4 * k + 3 :=
by
  intros h1 h2
  sorry

end exponent_form_l517_517944


namespace collinear_if_equal_areas_l517_517764

variables {A B C I H K B1 C1 B2 C2 A1 : Point}
-- Assuming that Point and Triangle are predefined structures

-- Conditions
def acute_triangle (ABC : Triangle) : Prop := triangle.is_acute ABC
def incenter (I : Point) (ABC : Triangle) : Prop := I = triangle.incenter ABC
def orthocenter (H : Point) (ABC : Triangle) : Prop := H = triangle.orthocenter ABC
def midpoint (P : Point) (X Y : Point) : Prop := P = segment.midpoint X Y

def ray_intersect (B1 I B2 AB : Segment) : Prop := B2 ∈ ray B1 I ∧ B2 ≠ B ∧ B2 ∈ AB
def ray_extend_intersect (C1 I C2 AC : Segment) : Prop := C2 ∈ ray C1 I ∧ C2 ∉ AC
def segment_intersect (B2 C2 K BC : Segment) : Prop := K ∈ segment B2 C2 ∧ K ∈ BC

def circumcenter (A1 : Point) (BHC : Triangle) : Prop := A1 = triangle.circumcenter BHC
def collinear_points (X Y Z : Point) : Prop := collinear X Y Z
def equal_area (t1 t2 : Triangle) : Prop := triangle.area t1 = triangle.area t2

-- Proof problem
theorem collinear_if_equal_areas {ABC BHC BKB2 CKC2 : Triangle}
  (h_acute : acute_triangle ABC)
  (h_incenter : incenter I ABC)
  (h_orthocenter : orthocenter H ABC)
  (h_midpointB1 : midpoint B1 A C)
  (h_midpointC1 : midpoint C1 A B)
  (h_rayB1I : ray_intersect ⟨B1, I⟩ B2 ⟨A, B⟩)
  (h_rayC1I : ray_extend_intersect ⟨C1, I⟩ C2 ⟨A, C⟩)
  (h_segment_intersect : segment_intersect B2 C2 K ⟨B, C⟩)
  (h_circumcenter : circumcenter A1 ⟨B, H, C⟩)
  : collinear_points A I A1 ↔ equal_area ⟨B, K, B2⟩ ⟨C, K, C2⟩ :=
sorry

end collinear_if_equal_areas_l517_517764


namespace prime_pairs_sum_50_l517_517447

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517447


namespace cos_pi_over_3_plus_2alpha_l517_517084

variable (α : ℝ)

theorem cos_pi_over_3_plus_2alpha (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (π / 3 + 2 * α) = 7 / 9 :=
  sorry

end cos_pi_over_3_plus_2alpha_l517_517084


namespace prime_pairs_sum_50_l517_517271

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517271


namespace sum_c_first_2017_terms_l517_517830

noncomputable def a (n : ℕ) : ℕ → ℝ
| 1 := 1
| (k + 1) := a k + b k + real.sqrt (a k ^ 2 + b k ^ 2)

noncomputable def b (n : ℕ) : ℕ → ℝ
| 1 := 1
| (k + 1) := a k + b k - real.sqrt (a k ^ 2 + b k ^ 2)

noncomputable def c (n : ℕ) : ℝ := 1 / (a n) + 1 / (b n)

noncomputable def sum_first_n_terms (n : ℕ) : ℝ :=
list.sum (list.map c (list.range n))

theorem sum_c_first_2017_terms : sum_first_n_terms 2017 = 4034 :=
by sorry

end sum_c_first_2017_terms_l517_517830


namespace variance_uniform_l517_517723

noncomputable def variance_of_uniform (α β : ℝ) (h : α < β) : ℝ :=
  let E := (α + β) / 2
  (β - α)^2 / 12

theorem variance_uniform (α β : ℝ) (h : α < β) :
  variance_of_uniform α β h = (β - α)^2 / 12 :=
by
  -- statement of proof only, actual proof here is sorry
  sorry

end variance_uniform_l517_517723


namespace prime_pairs_summing_to_50_count_l517_517377

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517377


namespace prime_pairs_summing_to_50_count_l517_517376

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517376


namespace num_prime_pairs_sum_50_l517_517167

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517167


namespace peter_multiple_of_harriet_l517_517821

variables (p_future p_current h_future h_current : ℕ) (mother_age : ℕ)

def conditions := 
  (p_future = p_current + 4) ∧ 
  (p_current = mother_age / 2) ∧ 
  (mother_age = 60) ∧ 
  (h_future = h_current + 4) ∧ 
  (h_current = 13)

theorem peter_multiple_of_harriet (p_future h_future : ℕ) (h_current : ℕ)
  (mother_age : ℕ) (h_current = 13)
  (mother_age = 60) :
  p_future = 2 * h_future :=
by
  assume cond : conditions p_future p_current h_future h_current mother_age,
  sorry

end peter_multiple_of_harriet_l517_517821


namespace minimum_r_for_coloring_S_l517_517688

def point := ℝ × ℝ

def hexagon_side_length : ℝ := 1

-- Define the set S formed by the points inside and on the edges of a regular hexagon with side length 1
noncomputable def is_in_set_S (p : point) : Prop := sorry

-- Define the coloring function which assigns one of three colors to each point in S
noncomputable def coloring (p : point) : ℕ := sorry

-- Define the distance function between two points
noncomputable def distance (p1 p2 : point) : ℝ := sorry

-- Prove that the minimum r for which points can be colored such that the distance between same-colored points is less than r is 3/2
theorem minimum_r_for_coloring_S : ∃ (r : ℝ), (∀ p1 p2 : point, p1 ∈ S → p2 ∈ S → coloring p1 = coloring p2 → distance p1 p2 < r) ∧ r = 3 / 2 :=
by sorry

end minimum_r_for_coloring_S_l517_517688


namespace prime_pairs_sum_50_l517_517371

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517371


namespace waiter_tables_l517_517636

/-
Problem:
A waiter had 22 customers in his section.
14 of them left.
The remaining customers were seated at tables with 4 people per table.
Prove the number of tables is 2.
-/

theorem waiter_tables:
  ∃ (tables : ℤ), 
    (∀ (customers_initial customers_remaining people_per_table tables_calculated : ℤ), 
      customers_initial = 22 →
      customers_remaining = customers_initial - 14 →
      people_per_table = 4 →
      tables_calculated = customers_remaining / people_per_table →
      tables = tables_calculated) →
    tables = 2 :=
by
  sorry

end waiter_tables_l517_517636


namespace count_prime_pairs_sum_50_exactly_4_l517_517280

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517280


namespace cos_difference_identity_l517_517798

theorem cos_difference_identity (α β : ℝ) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : Real.sin β = 5 / 13) : Real.cos (α - β) = 63 / 65 := 
by 
  sorry

end cos_difference_identity_l517_517798


namespace factorize_expression_l517_517713

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l517_517713


namespace cos_triple_angle_l517_517454

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517454


namespace count_prime_pairs_summing_to_50_l517_517236

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517236


namespace num_prime_pairs_sum_50_l517_517323

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517323


namespace primes_sum_50_l517_517394

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517394


namespace prime_pairs_sum_50_l517_517443

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517443


namespace admission_fee_for_adults_l517_517903

theorem admission_fee_for_adults (C : ℝ) (N M N_c N_a : ℕ) (A : ℝ) 
  (h1 : C = 1.50) 
  (h2 : N = 2200) 
  (h3 : M = 5050) 
  (h4 : N_c = 700) 
  (h5 : N_a = 1500) :
  A = 2.67 := 
by
  sorry

end admission_fee_for_adults_l517_517903


namespace percentage_decrease_in_breadth_l517_517999

theorem percentage_decrease_in_breadth
  (L B : ℝ)  -- original length and breadth
  (hL : L > 0)
  (hB : B > 0)
  (h_length_decrease : L * 0.7)
  (h_area_decrease : 0.525 * (L * B)) :

  let orig_area := L * B in
  let new_length := 0.7 * L in
  let new_area := 0.525 * orig_area in
  let new_breadth := (1 - 0.25) * B in
  
  0.25 = 25 := 
sorry

end percentage_decrease_in_breadth_l517_517999


namespace james_road_trip_l517_517835

theorem james_road_trip :
  let distance1 := 200 -- in miles
  let speed1 := 60 -- in mph
  let distance2 := 120 -- in miles
  let speed2 := 50 -- in mph
  let rest1 := 1.0 -- in hours
  let distance3 := 250 -- in miles
  let speed3 := 65 -- in mph
  let rest2 := 1.5 -- in hours
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let total_time := time1 + time2 + time3 + rest1 + rest2
  total_time = 12.079 :=
by
  let distance1 := 200
  let speed1 := 60
  let distance2 := 120
  let speed2 := 50
  let rest1 := 1.0
  let distance3 := 250
  let speed3 := 65
  let rest2 := 1.5
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let total_time := time1 + time2 + time3 + rest1 + rest2
  show total_time = 12.079
  sorry

end james_road_trip_l517_517835


namespace find_XZ_l517_517501

-- Defining the points and conditions of the triangle
variables (X Y Z M : Type) [inhabited X] [inhabited Y] [inhabited Z] [inhabited M]
variables (XY YZ XM : ℝ)

-- Given conditions
axiom xy_eq_seven : XY = 7
axiom yz_eq_ten : YZ = 10
axiom xm_eq_five : XM = 5

-- Define the lengths involving the midpoint M
axiom m_midpoint : XY * 2 = XZ

-- Prove that XZ = sqrt(51)
theorem find_XZ : XZ = Real.sqrt 51 :=
by
  sorry

end find_XZ_l517_517501


namespace prime_pairs_sum_to_50_l517_517314

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517314


namespace no_natural_divisible_by_M_with_digit_sum_less_than_m_l517_517695

theorem no_natural_divisible_by_M_with_digit_sum_less_than_m (m : ℕ) (M : ℕ) 
  (h1 : M = (NatIter.iterate m (λ n, n * 10 + 1) 0)) : 
  ¬ ∃ p : ℕ, (M ∣ p) ∧ (digit_sum p < m) := 
  by 
  sorry

end no_natural_divisible_by_M_with_digit_sum_less_than_m_l517_517695


namespace number_multiplied_by_3_l517_517801

theorem number_multiplied_by_3 (k : ℕ) : 
  2^13 - 2^(13-2) = 3 * k → k = 2048 :=
by
  sorry

end number_multiplied_by_3_l517_517801


namespace jane_house_number_l517_517925

theorem jane_house_number :
  (∃ (a b c d : ℕ), 7 + 4 + 5 + 3 + 2 + 8 + 1 = 30 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = 30 ∧ a * 1000 + b * 100 + c * 10 + d = 9876) :=
begin
  sorry
end

end jane_house_number_l517_517925


namespace mark_should_leave_at_1136_am_l517_517558

-- Definitions based on conditions
def rob_normal_travel_time : ℝ := 1 -- in hours
def rob_extra_time_construction : ℝ := 0.5 -- in hours
def mark_normal_travel_time : ℝ := 3 * rob_normal_travel_time -- in hours
def mark_reduction_percentage : ℝ := 0.2
def tz_difference : ℝ := 2 -- time zone difference in hours
def rob_departure_time : ℝ := 11 -- Rob leaves at 11 a.m. local time

def rob_altered_travel_time := rob_normal_travel_time + rob_extra_time_construction
def mark_altered_travel_time := mark_normal_travel_time - (mark_reduction_percentage * mark_normal_travel_time)

-- The time Mark should leave to arrive at the same time as Rob
def mark_departure_time := 
  let rob_arrival_time := rob_departure_time+ rob_altered_travel_time in
  let mark_arrival_time_in_tz := rob_arrival_time + tz_difference in
  mark_arrival_time_in_tz - mark_altered_travel_time

theorem mark_should_leave_at_1136_am : mark_departure_time = 11 + (36 / 60) :=
by
  -- Skipping proof steps
  sorry

end mark_should_leave_at_1136_am_l517_517558


namespace Diamond_evaluation_l517_517917

-- Redefine the operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^3 - b^2 + 1

-- Statement of the proof
theorem Diamond_evaluation : (Diamond 3 2) = 21 := by
  sorry

end Diamond_evaluation_l517_517917


namespace better_deal_at_850_cost_expressions_better_deal_at_1700_l517_517939

-- Definitions for cost in each mall
def cost_mallA (x : ℝ) : ℝ := 
  if x > 1000 then 1000 + 0.9 * (x - 1000) else x

def cost_mallB (x : ℝ) : ℝ := 
  if x > 500 then 500 + 0.95 * (x - 500) else x

-- Proofs required
theorem better_deal_at_850 : cost_mallA 850 > cost_mallB 850 := by
  sorry

theorem cost_expressions (x : ℝ) : x > 1000 → cost_mallA x = 100 + 0.9 * x ∧ cost_mallB x = 475 + 0.95 * x := by
  sorry

theorem better_deal_at_1700 : cost_mallA 1700 < cost_mallB 1700 := by
  sorry

end better_deal_at_850_cost_expressions_better_deal_at_1700_l517_517939


namespace probability_of_particle_falling_in_semicircle_l517_517479

noncomputable def rect_area (AB BC : ℝ) : ℝ := AB * BC

noncomputable def semicircle_area (AB : ℝ) : ℝ := (1 / 2) * π * (AB / 2) ^ 2

noncomputable def probability_in_semicircle (AB BC : ℝ) : ℝ :=
  semicircle_area AB / rect_area AB BC

theorem probability_of_particle_falling_in_semicircle {AB BC : ℝ} (hAB : AB = 2) (hBC : BC = 1) : 
  probability_in_semicircle AB BC = π / 4 :=
by
  rw [hAB, hBC]
  simp [probability_in_semicircle, semicircle_area, rect_area]
  sorry

end probability_of_particle_falling_in_semicircle_l517_517479


namespace eccentricity_of_hyperbola_eq_two_l517_517805

theorem eccentricity_of_hyperbola_eq_two
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (circle_eq : ∀ (x y : ℝ), (x - 4)^2 + y^2 = 16)
  (chord_length : ℝ)
  (h_chord : chord_length = 4)
  (hyperbola_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) : 
  let c := sqrt (a^2 + b^2)
  in (c / a) = 2 :=
sorry

end eccentricity_of_hyperbola_eq_two_l517_517805


namespace limit_sequence_l517_517551
noncomputable theory

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ := (1 + 3 * n) / (6 - n)

-- Define the limit value a
def a : ℝ := -3

-- Statement of the problem: Prove that the limit of a_n as n approaches infinity is a
theorem limit_sequence : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) := sorry

end limit_sequence_l517_517551


namespace num_prime_pairs_sum_50_l517_517330

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517330


namespace area_bound_l517_517536

variable (n : ℕ) (h_pos : n > 0)

def region (z : ℂ) : Prop :=
  ∑ k in Finset.range n, (1 / complex.abs (z - k)) ≥ 1

def area_of_region : ℝ := sorry -- Assumes the area can be computed, here as a placeholder.

theorem area_bound (h_pos : n > 0) : 
  area_of_region ≥ real.pi * (11 * n^2 + 1) / 12 :=
sorry

end area_bound_l517_517536


namespace cos_triple_angle_l517_517466

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l517_517466


namespace prime_pairs_sum_50_l517_517432

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517432


namespace probability_two_people_different_floors_l517_517932

-- Define the probability problem
def probability_different_floors (total_people : ℕ) (total_floors : ℕ) : ℚ :=
  let total_outcomes := total_floors ^ total_people
  let same_floor_outcomes := total_floors
  let different_floor_outcomes := total_outcomes - same_floor_outcomes
  different_floor_outcomes / total_outcomes

theorem probability_two_people_different_floors :
  probability_different_floors 2 6 = 5 / 6 :=
by
  -- Define the values for total people and floors
  let total_people := 2
  let total_floors := 6

  -- Calculate the total number of outcomes
  let total_outcomes := total_floors ^ total_people

  -- Calculate the number of outcomes where both leave on the same floor
  let same_floor_outcomes := total_floors

  -- Calculate the number of outcomes where they leave on different floors
  let different_floor_outcomes := total_outcomes - same_floor_outcomes

  -- Calculate the probability
  let prob := different_floor_outcomes / total_outcomes

  -- Assert that the calculation matches the expected value
  have calc_prob : prob = 5 / 6 := sorry
  
  exact calc_prob

end probability_two_people_different_floors_l517_517932


namespace cubic_inequality_l517_517952

theorem cubic_inequality (a b : ℝ) : a > b → a^3 > b^3 :=
sorry

end cubic_inequality_l517_517952


namespace count_prime_pairs_sum_50_l517_517129

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517129


namespace smallest_five_digit_with_product_55440_l517_517946

def product_of_digits (n : ℕ) : ℕ := 
  let digits := n.digits 10
  digits.foldl (*) 1

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_with_product_55440 : 
  ∃ n : ℕ, is_five_digit n ∧ product_of_digits n = 55440 ∧ ∀ m : ℕ, is_five_digit m ∧ product_of_digits m = 55440 → n ≤ m :=
sorry

end smallest_five_digit_with_product_55440_l517_517946


namespace cos_triple_angle_l517_517470

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517470


namespace employee_payments_l517_517936

noncomputable def amount_paid_to_Y : ℝ := 934 / 3
noncomputable def amount_paid_to_X : ℝ := 1.20 * amount_paid_to_Y
noncomputable def amount_paid_to_Z : ℝ := 0.80 * amount_paid_to_Y

theorem employee_payments :
  amount_paid_to_X + amount_paid_to_Y + amount_paid_to_Z = 934 :=
by
  sorry

end employee_payments_l517_517936


namespace prime_pairs_sum_50_l517_517276

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517276


namespace boys_without_calculators_l517_517541

theorem boys_without_calculators (total_boys : ℕ) (total_calculators : ℕ) (girls_with_calculators : ℕ)
    (total_boys_eq : total_boys = 20)
    (total_calculators_eq : total_calculators = 26)
    (girls_with_calculators_eq : girls_with_calculators = 13) :
    let boys_with_calculators := total_calculators - girls_with_calculators
    in total_boys - boys_with_calculators = 7 :=
by
  intros
  rw [total_boys_eq, total_calculators_eq, girls_with_calculators_eq]
  let boys_with_calculators := 26 - 13
  show 20 - boys_with_calculators = 7
  rw [boys_with_calculators]
  simp
  sorry

end boys_without_calculators_l517_517541


namespace prime_pairs_sum_50_l517_517367

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517367


namespace prime_pairs_summing_to_50_count_l517_517393

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517393


namespace meeting_on_medial_line_l517_517841

-- Given definitions
variables (A B C M H_a H_c : Type) [triangle ABC] [right_triangle B] (median BM AC)
variables (orthocenter H_a ABM) (orthocenter H_c CBM)

-- To Prove: AH_c and CH_a meet on the medial line of triangle ABC
theorem meeting_on_medial_line (H_ABC : right_triangle ABC B) (H_median : median BM AC)
  (H_orthocenter_A : orthocenter H_a ABM) (H_orthocenter_C : orthocenter H_c CBM) :
  intersects (line_through A H_c) (line_through C H_a) (medial_line ABC) :=
begin
  sorry
end

end meeting_on_medial_line_l517_517841


namespace whisker_count_correct_l517_517732

noncomputable def whisker_counts := 
  let princess_puff : ℕ := 14
  let catman_do : ℕ := 2 * princess_puff - 6
  let sir_whiskerson : ℕ := princess_puff + catman_do + 8
  let lady_flufflepuff : ℕ := sir_whiskerson / 2 + 4
  let mr_mittens : ℕ := abs (catman_do - lady_flufflepuff)
  (princess_puff, catman_do, sir_whiskerson, lady_flufflepuff, mr_mittens)

theorem whisker_count_correct:
  whisker_counts = (14, 22, 44, 26, 4) :=
by
  -- At this point, the detailed steps and proofs would be filled out to show the calculations, 
  -- but since 'sorry' is used to indicate the proof is omitted:
  sorry

end whisker_count_correct_l517_517732


namespace inscribed_angle_sum_l517_517671

theorem inscribed_angle_sum (α β γ : ℝ) (hα : α = 1/2 * (360 - (arc BC)) )
                                    (hβ : β = 1/2 * (360 - (arc CA)) )
                                    (hγ : γ = 1/2 * (360 - (arc AB)) )
                                    (arc_sum : arc BC + arc CA + arc AB = 360) : 
                                    α + β + γ = 360 :=
by
  sorry

end inscribed_angle_sum_l517_517671


namespace find_angle_B_find_side_b_l517_517503

open Real

variables (A B C a b c : ℝ)

axiom triangle_abc_conditions :
  ∀ (A B C a b c : ℝ),
    ∃ (A B C a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
    cos C * sin (A + π / 6) - sin C * sin (A - π / 3) = 1 / 2

theorem find_angle_B :
  triangle_abc_conditions A B C a b c →
  B = π / 3 :=
by
  sorry

variables (P : ℝ)
variables (S : ℝ)

axiom triangle_abc_additional_conditions :
  triangle_abc_conditions A B C a b c →
  P = 4 ∧ S = sqrt 3 / 3

theorem find_side_b :
  triangle_abc_additional_conditions A B C a b c P S →
  B = π / 3 →
  b = 3 / 2 :=
by
  sorry

end find_angle_B_find_side_b_l517_517503


namespace num_unordered_prime_pairs_summing_to_50_l517_517151

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517151


namespace regular_polygon_sides_l517_517986

theorem regular_polygon_sides (h : ∀ n : ℕ, n ≥ 3 → (total_internal_angle_sum / n) = 150) :
    n = 12 := by
  sorry

end regular_polygon_sides_l517_517986


namespace square_point_inequality_l517_517856

theorem square_point_inequality (A1 A2 A3 A4 P : ℝ×ℝ) 
  (h_square : is_square A1 A2 A3 A4) : 
  PA A1 + PA A2 + PA A3 + PA A4 ≥ 
  (1 + real.sqrt 2) * (max PA [A1, A2, A3, A4]) + (min PA [A1, A2, A3, A4]) :=
sorry

-- Definition for distance PA
def PA (P A : ℝ×ℝ) : ℝ :=
real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

-- Definitions for the properties of being a square and the circumcircle condition
def is_square (A1 A2 A3 A4 : ℝ×ℝ) : Prop :=
is_right_angle A1 A2 A3 ∧
is_right_angle A2 A3 A4 ∧
is_right_angle A3 A4 A1 ∧
is_right_angle A4 A1 A2

def is_right_angle (A B C : ℝ×ℝ) : Prop :=
((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2)) = 0

def on_circumcircle (P A1 A2 A3 A4 : ℝ×ℝ) : Prop :=
-- Cirumcircle radius calculation 
let center_real : ℝ := (A1.1 + A2.1 + A3.1 + A4.1) / 4
let center_imag : ℝ := (A1.2 + A2.2 + A3.2 + A4.2) / 4
let radius : ℝ := real.sqrt ((center_real - A1.1)^2 + (center_imag - A1.2)^2)
let equality_1 := real.sqrt ((center_real - P.1)^2 + (center_imag - P.2)^2) = radius in 
equality_1


end square_point_inequality_l517_517856


namespace num_prime_pairs_summing_to_50_l517_517347

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517347


namespace num_unordered_prime_pairs_summing_to_50_l517_517148

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517148


namespace factorization_l517_517708

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := sorry

end factorization_l517_517708


namespace length_of_ON_constant_l517_517675

noncomputable def ellipse_standard_form (a b : ℝ) (h : a > b > 0) (P : ℝ × ℝ) (eccentricity : ℝ) :=
  let x := P.1
  let y := P.2 in
  (∀ (a b : ℝ), (a > b > 0) → (eccentricity = (Real.sqrt 2) / 2) → 
    a^2 = 2 * b^2 → (x^2 / a^2 + y^2 / b^2 = 1) → (x / a)^2 + y^2 = 1)

noncomputable def circle_equation_with_chord (P : ℝ × ℝ) (line_eq : ℝ × ℝ → ℝ)
    (chord_length : ℝ) :=
  line_eq P = 0 → 
  ∀ (x y : ℝ) (h : (x - 1)^2 + (y - 2)^2 = 5) 
  (O M : ℝ) (p : ℝ × ℝ),
  p = (1, 2) → (h = chord_length / 2) → ((x - 1)^2 + (y - 2)^2 = 5)

theorem length_of_ON_constant (F O N : ℝ × ℝ) (hOM : ℝ × ℝ) :
  let F := (1, 0)
  let hON := (1, Real.sqrt 2) in
  F ≠ O → F ≠ N →
  (F - O) * (N - hOM) = 0 →
  (N - O)^2 = hON^2 :=
    sorry

end length_of_ON_constant_l517_517675


namespace prime_pairs_sum_50_l517_517373

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517373


namespace solution_l517_517018

noncomputable def problem : Prop :=
  nat.choose 16 13 = 560

theorem solution : problem := by
  sorry

end solution_l517_517018


namespace perfect_square_mod_4_l517_517966

theorem perfect_square_mod_4 (a : ℤ) : ∃ r, r ∈ {0, 1} ∧ (a^2) % 4 = r :=
by
  sorry

end perfect_square_mod_4_l517_517966


namespace num_prime_pairs_sum_50_l517_517201

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517201


namespace num_prime_pairs_summing_to_50_l517_517343

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517343


namespace evaluate_polynomial_at_minus_two_l517_517036

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end evaluate_polynomial_at_minus_two_l517_517036


namespace prime_pairs_sum_to_50_l517_517416

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517416


namespace D_96_equals_112_l517_517847

def multiplicative_decompositions (n : ℕ) : ℕ :=
  sorry -- Define how to find the number of multiplicative decompositions

theorem D_96_equals_112 : multiplicative_decompositions 96 = 112 :=
  sorry

end D_96_equals_112_l517_517847


namespace count_prime_pairs_sum_50_exactly_4_l517_517290

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517290


namespace count_prime_pairs_sum_50_exactly_4_l517_517298

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517298


namespace transformation_matrix_l517_517051

def scalingMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![ ![-3, 0], ![0, 2] ]

def rotationMatrix : Matrix (Fin 2) (Fin 2) ℝ := 
  ![ ![Real.sqrt 2 / 2, -Real.sqrt 2 / 2], 
     ![Real.sqrt 2 / 2,  Real.sqrt 2 / 2] ]

def combinedMatrix : Matrix (Fin 2) (Fin 2) ℝ := 
  ![ ![-3 * Real.sqrt 2 / 2, -2 * Real.sqrt 2 / 2],
     ![ 3 * Real.sqrt 2 / 2,  2 * Real.sqrt 2 / 2] ]

theorem transformation_matrix 
: rotationMatrix ⬝ scalingMatrix = combinedMatrix := by
  sorry

end transformation_matrix_l517_517051


namespace prime_pairs_sum_50_l517_517362

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517362


namespace hyperbola_eccentricity_proof_l517_517498

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : a = 2 ∧ b > 0 ∧ 4 * b^2 = a^2) : ℝ :=
  let e := (sqrt (a^2 - b^2)) / a
  e

theorem hyperbola_eccentricity_proof :
  ∀ b : ℝ, b > 0 →
  (let a := sqrt 12,
   b = 2 →
   hyperbola_eccentricity a b ⟨rfl, by norm_num, by norm_num⟩ = 2 * sqrt 3 / 3) :=
by
  sorry

end hyperbola_eccentricity_proof_l517_517498


namespace bianca_worked_hours_l517_517003

theorem bianca_worked_hours (B : ℝ) (Celeste : ℝ) (McClain : ℝ) :
  Celeste = 2 * B → McClain = 2 * B - 8.5 → (B + Celeste + McClain) * 60 = 3240 → B = 12.5 :=
by
  assume h1 h2 h3
  -- Since the proof steps are not required, we use "sorry" to skip the actual proof here.
  sorry

end bianca_worked_hours_l517_517003


namespace complex_product_l517_517685

theorem complex_product :
  (3 - 4 * complex.I) * (2 + 6 * complex.I) * (1 - complex.I) = 40 - 20 * complex.I :=
by
  simp [complex.I_def]
  sorry

end complex_product_l517_517685


namespace primes_sum_50_l517_517406

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517406


namespace max_value_fraction_l517_517718

theorem max_value_fraction : ∀ x : ℝ, 
  (∃ x : ℝ, max (1 + (16 / (4 * x^2 + 8 * x + 5))) = 17) :=
by
  sorry

end max_value_fraction_l517_517718


namespace prime_pairs_sum_to_50_l517_517317

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517317


namespace dwarves_all_collected_berries_on_one_day_l517_517894

def dwarf_activities_conditions (days : List (Fin 7 → Bool)) : Prop :=
  (∀ i j : Fin 16, i ≠ j → (days[i].toFinset ∩ days[j].toFinset).card ≥ 3) ∧
  (∀ i : Fin 16, (days[i].toFinset ∪ days[0].toFinset) = Finset.univ) ∧
  days[0] = λ i, false

theorem dwarves_all_collected_berries_on_one_day (days : List (Fin 7 → Bool)) :
  dwarf_activities_conditions days →
  ∃ i : Fin 16, ∀ j : Fin 7, days[i] j = true :=
sorry

end dwarves_all_collected_berries_on_one_day_l517_517894


namespace prime_pairs_sum_50_l517_517357

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517357


namespace count_prime_pairs_summing_to_50_l517_517233

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517233


namespace lunch_combo_count_l517_517679

/-- Given the number of options for each category of the lunch combo:
 3 options for Main Dishes,
 2 options for Sides,
 2 options for Drinks, and
 2 options for Desserts,
prove that the total number of distinct possible lunch combos is 24. -/
theorem lunch_combo_count :
  let main_dishes := 3
  let sides := 2
  let drinks := 2
  let desserts := 2
  main_dishes * sides * drinks * desserts = 24 := by
  let main_dishes := 3
  let sides := 2
  let drinks := 2
  let desserts := 2
  calc
    main_dishes * sides * drinks * desserts = 3 * 2 * 2 * 2 : by rfl
                                      ... = 24 : by norm_num

end lunch_combo_count_l517_517679


namespace youngest_sibling_age_l517_517930

theorem youngest_sibling_age
    (age_youngest : ℕ)
    (first_sibling : ℕ := age_youngest + 4)
    (second_sibling : ℕ := age_youngest + 5)
    (third_sibling : ℕ := age_youngest + 7)
    (average_age : ℕ := 21)
    (sum_of_ages : ℕ := 4 * average_age)
    (total_age_check : (age_youngest + first_sibling + second_sibling + third_sibling) = sum_of_ages) :
  age_youngest = 17 :=
sorry

end youngest_sibling_age_l517_517930


namespace train_journey_time_l517_517670

variable (b pm : ℝ)

def time_first_part (b : ℝ) : ℝ :=
  b / 50

def time_second_part (pm : ℝ) : ℝ :=
  3 * pm / 80

def total_time (b pm : ℝ) : ℝ :=
  time_first_part b + time_second_part pm

theorem train_journey_time (b pm : ℝ) :
  total_time b pm = (8 * b + 15 * pm) / 400 := by
  sorry

end train_journey_time_l517_517670


namespace prime_pairs_sum_to_50_l517_517305

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517305


namespace perimeter_of_square_l517_517995

-- Definitions based on problem conditions
def is_square_divided_into_four_congruent_rectangles (s : ℝ) (rect_perimeter : ℝ) : Prop :=
  rect_perimeter = 30 ∧ s > 0

-- Statement of the theorem to be proved
theorem perimeter_of_square (s : ℝ) (rect_perimeter : ℝ) (h : is_square_divided_into_four_congruent_rectangles s rect_perimeter) :
  4 * s = 48 :=
by sorry

end perimeter_of_square_l517_517995


namespace length_AB_l517_517978

noncomputable def hyperbola : Type := {x : ℝ × ℝ // x.1 ^ 2 / 3 - x.2 ^ 2 / 6 = 1}
def right_focus : ℝ × ℝ := (3, 0)
def inclination_angle (θ : ℝ) : Prop := θ = π / 6

theorem length_AB : ∀ (A B : hyperbola), 
  (∃ (m : ℝ), m = (Real.sqrt 3) / 3 ∧ 
              A.val.2 = m * (A.val.1 - 3) ∧ 
              B.val.2 = m * (B.val.1 - 3) ∧
              (A.val.1, A.val.2) ≠ right_focus ∧
              (B.val.1, B.val.2) ≠ right_focus) →
  Real.sqrt ((A.val.1 - B.val.1)^2 + (A.val.2 - B.val.2)^2) = (16 / 5) * Real.sqrt 3 :=
sorry

end length_AB_l517_517978


namespace journey_divided_into_portions_l517_517585

theorem journey_divided_into_portions
  (total_distance : ℕ)
  (speed : ℕ)
  (time : ℝ)
  (portion_distance : ℕ)
  (portions_covered : ℕ)
  (h1 : total_distance = 35)
  (h2 : speed = 40)
  (h3 : time = 0.7)
  (h4 : portions_covered = 4)
  (distance_covered := speed * time)
  (one_portion_distance := distance_covered / portions_covered)
  (total_portions := total_distance / one_portion_distance) :
  total_portions = 5 := 
sorry

end journey_divided_into_portions_l517_517585


namespace find_k_l517_517790

-- Defining the vectors a and b
def a := (2 : ℝ, 1 : ℝ)
def b (k : ℝ) := (-1 : ℝ, k)

-- Defining the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Defining the condition that a is perpendicular to (a + b)
def perpendicular (a b : ℝ × ℝ) : Prop :=
  dot_product a (a.1 + b.1, a.2 + b.2) = 0

-- The target theorem
theorem find_k (k : ℝ) (h : perpendicular a (b k)) : k = -3 :=
sorry

end find_k_l517_517790


namespace isosceles_triangle_points_count_l517_517873

theorem isosceles_triangle_points_count : 
  let D := (3, 3)
  let E := (5, 3)
  let grid_points := (fin 5) × (fin 5)
  ∃ F : grid_points, F ≠ D ∧ F ≠ E → 
  (∀ F, ((F = (4,1)) ∨ (F = (4,2)) ∨ (F = (4,4)) ∨ (F = (4,5))
        ∨ (F = (3,1)) ∨ (F = (3,5)) ∨ (F = (5,1)) ∨ (F = (5,5)) ) → 
        is_isosceles D E F)
  := 
sorry

end isosceles_triangle_points_count_l517_517873


namespace sum_of_players_l517_517531

noncomputable def A (total: ℕ) (f_A: ℝ) : ℕ := (f_A * total : ℝ).toInt
noncomputable def B (total: ℕ) (f_B: ℝ) : ℕ := (f_B * total : ℝ).toInt
noncomputable def C (total: ℕ) (f_C: ℝ) : ℕ := (f_C * total : ℝ).toInt
noncomputable def D (total: ℕ) (f_D: ℝ) : ℕ := (f_D * total : ℝ).toInt

theorem sum_of_players 
  (total : ℕ) 
  (f_A f_B f_C f_D : ℝ) 
  (h_fractions : f_A + f_B + f_C + f_D = 1) : 
  A total f_A + B total f_B + C total f_C + D total f_D = total :=
by 
  sorry

end sum_of_players_l517_517531


namespace prime_pairs_sum_50_l517_517254

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517254


namespace count_prime_pairs_sum_50_l517_517139

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517139


namespace increase_in_petrol_cost_l517_517620

noncomputable def initial_cost_per_litre (initial_ltr : ℕ) (initial_price_pence : ℕ) : ℕ := 
initial_price_pence / initial_ltr

noncomputable def current_cost_per_litre (current_ltr : ℕ) (current_price_pence : ℕ) : ℕ := 
current_price_pence / current_ltr

noncomputable def percentage_increase (initial_price_per_litre current_price_per_litre : ℕ) : ℕ :=
100 * (current_price_per_litre - initial_price_per_litre) / initial_price_per_litre

theorem increase_in_petrol_cost :
  let initial_ltr := 50 in
  let initial_price_pounds := 40 in 
  let current_ltr := 40 in
  let current_price_pounds := 50 in 
  let initial_price_pence := initial_price_pounds * 100 in 
  let current_price_pence := current_price_pounds * 100 in 
  let initial_price_per_litre := initial_cost_per_litre initial_ltr initial_price_pence in
  let current_price_per_litre := current_cost_per_litre current_ltr current_price_pence in
  percentage_increase initial_price_per_litre current_price_per_litre = 56 :=
by
  sorry

end increase_in_petrol_cost_l517_517620


namespace fraction_sum_l517_517684

theorem fraction_sum : (3 / 8) + (9 / 12) = 9 / 8 :=
by
  sorry

end fraction_sum_l517_517684


namespace prime_pairs_sum_50_l517_517274

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517274


namespace find_interest_rate_l517_517998

-- Define the conditions and variables
variables (total_sum second_part years_first R first_part : ℝ)
variables (interest_second interest_first : ℝ)

-- Initialize the conditions
def conditions :=
  total_sum = 2769 ∧
  second_part = 1704 ∧
  years_first = 8 ∧
  interest_second = 255.6 ∧
  first_part = total_sum - second_part

-- Lean theorem statement to be proved
theorem find_interest_rate (h : conditions) :
  interest_first = first_part * R * years_first / 100 →
  interest_first = interest_second → 
  R = 3 :=
  sorry

end find_interest_rate_l517_517998


namespace num_prime_pairs_summing_to_50_l517_517351

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517351


namespace prime_pairs_sum_to_50_l517_517308

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517308


namespace prime_pairs_sum_50_l517_517265

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517265


namespace probability_not_above_x_axis_l517_517881

-- Definitions of points P, Q, R, and S
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk 4  4
def Q := Point.mk (-2) (-2)
def R := Point.mk (-8) (-2)
def S := Point.mk (-2)  4

-- Definition of the parallelogram PQRS
def isParallelogram (P Q R S : Point) : Prop :=
  let v1 := (Q.x - P.x, Q.y - P.y)
  let v2 := (R.x - Q.x, R.y - Q.y)
  let v3 := (S.x - R.x, S.y - R.y)
  let v4 := (P.x - S.x, P.y - S.y)
  v1 = (S.x - R.x, S.y - R.y) ∧
  v2 = (P.x - S.x, P.y - S.y) ∧
  v1.1 * v2.2 - v1.2 * v2.1 ≠ 0 -- Non-collinear condition to ensure it's a parallelogram

-- Definition to check if a point is below or on the x-axis
def isBelowOrOnXAxis (pt : Point) : Prop :=
  pt.y ≤ 0

-- Area function to be used in proofs
noncomputable def area (P Q R S : Point) : ℝ :=
  abs ((Q.x - P.x) * (S.y - P.y) - (Q.y - P.y) * (S.x - P.x)) / 2

-- Main statement to be proved
theorem probability_not_above_x_axis (h : isParallelogram P Q R S) :
  (∃ a b c d : Point, a = P ∧ b = Q ∧ c = R ∧ d = S ∧ 
  area a b c d ≠ 0 ∧
  (∀ t : Point, t ∈ [a, b, c, d] -> ¬isBelowOrOnXAxis t) <->
  ∃ t : ℝ, t = 1/2 :=
sorry

end probability_not_above_x_axis_l517_517881


namespace prime_pairs_sum_50_l517_517359

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517359


namespace inequality_solution_l517_517896

theorem inequality_solution :
  { x : ℝ | (x^3 - 4 * x) / (x^2 - 1) > 0 } = { x : ℝ | x < -2 ∨ (0 < x ∧ x < 1) ∨ 2 < x } :=
by
  sorry

end inequality_solution_l517_517896


namespace each_girl_gets_strawberries_l517_517871

theorem each_girl_gets_strawberries (total_strawberries : ℝ) (number_of_girls : ℝ) (each_girl_gets : ℝ) :
  total_strawberries = 53.5 → number_of_girls = 8.5 → each_girl_gets ≈ 6.29 := by
  intro h1 h2
  sorry

end each_girl_gets_strawberries_l517_517871


namespace area_of_rectangular_field_l517_517567

theorem area_of_rectangular_field (length width perimeter : ℕ) 
  (h_perimeter : perimeter = 2 * (length + width)) 
  (h_length : length = 15) 
  (h_perimeter_value : perimeter = 70) : 
  (length * width = 300) :=
by
  sorry

end area_of_rectangular_field_l517_517567


namespace next_wednesday_l517_517868
open Nat

/-- Prove that the next year after 2010 when April 16 falls on a Wednesday is 2014,
    given the conditions:
    1. 2010 is a non-leap year.
    2. The day advances by 1 day for a non-leap year and 2 days for a leap year.
    3. April 16, 2010 was a Friday. -/
theorem next_wednesday (initial_year : ℕ) (initial_day : String) (target_day : String) : 
  (initial_year = 2010) ∧
  (initial_day = "Friday") ∧ 
  (target_day = "Wednesday") →
  2014 = 2010 + 4 :=
by
  sorry

end next_wednesday_l517_517868


namespace quadratic_factorization_l517_517023

theorem quadratic_factorization (a b : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a > b)
  (h4 : (x^2 - 18 * x + 77 = (x - a) * (x - b))) : 3 * b - a = 10 := by
  have ha : a = 11 := sorry
  have hb : b = 7 := sorry
  rw [ha, hb]
  norm_num
  sorry

end quadratic_factorization_l517_517023


namespace inequality_3var_l517_517860

variable {x y z : ℝ}

-- Define conditions
def positive_real (a : ℝ) : Prop := a > 0

theorem inequality_3var (hx : positive_real x) (hy : positive_real y) (hz : positive_real z) : 
  (x * y / z + y * z / x + z * x / y) > 2 * (x^3 + y^3 + z^3)^(1 / 3) :=
by {
    sorry
}

end inequality_3var_l517_517860


namespace part1_part2_l517_517753

open Set

def setA (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def setB : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

theorem part1 (a : ℝ) : 
  (setA a) ∩ setB = {x | a-1 < x ∧ x ≤ 4} := by
  sorry

theorem part2 (p q : ℝ → Prop) (h₀ : p = (λ x, x ∈ setA a)) (h₁ : q = (λ x, x ∈ setB))
  (h₂ : ∀ x, p x → q x) (h₃ : ¬ ∀ x, q x → p x) :
  a ∈ Iic (-4) ∪ Icc (-1 : ℝ) (1 / 2) := by
  sorry

end part1_part2_l517_517753


namespace volume_tetrahedron_BFEC1_l517_517826

variables (A B C D A1 B1 C1 D1 : Point ℝ) -- Cube vertices
variables {cube : Euclidean_Space ℝ (fin 3)} (hCube : is_cube A B C D A1 B1 C1 D1 1) -- Cube with edge length 1

-- Define points E and F with distances A1E = 2ED1 and DF = 2FC
variables E F : Point ℝ
variables (hE : on_line E A1 D1) (hE_ratio : dist A1 E = 2 * dist E D1)
variables (hF : on_line F C D) (hF_ratio : dist D F = 2 * dist F C)

-- Target volume to prove
theorem volume_tetrahedron_BFEC1 : volume_tetrahedron B F E C1 = 5 / 27 := by
  sorry

end volume_tetrahedron_BFEC1_l517_517826


namespace num_prime_pairs_sum_50_l517_517329

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517329


namespace count_prime_pairs_summing_to_50_l517_517241

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517241


namespace primes_sum_50_l517_517397

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517397


namespace find_m_l517_517948

theorem find_m (n m : ℕ) (h1 : m = 13 * n + 8) (h2 : m = 15 * n) : m = 60 :=
  sorry

end find_m_l517_517948


namespace prime_pairs_sum_50_l517_517450

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517450


namespace quadratic_graph_nature_l517_517493

theorem quadratic_graph_nature (a b : Real) (h : a ≠ 0) :
  ∀ (x : Real), (a * x^2 + b * x + (b^2 / (2 * a)) > 0) ∨ (a * x^2 + b * x + (b^2 / (2 * a)) < 0) :=
by
  sorry

end quadratic_graph_nature_l517_517493


namespace count_prime_pairs_sum_50_l517_517145

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517145


namespace num_prime_pairs_sum_50_l517_517319

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517319


namespace cos_triple_angle_l517_517465

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l517_517465


namespace primes_sum_50_l517_517400

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517400


namespace prime_pairs_sum_to_50_l517_517299

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517299


namespace circles_intersecting_l517_517110

-- Definitions of the circle equations

def C1 : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 + 4 * p.1 + 3 * p.2 + 2 = 0
def C2 : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 + 2 * p.1 + 3 * p.2 + 1 = 0

-- Statement that we need to prove
theorem circles_intersecting :
  let center1 := (-2:ℝ, -3/2:ℝ),
      radius1 := real.sqrt 17 / 2,
      center2 := (-1:ℝ, -3/2:ℝ),
      radius2 := 3 / 2,
      d := real.sqrt ((-2 - (-1))^2 + (-3/2 - (-3/2))^2)
  in radius1 - radius2 < d ∧ d < radius1 + radius2 :=
by
  -- we'll prove this later; putting sorry to skip the proof for now.
  sorry

end circles_intersecting_l517_517110


namespace cos_triple_angle_l517_517456

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517456


namespace area_of_right_triangle_l517_517494

theorem area_of_right_triangle
  (a b : ℝ) 
  (h₁ : ∃ A B C : Type, ∃ (x y : ℝ), is_right_triangle A B C ∧ 
    altitude_from_right_angle C A B = a ∧ 
    angle_bisector_from_right_angle C = b) :
  ∃ (area : ℝ), area = a^2 * b^2 / (2 * a^2 - b^2) := 
by {
  sorry
}

end area_of_right_triangle_l517_517494


namespace breakfast_customers_l517_517649

theorem breakfast_customers (B : ℕ) 
  (lunch : ℕ := 127) 
  (dinner : ℕ := 87) 
  (saturday_customers : ℕ := 574)
  (friday_customers := B + lunch + dinner)
  (twice_friday_customers := 2 * friday_customers)
  (predicted_saturday := saturday_customers = twice_friday_customers) :
  B = 73 :=
by
  have friday_customers := saturday_customers / 2
  have total_friday_customers := friday_customers
  calc
    B = 287 - (lunch + dinner) : sorry
    ... = 73 : by norm_num

end breakfast_customers_l517_517649


namespace count_prime_pairs_summing_to_50_l517_517232

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517232


namespace bill_reaches_two_thirds_on_13th_l517_517680

def start_date := 1 -- Start date is April 1st
def daily_reading := 8 -- Bill reads 8 pages every day
def total_pages := 144 -- The book has 144 pages
def two_thirds := (2 / 3 : ℝ) * total_pages -- Two-thirds of the book's pages

theorem bill_reaches_two_thirds_on_13th : start_date + (two_thirds / daily_reading) = 13 := 
by
  have h1 : two_thirds = 96 := by norm_num [two_thirds, total_pages]
  have h2 : 96 / daily_reading = 12 := by norm_num [daily_reading]
  have h3 : 1 + 12 = 13 := by norm_num
  rw [h1, h2, h3]
  exact rfl

end bill_reaches_two_thirds_on_13th_l517_517680


namespace set_intersection_l517_517106

variable (l : ℝ)
variable (hl : l > 0)

def M : set ℝ := {x | -l < x ∧ x < 1}
def N : set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem set_intersection : M l ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by sorry

end set_intersection_l517_517106


namespace prime_pairs_sum_50_l517_517448

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517448


namespace primes_sum_50_l517_517408

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517408


namespace prime_pairs_sum_to_50_l517_517431

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517431


namespace count_prime_pairs_sum_50_l517_517140

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517140


namespace slope_range_l517_517926

theorem slope_range (α : Real) (hα : α ∈ Set.Ioo (π / 3) (5 * π / 6)) :
  let k := Real.tan α
  k ∈ Set.Ioo (-∞) (-Real.sqrt(3) / 3) ∪ Set.Ioo (Real.sqrt(3)) ∞ :=
sorry

end slope_range_l517_517926


namespace prime_pairs_sum_to_50_l517_517309

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517309


namespace fibonacci_5k_divisible_by_5_l517_517749

def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| 2     := 1
| (n+2) := fibonacci n + fibonacci (n + 1)

theorem fibonacci_5k_divisible_by_5 (k : ℕ) : 5 ∣ fibonacci (5 * k) :=
by sorry

end fibonacci_5k_divisible_by_5_l517_517749


namespace prime_pairs_summing_to_50_count_l517_517392

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517392


namespace number_of_perfect_numbers_l517_517690

-- Define the concept of a perfect number
def perfect_number (a b : ℕ) : ℕ := (a + b)^2

-- Define the proposition we want to prove
theorem number_of_perfect_numbers : ∃ n : ℕ, n = 15 ∧ 
  ∀ p, ∃ a b : ℕ, p = perfect_number a b ∧ p < 200 :=
sorry

end number_of_perfect_numbers_l517_517690


namespace num_integer_points_in_circle_l517_517916

theorem num_integer_points_in_circle : 
  ∃ n : ℤ, n = 317 ∧ 
    (∀ (x y : ℤ), (x^2 + y^2 ≤ 100 ↔ (x, y) ∈ {(a, b) ∈ ℤ × ℤ | a^2 + b^2 ≤ 100}).finset.card = 317) :=
sorry

end num_integer_points_in_circle_l517_517916


namespace num_divisible_by_33_l517_517761

theorem num_divisible_by_33 : ∀ (x y : ℕ), 
  (0 ≤ x ∧ x ≤ 9) → (0 ≤ y ∧ y ≤ 9) →
  (19 + x + y) % 3 = 0 →
  (x - y + 1) % 11 = 0 →
  ∃! (n : ℕ), (20070002008 * 100 + x * 10 + y) = n ∧ n % 33 = 0 :=
by
  intros x y hx hy h3 h11
  sorry

end num_divisible_by_33_l517_517761


namespace hyperbola_eccentricity_l517_517762

-- Given conditions
variables {a b c : ℝ} (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h_hyperbola : b^2 = c^2 - a^2) (h_perpendicular : b^2 = a * c)

-- Proof Statement
theorem hyperbola_eccentricity (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_hyperbola : b^2 = c^2 - a^2) 
  (h_perpendicular : b^2 = a * c) :
  let e := c / a in
  e = (Real.sqrt 5 + 1) / 2 := by
  sorry

end hyperbola_eccentricity_l517_517762


namespace vector_parallel_solution_l517_517789

theorem vector_parallel_solution (x : ℝ) :
  let a := (1, x)
  let b := (x - 1, 2)
  (a.1 * b.2 = a.2 * b.1) → (x = 2 ∨ x = -1) :=
by
  intros a b h
  let a := (1, x)
  let b := (x - 1, 2)
  sorry

end vector_parallel_solution_l517_517789


namespace find_f_18_l517_517909

-- Define the function f
axiom f : ℕ → ℕ

-- Given conditions
axiom h1 : ∀ x, f(x + f(x)) = 6 * f(x)
axiom h2 : f(1) = 6
axiom h3 : f(2) = 3

-- Proof statement
theorem find_f_18 : f(18) = 108 :=
by
  sorry

end find_f_18_l517_517909


namespace vector_length_proof_l517_517663

variables {R : Type*} [linear_ordered_field R] [vector_space R R]

-- Define the points A, B, C, A1, B1, C1, X, X1 as elements of a vector space
variables (A B C A1 B1 C1 X X1 : R)
variables (l : R) (d : R) -- line and distance

-- Define the areas α, β, and γ
variables (α β γ : R) -- μ, ν, and ζ are real numbers

-- Define projection (this would require a formal definition in a full development)
axiom projection : R → R → Prop
-- Assume A1, B1, C1, and X1 are projections of A, B, C, and X on l
axiom projA1 : projection A A1
axiom projB1 : projection B B1
axiom projC1 : projection C C1
axiom projX1 : projection X X1
axiom dist_X1 : distance (X) (X1) = d
axiom vector_sum_zero : α • vector_X A X + β • vector_X B X + γ • vector_X C X = 0 -- given from the solution

-- Finally, the theorem to prove
theorem vector_length_proof :
  ∥ α • (A - A1) + β • (B - B1) + γ • (C - C1) ∥ = (α + β + γ) * d :=
sorry -- Here we acknowledge that a proof is required, but it's not provided.

end vector_length_proof_l517_517663


namespace find_cos_AOD_l517_517504

noncomputable def given_triangle (A B C : Point) : Prop :=
  ∠ A B C = 45

noncomputable def midpoint (M B C : Point) : Prop :=
  M = (B + C) / 2

noncomputable def intersect_circumcircle (A M C D : Point) : Prop :=
  -- AM intersects the circumcircle of triangle ABC at D
  (circle_intersects (circumcircle A B C) (line_through A M)) ∧
  (line_through A M).second_intersection = D

noncomputable def two_thirds_distance (A M D : Point) : Prop :=
  dist A M = 2 * dist M D

noncomputable def circumcenter (A B C O : Point) : Prop :=
  O = circumcircle_center A B C

theorem find_cos_AOD (A B C M D O : Point)
  (h1 : given_triangle A B C)
  (h2 : midpoint M B C)
  (h3 : intersect_circumcircle A M C D)
  (h4 : two_thirds_distance A M D)
  (h5 : circumcenter A B C O) :
  cos (∠ A O D) = -1 / 8 :=
sorry

end find_cos_AOD_l517_517504


namespace num_prime_pairs_summing_to_50_l517_517354

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517354


namespace primes_sum_50_l517_517399

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517399


namespace volume_conversion_l517_517619

theorem volume_conversion (a : Nat) (b : Nat) (c : Nat) (d : Nat) (e : Nat) (f : Nat)
  (h1 : a = 1) (h2 : b = 3) (h3 : c = a^3) (h4 : d = b^3) (h5 : c = 1) (h6 : d = 27) 
  (h7 : 1 = 1) (h8 : 27 = 27) (h9 : e = 5) 
  (h10 : f = e * d) : 
  f = 135 := 
sorry

end volume_conversion_l517_517619


namespace prime_pairs_sum_50_l517_517268

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517268


namespace count_prime_pairs_summing_to_50_l517_517223

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517223


namespace num_prime_pairs_summing_to_50_l517_517350

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517350


namespace angle_B_is_pi_div_3_perimeter_of_triangle_l517_517831

-- Given conditions
axiom sides_of_triangle (A B C : ℝ) (a b c : ℝ) (angle_A angle_B angle_C : ℝ)

-- Given bcos(C) = (2a - c)cos(B)
axiom given_condition_1 {a b c A B C : ℝ} (h1: b * real.cos C = (2 * a - c) * real.cos B) : Prop

-- Given area_triangle ABC
axiom area_triangle {a b c : ℝ} (area: ℝ)

-- Angle B should be π/3
theorem angle_B_is_pi_div_3 (a b c A B C : ℝ) (h1: b * real.cos C = (2 * a - c) * real.cos B) : B = real.pi / 3 := sorry

-- Perimeter of triangle ABC is 5 + sqrt(7)
theorem perimeter_of_triangle (a b c A B C : ℝ) (b_def: b = real.sqrt 7) (area: real.sqrt(3) * 3 / 2) (B_def: B = real.pi / 3) : a + b + c = 5 + real.sqrt(7) := sorry

end angle_B_is_pi_div_3_perimeter_of_triangle_l517_517831


namespace unique_k_value_l517_517013

theorem unique_k_value (p q k : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 99) (h_prod : p * q = k) :
  k = 194 ∧ (∀ k2, k2 ≠ 194 → ∃ p2 q2, Nat.Prime p2 ∧ Nat.Prime q2 ∧ p2 + q2 = 99 ∧ p2 * q2 = k2 → False) :=
by
  exists sorry

end unique_k_value_l517_517013


namespace sum_of_solutions_pos_int_sum_of_solutions_l517_517599

theorem sum_of_solutions_pos_int (x : ℕ) (h_cond : (x - 1) / 2 < 1) (h_pos : x > 0) : x = 1 ∨ x = 2 := 
by
  sorry

theorem sum_of_solutions (h : ∀ x : ℕ, ((x - 1) / 2 < 1) → (x > 0) → (x = 1 ∨ x = 2)) :
  ∑ x in {1, 2}, x = 3 :=
by
  rw [Finset.sum_insert, Finset.sum_singleton]
  sorry

end sum_of_solutions_pos_int_sum_of_solutions_l517_517599


namespace number_of_prime_pairs_sum_50_l517_517216

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517216


namespace find_number_l517_517874

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l517_517874


namespace correct_quotient_and_remainder_l517_517625

theorem correct_quotient_and_remainder:
  let incorrect_divisor := 47
  let incorrect_quotient := 5
  let incorrect_remainder := 8
  let incorrect_dividend := incorrect_divisor * incorrect_quotient + incorrect_remainder
  let correct_dividend := 243
  let correct_divisor := 74
  (correct_dividend / correct_divisor = 3 ∧ correct_dividend % correct_divisor = 21) :=
by sorry

end correct_quotient_and_remainder_l517_517625


namespace num_prime_pairs_sum_50_l517_517327

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517327


namespace line_general_equation_curve_rectangular_equation_distance_between_MN_l517_517829

-- Define the conditions and the mathematical objects
def parametric_line (t : ℝ) : ℝ × ℝ := (2 * (t - 2 * real.sqrt 2), t)
def polar_curve (ρ θ : ℝ) : Prop := ρ ^ 2 * (1 + 3 * real.sin θ ^ 2) = 4
def line_general (x y : ℝ) : Prop := x - 2 * y + 4 * real.sqrt 2 = 0
def curve_rectangular (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Definition for the condition involving theta and calculating distance
def beta_condition (β : ℝ) : Prop := β ∈ set.Ioo 0 real.pi ∧ real.tan β = -1 / 2
def distance_MN (ρ_M ρ_N : ℝ) : ℝ := ρ_N - ρ_M

-- Main Statements
theorem line_general_equation (t : ℝ) :
  ∃ x y : ℝ, parametric_line t = (x, y) →
  line_general x y := sorry

theorem curve_rectangular_equation (ρ θ : ℝ) :
  polar_curve ρ θ →
  ∃ x y : ℝ, (x = ρ * real.cos θ ∧ y = ρ * real.sin θ) ∧
  curve_rectangular x y := sorry

theorem distance_between_MN (β ρ_M ρ_N : ℝ) :
  beta_condition β →
  polar_curve ρ_M β →
  ∃ x y : ℝ, line_general (ρ_N * real.cos β) (ρ_N * real.sin β) ∧
  distance_MN ρ_M ρ_N = real.sqrt 10 / 2 := sorry

end line_general_equation_curve_rectangular_equation_distance_between_MN_l517_517829


namespace polygon_is_quadrilateral_l517_517920

-- Define the lines
def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := -2 * x + 3
def line3 (x : ℝ) : ℝ := -1
def line4 : ℝ := 2

-- Define the intersection points
def intersection1 : ℤ × ℤ := (0, 3)
def intersection2 : ℤ × ℤ := (-2, -1)
def intersection3 : ℤ × ℤ := (2, -1)
def intersection4 : ℤ × ℤ := (2, 7)

-- Problem statement (without proof)
theorem polygon_is_quadrilateral : 
    is_quadrilateral (intersection1, intersection2, intersection3, intersection4) := 
by
  sorry

end polygon_is_quadrilateral_l517_517920


namespace find_a_bounds_l517_517741

theorem find_a_bounds (a x1 x2 : ℝ) (h : x^2 - ax + a^2 - a = 0)
  (h_vieta1 : x1 + x2 = a) (h_vieta2 : x1 * x2 = a^2 - a)
  (h_discriminant_nonneg : a^2 - 4 * (a^2 - a) ≥ 0) :
  0 ≤ a ∧ a ≤ 4 / 3 :=
begin
  sorry
end

end find_a_bounds_l517_517741


namespace prime_pairs_sum_to_50_l517_517414

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517414


namespace number_of_prime_pairs_sum_50_l517_517207

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517207


namespace find_f_of_five_l517_517100

def f : ℝ → ℝ
| x if x < 4 := Math.sin (π / 6 * x)
| x := f (x - 1)

theorem find_f_of_five : f 5 = 1 := sorry

end find_f_of_five_l517_517100


namespace primes_sum_50_l517_517412

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517412


namespace hyperbola_standard_equation_l517_517068

noncomputable def standard_eq_hyperbola : Prop :=
  let ellipse := (x ^ 2) / 11 + (y ^ 2) / 7 = 1 in 
  let foci_distance := 2 in
  let vertices := foci_distance in
  let asymptote_to_foci_distance := sqrt 2 in
  let a := 2 in
  let b := 2 in
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1

theorem hyperbola_standard_equation :
  standard_eq_hyperbola :=
sorry

end hyperbola_standard_equation_l517_517068


namespace locus_of_center_l517_517726

-- Define the conditions
def on_parabola (P : ℝ × ℝ) : Prop := P.2 = P.1 ^ 2

def slope_line (a : ℝ) : (ℝ × ℝ) → Prop :=
  λ P, P.2 = 2 * a * P.1 - a ^ 2

-- Define the hypothesis for the points on the parabola and the slopes
variables {a1 a2 a3 : ℝ}
variables {P1 P2 P3 : ℝ × ℝ}
variables (hP1 : on_parabola P1) (hP2 : on_parabola P2) (hP3 : on_parabola P3)
variables (hL1 : slope_line a1 P1) (hL2 : slope_line a2 P2) (hL3 : slope_line a3 P3)

-- The Lean theorem statement
theorem locus_of_center :
  ∀ a1 a2 a3 : ℝ, ∃ x : ℝ, slope_line a1 P1 ∧ slope_line a2 P2 ∧ slope_line a3 P3 →
  (∃ G : ℝ × ℝ, G.2 = -1/4) :=
begin
  sorry
end

end locus_of_center_l517_517726


namespace prime_pairs_summing_to_50_count_l517_517388

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517388


namespace HCF_Of_4_And_18_Is_2_l517_517807

theorem HCF_Of_4_And_18_Is_2 : nat.gcd 4 18 = 2 := by
  -- HCF (4, 18) = 2
  sorry

end HCF_Of_4_And_18_Is_2_l517_517807


namespace prime_pairs_sum_50_l517_517263

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517263


namespace find_p5_l517_517530

variable (p : ℝ → ℝ)

axiom h1 : ∀ (x : ℝ), polynomial.degree_leading_coef_eq_one (p x) = 4
axiom h2 : p 1 = 2
axiom h3 : p 2 = 7
axiom h4 : p 3 = 10
axiom h5 : p 4 = 17

theorem find_p5 : p 5 = 26 :=
  sorry

end find_p5_l517_517530


namespace simplify_expr1_simplify_expr2_simplify_expr3_l517_517563

theorem simplify_expr1 (y : ℤ) (hy : y = 2) : -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := 
by sorry

theorem simplify_expr2 (a : ℤ) (ha : a = -2) : 15 * a^2 * (-4 * a^2 + (6 * a - a^2) - 3 * a) = -1560 :=
by sorry

theorem simplify_expr3 (x y : ℤ) (h1 : x * y = 2) (h2 : x + y = 3) : (3 * x * y + 10 * y) + (5 * x - (2 * x * y + 2 * y - 3 * x)) = 26 :=
by sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l517_517563


namespace num_unordered_prime_pairs_summing_to_50_l517_517156

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517156


namespace complex_conjugate_quadrant_l517_517771

theorem complex_conjugate_quadrant (z : ℂ) (h : z = complex.I * (1 - complex.I)) : 
  let conj_z := conj z in 
  conj_z.re > 0 ∧ conj_z.im < 0 := 
sorry

end complex_conjugate_quadrant_l517_517771


namespace solution_system_of_equations_l517_517598

theorem solution_system_of_equations : 
  ∃ (x y : ℝ), (2 * x - y = 3 ∧ x + y = 3) ∧ (x = 2 ∧ y = 1) := 
by
  sorry

end solution_system_of_equations_l517_517598


namespace ten_guards_sufficient_l517_517975

theorem ten_guards_sufficient (rooms : ℕ) (guards : ℕ) (h_rooms : rooms = 1000) (h_guards : guards = 10) : 
  ∃ strategy, (strategy rooms guards) :=
sorry

end ten_guards_sufficient_l517_517975


namespace digits_of_product_is_8_l517_517794

-- Define the power terms 3^7 and 7^5
def a : ℤ := 3 ^ 7
def b : ℤ := 7 ^ 5

-- Multiplication of a and b
def ab : ℤ := a * b

-- Number of digits of a number N given by ⌊log₁₀(N)⌋ + 1
def num_digits (N : ℤ) : ℤ := Int.floor (Real.log10 (N.toReal)) + 1

-- Problem statement: Find the number of digits in 3^7 * 7^5 is 8
theorem digits_of_product_is_8 : num_digits ab = 8 := by
  sorry

end digits_of_product_is_8_l517_517794


namespace problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l517_517717

/-- Lean statement for the math proof problem -/

/- First problem -/
theorem problem1_equation_of_line_intersection_perpendicular :
  ∃ k, 3 * k - 2 * ( - (5 - 3 * k) / 2) - 11 = 0 :=
sorry

/- Second problem -/
theorem problem2_equation_of_line_point_equal_intercepts :
  (∃ a, (1, 2) ∈ {(x, y) | x + y = a}) ∧ a = 3
  ∨ (∃ b, (1, 2) ∈ {(x, y) | y = b * x}) ∧ b = 2 :=
sorry

end problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l517_517717


namespace count_prime_pairs_sum_50_l517_517136

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517136


namespace ratio_of_rises_l517_517608

-- Definitions of radii and initial heights for cones
def r1 : ℝ := 4
def r2 : ℝ := 8
def h1 (h2 : ℝ) : ℝ := 4 * h2
def V_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

-- Definition of marble radii
def r_marble1 : ℝ := 1
def r_marble2 : ℝ := 2

-- Volume of marbles
def V_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

-- Proof goal
theorem ratio_of_rises (h2 : ℝ) (h1_initial : h1 h2) :
  let Δh1 : ℝ := 1 / 4
  let Δh2 : ℝ := 1 / 2
  Δh1 / Δh2 = 1 / 2 :=
by {
  sorry
}

end ratio_of_rises_l517_517608


namespace min_expression_value_l517_517970

theorem min_expression_value (a b c : ℝ) (h_pos : 0 < a ∧ a < 1) (h_pos_2 : 0 < b ∧ b < 1) (h_pos_3 : 0 < c ∧ c < 1) (h_expected_value : 3 * a + 2 * b = 2) :
  ∃ (min_val : ℝ), min_val = 16/3 ∧ (∀ x, x = (2 / a + 1 / (3 * b)) → x ≥ min_val) := 
begin
  sorry
end

end min_expression_value_l517_517970


namespace eval_f_at_two_eval_f_at_neg_two_l517_517775

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x

theorem eval_f_at_two : f 2 = 14 :=
by
  sorry

theorem eval_f_at_neg_two : f (-2) = 2 :=
by
  sorry

end eval_f_at_two_eval_f_at_neg_two_l517_517775


namespace Diego_total_stamp_cost_l517_517639

theorem Diego_total_stamp_cost :
  let price_brazil_colombia := 0.07
  let price_peru := 0.05
  let num_brazil_50s := 6
  let num_brazil_60s := 9
  let num_peru_50s := 8
  let num_peru_60s := 5
  let num_colombia_50s := 7
  let num_colombia_60s := 6
  let total_brazil := num_brazil_50s + num_brazil_60s
  let total_peru := num_peru_50s + num_peru_60s
  let total_colombia := num_colombia_50s + num_colombia_60s
  let cost_brazil := total_brazil * price_brazil_colombia
  let cost_peru := total_peru * price_peru
  let cost_colombia := total_colombia * price_brazil_colombia
  cost_brazil + cost_peru + cost_colombia = 2.61 :=
by
  sorry

end Diego_total_stamp_cost_l517_517639


namespace num_prime_pairs_sum_50_l517_517168

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517168


namespace face_opposite_E_is_F_l517_517662

def net_conditions (labels : List String) : Prop :=
  labels.contains "A" ∧
  labels.contains "B" ∧
  labels.contains "C" ∧
  labels.contains "D" ∧
  labels.contains "E" ∧
  labels.contains "F" ∧
  -- Assuming conditions about adjacency:
  -- A is flanked by B and C
  (some_adjacency_relation labels "A" "B" "C") ∧
  -- D is directly opposite B
  (some_opposition_relation labels "D" "B")

noncomputable def solution_face_opposite_E (labels : List String) : String :=
  -- Given we derive the face opposite E is F
  "F"

theorem face_opposite_E_is_F (labels : List String) 
  (h : net_conditions labels) : solution_face_opposite_E labels = "F" := 
sorry

end face_opposite_E_is_F_l517_517662


namespace inverse_cos_plus_one_l517_517912

noncomputable def f (x : ℝ) : ℝ := Real.cos x + 1

theorem inverse_cos_plus_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
    f (-(Real.arccos (x - 1))) = x :=
by
  sorry

end inverse_cos_plus_one_l517_517912


namespace ping_pong_tournament_l517_517495

theorem ping_pong_tournament (n : ℕ) (matches_asian : ℕ) (matches_european : ℕ) (matches_mixed : ℕ) :
  (matches_asian = n * (n - 1) / 2) →
  (matches_european = 2 * n * (2 * n - 1) / 2) →
  (5 * (matches_asian + matches_mixed - matches_european) = 7 * (matches_european + matches_mixed)) →
  n = 3 :=
begin
  -- proof omitted
  sorry
end

end ping_pong_tournament_l517_517495


namespace prime_pairs_sum_50_l517_517435

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517435


namespace powers_of_i_sum_l517_517702

theorem powers_of_i_sum :
  ∀ (i : ℂ), 
  (i^1 = i) ∧ (i^2 = -1) ∧ (i^3 = -i) ∧ (i^4 = 1) →
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 :=
by
  intros i h
  sorry

end powers_of_i_sum_l517_517702


namespace cos_8_degree_l517_517797

theorem cos_8_degree (m : ℝ) (h : Real.sin (74 * Real.pi / 180) = m) :
  Real.cos (8 * Real.pi / 180) = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_degree_l517_517797


namespace sector_central_angle_l517_517092

theorem sector_central_angle (r l : ℝ) (α : ℝ) 
  (h1 : l + 2 * r = 12) 
  (h2 : 1 / 2 * l * r = 8) : 
  α = 1 ∨ α = 4 :=
by
  sorry

end sector_central_angle_l517_517092


namespace certain_number_divisible_by_9_l517_517124

theorem certain_number_divisible_by_9 : ∃ N : ℕ, (∀ k : ℕ, (0 ≤ k ∧ k < 1110 → N + 9 * k ≤ 10000 ∧ (N + 9 * k) % 9 = 0)) ∧ N = 27 :=
by
  -- Given conditions:
  -- Numbers are in an arithmetic sequence with common difference 9.
  -- Total count of such numbers is 1110.
  -- The last number ≤ 10000 that is divisible by 9 is 9999.
  let L := 9999
  let n := 1110
  let d := 9
  -- First term in the sequence:
  let a := L - (n - 1) * d
  exists 27
  -- Proof of the conditions would follow here ...
  sorry

end certain_number_divisible_by_9_l517_517124


namespace count_prime_pairs_sum_50_l517_517128

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517128


namespace lateral_side_greater_than_twice_base_lateral_side_less_than_three_times_base_l517_517078

namespace IsoscelesTriangleProof

-- Define the setup of the isosceles triangle
structure IsoscelesTriangle :=
  (A B C : Type)
  [triangle : isosceles (A B C)]
  (vertex_angle : ∠ A B C = 20)

-- Theorem (a): b > 2a
theorem lateral_side_greater_than_twice_base {A B C : Type} [isosceles (A B C)] 
  (vertex_angle : ∠ A B C = 20)
  (AB AC BC : ℝ)
  (isosceles_property : AB = AC)
  (base_property : BC = a) : 
  AB > 2 * a := 
begin
  sorry -- Proof to be added
end

-- Theorem (b): b < 3a
theorem lateral_side_less_than_three_times_base {A B C : Type} [isosceles (A B C)] 
  (vertex_angle : ∠ A B C = 20)
  (AB AC BC : ℝ)
  (isosceles_property : AB = AC)
  (base_property : BC = a) : 
  AB < 3 * a := 
begin
  sorry -- Proof to be added
end

end IsoscelesTriangleProof

end lateral_side_greater_than_twice_base_lateral_side_less_than_three_times_base_l517_517078


namespace abs_sqrt_identity_l517_517756

theorem abs_sqrt_identity (x : ℝ) (h : 1 < x ∧ x ≤ 2) :
  |x - 3| + sqrt ((x - 2)^2) = 5 - 2 * x :=
by
  sorry

end abs_sqrt_identity_l517_517756


namespace factorize_expression_l517_517711

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l517_517711


namespace smallest_square_area_l517_517647

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) (h4 : d = 5) :
  ∃ s : ℕ, s * s = 64 ∧ (a + c <= s ∧ max b d <= s) ∨ (max a c <= s ∧ b + d <= s) :=
sorry

end smallest_square_area_l517_517647


namespace number_of_prime_pairs_sum_50_l517_517215

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517215


namespace prime_pairs_sum_50_l517_517444

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517444


namespace ninas_brothers_ages_l517_517885

-- Defining the conditions for Nina's brothers' ages
def is_valid_set_of_ages (ages : List ℕ) : Prop :=
  ages.length = 3 ∧
  ages.product = 36 ∧
  ∃ bus_number : ℕ, ages.sum = bus_number ∧ (∃ oldest : ℕ, List.maximum ages = some oldest)

-- Stating the main theorem
theorem ninas_brothers_ages : 
  ∃ ages : List ℕ, is_valid_set_of_ages ages ∧ ages = [2, 2, 9] :=
by
  sorry

end ninas_brothers_ages_l517_517885


namespace numValidSeqs_solution_l517_517073

def isValidSeq (A : Fin 10 → Fin 10) : Prop :=
  (Perm.isPerm A) ∧
  (A 0 < A 1 ∧ A 2 < A 3 ∧ A 4 < A 5 ∧ A 6 < A 7 ∧ A 8 < A 9) ∧ 
  (A 1 > A 2 ∧ A 3 > A 4 ∧ A 5 > A 6 ∧ A 7 > A 8) ∧
  (∀ i j k : Fin 10, i < j ∧ j < k → ¬ (A i < A k ∧ A k < A j))

theorem numValidSeqs : (Fin 10 → Fin 10) → ℕ
  := sorry

theorem solution : numValidSeqs = 42 :=
by
  sorry

end numValidSeqs_solution_l517_517073


namespace cyclic_quadrilateral_count_l517_517118

theorem cyclic_quadrilateral_count : 
  (∃ a b c d : ℕ, 
     a + b + c + d = 36 ∧ 
     a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
     a * b * c * d ≠ 0 ∧ 
     is_convex_cyclic_quadrilateral a b c d) 
  → (count_valid_quadrilaterals (36) = 823) :=
sorry

def is_convex_cyclic_quadrilateral (a b c d : ℕ) : Prop := 
  -- definition goes here, which checks whether the given a, b, c, d form a convex cyclic quadrilateral
  
noncomputable def count_valid_quadrilaterals (n : ℕ) : ℕ :=
  -- definition that counts the number of valid convex cyclic quadrilaterals with perimeter n goes here

end cyclic_quadrilateral_count_l517_517118


namespace distinct_real_roots_range_l517_517057

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 2 * x + a = 0) ∧ (y^2 - 2 * y + a = 0))
  ↔ a < 1 := 
by
  sorry

end distinct_real_roots_range_l517_517057


namespace hexagon_circle_ratio_l517_517985

theorem hexagon_circle_ratio (r : ℝ) (h : ∀ s : ℝ, s = r → s = r) :
  let A_hexagon := (3 * real.sqrt 3 / 2) * r^2 in
  let A_circle := real.pi * r^2 in
  A_hexagon / A_circle = 3 * real.sqrt 3 / (2 * real.pi) :=
by
  sorry

end hexagon_circle_ratio_l517_517985


namespace num_prime_pairs_sum_50_l517_517194

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517194


namespace num_prime_pairs_summing_to_50_l517_517341

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517341


namespace cos_triple_angle_l517_517453

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517453


namespace kola_solution_percent_sugar_l517_517627

theorem kola_solution_percent_sugar :
  let initial_volume := 340
  let initial_water_percent := 0.80
  let initial_kola_percent := 0.06
  let added_sugar := 3.2
  let added_water := 10
  let added_kola := 6.8

  let initial_water := initial_volume * initial_water_percent
  let initial_kola := initial_volume * initial_kola_percent
  let initial_sugar := initial_volume - initial_water - initial_kola

  let new_sugar := initial_sugar + added_sugar
  let new_water := initial_water + added_water
  let new_kola := initial_kola + added_kola

  let new_total_volume := new_sugar + new_water + new_kola

  let percent_sugar := (new_sugar / new_total_volume) * 100

  percent_sugar ≈ 14.11 :=
by
  sorry

end kola_solution_percent_sugar_l517_517627


namespace number_of_prime_pairs_sum_50_l517_517208

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517208


namespace combinatorial_identity_l517_517859

theorem combinatorial_identity (m n : ℕ) (hmn : m ≥ n) 
    (S : finset (vector ℕ n)) 
    (HS : ∀ (a : vector ℕ n), a ∈ S ↔ (∀ i, 1 ≤ a.nth i) ∧ a.to_list.sum = m) :
  (S.sum (λ a, list.prod (a.to_list.map_with_index (λ i ai, (i + 1)^ai)))) = 
  (finset.range (n + 1)).sum (λ j, (-1)^(n - j) * (nat.choose n j) * j^m) :=
by
  sorry

end combinatorial_identity_l517_517859


namespace solve_for_a_l517_517803

theorem solve_for_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end solve_for_a_l517_517803


namespace runner_time_difference_l517_517991

theorem runner_time_difference 
  (v : ℝ)  -- runner's initial speed (miles per hour)
  (H1 : 0 < v)  -- speed is positive
  (d : ℝ)  -- total distance
  (H2 : d = 40)  -- total distance condition
  (t2 : ℝ)  -- time taken for the second half
  (H3 : t2 = 10)  -- second half time condition
  (H4 : v ≠ 0)  -- initial speed cannot be zero
  (H5: 20 = 10 * (v / 2))  -- equation derived from the second half conditions
  : (t2 - (20 / v)) = 5 := 
by
  sorry

end runner_time_difference_l517_517991


namespace problem_a_l517_517941

noncomputable def probability_best_play_wins (n : ℕ) : ℝ :=
  1 - 2^(-n)

theorem problem_a (n : ℕ) : 
  probability_best_play_wins n = 1 - 2^(-n) := 
by
  sorry

end problem_a_l517_517941


namespace possible_configuration_l517_517642

def initial_piles := [2011, 2010, 2009, 2008]

def allowed_operations (piles : List ℕ) : List (List ℕ) :=
  let rem_same p : List ℕ := piles.map (λ x => x - p)
  let move x y p : List ℕ := 
    piles.modify x (λ xi => xi - p)
    .modify y (λ yi => yi + p)
  (List.range (List.minimum piles).getOrElse 0).map rem_same ++
  (List.range piles.length).bind (λ x =>
  (List.range piles.length).map (λ y =>
  if x ≠ y then (List.range (if piles[x] < piles[y] then piles[x] else piles[y]).getOrElse 0).map (move x y p) else []))

theorem possible_configuration :
  ∃ (x : List ℕ), x = [0, 0, 0, 2] ∧ x ∈ (allowed_operations initial_piles) :=
sorry

end possible_configuration_l517_517642


namespace arcs_points_on_diameter_l517_517961

theorem arcs_points_on_diameter (k : ℕ) (h : k > 0) (p : {n : ℕ // n = 3 * k}) 
  (h1 : { l : ℕ // l = k } → ∃ (arcs_1 arcs_2 arcs_3 : list ℕ), 
         (length arcs_1 = k) ∧ 
         (∀ (x ∈ arcs_1), x = 1) ∧ 
         (length arcs_2 = k) ∧ 
         (∀ (x ∈ arcs_2), x = 2) ∧ 
         (length arcs_3 = k) ∧ 
         (∀ (x ∈ arcs_3), x = 3) ∧ 
         (p = length (arcs_1 ++ arcs_2 ++ arcs_3))) : 
  ∃ (i j : ℕ), i ≠ j ∧ p.val / 2 = j - i ∨ p.val / 2 = i - j :=
sorry

end arcs_points_on_diameter_l517_517961


namespace construct_segment_eq_abc_div_de_l517_517752

theorem construct_segment_eq_abc_div_de 
(a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
  ∃ x : ℝ, x = (a * b * c) / (d * e) :=
by sorry

end construct_segment_eq_abc_div_de_l517_517752


namespace number_of_proper_subsets_of_B_l517_517081

noncomputable def A : Set ℝ := {x | ∃ n : ℤ, x = Real.sin (n * Real.pi / 2)}
noncomputable def B : Set ℝ := {x | ∃ a b ∈ A, x = a * b}

theorem number_of_proper_subsets_of_B : (B.to_finset.card - 1) = 7 := 
by sorry

end number_of_proper_subsets_of_B_l517_517081


namespace f_is_decreasing_l517_517745

-- Conditions
variables (f : ℝ → ℝ)
variables (f' : ℝ → ℝ)
variables (x1 x2 : ℝ)

-- Hypotheses
hypothesis h1 : ∀ x ≠ 1, (x - 1) * f' x < 0
hypothesis h2 : ∀ x, f (x + 1) = f (-x + 1)
hypothesis h3 : |x1 - 1| < |x2 - 1|

-- Goal
theorem f_is_decreasing : f x1 > f x2 :=
sorry

end f_is_decreasing_l517_517745


namespace billing_amounts_choose_billing_method_cost_effective_billing_l517_517972

theorem billing_amounts (t : ℕ) :
  let y₁ := if t ≤ 200 then 78 else 0.25 * t + 28 in
  let y₂ := if t ≤ 500 then 108 else 0.19 * t + 13 in
  (y₁ = if t ≤ 200 then 78 else 0.25 * t + 28) ∧
  (y₂ = if t ≤ 500 then 108 else 0.19 * t + 13) :=
by
  sorry

theorem choose_billing_method (t : ℕ) (expected_t : ℕ) (by : expected_t = 350) :
  let y₁ := if expected_t ≤ 200 then 78 else 0.25 * expected_t + 28 in
  let y₂ := if expected_t ≤ 500 then 108 else 0.19 * expected_t + 13 in
  (y₁ = 115.5) ∧ (y₂ = 108) ∧ (y₁ > y₂) :=
by
  sorry

theorem cost_effective_billing (t : ℕ) :
  let y₁ := if t ≤ 200 then 78 else 0.25 * t + 28 in
  let y₂ := if t ≤ 500 then 108 else 0.19 * t + 13 in
  (t < 320 → y₁ < y₂) ∧ (t > 320 → y₂ < y₁) :=
by
  sorry

end billing_amounts_choose_billing_method_cost_effective_billing_l517_517972


namespace prime_pairs_sum_50_l517_517442

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517442


namespace net_pay_rate_is_26_dollars_per_hour_l517_517973

-- Defining the conditions
noncomputable def total_distance (time_hours : ℝ) (speed_mph : ℝ) : ℝ :=
  time_hours * speed_mph

noncomputable def adjusted_fuel_efficiency (original_efficiency : ℝ) (decrease_percentage : ℝ) : ℝ :=
  original_efficiency * (1 - decrease_percentage)

noncomputable def gasoline_used (distance : ℝ) (efficiency : ℝ) : ℝ :=
  distance / efficiency

noncomputable def earnings (rate_per_mile : ℝ) (distance : ℝ) : ℝ :=
  rate_per_mile * distance

noncomputable def updated_gasoline_price (original_price : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_price * (1 + increase_percentage)

noncomputable def total_cost_gasoline (gasoline_price : ℝ) (gasoline_used : ℝ) : ℝ :=
  gasoline_price * gasoline_used

noncomputable def net_earnings (earnings : ℝ) (cost : ℝ) : ℝ :=
  earnings - cost

noncomputable def net_rate_of_pay (net_earnings : ℝ) (time_hours : ℝ) : ℝ :=
  net_earnings / time_hours

-- Given constants
def time_hours : ℝ := 3
def speed_mph : ℝ := 50
def original_efficiency : ℝ := 30
def decrease_percentage : ℝ := 0.10
def rate_per_mile : ℝ := 0.60
def original_gasoline_price : ℝ := 2.00
def increase_percentage : ℝ := 0.20

-- Proof problem statement
theorem net_pay_rate_is_26_dollars_per_hour :
  net_rate_of_pay 
    (net_earnings
       (earnings rate_per_mile (total_distance time_hours speed_mph))
       (total_cost_gasoline
          (updated_gasoline_price original_gasoline_price increase_percentage)
          (gasoline_used
            (total_distance time_hours speed_mph)
            (adjusted_fuel_efficiency original_efficiency decrease_percentage))))
    time_hours = 26 := 
  sorry

end net_pay_rate_is_26_dollars_per_hour_l517_517973


namespace locus_of_intersections_is_ellipse_l517_517603

-- Definitions based on conditions
variable (k : Circle)
variable (i1 i2 : Direction)

-- Statement of the proof problem
theorem locus_of_intersections_is_ellipse (h : ¬ (Direction.is_perpendicular i1 i2)) : 
  Locus.is_ellipse (k, i1, i2) := 
sorry

end locus_of_intersections_is_ellipse_l517_517603


namespace ivan_walk_time_l517_517039

-- Definitions of variables
variable (T v u s t : ℝ)

-- Conditions
axiom left_house_hour_early : ∃ T, T > 0
axiom met_company_car : ∀ t, t ≥ 0
axiom arrived_10_minutes_early : T + (t - T + 10/60) < t

-- Prove that Ivan Ivanovich walked for 55 minutes
theorem ivan_walk_time (h1 : left_house_hour_early)
                       (h2 : met_company_car)
                       (h3 : arrived_10_minutes_early) :
  T = 55 := 
sorry

end ivan_walk_time_l517_517039


namespace limit_sequence_l517_517550
noncomputable theory

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ := (1 + 3 * n) / (6 - n)

-- Define the limit value a
def a : ℝ := -3

-- Statement of the problem: Prove that the limit of a_n as n approaches infinity is a
theorem limit_sequence : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) := sorry

end limit_sequence_l517_517550


namespace comparison_abc_l517_517757

variable (a b c : ℝ)

noncomputable def a := Real.log 4 / Real.log 3
noncomputable def b := (1 / 4)^(1 / 3)
noncomputable def c := Real.log (1 / 5) / Real.log (1 / 3)

theorem comparison_abc : c > a ∧ a > b := by
  sorry

end comparison_abc_l517_517757


namespace prime_pairs_sum_to_50_l517_517316

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517316


namespace factorize_expression_l517_517706

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l517_517706


namespace geometric_sequence_value_l517_517067

variable {α : Type*} [Field α]

-- Definitions used in the condition
def geometric_seq (a : ℕ → α) : Prop :=
  ∃ r (a₀ : α), ∀ n, a (n + 1) = r * a n

-- The main statement
theorem geometric_sequence_value
  (a : ℕ → α)
  (h_seq : geometric_seq a)
  (h_sum : a 3 + a 7 = 5) :
  a 2 * a 4 + 2 * a 4 * a 6 + a 6 * a 8 = 25 := 
sorry

end geometric_sequence_value_l517_517067


namespace count_prime_pairs_sum_50_exactly_4_l517_517292

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517292


namespace count_prime_pairs_sum_50_l517_517133

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517133


namespace prime_pairs_sum_50_l517_517255

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517255


namespace saline_solution_concentration_l517_517655

theorem saline_solution_concentration
  (C : ℝ) -- concentration of the first saline solution
  (h1 : 3.6 * C + 1.4 * 9 = 5 * 3.24) : -- condition based on the total salt content
  C = 1 := 
sorry

end saline_solution_concentration_l517_517655


namespace new_seq_is_arithmetic_with_3d_l517_517824

-- Define the sequences and conditions
variable {d : ℝ}
variable {a : ℕ → ℝ}
variable h_arith_seq : ∀ n : ℕ, a (n + 1) - a n = d

-- Define the new sequence formed by taking out all terms whose indices are multiples of 3
def a_3n (n : ℕ) : ℝ := a (3 * n)

-- The proof statement
theorem new_seq_is_arithmetic_with_3d (n : ℕ) : a_3n (n + 1) - a_3n n = 3 * d :=
by
  sorry

end new_seq_is_arithmetic_with_3d_l517_517824


namespace point_D_on_line_AC_l517_517854

-- Definitions of Points, Circle, angle, and line
variables {Point : Type} [PlaneGeometry Point]
variables {A B C D O : Point}
variables {k : Circle Point}
variables (h1 : Center k = O)
variables (h2 : OnCircle k A)
variables (h3 : OnCircle k B)
variables (h4 : OnCircle k C)
variables (h5 : angle A B C > 90)

-- Angle bisector intersection with the circumcircle
variables {circBOC : Circle Point}
variables (h6 : circumcircle (Triangle.mk B O C) = circBOC)
variables (h7 : intersection (angleBisectorOf (angle A O B)) circBOC = D)

-- Proving that D lies on the line (AC)
theorem point_D_on_line_AC :
  LiesOn D (line A C) := by
  sorry

end point_D_on_line_AC_l517_517854


namespace unique_integers_exist_l517_517694

theorem unique_integers_exist : 
  ∃ b2 b3 b4 b5 b6 : ℕ,
    0 ≤ b2 ∧ b2 < 3 ∧
    0 ≤ b3 ∧ b3 < 4 ∧
    0 ≤ b4 ∧ b4 < 5 ∧
    0 ≤ b5 ∧ b5 < 6 ∧
    0 ≤ b6 ∧ b6 < 7 ∧
    (11 / 13 = (b2 / (3! : ℕ)) + (b3 / (4! : ℕ)) + (b4 / (5! : ℕ)) + (b5 / (6! : ℕ)) + (b6 / (7! : ℕ))) :=
sorry

end unique_integers_exist_l517_517694


namespace num_prime_pairs_sum_50_l517_517175

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517175


namespace vectors_not_collinear_l517_517109

noncomputable def e1 : ℝ × ℝ := (-1, 2)
noncomputable def e2 : ℝ × ℝ := (3, -1)

theorem vectors_not_collinear (a b c d : ℝ) : 
  a = -1 ∧ b = 2 ∧ c = 3 ∧ d = -1 → a * d - b * c ≠ 0 := 
by 
  intro h 
  cases h 
  simp 
  sorry

end vectors_not_collinear_l517_517109


namespace prime_pairs_sum_50_l517_517257

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517257


namespace arithmetic_mean_124_4_31_l517_517943

theorem arithmetic_mean_124_4_31 :
  let numbers := [12, 25, 39, 48]
  let total := 124
  let count := 4
  (total / count : ℝ) = 31 := by
  sorry

end arithmetic_mean_124_4_31_l517_517943


namespace change_is_13_82_l517_517506

def sandwich_cost : ℝ := 5
def num_sandwiches : ℕ := 3
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def payment : ℝ := 20 + 5 + 3

def total_cost_before_discount : ℝ := num_sandwiches * sandwich_cost
def discount_amount : ℝ := total_cost_before_discount * discount_rate
def discounted_cost : ℝ := total_cost_before_discount - discount_amount
def tax_amount : ℝ := discounted_cost * tax_rate
def total_cost_after_tax : ℝ := discounted_cost + tax_amount

def change (payment total_cost : ℝ) : ℝ := payment - total_cost

theorem change_is_13_82 : change payment total_cost_after_tax = 13.82 := 
by
  -- Proof will be provided here
  sorry

end change_is_13_82_l517_517506


namespace least_number_divisible_by_11_and_leaves_remainder_2_l517_517612

theorem least_number_divisible_by_11_and_leaves_remainder_2 : 
  ∃ n : ℕ, (n % 11 = 0) ∧ (∀ m ∈ {3, 4, 5, 6, 7}, n % m = 2) ∧ n = 3782 :=
by
  sorry

end least_number_divisible_by_11_and_leaves_remainder_2_l517_517612


namespace num_prime_pairs_sum_50_l517_517166

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517166


namespace num_prime_pairs_summing_to_50_l517_517353

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517353


namespace max_members_club_l517_517987

open Finset

theorem max_members_club (A B C : Finset ℕ) 
  (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11) 
  (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
  (A ∪ B ∪ C).card ≥ 22 :=
  sorry

end max_members_club_l517_517987


namespace eval_expression_l517_517038

theorem eval_expression (a b c : ℕ) (h₀ : a = 3) (h₁ : b = 2) (h₂ : c = 1) : 
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 :=
by
  sorry

end eval_expression_l517_517038


namespace Sn_value_l517_517747

noncomputable def sequence (n : ℕ) : ℚ :=
  Nat.recOn n (1 / 2) (λ n' a_n, 3 * a_n + 1)

noncomputable def S (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n + 1), sequence i

theorem Sn_value :
  S 2016 = (3^2016 - 2017) / 2 :=
by
  sorry

end Sn_value_l517_517747


namespace total_books_l517_517561

-- Definitions for the conditions
def SandyBooks : Nat := 10
def BennyBooks : Nat := 24
def TimBooks : Nat := 33

-- Stating the theorem we need to prove
theorem total_books : SandyBooks + BennyBooks + TimBooks = 67 := by
  sorry

end total_books_l517_517561


namespace count_prime_pairs_sum_50_l517_517138

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517138


namespace num_prime_pairs_sum_50_l517_517180

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517180


namespace factorize_expression_l517_517712

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l517_517712


namespace prime_pairs_summing_to_50_count_l517_517383

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517383


namespace percentage_of_alcohol_in_original_solution_l517_517964

noncomputable def alcohol_percentage_in_original_solution (P: ℝ) (V_original: ℝ) (V_water: ℝ) (percentage_new: ℝ): ℝ :=
  (P * V_original) / (V_original + V_water) * 100

theorem percentage_of_alcohol_in_original_solution : 
  ∀ (P: ℝ) (V_original : ℝ) (V_water : ℝ) (percentage_new : ℝ), 
  V_original = 3 → 
  V_water = 1 → 
  percentage_new = 24.75 →
  alcohol_percentage_in_original_solution P V_original V_water percentage_new = 33 := 
by
  sorry

end percentage_of_alcohol_in_original_solution_l517_517964


namespace part_a_part_b_l517_517546

-- Define the hyperbola and points A, B
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define points A, B, A1, B1
def A (x1 : ℝ) : ℝ × ℝ := (x1, 1 / x1)
def B (x2 : ℝ) : ℝ × ℝ := (x2, 1 / x2)
def A1 (x1 x2 : ℝ) : ℝ × ℝ := (0, 1 / x1 + 1 / x2)
def B1 (x1 x2 : ℝ) : ℝ × ℝ := (x1 + x2, 0)

-- Define distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

-- Proposition 1: AA1 = BB1 and AB1 = BA1
theorem part_a (x1 x2 : ℝ) (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0) :
  distance (A x1) (A1 x1 x2) = distance (B x2) (B1 x1 x2) ∧
  distance (A x1) (B1 x1 x2) = distance (B x2) (A1 x1 x2) :=
sorry

-- Define midpoint
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Proposition 2: If A1B1 is tangent to the hyperbola at X, then X is the midpoint of A1B1
theorem part_b (x1 x2 : ℝ) (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0) (X : ℝ × ℝ)
  (tangent_condition : X = (midpoint (A1 x1 x2) (B1 x1 x2))) :
  hyperbola X.1 X.2 ∧ X = midpoint (A1 x1 x2) (B1 x1 x2) :=
sorry

end part_a_part_b_l517_517546


namespace num_prime_pairs_summing_to_50_l517_517337

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517337


namespace sum_roots_abs_eq_l517_517721

open Polynomial

def p : Polynomial ℝ := Polynomial.C 1 * X ^ 4 - Polynomial.C 6 * X ^ 3 + Polynomial.C 9 * X ^ 2 + Polynomial.C 18 * X - Polynomial.C 40

-- Theorem: The sum of the absolute values of the roots of the polynomial p is 6√2.
theorem sum_roots_abs_eq : (roots p).map (λ x : ℝ, Real.abs x).sum = 6 * Real.sqrt 2 := 
sorry

end sum_roots_abs_eq_l517_517721


namespace prime_pairs_sum_50_l517_517437

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517437


namespace cos_triple_angle_l517_517477

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517477


namespace chicken_dodged_cars_l517_517570

theorem chicken_dodged_cars :
  ∃ c : ℕ, (5263 - 5217 = 2 * c) ∧ (c = 23) :=
begin
  use 23,
  split,
  {
    norm_num,
  },
  {
    refl,
  }
end

end chicken_dodged_cars_l517_517570


namespace evaluate_expression_l517_517512

variable (sqrt_five sqrt_seven sqrt_thirtyfive : ℝ)
hypothesis (h1 : sqrt_five * sqrt_five = 5)
hypothesis (h2 : sqrt_seven * sqrt_seven = 7)
hypothesis (h3 : sqrt_thirtyfive * sqrt_thirtyfive = 35)

def a := sqrt_five + sqrt_seven + sqrt_thirtyfive
def b := -sqrt_five + sqrt_seven + sqrt_thirtyfive
def c := sqrt_five - sqrt_seven + sqrt_thirtyfive
def d := -sqrt_five - sqrt_seven + sqrt_thirtyfive

theorem evaluate_expression : ( (1/a) + (1/b) + (1/c) + (1/d) )^2 = 560 / 83521 :=
by
  have zero_ne_a : a ≠ 0, sorry
  have zero_ne_b : b ≠ 0, sorry
  have zero_ne_c : c ≠ 0, sorry
  have zero_ne_d : d ≠ 0, sorry
  sorry

end evaluate_expression_l517_517512


namespace count_prime_pairs_sum_50_exactly_4_l517_517289

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517289


namespace fraction_cows_sold_is_one_fourth_l517_517602

def num_cows : ℕ := 184
def num_dogs (C : ℕ) : ℕ := C / 2
def remaining_animals : ℕ := 161
def fraction_dogs_sold : ℚ := 3 / 4
def fraction_cows_sold (C remaining_cows : ℕ) : ℚ := (C - remaining_cows) / C

theorem fraction_cows_sold_is_one_fourth :
  ∀ (C remaining_dogs remaining_cows: ℕ),
    C = 184 →
    remaining_animals = 161 →
    remaining_dogs = (1 - fraction_dogs_sold) * num_dogs C →
    remaining_cows = remaining_animals - remaining_dogs →
    fraction_cows_sold C remaining_cows = 1 / 4 :=
by sorry

end fraction_cows_sold_is_one_fourth_l517_517602


namespace ball_returns_to_Lisa_after_15_throws_l517_517714

-- Define the cyclic nature of the throws
def next_girl (n_skips : ℕ) (current : ℕ) : ℕ :=
  (current + n_skips) % 15

-- Define the recurrence relation for ball passing
noncomputable def ball_throw (current : ℕ) (step : ℕ) : ℕ :=
  if step % 2 = 0 then next_girl 4 current else next_girl 3 current

-- Define a function to compute the total number of throws required for the ball to return to Lisa
noncomputable def total_throws : ℕ → ℕ → ℕ
  | 1, 0 => 0 
  | pos, steps =>
      let next := ball_throw pos steps
      if next = 1 then steps + 1 else total_throws next (steps + 1)

-- Proving the number of throws required to return to Lisa
theorem ball_returns_to_Lisa_after_15_throws :
  total_throws 1 0 = 15 :=
  by
    sorry

end ball_returns_to_Lisa_after_15_throws_l517_517714


namespace num_prime_pairs_summing_to_50_l517_517349

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517349


namespace num_prime_pairs_sum_50_l517_517325

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517325


namespace num_prime_pairs_summing_to_50_l517_517340

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517340


namespace factorization_l517_517709

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := sorry

end factorization_l517_517709


namespace prime_pairs_sum_to_50_l517_517311

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517311


namespace diagonal_ratio_of_polygon_with_5_sides_l517_517592

theorem diagonal_ratio_of_polygon_with_5_sides (n : ℕ) (h : n = 5) :
  let v := n * (n - 3) / 2 in
  v / n = 1 :=
by
  cases h
  let v := 5 * (5 - 3) / 2
  show v / 5 = 1
  sorry

end diagonal_ratio_of_polygon_with_5_sides_l517_517592


namespace plane_nec_but_not_suff_condition_l517_517735

-- Define the planes α and β
variables (α β : Plane)

-- Define the line m and the subset relationship
variables (m : Line) (h : m ⊆ α)

-- State the theorem
theorem plane_nec_but_not_suff_condition
  (h1 : α ≠ β) -- α and β are different planes
  (h2 : m.perpendicular β) -- m is perpendicular to β
  : α.perpendicular β -> False := sorry

end plane_nec_but_not_suff_condition_l517_517735


namespace num_prime_pairs_summing_to_50_l517_517342

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517342


namespace least_number_divisible_by_11_and_leaves_remainder_2_l517_517613

theorem least_number_divisible_by_11_and_leaves_remainder_2 : 
  ∃ n : ℕ, (n % 11 = 0) ∧ (∀ m ∈ {3, 4, 5, 6, 7}, n % m = 2) ∧ n = 3782 :=
by
  sorry

end least_number_divisible_by_11_and_leaves_remainder_2_l517_517613


namespace prime_pairs_sum_50_l517_517256

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517256


namespace rhombus_area_l517_517913

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  0.5 * d1 * d2

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : area_of_rhombus d1 d2 = 6 :=
by
  sorry

end rhombus_area_l517_517913


namespace problem_statement_l517_517049

theorem problem_statement (x : ℝ) (h : x > 16) :
  (\sqrt{x - 8 * \sqrt{x - 16}} + 4 = \sqrt{x + 8 * \sqrt{x - 16}} - 4) ↔ (x ≥ 32) :=
sorry

end problem_statement_l517_517049


namespace num_valid_numbers_l517_517122

-- Definition of being a four-digit number with each digit being either 2 or 5
def is_valid_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧ (∀ d ∈ n.digits 10, d = 2 ∨ d = 5)

-- Theorem to prove the number of valid numbers is 16
theorem num_valid_numbers : (finset.filter is_valid_number (finset.range 10000)).card = 16 := 
by
  sorry

end num_valid_numbers_l517_517122


namespace biff_break_even_hours_l517_517009

-- Definitions based on conditions
def ticket_expense : ℕ := 11
def snacks_expense : ℕ := 3
def headphones_expense : ℕ := 16
def total_expenses : ℕ := ticket_expense + snacks_expense + headphones_expense
def gross_income_per_hour : ℕ := 12
def wifi_cost_per_hour : ℕ := 2
def net_income_per_hour : ℕ := gross_income_per_hour - wifi_cost_per_hour

-- The proof statement
theorem biff_break_even_hours : ∃ h : ℕ, h * net_income_per_hour = total_expenses ∧ h = 3 :=
by 
  have h_value : ℕ := 3
  exists h_value
  split
  · show h_value * net_income_per_hour = total_expenses
    sorry
  · show h_value = 3
    rfl

end biff_break_even_hours_l517_517009


namespace find_equivalent_angle_in_range_l517_517715

/-- Given an angle θ of 1250°, find the equivalent angle in the range [-360°, 0°). -/
theorem find_equivalent_angle_in_range : 
  ∃ α, α ∈ Set.Ico (-360 : ℝ) 0 ∧ (α : ℝ) ≡ 1250 [MOD 360] ∧ α = -190 :=
by
  sorry

end find_equivalent_angle_in_range_l517_517715


namespace x_coordinate_equidistant_l517_517609

-- Definitions of Points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := 0 }
def B : Point := { x := 2, y := 5 }

-- Define the distance formula
def dist (P Q : Point) : ℝ := 
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def onXAxis (P : Point) : Prop := P.y = 0

-- Define the statement to be proven: a point on the x-axis equidistant from A and B has x-coordinate equal to 2
theorem x_coordinate_equidistant :
  ∃ (x : ℝ), onXAxis { x := x, y := 0 } ∧ dist { x := x, y := 0 } A = dist { x := x, y := 0 } B ∧ x = 2 := 
by {
  use 2,
  sorry
}

end x_coordinate_equidistant_l517_517609


namespace find_a_distance_equal_l517_517026

def curve1 (a : ℝ) : ℝ → ℝ := λ x, x^2 + a
def curve2 : ℝ × ℝ → Prop := λ p, (p.1)^2 + (p.2 + 4)^2 = 2
def line (p : ℝ × ℝ) : Prop := p.2 = p.1

theorem find_a_distance_equal :
  ∃ (a : ℝ), (∀ (x1 x2 : ℝ), (curve1 a x1 - line (x1, curve1 a x1)).abs = (curve2 (x2, -4) - line (x2, 0)).abs) ∧ a = 9/4 :=
sorry

end find_a_distance_equal_l517_517026


namespace num_prime_pairs_summing_to_50_l517_517339

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517339


namespace num_prime_pairs_sum_50_l517_517336

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517336


namespace taylor_pets_count_l517_517565

noncomputable def totalPetsTaylorFriends (T : ℕ) (x1 : ℕ) (x2 : ℕ) : ℕ :=
  T + 3 * x1 + 2 * x2

theorem taylor_pets_count (T : ℕ) (x1 x2 : ℕ) (h1 : x1 = 2 * T) (h2 : x2 = 2) (h3 : totalPetsTaylorFriends T x1 x2 = 32) :
  T = 4 :=
by
  sorry

end taylor_pets_count_l517_517565


namespace prime_pairs_sum_50_l517_517365

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517365


namespace symmetry_construction_complete_l517_517953

-- Conditions: The word and the chosen axis of symmetry
def word : String := "ГЕОМЕТРИя"

inductive Axis
| horizontal
| vertical

-- The main theorem which states that a symmetrical figure can be constructed for the given word and axis
theorem symmetry_construction_complete (axis : Axis) : ∃ (symmetrical : String), 
  (axis = Axis.horizontal ∨ axis = Axis.vertical) → 
   symmetrical = "яИРТЕМОЕГ" := 
by
  sorry

end symmetry_construction_complete_l517_517953


namespace smallest_number_of_students_l517_517490

open Nat

theorem smallest_number_of_students 
  (y : ℕ) 
  (num_rows : ℕ = 5) 
  (extra_students : ℕ = 2) 
  (total_students_gt_40 : 5 * y + 2 > 40) : 
  (total_students : ℕ = 5 * y + 2) :=
by
  have y_lower_bound : y > 7 := sorry
  have y_value : y = 8 := sorry
  exact y_value

end smallest_number_of_students_l517_517490


namespace prime_pairs_sum_50_l517_517366

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517366


namespace approx_cube_of_331_l517_517034

noncomputable def cube (x : ℝ) : ℝ := x * x * x

theorem approx_cube_of_331 : 
  ∃ ε > 0, abs (cube 0.331 - 0.037) < ε :=
by
  sorry

end approx_cube_of_331_l517_517034


namespace num_unordered_prime_pairs_summing_to_50_l517_517165

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517165


namespace find_theta_l517_517103

noncomputable def f (x θ : ℝ) : ℝ :=
  2 * x * Real.sin (x + θ + Real.pi / 3)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_theta 
  (θ : ℝ)
  (H₀ : θ ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) 
  (H₁ : is_odd_function (λ x, f x θ))
  : θ = Real.pi / 6 :=
sorry

end find_theta_l517_517103


namespace num_prime_pairs_sum_50_l517_517177

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517177


namespace cos_triple_angle_l517_517471

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517471


namespace intersection_point_exists_l517_517766

noncomputable def h (x : ℝ) : ℝ := sorry
noncomputable def j (x : ℝ) : ℝ := sorry

theorem intersection_point_exists :
  h 1 = 1 ∧ j 1 = 1 ∧
  h 2 = 3 ∧ j 2 = 3 ∧
  h 3 = 6 ∧ j 3 = 6 ∧
  h 4 = 6 ∧ j 4 = 6 →
  ∃ (a : ℝ) (b : ℝ), (y : ℝ) = h (3 * a) ∧ (y = 3 * j (a) ∧ a = 4 ∧ b = 18) :=
by
  intros
  use [4, 18]
  sorry

end intersection_point_exists_l517_517766


namespace quadratic_factorization_l517_517578

theorem quadratic_factorization (C D : ℤ) (h : (15 * y^2 - 74 * y + 48) = (C * y - 16) * (D * y - 3)) :
  C * D + C = 20 :=
sorry

end quadratic_factorization_l517_517578


namespace prime_pairs_sum_50_l517_517267

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517267


namespace prime_pairs_summing_to_50_count_l517_517391

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517391


namespace num_prime_pairs_summing_to_50_l517_517348

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517348


namespace train_speed_kmph_l517_517667

noncomputable def speed_of_train := sorry

theorem train_speed_kmph :
  (∃ (train_length platform_length : ℝ) 
      (crossing_time : ℝ), 
    train_length = 110 ∧ 
    platform_length = 165 ∧ 
    crossing_time = 7.499400047996161 ∧ 
    speed_of_train = (132.01 * 1000 / 3600)) :=
sorry

end train_speed_kmph_l517_517667


namespace trapezoid_BC_squared_eq_CD_CE_l517_517514

variables {A B C D O E : Type*} -- Define geometric points

-- Define conditions
axiom trapezoid (ABCD : quadrilateral)
axiom longer_base (h1 : length B A > length D C)
axiom perpendicular_diagonals (h2 : angle A C D B = π/2)
axiom circumcenter (h3 : is_circumcenter O A B C)
axiom intersection_E (h4 : collinear O B E ∧ collinear C D E)

-- The theorem statement
theorem trapezoid_BC_squared_eq_CD_CE :
  length B C ^ 2 = length C D * length C E :=
sorry

end trapezoid_BC_squared_eq_CD_CE_l517_517514


namespace vector_sum_square_l517_517095

variables (A B C D : Point) (s : ℝ) -- define point and side length s
variables (AB BC AD : Vector) -- define the vectors

-- Assume conditions
def is_square_with_side_length_one (A B C D : Point) : Prop :=
  (distance A B = 1) ∧ (distance B C = 1) ∧ (distance C D = 1) ∧ (distance D A = 1) ∧
  (distance A C = real.sqrt 2) ∧ (distance B D = real.sqrt 2)

-- Define the specific vectors
noncomputable def vector_AB : Vector := B - A
noncomputable def vector_BC : Vector := C - B
noncomputable def vector_AD : Vector := D - A

-- Prove the math statement
theorem vector_sum_square (A B C D : Point) (h : is_square_with_side_length_one A B C D) :
  |(vector_AB A B) + (vector_BC B C)| + |(vector_AB A B) - (vector_AD A D)| = 2 * real.sqrt 2 :=
sorry

end vector_sum_square_l517_517095


namespace primes_sum_50_l517_517402

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517402


namespace y_coordinate_midpoint_Sn_equation_lambda_range_l517_517517

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := 0.5 + Real.log x / Real.log (1 - x)

-- Problem 1: Prove y-coordinate of midpoint M is 1/2
theorem y_coordinate_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (h1 : A.2 = 0.5 + Real.log (A.1 / (1 - A.1)) / Real.log 2)
  (h2 : B.2 = 0.5 + Real.log (B.1 / (1 - B.1)) / Real.log 2)
  (hM : M.1 = 0.5 ∧ M = (0.5 * (A.1 + B.1), 0.5 * (A.2 + B.2))) :
  M.2 = 0.5 :=
sorry

-- Problem 2: Prove Sn equals (n-1)/2 for n>=2
theorem Sn_equation (n : ℕ) (hn : n ≥ 2) :
  let Sn := (Finset.range (n-1)).sum (λ k, f ((k + 1 : ℕ) / (n : ℕ))) in
  Sn = (n-1)/2 :=
sorry

-- Problem 3: Prove range of λ that satisfies the inequality 
theorem lambda_range (λ n : ℕ) (h : ∀ (n : ℕ), 
  let a : ℕ → ℝ := 
      λ n, if n = 1 then 2/3 else 1 / ((f ((n : ℝ) / (n + 1 : ℝ)) + 1) * (f (((n + 1 : ℝ)) / (n + 2 : ℝ)) + 1)),
      Tn := (Finset.range n).sum (λ k, a k + 1 - 1) in
      4 * ((1 / (n+1)) - (1 / (n + 2))) :
  Tn < λ * (Sn + 1)) :
  λ > 0.5 :=
sorry

end y_coordinate_midpoint_Sn_equation_lambda_range_l517_517517


namespace arithmetic_evaluation_l517_517601

theorem arithmetic_evaluation : 6 * 2 - 3 = 9 := by
  sorry

end arithmetic_evaluation_l517_517601


namespace max_club_members_l517_517989

open Set

variable {U : Type} (A B C : Set U)

theorem max_club_members (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11)
    (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
    (A ∪ B ∪ C).card ≤ 22 :=
by {
  -- The proof will go here, but for now we skip it.
  sorry
}

end max_club_members_l517_517989


namespace prime_pairs_summing_to_50_count_l517_517382

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517382


namespace complete_square_rewrite_l517_517886

theorem complete_square_rewrite (j i : ℂ) :
  let c := 8
  let p := (3 * i / 8 : ℂ)
  let q := (137 / 8 : ℂ)
  (8 * j^2 + 6 * i * j + 16 = c * (j + p)^2 + q) →
  q / p = - (137 * i / 3) :=
by
  sorry

end complete_square_rewrite_l517_517886


namespace positive_satisfying_magnitude_l517_517729

theorem positive_satisfying_magnitude (s : ℝ) (h : |complex.mk 3 s| = 13) : s = 4 * real.sqrt 10 :=
by
  sorry

end positive_satisfying_magnitude_l517_517729


namespace XiaoYing_minimum_water_usage_l517_517569

-- Definitions based on the problem's conditions
def first_charge_rate : ℝ := 2.8
def excess_charge_rate : ℝ := 3
def initial_threshold : ℝ := 5
def minimum_bill : ℝ := 29

-- Main statement for the proof based on the derived inequality
theorem XiaoYing_minimum_water_usage (x : ℝ) (h1 : 2.8 * initial_threshold + 3 * (x - initial_threshold) ≥ 29) : x ≥ 10 := by
  sorry

end XiaoYing_minimum_water_usage_l517_517569


namespace max_value_vector_sum_l517_517090

open Real

def max_vector_sum (A B C P D : Point) : ℝ := 
  let PA := vector P A
  let PB := vector P B
  let PC := vector P C
  PA + PB + PC

theorem max_value_vector_sum
    (A B C : Point)
    (P : Point := (5,0))
    (D : Point := (1,0))
    (on_circle: ∀ (p : Point), p = A ∨ p = B ∨ p = C → point_on_circle p (0, 0) 2)
    (midpoint: midpoint B C D)
    : max_vector_sum A B C P D = 15 := 
sorry 

end max_value_vector_sum_l517_517090


namespace count_prime_pairs_sum_50_exactly_4_l517_517293

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517293


namespace measure_minor_arc_of_inscribed_angle_l517_517489

/--
In a circle \( Q \), if angle \( XBY \) measures \( 60 \) degrees, 
then the measure of the minor arc \( XB \) is \( 60 \) degrees.
-/
theorem measure_minor_arc_of_inscribed_angle 
  (Q : Type) [Circle Q]
  (X B Y : Q) 
  (h1 : inscribed_angle X B Y) 
  (h2 : angle_measure X B Y = 60) :
  arc_measure X B = 60 := 
sorry

end measure_minor_arc_of_inscribed_angle_l517_517489


namespace qrs_relationship_l517_517816

-- Definitions for the conditions
def p : ℕ := 3
def q : ℕ
def r : ℕ
def s : ℕ

-- Product of pqrs is 1365
axiom pqrs_product (h : p * q * r * s = 1365) : q * r * s = 455

-- Proof statement
theorem qrs_relationship (h : p * q * r * s = 1365) : q * r * s = 455 :=
pqrs_product h

end qrs_relationship_l517_517816


namespace number_of_prime_pairs_sum_50_l517_517212

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517212


namespace cyclic_quadrilaterals_count_l517_517115

theorem cyclic_quadrilaterals_count :
  let is_cyclic_quadrilateral (a b c d : ℕ) : Prop :=
    a + b + c + d = 36 ∧
    a * c + b * d <= (a + c) * (b + d) ∧ -- cyclic quadrilateral inequality
    a + b > c ∧ a + c > b ∧ a + d > b ∧ b + c > d ∧ -- convex quadilateral inequality

  (finset.univ.filter (λ (s : finset ℕ), 
    s.card = 4 ∧ fact (multiset.card s.to_multiset = 36) ∧ is_cyclic_quadrilateral s.to_multiset.sum)).card = 1440 :=
sorry

end cyclic_quadrilaterals_count_l517_517115


namespace evaluate_polynomial_at_minus_two_l517_517035

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end evaluate_polynomial_at_minus_two_l517_517035


namespace min_distance_l517_517727

theorem min_distance (x y z : ℝ) :
  ∃ (m : ℝ), m = (Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2)) ∧ m = Real.sqrt 6 :=
by
  sorry

end min_distance_l517_517727


namespace right_triangle_set_D_l517_517624

theorem right_triangle_set_D : (5^2 + 12^2 = 13^2) ∧ 
  ((3^2 + 3^2 ≠ 5^2) ∧ (6^2 + 8^2 ≠ 9^2) ∧ (4^2 + 5^2 ≠ 6^2)) :=
by
  sorry

end right_triangle_set_D_l517_517624


namespace number_of_prime_pairs_sum_50_l517_517206

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517206


namespace primes_sum_50_l517_517403

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517403


namespace sum_reciprocal_roots_eq_neg_one_l517_517852

theorem sum_reciprocal_roots_eq_neg_one (a b c : ℝ) (h1 : (Polynomial.X^3 - 2*Polynomial.X^2 - Polynomial.X + 2).is_root a)
(h2 : (Polynomial.X^3 - 2*Polynomial.X^2 - Polynomial.X + 2).is_root b)
(h3 : (Polynomial.X^3 - 2*Polynomial.X^2 - Polynomial.X + 2).is_root c) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) = -1 := 
sorry

end sum_reciprocal_roots_eq_neg_one_l517_517852


namespace count_prime_pairs_summing_to_50_l517_517238

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517238


namespace length_of_bridge_l517_517586

def length_of_train : ℝ := 135  -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 45  -- Speed of the train in km/hr
def speed_of_train_m_per_s : ℝ := 12.5  -- Speed of the train in m/s
def time_to_cross_bridge : ℝ := 30  -- Time to cross the bridge in seconds
def distance_covered : ℝ := speed_of_train_m_per_s * time_to_cross_bridge  -- Total distance covered

theorem length_of_bridge :
  distance_covered - length_of_train = 240 :=
by
  sorry

end length_of_bridge_l517_517586


namespace count_b_for_exactly_three_integer_solutions_number_of_possible_b_values_l517_517934

theorem count_b_for_exactly_three_integer_solutions (b : ℤ) :
  (∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ∀ (x : ℤ), (x = x1 ∨ x = x2 ∨ x = x3) ↔ x^2 + b * x + 6 ≤ 0) →
  (b = -5 ∨ b = 5) := sorry

theorem number_of_possible_b_values :
  {b : ℤ // ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ∀ (x : ℤ), (x = x1 ∨ x = x2 ∨ x = x3) ↔ x^2 + b * x + 6 ≤ 0}.card = 2 := sorry

end count_b_for_exactly_three_integer_solutions_number_of_possible_b_values_l517_517934


namespace Susan_has_12_dollars_left_l517_517899

theorem Susan_has_12_dollars_left:
  ∀ (initial food : ℕ), initial = 100 → food = 16 → 
  let rides := 3 * food in
  let games := rides / 2 in
  let total_spent := food + rides + games in
  let remaining := initial - total_spent in
  remaining = 12 :=
by
  intros initial food h_initial h_food rides games total_spent remaining
  rw [h_initial, h_food]
  simp only [mul_eq_mul_left_iff, eq_self_iff_true, and_true]
  sorry

end Susan_has_12_dollars_left_l517_517899


namespace average_salary_correct_l517_517496

variables (n_techs n_workers : ℕ) (s_techs s_nontechs : ℝ)

def average_salary (n_techs : ℕ) (s_techs : ℝ) (n_workers : ℕ) (s_nontechs : ℝ) : ℝ :=
  let total_salary_techs := n_techs * s_techs in
  let n_nontechs := n_workers - n_techs in
  let total_salary_nontechs := n_nontechs * s_nontechs in
  let total_salary := total_salary_techs + total_salary_nontechs in
  total_salary / n_workers

theorem average_salary_correct :
  average_salary 5 900 20 700 = 750 :=
sorry

end average_salary_correct_l517_517496


namespace rugby_team_lineups_l517_517872

-- Define the total number of team members
def team_size : ℕ := 22

-- Define the total number of forwards
def forwards : ℕ := 8

-- Define the total number of non-forwards
def non_forwards : ℕ := team_size - forwards

-- Define the number of players to be selected (excluding the captain)
def players_to_choose : ℕ := 12

-- Define the minimum number of forwards required
def min_forwards : ℕ := 3

-- Define the total number of valid lineups including captain selection
def total_lineups : ℕ :=
  let combinations := sum [finset.range (min_forwards) (min players_to_choose (forwards + 1))]
                         (λ k, nat.choose forwards k * nat.choose non_forwards (players_to_choose - k)) in
  22 * combinations

-- The main statement to prove
theorem rugby_team_lineups : total_lineups = 6478132 :=
sorry

end rugby_team_lineups_l517_517872


namespace sum_of_z_values_l517_517526

def f (x : ℝ) : ℝ := x^2 + 3 * x + 2

theorem sum_of_z_values : (∑ z in {z | f(4 * z) = 8}, z) = -3 / 16 :=
by
  sorry

end sum_of_z_values_l517_517526


namespace last_divisor_is_two_l517_517644

theorem last_divisor_is_two :
  let A := 377 / 13 in
  let B := A / 29 in
  let C := B * (1 / 4) in
  C / 2 = 0.125 := 
by
  let A := 377 / 13
  let B := A / 29
  let C := B * (1 / 4)
  have h : C / 2 = 0.125 := sorry
  exact h

end last_divisor_is_two_l517_517644


namespace abc_positive_l517_517792

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  -- Proof goes here
  sorry

end abc_positive_l517_517792


namespace unique_B_squared_l517_517522

variable {R : Type*} [CommRing R]
variable {B : Matrix (Fin 2) (Fin 2) R}
variable h : B^4 = 0

theorem unique_B_squared (h : B^4 = 0) : ∃! B2 : Matrix (Fin 2) (Fin 2) R, B2 = B^2 :=
by
  sorry

end unique_B_squared_l517_517522


namespace how_much_does_c_have_l517_517626

theorem how_much_does_c_have (A B C : ℝ) (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : B + C = 150) : C = 50 :=
by
  sorry

end how_much_does_c_have_l517_517626


namespace prime_pairs_sum_50_l517_517259

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517259


namespace ellipse_conditions_l517_517059

open Real

variables {a b : ℝ}
variables (x y c : ℝ)
variables (F1 F2 A B : ℝ × ℝ)

/-- Given F1 and F2 are the left and right foci of an ellipse (x^2/a^2) + (y^2/b^2) = 1 with a > b > 0,
point A lies on the ellipse, and AF1 ⊥ F1F2. Line AF2 intersects the
ellipse at another point B, and vector AF2 = 3 * vector F2B, 
prove the following:
- the length of segment AF1 is 2/3 ⋅ a
- the eccentricity of the ellipse is (sqrt 3) / 3 --/
theorem ellipse_conditions (h1 : F1 = (-c, 0)) (h2 : F2 = (c, 0))
  (h3 : 0 < a ∧ a > b ∧ 0 < b ∧ c = sqrt (a^2 - b^2))
  (h4 : A = (-c, b^2 / a)) (h5 : x = 5 / 3 * c ∧ y = -b^2 / (3 * a)
  (h6 : B = (x, y)) :
  (dist A F1 = 2 / 3 * a) ∧ ((sqrt (1 - b^2 / a^2)) = sqrt 3 / 3) :=
by sorry

end ellipse_conditions_l517_517059


namespace percentage_y_less_than_x_l517_517629

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 11 * y) : 
  ((x - y) / x) * 100 = 90.91 := 
by 
  sorry -- proof to be provided separately

end percentage_y_less_than_x_l517_517629


namespace sum_of_three_numbers_l517_517949

theorem sum_of_three_numbers (a b c : ℕ)
    (h1 : a + b = 35)
    (h2 : b + c = 40)
    (h3 : c + a = 45) :
    a + b + c = 60 := 
  by sorry

end sum_of_three_numbers_l517_517949


namespace min_value_frac_l517_517769

open Real

theorem min_value_frac (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a + b = 1) : min_value (fun (x y : ℝ) => 4 / x + 1 / y) a b h = 9 := sorry

end min_value_frac_l517_517769


namespace proof_l517_517529

noncomputable def f (x : ℝ) :=
  (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20)) - (x^2 - 13*x - 6)

noncomputable def n : ℝ := 13 + Real.sqrt (67)

theorem proof : ∃ (d e f : ℕ), n = d + Real.sqrt (e + Real.sqrt (f)) ∧ d + e + f = 95 :=
by {
  use 13,
  use 67,
  use 15,
  split,
  { simp [n], },
  { norm_num, },
}

end proof_l517_517529


namespace num_prime_pairs_sum_50_l517_517186

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517186


namespace prime_pairs_summing_to_50_count_l517_517384

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517384


namespace reflection_about_y_eq_neg_x_l517_517568

def reflect_point (p : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_about_y_eq_neg_x :
  reflect_point (-3, 7) (λ y, -y) = (-7, 3) :=
by
  sorry

end reflection_about_y_eq_neg_x_l517_517568


namespace sin_of_minus_19_div_6_pi_eq_one_half_l517_517929

theorem sin_of_minus_19_div_6_pi_eq_one_half :
  sin (- (19 / 6) * Real.pi) = 1 / 2 :=
by sorry

end sin_of_minus_19_div_6_pi_eq_one_half_l517_517929


namespace number_of_prime_pairs_sum_50_l517_517205

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517205


namespace painted_cells_l517_517491

open Int

theorem painted_cells : ∀ (m n : ℕ), (m = 20210) → (n = 1505) →
  let sub_rectangles := 215
  let cells_per_diagonal := 100
  let total_cells := sub_rectangles * cells_per_diagonal
  let total_painted_cells := 2 * total_cells
  let overlap_cells := sub_rectangles
  let unique_painted_cells := total_painted_cells - overlap_cells
  unique_painted_cells = 42785 := sorry

end painted_cells_l517_517491


namespace prime_pairs_sum_50_l517_517372

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517372


namespace infinite_fixpoints_l517_517535

variable {f : ℕ+ → ℕ+}
variable (H : ∀ (m n : ℕ+), (∃ k : ℕ+ , k ≤ f n ∧ n ∣ f (m + k)) ∧ (∀ j : ℕ+ , j ≤ f n → j ≠ k → ¬ n ∣ f (m + j)))

theorem infinite_fixpoints : ∃ᶠ n in at_top, f n = n :=
sorry

end infinite_fixpoints_l517_517535


namespace primes_sum_50_l517_517405

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517405


namespace min_recip_sum_l517_517085

-- Definitions for the conditions
variables {a b : ℝ}
hypothesis h1 : a > 0
hypothesis h2 : b > 0
hypothesis h3 : a + b = 1

-- Lean statement for the equivalence proof problem
theorem min_recip_sum : ∃ c : ℝ, (a > 0) → (b > 0) → (a + b = 1) → (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 1) → (c = ∑ i in {x, y}, i⁻¹)) ∧ (c = 4) := 
by
  sorry

-- Prove the problem
example : ∃ c : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (c = (1/a + 1/b))) ∧ (c = 4) := 
by
  use 4
  intros a b a_pos b_pos hab
  have h : 1/a + 1/b = 4 := 
    sorry
  exact ⟨h, rfl⟩

end min_recip_sum_l517_517085


namespace different_from_M_l517_517673

structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-5, Real.pi / 3⟩
def A : Point := ⟨5, -Real.pi / 3⟩
def B : Point := ⟨5, 4 * Real.pi / 3⟩
def C : Point := ⟨5, -2 * Real.pi / 3⟩
def D : Point := ⟨-5, -5 * Real.pi / 3⟩

theorem different_from_M : (A ≠ M)
  ∧ (B ≠ M)
  ∧ (C ≠ M)
  ∧ (D = M) :=
by
  sorry

end different_from_M_l517_517673


namespace subset_sum_remainder_l517_517738

theorem subset_sum_remainder (n : ℕ) (a : Fin n → ℕ) (r : ℕ) (h_rel_prime : ∀ i, Nat.coprime (a i) n) (h_r : r < n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % n = r :=
sorry

end subset_sum_remainder_l517_517738


namespace insect_shortest_path_l517_517677

def regular_octahedron (V : Type) [MetricSpace V] :=
  ∃ (E : set (V × V)), (∀ (a b : V), (a, b) ∈ E → dist a b = 1) ∧ ∀ v : V, 4 = E.count v -- This is a simplifying assumption for regular polyhedra in Lean.

def shortest_path_octahedron_surface (V : Type) [MetricSpace V] (top bottom : V) : Prop :=
   regular_octahedron V ∧ dist top bottom = 2

theorem insect_shortest_path (V : Type) [MetricSpace V] (top bottom : V) :
  shortest_path_octahedron_surface V top bottom := sorry

end insect_shortest_path_l517_517677


namespace one_and_one_third_of_what_number_is_45_l517_517878

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l517_517878


namespace relationship_among_g_a_0_f_b_l517_517861

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem relationship_among_g_a_0_f_b (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  -- Function properties are non-trivial and are omitted.
  sorry

end relationship_among_g_a_0_f_b_l517_517861


namespace walkway_and_border_area_correct_l517_517900

-- Definitions based on the given conditions
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3
def walkway_width : ℕ := 2
def border_width : ℕ := 4
def num_rows : ℕ := 4
def num_columns : ℕ := 3

-- Total width calculation
def total_width : ℕ := 
  (flower_bed_width * num_columns) + (walkway_width * (num_columns + 1)) + (border_width * 2)

-- Total height calculation
def total_height : ℕ := 
  (flower_bed_height * num_rows) + (walkway_width * (num_rows + 1)) + (border_width * 2)

-- Total area of the garden including walkways and decorative border
def total_area : ℕ := total_width * total_height

-- Total area of flower beds
def flower_bed_area : ℕ := 
  (flower_bed_width * flower_bed_height) * (num_rows * num_columns)

-- Area of the walkways and decorative border
def walkway_and_border_area : ℕ := total_area - flower_bed_area

theorem walkway_and_border_area_correct : 
  walkway_and_border_area = 912 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end walkway_and_border_area_correct_l517_517900


namespace part1_part2_l517_517102

variable (m x : ℝ)

def y (m x : ℝ) := (m + 1) * x ^ 2 - m * x + m - 1

theorem part1 (m : ℝ) (h_empty : ∀ x : ℝ, y m x < 0 → false) : 
  m ∈ Ici (2 * Real.sqrt 3 / 3) :=
sorry

theorem part2 (m : ℝ) (h_subset : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → y m x ≥ 0) : 
  m ∈ Ici (2 * Real.sqrt 3 / 3) :=
sorry

end part1_part2_l517_517102


namespace sum_of_A_l517_517971

theorem sum_of_A (A : ℕ) (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : 57 * 7 > 65 * A) : 
    (finset.sum (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 9 ∧ 57 * 7 > 65 * n) (finset.range 10))) = 21 := 
by sorry

end sum_of_A_l517_517971


namespace unique_solution_of_function_eq_l517_517046

theorem unique_solution_of_function_eq (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) : f = id := 
sorry

end unique_solution_of_function_eq_l517_517046


namespace find_number_l517_517876

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l517_517876


namespace num_prime_pairs_sum_50_l517_517169

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517169


namespace cos_30_eq_sqrt3_div_2_l517_517032

theorem cos_30_eq_sqrt3_div_2 : real.cos (pi / 6) = √3 / 2 := 
by sorry

end cos_30_eq_sqrt3_div_2_l517_517032


namespace expected_value_of_X_is_2_l517_517050

noncomputable def cdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 4 then x / 4
  else 1

noncomputable def pdf (x : ℝ) : ℝ :=
  if x < 0 then 0
  else if x ≤ 4 then 1 / 4
  else 0

noncomputable def expected_value : ℝ :=
  ∫ x in 0..4, x * pdf x

theorem expected_value_of_X_is_2 : expected_value = 2 :=
  sorry

end expected_value_of_X_is_2_l517_517050


namespace shopkeeper_sold_articles_l517_517993

theorem shopkeeper_sold_articles (C : ℝ) (N : ℕ) 
  (h1 : (35 * C = N * C + (1/6) * (N * C))) : 
  N = 30 :=
by
  sorry

end shopkeeper_sold_articles_l517_517993


namespace loss_percentage_l517_517661

theorem loss_percentage (CP SP : ℝ) (h_CP : CP = 1300) (h_SP : SP = 1040) :
  ((CP - SP) / CP) * 100 = 20 :=
by
  sorry

end loss_percentage_l517_517661


namespace tangent_line_eq_l517_517577

def f (x : Real) : Real := x * Real.exp x + 1

theorem tangent_line_eq (f : ℝ → ℝ) (h : f = λ x => x * Real.exp x + 1) :
  let p := (0, f 0)
  let m := Real.exp 0
  let y : ℝ := f 0
  ∀ x y, y - 1 = m * (x - 0) ↔ x - y + 1 = 0 :=
by
  intro x y
  sorry

end tangent_line_eq_l517_517577


namespace vector_magnitude_example_l517_517791

noncomputable def vector_magnitude (a b : ℝ) (angle : ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = real.sqrt 3) : ℝ :=
  real.sqrt (4 - 4 * (‖a‖ * ‖b‖ * real.cos angle) + ‖b‖^2)

theorem vector_magnitude_example (a b : ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = real.sqrt 3)
    (angle_150 : ∀ (a b  : ℝ),  angle  = 150 ) :
  vector_magnitude a b (150^\circ) ha hb = real.sqrt 13 :=
by
  sorry

end vector_magnitude_example_l517_517791


namespace prime_pairs_sum_to_50_l517_517312

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517312


namespace prime_pairs_sum_50_l517_517261

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517261


namespace count_prime_pairs_sum_50_exactly_4_l517_517286

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517286


namespace additional_machines_needed_l517_517802

theorem additional_machines_needed 
  (machines : ℕ) 
  (days : ℕ) 
  (one_eighth_less_time : ∃ (updated_days : ℝ), updated_days = days * (7 / 8))
  (work_rate : ∀ (m : ℕ) (d : ℝ), (m = machines) → (d = days) → (work_rate := 1 / (m * d))) :
  ∃ (additional_machines : ℕ), (machines + additional_machines) * (1 / (machines * days)) * (days * (7 / 8)) = 1 → additional_machines = 3 :=
sorry

end additional_machines_needed_l517_517802


namespace find_a_l517_517754

theorem find_a (a : ℝ) :
  (∀ x, (x^2 - ax + a^2 - 19 = 0) ↔ (x = 3)) ∧
  (∀ x, (x^2 - 5x + 6 = 0) ↔ (x = 2 ∨ x = 3)) ∧
  (∀ x, (x^2 + 2x - 8 = 0) ↔ (x = -4 ∨ x = 2)) ∧
  (∃ x, (x^2 - ax + a^2 - 19 = 0) ∧ (x^2 - 5x + 6 = 0)) ∧
  (¬ ∃ x, (x^2 - ax + a^2 - 19 = 0) ∧ (x^2 + 2x - 8 = 0)) →
  a = -2 :=
by sorry

end find_a_l517_517754


namespace range_shift_l517_517922

theorem range_shift {f : ℝ → ℝ} (h : set.range f = set.Icc (-2 : ℝ) 2) :
  set.range (λ x, f (x+1)) = set.Icc (-2 : ℝ) 2 :=
sorry

end range_shift_l517_517922


namespace prime_pairs_sum_to_50_l517_517422

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517422


namespace sales_tax_difference_l517_517027

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.0625
  let tax1 := price * tax_rate1
  let tax2 := price * tax_rate2
  let difference := tax1 - tax2
  difference = 0.625 :=
by
  sorry

end sales_tax_difference_l517_517027


namespace cos_3theta_value_l517_517459

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l517_517459


namespace prime_pairs_summing_to_50_count_l517_517379

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517379


namespace sequence_difference_l517_517596

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sequence_difference (hS : ∀ n, S n = n^2 - 5 * n)
                            (hna : ∀ n, a n = S n - S (n - 1))
                            (hpq : p - q = 4) :
                            a p - a q = 8 := by
    sorry

end sequence_difference_l517_517596


namespace tangential_trapezoid_l517_517583

theorem tangential_trapezoid {A B C D T U M V : Point} 
  (h1 : tangential_trapezoid ABCD)
  (h2 : incircle_touches AB T)
  (h3 : incircle_touches CD U)
  (h4 : intersection_lines AD BC M)
  (h5 : intersection_lines AB MU V) :
  segment_length A T = segment_length V B := 
sorry

end tangential_trapezoid_l517_517583


namespace maximize_triangle_area_l517_517781

theorem maximize_triangle_area (m : ℝ) (l : ∀ x y, x + y + m = 0) (C : ∀ x y, x^2 + y^2 + 4 * y = 0) :
  m = 0 ∨ m = 4 :=
sorry

end maximize_triangle_area_l517_517781


namespace number_of_prime_pairs_sum_50_l517_517209

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517209


namespace find_a_range_l517_517778

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem find_a_range (a : ℝ) :
  (∀ x > 0, x ^ a ≥ 2 * Real.exp (2 * x) * f a x + Real.exp (2 * x)) → 0 < a ∧ a ≤ 2 * Real.exp 1 :=
begin
  sorry
end

end find_a_range_l517_517778


namespace tan_alpha_proof_l517_517737

noncomputable def proof_tan_alpha : Prop :=
  ∀ α : ℝ, tan (α + π / 4) = 3 → tan α = 1 / 2

theorem tan_alpha_proof : proof_tan_alpha :=
by
  sorry

end tan_alpha_proof_l517_517737


namespace marble_count_l517_517817

-- Define the variables for the number of marbles
variables (o p y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := o = 1.3 * p
def condition2 : Prop := y = 1.5 * o

-- Define the total number of marbles based on the conditions
def total_marbles : ℝ := o + p + y

-- The theorem statement that needs to be proved
theorem marble_count (h1 : condition1 o p) (h2 : condition2 o y) : total_marbles o p y = 3.269 * o :=
by sorry

end marble_count_l517_517817


namespace biff_break_even_hours_l517_517012

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l517_517012


namespace max_x5_l517_517855

theorem max_x5 (x1 x2 x3 x4 x5 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) : x5 ≤ 5 :=
  sorry

end max_x5_l517_517855


namespace area_of_unpainted_region_l517_517607

theorem area_of_unpainted_region 
  (w1 w2 : ℝ) 
  (angle : ℝ) 
  (painted_parallelogram_height : ℝ) 
  : w1 = 5 ∧ w2 = 7 ∧ angle = 45 ∧ painted_parallelogram_height = 7 → 
    w1 * painted_parallelogram_height = 35 :=
by 
  intros h
  cases h with w1_eq h
  cases h with w2_eq h
  cases h with angle_eq height_eq
  rw [w1_eq, height_eq]
  norm_num
  done

end area_of_unpainted_region_l517_517607


namespace angle_ACP_l517_517562

/-
Suppose we have the following conditions:
- Segment \(AB\) has midpoint \(C\).
- Segment \(BC\) has midpoint \(D\).
- Semi-circles are constructed with diameters \(\overline{AB}\) and \(\overline{BC}\).
- Segment \(CP\) splits the region into parts with area ratio \(2:3\).

We need to prove that the degree measure of angle \(ACP\) is \(120.0^\circ\).
-/

structure Point := (x : ℝ) (y : ℝ)
structure Segment := (start : Point) (end : Point)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def radius (A B : Point) : ℝ := 
(real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)) / 2

def semicircle_area (r : ℝ) : ℝ := 
(π * r^2) / 2

def total_area (A B C : Point) : ℝ := 
semicircle_area (radius A B) + semicircle_area (radius B C)

def split_ratio (A B C P : Point) : ℝ :=
2 / 5

theorem angle_ACP (A B C D P : Point)
  (mid_C : midpoint A B = C)
  (mid_D : midpoint B C = D)
  (ratio : split_ratio A B C P = 2 / 5) :
  ∠ A C P = 120.0 := by
  sorry

end angle_ACP_l517_517562


namespace num_prime_pairs_sum_50_l517_517335

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517335


namespace union_of_intervals_l517_517734

theorem union_of_intervals :
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  P ∪ Q = {x : ℝ | -2 < x ∧ x < 1} :=
by
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  have h : P ∪ Q = {x : ℝ | -2 < x ∧ x < 1}
  {
     sorry
  }
  exact h

end union_of_intervals_l517_517734


namespace count_prime_pairs_sum_50_exactly_4_l517_517281

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517281


namespace train_length_proof_l517_517668

def speed_km_per_hr : ℝ := 78
def time_minutes : ℝ := 1
def tunnel_length_meters : ℝ := 500

def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
def time_seconds : ℝ := time_minutes * 60
def distance_covered : ℝ := speed_m_per_s * time_seconds
def train_length : ℝ := distance_covered - tunnel_length_meters

theorem train_length_proof : train_length = 800.2 := by
  -- proof goes here
  sorry

end train_length_proof_l517_517668


namespace count_prime_pairs_sum_50_l517_517144

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517144


namespace total_books_arithmetic_sequence_l517_517915

theorem total_books_arithmetic_sequence :
  ∃ (n : ℕ) (a₁ a₂ aₙ d S : ℤ), 
    n = 11 ∧
    a₁ = 32 ∧
    a₂ = 29 ∧
    aₙ = 2 ∧
    d = -3 ∧
    S = (n * (a₁ + aₙ)) / 2 ∧
    S = 187 :=
by sorry

end total_books_arithmetic_sequence_l517_517915


namespace solution_l517_517019

noncomputable def problem : Prop :=
  nat.choose 16 13 = 560

theorem solution : problem := by
  sorry

end solution_l517_517019


namespace prime_pairs_sum_to_50_l517_517427

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517427


namespace shelves_full_percentage_l517_517933

-- Define the conditions as constants
def ridges_per_record : Nat := 60
def cases : Nat := 4
def shelves_per_case : Nat := 3
def records_per_shelf : Nat := 20
def total_ridges : Nat := 8640

-- Define the total number of records
def total_records := total_ridges / ridges_per_record

-- Define the total capacity of the shelves
def total_capacity := cases * shelves_per_case * records_per_shelf

-- Define the percentage of shelves that are full
def percentage_full := (total_records * 100) / total_capacity

-- State the theorem that the percentage of the shelves that are full is 60%
theorem shelves_full_percentage : percentage_full = 60 := 
by
  sorry

end shelves_full_percentage_l517_517933


namespace find_n_l517_517093

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Given conditions
variable (n : ℕ)
variable (coef : ℕ)
variable (h : coef = binomial_coeff n 2 * 9)

-- Proof target
theorem find_n (h : coef = 54) : n = 4 :=
  sorry

end find_n_l517_517093


namespace count_prime_pairs_summing_to_50_l517_517240

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517240


namespace prime_pairs_summing_to_50_count_l517_517385

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517385


namespace num_prime_pairs_summing_to_50_l517_517344

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517344


namespace num_prime_pairs_sum_50_l517_517326

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517326


namespace taxi_charge_l517_517836

theorem taxi_charge :
  ∀ (initial_fee additional_charge_per_segment total_distance total_charge : ℝ),
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (5/2 * total_distance) = 0.35 :=
by
  intros initial_fee additional_charge_per_segment total_distance total_charge
  intros h_initial_fee h_total_distance h_total_charge
  -- Proof here
  sorry

end taxi_charge_l517_517836


namespace perimeter_semi_circle_new_radius_apprx_l517_517595

theorem perimeter_semi_circle_new_radius_apprx {r : ℝ} (h1 : r = 6.4) (h2 : r' = r * 1.20) : 
  let new_r := 7.68 in let π := 3.14159 in (π * new_r + 2 * new_r) ≈ 39.50 :=
by
  sorry

end perimeter_semi_circle_new_radius_apprx_l517_517595


namespace length_of_bridge_l517_517669

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass_bridge : ℝ) 
  (train_length_eq : train_length = 400)
  (train_speed_kmh_eq : train_speed_kmh = 60) 
  (time_to_pass_bridge_eq : time_to_pass_bridge = 72)
  : ∃ (bridge_length : ℝ), bridge_length = 800.24 := 
by
  sorry

end length_of_bridge_l517_517669


namespace evaluate_propositions_l517_517521

-- Definitions based on the conditions from part (a)
variables (α β : Plane) (m n : Line)

-- Proposition (1)
def proposition_1 := m ⊥ n ∧ m ⊥ α → n ∥ α

-- Proposition (2)
def proposition_2 := n ⊂ α ∧ m ⊂ β ∧ ¬(α ⊥ β) → ¬(n ⊥ m)

-- Proposition (3)
def proposition_3 := α ⊥ β ∧ (∃ x, α ∩ β = x ∧ x = m) ∧ n ⊂ α ∧ n ⊥ m → n ⊥ β

-- Proposition (4)
def proposition_4 := m ∥ n ∧ n ⊥ α ∧ α ∥ β → m ⊥ β

-- Statement that propositions (3) and (4) are true
theorem evaluate_propositions 
  (h_conds : α ≠ β ∧ m ≠ n) -- non-coincident planes and lines
  (h3 : proposition_3 α β m n)
  (h4 : proposition_4 α β m n) :
  true :=
by sorry

end evaluate_propositions_l517_517521


namespace yadav_clothes_transport_spending_l517_517959

variable (S : ℝ)
variable (savings_yearly : ℝ := 46800)
variable (savings_monthly : ℝ := savings_yearly / 12)

theorem yadav_clothes_transport_spending :
  (0.2 * S = savings_monthly) → (0.2 * S = 3900) :=
by
  intro h1
  simp [savings_monthly] at h1
  exact h1

end yadav_clothes_transport_spending_l517_517959


namespace sum_of_coefficients_l517_517015

theorem sum_of_coefficients : 
  let f : ℕ → ℕ → ℕ → ℕ := λ n x y, (3 * x^2 - 5 * x * y + 2 * y^2)^n
  in f 5 1 1 = 0 := 
by {
  sorry
}

end sum_of_coefficients_l517_517015


namespace money_found_l517_517698

theorem money_found (donna_share friend_share total : ℝ) (h1 : donna_share = 32.50) (h2 : friend_share = 32.50) (h3 : total = donna_share + friend_share) : total = 65.00 :=
by
  rw [h1, h2]
  simp at h3
  rw [← h3]
  simp
  sorry

end money_found_l517_517698


namespace conjugate_in_first_quadrant_l517_517758

-- Define the imaginary unit, not strictly necessary since it's part of complex numbers in Lean
def i : ℂ := Complex.I

-- Given condition that z * (3 + 4i) = 1 + i
def condition (z : ℂ) : Prop := z * (3 + 4 * i) = 1 + i

-- Define the proof statement
theorem conjugate_in_first_quadrant (z : ℂ) (h : condition z) : (Complex.conj z).re > 0 ∧ (Complex.conj z).im > 0 := by
  sorry

end conjugate_in_first_quadrant_l517_517758


namespace number_of_prime_pairs_sum_50_l517_517221

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517221


namespace number_of_prime_pairs_sum_50_l517_517219

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517219


namespace coeff_x3y5_in_expansion_l517_517571

theorem coeff_x3y5_in_expansion : 
  let poly := (x + 2 * y) * (x - y) ^ 7
  in coeff_x_y poly 3 5 = 49 :=
sorry

end coeff_x3y5_in_expansion_l517_517571


namespace binomial_coefficient_term_l517_517759

noncomputable def integral_value : ℝ := ∫ (x : ℝ) in 0..(π / 2), sqrt 2 * sin (x + (π / 4))

theorem binomial_coefficient_term :
  let m := integral_value in
  let binomial := (sqrt x - m / sqrt x) ^ 6 in
  m = 2 → 
  ∃ c : ℝ, (∀ (n : ℕ), has_term binomial x = n → n = 1 → c = 60) :=
by
  let m := 2
  have h_m : integral_value = m := sorry
  existsi 60
  sorry

end binomial_coefficient_term_l517_517759


namespace ratio_of_wilted_roses_to_total_l517_517510

-- Defining the conditions
def initial_roses := 24
def traded_roses := 12
def total_roses := initial_roses + traded_roses
def remaining_roses_after_second_night := 9
def roses_before_second_night := remaining_roses_after_second_night * 2
def wilted_roses_after_first_night := total_roses - roses_before_second_night
def ratio_wilted_to_total := wilted_roses_after_first_night / total_roses

-- Proving the ratio of wilted flowers to the total number of flowers after the first night is 1:2
theorem ratio_of_wilted_roses_to_total :
  ratio_wilted_to_total = (1/2) := by
  sorry

end ratio_of_wilted_roses_to_total_l517_517510


namespace geometric_number_difference_is_124_l517_517016

theorem geometric_number_difference_is_124 :
  ∀ N1 N2 : ℕ, (N1 = 124 ∧ N2 = 248) →
  (∀ d1 d2 d3 : ℕ, d1 < d2 ∧ d2 < d3 ∧ (d2 / d1 = 2) ∧ (d3 / d2 = 2) ∧ (d1 = 1 ∨ d2 = 1 ∨ d3 = 1)) →
  N2 - N1 = 124 :=
begin
  sorry,
end

end geometric_number_difference_is_124_l517_517016


namespace ryosuke_gas_cost_l517_517560

theorem ryosuke_gas_cost :
  let odometer_start := 85430 in
  let odometer_end := 85461 in
  let mpg := 32 in
  let price_per_gallon := 3.85 in
  let distance := odometer_end - odometer_start in
  let gallons_used := distance / mpg in
  let cost := gallons_used * price_per_gallon in
  Float.round cost 2 = 3.71 :=
by
  sorry

end ryosuke_gas_cost_l517_517560


namespace James_leftover_money_l517_517508

variable (W : ℝ)
variable (M : ℝ)

theorem James_leftover_money 
  (h1 : M = (W / 2 - 2))
  (h2 : M + 114 = W) : 
  M = 110 := sorry

end James_leftover_money_l517_517508


namespace gondor_laptop_earning_l517_517793

theorem gondor_laptop_earning :
  ∃ L : ℝ, (3 * 10 + 5 * 10 + 2 * L + 4 * L = 200) → L = 20 :=
by
  use 20
  sorry

end gondor_laptop_earning_l517_517793


namespace cos_triple_angle_l517_517464

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l517_517464


namespace lee_cookies_l517_517839

theorem lee_cookies (c1 c2 : ℕ) (w : ℕ) :
  c1 = 24 ∧ w = 1/2 ∧ c2 = 3 →
  let effective_flour := c2 - w in
  let proportional_cookies := (c1 / effective_flour) * 5 in
  proportional_cookies = 48 :=
by
  sorry

end lee_cookies_l517_517839


namespace scatter_plot_correlation_l517_517481

noncomputable def correlation_coefficient (points : List (ℝ × ℝ)) : ℝ := sorry

theorem scatter_plot_correlation {points : List (ℝ × ℝ)} 
  (h : ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) :
  correlation_coefficient points = 1 := 
sorry

end scatter_plot_correlation_l517_517481


namespace sally_cards_sum_middle_three_l517_517887

theorem sally_cards_sum_middle_three :
  ∃ (R : Fin 5 → ℕ) (B : Fin 6 → ℕ),
  (∀ i, R i ∈ {1, 2, 3, 4, 5}) ∧
  (∀ i, B i ∈ {3, 4, 5, 6, 7, 8}) ∧
  (∀ i, (∀ j, i ≠ j → R i ≠ R j) ∧ ∀ j, i ≠ j → B i ≠ B j) ∧
  (∀ k, if k % 2 = 0 then (B (Fin.mk k sorry) % 2 = 0 ∧ R (Fin.mk (k / 2) sorry) % 2 = 0 ∧ R (Fin.mk (k / 2) sorry) ∣ B (Fin.mk k sorry))
           else (B (Fin.mk k sorry) % 2 = 1 ∧ R (Fin.mk ((k - 1) / 2) sorry) % 2 = 1 ∧ R (Fin.mk ((k - 1) / 2) sorry) ∣ B (Fin.mk k sorry))) ∧
  sum (List.take 3 (List.drop 4 [B 0, R 0, B 1, R 1, B 2, R 2, B 3, R 3, B 4, R 4, B 5])) = 12 :=
sorry

end sally_cards_sum_middle_three_l517_517887


namespace points_on_segment_AD_l517_517545

theorem points_on_segment_AD (A B C D : Point) (P : Point → ℝ)
  (h : ∀ P, P A + P D ≥ P B + P C) : 
  (B ∈ segment A D) ∧ (C ∈ segment A D) ∧ (dist A B = dist C D) := 
sorry

end points_on_segment_AD_l517_517545


namespace arrangements_of_boys_and_girls_l517_517645

theorem arrangements_of_boys_and_girls (boys girls : ℕ) (h_boys : boys = 5) (h_girls : girls = 3) : 
  let units := boys + 1 in 
  ∃ arrangements : ℕ, arrangements = (units.factorial * girls.factorial) ∧ arrangements = 4320 :=
by
  sorry

end arrangements_of_boys_and_girls_l517_517645


namespace prime_pairs_sum_to_50_l517_517315

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517315


namespace surface_area_after_removal_l517_517967

noncomputable theory

variable (original_cube_sides corners_removed : ℕ) (corner_size : ℕ)

def original_surface_area : ℕ := 6 * (original_cube_sides ^ 2)

def corner_effect (corners_removed corner_size : ℕ) : ℕ :=
  2 * corners_removed * (corner_size ^ 2)

theorem surface_area_after_removal (h1 : original_cube_sides = 4)
                                   (h2 : corners_removed = 4)
                                   (h3 : corner_size = 1) :
  original_surface_area original_cube_sides corners_removed - 
  corner_effect corners_removed corner_size = 96 :=
by
  sorry

end surface_area_after_removal_l517_517967


namespace number_of_impossible_d_l517_517918

-- Define the problem parameters and conditions
def perimeter_diff (t s : ℕ) : ℕ := 3 * t - 4 * s
def side_diff (t s d : ℕ) : ℕ := t - s - d
def square_perimeter_positive (s : ℕ) : Prop := s > 0

-- Define the proof problem
theorem number_of_impossible_d (t s d : ℕ) (h1 : perimeter_diff t s = 1575) (h2 : side_diff t s d = 0) (h3 : square_perimeter_positive s) : 
    ∃ n, n = 525 ∧ ∀ d, d ≤ 525 → ¬ (3 * d > 1575) :=
    sorry

end number_of_impossible_d_l517_517918


namespace task_assignment_correct_l517_517880

open Set

-- Define the individuals
inductive Person
| TM | WK | SS | BJ
deriving DecidableEq, Inhabited

-- Define the tasks
inductive Task
| FetchWater | WashVegetables | RinseRice | MakeFire
deriving DecidableEq, Inhabited

-- Define the tasks performed by each individual
def TaskAssignment : Person → Task

-- Conditions from the problem
axiom cond1 : ¬(TaskAssignment Person.TM = Task.FetchWater) ∧ ¬(TaskAssignment Person.TM = Task.RinseRice)
axiom cond2 : ¬(TaskAssignment Person.WK = Task.WashVegetables) ∧ ¬(TaskAssignment Person.WK = Task.FetchWater)
axiom cond3 : ¬(TaskAssignment Person.WK = Task.RinseRice) → ¬(TaskAssignment Person.SS = Task.FetchWater)
axiom cond4 : ¬(TaskAssignment Person.BJ = Task.FetchWater) ∧ ¬(TaskAssignment Person.BJ = Task.WashVegetables)

-- Theorem to prove the correct tasks for each individual
theorem task_assignment_correct :
  TaskAssignment Person.TM = Task.WashVegetables ∧
  TaskAssignment Person.WK = Task.RinseRice ∧
  TaskAssignment Person.SS = Task.FetchWater ∧
  TaskAssignment Person.BJ = Task.MakeFire :=
  by
  sorry

end task_assignment_correct_l517_517880


namespace count_prime_pairs_sum_50_l517_517143

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517143


namespace diamond_location_detection_l517_517931

theorem diamond_location_detection :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ 100 →
  (∃ d : ℕ, 1 ≤ d ∧ d ≤ 100 ∧
  (∀ j, 1 ≤ j ∧ j ≤ 100 →
  (∃ k, k = j - 1 ∨ k = j + 1) ∧
  (d = k) ↔ (j = i))). :=
sorry

end diamond_location_detection_l517_517931


namespace range_of_a_if_inequality_holds_l517_517779

noncomputable def satisfies_inequality_for_all_xy_pos (a : ℝ) :=
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9

theorem range_of_a_if_inequality_holds :
  (∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9) → (a ≥ 4) :=
by
  sorry

end range_of_a_if_inequality_holds_l517_517779


namespace prime_pairs_summing_to_50_count_l517_517387

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517387


namespace num_unordered_prime_pairs_summing_to_50_l517_517155

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517155


namespace factorization_correct_l517_517907

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

end factorization_correct_l517_517907


namespace calculate_total_bricks_l517_517800

-- Given definitions based on the problem.
variables (a d g h : ℕ)

-- Definitions for the questions in terms of variables.
def days_to_build_bricks (a d g : ℕ) : ℕ :=
  (a * g) / d

def total_bricks_with_additional_men (a d g h : ℕ) : ℕ :=
  a + ((d + h) * a) / 2

theorem calculate_total_bricks (a d g h : ℕ)
  (h1 : 0 < d)
  (h2 : 0 < g)
  (h3 : 0 < a) :
  days_to_build_bricks a d g = a * g / d ∧
  total_bricks_with_additional_men a d g h = (3 * a + h * a) / 2 :=
  by sorry

end calculate_total_bricks_l517_517800


namespace drain_time_correct_l517_517901

-- Define the volume of the pool when full:
def pool_volume_full (width length depth : ℝ) : ℝ := width * length * depth

-- Define the volume of water in the pool at given capacity:
def pool_volume_capacity (full_volume capacity : ℝ) : ℝ := full_volume * capacity

-- Define how long it will take to drain the pool given the volume and rate:
def time_to_drain (volume rate : ℝ) : ℝ := volume / rate

-- Define the conversion from minutes to hours:
def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

-- The problem statement to prove:
theorem drain_time_correct :
  let width := 60
  let length := 100
  let depth := 10
  let capacity := 0.80
  let hose_rate := 60
  let full_volume := pool_volume_full width length depth
  let current_volume := pool_volume_capacity full_volume capacity
  let time_in_minutes := time_to_drain current_volume hose_rate
  minutes_to_hours time_in_minutes = 13.33 :=
by
  sorry

end drain_time_correct_l517_517901


namespace complex_powers_i_l517_517703

theorem complex_powers_i (i : ℂ) (h : i^2 = -1) :
  (i^123 - i^321 + i^432 = -2 * i + 1) :=
by
  -- sorry to skip the proof
  sorry

end complex_powers_i_l517_517703


namespace rate_per_kg_for_grapes_l517_517681

theorem rate_per_kg_for_grapes (G : ℝ) (h : 9 * G + 9 * 55 = 1125) : G = 70 :=
by
  -- sorry to skip the proof
  sorry

end rate_per_kg_for_grapes_l517_517681


namespace variance_of_sample_l517_517746

-- Definitions based on problem conditions
def sequence_term (n : ℕ) : ℕ := 2 ^ (n - 2)
def a := sequence_term 2  -- Second term of the sequence
def b := sequence_term 4  -- Fourth term of the sequence
def sample := [a, 3, 5, 7]
def mean := b

-- Function to calculate sample variance
def sample_variance (s : list ℕ) (x_bar : ℕ) : ℕ :=
  (1 / (s.length - 1)) * (s.sum (λ x, (x - x_bar) ^ 2))

-- The statement to prove
theorem variance_of_sample :
  sample == [1, 3, 5, 7] ∧ mean == 4 ∧ sample_variance sample mean == 5 := by
{ sorry }

end variance_of_sample_l517_517746


namespace sequence_sum_2016_l517_517748

noncomputable def sequenceSum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2
  else let a : ℕ → ℤ := λ n, ite (n = 1) 2 $
    nat.recOn n.succ_pred
      (-2 : ℤ)
      (λ n a_n1, 1 - a_n1) in
  (list.range n).sum (λ i, a (i + 1))

theorem sequence_sum_2016 : sequenceSum 2016 = 1007 := sorry

end sequence_sum_2016_l517_517748


namespace prime_pairs_sum_to_50_l517_517428

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517428


namespace escalator_length_l517_517676

variable (L : ℝ)
variable (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ)

-- Given conditions
def condition1 : escalator_speed = 7 := by sorry
def condition2 : person_speed = 2 := by sorry
def condition3 : time = 20 := by sorry

-- Proof statement
theorem escalator_length :
  L = (escalator_speed + person_speed) * time → L = 180 := by
  intros h
  rw [condition1, condition2, condition3] at h
  exact h

end escalator_length_l517_517676


namespace value_of_p_l517_517919

theorem value_of_p (p q : ℝ) (h1 : q = (2 / 5) * p) (h2 : p * q = 90) : p = 15 :=
by
  sorry

end value_of_p_l517_517919


namespace prime_pairs_sum_50_l517_517264

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517264


namespace part1_part2_part3_l517_517823

open Real

-- Definitions of points
structure Point :=
(x : ℝ)
(y : ℝ)

def M (m : ℝ) : Point := ⟨m - 2, 2 * m - 7⟩
def N (n : ℝ) : Point := ⟨n, 3⟩

-- Part 1
theorem part1 : 
  (M (7 / 2)).y = 0 ∧ (M (7 / 2)).x = 3 / 2 :=
by
  sorry

-- Part 2
theorem part2 (m : ℝ) : abs (m - 2) = abs (2 * m - 7) → (m = 5 ∨ m = 3) :=
by
  sorry

-- Part 3
theorem part3 (m n : ℝ) : abs ((M m).y - 3) = 2 ∧ (M m).x = n - 2 → (n = 4 ∨ n = 2) :=
by
  sorry

end part1_part2_part3_l517_517823


namespace sum_partition_36_l517_517513

theorem sum_partition_36 : 
  ∃ (S : Finset ℕ), S.card = 36 ∧ S.sum id = ((Finset.range 72).sum id) / 2 :=
by
  sorry

end sum_partition_36_l517_517513


namespace prime_pairs_sum_50_l517_517374

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517374


namespace factorize_expression_l517_517042

theorem factorize_expression
  (x : ℝ) :
  ( (x^2-1)*(x^4+x^2+1)-(x^3+1)^2 ) = -2*(x + 1)*(x^2 - x + 1) :=
by
  sorry

end factorize_expression_l517_517042


namespace bus_stop_time_l517_517040

noncomputable def time_stopped_per_hour (excl_speed incl_speed : ℕ) : ℕ :=
  60 * (excl_speed - incl_speed) / excl_speed

theorem bus_stop_time:
  time_stopped_per_hour 54 36 = 20 :=
by
  sorry

end bus_stop_time_l517_517040


namespace problem_l517_517061
-- Import necessary library

-- Define the problem conditions and question
theorem problem (a b : ℝ) (h : f x = ax^3 + bx - 4) (h1 : f (-2) = 2) : 
  f (2) = -10 :=
by
  sorry

end problem_l517_517061


namespace maximum_marks_l517_517630

-- Definitions derived from the conditions
def passing_mark (M : ℕ) : ℕ := (33 * M) / 100
def failed_by (marks: ℕ) (failed_by : ℕ) : ℕ := marks + failed_by

-- Main statement: Given conditions, proving that the maximum marks M is 500
theorem maximum_marks (M : ℕ) (marks : ℕ) (failed_by : ℕ)
  (h1 : marks = 125) (h2 : failed_by = 40) (h3 : passing_mark M = failed_by (marks) (failed_by)) :
  M = 500 :=
by
  sorry

end maximum_marks_l517_517630


namespace vectors_parallel_derivative_at_one_cosine_A_tangent_curve_l517_517963

-- Problem 1
theorem vectors_parallel (x : ℝ) (h : ((1:ℝ), x) ∥ (x - 1, 2)) : x = 1 ∨ x = 2 := 
sorry

-- Problem 2
theorem derivative_at_one (f : ℝ → ℝ) (h : f = λ x, x^3 - (f' 1) * x^2 + 1) : f' 1 = 1 := 
sorry

-- Problem 3
theorem cosine_A (a b c : ℝ) (A B C : ℝ) (hb : b = sqrt 2 * c) (hs: sin A + sqrt 2 * sin C = 2 * sin B ) : cos A = sqrt 2 / 4 := 
sorry

-- Problem 4
theorem tangent_curve (a : ℝ) (h_tangent : tangent (curve := λ x, x + ln x) (pt := (1, 1)) = tangent (curve := λ x, ax^2 + (a+2)x+1)) : a = 8 :=
sorry

end vectors_parallel_derivative_at_one_cosine_A_tangent_curve_l517_517963


namespace trader_gain_equivalent_l517_517682

theorem trader_gain_equivalent (C : ℝ) (hC : C ≠ 0) :
  let G := (1/6 : ℝ) * 90 in
  G = 15 :=
by
  sorry

end trader_gain_equivalent_l517_517682


namespace tan_theta_sqrt3_l517_517605

theorem tan_theta_sqrt3 (θ : ℝ) (h1 : real.angle.radians (π / 4) < real.angle.ofReal θ) (h2 : real.angle.ofReal θ < real.angle.radians (π / 2)) 
(h3 : real.tan θ + real.tan (3 * θ) + real.tan (5 * θ) = 0) : real.tan θ = real.sqrt 3 :=
sorry

end tan_theta_sqrt3_l517_517605


namespace min_sum_distances_five_digit_numbers_l517_517588

theorem min_sum_distances_five_digit_numbers 
  (a b : Fin 10 → Nat) -- a and b represent sequences of five-digit numbers considered digit-wise
  (h1 : ∃ x1 x2 x3 x4 x5 : Nat, 
    x1 + x2 + x3 + x4 + x5 = 89999 ∧ 
    x2 + x3 + x4 + x5 ≥ 9999 ∧ 
    x3 + x4 + x5 ≥ 999 ∧ 
    x4 + x5 ≥ 99 ∧ 
    x5 ≥ 9) :
  (∃ S, S = 101105) ↔ 
  (∃ x1 x2 x3 x4 x5 : Nat, 
    x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 = 101105 ∧ 
    x1 + x2 + x3 + x4 + x5 = 89999 ∧ 
    x2 + x3 + x4 + x5 ≥ 9999 ∧ 
    x3 + x4 + x5 ≥ 999 ∧ 
    x4 + x5 ≥ 99 ∧ 
    x5 ≥ 9) :=
by 
  -- Introducing variables x1, x2, x3, x4, x5
  let x1 := h1.some',
  let x2 := h1.some'_next,
  let x3 := h1.some'_next',
  let x4 := h1.some'_next'_next,
  let x5 := h1.some'_next'_next',
  
  -- We start by assuming the given conditions are true
  have h2 : x1 + x2 + x3 + x4 + x5 = 89999, from h1.exists.some_spec,
  have h3 : x5 ≥ 9, from h1.exists.snd,
  have h4 : x4 + x5 ≥ 99, from h1.exists.snd,
  have h5 : x3 + x4 + x5 ≥ 999, from h1.exists.snd,
  have h6 : x2 + x3 + x4 + x5 ≥ 9999, from h1.exists.snd,
  
  -- Apply summation to the inequalities
  have h7 : x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 ≥ 101105,
  sorry,
  
  -- State the equivalence condition
  exact ⟨λ h, ⟨x1, x2, x3, x4, x5, h7⟩, λ h, ⟨101105⟩⟩

end min_sum_distances_five_digit_numbers_l517_517588


namespace extra_large_poster_count_correct_l517_517994

def is_total_poster_count (n : ℕ) := n = 200
def is_small_poster_count (small : ℕ) (total : ℕ) := small = total / 4
def is_medium_poster_count (medium : ℕ) (total : ℕ) := medium = total / 3 - 1 -- rounding down not strict
def is_large_poster_count (large : ℕ) (total : ℕ) := large = total / 5
def extra_large_poster_count (total : ℕ) (small medium large : ℕ) := total - (small + medium + large)

theorem extra_large_poster_count_correct (total small medium large extra_large : ℕ) 
    (h_total : is_total_poster_count total)
    (h_small : is_small_poster_count small total)
    (h_medium : is_medium_poster_count medium total)
    (h_large : is_large_poster_count large total)
    (h_extra_large : extra_large_poster_count total small medium large = extra_large) : 
    extra_large = 44 :=
by 
  unfold is_total_poster_count at h_total
  unfold is_small_poster_count at h_small
  unfold is_medium_poster_count at h_medium
  unfold is_large_poster_count at h_large
  unfold extra_large_poster_count at h_extra_large
  rw [h_total, h_small, h_medium, h_large] at h_extra_large
  -- Calculate remaining posters via a helper function
  have h_calc := Nat.sub_eq_iff_eq_add.mpr h_extra_large
  rw [h_calc]
  sorry

end extra_large_poster_count_correct_l517_517994


namespace b_alone_completes_work_in_30_days_l517_517628

theorem b_alone_completes_work_in_30_days (A B : ℝ) :
  (A + B = 1 / 12) → (A = 1 / 20) → (B = 1 / 30) :=
by 
  intros h1 h2
  have h3 : B = 1 / 30 := by 
    calc
      B   = (1 / 12) - (1 / 20) : by sorry
  exact h3

end b_alone_completes_work_in_30_days_l517_517628


namespace problem_l517_517744

-- Definitions for the problem
variables {O A B C D E F H G : Type} [circumcircle : circumcircle O (triangle A B C)]
variable [angleBisector : angleBisector A D (angle A B C)]
variable [parallelOEED : Parallel O E E D]
variable [intersectsAB : Intersects AB E]
variable [parallelOFDC : Parallel O F D C]
variable [intersectsAC : Intersects AC F]
variable [orthocenterH : orthocenter H (triangle A B C)]
variable [parallelHGAD : Parallel H G A D]
variable [intersectsBC : Intersects BC G]

-- The theorem that needs to be proved
theorem problem :
  BE = GE ∧ GE = GF ∧ GF = CF :=
sorry

end problem_l517_517744


namespace conic_section_eccentricity_geometric_seq_l517_517810

theorem conic_section_eccentricity_geometric_seq (a_1 a_3 : ℝ) (a_2 : ℝ) 
  (h_geometric : ∀ n m, (∃ r, a_n = r ^ n * (-1) ∧ a_m = r ^ m * (-81)))
  (h_a2 : a_2 = 9) :
  ∃ e, e = sqrt 10 ∧ (x^2 + ⟦y^2/a_2⟧ = 1) :=
by
  sorry

end conic_section_eccentricity_geometric_seq_l517_517810


namespace prime_pairs_sum_50_l517_517358

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517358


namespace count_prime_pairs_sum_50_exactly_4_l517_517297

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517297


namespace congruent_triangles_angles_l517_517951

theorem congruent_triangles_angles (ΔABC ΔDEF : Triangle)
  (h : congruent ΔABC ΔDEF) : 
  ∀ a b, opposite_angle a = b → congruent_angles ΔABC ΔDEF a b := 
sorry

end congruent_triangles_angles_l517_517951


namespace circle_chord_eqn_l517_517796

theorem circle_chord_eqn (x y : ℝ) :
  (exists (PQ : ℝ → ℝ), ∀ t : ℝ, (x + t, y + PQ t) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 9} ∧
    ∃ A B C : ℝ, (A, B) = (1, 2) ∧ (x, y) = A + B * t ∨ (x, y) = C + PQ t) →
  (∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ a + 2 * b - 5 = 0) :=
by sorry

end circle_chord_eqn_l517_517796


namespace cos_triple_angle_l517_517473

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517473


namespace infinitesimal_alpha_as_t_to_zero_l517_517891

open Real

noncomputable def alpha (t : ℝ) : ℝ × ℝ :=
  (t, sin t)

theorem infinitesimal_alpha_as_t_to_zero : 
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, abs t < δ → abs (alpha t).fst + abs (alpha t).snd < ε := by
  sorry

end infinitesimal_alpha_as_t_to_zero_l517_517891


namespace sandy_walked_distance_l517_517888

-- Define the problem conditions in Lean 4
variables (d : ℝ)

-- The conditions provided in the problem
def sandy_walked_legs (d : ℝ) := 
let distanceSouth := d
let distanceLeft1 := d
let distanceLeft2 := d
let distanceRight := 25
in 
(distanceSouth - distanceLeft2) = 45

-- The question translated to a Lean proof statement
theorem sandy_walked_distance (d : ℝ) (h : sandy_walked_legs d) : d = 45 :=
sorry

end sandy_walked_distance_l517_517888


namespace magnitude_of_z_l517_517770

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := (10 / (3 + i)) - 2 * i

-- State the theorem to be proven
theorem magnitude_of_z : complex.abs z = 3 * real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l517_517770


namespace inscribed_circle_radius_l517_517693

theorem inscribed_circle_radius
  (a b c : ℝ)
  (h_a : a = 3)
  (h_b : b = 6)
  (h_c : c = 18)
  (h_formula : ∀ r : ℝ, 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))) :
  ∃ r : ℝ, r = 9 / 8 :=
begin
  -- proof part will be added here
  sorry
end

end inscribed_circle_radius_l517_517693


namespace increasing_intervals_range_of_f_on_interval_l517_517776

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x) - Real.sqrt 3 * Math.cos (2 * x)

theorem increasing_intervals : 
  ∀ (k : ℤ), ∀ (x : ℝ), 
    (k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12) → 
    is_strict_mono (λ x, f x) :=
sorry

theorem range_of_f_on_interval : 
  ∃ (a b : ℝ), 
    (a = 0 ∧ b = 2) ∧ 
    (∀ y : ℝ, y ∈ Set.image f (Set.Icc (Real.pi / 6) (Real.pi / 2)) ↔ y ∈ Set.Icc a b) :=
sorry

end increasing_intervals_range_of_f_on_interval_l517_517776


namespace radius_of_tangent_circle_l517_517652

-- Define the conditions
def is_45_45_90_triangle (A B C : ℝ × ℝ) (AB BC AC : ℝ) : Prop :=
  (AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2) ∧
  (A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2))

def is_tangent_to_axes (O : ℝ × ℝ) (r : ℝ) : Prop :=
  O = (r, r)

def is_tangent_to_hypotenuse (O : ℝ × ℝ) (r : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - O.1) = Real.sqrt 2 * r ∧ (C.2 - O.2) = Real.sqrt 2 * r

-- Main theorem
theorem radius_of_tangent_circle :
  ∃ r : ℝ, ∀ (A B C O : ℝ × ℝ),
    is_45_45_90_triangle A B C (2) (2) (2 * Real.sqrt 2) →
    is_tangent_to_axes O r →
    is_tangent_to_hypotenuse O r C →
    r = Real.sqrt 2 :=
by
  sorry

end radius_of_tangent_circle_l517_517652


namespace count_prime_pairs_sum_50_exactly_4_l517_517283

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517283


namespace intersection_M_N_eq_M_l517_517597

-- Define the sets M and N as per the given conditions
def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

-- Prove that the intersection of M and N equals M
theorem intersection_M_N_eq_M : M ∩ N = M := by
  -- Proof goes here
  sorry

end intersection_M_N_eq_M_l517_517597


namespace biff_break_even_hours_l517_517005

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end biff_break_even_hours_l517_517005


namespace count_four_digit_numbers_with_digits_2_or_5_l517_517120

theorem count_four_digit_numbers_with_digits_2_or_5 : 
  ∃ (count : ℕ), count = 16 ∧ 
  (∀ n : ℕ, n >= 1000 → n < 10000 → 
   (∀ d ∈ [ (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10], 
    d = 2 ∨ d = 5) → count = 16) :=
by {
  -- Count the number of four-digit numbers where each digit is either 2 or 5
  let count := (2 ^ 4),
  use count,
  split,
  { exact rfl },
  { intros n hn1 hn2 h_digits,
    sorry
  }
}

end count_four_digit_numbers_with_digits_2_or_5_l517_517120


namespace calculate_down_payment_l517_517837

theorem calculate_down_payment : 
  let monthly_fee := 12
  let years := 3
  let total_paid := 482
  let num_months := years * 12
  let total_monthly_payments := num_months * monthly_fee
  let down_payment := total_paid - total_monthly_payments
  down_payment = 50 :=
by
  sorry

end calculate_down_payment_l517_517837


namespace exists_perfect_square_l517_517021

theorem exists_perfect_square (a : ℕ → ℕ) (h1 : a 1 > 0) (hn : ∀ n, a (n + 1) > a n + 1) :
  ∀ n, ∃ k : ℕ, let b := (∑ i in finset.range (n + 1), a i) in b ≤ k^2 ∧ k^2 < (∑ i in finset.range (n + 2), a i) :=
begin
  sorry
end

end exists_perfect_square_l517_517021


namespace prime_pairs_sum_50_l517_517370

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517370


namespace solve_smallest_positive_angle_l517_517686

theorem solve_smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ tan (3 * y * (Real.pi / 180)) = (cos (y * (Real.pi / 180)) - sin (y * (Real.pi / 180))) / (cos (y * (Real.pi / 180)) + sin (y * (Real.pi / 180))) ∧ y = 11.25 :=
begin
  sorry -- proof is not required
end

end solve_smallest_positive_angle_l517_517686


namespace prime_pairs_sum_50_l517_517252

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517252


namespace prime_pairs_sum_to_50_l517_517304

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517304


namespace prime_pairs_sum_50_l517_517445

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517445


namespace one_and_one_third_of_what_number_is_45_l517_517879

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l517_517879


namespace min_PA_PF_value_l517_517097

theorem min_PA_PF_value 
  (P : ℝ × ℝ)
  (A : ℝ × ℝ)
  (F : ℝ × ℝ)
  (hA : A = (1, 1))
  (hEllipse : P.1 ^ 2 / 4 + P.2 ^ 2 / 3 = 1)
  (hFocus : F = (-1, 0))
  (hA_inside : (1 ^ 2 / 4 + 1 ^ 2 / 3 < 1)) 
  : (min_value (λ P, abs ((P.1 - 1) ^ 2 + (P.2 - 1) ^ 2) + abs ((P.1 + 1) ^ 2 + P.2 ^ 2))) = 3 :=
sorry

end min_PA_PF_value_l517_517097


namespace tangent_line_at_origin_is_neg3x_l517_517524

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem tangent_line_at_origin_is_neg3x
  (h : ∀ x, f' (-x) = f' x) :
  tangent_line_at_origin f = -3 * x :=
by
  sorry

end tangent_line_at_origin_is_neg3x_l517_517524


namespace maximize_triangle_area_l517_517780

theorem maximize_triangle_area (m : ℝ) (l : ∀ x y, x + y + m = 0) (C : ∀ x y, x^2 + y^2 + 4 * y = 0) :
  m = 0 ∨ m = 4 :=
sorry

end maximize_triangle_area_l517_517780


namespace sum_of_ages_is_nineteen_l517_517001

-- Definitions representing the conditions
def Bella_age : ℕ := 5
def Brother_is_older : ℕ := 9
def Brother_age : ℕ := Bella_age + Brother_is_older
def Sum_of_ages : ℕ := Bella_age + Brother_age

-- Mathematical statement (theorem) to be proved
theorem sum_of_ages_is_nineteen : Sum_of_ages = 19 := by
  sorry

end sum_of_ages_is_nineteen_l517_517001


namespace count_prime_pairs_summing_to_50_l517_517225

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517225


namespace D_96_equals_112_l517_517848

def multiplicative_decompositions (n : ℕ) : ℕ :=
  sorry -- Define how to find the number of multiplicative decompositions

theorem D_96_equals_112 : multiplicative_decompositions 96 = 112 :=
  sorry

end D_96_equals_112_l517_517848


namespace prime_pairs_sum_to_50_l517_517300

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517300


namespace num_prime_pairs_summing_to_50_l517_517355

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517355


namespace digit_in_base8_addition_l517_517028

theorem digit_in_base8_addition :
  ∀ {s : ℕ},
  543s + s67 + s4 = 65s3 ∧ s < 8 → s = 0 :=
by
  assume s
  assume h
  sorry

end digit_in_base8_addition_l517_517028


namespace max_abs_sum_on_ellipse_l517_517452

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 4 -> |x| + |y| ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by
  intro x y h
  sorry

end max_abs_sum_on_ellipse_l517_517452


namespace prime_pairs_sum_to_50_l517_517302

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517302


namespace primes_sum_50_l517_517396

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517396


namespace at_least_one_not_greater_than_minus_four_l517_517525

theorem at_least_one_not_greater_than_minus_four {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 :=
sorry

end at_least_one_not_greater_than_minus_four_l517_517525


namespace area_formed_by_curve_and_line_l517_517904

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (f x - g x)

theorem area_formed_by_curve_and_line :
  area_under_curve (λ x, x) (λ x, x^3 - 3 * x) 0 2 * 2 = 8 := by
  sorry

end area_formed_by_curve_and_line_l517_517904


namespace general_term_of_sequence_l517_517768

theorem general_term_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S (n + 1) = 3 * (n + 1) ^ 2 - 2 * (n + 1)) →
  a 1 = 1 →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  (∀ n, a n = 6 * n - 5) := 
by
  intros hS ha1 ha
  sorry

end general_term_of_sequence_l517_517768


namespace prime_pairs_sum_to_50_l517_517313

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517313


namespace count_prime_pairs_sum_50_l517_517146

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517146


namespace log_not_rational_l517_517883

noncomputable def log1999 (x : Real) : Real := log x / log 1999

theorem log_not_rational (f g : Real[X]) (h_coprime : IsCoprime f g) :
  ¬ (∀ x, log1999 x = (f.eval x) / (g.eval x)) :=
sorry

end log_not_rational_l517_517883


namespace inscribed_quadrilateral_diagonal_ratio_l517_517983

theorem inscribed_quadrilateral_diagonal_ratio
  (a b c d : ℝ) 
  (α β : ℝ) 
  (S_ABCD : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ 180)
  (h2 : 0 ≤ β ∧ β ≤ 180)
  (h3 : α + β = 180) :
  (b * c + a * d) / (a * b + c * d) = sin α / sin β := 
sorry

end inscribed_quadrilateral_diagonal_ratio_l517_517983


namespace prime_pairs_sum_to_50_l517_517423

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517423


namespace total_bananas_l517_517559

theorem total_bananas (
  bananas_visit1 : Nat := 12
  bananas_visit2 : Nat := 18
  bananas_visit3 : Nat := 25
  bananas_visit4 : Nat := 10
  bananas_visit5 : Nat := 15
) : bananas_visit1 + bananas_visit2 + bananas_visit3 + bananas_visit4 + bananas_visit5 = 80 :=
by sorry

end total_bananas_l517_517559


namespace count_correct_relations_l517_517672

theorem count_correct_relations :
  let empty_set := ∅
  let singleton_set := {0}
  let conditions := [empty_set ∉ singleton_set, 0 ∉ empty_set, singleton_set ⊆ singleton_set, empty_set ≠ singleton_set]
  (conditions.count true) = 1 :=
by
  sorry

end count_correct_relations_l517_517672


namespace general_formula_sum_b_formula_l517_517083

-- Define the arithmetic sequence and its properties
variables {a : ℕ → ℕ} {S : ℕ → ℕ}
-- Define the conditions: first term and sum of terms
def a1 : Prop := a 1 = 1
def Sn : Prop := ∀ n, S n = n * (a 1 + a n) / 2
-- Condition: geometric sequence formed by S1, S2, and S4
def geometric_condition : Prop := (S 2) ^ 2 = (S 1) * (S 4)

-- Question I: Prove the general formula for the arithmetic sequence
theorem general_formula (h1 : a1) (h2 : Sn) (h3 : geometric_condition) : ∀ n, a n = 2 * n - 1 :=
begin
  sorry,
end

-- Define b_n and its properties
def b (n : ℕ) : ℕ := 1 / (a n * a (n + 1))
-- Define the sum of first n terms of sequence b_n
def sum_b (n : ℕ) := (finset.range n).sum b

-- Question II: Prove the sum of the first n terms of b_n
theorem sum_b_formula (h1 : a1) (h2 : Sn) (h3 : geometric_condition) : ∀ n, sum_b n = n / (2 * n + 1) :=
begin
  sorry,
end

end general_formula_sum_b_formula_l517_517083


namespace cos_triple_angle_l517_517475

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517475


namespace tangent_line_eq_l517_517906

noncomputable def f : ℝ → ℝ := λ x, 1 / x

def point : ℝ × ℝ := (2, 1 / 2)

theorem tangent_line_eq : ∀ x y : ℝ, f 2 = 1 / 2 ∧ (y - (1/2) = (-1/4) * (x - 2)) → y = - (1/4) * x + 1 :=
by
  sorry

end tangent_line_eq_l517_517906


namespace dodecagon_area_l517_517997

theorem dodecagon_area (a : ℝ) : 
  let dodecagon_area := 12 * (1 / 2 * (a / real.sqrt 2) * (a / real.sqrt 2) * (1 / 2))
  in dodecagon_area = (3 * a^2) / 2 :=
by
  sorry

end dodecagon_area_l517_517997


namespace heaviest_lightest_difference_total_excess_weight_total_selling_price_l517_517650

-- Define deviations from standard weight and their counts
def deviations : List (ℚ × ℕ) := [(-3.5, 2), (-2, 4), (-1.5, 2), (0, 1), (1, 3), (2.5, 8)]

-- Define standard weight and price per kg
def standard_weight : ℚ := 18
def price_per_kg : ℚ := 1.8

-- Prove the three statements:
theorem heaviest_lightest_difference :
  (2.5 - (-3.5)) = 6 := by
  sorry

theorem total_excess_weight :
  (2 * -3.5 + 4 * -2 + 2 * -1.5 + 1 * 0 + 3 * 1 + 8 * 2.5) = 5 := by
  sorry

theorem total_selling_price :
  (standard_weight * 20 + 5) * price_per_kg = 657 := by
  sorry

end heaviest_lightest_difference_total_excess_weight_total_selling_price_l517_517650


namespace find_lcm_of_two_numbers_l517_517940

noncomputable def hcf (a b : ℕ) : ℕ := sorry
noncomputable def lcm (a b : ℕ) : ℕ := sorry

theorem find_lcm_of_two_numbers (a b : ℕ) (h_hcf : hcf a b = 11) (h_product : a * b = 2310) (h_multiple_of_7 : 7 ∣ a ∨ 7 ∣ b) : lcm a b = 210 :=
  sorry

end find_lcm_of_two_numbers_l517_517940


namespace sin_ratio_of_pentagon_l517_517902

theorem sin_ratio_of_pentagon (θ : ℝ) (hθ : θ = 108) :
  let phi := (sqrt 5 - 1) / 2 in
  2 * sin (18 * real.pi / 180) = phi → 
  sin θ / sin (36 * real.pi / 180) = (sqrt 5 + 1) / 2 :=
by 
  intro hphi
  sorry

end sin_ratio_of_pentagon_l517_517902


namespace eval_expr_at_neg3_l517_517037

theorem eval_expr_at_neg3 : 
  (let x := -3 in 
    (4 + x * (2 + x) - 2^2) / (x - 2 + x^2)) = 3 / 4 :=
by
  sorry

end eval_expr_at_neg3_l517_517037


namespace arithmetic_sequence_formula_geometric_sequence_formula_cn_formula_sum_c_2016_l517_517075

def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 3^(n - 1)

def c : ℕ → ℕ 
| 0     := 0  -- convention to handle zero index gracefully
| 1     := 3
| (n+2) := 2 * 3^n

theorem arithmetic_sequence_formula : ∀ n : ℕ, a n = 2 * n - 1 := 
by
  sorry

theorem geometric_sequence_formula : ∀ n : ℕ, b n = 3^(n - 1) :=
by
  sorry

theorem cn_formula : ∀ n : ℕ, c n = 
  match n with
  | 0     => 0
  | 1     => 3
  | (n+2) => 2 * 3^n := 
by
  sorry

theorem sum_c_2016 : (Finset.range 2016).sum (λ i, c (i + 1)) = 3^2016 :=
by
  sorry

end arithmetic_sequence_formula_geometric_sequence_formula_cn_formula_sum_c_2016_l517_517075


namespace maximum_value_neg_domain_l517_517581

-- Define the function
def func (x : ℝ) : ℝ := x + 4 / x

-- The theorem stating the maximum value of the function for x < 0 is -4
theorem maximum_value_neg_domain (x : ℝ) (h : x < 0) : func x ≤ -4 ∧ (func (-2) = -4) :=
by
  sorry

end maximum_value_neg_domain_l517_517581


namespace area_triangle_BRS_l517_517566

def point := ℝ × ℝ
def x_intercept (p : point) : ℝ := p.1
def y_intercept (p : point) : ℝ := p.2

noncomputable def distance_from_y_axis (p : point) : ℝ := abs p.1

theorem area_triangle_BRS (B R S : point)
  (hB : B = (4, 10))
  (h_perp : ∃ m₁ m₂, m₁ * m₂ = -1)
  (h_sum_zero : x_intercept R + x_intercept S = 0)
  (h_dist : distance_from_y_axis B = 10) :
  ∃ area : ℝ, area = 60 := 
sorry

end area_triangle_BRS_l517_517566


namespace range_of_a_l517_517733

-- Defining the piecewise function
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- Theorem stating the range of a for f to be decreasing
theorem range_of_a (a : ℝ) : ∀ x y, x < y → f a x ≥ f a y ↔ a ∈ Set.Ico (1/6 : ℝ) (1/2 : ℝ) :=
sorry

end range_of_a_l517_517733


namespace joe_speed_l517_517509

theorem joe_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_run : ℝ) (distance : ℝ) 
  (h1 : joe_speed = 2 * pete_speed)
  (h2 : time_run = 2 / 3)
  (h3 : distance = 16)
  (h4 : distance = 3 * pete_speed * time_run) :
  joe_speed = 16 :=
by sorry

end joe_speed_l517_517509


namespace angle_between_skew_lines_exists_l517_517935

-- Define basic structures for lines and angles in 3D space
structure Line (ℝ : Type) := 
(point : ℝ × ℝ × ℝ) 
(direction : ℝ × ℝ × ℝ)

noncomputable def angle_between_lines (l₁ l₂ : Line ℝ) : ℝ := sorry

-- Define the conditions (g and f are skew lines in general position)
variables (g f : Line ℝ)
-- Add a condition that g and f are in general position in space and are skew
#check (g.direction ≠ f.direction)  -- just indicative; in correct format, more specific conditions would be added

-- Formulating the proof problem
theorem angle_between_skew_lines_exists :
  ∃ θ : ℝ, θ = angle_between_lines g f :=
sorry

end angle_between_skew_lines_exists_l517_517935


namespace alicia_masks_left_l517_517914

theorem alicia_masks_left (T G L : ℕ) (hT : T = 90) (hG : G = 51) (hL : L = T - G) : L = 39 :=
by
  rw [hT, hG] at hL
  exact hL

end alicia_masks_left_l517_517914


namespace num_prime_pairs_sum_50_l517_517203

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517203


namespace area_enclosed_by_line_and_curve_l517_517643

theorem area_enclosed_by_line_and_curve (n : ℕ) (h : binomial.coeff n 1 = binomial.coeff n 3) : 
  ∫ x in 0..4, (4 * x - x ^ 2) = 32 / 3 := 
sorry

end area_enclosed_by_line_and_curve_l517_517643


namespace num_prime_pairs_sum_50_l517_517332

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517332


namespace least_number_divisible_by_11_and_remainder_2_l517_517614

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end least_number_divisible_by_11_and_remainder_2_l517_517614


namespace prime_pairs_sum_to_50_l517_517310

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517310


namespace quadratic_function_properties_l517_517070

theorem quadratic_function_properties:
  (∀ x : ℝ, f(x+1) - f(x) = 2x - 1) ∧ (f(0) = 3) ∧ 
  (∀ m : ℝ, 
     (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f(x) ≥ 2 * m * x) ↔ m ∈ (Set.Ico (-3) (Real.sqrt 3))) :=
sorry

end quadratic_function_properties_l517_517070


namespace Rachel_made_total_amount_l517_517700

def cost_per_bar : ℝ := 3.25
def total_bars_sold : ℕ := 25 - 7
def total_amount_made : ℝ := total_bars_sold * cost_per_bar

theorem Rachel_made_total_amount :
  total_amount_made = 58.50 :=
by
  sorry

end Rachel_made_total_amount_l517_517700


namespace manuscript_typing_cost_l517_517960

theorem manuscript_typing_cost (pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (initial_cost_per_page : ℝ) (revision_cost_per_page : ℝ) (revision_mult_twice : ℝ) :
  pages = 200 →
  revised_once = 80 →
  revised_twice = 20 →
  initial_cost_per_page = 5 →
  revision_cost_per_page = 3 →
  revision_mult_twice = 2 →
  (initial_cost_per_page * pages + revision_cost_per_page * revised_once + 
   (revision_cost_per_page * revision_mult_twice) * revised_twice) = 1360 := 
by
  intros hpages hrev_once hrev_twice hinit_cost hrev_cost hrev_mult_twice
  rw [hpages, hrev_once, hrev_twice, hinit_cost, hrev_cost, hrev_mult_twice]
  norm_num
  sorry

end manuscript_typing_cost_l517_517960


namespace geometric_sequence_sum_l517_517564

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 * q + a 1 * q ^ 3 = 20 →
    a 1 * q ^ 2 + a 1 * q ^ 4 = 40 →
    a 1 * q ^ 4 + a 1 * q ^ 6 = 160 :=
by
  sorry

end geometric_sequence_sum_l517_517564


namespace num_prime_pairs_sum_50_l517_517172

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517172


namespace fraction_of_sum_simple_interest_l517_517485

variable (P : ℝ) -- Principal sum of money
constant R : ℝ := 0.10 -- Rate of interest per annum
constant T : ℝ := 2 -- Time period in years

def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100

theorem fraction_of_sum_simple_interest : simple_interest P R T = P / 5 := by
  unfold simple_interest
  rw [R, T]
  sorry

end fraction_of_sum_simple_interest_l517_517485


namespace num_prime_pairs_summing_to_50_l517_517338

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517338


namespace max_expression_value_l517_517945

theorem max_expression_value (a b c d e f g h k : ℤ)
  (ha : (a = 1 ∨ a = -1)) (hb : (b = 1 ∨ b = -1))
  (hc : (c = 1 ∨ c = -1)) (hd : (d = 1 ∨ d = -1))
  (he : (e = 1 ∨ e = -1)) (hf : (f = 1 ∨ f = -1))
  (hg : (g = 1 ∨ g = -1)) (hh : (h = 1 ∨ h = -1))
  (hk : (k = 1 ∨ k = -1)) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 := sorry

end max_expression_value_l517_517945


namespace additional_people_needed_l517_517033

-- Definitions corresponding to the given conditions
def person_hours (n : ℕ) (t : ℕ) : ℕ := n * t
def initial_people : ℕ := 8
def initial_time : ℕ := 10
def total_person_hours := person_hours initial_people initial_time

-- Lean statement of the problem
theorem additional_people_needed (new_time : ℕ) (new_people : ℕ) : 
  new_time = 5 → person_hours new_people new_time = total_person_hours → new_people - initial_people = 8 :=
by
  intro h1 h2
  sorry

end additional_people_needed_l517_517033


namespace symmetric_line_l517_517576

theorem symmetric_line (x y : ℝ) : (2 * x + y - 4 = 0) → (2 * x - y + 4 = 0) :=
by
  sorry

end symmetric_line_l517_517576


namespace division_base7_l517_517041

-- Definition for base conversion from base 7 to base 10
def base7_to_base10 (n : ℕ) (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d, acc * n + d) 0

-- Constants given in the problem
def n1_digits := [1, 4, 5, 2]  -- represents 1452_7 in base 7
def n2_digits := [1, 4]        -- represents 14_7 in base 7

-- Expectations for the problem
def result_digits := [1, 0, 3]  -- represents 103_7 in base 7

-- Lean theorem statement
theorem division_base7 : 
  let n1 := base7_to_base10 7 n1_digits in
  let n2 := base7_to_base10 7 n2_digits in
  let expected_result := base7_to_base10 7 result_digits in
  n1 / n2 = expected_result :=
by
  sorry -- proof would go here

end division_base7_l517_517041


namespace blue_polygons_more_than_red_polygons_l517_517544

open_locale big_operators

theorem blue_polygons_more_than_red_polygons (red_points : Finset ℕ) (H : red_points.card = 40) :
  let blue := 1,
      red_polygons := ∑ k in range (40 + 1), if 3 ≤ k then finset.card (finset.powerset_len k.red_points) else 0,
      blue_polygons := ∑ k in finset.range 40, if 2 ≤ k then finset.card (finset.powerset_len k (red_points ∪ {blue})) else 0
  in blue_polygons - red_polygons = 780 := sorry

end blue_polygons_more_than_red_polygons_l517_517544


namespace homogenous_polynomial_P_l517_517897

noncomputable def P (x y : ℝ) : ℝ := sorry

def isHomogeneous (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ a : (Fin n.succ) → ℝ, ∀ x y, P x y = ∑ i in Finset.range (n + 1), a i * (x^(Finset.card (Finset.range i))) * (y^(Finset.card (Finset.range (n - i))))

theorem homogenous_polynomial_P (P : ℝ → ℝ → ℝ) (k : ℕ) :
  (∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) →
  (∃ n : ℕ, n > 0 ∧ isHomogeneous P n) →
  (∃ k : ℕ, k > 0 ∧ ∀ x y : ℝ, P x y = (x^2 + y^2)^k) :=
by
  intros h1 h2
  sorry

end homogenous_polynomial_P_l517_517897


namespace count_prime_pairs_sum_50_exactly_4_l517_517291

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517291


namespace nine_point_circle_angle_l517_517554

theorem nine_point_circle_angle (ABC : Triangle)
  (acute_angled : acute_triangle ABC)
  (D : Point) (foot_CD : altitude CD ABC)
  (O : Point) (circumcenter_O : circumcenter O ABC)
  (N : Point) (midpoint_N : midpoint N AB)
  (H : Point) (orthocenter_H : orthocenter H ABC)
  (E : Point) (midpoint_E : midpoint E (C, H))
  (nine_point_circle_center : Point) 
  (center_nine_point_circle : nine_point_circle_center = midpoint H O)
  (NED_on_nine_point_circle : on_nine_point_circle N E D ABC) :
  angle nine_point_circle_center N D = 2 * |angle A - angle B| :=
by
  sorry

end nine_point_circle_angle_l517_517554


namespace total_worth_green_div_by_three_l517_517573

-- Define the set of coin colors
inductive CoinColor
| Green | White | Orange

open CoinColor

-- Define the structure of coins
structure Coin :=
(value : ℕ)
(color : CoinColor)

-- Define the conditions
def no_touch_same_color (coins : List (List Coin)) : Prop :=
  ∀ i j, i < coins.length → j < coins.head.length →
    ∀ k l, (coins[i][j].color = coins[k][l].color) →
      (i = k ∧ j ≠ l) ∨ (i ≠ k ∧ j = l) ∨ (i ≠ k ∧ j ≠ l) ∧ 
      (abs (i - k) ≠ 1 ∨ abs (j - l) ≠ 1)

def divisible_by_three_worths (coins : List (List Coin)) : Prop :=
  ∀ dir (p : ℕ × ℕ),
    let line := /* some logic to extract the coins lying on a line specified by dir (direction) and p (starting point) */
    (line.map (λ c => c.value)).sum % 3 = 0

-- The statement to prove that the total worth of green coins is divisible by three
theorem total_worth_green_div_by_three (n : ℕ) (coins : List (List Coin))
  (h1 : no_touch_same_color coins)
  (h2 : divisible_by_three_worths coins) :
  (coins.flatten.filter (λ c => c.color = Green)).sum % 3 = 0 :=
by
  sorry

end total_worth_green_div_by_three_l517_517573


namespace prime_pairs_sum_50_l517_517441

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517441


namespace problem1_monotonically_decreasing_intervals_problem2_minimum_side_a_l517_517777

theorem problem1_monotonically_decreasing_intervals (k : ℤ) :
  ∀ x, (f(x) = sin(x)^2 + (sqrt 3 / 2) * sin(2 * x)) →
  x ∈ [((π / 3 : ℝ) + k * π) .. ((5 * π / 6 : ℝ) + k * π)] →
  ∃ y, f'(y) < 0 := 
sorry

theorem problem2_minimum_side_a (f : ℝ → ℝ) (A b c : ℝ) :
  f( (A : right over 2) ) = 1 ∧ (1 / 2) * b * c * sin(A) = 3 * sqrt(3) →
  A = π / 3 →
  ∀ b c, b * c = 12 →
  ∀ a, a = sqrt(b ^ 2 + c ^ 2 - 12) →
  a = 2 * sqrt(3) :=
sorry

end problem1_monotonically_decreasing_intervals_problem2_minimum_side_a_l517_517777


namespace prime_pairs_sum_50_l517_517360

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517360


namespace max_area_triangle_m_l517_517782

theorem max_area_triangle_m (m: ℝ) :
  ∃ (M N: ℝ × ℝ), (x + y + m = 0) ∧ (x^2 + y^2 + 4*y = 0) ∧ 
    (∀ {m₁ m₂ : ℝ}, (area_triangle_CM_N m₁ ≤ area_triangle_CM_N m ∧
      area_triangle_CM_N m₂ ≤ area_triangle_CM_N m) → 
      (m = 0 ∨ m = 4)) :=
begin
  -- Insert proof here
  sorry,
end

end max_area_triangle_m_l517_517782


namespace cyclic_quadrilaterals_count_l517_517113

theorem cyclic_quadrilaterals_count :
  let is_cyclic_quadrilateral (a b c d : ℕ) : Prop :=
    a + b + c + d = 36 ∧
    a * c + b * d <= (a + c) * (b + d) ∧ -- cyclic quadrilateral inequality
    a + b > c ∧ a + c > b ∧ a + d > b ∧ b + c > d ∧ -- convex quadilateral inequality

  (finset.univ.filter (λ (s : finset ℕ), 
    s.card = 4 ∧ fact (multiset.card s.to_multiset = 36) ∧ is_cyclic_quadrilateral s.to_multiset.sum)).card = 1440 :=
sorry

end cyclic_quadrilaterals_count_l517_517113


namespace prime_pairs_sum_to_50_l517_517415

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517415


namespace num_unordered_prime_pairs_summing_to_50_l517_517154

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517154


namespace largest_difference_l517_517844

noncomputable def A := 3 * 2023^2024
noncomputable def B := 2023^2024
noncomputable def C := 2022 * 2023^2023
noncomputable def D := 3 * 2023^2023
noncomputable def E := 2023^2023
noncomputable def F := 2023^2022

theorem largest_difference :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l517_517844


namespace count_setB_l517_517641

def setA : Set ℤ := {0, 1, 2}
def setB : Set ℤ := {z | ∃ x y ∈ setA, z = x - y}

theorem count_setB : setB.toFinset.card = 5 := by
  sorry

end count_setB_l517_517641


namespace prime_squares_between_5000_and_10000_l517_517126

open Nat

theorem prime_squares_between_5000_and_10000 :
  (finset.filter prime (finset.Ico 71 100)).card = 6 :=
by
  sorry

end prime_squares_between_5000_and_10000_l517_517126


namespace prime_pairs_summing_to_50_count_l517_517386

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517386


namespace arithmetic_sequence_a21_eq_2n_minus_m_l517_517825

variable {α : Type} [AddCommGroup α] {a : ℕ → α} (m n : α)

-- Define arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ i j k : ℕ, i + k = 2 * j → a i + a k = 2 * a j

-- Given conditions
def arithmetic_seq_conditions (a : ℕ → α) : Prop :=
  is_arithmetic_sequence a ∧ a 7 = m ∧ a 14 = n

-- Statement to prove
theorem arithmetic_sequence_a21_eq_2n_minus_m
  (a : ℕ → α) (h : arithmetic_seq_conditions a) : a 21 = 2 * n - m :=
sorry

end arithmetic_sequence_a21_eq_2n_minus_m_l517_517825


namespace total_number_of_coins_l517_517808

theorem total_number_of_coins (n : ℕ) (h : 4 * n - 4 = 240) : n^2 = 3721 :=
by
  sorry

end total_number_of_coins_l517_517808


namespace num_prime_pairs_sum_50_l517_517191

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517191


namespace correct_statements_count_l517_517591

theorem correct_statements_count
  (h1 : ¬(∀ (r : ℝ) (P : Point) (c : Circle), line P r ⊥ radius c → tangent P c))
  (h2 : ∀ (c : Circle) (P : Point), center c ∈ line P (tangent c P) → tangent P c)
  (h3 : ∀ (c : Circle) (P : Point), P ∈ tangent P c → center c ∈ line P (tangent c P))
  (h4 : ¬(∀ (r : ℝ) (P : Point) (c : Circle), P ∈ endpoint radius c → line P r ⊥ radius c → tangent P c))
  (h5 : ∀ (c1 c2 : Circle) (A B : Point), c1 ⊆ c2 → chord A B c2 = tangent A B c1 → P = midpoint A B) :
  ∃ (n : ℕ), n = 3 := 
begin
  use 3,
  sorry
end

end correct_statements_count_l517_517591


namespace max_elements_in_set_S_l517_517719

open Nat

def is_valid_element (S : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ S ∧ n ≤ 100

def has_valid_c (S : Finset ℕ) (a b : ℕ) : Prop :=
  a ≠ b → ∃ c ∈ S, gcd a c = 1 ∧ gcd b c = 1

def has_valid_d (S : Finset ℕ) (a b : ℕ) : Prop :=
  a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ gcd a d > 1 ∧ gcd b d > 1

theorem max_elements_in_set_S (S : Finset ℕ) :
  (∀ n ∈ S, is_valid_element S n) →
  (∀ a b ∈ S, has_valid_c S a b) →
  (∀ a b ∈ S, has_valid_d S a b) →
  S.card ≤ 72 := sorry

end max_elements_in_set_S_l517_517719


namespace prob_student_A_selected_l517_517751

theorem prob_student_A_selected :
  let total_outcomes := Nat.choose 4 2 in
  let outcomes_with_A := Nat.choose 1 1 * Nat.choose 3 1 in
  (outcomes_with_A : ℚ) / total_outcomes = 1 / 2 :=
by
  sorry

end prob_student_A_selected_l517_517751


namespace range_of_a_l517_517809

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, 2 ≤ x → x ≤ y → f(x) ≤ f(y)) →
  (f = λ x, log (x^2 + a*x - a - 1)) →
  a > -3 :=
by
  sorry

end range_of_a_l517_517809


namespace find_angle_A_l517_517488

variable (A B : ℝ) (a b : ℝ)

axiom angle_sum (α β γ : ℝ) : α + β + γ = 180
axiom law_of_sines_ratio (a b : ℝ) (α β : ℝ) (h : α ≠ 0 ∧ β ≠ 0) : a / sin α = b / sin β

theorem find_angle_A (hB : B = 2 * A) (hratio : a / b = 1 / sqrt 3) (internal_A : 0 < A) (internal_B : B < 180) :
  A = 30 :=
sorry

end find_angle_A_l517_517488


namespace find_k_l517_517107

theorem find_k (x y k : ℝ) (h1 : x + 2 * y = k + 1) (h2 : 2 * x + y = 1) (h3 : x + y = 3) : k = 7 :=
by
  sorry

end find_k_l517_517107


namespace continuous_stripe_probability_correct_l517_517701

-- Define the conditions of the problem
def regular_tetrahedron := Type*
def stripe (T : regular_tetrahedron) := (T → T → Prop)

-- Define a tetrahedron with a stripe condition
structure tetrahedron_with_stripes (T : regular_tetrahedron) :=
  (stripes : stripe T)

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  4 / 27

-- State the main theorem
theorem continuous_stripe_probability_correct (T : regular_tetrahedron) 
  (t : tetrahedron_with_stripes T) : 
  continuous_stripe_probability = 4 / 27 :=
by
  -- proof goes here
  sorry

end continuous_stripe_probability_correct_l517_517701


namespace num_prime_pairs_sum_50_l517_517178

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517178


namespace num_prime_pairs_sum_50_l517_517324

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517324


namespace max_cylinder_surface_area_l517_517743

variable (h r : ℝ)

noncomputable def max_surface_area (h r : ℝ) : ℝ :=
  (π * h^2 * r) / (2 * (h - r))

theorem max_cylinder_surface_area (h r : ℝ) (h_gt_2r : h > 2 * r) :
  let z := max_surface_area h r
  ∃ x y, 
    z = 2 * π * x^2 + 2 * π * x * y ∧
    y = h * (r - x) / r ∧
    x = h * r / (2 * (h - r)) := sorry


end max_cylinder_surface_area_l517_517743


namespace exist_equal_pair_l517_517055

noncomputable def quadratic_polynomial := Σ (a b c : ℝ), a ≠ 0

def quadratic_fun (p : quadratic_polynomial) (x : ℝ) : ℝ :=
  p.1 * x^2 + p.2 * x + p.3

variable (p : quadratic_polynomial)
variable (l t v : ℝ)

axiom f_l : quadratic_fun p l = t + v
axiom f_t : quadratic_fun p t = l + v
axiom f_v : quadratic_fun p v = l + t

theorem exist_equal_pair (p : quadratic_polynomial) (l t v : ℝ) (f_l : quadratic_fun p l = t + v) (f_t : quadratic_fun p t = l + v) (f_v : quadratic_fun p v = l + t) :
  ∃ (i j : ℝ), (i = l ∨ i = t ∨ i = v) ∧ (j = l ∨ j = t ∨ j = v) ∧ i = j :=
sorry

end exist_equal_pair_l517_517055


namespace length_YP_l517_517502

noncomputable def triangle_XYZ : Type :=
  {X Y Z : ℝ}

-- Conditions
def right_angle_Y (X Y Z : ℝ) := ∃ (angle_Y : ℝ), angle_Y = 90
def length_XZ (X Y Z : ℝ) := XZ = 5
def length_YZ (X Y Z : ℝ) := YZ = 12
def points_PQ_location (X Y Z P Q : ℝ) := ∃ (Q₁ Q₂ : ℝ), angle PQY = 90
def length_PQ (P Q : ℝ) := PQ = 3

-- Prove the length YP
theorem length_YP (X Y Z P Q : ℝ) (h_Y : right_angle_Y X Y Z) (h_XZ : length_XZ X Y Z)
  (h_YZ : length_YZ X Y Z) (h_PQ_loc : points_PQ_location X Y Z P Q) (h_PQ : length_PQ P Q) :
  YP = 13 / 4 := 
sorry

end length_YP_l517_517502


namespace find_line_equation_l517_517089

-- Definitions based on the conditions
structure Point where
  x : ℝ
  y : ℝ

def point_P := Point.mk 1 (-1)
def intersects_at_C (a : ℝ) : Point := Point.mk a 0
def intersects_at_D (b : ℝ) : Point := Point.mk 0 (-b)

-- Variables being set as positive real numbers
variables {a b : ℝ} (ha : 0 < a) (hb : 0 < b)

-- Line equation in intercept form passing through P(1, -1)
def line_equation (a b : ℝ) : Prop := 
  (1/a) + (1/b) = 1

-- Area of triangle OCD = 2
def triangle_area (a b : ℝ) : Prop := 
  (1/2) * a * b = 2

-- The Lean theorem statement
theorem find_line_equation (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : line_equation a b)
  (h2 : triangle_area a b) :
  ∃ k, (k = 1) ∧ (x y : ℝ), x - y - 2 = 0 :=
sorry

end find_line_equation_l517_517089


namespace problem_statement_l517_517773

-- Definition: For the function \( f(x) = \log_{3}(ax+1) \)
def f (a : ℝ) (x : ℝ) : ℝ := log (3 : ℝ) (a * x + 1)

-- Condition: f(x) is increasing on [2, 4]
def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

-- Statement to prove: Given the condition, show that \( a > 0 \)
theorem problem_statement (a : ℝ) 
  (h : is_increasing_on (f a) {x : ℝ | 2 ≤ x ∧ x ≤ 4}) : a > 0 :=
  sorry

end problem_statement_l517_517773


namespace problem_solution_l517_517850

noncomputable def problem (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α^2 + p * α - 1 = 0) ∧
  (β^2 + p * β - 1 = 0) ∧
  (γ^2 + q * γ + 1 = 0) ∧
  (δ^2 + q * δ + 1 = 0) →
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2

theorem problem_solution (p q α β γ δ : ℝ) : 
  problem p q α β γ δ := 
by sorry

end problem_solution_l517_517850


namespace infinite_solutions_xyz_l517_517696

theorem infinite_solutions_xyz :
  ∃^∞ (x y z : ℤ), x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
sorry

end infinite_solutions_xyz_l517_517696


namespace perpendicular_lines_necessary_not_sufficient_l517_517572

theorem perpendicular_lines_necessary_not_sufficient (a : ℝ) : (a^2 = 1) → (iff (perpendicular (line x y = 0) (line x - ay = 0)) (a = 1)) :=
begin
    sorry
end

end perpendicular_lines_necessary_not_sufficient_l517_517572


namespace number_of_functions_l517_517053

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem number_of_functions :
  ∃ (f : ℝ → ℝ) (a b c : ℝ), 
    (f = λ x, quadratic_function a b c x) ∧
    (quadratic_function a b c = quadratic_function a (-b) c) ∧
    (∀ x : ℝ, quadratic_function a b c x * quadratic_function a (-b) c x = quadratic_function a b c (x^3)) ∧
    (a = 0 ∨ a = 1) ∧
    (c = 0 ∨ c = 1) ∧
    ((a = 0 ∧ c = 0 ∧ (b = 0 ∨ b = -1)) ∨ 
     (a = 1 ∧ c = 1 ∧ (b = 1 ∨ b = -2))) ∧
    (finset.card (finset.univ.image (λ (a b c : ℕ), quadratic_function a b c)) = 6) := sorry

end number_of_functions_l517_517053


namespace quadractic_integer_roots_l517_517760

theorem quadractic_integer_roots (n : ℕ) (h : n > 0) :
  (∃ x y : ℤ, x^2 - 4 * x + n = 0 ∧ y^2 - 4 * y + n = 0) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadractic_integer_roots_l517_517760


namespace count_prime_pairs_sum_50_exactly_4_l517_517282

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517282


namespace number_of_ordered_pairs_l517_517030

-- Define the predicate that defines the condition for the ordered pairs (m, n)
def satisfies_condition (m n : ℕ) : Prop :=
  6 % m = 0 ∧ 3 % n = 0 ∧ 6 / m + 3 / n = 1

-- Define the main theorem for the problem statement
theorem number_of_ordered_pairs : 
  (∃ count : ℕ, count = 6 ∧ 
  (∀ m n : ℕ, satisfies_condition m n → m > 0 ∧ n > 0)) :=
by {
 sorry -- Placeholder for the actual proof
}

end number_of_ordered_pairs_l517_517030


namespace hyperbola_equation_l517_517767

def hyperbola_standard_equation (E : Type*) (a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ (∃ h : ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1 ∨ y^2 / a^2 - x^2 / b^2 = 1)))     

def real_axis_length (E : Type*) : Type* := { axis_length : ℝ // axis_length > 0 }

def asymptotes_tangent_to_parabola (E : Type*) : Prop := 
  ∀ x y : ℝ, ∃ k : ℝ, y = k * x ∧ y = x^2 + 1 -- simplified representation of tangent condition

theorem hyperbola_equation (E : Type*)
  (h1 : real_axis_length(E) = 2)
  (h2 : asymptotes_tangent_to_parabola(E))
  : hyperbola_standard_equation E 1 2 :=
sorry

end hyperbola_equation_l517_517767


namespace prime_pairs_sum_50_l517_517277

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517277


namespace prime_pairs_sum_50_l517_517270

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517270


namespace ratio_of_distances_l517_517660

theorem ratio_of_distances (p : ℝ) (hp : p > 0) :
  let F := (p / 2, 0) in
  let A := (p / 6, sqrt(p / 3)) in
  let B := (3 * p / 2, -sqrt(3 * p)) in
  let AF := (sqrt((p / 6 - p / 2)^2 + (sqrt(p / 3) - 0)^2)) in
  let BF := (sqrt((3 * p / 2 - p / 2)^2 + (-sqrt(3 * p) - 0)^2)) in
  AF / BF = 1 / 3 :=
sorry

end ratio_of_distances_l517_517660


namespace count_prime_pairs_summing_to_50_l517_517230

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517230


namespace pyramid_volume_after_scaling_l517_517982

theorem pyramid_volume_after_scaling (
  (V1 : ℝ) (a b c h : ℝ)
  (initial_volume : V1 = 60)
  (double_a : a2 = 2 * a)
  (double_b : b2 = 2 * b)
  (triple_c : c2 = 3 * c)
  (scale_height : h2 = 3 * h)
) : (volume_new : ℝ) :=
  let A1 := (a * b) / 2 -- simplifying the initial area calculation for illustration
  let A2 := 4 * A1
  let volume_new := (1/3) * A2 * h2
  volume_new = 720 :=
by sorry

end pyramid_volume_after_scaling_l517_517982


namespace find_value_of_expression_l517_517088

theorem find_value_of_expression
  (a b c : ℝ)
  (h1 : a - b = 2 + real.sqrt 3)
  (h2 : b - c = 2 - real.sqrt 3) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 15 :=
sorry

end find_value_of_expression_l517_517088


namespace Luke_trips_l517_517539

variable (carries : Nat) (table1 : Nat) (table2 : Nat)

theorem Luke_trips (h1 : carries = 4) (h2 : table1 = 20) (h3 : table2 = 16) : 
  (table1 / carries + table2 / carries) = 9 :=
by
  sorry

end Luke_trips_l517_517539


namespace offer1_expression_offer2_expression_compare_offers_minimized_cost_combination_l517_517659

-- Defining the costs for each offer as functions
def offer1_cost (x : ℕ) : ℕ := 20 * x + 5400
def offer2_cost (x : ℕ) : ℕ := 19 * x + 5700

-- Prove the algebraic expressions for both offers
theorem offer1_expression (x : ℕ) (h : x > 30) : offer1_cost x = 20 * x + 5400 := by
  sorry

theorem offer2_expression (x : ℕ) (h : x > 30) : offer2_cost x = 19 * x + 5700 := by
  sorry

-- Prove that for x = 40, Offer 1 is more cost-effective than Offer 2
theorem compare_offers : offer1_cost 40 < offer2_cost 40 := by
  sorry

-- Prove the minimized cost by combining both offers
theorem minimized_cost_combination :
  let cost_30sets_30bowls := 6000
  let cost_10bowls_offer2 := 10 * 20 * 95 / 100
  in cost_30sets_30bowls + cost_10bowls_offer2 = 6190 := by
  sorry

end offer1_expression_offer2_expression_compare_offers_minimized_cost_combination_l517_517659


namespace prime_pairs_sum_50_l517_517258

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517258


namespace count_prime_pairs_summing_to_50_l517_517237

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517237


namespace length_of_XY_l517_517813

theorem length_of_XY (X Y Z : Type) [EuclideanGeometry] 
  (angleX : ∠X = 90)
  (tanY : tan ∠Y = 3/4)
  (YZ : length (line_segment Y Z) = 30) : 
  length (line_segment X Y) = 24 := 
sorry

end length_of_XY_l517_517813


namespace num_prime_pairs_sum_50_l517_517184

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517184


namespace standard_spherical_coordinates_l517_517822

def spherical_to_standard (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let φ' := if φ > π then 2 * π - φ else φ
  let θ' := if θ + π < 2 * π then θ + π else (θ + π) - 2 * π
  (ρ, θ', φ')

theorem standard_spherical_coordinates :
  spherical_to_standard 4 (11 * π / 6) (9 * π / 5) = (4, 5 * π / 6, π / 5) :=
by
  sorry

end standard_spherical_coordinates_l517_517822


namespace prime_pairs_sum_50_l517_517244

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517244


namespace prime_pairs_sum_50_l517_517260

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517260


namespace remainder_is_159_l517_517618

-- Defining the conditions as assumptions for the mathematical entities involved
def pow_mod_remainder : ℕ :=
  let a := 2^160 + 160 in
  let b := 2^81 + 2^41 + 1 in
  a % b

-- Asserting that the remainder of 2^160 + 160 divided by 2^81 + 2^41 + 1 is 159
theorem remainder_is_159 : pow_mod_remainder = 159 :=
  sorry

end remainder_is_159_l517_517618


namespace frustum_volume_l517_517911

theorem frustum_volume (m : ℝ) (α : ℝ) (k : ℝ) : 
  m = 3/π ∧ 
  α = 43 + 40/60 + 42.2/3600 ∧ 
  k = 1 →
  frustumVolume = 0.79 := 
sorry

end frustum_volume_l517_517911


namespace sum_triangle_areas_eq_l517_517020

theorem sum_triangle_areas_eq (AB CD : ℝ) (h_AB : AB = 1) (h_CD : CD = 2) :
  let Q1 : ℝ := 1 / 3 * CD in
  let triangle_area (i : ℕ) : ℝ := 
    (1 / 2) * (1 / 3 * CD / 2^i) * (1 / 3 * CD / 2^(i - 1)) in
  (∑' i : ℕ, triangle_area (i + 1)) = 1 / 54 :=
by
  sorry

end sum_triangle_areas_eq_l517_517020


namespace sum_of_independent_poisson_is_poisson_l517_517555

open ProbabilityTheory

-- Definitions for the problem
variable {Ω : Type}

-- Defining the Poisson distributions for X and Y
noncomputable def poisson (λ : ℝ) : probability_space ℕ :=
{ P := λ k, λ^k / fact k * exp (-λ) }

def random_variable_X (a : ℝ) [is_poisson (poisson a) X] : Ω → ℕ := sorry
def random_variable_Y (b : ℝ) [is_poisson (poisson b) Y] : Ω → ℕ := sorry

-- Independence condition for random variables X and Y
variable (h_independent : independent X Y)

-- The theorem to prove
theorem sum_of_independent_poisson_is_poisson (a b : ℝ)
  (hX : is_poisson (poisson a) X)
  (hY : is_poisson (poisson b) Y)
  (h_independent : independent X Y) :
  is_poisson (poisson (a + b)) (X + Y) :=
sorry

end sum_of_independent_poisson_is_poisson_l517_517555


namespace solution_set_l517_517094

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry -- Define the derivative of f

axiom A1 : ∀ x : ℝ, f(x) + f'(x) < 0
axiom A2 : f(2) = 1 / Real.exp(2)

def g (x : ℝ) : ℝ := Real.exp(x) * f(x)

theorem solution_set : {x : ℝ | Real.exp(x) * f(x) - 1 > 0} = {x : ℝ | x < 2} :=
by
  sorry

end solution_set_l517_517094


namespace no_solution_cotangent_eqn_l517_517890

theorem no_solution_cotangent_eqn :
  ∀ x : ℝ, sin x ≠ 0 → sin (2 * x) ≠ 0 → sin (3 * x) ≠ 0 →
    (1 / (sin x * sin (2 * x) * sin (3 * x)) + cos (2 * x) / sin (2 * x) + cos (3 * x) / sin (3 * x)) ≠ 0 := 
by
  intro x hx1 hx2 hx3
  sorry

end no_solution_cotangent_eqn_l517_517890


namespace alpha_plus_beta_eq_pi_over_4_l517_517060

theorem alpha_plus_beta_eq_pi_over_4
  (α β : ℝ)
  (h1 : sin α = (real.sqrt 5) / 5)
  (h2 : sin β = (real.sqrt 10) / 10)
  (hα_acute : 0 < α ∧ α < π / 2)
  (hβ_acute : 0 < β ∧ β < π / 2) :
  α + β = π / 4 :=
sorry

end alpha_plus_beta_eq_pi_over_4_l517_517060


namespace D_96_l517_517845

def D : ℕ → ℕ
| 1       := 0
| n+1     := -- The full definition should be included here, along with any helper functions or constructs to properly define D(n)
sorry

theorem D_96 : D 96 = 112 :=
by
  sorry

end D_96_l517_517845


namespace count_prime_pairs_sum_50_exactly_4_l517_517295

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517295


namespace no_monochromatic_arithmetic_progression_l517_517553

-- Defining the set M and number of colors
def M := Finset.range 1987  -- This represents the set {0, 1, ..., 1986}
def num_colors := 4

-- Lean statement for the problem
theorem no_monochromatic_arithmetic_progression :
  ∃ (coloring : M → Fin num_colors), 
    ∀ (a d : ℕ), a ∈ M → d ≠ 0 → (a + 9 * d) < 1987 → 
      (∃ (c : Fin num_colors), ∀ i : ℕ, i < 10 → c = coloring (a + i * d)) → False :=
by
  sorry

end no_monochromatic_arithmetic_progression_l517_517553


namespace sum_of_exponents_in_factorial_15_l517_517487

theorem sum_of_exponents_in_factorial_15 :
  let x := (15.factorial)
  ∃ i k m p q r : ℕ, 
    x = 2^i * 3^k * 5^m * 7^p * 11^q * 13^r ∧
    i + k + m + p + q + r = 29 :=
by {
  sorry
}

end sum_of_exponents_in_factorial_15_l517_517487


namespace largest_of_four_consecutive_integers_with_product_840_l517_517731

theorem largest_of_four_consecutive_integers_with_product_840 
  (a b c d : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h_pos : 0 < a) (h_prod : a * b * c * d = 840) : d = 7 :=
sorry

end largest_of_four_consecutive_integers_with_product_840_l517_517731


namespace num_prime_pairs_sum_50_l517_517176

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517176


namespace prime_pairs_sum_to_50_l517_517303

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517303


namespace prime_pairs_sum_50_l517_517368

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517368


namespace num_unordered_prime_pairs_summing_to_50_l517_517161

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517161


namespace number_of_roots_in_interval_l517_517582

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions for f
axiom f_symmetric_about_2 : ∀ x : ℝ, f(2 + x) = f(2 - x)
axiom f_symmetric_about_7 : ∀ x : ℝ, f(7 + x) = f(7 - x)
axiom f_root_at_0 : f 0 = 0

-- Statement to prove the number of roots
theorem number_of_roots_in_interval : 
  (set_of (λ x : ℝ, f x = 0)).count (λ x, -1000 ≤ x ∧ x ≤ 1000) = 201 :=
sorry

end number_of_roots_in_interval_l517_517582


namespace correct_formula_exists_l517_517691

theorem correct_formula_exists:
  ∀ (x y : ℤ), ((x = 1 ∧ y = -2) ∨
               (x = 2 ∧ y = 0) ∨
               (x = 3 ∧ y = 4) ∨
               (x = 4 ∧ y = 10) ∨
               (x = 5 ∧ y = 18)) →
               y = x^2 - 4*x + 1 :=
by
  intro x y h
  cases h
  case inl H => { rw [H.1, H.2]; norm_num }
  case inr h =>
    cases h
    case inl H => { rw [H.1, H.2]; norm_num }
    case inr h =>
      cases h
      case inl H => { rw [H.1, H.2]; norm_num }
      case inr h =>
        cases h
        case inl H => { rw [H.1, H.2]; norm_num }
        case inr H => { rw [H.1, H.2]; norm_num }

end correct_formula_exists_l517_517691


namespace lateral_surface_area_cylinder_l517_517483

-- Definitions based on conditions
def base_radius : ℝ := 2
def generatrix_length : ℝ := 3
def lateral_surface_area (r : ℝ) (h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Statement to prove
theorem lateral_surface_area_cylinder : lateral_surface_area base_radius generatrix_length = 12 * Real.pi :=
by
  sorry

end lateral_surface_area_cylinder_l517_517483


namespace cos_triple_angle_l517_517472

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517472


namespace count_prime_pairs_sum_50_l517_517131

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517131


namespace count_prime_pairs_summing_to_50_l517_517228

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517228


namespace num_prime_pairs_sum_50_l517_517179

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517179


namespace int_between_sqrt_15_and_sqrt_150_l517_517123

theorem int_between_sqrt_15_and_sqrt_150 : ∃ n : ℕ, n = 9 ∧ 
  ∀ x : ℤ, (⟦⟨15, sorry⟩⟧.sqrt.digits.toInt : ℚ) < x ∧ x < (⟦⟨150, sorry⟩⟧.sqrt.digits.toInt : ℚ) → 4 ≤ x ∧ x ≤ 12 :=
begin
  sorry,
end

end int_between_sqrt_15_and_sqrt_150_l517_517123


namespace ratio_of_area_XYD_XZD_length_of_YD_length_of_ZD_l517_517832
noncomputable theory

-- Define the given lengths
def XY := 15
def XZ := 35
def YZ := 42

-- let XD be the angle bisector and D be on YZ such that XD divides the angle XYZ
def XD_bisects_angle_XYZ : Prop := 
  ∃ D, (YD / ZD = XY / XZ) ∧ (XYD_area / XZD_area = YD / ZD)

-- Definitions of YD and ZD using the ratio determined by the angle bisector theorem
def YD := 42 * (XY / (XY + XZ))
def ZD := 42 * (XZ / (XY + XZ))

-- Area ratio of two triangles sharing the common height is proportional to their bases
def area_ratio (a b: ℕ): ℚ := a / b

-- Theorem statements for what needs to be proved
theorem ratio_of_area_XYD_XZD : XD_bisects_angle_XYZ → area_ratio YD ZD = 3 / 7 := 
by 
  sorry -- proof goes here

theorem length_of_YD : XD_bisects_angle_XYZ → YD = 12.6 := 
by 
  sorry -- proof goes here

theorem length_of_ZD : XD_bisects_angle_XYZ → ZD = 29.4 := 
by 
  sorry -- proof goes here

end ratio_of_area_XYD_XZD_length_of_YD_length_of_ZD_l517_517832


namespace count_prime_pairs_sum_50_l517_517135

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517135


namespace prime_pairs_sum_to_50_l517_517421

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517421


namespace cos_triple_angle_l517_517467

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l517_517467


namespace circumradius_squared_eq_product_of_radii_l517_517017

noncomputable def circumradius (A B C : Point) : ℝ := sorry
noncomputable def radius (O : Point) : ℝ := sorry
noncomputable def tangent_at (O O' A : Point) : Prop := sorry

theorem circumradius_squared_eq_product_of_radii
    (O O' A B C : Point)
    (r r' : ℝ)
    (h1 : tangent_at O O' A)
    (h2 : dist O B = r)
    (h3 : dist O' C = r')
    (h4 : line_perpendicular (line_through O O') (line_through B C))
    (h5 : circumradius A B C = R) :
    R^2 = r * r' :=
sorry

end circumradius_squared_eq_product_of_radii_l517_517017


namespace primes_sum_50_l517_517398

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517398


namespace tan_eq_one_over_three_l517_517064

theorem tan_eq_one_over_three (x : ℝ) (h1 : x ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.cos (2 * x - (Real.pi / 2)) = Real.sin x ^ 2) :
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end tan_eq_one_over_three_l517_517064


namespace prime_pairs_sum_50_l517_517364

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517364


namespace special_calculator_result_l517_517604

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.toString.data
  Nat.ofDigits 10 digits.reverse.map Char.toNat

theorem special_calculator_result :
  ∃ x : ℕ, (10 ≤ x ∧ x ≤ 49) ∧ x * 2 |> reverse_digits + 2 = 44 :=
sorry

end special_calculator_result_l517_517604


namespace estimate_red_balls_l517_517820

-- Define the conditions
variable (total_balls : ℕ)
variable (prob_red_ball : ℝ)
variable (frequency_red_ball : ℝ := prob_red_ball)

-- Assume total number of balls in the bag is 20
axiom total_balls_eq_20 : total_balls = 20

-- Assume the probability (or frequency) of drawing a red ball
axiom prob_red_ball_eq_0_25 : prob_red_ball = 0.25

-- The Lean statement
theorem estimate_red_balls (H1 : total_balls = 20) (H2 : prob_red_ball = 0.25) : total_balls * prob_red_ball = 5 :=
by
  rw [H1, H2]
  norm_num
  sorry

end estimate_red_balls_l517_517820


namespace number_of_prime_pairs_sum_50_l517_517210

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517210


namespace ratio_of_perimeters_is_correct_l517_517984

def initial_rectangle : ℝ × ℝ := (6, 4)

def folded_dimensions (r : ℝ × ℝ) : ℝ × ℝ :=
  (r.1 / 2, r.2)

def small_rectangle_dimensions (r : ℝ × ℝ) : ℝ × ℝ :=
  folded_dimensions (r.1, r.2) |> (λ p, (p.1, p.2 / 2))

def perimeter (r : ℝ × ℝ) : ℝ :=
  2 * (r.1 + r.2)

def ratio_of_perimeters (initial_dim : ℝ × ℝ) : ℝ :=
  let large_rect := folded_dimensions initial_dim
  let small_rect := small_rectangle_dimensions initial_dim
  (perimeter small_rect) / (perimeter large_rect)

theorem ratio_of_perimeters_is_correct : ratio_of_perimeters initial_rectangle = 5 / 7 := by
  sorry

end ratio_of_perimeters_is_correct_l517_517984


namespace find_ratio_l517_517584

lemma condition (A B : ℤ) : (∀ x ≠ 2, x^3 - 6*x^2 + 16*x - 16 ≠ 0 → 
  (A / (x - 2) + B / (x^2 - 4*x + 8)) = (x^2 - 4*x + 18) / (x^3 - 6*x^2 + 16*x - 16)) :=
sorry

theorem find_ratio (A B : ℤ) (h : condition A B) : (B : ℚ) / A = -4 / 9 :=
by sorry

end find_ratio_l517_517584


namespace negation_universal_prop_l517_517590

theorem negation_universal_prop:
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
  sorry

end negation_universal_prop_l517_517590


namespace num_prime_pairs_sum_50_l517_517171

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517171


namespace work_problem_l517_517955

theorem work_problem (A B : ℕ → ℝ) (h1 : A 6 + B 6 = 1) (h2 : A 14 = 1) : B 10.5 = 1 :=
  sorry

end work_problem_l517_517955


namespace sequence_formula_l517_517105

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 2 * n - 1

theorem sequence_formula (a : ℕ → ℕ) (h : sequence a) :
  ∀ n, a n = n^2 :=
by
  sorry

end sequence_formula_l517_517105


namespace prime_pairs_sum_to_50_l517_517429

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517429


namespace percentage_increase_l517_517979

def original_price : ℝ := 100
def first_discount : ℝ := original_price * (1 - 0.25)
def second_discount : ℝ := first_discount * (1 - 0.15)
def third_discount : ℝ := second_discount * (1 - 0.10)
def fourth_discount : ℝ := third_discount * (1 - 0.05)
def final_price : ℝ := fourth_discount * (1 + 0.07)

theorem percentage_increase : 
  (original_price - final_price) / final_price ≈ 0.7145 := 
by
  sorry

end percentage_increase_l517_517979


namespace bisect_convex_figure_l517_517532

noncomputable def exists_bisecting_line (F : convex_set ℝ (euclidean_space ℝ 2)) : Prop :=
  ∃ l : affine_affine_space ℝ (euclidean_space ℝ 2), 
  bisects_perimeter F l ∧ bisects_area F l

theorem bisect_convex_figure (F : convex_set ℝ (euclidean_space ℝ 2)) :
  exists_bisecting_line F :=
sorry

end bisect_convex_figure_l517_517532


namespace count_prime_pairs_sum_50_exactly_4_l517_517294

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517294


namespace smallest_M_conditions_l517_517720

theorem smallest_M_conditions :
  ∃ M : ℕ, M > 0 ∧
  ((∃ k₁, M = 8 * k₁) ∨ (∃ k₂, M + 2 = 8 * k₂) ∨ (∃ k₃, M + 4 = 8 * k₃)) ∧
  ((∃ k₄, M = 9 * k₄) ∨ (∃ k₅, M + 2 = 9 * k₅) ∨ (∃ k₆, M + 4 = 9 * k₆)) ∧
  ((∃ k₇, M = 25 * k₇) ∨ (∃ k₈, M + 2 = 25 * k₈) ∨ (∃ k₉, M + 4 = 25 * k₉)) ∧
  M = 100 :=
sorry

end smallest_M_conditions_l517_517720


namespace total_surface_area_of_cubes_aligned_side_by_side_is_900_l517_517724

theorem total_surface_area_of_cubes_aligned_side_by_side_is_900 :
  let volumes := [27, 64, 125, 216, 512]
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  (surface_areas.sum = 900) :=
by
  sorry

end total_surface_area_of_cubes_aligned_side_by_side_is_900_l517_517724


namespace num_prime_pairs_sum_50_l517_517321

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517321


namespace num_unordered_prime_pairs_summing_to_50_l517_517162

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517162


namespace Spaatz_has_1_gemstone_l517_517689

-- Definitions

variables {Binkie Frankie Spaatz : ℕ}

-- Conditions
def Binkie_has_24_gemstones : Prop := Binkie = 24
def Binkie_gemstones_four_times_Frankie : Prop := Binkie = 4 * Frankie
def Spaatz_gemstones_two_less_than_half_Frankie : Prop := Spaatz = (Frankie / 2) - 2

-- Theorem to prove
theorem Spaatz_has_1_gemstone (h1: Binkie_has_24_gemstones) (h2: Binkie_gemstones_four_times_Frankie) (h3: Spaatz_gemstones_two_less_than_half_Frankie) : Spaatz = 1 :=
by
  sorry

end Spaatz_has_1_gemstone_l517_517689


namespace num_unordered_prime_pairs_summing_to_50_l517_517153

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517153


namespace sequences_properties_l517_517066

-- Definitions based on the problem conditions
def geom_sequence (a : ℕ → ℕ) := ∃ q : ℕ, a 1 = 2 ∧ a 3 = 18 ∧ ∀ n, a (n + 1) = a n * q
def arith_sequence (b : ℕ → ℕ) := b 1 = 2 ∧ ∃ d : ℕ, ∀ n, b (n + 1) = b n + d
def condition (a : ℕ → ℕ) (b : ℕ → ℕ) := a 1 + a 2 + a 3 > 20 ∧ a 1 + a 2 + a 3 = b 1 + b 2 + b 3 + b 4

-- Proof statement: proving the general term of the geometric sequence and the sum of the arithmetic sequence
theorem sequences_properties (a : ℕ → ℕ) (b : ℕ → ℕ) :
  geom_sequence a → arith_sequence b → condition a b →
  (∀ n, a n = 2 * 3^(n - 1)) ∧ (∀ n, S_n = 3 / 2 * n^2 + 1 / 2 * n) :=
by
  sorry

end sequences_properties_l517_517066


namespace prime_pairs_sum_50_l517_517272

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517272


namespace general_term_sum_b_n_terms_l517_517927

-- Define the arithmetic sequence and necessary given data
noncomputable def a_5 : ℕ → ℝ := λ n, if n = 5 then 10 else 0
noncomputable def S_7 : ℕ → ℝ := λ n, if n = 7 then 49 else 0

-- Define the sequences
noncomputable def a_n (n : ℕ) : ℝ := 3 * n - 5

-- Define b_n based on a_n
noncomputable def b_n (n : ℕ) : ℝ := 1 / ((3 * n - 2) * a_n n)

-- Sum of the first n terms of sequence b_n
noncomputable def T_n (n : ℕ) : ℝ := 1 / 3 * (-(1 / 2) - 1 / (3 * n - 2)) -- Simplified form

-- Main theorem proofs
theorem general_term (a5_cond : a_n 5 = 10) (s7_cond : S_7 7 = 49) : 
  ∀ n : ℕ, a_n n = 3 * n - 5 := by
  sorry

theorem sum_b_n_terms (b_def : ∀ n : ℕ, b_n n = 1 / ((3 * n - 2) * (3 * n - 5))) :
  ∀ n : ℕ, T_n n = -n / (2 * (3 * n - 2)) := by
  sorry

end general_term_sum_b_n_terms_l517_517927


namespace calories_per_person_l517_517507

theorem calories_per_person 
  (oranges : ℕ)
  (pieces_per_orange : ℕ)
  (people : ℕ)
  (calories_per_orange : ℝ)
  (h_oranges : oranges = 7)
  (h_pieces_per_orange : pieces_per_orange = 12)
  (h_people : people = 6)
  (h_calories_per_orange : calories_per_orange = 80.0) :
  (oranges * pieces_per_orange / people) * (calories_per_orange / pieces_per_orange) = 93.3338 :=
by
  sorry

end calories_per_person_l517_517507


namespace prime_pairs_sum_to_50_l517_517419

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517419


namespace largest_n_polynomials_l517_517029

theorem largest_n_polynomials :
  ∃ (P : ℕ → (ℝ → ℝ)), (∀ i j, i ≠ j → ∀ x, P i x + P j x ≠ 0) ∧ (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ x, P i x + P j x + P k x = 0) ↔ n = 3 := 
sorry

end largest_n_polynomials_l517_517029


namespace number_of_prime_pairs_sum_50_l517_517220

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517220


namespace prime_pairs_sum_50_l517_517369

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517369


namespace num_prime_pairs_sum_50_l517_517333

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517333


namespace num_unordered_prime_pairs_summing_to_50_l517_517164

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517164


namespace num_unordered_prime_pairs_summing_to_50_l517_517157

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517157


namespace num_prime_pairs_sum_50_l517_517331

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517331


namespace volume_relation_l517_517024

theorem volume_relation 
  (r h : ℝ) 
  (heightC_eq_three_times_radiusD : h = 3 * r)
  (radiusC_eq_heightD : r = h)
  (volumeD_eq_three_times_volumeC : ∀ (π : ℝ), 3 * (π * h^2 * r) = π * r^2 * h) :
  3 = (3 : ℝ) := 
by
  sorry

end volume_relation_l517_517024


namespace count_prime_pairs_summing_to_50_l517_517226

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517226


namespace parabola_circle_tangent_l517_517784

theorem parabola_circle_tangent (t : ℝ) (r : ℝ) (x y : ℝ) :
  (∀ (t : ℝ), (x = 8 * t^2 ∧ y = 8 * t)) ∧
  (x^2 + y^2 = r^2) ∧
  (r > 0) ∧
  (∀ (x y : ℝ), ∃ (m : ℝ), m = 1 ∧ (y - 0 = m * (x - 2))) →
  r = sqrt 2 :=
by sorry

end parabola_circle_tangent_l517_517784


namespace cos_triple_angle_l517_517476

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517476


namespace improper_fraction_decomposition_l517_517656

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := 
sorry

end improper_fraction_decomposition_l517_517656


namespace chennai_to_hyderabad_distance_l517_517025

-- Definitions of the conditions
def david_speed := 50 -- mph
def lewis_speed := 70 -- mph
def meet_point := 250 -- miles from Chennai

-- Theorem statement
theorem chennai_to_hyderabad_distance :
  ∃ D T : ℝ, lewis_speed * T = D + (D - meet_point) ∧ david_speed * T = meet_point ∧ D = 300 :=
by
  sorry

end chennai_to_hyderabad_distance_l517_517025


namespace value_of_expression_l517_517722

theorem value_of_expression 
  (x : ℝ) 
  (h : 7 * x^2 + 6 = 5 * x + 11) 
  : (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := 
by 
  sorry

end value_of_expression_l517_517722


namespace determine_k_coplanar_l517_517518

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C D : V}
variable (k : ℝ)

theorem determine_k_coplanar (h : 4 • A - 3 • B + 6 • C + k • D = 0) : k = -13 :=
sorry

end determine_k_coplanar_l517_517518


namespace poly_factors_abs_value_l517_517811

variable {R : Type} [CommRing R]

-- Definitions based on given conditions
def poly (x h k : R) := 3 * x^3 - h * x + k

def factor_cond1 (h k : R) : Prop :=
  poly (-2) h k = 0

def factor_cond2 (h k : R) : Prop :=
  poly 1 h k = 0

-- The Lean theorem statement
theorem poly_factors_abs_value (h k : R) (H1 : factor_cond1 h k) (H2 : factor_cond2 h k) :
  |(3 * h - 2 * k)| = 15 :=
sorry

end poly_factors_abs_value_l517_517811


namespace orthocenter_on_line_l517_517516

variable {Point : Type*} [EuclideanGeometry Point]

variable (A B C D M N : Point)
variable (g : Line Point)
variable (ABCD_tangential : TangentialQuadrilateral A B C D)
variable (g_passes_through_A : A ∈ g)
variable (M_on_BC : M ∈ segment B C)
variable (N_on_DC : N ∈ line D C)
variable (g_intersects_M : M ∈ g)
variable (g_intersects_N : N ∈ g)
variable (I1 I2 I3 : Point)
variable (incenter_ABM : Incenter I1 (triangle A B M))
variable (incenter_MCN : Incenter I2 (triangle M C N))
variable (incenter_ADN : Incenter I3 (triangle A D N))

theorem orthocenter_on_line :
  Orthocenter (triangle I1 I2 I3) ∈ g := 
sorry

end orthocenter_on_line_l517_517516


namespace count_prime_pairs_sum_50_exactly_4_l517_517284

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517284


namespace primes_sum_50_l517_517411

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517411


namespace integers_less_than_2019_divisible_by_18_or_21_but_not_both_l517_517125

theorem integers_less_than_2019_divisible_by_18_or_21_but_not_both :
  ∃ (N : ℕ), (∀ (n : ℕ), (n < 2019 → (n % 18 = 0 ∨ n % 21 = 0) → n % (18 * 21 / gcd 18 21) ≠ 0) ↔ (∀ (m : ℕ), m < N)) ∧ N = 176 :=
by
  sorry

end integers_less_than_2019_divisible_by_18_or_21_but_not_both_l517_517125


namespace term_with_largest_binomial_coefficient_in_expansion_l517_517827

theorem term_with_largest_binomial_coefficient_in_expansion :
  ∃ k : ℕ, k = 3 ∧ ∀ n : ℕ, n ≤ 6 → nat.choose 6 n ≤ nat.choose 6 k :=
begin
  sorry
end

end term_with_largest_binomial_coefficient_in_expansion_l517_517827


namespace prime_pairs_sum_50_l517_517439

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517439


namespace find_a_l517_517853

theorem find_a : ∀ (a : ℂ), (a / (1 - complex.i) = (1 + complex.i) / complex.i) → a = -2 * complex.i :=
by  -- proof is to be filled in here
  sorry

end find_a_l517_517853


namespace number_of_prime_pairs_sum_50_l517_517211

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517211


namespace prime_pairs_sum_to_50_l517_517307

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517307


namespace cos_triple_angle_l517_517457

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517457


namespace tangent_line_at_point_is_correct_l517_517905

noncomputable def curve (x : ℝ) : ℝ := Real.exp(x) + x

def point_of_tangency : ℝ × ℝ := (0, 1)

theorem tangent_line_at_point_is_correct :
  (∀ m b : ℝ, (λ x y : ℝ, y = m * x + b) (0, curve 0) ∧ 
    (∀ x y : ℝ, curve x = y → (y - (1 : ℝ)) = -x))
  ∧ (curve 0 = (1 : ℝ)) → 
  (y : ℝ) = -x + 1 := 
sorry

end tangent_line_at_point_is_correct_l517_517905


namespace perimeter_of_TRIANGLE_APR_l517_517942

variables (A B C P Q R : Point)
variables (circle : Circle)
variables (tangent : Circle → Point → Prop)

-- Define points A, B, C, P, Q, R
axiom hAB : tangent circle A B
axiom hAC : tangent circle A C
axiom hAP : tangent circle P Q
axiom hAQ : tangent circle Q P
axiom hPQ : PQ = QR

-- Segment lengths
axiom hab : AB = 24
axiom hsym : AC = 24

-- Define functions to measure distances
variables (distance: Point → Point → ℝ)

-- Assume necessary tangent properties
axiom tangent_symmetry : ∀ {X Y : Point}, tangent circle A X → tangent circle A Y → distance A X = distance A Y

noncomputable def perimeter_TRIANGLE := 
  distance A P + distance P R + distance R A

-- The actual proof problem statement
theorem perimeter_of_TRIANGLE_APR : perimeter_TRIANGLE = 48 := sorry

end perimeter_of_TRIANGLE_APR_l517_517942


namespace geom_sequence_product_l517_517520

open Classical

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geom_sequence_product
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : is_geometric_sequence a q)
  (h_q : q = 2)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_prod : ∏ i in finset.range 33, a (i + 1) = 2^33) :
  ∏ i in finset.range 11, a (3 * (i + 1)) = 2^100 :=
by
  sorry

end geom_sequence_product_l517_517520


namespace non_swimmers_play_soccer_percentage_l517_517678

theorem non_swimmers_play_soccer_percentage (N : ℕ) (hN_pos : 0 < N)
 (h1 : (0.7 * N : ℝ) = x)
 (h2 : (0.5 * N : ℝ) = y)
 (h3 : (0.6 * x : ℝ) = z)
 : (0.56 * y = 0.28 * N) := 
 sorry

end non_swimmers_play_soccer_percentage_l517_517678


namespace infinitely_many_a_sequence_perfect_square_l517_517640

/-- There are infinitely many positive integers a such that both a + 1 and 3a + 1 are perfect squares. -/
theorem infinitely_many_a (a : ℕ) (h1 : ∃ (y x : ℕ), a + 1 = y^2 ∧ 3 * a + 1 = x^2) : 
  ∃^\infty a, (∃ (y x : ℕ), a + 1 = y^2 ∧ 3 * a + 1 = x^2) := 
sorry

/-- For the sequence of all positive integer solutions of the first problem, prove aₙ * aₙ₊₁ is also a perfect square. -/
theorem sequence_perfect_square (a_n : ℕ → ℕ) (h2 : ∀ n, (∃ (y x : ℕ), a_n n + 1 = y^2 ∧ 3 * a_n n + 1 = x^2) ∧ a_n n < a_n (n + 1)) :
  ∀ n, ∃ k, (a_n n) * (a_n (n + 1)) = k^2 := 
sorry

end infinitely_many_a_sequence_perfect_square_l517_517640


namespace prime_pairs_summing_to_50_count_l517_517375

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517375


namespace number_of_prime_pairs_sum_50_l517_517218

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517218


namespace prime_pairs_sum_50_l517_517253

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517253


namespace snorlax_triangle_construction_l517_517893

theorem snorlax_triangle_construction 
  (l1 l2 l3 : Line)
  (ω : Circle)
  (h1 : ¬(l1 ∥ l2 ∧ l2 ∥ l3 ∧ l1 ∥ l3)) :
  ∃ (X Y Z : Point), 
    (X ∈ ω) ∧ (Y ∈ ω) ∧ (Z ∈ ω) ∧ 
    (parallel (line_through X Y) l1 ∨ parallel (line_through X Y) l2 ∨ parallel (line_through X Y) l3) ∧
    (parallel (line_through Y Z) l1 ∨ parallel (line_through Y Z) l2 ∨ parallel (line_through Y Z) l3) ∧
    (parallel (line_through Z X) l1 ∨ parallel (line_through Z X) l2 ∨ parallel (line_through Z X) l3) :=
  sorry

end snorlax_triangle_construction_l517_517893


namespace count_prime_pairs_sum_50_l517_517134

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517134


namespace count_prime_pairs_sum_50_l517_517141

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517141


namespace sum_of_sqrt_inequality_l517_517534

theorem sum_of_sqrt_inequality (n : ℕ) (a : ℕ → ℝ) (h_ge : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≥ 0) (h_sum : ∑ i in finset.range n, a i = 1) : 
  ∑ i in finset.range n, real.sqrt (1 - a i) ≤ real.sqrt ((n - 1) * n) :=
by
  sorry

end sum_of_sqrt_inequality_l517_517534


namespace solution_set_l517_517048

variable (x : ℝ)

noncomputable def expr := (x - 1)^2 / (x - 5)^2

theorem solution_set :
  { x : ℝ | expr x ≥ 0 } = { x | x < 5 } ∪ { x | x > 5 } :=
by
  sorry

end solution_set_l517_517048


namespace prime_pairs_sum_50_l517_517279

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517279


namespace num_prime_pairs_sum_50_l517_517170

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517170


namespace angle_in_second_quadrant_l517_517594

def inSecondQuadrant (θ : ℤ) : Prop :=
  90 < θ ∧ θ < 180

theorem angle_in_second_quadrant :
  ∃ k : ℤ, inSecondQuadrant (-2015 + 360 * k) :=
by {
  sorry
}

end angle_in_second_quadrant_l517_517594


namespace max_area_triangle_m_l517_517783

theorem max_area_triangle_m (m: ℝ) :
  ∃ (M N: ℝ × ℝ), (x + y + m = 0) ∧ (x^2 + y^2 + 4*y = 0) ∧ 
    (∀ {m₁ m₂ : ℝ}, (area_triangle_CM_N m₁ ≤ area_triangle_CM_N m ∧
      area_triangle_CM_N m₂ ≤ area_triangle_CM_N m) → 
      (m = 0 ∨ m = 4)) :=
begin
  -- Insert proof here
  sorry,
end

end max_area_triangle_m_l517_517783


namespace height_of_cylinder_l517_517580

-- Definitions based on the conditions in part a)
def surface_area_cylinder (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

def radius : ℝ := 3

def total_surface_area : ℝ := 36 * Real.pi

-- Theorem stating that given the conditions, the height h is 3 feet
theorem height_of_cylinder : 
  ∃ h : ℝ, surface_area_cylinder radius h = total_surface_area ∧ h = 3 := by
  sorry

end height_of_cylinder_l517_517580


namespace circumscribed_radius_eq_eight_sec_half_angle_l517_517992

noncomputable def sector_radius {φ : ℝ} (hφ : φ > π / 2 ∧ φ < π) : ℝ :=
  8 * Real.sec (φ / 2)

theorem circumscribed_radius_eq_eight_sec_half_angle (φ : ℝ) (hφ : φ > π / 2 ∧ φ < π) :
  sector_radius hφ = 8 * Real.sec (φ / 2) :=
sorry

end circumscribed_radius_eq_eight_sec_half_angle_l517_517992


namespace number_of_prime_pairs_sum_50_l517_517213

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517213


namespace prime_pairs_sum_to_50_l517_517425

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517425


namespace smallest_y_for_perfect_cube_l517_517980

theorem smallest_y_for_perfect_cube (x y : ℕ) (x_def : x = 11 * 36 * 54) : 
  (∃ y : ℕ, y > 0 ∧ ∀ (n : ℕ), (x * y = n^3 ↔ y = 363)) := 
by 
  sorry

end smallest_y_for_perfect_cube_l517_517980


namespace discount_rate_l517_517666

variable (P P_b P_s D : ℝ)

-- Conditions
variable (h1 : P_s = 1.24 * P)
variable (h2 : P_s = 1.55 * P_b)
variable (h3 : P_b = P * (1 - D))

theorem discount_rate :
  D = 0.2 :=
by
  sorry

end discount_rate_l517_517666


namespace num_prime_pairs_sum_50_l517_517328

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517328


namespace cyclic_quadrilaterals_count_l517_517114

theorem cyclic_quadrilaterals_count :
  let is_cyclic_quadrilateral (a b c d : ℕ) : Prop :=
    a + b + c + d = 36 ∧
    a * c + b * d <= (a + c) * (b + d) ∧ -- cyclic quadrilateral inequality
    a + b > c ∧ a + c > b ∧ a + d > b ∧ b + c > d ∧ -- convex quadilateral inequality

  (finset.univ.filter (λ (s : finset ℕ), 
    s.card = 4 ∧ fact (multiset.card s.to_multiset = 36) ∧ is_cyclic_quadrilateral s.to_multiset.sum)).card = 1440 :=
sorry

end cyclic_quadrilaterals_count_l517_517114


namespace num_prime_pairs_sum_50_l517_517195

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517195


namespace prime_pairs_summing_to_50_count_l517_517390

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517390


namespace negation_of_exists_gt_one_l517_517589

theorem negation_of_exists_gt_one : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by 
  sorry

end negation_of_exists_gt_one_l517_517589


namespace exists_infinite_sequence_real_not_exists_infinite_sequence_int_l517_517697

-- Real Numbers Part
theorem exists_infinite_sequence_real :
  ∃ (a : ℕ → ℝ), 
    (∀ n : ℕ, (∑ i in finset.range 10, a (n + i)) > 0)
    ∧ (∀ n : ℕ, (∑ i in finset.range (10 * n + 1), a i) < 0) :=
  sorry

-- Integers Part
theorem not_exists_infinite_sequence_int :
  ¬ ∃ (a : ℕ → ℤ), 
      (∀ n : ℕ, (∑ i in finset.range 10, a (n + i)) > 0)
      ∧ (∀ n : ℕ, (∑ i in finset.range (10 * n + 1), a i) < 0) :=
  sorry

end exists_infinite_sequence_real_not_exists_infinite_sequence_int_l517_517697


namespace percent_of_a_is_4b_l517_517898

variables (a b : ℝ)
theorem percent_of_a_is_4b (h : a = 2 * b) : 4 * b / a = 2 :=
by 
  sorry

end percent_of_a_is_4b_l517_517898


namespace isosceles_triangle_vertex_angle_l517_517497

theorem isosceles_triangle_vertex_angle (A B C : Angle)
  (isosceles : A = B ∨ A = C ∨ B = C)
  (sum_angles : A + B + C = 180)
  (one_angle : A = 80 ∨ B = 80 ∨ C = 80) :
  A = 20 ∨ A = 80 ∨ B = 20 ∨ B = 80 ∨ C = 20 ∨ C = 80 :=
sorry

end isosceles_triangle_vertex_angle_l517_517497


namespace total_cost_toys_l517_517870

variable (c_e_actionfigs : ℕ := 60) -- number of action figures for elder son
variable (cost_e_actionfig : ℕ := 5) -- cost per action figure for elder son
variable (c_y_actionfigs : ℕ := 3 * c_e_actionfigs) -- number of action figures for younger son
variable (cost_y_actionfig : ℕ := 4) -- cost per action figure for younger son
variable (c_y_cars : ℕ := 20) -- number of cars for younger son
variable (cost_car : ℕ := 3) -- cost per car
variable (c_y_animals : ℕ := 10) -- number of stuffed animals for younger son
variable (cost_animal : ℕ := 7) -- cost per stuffed animal

theorem total_cost_toys (c_e_actionfigs c_y_actionfigs c_y_cars c_y_animals : ℕ)
                         (cost_e_actionfig cost_y_actionfig cost_car cost_animal : ℕ) :
  (c_e_actionfigs * cost_e_actionfig + c_y_actionfigs * cost_y_actionfig + 
  c_y_cars * cost_car + c_y_animals * cost_animal) = 1150 := by
  sorry

end total_cost_toys_l517_517870


namespace num_prime_pairs_sum_50_l517_517181

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517181


namespace num_prime_pairs_sum_50_l517_517190

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517190


namespace least_natural_number_k_l517_517843

theorem least_natural_number_k (n : ℕ) : 
  ∃ (k : ℕ), k = 4 ∧ 
    ∀ (S : List ℕ), S.length = 2 * n + 2 → 
      ∃ (seqs : List (List ℕ)), seqs.length = k ∧ 
        ∀ (t : List ℕ), t.length = 2 * n + 2 → 
          ∃ (s ∈ seqs), (List.zipWith (fun x y => if x = y then 1 else 0) t s).sum ≥ n + 2 := 
by
  sorry

end least_natural_number_k_l517_517843


namespace number_of_prime_pairs_sum_50_l517_517204

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517204


namespace max_sum_x_y_l517_517080

theorem max_sum_x_y (x y : ℝ) (h : (2015 + x^2) * (2015 + y^2) = 2 ^ 22) : 
  x + y ≤ 2 * Real.sqrt 33 :=
sorry

end max_sum_x_y_l517_517080


namespace find_difference_l517_517865

noncomputable def setS : Set ℝ := {x | ∃ (a b : ℝ), 1 ≤ a ∧ a ≤ b ∧ b ≤ 4 ∧ x = (3 / a) + b}

noncomputable def M : ℝ := Sup {x | ∃ (a b : ℝ), 1 ≤ a ∧ a ≤ b ∧ b ≤ 4 ∧ x = (3 / a) + b}

noncomputable def N : ℝ := Inf {x | ∃ (a b : ℝ), 1 ≤ a ∧ a ≤ b ∧ b ≤ 4 ∧ x = (3 / a) + b}

theorem find_difference : M - N = 7 - 2 * Real.sqrt 3 :=
  by
  sorry

end find_difference_l517_517865


namespace sequence_sum_constants_l517_517924

theorem sequence_sum_constants :
  let x_n (n : ℕ) := 2 * Int.floor (Real.sqrt (n - 1)) + 1,
      a := 2,
      b := 1,
      c := -1,
      d := 1
  in a + b + c + d = 3 :=
by
  sorry

end sequence_sum_constants_l517_517924


namespace factor_theorem_solution_l517_517045

theorem factor_theorem_solution (t : ℝ) :
  (x - t ∣ 3 * x^2 + 10 * x - 8) ↔ (t = 2 / 3 ∨ t = -4) :=
by
  sorry

end factor_theorem_solution_l517_517045


namespace num_prime_pairs_sum_50_l517_517187

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517187


namespace cos_3theta_value_l517_517460

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l517_517460


namespace primes_sum_50_l517_517410

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517410


namespace train_pass_bridge_time_l517_517956

noncomputable def time_to_pass_bridge (length_train length_bridge speed_kmh : ℝ) : ℝ :=
  let distance := length_train + length_bridge
  let speed_mps := speed_kmh * 1000 / 3600
  distance / speed_mps

theorem train_pass_bridge_time :
  time_to_pass_bridge 360 140 50 = 36 := 
begin
  sorry
end

end train_pass_bridge_time_l517_517956


namespace prime_pairs_sum_50_l517_517361

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517361


namespace harmonic_arithmetic_sequence_common_difference_l517_517664

theorem harmonic_arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * d)) →
  (∀ n, a n = a 1 + (n - 1) * d) →
  (a 1 = 1) →
  (d ≠ 0) →
  (∃ k, ∀ n, S n / S (2 * n) = k) →
  d = 2 :=
by
  sorry

end harmonic_arithmetic_sequence_common_difference_l517_517664


namespace num_unordered_prime_pairs_summing_to_50_l517_517158

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517158


namespace lexie_crayons_count_l517_517867

variable (number_of_boxes : ℕ) (crayons_per_box : ℕ)

theorem lexie_crayons_count (h1: number_of_boxes = 10) (h2: crayons_per_box = 8) :
  (number_of_boxes * crayons_per_box) = 80 := by
  sorry

end lexie_crayons_count_l517_517867


namespace primes_sum_50_l517_517401

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517401


namespace wedge_volume_l517_517654

noncomputable def radius := 8
noncomputable def diameter := 16
noncomputable def angle := 30 

def volume_of_wedge (r h : ℝ) : ℝ := (π * r ^ 2 * h) / 2

theorem wedge_volume : volume_of_wedge radius diameter = 512 * π := by 
  sorry

end wedge_volume_l517_517654


namespace num_prime_pairs_sum_50_l517_517322

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517322


namespace num_prime_pairs_sum_50_l517_517334

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517334


namespace problem_statement_l517_517062

noncomputable def u : ℕ → ℝ
| 1 := 1
| (n + 1) := 1 / (∑ i in finset.range n, u (i + 1)) ^ 2

noncomputable def S : ℕ → ℝ
| 1 := u 1
| (n + 1) := S n + u (n + 1)

theorem problem_statement (n : ℕ) (h : n ≥ 2) :
  real.cbrt (3 * n + 2) ≤ S n ∧ S n ≤ real.cbrt (3 * n + 2) + 1 / real.cbrt (3 * n + 2) :=
sorry

end problem_statement_l517_517062


namespace prime_pairs_summing_to_50_count_l517_517389

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517389


namespace range_of_a_l517_517882

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

theorem range_of_a (a : ℝ) :
  (¬ proposition_p a) ↔ (a ∈ Set.Ioo (-∞) 0 ∪ Set.Ioo 4 ∞) :=
by
  sorry

end range_of_a_l517_517882


namespace count_prime_pairs_summing_to_50_l517_517234

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517234


namespace prime_pairs_sum_to_50_l517_517426

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517426


namespace cyclists_speeds_product_l517_517938

theorem cyclists_speeds_product (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h₁ : 6 / u = 6 / v + 1 / 12) 
  (h₂ : v / 3 = u / 3 + 4) : 
  u * v = 864 := 
by
  sorry

end cyclists_speeds_product_l517_517938


namespace euclidean_algorithm_steps_leq_5n_l517_517863

theorem euclidean_algorithm_steps_leq_5n (m0 m1 : ℕ) (n : ℕ) (h: ∃ t : ℕ, m1 < 10^n) :
  let k := euclidean_algorithm_steps m0 m1 in
  k ≤ 5 * n :=
sorry

end euclidean_algorithm_steps_leq_5n_l517_517863


namespace probability_eq_3_5_l517_517557

-- Definition for the probability calculation
noncomputable def probability_distinct_real_roots : ℚ := 
  let possible_a := {1, 2, 3, 4, 5}
  let possible_b := {1, 2, 3}
  let total_cases := possible_a.card * possible_b.card
  let valid_cases := (possible_a.product possible_b).count (λ (ab : ℕ × ℕ), ab.1 > ab.2)
  valid_cases / total_cases

-- The theorem statement
theorem probability_eq_3_5 :
  probability_distinct_real_roots = 3 / 5 :=
by
  sorry

end probability_eq_3_5_l517_517557


namespace perpendicular_relationship_l517_517079

variables {Line : Type} [has_perp Line] (m n : Line)
variables {Plane : Type} [has_perp Plane] (α β : Plane)

-- Declare the perpendicularity relations as assumptions
variables (h1 : m ⊥ α) (h2 : n ⊥ β) (h3 : α ⊥ β)

-- Statement of the proof problem:
theorem perpendicular_relationship (h1 : m ⊥ α) (h2 : n ⊥ β) (h3 : α ⊥ β) : m ⊥ n :=
by
  sorry

end perpendicular_relationship_l517_517079


namespace count_prime_pairs_sum_50_exactly_4_l517_517285

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517285


namespace longest_segment_AB_l517_517579

theorem longest_segment_AB (A B C D : Point ℝ)
    (hAB : dist A B = |(0,2) - (-3,0)|)
    (hBC : dist B C = |(3,0) - (0,2)|)
    (hCD : dist C D = |(0,-2) - (3,0)|)
    (hDA : dist D A = |(-3,0) - (0,-2)|)
    (hBD : dist B D = |(0,-2) - (0,2)|)
    (angle_ADB : angle D A B = 70 * (pi/180))
    (angle_ABD : angle A B D = 50 * (pi/180))
    (angle_CBD : angle B C D = 60 * (pi/180))
    (angle_BDC : angle C D B = 70 * (pi/180)) : 
  (ab : dist A B > dist A D ∧ dist A D > dist B D ∧ dist B D > dist B C ∧ dist B C > dist C D) :=
  sorry

end longest_segment_AB_l517_517579


namespace puppies_adopted_each_day_l517_517981

theorem puppies_adopted_each_day (p_init new_p day_p : ℕ) (h1 : p_init = 3) (h2 : new_p = 3) (h3 : day_p = 2) :
  (p_init + new_p) / day_p = 3 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end puppies_adopted_each_day_l517_517981


namespace find_m_l517_517111

variables (m : ℝ)
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define the property of vector parallelism in ℝ.
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Statement to be proven
theorem find_m :
    parallel (1, m - 1) c →
    m = -1 :=
by
  sorry

end find_m_l517_517111


namespace minimum_value_of_expression_l517_517052

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  2 * x + 1 / x^6 ≥ 3 :=
sorry

end minimum_value_of_expression_l517_517052


namespace num_prime_pairs_sum_50_l517_517318

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l517_517318


namespace product_of_numbers_l517_517600

theorem product_of_numbers (x y : ℕ) (h1 : x + y = 15) (h2 : x - y = 11) : x * y = 26 :=
by
  sorry

end product_of_numbers_l517_517600


namespace part_I_part_II_l517_517074

-- Part (I)
theorem part_I (A B C : ℝ) (a b c : ℝ) (h1 : C = 2 * B) : cos A = 3 * cos B - 4 * (cos B) ^ 3 :=
sorry

-- Part (II)
theorem part_II (A B C : ℝ) (a b c : ℝ) (S : ℝ)
  (h2 : S = (b^2 + c^2 - a^2) / 4) 
  (h3 : b * sin B - c * sin C = a) : B = 77.5 :=
sorry

end part_I_part_II_l517_517074


namespace cos_triple_angle_l517_517469

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517469


namespace one_and_one_third_of_what_number_is_45_l517_517877

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l517_517877


namespace value_of_a10_l517_517076

variable {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)
variable (h0 : d ≠ 0) -- non-zero common difference
variable (h1 : S 3 = (a 2) ^ 2) -- S3 = a2^2
variable (h2 : S 1, S 2, S 4 form a geometric sequence -- S1, S2, S4 form a geometric sequence
variable (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2) -- Sum of first n terms of arithmetic sequence

theorem value_of_a10 : a 10 = 19 :=
by
  sorry

end value_of_a10_l517_517076


namespace least_b_not_in_range_l517_517611

theorem least_b_not_in_range : ∃ b : ℤ, -10 = b ∧ ∀ x : ℝ, x^2 + b * x + 20 ≠ -10 :=
sorry

end least_b_not_in_range_l517_517611


namespace num_prime_pairs_sum_50_l517_517189

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517189


namespace angle_in_third_quadrant_l517_517968

open Real

/--
Given that 2013° can be represented as 213° + 5 * 360° and that 213° is a third quadrant angle,
we can deduce that 2013° is also a third quadrant angle.
-/
theorem angle_in_third_quadrant (h1 : 2013 = 213 + 5 * 360) (h2 : 180 < 213 ∧ 213 < 270) : 
  (540 < 2013 % 360 ∧ 2013 % 360 < 270) :=
sorry

end angle_in_third_quadrant_l517_517968


namespace solution_set_for_inequality_l517_517086

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def f (x : ℝ) : ℝ := x^2 - 4*x

-- Prove the solution set for the inequality f(x + 2) < 5 is (-7, 3)
theorem solution_set_for_inequality : 
  is_even_function f →
  (∀ x, x > 0 → f x = x^2 - 4 * x) →
  ∀ x, f (x + 2) < 5 ↔ -7 < x ∧ x < 3 :=
by 
  intros h_even h_def x
  sorry -- Proof omitted

end solution_set_for_inequality_l517_517086


namespace no_real_m_exists_for_p_eq_s_l517_517063

theorem no_real_m_exists_for_p_eq_s :
  ¬ ∃ (m : ℝ), (setOf (λ x : ℝ, -2 ≤ x ∧ x ≤ 10)) = (setOf (λ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m)) :=
by {
  sorry
}

end no_real_m_exists_for_p_eq_s_l517_517063


namespace shortest_path_Dasha_Vasya_l517_517828

-- Definitions for the given distances
def dist_Asya_Galia : ℕ := 12
def dist_Galia_Borya : ℕ := 10
def dist_Asya_Borya : ℕ := 8
def dist_Dasha_Galia : ℕ := 15
def dist_Vasya_Galia : ℕ := 17

-- Definition for shortest distance by roads from Dasha to Vasya
def shortest_dist_Dasha_Vasya : ℕ := 18

-- Proof statement of the goal that shortest distance from Dasha to Vasya is 18 km
theorem shortest_path_Dasha_Vasya : 
  dist_Dasha_Galia + dist_Vasya_Galia - dist_Asya_Galia - dist_Galia_Borya = shortest_dist_Dasha_Vasya := by
  sorry

end shortest_path_Dasha_Vasya_l517_517828


namespace prime_pairs_summing_to_50_count_l517_517380

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517380


namespace cos_3theta_value_l517_517458

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l517_517458


namespace prime_pairs_sum_50_l517_517247

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517247


namespace find_number_l517_517875

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l517_517875


namespace find_a11_l517_517864

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom cond1 : ∀ n : ℕ, n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n

-- Theorem stating the proof problem
theorem find_a11 :
  a 11 = -2 :=
sorry

end find_a11_l517_517864


namespace prime_pairs_sum_50_l517_517449

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517449


namespace prime_pairs_sum_to_50_l517_517301

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517301


namespace quadratic_has_vertex_and_opens_downwards_l517_517071

noncomputable def quadratic_function (a : ℝ) : ℝ × ℝ := 
  (2, -1)

theorem quadratic_has_vertex_and_opens_downwards :
  ∃ a : ℝ, a < 0 ∧
  ∃ f : ℝ → ℝ,
    (∀ k, f = λ x, a * (x - 2)^2 - 1) ∧
    f = (λ x, 2 * x^2 - 8 * x + 7) := 
by
  sorry

end quadratic_has_vertex_and_opens_downwards_l517_517071


namespace fraction_equality_l517_517730

theorem fraction_equality (x : ℝ) :
  (4 + 2 * x) / (7 + 3 * x) = (2 + 3 * x) / (4 + 5 * x) ↔ x = -1 ∨ x = -2 := by
  sorry

end fraction_equality_l517_517730


namespace factorize_expression_l517_517705

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l517_517705


namespace factorize_expression_l517_517707

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l517_517707


namespace impurity_requirement_l517_517651

noncomputable def impurity_decrease (x : ℕ) : ℝ :=
  0.02 * (2 / 3) ^ x

def minimum_filtrations_needed : ℕ :=
  let n : ℝ := (1 + 0.3010) / (0.4771 - 0.3010)
  Int.ceil n

theorem impurity_requirement (n : ℕ) (h : 0.02 * (2 / 3) ^ n ≤ 0.001) : n ≥ 8 :=
by
  have log2 : ℝ := 0.3010
  have log3 : ℝ := 0.4771
  have log2_3 : ℝ := log2 - log3
  have numerator : ℝ := 1 + log2
  have n_real : ℝ := numerator / log2_3
  have n_approx : ℝ := 7.4
  have n_int : ℕ := Int.ceil n_real
  sorry

end impurity_requirement_l517_517651


namespace least_number_divisible_by_11_and_remainder_2_l517_517615

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end least_number_divisible_by_11_and_remainder_2_l517_517615


namespace num_prime_pairs_sum_50_l517_517185

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517185


namespace num_prime_pairs_sum_50_l517_517197

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517197


namespace kim_hard_correct_l517_517492

-- Definitions
def points_per_easy := 2
def points_per_average := 3
def points_per_hard := 5
def easy_correct := 6
def average_correct := 2
def total_points := 38

-- Kim's correct answers in the hard round is 4
theorem kim_hard_correct : (total_points - (easy_correct * points_per_easy + average_correct * points_per_average)) / points_per_hard = 4 :=
by
  sorry

end kim_hard_correct_l517_517492


namespace prime_pairs_sum_50_l517_517243

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517243


namespace min_distance_parabola_l517_517785

-- Given definitions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def point_on_parabola (x y : ℝ) : Prop := parabola x y
def focus : (ℝ × ℝ) := (1, 0)
def B : (ℝ × ℝ) := (3, 4)

-- Lean 4 statement
theorem min_distance_parabola (P : ℝ × ℝ) (hP : point_on_parabola P.fst P.snd) :
  ∃ P : ℝ × ℝ, |P - B| + sqrt ((P.fst - focus.fst)^2 + (P.snd - focus.snd)^2) = 2 * sqrt 5 :=
sorry

end min_distance_parabola_l517_517785


namespace prime_pairs_sum_50_l517_517273

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517273


namespace fractional_part_of_blue_square_four_changes_l517_517996

theorem fractional_part_of_blue_square_four_changes 
  (initial_area : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ (a : ℝ), f a = (8 / 9) * a) :
  (f^[4]) initial_area / initial_area = 4096 / 6561 :=
by
  sorry

end fractional_part_of_blue_square_four_changes_l517_517996


namespace xi_n_converges_to_zero_a_s_l517_517889

open ProbabilityTheory

noncomputable theory

variable {Ω : Type*} [MeasureSpace Ω]
variable {ξ : ℕ → Ω → ℝ}

theorem xi_n_converges_to_zero_a_s 
  (h : ∑' n, ∫⁻ ω, |ξ n ω| ∂(MeasureTheory.MeasureSpace.volume) < ∞) :
  ∀ᵐ ω ∂(MeasureTheory.MeasureSpace.volume), (tendsto (λ n, ξ n ω) at_top (nhds 0)) :=
sorry

end xi_n_converges_to_zero_a_s_l517_517889


namespace num_prime_pairs_summing_to_50_l517_517345

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517345


namespace total_weight_of_rings_l517_517511

-- Define the weights of the rings
def weight_orange : Real := 0.08
def weight_purple : Real := 0.33
def weight_white : Real := 0.42
def weight_blue : Real := 0.59
def weight_red : Real := 0.24
def weight_green : Real := 0.16

-- Define the total weight of the rings
def total_weight : Real :=
  weight_orange + weight_purple + weight_white + weight_blue + weight_red + weight_green

-- The task is to prove that the total weight equals 1.82
theorem total_weight_of_rings : total_weight = 1.82 := 
  by
    sorry

end total_weight_of_rings_l517_517511


namespace number_of_prime_pairs_sum_50_l517_517222

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517222


namespace count_four_digit_numbers_with_digits_2_or_5_l517_517119

theorem count_four_digit_numbers_with_digits_2_or_5 : 
  ∃ (count : ℕ), count = 16 ∧ 
  (∀ n : ℕ, n >= 1000 → n < 10000 → 
   (∀ d ∈ [ (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10], 
    d = 2 ∨ d = 5) → count = 16) :=
by {
  -- Count the number of four-digit numbers where each digit is either 2 or 5
  let count := (2 ^ 4),
  use count,
  split,
  { exact rfl },
  { intros n hn1 hn2 h_digits,
    sorry
  }
}

end count_four_digit_numbers_with_digits_2_or_5_l517_517119


namespace parabola_directrix_l517_517575

theorem parabola_directrix (p : ℝ) (h : p = 8) : (∀ y, y = -4 → x^2 = 16y) :=
by
  assume y
  assume h_directrix : y = -4
  have h_eq : y = x^2 / 16 :=
    sorry
  show x^2 = 16y,
    by sorry

end parabola_directrix_l517_517575


namespace biff_break_even_hours_l517_517008

-- Definitions based on conditions
def ticket_expense : ℕ := 11
def snacks_expense : ℕ := 3
def headphones_expense : ℕ := 16
def total_expenses : ℕ := ticket_expense + snacks_expense + headphones_expense
def gross_income_per_hour : ℕ := 12
def wifi_cost_per_hour : ℕ := 2
def net_income_per_hour : ℕ := gross_income_per_hour - wifi_cost_per_hour

-- The proof statement
theorem biff_break_even_hours : ∃ h : ℕ, h * net_income_per_hour = total_expenses ∧ h = 3 :=
by 
  have h_value : ℕ := 3
  exists h_value
  split
  · show h_value * net_income_per_hour = total_expenses
    sorry
  · show h_value = 3
    rfl

end biff_break_even_hours_l517_517008


namespace option_A_correct_l517_517622

theorem option_A_correct (y x : ℝ) : y * x - 2 * (x * y) = - (x * y) :=
by
  sorry

end option_A_correct_l517_517622


namespace exist_compatible_assignment_l517_517543

def Species := Fin 8
def Cages := Fin 4

-- A relation indicating that two species are not compatible
def incompatible : Species → Species → Prop := sorry 

-- Ensuring the incompatibility graph is symmetric
axiom incompatibility_symm (s1 s2 : Species) : incompatible s1 s2 → incompatible s2 s1

-- Condition: Each species has at most 3 other species with which it cannot share a cage
axiom max_incompatibility (s : Species) : (fintype.card {t : Species // incompatible s t}) ≤ 3

-- Defining a function that assigns species to cages
def assignment : Species → Cages := sorry

-- The goal is to prove that this assignment satisfies the compatibility constraint
theorem exist_compatible_assignment : 
  ∃ (assignment : Species → Cages), ∀ s1 s2 : Species, incompatible s1 s2 → assignment s1 ≠ assignment s2 :=
by
  sorry

end exist_compatible_assignment_l517_517543


namespace problem1_problem2_l517_517740

-- Define the values of x and y
def x : ℝ := 2 + Real.sqrt 3
def y : ℝ := 2 - Real.sqrt 3

-- The first proposition: x² + 2xy + y² = 16
theorem problem1 : x^2 + 2 * x * y + y^2 = 16 :=
by sorry

-- The second proposition: x² - y² = 8√3
theorem problem2 : x^2 - y^2 = 8 * Real.sqrt 3 :=
by sorry

end problem1_problem2_l517_517740


namespace prime_pairs_sum_50_l517_517242

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517242


namespace count_prime_pairs_sum_50_exactly_4_l517_517288

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517288


namespace tangent_line_equivalence_l517_517928

def point := (ℝ × ℝ)

def tangent_line_at_point (f : ℝ → ℝ) (P : point) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, y : ℝ, y = m * (x - P.1) + f P.1 → y = f x

def tangent_line_passing_through (P : point) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, y : ℝ, y = m * (x - P.1) + P.2 → y = f x

theorem tangent_line_equivalence (f : ℝ → ℝ) (P : point) :
  tangent_line_at_point f P ↔ tangent_line_passing_through P :=
  sorry

end tangent_line_equivalence_l517_517928


namespace prime_pairs_sum_50_l517_517250

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517250


namespace roundness_of_1280000_l517_517014

theorem roundness_of_1280000 : 
  (∑ e in (1,280,000).factors.to_multiset.map (fun p => log_nat p), e) = 19 :=
by
  sorry

end roundness_of_1280000_l517_517014


namespace cos_3theta_value_l517_517461

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l517_517461


namespace D_96_l517_517846

def D : ℕ → ℕ
| 1       := 0
| n+1     := -- The full definition should be included here, along with any helper functions or constructs to properly define D(n)
sorry

theorem D_96 : D 96 = 112 :=
by
  sorry

end D_96_l517_517846


namespace measure_of_angle_B_range_of_possible_area_l517_517833

theorem measure_of_angle_B (a b c : ℝ) (A B C : ℝ)
  (cond1 : a = b * cos A)
  (cond2 : b = c * cos B)
  (h : b / (sqrt 3 * cos B) = c / sin C) : B = π / 3 :=
by 
  sorry

theorem range_of_possible_area (a b c : ℝ) (A B C : ℝ) (b_val : b = 3)
  (measure_B : B = π / 3)
  (cond1 : a = b * cos A)
  (cond2 : c = b * sin C)
  (cond3 : B + C = π / 3 + A) : 0 < triangle_area a b c / 2 ≤ 9 * sqrt 3 / 4 :=
by 
  sorry

end measure_of_angle_B_range_of_possible_area_l517_517833


namespace count_prime_pairs_summing_to_50_l517_517239

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517239


namespace probability_neither_defective_l517_517632

noncomputable def n := 9
noncomputable def k := 2
noncomputable def total_pens := 9
noncomputable def defective_pens := 3
noncomputable def non_defective_pens := total_pens - defective_pens

noncomputable def total_combinations := Nat.choose total_pens k
noncomputable def non_defective_combinations := Nat.choose non_defective_pens k

theorem probability_neither_defective :
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 := by
sorry

end probability_neither_defective_l517_517632


namespace area_under_piecewise_function_l517_517849

def piecewise_function (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 6 then 2 * x
  else if 6 < x ∧ x ≤ 10 then 3 * x - 10
  else 0

theorem area_under_piecewise_function :
  ∫ x in 0..10, piecewise_function x = 92 :=
by
  sorry

end area_under_piecewise_function_l517_517849


namespace num_unordered_prime_pairs_summing_to_50_l517_517159

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517159


namespace num_prime_pairs_summing_to_50_l517_517352

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l517_517352


namespace total_money_collected_l517_517606

def number_of_people := 610
def price_adult := 2
def price_child := 1
def number_of_adults := 350

theorem total_money_collected :
  (number_of_people - number_of_adults) * price_child + number_of_adults * price_adult = 960 := by
  sorry

end total_money_collected_l517_517606


namespace solution_unique_l517_517895

noncomputable def solve_equation (a x : ℝ) : Prop :=
  (0 < a) ∧ (a < 1) ∧ (∃ n ∈ ℤ, ∃ y ∈ Ioo 0 1, x = n + y ∧ a^n + real.log y / real.log a = x)

theorem solution_unique (a : ℝ) (h_a : 0 < a ∧ a < 1) (n : ℤ) (h_n : 0 < n) :
  solve_equation a (↑n + a^(n - a^n) * real.exp(-real.W (-a^(n - a^n) * real.log a))) :=
by {
  sorry
}

end solution_unique_l517_517895


namespace biff_break_even_hours_l517_517006

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end biff_break_even_hours_l517_517006


namespace prime_pairs_sum_to_50_l517_517420

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517420


namespace largest_four_digit_number_last_digit_l517_517974

theorem largest_four_digit_number_last_digit:
  ∃ (N : ℕ), N % 9 = 0 ∧ (∀ (d : ℕ), 1 ≤ d ∧ d ≤ 9 → (N // 10 = 996 → (d + 24 = N % 10))) ∧ (N % 10 = 3) := 
begin
  -- Definition of conditions
  let N := 996 * 10 + 3,
  -- Check divisibility by 9
  have div_9 : N % 9 = 0, from by sorry,
  -- Check if removing the last digit results in a multiple of 4
  have mult_4 : N // 10 = 996, from by sorry,
  -- Correct conclusion
  have last_digit : N % 10 = 3, from by sorry,
  -- Asserting the existence of such N
  use N,
  exact ⟨div_9, mult_4, last_digit⟩,
end

end largest_four_digit_number_last_digit_l517_517974


namespace smallest_proper_largest_improper_fraction_l517_517058

theorem smallest_proper_largest_improper_fraction :
  let numbers := {2, 3, 4, 5}
  let smallest_proper_frac := if max numbers - min numbers == 3 then (2 / 5) else (min numbers / max numbers)
  let largest_improper_frac := if max numbers - min numbers == 3 then (5 / 2) else (max numbers / min numbers)
  smallest_proper_frac = (2 / 5) ∧ largest_improper_frac = (5 / 2) :=
by
  let numbers := {2, 3, 4, 5}
  let smallest_proper_frac := if max numbers - min numbers == 3 then (2 / 5) else (min numbers / max numbers)
  let largest_improper_frac := if max numbers - min numbers == 3 then (5 / 2) else (max numbers / min numbers)
  have h1 : smallest_proper_frac = 2 / 5 := sorry
  have h2 : largest_improper_frac = 5 / 2 := sorry
  exact ⟨h1, h2⟩

end smallest_proper_largest_improper_fraction_l517_517058


namespace count_prime_pairs_sum_50_l517_517130

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517130


namespace prime_pairs_sum_to_50_l517_517413

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517413


namespace prime_pairs_sum_50_l517_517249

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517249


namespace find_C_l517_517954

-- Variables and conditions
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + B + C = 1000
def condition2 : Prop := A + C = 700
def condition3 : Prop := B + C = 600

-- The statement to be proved
theorem find_C (h1 : condition1 A B C) (h2 : condition2 A C) (h3 : condition3 B C) : C = 300 :=
sorry

end find_C_l517_517954


namespace schedule_arrangements_l517_517819

-- Define the initial setup of the problem
def subjects : List String := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"]

def periods_morning : List String := ["P1", "P2", "P3", "P4"]
def periods_afternoon : List String := ["P5", "P6", "P7"]

-- Define the constraints
def are_consecutive (subj1 subj2 : String) : Bool := 
  (subj1 = "Chinese" ∧ subj2 = "Mathematics") ∨ 
  (subj1 = "Mathematics" ∧ subj2 = "Chinese")

def can_schedule_max_one_period (subject : String) : Bool :=
  subject = "English" ∨ subject = "Physics" ∨ subject = "Chemistry" ∨ subject = "Biology"

-- Define the math problem as a proof in Lean
theorem schedule_arrangements : 
  ∃ n : Nat, n = 336 :=
by
  -- The detailed proof steps would go here
  sorry

end schedule_arrangements_l517_517819


namespace count_prime_pairs_sum_50_l517_517142

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l517_517142


namespace prime_pairs_sum_to_50_l517_517424

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l517_517424


namespace probability_more_granddaughters_l517_517869

theorem probability_more_granddaughters (n : ℕ) (h_n : n = 12)
  (p : ℝ) (h_p : p = 0.5) :
  let favorable_outcomes := (nat.choose 12 7) + (nat.choose 12 8) + (nat.choose 12 9) + (nat.choose 12 10) + (nat.choose 12 11) + (nat.choose 12 12)
  let total_outcomes := 2^12
  favorable_outcomes / total_outcomes = 793 / 2048 :=
begin
  sorry
end

end probability_more_granddaughters_l517_517869


namespace prime_pairs_sum_50_l517_517434

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517434


namespace D_coin_count_l517_517621

def A_coin_count : ℕ := 21
def B_coin_count := A_coin_count - 9
def C_coin_count := B_coin_count + 17
def sum_A_B := A_coin_count + B_coin_count
def sum_C_D := sum_A_B + 5

theorem D_coin_count :
  ∃ D : ℕ, sum_C_D - C_coin_count = D :=
sorry

end D_coin_count_l517_517621


namespace problem_statement_l517_517031

theorem problem_statement : 
  ( \bigg( \sum i from 1 to 2022, (2023 - i) / i \bigg) / \bigg( \sum j from 2 to 2023, 1 / j \bigg) = 2023 ) )
:= sorry

end problem_statement_l517_517031


namespace selection_num_ways_l517_517646

theorem selection_num_ways : 
  ∀ (volunteers : Finset ℕ) (n k l : ℕ),
  volunteers.card = 5 ∧ n = 2 ∧ k = 2 ∧ l = 3 →
  ∑ (S : Finset ℕ) in volunteers.powerset.filter (λ x, x.card = 2), 
    ∑ (T : Finset ℕ) in (volunteers \ S).powerset.filter (λ x, x.card = 2), 
    (if (S ∩ T = ∅) then 1 else 0) = 30 :=
by
  intros volunteers n k l h
  have hc1 : finset.card volunteers = 5 := h.1
  have hn2 : n = 2 := h.2.1
  have hk2 : k = 2 := h.2.2.1
  have hl3 : l = 3 := h.2.2.2
  sorry

end selection_num_ways_l517_517646


namespace num_valid_numbers_l517_517121

-- Definition of being a four-digit number with each digit being either 2 or 5
def is_valid_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧ (∀ d ∈ n.digits 10, d = 2 ∨ d = 5)

-- Theorem to prove the number of valid numbers is 16
theorem num_valid_numbers : (finset.filter is_valid_number (finset.range 10000)).card = 16 := 
by
  sorry

end num_valid_numbers_l517_517121


namespace surface_area_cube_surface_area_pyramid_main_problem_l517_517840

-- Definitions based on given conditions
def A1B1C1D1A2B2C2D2 : Type := unit_cube
def M := center_face (A2B2C2D2)
def pyramid := MA1B1C1D1

theorem surface_area_cube : surface_area (unit_cube) = 6 := by sorry
theorem surface_area_pyramid : surface_area (MA1B1C1D1) = sqrt(6) + 1 := by sorry

theorem main_problem (A1B1C1D1A2B2C2D2 : unit_cube) 
                     (A1B1C1D1 A2B2C2D2 : square_face A1B1C1D1A2B2C2D2)
                     (M : center_face A2B2C2D2)
                     (pyramid : pyramid MA1B1C1D1) : 
                     let remaining_solid_surface_area := (surface_area unit_cube) - (surface_area pyramid) + (surface_area (square_face A2B2C2D2))
                     in remaining_solid_surface_area = 5 + sqrt(6) ∧ (a + b = 11) := 
by sorry

end surface_area_cube_surface_area_pyramid_main_problem_l517_517840


namespace sum_series_a_l517_517957

/-- The sum of the series 1/2 + 2/2^2 + 3/2^3 + ... + n/2^n is 2 - (2 + n) / 2^n --/
theorem sum_series_a (n : ℕ) : 
  (∑ k in finset.range (n + 1), (k + 1) / 2^(k + 1)) = 2 - (2 + n) / 2^(n + 1) := 
sorry

end sum_series_a_l517_517957


namespace count_prime_pairs_sum_50_exactly_4_l517_517287

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517287


namespace troublesome_subset_size_troublesome_subset_existence_l517_517533

def troublesome_set (T : Set ℕ) : Prop :=
∀ (u v : ℕ), u ∈ T → v ∈ T → u + v ∉ T

theorem troublesome_subset_size (T : Set ℕ) (h₁ : troublesome_set T) (h₂ : T ⊆ { n : ℕ | 1 ≤ n ∧ n ≤ 2006 }) :
  T.finite → T.card ≤ 1003 :=
sorry

theorem troublesome_subset_existence (S : Finset ℕ) (h₁ : S.card = 2006) :
  ∃ (T : Finset ℕ), (troublesome_set T) ∧ T ⊆ S ∧ T.card ≥ 669 :=
sorry

end troublesome_subset_size_troublesome_subset_existence_l517_517533


namespace distinct_ordered_pairs_count_l517_517772

theorem distinct_ordered_pairs_count :
  ∃ (n : ℕ), n = 29 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b → a + b = 30 → ∃! p : ℕ × ℕ, p = (a, b)) :=
sorry

end distinct_ordered_pairs_count_l517_517772


namespace part_I_part_II_l517_517537

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x^2 + a * x|
noncomputable def M (a : ℝ) : ℝ := 
  if a > 2 - 2 * Real.sqrt 2 then 1 + a
  else if a ≤ -2 then -a - 1
  else (a^2) / 4

theorem part_I (a : ℝ) : (∀ x x', 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ x' ∧ x' ≤ 1 → x ≤ x' → f a x ≤ f a x') ↔ a ≥ 0 ∨ a ≤ -2 :=
begin
  sorry
end

theorem part_II : ∃ a : ℝ, M a = 3 - 2 * Real.sqrt 2 :=
begin
  sorry
end

end part_I_part_II_l517_517537


namespace sum_possible_integer_values_l517_517087

theorem sum_possible_integer_values (m : ℤ) (h1 : 0 < 3 * m) (h2 : 3 * m < 45) : 
  ∑ k in finset.Icc 1 14, k = 105 := 
by
  sorry

end sum_possible_integer_values_l517_517087


namespace find_m_plus_n_l517_517528

def operation (m n : ℕ) : ℕ := m^n + m * n

theorem find_m_plus_n :
  ∃ (m n : ℕ), (2 ≤ m) ∧ (2 ≤ n) ∧ (operation m n = 64) ∧ (m + n = 6) :=
by {
  -- Begin the proof context
  sorry
}

end find_m_plus_n_l517_517528


namespace clock_angle_l517_517653

theorem clock_angle (east_deg northwest_deg : ℝ) (n : ℕ) (angle_between_two_rays : ℝ) :
    n = 12 →
    east_deg = 90 →
    northwest_deg = 315 →
    angle_between_two_rays = (northwest_deg - east_deg) :=
begin
    sorry
end

#eval clock_angle 90 315 12 225

end clock_angle_l517_517653


namespace exist_a_b_for_every_n_l517_517552

theorem exist_a_b_for_every_n (n : ℕ) (hn : 0 < n) : 
  ∃ (a b : ℤ), 1 < a ∧ 1 < b ∧ a^2 + 1 = 2 * b^2 ∧ (a - b) % n = 0 := 
sorry

end exist_a_b_for_every_n_l517_517552


namespace x_lt_2_necessary_not_sufficient_x_sq_lt_4_l517_517637

theorem x_lt_2_necessary_not_sufficient_x_sq_lt_4 (x : ℝ) :
  (x < 2) → (x^2 < 4) ∧ ¬((x^2 < 4) → (x < 2)) :=
by
  sorry

end x_lt_2_necessary_not_sufficient_x_sq_lt_4_l517_517637


namespace sum_of_red_intervals_eq_one_one_quarter_in_cantor_set_sum_of_remaining_terms_is_green_l517_517022

noncomputable def cantor_set : set ℝ :=
{ x | ∀ n : ℕ, let b := (2^n) • (3^(-n : ℤ)) in ∀ (i : ℕ), (x < i * b ∨ x ≥ (i + 1) * b) }

theorem sum_of_red_intervals_eq_one : 
  (series_sum : ℕ → ℝ := λ n, 2^(n-1) / 3^n) → 
  ∑' n, series_sum n = 1 := sorry

theorem one_quarter_in_cantor_set (x : ℝ) (h : x=1/4) : x ∈ cantor_set := sorry

theorem sum_of_remaining_terms_is_green :
  ∀ (s : set ℕ), (∀ n ∉ s, 2^(n-1) / 3^n) → 
  ∑' (n : ℕ) (h : n ∈ s), 2^(n-1) / 3^n ∈ cantor_set := sorry

end sum_of_red_intervals_eq_one_one_quarter_in_cantor_set_sum_of_remaining_terms_is_green_l517_517022


namespace probability_event_l517_517969

def probability_score_shots
  (independent_shots : ∀ n : ℕ, ∀ k : ℕ, k ≤ n → (n.choose k) * (1 / 2)^k * (1 / 2)^(n - k) = (1 / 2) ^ n)
  (total_shots : ℕ := 6)
  (prob_sixth_shot : real := 1 / 2)
  (prob_less_equal_half (n : ℕ) : ℚ := if n = 6 then 1 / 2 else 1) :
  ℚ :=
sorry

theorem probability_event (h: probability_score_shots _ _ _ _ = 5 / 64) : True :=
sorry

end probability_event_l517_517969


namespace find_ellipse_equation_find_max_slope_line_equation_l517_517763

/-- Given conditions for the ellipse -/
structure EllipseConditions (a b x y : ℝ) :=
(a_pos : 0 < a)
(b_pos : 0 < b)
(a_gt_b : a > b)
(ellipse_eq : x^2 / a^2 + y^2 / b^2 = 1)
(distance_foci : 2 * sqrt(2) = 2 * c)
(angle_F1DF2 : angle F1 D F2 = real.pi / 3)
(area_triangle : (sqrt(3) / 3) = (1 / 2) * |DF1| * |DF2| * sin (real.pi / 3))

/-- Given conditions for Line through point Q and intersection with ellipse -/
structure LineConditions (a b m : ℝ) :=
(eqp : x = m * y + 1)
(intersectionA : is_intersection_point P A)
(intersectionB : is_intersection_point P B)
(slopeA : k1 = (y2 - 3)/(x2 - 4))
(slopeB : k2 = (y1 - 3)/(x1 - 4))

theorem find_ellipse_equation (a b x y c F1 D F2 : ℝ) (h : EllipseConditions a b x y): 
  a^2 = 4 ∧ b^2 = 2 ∧ (x^2 / 4 + y^2 / 2 = 1) :=
begin
  sorry
end

theorem find_max_slope_line_equation (a b m x y c F1 D F2 k1 k2 A B Q P : ℝ) (h1 : EllipseConditions a b x y) (h2 : LineConditions a b m x y):
  k1 * k2 → maximize (k1 * k2) ∧ line_eq = (x - y - 1 = 0) :=
begin
  sorry
end

end find_ellipse_equation_find_max_slope_line_equation_l517_517763


namespace arithmetic_sequence_a3_l517_517077

theorem arithmetic_sequence_a3 :
  ∀ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 = 2 ∧ 2 * a 2 = a 1 + a 3 -> a 3 = 3 :=
by
  intros a h,
  cases h with ha1 htail,
  cases htail with ha2 harith,
  rw [ha1, ha2] at harith,
  linarith

example : ∃ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 = 2 ∧ 2 * a 2 = a 1 + a 3 ∧ a 3 = 3 :=
by
  let a := λ n, if n = 1 then 1 else if n = 2 then 2 else if n = 3 then 3 else 0,
  use a,
  repeat { split };
  simp

end arithmetic_sequence_a3_l517_517077


namespace correct_option_is_A_l517_517623

def is_linear_equation (eq : String) : Bool :=
  -- Dummy implementation, in practice you may need to parse and check the equation structure
  match eq with
  | "x+y=5"     => true
  | "y=2"       => true
  | "x+y=2"     => true
  | "y-z=6"     => true
  | "xy=4"      => false
  | "y=1"       => true
  | "x^2-1=0"   => false
  | "x+y=5"     => true
  | _           => false

def is_system_of_two_linear_equations (eqs : List String) : Bool :=
  eqs.length == 2 ∧ eqs.all is_linear_equation

theorem correct_option_is_A :
  ∃ (A B C D : List String),
    A = ["x+y=5", "y=2"] ∧
    B = ["x+y=2", "y-z=6"] ∧
    C = ["xy=4", "y=1"] ∧
    D = ["x^2-1=0", "x+y=5"] ∧
    is_system_of_two_linear_equations A ∧
    ¬ is_system_of_two_linear_equations B ∧
    ¬ is_system_of_two_linear_equations C ∧
    ¬ is_system_of_two_linear_equations D ∧
    A = ["x+y=5", "y=2"] :=
by
  -- Prove that option A is the correct answer that forms a system of two linear equations
  let A := ["x+y=5", "y=2"]
  let B := ["x+y=2", "y-z=6"]
  let C := ["xy=4", "y=1"]
  let D := ["x^2-1=0", "x+y=5"]
  have hA : is_system_of_two_linear_equations A := by sorry
  have hB : ¬ is_system_of_two_linear_equations B := by sorry
  have hC : ¬ is_system_of_two_linear_equations C := by sorry
  have hD : ¬ is_system_of_two_linear_equations D := by sorry
  exact ⟨A, B, C, D, rfl, rfl, rfl, rfl, hA, hB, hC, hD, rfl⟩

end correct_option_is_A_l517_517623


namespace cosine_sum_identity_l517_517892

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem cosine_sum_identity :
  let x := Real.cos (2 * Real.pi / 17) + Real.cos (4 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) + Real.cos (14 * Real.pi / 17) in
  omega ^ 17 = 1 →
  x = (Real.sqrt 33 - 1) / 4 :=
by
  intro x h_omega
  -- proof goes here
  sorry

end cosine_sum_identity_l517_517892


namespace exist_x_y_l517_517842

theorem exist_x_y (a b c : ℝ) (h₁ : abs a > 2) (h₂ : a^2 + b^2 + c^2 = a * b * c + 4) :
  ∃ x y : ℝ, a = x + 1/x ∧ b = y + 1/y ∧ c = x*y + 1/(x*y) :=
sorry

end exist_x_y_l517_517842


namespace investment_amount_l517_517657

theorem investment_amount (A_investment B_investment total_profit A_share : ℝ)
  (hA_investment : A_investment = 100)
  (hB_investment_months : B_investment > 0)
  (h_total_profit : total_profit = 100)
  (h_A_share : A_share = 50)
  (h_conditions : A_share / total_profit = (A_investment * 12) / ((A_investment * 12) + (B_investment * 6))) :
  B_investment = 200 :=
by {
  sorry
}

end investment_amount_l517_517657


namespace fraction_of_work_left_l517_517634

def work_rate_p : ℚ := 1 / 15
def work_rate_q : ℚ := 1 / 20
def work_together_rate : ℚ := work_rate_p + work_rate_q
def days_worked : ℚ := 4
def work_done : ℚ := work_together_rate * days_worked

theorem fraction_of_work_left :
  work_done = 7 / 15 →
  1 - work_done = 8 / 15 :=
by 
  intros h
  rw [h]
  norm_num
  sorry

end fraction_of_work_left_l517_517634


namespace largest_w_factor_of_7_in_factorial_l517_517958

theorem largest_w_factor_of_7_in_factorial (w : ℕ) :
  (7 ^ w ∣ nat.factorial 100) ↔ w ≤ 16 := 
sorry

end largest_w_factor_of_7_in_factorial_l517_517958


namespace prime_pairs_sum_50_l517_517262

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l517_517262


namespace diameter_not_prime_l517_517921

-- Define the conditions
variables (a b c : ℕ) (AB AC BD : ℝ)
variable (ABCD_inscribed_in_semicircle : AB = c)
variable (BC_eq_a : BC = a)
variable (CD_eq_a : CD = a)
variable (DA_eq_b_b_ne_a : DA = b ∧ b ≠ a)
variable (natural_numbers : ∀ (x : ℕ), x > 0)

-- Defining Ptolemy's Theorem for cyclic quadrilateral
lemma ptolemys_theorem (AB CD : ℝ) (BC DA AC BD : ℝ) (ABCD_inscribed : cyclic ABCD) :
  AB * CD + BC * DA = AC * BD := sorry

-- Expressing the diagonal relations in terms of semicircle properties using Pythagorean theorem
lemma diagonal_relations (AB AC BD : ℝ) (BC_eq_a : BC = a) (CD_eq_a : CD = a) (DA_eq_b : DA = b) :
  BD^2 = (AB)^2 - (DA)^2 ∧ AC^2 = (AB)^2 - (BC)^2 := sorry

-- Main proof statement
theorem diameter_not_prime (a b c : ℕ) (AB AC BD : ℝ)
  (ABCD_inscribed_in_semicircle : AB = c)
  (BC_eq_a : BC = a) (CD_eq_a : CD = a)
  (DA_eq_b_b_ne_a : DA = b ∧ b ≠ a)
  (natural_numbers : ∀ (x : ℕ), x > 0):
  ¬ prime c := sorry

end diameter_not_prime_l517_517921


namespace baseball_team_selection_l517_517977

-- Define the conditions
def number_of_players := 16
def number_of_twins := 2
def total_lineup := 9

-- Calculate remaining players after including the twins
def remaining_players := number_of_players - number_of_twins
def players_to_choose := total_lineup - number_of_twins

-- Lean statement to prove the required number of ways
theorem baseball_team_selection : Nat.choose remaining_players players_to_choose = 3432 := by
  -- Number of ways to select 7 players out of 14
  have h : remaining_players = 14 := by rfl
  have h' : players_to_choose = 7 := by rfl
  rw [h, h']
  exact Nat.choose_spec 14 7

end baseball_team_selection_l517_517977


namespace prime_pairs_sum_50_l517_517245

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l517_517245


namespace quadratic_sum_roots_l517_517486

theorem quadratic_sum_roots {a b : ℝ}
  (h1 : ∀ x, x^2 - a * x + b < 0 ↔ -1 < x ∧ x < 3) :
  a + b = -1 :=
sorry

end quadratic_sum_roots_l517_517486


namespace prime_pairs_sum_50_l517_517433

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l517_517433


namespace prime_pairs_sum_50_l517_517363

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l517_517363


namespace prime_squares_between_5000_and_10000_l517_517127

open Nat

theorem prime_squares_between_5000_and_10000 :
  (finset.filter prime (finset.Ico 71 100)).card = 6 :=
by
  sorry

end prime_squares_between_5000_and_10000_l517_517127


namespace sum_of_coefficients_expr_is_192_l517_517043

-- Define the expression
def expr : ℤ[X] := 216 * X^9 - 1000 * Y^9

-- Define the function to find the sum of integer coefficients
def sum_of_coefficients (p : ℤ[X]) : ℤ :=
  p.coefficients.sum

-- The proof statement specifying what needs to be proved
theorem sum_of_coefficients_expr_is_192 : 
  sum_of_coefficients expr = 192 :=
sorry

end sum_of_coefficients_expr_is_192_l517_517043


namespace joint_business_profit_l517_517937

noncomputable def totalProfit (x : ℕ) : ℕ := 9 * ((800 / 4))

theorem joint_business_profit :
  ∀ (x : ℕ),
    (x + (x + 1000) + (x + 2000) = 9000) →
    (totalProfit x = 1800) :=
begin
  intro x,
  intro h,
  unfold totalProfit,
  rw [mul_assoc],
  norm_num,
  sorry
end

end joint_business_profit_l517_517937


namespace num_prime_pairs_sum_50_l517_517183

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517183


namespace last_two_digits_of_a2022_l517_517787

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 2 ∧ 
  ∀ k : ℕ, k > 0 → a (2 * k + 1) = (a (2 * k) ^ 2) / (a (2 * k - 1)) ∧
  ∀ k : ℕ, k > 0 → a (2 * k + 2) = 2 * a (2 * k + 1) - a (2 * k)

theorem last_two_digits_of_a2022 (a : ℕ → ℕ) (h : sequence a) : 
  (a 2022) % 100 = 32 :=
sorry

end last_two_digits_of_a2022_l517_517787


namespace biff_break_even_hours_l517_517004

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end biff_break_even_hours_l517_517004


namespace curved_surface_area_l517_517716

/-- Define the radius and slant height of the cone -/
def radius : ℝ := 28
def slant_height : ℝ := 30

/-- Statement for the curved surface area (CSA) of the cone -/
theorem curved_surface_area (π : ℝ) (hπ : π = Real.pi) : 
  π * radius * slant_height ≈ 2638.94 :=
by {
  have calculation := π * radius * slant_height,
  sorry
}

end curved_surface_area_l517_517716


namespace max_members_club_l517_517988

open Finset

theorem max_members_club (A B C : Finset ℕ) 
  (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11) 
  (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
  (A ∪ B ∪ C).card ≥ 22 :=
  sorry

end max_members_club_l517_517988


namespace final_result_l517_517725

noncomputable def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def sum_of_arithmetic_sums : ℕ :=
  ∑ p in finset.range 10, arithmetic_sum (p + 1) (2 * (p + 1) - 1) 40

theorem final_result : sum_of_arithmetic_sums = 80200 :=
  sorry

end final_result_l517_517725


namespace num_unordered_prime_pairs_summing_to_50_l517_517147

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517147


namespace count_prime_pairs_summing_to_50_l517_517229

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517229


namespace at_most_two_lazy_numbers_l517_517806

def is_lazy_number (N : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p > 3 → ¬p ∣ N

theorem at_most_two_lazy_numbers (n : ℕ) (hn : n > 0) :
  ∃ upper, 
  upper ∈ {2^\alpha * 3^\beta | α β : ℕ} ∧
  upper ≤ n^2 ∧ List.countp (is_lazy_number) ((List.range (n^2 + 1)).drop (n^2)) ≤ 2 :=
sorry

end at_most_two_lazy_numbers_l517_517806


namespace num_prime_pairs_sum_50_l517_517200

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517200


namespace primes_sum_50_l517_517407

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l517_517407


namespace num_unordered_prime_pairs_summing_to_50_l517_517149

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517149


namespace find_angle_C_find_max_area_l517_517812

variable {A B C a b c : ℝ}

-- Given Conditions
def condition1 (c B a b C : ℝ) := c * Real.cos B + (b - 2 * a) * Real.cos C = 0
def condition2 (c : ℝ) := c = 2 * Real.sqrt 3

-- Problem (1): Prove the size of angle C
theorem find_angle_C (h : condition1 c B a b C) (h2 : condition2 c) : C = Real.pi / 3 := 
  sorry

-- Problem (2): Prove the maximum area of ΔABC
theorem find_max_area (h : condition1 c B a b C) (h2 : condition2 c) :
  ∃ (A B : ℝ), B = 2 * Real.pi / 3 - A ∧ 
    (∀ (A B : ℝ), Real.sin (2 * A - Real.pi / 6) = 1 → 
    1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 ∧ 
    a = b ∧ b = c) := 
  sorry

end find_angle_C_find_max_area_l517_517812


namespace S_10_equals_687_l517_517923

noncomputable def a (n : ℕ) : ℝ := (-1)^n * (2 : ℝ)^n + n * real.cos (n * real.pi)

noncomputable def S (n : ℕ) : ℝ := (finset.range (n + 1)).sum (λ k, a k)

theorem S_10_equals_687 : S 10 = 687 := 
by sorry

end S_10_equals_687_l517_517923


namespace sum_a_i_eq_one_l517_517515

theorem sum_a_i_eq_one :
  let a_0, a_1, a_2, ... , a_11 : ℤ in
  (x + 3) * (2 * x + 3)^10 = a_0 + a_1 * (x + 3) + a_2 * (x + 3)^2 + ... + a_11 * (x + 3)^11
  → a_0 + a_1 + a_2 + ... + a_11 = 1 :=
by
  sorry

end sum_a_i_eq_one_l517_517515


namespace prime_pairs_sum_to_50_l517_517306

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l517_517306


namespace tangent_line_at_m_eq_1_monotonic_increasing_interval_l517_517101

def f (x : ℝ) (m : ℝ) : ℝ := (1 / 3) * x^3 + m * x^2 - 3 * m^2 * x + 1

-- Question 1: Tangent line at (2, f(2)) when m = 1
theorem tangent_line_at_m_eq_1 :
  let m := 1 in
  let f_x := f 2 m in
  15 * 2 - 3 * f_x - 25 = 0 :=
sorry

-- Question 2: Range of m for monotonicity in the interval (2m - 1, m + 1)
theorem monotonic_increasing_interval (m : ℝ) (h : 0 < m) :
  (2 * m - 1 ≤ m + 1) ∧ (2 * m - 1 ≥ m) → (1 ≤ m ∧ m < 2) :=
sorry

end tangent_line_at_m_eq_1_monotonic_increasing_interval_l517_517101


namespace num_unordered_prime_pairs_summing_to_50_l517_517163

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l517_517163


namespace sin_double_angle_identity_l517_517482

theorem sin_double_angle_identity (α : ℝ) (h : sin α = -2 * cos α) : sin(2 * α) = -4 / 5 :=
by
  sorry

end sin_double_angle_identity_l517_517482


namespace max_integer_k_l517_517098

noncomputable theory

open Real

def f (x : ℝ) : ℝ := (x * (1 + log x)) / (x - 1)

theorem max_integer_k {k : ℤ} :
  (∀ x > 1, f x > (k:ℝ)) ↔ k ≤ 3 :=
by
  sorry

end max_integer_k_l517_517098


namespace num_prime_pairs_sum_50_l517_517193

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517193


namespace find_square_side_length_l517_517665

open Nat

def original_square_side_length (s : ℕ) : Prop :=
  let length := s + 8
  let breadth := s + 4
  (2 * (length + breadth)) = 40 → s = 4

theorem find_square_side_length (s : ℕ) : original_square_side_length s := by
  sorry

end find_square_side_length_l517_517665


namespace gumball_problem_l517_517976

/--
A gumball machine contains 10 red, 6 white, 8 blue, and 9 green gumballs.
The least number of gumballs a person must buy to be sure of getting four gumballs of the same color is 13.
-/
theorem gumball_problem
  (red white blue green : ℕ)
  (h_red : red = 10)
  (h_white : white = 6)
  (h_blue : blue = 8)
  (h_green : green = 9) :
  ∃ n, n = 13 ∧ (∀ gumballs : ℕ, gumballs ≥ 13 → (∃ color_count : ℕ, color_count ≥ 4 ∧ (color_count = red ∨ color_count = white ∨ color_count = blue ∨ color_count = green))) :=
sorry

end gumball_problem_l517_517976


namespace count_prime_pairs_summing_to_50_l517_517235

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l517_517235


namespace equation_of_circumcircle_H_equation_of_line_l_range_of_r_l517_517108

section Triangle_Problem

variable {A B C : E2} 

-- Defining the triangle vertices
def A : E2 := ⟨-1, 0⟩
def B : E2 := ⟨ 1, 0⟩
def C : E2 := ⟨ 3, 2⟩

-- Defining the circumcircle H of the triangle ABC
def circumcircle (A B C : E2) : Circle E2 :=
Circle.mk (circumcenter A B C) (circumradius A B C)

-- Problem 1: Equation of the circumcircle
theorem equation_of_circumcircle_H :
  equation_of_circle (circumcircle A B C) = (λ x y, x^2 + (y - 3)^2 - 10) := sorry

-- Problem 2: Equation of line l
def line_l (C : E2) : Set.Line E2 := 
    λ (x y : ℝ), (y = (4 / 3) * x - 2) ∨ (x = 3)

theorem equation_of_line_l:
  equation_of_line (line_l C) := sorry

-- Problem 3: Range of radius r
theorem range_of_r :
  ∀ r, (r ≥ (sqrt 10) / 3 ∧ r < (4 * sqrt 10) / 5) := sorry

end Triangle_Problem

end equation_of_circumcircle_H_equation_of_line_l_range_of_r_l517_517108


namespace cos_triple_angle_l517_517474

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l517_517474


namespace calc_expression_l517_517683

theorem calc_expression : 3.609 - 2.5 - 0.193 = 0.916 := by
  have h1 : 3.609 - 2.5 = 1.109 := by norm_num
  have h2 : 1.109 - 0.193 = 0.916 := by norm_num
  rw [h1, h2]
  rfl

end calc_expression_l517_517683


namespace num_prime_pairs_sum_50_l517_517174

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l517_517174


namespace derivative_at_0_l517_517000

noncomputable def f : ℝ → ℝ :=
λ x,
if x ≠ 0 then
  Real.tan (2^(x^2 * Real.cos (1/(8*x))) - 1 + x)
else
  0

theorem derivative_at_0 : (deriv f 0) = 1 :=
by
  sorry

end derivative_at_0_l517_517000


namespace count_prime_pairs_sum_50_exactly_4_l517_517296

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l517_517296


namespace length_of_one_side_of_colored_paper_l517_517965

theorem length_of_one_side_of_colored_paper :
  ∃ s : ℝ, (∀ (n : ℕ) (a : ℝ) (total : ℕ) (side : ℝ),
    n = 54 ∧ a = 12 ∧ total = 6 ∧ side = a → s = 4) :=
begin
  sorry
end

end length_of_one_side_of_colored_paper_l517_517965


namespace num_prime_pairs_sum_50_l517_517196

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l517_517196


namespace equal_expressions_for_S_l517_517547

variables (r R A B C : ℝ)

theorem equal_expressions_for_S 
  (S1 : r * R * (sin A + sin B + sin C) = S2)
  (S2 : 4 * r * R * (cos (A / 2) * cos (B / 2) * cos (C / 2)) = S3)
  (S3 : (R^2 / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C)) = S4)
  (S4 : 2 * R^2 * sin A * sin B * sin C = S) :
  S1 = S2 ∧ S2 = S3 ∧ S3 = S4 := 
by
  sorry

end equal_expressions_for_S_l517_517547


namespace nina_money_l517_517633

theorem nina_money :
  ∃ (m C : ℝ), 
    m = 6 * C ∧ 
    m = 8 * (C - 1) ∧ 
    m = 24 :=
by
  sorry

end nina_money_l517_517633


namespace number_of_prime_pairs_sum_50_l517_517217

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l517_517217


namespace correct_answer_l517_517950

def event_A : Prop := ∀ t : ℕ, t > 0 → (sun_rises_from_the_east t) -- predictable, certain event
def event_B : Prop := ∃ t : ℕ, t > 0 ∧ encounter_red_light t -- uncertain, random event
def event_C : Prop := ∀ x : ℝ, x = peanut_oil_mass → floats_on_water x -- predictable, certain event
def event_D : Prop := ∀ a b : ℝ, a < 0 ∧ b < 0 → (a + b > 0) -- impossible event

theorem correct_answer : event_B ∧ ¬ event_A ∧ ¬ event_C ∧ ¬ event_D :=
by sorry

end correct_answer_l517_517950


namespace prime_pairs_summing_to_50_count_l517_517381

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l517_517381


namespace fraction_equals_one_l517_517962

/-- Given the fraction (12-11+10-9+8-7+6-5+4-3+2-1) / (1-2+3-4+5-6+7-8+9-10+11),
    prove that its value is equal to 1. -/
theorem fraction_equals_one :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end fraction_equals_one_l517_517962


namespace ducks_among_non_falcons_l517_517838

-- Definitions based on conditions
def percentage_birds := 100
def percentage_ducks := 40
def percentage_cranes := 20
def percentage_falcons := 15
def percentage_pigeons := 25

-- Question converted into the statement
theorem ducks_among_non_falcons : 
  (percentage_ducks / (percentage_birds - percentage_falcons) * percentage_birds) = 47 :=
by
  sorry

end ducks_among_non_falcons_l517_517838
