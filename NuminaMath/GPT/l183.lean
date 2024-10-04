import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Floor
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Analysis.Calculus.Distance
import Mathlib.Analysis.Calculus.Inverse
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Intervals
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Integration
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import data.nat.gcd

namespace min_area_right_triangle_l183_183203

/-- Minimum area of a right triangle with sides 6 and 8 units long. -/
theorem min_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8)
: 15.87 ≤ min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2))) :=
by
  sorry

end min_area_right_triangle_l183_183203


namespace max_limping_knight_moves_l183_183238

-- Definitions of moves and the board
inductive Move
| normal
| short

structure Board := (rows : ℕ) (cols : ℕ)
def Chessboard : Board := ⟨5, 6⟩

-- Function to check if a cell is within the bounds of the board
def is_within_bounds (b : Board) (r c : ℕ) : Prop :=
  r < b.rows ∧ c < b.cols

-- Predicate to determine if a knight's move is valid
def valid_knight_move (b : Board) (r1 c1 r2 c2 : ℕ) (m : Move) : Prop :=
  is_within_bounds b r2 c2 ∧
  (
    (m = Move.normal ∧ ((abs (r1 - r2) = 2 ∧ abs (c1 - c2) = 1) ∨ (abs (r1 - r2) = 1 ∧ abs (c1 - c2) = 2))) ∨
    (m = Move.short ∧ abs (r1 - r2) = 1 ∧ abs (c1 - c2) = 1)
  )

-- Define the concept of a limping knight move sequence
def limping_knight_moves (start : (ℕ × ℕ)) (seq : List (ℕ × ℕ)) : Prop :=
  (∀ i < seq.length, valid_knight_move Chessboard (seq.nth i).1.1 (seq.nth i).1.2 (seq.nth (i+1)).1.1 (seq.nth (i+1)).1.2 (if i % 2 = 0 then Move.normal else Move.short)) ∧
  seq.nodup

-- The lean theorem statement
theorem max_limping_knight_moves :
  ∀ (start : ℕ × ℕ), is_within_bounds Chessboard start.1 start.2 →
  ∃ seq : List (ℕ × ℕ), seq.head = start ∧ limping_knight_moves start seq ∧ seq.length ≤ 25 :=
by sorry

end max_limping_knight_moves_l183_183238


namespace least_positive_angle_is_75_l183_183711

noncomputable def least_positive_angle (θ : ℝ) : Prop :=
  cos (10 * Real.pi / 180) = sin (15 * Real.pi / 180) + sin θ

theorem least_positive_angle_is_75 :
  least_positive_angle (75 * Real.pi / 180) :=
by
  sorry

end least_positive_angle_is_75_l183_183711


namespace sum_of_exponentials_is_zero_l183_183662

theorem sum_of_exponentials_is_zero : 
  (∑ k in Finset.range 16, Complex.exp ((2 * Real.pi * k.succ : ℝ) * Complex.I / 17)) = 0 :=
by sorry

end sum_of_exponentials_is_zero_l183_183662


namespace problem_statement_l183_183022

theorem problem_statement (x : ℝ) (p : Prop := (x^2 - x - 2 < 0)) (q : Prop := (log 2 x < 1)) :
  (p → q) ∧ (¬(q → p)) :=
by
  -- proofs go here
  sorry

end problem_statement_l183_183022


namespace total_litter_weight_l183_183007

-- Definitions of the conditions
def gina_bags : ℕ := 2
def neighborhood_multiplier : ℕ := 82
def bag_weight : ℕ := 4

-- Representing the total calculation
def neighborhood_bags : ℕ := neighborhood_multiplier * gina_bags
def total_bags : ℕ := neighborhood_bags + gina_bags

def total_weight : ℕ := total_bags * bag_weight

-- Statement of the problem
theorem total_litter_weight : total_weight = 664 :=
by
  sorry

end total_litter_weight_l183_183007


namespace sum_of_exponentials_is_zero_l183_183661

theorem sum_of_exponentials_is_zero : 
  (∑ k in Finset.range 16, Complex.exp ((2 * Real.pi * k.succ : ℝ) * Complex.I / 17)) = 0 :=
by sorry

end sum_of_exponentials_is_zero_l183_183661


namespace max_slope_of_circle_l183_183171

theorem max_slope_of_circle (x y : ℝ) 
  (h : x^2 + y^2 - 6 * x - 6 * y + 12 = 0) : 
  ∃ k : ℝ, k = 3 + 2 * Real.sqrt 2 ∧ ∀ k' : ℝ, (x = 0 → k' = 0) ∧ (x ≠ 0 → y = k' * x → k' ≤ k) :=
sorry

end max_slope_of_circle_l183_183171


namespace combined_depths_underwater_l183_183259

theorem combined_depths_underwater :
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  Ron_depth + Sam_depth + Dean_depth = 13 :=
by
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  show Ron_depth + Sam_depth + Dean_depth = 13
  sorry

end combined_depths_underwater_l183_183259


namespace division_probability_l183_183287

def set_r : Set ℤ := {r | -4 < r ∧ r < 7}
def set_k : Set ℤ := {k | 2 < k ∧ k < 9}

theorem division_probability : 
  let valid_pairs := (set_r.product set_k).count (λ (rk : ℤ × ℤ), rk.2 ∣ rk.1)
  let total_pairs := set_r.card * set_k.card
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 1 / 5 :=
by 
  sorry

end division_probability_l183_183287


namespace find_other_divisor_l183_183347

noncomputable def is_divisible (a b : ℕ) : Prop := b % a = 0

theorem find_other_divisor (n : ℕ) (h : n = 1010 - 2)
    (h16 : is_divisible 16 n)
    (h18 : is_divisible 18 n)
    (h21 : is_divisible 21 n)
    (h28 : is_divisible 28 n) :
    ∃ (d : ℕ), d ≠ 16 ∧ d ≠ 18 ∧ d ≠ 21 ∧ d ≠ 28 ∧ is_divisible d n :=
begin
  sorry
end

end find_other_divisor_l183_183347


namespace sum_arithmetic_seq_mk_l183_183831

open Nat

variable {α : Type*} [LinearOrderedField α]

def arithmetic_seq {α : Type*} (a d : α) (n : ℕ) : α :=
  a + d * (n - 1)

theorem sum_arithmetic_seq_mk
  (aₙ aₖ : α) (k m : ℕ) (hk : k ≠ 0) (hm : m ≠ 0) (hkm : m ≠ k)
  (hₙ : aₙ = (1 : α) / k) (hₖ : aₖ = (1 : α) / m)
  (d : α) (hd : d = (aₖ - aₙ) / (k - m)) :
  let a₁ : α := aₙ - (n : α -1) * d
  let mk := m * k
  let a_mk := arithmetic_seq a₁ d mk
  (sum_range mk (λ n, a₁ + d * n) = (mk + 1) / 2) :=
sorry

end sum_arithmetic_seq_mk_l183_183831


namespace simplify_expression_l183_183643

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183643


namespace arithmetic_sum_l183_183744

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n * d)

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum :
  ∀ (a d : ℕ),
  arithmetic_sequence a d 2 + arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 12 →
  sum_first_n_terms a d 7 = 28 :=
by
  sorry

end arithmetic_sum_l183_183744


namespace band_fundraising_goal_exceed_l183_183956

theorem band_fundraising_goal_exceed
    (goal : ℕ)
    (basic_wash_cost deluxe_wash_cost premium_wash_cost cookie_cost : ℕ)
    (basic_wash_families deluxe_wash_families premium_wash_families sold_cookies : ℕ)
    (total_earnings : ℤ) :
    
    goal = 150 →
    basic_wash_cost = 5 →
    deluxe_wash_cost = 8 →
    premium_wash_cost = 12 →
    cookie_cost = 2 →
    basic_wash_families = 10 →
    deluxe_wash_families = 6 →
    premium_wash_families = 2 →
    sold_cookies = 30 →
    total_earnings = 
        (basic_wash_cost * basic_wash_families +
         deluxe_wash_cost * deluxe_wash_families +
         premium_wash_cost * premium_wash_families +
         cookie_cost * sold_cookies : ℤ) →
    (goal : ℤ) - total_earnings = -32 :=
by
  intros h_goal h_basic h_deluxe h_premium h_cookie h_basic_fam h_deluxe_fam h_premium_fam h_sold_cookies h_total_earnings
  sorry

end band_fundraising_goal_exceed_l183_183956


namespace mixed_groups_count_l183_183975

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183975


namespace ring_width_l183_183585

noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def outerCircumference : ℝ := 528 / 7

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

theorem ring_width :
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  r_outer - r_inner = 4 :=
by
  -- Definitions for inner and outer radius
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  -- Proof goes here
  sorry

end ring_width_l183_183585


namespace square_side_length_false_l183_183170

theorem square_side_length_false (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 8) (h2 : side_length = 4) :
  ¬(4 * side_length = perimeter) :=
by
  sorry

end square_side_length_false_l183_183170


namespace triangle_area_ratio_l183_183091

theorem triangle_area_ratio (P Q R S : Type) [lin_univ : MetricSpace Q] [PointBased : MetricSpace P] [TriPQR : Triangle P Q R]
  (h1 : PQ = 45) (h2 : PR = 75) (h3 : QR = 64) (h4 : AngleBisector P S Q R) :
  (areaRatio PQS PRS = 3 / 5) :=
sorry

end triangle_area_ratio_l183_183091


namespace area_of_shaded_triangle_l183_183085

-- Definitions of the conditions
def AC := 4
def BC := 3
def BD := 10
def CD := BD - BC

-- Statement of the proof problem
theorem area_of_shaded_triangle :
  (1 / 2 * CD * AC = 14) := by
  sorry

end area_of_shaded_triangle_l183_183085


namespace mixed_groups_count_l183_183971

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183971


namespace circumscribed_polygon_triangle_l183_183898

theorem circumscribed_polygon_triangle
  (polygon: Type) (n: ℕ) [fintype polygon] [decidable_eq polygon] (sides : polygon → ℝ) 
  (in_circle : ℝ) (circumscribed : ∀ (a b: polygon), a ≠ b → side a > 0)
  (largest_side : ∃ (AB: polygon), ∀ (e : polygon), sides e ≤ sides AB)
  (side_tangent_point : ∀ (e: polygon), ∃ K: in_circle, K touches (sides e))
  (neighbor_sides: ∀ (AB BC AD: polygon), sides AB ≠ sides BC → sides AB ≠ sides AD)
  (tangent_segments: ∀ (B A: polygon), ∃ BK AK : in_circle, BK + AK = sides AB)
  (ineq_BC: ∀BC BK, sides BC > BK)
  (ineq_AD: ∀AD AK, sides AD > AK)
  (sum_tangents: ∀BK AK, BK + AK = sides AB)
  :
  ∃ (AB AD BC: polygon),
  sides BC + sides AD > sides AB :=
by 
  -- proof steps will come here
  sorry

end circumscribed_polygon_triangle_l183_183898


namespace part1_part2_l183_183755

section part1

variables (a : ℝ) (x : ℝ)
def A : set ℝ := {x | 2 * a - 1 < x ∧ x < -a}
def B : set ℝ := {x | |x - 1| < 2 }

theorem part1 (h : a = -1) : (A ᶜ ∪ B) = {x | x ≤ -3 ∨ x > -1} :=
sorry

end part1

section part2

variables (a : ℝ) (x : ℝ)
def A : set ℝ := {x | 2 * a - 1 < x ∧ x < -a}
def B : set ℝ := {x | |x - 1| < 2 }

theorem part2 : (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∉ A ∧ x ∈ B) ↔ 0 ≤ a :=
sorry

end part2

end part1_part2_l183_183755


namespace least_positive_angle_l183_183706

theorem least_positive_angle (θ : ℝ) (h : θ > 0 ∧ θ ≤ 360) : 
  (cos 10 = sin 15 + sin θ) → θ = 32.5 :=
by 
  sorry

end least_positive_angle_l183_183706


namespace probability_diagonals_intersect_l183_183590

-- Define a structure for the heptagon
structure Heptagon where
  vertices : Finset ℕ
  h_vertices : vertices.card = 7

-- Define a function for combinations (n choose k)
def combination (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Define the total number of diagonals, sides, and pairs of diagonals
def total_pairs (n : ℕ) : ℕ := combination n 2

def sides (n : ℕ) : ℕ := n

def diagonals (n : ℕ) : ℕ := total_pairs n - sides n

def pairs_of_diagonals (n : ℕ) : ℕ := combination (diagonals n) 2

def intersecting_pairs (n : ℕ) : ℕ := combination n 4

-- Define the probability of intersection
def probability_of_intersection (n : ℕ) (h : n = 7) : ℚ :=
  intersecting_pairs n / pairs_of_diagonals n

-- The main theorem asserting the required probability for a regular heptagon
theorem probability_diagonals_intersect :
  probability_of_intersection 7 rfl = 5 / 13 :=
by sorry

end probability_diagonals_intersect_l183_183590


namespace hyperbola_eccentricity_l183_183944

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) :
  (∃ e : ℝ, e = 3^.sqrt) →
  (∃ a b c : ℝ, a^2 = m ∧ b^2 = 4 ∧ c = (a^2 + b^2)^.sqrt ∧ e = c / a) →
  m = 2 :=
by
  sorry

end hyperbola_eccentricity_l183_183944


namespace power_identity_l183_183375

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end power_identity_l183_183375


namespace angle_equivalence_modulo_l183_183835

-- Defining the given angles
def theta1 : ℤ := -510
def theta2 : ℤ := 210

-- Proving that the angles are equivalent modulo 360
theorem angle_equivalence_modulo : theta1 % 360 = theta2 % 360 :=
by sorry

end angle_equivalence_modulo_l183_183835


namespace range_of_a_l183_183407

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x,
  if x ≤ 0 then -x^2 + (2 * a - 2) * x
  else x^3 - (3 * a + 3) * x^2 + a * x

noncomputable def f' (a : ℝ) : ℝ → ℝ := λ x,
  if x ≤ 0 then -2 * x + 2 * a - 2
  else 3 * x^2 - 6 * (a + 1) * x + a

theorem range_of_a :
  (∀ (x1 x2 x3 : ℝ), (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) → (f' a x1 = f' a x2 ∧ f' a x1 = f' a x3)) →
  (-1 < a ∧ a < 2) :=
by {
  intro h,
  sorry
}

end range_of_a_l183_183407


namespace number_of_full_houses_l183_183893

theorem number_of_full_houses :
  let V := 13 in
  let choose (n k : ℕ) := Nat.choose n k in
  (V * (V - 1)) * (choose 4 3) * (choose 4 2) = 3744 := by
  let V := 13
  let choose (n k : ℕ) := Nat.choose n k
  calc
    (V * (V - 1)) * (choose 4 3) * (choose 4 2)
      = 13 * 12 * 4 * 6 : by rfl
    ... = 3744 : by norm_num

end number_of_full_houses_l183_183893


namespace find_tan_value_l183_183808

theorem find_tan_value (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log a 16 = 2) : tan (a * real.pi / 3) = real.sqrt 3 :=
  sorry

end find_tan_value_l183_183808


namespace alice_gave_rattle_to_first_l183_183186

-- Definitions based on the problem statements
def FirstBrother : Type := 
{ owns_rattle : Prop } -- First brother first says ownership

def SecondBrother : Type :=
{ owns_rattle : Prop } -- Second brother responds to the rattle ownership query

variables (firstBrother secondBrother : Prop)

-- The conditions based on their answers
def firstBrother_answer_yes (firstBrother : FirstBrother) := 
  firstBrother.owns_rattle

def secondBrother_answer_no (secondBrother : SecondBrother) :=
  ¬ secondBrother.owns_rattle

-- The main theorem statement that we need to prove
theorem alice_gave_rattle_to_first (firstBrother : FirstBrother) : firstBrother_answer_yes firstBrother → secondBrother_answer_no secondBrother → (∃ (owner : FirstBrother), firstBrother.owns_rattle) :=
by 
  sorry

end alice_gave_rattle_to_first_l183_183186


namespace distance_to_asymptote_l183_183927

noncomputable def distance_from_asymptote : ℝ :=
  let x0 := 3
  let y0 := 0
  let A := 3
  let B := -4
  let C := 0
  @Real.sqrt ((A^2) + (B^2))^{-1} * abs (A * x0 + B * y0 + C)

theorem distance_to_asymptote : distance_from_asymptote = (9 / 5) := sorry

end distance_to_asymptote_l183_183927


namespace relation_P_Q_l183_183531

def P : Set ℝ := {x | x ≠ 0}
def Q : Set ℝ := {x | x > 0}
def complement_P : Set ℝ := {0}

theorem relation_P_Q : Q ∩ complement_P = ∅ := 
by sorry

end relation_P_Q_l183_183531


namespace eight_consecutive_odd_sum_l183_183686

theorem eight_consecutive_odd_sum (S : ℤ) (h : S ∈ {120, 200, 248, 296, 344}) : 
  ¬ ∃ n : ℤ, (n % 2 = 1) ∧ (S = 8 * n + 56) :=
by
  sorry

end eight_consecutive_odd_sum_l183_183686


namespace range_of_k_l183_183440

-- Let K be a real number
variable (k : ℝ)

-- Define the conditions for the problem
def cond1 : Prop := ∀ x : ℝ, 0 < x → log 10 (k * x) = 2 * log 10 (x + 1)

-- The equivalent quadratic equation derived from the logarithmic condition
def quadratic_eq (k : ℝ) : ℝ → ℝ := λ (x : ℝ), x^2 + (2 - k) * x + 1

-- Discriminant analysis to ensure only one real root
def discriminant_eq_zero (k : ℝ) : Prop := (2 - k)^2 - 4 * 1 * 1 = 0

-- Prove that for the given conditions, the range of k is (-∞, 0) ∪ {4}
theorem range_of_k (k : ℝ) (h : cond1 k) : k ∈ set.Ioo (-∞) 0 ∪ {4} :=
sorry

end range_of_k_l183_183440


namespace project_assignment_total_count_l183_183184

theorem project_assignment_total_count :
  (∃ f : Fin 5 → Fin 3, 
      ∀ t : Fin 3, (∀ i : Fin 5, (f i) = t → (t.val ≠ 0 ∨ (card {i | f i = t} ≤ 2)))) ∧ 
    (∀ t : Fin 3, (card {i : Fin 5 | f i = t}) ≥ 1)
    → ∃ n, n = 130 := 
begin
  sorry
end

end project_assignment_total_count_l183_183184


namespace trigonometric_solutions_l183_183228

theorem trigonometric_solutions :
  (∀ α : ℝ, tan α = -4 / 3 ∧ (∃ k : ℤ, (2 * k * π < α ∧ α < 2 * k * π + π)) → 
    sin α = -4 / 5 ∧ cos α = 3 / 5) ∧
  (sin (25 * π / 6) + cos (26 * π / 3) + tan (-25 * π / 4) = -1) :=
by
  sorry

end trigonometric_solutions_l183_183228


namespace polygon_interior_angles_l183_183699

theorem polygon_interior_angles (n : ℕ) (h1 : ∀ i : fin n, 170 ≤ 177) : (n-2) * 180 = 177 * n → n = 120 :=
by
  sorry

end polygon_interior_angles_l183_183699


namespace opposite_of_8_is_neg8_l183_183169

theorem opposite_of_8_is_neg8 : ∃ x : ℤ, 8 + x = 0 ∧ x = -8 :=
by
  use -8
  split
  · exact by norm_num
  · rfl

end opposite_of_8_is_neg8_l183_183169


namespace shortest_distance_to_circle_is_one_l183_183196

-- Define the conditions: the circle equation
def circle (x y : ℝ) : Prop := x^2 - 8 * x + y^2 - 6 * y + 9 = 0

-- Define the function to calculate the shortest distance
def shortest_distance_from_origin_to_circle : ℝ :=
  let c_x := 4 in
  let c_y := 3 in
  let radius := 4 in
  let origin_distance := Real.sqrt ((c_x ^ 2) + (c_y ^ 2)) in
  origin_distance - radius

-- Theorem stating shortest distance is 1
theorem shortest_distance_to_circle_is_one : shortest_distance_from_origin_to_circle = 1 := by
  sorry

end shortest_distance_to_circle_is_one_l183_183196


namespace clock_face_rectangles_l183_183151

theorem clock_face_rectangles : 
    let points := (1 : ℕ) .. 12
    ∃ (rectangles : ℕ), rectangles = 15 ∧ 
    (∀ (quad : set ℕ), quad ⊆ points ∧ quad.card = 4 → 
     (quad_is_rectangle quad ↔ quad ⊆ set.of_unique_elements {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12})) :=
begin
  sorry,
end

end clock_face_rectangles_l183_183151


namespace simplify_complex_fraction_l183_183907

theorem simplify_complex_fraction :
  (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i :=
by
  sorry

end simplify_complex_fraction_l183_183907


namespace inverse_function_a_eq_1_max_value_range_of_a_l183_183735

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 / (1 + a * 2^x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - f (x - 1) a
noncomputable def f_inv (x : ℝ) : ℝ := Real.log ((1 - x) / x)

-- Statement 1: Inverse of f when a = 1
theorem inverse_function_a_eq_1 (x : ℝ) (hx : 0 < x ∧ x < 1) :
  (∀ x ∈ Ioo 0 1, f_inv x = Real.log ((1 - x) / x)) :=
  sorry

-- Statement 2: Maximum value of y = f(x) * f(-x)
theorem max_value (a : ℝ) (ha : 0 < a) :
  (∃ x : ℝ, ∀ x : ℝ, f x a * f (-x) a ≤ 1 / (a + 1)^2) :=
  sorry

-- Statement 3: Range of values for a such that g(x) >= g(0) for x ∈ (-∞, 0]
theorem range_of_a (a : ℝ) (ha : 0 < a) :
  (∀ x ∈ Iic 0, g x a ≥ g 0 a) ↔ (0 < a ∧ a ≤ Real.sqrt 2) :=
  sorry

end inverse_function_a_eq_1_max_value_range_of_a_l183_183735


namespace triangle_area_l183_183816

theorem triangle_area {α β γ : Type} [linear_ordered_field α] [has_sin γ] [topological_space γ]
  [topological_space β] [topological_space α] [has_coe_t α γ] [has_coe_t β γ] [has_coe_t γ α]
  [has_coe_t α β]: 
  (a : α = 6) (B : β = 30) (C : β = 120) :
  let A := 180 - B - C in
  let b := a in
  (1 / 2 * a * b * sin C = 9 * sqrt 3) :=
by
  -- The proof goes here
  sorry

end triangle_area_l183_183816


namespace circumcircle_BCN_tangent_incircle_l183_183105

variables {A B C D H M N : Type}
variables [Triangle ABC] [Incircle ω ABC] [TangencyPoint D ω BC]
variables [Foot H A BC] [Midpoint M A H] [LineSegment DM]
variables [Intersection N DM ω]

theorem circumcircle_BCN_tangent_incircle :
  Tangent (Circumcircle BCN) ω :=
sorry

end circumcircle_BCN_tangent_incircle_l183_183105


namespace exp_add_l183_183139

theorem exp_add (a : ℝ) (x₁ x₂ : ℝ) : a^(x₁ + x₂) = a^x₁ * a^x₂ :=
sorry

end exp_add_l183_183139


namespace area_of_rectangle_l183_183583

theorem area_of_rectangle (S R L B A : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S^2 = 1600)
  (h4 : B = 10)
  (h5 : A = L * B) : 
  A = 160 := 
sorry

end area_of_rectangle_l183_183583


namespace solve_equation_l183_183088

theorem solve_equation (x : ℝ) (hx : x ≠ 0) 
  (h : 1 / 4 + 8 / x = 13 / x + 1 / 8) : 
  x = 40 :=
sorry

end solve_equation_l183_183088


namespace probability_one_doctor_one_nurse_l183_183074

/-- Probability of selecting exactly 1 doctor and 1 nurse
    from 3 doctors and 2 nurses out of 5 individuals --/
theorem probability_one_doctor_one_nurse :
  let total_ways := ∑(k in range 5.choose 2, 1),
      doctor_nurse_pairs := 3.choose 1 * 2.choose 1 in
  (doctor_nurse_pairs.to_rat / total_ways.to_rat) = (3/5 : ℚ) := 
sorry

end probability_one_doctor_one_nurse_l183_183074


namespace length_of_goods_train_l183_183603

-- Conditions given in the problem
def speed_mans_train_kmph : ℝ := 60
def speed_goods_train_kmph : ℝ := 30
def time_taken_to_pass_s : ℝ := 12

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 5 / 18

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_mans_train_kmph + speed_goods_train_kmph)

-- Declaration of the proof
theorem length_of_goods_train : real :=
  let length_goods_train_m := relative_speed_mps * time_taken_to_pass_s
  length_goods_train_m = 300 := sorry

end length_of_goods_train_l183_183603


namespace min_value_of_expression_l183_183416

theorem min_value_of_expression (a b c m : ℝ) (h1 : 0 < a) (h2 : 0 < c)
    (h3 : a + b * m + c - 2 = 0) (h4 : ∀ l : ℝ, abs (4 * a + 0 * b + c - 2) / sqrt (a ^ 2 + b ^ 2) ≤ 3) :
    (1 / (2 * a) + 2 / c) = 9 / 4 :=
by
  sorry

end min_value_of_expression_l183_183416


namespace find_constants_l183_183310

theorem find_constants : 
  ∃ (a b : ℝ), a • (⟨1, 4⟩ : ℝ × ℝ) + b • (⟨3, -2⟩ : ℝ × ℝ) = (⟨5, 6⟩ : ℝ × ℝ) ∧ a = 2 ∧ b = 1 :=
by 
  sorry

end find_constants_l183_183310


namespace zeros_in_decimal_representation_l183_183057

theorem zeros_in_decimal_representation :
  let x := 1 / (2^3 * 5^7) in
  let adjusted := (2^4) / (10^7) in
  adjusted = (16 / 10^7) →
  count_zeros_decimal adjusted = 5 :=
by
  sorry

end zeros_in_decimal_representation_l183_183057


namespace distance_to_asymptote_l183_183928

noncomputable def distance_from_asymptote : ℝ :=
  let x0 := 3
  let y0 := 0
  let A := 3
  let B := -4
  let C := 0
  @Real.sqrt ((A^2) + (B^2))^{-1} * abs (A * x0 + B * y0 + C)

theorem distance_to_asymptote : distance_from_asymptote = (9 / 5) := sorry

end distance_to_asymptote_l183_183928


namespace mixed_groups_count_l183_183981

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183981


namespace ratio_of_marbles_l183_183303

def marbles_problem : Prop :=
  let jar1 := 80
  let jar3 := jar1 / 4
  let total_marbles := 260
  ∃ (jar2 : ℕ), jar1 + jar2 + jar3 = total_marbles ∧ jar2 / jar1 = 2

theorem ratio_of_marbles : marbles_problem :=
by
  let jar1 := 80
  let jar3 := jar1 / 4
  let total_marbles := 260
  use 160
  sorry

end ratio_of_marbles_l183_183303


namespace mike_needs_to_save_weeks_l183_183880

-- Definitions based on conditions
def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def discount_phone : ℝ := 0.10
def discount_smartwatch : ℝ := 0.15
def sales_tax : ℝ := 0.07
def initial_percentage : ℝ := 0.40
def weekly_savings : ℝ := 100

-- Proof problem statement
theorem mike_needs_to_save_weeks :
  let total_cost := (phone_cost * (1 - discount_phone)) + (smartwatch_cost * (1 - discount_smartwatch)) in
  let final_amount := total_cost * (1 + sales_tax) in
  let mike_has := final_amount * initial_percentage in
  let amount_needed := final_amount - mike_has in
  let weeks_needed := ⌈amount_needed / weekly_savings⌉ in
  weeks_needed = 11 := 
by sorry

end mike_needs_to_save_weeks_l183_183880


namespace sequence_geometric_and_general_term_sum_of_sequence_l183_183035

theorem sequence_geometric_and_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, S k = 2 * a k - k) : 
  (a 0 = 1) ∧ 
  (∀ k : ℕ, a (k + 1) = 2 * a k + 1) ∧ 
  (∀ k : ℕ, a k = 2^k - 1) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, a k = 2^k - 1)
  (h2 : ∀ k : ℕ, b k = 1 / a (k+1) + 1 / (a k * a (k+1))) :
  T n = 1 - 1 / (2^(n+1) - 1) :=
sorry

end sequence_geometric_and_general_term_sum_of_sequence_l183_183035


namespace volume_CO2_is_7_l183_183587

-- Definitions based on conditions
def Avogadro_law (V1 V2 : ℝ) : Prop := V1 = V2
def molar_ratio (V_CO2 V_O2 : ℝ) : Prop := V_CO2 = 1 / 2 * V_O2
def volume_O2 : ℝ := 14

-- Statement to be proved
theorem volume_CO2_is_7 : ∃ V_CO2 : ℝ, molar_ratio V_CO2 volume_O2 ∧ V_CO2 = 7 := by
  sorry

end volume_CO2_is_7_l183_183587


namespace part_1_geometric_and_general_term_part_2_range_of_a_l183_183788

def sequence (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := (5 * sequence a n - 8) / (sequence a n - 1)

theorem part_1_geometric_and_general_term (a: ℝ) (h : a = 3) :
  ∃ r : ℝ, ∀ n : ℕ, (sequence a n - 2) / (sequence a n - 4) = (-1) * (3 ^ n) ∧ 
  ∀ n : ℕ, sequence a n = 4 - (2 / (3 ^ n + 1)) :=
sorry

theorem part_2_range_of_a (a : ℝ) (h : ∀ n : ℕ, sequence a n > 3) :
  a > 3 :=
sorry

end part_1_geometric_and_general_term_part_2_range_of_a_l183_183788


namespace mixed_groups_count_l183_183972

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183972


namespace number_of_polynomials_is_seven_l183_183460

-- Definitions of what constitutes a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4*x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/5x" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Given set of algebraic expressions
def expressions : List String := 
  ["3/4*x^2", "3ab", "x+5", "y/5x", "-1", "y/3", "a^2-b^2", "a"]

-- Count the number of polynomials in the given expressions
def count_polynomials (exprs : List String) : Nat :=
  exprs.foldr (fun expr count => if is_polynomial expr then count + 1 else count) 0

theorem number_of_polynomials_is_seven : count_polynomials expressions = 7 :=
  by
    sorry

end number_of_polynomials_is_seven_l183_183460


namespace problem1_problem2_problem3_l183_183414

def f (x : ℝ) (b : ℝ) := -x^3 + x^2 + b
def g (x : ℝ) (a : ℝ) := a * Real.log x

-- condition 1
theorem problem1 (b : ℝ) (h1 : ∀ x ∈ Icc (-1/2 : ℝ) 1, f x b ≤ 3/8) : b = 0 := sorry

-- condition 2
theorem problem2 (a : ℝ) (h2 : ∀ x ∈ Icc 1 Real.exp 1, g x a ≥ -x^2 + (a + 2) * x) : a ≤ -1 := sorry

-- condition 3
def F (x : ℝ) (a b : ℝ) :=
  if x < 1 then f x b else g x a

theorem problem3 (a : ℝ) (hb : 0 = 0) (h2 : ∀ x ∈ Icc 1 (Real.exp 1), g x a ≥ -x^2 + (a + 2) * x) (ha_pos : 0 < a) :
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ P.1 * Q.1 = 0 ∧
  F P.1 a 0 = P.2 ∧ F Q.1 a 0 = Q.2 ∧
  (∃ u v : ℝ, P = (u, 0) ∧ Q = (0, v)) := sorry

end problem1_problem2_problem3_l183_183414


namespace math_problem_equivalent_l183_183286

-- Given that the problem requires four distinct integers a, b, c, d which are less than 12 and invertible modulo 12.
def coprime_with_12 (x : ℕ) : Prop := Nat.gcd x 12 = 1

theorem math_problem_equivalent 
  (a b c d : ℕ) (ha : coprime_with_12 a) (hb : coprime_with_12 b) 
  (hc : coprime_with_12 c) (hd : coprime_with_12 d) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c)
  (hbd : b ≠ d) (hcd : c ≠ d) :
  ((a * b * c * d) + (a * b * c) + (a * b * d) + (a * c * d) + (b * c * d)) * Nat.gcd (a * b * c * d) 12 = 1 :=
sorry

end math_problem_equivalent_l183_183286


namespace equilateral_triangle_of_medians_and_angles_l183_183456

theorem equilateral_triangle_of_medians_and_angles 
  {A B C F E : Point}
  (hA : is_triangle A B C)
  (hF : is_median A B F)
  (hE : is_median C B E)
  (hBAF : ∠ B A F = 30)
  (hBCE : ∠ B C E = 30) : 
  is_equilateral A B C := 
sorry

end equilateral_triangle_of_medians_and_angles_l183_183456


namespace minimum_sum_sequence_l183_183855

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n * (a_n 1 + a_n n)) / 2

theorem minimum_sum_sequence : ∃ n : ℕ, S_n n = (n - 24) * (n - 24) - 24 * 24 ∧ (∀ m : ℕ, S_n m ≥ S_n n) ∧ n = 24 := 
by {
  sorry -- Proof omitted
}

end minimum_sum_sequence_l183_183855


namespace circle_center_line_distance_l183_183157

noncomputable def distance_from_center_to_line (x y: ℝ) : ℝ :=
| x * 1 + y * 1 - 2 | / Real.sqrt (1^2 + 1^2)

theorem circle_center_line_distance : 
  (distance_from_center_to_line 1 0 = Real.sqrt 2 / 2) :=
sorry

end circle_center_line_distance_l183_183157


namespace mean_of_roots_of_cubic_eq_l183_183716

theorem mean_of_roots_of_cubic_eq (h : Polynomial.eval 0 (Polynomial.Coeff [−30, 1, 6, 1]) = 0) :
  (∃ x1 x2 x3, x1 + x2 + x3 = 0 ∧ x1 * x2 + x2 * x3 + x3 * x1 = 30 ∧ x1 * x2 * x3 = 0) →
  ∃ mean : RationalNumber,
  mean = -(4 / 3) :=
by
  sorry

end mean_of_roots_of_cubic_eq_l183_183716


namespace geom_sequence_ratio_l183_183746

theorem geom_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∃ q, ∀ n, a n = a 0 * q^n)
  (h2 : 8 * a 1 + a 4 = 0)
  (h3 : ∀ n, S n = a 0 * (1 - (h1.some)^n) / (1 - h1.some)) :
  S 3 / S 1 = 5 := 
sorry

end geom_sequence_ratio_l183_183746


namespace mixed_groups_count_l183_183995

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183995


namespace women_work_hours_per_day_l183_183230

theorem women_work_hours_per_day :
  let men_work : ℕ := 15 * 21 * 8 in
  let women_work_equivalent : ℕ := 21 * 36 * ((2 : ℕ) / 3) in
  let women_work_hours := 2520 / (504) in
  women_work_hours = 5 :=
  by
    sorry

end women_work_hours_per_day_l183_183230


namespace jonah_profit_l183_183357

def cost_per_pineapple (quantity : ℕ) : ℝ :=
  if quantity > 50 then 1.60 else if quantity > 40 then 1.80 else 2.00

def total_cost (quantity : ℕ) : ℝ :=
  cost_per_pineapple quantity * quantity

def bundle_revenue (bundles : ℕ) : ℝ :=
  bundles * 20

def single_ring_revenue (rings : ℕ) : ℝ :=
  rings * 4

def total_revenue (bundles : ℕ) (rings : ℕ) : ℝ :=
  bundle_revenue bundles + single_ring_revenue rings

noncomputable def profit (quantity bundles rings : ℕ) : ℝ :=
  total_revenue bundles rings - total_cost quantity

theorem jonah_profit : profit 60 35 150 = 1204 := by
  sorry

end jonah_profit_l183_183357


namespace slope_of_line_l183_183000

theorem slope_of_line (x y : ℝ) (h : 6 * x + 7 * y - 3 = 0) : - (6 / 7) = -6 / 7 := 
by
  sorry

end slope_of_line_l183_183000


namespace systematic_sampling_correct_l183_183823

def students_total := 400
def sample_size := 50
def first_drawn := 5
def buildings_A_B_C (n : ℕ) : ℕ := 
if n ≤ 200 then 1 else if n ≤ 300 then 2 else 3

theorem systematic_sampling_correct :
  let interval := students_total / sample_size in
  let drawn_set := {n | ∃ k, n = first_drawn + k * interval ∧ n ≤ students_total} in
  let count_in_building (b : ℕ) := (drawn_set.filter (λ n, buildings_A_B_C n = b)).card in
  count_in_building 1 = 25 ∧
  count_in_building 2 = 12 ∧ 
  count_in_building 3 = 13 :=
by
  sorry

end systematic_sampling_correct_l183_183823


namespace min_value_of_expression_l183_183736

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) : x^2 + (1 / 4) * y^2 ≥ 1 / 8 :=
sorry

end min_value_of_expression_l183_183736


namespace john_uses_six_pounds_of_vegetables_l183_183477

-- Define the given conditions:
def pounds_of_beef_bought : ℕ := 4
def pounds_beef_used_in_soup := pounds_of_beef_bought - 1
def pounds_of_vegetables_used := 2 * pounds_beef_used_in_soup

-- Statement to prove:
theorem john_uses_six_pounds_of_vegetables : pounds_of_vegetables_used = 6 :=
by
  sorry

end john_uses_six_pounds_of_vegetables_l183_183477


namespace initial_price_of_sugar_per_kg_l183_183952

theorem initial_price_of_sugar_per_kg
  (initial_price : ℝ)
  (final_price : ℝ)
  (required_reduction : ℝ)
  (initial_price_eq : initial_price = 6)
  (final_price_eq : final_price = 7.5)
  (required_reduction_eq : required_reduction = 0.19999999999999996) :
  initial_price = 6 :=
by
  sorry

end initial_price_of_sugar_per_kg_l183_183952


namespace john_vegetables_l183_183475

theorem john_vegetables (beef_used vege_used : ℕ) :
  beef_used = 4 - 1 →
  vege_used = 2 * beef_used →
  vege_used = 6 :=
by
  intros h_beef_used h_vege_used
  unfold beef_used vege_used
  exact sorry

end john_vegetables_l183_183475


namespace meal_cost_l183_183155

theorem meal_cost :
  ∃ (s c p : ℝ),
  (5 * s + 8 * c + 2 * p = 5.40) ∧
  (3 * s + 11 * c + 2 * p = 4.95) ∧
  (s + c + p = 1.55) :=
sorry

end meal_cost_l183_183155


namespace problem_solution_l183_183150

theorem problem_solution (y : ℤ)
  (h1 : 2 + y ≡ 9 [MOD 27])
  (h2 : 4 + y ≡ 8 [MOD 125])
  (h3 : 6 + y ≡ 49 [MOD 343]) : 
  y ≡ 1 [MOD 105] :=
sorry

end problem_solution_l183_183150


namespace hiker_comparison_l183_183193

theorem hiker_comparison
  (a : ℝ) -- Step length of the second hiker
  (n : ℝ) -- Number of steps the second hiker makes in a certain period
  (h1 : 0 < a) -- Step length is positive
  (h2 : 0 < n) -- Number of steps is positive
  (l1 : ℝ := 0.9 * a) -- Step length of the first hiker (10% shorter)
  (f1 : ℝ := 1.1 * n) -- Step frequency of the first hiker (10% more)
  (d2 : ℝ := a * n) -- Distance covered by the second hiker
  (d1 : ℝ := l1 * f1) -- Distance covered by the first hiker
  : d2 > d1 :=
begin
  -- Conditions: d1 = (0.9 * a) * (1.1 * n) = 0.99 * a * n
  -- d2 = a * n
  -- We need to show a * n > 0.99 * a * n
  show a * n > 0.99 * a * n,
  calc
    a * n = 1 * a * n : by simp
    ... > 0.99 * a * n : by nlinarith [h1, h2]
end

end hiker_comparison_l183_183193


namespace shared_side_triangle_inequality_l183_183223

theorem shared_side_triangle_inequality 
  (P Q R S : Type) 
  [AddGroup P] [AddGroup Q] [AddGroup R] [AddGroup S]
  (PQ PR SR QS QR : ℝ)
  (hPQ : PQ = 7) 
  (hPR : PR = 15) 
  (hSR : SR = 10) 
  (hQS : QS = 25) 
  (hTriangleInequalityPQR : QR > PR - PQ) 
  (hTriangleInequalitySQR : QR > QS - SR) : 
  QR ≥ 15 := 
by {
  intros,
  have h1 : QR > 15 - 7 := by exact hTriangleInequalityPQR,
  have h2 : QR > 25 - 10 := by exact hTriangleInequalitySQR,
  have minQR := max (PR - PQ) (QS - SR),
  linarith,
  sorry
}

end shared_side_triangle_inequality_l183_183223


namespace man_speed_is_10_km_per_hr_l183_183604
-- Definitions based on the conditions
def time_in_hours : ℝ := 36 / 60
def distance_in_km : ℝ := 6

-- The proof statement asserting the speed is 10 km/hr
theorem man_speed_is_10_km_per_hr : (distance_in_km / time_in_hours) = 10 := 
by
  -- Insert proof here
  sorry

end man_speed_is_10_km_per_hr_l183_183604


namespace find_a3_l183_183067

theorem find_a3 (a0 a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, x^4 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4) →
  a3 = -8 :=
by
  sorry

end find_a3_l183_183067


namespace rachel_made_18_dollars_l183_183688

/--
Each chocolate bar in a box costs $2. There are 13 bars in the box. 
Rachel sold all but 4 bars. Prove that Rachel made 18 dollars. 
-/
theorem rachel_made_18_dollars (cost_per_bar : ℕ) (total_bars : ℕ) (bars_left : ℕ) : 
  cost_per_bar = 2 ∧ total_bars = 13 ∧ bars_left = 4 → (total_bars - bars_left) * cost_per_bar = 18 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  exact rfl

end rachel_made_18_dollars_l183_183688


namespace coffee_per_donut_l183_183481

-- Definitions
def pots_cost (cost: ℕ) (price_per_pot: ℕ) : ℕ := cost / price_per_pot
def total_coffee (pots: ℕ) (ounces_per_pot: ℕ) : ℕ := pots * ounces_per_pot
def total_donuts (dozens: ℕ) (donuts_per_dozen: ℕ) : ℕ := dozens * donuts_per_dozen
def ounces_per_donut (total_ounces: ℕ) (total_donuts: ℕ) : ℕ := total_ounces / total_donuts

-- Main Statement
theorem coffee_per_donut :
  let pots := pots_cost 18 3 in
  let coffee := total_coffee pots 12 in
  let donuts := total_donuts 3 12 in
  ounces_per_donut coffee donuts = 2 :=
by
  intros
  sorry

end coffee_per_donut_l183_183481


namespace double_layer_cake_cost_l183_183278

theorem double_layer_cake_cost :
  ∃ x : ℝ, 28 + 5 * x = 63 ∧ x = 7 :=
begin
  use 7,
  split,
  {
    norm_num,
  },
  {
    refl,
  }
end

end double_layer_cake_cost_l183_183278


namespace find_intercept_l183_183038

-- Define the data points
def data_points : List (ℝ × ℝ) := [(0, 1), (1, 2), (2, 4), (3, 5)]

-- Define the regression equation slope
def slope : ℝ := 1.4

-- Calculate the average of the x values
def avg_x : ℝ := (0 + 1 + 2 + 3) / 4

-- Calculate the average of the y values
def avg_y : ℝ := (1 + 2 + 4 + 5) / 4

-- Formulate the theorem to find the intercept a
theorem find_intercept : ∃ (a : ℝ), avg_y = slope * avg_x + a :=
by
  sorry

end find_intercept_l183_183038


namespace ways_to_write_4020_l183_183854

theorem ways_to_write_4020 : 
  let M := {n : ℕ | ∃ (b3 b2 b1 b0 : ℕ), (4020 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0) ∧ (0 ≤ b3 ∧ b3 ≤ 99) ∧ (0 ≤ b2 ∧ b2 ≤ 99) ∧ (0 ≤ b1 ∧ b1 ≤ 99) ∧ (0 ≤ b0 ∧ b0 ≤ 99)} in
  M.card = 40000 :=
by sorry

end ways_to_write_4020_l183_183854


namespace probability_of_sum_le_two_thirds_l183_183142

theorem probability_of_sum_le_two_thirds
  (x y : ℝ)
  (hx : 0 ≤ x ∧ x ≤ 1)
  (hy : 0 ≤ y ∧ y ≤ 1) :
  (1:ℝ) / (1:ℝ) = (((∫ x in 0..1, ∫ y in 0..1, ite (x + y <= 2/3) 1 0) : ℝ) / (∫ x in 0..1, ∫ y in 0..1, 1 : ℝ)) := 
by {
  --continued proof goes here
  sorry 
}

end probability_of_sum_le_two_thirds_l183_183142


namespace rectangular_coords_of_curve_C_alpha_value_given_midpoint_l183_183464

/-- Problem 1: Convert polar coordinate equation to rectangular coordinate equation -/

theorem rectangular_coords_of_curve_C :
  ∀ (ρ θ : ℝ), ρ = 4 * cos θ / (1 - (cos θ)^2) →
  ∃ x y : ℝ, y^2 = 4*x ∧ x = ρ * cos θ ∧ y = ρ * sin θ :=
by
  intro ρ θ h
  have h1 : ρ * sin θ * sin θ = 4 * cos θ := by sorry
  have h2 : ρ = 4 * cos θ / (1 - cos θ * cos θ) := by sorry
  use [ρ * cos θ, ρ * sin θ]
  split
  . exact h1
  . split
  . exact (h2 ▸ ρ * cos θ)
  . exact (h2 ▸ ρ * sin θ)
  done

/-- Problem 2: Determine the value of α given the midpoint conditions -/

theorem alpha_value_given_midpoint :
  ∀ (α : ℝ), (∀ t, (2 + t * cos α, 2 + t * sin α) ∈ {(x, y) | y^2 = 4*x}) →
  (∃ (a b : ℝ), (a + b = 0) ∧ (2 + a * cos α = 2 + b * cos α) ∧ (2 + a * sin α = 2 + b * sin α)) →
  0 ≤ α ∧ α < π →
  α = π / 4 :=
by
  intro α h_intersects h_midpoint h_alpha_range
  have h_eqns : ∀ t, (2 + t * sin α)^2 = 4 * (2 + t * cos α) := by sorry
  have h_sum : ∀ t1 t2, t1 + t2 = 0 := by sorry
  have h_mid : (4 * sin α - 4 * cos α = 0) := by sorry
  have h_tan : tan α = 1 := by sorry
  have α_range : 0 ≤ α ∧ α < π := h_alpha_range
  exact (by calc α = acos (cos α) : by sorry)

end rectangular_coords_of_curve_C_alpha_value_given_midpoint_l183_183464


namespace hyperbola_eccentricity_l183_183293

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (c : ℝ) (A B F : ℝ × ℝ)
variable (hF : F = (c, 0))
variable (hB : B = (0, b))
variable (hFA_2AB : ∃ A, A = (c / 3, 2 * b / 3) ∧ ∥A - F∥ = 2 * ∥B - A∥)
variable (asymptote : A.y = (b / a) * A.x)

theorem hyperbola_eccentricity {e : ℝ} : 
  ∃ (e : ℝ), e = 2 
:= sorry

end hyperbola_eccentricity_l183_183293


namespace a1_a2_a3_sum_l183_183806

-- Given conditions and hypothesis
variables (a0 a1 a2 a3 : ℝ)
axiom H : ∀ x : ℝ, 1 + x + x^2 + x^3 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3

-- Goal statement to be proven
theorem a1_a2_a3_sum : a1 + a2 + a3 = -3 :=
sorry

end a1_a2_a3_sum_l183_183806


namespace smallest_area_right_triangle_l183_183199

theorem smallest_area_right_triangle (a b : ℕ) (h1 : a = 6) (h2 : b = 8)
  (c : ℝ) (h3 : c^2 = a^2 + b^2) :
  let A_leg := 1/2 * a * b
  let B_leg := 1/2 * a * (real.sqrt (b^2 - a^2))
  min A_leg B_leg = 6 * real.sqrt 7 :=
by {
  -- Proof goes here
  sorry
}

end smallest_area_right_triangle_l183_183199


namespace logarithm_decreasing_l183_183533

theorem logarithm_decreasing (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) : 
  ∀ x y : ℝ, 0 < x → x < y → log a x > log a y :=
by
  sorry

end logarithm_decreasing_l183_183533


namespace at_least_one_negative_l183_183513

-- Defining the circle partition and the properties given in the problem.
def circle_partition (a : Fin 7 → ℤ) : Prop :=
  ∃ (l1 l2 l3 : Finset (Fin 7)),
    l1.card = 4 ∧ l2.card = 4 ∧ l3.card = 4 ∧
    (∀ i ∈ l1, ∀ j ∉ l1, a i + a j = 0) ∧
    (∀ i ∈ l2, ∀ j ∉ l2, a i + a j = 0) ∧
    (∀ i ∈ l3, ∀ j ∉ l3, a i + a j = 0) ∧
    ∃ i, a i = 0

-- The main theorem to prove.
theorem at_least_one_negative : 
  ∀ (a : Fin 7 → ℤ), 
  circle_partition a → 
  ∃ i, a i < 0 :=
by
  sorry

end at_least_one_negative_l183_183513


namespace basic_astrophysics_degrees_l183_183596

theorem basic_astrophysics_degrees 
  (p_microphotonics p_home_electronics p_food_additives p_genetically_modified_microorganisms p_industrial_lubricants : ℤ)
  (h1 : p_microphotonics = 14)
  (h2 : p_home_electronics = 24)
  (h3 : p_food_additives = 15)
  (h4 : p_genetically_modified_microorganisms = 19)
  (h5 : p_industrial_lubricants = 8) :
  let total_percentage := p_microphotonics + p_home_electronics + p_food_additives + p_genetically_modified_microorganisms + p_industrial_lubricants
  in 360 * (100 - total_percentage) / 100 = 72 := 
by 
  sorry

end basic_astrophysics_degrees_l183_183596


namespace proof_sin_2α_proof_β_l183_183759

variable (α β : ℝ)

axiom tan_α : 0 < β ∧ β < α ∧ α < (Real.pi / 2) ∧ Real.tan α = 4 * Real.sqrt 3
axiom cos_diff : Real.cos (α - β) = 13 / 14

noncomputable def sin_2α : ℝ :=
  2 * (4 * Real.sqrt 3 / 7) * (1 / 7)

theorem proof_sin_2α : sin_2α α = 8 * Real.sqrt 3 / 49 :=
  sorry

theorem proof_β : β = Real.pi / 3 :=
  sorry

end proof_sin_2α_proof_β_l183_183759


namespace probability_of_distance_lt_8000_l183_183528

def distance_pairs : List (String × String × Nat) := [
  ("Bangkok", "Cape Town", 6300),
  ("Bangkok", "Honolulu", 6609),
  ("Bangkok", "London", 5944),
  ("Cape Town", "Honolulu", 11535),
  ("Cape Town", "London", 5989),
  ("Honolulu", "London", 7240)
]

def count_lt_8000 (pairs : List (String × String × Nat)) : Nat :=
  pairs.countp (λ p => p.2.2 < 8000)

def probability_less_8000 : ℚ :=
  count_lt_8000 distance_pairs / distance_pairs.length

theorem probability_of_distance_lt_8000 :
  probability_less_8000 = 5 / 6 := by
  sorry

end probability_of_distance_lt_8000_l183_183528


namespace least_integral_qr_length_l183_183226

theorem least_integral_qr_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7)
    (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
    ∃ (QR : ℕ), QR = 15 ∧ ∀ (QR' : ℝ), QR' > real.max (PR - PQ) (QS - SR) → QR' ≥ 15 := 
begin
    sorry -- proving the statement
end

end least_integral_qr_length_l183_183226


namespace area_of_triangle_DEF_is_16_l183_183462

noncomputable def area_of_triangle_DEF : ℝ :=
  let side_pqrs := Math.sqrt 64 -- side length of PQRS
  let midpoint_pq := side_pqrs / 2
  let small_square_side := 2
  let remaining_length := side_pqrs - 2 * small_square_side
  let base_ef := side_pqrs - 2 * small_square_side
  let height_dt := remaining_length / 2 in
  0.5 * base_ef * height_dt

theorem area_of_triangle_DEF_is_16 :
  area_of_triangle_DEF = 16 := by
  sorry

end area_of_triangle_DEF_is_16_l183_183462


namespace angle_between_vectors_l183_183761

noncomputable def magnitude_angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  let angle := arccos ((a - b) • (a + b) / (‖a‖ * ‖b‖))
  if abs (‖a‖ - ‖b‖) = 0 ∧ abs (‖a - b‖ - sqrt 3 * ‖a + b‖) = 0 then pi * (2 / 3)
  else angle

theorem angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) (h₁ : ‖a‖ = ‖b‖) (h₂ : ‖a - b‖ = sqrt 3 * ‖a + b‖) :
  magnitude_angle_between_vectors a b = 2 * pi / 3 := by
  sorry

end angle_between_vectors_l183_183761


namespace find_C_l183_183867

theorem find_C (A B C D E : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) (h5 : E < 10) 
  (h : 4 * (10 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) + 4) = 400000 + (10000 * A + 1000 * B + 100 * C + 10 * D + E)) : 
  C = 2 :=
sorry

end find_C_l183_183867


namespace simplify_expression_l183_183649

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183649


namespace harmonic_sum_induction_l183_183405

theorem harmonic_sum_induction (n : ℕ) : 
  (∑ i in finset.range (2^n), (1 / (i + 1) : ℝ)) > n / 2 :=
sorry

end harmonic_sum_induction_l183_183405


namespace problem_l183_183750

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b : V} {m n : ℝ}
variables {OA : V} (G : V → V → V → V) (M : V → V → V) (P Q O : V)
noncomputable def centroid (v1 v2 v3 : V) : V := (v1 + v2 + v3) / 3
noncomputable def midpoint (v1 v2 : V) : V := (v1 + v2) / 2

-- G is the centroid of triangle ABO
def G_is_centroid : Prop := G a b O = centroid a b O

-- M is the midpoint of side AB
def M_is_midpoint : Prop := M a b = midpoint a b

-- Given vector relations
variables (OP : V) (OQ : V)
def OP_belongs : Prop := OP = m • a
def OQ_belongs : Prop := OQ = n • b

-- Line PQ passes through the centroid G
def collinearPGQ : Prop := ∃ λ : ℝ, (G P Q - OP = λ • (OQ - G P Q))

theorem problem (hG : G_is_centroid) (hM : M_is_midpoint) (h1 : OP_belongs) (h2 : OQ_belongs) (hcollinear : collinearPGQ) : 
  (G a b O - a + G a b O - b + G a b O - O = 0) ∧ 
  (1 / m + 1 / n = 3) :=
by {
  sorry
}

end problem_l183_183750


namespace position_of_permutation_l183_183137

theorem position_of_permutation :
  ∀ (num1 num2: ℕ), num1 = 11234567 ∧ num2 = 46753211 →
  let permutations := (multiset.permutations (list.of_digits (num1.digits))).to_list in
  let sorted_permutations := list.sort (λ a b, nat.of_digits a < nat.of_digits b) permutations in
  list.index_of (num2.digits) sorted_permutations + 1 = 12240 :=
sorry

end position_of_permutation_l183_183137


namespace exists_zero_in_interval_l183_183782

variable {x t : ℝ} {m : ℝ}
def f (x : ℝ) (m : ℝ) := x^2 + x + m

theorem exists_zero_in_interval (m_pos : 0 < m) (f_t_neg : f t m < 0) : 
  ∃ c ∈ set.Ioo t (t + 1), f c m = 0 := by
sorry

end exists_zero_in_interval_l183_183782


namespace min_area_right_triangle_l183_183204

/-- Minimum area of a right triangle with sides 6 and 8 units long. -/
theorem min_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8)
: 15.87 ≤ min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2))) :=
by
  sorry

end min_area_right_triangle_l183_183204


namespace find_ratio_of_a_and_b_l183_183737

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

theorem find_ratio_of_a_and_b (a b : ℝ) 
  (h1 : ∀ x, f x a b ≤ 10) 
  (h2 : f 1 a b = 10) 
  (h3 : ∀ x, f' x a b = 0 → x = 1) :
  a / b = -2 / 3 :=
sorry

end find_ratio_of_a_and_b_l183_183737


namespace circle_circumference_l183_183240

-- Define the radius of the circle
def radius : ℝ := 11

-- Define the value of π (pi)
def pi_approx : ℝ := 3.14159

-- Define the formula for the circumference
def circumference (r : ℝ) : ℝ := 2 * pi_approx * r

-- State the theorem to prove
theorem circle_circumference : circumference radius = 69.11 :=
by
  -- Using the given radius and value of pi, calculate the circumference
  have h1 : circumference radius = 2 * pi_approx * radius := rfl
  -- Plug in the values
  have h2 : circumference radius = 2 * 3.14159 * 11 := by rw [h1, radius, pi_approx]
  -- Perform the calculation
  have h3 : 2 * 3.14159 * 11 = 69.11498 := by norm_num
  -- Round to two decimal places
  have h4 : (69.11498 : ℝ).round = 69.11 := by linarith
  -- Conclude the proof
  exact h4

end circle_circumference_l183_183240


namespace andrea_sod_rectangles_l183_183275

def section_1_length : ℕ := 35
def section_1_width : ℕ := 42
def section_2_length : ℕ := 55
def section_2_width : ℕ := 86
def section_3_length : ℕ := 20
def section_3_width : ℕ := 50
def section_4_length : ℕ := 48
def section_4_width : ℕ := 66

def sod_length : ℕ := 3
def sod_width : ℕ := 4

def area (length width : ℕ) : ℕ := length * width
def sod_area : ℕ := area sod_length sod_width

def rectangles_needed (section_length section_width sod_area : ℕ) : ℕ :=
  (area section_length section_width + sod_area - 1) / sod_area

def total_rectangles_needed : ℕ :=
  rectangles_needed section_1_length section_1_width sod_area +
  rectangles_needed section_2_length section_2_width sod_area +
  rectangles_needed section_3_length section_3_width sod_area +
  rectangles_needed section_4_length section_4_width sod_area

theorem andrea_sod_rectangles : total_rectangles_needed = 866 := by
  sorry

end andrea_sod_rectangles_l183_183275


namespace ratio_volume_surface_area_l183_183613

-- Define the structure formed by 8 unit cubes.
structure EightUnitCubes where
  volume : ℕ
  surfaceArea : ℕ

-- Specify the parameters given in the problem.
def structureParams : EightUnitCubes :=
  { volume := 8, surfaceArea := 24 }

-- Theorem stating the problem's request.
theorem ratio_volume_surface_area (s : EightUnitCubes) : s.volume / s.surfaceArea = 1 / 3 :=
by
  -- Define the values explicitly
  let v := 8
  let sa := 24
  -- The ratio v / sa should be 1 / 3
  show v / sa = 1 / 3
  sorry

end ratio_volume_surface_area_l183_183613


namespace proof_problem_l183_183061

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end proof_problem_l183_183061


namespace volume_of_original_cone_is_correct_l183_183598

noncomputable def volume_of_original_cone : ℝ :=
  sorry

theorem volume_of_original_cone_is_correct :
  (∃ (V_cyl : ℝ) (V_frust : ℝ), V_cyl = 21 ∧ V_frust = 91) → volume_of_original_cone = 94.5 :=
begin
  sorry
end

end volume_of_original_cone_is_correct_l183_183598


namespace sum_2017_l183_183837

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = (-1) ^ n * (a n + 1)

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  S 0 = 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem sum_2017 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_seq : sequence a) (h_sum : sum_first_n_terms a S) :
  S 2017 = -1007 := 
by
  sorry

end sum_2017_l183_183837


namespace area_of_triangle_ABC_l183_183054

-- Defines the function f(x)
noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6) + 1 / 2

-- Given conditions for the triangle
def a := 1
def b_and_c := 2
def f_A := 1
def A := Real.pi / 3

-- Main theorem to prove
theorem area_of_triangle_ABC :
  ∃ (b c : ℝ), f A = 1 ∧ b + c = 2 ∧ 
  (1 = b*c * (Real.sin A)) ∧
  (1 / 2 * b * c * (Real.sin A) = sqrt 3 / 4) := 
sorry

end area_of_triangle_ABC_l183_183054


namespace total_cans_to_collect_l183_183126

def cans_for_project (marthas_cans : ℕ) (additional_cans_needed : ℕ) (total_cans_needed : ℕ) : Prop :=
  ∃ diegos_cans : ℕ, diegos_cans = (marthas_cans / 2) + 10 ∧ 
  total_cans_needed = marthas_cans + diegos_cans + additional_cans_needed

theorem total_cans_to_collect : 
  cans_for_project 90 5 150 :=
by
  -- Insert proof here in actual usage
  sorry

end total_cans_to_collect_l183_183126


namespace modulus_of_z_l183_183378

theorem modulus_of_z (z : ℂ) (h : (z - 2) * complex.I = 1 + complex.I) : complex.abs z = real.sqrt 10 := 
sorry

end modulus_of_z_l183_183378


namespace exists_lambda_and_poly_poly_condition_for_lambda_2_l183_183701

variable {k : ℕ}

-- Main statement for the first part of the problem
theorem exists_lambda_and_poly (λ : ℝ) :
  (∀ (P : ℕ → ℝ) (n : ℕ), (P 1 + P 3 + P 5 + ⋯ + P (2*n-1)) / n = λ * P n) →
  λ = 2^k / (k + 1) :=
sorry

-- Main statement for the second part of the problem
noncomputable def poly_for_lambda_2 (a : ℝ) : ℝ[X] :=
  Polynomial.C a * Polynomial.X * (Polynomial.X^2 - 1)

theorem poly_condition_for_lambda_2 (P : ℝ[X]) :
  (∀ (n : ℕ), (λ P : ℕ → ℝ, polynomial.eval P (2*n-1)) = 2 * P.eval n) →
  P = poly_for_lambda_2 :=
sorry

end exists_lambda_and_poly_poly_condition_for_lambda_2_l183_183701


namespace total_birds_remaining_l183_183549

theorem total_birds_remaining (grey_birds_in_cage : ℕ) (white_birds_next_to_cage : ℕ) :
  (grey_birds_in_cage = 40) →
  (white_birds_next_to_cage = grey_birds_in_cage + 6) →
  (1/2 * grey_birds_in_cage = 20) →
  (1/2 * grey_birds_in_cage + white_birds_next_to_cage = 66) :=
by 
  intros h_grey_birds h_white_birds h_grey_birds_freed
  sorry

end total_birds_remaining_l183_183549


namespace dot_product_ab_magnitude_a_plus_b_perpendicular_vectors_l183_183053

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
axiom norm_a : ∥a∥ = 4
axiom norm_b : ∥b∥ = 8
axiom angle_ab : real.angle.cos (real.angle.between a b) = cos (2 * real.pi / 3)

-- Mathematically equivalent proof problem

-- (1) Prove that the dot product a • b = -16
theorem dot_product_ab : ⟪a, b⟫ = -16 :=
sorry

-- (2) Prove that the magnitude of a + b = 4√3
theorem magnitude_a_plus_b : ∥ a + b ∥ = 4 * real.sqrt 3 :=
sorry

-- (3) Prove that when k = -7, (a + 2b) is perpendicular to (k a - b)
theorem perpendicular_vectors (k : ℝ) (h : k = -7) : ⟪a + 2 • b, k • a - b⟫ = 0 :=
sorry

end dot_product_ab_magnitude_a_plus_b_perpendicular_vectors_l183_183053


namespace exists_solution_in_interval_l183_183685

noncomputable def f (x : ℝ) : ℝ := x^3 - 2^x

theorem exists_solution_in_interval : ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f x = 0 :=
by {
  -- Use the Intermediate Value Theorem, given f is continuous on [1, 2]
  sorry
}

end exists_solution_in_interval_l183_183685


namespace seats_not_occupied_l183_183327

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_l183_183327


namespace fifth_derivative_correct_l183_183339

noncomputable def fifth_derivative (y : ℝ → ℝ) (f : ℝ → ℝ) :=
  deriv^[5] y = f

theorem fifth_derivative_correct :
  fifth_derivative (λ x, (4 * x + 3) * 2^(-x)) 
  (λ x, (- (Real.log 2)^5 * (4 * x + 3) + 20 * (Real.log 2)^4) * 2^(-x)) :=
by
  sorry

end fifth_derivative_correct_l183_183339


namespace mixed_groups_count_l183_183969

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183969


namespace mixed_groups_count_l183_183985

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183985


namespace final_total_cost_l183_183611

theorem final_total_cost (spiral_notebook_cost : ℕ) (planner_cost : ℕ) (spiral_notebook_discount_threshold : ℕ) 
  (spiral_notebook_discount_rate : ℚ) (planner_discount_threshold : ℕ) (planner_discount_rate : ℚ)
  (tax_rate : ℚ) (num_spiral_notebooks : ℕ) (num_planners : ℕ) :
  spiral_notebook_cost = 15 ∧ planner_cost = 10 ∧ spiral_notebook_discount_threshold = 5 ∧ spiral_notebook_discount_rate = 0.20 ∧ 
  planner_discount_threshold = 10 ∧ planner_discount_rate = 0.15 ∧ tax_rate = 0.07 ∧ 
  num_spiral_notebooks = 6 ∧ num_planners = 12 →
  let total_spiral_notebook_cost := if num_spiral_notebooks >= spiral_notebook_discount_threshold 
                                    then num_spiral_notebooks * spiral_notebook_cost * (1 - spiral_notebook_discount_rate)
                                    else num_spiral_notebooks * spiral_notebook_cost in
  let total_planner_cost := if num_planners >= planner_discount_threshold 
                            then num_planners * planner_cost * (1 - planner_discount_rate)
                            else num_planners * planner_cost in
  let total_before_tax := total_spiral_notebook_cost + total_planner_cost in
  let total_cost_with_tax := total_before_tax * (1 + tax_rate) in
  total_cost_with_tax = 186.18 :=
by {
  sorry
}

end final_total_cost_l183_183611


namespace distance_from_point_to_asymptote_l183_183936

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l183_183936


namespace statement_B_correct_statement_D_correct_l183_183107

-- Definitions based on the conditions
def S (n : ℕ) (a1 d : ℝ) : ℝ :=
  n * a1 + (n * (n - 1)) / 2 * d

-- Statement B: If the sequence {S_n} has a minimum term, then d > 0
theorem statement_B_correct (a1 d : ℝ) (h : d ≠ 0) :
  (∃ n : ℕ, ∀ m : ℕ, S n a1 d ≤ S m a1 d) → d > 0 := by
  sorry

-- Statement D: If for any n ∈ ℕ*, S_n > 0, then the sequence {S_n} is increasing
theorem statement_D_correct (a1 d : ℝ) (h : d ≠ 0) :
  (∀ n : ℕ, 0 < n → S n a1 d > 0) → (∀ n : ℕ, S n a1 d < S (n + 1) a1 d) := by
  sorry

end statement_B_correct_statement_D_correct_l183_183107


namespace quadratic_coefficients_l183_183834

theorem quadratic_coefficients (x : ℝ) : 
  (2 * x^2 - 1 = 6 * x) → 
  (∃ a b c : ℚ, a = 2 ∧ b = -6 ∧ c = -1 ∧ a * x^2 + b * x + c = 0) := 
by intros h;
  use [2, -6, -1];
  split;
  { refl };
  split;
  { refl };
  split;
  { refl };
  simp [h];
  sorry

end quadratic_coefficients_l183_183834


namespace log_equation_solution_l183_183433

theorem log_equation_solution (x : ℝ) (h : log 9 x = 2.4) : x = (Real.root 81 5) ^ 6 := by
  sorry

end log_equation_solution_l183_183433


namespace solve_m_correct_l183_183403

noncomputable def solve_for_m (Q t h : ℝ) : ℝ :=
  if h >= 0 ∧ Q > 0 ∧ t > 0 then
    (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h))
  else
    0 -- Define default output for invalid inputs

theorem solve_m_correct (Q t h : ℝ) (m : ℝ) :
  Q = t / (1 + Real.sqrt h)^m → m = (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h)) :=
by
  intros h1
  rw [h1]
  sorry

end solve_m_correct_l183_183403


namespace minimum_driving_age_l183_183100

noncomputable def kimiko_age : ℕ := 26
noncomputable def kayla_age : ℕ := kimiko_age / 2
noncomputable def waiting_years : ℕ := 5

theorem minimum_driving_age :
  let minimum_age := kayla_age + waiting_years in
  minimum_age = 18 :=
by
  let minimum_age := kayla_age + waiting_years
  exact Eq.refl 18
sorry

end minimum_driving_age_l183_183100


namespace sin_alpha_add_pi_div_4_l183_183757

theorem sin_alpha_add_pi_div_4 (α : ℝ)
  (h : (sin α - 3 / 5) + (cos α - 4 / 5) * Complex.i = 0 + (cos α - 4 / 5) * Complex.i) :
  sin (α + π / 4) = - (Real.sqrt 2 / 10) :=
by sorry

end sin_alpha_add_pi_div_4_l183_183757


namespace f_diff_l183_183372

def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n)).sum (λ k => (1 : ℚ) / (k + 1))

theorem f_diff (n : ℕ) : f (n + 1) - f n = (1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2)) :=
by
  sorry

end f_diff_l183_183372


namespace distance_from_point_to_asymptote_l183_183934

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l183_183934


namespace complex_mul_l183_183858

theorem complex_mul (i : ℂ) (h : i^2 = -1) :
    (1 - i) * (1 + 2 * i) = 3 + i :=
by
  sorry

end complex_mul_l183_183858


namespace quadrilateral_perimeter_sum_l183_183608

theorem quadrilateral_perimeter_sum :
  let d1 := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let d2 := real.sqrt ((7 - 4)^2 + (3 - 6)^2)
  let d3 := real.sqrt ((5 - 7)^2 + (0 - 3)^2)
  let d4 := real.sqrt ((1 - 5)^2 + (2 - 0)^2)
  let p := 2
  let q := 1
  let r := 3
  d1 + d2 + d3 + d4 = p * real.sqrt 5 + q * real.sqrt 13 + r * real.sqrt 2 →
  (p + q + r) = 6 := by
  sorry

end quadrilateral_perimeter_sum_l183_183608


namespace minimum_area_of_triangle_ABC_l183_183485

noncomputable def A : ℝ × ℝ × ℝ := (-2, 0, 3)
noncomputable def B : ℝ × ℝ × ℝ := (2, 3, 4)
noncomputable def C (s : ℝ) : ℝ × ℝ × ℝ := (s, 4, 2)

def area (A B C : ℝ × ℝ × ℝ) : ℝ :=
  0.5 * Real.sqrt (
    (Real.sqrt (Real.pow (B.1 - A.1) 2 + Real.pow (B.2 - A.2) 2 + Real.pow (B.3 - A.3) 2)) *
    (Real.sqrt (Real.pow (C.1 - A.1) 2 + Real.pow (C.2 - A.2) 2 + Real.pow (C.3 - A.3) 2)) -
    Real.pow (
      Real.dot_product
        ⟨(B.1 - A.1), (B.2 - A.2), (B.3 - A.3)⟩
        ⟨(C.1 - A.1), (C.2 - A.2), (C.3 - A.3)⟩
    ) 2
  )

theorem minimum_area_of_triangle_ABC :
  ∃ s : ℝ, area A B (C s) = (Real.sqrt (51.5)) / 2 :=
by
  use 3.5  -- from the solution steps
  sorry

end minimum_area_of_triangle_ABC_l183_183485


namespace a_2017_value_l183_183018

variable (a S : ℕ → ℤ)

-- Given conditions
axiom a1 : a 1 = 1
axiom a_nonzero : ∀ n, a n ≠ 0
axiom a_S_relation : ∀ n, a n * a (n + 1) = 2 * S n - 1

theorem a_2017_value : a 2017 = 2017 := by
  sorry

end a_2017_value_l183_183018


namespace pine_tree_neighbors_l183_183277

theorem pine_tree_neighbors 
  (trees : Finset ℕ)
  (h_total : trees.card = 2019)
  (h_pines : ∃ pines : Finset ℕ, pines.card = 1009 ∧ pines ⊆ trees)
  (h_firs : ∃ firs : Finset ℕ, firs.card = 1010 ∧ firs ⊆ trees) :
  ∃ t ∈ trees, ∃ t2 t3 ∈ trees, t2 ≠ t ∧ t3 ≠ t ∧ t2 ∈ h_pines ∧ t3 ∈ h_pines ∧ t2 = t + 1 ∧ t3 = t + 2 % 2019 :=
sorry

end pine_tree_neighbors_l183_183277


namespace equal_angles_l183_183220

-- Define the elements of the problem
variables (A B C D P Q M : Point)
variables (l : Line)
variables [is_parallelogram A B C D]
variables [perpendicular l (line_through B C)]
variables [tangent_to_circle l P]
variables [tangent_to_circle l Q]
variables [passes_through D C P]
variables [passes_through D C Q]
variables [midpoint M A B]

-- State the theorem
theorem equal_angles (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : D ≠ A)
: angle D M P = angle D M Q :=
by sorry

end equal_angles_l183_183220


namespace part1_part2_l183_183042

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2 * x - 2) / (x ^ 2 + 1)

-- Define the function g(x)
def g (x : ℝ) : ℝ := Real.log x

-- Prove that f(-1) = -2
theorem part1 (x : ℝ) : f (-1) = -2 := by
  sorry

-- Prove that g(x) ≥ f(x) for x ∈ [1, +∞)
theorem part2 (x : ℝ) (hx : 1 ≤ x) : g x ≥ f x := by
  sorry

end part1_part2_l183_183042


namespace union_A_B_intersection_A_complement_B_l183_183791

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2
def setB (x : ℝ) : Prop := x * (x - 4) ≤ 0

theorem union_A_B : {x : ℝ | setA x} ∪ {x : ℝ | setB x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_complement_B : {x : ℝ | setA x} ∩ {x : ℝ | ¬ setB x} = {x : ℝ | -1 ≤ x ∧ x < 0} :=
by
  sorry

end union_A_B_intersection_A_complement_B_l183_183791


namespace unique_pair_prime_m_positive_l183_183104

theorem unique_pair_prime_m_positive (p m : ℕ) (hp : Nat.Prime p) (hm : 0 < m) :
  p * (p + m) + p = (m + 1) ^ 3 → (p = 2 ∧ m = 1) :=
by
  sorry

end unique_pair_prime_m_positive_l183_183104


namespace mixed_groups_count_l183_183987

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183987


namespace hyperbola_integer_points_l183_183425

theorem hyperbola_integer_points :
  (∃ n : ℕ, prime n ∧ 2013 = n ^ 3) ∧
  ∀ x y : ℤ, y = 2013 / x → x * y = 2013 → 16 :=
sorry

end hyperbola_integer_points_l183_183425


namespace red_car_initial_distance_ahead_l183_183192

theorem red_car_initial_distance_ahead 
    (Speed_red Speed_black : ℕ) (Time : ℝ)
    (H1 : Speed_red = 10)
    (H2 : Speed_black = 50)
    (H3 : Time = 0.5) :
    let Distance_black := Speed_black * Time
    let Distance_red := Speed_red * Time
    Distance_black - Distance_red = 20 := 
by
  let Distance_black := Speed_black * Time
  let Distance_red := Speed_red * Time
  sorry

end red_car_initial_distance_ahead_l183_183192


namespace first_system_solution_second_system_solution_l183_183148

theorem first_system_solution (x y : ℝ) (h₁ : 3 * x - y = 8) (h₂ : 3 * x - 5 * y = -20) : 
  x = 5 ∧ y = 7 := 
by
  sorry

theorem second_system_solution (x y : ℝ) (h₁ : x / 3 - y / 2 = -1) (h₂ : 3 * x - 2 * y = 1) : 
  x = 3 ∧ y = 4 := 
by
  sorry

end first_system_solution_second_system_solution_l183_183148


namespace polynomial_divisibility_l183_183116

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry
noncomputable def R : Polynomial ℝ := sorry
noncomputable def S : Polynomial ℝ := sorry

theorem polynomial_divisibility
  (h : P (x^3) + x * Q (x^5) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S (x)) :
  x - 1 ∣ P (x) :=
  sorry

end polynomial_divisibility_l183_183116


namespace hidden_lattice_points_l183_183905

theorem hidden_lattice_points (ell : ℤ) : ∃ (x y : ℤ), 
  ∀ (i j : ℤ), 
  0 ≤ i ∧ i < ell → 0 ≤ j ∧ j < ell →
  (∃ (pi : ℕ), Nat.Prime pi ∧ (x + i ≡ 0 [MOD pi] ∧ y + j ≡ 0 [MOD pi])) := sorry

end hidden_lattice_points_l183_183905


namespace average_of_multiplied_numbers_l183_183582

theorem average_of_multiplied_numbers (numbers : List ℝ) (avg : ℝ) (h_avg : avg = 21)
  (h_len : numbers.length = 8) (h_sum : numbers.sum = 8 * avg) :
  let new_numbers := numbers.map (λ x => x * 8)
  (new_avg : ℝ) := new_avg = new_numbers.sum / new_numbers.length
  in new_avg = 168 := 
by
  sorry

end average_of_multiplied_numbers_l183_183582


namespace expression_not_defined_at_12_l183_183362

theorem expression_not_defined_at_12 : 
  ¬ ∃ x, x^2 - 24 * x + 144 = 0 ∧ (3 * x^3 + 5) / (x^2 - 24 * x + 144) = 0 :=
by
  intro h
  cases h with x hx
  have hx2 : x^2 - 24 * x + 144 = 0 := hx.1
  have denom_zero : x^2 - 24 * x + 144 = 0 := by sorry
  subst denom_zero
  sorry

end expression_not_defined_at_12_l183_183362


namespace shared_side_triangle_inequality_l183_183224

theorem shared_side_triangle_inequality 
  (P Q R S : Type) 
  [AddGroup P] [AddGroup Q] [AddGroup R] [AddGroup S]
  (PQ PR SR QS QR : ℝ)
  (hPQ : PQ = 7) 
  (hPR : PR = 15) 
  (hSR : SR = 10) 
  (hQS : QS = 25) 
  (hTriangleInequalityPQR : QR > PR - PQ) 
  (hTriangleInequalitySQR : QR > QS - SR) : 
  QR ≥ 15 := 
by {
  intros,
  have h1 : QR > 15 - 7 := by exact hTriangleInequalityPQR,
  have h2 : QR > 25 - 10 := by exact hTriangleInequalitySQR,
  have minQR := max (PR - PQ) (QS - SR),
  linarith,
  sorry
}

end shared_side_triangle_inequality_l183_183224


namespace avg_price_pen_is_correct_l183_183591

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end avg_price_pen_is_correct_l183_183591


namespace transformed_function_symmetry_l183_183160

noncomputable def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = f x

theorem transformed_function_symmetry :
  let f := λ x : ℝ, Real.sin (6 * x + Real.pi / 4)
  let g := λ x : ℝ, Real.sin (2 * x)
  ∃ p : ℝ × ℝ, symmetry_center g p ∧ p = (Real.pi / 2, 0) := 
by
  sorry

end transformed_function_symmetry_l183_183160


namespace carrots_as_potatoes_l183_183504

variable (G O C P : ℕ)

theorem carrots_as_potatoes :
  G = 8 →
  G = (1 / 3 : ℚ) * O →
  O = 2 * C →
  P = 2 →
  (C / P : ℚ) = 6 :=
by intros hG1 hG2 hO hP; sorry

end carrots_as_potatoes_l183_183504


namespace no_natural_has_2021_trailing_zeros_l183_183315

-- Define the function f(n) which computes the number of trailing zeros in n!
def trailing_zeros (n : ℕ) : ℕ :=
  let rec aux (k : ℕ) (acc : ℕ) : ℕ :=
    if k > n then acc
    else aux (k * 5) (acc + n / k)
  aux 5 0

-- Prove that there does not exist a natural number n such that the number of trailing zeros in n! is exactly 2021
theorem no_natural_has_2021_trailing_zeros :
  ¬ ∃ n : ℕ, trailing_zeros n = 2021 :=
by {
  intro h,
  sorry
}

end no_natural_has_2021_trailing_zeros_l183_183315


namespace evaluate_expression_l183_183331

theorem evaluate_expression :
  let a := 2023 in
  ∃ p q : ℕ, (p % q = 2 * a + 1 ∧ p.gcd q = 1 ∧ p = 4047) :=
begin
  sorry
end

end evaluate_expression_l183_183331


namespace area_after_transformation_l183_183106

open Matrix

theorem area_after_transformation {R : Type*} [has_measure R] (area_R : volume R = 15) :
  let M := (33 : Matrix (Fin 2) (Fin 2) ℝ) := ![![3, 4], ![8, -2]]
  abs (det M) * 15 = 570 :=
by
  sorry

end area_after_transformation_l183_183106


namespace sum_of_two_lowest_scores_dropped_l183_183473

theorem sum_of_two_lowest_scores_dropped (S S' : ℕ) (test_scores : Fin 6 → ℕ) 
  (h1 : (∑ i, test_scores i) = 6 * 90) 
  (dropped_scores_sum : S' = 4 * 85) 
  (S := ∑ i, test_scores i) : 
  S - S' = 200 :=
sorry

end sum_of_two_lowest_scores_dropped_l183_183473


namespace mixed_groups_count_l183_183978

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183978


namespace least_positive_angle_l183_183713

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end least_positive_angle_l183_183713


namespace number_of_valid_n_l183_183003

theorem number_of_valid_n :
  {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 35 n = 7}.to_finset.card = 23 :=
by
  sorry

end number_of_valid_n_l183_183003


namespace expression_not_defined_at_12_l183_183363

theorem expression_not_defined_at_12 : 
  ¬ ∃ x, x^2 - 24 * x + 144 = 0 ∧ (3 * x^3 + 5) / (x^2 - 24 * x + 144) = 0 :=
by
  intro h
  cases h with x hx
  have hx2 : x^2 - 24 * x + 144 = 0 := hx.1
  have denom_zero : x^2 - 24 * x + 144 = 0 := by sorry
  subst denom_zero
  sorry

end expression_not_defined_at_12_l183_183363


namespace initial_amount_l183_183236

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end initial_amount_l183_183236


namespace smallest_area_right_triangle_l183_183197

theorem smallest_area_right_triangle (a b : ℕ) (h1 : a = 6) (h2 : b = 8)
  (c : ℝ) (h3 : c^2 = a^2 + b^2) :
  let A_leg := 1/2 * a * b
  let B_leg := 1/2 * a * (real.sqrt (b^2 - a^2))
  min A_leg B_leg = 6 * real.sqrt 7 :=
by {
  -- Proof goes here
  sorry
}

end smallest_area_right_triangle_l183_183197


namespace sum_abs_b_i_l183_183728

noncomputable def R (x : ℝ) : ℝ := 1 - (1 / 4) * x + (1 / 8) * x^2

noncomputable def S (x : ℝ) : ℝ :=
  (R x) * (R (x^2)) * (R (x^4)) * (R (x^6)) * (R (x^8))

theorem sum_abs_b_i : ∑ i in (Finset.range 41), |(S 1) - (S (-1))| = 100000 / 32768 :=
by
  sorry

end sum_abs_b_i_l183_183728


namespace vector_combination_l183_183073

/-- In triangle ABC, points M and N satisfy \overline{AM} = 2\overline{MC} and \overline{BN} = \overline{NC}.
    We need to prove that \overline{MN} = \frac{1}{2}\overline{AB} - \frac{1}{6}\overline{AC} --/
theorem vector_combination 
  (A B C M N : Point)
  (h1 : segment_ratio A M M C 2)
  (h2 : segment_ratio B N N C 1) :
  vector (M, N) = 1 / 2 * vector (A, B) - 1 / 6 * vector (A, C) :=
sorry

end vector_combination_l183_183073


namespace parallelogram_point_P_l183_183866

variable {A B C D P : Type} [EuclideanGeometry A B C D P]

theorem parallelogram_point_P (P : Point) (A B C D : Point) 
  (h_parallelogram: is_parallelogram A B C D) 
  (h_angle: angle A P B + angle C P D = 180) :
  (dist A B) * (dist A D) = (dist B P) * (dist D P) + (dist A P) * (dist C P) := 
sorry

end parallelogram_point_P_l183_183866


namespace tan_triple_angle_l183_183438

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l183_183438


namespace max_abs_ax_plus_b_l183_183048

theorem max_abs_ax_plus_b (a b c : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2 :=
by
  sorry

end max_abs_ax_plus_b_l183_183048


namespace solve_inequality_l183_183523

noncomputable def inequality_solutions (x y : ℝ) : Prop :=
  (∃ n k : ℤ, (x = (π / 4) + π * n) ∧ (y = π * k))

theorem solve_inequality (x y : ℝ) :
  (sqrt 3 * tan x - cbrt (sqrt (sin y)) - sqrt (3 / (cos x ^ 2) + sqrt (sin y) - 6) ≥ sqrt 3) ↔
  inequality_solutions x y := sorry

end solve_inequality_l183_183523


namespace acute_angles_first_quadrant_l183_183210

-- Definitions based on the problem
def is_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def same_terminal_side (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + k * 360

-- Theorem stating that acute angles are always first quadrant angles
theorem acute_angles_first_quadrant (θ : ℝ) :
  is_acute θ → is_first_quadrant θ :=
by
  intro h,
  sorry

end acute_angles_first_quadrant_l183_183210


namespace sin_double_angle_identity_l183_183732

theorem sin_double_angle_identity (α : ℝ) (h1 : cos α = 3 / 5) (h2 : α ∈ Ioo (- (π / 2)) 0) : 
  sin (2 * α) = -24 / 25 := 
sorry

end sin_double_angle_identity_l183_183732


namespace investment_duration_is_two_years_l183_183705

-- Definitions of given conditions
def P : ℝ := 20000
def r : ℝ := 0.04
def n : ℕ := 2
def CI : ℝ := 1648.64
def A : ℝ := P + CI

-- Target statement
theorem investment_duration_is_two_years :
  ∃ t : ℝ, t = 2 ∧ A = P * (1 + r / n)^(n * t) :=
by
  have hA : A = 20000 + 1648.64 := rfl
  have hP : P = 20000 := rfl
  have hr : r = 0.04 := rfl
  have hn : n = 2 := rfl
  have hCI : CI = 1648.64 := rfl
  use 2
  split
  -- to check the final truth via given properties.
  sorry

end investment_duration_is_two_years_l183_183705


namespace complex_operation_l183_183062

theorem complex_operation (z : ℂ) (hz : z = 4 + 3 * complex.i) : 
  (conj z) / complex.norm z = (4/5 : ℝ) - (3/5 : ℝ) * complex.i :=
by 
  sorry

end complex_operation_l183_183062


namespace frog_jumping_sequences_count_l183_183113

/-- 
A frog starts at vertex A of a regular hexagon ABCDEF and can jump to one of the two adjacent vertices at each jump.
The frog stops jumping if it reaches vertex D within 5 jumps or after completing 5 jumps. 
Prove that the total number of different jumping sequences from A until the frog stops is 26.
-/
theorem frog_jumping_sequences_count : 
  ∀ (A B C D E F : Type) (jump : A → B → Prop) (jumps : ℕ) 
    (stop_if_D : D → Prop) (max_jumps : jumps ≤ 5),
    let adjacents := ∀ a : A, jump a B ∨ jump a F,
    jump_sequences A D max_jumps stop_if_D = 26 := 
sorry

end frog_jumping_sequences_count_l183_183113


namespace triangle_A1B1C1_sides_l183_183368

theorem triangle_A1B1C1_sides
  (a b c x y z R : ℝ) 
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_positive_c : c > 0)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_positive_R : R > 0) :
  (↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R), ↑c * ↑x / (2 * ↑R)) = (↑c * ↑x / (2 * ↑R), ↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R)) :=
by sorry

end triangle_A1B1C1_sides_l183_183368


namespace range_of_g_l183_183683

noncomputable def g (x : ℝ) : ℝ := arctan (2 * x) + arctan ((2 - 3 * x) / (2 + 3 * x)) + arctan x

theorem range_of_g : set.Icc (-(π / 4)) (3 * π / 4) = set.range g := sorry

end range_of_g_l183_183683


namespace f_zero_roots_count_l183_183384

noncomputable def f (x : ℝ) : ℝ :=
if h₀ : 0 < x ∧ x < 2/3 then sin (Real.pi * x)
else if x = 3/2 then 0
else sorry -- continued definition based on periodicity and oddness

theorem f_zero_roots_count : 
  (∀ x : ℝ, f (-x) = -f x) →   
  (∀ x : ℝ, f (x + 3) = f x) → 
  (∀ x : ℝ, 0 < x ∧ x < 2/3 → f x = sin (Real.pi * x)) →
  (f (3/2) = 0) →
  ∃ count : ℕ, count = 9 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 6 → (f x = 0) → (count := count + 1)) :=
sorry

end f_zero_roots_count_l183_183384


namespace lane_total_investment_l183_183482

variable (T : ℝ)

def lane_investment (T : ℝ) (interest_total : ℝ) :=
  let invested_at_8_percent := 17000
  let interest_from_8_percent := 0.08 * invested_at_8_percent
  let interest_from_7_percent := 0.07 * (T - invested_at_8_percent)
  interest_from_8_percent + interest_from_7_percent = interest_total

theorem lane_total_investment 
  (interest_total : ℝ) 
  (h : lane_investment T interest_total := 1710) :
  T = 22000 := 
sorry

end lane_total_investment_l183_183482


namespace cylinder_base_area_l183_183241

-- Definitions: Adding variables and hypotheses based on the problem statement.
variable (A_c A_r : ℝ) -- Base areas of the cylinder and the rectangular prism
variable (h1 : 8 * A_c = 6 * A_r) -- Condition from the rise in water levels
variable (h2 : A_c + A_r = 98) -- Sum of the base areas
variable (h3 : A_c / A_r = 3 / 4) -- Ratio of the base areas

-- Statement: The goal is to prove that the base area of the cylinder is 42.
theorem cylinder_base_area : A_c = 42 :=
by
  sorry

end cylinder_base_area_l183_183241


namespace difference_of_squares_650_550_l183_183568

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 :=
by sorry

end difference_of_squares_650_550_l183_183568


namespace sum_of_exponentials_eq_neg_one_l183_183665

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_exponentials_eq_neg_one : 
  (omega + omega^2 + omega^3 + ⋯ + omega^16) = -1 :=
by
  have h1 : omega ^ 17 = 1 := by 
    unfold omega
    -- Proof of omega^17 = 1 would go here
    sorry
  have h2 : omega ^ 16 = omega⁻¹ := by
    rw [← Complex.exp_nat_mul]
    -- Proof of omega^16 = omega⁻¹ would go here
    sorry
  -- Proof that the sum equals -1 would go here
  sorry

end sum_of_exponentials_eq_neg_one_l183_183665


namespace fourth_arithmetic_sequence_equation_l183_183766

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ) (h : is_arithmetic_sequence a)
variable (h1 : a 1 - 2 * a 2 + a 3 = 0)
variable (h2 : a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0)
variable (h3 : a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0)

-- Theorem statement to be proven
theorem fourth_arithmetic_sequence_equation : a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0 :=
by
  sorry

end fourth_arithmetic_sequence_equation_l183_183766


namespace least_integral_qr_length_l183_183225

theorem least_integral_qr_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7)
    (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
    ∃ (QR : ℕ), QR = 15 ∧ ∀ (QR' : ℝ), QR' > real.max (PR - PQ) (QS - SR) → QR' ≥ 15 := 
begin
    sorry -- proving the statement
end

end least_integral_qr_length_l183_183225


namespace tetrahedron_RS_length_l183_183172

noncomputable def tetrahedron_edges := {a b c d e f : ℕ // {a, b, c, d, e, f} = {9, 15, 20, 29, 38, 43}}

theorem tetrahedron_RS_length {PQ : ℕ} {RS : ℕ} (hPQ : PQ = 43)
  (h : tetrahedron_edges) : RS = 15 :=
sorry

end tetrahedron_RS_length_l183_183172


namespace value_of_expression_l183_183498

theorem value_of_expression (x : ℤ) (h : x = -2023) : 
  | ||x| - x - |x| | - x = 4046 := by
  sorry

end value_of_expression_l183_183498


namespace extreme_values_f_inequality_f_g_l183_183747

def f (x : ℝ) := x * Real.log x
def g (x a : ℝ) := -x^2 - a*x - 4

theorem extreme_values_f :
  ∃ (x : ℝ), x > 0 ∧ f x = -1 / Real.exp(1) := 
sorry

theorem inequality_f_g (a : ℝ) :
  (∀ x > 0, f x > (1 / 3) * g x a) → a > -5 :=
sorry

end extreme_values_f_inequality_f_g_l183_183747


namespace isosceles_triangle_l183_183620

-- Definitions for geometric entities
variables {A B C O B' C' P Q : Type*}

-- Assume basic properties of points, segments, triangles, and circumcircle
variable [TriangleGeometry A B C]
variable [Circumcenter O A B C]
variable [Intersection BO AC B']
variable [Intersection CO AB C']
variable [IntersectionLineCircumcircle B'C' (Circle A B C) P Q]
variable (AP_eq_AQ : (Segment A P) = (Segment A Q))

-- Prove that triangle ABC is isosceles
theorem isosceles_triangle (h : AP_eq_AQ) : (Segment A B) = (Segment A C) :=
sorry

end isosceles_triangle_l183_183620


namespace friends_prove_l183_183078

theorem friends_prove (a b c d : ℕ) (h1 : 3^a * 7^b = 3^c * 7^d) (h2 : 3^a * 7^b = 21) :
  (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by {
  sorry
}

end friends_prove_l183_183078


namespace problem_statement_l183_183870

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x ≥ 2}
def setC (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a ≥ 0}

theorem problem_statement (a : ℝ):
  (setA ∩ setB = {x : ℝ | 2 ≤ x ∧ x < 3}) ∧ 
  (setA ∪ setB = {x : ℝ | x ≥ -1}) ∧ 
  (setB ⊆ setC a → a > -4) :=
by
  sorry

end problem_statement_l183_183870


namespace total_grapes_l183_183267

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l183_183267


namespace bucket_ratio_l183_183850

theorem bucket_ratio :
  ∀ (leak_rate : ℚ) (duration : ℚ) (bucket_capacity : ℚ),
  leak_rate = 1.5 ∧ duration = 12 ∧ bucket_capacity = 36 →
  bucket_capacity / (leak_rate * duration) = 2 :=
by
  intros leak_rate duration bucket_capacity h
  have h_rate := h.1
  have h_duration := h.2.1
  have h_capacity := h.2.2
  rw [h_rate, h_duration, h_capacity]
  norm_num
  sorry

end bucket_ratio_l183_183850


namespace infinite_compositions_l183_183103

noncomputable def f (n : ℕ) (i : ℕ) : ℕ := 2 * i % n
noncomputable def g (n : ℕ) (i : ℕ) : ℕ := (2 * i + 1) % n

theorem infinite_compositions (n : ℕ) (h1 : n > 1) (ℓ m : ℕ) (h2 : ℓ < n) (h3 : m < n) : ∃ (H : ℕ → ℕ), ∀ k : ℕ, (∃ h : ℕ → (ℕ → ℕ), (∃ k : ℕ, h k ℓ = m)) :=
begin
  sorry
end

end infinite_compositions_l183_183103


namespace total_tickets_spent_l183_183306

def tickets_first_trip : ℕ := 2 + 10 + 2
def tickets_second_trip : ℕ := 3 + 7 + 5
def tickets_third_trip : ℕ := 8 + 15 + 4
def tickets_fourth_trip : ℕ := 10 + 8
def tickets_fifth_trip : ℕ := 15 / 2 + 5 / 2

theorem total_tickets_spent :
  tickets_first_trip + tickets_second_trip + tickets_third_trip + tickets_fourth_trip + tickets_fifth_trip = 84 :=
by
  -- Definitions of tickets for each trip
  have h1 : tickets_first_trip = 14 := rfl
  have h2 : tickets_second_trip = 15 := rfl
  have h3 : tickets_third_trip = 27 := rfl
  have h4 : tickets_fourth_trip = 18 := rfl
  have h5 : tickets_fifth_trip = 10 := by norm_num
  calc
    tickets_first_trip + tickets_second_trip + tickets_third_trip + tickets_fourth_trip + tickets_fifth_trip
    = 14 + 15 + 27 + 18 + 10 : by rw [h1, h2, h3, h4, h5]
    ... = 84 : by norm_num

end total_tickets_spent_l183_183306


namespace simplify_complex_fraction_l183_183912

/-- The simplified form of (5 + 7 * I) / (2 + 3 * I) is (31 / 13) - (1 / 13) * I. -/
theorem simplify_complex_fraction : (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i := 
by {
    sorry
}

end simplify_complex_fraction_l183_183912


namespace noah_sales_value_l183_183507

def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def price_large : ℕ := 60
def price_small : ℕ := 30

def this_month_large_sales : ℕ := 2 * last_month_large_sales
def this_month_small_sales : ℕ := 2 * last_month_small_sales

def this_month_large_sales_value : ℕ := this_month_large_sales * price_large
def this_month_small_sales_value : ℕ := this_month_small_sales * price_small

def this_month_total_sales : ℕ := this_month_large_sales_value + this_month_small_sales_value

theorem noah_sales_value :
  this_month_total_sales = 1200 :=
by
  sorry

end noah_sales_value_l183_183507


namespace min_value_f_1998_l183_183861

theorem min_value_f_1998
  (N : Type)
  (f : N → N)
  (h : ∀ (s t : N), f (t^2 * f s) = s * (f t)^2) :
  f 1998 = 1998 :=
sorry

end min_value_f_1998_l183_183861


namespace sum_of_exponentials_is_zero_l183_183660

theorem sum_of_exponentials_is_zero : 
  (∑ k in Finset.range 16, Complex.exp ((2 * Real.pi * k.succ : ℝ) * Complex.I / 17)) = 0 :=
by sorry

end sum_of_exponentials_is_zero_l183_183660


namespace base9_arithmetic_l183_183918

theorem base9_arithmetic : 
  let sum := ((3 * 9^2 + 7 * 9 + 4) + (6 * 9^2 + 2 * 9 + 5)) in
  let difference := (sum - (2 * 9^2 + 6 * 9 + 1)) in
  difference = (7 * 9^2 + 3 * 9 + 8) := by
  sorry

end base9_arithmetic_l183_183918


namespace find_y_l183_183435

theorem find_y (y: ℕ)
  (h1: ∃ (k : ℕ), y = 9 * k)
  (h2: y^2 > 225)
  (h3: y < 30)
: y = 18 ∨ y = 27 := 
sorry

end find_y_l183_183435


namespace find_unique_positive_integer_l183_183291

open Nat

theorem find_unique_positive_integer (n : ℕ) (h : 3 * 2^3 + 4 * 2^4 + 5 * 2^5 + ∑ i in range (n - 5), (i + 6) * 2^(i + 6) = 2^(n + 8)) : 
    n = 129 := 
by 
  sorry

end find_unique_positive_integer_l183_183291


namespace cyclic_quadrilateral_angles_l183_183555

theorem cyclic_quadrilateral_angles (a b c d : ℝ) :
  (a + b + c + d = 360) ∧
  (a + c = 180) ∧
  (b + d = 180) ∧
  (a / 45 = b / 90) ∧
  (b / 90 = c / 135) ∧
  (c / 135 = d / 90) →
  {a, b, c, d} = {45, 90, 135, 90} := 
sorry

end cyclic_quadrilateral_angles_l183_183555


namespace NoahClosetsFit_l183_183884

-- Declare the conditions as Lean variables and proofs
variable (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
variable (H1 : AliClosetCapacity = 200)
variable (H2 : NoahClosetsRatio = 1 / 4)
variable (H3 : NoahClosetsCount = 2)

-- Define the total number of jeans both of Noah's closets can fit
noncomputable def NoahTotalJeans : ℕ := (AliClosetCapacity * NoahClosetsRatio) * NoahClosetsCount

-- Theorem to prove
theorem NoahClosetsFit (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
  (H1 : AliClosetCapacity = 200) 
  (H2 : NoahClosetsRatio = 1 / 4) 
  (H3 : NoahClosetsCount = 2) 
  : NoahTotalJeans AliClosetCapacity NoahClosetsRatio NoahClosetsCount = 100 := 
  by 
    sorry

end NoahClosetsFit_l183_183884


namespace part_a_R_part_b_R_l183_183300

noncomputable def P_a := λ x : ℤ, x^6 - 6 * x^4 - 4 * x^3 + 9 * x^2 + 12 * x + 4
noncomputable def Q_a := λ x : ℤ, x^4 + x^3 - 3 * x^2 - 5 * x - 2

theorem part_a_R:
  ∀ x : ℤ, (P_a x) / (Q_a x) = x^2 - x - 2 := sorry

noncomputable def P_b := λ x : ℤ, x^5 + x^4 - 2 * x^3 - 2 * x^2 + x + 1
noncomputable def Q_b := λ x : ℤ, x^3 + x^2 - x - 1

theorem part_b_R:
  ∀ x : ℤ, (P_b x) / (Q_b x) = x^2 - 1 := sorry

end part_a_R_part_b_R_l183_183300


namespace ab_cd_eq_one_l183_183320

theorem ab_cd_eq_one (a b c d : ℕ) (p : ℕ) 
  (h_div_a : a % p = 0)
  (h_div_b : b % p = 0)
  (h_div_c : c % p = 0)
  (h_div_d : d % p = 0)
  (h_div_ab_cd : (a * b - c * d) % p = 0) : 
  (a * b - c * d) = 1 :=
sorry

end ab_cd_eq_one_l183_183320


namespace simplify_expression_l183_183648

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183648


namespace grace_apples_after_6_weeks_l183_183281

def apples_per_day_bella : ℕ := 6

def days_per_week : ℕ := 7

def fraction_apples_bella_consumes : ℚ := 1/3

def weeks : ℕ := 6

theorem grace_apples_after_6_weeks :
  let apples_per_week_bella := apples_per_day_bella * days_per_week
  let apples_per_week_grace := apples_per_week_bella / fraction_apples_bella_consumes
  let remaining_apples_week := apples_per_week_grace - apples_per_week_bella
  let total_apples := remaining_apples_week * weeks
  total_apples = 504 := by
  sorry

end grace_apples_after_6_weeks_l183_183281


namespace staircase_length_ratio_l183_183426

theorem staircase_length_ratio (floors : ℕ) (h1 : floors ≥ 4) : 
  (3 = 1) -> (3 = 1) sorry := sorry

end staircase_length_ratio_l183_183426


namespace noah_sales_value_l183_183508

def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def price_large : ℕ := 60
def price_small : ℕ := 30

def this_month_large_sales : ℕ := 2 * last_month_large_sales
def this_month_small_sales : ℕ := 2 * last_month_small_sales

def this_month_large_sales_value : ℕ := this_month_large_sales * price_large
def this_month_small_sales_value : ℕ := this_month_small_sales * price_small

def this_month_total_sales : ℕ := this_month_large_sales_value + this_month_small_sales_value

theorem noah_sales_value :
  this_month_total_sales = 1200 :=
by
  sorry

end noah_sales_value_l183_183508


namespace distance_from_point_to_hyperbola_asymptote_l183_183933

noncomputable def distance_to_asymptote (x1 y1 a b : ℝ) : ℝ :=
  abs (a * x1 + b * y1) / real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_hyperbola_asymptote :
  distance_to_asymptote 3 0 3 (-4) = 9 / 5 :=
by
  sorry

end distance_from_point_to_hyperbola_asymptote_l183_183933


namespace mixed_groups_count_l183_183970

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183970


namespace task1_task2_l183_183753

open Set

variable {α : Type*}

def A : Set ℝ := {x | |x| ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2 * m + 1}
def CU_A := {x : ℝ | x > 3 ∨ x < -3}

-- Task 1: If m = 3, prove (CU_A) ∩ B m = {x | 3 < x ∧ x < 7}
theorem task1 (m : ℝ) (h : m = 3) : (CU_A ∩ B m) = {x | 3 < x ∧ x < 7} :=
sorry

-- Task 2: If A ∪ B = A, find the range of m
theorem task2 (A ∪ B m = A) : -2 ≤ m ∧ m ≤ 1 :=
sorry

end task1_task2_l183_183753


namespace second_player_wins_with_optimal_play_l183_183962

-- Define the conditions as hypotheses
def boxes : Finset (Fin 11) := Finset.univ

def move (state : Fin 11 → ℕ) (not_box : Fin 11) : Fin 11 → ℕ :=
  fun box => if box = not_box then state box else state box + 1

def win (state : Fin 11 → ℕ) : Prop :=
  ∃ box, state box = 21

-- Define the theorem to be proven
theorem second_player_wins_with_optimal_play :
  ∃ moves : (Fin 11 → ℕ) → (Fin 11 → ℕ) ℕ → (Fin 11 → ℕ),
  ∀ (initial_state : Fin 11 → ℕ),
    (initial_state = fun _ => 0) →
      ((∀ n, n < 22 → ∃ state, state = (moves initial_state) n → win state) →  -- First player doesn't win
       ¬ win (moves initial_state) 22) → -- First player triggers the end state on turn 22
    win (moves initial_state) 23 := -- Second player wins on turn 23
sorry

end second_player_wins_with_optimal_play_l183_183962


namespace intersection_eq_l183_183756

def setM (x : ℝ) : Prop := x > -1
def setN (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem intersection_eq : {x : ℝ | setM x} ∩ {x | setN x} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end intersection_eq_l183_183756


namespace one_integral_root_exists_l183_183682

theorem one_integral_root_exists :
    ∃! x : ℤ, x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end one_integral_root_exists_l183_183682


namespace mixed_groups_count_l183_183992

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183992


namespace gray_region_area_correct_l183_183288

-- Define circles C and D with their centers and radii
def circle_C := {center := (3, 5), radius := 5}
def circle_D := {center := (13, 5), radius := 5}

-- Define the gray region area calculation
def gray_region_area (circle_C : {center : ℝ × ℝ, radius : ℝ}) 
                     (circle_D : {center : ℝ × ℝ, radius : ℝ}) : ℝ :=
  let rect_area := 10 * 5 in
  let sector_area := 2 * (1/4 * π * (circle_C.radius)^2) in
  rect_area - sector_area

-- Prove that the calculated area of the gray region is as expected
theorem gray_region_area_correct : 
  gray_region_area circle_C circle_D = 50 - 12.5 * π := sorry

end gray_region_area_correct_l183_183288


namespace y_intercept_of_function_l183_183917

theorem y_intercept_of_function :
  (let f (x : ℝ) := (4 * (x + 3) * (x - 2) - 24) / (x + 4) in f 0 = -12) :=
by
  let f (x : ℝ) := (4 * (x + 3) * (x - 2) - 24) / (x + 4)
  have h : f 0 = (4 * (0 + 3) * (0 - 2) - 24) / (0 + 4), by rw f
  -- Further calculations simplified in steps
  have h1 : 4 * (0 + 3) * (0 - 2) = -24, by ring
  have h2 : -24 - 24 = -48, by linarith
  have h3 : -48 / 4 = -12, by norm_num
  rw [h, h1, h2, h3]
  exact rfl

end y_intercept_of_function_l183_183917


namespace initial_amount_is_800_l183_183235

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end initial_amount_is_800_l183_183235


namespace series_sum_result_l183_183352

noncomputable def seriesSum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (1 / (k+1) / ((k+1)+2))

theorem series_sum_result (n : ℕ) :
  seriesSum n = (1/2) * ((3/2) - (1/(n+1)) - (1/(n+2))) :=
by
  sorry

end series_sum_result_l183_183352


namespace determine_a_l183_183409

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.exp x - a)^2

theorem determine_a (a x₀ : ℝ)
  (h₀ : f x₀ a ≤ 1/2) : a = 1/2 :=
sorry

end determine_a_l183_183409


namespace probability_third_smallest_is_five_l183_183902

theorem probability_third_smallest_is_five :
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  probability = Rat.ofInt 35 / 132 :=
by
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  show probability = Rat.ofInt 35 / 132
  sorry

end probability_third_smallest_is_five_l183_183902


namespace number_of_sets_xy_eq_6_and_yz_eq_15_l183_183720

theorem number_of_sets_xy_eq_6_and_yz_eq_15 : 
  { (x, y, z) : ℕ × ℕ × ℕ // x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y = 6 ∧ y * z = 15 }.to_finset.card = 2 :=
by
  sorry

end number_of_sets_xy_eq_6_and_yz_eq_15_l183_183720


namespace distinct_collections_in_bag_l183_183134

-- Definitions
def letters := ['A', 'L', 'G', 'E', 'B', 'R', 'A', 'I', 'C', 'S']
def vowels := ['A', 'A', 'E', 'I']
def consonants := ['L', 'G', 'B', 'B', 'R', 'C', 'C', 'S']

-- Problem Statement
theorem distinct_collections_in_bag : 
  (∃ (two_vowels : set char) (two_consonants : set char), 
    two_vowels ⊆ letters ∧ 
    two_consonants ⊆ letters ∧ 
    two_vowels.card = 2 ∧ 
    two_consonants.card = 2 ∧ 
    two_vowels ⊆ vowels ∧ 
    two_consonants ⊆ consonants) →
  68 := 
by 
  sorry

end distinct_collections_in_bag_l183_183134


namespace sum_of_consecutive_positive_odd_numbers_l183_183207

theorem sum_of_consecutive_positive_odd_numbers (n : ℕ) : 
  (∃ k ≥ 2, ∃ a : ℕ, n = ∑ i in finset.range k, (a + 2 * i)) ↔ (4 ∣ n ∨ (n % 2 = 1 ∧ ¬n.prime ∧ n > 1)) :=
by
  sorry

end sum_of_consecutive_positive_odd_numbers_l183_183207


namespace odd_digits_divisible_by_power_of_5_l183_183141

theorem odd_digits_divisible_by_power_of_5 (n : ℕ) (h : 0 < n) :
  ∃ x, (Nat.digits 10 x).All (λ d, d % 2 = 1) ∧ (Nat.digits 10 x).length = n ∧ x % (5^n) = 0 := 
sorry

end odd_digits_divisible_by_power_of_5_l183_183141


namespace find_k_l183_183698

theorem find_k (α : ℝ) (h : (sin α + 1 / sin α)^2 + (cos α + 1 / cos α)^2 + 2 = k + 2 * (sin α / cos α)^2 + 2 * (cos α / sin α)^2) : 
  k = 5 := 
sorry

end find_k_l183_183698


namespace lola_blueberry_pies_count_l183_183875

-- Let's define the conditions
def lola_mini_cupcakes := 13
def lola_pop_tarts := 10
def lulu_mini_cupcakes := 16
def lulu_pop_tarts := 12
def lulu_blueberry_pies := 14
def total_pastries := 73

-- Final statement: Prove the number of blueberry pies Lola baked
theorem lola_blueberry_pies_count : 
  let lola_pastries := total_pastries - (lulu_mini_cupcakes + lulu_pop_tarts + lulu_blueberry_pies),
      lola_blueberry_pies := lola_pastries - (lola_mini_cupcakes + lola_pop_tarts)
  in lola_blueberry_pies = 8 := sorry

end lola_blueberry_pies_count_l183_183875


namespace relationship_y1_y2_l183_183023

theorem relationship_y1_y2 (a : ℝ) (h_a : a < 0) (y1 y2 : ℝ) :
  let A := (-1 : ℝ, y1)
  let B := (2 : ℝ, y2)
  A.snd = a * (A.fst + 2) ^ 2 + 3 →
  B.snd = a * (B.fst + 2) ^ 2 + 3 →
  y1 > y2 :=
by
  sorry

end relationship_y1_y2_l183_183023


namespace production_volume_march_l183_183600

theorem production_volume_march (x y : ℝ) (h : y = x + 1) (hx : x = 3) : y = 4 :=
by 
  have hy : y = 3 + 1 := by rw [hx, h]
  rw hy
  norm_num


end production_volume_march_l183_183600


namespace distribution_ways_3x3_chessboard_l183_183159

theorem distribution_ways_3x3_chessboard :
  let numbers := (Finset.range 1 10).val in
  let chessboard := Matrix (Fin 3) (Fin 3) ℕ in
  let is_valid (m : chessboard) :=
    (∀ i, Finset.min (Finset.image (m i) Finset.univ) = 4 ∨
          Finset.min (Finset.image (m i) Finset.univ) ≥ 4) ∧
    (∀ j, Finset.max (Finset.image (λ i, m i j) Finset.univ) = 4 ∨
          Finset.max (Finset.image (λ i, m i j) Finset.univ) ≤ 4) in
  let count_distributions := (count $ filter is_valid (Finset.univ.powerset 9).val) in
  count_distributions = 25920 :=
sorry

end distribution_ways_3x3_chessboard_l183_183159


namespace quality_inspection_population_l183_183006

-- Definitions based on given conditions
def population (total_products : ℕ) := total_products
def sample (total_products sample_size : ℕ) := sample_size
def is_population (p : ℕ) := p = 80
def is_sample (s : ℕ) := s = 10
def sample_size_is_not_population_size (s p : ℕ) := s ≠ p
def sample_size_correct (s : ℕ) := s = 10

-- Theorem statements as per correct answers
theorem quality_inspection_population (total_products sample_size : ℕ) 
  (hp : is_population (population total_products))
  (hs : is_sample (sample total_products sample_size))
  (hsnp : sample_size_is_not_population_size sample_size total_products)
  (hsc : sample_size_correct sample_size) :
    hp ∧ hs ∧ hsnp ∧ hsc := sorry

end quality_inspection_population_l183_183006


namespace distance_to_asymptote_l183_183939

/-- Define the hyperbola equation as a predicate --/
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

/-- Define the asymptote equations as predicates --/
def asymptote1 (x y : ℝ) : Prop := 3 * x - 4 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- Define the distance formula from a point to a line --/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2))

/-- Proof statement that the distance from (3,0) to asymptote is 9/5 --/
theorem distance_to_asymptote : 
  distance_from_point_to_line 3 0 3 (-4) 0 = 9 / 5 :=
by
  -- the main proof computation goes here
  sorry

end distance_to_asymptote_l183_183939


namespace cosine_angle_vec_sum_and_diff_l183_183420

noncomputable def vec_a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def magnitude_b : ℝ := 2
noncomputable def angle_ab : ℝ := real.pi / 3

theorem cosine_angle_vec_sum_and_diff :
  let b := (0, 0) in -- We need to define b such that |b| = 2, angle_ab = π/3 with a.
  by sorry 
  ∃ (b : ℝ × ℝ), 
  (real.sqrt ((vec_a.1)^2 + (vec_a.2)^2)) = 2 ∧ 
  (real.sqrt ((b.1)^2 + (b.2)^2)) = 2 ∧ 
  real.angle vec_a b = real.pi/3 ∧
  real.cos ((real.sqrt ((vec_add vec_a b).1 + (vec_add vec_a b).2) ).1 + (vec_sub vec_a b).2)  = (1/2) :=
sorry

end cosine_angle_vec_sum_and_diff_l183_183420


namespace find_k_l183_183822

theorem find_k
  (k : ℝ)
  (AB : ℝ × ℝ := (3, 1))
  (AC : ℝ × ℝ := (2, k))
  (BC : ℝ × ℝ := (2 - 3, k - 1))
  (h_perpendicular : AB.1 * BC.1 + AB.2 * BC.2 = 0)
  : k = 4 :=
sorry

end find_k_l183_183822


namespace expression_of_quadratic_function_range_of_a_for_max_positive_l183_183016
noncomputable theory

-- Definitions from the conditions in a)
def quadratic_function_with_leading_coeff_a (f : ℝ → ℝ) (a : ℝ) :=
  ∃ b c, f = λ x, a * x^2 + b * x + c

def solution_set_of_inequality (f : ℝ → ℝ) (a : ℝ) :=
  (∀ x, f x > -2 * x ↔ 1 < x ∧ x < 3)

-- Proof questions translated to Lean 4 statements
theorem expression_of_quadratic_function
  (f : ℝ → ℝ) (a : ℝ)
  (hf : quadratic_function_with_leading_coeff_a f a)
  (hsol : solution_set_of_inequality f a)
  (heqroot : ∀ x, f x + 6 * a = 0 → x = 2 - 2 * a) :
  f = λ x, (-1 / 5) * x^2 - (6 / 5) * x - (3 / 5) :=
sorry

theorem range_of_a_for_max_positive
  (f : ℝ → ℝ) (a : ℝ)
  (hf : quadratic_function_with_leading_coeff_a f a)
  (hsol : solution_set_of_inequality f a)
  (hmax : ∃ M, ∀ x, f x ≤ M ∧ M > 0) :
  a ∈ Set.Ioo (-2 - Real.sqrt 3) (-2 + Real.sqrt 3) ∪ Set.Iio 0 :=
sorry

end expression_of_quadratic_function_range_of_a_for_max_positive_l183_183016


namespace surface_area_prism_l183_183381

/-- 
Given a prism with a regular triangular base and lateral edges perpendicular to the base, 
a sphere with a volume of 4π/3 is tangent to all faces of the prism, 
prove that the surface area of this prism is 18√3.
-/
theorem surface_area_prism : 
  ∀ (R h a : ℝ), 
  R = 1 → 
  h = 2 * R → 
  (∃ a, (1 / 3) * (√3 / 2) * a = 1) → 
  a = 2 * √3 → 
  (3 * a * h + 2 * (√3 / 4) * a^2) = 18 * √3 := 
by 
  intros R h a hR eq_h ex_a eq_a;
  sorry

end surface_area_prism_l183_183381


namespace find_a_of_complex_eq_l183_183068

theorem find_a_of_complex_eq (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * (⟨1, -a⟩ : ℂ) = 2) : a = 1 :=
by
  sorry

end find_a_of_complex_eq_l183_183068


namespace true_propositions_count_is_zero_l183_183771

theorem true_propositions_count_is_zero :
  (¬ (0 > -complex.I)) ∧ 
  (∀ (z w : ℂ), (w = conj z ↔ (z + w).im = 0)) ∧ 
  (∀ (x y : ℂ), (x + y * complex.I = 1 + complex.I ↔ x = 1 ∧ y = 1)) ∧ 
  (¬ (function.bijective (λ a : ℝ, a * complex.I))) → 
  (number of true propositions is 0) :=
by {
  sorry
}

end true_propositions_count_is_zero_l183_183771


namespace diagonal_placement_l183_183318

-- Predicate for a valid diagonal placement
def is_valid_diagonal_placement (grid : List (List (Bool × Bool))) : Prop :=
  ∀ (r1 c1 r2 c2 : Nat), (r1, c1) ≠ (r2, c2) →
  (grid[r1][c1].1 = grid[r2][c2].1 → grid[r1][c1].1 = false) ∧
  (grid[r1][c1].2 = grid[r2][c2].2 → grid[r1][c1].2 = false)

-- Predicate to check if no additional diagonals can be added without violating the rules
def is_maximal_diagonal_placement (grid : List (List (Bool × Bool))) : Prop :=
  ∀ (r c : Nat), grid[r][c].1 || grid[r][c].2 →
  (r1 r2 c1 c2 : Nat) → (c1 ≠ c2 ∨ r1 ≠ r2) →
  ¬ (r1 ≤ 3 ∧ r2 ≤ 3 ∧ c1 ≤ 3 ∧ c2 ≤ 3)

theorem diagonal_placement (n : Nat) (grid : List (List (Bool × Bool)))
  (h_grid_size : grid.length = 4 ∧ ∀ r, grid[r].length = 4)
  (h_diags_count : grid.map (λ row, row.filter (λ cell, cell.1 || cell.2)).foldl (λ sum row, sum + row.length) 0 = 8)
  (h_valid_placement : is_valid_diagonal_placement grid)
  (h_maximal_placement : is_maximal_diagonal_placement grid) :
  is_valid_diagonal_placement grid ∧ is_maximal_diagonal_placement grid := by
  sorry

end diagonal_placement_l183_183318


namespace PQ_length_squared_PS_length_squared_QS_length_squared_altitudes_intersect_at_O_l183_183945

-- Definitions of points and distances
variables (A B C D P Q S O : Type)
variables (p q s R : ℝ)  -- distances from points to O and the radius

-- Conditions as assumptions
variables (circumscribed : inscribed A B C D O)
variables (P_intersect : extension_intersect A B D C P)
variables (Q_intersect : extension_intersect B C A D Q)
variables (S_intersect : diagonals_intersect A C B D S)
variables (dist_P : dist O P = p)
variables (dist_Q : dist O Q = q)
variables (dist_S : dist O S = s)
variables (radius : circumscribed_circle_radius A B C D R)

-- Proof of side lengths for triangle PQS
theorem PQ_length_squared : dist P Q ^ 2 = p ^ 2 + q ^ 2 - 2 * R ^ 2 :=
by sorry

theorem PS_length_squared : dist P S ^ 2 = p ^ 2 + s ^ 2 - 2 * R ^ 2 :=
by sorry

theorem QS_length_squared : dist Q S ^ 2 = q ^ 2 + s ^ 2 - 2 * R ^ 2 :=
by sorry

-- Proof that the altitudes intersect at O
theorem altitudes_intersect_at_O : altitudes_intersect_at P Q S O :=
by sorry

end PQ_length_squared_PS_length_squared_QS_length_squared_altitudes_intersect_at_O_l183_183945


namespace sum_of_coefficients_l183_183001

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ) :
  (2 - 0)^7 = a_0 + a_1 * (1 + 0) + a_2 * (1 + 0)^2 + a_3 * (1 + 0)^3 + 
              a_4 * (1 + 0)^4 + a_5 * (1 + 0)^5 + a_6 * (1 + 0)^6 + 
              a_7 * (1 + 0)^7 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 129 :=
begin
  sorry
end

end sum_of_coefficients_l183_183001


namespace instantaneous_velocity_at_1s_l183_183031

noncomputable theory

def displacement (t : ℝ) : ℝ := -t^2

theorem instantaneous_velocity_at_1s :
  deriv displacement 1 = -2 :=
by
  unfold displacement
  simp
  sorry

end instantaneous_velocity_at_1s_l183_183031


namespace A_inter_B_empty_iff_l183_183869

variable (m : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem A_inter_B_empty_iff : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by
  sorry

end A_inter_B_empty_iff_l183_183869


namespace sum_of_16_roots_of_unity_l183_183659

theorem sum_of_16_roots_of_unity : 
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17) in
  ∑ i in finset.range 16, ω^(i+1) = -1 :=
by
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17)
  have h₁ : ω^17 = 1 := sorry
  have h₂ : ∑ i in finset.range 17, ω^i = 0 := sorry
  sorry

end sum_of_16_roots_of_unity_l183_183659


namespace area_of_square_with_diagonal_two_l183_183443

theorem area_of_square_with_diagonal_two {a d : ℝ} (h : d = 2) (h' : d = a * Real.sqrt 2) : a^2 = 2 := 
by
  sorry

end area_of_square_with_diagonal_two_l183_183443


namespace part1_part2_l183_183489

def U := ℝ
def A := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1}

theorem part1 (m : ℝ) (hm: m = 3) : 
  A ∩ (set.univ \ B m) = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by
  rw [hm, set.compl_set_of, set.univ_diff, set.inter_set_of]
  sorry

theorem part2 (H : ∀ x, (x ∈ B 3) → (x ∈ A)): 
  set.Icc 1 2 = {m : ℝ | 1 ≤ m ∧ m ≤ 2} :=
by
  sorry

end part1_part2_l183_183489


namespace number_of_integers_such_that_P_n_leq_0_l183_183672

def P (x : ℤ) : ℤ :=
  (x - 1^2) * (x - 2^2) * ... * (x - 50^2)

theorem number_of_integers_such_that_P_n_leq_0 :
  ∃ (n : ℕ), n = 1300 ∧ ∀ k : ℤ, P(k) <= 0 ↔ (k <= n) :=
sorry

end number_of_integers_such_that_P_n_leq_0_l183_183672


namespace power_identity_l183_183376

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end power_identity_l183_183376


namespace area_of_circle_l183_183785
-- import necessary libraries

noncomputable theory
open Real

-- define the conditions: line and circle
def line (a : ℝ) : ℝ → ℝ := λ x, a * x
def circle (a : ℝ) : ℝ × ℝ → Prop :=
  λ p, let ⟨x, y⟩ := p in x^2 + y^2 - 2*a*x - 2*y + 2 = 0

-- define the hypothesis that ΔABC is an equilateral triangle
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d := λ p q, Real.mul (Real.sqrt (Real.pow (p.1 - q.1) 2 + Real.pow (p.2 - q.2) 2)) (Real.sqrt 3) / 2
  in dist A B = d B C

-- define the problem as a theorem statement
theorem area_of_circle (a : ℝ) :
  (∀ A B : ℝ × ℝ, line a (A.1) = A.2 ∧ circle a A ∧ line a (B.1) = B.2 ∧ circle a B → is_equilateral_triangle A B (a,1)) →
  let R := Real.sqrt (a^2 - 1) in π * R^2 = 6 * π :=
by 
  -- because we are only writing the statement, we end the proof here with sorry
  sorry

end area_of_circle_l183_183785


namespace seats_not_occupied_l183_183328

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_l183_183328


namespace even_perfect_square_factors_count_l183_183423

open Nat

def is_even (n : ℕ) := n % 2 = 0

theorem even_perfect_square_factors_count :
  let a_choices := filter (λ a, is_even a ∧ 1 ≤ a ∧ a ≤ 6) (range 7)
  let b_choices := filter (λ b, is_even b ∧ b ≤ 9) (range 10)
  let c_choices := filter (λ c, is_even c ∧ c ≤ 2) (range 3)
  a_choices.length * b_choices.length * c_choices.length = 30 :=
by
  -- Definitions:
  let a_choices := filter (λ a, is_even a ∧ 1 ≤ a ∧ a ≤ 6) (range 7)
  let b_choices := filter (λ b, is_even b ∧ b ≤ 9) (range 10)
  let c_choices := filter (λ c, is_even c ∧ c ≤ 2) (range 3)
  -- Now let's prove the lengths:
  have a_count : a_choices.length = 3 := sorry
  have b_count : b_choices.length = 5 := sorry
  have c_count : c_choices.length = 2 := sorry
  -- Multiplying the counted choices:
  have final_count : (a_choices.length * b_choices.length * c_choices.length = 3 * 5 * 2) := sorry
  -- Therefore:
  have : 3 * 5 * 2 = 30 := by norm_num
  show a_choices.length * b_choices.length * c_choices.length = 30 from final_count.trans this

end even_perfect_square_factors_count_l183_183423


namespace jerry_reaches_six_at_least_once_during_tosses_jerry_a_plus_b_l183_183097

-- Definitions based on conditions
def fair_coin (p : ℕ) := 
  (nat.choose p 8) * 2 / 2^p

def total_prob (p : ℕ) := 
  (nat.choose p 8) / 2^p

theorem jerry_reaches_six_at_least_once_during_tosses
  (p : ℕ) (h : p = 10) :
  (fair_coin p) = (45 / 512) :=
begin
  sorry
end

theorem jerry_a_plus_b 
  (a b : ℕ) (h1 : a = 45) (h2 : b = 512) : 
  a + b = 557 :=
begin
  sorry
end

end jerry_reaches_six_at_least_once_during_tosses_jerry_a_plus_b_l183_183097


namespace range_f_l183_183667

noncomputable def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

theorem range_f : set.range f = set.Icc (-8 : ℝ) (18 : ℝ) :=
sorry

end range_f_l183_183667


namespace infinitely_many_lines_parallel_to_plane_l183_183244

noncomputable theory
open_locale classical

open Set

-- Define the given plane as a set of points in ℝ³
def given_plane (P : Set (ℝ × ℝ × ℝ)) : Prop := ∃ a b c d : ℝ, P = {p : ℝ × ℝ × ℝ | a * p.1 + b * p.2 + c * p.3 = d}

-- Define a point outside the given plane
def point_outside_plane (P : Set (ℝ × ℝ × ℝ)) (Q : ℝ × ℝ × ℝ) : Prop := 
  given_plane P ∧ ¬ (Q ∈ P)

-- Theorem stating the existence of infinitely many lines parallel to the plane through the point outside the plane.
theorem infinitely_many_lines_parallel_to_plane 
  (P : Set (ℝ × ℝ × ℝ)) (Q : ℝ × ℝ × ℝ)
  (h1 : given_plane P)
  (h2 : point_outside_plane P Q) : 
  ∃ L : Set (Set (ℝ × ℝ × ℝ)), 
  (∀ l ∈ L, ∀ p1 p2 ∈ l, p1 ≠ p2) ∧ -- each set in L is a line through distinct points
  (∀ l ∈ L, ∀ l' ∈ L, l ≠ l' → l ∩ l' = ∅) ∧ -- lines in L are distinct
  ∀ l ∈ L, Q ∈ l ∧ ∀ p ∈ l, p ∉ P := -- each line in L passes through Q and is parallel to P
  sorry

end infinitely_many_lines_parallel_to_plane_l183_183244


namespace necessary_and_sufficient_condition_l183_183392

theorem necessary_and_sufficient_condition (a b : ℝ) (h : a * b ≠ 0) : 
  a - b = 1 ↔ a^3 - b^3 - a * b - a^2 - b^2 = 0 := by
  sorry

end necessary_and_sufficient_condition_l183_183392


namespace mixed_groups_count_l183_183979

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183979


namespace mixed_groups_count_l183_183968

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183968


namespace area_ratio_half_l183_183500

variables {A B C M P D : Type} [AffineSpace ℚ A] (A B C M P D : A)
variables [midpoint : AffineMap ℚ (triangle_space A B) A B M] 
variables [on_segment_P : line_segment A M P] 
variables [parallel_MD_PC : ∀ G : line_segment B C A B, C, parallel_segment P D C] 

theorem area_ratio_half 
  (midpoint_M : midpoint A B M)
  (P_on_AB_between_A_and_M : on_segment_P A M P)
  (MD_parallel_PC : ∀ G, parallel_MD_PC (G A) (segment P) (segment C) D) :
  ∃ r : ℚ, r = 1 / 2 := 
by
  sorry -- proof goes here

end area_ratio_half_l183_183500


namespace find_function_that_satisfies_eq_l183_183864

theorem find_function_that_satisfies_eq :
  ∀ (f : ℕ → ℕ), (∀ (m n : ℕ), f (m + f n) = f (f m) + f n) → (∀ n : ℕ, f n = n) :=
by
  intro f
  intro h
  sorry

end find_function_that_satisfies_eq_l183_183864


namespace distinct_lines_l183_183865

theorem distinct_lines :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → 
  {y | ∃ x : ℝ, y = (frac a (abs a) + frac b (abs b) + frac c (abs c)) * x}.finite.card = 4 :=
by sorry

end distinct_lines_l183_183865


namespace sum_formula_l183_183899

theorem sum_formula (m : ℕ) (h : 0 < m) : 
  (∑ k in Finset.range m, (m * (m-1) * ... * (m-k+1) * k) / (m ^ (k+1))) = 1 := 
sorry

end sum_formula_l183_183899


namespace mixed_groups_count_l183_183966

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183966


namespace chickens_pigs_legs_l183_183451

variable (x : ℕ)

-- Define the conditions
def sum_chickens_pigs (x : ℕ) : Prop := x + (70 - x) = 70
def total_legs (x : ℕ) : Prop := 2 * x + 4 * (70 - x) = 196

-- Main theorem to prove the given mathematical statement
theorem chickens_pigs_legs (x : ℕ) (h1 : sum_chickens_pigs x) (h2 : total_legs x) : (2 * x + 4 * (70 - x) = 196) :=
by sorry

end chickens_pigs_legs_l183_183451


namespace binary_multiplication_division_result_l183_183697

theorem binary_multiplication_division_result : 
  let a := 11100₂ 
  let b := 11010₂ 
  let c := 100₂ in
  (a * b / c) = 10100110₂ :=
by 
  sorry

end binary_multiplication_division_result_l183_183697


namespace sum_of_16_roots_of_unity_l183_183657

theorem sum_of_16_roots_of_unity : 
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17) in
  ∑ i in finset.range 16, ω^(i+1) = -1 :=
by
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17)
  have h₁ : ω^17 = 1 := sorry
  have h₂ : ∑ i in finset.range 17, ω^i = 0 := sorry
  sorry

end sum_of_16_roots_of_unity_l183_183657


namespace no_valid_n_l183_183359

-- Definition of the polynomial corresponding to the base-n number 321032_n
def f (n : ℕ) : ℕ :=
  3 + 2 * n + n^2 + 3 * n^4 + 2 * n^5

-- The theorem statement that leads to the conclusion there are no such n
theorem no_valid_n (h : 2 ≤ 100) :
  ∃ n : ℕ, (2 ≤ n ∧ n ≤ 100) ∧ f(n) % 9 = 0 :=
  false :=
by
  sorry

end no_valid_n_l183_183359


namespace set_A_correct_l183_183121

-- Definition of the sets and conditions
def A : Set ℤ := {-3, 0, 2, 6}
def B : Set ℤ := {-1, 3, 5, 8}

theorem set_A_correct : 
  (∃ a1 a2 a3 a4 : ℤ, A = {a1, a2, a3, a4} ∧ 
  {a1 + a2 + a3, a1 + a2 + a4, a1 + a3 + a4, a2 + a3 + a4} = B) → 
  A = {-3, 0, 2, 6} :=
by 
  sorry

end set_A_correct_l183_183121


namespace distance_from_point_to_asymptote_l183_183938

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l183_183938


namespace prime_pairs_l183_183700

theorem prime_pairs (p q : ℕ) : 
  p < 2005 → q < 2005 → 
  Prime p → Prime q → 
  (q ∣ p^2 + 4) → 
  (p ∣ q^2 + 4) → 
  (p = 2 ∧ q = 2) :=
by sorry

end prime_pairs_l183_183700


namespace trigonometric_sum_eq_zero_l183_183625

theorem trigonometric_sum_eq_zero :
  sin (-1071 * real.pi / 180) * sin (99 * real.pi / 180) +
  sin (-171 * real.pi / 180) * sin (-261 * real.pi / 180) +
  tan (-1089 * real.pi / 180) * tan (-540 * real.pi / 180) = 0 :=
sorry

end trigonometric_sum_eq_zero_l183_183625


namespace price_after_two_reductions_l183_183253

-- Define the two reductions as given in the conditions
def first_day_reduction (P : ℝ) : ℝ := P * 0.88
def second_day_reduction (P : ℝ) : ℝ := first_day_reduction P * 0.9

-- Main theorem: Price on the second day is 79.2% of the original price
theorem price_after_two_reductions (P : ℝ) : second_day_reduction P = 0.792 * P :=
by
  sorry

end price_after_two_reductions_l183_183253


namespace probability_dice_product_multiple_of_4_l183_183145

theorem probability_dice_product_multiple_of_4 :
  let sam_dice := fin 12
  let alex_dice := fin 6
  let favorable_cases := (3 / 12) + (2 / 12) * (2 / 6)
  let total_probability := favorable_cases
  total_probability = 11 / 36 :=
by
  sorry

end probability_dice_product_multiple_of_4_l183_183145


namespace balance_difference_l183_183619

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  (round diff = 3553) :=
by 
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  have h : round diff = 3553 := sorry
  assumption

end balance_difference_l183_183619


namespace proof_investment_values_l183_183890

def AA_init : ℝ := 150
def BB_init : ℝ := 120
def CC_init : ℝ := 100

def AA_1y : ℝ := AA_init * 1.15
def BB_1y : ℝ := BB_init * 0.70
def CC_1y : ℝ := CC_init

def AA_2y : ℝ := AA_1y * 0.85
def BB_2y : ℝ := BB_1y * 1.20
def CC_2y : ℝ := CC_1y

def AA_3y : ℝ := AA_2y * 1.10
def BB_3y : ℝ := BB_2y * 0.95
def CC_3y : ℝ := CC_2y * 1.05

theorem proof_investment_values : BB_3y < CC_3y ∧ CC_3y < AA_3y :=
by
  have h1 : AA_3y = 146.625 * 1.10 := rfl
  have h2 : BB_3y = 100.8 * 0.95 := rfl
  have h3 : CC_3y = 100 * 1.05 := rfl
  rw [h1, h2, h3]
  have hAA_3y : AA_3y = 161.2875 := by norm_num
  have hBB_3y : BB_3y = 95.76 := by norm_num
  have hCC_3y : CC_3y = 105 := by norm_num
  rw [hAA_3y, hBB_3y, hCC_3y]
  apply and.intro
  · norm_num
  · norm_num

end proof_investment_values_l183_183890


namespace smallest_n_65_l183_183540

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def smallest_n (n : ℕ) : Prop :=
  ∃ (numbers : ℕ → ℕ),                    -- a function assigning a number to each circle
    (∀ i j, ¬ connected i j → gcd ((numbers i) ^ 2 + (numbers j) ^ 2) n = 1) →
    (∀ i j, connected i j → gcd ((numbers i) ^ 2 + (numbers j) ^ 2) n > 1)

theorem smallest_n_65 : smallest_n 65 :=
by sorry

end smallest_n_65_l183_183540


namespace boiling_temperature_l183_183095

-- Definitions according to conditions
def initial_temperature : ℕ := 41

def temperature_increase_per_minute : ℕ := 3

def pasta_cooking_time : ℕ := 12

def mixing_and_salad_time : ℕ := pasta_cooking_time / 3

def total_evening_time : ℕ := 73

-- Conditions and the problem statement in Lean
theorem boiling_temperature :
  initial_temperature + (total_evening_time - (pasta_cooking_time + mixing_and_salad_time)) * temperature_increase_per_minute = 212 :=
by
  -- Here would be the proof, skipped with sorry
  sorry

end boiling_temperature_l183_183095


namespace xyz_sum_l183_183065

theorem xyz_sum (x y z : ℕ) (h1 : xyz = 240) (h2 : xy + z = 46) (h3 : x + yz = 64) : x + y + z = 20 :=
sorry

end xyz_sum_l183_183065


namespace calculate_a_minus_b_l183_183626

theorem calculate_a_minus_b (a b : ℝ) 
  (h : {1, a, b / a} = {0, a^2, a + b}) : a - b = -1 :=
sorry

end calculate_a_minus_b_l183_183626


namespace circumcenter_iff_perimeters_l183_183114

variables {O A B C A1 B1 C1 : Type*}
variables [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]
variables [Point O A B C A1 B1 C1]

def is_acute_triangle (A B C : Type*) [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))] :=
  ∀ (a b c : Type*), 
  (dist a b)^2 + (dist b c)^2 > (dist a c)^2  ∧ 
  (dist b c)^2 + (dist a c)^2 > (dist a b)^2  ∧ 
  (dist a b)^2 + (dist a c)^2 > (dist b c)^2

def is_perpendicular (O P : Point) (line : Line) := (O - line.point1) ⬝ (line.point2 - line.point1) = 0

variables (OA1perpendicular : is_perpendicular O A1 (Line.mk B C))
variables (OB1perpendicular : is_perpendicular O B1 (Line.mk C A))
variables (OC1perpendicular : is_perpendicular O C1 (Line.mk A B))

def perimeter (points : list (Point)) := 
  points.pairwise_list.map dist.zip
  |>.sum

theorem circumcenter_iff_perimeters (O A B C A1 B1 C1 : Type*)
  [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]
  [Point O A B C A1 B1 C1]
  (h_ac : is_acute_triangle A B C)
  (h1 : OA1perpendicular)
  (h2 : OB1perpendicular)
  (h3 : OC1perpendicular) : 
  (is_circumcenter O A B C) ↔ 
  perimeter [A1, B1, C1] ≥ perimeter [A, B1, C1] ∧
  perimeter [A1, B1, C1] ≥ perimeter [B, C1, A1] ∧
  perimeter [A1, B1, C1] ≥ perimeter [C, A1, B1] :=
sorry

end circumcenter_iff_perimeters_l183_183114


namespace penalty_kicks_calculation_l183_183152

def totalPlayers := 24
def goalkeepers := 4
def nonGoalkeeperShootsAgainstOneGoalkeeper := totalPlayers - 1
def totalPenaltyKicks := goalkeepers * nonGoalkeeperShootsAgainstOneGoalkeeper

theorem penalty_kicks_calculation : totalPenaltyKicks = 92 := by
  sorry

end penalty_kicks_calculation_l183_183152


namespace all_logarithmic_are_monotonic_exists_int_divisible_by_11_and_9_for_all_x_gt_zero_x_add_inv_ge_two_exists_x0_in_Z_log2_x0_gt_2_l183_183313

-- Problem 1: Prove all logarithmic functions are monotonic functions
theorem all_logarithmic_are_monotonic : 
  ∀ (f : ℝ → ℝ), (∀ a b, a < b → f a ≤ f b) → monotone f := 
begin
  sorry
end

-- Problem 2: Prove there exists at least one integer that is divisible by both 11 and 9
theorem exists_int_divisible_by_11_and_9 : 
  ∃ (n : ℤ), (11 ∣ n) ∧ (9 ∣ n) :=
begin
  sorry
end

-- Problem 3: Prove for all x > 0, x + 1/x ≥ 2
theorem for_all_x_gt_zero_x_add_inv_ge_two : 
  ∀ (x : ℝ), x > 0 → x + 1 / x ≥ 2 :=
begin
  sorry
end

-- Problem 4: Prove there exists an x0 in ℤ such that log2(x0) > 2
theorem exists_x0_in_Z_log2_x0_gt_2 : 
  ∃ (x0 : ℤ), real.log x0 / real.log 2 > 2 :=
begin
  sorry
end

end all_logarithmic_are_monotonic_exists_int_divisible_by_11_and_9_for_all_x_gt_zero_x_add_inv_ge_two_exists_x0_in_Z_log2_x0_gt_2_l183_183313


namespace analytical_expression_of_f_range_of_m_l183_183809

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem analytical_expression_of_f :
  (∀ (x : ℝ), f (1 + x) = f (1 - x)) ∧ (∀ (x : ℝ), ∃ (m : ℝ), f x ≥ 2*x + 1 + m) ∧
  (m ∈ Icc 0 1 → f 0 = 0 ∧ (∃ (min_val : ℝ), f min_val = -1)) → 
  f = fun x => x^2 - 2*x :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc 0 1, f x ≥ 2*x + 1 + m) → m ∈ Iio (-4) :=
sorry

end analytical_expression_of_f_range_of_m_l183_183809


namespace cricket_team_right_handed_count_l183_183132

theorem cricket_team_right_handed_count 
  (total throwers non_throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h_total : total = 70)
  (h_throwers : throwers = 37)
  (h_non_throwers : non_throwers = total - throwers)
  (h_left_handed_non_throwers : left_handed_non_throwers = non_throwers / 3)
  (h_right_handed_non_throwers : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h_all_throwers_right_handed : ∀ (t : ℕ), t = throwers → t = right_handed_non_throwers + (total - throwers) - (non_throwers / 3)) :
  right_handed_non_throwers + throwers = 59 := 
by 
  sorry

end cricket_team_right_handed_count_l183_183132


namespace distance_from_point_to_hyperbola_asymptote_l183_183931

noncomputable def distance_to_asymptote (x1 y1 a b : ℝ) : ℝ :=
  abs (a * x1 + b * y1) / real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_hyperbola_asymptote :
  distance_to_asymptote 3 0 3 (-4) = 9 / 5 :=
by
  sorry

end distance_from_point_to_hyperbola_asymptote_l183_183931


namespace convert_245_deg_to_rad_l183_183301

-- Define the degree to radian conversion factor
def degree_to_radian (d : ℤ) : ℝ := d * (Real.pi / 180)

-- Theorem: convert 245 degrees to radians
theorem convert_245_deg_to_rad :
  degree_to_radian 245 = 245 * (Real.pi / 180) :=
by
  sorry

end convert_245_deg_to_rad_l183_183301


namespace gcd_lcm_difference_multiplied_smaller_equals_160_l183_183311

theorem gcd_lcm_difference_multiplied_smaller_equals_160 :
  let a := 8
  let b := 12
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  let smaller := Nat.min a b
  (lcm_ab - gcd_ab) * smaller = 160 := 
by
  let a := 8
  let b := 12
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  let smaller := Nat.min a b
  have h1 : gcd_ab = 4 := rfl
  have h2 : lcm_ab = 24 := rfl
  have h3 : smaller = 8 := rfl
  calc
    (lcm_ab - gcd_ab) * smaller 
    = (24 - 4) * 8 : by rw [h1, h2, h3]
    = 20 * 8 : by rfl
    = 160 : by rfl

end gcd_lcm_difference_multiplied_smaller_equals_160_l183_183311


namespace ratio_of_average_speeds_l183_183216

def distance_A_to_B : ℝ := 480
def distance_A_to_C : ℝ := 300
def time_Eddy : ℝ := 3
def time_Freddy : ℝ := 4

def avg_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem ratio_of_average_speeds :
  avg_speed distance_A_to_B time_Eddy / avg_speed distance_A_to_C time_Freddy = 32 / 15 :=
by
  sorry

end ratio_of_average_speeds_l183_183216


namespace statement_T_true_for_given_values_l183_183488

/-- Statement T: If the sum of the digits of a whole number m is divisible by 9, 
    then m is divisible by 9.
    The given values to check are 45, 54, 81, 63, and none of these. --/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem statement_T_true_for_given_values :
  ∀ (m : ℕ), (m = 45 ∨ m = 54 ∨ m = 81 ∨ m = 63) →
    (is_divisible_by_9 (sum_of_digits m) → is_divisible_by_9 m) :=
by
  intros m H
  cases H
  case inl H1 => sorry
  case inr H2 =>
    cases H2
    case inl H1 => sorry
    case inr H2 =>
      cases H2
      case inl H1 => sorry
      case inr H2 => sorry

end statement_T_true_for_given_values_l183_183488


namespace simplify_complex_fraction_l183_183911

/-- The simplified form of (5 + 7 * I) / (2 + 3 * I) is (31 / 13) - (1 / 13) * I. -/
theorem simplify_complex_fraction : (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i := 
by {
    sorry
}

end simplify_complex_fraction_l183_183911


namespace mixed_groups_count_l183_183977

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183977


namespace zachary_needs_more_money_l183_183005

def cost_in_usd_football (euro_to_usd : ℝ) (football_cost_eur : ℝ) : ℝ :=
  football_cost_eur * euro_to_usd

def cost_in_usd_shorts (gbp_to_usd : ℝ) (shorts_cost_gbp : ℝ) (pairs : ℕ) : ℝ :=
  shorts_cost_gbp * pairs * gbp_to_usd

def cost_in_usd_shoes (shoes_cost_usd : ℝ) : ℝ :=
  shoes_cost_usd

def cost_in_usd_socks (jpy_to_usd : ℝ) (socks_cost_jpy : ℝ) (pairs : ℕ) : ℝ :=
  socks_cost_jpy * pairs * jpy_to_usd

def cost_in_usd_water_bottle (krw_to_usd : ℝ) (water_bottle_cost_krw : ℝ) : ℝ :=
  water_bottle_cost_krw * krw_to_usd

def total_cost_before_discount (cost_football_usd cost_shorts_usd cost_shoes_usd
                                cost_socks_usd cost_water_bottle_usd : ℝ) : ℝ :=
  cost_football_usd + cost_shorts_usd + cost_shoes_usd + cost_socks_usd + cost_water_bottle_usd

def discounted_total_cost (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost * (1 - discount)

def additional_money_needed (discounted_total_cost current_money : ℝ) : ℝ :=
  discounted_total_cost - current_money

theorem zachary_needs_more_money (euro_to_usd : ℝ) (gbp_to_usd : ℝ) (jpy_to_usd : ℝ) (krw_to_usd : ℝ)
  (football_cost_eur : ℝ) (shorts_cost_gbp : ℝ) (pairs_shorts : ℕ) (shoes_cost_usd : ℝ)
  (socks_cost_jpy : ℝ) (pairs_socks : ℕ) (water_bottle_cost_krw : ℝ) (current_money_usd : ℝ)
  (discount : ℝ) : additional_money_needed 
      (discounted_total_cost
          (total_cost_before_discount
            (cost_in_usd_football euro_to_usd football_cost_eur)
            (cost_in_usd_shorts gbp_to_usd shorts_cost_gbp pairs_shorts)
            (cost_in_usd_shoes shoes_cost_usd)
            (cost_in_usd_socks jpy_to_usd socks_cost_jpy pairs_socks)
            (cost_in_usd_water_bottle krw_to_usd water_bottle_cost_krw)) 
          discount) 
      current_money_usd = 7.127214 := 
sorry

end zachary_needs_more_money_l183_183005


namespace number_of_odd_divisors_l183_183565

theorem number_of_odd_divisors (n : ℕ) (h : n = 2^3 * 5^2 * 11) : 
  (finset.filter (λ x : ℕ, (x % 2 = 1)) (finset.divisors n)).card = 6 :=
by sorry

end number_of_odd_divisors_l183_183565


namespace average_speed1_average_speed2_l183_183536

-- Define the total distance function based on time
def total_distance (t : ℕ) : ℕ :=
  if t = 6 then 0 else
  if t = 7 then 40 else
  if t = 8 then 80 else
  if t = 9 then 120 else
  if t = 10 then 160 else
  if t = 11 then 180 else
  if t = 12 then 200 else 
  if t = 13 then 220 else
  if t = 14 then 240 else 0

-- Proof Problem 1: Prove Tom's average speed from 6 a.m. to 2 p.m. is 30 miles per hour
theorem average_speed1 : 
  let total_distance_6am_to_2pm := total_distance 14 - total_distance 6,
      total_time_6am_to_2pm := 8 in
  total_distance_6am_to_2pm / total_time_6am_to_2pm = 30 := by
    sorry

-- Proof Problem 2: Prove Tom's average speed from 10 a.m. to 12 p.m. is 15 miles per hour
theorem average_speed2 : 
  let distance_10am_to_12pm := total_distance 12 - total_distance 10,
      time_10am_to_12pm := 2 in
  distance_10am_to_12pm / time_10am_to_12pm = 15 := by
    sorry

end average_speed1_average_speed2_l183_183536


namespace least_possible_both_proof_l183_183448

variables (students_blue_eyes students_water_bottle students_total : ℕ)

def least_possible_both (students_blue_eyes students_water_bottle students_total : ℕ) : ℕ :=
  students_blue_eyes - (students_total - students_water_bottle)

theorem least_possible_both_proof :
  students_blue_eyes = 18 → students_water_bottle = 25 → students_total = 35 →
  least_possible_both 18 25 35 = 8 :=
by
  intros h1 h2 h3
  -- Students who do not have a water bottle
  have h4 : students_total - students_water_bottle = 10, by {
    rw [h2, h3],
    norm_num,
  }
  -- Maximum number of students with blue eyes who do not have a water bottle
  have h5 : students_blue_eyes - (students_total - students_water_bottle) = 8, by {
    rw [h1, h4],
    norm_num,
  }
  exact h5

end least_possible_both_proof_l183_183448


namespace problem1_problem2_problem3_l183_183627

-- Problem 1
theorem problem1 : 
  2 * real.sqrt 3 * real.cbrt 1.5 * real.root 6 12 = 6 :=
sorry

-- Problem 2
theorem problem2 : 
  real.log10 (3 / 7) + real.log10 70 - real.log10 3 - real.sqrt ((real.log10 3)^2 - real.log10 9 + 1) = real.log10 3 :=
sorry

-- Problem 3
theorem problem3 (α : ℝ) (h : real.tan α = 2) : 
  4 * (real.sin α)^2 - 3 * real.sin α * real.cos α - 5 * (real.cos α)^2 = 1 :=
sorry

end problem1_problem2_problem3_l183_183627


namespace cube_edge_length_increase_l183_183812

-- Define the original edge length and volume of the cube
variables (a : ℝ)

-- Define the condition: original volume of the cube
def original_volume := a^3

-- Define the new volume, which is 8 times the original
def new_volume := 8 * original_volume a

-- Prove that the new edge length is 2 times the original
theorem cube_edge_length_increase : (∃ x : ℝ, x^3 = new_volume a) → ∃ b : ℝ, b = 2 * a :=
by
  intro hx
  use 2 * a
  calc
    (2 * a)^3 = 2^3 * a^3 : by rw [mul_pow]
          ... = 8 * a^3 : by norm_num
          ... = new_volume a : by rw [new_volume]
  sorry

end cube_edge_length_increase_l183_183812


namespace face_opposite_of_E_l183_183669

-- Definitions of faces and their relationships
inductive Face : Type
| A | B | C | D | E | F | x

open Face

-- Adjacency relationship
def is_adjacent_to (f1 f2 : Face) : Prop :=
(f1 = x ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = x ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D)) ∨
(f1 = E ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = E ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D))

-- Non-adjacency relationship
def is_opposite (f1 f2 : Face) : Prop :=
∀ f : Face, is_adjacent_to f1 f → ¬ is_adjacent_to f2 f

-- Theorem to prove that F is opposite of E
theorem face_opposite_of_E : is_opposite E F :=
sorry

end face_opposite_of_E_l183_183669


namespace average_of_distinct_numbers_l183_183696

theorem average_of_distinct_numbers (A B C D : ℕ) (hA : A = 1 ∨ A = 3 ∨ A = 5 ∨ A = 7)
                                   (hB : B = 1 ∨ B = 3 ∨ B = 5 ∨ B = 7)
                                   (hC : C = 1 ∨ C = 3 ∨ C = 5 ∨ C = 7)
                                   (hD : D = 1 ∨ D = 3 ∨ D = 5 ∨ D = 7)
                                   (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
    (A + B + C + D) / 4 = 4 := by
  sorry

end average_of_distinct_numbers_l183_183696


namespace zero_not_in_range_g_l183_183495

def g (x : ℝ) : ℤ :=
if x > -3 then
  Int.ceil (2 / (x + 3))
else
  Int.floor (2 / (x + 3))

theorem zero_not_in_range_g :
  ¬ ∃ x : ℝ, g x = 0 := 
sorry

end zero_not_in_range_g_l183_183495


namespace complex_sub_mul_l183_183146

variable (z1 z2 z3 : ℂ)
variable h1 : z1 = (4 - 3*complex.i)
variable h2 : z2 = (7 - 5*complex.i)
variable h3 : z3 = (1 + 2*complex.i)

theorem complex_sub_mul :
  ((z1 - z2) * z3) = (-7 - 4*complex.i) :=
by {
  rw [h1, h2, h3],
  -- Using the fact that complex numbers obey certain arithmetic rules
  have h_sub : z1 - z2 = (4 - 3*complex.i) - (7 - 5*complex.i), by rw [h1, h2],
  rw [h_sub],
  have h_mul : (4 - 3*complex.i) - (7 - 5*complex.i) = -3 + 2*complex.i, by sorry,
  rw [h_mul],
  have h_final : (-3 + 2*complex.i) * (1 + 2*complex.i) = -7 - 4*complex.i, by sorry,
  rw [h_final],
}

end complex_sub_mul_l183_183146


namespace simplify_expression_l183_183644

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183644


namespace angle_skew_lines_l183_183461

-- Definitions and conditions
variable (A B C D A1 B1 C1 D1 P : Point) (cube : Cube A B C D A1 B1 C1 D1) (P_on_AD1 : LineSegment A D1 P) (parallel_A1B_D1C : Parallel A1 B D1 C)

-- Theoretical angle
def theta (P : Point) :
  Angle (skewLines C P) (skewLines B A1)

-- Proof statement
theorem angle_skew_lines (theta : Angle) :
  0 < theta ∧ theta ≤ π / 3 :=
sorry

end angle_skew_lines_l183_183461


namespace cricket_matches_total_l183_183154

theorem cricket_matches_total
  (n : ℕ)
  (avg_all : ℝ)
  (avg_first4 : ℝ)
  (avg_last3 : ℝ)
  (h_avg_all : avg_all = 56)
  (h_avg_first4 : avg_first4 = 46)
  (h_avg_last3 : avg_last3 = 69.33333333333333)
  (h_total_runs : n * avg_all = 4 * avg_first4 + 3 * avg_last3) :
  n = 7 :=
by
  sorry

end cricket_matches_total_l183_183154


namespace volume_of_intersection_l183_183249

variable (V α : ℝ)
variable (hV : V > 0)
variable (hα : 0 < α ∧ α < π)

theorem volume_of_intersection (hV : V > 0) (hα : 0 < α ∧ α < π) : 
  volume_of_intersection V α = (1 + (Real.tan (α / 2))^2) / ((1 + Real.tan (α / 2))^2) * V :=
sorry

end volume_of_intersection_l183_183249


namespace bob_corn_stalks_per_row_l183_183282

noncomputable def corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  ℕ :=
  (bushels * stalks_per_bushel) / rows

theorem bob_corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  rows = 5 → bushels = 50 → stalks_per_bushel = 8 → corn_stalks_per_row rows bushels stalks_per_bushel = 80 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  unfold corn_stalks_per_row
  rfl

end bob_corn_stalks_per_row_l183_183282


namespace max_radius_four_spheres_l183_183066

theorem max_radius_four_spheres (R : ℝ) (r : ℝ) (hR : R = 2 + Real.sqrt 6) :
  r ≤ 2 :=
by
  have h1 : R ≥ (Real.sqrt 6 * r) / 2 + r,
  { rwa [hR] at *; sorry }

  sorry

end max_radius_four_spheres_l183_183066


namespace probability_committee_mixed_l183_183529

noncomputable def probability_of_mixed_committee : ℚ := 
let boys := 12 in
let girls := 18 in
let total := 30 in
let committee_size := 5 in
let all_boy_committees := Nat.choose boys committee_size in
let all_girl_committees := Nat.choose girls committee_size in
let total_committees := Nat.choose total committee_size in
1 - ((all_boy_committees + all_girl_committees) / total_committees : ℚ)

theorem probability_committee_mixed :
  probability_of_mixed_committee = 133146 / 142506 := 
sorry

end probability_committee_mixed_l183_183529


namespace combinatorial_identity_l183_183002

open Nat

theorem combinatorial_identity (n : ℕ) (h : 0 < n) : 
  (finset.sum (finset.range (n + 1)) 
    (λ k, nat.choose n k * 2 ^ k * nat.choose (n - k) (nat.floor ((n - k) / 2)))) = 
  nat.choose (2 * n + 1) n := 
begin
  sorry
end

end combinatorial_identity_l183_183002


namespace triangle_area_l183_183813

theorem triangle_area (a b c : ℝ) (B C : ℝ) (h_a : a = 6) (h_B : B = π/6) (h_C : C = 2*π/3) :
  let A := π/6 in
  let S := 1/2 * a * b * Real.sin C in
  S = 9 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l183_183813


namespace joe_possible_lists_l183_183589

theorem joe_possible_lists : 
  (number_of_balls : ℕ) (pick_and_replace_times : ℕ) 
  (number_of_balls = 15) ∧ (pick_and_replace_times = 4) →
  (possible_lists : ℕ) (possible_lists = number_of_balls ^ pick_and_replace_times) → 
  possible_lists = 15 ^ 4 := 
by 
  intros h1 h2 
  sorry

end joe_possible_lists_l183_183589


namespace jason_safe_combination_count_l183_183842

theorem jason_safe_combination_count : 
  let valid_combinations_count := 
    (1 * 3 * 3 * 3 * 3 * 3) +   -- Starting with 1
    (1 * 2 * 3 * 3 * 3 * 3) +   -- Starting with 3
    (1 * 1 * 3 * 3 * 3 * 3) +   -- Starting with 5
    (1 * 2 * 3 * 3 * 3 * 3) +   -- Starting with 2
    (1 * 1 * 3 * 3 * 3 * 3)     -- Starting with 4
  in 
  valid_combinations_count = 729 :=
by
  -- Placeholder 'sorry' to indicate that proof needs to be provided
  sorry

end jason_safe_combination_count_l183_183842


namespace min_area_right_triangle_l183_183205

/-- Minimum area of a right triangle with sides 6 and 8 units long. -/
theorem min_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8)
: 15.87 ≤ min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2))) :=
by
  sorry

end min_area_right_triangle_l183_183205


namespace perimeter_of_ABCD_l183_183668

-- Given conditions
variables (A B C D E : Type)
variables (AE : ℝ) (BE : ℝ) (CE : ℝ)
axiom right_angle_triangle1 : ABE ⊆ right_angle_triangle ⟨A, B, E⟩
axiom right_angle_triangle2 : BCE ⊆ right_angle_triangle ⟨B, C, E⟩
axiom right_angle_triangle3 : CDE ⊆ right_angle_triangle ⟨C, D, E⟩
axiom angle_AEB_45 : ∠AEB = 45
axiom angle_BEC_45 : ∠BEC = 45
axiom angle_CED_45 : ∠CED = 45
axiom AE_length : AE = 32

-- Prove the perimeter of quadrilateral ABCD
theorem perimeter_of_ABCD : ∃ (perimeter : ℝ), perimeter = 48 + 16 * Real.sqrt 2 := by
  sorry

end perimeter_of_ABCD_l183_183668


namespace piano_lesson_hours_l183_183471

theorem piano_lesson_hours :
  ∃ x : ℕ, 
    let clarinet_cost := 40 * 3 * 52 in
    let piano_cost := 28 * x * 52 in
    piano_cost = clarinet_cost + 1040 ∧ x = 5 :=
begin
  sorry
end

end piano_lesson_hours_l183_183471


namespace problem_l183_183739

noncomputable def hyperbola_equation (a b : ℝ) : Prop :=
  a = 1 ∧ b = sqrt 2 ∧ (∀ x y : ℝ, (x^2) - ((y^2)/(b^2)) = 1 ↔ (x, y) ∈ ({(x, y) | x^2 - y^2 / 2 = 1}))

noncomputable def perpendicular_points (x1 x2 y1 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem problem
  (x y m : ℝ)
  (h1 : ∀ x y : ℝ, (x^2) - ((y^2)/2) = 1)
  (h2 : ∀ x y m : ℝ, (x - y + m = 0))
  (h3 : perpendicular_points x1 x2 y1 y2) :
  m = 2 ∨ m = -2 :=
sorry

end problem_l183_183739


namespace area_of_T_l183_183487

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![[3, 4], [6, -2]]

theorem area_of_T' (area_T : ℝ) (h1 : area_T = 8) :
    let det := (3 * (-2) - 4 * 6 : ℝ)
    let scaling_factor := abs det
    let area_T' := scaling_factor * area_T
    area_T' = 240 :=
by
  let det : ℝ := 3 * (-2) - 4 * 6
  let scaling_factor : ℝ := abs det
  let area_T' : ℝ := scaling_factor * area_T
  have h2 : det = -30 := by norm_num
  have h3 : scaling_factor = 30 := by simp [h2]
  have h4 : area_T' = 30 * 8 := by simp [h3, h1]
  norm_num at h4
  exact h4

end area_of_T_l183_183487


namespace inequality_selection_l183_183229

theorem inequality_selection (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) 
  (h₃ : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 ∧ (∀ x, |x + a| + |x - b| + c = 4 → x = (a - b)/2) ∧ (a = 8 / 7 ∧ b = 18 / 7 ∧ c = 2 / 7) :=
by
  sorry

end inequality_selection_l183_183229


namespace number_is_16_l183_183217

theorem number_is_16 (n : ℝ) (h : (1/2) * n + 5 = 13) : n = 16 :=
sorry

end number_is_16_l183_183217


namespace total_combined_grapes_l183_183264

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l183_183264


namespace range_of_m_intersection_l183_183751

noncomputable def range_of_m (m : ℝ) : Prop :=
  (m ≠ 0) ∧ 
  ∀ (A B : ℝ × ℝ), 
    A = (1, 2) ∧ B = (2, 11) →
    ((m - 6/m - 2 + 1) * (2*(m - 6/m) - 11 + 1) ≤ 0) →
    (m ∈ set.Icc (-2) (-1) ∨ m ∈ set.Icc 3 6)

theorem range_of_m_intersection :
  range_of_m m :=
sorry

end range_of_m_intersection_l183_183751


namespace find_p6_l183_183859

noncomputable def p (x : ℕ) : ℕ := sorry

axiom p_conditions :
  p 1 = 3 ∧
  p 2 = 7 ∧
  p 3 = 13 ∧
  p 4 = 21 ∧
  p 5 = 31

theorem find_p6 (h : p_conditions) : p 6 = 158 :=
sorry

end find_p6_l183_183859


namespace x_plus_y_value_l183_183008

theorem x_plus_y_value (x y : ℕ) (h1 : 2^x = 8^(y + 1)) (h2 : 9^y = 3^(x - 9)) : x + y = 27 :=
by
  sorry

end x_plus_y_value_l183_183008


namespace parallelogram_area_l183_183117

def v : ℝ × ℝ := (7, -5)
def w : ℝ × ℝ := (13, -4)

theorem parallelogram_area :
  let matrix := ![![7,13],![-5,-4]] in
  matrix.det.abs = 37 :=
by
  sorry

end parallelogram_area_l183_183117


namespace exists_set_satisfying_properties_l183_183316

theorem exists_set_satisfying_properties :
  ∃ (M : Finset ℕ),
    (M.card = 1992) ∧
    (∀ x ∈ M, ∃ m k : ℕ, m > 0 ∧ k ≥ 2 ∧ x = m ^ k) ∧
    (∀ S ⊆ M, ∃ m k : ℕ, m > 0 ∧ k ≥ 2 ∧ ∑ x in S, x = m ^ k) :=
sorry

end exists_set_satisfying_properties_l183_183316


namespace julia_age_correct_l183_183480

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end julia_age_correct_l183_183480


namespace propositions_incorrect_l183_183387

-- Assume the definitions of planes and lines in a suitable way
noncomputable theory
open_locale classical

-- Definitions for planes and lines
variables (α β : Type*) [Plane α] [Plane β] (a b : Type*) [Line a] [Line b]

-- Define sets
def A := {γ : Type* // Plane γ ∧ perpendicular γ α}
def B := {γ : Type* // Plane γ ∧ perpendicular γ β}
def M := {c : Type* // Line c ∧ perpendicular c a}
def N := {c : Type* // Line c ∧ perpendicular c b}

-- Lean statement to check the propositions
theorem propositions_incorrect :
  (A ∩ B ≠ ∅ → ¬parallel α β) ∧
  (parallel α β → ∀ (x : Type*) (hx : Plane x ∧ perpendicular x α), hb : Plane x ∧ perpendicular x β → x ∈ B) ∧
  (skew a b → ¬(M ∩ N = ∅)) ∧
  (intersect a b → ∀ (c : Type*) (hc : Line c ∧ perpendicular c a), ¬(perpendicular c b ∧ c ∈ N)) :=
sorry

end propositions_incorrect_l183_183387


namespace concyclic_points_l183_183554

variables (A B C D E X : Type)
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] [InnerProductSpace ℝ X]

-- Given conditions
variables (pABCD : Parallelogram A B C D) (pE_on_AB : E ∈ Line A B) (ordered_ABC : Collinear ℝ [A, B, E])
variables (BC_eq_BE : dist B C = dist B E)
variables (A_perp_CE : Perpendicular (line_through A) (line_through C E)) (X_intersection : X = intersection (perpendicular_bisector A E) (perpendicular_from A C E))

-- Prove
theorem concyclic_points : Concyclic A B D X := sorry

end concyclic_points_l183_183554


namespace relationship_y123_l183_183388

variable (A B C : ℝ) (m y_1 y_2 y_3 : ℝ)

noncomputable def y1_value := 15 + m
noncomputable def y2_value := 3 + m
noncomputable def y3_value := m

theorem relationship_y123 (m : ℝ) :
  y3_value m < y2_value m ∧ y2_value m < y1_value m :=
by {
  unfold y3_value y2_value y1_value,
  exact ⟨by linarith, by linarith⟩,
}

end relationship_y123_l183_183388


namespace polynomial_P_10_val_l183_183862

statement : Prop :=
  ∃ (a b c : ℝ),
    let P : ℝ → ℝ := λ x, a * x^2 + b * x + c in
    P(1) = 20 ∧ P(-1) = 22 ∧ P(P(0)) = 400 ∧ P(10) = 1595

theorem polynomial_P_10_val : statement :=
  sorry

end polynomial_P_10_val_l183_183862


namespace john_vegetables_l183_183474

theorem john_vegetables (beef_used vege_used : ℕ) :
  beef_used = 4 - 1 →
  vege_used = 2 * beef_used →
  vege_used = 6 :=
by
  intros h_beef_used h_vege_used
  unfold beef_used vege_used
  exact sorry

end john_vegetables_l183_183474


namespace net_salary_change_l183_183955

theorem net_salary_change (S : ℝ) :
  let salary_after_first_increase := S * 1.25,
      salary_after_first_reduction := salary_after_first_increase * 0.85,
      salary_after_second_increase := salary_after_first_reduction * 1.10,
      final_salary := salary_after_second_increase * 0.80
  in final_salary = S * 0.935 :=
by
  sorry

end net_salary_change_l183_183955


namespace cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l183_183058

-- Definitions for geometric objects
def cube : Type := sorry
def regular_tetrahedron : Type := sorry

-- Definitions for axes of symmetry
def axes_of_symmetry (shape : Type) : Nat := sorry

-- Theorem statements
theorem cube_axes_of_symmetry : axes_of_symmetry cube = 13 := 
by 
  sorry

theorem regular_tetrahedron_axes_of_symmetry : axes_of_symmetry regular_tetrahedron = 7 :=
by 
  sorry

end cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l183_183058


namespace average_speed_last_segment_l183_183099

-- Define the conditions from the problem
def total_distance : ℝ := 120
def total_duration : ℝ := 2
def first_segment_time : ℝ := 45 / 60
def second_segment_time : ℝ := 45 / 60
def last_segment_time : ℝ := 30 / 60
def first_segment_speed : ℝ := 50
def second_segment_speed : ℝ := 60

-- Define the theorem to prove the average speed during the last segment

theorem average_speed_last_segment : 
  (120 - (first_segment_speed * first_segment_time + second_segment_speed * second_segment_time)) / last_segment_time = 75 := 
by
  sorry

end average_speed_last_segment_l183_183099


namespace checkerboard_black_squares_435_l183_183516

theorem checkerboard_black_squares_435 :
  ∃ (n : ℕ), n = 29 ∧
    (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
      (i + j) % 2 = 0 →
      black i j) ∧
    (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
      (i + j) % 2 = 1 →
      red i j) ∧
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ n →
      ∀ (c : ℕ), (black_row_count : ℕ) = (n / 2 + 1) ∧  -- counting black squares in each row, rounded up
      (total_black_squares : ℕ) = black_row_count * n)
  ∧ total_black_squares = 435 :=
begin
  sorry
end

end checkerboard_black_squares_435_l183_183516


namespace least_positive_angle_is_75_l183_183710

noncomputable def least_positive_angle (θ : ℝ) : Prop :=
  cos (10 * Real.pi / 180) = sin (15 * Real.pi / 180) + sin θ

theorem least_positive_angle_is_75 :
  least_positive_angle (75 * Real.pi / 180) :=
by
  sorry

end least_positive_angle_is_75_l183_183710


namespace incorrect_statement_C_l183_183727

theorem incorrect_statement_C (k : ℝ) (h_k : -1/4 ≤ k ∧ k < 0) :
  ¬(∃ x1 x2 x3 x4 : ℝ, (abs ((1 / 2)^(abs x1) - 1 / 2))^2 - abs ((1 / 2)^(abs x1) - 1 / 2) - k = 0 ∧ 
                         (abs ((1 / 2)^(abs x2) - 1 / 2))^2 - abs ((1 / 2)^(abs x2) - 1 / 2) - k = 0 ∧
                         (abs ((1 / 2)^(abs x3) - 1 / 2))^2 - abs ((1 / 2)^(abs x3) - 1 / 2) - k = 0 ∧
                         (abs ((1 / 2)^(abs x4) - 1 / 2))^2 - abs ((1 / 2)^(abs x4) - 1 / 2) - k = 0) :=
by {
  sorry
}

end incorrect_statement_C_l183_183727


namespace asian_games_discount_equation_l183_183319

variable (a : ℝ)

theorem asian_games_discount_equation :
  168 * (1 - a / 100)^2 = 128 :=
sorry

end asian_games_discount_equation_l183_183319


namespace intersection_A_B_range_m_l183_183292

def f (x : ℝ) : ℝ := log (x^2 - x - 2)
def g (x : ℝ) : ℝ := sqrt (3 - abs x)

def A : set ℝ := {x | x^2 - x - 2 > 0}
def B : set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

theorem intersection_A_B  : A ∩ B = {x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3)} :=
by
  sorry

def C (m : ℝ) : set ℝ := {x | m - 1 < x ∧ x < m + 2}

theorem range_m (m : ℝ) : C m ⊆ B ↔ -2 ≤ m ∧ m ≤ 1 :=
by
  sorry

end intersection_A_B_range_m_l183_183292


namespace last_step_erased_numbers_l183_183168

/-- The sequence of numbers 1, 2, ..., 100 with a process of erasing numbers in steps
such that numbers with no divisors among the remaining numbers, except themselves,
are erased. Prove that 64 and 96 are erased in the last step. -/
theorem last_step_erased_numbers :
  let initial_numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 100}
  ∃ (step : ℕ → finset ℕ),
    step 0 = initial_numbers ∧
    (∀ n, step (n + 1) = step n \ {k ∈ step n | ∀ d ∈ step n \ {k}, d ∣ k}) ∧
    ∀ k, k ∈ step 6 → k = 64 ∨ k = 96 :=
by sorry

end last_step_erased_numbers_l183_183168


namespace number_of_factors_2_6_7_3_3_4_l183_183676

theorem number_of_factors_2_6_7_3_3_4 : 
  let n := (2^6) * (7^3) * (3^4) in 
  ∃ num_factors : ℕ, num_factors = 140 ∧ 
    (∀ a b c : ℕ, (0 ≤ a ∧ a ≤ 6) ∧ (0 ≤ b ∧ b ≤ 3) ∧ (0 ≤ c ∧ c ≤ 4)) → 
      (2^a) * (7^b) * (3^c) ∣ n := 
begin
  let n := (2^6) * (7^3) * (3^4),
  use 140,
  split,
  { exact 140 },
  { intros a b c ha hb hc,
    sorry
  }
end

end number_of_factors_2_6_7_3_3_4_l183_183676


namespace y_intercept_of_line_l183_183545

theorem y_intercept_of_line (x y : ℝ) (h : x + 2 * y - 1 = 0) (hx : x = 0) : y = 1 / 2 :=
by
  calc
    x + 2 * y - 1 = 0 : h
    0 + 2 * y - 1 = 0 : by rw [hx]
    2 * y - 1 = 0 := by simp [hx]
    2 * y = 1 := by linarith
    y = 1 / 2 := by linarith

end y_intercept_of_line_l183_183545


namespace ratio_sum_2_or_4_l183_183118

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end ratio_sum_2_or_4_l183_183118


namespace question1_question2_l183_183740

variable (a b c : ℝ)
variable (f : ℝ → ℝ) 
variable (A B C D : ℝ)

-- Conditions
def conditions : Prop :=
  c > b ∧ b > a ∧ ∀ x, f x = a*x^2 + 2*b*x + c ∧ f 1 = 0 ∧ ∃ x₀, f x₀ = -a

-- Questions
-- 1. Prove \(0 \leq \frac{b}{a} < 1\)
theorem question1 (h : conditions a b c f) : 0 ≤ b / a ∧ b / a < 1 :=
sorry

-- 2. Prove the range \(-1 + \frac{4}{21} < \frac{b}{a} < -1 + \frac{\sqrt{15}}{3}\)
theorem question2 (h : conditions a b c f) 
    (ha : y = |f(x)| ∧ ∀ i ∈ {A B C D}, f i = y) 
    (obtuse_triangle : ∃A B C D, segment(A,B) ∧ segment(B,C) ∧ segment(C,D) ∧ triangle_obtuse (segment(A,B)) (segment(B,C)) (segment(C,D))): 
    -1 + 4/21 < b / a ∧ b / a < -1 + sqrt(15) / 3 :=
sorry

end question1_question2_l183_183740


namespace seats_not_occupied_l183_183326

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l183_183326


namespace min_value_squared_l183_183511

noncomputable def z : ℂ := sorry  -- you can define it explicitly if needed

namespace ComplexPlaneProblem

theorem min_value_squared
  (h_area : let z : ℂ in (0, z, 1/z, z - 1/z) forms_parallelogram ∧ (parallelogram_area 0 z (1/z) (z - 1/z) = 15 / 17))
  (h_real_part : let z : ℂ in (z.re > 0)) : 
  ∃ d: ℝ, (d^2 = 176 / 34) ∧ (d = min_value (abs (z - 1/z))) := 
sorry

end ComplexPlaneProblem

end min_value_squared_l183_183511


namespace length_WZ_of_right_triangle_l183_183227

theorem length_WZ_of_right_triangle (XY XZ : ℝ) (WZ : ℝ) (hXY : XY = 45) (hXZ : XZ = 120)
  (hYZ : YZ = Real.sqrt (45^2 + 120^2))
  (hArea1 : ½ * XY * XZ = 2700)
  (hWX_hArea2 : ½ * hYZ * WX = 2700)
  (hWX : WX = 40)
  (hWZ_square : WZ^2 = 120^2 - 40^2) :
  WZ = 80 * Real.sqrt 2 := 
begin
  sorry
end

end length_WZ_of_right_triangle_l183_183227


namespace probability_win_more_than_5000_l183_183581

def boxes : Finset ℕ := {5, 500, 5000}
def keys : Finset (Finset ℕ) := { {5}, {500}, {5000} }

noncomputable def probability_correct_key (box : ℕ) : ℚ :=
  if box = 5000 then 1 / 3 else if box = 500 then 1 / 2 else 1

theorem probability_win_more_than_5000 :
    (probability_correct_key 5000) * (probability_correct_key 500) = 1 / 6 :=
by
  -- Proof is omitted
  sorry

end probability_win_more_than_5000_l183_183581


namespace no_six_consecutive_nat_num_sum_eq_2015_l183_183093

theorem no_six_consecutive_nat_num_sum_eq_2015 :
  ∀ (a b c d e f : ℕ),
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e + 1 = f →
  a * b * c + d * e * f ≠ 2015 :=
by
  intros a b c d e f h
  sorry

end no_six_consecutive_nat_num_sum_eq_2015_l183_183093


namespace square_area_triangle_ADC_l183_183557

-- Conditions:
def radius_small_circle := (2 : ℝ)
def radius_large_circle := (4 : ℝ)
def center := (0 : ℝ, 0 : ℝ)
def point_A := (4, 0)  -- Assume an equilateral triangle starting at 0 degrees
def point_B := (2, 2 * Real.sqrt 3) -- 120 degrees rotation
def point_C := (-2, 2 * Real.sqrt 3) -- 240 degrees rotation
def point_D := (2, 0) -- Intersection of PB with the small circle

-- Goal:
theorem square_area_triangle_ADC : 
  let area_ADC := (Real.sqrt 3 * ((point_A.1 - point_D.1) * (point_C.2 - point_D.2) 
                        - (point_A.2 - point_D.2) * (point_C.1 - point_D.1))) / 4 in
  (area_ADC ^ 2) = 192 :=
sorry

end square_area_triangle_ADC_l183_183557


namespace book_area_l183_183232

theorem book_area (length width : ℕ) (h1 : length = 2) (h2 : width = 3) : length * width = 6 :=
by 
  rw [h1, h2]
  norm_num

end book_area_l183_183232


namespace variable_v_value_l183_183063

theorem variable_v_value (w x v : ℝ) (h1 : 2 / w + 2 / x = 2 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) :
  v = 0.25 :=
sorry

end variable_v_value_l183_183063


namespace total_games_for_18_players_l183_183231

-- Define the number of players
def num_players : ℕ := 18

-- Define the function to calculate total number of games
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Theorem statement asserting the total number of games for 18 players
theorem total_games_for_18_players : total_games num_players = 612 :=
by
  -- proof goes here
  sorry

end total_games_for_18_players_l183_183231


namespace candy_left_l183_183472

variable (x : ℕ)

theorem candy_left (x : ℕ) : x - (18 + 7) = x - 25 :=
by sorry

end candy_left_l183_183472


namespace expression_simplification_l183_183635

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183635


namespace min_people_hat_glove_not_scarf_l183_183453

theorem min_people_hat_glove_not_scarf (n : ℕ) 
  (h_gloves : n * 3 / 8)
  (h_hats : n * 5 / 6)
  (h_scarves : n * 1 / 4) :
  n = 24 →
  ∃ x, x = 11 ∧ 
  x = h_gloves + h_hats - (n - h_scarves) :=
begin
  sorry
end

end min_people_hat_glove_not_scarf_l183_183453


namespace solve_equation_l183_183916

theorem solve_equation (x : ℝ) (h : (x - 60) / 3 = (4 - 3 * x) / 6) : x = 124 / 5 := by
  sorry

end solve_equation_l183_183916


namespace vector_MN_l183_183383

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]

-- Define the points in space as vectors.
variables (O A B C M N : V)
variables (a b c : V)

-- Hypotheses
noncomputable def O_a : O + a = A := sorry
noncomputable def O_b : O + b = B := sorry
noncomputable def O_c : O + c = C := sorry
noncomputable def M_O_A : M = O + (2 / 3) • a := sorry
noncomputable def N_b_c : N = (1 / 2) • (B + C) := sorry

-- The proof problem statement
theorem vector_MN :
  ∃ (MN : V), MN = - (2 / 3) • a + (1 / 2) • b + (1 / 2) • c :=
sorry

end vector_MN_l183_183383


namespace distance_to_asymptote_l183_183924

noncomputable def distance_from_asymptote : ℝ :=
  let x0 := 3
  let y0 := 0
  let A := 3
  let B := -4
  let C := 0
  @Real.sqrt ((A^2) + (B^2))^{-1} * abs (A * x0 + B * y0 + C)

theorem distance_to_asymptote : distance_from_asymptote = (9 / 5) := sorry

end distance_to_asymptote_l183_183924


namespace mixed_groups_count_l183_183991

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183991


namespace sqrt_div_simplify_l183_183333

def squared_expr_eq_fraction (p q : ℝ) : Prop :=
  ( ( (1 / 3) ^ 2 + (1 / 4) ^ 2) / ((1 / 5) ^ 2 + (1 / 6) ^ 2) ) = 25 * p / (61 * q)

theorem sqrt_div_simplify (p q : ℝ) (h : squared_expr_eq_fraction p q) : 
  (sqrt p) / (sqrt q) = 5 / 2 :=
  sorry

end sqrt_div_simplify_l183_183333


namespace general_equation_of_line_l183_183396

theorem general_equation_of_line (l : ℝ → ℝ)
    (h_passes_point : l 1 = 0)
    (h_slope : ∃ k, k = Real.tan (π / 3) ∧ ∀ x, l x = k * (x - 1)) :
    ∃ a b c : ℝ, a = √3 ∧ b = -1 ∧ c = -√3 ∧ ∀ x y, y = l x → √3 * x - y - √3 = 0 :=
by
  sorry

end general_equation_of_line_l183_183396


namespace cost_of_candy_bar_l183_183671

theorem cost_of_candy_bar (initial_amount remaining_amount cost : ℝ)
  (h_initial : initial_amount = 5)
  (h_remaining : remaining_amount = 3) :
  cost = initial_amount - remaining_amount :=
by {
  rw [h_initial, h_remaining],
  exact rfl,
}

end cost_of_candy_bar_l183_183671


namespace simplify_expression_eq_square_l183_183632

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183632


namespace terminating_decimals_count_l183_183726

theorem terminating_decimals_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 29 ∧ (∀ (d : ℕ), (30 = d * n → (∀ (p : ℕ), p.prime → p ∣ d → p = 2 ∨ p = 5)))}.card = 20 :=
by sorry

end terminating_decimals_count_l183_183726


namespace fourth_proportional_segment_l183_183790

theorem fourth_proportional_segment 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  : ∃ x : ℝ, x = (b * c) / a := 
by
  sorry

end fourth_proportional_segment_l183_183790


namespace derivative_at_2_l183_183774

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_2 : (deriv f 2) = -120 :=
by
  sorry

end derivative_at_2_l183_183774


namespace find_D_plus_E_plus_F_l183_183534

noncomputable def g (x : ℝ) (D E F : ℝ) : ℝ := (x^2) / (D * x^2 + E * x + F)

theorem find_D_plus_E_plus_F (D E F : ℤ) 
  (h1 : ∀ x : ℝ, x > 3 → g x D E F > 0.3)
  (h2 : ∀ x : ℝ, ¬(D * x^2 + E * x + F = 0 ↔ (x = -3 ∨ x = 2))) :
  D + E + F = -8 :=
sorry

end find_D_plus_E_plus_F_l183_183534


namespace stormi_charge_per_car_l183_183524

-- Define the relevant conditions and variables
variables (x : ℕ) -- Amount Stormi charges for washing each car
variables (cars_washed lawns_mowed : ℕ) (lawn_charge bike_cost additional_amount savings_from_mowing : ℕ)

-- Assign the values to the conditions
def stormi_conditions := cars_washed = 3 ∧ 
                        lawns_mowed = 2 ∧ 
                        lawn_charge = 13 ∧ 
                        bike_cost = 80 ∧ 
                        additional_amount = 24 ∧ 
                        savings_from_mowing = 2 * 13

-- Prove the amount Stormi charges for washing each car, given the conditions
theorem stormi_charge_per_car (h : stormi_conditions):
  x = 10 :=
begin
  sorry
end

end stormi_charge_per_car_l183_183524


namespace range_of_k_l183_183570

-- Define the function f(x) = e^x
def f (x : ℝ) : ℝ := Real.exp x

-- Define the line y = kx + 1
def line (k x : ℝ) : ℝ := k * x + 1

-- The theorem stating the equivalence
theorem range_of_k (k : ℝ) :
  (∀ x > 0, f x > line k x) → k ≤ 1 :=
by
  sorry

end range_of_k_l183_183570


namespace total_grapes_l183_183268

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l183_183268


namespace distance_from_point_to_hyperbola_asymptote_l183_183930

noncomputable def distance_to_asymptote (x1 y1 a b : ℝ) : ℝ :=
  abs (a * x1 + b * y1) / real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_hyperbola_asymptote :
  distance_to_asymptote 3 0 3 (-4) = 9 / 5 :=
by
  sorry

end distance_from_point_to_hyperbola_asymptote_l183_183930


namespace probability_theta_in_first_quadrant_l183_183144

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := real.sqrt (a.1^2 + a.2^2)
  let norm_b := real.sqrt (b.1^2 + b.2^2)
  real.arccos (dot_product / (norm_a * norm_b))

def is_in_first_quadrant_or_on_boundary (θ : ℝ) : Prop :=
  0 < θ ∧ θ ≤ real.pi / 2

def number_of_favorable_outcomes : ℕ := 
  6 + 5 + 4 + 3 + 2 + 1

theorem probability_theta_in_first_quadrant :
  let total_outcomes := 36
  let favorable_outcomes := number_of_favorable_outcomes
  let probability := favorable_outcomes / total_outcomes in
  probability = (7 / 12 : ℝ) := 
by
  sorry

end probability_theta_in_first_quadrant_l183_183144


namespace incorrect_options_in_ball_draws_l183_183076

/-- Proving the incorrect options in a probability scenario with ball draws -/
theorem incorrect_options_in_ball_draws :
  let balls : set ℕ := {1, 2, 3, 4},
      draw_with_replacement : list (ℕ × ℕ) := [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4),
                                                 (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)],
      draw_without_replacement : list (ℕ × ℕ) := [(1,2), (1,3), (1,4), (2,1), (2,3), (2,4),
                                                     (3,1), (3,2), (3,4), (4,1), (4,2), (4,3)],
      event_A : (ℕ × ℕ) → Prop := λ draw, draw.fst < 3,
      event_B : (ℕ × ℕ) → Prop := λ draw, draw.snd < 3,
      complement_event_B : (ℕ × ℕ) → Prop := λ draw, draw.snd ≥ 3,
      P := λ (l : list (ℕ × ℕ)) (f : (ℕ × ℕ) → Prop),
           (l.filter f).length / l.length in
  ¬ (∃ (option : string), option = "A") ∨
  ¬ (∃ (option : string), option = "C") ∨
  ¬ (∃ (option : string), option = "D")
:= sorry

end incorrect_options_in_ball_draws_l183_183076


namespace mass_order_l183_183212

variable {a b c d : ℝ}

theorem mass_order (h1: a + b = c + d) (h2: a + d > b + c) (h3: b > a + c) :
    d > b ∧ b > a ∧ a > c :=
begin
  sorry
end

end mass_order_l183_183212


namespace unoccupied_seats_l183_183323

theorem unoccupied_seats 
    (seats_per_row : ℕ) 
    (rows : ℕ) 
    (seatable_fraction : ℚ) 
    (total_seats := seats_per_row * rows) 
    (seatable_seats_per_row := (seatable_fraction * seats_per_row)) 
    (seatable_seats := seatable_seats_per_row * rows) 
    (unoccupied_seats := total_seats - seatable_seats) {
  seats_per_row = 8, 
  rows = 12, 
  seatable_fraction = 3/4 
  : unoccupied_seats = 24 :=
by
  sorry

end unoccupied_seats_l183_183323


namespace complex_sum_correct_l183_183122

-- Let's define the conditions first
variables {z : ℂ} (hz : abs z = 1)

-- The quadratic equation zx^2 + 2zx + 2 = 0 has a real root
variable (has_real_root : ∃ x : ℝ, z * x^2 + 2 * z * x + 2 = 0)

-- Sum of all such complex numbers z
def sum_of_complex_numbers_eq : Prop :=
  ∑ z in {z : ℂ | abs z = 1 ∧ (∃ x : ℝ, z * x^2 + 2 * z * x + 2 = 0)}, z = -3/2

-- Now we state the theorem
theorem complex_sum_correct (hz : abs z = 1) (has_real_root : ∃ x : ℝ, z * x^2 + 2 * z * x + 2 = 0) :
  sum_of_complex_numbers_eq :=
sorry

end complex_sum_correct_l183_183122


namespace tires_bought_l183_183149

variable (cost_per_tire total_cost : ℕ)
variable (h1 : cost_per_tire = 60)
variable (h2 : total_cost = 240)

theorem tires_bought : (total_cost / cost_per_tire) = 4 :=
by
  rw [h1, h2]
  sorry

end tires_bought_l183_183149


namespace complex_solution_l183_183070

theorem complex_solution (a : ℝ) (h : (a + complex.I) * (1 - a * complex.I) = 2) : a = 1 :=
by 
  sorry

end complex_solution_l183_183070


namespace common_terms_sum_l183_183350

noncomputable def arithmetic_progression (n : ℕ) : ℕ :=
  5 + 3 * n

noncomputable def geometric_progression (k : ℕ) : ℕ :=
  20 * 2^k

theorem common_terms_sum : ∑ i in (Finset.range 10), (arithmetic_progression i) = 6990500 :=
sorry

end common_terms_sum_l183_183350


namespace part1_monotonicity_for_a_eq1_part2_inequality_for_a_eq_minus3_l183_183411

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - a * Real.sin x - 1

theorem part1_monotonicity_for_a_eq1 :
  ∀ x ∈ Ioo (-π / 2) π, x < 0 → 
  let g := λ x, (Real.exp x - Real.sin x - 1) / Real.exp x in g x < g (x + 1) :=
sorry

theorem part2_inequality_for_a_eq_minus3 :
  ∀ x ∈ Ioo 0 (1 / 0), 
  f x (-3) < Real.exp x + x + 1 - 2 * Real.exp (-2 * x) :=
sorry

end part1_monotonicity_for_a_eq1_part2_inequality_for_a_eq_minus3_l183_183411


namespace subset_A_B_def_l183_183419

variable (A : Set ℝ) (B : Set ℝ)
variable (a : ℝ)

def A_def : Set ℝ := { -1, 1 }
def B_def (a : ℝ) : Set ℝ := { x | a * x + 1 = 0 }

theorem subset_A_B_def :
  (B ⊆ A) ↔ (a ∈ { -1, 1 }) :=
by
  sorry

end subset_A_B_def_l183_183419


namespace early_arrival_l183_183689
variable (T H : ℝ) -- The usual time man and wife arrive at the station and reach home.
variable (E : ℝ) -- The early arrival time.
variable (W S : ℝ) -- Walking and driving speeds.
variable (diff : ℝ) -- The difference in the arrival times which is 30 minutes as given.

theorem early_arrival (usual_time_to_home : H - T) 
                      (early : E = T - 30) 
                      (early_diff : H - T - (T - E) = H - T - 30)
                      : E = 30 :=
sorry

end early_arrival_l183_183689


namespace identify_quadratic_equation_l183_183616

-- Definitions of the equations
def eqA : Prop := ∀ x : ℝ, x^2 + 1/x^2 = 4
def eqB : Prop := ∀ (a b x : ℝ), a*x^2 + b*x - 3 = 0
def eqC : Prop := ∀ x : ℝ, (x - 1)*(x + 2) = 1
def eqD : Prop := ∀ (x y : ℝ), 3*x^2 - 2*x*y - 5*y^2 = 0

-- Definition that identifies whether a given equation is a quadratic equation in one variable
def isQuadraticInOneVariable (eq : Prop) : Prop := 
  ∃ (a b c : ℝ) (a0 : a ≠ 0), ∀ x : ℝ, eq = (a * x^2 + b * x + c = 0)

theorem identify_quadratic_equation :
  isQuadraticInOneVariable eqC :=
by
  sorry

end identify_quadratic_equation_l183_183616


namespace relative_error_comparison_l183_183272

-- Define the errors and the lengths of the lines
def error_first : ℝ := 0.05
def length_first : ℝ := 50
def error_second : ℝ := 0.4
def length_second : ℝ := 200

-- Define the relative errors
def relative_error_first : ℝ := error_first / length_first
def relative_error_second : ℝ := error_second / length_second

-- Prove the comparison between the two relative errors
theorem relative_error_comparison : relative_error_second > relative_error_first :=
by
  sorry

end relative_error_comparison_l183_183272


namespace usual_time_is_3_l183_183255

noncomputable def train_usual_time (T : ℕ) :=
  (5 / 4 * T = T + 3 / 4) → T = 3

theorem usual_time_is_3 (T : ℕ) : train_usual_time T :=
by
  assume h: (5 / 4 * T = T + 3 / 4)
  sorry

end usual_time_is_3_l183_183255


namespace sale_in_fifth_month_l183_183601

theorem sale_in_fifth_month (Sale1 Sale2 Sale3 Sale4 Sale6 AvgSale : ℤ) 
(h1 : Sale1 = 6435) (h2 : Sale2 = 6927) (h3 : Sale3 = 6855) (h4 : Sale4 = 7230) 
(h5 : Sale6 = 4991) (h6 : AvgSale = 6500) : (39000 - (Sale1 + Sale2 + Sale3 + Sale4 + Sale6)) = 6562 :=
by
  sorry

end sale_in_fifth_month_l183_183601


namespace cube_face_sum_l183_183522

theorem cube_face_sum (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) :
  (a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1287) →
  (a + d + b + e + c + f = 33) :=
by
  sorry

end cube_face_sum_l183_183522


namespace biē_nào_surface_area_sum_l183_183829

-- Define the tetrahedron and its properties
structure Tetrahedron (V : Type) [EuclideanSpace V] :=
(M A B C : V)
(MA_perpendicular_ABC : MA ⊥ plane (A, B, C))
(MA_length : ∥M - A∥ = 2)
(AB_length : ∥A - B∥ = 2)
(BC_length : ∥B - C∥ = 2)

noncomputable def sum_surface_areas {V : Type} [EuclideanSpace V] (t : Tetrahedron V) : ℝ :=
  let circumscribed_radius := sqrt 3    -- Radius of the circumscribed sphere
  let circumscribed_area := 4 * π * (circumscribed_radius ^ 2)
  let inscribed_radius := sqrt 2 - 1    -- Radius of the inscribed sphere
  let inscribed_area := 4 * π * (inscribed_radius ^ 2)
  circumscribed_area + inscribed_area

theorem biē_nào_surface_area_sum (t : Tetrahedron ℝ) :
  sum_surface_areas t = 24 * π - 8 * sqrt 2 * π :=
sorry

end biē_nào_surface_area_sum_l183_183829


namespace initial_distance_l183_183330

theorem initial_distance (speed_enrique speed_jamal : ℝ) (hours : ℝ) 
  (h_enrique : speed_enrique = 16) 
  (h_jamal : speed_jamal = 23) 
  (h_time : hours = 8) 
  (h_difference : speed_jamal = speed_enrique + 7) : 
  (speed_enrique * hours + speed_jamal * hours = 312) :=
by 
  sorry

end initial_distance_l183_183330


namespace reciprocal_check_C_l183_183572

theorem reciprocal_check_C : 0.1 * 10 = 1 := 
by 
  sorry

end reciprocal_check_C_l183_183572


namespace parabola_tangent_line_l183_183486

noncomputable def gcd (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem parabola_tangent_line (a b c : ℕ) (h1 : a^2 + (104 / 5) * a * b - 4 * b * c = 0)
  (h2 : b^2 - 5 * a^2 + 4 * a * c = 0) (hgcd : gcd a b c = 1) :
  a + b + c = 17 := by
  sorry

end parabola_tangent_line_l183_183486


namespace area_of_triangle_ABC_l183_183309

def point := (ℝ, ℝ)

noncomputable def area_of_triangle (A B C : point) : ℝ := 
    let base := (C.2 - A.2).abs in
    let height := (B.1 - A.1).abs in
    (1 / 2) * base * height

theorem area_of_triangle_ABC : area_of_triangle (0, 0) (1, 7) (0, 8) = 28 := 
by 
  sorry

end area_of_triangle_ABC_l183_183309


namespace rectangle_solution_l183_183563

theorem rectangle_solution (x : ℝ)
  (h_dims : (x - 3) * (3x + 4) = 12x - 9) :
  x = (17 + 5 * Real.sqrt 13) / 6 :=
sorry

end rectangle_solution_l183_183563


namespace product_of_solutions_l183_183345

theorem product_of_solutions :
  (∃ x y : ℝ, (|x^2 - 6 * x| + 5 = 41) ∧ (|y^2 - 6 * y| + 5 = 41) ∧ x ≠ y ∧ x * y = -36) :=
by
  sorry

end product_of_solutions_l183_183345


namespace minimum_value_l183_183028

theorem minimum_value 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : ab - a - 2 * b = 0) : 
  (a^2 / 4 - 2 / a + b^2 - 1 / b) ≥ 7 := 
begin
  sorry
end

end minimum_value_l183_183028


namespace exponent_property_l183_183543

theorem exponent_property : (-2)^2004 + 3 * (-2)^2003 = -2^2003 :=
by 
  sorry

end exponent_property_l183_183543


namespace arithmetic_geometric_seq_sum_5_l183_183020

-- Define the arithmetic-geometric sequence a_n
def a (n : ℕ) : ℤ := sorry

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℤ := sorry

-- Condition: a_1 = 1
axiom a1 : a 1 = 1

-- Condition: a_{n+2} + a_{n+1} - 2 * a_{n} = 0 for all n ∈ ℕ_+
axiom recurrence (n : ℕ) : a (n + 2) + a (n + 1) - 2 * a n = 0

-- Prove that S_5 = 11
theorem arithmetic_geometric_seq_sum_5 : S 5 = 11 := 
by
  sorry

end arithmetic_geometric_seq_sum_5_l183_183020


namespace monotonicity_of_g_inequality_f_l183_183413

noncomputable theory

-- Define the function f(x) for any real number a
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

-- Part 1: Show monotonicity properties for g(x) when a = 1
def g (x : ℝ) : ℝ := (f 1 x) / Real.exp x

-- Part 1: Theorem statement for the monotonicity of g(x)
theorem monotonicity_of_g : 
  (∀ x ∈ Ioo (-Real.pi/2) 0, g x decreases_on Ioo (-Real.pi/2) 0) ∧ 
  (∀ x ∈ Ioo 0 (3 * Real.pi / 2), g x increases_on Ioo 0 (3 * Real.pi / 2)) :=
sorry

-- Part 2: Theorem statement for the inequality when a = -3
theorem inequality_f : 
  ∀ x ∈ Ioi 0, f (-3) x < Real.exp x + x + 1 - 2 * Real.exp (-2 * x) :=
sorry

end monotonicity_of_g_inequality_f_l183_183413


namespace maximize_points_l183_183961

theorem maximize_points :
  let piles := (List.range 100).map (fun _ => 400)
  let total_stones := 100 * 400
  let max_points := 3920000
  (∃ f : List Int → List Int × List Int, ∑ i in List.natSucc 0 (total_stones / 2).pred, 
    | f([])._1 - f([])._2 | ≤ max_points) := 
sorry

end maximize_points_l183_183961


namespace correct_number_of_propositions_l183_183748

-- Definitions based on conditions from the problem
variable (a b c : Line)
variable (α β : Plane)
variable (L : ℕ)
variable (prop1 prop2 prop3 prop4 : Prop)

-- Propositions based on given conditions
def proposition1 : Prop := a ⊆ α ∧ b ⊆ α ∧ c ⊥ a ∧ c ⊥ b → c ⊥ α
def proposition2 : Prop := b ⊆ α ∧ a ∥ b → a ∥ α
def proposition3 : Prop := a ∥ α ∧ α ∩ β = b → a ∥ b
def proposition4 : Prop := a ⊥ α ∧ b ⊥ α → a ∥ b

-- Number of correct propositions
def number_correct (prop1 prop2 prop3 prop4 : Prop) : ℕ := 
  (if proposition1 then 1 else 0) +
  (if proposition2 then 1 else 0) +
  (if proposition3 then 1 else 0) +
  (if proposition4 then 1 else 0)

-- Theorem statement
theorem correct_number_of_propositions : 
  number_correct α β prop1 prop2 prop3 prop4 = 1 :=
  by sorry

end correct_number_of_propositions_l183_183748


namespace jerry_birthday_games_l183_183843

def jerry_original_games : ℕ := 7
def jerry_total_games_after_birthday : ℕ := 9
def games_jerry_got_for_birthday (original total : ℕ) : ℕ := total - original

theorem jerry_birthday_games :
  games_jerry_got_for_birthday jerry_original_games jerry_total_games_after_birthday = 2 := by
  sorry

end jerry_birthday_games_l183_183843


namespace total_coins_last_month_l183_183279

theorem total_coins_last_month (m s : ℝ) : 
  (100 = 1.25 * m) ∧ (100 = 0.80 * s) → m + s = 205 :=
by sorry

end total_coins_last_month_l183_183279


namespace maryann_free_time_l183_183127

theorem maryann_free_time
    (x : ℕ)
    (expensive_time : ℕ := 8)
    (friends : ℕ := 3)
    (total_time : ℕ := 42)
    (lockpicking_time : 3 * (x + expensive_time) = total_time) : 
    x = 6 :=
by
  sorry

end maryann_free_time_l183_183127


namespace number_of_ways_to_choose_subsets_l183_183609

open Finset

def setT : Finset ℕ := {0, 1, 2, 3, 4, 5}

def valid_subsets (T : Finset ℕ) (A B : Finset ℕ) : Prop :=
(A ∪ B = T) ∧ (A ∩ B).card = 3

theorem number_of_ways_to_choose_subsets :
  let S := setT in ∃ (n : ℕ), n = 80 :=
by
  sorry

end number_of_ways_to_choose_subsets_l183_183609


namespace twelfth_even_multiple_of_5_l183_183566

theorem twelfth_even_multiple_of_5 : 
  ∃ n : ℕ, n > 0 ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ ∀ m, (m > 0 ∧ (m % 2 = 0) ∧ (m % 5 = 0) ∧ m < n) → (m = 10 * (fin (n / 10) - 1)) := 
sorry

end twelfth_even_multiple_of_5_l183_183566


namespace minimum_value_of_func_l183_183312

noncomputable def func (x : ℝ) : ℝ := cos (2 * x) - 6 * cos x + 6

theorem minimum_value_of_func: ∃ x : ℝ, func x = 1 :=
by
  sorry

end minimum_value_of_func_l183_183312


namespace determinant_of_cos2_tan_1_l183_183499

variables {A B C : ℝ}

-- Condition: A, B, and C are angles of a triangle
def angles_of_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

theorem determinant_of_cos2_tan_1 (h : angles_of_triangle A B C) :
  let m := ![
    #[cos A ^ 2, tan A, 1],
    #[cos B ^ 2, tan B, 1],
    #[cos C ^ 2, tan C, 1]
  ] in
  matrix.det m = 0 :=
by sorry

end determinant_of_cos2_tan_1_l183_183499


namespace equal_segments_PE_EQ_l183_183525

-- Given data as definitions
variables {A B C L M E P Q : Type}
  [is_triangle ABC] -- Assuming a type class that represents that ABC is a triangle
  [is_circumcircle ABC] -- Assuming a type class that represents the circumcircle of triangle ABC
  (mid_AC : midpoint AC M) -- M is the midpoint of AC
  (bisect_ABC : angle_bisector ABC B L) -- The angle bisector intersects circumcircle at B and L
  (E_on_arc_ABC : on_arc_ABC E) -- E is chosen on the arc ABC
  (EM_parallel_BL : parallel E M B L) -- EM is parallel to BL
  (P_intersect_AB_EL : intersects_line AB EL P) -- P is the intersection of AB with EL
  (Q_intersect_BC_EL : intersects_line BC EL Q) -- Q is the intersection of BC with EL

-- Problem statement: prove that PE = EQ
theorem equal_segments_PE_EQ :
  distance P E = distance Q E :=
sorry

end equal_segments_PE_EQ_l183_183525


namespace right_triangles_impossible_l183_183885

/-- On a circle, 100 points are marked. Can there be exactly 1000 right triangles,
all vertices of which are these marked points?
-/
theorem right_triangles_impossible (h : ¬(∃ t : ℕ, t = 1000 ∧ 
  ∃ n : ℕ, n = 100 ∧ 
  let pairs := n / 2 in  
  let triangles_per_pair := n - 2 in 
  t = pairs * triangles_per_pair)): 
  True :=
begin
  sorry
end

end right_triangles_impossible_l183_183885


namespace zero_point_interval_l183_183537

def f (x : ℝ) : ℝ := (1 / 3)^x - x + 1

theorem zero_point_interval : ∃ c ∈ (Ioo 1 2), f c = 0 := 
by
  -- we will assume the continuity of the function f and then show the existence of a zero in the interval (1, 2)
  have h_cont : continuous f := sorry, -- assume continuity for now
  have h1 : f 1 > 0 := by 
  { simp [f],
    norm_num },
  have h2 : f 2 < 0 := by 
  { simp [f],
    norm_num },
  -- establish the zero in the interval (1, 2)
  obtain ⟨c, hc⟩ := intermediate_value_Ioo h_cont h1 h2,
  exact ⟨c, hc⟩

end zero_point_interval_l183_183537


namespace probability_converges_to_one_third_l183_183923

theorem probability_converges_to_one_third (n : ℕ) (p : ℕ → ℝ)
  (h_recurrence : ∀ n, p (n + 1) = p n - 1 / 2 * (p n)^2)
  (h_initial : p 0 = some_value) : 
  filter.tendsto p filter.at_top (nhds (1 / 3)) :=
sorry

end probability_converges_to_one_third_l183_183923


namespace total_books_in_school_l183_183455

theorem total_books_in_school (tables_A tables_B tables_C : ℕ)
  (books_per_table_A books_per_table_B books_per_table_C : ℕ → ℕ)
  (hA : tables_A = 750)
  (hB : tables_B = 500)
  (hC : tables_C = 850)
  (h_books_per_table_A : ∀ n, books_per_table_A n = 3 * n / 5)
  (h_books_per_table_B : ∀ n, books_per_table_B n = 2 * n / 5)
  (h_books_per_table_C : ∀ n, books_per_table_C n = n / 3) :
  books_per_table_A tables_A + books_per_table_B tables_B + books_per_table_C tables_C = 933 :=
by sorry

end total_books_in_school_l183_183455


namespace remainder_127_14_l183_183571

theorem remainder_127_14 : ∃ r : ℤ, r = 127 - (14 * 9) ∧ r = 1 := by
  sorry

end remainder_127_14_l183_183571


namespace evaluate_g_expression_l183_183297

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

theorem evaluate_g_expression : 3 * g 2 + 2 * g (-2) = 90 :=
by {
  -- Definitions directly used in the problem
  have h₁ : g 2 = 10 := by sorry,
  have h₂ : g (-2) = 30 := by sorry,
  -- Using the definitions to prove the problem
  calc
  3 * g 2 + 2 * g (-2) = 3 * 10 + 2 * 30 : by rw [h₁, h₂]
                     ... = 30 + 60       : by norm_num
                     ... = 90            : by norm_num
}

end evaluate_g_expression_l183_183297


namespace mixed_groups_count_l183_183990

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183990


namespace cone_volume_correct_l183_183922

def cone_volume (AB BC AM : ℝ) : ℝ :=
  let AC := real.sqrt (AB^2 + BC^2) in
  let r := AC / 2 in
  let h := real.sqrt (AB * (AB - AM) / 2) in
  (1/3) * real.pi * r^2 * h

theorem cone_volume_correct : cone_volume 8 6 5 = 25 * real.pi * real.sqrt(15) / 3 :=
by sorry

end cone_volume_correct_l183_183922


namespace product_digits_15_l183_183214

noncomputable def numberOfDigits (n : ℕ) : ℕ := sorry

theorem product_digits_15 :
  let A := 111111 in
  let B := 1111111111 in
  numberOfDigits (A * B) = 15 :=
sorry

end product_digits_15_l183_183214


namespace max_value_of_quadratic_l183_183681

theorem max_value_of_quadratic (x : ℝ) : 
  ∃ x : ℝ, (∀ y : ℝ, 10 * y - 5 * y^2 ≤ 10 * x - 5 * x^2) ∧ (10 * x - 5 * x^2 = 5) :=
begin
  sorry

end max_value_of_quadratic_l183_183681


namespace intersection_x_value_l183_183180

theorem intersection_x_value :
  (∃ x y : ℝ, y = 5 * x - 20 ∧ y = 110 - 3 * x ∧ x = 16.25) := sorry

end intersection_x_value_l183_183180


namespace distance_to_asymptote_l183_183942

/-- Define the hyperbola equation as a predicate --/
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

/-- Define the asymptote equations as predicates --/
def asymptote1 (x y : ℝ) : Prop := 3 * x - 4 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- Define the distance formula from a point to a line --/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2))

/-- Proof statement that the distance from (3,0) to asymptote is 9/5 --/
theorem distance_to_asymptote : 
  distance_from_point_to_line 3 0 3 (-4) 0 = 9 / 5 :=
by
  -- the main proof computation goes here
  sorry

end distance_to_asymptote_l183_183942


namespace factorize_1_factorize_2_l183_183334

-- Proof problem 1: Prove x² - 6x + 9 = (x - 3)²
theorem factorize_1 (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by sorry

-- Proof problem 2: Prove x²(y - 2) - 4(y - 2) = (y - 2)(x + 2)(x - 2)
theorem factorize_2 (x y : ℝ) : x^2 * (y - 2) - 4 * (y - 2) = (y - 2) * (x + 2) * (x - 2) :=
by sorry

end factorize_1_factorize_2_l183_183334


namespace distance_to_asymptote_l183_183943

/-- Define the hyperbola equation as a predicate --/
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

/-- Define the asymptote equations as predicates --/
def asymptote1 (x y : ℝ) : Prop := 3 * x - 4 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- Define the distance formula from a point to a line --/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2))

/-- Proof statement that the distance from (3,0) to asymptote is 9/5 --/
theorem distance_to_asymptote : 
  distance_from_point_to_line 3 0 3 (-4) 0 = 9 / 5 :=
by
  -- the main proof computation goes here
  sorry

end distance_to_asymptote_l183_183943


namespace smallest_solution_exists_l183_183722

noncomputable def ceil (x : ℝ) : ℤ := int.ceil x
noncomputable def frac_part (x : ℝ) : ℝ := x - int.floor x

theorem smallest_solution_exists :
  ∃ x : ℝ, ceil x = 8 + 90 * frac_part x ∧ x = 8.0111111 :=
sorry

end smallest_solution_exists_l183_183722


namespace plates_arrangement_l183_183246

theorem plates_arrangement :
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  non_adjacent_green_arrangements = 588 :=
by
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  sorry

end plates_arrangement_l183_183246


namespace triangle_area_l183_183814

theorem triangle_area (a b c : ℝ) (B C : ℝ) (h_a : a = 6) (h_B : B = π/6) (h_C : C = 2*π/3) :
  let A := π/6 in
  let S := 1/2 * a * b * Real.sin C in
  S = 9 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l183_183814


namespace max_sum_of_positives_l183_183445

theorem max_sum_of_positives (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 1 / x + 1 / y = 5) : x + y ≤ 4 :=
sorry

end max_sum_of_positives_l183_183445


namespace sequence_term_100_l183_183417

theorem sequence_term_100 :
  let a : ℕ → ℕ := λ n =>
    if n = 0 then 2
    else (a (n - 1) + 2 * (n - 1))
  in
  a 99 = 9902 := by
  sorry

end sequence_term_100_l183_183417


namespace find_m_l183_183792

variable (m : ℝ)
def vector_a : ℝ × ℝ := (1, 3)
def vector_b : ℝ × ℝ := (m, -2)

theorem find_m (h : (1 + m) + 3 = 0) : m = -4 := by
  sorry

end find_m_l183_183792


namespace polynomial_prime_divisor_exists_l183_183294

theorem polynomial_prime_divisor_exists (P : ℤ → ℤ) (d : ℕ → ℤ) (h_distinct: ∀ i j, i ≠ j → d i ≠ d j) :
    ∃ N : ℤ, ∀ x : ℤ, x ≥ N → ∃ p : ℕ, Nat.Prime p ∧ p > 20 ∧ p ∣ P x :=
by
  sorry

end polynomial_prime_divisor_exists_l183_183294


namespace cistern_fill_time_proof_l183_183576

noncomputable def cistern_full_time (rate_fill rate_empty : ℕ) : ℕ :=
  let net_rate := rate_fill - rate_empty in
  100

theorem cistern_fill_time_proof (C : ℕ) :
  let rate_A := C / 20
  let rate_B := C / 25
  rate_A > rate_B →
  cistern_full_time rate_A rate_B = 100 :=
by 
  intros
  rw [cistern_full_time]
  sorry

end cistern_fill_time_proof_l183_183576


namespace prism_volume_approx_l183_183187

theorem prism_volume_approx (a b c : Real) 
  (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) : 
  Real.round (a * b * c) = 426 := by
  sorry

end prism_volume_approx_l183_183187


namespace value_is_correct_l183_183801

def number : ℕ := 150
def fraction : ℚ := 3/5
def percentage : ℚ := 40/100

theorem value_is_correct : percentage * (fraction * number) = 36 := 
by
  have h1 : fraction * number = 90 := by
    norm_num
  have h2 : percentage * 90 = 36 := by
    norm_num
  exact Eq.trans (by rw [h1]) h2

end value_is_correct_l183_183801


namespace julia_played_with_34_kids_l183_183849

-- Define the number of kids Julia played with on each day
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Define the total number of kids Julia played with
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Prove given conditions
theorem julia_played_with_34_kids :
  totalKids = 34 :=
by
  sorry

end julia_played_with_34_kids_l183_183849


namespace complex_number_properties_l183_183021

-- Definitions for the complex numbers z1 and z2
def z1 : ℂ := 2 - I
def z2 : ℂ := 2 * I

-- Proposition for the proof
theorem complex_number_properties :
  (z2.im ≠ 0 ∧ z2.re = 0) ∧ (conj z1 = 2 + I) :=
by
  sorry

end complex_number_properties_l183_183021


namespace fruit_cost_l183_183276

theorem fruit_cost (apple_cost_per_3_pounds banana_cost_per_2_pounds : ℕ) 
(h1 : apple_cost_per_3_pounds = 3)
(h2 : banana_cost_per_2_pounds = 2) : 
3 * (9 / 3) + 2 * (6 / 2) = 15 :=
by
  rw [h1, h2]
  norm_num

end fruit_cost_l183_183276


namespace symmetry_center_of_f_l183_183441

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) + cos (ω * x)

def has_minimum_positive_period (ω : ℝ) : Prop :=
  ∃ T > 0, T = π ∧ ∀ x, f ω (x + T) = f ω x

def is_symmetry_center (ω : ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, f ω (2 * c.1 - x) = f ω x

theorem symmetry_center_of_f : ∀ (ω : ℝ), 
  ω > 0 → has_minimum_positive_period ω → is_symmetry_center ω (-π / 8, 0) :=
by
  intros ω hω hperiod
  sorry

end symmetry_center_of_f_l183_183441


namespace smallest_range_among_beneficiaries_l183_183258

theorem smallest_range_among_beneficiaries :
  ∀ (X : ℝ) (max_amount : ℝ),
  (max_amount = 80000) ∧ (1.4^7 * X = max_amount) →
  (range = max_amount - X) ∧ 
  (range = 72412) := 
by
  intros X max_amount h,
  -- You may assume the hypothesis h contains the necessary information to derive the conclusion
  sorry

end smallest_range_among_beneficiaries_l183_183258


namespace find_n_l183_183404

axiom initial_x : ℕ
axiom initial_n : ℕ
axiom loop_condition : ∀ x n, x = 169 → n = 0 → (∀ x n, x < n^2 ↔ x - n < (n + 1)^2)

theorem find_n (x n : ℕ) : x = 169 → n = 0 → (loop_condition x n) → n = 12 :=
by
  intro hx hn hc
  sorry

end find_n_l183_183404


namespace sequence_nonpositive_l183_183418

-- Definitions and conditions
def sequence (a : ℕ → ℝ) (n : ℕ) :=
  a 0 = 0 ∧ a n = 0 ∧ ∀ k, 1 ≤ k ∧ k ≤ n-1 → a (k-1) + a (k+1) - 2 * a k ≥ 0

-- The theorem to be proven
theorem sequence_nonpositive (a : ℕ → ℝ) (n : ℕ) (h : sequence a n) :
  ∀ k, 0 ≤ k ∧ k ≤ n → a k ≤ 0 :=
sorry

end sequence_nonpositive_l183_183418


namespace parabola_standard_equation_l183_183958

-- Define the focus location based on the problem condition
def focus_location : ℝ × ℝ := (1, 0)

-- Define the standard equation of the parabola as per the given focus
def parabola_equation (p : ℝ) : Prop :=
  p = 2 → y^2 = 4*x

-- State the proposition
theorem parabola_standard_equation : 
  parabola_equation 2 := 
by 
  sorry

end parabola_standard_equation_l183_183958


namespace problem_inequality_l183_183177

noncomputable def a_sequence (n : Nat) : ℝ :=
  2 * n + 1

noncomputable def b_sequence (n : Nat) : ℝ :=
  8 ^ (n - 1)

noncomputable def S_n (n : Nat) : ℝ :=
  n * (n + 2)

theorem problem_inequality (n : Nat) (n_pos : 0 < n) :
  (∑ i in Finset.range (n + 1), 1 / S_n i) < 3 / 4 := sorry

end problem_inequality_l183_183177


namespace quadratic_function_points_l183_183948

theorem quadratic_function_points (a c y1 y2 y3 y4 : ℝ) (h_a : a < 0)
    (h_A : y1 = a * (-2)^2 - 4 * a * (-2) + c)
    (h_B : y2 = a * 0^2 - 4 * a * 0 + c)
    (h_C : y3 = a * 3^2 - 4 * a * 3 + c)
    (h_D : y4 = a * 5^2 - 4 * a * 5 + c)
    (h_condition : y2 * y4 < 0) : y1 * y3 < 0 :=
by
  sorry

end quadratic_function_points_l183_183948


namespace sum_binom_1_div_m_plus_1_sum_binom_neg_1_pow_m_div_m_plus_1_l183_183514

-- Define the first theorem
theorem sum_binom_1_div_m_plus_1 (n : ℕ) :
    ∑ m in finset.range (n+1), (if m > 0 then (1/(m+1 : ℚ)) * nat.choose n m else 0) = (2^(n+1) - 1) / (n+1 : ℚ) :=
by
  sorry

-- Define the second theorem
theorem sum_binom_neg_1_pow_m_div_m_plus_1 (n : ℕ) :
    ∑ m in finset.range (n+1), (if m > 0 then ((-1)^m / (m+1 : ℚ)) * nat.choose n m else 0) = 1 / (n+1 : ℚ) :=
by
  sorry

end sum_binom_1_div_m_plus_1_sum_binom_neg_1_pow_m_div_m_plus_1_l183_183514


namespace find_min_value_l183_183717

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem find_min_value (a : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, f x a ≥ (if a < -1 then 2 * a else if a ≤ 1 then -1 - a^2 else -2 * a)) ∧
  (∃ x ∈ set.Icc (-1 : ℝ) 1, f x a = (if a < -1 then 2 * a else if a ≤ 1 then -1 - a^2 else -2 * a)) :=
by
  sorry

end find_min_value_l183_183717


namespace divide_quadrilateral_into_three_trapezoids_l183_183519

variables {A B C D M N K : Type} 
variables [ordered_add_comm_group A] 
variables [ordered_add_comm_group B]
variables [ordered_add_comm_group C]
variables [ordered_add_comm_group D]
variables [ordered_add_comm_group M]
variables [ordered_add_comm_group N]
variables [ordered_add_comm_group K]

structure quadrilateral (A B C D : Type) :=
(Angle_B_largest : ∀ (angle: A B C D), angle ≤ angle B)
(segment_BM_parallel_AD : ∀ (BM AD), BM ∥ AD)
(segment_MN_parallel_BC : ∀ (MN BC), MN ∥ BC)
(segment_MK_parallel_CD : ∀ (MK CD), MK ∥ CD)

theorem divide_quadrilateral_into_three_trapezoids 
  (ABC_quad : quadrilateral A B C D) : 
  ∃ (BM BN BK : Type), BM ∥ AD ∧ BN ∥ BC ∧ BK ∥ CD :=
sorry

end divide_quadrilateral_into_three_trapezoids_l183_183519


namespace fraction_sum_correct_l183_183211

example : Real := 3.0035428163476343
def frac1 := 2007 / 2999
def frac2 := 11002 / 5998
def frac3 := 2001 / 3999

theorem fraction_sum_correct : frac1 + frac2 + frac3 = example := by
  sorry

end fraction_sum_correct_l183_183211


namespace find_value_of_x_squared_plus_inverse_squared_l183_183055

theorem find_value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + (1/x) = 2) : x^2 + (1/x^2) = 2 :=
sorry

end find_value_of_x_squared_plus_inverse_squared_l183_183055


namespace lambda_3_sufficient_but_not_necessary_l183_183024

open Real

noncomputable def vec_a (λ : ℝ) : ℝ × ℝ := (3, λ)
noncomputable def vec_b (λ : ℝ) : ℝ × ℝ := (λ - 1, 2)

def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem lambda_3_sufficient_but_not_necessary : 
  ∀ (λ : ℝ), are_parallel (vec_a λ) (vec_b λ) 
  ↔ λ = 3 ∨ λ = -2 → λ = 3 ∧ (∃ λ' ≠ 3, are_parallel (vec_a λ') (vec_b λ')) := by
  sorry

end lambda_3_sufficient_but_not_necessary_l183_183024


namespace find_a_l183_183779

-- Define the function f and its derivative
def f (a : ℝ) (f'_2 : ℝ) (x : ℝ) : ℝ := a * x^3 + f'_2 * x^2 + 3
def f' (a : ℝ) (f'_2 : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * f'_2 * x

theorem find_a (h1 : f'(a, f'(a, f'(a, -4*a, 2), 2), 1) = -5) : a = 1 :=
by sorry

end find_a_l183_183779


namespace number_of_triangles_l183_183256

-- Define the points and conditions
def pointP : (ℚ × ℚ) := (0, 0)

def isLatticePoint (p : ℚ × ℚ) : Prop :=
  ∃ m n : ℤ, p = (m, n)

def onLine2x (p : ℚ × ℚ) : Prop :=
  p.snd = 2 * p.fst

def onLine3x (p : ℚ × ℚ) : Prop :=
  p.snd = 3 * p.fst

def area (P Q R : (ℚ × ℚ)) : ℚ :=
  1 / 2 * abs (P.1 * Q.2 + Q.1 * R.2 + R.1 * P.2 - P.2 * Q.1 - Q.2 * R.1 - R.2 * P.1)

def isTriangle (P Q R : (ℚ × ℚ)) : Prop :=
  area P Q R = 500000

-- Main theorem
theorem number_of_triangles :
  let possiblePairs := { qr : (ℤ × ℤ) | qr.1 * qr.2 = 1000000 } in
  possiblePairs.count = 49 :=
sorry

end number_of_triangles_l183_183256


namespace simplify_complex_fraction_l183_183910

/-- The simplified form of (5 + 7 * I) / (2 + 3 * I) is (31 / 13) - (1 / 13) * I. -/
theorem simplify_complex_fraction : (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i := 
by {
    sorry
}

end simplify_complex_fraction_l183_183910


namespace regression_lines_common_point_l183_183189

variables (n m : ℕ) (s t : ℝ)
-- Define the conditions for the given problem
-- Students A and B conducted n (n=10) and m (m=15) experiments respectively
def experiments_A : ℕ := n
def experiments_B : ℕ := m
-- Average values of x and y for both datasets are s and t respectively
def avg_x (dataset : Type) : ℝ := s
def avg_y (dataset : Type) : ℝ := t 

-- Define the regression lines, here we assume they are given as functions
def regression_line_1 (x : ℝ) : ℝ := sorry
def regression_line_2 (x : ℝ) : ℝ := sorry

-- State the theorem
theorem regression_lines_common_point:
  n = 10 → m = 15 →
  avg_x ℝ = s → avg_y ℝ = t →
  regression_line_1 s = t ∧ regression_line_2 s = t :=
by {
  intros h_expA h_expB h_avgx h_avgy,
  sorry
}

end regression_lines_common_point_l183_183189


namespace calc_ratio_of_d_to_s_l183_183452

theorem calc_ratio_of_d_to_s {n s d : ℝ} (h_n_eq_24 : n = 24)
    (h_tiles_area_64_pct : (576 * s^2) = 0.64 * (n * s + d)^2) : 
    d / s = 6 / 25 :=
by
  sorry

end calc_ratio_of_d_to_s_l183_183452


namespace minimum_integral_l183_183102

-- Define the function space F with conditions
noncomputable def F : Set (ℝ → ℝ) :=
  {f | ∃ (f' : ℝ → ℝ), Continuous f' ∧ Differentiable ℝ f ∧ f 0 = 0 ∧ f 1 = 1}

-- Define the functional to be minimized
noncomputable def J (f : ℝ → ℝ) : ℝ :=
  ∫ (0:ℝ) 1, (sqrt (1 + x^2)) * (derivative f x)^2

theorem minimum_integral : ∃ f ∈ F, J f = π / (4 * (log (1 + sqrt 2))^2) :=
  sorry  -- Proof goes here

end minimum_integral_l183_183102


namespace heather_ends_up_with_45_blocks_l183_183794

-- Conditions
def initialBlocks (Heather : Type) : ℕ := 86
def sharedBlocks (Heather : Type) : ℕ := 41

-- The theorem to prove
theorem heather_ends_up_with_45_blocks (Heather : Type) :
  (initialBlocks Heather) - (sharedBlocks Heather) = 45 :=
by
  sorry

end heather_ends_up_with_45_blocks_l183_183794


namespace dheo_total_bills_and_coins_l183_183687

theorem dheo_total_bills_and_coins 
  (total_bill : ℕ)
  (num_20_peso_bills : ℕ)
  (num_5_peso_coins : ℕ)
  (value_20_peso_bill : ℕ := 20)
  (value_5_peso_coin : ℕ := 5)
  (number_of_20_peso_bills : ℕ := 11)
  (number_of_5_peso_coins : ℕ := 11) :
  total_bill = (number_of_20_peso_bills * value_20_peso_bill) + (number_of_5_peso_coins * value_5_peso_coin) →
  (num_20_peso_bills = number_of_20_peso_bills) →
  (num_5_peso_coins = number_of_5_peso_coins) →
  (num_20_peso_bills + num_5_peso_coins) = 22 :=
by
  intros h_total_bill h_num_20_bills h_num_5_coins
  rw [h_num_20_bills, h_num_5_coins]
  exact rfl

end dheo_total_bills_and_coins_l183_183687


namespace fruit_discard_percentage_l183_183614

theorem fruit_discard_percentage :
  (let initial := 100.0;
       pears_day1_sold := 0.2 * initial;
       pears_day1_remaining := initial - pears_day1_sold;
       pears_day1_discarded := 0.3 * pears_day1_remaining;
       pears_day2_remaining1 := pears_day1_remaining - pears_day1_discarded;
       pears_day2_sold := 0.1 * pears_day2_remaining1;
       pears_day2_remaining2 := pears_day2_remaining1 - pears_day2_sold;
       pears_day2_discarded := 0.2 * pears_day2_remaining2;
       pears_total_discarded := pears_day1_discarded + pears_day2_discarded;

       apples_day1_sold := 0.25 * initial;
       apples_day1_remaining := initial - apples_day1_sold;
       apples_day1_discarded := 0.15 * apples_day1_remaining;
       apples_day2_remaining1 := apples_day1_remaining - apples_day1_discarded;
       apples_day2_sold := 0.15 * apples_day2_remaining1;
       apples_day2_remaining2 := apples_day2_remaining1 - apples_day2_sold;
       apples_day2_discarded := 0.1 * apples_day2_remaining2;
       apples_total_discarded := apples_day1_discarded + apples_day2_discarded;

       oranges_day1_sold := 0.3 * initial;
       oranges_day1_remaining := initial - oranges_day1_sold;
       oranges_day1_discarded := 0.35 * oranges_day1_remaining;
       oranges_day2_remaining1 := oranges_day1_remaining - oranges_day1_discarded;
       oranges_day2_sold := 0.2 * oranges_day2_remaining1;
       oranges_day2_remaining2 := oranges_day2_remaining1 - oranges_day2_sold;
       oranges_day2_discarded := 0.3 * oranges_day2_remaining2;
       oranges_total_discarded := oranges_day1_discarded + oranges_day2_discarded
   in
     (pears_total_discarded / initial * 100 = 34.08) ∧
     (apples_total_discarded / initial * 100 = 16.66875) ∧
     (oranges_total_discarded / initial * 100 = 35.42))
:= by
  sorry

end fruit_discard_percentage_l183_183614


namespace derivative_given_condition_l183_183798

variable (f : ℝ → ℝ) (x0 : ℝ)

theorem derivative_given_condition :
  (∀ Δx : ℝ, Δx ≠ 0 → ∃ l : ℝ, is_limit (λ Δx, (f (x0 + 2 * Δx) - f x0) / Δx) l (0 : ℝ) ∧ l = 1) →
  deriv f x0 = 1 / 2 :=
by
  intro h
  sorry

end derivative_given_condition_l183_183798


namespace minimum_value_of_f_l183_183718

def f (x : ℝ) : ℝ := |3 - x| + |x - 2|

theorem minimum_value_of_f : ∃ x0 : ℝ, (∀ x : ℝ, f x0 ≤ f x) ∧ f x0 = 1 := 
by
  sorry

end minimum_value_of_f_l183_183718


namespace sum_of_squares_and_cubes_l183_183904

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  ∃ x1 x2 : ℤ, a^2 - 2*b = x1^2 + x2^2 ∧ 3*a*b - a^3 = x1^3 + x2^3 :=
by
  sorry

end sum_of_squares_and_cubes_l183_183904


namespace weatherman_interview_sam_champion_childhood_dream_present_weather_tech_study_of_weather_science_l183_183562

theorem weatherman_interview :
  (Judging from the writing style (text: Type), text is B) :=
begin
 sorry,
end

theorem sam_champion_childhood_dream :
  (As a child (sam_child_dream: Type), sam_child_dream is C) :=
begin
 sorry,
end

theorem present_weather_tech :
  (Present weather forecasting technology (weather_tech: Type), weather_tech is D) :=
begin
 sorry,
end

theorem study_of_weather_science :
  (The study of weather science (weather_science_study: Type), weather_science_study is A) :=
begin
 sorry,
end

end weatherman_interview_sam_champion_childhood_dream_present_weather_tech_study_of_weather_science_l183_183562


namespace area_of_triangle_length_of_a_l183_183466

-- Given conditions
variables {A B C : Type*} [IsTriangle A B C]
variables {a b c : ℝ}
variables (angle_A : ℝ)

-- Given conditions expressions
axiom b_def : b = 2
axiom c_def : c = sqrt 3
axiom angle_A_def : angle_A = π / 6

-- Prove the area of the triangle
theorem area_of_triangle :
  (1/2) * b * c * sin angle_A = sqrt 3 / 2 :=
by
  rw [b_def, c_def, angle_A_def]
  -- Proof steps would go here
  sorry

-- Prove the length of side a
theorem length_of_a :
  sqrt (b^2 + c^2 - 2 * b * c * cos angle_A) = 1 :=
by
  rw [b_def, c_def, angle_A_def]
  -- Proof steps would go here
  sorry

end area_of_triangle_length_of_a_l183_183466


namespace star_m_eq_21_l183_183491

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

def S : Set ℕ := { n | digit_sum n = 15 ∧ n < 10^6 }

def m : ℕ := S.card

theorem star_m_eq_21 : digit_sum m = 21 :=
by
  sorry

end star_m_eq_21_l183_183491


namespace cos_eq_zero_necessary_not_sufficient_l183_183797

-- Define the main problem statement
def cos_eq_zero_iff_sin_eq_one (α : ℝ) : Prop :=
  (cos α = 0) -> (sin α = 1)

-- Define the necessary but not sufficient condition
theorem cos_eq_zero_necessary_not_sufficient {α : ℝ} :
  (cos_eq_zero_iff_sin_eq_one α) ∧ ¬ (cos_eq_zero_iff_sin_eq_one α) :=
by
  sorry

end cos_eq_zero_necessary_not_sufficient_l183_183797


namespace mixed_groups_count_l183_183973

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183973


namespace ratio_of_areas_l183_183558

theorem ratio_of_areas (O P X : Type) [normed_space ℝ X]
  (h_eq : ∥OX∥ = (1 / 3) * ∥OP∥) :
  let area_ratio := (∥OX∥ / ∥OP∥)^2 in area_ratio = 1 / 9 := by
  sorry

end ratio_of_areas_l183_183558


namespace range_of_y_eq_2_sin_sq_x_l183_183953

open Real

noncomputable def range_of_function : set ℝ := 
  {y : ℝ | ∃ x : ℝ, y = 2 * sin x ^ 2 }

theorem range_of_y_eq_2_sin_sq_x : range_of_function = set.Icc 0 2 := 
by
  sorry

end range_of_y_eq_2_sin_sq_x_l183_183953


namespace find_a_l183_183123

-- Define that f(x) is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

-- Define the theorem we need to prove
theorem find_a (a : ℝ) (h : is_even_function (f a)) : a = -1 :=
by
  sorry

#check find_a

end find_a_l183_183123


namespace polynomial_evaluation_l183_183397

theorem polynomial_evaluation (n : ℕ) (p : ℕ → ℝ) 
  (h_poly : ∀ k, k ≤ n → p k = 1 / (Nat.choose (n + 1) k)) :
  p (n + 1) = if n % 2 = 0 then 1 else 0 :=
by
  sorry

end polynomial_evaluation_l183_183397


namespace hyperbola_eccentricity_l183_183045

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : b^2 = a^2 - (a^2 - b^2))
  (h4 : (∀F1 F2 M : ℝ, let F1 := (a - (a^2 - b^2).sqrt, b),
                           F2 := (a + (a^2 - b^2).sqrt, b),
                           M := (a, b)
           in (F1, F2, M) ∧ (|((M.1 - F1.1)^2 + (M.2 - F1.2)^2).sqrt - ((M.1 - F2.1)^2 + (M.2 - F2.2)^2).sqrt| = 2 * b))
 : (∃e : ℝ, e^2 = (√5 + 1) / 2) :=
begin
  sorry
end

end hyperbola_eccentricity_l183_183045


namespace parallelogram_side_lengths_and_diagonal_l183_183166

-- Definitions and conditions

def parallelogram_angle_60 (P: Type) [metric_space P] (A B C D : P) : Prop :=
  ∃ (θ : ℝ), θ = 60 ∧ angle A B C = θ

def shorter_diagonal (P: Type) [metric_space P] (A B C D : P) (O : P) : Prop :=
  (dist A C = 2 * sqrt 31) ∧ midpoint ℝ A C = O

def perpendicular_length (P: Type) [metric_space P] (A B C D : P) (O N M: P) : Prop :=
  dist O N = (sqrt 75) / 2 ∧ dist O M = dist O N ∧ dist B N = sqrt 75

-- Prove the lengths of the sides and the larger diagonal
theorem parallelogram_side_lengths_and_diagonal {P: Type} [metric_space P] 
  {A B C D O N M : P} :
  parallelogram_angle_60 P A B C D →
  shorter_diagonal P A B C D O →
  perpendicular_length P A B C D O N M →
  dist A B = 10 ∧ dist A D = 14 ∧ dist A D = 2 * sqrt 91 := 
sorry  -- this is where the proof would go

end parallelogram_side_lengths_and_diagonal_l183_183166


namespace calc_distance_C_to_D_l183_183089

def distance_from_C_to_D : ℝ :=
  let side_smaller_sq := 8 / 4
  let side_larger_sq := Real.sqrt 64
  let horizontal_side := side_smaller_sq + side_larger_sq
  let vertical_side := side_larger_sq - side_smaller_sq
  Real.sqrt (horizontal_side^2 + vertical_side^2)

theorem calc_distance_C_to_D :
  distance_from_C_to_D = 11.7 :=
  sorry

end calc_distance_C_to_D_l183_183089


namespace possible_values_of_expression_l183_183760

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (k : ℕ), k ∈ [0, 1, 2, 3, 4] ∧ 
  (∑ (n : ℕ) in finset.range 4, 
    (if n = 0 then a else if n = 1 then b else if n = 2 then c else d) / 
    |(if n = 0 then a else if n = 1 then b else if n = 2 then c else d)|) +
  (abcd / |abcd|) ∈ ({5, 1, -1, -5} : set ℝ) := sorry

end possible_values_of_expression_l183_183760


namespace not_a_parabola_l183_183437

theorem not_a_parabola {θ : ℝ} (h : ∀ x y : ℝ, x^2 + y^2 * cos θ ≠ 4) : False :=
sorry

end not_a_parabola_l183_183437


namespace distinct_triplet_inequality_l183_183725

theorem distinct_triplet_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  abs (a / (b - c)) + abs (b / (c - a)) + abs (c / (a - b)) ≥ 2 := 
sorry

end distinct_triplet_inequality_l183_183725


namespace determine_a_l183_183056

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end determine_a_l183_183056


namespace relative_errors_are_equal_l183_183273

-- Define the conditions
def line_length1 : ℝ := 25 -- Length of the first line in inches
def error1 : ℝ := 0.05 -- Error in the first measurement in inches
def line_length2 : ℝ := 50 -- Length of the second line in inches
def error2 : ℝ := 0.1 -- Error in the second measurement in inches

-- Define the relative errors
def relative_error1 : ℝ := (error1 / line_length1) * 100
def relative_error2 : ℝ := (error2 / line_length2) * 100

-- State the theorem that we want to prove
theorem relative_errors_are_equal : relative_error1 = relative_error2 :=
by
  -- Skip the proof here
  sorry

end relative_errors_are_equal_l183_183273


namespace jerusha_earnings_l183_183845

variable (L : ℝ) 

theorem jerusha_earnings (h1 : L + 4 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l183_183845


namespace function_relationship_selling_price_for_profit_max_profit_l183_183612

-- Step (1): Prove the function relationship between y and x
theorem function_relationship (x y: ℝ) (h1 : ∀ x, y = -2*x + 80)
  (h2 : x = 22 ∧ y = 36 ∨ x = 24 ∧ y = 32) :
  y = -2*x + 80 := by
  sorry

-- Step (2): Selling price per book for a 150 yuan profit per week
theorem selling_price_for_profit (x: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) (profit : ℝ)
  (h_profit : profit = (x - 20) * (-2*x + 80)) (h2 : profit = 150) : 
  x = 25 := by
  sorry

-- Step (3): Maximizing the weekly profit
theorem max_profit (x w: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) 
  (profit : ∀ x, w = (x - 20) * (-2*x + 80)) :
  w = 192 ∧ x = 28 := by
  sorry

end function_relationship_selling_price_for_profit_max_profit_l183_183612


namespace students_with_same_number_of_friends_l183_183269

theorem students_with_same_number_of_friends :
    ∀ (students : Finset ℕ) (n : ℕ) (friends : ℕ → Finset ℕ),
    students.card = 30 →
    ∀ s ∈ students, friends s ⊆ students ∧ s ∉ friends s →
    ∃ s₁ s₂ ∈ students, s₁ ≠ s₂ ∧ friends s₁.card = friends s₂.card :=
by
  sorry

end students_with_same_number_of_friends_l183_183269


namespace grapes_total_sum_l183_183261

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l183_183261


namespace problem_geometric_sequence_arithmetic_and_product_l183_183014

theorem problem_geometric_sequence_arithmetic_and_product
  (b : ℕ+ → ℝ)
  (a : ℕ+ → ℝ)
  (m : ℝ)
  (h1 : ∀ n : ℕ+, b n = 3 ^ (a n))
  (h2 : a 8 + a 13 = m) 
  (q : ℝ) 
  (hq : ∀ n : ℕ+, b (n+1) = q * b n) :
  -- 1. The sequence {a_n} is an arithmetic sequence with difference log_3 q
  (∃ d : ℝ, ∀ n : ℕ+, a (n+1) - a n = d ∧ d = log q / log 3) ∧
  -- 2. The product of the first 20 terms of {b_n} is 3^(10m)
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 * b 8 * b 9 * b 10 * b 11 * b 12 * b 13 * b 14 * b 15 * b 16 * b 17 * b 18 * b 19 * b 20 = 3 ^ (10 * m)) :=
sorry

end problem_geometric_sequence_arithmetic_and_product_l183_183014


namespace sum_u_n_eq_zero_l183_183674

noncomputable def u0 : ℝ × ℝ := ⟨2, 2⟩
noncomputable def z0 : ℝ × ℝ := ⟨3, -3⟩

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let u_norm_sq := u.1 * u.1 + u.2 * u.2
  let scalar := dot_product / u_norm_sq
  (scalar * u.1, scalar * u.2)

noncomputable def u_n (n : ℕ) : ℝ × ℝ :=
  Nat.rec z0 (λ n z_nm1, proj u0 z_nm1) n

noncomputable def sum_u_n : ℝ × ℝ :=
  (0, 0)

theorem sum_u_n_eq_zero : ∀ n, (u_n (n+1) = (0, 0)) → sum_u_n = (0, 0) :=
by
  intros
  sorry

end sum_u_n_eq_zero_l183_183674


namespace range_of_a_l183_183072

noncomputable def f (x : ℝ) := x * Real.log x - x ^ 3 + x ^ 2 - a * Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≤ 0) → a ∈ Set.Ici 0 := by
  sorry

end range_of_a_l183_183072


namespace partition_sum_lt_one_l183_183723

theorem partition_sum_lt_one (n : ℕ) (h_pos : n > 0)
    (P : fin n → finset (fin (2 * n))) (h_partition : ∀ i, ∃ (a b : fin (2 * n)), a ≠ b ∧ P i = {a, b})
    (p : fin n → ℕ) (h_p : ∀ i, ∃ a b, a ≠ b ∧ P i = {a, b} ∧ p i = a.val * b.val) :
    (∑ i in finset.fin_range n, ((1 : ℚ) / p i)) < 1 := 
begin
  sorry
end

end partition_sum_lt_one_l183_183723


namespace proof_problem_l183_183046

noncomputable def l1 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0
noncomputable def l2 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

theorem proof_problem (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y → x^2 + y^2 + x - y ≠ 0) ∧ 
  (∀ x : ℝ, ∀ y : ℝ, l1 a x y.1 → l2 a x y.2 → |(x, y)| ≤ sqrt(2)) ∧
  (∀ a : ℝ, ∃ x y : ℝ, l1 a x y ∧ l2 a x y ∧ x = 0 ∧ y = 1) ∧ 
  (∀ a : ℝ, ∃ x y : ℝ, l1 a x y ∧ l2 a x y ∧ x = -1 ∧ y = 0) :=
begin
  sorry
end

end proof_problem_l183_183046


namespace measure_of_angle_B_l183_183468

-- Definitions based on the conditions
def angle_A := 45
def side_a := 6
def side_b := 3 * Real.sqrt 2

-- The main theorem statement to prove
theorem measure_of_angle_B (angle_A = 45) (side_a = 6) (side_b = 3 * Real.sqrt 2) :
  ∃ B, B = 30 :=
by
  sorry -- proof not required

end measure_of_angle_B_l183_183468


namespace correct_option_c_l183_183919

variables (l m : Type) [decidable_eq l] [decidable_eq m]
variables (α β : Type) [decidable_eq α] [decidable_eq β]

variables [has_parallel l m] [has_parallel l α] [has_parallel m α]
variables [has_perp l m] [has_perp l α] [has_perp l β]
variables [has_perp m α] [has_perp m β]
variables [has_subset m α] [has_subset l β]

theorem correct_option_c (hl_parallel_alpha : l ∥ α)
  (hm_parallel_beta : m ∥ β)
  (hl_perp_alpha : l ⊥ α)
  (h_alpha_parallel_beta : α ∥ β) :
  l ⊥ m :=
begin
  sorry
end

end correct_option_c_l183_183919


namespace optionA_correct_optionB_incorrect_optionC_incorrect_optionD_incorrect_l183_183208

theorem optionA_correct: real.sqrt 8 + real.sqrt 2 = 3 * real.sqrt 2 := sorry

theorem optionB_incorrect: ¬ (3 * real.sqrt 6 - real.sqrt 6 = 3) := sorry

theorem optionC_incorrect: ¬ (2 * real.sqrt 7 * (3 * real.sqrt 7) = 6 * real.sqrt 7) := sorry

theorem optionD_incorrect: ¬ (real.sqrt 27 / real.sqrt 3 = 9) := sorry

end optionA_correct_optionB_incorrect_optionC_incorrect_optionD_incorrect_l183_183208


namespace cookies_remaining_l183_183304

theorem cookies_remaining :
  ∀ (initial_white_cookies : ℕ) 
    (initial_black_cookies : ℕ) 
    (remaining_white_cookies : ℕ) 
    (remaining_black_cookies : ℕ),

  initial_white_cookies = 80 →
  initial_black_cookies = initial_white_cookies + 50 →
  remaining_white_cookies = initial_white_cookies - (3 * initial_white_cookies / 4) →
  remaining_black_cookies = initial_black_cookies / 2 →
  remaining_white_cookies + remaining_black_cookies = 85 :=
by
  intros initial_white_cookies initial_black_cookies remaining_white_cookies remaining_black_cookies
  assume h1 : initial_white_cookies = 80
  assume h2 : initial_black_cookies = initial_white_cookies + 50
  assume h3 : remaining_white_cookies = initial_white_cookies - (3 * initial_white_cookies / 4)
  assume h4 : remaining_black_cookies = initial_black_cookies / 2
  sorry

end cookies_remaining_l183_183304


namespace identify_monomials_identify_polynomials_l183_183553

-- Definitions based on the problem conditions
def monomial (expr : String) : Prop :=
  expr = "7x^5" ∨ expr = "-2x^2y^3" ∨ expr = "8"

def polynomial (expr : String) : Prop :=
  expr = "3xy+6" ∨ expr = "x/3+y/3"

-- Given list of expressions
def expressions : List String := ["7x^5", "3xy+6", "-2x^2y^3", "x/3+y/3", "8", "s=ab"]

-- Statements to prove 
theorem identify_monomials : (expressions.filter monomial) = ["7x^5", "-2x^2y^3", "8"] := by
  sorry

theorem identify_polynomials : (expressions.filter polynomial) = ["3xy+6", "x/3+y/3"] := by
  sorry

end identify_monomials_identify_polynomials_l183_183553


namespace divisible_by_17_l183_183896

theorem divisible_by_17 (n : ℕ) (hn : n > 0) :
  (nat.odd n → 17 ∣ (2^(4*n) - 1)) ∧ (nat.even n → 17 ∣ (2^(4*n) + 1)) :=
by
  sorry

end divisible_by_17_l183_183896


namespace complement_of_A_in_U_l183_183874

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : U \ A = {2, 4} := 
by
  sorry

end complement_of_A_in_U_l183_183874


namespace mixed_groups_count_l183_183983

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183983


namespace quadrilateral_perimeter_of_rectangle_midpoints_l183_183721

theorem quadrilateral_perimeter_of_rectangle_midpoints (d : ℝ) (h : d = 8) :
  let a b : ℝ := sorry in
  (a^2 + b^2 = d^2) →
  ((1/2) * Real.sqrt (a^2 + b^2) = 4) →
  4 * ((1/2) * Real.sqrt (a^2 + b^2)) = 16 :=
by
  intros a b h_eq hb
  rw [h_eq] at hb
  sorry

end quadrilateral_perimeter_of_rectangle_midpoints_l183_183721


namespace seats_not_occupied_l183_183325

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l183_183325


namespace last_digit_of_S_l183_183680

theorem last_digit_of_S : 
  let S := 54^2019 + 28^2021 in S % 10 = 2 :=
by
  sorry

end last_digit_of_S_l183_183680


namespace distance_from_point_to_hyperbola_asymptote_l183_183932

noncomputable def distance_to_asymptote (x1 y1 a b : ℝ) : ℝ :=
  abs (a * x1 + b * y1) / real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_hyperbola_asymptote :
  distance_to_asymptote 3 0 3 (-4) = 9 / 5 :=
by
  sorry

end distance_from_point_to_hyperbola_asymptote_l183_183932


namespace maximal_moves_lamps_l183_183181

theorem maximal_moves_lamps (n : ℕ) (h : n > 1) : 
  ∃ (moves : ℕ), moves = n - 1 ∧ 
  ∀ config : list bool, 
    init_config config n →
    valid_moves config →
    process_terminates config moves := sorry

-- Definitions required for the theorem to make sense (not proofs, just structure):

def init_config (config : list bool) (n : ℕ) : Prop :=
  config.length = 2 * n - 1 ∧ (nth config (n - 1) = some true) ∧
  (∀ i, i ≠ n - 1 → nth config i = some false)

def valid_moves (config : list bool) : Prop :=
  ∃ (L : ℕ), L ≥ 3 ∧
  ∃ (start end_ : ℕ), start < end_ ∧
  (nth config start = some false) ∧
  (nth config end_ = some false) ∧
  (∀ k, start < k ∧ k < end_ → nth config k = some true)

def process_terminates (config : list bool) (moves : ℕ) : Prop :=
  ∀ move_count, move_count ≤ moves →
  (move_count = moves → ∀ i j, j > i + 1 → 
   nth config i = some false → nth config j = some false → 
   (∃ k, i < k ∧ k < j ∧ nth config k ≠ some true))

end maximal_moves_lamps_l183_183181


namespace hyperbola_eccentricity_l183_183784

-- Define the hyperbola and asymptote conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def asymptote_condition (a b : ℝ) : Prop :=
  (b^2 / a^2) = 1 / 2

-- Define the parameters (a must be positive, b must be positive, and the specific asymptote condition)
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hahb : asymptote_condition a b)

-- Define the eccentricity e
def eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in 
  c / a

-- Statement of the problem with the given conditions
theorem hyperbola_eccentricity : eccentricity a b = Real.sqrt 6 / 2 :=
by
  -- Import necessary definitions and provide proof outline
  sorry

end hyperbola_eccentricity_l183_183784


namespace avg_var_sample2_l183_183398

variable {α : Type*} [fintype α] [decidable_eq α] (x : α → ℝ) (n : ℕ)
variable (sample1 : α → ℝ) (sample2 : α → ℝ)

def average (s : α → ℝ) : ℝ := (∑ i, s i) / (fintype.card α)
def variance (s : α → ℝ) : ℝ := (∑ i, (s i - average s) ^ 2) / (fintype.card α)

-- Given Conditions
axiom avg_sample1 : average sample1 = 10
axiom var_sample1 : variance sample1 = 2

-- Definition of sample2 derived from x being shifted by constant values
def sample1 := λ i, 1 + x i
def sample2 := λ i, 2 + x i

-- Required to Prove
theorem avg_var_sample2 :
  average sample2 = 11 ∧ variance sample2 = 2 := 
  sorry

end avg_var_sample2_l183_183398


namespace exercise_l183_183112

theorem exercise (a b d : ℝ) (x : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : d ≠ 0)
  (h4 : ∀ x, (g : ℝ → ℝ := λ x, (2 * a * x - b) / (d * x - 2 * b)) ) : 2 * a - 2 * b = 0 := 
sorry

end exercise_l183_183112


namespace lydia_age_when_planted_l183_183840

-- Definition of the conditions
def years_to_bear_fruit : ℕ := 7
def lydia_age_when_fruit_bears : ℕ := 11

-- Lean 4 statement to prove Lydia's age when she planted the tree
theorem lydia_age_when_planted (a : ℕ) : a = lydia_age_when_fruit_bears - years_to_bear_fruit :=
by
  have : a = 4 := by sorry
  exact this

end lydia_age_when_planted_l183_183840


namespace arithmetic_sequence_no_three_l183_183019

open Nat

theorem arithmetic_sequence_no_three (d : ℕ) (h_pos : 0 < d) (h : d ∈ (factors 80)) : d ≠ 3 := by
  sorry

end arithmetic_sequence_no_three_l183_183019


namespace parabola_vertex_x_coord_l183_183162

open Real

theorem parabola_vertex_x_coord {a b c x : ℝ} 
  (h1 : (2:ℝ) * a + b * 2 + c = 9)
  (h2 : (8:ℝ) * a + b * 8 + c = 9)
  (h3 : (3:ℝ) * a + b * 3 + c = 4) : 
  x = 5 :=
begin
  sorry
end

end parabola_vertex_x_coord_l183_183162


namespace mixed_groups_count_l183_183998

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l183_183998


namespace sin_double_angle_l183_183734

theorem sin_double_angle (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) = 4 / 5 :=
sorry

end sin_double_angle_l183_183734


namespace max_min_A_union_B_l183_183852

noncomputable def A_set (a b : Fin 2n → ℝ) : Set ℝ :=
  {x | ∃ i : Fin 2n, x = ∑ j : Fin 2n, a i * b j / (a i * b j + 1) ∧ x ≠ 0}

noncomputable def B_set (a b : Fin 2n → ℝ) : Set ℝ :=
  {x | ∃ j : Fin 2n, x = ∑ i : Fin 2n, a i * b j / (a i * b j + 1) ∧ x ≠ 0}

theorem max_min_A_union_B (n : ℕ) (a b : Fin 2n → ℝ)
  (h_sum_a : ∑ i : Fin 2n, a i = n)
  (h_sum_b : ∑ j : Fin 2n, b j = n)
  (h_nonneg_a : ∀ i : Fin 2n, 0 ≤ a i)
  (h_nonneg_b : ∀ j : Fin 2n, 0 ≤ b j)
  : max (min (A_set a b) ∪ (B_set a b)) = n / 2 :=
  sorry

end max_min_A_union_B_l183_183852


namespace find_a_of_complex_eq_l183_183069

theorem find_a_of_complex_eq (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * (⟨1, -a⟩ : ℂ) = 2) : a = 1 :=
by
  sorry

end find_a_of_complex_eq_l183_183069


namespace no_intersection_of_lines_l183_183243

theorem no_intersection_of_lines :
  ¬ ∃ (s v : ℝ) (x y : ℝ),
    (x = 1 - 2 * s ∧ y = 4 + 6 * s) ∧
    (x = 3 - v ∧ y = 10 + 3 * v) :=
by {
  sorry
}

end no_intersection_of_lines_l183_183243


namespace player2_wins_Y2KGame_l183_183921

-- Definition of the Y2K Game grid and rules.
def Y2KGame : Type := ℕ → option Char 

-- A function to check if a given state contains 'SOS'.
def contains_SOS (g : Y2KGame) : Prop := 
  ∃ i : ℕ, g i = some 'S' ∧ g (i + 1) = some 'O' ∧ g (i + 2) = some 'S'

-- Conditions as an inductive type.
inductive Y2KGame_conditions : Y2KGame -> Prop
| initial (g : Y2KGame) : 
    (∀ i : ℕ, i < 2000 -> g i = none) -> Y2KGame_conditions g

-- Definition capturing the draw scenario (all boxes filled without 'SOS').
def is_draw (g : Y2KGame) : Prop := 
  ∀ i : ℕ, i < 2000 -> (g i ≠ none) ∧ ¬contains_SOS g

-- Definition of the Y2K Game, Player 2's winning strategy.
def player2_winning_strategy (g : Y2KGame) : Prop :=
  ∀ g' : Y2KGame, Y2KGame_conditions g' →
    (contains_SOS g' ∨ is_draw g') → ∃ g'' : Y2KGame, contains_SOS g''

-- The main theorem statement.
theorem player2_wins_Y2KGame : 
  ∀ g : Y2KGame, Y2KGame_conditions g → player2_winning_strategy g :=
begin
  admit,
end

end player2_wins_Y2KGame_l183_183921


namespace kite_area_correct_l183_183731

open Real

structure Point where
  x : ℝ
  y : ℝ

def Kite (p1 p2 p3 p4 : Point) : Prop :=
  let triangle_area (a b c : Point) : ℝ :=
    abs (0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)))
  triangle_area p1 p2 p4 + triangle_area p1 p3 p4 = 102

theorem kite_area_correct : ∃ (p1 p2 p3 p4 : Point), 
  p1 = Point.mk 0 10 ∧ 
  p2 = Point.mk 6 14 ∧ 
  p3 = Point.mk 12 10 ∧ 
  p4 = Point.mk 6 0 ∧ 
  Kite p1 p2 p3 p4 :=
by
  sorry

end kite_area_correct_l183_183731


namespace mixed_groups_count_l183_183988

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183988


namespace smallest_area_right_triangle_l183_183198

theorem smallest_area_right_triangle (a b : ℕ) (h1 : a = 6) (h2 : b = 8)
  (c : ℝ) (h3 : c^2 = a^2 + b^2) :
  let A_leg := 1/2 * a * b
  let B_leg := 1/2 * a * (real.sqrt (b^2 - a^2))
  min A_leg B_leg = 6 * real.sqrt 7 :=
by {
  -- Proof goes here
  sorry
}

end smallest_area_right_triangle_l183_183198


namespace ai_squared_eq_ad_mul_ae_l183_183602

theorem ai_squared_eq_ad_mul_ae {A B C D E I : Point} (hTriangleABC : Triangle A B C)
  (hI : Incenter I A B C) (hDEparallelBC: Parallel D E B C)
  (hDE_tangent_to_incircle : Tangent D E (Incircle I A B C))
  (hDE_intersects_circumcircle : Intersects D E (Circumcircle A B C)) :
  (AI^2 = AD * AE) :=
  sorry

end ai_squared_eq_ad_mul_ae_l183_183602


namespace area_of_right_triangle_l183_183125

theorem area_of_right_triangle (O1 O2 : Point) (r1 r2 : ℝ) (A B S : Point)
  (hA : dist O1 S = r1 * sqrt 2)
  (hB : dist O2 S = r2 * sqrt 2)
  (h_perp : ∠(tangent_line A S) (tangent_line B S) = π / 2)
  (h_outer_tangent : S ∈ (external_tangent (circle O1 r1) (circle O2 r2))) :
  triangle_area A B S = r1 * r2 :=
by
  sorry

end area_of_right_triangle_l183_183125


namespace mixed_groups_count_l183_183997

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183997


namespace sequence_general_term_l183_183742

def Sn (n : ℕ) : ℕ := 4 * n ^ 2 + 2 * n

theorem sequence_general_term :
  ∀ (n : ℕ), n ≥ 1 → let a_n :=
    if n = 1 then Sn 1 else Sn n - Sn (n - 1)
  in a_n = 8 * n - 2 :=
by
  sorry

end sequence_general_term_l183_183742


namespace jerusha_earnings_l183_183844

variable (L : ℝ) 

theorem jerusha_earnings (h1 : L + 4 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l183_183844


namespace find_projection_vector_l183_183298

variables {t s : ℝ}

/-- Line k parametrization -/
def line_k (t : ℝ) : ℝ × ℝ :=
(2 + 3 * t, 3 + 2 * t)

/-- Line n parametrization -/
def line_n (s : ℝ) : ℝ × ℝ :=
(1 + 3 * s, 5 + 2 * s)

/-- Point C on line k -/
def point_C (t : ℝ) : ℝ × ℝ :=
line_k t

/-- Point D on line n -/
def point_D (s : ℝ) : ℝ × ℝ :=
line_n s

/-- Vector DC -/
def vector_DC (t s : ℝ) : ℝ × ℝ :=
(point_C t).sub (point_D s)

/-- Normal vector to line n -/
def normal_vector_n : ℝ × ℝ :=
(-2, 3)

/-- Projection vector satisfying w1 + w2 = 3 -/
theorem find_projection_vector
  (w : ℝ × ℝ)
  (hw : w.fst + w.snd = 3) :
  w = (-6, 9) :=
sorry

end find_projection_vector_l183_183298


namespace GF_perp_DE_l183_183827

section Geometry

variables {A B C D E F G : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F] [inhabited G]

noncomputable def isosceles_triangle (A B C : Type) [inhabited A] [inhabited B] [inhabited C] : Prop :=
  dist A B = dist A C

variables {dist : ∀ {X Y : Type} [inhabited X] [inhabited Y], X → Y → Real}

noncomputable def on_extension_of (D AB : Type) [inhabited D] [inhabited AB] : Prop := sorry -- Implementation depends on formal definition in geometry library.

noncomputable def intersection_point (DE BC F : Type) [inhabited DE] [inhabited BC] [inhabited F] : Prop := sorry -- Implementation depends on formal definition in geometry library.

noncomputable def circle_passing_through (pts : list Type) [∀ T, inhabited T] : Prop := sorry -- Implementation depends on formal definition in geometry library.

variables (isosceles_triangle ABC)
variables (on_extension_of D AB)
variables (on_segment E AC)
variables (dist C E = dist B D)
variables (intersection_point DE BC F)
variables (circle_passing_through [B, D, F])
variables (circle_passing_through [A, B, C])

theorem GF_perp_DE : ⊥ (line G F) (line D E) :=
  sorry

end Geometry

end GF_perp_DE_l183_183827


namespace find_a5_l183_183017

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 2) ∧ (∀ n, a (n + 1) = a n + 2)

theorem find_a5 (a : ℕ → ℕ) (h : sequence a) : a 5 = 10 :=
by
  sorry

end find_a5_l183_183017


namespace solution_set_of_inequality_l183_183032

variable {f : ℝ → ℝ}

-- Condition: Domain of f(x) is (0, +∞)
def dom_f : Set ℝ := Set.Ioi 0

-- Condition: f(x) + x * f''(x) > 0
def cond_f (x : ℝ) : Prop := f(x) + x * f'' x > 0

theorem solution_set_of_inequality :
  (∀ x ∈ dom_f, cond_f x) →
  (∀ x, (x-1) * f (x^2-1) < f (x+1) ↔ 1 < x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l183_183032


namespace hexahedron_volume_l183_183090

open Real

noncomputable def volume_of_hexahedron (AB A1B1 AA1 : ℝ) : ℝ :=
  let S_base := (3 * sqrt 3 / 2) * AB^2
  let S_top := (3 * sqrt 3 / 2) * A1B1^2
  let h := AA1
  (1 / 3) * h * (S_base + sqrt (S_base * S_top) + S_top)

theorem hexahedron_volume : volume_of_hexahedron 2 3 (sqrt 10) = 57 * sqrt 3 / 2 := by
  sorry

end hexahedron_volume_l183_183090


namespace sum_of_exponentials_eq_neg_one_l183_183664

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_exponentials_eq_neg_one : 
  (omega + omega^2 + omega^3 + ⋯ + omega^16) = -1 :=
by
  have h1 : omega ^ 17 = 1 := by 
    unfold omega
    -- Proof of omega^17 = 1 would go here
    sorry
  have h2 : omega ^ 16 = omega⁻¹ := by
    rw [← Complex.exp_nat_mul]
    -- Proof of omega^16 = omega⁻¹ would go here
    sorry
  -- Proof that the sum equals -1 would go here
  sorry

end sum_of_exponentials_eq_neg_one_l183_183664


namespace line_y_intercept_eq_line_distance_eq_l183_183770

theorem line_y_intercept_eq (a : ℝ) (l : ∀ x y : ℝ, x + (a + 1)*y + 2 - a = 0) : 
  (∀ y₀ : ℝ, l 0 y₀ = 0 → y₀ = 2) → (x - 3*y + 6 = 0) :=
by sorry

theorem line_distance_eq (a : ℝ) (l : ∀ x y : ℝ, x + (a + 1)*y + 2 - a = 0) :
  (∀ d : ℝ, d = 1 → 
    d = abs(2 - a) / sqrt(1^2 + (a + 1)^2)) → (3*x + 4*y + 5 = 0) :=
by sorry

end line_y_intercept_eq_line_distance_eq_l183_183770


namespace ratio_twice_width_to_length_l183_183951

theorem ratio_twice_width_to_length (L W : ℝ) (k : ℤ)
  (h1 : L = 24)
  (h2 : W = 13.5)
  (h3 : L = k * W - 3) :
  2 * W / L = 9 / 8 := by
  sorry

end ratio_twice_width_to_length_l183_183951


namespace find_m_plus_b_l183_183949

-- Given conditions as definitions
def point1 : (ℝ × ℝ) := (1, 1)
def point2 : (ℝ × ℝ) := (9, 5)

-- Define the midpoint calculation
def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the slope calculation
def slope (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the slope of the line of reflection as the negative reciprocal
def reflection_slope (m : ℝ) : ℝ := -1 / m

-- State what needs to be proven
theorem find_m_plus_b (m b : ℝ) (H1 : point2 = (2 * midpoint point1 point2).1 - point1.1, (2 * midpoint point1 point2).2 - point1.2)
                      (H2 : m = reflection_slope (slope point1 point2))
                      (H3 : midpoint point1 point2 = (5, 3))
                      (H4 : point1.2 = m * point1.1 + b ∧ point2.2 = m * point2.1 + b) :
  m + b = 11 :=
sorry

end find_m_plus_b_l183_183949


namespace length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l183_183826

noncomputable def spring_length (x : ℝ) : ℝ :=
  2 * x + 18

-- Problem (1)
theorem length_at_4kg : (spring_length 4) = 26 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (2)
theorem length_increases_by_2 : ∀ (x y : ℝ), y = x + 1 → (spring_length y) = (spring_length x) + 2 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (3)
theorem relationship_linear : ∃ (k b : ℝ), (∀ x, spring_length x = k * x + b) ∧ k = 2 ∧ b = 18 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (4)
theorem length_at_12kg : (spring_length 12) = 42 :=
  by
    -- The complete proof is omitted.
    sorry

end length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l183_183826


namespace simplify_expression_l183_183653

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183653


namespace distance_from_O_to_plane_l183_183610

-- Definitions and conditions
variable (O : Point) (radius : ℝ) (O_plane_distance : ℝ) (triangle_sides : ℝ × ℝ × ℝ)
variable (tangent_to_sphere : Bool)

-- Given conditions
def sphere_center_O (A : Point) (B : Point) (C : Point) : Prop :=
  radius = 5 ∧ triangle_sides = (13, 13, 10) ∧ tangent_to_sphere ∧
  plane_of_triangle_tangent_to_sphere (triangle_sides, radius) 

-- Proof goal
theorem distance_from_O_to_plane : sphere_center_O O → O_plane_distance = (5 * sqrt 5) / 3 :=
  by
    sorry

end distance_from_O_to_plane_l183_183610


namespace time_to_walk_2_miles_l183_183060

/-- I walked 2 miles in a certain amount of time. -/
def walked_distance : ℝ := 2

/-- If I maintained this pace for 8 hours, I would walk 16 miles. -/
def pace_condition (pace : ℝ) : Prop :=
  pace * 8 = 16

/-- Prove that it took me 1 hour to walk 2 miles. -/
theorem time_to_walk_2_miles (t : ℝ) (pace : ℝ) (h1 : walked_distance = pace * t) (h2 : pace_condition pace) :
  t = 1 :=
sorry

end time_to_walk_2_miles_l183_183060


namespace find_m_l183_183804

theorem find_m (m : ℝ) : (∀ x > 0, x^2 - 2 * (m^2 + m + 1) * Real.log x ≥ 1) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end find_m_l183_183804


namespace solution_set_of_inequality_l183_183174

theorem solution_set_of_inequality (x : ℝ) : 
  (|x - 1| + |x - 2| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by sorry

end solution_set_of_inequality_l183_183174


namespace part1_monotonicity_for_a_eq1_part2_inequality_for_a_eq_minus3_l183_183410

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - a * Real.sin x - 1

theorem part1_monotonicity_for_a_eq1 :
  ∀ x ∈ Ioo (-π / 2) π, x < 0 → 
  let g := λ x, (Real.exp x - Real.sin x - 1) / Real.exp x in g x < g (x + 1) :=
sorry

theorem part2_inequality_for_a_eq_minus3 :
  ∀ x ∈ Ioo 0 (1 / 0), 
  f x (-3) < Real.exp x + x + 1 - 2 * Real.exp (-2 * x) :=
sorry

end part1_monotonicity_for_a_eq1_part2_inequality_for_a_eq_minus3_l183_183410


namespace area_of_polygon_l183_183317

noncomputable def polygonArea : ℝ :=
  let vertices : list (ℝ × ℝ) :=
    [(0,0), (1,0), (2,1), (2,0), (3,0), (3,1), (3,2), (2,2), (2,3), (1,2), (0,2), (0,1)]
  1 -- This represents the mathematically computed area using predefined methods for polygons (to be proven)

theorem area_of_polygon : polygonArea = 2 :=
by
  sorry

end area_of_polygon_l183_183317


namespace chicken_wings_per_person_l183_183242

theorem chicken_wings_per_person (friends total_wings wings_per_person : ℕ)
  (h_friends : friends = 9)
  (h_total_wings : total_wings = 2 + 25)
  (h_wings_per_person : wings_per_person = total_wings / friends) :
  wings_per_person = 3 :=
  by {
    have h_total : total_wings = 27,
    calc total_wings = 2 + 25 : by rw [h_total_wings]
                  ... = 27    : by norm_num,
    rw [h_total] at h_wings_per_person,
    have h_friends' : friends = 9 := by norm_num,
    norm_num at h_friends',
    rw [h_friends'],
    exact h_wings_per_person,
  }

end chicken_wings_per_person_l183_183242


namespace expression_equals_41_l183_183799

theorem expression_equals_41 (x : ℝ) (h : 3*x^2 + 9*x + 5 ≠ 0) : 
  (3*x^2 + 9*x + 15) / (3*x^2 + 9*x + 5) = 41 :=
by
  sorry

end expression_equals_41_l183_183799


namespace derivative_at_0_l183_183856

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

theorem derivative_at_0 : (derivative f) 0 = 2 := by
  sorry

end derivative_at_0_l183_183856


namespace evaluate_expression_l183_183693

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 :=
by
  sorry

end evaluate_expression_l183_183693


namespace circle_symmetry_l183_183532

theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ x^2 + y^2 = 1) ↔ a = 2) :=
sorry

end circle_symmetry_l183_183532


namespace num_suitable_arrays_l183_183795

theorem num_suitable_arrays : 
  ∃ (count : ℕ), count = 1060 ∧ 
  (∀ (arr : Fin 6 → Fin 6 → ℤ), 
    (∀ i, (∑ j, arr i j) = 0) ∧ 
    (∀ j, (∑ i, arr i j) = 0) ∧ 
    (∀ i j, arr i j = 1 ∨ arr i j = -1) 
    → count = 1060) := sorry

end num_suitable_arrays_l183_183795


namespace repeating_decimal_sum_correct_l183_183332

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  x + y - z

theorem repeating_decimal_sum_correct :
  repeating_decimal_sum = 4 / 9 :=
by
  sorry

end repeating_decimal_sum_correct_l183_183332


namespace range_of_k_l183_183415

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := λ x, 4 * x / (x^2 + 1)
noncomputable def g (k : ℝ) : ℝ → ℝ := λ x, cos (2 * real.pi * x) + k * cos (real.pi * x)

-- State the theorem given the identified conditions
theorem range_of_k (k : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g k x2) → (k ≥ 2 * real.sqrt 2 ∨ k ≤ -2 * real.sqrt 2) :=
by
  sorry

end range_of_k_l183_183415


namespace sequence_explicit_formula_l183_183724

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n + Math.sqrt (a (n + 1) + a n)

theorem sequence_explicit_formula (n : ℕ) : a n = (n * (n + 3)) / 2 + 1 :=
by sorry

end sequence_explicit_formula_l183_183724


namespace harmonic_alternating_sum_l183_183560

theorem harmonic_alternating_sum (n : ℕ) (h : n ≥ 1) :
  (Finset.range (2 * n)).sum (λ k, if k % 2 = 0 then -(1 : ℚ) / (k + 1) else 1 / (k + 1)) =
  (Finset.range (n + 1)).sum (λ k, 1 / (n + 1 + k) : ℚ) :=
sorry

end harmonic_alternating_sum_l183_183560


namespace cos2α_values_l183_183370

variable {α β : ℝ}

-- Conditions
def cond1 (h : α) (h' : β) : Prop := sin α = 2 * sin β
def cond2 (h : α) (h' : β) : Prop := tan α = 3 * tan β

-- Statement of the problem
theorem cos2α_values (α β : ℝ) (h1 : cond1 α β) (h2 : cond2 α β) : cos (2 * α) = -1/4 ∨ cos (2 * α) = 1 :=
sorry

end cos2α_values_l183_183370


namespace range_of_a_l183_183050

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Lean statement for the problem
theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : -1 < a ∧ a ≤ 1 := 
by
  -- Proof is skipped
  sorry

end range_of_a_l183_183050


namespace cara_total_debt_l183_183579

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem cara_total_debt :
  let P := 54
  let R := 0.05
  let T := 1
  let I := simple_interest P R T
  let total := P + I
  total = 56.7 :=
by
  sorry

end cara_total_debt_l183_183579


namespace find_minimal_distance_point_l183_183337

open Real

noncomputable def distances_sum (P: Point) (A B C: Point): ℝ := 
  dist P A + dist P B + dist P C

def minimal_distances_sum (P: Point) (A B C: Point): Prop :=
  ∀ Q: Point, distances_sum P A B C ≤ distances_sum Q A B C

theorem find_minimal_distance_point :
  ∃ P: Point, minimal_distances_sum P ⟨(0, 0)⟩ ⟨(2, 0)⟩ ⟨(0, sqrt 3)⟩ ∧ P = ⟨1, (5.5 * sqrt 3 - 3) / 13⟩ :=
sorry

end find_minimal_distance_point_l183_183337


namespace polynomial_is_square_of_quadratic_l183_183131

theorem polynomial_is_square_of_quadratic (a b : ℚ) :
  (∃ A B : ℚ, x^4 + x^3 + 2x^2 + a*x + b = (x^2 + A*x + B)^2) →
  b = 49/64 :=
by
  sorry

end polynomial_is_square_of_quadratic_l183_183131


namespace number_of_ordered_17_tuples_l183_183719

noncomputable def countOrdered17Tuples : ℕ :=
  let count := 12378
  count

theorem number_of_ordered_17_tuples (a : Fin 17 → ℤ) (S : ℤ) :
  (∀ i, (a i)^2 = S - (a i)) →
  (S = ∑ i, a i) →
  ∃ l, l = 12378 :=
by
  intro h1 h2
  use 12378
  sorry

end number_of_ordered_17_tuples_l183_183719


namespace log_pow_two_l183_183805

theorem log_pow_two (x : ℝ) : log 101600 = 2 * log 102 :=
by
  have h : 101600 = 102 * 102,
  { norm_num, },
  rw [h, log_mul],
  norm_num,
  sorry

end log_pow_two_l183_183805


namespace domain_of_f_parity_of_f_inverse_of_f_range_of_f_gt_zero_l183_183772

noncomputable def f (x : ℝ) : ℝ := log ((x + 5) / (x - 5))

theorem domain_of_f :
  (∃ x : ℝ, x > 5 ∨ x < -5) ↔ ∀ x : ℝ, x > 5 ∨ x < -5 → x ∈ domain f := 
sorry

theorem parity_of_f :
  ∀ x : ℝ, f (-x) = -f x :=
sorry

noncomputable def f_inv (y : ℝ) : ℝ := 5 * (10^y + 1) / (10^y - 1)

theorem inverse_of_f :
  ∀ y : ℝ, y ≠ 0 → (f_inv (f y) = y ∧ f (f_inv y) = y) :=
sorry

theorem range_of_f_gt_zero :
  ∀ x : ℝ, f x > 0 ↔ x > 5 :=
sorry

end domain_of_f_parity_of_f_inverse_of_f_range_of_f_gt_zero_l183_183772


namespace mixed_groups_count_l183_183980

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183980


namespace proof_equation_analogy_l183_183401

theorem proof_equation_analogy
    (h : ∀ x : ℝ, 3^x + 4^x = 5^x ↔ x = 2) :
    ∀ x : ℝ, 3^x + 4^x + 5^x = 6^x ↔ x = 3 :=
by {
    sorry
}

end proof_equation_analogy_l183_183401


namespace total_birds_remaining_l183_183550

theorem total_birds_remaining (grey_birds_in_cage : ℕ) (white_birds_next_to_cage : ℕ) :
  (grey_birds_in_cage = 40) →
  (white_birds_next_to_cage = grey_birds_in_cage + 6) →
  (1/2 * grey_birds_in_cage = 20) →
  (1/2 * grey_birds_in_cage + white_birds_next_to_cage = 66) :=
by 
  intros h_grey_birds h_white_birds h_grey_birds_freed
  sorry

end total_birds_remaining_l183_183550


namespace base6_addition_sum_l183_183431

theorem base6_addition_sum 
  (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : Q ≠ R) 
  (h3 : P ≠ R) 
  (h4 : P < 6) 
  (h5 : Q < 6) 
  (h6 : R < 6) 
  (h7 : 2*R % 6 = P) 
  (h8 : 2*Q % 6 = R)
  : P + Q + R = 7 := 
  sorry

end base6_addition_sum_l183_183431


namespace part_1_part_2_l183_183052

variables (a b : ℝ × ℝ)

def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

axiom given_conditions :
  a = (1, -2) ∧ 
  vector_magnitude b = 2 * real.sqrt 5 ∧
  (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) • (2 * a.1 + b.1, 2 * a.2 + b.2) = -20

theorem part_1 : 
  a.1 * b.2 - a.2 * b.1 = 0 → 
  (b = (-2, 4) ∨ b = (2, -4)) :=
by 
  sorry

theorem part_2 : 
  (b = (-2, 4) ∨ b = (2, -4)) → 
  ∃ θ, cos θ = -1 / 2 ∧ 
  θ = 2 * real.pi / 3 :=
by 
  sorry

end part_1_part_2_l183_183052


namespace mixed_groups_count_l183_183994

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183994


namespace common_difference_of_arithmetic_progression_l183_183704

theorem common_difference_of_arithmetic_progression (a T₁₀ : ℤ) (d : ℤ) :
  a = 8 ∧ T₁₀ = 26 → d = 2 :=
by
  intros h
  cases h with ha hT₁₀ sorry

end common_difference_of_arithmetic_progression_l183_183704


namespace mean_median_difference_l183_183821

/-
In a history exam:
- 15% of students scored 60 points
- 20% of students scored 75 points
- 25% of students scored 85 points
- 10% of students scored 90 points
- the remaining students scored 100 points.
Prove that the difference between the mean and the median score is 0.75.
-/

noncomputable def difference_mean_median : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℚ :=
λ total_students percentage_60 percentage_75 percentage_85 percentage_90 percentage_100 score_60 score_75 score_85 score_90 score_100,
  let students_60 := total_students * percentage_60 / 100 in
  let students_75 := total_students * percentage_75 / 100 in
  let students_85 := total_students * percentage_85 / 100 in
  let students_90 := total_students * percentage_90 / 100 in
  let students_100 := total_students * percentage_100 / 100 in
  let mean_score := ((score_60 * students_60 + score_75 * students_75 + score_85 * students_85 + score_90 * students_90 + score_100 * students_100) : ℚ) / total_students in
  let median_score := score_85 in -- Median as determined since it was between 20th and 21st students
  abs (mean_score - median_score)

theorem mean_median_difference :
  difference_mean_median 40 15 20 25 10 30 60 75 85 90 100 = 0.75 := 
sorry

end mean_median_difference_l183_183821


namespace max_value_frac_inv_sum_l183_183860

theorem max_value_frac_inv_sum (x y : ℝ) (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b)
  (h3 : a^x = 6) (h4 : b^y = 6) (h5 : a + b = 2 * Real.sqrt 6) :
  ∃ m, m = 1 ∧ (∀ x y a b, (1 < a) → (1 < b) → (a^x = 6) → (b^y = 6) → (a + b = 2 * Real.sqrt 6) → 
  (∃ n, (n = (1/x + 1/y)) → n ≤ m)) :=
by
  sorry

end max_value_frac_inv_sum_l183_183860


namespace finite_set_cardinality_l183_183496

-- Define the main theorem statement
theorem finite_set_cardinality (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ)
  (hm : m ≥ 2)
  (hB : ∀ k : ℕ, k ∈ Finset.range m.succ → (B k).sum id = m^k) :
  A.card ≥ m / 2 := 
sorry

end finite_set_cardinality_l183_183496


namespace sum_geom_seq_product_range_l183_183767

noncomputable def ge_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^n

lemma geom_seq_sum_range (a₂ a₅ : ℝ) (n : ℕ) (h₁ : a₂ = 1) (h₂ : a₅ = 1 / 8) :
  ∃ r : ℝ, a₅ = a₂ * r^3 ∧
  r = 1 / 2 ∧
  ∃ a₁ : ℝ, a₁ = a₂ / r ∧
  ∃ a₃ : ℝ, a₃ = a₂ * r ∧
  ∃ S : ℝ, S = (a₁ * a₂) * ((1 - (1 / 4) ^ n) / (1 - 1 / 4)) ∧
  S ∈ set.Ico (2 : ℝ) (8 / 3) :=
by 
  have r : ℝ := 1 / 2,
  have a₁ : ℝ = a₂ / r,
  have a₃ : ℝ = a₂ * r,
  have S : ℝ = (a₁ * a₂) * ((1 - (1 / 4) ^ n) / (1 - 1 / 4)),
  use r,
  exact ⟨h₂, rfl, rfl, h₁ / rfl, h₁ * rfl, S, sorry⟩

theorem sum_geom_seq_product_range (a₂ a₅ : ℝ) (n : ℕ) (h₁ : a₂ = 1) (h₂ : a₅ = 1 / 8) : 
  ∃ S : ℝ, S ∈ set.Ico (2 : ℝ) (8 / 3) :=
  geom_seq_sum_range a₂ a₅ n h₁ h₂

end sum_geom_seq_product_range_l183_183767


namespace equal_segments_l183_183092

theorem equal_segments (A B C D P Q : Point)
  (h_triangle_ABC : ∠ACB = 90°)
  (h_perpendicular : Perpendicular C D AB)
  (h_angle_bisector_ADC : ∠ADC / 2 intersects AC at P)
  (h_angle_bisector_BDC : ∠BDC / 2 intersects BC at Q) :
  distance C P = distance C Q := 
sorry

end equal_segments_l183_183092


namespace passing_marks_l183_183575

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end passing_marks_l183_183575


namespace balanced_partition_of_weights_l183_183252

-- Defining the problem statement
def valid_distribution (weights : Fin 100 → ℝ) : Prop :=
  ∀ (A B : Finset (Fin 100)), A.card = 50 → B.card = 50 → (A ∪ B = Finset.univ) → A ∩ B = ∅ →
  abs (Finset.sum A weights - Finset.sum B weights) ≤ 20

theorem balanced_partition_of_weights :
  ∃ (A B : Finset (Fin 100)), A.card = 50 ∧ B.card = 50 ∧ (A ∪ B = Finset.univ) ∧ A ∩ B = ∅ ∧
  (∀ i j, abs (weights i - weights j) ≤ 20) → abs (Finset.sum A weights - Finset.sum B weights) ≤ 20 :=
sorry

end balanced_partition_of_weights_l183_183252


namespace relationship_between_a_b_c_l183_183011

noncomputable def a : ℝ := 81 ^ 31
noncomputable def b : ℝ := 27 ^ 41
noncomputable def c : ℝ := 9 ^ 61

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_between_a_b_c_l183_183011


namespace sum_of_16_roots_of_unity_l183_183658

theorem sum_of_16_roots_of_unity : 
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17) in
  ∑ i in finset.range 16, ω^(i+1) = -1 :=
by
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17)
  have h₁ : ω^17 = 1 := sorry
  have h₂ : ∑ i in finset.range 17, ω^i = 0 := sorry
  sorry

end sum_of_16_roots_of_unity_l183_183658


namespace solution_set_l183_183494

-- Define the odd function f
def f (x : ℝ) : ℝ := sorry  -- this is a placeholder

-- f is odd: f(-x) = -f(x) for all x
axiom f_odd : ∀ x, f (-x) = -f x

-- f(1) = 1
axiom f_at_1 : f 1 = 1

-- Define the derivative f'(x)
noncomputable def f' (x : ℝ) : ℝ := sorry  -- this is a placeholder

-- When x > 0, f(x) + x f'(x) > 1/x
axiom f_deriv_cond : ∀ x, x > 0 → f x + x * f' x > 1 / x

-- Main theorem to prove the solution set of the inequality x f(x) > 1 + ln |x|
theorem solution_set (x : ℝ) : (x f x > 1 + Real.log (| x |)) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 1) :=
by
  sorry

end solution_set_l183_183494


namespace mixed_groups_count_l183_183976

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183976


namespace incorrect_option_D_l183_183369

noncomputable theory

-- Define the arithmetic sequence and its sum S_n
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d
def sum_of_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := n * a + (n * (n - 1) / 2) * d

-- Given conditions
variables {a d : ℤ}
axiom S5_lt_S6 : sum_of_arithmetic_sequence a d 5 < sum_of_arithmetic_sequence a d 6
axiom S6_eq_S7_gt_S8 : sum_of_arithmetic_sequence a d 6 = sum_of_arithmetic_sequence a d 7 ∧ sum_of_arithmetic_sequence a d 7 > sum_of_arithmetic_sequence a d 8

-- Statement to prove
theorem incorrect_option_D : sum_of_arithmetic_sequence a d 9 < sum_of_arithmetic_sequence a d 5 :=
sorry

end incorrect_option_D_l183_183369


namespace number_of_conditions_for_parallel_planes_l183_183360

open EuclideanGeometry

-- Given conditions as variables in Lean
variables (M N Q : Plane) 
variable (l m : Line) 

-- Condition 1: Both M and N are perpendicular to plane Q
axiom h1 : M ⟂ Q ∧ N ⟂ Q

-- Condition 2: Both M and N are parallel to plane Q
axiom h2 : M ∥ Q ∧ N ∥ Q

-- Condition 3: Three non-collinear points in plane M are equidistant from plane N
axiom h3 : ∃ (A B C : Point), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ∈ M ∧ B ∈ M ∧ C ∈ M ∧ distance A N = distance B N ∧ distance B N = distance C N

-- Condition 4: Line l is outside plane M, and m consists of two lines within plane M, and \(l \parallel M\), \(m \parallel N\)
axiom h4 : l ∉ M ∧ ∃ (a b : Line), a ∈ M ∧ b ∈ M ∧ l ∥ M ∧ (a ∥ N ∧ b ∥ N)

-- Condition 5: Lines l and m are skew lines, \(l \parallel M\), \(m \parallel M\), \(l \parallel N\), \(m \parallel N\)
axiom h5 : skew l m ∧ l ∥ M ∧ m ∥ M ∧ l ∥ N ∧ m ∥ N

-- Proof statement
theorem number_of_conditions_for_parallel_planes (M N Q : Plane) (l m : Line) (h1 : M ⟂ Q ∧ N ⟂ Q)
  (h2 : M ∥ Q ∧ N ∥ Q) (h3 : ∃ (A B C : Point), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ∈ M ∧ B ∈ M ∧ C ∈ M ∧ distance A N = distance B N ∧ distance B N = distance C N)
  (h4 : l ∉ M ∧ ∃ (a b : Line), a ∈ M ∧ b ∈ M ∧ l ∥ M ∧ (a ∥ N ∧ b ∥ N)) (h5 : skew l m ∧ l ∥ M ∧ m ∥ M ∧ l ∥ N ∧ m ∥ N) : 
  ∃ (C : ℕ), (C = 2) := 
by
  sorry

end number_of_conditions_for_parallel_planes_l183_183360


namespace common_terms_sum_l183_183351

noncomputable def arithmetic_progression (n : ℕ) : ℕ :=
  5 + 3 * n

noncomputable def geometric_progression (k : ℕ) : ℕ :=
  20 * 2^k

theorem common_terms_sum : ∑ i in (Finset.range 10), (arithmetic_progression i) = 6990500 :=
sorry

end common_terms_sum_l183_183351


namespace count_numbers_including_digit_3_l183_183428

-- Definitions of the relevant sets based on conditions identified
def A : Finset ℕ := Finset.filter (λ n, (n / 100) % 10 = 3) (Finset.range' 200 300)
def B : Finset ℕ := Finset.filter (λ n, (n / 10) % 10 = 3) (Finset.range' 200 300)
def C : Finset ℕ := Finset.filter (λ n, n % 10 = 3) (Finset.range' 200 300)

-- Lean statement proving the required count
theorem count_numbers_including_digit_3 : 
  Finset.card (A ∪ B ∪ C) = 138 := by
  sorry

end count_numbers_including_digit_3_l183_183428


namespace probability_b_greater_than_a_l183_183901

open Finset

def setA : Finset ℕ := {1, 2, 3, 4, 5}
def setB : Finset ℕ := {1, 2, 3}

def total_outcomes : ℕ := setA.card * setB.card

def favorable_outcomes : ℤ := ((setA.product setB).filter (λ p => p.2 > p.1)).card

theorem probability_b_greater_than_a :
  (favorable_outcomes : ℚ) / total_outcomes = (1 : ℚ) / 5 :=
by
  sorry

end probability_b_greater_than_a_l183_183901


namespace total_coins_last_month_l183_183280

theorem total_coins_last_month (m s : ℝ) : 
  (100 = 1.25 * m) ∧ (100 = 0.80 * s) → m + s = 205 :=
by sorry

end total_coins_last_month_l183_183280


namespace sum_valid_x_eq_95_by_6_l183_183358

-- Definitions
def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x
def ceil (x : ℝ) : ℤ := Int.ceil x

def valid_x (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 6 ∧ (floor x : ℝ) * frac x * (ceil x : ℝ) = 1 ∧ frac x ≠ 0

-- Main theorem
theorem sum_valid_x_eq_95_by_6 : 
  let xs := {x : ℝ | valid_x x} in
  let ms := xs.to_list.sort (λ x y => x < y) in
  xs.card = 5 → 
  list.sum ms = 95 / 6 :=
by sorry

end sum_valid_x_eq_95_by_6_l183_183358


namespace total_minutes_of_game_and_ceremony_l183_183607

-- Define the components of the problem
def game_hours : ℕ := 2
def game_additional_minutes : ℕ := 35
def ceremony_minutes : ℕ := 25

-- Prove the total minutes is 180
theorem total_minutes_of_game_and_ceremony (h: game_hours = 2) (ga: game_additional_minutes = 35) (c: ceremony_minutes = 25) :
  (game_hours * 60 + game_additional_minutes + ceremony_minutes) = 180 :=
  sorry

end total_minutes_of_game_and_ceremony_l183_183607


namespace work_completion_time_l183_183577

theorem work_completion_time 
    (A B : ℝ) 
    (h1 : A = 2 * B) 
    (h2 : (A + B) * 18 = 1) : 
    1 / A = 27 := 
by 
    sorry

end work_completion_time_l183_183577


namespace pens_given_to_Sharon_l183_183574

variable (initial_pens mike_pens doubled_pens final_pens given_to_sharon : ℕ)

def number_of_pens_given (initial_pens mike_pens doubled_pens final_pens : ℕ) : ℕ :=
  (initial_pens + mike_pens) * 2 - final_pens

theorem pens_given_to_Sharon :
  initial_pens = 7 → 
  mike_pens = 22 → 
  doubled_pens = (initial_pens + mike_pens) * 2 → 
  final_pens = 39 → 
  given_to_sharon = number_of_pens_given initial_pens mike_pens doubled_pens final_pens → 
  given_to_sharon = 19 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h3
  dsimp [number_of_pens_given]
  rw h3
  dsimp
  linarith

end pens_given_to_Sharon_l183_183574


namespace probability_distance_condition_l183_183552

def cylinder_base_radius := 1
def cylinder_height := 3

def O1 := (0, 0, cylinder_height)
def O2 := (0, 0, 0)

def point_in_cylinder (P : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := P
  x^2 + y^2 ≤ cylinder_base_radius^2 ∧ 0 ≤ z ∧ z ≤ cylinder_height

def distance (P1 P2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2 + (P1.3 - P2.3)^2)

def distance_condition (P : ℝ × ℝ × ℝ) : Prop :=
  distance P O1 > 1 ∧ distance P O2 > 1

theorem probability_distance_condition : ∀ (P : ℝ × ℝ × ℝ),
  point_in_cylinder P → 
  ∃ (probability : ℝ), (probability = x) := 
sorry

end probability_distance_condition_l183_183552


namespace interval_of_decrease_l183_183033

theorem interval_of_decrease (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x) :
  ∀ x0 : ℝ, ∀ x1 : ℝ, x0 ≥ 3 → x0 ≤ x1 → f (x1 - 3) ≤ f (x0 - 3) := sorry

end interval_of_decrease_l183_183033


namespace find_abc_l183_183395

variables {a b c : ℕ}

theorem find_abc (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : abc ∣ ((a * b - 1) * (b * c - 1) * (c * a - 1))) : a = 2 ∧ b = 3 ∧ c = 5 :=
by {
    sorry
}

end find_abc_l183_183395


namespace circle_intersection_line_l183_183135

theorem circle_intersection_line :
  let c1 := (∃ x y : ℝ, (x + 4)^2 + (y + 10)^2 = 225)
  let c2 := (∃ x y : ℝ, (x - 8)^2 + (y - 6)^2 = 104)
  ∀ x y : ℝ, (c1 ∧ c2) → x + y = -2 :=
by
  intros x y hc
  cases hc with h1 h2
  -- Proof goes here
  sorry

end circle_intersection_line_l183_183135


namespace range_of_m_l183_183778

noncomputable def f : ℝ → ℝ := λ x, Real.sin x - x

theorem range_of_m (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2)
  (h : f (Real.cos θ ^ 2 + 2 * θ * Real.sin θ) + f (-2 - 2 * θ) > 0) :
  θ ≥ -1 / 2 :=
sorry

end range_of_m_l183_183778


namespace find_angle_CBO_l183_183618

theorem find_angle_CBO :
  ∀ (BAO CAO CBO ABO ACO BCO AOC : ℝ), 
  BAO = CAO → 
  CBO = ABO → 
  ACO = BCO → 
  AOC = 110 →
  CBO = 20 :=
by
  intros BAO CAO CBO ABO ACO BCO AOC hBAO_CAOC hCBO_ABO hACO_BCO hAOC
  sorry

end find_angle_CBO_l183_183618


namespace project_completion_time_l183_183254

-- Define the time taken for the first part of the project
def time_first_part (days : ℕ) : Prop :=
  days = 30

-- Define the team size and work done in the first part
def team_size (people : ℕ) : Prop :=
  people = 8

def work_done_first_part (fraction : ℚ) : Prop :=
  fraction = 2 / 3

-- Define the additional team members added
def additional_people (people : ℕ) : Prop :=
  people = 4

-- Define the time to complete the remaining work
def time_remaining_work (days : ℕ) : Prop :=
  days = 10

-- Define the total time to complete the project
def total_time (days : ℕ) : Prop :=
  days = 40

-- Theorem statement combining all conditions to achieve the proof
theorem project_completion_time :
  ∃ (days_first_part days_remaining total_days : ℕ)
    (initial_team additional_team : ℕ)
    (work_fraction : ℚ),
    time_first_part days_first_part ∧
    team_size initial_team ∧
    work_done_first_part work_fraction ∧
    additional_people additional_team ∧
    time_remaining_work days_remaining ∧
    total_time total_days ∧
    total_days = days_first_part + days_remaining :=
begin
  sorry
end

end project_completion_time_l183_183254


namespace trigonometric_identity_l183_183960

theorem trigonometric_identity :
  (1 / Real.cos 80) - (Real.sqrt 3 / Real.cos 10) = 4 :=
by
  sorry

end trigonometric_identity_l183_183960


namespace integer_pairs_count_l183_183344

theorem integer_pairs_count :
  let f (x : ℤ) := log 5 x
  let g (x : ℤ) := 110 + x - 5 ^ 110
  let s : ℤ := (1 + 5 ^ 110) * 5 ^ 110 / 2 - (5 ^ 111 - 445) / 4
  ∃ pairs : ℤ × ℤ,
  (∀ x y, (y ≥ 110 + x - 5 ^ 110) ∧ (y ≤ log 5 x)) →
  (pairs = s) :=
by
  sorry

end integer_pairs_count_l183_183344


namespace Kyle_monthly_income_l183_183101

theorem Kyle_monthly_income :
  let rent := 1250
  let utilities := 150
  let retirement_savings := 400
  let groceries_eatingout := 300
  let insurance := 200
  let miscellaneous := 200
  let car_payment := 350
  let gas_maintenance := 350
  rent + utilities + retirement_savings + groceries_eatingout + insurance + miscellaneous + car_payment + gas_maintenance = 3200 :=
by
  -- Informal proof was provided in the solution.
  sorry

end Kyle_monthly_income_l183_183101


namespace difference_in_spending_l183_183305

-- Condition: original prices and discounts
def original_price_candy_bar : ℝ := 6
def discount_candy_bar : ℝ := 0.25
def original_price_chocolate : ℝ := 3
def discount_chocolate : ℝ := 0.10

-- The theorem to prove
theorem difference_in_spending : 
  (original_price_candy_bar * (1 - discount_candy_bar) - original_price_chocolate * (1 - discount_chocolate)) = 1.80 :=
by
  sorry

end difference_in_spending_l183_183305


namespace mixed_groups_count_l183_183986

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183986


namespace ball_distribution_l183_183427

theorem ball_distribution (n k : ℕ) (h_n : n = 6) (h_k : k = 3) :
  ∃ (count : ℕ), count = 540 ∧
    (∑ (S in Finset.partitions n), (if S.card = k then 1 else 0)) * k! = count :=
by
  let valid_distributions := (∑ S in Finset.partitions n, if S.card = k then 1 else 0) * Nat.factorial k
  have : valid_distributions = 540 :=
    sorry -- Proof
  use [valid_distributions, this] 
  exact ⟨this, rfl⟩

end ball_distribution_l183_183427


namespace probability_distance_less_than_one_inside_triangle_l183_183839

theorem probability_distance_less_than_one_inside_triangle 
  (A B C : Point) 
  (h_equilateral : is_equilateral_triangle A B C 2) : 
    (∃ P : Point, inside_triangle A B C P ∧ 
    (dist P A < 1 ∨ dist P B < 1 ∨ dist P C < 1)) → 
  (probability (λ P, dist P A < 1 ∨ dist P B < 1 ∨ dist P C < 1) inside_triangle_region) = (π * √3) / 9 :=
sorry

end probability_distance_less_than_one_inside_triangle_l183_183839


namespace value_of_a_compare_f_values_range_of_f_l183_183765

-- Definitions of conditions
def f (a x : ℝ) : ℝ := a ^ x

-- 1. Prove that the value of \( a \) is \( \frac{1}{3} \)
theorem value_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a 2 = 1/9) : a = 1/3 :=
sorry

-- 2. Prove that \( f(2) \geq f(b^2 + 2) \) given the value of \( a \)
theorem compare_f_values (a b : ℝ) (h_value_of_a : a = 1/3) : f a 2 ≥ f a (b^2 + 2) :=
sorry

-- 3. Prove the range of the function \( f(x) = a^{x^2 - 2x} \) for \( x \geq 0 \)
theorem range_of_f (a : ℝ) (h_value_of_a : a = 1/3) : (∀ x ≥ 0, 0 < f a (x^2 - 2*x) ∧ f a (x^2 - 2*x) ≤ 3) :=
sorry

end value_of_a_compare_f_values_range_of_f_l183_183765


namespace find_ingrid_tax_rate_l183_183848

-- Define the given conditions
def john_tax_rate : ℝ := 0.30
def john_income : ℝ := 56000
def ingrid_income : ℝ := 74000
def combined_tax_rate : ℝ := 0.3569

-- Define the amount of tax paid by John
def john_tax : ℝ := john_tax_rate * john_income

-- Calculate the combined income
def combined_income : ℝ := john_income + ingrid_income

-- Define the total tax paid
def total_tax_paid : ℝ := combined_tax_rate * combined_income

-- Calculate the tax paid by Ingrid
def ingrid_tax : ℝ := total_tax_paid - john_tax

-- Define Ingrid's tax rate
def ingrid_tax_rate : ℝ := ingrid_tax / ingrid_income

-- The theorem to be proved
theorem find_ingrid_tax_rate : ingrid_tax_rate ≈ 0.40 := by
  sorry

end find_ingrid_tax_rate_l183_183848


namespace repeating_decimal_three_digits_l183_183302

theorem repeating_decimal_three_digits (N : ℕ) (hN : N < 1000) :
  (N = 37 ∨ N = 296) ↔ (real.of_rat (↑N / 999)).to_decimal =  ("0." ++ "037" ++ "..." ∨ "0." ++ "296" ++ "...") := sorry

end repeating_decimal_three_digits_l183_183302


namespace rectangle_length_width_l183_183250

theorem rectangle_length_width (x y : ℝ) (h1 : 2 * (x + y) = 26) (h2 : x * y = 42) : 
  (x = 7 ∧ y = 6) ∨ (x = 6 ∧ y = 7) :=
by
  sorry

end rectangle_length_width_l183_183250


namespace dot_product_AD_BC_l183_183527

variable (A B C D O : Point)
variable (vec_a vec_b : Vector)
variable (AB CD AO BO OC OD AD BC : ℝ)

-- Conditions
-- AB = 101
axiom condition1 : AB = 101

-- CD = 20
axiom condition2 : CD = 20

-- AO and OC relationship from similar triangles
axiom condition3 : OC = (20 / 101) * AO

-- BO and OD relationship from similar triangles
axiom condition4 : OD = (20 / 101) * BO

-- Perpendicular diagonals imply dot product is zero
axiom condition5 : vec_a.dot vec_b = 0

-- Pythagorean theorem on the right-angled triangle AOB
axiom condition6 : vec_a.norm_sq + vec_b.norm_sq = AB^2

-- Define vectors AD and BC in terms of AO and BO
def AD : Vector := vec_a + ((20 / 101) * vec_b)
def BC : Vector := vec_b + ((20 / 101) * vec_a)

-- The theorem we want to prove
theorem dot_product_AD_BC : AD.dot(BC) = 2020 :=
by
  -- Since it's a statement only, we just add sorry to indicate the proof is not present.
  sorry

end dot_product_AD_BC_l183_183527


namespace solution_l183_183692

noncomputable def problem_statement : Prop :=
  sqrt (7 + 4 * sqrt 3) - sqrt (7 - 4 * sqrt 3) = 2 * sqrt 3

theorem solution : problem_statement :=
by
  sorry -- Proof placeholder

end solution_l183_183692


namespace petes_original_number_l183_183891

theorem petes_original_number :
  ∃ x : ℤ, 5 * (3 * x - 5) = 200 ∧ x = 15 :=
by
  existsi (15 : ℤ)
  split
  · calc
      5 * (3 * 15 - 5) = 5 * (45 - 5) : by ring
                      ... = 5 * 40 : by ring
                      ... = 200 : by norm_num
  · rfl

end petes_original_number_l183_183891


namespace range_m_satisfies_conditions_l183_183012

def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)
def g (x : ℝ) : ℝ := 2^x - 2

theorem range_m_satisfies_conditions :
  ∀ m : ℝ,
    (-4 < m ∧ m < -2) ↔
      (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧
      (∀ x : ℝ, x < -4 → f m x * g x < 0) :=
begin
  sorry
end

end range_m_satisfies_conditions_l183_183012


namespace KirpichWithdrawThreeCards_KirpichProbabilityAllFourCards_l183_183888

-- Constants based on the conditions
constant numCards : ℕ := 4
constant numTries : ℕ := 3
constant numPINS : ℕ := 4

-- Define the results expected in both problems
-- Part (a): Show that Kirpich can withdraw money from three cards
theorem KirpichWithdrawThreeCards (cards : Fin numCards → Fin numPINS) (pin_attempt : Fin numCards → Fin numTries → Fin numPINS) :
  ∃ (used_cards : Set (Fin numCards)), used_cards.card ≥ 3 ∧
  ∀ (c : Fin numCards) (p : Fin numPINS), c ∈ used_cards → p ∉ pin_attempt c ' (Fin.mk 2 sorry) := sorry

-- Part (b): Probability calculation
open ProbabilityTheory

noncomputable def probability_success_all_cards : ℚ := (1/4) * (1/3) * (1/2) * 1

theorem KirpichProbabilityAllFourCards :
  probability_success_all_cards = 1 / 24 := sorry

end KirpichWithdrawThreeCards_KirpichProbabilityAllFourCards_l183_183888


namespace max_abs_z_correct_l183_183436

noncomputable def max_abs_z (z : ℂ) : ℝ :=
  if |z + 3 + 4 * ℂ.I| ≤ 2 then 7 else 0

theorem max_abs_z_correct : ∀ (z : ℂ), |z + 3 + 4 * ℂ.I| ≤ 2 → |z| ≤ 7 :=
by
  intro z h
  have h_dist : |z + 3 + 4*ℂ.I| ≤ 2 := h
  -- Proof steps would go here (hypothetically)
  -- Using sorry to skip the actual proof
  sorry

end max_abs_z_correct_l183_183436


namespace expression_simplification_l183_183636

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183636


namespace erase_proper_divisors_l183_183881

-- Define the proper divisor concept
def proper_divisors (n : ℕ) : set ℕ := {d | d ∣ n ∧ d ≠ 1 ∧ d ≠ n}

-- Define the main theorem
theorem erase_proper_divisors (a b : ℕ) (ha : ∃ p q, a = p * q ∧ 1 < p ∧ 1 < q) (hb : ∃ r s, b = r * s ∧ 1 < r ∧ 1 < s) 
  (h_no_common_proper_divisors : (proper_divisors a) ∩ (proper_divisors b) = ∅) : 
  ∃ (erase_A : set ℕ) (erase_B : set ℕ), 
    (erase_A ⊆ proper_divisors a ∧ erase_A.card ≤ (proper_divisors a).card / 2) ∧ 
    (erase_B ⊆ proper_divisors b ∧ erase_B.card ≤ (proper_divisors b).card / 2) ∧
    ∀ c ∈ (proper_divisors a \ erase_A), ∀ d ∈ (proper_divisors b \ erase_B), ¬ (a + b) ∣ (c + d) := 
begin
  -- Proof omitted
  sorry
end

end erase_proper_divisors_l183_183881


namespace interval_monotonicity_max_value_inequality_g_local_min_l183_183777

-- (1) Prove the intervals of monotonicity for f(x)
theorem interval_monotonicity (a : ℝ) (hapos : a > 0) :
  (∀ x : ℝ, -2 * a < x ∧ x < 1 / a - 2 * a → deriv (λ x, Real.log (x + 2 * a) - a * x) x > 0) ∧
  (∀ x : ℝ, x > 1 / a - 2 * a → deriv (λ x, Real.log (x + 2 * a) - a * x) x < 0) :=
sorry

-- (2) Given M(a), prove the inequality given c_1 and c_2
theorem max_value_inequality (a_1 a_2 : ℝ) (hapos1 : a_1 > 0) (hapos2 : a_2 > 0)
  (hineq : a_2 > a_1) (heq : 2 * a_1^2 - 1 - Real.log a_1 = 2 * a_2^2 - 1 - Real.log a_2) :
  a_1 * a_2 < 1 / 4 :=
sorry

-- (3) Prove x_0 is a local minimum of g(x) given a > 2
theorem g_local_min (a : ℝ) (hapos : a > 2) (x_0 : ℝ)
  (hx0 : (λ x, Real.log (x + 2 * a) - a * x) x_0 = 0) :
  ∀ x : ℝ, -2 * a < x ∧ x < 1 / a - 2 * a → (λ x, |Real.log (x + 2 * a) - a * x| + x) x_0 ≤ (λ x, |Real.log (x + 2 * a) - a * x| + x) x :=
sorry

end interval_monotonicity_max_value_inequality_g_local_min_l183_183777


namespace gas_cost_correct_l183_183900

/-- Define the initial odometer reading --/
def initial_odometer : ℕ := 74580

/-- Define the final odometer reading --/
def final_odometer : ℕ := 74610

/-- Define the car mileage (miles per gallon) --/
def car_mileage : ℕ := 25

/-- Define the gas price per gallon --/
def gas_price : ℝ := 4.20

/-- Define a function to calculate the cost of the trip's gas --/
def gas_cost (initial : ℕ) (final : ℕ) (mileage : ℕ) (price : ℝ) : ℝ :=
  let distance := final - initial
  let gallons_used := (distance : ℝ) / (mileage : ℝ)
  gallons_used * price

/-- Prove that the cost of gas for the trip is $5.04 --/
theorem gas_cost_correct : gas_cost initial_odometer final_odometer car_mileage gas_price = 5.04 :=
by
  sorry

end gas_cost_correct_l183_183900


namespace infinite_rationals_approx_l183_183863

theorem infinite_rationals_approx (x : ℝ) (hx : Irrational x ∧ x > 0) : 
  ∃ᶠ (p q : ℕ) in at_top, (q > 0) ∧ (|x - (p / q : ℝ)| < 1 / q^2) :=
sorry

end infinite_rationals_approx_l183_183863


namespace divide_octagon_into_parallelograms_l183_183520

structure Octagon (α : Type) [LinearOrder α] :=
(vertices : Fin₈ → α × α)
(convex : ∀ i j k : Fin₈, 
  (vertices i).1 * ((vertices j).2 - (vertices k).2) + 
  (vertices j).1 * ((vertices k).2 - (vertices i).2) + 
  (vertices k).1 * ((vertices i).2 - (vertices j).2) ≥ 0)
(parallel_and_equal : ∀ i : ℕ, 
  ((vertices (Fin.mk i (by linarith [i % 8, Nat.mod_lt i (by norm_num : 0 < 8)]))).fst) 
  = ((vertices (Fin.mk (i+4) (by linarith [i+4 % 8, Nat.mod_lt (i+4) (by norm_num : 0 < 8)]))).fst) ∧
  ((vertices (Fin.mk i (by linarith [i % 8, Nat.mod_lt i (by norm_num : 0 < 8)]))).snd) 
  = ((vertices (Fin.mk (i+4) (by linarith [i+4 % 8, Nat.mod_lt (i+4) (by norm_num : 0 < 8)])])).snd))

theorem divide_octagon_into_parallelograms (O : Octagon ℝ) : 
  ∃ P : Fin₄ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ), 
  ∀ i : Fin₄, parallelogram (P i) := sorry

end divide_octagon_into_parallelograms_l183_183520


namespace mixed_groups_count_l183_183989

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183989


namespace mixed_groups_count_l183_183999

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l183_183999


namespace determine_y_l183_183429

theorem determine_y :
  (10 ^ (Real.log10 16) = 10 * y + 6) → y = 1 :=
by
  sorry

end determine_y_l183_183429


namespace min_marbles_to_draw_l183_183233

theorem min_marbles_to_draw (reds greens blues yellows oranges purples : ℕ)
  (h_reds : reds = 35)
  (h_greens : greens = 25)
  (h_blues : blues = 24)
  (h_yellows : yellows = 18)
  (h_oranges : oranges = 15)
  (h_purples : purples = 12)
  : ∃ n : ℕ, n = 103 ∧ (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r < 20 ∧ g < 20 ∧ b < 20 ∧ y < 20 ∧ o < 20 ∧ p < 20 → r + g + b + y + o + p < n) ∧
      (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r + g + b + y + o + p = n → r = 20 ∨ g = 20 ∨ b = 20 ∨ y = 20 ∨ o = 20 ∨ p = 20) :=
sorry

end min_marbles_to_draw_l183_183233


namespace seats_not_occupied_l183_183324

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l183_183324


namespace minimum_value_of_linear_expression_l183_183803

theorem minimum_value_of_linear_expression :
  ∀ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 → 2 * x + y ≥ -5 :=
by
  sorry

end minimum_value_of_linear_expression_l183_183803


namespace rachel_picked_total_apples_l183_183515

-- Define the conditions
def num_trees : ℕ := 4
def apples_per_tree_picked : ℕ := 7
def apples_remaining : ℕ := 29

-- Define the total apples picked
def total_apples_picked : ℕ := num_trees * apples_per_tree_picked

-- Formal statement of the goal
theorem rachel_picked_total_apples : total_apples_picked = 28 := 
by
  sorry

end rachel_picked_total_apples_l183_183515


namespace find_b_c_l183_183382

-- Definitions and the problem statement
theorem find_b_c (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = 1) (h2 : x2 = -2) 
  (h_eq : ∀ x, x^2 - b * x + c = (x - x1) * (x - x2)) :
  b = -1 ∧ c = -2 :=
by
  sorry

end find_b_c_l183_183382


namespace additive_function_property_l183_183851

theorem additive_function_property {a : ℝ} {n : ℕ} (f : ℕ → ℝ → ℝ)
  (h_add : ∀ i, (x y : ℝ), f i (x + y) = f i x + f i y)
  (h_mult : ∀ x : ℝ, (∏ i in finset.range n, f i x) = a * x ^ n) :
  ∃ (b : ℝ) (i : ℕ), i < n ∧ (∀ x : ℝ, f i x = b * x) := 
sorry

end additive_function_property_l183_183851


namespace investment_of_D_l183_183622

/--
Given C and D started a business where C invested Rs. 1000 and D invested some amount.
They made a total profit of Rs. 500, and D's share of the profit is Rs. 100.
So, how much did D invest in the business?
-/
theorem investment_of_D 
  (C_invested : ℕ) (D_share : ℕ) (total_profit : ℕ) 
  (H1 : C_invested = 1000) 
  (H2 : D_share = 100) 
  (H3 : total_profit = 500) 
  : ∃ D : ℕ, D = 250 :=
by
  sorry

end investment_of_D_l183_183622


namespace planes_parallel_if_line_perpendicular_to_both_l183_183749

section PlanesAndLines

variables {Point : Type} [AffineSpace Point ℝ]
variable {a : Line Point}
variables {α β : Plane Point}

-- Condition definitions
def perp_to_plane (l : Line Point) (p : Plane Point) : Prop :=
∀ (P1 P2 : Point), P1 ≠ P2 → P1 ∈ p → P2 ∈ p → ∀ A ∈ l, ∃ B ∈ l, B ≠ A ∧ B ∈ line_through P1 P2

def parallel_to_plane (p q : Plane Point) : Prop :=
∀ (P1 P2 : Point), P1 ≠ P2 → P1 ∈ p → P2 ∈ p → ∃ (Q1 Q2 : Point), Q1 ≠ Q2 ∧ Q1 ∈ q ∧ Q2 ∈ q ∧ line_through P1 P2 = line_through Q1 Q2

-- The theorem that needs proving
theorem planes_parallel_if_line_perpendicular_to_both (h1 : perp_to_plane a α) (h2 : perp_to_plane a β) :
  parallel_to_plane α β :=
sorry

end PlanesAndLines

end planes_parallel_if_line_perpendicular_to_both_l183_183749


namespace bees_flew_in_l183_183182

theorem bees_flew_in (initial_bees additional_bees total_bees : ℕ) 
  (h1 : initial_bees = 16) (h2 : total_bees = 25) 
  (h3 : initial_bees + additional_bees = total_bees) : additional_bees = 9 :=
by sorry

end bees_flew_in_l183_183182


namespace circle_tangent_incicle_l183_183741

open EuclideanGeometry

-- Define a scalene (no equal sides) acute-angled (all angles < 90°) triangle ABC
variable {ABC : Triangle}
variable {a b c : ℝ}
variable {p r : ℝ}

-- Define the points K, D, M, N
variable (K D M N : Point)

-- Define the incircle I of triangle ABC, touching at BC at K
variable {I : Circle}
variable (H1 : I.Touches ABC.BC K)

-- Define AD as the altitude from A to BC and M as the midpoint of AD
variable (H2 : PerpendicularFromLineAndPoint AD ABC.BC ABC.A D)
variable (H3 : Midpoint M AD)

-- Define KM extended to intersect I at N
variable (H4 : LineThroughAndIntersectCircle KM I N)

-- Prove: The circle passing through points B, N, C is tangent to incircle I
theorem circle_tangent_incicle :
  Tangent (Circumcircle B N C) I :=
sorry

end circle_tangent_incicle_l183_183741


namespace simplify_expression_l183_183654

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183654


namespace distance_center_to_point_l183_183623

noncomputable def circle_center (x y : ℝ) (h : x^2 + y^2 = 6 * x - 2 * y - 15) : ℝ × ℝ :=
  (3, -1)

noncomputable def distance_to_point (cx cy px py : ℝ) : ℝ :=
  real.sqrt ((px - cx)^2 + (py - cy)^2)

theorem distance_center_to_point : 
  (distance_to_point 3 (-1) (-2) 5) = real.sqrt 61 := by
  sorry

end distance_center_to_point_l183_183623


namespace degree_of_divisor_l183_183606

theorem degree_of_divisor (f q r d : Polynomial ℝ)
  (h_f : f.degree = 15)
  (h_q : q.degree = 9)
  (h_r : r = Polynomial.C 5 * X^4 + Polynomial.C 3 * X^3 - Polynomial.C 2 * X^2 + Polynomial.C 9 * X - Polynomial.C 7)
  (h_div : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_l183_183606


namespace all_rationals_as_sum_of_q_n_l183_183895

theorem all_rationals_as_sum_of_q_n :
  ∀ (r : ℚ), ∃ (n : ℕ) (a : ℕ → ℚ), (∀ i, a i = (i - 1) / (i + 2) ∧ n ≥ 0) ∧ 
            r = ∑ i in Finset.range n, a i :=
by
  sorry

end all_rationals_as_sum_of_q_n_l183_183895


namespace min_side_length_of_cube_l183_183379

-- Defining the main constants
def radius : ℝ := 1
def cube_min_side_length := 4

-- Formalizing the problem statement in Lean
theorem min_side_length_of_cube (s : ℝ) :
  (∀ (sphere_1 sphere_2 : ℝ), sphere_1 = radius ∧ sphere_2 = radius 
  → s = cube_min_side_length) :=
begin
  sorry
end

end min_side_length_of_cube_l183_183379


namespace choir_row_lengths_l183_183239

theorem choir_row_lengths : 
  let n := 96 in
  let valid_row_lengths := {d ∈ finset.Icc 5 20 | n % d = 0} in
  finset.card valid_row_lengths = 4 :=
by 
  let n := 96
  let valid_row_lengths := {d ∈ finset.Icc 5 20 | n % d = 0}
  have : finset.card valid_row_lengths = 4
  from sorry
  this

end choir_row_lengths_l183_183239


namespace max_pairs_distance_at_most_n_l183_183079

theorem max_pairs_distance_at_most_n
  (n : ℕ)
  (h_n : n ≥ 3)
  (points : fin n → ℝ × ℝ)
  (d : ℝ)
  (h_d : ∀ i j, i ≠ j → dist (points i) (points j) ≤ d)
  (h_max_d : ∃ i j, i ≠ j ∧ dist (points i) (points j) = d) :
  ∃ pairs : finset (fin n × fin n), 
    (∀ p ∈ pairs, dist (points p.1) (points p.2) = d) ∧ pairs.card ≤ n := 
sorry

end max_pairs_distance_at_most_n_l183_183079


namespace valentine_cards_l183_183133

theorem valentine_cards (x y : ℕ) (h : x * y = x + y + 18) : x * y = 40 :=
by
  sorry

end valentine_cards_l183_183133


namespace area_of_inscribed_triangle_l183_183257

noncomputable def triangle_area : ℝ := 
  (50 / (Real.pi^2)) * (1 + Real.sin (54 * Real.pi / 180) + Real.sin (36 * Real.pi / 180))

theorem area_of_inscribed_triangle :
  (let r := 10 / Real.pi 
   let θ := 18 * Real.pi / 180 in 
   let A := 90 * Real.pi / 180
   let B := 126 * Real.pi / 180
   let C := 144 * Real.pi / 180 in 
   (1 / 2) * r^2 * (Real.sin A + Real.sin B + Real.sin C) = triangle_area) := 
by 
  sorry

end area_of_inscribed_triangle_l183_183257


namespace samia_walk_distance_l183_183518

/-- Given the conditions of Samia's journey: cycling at 15 km/h for half the distance,
     and walking at 6 km/h for the other half, with a total travel time of 36 minutes,
     prove that the distance Samia walked is 2.6 kilometers. -/
theorem samia_walk_distance :
  ∃ (x : ℝ), x = 2.6 ∧ (
    ∀ (d : ℝ), 
    (d = 2 * x) → 
    (let t_bike := d / 2 / 15 in
     let t_walk := d / 2 / 6 in
     t_bike + t_walk = 36 / 60)
  ) :=
begin
  use 2.6,
  split,
  { refl },
  { 
    intros d hd,
    rw hd,
    let t_bike := 2.6 / 15,
    let t_walk := 2.6 / 6,
    have h : t_bike + t_walk = 36 / 60,
    { sorry },
    exact h
  }
end

end samia_walk_distance_l183_183518


namespace centipede_sock_shoe_order_l183_183595

theorem centipede_sock_shoe_order :
  ∃ (n : ℕ), n = 10 ∧ (∀ (m : ℕ), m = 20! / 2^10) :=
by
  existsi 10
  splithypothesis
  existsi (20!/2^10)
  sorry

end centipede_sock_shoe_order_l183_183595


namespace distance_to_asymptote_l183_183925

noncomputable def distance_from_asymptote : ℝ :=
  let x0 := 3
  let y0 := 0
  let A := 3
  let B := -4
  let C := 0
  @Real.sqrt ((A^2) + (B^2))^{-1} * abs (A * x0 + B * y0 + C)

theorem distance_to_asymptote : distance_from_asymptote = (9 / 5) := sorry

end distance_to_asymptote_l183_183925


namespace two_digit_pairs_with_reversed_sum_and_difference_l183_183175

theorem two_digit_pairs_with_reversed_sum_and_difference :
  ∃ (n : ℕ), n = 9 ∧
    ∃ A B : ℕ,
      (A < 100 ∧ B < 100 ∧
      let S := A + B,
      let F := A - B,
      let x := S / 10, let y := S % 10,
      let u := F / 10, let v := F % 10 in
        S < 100 ∧ F < 100 ∧
        S = 10 * x + y ∧
        F = 10 * v + u ∧
        x = v ∧ y = u) := by
    sorry

end two_digit_pairs_with_reversed_sum_and_difference_l183_183175


namespace sum_of_first_10_terms_l183_183026

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  -- The "gn" function represents the sequence "a"
  32 * (1 - (1 / 4)^n) / (1 - (1 / 4))

theorem sum_of_first_10_terms : 
  let a : ℕ → ℕ := λ n, 32 * (1 / 4)^(n - 1)
  (S_n (a) : ℕ → ℕ) := λ n, (32 * (1 - (1 / 4)^n) / (1 - 1 / 4))
  ((S_6 (S_n a)) / (S_3 (S_n a)) = 65 / 64) →
  (sum (λ n, |log_2 a n|) (range 10) = 58) :=
sorry

end sum_of_first_10_terms_l183_183026


namespace ryan_hours_difference_l183_183694

theorem ryan_hours_difference :
  let hours_english := 6
  let hours_chinese := 7
  hours_chinese - hours_english = 1 := 
by
  -- this is where the proof steps would go
  sorry

end ryan_hours_difference_l183_183694


namespace find_number_l183_183422

theorem find_number (n : ℕ) (h : (1 / 2 : ℝ) * n + 5 = 13) : n = 16 := 
by
  sorry

end find_number_l183_183422


namespace total_combined_grapes_l183_183265

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l183_183265


namespace find_length_AP_l183_183828

noncomputable def radius := 1 / 2
def A : ℝ × ℝ := (-1, 1 / 2)
def M : ℝ × ℝ := (0, -1 / 2)
def P : ℝ × ℝ := (1 / 2, 0)

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = radius^2
def line_am (x y : ℝ) : Prop := y = -x + 1 / 2 

theorem find_length_AP :
  (A.1 - P.1)^2 + (A.2 - P.2)^2 = (√10 / 2)^2 := by
  sorry

end find_length_AP_l183_183828


namespace additional_mangoes_l183_183130

variable (original_price : ℝ)
variable (price_reduction : ℝ)
variable (total_amount_spent : ℝ)

-- Define the original number of mangoes Mr. John bought
def original_number_of_mangoes (p : ℝ) : ℝ :=
  total_amount_spent / p

-- Define the new price per mango after reduction
def new_price_per_mango (p : ℝ) (r : ℝ) : ℝ :=
  p - (r * p)

-- Define the new number of mangoes Mr. John can buy at the reduced price
def new_number_of_mangoes (np : ℝ) : ℝ :=
  total_amount_spent / np

-- Prove the number of additional mangoes
theorem additional_mangoes (original_price : ℝ) (price_reduction : ℝ) (total_amount_spent : ℝ)
  (h₁ : original_price = 366.67 / 110) -- original price per mango calculation
  (h₂ : price_reduction = 0.10) -- price reduction by 10%
  (h₃ : total_amount_spent = 360) -- total amount Mr. John spent
  : new_number_of_mangoes (new_price_per_mango original_price price_reduction) - original_number_of_mangoes original_price = 12 := 
sorry

end additional_mangoes_l183_183130


namespace central_cell_value_l183_183075

-- Conditions
def is_arithmetic_progression (a : ℕ) (d : ℕ => ℕ → ℕ) :=
  ∀ n m : ℕ, 1 ≤ n ∧ n ≤ 7 → 1 ≤ m ∧ m ≤ 7 → (d n = a + (n - 1) * d)

def first_row_first_elem := 3
def first_row_last_elem := 143
def last_row_first_elem := 82
def last_row_last_elem := 216

-- To be proven
theorem central_cell_value : 
  ∃ x : ℕ, x = 111 :=
by sorry

end central_cell_value_l183_183075


namespace time_to_complete_job_together_l183_183586

open Real

def work_rate_a : ℝ := 1/10
def work_rate_b : ℝ := 1/15
def combined_work_rate : ℝ := work_rate_a + work_rate_b

theorem time_to_complete_job_together : 1 / combined_work_rate = 6 := by
  sorry

end time_to_complete_job_together_l183_183586


namespace expression_simplification_l183_183640

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183640


namespace proof_f_prime_1_zero_l183_183807

noncomputable def function_f (x : ℝ) (f_prime_1 : ℝ) : ℝ :=
  (1 / 3) * x^3 - f_prime_1 * x^2 - x

theorem proof_f_prime_1_zero :
  ∃ (f_prime_1 : ℝ), (∀ x, deriv (λ x, function_f x f_prime_1) x = x^2 - 2 * f_prime_1 * x - 1) → 
  (deriv (λ x, function_f x f_prime_1) 1 = 0) :=
by
  sorry

end proof_f_prime_1_zero_l183_183807


namespace jakes_present_weight_l183_183064

theorem jakes_present_weight (J S : ℕ) (h1 : J - 32 = 2 * S) (h2 : J + S = 212) : J = 152 :=
by
  sorry

end jakes_present_weight_l183_183064


namespace mixed_groups_count_l183_183963

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183963


namespace power_function_value_l183_183047

theorem power_function_value (a : ℝ) (h : (1/4)^a = 1/2) : 4^a = 2 :=
by
  sorry

end power_function_value_l183_183047


namespace number_of_students_l183_183218

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N) (h2 : (T - 250) / (N - 5) = 90) : N = 20 :=
sorry

end number_of_students_l183_183218


namespace sum_of_elements_in_A_star_B_eq_6_l183_183124

def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}
def star (A B : Set ℤ) : Set ℤ := {x | ∃ (a ∈ A) (b ∈ B), x = a * b}

theorem sum_of_elements_in_A_star_B_eq_6 :
  (∑ x in (star A B), x) = 6 := by
  sorry

end sum_of_elements_in_A_star_B_eq_6_l183_183124


namespace balloon_count_l183_183367

-- Conditions
def Fred_balloons : ℕ := 5
def Sam_balloons : ℕ := 6
def Mary_balloons : ℕ := 7
def total_balloons : ℕ := 18

-- Proof statement
theorem balloon_count :
  Fred_balloons + Sam_balloons + Mary_balloons = total_balloons :=
by
  exact Nat.add_assoc 5 6 7 ▸ rfl

end balloon_count_l183_183367


namespace proof_problem_l183_183775

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem proof_problem :
  (∃ A ω φ, (A = 2) ∧ (ω = 2) ∧ (φ = Real.pi / 4) ∧
  f (3 * Real.pi / 8) = 0 ∧
  f (Real.pi / 8) = 2 ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≤ 2) ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -Real.sqrt 2) ∧
  f (-Real.pi / 4) = -Real.sqrt 2) :=
sorry

end proof_problem_l183_183775


namespace students_in_both_band_and_chorus_l183_183546

-- Definitions of conditions
def total_students := 250
def band_students := 90
def chorus_students := 120
def band_or_chorus_students := 180

-- Theorem statement to prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : 
  (band_students + chorus_students - band_or_chorus_students) = 30 := 
by sorry

end students_in_both_band_and_chorus_l183_183546


namespace countable_inter_set_l183_183221

noncomputable def balls_family (n : ℕ) (h : n > 1) : Type :=
{ S : set (set (euclidean_space ℝ (fin n)))
  // ∀ B₁ B₂ ∈ S, B₁ ≠ B₂ → ∃ x, (B₁ ∩ B₂ = {x}) ∨ (B₁ ∩ B₂ = ∅) }

def intersection_set (n : ℕ) (h : n > 1) : set (euclidean_space ℝ (fin n)) :=
{ x | ∃ (S : set (set (euclidean_space ℝ (fin n)))), x ∈ ⋃ B₁∈S, ⋃ B₂∈S, B₁ ≠ B₂ ∧ x ∈ B₁ ∧ x ∈ B₂ }

theorem countable_inter_set {n : ℕ} (h : n > 1) (S : balls_family n h) :
   countable (intersection_set n h) :=
sorry

end countable_inter_set_l183_183221


namespace triangle_properties_l183_183825

noncomputable theory

variables (a b c A B C : ℝ)

def quadratic_roots : Prop :=
  a^2 - 2 * real.sqrt 3 * a + 2 = 0 ∧
  b^2 - 2 * real.sqrt 3 * b + 2 = 0

def sine_condition : Prop :=
  2 * real.sin (A + B) - real.sqrt 3 = 0

def acute_triangle : Prop :=
  A + B = 120 ∧ C = 60

def cosine_rule : Prop :=
  c^2 = (a + b)^2 - 3 * a * b

def triangle_area : Prop :=
  (1 / 2) * a * b * real.sin C = real.sqrt 3 / 2

theorem triangle_properties
  (h1 : quadratic_roots a b)
  (h2 : sine_condition A B)
  (h3 : acute_triangle A B C)
  (h4 : cosine_rule a b c 60)
  (h5 : triangle_area a b 60) :
  C = 60 ∧ c = real.sqrt 6 ∧ (1 / 2) * a * b * real.sin 60 = real.sqrt 3 / 2 :=
begin
  sorry
end

end triangle_properties_l183_183825


namespace centroid_ineq_l183_183115

variable (A B C O : Type)
variable [has_dist A B C O]
variable [InTriangle O A B C] -- Assuming InTriangle represents O is the centroid of triangle ABC
variable (AB BC CA OA OB OC : ℝ)
variable (H1 : s_1 = OA + OB + OC) 
variable (H2 : s_2 = (1 / 2) * (AB + BC + CA))

theorem centroid_ineq (s_1 s_2 : ℝ) (H1 : s_1 = OA + OB + OC) (H2 : s_2 = (1/2) * (AB + BC + CA)) : 
  s_1 < s_2 :=
sorry

end centroid_ineq_l183_183115


namespace find_a_l183_183013

def is_digit (a : ℕ) : Prop := 1 ≤ a ∧ a ≤ 9

theorem find_a (a : ℕ) (ha : is_digit a) (h : 0.1 * a + 0.00 * a .. = 1 / a) : 
  a = 6 :=
sorry

end find_a_l183_183013


namespace largest_k_l183_183715

-- Definitions
def euler_totient (n : ℕ) : ℕ := 
  n - Finset.card { m : Finset.Icc 1 n | Nat.gcd m n ≠ 1}

def sum_of_divisors (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (λ m => n % m = 0).sum

-- Main statement
theorem largest_k
  (φ : ℕ → ℕ)
  (σ : ℕ → ℕ)
  (phi_def : ∀ n, φ n = euler_totient n)
  (sigma_def : ∀ n, σ n = sum_of_divisors n)
  (h1 : 641 ∣ 2 ^ 32 + 1) :
  ∃ k, (∀ m, φ (σ (2 ^ m)) = 2 ^ m → m ≤ k) ∧ k = 31 :=
by
  sorry

end largest_k_l183_183715


namespace min_value_is_two_l183_183752

noncomputable def min_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 3 * b = 4) : ℝ :=
  Inf (setOf (λ x, x = (1 / (a + 1)) + (3 / (b + 1)))) 

theorem min_value_is_two : ∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 4 → (1 / (a + 1)) + (3 / (b + 1)) = 2 :=
by
  sorry

end min_value_is_two_l183_183752


namespace transformed_avg_and_stddev_l183_183030

variable {n : ℕ}
variable {x : Fin n → ℝ}

def avg (xs : Fin n → ℝ) : ℝ := (Finset.univ.sum xs) / n

def stddev (xs : Fin n → ℝ) : ℝ :=
(real.sqrt (Finset.univ.sum (λ i, (xs i - avg xs) ^ 2))) / real.sqrt n 

theorem transformed_avg_and_stddev (h_avg : avg x = 10) (h_stddev : stddev x = 2) :
  avg (λ i, 2 * x i - 1) = 19 ∧ stddev (λ i, 2 * x i - 1) = 4 :=
by
  sorry

end transformed_avg_and_stddev_l183_183030


namespace juice_left_l183_183597

theorem juice_left (total consumed : ℚ) (h_total : total = 1) (h_consumed : consumed = 4 / 6) :
  total - consumed = 2 / 6 ∨ total - consumed = 1 / 3 :=
by
  sorry

end juice_left_l183_183597


namespace trajectory_of_midpoint_l183_183377

theorem trajectory_of_midpoint (M : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) ∧
  (P.1 = M.1 ∧ P.2 = 2 * M.2) ∧ 
  (N.1 = P.1 ∧ N.2 = 0) ∧ 
  (M.1 = (P.1 + N.1) / 2 ∧ M.2 = (P.2 + N.2) / 2)
  → M.1^2 + 4 * M.2^2 = 1 := 
by
  sorry

end trajectory_of_midpoint_l183_183377


namespace total_cost_of_trip_l183_183355

theorem total_cost_of_trip :
  let adult_count := 5 in
  let child_count := 2 in
  let concessions_cost := 12 in
  let child_ticket_cost := 7 in
  let adult_ticket_cost := 10 in
  let total_adult_cost := adult_count * adult_ticket_cost in
  let total_child_cost := child_count * child_ticket_cost in
  total_adult_cost + total_child_cost + concessions_cost = 76 :=
by
  -- Definitions and conditions
  let adult_count := 5;
  let child_count := 2;
  let concessions_cost := 12;
  let child_ticket_cost := 7;
  let adult_ticket_cost := 10;
  let total_adult_cost := adult_count * adult_ticket_cost;
  let total_child_cost := child_count * child_ticket_cost;
  -- The proof goal
  have : total_adult_cost + total_child_cost + concessions_cost = 76,
    from sorry,
  exact this

end total_cost_of_trip_l183_183355


namespace a3_is_4_l183_183787

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 3
  else seq (n - 1) + 1 / seq (n - 2)

-- Statement to prove
theorem a3_is_4 : seq 3 = 4 :=
by
  -- Proof is to be provided, currently using sorry
  sorry

end a3_is_4_l183_183787


namespace find_x_l183_183119

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - q.1, p.2 + q.2)

theorem find_x (x y : ℤ) :
  star (3, 3) (0, 0) = star (x, y) (3, 2) → x = 6 :=
by
  intro h
  sorry

end find_x_l183_183119


namespace largest_base_sum_digits_l183_183679

-- Define the function to compute sum of digits in a given base
def sum_of_digits (n b : ℕ) : ℕ :=
  let digits := (nat.digits b n) in
  digits.sum

-- Translate the problem conditions to Lean and state the theorem
theorem largest_base_sum_digits (b : ℕ) : 
  (b = 7) ∧ (∀ n, sum_of_digits ((nat.pow (12 % b) 4) b) b ≠ 35) :=
by
  sorry

end largest_base_sum_digits_l183_183679


namespace mixed_groups_count_l183_183965

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183965


namespace probability_two_students_different_classes_l183_183959

theorem probability_two_students_different_classes:
  let students := {1, 2, 3, 'A', 'B'}
  let same_class := {1, 2, 3}
  let diff_class := {'A', 'B'}
  let total_ways := Nat.choose 5 2
  let different_class_ways := 7
  let probability := (different_class_ways: ℚ) / total_ways in
  probability = 0.7 :=
by
  sorry

end probability_two_students_different_classes_l183_183959


namespace tangerines_times_persimmons_l183_183183

-- Definitions from the problem conditions
def apples : ℕ := 24
def tangerines : ℕ := 6 * apples
def persimmons : ℕ := 8

-- Statement to be proved
theorem tangerines_times_persimmons :
  tangerines / persimmons = 18 := by
  sorry

end tangerines_times_persimmons_l183_183183


namespace simplify_expression_eq_square_l183_183628

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183628


namespace marble_pairs_count_l183_183059

-- Definitions of the problem conditions
def my_marbles : Finset ℕ := {1, 2, 3, 4, 5}
def mathew_marbles : Finset ℕ := Finset.range 13 \ {0}

-- Defining the main theorem
theorem marble_pairs_count :
  (card ((my_marbles.product my_marbles).filter (λ p, ∃ x ∈ mathew_marbles, p.1 + p.2 = x + 1))) = 24 :=
sorry

end marble_pairs_count_l183_183059


namespace find_n_l183_183343

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : ∃ k : ℤ, 721 = n + 360 * k): n = 1 :=
sorry

end find_n_l183_183343


namespace smallest_number_condition_l183_183584

theorem smallest_number_condition :
  ∃ n, 
  (n > 0) ∧ 
  (∀ k, k < n → (n - 3) % 12 = 0 ∧ (n - 3) % 16 = 0 ∧ (n - 3) % 18 = 0 ∧ (n - 3) % 21 = 0 ∧ (n - 3) % 28 = 0 → k = 0) ∧
  (n - 3) % 12 = 0 ∧
  (n - 3) % 16 = 0 ∧
  (n - 3) % 18 = 0 ∧
  (n - 3) % 21 = 0 ∧
  (n - 3) % 28 = 0 ∧
  n = 1011 :=
sorry

end smallest_number_condition_l183_183584


namespace increasing_function_a_range_l183_183039

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4 * a * x else (2 * a + 3) * x - 4 * a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end increasing_function_a_range_l183_183039


namespace liya_preferences_l183_183883

theorem liya_preferences (n : ℕ) (hn : n % 3 = 0) (hlast : n % 10 = 0) : ∃ l, l = 0 :=
by {
  use 0,
  sorry
}

end liya_preferences_l183_183883


namespace total_theme_parks_l183_183094

theorem total_theme_parks 
  (J V M N : ℕ) 
  (hJ : J = 35)
  (hV : V = J + 40)
  (hM : M = J + 60)
  (hN : N = 2 * M) 
  : J + V + M + N = 395 :=
sorry

end total_theme_parks_l183_183094


namespace find_a_when_C_on_plane_l183_183763

theorem find_a_when_C_on_plane
  (a : ℝ)
  (P A B C : ℝ × ℝ × ℝ)
  (hP : P = (2, 0, 0))
  (hA : A = (1, -3, 2))
  (hB : B = (8, -1, 4))
  (hC : C = (2 * a + 1, a + 1, 2))
  (hC_on_plane : ∃ (λ₁ λ₂ : ℝ), 
    C = (λ₁ * ((1, -3, 2) - (2, 0, 0)) + λ₂ * ((8, -1, 4) - (2, 0, 0)) + (2, 0, 0))) : 
  a = 16 :=
by
  sorry

end find_a_when_C_on_plane_l183_183763


namespace distance_relation_l183_183830

noncomputable theory

-- Definitions for the problem
def polar_curve_eq (ρ θ : ℝ) : Prop := ρ * (sin θ)^2 = 2 * cos θ
def parametric_line_eq (t : ℝ) : ℝ × ℝ :=
  (-2 - (real.sqrt 2 / 2) * t, -4 - (real.sqrt 2 / 2) * t)
def point_P := (-2, -4)

-- Cartesian equation of curve C
def cartesian_curve_eq (x y : ℝ) : Prop := y^2 = 2 * x

-- Standard equation of line l
def standard_line_eq (x y : ℝ) : Prop := x - y - 2 = 0

-- Distances and points
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
def point_A : ℝ × ℝ := (real.sqrt 5 + 3, real.sqrt 5 + 1)
def point_B : ℝ × ℝ := (3 - real.sqrt 5, 1 - real.sqrt 5)
def PA := distance point_P point_A
def PB := distance point_P point_B
def AB := distance point_A point_B

-- The main proof statement
theorem distance_relation :
  (polar_curve_eq ρ θ) ∧ 
  (parametric_line_eq t = point_P) ∧ 
  (∀ x y, (cartesian_curve_eq x y ↔ polar_curve_eq ρ θ ∧ parametric_line_eq t (x, y))) ∧ 
  (∀ x y, (standard_line_eq x y ↔ parametric_line_eq t = point_P)) ∧ 
  (|PA| * |PB| = |AB|^2) :=
sorry

end distance_relation_l183_183830


namespace area_between_curves_l183_183153

open Real Integral

theorem area_between_curves : 
  (∫ x in (0 : ℝ)..(1 : ℝ), (sqrt x - x)) = (1 / 6) :=
by
  sorry

end area_between_curves_l183_183153


namespace simplify_expression_l183_183647

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183647


namespace number_of_integers_in_interval_is_one_l183_183399

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f' x = 2 * x - 9

def integer_at_origin (f : ℝ → ℝ) : Prop :=
  ∃ n : ℤ, f 0 = n

def number_of_integers_within_interval (f : ℝ → ℝ) : ℕ :=
  {n | ∃ x ∈ Ioo 4 5, f x = n}.finite.to_finset.card

theorem number_of_integers_in_interval_is_one
  (f : ℝ → ℝ) (h_der : quadratic_function f) (h_int_origin : integer_at_origin f) :
  number_of_integers_within_interval f = 1 :=
  sorry

end number_of_integers_in_interval_is_one_l183_183399


namespace mixed_groups_count_l183_183984

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l183_183984


namespace f_1987_eq_5_l183_183802

noncomputable def f : ℕ → ℝ := sorry

axiom f_def : ∀ x : ℕ, x ≥ 0 → ∃ y : ℝ, f x = y
axiom f_one : f 1 = 2
axiom functional_eq : ∀ a b : ℕ, a ≥ 0 → b ≥ 0 → f (a + b) = f a + f b - 3 * f (a * b) + 1

theorem f_1987_eq_5 : f 1987 = 5 := sorry

end f_1987_eq_5_l183_183802


namespace expression_simplification_l183_183641

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183641


namespace martin_average_speed_l183_183877

theorem martin_average_speed:
  let total_distance := 36
  let distance_part1 := 12
  let speed_part1 := 3
  let distance_part2 := 12
  let speed_part2 := 4
  let distance_part3 := 12
  let speed_part3 := 2
  let time_part1 := distance_part1 / speed_part1
  let time_part2 := distance_part2 / speed_part2
  let time_part3 := distance_part3 / speed_part3
  let total_time := time_part1 + time_part2 + time_part3
  let average_speed := total_distance / total_time
  average_speed ≈ 2.77 := sorry

end martin_average_speed_l183_183877


namespace simplify_expression_l183_183521

theorem simplify_expression (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) (h3 : a ≠ 3) :
  (a - 3) / (a^2 - 4) / (1 - 5 / (a + 2)) = 1 / (a - 2) :=
begin
  sorry
end

end simplify_expression_l183_183521


namespace triangle_ABC_right_angled_line_through_A_reciprocal_intercepts_line_l_A_distance_2_l183_183459

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (1, 0)

-- Question (I): Prove triangle ABC is a right-angled triangle
theorem triangle_ABC_right_angled : 
  (1 - (-1)) * (1 - 0) + (2 - 0) * (-1 - 1) = 0 :=
sorry

-- Question (II): Prove the equation of the line passing through A with reciprocal intercepts is correct
def reciprocal_line_1 : ℝ → ℝ × ℝ → Prop := λ a p, (fst p)/a + a*(snd p) = 1
def reciprocal_line_2 := λ x y, 4*x + y + 2 = 0 ∨ x + y - 1 = 0

theorem line_through_A_reciprocal_intercepts :
  (reciprocal_line_1 (-1, 2) a ∧ reciprocal_line_2 a (-1, 2)) :=
sorry

-- Question (III): Prove the equation of line l passing through A at a distance 2 from C
def distance_to_line : ℝ × ℝ → (ℝ → ℝ) → ℝ := 
λ p f, abs ((f (fst p)) - (snd p)) / sqrt ((f'(fst p))^2 + 1)
def line_l : ℝ → (ℝ → ℝ) → Prop := 
  λ k f, ∃ k, f = λ x, k*x + 2 ∨ f = λ x, k*(x + 1)

theorem line_l_A_distance_2 : 
  (distance_to_line (1, 0) (line_l (-1, 2)) = 2) ∧ 
  (line_l (fst A) = (-1, 2) ∧ line_l (snd A) = (2 - x)) :=
sorry

end triangle_ABC_right_angled_line_through_A_reciprocal_intercepts_line_l_A_distance_2_l183_183459


namespace zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l183_183408

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 2 then 2^x + a else a - x

theorem zero_of_f_a_neg_sqrt2 : 
  ∀ x, f x (- Real.sqrt 2) = 0 ↔ x = 1/2 :=
by
  sorry

theorem range_of_a_no_zero :
  ∀ a, (¬∃ x, f x a = 0) ↔ a ∈ Set.Iic (-4) ∪ Set.Ico 0 2 :=
by
  sorry

end zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l183_183408


namespace seats_not_occupied_l183_183329

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_l183_183329


namespace no_intersection_range_k_l183_183406

def problem_statement (k : ℝ) : Prop :=
  ∀ (x : ℝ),
    ¬(x > 1 ∧ x + 1 = k * x + 2) ∧ ¬(x < 1 ∧ -x - 1 = k * x + 2) ∧ 
    (x = 1 → (x + 1 ≠ k * x + 2 ∧ -x - 1 ≠ k * x + 2))

theorem no_intersection_range_k :
  ∀ (k : ℝ), problem_statement k ↔ -4 ≤ k ∧ k < -1 :=
sorry

end no_intersection_range_k_l183_183406


namespace largest_and_smallest_areas_l183_183271

theorem largest_and_smallest_areas (P : ℝ) (hP : 0 < P) :
  let A_circle := P^2 / (4 * Real.pi),
      A_square := P^2 / 16,
      A_triangle := (Real.sqrt 3 * P^2) / 36
  in A_circle > A_square ∧ A_square > A_triangle :=
by
  sorry

end largest_and_smallest_areas_l183_183271


namespace lucy_age_l183_183876

theorem lucy_age (Inez_age : ℕ) (Zack_age : ℕ) (Jose_age : ℕ) (Lucy_age : ℕ) 
  (h1 : Inez_age = 18) 
  (h2 : Zack_age = Inez_age + 4) 
  (h3 : Jose_age = Zack_age - 6) 
  (h4 : Lucy_age = Jose_age + 2) : 
  Lucy_age = 18 := by
sorry

end lucy_age_l183_183876


namespace library_books_difference_l183_183538

theorem library_books_difference (total_books : ℕ) (borrowed_percentage : ℕ) 
  (initial_books : total_books = 400) 
  (percentage_borrowed : borrowed_percentage = 30) :
  (total_books - (borrowed_percentage * total_books / 100)) = 280 :=
by
  sorry

end library_books_difference_l183_183538


namespace third_roll_is_five_l183_183656

noncomputable def probability_of_third_five (fair_die : ℕ → ℚ) (biased_die : ℕ → ℚ) (chosen_die : bool) : ℚ :=
if chosen_die then biased_die 5 else fair_die 5

theorem third_roll_is_five (prob_fair_five : ℚ)
                          (prob_biased_five : ℚ)
                          (prob_chosen_die : ℚ)
                          (prob_fair_die_twice : ℚ)
                          (prob_biased_die_twice : ℚ)
                          (prob_two_fives : prob_fair_die_twice + prob_biased_die_twice = 1)
                          (prob_three_fives : ℚ) :
  let prob_use_fair := prob_fair_die_twice / prob_two_fives in
  let prob_use_biased := prob_biased_die_twice / prob_two_fives in
  prob_three_fives = (prob_use_fair * prob_fair_five + prob_use_biased * prob_biased_five) →
  prob_three_fives = 223/297 :=
begin
  sorry
end

end third_roll_is_five_l183_183656


namespace sum_of_exponentials_eq_neg_one_l183_183663

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_exponentials_eq_neg_one : 
  (omega + omega^2 + omega^3 + ⋯ + omega^16) = -1 :=
by
  have h1 : omega ^ 17 = 1 := by 
    unfold omega
    -- Proof of omega^17 = 1 would go here
    sorry
  have h2 : omega ^ 16 = omega⁻¹ := by
    rw [← Complex.exp_nat_mul]
    -- Proof of omega^16 = omega⁻¹ would go here
    sorry
  -- Proof that the sum equals -1 would go here
  sorry

end sum_of_exponentials_eq_neg_one_l183_183663


namespace simplify_expression_l183_183915

theorem simplify_expression (a : ℤ) (ha : a = -2) : 
  3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a)) = 10 := 
by 
  sorry

end simplify_expression_l183_183915


namespace seven_circle_divisors_exists_non_adjacent_divisors_l183_183903

theorem seven_circle_divisors_exists_non_adjacent_divisors (a : Fin 7 → ℕ)
  (h_adj : ∀ i : Fin 7, a i ∣ a (i + 1) % 7 ∨ a (i + 1) % 7 ∣ a i) :
  ∃ (i j : Fin 7), i ≠ j ∧ j ≠ i + 1 % 7 ∧ j ≠ i + 6 % 7 ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end seven_circle_divisors_exists_non_adjacent_divisors_l183_183903


namespace find_c_l183_183393

variables (x b : ℝ)

def c := x^3 - (1/x)^3  -- Define the value to be computed

theorem find_c (h : x - 1/x = 2 * b) (hb : 2 * b = 3) : c = 36 :=
by
  have h₁ : x - 1/x = 3 := by rw [←hb, h]
  have h₂ : (x - 1/x)^2 = 9 := by rw [h₁]; norm_num
  have h₃ : x^2 - 2 + 1/x^2 = 9 := by rw [←h₂]; ring
  have h₄ : x^2 + 1/x^2 = 11 := by linarith
  have h₅ : x^3 - 1/x^3 = (x - 1/x) * (x^2 + 1 + 1/x^2) := by ring
  have h₆ : x^2 + 1 + 1/x^2 = 12 := by rw [←h₄]; norm_num
  rw [h₆, h₁] at h₅; norm_num at h₅; exact h₅

end find_c_l183_183393


namespace sufficient_but_not_necessary_condition_l183_183530

theorem sufficient_but_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x > 0 ∧ y > 0 → (x / y + y / x ≥ 2)) ∧ ¬((x / y + y / x ≥ 2) → (x > 0 ∧ y > 0)) :=
sorry

end sufficient_but_not_necessary_condition_l183_183530


namespace geometric_sequence_sum_l183_183833

def a (n : ℕ) : ℕ := 3 * (2 ^ (n - 1))

theorem geometric_sequence_sum :
  a 1 = 3 → a 4 = 24 → (a 3 + a 4 + a 5) = 84 :=
by
  intros h1 h4
  sorry

end geometric_sequence_sum_l183_183833


namespace vec_subtraction_l183_183421

-- Definitions
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Condition: a is parallel to b
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Main theorem
theorem vec_subtraction (m : ℝ) (h : are_parallel a (b m)) :
  2 • a - b m = (4, -8) :=
sorry

end vec_subtraction_l183_183421


namespace maximum_value_l183_183800

noncomputable theory

-- Define the function f(x)
def f (x a : ℝ) : ℝ := (x^2 + a * x + 1) * Real.exp x

-- Condition that x = 3 is an extremum point of f(x)
def is_extremum (f : ℝ → ℝ) (x : ℝ) : Prop := (deriv f x = 0)

-- Prove that the maximum value of f(x) is 6e^{-1} given the conditions
theorem maximum_value (a : ℝ) (h_extremum : is_extremum (λ x, f x a) 3)
: ∃ x, f x a = 6 * Real.exp (-1) :=
sorry

end maximum_value_l183_183800


namespace median_of_data_set_l183_183743

noncomputable def data_set := [12, 5, 9, 5, 14]

def median (l : List ℝ) : ℝ :=
let sorted := l.qsort (· < ·)
in sorted.nthLe (sorted.length / 2) (by dec_trivial) -- middle element

theorem median_of_data_set : median data_set = 9 := by
  sorry

end median_of_data_set_l183_183743


namespace good_n_tuples_l183_183853

theorem good_n_tuples (n : ℕ) (a : Fin n → ℕ) :
  (∀ i, a i > 0) →
  (∑ i, a i = 2 * n) →
  (∀ s : Finset (Fin n), s.sum a ≠ n) →
  (∀ i, (a i = 2) ∨ (a i = 1) ∨ (∃ j, a j = n + 1)) :=
by sorry

end good_n_tuples_l183_183853


namespace swap_black_gray_pieces_l183_183158

def initial_state : List (Option String) := [some "B", some "B", none, some "G", some "G", none, none, none, none, none]

def swapped_state : List (Option String) := [none, none, none, none, none, some "B", some "B", some "B", some "G", some "G"]

noncomputable def can_swap (initial final : List (Option String)) : Prop :=
  ∃ moves : List (List (Option String)), moves.head = initial ∧ moves.last = final ∧ 
    (∀ i, i < moves.length - 1 → is_valid_move moves.nth_le i (i + 1))

def is_valid_move (current next : List (Option String)) : Prop :=
  -- Define the conditions that make a move valid
  sorry

theorem swap_black_gray_pieces : can_swap initial_state swapped_state := 
  sorry

end swap_black_gray_pieces_l183_183158


namespace vector_projection_max_value_proof_l183_183789

open Real

noncomputable def vector_projection_max_value
    (a b c e : EuclideanSpace ℝ (Fin 2))
    (h_a : ‖a‖ = 1)
    (h_b : ‖b‖ = 1)
    (h_c : ‖c‖ = 1)
    (h_ab : dot_product a b = 1 / 2)
    (h_bc : dot_product b c = 1 / 2) :=
  max_value := 5

theorem vector_projection_max_value_proof (a b c e : EuclideanSpace ℝ (Fin 2))
    (h_a : ‖a‖ = 1)
    (h_b : ‖b‖ = 1)
    (h_c : ‖c‖ = 1)
    (h_ab : dot_product a b = 1 / 2)
    (h_bc : dot_product b c = 1 / 2) :
  (∀ e, ‖e‖ = 1 → max 
    (abs (dot_product a e) + 2 * abs (dot_product b e) + 3 * abs (dot_product c e)) = 5) :=
sorry

end vector_projection_max_value_proof_l183_183789


namespace shampoo_time_l183_183841

noncomputable def rate (hours : ℝ) : ℝ := 1 / hours

noncomputable def combined_rate_main_floor :=
  rate 3 + rate 6 + rate 4 + rate 5

noncomputable def time_for_main_floor := 1 / combined_rate_main_floor

noncomputable def time_for_second_floor := time_for_main_floor / 2

noncomputable def total_time := time_for_main_floor + time_for_second_floor

theorem shampoo_time :
  total_time = 1.5789 := 
begin
  sorry
end

end shampoo_time_l183_183841


namespace problem_I_minimal_period_and_intervals_problem_II_min_abs_a_if_g_even_l183_183776

def f (x : ℝ) : ℝ := cos x * (sin x - real.sqrt 3 * cos x) + real.sqrt 3 / 2

def g (x a : ℝ) : ℝ := f (x + a)

theorem problem_I_minimal_period_and_intervals :
  (∀ x, f(x) = sin (2 * x - π / 3)) ∧
  (∀ T, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ k : ℤ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) → (∀ x, f'(x) > 0)) :=
sorry

theorem problem_II_min_abs_a_if_g_even (a : ℝ) (h : ∀ x, g x a = g (-x) a) :
  ∃ k : ℤ, a = k * π / 2 + 5 * π / 12 ∧ |a| = π / 12 :=
sorry

end problem_I_minimal_period_and_intervals_problem_II_min_abs_a_if_g_even_l183_183776


namespace solution_set_of_inequality_l183_183348

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by
  sorry

end solution_set_of_inequality_l183_183348


namespace volume_of_fifth_section_l183_183458

theorem volume_of_fifth_section
  (a : ℕ → ℚ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence constraint
  (h_sum_top_four : a 0 + a 1 + a 2 + a 3 = 3)  -- Sum of the top four sections
  (h_sum_bottom_three : a 6 + a 7 + a 8 = 4)  -- Sum of the bottom three sections
  : a 4 = 67 / 66 := sorry

end volume_of_fifth_section_l183_183458


namespace triangle_count_equals_two_l183_183385

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def count_valid_triangles : ℕ :=
  let lengths := [3, 4, 7, 9]
  let combinations := [(3, 4, 7), (3, 4, 9), (3, 7, 9), (4, 7, 9)]
  combinations.count (λ (t : ℕ × ℕ × ℕ), is_triangle t.1 t.2 t.3)

theorem triangle_count_equals_two : count_valid_triangles = 2 := by
  sorry

end triangle_count_equals_two_l183_183385


namespace g_at_5_l183_183161

variable (g : ℝ → ℝ)

-- Define the condition on g
def functional_condition : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1

-- The statement proven should be g(5) = 8 given functional_condition
theorem g_at_5 (h : functional_condition g) : g 5 = 8 := by
  sorry

end g_at_5_l183_183161


namespace time_to_cross_signal_pole_l183_183593

-- Define given conditions
def length_train : ℕ := 300 /- 300 meters -/
def time_cross_platform : ℕ := 36 /- 36 seconds -/
def length_platform : ℕ := 300 /- 300 meters -/

-- Define the proof problem
theorem time_to_cross_signal_pole :
  let speed := (length_train + length_platform) / time_cross_platform in
  (length_train : ℚ) / speed = 18 :=
by
  sorry

end time_to_cross_signal_pole_l183_183593


namespace car_speed_reduction_and_increase_l183_183594

theorem car_speed_reduction_and_increase (V x : ℝ)
  (h1 : V > 0) -- V is positive
  (h2 : V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100)) :
  x = 20 :=
sorry

end car_speed_reduction_and_increase_l183_183594


namespace three_digit_even_count_l183_183366

-- Define the set of allowable digits
def S := {0, 1, 2, 3, 4, 5}

-- Define the function that counts the number of valid three-digit even numbers without repeating digits
def evenCount (S : Set ℕ) : ℕ :=
  -- The function body is omitted here
  sorry

-- The theorem to prove
theorem three_digit_even_count : evenCount S = 52 := by
  sorry

end three_digit_even_count_l183_183366


namespace total_cost_is_2089_l183_183882

noncomputable def vanilla_price : ℝ := 0.99
noncomputable def chocolate_price : ℝ := 1.29
noncomputable def strawberry_price : ℝ := 1.49

noncomputable def num_vanilla_cones : ℕ := 6
noncomputable def num_chocolate_cones : ℕ := 8
noncomputable def num_strawberry_cones : ℕ := 6

noncomputable def discount_vanilla_cone_list : list ℝ := 
  (list.repeat vanilla_price 4) ++ (list.repeat (vanilla_price / 2) 2)

noncomputable def discount_chocolate_cone_list : list ℝ := 
  list.repeat chocolate_price 6

noncomputable def discount_strawberry_cone_list : list ℝ := 
  (list.repeat strawberry_price 4) ++
  (list.repeat (strawberry_price * (1 - 0.25)) 2)

noncomputable def total_cost : ℝ :=
  (discount_vanilla_cone_list.sum) + 
  (discount_chocolate_cone_list.sum) + 
  (discount_strawberry_cone_list.sum)

theorem total_cost_is_2089 : total_cost = 20.89 := sorry

end total_cost_is_2089_l183_183882


namespace last_score_86_l183_183505

theorem last_score_86 (scores : List ℕ)
  (H1 : scores ~ List [73, 78, 84, 86, 97])
  (H2 : ∀ (prefix : List ℕ), prefix ⊆ scores → prefix ≠ [] →
    (list.sum prefix) % (list.length prefix) = 0) : 
  ∃ (last_score : ℕ), 
    last_score ∈ scores ∧ 
    (list.sum (scores.erase last_score)) % 4 = 2 ∧ 
    last_score = 86 :=
by
  sorry

end last_score_86_l183_183505


namespace find_k_value_l183_183353

noncomputable def csc (x : ℝ) : ℝ := 1 / sin x
noncomputable def sec (x : ℝ) : ℝ := 1 / cos x
noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

theorem find_k_value : ∀ α : ℝ, 
  2 * (sin α + csc α) ^ 2 + 2 * (cos α + sec α) ^ 2 = 14 + 2 * tan α ^ 2 + 2 * cot α ^ 2 :=
by
  sorry

end find_k_value_l183_183353


namespace simplify_complex_fraction_l183_183909

theorem simplify_complex_fraction :
  (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i :=
by
  sorry

end simplify_complex_fraction_l183_183909


namespace range_of_m_if_p_or_q_is_false_l183_183373

variable {x m : ℝ}

def p := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m_if_p_or_q_is_false (hp : ¬ p) (hq : ¬ q) : 2 ≤ m :=
by sorry

end range_of_m_if_p_or_q_is_false_l183_183373


namespace number_of_squares_in_L_shaped_grid_l183_183295

def is_square_in_grid (n : ℕ) (grid : ℕ × ℕ → Prop) (x y size : ℕ) : Prop :=
  ∀ i j, i < size → j < size → grid (x + i, y + j)

noncomputable def count_squares (n : ℕ) (grid : ℕ × ℕ → Prop) : ℕ :=
  ∑ size in finset.range n.succ, ∑ x in finset.range (n + 1 - size), ∑ y in finset.range (n + 1 - size), if is_square_in_grid n grid x y size then 1 else 0

def L_shaped_grid (x y : ℕ) : Prop :=
  (0 ≤ x ∧ x < 5 ∧ 0 ≤ y ∧ y < 6) ∨ (4 ≤ x ∧ x < 6 ∧ 2 ≤ y ∧ y < 6)

theorem number_of_squares_in_L_shaped_grid : count_squares 6 L_shaped_grid = 61 :=
by
  sorry

end number_of_squares_in_L_shaped_grid_l183_183295


namespace ratio_lateral_surface_base_l183_183163

variable (a : ℝ)

def radius_of_cylinder (a : ℝ) : ℝ := a / π
def lateral_surface_area (a : ℝ) : ℝ := 4 * a^2
def base_area (a : ℝ) : ℝ := π * (radius_of_cylinder a)^2
def ratio (a : ℝ) : ℝ := lateral_surface_area a / base_area a

theorem ratio_lateral_surface_base (a : ℝ) : ratio a = 4 * π :=
by
  sorry

end ratio_lateral_surface_base_l183_183163


namespace Harriet_siblings_product_l183_183793

variable (Harry_sisters : Nat)
variable (Harry_brothers : Nat)
variable (Harriet_sisters : Nat)
variable (Harriet_brothers : Nat)

theorem Harriet_siblings_product:
  Harry_sisters = 4 -> 
  Harry_brothers = 6 ->
  Harriet_sisters = Harry_sisters -> 
  Harriet_brothers = Harry_brothers ->
  Harriet_sisters * Harriet_brothers = 24 :=
by
  intro hs hb hhs hhb
  rw [hhs, hhb]
  sorry

end Harriet_siblings_product_l183_183793


namespace point_first_quadrant_l183_183541

def i : ℂ := complex.I

def z : ℂ := (2 * i - 1) / i

def is_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem point_first_quadrant : is_first_quadrant z :=
by
  sorry

end point_first_quadrant_l183_183541


namespace unique_rational_line_through_irrational_point_l183_183450

-- Define rational points
def is_rational_point (p : ℝ × ℝ) :=
  ∃ (r1 r2 : ℚ), p = (r1, r2)

-- Define the point (a, 0) with a as irrational
variable (a : ℝ)
axiom h_irrational_a : irrational a
def point_a_zero := (a, 0)

-- Lean 4 statement for the problem
theorem unique_rational_line_through_irrational_point :
  ∀ l : ℝ × ℝ → Prop,
    (∃ (p1 p2 : ℝ × ℝ), is_rational_point p1 ∧ is_rational_point p2 ∧ p1 ≠ p2 ∧ l p1 ∧ l p2) ∧ l point_a_zero
    → l = (λ p : ℝ × ℝ, p.2 = 0) :=
sorry

end unique_rational_line_through_irrational_point_l183_183450


namespace trigonometric_comparison_l183_183493

noncomputable def a : ℝ := 2 * Real.sin (13 * Real.pi / 180) * Real.cos (13 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.tan (76 * Real.pi / 180) / (1 + Real.tan (76 * Real.pi / 180)^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem trigonometric_comparison : b > a ∧ a > c := by
  sorry

end trigonometric_comparison_l183_183493


namespace g_at_3_eq_10_by_7_l183_183434

def g (x : ℝ) : ℝ := (x^2 + 1) / (4 * x - 5)

-- We state the theorem we need to prove
theorem g_at_3_eq_10_by_7 : g 3 = 10 / 7 := sorry

end g_at_3_eq_10_by_7_l183_183434


namespace pow_mod_eq_l183_183564

theorem pow_mod_eq : (17 ^ 2001) % 23 = 11 := 
by {
  sorry
}

end pow_mod_eq_l183_183564


namespace racket_price_l183_183098

theorem racket_price (cost_sneakers : ℕ) (cost_outfit : ℕ) (total_spent : ℕ) 
  (h_sneakers : cost_sneakers = 200) 
  (h_outfit : cost_outfit = 250) 
  (h_total : total_spent = 750) : 
  (total_spent - cost_sneakers - cost_outfit) = 300 :=
sorry

end racket_price_l183_183098


namespace arithmetic_sequence_general_formula_l183_183027

noncomputable def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_formula {a : ℕ → ℤ} (h_seq : arithmetic_seq a) 
  (h_a1 : a 1 = 6) (h_a3a5 : a 3 + a 5 = 0) : 
  ∀ n, a n = 8 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l183_183027


namespace shaded_triangle_probability_l183_183832

noncomputable def total_triangles : ℕ := 5
noncomputable def shaded_triangles : ℕ := 2
noncomputable def probability_shaded : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : probability_shaded = 2 / 5 :=
by
  sorry

end shaded_triangle_probability_l183_183832


namespace angle_of_intersection_with_y_axis_l183_183677

def f (x : ℝ) : ℝ := Real.exp x - x

theorem angle_of_intersection_with_y_axis : 
  ∀ (x : ℝ), x = 0 → f' x = 0 :=
by
  intros x hx
  rw hx
  calc 
    (Real.exp 0 - 1) = (1 - 1) := by rw Real.exp_zero
            ...     = 0         := by norm_num


end angle_of_intersection_with_y_axis_l183_183677


namespace number_of_possible_values_for_s_l183_183167

-- Problem statement: proving the number of possible values for s
theorem number_of_possible_values_for_s :
  let R1 := (0.2112 : ℝ)
      R2 := (0.2360 : ℝ)
      four_place_decimals := {s : ℝ | ∃ (a b c d : ℤ), s = (a / 10 + b / 100 + c / 1000 + d / 10000) ∧ 0 ≤ a < 10 ∧ 0 ≤ b < 10 ∧ 0 ≤ c < 10 ∧ 0 ≤ d < 10}
      valid_s := {s : ℝ | R1 ≤ s ∧ s ≤ R2 ∧ s ∈ four_place_decimals}
  in 249 = set.card valid_s :=
begin
  sorry
end

end number_of_possible_values_for_s_l183_183167


namespace problem_1_problem_2_l183_183773

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

theorem problem_1 (a : ℝ) (h : a > 0 ∧ a ≠ 1) (x : ℝ) : f a (-x) = -f a x :=
by sorry

theorem problem_2 (a : ℝ) (h : a > 0 ∧ a ≠ 1) (m : ℝ) : (∀ x ∈ set.Icc (-1 : ℝ) 1, f a x ≥ m) ↔ m ≤ 1 :=
by sorry

end problem_1_problem_2_l183_183773


namespace more_girls_than_boys_l183_183449

theorem more_girls_than_boys (total students : ℕ) (girls boys : ℕ) (h1 : total = 41) (h2 : girls = 22) (h3 : girls + boys = total) : (girls - boys) = 3 :=
by
  sorry

end more_girls_than_boys_l183_183449


namespace diameter_of_large_circle_l183_183147

-- Given conditions
def small_radius : ℝ := 3
def num_small_circles : ℕ := 6

-- Problem statement: Prove the diameter of the large circle
theorem diameter_of_large_circle (r : ℝ) (n : ℕ) (h_radius : r = small_radius) (h_num : n = num_small_circles) :
  ∃ (R : ℝ), R = 9 * 2 := 
sorry

end diameter_of_large_circle_l183_183147


namespace similar_triangle_longest_side_l183_183164

theorem similar_triangle_longest_side (a b c : ℕ) (h_a : a = 8) (h_b : b = 10) (h_c : c = 12) (perimeter_similar : ℕ) (h_perimeter : perimeter_similar = 150) : 
  let x := perimeter_similar / (a + b + c),
      a' := a * x,
      b' := b * x,
      c' := c * x 
  in a' = 60 ∧ b' + c' > a' ∧ a' + c' > b' ∧ a' + b' > c' :=
by
  sorry

end similar_triangle_longest_side_l183_183164


namespace sum_of_numbers_l183_183954

theorem sum_of_numbers (a b c : ℝ) (h_ratio : a / 1 = b / 2 ∧ b / 2 = c / 3) (h_sum_squares : a^2 + b^2 + c^2 = 2744) : 
  a + b + c = 84 := 
sorry

end sum_of_numbers_l183_183954


namespace probability_of_large_subtriangle_area_l183_183188

def point_in_triangle (ABC : Triangle) : Type := sorry
def area_of_triangle (t : Triangle) : ℝ := sorry

theorem probability_of_large_subtriangle_area (ABC : Triangle) (M : point_in_triangle ABC) :
  (random_point M → ∃ T1 T2 T3 : Triangle, area_of_triangle T1 > area_of_triangle T2 + area_of_triangle T3) → 
  (probability (M ∈ ABC) = 3/4) :=
sorry

end probability_of_large_subtriangle_area_l183_183188


namespace triangle_area_l183_183815

theorem triangle_area {α β γ : Type} [linear_ordered_field α] [has_sin γ] [topological_space γ]
  [topological_space β] [topological_space α] [has_coe_t α γ] [has_coe_t β γ] [has_coe_t γ α]
  [has_coe_t α β]: 
  (a : α = 6) (B : β = 30) (C : β = 120) :
  let A := 180 - B - C in
  let b := a in
  (1 / 2 * a * b * sin C = 9 * sqrt 3) :=
by
  -- The proof goes here
  sorry

end triangle_area_l183_183815


namespace initial_amount_is_800_l183_183234

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end initial_amount_is_800_l183_183234


namespace construction_paper_initial_count_l183_183820

theorem construction_paper_initial_count 
    (b r d : ℕ)
    (ratio_cond : b = 2 * r)
    (daily_usage : ∀ n : ℕ, n ≤ d → n * 1 = b ∧ n * 3 = r)
    (last_day_cond : 0 = b ∧ 15 = r):
    b + r = 135 :=
sorry

end construction_paper_initial_count_l183_183820


namespace total_birds_remaining_l183_183548

-- Definitions from conditions
def initial_grey_birds : ℕ := 40
def additional_white_birds : ℕ := 6
def white_birds (grey_birds: ℕ) : ℕ := grey_birds + additional_white_birds
def remaining_grey_birds (grey_birds: ℕ) : ℕ := grey_birds / 2

-- Proof problem
theorem total_birds_remaining : 
  let grey_birds := initial_grey_birds;
  let white_birds_next_to_cage := white_birds(grey_birds);
  let grey_birds_remaining := remaining_grey_birds(grey_birds);
  (grey_birds_remaining + white_birds_next_to_cage) = 66 :=
by {
  sorry
}

end total_birds_remaining_l183_183548


namespace coplanar_points_l183_183702

-- Define the points
def p1 : ℝ × ℝ × ℝ := (0, 0, 0)
def p2 (b : ℝ) : ℝ × ℝ × ℝ := (1, b, 0)
def p3 (b : ℝ) : ℝ × ℝ × ℝ := (0, 1, b)
def p4 (b : ℝ) : ℝ × ℝ × ℝ := (b, 0, 2)

-- Define the vectors based on points
def v1 (b : ℝ) : ℝ × ℝ × ℝ := (1, b, 0)
def v2 (b : ℝ) : ℝ × ℝ × ℝ := (0, 1, b)
def v3 (b : ℝ) : ℝ × ℝ × ℝ := (b, 0, 2)

-- Define the determinant of the matrix formed by vectors
def det (b : ℝ) : ℝ :=
  1 * (1 * 2 - 0 * b) +
  b * (b * b - 1 * 0)

-- Noncomputable proof / example indicating the solution found
theorem coplanar_points (b : ℝ) : det b = 0 → b = -real.cbrt 2 :=
by sorry

end coplanar_points_l183_183702


namespace common_term_sequence_7n_l183_183156

theorem common_term_sequence_7n (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (7 / 9) * (10^n - 1) :=
by
  sorry

end common_term_sequence_7n_l183_183156


namespace expression_simplification_l183_183639

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183639


namespace temperature_at_6_km_l183_183470

-- Define the initial conditions
def groundTemperature : ℝ := 25
def temperatureDropPerKilometer : ℝ := 5

-- Define the question which is the temperature at a height of 6 kilometers
def temperatureAtHeight (height : ℝ) : ℝ :=
  groundTemperature - temperatureDropPerKilometer * height

-- Prove that the temperature at 6 kilometers is -5 degrees Celsius
theorem temperature_at_6_km : temperatureAtHeight 6 = -5 := by
  -- Use expected proof  
  simp [temperatureAtHeight, groundTemperature, temperatureDropPerKilometer]
  sorry

end temperature_at_6_km_l183_183470


namespace greatest_length_of_pieces_l183_183129

theorem greatest_length_of_pieces (a b c : ℕ) (h1 : a = 28) (h2 : b = 42) (h3 : c = 70) :
  Nat.gcd (Nat.gcd a b) c = 14 :=
by
  rw [h1, h2, h3]
  exact Nat.gcd_gcd_gcd 28 42 70
-- We will skip the proof steps and return 'sorry' for now
  sorry

end greatest_length_of_pieces_l183_183129


namespace eccentricity_range_ellipses_l183_183892

noncomputable def ellipse_eccentricity_range 
  (a b : ℝ) (ha : a > 1) (hb : b > 1) (h_ab : a > b > 1) : set ℝ :=
  {e | ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
                  ((x, y) • (x - a, y) = 0) ∧ 
                  e = (√(a^2 - b^2) / a) 
  }

theorem eccentricity_range_ellipses (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a > b) :
  ellipse_eccentricity_range a b ha hb hab = {e : ℝ | (sqrt 2 / 2) < e ∧ e < 1} :=
sorry

end eccentricity_range_ellipses_l183_183892


namespace range_of_m_l183_183389

-- Define the points and hyperbola condition
section ProofProblem

variables (m y₁ y₂ : ℝ)

-- Given conditions
def point_A_hyperbola : Prop := y₁ = -3 - m
def point_B_hyperbola : Prop := y₂ = (3 + m) / 2
def y1_greater_than_y2 : Prop := y₁ > y₂

-- The theorem to prove
theorem range_of_m (h1 : point_A_hyperbola m y₁) (h2 : point_B_hyperbola m y₂) (h3 : y1_greater_than_y2 y₁ y₂) : m < -3 :=
by { sorry }

end ProofProblem

end range_of_m_l183_183389


namespace chameleons_all_green_l183_183886

theorem chameleons_all_green :
  ∃ j r v : ℕ, j = 7 ∧ r = 10 ∧ v = 17 ∧ (j + r + v = 34) ∧
  (∀ t1 t2 t3 : ℕ, 
    let meet := t1 - t2 in 
    ¬(t1 = t2) → t3 % 3 = (v - r) % 3 ∧ (v - r) % 3 = (r - j) % 3 ∧ (r - j) % 3 = (j - v) % 3) →
    ∀ t : ℕ, 
      (∀ c : ℕ, c ∈ {r, j} → c ≠ t) → v = 34 :=
begin
  sorry
end

end chameleons_all_green_l183_183886


namespace domino_arrangements_l183_183483

def number_of_distinct_paths_6x5_grid: ℕ := 126

theorem domino_arrangements : 
  (∀ (grid_width grid_height : ℕ), grid_width = 6 → grid_height = 5 →
   (∃ (dominoes : ℕ), dominoes = 5 →
   (∀ (movement_r movement_d : ℕ), movement_r = 5 → movement_d = 4 →
   (number_of_distinct_paths grid_width grid_height dominoes
   movement_r movement_d = number_of_distinct_paths_6x5_grid)))) :=
sorry

noncomputable def number_of_distinct_paths (grid_width grid_height dominoes movement_r movement_d : ℕ) : ℕ :=
  if grid_width = 6 ∧ grid_height = 5 ∧ dominoes = 5 ∧ movement_r = 5 ∧ movement_d = 4 then 126 else 0

end domino_arrangements_l183_183483


namespace expression_simplification_l183_183638

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183638


namespace pet_store_initial_gerbils_l183_183248

-- Define sold gerbils
def sold_gerbils : ℕ := 69

-- Define left gerbils
def left_gerbils : ℕ := 16

-- Define the initial number of gerbils
def initial_gerbils : ℕ := sold_gerbils + left_gerbils

-- State the theorem to be proved
theorem pet_store_initial_gerbils : initial_gerbils = 85 := by
  -- This is where the proof would go
  sorry

end pet_store_initial_gerbils_l183_183248


namespace range_of_x_l183_183308

def op (a b : ℝ) : ℝ :=
if a > b then a * b + b else a * b - b

theorem range_of_x (x : ℝ) : 3 ⊕ (x + 2) > 0 ↔ (-2 < x ∧ x < 1) ∨ (x > 1) :=
by
  let op (a b : ℝ) : ℝ := if a > b then a * b + b else a * b - b
  have h1 : ∀ x, 3 > x + 2 → 3 ⊕ (x + 2) = 4 * x + 8,
    intro x,
    intro h,
    rw [op, (lt_trans h (by linarith)), mul_add, mul_comm, add_comm (3 * (x + 2)), mul_add],
    sorry,

  have h2 : ∀ x, 3 < x + 2 → 3 ⊕ (x + 2) = 2 * x + 4,
    intro x,
    intro h,
    rw [op, (gt_trans h (by linarith [h])), mul_add, mul_comm, sub_add, add_sub, sub_add],
    sorry,

  split,
  intro h,
  by_cases h1 : 3 > x + 2,
    left,
    split,
    linarith [h1],
    have : (4 : ℝ) > 0 :=
      by linarith,
    linarith [this, show 3 ⊕ (x + 2) = 4 * x + 8 from h1 x h1, h],
    right,
    have : (2 : ℝ) > 0 :=
      by linarith,
    linarith [show 3 ⊕ (x + 2) = 2 * x + 4 from h2 x (not_lt.mp h1), h],
  
  intro h,
  cases h with h h;
  linarith [show 3 ⊕ (x + 2) = 4 * x + 8 from h1 x (by linarith), show 3 ⊕ (x + 2) = 2 * x + 4 from h2 x (by linarith)],

end range_of_x_l183_183308


namespace find_a_b_f_inequality_l183_183041

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

-- a == 1 and b == 1 from the given conditions
theorem find_a_b (e : ℝ) (h_e : e = Real.exp 1) (b : ℝ) (a : ℝ) 
  (h_tangent : ∀ x, f x a = (e - 2) * x + b → a = 1 ∧ b = 1) : a = 1 ∧ b = 1 :=
sorry

-- prove f(x) > x^2 + 4x - 14 for x >= 0
theorem f_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ x : ℝ, 0 ≤ x → f x 1 > x^2 + 4 * x - 14 :=
sorry

end find_a_b_f_inequality_l183_183041


namespace range_of_k_l183_183442

-- Define the function
noncomputable def my_function (k x : ℝ) : ℝ := k * x^2 - 4 * x - 8

-- State the theorem
theorem range_of_k (k : ℝ) : 
  (∀ x ∈ set.Icc (4 : ℝ) (16 : ℝ), deriv (my_function k) x < 0) → 
  k ∈ set.Iic ((1 : ℝ) / 8) :=
sorry

end range_of_k_l183_183442


namespace arithmetic_sequence_G_minus_L_l183_183270

-- Define conditions for the arithmetic sequence
def is_arithmetic_sequence (seq : ℕ → ℝ) := 
  ∃ d, ∀ n, seq n = 75 + (n - 1) * d

-- Define the sequence and properties
variables (seq : ℕ → ℝ) (d : ℝ)
-- The arithmetic sequence is 300 terms long
variable (n : ℕ) (h_n : n = 300)
-- Terms bounds: each term in the sequence is at least 5 and at most 150
variable (h_bounds : ∀ k: ℕ, 1 ≤ k ∧ k ≤ n → (5 : ℝ) ≤ seq k ∧ seq k ≤ 150)
-- The sum of the terms is 22500
variable (h_sum : ∑ i in finset.range n, seq (i + 1) = 22500)

-- Define L and G as the least and greatest possible value of the 75th term
def L : ℝ := 75 - 225 * (70 / 299)
def G : ℝ := 75 + 225 * (70 / 299)

-- Prove that G - L = 31500 / 299
theorem arithmetic_sequence_G_minus_L (h_arith : is_arithmetic_sequence seq) : 
  G - L = 31500 / 299 :=
sorry

end arithmetic_sequence_G_minus_L_l183_183270


namespace ellipse_problem_l183_183745

-- Definitions from the conditions
def is_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) : Prop :=
  e = real.sqrt 6 / 3 ∧ 2 * a = 2 * real.sqrt 3

def is_line (k m : ℝ) : Prop :=
  ∀ x y : ℝ, y = k * x + m

noncomputable def ellipse_eq (a : ℝ) : ℝ := a^2

def proof_distance (a k m : ℝ) (h : a > 0 ∧ k > 0) : Prop :=
  ∀ (x1 x2 y1 y2 : ℝ), 
  (1 + 3*k^2)*x1^2 + 6*k*m*x1 + 3*m^2 - 3 = 0 ∧
  x1 * x2 + y1 * y2 = 0 ∧
  (1 + k^2) *  (3*m^2 - 3/(1 + 3*k^2)) - k*m * (6*k*m / (1 + 3*k^2)) + m^2 = 0 → 
  real.abs m / real.sqrt (1 + k^2) = real.sqrt 3 / 2

-- Main theorem to be proven
theorem ellipse_problem :
  ∃ (a b k m : ℝ), 
  a > b ∧ b > 0 ∧ is_ellipse a b (a > b ∧ b > 0) (real.sqrt 6 / 3) ∧
  is_line k m ∧
  proof_distance a k m (a > 0 ∧ k > 0) :=
sorry

end ellipse_problem_l183_183745


namespace angle_coterminal_l183_183354

theorem angle_coterminal (k : ℤ) : 
  ∃ α : ℝ, α = 30 + k * 360 :=
sorry

end angle_coterminal_l183_183354


namespace max_intersection_distance_l183_183084

theorem max_intersection_distance :
  let C1_x (α : ℝ) := 2 + 2 * Real.cos α
  let C1_y (α : ℝ) := 2 * Real.sin α
  let C2_x (β : ℝ) := 2 * Real.cos β
  let C2_y (β : ℝ) := 2 + 2 * Real.sin β
  let l1 (α : ℝ) := α
  let l2 (α : ℝ) := α - Real.pi / 6
  (0 < Real.pi / 2) →
  let OP (α : ℝ) := 4 * Real.cos α
  let OQ (α : ℝ) := 4 * Real.sin (α - Real.pi / 6)
  let pq_prod (α : ℝ) := OP α * OQ α
  ∀α, 0 < α ∧ α < Real.pi / 2 → pq_prod α ≤ 4 := by
  sorry

end max_intersection_distance_l183_183084


namespace shared_bill_approx_16_99_l183_183178

noncomputable def calculate_shared_bill (total_bill : ℝ) (num_people : ℕ) (tip_rate : ℝ) : ℝ :=
  let tip := total_bill * tip_rate
  let total_with_tip := total_bill + tip
  total_with_tip / num_people

theorem shared_bill_approx_16_99 :
  calculate_shared_bill 139 9 0.10 ≈ 16.99 :=
sorry

end shared_bill_approx_16_99_l183_183178


namespace incorrect_option_B_l183_183729

variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

-- Conditions:
-- 1. m is parallel to α
-- 2. n is parallel to β
-- 3. α is perpendicular to β
-- Given these conditions, we need to prove the following statement is incorrect:
-- "m is perpendicular to n or m is parallel to n"

theorem incorrect_option_B (h1 : m ∥ α) (h2 : n ∥ β) (h3 : α ⟂ β) : ¬ (m ⟂ n ∨ m ∥ n) :=
sorry

end incorrect_option_B_l183_183729


namespace quartic_solution_l183_183335

theorem quartic_solution (x : ℝ) : 
  (√[4] x = 15 / (8 - √[4] x)) ↔ (x = 81 ∨ x = 625) :=
by
  sorry

end quartic_solution_l183_183335


namespace brownies_each_l183_183730

theorem brownies_each (num_columns : ℕ) (num_rows : ℕ) (total_people : ℕ) (total_brownies : ℕ) 
(h1 : num_columns = 6) (h2 : num_rows = 3) (h3 : total_people = 6) 
(h4 : total_brownies = num_columns * num_rows) : 
total_brownies / total_people = 3 := 
by
  -- Placeholder for the actual proof
  sorry

end brownies_each_l183_183730


namespace set_equality_proof_l183_183873

theorem set_equality_proof :
  (∃ (u : ℤ), ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l) ↔
  (∃ (u : ℤ), ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r) :=
sorry

end set_equality_proof_l183_183873


namespace stickers_given_l183_183879

theorem stickers_given (initial_stickers bought_stickers birthday_stickers used_stickers left_stickers : ℕ)
  (h1 : initial_stickers = 20)
  (h2 : bought_stickers = 26)
  (h3 : birthday_stickers = 20)
  (h4 : used_stickers = 58)
  (h5 : left_stickers = 2) :
  (initial_stickers + bought_stickers + birthday_stickers) - (used_stickers + left_stickers) = 6 :=
by
  rw [h1, h2, h3, h4, h5]
  sorry

end stickers_given_l183_183879


namespace circle_eq_of_tangent_and_chord_len_l183_183341

theorem circle_eq_of_tangent_and_chord_len (a : ℚ) :
  ∀ (x y : ℚ),
  (C : ℚ × ℚ) →
  3 * C.1 - 4 * C.2 = 0 → -- Center C bounds
  (r : ℚ) →
  r = 3 * abs a → -- Circle is tangent to x-axis
  ∀ (k : ℚ),
  let dist_to_line := abs (C.1 - C.2) / real.sqrt 2 in
  let chord_length := 2 * real.sqrt 17 in
  by sorry in
  dist_to_line ^ 2 + chord_length / 2 ^ 2 = (3 * abs a) ^ 2 →
  let eq1 := (x - 4 * real.sqrt 2) ^ 2 + (y - 3 * real.sqrt 2) ^ 2 - 18 in
  let eq2 := (x + 4 * real.sqrt 2) ^ 2 + (y + 3 * real.sqrt 2) ^ 2 - 18 in
  eq1 = 0 ∨ eq2 = 0

end circle_eq_of_tangent_and_chord_len_l183_183341


namespace increasing_on_interval_l183_183678

open Real

theorem increasing_on_interval (x : ℝ) (h : (π / 2) < x ∧ x < π) :
  ∀ y, y ∈ {fun x => cos (2 * x)} →
    StrictMonoOn y ((π / 2), π) := by
sorry

end increasing_on_interval_l183_183678


namespace fifteenth_number_in_base_5_l183_183082

theorem fifteenth_number_in_base_5 :
  ∃ n : ℕ, n = 15 ∧ (n : ℕ) = 3 * 5^1 + 0 * 5^0 :=
by
  sorry

end fifteenth_number_in_base_5_l183_183082


namespace cricket_team_new_win_percentage_l183_183447

theorem cricket_team_new_win_percentage (total_matches_initially : ℕ)
  (percentage_won_initially : ℚ)
  (matches_won_streak : ℕ)
  (expected_percentage : ℚ) :
  total_matches_initially = 120 → 
  percentage_won_initially = 22 / 100 →
  matches_won_streak = 75 →
  expected_percentage ≈ 5179 / 100 :=
by
  intros h1 h2 h3
  let initial_wins := (percentage_won_initially * total_matches_initially).to_nat
  let total_wins := initial_wins + matches_won_streak
  let total_matches := total_matches_initially + matches_won_streak
  let new_win_percentage := (total_wins.to_rat / total_matches.to_rat) * 100
  have : new_win_percentage ≈ expected_percentage := sorry
  exact this

end cricket_team_new_win_percentage_l183_183447


namespace distance_from_point_to_asymptote_l183_183937

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l183_183937


namespace smallest_area_6_8_l183_183200

noncomputable def smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) : ℝ :=
  let c1 := Real.sqrt (a^2 + b^2)
  let area1 := 0.5 * a * b
  let c2 := Real.sqrt (b^2 - a^2)
  let area2 := if h : b > a then 0.5 * a * c2 else 0.5 * b * c2
  min area1 area2

theorem smallest_area_6_8 : smallest_area_of_right_triangle 6 8 6 8 = 15.87 :=
sorry

end smallest_area_6_8_l183_183200


namespace num_solutions_l183_183796

-- Define the equation condition
def equation_condition (x y : ℤ) : Prop := x^4 + y^2 = 2 * y + 3

-- Define the main theorem to prove
theorem num_solutions : 
  (Finset.univ : Finset (ℤ × ℤ)).filter (λ p, equation_condition p.1 p.2).card = 2 :=
sorry

end num_solutions_l183_183796


namespace determine_triangle_shape_l183_183817

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (AB AC : euclidean_space ℝ (fin 3))

noncomputable def triangle_shape (AB AC : euclidean_space ℝ (fin 3)) : Prop :=
(|AB| = 4) ∧ (|AC| = 4) ∧ (dot_product AB AC = 8) → 
(euclidean_distance A B = euclidean_distance A C ∧ 
 ∃ (angle: ℝ), (angle = real.arccos (8 / (4 * 4))) ∧ angle = π / 3 ∧ 
 triangle_is_equilateral A B C)

theorem determine_triangle_shape : 
(triangle_shape AB AC) ↔ (euclidean_distance A B = 4) ∧ (euclidean_distance A C = 4) ∧ (euclidean.distance A B = euclidean.distance A C) ∧ (dot_product AB AC = 8) → 
(triangle_is_equilateral A B C) :=
by sorry

end determine_triangle_shape_l183_183817


namespace total_new_cans_from_3125_l183_183356

def number_of_new_cans (initial_cans new_cans_per_5: ℕ) : ℕ := 
  if initial_cans < 5 then 0 
  else let new_cans := initial_cans / 5 in 
       new_cans + number_of_new_cans new_cans new_cans_per_5

theorem total_new_cans_from_3125 : number_of_new_cans 3125 5 = 781 := by
  sorry

end total_new_cans_from_3125_l183_183356


namespace distance_from_point_to_asymptote_l183_183935

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l183_183935


namespace area_of_closed_figure_l183_183526

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then sqrt (1 - (x + 1)^2) else
if x ≤ 1 then x^2 - x else 0

theorem area_of_closed_figure : 
  (∫ x in -2..0, sqrt (1 - (x + 1)^2)) + (∫ x in 0..1, x^2 - x) = (Real.pi / 2) + (1 / 6) :=
sorry

end area_of_closed_figure_l183_183526


namespace simplify_complex_fraction_l183_183913

/-- The simplified form of (5 + 7 * I) / (2 + 3 * I) is (31 / 13) - (1 / 13) * I. -/
theorem simplify_complex_fraction : (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i := 
by {
    sorry
}

end simplify_complex_fraction_l183_183913


namespace jerry_expected_candies_l183_183819

noncomputable def blue_eggs := rat.mk 4 10
noncomputable def purple_eggs := rat.mk 3 10
noncomputable def red_eggs := rat.mk 2 10
noncomputable def green_eggs := rat.mk 1 10

-- Expected number of candies for each egg color
noncomputable def E_blue :=
  (rat.mk 1 3) * 3 + (rat.mk 1 2) * 2 + (rat.mk 1 6) * 0

noncomputable def E_purple :=
  (rat.mk 1 2) * 5 + (rat.mk 1 2) * 0

noncomputable def E_red :=
  (rat.mk 3 4) * 1 + (rat.mk 1 4) * 4

noncomputable def E_green :=
  (rat.mk 1 2) * 6 + (rat.mk 1 2) * 8

-- Overall expected number of candies
noncomputable def expected_candies :=
  blue_eggs * E_blue + purple_eggs * E_purple +
  red_eggs * E_red + green_eggs * E_green

theorem jerry_expected_candies : expected_candies = rat.mk 26 10 :=
  by 
    -- Verification of expected candies calculation
    have h1 : E_blue = 2 := sorry
    have h2 : E_purple = 2.5 := sorry
    have h3 : E_red = 1.75 := sorry
    have h4 : E_green = 7 := sorry
    sorry

end jerry_expected_candies_l183_183819


namespace solve_inequality_min_value_F_l183_183374

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)
def m := 3    -- Arbitrary constant, m + n = 7 implies n = 4
def n := 4

-- First statement: Solve the inequality f(x) ≥ (m + n)x
theorem solve_inequality (x : ℝ) : f x ≥ (m + n) * x ↔ x ≤ 0 := by
  sorry

noncomputable def F (x y : ℝ) : ℝ := max (abs (x^2 - 4 * y + m)) (abs (y^2 - 2 * x + n))

-- Second statement: Find the minimum value of F
theorem min_value_F (x y : ℝ) : (F x y) ≥ 1 ∧ (∃ x y, (F x y) = 1) := by
  sorry

end solve_inequality_min_value_F_l183_183374


namespace diff_eq_solution_l183_183342

noncomputable def general_solution (C1 C2 : ℝ) (y : ℝ → ℝ) : Prop :=
  y = λ x : ℝ, C1 * Real.arctan x + x^3 - 3 * x + C2

theorem diff_eq_solution (C1 C2 : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x : ℝ, (1 + x^2) * (y'' x) + 2 * x * (y' x) = 12 * x^3) ∧ general_solution C1 C2 y :=
begin
  sorry
end

end diff_eq_solution_l183_183342


namespace probability_three_draws_one_white_one_red_probability_two_draws_one_white_one_red_one_other_l183_183551

-- Conditions
constant red_balls : ℕ := 3
constant white_balls : ℕ := 2

def total_balls : ℕ := red_balls + white_balls
def num_draws : ℕ := 3
def event_A_probability : ℝ := (red_balls / total_balls) * (white_balls / (total_balls - 1)) + (white_balls / total_balls) * (red_balls / (total_balls - 1))

-- Problem Statement
theorem probability_three_draws_one_white_one_red :
  let P_A := event_A_probability in
  let P_3_3 := P_A ^ num_draws in
  P_3_3 = 0.216 := by
  sorry

theorem probability_two_draws_one_white_one_red_one_other :
  let P_A := event_A_probability in
  let P_3_2 := (3.choose 2) * (P_A ^ 2) * ((1 - P_A) ^ 1) in
  P_3_2 = 0.432 := by
  sorry

end probability_three_draws_one_white_one_red_probability_two_draws_one_white_one_red_one_other_l183_183551


namespace probability_of_odd_number_l183_183517

theorem probability_of_odd_number (total_outcomes : ℕ) (odd_outcomes : ℕ) (h1 : total_outcomes = 6) (h2 : odd_outcomes = 3) : (odd_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry 

end probability_of_odd_number_l183_183517


namespace quadrilateral_diagonal_length_l183_183703

theorem quadrilateral_diagonal_length (d : ℝ) 
  (h_offsets : true) 
  (area_quadrilateral : 195 = ((1 / 2) * d * 9) + ((1 / 2) * d * 6)) : 
  d = 26 :=
by 
  sorry

end quadrilateral_diagonal_length_l183_183703


namespace find_x0_l183_183394

-- Definitions from conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f : ℝ → ℝ :=
  fun x => if x < 0 then 3^x else -3^(-x)

-- Proof statement
theorem find_x0 (x0 : ℝ) (h1 : odd_function f) (h2 : f x0 = -1/9) : x0 = 2 :=
by
  sorry

end find_x0_l183_183394


namespace problem_f_of_f_neg2_l183_183040

def f (x : ℝ) : ℝ :=
  if 1 < x then real.log x / real.log 2 else (1 / 2) ^ x

theorem problem_f_of_f_neg2 :
  f (f (-2)) = 2 :=
by
  sorry

end problem_f_of_f_neg2_l183_183040


namespace trapezoid_perimeter_l183_183465

open Real

theorem trapezoid_perimeter 
  (EF GH height : ℝ)
  (h1 : EF = 10)
  (h2 : GH = 20)
  (h3 : height = 5)
  (h4 : EF = GH) : 
  EF + GH + 2 * sqrt(height ^ 2 + ((GH - EF) / 2) ^ 2) = 30 + 10 * sqrt 2 := 
by
  simp only [h1, h2, h3, sqrt_sqr]
  repeat { rw ←add_assoc }
  rw ←mul_assoc
  rw [(10 + 10) + 10 * sqrt 2]
  sorry

end trapezoid_perimeter_l183_183465


namespace hyperbola_equation_l183_183015

    -- Define an environment for the hyperbola
    noncomputable def hyperbola (a b : ℝ) : Prop :=
      ∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

    -- Define the standard form of the parabola
    def parabola_focus : ℝ := real.sqrt 3

    -- Define the condition for one asymptote having a slope of sqrt(2)
    def asymptote_slope (a b : ℝ) : Prop :=
      b = real.sqrt 2 * a

    -- Define the condition for the right focus coinciding with the parabola focus
    def right_focus (a b : ℝ) : Prop :=
      (real.sqrt 3)^2 = a^2 + b^2

    -- Main theorem to prove
    theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
      (h3 : asymptote_slope a b) (h4 : right_focus a b) :
      hyperbola 1 (real.sqrt 2) := by
    -- proof omitted
    sorry
    
end hyperbola_equation_l183_183015


namespace temperature_rise_result_l183_183176

def initial_temperature : ℤ := -2
def rise : ℤ := 3

theorem temperature_rise_result : initial_temperature + rise = 1 := 
by 
  sorry

end temperature_rise_result_l183_183176


namespace minimum_queries_needed_l183_183887

-- Declare the parameters and the problem as a theorem statement in Lean
theorem minimum_queries_needed {n : Nat} (hn : n = 2005) :
  ∃ k : Nat, (∀ cards : Finset (Fin 2005), cards.card = 3 → setOfQueries cards) ∧ k = 1003 :=
sorry

end minimum_queries_needed_l183_183887


namespace sqrt_product_simplify_l183_183283

theorem sqrt_product_simplify (x : ℝ) (hx : 0 ≤ x):
  Real.sqrt (48*x) * Real.sqrt (3*x) * Real.sqrt (50*x) = 60 * x * Real.sqrt x := 
by
  sorry

end sqrt_product_simplify_l183_183283


namespace third_oldest_bev_l183_183314

theorem third_oldest_bev (D B E A C : ℕ) (h1 : D > B) (h2 : B > E) (h3 : A > E) (h4 : B > A) (h5 : C > B) : 
  third_oldest D B E A C = B :=
sorry

end third_oldest_bev_l183_183314


namespace linear_function_difference_l183_183857

theorem linear_function_difference (g : ℝ → ℝ) (h_linear : ∀ x y z w : ℝ, g(x) - g(y) = (g(z) - g(w)) / (x - y) * (z - w))
  (h_condition : g(8) - g(3) = 15) : g(13) - g(3) = 30 :=
sorry

end linear_function_difference_l183_183857


namespace cos_double_angle_transform_l183_183432

theorem cos_double_angle_transform (α : ℝ) (h1 : cos (α - π / 3) = 2 / 3) (h2 : 0 < α ∧ α < π / 2) : 
  cos (2 * α - 2 * π / 3) = -1 / 9 :=
by
  sorry

end cos_double_angle_transform_l183_183432


namespace sufficient_but_not_necessary_condition_l183_183371

variable (a b x y : ℝ)

theorem sufficient_but_not_necessary_condition (ha : a > 0) (hb : b > 0) :
  ((x > a ∧ y > b) → (x + y > a + b ∧ x * y > a * b)) ∧
  ¬((x + y > a + b ∧ x * y > a * b) → (x > a ∧ y > b)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l183_183371


namespace chlorine_discount_l183_183245

theorem chlorine_discount
  (cost_chlorine : ℕ)
  (cost_soap : ℕ)
  (num_chlorine : ℕ)
  (num_soap : ℕ)
  (discount_soap : ℤ)
  (total_savings : ℤ)
  (price_chlorine : ℤ)
  (price_soap_after_discount : ℤ)
  (total_price_before_discount : ℤ)
  (total_price_after_discount : ℤ)
  (goal_discount : ℤ) :
  cost_chlorine = 10 →
  cost_soap = 16 →
  num_chlorine = 3 →
  num_soap = 5 →
  discount_soap = 25 →
  total_savings = 26 →
  price_soap_after_discount = (1 - (discount_soap / 100)) * 16 →
  total_price_before_discount = (num_chlorine * cost_chlorine) + (num_soap * cost_soap) →
  total_price_after_discount = (num_chlorine * ((100 - goal_discount) / 100) * cost_chlorine) + (num_soap * 12) →
  total_price_before_discount - total_price_after_discount = total_savings →
  goal_discount = 20 :=
by
  intros
  sorry

end chlorine_discount_l183_183245


namespace apple_cost_l183_183878

theorem apple_cost (A : ℝ) (h_discount : ∃ (n : ℕ), 15 = (5 * (5: ℝ) * A + 3 * 2 + 2 * 3 - n)) : A = 1 :=
by
  sorry

end apple_cost_l183_183878


namespace derivative_of_m_l183_183439

noncomputable def m (x : ℝ) : ℝ := (2 : ℝ)^x / (1 + x)

theorem derivative_of_m (x : ℝ) : 
  deriv m x = (2^x * (1 + x) * Real.log 2 - 2^x) / (1 + x)^2 :=
by
  sorry

end derivative_of_m_l183_183439


namespace total_toys_l183_183274

theorem total_toys (K A L : ℕ) (h1 : A = K + 30) (h2 : L = 2 * K) (h3 : K + A = 160) : 
    K + A + L = 290 :=
by
  sorry

end total_toys_l183_183274


namespace simplify_complex_fraction_l183_183906

theorem simplify_complex_fraction :
  (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i :=
by
  sorry

end simplify_complex_fraction_l183_183906


namespace compute_c_plus_d_l183_183109

theorem compute_c_plus_d (c d : ℝ) 
  (h1 : c^3 - 18 * c^2 + 25 * c - 75 = 0) 
  (h2 : 9 * d^3 - 72 * d^2 - 345 * d + 3060 = 0) : 
  c + d = 10 := 
sorry

end compute_c_plus_d_l183_183109


namespace algebraic_expression_difference_l183_183138

theorem algebraic_expression_difference (k : Nat) :
  (1 - ∑ i in finset.range (2 * k), (-1)^i / (i+1)) - (1 - ∑ i in finset.range (2 * (k + 1)), (-1)^i / (i+1)) 
  = (1 / ↑(2 * k + 1)) - (1 / ↑(2 * k + 2)) :=
by
  sorry

end algebraic_expression_difference_l183_183138


namespace distance_P_to_plane_l183_183836

def distance_point_to_plane (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_P_to_plane : distance_point_to_plane 2 1 1 3 4 12 4 = 2 := 
by 
  sorry 

end distance_P_to_plane_l183_183836


namespace area_of_circumcircle_of_triangle_l183_183838

noncomputable def area_of_circumcircle (A B C : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] (angle_B : ℝ) (AB BC : ℝ) :=
let AC := real.sqrt (AB^2 + BC^2 - 2 * AB * BC * real.cos angle_B) in
let radius := AC / (2 * real.sin angle_B) in
π * radius^2

theorem area_of_circumcircle_of_triangle {A B C : Type*} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] :
  area_of_circumcircle A B C (π / 3) 8 5 = 49 * π / 3 := sorry

end area_of_circumcircle_of_triangle_l183_183838


namespace JerushaEarnings_is_68_l183_183846

-- Define the conditions
def LottiesEarnings := ℝ
def JerushaEarnings (L : LottiesEarnings) := 4 * L

-- Condition 1: Jerusha's earning is 4 times Lottie's earnings
def condition1 (L : LottiesEarnings) : Prop := JerushaEarnings L = 4 * L

-- Condition 2: The total earnings of Jerusha and Lottie is $85
def condition2 (L : LottiesEarnings) : Prop := JerushaEarnings L + L = 85

-- The theorem to prove Jerusha's earnings is $68
theorem JerushaEarnings_is_68 (L : LottiesEarnings) (h1 : condition1 L) (h2 : condition2 L) : JerushaEarnings L = 68 := 
by 
  sorry

end JerushaEarnings_is_68_l183_183846


namespace sales_this_month_l183_183509

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end sales_this_month_l183_183509


namespace find_transform_l183_183179

structure Vector3D (α : Type) := (x y z : α)

def T (u : Vector3D ℝ) : Vector3D ℝ := sorry

axiom linearity (a b : ℝ) (u v : Vector3D ℝ) : T (Vector3D.mk (a * u.x + b * v.x) (a * u.y + b * v.y) (a * u.z + b * v.z)) = 
                      Vector3D.mk (a * (T u).x + b * (T v).x) (a * (T u).y + b * (T v).y) (a * (T u).z + b * (T v).z)

axiom cross_product (u v : Vector3D ℝ) : T (Vector3D.mk (u.y * v.z - u.z * v.y) (u.z * v.x - u.x * v.z) (u.x * v.y - u.y * v.x)) = 
                    (Vector3D.mk ((T u).y * (T v).z - (T u).z * (T v).y) ((T u).z * (T v).x - (T u).x * (T v).z) ((T u).x * (T v).y - (T u).y * (T v).x))

axiom transform1 : T (Vector3D.mk 3 3 7) = Vector3D.mk 2 (-4) 5
axiom transform2 : T (Vector3D.mk (-2) 5 4) = Vector3D.mk 6 1 0

theorem find_transform : T (Vector3D.mk 5 15 11) = Vector3D.mk a b c := sorry

end find_transform_l183_183179


namespace number_of_elements_in_B_l183_183009

def A : Set ℕ := {2, 3, 4}

def B : Set ℕ := {x | ∃ (m n : ℕ), m ∈ A ∧ n ∈ A ∧ m ≠ n ∧ x = m * n}

theorem number_of_elements_in_B : Fintype.card B = 3 :=
by
  sorry

end number_of_elements_in_B_l183_183009


namespace part_a_part_b_l183_183222

noncomputable def exists_diff_func (f : ℝ → ℝ) (domain : Set ℝ) (f_differentiable : ∀ x ∈ domain, DifferentiableAt ℝ f x) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x ∈ domain, g (deriv f x) = x)

theorem part_a :
  exists_diff_func f (Set.Ioi 0) (λ x hx, differentiable_at_real_of_Ioo hx) :=
sorry

theorem part_b :
  ¬ exists_diff_func f Set.univ differentiable_at_real :=
sorry

end part_a_part_b_l183_183222


namespace probability_of_eventA_is_1_over_4_probability_of_eventB_is_1_over_2_l183_183185

-- Conditions for problem (I)
def drawnBallsWithoutReplacement := [(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), (3,1), (3,2), (3,4), (4,1), (4,2), (4,3) : List (ℕ × ℕ)]

def eventA (p : ℕ × ℕ) : Prop := ∃ a b, p = (a, b) ∧ a % 2 = 0 ∧ (a + b) % 3 = 0

-- Conditions for problem (II)
def drawnBallsWithReplacement := [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4) : List (ℕ × ℕ)]

def eventB (p : ℕ × ℕ) : Prop := ∃ a b, p = (a, b) ∧ a^2 + b^2 < 16

-- Correct answers
theorem probability_of_eventA_is_1_over_4 : (drawnBallsWithoutReplacement.filter eventA).length = (drawnBallsWithoutReplacement.length / 4) := sorry

theorem probability_of_eventB_is_1_over_2 : (drawnBallsWithReplacement.filter eventB).length = (drawnBallsWithReplacement.length / 2) := sorry

end probability_of_eventA_is_1_over_4_probability_of_eventB_is_1_over_2_l183_183185


namespace total_birds_remaining_l183_183547

-- Definitions from conditions
def initial_grey_birds : ℕ := 40
def additional_white_birds : ℕ := 6
def white_birds (grey_birds: ℕ) : ℕ := grey_birds + additional_white_birds
def remaining_grey_birds (grey_birds: ℕ) : ℕ := grey_birds / 2

-- Proof problem
theorem total_birds_remaining : 
  let grey_birds := initial_grey_birds;
  let white_birds_next_to_cage := white_birds(grey_birds);
  let grey_birds_remaining := remaining_grey_birds(grey_birds);
  (grey_birds_remaining + white_birds_next_to_cage) = 66 :=
by {
  sorry
}

end total_birds_remaining_l183_183547


namespace minimum_pencils_needed_l183_183599

theorem minimum_pencils_needed (red_pencils blue_pencils : ℕ) (total_pencils : ℕ) 
  (h_red : red_pencils = 7) (h_blue : blue_pencils = 4) (h_total : total_pencils = red_pencils + blue_pencils) :
  (∃ n : ℕ, n = 8 ∧ n ≤ total_pencils ∧ (∀ m : ℕ, m < 8 → (m < red_pencils ∨ m < blue_pencils))) :=
by
  sorry

end minimum_pencils_needed_l183_183599


namespace max_tied_teams_in_tournament_l183_183454

theorem max_tied_teams_in_tournament :
  ∀ (n : ℕ), n = 8 →
  (∀ (team : ℕ), team < 8 → 
    (∃ (wins : ℕ), wins ≤ 7 ∧ wins ≥ 0)) →
  (∃ m, m = 3 ∧ 
    ∀ k, k > 3 → 
    ¬(∃ (tie_teams : ℕ → ℕ), 
      (∀ t, t < k → tie_teams t ≤ 7) ∧ 
      (∑ t in finset.range k, tie_teams t = 28))) :=
begin
  -- Definitions and conditions
  intro n,
  intro n_eq_8,
  intros H,
  -- Construct proof here
  sorry,
end

end max_tied_teams_in_tournament_l183_183454


namespace simplify_expression_l183_183645

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183645


namespace distance_to_asymptote_l183_183940

/-- Define the hyperbola equation as a predicate --/
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

/-- Define the asymptote equations as predicates --/
def asymptote1 (x y : ℝ) : Prop := 3 * x - 4 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- Define the distance formula from a point to a line --/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2))

/-- Proof statement that the distance from (3,0) to asymptote is 9/5 --/
theorem distance_to_asymptote : 
  distance_from_point_to_line 3 0 3 (-4) 0 = 9 / 5 :=
by
  -- the main proof computation goes here
  sorry

end distance_to_asymptote_l183_183940


namespace mixed_groups_count_l183_183982

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l183_183982


namespace dirichlet_not_riemann_integrable_dirichlet_lebesgue_integrable_and_value_l183_183920

noncomputable def dirichlet_function (x : ℝ) : ℝ :=
  if ¬ ∃ r : ℚ, ↑r = x then 1 else 0

open measure_theory 

theorem dirichlet_not_riemann_integrable :
  ¬ ∃ f, is_Riemann_integrable_on (dirichlet_function) (set.Icc 0 1) :=
sorry

theorem dirichlet_lebesgue_integrable_and_value :
  integrable_on dirichlet_function (set.Icc 0 1) measure_theory.measure_space.volume ∧ 
  ∫ x in (set.Icc 0 1), (dirichlet_function x) = 1 :=
sorry

end dirichlet_not_riemann_integrable_dirichlet_lebesgue_integrable_and_value_l183_183920


namespace log_product_identity_l183_183284

theorem log_product_identity :
    (Real.log 9 / Real.log 8) * (Real.log 32 / Real.log 9) = 5 / 3 := 
by 
  sorry

end log_product_identity_l183_183284


namespace find_S_l183_183889

theorem find_S :
  (1/4 : ℝ) * (1/6 : ℝ) * S = (1/5 : ℝ) * (1/8 : ℝ) * 160 → S = 96 :=
by
  intro h
  -- Proof is omitted
  sorry 

end find_S_l183_183889


namespace find_a_given_solution_set_l183_183173

theorem find_a_given_solution_set :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 ↔ x^2 + a * x + 6 ≤ 0) → a = -5 :=
by
  sorry

end find_a_given_solution_set_l183_183173


namespace least_positive_angle_l183_183712

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end least_positive_angle_l183_183712


namespace largest_option_l183_183758

-- Let a and b be real numbers such that 0 < a < b and a + b = 1
variables {a b : ℝ}
variable h1 : 0 < a
variable h2 : a < b
variable h3 : a + b = 1

-- Define the four options
def option_A := (1 : ℝ) / 2
def option_B := a^2 + b^2
def option_C := 2 * a * b
def option_D := b

-- We need to prove that option D is the largest
theorem largest_option : option_D > option_A ∧ option_D > option_B ∧ option_D > option_C :=
by
  sorry

end largest_option_l183_183758


namespace problem_solution_l183_183490

noncomputable def greatest_integer_not_exceeding (z : ℝ) : ℤ := Int.floor z

theorem problem_solution (x : ℝ) (y : ℝ) 
  (h1 : y = 4 * greatest_integer_not_exceeding x + 4)
  (h2 : y = 5 * greatest_integer_not_exceeding (x - 3) + 7)
  (h3 : x > 3 ∧ ¬ ∃ (n : ℤ), x = ↑n) :
  64 < x + y ∧ x + y < 65 :=
by
  sorry

end problem_solution_l183_183490


namespace ada_original_seat_l183_183081

theorem ada_original_seat:
  (friends : List string) (seats : Fin 5) 
  (Ada_exited : friends.contains "Ada")
  (Bea_moves_right : true) 
  (Ceci_stays : true)
  (Dee_moves_left : true)
  (Edie_Fay_switch : true)
  (Ada_returns : true) 
  (Ada_seat : seats = 3) :
  (original_ada_seat : Fin 5) := 
  original_ada_seat = 4 :=
sorry

end ada_original_seat_l183_183081


namespace congruence_problem_l183_183430

theorem congruence_problem (x : ℤ) (h : 5 * x + 9 ≡ 3 [MOD 19]) : 3 * x + 14 ≡ 18 [MOD 19] :=
sorry

end congruence_problem_l183_183430


namespace simplify_expression_eq_square_l183_183629

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183629


namespace num_solns_in_interval_l183_183111

theorem num_solns_in_interval :
  (∃ x ∈ set.Icc (0 : ℝ) real.pi, 
    6 * (real.sin x)^4 + 6 * (real.cos x)^4 
  - 4 * (real.sin x)^6 - 4 * (real.cos x)^6 
  = 2 * ((real.sin x)^2 + (real.cos x)^2)) :=
sorry

end num_solns_in_interval_l183_183111


namespace train_length_l183_183559

theorem train_length (x y : ℝ) : (∃ (L_A : ℝ), train_A_speed = 72 ∧ train_B_speed = x ∧ platform_length = 240 ∧ time_to_cross_platform = 26 ∧ L_A = 280) :=
by
  let train_A_speed_kmph := 72
  let train_A_speed_mps := 20
  let platform_length := 240
  let time_to_cross_platform := 26
  have train_length : 240 + L_A = 20 * 26 := by sorry
  have L_A := 280
  exact ⟨L_A, train_A_speed, train_B_speed, platform_length, time_to_cross_platform⟩

end train_length_l183_183559


namespace all_black_after_rotations_l183_183592

theorem all_black_after_rotations : 
  ∀ (grid : array (fin 4) (array (fin 4) Prop)),
  (∀ i j, (i = j ∨ i + j = 3) -> grid[i][j] ∧ (¬ ∃ k l, k ≠ l ∧ k + l ≠ 3 -> ¬grid[k][l])) →
  (∀ i j, i = j ∧ i + j = 3 ∨ i ≠ j ∧ i + j ≠ 3 -> (rotate_90_clockwise(grid))[i][j] = true ∧ rotate_90_clockwise(grid)[i][j] = true) →
  (∀ i j, i ≠ j ∧ i + j ≠ 3 ∧ j = i -> (rotate_90_clockwise(rotate_90_clockwise(grid)))[i][j] = true) →
  ∀ i j, grid[i][j] = true :=
begin
  sorry
end

end all_black_after_rotations_l183_183592


namespace sum_harmonic_geq_half_n_plus_two_l183_183049

theorem sum_harmonic_geq_half_n_plus_two (n : ℕ) (hn : n > 0) :
  (finset.range (2^n + 1)).sum (λ k, 1 / (k + 1 : ℝ)) ≥ (n + 2) / 2 :=
sorry

end sum_harmonic_geq_half_n_plus_two_l183_183049


namespace general_term_proof_sum_bound_proof_l183_183036

noncomputable def geom_seq (a₄ : ℕ) (mean_a₂_a₃ : ℕ) (a : ℕ → ℕ) : Prop :=
a 4 = a₄ ∧ (a 2 + a 3) / 2 = mean_a₂_a₃

noncomputable def geom_seq_general_term (a₄ : ℕ) (mean_a₂_a₃ : ℕ) (a : ℕ → ℕ) : Prop :=
∀ n, a n = 3^n

theorem general_term_proof (a : ℕ → ℕ) (a₄ mean_a₂_a₃ : ℕ) :
geom_seq a₄ mean_a₂_a₃ a → geom_seq_general_term a₄ mean_a₂_a₃ a :=
by sorry

noncomputable def seq_sum_bound (T : ℕ → ℝ) : Prop :=
∀ n, T n < 1 / 2

noncomputable def T (n : ℕ) : ℝ := 
∑ i in finset.range n, 1 / (2 * i.succ_nat ^ 2 - 1)

theorem sum_bound_proof : seq_sum_bound T :=
by sorry

end general_term_proof_sum_bound_proof_l183_183036


namespace find_lambda_l183_183446

variables (e1 e2 : ℝ → ℝ → ℝ)
variable (λ : ℝ)

-- Assume e1 and e2 are unit vectors
axioms (he1 : ∥e1∥ = 1) (he2 : ∥e2∥ = 1) 
-- Assume the angle between e1 and e2 is π/3
axiom angle_e1_e2 : real.angle (e1 1 0) (e2 0 1) = real.pi / 3

-- Assume vector a = e1 + λ * e2 with |a| = sqrt(3)/2
def a := e1 + λ • e2
axiom ha : ∥a∥ = real.sqrt 3 / 2

theorem find_lambda : λ = -1 / 2 := by
  sorry

end find_lambda_l183_183446


namespace john_savings_percentage_l183_183479

theorem john_savings_percentage :
  ∀ (savings discounted_price total_price original_price : ℝ),
  savings = 4.5 →
  total_price = 49.5 →
  total_price = discounted_price * 1.10 →
  original_price = discounted_price + savings →
  (savings / original_price) * 100 = 9 := by
  intros
  sorry

end john_savings_percentage_l183_183479


namespace carpet_rate_l183_183338

theorem carpet_rate (length breadth cost area: ℝ) (h₁ : length = 13) (h₂ : breadth = 9) (h₃ : cost = 1872) (h₄ : area = length * breadth) :
  cost / area = 16 := by
  sorry

end carpet_rate_l183_183338


namespace distinct_natural_primes_l183_183336

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem distinct_natural_primes :
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧
  is_prime (a * b + c * d) ∧
  is_prime (a * c + b * d) ∧
  is_prime (a * d + b * c) := by
  sorry

end distinct_natural_primes_l183_183336


namespace incorrect_statements_l183_183080

variable {Ω : Type*} [probability_space Ω]

def A1 : event Ω := sorry
def A2 : event Ω := sorry
def A3 : event Ω := sorry

axiom P_A1 : P A1 = 0.2
axiom P_A2 : P A2 = 0.3
axiom P_A3 : P A3 = 0.5
axiom P_non_neg {e : event Ω} : 0 ≤ P e 

theorem incorrect_statements :
  (¬(mutually_exclusive (A1 ∪ A2) A3 ∧ (A1 ∪ A2) ∪ A3 = compl (A1 ∪ A2))) ∧
  (¬((A1 ∪ A2) ∪ A3 = Ω)) ∧
  (¬(P (A2 ∪ A3) = 0.8)) ∧
  (P (A1 ∪ A2) ≤ 0.5) :=
by
  sorry

end incorrect_statements_l183_183080


namespace highest_average_speed_interval_l183_183535

theorem highest_average_speed_interval
  (d : ℕ → ℕ)
  (h0 : d 0 = 45)        -- Distance from 0 to 30 minutes
  (h1 : d 1 = 135)       -- Distance from 30 to 60 minutes
  (h2 : d 2 = 255)       -- Distance from 60 to 90 minutes
  (h3 : d 3 = 325) :     -- Distance from 90 to 120 minutes
  (1 / 2) * ((d 2 - d 1 : ℕ) : ℝ) > 
  max ((1 / 2) * ((d 1 - d 0 : ℕ) : ℝ)) 
      (max ((1 / 2) * ((d 3 - d 2 : ℕ) : ℝ))
          ((1 / 2) * ((d 3 - d 1 : ℕ) : ℝ))) :=
by
  sorry

end highest_average_speed_interval_l183_183535


namespace number_of_persons_l183_183617

theorem number_of_persons
    (total_amount : ℕ) 
    (amount_per_person : ℕ) 
    (h1 : total_amount = 42900) 
    (h2 : amount_per_person = 1950) :
    total_amount / amount_per_person = 22 :=
by
  sorry

end number_of_persons_l183_183617


namespace grapes_total_sum_l183_183260

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l183_183260


namespace bank_balance_after_five_years_l183_183247

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem bank_balance_after_five_years :
  let P0 := 5600
  let r1 := 0.03
  let r2 := 0.035
  let r3 := 0.04
  let r4 := 0.045
  let r5 := 0.05
  let D := 2000
  let A1 := compoundInterest P0 r1 1 1
  let A2 := compoundInterest A1 r2 1 1
  let A3 := compoundInterest (A2 + D) r3 1 1
  let A4 := compoundInterest A3 r4 1 1
  let A5 := compoundInterest A4 r5 1 1
  A5 = 9094.2 := by
  sorry

end bank_balance_after_five_years_l183_183247


namespace find_interest_rate_l183_183818

-- Conditions
def principal1 : ℝ := 100
def rate1 : ℝ := 0.05
def time1 : ℕ := 48

def principal2 : ℝ := 600
def time2 : ℕ := 4

-- The given interest produced by the first amount
def interest1 : ℝ := principal1 * rate1 * time1

-- The interest produced by the second amount should be the same
def interest2 (rate2 : ℝ) : ℝ := principal2 * rate2 * time2

-- The interest rate to prove
def rate2_correct : ℝ := 0.1

theorem find_interest_rate :
  ∃ rate2 : ℝ, interest2 rate2 = interest1 ∧ rate2 = rate2_correct :=
by
  sorry

end find_interest_rate_l183_183818


namespace ratio_of_population_is_correct_l183_183824

noncomputable def ratio_of_population (M W C : ℝ) : ℝ :=
  (M / (W + C)) * 100

theorem ratio_of_population_is_correct
  (M W C : ℝ) 
  (hW: W = 0.9 * M)
  (hC: C = 0.6 * (M + W)) :
  ratio_of_population M W C = 49.02 := 
by
  sorry

end ratio_of_population_is_correct_l183_183824


namespace question_1_monotonic_and_extreme_values_question_2_range_of_k_for_two_zeros_l183_183044

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x * exp x - (x^2) / 2 - x
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * exp x - x
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := f k x - g k x

theorem question_1_monotonic_and_extreme_values :
  (∀ x : ℝ, f 1 x = x * exp x - (x^2) / 2 - x)
  → (∀ x ∈ Icc (-∞ : ℝ) (-1 : ℝ), (f 1)' x > 0)
  ∧ (∀ x ∈ Icc (-1 : ℝ) 0, (f 1)' x < 0)
  ∧ (∀ x ∈ Icc 0 (∞ : ℝ), (f 1)' x > 0)
  ∧ (f 1 (-1) = (1 / 2 - 1 / exp 1))
  ∧ (f 1 0 = 0) :=
  by
    sorry

theorem question_2_range_of_k_for_two_zeros :
  (∀ x : ℝ, F x = k * (x - 1) * exp x - (x^2) / 2)
  → (∃ k : ℝ, F x = 0)
  ↔ k ∈ set_of (λ k, k < 0) :=
  by
    sorry

end question_1_monotonic_and_extreme_values_question_2_range_of_k_for_two_zeros_l183_183044


namespace smallest_area_6_8_l183_183202

noncomputable def smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) : ℝ :=
  let c1 := Real.sqrt (a^2 + b^2)
  let area1 := 0.5 * a * b
  let c2 := Real.sqrt (b^2 - a^2)
  let area2 := if h : b > a then 0.5 * a * c2 else 0.5 * b * c2
  min area1 area2

theorem smallest_area_6_8 : smallest_area_of_right_triangle 6 8 6 8 = 15.87 :=
sorry

end smallest_area_6_8_l183_183202


namespace calculate_f12_l183_183673

def f : ℕ → ℕ
| 1       := 1
| 2       := 4
| (n + 3) := f (n + 2) - f (n + 1) + 2 * (n + 3)

theorem calculate_f12 : f 12 = 25 := by
  sorry

end calculate_f12_l183_183673


namespace initial_amount_l183_183237

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end initial_amount_l183_183237


namespace finitely_many_good_integers_l183_183386

theorem finitely_many_good_integers (f : ℕ → ℕ) (m : ℕ) (positive_leading_coeff : ∃ a b, ∀ n, f(n) = a * n^m + b ∧ a > 0) :
  ∃ N, ∀ n > N, ¬∃ kn : ℕ, kn > 0 ∧ n! + 1 = (f n) ^ kn :=
sorry

end finitely_many_good_integers_l183_183386


namespace problem_a2016_b2016_l183_183251

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 0

theorem problem_a2016_b2016 :
  {a, 1, b / a} = {a^2, a + b, 0} → a = 1 → b = 0 → a ^ 2016 + b ^ 2016 = 1 :=
by
  sorry

end problem_a2016_b2016_l183_183251


namespace inscribed_circle_diameter_l183_183340

-- Define the sides of the triangle
def AB : ℝ := 13
def AC : ℝ := 5
def BC : ℝ := 12

-- Define the semiperimeter
def s : ℝ := (AB + AC + BC) / 2

-- Define the area using Heron's formula
def K : ℝ := real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- Define the diameter of the inscribed circle
def d : ℝ := 2 * r

-- Theorem statement
theorem inscribed_circle_diameter : d = 4 := by
  -- Proof will go here
  sorry

end inscribed_circle_diameter_l183_183340


namespace f50_zero_values_count_l183_183110

noncomputable def f0 (x : ℝ) : ℝ :=
if x < -50 then x + 100
else if x < 50 then -x
else x - 100

noncomputable def f : ℕ → ℝ → ℝ
| 0     => f0
| (n+1) => λ x, |(f n x)| - 1

theorem f50_zero_values_count : 
  ∃ S : set ℝ, (∀ x ∈ S, f 50 x = 0) ∧ S.finite ∧ S.card = 149 :=
by
  sorry

end f50_zero_values_count_l183_183110


namespace mixed_groups_count_l183_183996

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183996


namespace function_shifts_equiv_l183_183781

-- Define the initial function
def f (x : ℝ) : ℝ := 1 / x

-- Define the transformation properties
def shift_right (g : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, g (x - a)
def shift_down (g : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, g x - b

-- State the theorem
theorem function_shifts_equiv :
  ∀ x : ℝ, ((shift_down (shift_right f 2) 1) x) = (1 / (x - 2) - 1) :=
by
  sorry

end function_shifts_equiv_l183_183781


namespace megan_markers_final_count_l183_183128

theorem megan_markers_final_count :
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  final_markers = 582 :=
by
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  have h : final_markers = 582 := sorry
  exact h

end megan_markers_final_count_l183_183128


namespace sales_this_month_l183_183510

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end sales_this_month_l183_183510


namespace coefficient_x15_in_2x_plus_3_pow_20_l183_183463

theorem coefficient_x15_in_2x_plus_3_pow_20 :
  (finset.range 21).sum (λ k, if k = 15 then nat.choose 20 k * 2^k * 3^(20 - k) else 0) = 123483323328 :=
by {
  sorry
}

end coefficient_x15_in_2x_plus_3_pow_20_l183_183463


namespace mixed_groups_count_l183_183967

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183967


namespace valid_parameterizations_l183_183165

open Real

def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2 * p.1 - 7

def valid_parametrization (p d : ℝ × ℝ) : Prop :=
  lies_on_line p ∧ is_scalar_multiple d (1, 2)

theorem valid_parameterizations :
  valid_parametrization (4, 1) (-2, -4) ∧ 
  ¬ valid_parametrization (12, 17) (5, 10) ∧ 
  valid_parametrization (3.5, 0) (1, 2) ∧ 
  valid_parametrization (-2, -11) (0.5, 1) ∧ 
  valid_parametrization (0, -7) (10, 20) :=
by {
  sorry
}

end valid_parameterizations_l183_183165


namespace sin_minus_cos_value_l183_183569

theorem sin_minus_cos_value (α : ℝ) (h1 : α ∈ set.Ioo (π / 2) π) 
    (h2 : (sin (π - α) - cos (π + α)) = √2 / 3) : 
    (sin α - cos α) = 4 / 3 := 
begin
  sorry
end

end sin_minus_cos_value_l183_183569


namespace monotonicity_of_g_inequality_f_l183_183412

noncomputable theory

-- Define the function f(x) for any real number a
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

-- Part 1: Show monotonicity properties for g(x) when a = 1
def g (x : ℝ) : ℝ := (f 1 x) / Real.exp x

-- Part 1: Theorem statement for the monotonicity of g(x)
theorem monotonicity_of_g : 
  (∀ x ∈ Ioo (-Real.pi/2) 0, g x decreases_on Ioo (-Real.pi/2) 0) ∧ 
  (∀ x ∈ Ioo 0 (3 * Real.pi / 2), g x increases_on Ioo 0 (3 * Real.pi / 2)) :=
sorry

-- Part 2: Theorem statement for the inequality when a = -3
theorem inequality_f : 
  ∀ x ∈ Ioi 0, f (-3) x < Real.exp x + x + 1 - 2 * Real.exp (-2 * x) :=
sorry

end monotonicity_of_g_inequality_f_l183_183412


namespace max_objective_value_l183_183810

theorem max_objective_value (x y : ℝ) (h1 : x - y - 2 ≥ 0) (h2 : 2 * x + y - 2 ≤ 0) (h3 : y + 4 ≥ 0) :
  ∃ (z : ℝ), z = 4 * x + 3 * y ∧ z ≤ 8 :=
sorry

end max_objective_value_l183_183810


namespace sum_of_remaining_numbers_l183_183087

def remaining_numbers_sum (n : ℕ) : ℕ :=
  let numbers := list.range (n + 1).tail
  let cross_out (lst : list ℕ) (k : ℕ) : list ℕ :=
    if h : lst.length > 2 then
      let i := (k - 1) % lst.length
      (lst.take i) ++ (lst.drop (i + 1))
    else lst
  let rec loop (lst : list ℕ) (k : ℕ) (step : ℕ) : list ℕ :=
    if lst.length ≤ 2 then lst
    else loop (cross_out lst (step + k)) k (step + 1)
  10 (loop numbers 3 0).sum

theorem sum_of_remaining_numbers : remaining_numbers_sum 10 = 10 :=
  by sorry

end sum_of_remaining_numbers_l183_183087


namespace JerushaEarnings_is_68_l183_183847

-- Define the conditions
def LottiesEarnings := ℝ
def JerushaEarnings (L : LottiesEarnings) := 4 * L

-- Condition 1: Jerusha's earning is 4 times Lottie's earnings
def condition1 (L : LottiesEarnings) : Prop := JerushaEarnings L = 4 * L

-- Condition 2: The total earnings of Jerusha and Lottie is $85
def condition2 (L : LottiesEarnings) : Prop := JerushaEarnings L + L = 85

-- The theorem to prove Jerusha's earnings is $68
theorem JerushaEarnings_is_68 (L : LottiesEarnings) (h1 : condition1 L) (h2 : condition2 L) : JerushaEarnings L = 68 := 
by 
  sorry

end JerushaEarnings_is_68_l183_183847


namespace distance_to_asymptote_l183_183941

/-- Define the hyperbola equation as a predicate --/
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

/-- Define the asymptote equations as predicates --/
def asymptote1 (x y : ℝ) : Prop := 3 * x - 4 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- Define the distance formula from a point to a line --/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2))

/-- Proof statement that the distance from (3,0) to asymptote is 9/5 --/
theorem distance_to_asymptote : 
  distance_from_point_to_line 3 0 3 (-4) 0 = 9 / 5 :=
by
  -- the main proof computation goes here
  sorry

end distance_to_asymptote_l183_183941


namespace binary_to_decimal_1101_l183_183621

theorem binary_to_decimal_1101 : "1101"_2 = 13 :=
by sorry

end binary_to_decimal_1101_l183_183621


namespace unique_solution_condition_l183_183029

-- Define p and q as real numbers
variables (p q : ℝ)

-- The Lean statement to prove a unique solution when q ≠ 4
theorem unique_solution_condition : (∀ x : ℝ, (4 * x - 7 + p = q * x + 2) ↔ (q ≠ 4)) :=
by
  sorry

end unique_solution_condition_l183_183029


namespace smallest_n_for_triangle_area_l183_183290

theorem smallest_n_for_triangle_area :
  ∃ n : ℕ, 10 * n^4 - 8 * n^3 - 52 * n^2 + 32 * n - 24 > 10000 ∧ ∀ m : ℕ, 
  (m < n → ¬ (10 * m^4 - 8 * m^3 - 52 * m^2 + 32 * m - 24 > 10000)) :=
sorry

end smallest_n_for_triangle_area_l183_183290


namespace f_positive_l183_183762

variable (f : ℝ → ℝ)

-- f is a differentiable function on ℝ
variable (hf : differentiable ℝ f)

-- Condition: (x+1)f(x) + x f''(x) > 0
variable (H : ∀ x, (x + 1) * f x + x * (deriv^[2]) f x > 0)

-- Prove: ∀ x, f x > 0
theorem f_positive : ∀ x, f x > 0 := 
by
  sorry

end f_positive_l183_183762


namespace tan_theta_solution_l183_183733

theorem tan_theta_solution (θ : ℝ) (h : (1 + sin (2 * θ)) / (cos θ ^ 2 - sin θ ^ 2) = -3) : 
  tan θ = 2 := 
by 
  sorry

end tan_theta_solution_l183_183733


namespace total_surface_area_correct_l183_183206

-- Defining the dimensions of the rectangular solid
def length := 10
def width := 9
def depth := 6

-- Definition of the total surface area of a rectangular solid
def surface_area (l w d : ℕ) := 2 * (l * w + w * d + l * d)

-- Proposition that the total surface area for the given dimensions is 408 square meters
theorem total_surface_area_correct : surface_area length width depth = 408 := 
by
  sorry

end total_surface_area_correct_l183_183206


namespace sin_150_eq_half_l183_183289

open Real

theorem sin_150_eq_half : sin (150 * pi / 180) = 1 / 2 :=
by
  -- Given the conditions
  have h1 : 150 * pi / 180 = pi - (30 * pi / 180) := by norm_num
  have h2 : sin (pi - (30 * pi / 180)) = sin (30 * pi / 180) := Real.sin_sub_pi_div_two 30
  have h3 : sin (30 * pi / 180) = 1 / 2 := by norm_num ; rw [Real.sin_pi_div_six]
  -- Combining the conditions to get the desired result
  rw [h1, h2, h3]
  -- sorry  -- This placeholder can be replaced by the actual proof steps if necessary

end sin_150_eq_half_l183_183289


namespace f_monotonically_increasing_on_interval_l183_183143

noncomputable def f (x : ℝ) : ℝ := 5 * (sin (3 * x)) + 5 * (sqrt 3) * (cos (3 * x))

theorem f_monotonically_increasing_on_interval :
  monotone_on f (set.Icc 0 (π / 20)) :=
sorry

end f_monotonically_increasing_on_interval_l183_183143


namespace domain_of_g_l183_183296

theorem domain_of_g (t : ℝ) : (t - 1)^2 + (t + 1)^2 + t ≠ 0 :=
  by
  sorry

end domain_of_g_l183_183296


namespace complex_fraction_l183_183492

open Complex

theorem complex_fraction
  (a b : ℂ)
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 := 
by
  sorry

end complex_fraction_l183_183492


namespace smallest_area_6_8_l183_183201

noncomputable def smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) : ℝ :=
  let c1 := Real.sqrt (a^2 + b^2)
  let area1 := 0.5 * a * b
  let c2 := Real.sqrt (b^2 - a^2)
  let area2 := if h : b > a then 0.5 * a * c2 else 0.5 * b * c2
  min area1 area2

theorem smallest_area_6_8 : smallest_area_of_right_triangle 6 8 6 8 = 15.87 :=
sorry

end smallest_area_6_8_l183_183201


namespace compound_proposition_false_l183_183120

def p (x : ℝ) : Prop := e^x > 1 → x > 0
def q (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

theorem compound_proposition_false (h₁ : ∀ x : ℝ, p x) (h₂ : ¬ (∀ a b : ℝ, q a b)) : ¬ (∀ x a b : ℝ, p x ∧ q a b) :=
by
  sorry

end compound_proposition_false_l183_183120


namespace max_perpendicular_pairs_l183_183361

-- Definition of lines and sets of lines
def line := Type
def setA : set line := sorry  -- Set A containing 10 parallel lines
def setB : set line := sorry  -- Set B containing 10 parallel lines

-- Condition that each line in setA is perpendicular to every line in setB
def perpendicular (a b : line) : Prop := sorry
axiom every_setA_perpendicular_setB (a : line) (b : line) : a ∈ setA → b ∈ setB → perpendicular a b

-- Definition of perpendicular pairs
def perpendicular_pair (a b : line) : Prop := perpendicular a b

-- Proof statement for maximum number of perpendicular pairs
theorem max_perpendicular_pairs :
  ∃ (pairs : set (line × line)), 
  ∀ (a : line) (b : line), (a, b) ∈ pairs ↔ a ∈ setA ∧ b ∈ setB ∧ perpendicular_pair a b :=
sorry

end max_perpendicular_pairs_l183_183361


namespace polynomial_remainder_l183_183346

theorem polynomial_remainder (x : ℂ) :
  (x ^ 2030 + 1) % (x ^ 6 - x ^ 4 + x ^ 2 - 1) = x ^ 2 - 1 :=
by
  sorry

end polynomial_remainder_l183_183346


namespace complex_solution_l183_183071

theorem complex_solution (a : ℝ) (h : (a + complex.I) * (1 - a * complex.I) = 2) : a = 1 :=
by 
  sorry

end complex_solution_l183_183071


namespace collinear_condition_right_angled_triangle_values_l183_183051

-- Variables
variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variable (k : ℝ)
variable (BC AC : A)

-- Collinearity (Part 1)
theorem collinear_condition (hBC : BC = (2 - k, 3)) (hAC : AC = (2, 4)) :
  k = 1/2 :=
sorry

-- Right-Angled Triangle (Part 2)
theorem right_angled_triangle_values (hBC : BC = (2 - k, 3)) (hAC : AC = (2, 4)) :
  k = -2 ∨ k = -1 ∨ k = 3 ∨ k = 8 :=
sorry

end collinear_condition_right_angled_triangle_values_l183_183051


namespace least_positive_angle_is_75_l183_183709

noncomputable def least_positive_angle (θ : ℝ) : Prop :=
  cos (10 * Real.pi / 180) = sin (15 * Real.pi / 180) + sin θ

theorem least_positive_angle_is_75 :
  least_positive_angle (75 * Real.pi / 180) :=
by
  sorry

end least_positive_angle_is_75_l183_183709


namespace adam_first_year_students_l183_183615

theorem adam_first_year_students (X : ℕ) 
  (remaining_years_students : ℕ := 9 * 50)
  (total_students : ℕ := 490) 
  (total_years_students : X + remaining_years_students = total_students) : X = 40 :=
by { sorry }

end adam_first_year_students_l183_183615


namespace convex_polygon_enclosed_by_triangle_l183_183215

theorem convex_polygon_enclosed_by_triangle (U : set ℝ) (hU_convex : convex U) (hU_area : measure_theory.measure.area U = 1) :
  ∃ (T : set ℝ), convex T ∧ measure_theory.measure.area T ≤ 2 ∧ U ⊆ T :=
by
  sorry

end convex_polygon_enclosed_by_triangle_l183_183215


namespace incorrect_statement_B_l183_183573

theorem incorrect_statement_B :
  (∀ x: ℝ, 0 ≤ x → sqrt (0: ℝ) = 0) → -- Condition A
  (¬(sqrt ((-2)^2: ℝ) = 2)) →           -- Condition B is incorrect
  (∀ x: ℝ, 0 < x → sqrt x = (-sqrt x)) → -- Condition C
  (∀ x: ℝ, 0 < x → sqrt x > -x) →        -- Condition D
  ¬(sqrt ((-2)^2: ℝ) = 2) :=             -- The conclusion
begin
  intros hA hB hC hD,
  exact hB,
end

end incorrect_statement_B_l183_183573


namespace expression_undefined_at_12_l183_183364

theorem expression_undefined_at_12 :
  ¬ ∃ x : ℝ, x = 12 ∧ (x^2 - 24 * x + 144 = 0) →
  (∃ y : ℝ, y = (3 * x^3 + 5) / (x^2 - 24 * x + 144)) :=
by
  sorry

end expression_undefined_at_12_l183_183364


namespace homework_checked_on_friday_given_not_checked_until_thursday_l183_183561

open ProbabilityTheory

variables {Ω : Type} {P : ProbabilitySpace Ω}
variables (S : Event Ω) (A : Event Ω) (B : Event Ω)
variables [Fact (Probability S = 1 / 2)]
variables [Fact (Probability (Sᶜ ∩ B) = 1 / 10)]
variables [Fact (Probability A = 3 / 5)]
variables [Fact (A = S ∪ B)]
variables [Fact (Aᶜ = Sᶜ ∩ Aᶜ)]

theorem homework_checked_on_friday_given_not_checked_until_thursday :
  condProb B A = 1 / 6 := sorry

end homework_checked_on_friday_given_not_checked_until_thursday_l183_183561


namespace rent_percentage_l183_183580

variable (E : ℝ)

def rent_last_year (E : ℝ) : ℝ := 0.20 * E 
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 0.25 * (earnings_this_year E)

-- Prove that the rent this year is 143.75% of the rent last year
theorem rent_percentage : (rent_this_year E) = 1.4375 * (rent_last_year E) :=
by
  sorry

end rent_percentage_l183_183580


namespace calculate_expression_l183_183285

theorem calculate_expression :
  (-0.25) ^ 2014 * (-4) ^ 2015 = -4 :=
by
  sorry

end calculate_expression_l183_183285


namespace hyperbola_equation_l183_183037

open Real

noncomputable def hyperbola_foci := (0, sqrt 10), (0, -sqrt 10)

noncomputable def point_on_hyperbola (M : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, sqrt 10)
  let F₂ : ℝ × ℝ := (0, -sqrt 10)
  (F₁.fst - M.fst) * (F₂.fst - M.fst) + (F₁.snd - M.snd) * (F₂.snd - M.snd) = 0 ∧
  (sqrt ((F₁.fst - M.fst)^2 + (F₁.snd - M.snd)^2)) *
  (sqrt ((F₂.fst - M.fst)^2 + (F₂.snd - M.snd)^2)) = 2

theorem hyperbola_equation
  (M : ℝ × ℝ)
  (hM : point_on_hyperbola M) :
  ∃ a b : ℝ, (a = 3) ∧ (b^2 = 1) ∧ (y^2 / a^2 - x^2 / b^2 = 1) :=
sorry

end hyperbola_equation_l183_183037


namespace points_collinear_l183_183588

theorem points_collinear : 
  ∀ (A B C : ℝ×ℝ), A = (1, 2) ∧ B = (3, 5) ∧ C = (9, 14) → collinear {A, B, C} :=
by
  intro A B C h
  cases h with ha hbc
  cases hbc with hb hc
  simp only at ha hb hc
  sorry

end points_collinear_l183_183588


namespace polyhedron_vertex_assignment_l183_183690

theorem polyhedron_vertex_assignment (P : Polyhedron) 
    (edge_coloring : ∀ e : P.edges, Color) 
    (vertex_edges_three_colored : ∀ v : P.vertices, Card (set_of_colors v.edges) = 3) 
    : ∃ z : P.vertices -> ℂ, (∀ v : P.vertices, z v ≠ 1) ∧
      (∀ f : P.faces, (List.foldr (*) 1 (map (λ v => z v) f.vertices)) = 1) :=
sorry

end polyhedron_vertex_assignment_l183_183690


namespace pq_is_true_l183_183390

-- Define proposition p
def p : Prop := ∃ α : ℝ, Real.cos (π - α) = Real.cos α

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- The theorem to be proven
theorem pq_is_true : p ∧ q :=
by
  -- Provide proof for p and q (skipped here)
  sorry

end pq_is_true_l183_183390


namespace mixed_groups_count_l183_183974

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l183_183974


namespace symmetry_center_l183_183400

noncomputable def function_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

theorem symmetry_center (ω : ℝ) (φ : ℝ) (x : ℝ) :
  (0 < ω) ∧ (|φ| < π / 2) →
  function_period (λ x, Real.sin (ω * x + φ)) (4 * π) →
  ∀ x, Real.sin (ω * x + φ) ≤ Real.sin (ω * (π / 3) + φ) →
  ∃ k : ℤ, x = 2 * k * π - 2 * π / 3 :=
begin
  sorry
end

end symmetry_center_l183_183400


namespace function_properties_l183_183380

/-- Given a function f(2^x) = x^2 - 2 * a * x + 3, prove:
 1. The explicit expression for y = f(x) is (log2 x)^2 - 2 * a * log2 x + 3.
 2. If the function y = f(x) has a minimum value of -1 on the interval [1/2, 8], then a = -5/2 or a = 2.
-/
theorem function_properties (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (2^x) = x^2 - 2 * a * x + 3) ∧
  ( ∃ x : ℝ, 1 / 2 <= x ∧ x <= 8 ∧ f x = -1 )
  → (f = λ x, (Real.log2 x)^2 - 2 * a * (Real.log2 x) + 3) ∧ (a = -5/2 ∨ a = 2) :=
by
  sorry

end function_properties_l183_183380


namespace incenter_length_isosceles_right_triangle_length_l183_183542

theorem incenter_length_isosceles_right_triangle_length
  (A B C : Point)
  (h_triangle : is_isosceles_right_triangle A B C)
  (h_AC : dist A C = 4 * Real.sqrt 5)
  (h_angle : right_angle A):
  let I := incenter A B C,
      BI := dist B I
  in BI = (Real.sqrt 10 - Real.sqrt 5) * (Real.sqrt 2 + 1) := 
sorry

end incenter_length_isosceles_right_triangle_length_l183_183542


namespace real_number_condition_l183_183786

theorem real_number_condition 
  (x : ℝ) 
  (p : |x-1| ≥ 2) 
  (q : x ∈ ℤ) 
  (h1 : ¬((|x-1| ≥ 2) ∧ (x ∈ ℤ)))
  (h2 : ¬(¬(x ∈ ℤ))) : 
  x ∈ {0, 1, 2} := 
by 
  sorry

end real_number_condition_l183_183786


namespace arithmetic_sequence_sum_l183_183402

def a (n : ℕ) : ℤ := 2 + (n - 1) * 2 -- Definition of the nth term of the arithmetic sequence
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2 -- Sum of the first n terms
def H := (S n : ℕ) -- Convert to natural numbers if necessary

theorem arithmetic_sequence_sum : 
  let S (n : ℕ) := n^2 + n in 
  (∑ n in finset.range 100, 1 / S (n + 1)) = 100 / 101 := 
    sorry

end arithmetic_sequence_sum_l183_183402


namespace power_function_value_l183_183783

theorem power_function_value {α : ℝ} (h : 3^α = Real.sqrt 3) : (9 : ℝ)^α = 3 :=
by sorry

end power_function_value_l183_183783


namespace orthic_triangle_perimeter_l183_183897

variable {ABC : Triangle}
variable {A B C H D E F : Point}

/-- Prove the perimeter of the orthic triangle DEF is less than twice the length of any altitude in an acute-angled triangle ABC. -/
theorem orthic_triangle_perimeter (h1 : triangle.is_acute ABC)
  (h2 : triangle.orthocenter H ABC)
  (h3 : altitude_foot D A BC)
  (h4 : altitude_foot E B AC)
  (h5 : altitude_foot F C AB) :
  perimeter (triangle DEF) < 2 * altitudes.min_length ABC :=
sorry

end orthic_triangle_perimeter_l183_183897


namespace unoccupied_seats_l183_183321

theorem unoccupied_seats 
    (seats_per_row : ℕ) 
    (rows : ℕ) 
    (seatable_fraction : ℚ) 
    (total_seats := seats_per_row * rows) 
    (seatable_seats_per_row := (seatable_fraction * seats_per_row)) 
    (seatable_seats := seatable_seats_per_row * rows) 
    (unoccupied_seats := total_seats - seatable_seats) {
  seats_per_row = 8, 
  rows = 12, 
  seatable_fraction = 3/4 
  : unoccupied_seats = 24 :=
by
  sorry

end unoccupied_seats_l183_183321


namespace angle_B_triangle_perimeter_l183_183467

variable {A B C a b c : Real}

-- Definitions and conditions for part 1
def sides_relation (a b c : ℝ) (A : ℝ) : Prop :=
  2 * c = a + 2 * b * Real.cos A

-- Definitions and conditions for part 2
def triangle_area (a b c : ℝ) (B : ℝ) : Prop :=
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3

def side_b_value (b : ℝ) : Prop :=
  b = Real.sqrt 13

-- Theorem statement for part 1 
theorem angle_B (a b c A : ℝ) (h1: sides_relation a b c A) : B = Real.pi / 3 :=
sorry

-- Theorem statement for part 2 
theorem triangle_perimeter (a b c B : ℝ) (h1 : triangle_area a b c B) (h2 : side_b_value b) (h3 : B = Real.pi / 3) : a + b + c = 5 + Real.sqrt 13 :=
sorry

end angle_B_triangle_perimeter_l183_183467


namespace simplify_expression_eq_square_l183_183633

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183633


namespace constant_sum_of_squares_l183_183219

def isEquidistant (C D O : Point) : Prop :=
  dist C O = dist D O

theorem constant_sum_of_squares
  (O A B C D P : Point)
  (h_diameter : on_diameter A B O)
  (C_D_equidistant : isEquidistant C D O)
  (P_on_circumference : on_circumference P O)
  (circ_center : center O)
  (circle_points : points_on_circle O [A, B, C, D, P]) :
  dist P C ^ 2 + dist P D ^ 2 = k :=
sorry

end constant_sum_of_squares_l183_183219


namespace symmetric_lines_l183_183444

-- Define the lines l1 and l2 with their equations
def l1 : (ℝ → ℝ → Prop) := λ x y, x - 3 * y + 2 = 0
def l2 (m b : ℝ) : (ℝ → ℝ → Prop) := λ x y, m * x - y + b = 0

-- Define the symmetry condition with respect to the x-axis
def is_symmetric_wrt_x_axis (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y ↔ l2 x (-y)

-- State the theorem
theorem symmetric_lines (m b : ℝ)
  (h : is_symmetric_wrt_x_axis l1 (l2 m b)) : m + b = -1 :=
by
  sorry

end symmetric_lines_l183_183444


namespace cube_increasing_on_reals_l183_183010

theorem cube_increasing_on_reals (a b : ℝ) (h : a < b) : a^3 < b^3 :=
sorry

end cube_increasing_on_reals_l183_183010


namespace part1_part2_part3_l183_183780

def f (x a : ℝ) : ℝ := x^3 - 3 * a * x
def g (x a : ℝ) : ℝ := abs (f x a)
def F (a : ℝ) : ℝ :=
  if a ≤ 1/4 then 1 - 3 * a
  else if 1/4 < a ∧ a < 1 then 2 * a * (real.sqrt a)
  else 3 * a - 1

theorem part1 (h : real) : f 1 1 = -2 := by sorry
theorem part2 (a : ℝ) (h : ∀ m ∈ ℝ, ¬(∃ x, f x a = -x - m)) : a < 1/3 := by sorry
theorem part3 (a : ℝ) : (∀ x ∈ [-1, 1], g x a ≤ F a) ∧ (∃ x ∈ [-1, 1], g x a = F a) := by sorry

end part1_part2_part3_l183_183780


namespace correct_probability_statement_l183_183209

theorem correct_probability_statement :
  (∀ flips_heads : ℕ, flips_heads = 3 → ¬(probability 10 flips_heads) = 3 / 10) ∧
  (∀ tickets : ℕ, tickets = 1000 → ¬(probability 1000 1) = 1) ∧
  (∀ forecast : ℚ, forecast = 7 / 10 → ¬(forecast = 1)) ∧
  (probability 6 1 = 1 / 6) :=
by
  sorry

end correct_probability_statement_l183_183209


namespace expression_simplification_l183_183637

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l183_183637


namespace length_AB_l183_183868

-- Define the coordinates of points A and B
def A : (ℝ × ℝ × ℝ) := (2, -3, 5)
def B : (ℝ × ℝ × ℝ) := (2, -3, -5)

-- Define a function to calculate the Euclidean distance between two points in ℝ³
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)

-- Prove that the distance between A and B is 10
theorem length_AB : distance A B = 10 := by
  dsimp [distance, A, B]
  rw [← Real.sqrt_sq (by norm_num : 0 ≤ 10)]
  norm_num
  sorry

end length_AB_l183_183868


namespace rowing_tide_time_saved_l183_183578

theorem rowing_tide_time_saved :
  (∀ (d₁ : ℝ) (t₁ : ℝ) (d₂ : ℝ) (t₂ : ℝ),
  ((d₁ = 5) ∧ (t₁ = 1) ∧ (d₂ = 40) ∧ (t₂ = 10) ∧ t₁ ≠ 0 ∧ t₂ ≠ 0) →
  let speed_with_tide := d₁ / t₁ in
  let speed_against_tide := d₂ / t₂ in
  let time_with_tide := d₂ / speed_with_tide in
  let time_saved := t₂ - time_with_tide in
  (time_saved = 2))
  :=
sorry -- Proof omitted

end rowing_tide_time_saved_l183_183578


namespace range_of_m_l183_183391

theorem range_of_m (a b m : ℝ) (h₀ : a > 0) (h₁ : b > 1) (h₂ : a + b = 2) (h₃ : ∀ m, (4/a + 1/(b-1)) > m^2 + 8*m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l183_183391


namespace ratio_AC_AP_l183_183083

variable {ℝ : Type*} [linear_ordered_field ℝ]

structure Parallelogram (A B C D M N P : ℝ × ℝ) :=
  (parallelogram : line_parallel (A, B) (C, D) ∧ line_parallel (A, D) (B, C))
  (AM_AB_ratio : dist A M / dist A B = 17 / 1000)
  (AN_AD_ratio : dist A N / dist A D = 17 / 2009)
  (P_intersection : point_on_line P (A, C) ∧ point_on_line P (M, N))

theorem ratio_AC_AP (A B C D M N P : ℝ × ℝ) 
  [Parallelogram A B C D M N P] :
  dist A C / dist A P = 175 :=
sorry

end ratio_AC_AP_l183_183083


namespace simplify_expression_eq_square_l183_183634

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183634


namespace white_surface_area_fraction_l183_183299

theorem white_surface_area_fraction
    (total_cubes : ℕ)
    (white_cubes : ℕ)
    (red_cubes : ℕ)
    (edge_length : ℕ)
    (white_exposed_area : ℕ)
    (total_surface_area : ℕ)
    (fraction : ℚ)
    (h1 : total_cubes = 64)
    (h2 : white_cubes = 14)
    (h3 : red_cubes = 50)
    (h4 : edge_length = 4)
    (h5 : white_exposed_area = 6)
    (h6 : total_surface_area = 96)
    (h7 : fraction = 1 / 16)
    (h8 : white_cubes + red_cubes = total_cubes)
    (h9 : 6 * (edge_length * edge_length) = total_surface_area)
    (h10 : white_exposed_area / total_surface_area = fraction) :
    fraction = 1 / 16 := by
    sorry

end white_surface_area_fraction_l183_183299


namespace probability_of_karnataka_student_l183_183512

-- Defining the conditions

-- Number of students from each region
def total_students : ℕ := 10
def maharashtra_students : ℕ := 4
def karnataka_students : ℕ := 3
def goa_students : ℕ := 3

-- Number of students to be selected
def students_to_select : ℕ := 4

-- Total ways to choose 4 students out of 10
def C_total : ℕ := Nat.choose total_students students_to_select

-- Ways to select 4 students from the 7 students not from Karnataka
def non_karnataka_students : ℕ := maharashtra_students + goa_students
def C_non_karnataka : ℕ := Nat.choose non_karnataka_students students_to_select

-- Probability calculations
def P_no_karnataka : ℚ := C_non_karnataka / C_total
def P_at_least_one_karnataka : ℚ := 1 - P_no_karnataka

-- The statement to be proved
theorem probability_of_karnataka_student :
  P_at_least_one_karnataka = 5 / 6 :=
sorry

end probability_of_karnataka_student_l183_183512


namespace unoccupied_seats_l183_183322

theorem unoccupied_seats 
    (seats_per_row : ℕ) 
    (rows : ℕ) 
    (seatable_fraction : ℚ) 
    (total_seats := seats_per_row * rows) 
    (seatable_seats_per_row := (seatable_fraction * seats_per_row)) 
    (seatable_seats := seatable_seats_per_row * rows) 
    (unoccupied_seats := total_seats - seatable_seats) {
  seats_per_row = 8, 
  rows = 12, 
  seatable_fraction = 3/4 
  : unoccupied_seats = 24 :=
by
  sorry

end unoccupied_seats_l183_183322


namespace simplify_expression_l183_183642

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183642


namespace value_of_power_log_base_l183_183544

theorem value_of_power_log_base (b x : ℝ) (hx: x > 0) (hb : b > 0) (hb_not_one : b ≠ 1) 
    (log_eq : logb b x = log b x) : b ^ logb b x = x :=
by
  rw [log_eq]
  sorry

example : 5 ^ log 5 3 = 3 :=
by
  have log_seq := @value_of_power_log_base 5 3
  apply log_seq
  apply sorry -- proof of conditions

end value_of_power_log_base_l183_183544


namespace total_combined_grapes_l183_183263

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l183_183263


namespace quadratic_rewriting_l183_183478

theorem quadratic_rewriting (b n : ℝ) (h₁ : 0 < n)
  (h₂ : ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) :
  b = 4 * Real.sqrt 13 :=
by
  sorry

end quadratic_rewriting_l183_183478


namespace triangle_angle_conditions_l183_183556

theorem triangle_angle_conditions
  (A B C A1 B1 B2 C2 C3 A3 : Point)
  (h_triangle : IsTriangle A B C)
  (h_square_AB : IsSquare A B B1 A1)
  (h_square_BC : IsSquare B C C2 B2)
  (h_square_CA : IsSquare C A A3 C3)
  (h_circle_1 : OnCircle A1 B1 B2 C2 C3 A3) :
  (∠ A B C = 60 ∧ ∠ B C A = 60 ∧ ∠ C A B = 60) ∨
  (∠ A B C = 45 ∧ ∠ B C A = 90 ∧ ∠ C A B = 45) ∨
  (∠ A B C = 90 ∧ ∠ B C A = 45 ∧ ∠ C A B = 45) :=
sorry

end triangle_angle_conditions_l183_183556


namespace relationship_between_zeros_l183_183768

theorem relationship_between_zeros (x1 x2 : ℝ) 
  (h1 : 0 < x1 ∧ x1 < 1)
  (h2 : 1 < x2)
  (zero_of_f1 : e^(-x1) = -real.log x1)
  (zero_of_f2 : e^(-x2) = real.log x2) :
  0 < x1 * x2 ∧ x1 * x2 < 1 :=
sorry

end relationship_between_zeros_l183_183768


namespace g_value_at_5_l183_183947

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_5 (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x ^ 2) : g 5 = 1 := 
by 
  sorry

end g_value_at_5_l183_183947


namespace max_OM_ON_l183_183469

theorem max_OM_ON (a b : ℝ) : 
    (∃ A B C M N O : ℝ³, 
        let |BC| := a,
            let |AC| := b,
            midpoint M BC,
            midpoint N AC,
            center_square O AB,
            |OM + ON| ≤ (sqrt 2 + 1) / 2 * (a + b)
        ) :=
sorry

end max_OM_ON_l183_183469


namespace possible_values_l183_183004

noncomputable def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

theorem possible_values (x y z : ℝ) 
  (h : matrix x y z.det = 0) :
  ∃ v, v = (x / (y + z) + y / (x + z) + z / (x + y)) ∧ (v = -3 ∨ v = 3 / 2) := 
sorry

end possible_values_l183_183004


namespace addition_problem_base6_l183_183349

theorem addition_problem_base6 (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 :=
by
  sorry

end addition_problem_base6_l183_183349


namespace sequence_takes_every_positive_integer_value_once_l183_183484

def y : ℕ → ℕ
| 1           := 1
| (2*k)       := if even k then 2 * y k else 2 * y k + 1
| (2*k + 1) := if even k then 2 * y k + 1 else 2 * y k

theorem sequence_takes_every_positive_integer_value_once :
  ∀ n : ℕ, n > 0 → y n = n := 
by
  sorry

end sequence_takes_every_positive_integer_value_once_l183_183484


namespace exists_colored_triangle_l183_183501

structure Point := (x : ℝ) (y : ℝ)
inductive Color
| red
| blue

def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)
  
def same_color_triangle_exists (S : Finset Point) (color : Point → Color) : Prop :=
  ∃ (A B C : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧
                    (color A = color B ∧ color B = color C) ∧
                    ¬ collinear A B C ∧
                    (∃ (X Y Z : Point), 
                      ((X ∈ S ∧ color X ≠ color A ∧ (X ≠ A ∧ X ≠ B ∧ X ≠ C)) ∧ 
                       (Y ∈ S ∧ color Y ≠ color A ∧ (Y ≠ A ∧ Y ≠ B ∧ Y ≠ C)) ∧
                       (Z ∈ S ∧ color Z ≠ color A ∧ (Z ≠ A ∧ Z ≠ B ∧ Z ≠ C)) → 
                       False))

theorem exists_colored_triangle 
  (S : Finset Point) (h1 : 5 ≤ S.card) (color : Point → Color) 
  (h2 : ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → (color A = color B ∧ color B = color C) → ¬ collinear A B C) 
  : same_color_triangle_exists S color :=
sorry

end exists_colored_triangle_l183_183501


namespace train_speed_including_stoppages_l183_183695

theorem train_speed_including_stoppages (s : ℝ) (t : ℝ) (running_time_fraction : ℝ) :
  s = 48 ∧ t = 1/4 ∧ running_time_fraction = (1 - t) → (s * running_time_fraction = 36) :=
by
  sorry

end train_speed_including_stoppages_l183_183695


namespace sum_sqrt_bounds_l183_183140

theorem sum_sqrt_bounds (n : ℕ) (h : 0 < n) : 
  (2 / 3 * n * Real.sqrt n) < 
  (∑ k in Finset.range n.succ, Real.sqrt k) ∧ 
  (∑ k in Finset.range n.succ, Real.sqrt k) < 
  ((4 * n + 3) / 6 * Real.sqrt n) :=
by
  sorry

end sum_sqrt_bounds_l183_183140


namespace triangle_inequality_l183_183502

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 :=
sorry

end triangle_inequality_l183_183502


namespace simplify_expression_l183_183651

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183651


namespace simplify_expression_eq_square_l183_183631

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183631


namespace hyperbola_transformation_l183_183191

def equation_transform (x y : ℝ) : Prop :=
  y = (1 - 3 * x) / (2 * x - 1)

def coordinate_shift (x y X Y : ℝ) : Prop :=
  X = x - 0.5 ∧ Y = y + 1.5

theorem hyperbola_transformation (x y X Y : ℝ) :
  equation_transform x y →
  coordinate_shift x y X Y →
  (X * Y = -0.25) :=
by
  sorry

end hyperbola_transformation_l183_183191


namespace magnitude_vector_minus_double_b_l183_183764

variables (a b : V) (angle : Real) (norm_a norm_b : ℝ)
variables [InnerProductSpace ℝ V]

def vector_magnitude (v : V) : ℝ := Real.sqrt (inner v v)

theorem magnitude_vector_minus_double_b 
  (h_angle : angle = 150 * Real.pi / 180)
  (h_mag_a : vector_magnitude a = 2)
  (h_mag_b : vector_magnitude b = Real.sqrt 3)
  (h_a_dot_b : inner a b = 2 * Real.sqrt 3 * (Real.cos (150 * Real.pi / 180))) :
  vector_magnitude (a - 2 • b) = 2 :=
by
  sorry

end magnitude_vector_minus_double_b_l183_183764


namespace evaluate_expression_l183_183691

theorem evaluate_expression :
  (27 * 16) ^ (-3)⁻¹ = 1 / (3 * 2 ^ (4 / 3)) :=
by
  -- Conditions:
  let a := 27 
  let b := 16 
  have h1 : a = 3^3 := by sorry
  have h2 : b = 2^4 := by sorry
  -- Definition of the Exponent
  let exp := -3⁻¹
  have exp_simplified : exp = -1/3 := by sorry
  -- Proof
  sorry

end evaluate_expression_l183_183691


namespace last_number_on_blackboard_l183_183213

-- Define the initial sequence of numbers on the blackboard
def initial_sequence : List ℚ :=
  List.range (2 * 1008 + 1).map (λ n : ℕ, (n - 1008) / 1008)

-- Define the operation that takes two numbers and writes their transformation back
def operation (x y : ℚ) : ℚ :=
  x + 7 * x * y + y

-- Lean theorem statement for the problem
theorem last_number_on_blackboard : 
  let final_number := 
    sorry -- Placeholder for the sequence of 2016 operations.
  in final_number = -144 / 1008 :=
sorry

end last_number_on_blackboard_l183_183213


namespace distance_to_asymptote_l183_183926

noncomputable def distance_from_asymptote : ℝ :=
  let x0 := 3
  let y0 := 0
  let A := 3
  let B := -4
  let C := 0
  @Real.sqrt ((A^2) + (B^2))^{-1} * abs (A * x0 + B * y0 + C)

theorem distance_to_asymptote : distance_from_asymptote = (9 / 5) := sorry

end distance_to_asymptote_l183_183926


namespace complement_intersect_A_B_range_of_a_l183_183754

-- Definitions for sets A and B
def setA : Set ℝ := {x | -2 < x ∧ x < 0}
def setB : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- First statement to prove
theorem complement_intersect_A_B : (setAᶜ ∩ setB) = {x | x ≥ 0} :=
  sorry

-- Definition for set C
def setC (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

-- Second statement to prove
theorem range_of_a (a : ℝ) : (setC a ⊆ setA) ↔ (a ≤ -1) ∨ (-1 ≤ a ∧ a ≤ -1 / 2) :=
  sorry

end complement_intersect_A_B_range_of_a_l183_183754


namespace sin_R_of_right_triangle_l183_183457

theorem sin_R_of_right_triangle (P Q R : ℝ) 
  (h1 : P + Q + R = π / 2)
  (h2 : sin P = 3 / 5)
  (h3 : sin Q = 1) :
  sin R = 4 / 5 :=
begin
  sorry
end

end sin_R_of_right_triangle_l183_183457


namespace limit_log_sine_eq_one_div_e_l183_183624

open Real

theorem limit_log_sine_eq_one_div_e :
  tendsto (λ x : ℝ, (1 - log (1 + x^(1/3)))^(x/(sin (x^(1/3)))^4)) (𝓝 0) (𝓝 (exp (-1))) := 
  sorry

end limit_log_sine_eq_one_div_e_l183_183624


namespace midpoint_calculation_l183_183872

theorem midpoint_calculation :
  let A := (30, 10) : ℝ × ℝ
  let B := (6, 3) : ℝ × ℝ
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (2 * C.1 - 4 * C.2) = 10 :=
by
  let A := (30, 10) : ℝ × ℝ
  let B := (6, 3) : ℝ × ℝ
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  have hC1 : C.1 = 18 := by
    calc
      C.1 = (A.1 + B.1) / 2 : rfl
      ... = (30 + 6) / 2 : rfl
      ... = 36 / 2 : rfl
      ... = 18 : rfl
  have hC2 : C.2 = 6.5 := by
    calc
      C.2 = (A.2 + B.2) / 2 : rfl
      ... = (10 + 3) / 2 : rfl
      ... = 13 / 2 : rfl
      ... = 6.5 : rfl
  calc
    2 * C.1 - 4 * C.2 = 2 * 18 - 4 * 6.5 : by rw [hC1, hC2]
    ... = 36 - 26 : rfl
    ... = 10 : rfl
  sorry

end midpoint_calculation_l183_183872


namespace A_squared_finite_possibilities_l183_183108

open Matrix

-- Defining a 3x3 real matrix
variable (A : Matrix (Fin 3) (Fin 3) ℝ)

/-- Given that A⁴ = 0, we need to prove that the number of different matrices that A² can be is finite. -/
theorem A_squared_finite_possibilities (hA : A ^ 4 = 0) :
  {B : Matrix (Fin 3) (Fin 3) ℝ // ∃ (C : Matrix (Fin 3) (Fin 3) ℝ), A = C ∧ B = C ^ 2}.finite :=
sorry

end A_squared_finite_possibilities_l183_183108


namespace possible_values_of_a_l183_183043

noncomputable def f (x a : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 2 * a * x + 2 else x + 9 / x - 3 * a

theorem possible_values_of_a (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ 1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end possible_values_of_a_l183_183043


namespace block_path_length_l183_183670

theorem block_path_length
  (length width height : ℝ) 
  (dot_distance : ℝ) 
  (rolls_to_return : ℕ) 
  (π : ℝ) 
  (k : ℝ)
  (H1 : length = 2) 
  (H2 : width = 1) 
  (H3 : height = 1)
  (H4 : dot_distance = 1)
  (H5 : rolls_to_return = 2) 
  (H6 : k = 4) 
  : (2 * rolls_to_return * length * π = k * π) :=
by sorry

end block_path_length_l183_183670


namespace graph_shift_l183_183190

/- 
  Given the functions f(x) = 2^{x+2} and g(x) = 2^{x-1}, 
  prove that to obtain the graph of f(x) it is only necessary 
  to shift the graph of g(x) to the left by 3 units.
-/

def f (x : ℝ) : ℝ := 2^(x + 2)
def g (x : ℝ) : ℝ := 2^(x - 1)

theorem graph_shift :
  ∃ shift : ℝ, ∀ x : ℝ, f(x) = g(x + shift) ∧ shift = -3 :=
by
  sorry

end graph_shift_l183_183190


namespace simplify_expression_eq_square_l183_183630

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l183_183630


namespace numbers_from_1_to_39_usable_by_five_threes_l183_183194

def uses_five_threes (expr : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), (∃ (three_uses: ℕ), expr three_uses = n) → (three_uses = 5)

theorem numbers_from_1_to_39_usable_by_five_threes :
  ∀ (n : ℕ), (n ≥ 1 ∧ n ≤ 39) → (∃ (expr: ℕ → ℕ), uses_five_threes expr ∧ expr 3 = n) :=
by
  intro n hn
  sorry

end numbers_from_1_to_39_usable_by_five_threes_l183_183194


namespace monotonicity_of_f_smallest_positive_a_two_zeros_derivative_positive_at_avg_l183_183871

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a-2)*x - a*Real.log x

/-- 
1. Prove the intervals of monotonicity:
    - If \( a \leq 0 \), then \( f(x) \) increases on \( (0, \infty) \).
    - If \( a > 0 \), then \( f(x) \) increases on \( \left( \frac{a}{2}, +\infty \right) \) and decreases on \( \left( 0, \frac{a}{2} \right) \).
--/
theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, 0 < x ∧ x < y → f a x < f a y) ∧
  (a > 0 → (∀ x : ℝ, 0 < x ∧ x < a / 2 → f a x > f a (x + 1)) ∧ 
    (∀ x : ℝ, x > a / 2 → f a x < f a (x + 1))) :=
sorry

/-- 
2. Prove that the smallest positive integer \( a \) such that \( f(x) \) has two zeros is \( 3 \).
--/
theorem smallest_positive_a_two_zeros :
  ∃ a : ℕ, 0 < a ∧ (∀ x : ℝ, f a x = 0 → ∃ y : ℝ, y ≠ x ∧ f a y = 0) ∧ 
    ∀ b : ℕ, 0 < b ∧ b < a → ∀ x : ℝ, ¬(f b x = 0 ∧ ∃ y : ℝ, y ≠ x ∧ f b y = 0) :=
sorry

/-- 
3. For \( a > 0 \), if the equation \( f(x) = c \) has two distinct real roots \( x_1 \) and \( x_2 \), then \( f'\left( \frac{x_1 + x_2}{2} \right) > 0 \).
--/
theorem derivative_positive_at_avg (a c : ℝ) (x1 x2 : ℝ) (h1 : x1 ≠ x2) (hx1 : f a x1 = c) (hx2 : f a x2 = c) :
  a > 0 → (f a)'.eval ((x1 + x2)/2) > 0 :=
sorry

end monotonicity_of_f_smallest_positive_a_two_zeros_derivative_positive_at_avg_l183_183871


namespace selected_40th_is_795_l183_183136

-- Definitions of constants based on the problem conditions
def total_participants : ℕ := 1000
def selections : ℕ := 50
def equal_spacing : ℕ := total_participants / selections
def first_selected_number : ℕ := 15
def nth_selected_number (n : ℕ) : ℕ := (n - 1) * equal_spacing + first_selected_number

-- The theorem to prove the 40th selected number is 795
theorem selected_40th_is_795 : nth_selected_number 40 = 795 := 
by 
  -- Skipping the detailed proof
  sorry

end selected_40th_is_795_l183_183136


namespace expression_undefined_at_12_l183_183365

theorem expression_undefined_at_12 :
  ¬ ∃ x : ℝ, x = 12 ∧ (x^2 - 24 * x + 144 = 0) →
  (∃ y : ℝ, y = (3 * x^3 + 5) / (x^2 - 24 * x + 144)) :=
by
  sorry

end expression_undefined_at_12_l183_183365


namespace john_uses_six_pounds_of_vegetables_l183_183476

-- Define the given conditions:
def pounds_of_beef_bought : ℕ := 4
def pounds_beef_used_in_soup := pounds_of_beef_bought - 1
def pounds_of_vegetables_used := 2 * pounds_beef_used_in_soup

-- Statement to prove:
theorem john_uses_six_pounds_of_vegetables : pounds_of_vegetables_used = 6 :=
by
  sorry

end john_uses_six_pounds_of_vegetables_l183_183476


namespace ordered_triples_count_l183_183424

theorem ordered_triples_count :
  { (a, b, c) : ℤ × ℤ × ℤ // |a - b| + c = 23 ∧ a^2 - b * c = 119 }.toFinset.card = 1 :=
by
  sorry

end ordered_triples_count_l183_183424


namespace transformed_mean_l183_183034

variables {n : ℕ} (x : Fin n → ℝ)

def mean (x : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i, x i)) / n

theorem transformed_mean {x : Fin n → ℝ} (h : mean x = 5) :
  mean (λ i, 2 * x i + 1) = 11 :=
sorry

end transformed_mean_l183_183034


namespace least_positive_angle_l183_183707

theorem least_positive_angle (θ : ℝ) (h : θ > 0 ∧ θ ≤ 360) : 
  (cos 10 = sin 15 + sin θ) → θ = 32.5 :=
by 
  sorry

end least_positive_angle_l183_183707


namespace problem_l183_183894

noncomputable def log10 : ℝ → ℝ := λ x, Real.log x / Real.log 10

theorem problem
  (x y z : ℝ)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_z : z > 0)
  (h1 : x * y * z = 10^81)
  (h2 : log10 x * log10 (y * z) + log10 y * log10 z = 468) :
  Real.sqrt ((log10 x)^2 + (log10 y)^2 + (log10 z)^2) = 75 :=
sorry

end problem_l183_183894


namespace marco_dad_strawberries_weight_l183_183503

theorem marco_dad_strawberries_weight (marco_weight total_weight dad_weight : ℕ)
    (h1 : marco_weight = 15)
    (h2 : total_weight = 37)
    (h3 : total_weight = marco_weight + dad_weight) : 
    dad_weight = 22 :=
by {
  rw [←h2, h1] at h3,
  sorry
}

end marco_dad_strawberries_weight_l183_183503


namespace mixed_groups_count_l183_183964

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l183_183964


namespace nat_pairs_exp_eq_l183_183675

theorem nat_pairs_exp_eq (a b : ℕ) : a^b = b^a ↔ (a = b) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) := 
by
  sorry

end nat_pairs_exp_eq_l183_183675


namespace find_k_l183_183539

-- Conditions:
-- The roots are in the ratio 3:2 and are based on Vieta's formula for sum and product of roots.

noncomputable def roots_ratio (a : ℝ) : Prop :=
  let sum_roots := 3 * a + 2 * a
  let product_roots := 3 * a * 2 * a
  sum_roots = -10 ∧ product_roots = k

theorem find_k (k : ℝ) (h : ∃ a : ℝ, roots_ratio a) : k = 24 :=
by
  sorry

end find_k_l183_183539


namespace complex_solution_l183_183769

theorem complex_solution (x y : ℝ) (z : ℂ) (h1 : z = x + y * complex.I) (h2 : z = complex.inv complex.I) : x + y = -1 :=
by
  sorry

end complex_solution_l183_183769


namespace problem_solution_l183_183684

theorem problem_solution :
  (∀ (a b : Point), ¬ (segment a b = distance a b))
  ∧ (¬ ∀ (α β : Angle), corresponding α β → α = β)
  ∧ (¬ ∀ (α β : Angle), complementary α β → adjacent_complementary α β)
  ∧ (∀ (P : Point) (L : Line), ∃! (PL : Perpendicular_Line), passes_through PL P ∧ perpendicular PL L)
  ∧ (¬ ∀ (d c : Diameter_Chord), bisects d c → ⟂ d c)
  ∧ (∀ (l c : Line_Chord), perpendicular_bisects l c → passes_center l)
  ∧ (∀ (d c : Diameter_Chord), perpendicular d c → bisects_arc d c)
  ∧ (∀ (d a : Diameter_Arc), bisects_arc d a → ⟂ d (chord_corresponding a) ∧ bisects_chord d (chord_corresponding a)) :=
begin
  sorry
end

end problem_solution_l183_183684


namespace ratio_initial_to_doubled_l183_183605

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := 
by
  sorry

end ratio_initial_to_doubled_l183_183605


namespace simplify_expression_l183_183655

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183655


namespace simplify_expression_l183_183652

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183652


namespace simplify_complex_fraction_l183_183908

theorem simplify_complex_fraction :
  (5 + 7 * Complex.i) / (2 + 3 * Complex.i) = (31 / 13) - (1 / 13) * Complex.i :=
by
  sorry

end simplify_complex_fraction_l183_183908


namespace least_positive_angle_l183_183708

theorem least_positive_angle (θ : ℝ) (h : θ > 0 ∧ θ ≤ 360) : 
  (cos 10 = sin 15 + sin θ) → θ = 32.5 :=
by 
  sorry

end least_positive_angle_l183_183708


namespace either_p_or_q_false_suff_not_p_true_l183_183497

theorem either_p_or_q_false_suff_not_p_true (p q : Prop) : (p ∨ q = false) → (¬p = true) :=
by
  sorry

end either_p_or_q_false_suff_not_p_true_l183_183497


namespace eccentricity_minimization_l183_183738

theorem eccentricity_minimization (a b x y : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola_C : (x^2 / a^2) - (y^2 / b^2) = 1)
  (k1 k2 : ℝ) (h_kk : k1 * k2 = b^2 / a^2)
  (h_minimize : ∀ t : ℝ, differentiable_at ℝ (λ t, (2 / t) + real.log t) t → ((2 / t) + real.log t).deriv t = 0 → t = 2)
  : (sqrt (1 + b^2 / a^2) = sqrt 3) :=
begin
  sorry -- Proof goes here
end

end eccentricity_minimization_l183_183738


namespace least_positive_angle_l183_183714

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end least_positive_angle_l183_183714


namespace jayden_total_rounded_l183_183096

noncomputable def round_to_nearest_ten_cents (x : ℝ) : ℝ :=
  (Real.floor (10 * x + 0.5)) / 10

def total_rounded_amount (a1 a2 a3 a4 : ℝ) : ℝ :=
  round_to_nearest_ten_cents a1 +
  round_to_nearest_ten_cents a2 +
  round_to_nearest_ten_cents a3 +
  round_to_nearest_ten_cents a4

theorem jayden_total_rounded
  (a1 a2 a3 a4 : ℝ)
  (h1 : a1 = 2.93)
  (h2 : a2 = 7.58)
  (h3 : a3 = 12.49)
  (h4 : a4 = 15.65) :
  total_rounded_amount a1 a2 a3 a4 = 38.70 :=
by
  rw [h1, h2, h3, h4]
  sorry

end jayden_total_rounded_l183_183096


namespace simplify_expression_l183_183646

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l183_183646


namespace twelfth_even_multiple_of_5_l183_183567

theorem twelfth_even_multiple_of_5 : 
  ∃ n : ℕ, n > 0 ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ ∀ m, (m > 0 ∧ (m % 2 = 0) ∧ (m % 5 = 0) ∧ m < n) → (m = 10 * (fin (n / 10) - 1)) := 
sorry

end twelfth_even_multiple_of_5_l183_183567


namespace total_grapes_l183_183266

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l183_183266


namespace largest_corner_sum_l183_183946

noncomputable def sum_faces (cube : ℕ → ℕ) : Prop :=
  cube 1 + cube 7 = 8 ∧ 
  cube 2 + cube 6 = 8 ∧ 
  cube 3 + cube 5 = 8 ∧ 
  cube 4 + cube 4 = 8

theorem largest_corner_sum (cube : ℕ → ℕ) 
  (h : sum_faces cube) : 
  ∃ n, n = 17 ∧ 
  ∀ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
            (cube a = 7 ∧ cube b = 6 ∧ cube c = 4 ∨ 
             cube a = 6 ∧ cube b = 4 ∧ cube c = 7 ∨ 
             cube a = 4 ∧ cube b = 7 ∧ cube c = 6)) → 
            a + b + c = n := sorry

end largest_corner_sum_l183_183946


namespace largest_k_divides_2010_factorial_l183_183666

theorem largest_k_divides_2010_factorial :
  ∀ (k : ℕ), (↑2010 : ℕ) = 2 * 3 * 5 * 67 → 
    (∀ k', k' > k → ¬(2010 ^ k' ∣ (nat.factorial 2010))) →
    (2010 ^ k ∣ (nat.factorial 2010)) → 
    k = 30 := 
by
  intros k fact cond1 cond2
  sorry

end largest_k_divides_2010_factorial_l183_183666


namespace radian_measure_15_degrees_l183_183195

theorem radian_measure_15_degrees : (15 * (Real.pi / 180)) = (Real.pi / 12) :=
by
  sorry

end radian_measure_15_degrees_l183_183195


namespace smallest_k_l183_183957

-- Define the non-decreasing property of digits in a five-digit number
def non_decreasing (n : Fin 5 → ℕ) : Prop :=
  n 0 ≤ n 1 ∧ n 1 ≤ n 2 ∧ n 2 ≤ n 3 ∧ n 3 ≤ n 4

-- Define the overlap property in at least one digit
def overlap (n1 n2 : Fin 5 → ℕ) : Prop :=
  ∃ i : Fin 5, n1 i = n2 i

-- The main theorem stating the problem
theorem smallest_k {N1 Nk : Fin 5 → ℕ} :
  (∀ n : Fin 5 → ℕ, non_decreasing n → overlap N1 n ∨ overlap Nk n) → 
  ∃ (k : Nat), k = 2 :=
sorry

end smallest_k_l183_183957


namespace simplify_expression_l183_183650

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l183_183650


namespace rectangle_area_l183_183950

-- Definitions of the conditions
variables (Length Width Area : ℕ)
variable (h1 : Length = 4 * Width)
variable (h2 : Length = 20)

-- Statement to prove
theorem rectangle_area : Area = Length * Width → Area = 100 :=
by
  sorry

end rectangle_area_l183_183950


namespace grapes_total_sum_l183_183262

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l183_183262


namespace sqrt_condition_l183_183811

theorem sqrt_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) → x ≥ 2 :=
by
  sorry

end sqrt_condition_l183_183811


namespace fish_remaining_l183_183506

def initial_fish : ℝ := 47.0
def given_away_fish : ℝ := 22.5

theorem fish_remaining : initial_fish - given_away_fish = 24.5 :=
by
  sorry

end fish_remaining_l183_183506


namespace shift_graph_right_pi_over_3_eq_2sinx_l183_183307

def matrix_det (a1 a2 a3 a4 : ℝ) : ℝ := a1 * a4 - a2 * a3

def f (x : ℝ) : ℝ := 
  let a1 := sin (π - x)
  let a2 := sqrt 3
  let a3 := cos (π + x)
  let a4 := 1
  sin (π - x) - sqrt 3 * cos (π + x)

theorem shift_graph_right_pi_over_3_eq_2sinx :
  ∀ x : ℝ, f (x + π / 3) = 2 * sin x :=
by
  sorry

end shift_graph_right_pi_over_3_eq_2sinx_l183_183307


namespace mixed_groups_count_l183_183993

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l183_183993


namespace num_of_trucks_l183_183077

-- Define constants for given conditions
def service_cost_per_vehicle : ℝ := 2.2
def cost_per_liter : ℝ := 0.7
def number_of_minivans : ℕ := 3
def minivan_tank_size : ℝ := 65.0
def truck_tank_multiplier : ℝ := 2.2
def total_cost : ℝ := 347.7

-- Define the function to calculate the total cost of filling up a vehicle
def total_cost_per_vehicle (tank_size : ℝ) (service_cost : ℝ) (fuel_cost_per_liter : ℝ) : ℝ :=
  (tank_size * fuel_cost_per_liter) + service_cost

-- Calculate the total cost for the mini-vans
def total_cost_minivans : ℝ :=
  total_cost_per_vehicle minivan_tank_size service_cost_per_vehicle cost_per_liter * number_of_minivans

-- Calculate the tank size for one truck
def truck_tank_size : ℝ := minivan_tank_size * truck_tank_multiplier

-- Calculate the total cost for one truck
def total_cost_per_truck : ℝ :=
  total_cost_per_vehicle truck_t_tzcarank_s_rize service_cost_per_vehicle cost_per_liter

-- Given conditions and total cost, prove the number of trucks filled up is 2
theorem num_of_trucks 
  (total_cost_minivans = total_cost_per_vehicle minivan_tank_size service_cost_per_vehicle cost_per_liter * number_of_minivans)
  (total_cost_trucks = total_cost - total_cost_minivans)
  (total_cost_per_truck = total_cost_per_vehicle truck_tank_size service_cost_per_vehicle cost_per_liter) :
  total_cost_trucks / total_cost_per_truck = 2 := sorry

end num_of_trucks_l183_183077


namespace cube_side_length_l183_183086

theorem cube_side_length (V : ℝ) (hV : V = 8) : ∃ x : ℝ, x^3 = V ∧ x = 2 := 
by
  use 2
  split
  · rw [hV]
    norm_num
  sorry

end cube_side_length_l183_183086


namespace distance_from_point_to_hyperbola_asymptote_l183_183929

noncomputable def distance_to_asymptote (x1 y1 a b : ℝ) : ℝ :=
  abs (a * x1 + b * y1) / real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_hyperbola_asymptote :
  distance_to_asymptote 3 0 3 (-4) = 9 / 5 :=
by
  sorry

end distance_from_point_to_hyperbola_asymptote_l183_183929


namespace simplify_expression_l183_183914

theorem simplify_expression :
  (∀ (x : ℝ), x = 4^6 → (sqrt (real.cbrt (sqrt x))) = 1/2) :=
by
  intro x h4096
  sorry

end simplify_expression_l183_183914


namespace surface_area_eq_8pi_l183_183025

-- Define the equilateral triangle with side length sqrt(3)
def equilateral_triangle (a : ℝ) (h_a : a = real.sqrt 3) : Prop :=
  ∃ (A B C : ℝ × ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  dist A B = a ∧ dist B C = a ∧ dist A C = a

-- Define the surface area of the circumscribed sphere of the tetrahedron
def surface_area_of_circumscribed_sphere (T : Type) (a : ℝ) (h_a : a = real.sqrt 3) : Prop :=
  ∃ (S : ℝ), S = 8 * real.pi

theorem surface_area_eq_8pi (a : ℝ) (h_a : a = real.sqrt 3) :
  ∃ (S : ℝ), S = 8 * real.pi :=
sorry

end surface_area_eq_8pi_l183_183025
