import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Sum
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Interval
import Mathlib.Geometry.Euclidean.Triangle.Basic
import Mathlib.Integration
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Set
import Mathlib.Tactic
import Mathlib.Topology.Instances.Real
import Real

namespace circle_center_and_tangency_l592_592382

theorem circle_center_and_tangency (m : ℝ) :
  let C1 := { center := (-1 : ℝ, -1), radius := 1 }
  let C2 := { center := (2 : ℝ, 3), radius := real.sqrt (13 - m) }
  (∀ x y : ℝ, x^2 + y^2 + 2*x + 2*y + 1 = 0 → (x, y) = (-1, -1)) ∧
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 6*y + m = 0 → sorry) → 
  abs (dist C1.center C2.center - (C1.radius + C2.radius)) < 1e-6 →
  m = -3 :=
sorry

end circle_center_and_tangency_l592_592382


namespace perimeters_of_rectangles_l592_592599

theorem perimeters_of_rectangles (a b : ℕ) (h1 : 2 * (a + b) = 20) (h2 : 2 * (a + 4 * b) = 56) : 
  2 * (2 * a + 2 * b) = 40 ∧ 2 * (4 * a + b) = 44 :=
by
  -- given conditions and necessary values
  have hab : a + b = 10, from Nat.eq_of_mul_eq_mul_left (by norm_num) h1,
  have hab2 : a + 4 * b = 28, from Nat.eq_of_mul_eq_mul_left (by norm_num) h2,
  sorry  -- skipping the proof for now

end perimeters_of_rectangles_l592_592599


namespace spider_leg_pressure_l592_592492

/--
A giant spider is discovered. It weighs 2.5 times the previous largest spider, 
which weighed 6.4 ounces. Each of its legs has a cross-sectional area of 0.5 
square inches. How much pressure in ounces per square inch does each leg undergo?
-/
theorem spider_leg_pressure
  (weight_previous_spider : ℝ := 6.4)
  (weight_ratio : ℝ := 2.5)
  (cross_sectional_area : ℝ := 0.5)
  (number_of_legs : ℕ := 8) :
  let weight_giant_spider := weight_ratio * weight_previous_spider in
  let weight_per_leg := weight_giant_spider / number_of_legs in
  let pressure_per_leg := weight_per_leg / cross_sectional_area in
  pressure_per_leg = 4 :=
by
  let weight_giant_spider := weight_ratio * weight_previous_spider
  let weight_per_leg := weight_giant_spider / number_of_legs
  let pressure_per_leg := weight_per_leg / cross_sectional_area
  sorry

end spider_leg_pressure_l592_592492


namespace correlation_1_and_3_l592_592778

-- Define the conditions as types
def relationship1 : Type := ∀ (age : ℕ) (fat_content : ℝ), Prop
def relationship2 : Type := ∀ (curve_point : ℝ × ℝ), Prop
def relationship3 : Type := ∀ (production : ℝ) (climate : ℝ), Prop
def relationship4 : Type := ∀ (student : ℕ) (student_ID : ℕ), Prop

-- Define what it means for two relationships to have a correlation
def has_correlation (rel1 rel2 : Type) : Prop := 
  -- Some formal definition of correlation suitable for the context
  sorry

-- Theorem stating that relationships (1) and (3) have a correlation
theorem correlation_1_and_3 :
  has_correlation relationship1 relationship3 :=
sorry

end correlation_1_and_3_l592_592778


namespace converse_and_inverse_not_true_l592_592648

def quadrilateral := Type
def rectangle : quadrilateral → Prop := sorry
def opposite_sides_equal : quadrilateral → Prop := sorry

theorem converse_and_inverse_not_true (q : quadrilateral) :
  (rectangle q → opposite_sides_equal q) →
  (¬(opposite_sides_equal q → rectangle q)) ∧ (¬(¬rectangle q → ¬opposite_sides_equal q)) :=
by
  intros h
  split
  { intro converse
    sorry },
  { intro inverse
    sorry }

end converse_and_inverse_not_true_l592_592648


namespace inner_cube_surface_area_eq_l592_592508

noncomputable def outer_cube_surface_area := 150
def surface_area_of_cube (s : ℝ) : ℝ := 6 * s^2

-- Given
theorem inner_cube_surface_area_eq (s₁ s₂ r l : ℝ) (h1: surface_area_of_cube s₁ = 150) 
  (h2: r = s₁ / 2) (h3: l * real.sqrt 3 = 2 * r) : surface_area_of_cube l = 50 := by
  sorry

end inner_cube_surface_area_eq_l592_592508


namespace triangle_acute_angle_l592_592802

theorem triangle_acute_angle 
  (a b c : ℝ) 
  (h1 : a^3 = b^3 + c^3)
  (h2 : a > b)
  (h3 : a > c)
  (h4 : b > 0) 
  (h5 : c > 0) 
  (h6 : a > 0) 
  : 
  (a^2 < b^2 + c^2) :=
sorry

end triangle_acute_angle_l592_592802


namespace product_of_first_three_terms_is_960_l592_592801

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l592_592801


namespace jackson_house_visits_l592_592292

theorem jackson_house_visits
  (days_per_week : ℕ)
  (total_goal : ℕ)
  (monday_earnings : ℕ)
  (tuesday_earnings : ℕ)
  (earnings_per_4_houses : ℕ)
  (houses_per_4 : ℝ)
  (remaining_days := days_per_week - 2)
  (remaining_goal := total_goal - monday_earnings - tuesday_earnings)
  (daily_goal := remaining_goal / remaining_days)
  (earnings_per_house := houses_per_4 / 4)
  (houses_per_day := daily_goal / earnings_per_house) :
  days_per_week = 5 ∧
  total_goal = 1000 ∧
  monday_earnings = 300 ∧
  tuesday_earnings = 40 ∧
  earnings_per_4_houses = 10 ∧
  houses_per_4 = earnings_per_4_houses.toReal →
  houses_per_day = 88 := 
by 
  sorry

end jackson_house_visits_l592_592292


namespace vlad_taller_than_sister_l592_592822

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end vlad_taller_than_sister_l592_592822


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592032

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592032


namespace product_of_first_three_terms_is_960_l592_592800

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l592_592800


namespace probability_sum_of_remaining_bills_l592_592530

def bagA : List ℕ := [10, 10, 1, 1, 1]
def bagB : List ℕ := [5, 5, 5, 5, 1, 1, 1]
def totalWaysA := Nat.choose 5 2
def totalWaysB := Nat.choose 7 2

def remainingSums (bag : List ℕ) (draw : List ℕ) : ℕ :=
  bag.sum - draw.sum

def isFavored (remA : ℕ) (remB : ℕ) : Bool :=
  remA > remB

theorem probability_sum_of_remaining_bills :
  ∑ (drawA ∈ (bagA.combinations 2)) (drawB ∈ (bagB.combinations 2)),
    if isFavored (remainingSums bagA drawA) (remainingSums bagB drawB)
    then (1 : ℚ)
    else (0 : ℚ) = (9 / 35 : ℚ) :=
by
  sorry

end probability_sum_of_remaining_bills_l592_592530


namespace sin_cos_identity_l592_592165

theorem sin_cos_identity : sin (π / 12) * cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_identity_l592_592165


namespace locus_of_point_P_is_circle_centered_at_M_l592_592137

theorem locus_of_point_P_is_circle_centered_at_M 
  (ABC : Triangle ℝ) (P : Point) (M : Point)
  (h_orthocenter : is_orthocenter M ABC)
  (A1 B1 C1 : Point)
  (h_perpendiculars : is_perpendicular_from_to P (altitudes ABC) (A1, B1, C1))
  (h_similarity_at_infinity : is_similar_at_infinity (Triangle.mk A1 B1 C1) ABC) :
  exists circle_center_radius : Circle,
  circle_center_radius.center = M ∧
  circle_center_radius.radius = diameter (circumcircle ABC) :=
sorry

end locus_of_point_P_is_circle_centered_at_M_l592_592137


namespace at_most_one_perfect_square_l592_592506

theorem at_most_one_perfect_square (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) :
  ∃ k, ∀ n, is_square (a n) → n = k :=
sorry

end at_most_one_perfect_square_l592_592506


namespace angle_C_value_b_l592_592668

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Conditions
axiom angle_relation : c * sin B = sqrt 3 * b * cos C
axiom side_relation : a^2 - c^2 = 2 * b^2

-- Problem (I): Find measure of angle C
theorem angle_C :
  C = Real.pi / 3 :=
by sorry

-- Conditions for Problem (II)
axiom area_relation : 1 / 2 * a * b * sin C = 21 * sqrt 3
axiom found_A : a = 3 * b

-- Problem (II): Find the value of b
theorem value_b :
  b = 2 * sqrt 7 :=
by sorry

end angle_C_value_b_l592_592668


namespace perpendicular_NR_QM_l592_592379

variable {P Q R K M N : Type}
variable [EuclideanGeometry P Q R K M N]

-- Assume basic geometric properties
variable (triangle_PQR : triangle P Q R)
variable (bisector_QK : angle_bisector Q K P R)
variable (circumcircle_PQR : circ P Q R)
variable (circumcircle_PKM : circ P K M)
variable (extension_PQ : extension_side P Q)
variable (PQ_extension_intersect : intersects (circ P K M) (extension_side P Q) N)

-- Hypotheses
hypothesis (H1 : QK intersects circumference of triangle PQR at point M distinct from Q)
hypothesis (H2 : M ≠ Q)
hypothesis (H3 : closest_approach circumcircle_PQR M)
hypothesis (H4 : circumcircular_segments triangle_PQR M intersect (extension_side P Q) at point N)

-- Required to prove
theorem perpendicular_NR_QM : perpendicular NR QM :=
by
  sorry

end perpendicular_NR_QM_l592_592379


namespace original_average_rent_l592_592495

theorem original_average_rent 
    (n_friends : ℕ)
    (new_mean_rent : ℕ)
    (increased_friend_rent : ℕ)
    (increased_percent : ℝ)
    (new_mean_value : ℕ)
    : n_friends = 4 → new_mean_rent = 880 → increased_friend_rent = 1600 → increased_percent = 0.20 → 
      (new_mean_value = ((4 * 880) - 1600 + 1920) / 4) → 
      (let original_avg_rent := 4 * 800 in original_avg_rent / 4 = 800) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end original_average_rent_l592_592495


namespace volume_removed_percentage_l592_592108

noncomputable def original_volume : ℕ := 20 * 15 * 10

noncomputable def cube_volume : ℕ := 4 * 4 * 4

noncomputable def total_volume_removed : ℕ := 8 * cube_volume

noncomputable def percentage_volume_removed : ℝ :=
  (total_volume_removed : ℝ) / (original_volume : ℝ) * 100

theorem volume_removed_percentage :
  percentage_volume_removed = 512 / 30 := sorry

end volume_removed_percentage_l592_592108


namespace sin_315_eq_neg_sqrt2_div_2_l592_592964

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592964


namespace forgot_homework_percentage_l592_592001

def GroupA := 20
def GroupB := 80
def GroupC := 50
def GroupD := 100

def GroupAForgotPercentage := 20
def GroupBForgotPercentage := 15
def GroupCForgotPercentage := 25
def GroupDForgotPercentage := 10

def TotalStudents := GroupA + GroupB + GroupC + GroupD

def GroupAForgot := GroupA * GroupAForgotPercentage / 100
def GroupBForgot := GroupB * GroupBForgotPercentage / 100
def GroupCForgot := GroupC * GroupCForgotPercentage / 100
def GroupDForgot := GroupD * GroupDForgotPercentage / 100

def TotalForgot := GroupAForgot + GroupBForgot + GroupCForgot + GroupDForgot

def PercentageForgot := (TotalForgot / TotalStudents) * 100

theorem forgot_homework_percentage (h1 : GroupA = 20) (h2 : GroupB = 80) (h3 : GroupC = 50) (h4 : GroupD = 100) 
  (h5 : GroupAForgotPercentage = 20) (h6 : GroupBForgotPercentage = 15) (h7 : GroupCForgotPercentage = 25) 
  (h8 : GroupDForgotPercentage = 10) : PercentageForgot = 15.6 :=
by
  sorry

end forgot_homework_percentage_l592_592001


namespace parabola_equation_length_AB_l592_592642

open Real

-- Conditions
def parabola_eq (x y : ℝ) (p : ℝ) : Prop := y^2 = 2 * p * x
def focus_on_line (x y : ℝ) : Prop := x + y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def angle_45_line (x y : ℝ) : Prop := y = x - 1

-- Main Statements to Prove
theorem parabola_equation (p : ℝ) (hp : 0 < p) :
  (∃ x y, focus_on_line x y ∧ parabola_eq x y p) →
  ∀ x y, parabola_eq x y 2 → parabola x y :=
sorry

theorem length_AB :
  parabola (y^2 - 4 * x = 0) →
  (∃ x₁ x₂ y₁ y₂,
     angle_45_line x₁ y₁ ∧
     angle_45_line x₂ y₂ ∧
     x₁ ≠ x₂ ∧
     parabola x₁ y₁ ∧
     parabola x₂ y₂ ∧
     sqrt (1 + 1) * (sqrt ((x₁ + x₂)^2 - 4 * x₁ * x₂)) = 8) :=
sorry

end parabola_equation_length_AB_l592_592642


namespace remaining_wire_in_cm_l592_592504

theorem remaining_wire_in_cm (total_mm : ℝ) (per_mobile_mm : ℝ) (conversion_factor : ℝ) :
  total_mm = 117.6 →
  per_mobile_mm = 4 →
  conversion_factor = 10 →
  ((total_mm % per_mobile_mm) / conversion_factor) = 0.16 :=
by
  intros htotal hmobile hconv
  sorry

end remaining_wire_in_cm_l592_592504


namespace product_invertible_integers_mod_120_eq_one_l592_592323

theorem product_invertible_integers_mod_120_eq_one :
  let n := ∏ i in (multiset.filter (λ x, Nat.coprime x 120) (multiset.range 120)), i
  in n % 120 = 1 := 
by
  sorry

end product_invertible_integers_mod_120_eq_one_l592_592323


namespace sum_proper_divisors_600_correct_l592_592066

-- Define the prime factorization of 600.
def prime_factors_600 : list (ℕ × ℕ) := [(2, 3), (3, 1), (5, 2)]

-- Define the sigma function based on the prime factorization
def sigma_600 : ℕ := (1 + 2 + 2^2 + 2^3) * (1 + 3) * (1 + 5 + 5^2)

-- From the provided summation, sigma(600) = 1860.
def sum_divisors_600 : ℕ := 1860

-- Define the condition which computes the sum of proper divisors
def sum_proper_divisors_600 : ℕ := sum_divisors_600 - 600

-- Prove that the sum of the proper divisors of 600 is 1260
theorem sum_proper_divisors_600_correct : sum_proper_divisors_600 = 1260 := by
  sorry

end sum_proper_divisors_600_correct_l592_592066


namespace sin_315_equals_minus_sqrt2_div_2_l592_592980

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592980


namespace chessboard_edge_count_l592_592011

theorem chessboard_edge_count (n : ℕ) 
  (border_white : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ w : ℕ, w ≥ n)) 
  (border_black : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ b : ℕ, b ≥ n)) :
  ∃ e : ℕ, e ≥ n :=
sorry

end chessboard_edge_count_l592_592011


namespace min_value_lambda_mu_l592_592257

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C P M N : V)
variables (λ μ : ℝ)

-- Condition: P is a point on the side BC such that \overrightarrow{BP} = \frac{1}{2}\overrightarrow{PC}.
def point_on_side : Prop :=
  ∃ t : ℝ, t = 1 / 3 ∧ P = B + t • (C - B)

-- Condition: Given expressions for AM and AN with λ, μ > 0.
def expressions_for_AM_AN : Prop :=
  M = A + λ • (B - A) ∧ N = A + μ • (C - A) ∧ λ > 0 ∧ μ > 0

-- Condition: M and N lie on a line passing through P.
def line_passing_through : Prop :=
  ∃ k1 k2 : ℝ, M = A + k1 • (P - A) ∧ N = A + k2 • (P - A) ∧ k1 > 0 ∧ k2 > 0

-- Prove that the minimum value of λ + 2μ is 2/3.
theorem min_value_lambda_mu (h1 : point_on_side A B C P) (h2 : expressions_for_AM_AN A B C M N λ μ) (h3 : line_passing_through A M N P) : λ + 2 * μ = 2 / 3 :=
by sorry

end min_value_lambda_mu_l592_592257


namespace sin_315_degree_l592_592953

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592953


namespace reinforcement_left_after_days_l592_592868

theorem reinforcement_left_after_days
  (initial_men : ℕ) (initial_days : ℕ) (remaining_days : ℕ) (men_left : ℕ)
  (remaining_men : ℕ) (x : ℕ) :
  initial_men = 400 ∧
  initial_days = 31 ∧
  remaining_days = 8 ∧
  men_left = initial_men - remaining_men ∧
  remaining_men = 200 ∧
  400 * 31 - 400 * x = 200 * 8 →
  x = 27 :=
by
  intros h
  sorry

end reinforcement_left_after_days_l592_592868


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592947

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592947


namespace sum_of_odd_integers_between_15_and_41_l592_592805

theorem sum_of_odd_integers_between_15_and_41 : 
  (∑ i in (finset.range ((41 - 15) / 2 + 1)).map (λ x, 15 + 2 * x)) = 392 :=
by
  sorry

end sum_of_odd_integers_between_15_and_41_l592_592805


namespace length_of_PS_l592_592284

theorem length_of_PS 
  (PQ QR PR : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 15)
  (hPR : PR = 17)
  {P Q R S : Type}
  (PQR : ∠ QPR → 90°)
  (PS_bisects_QPR : PS = angleBisector (∠ QPR)) : PS = 8 := 
sorry

end length_of_PS_l592_592284


namespace distinct_squares_with_at_least_6_black_l592_592857

-- Define the checkerboard
def checkerboard := List (List Bool)

-- Define the property of containing at least 6 black squares
def contains_at_least_6_black_squares (square : checkerboard) : Prop :=
  square.flatten.count (· = false) ≥ 6

-- Define a function to count squares of arbitrary size on a 9x9 checkerboard
def count_squares_at_least_6_black (board : checkerboard) : Nat :=
  let sizes := [4, 5, 6, 7, 8, 9]
  sizes.sum (λ size, (9 - size + 1) * (9 - size + 1))

-- The proof goal
theorem distinct_squares_with_at_least_6_black (board : checkerboard) (h : ∀ x y, board.get x y = x % 2 = y % 2) :
  count_squares_at_least_6_black board = 91 :=
by
  sorry

end distinct_squares_with_at_least_6_black_l592_592857


namespace spider_leg_pressure_l592_592493

/--
A giant spider is discovered. It weighs 2.5 times the previous largest spider, 
which weighed 6.4 ounces. Each of its legs has a cross-sectional area of 0.5 
square inches. How much pressure in ounces per square inch does each leg undergo?
-/
theorem spider_leg_pressure
  (weight_previous_spider : ℝ := 6.4)
  (weight_ratio : ℝ := 2.5)
  (cross_sectional_area : ℝ := 0.5)
  (number_of_legs : ℕ := 8) :
  let weight_giant_spider := weight_ratio * weight_previous_spider in
  let weight_per_leg := weight_giant_spider / number_of_legs in
  let pressure_per_leg := weight_per_leg / cross_sectional_area in
  pressure_per_leg = 4 :=
by
  let weight_giant_spider := weight_ratio * weight_previous_spider
  let weight_per_leg := weight_giant_spider / number_of_legs
  let pressure_per_leg := weight_per_leg / cross_sectional_area
  sorry

end spider_leg_pressure_l592_592493


namespace simplify_fraction_l592_592761

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l592_592761


namespace set_subset_of_inter_union_l592_592734

variable {α : Type} [Nonempty α]
variables {A B C : Set α}

-- The main theorem based on the problem statement
theorem set_subset_of_inter_union (h : A ∩ B = B ∪ C) : C ⊆ B :=
by
  sorry

end set_subset_of_inter_union_l592_592734


namespace max_stage_for_less_than_2014_squares_l592_592006

def number_of_squares (k : ℕ) : ℕ :=
  2 * k^2 - 2 * k + 1

theorem max_stage_for_less_than_2014_squares :
  ∃ (k : ℕ), ∀ (m : ℕ), (m ≤ k) → number_of_squares(m) < 2014 := sorry

end max_stage_for_less_than_2014_squares_l592_592006


namespace question1_question2_l592_592907

-- Problem 1
theorem question1 (π : ℝ) : 
  sqrt((π - 13/4)^2) + (16/49)^(-1/2:ℝ) + (-8)^(2/3:ℝ) + 8^(0.25:ℝ) * 2^(1/4:ℝ) + ((π - 2)^3)^(1/3:ℝ) = 9 :=
sorry

-- Problem 2
theorem question2 : 
  1/2 * log 25 + log 2 - log (sqrt 0.1) - (log 9 / log 2) * (log 2 / log 3) = -1/2 :=
sorry

end question1_question2_l592_592907


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592044

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592044


namespace prob_john_meets_train_l592_592296

noncomputable def john_arrival_time : MeasureTheory.Measure ℝ := MeasureTheory.Measure.unif 90 -- John arrives uniformly between 1:30 (0 minutes) and 3:00 (90 minutes)
noncomputable def train_arrival_time : MeasureTheory.Measure ℝ := MeasureTheory.Measure.unif 60 -- Train arrives uniformly between 2:00 (0 minutes) and 3:00 (60 minutes)
noncomputable def prob_train_at_station_when_john_arrives : ℝ := 4/27

theorem prob_john_meets_train :
  MeasureTheory.Probability (set.prod {j | j > -0.5} {t | t < j + 20}) = 4/27 := sorry

end prob_john_meets_train_l592_592296


namespace sum_of_digits_of_expression_l592_592050

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592050


namespace modern_population_growth_model_causes_older_age_structure_l592_592384

-- Definitions based on conditions
def modern_population_growth_model (P : Type) := 
  (average_lifespan_increases : Prop) × (generational_turnover_slows_down : Prop)

-- Population age structure definition
def population_age_structure_older (P : Type) := Prop

-- The statement we want to prove
theorem modern_population_growth_model_causes_older_age_structure (P : Type)
  (h : modern_population_growth_model P) :
  population_age_structure_older P :=
sorry

end modern_population_growth_model_causes_older_age_structure_l592_592384


namespace vlad_taller_by_41_inches_l592_592825

/-- Vlad's height is 6 feet and 3 inches. -/
def vlad_height_feet : ℕ := 6

def vlad_height_inches : ℕ := 3

/-- Vlad's sister's height is 2 feet and 10 inches. -/
def sister_height_feet : ℕ := 2

def sister_height_inches : ℕ := 10

/-- There are 12 inches in a foot. -/
def inches_in_a_foot : ℕ := 12

/-- Convert height in feet and inches to total inches. -/
def convert_to_inches (feet inches : ℕ) : ℕ :=
  feet * inches_in_a_foot + inches

/-- Proof that Vlad is 41 inches taller than his sister. -/
theorem vlad_taller_by_41_inches : convert_to_inches vlad_height_feet vlad_height_inches - convert_to_inches sister_height_feet sister_height_inches = 41 :=
by
  -- Start the proof
  sorry

end vlad_taller_by_41_inches_l592_592825


namespace smallest_planes_covering_S_l592_592728

def S (n : ℕ) : set (ℕ × ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ p.3 ≤ n ∧ p.1 + p.2 + p.3 > 0}

theorem smallest_planes_covering_S (n : ℕ) (hn : n > 1) :
    ∃ k : ℕ, ∀ x y z, (x, y, z) ∈ S n → (x + y + z ≤ k) ∧ (∀ i, (x = i ∨ y = i ∨ z = i) → i > 0) → k ≥ 3 * n :=
sorry

end smallest_planes_covering_S_l592_592728


namespace round_trip_average_speed_l592_592838

theorem round_trip_average_speed : 
  (∀ a : ℝ, a > 0 → 
  let time_go := a / 6 in
  let time_back := a / 4 in 
  let avg_speed := 2 * a / (time_go + time_back) in 
  avg_speed = 4.8) :=
begin
  intros a ha,
  let time_go := a / 6,
  let time_back := a / 4,
  let avg_speed := 2 * a / (time_go + time_back),
  have : avg_speed = 4.8,
  { sorry },
  exact this,
end

end round_trip_average_speed_l592_592838


namespace three_alpha_four_plus_eight_beta_three_eq_876_l592_592625

variable (α β : ℝ)

-- Condition 1: α and β are roots of the equation x^2 - 3x - 4 = 0
def roots_of_quadratic : Prop := α^2 - 3 * α - 4 = 0 ∧ β^2 - 3 * β - 4 = 0

-- Question: 3α^4 + 8β^3 = ?
theorem three_alpha_four_plus_eight_beta_three_eq_876 
  (h : roots_of_quadratic α β) : (3 * α^4 + 8 * β^3 = 876) := sorry

end three_alpha_four_plus_eight_beta_three_eq_876_l592_592625


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592929

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592929


namespace complex_modulus_problem_l592_592247

-- Define what it means for i to be the imaginary unit (i^2 = -1)
def isImaginaryUnit (i : ℂ) : Prop := i^2 = -1

-- Define the conditions of the problem
variables (i : ℂ) (z : ℂ)
hypothesis (h_imag_unit : isImaginaryUnit i)
hypothesis (h_eq : (1 - i * z) = 1)

-- The goal is to prove that |2z - 3| = √5
theorem complex_modulus_problem :
  |2 * z - 3| = real.sqrt 5 :=
sorry

end complex_modulus_problem_l592_592247


namespace angela_action_figures_left_l592_592118

theorem angela_action_figures_left :
  ∀ (initial_collection : ℕ), 
  initial_collection = 24 → 
  (let sold := initial_collection / 4 in
   let remaining_after_sold := initial_collection - sold in
   let given_to_daughter := remaining_after_sold / 3 in
   let remaining_after_given := remaining_after_sold - given_to_daughter in
   remaining_after_given = 12) :=
by
  intros
  sorry

end angela_action_figures_left_l592_592118


namespace distinct_numbers_floor_div_l592_592161

theorem distinct_numbers_floor_div (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500) :
  finset.card (finset.image (λ n, ⌊(n^2 : ℝ)/500⌋) (finset.range 501)) = 376 := 
by
  sorry

end distinct_numbers_floor_div_l592_592161


namespace _l592_592576

lemma right_triangle_angles (AB BC AC : ℝ) (α β : ℝ)
  (h1 : AB = 1) 
  (h2 : BC = Real.sin α)
  (h3 : AC = Real.cos α)
  (h4 : AB^2 = BC^2 + AC^2) -- Pythagorean theorem for the right triangle
  (h5 : α = (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1))) :
  β = 90 - (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) :=
sorry

end _l592_592576


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592048

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592048


namespace max_largest_number_l592_592510

theorem max_largest_number (n : ℕ) (arcs : Fin n → ℕ × ℕ)
  (h1 : n = 1000)
  (h2 : ∀ i : Fin n, (arcs i).fst + (arcs i).snd ∣ (arcs (Fin.succ i)).fst * (arcs (Fin.succ i)).snd) :
  ∃ (k : ℕ), k = 2001 ∧ ∀ i : Fin n, (arcs i).fst ≤ k ∧ (arcs i).snd ≤ k :=
sorry

end max_largest_number_l592_592510


namespace product_invertibles_mod_120_eq_1_l592_592331

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def product_of_invertibles_mod_n (n : ℕ) :=
  List.prod (List.filter (fun x => is_coprime x n) (List.range n))

theorem product_invertibles_mod_120_eq_1 :
  product_of_invertibles_mod_n 120 % 120 = 1 := 
sorry

end product_invertibles_mod_120_eq_1_l592_592331


namespace arithmetic_sequence_general_term_l592_592200

theorem arithmetic_sequence_general_term (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 7)
  (h_a7 : a 7 = 3) :
  ∀ n, a n = -↑n + 10 :=
by
  sorry

end arithmetic_sequence_general_term_l592_592200


namespace spider_leg_pressure_l592_592490

-- Definitions based on given conditions
def weight_of_previous_spider := 6.4 -- ounces
def weight_multiplier := 2.5
def cross_sectional_area := 0.5 -- square inches
def number_of_legs := 8

-- Theorem stating the problem
theorem spider_leg_pressure : 
  (weight_multiplier * weight_of_previous_spider) / number_of_legs / cross_sectional_area = 4 := 
by 
  sorry

end spider_leg_pressure_l592_592490


namespace cost_of_cd_l592_592299

theorem cost_of_cd 
  (cost_film : ℕ) (cost_book : ℕ) (total_spent : ℕ) (num_cds : ℕ) (total_cost_films : ℕ)
  (total_cost_books : ℕ) (cost_cd : ℕ) : 
  cost_film = 5 → cost_book = 4 → total_spent = 79 →
  total_cost_films = 9 * cost_film → total_cost_books = 4 * cost_book →
  total_spent = total_cost_films + total_cost_books + num_cds * cost_cd →
  num_cds = 6 →
  cost_cd = 3 := 
by {
  -- proof would go here
  sorry
}

end cost_of_cd_l592_592299


namespace number_of_players_l592_592810

theorem number_of_players (x y z : ℕ) 
  (h1 : x + y + z = 10)
  (h2 : x * y + y * z + z * x = 31) : 
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 3) ∨ (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 3) ∨ (x = 5 ∧ y = 3 ∧ z = 2) :=
sorry

end number_of_players_l592_592810


namespace minor_premise_l592_592562

-- Definitions
def Rectangle : Type := sorry
def Square : Type := sorry
def Parallelogram : Type := sorry

axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle
axiom square_is_parallelogram : Square → Parallelogram

-- Problem statement
theorem minor_premise : ∀ (S : Square), ∃ (R : Rectangle), square_is_rectangle S = R :=
by
  sorry

end minor_premise_l592_592562


namespace cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l592_592211

theorem cos_alpha_plus_5pi_over_12_eq_neg_1_over_3
  (α : ℝ)
  (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l592_592211


namespace max_value_A_l592_592088

theorem max_value_A (n : ℕ) (x y : Fin n → ℝ) 
  (h_condition : (∑ i, (x i) ^ 2) + (∑ i, (y i) ^ 2) ≤ 1) :
  let sum_x := ∑ i in Finset.range n, x i
  let sum_y := ∑ i in Finset.range n, y i
  let A := (3 * sum_x - 5 * sum_y) * (5 * sum_x + 3 * sum_y)
in A ≤ 17 * n ∧ A = 17 * n := sorry

end max_value_A_l592_592088


namespace probability_of_C_l592_592097

theorem probability_of_C (P : ℕ → ℚ) (P_total : P 1 + P 2 + P 3 = 1)
  (P_A : P 1 = 1/3) (P_B : P 2 = 1/2) : P 3 = 1/6 :=
by
  sorry

end probability_of_C_l592_592097


namespace f_monotonicity_l592_592391

noncomputable def f (x : ℝ) : ℝ := (√(1 - cos (2 * x))) / (cos x)

theorem f_monotonicity :
  (∀ x, 0 ≤ x ∧ x < π/2 → (∀ y, 0 ≤ y ∧ y < π/2 → x < y → f x < f y)) ∧
  (∀ x, π/2 < x ∧ x ≤ π → (∀ y, π/2 < y ∧ y ≤ π → x < y → f x < f y)) ∧
  (∀ x, π ≤ x ∧ x < (3 * π / 2) → (∀ y, π ≤ y ∧ y < (3 * π / 2) → x < y → f x > f y)) ∧
  (∀ x, (3 * π / 2) < x ∧ x ≤ 2 * π → (∀ y, (3 * π / 2) < y ∧ y ≤ 2 * π → x < y → f x > f y)) :=
  sorry

end f_monotonicity_l592_592391


namespace sin_315_equals_minus_sqrt2_div_2_l592_592982

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592982


namespace sum_of_ratios_in_triangle_l592_592287

theorem sum_of_ratios_in_triangle
  (A B C D E F : Type)
  [is_midpoint D A B]
  [is_trisection_point E A C]
  [is_ratio_point F D E 2 1]
  : (EF / FB) + (AF / FC) = 4.5 :=
sorry

end sum_of_ratios_in_triangle_l592_592287


namespace correct_equation_among_options_l592_592113

theorem correct_equation_among_options :
  (∀ (n m : ℝ), ( (n / m)^7 ≠ n^7 * m^(1 / 7)) ∧ 
  (sqrt (∛(9:ℝ)) = ∛(3)) ∧ 
  (∀ (x y : ℝ), sqrt((x ^ 3 + y ^ 3) ^ (1 / 4)) ≠ (x + y) ^ (3 / 4)) ∧ 
  (sqrt(∛(9)) = ∛(3))) := 
by 
  sorry

end correct_equation_among_options_l592_592113


namespace product_invertibles_mod_120_l592_592314

theorem product_invertibles_mod_120 :
  let n := (list.filter (λ k, Nat.coprime k 120) (list.range 120)).prod
  in n % 120 = 119 :=
by
  sorry

end product_invertibles_mod_120_l592_592314


namespace z_share_of_earnings_l592_592847

theorem z_share_of_earnings
  (x_work_rate : ℚ := 1/2)
  (y_work_rate : ℚ := 1/4)
  (z_work_rate : ℚ := 1/6)
  (total_earnings : ℚ := 2000) :
  let combined_work_rate := x_work_rate + y_work_rate + z_work_rate
  in let z_share := (z_work_rate / combined_work_rate) * total_earnings
  in z_share = 363.64 := sorry

end z_share_of_earnings_l592_592847


namespace fraction_addition_l592_592900

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l592_592900


namespace sum_of_prime_factors_of_2550_l592_592065

theorem sum_of_prime_factors_of_2550 : 
  let n := 2550
  let prime_factors := {2, 3, 5, 17}
  finset.sum prime_factors id = 27 := 
by 
  let n := 2550
  let prime_factors := {2, 3, 5, 17}
  have h : finset.sum prime_factors id = 27 := by
    sorry
  exact h

end sum_of_prime_factors_of_2550_l592_592065


namespace product_invertibles_mod_120_eq_1_l592_592327

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def product_of_invertibles_mod_n (n : ℕ) :=
  List.prod (List.filter (fun x => is_coprime x n) (List.range n))

theorem product_invertibles_mod_120_eq_1 :
  product_of_invertibles_mod_n 120 % 120 = 1 := 
sorry

end product_invertibles_mod_120_eq_1_l592_592327


namespace product_of_first_three_terms_of_arithmetic_sequence_l592_592798

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l592_592798


namespace inequality_exponentiation_l592_592613

theorem inequality_exponentiation (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb : b < 1) (hc : c > 1) : 
  a * b^c > b * a^c := 
sorry

end inequality_exponentiation_l592_592613


namespace prob_same_color_is_correct_l592_592569

noncomputable def prob_same_color : ℚ :=
  let green_prob := (8 : ℚ) / 10
  let red_prob := (2 : ℚ) / 10
  (green_prob)^2 + (red_prob)^2

theorem prob_same_color_is_correct :
  prob_same_color = 17 / 25 := by
  sorry

end prob_same_color_is_correct_l592_592569


namespace sum_of_divisors_of_36_l592_592440

theorem sum_of_divisors_of_36 : (∑ d in (List.range 37).filter (λ n, 36 % n = 0), d) = 91 := 
by 
  sorry

end sum_of_divisors_of_36_l592_592440


namespace positive_root_of_equation_l592_592590

theorem positive_root_of_equation :
  ∃ a b : ℤ, (a + b * Real.sqrt 3)^3 - 5 * (a + b * Real.sqrt 3)^2 + 2 * (a + b * Real.sqrt 3) - Real.sqrt 3 = 0 ∧
    a + b * Real.sqrt 3 > 0 ∧
    (a + b * Real.sqrt 3) = 3 + Real.sqrt 3 := 
by
  sorry

end positive_root_of_equation_l592_592590


namespace sum_of_proper_divisors_600_l592_592069

-- Define a function to calculate the sum of divisors
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in finset.range n.succ, if n % i = 0 then i else 0

-- Define a function to calculate the sum of proper divisors
def sum_of_proper_divisors (n : ℕ) : ℕ :=
  sum_of_divisors n - n

-- Assert that the sum of the proper divisors of 600 is 1260
theorem sum_of_proper_divisors_600 : sum_of_proper_divisors 600 = 1260 :=
  sorry

end sum_of_proper_divisors_600_l592_592069


namespace zach_min_tshirt_packages_l592_592839

theorem zach_min_tshirt_packages (tshirt_per_package hats_per_package : ℕ) :
  (tshirt_per_package = 12) →
  (hats_per_package = 10) →
  ∃ (n : ℕ), (n * tshirt_per_package = m * hats_per_package ∧ n = 5) :=
by
  intros h1 h2
  let t := 12
  let h := 10
  have h_tshirt : tshirt_per_package = t := h1
  have h_hats : hats_per_package = h := h2
  use 5
  split
  · have : 60 = 5 * 12 := by norm_num
    exact this.symm
  · exact rfl

end zach_min_tshirt_packages_l592_592839


namespace lara_puts_flowers_in_vase_l592_592711

theorem lara_puts_flowers_in_vase : 
  ∀ (total_flowers mom_flowers flowers_given_more : ℕ), 
    total_flowers = 52 →
    mom_flowers = 15 →
    flowers_given_more = 6 →
  (total_flowers - (mom_flowers + (mom_flowers + flowers_given_more))) = 16 :=
by
  intros total_flowers mom_flowers flowers_given_more h1 h2 h3
  sorry

end lara_puts_flowers_in_vase_l592_592711


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592930

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592930


namespace true_propositions_l592_592636

variables (prop1 prop2 prop3 prop4 : Prop)
-- Conditions from a)
axiom H1 : prop1 = (exists (P : Type) (l : P -> Prop) (h : ∀ p : P, ∃! l : Prop, l = l))
axiom H2 : prop2 = (exists (P : Type) (l : P -> Prop) (h : ∀ p : P, ∃ l : Prop, l = l))
axiom H3 : prop3 = (exists (P : Type) (l1 l2 l3 : P -> Prop) (h1 : ∀ p : P, l1 p = l2 p → l3 p))
axiom H4 : prop4 = (exists (P : Type) (l1 l2 : P -> Prop) (h : ∀ p : P, l1 p → l2 p → l1 p))

-- Goal
theorem true_propositions : {prop1, prop3, prop4} = {p | p = prop1 ∨ p = prop3 ∨ p = prop4} :=
sorry

end true_propositions_l592_592636


namespace conditional_extremum_l592_592578

noncomputable def f (x₁ x₂ : ℝ) : ℝ :=
  x₁^2 + x₂^2 - x₁ * x - 2 + x₁ + x₂ - 4

def constraint (x₁ x₂ : ℝ) : Prop :=
  x₁ + x₂ + 3 = 0

theorem conditional_extremum :
  ∃ x₁ x₂ : ℝ, constraint x₁ x₂ ∧ f x₁ x₂ = -4.5 :=
by
  use (-1.5, -1.5)
  split
  · -- Constraints satisfaction
    sorry
  · -- Function value satisfaction
    sorry

end conditional_extremum_l592_592578


namespace correctness_of_propositions_l592_592885

-- Define conditions as propositions
variables (line l : Type) (plane α β : Type)
variables (points_on_line : list l) (points_on_plane : list α)
variable (skew_lines : Type)
variable h1 : ∀ (points_on_line : list l), (∀ p ∈ points_on_line, ∃ d, point_eq_dist_to_plane p α d) → l ∥ α
variable h2 : ∀ (points_on_plane : list α), (∀ p1 p2 p3 ∈ points_on_plane, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) → (∀ p ∈ points_on_plane, ∃ d, point_eq_dist_to_plane p β d) → α ∥ β
variable h3 : ∀ (l : Type), (plane ⊥ l α ∧ plane ⊥ l β) → α ∥ β
variable h4 : ∀ (l : Type), (plane ∥ l α ∧ plane ∥ l β) → α ∥ β
variable h5 : ∀ (a b : skew_lines), (a ∈ α ∧ b ∥ α ∧ b ∈ β ∧ a ∥ β) → α ∥ β

-- State correctness of propositions
theorem correctness_of_propositions : 
    (∀ (h3 : ∀ (l : Type), (plane ⊥ l α ∧ plane ⊥ l β) → α ∥ β), True) ∧
    (∀ (h5 : ∀ (a b : skew_lines), (a ∈ α ∧ b ∥ α ∧ b ∈ β ∧ a ∥ β) → α ∥ β), True) ∧
    (∀ (h1 : ∀ (points_on_line : list l), (∀ p ∈ points_on_line, ∃ d, point_eq_dist_to_plane p α d) → l ∥ α), False) ∧
    (∀ (h2 : ∀ (points_on_plane : list α), (∀ p1 p2 p3 ∈ points_on_plane, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) → (∀ p ∈ points_on_plane, ∃ d, point_eq_dist_to_plane p β d) → α ∥ β), False) ∧
    (∀ (h4 : ∀ (l : Type), (plane ∥ l α ∧ plane ∥ l β) → α ∥ β), False) :=
by
    sorry

end correctness_of_propositions_l592_592885


namespace smallest_discount_l592_592169

theorem smallest_discount (n : ℕ) (h1 : (1 - 0.12) * (1 - 0.18) = 0.88 * 0.82)
  (h2 : (1 - 0.08) * (1 - 0.08) * (1 - 0.08) = 0.92 * 0.92 * 0.92)
  (h3 : (1 - 0.20) * (1 - 0.10) = 0.80 * 0.90) :
  (29 > 27.84 ∧ 29 > 22.1312 ∧ 29 > 28) :=
by {
  sorry
}

end smallest_discount_l592_592169


namespace ab_fraction_l592_592399

theorem ab_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 9) (h2 : a * b = 20) : 
  (1 / a + 1 / b) = 9 / 20 := 
by 
  sorry

end ab_fraction_l592_592399


namespace segment_GH_length_l592_592687

-- Define the lengths of the segments as given conditions
def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

-- Define the length of segment GH to be proved
def length_GH : ℕ := length_AB + length_CD + length_FE

-- The theorem that needs to be proved
theorem segment_GH_length : length_GH = 29 := by
  -- We can use the definitions of length_AB, length_FE, and length_CD directly here
  unfold length_GH
  rw [length_AB, length_CD, length_FE]
  norm_num
  sorry

end segment_GH_length_l592_592687


namespace value_of_c_l592_592442

theorem value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 10 < 0) ↔ (x < 2 ∨ x > 8)) → c = 10 :=
by
  sorry

end value_of_c_l592_592442


namespace median_number_of_books_l592_592669

theorem median_number_of_books :
  ∀ (students : List ℚ), 
  students = [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8] →
  students.length = 12 →
  let sorted_students := students.sorted in
  let n := sorted_students.length in
  2 ∣ n →
  (sorted_students[n/2 - 1] + sorted_students[n/2]) / 2 = (4.5 : ℚ) :=
by
  intro students h1 h2 sorted_students n h3
  have h4 : sorted_students = [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8] := by
    sorry -- the sorted list equal to the given list
  have h5 : n = 12 := by
    sorry -- the length of the list is 12
  have h6 : n / 2 - 1 = 5 := by
    sorry -- computing the middle index 1 (0-based)
  have h7 : n / 2 = 6 := by
    sorry -- computing the middle index 2 (0-based)
  have h8 : sorted_students[5] = 4 := by
    sorry -- the 6th element (0-based) is 4
  have h9 : sorted_students[6] = 5 := by
    sorry -- the 7th element (0-based) is 5
  have h10 : (4 + 5) / 2 = 4.5 := by
    sorry -- the median computation
  exact h10

end median_number_of_books_l592_592669


namespace complex_number_quadrant_l592_592682

def i_squared : ℂ := -1

def z (i : ℂ) : ℂ := (-2 + i) * i^5

def in_quadrant_III (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant 
  (i : ℂ) (hi : i^2 = -1) (z_val : z i = (-2 + i) * i^5) :
  in_quadrant_III (z i) :=
sorry

end complex_number_quadrant_l592_592682


namespace distinct_numbers_floor_div_l592_592160

theorem distinct_numbers_floor_div (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500) :
  finset.card (finset.image (λ n, ⌊(n^2 : ℝ)/500⌋) (finset.range 501)) = 376 := 
by
  sorry

end distinct_numbers_floor_div_l592_592160


namespace find_range_a_l592_592645

-- Define sets A and B given the conditions
def A : Set ℝ := { x : ℝ | (1 / 2)^(x^2 - x - 6) < 1 }

def B (a : ℝ) : Set ℝ := { x : ℝ | log 4 (x + a) < 1 }

-- Define the intersection condition for sets A and B to be empty
def A_inter_B_empty (a : ℝ) : Prop := A ∩ B a = ∅

-- Translate the given condition and correct answer into a Lean theorem
theorem find_range_a (a : ℝ) : A_inter_B_empty a ↔ 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end find_range_a_l592_592645


namespace locus_of_Q_is_circle_l592_592187

open Complex

noncomputable def locus_of_Q (C_center : ℂ) (r : ℝ) (A : ℂ) : set ℂ :=
  let P := λ θ : ℝ, r * exp (θ * I)
  let Q := λ p : ℂ, (1 - I) * A + I * p in
  {q : ℂ | ∃ θ : ℝ, q = Q (P θ)}

theorem locus_of_Q_is_circle (C_center : ℂ) (r : ℝ) (A : ℂ) (A_exterior : abs A > r) :
  locus_of_Q C_center r A = {q : ℂ | abs (q - ((1 - I) * A)) = r * sqrt 2} :=
sorry

end locus_of_Q_is_circle_l592_592187


namespace sum_of_digits_of_expression_l592_592040

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592040


namespace num_full_servings_l592_592499

-- Define the original amount of peanut butter as a rational number
def original_amount : ℚ := 35 + 2/3

-- Define the used amount of peanut butter as a rational number
def used_amount : ℚ := 5 + 1/3

-- Define the amount of peanut butter per serving as a rational number
def serving_size : ℚ := 3

-- Define the remaining peanut butter after using some for the recipe
def remaining_amount : ℚ := original_amount - used_amount

-- Define the total number of servings that can be made from the remaining amount
def num_servings : ℚ := remaining_amount / serving_size

-- State the theorem that we need to prove
theorem num_full_servings : num_servings.floor = 10 := 
by
  sorry

end num_full_servings_l592_592499


namespace g_crosses_horizontal_asymptote_at_l592_592176

noncomputable def g (x : ℝ) : ℝ :=
  (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 3)

theorem g_crosses_horizontal_asymptote_at :
  ∃ x : ℝ, g(x) = 3 ↔ x = 17 / 8 :=
by
  unfold g
  sorry

end g_crosses_horizontal_asymptote_at_l592_592176


namespace rational_coeff_quadratic_roots_rational_or_irrational_l592_592451

theorem rational_coeff_quadratic_roots_rational_or_irrational
    (a b c : ℚ) 
    (h : a ≠ 0) : 
    ∀ (x₀ x₁ : ℚ), 
    (x₀ * x₀ = b^2 - 4 * a * c) → 
    (x₀ + x₁ = -b / a) ∧ (x₀ * x₁ = c / a) → 
    (∃ d : ℚ, d * d = b^2 - 4 * a * c) → 
    (x₀ ∈ ℚ ∧ x₁ ∈ ℚ) ∨ (x₀ ∉ ℚ ∧ x₁ ∉ ℚ) :=
begin
    sorry
end

end rational_coeff_quadratic_roots_rational_or_irrational_l592_592451


namespace problem1_smallest_positive_period_problem1_range_problem2_solution_l592_592723

noncomputable def f (x : ℝ) : ℝ := sin x * sin x + sqrt 3 * sin x * cos x - 1 / 2

theorem problem1_smallest_positive_period :
  ∃ p > 0, ∀ x, f (x + p) = f x :=
sorry

theorem problem1_range :
  ∀ y, f x = y → y ∈ Icc (-1 : ℝ) 1 :=
sorry

variables (a b c A B C : ℝ)
axiom condition_acute_angle : A < π / 2
axiom condition_lengths : a = 2 * sqrt 3 ∧ c = 4

theorem problem2_solution :
  f A = 1 → A = π / 3 ∧ b = 2 :=
sorry

end problem1_smallest_positive_period_problem1_range_problem2_solution_l592_592723


namespace sum_of_roots_eq_eight_l592_592377

open Real

theorem sum_of_roots_eq_eight (f : ℝ → ℝ) 
  (h_symm : ∀ x, f(2 + x) = f(2 - x)) 
  (h_roots : set.countable {x | f x = 0}) 
  (h_distinct : (∃ (r1 r2 r3 r4 : ℝ), 
               r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ 
               r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧
               f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0 ∧ f r4 = 0)) : 
  (∀ (r1 r2 r3 r4 : ℝ), 
       f r1 = 0 → f r2 = 0 → f r3 = 0 → f r4 = 0 → 
       r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 → 
       r1 + r2 + r3 + r4 = 8) := 
by
  sorry

end sum_of_roots_eq_eight_l592_592377


namespace product_of_roots_eq_l592_592143

theorem product_of_roots_eq :
  ∀ {a b c : ℝ} (h : a ≠ 0), (24 * (1 / 24) * 200 / 24 = 25 / 3) → 
    0 = 24 * (1 / 24 - b + 200 / 24 / a) :=
by
  intros a b c h ha
  have :
  h : a ≠ 0
  show : 0 = 24 * (1 / 24) - 25 / 3
  sorry

end product_of_roots_eq_l592_592143


namespace ben_paints_area_l592_592884

variable (allen_ratio : ℕ) (ben_ratio : ℕ) (total_area : ℕ)
variable (total_ratio : ℕ := allen_ratio + ben_ratio)
variable (part_size : ℕ := total_area / total_ratio)

theorem ben_paints_area 
  (h1 : allen_ratio = 2)
  (h2 : ben_ratio = 6)
  (h3 : total_area = 360) : 
  ben_ratio * part_size = 270 := sorry

end ben_paints_area_l592_592884


namespace kaleb_spent_amount_l592_592525

def tickets_spent (initial_tickets ferris_wheel_tickets bumper_cars_tickets roller_coaster_tickets : ℕ) : ℕ :=
  initial_tickets - ferris_wheel_tickets - bumper_cars_tickets - roller_coaster_tickets

def total_cost (tickets_used ticket_price : ℕ) : ℕ :=
  tickets_used * ticket_price

theorem kaleb_spent_amount :
  let initial_tickets := 6
  let ferris_wheel_tickets := 2
  let bumper_cars_tickets := 1
  let roller_coaster_tickets := 2
  let ticket_price := 9
  let tickets_used := ferris_wheel_tickets + bumper_cars_tickets + roller_coaster_tickets
  total_cost tickets_used ticket_price = 45 :=
by
  let initial_tickets := 6
  let ferris_wheel_tickets := 2
  let bumper_cars_tickets := 1
  let roller_coaster_tickets := 2
  let ticket_price := 9
  have tickets_used := ferris_wheel_tickets + bumper_cars_tickets + roller_coaster_tickets
  show total_cost tickets_used ticket_price = 45
  sorry

end kaleb_spent_amount_l592_592525


namespace vlad_taller_than_sister_l592_592823

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end vlad_taller_than_sister_l592_592823


namespace work_required_to_pump_liquid_l592_592908

/-- Calculation of work required to pump a liquid of density ρ out of a parabolic boiler. -/
theorem work_required_to_pump_liquid
  (ρ g H a : ℝ)
  (h_pos : 0 < H)
  (a_pos : 0 < a) :
  ∃ (A : ℝ), A = (π * ρ * g * H^3) / (6 * a^2) :=
by
  -- TODO: Provide the proof.
  sorry

end work_required_to_pump_liquid_l592_592908


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592926

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592926


namespace greatest_three_digit_number_l592_592830

theorem greatest_three_digit_number :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ N % 8 = 2 ∧ N % 7 = 4 ∧ N = 978 :=
by
  sorry

end greatest_three_digit_number_l592_592830


namespace find_length_of_nm_l592_592005

theorem find_length_of_nm 
  (O₁ O₂ A B M N C : Point)
  (h_circle₁ : Circle O₁ (dist O₁ C))
  (h_circle₂ : Circle O₂ (dist O₂ C))
  (h_intersect : C ∈ (⋂ (Circle O₁) (Circle O₂)))
  (h_ab : Line A B)
  (h_mn : Line M N)
  (h_pass_through : C ∈ h_ab ∧ C ∈ h_mn)
  (h_parallel : parallel (Line O₁ O₂) h_ab)
  (h_angle_alpha : angle_between (Line O₁ O₂) h_mn = α)
  (h_ab_length : dist A B = a) : 
  dist M N = a * cos α :=
sorry

end find_length_of_nm_l592_592005


namespace price_of_rice_packet_l592_592821

-- Definitions based on conditions
def initial_amount : ℕ := 500
def wheat_flour_price : ℕ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℕ := 150
def remaining_balance : ℕ := 235
def total_spending (P : ℕ) : ℕ := initial_amount - remaining_balance

-- Theorem to prove
theorem price_of_rice_packet (P : ℕ) (h: 2 * P + wheat_flour_quantity * wheat_flour_price + soda_price = total_spending P) : P = 20 :=
sorry

end price_of_rice_packet_l592_592821


namespace tetrahedron_volume_l592_592380

-- Definition of the required constants and variables
variables {S1 S2 S3 S4 r : ℝ}

-- The volume formula we need to prove
theorem tetrahedron_volume :
  (V = 1/3 * (S1 + S2 + S3 + S4) * r) :=
sorry

end tetrahedron_volume_l592_592380


namespace max_val_l592_592366

-- Define the conditions and parameters
variables (n : Fin 2004 → ℕ)
variable (sum_n : ∑ i in Finset.range 2004, (i * n i) = 2003)

-- Main theorem statement
theorem max_val (N : ℕ) (hN : N = ∑ i in Finset.range 2004, n i)
  (h_sum_n : ∑ i in Finset.range 2004, (i * n i) = 2003) :
  let expr := (2003 - N)
  in expr = 2001 :=
by
  sorry

end max_val_l592_592366


namespace shipping_cost_l592_592740

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

/-- Lizzy's total shipping cost for 540 pounds of fish packed in 30-pound crates at $1.5 per crate is $27. -/
theorem shipping_cost : (total_weight / weight_per_crate) * cost_per_crate = 27 := by
  sorry

end shipping_cost_l592_592740


namespace average_runs_in_second_set_l592_592381

theorem average_runs_in_second_set
  (avg_first_set : ℕ → ℕ → ℕ)
  (avg_all_matches : ℕ → ℕ → ℕ)
  (avg1 : ℕ := avg_first_set 20 30)
  (avg2 : ℕ := avg_all_matches 30 25) :
  ∃ (A : ℕ), A = 15 := by
  sorry

end average_runs_in_second_set_l592_592381


namespace average_speed_l592_592514

theorem average_speed :
  ∀ (initial_odometer final_odometer total_time : ℕ), 
    initial_odometer = 2332 →
    final_odometer = 2772 →
    total_time = 8 →
    (final_odometer - initial_odometer) / total_time = 55 :=
by
  intros initial_odometer final_odometer total_time h_initial h_final h_time
  sorry

end average_speed_l592_592514


namespace average_minutes_run_is_20_56_l592_592262

def sixth_graders_run (s : ℕ) : ℝ := 20 * 3 * s
def seventh_graders_run (s : ℕ) : ℝ := 25 * s
def eighth_graders_run (s : ℕ) : ℝ := 15 * (s / 2)

def total_minutes_run (s : ℕ) : ℝ :=
  sixth_graders_run s + seventh_graders_run s + eighth_graders_run s

def total_students (s : ℕ) : ℝ :=
  3 * s + s + (s / 2)

def average_minutes_run (s : ℕ) : ℝ :=
  total_minutes_run s / total_students s

theorem average_minutes_run_is_20_56 (s : ℕ) : average_minutes_run s = 20.56 := sorry

end average_minutes_run_is_20_56_l592_592262


namespace area_enclosed_by_curves_l592_592577

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := real.sqrt x

-- Define the integral expression for the area
noncomputable def area : ℝ :=
  ∫ x in 0..1, (curve2 x - curve1 x)

-- Theorem stating the area is 1/6
theorem area_enclosed_by_curves : area = 1 / 6 :=
by
  sorry

end area_enclosed_by_curves_l592_592577


namespace height_of_stack_of_pipes_l592_592003

-- Lean 4 statement
theorem height_of_stack_of_pipes :
  ∀ (n : ℕ) (d : ℝ), n = 3 → d = 12 → (h : ℝ), h = n * d → h = 36 :=
by
  intros n d h h_eq_n_mul_d hn hd
  sorry

end height_of_stack_of_pipes_l592_592003


namespace proposition1_proposition2_proposition3_proposition4_l592_592221

theorem proposition1 (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by sorry

theorem proposition2 (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (ac > bd) :=
by sorry

theorem proposition3 (a b : ℝ) (h1 : a > b) : ¬ (1 / (a - b) > 1 / a) :=
by sorry

theorem proposition4 (a b : ℝ) (h1 : 1 / a < 1 / b) (h2 : 1 / b < 0) : ab < b^2 :=
by sorry

example : (proposition1 ∧ proposition4) ∧ (proposition2 ∧ proposition3) :=
by
  constructor;
  {
    split;
    { exact proposition1, sorry };
    { exact proposition4, sorry }
  }; 
  {
    split;
    { exact proposition2, sorry };
    { exact proposition3, sorry }
  }

end proposition1_proposition2_proposition3_proposition4_l592_592221


namespace find_y_l592_592720

def star (a b : ℝ) : ℝ := (sqrt (a^2 + b)) / (sqrt (a^2 - b))

theorem find_y (y : ℝ) (h : star y 15 = 5) : y = sqrt 65 / 2 :=
by
  sorry

end find_y_l592_592720


namespace deborah_total_cost_l592_592559

-- Standard postage per letter
def stdPostage : ℝ := 1.08

-- Additional charge for international shipping per letter
def intlAdditional : ℝ := 0.14

-- Number of domestic and international letters
def numDomestic : ℕ := 2
def numInternational : ℕ := 2

-- Expected total cost for four letters
def expectedTotalCost : ℝ := 4.60

theorem deborah_total_cost :
  (numDomestic * stdPostage) + (numInternational * (stdPostage + intlAdditional)) = expectedTotalCost :=
by
  -- proof skipped
  sorry

end deborah_total_cost_l592_592559


namespace chairs_to_remove_l592_592484

-- Defining the conditions
def chairs_per_row : Nat := 15
def total_chairs : Nat := 180
def expected_attendees : Nat := 125

-- Main statement to prove
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ) : 
  chairs_per_row = 15 → 
  total_chairs = 180 → 
  expected_attendees = 125 → 
  ∃ n, total_chairs - (chairs_per_row * n) = 45 ∧ n * chairs_per_row ≥ expected_attendees := 
by
  intros h1 h2 h3
  sorry

end chairs_to_remove_l592_592484


namespace average_age_group_C_l592_592672

def total_age_class (number_of_students_class : ℕ) (average_age_class : ℕ) : ℕ :=
  number_of_students_class * average_age_class

def total_age_group (number_of_students_group : ℕ) (average_age_group : ℕ) : ℕ :=
  number_of_students_group * average_age_group

def average_age_group_C_before_changes (total_age_class : ℕ) (total_age_group_A : ℕ) (total_age_group_B : ℕ) (number_of_students_group_C : ℕ) : ℕ :=
  (total_age_class - (total_age_group_A + total_age_group_B)) / number_of_students_group_C

theorem average_age_group_C (number_of_students_class : ℕ) (average_age_class : ℕ)
(number_of_students_group_A : ℕ) (average_age_group_A : ℕ)
(number_of_students_group_B : ℕ) (average_age_group_B : ℕ) 
(number_of_students_group_C_after_2_move : ℕ) (average_age_group_C_after_2_move : ℕ) : 
average_age_group_C_before_changes (total_age_class number_of_students_class average_age_class)
(total_age_group number_of_students_group_A average_age_group_A)
(total_age_group number_of_students_group_B average_age_group_B)
(number_of_students_class - (number_of_students_group_A + number_of_students_group_B))
= 27.5 :=
by sorry

end average_age_group_C_l592_592672


namespace complete_square_transformation_l592_592776

theorem complete_square_transformation :
  ∀ x : ℝ, x^2 + 6 * x - 5 = 0 → (x + 3)^2 = 14 := 
by 
  intros x h,
  sorry

end complete_square_transformation_l592_592776


namespace piecewise_function_algorithm_structure_l592_592229

def piecewise_function (x : ℝ) : ℝ :=
  if x > 0 then log x else 2^x

theorem piecewise_function_algorithm_structure :
  ∃ sequential_structure selection_structure, True := 
by
  exists "Sequential structure"
  exists "Selection structure"
  trivial

end piecewise_function_algorithm_structure_l592_592229


namespace distance_from_circle_center_to_line_l592_592692

theorem distance_from_circle_center_to_line :
  let ρ := λ θ : ℝ, 4 * Real.cos θ,
      circle_center_x := 2,
      circle_center_y := 0,
      line := λ θ : ℝ, 2 * Real.sqrt 2 / Real.sin (θ + π / 4),
      distance (x1 y1 A B C : ℝ) : ℝ := (x1 * A + y1 * B + C).abs / (A^2 + B^2).sqrt
  in distance circle_center_x circle_center_y 1 1 -4 = Real.sqrt 2 :=
by
  sorry

end distance_from_circle_center_to_line_l592_592692


namespace sin_315_degree_l592_592954

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592954


namespace sequence_monotonic_and_bounded_l592_592107

theorem sequence_monotonic_and_bounded :
  ∀ (a : ℕ → ℝ), (a 1 = 1 / 2) → (∀ n, a (n + 1) = 1 / 2 + (a n)^2 / 2) →
    (∀ n, a n < 2) ∧ (∀ n, a n < a (n + 1)) :=
by
  sorry

end sequence_monotonic_and_bounded_l592_592107


namespace S_when_R_is_16_and_T_is_1_div_4_l592_592452

theorem S_when_R_is_16_and_T_is_1_div_4 :
  ∃ (S : ℝ), (∀ (R S T : ℝ) (c : ℝ), (R = c * S / T) →
  (2 = c * 8 / (1/2)) → c = 1 / 8) ∧
  (16 = (1/8) * S / (1/4)) → S = 32 :=
sorry

end S_when_R_is_16_and_T_is_1_div_4_l592_592452


namespace additional_cocoa_powder_needed_is_zero_l592_592807

-- Conditions
def recipe_ratio : ℝ := 0.4
def cake_weight : ℝ := 450
def cocoa_already_given : ℝ := 259

-- The question transformed into a proof statement
theorem additional_cocoa_powder_needed_is_zero :
  let required_cocoa_powder := cake_weight * recipe_ratio in
  let additional_cocoa_powder_needed := required_cocoa_powder - cocoa_already_given in
  additional_cocoa_powder_needed = 0 → cocoa_already_given - required_cocoa_powder = 79 :=
by
  sorry

end additional_cocoa_powder_needed_is_zero_l592_592807


namespace computer_price_increase_l592_592254

theorem computer_price_increase (c : ℕ) (h : 2 * c = 540) : c + (c * 30 / 100) = 351 :=
by
  sorry

end computer_price_increase_l592_592254


namespace lizzys_shipping_cost_l592_592743

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

def number_of_crates (W w : ℝ) : ℝ := W / w
def total_shipping_cost (c n : ℝ) : ℝ := c * n

theorem lizzys_shipping_cost :
  total_shipping_cost cost_per_crate (number_of_crates total_weight weight_per_crate) = 27 := 
by {
  sorry
}

end lizzys_shipping_cost_l592_592743


namespace find_a_extreme_values_and_c_range_l592_592223

-- Given function f(x) and its conditions
def f (x a : ℝ) : ℝ := x^3 + 3 * a * x^2 - 9 * x + 5
def f_prime (x a : ℝ) : ℝ := 3 * x^2 + 6 * a * x - 9

-- Main statement
theorem find_a_extreme_values_and_c_range (a c : ℝ) (h1 : f_prime 1 a = 0) :
  a = 1 ∧
  (f (-3) 1 = 32 ∧ f 1 1 = 0) ∧
  (∀ x : ℝ, x ∈ set.Icc (-4) 4 → f x 1 < c ^ 2 → (c > 9 ∨ c < -9)) :=
by
  sorry

end find_a_extreme_values_and_c_range_l592_592223


namespace knight_ship_speed_l592_592522

-- Define the variables and constants
variables (v_0 : ℝ) -- Speed of Knight ship in still water
constant (current_speed : ℝ) := 0.5 -- Speed of current in meters per second
constant (total_time : ℝ) := 7200 -- Total travel time in seconds
constant (meeting_time : ℝ) := 600 -- Time to meet in seconds

-- Define the effective speeds
def downstream_speed : ℝ := v_0 + current_speed
def upstream_speed : ℝ := v_0 - current_speed

-- Define the total distance equation
axiom distance_equation : 
  total_time = 2 * ((meeting_time / downstream_speed) + (meeting_time / upstream_speed)) + 2 * meeting_time

-- Theorem to prove the speed of the Knight ship
theorem knight_ship_speed : v_0 = 6 :=
by sorry

end knight_ship_speed_l592_592522


namespace system_of_equations_l592_592606

theorem system_of_equations (x y : ℝ) 
  (h1 : 2019 * x + 2020 * y = 2018) 
  (h2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 :=
by sorry

end system_of_equations_l592_592606


namespace fraction_addition_l592_592891

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l592_592891


namespace painter_can_turn_all_black_l592_592865

-- Definition of the board dimensions and initial conditions
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (color : ℕ × ℕ → bool) -- True for black, False for white
  (initial_checkerboard : ∀ i j, color (i, j) = (i + j) % 2 = 1)

-- Initial conditions
def board_2012_2013 := Board.mk 2012 2013 (λ (i j), (i + j) % 2 = 1) sorry

-- The main theorem
theorem painter_can_turn_all_black (b : Board) (h : b = board_2012_2013) 
  (start_at_corner : b.color (0, 0) = (0 + 0) % 2 = 1):
  ∃ (moves : list (ℕ × ℕ)), 
    (∀ (pos : ℕ × ℕ), pos ∈ moves →
      pos.fst < b.rows ∧ pos.snd < b.cols) ∧
    (∃ color_after_moves : ℕ × ℕ → bool,
      color_after_moves = (λ (i j), true)) :=
begin
  sorry
end

end painter_can_turn_all_black_l592_592865


namespace range_of_a_l592_592637

noncomputable def f (a x : ℝ) : ℝ := (Real.log (x^2 - a * x + 5)) / (Real.log a)

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha0 : 0 < a) (ha1 : a ≠ 1) 
  (hx₁x₂ : x₁ < x₂) (hx₂ : x₂ ≤ a / 2) 
  (hf : (f a x₂ - f a x₁ < 0)) : 
  1 < a ∧ a < 2 * Real.sqrt 5 := 
sorry

end range_of_a_l592_592637


namespace volume_space_inside_sphere_outside_cylinder_l592_592877

-- Definitions of given conditions
def base_radius (cylinder : Type) := 4
def sphere_radius (sphere : Type) := 7

-- The Lean statement of the proof problem
theorem volume_space_inside_sphere_outside_cylinder :
  let r_s := sphere_radius ()
  let r_c := base_radius ()
  let h := 2 * r_s
  let V_sphere := (4 / 3) * Real.pi * r_s^3
  let V_cylinder := Real.pi * r_c^2 * h
  let V_space := V_sphere - V_cylinder
  V_space = (700 / 3) * Real.pi :=
  sorry

end volume_space_inside_sphere_outside_cylinder_l592_592877


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592933

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592933


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592937

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592937


namespace remainder_three_ab_l592_592724

theorem remainder_three_ab (n : ℕ) (a b : ℤ) [invertible b] (h1 : a ≡ 2 * (⅟ b) [ZMOD n]) : (3 * a * b) ≡ 6 [ZMOD n] :=
by sorry

end remainder_three_ab_l592_592724


namespace sin_theta_value_l592_592250

theorem sin_theta_value (θ : ℝ) 
  (h1 : sin θ + cos θ = 7/5) 
  (h2 : tan θ < 1) : sin θ = 3/5 :=
sorry

end sin_theta_value_l592_592250


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592996

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592996


namespace max_value_quadratic_l592_592368

theorem max_value_quadratic (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : x * (1 - x) ≤ 1 / 4 :=
by
  sorry

end max_value_quadratic_l592_592368


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592997

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592997


namespace length_GH_is_29_l592_592686

-- Define the known lengths of the segments
def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

-- State the problem to prove that the length of segment GH is 29
theorem length_GH_is_29 (x : ℕ) : 
  (let side_length_fourth_square := x + length_FE in
   let side_length_third_square := side_length_fourth_square + length_CD in
   let side_length_second_square := side_length_third_square + length_AB in
   let side_length_first_square := x in
   side_length_second_square - side_length_first_square = 29
  ) :=
sorry

end length_GH_is_29_l592_592686


namespace parameter_values_two_solutions_l592_592575

theorem parameter_values_two_solutions :
  {a : ℝ |
    ∃! (x y : ℝ), 
    (x-a)^2 + y^2 = 64 ∧ 
    ((|x| - 8)^2 + (|y| - 15)^2 = 289)} = 
    ({-28} ∪ (Icc -24 -8) ∪ (Ico 8 24) ∪ {28}) :=
by sorry

end parameter_values_two_solutions_l592_592575


namespace polynomial_square_b_value_l592_592786

theorem polynomial_square_b_value (a b : ℚ) :
  (∃ (A B : ℚ), 2 * A = 1 ∧ A^2 + 2 * B = 2 ∧ 2 * A * B = a ∧ B^2 = b) → b = 49 / 64 :=
by
  intros,
  sorry

end polynomial_square_b_value_l592_592786


namespace nonagon_product_l592_592754

-- Assume conditions
def A1 := Point -- Assume point on unit circle
def A2 := rotate A1 40 -- 360/9 = 40 degree rotation
def A3 := rotate A1 80
def A4 := rotate A1 120
def A5 := rotate A1 160

noncomputable def sin_deg (deg : ℝ) : ℝ := Real.sin (deg * Real.pi / 180)

-- Define the lengths
def length_A1_A2 := 2 * sin_deg 20
def length_A1_A3 := 2 * sin_deg 40
def length_A1_A4 := 2 * sin_deg 60
def length_A1_A5 := 2 * sin_deg 80

-- The final proposition that needs to be proved
theorem nonagon_product :
  length_A1_A2 * length_A1_A3 * length_A1_A4 * length_A1_A5 = 3 :=
by sorry

end nonagon_product_l592_592754


namespace trapezoid_area_l592_592110

-- Definitions used in the problem conditions
def base1 (y : ℝ) : ℝ := 3 * y
def base2 (y : ℝ) : ℝ := 4 * y
def height (y : ℝ) : ℝ := y

-- The Lean statement for the proof problem
theorem trapezoid_area (y : ℝ) : 
  (1/2) * (base1 y + base2 y) * (height y) = 7 * y^2 / 2 :=
by
  sorry

end trapezoid_area_l592_592110


namespace card_S_l592_592726

def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 5 * n - 1

def S : Finset ℕ := 
  (Finset.range 2016).image a ∩ (Finset.range (a 2015 + 1)).image b

theorem card_S : S.card = 504 := 
  sorry

end card_S_l592_592726


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592026

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592026


namespace sasha_wins_2023_sasha_wins_2022_l592_592417

def game_initial_state : Type :=
  { N : ℕ } 

def sasha_turn : ℕ := 
  1 

def misha_options : List ℕ := 
  [1, 3, 5, 7, 9]

def vitalik_options : List ℕ := 
  [2, 4, 6, 8, 10]

theorem sasha_wins_2023 (N : ℕ) : N = 2023 ∧ 
  ( ∀ s, s = sasha_turn ) ∧ 
  ( ∀ m, m ∈ misha_options ) ∧ 
  ( ∀ v, v ∈ vitalik_options ) → 
  ∃ winner : String, winner = "Sasha" :=
sorry

theorem sasha_wins_2022 (N : ℕ) : N = 2022 ∧ 
  ( ∀ s, s = sasha_turn ) ∧ 
  ( ∀ m, m ∈ misha_options ) ∧ 
  ( ∀ v, v ∈ vitalik_options ) → 
  ∃ winner : String, winner = "Sasha" :=
sorry

end sasha_wins_2023_sasha_wins_2022_l592_592417


namespace sin_315_eq_neg_sqrt2_div_2_l592_592919

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592919


namespace min_value_l592_592195

-- Definition of the sequence {a_n}
def sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (a 1 = 2) ∧ (∀ n : ℕ, n > 0 → a (n + 1) - a n = 2 * n)

-- The minimum value proof
theorem min_value (a : ℕ → ℕ) (h: sequence a) : ∀ n : ℕ, n > 0 → (a n) / n ≥ 2 :=
by
  -- Proof follows from the problem conditions and solution.
  sorry

end min_value_l592_592195


namespace beetle_total_distance_l592_592861

-- Define positions as given in the conditions
def startPos : ℤ := 4
def pos1 : ℤ := -7
def pos2 : ℤ := 3
def pos3 : ℤ := -4

-- Calculate distances for each segment
def dist1 : ℤ := abs (pos1 - startPos)
def dist2 : ℤ := abs (pos2 - pos1)
def dist3 : ℤ := abs (pos3 - pos2)

-- Calculate the total distance
def totalDistance : ℤ := dist1 + dist2 + dist3

-- The theorem to state the total distance crawled by the beetle is 28 units
theorem beetle_total_distance : totalDistance = 28 := by
  unfold totalDistance dist1 dist2 dist3
  unfold startPos pos1 pos2 pos3
  simp
  exact dec_trivial

end beetle_total_distance_l592_592861


namespace product_invertibles_mod_120_l592_592312

theorem product_invertibles_mod_120 :
  let n := (list.filter (λ k, Nat.coprime k 120) (list.range 120)).prod
  in n % 120 = 119 :=
by
  sorry

end product_invertibles_mod_120_l592_592312


namespace triangle_unique_solution_l592_592251

theorem triangle_unique_solution (k : ℝ) (hABC : ∠ABC = 60) (hAC : AC = 12) (hBC : BC = k) :
  (0 < k ∧ k ≤ 12) ∨ (k = 8 * sqrt 3) → ∃! Δ : triangle, Δ.has_property :=
sorry

end triangle_unique_solution_l592_592251


namespace find_variance_l592_592621

noncomputable def variance {n : ℕ} (X : ℕ → ℝ) : ℝ :=
  let p := 0.5 in
  n * p * (1 - p)

theorem find_variance (X : ℕ → ℝ) (n : ℕ) (p : ℝ) (h1 : p = 0.5) (h2 : X = fun k => if k ≤ n then real.of_nat k else 0) (h3 : 16 = n * p) :
  variance X = 8 :=
by
  sorry

end find_variance_l592_592621


namespace line_intersects_x_axis_l592_592526

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end line_intersects_x_axis_l592_592526


namespace x_investment_amount_l592_592084

variable (X : ℝ)
variable (investment_y : ℝ := 15000)
variable (total_profit : ℝ := 1600)
variable (x_share : ℝ := 400)

theorem x_investment_amount :
  (total_profit - x_share) / investment_y = x_share / X → X = 5000 :=
by
  intro ratio
  have h1: 1200 / 15000 = 400 / 5000 :=
    by sorry
  have h2: X = 5000 :=
    by sorry
  exact h2

end x_investment_amount_l592_592084


namespace tile_arrangements_count_l592_592353

-- Define the main function to count the number of possible arrangements
def countTileArrangements : Nat :=
  let colors := 3  -- number of colors
  let positions := 4  -- number of positions
  let combinations := (Finset.card (Finset.choose 2 (Finset.range positions)))  -- choosing 2 out of 4 positions
  let permutations := 2.factorial  -- permuting the remaining 2 colors
  colors * (combinations * permutations)

theorem tile_arrangements_count :
  countTileArrangements = 36 :=
by
  sorry

end tile_arrangements_count_l592_592353


namespace number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l592_592373

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of rooms
def total_rooms : ℕ := 2

-- For the first question, define: each room must receive exact three people
def room_receives_three_people (n m : ℕ) : Prop :=
  n = 3 ∧ m = 3

-- For the second question, define: each room must receive at least one person
def room_receives_at_least_one_person (n m : ℕ) : Prop :=
  n ≥ 1 ∧ m ≥ 1

theorem number_of_distributions_room_receives_three_people :
  ∃ (ways : ℕ), ways = 20 :=
by
  sorry

theorem number_of_distributions_room_receives_at_least_one_person :
  ∃ (ways : ℕ), ways = 62 :=
by
  sorry

end number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l592_592373


namespace number_of_bags_needed_l592_592076

def cost_corn_seeds : ℕ := 50
def cost_fertilizers_pesticides : ℕ := 35
def cost_labor : ℕ := 15
def profit_percentage : ℝ := 0.10
def price_per_bag : ℝ := 11

theorem number_of_bags_needed (total_cost : ℕ) (total_revenue : ℝ) (num_bags : ℝ) :
  total_cost = cost_corn_seeds + cost_fertilizers_pesticides + cost_labor →
  total_revenue = ↑total_cost + (↑total_cost * profit_percentage) →
  num_bags = total_revenue / price_per_bag →
  num_bags = 10 := 
by
  sorry

end number_of_bags_needed_l592_592076


namespace closest_westbound_vehicles_in_150_mile_section_l592_592412

theorem closest_westbound_vehicles_in_150_mile_section
  (speed_eastbound : ℕ)
  (speed_westbound : ℕ)
  (vehicles_passed : ℕ)
  (time_interval_minutes : ℕ)
  (section_length_miles : ℕ)
  (vehicles_in_section : ℕ) :
  speed_eastbound = 60 →
  speed_westbound = 50 →
  vehicles_passed = 40 →
  time_interval_minutes = 10 →
  section_length_miles = 150 →
  vehicles_in_section = 300 :=
begin
  sorry
end

end closest_westbound_vehicles_in_150_mile_section_l592_592412


namespace wendi_chickens_l592_592433

theorem wendi_chickens : 
  let initial := 12 in 
  let after_few_days := 3 * initial - 8 in 
  let after_dog := initial + after_few_days - 2 in 
  let final_brought_home := 2 * (3 * after_dog - 10) in 
  initial + after_few_days - 2 + final_brought_home = 246 := 
by
  sorry

end wendi_chickens_l592_592433


namespace inequality_proof_l592_592344

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (habc : a * b * c = 1)

theorem inequality_proof :
  (a + 1 / b)^2 + (b + 1 / c)^2 + (c + 1 / a)^2 ≥ 3 * (a + b + c + 1) :=
by
  sorry

end inequality_proof_l592_592344


namespace sin_315_eq_neg_sqrt2_div_2_l592_592918

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592918


namespace one_of_integers_is_one_l592_592464

theorem one_of_integers_is_one 
  {k : ℕ} (hpos : 0 < k) 
  (a : Fin k → ℕ) (hpos_a : ∀ i, 1 ≤ a i) 
  (hdiv : (finset.univ.prod a) ∣ 
          (finset.univ.prod (λ i, 2^(a i - 1) + 1))) : 
  ∃ i, a i = 1 := 
sorry

end one_of_integers_is_one_l592_592464


namespace product_invertibles_mod_120_eq_1_l592_592328

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def product_of_invertibles_mod_n (n : ℕ) :=
  List.prod (List.filter (fun x => is_coprime x n) (List.range n))

theorem product_invertibles_mod_120_eq_1 :
  product_of_invertibles_mod_n 120 % 120 = 1 := 
sorry

end product_invertibles_mod_120_eq_1_l592_592328


namespace find_value_of_a_l592_592178

axiom a : ℤ
def A := {4, a^2}
def B := {a - 6, a + 1, 9}
def C := A ∩ B

theorem find_value_of_a (h : C = {9}) : a = -3 :=
by
  sorry

end find_value_of_a_l592_592178


namespace FD_eq_FM_l592_592812

open EuclideanGeometry

-- Define the points A, B, C, D, F, and M on a circle with specific conditions.
variable (A B C D F M : Point)

-- Assume the conditions:
axiom on_circle_A_B_C : ∀ (M : Point), Circle M → on_circle M A ∧ on_circle M B ∧ on_circle M C
axiom AB_eq_BC : |AB| = |BC|
axiom BCD_equilateral : equilateral B C D
axiom second_intersection_AD_F : second_intersection (Line_AD : Line A D) Circle M F

-- Theorem to prove that |FD| = |FM|
theorem FD_eq_FM :
  distance F D = distance F M := by
  sorry

end FD_eq_FM_l592_592812


namespace difference_of_sums_l592_592436

noncomputable def sum_of_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem difference_of_sums : 
  sum_of_first_n_even 2004 - sum_of_first_n_odd 2003 = 6017 := 
by sorry

end difference_of_sums_l592_592436


namespace broken_line_count_l592_592473

def num_right_moves : ℕ := 9
def num_up_moves : ℕ := 10
def total_moves : ℕ := num_right_moves + num_up_moves
def num_broken_lines : ℕ := Nat.choose total_moves num_right_moves

theorem broken_line_count : num_broken_lines = 92378 := by
  sorry

end broken_line_count_l592_592473


namespace max_classes_bound_l592_592816

theorem max_classes_bound (P : ℕ) (hP : P > 0) (m : ℕ) 
  (A : Fin m → Finset (Fin P))
  (h1 : ∀ i j : Fin m, i ≠ j → A i ≠ A j)
  (h2 : ∀ i j : Fin m, i ≠ j → (A i ∩ A j).nonempty) : 
  m ≤ 2^(P-1) := 
sorry

end max_classes_bound_l592_592816


namespace sum_of_squares_of_roots_l592_592164

theorem sum_of_squares_of_roots : 
  (∀ x, x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3) → (2^2 + 3^2 = 13) := 
by 
  assume h : ∀ x, x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3,
  have s1 : 2 = 2 := rfl,
  have s2 : 3 = 3 := rfl,
  have : (2^2 + 3^2 = 2*2 + 3*3) := rfl,
  have : (2*2 + 3*3 = 4 + 9) := rfl,
  show 4 + 9 = 13 from rfl,
  sorry

end sum_of_squares_of_roots_l592_592164


namespace maximize_angle_x_coordinate_l592_592271

def point (x y : ℝ) := (x, y : ℝ × ℝ)

def M : ℝ × ℝ := point (-1) 2
def N : ℝ × ℝ := point 1 4

theorem maximize_angle_x_coordinate :
  ∃ x0 : ℝ, ∀ P : ℝ × ℝ, P = point x0 0 → (angle_at_max P M N → x0 = 1) :=
sorry

end maximize_angle_x_coordinate_l592_592271


namespace product_invertible_integers_mod_120_eq_one_l592_592322

theorem product_invertible_integers_mod_120_eq_one :
  let n := ∏ i in (multiset.filter (λ x, Nat.coprime x 120) (multiset.range 120)), i
  in n % 120 = 1 := 
by
  sorry

end product_invertible_integers_mod_120_eq_one_l592_592322


namespace range_of_a_value_of_f_neg_half_l592_592194

-- Definition of conditions
def condition1 (a : ℝ) : Prop := (a^(1/2) ≤ 3)
def condition2 (a : ℝ) : Prop := (Real.log 3 / Real.log a ≤ 1/2)
def f (a : ℝ) (m x : ℝ) : ℝ := m * x^a + Real.log ((1 + x)^a) - a * Real.log (1 - x) - 2

-- Problem (1)
theorem range_of_a (a : ℝ) : condition1 a → condition2 a → (0 < a ∧ a < 1) ∨ a = 9 := 
sorry

-- Problem (2)
theorem value_of_f_neg_half (a : ℝ) (m : ℝ) (h : a = 9) : 
    f a m (1/2) = a → f a m (-1/2) = -13 := 
sorry

end range_of_a_value_of_f_neg_half_l592_592194


namespace Q_eq_N_l592_592647

def P : set ℝ := {y | y = x^2 + 1}
def Q : set ℝ := {y | ∃ x, y = x^2 + 1}
def R : set ℝ := {x | ∃ y, y = x^2 + 1}
def M : set (ℝ × ℝ) := {(x, y) | y = x^2 + 1}
def N : set ℝ := {x | x ≥ 1}

theorem Q_eq_N : Q = N := by
  sorry

end Q_eq_N_l592_592647


namespace original_sphere_radius_l592_592109

-- Define the given conditions
def r : ℝ := 4 * real.cbrt 2
def hemisphere_volume : ℝ := (2 / 3) * real.pi * r^3

-- Problem statement:  Given conditions and prove results
theorem original_sphere_radius :
  let R := real.cbrt 4 in
  let original_sphere_volume := (4 / 3) * real.pi * R^3 / 2 in 
  let new_sphere_radius := 2 * R in
  let new_sphere_volume := (4 / 3) * real.pi * new_sphere_radius^3 in
  hemisphere_volume = 2 * original_sphere_volume ∧
  R = real.cbrt 4 ∧
  new_sphere_volume = (64 / 3) * real.pi * real.cbrt 4 :=
by
  let R := real.cbrt 4
  let original_sphere_volume := (4 / 3) * real.pi * R^3
  let new_sphere_radius := 2 * R
  let new_sphere_volume := (4 / 3) * real.pi * new_sphere_radius^3
  sorry

end original_sphere_radius_l592_592109


namespace percentageErrorIs95_l592_592701

-- Define the original number N
variable (N : ℝ)

-- Define the correct result which is 2N
def correctResult : ℝ := 2 * N

-- Define the incorrect result which is N / 10
def incorrectResult : ℝ := N / 10

-- Define the error as the difference between the correct result and the incorrect result
def error : ℝ := correctResult N - incorrectResult N

-- Define the percentage of error
def percentageError : ℝ := (error N / correctResult N) * 100

-- The theorem stating the percentage of error is 95%
theorem percentageErrorIs95 : percentageError N = 95 := by 
  sorry

end percentageErrorIs95_l592_592701


namespace analytical_expression_sum_of_solutions_eq_2pi_div_3_l592_592227

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytical_expression :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (∃ x, f x = 2) ∧ 
  (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → |x₁ - x₂| = Real.pi / 2) :=
by
  -- Given conditions
  have A_pos : 2 > 0 := by linarith
  have omega_pos : 2 > 0 := by linarith

  -- Questions to answer
  have fc1 : ∀ x : ℝ, f x = 2 * Real.sin (2 * x + Real.pi / 6) := sorry
  have fc2 : ∃ x, f x = 2 := sorry
  have fc3 : ∀ x₁ x₂ : ℝ, f x₁ = f x₂ → |x₁ - x₂| = Real.pi / 2 := sorry

  exact ⟨fc1, fc2, fc3⟩

theorem sum_of_solutions_eq_2pi_div_3 :
  ∑ x in [(-Real.pi / 6), (5 * Real.pi / 6), (-Real.pi / 2), (Real.pi / 2)], x = 2 * Real.pi / 3 :=
by
  sorry

end analytical_expression_sum_of_solutions_eq_2pi_div_3_l592_592227


namespace trapezoid_perimeter_l592_592281

noncomputable def perimeter_trapezoid 
  (AB CD AD BC : ℝ) 
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : ℝ :=
AB + BC + CD + AD

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : perimeter_trapezoid AB CD AD BC h_AB_CD_parallel h_AD_perpendicular h_BC_perpendicular h_AB_eq h_CD_eq h_height = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end trapezoid_perimeter_l592_592281


namespace nth_number_in_sequence_40_l592_592554

def number_in_row (n : ℕ) : ℕ :=
  3 * n

def nth_number_in_sequence (k : ℕ) : ℕ :=
  let rec find_row (k : ℕ) (n : ℕ) (count : ℕ) : (ℕ × ℕ) :=
    if k <= count then (n, k - (count - n^3))
    else find_row k (n + 1) (count + (n + 1)^3)
  let (row, pos) := find_row k 1 1
  number_in_row row

theorem nth_number_in_sequence_40 : nth_number_in_sequence 40 = 12 := 
by
  sorry

end nth_number_in_sequence_40_l592_592554


namespace product_of_invertibles_mod_120_l592_592336

open Nat

theorem product_of_invertibles_mod_120 :
  let m := 120
  let invertibles := { x | x < m ∧ gcd x m = 1 }
  ∏ a in invertibles, a % m = 119 :=
by
  sorry

end product_of_invertibles_mod_120_l592_592336


namespace stddev_of_sample_data_is_sqrt_two_l592_592204

-- Define the sample data
def sample_data : List ℝ := [3, 5, 7, 4, 6]

-- Calculate the mean of the sample data
def mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length)

-- Define the variance calculation for the sample data
def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (λ x => (x - m)^2)).sum / data.length

-- Define the standard deviation calculation as the square root of the variance
def stddev (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

-- The theorem to prove
theorem stddev_of_sample_data_is_sqrt_two :
  stddev sample_data = Real.sqrt 2 :=
by
  sorry

end stddev_of_sample_data_is_sqrt_two_l592_592204


namespace fraction_addition_l592_592897

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l592_592897


namespace fraction_addition_l592_592890

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l592_592890


namespace sum_of_digits_l592_592060

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592060


namespace minimum_positive_announcements_l592_592524

theorem minimum_positive_announcements (x y : ℕ) (h : x * (x - 1) = 132) (positive_products negative_products : ℕ)
  (hp : positive_products = y * (y - 1)) (hn : negative_products = (x - y) * (x - y - 1)) 
  (h_sum : positive_products + negative_products = 132) : 
  y = 2 :=
by sorry

end minimum_positive_announcements_l592_592524


namespace calculate_wholesale_price_l592_592106

noncomputable def retail_price : ℝ := 108

noncomputable def selling_price (retail_price : ℝ) : ℝ := retail_price * 0.90

noncomputable def selling_price_alt (wholesale_price : ℝ) : ℝ := wholesale_price * 1.20

theorem calculate_wholesale_price (W : ℝ) (R : ℝ) (SP : ℝ)
  (hR : R = 108)
  (hSP1 : SP = selling_price R)
  (hSP2 : SP = selling_price_alt W) : W = 81 :=
by
  -- Proof omitted
  sorry

end calculate_wholesale_price_l592_592106


namespace range_quadratic_l592_592791

-- Define the function and the interval
def f (x: ℝ) := x^2 - 2 * x - 3

-- Prove the range of the function over the interval [-2, 2]
theorem range_quadratic :
  set.range (f ∘ (set.Icc (-2: ℝ) 2)) = set.Icc (-4: ℝ) 5 :=
sorry

end range_quadratic_l592_592791


namespace radius_of_tangent_sphere_l592_592881

theorem radius_of_tangent_sphere (r1 r2 : ℝ) (h1 : r1 = 25) (h2 : r2 = 7) : 
  ∃ r : ℝ, r = 5 * Real.sqrt 7 := 
by
  use 5 * Real.sqrt 7
  have r_eq : 5 * Real.sqrt 7 = 5 * Real.sqrt 7 := rfl
  exact r_eq
sorry

end radius_of_tangent_sphere_l592_592881


namespace product_invertible_integers_mod_120_eq_one_l592_592321

theorem product_invertible_integers_mod_120_eq_one :
  let n := ∏ i in (multiset.filter (λ x, Nat.coprime x 120) (multiset.range 120)), i
  in n % 120 = 1 := 
by
  sorry

end product_invertible_integers_mod_120_eq_one_l592_592321


namespace geo_seq_decreasing_l592_592615

variables (a_1 q : ℝ) (a : ℕ → ℝ)
-- Define the geometric sequence
def geo_seq (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q ^ n

-- The problem statement as a Lean theorem
theorem geo_seq_decreasing (h1 : a_1 * (q - 1) < 0) (h2 : q > 0) :
  ∀ n : ℕ, geo_seq a_1 q (n + 1) < geo_seq a_1 q n :=
by
  sorry

end geo_seq_decreasing_l592_592615


namespace one_coin_tails_l592_592671

theorem one_coin_tails (n : ℕ) :
  (∃ k : ℕ, k < 2 * n + 1 ∧ ∀ j : ℕ, j < 2 * n + 1 → j ≠ k → coin_state j = heads) ∧
  (coin_state k = tails) :=
by sorry

end one_coin_tails_l592_592671


namespace problem_bounds_l592_592640

noncomputable def f (a x : ℝ) := (2 - x) * exp(x) + a * (x - 1) ^ 2

theorem problem_bounds (a : ℝ) : (∀ x : ℝ, f a x ≤ 2 * exp(x)) ↔ a ≤ (1 - Real.sqrt 2) * exp(1 - Real.sqrt 2) / 2 :=
by sorry

end problem_bounds_l592_592640


namespace simplify_expression_l592_592556

variable (x y : ℝ)

theorem simplify_expression (A B : ℝ) (hA : A = x^2) (hB : B = y^2) :
  (A + B) / (A - B) + (A - B) / (A + B) = 2 * (x^4 + y^4) / (x^4 - y^4) :=
by {
  sorry
}

end simplify_expression_l592_592556


namespace expected_value_of_winnings_is_5_l592_592079

namespace DiceGame

def sides : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 2 * roll else 0

noncomputable def expectedValue : ℚ :=
  (winnings 2 + winnings 4 + winnings 6 + winnings 8) / 8

theorem expected_value_of_winnings_is_5 :
  expectedValue = 5 := by
  sorry

end DiceGame

end expected_value_of_winnings_is_5_l592_592079


namespace impossible_to_cover_square_l592_592715

theorem impossible_to_cover_square (S : ℝ) (small_squares : ℝ) (num_small_squares : ℕ) 
  (hS : S = 7) (h_small_squares : small_squares = 3) (h_num_small_squares : num_small_squares = 8) : 
  ¬(∃ (positions : fin num_small_squares → (ℝ × ℝ)), ∀ i j : fin num_small_squares, 
    i ≠ j → (positions i).1 + small_squares ≤ (positions j).1 ∨ 
             (positions j).1 + small_squares ≤ (positions i).1 ∨ 
             (positions i).2 + small_squares ≤ (positions j).2 ∨ 
             (positions j).2 + small_squares ≤ (positions i).2) :=
by
  sorry

end impossible_to_cover_square_l592_592715


namespace sin_315_equals_minus_sqrt2_div_2_l592_592977

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592977


namespace sum_of_digits_l592_592059

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592059


namespace remainder_of_power_l592_592439

variables {R : Type*} [CommRing R] (x : R)

theorem remainder_of_power :
  let poly := (x + 2) ^ 1004,
      divisor := x^2 - x + 1,
      remainder := -x
  in
  ∃ q : R, poly = q * divisor + remainder :=
by
  sorry

end remainder_of_power_l592_592439


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592017

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592017


namespace mixtilinear_excircle_excenter_l592_592122

variables (A B C D E T S M P Q : Point)
variables (ω_ABC ω_TDE ω_SBC ω_BD ω_CE : Circle)

noncomputable def is_excenter (A B C Q : Point) : Prop := sorry

theorem mixtilinear_excircle_excenter 
  (h₁ : touches (A_mix_excircle A) (ω_ABC) AB AC T D E)
  (h₂ : S = intersection (line AT) (ω_TDE))
  (h₃ : M = intersection_two_circles ω_SBC ω_TDE)
  (h₄ : P = intersection_two_circles (ω_BD M) (ω_CE M))
  (h₅ : Q = symmetric_point P T) :
  is_excenter A B C Q :=
sorry

end mixtilinear_excircle_excenter_l592_592122


namespace exists_sequences_satisfying_conditions_l592_592085

noncomputable def satisfies_conditions (n : ℕ) (hn : Odd n) 
  (a : Fin n → ℕ) (b : Fin n → ℕ) : Prop :=
  ∀ (k : Fin n), 0 < k.val → k.val < n →
    ∀ (i : Fin n),
      let in3n := 3 * n;
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n ≠
      (a i + b i) % in3n ∧
      (a i + b i) % in3n ≠
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ∧
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ≠
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n

theorem exists_sequences_satisfying_conditions :
  ∀ n : ℕ, Odd n → ∃ (a : Fin n → ℕ) (b : Fin n → ℕ),
    satisfies_conditions n sorry a b :=
sorry

end exists_sequences_satisfying_conditions_l592_592085


namespace largest_prime_factor_l592_592585

theorem largest_prime_factor (a b c : ℕ) (h₁ : a = 20) (h₂ : b = 15) (h₃ : c = 10) :
  let expr := a^4 + b^4 - c^5 in
  ∃ p : ℕ, nat.prime p ∧ p >= 2 ∧ (p dvd expr) ∧ ∀ q : ℕ, q ≠ p → nat.prime q → q dvd expr → q ≤ p :=
sorry

end largest_prime_factor_l592_592585


namespace sin_315_equals_minus_sqrt2_div_2_l592_592984

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592984


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592948

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592948


namespace find_cost_price_of_peaches_find_a_l592_592145

-- Define the parameters and variables used in the problem
variables (x a : ℝ)

-- Define the constants provided in the problem 
def peaches_sold_weight1 : ℝ := 100
def apples_sold_weight : ℝ := 50
def peaches_selling_price1 : ℝ := 16
def apples_selling_price : ℝ := 20
def profit1 : ℝ := 1800

def peaches_sold_weight2 : ℝ := 300
def peaches_selling_price2_day1 : ℝ := 17
def peaches_remaining_weight : ℝ := 20
def profit2 : ℝ := 2980

-- Define conditions derived from the problem statement
def apples_cost_price : ℝ := 1.2 * x
def profit_eqn1 : Prop := (peaches_selling_price1 - x) * peaches_sold_weight1 + (apples_selling_price - apples_cost_price) * apples_sold_weight = profit1
def final_price_day2 : ℝ := 16 - 0.1 * a

-- Translate this to a proof problem
theorem find_cost_price_of_peaches (h1 : profit_eqn1) :
  x = 5 :=
sorry

-- Define equation for peaches on the second day
def profit_eqn2 : Prop :=
  (peaches_selling_price2_day1 * (8 * a)) + (final_price_day2 * (peaches_sold_weight2 - (8 * a) - peaches_remaining_weight)) - (x * peaches_sold_weight2) = profit2

-- Translate this into a proof problem
theorem find_a (h2 : profit_eqn2) :
  a = 25 :=
sorry

end find_cost_price_of_peaches_find_a_l592_592145


namespace total_worth_of_items_total_worth_of_items_final_l592_592294

theorem total_worth_of_items (F : ℝ) (tax_free_amount tax_rate total_worth taxable_amount : ℝ) 
  (hF : F = 39.7) 
  (hTaxRate : tax_rate = 0.06) 
  (hTaxFreeAmount : tax_free_amount = 0.30 * (taxable_amount + F)) :
  total_worth = taxable_amount + F := 
by
  sorry

# Here we are proving that the total worth is the sum of taxable and tax-free items
# under the given conditions. 

variables {F : ℝ} {taxable_amount : ℝ} {total_worth : ℝ}

def calc_taxable_amount (F taxable_sales_tax : ℝ) : ℝ :=
  taxable_sales_tax * 39.7 * (1/taxable_sales_tax - 0.3)

theorem total_worth_of_items_final (hF : F = 39.7) 
  (tax_free_amount : ℝ) (tax_rate : ℝ)
  (h_tax_rate_eq : tax_rate = 0.06)
  (h_tax_free_amount : tax_free_amount = 0.30 * (49.625 + F)) :
  total_worth = 49.625 + 39.7 :=
by
  sorry

# We start by setting up the base values and aim to show that given 
# these conditions, the total worth sums up to 89.325.

end total_worth_of_items_total_worth_of_items_final_l592_592294


namespace find_m_l592_592231

variables {a b : EuclideanSpace ℝ (Fin 2)}
variables (m : ℝ)
variables [NormedAddCommGroup a] [NormedAddCommGroup b]

noncomputable def vector_a : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def vector_b : EuclideanSpace ℝ (Fin 2) := sorry

axiom norm_vector_a : ‖vector_a‖ = sqrt 3
axiom norm_vector_b : ‖vector_b‖ = 2
axiom angle_between_a_b : real.angle θ := real.pi / 6
axiom perpendicular_condition : inner (vector_a - m • vector_b) vector_a = 0

theorem find_m : m = 1 :=
sorry

end find_m_l592_592231


namespace paul_money_duration_l592_592465

theorem paul_money_duration
  (mow_earnings : ℕ)
  (weed_earnings : ℕ)
  (weekly_expenses : ℕ)
  (earnings_mow : mow_earnings = 3)
  (earnings_weed : weed_earnings = 3)
  (expenses : weekly_expenses = 3) :
  (mow_earnings + weed_earnings) / weekly_expenses = 2 := 
by
  sorry

end paul_money_duration_l592_592465


namespace tangent_line_slope_l592_592666

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 3 * x - 1

theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, f(x₀) = k * x₀ + 2 ∧ deriv f x₀ = k) → k = 2 := by
  sorry

end tangent_line_slope_l592_592666


namespace lesser_fraction_solution_l592_592409

noncomputable def lesser_fraction (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) : ℚ :=
  if x ≤ y then x else y

theorem lesser_fraction_solution (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) :
  lesser_fraction x y h₁ h₂ = (7 - Real.sqrt 17) / 16 := by
  sorry

end lesser_fraction_solution_l592_592409


namespace general_term_formula_l592_592612

noncomputable def xSeq : ℕ → ℝ
| 0       => 3
| (n + 1) => (xSeq n)^2 + 2 / (2 * (xSeq n) - 1)

theorem general_term_formula (n : ℕ) : 
  xSeq n = (2 * 2^2^n + 1) / (2^2^n - 1) := 
sorry

end general_term_formula_l592_592612


namespace sum_of_digits_of_expression_l592_592056

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592056


namespace value_of_a_l592_592218

theorem value_of_a (a : ℝ) :
  ((abs ((1) - (2) + a)) = 1) ↔ (a = 0 ∨ a = 2) :=
by
  sorry

end value_of_a_l592_592218


namespace checkerboard_square_count_l592_592855

/-- 
Prove that the total number of distinct squares with sides on the grid lines of a 
9 by 9 checkerboard and containing at least 6 black squares, can be drawn on the checkerboard.
-/
theorem checkerboard_square_count : 
  ∀ (board : ℕ) [hboard : board = 9], 
  (∑ (n : ℕ) in {4, 5, 6, 7, 8, 9}, (board - n + 1) * (board - n + 1)) = 91 :=
begin
  intros,
  -- This part contains the proof details, skipped for brevity
  sorry
end

end checkerboard_square_count_l592_592855


namespace sum_of_digits_l592_592064

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592064


namespace sum_of_middle_three_l592_592363

def is_valid_permutation (reds blues : List ℕ) : Prop :=
  reds.length = 4 ∧
  blues.length = 5 ∧
  ∀ i, (i < 4 ∧ reds.nth i ≠ none ∧ blues.nth (i + 1) ≠ none ∧ reds.nth_le i (by simp [*]) = 1) → 
  ∃ j, (reds.nth_le i (by simp [*]) ∣ blues.nth_le j (by simp [*]))

def card_sequence := [2, 1, 4, 2, 6, 4, 5, 3, 3]

def middle_three_sum (cards : List ℕ) : ℕ :=
  cards.nth_le 3 (by simp) + cards.nth_le 4 (by simp) + cards.nth_le 5 (by simp)

theorem sum_of_middle_three : middle_three_sum card_sequence = 12 :=
by
  sorry

end sum_of_middle_three_l592_592363


namespace arccos_neg_one_eq_pi_l592_592547

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l592_592547


namespace product_of_first_three_terms_of_arithmetic_sequence_l592_592797

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l592_592797


namespace expected_turns_l592_592485

theorem expected_turns (n : ℕ) : 
  let comb := (λ (a b : ℕ), Nat.binomial a b) in
  let expectation := n + 1/2 - (n - 1/2) * comb (2 * n - 2) (n - 1) / 2^(2 * n - 2) in
  E = expectation :=
sorry

end expected_turns_l592_592485


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592935

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592935


namespace distance_between_joe_and_gracie_l592_592738

open Complex

noncomputable def joe_point : ℂ := 2 + 3 * I
noncomputable def gracie_point : ℂ := -2 + 2 * I
noncomputable def distance := abs (joe_point - gracie_point)

theorem distance_between_joe_and_gracie :
  distance = Real.sqrt 17 := by
  sorry

end distance_between_joe_and_gracie_l592_592738


namespace triangle_inequality_l592_592698

theorem triangle_inequality
  (A B C M : Type)
  [add_comm_group A] [vector_space ℝ A]
  [add_comm_group B] [vector_space ℝ B]
  [add_comm_group C] [vector_space ℝ C]
  [add_comm_group M] [vector_space ℝ M]
  (S : ℝ)
  (area_ABC : ℝ)
  (AM BM CM BC AC AB : ℝ)
  (h_area : S = area_ABC)
  (h_AM: AM = dist A M)
  (h_BM: BM = dist B M)
  (h_CM: CM = dist C M)
  (h_BC: BC = dist B C)
  (h_AC: AC = dist A C)
  (h_AB: AB = dist A B) :
  4 * S ≤ AM * BC + BM * AC + CM * AB := by
  sorry

end triangle_inequality_l592_592698


namespace sum_k_over_k_plus_one_binom_eq_expression_sum_neg_one_pow_k_minus_one_k_over_k_plus_one_binom_eq_expression_l592_592755

-- First statement: sum (k / (k + 1) * binomial) equals the expression
theorem sum_k_over_k_plus_one_binom_eq_expression (n : ℕ) :
  ∑ k in Finset.range(n+1), (if k = 0 then 0 else (k : ℚ) / (k + 1) * Nat.choose n k) = 
  (↑(n-1) * 2^n + 1) / (n + 1) := by
  sorry

-- Second statement: sum ((-1)^(k-1) * k / (k + 1) * binomial) equals the expression
theorem sum_neg_one_pow_k_minus_one_k_over_k_plus_one_binom_eq_expression (n : ℕ) :
  ∑ k in Finset.range(n+1), (if k = 0 then 0 else ((-1)^(k-1) : ℚ) * (k : ℚ) / (k + 1) * Nat.choose n k) = 
  1 / (n + 1) := by
  sorry

end sum_k_over_k_plus_one_binom_eq_expression_sum_neg_one_pow_k_minus_one_k_over_k_plus_one_binom_eq_expression_l592_592755


namespace solve_inequality_l592_592803

theorem solve_inequality (x : ℝ) (h : x / 3 - 2 < 0) : x < 6 :=
sorry

end solve_inequality_l592_592803


namespace product_of_coprimes_mod_120_l592_592320

open Nat

noncomputable def factorial_5 : ℕ := 5!

def is_coprime_to_120 (x : ℕ) : Prop := gcd x 120 = 1

def coprimes_less_than_120 : List ℕ :=
  (List.range 120).filter is_coprime_to_120

def product_of_coprimes_less_than_120 : ℕ :=
  coprimes_less_than_120.foldl (*) 1

theorem product_of_coprimes_mod_120 : 
  (product_of_coprimes_less_than_120 % 120) = 1 :=
sorry

end product_of_coprimes_mod_120_l592_592320


namespace find_x_l592_592616

theorem find_x 
  (x : ℝ) 
  (h1 : ∃ θ : ℝ, P (x, 3) ∧ cos θ = -4/5) : 
  x = -4 := sorry

end find_x_l592_592616


namespace correct_product_l592_592739

theorem correct_product (a b c : ℕ) (ha : 10 * c + 1 = a) (hb : 10 * c + 7 = a) 
(hl : (10 * c + 1) * b = 255) (hw : (10 * c + 7 + 6) * b = 335) : 
  a * b = 285 := 
  sorry

end correct_product_l592_592739


namespace sin_cos_product_l592_592207

-- Condition: cos α - sin α = 1/2
-- Conclusion to prove: sin α cos α = 3/8
theorem sin_cos_product (α : Real) (h : cos α - sin α = 1 / 2) : sin α * cos α = 3 / 8 :=
sorry

end sin_cos_product_l592_592207


namespace sufficient_but_not_necessary_condition_circle_l592_592563

theorem sufficient_but_not_necessary_condition_circle {a : ℝ} (h : a = 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 → (∀ a, a < 2 → (x - 1)^2 + (y + 1)^2 = 2 - a) :=
by
  sorry

end sufficient_but_not_necessary_condition_circle_l592_592563


namespace speed_of_current_l592_592081

/-- The given conditions of the problem. -/
def upstream_time := 25 / 60 -- time in hours
def downstream_time := 12 / 60 -- time in hours
def distance := 1 -- distance in km

/-- Speed calculations based on the conditions. -/
def speed_upstream := distance / upstream_time
def speed_downstream := distance / downstream_time

/-- Let B be the speed of the boat in still water, and C be the speed of the current. -/
def B := (speed_upstream + speed_downstream) / 2
def C := (speed_downstream - speed_upstream) / 2

/-- The goal is to show that the speed of the current C is 1.3 km/h. -/
theorem speed_of_current : C = 1.3 := by
  compute speed_upstream and speed_downstream first
  have h1 : speed_upstream = 2.4 := by sorry
  have h2 : speed_downstream = 5 := by sorry
  -- Now, considering B and C as defined
  have h3 : B = 3.7 := by sorry
  show C = 1.3 from sorry

end speed_of_current_l592_592081


namespace sum_of_digits_l592_592058

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592058


namespace line_passes_through_parabola_vertex_l592_592596

theorem line_passes_through_parabola_vertex : 
  ∃ (c : ℝ), (∀ (x : ℝ), y = 2 * x + c → ∃ (x0 : ℝ), (x0 = 0 ∧ y = c^2)) ∧ 
  (∀ (c1 c2 : ℝ), (y = 2 * x + c1 ∧ y = 2 * x + c2 → c1 = c2)) → 
  ∃ c : ℝ, c = 0 ∨ c = 1 :=
by 
  -- Proof should be inserted here
  sorry

end line_passes_through_parabola_vertex_l592_592596


namespace product_of_coprimes_mod_120_l592_592315

open Nat

noncomputable def factorial_5 : ℕ := 5!

def is_coprime_to_120 (x : ℕ) : Prop := gcd x 120 = 1

def coprimes_less_than_120 : List ℕ :=
  (List.range 120).filter is_coprime_to_120

def product_of_coprimes_less_than_120 : ℕ :=
  coprimes_less_than_120.foldl (*) 1

theorem product_of_coprimes_mod_120 : 
  (product_of_coprimes_less_than_120 % 120) = 1 :=
sorry

end product_of_coprimes_mod_120_l592_592315


namespace length_BC_of_circle_l592_592355

theorem length_BC_of_circle (r : ℝ) (AB lenOC lenCD : ℝ) (A B C D : Point)
  (AB : line_segment A B) (OC : line_segment O C) (OD : line_segment O D) (CD : line_segment C D)
  (radius_A : distance O A = r)
  (radius_B : distance O B = r)
  (AB_eq : distance A B = AB)
  (C_midpoint_minor_arc : is_midpoint_minor_arc C A B O)
  (lenOC_def : distance O C = lenOC)
  (lenCD_def : distance C D = lenCD) :
  distance B C = 14 - 2 * sqrt 33 :=
by
  sorry

end length_BC_of_circle_l592_592355


namespace brooke_butter_price_l592_592534

variables (price_per_gallon_of_milk : ℝ)
variables (gallons_to_butter_conversion : ℝ)
variables (number_of_cows : ℕ)
variables (milk_per_cow : ℝ)
variables (number_of_customers : ℕ)
variables (milk_demand_per_customer : ℝ)
variables (total_earnings : ℝ)

theorem brooke_butter_price :
    price_per_gallon_of_milk = 3 →
    gallons_to_butter_conversion = 2 →
    number_of_cows = 12 →
    milk_per_cow = 4 →
    number_of_customers = 6 →
    milk_demand_per_customer = 6 →
    total_earnings = 144 →
    (total_earnings - number_of_customers * milk_demand_per_customer * price_per_gallon_of_milk) /
    (number_of_cows * milk_per_cow - number_of_customers * milk_demand_per_customer) *
    gallons_to_butter_conversion = 1.50 :=
by { sorry }

end brooke_butter_price_l592_592534


namespace prob_below_8_correct_l592_592507

-- Defining the probabilities of hitting the 10, 9, and 8 rings
def prob_10 : ℝ := 0.20
def prob_9 : ℝ := 0.30
def prob_8 : ℝ := 0.10

-- Defining the event of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10 + prob_9 + prob_8)

-- The main theorem to prove: the probability of scoring below 8 is 0.40
theorem prob_below_8_correct : prob_below_8 = 0.40 :=
by 
  -- We need to show this proof in a separate proof phase
  sorry

end prob_below_8_correct_l592_592507


namespace cot_sum_arccot_roots_l592_592340

noncomputable def P (z : ℂ) : ℂ := z^25 - 5 * z^24 + 14 * z^23 - 30 * z^22 + ... + 676

-- Define the 25 roots of the polynomial
variables {z : ℕ → ℂ} (h_roots : ∀ k, 1 ≤ k ∧ k ≤ 25 → P (z k) = 0)

theorem cot_sum_arccot_roots :
  cot (∑ k in finset.range 25, arccot (z k)) = result :=
by sorry

end cot_sum_arccot_roots_l592_592340


namespace integral_sqrt_9_minus_x_squared_l592_592572

noncomputable def evaluate_integral : ℝ :=
  ∫ x in -3..3, real.sqrt (9 - x^2)

theorem integral_sqrt_9_minus_x_squared :
  evaluate_integral = (9 * real.pi) / 2 :=
sorry

end integral_sqrt_9_minus_x_squared_l592_592572


namespace surface_area_of_circumscribed_sphere_l592_592430

theorem surface_area_of_circumscribed_sphere
  (PM NR PN MR MN PR : ℝ)
  (hPM : PM = sqrt 10)
  (hNR : NR = sqrt 10)
  (hPN : PN = sqrt 13)
  (hMR : MR = sqrt 13)
  (hMN : MN = sqrt 5)
  (hPR : PR = sqrt 5) :
  let r := sqrt 14 / 2 in
  4 * π * r^2 = 14 * π := 
by
  sorry

end surface_area_of_circumscribed_sphere_l592_592430


namespace angle_DAE_20_degrees_l592_592282

open EuclideanGeometry

noncomputable def triangle_data (A B C D O E : Point) :=
  (angle A C B = 40) ∧ (angle C B A = 60) ∧
  (perpendicular D A (line_through B C)) ∧
  (is_circumcenter O A B C) ∧
  (is_diameter_end E A O)

theorem angle_DAE_20_degrees (A B C D O E : Point) 
  (h : triangle_data A B C D O E) : angle D A E = 20 := 
sorry

end angle_DAE_20_degrees_l592_592282


namespace angle_QRS_l592_592273

-- Given definitions
def Triangle (a b c : Type) := a = b ∧ b = c ∧ c = a
def is_equilateral (a b c : Type) := Triangle a b c
def is_isosceles (a b c : Type) := a = b ∧ a = c

variables {P Q R S : Type}

-- Conditions
axiom PQS_equilateral : is_equilateral PQ PS QS
axiom PQR_isosceles : is_isosceles PQ PR PS
axiom PSR_isosceles : is_isosceles PS PR PS
axiom angle_equality : ∀ {x}, x = angle RPQ ↔ x = angle RPS 
axiom angle_sum_at_point : angle RPQ + angle RPS + angle QPS = 360

-- Proof goal
theorem angle_QRS :
  angle QRS = 30 := sorry

end angle_QRS_l592_592273


namespace prob_exactly_two_consecutive_l592_592600

theorem prob_exactly_two_consecutive 
  (balls : Finset ℕ) 
  (numbered : balls = {1, 2, 3, 4, 5, 6}) 
  (total_events : ℕ := Nat.choose 6 3)
  (consec_events : ℕ := 12) :
  let n := total_events,
      m := consec_events,
      p := m / n in
  p = (3 : ℚ) / 5 :=
by
  sorry

end prob_exactly_two_consecutive_l592_592600


namespace sin_315_eq_neg_sqrt2_div_2_l592_592966

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592966


namespace sum_of_interior_angles_heptagon_l592_592806

theorem sum_of_interior_angles_heptagon (n : ℕ) (h : n = 7) : (n - 2) * 180 = 900 := by
  sorry

end sum_of_interior_angles_heptagon_l592_592806


namespace arccos_neg_one_l592_592549

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l592_592549


namespace sin_315_eq_neg_sqrt2_div_2_l592_592913

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592913


namespace cos_48_equals_sqrt3_minus1_div2_l592_592130

noncomputable def cos24 : ℝ := (sqrt (2 + sqrt 3)) / 2
noncomputable def cos48 : ℝ := (sqrt 3 - 1) / 2

theorem cos_48_equals_sqrt3_minus1_div2 :
  cos 48 = (sqrt 3 - 1) / 2 :=
by
  let x := cos 24
  have h1 : cos 24 = cos24,
    sorry
  have h2 : cos 48 = 2 * x ^ 2 - 1,
    sorry
  have h3 : cos 72 = 4 * x ^ 3 - 3 * x,
    sorry
  have h4 : cos 72 = cos 72,
    sorry
  have h5 : cos 48 = (sqrt 3 - 1) / 2,
    sorry
  exact h5

end cos_48_equals_sqrt3_minus1_div2_l592_592130


namespace line_intersects_x_axis_l592_592527

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end line_intersects_x_axis_l592_592527


namespace first_player_win_condition_l592_592002

def player_one_wins (p q : ℕ) : Prop :=
  p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 4 ∨
  q % 5 = 0 ∨ q % 5 = 1 ∨ q % 5 = 4

theorem first_player_win_condition (p q : ℕ) :
  player_one_wins p q ↔
  (∃ (a b : ℕ), (a, b) = (p, q) ∧ (a % 5 = 0 ∨ a % 5 = 1 ∨ a % 5 = 4 ∨ 
                                     b % 5 = 0 ∨ b % 5 = 1 ∨ b % 5 = 4)) :=
sorry

end first_player_win_condition_l592_592002


namespace sum_proper_divisors_600_correct_l592_592067

-- Define the prime factorization of 600.
def prime_factors_600 : list (ℕ × ℕ) := [(2, 3), (3, 1), (5, 2)]

-- Define the sigma function based on the prime factorization
def sigma_600 : ℕ := (1 + 2 + 2^2 + 2^3) * (1 + 3) * (1 + 5 + 5^2)

-- From the provided summation, sigma(600) = 1860.
def sum_divisors_600 : ℕ := 1860

-- Define the condition which computes the sum of proper divisors
def sum_proper_divisors_600 : ℕ := sum_divisors_600 - 600

-- Prove that the sum of the proper divisors of 600 is 1260
theorem sum_proper_divisors_600_correct : sum_proper_divisors_600 = 1260 := by
  sorry

end sum_proper_divisors_600_correct_l592_592067


namespace bunnies_burrow_exit_counts_l592_592505

theorem bunnies_burrow_exit_counts :
  let groupA_bunnies := 40
  let groupA_rate := 3  -- times per minute per bunny
  let groupB_bunnies := 30
  let groupB_rate := 5 / 2 -- times per minute per bunny
  let groupC_bunnies := 30
  let groupC_rate := 8 / 5 -- times per minute per bunny
  let total_bunnies := 100
  let minutes_per_day := 1440
  let days_per_week := 7
  let pre_change_rate_per_min := groupA_bunnies * groupA_rate + groupB_bunnies * groupB_rate + groupC_bunnies * groupC_rate
  let post_change_rate_per_min := pre_change_rate_per_min * 0.5
  let total_pre_change_counts := pre_change_rate_per_min * minutes_per_day * days_per_week
  let total_post_change_counts := post_change_rate_per_min * minutes_per_day * (days_per_week * 2)
  total_pre_change_counts + total_post_change_counts = 4897920 := by
    sorry

end bunnies_burrow_exit_counts_l592_592505


namespace segment_GH_length_l592_592688

-- Define the lengths of the segments as given conditions
def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

-- Define the length of segment GH to be proved
def length_GH : ℕ := length_AB + length_CD + length_FE

-- The theorem that needs to be proved
theorem segment_GH_length : length_GH = 29 := by
  -- We can use the definitions of length_AB, length_FE, and length_CD directly here
  unfold length_GH
  rw [length_AB, length_CD, length_FE]
  norm_num
  sorry

end segment_GH_length_l592_592688


namespace product_invertible_integers_mod_120_eq_one_l592_592326

theorem product_invertible_integers_mod_120_eq_one :
  let n := ∏ i in (multiset.filter (λ x, Nat.coprime x 120) (multiset.range 120)), i
  in n % 120 = 1 := 
by
  sorry

end product_invertible_integers_mod_120_eq_one_l592_592326


namespace cistern_empty_time_l592_592862

/-- This defines the rate of filling the cistern without a leak. -/
def fill_rate_no_leak : ℚ := 1 / 7

/-- This defines the rate of filling the cistern with the leak. -/
def fill_rate_with_leak : ℚ := 1 / 8

/-- This defines the leak rate, calculated as the difference between the filling rates without and with the leak. -/
def leak_rate : ℚ := fill_rate_no_leak - fill_rate_with_leak

/-- This theorem states that the time for the leak to empty the full cistern (time to empty) equals 56 hours. -/
theorem cistern_empty_time : (1 / leak_rate) = 56 := 
by 
sory

end cistern_empty_time_l592_592862


namespace geometric_series_sum_first_10_terms_l592_592690

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

noncomputable def sum_geom_series (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q ^ n) / (1 - q)

theorem geometric_series_sum_first_10_terms : 
  (a1 a4 : ℝ) (n : ℕ) (h1 : a1 = 1) (h4 : a4 = 1/8) (h_n : n = 10) :
  sum_geom_series a1 (1 / 2) n = 2 - 1 / 2^9 :=
by
  sorry

end geometric_series_sum_first_10_terms_l592_592690


namespace sum_of_proper_divisors_600_l592_592068

-- Define a function to calculate the sum of divisors
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in finset.range n.succ, if n % i = 0 then i else 0

-- Define a function to calculate the sum of proper divisors
def sum_of_proper_divisors (n : ℕ) : ℕ :=
  sum_of_divisors n - n

-- Assert that the sum of the proper divisors of 600 is 1260
theorem sum_of_proper_divisors_600 : sum_of_proper_divisors 600 = 1260 :=
  sorry

end sum_of_proper_divisors_600_l592_592068


namespace arccos_neg_one_l592_592548

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l592_592548


namespace correctness_of_statements_l592_592630

open_locale classical

/-- Definitions based on the given conditions -/
def AB : ℝ × ℝ × ℝ := (2, -1, -4)
def AD : ℝ × ℝ × ℝ := (4, 2, 0)
def AP : ℝ × ℝ × ℝ := (-1, 2, -1)

/-- Dot product function -/
def dot_prod (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

/-- Define BD as AD - AB -/
def BD : ℝ × ℝ × ℝ := (AD.1 - AB.1, AD.2 - AB.2, AD.3 - AB.3)

/-- The theorem stating the correctness of statements ①, ②, and ③ -/
theorem correctness_of_statements :
  (dot_prod AP AB = 0) ∧
  (dot_prod AP AD = 0) ∧
  (dot_prod AP AB = 0 ∧ dot_prod AP AD = 0 ∧ (∃ p, AP = p • BD) = false) :=
by { sorry }

end correctness_of_statements_l592_592630


namespace bead_color_2014_l592_592593

/-- Prove the color of the 2014th bead in the given repeating sequence of colored beads. -/
theorem bead_color_2014 : 
  let red_beads := 5
  let white_beads := 4
  let yellow_beads := 3
  let blue_beads := 2
  let total_beads := 2014
  let cycle_length := red_beads + white_beads + yellow_beads + blue_beads
  let position_in_cycle := 
    if total_beads % cycle_length = 0 
    then cycle_length 
    else total_beads % cycle_length
  let bead_color := 
    if position_in_cycle <= red_beads 
    then "red"
    else if position_in_cycle <= red_beads + white_beads 
    then "white"
    else if position_in_cycle <= red_beads + white_beads + yellow_beads 
    then "yellow"
    else "blue"
  in bead_color = "yellow" :=
by 
  let red_beads := 5
  let white_beads := 4
  let yellow_beads := 3
  let blue_beads := 2
  let total_beads := 2014
  let cycle_length := red_beads + white_beads + yellow_beads + blue_beads
  have position_in_cycle := total_beads % cycle_length
  have bead_color := 
    if position_in_cycle <= red_beads 
    then "red"
    else if position_in_cycle <= red_beads + white_beads 
    then "white"
    else if position_in_cycle <= red_beads + white_beads + yellow_beads 
    then "yellow"
    else "blue"
  have h : bead_color = "yellow"
  sorry

end bead_color_2014_l592_592593


namespace ellipse_slope_product_constant_l592_592220

variables {a b c k₁ k₂ : ℝ}
variables (x y : ℝ)

/-- Given conditions for the ellipse and other geometric constraints,
  prove that the product of the slopes k₁ and k₂ is a constant. -/
theorem ellipse_slope_product_constant
  (h₁ : a = 2)
  (h₂ : a > b)
  (h₃ : b > 0)
  (h₄ : c / a = sqrt (3 / 4))
  (h₅ : a^2 = b^2 + c^2)
  (h₆ : k₁ ≠ 0)
  (h₇ : 4 * y^2 + x^2 = 4) -- equation of the ellipse
  (h₈ : (-1, sqrt 3 / 2) ∈ set_of (λ p : ℝ × ℝ, 4 * p.2^2 + p.1^2 = 4))
  (h₉ : (1, 0) ∈ line_through (line_through_origin_with_slope k₁))
  (h₁₀ : E = line_intersect_ellipse k₁ (1, 0))
  (h₁₁ : F = line_intersect_ellipse k₁ (1, 0))
  (h₁₂ : (3, y₁) = line_intersect_line (x₁, y₁) (2, 0) 3)
  (h₁₃ : (3, y₂) = line_intersect_line (x₂, y₂) (2, 0) 3)
  (h₁₄ : P = midpoint (3, y₁) (3, y₂))
  (h₁₅ : k₂ = slope P (1, 0))
  : k₁ * k₂ = -1 / 4 := 
sorry

end ellipse_slope_product_constant_l592_592220


namespace solution_set_f_x_plus_2_lt_5_l592_592623

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4 * x else (abs x)^2 - 4 * abs x

theorem solution_set_f_x_plus_2_lt_5 :
  { x : ℝ | f (x + 2) < 5 } = set.Ioo (-7 : ℝ) 3 :=
by
  sorry

end solution_set_f_x_plus_2_lt_5_l592_592623


namespace sin_315_eq_neg_sqrt2_div_2_l592_592973

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592973


namespace train_crosses_pole_in_given_time_l592_592699

noncomputable def train_crossing_time (l : ℝ) (v_kmh : ℝ) : ℝ :=
  let v_ms := v_kmh * 1000 / 3600
  l / v_ms

theorem train_crosses_pole_in_given_time :
  train_crossing_time 150 122 ≈ 4.43 := by
  sorry

end train_crosses_pole_in_given_time_l592_592699


namespace solve_n_from_equation_l592_592248

-- Given condition, we define the equation
def equation (n : ℝ) : Prop :=
  (real.sqrt (2 * n))^2 + (12 * n) / 4 - 7 = 64

-- The theorem statement that we need to prove
theorem solve_n_from_equation : ∃ (n : ℝ), equation n ∧ n = 14.2 :=
by {
  sorry  -- Placeholder for the proof
}

end solve_n_from_equation_l592_592248


namespace problem1_problem2_l592_592306

-- Definitions for conditions
variable {ℂ : Type} [IsComplex ℂ]

-- Definition of the conjugate
def conj (z : ℂ) : ℂ := complex.conj z

-- Problem 1: If z is a pure imaginary number and satisfies |z - conj(z)| = 2√3, then z = ±√3i.
theorem problem1 (b : ℝ) (z : ℂ) (h1 : z = b * I) (h2 : |z - conj(z)| = 2 * sqrt 3) : z = sqrt 3 * I ∨ z = -sqrt 3 * I :=
sorry

-- Problem 2: If z - conj(z^2) is a real number and |z - conj(z)| = 2√3, then |z| = √13 / 2.
theorem problem2 (a b : ℝ) (z : ℂ) (h1 : z = a + b * I) (h2 : |z - conj(z)| = 2 * sqrt 3) (h3 : (z - conj(z^2)).im = 0) : |z| = sqrt 13 / 2 :=
sorry

end problem1_problem2_l592_592306


namespace number_of_two_bedroom_units_l592_592479

-- Define the total number of units and costs
variables (x y : ℕ)
def total_units := (x + y = 12)
def total_cost := (360 * x + 450 * y = 4950)

-- The target is to prove that there are 7 two-bedroom units
theorem number_of_two_bedroom_units : total_units ∧ total_cost → y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l592_592479


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592041

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592041


namespace sum_of_digits_of_expression_l592_592036

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592036


namespace product_invertibles_mod_120_l592_592313

theorem product_invertibles_mod_120 :
  let n := (list.filter (λ k, Nat.coprime k 120) (list.range 120)).prod
  in n % 120 = 119 :=
by
  sorry

end product_invertibles_mod_120_l592_592313


namespace vector_magnitude_problem_l592_592622

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem vector_magnitude_problem : 
  ∀ (x : ℝ), 
  parallel a (a - b x) → 
  |(1 : ℝ, 2 : ℝ) + b (1 / 2)| = (3 * Real.sqrt 5 / 2) := 
by
  sorry

end vector_magnitude_problem_l592_592622


namespace johns_initial_trempons_l592_592707

theorem johns_initial_trempons : ∃ x : ℕ, 
  (∀ y z w : ℕ, y = w / 2 + 0.5 → 
                z = y / 2 + 0.5 → 
                w = z / 2 + 0.5 → 
                x = w) ∧ 
  x = 3 :=
sorry

end johns_initial_trempons_l592_592707


namespace arithmetic_seq_common_diff_l592_592677

theorem arithmetic_seq_common_diff (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 0 + a 2 = 10) 
  (h2 : a 3 + a 5 = 4)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = -1 := 
  sorry

end arithmetic_seq_common_diff_l592_592677


namespace relationship_between_a_and_b_l592_592181

noncomputable def a : ℝ := ∫ x in (Real.pi / 2)..2, Real.sin x
noncomputable def b : ℝ := ∫ x in 0..1, Real.cos x

theorem relationship_between_a_and_b : a < b :=
by
  -- Definition of a and b respectively
  rw [a, b]
  sorry

end relationship_between_a_and_b_l592_592181


namespace sqrt_mul_eq_four_l592_592835

theorem sqrt_mul_eq_four : sqrt 2 * sqrt 8 = 4 := by
  have h1 : sqrt (2 * 4) = sqrt 8 := by rw [mul_comm, sqrt_mul (zero_le_two) (zero_le_four), Real.sqrt4, sqrt_mul_self zero_le_two]
  have h2 : sqrt 8 = 2 * sqrt 2 := by rw [sqrt_eq_rpow, Real.rpow_mul, Real.rpow_two, sqrt_sq zero_le_two]
  have h3 : sqrt 2 * (2 * sqrt 2) = 2 * (sqrt 2 * sqrt 2) := by ring
  have h4 : sqrt 2 * sqrt 2 = 2 := Real.sqrt_mul_self zero_le_two
  rw [h2, h3, h4]
  exact rfl

end sqrt_mul_eq_four_l592_592835


namespace xyz_solution_l592_592155

noncomputable def solve_xyz : set (ℝ × ℝ × ℝ) :=
  { (t, t, 0) | t : ℝ } ∪ { (0, t, t) | t : ℝ }

theorem xyz_solution :
  ∀ x y z : ℝ, (x - y + z)^2 = x^2 - y^2 + z^2 ↔ (x, y, z) ∈ solve_xyz :=
by
  sorry

end xyz_solution_l592_592155


namespace wrapping_paper_fraction_l592_592297

theorem wrapping_paper_fraction (s l : ℚ) (h1 : 4 * s + 2 * l = 5 / 12) (h2 : l = 2 * s) :
  s = 5 / 96 ∧ l = 5 / 48 :=
by
  sorry

end wrapping_paper_fraction_l592_592297


namespace correct_statements_l592_592639

def f (x : ℝ) : ℝ := |Real.cos x| * Real.sin x

theorem correct_statements : 
  (f (2014 * Real.pi / 3) = -Real.sqrt 3 / 4) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → MonotoneOn f (Set.Icc (-Real.pi / 4) (Real.pi / 4))) :=
by
  sorry

end correct_statements_l592_592639


namespace sum_abs_bound_l592_592186

theorem sum_abs_bound (n : ℕ) (x : Fin n → ℝ) 
  (h1 : ∀ i, x i ∈ Set.Icc (-1 : ℝ) 1)
  (h2 : ∑ i, (x i) ^ 3 = 0) : 
  |∑ i, x i| ≤ n / 3 := 
sorry

end sum_abs_bound_l592_592186


namespace sum_of_digits_of_expression_l592_592052

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592052


namespace sara_height_l592_592364

def Julie := 33
def Mark := Julie + 1
def Roy := Mark + 2
def Joe := Roy + 3
def Sara := Joe + 6

theorem sara_height : Sara = 45 := by
  sorry

end sara_height_l592_592364


namespace remainder_is_4_over_3_l592_592448

noncomputable def original_polynomial (z : ℝ) : ℝ := 3 * z ^ 3 - 4 * z ^ 2 - 14 * z + 3
noncomputable def divisor (z : ℝ) : ℝ := 3 * z + 5
noncomputable def quotient (z : ℝ) : ℝ := z ^ 2 - 3 * z + 1 / 3

theorem remainder_is_4_over_3 :
  ∃ r : ℝ, original_polynomial z = divisor z * quotient z + r ∧ r = 4 / 3 :=
sorry

end remainder_is_4_over_3_l592_592448


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592018

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592018


namespace perfect_square_factors_18000_l592_592656

theorem perfect_square_factors_18000 : 
  ∃ n : ℕ, (∀ (a b c : ℕ), (0 ≤ a ∧ a ≤ 3) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (0 ≤ c ∧ c ≤ 3) ∧ (even a ∧ even b ∧ even c) →
    n = (2 ^ a) * (3 ^ b) * (5 ^ c) ∧ n ∣ 18000 → (a = 0 ∨ a = 2) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2)) ∧ n = 8 :=
sorry

end perfect_square_factors_18000_l592_592656


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592025

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592025


namespace cost_of_whitewashing_room_l592_592386

theorem cost_of_whitewashing_room :
  let length := 25
  let width := 15
  let height := 12
  let cost_per_sq_ft := 5
  let door_height := 6
  let door_width := 3
  let window_height := 4
  let window_width := 3
  let num_windows := 3
  let perimeter := 2 * (length + width)
  let wall_area := perimeter * height
  let door_area := door_height * door_width
  let window_area := window_height * window_width
  let total_window_area := num_windows * window_area
  let actual_wall_area := wall_area - door_area - total_window_area
  let total_cost := actual_wall_area * cost_per_sq_ft
  in
  total_cost = 4530 :=
by
  sorry

end cost_of_whitewashing_room_l592_592386


namespace integral1_integral2_derivative1_derivative2_l592_592853

section
variable (x : ℝ)

-- Definition of the first integrand
def integrand1 := (3 * x^2 + 4 * x^3)

-- Proof for the first integral
theorem integral1 : ∫ x in 0..2, integrand1 x = 24 := by
  -- proof goes here
  sorry

-- Definition of the second integrand
def integrand2 := (exp x + 2 * x)

-- Proof for the second integral
theorem integral2 : ∫ x in 0..1, integrand2 x = Real.exp 1 := by
  -- proof goes here
  sorry
end

section
variable (x : ℝ)

-- Definition of the first function for differentiation
def func1 := (x^2 + Real.sin (2 * x)) / Real.exp x

-- Proof for the first derivative
theorem derivative1 : deriv func1 x = (2 * (1 + Real.cos (2 * x)) - 2 * x - Real.sin (2 * x)) / Real.exp x := by
  -- proof goes here
  sorry

-- Definition of the second function for differentiation
def func2 := Real.log (2 * x + 1) - Real.log (2 * x - 1)

-- Proof for the second derivative
theorem derivative2 (h : x > 1 / 2) : deriv func2 x = -4 / (4 * x^2 - 1) := by
  -- proof goes here
  sorry
end

end integral1_integral2_derivative1_derivative2_l592_592853


namespace arithmetic_sequence_sum_l592_592804

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 + a 6 = 18) :
  S 10 = 90 :=
sorry

end arithmetic_sequence_sum_l592_592804


namespace fraction_addition_l592_592903

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l592_592903


namespace fraction_product_l592_592906

theorem fraction_product :
  (∏ k in Finset.range 668, ((3 * k + 4) : ℚ) / (3 * k + 7)) = (4 / 2007) :=
by
  sorry

end fraction_product_l592_592906


namespace pages_written_in_a_year_l592_592293

def pages_per_friend_per_letter : ℕ := 3
def friends : ℕ := 2
def letters_per_week : ℕ := 2
def weeks_per_year : ℕ := 52

theorem pages_written_in_a_year : 
  (pages_per_friend_per_letter * friends * letters_per_week * weeks_per_year) = 624 :=
by
  sorry

end pages_written_in_a_year_l592_592293


namespace fraction_addition_l592_592892

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l592_592892


namespace book_arrangement_l592_592356

theorem book_arrangement :
  let num_books : Nat := 11
  let asian_books : Nat := 3
  let european_books : Nat := 4
  let african_books : Nat := 4
  (num_books = asian_books + european_books + african_books) →
  let total_arrangements := (6! * 3! * 4!) 
  total_arrangements = 103680 := 
by
  intros
  sorry

end book_arrangement_l592_592356


namespace centroid_of_abd_lies_on_cf_l592_592714

-- Define the isosceles trapezoid and projections
variables {A B C D F : Type} 
variable [AffineSpace ℝ A]

-- Conditions provided in part (a)
variables (isIsoscelesTrapezoid : is_isosceles_trapezoid A B C D)
variable (F_projection : is_projection D A B F)

-- Statement we need to prove (the centroid of triangle ABD lies on CF)
theorem centroid_of_abd_lies_on_cf (h1 : is_isosceles_trapezoid A B C D) (h2 : is_projection D A B F) :
  ∃ G : A, (is_centroid G A B D) ∧ (lies_on G C F) :=
by 
  sorry

end centroid_of_abd_lies_on_cf_l592_592714


namespace ice_per_person_l592_592540

def number_of_people := 15
def total_money_spent := 9 -- in dollars
def cost_per_pack := 3 -- in dollars per pack
def bags_per_pack := 10

theorem ice_per_person :
  let total_packs := total_money_spent / cost_per_pack in
  let total_bags := total_packs * bags_per_pack in
  total_bags / number_of_people = 2 :=
by
  sorry

end ice_per_person_l592_592540


namespace arithmetic_sequence_properties_l592_592719

theorem arithmetic_sequence_properties (a_n S_n : ℕ → ℤ)
  (h1 : a_n 1 = -7)
  (h2 : S_n 3 = -15)
  (h3 : ∀ n, S_n n = n * (a_n 1 + (n - 1) * (a_n 2 - a_n 1)) / 2) :
  (∀ n, a_n n = 2 * n - 9) ∧ (∃ n, S_n n = -16) :=
begin
  sorry
end

end arithmetic_sequence_properties_l592_592719


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592999

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592999


namespace sin_double_angle_identity_l592_592607

open Real

theorem sin_double_angle_identity (α : ℝ) (h : sin (α - π / 4) = 3 / 5) : sin (2 * α) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l592_592607


namespace cost_of_dozen_pens_l592_592385

theorem cost_of_dozen_pens
  (cost_three_pens_five_pencils : ℝ)
  (cost_one_pen : ℝ)
  (pen_to_pencil_ratio : ℝ)
  (h1 : 3 * cost_one_pen + 5 * (cost_three_pens_five_pencils / 8) = 260)
  (h2 : cost_one_pen = 65)
  (h3 : cost_one_pen / (cost_three_pens_five_pencils / 8) = 5/1)
  : 12 * cost_one_pen = 780 := by
    sorry

end cost_of_dozen_pens_l592_592385


namespace expected_yield_is_correct_l592_592351

-- Define the garden dimensions in steps
def steps_length := 18
def steps_width := 25

-- Define the conversion factor from steps to feet
def step_length := 3

-- Calculate the garden dimensions in feet
def length_in_feet := steps_length * step_length
def width_in_feet := steps_width * step_length

-- Calculate the area of the garden in square feet
def garden_area := length_in_feet * width_in_feet

-- Define the expected yield per square foot of garden
def yield_per_square_foot := 3 / 4

-- Calculate the total expected yield in pounds
def expected_yield := garden_area * yield_per_square_foot

-- Prove the expected yield is 3037.5 pounds
theorem expected_yield_is_correct : expected_yield = 3037.5 := by
  -- Sorry is used here as a placeholder for the proof
  sorry

end expected_yield_is_correct_l592_592351


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592936

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592936


namespace length_GH_is_29_l592_592685

-- Define the known lengths of the segments
def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

-- State the problem to prove that the length of segment GH is 29
theorem length_GH_is_29 (x : ℕ) : 
  (let side_length_fourth_square := x + length_FE in
   let side_length_third_square := side_length_fourth_square + length_CD in
   let side_length_second_square := side_length_third_square + length_AB in
   let side_length_first_square := x in
   side_length_second_square - side_length_first_square = 29
  ) :=
sorry

end length_GH_is_29_l592_592685


namespace markup_amount_l592_592459

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.35
def net_profit : ℝ := 18

def overhead : ℝ := purchase_price * overhead_percentage
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + net_profit
def markup : ℝ := selling_price - purchase_price

theorem markup_amount : markup = 34.80 := by
  sorry

end markup_amount_l592_592459


namespace painting_problem_l592_592570

theorem painting_problem :
  let n := 8         -- number of balls
  let red := 5       -- number of red balls
  let white := 3     -- number of white balls
  ∃! (paintings : ℕ), 
  (∃ s : list ℕ, s.length = n ∧ 
                s.count 1 = red ∧ 
                s.count 0 = white ∧ 
                ∃ i : ℕ, 
                i + 2 < n ∧ 
                (s.drop i).take 3 = [1, 1, 1]) →
  paintings = 24 :=
begin
  sorry
end

end painting_problem_l592_592570


namespace students_in_both_math_and_chem_l592_592673

theorem students_in_both_math_and_chem (students total math physics chem math_physics physics_chem : ℕ) :
  total = 36 →
  students ≤ 2 →
  math = 26 →
  physics = 15 →
  chem = 13 →
  math_physics = 6 →
  physics_chem = 4 →
  math + physics + chem - math_physics - physics_chem - students = total →
  students = 8 := by
  intros h_total h_students h_math h_physics h_chem h_math_physics h_physics_chem h_equation
  sorry

end students_in_both_math_and_chem_l592_592673


namespace smallest_discount_advantageous_l592_592174

def discount_20_twice (x : ℝ) : ℝ := (1 - 0.20) * (1 - 0.20) * x
def discount_15_thrice (x : ℝ) : ℝ := (1 - 0.15) * (1 - 0.15) * (1 - 0.15) * x
def discount_30_then_10 (x : ℝ) : ℝ := (1 - 0.30) * (1 - 0.10) * x
def discount_40_then_5_twice (x : ℝ) : ℝ := (1 - 0.40) * (1 - 0.05) * (1 - 0.05) * x

def effective_discount (final_price : ℝ) (original_price : ℝ) : ℝ :=
  original_price - final_price

def min_n (x : ℝ) : ℝ := 1 - max (0.64) (0.614125) (0.63) (0.5415)

theorem smallest_discount_advantageous (x : ℝ) :
  let n := 46 in
  effective_discount ((1 - n / 100) * x) x > max (effective_discount (discount_20_twice x) x)
                                                (effective_discount (discount_15_thrice x) x)
                                                (effective_discount (discount_30_then_10 x) x)
                                                (effective_discount (discount_40_then_5_twice x) x) :=
by sorry

end smallest_discount_advantageous_l592_592174


namespace trains_cross_time_l592_592083

theorem trains_cross_time 
  (len_train1 len_train2 : ℕ) 
  (speed_train1_kmph speed_train2_kmph : ℕ) 
  (len_train1_eq : len_train1 = 200) 
  (len_train2_eq : len_train2 = 300) 
  (speed_train1_eq : speed_train1_kmph = 70) 
  (speed_train2_eq : speed_train2_kmph = 50) 
  : (500 / (120 * 1000 / 3600)) = 15 := 
by sorry

end trains_cross_time_l592_592083


namespace problem_solution_l592_592594

-- Define the operation otimes
def otimes (x y : ℚ) : ℚ := (x * y) / (x + y / 3)

-- Define the specific values x and y
def x : ℚ := 4
def y : ℚ := 3/2 -- 1.5 in fraction form

-- Prove the mathematical statement
theorem problem_solution : (0.36 : ℚ) * (otimes x y) = 12 / 25 := by
  sorry

end problem_solution_l592_592594


namespace smallest_c_in_progressions_l592_592307

def is_arithmetic_progression (a b c : ℤ) : Prop := b - a = c - b

def is_geometric_progression (b c a : ℤ) : Prop := c^2 = a*b

theorem smallest_c_in_progressions :
  ∃ (a b c : ℤ), is_arithmetic_progression a b c ∧ is_geometric_progression b c a ∧ 
  (∀ (a' b' c' : ℤ), is_arithmetic_progression a' b' c' ∧ is_geometric_progression b' c' a' → c ≤ c') ∧ c = 2 :=
by
  sorry

end smallest_c_in_progressions_l592_592307


namespace characteristic_sequence_subset_A_sum_intersection_P_Q_cardinality_l592_592175

-- Given set E and the definition of characteristic sequences
def E : Set ℕ := {a | a ≥ 1 ∧ a ≤ 100}

-- Defining characteristic sequence of a subset X
-- For subset \{a_1, a_2, ..., a_n\}
def characteristic_sequence (X : Set ℕ) : ℕ → ℕ
| n => if n ∈ X then 1 else 0

-- Specific subset {a_1, a_3, a_5}
def subset_A := {1, 3, 5}

-- Specific subset characteristic sequences for conditions in (2)
def P_characteristic_sequence (i : ℕ) : ℕ :=
if i % 2 = 1 then 1 else 0

def Q_characteristic_sequence (i : ℕ) : ℕ :=
if i % 3 = 1 then 1 else 0

-- Questions to prove:

-- (1) Prove the sum of the first three items of the characteristic sequence of subset {a1, a3, a5} is 2.
theorem characteristic_sequence_subset_A_sum : 
    characteristic_sequence subset_A 1 + 
    characteristic_sequence subset_A 2 + 
    characteristic_sequence subset_A 3 = 2 := 
sorry

-- (2) Prove the number of elements in P ∩ Q is 17.
theorem intersection_P_Q_cardinality : 
    (Finset.filter (λ n, P_characteristic_sequence n = 1 ∧ Q_characteristic_sequence n = 1) 
    (Finset.range 101)).card = 17 := 
sorry

end characteristic_sequence_subset_A_sum_intersection_P_Q_cardinality_l592_592175


namespace total_songs_listened_l592_592431

theorem total_songs_listened (vivian_daily : ℕ) (fewer_songs : ℕ) (days_in_june : ℕ) (weekend_days : ℕ) :
  vivian_daily = 10 →
  fewer_songs = 2 →
  days_in_june = 30 →
  weekend_days = 8 →
  (vivian_daily * (days_in_june - weekend_days)) + ((vivian_daily - fewer_songs) * (days_in_june - weekend_days)) = 396 := 
by
  intros h1 h2 h3 h4
  sorry

end total_songs_listened_l592_592431


namespace only_prime_in_sequence_47_l592_592241

def is_sequence_47 (n : ℕ) : ℕ :=
  47 * (10 ^ (2 * n) + 10 ^ (2 * n - 2) + ... + 10 ^ 2 + 1)

theorem only_prime_in_sequence_47 :
  ∀ n,  (is_sequence_47 n).prime ↔ n = 0 := by
  -- use induction or other methods to show that only n = 0 yields a prime number in this sequence
  sorry

end only_prime_in_sequence_47_l592_592241


namespace eq1_eq2_eq3_l592_592820

theorem eq1 (x : ℝ) : (x - 2)^2 - 5 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by 
  intro h
  sorry

theorem eq2 (x : ℝ) : x^2 + 4 * x = -3 → x = -1 ∨ x = -3 := 
by 
  intro h
  sorry
  
theorem eq3 (x : ℝ) : 4 * x * (x - 2) = x - 2 → x = 2 ∨ x = 1/4 := 
by 
  intro h
  sorry

end eq1_eq2_eq3_l592_592820


namespace count_arithmetic_sequence_l592_592657

theorem count_arithmetic_sequence : 
  let seq := list.range' 150 27 |>.filter (λ n => n % 4 == 2)
  seq.length = 54 :=
by
  sorry

end count_arithmetic_sequence_l592_592657


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592943

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592943


namespace volume_displaced_squared_l592_592866

theorem volume_displaced_squared
  (radius : ℝ) (height : ℝ) (side_length : ℝ) (w : ℝ)
  (h_radius : radius = 5)
  (h_height : height = 12)
  (h_side_length : side_length = 10)
  (h_w : w = (1 / 6) * 75 * (real.sqrt 6)) :
  w^2 = 937.5 :=
by
  rw [h_w]
  sorry

end volume_displaced_squared_l592_592866


namespace no_valid_circle_arrangement_l592_592702

theorem no_valid_circle_arrangement :
  ¬ ∃ (f : Fin 2022 → ℕ), 
    (∀ i : Fin 2022, 1 ≤ f i ∧ f i ≤ 2022) ∧ 
    (∀ i : Fin 2022, 
      let j := (i + 1) % 2022 in 
      f i % (f i - f j) = 0 ∧ (f i - f j) ≠ 0) := 
sorry

end no_valid_circle_arrangement_l592_592702


namespace zero_of_fn_exists_between_2_and_3_l592_592414

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 3 * x - 9

theorem zero_of_fn_exists_between_2_and_3 :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
sorry

end zero_of_fn_exists_between_2_and_3_l592_592414


namespace minimum_value_analysis_l592_592395

theorem minimum_value_analysis
  (a : ℝ) (m n : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : 2 * m + n = 2)
  (h4 : m > 0)
  (h5 : n > 0) :
  (2 / m + 1 / n) ≥ 9 / 2 :=
sorry

end minimum_value_analysis_l592_592395


namespace hypotenuse_not_increase_more_than_sqrt2_l592_592146

theorem hypotenuse_not_increase_more_than_sqrt2 (x y : ℝ) :
  sqrt ((x + 1)^2 + (y + 1)^2) ≤ sqrt (x^2 + y^2) + sqrt 2 :=
sorry

end hypotenuse_not_increase_more_than_sqrt2_l592_592146


namespace sum_of_zeta_fractionals_l592_592170

noncomputable def RiemannZeta (y : ℝ) : ℝ := ∑' m : ℕ, if m > 0 then 1 / (m ^ y) else 0

theorem sum_of_zeta_fractionals :
  (∑ j in (Set.Ici 3 : Set ℕ), RiemannZeta (2 * j) % 1) = -3 / 16 := by
  sorry

end sum_of_zeta_fractionals_l592_592170


namespace volume_of_cone_l592_592665

noncomputable def lateral_surface_area : ℝ := 8 * Real.pi

theorem volume_of_cone (l r h : ℝ)
  (h_lateral_surface : l * Real.pi = 2 * lateral_surface_area)
  (h_radius : l = 2 * r)
  (h_height : h = Real.sqrt (l^2 - r^2)) :
  (1/3) * Real.pi * r^2 * h = (8 * Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry

end volume_of_cone_l592_592665


namespace sum_of_first_nine_terms_l592_592232

def sequence (k : ℝ) (n : ℕ) : ℝ :=
  k * (n - 5) - 2

def sum_first_nine_terms (k : ℝ) : ℝ :=
  ∑ n in Finset.range 9, sequence k (n + 1)

theorem sum_of_first_nine_terms (k : ℝ) : sum_first_nine_terms k = -18 := by
  let S_9 := ∑ n in Finset.range 9, sequence k (n + 1)
  calc
    S_9 = ∑ n in Finset.range 9, (k * (n + 1 - 5) - 2)  : rfl
    ... = k * (∑ n in Finset.range 9, (n + 1 - 5)) - 2 * 9 : by
      simp [sum_mul, sub_mul, sum_sub_distrib]
    ... = k * -20 - 18 : by
      simp [Finset.sum_range_succ', Nat.add_sub, sum_eq_zero, Nat.sub_sub]
        case ih {
          ring,
        }
    ... = -18 : by
      sorry

end sum_of_first_nine_terms_l592_592232


namespace length_of_PS_in_right_triangle_l592_592286

theorem length_of_PS_in_right_triangle 
  (P Q R S : Type) 
  (PQ QR PR PS : ℝ)
  (hPQ : PQ = 8) 
  (hQR : QR = 15) 
  (hPR : PR = 17)
  (is_right_triangle : PQ^2 + QR^2 = PR^2)
  (angle_bisector : PS is_angle_bisector_of ∠ P Q R) : 
  PS = 15 * real.sqrt 1065 / 32 := 
sorry

end length_of_PS_in_right_triangle_l592_592286


namespace product_invertible_integers_mod_120_eq_one_l592_592324

theorem product_invertible_integers_mod_120_eq_one :
  let n := ∏ i in (multiset.filter (λ x, Nat.coprime x 120) (multiset.range 120)), i
  in n % 120 = 1 := 
by
  sorry

end product_invertible_integers_mod_120_eq_one_l592_592324


namespace cos_alpha_value_l592_592208

open Real

theorem cos_alpha_value (α : ℝ) (h_cos : cos (α - π/6) = 15/17) (h_range : π/6 < α ∧ α < π/2) : 
  cos α = (15 * Real.sqrt 3 - 8) / 34 :=
by
  sorry

end cos_alpha_value_l592_592208


namespace find_common_ratio_of_geometric_sequence_l592_592689

noncomputable def geometric_sequence_q (a : ℕ → ℝ) (S_6 : ℝ) (cond1 : S_6 = 120) 
  (cond2 : a 0 + a 2 + a 4 = 30) : ℝ :=
  let q := (a 1 + a 3 + a 5) / (a 0 + a 2 + a 4) in
  q

theorem find_common_ratio_of_geometric_sequence (a : ℕ → ℝ)  (S_6 : ℝ) 
  (cond1 : S_6 = 120) (cond2 : a 0 + a 2 + a 4 = 30) :
  geometric_sequence_q a S_6 cond1 cond2 = 3 :=
sorry

end find_common_ratio_of_geometric_sequence_l592_592689


namespace probability_product_divisible_by_4_l592_592601

open Probability

theorem probability_product_divisible_by_4 :
  (∑ x in finset.filter (λ (x : ℕ × ℕ), ((x.1 * x.2) % 4 = 0)) (finset.product (finset.range 8) (finset.range 8)), 1) / 64 = 25/64 :=
by sorry

end probability_product_divisible_by_4_l592_592601


namespace students_playing_both_football_and_cricket_l592_592750

variable (F C N Total B : ℕ)
variable (plays_football_and_cricket : F + C - B + N = Total)

theorem students_playing_both_football_and_cricket :
  F = 325 → C = 175 → N = 50 → Total = 460 → B = 90 :=
by
  intros hF hC hN hTotal
  have h : 325 + 175 - B + 50 = 460 := by
    rw [hF, hC, hN, hTotal] at plays_football_and_cricket
    exact plays_football_and_cricket
  linarith

end students_playing_both_football_and_cricket_l592_592750


namespace find_line_through_P_l592_592235

def Sphere : Type :=
  { center : ℝ × ℝ × ℝ, radius : ℝ }

variables (O1 O2 : Sphere) (P : ℝ × ℝ × ℝ)

-- Hypotheses
axiom sphere_intersection : (∃ (C : ℝ × ℝ × ℝ) (r : ℝ), C = P ∧ r > 0 ∧ 
  dist O1.center C = r ∧ dist O2.center C = r )

axiom angle_with_horizontal : (∃ (l : ℝ×ℝ×ℝ → ℝ×ℝ×ℝ), 
  ∃ (θ : ℝ), θ = 30 * (π / 180) ∧ 
  ∀ θ, θ = atan2 (l.2 - P.2) (sqrt ((l.1 - P.1)^2 + (l.3 - P.3)^2)))

axiom symmetric_property : 
  ∃ A B : ℝ × ℝ × ℝ, A ≠ B ∧ 
  dist P A = dist P B ∧ 
  dist P A = r

theorem find_line_through_P : 
  ∃ l : ℝ×ℝ×ℝ → ℝ×ℝ×ℝ, 
  ∃ A B : ℝ × ℝ × ℝ, 
  A ≠ B ∧
  dist P A = dist P B ∧
  (l P).1 = A.1 ∧ (l P).2 = A.2 ∧ (l P).3 = A.3 ∧
  (l P).1 = B.1 ∧ (l P).2 = B.2 ∧ (l P).3 = B.3 :=
sorry

end find_line_through_P_l592_592235


namespace remaining_volume_of_cube_l592_592101

theorem remaining_volume_of_cube (s : ℝ) (r : ℝ) (h : ℝ) (π : ℝ) 
    (cube_volume : s = 5) 
    (cylinder_radius : r = 1.5) 
    (cylinder_height : h = 5) :
    s^3 - π * r^2 * h = 125 - 11.25 * π := by
  sorry

end remaining_volume_of_cube_l592_592101


namespace sqrt_sum_of_three_powers_equals_seventy_five_l592_592833

theorem sqrt_sum_of_three_powers_equals_seventy_five :
  sqrt (5^4 + 5^4 + 5^4) = 75 :=
by sorry

end sqrt_sum_of_three_powers_equals_seventy_five_l592_592833


namespace giant_spider_leg_pressure_l592_592487

-- Defining all conditions
def previous_spider_weight : ℝ := 6.4
def weight_multiplier : ℝ := 2.5
def num_legs : ℕ := 8
def leg_area : ℝ := 0.5

-- Compute the total weight of the giant spider
def giant_spider_weight : ℝ := previous_spider_weight * weight_multiplier

-- Compute the weight per leg
def weight_per_leg : ℝ := giant_spider_weight / num_legs

-- Calculate the pressure each leg undergoes
def pressure_per_leg : ℝ := weight_per_leg / leg_area

-- Theorem stating the expected pressure
theorem giant_spider_leg_pressure : pressure_per_leg = 4 :=
by
  -- Sorry is used here to skip the proof steps
  sorry

end giant_spider_leg_pressure_l592_592487


namespace cups_of_flour_already_put_in_correct_l592_592744

-- Let F be the number of cups of flour Mary has already put in
def cups_of_flour_already_put_in (F : ℕ) : Prop :=
  let total_flour_needed := 12
  let cups_of_salt := 7
  let additional_flour_needed := cups_of_salt + 3
  F = total_flour_needed - additional_flour_needed

-- Theorem stating that F = 2
theorem cups_of_flour_already_put_in_correct (F : ℕ) : cups_of_flour_already_put_in F → F = 2 :=
by
  intro h
  sorry

end cups_of_flour_already_put_in_correct_l592_592744


namespace all_statements_correct_l592_592136

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem all_statements_correct (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (¬(0 ∈ Set.range (f b))) ∧
  (∀ x, 0 < x ∧ x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) := by
  unfold f
  sorry

end all_statements_correct_l592_592136


namespace continuity_and_discontinuity_at_zero_l592_592288

noncomputable def f (x : ℝ) : ℝ := 5^(1/x) / (1 + 5^(1/x))

theorem continuity_and_discontinuity_at_zero :
  ∀ ε > 0, ∃ δ > 0, (∀ x, 0 < abs x → abs x < δ → abs (f x - f (0)) < ε) ∧ 
  (∀ ε > 0, ∃ δ > 0, (∀ x, 0 < x ∧ x < δ → abs (f x - 1) < ε)) ∧ 
  (∀ ε > 0, ∃ δ > 0, (∀ x, -δ < x ∧ x < 0 → abs (f x) < ε)) :=
begin
  sorry
end

end continuity_and_discontinuity_at_zero_l592_592288


namespace sin_315_equals_minus_sqrt2_div_2_l592_592987

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592987


namespace expected_value_of_biased_coin_l592_592094

noncomputable def expected_value : ℚ :=
  (2 / 3) * 5 + (1 / 3) * -6

theorem expected_value_of_biased_coin :
  expected_value = 4 / 3 := by
  sorry

end expected_value_of_biased_coin_l592_592094


namespace sum_of_products_nonzero_l592_592678

theorem sum_of_products_nonzero :
  ∀ (a : Fin 2021 × Fin 2021 → ℤ), 
  (∀ i j, a(i, j) = 1 ∨ a(i, j) = -1) →
  let P := λ j, ∏ i : Fin 2021, a(i, j)
  let R := λ i, ∏ j : Fin 2021, a(i, j)
  (∑ j : Fin 2021, P j + ∑ i : Fin 2021, R i) ≠ 0 :=
by
  sorry

end sum_of_products_nonzero_l592_592678


namespace fraction_addition_l592_592894

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l592_592894


namespace leo_final_amount_l592_592302

theorem leo_final_amount (total_amount : ℝ) (ryan_share sarah_share ryan_owes_sarah_leo sarah_owes_sarah_leo leo_owes_ryan leo_owes_sarah : ℝ) :
  total_amount = 72 →
  ryan_share = (2 / 5) * total_amount →
  sarah_share = (1 / 4) * total_amount →
  ryan_owes_sarah_leo = 8 →
  sarah_owes_sarah_leo = 10 →
  leo_owes_ryan = 6 →
  leo_owes_sarah = 4 →
  let leo_original_share := total_amount - (ryan_share + sarah_share)
  let ryan_net_transfer := ryan_owes_sarah_leo - leo_owes_ryan
  let sarah_net_transfer := sarah_owes_sarah_leo - leo_owes_sarah
  leo_original_share + ryan_net_transfer + sarah_net_transfer = 33.2 :=
by {
  intros,
  let leo_original_share := total_amount - (ryan_share + sarah_share),
  let ryan_net_transfer := ryan_owes_sarah_leo - leo_owes_ryan,
  let sarah_net_transfer := sarah_owes_sarah_leo - leo_owes_sarah,
  sorry
}

end leo_final_amount_l592_592302


namespace interval_of_x₀_l592_592624

-- Definition of the problem
variable (x₀ : ℝ)

-- Conditions
def condition_1 := x₀ > 0 ∧ x₀ < Real.pi
def condition_2 := Real.sin x₀ + Real.cos x₀ = 2 / 3

-- Proof problem statement
theorem interval_of_x₀ 
  (h1 : condition_1 x₀)
  (h2 : condition_2 x₀) : 
  x₀ > 7 * Real.pi / 12 ∧ x₀ < 3 * Real.pi / 4 := 
sorry

end interval_of_x₀_l592_592624


namespace pyramid_lateral_surface_area_l592_592876

noncomputable def lateralSurfaceAreaOfPyramid (n : ℕ) (R α : ℝ) : ℝ :=
  (n * R^2 * (Real.cot (α / 2))^2 * Real.tan (Real.pi / n)) / Real.cos α

theorem pyramid_lateral_surface_area (n : ℕ) (R α : ℝ) : 
  n > 2 →
  α > 0 → α < Real.pi →
  lateralSurfaceAreaOfPyramid n R α = 
    (n * R^2 * (Real.cot (α / 2))^2 * Real.tan (Real.pi / n)) / Real.cos α := 
by
  sorry

end pyramid_lateral_surface_area_l592_592876


namespace sin_315_eq_neg_sqrt2_div_2_l592_592916

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592916


namespace area_of_square_II_l592_592265

theorem area_of_square_II {a b : ℝ} (h : a > b) (d : ℝ) (h1 : d = a - b)
    (A1_A : ℝ) (h2 : A1_A = (a - b)^2 / 2) (A2_A : ℝ) (h3 : A2_A = 3 * A1_A) :
  A2_A = 3 * (a - b)^2 / 2 := by
  sorry

end area_of_square_II_l592_592265


namespace max_discount_benefit_l592_592676

theorem max_discount_benefit {S X : ℕ} (P : ℕ → Prop) :
  S = 1000 →
  X = 99 →
  (∀ s1 s2 s3 s4 : ℕ, s1 ≥ s2 ∧ s2 ≥ s3 ∧ s3 ≥ s4 ∧ s4 ≥ X ∧ s1 + s2 + s3 + s4 = S →
  ∃ N : ℕ, P N ∧ N = 504) := 
by
  intros hS hX
  sorry

end max_discount_benefit_l592_592676


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592945

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592945


namespace product_of_first_nine_terms_l592_592275

-- Declare the geometric sequence and given condition
variable {α : Type*} [Field α]
variable {a : ℕ → α}
variable (r : α) (a1 : α)

-- Define that the sequence is geometric
def is_geometric_sequence (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  ∀ n : ℕ, a n = a1 * r ^ n

-- Given a_5 = -2 in the sequence
def geometric_sequence_with_a5 (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  is_geometric_sequence a r a1 ∧ a 5 = -2

-- Prove that the product of the first 9 terms is -512
theorem product_of_first_nine_terms 
  (a : ℕ → α) 
  (r : α) 
  (a₁ : α) 
  (h : geometric_sequence_with_a5 a r a₁) : 
  (a 0) * (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = -512 := 
by
  sorry

end product_of_first_nine_terms_l592_592275


namespace rectangle_area_l592_592875

open Real

theorem rectangle_area :
  ∃ (l w : ℝ), ((l + 2) * w = l * w + 10) ∧ (l * (w - 3) = l * w - 18) ∧ (l * w = 30) :=
by {
  -- Definitions for conditions
  let l := 6
  let w := 5

  have h1 : (l + 2) * w = l * w + 10 := by norm_num
  have h2 : l * (w - 3) = l * w - 18 := by norm_num

  -- Original area proof
  have h3 : l * w = 30 := by norm_num

  -- Conclusion
  use [l, w],
  exact ⟨h1, h2, h3⟩
}

end rectangle_area_l592_592875


namespace investment_total_l592_592078

theorem investment_total (x y : ℝ) (h₁ : 0.08 * x + 0.05 * y = 490) (h₂ : x = 3000 ∨ y = 3000) : x + y = 8000 :=
by
  sorry

end investment_total_l592_592078


namespace probability_min_diff_three_or_greater_probability_of_min_diff_ge_three_l592_592419

open Finset

theorem probability_min_diff_three_or_greater :
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  in (∑ x in s.powerset, if x.card = 3 then 
    if (∀ (a b : ℕ), a ∈ x → b ∈ x → a ≠ b → abs (a - b) ≥ 3) then 1 else 0 else 0) = 6 :=
sorry

theorem probability_of_min_diff_ge_three :
  let total := Nat.choose 9 3 in
  let valid := 6 in
  valid / total = 1 / 14 :=
sorry

end probability_min_diff_three_or_greater_probability_of_min_diff_ge_three_l592_592419


namespace lock_combination_correct_l592_592367

def distinct_digits (d : Finset ℕ) : Prop := d.card = d.to_list.nodup.card

noncomputable def STARS (S T A R : ℕ) := S * 12^4 + T * 12^3 + A * 12^2 + R * 12 + S
noncomputable def RATS  (S T A R : ℕ) := R * 12^3 + A * 12^2 + T * 12 + S
noncomputable def ARTS  (S T A R : ℕ) := A * 12^3 + R * 12^2 + T * 12 + S
noncomputable def START (S T A R : ℕ) := S * 12^4 + T * 12^3 + A * 12^2 + R * 12 + T

theorem lock_combination_correct (S T A R : ℕ) (h1: STARS S T A R + RATS S T A R + ARTS S T A R = START S T A R)
    (h2: distinct_digits {S, T, A, R}) : (5 * 12^2 + 7 * 12 + 1 = 805) := by
  sorry

end lock_combination_correct_l592_592367


namespace complement_U_A_l592_592650

-- Define the universal set U and the subset A
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

-- Define the complement of A relative to the universal set U
def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- The theorem we want to prove
theorem complement_U_A : complement U A = {2} := by
  sorry

end complement_U_A_l592_592650


namespace election_vote_percentage_l592_592691

theorem election_vote_percentage 
  (total_students : ℕ)
  (winner_percentage : ℝ)
  (loser_percentage : ℝ)
  (vote_difference : ℝ)
  (P : ℝ)
  (H1 : total_students = 2000)
  (H2 : winner_percentage = 0.55)
  (H3 : loser_percentage = 0.45)
  (H4 : vote_difference = 50)
  (H5 : 0.1 * P * (total_students / 100) = vote_difference) :
  P = 25 := 
sorry

end election_vote_percentage_l592_592691


namespace product_of_first_three_terms_is_960_l592_592799

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l592_592799


namespace sin_315_eq_neg_sqrt2_div_2_l592_592967

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592967


namespace elective_course_schemes_l592_592111

theorem elective_course_schemes : Nat.choose 4 2 = 6 := by
  sorry

end elective_course_schemes_l592_592111


namespace remainder_of_n_when_divided_by_7_l592_592168

theorem remainder_of_n_when_divided_by_7 (n : ℕ) :
  (n^2 ≡ 2 [MOD 7]) ∧ (n^3 ≡ 6 [MOD 7]) → (n ≡ 3 [MOD 7]) :=
by sorry

end remainder_of_n_when_divided_by_7_l592_592168


namespace class_average_correct_l592_592260

-- Define the constants as per the problem data
def total_students : ℕ := 30
def students_group_1 : ℕ := 24
def students_group_2 : ℕ := 6
def avg_score_group_1 : ℚ := 85 / 100  -- 85%
def avg_score_group_2 : ℚ := 92 / 100  -- 92%

-- Calculate total scores and averages based on the defined constants
def total_score_group_1 : ℚ := students_group_1 * avg_score_group_1
def total_score_group_2 : ℚ := students_group_2 * avg_score_group_2
def total_class_score : ℚ := total_score_group_1 + total_score_group_2
def class_average : ℚ := total_class_score / total_students

-- Goal: Prove that class_average is 86.4%
theorem class_average_correct : class_average = 86.4 / 100 := sorry

end class_average_correct_l592_592260


namespace midpoint_calculation_l592_592438

def point := ℝ × ℝ

def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_calculation : midpoint (7, -6) (-3, 4) = (2, -1) :=
  by
    simp [midpoint]
    norm_num

end midpoint_calculation_l592_592438


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592028

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592028


namespace arithmetic_seq_and_inequality_l592_592777

-- Condition 1: First term of the arithmetic sequence
def a1 : ℕ := 1

-- Define the general arithmetic sequence term
def a (n : ℕ) (d : ℕ) : ℕ := a1 + (n - 1) * d

-- Theorem statement to prove the general term formula and the inequality condition
theorem arithmetic_seq_and_inequality :
  let d := 2
  let a_n := λ n, a n d
  (a 3 d + a 5 d = a 4 d + 7) →
  (a 1 d = 1) →
  a_n n = 2 * n - 1 ∧ (∃ n, 1 < n ∧ n < 5 ∧ (S n < 3 * a_n n - 2))
:= by
  intros h1 h2
  sorry

end arithmetic_seq_and_inequality_l592_592777


namespace calculate_expression_l592_592909

theorem calculate_expression : -4^2 * (-1)^2022 = -16 :=
by
  sorry

end calculate_expression_l592_592909


namespace bin_to_base4_correct_l592_592826

-- Define the binary input
def binInput : List Nat := [1, 0, 1, 0, 1, 1, 1, 0]

-- Define the expected base 4 output
def base4Output : List Nat := [2, 2, 3, 2]

-- The theorem to prove the conversion correctness
theorem bin_to_base4_correct : 
  binInput = [1, 0, 1, 0, 1, 1, 1, 0] → 
  base4Output = [2, 2, 3, 2] → 
  base4Output = convertBinToBase4 binInput := 
by 
  sorry

-- Define the necessary conversion function (this would be filled in with the actual conversion code)
noncomputable def convertBinToBase4 (bin : List Nat) : List Nat := 
  sorry

end bin_to_base4_correct_l592_592826


namespace evaluate_fractional_exponent_l592_592571

theorem evaluate_fractional_exponent :
  ( (1 / 8) ^ (-1 / 3) = 2 ) :=
by
  sorry

end evaluate_fractional_exponent_l592_592571


namespace product_first_three_terms_arithmetic_seq_l592_592795

theorem product_first_three_terms_arithmetic_seq :
  ∀ (a₇ d : ℤ), 
  a₇ = 20 → d = 2 → 
  let a₁ := a₇ - 6 * d in
  let a₂ := a₁ + d in
  let a₃ := a₂ + d in
  a₁ * a₂ * a₃ = 960 := 
by
  intros a₇ d a₇_20 d_2
  let a₁ := a₇ - 6 * d
  let a₂ := a₁ + d
  let a₃ := a₂ + d
  sorry

end product_first_three_terms_arithmetic_seq_l592_592795


namespace rectangle_perimeter_l592_592135

theorem rectangle_perimeter {b : ℕ → ℕ} {W H : ℕ}
  (h1 : ∀ i, b i ≠ b (i+1))
  (h2 : b 9 = W / 2)
  (h3 : gcd W H = 1)

  (h4 : b 1 + b 2 = b 3)
  (h5 : b 1 + b 3 = b 4)
  (h6 : b 3 + b 4 = b 5)
  (h7 : b 4 + b 5 = b 6)
  (h8 : b 2 + b 3 + b 5 = b 7)
  (h9 : b 2 + b 7 = b 8)
  (h10 : b 1 + b 4 + b 6 = b 9)
  (h11 : b 6 + b 9 = b 7 + b 8) : 
  2 * (W + H) = 266 :=
  sorry

end rectangle_perimeter_l592_592135


namespace average_candies_per_bag_l592_592704

theorem average_candies_per_bag :
  let candies := [5, 7, 9, 12, 12, 15, 15, 18, 25]
  let num_bags := 9
  let total_candies := candies.sum
  let average := (total_candies : ℝ) / num_bags
  Nat.round average = 13 :=
by
  let candies := [5, 7, 9, 12, 12, 15, 15, 18, 25]
  let num_bags := 9
  let total_candies := candies.sum
  let average := (total_candies : ℝ) / num_bags
  sorry

end average_candies_per_bag_l592_592704


namespace greatest_n_factor_of_3_in_factorial_16_l592_592661

theorem greatest_n_factor_of_3_in_factorial_16 (n : ℕ) (m : ℕ) (h1 : m = 3^n) : 
  ∃ k : ℕ, k = 6 ∧ k * 3 ^ n ∣ nat.factorial 16 :=
begin
  use 6,
  split,
  { refl },
  {
    rw h1,
    sorry
  }
end

end greatest_n_factor_of_3_in_factorial_16_l592_592661


namespace josef_timothy_game_l592_592708

theorem josef_timothy_game : 
  ∃ n : ℕ, n = (∏ d in (finset.range 901).filter (λ k, 900 % k = 0), 1) ∧ n = 27 :=
by
  sorry

end josef_timothy_game_l592_592708


namespace minimize_sum_first_n_terms_l592_592199

noncomputable def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

noncomputable def sum_first_n_terms (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n-1) / 2) * d

theorem minimize_sum_first_n_terms (a₁ : ℤ) (a₃_plus_a₅ : ℤ) (n_min : ℕ) :
  a₁ = -9 → a₃_plus_a₅ = -6 → n_min = 5 := by
  sorry

end minimize_sum_first_n_terms_l592_592199


namespace sequence_general_formula_sum_first_n_terms_l592_592349

-- Definition and conditions
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - a 1
def arithmetic_seq (a1 a2 a3 : ℕ) : Prop := a1 + a3 = 2 * (a2 + 1)

-- Given sequences and initial conditions
variables {a : ℕ → ℕ}

-- Proof statements
theorem sequence_general_formula
  (S_n_eq : ∀ n, S n a = 2 * a n - a 1)
  (arith_seq : arithmetic_seq (a 1) (a 2) (a 3)) :
  ∀ n, a n = 2^n := sorry

noncomputable def b (n : ℕ) : ℕ := nat.log 2 (a n)
def T (n : ℕ) (a : ℕ → ℕ) : ℕ := ∑ i in finset.range n, a i * b i

theorem sum_first_n_terms
  (a_n_eq : ∀ n, a n = 2 ^ n) :
  ∀ n, T n a = (n - 1) * 2 ^ (n + 1) + 2 := sorry

end sequence_general_formula_sum_first_n_terms_l592_592349


namespace fraction_of_field_planted_l592_592148

theorem fraction_of_field_planted (AB AC : ℕ) (x : ℕ) (shortest_dist : ℕ) (hypotenuse : ℕ)
  (S : ℕ) (total_area : ℕ) (planted_area : ℕ) :
  AB = 5 ∧ AC = 12 ∧ hypotenuse = 13 ∧ shortest_dist = 2 ∧ x * x = S ∧ 
  total_area = 30 ∧ planted_area = total_area - S →
  (planted_area / total_area : ℚ) = 2951 / 3000 :=
by
  sorry

end fraction_of_field_planted_l592_592148


namespace jackson_house_visits_l592_592291

theorem jackson_house_visits
  (days_per_week : ℕ)
  (total_goal : ℕ)
  (monday_earnings : ℕ)
  (tuesday_earnings : ℕ)
  (earnings_per_4_houses : ℕ)
  (houses_per_4 : ℝ)
  (remaining_days := days_per_week - 2)
  (remaining_goal := total_goal - monday_earnings - tuesday_earnings)
  (daily_goal := remaining_goal / remaining_days)
  (earnings_per_house := houses_per_4 / 4)
  (houses_per_day := daily_goal / earnings_per_house) :
  days_per_week = 5 ∧
  total_goal = 1000 ∧
  monday_earnings = 300 ∧
  tuesday_earnings = 40 ∧
  earnings_per_4_houses = 10 ∧
  houses_per_4 = earnings_per_4_houses.toReal →
  houses_per_day = 88 := 
by 
  sorry

end jackson_house_visits_l592_592291


namespace convex_polygon_intersection_of_half_planes_l592_592357

noncomputable theory

variables {P : Type} [Polygon P] [ConvexPolygon P]

-- Statement: Any convex polygon P can be represented as the intersection of a finite number of half-planes
theorem convex_polygon_intersection_of_half_planes (P : Type) [Polygon P] [ConvexPolygon P] :
    ∃ (H : finset (half_plane P)), P = ⋂₀ H :=
sorry

end convex_polygon_intersection_of_half_planes_l592_592357


namespace gcf_lcm_problem_l592_592535

open Nat

-- Definitions based on given conditions
def n1 := 9
def n2 := 21
def n3 := 14
def n4 := 15

def lcm1 := lcm n1 n2
def lcm2 := lcm n3 n4

-- The equivalent proof problem statement
theorem gcf_lcm_problem :
  gcd lcm1 lcm2 = 21 :=
sorry

end gcf_lcm_problem_l592_592535


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592994

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592994


namespace probability_A_or_B_selected_l592_592177

open Finset

-- Constants representing the students
inductive Student : Type
| A | B | C | D

-- The set of all students
def students : Finset Student := {Student.A, Student.B, Student.C, Student.D}

-- The event of selecting exactly two students
def event (s1 s2 : Student) : Finset Student := {s1, s2}

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := (n.choose k)

-- Main theorem statement
theorem probability_A_or_B_selected : 
  (comb 2 1 * comb 2 1 : ℚ) / comb 4 2 = 2 / 3 := by
  sorry

end probability_A_or_B_selected_l592_592177


namespace find_f_x_l592_592182

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x, f (1 + x) = x^2 + 2 * x - 1) : ∀ x, f x = x^2 - 1 :=
begin
  sorry
end

end find_f_x_l592_592182


namespace product_of_invertibles_mod_120_l592_592337

open Nat

theorem product_of_invertibles_mod_120 :
  let m := 120
  let invertibles := { x | x < m ∧ gcd x m = 1 }
  ∏ a in invertibles, a % m = 119 :=
by
  sorry

end product_of_invertibles_mod_120_l592_592337


namespace range_of_a_l592_592753

theorem range_of_a (a : ℝ) : 
  (let p := ∀ x : ℝ, x^2 - (a+1)*x + 1 > 0 in
   let q := ∀ x : ℝ, ∃ f : ℝ → ℝ, f x = (a+1)^x ∧ monotone f in
   ¬ (p ∧ q) ∧ (p ∨ q)) ↔ (-3 < a ∧ a ≤ 0) ∨ (a ≥ 1) :=
by
  sorry

end range_of_a_l592_592753


namespace centers_of_equilateral_triangles_outside_triangle_form_equilateral_l592_592092

theorem centers_of_equilateral_triangles_outside_triangle_form_equilateral
  (ABC : Triangle)
  (O1 O2 O3: Point)
  (h1 : ∃ Δ_1 Δ_2 Δ_3 : Triangle, O1 = Δ_1.center ∧ O2 = Δ_2.center ∧ O3 = Δ_3.center ∧ 
    is_equilateral Δ_1 ∧ is_equilateral Δ_2 ∧ is_equilateral Δ_3 ∧ 
    is_on_side_of_triangle Δ_1 ABC ∧ is_on_side_of_triangle Δ_2 ABC ∧ is_on_side_of_triangle Δ_3 ABC) :
  is_equilateral (Triangle.mk O1 O2 O3) :=
sorry

end centers_of_equilateral_triangles_outside_triangle_form_equilateral_l592_592092


namespace three_term_arithmetic_seq_l592_592560

noncomputable def arithmetic_sequence_squares (x y z : ℤ) : Prop :=
  x^2 + z^2 = 2 * y^2

theorem three_term_arithmetic_seq (x y z : ℤ) :
  (∃ a b : ℤ, a = (x + z) / 2 ∧ b = (x - z) / 2 ∧ x^2 + z^2 = 2 * y^2) ↔
  arithmetic_sequence_squares x y z :=
by
  sorry

end three_term_arithmetic_seq_l592_592560


namespace min_sum_of_areas_l592_592854

theorem min_sum_of_areas:
  ∃ (x: ℝ) (A1 A2: ℝ),
  x + (12 - x) = 12 ∧
  A1 = (sqrt 3 / 36) * x^2 ∧
  A2 = (sqrt 3 / 36) * (12 - x)^2 ∧
  (∀ (y: ℝ), (sqrt 3 / 36) * y^2 + (sqrt 3 / 36) * (12 - y)^2 ≥ 2 * sqrt 3) ∧
  A1 + A2 = 2 * sqrt 3 :=
by
  sorry

end min_sum_of_areas_l592_592854


namespace angle_relationship_l592_592695

-- Definitions for conditions
variables (A B C D E F : Type*)
variables (AB AC BC: ℝ)
variables (a b c : ℝ)
variables [IsIsoscelesTriangle : is_isosceles_triangle A B C]
variables [IsInscribedIsoscelesTriangle : is_isosceles_triangle D E F]
variables (midpointD : is_midpoint D A B)
variables (midpointF : is_midpoint F A C)
variables (angle_BFD angle_ADE angle_FEC : ℝ)

-- Define properties from problem statement
axiom angle_BFD_eq_a : angle_BFD = a
axiom angle_ADE_eq_b : angle_ADE = b
axiom angle_FEC_eq_c : angle_FEC = c

-- Define the theorem to prove
theorem angle_relationship (h1 : angle_ADE = b) (h2 : angle_FEC = c) : c = 180 - 2 * b :=
by sorry

end angle_relationship_l592_592695


namespace exchange_rate_5_CAD_to_JPY_l592_592859

theorem exchange_rate_5_CAD_to_JPY :
  (1 : ℝ) * 85 * 5 = 425 :=
by
  sorry

end exchange_rate_5_CAD_to_JPY_l592_592859


namespace tetrahedron_volume_correct_l592_592009

-- Define the conditions of the problem
def side_length : ℝ := 30
def half_side_length : ℝ := side_length / 2
def base_triangle_area : ℝ := (half_side_length * half_side_length) / 2
def height_of_tetrahedron : ℝ := side_length
def expected_volume : ℝ := (1 / 3) * base_triangle_area * height_of_tetrahedron

-- State the theorem
theorem tetrahedron_volume_correct :
  expected_volume = 1125 :=
by
  -- Provide the proof here (omitted for this task)
  sorry

end tetrahedron_volume_correct_l592_592009


namespace Alex_meets_train_probability_l592_592515

noncomputable def probability_Alex_meets_train : ℚ := 11 / 72

theorem Alex_meets_train_probability :
  let time_range : set (ℚ × ℚ) := {xy | 0 ≤ xy.1 ∧ xy.1 ≤ 60 ∧ 0 ≤ xy.2 ∧ xy.2 ≤ 60}
  let shaded_region : set (ℚ × ℚ) := {xy | xy.2 - xy.1 ≤ 10 ∧ xy.2 - xy.1 ≥ -10}
  probability (shaded_region) / probability (time_range) = 11 / 72 := 
sorry

end Alex_meets_train_probability_l592_592515


namespace alpha_quadrant_and_trig_values_l592_592604

theorem alpha_quadrant_and_trig_values (α : ℝ) (M : ℝ × ℝ) (m : ℝ) (O : ℝ × ℝ) (h1 : 1 / |Real.sin α| = -1 / Real.sin α)
  (h2 : 0 < Real.cos α) (h3 : M = (3/5, m)) (h4 : Real.sqrt ((3/5)^2 + m^2) = 1) :
  (α ∈ set.Ioo (3*π/2) (2*π)) ∧ (m = -4/5) ∧ (Real.sin α = -4/5) :=
by
  sorry

end alpha_quadrant_and_trig_values_l592_592604


namespace solve_f_l592_592467

open Nat

theorem solve_f (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) + f n = 2 * n + 3) : f 1993 = 1994 := by
  -- assumptions and required proof
  sorry

end solve_f_l592_592467


namespace solve_f_inv_zero_l592_592722

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)
noncomputable def f_inv (a b x : ℝ) : ℝ := sorry -- this is where the inverse function definition would go

theorem solve_f_inv_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : f_inv a b 0 = (1 / b) :=
by sorry

end solve_f_inv_zero_l592_592722


namespace range_of_f_l592_592730

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f : ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → 
  Set.range f x = Set.Icc (Real.pi^4 / 16) (Real.pi^4 / 8) :=
by
  intro x hx
  sorry

end range_of_f_l592_592730


namespace number_of_steaks_needed_l592_592421

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end number_of_steaks_needed_l592_592421


namespace pairs_satisfaction_l592_592566

-- Definitions for the conditions given
def condition1 (x y : ℝ) : Prop := y = (x + 2)^2
def condition2 (x y : ℝ) : Prop := x * y + 2 * y = 2

-- The statement that we need to prove
theorem pairs_satisfaction : 
  (∃ x y : ℝ, condition1 x y ∧ condition2 x y) ∧ 
  (∃ x1 x2 : ℂ, x^2 + -2*x + 1 = 0 ∧ ¬∃ (y : ℝ), y = (x1 + 2)^2 ∨ y = (x2 + 2)^2) :=
by
  sorry

end pairs_satisfaction_l592_592566


namespace kinetic_energy_reduction_collisions_l592_592874

theorem kinetic_energy_reduction_collisions (E_0 : ℝ) (n : ℕ) :
  (1 / 2)^n * E_0 = E_0 / 64 → n = 6 :=
by
  sorry

end kinetic_energy_reduction_collisions_l592_592874


namespace work_problem_l592_592843

theorem work_problem (A B : ℝ) (hA : A = 1/4) (hB : B = 1/12) :
  (2 * (A + B) + 4 * B = 1) :=
by
  -- Work rate of A and B together
  -- Work done in 2 days by both
  -- Remaining work and time taken by B alone
  -- Final Result
  sorry

end work_problem_l592_592843


namespace granger_buys_3_jars_of_peanut_butter_l592_592655

theorem granger_buys_3_jars_of_peanut_butter :
  ∀ (spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count: ℕ),
    spam_cost = 3 → peanut_butter_cost = 5 → bread_cost = 2 →
    spam_count = 12 → loaf_count = 4 → total_cost = 59 →
    spam_cost * spam_count + bread_cost * loaf_count + peanut_butter_cost * peanut_butter_count = total_cost →
    peanut_butter_count = 3 :=
by
  intros spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count
  intros hspam_cost hpeanut_butter_cost hbread_cost hspam_count hloaf_count htotal_cost htotal
  sorry  -- The proof step is omitted as requested.

end granger_buys_3_jars_of_peanut_butter_l592_592655


namespace functional_equation_solution_l592_592150

open Nat

theorem functional_equation_solution :
  (∀ (f : ℕ → ℕ), 
    (∀ (x y : ℕ), 0 ≤ y + f x - (Nat.iterate f (f y) x) ∧ (y + f x - (Nat.iterate f (f y) x) ≤ 1)) →
    (∀ n, f n = n + 1)) :=
by
  intro f h
  sorry

end functional_equation_solution_l592_592150


namespace beautiful_fold_probability_l592_592746

def square (A B C D : Point) : Prop := -- definition of square with vertices A B C D

def beautiful_fold (A B C D : Point) (F : Point) : Prop := 
  -- definition that fold passing through F intersects AB & CD, dividing square into four equal right triangles

theorem beautiful_fold_probability (A B C D : Point) (F : Point) :
  square A B C D → random_point_on_square F A B C D →
  probability (beautiful_fold A B C D F) = 1 / 2 := 
sorry

end beautiful_fold_probability_l592_592746


namespace domain_range_of_f_l592_592387

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - (1/2)^x)

theorem domain_range_of_f :
  (∀ x : ℝ, f x ≥ 0 ↔ x ∈ set.Ici 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, y = f x ↔ y ∈ set.Ico 0 1) :=
by
  sorry

end domain_range_of_f_l592_592387


namespace infinitely_many_odd_composites_l592_592360

theorem infinitely_many_odd_composites :
  ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n ≥ 1 → composite (2 ^ (2 ^ n) + k)) := by
  sorry

end infinitely_many_odd_composites_l592_592360


namespace geom_sum_converges_to_4_over_3_l592_592910

noncomputable def geom_sum_2d : ℝ :=
  ∑' j : ℕ, ∑' k : ℕ, 4 ^ (- (k + 2 * j + (k + j) ^ 2))

theorem geom_sum_converges_to_4_over_3 : 
  geom_sum_2d = (4 / 3) :=
by sorry

end geom_sum_converges_to_4_over_3_l592_592910


namespace find_b6_l592_592396

def fib (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

theorem find_b6 (b : ℕ → ℕ) (b1 b2 : ℕ)
  (h1 : b 1 = b1) (h2 : b 2 = b2) (h3 : b 5 = 55)
  (hfib : fib b) : b 6 = 84 :=
  sorry

end find_b6_l592_592396


namespace benny_start_cards_l592_592124

--- Benny bought 4 new cards before the dog ate half of his collection.
def new_cards : Int := 4

--- The remaining cards after the dog ate half of the collection is 34.
def remaining_cards : Int := 34

--- The total number of cards Benny had before adding the new cards and the dog ate half.
def total_before_eating := remaining_cards * 2

theorem benny_start_cards : total_before_eating - new_cards = 64 :=
sorry

end benny_start_cards_l592_592124


namespace triangle_to_parallelogram_l592_592138

variables {A B C D E : Type*} [linear_ordered_field A]

-- Define the points and midline conditions
def is_midpoint (p1 p2 pm : A) : Prop := (pm = (p1 + p2) / 2)

def Midline_Properties (a b c d e : A) : Prop :=
  is_midpoint a b d ∧ is_midpoint a c e ∧ (d - e) = (b - c) / 2 ∧ (d ≠ e)

-- Define the main theorem statement
theorem triangle_to_parallelogram (a b c d e : A) (H : Midline_Properties a b c d e) :
  ∃ x y z w : A, (is_midpoint x y z ∧ is_midpoint x w z ∧ -- Forming parallelogram conditions
                  (y - z) = (y - w) / 2 ∧ (y = w) ∧ (z ≠ w)) :=
sorry 

end triangle_to_parallelogram_l592_592138


namespace seven_balls_expected_positions_l592_592365

theorem seven_balls_expected_positions :
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  expected_positions = 3.61 :=
by
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  exact sorry

end seven_balls_expected_positions_l592_592365


namespace circle_center_coordinates_l592_592770

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (c = (1, -2)) ∧ 
  (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - 1)^2 + (y + 2)^2 = 9)) :=
by
  sorry

end circle_center_coordinates_l592_592770


namespace unique_solution_f_eq_x_squared_l592_592574

noncomputable def f (x : ℝ) : ℝ := sorry

theorem unique_solution_f_eq_x_squared :
  (∀ x : ℝ, f x = Real.sup (set_of (λ y : ℝ, 2 * x * y - f y))) →
  (∀ x : ℝ, f x = x^2) :=
begin
  intro h,
  -- proof to be filled
  sorry
end

end unique_solution_f_eq_x_squared_l592_592574


namespace imaginary_part_of_z_l592_592213

open Complex

theorem imaginary_part_of_z : 
  let z := (1 + 2 * Complex.i) * (2 - Complex.i)
  Im z = 3 :=
by {
  let z := (1 + 2 * Complex.i) * (2 - Complex.i),
  have : z = 4 + 3 * Complex.i := by norm_num [Complex.mul],
  rw this,
  exact rfl,
}

end imaginary_part_of_z_l592_592213


namespace jackson_collection_goal_l592_592289

theorem jackson_collection_goal 
  (days_in_week : ℕ)
  (goal : ℕ)
  (earned_mon : ℕ)
  (earned_tue : ℕ)
  (avg_collect_per_4house : ℕ)
  (remaining_days : ℕ)
  (remaining_goal : ℕ)
  (daily_target : ℕ)
  (collect_per_house : ℚ)
  :
  days_in_week = 5 →
  goal = 1000 →
  earned_mon = 300 →
  earned_tue = 40 →
  avg_collect_per_4house = 10 →
  remaining_goal = goal - earned_mon - earned_tue →
  remaining_days = days_in_week - 2 →
  daily_target = remaining_goal / remaining_days →
  collect_per_house = avg_collect_per_4house / 4 →
  (daily_target : ℚ) / collect_per_house = 88 := 
by sorry

end jackson_collection_goal_l592_592289


namespace maximize_cone_surface_area_l592_592188

noncomputable def cone_surface_area_maximized_ratio (V : ℝ) : Prop :=
  ∀ (R H : ℝ), (H / R = 1) ↔ let r := R in let h := H in (1 / 3 * π * r ^ 2 * h = V)

theorem maximize_cone_surface_area (V : ℝ) : cone_surface_area_maximized_ratio V :=
by
  sorry

end maximize_cone_surface_area_l592_592188


namespace money_at_fair_l592_592072

theorem money_at_fair (M L D : ℕ) (hL : L = 16) (hD : D = 71) (h : M - L = D) : M = 87 :=
by
  rw [hL, hD] at h
  rw ← add_eq_of_eq_sub' h
  exact eq.symm (by simp)

end money_at_fair_l592_592072


namespace percentage_of_failed_candidates_l592_592267

theorem percentage_of_failed_candidates :
  let total_candidates := 2000
  let girls := 900
  let boys := total_candidates - girls
  let boys_passed := 32 / 100 * boys
  let girls_passed := 32 / 100 * girls
  let total_passed := boys_passed + girls_passed
  let total_failed := total_candidates - total_passed
  let percentage_failed := (total_failed / total_candidates) * 100
  percentage_failed = 68 :=
by
  -- Proof goes here
  sorry

end percentage_of_failed_candidates_l592_592267


namespace number_of_houses_in_block_l592_592809

theorem number_of_houses_in_block (pieces_per_house pieces_per_block : ℕ) (h1 : pieces_per_house = 32) (h2 : pieces_per_block = 640) :
  pieces_per_block / pieces_per_house = 20 :=
by
  sorry

end number_of_houses_in_block_l592_592809


namespace angela_action_figures_l592_592115

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l592_592115


namespace max_zeros_in_table_l592_592261

instance : DecidableEq ℕ := Classical.decEq _

-- Definition of our table
def isDistinctSums (matrix : List (List ℕ)) : Prop :=
  let rowSums := matrix.map (λ row => List.sum row)
  let columnSums := (List.transpose matrix).map (λ col => List.sum col)
  (rowSums ++ columnSums).nodup

theorem max_zeros_in_table (matrix : List (List ℕ)) (h_size : matrix.length = 3) (h_width : (∀ row ∈ matrix, row.length = 4)) 
  (h_distinct : isDistinctSums matrix) : ∃ zeros, zeros = 8 :=
by
  sorry

end max_zeros_in_table_l592_592261


namespace lara_flowers_in_vase_l592_592709

theorem lara_flowers_in_vase:
  ∀ (total_stems mom_flowers extra_flowers: ℕ),
  total_stems = 52 →
  mom_flowers = 15 →
  extra_flowers = 6 →
  let grandma_flowers := mom_flowers + extra_flowers in
  let given_away := mom_flowers + grandma_flowers in
  let in_vase := total_stems - given_away in
  in_vase = 16 :=
by
  intros total_stems mom_flowers extra_flowers
  intros h1 h2 h3
  let grandma_flowers := mom_flowers + extra_flowers
  let given_away := mom_flowers + grandma_flowers
  let in_vase := total_stems - given_away
  rw [h1, h2, h3]
  exact sorry

end lara_flowers_in_vase_l592_592709


namespace tank_capacity_calculation_l592_592705

theorem tank_capacity_calculation :
  ∀ (balloons : ℕ) (liters_per_balloon : ℕ) (tanks : ℕ) (total_liters : ℕ) (tank_capacity : ℕ),
  balloons = 1000 →
  liters_per_balloon = 10 →
  tanks = 20 →
  total_liters = balloons * liters_per_balloon →
  tank_capacity = total_liters / tanks →
  tank_capacity = 500 :=
by
  intros balloons liters_per_balloon tanks total_liters tank_capacity
  intro h_balloon_count
  intro h_liters_per_balloon
  intro h_tank_count
  intro h_total_liters
  intro h_tank_capacity
  rw [h_balloon_count, h_liters_per_balloon, h_tank_count] at *
  simp at *
  exact h_tank_capacity

end tank_capacity_calculation_l592_592705


namespace tangent_line_at_origin_is_y_eq_x_l592_592638

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + (a-1)*x^2 + a*x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem tangent_line_at_origin_is_y_eq_x
  (a : ℝ)
  (h_odd : is_odd (f x a)) :
  ∃ m : ℝ, m = 1 :=
sorry

end tangent_line_at_origin_is_y_eq_x_l592_592638


namespace male_population_half_total_l592_592402

theorem male_population_half_total (total_population : ℕ) (segments : ℕ) (male_segment : ℕ) :
  total_population = 800 ∧ segments = 4 ∧ male_segment = 1 ∧ male_segment = segments / 2 →
  total_population / 2 = 400 :=
by
  intro h
  sorry

end male_population_half_total_l592_592402


namespace find_y_l592_592149

theorem find_y (
  A B C D O : Point,
  distAO : dist A O = 7,
  distBO : dist B O = 6,
  distCO : dist C O = 12,
  distDO : dist D O = 9,
  angleAOC_eq_angleBOD : ∠A O C = ∠B O D) :
  ∃ y, y = Real.sqrt 203 :=
by
  have cos_phi := (7^2 + 6^2 - 9^2) / (2 * 7 * 6),
  have y_squared := 7^2 + 12^2 - 2 * 7 * 12 * cos_phi,
  exact ⟨Real.sqrt 203, by sorry⟩

end find_y_l592_592149


namespace eval_f_f_neg2_l592_592222

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x else x * x

theorem eval_f_f_neg2 : f (f (-2)) = 8 :=
by sorry

end eval_f_f_neg2_l592_592222


namespace find_x_axis_intercept_l592_592529

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end find_x_axis_intercept_l592_592529


namespace problem_solution_l592_592470

noncomputable def problem_expr : ℝ :=
  (64 + 5 * 12) / (180 / 3) + Real.sqrt 49 - 2^3 * Nat.factorial 4

theorem problem_solution : problem_expr = -182.93333333 :=
by 
  sorry

end problem_solution_l592_592470


namespace cos_48_degrees_l592_592132

noncomputable def cos_value (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem cos_48_degrees : 
  let x := cos_value 12 in
  4 * x^3 - 3 * x = (1 + Real.sqrt 5) / 4 →
  cos_value 48 = (1 / 2) * x + (Real.sqrt 3 / 2) * Real.sqrt (1 - x^2) :=
by
  sorry

end cos_48_degrees_l592_592132


namespace arccos_neg_one_eq_pi_l592_592545

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l592_592545


namespace sin_315_degree_l592_592962

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592962


namespace polynomial_degree_l592_592561

variable (b c d e f g : ℝ) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) (d_nonzero : d ≠ 0) 
          (e_nonzero : e ≠ 0) (f_nonzero : f ≠ 0) (g_nonzero : g ≠ 0)
variable (x y : ℝ)

theorem polynomial_degree :
  polynomial.degree ((polynomial.C (b*x^8 + c*x^2 + d) * polynomial.C (x^4 + e*x^3 + f*y^2) * polynomial.C (x + g)):= (polynomial.degree (polynomial.C (b*x^13))).

-- Additional conditions to turn into Lean calculus
/-
variables  
polynomial.degree(C)
polynomial. degree(C)
polynomial.degree (polynomial.C (b*x^13)) = 13
--polynomial.degree polynomial

-/
pokey sorry

end polynomial_degree_l592_592561


namespace exists_a_perfect_power_l592_592567

def is_perfect_power (n : ℕ) : Prop :=
  ∃ b k : ℕ, b > 0 ∧ k ≥ 2 ∧ n = b^k

theorem exists_a_perfect_power :
  ∃ a > 0, ∀ n, 2015 ≤ n ∧ n ≤ 2558 → is_perfect_power (n * a) :=
sorry

end exists_a_perfect_power_l592_592567


namespace intersection_A_B_l592_592646

def A : Set ℝ := {x | x * (x - 4) < 0}
def B : Set ℝ := {0, 1, 5}

theorem intersection_A_B : (A ∩ B) = {1} := by
  sorry

end intersection_A_B_l592_592646


namespace find_quadratic_eq_l592_592819

theorem find_quadratic_eq (x y : ℝ) (hx : x + y = 10) (hy : |x - y| = 12) :
    ∃ a b c : ℝ, a = 1 ∧ b = -10 ∧ c = -11 ∧ (x^2 + b * x + c = 0) ∧ (y^2 + b * y + c = 0) := by
  sorry

end find_quadratic_eq_l592_592819


namespace people_ratio_l592_592660

theorem people_ratio (pounds_coal : ℕ) (days1 : ℕ) (people1 : ℕ) (pounds_goal : ℕ) (days2 : ℕ) :
  pounds_coal = 10000 → days1 = 10 → people1 = 10 → pounds_goal = 40000 → days2 = 80 →
  (people1 * pounds_goal * days1) / (pounds_coal * days2) = 1 / 2 :=
by
  sorry

end people_ratio_l592_592660


namespace wire_division_l592_592667

theorem wire_division (initial_length : ℝ) (num_parts : ℕ) (final_length : ℝ) :
  initial_length = 69.76 ∧ num_parts = 8 ∧
  final_length = (initial_length / num_parts) / num_parts →
  final_length = 1.09 :=
by
  sorry

end wire_division_l592_592667


namespace sin_315_equals_minus_sqrt2_div_2_l592_592989

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592989


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592995

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592995


namespace find_value_of_x_l592_592721

def inf_sqrt (d : ℝ) : ℝ := d + inf_sqrt d

noncomputable def bowtie (c d : ℝ) : ℝ := c + Real.sqrt (inf_sqrt d)

theorem find_value_of_x (x : ℝ) (h : bowtie 3 x = 12) : x = 72 :=
begin
  sorry,
end

end find_value_of_x_l592_592721


namespace find_g3_l592_592779

-- Define a function g from ℝ to ℝ
variable (g : ℝ → ℝ)

-- Condition: ∀ x, g(3^x) + 2 * x * g(3^(-x)) = 3
axiom condition : ∀ x : ℝ, g (3^x) + 2 * x * g (3^(-x)) = 3

-- The theorem we need to prove
theorem find_g3 : g 3 = -3 := 
by 
  sorry

end find_g3_l592_592779


namespace perpendicular_vector_l592_592234

theorem perpendicular_vector {a : ℝ × ℝ} (h : a = (1, -2)) : ∃ (b : ℝ × ℝ), b = (2, 1) ∧ (a.1 * b.1 + a.2 * b.2 = 0) :=
by 
  sorry

end perpendicular_vector_l592_592234


namespace sum_of_digits_of_expression_l592_592039

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592039


namespace cos_48_equals_sqrt3_minus1_div2_l592_592131

noncomputable def cos24 : ℝ := (sqrt (2 + sqrt 3)) / 2
noncomputable def cos48 : ℝ := (sqrt 3 - 1) / 2

theorem cos_48_equals_sqrt3_minus1_div2 :
  cos 48 = (sqrt 3 - 1) / 2 :=
by
  let x := cos 24
  have h1 : cos 24 = cos24,
    sorry
  have h2 : cos 48 = 2 * x ^ 2 - 1,
    sorry
  have h3 : cos 72 = 4 * x ^ 3 - 3 * x,
    sorry
  have h4 : cos 72 = cos 72,
    sorry
  have h5 : cos 48 = (sqrt 3 - 1) / 2,
    sorry
  exact h5

end cos_48_equals_sqrt3_minus1_div2_l592_592131


namespace thyme_pots_count_l592_592121

theorem thyme_pots_count
  (basil_pots : ℕ := 3)
  (rosemary_pots : ℕ := 9)
  (leaves_per_basil_pot : ℕ := 4)
  (leaves_per_rosemary_pot : ℕ := 18)
  (leaves_per_thyme_pot : ℕ := 30)
  (total_leaves : ℕ := 354)
  : (total_leaves - (basil_pots * leaves_per_basil_pot + rosemary_pots * leaves_per_rosemary_pot)) / leaves_per_thyme_pot = 6 :=
by
  sorry

end thyme_pots_count_l592_592121


namespace arccos_neg_one_eq_pi_l592_592544

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l592_592544


namespace shoe_store_restock_l592_592098

-- Definitions based on given conditions
def Variance (sizes : List ℕ) : ℝ := sorry  -- Assume an arbitrary definition.
def Mode (sizes : List ℕ) : ℕ := sorry  -- Assume an arbitrary definition.
def Median (sizes : List ℕ) : ℝ := sorry  -- Assume an arbitrary definition.
def Mean (sizes : List ℕ) : ℝ := sorry  -- Assume an arbitrary definition.

-- Problem statement as a proof in Lean
theorem shoe_store_restock (sizes : List ℕ) :
  (Mode sizes = sizes.maxBy (λ x => List.filter (λ y => y = x) sizes).length) : Prop :=
sorry

end shoe_store_restock_l592_592098


namespace arccos_solution_l592_592374

theorem arccos_solution (x : ℝ) (h₁ : -1 / 3 ≤ x) (h₂ : x ≤ 1 / 3) :
  arccos (3 * x) - arccos (2 * x) = π / 6 ↔ x = -1 / 3 :=
by
  sorry

end arccos_solution_l592_592374


namespace number_of_steaks_needed_l592_592420

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end number_of_steaks_needed_l592_592420


namespace a7_value_l592_592196

def sequence (n : ℕ) : ℚ
| 0       := 2
| (n + 1) := (sequence n - 1) / (sequence n)

theorem a7_value : sequence 6 = 2 :=
sorry

end a7_value_l592_592196


namespace complex_number_satisfies_equation_l592_592219

def Z : ℂ := 1 - 1 * Complex.i

theorem complex_number_satisfies_equation : (1 + Complex.i) * Z = 2 :=
by
  -- proof here
  sorry

end complex_number_satisfies_equation_l592_592219


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592027

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592027


namespace prime_saturated_96_l592_592103

def is_prime_factor_product (n : ℕ) (prod : ℕ) : Prop :=
  (∃ (p1 p2 : ℕ), p1 * p2 = prod ∧ nat.prime p1 ∧ nat.prime p2) ∧ prod < n

theorem prime_saturated_96 :
  is_prime_factor_product 96 6 :=
by {
  sorry
}

end prime_saturated_96_l592_592103


namespace terminal_side_neg_1060_l592_592410

-- Definitions of the conditions.
def angle_mod_360 (θ : ℝ) : ℝ := θ % 360

-- The given problem as a Lean 4 statement.
theorem terminal_side_neg_1060 (θ : ℝ) : angle_mod_360 (-1060) = 20 ∧ 
  (20 < 90) :=
begin
  sorry
end

end terminal_side_neg_1060_l592_592410


namespace negation_of_universal_proposition_l592_592620

theorem negation_of_universal_proposition :
  (∀ x : ℝ, 0 < x → 2^x > 1) ↔ (∃ x : ℝ, 0 < x ∧ 2^x ≤ 1) :=
sorry

end negation_of_universal_proposition_l592_592620


namespace propA_iff_propB_l592_592427

-- Definitions for the conditions
variable (α β : Plane) (l m : Line)
hypothesis H1 : l ⊂ α
hypothesis H2 : m ⊂ α
hypothesis H3 : ¬(l ⊂ β)
hypothesis H4 : ¬(m ⊂ β)
hypothesis H5 : l ∩ m ≠ ∅

-- Proposition A: At least one of l and m intersects with β.
def PropA : Prop := l ∩ β ≠ ∅ ∨ m ∩ β ≠ ∅

-- Proposition B: Plane α intersects with β.
def PropB : Prop := α ∩ β ≠ ∅

-- Proving Proposition A is necessary and sufficient for Proposition B
theorem propA_iff_propB : PropA α β l m ↔ PropB α β l m := by
  sorry

end propA_iff_propB_l592_592427


namespace M_intersect_N_l592_592469

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def intersection (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∈ N}

theorem M_intersect_N :
  intersection M N = {x | 1 ≤ x ∧ x < 2} := 
sorry

end M_intersect_N_l592_592469


namespace arccos_neg_one_l592_592550

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l592_592550


namespace parabola_points_l592_592205

noncomputable def y1 : ℝ := -2 * (-3 - 1)^2 + 3
noncomputable def y2 : ℝ := -2 * (2 - 1)^2 + 3

theorem parabola_points:
  let y1 := -2 * (-3 - 1)^2 + 3,
      y2 := -2 * (2 - 1)^2 + 3
  in y1 < y2 ∧ y2 < 3 :=
by
  let y1 := -2 * (-3 - 1)^2 + 3
  let y2 := -2 * (2 - 1)^2 + 3
  sorry

end parabola_points_l592_592205


namespace sum_fraction_inequality_l592_592471

theorem sum_fraction_inequality : 
  \(\sum_{n=1}^{2007} \frac{1}{n^3 + 3n^2 + 2n} < \frac{1}{4}\) :=
by
  sorry

end sum_fraction_inequality_l592_592471


namespace range_of_a_cos_theta_sub_pi_six_l592_592852

theorem range_of_a (a : ℝ) 
  (h1 : (3 * a - 9 < 0))
  (h2 : (a + 2 > 0)) : 
  (-2 < a ∧ a < 3) :=
sorry

theorem cos_theta_sub_pi_six 
  (θ : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P = (- real.sqrt 3, real.sqrt 6) ∧ cos θ = - (real.sqrt 3 / 3) ∧ sin θ = (real.sqrt 6 / 3)): 
  cos (θ - (real.pi / 6)) = - 1 / 2 + real.sqrt 6 / 6 :=
sorry

end range_of_a_cos_theta_sub_pi_six_l592_592852


namespace parallelogram_AC_value_l592_592259
-- import the necessary library

-- define the main proof problem
theorem parallelogram_AC_value 
  (BA BC BD AC : ℝ) 
  (parallelogram_ABCD : true) -- we state that ABCD is a parallelogram
  (h : ℝ) 
  (h_def : AC = h)
  (cond_BA : BA = 3) 
  (cond_BC : BC = 4) 
  (cond_BD : BD = sqrt 37) :
  h = sqrt 13 := 
sorry

end parallelogram_AC_value_l592_592259


namespace problem1_problem2_l592_592128

-- Definition of logarithms and their properties might be required
noncomputable def log2 := Real.log (2 : ℝ)
noncomputable def log5 := Real.log (5 : ℝ)
noncomputable def log8 := Real.log (8 : ℝ)
noncomputable def log50 := Real.log (50 : ℝ)
noncomputable def log40 := Real.log (40 : ℝ)
noncomputable def log_sqrt2_inv4 := Real.logBase (Real.sqrt 2) (1 / 4)

-- Proof Problem 1
theorem problem1 : (log2 + log5 - log8) / (log50 - log40) = 1 := sorry

-- Proof Problem 2
theorem problem2 : (2 : ℝ)^(2 + log_sqrt2_inv4) = 1 / 4 := sorry

end problem1_problem2_l592_592128


namespace log_residue_of_f_with_contour_C_is_3_l592_592587

noncomputable
def f (z : ℂ) : ℂ := (complex.cosh z) / (complex.exp (complex.I * z) - 1)

theorem log_residue_of_f_with_contour_C_is_3 :
  ∃ (C : set ℂ), ∀ z : ℂ, abs z = 8 → (C = {z | abs z = 8} → logarithmic_residue f C = 3) :=
sorry

end log_residue_of_f_with_contour_C_is_3_l592_592587


namespace train_speed_correct_l592_592127

/-- Define the length of the train in meters -/
def length_train : ℝ := 120

/-- Define the length of the bridge in meters -/
def length_bridge : ℝ := 160

/-- Define the time taken to pass the bridge in seconds -/
def time_taken : ℝ := 25.2

/-- Define the expected speed of the train in meters per second -/
def expected_speed : ℝ := 11.1111

/-- Prove that the speed of the train is 11.1111 meters per second given conditions -/
theorem train_speed_correct :
  (length_train + length_bridge) / time_taken = expected_speed :=
by
  sorry

end train_speed_correct_l592_592127


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592998

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592998


namespace probability_region_C_l592_592096

theorem probability_region_C (P_A P_B P_C P_D : ℚ) 
  (h₁ : P_A = 1/4) 
  (h₂ : P_B = 1/3) 
  (h₃ : P_A + P_B + P_C + P_D = 1) : 
  P_C = 5/12 := 
by 
  sorry

end probability_region_C_l592_592096


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592949

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592949


namespace proposition_false_l592_592790

theorem proposition_false (a b : ℝ) (h : a + b > 0) : ¬ (a > 0 ∧ b > 0) := 
by {
  sorry -- this is a placeholder for the proof
}

end proposition_false_l592_592790


namespace largest_prime_factor_9_3_plus_8_5_minus_4_5_l592_592586

theorem largest_prime_factor_9_3_plus_8_5_minus_4_5 :
  ∃ p : ℕ, p.prime ∧ (∀ q : ℕ, q.prime ∧ q ∣ (9^3 + 8^5 - 4^5) → q ≤ p) ∧ p = 31 :=
begin
  sorry
end

end largest_prime_factor_9_3_plus_8_5_minus_4_5_l592_592586


namespace unique_n_l592_592153

noncomputable theory

open Nat

theorem unique_n (n : ℕ) : 
  (∃ p : ℕ, prime p ∧ ∃ a : ℕ, p^n - (p - 1)^n = 3^a) ↔ (n = 2) := 
  by
  sorry

end unique_n_l592_592153


namespace range_of_a_l592_592189

def f (a x : ℝ) : ℝ := Real.log (1 + a * x) - (2 * x) / (x + 2)

theorem range_of_a (a : ℝ) (h1 : 0 < a)
  (hx1 : x1 = 2 * (Real.sqrt (a * (1 - a))) / a) 
  (hx2 : x2 = -2 * (Real.sqrt (a * (1 - a))) / a)
  (h_extrema : f a x1 + f a x2 > 0) : 
  a ∈ Ioo (1 / 2) 1 :=
sorry

end range_of_a_l592_592189


namespace range_of_m_l592_592641

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x
  else Real.log x / Real.log (1 / 3)

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x ≤ (5 / 4) * m - m^2) ↔ (1 / 4 ≤ m ∧ m ≤ 1) :=
by
  sorry

end range_of_m_l592_592641


namespace range_of_a_l592_592851

variable (f : ℝ → ℝ)

-- Conditions
def is_monotonically_decreasing : Prop := ∀ x y, x < y → f x > f y
def is_odd_function : Prop := ∀ x, f (-x) = -f x
def domain_condition (a : ℝ) : Prop := 2 - a ∈ set.Ioo (-2 : ℝ) (2 : ℝ) ∧ 2 * a - 3 ∈ set.Ioo (-2 : ℝ) (2 : ℝ)

-- Main problem
theorem range_of_a (a : ℝ) (h1 : is_monotonically_decreasing f) (h2 : is_odd_function f) (h3 : domain_condition f a) :
    f (2 - a) + f (2 * a - 3) < 0 → a ∈ set.Ioo (1 : ℝ) (5 / 2 : ℝ) := sorry

end range_of_a_l592_592851


namespace homogeneous_degree_zero_l592_592446

variable {R : Type*} [CommRing R]

def algebraic_function (f : R → R → R) : Prop :=
∀ (x y : R) (k : R), k ≠ 0 → f (k * x) (k * y) = f x y

theorem homogeneous_degree_zero (f : R → R → R) (H : algebraic_function f) :
    ∀ (x y : R) (k : R), k ≠ 0 → f (k * x) (k * y) = f (x y) :=
begin
  sorry
end

end homogeneous_degree_zero_l592_592446


namespace product_invertibles_mod_120_l592_592310

theorem product_invertibles_mod_120 :
  let n := (list.filter (λ k, Nat.coprime k 120) (list.range 120)).prod
  in n % 120 = 119 :=
by
  sorry

end product_invertibles_mod_120_l592_592310


namespace find_num_cases_l592_592588

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def dice := {1, 2, 3, 4, 5, 6}

theorem find_num_cases : 
  (∃! n : ℕ, n ∈ dice ∧ is_odd n ∧ is_multiple_of_3 n) ↔ 1 := by
  sorry

end find_num_cases_l592_592588


namespace complement_A_inter_B_l592_592651

def U : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }
def A : Set ℤ := { x | x * (x - 1) = 0 }
def B : Set ℤ := { x | -1 < x ∧ x < 2 }

theorem complement_A_inter_B {U A B : Set ℤ} :
  A ⊆ U → B ⊆ U → 
  (A ∩ B) ⊆ (U ∩ A ∩ B) → 
  (U \ (A ∩ B)) = { -1, 2 } :=
by 
  sorry

end complement_A_inter_B_l592_592651


namespace mean_of_points_l592_592603

theorem mean_of_points :
  let points := [81, 73, 83, 86, 73] in
  (list.sum points : ℝ) / (list.length points : ℝ) = 79.2 :=
by
  sorry

end mean_of_points_l592_592603


namespace count_terms_with_last_digit_three_l592_592878

theorem count_terms_with_last_digit_three :
  let seq := λ n: ℕ, (7 ^ n) % 10 in
  (Finset.card (Finset.filter (λ n, seq n = 3) (Finset.range 2012))) = 503 := sorry

end count_terms_with_last_digit_three_l592_592878


namespace units_digit_sum_l592_592466

theorem units_digit_sum :
  let N := 3 ^ 1001 + 7 ^ 1002 + 13 ^ 1003 in
  (N % 10) = 9 := by
  sorry

end units_digit_sum_l592_592466


namespace trigonometric_equation_solution_l592_592840

open Real

-- Define a noncomputable function for the solution's domain of validity
noncomputable def domain_validity (t : ℝ) : Prop := 
\(\cos t \neq 0 \wedge \cos 3t \neq 0 \wedge \tan t \neq 0 \wedge \tan 3t \neq 0\)

-- Define the mathematical problem
theorem trigonometric_equation_solution (t : ℝ) :
  domain_validity t → (frac (\cos (3 * t))^3 (tan t) + frac (\cos t)^2 (tan (3 * t)) = 0 ↔ ∃ m : ℤ, t = (π / 4) * (2 * m + 1)) :=
sorry

end trigonometric_equation_solution_l592_592840


namespace alpha_plus_beta_l592_592071

def alpha (α β : ℕ) : Prop := α = 4
def beta (α β : ℕ) : Prop := β = 50

theorem alpha_plus_beta : ∃ α β : ℕ, (α = 4 ∧ β = 50) ∧ (α + β = 54) :=
by
  use 4
  use 50
  split
  { split
    { exact rfl }
    { exact rfl } }
  { norm_num }

end alpha_plus_beta_l592_592071


namespace triangle_side_bc_length_l592_592696

noncomputable def triangle_side_length : ℝ := sqrt 5

theorem triangle_side_bc_length (a b c A B C : ℝ) 
    (hA : A = π / 4) 
    (h1 : sin A + sin (B - C) = 2 * sqrt 2 * sin (2 * C))
    (area : 1/2 * b * c * sin A = 1) : 
    a = triangle_side_length := 
sorry

end triangle_side_bc_length_l592_592696


namespace optimal_feeding_program_l592_592675

theorem optimal_feeding_program :
  ∃ x y : ℝ, 
  (x + y ≥ 4.5) ∧ 
  (x + 2y ≥ 6) ∧ 
  (y ≥ 1) ∧ 
  (30 * x + 120 * y = 240) ∧ 
  (3 * x + 24 * y = 36) := 
by
  use 4, 1
  split
  { norm_num } -- (4 + 1 >= 4.5)
  split
  { norm_num } -- (4 + 2*1 >= 6)
  split
  { norm_num } -- (1 >= 1)
  split
  { norm_num } -- (30 * 4 + 120 * 1 = 240)
  { norm_num } -- (3 * 4 + 24 * 1 = 36)

end optimal_feeding_program_l592_592675


namespace polynomial_square_b_value_l592_592785

theorem polynomial_square_b_value (a b : ℚ) :
  (∃ (A B : ℚ), 2 * A = 1 ∧ A^2 + 2 * B = 2 ∧ 2 * A * B = a ∧ B^2 = b) → b = 49 / 64 :=
by
  intros,
  sorry

end polynomial_square_b_value_l592_592785


namespace curve_scaling_transformation_l592_592278

theorem curve_scaling_transformation :
  ∀ (x y : ℝ),
  (∃ x' y' : ℝ, x' = 5 * x ∧ y' = 3 * y ∧ x'^2 + y'^2 = 1) →
  25 * x^2 + 9 * y^2 = 1 :=
by
  intros x y h
  cases h with x' hx
  cases hx with y' hy
  cases hy with hx' hy'
  cases hy' with hy' h_curve
  sorry

end curve_scaling_transformation_l592_592278


namespace valid_pairs_l592_592141

open Nat

theorem valid_pairs (a p : ℕ) (hp : Prime p) (ha_pos : a > 0) (hp_pos : p > 0)
  (h : a % p + a % (2 * p) + a % (3 * p) + a % (4 * p) = a + p) :
  (a = 3 * p ∨ (a = 1 ∧ p = 3) ∨ (a = 17 ∧ p = 3)) :=
sorry

end valid_pairs_l592_592141


namespace general_term_l592_592731

-- Definitions of the sequences and conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ}
variable (d : ℝ)
hypothesis h1 : d ≠ 0
hypothesis h2 : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * d
hypothesis h3 : ∀ n, b n = sqrt (8 * S n + 2 * n)
hypothesis h4 : ∀ n, b (n + 1) - b n = d

-- The statement of the problem to prove
theorem general_term (n : ℕ) : a n = 4 * n - 9 / 4 :=
by
  -- proof steps can be filled in here
  sorry

end general_term_l592_592731


namespace sum_of_digits_of_expression_l592_592034

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592034


namespace sin_315_eq_neg_sqrt2_div_2_l592_592920

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592920


namespace triminos_vs_dominoes_l592_592190

theorem triminos_vs_dominoes (n : ℕ) : 
  (number_of_ways_triminos (3 * n) > number_of_ways_dominoes (2 * n)) := 
sorry

end triminos_vs_dominoes_l592_592190


namespace expected_books_total_l592_592871

-- Define the sets of symmetric and non-symmetric letters
def symmetricLetters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'H', 'I', 'M', 'O', 'T', 'U', 'V', 'W', 'X', 'Y'}
def nonSymmetricLetters : Finset Char := {'F', 'G', 'J', 'K', 'L', 'N', 'P', 'Q', 'R', 'S', 'Z'}

-- Define the number of symmetric and non-symmetric letters
def numSymmetricLetters : ℕ := symmetricLetters.card
def numNonSymmetricLetters : ℕ := nonSymmetricLetters.card

-- Define the probability of taking a book from symmetric and non-symmetric letters
def probSymmetric : ℚ := 1 / 2
def probNonSymmetric : ℚ := 1 / 4

-- Define the expected number of books taken from symmetric and non-symmetric letters
def expectedSymmetric : ℚ := numSymmetricLetters * probSymmetric
def expectedNonSymmetric : ℚ := numNonSymmetricLetters * probNonSymmetric

-- The expected total number of books taken
def expectedTotalBooks : ℚ := expectedSymmetric + expectedNonSymmetric

-- The main theorem stating the expected number of books taken
theorem expected_books_total : expectedTotalBooks = 41 / 4 :=
by
  have h1 : numSymmetricLetters = 15 := by sorry
  have h2 : numNonSymmetricLetters = 11 := by sorry
  have h3 : expectedSymmetric = 15 * (1 / 2) := by sorry
  have h4 : expectedNonSymmetric = 11 * (1 / 4) := by sorry
  have h5 : expectedTotalBooks = (15 * (1 / 2)) + (11 * (1 / 4)) := by sorry
  have h6 : (15 * (1 / 2)) = 15 / 2 := by sorry
  have h7 : (11 * (1 / 4)) = 11 / 4 := by sorry
  have h8 : (15 / 2) + (11 / 4) = 30 / 4 + 11 / 4 := by sorry
  have h9 : 30 / 4 + 11 / 4 = 41 / 4 := by sorry
  exact h9

end expected_books_total_l592_592871


namespace sin_315_degree_l592_592963

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592963


namespace correct_ordering_of_f_l592_592727
noncomputable def f (x : ℝ) := x^2 - real.pi * x
def alpha := real.arcsin (1/3)
def beta := real.arctan (5/4)
def gamma := real.arccos (-1/3)
def delta := real.arccot (-5/4)

theorem correct_ordering_of_f :
  f alpha > f delta ∧ f delta > f beta ∧ f beta > f gamma :=
sorry

end correct_ordering_of_f_l592_592727


namespace sin_315_eq_neg_sqrt2_div_2_l592_592923

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592923


namespace compute_multiplied_difference_l592_592552

theorem compute_multiplied_difference (a b : ℕ) (h_a : a = 25) (h_b : b = 15) :
  3 * ((a + b) ^ 2 - (a - b) ^ 2) = 4500 := by
  sorry

end compute_multiplied_difference_l592_592552


namespace probability_of_MD1_geq_2ME_l592_592354

noncomputable def geometric_probability (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 ≥ 4 * ((x - 3)^2 + (y - 3)^2 + (z - 3)^2)

theorem probability_of_MD1_geq_2ME : 
  let edge_length := 4
  let center := (4, 4, 4)
  let radius := 2 * sqrt 3
  let volume_cube := edge_length ^ 3
  let volume_sphere_octant := (1 / 8) * (4 / 3 * π * radius ^ 3)
  volume_sphere_octant / volume_cube = sqrt 3 * π / 16 := sorry

end probability_of_MD1_geq_2ME_l592_592354


namespace black_pens_removed_l592_592498

theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
    (blue_removed : ℕ) (pens_left : ℕ)
    (h_initial_pens : initial_blue = 9 ∧ initial_black = 21 ∧ initial_red = 6)
    (h_blue_removed : blue_removed = 4)
    (h_pens_left : pens_left = 25) :
    initial_blue + initial_black + initial_red - blue_removed - (initial_blue + initial_black + initial_red - blue_removed - pens_left) = 7 :=
by
  rcases h_initial_pens with ⟨h_ib, h_ibl, h_ir⟩
  simp [h_ib, h_ibl, h_ir, h_blue_removed, h_pens_left]
  sorry

end black_pens_removed_l592_592498


namespace product_of_first_three_terms_of_arithmetic_sequence_l592_592796

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l592_592796


namespace honey_contains_20_percent_water_l592_592073

theorem honey_contains_20_percent_water :
  ∀ (total_nectar weight_honey weight_water percentage_water_nectar : ℝ),
  total_nectar = 1.6 →
  weight_honey = 1 → 
  percentage_water_nectar = 50 →
  let initial_water := total_nectar * (percentage_water_nectar / 100),
      solid_fraction := total_nectar - initial_water,
      water_removed := total_nectar - weight_honey,
      remaining_water := initial_water - water_removed in
  (remaining_water / weight_honey) * 100 = 20 := 
by
  intros total_nectar weight_honey weight_water percentage_water_nectar h1 h2 h3 
  let initial_water := total_nectar * (percentage_water_nectar / 100)
  let solid_fraction := total_nectar - initial_water
  let water_removed := total_nectar - weight_honey
  let remaining_water := initial_water - water_removed
  sorry

end honey_contains_20_percent_water_l592_592073


namespace sum_of_f_l592_592618

def f (x : ℝ) : ℝ := sorry

variables (a : ℝ) (f_properties : ∀ x : ℝ, f x = -f (-x) ∧ f (x + 2) = f (-x - 2)) (h₁ : f 1 = a)

theorem sum_of_f :
  f 1 + f 3 + f 5 + ... + f 2019 = 2 * a :=
sorry

end sum_of_f_l592_592618


namespace imaginary_part_is_minus_one_l592_592614

def z : ℂ := (1 - complex.i) / (complex.i ^ 3)
def conjugate_z : ℂ := conj z
def imaginary_part_conjugate_z : ℝ := complex.im conjugate_z

theorem imaginary_part_is_minus_one : imaginary_part_conjugate_z = -1 :=
  sorry

end imaginary_part_is_minus_one_l592_592614


namespace algebraic_expression_value_l592_592246

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : 2*a^2 - 4*a + 2023 = 2025 :=
sorry

end algebraic_expression_value_l592_592246


namespace john_and_linda_upward_distance_l592_592503

open Real -- Open the Real namespace for convenience

def john_location : ℝ × ℝ := (3, -15)
def linda_location : ℝ × ℝ := (-2, 20)
def maria_location : ℝ × ℝ := (0.5, 5)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def upward_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  p2.2 - p1.2

theorem john_and_linda_upward_distance :
  upward_distance (midpoint john_location linda_location) maria_location = 2.5 :=
by
  sorry

end john_and_linda_upward_distance_l592_592503


namespace sin_315_degree_l592_592951

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592951


namespace unique_cd_exists_l592_592361

open Real

theorem unique_cd_exists (h₀ : 0 < π / 2):
  ∃! (c d : ℝ), (0 < c) ∧ (c < π / 2) ∧ (0 < d) ∧ (d < π / 2) ∧ (c < d) ∧ 
  (sin (cos c) = c) ∧ (cos (sin d) = d) := sorry

end unique_cd_exists_l592_592361


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592928

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592928


namespace complex_sum_example_l592_592166

open Complex

theorem complex_sum_example :
  (∑ k in Finset.range 33, (i^k * cos (30 + 90 * k) + i^(k+1) * sin (120 + 90 * k%4))) = (Complex.mk (17 * Real.sqrt 3 / 2) 8) :=
by
  sorry

end complex_sum_example_l592_592166


namespace correct_operation_l592_592074

theorem correct_operation :
  (∀ a : ℝ, (a^4)^2 ≠ a^6) ∧
  (∀ a b : ℝ, (a - b)^2 ≠ a^2 - ab + b^2) ∧
  (∀ a b : ℝ, 6 * a^2 * b / (2 * a * b) = 3 * a) ∧
  (∀ a : ℝ, a^2 + a^4 ≠ a^6) :=
by {
  sorry
}

end correct_operation_l592_592074


namespace simple_interest_principal_l592_592880

theorem simple_interest_principal (R : ℝ) (P : ℝ) (h : P * 7 * (R + 2) / 100 = P * 7 * R / 100 + 140) : P = 1000 :=
by
  sorry

end simple_interest_principal_l592_592880


namespace proof_problem_l592_592619

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 1)
noncomputable def g (x : ℝ) : ℝ := real.sin x

variables {a b c d : ℝ}

theorem proof_problem
  (h1 : a > b ∧ b ≥ 1)
  (h2 : c > d ∧ d > 0)
  (h3 : f a - f b = real.pi)
  (h4 : g c - g d = real.pi / 10) :
  a + d - b - c < 9 * real.pi / 10 := 
sorry

end proof_problem_l592_592619


namespace prove_BH_eq_BM_plus_BN_prove_B_M_O_N_concyclic_l592_592277

variables {A B C D E F G H M N O : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables [MetricSpace G] [MetricSpace H] [MetricSpace M]
variables [MetricSpace N] [MetricSpace O]
variables {R : ℝ}

-- Definitions for vertices and intersections based on given conditions
def is_cyclic_quadrilateral (A B C D : Type) : Prop :=
∃ O : Type, inscribed_circumference A B C D O

def intersection_points {X Y Z W : Type} (E F : Type) : Prop :=
opposite_sides_intersection X Y Z W E ∧ opposite_sides_intersection X Z Y W F

def is_orthocenter (H A B C : Type) : Prop :=
orthocenter_of_triangle H A B C

def similar_triangles (X Y Z W : Type) : Prop :=
similar_triangle_construction X Y Z W

-- Conditions in Lean definitions
axiom quad_inscribed_circle : is_cyclic_quadrilateral A B C D
axiom points_intersections : intersection_points A B C D E F
axiom diagonals_intersection : G = diag_intersect A C B D
axiom orthocenter_condition : is_orthocenter H A B C
axiom triangle_similarity_1 : similar_triangles B M G A H E
axiom triangle_similarity_2 : similar_triangles B N G C H F

-- The propositions to prove
theorem prove_BH_eq_BM_plus_BN : BH = BM + BN :=
sorry

theorem prove_B_M_O_N_concyclic : cyclic (B, M, O, N) :=
sorry

end prove_BH_eq_BM_plus_BN_prove_B_M_O_N_concyclic_l592_592277


namespace bar_charts_as_line_charts_l592_592531

-- Given that line charts help to visualize trends of increase and decrease
axiom trends_visualization (L : Type) : Prop

-- Bar charts can be drawn as line charts, which helps in visualizing trends
theorem bar_charts_as_line_charts (L B : Type) (h : trends_visualization L) : trends_visualization B := sorry

end bar_charts_as_line_charts_l592_592531


namespace regular_polygon_proof_l592_592010

theorem regular_polygon_proof (k m n : ℕ) (h1 : 2 < k)
  (h2 : ∀ i, i ∈ ((finset.range k).map (λ i, rotate_right i (equilateral_polygons n m))).val)
  : (m = 3 ∧ n = k) ∨ (m = 4 ∧ k = 6 ∧ n = 12) :=
sorry

end regular_polygon_proof_l592_592010


namespace remaining_cookies_l592_592747

theorem remaining_cookies : 
  let naomi_cookies := 53
  let oliver_cookies := 67
  let penelope_cookies := 29
  let total_cookies := naomi_cookies + oliver_cookies + penelope_cookies
  let package_size := 15
  total_cookies % package_size = 14 :=
by
  sorry

end remaining_cookies_l592_592747


namespace solve_for_a_l592_592184

def f : ℝ → ℝ :=
  λ x, if x < 0 then -2 * x else x^2 - 1

def equation_holds (a : ℝ) : Prop :=
  ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f x1 + 2 * sqrt (1 - x1^2) + abs (f x1 - 2 * sqrt (1 - x1^2)) - 2 * a * x1 - 4 = 0 ∧
                    f x2 + 2 * sqrt (1 - x2^2) + abs (f x2 - 2 * sqrt (1 - x2^2)) - 2 * a * x2 - 4 = 0 ∧
                    f x3 + 2 * sqrt (1 - x3^2) + abs (f x3 - 2 * sqrt (1 - x3^2)) - 2 * a * x3 - 4 = 0 ∧
                    x3 - x2 = 2 * (x2 - x1)

theorem solve_for_a : ∃ a : ℝ, equation_holds a ∧ a = (-3 + sqrt 17) / 2 :=
by
  sorry

end solve_for_a_l592_592184


namespace at_most_6_large_faces_l592_592104
open EuclideanSpace 

noncomputable theory

-- Define the sphere and polyhedron
def circumscribed_around_sphere (P : Polyhedron) (S : Sphere) : Prop := sorry
def large_face (F : Face) (S : Sphere) : Prop := sorry

-- Define the polyhedron P and sphere S as given
variables (P : Polyhedron) (S : Sphere)

-- Assume P is circumscribed around S
variable (h1 : circumscribed_around_sphere P S)

-- The theorem statement: there are at most 6 large faces
theorem at_most_6_large_faces : 
  ∑ (F : Face) in P.faces, large_face F S → P.faces.card ≤ 6 := sorry

end at_most_6_large_faces_l592_592104


namespace sum_of_prime_divisors_1800_l592_592832

theorem sum_of_prime_divisors_1800 : 
  let n := 1800
  prime_factors n = {2, 3, 5} → 
  (2 + 3 + 5 = 10) := 
by
  intros n h_prime_factors
  sorry

end sum_of_prime_divisors_1800_l592_592832


namespace sum_of_digits_of_expression_l592_592038

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592038


namespace product_of_coprimes_mod_120_l592_592317

open Nat

noncomputable def factorial_5 : ℕ := 5!

def is_coprime_to_120 (x : ℕ) : Prop := gcd x 120 = 1

def coprimes_less_than_120 : List ℕ :=
  (List.range 120).filter is_coprime_to_120

def product_of_coprimes_less_than_120 : ℕ :=
  coprimes_less_than_120.foldl (*) 1

theorem product_of_coprimes_mod_120 : 
  (product_of_coprimes_less_than_120 % 120) = 1 :=
sorry

end product_of_coprimes_mod_120_l592_592317


namespace find_equations_of_parabola_and_hyperbola_l592_592468

theorem find_equations_of_parabola_and_hyperbola
        (vertex_origin : ∀ (x y : ℝ), y^2 = 2 * p * x)
        (directrix_through_focus : ∀ (x y : ℝ), x^2 - y^2 = 1)
        (directrix_perpendicular_real_axis : ∀ (x y : ℝ), x y = 0)
        (intersection_M : ∀ (x y : ℝ), (x, y) ∈ intersect_set (λ z, z^2 = 4*z) (λ z, 4*(z.1)^2 - z.2^2 = 1))
        : (∀ (x y : ℝ), y^2 = 4 * x) ∧ (∀ (x y : ℝ), 4 * x^2 - y^2 = 1) :=
by
    sorry

end find_equations_of_parabola_and_hyperbola_l592_592468


namespace complement_intersection_l592_592718

-- Define the sets
def I : set ℕ := {1, 2, 3, 4, 5}
def A : set ℕ := {1, 2}
def B : set ℕ := {1, 3, 5}

-- Define the complement of A in I
def C_I_A : set ℕ := I \ A

-- The main theorem stating the desired equality
theorem complement_intersection :
  C_I_A ∩ B = {3, 5} := by
-- Placeholder for the proof
sorry

end complement_intersection_l592_592718


namespace range_of_m_l592_592253

theorem range_of_m (m : ℝ) : 
    (∀ x y : ℝ, (x^2 / (4 - m) + y^2 / (m - 3) = 1) → 
    4 - m > 0 ∧ m - 3 > 0 ∧ m - 3 > 4 - m) → 
    (7/2 < m ∧ m < 4) :=
sorry

end range_of_m_l592_592253


namespace find_list_price_l592_592882

noncomputable def list_price (x : ℝ) (alice_price_diff bob_price_diff : ℝ) (alice_comm_fraction bob_comm_fraction : ℝ) : Prop :=
  alice_comm_fraction * (x - alice_price_diff) = bob_comm_fraction * (x - bob_price_diff)

theorem find_list_price : list_price 40 15 25 0.15 0.25 :=
by
  sorry

end find_list_price_l592_592882


namespace range_of_a_l592_592233

def condition1 (a : ℝ) : Prop := (2 - a) ^ 2 < 1
def condition2 (a : ℝ) : Prop := (3 - a) ^ 2 ≥ 1

theorem range_of_a (a : ℝ) (h1 : condition1 a) (h2 : condition2 a) :
  1 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l592_592233


namespace problem1_problem2_l592_592605

-- Proof Problem 1: Statement
theorem problem1 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hTα : tan α = 2) : 
  (2 * sin α - cos (π - α)) / (3 * sin α - sin (π / 2 + α)) = 1 := 
sorry

-- Proof Problem 2: Statement
theorem problem2 (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hTα : tan α = 2) (hSinDiff : sin (α - β) = 3 / 5) : 
  cos β = 2 * sqrt 5 / 5 :=
sorry

end problem1_problem2_l592_592605


namespace find_subtracted_value_l592_592872

theorem find_subtracted_value (n : ℕ) (h₁ : n = 848) (h₂ : ∃ x : ℕ, (n / 8) - x = 6) : ∃ x : ℕ, x = 100 :=
by
  have h₃ : n / 8 = 106 := by rw [h₁]; norm_num
  obtain ⟨x, h₄⟩ := h₂
  use x
  rw [h₃] at h₄
  linarith

end find_subtracted_value_l592_592872


namespace total_spent_l592_592301

-- Define the prices of individual items
def cost_barrette : ℕ := 4
def cost_comb : ℕ := 2
def cost_hairband : ℕ := 3
def cost_hairtie : ℕ := 2.5

-- Define the quantities purchased by Kristine and Crystal
def kristine_items : ℕ := 1 + 1 + 2  -- 1 set of barrettes, 1 comb, 2 hairbands
def crystal_items : ℕ := 3 + 1 + 2  -- 3 sets of barrettes, 1 comb, 2 packs of hair ties

-- Calculate the total cost before discount and tax
def kristine_total : ℕ := cost_barrette + cost_comb + (2 * cost_hairband)
def crystal_total : ℕ := (3 * cost_barrette) + cost_comb + (2 * cost_hairtie)

-- Discount condition
def discount_threshold : ℕ := 5
def discount_rate : ℝ := 0.15

-- Sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Final amount to spend after applying discount and tax
def final_amount : ℝ :=
  let kristine_total_af := krisine_total
  let kristine_tax := kristine_total_af * sales_tax_rate
  let kristine_total_after_tax := kristine_total_af + kristine_tax

  let crystal_discount := if crystal_items > discount_threshold then crystal_total * discount_rate else 0
  let crystal_total_af := crystal_total - crystal_discount
  let crystal_tax := crystal_total_af * sales_tax_rate
  let crystal_total_after_tax := crystal_total_af + crystal_tax

  kristine_total_after_tax + crystal_total_after_tax

-- Prove that the combined total is $30.40
theorem total_spent : final_amount = 30.40 :=
  sorry

end total_spent_l592_592301


namespace max_value_A_l592_592089

theorem max_value_A (n : ℕ) (x y : Fin n → ℝ) 
  (h_condition : (∑ i, (x i) ^ 2) + (∑ i, (y i) ^ 2) ≤ 1) :
  let sum_x := ∑ i in Finset.range n, x i
  let sum_y := ∑ i in Finset.range n, y i
  let A := (3 * sum_x - 5 * sum_y) * (5 * sum_x + 3 * sum_y)
in A ≤ 17 * n ∧ A = 17 * n := sorry

end max_value_A_l592_592089


namespace pentagon_area_l592_592589

-- Define the coordinates of the vertices of the pentagon
def A := (0, 0) : Real × Real
def B := (20, 0) : Real × Real
def C := (40, 0) : Real × Real
def D := (50, 30) : Real × Real
def E := (20, 50) : Real × Real
def F := (0, 40) : Real × Real

-- Define a theorem to state the area of the shaded region
theorem pentagon_area : 
  let pentagon := [A, B, D, E, F]
  area pentagon = 2150 := 
  sorry

end pentagon_area_l592_592589


namespace part1_part2_l592_592230

-- Define the function, assumptions, and the proof for the first part
theorem part1 (m : ℝ) (x : ℝ) :
  (∀ x > 1, -m * (0 * x + 1) * Real.log x + x - 0 ≥ 0) →
  m ≤ Real.exp 1 := sorry

-- Define the function, assumptions, and the proof for the second part
theorem part2 (x : ℝ) :
  (∀ x > 0, (x - 1) * (-(x + 1) * Real.log x + x - 1) ≤ 0) := sorry

end part1_part2_l592_592230


namespace pyramid_equation_l592_592411

-- Define points A, B, C, D, E, F, G in the appropriate space
variables {A B C D E F G : Point}

-- Conditions
axiom midpoint_E : midpoint B C E
axiom midpoint_F : midpoint C A F
axiom midpoint_G : midpoint A B G

-- Pyramid properties and distances to be proved
theorem pyramid_equation
  (h1 : distance D A ^ 2 - distance D B ^ 2 + distance D C ^ 2
          + distance A E ^ 2 + distance B F ^ 2 + distance C G ^ 2 =
        distance D E ^ 2 + distance D F ^ 2 + distance D G ^ 2
          + 4 * (distance E F ^ 2 + distance F G ^ 2 + distance G E ^ 2)) :
  true :=
begin
  sorry,
end

end pyramid_equation_l592_592411


namespace angle_DCA_in_triangle_ABC_l592_592670

theorem angle_DCA_in_triangle_ABC
  (A B C D : Type)
  [triangle ABC]
  [on_side D AB]
  : ∠B = π / 3 
  → BD = 1 
  → AC = √3 
  → DA = DC 
  → (∠ DCA = π / 6 ∨ ∠ DCA = π / 18) :=
by 
  -- All conditions are introduced as hypotheses.
  intro hB hBD hAC hDA_DC 
  -- Proof omitted.
  sorry

end angle_DCA_in_triangle_ABC_l592_592670


namespace decreasing_interval_of_f_x_minus_1_l592_592664

-- Given that the derivative of f(x) is 2x - 4
def f_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, deriv f x = 2 * x - 4

-- Prove that the function f(x-1) is decreasing on the interval (-∞, 3)
theorem decreasing_interval_of_f_x_minus_1 (f : ℝ → ℝ) (h : f_derivative f) : 
  ∀ x : ℝ, x < 3 → deriv (λ x, f (x - 1)) x < 0 :=
by
  sorry

end decreasing_interval_of_f_x_minus_1_l592_592664


namespace arithmetic_sequence_common_difference_l592_592212

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h_a30 : a 30 = 100)
  (h_a100 : a 100 = 30) :
  d = -1 := sorry

end arithmetic_sequence_common_difference_l592_592212


namespace angle_CDE_eq_45_l592_592236

universe u
noncomputable theory

open Set
open Classical
open Real
open EuclideanGeometry

variable {P Q R A B C D E : Type u}

section triangle_setup

variables {A B C D E : Point}

-- Given definitions and conditions
def triangle (A B C : Point) : Prop := ¬Collinear A B C

def is_altitude (A B C : Point) (D : Point) : Prop :=
  ∃ l : Line, LineContains (LineSegment A D) l ∧ Perpendicular l (LineContaining B C)

def is_angle_bisector (B C E : Point) : Prop :=
  AngleBisector B C E

def angle_AEB_eq_45 (A B E : Point) : Prop :=
  ∠ (Angle B A E) = Real.pi / 4

-- Prove that ∠CDE = 45 degrees
theorem angle_CDE_eq_45
  (habc : triangle A B C)
  (hAD_altitude : is_altitude A B C D)
  (hBE_bisector : is_angle_bisector B C E)
  (hAngle_AEB : angle_AEB_eq_45 A B E) :
  ∠ (Angle C D E) = Real.pi / 4 := 
sorry

end triangle_setup

end angle_CDE_eq_45_l592_592236


namespace boarders_joined_l592_592405

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ)
  (final_boarders : ℕ) (x : ℕ)
  (ratio_initial : initial_boarders * 16 = initial_day_scholars * 7)
  (ratio_final : final_boarders * 2 = initial_day_scholars)
  (final_boarders_eq : final_boarders = initial_boarders + x)
  (initial_boarders_val : initial_boarders = 560)
  (initial_day_scholars_val : initial_day_scholars = 1280)
  (final_boarders_val : final_boarders = 640) :
  x = 80 :=
by
  sorry

end boarders_joined_l592_592405


namespace ben_total_distance_walked_l592_592123

-- Definitions based on conditions
def walking_speed : ℝ := 4  -- 4 miles per hour.
def total_time : ℝ := 2  -- 2 hours.
def break_time : ℝ := 0.25  -- 0.25 hours (15 minutes).

-- Proof goal: Prove that the total distance walked is 7.0 miles.
theorem ben_total_distance_walked : (walking_speed * (total_time - break_time) = 7.0) :=
by
  sorry

end ben_total_distance_walked_l592_592123


namespace employee_B_payment_l592_592426

theorem employee_B_payment (total_payment A_payment B_payment : ℝ) 
    (h1 : total_payment = 450) 
    (h2 : A_payment = 1.5 * B_payment) 
    (h3 : total_payment = A_payment + B_payment) : 
    B_payment = 180 := 
by
  sorry

end employee_B_payment_l592_592426


namespace smallest_possible_sector_angle_l592_592706

/-- 
Prove that the smallest possible integer angle in the division of a circle 
into 16 sectors, where the central angles form an arithmetic sequence, is 15 degrees.
-/
theorem smallest_possible_sector_angle :
  ∃ (a : ℕ) (d : ℕ),
    (∀ n, 1 ≤ n ∧ n ≤ 16 → ∃ k : ℕ, k = a + (n - 1) * d) ∧
    (∑ n in Finset.range 16, (a + n * d)) = 360 ∧
    a = 15 :=
sorry

end smallest_possible_sector_angle_l592_592706


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592925

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592925


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592020

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592020


namespace perimeter_of_shaded_region_correct_l592_592683

noncomputable def perimeter_of_shaded_region : ℝ :=
  let r := 7
  let perimeter := 2 * r + (3 / 4) * (2 * Real.pi * r)
  perimeter

theorem perimeter_of_shaded_region_correct :
  perimeter_of_shaded_region = 14 + 10.5 * Real.pi :=
by
  sorry

end perimeter_of_shaded_region_correct_l592_592683


namespace purely_imaginary_condition_l592_592773

theorem purely_imaginary_condition (a b : ℝ) :
  (∃ z : ℂ, z = complex.mk 0 a ∧ (b = 0 ∧ a ≠ 0)) ↔ (b = 0 ∧ a ≠ 0) :=
by sorry

end purely_imaginary_condition_l592_592773


namespace platform_length_correct_l592_592453

noncomputable def speed_of_train_kmph := 72
noncomputable def time_to_cross_man_seconds := 20
noncomputable def time_to_cross_platform_seconds := 30

def speed_of_train_mps := (speed_of_train_kmph * 1000) / 3600

def length_of_train := speed_of_train_mps * time_to_cross_man_seconds
def distance_train_platform := speed_of_train_mps * time_to_cross_platform_seconds

def length_of_platform := distance_train_platform - length_of_train

theorem platform_length_correct :
  length_of_platform = 200 :=
by
  unfold speed_of_train_mps
  unfold length_of_train
  unfold distance_train_platform
  unfold length_of_platform
  change length_of_platform with (distance_train_platform - length_of_train)
  change distance_train_platform with (speed_of_train_mps * time_to_cross_platform_seconds)
  change length_of_train with (speed_of_train_mps * time_to_cross_man_seconds)
  have h1 : speed_of_train_mps = (72 * 1000) / 3600 := by rfl
  have h2 : length_of_train = (20 * ((72 * 1000) / 3600)) := by rw [h1]
  have h3 : distance_train_platform = (30 * ((72 * 1000) / 3600)) := by rw [h1]
  rw [h2, h3]
  norm_num


end platform_length_correct_l592_592453


namespace simplify_complex_fraction_l592_592372

theorem simplify_complex_fraction :
  (2 + 4*complex.I) / (2 - 4*complex.I) - (2 - 4*complex.I) / (2 + 4*complex.I) = 4*complex.I / 5 := by
  sorry

end simplify_complex_fraction_l592_592372


namespace number_of_two_bedroom_units_l592_592478

-- Define the total number of units and costs
variables (x y : ℕ)
def total_units := (x + y = 12)
def total_cost := (360 * x + 450 * y = 4950)

-- The target is to prove that there are 7 two-bedroom units
theorem number_of_two_bedroom_units : total_units ∧ total_cost → y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l592_592478


namespace marbles_remaining_l592_592139

def original_marbles : Nat := 64
def given_marbles : Nat := 14
def remaining_marbles : Nat := original_marbles - given_marbles

theorem marbles_remaining : remaining_marbles = 50 :=
  by
    sorry

end marbles_remaining_l592_592139


namespace coloring_scheme_formula_l592_592429

noncomputable def number_of_coloring_schemes (m n : ℕ) : ℕ :=
  if h : (m ≥ 2) ∧ (n ≥ 2) then
    m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n
  else 0

-- Formal statement verifying the formula for coloring schemes
theorem coloring_scheme_formula (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  number_of_coloring_schemes m n = m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n :=
by sorry

end coloring_scheme_formula_l592_592429


namespace reflection_segment_lengths_reflection_angles_l592_592191

-- Define the conditions using points and triangles
variables {A B C P P1 P2 P3 : Type} [In A B C P]
  
-- Assumptions based on the problem description
def triangle_in (A B C P : Type) := ∃ (P1 P2 P3 : Type), in_triangle A B C P ∧
  P1 = reflect_over_line P A B ∧
  P2 = reflect_over_line P B C ∧
  P3 = reflect_over_line P C A

-- Define the first part of the proof: segment lengths
theorem reflection_segment_lengths : 
  ∀ {A B C P P1 P2 P3 : Type} [In A B C P], 
  triangle_in A B C P → 
    (distance P1 P2 = distance P B * real.sqrt (2 * (1 - real.cos (2 * angle B)))) 
  ∧ (distance P2 P3 = distance P C * real.sqrt (2 * (1 - real.cos (2 * angle C)))) 
  ∧ (distance P3 P1 = distance P A * real.sqrt (2 * (1 - real.cos (2 * angle A)))) :=
sorry

-- Define the second part of the proof: angles between reflections
theorem reflection_angles : 
  ∀ {A B C P P1 P2 P3 : Type} [In A B C P], 
  triangle_in A B C P → 
    (angle P1 P2 P3 = angle B P C - angle A) 
  ∧ (angle P2 P3 P1 = angle C P A - angle B) 
  ∧ (angle P3 P1 P2 = angle A P B - angle C) :=
sorry

end reflection_segment_lengths_reflection_angles_l592_592191


namespace determine_location_with_coords_l592_592112

-- Define the conditions as a Lean structure
structure Location where
  longitude : ℝ
  latitude : ℝ

-- Define the specific location given in option ①
def location_118_40 : Location :=
  {longitude := 118, latitude := 40}

-- Define the theorem and its statement
theorem determine_location_with_coords :
  ∃ loc : Location, loc = location_118_40 := 
  by
  sorry -- Placeholder for the proof

end determine_location_with_coords_l592_592112


namespace sum_of_squares_l592_592768

theorem sum_of_squares :
  (∃ (b1 b2 b3 b4 b5 b6 : ℝ),
    ∀ θ : ℝ,
    cos(θ)^6 = b1 * cos(θ) + b2 * cos(2 * θ) + b3 * cos(3 * θ) + b4 * cos(4 * θ) + b5 * cos(5 * θ) + b6 * cos(6 * θ))
    → b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 = 131/128 :=
begin
  sorry
end

end sum_of_squares_l592_592768


namespace count_desired_numbers_l592_592658

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def containsDigit (n : ℕ) (d : ℕ) : Prop :=
  (n / 100 = d) ∨ (n / 10 % 10 = d) ∨ (n % 10 = d)

def isDesiredNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ containsDigit n 2 ∧ containsDigit n 3

theorem count_desired_numbers : (setOf isDesiredNumber).card = 52 :=
by
  sorry

end count_desired_numbers_l592_592658


namespace consecutive_product_solution_l592_592151

theorem consecutive_product_solution :
  ∀ (n : ℤ), (∃ a : ℤ, n^4 + 8 * n + 11 = a * (a + 1)) ↔ n = 1 :=
by
  sorry

end consecutive_product_solution_l592_592151


namespace sum_of_digits_of_expression_l592_592033

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592033


namespace percentage_error_is_94_l592_592105

theorem percentage_error_is_94 (x : ℝ) (hx : 0 < x) :
  let correct_result := 4 * x
  let error_result := x / 4
  let error := |correct_result - error_result|
  let percentage_error := (error / correct_result) * 100
  percentage_error = 93.75 := by
    sorry

end percentage_error_is_94_l592_592105


namespace sin_315_degree_l592_592956

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592956


namespace parallel_lines_condition_l592_592784

theorem parallel_lines_condition (m n : ℝ) :
  (∃x y, (m * x + y - n = 0) ∧ (x + m * y + 1 = 0)) →
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1) :=
by
  sorry

end parallel_lines_condition_l592_592784


namespace bees_leg_count_l592_592474

-- Define the number of legs per bee
def legsPerBee : Nat := 6

-- Define the number of bees
def numberOfBees : Nat := 8

-- Calculate the total number of legs for 8 bees
def totalLegsForEightBees : Nat := 48

-- The theorem statement
theorem bees_leg_count : (legsPerBee * numberOfBees) = totalLegsForEightBees := 
by
  -- Skipping the proof by using sorry
  sorry

end bees_leg_count_l592_592474


namespace infinite_solutions_sum_l592_592156

theorem infinite_solutions_sum (A B C : ℝ) 
  (hA : A = 3) 
  (hB : 3 * B + 40 = 5 * B + 24) 
  (hC : C = 5 / 3 * B) 
  (hEq : ∀ x, (x + B) * (A * x + 40) = 3 * (x + C) * (x + 8)) :
  (-(8 : ℝ) + -((40 : ℝ) / 3) = (-64 : ℝ) / 3) :=
by
  have h1 : 40 * B = 24 * C := 
    by rw [hC]; linarith
  have h2 : B = 8 := 
    by linarith [h1]
  have h3 : C = 40 / 3 := 
    by rw [hC, h2]; linarith
  rw [h2, h3]
  linarith

end infinite_solutions_sum_l592_592156


namespace diff_lines_not_parallel_perpendicular_same_plane_l592_592609

-- Variables
variables (m n : Type) (α β : Type)

-- Conditions
-- m and n are different lines, which we can assume as different types (or elements of some type).
-- α and β are different planes, which we can assume as different types (or elements of some type).
-- There exist definitions for parallel and perpendicular relationships between lines and planes.

def areParallel (x y : Type) : Prop := sorry
def arePerpendicularToSamePlane (x y : Type) : Prop := sorry

-- Theorem Statement
theorem diff_lines_not_parallel_perpendicular_same_plane
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : ¬ areParallel m n) :
  ¬ arePerpendicularToSamePlane m n :=
sorry

end diff_lines_not_parallel_perpendicular_same_plane_l592_592609


namespace highest_elevation_l592_592873

-- Define the function for elevation as per the conditions
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2

-- Prove that the highest elevation reached is 500 meters
theorem highest_elevation : (exists t : ℝ, elevation t = 500) ∧ (∀ t : ℝ, elevation t ≤ 500) := sorry

end highest_elevation_l592_592873


namespace max_value_expression_l592_592086

theorem max_value_expression (n : ℕ) (x y : Fin n → ℝ)
  (h : (∑ i, (x i)^2 + ∑ i, (y i)^2) ≤ 1) :
  let A := (3 * ∑ i, x i - 5 * ∑ i, y i) * (5 * ∑ i, x i + 3 * ∑ i, y i)
  in A ≤ 17 * n :=
sorry

end max_value_expression_l592_592086


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592030

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592030


namespace product_invertibles_mod_120_l592_592309

theorem product_invertibles_mod_120 :
  let n := (list.filter (λ k, Nat.coprime k 120) (list.range 120)).prod
  in n % 120 = 119 :=
by
  sorry

end product_invertibles_mod_120_l592_592309


namespace nurses_count_l592_592808

theorem nurses_count (total_medical_staff : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (total_ratio_parts : ℕ) (h1 : total_medical_staff = 200) 
  (h2 : ratio_doctors = 4) (h3 : ratio_nurses = 6) (h4 : total_ratio_parts = ratio_doctors + ratio_nurses) :
  (ratio_nurses * total_medical_staff) / total_ratio_parts = 120 :=
by
  sorry

end nurses_count_l592_592808


namespace calculate_difference_l592_592226

def f (x : ℝ) : ℝ := 3 ^ x

theorem calculate_difference (x : ℝ) : f (x + 1) - f x = 2 * f x := 
by
  sorry

end calculate_difference_l592_592226


namespace cistern_problem_l592_592100

noncomputable def cistern_problem_statement : Prop :=
∀ (x : ℝ),
  (1 / 5 - 1 / x = 1 / 11.25) → x = 9

theorem cistern_problem : cistern_problem_statement :=
sorry

end cistern_problem_l592_592100


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592950

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592950


namespace penguin_fish_consumption_l592_592870

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end penguin_fish_consumption_l592_592870


namespace fifty_percent_greater_l592_592082

theorem fifty_percent_greater (x : ℕ) (h : x = 88 + (88 / 2)) : x = 132 := 
by {
  sorry
}

end fifty_percent_greater_l592_592082


namespace tommy_needs_4_steaks_l592_592423

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end tommy_needs_4_steaks_l592_592423


namespace arithmetic_sequence_first_term_l592_592681

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 9) (h2 : 2 * a 3 = a 2 + 6) : a 1 = -3 :=
by
  -- a_5 = a_1 + 4d
  have h3 : a 5 = a 1 + 4 * d := sorry
  
  -- 2a_3 = a_2 + 6, which means 2 * (a_1 + 2d) = (a_1 + d) + 6
  have h4 : 2 * (a 1 + 2 * d) = (a 1 + d) + 6 := sorry
  
  -- solve the system of linear equations to find a_1 = -3
  sorry

end arithmetic_sequence_first_term_l592_592681


namespace triangle_ABC_angles_exist_l592_592697

theorem triangle_ABC_angles_exist
  (ABC : Type) [triangle ABC]
  (Q P : point ABC)
  (M N : point ABC)
  (midpoint_Q : midpoint Q A B)
  (midpoint_P : midpoint P B C)
  (MC_MB_ratio : MC / MB = 1 / 5)
  (AN_NC_ratio : AN / NC = 1 / 2) :
  (A_angle = arctan 2) ∧ (B_angle = arctan 3) ∧ (C_angle = 45) := 
sorry

end triangle_ABC_angles_exist_l592_592697


namespace man_double_son_age_in_2_years_l592_592502

def present_age_son : ℕ := 25
def age_difference : ℕ := 27
def years_to_double_age : ℕ := 2

theorem man_double_son_age_in_2_years 
  (S : ℕ := present_age_son)
  (M : ℕ := S + age_difference)
  (Y : ℕ := years_to_double_age) : 
  M + Y = 2 * (S + Y) :=
by sorry

end man_double_son_age_in_2_years_l592_592502


namespace inequality_chain_l592_592608

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end inequality_chain_l592_592608


namespace sum_even_numbers_multiple_106_l592_592408

theorem sum_even_numbers_multiple_106 (n : ℕ) (hn : n = 211) : 
  let sum := (105 / 2) * (2 + 210) in
  sum / 106 = 105 :=
by
  sorry

end sum_even_numbers_multiple_106_l592_592408


namespace sum_of_digits_l592_592057

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592057


namespace find_Q_l592_592091

variable (Q U P k : ℝ)

noncomputable def varies_directly_and_inversely : Prop :=
  P = k * (Q / U)

theorem find_Q (h : varies_directly_and_inversely Q U P k)
  (h1 : P = 6) (h2 : Q = 8) (h3 : U = 4)
  (h4 : P = 18) (h5 : U = 9) :
  Q = 54 :=
sorry

end find_Q_l592_592091


namespace spider_leg_pressure_l592_592489

-- Definitions based on given conditions
def weight_of_previous_spider := 6.4 -- ounces
def weight_multiplier := 2.5
def cross_sectional_area := 0.5 -- square inches
def number_of_legs := 8

-- Theorem stating the problem
theorem spider_leg_pressure : 
  (weight_multiplier * weight_of_previous_spider) / number_of_legs / cross_sectional_area = 4 := 
by 
  sorry

end spider_leg_pressure_l592_592489


namespace vector_length_b_l592_592217

open Real

variables {V : Type*} [inner_product_space ℝ V] {a b : V}

theorem vector_length_b {a b : V} (h₁ : real.angle a b = real.pi / 6) (h₂ : ∥a∥ = 1) (h₃ : ∥2 • a - b∥ = 1) : ∥b∥ = sqrt 3 :=
begin
  sorry
end

end vector_length_b_l592_592217


namespace midpoint_of_CD_by_intersection_and_tangency_l592_592343

-- Define the circles and points
variables {k₁ k₂ : Circle}
variables {A B C D M : Point}

-- Define the tangency and intersection conditions
variables (h₁ : k₁.intersectsAt A B) 
          (h₂ : k₂.intersectsAt A B)
          (h₃ : k₁.tangentAt C)
          (h₄ : k₂.tangentAt D)
          (h₅ : meetsAt (lineThrough A B) M t)
          (h₆ : tangent t C k₁)
          (h₇ : tangent t D k₂)
          
-- Midpoint definition
def is_midpoint (M C D : Point) : Prop := dist M C = dist M D

-- Main proposition
theorem midpoint_of_CD_by_intersection_and_tangency:
  is_midpoint M C D :=
sorry

end midpoint_of_CD_by_intersection_and_tangency_l592_592343


namespace zach_more_points_than_ben_l592_592080

theorem zach_more_points_than_ben (zach_points : ℕ) (ben_points : ℕ) (h1 : zach_points = 42) (h2 : ben_points = 21) :
(zach_points - ben_points = 21) :=
by
  rw [h1, h2]
  simp

end zach_more_points_than_ben_l592_592080


namespace initial_speed_increase_l592_592095

variables (S : ℝ) (P : ℝ)

/-- Prove that the initial percentage increase in speed P is 0.3 based on the given conditions: 
1. After the first increase by P, the speed becomes S + PS.
2. After the second increase by 10%, the final speed is (S + PS) * 1.10.
3. The total increase results in a speed that is 1.43 times the original speed S. -/
theorem initial_speed_increase (h : (S + P * S) * 1.1 = 1.43 * S) : P = 0.3 :=
sorry

end initial_speed_increase_l592_592095


namespace sin_315_eq_neg_sqrt2_div_2_l592_592921

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592921


namespace cost_of_orange_juice_l592_592171

def cost_of_bagel : ℝ := 0.95
def cost_of_sandwich : ℝ := 4.65
def cost_of_milk : ℝ := 1.15
def additional_cost_for_lunch : ℝ := 4

def cost_of_breakfast (cost_of_oj : ℝ) : ℝ := cost_of_bagel + cost_of_oj
def cost_of_lunch : ℝ := cost_of_sandwich + cost_of_milk

theorem cost_of_orange_juice :
  ∃ (cost_of_oj : ℝ), cost_of_lunch = cost_of_breakfast(cost_of_oj) + additional_cost_for_lunch ∧ cost_of_oj = 0.85 :=
by
  sorry

end cost_of_orange_juice_l592_592171


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592023

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592023


namespace simplify_trig_expression_l592_592763

theorem simplify_trig_expression (x : ℝ) : 
  √2 * Real.cos x - √6 * Real.sin x = 2 * √2 * Real.cos (π/3 + x) :=
by
  sorry

end simplify_trig_expression_l592_592763


namespace find_total_time_l592_592167

def work_rates (V : ℝ) (x1 x2 x3 x4 x5 : ℝ) : Prop := 
  (x1 + x2 + x3 = V) ∧ 
  (x2 + x4 + x5 = V) ∧ 
  (x1 + x5 = V / 2) ∧ 
  (x3 + x4 = V / 2)

def total_time (x1 x2 x3 x4 x5 : ℝ) : ℝ := 
  1 / (x1 + x2 + x3 + x4 + x5)

theorem find_total_time (V x1 x2 x3 x4 x5 : ℝ) 
  (h : work_rates V x1 x2 x3 x4 x5) :
  total_time x1 x2 x3 x4 x5 = 1 / 2 := by 
    sorry

end find_total_time_l592_592167


namespace find_f_f_eighth_l592_592183

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 2 ^ x

theorem find_f_f_eighth : f (f (1 / 8)) = 1 / 8 := by
  sorry

end find_f_f_eighth_l592_592183


namespace average_increase_l592_592093

-- Definitions
def runs_11 := 90
def avg_11 := 40

-- Conditions
def total_runs_before (A : ℕ) := A * 10
def total_runs_after (runs_11 : ℕ) (total_runs_before : ℕ) := total_runs_before + runs_11
def increased_average (avg_11 : ℕ) (avg_before : ℕ) := avg_11 = avg_before + 5

-- Theorem stating the equivalent proof problem
theorem average_increase
  (A : ℕ)
  (H1 : total_runs_after runs_11 (total_runs_before A) = 40 * 11)
  (H2 : avg_11 = 40) :
  increased_average 40 A := 
sorry

end average_increase_l592_592093


namespace largest_systematic_sample_number_l592_592255

theorem largest_systematic_sample_number : 
  ∀ (pop_size sample_size : ℕ), 
    (pop_size = 60) → 
    (sample_size = 10) → 
    (∀ (x ∈ finset.range 60), x % 6 = 3 → x + 6 * (sample_size - 1) < pop_size) → 
    ∃ max_num, max_num = 57 := 
by 
  intros pop_size sample_size pop_size_eq sample_size_eq condition;
  sorry

end largest_systematic_sample_number_l592_592255


namespace greatest_three_digit_number_l592_592829

theorem greatest_three_digit_number :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ N % 8 = 2 ∧ N % 7 = 4 ∧ N = 978 :=
by
  sorry

end greatest_three_digit_number_l592_592829


namespace factor_count_of_polynomial_l592_592834

theorem factor_count_of_polynomial : 
  ∃ f1 f2 f3 f4 : ℤ[X], (x : ℤ[X])^11 - x = f1 * f2 * f3 * f4 ∧ 
  irreducible f1 ∧ irreducible f2 ∧ irreducible f3 ∧ irreducible f4 := sorry

end factor_count_of_polynomial_l592_592834


namespace sin_315_equals_minus_sqrt2_div_2_l592_592978

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592978


namespace sin_315_equals_minus_sqrt2_div_2_l592_592981

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592981


namespace eccentricity_of_ellipse_equilateral_triangle_l592_592216

theorem eccentricity_of_ellipse_equilateral_triangle (c b a e : ℝ)
  (h1 : b = Real.sqrt (3 * c))
  (h2 : a = Real.sqrt (b^2 + c^2)) 
  (h3 : e = c / a) :
  e = 1 / 2 :=
by {
  sorry
}

end eccentricity_of_ellipse_equilateral_triangle_l592_592216


namespace avg_value_of_S_is_55_over_3_sum_of_p_and_q_is_58_l592_592172

-- Definition of the sum S for a given permutation of integers
def S (a : Fin 10 → ℕ) : ℕ :=
  (|a 0 - a 1| + |a 2 - a 3| + |a 4 - a 5| + |a 6 - a 7| + |a 8 - a 9|)

-- The main problem statement
theorem avg_value_of_S_is_55_over_3 :
  let permutations := {a : Fin 10 → ℕ // ∀ i, a i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ ∀ i j, i ≠ j → a i ≠ a j}
  let avg_value := (1 / (10.perm 10).card : ℚ) * ∑ a in permutations, S a
  sorry :=
  avg_value = 55 / 3

theorem sum_of_p_and_q_is_58 :
  let (p, q) := (55, 3)
  p + q = 58 :=
  by 
  rfl -- simple identity proof

end avg_value_of_S_is_55_over_3_sum_of_p_and_q_is_58_l592_592172


namespace sum_infinite_series_l592_592911

theorem sum_infinite_series : 
  ∑' n : ℕ, (2 * (n + 1) + 3) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2) * ((n + 1) + 3)) = 9 / 4 := by
  sorry

end sum_infinite_series_l592_592911


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592021

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592021


namespace minimum_value_inequality_l592_592729

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 :=
sorry

end minimum_value_inequality_l592_592729


namespace det_rotation_45_deg_l592_592305

open Matrix

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem det_rotation_45_deg : det (rotation_matrix (Real.pi / 4)) = 1 :=
by
  sorry

end det_rotation_45_deg_l592_592305


namespace ratio_of_areas_l592_592099

-- Define the original area of the garden
def original_area (π R : ℝ) : ℝ :=
  π * R^2

-- Define the new area of the garden after doubling the diameter
def new_area (π R : ℝ) : ℝ :=
  π * (2 * R)^2

-- State the ratio of original area to the new area
theorem ratio_of_areas (π R : ℝ) : original_area π R / new_area π R = 1 / 4 :=
by
  unfold original_area new_area
  field_simp
  ring

end ratio_of_areas_l592_592099


namespace twenty_fifth_digit_of_sum_fraction_is_zero_l592_592435

theorem twenty_fifth_digit_of_sum_fraction_is_zero :
  (∀ n, n ≥ 25 → (decimal_expansion (\frac{1}{8} + \frac{1}{5}) n = 0)) :=
by
  sorry

end twenty_fifth_digit_of_sum_fraction_is_zero_l592_592435


namespace angela_action_figures_left_l592_592120

theorem angela_action_figures_left :
  ∀ (initial_collection : ℕ), 
  initial_collection = 24 → 
  (let sold := initial_collection / 4 in
   let remaining_after_sold := initial_collection - sold in
   let given_to_daughter := remaining_after_sold / 3 in
   let remaining_after_given := remaining_after_sold - given_to_daughter in
   remaining_after_given = 12) :=
by
  intros
  sorry

end angela_action_figures_left_l592_592120


namespace initial_amount_of_money_l592_592889

-- Definitions based on conditions in a)
variables (n : ℚ) -- Bert left the house with n dollars
def after_hardware_store := (3 / 4) * n
def after_dry_cleaners := after_hardware_store - 9
def after_grocery_store := (1 / 2) * after_dry_cleaners
def after_bookstall := (2 / 3) * after_grocery_store
def after_donation := (4 / 5) * after_bookstall

-- Theorem statement
theorem initial_amount_of_money : after_donation = 27 → n = 72 :=
by
  sorry

end initial_amount_of_money_l592_592889


namespace product_first_three_terms_arithmetic_seq_l592_592794

theorem product_first_three_terms_arithmetic_seq :
  ∀ (a₇ d : ℤ), 
  a₇ = 20 → d = 2 → 
  let a₁ := a₇ - 6 * d in
  let a₂ := a₁ + d in
  let a₃ := a₂ + d in
  a₁ * a₂ * a₃ = 960 := 
by
  intros a₇ d a₇_20 d_2
  let a₁ := a₇ - 6 * d
  let a₂ := a₁ + d
  let a₃ := a₂ + d
  sorry

end product_first_three_terms_arithmetic_seq_l592_592794


namespace krikor_speed_increase_l592_592748

/--
Krikor traveled to work on two consecutive days, Monday and Tuesday, at different speeds.
Both days, he covered the same distance. On Monday, he traveled for 0.5 hours, and on
Tuesday, he traveled for \( \frac{5}{12} \) hours. Prove that the percentage increase in his speed 
from Monday to Tuesday is 20%.
-/
theorem krikor_speed_increase :
  ∀ (v1 v2 : ℝ), (0.5 * v1 = (5 / 12) * v2) → (v2 = (6 / 5) * v1) → 
  ((v2 - v1) / v1 * 100 = 20) :=
by
  -- Proof goes here
  sorry

end krikor_speed_increase_l592_592748


namespace octagon_perimeter_correct_l592_592013

noncomputable def octagon_perimeter (horizontal_vertical_length : ℝ) (diagonal_length : ℝ) : ℝ :=
  4 * horizontal_vertical_length + 4 * diagonal_length

theorem octagon_perimeter_correct :
  octagon_perimeter 3 (2 * Real.sqrt 2) = 12 + 8 * Real.sqrt 2 := by
s sorry

end octagon_perimeter_correct_l592_592013


namespace lara_flowers_in_vase_l592_592710

theorem lara_flowers_in_vase:
  ∀ (total_stems mom_flowers extra_flowers: ℕ),
  total_stems = 52 →
  mom_flowers = 15 →
  extra_flowers = 6 →
  let grandma_flowers := mom_flowers + extra_flowers in
  let given_away := mom_flowers + grandma_flowers in
  let in_vase := total_stems - given_away in
  in_vase = 16 :=
by
  intros total_stems mom_flowers extra_flowers
  intros h1 h2 h3
  let grandma_flowers := mom_flowers + extra_flowers
  let given_away := mom_flowers + grandma_flowers
  let in_vase := total_stems - given_away
  rw [h1, h2, h3]
  exact sorry

end lara_flowers_in_vase_l592_592710


namespace angela_action_figures_l592_592117

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l592_592117


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592934

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592934


namespace problem_f_f_half_l592_592308

def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then |x - 1| - 2 else 1 / (1 + x^2)

theorem problem_f_f_half : f (f (1/2)) = 4 / 13 :=
by
  sorry

end problem_f_f_half_l592_592308


namespace unique_root_l592_592090

noncomputable def circ (y z : ℝ) : ℝ := (y + z + |y - z|) / 2

def X (a : ℝ) : set ℝ := {a}

theorem unique_root (a b x : ℝ) (y z : ℝ) (hx : x ∈ X a) (hy : y ∈ X a) (hz : z ∈ X a) :
  (circ y z = y ∨ circ y z = z) →
  (circ a x = b → x = a ∧ b = a ∨ (b > a → x = b)) →
  (∃ x', circ a x' = b ∧ x' ∈ X a) :=
by
  sorry

end unique_root_l592_592090


namespace solve_PF₂_l592_592632

-- Definitions
noncomputable def ellipse_eq := ∀ (x y : ℝ), (x^2 / 9) + (y^2 / 6) = 1
def focus_distance (p f: ℝ × ℝ) : ℝ := (real.sqrt((fst p - fst f)^2 + (snd p - snd f)^2))
variable (F₁ : ℝ × ℝ)
variable (F₂ : ℝ × ℝ)
variable (P : ℝ × ℝ)

-- Conditions
axiom ellipse_condition : ellipse_eq (fst P) (snd P)
axiom focus1_distance_cond : focus_distance P F₁ = 2

-- Goal
theorem solve_PF₂ :
  focus_distance P F₂ = 4 := 
sorry

end solve_PF₂_l592_592632


namespace max_value_of_expression_l592_592725

theorem max_value_of_expression {a x1 x2 : ℝ}
  (h1 : x1^2 + a * x1 + a = 2)
  (h2 : x2^2 + a * x2 + a = 2)
  (h1_ne_x2 : x1 ≠ x2) :
  ∃ a : ℝ, (x1 - 2 * x2) * (x2 - 2 * x1) = -63 / 8 :=
by
  sorry

end max_value_of_expression_l592_592725


namespace largest_possible_a_l592_592345

theorem largest_possible_a 
  (a b c d : ℕ) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 :=
sorry

end largest_possible_a_l592_592345


namespace rational_function_solution_l592_592767

theorem rational_function_solution (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g (1 / x) + 3 * g x / x = x^3) :
  g (-3) = 135 / 4 := 
sorry

end rational_function_solution_l592_592767


namespace curve_crosses_itself_l592_592555

-- Definitions of the parametric equations
def x (t k : ℝ) : ℝ := t^2 + k
def y (t k : ℝ) : ℝ := t^3 - k * t + 5

-- The main theorem statement
theorem curve_crosses_itself (k : ℝ) (ha : ℝ) (hb : ℝ) :
  ha ≠ hb →
  x ha k = x hb k →
  y ha k = y hb k →
  k = 9 ∧ x ha k = 18 ∧ y ha k = 5 :=
by
  sorry

end curve_crosses_itself_l592_592555


namespace sin_315_degree_l592_592958

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592958


namespace probability_even_sum_of_three_fair_coins_l592_592813

-- Variables
variables {c : ℕ} {d6 d4 : ℕ}
-- Definitions based on conditions.
-- C is the number of heads resulting from tossing three fair coins.
def C := (finset.range 4).sum (λ c, if c = 0 then 1 / (2 ^ 3) else if c = 1 then 3 / (2 ^ 3) else if c = 2 then 3 / (2 ^ 3) else 1 / (2 ^ 3))

-- Definitions for the dices, 6-sided and 4-sided.
def D_6 := finset.range 6
def D_4 := finset.range 4

-- Even sum condition
def sum_even (c : ℕ) (heads : finset ℕ) (tails : finset ℕ) : Prop :=
  (finset.sum (heads ∪ tails) id) % 2 = 0

-- Probability calculation for the corresponding coin tosses and dice rolls being summed even
noncomputable def probability_even_sum : ℚ :=
  ((C.sum (λ c, if c = 0 then 5 / 8 else if c = 1 then 21 / 48 else if c = 2 then (uncertain) else 1 / 8) )) 

theorem probability_even_sum_of_three_fair_coins :
  probability_even_sum = 43 / 72 :=
sorry

end probability_even_sum_of_three_fair_coins_l592_592813


namespace area_increase_l592_592458

/-- Given the original length L and original breadth B of a rectangle, 
prove that increasing the length by 3% and the breadth by 6% increases the area by 9.18%. -/
theorem area_increase (L B : ℝ) :
  let L_new := L * 1.03,
      B_new := B * 1.06,
      A_original := L * B,
      A_new := L_new * B_new,
      increase := ((A_new - A_original) / A_original) * 100 in
  increase = 9.18 :=
by
  sorry

end area_increase_l592_592458


namespace product_tangent_intersection_l592_592737

/--
Let the tangent line at the point (1, 1) on the curve y = x^(n+1) (where n ∈ ℤ*) intersect the x-axis at the point with
the x-coordinate x_n. The value of the product x_1 * x_2 * x_3 * ... * x_n is 1/(n+1).
-/
theorem product_tangent_intersection (n : ℤ) (h : n > 0) :
  let x_k := λ k : ℤ, (k : ℚ) / (k + 1)
  in ∏ k in Finset.range(n + 1), x_k k = 1 / (n + 1) := 
by 
  sorry

end product_tangent_intersection_l592_592737


namespace find_special_primes_l592_592152

open Nat

theorem find_special_primes :
  ∃ p, (p = 3 ∨ p = 7) ∧ (Prime p) ∧ 
  (∀ k, 1 ≤ k → k ≤ (p-1) / 2 → Prime (1 + k * (p - 1))) :=
by
  sorry

end find_special_primes_l592_592152


namespace height_second_cylinder_l592_592404

-- Variables for radii and heights
variables (r1 r2 h1 h2 : ℝ)

-- Conditions
axiom radius_relation : r2 = 1.2 * r1
axiom height_first_cylinder : h1 = 15
axiom volumes_equal : π * r1^2 * h1 = π * r2^2 * h2

-- The statement to be proved
theorem height_second_cylinder : h2 = 10.5 :=
begin
  sorry
end

end height_second_cylinder_l592_592404


namespace product_invertibles_mod_120_eq_1_l592_592329

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def product_of_invertibles_mod_n (n : ℕ) :=
  List.prod (List.filter (fun x => is_coprime x n) (List.range n))

theorem product_invertibles_mod_120_eq_1 :
  product_of_invertibles_mod_n 120 % 120 = 1 := 
sorry

end product_invertibles_mod_120_eq_1_l592_592329


namespace sum_of_roots_of_polynomial_l592_592016

theorem sum_of_roots_of_polynomial :
  let f : Polynomial ℝ := Polynomial.mk [10, -14, -3, 1] in
  f.roots.sum = 3 :=
begin
  -- Sorry, as we are providing only the statement
  sorry
end

end sum_of_roots_of_polynomial_l592_592016


namespace sum_of_subsets_equal_l592_592716

theorem sum_of_subsets_equal
  (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (x : Fin m → ℕ) (y : Fin n → ℕ)
  (hx_mean : (∑ i in Finset.univ, x i) < m * (n + 1))
  (hy_mean : (∑ j in Finset.univ, y j) < n * (m + 1)) :
  ∃ (I : Finset (Fin m)) (J : Finset (Fin n)), I.nonempty ∧ J.nonempty ∧ (∑ i in I, x i) = (∑ j in J, y j) :=
by
  sorry

end sum_of_subsets_equal_l592_592716


namespace tommy_needs_4_steaks_l592_592422

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end tommy_needs_4_steaks_l592_592422


namespace sin_315_eq_neg_sqrt2_div_2_l592_592975

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592975


namespace product_of_coprimes_mod_120_l592_592319

open Nat

noncomputable def factorial_5 : ℕ := 5!

def is_coprime_to_120 (x : ℕ) : Prop := gcd x 120 = 1

def coprimes_less_than_120 : List ℕ :=
  (List.range 120).filter is_coprime_to_120

def product_of_coprimes_less_than_120 : ℕ :=
  coprimes_less_than_120.foldl (*) 1

theorem product_of_coprimes_mod_120 : 
  (product_of_coprimes_less_than_120 % 120) = 1 :=
sorry

end product_of_coprimes_mod_120_l592_592319


namespace vlad_taller_by_41_inches_l592_592824

/-- Vlad's height is 6 feet and 3 inches. -/
def vlad_height_feet : ℕ := 6

def vlad_height_inches : ℕ := 3

/-- Vlad's sister's height is 2 feet and 10 inches. -/
def sister_height_feet : ℕ := 2

def sister_height_inches : ℕ := 10

/-- There are 12 inches in a foot. -/
def inches_in_a_foot : ℕ := 12

/-- Convert height in feet and inches to total inches. -/
def convert_to_inches (feet inches : ℕ) : ℕ :=
  feet * inches_in_a_foot + inches

/-- Proof that Vlad is 41 inches taller than his sister. -/
theorem vlad_taller_by_41_inches : convert_to_inches vlad_height_feet vlad_height_inches - convert_to_inches sister_height_feet sister_height_inches = 41 :=
by
  -- Start the proof
  sorry

end vlad_taller_by_41_inches_l592_592824


namespace trajectory_eq_max_area_exists_line_l592_592629

variables {x y k : ℝ}
variables (A : (ℝ×ℝ)) (B : (ℝ×ℝ)) (P : (ℝ×ℝ)) (H : (ℝ×ℝ)) (E : (ℝ×ℝ)) (l : ℝ -> ℝ)

/-- Given point P is on the circle of radius 2 centered at the origin, and E is the midpoint
    of the perpendicular dropped from P to the x-axis with H as the foot of the perpendicular. -/
def on_circle (P : (ℝ×ℝ)) : Prop := (P.1 * P.1) + (P.2 * P.2) = 4

/-- Given E is the midpoint of the perpendicular dropped from P to the x-axis -/
def mid_point (E P H : (ℝ×ℝ)) : Prop := E = ((P.1, P.2/2) : (ℝ×ℝ))

/-- Condition for perpendicular foot H -/
def perpend_foot (P H : (ℝ×ℝ)) : Prop := H = (P.1, 0)

/-- Trajectory equation for the midpoint E -/
theorem trajectory_eq (hx : on_circle P) (e_def : mid_point E P H) (h_def : perpend_foot  P H) : 
  (E.1 * E.1 / 4) + (E.2 * E.2) = 1 :=
sorry

/-- The maximum area of triangle OAB -/
theorem max_area (B : (ℝ×ℝ)) 
  (eqB : (B.1 * B.1 / 4) + (B.2 * B.2) = 1)
  (A_def : A = (0, 1))
  : ((B.1 = 2) ∨ (B.1 = -2)) → ((1 / 2) * (abs B.1)) = 1 :=
sorry

/-- Determining k for the line equation y = kx + m with conditions -/
def line_eq (k x m : ℝ) : (ℝ × ℝ) := (x, k * x + m)

theorem exists_line (k m : ℝ) (hx : k ≠ 0) 
  (cond1 : 4*k^2 - m^2 + 1 > 0)
  (cond2 : -3 < m ∧ m < 0)
  : k ∈ set.Ioo (-real.sqrt 2) 0 ∪ set.Ioo 0 (real.sqrt 2) :=
sorry

end trajectory_eq_max_area_exists_line_l592_592629


namespace fraction_addition_l592_592896

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l592_592896


namespace sin_315_degree_l592_592960

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592960


namespace yellow_highlighters_count_l592_592258

theorem yellow_highlighters_count 
  (Y : ℕ) 
  (pink_highlighters : ℕ := Y + 7) 
  (blue_highlighters : ℕ := Y + 12) 
  (total_highlighters : ℕ := Y + pink_highlighters + blue_highlighters) : 
  total_highlighters = 40 → Y = 7 :=
by
  sorry

end yellow_highlighters_count_l592_592258


namespace bridge_trapezoid_larger_angle_l592_592114

/-- We have a bridge made up of 12 congruent isosceles trapezoids forming a circular arch. -/
def num_trapezoids : ℕ := 12

/-- Each trapezoid contributes to a 360-degree circle. -/
def circle_degrees : ℝ := 360

/-- The central angle subtended by one pair of trapezoids' legs. -/
def central_angle (n : ℝ) : ℝ := circle_degrees / n

/-- Since trapezoids are isosceles, their legs split the central angle equally. -/
def half_central_angle : ℝ := central_angle num_trapezoids / 2

/-- Calculate the smaller interior angle at the vertex. -/
def smaller_interior_angle : ℝ := 180 - half_central_angle

/-- The larger interior angle adjacent to the longer base. -/
def larger_interior_angle (θ : ℝ) : ℝ := 180 - θ

/-- Given the conditions, prove that the larger interior angle of each trapezoid is 97.5 degrees. -/
theorem bridge_trapezoid_larger_angle : larger_interior_angle (smaller_interior_angle / 2) = 97.5 :=
sorry

end bridge_trapezoid_larger_angle_l592_592114


namespace cos_48_degrees_l592_592133

noncomputable def cos_value (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem cos_48_degrees : 
  let x := cos_value 12 in
  4 * x^3 - 3 * x = (1 + Real.sqrt 5) / 4 →
  cos_value 48 = (1 / 2) * x + (Real.sqrt 3 / 2) * Real.sqrt (1 - x^2) :=
by
  sorry

end cos_48_degrees_l592_592133


namespace indochina_no_hunters_l592_592518

universe u

noncomputable def problem : Prop :=
let People := Type u in
let Weight : People → ℝ := sorry in
let BornInIndochina : Set People := sorry in
let CollectsStamps : Set People := sorry in
let HuntsBears : Set People := sorry in
let E := {p : People | p ∈ BornInIndochina ∧ p ∉ HuntsBears} in
let A := {p : People | p ∈ E ∧ p ∈ CollectsStamps} in
let B := {p : People | p ∈ E ∧ p ∉ CollectsStamps} in
all_of (∃ q : People, Weight q < 100 ∧ q ∉ CollectsStamps → q ∉ BornInIndochina) → 
(∀ r : People, r ∈ BornInIndochina ∧ r ∈ CollectsStamps → r ∈ HuntsBears) →
(∀ s : People, s ∈ BornInIndochina → s ∈ HuntsBears ∨ s ∈ CollectsStamps) →
A = ∅ ∧ B = ∅

theorem indochina_no_hunters : problem := by
  sorry

end indochina_no_hunters_l592_592518


namespace swimming_speed_in_still_water_l592_592501

variable (V_m V_s : ℝ)

-- Define the conditions
def stream_speed : ℝ := 1.6666666666666667

def time_ratio (t : ℝ) : Prop := (V_m + V_s) * t = (V_m - V_s) * 2 * t

-- Define the theorem to prove
theorem swimming_speed_in_still_water (h : time_ratio t) : V_m = 5 := by
  have h2 := congr_fun h t
  -- Here, we would normally proceed with the proof steps already known.
  sorry

end swimming_speed_in_still_water_l592_592501


namespace find_value_l592_592407

variable (number : ℝ) (V : ℝ)

theorem find_value
  (h1 : number = 8)
  (h2 : 0.75 * number + V = 8) : V = 2 := by
  sorry

end find_value_l592_592407


namespace directrix_of_parabola_l592_592582

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end directrix_of_parabola_l592_592582


namespace sin_315_eq_neg_sqrt2_div_2_l592_592971

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592971


namespace count_solutions_sin_equation_l592_592564

theorem count_solutions_sin_equation : 
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 3 * (Real.sin x)^4 - 7 * (Real.sin x)^3 + 5 * (Real.sin x)^2 - Real.sin x = 0) ∧ S.card = 4 :=
by
  sorry

end count_solutions_sin_equation_l592_592564


namespace tom_hockey_games_l592_592004

theorem tom_hockey_games (g_this_year g_last_year : ℕ) 
  (h1 : g_this_year = 4)
  (h2 : g_last_year = 9) 
  : g_this_year + g_last_year = 13 := 
by
  sorry

end tom_hockey_games_l592_592004


namespace problem_statement_l592_592649

open Set

variable (U : Set ℝ) (M N : Set ℝ)

def U := { x : ℝ | True }
def M := { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }
def N := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }

theorem problem_statement : M ∩ (U \ N) = { x : ℝ | -1 ≤ x ∧ x < 1 } := by
  sorry

end problem_statement_l592_592649


namespace shares_proportion_l592_592879

theorem shares_proportion (C D : ℕ) (h1 : D = 1500) (h2 : C = D + 500) : C / Nat.gcd C D = 4 ∧ D / Nat.gcd C D = 3 := by
  sorry

end shares_proportion_l592_592879


namespace arccos_neg_one_eq_pi_l592_592542

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l592_592542


namespace remainder_of_N_mod_1000_l592_592341

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def is_constant_function (f : ℕ → ℕ) : Prop :=
  ∃ c ∈ A, ∀ x ∈ A, f (f x) = c

noncomputable def N : ℕ :=
  8 * ∑ k in Finset.range 8 \ {0}, Nat.choose 7 k * k ^ (7 - k)

theorem remainder_of_N_mod_1000 : N % 1000 = 576 := by
  sorry

end remainder_of_N_mod_1000_l592_592341


namespace coefficient_x_80_l592_592126

noncomputable def polynomial : Polynomial ℚ :=
  (finset.range 81).sum (λ n, polynomial.x ^ n) ^ 3

theorem coefficient_x_80 : polynomial.coeff 80 polynomial = 3321 := by
  sorry

end coefficient_x_80_l592_592126


namespace product_invertible_integers_mod_120_eq_one_l592_592325

theorem product_invertible_integers_mod_120_eq_one :
  let n := ∏ i in (multiset.filter (λ x, Nat.coprime x 120) (multiset.range 120)), i
  in n % 120 = 1 := 
by
  sorry

end product_invertible_integers_mod_120_eq_one_l592_592325


namespace smaller_quadrilateral_area_l592_592568

variable (A B C D : Type) [convex_quadrilateral A] [convex_quadrilateral B] [convex_quadrilateral C] [convex_quadrilateral D]

/-- Each side of a convex quadrilateral is divided into five equal parts, 
and the corresponding points on opposite sides are connected to form a smaller quadrilateral.
Prove that the area of the smaller quadrilateral is 25 times smaller than the area of the original quadrilateral. -/
theorem smaller_quadrilateral_area (quad : convex_quadrilateral) :
  let original_area := convex_quadrilateral.area quad in
  let sub_quad := divide_and_connect quad 5 in
  convex_quadrilateral.area sub_quad = original_area / 25 :=
begin
  sorry
end

end smaller_quadrilateral_area_l592_592568


namespace exist_h_l592_592848

-- Define the set S of continuous real-valued functions
def S := {f : ℝ → ℝ // continuous f}

-- Define a linear map φ : S → S satisfying the given property
noncomputable def linear_map (φ : S → S) (H : 
  ∀ f g : S, ∀ a b : ℝ, (a < b) → (∀ x : ℝ, a < x ∧ x < b → f.1 x = g.1 x) → 
  ∀ x : ℝ, a < x ∧ x < b → (φ f).1 x = (φ g).1 x) : Prop :=
∃ h ∈ S, ∀ f : S, ∀ x : ℝ, (φ f).1 x = h.1 x * f.1 x

-- The math problem rephrased as a Lean 4 statement
theorem exist_h {φ : S → S}
  (H : ∀ f g : S, ∀ a b : ℝ, (a < b) → (∀ x : ℝ, a < x ∧ x < b → f.1 x = g.1 x) → 
  ∀ x : ℝ, a < x ∧ x < b → (φ f).1 x = (φ g).1 x) : 
  ∃ h ∈ S, ∀ f : S, ∀ x : ℝ, (φ f).1 x = h.1 x * f.1 x :=
sorry

end exist_h_l592_592848


namespace greatest_three_digit_number_l592_592828

theorem greatest_three_digit_number (n : ℕ) :
  (n % 8 = 2) ∧ (n % 7 = 4) ∧ (100 ≤ n ∧ n ≤ 999) → n = 970 :=
begin
  sorry
end

end greatest_three_digit_number_l592_592828


namespace distance_between_nails_l592_592418

theorem distance_between_nails (banner_length : ℕ) (num_nails : ℕ) (end_distance : ℕ) :
  banner_length = 20 → num_nails = 7 → end_distance = 1 → 
  (banner_length - 2 * end_distance) / (num_nails - 1) = 3 :=
by
  intros
  sorry

end distance_between_nails_l592_592418


namespace base8_addition_l592_592397

/-- Converting decimal integers to base 8 -/
def to_base_8(n : ℕ) : list ℕ :=
  if n = 0 then [0] else
    let rec digits (n : ℕ) : list ℕ :=
      if n = 0 then [] else (n % 8) :: digits (n / 8)
    digits n

/-- Adding two numbers in base 8 -/
def add_base_8 (a b : list ℕ) : list ℕ := 
  let rec add_with_carry (a b : list ℕ) (carry : ℕ) : list ℕ :=
    match a, b with
    | [], [] => if carry = 0 then [] else [carry]
    | x::xs, [] => let s := x + carry in (s % 8) :: add_with_carry xs [] (s / 8)
    | [], y::ys => let s := y + carry in (s % 8) :: add_with_carry [] ys (s / 8)
    | x::xs, y::ys => 
        let s := x + y + carry in
        (s % 8) :: add_with_carry xs ys (s / 8)
  add_with_carry a b 0

/-- The main theorem statement -/
theorem base8_addition : 
  to_base_8 624 = [0, 6, 1, 1] ∧ to_base_8 112 = [0, 6, 1] →
  add_base_8 (to_base_8 624) (to_base_8 112) = [0, 4, 3, 1] :=
by
  sorry

end base8_addition_l592_592397


namespace sin_315_eq_neg_sqrt2_div_2_l592_592972

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592972


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592019

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592019


namespace smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l592_592015

def is_not_prime (n : ℕ) := ¬ Prime n
def is_not_square (n : ℕ) := ∀ m : ℕ, m * m ≠ n
def no_prime_factors_less_than (n k : ℕ) := ∀ p : ℕ, Prime p → p < k → ¬ p ∣ n
def smallest_integer_prop (n : ℕ) := is_not_prime n ∧ is_not_square n ∧ no_prime_factors_less_than n 60

theorem smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60 : ∃ n : ℕ, smallest_integer_prop n ∧ n = 4087 :=
by
  sorry

end smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l592_592015


namespace find_x_l592_592732

def matrix_A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![1, x]]

def matrix_B : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![-1, 1]]

def matrix_C : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![-1, -2]]

theorem find_x (x : ℝ) (h : matrix_B ⬝ matrix_A x = matrix_C) : x = 2 :=
by
  sorry

end find_x_l592_592732


namespace initial_percentage_females_l592_592413

theorem initial_percentage_females (E : ℕ) (P : ℝ) 
  (h1 : E + 24 = 288) 
  (h2 : 0.55 * 288 = 158.4) 
  (h3 : E = 264) : P ≈ 59.85 :=
by sorry

end initial_percentage_females_l592_592413


namespace max_value_expression_l592_592087

theorem max_value_expression (n : ℕ) (x y : Fin n → ℝ)
  (h : (∑ i, (x i)^2 + ∑ i, (y i)^2) ≤ 1) :
  let A := (3 * ∑ i, x i - 5 * ∑ i, y i) * (5 * ∑ i, x i + 3 * ∑ i, y i)
  in A ≤ 17 * n :=
sorry

end max_value_expression_l592_592087


namespace sin_315_equals_minus_sqrt2_div_2_l592_592985

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592985


namespace minimum_queens_required_to_cover_chessboard_l592_592014

theorem minimum_queens_required_to_cover_chessboard :
  ∃ n : ℕ, n = 5 ∧ (∀ (board : array 8 (array 8 (option queen))),
    (∀ pos : fin 8 × fin 8, 
    pos.1 ≠ pos.2 → (board.read pos.1).read pos.2 ≠ none → 
    ∃ (q : queen), q.can_reach pos ∧ board.contains q) ∨
    (∀ pos : fin 8 × fin 8, board.read pos.1).read pos.2 = none)
  :=
sorry

end minimum_queens_required_to_cover_chessboard_l592_592014


namespace line_intersects_curve_l592_592680

variables (t p alpha theta : ℝ)
variables (α_pos : 0 < alpha) (α_lt_pi : alpha < π) (p_pos : 0 < p)

noncomputable def parametric_eq_line_l := (t * cos alpha, t * sin alpha)

noncomputable def polar_eq_curve_C := p / (1 - cos theta)

theorem line_intersects_curve (OA OB : ℝ) (h1 : OA = (1 + cos alpha) * p / sin(alpha) ^ 2)
  (h2 : OB = (1 - cos alpha) * p / sin(alpha) ^ 2) :
  1 / OA + 1 / OB = 2 / p :=
by
  sorry

end line_intersects_curve_l592_592680


namespace other_factor_120_l592_592472

def other_factor_of_60n (n : ℕ) (h : n ≥ 8) (d : ℕ) (h₁ : 4 * d = 60 * n) : ℕ :=
  d

theorem other_factor_120 (n : ℕ) (h : n ≥ 8) (d : ℕ) (h₁ : 4 * d = 60 * n) : d = 120 :=
by
  have h₂ : 60 * 8 = 480 := by norm_num
  have h₃ : 4 * d = 480 := by rw [←h₁, h₂, ←mul_assoc, mul_comm 60 8]
  have h₄ : d = 480 / 4 := (nat.mul_right_inj (by norm_num : 4 > 0)).mp h₃
  simp at h₄
  exact h₄

end other_factor_120_l592_592472


namespace sum_of_roots_eq_neg_one_third_l592_592735

variable {a b c d : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (h4 : a > b > c)
variable (h5 : b = a - d) (h6 : c = a - 2 * d)
variable (h7 : d > 0)

theorem sum_of_roots_eq_neg_one_third :
  let Δ := (a - d)^2 - 4 * a * (a - 2 * d) in
  Δ > 0 → - (a - d) / a = -1 / 3 := 
by
  intro h_discriminant_positive
  sorry

end sum_of_roots_eq_neg_one_third_l592_592735


namespace minimum_value_expression_l592_592339

theorem minimum_value_expression (x y : ℝ) : 
  ∃ m : ℝ, ∀ x y : ℝ, 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ m ∧ m = 3 :=
sorry

end minimum_value_expression_l592_592339


namespace true_propositions_l592_592304

variables (α β : Type) [plane α] [plane β] 
variables (l m n : Type) [line l] [line m] [line n]

-- Define parallel relation between planes
def parallel_planes (p q : Type) [plane p] [plane q] : Prop := sorry
-- Define subset relation between lines and planes
def line_in_plane (l : Type) [line l] (p : Type) [plane p] : Prop := sorry
-- Define perpendicular relation between a line and a plane
def line_perpendicular_plane (l : Type) [line l] (p : Type) [plane p] : Prop := sorry

-- The statement to be proved
theorem true_propositions :
  (parallel_planes α β ∧ line_in_plane l α → parallel_planes l β) ∧ 
  ¬(line_in_plane m α ∧ line_in_plane n α ∧ parallel_planes m β ∧ parallel_planes n β → parallel_planes α β) ∧
  (parallel_planes l α ∧ line_perpendicular_plane l β → line_perpendicular_plane α β) ∧
  ¬(line_in_plane m α ∧ line_in_plane n α ∧ line_perpendicular_plane l m ∧ line_perpendicular_plane l n → line_perpendicular_plane l α) :=
by sorry

end true_propositions_l592_592304


namespace diagonal_sequence_correct_l592_592751

noncomputable def fillGrid : ℕ → ℕ → char 
| 1, 1 := 'X'
| 1, 2 := 'X'
| 1, 3 := 'C'
| 2, 1 := 'X'
| 2, 2 := 'A'
| 2, 3 := 'C'
| 3, 1 := 'X'
| 3, 2 := 'C'
| 3, 3 := 'X'
| _, _ := 'X'

def getDiagonalSequence : list char :=
  [fillGrid 1 1, fillGrid 2 2, fillGrid 3 3, fillGrid 1 3, fillGrid 2 3, fillGrid 3 2]

theorem diagonal_sequence_correct : 
  getDiagonalSequence = ['X', 'X', 'C', 'X', 'A', 'C'] :=
begin
  simp [getDiagonalSequence, fillGrid],
end

end diagonal_sequence_correct_l592_592751


namespace banker_l592_592769

-- Define the conditions in Lean
variable (S : ℝ) -- The sum due
constant BG : ℝ := 100 -- The banker's gain is Rs. 100
constant r : ℝ := 0.25 -- Annual interest rate
constant n : ℕ := 2 -- Compounded semi-annually
constant t : ℝ := 5 -- Number of years

noncomputable def future_value (P : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

noncomputable def true_discount : ℝ :=
  S / (1 + r / n)^(n * t)

noncomputable def banker_discount (TD : ℝ) : ℝ :=
  BG + TD

theorem banker's_discount_is_correct (S : ℝ) :
  let TD := true_discount S in
  banker_discount TD = 148.75 :=
by
  sorry

end banker_l592_592769


namespace no_possible_arrangement_l592_592538

theorem no_possible_arrangement :
  ¬∃ (f : ℕ × ℕ → ℕ), (∀ (m n : ℕ), 100 < m → 100 < n → (∑ i in finset.range m, ∑ j in finset.range n, f (i, j)) % (m + n) = 0) :=
by
  sorry

end no_possible_arrangement_l592_592538


namespace chessboard_number_determination_l592_592849

theorem chessboard_number_determination (d_n : ℤ) (a_n b_n a_1 b_1 c_0 d_0 : ℤ) :
  (∀ i j : ℤ, d_n + a_n = b_n + a_1 + b_1 - (c_0 + d_0) → 
   a_n + b_n = c_0 + d_0 + d_n) →
  ∃ x : ℤ, x = a_1 + b_1 - d_n ∧ 
  x = d_n + (a_1 - c_0) + (b_1 - d_0) :=
by
  sorry

end chessboard_number_determination_l592_592849


namespace second_discount_percentage_l592_592519

def normal_price : ℝ := 49.99
def first_discount : ℝ := 0.10
def final_price : ℝ := 36.0

theorem second_discount_percentage : 
  ∃ p : ℝ, (((normal_price - (first_discount * normal_price)) - final_price) / (normal_price - (first_discount * normal_price))) * 100 = p ∧ p = 20 :=
by
  sorry

end second_discount_percentage_l592_592519


namespace fraction_addition_l592_592901

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l592_592901


namespace smallest_positive_base_ten_seven_binary_digits_l592_592443

theorem smallest_positive_base_ten_seven_binary_digits : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, (m > 0 ∧ m < n) → m.bit_length < 7) ∧ n.bit_length = 7 := 
begin
  use 64,
  split,
  { exact nat.succ_pos' 63, },
  split,
  { intros m hm,
    rw [nat.lt_succ_iff],
    apply nat.lt_of_le_of_lt (nat.bit_length_le_self m) (nat.lt_succ_self 63) },
  { exact rfl }
end

end smallest_positive_base_ten_seven_binary_digits_l592_592443


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592932

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592932


namespace composite_a2_b2_l592_592792

-- Introduce the main definitions according to the conditions stated in a)
theorem composite_a2_b2 (x1 x2 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (a b : ℤ) 
  (ha : a = -(x1 + x2)) (hb : b = x1 * x2 - 1) : 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ (a^2 + b^2) = m * n := 
by 
  sorry

end composite_a2_b2_l592_592792


namespace angle_B1MC1_eq_phi_l592_592749

open EuclideanGeometry

variables {A B C C1 B1 A1 M : Point}
variables {AB AC : Real}
variables {φ : Real}

-- Conditions
constant h_ABC : Triangle A B C
constant h_AB_ne_AC : AB ≠ AC
constant h_isosceles_AC1B : isosceles_triangle A C1 B φ 
constant h_isosceles_AB1C : isosceles_triangle A B1 C φ 
constant median_AA1 : median A A1 B C
constant h_M_on_median : M_on_median M A1 A
constant h_M_eqidist_B1_C1 : dist M B1 = dist M C1

-- Question
theorem angle_B1MC1_eq_phi 
(h_ABC : Triangle A B C)
(h_AB_ne_AC : AB ≠ AC)
(h_isosceles_AC1B : isosceles_triangle A C1 B φ)
(h_isosceles_AB1C : isosceles_triangle A B1 C φ)
(median_AA1 : median A A1 B C)
(h_M_on_median : M_on_median M A1 A)
(h_M_eqidist_B1_C1 : dist M B1 = dist M C1) : 
∠ B1 M C1 = φ := 
sorry

end angle_B1MC1_eq_phi_l592_592749


namespace ellipse_proof_l592_592520

def fociA : (ℝ × ℝ) := (2, 1)
def fociB : (ℝ × ℝ) := (2, 5)
def pointP : (ℝ × ℝ) := (-3, 3)

def a : ℝ := Real.sqrt 29
def k : ℝ := 3

theorem ellipse_proof : (a + k) = Real.sqrt 29 + 3 := by
  sorry

end ellipse_proof_l592_592520


namespace shaded_area_eq_2_25_l592_592157

/-- Problem statement:
Prove that the area of the shaded region defined by the lines passing through points (0, 3) and (6, 3)
for the first line, and points (0, 6) and (3, 0) for the second line, from \(x = 0\) to the 
intersection point is equal to \(2.25\) square units.
-/

noncomputable def line1 (x : ℝ) : ℝ := 3

noncomputable def line2 (x : ℝ) : ℝ := -2 * x + 6

theorem shaded_area_eq_2_25 :
  let f := λ x : ℝ, line2 x - line1 x in
  ∫ x in (0 : ℝ)..1.5, f x = 2.25 :=
by
  -- Definitions and integration setup
  let f := λ x : ℝ, line2 x - line1 x
  sorry

end shaded_area_eq_2_25_l592_592157


namespace problem_A_value_l592_592243

theorem problem_A_value (x y A : ℝ) (h : (x + 2 * y) ^ 2 = (x - 2 * y) ^ 2 + A) : A = 8 * x * y :=
by {
    sorry
}

end problem_A_value_l592_592243


namespace g_29_eq_27_l592_592393

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x : ℝ, g (x + g x) = 3 * g x
axiom initial_condition : g 2 = 9

theorem g_29_eq_27 : g 29 = 27 := by
  sorry

end g_29_eq_27_l592_592393


namespace greatest_three_digit_number_l592_592827

theorem greatest_three_digit_number (n : ℕ) :
  (n % 8 = 2) ∧ (n % 7 = 4) ∧ (100 ≤ n ∧ n ≤ 999) → n = 970 :=
begin
  sorry
end

end greatest_three_digit_number_l592_592827


namespace smaller_octagon_area_fraction_l592_592263

noncomputable def ratio_of_areas_of_octagons (A B C D E F G H : ℝ) : ℝ :=
  -- Assume A B C D E F G H represent vertices of a regular octagon
  let larger_octagon_area := sorry in  -- Compute area of the larger octagon based on its geometric properties
  let smaller_octagon_area := sorry in -- Compute area of the smaller octagon based on its geometric properties and the midpoints
  smaller_octagon_area / larger_octagon_area

theorem smaller_octagon_area_fraction (A B C D E F G H : ℝ) 
  (h_regular : regular_octagon A B C D E F G H)
  (h_midpoints : smaller_octagon_created_by_midpoints A B C D E F G H)
  (h_angle_center : angle_at_center A B C D E F G H = 45) :
  ratio_of_areas_of_octagons A B C D E F G H = 1 / 2 :=
by
  sorry

end smaller_octagon_area_fraction_l592_592263


namespace product_invertibles_mod_120_eq_1_l592_592332

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def product_of_invertibles_mod_n (n : ℕ) :=
  List.prod (List.filter (fun x => is_coprime x n) (List.range n))

theorem product_invertibles_mod_120_eq_1 :
  product_of_invertibles_mod_n 120 % 120 = 1 := 
sorry

end product_invertibles_mod_120_eq_1_l592_592332


namespace sin_315_eq_neg_sqrt2_div_2_l592_592974

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592974


namespace binomial_coefficient_divisible_by_p_l592_592758

theorem binomial_coefficient_divisible_by_p (p k : ℕ) (hp : Nat.Prime p) (hk1 : 0 < k) (hk2 : k < p) :
  p ∣ (Nat.factorial p / (Nat.factorial k * Nat.factorial (p - k))) :=
by
  sorry

end binomial_coefficient_divisible_by_p_l592_592758


namespace quadratic_eq_root_conjugate_l592_592193

theorem quadratic_eq_root_conjugate (b c : ℝ) (i : ℂ) (h₁ : 0 + 1 * i = complex.I)
  (h₂ : ∃ (x : ℂ), x*x + b*x + c = 0 ∧ x = 5 + 3*complex.I) : c = 34 := 
by
  sorry

end quadratic_eq_root_conjugate_l592_592193


namespace range_D_has_frequency_0_2_l592_592644

def sample : List ℕ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

def total_data_points := 20
def frequency_to_check_for := 0.2
def frequency_count := total_data_points * frequency_to_check_for  -- This should be 4

def contains_4_data_points_in_range (lower_bound upper_bound : ℤ) : Prop := 
  (sample.filter (λ x => lower_bound.to_nat ≤ x ∧ x < (upper_bound+1).to_nat)).length = 4

theorem range_D_has_frequency_0_2 : contains_4_data_points_in_range 11.5 13.5 := 
by 
  sorry  -- proof to be filled

end range_D_has_frequency_0_2_l592_592644


namespace matrix_sequence_product_l592_592134

-- Definitions of the matrix multiplication function for given matrices.
def A (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![1, 2*n], ![0, 1]]

-- Matrix multiplication function for a sequence of matrices.
def matrix_product (f : ℕ → Matrix (Fin 2) (Fin 2) ℕ) (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.finRange n).map f |>.foldl (· ⬝ ·) 1

-- The problem statement: Prove the product of the given matrices equals the desired result.
theorem matrix_sequence_product :
  matrix_product A 50 = ![![1, 2550], ![0, 1]] :=
by
  sorry

end matrix_sequence_product_l592_592134


namespace sin_315_degree_l592_592952

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592952


namespace distinct_floor_values_count_l592_592163

theorem distinct_floor_values_count : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 500 → 
  (count_distinct (λ n, nat.floor (n^2 / (500 : ℚ))) (finset.range 501)) = 376 :=
by sorry

end distinct_floor_values_count_l592_592163


namespace angela_action_figures_l592_592116

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l592_592116


namespace Gretchen_walks_distance_l592_592237

section GretchenWalking

variables (meet_hours desk_hours break_hours walking_break_minutes sitting_interval walking_interval walking_speed : ℕ)

def total_walking_time (desk_hours break_hours walking_break_minutes : ℕ) (sitting_interval walking_interval : ℕ) : ℕ :=
  let desk_minutes := desk_hours * 60
      lunch_break_minutes := break_hours * 60
      sitting_lunch_break := lunch_break_minutes - walking_break_minutes
      total_sitting_minutes := desk_minutes + sitting_lunch_break
      intervals := total_sitting_minutes / sitting_interval
  in intervals * walking_interval + walking_break_minutes

def walking_distance (walking_time : ℕ) (walking_speed : ℕ) : ℚ :=
  walking_time / 60 * walking_speed

theorem Gretchen_walks_distance :
  total_walking_time 4 2 30 75 15 * 3 / 60 = 4.5 :=
by
  sorry

end GretchenWalking

end Gretchen_walks_distance_l592_592237


namespace max_largest_integer_l592_592663

theorem max_largest_integer (A B C D E : ℕ) 
  (h1 : A ≤ B) 
  (h2 : B ≤ C) 
  (h3 : C ≤ D) 
  (h4 : D ≤ E)
  (h5 : (A + B + C + D + E) / 5 = 60) 
  (h6 : E - A = 10) : 
  E ≤ 290 :=
sorry

end max_largest_integer_l592_592663


namespace angela_action_figures_left_l592_592119

theorem angela_action_figures_left :
  ∀ (initial_collection : ℕ), 
  initial_collection = 24 → 
  (let sold := initial_collection / 4 in
   let remaining_after_sold := initial_collection - sold in
   let given_to_daughter := remaining_after_sold / 3 in
   let remaining_after_given := remaining_after_sold - given_to_daughter in
   remaining_after_given = 12) :=
by
  intros
  sorry

end angela_action_figures_left_l592_592119


namespace sin_315_eq_neg_sqrt2_div_2_l592_592917

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592917


namespace probability_of_b_l592_592627

-- Definitions from conditions
variables (a b : Prop)
def p (e : Prop) : ℝ := sorry -- Assume a general probability function

-- Given conditions
axiom p_a : p a = 5 / 7
axiom p_a_and_b : p (a ∩ b) = 0.28571428571428575
axiom independent_events : ∀ A B : Prop, A ∩ B = A ∧ B → p (A ∧ B) = p A * p B

-- Theorem to prove
theorem probability_of_b : p b = 0.4 := by
  sorry

end probability_of_b_l592_592627


namespace length_of_PS_l592_592283

theorem length_of_PS 
  (PQ QR PR : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 15)
  (hPR : PR = 17)
  {P Q R S : Type}
  (PQR : ∠ QPR → 90°)
  (PS_bisects_QPR : PS = angleBisector (∠ QPR)) : PS = 8 := 
sorry

end length_of_PS_l592_592283


namespace train_crossing_pole_time_l592_592700

theorem train_crossing_pole_time :
  ∀ (length_of_train : ℝ) (speed_km_per_hr : ℝ) (t : ℝ),
    length_of_train = 45 →
    speed_km_per_hr = 108 →
    t = 1.5 →
    t = length_of_train / (speed_km_per_hr * 1000 / 3600) := 
  sorry

end train_crossing_pole_time_l592_592700


namespace shortest_chord_length_l592_592626

open Real

-- Definitions for points A, B, D
def A := (1 : ℝ, 0 : ℝ)
def B := (2 : ℝ, -1 : ℝ)
def D := (1 : ℝ, -1/2)

-- Circle center condition
def center_lies_on_line (center : ℝ × ℝ) : Prop := 
  center.1 + center.2 = 0

-- Circle passing through points condition
def passes_through_point (center : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop :=
  (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

-- The shortest chord length through point D
theorem shortest_chord_length :
  -- There exists a center and radius such that these conditions hold
  (∃ (center : ℝ × ℝ) (r : ℝ),
    center_lies_on_line center ∧
    passes_through_point center r A ∧
    passes_through_point center r B ∧
    -- This statements asserts the length we need to prove
    let dist := sqrt ((center.1 - D.1)^2 + (center.2 - D.2)^2) in
    2 * sqrt (r^2 - dist^2) = sqrt 3) :=
begin
  sorry
end

end shortest_chord_length_l592_592626


namespace initial_water_amount_l592_592475

theorem initial_water_amount (W : ℝ) 
  (evap_per_day : ℝ := 0.0008) 
  (days : ℤ := 50) 
  (percentage_evap : ℝ := 0.004) 
  (evap_total : ℝ := evap_per_day * days) 
  (evap_eq : evap_total = percentage_evap * W) : 
  W = 10 := 
by
  sorry

end initial_water_amount_l592_592475


namespace total_age_l592_592497

-- Define the conditions
variables {A B C : ℕ}

-- Given conditions
def condition1 : Prop := A = B + 2
def condition2 : Prop := B = 2 * C
def condition3 : Prop := B = 8

-- Theorem statement
theorem total_age : condition1 → condition2 → condition3 → A + B + C = 22 :=
by
  intro h1 h2 h3
  -- Proof would go here
  sorry

end total_age_l592_592497


namespace sum_of_digits_of_expression_l592_592055

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592055


namespace min_translation_for_symmetry_l592_592817

noncomputable def f (x : Real) : Real := sqrt 3 * Real.sin x - Real.cos x

theorem min_translation_for_symmetry (a : Real) (h : a > 0)
  (h_sym : ∀ x, f (x - a) = f (- (x - a))) :
  a = π / 3 :=
by
  sorry

end min_translation_for_symmetry_l592_592817


namespace ten_digit_number_property_l592_592541

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def product_of_digits (n : ℕ) : ℕ :=
  n.digits 10).prod

theorem ten_digit_number_property :
  ∃ (n : ℕ), no_zero_digits n ∧ (product_of_digits n + n) = 1111111631 ∧ (product_of_digits n = product_of_digits (n + product_of_digits n)) :=
  ⟨1111111613, 
    by {
      sorry, -- proof that 1111111613 contains no zeros
    },
    by {
      sorry, -- proof that product of digits of 1111111613 + 1111111613 = 1111111631
    },
    by {
      sorry, -- proof that product of digits of 1111111613 = product of digits of 1111111613 + product_of_digits 1111111613
    }⟩

end ten_digit_number_property_l592_592541


namespace solve_quadratic_equation_solve_linear_factor_equation_l592_592375

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x^2 - 6 * x + 1 = 0 → (x = 3 - 2 * Real.sqrt 2 ∨ x = 3 + 2 * Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

theorem solve_linear_factor_equation :
  ∀ (x : ℝ), x * (2 * x - 1) = 2 * (2 * x - 1) → (x = 1 / 2 ∨ x = 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_equation_solve_linear_factor_equation_l592_592375


namespace sum_of_distances_l592_592276

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
(√3 * Real.cos α, Real.sin α)

def line_parametric (t : ℝ) : ℝ × ℝ :=
(1 + (√2 / 2) * t, (√2 / 2) * t)

def curve_polar (ρ θ : ℝ) : Prop :=
ρ^2 + 2 * ρ^2 * (Real.sin θ)^2 - 3 = 0

theorem sum_of_distances (t₁ t₂ : ℝ)
  (h₁ : ∃ t, line_parametric t = (1 + (√2 / 2) * t₁, 0))
  (h₂ : ∀ ρ θ, curve_polar ρ θ)
  (h₃ : ∃ t, line_parametric t = curve_parametric t₁)
  (h₄ : ∃ t, line_parametric t = curve_parametric t₂) :
  (t₁ + t₂) ^ 2 - 2 * t₁ * t₂ = 5 / 2 :=
sorry

end sum_of_distances_l592_592276


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592991

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592991


namespace sum_shade_length_l592_592679

-- Define the arithmetic sequence and the given conditions
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (is_arithmetic : ∀ n, a (n + 1) = a n + d)

-- Define the shadow lengths for each term using the arithmetic progression properties
def shade_length_seq (seq : ArithmeticSequence) : ℕ → ℝ := seq.a

variables (seq : ArithmeticSequence)

-- Given conditions
axiom sum_condition_1 : seq.a 1 + seq.a 4 + seq.a 7 = 31.5
axiom sum_condition_2 : seq.a 2 + seq.a 5 + seq.a 8 = 28.5

-- Question to prove
theorem sum_shade_length : seq.a 3 + seq.a 6 + seq.a 9 = 25.5 :=
by
  -- proof to be filled in later
  sorry

end sum_shade_length_l592_592679


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592043

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592043


namespace product_invertibles_mod_120_eq_1_l592_592330

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def product_of_invertibles_mod_n (n : ℕ) :=
  List.prod (List.filter (fun x => is_coprime x n) (List.range n))

theorem product_invertibles_mod_120_eq_1 :
  product_of_invertibles_mod_n 120 % 120 = 1 := 
sorry

end product_invertibles_mod_120_eq_1_l592_592330


namespace arrangements_starting_with_vowel_l592_592239

theorem arrangements_starting_with_vowel (word : String) (letters : Multiset Char) (vowels : Set Char) :
  word = "basics" →
  letters = {'b', 'a', 's', 'i', 'c', 's'} →
  vowels = {'a', 'i'} →
  (∀ c ∈ letters, Multiset.count c letters = 2 → c = 's') →
  finset.card (Finset.filter (λ s, s[0] ∈ vowels) (Finset.univ : Finset (List Char)).attach) = 120 :=
by
  sorry

end arrangements_starting_with_vowel_l592_592239


namespace sin_315_eq_neg_sqrt2_div_2_l592_592914

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592914


namespace sum_of_digits_of_expression_l592_592035

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592035


namespace bin_add_convert_l592_592434

-- Definitions
def bin1 := 11111111
def bin2 := 11111
def dec_value (b : ℕ) : ℕ := (2^(b.digits).length - 1)

-- Main proof statement
theorem bin_add_convert (h1 : dec_value bin1 = 255) (h2 : dec_value bin2 = 31) : 
  let sum := h1 + h2 in 
  let base8 := nat.base_8_val sum in 
  let final_decimal := nat.base_10_of_base_8 base8 in 
  final_decimal = 286 := 
by 
  sorry

end bin_add_convert_l592_592434


namespace equilateral_triangle_product_l592_592789

theorem equilateral_triangle_product (p q : ℂ)
(h1 : (0 : ℂ) = 0 + 0 * complex.I)
(h2 : p = p + 13 * complex.I)
(h3 : q = q + 41 * complex.I)
(h4 : ∃ (α : ℂ), α = complex.of_real (-1/2) + complex.I * complex.sqrt (3)/2 ∧ q + 41 * complex.I = (p + 13 * complex.I) * α):
  p * q = -2123 := by
  sorry

end equilateral_triangle_product_l592_592789


namespace polynomial_root_third_symmetric_sum_l592_592883

theorem polynomial_root_third_symmetric_sum (roots : Fin 6 → ℕ) (h_sum : (∑ i, roots i) = 12) 
  (h_roots : Polynomial.eval₂ (Polynomial.C : ℕ → Polynomial ℤ) (Polynomial.X : Polynomial ℤ) 
    ((∏ i, (Polynomial.X - Polynomial.C (roots i)) : Polynomial ℤ)) = Polynomial.of_coeffs [-36, D, C, -B, A, 12, 1]):
  B = 76 := sorry

end polynomial_root_third_symmetric_sum_l592_592883


namespace num_subsets_A_l592_592179

-- Definitions based on given conditions
def M : Set ℝ := {a | |a| ≥ 2}
def A : Set ℝ := {a | (a - 2) * (a^2 - 3) = 0 ∧ a ∈ M}

-- The proof statement
theorem num_subsets_A : {2} ⊆ A ∧ ∀ a ∈ A, a = 2 → finset.card (set.to_finset A.powerset) = 2 := 
by 
  split 
  -- Proof steps can be added here if needed
  .. 
  sorry

end num_subsets_A_l592_592179


namespace roberts_percentage_gain_l592_592756

theorem roberts_percentage_gain:
  ∀ (S: ℝ), S > 0 → 
  let final_salary := (0.97 * 1.05 * 0.9 * 1.5 * S)
  in ((final_salary - S) / S) * 100 = 37.4975 := by
  intros S hS
  let final_salary := (0.97 * 1.05 * 0.9 * 1.5 * S)
  have hfinal : final_salary = 1.374975 * S := by sorry
  have hgain : (final_salary - S) / S * 100 = 37.4975 := by
    calc ((1.374975 * S - S) / S) * 100 = ((1.374975 - 1) * S / S) * 100 : by sorry
                                    ... = (0.374975 * S / S) * 100       : by sorry
                                    ... = 0.374975 * 100                 : by sorry
                                    ... = 37.4975                        : by sorry
  exact hgain

end roberts_percentage_gain_l592_592756


namespace silent_session_possible_l592_592674

-- Define a structure to represent a student in the class
structure Student :=
  (is_talker : Bool) -- Whether the student is a talker
  (friends : List Student) -- List of friends of the student

noncomputable def are_friends (s1 s2 : Student) : Bool :=
  s1 ∈ s2.friends

-- Define the main theorem
theorem silent_session_possible (students : List Student) :
  ((∀ s : Student, s.is_talker → ∃ f : Student, f ∈ s.friends ∧ ¬f.is_talker) ∧
  (∀ s : Student, s.is_talker → ¬ s.is_talker)) →
  ∃ invited : List Student, (∀ s : Student, s ∈ invited → ¬s.is_talker) ∧ (invited.length ≥ students.length / 2) :=
by
  sorry

end silent_session_possible_l592_592674


namespace Finn_initial_goldfish_l592_592592

variable (x : ℕ)

-- Defining the conditions
def number_of_goldfish_initial (x : ℕ) : Prop :=
  ∃ y z : ℕ, y = 32 ∧ z = 57 ∧ x = y + z 

-- Theorem statement to prove Finn's initial number of goldfish
theorem Finn_initial_goldfish (x : ℕ) (h : number_of_goldfish_initial x) : x = 89 := by
  sorry

end Finn_initial_goldfish_l592_592592


namespace relationship_l592_592347

noncomputable def f : ℝ → ℝ
| x := if 1 < x then log 0.2 x else 2 - 2 * x

def a := f 2 ^ 0.3
def b := f (log 0.3 2)
def c := f (log 3 2)

theorem relationship among a, b, and c : b > c > a := by sorry

end relationship_l592_592347


namespace fraction_addition_l592_592904

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l592_592904


namespace compute_S_15_l592_592558

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def first_element_in_set (n : ℕ) : ℕ := sum_first_n (n - 1) + 1

def last_element_in_set (n : ℕ) : ℕ := first_element_in_set n + n - 1

def S (n : ℕ) : ℕ := n * (first_element_in_set n + last_element_in_set n) / 2

theorem compute_S_15 : S 15 = 1695 := by
  sorry

end compute_S_15_l592_592558


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592993

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592993


namespace two_bedroom_units_l592_592476

theorem two_bedroom_units {x y : ℕ} 
  (h1 : x + y = 12) 
  (h2 : 360 * x + 450 * y = 4950) : 
  y = 7 := 
by
  sorry

end two_bedroom_units_l592_592476


namespace hyperbola_line_intersection_l592_592500

theorem hyperbola_line_intersection
  (A B m : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) (hm : m ≠ 0) :
  ∃ x y : ℝ, A^2 * x^2 - B^2 * y^2 = 1 ∧ Ax - By = m ∧ Bx + Ay ≠ 0 :=
by
  sorry

end hyperbola_line_intersection_l592_592500


namespace reciprocal_fraction_addition_l592_592445

theorem reciprocal_fraction_addition (a b c : ℝ) (h : a ≠ b) :
  (a + c) / (b + c) = b / a ↔ c = - (a + b) := 
by
  sorry

end reciprocal_fraction_addition_l592_592445


namespace lizzys_shipping_cost_l592_592742

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

def number_of_crates (W w : ℝ) : ℝ := W / w
def total_shipping_cost (c n : ℝ) : ℝ := c * n

theorem lizzys_shipping_cost :
  total_shipping_cost cost_per_crate (number_of_crates total_weight weight_per_crate) = 27 := 
by {
  sorry
}

end lizzys_shipping_cost_l592_592742


namespace product_of_coprimes_mod_120_l592_592318

open Nat

noncomputable def factorial_5 : ℕ := 5!

def is_coprime_to_120 (x : ℕ) : Prop := gcd x 120 = 1

def coprimes_less_than_120 : List ℕ :=
  (List.range 120).filter is_coprime_to_120

def product_of_coprimes_less_than_120 : ℕ :=
  coprimes_less_than_120.foldl (*) 1

theorem product_of_coprimes_mod_120 : 
  (product_of_coprimes_less_than_120 % 120) = 1 :=
sorry

end product_of_coprimes_mod_120_l592_592318


namespace cuboctahedron_volume_is_5_sqrt_2_div_3_l592_592070

def volume_cuboctahedron (a : ℝ) : ℝ := (5 * a^3 * (2^(1/2))) / 3

theorem cuboctahedron_volume_is_5_sqrt_2_div_3 :
  volume_cuboctahedron 1 = 5 * (2^(1/2)) / 3 :=
by
  sorry

end cuboctahedron_volume_is_5_sqrt_2_div_3_l592_592070


namespace polygons_congruent_l592_592425

variables {n : ℕ} (A B : Fin n → ℝ × ℝ)
  (eq_sides : ∀ i, dist (A i) (A ((i + 1) % n)) = dist (B i) (B ((i + 1) % n)))
  (eq_angles : ∃ (indices : Fin (n-3) → Fin n), ∀ j, angle (A (indices j)) (A ((indices j) + 1 % n)) (A ((indices j) + 2 % n)) 
                                            = angle (B (indices j)) (B ((indices j) + 1 % n)) (B ((indices j) + 2 % n)))

theorem polygons_congruent : A = B :=
begin
  sorry
end

end polygons_congruent_l592_592425


namespace fraction_addition_l592_592895

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l592_592895


namespace m_le_n_l592_592752

theorem m_le_n (k m n : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : m^2 + n = k^2 + k) : m ≤ n := 
sorry

end m_le_n_l592_592752


namespace sum_of_digits_of_expression_l592_592053

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592053


namespace test_score_l592_592781

theorem test_score (time_spent_preparing score_achieved max_score : ℝ) 
                   (h_score : score_achieved / 5 = 90 / 5)
                   (h_max_score : max_score = 140) :
  let score := (90 * 7) / 5 in
  score <= max_score ∧ score = 126 := by
  sorry

end test_score_l592_592781


namespace blue_cells_in_nxn_intersections_l592_592376

/-- Each cell in any 10x10 grid contains at least one blue cell
(This is a simplified version of the condition) -/
axiom blue_cell_in_any_10x10_grid 
  (B : ℕ → ℕ → Prop) 
  (h : ∀ i j : ℤ, (∃ a b : fin 10, B (i + a) (j + b))): true

/-- Prove that for any positive integer n, it is possible to select n rows and n columns so
that all of the n^2 cells in their intersections are blue. -/
theorem blue_cells_in_nxn_intersections
  (B : ℕ → ℕ → Prop)
  (h : ∀ i j : ℤ, (∃ a b : fin 10, B (i + a) (j + b))) :
  ∀ (n : ℕ), 0 < n → 
  ∃ (rows cols : fin n → ℤ), 
    ∀ r c : fin n, B (rows r) (cols c) :=
sorry

end blue_cells_in_nxn_intersections_l592_592376


namespace shift_left_result_l592_592780

def f (x : ℝ) : ℝ := (x - 1) ^ 2

def left_shift (h : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  h (x + a)

theorem shift_left_result : left_shift f 3 = λ x, (x + 2) ^ 2 := 
by 
  sorry

end shift_left_result_l592_592780


namespace sum_of_two_medians_leq_34P_sum_of_two_medians_geq_34p_l592_592359

-- Define the triangle sides and medians
variables {a b c : ℝ}
variables {ma mb mc : ℝ} -- medians corresponding to sides a, b, c

-- Define the perimeter and semi-perimeter
def P : ℝ := a + b + c
def p : ℝ := (a + b + c) / 2

-- Theorem statement for part (a)
theorem sum_of_two_medians_leq_34P (h1 : ma + mb + mc ≥ 0) :
  ∀ (a b c ma mb mc : ℝ), ma + mb ≤ 3/4 * (a + b + c) :=
sorry

-- Theorem statement for part (b)
theorem sum_of_two_medians_geq_34p (h1 : ma + mb + mc ≥ 0) :
  ∀ (a b c ma mb mc : ℝ), ma + mb ≥ 3/4 * ((a + b + c) / 2) :=
sorry

end sum_of_two_medians_leq_34P_sum_of_two_medians_geq_34p_l592_592359


namespace sum_of_digits_l592_592063

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592063


namespace spider_leg_pressure_l592_592494

/--
A giant spider is discovered. It weighs 2.5 times the previous largest spider, 
which weighed 6.4 ounces. Each of its legs has a cross-sectional area of 0.5 
square inches. How much pressure in ounces per square inch does each leg undergo?
-/
theorem spider_leg_pressure
  (weight_previous_spider : ℝ := 6.4)
  (weight_ratio : ℝ := 2.5)
  (cross_sectional_area : ℝ := 0.5)
  (number_of_legs : ℕ := 8) :
  let weight_giant_spider := weight_ratio * weight_previous_spider in
  let weight_per_leg := weight_giant_spider / number_of_legs in
  let pressure_per_leg := weight_per_leg / cross_sectional_area in
  pressure_per_leg = 4 :=
by
  let weight_giant_spider := weight_ratio * weight_previous_spider
  let weight_per_leg := weight_giant_spider / number_of_legs
  let pressure_per_leg := weight_per_leg / cross_sectional_area
  sorry

end spider_leg_pressure_l592_592494


namespace convex_polygon_trio_adj_covered_by_circle_l592_592358

theorem convex_polygon_trio_adj_covered_by_circle 
  (P : Type*) [convex_polygon P] :
  ∃ (trio : list (vertex P)), 
    adjacent_vertices trio ∧ 
    (∀ v ∈ P.vertices, in_circle (circle_through trio) v) :=
sorry

end convex_polygon_trio_adj_covered_by_circle_l592_592358


namespace num_pairs_of_books_l592_592242

/-- Given 11 books, the number of ways to choose 2 books is 55. -/
theorem num_pairs_of_books : ∀ (n : ℕ), n = 11 → (n * (n - 1)) / 2 = 55 :=
by
  intro n
  intro hn
  rw [hn]
  sorry

end num_pairs_of_books_l592_592242


namespace larger_root_exceeds_smaller_root_by_5_5_l592_592844

theorem larger_root_exceeds_smaller_root_by_5_5:
  let a := 2
      b := 5
      c := -12 in
  let discriminant := b^2 - 4 * a * c in
  let sqrt_discriminant := Nat.sqrt discriminant in
  let q1 := (-b + sqrt_discriminant) / (2 * a)
      q2 := (-b - sqrt_discriminant) / (2 * a) in
  q1 - q2 = 5.5 :=
by
  sorry

end larger_root_exceeds_smaller_root_by_5_5_l592_592844


namespace fifth_book_pages_l592_592352

-- definitions based on conditions
def pages_first_book := 20
def pages_second_book := 45
def pages_half_second_book := pages_second_book / 2
def pages_third_book := 32
def pages_total_fourth_book := 60
def pages_two_fifths_fourth_book := (2 / 5) * pages_total_fourth_book
def pages_three_fifths_fourth_book := (3 / 5) * pages_total_fourth_book
def total_pages := 200

-- Lean theorem statement
theorem fifth_book_pages :
  pages_first_book + pages_half_second_book + pages_half_second_book + pages_third_book + pages_two_fifths_fourth_book + 
  pages_three_fifths_fourth_book + (fetch fifth book pages) = total_pages →
  fetch fifth book pages = 43 :=
by
  simp [pages_first_book, pages_second_book, pages_half_second_book, pages_third_book, pages_total_fourth_book, pages_two_fifths_fourth_book, pages_three_fifths_fourth_book, total_pages]
  sorry

end fifth_book_pages_l592_592352


namespace no_prime_solution_l592_592463

theorem no_prime_solution (p : ℕ) (h_prime : Nat.Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end no_prime_solution_l592_592463


namespace max_value_l592_592757

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 3)
noncomputable def g (x : ℝ) : ℝ := 3 * sin (2 * x + 2 * π / 3) + 1

theorem max_value (x1 x2 : ℝ) (h1 : g x1 * g x2 = 16)
  (h2 : -3 * π / 2 ≤ x1 ∧ x1 ≤ 3 * π / 2)
  (h3 : -3 * π / 2 ≤ x2 ∧ x2 ≤ 3 * π / 2) :
  2 * x1 - x2 ≤ 35 * π / 12 :=
sorry

end max_value_l592_592757


namespace project_completion_days_l592_592480

-- A's work rate per day
def A_work_rate : ℚ := 1 / 20

-- B's work rate per day
def B_work_rate : ℚ := 1 / 30

-- Combined work rate per day
def combined_work_rate : ℚ := A_work_rate + B_work_rate

-- Work done by B alone in the last 5 days
def B_alone_work : ℚ := 5 * B_work_rate

-- Let variable x represent the number of days A and B work together
def x (x_days : ℚ) := x_days / combined_work_rate + B_alone_work = 1

theorem project_completion_days (x_days : ℚ) (total_days : ℚ) :
  A_work_rate = 1 / 20 → B_work_rate = 1 / 30 → combined_work_rate = 1 / 12 → x_days / 12 + 1 / 6 = 1 → x_days = 10 → total_days = x_days + 5 → total_days = 15 :=
by
  intros _ _ _ _ _ _
  sorry

end project_completion_days_l592_592480


namespace abs_diff_a_b_l592_592597

def tau (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ i, i > 0 ∧ n % i = 0).card

def S (n : ℕ) : ℕ := (finset.range (n + 1)).sum tau

def count_odd_S (N : ℕ) : ℕ := (finset.range (N + 1)).filter (λ i, S i % 2 = 1).card

def count_even_S (N : ℕ) : ℕ := (finset.range (N + 1)).filter (λ i, S i % 2 = 0).card

theorem abs_diff_a_b : |count_odd_S 2005 - count_even_S 2005| = 25 := 
sorry

end abs_diff_a_b_l592_592597


namespace nk_tournament_exists_l592_592887

def v2 (n : ℕ) : ℕ :=
  if h: n = 0 then 0 else Nat.find (λ k, Nat.coprime (n >> k) 2)

def is_tournament (n k : ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → ∀ (A B C D : ℕ), A ≠ B ∧ C ≠ D → 
    ((A meets B in round i) → (C meets D in round i) → (A meets C in round j) → (B meets D in round j))

theorem nk_tournament_exists (n k : ℕ) : 
  is_tournament n k ↔ k ≤ 2^(v2 n) - 1 :=
sorry

end nk_tournament_exists_l592_592887


namespace B_alone_in_24_days_l592_592841

variable (A B : Type) [Inhabited A] [Inhabited B]

-- Define the conditions
def work_together_in_8_days (do_work_together: A × B → ℝ → Prop) (a: A) (b: B) :=
  ∃ x: ℝ, x = 1 / 8 ∧ do_work_together (a, b) x

def A_alone_in_12_days (do_A_work : A → ℝ → Prop) (a: A) :=
  ∃ y: ℝ, y = 1 / 12 ∧ do_A_work a y

-- The theorem we need to prove
theorem B_alone_in_24_days (do_work_together: A × B → ℝ → Prop)
  (do_A_work : A → ℝ → Prop)
  (do_B_work : B → ℝ → Prop)
  (a: A) (b: B) :
  work_together_in_8_days do_work_together a b →
  A_alone_in_12_days do_A_work a →
  ∃ z: ℝ, z = 1 / 24 ∧ do_B_work b z :=
sorry

end B_alone_in_24_days_l592_592841


namespace sin_315_degree_l592_592955

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592955


namespace fraction_addition_l592_592898

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l592_592898


namespace measure_angle_PQR_given_conditions_l592_592272

-- Definitions based on conditions
variables {R P Q S : Type} [LinearOrder R] [AddGroup Q] [LinearOrder P] [LinearOrder S]

-- Assume given conditions
def is_straight_line (r s p : ℝ) : Prop := r + p = 2 * s

def is_isosceles_triangle (p s q : ℝ) : Prop := p = q

def angle (q s p : ℝ) := (q - s) - (s - p)

variables (r p q s : ℝ)

-- Define the given angles and equality conditions
def given_conditions : Prop := 
  is_straight_line r s p ∧
  angle q s p = 60 ∧
  is_isosceles_triangle p s q ∧
  r ≠ q 

-- The theorem we want to prove
theorem measure_angle_PQR_given_conditions : given_conditions r p q s → angle p q r = 120 := by
  sorry

end measure_angle_PQR_given_conditions_l592_592272


namespace hyperbola_asymptotes_l592_592214

-- Define the problem setup and required proof
theorem hyperbola_asymptotes :
  (∃ (foci : ℝ × ℝ), foci = (4, 0) ∧ 
     ((∃ (e_ellipse e_hyperbola : ℝ),
           e_ellipse = 4 / 5 ∧ e_hyperbola = 2 ∧ 
           e_ellipse + e_hyperbola = 14 / 5) ∧ 
          (∃ (a b : ℝ),
             a = 2 ∧ b = 2 * sqrt 3 ∧ 
             ∃ (asymptote : ℝ → Prop), 
               asymptote = (λ x, x = ± sqrt 3 * x)))))
 :=
by
  sorry

end hyperbola_asymptotes_l592_592214


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592045

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592045


namespace cartesian_equation_of_C_polar_equation_of_C_trajectory_of_M_trajectory_of_M_cartesian_l592_592270

-- Part 1
theorem cartesian_equation_of_C (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π) :
  ∀ x y : ℝ, (x = cos α) ∧ (y = 1 + sin α) →
  (x^2 + (y - 1)^2 = 1 ∧ (1 ≤ y ∧ y ≤ 2)) := sorry

theorem polar_equation_of_C (θ : ℝ) :
  ∀ ρ : ℝ, (ρ = 2 * sin θ) ∧ (π / 4 ≤ θ ∧ θ ≤ 3 * π / 4) := sorry

-- Part 2
theorem trajectory_of_M (θ : ℝ) (hθ : π / 4 ≤ θ ∧ θ ≤ 3 * π / 4) :
  ∀ ρ : ℝ, (ρ* sin θ = 2) := sorry

theorem trajectory_of_M_cartesian (x y : ℝ) :
  (y = 2) ∧ (x ∈ set.Icc (-2) 2) := sorry

end cartesian_equation_of_C_polar_equation_of_C_trajectory_of_M_trajectory_of_M_cartesian_l592_592270


namespace solve_linear_system_l592_592565

theorem solve_linear_system :
  ∃ (x y : ℚ), 
    3 * x - 2 * y = 1 ∧ 
    4 * x + 3 * y = 23 ∧ 
    x = 49 / 17 ∧ 
    y = 65 / 17 :=
by
  use 49 / 17, 65 / 17
  split
  . norm_num
  . norm_num

end solve_linear_system_l592_592565


namespace recurring_decimal_product_l592_592573

theorem recurring_decimal_product : (0.3333333333 : ℝ) * (0.4545454545 : ℝ) = (5 / 33 : ℝ) :=
sorry

end recurring_decimal_product_l592_592573


namespace find_consecutive_geometric_and_arithmetic_terms_l592_592814

theorem find_consecutive_geometric_and_arithmetic_terms :
  ∃ (a d : ℕ), a = 2 ∧ d = 4 ∧ a + (a + 3 * d) + (a + 24 * d) = 114
  ∧ (a + 3 * d)^2 = a * (a + 24 * d) 
  ∧ (2, 14, 98).1 = a 
  ∧ (2, 14, 98).2 = a + 3 * d
  ∧ (2, 14, 98).3 = a + 24 * d :=
by
  sorry

end find_consecutive_geometric_and_arithmetic_terms_l592_592814


namespace arccos_neg_one_eq_pi_l592_592543

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l592_592543


namespace sum_of_digits_of_expression_l592_592054

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592054


namespace determine_X_with_7_gcd_queries_l592_592482

theorem determine_X_with_7_gcd_queries : 
  ∀ (X : ℕ), (X ≤ 100) → ∃ (f : Fin 7 → ℕ × ℕ), 
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) ∧ (∃ (Y : Fin 7 → ℕ), 
      (∀ i, Y i = Nat.gcd (X + (f i).1) (f i).2) → 
        (∀ (X' : ℕ), (X' ≤ 100) → ((∀ i, Y i = Nat.gcd (X' + (f i).1) (f i).2) → X' = X))) :=
sorry

end determine_X_with_7_gcd_queries_l592_592482


namespace product_of_coprimes_mod_120_l592_592316

open Nat

noncomputable def factorial_5 : ℕ := 5!

def is_coprime_to_120 (x : ℕ) : Prop := gcd x 120 = 1

def coprimes_less_than_120 : List ℕ :=
  (List.range 120).filter is_coprime_to_120

def product_of_coprimes_less_than_120 : ℕ :=
  coprimes_less_than_120.foldl (*) 1

theorem product_of_coprimes_mod_120 : 
  (product_of_coprimes_less_than_120 % 120) = 1 :=
sorry

end product_of_coprimes_mod_120_l592_592316


namespace smallest_positive_period_max_value_in_interval_l592_592225

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 5 / 2 :=
sorry

end smallest_positive_period_max_value_in_interval_l592_592225


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592042

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592042


namespace geometric_locus_trianglular_arc_l592_592653

noncomputable def geometric_locus (A I : Point) (α : ℝ) (hα : 60 < α ∧ α < 90) : Set Point := 
  { B : Point | ∃ C : Point, 
    let A := angle_ABC A B C in
    let B := angle_ABC B C A in
    let C := angle_ABC C A B in
    incircle_center A B C = I ∧ 
    A < α ∧ B < α ∧ C < α }

theorem geometric_locus_trianglular_arc (A I : Point) (α : ℝ) (hα : 60 < α ∧ α < 90) :
  geometric_locus A I α hα = 
  { B : Point | ∃ (C : Point) (AB BI : Segment), 
    incircle_center A B C = I ∧ 
    ∃ arc_center : Point, 
    segment_span arc_center A = AB ∧ 
    spanning_angle arc_center ≤ α / 2 } :=
sorry

end geometric_locus_trianglular_arc_l592_592653


namespace directrix_of_parabola_l592_592580

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end directrix_of_parabola_l592_592580


namespace length_of_PS_in_right_triangle_l592_592285

theorem length_of_PS_in_right_triangle 
  (P Q R S : Type) 
  (PQ QR PR PS : ℝ)
  (hPQ : PQ = 8) 
  (hQR : QR = 15) 
  (hPR : PR = 17)
  (is_right_triangle : PQ^2 + QR^2 = PR^2)
  (angle_bisector : PS is_angle_bisector_of ∠ P Q R) : 
  PS = 15 * real.sqrt 1065 / 32 := 
sorry

end length_of_PS_in_right_triangle_l592_592285


namespace fraction_addition_l592_592902

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l592_592902


namespace parabola_integer_points_l592_592400

/-- The parabola Q has focus at (0, 2) and goes through the points (5, 4) and (-5, -4).
    We want to prove the number of points (x, y) on Q with integer coordinates such that 
    |5x + 4y| ≤ 1250 is 100. -/
theorem parabola_integer_points :
  let Q := {p : ℝ × ℝ | ∃ (x y : ℝ), (x, y) = p ∧ distance (x, y) (0, 2) = 
    distance (x, y) (x, -2)}
  ∀ (x y : ℤ), (x, y) ∈ Q ∧ abs (5 * x + 4 * y) ≤ 1250 → 
    ∃ (n : ℕ), n = 100 :=
by sorry

end parabola_integer_points_l592_592400


namespace quadratic_conditions_l592_592362

open Polynomial

noncomputable def exampleQuadratic (x : ℝ) : ℝ :=
-2 * x^2 + 12 * x - 10

theorem quadratic_conditions :
  (exampleQuadratic 1 = 0) ∧ (exampleQuadratic 5 = 0) ∧ (exampleQuadratic 3 = 8) :=
by
  sorry

end quadratic_conditions_l592_592362


namespace distinct_integers_in_sequence_count_l592_592536

theorem distinct_integers_in_sequence_count :
  let s := {k : ℕ | 1 ≤ k ∧ k ≤ 1000}
  let seq := λ k, (3 * k) ^ 2 / 3000
  (set.card (set.image (λ k, seq k) s) = 2583) :=
begin
  sorry
end

end distinct_integers_in_sequence_count_l592_592536


namespace find_varphi_intervals_of_increase_l592_592348

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_varphi (φ : ℝ) (h1 : -Real.pi < φ) (h2 : φ < 0)
  (h3 : ∃ k : ℤ, 2 * (Real.pi / 8) + φ = (Real.pi / 2) + k * Real.pi) :
  φ = -3 * Real.pi / 4 :=
sorry

theorem intervals_of_increase (m : ℤ) :
  ∀ x : ℝ, (π / 8 + m * π ≤ x ∧ x ≤ 5 * π / 8 + m * π) ↔
  Real.sin (2 * x - 3 * π / 4) > 0 :=
sorry

end find_varphi_intervals_of_increase_l592_592348


namespace sin_315_eq_neg_sqrt2_div_2_l592_592965

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592965


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592024

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592024


namespace jackson_collection_goal_l592_592290

theorem jackson_collection_goal 
  (days_in_week : ℕ)
  (goal : ℕ)
  (earned_mon : ℕ)
  (earned_tue : ℕ)
  (avg_collect_per_4house : ℕ)
  (remaining_days : ℕ)
  (remaining_goal : ℕ)
  (daily_target : ℕ)
  (collect_per_house : ℚ)
  :
  days_in_week = 5 →
  goal = 1000 →
  earned_mon = 300 →
  earned_tue = 40 →
  avg_collect_per_4house = 10 →
  remaining_goal = goal - earned_mon - earned_tue →
  remaining_days = days_in_week - 2 →
  daily_target = remaining_goal / remaining_days →
  collect_per_house = avg_collect_per_4house / 4 →
  (daily_target : ℚ) / collect_per_house = 88 := 
by sorry

end jackson_collection_goal_l592_592290


namespace evaluate_expression_l592_592370

theorem evaluate_expression (m n : ℕ) (h₁: m = 1) (h₂: n = 3) : 
  (3 * m^2 - 4 * m * n) - 2 * (m^2 + 2 * m * n) = -23 :=
by
  rw [h₁, h₂]
  simp [pow_two, mul_assoc, mul_comm, mul_left_comm]
  sorry

end evaluate_expression_l592_592370


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592940

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592940


namespace correct_graph_l592_592818

noncomputable def snail_position (time : ℝ) : ℝ :=
  constant_velocity * time -- Represents the snail's position as a function of time

noncomputable def rabbit_position (time : ℝ) : ℝ :=
  if time < t1 then v1 * time                              -- Initial sprint
  else if time < t2 then v1 * t1                           -- First rest
  else if time < t3 then v1 * t1 + v2 * (time - t2)        -- Second sprint
  else if time < t4 then v1 * t1 + v2 * (t3 - t2)          -- Second rest
  else if time < tf then v1 * t1 + v2 * (t3 - t2) + v3 * (time - t4) -- Final sprint
  else v1 * t1 + v2 * (t3 - t2) + v3 * (tf - t4)           -- Wait near the finish

theorem correct_graph :
  (snail_position, rabbit_position) represent "Snail’s graph rising steadily, rabbit’s showing two pauses and final wait." sorry

end correct_graph_l592_592818


namespace fifth_largest_divisor_of_2014000000_l592_592783

theorem fifth_largest_divisor_of_2014000000 :
  ∃ d : ℕ, d = 125875000 ∧ (∀ n : ℕ, 2 ≤ n → n-th_largest_divisor 2014000000 n = d) := by
sory

end fifth_largest_divisor_of_2014000000_l592_592783


namespace maximum_cookies_by_andy_l592_592815

-- Define the conditions
def total_cookies := 36
def cookies_by_andry (a : ℕ) := a
def cookies_by_alexa (a : ℕ) := 3 * a
def cookies_by_alice (a : ℕ) := 2 * a
def sum_cookies (a : ℕ) := cookies_by_andry a + cookies_by_alexa a + cookies_by_alice a

-- The theorem stating the problem and solution
theorem maximum_cookies_by_andy :
  ∃ a : ℕ, sum_cookies a = total_cookies ∧ a = 6 :=
by
  sorry

end maximum_cookies_by_andy_l592_592815


namespace train_length_l592_592511

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

theorem train_length (time_to_cross_bridge : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ) :
  time_to_cross_bridge = 16.665333439991468 →
  length_bridge = 150 →
  speed_kmph = 54 →
  let speed_mps := speed_kmph_to_mps speed_kmph in
  let total_distance := speed_mps * time_to_cross_bridge in
  let train_length := total_distance - length_bridge in
  train_length = 99.97999909987152 :=
by
  intros _ _ _
  let speed_mps := speed_kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_to_cross_bridge
  let train_length := total_distance - length_bridge
  show train_length = 99.97999909987152
  sorry

end train_length_l592_592511


namespace range_of_a_l592_592643

def p (a : ℝ) : Prop := (a-1)^2 - 4a^2 < 0
def q (a : ℝ) : Prop := 2 * a^2 - a > 1

theorem range_of_a (a : ℝ) (h : p(a) ∨ q(a)) : a < -1/2 ∨ a > 1/3 :=
sorry

end range_of_a_l592_592643


namespace find_t_l592_592342

noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ :=
  x^4 + p*x^3 + q*x^2 + s*x + t

theorem find_t {p q s t : ℝ}
  (h1 : ∀ r : ℝ, g r p q s t = 0 → r < 0 ∧ Int.mod (round r) 2 = 1)
  (h2 : p + q + s + t = 2047) :
  t = 5715 :=
sorry

end find_t_l592_592342


namespace triangle_third_side_length_l592_592266

theorem triangle_third_side_length (a b : ℕ) (h1 : a = 2) (h2 : b = 3) 
(h3 : ∃ x, x^2 - 10 * x + 21 = 0 ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)) :
  ∃ x, x = 3 := 
by 
  sorry

end triangle_third_side_length_l592_592266


namespace initial_men_in_camp_l592_592390

theorem initial_men_in_camp (days_initial men_initial : ℕ) (days_plus_thirty men_plus_thirty : ℕ)
(h1 : days_initial = 20)
(h2 : men_plus_thirty = men_initial + 30)
(h3 : days_plus_thirty = 5)
(h4 : (men_initial * days_initial) = (men_plus_thirty * days_plus_thirty)) :
  men_initial = 10 :=
by sorry

end initial_men_in_camp_l592_592390


namespace suitable_value_for_x_evaluates_to_neg1_l592_592371

noncomputable def given_expression (x : ℝ) : ℝ :=
  (x^3 + 2 * x^2) / (x^2 - 4 * x + 4) / (4 * x + 8) - 1 / (x - 2)

theorem suitable_value_for_x_evaluates_to_neg1 : 
  given_expression (-6) = -1 :=
by
  sorry

end suitable_value_for_x_evaluates_to_neg1_l592_592371


namespace exist_n_gon_perpendicular_bisectors_exist_n_gon_angle_bisectors_l592_592611

-- Define the conditions of the problem
def lines : Type := list line
def n := ℕ
def polygon := Type 

-- Define properties for Part (a): Perpendicular Bisectors
def is_perpendicular_bisectors (lngs : lines) (p : polygon) : Prop := 
    ∀ (i : ℕ), is_perpendicular_bisector (lngs.nth i) (side i p)

-- Define properties for Part (b): Angle Bisectors
def is_angle_bisectors (lngs : lines) (p : polygon) : Prop := 
    ∀ (i : ℕ), is_angle_bisector (lngs.nth i) (angle i p)

-- Statement for Part (a)
theorem exist_n_gon_perpendicular_bisectors (lngs : lines) (n : ℕ) :
    ∃ p : polygon, is_perpendicular_bisectors lngs p :=
by sorry

-- Statement for Part (b)
theorem exist_n_gon_angle_bisectors (lngs : lines) (n : ℕ) :
    ∃ p : polygon, is_angle_bisectors lngs p :=
by sorry

end exist_n_gon_perpendicular_bisectors_exist_n_gon_angle_bisectors_l592_592611


namespace problem1_problem2_l592_592733

def prop_p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem1 (a : ℝ) (h_a : a = 1) (h_pq : ∃ x, prop_p x a ∧ prop_q x) :
  ∃ x, 2 < x ∧ x < 3 :=
by sorry

theorem problem2 (h_qp : ∀ x (a : ℝ), prop_q x → prop_p x a) :
  ∃ a, 1 < a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l592_592733


namespace sum_of_reciprocals_l592_592197

-- Define the sequences a_n and b_n
def a_n (n : ℕ) : ℤ := n - 1
def b_n (n : ℕ) : ℤ := 3 * n - 2

-- Define the squared distance between points P_1 and P_{n+1}
def d_squared (n : ℕ) : ℤ := 10 * n^2

-- Define the main theorem to prove
theorem sum_of_reciprocals (n : ℕ) : (n ≥ 1) → 
    (∑ k in Finset.range (n + 1) \ {0, 1}, 1 / (d_squared k)) < 1 / 5 := 
begin
  sorry
end

end sum_of_reciprocals_l592_592197


namespace max_x_for_area_lt_2004_l592_592598

theorem max_x_for_area_lt_2004 :
  ∀ (x : ℝ), (∀ (A B C D E : ℝ),
    A < B ∧ B < C ∧ C < D ∧ D < E ∧
    ∃ h : ℝ, isosceles_triangle A B h ∧ isosceles_triangle B C h ∧
              isosceles_triangle C D h ∧ isosceles_triangle D E h ∧
              AZ = 4 * x ∧
              area_of_triangle (AX A x) (AY A x) (AZ A x) < 2004)
  → x ≤ 22 :=
sorry

end max_x_for_area_lt_2004_l592_592598


namespace ratio_of_bases_l592_592268

-- Definitions for an isosceles trapezoid
def isosceles_trapezoid (s t : ℝ) := ∃ (a b c d : ℝ), s = d ∧ s = a ∧ t = b ∧ (a + c = b + d)

-- Main theorem statement based on conditions and required ratio
theorem ratio_of_bases (s t : ℝ) (h1 : isosceles_trapezoid s t)
  (h2 : s = s) (h3 : t = t) : s / t = 3 / 5 :=
by { sorry }

end ratio_of_bases_l592_592268


namespace purely_imaginary_condition_l592_592774

theorem purely_imaginary_condition (a b : ℝ) :
  (∃ z : ℂ, z = complex.mk 0 a ∧ (b = 0 ∧ a ≠ 0)) ↔ (b = 0 ∧ a ≠ 0) :=
by sorry

end purely_imaginary_condition_l592_592774


namespace sin_315_eq_neg_sqrt2_div_2_l592_592968

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592968


namespace sin_315_eq_neg_sqrt2_div_2_l592_592976

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592976


namespace statement1_correct_statement2_correct_statement3_correct_statement4_correct_l592_592075

-- Define the function for statement 1 condition
def func1 (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- Define the function for statement 2 condition
def func2 (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) - Real.sin (2 * x)

-- Define vectors for statement 3 condition
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)

-- Define the function for statement 4 condition
def func4 (x a : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) - a

-- Proof that statement 1 is correct
theorem statement1_correct : 
  ∀ k : ℤ, ∀ x : ℝ, (k * Real.pi + Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 7 * Real.pi / 12) ↔
    func1 x is decreasing on the interval [k * Real.pi + Real.pi / 12, k * Real.pi + 7 * Real.pi / 12] := sorry

-- Proof that statement 2 is correct
theorem statement2_correct : 
  ∃ k : ℤ, ∃ x : ℝ, (x = k * Real.pi / 2 + Real.pi / 6) ↔ (func2 (Real.pi / 6) = 0) := sorry

-- Proof that statement 3 is correct
theorem statement3_correct :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = (3 * Real.sqrt 2) / 2 := sorry

-- Proof that statement 4 is correct
theorem statement4_correct :
  ∀ x₁ x₂ a : ℝ, x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 → 
    ∃ x : ℝ, func4 x₁ a = 0 ∧ func4 x₂ a = 0 → x₁ + x₂ = Real.pi / 6 := sorry

end statement1_correct_statement2_correct_statement3_correct_statement4_correct_l592_592075


namespace cats_awake_l592_592415

theorem cats_awake (total_cats : ℕ) (percentage_asleep : ℚ) (h_total_cats : total_cats = 235) (h_percentage_asleep : percentage_asleep = 0.83) : 
(total_cats - (total_cats * percentage_asleep).nat_floor = 40) :=
by sorry

end cats_awake_l592_592415


namespace max_int_y_l592_592831

theorem max_int_y (y : ℤ) : (y : ℚ) / 4 + 3 / 7 < 7 / 4 → y ≤ 5 :=
begin
  sorry
end

end max_int_y_l592_592831


namespace sin_315_eq_neg_sqrt2_div_2_l592_592922

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592922


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592944

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592944


namespace trajectory_of_M_l592_592628

noncomputable def point (x y : ℝ) := (x, y)

def A : ℝ × ℝ := point (-1) 0
def C : ℝ × ℝ := point 1 0

def is_on_circle (B : ℝ × ℝ) : Prop :=
  let (x, y) := B in (x - 1)^2 + y^2 = 16

def perpendicular_bisector_intersects (A B C M : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  let (xM, yM) := M
  -- Placeholder for the actual condition for intersection
  true

def trajectory_equation (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in (x^2 / 4) + (y^2 / 3) = 1

theorem trajectory_of_M (B M : ℝ × ℝ) (h1 : is_on_circle B) (h2 : perpendicular_bisector_intersects A B C M) :
  trajectory_equation M :=
sorry

end trajectory_of_M_l592_592628


namespace jake_earning_per_hour_l592_592703

-- Define the conditions
def mowing_lawn_time : ℕ := 1
def mowing_lawn_pay : ℕ := 15
def flower_planting_time : ℕ := 2
def flower_planting_charge : ℕ := 45

-- Prove that Jake wants to make $22.50 per hour
theorem jake_earning_per_hour : flower_planting_charge / flower_planting_time = 22.50 := by
  -- The value in Lean needs to be converted to real numbers since it's not inherently capable of decimal arithmetic on natural numbers.
  let total_charge := (flower_planting_charge : ℝ)
  let hours_worked := (flower_planting_time : ℝ)
  have h : total_charge / hours_worked = 22.50 := by
    calc
      total_charge / hours_worked = 45 / 2 : by rfl
      ... = 22.50 : by norm_num
  exact h

end jake_earning_per_hour_l592_592703


namespace car_A_speed_l592_592539

theorem car_A_speed (s_A s_B : ℝ) (d_AB d_extra t : ℝ) (h_s_B : s_B = 50) (h_d_AB : d_AB = 40) (h_d_extra : d_extra = 8) (h_time : t = 6) 
(h_distance_traveled_by_car_B : s_B * t = 300) 
(h_distance_difference : d_AB + d_extra = 48) :
  s_A = 58 :=
by
  sorry

end car_A_speed_l592_592539


namespace simplify_and_evaluate_l592_592764

theorem simplify_and_evaluate (a : ℝ) (h₁ : a^2 - 4 * a + 3 = 0) (h₂ : a ≠ 3) : 
  ( (a^2 - 9) / (a^2 - 3 * a) / ( (a^2 + 9) / a + 6 ) = 1 / 4 ) :=
by 
  sorry

end simplify_and_evaluate_l592_592764


namespace rhombus_angle_G_eq_135_l592_592269

variable (EFGH : Type) [IsRhombus EFGH]
variable (E F G H : EFGH)
variable (aE : Angle) (aF : Angle) (aG : Angle) (aH : Angle)
variable [AngleMeasure aE = 135]

theorem rhombus_angle_G_eq_135
  (h1 : aE = 135) :
  aG = 135 := sorry

end rhombus_angle_G_eq_135_l592_592269


namespace directrix_of_parabola_l592_592581

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end directrix_of_parabola_l592_592581


namespace find_two_digit_number_l592_592077

theorem find_two_digit_number
  (X : ℕ)
  (h1 : 57 + (10 * X + 6) = 123)
  (h2 : two_digit_number = 10 * X + 9) :
  two_digit_number = 69 :=
by
  sorry

end find_two_digit_number_l592_592077


namespace circle_equation_unique_circle_equation_l592_592252

-- Definitions based on conditions
def radius (r : ℝ) : Prop := r = 1
def center_in_first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0
def tangent_to_line (a b : ℝ) : Prop := (|4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2)) = 1
def tangent_to_x_axis (b : ℝ) : Prop := b = 1

-- Main theorem statement
theorem circle_equation_unique 
  {a b : ℝ} 
  (h_rad : radius 1) 
  (h_center : center_in_first_quadrant a b) 
  (h_tan_line : tangent_to_line a b) 
  (h_tan_x : tangent_to_x_axis b) :
  (a = 2 ∧ b = 1) :=
sorry

-- Final circle equation
theorem circle_equation : 
  (∀ a b : ℝ, ((a = 2) ∧ (b = 1)) → (x - a)^2 + (y - b)^2 = 1) :=
sorry

end circle_equation_unique_circle_equation_l592_592252


namespace purely_imaginary_iff_a1_b0_l592_592771

-- Definitions for the conditions
variable (a b : ℝ)

-- Statement to be proved in Lean
theorem purely_imaginary_iff_a1_b0 : (z : ℂ) (h : z = (a : ℂ) * complex.I + b) 
  (ha : a = 1) (hb : b = 0) :
  z.im /= 0 ∧ z.re = 0 ↔ z = (1 : ℂ) * complex.I + 0 := 
by sorry

end purely_imaginary_iff_a1_b0_l592_592771


namespace no_solution_exists_l592_592455

theorem no_solution_exists (x y : ℝ) : 9^(y + 1) / (1 + 4 / x^2) ≠ 1 :=
by
  sorry

end no_solution_exists_l592_592455


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592938

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592938


namespace area_of_region_l592_592158

theorem area_of_region (x y : ℝ) (h1 : y = |2 * x|) (h2 : x^2 + y^2 = 9) :
  area = 9 * arcsin (2) :=
sorry

end area_of_region_l592_592158


namespace sum_non_prime_between_60_and_70_l592_592441

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def non_prime_numbers_between_60_and_70 : List ℕ :=
  List.filter (λ n, ¬is_prime n) [61, 62, 63, 64, 65, 66, 67, 68, 69]

theorem sum_non_prime_between_60_and_70 :
  non_prime_numbers_between_60_and_70.sum = 407 :=
by
  -- Proof will be provided here
  sorry

end sum_non_prime_between_60_and_70_l592_592441


namespace arccos_neg_one_eq_pi_l592_592546

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l592_592546


namespace valid_mappings_l592_592517

-- Define sets A and B for each condition
def A1 := {0, 1, 2, 3}
def B1 := {1, 2, 3, 4}
def A2 := set.Ioi 0  -- positive real numbers
def B2 := set.univ -- since B2 = ℝ
def A3 := set.univ -- since A3 = ℕ
def B3 := set.univ
def A4 := set.univ -- since A4 = ℝ
def B4 := set.univ

-- Define the rules corresponding to each condition
def f1 (x : ℕ) : ℕ := x + 1
def f2 (x : ℝ) : ℝ := real.sqrt x
def f3 (x : ℕ) : ℕ := 3 * x
def f4 (x : ℝ) : ℝ := x⁻¹

-- Prove that the rules form valid mappings for conditions 1 and 3
theorem valid_mappings : 
  (∀ a ∈ A1, ∃ b ∈ B1, f1 a = b) ∧
  (∀ a ∈ A2, ∃! b ∈ B2, f2 a = b) ∧ 
  (∀ a ∈ A3, ∃ b ∈ B3, f3 a = b) ∧ 
  (∀ a ∈ A4, ∃ b ∈ B4, f4 a = b) → 
  (∀ a ∈ A1, f1 a ∈ B1) ∧ 
  (∀ a ∈ A3, f3 a ∈ B3) := 
by 
  sorry

end valid_mappings_l592_592517


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592946

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592946


namespace trapezoid_diagonal_inequality_l592_592280

theorem trapezoid_diagonal_inequality {A B C D : Type} 
  [IsTrapezoid ABCD] 
  (h1 : ∠A < ∠D) 
  (h2 : ∠D < 90) : 
  AC > BD :=
sorry

end trapezoid_diagonal_inequality_l592_592280


namespace find_lambda_value_l592_592654

variables (a : ℝ^3) (b : ℝ^3) (λ : ℝ)

def are_orthogonal (v w : ℝ^3) : Prop := 
  dot_product v w = 0

theorem find_lambda_value :
  let a := ![0, 1, -1]
  let b := ![1, 1, 0]
  are_orthogonal (a + λ • b) a → λ = -2 :=
by
  intros h
  sorry

end find_lambda_value_l592_592654


namespace arithmetic_sequence_exists_l592_592185

theorem arithmetic_sequence_exists (n : Nat) (nums : Finset Nat) (h_distinct : nums.card = n) (h_n_cases : n = 5 ∨ n = 1989) :
  ∃ (a d : Nat), d > 0 ∧ a ≤ d ∧ (nums.filter (λ x, ∃ k : Nat, x = a + k * d)).card = 3 ∨ (nums.filter (λ x, ∃ k : Nat, x = a + k * d)).card = 4 :=
by
  sorry

end arithmetic_sequence_exists_l592_592185


namespace tangent_line_at_P_tangent_line_passing_through_P_l592_592634

-- Given definitions from conditions
def curve (x : ℝ) : ℝ := (1 / 3) * x^3 + (4 / 3)
def point_P := (2, 4 : ℝ)

-- The equation of the curve's derivative
def curve_derivative (x : ℝ) : ℝ := x^2

-- Statement of the proof problem (1): equation of the tangent line at point_P
theorem tangent_line_at_P :
  let k := curve_derivative 2 in
  (k = 4) ∧ (4*2 - 4 - 4 = 0) :=
by
  sorry

-- Statement of the proof problem (2): equation of the tangent line passing through point_P
theorem tangent_line_passing_through_P :
  ∃ x₀ : ℝ, (curve x₀, x₀) ∧ 
    ((curve_derivative x₀) = x₀^2) ∧
    (4 = x₀^2 * 2 - (2 / 3) * x₀^3 + (4 / 3)) ∧
    (forall (x₀ = 2 ∨ x₀ = -1), 
      (∀ (x : ℝ), curve_derivative x₀ * x - (curve_derivative x₀ * x₀ - curve x₀) = curve (point_P.1) - point_P.2)) :=
by
  sorry

end tangent_line_at_P_tangent_line_passing_through_P_l592_592634


namespace longer_train_length_is_correct_l592_592428

noncomputable def longer_train_length := 
  let speed_first : ℝ := 60 * 1000 / 3600 in        -- conversion to m/s
  let speed_second : ℝ := 40 * 1000 / 3600 in       -- conversion to m/s
  let relative_speed : ℝ := speed_first + speed_second in
  let time_to_cross : ℝ := 12.59899208063355 in
  let shorter_train_length : ℝ := 160 in
  let total_distance_covered : ℝ := relative_speed * time_to_cross in
  total_distance_covered - shorter_train_length

theorem longer_train_length_is_correct : longer_train_length = 190 := 
by
  -- Don't forget sorry avoids entering the proof 
  sorry

end longer_train_length_is_correct_l592_592428


namespace giant_spider_leg_pressure_l592_592488

-- Defining all conditions
def previous_spider_weight : ℝ := 6.4
def weight_multiplier : ℝ := 2.5
def num_legs : ℕ := 8
def leg_area : ℝ := 0.5

-- Compute the total weight of the giant spider
def giant_spider_weight : ℝ := previous_spider_weight * weight_multiplier

-- Compute the weight per leg
def weight_per_leg : ℝ := giant_spider_weight / num_legs

-- Calculate the pressure each leg undergoes
def pressure_per_leg : ℝ := weight_per_leg / leg_area

-- Theorem stating the expected pressure
theorem giant_spider_leg_pressure : pressure_per_leg = 4 :=
by
  -- Sorry is used here to skip the proof steps
  sorry

end giant_spider_leg_pressure_l592_592488


namespace interest_payment_frequency_l592_592388

theorem interest_payment_frequency (i : ℝ) (EAR : ℝ) (n : ℕ) (h_i : i = 0.16) (h_EAR : EAR = 0.1664) :
  1.1664 = (1 + (i / n)) ^ n :=
by
  have h1 : 1.1664 = EAR + 1, by simp [h_EAR]
  rw [h1, ←h_i]
  sorry
 
end interest_payment_frequency_l592_592388


namespace find_n_l592_592584

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -3736 [MOD 6] → n = 2 := by
  sorry

end find_n_l592_592584


namespace intersection_point_on_angle_bisector_l592_592401

-- Definitions of points and properties
variables (A B C D P Q : Type) [add_comm_group A] [module D A] 
variables [affine_space A]
variables (ABCD : affine_subspace D A)  
variables {BC CD : affine_submodule D A} 
variables (hABCD : is_parallelogram ABCD) 
variables (hP : P ∈ BC) (hQ : Q ∈ CD)
variables (hBP_EQ_QD : dist B P = dist Q D)

-- Definitions of lines and angle bisectors
variables (BQ DP : affine_subspace D A)
variables (hIntersect : ∃ X, X ∈ BQ ∧ X ∈ DP)
variables (angle_bisector_BAD : affine_subspace D A)

-- Main goal
theorem intersection_point_on_angle_bisector 
  (hABCD : is_parallelogram ABCD) 
  (hP : P ∈ BC) (hQ : Q ∈ CD) 
  (hBP_EQ_QD : dist B P = dist Q D) 
  (hIntersect : ∃ X, X ∈ BQ ∧ X ∈ DP)
  :
  ∃ X, X ∈ angle_bisector_BAD ∧ X ∈ BQ ∧ X ∈ DP := 
sorry

end intersection_point_on_angle_bisector_l592_592401


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592029

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592029


namespace distances_not_equal_l592_592203

noncomputable theory

open_locale big_operators

variables {α : Type*} {δ : Type*} [linear_ordered_field α]
variables [metric_space δ] [has_dist δ]

-- Point A and Point B on a line
variables (A B : δ)

-- Function to denote red and blue points
variables (R B' : finset δ) -- Assuming finite sets for red and blue points

-- Definitions for distances
variables (d : δ → δ → α)

-- Sum definitions for S1 and S2 based on the problem statement
def S_1 := ∑ x in R, d A x + ∑ y in B', d B y
def S_2 := ∑ y in B', d A y + ∑ x in R, d B x

-- Proof statement
theorem distances_not_equal 
  (h_red_blue_size : R.card + B'.card = 57) -- There are 57 points in total
  : S_1 ≠ S_2 :=
sorry  -- Proof to be completed here.

end distances_not_equal_l592_592203


namespace max_employees_relocated_preferred_l592_592846

theorem max_employees_relocated_preferred :
  let total_employees := 200
  let relocated_X := 0.30 * total_employees
  let relocated_Y := 0.70 * total_employees
  let prefers_X := 0.60 * total_employees
  let prefers_Y := 0.40 * total_employees
  max_employees_relocated_preferred = 60 + 80 :=
begin
  let total_employees: ℝ := 200,
  let relocated_X: ℝ := 0.30 * total_employees,
  let relocated_Y: ℝ := 0.70 * total_employees,
  let prefers_X: ℝ := 0.60 * total_employees,
  let prefers_Y: ℝ := 0.40 * total_employees,
  -- we aim to prove that the maximal number of employees relocated to their preferred city is 140
  have h₁ : relocated_X ≤ prefers_X, from sorry, -- proof steps not required
  have h₂ : prefers_Y ≤ relocated_Y, from sorry,
  exact sorry, -- final proof of 60 + 80 = 140
end

end max_employees_relocated_preferred_l592_592846


namespace value_of_e_l592_592456

theorem value_of_e (a : ℕ) (e : ℕ) 
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * 45 * e) : 
  e = 49 := 
by 
  sorry

end value_of_e_l592_592456


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592992

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592992


namespace sum_c_i_squared_le_sum_a_i_squared_mul_sum_b_i_squared_l592_592610

variable {n : ℕ}
variable (a_i b_i : Fin n → ℝ)

def a (a_i : Fin n → ℝ) := (∑ i, a_i i) / n
def b (b_i : Fin n → ℝ) := (∑ i, b_i i) / n

def c_i (a_i b_i : Fin n → ℝ) (i : Fin n) : ℝ :=
  |a a_i * b_i i + a_i i * b b_i - a_i i * b_i i|

theorem sum_c_i_squared_le_sum_a_i_squared_mul_sum_b_i_squared :
  (∑ i, c_i a_i b_i i) ^ 2 ≤ (∑ i, (a_i i) ^ 2) * (∑ i, (b_i i) ^ 2) :=
sorry

end sum_c_i_squared_le_sum_a_i_squared_mul_sum_b_i_squared_l592_592610


namespace numbers_in_circle_are_zero_l592_592416

theorem numbers_in_circle_are_zero (a : Fin 55 → ℤ) 
  (h : ∀ i, a i = a ((i + 54) % 55) + a ((i + 1) % 55)) : 
  ∀ i, a i = 0 := 
by
  sorry

end numbers_in_circle_are_zero_l592_592416


namespace giant_spider_leg_pressure_l592_592486

-- Defining all conditions
def previous_spider_weight : ℝ := 6.4
def weight_multiplier : ℝ := 2.5
def num_legs : ℕ := 8
def leg_area : ℝ := 0.5

-- Compute the total weight of the giant spider
def giant_spider_weight : ℝ := previous_spider_weight * weight_multiplier

-- Compute the weight per leg
def weight_per_leg : ℝ := giant_spider_weight / num_legs

-- Calculate the pressure each leg undergoes
def pressure_per_leg : ℝ := weight_per_leg / leg_area

-- Theorem stating the expected pressure
theorem giant_spider_leg_pressure : pressure_per_leg = 4 :=
by
  -- Sorry is used here to skip the proof steps
  sorry

end giant_spider_leg_pressure_l592_592486


namespace luke_hotdogs_ratio_l592_592745

-- Definitions
def hotdogs_per_sister : ℕ := 2
def total_sisters_hotdogs : ℕ := 2 * 2 -- Ella and Emma together
def hunter_hotdogs : ℕ := 6 -- 1.5 times the total of sisters' hotdogs
def total_hotdogs : ℕ := 14

-- Ratio proof problem statement
theorem luke_hotdogs_ratio :
  ∃ x : ℕ, total_hotdogs = total_sisters_hotdogs + 4 * x + hunter_hotdogs ∧ 
    (4 * x = 2 * 1 ∧ x = 1) := 
by 
  sorry

end luke_hotdogs_ratio_l592_592745


namespace expression_value_l592_592766

noncomputable def expression (x b : ℝ) : ℝ :=
  (x / (x + b) + b / (x - b)) / (b / (x + b) - x / (x - b))

theorem expression_value (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  expression x b = -1 := 
by
  sorry

end expression_value_l592_592766


namespace sin_315_degree_l592_592957

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592957


namespace blair_num_bars_l592_592553

-- Define the total number of bars and percentages
def totalWeight : ℝ := 100 -- scaling weight to percentage
def numBrennanBars : ℕ := 24
def brennanWeightPercent : ℝ := 45
def numMayaBars : ℕ := 13
def mayaWeightPercent : ℝ := 26

-- Formula to calculate the remaining weight percentage for Blair
def blairWeightPercent : ℝ := totalWeight - brennanWeightPercent - mayaWeightPercent 

-- The condition that Blair's bars weigh the remaining percentage
def numBlairBars (n : ℕ) : Prop := (blairWeightPercent / n) ≥ (brennanWeightPercent / numBrennanBars) ∧ (blairWeightPercent / n) ≤ (mayaWeightPercent / numMayaBars)

-- Theorem to prove how many bars Blair received
theorem blair_num_bars : ∃ n, numBlairBars n ∧ n = 15 :=
by
  existsi 15
  split
  calc
    blairWeightPercent : ℝ := 29 -- Calculations for proof correctness
  sorry -- Proof skipped for this exercise

end blair_num_bars_l592_592553


namespace fraction_addition_l592_592899

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l592_592899


namespace george_initial_candy_l592_592602

theorem george_initial_candy (number_of_bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : number_of_bags = 8) (h2 : pieces_per_bag = 81) : 
  number_of_bags * pieces_per_bag = 648 := 
by 
  sorry

end george_initial_candy_l592_592602


namespace sum_of_digits_of_expression_l592_592051

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592051


namespace two_bedroom_units_l592_592477

theorem two_bedroom_units {x y : ℕ} 
  (h1 : x + y = 12) 
  (h2 : 360 * x + 450 * y = 4950) : 
  y = 7 := 
by
  sorry

end two_bedroom_units_l592_592477


namespace dist_half_length_l592_592403

variable {α : Type*} [MetricSpace α]

-- Definitions of points A, B, C, D, and center O
variables (A B C D O : α)

-- Assumptions: 
-- 1. Quadrilateral ABCD is inscribed in a circle centered at O.
def is_inscribed (A B C D O : α) [EuclideanSpace ℝ α] : Prop := 
  ∀ (P : α), P ∈ {A, B, C, D} → dist O P = ∃ r : ℝ, dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

-- 2. Diagonals AC and BD are perpendicular.
def are_perpendicular (A C B D : α) [InnerProductSpace ℝ α] : Prop :=
  ∃ (u v w x : α), (AC = u - v) ∧ (BD = w - x) ∧ inner u v = 0

-- Distance from O to the line AD
def dist_to_line {β : Type*} [MetricSpace β] (P Q : β) (x : β) : ℝ :=
  sorry -- definition needed for distance from point to line (can be defined via projection)

-- Length of segment BC
def length_segment (P Q : α) [MetricSpace α] : ℝ :=
  dist P Q

-- Main theorem statement
theorem dist_half_length
  (insc: is_inscribed A B C D O)
  (perp: are_perpendicular A C B D)
  : dist_to_line O A D = (1 / 2) * length_segment B C := 
sorry

end dist_half_length_l592_592403


namespace sin_315_eq_neg_sqrt2_div_2_l592_592924

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592924


namespace find_m_l592_592394

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, f x = x^2 - 3*x + 2*m) ∧ 
  (∀ x : ℝ, g x = 2*x^2 - 6*x + 5*m) ∧ 
  (f 3 = 2*m) ∧ 
  (g 3 = 5*m) ∧ 
  (3 * f 3 = 2 * g 3) -> 
  m = 0 :=
begin
  intros,
  sorry
end


end find_m_l592_592394


namespace find_f2_l592_592392

theorem find_f2 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f(x) + 3 * f(1 - x) = 4 * x^2) : 
  f 2 = -1 / 2 :=
by
  sorry

end find_f2_l592_592392


namespace sin_315_eq_neg_sqrt2_div_2_l592_592969

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592969


namespace brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l592_592454

noncomputable def brocard_vertex_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(a * b * c, c^3, b^3)

theorem brocard_vertex_coordinates_correct (a b c : ℝ) :
  brocard_vertex_trilinear_coordinates a b c = (a * b * c, c^3, b^3) :=
sorry

noncomputable def steiner_point_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(1 / (a * (b^2 - c^2)),
  1 / (b * (c^2 - a^2)),
  1 / (c * (a^2 - b^2)))

theorem steiner_point_coordinates_correct (a b c : ℝ) :
  steiner_point_trilinear_coordinates a b c = 
  (1 / (a * (b^2 - c^2)),
   1 / (b * (c^2 - a^2)),
   1 / (c * (a^2 - b^2))) :=
sorry

end brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l592_592454


namespace sin_315_equals_minus_sqrt2_div_2_l592_592979

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592979


namespace smallest_k_for_60_subset_l592_592303

def is_integer_coordinate (p : ℝ × ℝ) : Prop := ∃ (a b : ℤ), p = (a.toReal, b.toReal)

def in_same_subset (S : set (ℝ × ℝ)) (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ C ∈ S, 
  is_integer_coordinate C ∧
  1 / 2 * (abs ((fst A * snd B + fst B * snd C + fst C * snd A) - (snd A * fst B + snd B * fst C + snd C * fst A))) = k

theorem smallest_k_for_60_subset :
  ∃ S : set (ℝ × ℝ), (∀ p ∈ S, is_integer_coordinate p) ∧ (card S = 60) ∧ (∀ A B ∈ S, A ≠ B → in_same_subset S A B 210) :=
sorry

end smallest_k_for_60_subset_l592_592303


namespace larger_triangle_side_l592_592775

theorem larger_triangle_side (A1 A2 : ℕ) (k : ℕ) :
  A1 - A2 = 32 ∧ A1 = k ^ 2 * A2 ∧ (∃ n : ℕ, A2 = n) ∧ n1 = 5 ->
  k = 3 ∧ so the larger side is 15 :=
by
  sorry

end larger_triangle_side_l592_592775


namespace product_of_invertibles_mod_120_l592_592338

open Nat

theorem product_of_invertibles_mod_120 :
  let m := 120
  let invertibles := { x | x < m ∧ gcd x m = 1 }
  ∏ a in invertibles, a % m = 119 :=
by
  sorry

end product_of_invertibles_mod_120_l592_592338


namespace spiders_webs_l592_592249

theorem spiders_webs (spiders days webs_per_spider : ℕ) (h₁ : spiders = 5) (h₂ : days = 5) (h₃ : webs_per_spider = 1) : spiders * webs_per_spider = 5 :=
by
  simp [h₁, h₂, h₃]
  sorry

end spiders_webs_l592_592249


namespace find_b_if_polynomial_is_square_l592_592788

theorem find_b_if_polynomial_is_square (a b : ℚ) (h : ∃ g : ℚ[X], (g ^ 2) = (X^4 + X^3 + 2*X^2 + a*X + b)) : b = 49/64 :=
by {
  sorry
}

end find_b_if_polynomial_is_square_l592_592788


namespace count_irrational_numbers_l592_592886

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, b ≠ 0 ∧ x = a / b

def given_numbers : List ℝ := [
    22 / 7,
    -real.sqrt 5,
    real.pi / 2,
    real.cbrt 8, -- Alternatively, this can be written as 2 but for clarity, we use cbrt.
    3.14,
    0.0010010001 -- A non-repeating, non-terminating decimal
]

def irrational_count (l : List ℝ) : Nat := l.countp is_irrational

theorem count_irrational_numbers : 
  irrational_count given_numbers = 3 := by 
  sorry

end count_irrational_numbers_l592_592886


namespace badger_wins_l592_592521

-- Definitions based on conditions
structure Tree :=
(vertices : Type)
(edges : vertices -> vertices -> Prop)
(acyclic : ∀ v, edges v v -> false)
(undirected : ∀ u v, edges u v ↔ edges v u)
(connected : ∀ u v, u ≠ v -> ∃ p : list vertices, path edges p u ∧ last p v)
(nonempty : ∃ v, true)

def apples_on_vertices (V : Type) := V -> ℕ

-- Game play structure
inductive Player
| Armadillo
| Badger

inductive game_step (G : Tree) (apples : apples_on_vertices G.vertices) : Type
| initial (v : G.vertices) : game_step
| move (v : G.vertices) (next : G.vertices) (v_adj_next : G.edges v next) : game_step

def alternating_turns (steps : list (game_step G apples)) : Prop :=
∃ badger_turns armadillo_turns,
badger_turns = list.filter (λs, match s with | game_step.move _ _ k => k % 2 = 0 | _ => true end) steps ∧
armadillo_turns = list.filter (λs, match s with | game_step.move _ _ k => k % 2 = 1 | _ => false end) steps

-- Theorem statement
theorem badger_wins (G : Tree) (apples : apples_on_vertices G.vertices) :
  ∀ steps : list (game_step G apples), alternating_turns steps -> Badger can always ensure optimal strategy thus Armadillo cannot ensure more apples :
 sorry

end badger_wins_l592_592521


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592939

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592939


namespace find_unknown_number_l592_592406

theorem find_unknown_number (x : ℝ) : 
  (1000 * 7) / (x * 17) = 10000 → x = 24.285714285714286 := by
  sorry

end find_unknown_number_l592_592406


namespace triangle_area_is_zero_l592_592389

-- Given conditions
variables {a b c : ℝ}
hypothesis h1 : a + b + c = 6
hypothesis h2 : a * b + b * c + c * a = 11
hypothesis h3 : a * b * c = 6

-- To prove
theorem triangle_area_is_zero (ha : a + b + c = 6) (hb : a * b + b * c + c * a = 11) (hc : a * b * c = 6) : 
  let p := (a + b + c) / 2 in
  let K^2 := p * (p - a) * (p - b) * (p - c) in
  K = 0 :=
by
  let p := 3
  have : (p - a) * (p - b) * (p - c) = 0
  sorry

end triangle_area_is_zero_l592_592389


namespace sin_315_degree_l592_592959

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592959


namespace product_of_invertibles_mod_120_l592_592333

open Nat

theorem product_of_invertibles_mod_120 :
  let m := 120
  let invertibles := { x | x < m ∧ gcd x m = 1 }
  ∏ a in invertibles, a % m = 119 :=
by
  sorry

end product_of_invertibles_mod_120_l592_592333


namespace spider_leg_pressure_l592_592491

-- Definitions based on given conditions
def weight_of_previous_spider := 6.4 -- ounces
def weight_multiplier := 2.5
def cross_sectional_area := 0.5 -- square inches
def number_of_legs := 8

-- Theorem stating the problem
theorem spider_leg_pressure : 
  (weight_multiplier * weight_of_previous_spider) / number_of_legs / cross_sectional_area = 4 := 
by 
  sorry

end spider_leg_pressure_l592_592491


namespace sin_315_equals_minus_sqrt2_div_2_l592_592986

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592986


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592931

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592931


namespace find_ab_monotonicity_solve_inequality_l592_592228

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (4 + x^2)

axiom a_condition : ∀ x, f a b x = -f a b (-x)
axiom b_condition : f a b 1 = -2 / 5

theorem find_ab (a b : ℝ) (ax_condition : a_condition) (bx_condition : b_condition) :
  a = -2 ∧ b = 0 :=
sorry

theorem monotonicity (a b : ℝ) (ax_condition : a_condition) (bx_condition : b_condition) :
  ∀ x1 x2, -2 < x1 → x1 < x2 → x2 < 2 → f a b x1 > f a b x2 :=
sorry

theorem solve_inequality (a b : ℝ) (ax_condition : a_condition) (bx_condition : b_condition) :
  ∀ m, 0 < m ∧ m < sqrt 2 → f a b (m - 2) + f a b (m^2 - m) > 0 :=
sorry

end find_ab_monotonicity_solve_inequality_l592_592228


namespace fraction_addition_l592_592905

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l592_592905


namespace new_mean_rent_l592_592496

theorem new_mean_rent
  (num_friends : ℕ)
  (avg_rent : ℕ)
  (original_rent_increased : ℕ)
  (increase_percentage : ℝ)
  (new_mean_rent : ℕ) :
  num_friends = 4 →
  avg_rent = 800 →
  original_rent_increased = 1400 →
  increase_percentage = 0.2 →
  new_mean_rent = 870 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_l592_592496


namespace range_f_l592_592224

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x - Real.cos x

theorem range_f : Set.range f = Set.Icc (-2 : ℝ) 2 := 
by
  sorry

end range_f_l592_592224


namespace arithmetic_sequence_value_and_formula_monotonically_increasing_sequence_range_l592_592192

noncomputable def positive_sequence_arithmetic_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 1 = t) ∧ (∀ n, S n = (a 0 + ... + a (n-1))) ∧ (∀ n, 2 * S n = a n * a (n + 1))

theorem arithmetic_sequence_value_and_formula {a : ℕ → ℝ} {S : ℕ → ℝ} {t : ℝ} 
  (h : positive_sequence_arithmetic_conditions a S t) (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) :
  t = 1 ∧ ∀ n, a n = n :=
by sorry

noncomputable def positive_sequence_monotonically_increasing_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 1 = t) ∧ (∀ n, S n = (a 0 + ... + a (n-1))) ∧ (∀ n, 2 * S n = a n * a (n + 1))

theorem monotonically_increasing_sequence_range {a : ℕ → ℝ} {S : ℕ → ℝ} {t : ℝ} 
  (h : positive_sequence_monotonically_increasing_conditions a S t) (h_mono : ∀ n, a (n + 1) > a n) :
  t ∈ (0, 2) :=
by sorry

end arithmetic_sequence_value_and_formula_monotonically_increasing_sequence_range_l592_592192


namespace vinny_initial_weight_correct_l592_592007

noncomputable def vinny_initial_weight (W_5: ℝ) (L_5: ℝ) : ℝ :=
  let W_4 := W_5 + L_5 in
  let L_4 := 2 * L_5 in
  let W_3 := W_4 + L_4 in
  let L_3 := L_4 / 2 in
  let W_2 := W_3 + L_3 in
  let L_2 := L_3 / 2 in
  let W_1 := W_2 + L_2 in
  let L_1 := 20 in
  (W_1 + L_1) -- initial weight

theorem vinny_initial_weight_correct :
  vinny_initial_weight 250.5 12 = 324.5 :=
by
  simp [vinny_initial_weight]
  sorry

end vinny_initial_weight_correct_l592_592007


namespace remaining_student_number_l592_592483

-- Definitions based on given conditions
def total_students := 48
def sample_size := 6
def sampled_students := [5, 21, 29, 37, 45]

-- Interval calculation and pattern definition based on systematic sampling
def sampling_interval := total_students / sample_size
def sampled_student_numbers (n : Nat) : Nat := 5 + sampling_interval * (n - 1)

-- Prove the student number within the sample
theorem remaining_student_number : ∃ n, n ∉ sampled_students ∧ sampled_student_numbers n = 13 :=
by
  sorry

end remaining_student_number_l592_592483


namespace cherries_used_l592_592481

theorem cherries_used (initial remaining used : ℕ) (h_initial : initial = 77) (h_remaining : remaining = 17) (h_used : used = initial - remaining) : used = 60 :=
by
  rw [h_initial, h_remaining] at h_used
  simp at h_used
  exact h_used

end cherries_used_l592_592481


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592941

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592941


namespace distance_between_points_l592_592437

def point1 : ℝ × ℝ := (3.5, -2)
def point2 : ℝ × ℝ := (7.5, 5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 65 := by
  sorry

end distance_between_points_l592_592437


namespace hyperbola_equation_l592_592717

noncomputable def h : ℝ := -4
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1

theorem hyperbola_equation :
  (h + k + a + b) = 0 := by
  have h := -4
  have k := 2
  have a := 1
  have b := 1
  show (-4 + 2 + 1 + 1) = 0
  sorry

end hyperbola_equation_l592_592717


namespace initial_distance_between_stations_l592_592424

-- Define the properties and conditions
variables (distance_AB : ℝ)
variables (speed_A speed_B : ℝ) (dist_A_when_meet : ℝ)
variables (simultaneous_start : Prop) (same_route : Prop) (constant_speed : Prop)

-- Assume the conditions given
def conditions := simultaneous_start ∧ same_route ∧ constant_speed ∧ 
  (speed_A = 20) ∧ (speed_B = 20) ∧ (dist_A_when_meet = 100)

-- Define the initial distance
def initial_distance := distance_AB = 200

-- The theorem statement
theorem initial_distance_between_stations (h : conditions) : initial_distance :=
sorry

end initial_distance_between_stations_l592_592424


namespace cows_count_24_l592_592845

-- Declare the conditions as given in the problem.
variables (D C : Nat)

-- Define the total number of legs and heads and the given condition.
def total_legs := 2 * D + 4 * C
def total_heads := D + C
axiom condition : total_legs = 2 * total_heads + 48

-- The goal is to prove that the number of cows C is 24.
theorem cows_count_24 : C = 24 :=
by
  sorry

end cows_count_24_l592_592845


namespace distinguishable_arrangements_l592_592240

theorem distinguishable_arrangements :
  let brown_tiles := 1
  let purple_tiles := 1
  let green_tiles := 2
  let yellow_tiles := 3
  let total_tiles := brown_tiles + purple_tiles + green_tiles + yellow_tiles
  multiset_permutations total_tiles [brown_tiles, purple_tiles, green_tiles, yellow_tiles] = 420 :=
by
  let brown_tiles := 1
  let purple_tiles := 1
  let green_tiles := 2
  let yellow_tiles := 3
  let total_tiles := brown_tiles + purple_tiles + green_tiles + yellow_tiles

  -- Function to calculate permutations of a multiset
  def multiset_permutations (n : ℕ) (counts : list ℕ) : ℕ :=
    n.factorial / counts.prod (λ x, x.factorial)

  -- Check that our parameters produce 420 distinguishable arrangements
  have h : multiset_permutations total_tiles [brown_tiles, purple_tiles, green_tiles, yellow_tiles] = 420 := sorry
  exact h

end distinguishable_arrangements_l592_592240


namespace harrison_croissant_expenditure_l592_592238

-- Define the conditions
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def weeks_in_year : ℕ := 52

-- Define the total cost of croissants in a year
def total_cost (cost_regular cost_almond : ℝ) (weeks : ℕ) : ℝ :=
  (weeks * cost_regular) + (weeks * cost_almond)

-- State the proof problem
theorem harrison_croissant_expenditure :
  total_cost cost_regular_croissant cost_almond_croissant weeks_in_year = 468.00 :=
by
  sorry

end harrison_croissant_expenditure_l592_592238


namespace angle_neg1120_in_fourth_quadrant_l592_592378

def angle_in_fourth_quadrant (θ : ℤ) : Prop :=
  let α := (θ % 360 + 360) % 360 in 270 ≤ α ∧ α < 360

theorem angle_neg1120_in_fourth_quadrant : angle_in_fourth_quadrant (-1120) :=
by
  sorry

end angle_neg1120_in_fourth_quadrant_l592_592378


namespace solve_x4_plus_100_eq_0_l592_592154

open Complex

theorem solve_x4_plus_100_eq_0 :
  ∀ x : ℂ, x^4 + 100 = 0 ↔ x = (√5 + √5 * Complex.i) ∨ x = (-√5 - √5 * Complex.i) ∨ x = (√5 * Complex.i - √5) ∨ x = (-√5 * Complex.i + √5) :=
by 
  sorry

end solve_x4_plus_100_eq_0_l592_592154


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592990

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592990


namespace problem_coefficient_term_problem_sum_coefficients_l592_592633

theorem problem_coefficient_term (n : ℕ) (a : ℝ) (h_n : n = 6)
  (h_coeff : (Nat.choose n 2) * 2^2 * a^(n - 2) = 960) 
  (h_a_pos : a > 0) (h_n_pos : n > 1) : a = 2 := 
  sorry

theorem problem_sum_coefficients 
  (n : ℕ) (a : ℝ) 
  (h_sum_coeffs : (a + 2)^n = 3^10)
  (h_na : n + a = 12) 
  (h_a_pos : a > 0) (h_n_pos : n > 1) : 
  (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3 + Nat.choose n 4 + Nat.choose n5 + Nat.choose n(n - 1) + Nat.choose n0) = 32 := sorry

end problem_coefficient_term_problem_sum_coefficients_l592_592633


namespace sum_distances_foci_l592_592635

-- Define variables for the problem
variables {x y m : ℝ} (B : Set ℝ)

-- Given conditions
def ellipse_eq (x y : ℝ) := (x^2 / 4) + (y^2 / m) = 1
def point_on_ellipse (B : Set ℝ) := B = {0, 4}

-- The goal is to prove that the sum of distances from any point on the ellipse to the two foci is 8
theorem sum_distances_foci (h₁ : ellipse_eq 0 4) (h₂ : point_on_ellipse {0, 4}) :
  ∃ a : ℝ, 2 * a = 8 :=
  sorry

end sum_distances_foci_l592_592635


namespace expansion_constant_term_l592_592142

noncomputable def constant_term_in_expansion : ℚ :=
  let binomial_coef := Nat.choose 6 4 in
  let value := binomial_coef * (-1)^4 * (1/2)^2 in
  value

theorem expansion_constant_term :
  constant_term_in_expansion = 15 / 4 := 
by
  sorry

end expansion_constant_term_l592_592142


namespace taking_statistics_is_23_l592_592523

-- Definitions for conditions
variable total_players : ℕ := 25
variable taking_physics : ℕ := 10
variable taking_both : ℕ := 6

-- The number of players taking statistics
def taking_statistics : ℕ :=
  total_players - (taking_physics - taking_both) + taking_both

-- Proof statement
theorem taking_statistics_is_23 : taking_statistics = 23 := by
  sorry

end taking_statistics_is_23_l592_592523


namespace find_x_axis_intercept_l592_592528

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end find_x_axis_intercept_l592_592528


namespace find_integer_n_l592_592583

def floor_sqrt (n : ℕ) : ℕ := ↑⌊(Real.sqrt n)⌋

def floor_sqrt_sqrt (n : ℕ) : ℕ := ↑⌊(Real.sqrt (Real.sqrt n))⌋

theorem find_integer_n :
  ∃ (n : ℕ), n + floor_sqrt n + floor_sqrt_sqrt n = 2017 ∧ n = 1967 :=
begin
  sorry
end

end find_integer_n_l592_592583


namespace line_equation_of_l1_l592_592215

def is_tangent_to_circle (l : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, m) := l
  let (h, k, r) := c
  abs (a * h + b * k + m) / real.sqrt (a^2 + b^2) = r

def is_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  let (a1, b1, _) := l1
  let (a2, b2, _) := l2
  a1 * b2 = a2 * b1

theorem line_equation_of_l1 :
  ∃ (m : ℝ), 
  (is_tangent_to_circle (3, 4, m) (0, -1, 1) ∧ is_parallel (3, 4, m) (3, 4, -6) ∧ 
    (m = -1 ∨ m = 9)) :=
by
  -- sorry to skip the proof
  sorry

end line_equation_of_l1_l592_592215


namespace find_train_length_l592_592512

noncomputable def speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 245.03
noncomputable def time_seconds : ℝ := 30
noncomputable def speed_ms : ℝ := (speed_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := speed_ms * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length

theorem find_train_length : train_length = 129.97 := 
by
  sorry

end find_train_length_l592_592512


namespace find_y_from_condition_l592_592449

variable (y : ℝ) (h : (3 * y) / 7 = 15)

theorem find_y_from_condition : y = 35 :=
by {
  sorry
}

end find_y_from_condition_l592_592449


namespace modular_home_total_cost_l592_592140

theorem modular_home_total_cost :
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  total_cost = 249000 := 
by
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  have h : total_cost = 249000 := sorry
  exact h

end modular_home_total_cost_l592_592140


namespace checkerboard_square_count_l592_592856

/-- 
Prove that the total number of distinct squares with sides on the grid lines of a 
9 by 9 checkerboard and containing at least 6 black squares, can be drawn on the checkerboard.
-/
theorem checkerboard_square_count : 
  ∀ (board : ℕ) [hboard : board = 9], 
  (∑ (n : ℕ) in {4, 5, 6, 7, 8, 9}, (board - n + 1) * (board - n + 1)) = 91 :=
begin
  intros,
  -- This part contains the proof details, skipped for brevity
  sorry
end

end checkerboard_square_count_l592_592856


namespace shipping_cost_l592_592741

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

/-- Lizzy's total shipping cost for 540 pounds of fish packed in 30-pound crates at $1.5 per crate is $27. -/
theorem shipping_cost : (total_weight / weight_per_crate) * cost_per_crate = 27 := by
  sorry

end shipping_cost_l592_592741


namespace equivalence_of_statements_l592_592450

theorem equivalence_of_statements 
  (Q P : Prop) :
  (Q → ¬ P) ↔ (P → ¬ Q) := sorry

end equivalence_of_statements_l592_592450


namespace simplify_fraction_l592_592760

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l592_592760


namespace directrix_of_parabola_l592_592579

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end directrix_of_parabola_l592_592579


namespace count_integers_expression_negative_l592_592173

theorem count_integers_expression_negative :
  ∃ n : ℕ, n = 4 ∧ 
  ∀ x : ℤ, x^4 - 60 * x^2 + 144 < 0 → n = 4 := by
  -- Placeholder for the proof
  sorry

end count_integers_expression_negative_l592_592173


namespace sum_of_digits_of_expression_l592_592049

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l592_592049


namespace find_integer_n_l592_592244

theorem find_integer_n (n : ℤ) :
  (⌊ (n^2 : ℤ) / 9 ⌋ - ⌊ n / 3 ⌋^2 = 3) → (n = 8 ∨ n = 10) :=
  sorry

end find_integer_n_l592_592244


namespace decimal_equiv_of_fraction_l592_592461

theorem decimal_equiv_of_fraction : (1 / 5) ^ 2 = 0.04 := by
  sorry

end decimal_equiv_of_fraction_l592_592461


namespace imaginary_part_eq_neg_one_l592_592631

def z : ℂ := 1 + complex.i

def conjugate_z : ℂ := 1 - complex.i

theorem imaginary_part_eq_neg_one : complex.im ((4 : ℂ) / z - conjugate_z) = -1 :=
by
  sorry

end imaginary_part_eq_neg_one_l592_592631


namespace sin_45_degrees_l592_592551

noncomputable def Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem sin_45_degrees : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_degrees_l592_592551


namespace distinct_squares_with_at_least_6_black_l592_592858

-- Define the checkerboard
def checkerboard := List (List Bool)

-- Define the property of containing at least 6 black squares
def contains_at_least_6_black_squares (square : checkerboard) : Prop :=
  square.flatten.count (· = false) ≥ 6

-- Define a function to count squares of arbitrary size on a 9x9 checkerboard
def count_squares_at_least_6_black (board : checkerboard) : Nat :=
  let sizes := [4, 5, 6, 7, 8, 9]
  sizes.sum (λ size, (9 - size + 1) * (9 - size + 1))

-- The proof goal
theorem distinct_squares_with_at_least_6_black (board : checkerboard) (h : ∀ x y, board.get x y = x % 2 = y % 2) :
  count_squares_at_least_6_black board = 91 :=
by
  sorry

end distinct_squares_with_at_least_6_black_l592_592858


namespace sin_315_eq_neg_sqrt2_div_2_l592_592970

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592970


namespace sum_of_digits_l592_592062

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592062


namespace sin_315_equals_minus_sqrt2_div_2_l592_592983

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592983


namespace total_coins_l592_592298
-- Import the necessary library

-- Defining the conditions
def quarters := 22
def dimes := quarters + 3
def nickels := quarters - 6

-- Main theorem statement
theorem total_coins : (quarters + dimes + nickels) = 63 := by
  sorry

end total_coins_l592_592298


namespace opposite_numbers_power_l592_592245

theorem opposite_numbers_power (a b : ℝ) (h : a + b = 0) : (a + b) ^ 2023 = 0 :=
by 
  sorry

end opposite_numbers_power_l592_592245


namespace min_cost_3_each_type_l592_592513

-- Definitions based on conditions
def num_orange_candies (O : ℕ) := O
def num_apple_candies (O : ℕ) := 2 * O
def num_grape_candies (G : ℕ) := G
def num_strawberry_candies (G : ℕ) := 2 * G

-- Main statement
theorem min_cost_3_each_type (O G : ℕ) 
  (h1 : num_orange_candies O = num_strawberry_candies G)
  (h2 : num_apple_candies O = 2 * num_strawberry_candies G)
  (h3 : G = 3) : 0.1 * (num_apple_candies O + num_orange_candies O + num_grape_candies G + num_strawberry_candies G) = 2.7 := 
by
  sorry

end min_cost_3_each_type_l592_592513


namespace infinite_common_terms_l592_592008

noncomputable section 

def sequence_a : ℕ → ℤ
| 0       := 2
| 1       := 14
| (n + 2) := 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0       := 2
| 1       := 14
| (n + 2) := 6 * sequence_b (n + 1) - sequence_b n

theorem infinite_common_terms (a_n b_n : ℕ → ℤ)
  (h₀ : a_n 0 = 2) (h₁ : a_n 1 = 14)
  (rec_a : ∀ n ≥ 2, a_n n = 14 * a_n (n - 1) + a_n (n - 2))
  (h'₀ : b_n 0 = 2) (h'₁ : b_n 1 = 14)
  (rec_b : ∀ n ≥ 2, b_n n = 6 * b_n (n - 1) - b_n (n - 2)) :
  ∃ f : ℕ → ℕ, function.injective f ∧ ∀ n, ∃ k, a_n n = b_n (f n) :=
begin
  sorry
end

end infinite_common_terms_l592_592008


namespace shop_length_l592_592398

def monthly_rent : ℝ := 2244
def width : ℝ := 18
def annual_rent_per_sqft : ℝ := 68

theorem shop_length : 
  (monthly_rent * 12 / annual_rent_per_sqft / width) = 22 := 
by
  -- Proof omitted
  sorry

end shop_length_l592_592398


namespace correct_equation_for_annual_consumption_l592_592867

-- Definitions based on the problem conditions
-- average_monthly_consumption_first_half is the average monthly electricity consumption in the first half of the year, assumed to be x
def average_monthly_consumption_first_half (x : ℝ) := x

-- average_monthly_consumption_second_half is the average monthly consumption in the second half of the year, i.e., x - 2000
def average_monthly_consumption_second_half (x : ℝ) := x - 2000

-- total_annual_consumption is the total annual electricity consumption which is 150000 kWh
def total_annual_consumption (x : ℝ) := 6 * average_monthly_consumption_first_half x + 6 * average_monthly_consumption_second_half x

-- The main theorem statement which we need to prove
theorem correct_equation_for_annual_consumption (x : ℝ) : total_annual_consumption x = 150000 :=
by
  -- equation derivation
  sorry

end correct_equation_for_annual_consumption_l592_592867


namespace blue_pill_cost_l592_592533

theorem blue_pill_cost :
  ∃ y : ℝ, ∀ (red_pill_cost blue_pill_cost : ℝ),
    (blue_pill_cost = red_pill_cost + 2) ∧
    (21 * (blue_pill_cost + red_pill_cost) = 819) →
    blue_pill_cost = 20.5 :=
by sorry

end blue_pill_cost_l592_592533


namespace prepaid_card_cost_correct_l592_592713

noncomputable def prepaid_phone_card_cost
    (cost_per_minute : ℝ) (call_minutes : ℝ) (remaining_credit : ℝ) : ℝ :=
  remaining_credit + (call_minutes * cost_per_minute)

theorem prepaid_card_cost_correct :
  let cost_per_minute := 0.16
  let call_minutes := 22
  let remaining_credit := 26.48
  prepaid_phone_card_cost cost_per_minute call_minutes remaining_credit = 30.00 := by
  sorry

end prepaid_card_cost_correct_l592_592713


namespace sandwich_cost_90_cents_l592_592295

theorem sandwich_cost_90_cents :
  let cost_bread := 0.15
  let cost_ham := 0.25
  let cost_cheese := 0.35
  (2 * cost_bread + cost_ham + cost_cheese) * 100 = 90 := 
by
  sorry

end sandwich_cost_90_cents_l592_592295


namespace find_b_if_polynomial_is_square_l592_592787

theorem find_b_if_polynomial_is_square (a b : ℚ) (h : ∃ g : ℚ[X], (g ^ 2) = (X^4 + X^3 + 2*X^2 + a*X + b)) : b = 49/64 :=
by {
  sorry
}

end find_b_if_polynomial_is_square_l592_592787


namespace sin_five_pi_over_six_l592_592537

theorem sin_five_pi_over_six : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
  sorry

end sin_five_pi_over_six_l592_592537


namespace ellipse_equation_perpendicular_slopes_l592_592201

-- Define the problem conditions and the proofs required

-- Part 1: Prove the equation of the ellipse C
theorem ellipse_equation
  (a b : ℝ) (h : a > b ∧ b > 0)
  (ellipse_foci_minor : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) →
    ((x^2 + y^2 = 1) ∨ (x^2 + y^2 = a^2 - b^2))) :
  (a = sqrt 2 ∧ b = 1) →
  ∀ (x y : ℝ), x^2 / 2 + y^2 = 1 :=
by
  sorry

-- Part 2: Discuss values of k for OA ⊥ OB for lines passing through (2,0)
theorem perpendicular_slopes
  (k : ℝ) (line_eq : ∀ (x y : ℝ), 
  (y = k * (x - 2)) → 
  (x^2 / 2 + y^2 = 1) ∧ (point_of_intersection : x = a ∨ x = b)) :
  (∀ x1 y1 x2 y2 : ℝ, (x1^2 / 2 + y1^2 = 1) ∧ (x2^2 / 2 + y2^2 = 1) ∧ 
    (y1 = k * (x1 - 2)) ∧ (y2 = k * (x2 - 2)) ∧ x1 ≠ x2 →
    ((x1 * x2 + y1 * y2 = 0) ↔ (k = sqrt 5 / 5 ∨ k = -sqrt 5 / 5))) :=
by 
  sorry

end ellipse_equation_perpendicular_slopes_l592_592201


namespace P_is_linear_l592_592209

-- Define the arithmetic sequence and common difference
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Define the polynomial P_n(x)
def P (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * nat.choose n k * (1 - x)^(n - k) * x^k

-- The main theorem we need to prove
theorem P_is_linear (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (h_arith : is_arithmetic_sequence a d) :
  ∃ b c : ℝ, ∀ x : ℝ, P a n x = b + c * x ∧ (c ≠ 0) := 
sorry

end P_is_linear_l592_592209


namespace sin_315_eq_neg_sqrt_2_div_2_l592_592942

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l592_592942


namespace sum_digits_2_pow_2010_5_pow_2012_7_l592_592031

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l592_592031


namespace cos_double_angle_identity_l592_592180

theorem cos_double_angle_identity (x : ℝ) (h : Real.sin (Real.pi / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 :=
sorry

end cos_double_angle_identity_l592_592180


namespace sum_digits_probability_l592_592888

noncomputable def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def numInRange : ℕ := 1000000

noncomputable def coefficient : ℕ :=
  Nat.choose 24 5 - 6 * Nat.choose 14 5

noncomputable def probability : ℚ :=
  coefficient / numInRange

theorem sum_digits_probability :
  probability = 7623 / 250000 :=
by
  sorry

end sum_digits_probability_l592_592888


namespace product_invertibles_mod_120_l592_592311

theorem product_invertibles_mod_120 :
  let n := (list.filter (λ k, Nat.coprime k 120) (list.range 120)).prod
  in n % 120 = 119 :=
by
  sorry

end product_invertibles_mod_120_l592_592311


namespace atmosphere_depth_l592_592864

/-- The height of the cone-shaped peak -/
def coneHeight : ℝ := 5000

/-- The fraction of the top volume above the atmosphere -/
def volumeFractionAboveAtmosphere : ℝ := 1 / 5

/-- The calculated depth of the atmosphere at the peak's base. 
This is what we're proving to match the given depth. -/
def depthOfAtmosphere : ℝ := 340

/-- 
Given the height of the cone and the fraction of its volume above 
the atmosphere, we want to prove that the depth of the atmosphere 
at the base of the peak is 340 meters.
-/
theorem atmosphere_depth (h : ℝ) (v_frac : ℝ) (d : ℝ) 
  (h_def : h = coneHeight) 
  (v_frac_def : v_frac = volumeFractionAboveAtmosphere)
  (d_def : d = depthOfAtmosphere) :
  d = h - (h * real.sqrt (real.sqrt (4 / 5))) :=
by
  rw [h_def, v_frac_def, d_def]
  sorry

end atmosphere_depth_l592_592864


namespace kit_time_to_ticket_window_l592_592300

def distance_covered (time : ℕ) (rate : ℕ) : ℕ := rate * time

def time_required (distance : ℕ) (rate : ℕ) : ℕ := distance / rate

theorem kit_time_to_ticket_window :
  let rate := 2 in
  let remaining_distance := 90 * 3 in
  time_required remaining_distance rate = 135 := by
  sorry

end kit_time_to_ticket_window_l592_592300


namespace pq_sum_l592_592279

def single_digit (n : ℕ) : Prop := n < 10

theorem pq_sum (P Q : ℕ) (hP : single_digit P) (hQ : single_digit Q)
  (hSum : P * 100 + Q * 10 + Q + P * 110 + Q + Q * 111 = 876) : P + Q = 5 :=
by 
  -- Here we assume the expected outcome based on the problem solution
  sorry

end pq_sum_l592_592279


namespace sin_315_degree_l592_592961

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l592_592961


namespace compound_interest_calculation_l592_592159

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  let A := P * (1 + r / n) ^ (n * t)
  in A - P

theorem compound_interest_calculation : 
  compound_interest 2500 0.25 1 6 = 9707.03 := 
  by
    -- skip the proof
    sorry

end compound_interest_calculation_l592_592159


namespace number_of_integer_points_EM_plus_MB_l592_592850

-- Definitions based on the problem conditions
structure Point where
  x : ℝ
  y : ℝ

def E := Point.mk 2 4

def M (t : ℝ) := Point.mk t t

def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def EM (t : ℝ) : ℝ := distance (M t) E

def B := Point.mk 4 4

def MB (t : ℝ) : ℝ := distance (M t) B

def EM_plus_MB (t : ℝ) : ℝ := EM t + MB t

theorem number_of_integer_points_EM_plus_MB : 
  {t : ℝ | 0 ≤ t ∧ t ≤ 4 ∧ ∃ n : ℤ, EM_plus_MB t = n}.to_finset.card = 5 :=
by
  sorry

end number_of_integer_points_EM_plus_MB_l592_592850


namespace sum_of_digits_l592_592061

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l592_592061


namespace quadrilateral_IO_equal_OJ_l592_592693

-- Define the basic setup and conditions
variables {A B C D E F G H I J O : Point}
variables {a b c d k h λ : ℝ}
variables {AB CD AD BC : Line}

-- Define points and condition relation AB = CD
def AB_line (A B : Point) (O : Point) := -- Definition equivalent to AB
def CD_line (C D : Point) (O : Point) := -- Definition equivalent to CD

-- The proof statement in Lean 4
theorem quadrilateral_IO_equal_OJ
  (AB_eq_CD : length AB = length CD)
  (OIntersectionACBD : is_intersection AC BD O)
  (EIntersectionAD : is_intersection AD ?line E)
  (FIntersectionBC : is_intersection BC ?line F)
  (GIntersectionAB : is_intersection AB ?line G)
  (HIntersectionCD : is_intersection CD ?line H)
  (IIntersectionGFBD : is_intersection GF BD I)
  (JIntersectionEHBD : is_intersection EH BD J):
  dist I O = dist O J :=
begin
  -- Statement to be proved with the imported libraries and conditions given
  sorry
end

end quadrilateral_IO_equal_OJ_l592_592693


namespace simplify_fraction_l592_592762

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l592_592762


namespace values_a_seq_general_a_seq_l592_592617

noncomputable def Sn (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum a

def a_seq : ℕ → ℚ
| 0       := 1
| (n + 1) := (Sn a_seq n) / 3

theorem values_a_seq :
  a_seq 1 = 1 / 3 ∧ a_seq 2 = 4 / 9 ∧ a_seq 3 = 16 / 27 :=
by {
  split; 
  { have hSum0 : Sn a_seq 1 = 1, by simp [Sn],
    have hSum1 : Sn a_seq 2 = 4 / 3, by simp [Sn, hSum0],
    exacts [
      show a_seq 1 = (Sn a_seq 1) / 3, by rw [a_seq, hSum0 / 3],
      show a_seq 2 = (Sn a_seq 2) / 3, by rw [a_seq, hSum1 / 3],
      show a_seq 3 = (Sn a_seq 3) / 3, by {
        have hSum2 : Sn a_seq 3 = 16 / 9, by {
          simp [Sn, hSum0, hSum1],
        },
        rw [a_seq, hSum2 / 3]
      }
    ]
  }
}

theorem general_a_seq (n : ℕ) :
  a_seq n = if n = 0 then 1 else (1 / 3) * (4 / 3)^(n - 1) :=
by {
  induction n with d hd,
  { refl },
  case succ: d hd {
    rw [a_seq, Sn],
    split_ifs,
    { simp },
    { sorry }
  }
}

end values_a_seq_general_a_seq_l592_592617


namespace max_distance_is_7_l592_592346

noncomputable def z2 : ℂ := 3 - 4 * complex.I
def R : ℝ := 2
def max_distance_from_z1_to_z2 (z1 : ℂ) (z2 : ℂ) : ℝ := complex.abs(z1 - z2)
def in_circle (z1 : ℂ) (R : ℝ) : Prop := complex.abs z1 ≤ R

theorem max_distance_is_7 (z1 : ℂ) (h : in_circle z1 R) : max_distance_from_z1_to_z2 z1 z2 ≤ 7 :=
by
  have h_dist_zero_to_z2 : complex.abs z2 = 5 := by simp [z2, complex.abs]
  have radius_add_dist : 2 + 5 = 7 := by simp
  exact sorry

end max_distance_is_7_l592_592346


namespace intersection_of_A_and_B_l592_592662

namespace SetsIntersectionProof

def setA : Set ℝ := { x | |x| ≤ 2 }
def setB : Set ℝ := { x | x < 1 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | -2 ≤ x ∧ x < 1 } :=
sorry

end SetsIntersectionProof

end intersection_of_A_and_B_l592_592662


namespace unit_digit_product_7858_1086_4582_9783_l592_592460

theorem unit_digit_product_7858_1086_4582_9783 : 
  (7858 * 1086 * 4582 * 9783) % 10 = 8 :=
by
  -- Given that the unit digits of the numbers are 8, 6, 2, and 3.
  let d1 := 7858 % 10 -- This unit digit is 8
  let d2 := 1086 % 10 -- This unit digit is 6
  let d3 := 4582 % 10 -- This unit digit is 2
  let d4 := 9783 % 10 -- This unit digit is 3
  -- We need to prove that the unit digit of the product is 8
  sorry -- The actual proof steps are skipped

end unit_digit_product_7858_1086_4582_9783_l592_592460


namespace stratified_sampling_sample_size_l592_592863

-- Definitions based on conditions
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def female_employees_in_sample : ℕ := 3

-- Proof statement
theorem stratified_sampling_sample_size : total_employees = 120 ∧ male_employees = 90 ∧ female_employees_in_sample = 3 → 
  (female_employees_in_sample + female_employees_in_sample * (male_employees / (total_employees - male_employees))) = 12 :=
sorry

end stratified_sampling_sample_size_l592_592863


namespace product_of_invertibles_mod_120_l592_592334

open Nat

theorem product_of_invertibles_mod_120 :
  let m := 120
  let invertibles := { x | x < m ∧ gcd x m = 1 }
  ∏ a in invertibles, a % m = 119 :=
by
  sorry

end product_of_invertibles_mod_120_l592_592334


namespace nine_x_eq_64_l592_592659

theorem nine_x_eq_64 (x : ℝ) : 9^x = 64 ↔ x = 3 * log 3 2 :=
sorry

end nine_x_eq_64_l592_592659


namespace sum_of_digits_of_expression_l592_592037

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l592_592037


namespace eval_diamond_expr_l592_592557

def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4
  | (1, 2) => 3
  | (1, 3) => 2
  | (1, 4) => 1
  | (2, 1) => 1
  | (2, 2) => 4
  | (2, 3) => 3
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 1
  | (3, 3) => 4
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 1
  | (4, 4) => 4
  | (_, _) => 0  -- This handles any case outside of 1,2,3,4 which should ideally not happen

theorem eval_diamond_expr : diamond (diamond 3 4) (diamond 2 1) = 2 := by
  sorry

end eval_diamond_expr_l592_592557


namespace range_of_a_l592_592206

noncomputable def valid_a (a : ℝ) : Prop :=
(a ≤ 2 / 3) ∧ (1 / 2 < a ∧ a < 1)

theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a)
    (h2 : ∀ x : ℝ, ∀ (y : ℝ), y = (2 * a - 1)^x → decreasing y) :
  valid_a a :=
by
  sorry

end range_of_a_l592_592206


namespace lara_puts_flowers_in_vase_l592_592712

theorem lara_puts_flowers_in_vase : 
  ∀ (total_flowers mom_flowers flowers_given_more : ℕ), 
    total_flowers = 52 →
    mom_flowers = 15 →
    flowers_given_more = 6 →
  (total_flowers - (mom_flowers + (mom_flowers + flowers_given_more))) = 16 :=
by
  intros total_flowers mom_flowers flowers_given_more h1 h2 h3
  sorry

end lara_puts_flowers_in_vase_l592_592712


namespace find_d_of_triple_product_l592_592811

variables {V : Type} [inner_product_space ℝ V]
variables (i j k l : V)
variables (v : V)
-- Assuming orthonormal vectors i, j, k, l
variable (h_orthonormal : orthonormal ℝ ![i, j, k, l])

theorem find_d_of_triple_product :
  (i × (v × i) + j × (v × j) + k × (v × k) + l × (v × l) = (3 : ℝ) • v) :=
sorry

end find_d_of_triple_product_l592_592811


namespace Andrey_Gleb_distance_l592_592782

theorem Andrey_Gleb_distance (AB VG : ℕ) (AG : ℕ) (BV : ℕ) (cond1 : AB = 600) (cond2 : VG = 600) (cond3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := 
sorry

end Andrey_Gleb_distance_l592_592782


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592022

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l592_592022


namespace students_chose_greek_food_l592_592383
  
theorem students_chose_greek_food (total_students : ℕ) (percentage_greek : ℝ) (h1 : total_students = 200) (h2 : percentage_greek = 0.5) :
  (percentage_greek * total_students : ℝ) = 100 :=
by
  rw [h1, h2]
  norm_num
  sorry

end students_chose_greek_food_l592_592383


namespace sin_315_eq_neg_sqrt2_div_2_l592_592915

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592915


namespace exists_similar_1995_digit_numbers_l592_592432

open Nat

theorem exists_similar_1995_digit_numbers :
  ∃ (A B C : ℕ),
    (A.to_digits  10).length = 1995 ∧
    (B.to_digits  10).length = 1995 ∧
    (C.to_digits  10).length = 1995 ∧
    (∀ x ∈ A.to_digits  10, x ≠ 0) ∧
    (∀ x ∈ B.to_digits  10, x ≠ 0) ∧
    (∀ x ∈ C.to_digits  10, x ≠ 0) ∧
    list.perm (A.to_digits  10) (B.to_digits  10) ∧
    list.perm (A.to_digits  10) (C.to_digits  10) ∧
    A + B = C :=
  by
  sorry

end exists_similar_1995_digit_numbers_l592_592432


namespace probability_at_least_one_defective_bulb_l592_592842

theorem probability_at_least_one_defective_bulb (total_bulbs : ℕ) (defective_bulbs : ℕ) (chosen_bulbs : ℕ) :
  total_bulbs = 20 → defective_bulbs = 4 → chosen_bulbs = 2 →
  let non_defective_bulbs := total_bulbs - defective_bulbs
  in
  let p_first_non_defective := (non_defective_bulbs : ℝ) / (total_bulbs : ℝ)
  in
  let p_second_non_defective := (non_defective_bulbs - 1 : ℝ) / (total_bulbs - 1 : ℝ)
  in
  let p_both_non_defective := p_first_non_defective * p_second_non_defective
  in
  let p_at_least_one_defective := 1 - p_both_non_defective
  in
  p_at_least_one_defective = 7/19 :=
by
  intros h1 h2 h3
  sorry

end probability_at_least_one_defective_bulb_l592_592842


namespace smallest_perfect_square_greater_than_x_l592_592202

theorem smallest_perfect_square_greater_than_x (x : ℤ)
  (h₁ : ∃ k : ℤ, k^2 ≠ x)
  (h₂ : x ≥ 0) :
  ∃ n : ℤ, n^2 > x ∧ ∀ m : ℤ, m^2 > x → n^2 ≤ m^2 :=
sorry

end smallest_perfect_square_greater_than_x_l592_592202


namespace logarithm_identity_l592_592369

theorem logarithm_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 4 * y^2 = 12 * x * y) : log 10 (x + 2 * y) - 2 * log 10 2 = (1 / 2) * (log 10 x + log 10 y) := 
by
  sorry

end logarithm_identity_l592_592369


namespace similar_triangles_l592_592652

namespace Geometry

variables {A B C P : Point} {circumABC : Circle}

def symmetric_point (p : Point) (l : Line) : Point := sorry

noncomputable def second_intersection (l : Line) (c : Circle) : Point := sorry

def triangle_similar (Δ₁ Δ₂ : Triangle) : Prop := sorry

theorem similar_triangles
  (ABC : Triangle) (circumABC : Circle)
  (P : Point)
  (HP : ∀ (Q : Point), Q ∈ circumABC ↔ ∃ (k : Line), Q ∈ k ∧ P ∈ k)
  (A1 := second_intersection (line_through A P) circumABC)
  (B1 := second_intersection (line_through B P) circumABC)
  (C1 := second_intersection (line_through C P) circumABC)
  (A2 := symmetric_point A1 (line_through B C))
  (B2 := symmetric_point B1 (line_through C A))
  (C2 := symmetric_point C1 (line_through A B)) :
  triangle_similar (triangle A1 B1 C1) (triangle A2 B2 C2) :=
sorry

end Geometry

end similar_triangles_l592_592652


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592046

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592046


namespace rectangle_perimeter_l592_592684

theorem rectangle_perimeter (z w : ℝ) (h : z > w) :
  (2 * ((z - w) + w)) = 2 * z := by
  sorry

end rectangle_perimeter_l592_592684


namespace sin_315_equals_minus_sqrt2_div_2_l592_592988

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l592_592988


namespace minimize_quadratic_expression_l592_592447

theorem minimize_quadratic_expression :
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, (y^2 - 6*y + 8) ≥ (x^2 - 6*x + 8) := by
sorry

end minimize_quadratic_expression_l592_592447


namespace sum_of_roots_l592_592144

/-!
# Proof problem
Prove that the sum of all real numbers x satisfying
(x^3 - 6x^2 + 11x - 3)^(x^2 - 6x + 8) = 1 is 6.
-/

theorem sum_of_roots : 
  ∑ x in { x : ℝ | (x^3 - 6 * x^2 + 11 * x - 3) ^ (x^2 - 6 * x + 8) = 1 }, x = 6 :=
by
  sorry

end sum_of_roots_l592_592144


namespace point_M_coordinates_l592_592694

noncomputable theory
open Real

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4
def is_left_half_x_axis (M : ℝ × ℝ) : Prop := M.1 < 0 ∧ M.2 = 0
def tangent_cond (M : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop := 
  tangent_to_circle M A circle_O ∧ 
  intersect_circle A B C circle_O1 ∧ 
  seg_len_eq A B B C

theorem point_M_coordinates :
  ∃ M : ℝ × ℝ, 
  is_left_half_x_axis M ∧ 
  (∃ A B C : ℝ × ℝ, tangent_cond M A B C) ∧ 
  M = (-4, 0) :=
begin
  sorry
end

end point_M_coordinates_l592_592694


namespace fraction_addition_l592_592893

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l592_592893


namespace original_number_l592_592012

theorem original_number (x : ℤ) (h : x + 24 ≡ 0 [MOD 27]) : x = 30 :=
sorry

end original_number_l592_592012


namespace part1_part2_l592_592210

noncomputable section

variables (α : ℝ)

def condition : Prop := sin α + cos α = -1 / 5

theorem part1 (h1 : condition α) :
  sin (π / 2 + α) * cos (π / 2 - α) = -12 / 25 := by
  sorry

theorem part2 (h1 : condition α) (h2 : π / 2 < α) (h3 : α < π):
  1 / sin (π - α) + 1 / cos (π - α) = 35 / 12 := by
  sorry

end part1_part2_l592_592210


namespace sum_of_areas_proof_l592_592462

noncomputable theory

-- Define the conditions
def leg_length : ℝ := 36
def initial_triangle_area (a : ℝ) : ℝ := (a * a) / 2

-- Define the area of an equilateral triangle with side length a
def equilateral_triangle_area (a : ℝ) : ℝ := (a * a * Math.sqrt 3) / 4

-- Define the geometric series sum function
def geometric_series_sum (a r : ℝ) : ℝ :=
  if hr : r < 1 then a / (1 - r) else 0

-- Define the sum of areas of the infinite series of equilateral triangles

def sum_of_areas_of_equilateral_triangles : ℝ :=
  geometric_series_sum (equilateral_triangle_area leg_length) (1/2)

-- Prove the total area of all equilateral triangles
theorem sum_of_areas_proof : sum_of_areas_of_equilateral_triangles = 324 :=
  sorry

end sum_of_areas_proof_l592_592462


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592047

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l592_592047


namespace sum_G_eq_676797_l592_592595

-- Define the function G
def G (n : ℕ) : ℕ := 2 * n^2 + 1

-- The sum of G(n) from n = 2 to n = 100
theorem sum_G_eq_676797 :
  ∑ n in Finset.range (100 - 1) + 2, G n = 676797 :=
by
  intros
  -- Proof omitted
  sorry

end sum_G_eq_676797_l592_592595


namespace simplify_fraction_l592_592759

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l592_592759


namespace coin_toss_sequences_l592_592264

theorem coin_toss_sequences : 
  ∃ (s : Set (List Bool)), 
    (∀ seq ∈ s, length seq = 17 ∧
      (∃ HH HT TH TT, 
        (count_subsequences seq [true, true] = HH ∧ 
         count_subsequences seq [true, false] = HT ∧ 
         count_subsequences seq [false, true] = TH ∧ 
         count_subsequences seq [false, false] = TT) ∧ 
         HH = 3 ∧ HT = 2 ∧ TH = 5 ∧ TT = 6)) ∧
    fintype.card s = 840 := 
sorry

/-- Helper function to count the number of times a particular subsequence occurs in a list. -/
noncomputable def count_subsequences (l : List Bool) (subseq : List Bool) : Nat :=
  sorry

end coin_toss_sequences_l592_592264


namespace sin_315_eq_neg_sqrt2_div_2_l592_592912

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l592_592912


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l592_592927

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l592_592927


namespace frictional_force_is_12N_l592_592509

-- Given conditions
variables (m1 m2 a μ : ℝ)
-- Constants
def g : ℝ := 9.8

-- Frictional force on the tank
def F_friction : ℝ := μ * m1 * g

-- Proof statement
theorem frictional_force_is_12N (m1_value : m1 = 3) (m2_value : m2 = 15) (a_value : a = 4) (μ_value : μ = 0.6) :
  m1 * a = 12 :=
by
  sorry

end frictional_force_is_12N_l592_592509


namespace table_arrangement_division_l592_592256

theorem table_arrangement_division (total_tables : ℕ) (rows : ℕ) (tables_per_row : ℕ) (tables_left_over : ℕ)
    (h1 : total_tables = 74) (h2 : rows = 8) (h3 : tables_per_row = total_tables / rows) (h4 : tables_left_over = total_tables % rows) :
    tables_per_row = 9 ∧ tables_left_over = 2 := by
  sorry

end table_arrangement_division_l592_592256


namespace bill_donuts_combination_l592_592125

theorem bill_donuts_combination :
  (∃ (kind_to_skip : Fin 5), let remaining_donuts := 8 - 4 in
   let cases := [
     -- Case 1: All remaining donuts are of one kind
     5,

     -- Case 2: Remaining donuts are of two different kinds
     5 * 4,

     -- Case 3: Remaining donuts are all different kinds
     Nat.choose 5 3 * Nat.factorial 3
   ] in
   5 * (cases.sum)) = 425 :=
by
  sorry

end bill_donuts_combination_l592_592125


namespace eugene_pencils_left_l592_592147

-- Define the total number of pencils Eugene initially has
def initial_pencils : ℝ := 234.0

-- Define the number of pencils Eugene gives away
def pencils_given_away : ℝ := 35.0

-- Define the expected number of pencils left
def expected_pencils_left : ℝ := 199.0

-- Prove the number of pencils left after giving away 35.0 equals 199.0
theorem eugene_pencils_left : initial_pencils - pencils_given_away = expected_pencils_left := by
  -- This is where the proof would go, if needed
  sorry

end eugene_pencils_left_l592_592147


namespace quadratic_roots_l592_592591

theorem quadratic_roots (a b C D : ℝ) :
  (a^2 + b^2 = 63) ∧ (a + b = -C) ∧ (a * b = D) →
  C^2 - 2 * D = 63 ∧ D ≤ 31.5 :=
by
  intro h,
  -- Extract conditions from hypothesis
  cases h with h1 h2,
  cases h2 with h2 h3,
  -- Proof omitted
  sorry

end quadratic_roots_l592_592591


namespace intersection_complement_N_l592_592350

open Set

variable (ℝ : Type) [LinearOrderedField ℝ]

def M := {x : ℝ | x^2 - 2*x - 3 < 0}
def N := {x : ℝ | 2*x < 2}
def complement_N := {x : ℝ | x ≥ 1}

theorem intersection_complement_N :
  M ∩ complement_N = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_complement_N_l592_592350


namespace cheryl_material_used_l592_592457

theorem cheryl_material_used :
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  used = (52 / 247 : ℚ) :=
by
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  have : used = (52 / 247 : ℚ) := sorry
  exact this

end cheryl_material_used_l592_592457


namespace inequality_solution_l592_592765

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 3 * x + 10) ≥ 0 ↔ x ≥ -2 := sorry

end inequality_solution_l592_592765


namespace max_mineral_value_l592_592837

/-- Jane discovers three types of minerals with given weights and values:
6-pound mineral chunks worth $16 each,
3-pound mineral chunks worth $9 each,
and 2-pound mineral chunks worth $3 each. 
There are at least 30 of each type available.
She can haul a maximum of 21 pounds in her cart.
Prove that the maximum value, in dollars, that Jane can transport is $63. -/
theorem max_mineral_value : 
  ∃ (value : ℕ), (∀ (x y z : ℕ), 6 * x + 3 * y + 2 * z ≤ 21 → 
    (x ≤ 30 ∧ y ≤ 30 ∧ z ≤ 30) → value ≥ 16 * x + 9 * y + 3 * z) ∧ value = 63 :=
by sorry

end max_mineral_value_l592_592837


namespace proof_sum_of_drawn_kinds_l592_592102

def kindsGrains : Nat := 40
def kindsVegetableOils : Nat := 10
def kindsAnimalFoods : Nat := 30
def kindsFruitsAndVegetables : Nat := 20
def totalKindsFood : Nat := kindsGrains + kindsVegetableOils + kindsAnimalFoods + kindsFruitsAndVegetables
def sampleSize : Nat := 20
def samplingRatio : Nat := sampleSize / totalKindsFood

def numKindsVegetableOilsDrawn : Nat := kindsVegetableOils / 5
def numKindsFruitsAndVegetablesDrawn : Nat := kindsFruitsAndVegetables / 5
def sumVegetableOilsAndFruitsAndVegetablesDrawn : Nat := numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn

theorem proof_sum_of_drawn_kinds : sumVegetableOilsAndFruitsAndVegetablesDrawn = 6 := by
  have h1 : totalKindsFood = 100 := by rfl
  have h2 : samplingRatio = 1 / 5 := by
    calc
      sampleSize / totalKindsFood
      _ = 20 / 100 := rfl
      _ = 1 / 5 := by norm_num
  have h3 : numKindsVegetableOilsDrawn = 2 := by
    calc
      kindsVegetableOils / 5
      _ = 10 / 5 := rfl
      _ = 2 := by norm_num
  have h4 : numKindsFruitsAndVegetablesDrawn = 4 := by
    calc
      kindsFruitsAndVegetables / 5
      _ = 20 / 5 := rfl
      _ = 4 := by norm_num
  calc
    sumVegetableOilsAndFruitsAndVegetablesDrawn
    _ = numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn := rfl
    _ = 2 + 4 := by rw [h3, h4]
    _ = 6 := by norm_num

end proof_sum_of_drawn_kinds_l592_592102


namespace arithmetic_sequence_an_smallest_m_l592_592736

-- Definitions for the sequence {a_n} and its sum S_n
def Sn (n : ℕ) := 3 * n^2 - 2 * n

-- The sequence terms {a_n}
def a_n : ℕ → ℕ
| 0 => 0
| n+1 => Sn (n+1) - Sn n

-- Definitions for the sequence sum T_n
def T_n (n : ℕ) := 1/2 * (1 - 1/(6*n + 1))

-- Conditions
axiom HSn (n : ℕ) : Sn n = 3 * n^2 - 2 * n

-- Prove (1) that {a_n} is an arithmetic sequence with common difference 6
theorem arithmetic_sequence_an : ∀ (n: ℕ), ∃ d : ℕ, ∀ m: ℕ, a_n (m + 1) - a_n m = d := sorry

-- Prove (2) the smallest positive integer m such that T_n < m / 20 for all n ∈ ℕ+
theorem smallest_m : ∀ n : ℕ+, T_n n < (10 : ℕ) / 20 := sorry

end arithmetic_sequence_an_smallest_m_l592_592736


namespace div_by_7_iff_sum_div_by_7_l592_592198

theorem div_by_7_iff_sum_div_by_7 (a b : ℕ) : 
  (101 * a + 10 * b) % 7 = 0 ↔ (a + b) % 7 = 0 := 
by
  sorry

end div_by_7_iff_sum_div_by_7_l592_592198


namespace value_of_F_at_2_l592_592532

-- Define F as a function from ℝ to ℝ
def F (x : ℝ) : ℝ :=
  sqrt(abs(x - 2)) + (10 / real.pi) * atan(sqrt(abs(x - 1)))

-- The main theorem states that F(2) equals 3 given that point (2,3) lies on the graph.
theorem value_of_F_at_2 : F 2 = 3 :=
by
  -- The proof that F(2) = 3 will be filled in here
  sorry

end value_of_F_at_2_l592_592532


namespace distinct_floor_values_count_l592_592162

theorem distinct_floor_values_count : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 500 → 
  (count_distinct (λ n, nat.floor (n^2 / (500 : ℚ))) (finset.range 501)) = 376 :=
by sorry

end distinct_floor_values_count_l592_592162


namespace purely_imaginary_iff_a1_b0_l592_592772

-- Definitions for the conditions
variable (a b : ℝ)

-- Statement to be proved in Lean
theorem purely_imaginary_iff_a1_b0 : (z : ℂ) (h : z = (a : ℂ) * complex.I + b) 
  (ha : a = 1) (hb : b = 0) :
  z.im /= 0 ∧ z.re = 0 ↔ z = (1 : ℂ) * complex.I + 0 := 
by sorry

end purely_imaginary_iff_a1_b0_l592_592772


namespace red_shoes_drawn_l592_592000

-- Define the main conditions
def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4
def probability_red : ℝ := 0.21428571428571427

-- Problem statement in Lean
theorem red_shoes_drawn (x : ℕ) (hx : ↑x / total_shoes = probability_red) : x = 2 := by
  sorry

end red_shoes_drawn_l592_592000


namespace product_first_three_terms_arithmetic_seq_l592_592793

theorem product_first_three_terms_arithmetic_seq :
  ∀ (a₇ d : ℤ), 
  a₇ = 20 → d = 2 → 
  let a₁ := a₇ - 6 * d in
  let a₂ := a₁ + d in
  let a₃ := a₂ + d in
  a₁ * a₂ * a₃ = 960 := 
by
  intros a₇ d a₇_20 d_2
  let a₁ := a₇ - 6 * d
  let a₂ := a₁ + d
  let a₃ := a₂ + d
  sorry

end product_first_three_terms_arithmetic_seq_l592_592793


namespace area_of_triangle_LEF_l592_592274

noncomputable def area_triangle_LEF : ℝ :=
let r : ℝ := 10
let EF_length : ℝ := 12
let LN_length : ℝ := 20
let EF_midpoint_to_P_distance : ℝ := Math.sqrt (r^2 - (EF_length / 2)^2) in
1/2 * EF_length * EF_midpoint_to_P_distance

theorem area_of_triangle_LEF :
  ∃ (L N P M E F : ℝ × ℝ),
    (dist P M = 2 * 10) ∧
    (L.1 < N.1) ∧ (N.1 < P.1) ∧ (P.1 < M.1) ∧
    (E.2 = F.2) ∧
    (dist E F = 12) ∧
    (L.1 < E.1) ∧ (F.1 < M.1) ∧
    (dist L N = 20) ∧
    (dist L P = dist P N) ∧ -- P is the center
    let area := 1/2 * 12 * 8 in
    area = 48 :=
by sorry

end area_of_triangle_LEF_l592_592274


namespace correct_assignment_statement_l592_592836

-- Definitions according to the problem conditions
def input_statement (x : Nat) : Prop := x = 3
def assignment_statement1 (A B : Nat) : Prop := A = B ∧ B = 2
def assignment_statement2 (T : Nat) : Prop := T = T * T
def output_statement (A : Nat) : Prop := A = 4

-- Lean statement for the problem. We need to prove that the assignment_statement2 is correct.
theorem correct_assignment_statement (T : Nat) : assignment_statement2 T :=
by sorry

end correct_assignment_statement_l592_592836


namespace least_number_to_subtract_l592_592444

theorem least_number_to_subtract (n : ℕ) (h : n = 9876543210) : 
  ∃ m, m = 6 ∧ (n - m) % 29 = 0 := 
sorry

end least_number_to_subtract_l592_592444


namespace A_remaining_time_equals_B_remaining_time_l592_592869

variable (d_A d_B remaining_Distance_A remaining_Time_A remaining_Distance_B remaining_Time_B total_Distance : ℝ)

-- Given conditions as definitions
def A_traveled_more : d_A = d_B + 180 := sorry
def total_distance_between_X_Y : total_Distance = 900 := sorry
def sum_distance_traveled : d_A + d_B = total_Distance := sorry
def B_remaining_time : remaining_Time_B = 4.5 := sorry
def B_remaining_distance : remaining_Distance_B = total_Distance - d_B := sorry

-- Prove that: A travels the same remaining distance in the same time as B
theorem A_remaining_time_equals_B_remaining_time :
  remaining_Distance_A = remaining_Distance_B ∧ remaining_Time_A = remaining_Time_B := sorry

end A_remaining_time_equals_B_remaining_time_l592_592869


namespace product_of_invertibles_mod_120_l592_592335

open Nat

theorem product_of_invertibles_mod_120 :
  let m := 120
  let invertibles := { x | x < m ∧ gcd x m = 1 }
  ∏ a in invertibles, a % m = 119 :=
by
  sorry

end product_of_invertibles_mod_120_l592_592335


namespace total_nails_brought_l592_592129

theorem total_nails_brought (nails_per_station : ℕ) (stations_visited : ℕ) (H1 : nails_per_station = 7) (H2 : stations_visited = 20) : nails_per_station * stations_visited = 140 :=
by
  rw [H1, H2]
  norm_num
  exact rfl

end total_nails_brought_l592_592129


namespace probability_both_groups_stop_same_round_l592_592516

noncomputable def probability_same_round : ℚ :=
  let probability_fair_coin_stop (n : ℕ) : ℚ := (1/2)^n
  let probability_biased_coin_stop (n : ℕ) : ℚ := (2/3)^(n-1) * (1/3)
  let probability_fair_coin_group_stop (n : ℕ) : ℚ := (probability_fair_coin_stop n)^3
  let probability_biased_coin_group_stop (n : ℕ) : ℚ := (probability_biased_coin_stop n)^3
  let combined_round_probability (n : ℕ) : ℚ := 
    probability_fair_coin_group_stop n * probability_biased_coin_group_stop n
  let total_probability : ℚ := ∑' n, combined_round_probability n
  total_probability

theorem probability_both_groups_stop_same_round :
  probability_same_round = 1 / 702 := by sorry

end probability_both_groups_stop_same_round_l592_592516


namespace probability_of_one_black_ball_l592_592860

theorem probability_of_one_black_ball (total_balls black_balls white_balls drawn_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_black : black_balls = 2)
  (h_white : white_balls = 2)
  (h_drawn : drawn_balls = 2) :
  ((Nat.choose black_balls 1) * (Nat.choose white_balls 1) : ℚ) / (Nat.choose total_balls drawn_balls) = 2 / 3 :=
by {
  -- Insert proof here
  sorry
}

end probability_of_one_black_ball_l592_592860
