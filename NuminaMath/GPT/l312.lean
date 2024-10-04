import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Complex.Basic
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.InfiniteSum
import Mathlib.Algebra.Ring
import Mathlib.Analysis.Geometry.conicsections
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Angle.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Permutation
import Mathlib.Data.Prime
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Pi
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.Probability
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Series
import Mathlib.Trigonometry.Basic

namespace jane_ate_twelve_pieces_l312_312878

theorem jane_ate_twelve_pieces 
  (total_pieces : ℕ)
  (num_people : ℕ)
  (equal_pieces_per_person : total_pieces % num_people = 0) :
  let pieces_each := total_pieces / num_people in
  total_pieces = 120 → num_people = 10 → pieces_each = 12 :=
by
  intros h_total_pieces h_num_people
  rw [h_total_pieces, h_num_people]
  show total_pieces / num_people = 12
  sorry

end jane_ate_twelve_pieces_l312_312878


namespace heptagon_diagonals_l312_312468

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l312_312468


namespace unique_function_satisfying_equation_l312_312013

theorem unique_function_satisfying_equation :
  ∃! g : ℝ → ℝ, ∀ x y : ℝ, g (x + g y) = x - y :=
by
  use (λ y : ℝ, -y)  -- Guess the function with the correct property
  split
  {
    intros x y
    exact (by linarith)
  }
  {
    intros g hg
    funext y
    have : g (0) = 0 := by linarith
    have g_neg_y := fun y => by rw [←(hg 0 y)]
    rw g_neg_y at *
    simp [this]
  }

end unique_function_satisfying_equation_l312_312013


namespace brother_highlighter_spending_l312_312935

variables {total_money : ℝ} (sharpeners notebooks erasers highlighters : ℝ)

-- Conditions
def total_given := total_money = 100
def sharpeners_cost := 2 * 5
def notebooks_cost := 4 * 5
def erasers_cost := 10 * 4
def heaven_expenditure := sharpeners_cost + notebooks_cost
def brother_remaining := total_money - (heaven_expenditure + erasers_cost)
def brother_spent_on_highlighters := brother_remaining = 30

-- Statement
theorem brother_highlighter_spending (h1 : total_given) 
    (h2 : brother_spent_on_highlighters) : brother_remaining = 30 :=
sorry

end brother_highlighter_spending_l312_312935


namespace Cindy_coins_l312_312330

theorem Cindy_coins (n : ℕ) (h1 : ∃ X Y : ℕ, n = X * Y ∧ Y > 1 ∧ Y < n) (h2 : ∀ Y, Y > 1 ∧ Y < n → ¬Y ∣ n → False) : n = 65536 :=
by
  sorry

end Cindy_coins_l312_312330


namespace days_in_week_l312_312343

theorem days_in_week {F D : ℕ} (h1 : F = 3 + 11) (h2 : F = 2 * D) : D = 7 :=
by
  sorry

end days_in_week_l312_312343


namespace sum_of_roots_eq_three_l312_312386

-- Definitions of the polynomials
def poly1 (x : ℝ) : ℝ := 3 * x^3 + 3 * x^2 - 9 * x + 27
def poly2 (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + 5

-- Theorem stating the sum of the roots of the given equation is 3
theorem sum_of_roots_eq_three : 
  (∀ a b c d e f g h i : ℝ, 
    (poly1 a = 0) → (poly1 b = 0) → (poly1 c = 0) → 
    (poly2 d = 0) → (poly2 e = 0) → (poly2 f = 0) →
    a + b + c + d + e + f = 3) := 
by
  sorry

end sum_of_roots_eq_three_l312_312386


namespace find_m_2n_3k_l312_312067

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_m_2n_3k (m n k : ℕ) (h1 : m + n = 2021) (h2 : is_prime (m - 3 * k)) (h3 : is_prime (n + k)) :
  m + 2 * n + 3 * k = 2025 ∨ m + 2 * n + 3 * k = 4040 := by
  sorry

end find_m_2n_3k_l312_312067


namespace change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l312_312437

-- Given f(x) = x^2 - 5x
def f (x : ℝ) : ℝ := x^2 - 5 * x

-- Prove the change in f(x) when x is increased by 2 is 4x - 6
theorem change_in_f_when_x_increased_by_2 (x : ℝ) : f (x + 2) - f x = 4 * x - 6 := by
  sorry

-- Prove the change in f(x) when x is decreased by 2 is -4x + 14
theorem change_in_f_when_x_decreased_by_2 (x : ℝ) : f (x - 2) - f x = -4 * x + 14 := by
  sorry

end change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l312_312437


namespace find_k_l312_312069

noncomputable def vec_a : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def vec_b : ℝ × ℝ := (0, -1)
noncomputable def vec_c (k : ℝ) : ℝ × ℝ := (k, Real.sqrt 3)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, u = λ • v

theorem find_k (k : ℝ) : collinear (vec_a - 2 • vec_b) (vec_c k) → k = 1 :=
by
  sorry

end find_k_l312_312069


namespace negation1_converse1_negation2_converse2_negation3_converse3_l312_312728

-- Definitions
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_prime (p : ℕ) : Prop := nat.prime p

-- Proof statements
theorem negation1 : ¬ (∀ x y : ℤ, is_odd x → is_odd y → is_even (x + y)) ↔ true := sorry
theorem converse1 : ¬ (∀ x y : ℤ, ¬ (is_odd x ∧ is_odd y) → ¬ is_even (x + y)) ↔ true := sorry

theorem negation2 : (∀ x y : ℤ, x * y = 0 → ¬ (x = 0) ∧ ¬ (y = 0)) ↔ false := sorry
theorem converse2 : (∀ x y : ℤ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ↔ true := sorry

theorem negation3 : (∀ p : ℕ, is_prime p → ¬ (is_odd p)) ↔ false := sorry
theorem converse3 : (∀ p : ℕ, ¬ is_prime p → ¬ (is_odd p)) ↔ false := sorry

end negation1_converse1_negation2_converse2_negation3_converse3_l312_312728


namespace parabola_example_l312_312064

theorem parabola_example (p : ℝ) (hp : p > 0)
    (h_intersect : ∀ x y : ℝ, y = x - p / 2 ∧ y^2 = 2 * p * x → ((x - p / 2)^2 = 2 * p * x))
    (h_AB : ∀ A B : ℝ × ℝ, A.2 = A.1 - p / 2 ∧ B.2 = B.1 - p / 2 ∧ |A.1 - B.1| = 8) :
    p = 2 := 
sorry

end parabola_example_l312_312064


namespace john_worked_period_l312_312133

theorem john_worked_period (A : ℝ) (n : ℕ) (h1 : 6 * A = 1 / 2 * (6 * A + n * A)) : n + 1 = 7 :=
by
  sorry

end john_worked_period_l312_312133


namespace total_bill_second_month_l312_312843

-- Define the fixed monthly charge for internet service and call charges
variables (F C : ℝ)

-- Condition: The total telephone bill for January
def january_bill := F + C = 52

-- Condition: The charge for calls in the second month is twice the charge for January
def second_month_call_charge := 2 * C

-- Define Elvin's total bill for the second month
def second_month_bill := F + second_month_call_charge

-- Proof: The total telephone bill for the second month is $52 + C
theorem total_bill_second_month
  (january_bill : january_bill)
  (second_month_call_charge : second_month_call_charge) :
  second_month_bill = 52 + C :=
sorry

end total_bill_second_month_l312_312843


namespace final_portfolio_correct_l312_312989

noncomputable def final_portfolio : ℝ :=
  let start_amount := 80
  let amount_q1_y1 := start_amount * 1.15
  let amount_q2_y1 := (amount_q1_y1 + 28) * 1.05
  let amount_q3_y1 := (amount_q2_y1 - 10) * 1.06
  let amount_q4_y1 := amount_q3_y1 * 0.97

  let amount_q1_y2 := (amount_q4_y1 + 40) * 1.10
  let amount_q2_y2 := (amount_q1_y2 - 20) * 0.96
  let amount_q3_y2 := (amount_q2_y2 + 12) * 1.02
  let amount_q4_y2 := (amount_q3_y2 - 15) * 0.93

  amount_q4_y2

theorem final_portfolio_correct : final_portfolio ≈ 132.232420960752 := 
  sorry

end final_portfolio_correct_l312_312989


namespace find_a4_l312_312033

-- Declare the conditions
variables {a : ℝ}
variables {a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}
variables (x : ℝ)

-- Condition 1: Polynomial equation
def poly_eq : Prop :=
  (x - a) * (x + 2)^5 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6

-- Condition 2: Sum of coefficients
def sum_coeff : Prop :=
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = -96

-- Statement to prove
theorem find_a4 (h1 : poly_eq x) (h2 : sum_coeff) : a_4 = -10 := 
sorry

end find_a4_l312_312033


namespace percentage_of_local_arts_students_is_50_l312_312967

-- Definitions
def total_students_arts := 400
def total_students_science := 100
def total_students_commerce := 120
def percent_local_science := 25 / 100
def percent_local_commerce := 85 / 100
def total_locals := 327

-- Problem statement in Lean
theorem percentage_of_local_arts_students_is_50
  (x : ℕ) -- Percentage of local arts students as a natural number
  (h1 : percent_local_science * total_students_science = 25)
  (h2 : percent_local_commerce * total_students_commerce = 102)
  (h3 : (x / 100 : ℝ) * total_students_arts + 25 + 102 = total_locals) :
  x = 50 :=
sorry

end percentage_of_local_arts_students_is_50_l312_312967


namespace ST_passes_through_midpoint_arc_ABC_l312_312992

variable {A B C L K S T : Type}
variable [Geometry A B C L K S T]
variable [AngleBisector A B C L]
variable [ChosenPointOnLine K (LineSegment B L)]
variable [ChosenPointOnExtensionLine S (ExtendedLine B L)]
variable [DiametricallyOppositePoint T K (CircumcircleOfTriangle A K C)]

theorem ST_passes_through_midpoint_arc_ABC :
  (Angle AKC - Angle ABC = 90) →
  (Angle ASC = 90) →
  (Circumcircle A K C) →
  PassesThroughMidpointArc ST (ArcMidpoint A B C) :=
by
  sorry

end ST_passes_through_midpoint_arc_ABC_l312_312992


namespace angle_BAF_eq_angle_CAG_l312_312965

-- Assume the necessary points and conditions
variables {A B C D E F G : Type*}
variables [euclidean_geometry A B C D E F G]
variables (P Q : Type*)
          
-- Assume the points on the sides and parallel condition
variables (hD : is_on_segment A B D)
variables (hE : is_on_segment A C E)
variables (hParallel : parallel DE BC)

-- Assume the intersection point
variables (hF : intersection_point BE CD F)

-- Assume the existence of circumcircles intersecting at points F and G
variables (hCircBDF : circumcircle B D F)
variables (hCircCEF : circumcircle C E F)
variables (hIntersection : intersects_at hCircBDF hCircCEF F G)

-- The theorem to be proved
theorem angle_BAF_eq_angle_CAG :
  ∠ BAF = ∠ CAG :=
sorry

end angle_BAF_eq_angle_CAG_l312_312965


namespace T_n_lt_7_over_2_l312_312406

noncomputable def a : ℕ → ℝ
| 0       => 0  -- no a_0 given in the original problem (we start from a_1)
| 1       => 3
| (n+1+1) => 3 * a (n+1)

def S : ℕ → ℝ
| 0       => 0
| (n+1) => (List.range (n+1)).sum (a ∘ (Function.iterate Nat.succ))

def b (n : ℕ) : ℝ := (4 * n + 1) / a n

def T (n : ℕ) : ℝ := (List.range n).sum (b ∘ (Function.iterate Nat.succ))

theorem T_n_lt_7_over_2 (n : ℕ) (hn : n > 0) : T n < (7 / 2) :=
begin
  sorry
end

end T_n_lt_7_over_2_l312_312406


namespace probability_factory_two_given_defective_l312_312753

variables (H1 H2 A : Prop)
variables (P : Prop → ℝ)

-- Given conditions
axiom P_H1 : P(H1) = 1/4
axiom P_H2 : P(H2) = 3/4
axiom P_A_given_H1 : P(A ∣ H1) = 0.02
axiom P_A_given_H2 : P(A ∣ H2) = 0.01

-- Definition of total probability theorem
noncomputable def P_A : ℝ := 
  P(A ∣ H1) * P(H1) + P(A ∣ H2) * P(H2)

-- Given conclusion to prove
theorem probability_factory_two_given_defective :
  P(H2 ∣ A) = 0.6 :=
by
  -- Using Bayes' theorem and provided conditions within the proof
  sorry

end probability_factory_two_given_defective_l312_312753


namespace union_I_n_is_all_odd_integers_l312_312021

def I_0 : Set ℤ := {-1, 1}

def I_n (n : ℕ) : Set ℤ :=
  {x | ∃ y ∈ I_n (n-1), x^2 - 2*x*y + y^2 = 4^n}

noncomputable def union_I_n : Set ℤ :=
  {z | ∃ (n : ℕ) (x ∈ I_n n), z = x}

theorem union_I_n_is_all_odd_integers : union_I_n = {z : ℤ | z % 2 = 1 ∨ z % 2 = -1} :=
by
  sorry

end union_I_n_is_all_odd_integers_l312_312021


namespace first_negative_term_position_l312_312982

def a1 : ℤ := 1031
def d : ℤ := -3
def nth_term (n : ℕ) : ℤ := a1 + (n - 1 : ℤ) * d

theorem first_negative_term_position : ∃ n : ℕ, nth_term n < 0 ∧ n = 345 := 
by 
  -- Placeholder for proof
  sorry

end first_negative_term_position_l312_312982


namespace point_in_second_quadrant_l312_312562

/-- Define the quadrants in the Cartesian coordinate system -/
def quadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

theorem point_in_second_quadrant :
  quadrant (-3) 2005 = "Second quadrant" :=
by
  sorry

end point_in_second_quadrant_l312_312562


namespace no_prime_divisible_by_56_l312_312529

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end no_prime_divisible_by_56_l312_312529


namespace equal_sets_power_sums_l312_312415

variable {x y : ℝ}
def A : Set ℝ := {x, y / x, 1}
def B : Set ℝ := {x^2, x + y, 0}

theorem equal_sets_power_sums (h : A = B) : x^2023 + y^2024 = -1 :=
  sorry

end equal_sets_power_sums_l312_312415


namespace roof_collapse_days_l312_312327

def leaves_per_pound : ℕ := 1000
def pounds_limit_of_roof : ℕ := 500
def leaves_per_day : ℕ := 100

theorem roof_collapse_days : (pounds_limit_of_roof * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

end roof_collapse_days_l312_312327


namespace cost_of_toilet_paper_roll_l312_312802

-- Definitions of the problem's conditions
def num_toilet_paper_rolls : Nat := 10
def num_paper_towel_rolls : Nat := 7
def num_tissue_boxes : Nat := 3

def cost_per_paper_towel : Real := 2
def cost_per_tissue_box : Real := 2

def total_cost : Real := 35

-- The function to prove
def cost_per_toilet_paper_roll (x : Real) :=
  num_toilet_paper_rolls * x + 
  num_paper_towel_rolls * cost_per_paper_towel + 
  num_tissue_boxes * cost_per_tissue_box = total_cost

-- Statement to prove
theorem cost_of_toilet_paper_roll : 
  cost_per_toilet_paper_roll 1.5 := 
by
  simp [num_toilet_paper_rolls, num_paper_towel_rolls, num_tissue_boxes, cost_per_paper_towel, cost_per_tissue_box, total_cost]
  sorry

end cost_of_toilet_paper_roll_l312_312802


namespace tobias_weeks_played_l312_312623

theorem tobias_weeks_played :
  (let nathan_hours_per_day := 3
     nathan_weeks := 2
     tobias_hours_per_day := 5
     total_hours_played := 77
     nathan_total_hours := nathan_hours_per_day * 7 * nathan_weeks
     tobias_total_hours := total_hours_played - nathan_total_hours
     tobias_weeks := tobias_total_hours / (tobias_hours_per_day * 7)
   in tobias_weeks = 1) :=
by
  sorry

end tobias_weeks_played_l312_312623


namespace tg_x_plus_tg_y_l312_312334

theorem tg_x_plus_tg_y (x y a b : ℝ) (h1 : sin (2 * x) + sin (2 * y) = a) (h2 : cos (2 * x) + cos (2 * y) = b) :
  tan x + tan y = 4 * a / (a ^ 2 + b ^ 2 + 2 * b) :=
sorry

end tg_x_plus_tg_y_l312_312334


namespace quotient_of_sum_of_distinct_remainders_l312_312341

def cube (n: ℤ) : ℤ := n ^ 3

noncomputable def distinct_remainders_sum : ℤ :=
  (Finset.range 16).val.filter (λ n, ∃ k in (Finset.range 16), n ≡ k^3 [MOD 16])
  |>.sum

theorem quotient_of_sum_of_distinct_remainders : 
  let m := distinct_remainders_sum in 
  m / 16 = 2 :=
by
  sorry

end quotient_of_sum_of_distinct_remainders_l312_312341


namespace base_conversion_l312_312671

theorem base_conversion (b : ℕ) (h : 1 * 6^2 + 4 * 6 + 2 = 2 * b^2 + b + 5) : b = 5 :=
by
  sorry

end base_conversion_l312_312671


namespace fraction_output_N_l312_312154

theorem fraction_output_N (X t_N t_T t_O : ℝ) 
  (hT_N : t_T = (3 / 4) * t_N)
  (hN_O : t_O = (3 / 2) * t_N) :
  let R_T := X / t_T;
      R_N := X / t_N;
      R_O := X / t_O;
      R_Total := R_T + R_N + R_O in
  (R_N / R_Total) = 1 / 3 :=
by
  sorry

end fraction_output_N_l312_312154


namespace ceil_neg_3_7_l312_312365

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l312_312365


namespace y_intercept_of_line_l312_312566

theorem y_intercept_of_line (a : ℝ) (k : ℝ → ℝ) 
  (h_slope : ∀ x, k(x) = 4 * a * x + (-4 * a^2)) :
  k(0) = -4 * a^2 :=
by
  sorry

end y_intercept_of_line_l312_312566


namespace largest_n_for_factoring_l312_312869

theorem largest_n_for_factoring :
  ∃ n, (∃ A B : ℤ, (3 * A = 3 * 108 + 1) ∧ (/3 * B * 108 = 2) ∧ 
  (3 * 36 + 3 = 111) ∧ (3 * 108 + A = n) )=
  (n = 325) := sorry
iddenLean_formatter.clonecreateAngular

end largest_n_for_factoring_l312_312869


namespace mom_went_by_car_l312_312618

def transportation_mode (speed : ℕ) : Prop :=
  speed = 70

theorem mom_went_by_car (speed : ℕ) (h : transportation_mode speed) : Prop :=
  speed = 70 ∧ "Mom went by car" :=
by
  sorry

end mom_went_by_car_l312_312618


namespace cement_needed_correct_l312_312808

noncomputable def cement_needed_for_tunnel 
  (r : ℝ) -- radius of circular cross-section in meters
  (h : ℝ) -- thickness of concrete layer in meters
  (length : ℝ) -- length of the tunnel in meters
  (cement_density : ℝ) -- cement density in kg/m³
  : ℝ := 
  let alpha_half := real.arccos ((r - h) / r) in
  let alpha := 2 * alpha_half in
  let sector_area := (r^2 * real.pi * alpha / (2 * real.pi)) in
  let triangle_base := real.sqrt (r^2 - (r - h)^2) in
  let triangle_area := 0.5 * (r - h) * triangle_base in
  let segment_area := sector_area - triangle_area in
  let volume := segment_area * length in
  let total_cement := (volume * cement_density) in
  total_cement / 100 

theorem cement_needed_correct 
  : cement_needed_for_tunnel 2.3 0.7 1000 150 = 50 := sorry

end cement_needed_correct_l312_312808


namespace numbers_below_9_and_24_not_prime_l312_312240

/-
  Given a spiral arrangement of natural numbers on graph paper,
  prove that the numbers directly below 9 and 24 in their respective columns are composite.
-/

theorem numbers_below_9_and_24_not_prime 
  (spiral : ℕ → ℕ × ℕ) 
  (is_prime : ℕ → Prop)
  (h_spiral_layout : ∀ n, spiral n = ⟨⟨(∃ k ≥ 1, spiral (2*k+1)^2 + k = n) ∨ (∃ k ≥ 2, spiral (2*k+1)^2 + k - 1 = n)⟩, is_prime (2*k+1)^2 + k→n(k2,k5,k6= composite number)⟩ )
  : ∀ k ≥ 1, ¬ is_prime (4*k^2 + 5*k + 1) ∧ ∀ k ≥ 2, ¬ is_prime (4*k^2 + 5*k).

sorry

end numbers_below_9_and_24_not_prime_l312_312240


namespace dot_product_BA_BC_find_b_l312_312572

-- Define the conditions using parameters and hypothesis

variables {A B C : Type}
variables (a b c : ℝ) (cosB : ℝ) (area : ℝ)
hypothesis (h_cosB : cosB = 4 / 5)
hypothesis (h_area : area = 24)
hypothesis (h_c_relation : c = 5 / 4 * a)

-- Proof problem 1: Prove the dot product value of vector BA and BC
theorem dot_product_BA_BC : ∃ (a c : ℝ), 2 * area = ac * sqrt(1 - cosB^2) ∧ 
   (cosB = 4 / 5 ∧ area = 24) → 
   (ac * cosB = 64) :=
by sorry

-- Proof problem 2: Prove that 'b' satisfies the given conditions
theorem find_b : c = 5 / 4 * a → (cosB = 4 / 5 ∧ area = 24) → 
   b = sqrt(a^2 + (5 / 4 * a)^2 - 2 * a * (5 / 4 * a) * (4 / 5)) → 
   b = 6 :=
by sorry

end dot_product_BA_BC_find_b_l312_312572


namespace common_ratio_geometric_sequence_l312_312430

theorem common_ratio_geometric_sequence
  (a_1 : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (geom_sum : ∀ n q, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q))
  (h_arithmetic : 2 * S 4 = S 5 + S 6)
  : (∃ q : ℝ, ∀ n : ℕ, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q)) → q = -2 :=
by
  sorry

end common_ratio_geometric_sequence_l312_312430


namespace max_discount_l312_312297

-- Definitions:
def cost_price : ℝ := 400
def sale_price : ℝ := 600
def desired_profit_margin : ℝ := 0.05

-- Statement:
theorem max_discount 
  (x : ℝ) 
  (hx : sale_price * (1 - x / 100) ≥ cost_price * (1 + desired_profit_margin)) :
  x ≤ 90 := 
sorry

end max_discount_l312_312297


namespace ceil_neg_3_7_l312_312356

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l312_312356


namespace contrapositive_proposition_contrapositive_version_l312_312398

variable {a b : ℝ}

theorem contrapositive_proposition (h : a + b = 1) : a^2 + b^2 ≥ 1/2 :=
sorry

theorem contrapositive_version : a^2 + b^2 < 1/2 → a + b ≠ 1 :=
by
  intros h
  intro hab
  apply not_le.mpr h
  exact contrapositive_proposition hab

end contrapositive_proposition_contrapositive_version_l312_312398


namespace smallest_perimeter_even_integer_triangl_l312_312254

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l312_312254


namespace correct_option_is_C_l312_312269

namespace ExponentProof

-- Definitions of conditions
def optionA (a : ℝ) : Prop := a^3 * a^4 = a^12
def optionB (a : ℝ) : Prop := a^3 + a^4 = a^7
def optionC (a : ℝ) : Prop := a^5 / a^3 = a^2
def optionD (a : ℝ) : Prop := (-2 * a)^3 = -6 * a^3

-- Proof problem stating that optionC is the only correct one
theorem correct_option_is_C : ∀ (a : ℝ), ¬ optionA a ∧ ¬ optionB a ∧ optionC a ∧ ¬ optionD a :=
by
  intro a
  sorry

end ExponentProof

end correct_option_is_C_l312_312269


namespace value_of_x_l312_312083

theorem value_of_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 := 
by
  sorry

end value_of_x_l312_312083


namespace employees_use_public_transportation_l312_312758

theorem employees_use_public_transportation
    (total_employees : ℕ)
    (drive_percentage : ℝ)
    (public_transportation_fraction : ℝ)
    (h1 : total_employees = 100)
    (h2 : drive_percentage = 0.60)
    (h3 : public_transportation_fraction = 0.50) :
    ((total_employees * (1 - drive_percentage)) * public_transportation_fraction) = 20 :=
by
    sorry

end employees_use_public_transportation_l312_312758


namespace number_of_arrangements_l312_312323

theorem number_of_arrangements (A B C D : Type) (events : Finset (Fin 3)) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_nonempty : ∀ e ∈ events, ∃ x ∈ ({A, B, C, D} : Finset Type), True)
  (h_single_event : ∀ e ∈ events, ∃! x ∈ ({A, B, C, D} : Finset Type), ∀ y ∈ ({A, B, C, D} : Finset Type), x = y → False)
  (h_ab_diff : ∀ e ∈ events, (A, B) ∉ ({(A, B)} : Finset (Type × Type)))
  : num_arrangements events 4 3 30 :=
sorry

end number_of_arrangements_l312_312323


namespace vec_op_l312_312023

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -2)
def two_a : ℝ × ℝ := (2 * 2, 2 * 1)
def result : ℝ × ℝ := (two_a.1 - b.1, two_a.2 - b.2)

theorem vec_op : (2 * a.1 - b.1, 2 * a.2 - b.2) = (2, 4) := by
  sorry

end vec_op_l312_312023


namespace largest_consecutive_odd_integers_sum_255_l312_312218

theorem largest_consecutive_odd_integers_sum_255 : 
  ∃ (n : ℤ), (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 255) ∧ (n + 8 = 55) :=
by
  sorry

end largest_consecutive_odd_integers_sum_255_l312_312218


namespace solve_quadratic_eq_l312_312397

theorem solve_quadratic_eq (b c : ℝ) :
  (∀ x : ℝ, |x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = -1) →
  b = -6 ∧ c = -7 :=
by
  intros h_abs_val_eq h_quad_eq
  sorry

end solve_quadratic_eq_l312_312397


namespace problem_statement_l312_312911

noncomputable def sin_theta (a : ℝ) (h : a > 0) : Prop :=
  let r := real.sqrt ((3 * a) ^ 2 + (4 * a) ^ 2)
  r = 5 * a ∧ real.sin (real.atan2 (4 * a) (3 * a)) = 4 / 5

noncomputable def sin_cos_expression (a : ℝ) (h : a > 0) : Prop :=
  let θ := real.atan2 (4 * a) (3 * a)
  real.sin (3 * real.pi / 2 - θ) + real.cos (θ - real.pi) = -6 / 5

theorem problem_statement (a : ℝ) (h : a > 0) : 
  sin_theta a h ∧ sin_cos_expression a h :=
  sorry

end problem_statement_l312_312911


namespace min_value_binom_l312_312959

theorem min_value_binom
  (a b : ℕ → ℕ)
  (n : ℕ) (hn : 0 < n)
  (h1 : ∀ n, a n = 2^n)
  (h2 : ∀ n, b n = 4^n) :
  ∃ n, 2^n + (1 / 2^n) = 5 / 2 :=
sorry

end min_value_binom_l312_312959


namespace average_postcards_per_day_l312_312844

theorem average_postcards_per_day : 
  ∀ (a d n : ℕ), 
  a = 10 → 
  d = 10 →
  n = 7 →
  (∑ i in range n, a + i * d) / n = 40 :=
by
  assume a d n
  intro h_a h_d h_n
  sorry

end average_postcards_per_day_l312_312844


namespace num_test_takers_l312_312186

def each_person took_5_tests := true
def test_score_ranges := [17, 28, 35, 45]
def min_possible_range := 45

theorem num_test_takers (n : ℕ) :
  each_person_took_5_tests →
  test_score_ranges =
    [17, 28, 35, 45] →
  min_possible_range = 45 →
  n = 2 :=
by
  intros
  exact sorry

end num_test_takers_l312_312186


namespace trapezoid_base_ratio_l312_312651

-- Define variables and conditions as per the problem
variables {a b h: ℝ}
def is_trapezoid (a b: ℝ) := a > b
def area_trapezoid (a b h: ℝ) := (1 / 2) * (a + b) * h
def area_quadrilateral (a b h: ℝ) := (1 / 2) * ((a - b) / 2) * (h / 2)

-- State the problem statement
theorem trapezoid_base_ratio (a b h: ℝ) (ha: a > b) (ht: area_quadrilateral a b h = (1 / 4) * area_trapezoid a b h) :
  a / b = 3 :=
sorry

end trapezoid_base_ratio_l312_312651


namespace circles_radius_proof_l312_312704

theorem circles_radius_proof :
  (∃ s : ℝ, 
    (∀ x y : ℝ, (x - s)^2 + y^2 = s^2 → x^2 + 4y^2 = 8)) →
  s = Real.sqrt (3 / 2) :=
by
  sorry

end circles_radius_proof_l312_312704


namespace triangle_angles_correct_l312_312692

noncomputable def triangle_angles (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
by sorry

theorem triangle_angles_correct :
  triangle_angles 3 (Real.sqrt 8) (2 + Real.sqrt 2) =
    (67.5, 22.5, 90) :=
by sorry

end triangle_angles_correct_l312_312692


namespace megatek_manufacturing_percentage_l312_312642

theorem megatek_manufacturing_percentage (total_degrees sector_degrees : ℝ)
    (h_circle: total_degrees = 360)
    (h_sector: sector_degrees = 252) :
    (sector_degrees / total_degrees) * 100 = 70 :=
by
  sorry

end megatek_manufacturing_percentage_l312_312642


namespace problem1_l312_312290

theorem problem1 
  (α β : Real)
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - Real.pi / 4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := sorry

end problem1_l312_312290


namespace false_converse_of_proposition_B_l312_312726

-- Proposition A
def parallel_lines_imply_equal_corresponding_angles (L1 L2 : Line) (h : parallel L1 L2) : 
    ∀ (T : Transversal), corresponding_angles_equal L1 L2 T := sorry

-- Proposition B
def square_implies_right_angles (sq : Square) : all_four_right_angles sq := sorry

def quadrilateral_four_right_angles (quad : Quadrilateral) (h : all_four_right_angles quad) : Square quad := sorry

-- Proposition C
def rhombus_implies_equal_sides (rhom : Rhombus) : all_sides_equal rhom := sorry

def quadrilateral_equal_sides (quad : Quadrilateral) (h : all_sides_equal quad) : Rhombus quad := sorry

-- Proposition D
def parallelogram_implies_diagonals_bisect (pgram : Parallelogram) : diagonals_bisect pgram := sorry

def quadrilateral_diagonals_bisect (quad : Quadrilateral) (h : diagonals_bisect quad) : Parallelogram quad := sorry

theorem false_converse_of_proposition_B :
  ∃ (quad : Quadrilateral), all_four_right_angles quad ∧ ¬Square quad := sorry

end false_converse_of_proposition_B_l312_312726


namespace number_of_students_l312_312580

-- Define the conditions
variable (n : ℕ) (jayden_rank_best jayden_rank_worst : ℕ)
variable (h1 : jayden_rank_best = 100)
variable (h2 : jayden_rank_worst = 100)

-- Define the question
theorem number_of_students (h1 : jayden_rank_best = 100) (h2 : jayden_rank_worst = 100) : n = 199 := 
  sorry

end number_of_students_l312_312580


namespace swap_adjacent_swap_vertices_l312_312099

-- Definitions for initial conditions of the problem
inductive Vertex (n : Nat) : Type
| mk (i : Fin n) : Vertex

def initial_pawn_position (n : Nat) (v : Vertex n) : Fin n := sorry -- Initial pawn position function (to be defined)

-- Definition of operations
def move_clockwise (n : Nat) (pawns : Fin n -> Fin n) : Fin n -> Fin n := 
  λ v, match v with
       | Vertex.mk (Fin.mk i hi) => Vertex.mk ((i + 1) % n)

def swap_A1_A2 (n : Nat) (pawns : Fin n -> Fin n) : Fin n -> Fin n :=
  λ v, match v with
       | Vertex.mk 0 => Vertex.mk 1
       | Vertex.mk 1 => Vertex.mk 0
       | Vertex.mk i => Vertex.mk i

-- Theorem statements
theorem swap_adjacent (n : Nat) (pawns : Fin n -> Fin n) : 
  ∀ i : Fin (n-1), ∃ (operations : List (Fin n -> Fin n)), 
  (List.foldr (λ f acc, f ∘ acc) id operations) pawns ((i + 1) % n) = pawns i :=
sorry

theorem swap_vertices (n : Nat) (pawns : Fin n -> Fin n) :
  ∀ i j : Fin n, i.val < j.val -> ∃ (operations : List (Fin n -> Fin n)),
  (List.foldr (λ f acc, f ∘ acc) id operations) pawns j = pawns i :=
sorry

end swap_adjacent_swap_vertices_l312_312099


namespace running_time_15mph_l312_312002

theorem running_time_15mph (x y z : ℝ) (h1 : x + y + z = 14) (h2 : 15 * x + 10 * y + 8 * z = 164) :
  x = 3 :=
sorry

end running_time_15mph_l312_312002


namespace num_divisors_36_l312_312498

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312498


namespace sum_of_possible_remainders_l312_312149

open Nat

theorem sum_of_possible_remainders : (∑ n in {0, 1, 2, 3, 4, 5, 6, 7, 8}, (n + 10) % 11) = 126 :=
by
  sorry

end sum_of_possible_remainders_l312_312149


namespace smoking_negative_correlation_l312_312742

theorem smoking_negative_correlation (H : "Smoking is harmful to health") : 
  "smoking has a negative correlation with health" :=
sorry

end smoking_negative_correlation_l312_312742


namespace ceiling_of_neg_3_7_l312_312373

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l312_312373


namespace maximum_sinA_plus_sinB_l312_312123

theorem maximum_sinA_plus_sinB {A B C a b c : ℝ} (h1 : 0 < A) (h2 : A < 2 * π / 3) 
  (h3 : 0 < B) (h4 : B < 2 * π / 3) (h5 : A + B = 2 * π / 3) : 
  (c * sin A = sqrt 3 * a * cos C) → 
  ∃ M, M = sqrt 3 ∧ ∀ x y, (0 < x ∧ x < 2 * π / 3 ∧ 0 < y ∧ y < 2 * π / 3 ∧ x + y = 2 * π / 3) → sin x + sin y ≤ M :=
sorry

end maximum_sinA_plus_sinB_l312_312123


namespace pascal_triangle_even_odd_count_l312_312944

theorem pascal_triangle_even_odd_count :
  let C (n k : ℕ) := factorial n / (factorial k * factorial (n - k))
  let is_even (n k : ℕ) := ∃b1 b2, k = b1 + b2 ∧ (nat.shiftl b1 (nat.log2 n - 1)) + nat.shiftl b2 1 = n
  let evens := finset.sum (finset.range 15) (λ n, finset.filter (λ k, is_even n k) (finset.range (n + 1))).card
  let odds := finset.sum (finset.range 15) (λ n, (n + 1) - (finset.filter (λ k, is_even n k) (finset.range (n + 1))).card)
  evens = 56 ∧ odds = 64 :=
by
  sorry

end pascal_triangle_even_odd_count_l312_312944


namespace average_inside_time_l312_312135

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end average_inside_time_l312_312135


namespace vectors_parallel_same_direction_l312_312412

variable {E : Type} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable (a b : E)

theorem vectors_parallel_same_direction
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : ∥a + b∥ = ∥a∥ + ∥b∥) :
  (∃ k : ℝ, k > 0 ∧ a = k • b) ∨ (∃ k : ℝ, k > 0 ∧ b = k • a) := 
by
  sorry

end vectors_parallel_same_direction_l312_312412


namespace trapezoid_base_ratio_l312_312667

variable {A B C D K M P Q L N : Type}
variable [LinearOrderedField A]
variable [LinearOrderedField B]
variable [LinearOrderedField C]
variable [LinearOrderedField D]

def is_trapezoid (A B C D : Type) (a b : ℝ) : Prop :=
  is_parallel (AD BC) ∧ AD.length = a ∧ BC.length = b ∧ a > b

def area (A B C D : Type) (a b : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

def quadrilateral_area (K L M N : Type) (a b h : ℝ) : ℝ :=
  (1 / 4) * (a - b) * h

theorem trapezoid_base_ratio (A B C D K M P Q L N : Type) (a b : ℝ) (h : ℝ) :
  is_trapezoid A B C D a b →
  quadrilateral_area K L M N a b h = (1 / 4) * area A B C D a b ↔ a / b = 3 :=
by 
  sorry

end trapezoid_base_ratio_l312_312667


namespace find_phi_l312_312919

open Real

theorem find_phi (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π) 
    (h3 : symmetric_center (fun x => sin (-2 * x + ϕ)) (π / 3, 0)) : 
    ϕ = 2 * π / 3 := 
sorry

end find_phi_l312_312919


namespace maximum_machine_speed_l312_312772

theorem maximum_machine_speed : 
  let xs := [8, 12, 14, 16]
  let ys := [5, 8, 9, 11]
  let n := xs.length
  let x_mean := (xs.sum) / n
  let y_mean := (ys.sum) / n
  let b := (∑ i in finset.range n, (xs.nth! i - x_mean) * (ys.nth! i - y_mean)) /
           (∑ i in finset.range n, (xs.nth! i - x_mean) ^ 2)
  let a := y_mean - b * x_mean
  b ≠ 0 →
  (ys.nth! 0 - (a + b * xs.nth! 0) = 0) ∧
  (ys.nth! 1 - (a + b * xs.nth! 1) = 0) ∧
  (ys.nth! 2 - (a + b * xs.nth! 2) = 0) ∧
  (ys.nth! 3 - (a + b * xs.nth! 3) = 0) →
  let x := (10 + a) / b in
  x ≤ 15 :=
by
  sorry

end maximum_machine_speed_l312_312772


namespace tolya_vs_vasya_l312_312698

open Real

-- Definitions based on the conditions
variables {n : ℕ} (a : Fin n → ℝ)
hypothesis h1 : n = 1000
hypothesis petya_condition : ∀ i : Fin n, abs (a i - a ((i + 1) % n)) ≥ 2 * v
hypothesis vasya_condition : ∀ i : Fin n, abs (a i - a ((i + 2) % n)) ≤ v

-- Statement to be proved
theorem tolya_vs_vasya (v : ℝ) (t : ℝ) : (∃ i : Fin n, abs (a i - a ((i + 3) % n)) = t) → t ≥ v :=
by
  intros h
  have h_t : ∀ i : Fin n, abs (a i - a ((i + 3) % n)) = t,
  { obtain ⟨i, hi⟩ := h, exact hi }
  -- Proof goes here
  sorry

end tolya_vs_vasya_l312_312698


namespace soccer_match_players_l312_312554

theorem soccer_match_players :
  ∀ (red_socks blue_socks green_socks : ℕ),
  red_socks = 12 → blue_socks = 10 → green_socks = 16 →
  (∃ players, players = 12 ∧ 
    (red_socks ≤ players ∧
     blue_socks + green_socks ≥ players ∧ 
     red_socks - min red_socks blue_socks <= green_socks)) :=
by
  intros red_socks blue_socks green_socks hred hblue hgreen
  use 12
  split
  assumption
  split
  -- goal is red_socks ≤ 12
  linarith
  split
  -- goal is blue_socks + green_socks ≥ 12
  linarith
  -- goal is red_socks - min red_socks blue_socks <= green_socks
  linarith [min_le_left red_socks blue_socks]

end soccer_match_players_l312_312554


namespace fractional_decimal_expansion_412th_digit_l312_312711

theorem fractional_decimal_expansion_412th_digit :
  let seq := "38297872340425531914893617"
  let cycle_length := 46
  let position := 412 % cycle_length
  seq.get (position - 1) == '3' :=
by
  let seq := "38297872340425531914893617"
  let cycle_length := 46
  let position := 412 % cycle_length
  have hcycle : position = 44 := by sorry
  have hdigit : seq.get (position - 1) = '3' := by sorry
  exact hdigit

end fractional_decimal_expansion_412th_digit_l312_312711


namespace perpendicular_vectors_l312_312457

variables (t k : ℝ)
def veca : ℝ × ℝ := ( √3 / 2, -1 / 2 )
def vecb : ℝ × ℝ := ( 1, √3 )
def vecx (t : ℝ) : ℝ × ℝ := (veca.1 + (t^2 - 3) * vecb.1, veca.2 + (t^2 - 3) * vecb.2)
def vecy (t k : ℝ) : ℝ × ℝ := (-k * veca.1 + t * vecb.1, -k * veca.2 + t * vecb.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem perpendicular_vectors (t : ℝ) : 
  dot_product (vecx t) (vecy t (4 * t * (t^2 - 3))) = 0 := sorry

end perpendicular_vectors_l312_312457


namespace triangle_is_right_triangle_l312_312636

theorem triangle_is_right_triangle (A B C : ℝ) (h : sin A ^ 2 = sin B ^ 2 + sin C ^ 2) : 
  A = π / 2 := 
sorry

end triangle_is_right_triangle_l312_312636


namespace exists_ellipse_l312_312009

/-- Definition of a circle with center O and radius R -/
structure Circle (O : ℝ × ℝ) (R : ℝ) :=
mk :: (center : ℝ × ℝ) (radius : ℝ)

/-- Definition of an ellipse with center O and semi-major and semi-minor axes -/
structure Ellipse (O : ℝ × ℝ) (a b : ℝ) :=
mk :: (center : ℝ × ℝ) (semi_major_axis : ℝ) (semi_minor_axis : ℝ)

/-- Definition of a tangent between a circle and an ellipse -/
def is_tangent (C : Circle) (E : Ellipse) (e : ℝ) : Prop :=
∃ P : ℝ × ℝ, (distance P C.center = C.radius) ∧ (distance P E.center = sqrt ((P.1 - E.center.1) ^ 2 / E.semi_major_axis^2 + (P.2 - E.center.2) ^ 2 / E.semi_minor_axis^2))

/-- Definition of intersection points -/
def intersect (C : Circle) (E : Ellipse) (A B : ℝ × ℝ) : Prop :=
(distance A C.center = C.radius) ∧ (distance B C.center = C.radius) ∧ (distance A E.center = sqrt ((A.1 - E.center.1) ^ 2 / E.semi_major_axis^2 + (A.2 - E.center.2) ^ 2 / E.semi_minor_axis^2)) ∧
(distance B E.center = sqrt ((B.1 - E.center.1) ^ 2 / E.semi_major_axis^2 + (B.2 - E.center.2) ^ 2 / E.semi_minor_axis^2))

/-- Main theorem statement asserting the existence of the desired ellipse -/
theorem exists_ellipse (O : ℝ × ℝ) (R : ℝ) (e : ℝ) (A B : ℝ × ℝ) (h_collinear : (A.1 - O.1) * (B.2 - O.2) ≠ (A.2 - O.2) * (B.1 - O.1))
  (C : Circle O R) :
  ∃ (E : Ellipse O 1 1), is_tangent C E e ∧ intersect C E A B :=
sorry

end exists_ellipse_l312_312009


namespace trapezoid_ratio_of_bases_l312_312658

theorem trapezoid_ratio_of_bases (a b : ℝ) (h : a > b)
    (H : (1 / 4) * (1 / 2) * (a - b) * h = (1 / 8) * (a + b) * h) : 
    a / b = 3 := 
sorry

end trapezoid_ratio_of_bases_l312_312658


namespace cube_root_sixteenth_power_eqn_l312_312004

theorem cube_root_sixteenth_power_eqn : (16^(1/3 : ℝ))^6 = 256 := 
by sorry

end cube_root_sixteenth_power_eqn_l312_312004


namespace bus_speed_without_stoppages_l312_312846

theorem bus_speed_without_stoppages (v : ℝ) (stoppage_time_per_hour : ℝ) (bus_avg_speed_with_stoppage : ℝ) :
  (stoppage_time_per_hour = 10 / 60) →
  (bus_avg_speed_with_stoppage = 45) →
  (v * (1 - stoppage_time_per_hour) = bus_avg_speed_with_stoppage * (1 - stoppage_time_per_hour)) →
  v = 54 :=
by {
  intros h_stoppage h_avg_speed h_eqn,
  sorry
}

end bus_speed_without_stoppages_l312_312846


namespace range_of_p_l312_312925

theorem range_of_p (p : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + p * x + 1 > 2 * x + p) → p > -1 := 
by
  sorry

end range_of_p_l312_312925


namespace find_x_intercept_l312_312063

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 12

noncomputable def line_eq (m x y : ℝ) : Prop := m * x + y + 3 * m - real.sqrt 3 = 0

noncomputable def distance (A B : ℝ) : Prop := |A - B| = 2 * real.sqrt 3

theorem find_x_intercept (m : ℝ) (x : ℝ) (hx : ∀ y, line_eq m x y) 
  (hc : ∀ x y, circle_eq x y) 
  (hd : distance 0 m) :
  x = -6 :=
by
  sorry

end find_x_intercept_l312_312063


namespace both_functions_monotonically_increasing_l312_312450

noncomputable def func1 (x : ℝ) : ℝ :=
  sin x + cos x

noncomputable def func2 (x : ℝ) : ℝ :=
  2 * sqrt 2 * sin x * cos x

theorem both_functions_monotonically_increasing :
  ∀ x y : ℝ, x ∈ Ioc (-π / 4) (π / 4) ∧ y ∈ Ioc (-π / 4) (π / 4) ∧ x < y →
    func1 x < func1 y ∧ func2 x < func2 y :=
sorry

end both_functions_monotonically_increasing_l312_312450


namespace circle_outside_hexagon_area_l312_312299

theorem circle_outside_hexagon_area :
  let r := (Real.sqrt 2) / 2
  let s := 1
  let area_circle := π * r^2
  let area_hexagon := 3 * Real.sqrt 3 / 2 * s^2
  area_circle - area_hexagon = (π / 2) - (3 * Real.sqrt 3 / 2) :=
by
  sorry

end circle_outside_hexagon_area_l312_312299


namespace relationship_among_abc_l312_312026

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l312_312026


namespace first_friend_added_15_l312_312585

def poundsAddedByFirstFriend : ℕ := 
  let jovanas_shells := 5
  let second_friends_shells := 17
  let total_shells := 37
  total_shells - jojanas_shells - second_friends_shells

-- Stating the theorem
theorem first_friend_added_15 : poundsAddedByFirstFriend = 15 := 
by
  sorry

end first_friend_added_15_l312_312585


namespace area_of_rectangle_l312_312616

-- Definitions of the geometric setup
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Rectangle :=
  (A B C D : Point)

def Diagonal (rect : Rectangle) : ℝ := 
  real.sqrt ((rect.A.x - rect.C.x)^2 + (rect.A.y - rect.C.y)^2)

-- Given conditions
def Segment_Division (DB : ℝ) (E F : ℝ) : Prop :=
  DB = 6 ∧ E = 2 ∧ F = 3

-- Prove the area of the rectangle
theorem area_of_rectangle (rect : Rectangle) (DB E F : ℝ)
  (h1 : Segment_Division DB E F)
  (h2 : DB = 2 + 3 + 1) :
  let area := 6 * real.sqrt 5 in
  real.round (area * 10) / 10 = 13.4 := 
by
  sorry

end area_of_rectangle_l312_312616


namespace julia_drove_miles_l312_312752

theorem julia_drove_miles :
  ∀ (daily_rental_cost cost_per_mile total_paid : ℝ),
    daily_rental_cost = 29 →
    cost_per_mile = 0.08 →
    total_paid = 46.12 →
    total_paid - daily_rental_cost = cost_per_mile * 214 :=
by
  intros _ _ _ d_cost_eq cpm_eq tp_eq
  -- calculation and proof steps will be filled here
  sorry

end julia_drove_miles_l312_312752


namespace card_drawing_certain_l312_312391

theorem card_drawing_certain (A B C : Type)
    (hearts : Finset A) (clubs : Finset B) (spades : Finset C)
    (cardinality_conditions : hearts.card = 5 ∧ clubs.card = 4 ∧ spades.card = 3)
    (total_cards : Finset (A ⊕ B ⊕ C))
    (total_size_condition : total_cards.card = 12)
    (draw_size : ℕ)
    (draw_size_condition : draw_size = 10) :
  ∀ (draw : Finset (A ⊕ B ⊕ C)), draw.card = 10 → 
    (∃ h ∈ hearts, ∃ c ∈ clubs, ∃ s ∈ spades, h ∈ draw ∧ c ∈ draw ∧ s ∈ draw) :=
by
  sorry

end card_drawing_certain_l312_312391


namespace oxygen_gas_produced_l312_312072

def molar_mass_KClO3 : ℝ := 122.6
def mass_KClO3 : ℝ := 245
def moles_O2_produced (moles_KClO3 : ℝ) : ℝ := (3 / 2) * moles_KClO3

theorem oxygen_gas_produced :
  let moles_KClO3 := mass_KClO3 / molar_mass_KClO3
  in moles_O2_produced moles_KClO3 = 3 :=
by
  sorry

end oxygen_gas_produced_l312_312072


namespace count_divisors_36_l312_312485

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312485


namespace hexagon_ratio_identity_l312_312141

theorem hexagon_ratio_identity
  (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (AB BC CD DE EF FA : ℝ)
  (angle_B angle_D angle_F : ℝ)
  (h1 : AB / BC * CD / DE * EF / FA = 1)
  (h2 : angle_B + angle_D + angle_F = 360) :
  (BC / AC * AE / EF * FD / DB = 1) := sorry

end hexagon_ratio_identity_l312_312141


namespace non_negative_integer_solutions_l312_312008

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem non_negative_integer_solutions :
  ∀ x y z : ℕ,
    x! + 2^y = z! →
    (x = 0 ∧ y = 0 ∧ z = 2) ∨
    (x = 1 ∧ y = 0 ∧ z = 2) ∨
    (x = 2 ∧ y = 2 ∧ z = 3) :=
begin
  sorry -- The proof is omitted
end

end non_negative_integer_solutions_l312_312008


namespace number_of_pairs_of_students_l312_312389

theorem number_of_pairs_of_students : 
  let n := 12 in 
  (n.choose 2) = 66 :=
by
  let n := 12
  have h : n.choose 2 = 66 := by sorry
  exact h

end number_of_pairs_of_students_l312_312389


namespace find_kn_l312_312380

section
variables (k n : ℝ)

def system_infinite_solutions (k n : ℝ) :=
  ∃ (y : ℝ → ℝ) (x : ℝ → ℝ),
  (∀ y, k * y + x y + n = 0) ∧
  (∀ y, |y - 2| + |y + 1| + |1 - y| + |y + 2| + x y = 0)

theorem find_kn :
  { (k, n) | system_infinite_solutions k n } = {(4, 0), (-4, 0), (2, 4), (-2, 4), (0, 6)} :=
sorry
end

end find_kn_l312_312380


namespace repeating_decimal_fraction_count_l312_312393

theorem repeating_decimal_fraction_count :
  (finset.range 30).filter (λ n, ¬(3 ∣ (n + 1))).card = 20 :=
by
  sorry

end repeating_decimal_fraction_count_l312_312393


namespace area_of_polygon_l312_312110

-- Define the conditions
variables (n : ℕ) (s : ℝ)
-- Given that polygon has 32 sides.
def sides := 32
-- Each side is congruent, and the total perimeter is 64 units.
def perimeter := 64
-- Side length of each side
def side_length := perimeter / sides

-- Area of the polygon we need to prove
def target_area := 96

theorem area_of_polygon : side_length * side_length * sides = target_area := 
by {
  -- Here proof would come in reality, we'll skip it by sorry for now.
  sorry
}

end area_of_polygon_l312_312110


namespace hypotenuse_of_45_45_90_triangle_l312_312625

-- Definitions directly from the conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def is_45_45_90 (a b c : ℕ) : Prop :=
  a = b ∧ c = a * (Real.sqrt 2)

-- Given values from conditions
def leg_length : ℕ := 8
def hypotenuse_length : ℝ := 8 * (Real.sqrt 2)

-- Lean 4 statement to prove the hypotenuse of the 45-45-90 triangle
theorem hypotenuse_of_45_45_90_triangle : 
  ∃ c : ℝ, is_45_45_90 leg_length leg_length c ∧ c = hypotenuse_length :=
by
  sorry

end hypotenuse_of_45_45_90_triangle_l312_312625


namespace artifacts_in_each_wing_l312_312777

theorem artifacts_in_each_wing (total_wings : ℕ) (artifact_factor : ℕ) (painting_wings : ℕ) 
  (large_painting : ℕ) (small_paintings_per_wing : ℕ) (remaining_artifact_wings : ℕ) 
  (total_paintings constant_total_paintings_expected : ℕ) (total_artifacts total_artifacts_expected : ℕ) 
  (artifacts_per_wing artifacts_per_wing_expected : ℕ) : 

  total_wings = 8 →
  artifact_factor = 4 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  remaining_artifact_wings = total_wings - painting_wings →
  total_paintings = painting_wings * small_paintings_per_wing + large_painting →
  total_artifacts = total_paintings * artifact_factor →
  artifacts_per_wing = total_artifacts / remaining_artifact_wings →
  artifacts_per_wing = 20 :=

by
    intros htotal_wings hartifact_factor hpainting_wings hlarge_painting hsmall_paintings_per_wing hermaining_artifact_wings htotal_paintings htotal_artifacts hartifacts_per_wing,
    sorry

end artifacts_in_each_wing_l312_312777


namespace employees_use_public_transport_l312_312756

-- Define the main assumptions based on the given conditions
def total_employees := 100
def drives_to_work := 0.60
def fraction_take_public_transport := 0.5

-- Define the problem statement
theorem employees_use_public_transport : 
  let drives := drives_to_work * total_employees in
  let doesnt_drive := total_employees - drives in
  let takes_public_transport := doesnt_drive * fraction_take_public_transport in
  takes_public_transport = 20 :=
by
  sorry

end employees_use_public_transport_l312_312756


namespace playground_total_children_l312_312230

theorem playground_total_children (boys girls : ℕ) (hb : boys = 27) (hg : girls = 35) : boys + girls = 62 := by
  rw [hb, hg]
  exact rfl -- This ensures the rewritten terms match the expected sum
  sorry -- Placeholder for proof confirmation

end playground_total_children_l312_312230


namespace find_length_KL_l312_312573

theorem find_length_KL 
  (J K L : Type)
  (JK JL KL : ℝ)
  (right_triangle : ∀ (J K L : Type), Prop)
  (h1 : right_triangle J K L)
  (tan_K_eq : ∀ (K : Type), Real.tan K = 4 / 3)
  (JK_eq : JK = 3) 
  : KL = 5 :=
by
  sorry

end find_length_KL_l312_312573


namespace gcd_expression_multiple_of_456_l312_312902

theorem gcd_expression_multiple_of_456 (a : ℤ) (h : ∃ k : ℤ, a = 456 * k) : 
  Int.gcd (3 * a^3 + a^2 + 4 * a + 57) a = 57 := by
  sorry

end gcd_expression_multiple_of_456_l312_312902


namespace total_volume_calculation_l312_312721

theorem total_volume_calculation (
  length width height : ℕ,
  cost_per_box total_spent : ℚ,
  volume_per_box total_boxes : ℕ
) (h₁ : length = 20) (h₂ : width = 20) (h₃ : height = 15) 
  (h₄ : cost_per_box = 0.90) (h₅ : total_spent = 459) 
  (h₆ : volume_per_box = length * width * height)
  (h₇ : total_boxes = total_spent / cost_per_box) :
  total_boxes * volume_per_box = 3060000 := sorry

end total_volume_calculation_l312_312721


namespace divisors_of_36_l312_312476

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312476


namespace proof_problem_l312_312097

open Real

-- Definitions of curves and transformations
def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def C2 := { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 }

-- Parametric equation of C2
def parametric_C2 := ∃ α : ℝ, (0 ≤ α ∧ α ≤ 2*π) ∧
  (C2 = { p : ℝ × ℝ | p.1 = 2 * cos α ∧ p.2 = (1/2) * sin α })

-- Equation of line l1 maximizing the perimeter of ABCD
def line_l1 (p : ℝ × ℝ): Prop :=
  p.2 = (1/4) * p.1

theorem proof_problem : parametric_C2 ∧
  ∀ (A B C D : ℝ × ℝ),
    (A ∈ C2 ∧ B ∈ C2 ∧ C ∈ C2 ∧ D ∈ C2) →
    (line_l1 A ∧ line_l1 B) → 
    (line_l1 A ∧ line_l1 B) ∧
    (line_l1 C ∧ line_l1 D) →
    y = (1 / 4) * x :=
sorry

end proof_problem_l312_312097


namespace difference_of_digits_4086_l312_312712

theorem difference_of_digits_4086 :
  ∃ x : ℕ, let largest := 1000 * 4 + 100 * x + 10 * 3 + 1,
               smallest := 1000 * 1 + 100 * 3 + 10 * 4 + x,
           in largest - smallest = 4086 :=
by
  sorry

end difference_of_digits_4086_l312_312712


namespace tan_A_eq_sqrt_3_l312_312428

-- Definitions given in the problem
def perimeter (a b c : ℝ) := a + b + c
def inradius (a b c : ℝ) (r : ℝ) := r
def sideBC (b c : ℝ) := b = c
def tanA (a b c : ℝ) : ℝ := (2 * tan (0.5 * atan ((sqrt (3)) / 3))) / (1 - (tan (0.5 * atan ((sqrt (3)) / 3)))^2)

-- Statement to prove
theorem tan_A_eq_sqrt_3 (a b c : ℝ) (r : ℝ) (h1 : perimeter a b c = 20)
                            (h2 : inradius a b c r = sqrt 3) (h3 : b = 7) :
                            tanA a b c = sqrt 3 :=
by
  sorry

end tan_A_eq_sqrt_3_l312_312428


namespace divisors_of_36_count_l312_312513

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312513


namespace weave_mats_l312_312295

theorem weave_mats (m n p q : ℕ) (h1 : m * n = p * q) (h2 : ∀ k, k = n → n * 2 = k * 2) :
  (8 * 2 = 16) :=
by
  -- This is where we would traditionally include the proof steps.
  sorry

end weave_mats_l312_312295


namespace hamburger_count_l312_312463

-- Define the number of condiments and their possible combinations
def condiment_combinations : ℕ := 2 ^ 10

-- Define the number of choices for meat patties
def meat_patties_choices : ℕ := 4

-- Define the total count of different hamburgers
def total_hamburgers : ℕ := condiment_combinations * meat_patties_choices

-- The theorem statement proving the total number of different hamburgers
theorem hamburger_count : total_hamburgers = 4096 := by
  sorry

end hamburger_count_l312_312463


namespace smoking_health_correlation_is_negative_l312_312744

-- Define the relationship and the negative correlation.
def smoking_is_harmful_to_health := ∀ (S H : Type), (smoking S) -> (health H) -> ((correlation S H) = negative_correlation)

-- The theorem stating the main problem to be proved.
theorem smoking_health_correlation_is_negative (S H : Type) (hs : smoking_is_harmful_to_health S H) :
  (correlation S H) = negative_correlation :=
by
  sorry

end smoking_health_correlation_is_negative_l312_312744


namespace symmetric_circle_eq_l312_312011

/-- The definition of the original circle equation. -/
def original_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The definition of the line of symmetry equation. -/
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

/-- The statement that the equation of the circle that is symmetric to the original circle 
    about the given line is (x - 4)^2 + (y + 1)^2 = 1. -/
theorem symmetric_circle_eq : 
  (∃ x y : ℝ, original_circle_eq x y ∧ line_eq x y) →
  (∀ x y : ℝ, (x - 4)^2 + (y + 1)^2 = 1) :=
by sorry

end symmetric_circle_eq_l312_312011


namespace ring_stack_distance_l312_312788

theorem ring_stack_distance :
  ∀ (n : ℕ) (a l : ℕ), a = 19 ∧ l = 2 ∧ n = 18 →
    (∑ k in (finset.range n).filter (λ x, x ≥ 1), (a - k + 1)) + 2 = 173 :=
by intros n a l h; sorry

end ring_stack_distance_l312_312788


namespace sum_of_squares_of_coefficients_l312_312818

theorem sum_of_squares_of_coefficients :
  let expr := 3 * (x^3 - x^2 + 4) - 5 * (x^2 - 2x + 3)
  let coeffs := [3, -8, 10, -3]
  let squares := coeffs.map (λ c => c^2)
  let sum_of_squares := squares.sum
  sum_of_squares = 182 := 
by
  let expr := 3 * (x^3 - x^2 + 4) - 5 * (x^2 - 2x + 3)
  let coeffs := [3, -8, 10, -3]
  let squares := coeffs.map (λ c => c^2)
  have h1: squares = [9, 64, 100, 9], by sorry
  have h2: sum_of_squares = 9 + 64 + 100 + 9, by sorry
  show sum_of_squares = 182, by sorry

end sum_of_squares_of_coefficients_l312_312818


namespace ordered_pairs_count_l312_312336

-- Define the conditions mathematically
def is_real (x y : ℤ) : Prop := ∃ k1 k2 : ℤ, i^x + i^y = (k1 + k2 * I) ∧ k2 = 0

theorem ordered_pairs_count :
  (∑ x in finset.Icc 1 200, ∑ y in finset.Icc (x + 1) 200, if is_real x y then 1 else 0) = 3651 := sorry

end ordered_pairs_count_l312_312336


namespace total_apples_correct_l312_312831

def craig_initial := 20.5
def judy_initial := 11.25
def dwayne_initial := 17.85
def eugene_to_craig := 7.15
def craig_to_dwayne := 3.5 / 2
def judy_to_sally := judy_initial / 2

def craig_final := craig_initial + eugene_to_craig - craig_to_dwayne
def dwayne_final := dwayne_initial + craig_to_dwayne
def judy_final := judy_initial - judy_to_sally
def sally_final := judy_to_sally

def total_apples := craig_final + judy_final + dwayne_final + sally_final

theorem total_apples_correct : total_apples = 56.75 := by
  -- skipping proof
  sorry

end total_apples_correct_l312_312831


namespace question_1_question_2_l312_312049

noncomputable def f (x m : ℝ) : ℝ := abs (x + m) - abs (2 * x - 2 * m)

theorem question_1 (x : ℝ) (m : ℝ) (h : m = 1/2) (h_pos : m > 0) : 
  (f x m ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

theorem question_2 (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, ∃ t : ℝ, f x m + abs (t - 3) < abs (t + 4)) ↔ (0 < m ∧ m < 7/2) :=
sorry

end question_1_question_2_l312_312049


namespace find_weight_B_l312_312737

-- Define the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions
def avg_weight_ABC := A + B + C = 135
def avg_weight_AB := A + B = 80
def avg_weight_BC := B + C = 86

-- The statement to be proved
theorem find_weight_B (h1: avg_weight_ABC A B C) (h2: avg_weight_AB A B) (h3: avg_weight_BC B C) : B = 31 :=
sorry

end find_weight_B_l312_312737


namespace divisors_of_36_count_l312_312516

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312516


namespace remainder_of_max_value_l312_312017

-- Definitions for binomial coefficient and gcd calculations
def binomial (n k : ℕ) : ℕ := nat.choose n k
def gcd (a b : ℕ) : ℕ := nat.gcd a b

-- Definitions for specific problem conditions
def A (k : ℕ) : ℕ := binomial 100 k
def B (k : ℕ) : ℕ := binomial 100 (k + 3)

-- Main Lean statement to be proven
theorem remainder_of_max_value :
  (max (λ k : ℕ, if 30 ≤ k ∧ k ≤ 70 then (A k) / (gcd (A k) (B k)) else 0) % 1000) = 664 :=
by
  sorry

end remainder_of_max_value_l312_312017


namespace find_AB_l312_312116

-- Definitions for given conditions
variable (ABCD : Type) [Nonempty ABCD]
variable {A B C D : Point ABCD}
variable [InCircle ABCD]
variable [Trapezoid ABCD A B C D]

-- Given values and assumptions
variable (a : ℝ) (α : ℝ)
variable (h_angle_bad : α = 45)
variable (h_parallel : Parallel A B D C)
variable (h_equal_sides : Distance A B = a ∧ Distance C D = a)
variable (h_area : Area ABCD = 10)

-- Goal: Find the length of AB
theorem find_AB :
  Distance A B = Real.sqrt (10 * Real.sqrt 2) :=
by
  sorry

end find_AB_l312_312116


namespace curve_lies_in_plane_l312_312976

theorem curve_lies_in_plane (φ : ℝ) (hφ : 0 ≤ φ ∧ φ ≤ 2 * Real.pi) :
    ∃ (A B C D : ℝ), A ≠ 0 ∧ B = 0 ∧ D = 0 ∧ (∀ (φ : ℝ), 0 ≤ φ ∧ φ ≤ 2 * Real.pi →
    A * Real.sin φ + B * Real.cos φ + C * Real.sin φ + D = 0) :=
begin
  use [1, 0, -1, 0],
  split,
  { exact one_ne_zero },
  split,
  { refl },
  split,
  { refl },
  intros φ hφ,
  simp,
  ring,
end

end curve_lies_in_plane_l312_312976


namespace cube_root_of_a_l312_312950

theorem cube_root_of_a (a : ℤ) (h1 : real.sqrt 59 < a) (h2 : a < real.sqrt 65) : real.sqrt (↑a ^ 3) = 2 := 
sorry

end cube_root_of_a_l312_312950


namespace brother_highlighters_spent_l312_312942

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end brother_highlighters_spent_l312_312942


namespace expected_value_is_correct_l312_312766

noncomputable def expected_winnings : ℝ :=
  (1/12 : ℝ) * (9 + 8 + 7 + 6 + 5 + 1 + 2 + 3 + 4 + 5 + 6 + 7)

theorem expected_value_is_correct : expected_winnings = 5.25 := by
  sorry

end expected_value_is_correct_l312_312766


namespace count_valid_pairs_is_7_l312_312073

def valid_pairs_count : Nat :=
  let pairs := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
  List.length pairs

theorem count_valid_pairs_is_7 (b c : ℕ) (hb : b > 0) (hc : c > 0) :
  (b^2 - 4 * c ≤ 0) → (c^2 - 4 * b ≤ 0) → valid_pairs_count = 7 :=
by
  sorry

end count_valid_pairs_is_7_l312_312073


namespace segment_length_parallel_l312_312962

/-- Let ABC be a triangle with sides AB = 300, BC = 350, and AC = 400.
    If an interior point P is drawn, and segments are drawn through P parallel to the sides of the triangle,
    and these segments are of equal length d, then d = 200.
--/
theorem segment_length_parallel (AB BC AC d : ℝ) (hAB : AB = 300) (hBC : BC = 350) (hAC : AC = 400)
    (h_parallel : ∀ P : ℝ × ℝ, (segment_parallel_length d AB BC AC P)) : d = 200 :=
by
  sorry

/-- Definition of the segments being parallel and of equal length d --/
def segment_parallel_length (d AB BC AC : ℝ) (P : ℝ × ℝ) :=
  let PE := d / AC * AB
  let EE' := BC - 2 * PE
  let PF := d / AC * BC
  let FF' := AC - 2 * PF
  EE' = FF'

end segment_length_parallel_l312_312962


namespace find_a_l312_312922

def f (a x : ℝ) : ℝ := (Real.log x) / (a * x + 1)
def f_prime_at (a x : ℝ) : ℝ := 
  ((a * x + 1 - a * x * Real.log x) / (x * (a * x + 1) ^ 2))

theorem find_a (a : ℝ) : 
  f_prime_at a Real.exp = 1 / (Real.exp * (1 - Real.exp) ^ 2)  ↔ 
  (a = -1 ∨ a = (Real.exp - 2) / Real.exp) := 
sorry

end find_a_l312_312922


namespace MinimumC_SumB_inverse_l312_312409

noncomputable def a : ℕ → ℕ 
| n := 2^n

def b : ℕ → ℕ 
| n := 2 * n - 1

def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (b i) / (a i)

noncomputable def B (n : ℕ) : ℕ := n^2

theorem MinimumC (c : ℤ) : ∀ n : ℕ, c > T n → 3 ≤ c := sorry

theorem SumB_inverse : ∀ n : ℕ, ∑ i in Finset.range n.succ, 1 / (B i) < 5 / 3 := sorry

end MinimumC_SumB_inverse_l312_312409


namespace number_of_diet_soda_bottles_l312_312769

theorem number_of_diet_soda_bottles (apples regular_soda total_bottles diet_soda : ℕ)
    (h_apples : apples = 36)
    (h_regular_soda : regular_soda = 80)
    (h_total_bottles : total_bottles = apples + 98)
    (h_diet_soda_eq : total_bottles = regular_soda + diet_soda) :
    diet_soda = 54 := by
  sorry

end number_of_diet_soda_bottles_l312_312769


namespace decimal_digits_fraction_l312_312346

theorem decimal_digits_fraction :
  let frac := (987654321 : ℚ) / (2^30 * 5^2)
  in real.truncate_digits frac = 30 :=
by
  let frac := (987654321 : ℚ) / (2^30 * 5^2)
  sorry

end decimal_digits_fraction_l312_312346


namespace smallest_perimeter_consecutive_even_triangle_l312_312250

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l312_312250


namespace angle_BPA_l312_312993

noncomputable def inside_square (A B C D P : ℝ × ℝ) := 
  ∃ (s : ℝ), 
    (s > 0) ∧ 
    A = (0, 0) ∧ 
    B = (s, 0) ∧ 
    C = (s, s) ∧ 
    D = (0, s) ∧
    ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ s) ∧ (0 ≤ y ∧ y ≤ s) ∧ P = (x, y)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem angle_BPA 
  (A B C D P : ℝ × ℝ)
  (h1 : inside_square A B C D P)
  (h2 : distance P A = k)
  (h3 : distance P B = 2 * k)
  (h4 : distance P C = 3 * k) :
  ∠ B P A = 135 :=
sorry

end angle_BPA_l312_312993


namespace profit_percentage_is_ten_l312_312795

-- Define the cost price (CP) and selling price (SP) as constants
def CP : ℝ := 90.91
def SP : ℝ := 100

-- Define a theorem to prove the profit percentage is 10%
theorem profit_percentage_is_ten : ((SP - CP) / CP) * 100 = 10 := 
by 
  -- Skip the proof.
  sorry

end profit_percentage_is_ten_l312_312795


namespace Tim_total_spent_l312_312082

theorem Tim_total_spent (lunch_cost : ℝ) (tip_percent : ℝ) (tip_amount : ℝ) (total_spent : ℝ) :
  lunch_cost = 50.20 ∧ tip_percent = 0.20 ∧ tip_amount = tip_percent * lunch_cost ∧ total_spent = lunch_cost + tip_amount
  → total_spent = 60.24 :=
by
  intros h
  cases h with lunch_cost_eq h
  cases h with tip_percent_eq h
  cases h with tip_amount_eq total_spent_eq
  rw [lunch_cost_eq, tip_percent_eq, tip_amount_eq, total_spent_eq]
  norm_num 
  exact total_spent_eq
  sorry

end Tim_total_spent_l312_312082


namespace valid_third_side_length_l312_312910

theorem valid_third_side_length : 4 < 6 ∧ 6 < 10 :=
by
  exact ⟨by norm_num, by norm_num⟩

end valid_third_side_length_l312_312910


namespace largest_n_for_factoring_l312_312871

theorem largest_n_for_factoring :
  ∃ n, (∃ A B : ℤ, (3 * A = 3 * 108 + 1) ∧ (/3 * B * 108 = 2) ∧ 
  (3 * 36 + 3 = 111) ∧ (3 * 108 + A = n) )=
  (n = 325) := sorry
iddenLean_formatter.clonecreateAngular

end largest_n_for_factoring_l312_312871


namespace solve_problem_l312_312841

universe u

structure Graph (V : Type u) :=
  (adj : V → V → Prop)
  (sym : symmetric adj . obviously)
  (loopless : irreflexive adj . obviously)
  (finite : Fintype V . obviously)

variable {V : Type u} [DecidableEq V] [Fintype V]

def initial_coloring (G : Graph V) : V → bool := λ v, false

def toggle (G : Graph V) (v : V) (c : V → bool) : V → bool :=
  λ u, if u = v ∨ G.adj v u then ¬ c u else c u

def can_make_every_vertex_white (G : Graph V) : Prop :=
  ∃ seq : list V, (seq.foldr (toggle G) (initial_coloring G) = λ _, true)

theorem solve_problem (G : Graph V) : can_make_every_vertex_white G :=
  sorry

end solve_problem_l312_312841


namespace find_lambda_l312_312912

variables {R : Type*} [Field R] 
variables (e1 e2 a b : R) (λ : R)

-- Definitions of vectors and basis
def basis_vectors := e1 ≠ 0 ∧ e2 ≠ 0

-- Definition of vectors a and b
def vector_a := -3 * e1 - 1 * e2
def vector_b := e1 - λ * e2

-- Definition of collinearity
def collinear (v1 v2 : R) := ∃ μ : R, v1 = μ * v2

-- Proof statement
theorem find_lambda 
  (h_basis : basis_vectors e1 e2)
  (h_a : vector_a e1 e2 a)
  (h_b : vector_b e1 e2 b λ)
  (h_collinear : collinear a b) : λ = -1/3 :=
sorry

end find_lambda_l312_312912


namespace num_divisors_of_36_l312_312486

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312486


namespace average_speed_l312_312842

theorem average_speed (initial final time : ℕ) (h_initial : initial = 2002) (h_final : final = 2332) (h_time : time = 11) : 
  (final - initial) / time = 30 := by
  sorry

end average_speed_l312_312842


namespace sufficient_condition_for_inequality_l312_312429

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end sufficient_condition_for_inequality_l312_312429


namespace find_natural_numbers_l312_312289

theorem find_natural_numbers (n : ℕ) (h : n ≥ 2) :
  (∃ a : Fin n → ℝ, { |a i - a j| | i j : Fin n, i < j }.card = n * (n - 1) / 2) ↔ n = 2 ∨ n = 3 ∨ n = 4 := 
sorry

end find_natural_numbers_l312_312289


namespace provisions_duration_l312_312768

theorem provisions_duration (initial_men: ℕ) (initial_days: ℕ) (days_elapsed: ℕ) (reinforcements: ℕ) : 
  initial_men = 2000 →
  initial_days = 54 →
  days_elapsed = 21 →
  reinforcements = 1300 →
  (initial_men * (initial_days - days_elapsed)) / (initial_men + reinforcements) = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end provisions_duration_l312_312768


namespace number_exceeds_80_by_120_l312_312305

theorem number_exceeds_80_by_120 : ∃ x : ℝ, x = 0.80 * x + 120 ∧ x = 600 :=
by sorry

end number_exceeds_80_by_120_l312_312305


namespace even_function_zeros_l312_312613

noncomputable def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

theorem even_function_zeros (m : ℝ) (h : ∀ x : ℝ, f x m = f (-x) m ) : 
  m = 1 ∧ (∀ x : ℝ, f x m = 0 → (x = 1 ∨ x = -1)) := by
  sorry

end even_function_zeros_l312_312613


namespace red_rose_ratio_l312_312620

-- Define the conditions as given in part a)
variables (R Y W : ℕ)
variable (totalRoses : ℕ := 80)

-- Mrs. Amaro has 80 roses in her garden
axiom h1 : R + Y + W = totalRoses

-- There are 75 roses that are either red or white.
axiom h2 : R + W = 75

-- One-fourth of the remaining roses after the red ones are yellow.
axiom h3 : Y = 1 / 4 * (totalRoses - R)

-- The ratio of red roses to the total number of roses is 3:4
theorem red_rose_ratio :
  R.toReal / totalRoses.toReal = 3 / 4 :=
by {
  -- The proof is skipped with 'sorry' as per instruction
  sorry
}

end red_rose_ratio_l312_312620


namespace repeating_decimal_to_fraction_l312_312377

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 2.3333333333333…) : x = 7 / 3 := 
sorry

end repeating_decimal_to_fraction_l312_312377


namespace compare_powers_l312_312821

theorem compare_powers :
  4^(1/4:ℝ) > 6^(1/6:ℝ) ∧ 6^(1/6:ℝ) > 16^(1/16:ℝ) ∧ 16^(1/16:ℝ) > 27^(1/27:ℝ) :=
sorry

end compare_powers_l312_312821


namespace roots_complex_conjugates_are_real_l312_312146

theorem roots_complex_conjugates_are_real (a b : ℝ) :
  (∀ z : ℂ, z^2 + (15 + a * complex.I) * z + (40 + b * complex.I) = 0 → (∃ x y : ℝ, z = x + y * complex.I ∧ (z + conj z = 2 * x) ∧ (z * conj z = x^2 + y^2))) →
  a = 0 ∧ b = 0 :=
sorry

end roots_complex_conjugates_are_real_l312_312146


namespace negation1_converse1_negation2_converse2_negation3_converse3_l312_312729

-- Definitions
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_prime (p : ℕ) : Prop := nat.prime p

-- Proof statements
theorem negation1 : ¬ (∀ x y : ℤ, is_odd x → is_odd y → is_even (x + y)) ↔ true := sorry
theorem converse1 : ¬ (∀ x y : ℤ, ¬ (is_odd x ∧ is_odd y) → ¬ is_even (x + y)) ↔ true := sorry

theorem negation2 : (∀ x y : ℤ, x * y = 0 → ¬ (x = 0) ∧ ¬ (y = 0)) ↔ false := sorry
theorem converse2 : (∀ x y : ℤ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ↔ true := sorry

theorem negation3 : (∀ p : ℕ, is_prime p → ¬ (is_odd p)) ↔ false := sorry
theorem converse3 : (∀ p : ℕ, ¬ is_prime p → ¬ (is_odd p)) ↔ false := sorry

end negation1_converse1_negation2_converse2_negation3_converse3_l312_312729


namespace concyclic_points_and_locus_center_l312_312705

theorem concyclic_points_and_locus_center (a : ℝ) :
  let line1 := λ (x : ℝ), a + 2 * x
  let line2 := λ (x : ℝ), a + (1/2) * x
  let hyp := λ (x : ℝ), 1 / x
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ), ((y = line1 x) ∨ (y = line2 x)) ∧ (y = hyp x) →
    ((x - O.1)^2 + (y - O.2)^2 = r^2) ∧ (O.2 = - (4 / 5) * O.1) := sorry

end concyclic_points_and_locus_center_l312_312705


namespace num_divisors_of_36_l312_312521

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312521


namespace triangle_area_l312_312815

/-- Given a triangle ABC with sides a and b, and an angle bisector l,
the area of the triangle is given by (l(a+b)/(4ab)) * sqrt(4a^2b^2 - l^2(a+b)^2). --/
theorem triangle_area (a b l : ℝ) (α : ℝ) (hA : α = real.arccos (l * (a + b) / (2 * a * b))) :
  1 / 2 * a * b * real.sin (2 * α) = (l * (a + b)) / (4 * a * b) * real.sqrt (4 * a^2 * b^2 - l^2 * (a + b)^2) :=
begin
  sorry
end

end triangle_area_l312_312815


namespace employees_use_public_transportation_l312_312759

theorem employees_use_public_transportation
    (total_employees : ℕ)
    (drive_percentage : ℝ)
    (public_transportation_fraction : ℝ)
    (h1 : total_employees = 100)
    (h2 : drive_percentage = 0.60)
    (h3 : public_transportation_fraction = 0.50) :
    ((total_employees * (1 - drive_percentage)) * public_transportation_fraction) = 20 :=
by
    sorry

end employees_use_public_transportation_l312_312759


namespace time_to_finish_typing_l312_312579

-- Definitions
def words_per_minute : ℕ := 38
def total_words : ℕ := 4560

-- Theorem to prove
theorem time_to_finish_typing : (total_words / words_per_minute) / 60 = 2 := by
  sorry

end time_to_finish_typing_l312_312579


namespace greatest_possible_k_l312_312979

-- Define the number of saunas
def num_saunas : ℕ := 2019

-- Define the conditions in Lean
structure Guest :=
(Knows : Guest → Prop)

structure Couple :=
(woman : Guest)
(man : Guest)

def max_couples (num_saunas : ℕ) : ℕ :=
  if num_saunas ≥ 2 then num_saunas - 1 else 0

-- Prove that the greatest possible k for the conditions is 2018
theorem greatest_possible_k (num_saunas = 2019) : max_couples num_saunas = 2018 :=
by
  sorry

end greatest_possible_k_l312_312979


namespace find_doctor_and_engineer_l312_312701

theorem find_doctor_and_engineer
  (Raja Omar Beatty : String)
  (is_doctor : String → Prop)
  (is_engineer : String → Prop)
  (is_musician : String → Prop)
  (youngest : ∀ (x : String), is_doctor x → ¬ (x = Omar ∧ has_brother x))
  (older_than : ∀ (x y : String), is_engineer y → is_musician x → older x y)
  (married_to_brother : ∀ (x y : String), x = Beatty ∧ y = Omar ∧ is_musician y)
  : (is_doctor Raja ∧ is_engineer Omar) ∧
      ∀ z, z ≠ Raja → z ≠ Omar → ¬ is_doctor z ∧ ¬ is_engineer z :=
by 
  sorry

def has_brother : String → Prop := sorry
def older (x y : String) : Prop := sorry
def musician (x : String) (y : String) : Prop := sorry

end find_doctor_and_engineer_l312_312701


namespace cannot_form_triangle_5_8_2_l312_312727

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem cannot_form_triangle_5_8_2 :
  ¬ satisfies_triangle_inequality 5 8 2 :=
by
  -- Applying the triangle inequality theorem:
  -- We need to check a + b > c, a + c > b, and b + c > a
  unfold satisfies_triangle_inequality
  split
  · -- case 1: check 5 + 8 > 2
    exact Nat.gt_lt_trans (Nat.add_lt_add_right (Nat.lt_succ_self 5) 8) (Nat.lt_succ_self 13)
  split
  · -- case 2: check 5 + 2 > 8
    simp
  · -- case 3: check 8 + 2 > 5
    exact Nat.add_lt_add_right (Nat.lt_succ_self 8) 2

end cannot_form_triangle_5_8_2_l312_312727


namespace remainder_div_by_8_l312_312723

noncomputable def bin_to_nat (s : String) : Nat :=
  if s.length = 0 then 0
  else if s.head = '0' then bin_to_nat s.tail
  else 2 ^ (s.length - 1) + bin_to_nat s.tail

theorem remainder_div_by_8 (bin_num : String) (h : bin_num = "110110011110") : 
  (bin_to_nat (bin_num.slice (bin_num.length - 3) bin_num.length)) % 8 = 6 := 
by 
  have h_last_three : bin_num.slice (bin_num.length - 3) bin_num.length = "110" := by 
    rw [h] 
    simp
  rw [h_last_three] 
  norm_num
  sorry

end remainder_div_by_8_l312_312723


namespace left_handed_women_percentage_l312_312094

noncomputable section

variables (x y : ℕ) (percentage : ℝ)

-- Conditions
def right_handed_ratio := 3
def left_handed_ratio := 1
def men_ratio := 3
def women_ratio := 2

def total_population_by_hand := right_handed_ratio * x + left_handed_ratio * x -- i.e., 4x
def total_population_by_gender := men_ratio * y + women_ratio * y -- i.e., 5y

-- Main Statement
theorem left_handed_women_percentage (h1 : total_population_by_hand = total_population_by_gender) :
    percentage = 25 :=
by
  sorry

end left_handed_women_percentage_l312_312094


namespace line_tangent_to_ellipse_l312_312957

theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x : ℝ, x^2 + 4 * (m * x + 1)^2 = 1) → m^2 = 3 / 4 :=
by
  sorry

end line_tangent_to_ellipse_l312_312957


namespace ceiling_of_neg_3_7_l312_312371

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l312_312371


namespace proof_statement_l312_312764

noncomputable def point := (ℝ × ℝ)
noncomputable def radius := real.sqrt 80
noncomputable def center : point := (0, 0)
noncomputable def distance (p₁ p₂ : point) : ℝ :=
  real.sqrt ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2)
  
noncomputable def is_on_circle (p : point) (center : point) (r : ℝ) : Prop :=
  distance p center = r

noncomputable def a := (a₁ : ℝ, b₁ : ℝ)
noncomputable def b := (a₂ : ℝ, b₂ : ℝ)
noncomputable def c := (a₃ : ℝ, b₃ : ℝ)
noncomputable def proof_problem : Prop :=
  is_on_circle b center radius ∧ (distance a b) = 8 ∧ (distance b c) = 3 ∧ 
  ∃ (center : point) (a b : point), 
    a₁ + 8 = 0 ∧ (-a₂ + 3) ∧ 
    (distance b center) ^ 2 = 80

theorem proof_statement : proof_problem :=
by
  sorry

end proof_statement_l312_312764


namespace problem_l312_312404

open Real

noncomputable def f (ω a x : ℝ) := (1 / 2) * (sin (ω * x) + a * cos (ω * x))

theorem problem (a : ℝ) 
  (hω_range : 0 < ω ∧ ω ≤ 1)
  (h_f_sym1 : ∀ x, f ω a x = f ω a (π/3 - x))
  (h_f_sym2 : ∀ x, f ω a (x - π) = f ω a (x + π))
  (x1 x2 : ℝ) 
  (h_x_in_interval1 : -π/3 < x1 ∧ x1 < 5*π/3)
  (h_x_in_interval2 : -π/3 < x2 ∧ x2 < 5*π/3)
  (h_distinct : x1 ≠ x2)
  (h_f_neg_half1 : f ω a x1 = -1/2)
  (h_f_neg_half2 : f ω a x2 = -1/2) :
  (f 1 (sqrt 3) x = sin (x + π/3)) ∧ (x1 + x2 = 7*π/3) :=
by
  sorry

end problem_l312_312404


namespace ceil_neg_3_7_l312_312360

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l312_312360


namespace second_largest_possible_n_div_170_fact_l312_312538

theorem second_largest_possible_n_div_170_fact (n : ℕ) : 
  ∃ n, (n = 40 ∧ ∃ m : ℕ, 170! / (10^n) = m) :=
sorry

end second_largest_possible_n_div_170_fact_l312_312538


namespace trajectory_of_P_eqn_l312_312897

noncomputable def point_A : ℝ × ℝ := (1, 0)

def curve_C (x : ℝ) : ℝ := x^2 - 2

def symmetric_point (Qx Qy Px Py : ℝ) : Prop :=
  Qx = 2 - Px ∧ Qy = -Py

theorem trajectory_of_P_eqn (Qx Qy Px Py : ℝ) (hQ_on_C : Qy = curve_C Qx)
  (h_symm : symmetric_point Qx Qy Px Py) :
  Py = -Px^2 + 4 * Px - 2 :=
by
  sorry

end trajectory_of_P_eqn_l312_312897


namespace equal_sides_of_circumscribed_quadrilateral_l312_312682

variable {A B C M A1 B1 : Type*} [BarycentricCoord]

def intersect_at_median (A B C A1 B1 : Type*) (M : Type*) : Prop := 
  (is_median A B C A1) ∧ (is_median B A C B1) ∧ (intersects_at A A1 B B1 M)

def is_circumscribed (A1 M B1 C : Type*) : Prop := 
  exists_radius (circumcircle_radius A1 M B1 C)

theorem equal_sides_of_circumscribed_quadrilateral
  (h1 : intersect_at_median A B C A1 B1 M)
  (h2 : is_circumscribed A1 M B1 C) : 
  AC = BC := 
sorry

end equal_sides_of_circumscribed_quadrilateral_l312_312682


namespace always_possible_to_sort_volumes_l312_312163

def is_sorted (l : List ℕ) : Prop :=
  ∀ a b, a < b → l.nth a < l.nth b

def can_swap (l : List ℕ) (i j : ℕ) : Prop :=
  abs (i - j) > 4

theorem always_possible_to_sort_volumes :
  ∀ (l : List ℕ),
    l.length = 10 → -- Condition 1: There are ten volumes
    (∀ i j, can_swap l i j → ∃ l', l' = l.swap i j) → -- Condition 2: Swap condition
    ∃ k, is_sorted k := -- Goal: It is always possible to sort
begin
  sorry
end

end always_possible_to_sort_volumes_l312_312163


namespace unusual_die_min_dots_l312_312627

def is_unusual_die (dots : Fin 6 → ℕ) : Prop :=
  ∀ i j : Fin 6, i ≠ j → abs (dots i - dots j) ≥ 2

noncomputable def min_total_dots_on_unusual_die : ℕ :=
  1 + 2 + 4 + 5 + 7 + 8

theorem unusual_die_min_dots : ∃ dots : Fin 6 → ℕ, is_unusual_die dots ∧ (∑ i, dots i) = 27 :=
by
  use (λ i, [1, 2, 4, 5, 7, 8].nth i.val).get_or_else 0
  sorry

end unusual_die_min_dots_l312_312627


namespace angle_CMN_90_l312_312196

variable {α : Type*} [EuclideanSpace α]

noncomputable def trapezoid (A B C D M N : α) : Prop :=
  let O : α := circumcenter A B D
  let L : α := symmetry O AD
  is_midpoint M A B ∧ is_symmetric N O AD ∧
  perp (AC) (BD) ∧ angle (CMN) = 90

theorem angle_CMN_90 (A B C D M N : α)
  (h_midpointM : is_midpoint M A B)
  (h_symmetricN : is_symmetric N (circumcenter A B D) AD)
  (h_perpendicular : perp AC BD) :
  angle CMN = 90 :=
sorry

end angle_CMN_90_l312_312196


namespace brenda_age_l312_312316

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l312_312316


namespace find_a_l312_312439

noncomputable def f (a x : ℝ) : ℝ := a^x + log a x

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : (max (f a 1) (f a 2)) + (min (f a 1) (f a 2)) = log 2 / log a + 6) : 
  a = 2 := 
sorry

end find_a_l312_312439


namespace ceiling_of_neg_3_7_l312_312370

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l312_312370


namespace sum_first_five_terms_l312_312422

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 d : ℝ, ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Define the specific condition a_5 + a_8 - a_10 = 2
def specific_condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 8 - a 10 = 2

-- Define the sum of the first five terms S₅
def S5 (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 

-- The statement to be proved
theorem sum_first_five_terms (a : ℕ → ℝ) (h₁ : arithmetic_sequence a) (h₂ : specific_condition a) : 
  S5 a = 10 :=
sorry

end sum_first_five_terms_l312_312422


namespace calculateDistance_l312_312786

variable (RingThickness : ℕ := 1)
variable (TopRingOutsideDiameter : ℕ := 20)
variable (BottomRingOutsideDiameter : ℕ := 3)

def insideDiameter (outsideDiameter : ℕ) : ℕ :=
  outsideDiameter - 2 * RingThickness

def numberOfRings : ℕ :=
  (TopRingOutsideDiameter - BottomRingOutsideDiameter) + 1

def sumOfInsideDiameters : ℕ :=
  let n := numberOfRings
  let a := insideDiameter TopRingOutsideDiameter
  let l := insideDiameter BottomRingOutsideDiameter
  n * (a + l) / 2

def totalDistance : ℕ :=
  sumOfInsideDiameters + 2 * RingThickness

theorem calculateDistance : totalDistance = 173 := by
  sorry

end calculateDistance_l312_312786


namespace trigonometric_identity_l312_312024

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (\frac (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) 
           (Real.sin (\frac Real.pi 2 - α) + Real.cos (\frac Real.pi 2 + α))) = 2 :=
by
  sorry

end trigonometric_identity_l312_312024


namespace decreasing_interval_monotonic_find_minimum_a_l312_312438

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x

theorem decreasing_interval_monotonic (x : ℝ) (hx : 0 < x) :
  (derivative f x < 0) → (1 ≤ x) := 
sorry

theorem find_minimum_a (a : ℝ) : (∀ x, 0 < x → f(x) ≤ (a / 2 - 1) * x^2 + a * x - 1) → (2 ≤ a) :=
sorry

end decreasing_interval_monotonic_find_minimum_a_l312_312438


namespace angle_EKB_ninety_degrees_l312_312167

/--
Given a pentagon \(ABCDE\) inscribed around a circle \(\Omega\) such that \(AB = BC = CD\),
and the side \(BC\) touches the circle at point \(K\), prove that \(\angle EKB = 90^\circ\).
-/
theorem angle_EKB_ninety_degrees 
  (ABCDE : Type)
  (A B C D E K : ABCDE)
  (Ω : ABCDE)
  (touches_Ω : ℕ → Set(ABCDE) → Prop)
  (is_tangent : ∀ {X Y : ABCDE}, touches_Ω 2 [X, Y] → ℕ)
  (equal_sides : AB = BC = CD)
  (BC_touches_Ω_at_K : touches_Ω 2 [B, C] ∧ is_tangent BC_touches_Ω_at_K = K)
  : ∠EKB = 90 := 
sorry

end angle_EKB_ninety_degrees_l312_312167


namespace find_square_area_l312_312276

noncomputable theory
open scoped Real

def square_area (d : ℝ) := (d^2) / 2

theorem find_square_area : square_area 3.8 = 7.22 := 
by 
  have h : square_area 3.8 = (3.8^2) / 2 := rfl
  have h2 : (3.8^2) = 14.44 := by norm_num
  rw h2 at h
  have h3 : 14.44 / 2 = 7.22 := by norm_num
  rw h3 at h
  exact h

end find_square_area_l312_312276


namespace f_is_odd_f_is_monotone_l312_312057

noncomputable def f (k x : ℝ) : ℝ := x + k / x

-- Proving f(x) is odd
theorem f_is_odd (k : ℝ) (hk : k ≠ 0) : ∀ x : ℝ, f k (-x) = -f k x :=
by
  intro x
  sorry

-- Proving f(x) is monotonically increasing on [sqrt(k), +∞) for k > 0
theorem f_is_monotone (k : ℝ) (hk : k > 0) : ∀ x1 x2 : ℝ, 
  x1 ∈ Set.Ici (Real.sqrt k) → x2 ∈ Set.Ici (Real.sqrt k) → x1 < x2 → f k x1 < f k x2 :=
by
  intro x1 x2 hx1 hx2 hlt
  sorry

end f_is_odd_f_is_monotone_l312_312057


namespace distance_between_points_l312_312853

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  (real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def p1 : point := (0, 12)
def p2 : point := (5, 6)

theorem distance_between_points : distance p1 p2 = real.sqrt 61 := by
  sorry

end distance_between_points_l312_312853


namespace smallest_triangle_perimeter_consecutive_even_l312_312265

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l312_312265


namespace marks_in_chemistry_l312_312587

theorem marks_in_chemistry (marks_english marks_math marks_physics marks_biology : ℕ)
  (average total_subjects : ℕ) (h1 : marks_english = 76) (h2 : marks_math = 60)
  (h3 : marks_physics = 82) (h4 : marks_biology = 85) (h5 : average = 74) 
  (h6 : total_subjects = 5) : ∃ C : ℕ, C = 67 :=
by
  have total_known_marks : ℕ := marks_english + marks_math + marks_physics + marks_biology,
    calc total_known_marks = 76 + 60 + 82 + 85 : by
          rw [h1, h2, h3, h4]
            rw [← h1, ← h2, ← h3, ← h4]
    rw [h1, h2, h3, h4]
  have total_marks : ℕ := total_subjects * average,
    calc total_marks = 5 * 74 : by
          rw [h6, h5]
            rw [← h6, ← h5]
    rw [h6, h5]
  have eq_C : C = total_marks - total_known_marks, from sorry,
    exists.intro C eq_C 

end marks_in_chemistry_l312_312587


namespace number_of_terms_l312_312206

-- Define the sequence according to the given pattern
def sequence (n : ℕ) : List ℕ :=
  [0, 3, 6, ..., 3n+6]

-- The theorem to prove that the number of terms in the sequence is n + 3
theorem number_of_terms (n : ℕ) :
  length (sequence n) = n + 3 :=
by
  sorry

end number_of_terms_l312_312206


namespace brother_highlighter_spending_l312_312936

variables {total_money : ℝ} (sharpeners notebooks erasers highlighters : ℝ)

-- Conditions
def total_given := total_money = 100
def sharpeners_cost := 2 * 5
def notebooks_cost := 4 * 5
def erasers_cost := 10 * 4
def heaven_expenditure := sharpeners_cost + notebooks_cost
def brother_remaining := total_money - (heaven_expenditure + erasers_cost)
def brother_spent_on_highlighters := brother_remaining = 30

-- Statement
theorem brother_highlighter_spending (h1 : total_given) 
    (h2 : brother_spent_on_highlighters) : brother_remaining = 30 :=
sorry

end brother_highlighter_spending_l312_312936


namespace brenda_age_l312_312315

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l312_312315


namespace find_erased_number_l312_312229

-- Assume we have 21 consecutive natural numbers with the central number N.
variable (N : ℕ)

-- The sum of the remaining 20 numbers is 2017 after removing one number.
axiom erased_sum (x : ℕ) (h : x ∈ {N-10, N-9, ..., N, ..., N+9, N+10}) : 
  21 * N - x = 2017

-- Prove that the number erased is 104.
theorem find_erased_number : 
  ∃ (x : ℕ), (∀ (n : ℕ), N - 10 ≤ n ∧ n ≤ N + 10 → n ≠ x) ∧ 
  (21 * N - x = 2017) ∧ 
  x = 104 :=
sorry

end find_erased_number_l312_312229


namespace sum_positive_integer_c_real_roots_l312_312834

theorem sum_positive_integer_c_real_roots (c : ℕ) (h : ∀ x : ℝ, (x > 4) → (10 * x^2 + 25 * x + c = 0)) :
  (∑ i in finset.range 16, if i > 0 then i else 0) = 120 :=
by
  sorry

end sum_positive_integer_c_real_roots_l312_312834


namespace focus_distance_to_vertex_l312_312882

def parabola_f : ℝ → ℝ := fun p => p / 2

theorem focus_distance_to_vertex (p : ℝ) (hp : p > 0)
  (angle_eq : ∀ theta, theta = Real.pi / 4)
  (chord_length : ∀ x y, x^2 - 2 * p * x - y^2 = 0 → (2 * x = 4)) :
  parabola_f p = 1 / 2 :=
sorry

end focus_distance_to_vertex_l312_312882


namespace drunk_drivers_traffic_class_l312_312971

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end drunk_drivers_traffic_class_l312_312971


namespace drunk_drivers_count_l312_312968

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end drunk_drivers_count_l312_312968


namespace snake_eats_three_birds_per_day_l312_312350

noncomputable def birds_eaten_per_snake (B : ℕ) : Prop :=
  let beetles_eaten_per_bird := 12
  let snakes_eaten_per_jaguar := 5
  let jaguars := 6
  let total_beetles_eaten := 1080
  jaguars * snakes_eaten_per_jaguar * B * beetles_eaten_per_bird = total_beetles_eaten

theorem snake_eats_three_birds_per_day : birds_eaten_per_snake 3 :=
by
  let B := 3
  let beetles_eaten_per_bird := 12
  let snakes_eaten_per_jaguar := 5
  let jaguars := 6
  let total_beetles_eaten := 1080
  have h : jaguars * snakes_eaten_per_jaguar * B * beetles_eaten_per_bird = total_beetles_eaten,
    sorry
  exact h

end snake_eats_three_birds_per_day_l312_312350


namespace count_divisors_36_l312_312484

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312484


namespace divisor_count_360_l312_312354

theorem divisor_count_360 : ∃ n : ℕ, n = 24 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 360) → (360 % k = 0 ↔ k ∈ divisors 360) → n = finset.card (divisors 360) :=
by
  -- Definition that n is 24
  let n := 24

  -- Proof that the number of divisors of 360 is equivalent to 24
  have h₁ : n = finset.card (divisors 360), from sorry

  -- Existence proof 
  use n

  -- Prove the conjunction
  split,
  { exact rfl },   -- Proof that n = 24
  {
    intros k hk hdiv,
    -- Proof that the count of the divisors of 360 is 24
    exact h₁,
  }

end divisor_count_360_l312_312354


namespace sum_of_Tp_l312_312877

theorem sum_of_Tp :
  let T (p : ℕ) := 25 * (149 * p - 49)
  ∑ i in Finset.range 20, T (i + 1) = 757750 :=
by
  sorry

end sum_of_Tp_l312_312877


namespace sum_of_24_consecutive_integers_is_square_l312_312216

theorem sum_of_24_consecutive_integers_is_square : ∃ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (24 * (2 * n + 23)) = k * k ∧ k * k = 324 :=
by
  sorry

end sum_of_24_consecutive_integers_is_square_l312_312216


namespace factor_tree_solution_l312_312096

noncomputable def factor_tree := λ X Y Z F G, (Y = 4 * F) → (F = 2) → (Z = 7 * G) → (G = 7) → (X = Y * Z)

theorem factor_tree_solution (X Y Z F G : ℕ) (h1 : Y = 4 * F) (h2 : F = 2) (h3 : Z = 7 * G) (h4 : G = 7) : X = 392 :=
by 
  have hY : Y = 8, 
    exact h1.symm.trans (congr_arg (λ a, 4 * a) h2)
  have hZ : Z = 49, 
    exact h3.symm.trans (congr_arg (λ a, 7 * a) h4)
  have hX: X = Y * Z,
    rw [hY, hZ],
  exact hX.symm.trans (by norm_num)

end factor_tree_solution_l312_312096


namespace circle_eq_standard_form_line_eq_l312_312032

-- Declaration of points
def A := (Real.sqrt 6, 1 : ℝ)
def B := (1, 0 : ℝ)
def C := (3, 2 : ℝ)

-- Prove that the standard equation of circle M is x^2 + (y - 3)^2 = 10
theorem circle_eq_standard_form :
  ∃ (D E F : ℝ), 
    (∀ (x y : ℝ), (x, y) = A → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
    (∀ (x y : ℝ), (x, y) = B → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x, y) = C → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x^2 + (y - 3)^2 = 10 → x^2 + y^2 + D * x + E * y + F = 0)) := sorry

-- Prove that a line l passing through point C with a chord length of 2 has equation x = 3 or 4x - 3y - 6 = 0
theorem line_eq :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), (y - 2 = k * (x - 3) ∨ x = 3) → 
      ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ (a = 4 ∧ b = -3 ∧ c = -6 ∨ a = 1 ∧ b = 0 ∧ c = -3)) := sorry

end circle_eq_standard_form_line_eq_l312_312032


namespace compare_log_fractions_l312_312533

-- Define a, b, c as per the given conditions
def a : ℝ := (Real.log 2) / 2
def b : ℝ := (Real.log 3) / 3
def c : ℝ := (Real.log 5) / 5

-- State the theorem to prove the desired inequality
theorem compare_log_fractions : c < a ∧ a < b := by
  sorry

end compare_log_fractions_l312_312533


namespace divisors_of_36_count_l312_312515

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312515


namespace candy_bars_per_bag_l312_312641

/-
Define the total number of candy bars and the number of bags
-/
def totalCandyBars : ℕ := 75
def numberOfBags : ℚ := 15.0

/-
Prove that the number of candy bars per bag is 5
-/
theorem candy_bars_per_bag : totalCandyBars / numberOfBags = 5 := by
  sorry

end candy_bars_per_bag_l312_312641


namespace num_incorrect_propositions_l312_312804

theorem num_incorrect_propositions : 
  (∃ (A B C D : Type) (AB BC CD DA : A = B = C = D), AB + BC + CD + DA = 0 → false) ∧ 
  (∃ (a b : Type) (a_b : a * b), |a| - |b| = |a + b| ↔ collinear a b → false) ∧ 
  (∃ (a b : Type) (a_b : a * b), collinear a b → parallel a b) ∧ 
  (∃ (O A B C P : Type) (OA OB OC OP : O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ P), OP = x * OA + y * OB + z * OC → coplanar P A B C → false) → 
  3 = 
sorry

end num_incorrect_propositions_l312_312804


namespace number_of_squares_sharing_two_vertices_l312_312600

-- Define the setup for the right triangle and squares
structure Triangle (A B C : Type) :=
(angle_B : ℝ) -- angle at vertex B

-- Define the problem statement
def right_triangle (A B C : Type) [Triangle A B C] :=
Triangle.angle_B = 90

-- Define the condition
axiom ΔABC_is_right (A B C : Type) [Triangle A B C] : right_triangle A B C

-- Define the theorem/proof problem
theorem number_of_squares_sharing_two_vertices (A B C : Type) [Triangle A B C] (h : right_triangle A B C) : 
  ∃ (n : ℕ), n = 2 :=
sorry

end number_of_squares_sharing_two_vertices_l312_312600


namespace problem1_problem2_l312_312929

-- Definitions corresponding to conditions
def p (m : ℝ) : Prop := ∃ x : ℝ, 0 < x ∧ x^2 - 2 * real.exp(1) * real.log x ≤ m
def q (m : ℝ) : Prop := ∀ x : ℝ, 2 ≤ x → (2 * x^2 - m * x + 2) ≤ (2 * (x + 1)^2 - m * (x + 1) + 2)

-- Theorem statements for the proof problems
theorem problem1 (m : ℝ) : ¬ (p m ∨ q m) → false :=
begin
  sorry
end

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → m > 8 ∨ m < 0 :=
begin
  sorry
end

end problem1_problem2_l312_312929


namespace find_x_from_angles_l312_312108

theorem find_x_from_angles (x : ℝ) (h : 3 * x + 7 * x + 4 * x + x = 360) : x = 24 :=
begin
  sorry
end

end find_x_from_angles_l312_312108


namespace seven_times_one_fifth_cubed_l312_312822

theorem seven_times_one_fifth_cubed : 7 * (1 / 5) ^ 3 = 7 / 125 := 
by 
  sorry

end seven_times_one_fifth_cubed_l312_312822


namespace smallest_perimeter_even_integer_triangl_l312_312257

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l312_312257


namespace three_digit_numbers_with_4_and_5_correct_l312_312531

def count_three_digit_numbers_with_4_and_5 : ℕ :=
  48

theorem three_digit_numbers_with_4_and_5_correct :
  count_three_digit_numbers_with_4_and_5 = 48 :=
by
  sorry -- proof goes here

end three_digit_numbers_with_4_and_5_correct_l312_312531


namespace count_divisors_36_l312_312480

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312480


namespace probability_age_between_30_and_40_l312_312972

-- Assume total number of people in the group is 200
def total_people : ℕ := 200

-- Assume 80 people have an age of more than 40 years
def people_age_more_than_40 : ℕ := 80

-- Assume 70 people have an age between 30 and 40 years
def people_age_between_30_and_40 : ℕ := 70

-- Assume 30 people have an age between 20 and 30 years
def people_age_between_20_and_30 : ℕ := 30

-- Assume 20 people have an age of less than 20 years
def people_age_less_than_20 : ℕ := 20

-- The proof problem statement
theorem probability_age_between_30_and_40 :
  (people_age_between_30_and_40 : ℚ) / (total_people : ℚ) = 7 / 20 :=
by
  sorry

end probability_age_between_30_and_40_l312_312972


namespace ratio_of_amounts_l312_312632

theorem ratio_of_amounts
    (initial_cents : ℕ)
    (given_to_peter_cents : ℕ)
    (remaining_nickels : ℕ)
    (nickel_value : ℕ := 5)
    (nickels_initial := initial_cents / nickel_value)
    (nickels_to_peter := given_to_peter_cents / nickel_value)
    (nickels_remaining := nickels_initial - nickels_to_peter)
    (nickels_given_to_randi := nickels_remaining - remaining_nickels)
    (cents_to_randi := nickels_given_to_randi * nickel_value)
    (cents_initial : initial_cents = 95)
    (cents_peter : given_to_peter_cents = 25)
    (nickels_left : remaining_nickels = 4)
    :
    (cents_to_randi / given_to_peter_cents) = 2 :=
by
  sorry

end ratio_of_amounts_l312_312632


namespace max_sphere_volume_in_prism_l312_312552

theorem max_sphere_volume_in_prism
  (AB BC : ℝ) (AA₁ : ℝ)
  (h_AB_BC : AB = 6) (h_BC_AB : BC = 8) (h_AA₁ : AA₁ = 3) :
  ∃ V : ℝ, V = (9 * real.pi) / 2 :=
by {
  sorry
}

end max_sphere_volume_in_prism_l312_312552


namespace diana_wins_probability_l312_312319

theorem diana_wins_probability :
  let a := (1 / 32 : ℝ)
  let r := (1 / 32 : ℝ)
  let geom_series_sum := a / (1 - r) in
  geom_series_sum = 1 / 31 :=
by
  let a := (1 / 32 : ℝ)
  let r := (1 / 32 : ℝ)
  have sum_geom_series : geom_series_sum = a / (1 - r),
  { unfold geom_series_sum a r,
    simp [div_eq_mul_inv, mul_comm],
    sorry }

#exit

end diana_wins_probability_l312_312319


namespace expression_evaluation_l312_312819

theorem expression_evaluation :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1 / 3))^2 = 5 :=
by
  sorry

end expression_evaluation_l312_312819


namespace num_divisors_of_36_l312_312488

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312488


namespace prime_power_lcm_condition_l312_312611

open Nat

theorem prime_power_lcm_condition (n : ℕ) (h : n ≥ 2) (d : ℕ → Prop) :
  (∃ p k : ℕ, prime p ∧ k ≥ 1 ∧ n = p^k) ↔ lcm (filter (λ m, m < n) (range (n + 1))) ≠ n :=
by
  sorry

end prime_power_lcm_condition_l312_312611


namespace find_S7_l312_312997

def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem find_S7 
  (a : ℕ → ℝ)
  (h1 : a 2 = 3)
  (h2 : a 6 = 11)
  (h_seq : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)) :
  arithmetic_sequence_sum a 7 = 49 :=
by
  sorry

end find_S7_l312_312997


namespace smallest_n_for_A_l312_312281

def A (n : ℕ) : ℕ := 
  let terms := (List.range (n+1)).map (fun k => 2^(10*k))
  let concatenated_str := terms.foldl (λ acc num => acc ++ toString num) ""
  concatenated_str.toNat

theorem smallest_n_for_A : ∃ n, n ≥ 3 ∧ A n % 2^170 = 2^(10*n) % 2^170 ∧ (∀ m, m ≥ 3 ∧ A m % 2^170 = 2^(10*m) % 2^170 → n <= m) :=
  ⟨14, 
  by
    simp
    sorry⟩

end smallest_n_for_A_l312_312281


namespace max_g_value_range_of_a_l312_312918

-- Define the function f: ℝ → ℝ as ln x
def f (x : ℝ) : ℝ := Real.log x

-- Define the function g: ℝ → ℝ as f(x+1) - x
def g (x : ℝ) : ℝ := f (x + 1) - x

-- Theorem 1: Proving that the maximum value of g(x) is 0 for x > -1
theorem max_g_value : ∀ x > -1, g x ≤ g 0 := 
by sorry

-- Theorem 2: Proving the range of a such that for any x > 0, f(x) ≤ ax ≤ x^2 + 1
theorem range_of_a : ∀ a, (∀ x > 0, f(x) ≤ a * x ∧ a * x ≤ x^2 + 1) ↔ (1 / Real.exp 1 ≤ a ∧ a ≤ 2) := 
by sorry

end max_g_value_range_of_a_l312_312918


namespace rowing_speed_is_1_l312_312773

def man_rows_750m_in_675s_against_stream : Prop := ∃ V1 : ℝ, V1 = 750 / 675
def man_returns_in_450s : Prop := ∃ V2 : ℝ, V2 = 750 / 450
def rowing_speed_in_still_water (V1 V2 : ℝ) : ℝ := (V1 + V2) / 2

theorem rowing_speed_is_1.389 :
  man_rows_750m_in_675s_against_stream ∧ man_returns_in_450s →
  ∃ V : ℝ, V = 1.389 := 
by
  sorry

end rowing_speed_is_1_l312_312773


namespace inequality_range_a_l312_312062

theorem inequality_range_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) : 
  (sqrt x + sqrt y ≤ a * sqrt (x + y)) ↔ a ≥ sqrt 2 :=
sorry

end inequality_range_a_l312_312062


namespace complement_of_union_is_neg3_l312_312066

open Set

variable (U A B : Set Int)

def complement_union (U A B : Set Int) : Set Int :=
  U \ (A ∪ B)

theorem complement_of_union_is_neg3 (U A B : Set Int) (hU : U = {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6})
  (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {-2, 3, 4, 5, 6}) :
  complement_union U A B = {-3} :=
by
  sorry

end complement_of_union_is_neg3_l312_312066


namespace divide_friends_to_teams_l312_312075

def num_ways_to_divide_friends(N : ℕ, T : ℕ) : ℕ :=
  T ^ N

-- Given conditions:
def friends : ℕ := 8
def teams : ℕ := 4

-- Define the correct answer
def expected_ways : ℕ := 65536

theorem divide_friends_to_teams :
  num_ways_to_divide_friends friends teams = expected_ways := by
  sorry

end divide_friends_to_teams_l312_312075


namespace sara_frosting_total_l312_312634

def cakes_baked_each_day : List Nat := [7, 12, 8, 10, 15]
def cakes_eaten_by_Carol : List Nat := [4, 6, 3, 2, 3]
def cans_per_cake_each_day : List Nat := [2, 3, 4, 3, 2]

def total_frosting_cans_needed : Nat :=
  let remaining_cakes := List.zipWith (· - ·) cakes_baked_each_day cakes_eaten_by_Carol
  let required_cans := List.zipWith (· * ·) remaining_cakes cans_per_cake_each_day
  required_cans.foldl (· + ·) 0

theorem sara_frosting_total : total_frosting_cans_needed = 92 := by
  sorry

end sara_frosting_total_l312_312634


namespace largest_value_fraction_l312_312953

theorem largest_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  ∃ z, z = (x^2 / (2 * y)) ∧ z ≤ 5 :=
by
  sorry

end largest_value_fraction_l312_312953


namespace DE_bisects_angle_ADF_l312_312740

theorem DE_bisects_angle_ADF (A B C D E F : Type) 
  [Points_on_triangle_sides D E F A B C]
  (h1 : BF = 2 * CF) 
  (h2 : CE = 2 * AE) 
  (h3 : ∠ DEF = 90) : (bisects_angle E D F) ∧ (bisects_angle D E A) :=
sorry

end DE_bisects_angle_ADF_l312_312740


namespace num_divisors_36_l312_312501

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312501


namespace geometric_sequence_sum_S6_l312_312431

theorem geometric_sequence_sum_S6 (S : ℕ → ℝ) (S_2_eq_4 : S 2 = 4) (S_4_eq_16 : S 4 = 16) :
  S 6 = 52 :=
sorry

end geometric_sequence_sum_S6_l312_312431


namespace find_a_n_find_min_m_l312_312407

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 1 else 3^(n-1)

noncomputable def sequence_b (n : ℕ) : ℕ → ℝ
| 0 := 0
| k+1 := 1 / ((1 + real.log (sequence_a (k+1)) / real.log 3) * (3 + real.log (sequence_a (k+1)) / real.log 3))

noncomputable def T_n (n : ℕ) : ℝ :=
∑ i in finset.range n, sequence_b i

theorem find_a_n (n : ℕ) : sequence_a n = 3^(n-1) :=
sorry

theorem find_min_m (m : ℝ) (h : ∀ n : ℕ, T_n n < m) : m = (3 / 4) :=
sorry

end find_a_n_find_min_m_l312_312407


namespace drunk_drivers_count_l312_312969

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end drunk_drivers_count_l312_312969


namespace b_minus_4a_eq_4_l312_312951

theorem b_minus_4a_eq_4
  (a : ℤ) (b : ℤ)
  (h1: a = -1) (h2: b = 0) :
  b - 4 * a = 4 :=
by
  rw [h1, h2]
  simp
  exact rfl

end b_minus_4a_eq_4_l312_312951


namespace solution_l312_312030

noncomputable def problem_statement (p : ℝ) (hp : p > 0) : Prop :=
  let parabola := λ x y, y^2 = 2 * p * x in
  let line := λ x y, y = (Real.sqrt 3) * (x - p / 2) in
  let intercepts := intersect parabola line in -- assuming a function to find intercepts
  let A := intercepts.head in
  let B := intercepts.tail.head in -- assuming A is in the 1st quadrant, B in the 4th
  let P := (p / 2, 0) in
  let AB := dist A B in
  let AP := dist A P in
  AB / AP = 7 / 12

theorem solution (p : ℝ) (hp : p > 0) : problem_statement p hp :=
sorry

end solution_l312_312030


namespace estimate_sqrt_expression_l312_312355

theorem estimate_sqrt_expression :
  5 < 3 * Real.sqrt 5 - 1 ∧ 3 * Real.sqrt 5 - 1 < 6 :=
by
  sorry

end estimate_sqrt_expression_l312_312355


namespace proof_problem_l312_312983

-- Define the conditions
def C1 (θ : ℝ) := 4 * Real.cos θ
def C2 (θ : ℝ) := 2 * Real.sin θ

-- Define the main statement 
theorem proof_problem (α : ℝ) (hα : α ∈ Set.Icc (Real.pi / 6) (Real.pi / 3)) :
  (∀ θ, C1 θ = 4 * cos θ → (ρ^2 = 4 * ρ * cos θ ↔ (x-2)^2 + y^2 = 4)) ∧ 
  (∃ (OA OB : ℝ), OA = 4 * cos α ∧ OB = 2 * cos α ∧ 
  1/2 * OA * OB = 1 + cos (2 * α)) ∧
  (1 + cos (2 * α) ∈ Set.Icc (1 / 2) (3 / 2)) :=
by
  sorry

end proof_problem_l312_312983


namespace smallest_k_for_exponential_inequality_l312_312279

theorem smallest_k_for_exponential_inequality :
  (∃ k : ℕ, 64^k > 4^17 ∧ ∀ n : ℕ, n < k → 64^n ≤ 4^17) :=
by
  let k := 6
  use k
  split
  -- Proof of 64^k > 4^17
  sorry
  -- Proof of minimality
  sorry

end smallest_k_for_exponential_inequality_l312_312279


namespace artifacts_per_wing_l312_312780

theorem artifacts_per_wing (total_wings : ℕ) (painting_wings : ℕ) 
    (large_paintings : ℕ) (small_paintings_per_wing : ℕ) 
    (artifact_ratio : ℕ) 
    (h_total_wings : total_wings = 8) 
    (h_painting_wings : painting_wings = 3) 
    (h_large_paintings : large_paintings = 1) 
    (h_small_paintings_per_wing : small_paintings_per_wing = 12) 
    (h_artifact_ratio : artifact_ratio = 4) :
    let total_paintings := large_paintings + small_paintings_per_wing * 2 in
    let total_artifacts := artifact_ratio * total_paintings in
    let artifact_wings := total_wings - painting_wings in
    total_artifacts / artifact_wings = 20 :=
by
    sorry

end artifacts_per_wing_l312_312780


namespace length_of_plot_l312_312735

theorem length_of_plot (breadth length : ℕ) 
                       (h1 : length = breadth + 26)
                       (fencing_cost total_cost : ℝ)
                       (h2 : fencing_cost = 26.50)
                       (h3 : total_cost = 5300)
                       (perimeter : ℝ) 
                       (h4 : perimeter = 2 * (breadth + length)) 
                       (h5 : total_cost = perimeter * fencing_cost) :
                       length = 63 :=
by
  sorry

end length_of_plot_l312_312735


namespace slope_of_tangent_l312_312247

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

def C : Point := ⟨2, 3⟩
def P : Point := ⟨7, 8⟩

-- Function to calculate the slope between two points
def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

-- Statement of the problem
theorem slope_of_tangent (C P : Point) (hC : C = ⟨2, 3⟩) (hP : P = ⟨7, 8⟩) :
  slope C P = 1 →
  -1 / slope C P = -1 := by
  sorry

end slope_of_tangent_l312_312247


namespace find_side_length_a_l312_312119

variable {a : ℝ} {b : ℝ} {c : ℝ} {A : ℝ} {B : ℝ} {C : ℝ}

theorem find_side_length_a (h1 : b = 7) (h2 : c = 6) (h3 : Real.cos (B - C) = 31/32) : a = Real.sqrt(299) / 2 := sorry

end find_side_length_a_l312_312119


namespace integral_f_l312_312152

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2
  else if x < Real.exp 1 then 1/x
  else 0

theorem integral_f : ∫ x in 0..Real.exp 1, f x = 4 / 3 :=
by
  sorry

end integral_f_l312_312152


namespace find_values_exists_l312_312881

def verify_coefficients (a b c d : ℤ) : Prop :=
  let poly := λ x : ℚ, (x ^ 2 + (a : ℚ) * x + (b : ℚ)) * (x ^ 2 + (c : ℚ) * x + (d : ℚ))
  (poly 1 = 1^4 + 2*1^3 - 3*1^2 + 12*1 - 8) ∧
  (poly 2 = 2^4 + 2*2^3 - 3*2^2 + 12*2 - 8)

theorem find_values_exists (a b c d : ℤ)
  (h1 : a + c = 2)
  (h2 : a * c + b + d = -3)
  (h3 : a * d + b * c = 12)
  (h4 : b * d = -8) : 
  a + b + c + d = sorry := by
    sorry

end find_values_exists_l312_312881


namespace smallest_perimeter_consecutive_even_triangle_l312_312253

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l312_312253


namespace number_of_smaller_pipes_needed_l312_312204

def radius (diameter : ℝ) : ℝ := diameter / 2

def cross_sectional_area (r : ℝ) : ℝ := π * r^2

def num_smaller_pipes (d_large d_small : ℝ) : ℝ :=
  (cross_sectional_area (radius d_large)) / (cross_sectional_area (radius d_small))

theorem number_of_smaller_pipes_needed :
  num_smaller_pipes 8 2 = 16 :=
by
  sorry

end number_of_smaller_pipes_needed_l312_312204


namespace prob_N6_mod_7_eq_1_l312_312807

theorem prob_N6_mod_7_eq_1 : 
  let N_max := 2021 in
  let favorable := 6 in
  let total_classes := 7 in
  (favorable / total_classes : ℚ) = 6 / 7 :=
sorry

end prob_N6_mod_7_eq_1_l312_312807


namespace max_minus_min_eq_32_l312_312048

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_minus_min_eq_32 : 
  let M := max (f (-3)) (max (f 3) (max (f (-2)) (f 2)))
  let m := min (f (-3)) (min (f 3) (min (f (-2)) (f 2)))
  M - m = 32 :=
by
  sorry

end max_minus_min_eq_32_l312_312048


namespace ratio_of_diagonal_lengths_of_squares_l312_312688

noncomputable def diagonal_length (side_length : ℝ) : ℝ := side_length * real.sqrt 2

theorem ratio_of_diagonal_lengths_of_squares (a b : ℝ) (h : b^2 = 4 * a^2) : 
  diagonal_length b / diagonal_length a = 2 := 
sorry

end ratio_of_diagonal_lengths_of_squares_l312_312688


namespace heptagon_diagonals_l312_312465

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end heptagon_diagonals_l312_312465


namespace base_angle_isosceles_triangle_l312_312037

theorem base_angle_isosceles_triangle (α : ℝ) (hα : α = 108) (isosceles : ∀ (a b c : ℝ), a = b ∨ b = c ∨ c = a) : α = 108 →
  α + β + β = 180 → β = 36 :=
by
  sorry

end base_angle_isosceles_triangle_l312_312037


namespace problems_per_hour_l312_312880

def num_math_problems : ℝ := 17.0
def num_spelling_problems : ℝ := 15.0
def total_hours : ℝ := 4.0

theorem problems_per_hour :
  (num_math_problems + num_spelling_problems) / total_hours = 8.0 := by
  sorry

end problems_per_hour_l312_312880


namespace inverse_h_l312_312151

def f (x : ℝ) : ℝ := 5 * x - 7
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h : (∀ x : ℝ, h (15 * x + 3) = x) :=
by
  -- Proof would go here
  sorry

end inverse_h_l312_312151


namespace total_ticket_cost_l312_312586

theorem total_ticket_cost :
  let price_adult := 12
  let price_child := 10
  let price_senior := 8
  let price_student := 9
  let num_adults := 4
  let num_children := 3
  let num_seniors := 2
  let num_students := 1
  let total_cost := num_adults * price_adult + num_children * price_child + num_seniors * price_senior + num_students * price_student
  total_cost = 103 :=
by
  let price_adult := 12
  let price_child := 10
  let price_senior := 8
  let price_student := 9
  let num_adults := 4
  let num_children := 3
  let num_seniors := 2
  let num_students := 1
  let total_cost := num_adults * price_adult + num_children * price_child + num_seniors * price_senior + num_students * price_student
  show total_cost = 103 from sorry

end total_ticket_cost_l312_312586


namespace probability_divisible_by_13_l312_312085

noncomputable def digit_sum_eq_44 (n : ℕ) : Prop :=
  n.digits.sum = 44 ∧ 10000 ≤ n ∧ n < 100000

noncomputable def count_five_digit_numbers : ℕ :=
  ((list.range 90000).filter (λ n, digit_sum_eq_44 (10000 + n))).length

noncomputable def count_divisible_by_13 : ℕ :=
  ((list.range 90000).filter (λ n, digit_sum_eq_44 (10000 + n) ∧ (10000 + n) % 13 = 0)).length

theorem probability_divisible_by_13 :
  (count_divisible_by_13.toFloat / count_five_digit_numbers.toFloat) = (2 / 25) :=
sorry

end probability_divisible_by_13_l312_312085


namespace range_of_a_l312_312055

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (3 * x + a / x - 2)

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) := 
sorry

end range_of_a_l312_312055


namespace common_intersection_point_l312_312239

variable {P : Type} [EuclideanGeometry P]

-- Definitions of points and tetrahedron vertices
variables (A B C D : P)

-- Definitions of the plane and projections
variable (plane_ABC : Plane P)
variable (proj_D_onto_ABC : P)
variable [hproj : IsOrthogonalProjectionToPlane D proj_D_onto_ABC plane_ABC]

-- Definitions of spheres using edges as diameters
variable (sphere_AD : Sphere P)
variable (sphere_BD : Sphere P)
variable (sphere_CD : Sphere P)
variable [hs_AD : SphereHasDiameter sphere_AD A D]
variable [hs_BD : SphereHasDiameter sphere_BD B D]
variable [hs_CD : SphereHasDiameter sphere_CD C D]

-- Definitions of circles
variable (circle_AD_inter_plane_ABC : Circle P)
variable (circle_BD_inter_plane_ABC : Circle P)
variable (circle_CD_inter_plane_ABC : Circle P)
variable [hc_AD_plane_ABC : IntersectionIsCircle sphere_AD plane_ABC circle_AD_inter_plane_ABC]
variable [hc_BD_plane_ABC : IntersectionIsCircle sphere_BD plane_ABC circle_BD_inter_plane_ABC]
variable [hc_CD_plane_ABC : IntersectionIsCircle sphere_CD plane_ABC circle_CD_inter_plane_ABC]

-- Proposition to prove there is a common intersection point
theorem common_intersection_point (A B C D : P) (plane_ABC : Plane P)
  (proj_D_onto_ABC : P) [IsOrthogonalProjectionToPlane D proj_D_onto_ABC plane_ABC]
  (sphere_AD sphere_BD sphere_CD : Sphere P)
  [SphereHasDiameter sphere_AD A D]
  [SphereHasDiameter sphere_BD B D]
  [SphereHasDiameter sphere_CD C D]
  (circle_AD_inter_plane_ABC circle_BD_inter_plane_ABC circle_CD_inter_plane_ABC : Circle P)
  [IntersectionIsCircle sphere_AD plane_ABC circle_AD_inter_plane_ABC]
  [IntersectionIsCircle sphere_BD plane_ABC circle_BD_inter_plane_ABC]
  [IntersectionIsCircle sphere_CD plane_ABC circle_CD_inter_plane_ABC] :
  proj_D_onto_ABC ∈ circle_AD_inter_plane_ABC ∧ 
  proj_D_onto_ABC ∈ circle_BD_inter_plane_ABC ∧ 
  proj_D_onto_ABC ∈ circle_CD_inter_plane_ABC :=
sorry

end common_intersection_point_l312_312239


namespace solution_set_of_inequality_l312_312441

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 2^(-x) - 1 else -x^2 + x

def solution_set (x : ℝ) : Prop := f(f(x)) ≤ 3

theorem solution_set_of_inequality : ∀ x, solution_set x ↔ x ≤ 2 := 
by
  sorry

end solution_set_of_inequality_l312_312441


namespace complex_conjugate_square_l312_312418

theorem complex_conjugate_square (a b : ℝ) (i : ℂ) (h1 : i = complex.I) (h2 : a + i = conj (2 - b * i)) :
  (a + b * i) ^ 2 = 3 + 4 * i :=
by
  sorry

end complex_conjugate_square_l312_312418


namespace arithmetic_sequence_exists_l312_312630

theorem arithmetic_sequence_exists (k : ℕ) (hk : k > 0) :
  ∃ (a b : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ k → (nat.coprime (a i) (b i)) ∧ 
           a i > 0 ∧ b i > 0 ∧ 
           (∀ j, j ≠ i → a i ≠ a j ∧ b i ≠ b j) ∧
           (∃ d, ∀ i, 1 ≤ i ∧ i ≤ k → (a i : ℚ) / (b i : ℚ) = ((a 1) : ℚ) / (b 1 : ℚ) + d * (i - 1))) :=
begin
  sorry
end

end arithmetic_sequence_exists_l312_312630


namespace time_to_traverse_nth_mile_l312_312790

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 3) : ∃ t : ℕ, t = (n - 2)^2 :=
by
  -- Given:
  -- Speed varies inversely as the square of the number of miles already traveled.
  -- Speed is constant for each mile.
  -- The third mile is traversed in 4 hours.
  -- Show that:
  -- The time to traverse the nth mile is (n - 2)^2 hours.
  sorry

end time_to_traverse_nth_mile_l312_312790


namespace johns_last_segment_speed_l312_312582

variable (total_distance : ℕ) (total_time_in_minutes : ℕ)
variable (first_segment_speed second_segment_speed : ℕ)

theorem johns_last_segment_speed :
  total_distance = 120 → total_time_in_minutes = 90 →
  first_segment_speed = 50 → second_segment_speed = 70 →
  (120 - 50 * 1/2 - 70 * 1/2) * 3 = 120 :=
by
  intros hd ht hs1 hs2
  rw [hd, ht, hs1, hs2]
  sorry

end johns_last_segment_speed_l312_312582


namespace find_z_value_l312_312436

-- We will define the variables and the given condition
variables {x y z : ℝ}

-- Translate the given condition into Lean
def given_condition (x y z : ℝ) : Prop := (1 / x^2 - 1 / y^2) = (1 / z)

-- State the theorem to prove
theorem find_z_value (x y z : ℝ) (h : given_condition x y z) : 
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end find_z_value_l312_312436


namespace correct_inequality_l312_312888

noncomputable section

open Real

variables (t : ℝ) (x y z : ℝ)
variables (h1 : t > 1) (hx : x = log t / log 2) (hy : y = log t / log 3) (hz : z = log t / log 5)

theorem correct_inequality : 3 * y < 2 * x ∧ 2 * x < 5 * z :=
by
  have h : log t > 0 := log_pos h1
  have h2 : 3 * (log t / log 3) < 2 * (log t / log 2) := 
    by sorry
  have h3 : 2 * (log t / log 2) < 5 * (log t / log 5) := 
    by sorry
  exact ⟨h2, h3⟩

end correct_inequality_l312_312888


namespace quadratic_ratio_l312_312212

/-- Given a quadratic polynomial, represent it in the form of a completed square
    and prove the ratio of the constants involved. -/
theorem quadratic_ratio :
  let d := 1011.5
  let e := 2023 - (d * d)
  (e / d) = -1009.75 := by
  let d := 1011.5
  let e := 2023 - (d * d)
  calc
    (e / d) = (2023 - (d * d)) / d : by sorry
           ... = -1009.75 : by sorry

end quadratic_ratio_l312_312212


namespace triangles_same_area_l312_312117

/-- Each key on the keyboard is represented by a congruent square with a side length of 1 unit.
    The points Q, A, Z, E, S are at the centers of specific keys on the keyboard. -/
variables {Q A Z E S : Point}
def key_side_length : ℝ := 1

/-- Define the coordinates of points Q, A, Z, E, S based on their positions on the keyboard grid. -/
def point_coords : Point → ℝ × ℝ
| Q := (2, 4)
| A := (1, 3)
| Z := (0, 2)
| E := (2, 0)
| S := (1, 1)

/-- Given point coordinates, calculate the area of a triangle with vertices p1, p2, p3. -/
def triangle_area (p1 p2 p3 : Point) : ℝ :=
  let (x1, y1) := point_coords p1
  let (x2, y2) := point_coords p2
  let (x3, y3) := point_coords p3
  in 1 / 2 * ((x1 * y2 + x2 * y3 + x3 * y1) - (y1 * x2 + y2 * x3 + y3 * x1))

/-- Theorem to prove the areas of triangles QAZ and ESZ are equal. -/
theorem triangles_same_area : triangle_area Q A Z = triangle_area E S Z :=
by
  sorry

end triangles_same_area_l312_312117


namespace two_n_plus_m_is_36_l312_312679

theorem two_n_plus_m_is_36 (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 :=
sorry

end two_n_plus_m_is_36_l312_312679


namespace attendance_percentage_l312_312207

theorem attendance_percentage (X : ℝ) :
  let projected_attendance := 1.25 * X,
      actual_attendance := 0.80 * X
  in (actual_attendance / projected_attendance) * 100 = 64 :=
by
  sorry

end attendance_percentage_l312_312207


namespace probability_units_digit_8_l312_312390

-- Definitions of the conditions from the problem
def is_in_range (n : ℕ) : Prop := n ∈ (Finset.range 51).erase 0

def units_digit_cycles (base : ℕ) : List ℕ :=
  if base = 3 then [1, 3, 9, 7] else
  if base = 7 then [7, 9, 3, 1] else []

-- Define the main theorem statement
theorem probability_units_digit_8 :
  ∀ (a b : ℕ), is_in_range a →
               is_in_range b →
               (units_digit_cycles 3) (a % 4) +
               (units_digit_cycles 7) (b % 4) % 10 = 8 →
               (∑ x in (Finset.range 51).erase 0, ∑ y in (Finset.range 51).erase 0, 1) = 169 →
               ∃ p : ℚ, p = 3 / 169 :=
begin
  intros,
  sorry,
end

end probability_units_digit_8_l312_312390


namespace max_n_for_factorable_quadratic_l312_312863

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l312_312863


namespace trajectory_is_ellipse_find_line_equation_l312_312980

-- Statement 1
theorem trajectory_is_ellipse :
  let M := (fun x y : ℝ => (x + 1)^2 + y^2 = 49 / 4)
  let N := (fun x y : ℝ => (x - 1)^2 + y^2 = 1 / 4)
  let moving_circle r (x y : ℝ) := M x y && N x y
  (internal_tangent_to M moving_circle) ∧ (external_tangent_to N moving_circle) →
  ellipse_with_foci_and_major_axis M N 2 4 :=
sorry

-- Statement 2
theorem find_line_equation (A B : ℝ × ℝ) :
  let curve_P := (fun x y : ℝ => x^2 / 4 + y^2 / 3 = 1)
  let l := (fun k : ℝ => y = k * (x - 1))
  line_through l (1, 0) ∧ intersects_curve curve_P l A B ∧ (vector_dot A B = -2) →
  l = (fun k : ℝ => y = ± sqrt(2) * (x - 1)) :=
sorry

end trajectory_is_ellipse_find_line_equation_l312_312980


namespace inverse_variation_l312_312746

theorem inverse_variation (k : ℝ) (h₁ : 5 * 4 = k / (2^3)) (h₂ : x = 4) : y = 1 / 2 :=
by
  have k_val : k = 160 :=
    calc
      k = 20 * 8 := by sorry
  have eq₁ : 5 * y = 160 / (x^3) := by sorry
  have eq₂ : 5 * y = 5 / 2 := by
    rw [h₂, eq₁]
    apply calc
      160 / (4^3) = 160 / 64 := by sorry
      ... = 5 / 2 := by sorry
  exact calc
    y = (5 / 2) / 5 := by sorry
    ... = 1 / 2 := by sorry

end inverse_variation_l312_312746


namespace teal_more_blue_l312_312294

def numSurveyed : ℕ := 150
def numGreen : ℕ := 90
def numBlue : ℕ := 50
def numBoth : ℕ := 40
def numNeither : ℕ := 20

theorem teal_more_blue : 40 + (numSurveyed - (numBoth + (numGreen - numBoth) + numNeither)) = 80 :=
by
  -- Here we simplify numerically until we get the required answer
  -- start with calculating the total accounted and remaining
  sorry

end teal_more_blue_l312_312294


namespace part1_single_solution_part2_range_values_l312_312548

-- Definitions based on the given conditions
def in_triangle_ABC (a b : ℝ) (B : ℝ) (A C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 0 < B ∧ B < 180 ∧ 
  A > 0 ∧ A < 180 ∧ C > 0 ∧ C < 180 ∧ 
  (A + B + C = 180) ∧ (a * Math.sin B = b * Math.sin A)

-- (1) Define the conditions for the first part of the question
def conditions_part1 : Prop := in_triangle_ABC 2 (Real.sqrt 2) 45 90 45 

-- Prove there is only one solution
theorem part1_single_solution : conditions_part1 → ∃! (A : ℝ), A = 90 :=
  by sorry

-- (2) Define the conditions for the second part of the question
def conditions_part2 (b : ℝ) : Prop :=
  in_triangle_ABC 2 b 45 60 75 ∧ 
  b ∈ Set.Ioo (Real.sqrt 2) 2

-- Prove the range of values for b when there are two solutions
theorem part2_range_values (b : ℝ) : (∃ A C, conditions_part2 b) → b ∈ Set.Ioo (Real.sqrt 2) 2 :=
  by sorry

end part1_single_solution_part2_range_values_l312_312548


namespace relationship_between_m_and_n_l312_312903

theorem relationship_between_m_and_n
  (a : ℝ) (m n : ℝ)
  (h_a : a = (Real.sqrt 5 - Real.sqrt 2) / 2)
  (h_f : ∀ x : ℝ, f x = a^x)
  (h_ineq : f m > f n) :
  m < n :=
sorry

end relationship_between_m_and_n_l312_312903


namespace artifacts_per_wing_l312_312781

theorem artifacts_per_wing (total_wings : ℕ) (painting_wings : ℕ) 
    (large_paintings : ℕ) (small_paintings_per_wing : ℕ) 
    (artifact_ratio : ℕ) 
    (h_total_wings : total_wings = 8) 
    (h_painting_wings : painting_wings = 3) 
    (h_large_paintings : large_paintings = 1) 
    (h_small_paintings_per_wing : small_paintings_per_wing = 12) 
    (h_artifact_ratio : artifact_ratio = 4) :
    let total_paintings := large_paintings + small_paintings_per_wing * 2 in
    let total_artifacts := artifact_ratio * total_paintings in
    let artifact_wings := total_wings - painting_wings in
    total_artifacts / artifact_wings = 20 :=
by
    sorry

end artifacts_per_wing_l312_312781


namespace one_letter_goal_l312_312228

def is_A_or_B : Char → Bool
| 'A' => true
| 'B' => true
| _ => false

-- Length 41 list of 'A' and 'B'
def valid_initial_list (l : List Char) : Prop :=
  (l.length = 41) ∧ (∀ c ∈ l, is_A_or_B c)

-- Transformations
def transform1 (l : List Char) : List Char := 
  if l.containsSublist ['A', 'B', 'A']
    then l.replaceSublist ['A', 'B', 'A'] ['B']
    else l.replaceSublist ['B'] ['A', 'B', 'A']

def transform2 (l : List Char) : List Char :=
  if l.containsSublist ['V', 'A', 'V']
    then l.replaceSublist ['V', 'A', 'V'] ['A']
    else l.replaceSublist ['A'] ['V', 'A', 'V']

-- Final goal
def can_reduce_to_one (l : List Char) : Prop :=
  ∃ f : ℕ → List Char, f 0 = l ∧ (∀ n, f (n + 1) = transform1 (transform2 (f n)) ∨ f (n + 1) = transform2 (transform1 (f n))) ∧ (∃ n, (f n).length = 1)

-- Statement to prove
theorem one_letter_goal (l : List Char) (h : valid_initial_list l) : can_reduce_to_one l :=
by sorry

end one_letter_goal_l312_312228


namespace numberOfDogs_l312_312763

variable (cupWeight : ℝ) (cupsPerMeal : ℕ) (mealsPerDay : ℕ) (daysPerMonth : ℕ) (bagsPurchased : ℕ)
variable (weightPerBag : ℝ)

def monthlyFoodPerDog : ℝ :=
  let dailyCups := cupsPerMeal * mealsPerDay
  let dailyPounds := dailyCups * cupWeight
  dailyPounds * daysPerMonth

def totalMonthlyFoodPurchased : ℝ :=
  bagsPurchased * weightPerBag

theorem numberOfDogs (h1 : cupWeight = 1/4) (h2 : cupsPerMeal = 6) (h3 : mealsPerDay = 2)
                    (h4 : daysPerMonth = 30) (h5 : bagsPurchased = 9) (h6 : weightPerBag = 20):
  totalMonthlyFoodPurchased = monthlyFoodPerDog * 2 :=
by
  sorry

end numberOfDogs_l312_312763


namespace sum_of_24_consecutive_integers_is_square_l312_312217

theorem sum_of_24_consecutive_integers_is_square : ∃ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (24 * (2 * n + 23)) = k * k ∧ k * k = 324 :=
by
  sorry

end sum_of_24_consecutive_integers_is_square_l312_312217


namespace juan_debt_exceeds_triple_l312_312991

theorem juan_debt_exceeds_triple :
  ∃ t : ℤ, t ≥ 0 ∧ (1 + 0.05)^t > 3 ∧ ∀ u : ℤ, u ≥ 0 → (u < t → (1 + 0.05)^u ≤ 3) :=
by
  sorry

end juan_debt_exceeds_triple_l312_312991


namespace trapezoid_base_ratio_l312_312655

theorem trapezoid_base_ratio
  (a b h : ℝ)  -- lengths of the bases and height
  (ha_gt_hb : a > b)
  (trapezoid_area : ℝ) 
  (quad_area : ℝ) 
  (h1 : trapezoid_area = (1/2) * (a + b) * h) 
  (h2 : quad_area = (1/2) * (a - b) * h / 4)
  (h3 : quad_area = trapezoid_area / 4) :
  a / b = 3 :=
by {
  sorry,
}

end trapezoid_base_ratio_l312_312655


namespace find_x_l312_312266

theorem find_x (a y x : ℤ) (h1 : y = 3) (h2 : a * y + x = 10) (h3 : a = 3) : x = 1 :=
by 
  sorry

end find_x_l312_312266


namespace value_of_b_l312_312709

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
sorry

end value_of_b_l312_312709


namespace polygon_similarity_nesting_equiv_l312_312707

-- Define the nesting relationship
def nesting (p q : Type) [Polygon p] [Polygon q] : Prop :=
  ∃ (transformation : Transformation), can_fit_inside p q transformation

-- Define the similarity relationship
def similar (p q : Type) [Polygon p] [Polygon q] : Prop :=
  ∃ (homothety : Homothety), homothety q = p

-- Main theorem statement
theorem polygon_similarity_nesting_equiv (p q : Type) [Polygon p] [Polygon q] :
  (∃ r : Type, [Polygon r] ∧ (similar r q) ∧ ¬nesting r p) ↔ ¬similar p q :=
by 
  sorry

end polygon_similarity_nesting_equiv_l312_312707


namespace lines_parallel_to_same_plane_positional_relationship_l312_312090

-- Definitions for our problem
variable (l1 l2 : Type) (P : Type)
variable [IsParallel l1 P]
variable [IsParallel l2 P]

theorem lines_parallel_to_same_plane_positional_relationship :
  (IsParallel l1 l2 ∨ Intersect l1 l2 ∨ Skew l1 l2) := 
by
  sorry

end lines_parallel_to_same_plane_positional_relationship_l312_312090


namespace ring_stack_distance_l312_312789

theorem ring_stack_distance :
  ∀ (n : ℕ) (a l : ℕ), a = 19 ∧ l = 2 ∧ n = 18 →
    (∑ k in (finset.range n).filter (λ x, x ≥ 1), (a - k + 1)) + 2 = 173 :=
by intros n a l h; sorry

end ring_stack_distance_l312_312789


namespace last_digits_of_numbers_divisible_by_4_l312_312622

-- Prove that there are 3 possible last digits for numbers divisible by 4
theorem last_digits_of_numbers_divisible_by_4 : {d : ℕ | d < 10 ∧ ∃ n : ℕ, (10 * n + d) % 4 = 0}.finite := by
  sorry

end last_digits_of_numbers_divisible_by_4_l312_312622


namespace trapezoid_ratio_of_bases_l312_312662

theorem trapezoid_ratio_of_bases (a b : ℝ) (h : a > b)
    (H : (1 / 4) * (1 / 2) * (a - b) * h = (1 / 8) * (a + b) * h) : 
    a / b = 3 := 
sorry

end trapezoid_ratio_of_bases_l312_312662


namespace hyperbola_equation_valid_dot_product_is_zero_l312_312038

noncomputable def hyperbola_center_origin (a b : ℝ) : Prop :=
  ∃ c : ℝ, a^2 = b^2 ∧ c = sqrt 6 ∧
    ∀ x y,
      (x^2 / 6 - y^2 / 6 = 1) ∧
      (4, -sqrt (10)).fst^2 / 6 - (4, -sqrt (10)).snd^2 / 6 = 1

noncomputable def dot_product_zero (a b x y : ℝ) : Prop :=
  ∃ (m : ℝ) (F1 F2 M : ℝ × ℝ),
    F1 = (-sqrt 6, 0) ∧
    F2 = (sqrt 6, 0) ∧
    (x = 3) ∧ ((x, y).fst^2 / 6 - (x, y).snd^2 / 6 = 1) ∧
    (m = sqrt 3 ∨ m = -sqrt 3) ∧
    (M = (3, m)) ∧
    (F1 = (-sqrt 6, 0)) ∧
    (F2 = (sqrt 6, 0)) ∧
    ((M.fst - F1.fst) * (M.fst - F2.fst) + (M.snd - F1.snd) * (M.snd - F2.snd) = 0)

theorem hyperbola_equation_valid : 
  hyperbola_center_origin 6 6 := by
  -- Proof omitted
  sorry

theorem dot_product_is_zero (M: ℝ × ℝ) : 
  dot_product_zero 6 6 M.fst M.snd := by
  -- Proof omitted
  sorry

end hyperbola_equation_valid_dot_product_is_zero_l312_312038


namespace largest_n_for_factoring_l312_312868

theorem largest_n_for_factoring :
  ∃ n, (∃ A B : ℤ, (3 * A = 3 * 108 + 1) ∧ (/3 * B * 108 = 2) ∧ 
  (3 * 36 + 3 = 111) ∧ (3 * 108 + A = n) )=
  (n = 325) := sorry
iddenLean_formatter.clonecreateAngular

end largest_n_for_factoring_l312_312868


namespace triangle_obtuse_angle_value_of_y_l312_312121

theorem triangle_obtuse_angle_value_of_y 
  (A B C θ : ℝ) 
  (h_C_obtuse : A + B < 90)
  (h_P_coords : (sin A - cos B, cos A - sin B))
: (sin θ / abs (sin θ) + abs (cos θ) / cos θ + tan θ / abs (tan θ) = -1) :=
sorry

end triangle_obtuse_angle_value_of_y_l312_312121


namespace num_divisors_of_36_l312_312487

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312487


namespace intersection_union_l312_312899

open Set

variable {α : Type*} [LinearOrder α]

def A (x : α) : Prop := -1 < x ∧ x < 7
def B (x : α) : Prop := 2 < x ∧ x < 10

theorem intersection (x : α) : (A x ∧ B x) ↔ (2 < x ∧ x < 7) := by
  sorry

theorem union (x : α) : (A x ∨ B x) ↔ (-1 < x ∧ x < 10) := by
  sorry

end intersection_union_l312_312899


namespace measure_of_angle_ABC_l312_312532

-- Define the angles involved and their respective measures
def angle_CBD : ℝ := 90 -- Given that angle CBD is a right angle
def angle_sum : ℝ := 160 -- Sum of the angles around point B
def angle_ABD : ℝ := 50 -- Given angle ABD

-- Define angle ABC to be determined
def angle_ABC : ℝ := angle_sum - (angle_ABD + angle_CBD)

-- Define the statement
theorem measure_of_angle_ABC :
  angle_ABC = 20 :=
by 
  -- Calculations omitted
  sorry

end measure_of_angle_ABC_l312_312532


namespace greatest_perimeter_approx_l312_312322

/-- Define the isosceles triangle -/
structure IsoscelesTriangle :=
(base : ℝ)
(height : ℝ)

/-- Define an equal area piece -/
structure EqualAreaPiece :=
(perimeter : ℝ)

/-- Calculate the perimeter of a given piece of the isosceles triangle -/
def perimeter_piece (triangle : IsoscelesTriangle) (k : ℕ) : ℝ :=
2 + Real.sqrt (triangle.height^2 + (2 * k)^2) + Real.sqrt (triangle.height^2 + (2 * (k + 1))^2)

/-- The given isosceles triangle with base 12 and height 15 -/
def given_triangle : IsoscelesTriangle :=
{base := 12, height := 15}

/-- The maximum perimeter among the 6 pieces -/
def max_perimeter : ℝ :=
List.maximum (List.map (λ k, perimeter_piece given_triangle k) [0, 1, 2, 3, 4])

/-- Theorem: The maximum perimeter is approximately 37.03 inches. -/
theorem greatest_perimeter_approx :
    abs (max_perimeter - 37.03) < 0.01 :=
sorry

end greatest_perimeter_approx_l312_312322


namespace abs_value_of_z_l312_312433

def complex_z : ℂ := (1 + 2 * complex.I) / complex.I

theorem abs_value_of_z :
  abs complex_z = real.sqrt 5 := 
sorry

end abs_value_of_z_l312_312433


namespace compute_p_plus_q_l312_312604

theorem compute_p_plus_q (p q : ℝ) (h : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = 8 ∧ a * b + b * c + a * c = p ∧ a * b * c = q)) :
  p + q = 27 :=
by
  obtain ⟨a, b, c, hab, hbc, hac, h1, h2, h3⟩ := h
  cases' (nat.mul_eq_one_dvd a (nat.lt_of_ne_of_ne hab (hac.symm.trans hbc).symm)).symm
  obtain rfl | rfl : a = 1 ∨ a = b by simp [h3] 
  contradiction
  sorry

end compute_p_plus_q_l312_312604


namespace number_of_divisors_of_36_l312_312503

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312503


namespace find_a_range_l312_312423

noncomputable def is_decreasing (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f y < f x

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, x > -1 → 3 * x^2 - a * x + 5 > 0) →
  is_decreasing (fun x => Real.logBase 0.5 (3 * x^2 - a * x + 5)) {x : ℝ | x > -1} →
  a ∈ Ioo (-8) (-6) :=
by
  sorry

end find_a_range_l312_312423


namespace part_I_part_II_part_III_l312_312113

-- Define the sequence {a_n}
def a : ℕ → ℚ
| 0       := 0       -- Placeholder, not used, a_1 should correspond to a 1
| 1       := 1
| (n + 2) := 2 * a (n + 1) / (2 + a (n + 1))

-- Define the sequence {b_n}
def b (n : ℕ) : ℚ := a n / n

-- Sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i, b (i + 1))

theorem part_I :
  a 2 = 2 / 3 ∧
  a 3 = 1 / 2 ∧
  a 4 = 2 / 5 :=
by
  sorry

theorem part_II : ∀ n : ℕ, a (n + 1) = 2 / (n + 2) :=
by
  sorry

theorem part_III : ∀ n : ℕ, S n = 2 * n / (n + 1) :=
by
  sorry

end part_I_part_II_part_III_l312_312113


namespace range_of_a_l312_312442

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then x^2 + 4
  else if 0 < x then sin (Real.pi * x)
  else 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (-1 ≤ x ∧ x < 0) ∨ (0 < x) → f x - a * x ≥ -1) ↔ (-6 ≤ a ∧ a ≤ 0) :=
sorry

end range_of_a_l312_312442


namespace ratio_DQ_EQ_l312_312120

theorem ratio_DQ_EQ (D E F Q : EudlPlane.Point) 
  (h_triangle_DEF : right_triangle D E F)
  (h_angle_DFE : ∠ D F E < 45)
  (h_DE : dist D E = 5)
  (h_point_Q_on_DE : Q ∈ seg D E)
  (h_angle_DQF : ∠ D Q F = 2 * ∠ D F Q)
  (h_FQ : dist F Q = 2) :
  (∃ (u v : ℕ) (w : ℕ), (w.prime ∧ (u + v * real.sqrt w) = ratio (dist D Q) (dist E Q)) ∧ 
  (u = 23 ∧ v = 5 ∧ w = 19)) :=
sorry

end ratio_DQ_EQ_l312_312120


namespace total_pencils_l312_312820

variable (C Y M D : ℕ)

-- Conditions
def cheryl_has_thrice_as_cyrus (h1 : C = 3 * Y) : Prop := true
def madeline_has_half_of_cheryl (h2 : M = 63 ∧ C = 2 * M) : Prop := true
def daniel_has_25_percent_of_total (h3 : D = (C + Y + M) / 4) : Prop := true

-- Total number of pencils for all four
theorem total_pencils (h1 : C = 3 * Y) (h2 : M = 63 ∧ C = 2 * M) (h3 : D = (C + Y + M) / 4) :
  C + Y + M + D = 289 :=
by { sorry }

end total_pencils_l312_312820


namespace num_false_prop_l312_312046

-- Defining the propositions
def prop1 : Prop := ∀ (l₁ l₂ l : Type) [linear_space l₁ l₂ l], 
  perpendicular l₁ l → perpendicular l₂ l → parallel l₁ l₂

def prop2 : Prop := ∀ (π₁ π₂ π : Plane), 
  perpendicular_plane π₁ π → perpendicular_plane π₂ π → parallel_plane π₁ π₂

def prop3 : Prop := ∀ (l₁ l₂ : Type) (π : Plane) [linear_space l₁ π l₂], 
  (angle_line_plane l₁ π = angle_line_plane l₂ π) → parallel l₁ l₂

def prop4 : Prop := ∀ (l₁ l₂ l₃ l₄ : Type) [linear_space l₁ l₂ l₃ l₄], 
  skew_lines l₁ l₂ → intersects l₃ l₁ ∧ intersects l₃ l₂ → intersects l₄ l₁ ∧ intersects l₄ l₂ → skew_lines l₃ l₄


-- Declaring the main theorem to prove the number of false propositions
theorem num_false_prop : ∑ (b : Bool) in [¬prop1, ¬prop2, ¬prop3, ¬prop4].to_finset, cond b 1 0 = 4 :=
by
  sorry

end num_false_prop_l312_312046


namespace number_of_divisors_of_36_l312_312508

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312508


namespace find_legs_of_right_triangle_l312_312043

theorem find_legs_of_right_triangle (x y a Δ : ℝ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : 2 * Δ = x * y) : 
  x = (Real.sqrt (a^2 + 4 * Δ) + Real.sqrt (a^2 - 4 * Δ)) / 2 ∧ 
  y = (Real.sqrt (a^2 + 4 * Δ) - Real.sqrt (a^2 - 4 * Δ)) / 2 :=
sorry

end find_legs_of_right_triangle_l312_312043


namespace find_roots_of_g_l312_312961

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - a*x - b
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := b*x^2 - a*x - 1

theorem find_roots_of_g :
  (∀ a b : ℝ, f 2 a b = 0 ∧ f 3 a b = 0 → ∃ (x1 x2 : ℝ), g x1 a b = 0 ∧ g x2 a b = 0 ∧
    (x1 = -1/2 ∨ x1 = -1/3) ∧ (x2 = -1/2 ∨ x2 = -1/3) ∧ x1 ≠ x2) :=
by
  sorry

end find_roots_of_g_l312_312961


namespace ceil_neg_3_7_l312_312357

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l312_312357


namespace min_emails_ensure_all_learn_l312_312353

theorem min_emails_ensure_all_learn (n : ℕ) (h : n = 18) : 
  ∃ m : ℕ, m = 34 ∧ 
           ∀ (method : (fin n → set (fin n)) → Prop), method (λ x, {x}) → 
                                                      method (λ _ , {0, 1, ..., n - 1}) → 
                                                      ∃ total_emails, total_emails = 34 :=
begin
  sorry
end

end min_emails_ensure_all_learn_l312_312353


namespace shaded_triangle_area_l312_312974

theorem shaded_triangle_area (AB BC : ℝ)
  (hAB : AB = 6)
  (hBC : BC = 8) :
  let area_ABC := (1 / 2) * AB * BC in
  let shaded_series := geomSeries ((1 / 4) * area_ABC) (1 / 4) in
  infinite_series_sum shaded_series = 8 :=
by
  sorry

end shaded_triangle_area_l312_312974


namespace youth_gathering_l312_312325

theorem youth_gathering (x : ℕ) (h1 : ∃ x, 9 * (2 * x + 12) = 20 * x) : 
  2 * x + 12 = 120 :=
by sorry

end youth_gathering_l312_312325


namespace smallest_triangle_perimeter_l312_312258

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l312_312258


namespace triangle_concurrency_l312_312750

theorem triangle_concurrency 
  (A B C A1 B1 C1 : Point)
  (isosceles : AB = BC)
  (A1_on_BC : on BC A1)
  (B1_on_CA : on CA B1)
  (C1_on_AB : on AB C1)
  (concurrent : lines_concurrent AA1 BB1 CC1) :
  (AC1 / C1B) = (sin (BAB1) * sin (CAA1)) / (sin (BAA1) * sin (CBB1)) :=
sorry

end triangle_concurrency_l312_312750


namespace parabola_coefficients_l312_312640

theorem parabola_coefficients 
  (a b c : ℝ) 
  (h_vertex : ∀ x : ℝ, (2 - (-2))^2 * a + (-2 * 2 * a + b) * (2 - (-2)) + (c - 5) = 0)
  (h_point : 9 = a * (2:ℝ)^2 + b * (2:ℝ) + c) : 
  a = 1 / 4 ∧ b = 1 ∧ c = 6 := 
by 
  sorry

end parabola_coefficients_l312_312640


namespace find_principal_l312_312876

-- Assume we are given the following conditions
variables (P : ℝ) (T R : ℝ)
axiom time_condition : T = 2
axiom rate_condition : R = 10 / 100
axiom interest_difference : (P * (1 + R)^T - P - P * R * T) = 12

-- The goal is to prove that the principal amount P is 1200
theorem find_principal : P = 1200 :=
by
  -- invoke the conditions to construct a system of equations
  have h1 : T = 2 := time_condition
  have h2 : R = 10 / 100 := rate_condition
  have h3 : (P * (1 + R)^T - P - P * R * T) = 12 := interest_difference
  -- use these to solve for P
  sorry

end find_principal_l312_312876


namespace trapezoid_base_ratio_l312_312648

-- Define variables and conditions as per the problem
variables {a b h: ℝ}
def is_trapezoid (a b: ℝ) := a > b
def area_trapezoid (a b h: ℝ) := (1 / 2) * (a + b) * h
def area_quadrilateral (a b h: ℝ) := (1 / 2) * ((a - b) / 2) * (h / 2)

-- State the problem statement
theorem trapezoid_base_ratio (a b h: ℝ) (ha: a > b) (ht: area_quadrilateral a b h = (1 / 4) * area_trapezoid a b h) :
  a / b = 3 :=
sorry

end trapezoid_base_ratio_l312_312648


namespace max_value_m_l312_312153

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end max_value_m_l312_312153


namespace solve_for_y_l312_312183

theorem solve_for_y
  (y : ℤ)
  (h : 5^(y+1) = 625) :
  y = 3 :=
sorry

end solve_for_y_l312_312183


namespace DX_eq_DY_l312_312824

-- Definitions: Parallelogram ABCD with AB > AD
variable (A B C D X Y : Point)
variable (AB AD CX CB AY CD DX DY : Length)
variable (BD_ray : Ray)

-- Conditions
axiom Parallelogram_ABCD : parallelogram A B C D
axiom AB_gt_AD : AB > AD
axiom CX_eq_CB : CX = CB
axiom AY_eq_AB : AY = AB
axiom X_Y_on_BD_ray : OnRay B D X ∧ OnRay B D Y
axiom notation_BD_ray : BD_ray = ray B D

theorem DX_eq_DY : DX = DY :=
by
  -- The required proof steps would be implemented here.
  sorry

end DX_eq_DY_l312_312824


namespace determine_percentage_of_yellow_in_darker_green_paint_l312_312016

noncomputable def percentage_of_yellow_in_darker_green_paint : Real :=
  let volume_light_green := 5
  let volume_darker_green := 1.66666666667
  let percentage_light_green := 0.20
  let final_percentage := 0.25
  let total_volume := volume_light_green + volume_darker_green
  let total_yellow_required := final_percentage * total_volume
  let yellow_in_light_green := percentage_light_green * volume_light_green
  (total_yellow_required - yellow_in_light_green) / volume_darker_green

theorem determine_percentage_of_yellow_in_darker_green_paint :
  percentage_of_yellow_in_darker_green_paint = 0.4 := by
  sorry

end determine_percentage_of_yellow_in_darker_green_paint_l312_312016


namespace log_increasing_interval_l312_312401

-- Definitions of the problem conditions
def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2 * x else x^2 + 2 * x

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

-- Statement of the proof problem
theorem log_increasing_interval:
  is_even f →
  (∀ x, (log 2) (x^2 - 4 * x - 5) = (log 2) (x^2 - 4 * x - 5)) →
  Ioi (5 : ℝ) ⊆ { x : ℝ | 0 < x^2 - 4 * x - 5 } :=
sorry

end log_increasing_interval_l312_312401


namespace no_prime_divisible_by_56_l312_312528

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end no_prime_divisible_by_56_l312_312528


namespace eccentricity_theorem_l312_312963

noncomputable def eccentricity_of_ellipse 
  (A B C : Type)
  [metric_space C] 
  (dist : C → C → ℝ)
  (triangle_ABC : A ∈ metric.affine_triangle B C) 
  (angle_A : ∠ B A C = π / 2)
  (tan_B : real.tan (∠ A B C) = 3 / 4) : ℝ :=
  let AB := 4 in
  let AC := 3 in
  let BC := 5 in
  let e := 1 / 2 in
  e

theorem eccentricity_theorem 
  (A B C : Type)
  [metric_space C] 
  (dist : C → C → ℝ)
  (triangle_ABC : A ∈ metric.affine_triangle B C) 
  (angle_A : ∠ B A C = π / 2)
  (tan_B : real.tan (∠ A B C) = 3 / 4) :
  eccentricity_of_ellipse A B C dist triangle_ABC angle_A tan_B = 1 / 2 :=
sorry

end eccentricity_theorem_l312_312963


namespace sqrt_sum_dif_multiplication_formula_step3_error_correct_result_l312_312288

theorem sqrt_sum_dif : real.sqrt 32 + real.sqrt 8 - real.sqrt 50 = real.sqrt 2 :=
by
  sorry

theorem multiplication_formula : ∃ a b : ℝ, (@eq (ℝ → ℝ → ℝ) (λx y, (x + y)^2) (λx y, x^2 + 2 * x * y + y^2)) ∧
                                           (@eq (ℝ → ℝ → ℝ) (λx y, (x - y)^2) (λx y, x^2 - 2 * x * y + y^2)) :=
by
  sorry

theorem step3_error (x : ℝ) : let a := (2 * real.sqrt 6) in (a^2 ≠ 12) :=
by
  sorry

theorem correct_result : ((real.sqrt 3 - real.sqrt 2)^2) * (5 + 2 * real.sqrt 6) = 1 :=
by
  sorry

end sqrt_sum_dif_multiplication_formula_step3_error_correct_result_l312_312288


namespace bisection_min_iterations_l312_312464

-- Define the interval and precision
def interval_start := 0
def interval_end := 4
def precision := 0.001

-- Define the inequality condition from the bisection method
def bisection_inequality (n : ℕ) : Prop :=
  interval_end - interval_start < 2^n * precision

-- Prove that the minimum n that satisfies the inequality is 12
theorem bisection_min_iterations : ∃ n, bisection_inequality n ∧ (∀ m, bisection_inequality m → n ≤ m) :=
  sorry

end bisection_min_iterations_l312_312464


namespace collinear_points_eq_sum_l312_312836

theorem collinear_points_eq_sum (a b : ℝ) :
  -- Collinearity conditions in ℝ³
  (∃ t1 t2 t3 t4 : ℝ,
    (2, a, b) = (a + t1 * (a - 2), 3 + t1 * (b - 3), b + t1 * (4 - b)) ∧
    (a, 3, b) = (a + t2 * (a - 2), 3 + t2 * (b - 3), b + t2 * (4 - b)) ∧
    (a, b, 4) = (a + t3 * (a - 2), 3 + t3 * (b - 3), b + t3 * (4 - b)) ∧
    (5, b, a) = (a + t4 * (a - 2), 3 + t4 * (b - 3), b + t4 * (4 - b))) →
  a + b = 9 :=
by
  sorry

end collinear_points_eq_sum_l312_312836


namespace infinite_primes_in_S_max_value_f_for_S_l312_312144

-- Define the set S as the set of primes with the desired property
def S (p : ℕ) : Prop := 
  Nat.Prime p ∧ ∃ r : ℕ, r > 0 ∧
  (DecimalExpansion (1 : ℝ) / p).LengthOfMinimalRepeatingBlock = 3 * r

-- Define the function f(k, p)
def f (k p : ℕ) : ℕ :=
  let dec := (DecimalExpansion (1 : ℝ) / p)
  let r := (dec.LengthOfMinimalRepeatingBlock / 3)
  dec.getDigit (k - 1) + dec.getDigit (k - 1 + r) + dec.getDigit (k - 1 + 2 * r)

-- Problem statement part (1): Prove that S contains infinitely many primes
theorem infinite_primes_in_S : ∃ inf : ℕ → ℕ, ∀ n : ℕ, S (inf n) := sorry

-- Problem statement part (2): Prove the maximum value of f(k, p) for k ≥ 1 and p ∈ S
theorem max_value_f_for_S : ∃ k : ℕ, ∃ p : ℕ, S p ∧ k ≥ 1 ∧ f(k, p) = 19 := sorry

end infinite_primes_in_S_max_value_f_for_S_l312_312144


namespace ceil_neg_3_7_l312_312359

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l312_312359


namespace solid_with_triangular_views_is_tetrahedron_l312_312960

theorem solid_with_triangular_views_is_tetrahedron :
  (∀ (solid : Type) (view_front : solid → Set Triangle) (view_top : solid → Set Triangle) (view_side : solid → Set Triangle),
  (∀ s, view_front s ∧ view_top s ∧ view_side s) → solid = Tetrahedron) :=
sorry

end solid_with_triangular_views_is_tetrahedron_l312_312960


namespace patrolman_direction_and_distance_patrolman_total_distance_patrolman_return_time_train_departure_time_l312_312307

-- Definitions
def journey_segments : List ℝ := [-5.4, 3.2, -3.6]
def constant_speed : ℝ := 6
def A_to_B_distance : ℝ := 30
def train_speed : ℝ := 180
def train_pass_time : ℝ := 2 + 18 / 60 -- in hours (2:18 PM)

-- Calculations
def net_distance : ℝ := journey_segments.sum
def patrolman_distance_from_A : ℝ := -net_distance
def total_distance : ℝ := journey_segments.sum (·.abs)
def return_journey_distance : ℝ := total_distance + patrolman_distance_from_A.abs
def return_time : ℝ := return_journey_distance / constant_speed
def return_time_pm : ℝ := 1 + return_time -- in PM hours
def walking_time : ℝ := train_pass_time - 1 -- Patrolman walking until 2:18 PM
def patrolman_distance_walked : ℝ := walking_time * constant_speed
def patrolman_position_at_pass : ℝ := -5.4 + patrolman_distance_walked - -5.8 -- Negative for direction correction
def distance_B_to_patrolman_at_pass: ℝ := A_to_B_distance + patrolman_position_at_pass 
def train_travel_time : ℝ := distance_B_to_patrolman_at_pass / train_speed
def train_departure_time_pm : ℝ := train_pass_time - train_travel_time -- in PM hours

-- Lean Statements
theorem patrolman_direction_and_distance (segment : List ℝ) : (segment.sum = -5.8) -> (patrolman_distance_from_A = 5.8) := 
by
  sorry

theorem patrolman_total_distance (segment : List ℝ) : (total_distance = 12.2) := 
by
  sorry

theorem patrolman_return_time : (return_time_pm = 4) :=
by
  sorry

theorem train_departure_time : (train_departure_time_pm = 2 + 9 / 60) := 
by
  sorry

end patrolman_direction_and_distance_patrolman_total_distance_patrolman_return_time_train_departure_time_l312_312307


namespace milk_selected_l312_312812

theorem milk_selected (N : ℕ) (hsoda : 0.5 * N = 100) (hmilk : 0.3 * N = 60) : true :=
by
  have total_students : N = 200 :=
    by
      have hn : 0.5 * N = 100 := hsoda
      sorry
  have students_selected_milk : 0.3 * N = 60 :=
    by
      have hn : N = total_students := by sorry
      sorry
  sorry

end milk_selected_l312_312812


namespace min_num_edges_chromatic_l312_312561

-- Definition of chromatic number.
def chromatic_number (G : SimpleGraph V) : ℕ := sorry

-- Definition of the number of edges in a graph as a function.
def num_edges (G : SimpleGraph V) : ℕ := sorry

-- Statement of the theorem.
theorem min_num_edges_chromatic (G : SimpleGraph V) (n : ℕ) 
  (chrom_num_G : chromatic_number G = n) : 
  num_edges G ≥ n * (n - 1) / 2 :=
sorry

end min_num_edges_chromatic_l312_312561


namespace find_CD_l312_312236

open Real EuclideanGeometry

-- Definitions and Conditions
/-- Two circles touch each other externally at point A. --/
axiom circles_touch_at (A : Point) (circle1 circle2 : Circle) : touches_externally A circle1 circle2

/-- The common tangent touches the first circle at point B and the second circle at point C. --/
axiom common_tangent (B C : Point) (circle1 circle2 : Circle) : is_common_tangent B circle1 ∧ is_common_tangent C circle2

/-- The line passing through points A and B intersects the second circle at point D. --/
axiom line_intersects (A B D : Point) (circle2 : Circle) : line_passes_through A B ∧ intersects B D circle2

/-- Given distances AB and AD. --/
axiom distances (A B D : Point) : dist A B = 5 ∧ dist A D = 4

-- Theorem to prove
theorem find_CD 
  (A B C D : Point) 
  (circle1 circle2 : Circle)
  (h1 : touches_externally A circle1 circle2)
  (h2 : is_common_tangent B circle1 ∧ is_common_tangent C circle2)
  (h3 : line_passes_through A B ∧ intersects B D circle2)
  (h4 : dist A B = 5)
  (h5 : dist A D = 4) :
  dist C D = 6 :=
by
  sorry

end find_CD_l312_312236


namespace ratio_HD_HA_l312_312556

-- Define the triangle side lengths
variables {a b c : ℝ} (h : a = 11) (h2 : b = 13) (h3 : c = 20)

-- Define the semi-perimeter and area
def s := (a + b + c) / 2
noncomputable def A := real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the altitude from vertex A and the lengths HA and HD
def h' := (2 * A) / c
def HA := h'
def HD := 0

-- Prove the ratio HD:HA is 0
theorem ratio_HD_HA : HD / HA = 0 :=
by
  have s_value : s = 22 := by sorry
  have A_value : A = 66 := by sorry
  have h_value : h' = 6.6 := by sorry
  have HA_value : HA = 6.6 := by sorry
  sorry

end ratio_HD_HA_l312_312556


namespace problem1_problem2_l312_312748

theorem problem1 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := 
sorry

theorem problem2 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin (↑(π/2) + α) * Real.cos (↑(5*π/2) - α) * Real.tan (↑(-π) + α)) / 
    (Real.tan (↑(7*π) - α) * Real.sin (↑π + α)) = Real.cos α := 
sorry

end problem1_problem2_l312_312748


namespace divisors_of_36_l312_312474

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312474


namespace find_line_equation_l312_312035

theorem find_line_equation (a : ℝ) (h : a < 3) :
  ∃ l : AffineMap ℝ (EuclideanSpace ℝ) ℝ,
    (∀ P Q : Point,
      (P ∉ l.range → P ∉ Sphere ℝ 2) ∧ (Q ∉ l.range → Q ∉ Sphere ℝ 2) ∧
      (P.1 + Q.1 = -4) ∧ (P.2 + Q.2 = 6))
    → 
    ∃ l_eq : String, l_eq = "x - y + 5 = 0" :=  
sorry

end find_line_equation_l312_312035


namespace count_three_digit_clubsuit_clubsuit_five_l312_312998

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def num_three_digit_for_clubsuit_clubsuit_eq_five : ℕ :=
  (List.range' 100 900).filter (λ x, sum_of_digits (sum_of_digits x) = 5) |>.length

theorem count_three_digit_clubsuit_clubsuit_five : num_three_digit_for_clubsuit_clubsuit_eq_five = 17 :=
  sorry

end count_three_digit_clubsuit_clubsuit_five_l312_312998


namespace no_partition_exists_l312_312708

noncomputable section

open Set

def partition_N (A B C : Set ℕ) : Prop := 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧  -- Non-empty sets
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧  -- Disjoint sets
  A ∪ B ∪ C = univ ∧  -- Covers the whole ℕ
  (∀ a ∈ A, ∀ b ∈ B, a + b + 2008 ∈ C) ∧
  (∀ b ∈ B, ∀ c ∈ C, b + c + 2008 ∈ A) ∧
  (∀ c ∈ C, ∀ a ∈ A, c + a + 2008 ∈ B)

theorem no_partition_exists : ¬ ∃ (A B C : Set ℕ), partition_N A B C :=
by
  sorry

end no_partition_exists_l312_312708


namespace obtuse_triangle_acute_triangle_l312_312125

noncomputable section
open Real

variables {α β γ : ℝ}
variables (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ)

/-- In any non-right triangle, the sum of the angles is π. -/
axiom angle_sum (h : α + β + γ = π) : α + β + γ = π

/-- The identity for the sum of the tangents of the angles in a non-right triangle. -/
lemma tangent_identity (hsum: α + β + γ = π) : 
  tan α + tan β + tan γ = tan α * tan β * tan γ :=
sorry

/-- In obtuse triangles, the sum of the tangents of the angles is less than the sum of the cotangents. -/
theorem obtuse_triangle (h_obtuse: max α (max β γ) > π / 2) : 
  tan α + tan β + tan γ < cot α + cot β + cot γ :=
sorry

/-- In acute triangles, the sum of the cotangents of the angles is less than the sum of the tangents. -/
theorem acute_triangle (h_acute: max α (max β γ) < π / 2) : 
  cot α + cot β + cot γ < tan α + tan β + tan γ :=
sorry

end obtuse_triangle_acute_triangle_l312_312125


namespace trapezoid_base_ratio_l312_312663

variable {A B C D K M P Q L N : Type}
variable [LinearOrderedField A]
variable [LinearOrderedField B]
variable [LinearOrderedField C]
variable [LinearOrderedField D]

def is_trapezoid (A B C D : Type) (a b : ℝ) : Prop :=
  is_parallel (AD BC) ∧ AD.length = a ∧ BC.length = b ∧ a > b

def area (A B C D : Type) (a b : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

def quadrilateral_area (K L M N : Type) (a b h : ℝ) : ℝ :=
  (1 / 4) * (a - b) * h

theorem trapezoid_base_ratio (A B C D K M P Q L N : Type) (a b : ℝ) (h : ℝ) :
  is_trapezoid A B C D a b →
  quadrilateral_area K L M N a b h = (1 / 4) * area A B C D a b ↔ a / b = 3 :=
by 
  sorry

end trapezoid_base_ratio_l312_312663


namespace second_largest_possible_n_div_170_fact_l312_312537

theorem second_largest_possible_n_div_170_fact (n : ℕ) : 
  ∃ n, (n = 40 ∧ ∃ m : ℕ, 170! / (10^n) = m) :=
sorry

end second_largest_possible_n_div_170_fact_l312_312537


namespace four_digit_numbers_count_l312_312301

open Finset

-- Define a digit type
inductive Digit : Type
| one : Digit
| two : Digit
| three : Digit

-- Define a four-digit number using only 1, 2, and 3 where all three digits must be used
-- and the same digit cannot appear adjacent.
def valid_numbers : Finset (List Digit) :=
  (univ.product (univ.product (univ.product univ)))
  .filter (λ d, 
    d.nth 0 = some Digit.one ∨ d.nth 0 = some Digit.two ∨ d.nth 0 = some Digit.three ∧
    d.nth 1 = some Digit.one ∨ d.nth 1 = some Digit.two ∨ d.nth 1 = some Digit.three ∧
    d.nth 2 = some Digit.one ∨ d.nth 2 = some Digit.two ∨ d.nth 2 = some Digit.three ∧
    d.nth 3 = some Digit.one ∨ d.nth 3 = some Digit.two ∨ d.nth 3 = some Digit.three ∧
    d.nth 0 ≠ d.nth 1 ∧ d.nth 1 ≠ d.nth 2 ∧ d.nth 2 ≠ d.nth 3 ∧
    (d.nth 0 = some Digit.one ∨ d.nth 1 = some Digit.one ∨ d.nth 2 = some Digit.one ∨ d.nth 3 = some Digit.one) ∧
    (d.nth 0 = some Digit.two ∨ d.nth 1 = some Digit.two ∨ d.nth 2 = some Digit.two ∨ d.nth 3 = some Digit.two) ∧
    (d.nth 0 = some Digit.three ∨ d.nth 1 = some Digit.three ∨ d.nth 2 = some Digit.three ∨ d.nth 3 = some Digit.three)
  )

theorem four_digit_numbers_count : valid_numbers.card = 18 := sorry

end four_digit_numbers_count_l312_312301


namespace walker_cyclist_catchup_l312_312696

-- Define constants, conditions, and concluding proof
theorem walker_cyclist_catchup : 
  ∀ (walker_speed cyclist_speed : ℝ) (stopping_time : ℝ),
    walker_speed = 4 ∧ cyclist_speed = 20 ∧ stopping_time = 5 → 
    ∃ catchup_time : ℝ, catchup_time = 25 := 
by 
  -- We state that walker_speed (in miles/hour) is 4, cyclist_speed (in miles/hour) is 20, and stopping_time (in minutes) is 5
  intros walker_speed cyclist_speed stopping_time h,
  -- Using the given conditions, we conclude that the walker will catch up to the cyclist after 25 minutes
  cases h with h1 h,
  cases h with h2 h3,
  use 25,
  sorry

end walker_cyclist_catchup_l312_312696


namespace equation_of_circle_length_of_chord_equation_of_MN_l312_312403

open Real EuclideanGeometry

-- Define the circle C centered at the origin
def circleC := mk_circle (0, 0) sqrt 4

-- Define the line l1: x - y - 2√2 = 0
def l1 := mk_line (1, -1, -2 * sqrt 2)

-- Define the line l2: 4x - 3y + 5 = 0
def l2 := mk_line (4, -3, 5)

-- Define point G at coordinates (1,3)
def G := (1, 3)

-- The proof statements
theorem equation_of_circle :
  ∃ r, circleC.equation = x^2 + y^2 = r^2 ∧ r^2 = 4 := sorry

theorem length_of_chord :
  ∃ l, chord_length circleC l2 = l ∧ l = 2 * sqrt 3 := sorry

theorem equation_of_MN :
  ∃ mn, tangent_line_through_point G circleC = mn ∧ mn.equation = x + 3y - 4 := sorry

end equation_of_circle_length_of_chord_equation_of_MN_l312_312403


namespace ceil_neg_3_7_l312_312364

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l312_312364


namespace correct_statements_about_microbial_counting_l312_312271

def hemocytometer_counts_bacteria_or_yeast : Prop :=
  true -- based on condition 1

def plate_streaking_allows_colony_counting : Prop :=
  false -- count is not done using the plate streaking method, based on the analysis

def dilution_plating_allows_colony_counting : Prop :=
  true -- based on condition 3  
  
def dilution_plating_count_is_accurate : Prop :=
  false -- colony count is often lower than the actual number, based on the analysis

theorem correct_statements_about_microbial_counting :
  (hemocytometer_counts_bacteria_or_yeast ∧ dilution_plating_allows_colony_counting)
= (plate_streaking_allows_colony_counting ∨ dilution_plating_count_is_accurate) :=
by sorry

end correct_statements_about_microbial_counting_l312_312271


namespace percentage_difference_is_40_l312_312304

-- Define the sizes and the length function
def smallest_size := 8
def largest_size := 17
def unit_increase := 1 / 5
def size15_length := 5.9

-- Function to calculate the length of the shoe given its size
def shoe_length (size : ℝ) : ℝ := size15_length + (size - 15) * unit_increase

-- Calculate the lengths for the smallest and largest sizes
def smallest_length := shoe_length smallest_size
def largest_length := shoe_length largest_size

-- Prove the percentage difference is 40%
theorem percentage_difference_is_40 :
  ((largest_length - smallest_length) / smallest_length) * 100 = 40 := 
sorry

end percentage_difference_is_40_l312_312304


namespace students_in_circle_l312_312975

theorem students_in_circle (n : ℕ) (students : Fin n → Prop) (h : ∀ i : Fin n, (students i ≠ students (i + 1)) ∧ (students i ≠ students (i - 1))) : n % 4 = 0 :=
sorry

end students_in_circle_l312_312975


namespace N_P_minus_N_Q_eq_285_l312_312739

def P_type_numbers (abcd : Nat × Nat × Nat × Nat) : Prop :=
  let (a, b, c, d) := abcd in (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (0 ≤ d ∧ d ≤ 9)
  ∧ (a > b) ∧ (b < c) ∧ (c > d)

def Q_type_numbers (abcd : Nat × Nat × Nat × Nat) : Prop :=
  let (a, b, c, d) := abcd in (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (0 ≤ d ∧ d ≤ 9)
  ∧ (a < b) ∧ (b > c) ∧ (c < d)

def N_P : Nat :=
  Finset.card (Finset.filter P_type_numbers (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.range 10)))))

def N_Q : Nat :=
  Finset.card (Finset.filter Q_type_numbers (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.range 10)))))

theorem N_P_minus_N_Q_eq_285 : N_P - N_Q = 285 := by
  sorry

end N_P_minus_N_Q_eq_285_l312_312739


namespace systematic_sampling_50_5_l312_312019

/-- There are 50 missiles numbered from 1 to 50.
Five missiles need to be selected using systematic sampling.
We need to prove that the sequence {3, 13, 23, 33, 43} follows this method
and is thus the most likely outcome. -/
theorem systematic_sampling_50_5 (missiles : Finset ℕ) :
  (∀ x ∈ missiles, 1 ≤ x ∧ x ≤ 50) ∧ missiles.card = 5 ∧
  (∃ interval : ℕ, interval = 10 ∧ ∀ (a b ∈ missiles), a ≠ b → (a % interval) = 3 → (b % interval) = a + interval % 50 ∧ b < 50) →
  missiles = {3, 13, 23, 33, 43} :=
sorry

end systematic_sampling_50_5_l312_312019


namespace find_adult_buffet_price_l312_312619

variable {A : ℝ} -- Let A be the price for the adult buffet
variable (children_cost : ℝ := 45) -- Total cost for the children's buffet
variable (senior_discount : ℝ := 0.9) -- Discount for senior citizens
variable (total_cost : ℝ := 159) -- Total amount spent by Mr. Smith
variable (num_adults : ℕ := 2) -- Number of adults (Mr. Smith and his wife)
variable (num_seniors : ℕ := 2) -- Number of senior citizens

theorem find_adult_buffet_price (h1 : children_cost = 45)
    (h2 : total_cost = 159)
    (h3 : ∀ x, num_adults * x + num_seniors * (senior_discount * x) + children_cost = total_cost)
    : A = 30 :=
by
  sorry

end find_adult_buffet_price_l312_312619


namespace standard_eq_circle_find_b_l312_312402

-- Define the setup and given conditions
variables (x y a b : ℝ)

def circle_C_eq (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 4

-- Part 1: Given the conditions, prove the standard equation of the circle
theorem standard_eq_circle : 
  (∀ (x y a : ℝ), 2a > 0 ∧ x - 2y = 0 ∧ (a=a^2 + (sqrt 3)^2 / (2a)^2) → circle_C_eq x y) := 
sorry

-- Part 2: Given the line intersects the circle, and the circle with AB diameter passes through origin, find b
theorem find_b (b : ℝ) :
  (∀ (x y : ℝ), circle_C_eq x y ∧ (y = -2x + b) ∧ 
   ∃ (A B : ℝ×ℝ), (A.1 + B.1) = 4b/5 ∧ (A.1 * B.1 + ((-2*A.1 + b) * (-2*B.1 + b)) = 0) →
    b = (5 + sqrt 15) / 2 ∨ b = (5 - sqrt 15) / 2) :=
sorry

end standard_eq_circle_find_b_l312_312402


namespace volume_of_prism_l312_312231

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) :
    a * b * c = 426 :=
sorry

end volume_of_prism_l312_312231


namespace continuous_function_property_l312_312851

theorem continuous_function_property (f : ℝ → ℝ)
  (h_cont : continuous f)
  (h_property : ∀ (n : ℤ) (x : ℝ), 0 < n -> f (x + (1 / n)) ≤ f x + (1 / n)) :
  ∃ a : ℝ, ∀ x : ℝ, f x = x + a :=
sorry

end continuous_function_property_l312_312851


namespace congruence_equiv_l312_312949

theorem congruence_equiv (x : ℤ) (h : 5 * x + 9 ≡ 3 [ZMOD 18]) : 3 * x + 14 ≡ 14 [ZMOD 18] :=
sorry

end congruence_equiv_l312_312949


namespace expression_evaluation_l312_312374

theorem expression_evaluation (a : ℝ) (h : a = 9) : ( (a ^ (1 / 3)) / (a ^ (1 / 5)) ) = a^(2 / 15) :=
by
  sorry

end expression_evaluation_l312_312374


namespace quadratic_coefficients_l312_312383

theorem quadratic_coefficients (x1 x2 p q : ℝ)
  (h1 : x1 - x2 = 5)
  (h2 : x1 ^ 3 - x2 ^ 3 = 35) :
  (x1 + x2 = -p ∧ x1 * x2 = q ∧ (p = 1 ∧ q = -6) ∨ 
   x1 + x2 = p ∧ x1 * x2 = q ∧ (p = -1 ∧ q = -6)) :=
by
  sorry

end quadratic_coefficients_l312_312383


namespace range_f_l312_312874

def f (x : ℝ) : ℝ :=
  4 * Real.cos ((Real.pi / 3) * Real.sin (x^2 + 6 * x + 10 - Real.sin x))

theorem range_f : Set.Icc 2 4 = Set.range (f) :=
sorry

end range_f_l312_312874


namespace solution_l312_312637

-- Define the conditions: x, y, z are positive integers
variable (x y z : ℕ) 
-- The positive integer condition is denoted by x > 0, y > 0, and z > 0

-- The mathematical equation to check
def equation := 1 + 2^x + 3^y = z^3

-- Translate the correct answer
theorem solution : (x = 2) ∧ (y = 1) ∧ (z = 2) → equation x y z :=
by
  intro h
  rcases h with ⟨hx, hy, hz⟩
  rw [hx, hy, hz]
  simp
  sorry

end solution_l312_312637


namespace trajectory_midpoint_l312_312044

noncomputable def circle (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

theorem trajectory_midpoint (x y : ℝ) (hC : circle (2 * x) (2 * y)) (hy : y ≠ 0) :
  x^2 + (y - 3/2)^2 = 9 / 4 :=
sorry

end trajectory_midpoint_l312_312044


namespace ratio_of_smaller_to_bigger_l312_312237

theorem ratio_of_smaller_to_bigger (S B : ℕ) (h_bigger: B = 104) (h_sum: S + B = 143) :
  S / B = 39 / 104 := sorry

end ratio_of_smaller_to_bigger_l312_312237


namespace sum_first_10_terms_l312_312191

variable {a : ℕ → ℝ}

-- Conditions
def arithmetic_sequence_with_common_difference (d : ℕ → ℝ) : Prop :=
  ∃ a_1: ℝ, ∀ n: ℕ, a n = a_1 + n * 2

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Main theorem
theorem sum_first_10_terms (h1 : arithmetic_sequence_with_common_difference a)
  (h2 : geometric_sequence (a 2) (a 4) (a 8)) :
  (∑ i in finset.range 10, a i) = 110 := 
by
  sorry

end sum_first_10_terms_l312_312191


namespace ratio_max_min_2x_minus_y_l312_312171

theorem ratio_max_min_2x_minus_y (x y : ℝ) (h : y = 3 * real.sqrt (1 - x^2 / 4)) :
  let z := 2 * x - y in
  (z = 4 ∨ z = -5) → z ≠ 0 → 
  (set.to_rat (4 / -5)) = -4 / 5 :=
by
  let z := 2 * x - y
  intro hxy hnonzero
  sorry

end ratio_max_min_2x_minus_y_l312_312171


namespace num_divisors_of_36_l312_312493

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312493


namespace artifacts_per_wing_l312_312779

theorem artifacts_per_wing (total_wings : ℕ) (painting_wings : ℕ) 
    (large_paintings : ℕ) (small_paintings_per_wing : ℕ) 
    (artifact_ratio : ℕ) 
    (h_total_wings : total_wings = 8) 
    (h_painting_wings : painting_wings = 3) 
    (h_large_paintings : large_paintings = 1) 
    (h_small_paintings_per_wing : small_paintings_per_wing = 12) 
    (h_artifact_ratio : artifact_ratio = 4) :
    let total_paintings := large_paintings + small_paintings_per_wing * 2 in
    let total_artifacts := artifact_ratio * total_paintings in
    let artifact_wings := total_wings - painting_wings in
    total_artifacts / artifact_wings = 20 :=
by
    sorry

end artifacts_per_wing_l312_312779


namespace surface_area_of_inscribed_sphere_l312_312222

theorem surface_area_of_inscribed_sphere (V : ℝ) (hV : V = 8) : ∃ S : ℝ, S = 4 * Real.pi :=
by
  let a := (V)^(1/3)
  have ha : a = 2, from sorry
  let r := a / 2
  have hr : r = 1, from sorry
  let S := 4 * Real.pi * r^2
  use S
  rw [hr]
  simp

-- Intention is to show that the surface area S of the inscribed sphere is 4 * π, given the volume of the cube is 8.

end surface_area_of_inscribed_sphere_l312_312222


namespace matrix_not_invertible_x_l312_312837

theorem matrix_not_invertible_x (x : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 + x, 9], ![4 - x, 10]]
  A.det = 0 ↔ x = 16 / 19 := sorry

end matrix_not_invertible_x_l312_312837


namespace angle_B_max_area_l312_312893

theorem angle_B (A B C : ℝ) (a b c : ℝ) (h1 : B > 0) (h2 : B < π / 2) 
  (m : ℝ × ℝ) (n : ℝ × ℝ) (hm : m = (2 * sin B, -sqrt 3))
  (hn : n = (cos (2 * B), 2 * cos (B / 2) ^ 2 - 1)) :
  m.1 * n.2 + m.2 * n.1 = 0 → B = π / 3 := 
sorry

theorem max_area (A B C : ℝ) (a b c : ℝ) (h1 : B = π / 3) (h2 : b = 2) :
  let ac := a * c in
  ac ≤ 4 → 
  let S := 1/2 * ac * (sin B) in
  S ≤ sqrt 3 :=
sorry

end angle_B_max_area_l312_312893


namespace mrs_smith_children_l312_312621

noncomputable def balloon_number (ages : list ℕ) : ℕ := sorry

theorem mrs_smith_children (ages : list ℕ) (m : ℕ) (a b : ℕ) :
  -- Conditions
  (∀ age ∈ ages, age < 10) ∧
  distinct ages ∧
  3 ∈ ages ∧
  ∃ m, (m = a * 1001 + b * 110) ∧
  (∀ age ∈ ages, m % age = 0) ∧
  a * 10 + b = #) → 
  -- Conclusion
  8 ∉ ages :=
sorry

end mrs_smith_children_l312_312621


namespace concave_quadrilaterals_count_l312_312337

noncomputable def count_concave_quadrilaterals : ℕ :=
  366 

theorem concave_quadrilaterals_count (a b c d : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 59)
  (h₁ : 0 ≤ c ∧ c ≤ 59) (h₂ : 1 ≤ b ∧ b ≤ 5) (h₃ : 1 ≤ d ∧ d ≤ 5)
  (h_distinct : (a, b) ≠ (c, d) ∧ (a, b) ≠ (1,0) ∧ (c, d) ≠ (1,0)) :
  ∃ n, n = count_concave_quadrilaterals :=
  by
    use count_concave_quadrilaterals
    exact sorry

end concave_quadrilaterals_count_l312_312337


namespace max_n_for_factorable_quadratic_l312_312860

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l312_312860


namespace trapezoid_ratio_of_bases_l312_312659

theorem trapezoid_ratio_of_bases (a b : ℝ) (h : a > b)
    (H : (1 / 4) * (1 / 2) * (a - b) * h = (1 / 8) * (a + b) * h) : 
    a / b = 3 := 
sorry

end trapezoid_ratio_of_bases_l312_312659


namespace trapezoid_base_ratio_l312_312652

-- Define variables and conditions as per the problem
variables {a b h: ℝ}
def is_trapezoid (a b: ℝ) := a > b
def area_trapezoid (a b h: ℝ) := (1 / 2) * (a + b) * h
def area_quadrilateral (a b h: ℝ) := (1 / 2) * ((a - b) / 2) * (h / 2)

-- State the problem statement
theorem trapezoid_base_ratio (a b h: ℝ) (ha: a > b) (ht: area_quadrilateral a b h = (1 / 4) * area_trapezoid a b h) :
  a / b = 3 :=
sorry

end trapezoid_base_ratio_l312_312652


namespace shaded_area_l312_312107

-- Defining the conditions
def total_area_of_grid : ℕ := 38
def base_of_triangle : ℕ := 12
def height_of_triangle : ℕ := 4

-- Using the formula for the area of a right triangle
def area_of_unshaded_triangle : ℕ := (base_of_triangle * height_of_triangle) / 2

-- The goal: Prove the area of the shaded region
theorem shaded_area : total_area_of_grid - area_of_unshaded_triangle = 14 :=
by
  sorry

end shaded_area_l312_312107


namespace problem1_problem2_problem3_l312_312291

-- Problem 1
theorem problem1 (α : ℝ) (h : (4:ℝ, -3:ℝ) ∈ terminal_side_point α) :
  2 * Real.sin α + Real.cos α = -2 / 5 := sorry

-- Problem 2
theorem problem2 (α a : ℝ) (ha : a ≠ 0) (h : (4*a, -3*a) ∈ terminal_side_point α) :
  2 * Real.sin α + Real.cos α = 2 / 5 ∨ 2 * Real.sin α + Real.cos α = -2 / 5 := sorry

-- Problem 3
theorem problem3 (α : ℝ) (h : ∃ (a : ℝ), a ≠ 0 ∧ ((4*a, 3*a) ∈ terminal_side_point α ∨ (-4*a, 3*a) ∈ terminal_side_point α ∨ (-4*a, -3*a) ∈ terminal_side_point α ∨ (4*a, -3*a) ∈ terminal_side_point α)) :
  2 * Real.sin α + Real.cos α = 2 ∨ 2 * Real.sin α + Real.cos α = 2 / 5 ∨ 2 * Real.sin α + Real.cos α = -2 ∨ 2 * Real.sin α + Real.cos α = -2 / 5 := sorry

end problem1_problem2_problem3_l312_312291


namespace percentage_local_commerce_students_l312_312553

theorem percentage_local_commerce_students (x : ℝ) (h_arts : 200 = 0.50 * 400)
    (h_science : 25 = 0.25 * 100) (h_total : 200 + 25 + (x / 100) * 120 = 327) :
    x = 85 := by
  -- Given the conditions convert them to Lean 4
  have h_local_arts : 200 = 200 := eq.refl 200
  have h_local_science : 25 = 25 := eq.refl 25
  -- skip the actual proof
  sorry

end percentage_local_commerce_students_l312_312553


namespace symmetric_center_proof_l312_312387

noncomputable def symmetric_center (k : ℤ) : ℝ × ℝ :=
  (k * real.pi / 2 + real.pi / 6, 0)

theorem symmetric_center_proof : ∀ k : ℤ, 
  symmetric_center k = (k * real.pi / 2 + real.pi / 6, 0) := by
  sorry

end symmetric_center_proof_l312_312387


namespace carriages_per_train_l312_312700

variable (c : ℕ)

theorem carriages_per_train :
  (∃ c : ℕ, (25 + 10) * c * 3 = 420) → c = 4 :=
by
  sorry

end carriages_per_train_l312_312700


namespace maximal_n_for_quadratic_factorization_l312_312858

theorem maximal_n_for_quadratic_factorization :
  ∃ n, n = 325 ∧ (∃ A B : ℤ, A * B = 108 ∧ n = 3 * B + A) :=
by
  use 325
  use 1, 108
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num
  sorry

end maximal_n_for_quadratic_factorization_l312_312858


namespace number_of_divisors_of_36_l312_312507

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312507


namespace sum_distances_greater_than_n_l312_312890

theorem sum_distances_greater_than_n 
  {n : ℕ} (n_ge2 : n ≥ 3) 
  (A : fin n → ℂ) 
  (unit_circle : ∀ k, complex.abs (A k) = 1) 
  (regularity : ∀ k1 k2, A k1 ≠ A k2 → |A 0 - A k1| = |A 0 - A k2|) 
  (P : ℂ) 
  (P_on_circle : complex.abs P = 1) : 
  ∑ k in finset.range n, complex.abs (P - A k) > n :=
sorry

end sum_distances_greater_than_n_l312_312890


namespace range_of_a_is_0_to_1_over_2e_l312_312392

noncomputable def range_of_a : set ℝ :=
  {a : ℝ | ∃ x y : ℝ, x ≠ y ∧ e^x * (y - x) - a * e^(2*y - x) = 0}

theorem range_of_a_is_0_to_1_over_2e :
  range_of_a = {a : ℝ | 0 < a ∧ a < 1 / (2 * Real.exp 1)} :=
by
  sorry

end range_of_a_is_0_to_1_over_2e_l312_312392


namespace right_angled_triangle_l312_312174

namespace TriangleProof

variables {A B C : ℝ}  -- Angles of triangle ABC
variables {a b c : ℝ}  -- Sides of triangle ABC

-- Main statement: If the given condition holds, then the triangle is right-angled.
theorem right_angled_triangle (h : Real.cot (A / 2) = (b + c) / a) : 
  ∃ (R : ℝ) (x : Triangle R), x.right_angled :=
begin
  sorry,
end

end TriangleProof

end right_angled_triangle_l312_312174


namespace nell_total_cards_l312_312159

theorem nell_total_cards (initial_cards : ℝ) (jeff_cards : ℝ) : initial_cards = 304.5 → jeff_cards = 276.25 → initial_cards + jeff_cards = 580.75 :=
by
  intro h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end nell_total_cards_l312_312159


namespace scooter_cost_l312_312461

variable (saved needed total_cost : ℕ)

-- The conditions given in the problem
def greg_saved_57 : saved = 57 := sorry
def greg_needs_33_more : needed = 33 := sorry

-- The proof goal
theorem scooter_cost (h1 : saved = 57) (h2 : needed = 33) :
  total_cost = saved + needed → total_cost = 90 := by
  sorry

end scooter_cost_l312_312461


namespace trapezoid_base_ratio_l312_312649

-- Define variables and conditions as per the problem
variables {a b h: ℝ}
def is_trapezoid (a b: ℝ) := a > b
def area_trapezoid (a b h: ℝ) := (1 / 2) * (a + b) * h
def area_quadrilateral (a b h: ℝ) := (1 / 2) * ((a - b) / 2) * (h / 2)

-- State the problem statement
theorem trapezoid_base_ratio (a b h: ℝ) (ha: a > b) (ht: area_quadrilateral a b h = (1 / 4) * area_trapezoid a b h) :
  a / b = 3 :=
sorry

end trapezoid_base_ratio_l312_312649


namespace parallelogram_side_lengths_l312_312693

theorem parallelogram_side_lengths (x y : ℝ) (h1 : 3 * x + 6 = 12) (h2 : 5 * y - 2 = 10) : x + y = 22 / 5 :=
by
  sorry

end parallelogram_side_lengths_l312_312693


namespace ammonium_nitrate_formed_l312_312385

-- Definitions based on conditions in the problem
def NH3_moles : ℕ := 3
def HNO3_moles (NH3 : ℕ) : ℕ := NH3 -- 1:1 molar ratio with NH3 for HNO3

-- Definition of the outcome
def NH4NO3_moles (NH3 NH4NO3 : ℕ) : Prop :=
  NH4NO3 = NH3

-- The theorem to prove that 3 moles of NH3 combined with sufficient HNO3 produces 3 moles of NH4NO3
theorem ammonium_nitrate_formed (NH3 NH4NO3 : ℕ) (h : NH3 = 3) :
  NH4NO3_moles NH3 NH4NO3 → NH4NO3 = 3 :=
by
  intro hn
  rw [h] at hn
  exact hn

end ammonium_nitrate_formed_l312_312385


namespace proof_speed_of_man_in_still_water_l312_312774

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  50 / 4 = v_m + v_s ∧ 30 / 6 = v_m - v_s

theorem proof_speed_of_man_in_still_water (v_m v_s : ℝ) :
  speed_of_man_in_still_water v_m v_s → v_m = 8.75 :=
by
  intro h
  sorry

end proof_speed_of_man_in_still_water_l312_312774


namespace left_handed_fraction_proof_l312_312326

-- Define the conditions
variables (x : ℝ)
variable (y : ℝ := (8/5) * x)

-- Number of participants
def red_participants : ℝ := 3 * x
def blue_participants : ℝ := 2 * x
def green_participants : ℝ := y

-- Number of left-handed participants
def left_handed_red : ℝ := 1 / 3 * red_participants
def left_handed_blue : ℝ := 2 / 3 * blue_participants
def left_handed_green : ℝ := 0

-- Totals
def total_participants : ℝ := red_participants + blue_participants + green_participants
def total_left_handed : ℝ := left_handed_red + left_handed_blue + left_handed_green

-- Fraction of left-handed participants
def fraction_left_handed : ℝ := total_left_handed / total_participants

-- Theorem to prove
theorem left_handed_fraction_proof : fraction_left_handed = (35 / 99) := 
by
  sorry

end left_handed_fraction_proof_l312_312326


namespace divisors_of_36_count_l312_312512

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312512


namespace gcd_plus_lcm_18_30_45_l312_312597

noncomputable def gcd_18_30_45 : Nat := Nat.gcd (Nat.gcd 18 30) 45

noncomputable def lcm_18_30_45 : Nat := Nat.lcm (Nat.lcm 18 30) 45

theorem gcd_plus_lcm_18_30_45 : gcd_18_30_45 + lcm_18_30_45 = 93 := by
  have h_gcd : gcd_18_30_45 = 3 := by sorry
  have h_lcm : lcm_18_30_45 = 90 := by sorry
  rw [h_gcd, h_lcm]
  rfl

end gcd_plus_lcm_18_30_45_l312_312597


namespace find_QT_l312_312978

variable (P Q R S T : Type)
variable [metric_space P]
variable [metric_space Q]
variable [metric_space R]
variable [metric_space S]
variable [metric_space T]
variable (dist : P → Q → ℝ)
variable (RS PQ : ℝ)

-- Conditions
-- P Q R S are points in a convex quadrilateral
-- \overline{RS} ⊥ \overline{PQ}
-- \overline{PQ} ⊥ \overline{RS}
-- RS = 52
-- PQ = 39
-- Line through Q perpendicular to PS intersects PQ at T
-- PT = 25

def convex_quadrilateral (A B C D: P) : Prop :=
  sorry -- Definition of convex quadrilateral

def perpendicular (A B C : P) : Prop :=
  sorry -- Definition of perpendicularity

def line_through (A B: P) : P → Prop :=
  sorry -- Definition of the line through points

-- The main theorem
theorem find_QT
  (P Q R S T : P)
  (h_convex : convex_quadrilateral P Q R S)
  (h_perp_RS_PQ : perpendicular R S T)
  (h_perp_PQ_RS : perpendicular Q P R)
  (h_RS : dist R S = 52)
  (h_PQ : dist P Q = 39)
  (h_PT : dist P T = 25)
  (h_line_through_Q : ∀ (X : P), line_through Q X → perpendicular Q P S → X = T):
  dist Q T = 14 :=
sorry

end find_QT_l312_312978


namespace nonneg_reals_inequality_l312_312178

theorem nonneg_reals_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16 * a * b * c * d := 
by 
  sorry

end nonneg_reals_inequality_l312_312178


namespace range_of_f_max_omega_for_increasing_l312_312053

noncomputable def f (x ω : ℝ) : ℝ := 4 * cos(ω * x - π / 6) * sin(π - ω * x) - sin(2 * ω * x - π / 2)

theorem range_of_f (ω : ℝ) (hω : ω > 0) : 
  ∀ x, f x ω ∈ set.interval (1 - real.sqrt 3) (1 + real.sqrt 3) :=
sorry

theorem max_omega_for_increasing (h : ∀ x, -3 * real.pi / 2 ≤ x ∧ x ≤ real.pi / 2 → f x ω ≤ f (x + 1) ω) :
  ∃ ω, ω > 0 ∧ ω ≤ 1 / 6 := 
sorry

end range_of_f_max_omega_for_increasing_l312_312053


namespace max_S_l312_312798

noncomputable def f (t : ℕ) : ℕ := -2 * t + 200

noncomputable def g (t : ℕ) : ℕ :=
if h : t ≤ 30 then
  1/2 * t + 30
else
  40

noncomputable def S (t : ℕ) : ℕ :=
if h : t ≤ 30 then
  -t^2 + 40 * t + 6000
else
  -80 * t + 8000

theorem max_S : ∃ t : ℕ, 1 ≤ t ∧ t ≤ 50 ∧ S t = 6400 :=
by {
  use 20,
  have t1 := nat.le_refl 20,
  have t2 := nat.lt_of_le_and_ne (le_succ 20) (ne_of_lt (by exact dec_trivial)),
  exact ⟨dec_trivial, t1, t2, dec_trivial⟩,
  
  sorry }

end max_S_l312_312798


namespace find_x_l312_312674

theorem find_x (x : ℕ) (hcf lcm : ℕ):
  (hcf = Nat.gcd x 18) → 
  (lcm = Nat.lcm x 18) → 
  (lcm - hcf = 120) → 
  x = 42 := 
by
  sorry

end find_x_l312_312674


namespace heptagon_diagonals_l312_312466

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end heptagon_diagonals_l312_312466


namespace line_parallel_to_axis_of_symmetry_l312_312020

variable (A B C D O E F : Type)
variable [IsKite A B C D]
variable [IntersectionPoint O (Diagonals A B C D)]
variable [PerpendicularFrom O (Side A D) E]
variable [PerpendicularFrom O (Side D C) F]

theorem line_parallel_to_axis_of_symmetry
  (hO : ∃ A B C D, O = intersection_of_diagonals A B C D)
  (hE : E = perpendicular_from_point O (side AD))
  (hF : F = perpendicular_from_point O (side DC)) :
  parallel (line EF) (axis_of_symmetry A B C D) :=
sorry

end line_parallel_to_axis_of_symmetry_l312_312020


namespace different_tower_heights_l312_312166

-- Definitions of conditions
def bricks_count : Nat := 100
def brick_dimensions : List ℕ := [5, 12, 20]

-- Tuple representing (minimum increment, another increment)
def increments : List ℕ := [7, 15]
def minimum_possible_height : ℕ := bricks_count * 5
def maximum_possible_height : ℕ := bricks_count * 20

-- Problem statement
theorem different_tower_heights :
    ∃ heights : Set ℕ,
    heights = {x | minimum_possible_height ≤ x ∧ x ≤ maximum_possible_height ∧ 
               ∃ k j, x = minimum_possible_height + k * increments.head + j * increments.tail.head} ∧
    heights.size = 1417 :=
  by
    -- Placeholder for proof
    sorry

end different_tower_heights_l312_312166


namespace find_AE_l312_312678

noncomputable theory
open_locale classical

variables (A B C D L E : Type)
variables [point : HasPoints A B C D L E]
variables [rhombus : Rhombus A B C D]
variables (h_perp : IsPerpendicular B L A D)
variables (h_BL : ∥B - L∥ = 8)
variables (h_ratio : ∥A - L∥ / ∥L - D∥ = 3 / 2)
variables (h_E_on_AC : lies_on E (AC A C))

theorem find_AE : ∥A - E∥ = 3 * real.sqrt 5 :=
sorry

end find_AE_l312_312678


namespace smallest_k_divides_l312_312875

-- Conditions and definitions
def polynomial := (λ z : ℂ, z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1)

-- Main statement
theorem smallest_k_divides :
  ∃ k : ℕ, 0 < k ∧ (∀ z : ℂ, polynomial z ∣ z^k - 1) ∧ (∀ m : ℕ, 0 < m ∧ m < k → ¬(∀ z : ℂ, polynomial z ∣ z^m - 1)) := 
sorry

end smallest_k_divides_l312_312875


namespace trapezoid_base_ratio_l312_312665

variable {A B C D K M P Q L N : Type}
variable [LinearOrderedField A]
variable [LinearOrderedField B]
variable [LinearOrderedField C]
variable [LinearOrderedField D]

def is_trapezoid (A B C D : Type) (a b : ℝ) : Prop :=
  is_parallel (AD BC) ∧ AD.length = a ∧ BC.length = b ∧ a > b

def area (A B C D : Type) (a b : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

def quadrilateral_area (K L M N : Type) (a b h : ℝ) : ℝ :=
  (1 / 4) * (a - b) * h

theorem trapezoid_base_ratio (A B C D K M P Q L N : Type) (a b : ℝ) (h : ℝ) :
  is_trapezoid A B C D a b →
  quadrilateral_area K L M N a b h = (1 / 4) * area A B C D a b ↔ a / b = 3 :=
by 
  sorry

end trapezoid_base_ratio_l312_312665


namespace symmetric_line_equation_l312_312201

theorem symmetric_line_equation (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  (λ x, ax + b) = (λ y, y * a + b) ↔ (λ x, y = (1 / a) * x - (b / a)) :=
sorry

end symmetric_line_equation_l312_312201


namespace problem1_problem2_l312_312449

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x * f a x - x) / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 1 - (f a x - 1) / Real.exp x

theorem problem1 (x : ℝ) (h₁ : x ≥ 5) : g 1 x < 1 :=
sorry

theorem problem2 (a : ℝ) (h₂ : a > Real.exp 2 / 4) : 
∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ h a x1 = 0 ∧ h a x2 = 0 :=
sorry

end problem1_problem2_l312_312449


namespace trapezoid_base_ratio_l312_312643

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l312_312643


namespace divisors_of_36_count_l312_312511

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312511


namespace brendas_age_l312_312312

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l312_312312


namespace seats_in_nth_row_l312_312551

variable (m n : ℕ)

theorem seats_in_nth_row (m n : ℕ) : number_of_seats n = m + (n - 1) :=
sorry

end seats_in_nth_row_l312_312551


namespace find_AFplusAG_l312_312109

-- Conditions
variables (A B C D E F G : Point)
variables (AB AC : ℝ) (AD DB AE EC AF DF AG GE : ℝ)
variables (AFplusAG : ℝ)

-- Given data
def triangle_ABC (A B C : Point) : Prop := triangle A B C
def sides (AB AC : ℝ) : Prop := AB = 180 ∧ AC = 204
def points_on_AB (D F : Point) : Prop := (on_line A B D) ∧ (on_line A B F)
def points_on_AC (E G : Point) : Prop := (on_line A C E) ∧ (on_line A C G)
def equal_area_triangulation (A B C D E F G : Point) : Prop := 
  (triangle_area A D C = 4 * triangle_area D B C) ∧ 
  (triangle_area A D E = 3 * triangle_area E D C) ∧ 
  (triangle_area A F E = 2 * triangle_area F E D) ∧ 
  (triangle_area A F G = triangle_area G E F)

-- Required proof statement
theorem find_AFplusAG (h1 : triangle_ABC A B C)
  (h2 : sides AB AC)
  (h3 : points_on_AB D F)
  (h4 : points_on_AC E G)
  (h5 : equal_area_triangulation A B C D E F G)
  (h6 : AD = 144 ∧ DB = 36 ∧ AE = 153 ∧ EC = 51 ∧ AF = 96 ∧ DF = 48 ∧ AG = 76.5 ∧ GE = 76.5):
  AF + AG = 172.5 :=
sorry

end find_AFplusAG_l312_312109


namespace find_a_values_l312_312995

theorem find_a_values (a : ℝ) (h1 : -1 ≤ a)
                      (h2 : ∃ h : a > -1, true)
                      (A : set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ a })
                      (B : set ℝ := { y : ℝ | ∃ x, x ∈ A ∧ y = x + 1 })
                      (C : set ℝ := { y : ℝ | ∃ x, x ∈ A ∧ y = x^2 }) :
                      B = C ↔ (a = 0 ∨ a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2) := 
by {
  sorry
}

end find_a_values_l312_312995


namespace sin_75_eq_sqrt6_add_sqrt2_div4_l312_312695

theorem sin_75_eq_sqrt6_add_sqrt2_div4 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
sorry

end sin_75_eq_sqrt6_add_sqrt2_div4_l312_312695


namespace tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l312_312854

noncomputable def tangent_line (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_point_is_x_minus_y_plus_1_eq_0:
  tangent_line 0 = 1 →
  ∀ x y, y = tangent_line x → x - y + 1 = 0 → y = x * Real.exp x + 1 →
  x = 0 ∧ y = 1 → x - y + 1 = 0 :=
by
  intro h_point x y h_tangent h_eq h_coord
  sorry

end tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l312_312854


namespace train_speed_kmph_l312_312311

-- Definitions for the given conditions
def speed_mps : ℝ := 45.0036
def conversion_factor : ℝ := 3.6

-- The speed in kmph derived from the given conditions
def speed_kmph : ℝ := speed_mps * conversion_factor

-- The theorem stating the equivalence
theorem train_speed_kmph : speed_kmph = 162.013 :=
by {
  unfold speed_kmph,
  sorry
}

end train_speed_kmph_l312_312311


namespace no_real_solution_to_inequality_l312_312379

theorem no_real_solution_to_inequality (x : ℝ) :
  ∀ x ∈ ℝ, (Real.cbrt (x^2) + 4 / (Real.cbrt (x^2) + 2) ≤ 0) → False := 
by {
  sorry
}

end no_real_solution_to_inequality_l312_312379


namespace probability_dana_exactly_6_coins_at_end_of_fifth_round_l312_312318

-- Definition of initial conditions
structure GameState :=
    (alice_coins : ℕ)
    (bob_coins : ℕ)
    (carlos_coins : ℕ)
    (dana_coins : ℕ)
    (green_balls : ℕ)
    (red_balls : ℕ)
    (white_balls : ℕ)

def initial_state : GameState :=
    { alice_coins := 3,
      bob_coins := 5,
      carlos_coins := 4,
      dana_coins := 6,
      green_balls := 1,
      red_balls := 1,
      white_balls := 2 }

-- Game rules and ball addition mechanics
def update_game_state (state : GameState) : GameState := sorry

def end_of_fifth_round (initial : GameState) : GameState := sorry

-- The main theorem to be proved
theorem probability_dana_exactly_6_coins_at_end_of_fifth_round :
  (end_of_fifth_round initial_state).dana_coins = 6 → (7/768) :=
sorry

end probability_dana_exactly_6_coins_at_end_of_fifth_round_l312_312318


namespace prod_inequality_l312_312029

theorem prod_inequality (n : ℕ) (a b c d e : Fin n → ℝ)
  (h_a : ∀ i, 1 < a i) (h_b : ∀ i, 1 < b i) (h_c : ∀ i, 1 < c i) 
  (h_d : ∀ i, 1 < d i) (h_e : ∀ i, 1 < e i) :
  let A := (∑ i, a i) / n
      B := (∑ i, b i) / n
      C := (∑ i, c i) / n
      D := (∑ i, d i) / n
      E := (∑ i, e i) / n in 
  (∏ i, (a i * b i * c i * d i * e i + 1) / (a i * b i * c i * d i * e i - 1)) 
  ≥ ( (A * B * C * D * E + 1) / (A * B * C * D * E - 1) )^n :=
by 
  sorry

end prod_inequality_l312_312029


namespace find_k_l312_312908

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem find_k (k : ℝ) (h : (∑ i in Finset.range (6 + 1), (binomial 6 i) * (k ^ 2) ^ i *(by exact 1 ^' from 0) - 6 * над уровной math_french_revolution laundry notation - 1).coeffs[idx(math_γ_trick)]) = 240) :
  k = 2 ∨ k = -2 :=
by
  sorry

end find_k_l312_312908


namespace sin_cos_quad_ineq_l312_312176

open Real

theorem sin_cos_quad_ineq (x : ℝ) : 
  2 * (sin x) ^ 4 + 3 * (sin x) ^ 2 * (cos x) ^ 2 + 5 * (cos x) ^ 4 ≤ 5 :=
by
  sorry

end sin_cos_quad_ineq_l312_312176


namespace mrs_lee_grandsons_prob_l312_312158

open ProbabilityTheory

-- Define the probability that number of grandsons is not equal to number of granddaughters
def prob_mrs_lee_grandsons_ne_granddaughters : ℚ :=
  472305 / 531441

theorem mrs_lee_grandsons_prob (n : ℕ) (p : ℚ) (h_n : n = 12) (h_p : p = 2 / 3) :
  P(X ≠ 6) = prob_mrs_lee_grandsons_ne_granddaughters := by
  sorry

end mrs_lee_grandsons_prob_l312_312158


namespace determine_a_b_l312_312884

def set_A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def set_B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def A_union_B_is_R (a b : ℝ) : Prop := set_A ∪ set_B a b = set.univ
def A_inter_B_is_interval (a b : ℝ) : Prop := set_A ∩ set_B a b = Ioc 3 4

theorem determine_a_b (a b : ℝ) (h1 : A_union_B_is_R a b) (h2 : A_inter_B_is_interval a b) :
  a = -3 ∧ b = -4 :=
sorry

end determine_a_b_l312_312884


namespace sugar_price_difference_l312_312238

theorem sugar_price_difference (a b : ℝ) (h : (3 / 5 * a + 2 / 5 * b) - (2 / 5 * a + 3 / 5 * b) = 1.32) :
  a - b = 6.6 :=
by
  sorry

end sugar_price_difference_l312_312238


namespace middleton_soccer_team_geography_players_l312_312223

theorem middleton_soccer_team_geography_players :
  ∀ (players : Finset α) (H G : Finset α),
  players.card = 25 →
  H.card = 10 →
  (H ∩ G).card = 5 →
  (H ∪ G).card = 25 →
  G.card = 20 := by
  intros players H G h_players h_H h_HG h_union
  sorry

end middleton_soccer_team_geography_players_l312_312223


namespace arithmetic_sequence_propositions_l312_312417

theorem arithmetic_sequence_propositions (a_n : ℕ → ℤ) (S : ℕ → ℤ)
  (h_S_def : ∀ n, S n = n * (a_n 1 + (a_n (n - 1))) / 2)
  (h_cond : S 6 > S 7 ∧ S 7 > S 5) :
  (∃ d, d < 0 ∧ S 11 > 0) :=
by
  sorry

end arithmetic_sequence_propositions_l312_312417


namespace triangle_centroid_projections_sum_l312_312118

-- Defining the problem conditions and the final proof statement
theorem triangle_centroid_projections_sum :
  ∀ (A B C G P Q R : Type) 
    [has_dist A] [has_dist B] [has_dist C]
    [is_centroid_of G A B C]
    (AB : ℝ) (AC : ℝ) (BC : ℝ) (hAB : dist A B = 6) (hAC : dist A C = 10) (hBC : dist B C = 8)
    (proj_GP : dist G P) (proj_GQ : dist G Q) (proj_GR : dist G R)
    (hG_proj : is_projection_of_centroid G P Q R A B C)
    : proj_GP + proj_GQ + proj_GR = 94 / 15 :=
by
  sorry

end triangle_centroid_projections_sum_l312_312118


namespace divisors_of_36_l312_312472

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312472


namespace find_g_half_l312_312832

-- Define the conditions of the problem
def g : ℝ → ℝ
def zero_le_one : Prop := (0 : ℝ) ≤ 1
def g_zero : Prop := g 0 = 0
def non_decreasing (x y : ℝ) : Prop := 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y
def symmetry (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x
def scaling (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 3

-- The proof problem statement
theorem find_g_half (x : ℝ) (hx : x = 1 / 2) : 
  zero_le_one →
  g_zero →
  non_decreasing x (1 / 2) →
  symmetry x →
  scaling x →
  g x = 1 / 2 :=
by
  sorry  -- Proof is not provided, only the statement.

end find_g_half_l312_312832


namespace sequence_terms_and_sum_l312_312041

theorem sequence_terms_and_sum {a_n b_n : ℕ → ℝ}
  (h_all_pos : ∀ n, 0 < a_n n)
  (h_geom_cond1 : 2 * a_n 1 + 3 * a_n 2 = 1)
  (h_geom_cond2 : a_n 3 ^ 2 = 9 * a_n 2 * a_n 6)
  (h_arith_cond1 : b_n 2 = 0)
  (h_arith_cond2 : b_n 6 + b_n 8 = 10) :
  (∀ n, a_n n = (1/3) ^ n) ∧
  (∀ n, b_n n = n - 2) ∧
  (∀ n, let S_n := ∑ i in finset.range (n + 1), a_n i * b_n i 
   in S_n = -1/4 * (1 + (2 * n - 1) * (1/3) ^ n)) :=
by
  sorry

end sequence_terms_and_sum_l312_312041


namespace skipping_contest_mode_l312_312754

theorem skipping_contest_mode :
  let skips := [165, 165, 165, 165, 165, 170, 170, 145, 150, 150] in
  multiset.mode (multiset.of_list skips) = 165 :=
by
  sorry

end skipping_contest_mode_l312_312754


namespace num_divisors_36_l312_312495

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312495


namespace construct_triangle_l312_312340

noncomputable def exists_triangle_with_params (k a r : ℝ) : Prop :=
  ∃ (sides : ℝ × ℝ × ℝ), 
    let (b, c, d) := sides in 
    b + c + d = k ∧ (b = a ∨ c = a ∨ d = a) ∧ 
    let s := (b + c + a) / 2 in
    let A := sqrt (s * (s - a) * (s - b) * (s - c)) in
    A / s = r

theorem construct_triangle (k a r : ℝ) (hk : 0 < k) (ha : 0 < a) (hr : 0 < r) :
  exists_triangle_with_params k a r :=
sorry

end construct_triangle_l312_312340


namespace find_a_l312_312453

theorem find_a (a : ℝ) :
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ a = 1 ∨ a = 5/3 :=
by
  sorry

end find_a_l312_312453


namespace solution_is_x_l312_312847

def find_x (x : ℝ) : Prop :=
  64 * (x + 1)^3 - 27 = 0

theorem solution_is_x : ∃ x : ℝ, find_x x ∧ x = -1 / 4 :=
by
  sorry

end solution_is_x_l312_312847


namespace smallest_sum_of_24_consecutive_integers_is_perfect_square_l312_312215

theorem smallest_sum_of_24_consecutive_integers_is_perfect_square :
  ∃ n : ℕ, (n > 0) ∧ (m : ℕ) ∧ (2 * n + 23 = m^2) ∧ (12 * (2 * n + 23) = 300) :=
by
  sorry

end smallest_sum_of_24_consecutive_integers_is_perfect_square_l312_312215


namespace cos_neg_750_eq_sqrt3_div_2_l312_312835

theorem cos_neg_750_eq_sqrt3_div_2 :
  cos (-750 * (π / 180)) = (real.sqrt 3) / 2 :=
by
  sorry

end cos_neg_750_eq_sqrt3_div_2_l312_312835


namespace three_digit_integers_l312_312945

theorem three_digit_integers (n : ℕ) :
  (∃ (a b : ℕ), 6 ≤ a ∧ a ≤ 9 ∧ 6 ≤ b ∧ b ≤ 9 ∧ n = 100 * a + 10 * b + 5) →
  ∃ (c : ℕ), n = c ∧ 6 ≤ c / 100 ∧ c / 100 ≤ 9 ∧ 6 ≤ (c / 10) % 10 ∧ (c / 10) % 10 ≤ 9 ∧ c % 10 = 5 ∧ nat.divisible_by_5 c :=
  sorry

end three_digit_integers_l312_312945


namespace volume_of_rectangular_prism_l312_312233

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end volume_of_rectangular_prism_l312_312233


namespace common_tangents_l312_312683

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

def num_common_tangents (q1_center q2_center : ℝ × ℝ) (q1_radius q2_radius : ℝ) : ℕ :=
  let dist := real.sqrt ((q1_center.1 - q2_center.1)^2 + (q1_center.2 - q2_center.2)^2) in
  if dist > q1_radius + q2_radius then 4 else 0 -- simplified for our specific case

theorem common_tangents :
  num_common_tangents (0, 0) (3, 4) 3 1 = 4 :=
sorry

end common_tangents_l312_312683


namespace trigonometric_identity_l312_312731

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + Real.pi / 6) / Real.sin (4 * α - Real.pi / 6) :=
sorry

end trigonometric_identity_l312_312731


namespace find_x4_minus_x1_l312_312396

noncomputable def quadratic_functions (f g : ℝ → ℝ) : Prop :=
  (is_quadratic f) ∧ (is_quadratic g) ∧ 
  (∀ x, g x = -f (75 - x)) ∧ 
  (contains_vertex g f) ∧ 
  (∃ x1 x2 x3 x4 : ℝ, x1 < x2 < x3 < x4 ∧ x3 - x2 = 120)

def solution := 
  ∃ m n p : ℕ, p ≠ 1 ∧ (∀ prime, p mod (prime ^ 2) ≠ 0) ∧
  x4 - x1 = (m : ℝ) + n * real.sqrt p ∧
  m + n + p = 602

theorem find_x4_minus_x1 (f g : ℝ → ℝ) (h : quadratic_functions f g) :
  solution :=
  sorry

end find_x4_minus_x1_l312_312396


namespace no_real_roots_max_interval_l312_312706

theorem no_real_roots_max_interval (d : ℕ) :
  ∃ (I : Set ℝ), I = Set.Ioo 1 (1 + (1 : ℝ) / d)
    ∧ (∀ (a : Fin (2 * d) → ℝ),
       (∀ i, a i ∈ I) →
       ∀ x : ℝ, Polynomial.eval x (Polynomial.monomial (2 * d) 1 +
               ∑ i in Finset.range (2 * d), Polynomial.monomial i (a i)) ≠ 0) :=
sorry

end no_real_roots_max_interval_l312_312706


namespace travel_paths_l312_312205

-- Definitions for conditions
def roads_AB : ℕ := 3
def roads_BC : ℕ := 2

-- The theorem statement
theorem travel_paths : roads_AB * roads_BC = 6 := by
  sorry

end travel_paths_l312_312205


namespace num_divisors_36_l312_312494

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312494


namespace abs_condition_iff_range_l312_312673

theorem abs_condition_iff_range (x : ℝ) : 
  (|x-1| + |x+2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 2) := 
sorry

end abs_condition_iff_range_l312_312673


namespace smallest_sum_of_24_consecutive_integers_is_perfect_square_l312_312214

theorem smallest_sum_of_24_consecutive_integers_is_perfect_square :
  ∃ n : ℕ, (n > 0) ∧ (m : ℕ) ∧ (2 * n + 23 = m^2) ∧ (12 * (2 * n + 23) = 300) :=
by
  sorry

end smallest_sum_of_24_consecutive_integers_is_perfect_square_l312_312214


namespace apples_in_box_l312_312226

theorem apples_in_box (total_fruit : ℕ) (one_fourth_oranges : ℕ) (half_peaches_oranges : ℕ) (apples_five_peaches : ℕ) :
  total_fruit = 56 →
  one_fourth_oranges = total_fruit / 4 →
  half_peaches_oranges = one_fourth_oranges / 2 →
  apples_five_peaches = 5 * half_peaches_oranges →
  apples_five_peaches = 35 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end apples_in_box_l312_312226


namespace z_in_fourth_quadrant_l312_312432

noncomputable def z : ℂ := (3 * Complex.I - 2) / (Complex.I - 1) * Complex.I

theorem z_in_fourth_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_fourth_quadrant_l312_312432


namespace ceiling_of_neg_3_7_l312_312369

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l312_312369


namespace find_x_l312_312751

-- Definitions of conditions
def initialSpeed (V : ℝ) : ℝ := V
def reducedSpeed (V : ℝ) (x : ℝ) : ℝ := V * (1 - x / 100)
def increasedSpeed (V : ℝ) (x : ℝ) : ℝ := (reducedSpeed V x) * (1 + 0.5 * x / 100)
def finalSpeed (V : ℝ) (x : ℝ) : ℝ := V * (1 - 0.6 * x / 100)

-- Main statement of the proof problem
theorem find_x (V : ℝ) : ∃ x : ℝ, increasedSpeed V x = finalSpeed V x ∧ x = 20 := by
  sorry

end find_x_l312_312751


namespace find_m_l312_312456

variables {m : ℝ}
def a := (1, m)
def b := (m - 1, 2)
def orthogonal (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0
def not_equal (u v : ℝ × ℝ) := u ≠ v

theorem find_m (h1: orthogonal (a.1 - b.1, a.2 - b.2) a) (h2: not_equal a b) : m = 1 :=
sorry

end find_m_l312_312456


namespace count_odd_functions_l312_312805

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

def f1 (x : ℝ) : ℝ := x ^ 3
def f2 (x : ℝ) : ℝ := 2 ^ x
def f3 (x : ℝ) : ℝ := x ^ 2 + 1
def f4 (x : ℝ) : ℝ := 2 * sin x

theorem count_odd_functions : 
  (if is_odd_function f1 then 1 else 0) + 
  (if is_odd_function f2 then 1 else 0) + 
  (if is_odd_function f3 then 1 else 0) + 
  (if is_odd_function f4 then 1 else 0) = 2 := 
sorry

end count_odd_functions_l312_312805


namespace trapezoid_base_ratio_l312_312654

theorem trapezoid_base_ratio
  (a b h : ℝ)  -- lengths of the bases and height
  (ha_gt_hb : a > b)
  (trapezoid_area : ℝ) 
  (quad_area : ℝ) 
  (h1 : trapezoid_area = (1/2) * (a + b) * h) 
  (h2 : quad_area = (1/2) * (a - b) * h / 4)
  (h3 : quad_area = trapezoid_area / 4) :
  a / b = 3 :=
by {
  sorry,
}

end trapezoid_base_ratio_l312_312654


namespace distinct_integer_sums_of_special_fractions_l312_312344

def is_special_fraction (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 21

def special_fractions : List (ℕ × ℕ) :=
  [(1, 20), (2, 19), (3, 18), (4, 17), (5, 16), (6, 15), (7, 14), (8, 13), (9, 12), (10, 11),
   (11, 10), (12, 9), (13, 8), (14, 7), (15, 6), (16, 5), (17, 4), (18, 3), (19, 2), (20, 1)]

def sum_two_special_fractions (s : List (ℕ × ℕ)) : List ℤ :=
  List.map (λ p, p.1 / p.2) s

theorem distinct_integer_sums_of_special_fractions :
  (List.eraseDup (List.map (λ (p q : (ℕ × ℕ)), 
    (p.1 / p.2) + (q.1 / q.2)) (List.product special_fractions special_fractions))).length = 11 :=
by
  sorry

end distinct_integer_sums_of_special_fractions_l312_312344


namespace rectangular_to_polar_3_3_l312_312830

def convert_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := real.arctan (y / x)
  (r, θ)

theorem rectangular_to_polar_3_3 :
  convert_to_polar 3 3 = (3 * real.sqrt 2, real.pi / 4) :=
by
  sorry

end rectangular_to_polar_3_3_l312_312830


namespace task_force_combinations_l312_312762

theorem task_force_combinations :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end task_force_combinations_l312_312762


namespace percentage_difference_is_20_l312_312155

/-
Given:
Height of sunflowers from Packet A = 192 inches
Height of sunflowers from Packet B = 160 inches

Show:
Percentage difference in height between Packet A and Packet B is 20%.
-/

-- Definitions of heights
def height_packet_A : ℤ := 192
def height_packet_B : ℤ := 160

-- Definition of percentage difference formula
def percentage_difference (hA hB : ℤ) : ℤ := ((hA - hB) * 100) / hB

-- Theorem statement
theorem percentage_difference_is_20 :
  percentage_difference height_packet_A height_packet_B = 20 :=
sorry

end percentage_difference_is_20_l312_312155


namespace base_ten_equivalent_l312_312242

theorem base_ten_equivalent : 
  let val := 5 * 8^4 + 2 * 8^3 + 6 * 8^2 + 4 * 8^1 + 3 * 8^0 in
  val = 21923 := by
  let val := 5 * 8^4 + 2 * 8^3 + 6 * 8^2 + 4 * 8^1 + 3 * 8^0
  show val = 21923
  sorry

end base_ten_equivalent_l312_312242


namespace square_diagonal_length_l312_312720

theorem square_diagonal_length (rect_length rect_width : ℝ) 
  (h1 : rect_length = 45) 
  (h2 : rect_width = 40) 
  (rect_area := rect_length * rect_width) 
  (square_area := rect_area) 
  (side_length := Real.sqrt square_area) 
  (diagonal := side_length * Real.sqrt 2) :
  diagonal = 60 :=
by
  -- Proof goes here
  sorry

end square_diagonal_length_l312_312720


namespace constant_term_correct_l312_312194

noncomputable def constant_term_in_expansion : ℤ :=
  let f := (1 + a + a^2) * (a - 1/a)^6
  f.coeff 0

theorem constant_term_correct :
  constant_term_in_expansion = -5 := 
sorry

end constant_term_correct_l312_312194


namespace largest_n_for_factored_polynomial_l312_312866

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l312_312866


namespace manager_salary_proof_l312_312192

noncomputable def manager_salary 
    (avg_salary_without_manager : ℝ) 
    (num_employees_without_manager : ℕ) 
    (increase_in_avg_salary : ℝ) 
    (new_total_salary : ℝ) : ℝ :=
    new_total_salary - (num_employees_without_manager * avg_salary_without_manager)

theorem manager_salary_proof :
    manager_salary 3500 100 800 (101 * (3500 + 800)) = 84300 :=
by
    sorry

end manager_salary_proof_l312_312192


namespace minimum_length_MX_l312_312628

theorem minimum_length_MX {A B C M O X : Point} (hM : midpoint B C M) 
  (hAB : dist A B = 17) (hAC : dist A C = 30) (hBC : dist B C = 19) 
  (circle_with_diam_AB : Circle A B X) : 
  ∃ X, X ∈ circle_with_diam_AB ∧ dist M X = 6.5 :=
begin
  sorry
end

end minimum_length_MX_l312_312628


namespace triangle_areas_inequality_l312_312984

-- Define a triangle as vertices A, B, C in a 2D plane.
variables {A B C E F : Type*} [affine_space ℝ]
variables [noncomputable] A B C D : E := midpoint ℝ A B
variables (E : E) (F : E)
variable (AC BC : line_segment ℝ A C) (BC : line_segment ℝ B C)

-- Theorem statement
theorem triangle_areas_inequality (hD : D = midpoint ℝ A B) (hE : point_on_line_segment ℝ E AC) (hF : point_on_line_segment ℝ F BC) :
  area (triangle D E F) ≤ area (triangle A E D) + area (triangle B F D) :=
sorry

end triangle_areas_inequality_l312_312984


namespace final_position_total_distance_l312_312797

-- Define the movements as a list
def movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

-- Prove that the final position of the turtle is 5 meters north of the starting point
theorem final_position (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum movements = 5 :=
by
  rw [h]
  sorry

-- Prove that the total distance crawled by the turtle is 47 meters
theorem total_distance (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum (List.map Int.natAbs movements) = 47 :=
by
  rw [h]
  sorry

end final_position_total_distance_l312_312797


namespace brother_spent_on_highlighters_l312_312939

theorem brother_spent_on_highlighters : 
  let total_money := 100
  let cost_sharpener := 5
  let num_sharpeners := 2
  let cost_notebook := 5
  let num_notebooks := 4
  let cost_eraser := 4
  let num_erasers := 10
  let total_spent_sharpeners := num_sharpeners * cost_sharpener
  let total_spent_notebooks := num_notebooks * cost_notebook
  let total_spent_erasers := num_erasers * cost_eraser
  let total_spent := total_spent_sharpeners + total_spent_notebooks + total_spent_erasers
  let remaining_money := total_money - total_spent
  remaining_money = 30 :=
begin
  sorry
end

end brother_spent_on_highlighters_l312_312939


namespace points_lie_on_hyperbola_l312_312394

-- Define the hyperbolic cosine function
def cosh (s : ℝ) : ℝ := (Real.exp s + Real.exp (-s)) / 2

-- Define the hyperbolic sine function
def sinh (s : ℝ) : ℝ := (Real.exp s - Real.exp (-s)) / 2

-- Define the variables x and y in terms of s
def x (s : ℝ) : ℝ := 2 * cosh s
def y (s : ℝ) : ℝ := 4 * sinh s

-- Define the theorem to be proven
theorem points_lie_on_hyperbola : ∀ s : ℝ, (x s)^2 - (y s)^2 / 8 = 1 := by
  sorry

end points_lie_on_hyperbola_l312_312394


namespace find_k_intersecting_lines_l312_312388

theorem find_k_intersecting_lines : 
  ∃ (k : ℚ), (∃ (x y : ℚ), y = 6 * x + 4 ∧ y = -3 * x - 30 ∧ y = 4 * x + k) ∧ k = -32 / 9 :=
by
  sorry

end find_k_intersecting_lines_l312_312388


namespace trapezoid_base_ratio_l312_312646

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l312_312646


namespace slower_speed_l312_312791

theorem slower_speed (x : ℝ) :
  (5 * (24 / x) = 24 + 6) → x = 4 := 
by
  intro h
  sorry

end slower_speed_l312_312791


namespace no_equal_sums_possible_l312_312574

theorem no_equal_sums_possible :
  ¬ (∃ (l1 l2 l3 l4 l5 : list ℕ),
    l1.length = 4 ∧ l2.length = 4 ∧ l3.length = 4 ∧ l4.length = 4 ∧ l5.length = 4 ∧
    (∀ l ∈ [l1, l2, l3, l4, l5], ∀ x ∈ l, x ∈ [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]) ∧
    sum l1 = sum l2 ∧ sum l2 = sum l3 ∧ sum l3 = sum l4 ∧ sum l4 = sum l5 ∧ 
    sum (l1 ++ l2 ++ l3 ++ l4 ++ l5) = 2 * (4 * 1 + 3 * 2 + 3 * 3) / 5) :=
sorry

end no_equal_sums_possible_l312_312574


namespace area_of_union_of_triangles_l312_312411

-- Define the vertices of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (7, 3)

-- Define the reflection function across the line x=5
def reflect_x5 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (10 - x, y)

-- Define the vertices of the reflected triangle
def A' : ℝ × ℝ := reflect_x5 A
def B' : ℝ × ℝ := reflect_x5 B
def C' : ℝ × ℝ := reflect_x5 C

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the union of both triangles is 22
theorem area_of_union_of_triangles : triangle_area A B C + triangle_area A' B' C' = 22 := by
  sorry

end area_of_union_of_triangles_l312_312411


namespace smallest_triangle_perimeter_consecutive_even_l312_312263

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l312_312263


namespace hyperbola_eqn_l312_312924

theorem hyperbola_eqn 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (circle : ∀ x y : ℝ, x^2 + y^2 - 6 * x + 5 = 0)
  (asympt_tangent : ∀ x y : ℝ, (y = (b / a) * x ∨ y = -(b / a) * x) →
                    (x^2 + y^2 - 6 * x + 5 = 0))
  (focus_center : (3, 0) = (3, 0)) :
  ∃ (a b : ℝ), a = √5 ∧ b = 2 ∧ (∀ x y : ℝ, x^2 / 5 - y^2 / 4 = 1) :=
by
  sorry

end hyperbola_eqn_l312_312924


namespace average_inside_time_l312_312136

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end average_inside_time_l312_312136


namespace brother_highlighter_spending_l312_312937

variables {total_money : ℝ} (sharpeners notebooks erasers highlighters : ℝ)

-- Conditions
def total_given := total_money = 100
def sharpeners_cost := 2 * 5
def notebooks_cost := 4 * 5
def erasers_cost := 10 * 4
def heaven_expenditure := sharpeners_cost + notebooks_cost
def brother_remaining := total_money - (heaven_expenditure + erasers_cost)
def brother_spent_on_highlighters := brother_remaining = 30

-- Statement
theorem brother_highlighter_spending (h1 : total_given) 
    (h2 : brother_spent_on_highlighters) : brother_remaining = 30 :=
sorry

end brother_highlighter_spending_l312_312937


namespace brendas_age_l312_312313

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l312_312313


namespace area_of_triangle_BCD_l312_312996

-- Define points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A, B, C, D
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨2*a, 0, 0⟩
def C (b : ℝ) : Point3D := ⟨0, 3*b, 0⟩
def D (c : ℝ) : Point3D := ⟨0, 0, 4*c⟩

-- Given conditions
def edges_perpendicular := ∀ (a b c : ℝ), ∃ (A B C D : Point3D),
  -- The points are defined
  A = ⟨0,0,0⟩ ∧
  B = ⟨2*a,0,0⟩ ∧
  C = ⟨0,3*b,0⟩ ∧
  D = ⟨0,0,4*c⟩ ∧
  -- The length conditions on the edges are perpendicular at A
  A.x * B.x + A.y * B.y + A.z * B.z = 0 ∧
  A.x * C.x + A.y * C.y + A.z * C.z = 0 ∧
  A.x * D.x + A.y * D.y + A.z * D.z = 0

-- Define areas of triangles as given
def area_triangle_ABC (a : ℝ) := a^2
def area_triangle_ACD (b : ℝ) := 9*b^2
def area_triangle_ADB (c : ℝ) := 8*c^2

-- Prove the area of triangle BCD is as expected
theorem area_of_triangle_BCD (a b c : ℝ) : 
  edges_perpendicular a b c →
  area_triangle_ABC a = a^2 →
  area_triangle_ACD b = 9*b^2 →
  area_triangle_ADB c = 8*c^2 →
  ∃ K : ℝ, K = 3*b*Real.sqrt(a^2 + 4*c^2) := by
  sorry

end area_of_triangle_BCD_l312_312996


namespace division_of_factorials_l312_312246

-- Definitions from conditions
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def diff (a b : ℕ) : ℕ := a - b

-- The proof problem statement
theorem division_of_factorials (a b : ℕ) (ha : a = 8) (hb : b = 2) :
  (factorial a) / (factorial (diff a b)) = 56 := by
  sorry

end division_of_factorials_l312_312246


namespace artifacts_in_each_wing_l312_312778

theorem artifacts_in_each_wing (total_wings : ℕ) (artifact_factor : ℕ) (painting_wings : ℕ) 
  (large_painting : ℕ) (small_paintings_per_wing : ℕ) (remaining_artifact_wings : ℕ) 
  (total_paintings constant_total_paintings_expected : ℕ) (total_artifacts total_artifacts_expected : ℕ) 
  (artifacts_per_wing artifacts_per_wing_expected : ℕ) : 

  total_wings = 8 →
  artifact_factor = 4 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  remaining_artifact_wings = total_wings - painting_wings →
  total_paintings = painting_wings * small_paintings_per_wing + large_painting →
  total_artifacts = total_paintings * artifact_factor →
  artifacts_per_wing = total_artifacts / remaining_artifact_wings →
  artifacts_per_wing = 20 :=

by
    intros htotal_wings hartifact_factor hpainting_wings hlarge_painting hsmall_paintings_per_wing hermaining_artifact_wings htotal_paintings htotal_artifacts hartifacts_per_wing,
    sorry

end artifacts_in_each_wing_l312_312778


namespace problem_statement_l312_312905

noncomputable def x : ℂ := sorry

theorem problem_statement (h : x - 1/x = 2 * complex.I) : 
    x^8000 - 1/x^8000 = 0 :=
by
  sorry

end problem_statement_l312_312905


namespace R_is_not_polynomial_l312_312631

def R (X : ℝ) : ℝ := real.sqrt (abs X)

theorem R_is_not_polynomial : ¬ ∃ (P : ℝ → ℝ), polynomial ℝ P ∧ (∀ X : ℝ, R X = P X) :=
by
  sorry

end R_is_not_polynomial_l312_312631


namespace right_triangle_count_l312_312612

theorem right_triangle_count (p : ℕ) (hp : Nat.Prime p) :
  ∃ (n : ℕ), (n = 36 ∧ p ≠ 2 ∧ p ≠ 997) ∨
             (n = 18 ∧ p = 2) ∨
             (n = 20 ∧ p = 997) :=
begin
  sorry,
end

end right_triangle_count_l312_312612


namespace roots_purely_imaginary_l312_312828

open Complex

/-- 
  If m is a purely imaginary number, then the roots of the equation 
  8z^2 + 4i * z - m = 0 are purely imaginary.
-/
theorem roots_purely_imaginary (m : ℂ) (hm : m.im ≠ 0 ∧ m.re = 0) : 
  ∀ z : ℂ, 8 * z^2 + 4 * Complex.I * z - m = 0 → z.im ≠ 0 ∧ z.re = 0 :=
by
  sorry

end roots_purely_imaginary_l312_312828


namespace longest_diagonal_length_l312_312308

theorem longest_diagonal_length (A : ℝ) (r : ℝ) (k : ℝ) (long_d : ℝ) (short_d : ℝ):
  A = 192 ∧ r = 4/3 ∧ A = 1/2 * long_d * short_d ∧ long_d = k * 4 ∧ short_d = k * 3 → 
  long_d = 16 * sqrt 2 :=
by
  sorry

end longest_diagonal_length_l312_312308


namespace series_sum_l312_312817

theorem series_sum :
  let s := λ n, if n % 4 == 1 then n else if n % 4 == 2 then -n else if n % 4 == 3 then -n else n
  (Finset.sum (Finset.range 2002) s) = 2001 :=
by
  sorry

end series_sum_l312_312817


namespace problem1_l312_312434

theorem problem1 (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  (a - 2) * (b - 2) = 2 :=
sorry

end problem1_l312_312434


namespace find_a_l312_312443

noncomputable def f (x a : ℝ) : ℝ := x / (x^2 + a)

theorem find_a (a : ℝ) (h_positive : a > 0) (h_max : ∀ x, x ∈ Set.Ici 1 → f x a ≤ f 1 a) :
  a = Real.sqrt 3 - 1 := by
  sorry

end find_a_l312_312443


namespace remainder_2457634_div_8_l312_312245

theorem remainder_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end remainder_2457634_div_8_l312_312245


namespace largest_lucky_number_l312_312932

theorem largest_lucky_number : 
  let a := 1
  let b := 4
  let lucky_number (x y : ℕ) := x + y + x * y
  let c1 := lucky_number a b
  let c2 := lucky_number b c1
  let c3 := lucky_number c1 c2
  c3 = 499 :=
by
  sorry

end largest_lucky_number_l312_312932


namespace balance_pitcher_with_saucers_l312_312607

-- Define the weights of the cup (C), pitcher (P), and saucer (S)
variables (C P S : ℝ)

-- Conditions provided in the problem
axiom cond1 : 2 * C + 2 * P = 14 * S
axiom cond2 : P = C + S

-- The statement to prove
theorem balance_pitcher_with_saucers : P = 4 * S :=
by
  sorry

end balance_pitcher_with_saucers_l312_312607


namespace doubling_n_constant_C_l312_312338

theorem doubling_n_constant_C (e n R r : ℝ) (h_pos_e : 0 < e) (h_pos_n : 0 < n) (h_pos_R : 0 < R) (h_pos_r : 0 < r)
  (C : ℝ) (hC : C = e^2 * n / (R + n * r^2)) :
  C = (2 * e^2 * n) / (R + 2 * n * r^2) := 
sorry

end doubling_n_constant_C_l312_312338


namespace hyperbola_point_A_l312_312086

def point (x y : ℝ) := (x, y)

section
open_locale real

variables (a b : ℝ) (h_ab : a > 0 ∧ b > 0)
variables (M A : ℝ × ℝ) (P F: ℝ × ℝ)

def hyperbola_eq := x^2 / a^2 - y^2 / b^2 = 1

def M := point (-1) (sqrt 3)

-- Assume M is symmetric with respect to the other asymptote to right focus F
def focus := point 2 0

def moving_point_on_hyperbola (P := point x y) :=
  P satisfies hyperbola_eq 

def point_A := point 3 1

def distance (p1 p2 : ℝ × ℝ) := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def PA := distance P point_A

def PF := distance P focus

theorem hyperbola_point_A (
  h1 : M = point (-1) (sqrt 3),
  h2 : (P satisfies hyperbola_eq),
  h3 : point_A = point 3 1,
  h4 : focus = point 2 0 ) :
  PA + 1/2 * PF = 5 / 2 :=
begin
  sorry -- Proof to be filled
end

end hyperbola_point_A_l312_312086


namespace exists_d_iff_n_eq_product_of_distinct_primes_smallest_d_for_x_powers_mod_n_l312_312273

open Nat Int

section PartA

theorem exists_d_iff_n_eq_product_of_distinct_primes (n : ℕ) (h : n ≥ 2) :
  (∃ d : ℕ, ∀ x : ℤ, x^(d + 1) ≡ x [MOD n]) ↔
  ∃ (p : ℕ → ℕ) (k : ℕ), (∀ i j, i ≠ j → p i ≠ p j) ∧ n = ∏ i in finRange k, p i :=
by
  sorry

end PartA

section PartB

theorem smallest_d_for_x_powers_mod_n (p : ℕ → ℕ) (k : ℕ) (hdistinct : ∀ i j, i ≠ j → p i ≠ p j) :
  let n := ∏ i in finRange k, p i
  ∀ d : ℕ, (∀ x : ℤ, x^(d + 1) ≡ x [MOD n]) ↔ d = lcm (finRange k).map (λ i => p i - 1) :=
by
  sorry

end PartB

end exists_d_iff_n_eq_product_of_distinct_primes_smallest_d_for_x_powers_mod_n_l312_312273


namespace find_area_of_triangle_KLM_l312_312732

open Real

noncomputable def area_of_triangle_KLM 
  (O S A B C K L M : Point)
  (sphere : Sphere)
  (pyramid SABC : Pyramid)
  (radius R : ℝ) 
  (alpha : ℝ) 
  (cross_section_area : ℝ)
  (angle_KSO : ℝ) 
  (SO : ℝ) 
  (planes_parallel : Plane)
  (triangle_area_KLM : ℝ) : Prop :=
sphere.touches S A B C &&
sphere.touches_edges S A K && 
sphere.touches_edges S B L && 
sphere.touches_edges S C M &&
cross_section_area = 2 &&
angle_KSO = arccos (sqrt 7 / 4) &&
SO = 9 &&
planes_parallel K L M A B C &&
triangle_area_KLM = 49 / 8

theorem find_area_of_triangle_KLM
  (O S A B C K L M : Point)
  (sphere : Sphere)
  (pyramid_SABC : Pyramid)
  (R alpha cross_section_area angle_KSO SO : ℝ)
  (planes_parallel : Plane) :
  sphere.touches S A B C →
  sphere.touches_edges S A K →
  sphere.touches_edges S B L →
  sphere.touches_edges S C M →
  cross_section_area = 2 →
  angle_KSO = arccos (sqrt 7 / 4) →
  SO = 9 →
  planes_parallel K L M A B C →
  ∃ (triangle_area_KLM : ℝ), 
    triangle_area_KLM = 49 / 8 :=
sorry

end find_area_of_triangle_KLM_l312_312732


namespace divisors_of_36_l312_312471

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312471


namespace martian_angle_measure_obtuse_l312_312081

theorem martian_angle_measure_obtuse (h1 : 800 = 800) (h2 : 360 = 360) : 
  let full_circle_clerts := 800
  let full_circle_degrees := 360
  let obtuse_angle_degrees := 120
  let angle_fraction := obtuse_angle_degrees / full_circle_degrees
  let obtuse_angle_clerts := angle_fraction * full_circle_clerts
  in obtuse_angle_clerts = 267 :=
by
  sorry

end martian_angle_measure_obtuse_l312_312081


namespace smallest_triangle_perimeter_consecutive_even_l312_312262

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l312_312262


namespace exists_always_white_cell_l312_312128

-- Assume there is an infinite grid plane where initially some finite number of cells are painted black.
-- We have a grid polygon M that can cover more than one cell.
-- M can be shifted in any direction on the grid without rotating.
-- If after a shift, exactly one cell of M lies on a white cell, then that white cell is painted black.
-- Prove that there exists at least one cell that will always remain white no matter how many shifts of M are performed.

theorem exists_always_white_cell
  (grid : ℤ × ℤ → Bool)  -- Infinite grid plane, true for black cells
  (initial_black_cells : Finset (ℤ × ℤ))  -- Initially painted black cells
  (M : Finset (ℤ × ℤ))  -- Grid polygon M covering more than one cell
  (hM : 1 < M.card)  -- M covers more than one cell
  (shift : ℤ × ℤ → Finset (ℤ × ℤ) → Finset (ℤ × ℤ))  -- Shifting function
  (h_shift : ∀ s : ℤ × ℤ, shift s M ⊆ grid)  -- Shifting preserves grid alignment
  :
  ∃ (white_cell : ℤ × ℤ), ∀ (shift_seq : List (ℤ × ℤ)), ¬(shift_seq.foldl (λ b s, shift s b) initial_black_cells) white_cell = true := 
sorry

end exists_always_white_cell_l312_312128


namespace value_of_f_at_pi_over_4_l312_312056

-- Given function f and its derivative conditions
def f (x : ℝ) : ℝ := f' (π / 4) * Real.cos x + Real.sin x

-- Required to find the value of f(π / 4)
theorem value_of_f_at_pi_over_4 : f (π / 4) = 1 := by
  sorry

end value_of_f_at_pi_over_4_l312_312056


namespace vector_dot_product_n_l312_312459

theorem vector_dot_product_n (n : ℚ) : 
  let a := (-1, 1 : ℚ × ℚ)
  let b := (n, 2 : ℚ × ℚ)
  a.1 * b.1 + a.2 * b.2 = 5 / 3 → 
  n = 1 / 3 := 
by
  intro h
  sorry

end vector_dot_product_n_l312_312459


namespace proof_problem_l312_312816

theorem proof_problem :
  (real.sqrt 3 + 2)^2023 * (real.sqrt 3 - 2)^2023 = -1 := 
sorry

end proof_problem_l312_312816


namespace ratio_of_speeds_l312_312287

-- Define the constants and variables
def A_speed_ratio_B_speed (v_A v_B : ℝ) : Prop :=
  (3 * v_A = Real.abs (-600 + 3 * v_B)) ∧ 
  (12 * v_A = Real.abs (-600 + 12 * v_B)) → 
  (v_A / v_B = 4 / 5)

-- State the main theorem
theorem ratio_of_speeds (v_A v_B : ℝ) :
  (A_speed_ratio_B_speed v_A v_B) :=
by {
  sorry
}

end ratio_of_speeds_l312_312287


namespace sequence_sum_2011_l312_312112

noncomputable def sequence : ℕ → ℝ
| 0     := 2  -- As we are given a_2011 = 2, we index it from a_0, implying a_2011 corresponds to a_0 in our definition.
| (n+1) := 1 - 1 / sequence n

theorem sequence_sum_2011 :
  (∑ i in Finset.range 2011, sequence i) = 1007 :=
sorry

end sequence_sum_2011_l312_312112


namespace archibald_percentage_wins_l312_312811

def archibald_wins : ℕ := 12
def brother_wins : ℕ := 18
def total_games_played : ℕ := archibald_wins + brother_wins

def percentage_archibald_wins : ℚ := (archibald_wins : ℚ) / (total_games_played : ℚ) * 100

theorem archibald_percentage_wins : percentage_archibald_wins = 40 := by
  sorry

end archibald_percentage_wins_l312_312811


namespace tangent_line_at_x0_range_of_t_with_three_extreme_points_max_value_of_m_l312_312052

noncomputable def f (x t : ℝ) : ℝ := (x^3 - 6*x^2 + 3*x + t) * Real.exp x

-- Problem 1
theorem tangent_line_at_x0 (t : ℝ) (hx : t = 1) : 
  (let f := fun x => (x^3 - 6*x^2 + 3*x + 1) * Real.exp x in
   let f' := fun x => (x^3 - 3*x^2 - 9*x + 4) * Real.exp x in
   f 0 = 1 ∧ f' 0 = 4) → 
  (∀ x, f x = f 0 + f' 0 * x := by sorry) :=
sorry

-- Problem 2
theorem range_of_t_with_three_extreme_points 
  (h : ∀ t, 
      (∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
      (let g := (fun x => x^3 - 3*x^2 - 9*x + t + 3) in 
       g x = 0 ∧ g y = 0 ∧ g z = 0))) : 
  -8 < t ∧ t < 24 := 
sorry

-- Problem 3
theorem max_value_of_m (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) : 
  (∀ x ∈ Set.Icc 1 5, f x t ≤ x) ∧ 
  (∀ x > 5, ∃ x ∈ Set.Icc 1 x, f x t > x) :=
sorry

end tangent_line_at_x0_range_of_t_with_three_extreme_points_max_value_of_m_l312_312052


namespace foldable_polygons_for_cube_l312_312209

open function

-- Define the initial polygon shape conditions
def base_polygon : Set (Finset (ℕ × ℕ)) :=
  {s | ∃ x y : ℕ, s = {(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)} ∧ 
  (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 1)}

-- Define the 12 possible positions for the additional square
def possible_positions : List (ℕ × ℕ) := 
  [(0,1), (1,0), (1,2), (2,1), (0,0), (0,2), (2,0), (2,2), 
  (3,1), (1,3), (3,3), (0,3)]

-- The main theorem stating the foldable polygons to form a cube
theorem foldable_polygons_for_cube : 
  (pos : possible_positions) →
     (∃ p ∈ base_polygon, ∃ q ∈ possible_positions,
       q ∈ p ∧ p ∈ cube_configurations) →
            ∀ pos ∈ base_polygon: Set (Finset (ℕ × ℕ)), 
                pos ∈ base_polygon ⟶ pos ⟶ base_polygon :=
by
  sorry

end foldable_polygons_for_cube_l312_312209


namespace count_divisors_36_l312_312481

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312481


namespace num_divisors_of_36_l312_312520

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312520


namespace second_largest_n_divides_170_factorial_l312_312540

-- Define the main problem
theorem second_largest_n_divides_170_factorial :
  let n := (170 / 5) + (34 / 5) + (6 / 5) - 1 in 
  n = 40 := 
by
  -- computation will verify this calculation
  have h : (170 / 5) + (34 / 5) + (6 / 5) = 41 :=
    by norm_num [int.div],
  have n_def : n = 41 - 1 :=
    by rw [show n = (170 / 5) + (34 / 5) + (6 / 5) - 1, from rfl, h],
  exact (n_def.trans (sub_self 1)).symm,
  sorry -- Proof omitted

end second_largest_n_divides_170_factorial_l312_312540


namespace find_q_l312_312848

def P (q x : ℝ) : ℝ := x^4 + 2 * q * x^3 - 3 * x^2 + 2 * q * x + 1

theorem find_q (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ P q x1 = 0 ∧ P q x2 = 0) → q < 1 / 4 :=
by
  sorry

end find_q_l312_312848


namespace monotonic_intervals_extreme_points_interval_exists_x0_monotonic_increasing_l312_312916

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  (Real.exp x) / (x^3) - (3 * k) / x - k * Real.log x

def f' (x : ℝ) (k : ℝ) : ℝ :=
  ((Real.exp x) * (x - 3)) / (x^4) + k * (3 / (x^2) - 1 / x)

theorem monotonic_intervals (k : ℝ) (x : ℝ) (hk : k ≤ 0) :
  (x > 3 → f' x k > 0) ∧ (0 < x ∧ x < 3 → f' x k < 0) :=
sorry

theorem extreme_points_interval (k : ℝ) :
  (∃ (x : ℝ), (1 < x ∧ x < 3) ∧ f' x k = 0) → (Real.exp 2 / 4 < k ∧ k < Real.exp 3 / 9) :=
sorry

theorem exists_x0_monotonic_increasing (k : ℝ) :
  ∃ x0 > 0, ∀ (x > x0), f' x k > 0 :=
sorry

end monotonic_intervals_extreme_points_interval_exists_x0_monotonic_increasing_l312_312916


namespace triangle_is_isosceles_l312_312122

noncomputable def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem triangle_is_isosceles (A B C a b c : ℝ) (h : a * Real.cos B = b * Real.cos A) :
  is_isosceles_triangle A B C a b c :=
begin
  -- Proof is omitted here
  sorry
end

end triangle_is_isosceles_l312_312122


namespace number_of_divisors_of_36_l312_312506

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312506


namespace correct_propositions_l312_312603

-- Define the lines and planes
variables {m n : Line} {α β : Plane}

-- Define the propositions
def Prop1 : Prop := m ⊥ α ∧ n ∥ α → m ⊥ n
def Prop2 : Prop := m ⊥ n ∧ n ∥ α → m ⊥ α
def Prop3 : Prop := m ⊥ α ∧ α ∥ β → m ⊥ β
def Prop4 : Prop := m ⊥ α ∧ m ⊥ β → α ∥ β

-- Define the main theorem to prove the correct propositions
theorem correct_propositions :
  Prop1 ∧ Prop3 ∧ Prop4 :=
by
  ⟨sorry, sorry, sorry⟩

end correct_propositions_l312_312603


namespace range_of_a_l312_312421

theorem range_of_a (f : ℝ → ℝ) (h_odd : ∀ x ∈ Ioo (-1 : ℝ) 1, f (-x) = -f x)
  (h_decreasing : ∀ x y ∈ Ico 0 1, x < y → f y ≤ f x) (a : ℝ) 
  (h_condition : f (1 - a) + f (1 - a^2) < 0) : a ∈ Ioo 0 1 := 
sorry

end range_of_a_l312_312421


namespace natural_number_solution_l312_312007

theorem natural_number_solution (a : ℕ) (h : (∃ n : ℕ, n = (a + 1 + nat.sqrt (a^5 + 2 * a^2 + 1)) / (a^2 + 1))) : a = 1 :=
sorry

end natural_number_solution_l312_312007


namespace count_five_digit_integers_l312_312469

-- Define the total number of different 5-digit positive integers
theorem count_five_digit_integers : 
  let first_digit_choices := 9 in 
  let remaining_digit_choices := 10 in 
  (first_digit_choices * remaining_digit_choices ^ 4) = 90000 :=
by sorry

end count_five_digit_integers_l312_312469


namespace tangent_at_point_is_correct_h_geq_neg2_for_x_geq_1_l312_312440

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x ^ 2 - Real.exp 1 * x - 2

noncomputable def h (x : ℝ) : ℝ := Real.exp x - 2 * x - Real.exp 1

theorem tangent_at_point_is_correct :
  ∃ (m b : ℝ), (λ x y, y = m * x + b) = λ x y, y + 3 = -2 * (x - 1) :=
sorry

theorem h_geq_neg2_for_x_geq_1 :
  ∀ (x : ℝ), (x ≥ 1) → (h x ≥ -2) :=
sorry

end tangent_at_point_is_correct_h_geq_neg2_for_x_geq_1_l312_312440


namespace problem_condition_l312_312928

variable (a : ℝ)
def p : Prop := ∀ n : ℕ+ , (-1 : ℤ) ^ n * (2 * a + 1) < 2 + (-1 : ℤ) ^ (n + 1) / (n : ℚ)
def q : Prop := ∀ x : ℝ, abs (x - 5/2) < a → abs (x^2 - 5) < 4

theorem problem_condition (h₁ : ¬ (p a ∧ q a)) (h₂ : p a ∨ q a) : 
  a ∈ Icc (-3/2) 0 ∪ Icc (1/4) 1/2 := sorry

end problem_condition_l312_312928


namespace find_number_l312_312838

theorem find_number (x : ℝ) : 
  (3 * x / 5 - 220) * 4 + 40 = 360 → x = 500 :=
by
  intro h
  sorry

end find_number_l312_312838


namespace value_of_x_l312_312077

theorem value_of_x (b x : ℝ) (h₀ : 1 < b) (h₁ : 0 < x) (h₂ : (2 * x) ^ (Real.logb b 2) - (3 * x) ^ (Real.logb b 3) = 0) : x = 1 / 6 :=
by {
  sorry
}

end value_of_x_l312_312077


namespace actual_average_speed_l312_312765

variable {t : ℝ} (h₁ : t > 0) -- ensure that time is positive
variable {v : ℝ} 

theorem actual_average_speed (h₂ : v > 0)
  (h3 : v * t = (v + 12) * (3 / 4 * t)) : v = 36 :=
by
  sorry

end actual_average_speed_l312_312765


namespace min_expression_value_l312_312610

theorem min_expression_value (n : ℕ) (a b : ℝ) (h1 : a + b = 2) (h2 : a > 0) (h3 : b > 0) : 
  \(\frac{1}{1 + a^n} + \frac{1}{1 + b^n} \ge 1\) :=
  sorry

end min_expression_value_l312_312610


namespace Smithtown_left_handed_women_percentage_l312_312092

theorem Smithtown_left_handed_women_percentage :
  ∃ (x y : ℕ), 
    (3 * x + x = 4 * x) ∧
    (3 * y + 2 * y = 5 * y) ∧
    (4 * x = 5 * y) ∧
    (x = y) → 
    let total_population := 4 * x
    let left_handed_women := x
    left_handed_women / total_population = 0.25 :=
sorry

end Smithtown_left_handed_women_percentage_l312_312092


namespace trapezoid_base_ratio_l312_312666

variable {A B C D K M P Q L N : Type}
variable [LinearOrderedField A]
variable [LinearOrderedField B]
variable [LinearOrderedField C]
variable [LinearOrderedField D]

def is_trapezoid (A B C D : Type) (a b : ℝ) : Prop :=
  is_parallel (AD BC) ∧ AD.length = a ∧ BC.length = b ∧ a > b

def area (A B C D : Type) (a b : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

def quadrilateral_area (K L M N : Type) (a b h : ℝ) : ℝ :=
  (1 / 4) * (a - b) * h

theorem trapezoid_base_ratio (A B C D K M P Q L N : Type) (a b : ℝ) (h : ℝ) :
  is_trapezoid A B C D a b →
  quadrilateral_area K L M N a b h = (1 / 4) * area A B C D a b ↔ a / b = 3 :=
by 
  sorry

end trapezoid_base_ratio_l312_312666


namespace range_of_m_l312_312054

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 + 4 * Real.log x

theorem range_of_m :
  ∀ x₀ (h₀ : 1 ≤ x₀ ∧ x₀ ≤ 3),
    let slope_f := (fun x => x + 4 / x) x₀ in
    ∀ m, (x₀ + 4 / x₀) = m → (m ∈ Set.Icc 4 5) :=
  begin
    intros x₀ h₀ slope_f m h,
    have h1 : slope_f = x₀ + 4 / x₀ := rfl,
    rw h1 at h,
    have h2 : m = x₀ + 4 / x₀ := h.symm,
    have h3 : 1 ≤ x₀ ∧ x₀ ≤ 3 := h₀,
    cases h3 with h4 h5,
    split,
    { linarith [(4 / x₀) + x₀],
      use 4 / x₀ + x₀, 
      split, linarith, linarith, sorry }, 
    { linarith [(4 / x₀) + x₀, sorry] },
  end

end range_of_m_l312_312054


namespace sin_a_sin_b_eq_sin_c_find_tan_B_l312_312547

variables {A B C a b c : ℝ}
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (cos_a_pos : (cos A) / a + (cos B) / b = (sin C) / c)
variables (cosine_rule_cond : b^2 + c^2 - a^2 = (6/5) * b * c)

-- Part I: Prove that sin A * sin B = sin C
theorem sin_a_sin_b_eq_sin_c 
  (h_cos_pos : cos_a_pos) : sin A * sin B = sin C :=
by
  sorry

-- Part II: Find tan B given the cosine rule condition
theorem find_tan_B 
  (h_cos_rule : cosine_rule_cond) : tan B = 4 :=
by
  sorry

end sin_a_sin_b_eq_sin_c_find_tan_B_l312_312547


namespace ceil_neg_3_7_l312_312363

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l312_312363


namespace two_digit_prime_count_l312_312946

theorem two_digit_prime_count : 
  let digits := {2, 3, 7, 8, 9}
  in (∃ primes : Finset ℕ, (primes.card = 6) ∧ (∀ p ∈ primes, p ∈ digits.product {d | d ∈ digits ∧ d ≠ d}.toFinset) ∧ (∀ p ∈ primes, Prime p)) := sorry

end two_digit_prime_count_l312_312946


namespace volume_of_prism_l312_312232

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) :
    a * b * c = 426 :=
sorry

end volume_of_prism_l312_312232


namespace find_f_neg_two_l312_312039

def is_even_function (f : ℝ → ℝ) (h : ℝ → ℝ) := ∀ x, h (-x) = h x

theorem find_f_neg_two (f : ℝ → ℝ) (h : ℝ → ℝ) (hx : ∀ x, h x = f (2*x) + x)
  (h_even : is_even_function f h) 
  (h_f_two : f 2 = 1) : 
  f (-2) = 3 :=
  by
    sorry

end find_f_neg_two_l312_312039


namespace proof_frost_and_decorate_in_10_minutes_l312_312329

def frost_and_decorate_in_10_minutes : Prop :=
  let Cagney_rate := 25 -- seconds per cupcake for Cagney
  let Lacey_rate := 35 -- seconds per cupcake for Lacey
  let break_time := 10 -- seconds break after every 6 cupcakes
  let decor_time := 5 -- seconds per cupcake for decorating
  let total_time := 600 -- total time in seconds (10 minutes)
  let combined_rate := Cagney_rate * Lacey_rate / (Cagney_rate + Lacey_rate) -- harmonic mean of their rates
  let cupcakes_per_cycle := 6 -- cupcakes before break
  let cycle_time := cupcakes_per_cycle * combined_rate + break_time -- time for one cycle including break
  let full_cycles := (total_time / cycle_time).to_nat -- number of full cycles they can complete
  let total_cupcakes := full_cycles * cupcakes_per_cycle -- total cupcakes frosted and decorated
  let remaining_time := (total_time - full_cycles * cycle_time).to_nat -- remaining time after full cycles
  let additional_cupcakes := (remaining_time / (combined_rate + decor_time)).to_nat -- additional cupcakes in the remaining time
  total_cupcakes + additional_cupcakes = 28

theorem proof_frost_and_decorate_in_10_minutes :
  frost_and_decorate_in_10_minutes :=
begin
  sorry
end

end proof_frost_and_decorate_in_10_minutes_l312_312329


namespace count_divisors_36_l312_312483

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312483


namespace oprah_winfrey_band_weights_l312_312563

theorem oprah_winfrey_band_weights :
  let weight_trombone := 10
  let weight_tuba := 20
  let weight_drum := 15
  let num_trumpets := 6
  let num_clarinets := 9
  let num_trombones := 8
  let num_tubas := 3
  let num_drummers := 2
  let total_weight := 245

  15 * x = total_weight - (num_trombones * weight_trombone + num_tubas * weight_tuba + num_drummers * weight_drum) 
  → x = 5 := by
  sorry

end oprah_winfrey_band_weights_l312_312563


namespace problem_solution_l312_312879

def heartsuit (x : ℝ) : ℝ :=
  (x + x^3) / 2

theorem problem_solution : heartsuit 1 + heartsuit (-1) + heartsuit 2 = 5 := by
  sorry

end problem_solution_l312_312879


namespace time_to_finish_typing_l312_312578

-- Definitions
def words_per_minute : ℕ := 38
def total_words : ℕ := 4560

-- Theorem to prove
theorem time_to_finish_typing : (total_words / words_per_minute) / 60 = 2 := by
  sorry

end time_to_finish_typing_l312_312578


namespace a_5_is_1_over_10_l312_312451

-- Define the sequence given the general term
def a (n : ℕ) [hn : Fact (0 < n)] := n / (n^2 + 25)

-- Define the property that the fifth term equals 1/10
theorem a_5_is_1_over_10 : a 5 = 1 / 10 :=
by
  sorry -- proof is omitted, as requested

end a_5_is_1_over_10_l312_312451


namespace intersection_A_B_l312_312898

-- Definitions of the sets A and B
def set_A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 7 }

-- Theorem statement to prove the intersection
theorem intersection_A_B : set_A ∩ set_B = { x | 3 ≤ x ∧ x < 7 } := by
  sorry

end intersection_A_B_l312_312898


namespace beans_fraction_remaining_l312_312697

/-- The weight of a glass jar is 25% of the weight of the jar filled with coffee beans.
After some of the beans have been removed, the weight of the jar and the remaining beans is 60% of the original total weight.
Prove that the fraction part of the beans that remain in the jar is 7/15. -/
theorem beans_fraction_remaining (J B B_remaining : ℝ) 
  (h1 : J = 0.25 * (J + B)) 
  (h2 : J + B_remaining = 0.60 * (J + B)) : 
  B_remaining / B = 7 / 15 := 
by 
  sorry ⟩

end beans_fraction_remaining_l312_312697


namespace log3_fraction_sq_l312_312172
-- Import the necessary library.

-- Define the main proof statement.
theorem log3_fraction_sq (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hlog : log 3 x = log y 81) (hsum : x + y = 36) :
  (log 3 (x / y)) ^ 2 = 1 :=
sorry

end log3_fraction_sq_l312_312172


namespace rightmost_three_digits_7_pow_1993_l312_312710

theorem rightmost_three_digits_7_pow_1993 :
  ∃ n : ℕ, 7^1993 ≡ n [MOD 1000] ∧ n = 343 := by
  have h0 : 7^0 ≡ 1 [MOD 1000] := by norm_num
  have h1 : 7^1 ≡ 7 [MOD 1000] := by norm_num
  have h2 : 7^2 ≡ 49 [MOD 1000] := by norm_num
  have h3 : 7^3 ≡ 343 [MOD 1000] := by norm_num
  have h4 : 7^4 ≡ 401 [MOD 1000] := by norm_num
  have h5 : 7^5 ≡ 807 [MOD 1000] := by norm_num
  have h6 : 7^6 ≡ 649 [MOD 1000] := by norm_num
  have h7 : 7^7 ≡ 543 [MOD 1000] := by norm_num
  have h8 : 7^8 ≡ 801 [MOD 1000] := by norm_num
  have h9 : 7^9 ≡ 607 [MOD 1000] := by norm_num
  have h10 : 7^10 ≡ 249 [MOD 1000] := by norm_num
  sorry

end rightmost_three_digits_7_pow_1993_l312_312710


namespace num_divisors_of_36_l312_312523

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312523


namespace work_completion_days_l312_312736

theorem work_completion_days
  (E_q : ℝ) -- Efficiency of q
  (E_p : ℝ) -- Efficiency of p
  (E_r : ℝ) -- Efficiency of r
  (W : ℝ)  -- Total work
  (H1 : E_p = 1.5 * E_q) -- Condition 1
  (H2 : W = E_p * 25) -- Condition 2
  (H3 : E_r = 0.8 * E_q) -- Condition 3
  : (W / (E_p + E_q + E_r)) = 11.36 := -- Prove the days_needed is 11.36
by
  sorry

end work_completion_days_l312_312736


namespace spatial_vector_theorem_l312_312570

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C D : V)
variables {λ μ x y z : ℝ}

-- Conditions
def point_O_outside_line_AB : Prop := 
  ¬(∃ t : ℝ, O = A + t • (B - A))

def point_C_on_line_AB : Prop :=
  ∃ t : ℝ, C = A + t • (B - A)

def OC_as_linear_combination_of_OA_and_OB (λ μ : ℝ) : Prop :=
  ∃ λ μ : ℝ, C - O = λ • (A - O) + μ • (B - O) ∧ λ + μ = 1

def not_collinear (A B C D : V) : Prop :=
  ¬Collinear ℝ ![A, B, C] ∧ ¬Collinear ℝ ![A, B, D] ∧ ¬Collinear ℝ ![A, C, D] ∧ ¬Collinear ℝ ![B, C, D]

def points_in_same_plane (A B C D : V) : Prop :=
  ∃ α β γ : ℝ, D = α • A + β • B + γ • C ∧ ¬(α = 0 ∧ β = 0 ∧ γ = 0)

-- Main statement to prove
theorem spatial_vector_theorem 
  (h1 : point_O_outside_line_AB O A B)
  (h2 : point_C_on_line_AB A B C)
  (h3 : OC_as_linear_combination_of_OA_and_OB O A B C λ μ)
  (h4 : points_in_same_plane A B C D)
  (h5 : not_collinear A B C D)
  (H : ∃ x y z : ℝ, D - O = x • (A - O) + y • (B - O) + z • (C - O)) :
  x + y + z = 1 :=
sorry

end spatial_vector_theorem_l312_312570


namespace trig_identity_l312_312885

def cos_sq_eq_sin (α : ℝ) : Prop :=
  cos α ^ 2 = sin α

theorem trig_identity (α : ℝ) (h : cos_sq_eq_sin α) : 
  (1 / sin α + cos α ^ 4) = 2 :=
by
  sorry

end trig_identity_l312_312885


namespace problem_1_problem_2_problem_3_l312_312855

noncomputable def tangent_line_eq_1 (x y : ℝ) : Prop :=
  y = -real.sqrt 3 * x + 4

noncomputable def tangent_line_eq_2_pos (x y : ℝ) : Prop :=
  y = (2 * real.sqrt 5 / 5) * x - (6 * real.sqrt 5 / 5)

noncomputable def tangent_line_eq_2_neg (x y : ℝ) : Prop :=
  y = -(2 * real.sqrt 5 / 5) * x + (6 * real.sqrt 5 / 5)

noncomputable def tangent_line_eq_3_pos (x y : ℝ) : Prop :=
  x + y = 2 * real.sqrt 2

noncomputable def tangent_line_eq_3_neg (x y : ℝ) : Prop :=
  x + y = -2 * real.sqrt 2

theorem problem_1 (P : ℝ × ℝ) (hP : P = (real.sqrt 3, 1)) : 
  tangent_line_eq_1 P.1 P.2 := sorry

theorem problem_2 (Q : ℝ × ℝ) (hQ : Q = (3, 0)) : 
  tangent_line_eq_2_pos Q.1 Q.2 ∨ tangent_line_eq_2_neg Q.1 Q.2 := sorry

theorem problem_3 (x y : ℝ) (h : x + y = 0) : 
  tangent_line_eq_3_pos x y ∨ tangent_line_eq_3_neg x y := sorry

end problem_1_problem_2_problem_3_l312_312855


namespace problem1_problem2_l312_312546

/-
Problem 1: Prove magnitude of angle B in triangle
Given:
  ∀ a b c A B C : ℝ,
  (0 < A ∧ A < Real.pi) ∧
  (a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c) ∧
  (Real.sin A * Real.sin B = Real.cos (C / 2)^2)
  → B = Real.pi / 6
-/
theorem problem1
  (a b c A B C : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c)
  (h3 : Real.sin A * Real.sin B = Real.cos (C / 2)^2)
  : B = Real.pi / 6 :=
  sorry

/-
Problem 2: Find the sum of the sequence S_n
Given:
  ∀ d a_n S_n B : ℝ,
  (a_n = λ n, 2 * n) ∧
  (S_n = ∑ i in finset.range n, 4 / (a_n i * a_n (i+1)))
  → S_n = n / (n + 1)
-/
theorem problem2
  (a_n S_n : ℕ → ℝ)
  (n : ℕ)
  (h1 : ∀ n, a_n n = 2 * n)
  (h2 : S_n = ∑ i in finset.range n, 4 / (a_n i * a_n (i + 1)))
  : S_n = n / (n + 1) :=
  sorry

end problem1_problem2_l312_312546


namespace egyptian_pyramid_angle_and_seked_l312_312199

noncomputable def inclination_angle_of_side_edges (face_angle: ℝ) : ℝ :=
  Real.arctan (Real.tan face_angle / Real.sqrt 2)

noncomputable def seked (edge_angle: ℝ) : ℝ :=
  Real.cos edge_angle

theorem egyptian_pyramid_angle_and_seked (h: inclination_angle_of_side_edges (52 * Real.pi / 180) ≈ 42 + 8 / 60 + 49 / 3600) :
  seked (inclination_angle_of_side_edges (52 * Real.pi / 180)) ≈ 0.74148 :=
by sorry

end egyptian_pyramid_angle_and_seked_l312_312199


namespace num_lines_passing_through_A_and_C_l312_312413

noncomputable def num_lines_through_point_and_tangent_to_parabola (A : ℝ × ℝ) (C : ℝ → ℝ → Prop) : ℕ :=
  if A = (0, 2) ∧ (∀ x y, C x y ↔ y^2 = 3 * x) then 3 else 0  

theorem num_lines_passing_through_A_and_C : 
  num_lines_through_point_and_tangent_to_parabola (0, 2) (λ x y, y^2 = 3 * x) = 3 := 
sorry

end num_lines_passing_through_A_and_C_l312_312413


namespace ceiling_of_neg_3_7_l312_312372

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l312_312372


namespace existence_of_f_l312_312349

noncomputable def f (t : ℝ) : ℝ :=
if |t| ≥ 2 then t^3 - 3 * t else 2 * (t - 1)^2

theorem existence_of_f :
  (∀ x : ℝ, f(2 * Real.cos x ^ 2) = 1 + Real.cos (4 * x)) ∧
  (∀ x : ℝ, x ≠ 0 → f(x + 1/x) = x^3 + 1/x^3) :=
by
  sorry

end existence_of_f_l312_312349


namespace find_angle_C_find_a_plus_b_l312_312964

-- Define the entity of the sides of the triangle and the area
variables (a b c : ℝ) 
variable (S : ℝ)

-- Define the required conditions for both parts of the problem
variable (h1 : a^2 + b^2 - c^2 = (4 * Real.sqrt 3 / 3) * S)
variable (h2 : c = Real.sqrt 3)
variable (h3 : S = Real.sqrt (3) / 2)

-- Define the triangle properties
def triangle_properties : Prop :=
  ∀ (A B C : ℝ), (a = S * 2 / (Real.sqrt 3)) → 
  (b = (S * 2 / (Real.sqrt 3)) * (Real.sin C / Real.cos C)) →

-- Find the measure of angle C
theorem find_angle_C : ∃ C : ℝ, h1 → C = Real.pi / 3 := 
sorry

-- Find the value of a + b given additional conditions
theorem find_a_plus_b : ∃ a b : ℝ, c = Real.sqrt 3 → S = (Real.sqrt 3) / 2 → a + b = 3 := 
sorry

end find_angle_C_find_a_plus_b_l312_312964


namespace quadratic_equation_in_y_roots_l312_312452

theorem quadratic_equation_in_y_roots {(p q : ℝ)} 
  (h : 1 + p + q ≠ 0)
  (x1 x2 : ℝ) 
  (hx : x1 * x2 = q)
  (hxsum : x1 + x2 = -p) : 
  ∃ y1 y2 : ℝ, 
  y1 = (x1 + x1^2) / (1 - x2) ∧ 
  y2 = (x2 + x2^2) / (1 - x1) ∧ 
  ((y^2 + (p*(1 + 3*q - p^2))/(1 + p + q) * y + (q * (1 - p + q))/(1 + p + q) = 0) :=
begin
  sorry
end

end quadratic_equation_in_y_roots_l312_312452


namespace apples_in_box_l312_312227

theorem apples_in_box (total_fruit : ℕ) (one_fourth_oranges : ℕ) (half_peaches_oranges : ℕ) (apples_five_peaches : ℕ) :
  total_fruit = 56 →
  one_fourth_oranges = total_fruit / 4 →
  half_peaches_oranges = one_fourth_oranges / 2 →
  apples_five_peaches = 5 * half_peaches_oranges →
  apples_five_peaches = 35 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end apples_in_box_l312_312227


namespace cube_root_sum_eq_one_l312_312694

theorem cube_root_sum_eq_one :
  (∛(5 + 2 * Real.sqrt 13) + ∛(5 - 2 * Real.sqrt 13)) = 1 := 
by
  sorry

end cube_root_sum_eq_one_l312_312694


namespace sector_area_correct_l312_312909

noncomputable def sector_area (θ r : ℝ) : ℝ :=
  (θ / (2 * Real.pi)) * (Real.pi * r^2)

theorem sector_area_correct : 
  sector_area (Real.pi / 3) 3 = (3 / 2) * Real.pi :=
by
  sorry

end sector_area_correct_l312_312909


namespace range_of_a_l312_312446

theorem range_of_a (a : ℝ) (e : ℝ) (x : ℝ) (ln : ℝ → ℝ) :
  (∀ x, (1 / e) ≤ x ∧ x ≤ e → (a - x^2 = -2 * ln x)) →
  (1 ≤ a ∧ a ≤ (e^2 - 2)) :=
by
  sorry

end range_of_a_l312_312446


namespace trapezoid_base_ratio_l312_312644

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l312_312644


namespace coffee_cost_l312_312584

theorem coffee_cost (days_in_april : ℕ) (coffees_per_day : ℕ) (total_spent : ℕ) (h1 : days_in_april = 30) (h2 : coffees_per_day = 2) (h3 : total_spent = 120) :
  total_spent / (days_in_april * coffees_per_day) = 2 :=
by
  rw [h1, h2, h3]
  norm_num

end coffee_cost_l312_312584


namespace angles_on_line_y_eq_sqrt3_x_double_angle_quadrants_l312_312293

theorem angles_on_line_y_eq_sqrt3_x (α : ℝ) (k : ℤ) : 
  (cos α = 1/2 ∧ sin α = sqrt 3 / 2) ↔ α = π / 3 + k * π :=
sorry

theorem double_angle_quadrants (α : ℝ) (k : ℤ) (h : π + 2*k*π < α ∧ α < 3*π/2 + 2*k*π) :
  (0 < 2*α ∧ 2*α < π) ∨ (π < 2*α ∧ 2*α < 2*π) :=
sorry

end angles_on_line_y_eq_sqrt3_x_double_angle_quadrants_l312_312293


namespace correct_fill_in_the_blank_l312_312189

def options : Set String := {"other", "others", "one", "ones"}
def sentence : String := "The CDs are on sale! Buy one and you get ______ completely free."

theorem correct_fill_in_the_blank :
  "one" ∈ options ∧ 
  (∀ opt ∈ options, 
    (opt = "other" → False)
    ∧ (opt = "others" → False)
    ∧ (opt = "ones" → False)
    ∧ (opt = "one" → True)
    → opt = "one") :=
by
  intros opt hopt,
  cases hopt,
  sorry

end correct_fill_in_the_blank_l312_312189


namespace maximal_n_for_quadratic_factorization_l312_312857

theorem maximal_n_for_quadratic_factorization :
  ∃ n, n = 325 ∧ (∃ A B : ℤ, A * B = 108 ∧ n = 3 * B + A) :=
by
  use 325
  use 1, 108
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num
  sorry

end maximal_n_for_quadratic_factorization_l312_312857


namespace triangulated_polygon_coloring_l312_312793

theorem triangulated_polygon_coloring 
  (V : Type) [Fintype V] [DecidableEq V]
  (E : Finset (Finset V)) 
  (triangulated : ∀ {T : Finset V}, T ∈ E → T.card = 3)
  (properly_colored : ∀ {u v w : V} {T : Finset V}, T = {u, v, w} → T ∈ E → 
                      (u % 2 ≠ v % 2 ∧ v % 2 ≠ w % 2 ∧ w % 2 ≠ u % 2)) 
  : ∃ (f : V → Fin 3), ∀ {u v : V}, u ∈ V → v ∈ V → u ≠ v → (∀ {T : Finset V}, T = {u, v} → T ∈ E → f u ≠ f v) :=
sorry

end triangulated_polygon_coloring_l312_312793


namespace picnic_weather_condition_l312_312454

variables (P Q : Prop)

theorem picnic_weather_condition (h : ¬P → ¬Q) : Q → P := 
by sorry

end picnic_weather_condition_l312_312454


namespace correct_exponent_division_l312_312725

theorem correct_exponent_division (a : ℝ) : (a^7) / (a^3) = a^4 :=
by calc
  (a^7) / (a^3) = a^(7 - 3) : by rw [div_eq_mul_inv, ← pow_sub]
              ... = a^4     : by norm_num

end correct_exponent_division_l312_312725


namespace expression_value_l312_312684

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 :=
by
  sorry

end expression_value_l312_312684


namespace smallest_m_for_integral_solutions_l312_312249

theorem smallest_m_for_integral_solutions :
  ∃ m : ℕ, m > 0 ∧ (∃ p q : ℤ, 10 * p * q = 660 ∧ p + q = m/10) ∧ m = 170 :=
by
  sorry

end smallest_m_for_integral_solutions_l312_312249


namespace sum_of_digits_l312_312220

theorem sum_of_digits :
  ∃ (a b : ℕ), (4 * 100 + a * 10 + 5) + 457 = (9 * 100 + b * 10 + 2) ∧
                (((9 + 2) - b) % 11 = 0) ∧
                (a + b = 4) :=
sorry

end sum_of_digits_l312_312220


namespace smallest_omega_l312_312926

noncomputable def f (omega x : ℝ) : ℝ :=
  let a1 := (√3 : ℝ)
  let a2 := Real.sin (omega * x)
  let a3 := (1 : ℝ)
  let a4 := Real.cos (omega * x)
  a1 * a4 - a2 * a3

theorem smallest_omega : ∃ ω > 0, (∀ x : ℝ, f ω (x + 2 * Real.pi / 3) = f ω (-x)) ∧ (ω = 5 / 4) :=
sorry

end smallest_omega_l312_312926


namespace interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l312_312345

section
open Real

noncomputable def interval1 (x : ℝ) : Real := log (1 - x ^ 2)
noncomputable def interval2 (x : ℝ) : Real := x * (1 + 2 * sqrt x)
noncomputable def interval3 (x : ℝ) : Real := log (abs x)

-- Function 1: p = ln(1 - x^2)
theorem interval1_increase_decrease :
  (∀ x : ℝ, -1 < x ∧ x < 0 → deriv interval1 x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv interval1 x < 0) := by
  sorry

-- Function 2: z = x(1 + 2√x)
theorem interval2_increasing :
  ∀ x : ℝ, x ≥ 0 → deriv interval2 x > 0 := by
  sorry

-- Function 3: y = ln|x|
theorem interval3_increase_decrease :
  (∀ x : ℝ, x < 0 → deriv interval3 x < 0) ∧
  (∀ x : ℝ, x > 0 → deriv interval3 x > 0) := by
  sorry

end

end interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l312_312345


namespace G_51_equals_36_and_1_3_l312_312954

noncomputable def G : ℕ → ℚ
| 1 := 3
| (n + 1) := (3 * G n + 2) / 3

theorem G_51_equals_36_and_1_3 :
  G 51 = 36 + 1/3 :=
by
  sorry

end G_51_equals_36_and_1_3_l312_312954


namespace largest_n_for_factored_polynomial_l312_312865

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l312_312865


namespace recurrence_relation_l312_312889

open Nat

def F : ℕ → ℕ
| 0     := 1
| (n+1) := ∑ k in range (n+1), F k * F (n-k)

theorem recurrence_relation (n : ℕ) (h : 0 < n) : 
  F n = ∑ k in range n, F k * F (n-1-k) := by sorry

end recurrence_relation_l312_312889


namespace average_of_first_two_and_last_three_numbers_l312_312669

theorem average_of_first_two_and_last_three_numbers 
    (numbers : list ℝ)
    (h1 : numbers.length = 7)
    (h2 : numbers.sum / 7 = 63)
    (a3 a4 : ℝ)
    (h3 : a3 = numbers.nth_le 2 h1)
    (h4 : a4 = numbers.nth_le 3 h1)
    (h5 : (a3 + a4) / 2 = 60) 
    : (numbers.take 2 ++ numbers.drop 4).sum / 5 = 64.2 :=
begin
  sorry
end

end average_of_first_two_and_last_three_numbers_l312_312669


namespace no_prime_divisible_by_56_l312_312527

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end no_prime_divisible_by_56_l312_312527


namespace artifacts_per_wing_l312_312784

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end artifacts_per_wing_l312_312784


namespace rounding_strategy_correct_l312_312810

-- Definitions of rounding functions
def round_down (n : ℕ) : ℕ := n - 1  -- Assuming n is a large integer, for simplicity
def round_up (n : ℕ) : ℕ := n + 1

-- Definitions for conditions
def cond1 (p q r : ℕ) : ℕ := round_down p / round_down q + round_down r
def cond2 (p q r : ℕ) : ℕ := round_up p / round_down q + round_down r
def cond3 (p q r : ℕ) : ℕ := round_down p / round_up q + round_down r
def cond4 (p q r : ℕ) : ℕ := round_down p / round_down q + round_up r
def cond5 (p q r : ℕ) : ℕ := round_up p / round_up q + round_down r

-- Theorem stating the correct condition
theorem rounding_strategy_correct (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) :
  cond3 p q r < p / q + r :=
sorry

end rounding_strategy_correct_l312_312810


namespace students_paid_half_l312_312549

theorem students_paid_half (F H : ℕ) 
  (h1 : F + H = 25)
  (h2 : 50 * F + 25 * H = 1150) : 
  H = 4 := by
  sorry

end students_paid_half_l312_312549


namespace only_quadratic_equation_in_one_variable_is_B_l312_312724

-- Definitions of each of the given equations
def option_A (a b c : ℝ) : Prop := a ≠ 0 ∧ a * x^2 + b * x + c = 0
def option_B : Prop := x^2 = 1
def option_C : Prop := x^2 + 2 * x = 2 / x
def option_D (y : ℝ) : Prop := x^2 + y^2 = 0

-- The proof problem to verify which equation is a quadratic equation in one variable
theorem only_quadratic_equation_in_one_variable_is_B (x : ℝ) :
  ¬(∃ a b c: ℝ, option_A a b c) ∧ option_B ∧ ¬option_C ∧ ∀ y : ℝ, ¬option_D y := 
by
  sorry

end only_quadratic_equation_in_one_variable_is_B_l312_312724


namespace max_n_for_factorable_quadratic_l312_312862

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l312_312862


namespace ratio_of_coeffs_l312_312031

theorem ratio_of_coeffs
  (a b c d e : ℝ) 
  (h_poly : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) : 
  d / e = 25 / 12 :=
by
  sorry

end ratio_of_coeffs_l312_312031


namespace num_divisors_of_36_l312_312492

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312492


namespace max_area_triangle_MNP_l312_312826

open Real

-- Define the basic setup for the triangle and points
variable (A B C : Point)
variable (E F P M : Point)
variable (ABC_right : right_angle C A B)
variable (hypotenuse_AB : dist A B = 1)
variable (angle_bisector : bisector (angle A C B) intersects_med AF BE at P and M)
variable (intersection_P : AF ∩ BE = {P})

-- Define the problem by stating the maximum value of the area
theorem max_area_triangle_MNP :
  exists (A B C E F P M : Point), right_angle C A B ∧ dist A B = 1 ∧ bisector (angle A C B) intersects_med AF BE at P and M ∧ AF ∩ BE = {P} ∧ (area_of_triangle M N P) ≤ (1 / 150) := sorry

end max_area_triangle_MNP_l312_312826


namespace polynomial_sum_of_squares_is_23456_l312_312535

theorem polynomial_sum_of_squares_is_23456 (p q r s t u : ℤ) :
  (∀ x, 1728 * x ^ 3 + 64 = (p * x ^ 2 + q * x + r) * (s * x ^ 2 + t * x + u)) →
  p ^ 2 + q ^ 2 + r ^ 2 + s ^ 2 + t ^ 2 + u ^ 2 = 23456 :=
by
  sorry

end polynomial_sum_of_squares_is_23456_l312_312535


namespace trailing_zeros_factorial_2014_l312_312071

-- Definitions based on the given conditions.
def countMultiples (n k : Nat) : Nat :=
  n / k

theorem trailing_zeros_factorial_2014 : 
  let num_5s := countMultiples 2014 5 + countMultiples 2014 25 + countMultiples 2014 125 + countMultiples 2014 625 
  in num_5s = 501 := 
by 
  sorry

end trailing_zeros_factorial_2014_l312_312071


namespace total_volume_l312_312300

open Real

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem total_volume {d_cylinder d_cone_top d_cone_bottom h_cylinder h_cone : ℝ}
  (h1 : d_cylinder = 2) (h2 : d_cone_top = 2) (h3 : d_cone_bottom = 1)
  (h4 : h_cylinder = 14) (h5 : h_cone = 4) :
  volume_cylinder (d_cylinder / 2) h_cylinder +
  volume_cone (d_cone_top / 2) h_cone =
  (46 / 3) * π :=
by
  sorry

end total_volume_l312_312300


namespace num_divisors_of_36_l312_312489

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312489


namespace prime_equally_spaced_on_unit_circle_l312_312074

theorem prime_equally_spaced_on_unit_circle :
  (∃! n : ℕ, n ≥ 2 ∧ Nat.Prime n ∧ (∀ z : Fin n → ℂ, (∀ i, complex.abs (z i) = 1) ∧ (∑ i, z i = 0) → (∃ d : ℂ, d ≠ 0 ∧ (∀ i, z i = complex.of_real (cos (2 * π * i / n : ℝ)) + complex.I * complex.of_real (sin (2 * π * i / n : ℝ))))))
  ∧ n = 2 ∨ n = 3 :=
by 
  sorry

end prime_equally_spaced_on_unit_circle_l312_312074


namespace isosceles_triangle_vertex_angle_l312_312036

theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  is_triangle a b c →
  is_isosceles a b c →
  (interior_angle a b c = 70 ∨ interior_angle b a c = 70 ∨ interior_angle b c a = 70) →
  (vertex_angle a b c = 40 ∨ vertex_angle a b c = 70) :=
by sorry

end isosceles_triangle_vertex_angle_l312_312036


namespace brother_highlighters_spent_l312_312941

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end brother_highlighters_spent_l312_312941


namespace find_certain_amount_l312_312541

def percent (p : ℝ) (n : ℝ) : ℝ := (p / 100) * n

theorem find_certain_amount (x : ℝ) (A : ℝ) (h1 : x = 840)
  (h2 : percent 25 x = percent 15 1500 - A) : A = 15 :=
by
  sorry

end find_certain_amount_l312_312541


namespace rihanna_money_left_l312_312633

-- Definitions of the item costs
def cost_of_mangoes : ℝ := 6 * 3
def cost_of_apple_juice : ℝ := 4 * 3.50
def cost_of_potato_chips : ℝ := 2 * 2.25
def cost_of_chocolate_bars : ℝ := 3 * 1.75

-- Total cost computation
def total_cost : ℝ := cost_of_mangoes + cost_of_apple_juice + cost_of_potato_chips + cost_of_chocolate_bars

-- Initial amount of money Rihanna has
def initial_money : ℝ := 50

-- Remaining money after the purchases
def remaining_money : ℝ := initial_money - total_cost

-- The theorem stating that the remaining money is $8.25
theorem rihanna_money_left : remaining_money = 8.25 := by
  -- Lean will require the proof here.
  sorry

end rihanna_money_left_l312_312633


namespace brendas_age_l312_312314

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l312_312314


namespace units_digit_G_1000_l312_312624

def modified_fermat_number (n : ℕ) : ℕ := 5^(5^n) + 6

theorem units_digit_G_1000 : (modified_fermat_number 1000) % 10 = 1 :=
by
  -- The proof goes here
  sorry

end units_digit_G_1000_l312_312624


namespace log_a_b_plus_log_b_a_l312_312042

noncomputable def log_property (a b : ℝ) : Prop :=
(a > 0 ∧ b > 0 ∧ a ≠ 1 ∧ b ≠ 1) ∧
(∀ x: ℝ, 0 < x → (log 10 x)^2 - 2 * log 10 x - 3 = 0 → (x = a ∨ x = b)) ∧
(log b / log a + log a / log b = -10 / 3)

theorem log_a_b_plus_log_b_a (a b : ℝ) : log_property a b := sorry

end log_a_b_plus_log_b_a_l312_312042


namespace left_handed_women_percentage_l312_312095

noncomputable section

variables (x y : ℕ) (percentage : ℝ)

-- Conditions
def right_handed_ratio := 3
def left_handed_ratio := 1
def men_ratio := 3
def women_ratio := 2

def total_population_by_hand := right_handed_ratio * x + left_handed_ratio * x -- i.e., 4x
def total_population_by_gender := men_ratio * y + women_ratio * y -- i.e., 5y

-- Main Statement
theorem left_handed_women_percentage (h1 : total_population_by_hand = total_population_by_gender) :
    percentage = 25 :=
by
  sorry

end left_handed_women_percentage_l312_312095


namespace parallel_vectors_acute_angle_condition_l312_312458

section
variables (a b : ℕ → ℝ)
def vector_a : ℕ → ℝ := λ i, [1, 1, 0].nth i 0
def vector_b : ℕ → ℝ := λ i, [-1, 0, 2].nth i 0

def add_vectors (u v : ℕ → ℝ) (k : ℝ) : ℕ → ℝ :=
λ i, u i + k * v i

def dot_product (u v : ℕ → ℝ) : ℝ :=
(u 0 * v 0) + (u 1 * v 1) + (u 2 * v 2)

theorem parallel_vectors (k : ℝ) :
  (∀ i, add_vectors vector_a vector_b k i = 2 * vector_a i + vector_b i) →
  k = 1/2 :=
sorry

theorem acute_angle_condition (k : ℝ) :
  dot_product (add_vectors vector_a vector_b k) (λ i, 2 * vector_a i + vector_b i) > 0 →
  k > -1 ∧ k ≠ 1/2 :=
sorry
end

end parallel_vectors_acute_angle_condition_l312_312458


namespace perp_lines_angles_l312_312955

theorem perp_lines_angles (α1 α2 : ℝ) (l1 l2 : Type*)
  [linear_ordered_field α1] [linear_ordered_field α2] (h_perp : l1 ⊥ l2) :
  (|α1 - α2| = 90) : ℝ :=
begin
  sorry
end

end perp_lines_angles_l312_312955


namespace largest_n_for_factored_polynomial_l312_312864

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l312_312864


namespace ceil_neg_3_7_l312_312367

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l312_312367


namespace floor_sum_inequality_l312_312173

theorem floor_sum_inequality (x : ℝ) (n : ℕ) (h : n > 0) : 
  (\sum k in Finset.range n + 1, ((⌊k * x⌋₊ : ℝ) / k) ) ≤ ⌊ (n : ℝ) * x ⌋₊ := sorry

end floor_sum_inequality_l312_312173


namespace y_explicit_and_range_l312_312420

theorem y_explicit_and_range (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2*(m-1)*x1 + m + 1 = 0) (h2 : x2^2 - 2*(m-1)*x2 + m + 1 = 0) :
  x1 + x2 = 2*(m-1) ∧ x1 * x2 = m + 1 ∧ (x1^2 + x2^2 = 4*m^2 - 10*m + 2) 
  ∧ ∀ (y : ℝ), (∃ m, y = 4*m^2 - 10*m + 2) → y ≥ 6 :=
by
  sorry

end y_explicit_and_range_l312_312420


namespace complex_number_solution_l312_312534

open Complex

noncomputable def solve_z (a : ℝ) (h : a ≥ 0) : ℂ :=
(a - real.sqrt (a ^ 2 + 4)) / 2 * I

theorem complex_number_solution (a : ℝ) (h : a ≥ 0) (z : ℂ) (hz : z * abs z + a * z + I = 0) :
  z = solve_z a h :=
begin
  sorry
end

end complex_number_solution_l312_312534


namespace parallelogram_distance_sum_constant_quadrilateral_distance_line_l312_312274

-- Part (a)
theorem parallelogram_distance_sum_constant (A B C D X : Point) (h : IsParallelogram A B C D) :
  AX^2 + CX^2 - BX^2 - DX^2 = 0 := 
sorry

-- Part (b)
theorem quadrilateral_distance_line (A B C D X : Point) (h1 : ¬IsParallelogram A B C D)
  (h2 : AX^2 + CX^2 = BX^2 + DX^2) : 
  ∃ l : Line, (∀ X, OnLine X l ↔ AX^2 + CX^2 = BX^2 + DX^2) ∧ l ⟂ LineConnecting (Midpoint A C) (Midpoint B D) :=
sorry

end parallelogram_distance_sum_constant_quadrilateral_distance_line_l312_312274


namespace approximation_accuracy_l312_312284

noncomputable def radius (k : Circle) : ℝ := sorry
def BG_equals_radius (BG : ℝ) (r : ℝ) := BG = r
def DB_equals_radius_sqrt3 (DB DG r : ℝ) := DB = DG ∧ DG = r * Real.sqrt 3
def cos_beta (cos_beta : ℝ) := cos_beta = 1 / (2 * Real.sqrt 3)
def sin_beta (sin_beta : ℝ) := sin_beta = Real.sqrt 11 / (2 * Real.sqrt 3)
def angle_BCH (angle_BCH : ℝ) (beta : ℝ) := angle_BCH = 120 - beta
def side_nonagon (a_9 r : ℝ) := a_9 = 2 * r * Real.sin 20
def bounds_sin_20 (sin_20 : ℝ) := 0.34195 < sin_20 ∧ sin_20 < 0.34205
def error_margin_low (BH_low a_9 r : ℝ) := 0.6839 * r < a_9
def error_margin_high (BH_high a_9 r : ℝ) := a_9 < 0.6841 * r

theorem approximation_accuracy
  (r : ℝ) (BG DB DG : ℝ) (beta : ℝ) (a_9 BH_low BH_high : ℝ)
  (h1 : BG_equals_radius BG r)
  (h2 : DB_equals_radius_sqrt3 DB DG r)
  (h3 : cos_beta (1 / (2 * Real.sqrt 3)))
  (h4 : sin_beta (Real.sqrt 11 / (2 * Real.sqrt 3)))
  (h5 : angle_BCH (120 - beta) beta)
  (h6 : side_nonagon a_9 r)
  (h7 : bounds_sin_20 (Real.sin 20))
  (h8 : error_margin_low BH_low a_9 r)
  (h9 : error_margin_high BH_high a_9 r) : 
  0.6861 * r < BH_high ∧ BH_low < 0.6864 * r :=
sorry

end approximation_accuracy_l312_312284


namespace num_divisors_36_l312_312500

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312500


namespace divisor_between_40_and_50_l312_312375

theorem divisor_between_40_and_50 (n : ℕ) (h1 : 40 ≤ n) (h2 : n ≤ 50) (h3 : n ∣ (2^36 - 1)) : n = 49 :=
sorry

end divisor_between_40_and_50_l312_312375


namespace artifacts_per_wing_l312_312782

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end artifacts_per_wing_l312_312782


namespace part1_part2_part3_l312_312018

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 2 then x else sorry -- given only these conditions, implementation of f(x) when x ∈ (0, 2] is x

theorem part1 (x : ℝ) (h : 2 < x ∧ x ≤ 4) : f x = 1 / (x - 2) :=
sorry

theorem part2 (m : ℝ) (hm: f m = 1) : m = 1 ∨ m = 3 :=
sorry

theorem part3 : ∑ i in (finset.range 2015).map (λ i, i+1) f = 4535 / 2 :=
sorry

end part1_part2_part3_l312_312018


namespace channel_bottom_width_l312_312195

theorem channel_bottom_width
  (area : ℝ)
  (top_width : ℝ)
  (depth : ℝ)
  (h_area : area = 880)
  (h_top_width : top_width = 14)
  (h_depth : depth = 80) :
  ∃ (b : ℝ), b = 8 ∧ area = (1/2) * (top_width + b) * depth := 
by
  sorry

end channel_bottom_width_l312_312195


namespace a_put_his_oxen_for_grazing_for_7_months_l312_312733

theorem a_put_his_oxen_for_grazing_for_7_months
  (x : ℕ)
  (a_oxen : ℕ := 10)
  (b_oxen : ℕ := 12)
  (b_months : ℕ := 5)
  (c_oxen : ℕ := 15)
  (c_months : ℕ := 3)
  (total_rent : ℝ := 105)
  (c_share : ℝ := 27) :
  (c_share / total_rent = (c_oxen * c_months) / ((a_oxen * x) + (b_oxen * b_months) + (c_oxen * c_months))) → (x = 7) :=
by
  sorry

end a_put_his_oxen_for_grazing_for_7_months_l312_312733


namespace smallest_perimeter_consecutive_even_triangle_l312_312251

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l312_312251


namespace solve_for_y_l312_312184

theorem solve_for_y (y : ℤ) (h : 5^(y + 1) = 625) : y = 3 := by
  sorry

end solve_for_y_l312_312184


namespace trapezoid_ratio_of_bases_l312_312660

theorem trapezoid_ratio_of_bases (a b : ℝ) (h : a > b)
    (H : (1 / 4) * (1 / 2) * (a - b) * h = (1 / 8) * (a + b) * h) : 
    a / b = 3 := 
sorry

end trapezoid_ratio_of_bases_l312_312660


namespace trapezoid_ratio_of_bases_l312_312661

theorem trapezoid_ratio_of_bases (a b : ℝ) (h : a > b)
    (H : (1 / 4) * (1 / 2) * (a - b) * h = (1 / 8) * (a + b) * h) : 
    a / b = 3 := 
sorry

end trapezoid_ratio_of_bases_l312_312661


namespace rectangular_tables_have_7_chairs_l312_312800

theorem rectangular_tables_have_7_chairs :
  ∀ (x : ℕ),
  let round_tables := 2 in
  let chairs_per_round_table := 6 in
  let rectangular_tables := 2 in
  let total_chairs := 26 in
  (round_tables * chairs_per_round_table + rectangular_tables * x = total_chairs) →
  x = 7 :=
by
  intro x
  let round_tables := 2
  let chairs_per_round_table := 6
  let rectangular_tables := 2
  let total_chairs := 26
  intro h
  sorry

end rectangular_tables_have_7_chairs_l312_312800


namespace six_digit_start_5_not_possible_l312_312275

theorem six_digit_start_5_not_possible :
  ∀ n : ℕ, (n ≥ 500000 ∧ n < 600000) → (¬ ∃ m : ℕ, (n * 10^6 + m) ^ 2 < 10^12 ∧ (n * 10^6 + m) ^ 2 ≥ 5 * 10^11 ∧ (n * 10^6 + m) ^ 2 < 6 * 10^11) :=
sorry

end six_digit_start_5_not_possible_l312_312275


namespace area_codes_even_product_l312_312310

def digits := {2, 4, 5}

theorem area_codes_even_product : ∃ n : ℕ, n = 26 ∧ ∀ (a b c : ℕ), a ∈ digits → b ∈ digits → c ∈ digits → (a * b * c).even → 3 ∣ (a + b + c - 6) :=
by
  sorry

end area_codes_even_product_l312_312310


namespace non_neg_int_solutions_l312_312833

theorem non_neg_int_solutions (n : ℕ) (a b : ℤ) :
  n^2 = a + b ∧ n^3 = a^2 + b^2 → n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end non_neg_int_solutions_l312_312833


namespace sum_of_squares_of_coeffs_eq_79_l312_312268

-- Define the expression 5(y^2 - 3y + 3) - 6(y^3 - 2y + 2)
def expr : ℚ[X] := 5 * (X^2 - 3 * X + 3) - 6 * (X^3 - 2 * X + 2)

-- Prove that the sum of the squares of the coefficients of the simplified expression is 79
theorem sum_of_squares_of_coeffs_eq_79 :
  (expr.coeff 3)^2 + (expr.coeff 2)^2 + (expr.coeff 1)^2 + (expr.coeff 0)^2 = 79 :=
by
  sorry

end sum_of_squares_of_coeffs_eq_79_l312_312268


namespace john_owes_more_than_twice_l312_312990

noncomputable def compound_interest_condition (P₀ : ℝ) (r : ℝ) (t : ℕ) : Prop :=
  P₀ * (1 + r)^t > 2 * P₀

theorem john_owes_more_than_twice (P₀ : ℝ) (r : ℝ) (t : ℕ) (h : P₀ = 1500) (hr : r = 0.03) :
  (∃ t : ℕ, compound_interest_condition P₀ r t) → 
  ∃ t : ℕ, compound_interest_condition P₀ r t ∧ t = 25 :=
begin
  sorry
end

end john_owes_more_than_twice_l312_312990


namespace num_divisors_of_36_l312_312490

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312490


namespace gcd_plus_lcm_18_30_45_l312_312596

noncomputable def gcd_18_30_45 : Nat := Nat.gcd (Nat.gcd 18 30) 45

noncomputable def lcm_18_30_45 : Nat := Nat.lcm (Nat.lcm 18 30) 45

theorem gcd_plus_lcm_18_30_45 : gcd_18_30_45 + lcm_18_30_45 = 93 := by
  have h_gcd : gcd_18_30_45 = 3 := by sorry
  have h_lcm : lcm_18_30_45 = 90 := by sorry
  rw [h_gcd, h_lcm]
  rfl

end gcd_plus_lcm_18_30_45_l312_312596


namespace gcf_lcm_sum_l312_312594

-- Definitions for GCD and LCM
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Let's define a function to get GCD and LCM for three numbers by pairwise computation
def gcd_three (a b c : ℕ) : ℕ := gcd a (gcd b c)
def lcm_three (a b c : ℕ) : ℕ := lcm a (lcm b c)

theorem gcf_lcm_sum :
  let C := gcd_three 18 30 45,
  let D := lcm_three 18 30 45
  in C + D = 93 := 
by
  -- Define the GCD and LCM of the three numbers
  let C := gcd_three 18 30 45
  let D := lcm_three 18 30 45
  -- We use sorry to skip the proof
  sorry

end gcf_lcm_sum_l312_312594


namespace find_f_2017_l312_312419

-- Define the function f
def f (x : ℝ) (a α b β : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

-- State the theorem with the given conditions
theorem find_f_2017 (a α b β : ℝ) (h : f 2016 a α b β = 5) : f 2017 a α b β = 3 :=
by
  sorry

end find_f_2017_l312_312419


namespace find_stock_rate_l312_312850

theorem find_stock_rate (annual_income : ℝ) (investment_amount : ℝ) (R : ℝ) 
  (h1 : annual_income = 2000) (h2 : investment_amount = 6800) : 
  R = 2000 / 6800 :=
by
  sorry

end find_stock_rate_l312_312850


namespace tan_A_eq_sqrt_3_l312_312427

-- Definitions given in the problem
def perimeter (a b c : ℝ) := a + b + c
def inradius (a b c : ℝ) (r : ℝ) := r
def sideBC (b c : ℝ) := b = c
def tanA (a b c : ℝ) : ℝ := (2 * tan (0.5 * atan ((sqrt (3)) / 3))) / (1 - (tan (0.5 * atan ((sqrt (3)) / 3)))^2)

-- Statement to prove
theorem tan_A_eq_sqrt_3 (a b c : ℝ) (r : ℝ) (h1 : perimeter a b c = 20)
                            (h2 : inradius a b c r = sqrt 3) (h3 : b = 7) :
                            tanA a b c = sqrt 3 :=
by
  sorry

end tan_A_eq_sqrt_3_l312_312427


namespace subset_implies_all_elements_l312_312187

variable {U : Type}

theorem subset_implies_all_elements (P Q : Set U) (hPQ : P ⊆ Q) (hP_nonempty : P ≠ ∅) (hQ_nonempty : Q ≠ ∅) :
  ∀ x ∈ P, x ∈ Q :=
by 
  sorry

end subset_implies_all_elements_l312_312187


namespace jaydee_typing_time_l312_312577

theorem jaydee_typing_time : 
  (∀ (wpm total_words : ℕ) (minutes_per_hour : ℕ),
    wpm = 38 ∧ total_words = 4560 ∧ minutes_per_hour = 60 → 
      (total_words / wpm) / minutes_per_hour = 2) :=
begin
  intros wpm total_words minutes_per_hour h,
  cases h with hwpm hwords_hours,
  cases hwords_hours with hwords hhours,
  rw [hwpm, hwords, hhours],
  norm_num,
end

end jaydee_typing_time_l312_312577


namespace rent_of_apartment_l312_312235

theorem rent_of_apartment (utils groceries roommate_payment : ℕ) (split : ℕ → ℕ → ℕ) : 
  utils = 114 → 
  groceries = 300 → 
  roommate_payment = 757 → 
  split (R + utils + groceries) 2 = roommate_payment → 
  R = 1100 :=
by
  intros h_utils h_groceries h_roommate_payment h_split
  rw [h_utils, h_groceries, h_roommate_payment] at h_split
  apply eq_of_add_eq_add_right
  exact h_split

end rent_of_apartment_l312_312235


namespace sum_of_two_numbers_l312_312626

theorem sum_of_two_numbers (x y : ℕ) (h : x = 11) (h1 : y = 3 * x + 11) : x + y = 55 := by
  sorry

end sum_of_two_numbers_l312_312626


namespace white_cell_never_painted_l312_312126

theorem white_cell_never_painted (G : Type) [infinite G] [finite (initially_black_cells : set G)] 
    (M : set G) (covers_more_than_one : ∃ p q : G, p ∈ M ∧ q ∈ M ∧ p ≠ q)
    (shift : M → set G) (aligns_with : shift M = M)
    (paint_black : ∀ m ∈ M, (m ∉ initially_black_cells) → 
        shift (λ x, if x = m then True else False) = initially_black_cells ∪ {m}) :
    ∃ w ∈ G, w ∉ initially_black_cells ∧ ∀ k, k ∈ shift M → w ∉ k := 
sorry

end white_cell_never_painted_l312_312126


namespace circles_intersect_l312_312455

def circle1 : set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 + 1)^2 = 1}
def circle2 : set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 + 3)^2 = 4}

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 := sorry

end circles_intersect_l312_312455


namespace angle_ECB_is_30_l312_312091

-- Define the geometric entities and properties
variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
variables [MetricSpace D] [MetricSpace E]
variables (ABC : Triangle A B C) (C_is_mid : Midpoint D C)
variables (C_is_alt : Altitude D A B C)

-- Given conditions
variables (AC_eq_BC : dist A C = dist B C)
variables (AB_eq_BC : dist A B = dist B C)
variables (angle_ACB_60 : angle A C B = 60 * pi / 180)
variables (CD_is_altitude : alt ABC C D)
variables (E_is_midpoint : dist D E = dist E C)

-- The statement of the proof problem
theorem angle_ECB_is_30 : angle E C B = 30 * pi / 180 :=
by sorry

end angle_ECB_is_30_l312_312091


namespace minimum_value_f_l312_312872

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2)

theorem minimum_value_f (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ (a : ℝ), (∀ (b : ℝ), f x y >= b) ∧ a = 10 := sorry

end minimum_value_f_l312_312872


namespace trapezoid_base_ratio_l312_312647

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l312_312647


namespace correct_options_l312_312270

-- Statement (A) verification
def slope_greater_than_45 (slope: ℚ) : Prop :=
  slope > 1

def line_a : Prop :=
  slope_greater_than_45 (5 / 4)

-- Statement (B) verification
def point_on_line_b (x y : ℝ) (m : ℝ) : Prop :=
  (2 + m) * x + 4 * y - 2 + m = 0

def check_point_on_line_b : Prop :=
  ∀ m : ℝ, point_on_line_b (-1) 1 m

-- Statement (C) verification
def distance_between_lines_c : Prop :=
  let A₁ := 1 in let B₁ := 1 in let C₁ := -1 in
  let A₂ := 2 in let B₂ := 2 in let C₂ := 1 in
  abs (C₂ + 2) / real.sqrt (A₂^2 + B₂^2) ≠ real.sqrt 2

-- Statement (D) verification
def exact_lines_between_points_d : Prop :=
  let d := real.sqrt ((-1 - 3)^2 + (2 + 1)^2) in
  let r₁ := 1 in let r₂ := 4 in
  d = r₁ + r₂ → false

-- Final statement combining all correct results for A and B, not C and D
def final_verification : Prop :=
  line_a ∧ check_point_on_line_b ∧ distance_between_lines_c ∧ exact_lines_between_points_d

-- Concluding the correctness of A and B
theorem correct_options : final_verification :=
by {
  unfold final_verification line_a check_point_on_line_b distance_between_lines_c exact_lines_between_points_d,
  split,
  {
    -- Proof for A here
    sorry,
  },
  {
    split,
    {
      -- Proof for B here
      sorry,
    },
    {
      split,
      {
        -- Proof for C here
        sorry,
      },
      {
        -- Proof for D here
        sorry,
      }
    }
  }
}

end correct_options_l312_312270


namespace car_dealership_sales_l312_312324

theorem car_dealership_sales (x : ℕ)
  (h1 : 5 * x = 30 * 8)
  (h2 : 30 + x = 78) : 
  x = 48 :=
sorry

end car_dealership_sales_l312_312324


namespace radius_inscribed_in_XYZ_l312_312716

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inscribed_radius (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  let K := area a b c
  in K / s

theorem radius_inscribed_in_XYZ :
  inscribed_radius 7 8 9 = Real.sqrt 5 :=
by
  sorry

end radius_inscribed_in_XYZ_l312_312716


namespace num_divisors_of_36_l312_312519

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312519


namespace quadratic_tangent_x_axis_l312_312087

theorem quadratic_tangent_x_axis
  (a b : ℝ) (h1 : a ≠ 0) (c : ℝ) (h2 : c = b^2 / (4 * a)) :
  ∃ x : ℝ, f(x) = ax^2 + bx + c ∧ f(x) = 0 ∧ derivative (λ x, f(x)) x = 0 := 
begin
  sorry
end

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

end quadratic_tangent_x_axis_l312_312087


namespace smallest_triangle_perimeter_l312_312259

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l312_312259


namespace conic_section_is_ellipse_l312_312347

theorem conic_section_is_ellipse (x y : ℝ) :
  sqrt (x^2 + (y - 4)^2) + sqrt ((x - 6)^2 + (y + 2)^2) = 12 →
  "E" := 
begin
  sorry
end

end conic_section_is_ellipse_l312_312347


namespace probability_of_pq_6p_3q_eq_3_l312_312076

theorem probability_of_pq_6p_3q_eq_3 :
  (∃ p ∈ finset.range 15, ∃ q ∈ int, (p + 1) * q = 6 * (p + 1) + 3 * q + 3) / 15 = 1 / 5 :=
sorry

end probability_of_pq_6p_3q_eq_3_l312_312076


namespace problem_part_1_problem_part_2_problem_part_3_problem_part_4_l312_312047

def f (x a : ℝ) : ℝ := x^2 - a * Real.log x

theorem problem_part_1 (x : ℝ) (h1 : 1 < x) :
  f x 2 > 0 := by sorry

theorem problem_part_2 (a : ℝ) :
  a ≤ 2 → (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f x a ≥ 1) := by sorry

theorem problem_part_3 (a : ℝ) :
  a ≥ 2 * Real.exp 1 ^ 2 → (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f x a ≥ Real.exp 1 ^ 2 - a) := by sorry

theorem problem_part_4 (a : ℝ) :
  2 < a ∧ a < 2 * Real.exp 1 ^ 2 → (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f x a ≥ - Real.log (a / 2) / 2) := by sorry

end problem_part_1_problem_part_2_problem_part_3_problem_part_4_l312_312047


namespace polynomial_root_sum_nonnegative_l312_312689

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem polynomial_root_sum_nonnegative 
  (m1 m2 k1 k2 b c p q : ℝ)
  (h1 : f m1 b c = 0) (h2 : f m2 b c = 0)
  (h3 : g k1 p q = 0) (h4 : g k2 p q = 0) :
  f k1 b c + f k2 b c + g m1 p q + g m2 p q ≥ 0 := 
by
  sorry  -- Proof placeholders

end polynomial_root_sum_nonnegative_l312_312689


namespace tangent_point_at_slope_one_l312_312435

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem proof problem
theorem tangent_point_at_slope_one : ∃ x : ℝ, derivative x = 1 ∧ x = 2 :=
by
  sorry

end tangent_point_at_slope_one_l312_312435


namespace circle_radius_l312_312298

theorem circle_radius (AB BM MC AD : ℝ) (h1 : AB = 10) (h2 : BM = 6) (h3 : MC = 6) (h4 : AD = 20) :
  let r := AD / 2
  in r = 10 :=
by
  let r := AD / 2
  sorry

end circle_radius_l312_312298


namespace cannot_form_basis_with_b_plus_c_l312_312886

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, 0)
def vector_c : ℝ × ℝ := (2, 4)
def vector_b_plus_c : ℝ × ℝ := (vector_b.1 + vector_c.1, vector_b.2 + vector_c.2)

theorem cannot_form_basis_with_b_plus_c :
  ¬ linear_independent ℝ ![vector_a, vector_b_plus_c] := by
sorry

end cannot_form_basis_with_b_plus_c_l312_312886


namespace number_of_divisors_of_36_l312_312504

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312504


namespace tan_A_of_triangle_l312_312426

theorem tan_A_of_triangle (A B C : Type)
  [Triangle A B C] 
  (perimeter_ABC : Perimeter A B C = 20)
  (radius_incircle : InscribedCircleRadius A B C = sqrt 3)
  (BC_eq_seven : SideLength B C = 7) :
  tan (Angle A) = sqrt 3 := 
sorry

end tan_A_of_triangle_l312_312426


namespace max_n_for_factorable_quadratic_l312_312861

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l312_312861


namespace g_periodic_f_2008_l312_312147

noncomputable def f : ℝ → ℝ := sorry

def g (x : ℝ) := f(x) - x

theorem g_periodic :
  (∀ x : ℝ, f(x + 3) ≤ f(x) + 3) →
  (∀ x : ℝ, f(x + 2) ≥ f(x) + 2) →
  ∃ T > 0, ∀ x : ℝ, g(x + T) = g(x) :=
by
  intros h1 h2
  use 6
  sorry

theorem f_2008 :
  (∀ x : ℝ, f(x + 3) ≤ f(x) + 3) →
  (∀ x : ℝ, f(x + 2) ≥ f(x) + 2) →
  f(994) = 992 →
  f(2008) = 2006 :=
by
  intros h1 h2 h3
  have g_period := g_periodic h1 h2
  sorry

end g_periodic_f_2008_l312_312147


namespace four_digit_numbers_with_two_identical_digits_l312_312381

open Nat

theorem four_digit_numbers_with_two_identical_digits :
  let valid_digits := {0,1,3,4,5,6,7,8,9} in
  let positions := {22xy, 2x2y, 2xy2, 2xxy, 2xyx, 2yxx} in
  let choices := λ x y, x ≠ y ∧ x ∈ valid_digits ∧ y ∈ valid_digits in
  let count_valid_numbers := 3 * 8 * 8 in
  2 * count_valid_numbers = 384 :=
by
  sorry

end four_digit_numbers_with_two_identical_digits_l312_312381


namespace rotate_triangle_360_hypotenuse_makes_two_cones_l312_312794

/-- 
A right-angled triangle is rotated 360 degrees around its hypotenuse.
We must show that the resulting spatial geometric body consists of two cones.
-/
theorem rotate_triangle_360_hypotenuse_makes_two_cones 
  {α β : ℝ} (hα : α > 0) (hβ : β > 0) :
  let γ := (α^2 + β^2)^(1/2)
  (rotate_360 (right_angle_triangle α β)) around γ = two_cones (height (γ, α, β)) := 
sorry

end rotate_triangle_360_hypotenuse_makes_two_cones_l312_312794


namespace ten_yuan_notes_count_l312_312138

theorem ten_yuan_notes_count (total_notes : ℕ) (total_change : ℕ) (item_cost : ℕ) (change_given : ℕ → ℕ → ℕ) (is_ten_yuan_notes : ℕ → Prop) :
    total_notes = 16 →
    total_change = 95 →
    item_cost = 5 →
    change_given 10 5 = total_change →
    (∃ x y : ℕ, x + y = total_notes ∧ 10 * x + 5 * y = total_change ∧ is_ten_yuan_notes x) → is_ten_yuan_notes 3 :=
by
  sorry

end ten_yuan_notes_count_l312_312138


namespace no_prime_roots_of_quadratic_l312_312328

theorem no_prime_roots_of_quadratic :
  ∀ (k : ℝ), ¬ (∃ p q : ℝ, nat.prime (int.of_nat p) ∧ nat.prime (int.of_nat q) ∧ p + q = 65 ∧ p * q = k) :=
by
  intro k
  sorry

end no_prime_roots_of_quadratic_l312_312328


namespace trapezoid_base_ratio_l312_312653

theorem trapezoid_base_ratio
  (a b h : ℝ)  -- lengths of the bases and height
  (ha_gt_hb : a > b)
  (trapezoid_area : ℝ) 
  (quad_area : ℝ) 
  (h1 : trapezoid_area = (1/2) * (a + b) * h) 
  (h2 : quad_area = (1/2) * (a - b) * h / 4)
  (h3 : quad_area = trapezoid_area / 4) :
  a / b = 3 :=
by {
  sorry,
}

end trapezoid_base_ratio_l312_312653


namespace sphere_hemisphere_cone_volume_ratio_l312_312687

theorem sphere_hemisphere_cone_volume_ratio (r : ℝ) (π : Real) :
  let V_sphere := (4/3) * π * r^3
  let V_hemisphere := (1/2) * (4/3) * π * (3 * r)^3
  let V_cone := (1/3) * π * r^2 * (2 * r)
  V_sphere / (V_hemisphere + V_cone) = (1 : ℝ) / 14 :=
by
  let V_sphere := (4/3) * π * r^3
  let V_hemisphere := (1/2) * (4/3) * π * (3 * r)^3
  let V_cone := (1/3) * π * r^2 * (2 * r)
  have V_combined : V_hemisphere + V_cone = 18 * π * r^3 + (2/3) * π * r^3 := sorry
  have ratio_sphere_combined : V_sphere / V_combined = (1 : ℝ) / 14 := sorry
  exact ratio_sphere_combined

end sphere_hemisphere_cone_volume_ratio_l312_312687


namespace billy_restaurant_bill_l312_312741

def adults : ℕ := 2
def children : ℕ := 5
def meal_cost : ℕ := 3

def total_people : ℕ := adults + children
def total_bill : ℕ := total_people * meal_cost

theorem billy_restaurant_bill : total_bill = 21 := 
by
  -- This is the placeholder for the proof.
  sorry

end billy_restaurant_bill_l312_312741


namespace work_completion_time_l312_312280

theorem work_completion_time (x_work_rate y_work_rate z_work_rate: ℝ) 
  (x_days y_days z_days: ℕ)
  (hx: x_work_rate = 1 / 20) 
  (hy: y_work_rate = 1 / 12) 
  (hz: z_work_rate = 1 / 15) 
  (hx_days: x_days = 4) 
  (hy_days: y_days = 3) 
  (hz_days: y_days + hx_days + hz_days = 9) : 
  4 * x_work_rate + 3 * (x_work_rate + y_work_rate) + (9 - 7) * (x_work_rate + y_work_rate + z_work_rate) = 1 := 
sorry

end work_completion_time_l312_312280


namespace area_of_circle_l312_312382

noncomputable def area_of_region (x y : ℝ) : ℝ :=
  π * 23

theorem area_of_circle :
  ∀ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 10 = 0 → area_of_region x y = 23 * π :=
by
  intros x y h
  sorry

end area_of_circle_l312_312382


namespace problem1_problem2_l312_312068

-- Define the vectors a, b, and c
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)
def vector_c : ℝ × ℝ := (-1, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Problem 1: Determine the angle θ between vectors a and c given x = π / 3
theorem problem1 (x : ℝ) (h : x = Real.pi / 3) :
  let a := vector_a x in
  let c := vector_c in
  let cos_theta := (dot_product a c) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (c.1 ^ 2 + c.2 ^ 2)) in
  cos_theta = - (Real.sqrt 3 / 2) :=
sorry

-- Problem 2: Find λ given x ∈ [-3π/8, π/4] and f(x) has a maximum value of 1/2
theorem problem2 (λ : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-3 * Real.pi / 8) (Real.pi / 4) → 
  let f := λ * dot_product (vector_a x) (vector_b x) in 
  f ≤ 1/2 ∧ (∃ y : ℝ, y ∈ Set.Icc (-3 * Real.pi / 8) (Real.pi / 4) ∧ f = 1/2)) →
  λ = 1/2 :=
sorry

end problem1_problem2_l312_312068


namespace numberOfValidSequences_l312_312599

-- Define the set T and the sequence generation rule
def validTriple (b_1 b_2 b_3 : ℕ) : Prop :=
  1 ≤ b_1 ∧ b_1 ≤ 15 ∧ 1 ≤ b_2 ∧ b_2 ≤ 15 ∧ 1 ≤ b_3 ∧ b_3 ≤ 15

def generatesZeroSequence (b_1 b_2 b_3 : ℕ) : Prop :=
  ∃ n ≥ 4, ∃ b : ℕ → ℕ, b 1 = b_1 ∧ b 2 = b_2 ∧ b 3 = b_3 ∧
  (∀ n ≥ 4, b n = b (n - 1) * |b (n - 2) - b (n - 3)|) ∧ b n = 0

theorem numberOfValidSequences : ∑ b_1 in finset.range 16, ∑ b_2 in finset.range 16, ∑ b_3 in finset.range 16, 
  if validTriple b_1 b_2 b_3 ∧ generatesZeroSequence b_1 b_2 b_3 then 1 else 0 = 1155 := 
sorry

end numberOfValidSequences_l312_312599


namespace national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l312_312719

-- Question 5
theorem national_currency_depreciation (term : String) : term = "Devaluation" := 
sorry

-- Question 6
theorem bond_annual_coupon_income 
  (purchase_price face_value annual_yield annual_coupon : ℝ) 
  (h_price : purchase_price = 900)
  (h_face : face_value = 1000)
  (h_yield : annual_yield = 0.15) 
  (h_coupon : annual_coupon = 135) : 
  annual_coupon = annual_yield * purchase_price := 
sorry

-- Question 7
theorem dividend_yield 
  (num_shares price_per_share total_dividends dividend_yield : ℝ)
  (h_shares : num_shares = 1000000)
  (h_price : price_per_share = 400)
  (h_dividends : total_dividends = 60000000)
  (h_yield : dividend_yield = 15) : 
  dividend_yield = (total_dividends / num_shares / price_per_share) * 100 :=
sorry

-- Question 8
theorem tax_deduction 
  (insurance_premium annual_salary tax_return : ℝ)
  (h_premium : insurance_premium = 120000)
  (h_salary : annual_salary = 110000)
  (h_return : tax_return = 14300) : 
  tax_return = 0.13 * min insurance_premium annual_salary := 
sorry

end national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l312_312719


namespace line_intersects_y_axis_at_point_intersection_at_y_axis_l312_312813

theorem line_intersects_y_axis_at_point :
  ∃ y, 5 * 0 - 7 * y = 35 := sorry

theorem intersection_at_y_axis :
  (∃ y, 5 * 0 - 7 * y = 35) → 0 - 7 * (-5) = 35 := sorry

end line_intersects_y_axis_at_point_intersection_at_y_axis_l312_312813


namespace speed_of_stream_l312_312278

variable {D : ℝ} (v : ℝ)

-- Conditions
def speed_in_still_water : ℝ := 48
def time_upstream (D : ℝ) (v : ℝ) : ℝ := D / (speed_in_still_water - v)
def time_downstream (D : ℝ) (v : ℝ) : ℝ := D / (speed_in_still_water + v)
def condition : Prop := (time_upstream D v) = 2 * (time_downstream D v)

-- Statement to prove
theorem speed_of_stream (h : condition D v) : v = 16 := by
  sorry

end speed_of_stream_l312_312278


namespace hyperbola_standard_eq_l312_312065

-- Define the conditions explicitly.
def is_hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0)

def distance_from_focus_to_asymptote (a b : ℝ) (focus : ℝ × ℝ) (distance : ℝ) : Prop := 
  let (f_x, f_y) := focus in
  (f_x * b) / (Real.sqrt (a^2 + b^2)) = distance

-- Lean statement to prove the standard equation of the hyperbola.
theorem hyperbola_standard_eq (a b : ℝ) (focus : ℝ × ℝ) (distance : ℝ) 
  (h_hyperbola : is_hyperbola a b)
  (h_focus : focus = (3, 0))
  (h_distance : distance = Real.sqrt 5)
  (h_condition : distance_from_focus_to_asymptote a b focus distance) :
  (a = 2) ∧ (b = Real.sqrt 5) ↔ (∃ c d : ℝ, (c = 4) ∧ (d = 5) ∧ ( ∀ x y : ℝ, (x^2 / c - y^2 / d = 1))) :=
by
  sorry

end hyperbola_standard_eq_l312_312065


namespace tan_diff_sin_double_l312_312070

theorem tan_diff (α : ℝ) (h : Real.tan α = 2) : 
  Real.tan (α - Real.pi / 4) = 1 / 3 := 
by 
  sorry

theorem sin_double (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end tan_diff_sin_double_l312_312070


namespace sum_f_from_1_to_2017_l312_312448

-- Define the piecewise function f
def f : ℤ → ℝ
| x := if x ≤ 0 then 2 ^ x else f(x - 2)

-- Define the theorem to prove the sum
theorem sum_f_from_1_to_2017 :
  let sum_f := ∑ i in finset.range 2017, f (i + 1)
  sum_f = 3025 / 2 :=
by
  sorry

end sum_f_from_1_to_2017_l312_312448


namespace ceil_neg_3_7_l312_312366

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l312_312366


namespace four_digit_number_divisibility_l312_312639

theorem four_digit_number_divisibility 
  (E V I L : ℕ) 
  (hE : 0 ≤ E ∧ E < 10) 
  (hV : 0 ≤ V ∧ V < 10) 
  (hI : 0 ≤ I ∧ I < 10) 
  (hL : 0 ≤ L ∧ L < 10)
  (h1 : (1000 * E + 100 * V + 10 * I + L) % 73 = 0) 
  (h2 : (1000 * V + 100 * I + 10 * L + E) % 74 = 0)
  : 1000 * L + 100 * I + 10 * V + E = 5499 := 
  sorry

end four_digit_number_divisibility_l312_312639


namespace range_a_range_k_l312_312917

def f (x : ℝ) (a : ℝ) : ℝ := (1 / x) - x + a * Real.log x

theorem range_a (x1 x2 a : ℝ) (h1 : x2 > x1) (h2 : x1 > 0) (h3 : x2 > 0) 
(h4 : ∃ x1 x2, x1 + x2 = a ∧ x1 * x2 = 1) : a ∈ Ioi (2 : ℝ) := 
sorry

theorem range_k (x1 x2 a k : ℝ) (h1 : x2 > x1) (h2 : x1 > 0) (h3 : x2 > 0) 
(h4 : f x2 a - f x1 a ≥ k * a - 3) (h5 : ∃ x1 x2, x1 + x2 = a ∧ x1 * x2 = 1) : 
k ∈ Iic (2 * Real.log 2) :=
sorry

end range_a_range_k_l312_312917


namespace trapezoid_base_ratio_l312_312656

theorem trapezoid_base_ratio
  (a b h : ℝ)  -- lengths of the bases and height
  (ha_gt_hb : a > b)
  (trapezoid_area : ℝ) 
  (quad_area : ℝ) 
  (h1 : trapezoid_area = (1/2) * (a + b) * h) 
  (h2 : quad_area = (1/2) * (a - b) * h / 4)
  (h3 : quad_area = trapezoid_area / 4) :
  a / b = 3 :=
by {
  sorry,
}

end trapezoid_base_ratio_l312_312656


namespace ceil_neg_3_7_l312_312362

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := sorry

end ceil_neg_3_7_l312_312362


namespace optimal_discount_order_l312_312200

variables (p : ℝ) (d1 : ℝ) (d2 : ℝ)

-- Original price of "Stars Beyond" is 30 dollars
def original_price : ℝ := 30

-- Fixed discount is 5 dollars
def discount_5 : ℝ := 5

-- 25% discount represented as a multiplier
def discount_25 : ℝ := 0.75

-- Applying $5 discount first and then 25% discount
def price_after_5_then_25_discount := discount_25 * (original_price - discount_5)

-- Applying 25% discount first and then $5 discount
def price_after_25_then_5_discount := (discount_25 * original_price) - discount_5

-- The additional savings when applying 25% discount first
def additional_savings := price_after_5_then_25_discount - price_after_25_then_5_discount

theorem optimal_discount_order : 
  additional_savings = 1.25 :=
sorry

end optimal_discount_order_l312_312200


namespace correct_number_of_statements_l312_312999

theorem correct_number_of_statements (a b : ℤ) :
  (¬ (∃ h₁ : Even (a + 5 * b), ¬ Even (a - 7 * b)) ∧
   ∃ h₂ : a + b % 3 = 0, ¬ ((a % 3 = 0) ∧ (b % 3 = 0)) ∧
   ∃ h₃ : Prime (a + b), Prime (a - b)) →
   1 = 1 :=
by
  sorry

end correct_number_of_statements_l312_312999


namespace golden_ellipse_properties_l312_312894

-- Definitions based on given conditions
variables (a b c : ℝ) (e : ℝ := (Real.sqrt 5 - 1) / 2) (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
def ellipse_eq : Prop := (e * e + e - 1 = 0 ∧ a * a - c * c = b * b ∧ a * c = b * b)

-- Statement 1 (geometric sequence)
def statement_1 : Prop := a * a = b * b * c ∧ b * b = a * c

-- Statement 2 (angle between vectors)
def vector_EF1 : ℝ × ℝ := (-c, -b)
def vector_EB : ℝ × ℝ := (a, -b)
def statement_2 : Prop := vector_EF1.fst * vector_EB.fst + vector_EF1.snd * vector_EB.snd = 0

-- Statement 3 (inscribed circle passes through foci)
def rhombus_area := 2 * a * 2 * b
def inscribed_circle_radius := r : ℝ := a * b / (Real.sqrt (a * a + b * b))
def statement_3 : Prop := r = c

-- Final proof goal
theorem golden_ellipse_properties :
  ellipse_eq a b c e h_a_gt_b h_b_gt_0 →
  statement_1 a b c ∧
  statement_2 a b c ∧
  statement_3 a b c :=
sorry

end golden_ellipse_properties_l312_312894


namespace gym_membership_total_cost_l312_312132

-- Definitions for the conditions stated in the problem
def first_gym_monthly_fee : ℕ := 10
def first_gym_signup_fee : ℕ := 50
def first_gym_discount_rate : ℕ := 10
def first_gym_personal_training_cost : ℕ := 25
def first_gym_sessions_per_year : ℕ := 52

def second_gym_multiplier : ℕ := 3
def second_gym_monthly_fee : ℕ := 3 * first_gym_monthly_fee
def second_gym_signup_fee_multiplier : ℕ := 4
def second_gym_discount_rate : ℕ := 10
def second_gym_personal_training_cost : ℕ := 45
def second_gym_sessions_per_year : ℕ := 52

-- Proof of the total amount John paid in the first year
theorem gym_membership_total_cost:
  let first_gym_annual_cost := (first_gym_monthly_fee * 12) +
                                (first_gym_signup_fee * (100 - first_gym_discount_rate) / 100) +
                                (first_gym_personal_training_cost * first_gym_sessions_per_year)
  let second_gym_annual_cost := (second_gym_monthly_fee * 12) +
                                (second_gym_monthly_fee * second_gym_signup_fee_multiplier * (100 - second_gym_discount_rate) / 100) +
                                (second_gym_personal_training_cost * second_gym_sessions_per_year)
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  total_annual_cost = 4273 := by
  -- Declaration of the variables used in the problem
  let first_gym_annual_cost := 1465
  let second_gym_annual_cost := 2808
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  -- Simplify and verify the total cost
  sorry

end gym_membership_total_cost_l312_312132


namespace divisibility_mn_l312_312609

open Nat

theorem divisibility_mn (a b m n : ℕ) (h_coprime : coprime a b) (h_a_gt_one : 1 < a)
  (h_div : (a^m + b^m) % (a^n + b^n) = 0) : n ∣ m :=
sorry

end divisibility_mn_l312_312609


namespace find_x2_x1_x3_product_l312_312606

noncomputable def sqrt_4050 : ℝ := Real.sqrt 4050

def poly (x : ℝ) : ℝ := sqrt_4050 * x^3 - 8101 * x^2 + 4

theorem find_x2_x1_x3_product :
  ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0 ∧
  (x2 * (x1 + x3) = 2) :=
  sorry

end find_x2_x1_x3_product_l312_312606


namespace area_of_shaded_region_l312_312873

-- Define the vertices of the polygon
def vertices : List (ℝ × ℝ) := [(0, 0), (15, 0), (45, 30), (45, 45), (30, 45), (0, 15)]

-- Define the function to calculate the area of the polygon using the formula for the area of a polygon
def polygon_area (verts : List (ℝ × ℝ)) : ℝ :=
  let cross_product (a b : ℝ × ℝ) : ℝ := a.1 * b.2 - a.2 * b.1
  let shifted := verts.tail ++ verts.take 1
  0.5 * (List.sumr (List.zipWith cross_product verts shifted)).abs

-- Theorem to prove the area of the shaded region
theorem area_of_shaded_region : polygon_area vertices = 787.5 := by
  -- The proof will go here
  sorry

end area_of_shaded_region_l312_312873


namespace count_divisors_36_l312_312478

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312478


namespace find_points_on_line_with_distance_l312_312384

def point_on_line (t : ℝ) : ℝ × ℝ :=
  (-2 - real.sqrt 2 * t, 3 + real.sqrt 2 * t)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem find_points_on_line_with_distance :
  {p : ℝ × ℝ // (∃ t : ℝ, p = point_on_line t) ∧ distance p (-2, 3) = real.sqrt 2} =
  {(-3, 4), (-1, 2)} :=
sorry

end find_points_on_line_with_distance_l312_312384


namespace white_cell_never_painted_l312_312127

theorem white_cell_never_painted (G : Type) [infinite G] [finite (initially_black_cells : set G)] 
    (M : set G) (covers_more_than_one : ∃ p q : G, p ∈ M ∧ q ∈ M ∧ p ≠ q)
    (shift : M → set G) (aligns_with : shift M = M)
    (paint_black : ∀ m ∈ M, (m ∉ initially_black_cells) → 
        shift (λ x, if x = m then True else False) = initially_black_cells ∪ {m}) :
    ∃ w ∈ G, w ∉ initially_black_cells ∧ ∀ k, k ∈ shift M → w ∉ k := 
sorry

end white_cell_never_painted_l312_312127


namespace solids_of_revolution_l312_312321

def isSolidOfRevolution (solid : String) : Prop :=
  solid = "Cylinder" ∨ solid = "Sphere"

theorem solids_of_revolution :
  ∀ (solids : List String), solids = ["Cylinder", "Hexagonal pyramid", "Cube", "Sphere", "Tetrahedron"] →
    List.filter isSolidOfRevolution solids = ["Cylinder", "Sphere"] :=
by
  intros solids hsolids
  sorry

end solids_of_revolution_l312_312321


namespace artifacts_per_wing_l312_312783

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end artifacts_per_wing_l312_312783


namespace right_focus_circle_radius_tangent_asymptotes_l312_312060

noncomputable def radius_of_circle_centered_at_focus_tangent_to_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in
  let asymptote_slope := b / a in
  b

theorem right_focus_circle_radius_tangent_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let c := Real.sqrt (a^2 + b^2) in
  radius_of_circle_centered_at_focus_tangent_to_asymptotes a b ha hb = b :=
by
  let c := Real.sqrt (a^2 + b^2)
  simp [radius_of_circle_centered_at_focus_tangent_to_asymptotes]
  sorry

end right_focus_circle_radius_tangent_asymptotes_l312_312060


namespace circle_area_is_39_9_pi_l312_312629

noncomputable def point := (ℝ × ℝ)

def A : point := (4, 15)
def B : point := (14, 9)

def tangent_intersects_x_axis (ω : set point) : Prop :=
  ∃ x : ℝ, (∀ p ∈ ω, (tangent_line p).intersection_at(−x) = 0)

def on_circle (p : point) (ω : set point) : Prop :=
  ∃ r : ℝ, ∀ p' ∈ ω, dist p p' = r

def circle_area (ω : set point) :=
  let r := classical.some (exists_radius (on_circle A ω)) in
  π * r ^ 2

theorem circle_area_is_39_9_pi {ω : set point} 
  (h1 : on_circle A ω) 
  (h2 : on_circle B ω)
  (h3 : tangent_intersects_x_axis ω) :
  circle_area ω = 39.9 * π :=
sorry

end circle_area_is_39_9_pi_l312_312629


namespace monotonic_decreasing_interval_l312_312202

-- Define the function
def f : ℝ → ℝ := λ x, sin (π / 6 - x)

-- Define the domain
def domain : set ℝ := set.Icc 0 (3 * π / 2)

-- Define the interval where we claim the function is decreasing
def decreasing_interval : set ℝ := set.Icc 0 (2 * π / 3)

-- The proof statement
theorem monotonic_decreasing_interval :
  ∀ x₁ x₂ ∈ decreasing_interval, x₁ < x₂ → f x₁ ≥ f x₂ :=
by
  sorry

end monotonic_decreasing_interval_l312_312202


namespace arc_length_of_curve_l312_312283

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (x - x^2) - Real.arccos (sqrt x) + 5

theorem arc_length_of_curve :
  ∫ x in (1 / 9 : ℝ)..1, sqrt (1 + (Real.sqrt ((1 - x) / x)) ^ 2) = (4 : ℝ) / 3 :=
by sorry

end arc_length_of_curve_l312_312283


namespace first_alloy_mass_l312_312100

theorem first_alloy_mass (x : ℝ) : 
  (0.12 * x + 2.8) / (x + 35) = 9.454545454545453 / 100 → 
  x = 20 :=
by
  intro h
  sorry

end first_alloy_mass_l312_312100


namespace find_B_l312_312981

-- Define the translation function for points in ℝ × ℝ.
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

-- Given conditions
def A : ℝ × ℝ := (2, 2)
def A' : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-1, 1)

-- The vector v representing the translation from A to A'
def v : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Proving the coordinates of B' after applying the same translation vector v to B
theorem find_B' : translate B v = (-5, -3) :=
by
  -- translation function needs to be instantiated with the correct values.
  -- Since this is just a Lean 4 statement, we'll not include the proof here and leave it as a sorry.
  sorry

end find_B_l312_312981


namespace ines_amount_left_l312_312987

noncomputable def total_cost_before_discounts : ℝ :=
  (3 * 2) + (2 * 3.5) + (4 * 1.25) + (1 * 4) + (2 * 2.5)

noncomputable def discount : ℝ := 
  if total_cost_before_discounts > 10 then 0.15 * total_cost_before_discounts else 0

noncomputable def cost_after_discount : ℝ := 
  total_cost_before_discounts - discount

noncomputable def tax : ℝ := 0.05 * cost_after_discount

noncomputable def cost_after_tax : ℝ := 
  cost_after_discount + tax

noncomputable def surcharge : ℝ := 
  0.02 * cost_after_tax

noncomputable def final_cost : ℝ := 
  cost_after_tax + surcharge

noncomputable def rounded_final_cost : ℝ := 
  Float.ceil (final_cost * 100) / 100

noncomputable def amount_left : ℝ := 
  20 - rounded_final_cost

theorem ines_amount_left : 
  amount_left = -4.58 :=
by
  sorry

end ines_amount_left_l312_312987


namespace product_of_possible_values_b_l312_312681

theorem product_of_possible_values_b : 
  ∀ (b : ℝ), (y1 = 3) → (y2 = 7) → (x1 = 2) → form_square y1 y2 x1 b → y2 - y1 = 4 
  ∧ x1 - 4 ≤ x1 + 4 → b = -2 ∨ b = 6 → (-2) * 6 = -12 :=
by 
  intros b y1 y2 x1 form_square hy hz hx1 hsq hdist hvalues,
  have hy : y2 - y1 = 4, from by linarith [hz, hy],
  cases hvalues with hval_left hval_right,
  { rw hval_left,
    ring },
  { rw hval_right,
    ring },
  exact hdist

end product_of_possible_values_b_l312_312681


namespace snow_at_least_once_probability_l312_312165

def P_snow_first_two_days : ℚ := 1 / 2
def P_no_snow_first_two_days : ℚ := 1 - P_snow_first_two_days
def P_snow_next_four_days_if_snow_first_two_days : ℚ := 1 / 3
def P_no_snow_next_four_days_if_snow_first_two_days : ℚ := 1 - P_snow_next_four_days_if_snow_first_two_days
def P_snow_next_four_days_if_no_snow_first_two_days : ℚ := 1 / 4
def P_no_snow_next_four_days_if_no_snow_first_two_days : ℚ := 1 - P_snow_next_four_days_if_no_snow_first_two_days

def P_no_snow_first_two_days_total : ℚ := P_no_snow_first_two_days ^ 2
def P_no_snow_next_four_days_given_no_snow_first_two_days : ℚ := P_no_snow_next_four_days_if_no_snow_first_two_days ^ 4
def P_no_snow_next_four_days_given_snow_first_two_days : ℚ := P_no_snow_next_four_days_if_snow_first_two_days ^ 4

def P_no_snow_all_days : ℚ := 
  P_no_snow_first_two_days_total * P_no_snow_next_four_days_given_no_snow_first_two_days +
  (1 - P_no_snow_first_two_days_total) * P_no_snow_next_four_days_given_snow_first_two_days

def P_snow_at_least_once : ℚ := 1 - P_no_snow_all_days

theorem snow_at_least_once_probability : P_snow_at_least_once = 29 / 32 :=
by
  -- sorry to indicate that the proof is skipped
  sorry

end snow_at_least_once_probability_l312_312165


namespace num_divisors_36_l312_312496

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312496


namespace smoking_health_correlation_is_negative_l312_312745

-- Define the relationship and the negative correlation.
def smoking_is_harmful_to_health := ∀ (S H : Type), (smoking S) -> (health H) -> ((correlation S H) = negative_correlation)

-- The theorem stating the main problem to be proved.
theorem smoking_health_correlation_is_negative (S H : Type) (hs : smoking_is_harmful_to_health S H) :
  (correlation S H) = negative_correlation :=
by
  sorry

end smoking_health_correlation_is_negative_l312_312745


namespace relationship_between_a_b_c_l312_312027

variable (a b c : ℝ)
variable (h_a : a = 0.4 ^ 0.2)
variable (h_b : b = 0.4 ^ 0.6)
variable (h_c : c = 2.1 ^ 0.2)

-- Prove the relationship c > a > b
theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l312_312027


namespace minimum_abs_phi_l312_312447

theorem minimum_abs_phi (ϕ : ℝ) (k : ℤ) :
  (y : ℝ → ℝ) = 3 * cos (2 * x + ϕ) 
  ∧ (∀ x : ℝ, y (2 * (2 * π / 3) + ϕ) = y (k * π + π / 2))
  → |k * π - (5 * π / 6)| = π / 6 := sorry

end minimum_abs_phi_l312_312447


namespace quarter_pounder_cost_l312_312342

theorem quarter_pounder_cost :
  let fries_cost := 2 * 1.90
  let milkshakes_cost := 2 * 2.40
  let min_purchase := 18
  let current_total := fries_cost + milkshakes_cost
  let amount_needed := min_purchase - current_total
  let additional_spending := 3
  let total_cost := amount_needed + additional_spending
  total_cost = 12.40 :=
by
  sorry

end quarter_pounder_cost_l312_312342


namespace Smithtown_left_handed_women_percentage_l312_312093

theorem Smithtown_left_handed_women_percentage :
  ∃ (x y : ℕ), 
    (3 * x + x = 4 * x) ∧
    (3 * y + 2 * y = 5 * y) ∧
    (4 * x = 5 * y) ∧
    (x = y) → 
    let total_population := 4 * x
    let left_handed_women := x
    left_handed_women / total_population = 0.25 :=
sorry

end Smithtown_left_handed_women_percentage_l312_312093


namespace find_a_b_l312_312591

theorem find_a_b (a b : ℕ) (h1 : (a^3 - a^2 + 1) * (b^3 - b^2 + 2) = 2020) : 10 * a + b = 53 :=
by {
  -- Proof to be completed
  sorry
}

end find_a_b_l312_312591


namespace distance_BC_l312_312555

def A : ℝ × ℝ × ℝ := (1, -2, 3)
def B : ℝ × ℝ × ℝ := (1, 2, 3)
def C : ℝ × ℝ × ℝ := (1, 2, -3)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_BC : distance B C = 6 :=
  by
  -- proof to be provided
  sorry

end distance_BC_l312_312555


namespace trigonometric_expression_value_l312_312025

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
sorrY

end trigonometric_expression_value_l312_312025


namespace no_prime_divisible_by_56_l312_312526

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end no_prime_divisible_by_56_l312_312526


namespace interval_contains_zero_l312_312680

def f (x : ℝ) : ℝ := 2^x - 5

theorem interval_contains_zero : ∃ m : ℕ, f m ≤ 0 ∧ f (m + 1) > 0 ∧ m = 2 := by
  sorry

end interval_contains_zero_l312_312680


namespace largest_n_for_factoring_l312_312870

theorem largest_n_for_factoring :
  ∃ n, (∃ A B : ℤ, (3 * A = 3 * 108 + 1) ∧ (/3 * B * 108 = 2) ∧ 
  (3 * 36 + 3 = 111) ∧ (3 * 108 + A = n) )=
  (n = 325) := sorry
iddenLean_formatter.clonecreateAngular

end largest_n_for_factoring_l312_312870


namespace employees_use_public_transport_l312_312757

-- Define the main assumptions based on the given conditions
def total_employees := 100
def drives_to_work := 0.60
def fraction_take_public_transport := 0.5

-- Define the problem statement
theorem employees_use_public_transport : 
  let drives := drives_to_work * total_employees in
  let doesnt_drive := total_employees - drives in
  let takes_public_transport := doesnt_drive * fraction_take_public_transport in
  takes_public_transport = 20 :=
by
  sorry

end employees_use_public_transport_l312_312757


namespace sum_of_squares_of_combinations_l312_312747

-- We are stating the assumptions as definitions.
def binomial (n k : ℕ) := nat.choose n k

axiom binomial_identity (n : ℕ) : binomial (n + 1) 3 - binomial n 3 = binomial n 2

theorem sum_of_squares_of_combinations : 
  ∑ n in finset.range 9, binomial (n + 2) 2 ^ 2 = 165 := 
sorry

end sum_of_squares_of_combinations_l312_312747


namespace initial_number_of_men_l312_312668

theorem initial_number_of_men
  (M : ℕ) (A : ℕ)
  (h1 : ∀ A_new : ℕ, A_new = A + 4)
  (h2 : ∀ total_age_increase : ℕ, total_age_increase = (2 * 52) - (36 + 32))
  (h3 : ∀ sum_age_men : ℕ, sum_age_men = M * A)
  (h4 : ∀ new_sum_age_men : ℕ, new_sum_age_men = sum_age_men + ((2 * 52) - (36 + 32))) :
  M = 9 := 
by
  -- Proof skipped
  sorry

end initial_number_of_men_l312_312668


namespace pastries_combination_l312_312306

def total_pastries : ℕ := 8
def pastry_types : ℕ := 4

theorem pastries_combination (n k : ℕ) (h_n : n = total_pastries - pastry_types) (h_k : k = pastry_types) :
  ∃ c : ℕ, c = Nat.choose (n + k - 1) (k - 1) ∧ c = 35 :=
by
  use Nat.choose (n + k - 1) (k - 1)
  split
  · rfl
  · sorry

end pastries_combination_l312_312306


namespace maximal_n_for_quadratic_factorization_l312_312856

theorem maximal_n_for_quadratic_factorization :
  ∃ n, n = 325 ∧ (∃ A B : ℤ, A * B = 108 ∧ n = 3 * B + A) :=
by
  use 325
  use 1, 108
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num
  sorry

end maximal_n_for_quadratic_factorization_l312_312856


namespace alice_has_winning_strategy_l312_312148

def game_conditions (n : ℕ) (hn : n > 0) : Prop :=
  ∃ (winning_player : string), winning_player = "Alice"

theorem alice_has_winning_strategy (n : ℕ) (hn : n > 0) : (∃ (winning_player : string), winning_player = "Alice") :=
by
  sorry

end alice_has_winning_strategy_l312_312148


namespace max_different_numbers_in_table_l312_312098

/-- In each cell of a 100 × 100 table, a natural number is written. Each row 
contains at least 10 different numbers, and in every four consecutive rows, 
there are no more than 15 different numbers. The maximum number of different 
numbers in the table is 175.  -/
theorem max_different_numbers_in_table : 
  (∀ (i : ℕ), i < 100 → ∃ (s : Finset ℕ), s.card ≥ 10 ∧ ∀ j < 100, table i j ∈ s)
  → (∀ (i : ℕ), (∀ j, i ≤ j → j < i + 4 → ∃ (s : Finset ℕ), s.card ≤ 15 ∧ ∀ k, i ≤ k → k < i + 4 → table k j ∈ s))
  → ∃ (s : Finset ℕ), s.card = 175 :=
sorry

end max_different_numbers_in_table_l312_312098


namespace problem_statement_l312_312079

variables {k : ℝ} {f : ℝ → ℝ}

-- Define the conditions
def cond1 : Prop := k > 0
def cond2 (x : ℝ) (hx : x > 0) : Prop := (f (x^2 + 1)) ^ (Real.sqrt x) = k

-- State the main theorem
theorem problem_statement (y : ℝ) (hy : y > 0) (hk : cond1) (H : ∀ x > 0, cond2 x): 
    (f ((9 + y^2) / y^2)) ^ (Real.sqrt (12 / y)) = k^2 :=
sorry

end problem_statement_l312_312079


namespace problem_equiv_proof_l312_312241

theorem problem_equiv_proof : 
  (∑ i in {2, 4, 6}, i ^ 2 / ∑ i in {1, 3, 5}, i ^ 2) -
  (∑ i in {1, 3, 5}, i ^ 2 / ∑ i in {2, 4, 6}, i ^ 2) = 1911 / 1960 := 
by sorry

end problem_equiv_proof_l312_312241


namespace find_a_and_c_range_for_m_l312_312915

def nat_star : Set ℕ := {n : ℕ | n > 0}

variable {a c : ℕ} (m : ℝ) (f : ℚ → ℝ)

-- Define the given function f(x) = ax^2 + 2x + c
def f (x : ℝ) : ℝ := a * x ^ 2 + 2 * x + c

-- Given conditions
def condition1 := f 1 = 5
def condition2 := 6 < f 2 ∧ f 2 < 11
axiom nat_star_condition : a ∈ nat_star
axiom nat_star_condition_c : c ∈ nat_star

-- Goal statement combining all the conditions
theorem find_a_and_c (h₁ : condition1) (h₂ : condition2) (hna : nat_star_condition) (hnc : nat_star_condition_c) : a = 1 ∧ c = 2 := sorry

-- Inequality condition for m
def holds (m : ℝ) : Prop := ∀ x : ℝ, f x - 2 * m * x ≤ 1
theorem range_for_m (h₁ : condition1) (h₂ : condition2) (hna : nat_star_condition) (hnc : nat_star_condition_c) (h : holds m) : m ≥ 1 := sorry

end find_a_and_c_range_for_m_l312_312915


namespace intersection_points_form_similar_triangle_l312_312221

theorem intersection_points_form_similar_triangle (ABC A_1B_1C_1 : Triangle) 
    (alpha : ℝ) (h : alpha < 180) (O : Point)
    (rotates_around_circumcircle : rotates_around ABC A_1B_1C_1 O alpha) :
    similar (intersection_triangle ABC A_1B_1C_1) ABC :=
sorry

end intersection_points_form_similar_triangle_l312_312221


namespace part_1_part_2_l312_312601

open Nat

-- Definitions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (a d : ℝ) := ∀ n, a_n n = a + (n-1) * d
def sum_first_n_terms (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) := ∀ n, S_n n = (n * (a_n n + a_n 1) / 2)
def b_n (n : ℕ) (S_n : ℕ → ℝ) (c : ℝ) := (n * S_n n) / ((n:ℝ)^2 + c)

-- First part of the problem
theorem part_1 (a d : ℝ) (S_n : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n, S_n n = (n:ℝ)^2 * a) →
  (∀ k n, S_n (n*k) = (n:ℝ) * (n:ℝ) * S_n k) :=
sorry

-- Second part of the problem
theorem part_2 (a d : ℝ) (S_n : ℕ → ℝ) (c : ℝ) :
  (∀ n m, b_n (n+m) S_n c = b_n n S_n c + b_n m S_n c) →
  c = 0 :=
sorry

end part_1_part_2_l312_312601


namespace minimum_positive_period_pi_collinear_O_P_C_implies_OA_plus_OB_l312_312900

-- Define vectors and point P in terms of alpha
noncomputable def vec_OA (α : ℝ) : ℝ × ℝ := (real.sin α, 1)
noncomputable def vec_OB (α : ℝ) : ℝ × ℝ := (real.cos α, 0)
noncomputable def vec_OC (α : ℝ) : ℝ × ℝ := (-real.sin α, 2)

-- Define vector functions
noncomputable def vec_AB (α : ℝ) : ℝ × ℝ := (real.cos α - real.sin α, -1)
noncomputable def vec_BP (α : ℝ) : ℝ × ℝ := (2 * real.cos α - real.sin α - real.cos α, -1)
noncomputable def vec_CA (α : ℝ) : ℝ × ℝ := (2 * real.sin α, -1)
noncomputable def vec_PB (α : ℝ) : ℝ × ℝ := (real.sin α - real.cos α, 1)

-- Define function f(α)
noncomputable def f (α : ℝ) : ℝ := (real.sin α - real.cos α) * (2 * real.sin α) - 1

-- Theorems to prove
theorem minimum_positive_period_pi (α : ℝ) : f(α) = -real.sqrt 2 * real.sin (2 * α + real.pi / 4) → ∃ T : ℝ, T = real.pi :=
sorry

theorem collinear_O_P_C_implies_OA_plus_OB (α : ℝ) : 
  (α = real.atan (4 / 3)) → |vec_OA α + vec_OB α| = real.sqrt 74 / 5 :=
sorry

end minimum_positive_period_pi_collinear_O_P_C_implies_OA_plus_OB_l312_312900


namespace parallelogram_area_l312_312106

open Real

def line1 (p : ℝ × ℝ) : Prop := p.2 = 2
def line2 (p : ℝ × ℝ) : Prop := p.2 = -2
def line3 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 - 10 = 0
def line4 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 + 20 = 0

theorem parallelogram_area :
  ∃ D : ℝ, D = 30 ∧
  (∀ p : ℝ × ℝ, line1 p ∨ line2 p ∨ line3 p ∨ line4 p) :=
sorry

end parallelogram_area_l312_312106


namespace gcf_lcm_sum_l312_312595

-- Definitions for GCD and LCM
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Let's define a function to get GCD and LCM for three numbers by pairwise computation
def gcd_three (a b c : ℕ) : ℕ := gcd a (gcd b c)
def lcm_three (a b c : ℕ) : ℕ := lcm a (lcm b c)

theorem gcf_lcm_sum :
  let C := gcd_three 18 30 45,
  let D := lcm_three 18 30 45
  in C + D = 93 := 
by
  -- Define the GCD and LCM of the three numbers
  let C := gcd_three 18 30 45
  let D := lcm_three 18 30 45
  -- We use sorry to skip the proof
  sorry

end gcf_lcm_sum_l312_312595


namespace mitchell_more_than_antonio_l312_312157

-- Definitions based on conditions
def mitchell_pencils : ℕ := 30
def total_pencils : ℕ := 54

-- Definition of the main question
def antonio_pencils : ℕ := total_pencils - mitchell_pencils

-- The theorem to be proved
theorem mitchell_more_than_antonio : mitchell_pencils - antonio_pencils = 6 :=
by
-- Proof is omitted
sorry

end mitchell_more_than_antonio_l312_312157


namespace find_cd_l312_312722

def g (c d x : ℝ) : ℝ := c * x^3 - 4 * x^2 + d * x - 7

theorem find_cd :
  let c := -1 / 3
  let d := 28 / 3
  g c d 2 = -7 ∧ g c d (-1) = -20 :=
by sorry

end find_cd_l312_312722


namespace solution_l312_312197

-- Definition of the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * m * x + 1

-- Statement of the problem
theorem solution (m : ℝ) (x : ℝ) (h : quadratic_equation m x = (m - 2) * x^2 + 3 * m * x + 1) : m ≠ 2 :=
by
  sorry

end solution_l312_312197


namespace boat_license_combinations_l312_312785

theorem boat_license_combinations :
  let letters := {A, M, Z}
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  in
  (∃ f : vector letters 1 × vector digits 6,
   true) →
  3 * 10^6 = 3000000 :=
by
  sorry

end boat_license_combinations_l312_312785


namespace find_value_of_n_l312_312061

-- Given conditions
def hyperbola_has_focus (m : ℝ) : Prop :=
  ∃ x y : ℝ,  x = 0 ∧ y = 2 ∧ (x^2/m - y^2/(3*m) = 1)

def ellipse_has_focal_length (n m : ℝ) : Prop :=
  ∃ f : ℝ, f = 4 ∧ y^2/n - x^2/m = 1

-- Theorem statement to prove that n = 5 under given conditions
theorem find_value_of_n (m n : ℝ) (h1 : hyperbola_has_focus m) (h2 : ellipse_has_focal_length n m) : n = 5 := 
sorry

end find_value_of_n_l312_312061


namespace moles_of_HNO3_l312_312014

theorem moles_of_HNO3 (HNO3 NaHCO3 NaNO3 : ℝ)
  (h1 : NaHCO3 = 1) (h2 : NaNO3 = 1) :
  HNO3 = 1 :=
by sorry

end moles_of_HNO3_l312_312014


namespace artifacts_in_each_wing_l312_312776

theorem artifacts_in_each_wing (total_wings : ℕ) (artifact_factor : ℕ) (painting_wings : ℕ) 
  (large_painting : ℕ) (small_paintings_per_wing : ℕ) (remaining_artifact_wings : ℕ) 
  (total_paintings constant_total_paintings_expected : ℕ) (total_artifacts total_artifacts_expected : ℕ) 
  (artifacts_per_wing artifacts_per_wing_expected : ℕ) : 

  total_wings = 8 →
  artifact_factor = 4 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  remaining_artifact_wings = total_wings - painting_wings →
  total_paintings = painting_wings * small_paintings_per_wing + large_painting →
  total_artifacts = total_paintings * artifact_factor →
  artifacts_per_wing = total_artifacts / remaining_artifact_wings →
  artifacts_per_wing = 20 :=

by
    intros htotal_wings hartifact_factor hpainting_wings hlarge_painting hsmall_paintings_per_wing hermaining_artifact_wings htotal_paintings htotal_artifacts hartifacts_per_wing,
    sorry

end artifacts_in_each_wing_l312_312776


namespace small_s_team_count_l312_312564

theorem small_s_team_count (G : ℕ) (hG : G = 36) : ∃ n : ℕ, (n * (n - 1)) / 2 = 36 ∧ n = 9 :=
by
  use 9
  split
  simp [*, Nat.mul_sub] at *
  sorry

end small_s_team_count_l312_312564


namespace smallest_perimeter_even_integer_triangl_l312_312256

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l312_312256


namespace domain_of_function_l312_312676

def f (x : ℝ) : ℝ := (x - 1) / real.sqrt (3 * x + 2) + x^0

theorem domain_of_function :
  ∀ x : ℝ,
    (3 * x + 2 > 0) ∧ (x ≠ 0) ↔ (-2 / 3 < x ∧ x < 0 ∨ 0 < x) :=
  sorry

end domain_of_function_l312_312676


namespace integer_a_for_factoring_l312_312012

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end integer_a_for_factoring_l312_312012


namespace q_transformation_factor_l312_312543

-- Define the initial condition
def q (w : ℝ) (h : ℝ) (z : ℝ) : ℝ :=
  5 * w / (4 * h * z^2)

-- Define the new values after modification
def q_new (w : ℝ) (h : ℝ) (z : ℝ) : ℝ :=
  20 * w / (8 * h * (3 * z)^2)

-- Prove that q_new is (5 / 18) times the original q
theorem q_transformation_factor (w h z : ℝ) (hq : h ≠ 0) (hz : z ≠ 0) :
  q_new w h z = (5 / 18) * q w h z :=
by
  unfold q
  unfold q_new
  sorry

end q_transformation_factor_l312_312543


namespace binomial_sum_excluding_constant_l312_312376

theorem binomial_sum_excluding_constant :
  let f := (λ x : ℚ, (2/√x - x)^9) in
  (∑ k in (finset.range 10).erase 3, (1 : ℚ) * binom 9 k * (2/√1)^(9-k) * (-1)^k) = 5377 :=
by 
  let f := (λ x : ℚ, (2/√x - x)^9)
  have h_sum : f 1 = (2/√1 - 1)^9 := rfl,
  rw [h_sum, pow_one],
sorry

end binomial_sum_excluding_constant_l312_312376


namespace number_of_roots_eq_four_l312_312913

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x
  else if x <= 1 then (x-1)^2 + 2*(x-1) + 1
  else (x-1)^2 + 1

theorem number_of_roots_eq_four :
  (∃ x ∈ set.Icc (-2 : ℝ) 2, x - f x = 0) → 4 :=
  sorry

end number_of_roots_eq_four_l312_312913


namespace divisors_of_36_l312_312473

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312473


namespace problem1_problem2_problem3_l312_312292

-- 1. Given: ∃ x ∈ ℤ, x^2 - 2x - 3 = 0
--    Show: ∀ x ∈ ℤ, x^2 - 2x - 3 ≠ 0
theorem problem1 : (∃ x : ℤ, x^2 - 2 * x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2 * x - 3 ≠ 0) := sorry

-- 2. Given: ∀ x ∈ ℝ, x^2 + 3 ≥ 2x
--    Show: ∃ x ∈ ℝ, x^2 + 3 < 2x
theorem problem2 : (∀ x : ℝ, x^2 + 3 ≥ 2 * x) ↔ (∃ x : ℝ, x^2 + 3 < 2 * x) := sorry

-- 3. Given: If x > 1 and y > 1, then x + y > 2
--    Show: If x ≤ 1 or y ≤ 1, then x + y ≤ 2
theorem problem3 : (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ↔ (∀ x y : ℝ, x ≤ 1 ∨ y ≤ 1 → x + y ≤ 2) := sorry

end problem1_problem2_problem3_l312_312292


namespace dot_product_is_one_half_l312_312934

variables {α : ℝ}

/-- Given vectors a and b as defined, their dot product is 1/2. -/
theorem dot_product_is_one_half (α : ℝ) : 
  let a := (Real.cos α, Real.sin α),
      b := (Real.cos (Real.pi / 3 + α), Real.sin (Real.pi / 3 + α)) in
  (a.1 * b.1 + a.2 * b.2) = 1 / 2 :=
by
  let a := (Real.cos α, Real.sin α)
  let b := (Real.cos (Real.pi / 3 + α), Real.sin (Real.pi / 3 + α))
  sorry

end dot_product_is_one_half_l312_312934


namespace expected_rolls_to_at_least_3_l312_312164

/-- Expected number of rolls to achieve a sum of at least 3 on a fair six-sided die --/
theorem expected_rolls_to_at_least_3 
  (X : ℕ → Probability measure_on ℝ) 
  (hX : ∀ n, X n = sum (range 1 7) ((λ i, ite i = (1 : ℝ), 1) + (λ i, ite i = (2 : ℝ), 2))) : 
  (∑ i in range 1 7, X i) / 6 = 1.36 := 
sorry

end expected_rolls_to_at_least_3_l312_312164


namespace problem_statement_l312_312592

variable {a : ℕ → ℂ} -- assuming coefficients a_i are complex numbers for generality
noncomputable def polynomial_expansion : ℂ → ℕ → ℂ 
| x, n := if n = 0 then (x + 1) * (2 * x + 1)^(10) else 0 -- formal representation of the polynomial, focusing on nth term expansion

theorem problem_statement (a : ℕ → ℂ) :
  (∀ x : ℂ, (x + 1) * (2 * x + 1)^(10) = ∑ i in finset.range 12, a i * (x + 2)^i) → 
  ((∑ i in finset.range 11, a (i + 1)) = -310) :=
by
  sorry

end problem_statement_l312_312592


namespace max_additional_spheres_l312_312767

noncomputable def frustum : Type := sorry

structure Sphere (r : ℝ) :=
(center : frustum)
(radius : ℝ := r)

axiom sphere_O1 : Sphere 2
axiom sphere_O2 : Sphere 3

axiom frustum_conditions {con : Prop} :
  con ↔
  -- height of the frustum is 8 units
  ∃ h : ℝ, h = 8 ∧
  -- Sphere O1 is tangent to the upper base and side surface
  ∀ (O1 : Sphere 2),
    is_on_axis O1 ∧
    is_tangent_to_upper_base O1 ∧
    is_tangent_to_side_surface O1 :=
  sorry

axiom sphere_O2_conditions {con : Prop} :
  con ↔
  ∃ (O2 : Sphere 3),
  -- Sphere O2 is tangent to Sphere O1, lower base of frustum, and side surface
  is_tangent_to_sphere O2 sphere_O1 ∧
  is_tangent_to_lower_base O2 ∧
  is_tangent_to_side_surface O2 := 
  sorry

theorem max_additional_spheres :
  frustum_conditions →
  sphere_O2_conditions →
  (max_additional_spheres 3 = 2) :=
sorry

end max_additional_spheres_l312_312767


namespace angle_W_in_quadrilateral_l312_312190

theorem angle_W_in_quadrilateral 
  (W X Y Z : ℝ) 
  (h₀ : W + X + Y + Z = 360) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) : 
  W = 206 :=
by
  sorry

end angle_W_in_quadrilateral_l312_312190


namespace brother_spent_on_highlighters_l312_312938

theorem brother_spent_on_highlighters : 
  let total_money := 100
  let cost_sharpener := 5
  let num_sharpeners := 2
  let cost_notebook := 5
  let num_notebooks := 4
  let cost_eraser := 4
  let num_erasers := 10
  let total_spent_sharpeners := num_sharpeners * cost_sharpener
  let total_spent_notebooks := num_notebooks * cost_notebook
  let total_spent_erasers := num_erasers * cost_eraser
  let total_spent := total_spent_sharpeners + total_spent_notebooks + total_spent_erasers
  let remaining_money := total_money - total_spent
  remaining_money = 30 :=
begin
  sorry
end

end brother_spent_on_highlighters_l312_312938


namespace solution_set_of_inequality_l312_312849

open Set

theorem solution_set_of_inequality :
  {x : ℝ | (x ≠ -2) ∧ (x ≠ -8) ∧ (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5)} =
  {x : ℝ | (-8 < x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 4)} :=
by
  sorry

end solution_set_of_inequality_l312_312849


namespace find_root_of_equation_l312_312285

theorem find_root_of_equation (a b c d x : ℕ) (h_ad : a + d = 2016) (h_bc : b + c = 2016) (h_ac : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 1008 :=
by
  sorry

end find_root_of_equation_l312_312285


namespace officers_count_l312_312977

theorem officers_count (average_salary_all : ℝ) (average_salary_officers : ℝ) 
    (average_salary_non_officers : ℝ) (num_non_officers : ℝ) (total_salary : ℝ) : 
    average_salary_all = 120 → 
    average_salary_officers = 470 →  
    average_salary_non_officers = 110 → 
    num_non_officers = 525 → 
    total_salary = average_salary_all * (num_non_officers + O) → 
    total_salary = average_salary_officers * O + average_salary_non_officers * num_non_officers → 
    O = 15 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end officers_count_l312_312977


namespace general_term_formula_sum_formula_max_sum_value_l312_312827

-- Define the initial condition for the arithmetic sequence
def a₁ : ℤ := 28
def d : ℤ := -2

-- Define the nth term of the arithmetic sequence
def a (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

-- Prove the general formula for the nth term of the sequence
theorem general_term_formula (n : ℕ) : a n = 30 - 2 * n :=
by
  sorry

-- Prove the formula for the sum of the first n terms
theorem sum_formula (n : ℕ) : S n = -n^2 + 29 * n :=
by
  sorry

-- Prove the maximum value of S(n) and corresponding n
theorem max_sum_value : 
  let max_value := 210 in
  (S 14 = max_value ∧ S 15 = max_value) ∧ 
  (max_value = -14^2 + 29 * 14) ∧ 
  (max_value = -15^2 + 29 * 15) :=
by
  sorry

end general_term_formula_sum_formula_max_sum_value_l312_312827


namespace acute_angle_condition_l312_312399

variables {a b : ℝ}
def magnitude_a : ℝ := real.sqrt 2
def magnitude_b : ℝ := 1
def angle_ab := real.pi / 4 -- 45 degrees in radians
def dot_product_ab := magnitude_a * magnitude_b * real.cos angle_ab

noncomputable def vector_angle_acute (λ : ℝ) : Prop :=
 (2 * a - λ * b) • (λ * a - 3 * b) > 0

theorem acute_angle_condition (λ : ℝ) :
  magnitude_a = real.sqrt 2 → magnitude_b = 1 → angle_ab = real.pi / 4 →
  (1 < λ ∧ λ < 6 ∧ λ ≠ real.sqrt 6) → vector_angle_acute λ :=
by
  sorry

end acute_angle_condition_l312_312399


namespace ellipse_characterize_dot_product_range_l312_312895

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1

noncomputable def eccentricity (a c : ℝ) : Prop :=
  c/a = Real.sqrt 2 / 2

noncomputable def condition1 (a b : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P ≠ (-a,0) ∧ P ≠ (a,0) ∧
  (λ PA PB : ℝ × ℝ, (PA.1 - PB.1)^2 + (PA.2 - PB.2)^2 - 2) PA PB = -2

theorem ellipse_characterize (a b c d : ℝ) :
  eccentricity a c ∧ condition1 a b →
  ellipse_equation 4 2 :=
sorry

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_range (a b c d : ℝ) (F1 F2 M N : ℝ × ℝ) :
  eccentricity a c ∧ condition1 a b ∧ 
  (F1 = (-Real.sqrt 2, 0)) ∧ (F2 = (Real.sqrt 2, 0)) →
  (-2 <= dot_product (F2.1 - M.1, F2.2 - M.2) (F2.1 - N.1, F2.2 - N.2) ∧
  dot_product (F2.1 - M.1, F2.2 - M.2) (F2.1 - N.1, F2.2 - N.2) <= 7) :=
sorry

end ellipse_characterize_dot_product_range_l312_312895


namespace equilateral_ABC_l312_312140

variable {A B C I X Y Z : Type}
variable [Triangle A B C I] -- assuming there is a Triangle structure which includes incenter property
variable [IsIncenter I A B C]
variable [IsIncenter X B I C]
variable [IsIncenter Y C I A]
variable [IsIncenter Z A I B]

theorem equilateral_ABC (h : IsEquilateralTriangle X Y Z) : IsEquilateralTriangle A B C := sorry

end equilateral_ABC_l312_312140


namespace second_rate_of_return_l312_312000

namespace Investment

def total_investment : ℝ := 33000
def interest_total : ℝ := 970
def investment_4_percent : ℝ := 13000
def interest_rate_4_percent : ℝ := 0.04

def amount_second_investment : ℝ := total_investment - investment_4_percent
def interest_from_first_part : ℝ := interest_rate_4_percent * investment_4_percent
def interest_from_second_part (R : ℝ) : ℝ := R * amount_second_investment

theorem second_rate_of_return : (∃ R : ℝ, interest_from_first_part + interest_from_second_part R = interest_total) → 
  R = 0.0225 :=
by
  intro h
  sorry

end Investment

end second_rate_of_return_l312_312000


namespace paths_from_top_to_bottom_l312_312825

def regular_octahedron (V F : Type) := sorry

theorem paths_from_top_to_bottom
  (V : Type) [fintype V] [decidable_eq V]
  (F : Type) [fintype F] [decidable_eq F]
  (octahedron : regular_octahedron V F)
  (top bottom : V)
  (adj : V → finset F)
  (adj_faces : F → finset F)
  (hf1 : ∀ v : V, card (adj v) = 4)
  (hf2 : ∀ f : F, card (adj_faces f) = 3)
  (hf3 : top ≠ bottom)
  (hf4 : top ∈ ⋃ f ∈ adj bottom, adj_faces f)
  (hf5 : ∀ f ∈ adj top, bottom ∈ adj_faces f) :
  (2 * 4 = 8) := by
  sorry

end paths_from_top_to_bottom_l312_312825


namespace sum_formable_l312_312571
noncomputable def alpha : ℝ := (Real.sqrt 29 - 1) / 2

def valid_denominations (n : ℕ) : Prop := 
  ∀ k : ℕ, k > 0 → irrational (alpha^k)

theorem sum_formable 
  (n : ℕ) (h1 : alpha > 2) 
  (h2 : valid_denominations n) : 
  ∃ (m : ℕ), ∀ (sum : ℕ), sum = n → 
    ∃ (c1 c2 ... cm : ℕ), 
      sum = c1 + c2 * alpha + ... + cm * alpha^(m-1) ∧ 
      ∀ i, c_i ≤ 6 :=
  sorry

end sum_formable_l312_312571


namespace find_B_l312_312339

noncomputable def poly_roots : {roots : list ℕ // (roots.product = 81) ∧ (roots.sum = 10)} := sorry

theorem find_B (roots : list ℕ) (h1 : roots.product = 81) (h2 : roots.sum = 10) :
  let B := - (4 + 36 + 36) in
  B = -76 :=
by
  have h3 : (4 + 36 + 36) = 76 := by simp
  simp [B, h3]
  rfl

end find_B_l312_312339


namespace min_value_expression_min_value_min_value_at_y_main_theorem_l312_312907

theorem min_value_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) : 
  2*x + 3*y^2 = 2 * (1 - 2*y) + 3*y^2 :=
by
  sorry

theorem min_value (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) : 
  2*x + 3*y^2 ≥ (3y^2 - 4y + 2) :=
by
  sorry

theorem min_value_at_y (hx : 1 - 2*(1/2) ≥ 0) : 
  2*(1 - 2*(1/2)) + 3*(1/2)^2 = 3/4 :=
by
  sorry

theorem main_theorem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) : 
  2*x + 3*y^2 = 3/4 :=
by
  sorry

end min_value_expression_min_value_min_value_at_y_main_theorem_l312_312907


namespace increasing_function_range_l312_312089

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x^2 + a * x
  else (4 - a / 2) * x + 2

theorem increasing_function_range (a : ℝ) :
  (∀ x, (0 < x → 2 * x + a > 0) ∧ (x ≤ 1 → 4 - a / 2 > 0)) ∧ 
  ((1 + a = (4 - a / 2) * 1 + 2) → a = 10 / 3) →
  10 / 3 ≤ a ∧ a < 8 :=
sorry

end increasing_function_range_l312_312089


namespace ceil_neg_3_7_l312_312361

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l312_312361


namespace trapezoid_base_ratio_l312_312645

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l312_312645


namespace induced_subgraph_exists_l312_312560

-- Given a graph G with e edges and n vertices, and the degrees of the vertices d_1, d_2, ..., d_n
variables {V : Type*} [fintype V] {G : simple_graph V}

-- Given an integer k such that k < min {d_1, d_2, ..., d_n}
variable {k : ℕ}

-- Given the degree of each vertex
variables (d : V → ℕ)

-- Assumption k < min {d_1, d_2, ..., d_n}
variable (h_min : ∀ (v : V), k < d v)

-- Assuming degrees sum up to 2e (Handshaking Lemma)
variable (degree_sum : ∑ (v : V), d v = 2 * G.edge_finset.card)

-- Assumption on the graph G being finite and having n vertices
variable [fintype V]

-- Define induced subgraph that avoids K_{k+1}
def avoids_complete_subgraph (H : simple_graph V) :=
  ∀ (W : finset V), W.card = k + 1 → ¬H.is_clique W

-- Prove that required induced subgraph H exists
theorem induced_subgraph_exists : 
  ∃ (H : simple_graph V), 
    H ≤ G ∧ 
    (avoids_complete_subgraph k H) ∧ 
    finset.card H.vertices ≥ k * (fintype.card V)^2 / (2 * G.edge_finset.card + fintype.card V) :=
sorry

end induced_subgraph_exists_l312_312560


namespace second_largest_n_divides_170_factorial_l312_312539

-- Define the main problem
theorem second_largest_n_divides_170_factorial :
  let n := (170 / 5) + (34 / 5) + (6 / 5) - 1 in 
  n = 40 := 
by
  -- computation will verify this calculation
  have h : (170 / 5) + (34 / 5) + (6 / 5) = 41 :=
    by norm_num [int.div],
  have n_def : n = 41 - 1 :=
    by rw [show n = (170 / 5) + (34 / 5) + (6 / 5) - 1, from rfl, h],
  exact (n_def.trans (sub_self 1)).symm,
  sorry -- Proof omitted

end second_largest_n_divides_170_factorial_l312_312539


namespace largest_inscribed_square_side_length_l312_312988

noncomputable def side_length_of_largest_inscribed_square (a : ℝ) : ℝ :=
  10 - (5 * Real.sqrt 6 - 5 * Real.sqrt 2) / 2

theorem largest_inscribed_square_side_length :
  let a := 20 in
  side_length_of_largest_inscribed_square a = 10 - (5 * Real.sqrt 6 - 5 * Real.sqrt 2) / 2 :=
by trivial -- To be replaced with the actual proof

end largest_inscribed_square_side_length_l312_312988


namespace domain_g_l312_312956

def is_in_domain (f : ℝ → ℝ) (a b x : ℝ) : Prop :=
  a ≤ x ∧ x ≤ b

def domain_f := [-3, 5]

def f : ℝ → ℝ := sorry  -- Definition of f is not provided, hence it's a placeholder

theorem domain_g (g : ℝ → ℝ) (f : ℝ → ℝ) :
  (∀ x, is_in_domain f (-3) 5 x) →
  (∀ x, is_in_domain f (-1) 4 x →
        g(x) = f(x + 1) + f(x - 2)) :=
by
  -- Proof needs to be filled in
  sorry

end domain_g_l312_312956


namespace subset_implies_uniform_membership_l312_312896

theorem subset_implies_uniform_membership {A B : Set α} 
  (hA_nonempty : ∃ x, x ∈ A) 
  (hB_nonempty : ∃ y, y ∈ B) 
  (hA_subset_B : ∀ x, x ∈ A → x ∈ B) : 
  ∀ x, x ∈ A → x ∈ B := by
    assume x hx,
    exact hA_subset_B x hx

end subset_implies_uniform_membership_l312_312896


namespace binary_arithmetic_l312_312332

theorem binary_arithmetic :
  let a := 0b1101
  let b := 0b0110
  let c := 0b1011
  let d := 0b1001
  a + b - c + d = 0b10001 := by
sorry

end binary_arithmetic_l312_312332


namespace find_number_l312_312277

theorem find_number (x : ℝ) (h : (1/4) * x = (1/5) * (x + 1) + 1) : x = 24 := 
sorry

end find_number_l312_312277


namespace smallest_triangle_perimeter_l312_312261

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l312_312261


namespace number_of_divisors_of_36_l312_312502

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312502


namespace trade_ratio_blue_per_red_l312_312168

-- Define the problem conditions
def initial_total_marbles : ℕ := 10
def blue_percentage : ℕ := 40
def kept_red_marbles : ℕ := 1
def final_total_marbles : ℕ := 15

-- Find the number of blue marbles initially
def initial_blue_marbles : ℕ := (blue_percentage * initial_total_marbles) / 100

-- Calculate the number of red marbles initially
def initial_red_marbles : ℕ := initial_total_marbles - initial_blue_marbles

-- Calculate the number of red marbles traded
def traded_red_marbles : ℕ := initial_red_marbles - kept_red_marbles

-- Calculate the number of marbles received from the trade
def traded_marbles : ℕ := final_total_marbles - (initial_blue_marbles + kept_red_marbles)

-- The number of blue marbles received per each red marble traded
def blue_per_red : ℕ := traded_marbles / traded_red_marbles

-- Theorem stating that Pete's friend trades 2 blue marbles for each red marble
theorem trade_ratio_blue_per_red : blue_per_red = 2 := by
  -- Proof steps would go here
  sorry

end trade_ratio_blue_per_red_l312_312168


namespace find_area_AFPE_l312_312142

-- Definitions based on the conditions in the problem
variables {S_BPF S_BPC S_CPE S_PEF S_AEF S_AFPE : ℝ}

-- Given areas
def S_BPF := 4
def S_BPC := 8
def S_CPE := 13

-- Relationship in quadrilateral BCEF where P is intersection of BE and CF
axiom area_relationship : S_BPF * S_CPE = S_BPC * S_PEF

-- Solution derived:
def S_PEF := 52 / 8

-- Prove that the area of AFPE is 143
theorem find_area_AFPE (h1 : S_BPF = 4) (h2 : S_BPC = 8) (h3 : S_CPE = 13) : S_AFPE = 143 :=
by
  have h_area_relationship : S_PEF = 13 / 2 :=
    calc
    S_PEF = 4 * 13 / 8 : by rw [mul_div_right_comm, h1, h3, h2]
    ... = 52 / 8      : by norm_num
    ... = 13 / 2      : by norm_num
  sorry

end find_area_AFPE_l312_312142


namespace volume_increase_by_eight_l312_312544

theorem volume_increase_by_eight {r : ℝ} (V : ℝ) (h : V = (4 / 3) * Real.pi * r^3) :
  let r' := 2 * r in
  let V' := (4 / 3) * Real.pi * r'^3 in
  V' = 8 * V :=
by
  sorry

end volume_increase_by_eight_l312_312544


namespace divisors_of_36_count_l312_312517

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312517


namespace nested_sum_evaluation_l312_312335

noncomputable def inner_sum (n : ℕ) (k : ℕ) : ℤ :=
if n ≥ 3 ∧ k ≥ 2 ∧ k ≤ n - 1 then k / 3^(n + k) else 0

noncomputable def nested_sum : ℚ :=
∑' n, ∑' k, inner_sum n k

theorem nested_sum_evaluation : nested_sum = 9 / 64 := by
  sorry

end nested_sum_evaluation_l312_312335


namespace karen_total_cost_l312_312588

noncomputable def calculate_total_cost (burger_price sandwich_price smoothie_price : ℝ) (num_smoothies : ℕ)
  (discount_rate tax_rate : ℝ) (order_time : ℕ) : ℝ :=
  let total_cost_before_discount := burger_price + sandwich_price + (num_smoothies * smoothie_price)
  let discount := if total_cost_before_discount > 15 ∧ order_time ≥ 1400 ∧ order_time ≤ 1600 then total_cost_before_discount * discount_rate else 0
  let reduced_price := total_cost_before_discount - discount
  let tax := reduced_price * tax_rate
  reduced_price + tax

theorem karen_total_cost :
  calculate_total_cost 5.75 4.50 4.25 2 0.20 0.12 1545 = 16.80 :=
by
  sorry

end karen_total_cost_l312_312588


namespace students_speaking_both_languages_l312_312734

theorem students_speaking_both_languages:
  ∀ (total E T N B : ℕ),
    total = 150 →
    E = 55 →
    T = 85 →
    N = 30 →
    (total - N) = 120 →
    (E + T - B) = 120 → B = 20 :=
by
  intros total E T N B h_total h_E h_T h_N h_langs h_equiv
  sorry

end students_speaking_both_languages_l312_312734


namespace exists_always_white_cell_l312_312129

-- Assume there is an infinite grid plane where initially some finite number of cells are painted black.
-- We have a grid polygon M that can cover more than one cell.
-- M can be shifted in any direction on the grid without rotating.
-- If after a shift, exactly one cell of M lies on a white cell, then that white cell is painted black.
-- Prove that there exists at least one cell that will always remain white no matter how many shifts of M are performed.

theorem exists_always_white_cell
  (grid : ℤ × ℤ → Bool)  -- Infinite grid plane, true for black cells
  (initial_black_cells : Finset (ℤ × ℤ))  -- Initially painted black cells
  (M : Finset (ℤ × ℤ))  -- Grid polygon M covering more than one cell
  (hM : 1 < M.card)  -- M covers more than one cell
  (shift : ℤ × ℤ → Finset (ℤ × ℤ) → Finset (ℤ × ℤ))  -- Shifting function
  (h_shift : ∀ s : ℤ × ℤ, shift s M ⊆ grid)  -- Shifting preserves grid alignment
  :
  ∃ (white_cell : ℤ × ℤ), ∀ (shift_seq : List (ℤ × ℤ)), ¬(shift_seq.foldl (λ b s, shift s b) initial_black_cells) white_cell = true := 
sorry

end exists_always_white_cell_l312_312129


namespace remainder_98_pow_24_mod_100_l312_312717

theorem remainder_98_pow_24_mod_100 : (98 ^ 24) % 100 = 16 := 
by {
  have h1 : 98 % 100 = (-2) % 100 := by norm_num,
  have h2 : (-2) ^ 24 = 2 ^ 24 := by norm_num,
  have h3 : (2 ^ 24) % 100 = 16 := by norm_num,
  rw [← h1, h2],
  exact h3,
}

end remainder_98_pow_24_mod_100_l312_312717


namespace smallest_positive_period_monotonically_decreasing_intervals_find_a_l312_312445

noncomputable def f (x a : ℝ) : ℝ := 
  sin (2 * x + π / 6) + sin (2 * x - π / 6) + cos (2 * x) + a

theorem smallest_positive_period (a : ℝ) : 
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ T = π :=
sorry

theorem monotonically_decreasing_intervals (a : ℝ) : 
  ∀ k : ℤ, ∃ I : set ℝ, I = set.Icc (k * π + π / 6) (k * π + 2 * π / 3) ∧ 
  ∀ x ∈ I, ∃ y ∈ I, x ≠ y → f x a > f y a :=
sorry

theorem find_a (a : ℝ) : 
  (∀ x ∈ set.Icc 0 (π / 2), ∀ y ∈ set.Icc 0 (π / 2), f x a > f y a ∨ f x a = -2) ∧ 
  f (π / 2) a = -2 → 
  a = -1 :=
sorry

end smallest_positive_period_monotonically_decreasing_intervals_find_a_l312_312445


namespace brother_highlighters_spent_l312_312943

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end brother_highlighters_spent_l312_312943


namespace molecular_weight_H2O_correct_l312_312244

-- Define the atomic weights of hydrogen and oxygen, and the molecular weight of H2O
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight calculation of H2O
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + atomic_weight_O

-- Theorem to state the molecular weight of H2O is approximately 18.016 g/mol
theorem molecular_weight_H2O_correct : molecular_weight_H2O = 18.016 :=
by
  -- Putting the value and calculation
  sorry

end molecular_weight_H2O_correct_l312_312244


namespace exists_chess_order_2x2_square_l312_312845

theorem exists_chess_order_2x2_square:
  (∀ i j, (i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99) → color i j = Black) →
  (∀ i j, i < 99 ∧ j < 99 → 
    (∃ p q, color i j = p ∧ color (i+1) j = q ∧ p ≠ q ∨ 
    ∃ p q, color i j = p ∧ color i (j+1) = q ∧ p ≠ q ∨ 
    ∃ p q, color (i+1) j = p ∧ color (i+1) (j+1) = q ∧ p ≠ q ∨ 
    ∃ p q, color i (j+1) = p ∧ color (i+1) (j+1) = q ∧ p ≠ q)) →
  ∃ i j, (i < 99 ∧ j < 99) ∧ 
    ((color i j = Black ∧ color (i+1) (j+1) = Black ∧ 
     color (i+1) j = White ∧ color i (j+1) = White) ∨ 
    ( color i j = White ∧ color (i+1) (j+1) = White ∧ 
     color (i+1) j = Black ∧ color i (j+1) = Black)) :=
sorry

end exists_chess_order_2x2_square_l312_312845


namespace true_propositions_l312_312414

axiom SkewLines (a b : Type) : Prop

axiom is_parallel (a b : Type) : Prop
axiom is_perpendicular (a b : Type) : Prop
axiom is_parallel_to_plane (a : Type) (α : Type) : Prop
axiom is_perpendicular_to_plane (a : Type) (α : Type) : Prop

noncomputable def Prop1 : Prop := ∀ (a b : Type), SkewLines a b → ¬ is_parallel a b ∧ ¬ (a = b)
noncomputable def Prop2 : Prop := ∀ (a b : Type) (α : Type), SkewLines a b → is_parallel_to_plane a α → ¬ is_parallel_to_plane b α
noncomputable def Prop3 : Prop := ∀ (a b : Type) (α : Type), SkewLines a b → is_perpendicular_to_plane a α → ¬ is_perpendicular_to_plane b α
noncomputable def Prop4 : Prop := ∀ (a b : Type) (π : Type), SkewLines a b → ¬ (is_parallel (project_onto_plane a π) (project_onto_plane b π))

theorem true_propositions : Prop1 ∧ Prop3 :=
by
  have p1 : Prop1 := sorry,
  have p3 : Prop3 := sorry,
  exact ⟨p1, p3⟩

end true_propositions_l312_312414


namespace hypotenuse_length_l312_312034

-- Define the properties of the right-angled triangle
variables (α β γ : ℝ) (a b c : ℝ)
-- Right-angled triangle condition
axiom right_angled_triangle : α = 30 ∧ β = 60 ∧ γ = 90 → c = 2 * a

-- Given side opposite 30° angle is 6 cm
axiom side_opposite_30_is_6cm : a = 6

-- Proof that hypotenuse is 12 cm
theorem hypotenuse_length : c = 12 :=
by 
  sorry

end hypotenuse_length_l312_312034


namespace ellipse_standard_equation_l312_312045

-- Define the foci and point P
def foci := [(-1, 0), (1, 0)]
def P := (2, 0)

-- Define the problem statement
theorem ellipse_standard_equation (foci : List (ℝ × ℝ)) (P : ℝ × ℝ) :
  foci = [(-1,0), (1,0)] ∧ P = (2,0) →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≥ b ∧ 
  (a = 2) ∧ (b = Real.sqrt 3) ∧ 
  (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1, p.2) = (x, y) → 
    x^2 / 4 + y^2 / 3 = 1)) :=
by
  sorry

end ellipse_standard_equation_l312_312045


namespace amount_pop_spend_l312_312180

theorem amount_pop_spend
  (total_spent : ℝ)
  (ratio_snap_crackle : ℝ)
  (ratio_crackle_pop : ℝ)
  (spending_eq : total_spent = 150)
  (snap_crackle : ratio_snap_crackle = 2)
  (crackle_pop : ratio_crackle_pop = 3)
  (snap : ℝ)
  (crackle : ℝ)
  (pop : ℝ)
  (snap_eq : snap = ratio_snap_crackle * crackle)
  (crackle_eq : crackle = ratio_crackle_pop * pop)
  (total_eq : snap + crackle + pop = total_spent) :
  pop = 15 := 
by
  sorry

end amount_pop_spend_l312_312180


namespace proportion_solution_l312_312088

-- Define the given proportion condition as a hypothesis
variable (x : ℝ)

-- The definition is derived directly from the given problem
def proportion_condition : Prop := x / 5 = 1.2 / 8

-- State the theorem using the given proportion condition to prove x = 0.75
theorem proportion_solution (h : proportion_condition x) : x = 0.75 :=
  by
    sorry

end proportion_solution_l312_312088


namespace painted_prism_probability_l312_312839

-- Define a rectangular prism with independently painted faces
def rectangular_prism : Type := array 6 bool  -- We use bool to represent colors: false = red, true = blue

-- Each face is painted independently with a probability of 1/2
def paint_probability (prism : rectangular_prism) : ℝ :=
  (1 / 2) ^ 6

-- Condition that the prism can be placed so that the two visible vertical faces and their opposing faces are the same color
def suitable_arrangements (prism : rectangular_prism) : Prop :=
  -- Check if there exists a valid placement
  ((prism[0] = prism[1] ∧ prism[2] = prism[3] ∧ prism[0] = prism[2]) ∨
   (prism[0] = prism[2] ∧ prism[1] = prism[3] ∧ prism[0] = prism[1]) ∨
   (prism[0] = prism[3] ∧ prism[1] = prism[2] ∧ prism[0] = prism[1]))

theorem painted_prism_probability :
  probability (λ prism, suitable_arrangements prism) = 5 / 16 :=
begin
  sorry
end

end painted_prism_probability_l312_312839


namespace maximal_n_for_quadratic_factorization_l312_312859

theorem maximal_n_for_quadratic_factorization :
  ∃ n, n = 325 ∧ (∃ A B : ℤ, A * B = 108 ∧ n = 3 * B + A) :=
by
  use 325
  use 1, 108
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num
  sorry

end maximal_n_for_quadratic_factorization_l312_312859


namespace num_divisors_of_36_l312_312522

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312522


namespace divisors_of_36_l312_312470

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312470


namespace given_condition_l312_312078

theorem given_condition (f : Real → Real) :
  (∀ x : Real, f (Real.sin x) = Real.cos (2 * x)) →
  f (Real.sin (5 * Real.pi / 6)) ≠ Real.sqrt 3 / 2 ∧
  f (Real.cos (2 * Real.pi / 3)) = 1 / 2 ∧
  f (Real.cos x) ≠ -Real.sin (2 * x) ∧
  f (Real.cos x) = -Real.cos (2 * x) :=
by
  intro h
  split
  sorry -- Proof for f (Real.sin (5 * Real.pi / 6)) ≠ Real.sqrt 3 / 2
  split
  sorry -- Proof for f (Real.cos (2 * Real.pi / 3)) = 1 / 2
  split
  sorry -- Proof for f (Real.cos x) ≠ -Real.sin (2 * x)
  sorry -- Proof for f (Real.cos x) = -Real.cos (2 * x)

end given_condition_l312_312078


namespace cosine_angle_BHD_eq_zero_l312_312565

variables (D H G F B : Type)
variables [inner_product_space ℝ D] [inner_product_space ℝ H] [inner_product_space ℝ G]
variables [inner_product_space ℝ F] [inner_product_space ℝ B]
variables (DH HG FB BHD : ℝ)

-- Conditions
axiom angle_DHG : angle DH G = 30
axiom angle_FHB : angle F H B = 45
axiom side_length_cube : ∀ (x : D H G F B), ∥x∥ = 2

-- To Prove
theorem cosine_angle_BHD_eq_zero : cos (angle BHD) = 0 :=
sorry

end cosine_angle_BHD_eq_zero_l312_312565


namespace two_trains_clear_time_l312_312738

def train1_length : ℝ := 150
def train2_length : ℝ := 165
def train1_speed_kmh : ℝ := 80
def train2_speed_kmh : ℝ := 65
def kmh_to_ms (s : ℝ) : ℝ := s * (5.0 / 18.0)

theorem two_trains_clear_time :
  let total_length := train1_length + train2_length,
      relative_speed := kmh_to_ms (train1_speed_kmh + train2_speed_kmh),
      time := total_length / relative_speed
  in time ≈ 7.82 := 
by
  sorry

end two_trains_clear_time_l312_312738


namespace sum_integer_solutions_l312_312638

-- Define the inequality function
noncomputable def inequality (x : ℝ) : ℝ :=
  9 * ((|x + 4| - |x - 2|) / (|3 * x + 14| - |3 * x - 8|)) +
  11 * ((|x + 4| + |x - 2|) / (|3 * x + 14| + |3 * x - 8|))

-- Define the range condition
def inRange (x : ℝ) : Prop := abs x < 110

-- Define the main theorem we need to prove
theorem sum_integer_solutions : 
  (∑ n in (Finset.filter inRange (Finset.Ico (Int.floor (-110 : ℝ)) (Int.ceil 110))),
  if inequality n ≤ 6 then (n : ℝ) else 0) = -6 :=
by
  sorry

end sum_integer_solutions_l312_312638


namespace smallest_number_is_3_l312_312702

theorem smallest_number_is_3 (a b c : ℝ) (h1 : (a + b + c) / 3 = 7) (h2 : a = 9 ∨ b = 9 ∨ c = 9) : min (min a b) c = 3 := 
sorry

end smallest_number_is_3_l312_312702


namespace probability_of_drawing_red_ball_l312_312699

theorem probability_of_drawing_red_ball : 
  let box1_initial_white := 2
  let box1_initial_red := 4
  let box2_initial_white := 5
  let box2_initial_red := 3

  let p_white_from_box1 := (box1_initial_white : ℚ) / (box1_initial_white + box1_initial_red)
  let p_red_from_box1 := (box1_initial_red : ℚ) / (box1_initial_white + box1_initial_red)

  let box2_after_white_in_white := box2_initial_white + 1
  let box2_after_white_in_red := box2_initial_red
  let box2_after_red_in_white := box2_initial_white
  let box2_after_red_in_red := box2_initial_red + 1

  let p_red_from_box2_after_white := (box2_after_white_in_red : ℚ) / (box2_after_white_in_white + box2_after_white_in_red)
  let p_red_from_box2_after_red := (box2_after_red_in_red : ℚ) / (box2_after_red_in_white + box2_after_red_in_red)

  let probability := p_white_from_box1 * p_red_from_box2_after_white + p_red_from_box1 * p_red_from_box2_after_red

  probability = 11 / 27 := by
  -- Proof: The detailed steps are omitted
  sorry

end probability_of_drawing_red_ball_l312_312699


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l312_312930

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {4, 5, 6, 7, 8, 9}
def B : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem union_of_A_and_B : A ∪ B = U := by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {4, 5, 6} := by
  sorry

theorem complement_of_intersection : U \ (A ∩ B) = {1, 2, 3, 7, 8, 9} := by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l312_312930


namespace determine_radius_of_semicircle_l312_312213

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem determine_radius_of_semicircle :
  radius_of_semicircle 32.392033717615696 = 6.3 :=
by
  sorry

end determine_radius_of_semicircle_l312_312213


namespace find_m_of_binomial_coeff_l312_312904

-- Theorem statement: given the conditions, prove that m = 2
theorem find_m_of_binomial_coeff (m : ℤ) 
  (h : (choose 4 2 * m^2) = (choose 4 3 * m + 16)) : 
  m = 2 :=
sorry

end find_m_of_binomial_coeff_l312_312904


namespace possible_omega_values_l312_312920

open Real

theorem possible_omega_values (ω : ℝ) :
  ω > 0 → 
  (∀ x ∈ Ioo 0 π, sin (ω * x + π / 3) = 0) → 
  (5 / 3 < ω ∧ ω ≤ 8 / 3) :=
by
  intros ω_pos h
  sorry

end possible_omega_values_l312_312920


namespace sin_A_correct_l312_312101

theorem sin_A_correct (A B C : Type) [triangle : Triangle A B C]
  (hC : ∠ C = 90)
  (BC : length B C = 3)
  (AB : length A B = 4) :
  sin (angle A) = 3 / 4 := 
sorry

end sin_A_correct_l312_312101


namespace problem_a51_l312_312114

-- Definitions of given conditions
variable {a : ℕ → ℤ}
variable (h1 : ∀ n : ℕ, a (n + 2) - 2 * a (n + 1) + a n = 16)
variable (h2 : a 63 = 10)
variable (h3 : a 89 = 10)

-- Proof problem statement
theorem problem_a51 :
  a 51 = 3658 :=
by
  sorry

end problem_a51_l312_312114


namespace relationship_among_a_b_c_l312_312602

-- Definitions as per given conditions
def a : ℝ := Real.tan (135 * Real.pi / 180)
def b : ℝ := Real.cos (Real.cos 0)
def c : ℝ := (x^2 + 1/2)^0

-- Proof statement
theorem relationship_among_a_b_c : c > b ∧ b > a := 
by
  sorry

end relationship_among_a_b_c_l312_312602


namespace sequence_bounds_l312_312685

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 2 else (sequence (n-1))^5 + 1 / (5 * (sequence (n-1)))

theorem sequence_bounds : ∀ n : ℕ, 1 ≤ n → (1 / 5 : ℝ) ≤ sequence n ∧ sequence n ≤ 2 :=
by
  assume n hn
  sorry

end sequence_bounds_l312_312685


namespace nice_people_total_l312_312801

theorem nice_people_total :
  let num_barry := 50 in
  let num_kevin := 40 in
  let num_julie := 200 in
  let num_joe := 120 in
  let num_alex := 150 in
  let num_lauren := 90 in
  let num_chris := 80 in
  let num_taylor := 100 in
  let per_barry := 1.0 in
  let per_kevin := 0.5 in
  let per_julie := 0.75 in
  let per_joe := 0.1 in
  let per_alex := 0.85 in
  let per_lauren := 2 / 3 in
  let per_chris := 0.25 in
  let per_taylor := 0.95 in
  let nice_barry := num_barry * per_barry in
  let nice_kevin := num_kevin * per_kevin in
  let nice_julie := num_julie * per_julie in
  let nice_joe := num_joe * per_joe in
  let nice_alex := real.floor (num_alex * per_alex) in
  let nice_lauren := num_lauren * per_lauren in
  let nice_chris := num_chris * per_chris in
  let nice_taylor := num_taylor * per_taylor in
  nice_barry + nice_kevin + nice_julie + nice_joe + nice_alex + nice_lauren + nice_chris + nice_taylor = 534 := 
by
  sorry

end nice_people_total_l312_312801


namespace find_f_of_odd_function_periodic_l312_312424

noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem find_f_of_odd_function_periodic (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_periodic : ∀ x k : ℤ, f x = f (x + 3 * k))
    (α : ℝ) (h_tan : Real.tan α = 3) :
  f (2015 * Real.sin (2 * (arctan 3))) = 0 :=
sorry

end find_f_of_odd_function_periodic_l312_312424


namespace ceil_neg_3_7_l312_312358

def x : ℝ := -3.7

def ceil_function (x : ℝ) : ℤ := Int.ceil x

theorem ceil_neg_3_7 : ceil_function x = -3 := 
by 
  -- Translate the conditions into Lean 4 conditions, 
  -- equivalently define x and the function ceil_function
  sorry

end ceil_neg_3_7_l312_312358


namespace divisors_of_36_count_l312_312514

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312514


namespace trapezoid_base_ratio_l312_312650

-- Define variables and conditions as per the problem
variables {a b h: ℝ}
def is_trapezoid (a b: ℝ) := a > b
def area_trapezoid (a b h: ℝ) := (1 / 2) * (a + b) * h
def area_quadrilateral (a b h: ℝ) := (1 / 2) * ((a - b) / 2) * (h / 2)

-- State the problem statement
theorem trapezoid_base_ratio (a b h: ℝ) (ha: a > b) (ht: area_quadrilateral a b h = (1 / 4) * area_trapezoid a b h) :
  a / b = 3 :=
sorry

end trapezoid_base_ratio_l312_312650


namespace divisors_of_36_l312_312477

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312477


namespace series_problem_l312_312806

theorem series_problem (m : ℝ) :
  let a₁ := 9
  let a₂ := 3
  let b₁ := 9
  let b₂ := 3 + m
  let S₁ := a₁ / (1 - (a₂ / a₁))
  let S₂ := b₁ / (1 - (b₂ / b₁))
  S₂ = 3 * S₁ → m = 4 :=
by
  let a₁ := 9
  let a₂ := 3
  let b₁ := 9
  let b₂ := 3 + m
  let S₁ := a₁ / (1 - (a₂ / a₁))
  let S₂ := b₁ / (1 - (b₂ / b₁))
  have h_sum_equal : S₂ = 3 * S₁ := by assumption
  sorry

end series_problem_l312_312806


namespace simplify_radical_l312_312536

theorem simplify_radical :
  let s := 1 / (2 - real.cbrt 3)
  s = 2 + real.cbrt 3 :=
by
  sorry

end simplify_radical_l312_312536


namespace transformation_represents_A_l312_312059

noncomputable def f : ℝ → ℝ :=
λ x, if x >= -3 ∧ x <= 0 then -2 - x
     else if x >= 0 ∧ x <= 2 then real.sqrt(4 - (x - 2) ^ 2) - 2
     else if x >= 2 ∧ x <= 3 then 2 * (x - 2)
     else 0

def transformed_f (x : ℝ) : ℝ := f (2 * x + 1)

theorem transformation_represents_A :
  (graph transformed_f) = (graph A) := by
  sorry

end transformation_represents_A_l312_312059


namespace infinite_solutions_exists_l312_312179

theorem infinite_solutions_exists : 
  ∃ (S : Set (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ S → 2 * a^2 - 3 * a + 1 = 3 * b^2 + b) 
  ∧ Set.Infinite S :=
sorry

end infinite_solutions_exists_l312_312179


namespace comparison_of_y1_and_y2_l312_312103

variable {k y1 y2 : ℝ}

theorem comparison_of_y1_and_y2 (hk : 0 < k)
    (hy1 : y1 = k)
    (hy2 : y2 = k / 4) :
    y1 > y2 := by
  sorry

end comparison_of_y1_and_y2_l312_312103


namespace exists_polygon_le_9_exists_polygon_le_8_l312_312840

/-- Definition of a polygon -/
structure Polygon (n : ℕ) :=
(vertices : Fin n → Point)
(sides : (Fin n) → Line)
(property : ∀ i : Fin n, ∃ j : Fin n, j ≠ i ∧ vertices j ∈ (sides i).points)

/-- The theorem to prove the existence of a polygon with at most 9 vertices adhering to the given property -/
theorem exists_polygon_le_9 :
  ∃ (n : ℕ) (p : Polygon n), n ≤ 9 ∧ (∀ i : Fin n, ∃ j : Fin n, j ≠ i ∧ p.vertices j ∈ (p.sides i).points) :=
sorry

/-- The theorem to prove the existence of a polygon with at most 8 vertices adhering to the given property -/
theorem exists_polygon_le_8 :
  ∃ (n : ℕ) (p : Polygon n), n ≤ 8 ∧ (∀ i : Fin n, ∃ j : Fin n, j ≠ i ∧ p.vertices j ∈ (p.sides i).points) :=
sorry

end exists_polygon_le_9_exists_polygon_le_8_l312_312840


namespace total_distance_traveled_l312_312169

theorem total_distance_traveled (A B D : ℝ) (hA : A = 4000) (hB : B = 4500) :
  let BD := Real.sqrt (B^2 - A^2)
  in A + B + BD = 8500 + 50 * Real.sqrt (1700) := 
by
  sorry

end total_distance_traveled_l312_312169


namespace negation_of_dot_product_property_l312_312211

variables {p q : ℝ^n}

theorem negation_of_dot_product_property (n : ℕ) :
  ∀ (p q : ℝ^n), |p ⬝ q| ≠ |p| * |q| :=
by
  sorry

end negation_of_dot_product_property_l312_312211


namespace ratio_AD_AB_l312_312986

-- Triangle-specific definitions and angles
def Triangle := Type

-- Define the main theorem
theorem ratio_AD_AB {ABC : Triangle} 
  (angle_A : ABC → ℝ) 
  (angle_B : ABC → ℝ) 
  (DE : Set ℝ)  -- This represents the line DE and its properties
  (D : ABC → ℝ)
  (E : ABC → ℝ)
  (angle_A_60 : angle_A ABC = 60)
  (angle_B_45 : angle_B ABC = 45)
  (on_line : D ABC ∈ DE ∧ E ABC ∈ AC)
  (angle_ADE_75 : ∃ x, x = 75 ∧ ∀ y ∈ (DE : Set ℝ), y = x)  -- Angle ADE = 75°
  (equal_area_divide : dividesAreaEqually ABC DE) :  -- DE divides ABC into equal areas
  ∃ r, r = 1 / 2 ∧ r = (D ABC / LineSegment AB) :=
sorry

end ratio_AD_AB_l312_312986


namespace novel_pages_count_l312_312581

def joel_reading_schedule (P : ℕ) : Prop :=
  let pages_first_4_days := 4 * 42 in
  let pages_next_2_days := 2 * 48 in
  let pages_first_6_days := pages_first_4_days + pages_next_2_days in
  let pages_last_day := 14 in
  P = pages_first_6_days + pages_last_day

theorem novel_pages_count : ∃ P : ℕ, joel_reading_schedule P ∧ P = 278 :=
by
  let pages_first_4_days := 4 * 42
  let pages_next_2_days := 2 * 48
  let pages_first_6_days := pages_first_4_days + pages_next_2_days
  let pages_last_day := 14
  existsi (pages_first_6_days + pages_last_day)
  split
  . unfold joel_reading_schedule
    unfold pages_first_4_days pages_next_2_days pages_first_6_days pages_last_day
    rfl
  . rfl

end novel_pages_count_l312_312581


namespace max_negative_factors_l312_312958

theorem max_negative_factors (a b c d e f g h : ℤ) :
  a * b * c * d * e * f * g * h < 0 → 
  (∑ n in {a, b, c, d, e, f, g, h}, if n < 0 then 1 else 0) ≤ 7 :=
sorry

end max_negative_factors_l312_312958


namespace max_value_of_f_ratio_BC_AB_in_triangle_ABC_l312_312051

section Problem

def f (x : ℝ) : ℝ := sqrt 3 * (sin x)^2 + sin x * cos x - sqrt 3 / 2

theorem max_value_of_f:
  ∃ x ∈ Ioo 0 (π / 2), f x = 1 := sorry

theorem ratio_BC_AB_in_triangle_ABC
  (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = π)
  (hA_lt_B : A < B) (h_fA : f A = 1 / 2) (h_fB : f B = 1 / 2)
  : BC / AB = sqrt 2 := sorry 

end Problem

end max_value_of_f_ratio_BC_AB_in_triangle_ABC_l312_312051


namespace complement_A_eq_B_subset_complement_A_l312_312593

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 + 4 * x > 0 }
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1 }

-- The universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Complement of A in U
def complement_U_A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}

-- Proof statement for part (1)
theorem complement_A_eq : complement_U_A = {x | -4 ≤ x ∧ x ≤ 0} :=
  sorry 

-- Proof statement for part (2)
theorem B_subset_complement_A (a : ℝ) : B a ⊆ complement_U_A ↔ -3 ≤ a ∧ a ≤ -1 :=
  sorry 

end complement_A_eq_B_subset_complement_A_l312_312593


namespace area_of_quadrilateral_l312_312177

-- Define the quadrilateral ABCD with the given conditions
structure Quadrilateral :=
  (A B C D : Type)
  (AB BC AD DC : ℚ)
  (right_angle_B : Prop)
  (right_angle_D : Prop)
  (AC : ℚ)
  (AC_len : AC = 5)
  (int_sides : AB = 3 ∨ AB = 4 ∨ BC = 3 ∨ BC = 4 ∨ AD = 3 ∨ AD = 4 ∨ DC = 3 ∨ DC = 4)
  (one_odd : ∃ x, (x = 3 ∨ x = 4) ∧ odd x)

-- Define the theorem to prove that the area of quadrilateral ABCD is 12
theorem area_of_quadrilateral : ∀ (q : Quadrilateral), q.AC = 5 → (q.AB = 3 ∨ q.AB = 4 ∨ q.BC = 3 ∨ q.BC = 4 ∨ q.AD = 3 ∨ q.AD = 4 ∨ q.DC = 3 ∨ q.DC = 4) → (∃ x, (x = 3 ∨ x = 4) ∧ odd x) → (area_of_quadrilateral q = 12) :=
by
  sorry

end area_of_quadrilateral_l312_312177


namespace max_profit_at_six_l312_312690

noncomputable def y1 (x : ℝ) : ℝ := 17 * x^2
noncomputable def y2 (x : ℝ) : ℝ := 2 * x^3 - x^2
noncomputable def profit (x : ℝ) : ℝ := y1 x - y2 x

theorem max_profit_at_six :
  ∀ x : ℝ, x = 6 -> ∀ x' : ℝ, x' > 0 → profit x ≤ profit x' :=
begin
    -- we need to show the proof
    sorry
end

end max_profit_at_six_l312_312690


namespace volume_of_rectangular_prism_l312_312234

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end volume_of_rectangular_prism_l312_312234


namespace brenda_age_l312_312317

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l312_312317


namespace angle_between_hour_and_minute_hand_7_36_l312_312243

theorem angle_between_hour_and_minute_hand_7_36 : 
  ∀ (hour minute : ℕ), 
    hour = 7 → minute = 36 → 
    let minute_angle := (minute / 60.0) * 360.0 in
    let hour_angle := (hour + minute / 60.0) * 30.0 in
    abs (hour_angle - minute_angle) = 12 :=
by
  intros hour minute h_hour h_minute
  let minute_angle := (minute / 60.0) * 360.0
  let hour_angle := (hour + minute / 60.0) * 30.0
  have h1 : minute_angle = 216.0, by sorry
  have h2 : hour_angle = 228.0, by sorry
  calc abs (hour_angle - minute_angle) 
        = abs (228.0 - 216.0) : by rw [h1, h2]
    ... = abs 12.0 : by norm_num
    ... = 12 : by norm_num


end angle_between_hour_and_minute_hand_7_36_l312_312243


namespace symmetric_point_l312_312115

-- Define the three-dimensional point symmetry with respect to the x-axis
def symmetric_with_respect_to_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, -p.3)

-- Define the specific point in question
def point_A : ℝ × ℝ × ℝ := (-2, 1, 4)

-- Define what the symmetric point should be
def point_B : ℝ × ℝ × ℝ := (-2, -1, -4)

-- Theorem to prove that the symmetric of point_A with respect to the x-axis is point_B
theorem symmetric_point : symmetric_with_respect_to_x_axis point_A = point_B :=
  by
  sorry

end symmetric_point_l312_312115


namespace fruit_vendor_l312_312883

theorem fruit_vendor (x y a b : ℕ) (C1 : 60 * x + 40 * y = 3100) (C2 : x + y = 60) 
                     (C3 : 15 * a + 20 * b = 600) (C4 : 3 * a + 4 * b = 120)
                     (C5 : 3 * a + 4 * b + 3 * (x - a) + 4 * (y - b) = 250) :
  (x = 35 ∧ y = 25) ∧ (820 - 12 * a - 16 * b = 340) ∧ (a + b = 52 ∨ a + b = 53) :=
by
  sorry

end fruit_vendor_l312_312883


namespace relationship_between_products_l312_312952

variable {a₁ a₂ b₁ b₂ : ℝ}

theorem relationship_between_products (h₁ : a₁ < a₂) (h₂ : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := 
sorry

end relationship_between_products_l312_312952


namespace area_ABC_l312_312111

namespace TriangleProof

variable {A B C P : Type} [EuclideanSpace ℝ P]

-- Define the given conditions
variable (m : ℝ) (h_m : m > 0)
variable (PA PC AB : P → ℝ^3)
variable (h_condition : PA + PC = λ p, m * AB p)
variable (area_ABP : ℝ := 6)

-- Define a function to express the area of a triangle given its vertices.
noncomputable def area_triangle (a b c : P → ℝ^3) : ℝ :=
  -- some formula to compute the area of ΔABC
  sorry

-- The main statement: prove the area of ΔABC using the conditions given.
theorem area_ABC : area_triangle A B C = 12 :=
by
  sorry

end TriangleProof

end area_ABC_l312_312111


namespace bogdan_enescu_inequality_l312_312994

theorem bogdan_enescu_inequality 
  (x : ℕ → ℝ) 
  (n : ℕ) 
  (h_pos : ∀ i, 1 ≤ i → i ≤ n → 0 < x i) 
  (n_pos : 0 < n) :
  (∑ i in finset.range n, 1 / (1 + (finset.range (i+1)).sum x)) < 
  real.sqrt (∑ i in finset.range n, 1 / (x i.succ)) :=
sorry

end bogdan_enescu_inequality_l312_312994


namespace pizza_toppings_l312_312792

theorem pizza_toppings (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  rw [h, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_succ_succ]
  -- Sorry to skip the detailed calculations
  sorry

end pizza_toppings_l312_312792


namespace range_of_c_l312_312947

variable {a c : ℝ}

theorem range_of_c (h : a ≥ 1 / 8) (sufficient_but_not_necessary : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 := 
sorry

end range_of_c_l312_312947


namespace number_of_solutions_eq_l312_312530

theorem number_of_solutions_eq : 
  (∃ x : ℕ, (⟦(x / 20 : ℝ)⟧ = ⟦(x / 17 : ℝ)⟧) ∧ (x > 0)) → 
  ∃ count_x : ℕ, count_x = 56 :=
by
  sorry

end number_of_solutions_eq_l312_312530


namespace number_of_divisors_of_36_l312_312509

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312509


namespace probability_of_event_a_l312_312084

variables {Ω : Type*} [probability_space Ω]
variables (a b : event Ω)

noncomputable def P : event Ω → ℝ := sorry

axiom prob_b : P b = 1 / 2
axiom prob_a_and_b : P (a ∩ b) = 3 / 8
axiom prob_not_a_and_not_b : P (¬a ∩ ¬b) = 0.125

theorem probability_of_event_a : P a = 0.75 :=
by {
  sorry
}

end probability_of_event_a_l312_312084


namespace alternating_series_sum_l312_312331

theorem alternating_series_sum :
  ∑ k in Finset.range 99, if k % 2 = 0 then (k + 1: ℤ) else -(k + 1: ℤ) = 50 :=
by
  sorry

end alternating_series_sum_l312_312331


namespace arithmetic_sequence_a2_a8_l312_312105

theorem arithmetic_sequence_a2_a8 (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : a 3 + a 4 + a 5 + a 6 + a 7 = 450) :
  a 2 + a 8 = 180 :=
by
  sorry

end arithmetic_sequence_a2_a8_l312_312105


namespace BaSO4_molecular_weight_l312_312010

noncomputable def Ba : ℝ := 137.327
noncomputable def S : ℝ := 32.065
noncomputable def O : ℝ := 15.999
noncomputable def BaSO4 : ℝ := Ba + S + 4 * O

theorem BaSO4_molecular_weight : BaSO4 = 233.388 := by
  sorry

end BaSO4_molecular_weight_l312_312010


namespace jaydee_typing_time_l312_312576

theorem jaydee_typing_time : 
  (∀ (wpm total_words : ℕ) (minutes_per_hour : ℕ),
    wpm = 38 ∧ total_words = 4560 ∧ minutes_per_hour = 60 → 
      (total_words / wpm) / minutes_per_hour = 2) :=
begin
  intros wpm total_words minutes_per_hour h,
  cases h with hwpm hwords_hours,
  cases hwords_hours with hwords hhours,
  rw [hwpm, hwords, hhours],
  norm_num,
end

end jaydee_typing_time_l312_312576


namespace count_divisors_36_l312_312482

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312482


namespace correct_points_and_order_l312_312104

structure TeamScores where
  wins : Nat
  losses : Nat
  draws : Nat
  bonus_wins : Nat
  extra_bonus_matches : Nat

def points (team : TeamScores) : Nat :=
  (team.wins * 3) + (team.draws * 1) + (team.bonus_wins * 2) + (team.extra_bonus_matches * 1)

def TeamSoccerStars := TeamScores.mk 18 5 7 6 4
def LightningStrikers := TeamScores.mk 15 8 7 5 3
def GoalGrabbers := TeamScores.mk 21 5 4 4 9
def CleverKickers := TeamScores.mk 11 10 9 2 1

def total_points_and_order_correct : Prop :=
  (points TeamSoccerStars = 77) ∧
  (points LightningStrikers = 65) ∧
  (points GoalGrabbers = 84) ∧
  (points CleverKickers = 47) ∧
  ([GoalGrabbers, TeamSoccerStars, LightningStrikers, CleverKickers]
   = List.sort (fun a b => points a > points b) 
     [GoalGrabbers, TeamSoccerStars, LightningStrikers, CleverKickers])

theorem correct_points_and_order : total_points_and_order_correct :=
  by
  sorry

end correct_points_and_order_l312_312104


namespace sufficient_but_not_necessary_condition_l312_312080

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x * (x - 1) < 0 → x < 1) ∧ ¬(x < 1 → x * (x - 1) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l312_312080


namespace hyperbola_eccentricity_l312_312416

def hyperbola (a b : ℝ) := { p : ℝ × ℝ // (p.fst ^ 2 / a ^ 2) - (p.snd ^ 2 / b ^ 2) = 1 }

def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((sqrt(a^2 + b^2), 0), (-sqrt(a^2 + b^2), 0))

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2))

theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0) (p : ℝ × ℝ)
  (hp : p ∈ hyperbola a b)
  (dist_pf1_pf2_eq : distance p (foci a b).fst * distance p (foci a b).snd = 8 * a^2)
  (angle_pf1f2_30 : angle (foci a b).fst (foci a b).snd p = real.pi / 6) :
  eccentricity (hyperbola a b) = sqrt 3 :=
begin
  sorry
end

end hyperbola_eccentricity_l312_312416


namespace triangle_diameter_ratio_l312_312410

theorem triangle_diameter_ratio (a b c : ℕ) (h : a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2) :
  (let r_inscribed := (a + b - c) / 2 in
   let d_inscribed := 2 * r_inscribed in
   d_inscribed / c = 4 / 13) :=
by
  sorry

end triangle_diameter_ratio_l312_312410


namespace solve_x_l312_312181

theorem solve_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.84) : x = 72 := 
by
  sorry

end solve_x_l312_312181


namespace triangle_obtuse_if_midline_gt_median_l312_312557

-- Lean statement for the theorem
theorem triangle_obtuse_if_midline_gt_median 
  (A B C M N K : Point) 
  (triangle_ABC : Triangle A B C) 
  (M_mid_AB : Midpoint M A B)
  (N_mid_BC : Midpoint N B C)
  (K_mid_AC : Midpoint K A C)
  (MN_is_midline : LineSegment M N)
  (AN_is_median : LineSegment A N)
  (BK_is_median : LineSegment B K)
  (MN_gt_AN : length MN > length AN ∨ length MN > length BK) :
  obtuse_triangle A B C := 
sorry

end triangle_obtuse_if_midline_gt_median_l312_312557


namespace student_minimum_earnings_l312_312796

def minimum_weekly_earnings (library_hours construction_hours library_rate construction_rate total_hours: ℕ) (h1: library_hours = 10) (h2: library_rate = 8) (h3: construction_rate = 15) (h4: total_hours = 25) : ℕ :=
  library_rate * library_hours + construction_rate * (total_hours - library_hours)

theorem student_minimum_earnings :
  minimum_weekly_earnings 10 15 8 15 25 10 8 15 25 = 305 :=
by
sory

end student_minimum_earnings_l312_312796


namespace bacteria_growth_how_many_hours_l312_312670

theorem bacteria_growth:
  ∀ (t: ℕ), 5 * t = 30 →
  initially_bacteria = 200 →
  bacteria_every_interval = 3 →
  interval_hour = 5 →
  initial_bacteria * (bacteria_every_interval ^ t) = 145800 :=
by
  intros t ht ha hb hc
  cases ht with 
  | intro h =>
  sorry

def bacteria_initial: ℕ := 200
def final_bacteria: ℕ := 145800

theorem how_many_hours 
  (interval: ℕ) ( multiple: ℕ) (initial_bacteria: ℕ)  (time_needed: ℕ): 
  initial_bacteria * multiple^time_needed = final_bacteria-> interval * time_needed = 30
  := 
by
  intros h1
  have h:= mul_eq_one interval time_needed
  
  cases h with 
  | intro h_cont =>
  sorry 

end bacteria_growth_how_many_hours_l312_312670


namespace games_played_l312_312001

theorem games_played (x : ℕ) : 
  (∃ x, ∀ Petya_games Kolya_games Vasya_games : ℕ,
  Petya_games = x / 2 ∧
  Kolya_games = x / 3 ∧
  Vasya_games = x / 5 ∧
  (x % 2 = 0) ∧
  (x % 3 = 0) ∧ 
  (x % 5 = 0) ∧
  (0 ≤ Petya_games ∧ 0 ≤ Kolya_games ∧ 0 ≤ Vasya_games) ∧
  (Petya_games, Kolya_games and Vasya_games are whole numbers)
  (Petya and Kolya played only one game))
  (x = 30) sorry

end games_played_l312_312001


namespace four_point_questions_l312_312272

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 :=
sorry

end four_point_questions_l312_312272


namespace arithmetic_expression_evaluation_l312_312814

theorem arithmetic_expression_evaluation :
  1325 + (180 / 60) * 3 - 225 = 1109 :=
by
  sorry -- To be filled with the proof steps

end arithmetic_expression_evaluation_l312_312814


namespace integral_rational_term_expansion_l312_312568

theorem integral_rational_term_expansion :
  ∫ x in 0.0..1.0, x ^ (1/6 : ℝ) = 6/7 := by
  sorry

end integral_rational_term_expansion_l312_312568


namespace angle_BAD_is_105_l312_312170

noncomputable def triangle_angles (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
  let ∠ABC := 75
  let ∠BAC := 105
  ∃ ∠ABD ∠DBC, ∠ABD = 15 ∧ ∠DBC = 60 ∧ ∠BAD = 105

theorem angle_BAD_is_105 (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (D_is_on_AC : D ∈ AC) (angle_ABD : 15) (angle_DBC : 60) :
  ∃ angle_BAD, angle_BAD = 105 :=
begin
  sorry
end

end angle_BAD_is_105_l312_312170


namespace cyclic_sum_inequality_l312_312400

open Real

theorem cyclic_sum_inequality
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / c + b^2 / a + c^2 / b) + (b^2 / c + c^2 / a + a^2 / b) + (c^2 / a + a^2 / b + b^2 / c) + 
  7 * (a + b + c) 
  ≥ ((a + b + c)^3) / (a * b + b * c + c * a) + (2 * (a * b + b * c + c * a)^2) / (a * b * c) := 
sorry

end cyclic_sum_inequality_l312_312400


namespace simplify_fraction_l312_312906

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hx2 : x^2 - (1 / y) ≠ 0) (hy2 : y^2 - (1 / x) ≠ 0) :
  (x^2 - 1 / y) / (y^2 - 1 / x) = (x * (x^2 * y - 1)) / (y * (y^2 * x - 1)) :=
sorry

end simplify_fraction_l312_312906


namespace find_2023rd_term_l312_312395

theorem find_2023rd_term : 
  let groups := λ n => (fintype.elems (fin (n+1))).map (λ f => f + (n * (n+1) / 2) + 1)
  let sequence := list.join (list.map groups (list.range 2023))
  (sequence.nth (2023 - 1)).get_or_else 0 = 2023 :=
by {
  -- Placeholder for the proof
  sorry
}

end find_2023rd_term_l312_312395


namespace compute_2A_cubed_l312_312333

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[2, -2], [3, -1]]

theorem compute_2A_cubed : 2 • (A^3) = ![![ -20, 12], [-18, -2]] := by
  sorry

end compute_2A_cubed_l312_312333


namespace evaluate_Q_min_value_l312_312686

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 - 3*x^2 + 6*x - 5

theorem evaluate_Q_min_value :
  let Q1 := Q 1
  let prod_zeros := -5 -- product of the zeros (leading coefficient is 1)
  let sum_coeffs := -3 -- sum of the coefficients
  -- Note: We bypass the exact value for product of nonreal zeros and sum of real zeros
  (Q1 = -3) ∧ (Q1 ≤ prod_zeros) ∧ (Q1 ≤ sum_coeffs) :=
by
  have Q1 : Q 1 = -3 := by norm_num;
  have prod_zeros : -5 = -5 := by norm_num;
  have sum_coeffs : -3 = -3 := by norm_num;
  exact ⟨Q1, by linarith, by linarith⟩

end evaluate_Q_min_value_l312_312686


namespace solvable_system_l312_312286

theorem solvable_system (p : ℝ) : 
  (∃ (y : ℝ), ∃ (x : ℝ), ([⌊x⌋] : ℝ) = x /\
    2 * x + y = 3 / 2 ∧ 
    3 * x - 2 * y = p) ↔ 
  (∃ (k : ℤ), p = 7 * k - 3) := 
begin
  sorry
end

end solvable_system_l312_312286


namespace find_nickels_l312_312545

noncomputable def num_quarters1 := 25
noncomputable def num_dimes := 15
noncomputable def num_quarters2 := 15
noncomputable def value_quarter := 25
noncomputable def value_dime := 10
noncomputable def value_nickel := 5

theorem find_nickels (n : ℕ) :
  value_quarter * num_quarters1 + value_dime * num_dimes = value_quarter * num_quarters2 + value_nickel * n → 
  n = 80 :=
by
  sorry

end find_nickels_l312_312545


namespace sequence_reaches_integer_l312_312282

theorem sequence_reaches_integer (x : ℚ) : ∃ (f : ℕ → ℚ), 
  f 0 = x ∧ 
  (∀ n, f (n+1) = 2 * f n ∨ f (n+1) = 2 * f n + 1 / (n+1)) ∧
  (∃ n, (f n).denom = 1) :=
begin
  sorry
end

end sequence_reaches_integer_l312_312282


namespace smallest_four_digit_divisible_by_53_ending_in_3_l312_312248

theorem smallest_four_digit_divisible_by_53_ending_in_3 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n % 10 = 3 ∧ n = 1113 := 
by
  sorry

end smallest_four_digit_divisible_by_53_ending_in_3_l312_312248


namespace repeated_roots_coincide_l312_312351

noncomputable theory
open Classical

variables {R : Type*} [CommRing R] [IsDomain R]

def repeated_root (f : Polynomial R) (r : R) : Prop :=
  (Polynomial.derivative f).eval r = 0 ∧ f.eval r = 0

theorem repeated_roots_coincide
  {P Q : Polynomial ℝ}
  (hP : ∃ u : ℝ, P = Polynomial.C u * (X - C u)^2)
  (hQ : ∃ v : ℝ, Q = Polynomial.C v * (X - C v)^2)
  (hP_Q : ∃ w : ℝ, P + Q = Polynomial.C w * (X - C w)^2) :
  ∃ t : ℝ, (∃ u : ℝ, P = Polynomial.C u * (X - C u)^2 ∧ u = t) ∧ (∃ v : ℝ, Q = Polynomial.C v * (X - C v)^2 ∧ v = t) :=
begin
  sorry
end

end repeated_roots_coincide_l312_312351


namespace max_permutations_Q_l312_312608

open Fintype Finset

def valid_permutations (Q : Finset (Equiv.Perm (Fin 100))) : Prop :=
  ∀ a b : Fin 100, a ≠ b → 
  (Finset.univ.filter (λ σ : Equiv.Perm (Fin 100), σ (a) + 1 = σ (b))).card ≤ 1

theorem max_permutations_Q : 
  ∃ (Q : Finset (Equiv.Perm (Fin 100))), 
  valid_permutations Q ∧ Q.card = 100 :=
sorry

end max_permutations_Q_l312_312608


namespace length_IJ_eq_AH_l312_312150

-- Defining the geometric configuration
variables {A B C H G I J : Type*}
variables [euclidean_geometry] -- Assuming a Euclidean geometry context

-- Conditions
-- Acute triangle ABC
axiom acuteTriangle (ABC: Triangle) : acute ABC
-- Orthocenter of triangle ABC is H
axiom orthocenter (ABC: Triangle) (H: Point) : H = orthocenter ABC
-- G is the intersection of the line parallel to AB through H with the line parallel to AH through B
axiom G_intersection (H A B G: Point) : is_parallel H A B ∧ is_parallel B H A G
-- I is the point on the line GH such that AC bisects the segment HI
axiom I_bisect_HI (H G I C: Point) : is_parallel G H I ∧ AC bisects HI
-- J is the second intersection of AC with the circumcircle of triangle CGI
axiom J_circumcircle (C G I J: Point) : J = second_intersection AC (circumcircle C G I)

-- Prove that IJ = AH
theorem length_IJ_eq_AH
  (ABC: Triangle) 
  (H A B C G I J: Point) 
  (acuteTriangle: Triangle → Prop) 
  (H_eq: H = orthocenter ABC)
  (G_parallel: is_parallel H A B ∧ is_parallel B H A G)
  (I_bisect: is_parallel G H I ∧ AC bisects HI)
  (J_intersect: J = second_intersection AC (circumcircle C G I)):
  length IJ = length AH :=
sorry

end length_IJ_eq_AH_l312_312150


namespace range_of_f_l312_312923

def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 3)

theorem range_of_f :
  set.range (f ∘ (λ x, x : ℝ)) ∩ set.Icc 0 (Real.pi / 3) = set.Icc (-3 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end range_of_f_l312_312923


namespace ellipse_foci_coordinates_l312_312852

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, x^2 / 9 + y^2 / 5 = 1 → (x = 2 ∧ y = 0) ∨ (x = -2 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l312_312852


namespace average_speed_l312_312003

theorem average_speed
  (distance : ℝ)
  (time_hours : ℝ)
  (time_minutes : ℝ)
  (total_time : ℝ := time_hours + time_minutes / 60)
  (speed : ℝ := distance / total_time)
  (distance_eq : distance = 280)
  (time_hours_eq : time_hours = 2)
  (time_minutes_eq : time_minutes = 20)
  (approx_eq : ∀ (x y : ℝ), (|x - y| ≤ 0.01) → x ≈ y) :
  speed ≈ 120 := 
by
  sorry

end average_speed_l312_312003


namespace oven_capacity_correct_l312_312575

-- Definitions for the conditions
def dough_time := 30 -- minutes
def bake_time := 30 -- minutes
def pizzas_per_batch := 3
def total_time := 5 * 60 -- minutes (5 hours)
def total_pizzas := 12

-- Calculation of the number of batches
def batches_needed := total_pizzas / pizzas_per_batch

-- Calculation of the time for making dough
def dough_preparation_time := batches_needed * dough_time

-- Calculation of the remaining time for baking
def remaining_baking_time := total_time - dough_preparation_time

-- Calculation of the number of 30-minute baking intervals
def baking_intervals := remaining_baking_time / bake_time

-- Calculation of the capacity of the oven
def oven_capacity := total_pizzas / baking_intervals

theorem oven_capacity_correct : oven_capacity = 2 := by
  sorry

end oven_capacity_correct_l312_312575


namespace john_walking_distance_l312_312583

def total_distance : ℝ := 10
def skateboard_speed : ℝ := 10
def walk_speed : ℝ := 6
def total_time : ℝ := 66 / 60

theorem john_walking_distance : 
  ∃ (walk_distance : ℝ), walk_distance = 5 ∧ 
  total_distance = walk_distance + (total_distance - walk_distance) ∧ 
  (total_distance - walk_distance) / skateboard_speed + walk_distance / walk_speed = total_time :=
sorry

end john_walking_distance_l312_312583


namespace cookie_combinations_l312_312761

theorem cookie_combinations (kinds cookies : Nat) (h_kinds : kinds = 4) (h_cookies : cookies = 8) :
  ∃ n, n = 46 ∧ n = (number_combinations kinds cookies) :=
by
  sorry

-- Helper definition to count the number of valid combinations
noncomputable def number_combinations (kinds cookies : Nat) : Nat :=
  sorry

end cookie_combinations_l312_312761


namespace find_vertex_of_parabola_l312_312198

theorem find_vertex_of_parabola :
  ∃ (x y : ℚ), 2*y^2 + 8*y + 3*x + 7 = 0 ∧ (x = 1/3 ∧ y = -2) :=
by
  use [1/3, -2]
  split
  { norm_num }
  { norm_num }

end find_vertex_of_parabola_l312_312198


namespace smallest_triangle_perimeter_consecutive_even_l312_312264

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l312_312264


namespace smoking_negative_correlation_l312_312743

theorem smoking_negative_correlation (H : "Smoking is harmful to health") : 
  "smoking has a negative correlation with health" :=
sorry

end smoking_negative_correlation_l312_312743


namespace magnitude_a_plus_2b_l312_312933

variables (a b : ℝ × ℝ)
def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- Given conditions
def a := (1, 0)  -- \(\overrightarrow{a}\)
def b_magnitude := 1
def theta := 120 * real.pi / 180  -- angle in radians

-- Assume \(\overrightarrow{b}\) has magnitude 1 and the angle between a and b is 120 degrees.
axiom b_has_magnitude : vector_magnitude b = 1
axiom angle_a_b : real.cos theta = -1 / 2
-- Prove the required result statement
theorem magnitude_a_plus_2b : vector_magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = real.sqrt 3 :=
sorry

end magnitude_a_plus_2b_l312_312933


namespace min_colors_for_2x2_unique_l312_312124

theorem min_colors_for_2x2_unique : ∃ (n : ℕ), (∀ (coloring : ℤ × ℤ → ℕ), 
  (∀ (i j : ℤ), i ≥ 0 → j ≥ 0 → ∀ (di dj : ℕ), (di, dj) = (1, 1) ∨ (di, dj) = (1, 0) ∨ (di, dj) = (0, 1) ∨ (di, dj) = (0, 0) → 
  coloring (i + di, j + dj) ≠ coloring (i, j)) →
  n = 8 :=
begin
  sorry
end

end min_colors_for_2x2_unique_l312_312124


namespace final_result_after_subtracting_150_l312_312309

theorem final_result_after_subtracting_150
  (chosen_number : ℕ)
  (multiply_factor : ℕ)
  (subtract_value : ℕ)
  (h_chosen : chosen_number = 40)
  (h_factor : multiply_factor = 7)
  (h_subtract : subtract_value = 150) :
  (chosen_number * multiply_factor - subtract_value) = 130 :=
by
  -- Conditions given directly in the problem
  simp [h_chosen, h_factor, h_subtract]
  sorry

end final_result_after_subtracting_150_l312_312309


namespace find_n_value_l312_312015

noncomputable def sum_series (n : ℕ) : ℝ :=
∑ k in Finset.range n, (1 / (Real.sqrt (k + 1) + Real.sqrt (k + 2)))

theorem find_n_value : ∃ n : ℕ, sum_series n = 2011 ∧ n = 4048143 := by
  sorry

end find_n_value_l312_312015


namespace num_divisors_of_36_l312_312524

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312524


namespace find_OM_l312_312966

theorem find_OM (O M A B : Point) (h1 : chord AB O M)
  (h2 : angle_ABM_diameter = 60)
  (h3 : distance A M = 10)
  (h4 : distance B M = 4) :
  distance O M = 6 := 
sorry

end find_OM_l312_312966


namespace smallest_perimeter_consecutive_even_triangle_l312_312252

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l312_312252


namespace ratio_proof_l312_312770

theorem ratio_proof (x y z s : ℝ) (h1 : x < y) (h2 : y < z)
    (h3 : (x : ℝ) / y = y / z) (h4 : x + y + z = s) (h5 : x + y = z) :
    (x / y = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end ratio_proof_l312_312770


namespace no_real_roots_of_quadratic_l312_312203

theorem no_real_roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -2 ∧ c = 3) (h_eqn : ∀ x : ℝ, a * x^2 + b * x + c = 0) :
  b^2 - 4 * a * c < 0 :=
by {
  have ha : a = 1 := h_eq.1,
  have hb : b = -2 := h_eq.2.1,
  have hc : c = 3 := h_eq.2.2,
  calc 
    b^2 - 4 * a * c = (-2)^2 - 4 * 1 * 3 : by rw [ha, hb, hc]
    ... = 4 - 12 : by norm_num
    ... = -8 : by norm_num,
  sorry
}

end no_real_roots_of_quadratic_l312_312203


namespace sum_of_lowest_two_scores_l312_312615
noncomputable def total_score (scores : List ℚ) : ℚ := scores.sum
noncomputable def average (scores : List ℚ) : ℚ := scores.sum / scores.length
noncomputable def median (scores : List ℚ) : ℚ := (scores.sort)[scores.length / 2]
noncomputable def mode (scores : List ℚ) : ℚ := scores.max_by (λ x, (scores.count x, x))

theorem sum_of_lowest_two_scores 
  (scores : List ℚ) 
  (h_average : average scores = 90) 
  (h_median : median scores = 91) 
  (h_mode : mode scores = 93) 
  (h_length : scores.length = 5) : 
  (∃ a b c d e : ℚ, scores = [a, b, c, d, e] ∧ (a + b) = 173) := 
sorry

end sum_of_lowest_two_scores_l312_312615


namespace dilation_image_l312_312675

open Complex

noncomputable def dilation_center := (1 : ℂ) + (3 : ℂ) * I
noncomputable def scale_factor := -3
noncomputable def initial_point := -I
noncomputable def target_point := (4 : ℂ) + (15 : ℂ) * I

theorem dilation_image :
  let c := dilation_center
  let k := scale_factor
  let z := initial_point
  let z_prime := target_point
  z_prime = c + k * (z - c) := 
  by
    sorry

end dilation_image_l312_312675


namespace find_y_given_conditions_l312_312771

theorem find_y_given_conditions :
  ∃ y : ℝ, y > 0 ∧ dist (3, 7) (-5, y) = 12 ∧ y = 7 + 4 * real.sqrt 5 :=
by {
  sorry
}

end find_y_given_conditions_l312_312771


namespace youseff_lives_6_blocks_from_office_l312_312730

-- Definitions
def blocks_youseff_lives_from_office (x : ℕ) : Prop :=
  ∃ t_walk t_bike : ℕ,
    t_walk = x ∧
    t_bike = (20 * x) / 60 ∧
    t_walk = t_bike + 4

-- Main theorem
theorem youseff_lives_6_blocks_from_office (x : ℕ) (h : blocks_youseff_lives_from_office x) : x = 6 :=
  sorry

end youseff_lives_6_blocks_from_office_l312_312730


namespace divisors_of_36_l312_312475

def is_divisor (n : Int) (d : Int) : Prop := d ≠ 0 ∧ n % d = 0

def positive_divisors (n : Int) : List Int := 
  List.filter (λ d, d > 0 ∧ is_divisor n d) (List.range (Int.toNat n + 1))

def total_divisors (n : Int) : List Int :=
  positive_divisors n ++ List.map (λ d, -d) (positive_divisors n)

theorem divisors_of_36 : ∃ d, d = 36 ∧ (total_divisors d).length = 18 := by
  sorry

end divisors_of_36_l312_312475


namespace percent_red_tint_new_mixture_l312_312775

namespace MixtureProblem

-- Definitions of the given conditions
def original_volume : ℕ := 40
def red_tint_percentage : ℝ := 0.20
def added_red_tint : ℕ := 8

-- Function to calculate new mixture percentage
def new_red_tint_percentage (orig_vol : ℕ) (orig_pct : ℝ) (added_red : ℕ) : ℝ :=
  let original_red_tint := orig_vol * orig_pct
  let total_red_tint := original_red_tint + added_red
  let new_total_volume := orig_vol + added_red
  (total_red_tint / new_total_volume) * 100

-- Statement to prove
theorem percent_red_tint_new_mixture : 
  new_red_tint_percentage original_volume red_tint_percentage added_red_tint = 33.33 := by
  sorry

end MixtureProblem

end percent_red_tint_new_mixture_l312_312775


namespace smallest_triangle_perimeter_l312_312260

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l312_312260


namespace prob_eventA_and_eventB_l312_312703

def boxA := {1, 2, ..., 25}
def boxB := {1, 2, ..., 30}

def eventA := {n ∈ boxA | n < 20}
def eventB := {n ∈ boxB | Nat.Prime n ∨ n > 28}

def probA : ℚ := 19 / 25
def probB : ℚ := 11 / 30

theorem prob_eventA_and_eventB : 
  probA * probB = 209 / 750 :=
by 
  sorry

end prob_eventA_and_eventB_l312_312703


namespace num_divisors_36_l312_312499

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312499


namespace original_number_19_l312_312267

theorem original_number_19 (k : ℤ) : ∃ N : ℤ, N + 4 = 23 * k ∧ N = 19 := by
  use 19
  split
  case left =>
    use 1
    sorry
  case right =>
    sorry

end original_number_19_l312_312267


namespace find_x_l312_312145

noncomputable def star (a b : ℝ) : ℝ := (real.sqrt (a + b + 36)) / (real.sqrt (a - b))

theorem find_x (x : ℝ) (h : star x 36 = 9) : x = 37 :=
sorry

end find_x_l312_312145


namespace brother_spent_on_highlighters_l312_312940

theorem brother_spent_on_highlighters : 
  let total_money := 100
  let cost_sharpener := 5
  let num_sharpeners := 2
  let cost_notebook := 5
  let num_notebooks := 4
  let cost_eraser := 4
  let num_erasers := 10
  let total_spent_sharpeners := num_sharpeners * cost_sharpener
  let total_spent_notebooks := num_notebooks * cost_notebook
  let total_spent_erasers := num_erasers * cost_eraser
  let total_spent := total_spent_sharpeners + total_spent_notebooks + total_spent_erasers
  let remaining_money := total_money - total_spent
  remaining_money = 30 :=
begin
  sorry
end

end brother_spent_on_highlighters_l312_312940


namespace decks_left_is_3_l312_312303

-- Given conditions
def price_per_deck := 2
def total_decks_start := 5
def money_earned := 4

-- The number of decks sold
def decks_sold := money_earned / price_per_deck

-- The number of decks left
def decks_left := total_decks_start - decks_sold

-- The theorem to prove 
theorem decks_left_is_3 : decks_left = 3 :=
by
  -- Here we put the steps to prove
  sorry

end decks_left_is_3_l312_312303


namespace ratio_of_polygon_sides_l312_312175

open Real

theorem ratio_of_polygon_sides (n : ℕ) (a : Fin n → ℝ) (h : ∀ i : Fin n, 0 < a i) (hl : ∀ i j : Fin n, a i > a j → j.1 ≤ i.1) (poly_ineq : ∀ k : Fin n, a k < (∑ j in Finset.erase (Finset.univ : Finset (Fin n)) k, a j)) :
  ∃ i j : Fin n, i ≠ j ∧ (1 / 2) < a i / a j ∧ a i / a j < 2 :=
by
  sorry

end ratio_of_polygon_sides_l312_312175


namespace apples_in_box_l312_312224

theorem apples_in_box :
  (∀ (o p a : ℕ), 
    (o = 1 / 4 * 56) ∧ 
    (p = 1 / 2 * o) ∧ 
    (a = 5 * p) → 
    a = 35) :=
  by sorry

end apples_in_box_l312_312224


namespace weight_lifting_requirement_l312_312188

-- Definitions based on conditions
def weight_25 : Int := 25
def weight_10 : Int := 10
def lifts_25 := 16
def total_weight_25 := 2 * weight_25 * lifts_25

def n_lifts_10 (n : Int) := 2 * weight_10 * n

-- Problem statement and theorem to prove
theorem weight_lifting_requirement (n : Int) : n_lifts_10 n = total_weight_25 ↔ n = 40 := by
  sorry

end weight_lifting_requirement_l312_312188


namespace sum_of_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l312_312219

-- Problem 1
theorem sum_of_reciprocals (x y : ℝ) (hx : x + y = 50) (hxy : x * y = 25) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

-- Problem 2
theorem perpendicular_lines (a b : ℝ) (ha : a = 2) 
  (eq1 : ∀ x y : ℝ, ax + 2y + 1 = 0) (eq2 : ∀ x y : ℝ, 3x + by + 5 = 0) : 
  b = -3 := 
by
  sorry

-- Problem 3
theorem equilateral_triangle_perimeter (A : ℝ) (hA : A = 100 * real.sqrt 3) : 
  ∃ (p : ℝ), p = 60 :=
by
  sorry

-- Problem 4
theorem polynomial_divisibility (p q : ℝ) (hp : p = 60) 
  (hq : ∀ x : ℝ, (x + 2) ∣ (x^3 - 2*x^2 + p*x + q)) : 
  q = 136 := 
by
  sorry

end sum_of_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l312_312219


namespace num_divisors_36_l312_312497

theorem num_divisors_36 : ∃ n : ℕ, n = 18 ∧ ∀ d : ℤ, (d ≠ 0 → 36 % d = 0) → nat_abs d ∣ 36 :=
by
  sorry

end num_divisors_36_l312_312497


namespace problem_statement_l312_312948

noncomputable def a : ℝ := Real.log 6 / Real.log 2
noncomputable def b : ℝ := Real.log 3 / Real.log 2

theorem problem_statement (a b : ℝ) (h₁ : 2 ^ a = 6) (h₂ : b = Real.log 3 / Real.log 2) : a - b = 1 := 
by {
      sorry
}

end problem_statement_l312_312948


namespace femaleRainbowTroutCount_l312_312550

noncomputable def numFemaleRainbowTrout : ℕ := 
  let numSpeckledTrout := 645
  let numFemaleSpeckled := 200
  let numMaleSpeckled := 445
  let numMaleRainbow := 150
  let totalTrout := 1000
  let numRainbowTrout := totalTrout - numSpeckledTrout
  numRainbowTrout - numMaleRainbow

theorem femaleRainbowTroutCount : numFemaleRainbowTrout = 205 := by
  -- Conditions
  let numSpeckledTrout : ℕ := 645
  let numMaleSpeckled := 2 * 200 + 45
  let totalTrout := 645 + 355
  let numRainbowTrout := totalTrout - numSpeckledTrout
  let numFemaleRainbow := numRainbowTrout - 150
  
  -- The proof would proceed here
  sorry

end femaleRainbowTroutCount_l312_312550


namespace ninth_term_of_geometric_sequence_l312_312829

theorem ninth_term_of_geometric_sequence :
  let a1 := (5 : ℚ)
  let r := (3 / 4 : ℚ)
  (a1 * r^8) = (32805 / 65536 : ℚ) :=
by {
  sorry
}

end ninth_term_of_geometric_sequence_l312_312829


namespace subset_A_of_intersection_l312_312892

-- Given conditions
def B := {x : ℝ | x ≥ 0}
def A := {1, 2}

-- Statement to prove
theorem subset_A_of_intersection (A : Set ℝ) (h1 : ∀ x ∈ A, x ≥ 0) (h2 : ∀ x ∈ A, x ∈ B) : A ⊆ {1, 2} :=
by
  sorry

end subset_A_of_intersection_l312_312892


namespace decreasing_interval_of_f_l312_312749

def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

theorem decreasing_interval_of_f :
  { x : ℝ // 0 < x ∧ x < 2 } = set.Ioo 0 2 := 
sorry

end decreasing_interval_of_f_l312_312749


namespace thirty_five_billion_in_scientific_notation_l312_312691

def n : ℕ := 35000000000

theorem thirty_five_billion_in_scientific_notation : n = 3.5 * 10^10 :=
  sorry

end thirty_five_billion_in_scientific_notation_l312_312691


namespace largest_n_for_factored_polynomial_l312_312867

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l312_312867


namespace orthocenter_circumcenter_l312_312985

theorem orthocenter_circumcenter (A B C H Ω : Point) (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 10)
  (hH : vectorEq H = 0 • vectorEq A + 0 • vectorEq B + 1 • vectorEq C)
  (hΩ : vectorEq Ω = 0 • vectorEq A + (1 / 2) • vectorEq B + (1 / 2) • vectorEq C)
  : ∀ (p q r u v w : ℝ), (p + q + r = 1 ∧ u + v + w = 1)
      → ((p, q, r) = (0, 0, 1) ∧ (u, v, w) = (0, 1 / 2, 1 / 2)) :=
  sorry

end orthocenter_circumcenter_l312_312985


namespace base5_div_l312_312559

-- Definitions for base 5 numbers
def n1 : ℕ := (2 * 125) + (4 * 25) + (3 * 5) + 4  -- 2434_5 in base 10 is 369
def n2 : ℕ := (1 * 25) + (3 * 5) + 2              -- 132_5 in base 10 is 42
def d  : ℕ := (2 * 5) + 1                          -- 21_5 in base 10 is 11

theorem base5_div (res : ℕ) : res = (122 : ℕ) → (n1 + n2) / d = res :=
by sorry

end base5_div_l312_312559


namespace num_divisors_of_36_l312_312518

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312518


namespace sequence_a_n_l312_312408

-- Define the sequence and sum condition
def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum a

theorem sequence_a_n (a : ℕ → ℕ) (H : ∀ n, S_n a n = 2 * a n - 2) : ∀ n, a n = 2 ^ n :=
by
  intro n
  sorry

end sequence_a_n_l312_312408


namespace more_than_two_thirds_millet_on_Friday_l312_312161

theorem more_than_two_thirds_millet_on_Friday :
  ∀ (n : ℕ), 
  (initial_millet : ℝ) // initial_millet = 0.5 * 0.3 → 
  (additional_millet : ℝ) // additional_millet = 0.5 * 0.3 →
  (remaining_millet : ℕ → ℝ) // 
    remaining_millet 0 = initial_millet →
    (∀ k, remaining_millet (k + 1) = 0.7 * remaining_millet k + additional_millet) →
    ∀ (total_seeds : ℕ → ℝ) // 
      total_seeds 0 = 0.5 →
      (∀ k, total_seeds (k + 1) = 0.5 + 0.7 * remaining_millet k) →
      (2 / 3 < (remaining_millet n) / (total_seeds n)) →
  n = 5 := sorry

end more_than_two_thirds_millet_on_Friday_l312_312161


namespace angie_and_diego_probability_l312_312809

theorem angie_and_diego_probability :
  let people := ["Angie", "Bridget", "Carlos", "Diego"]
  let arrangements := {arr : list String // arr.permutations}
  let seat_adjacent := ∀ p ∈ arrangements, p.nth 0 = some "Angie" → 
                       (p.nth 1 = some "Diego" ∨ p.nth 3 = some "Diego")
  ∃ favorable_counts = 
    list.count (λ a, seat_adjacent a) arrangements in
  favorable_counts / arrangements.size = 2 / 3 := sorry

end angie_and_diego_probability_l312_312809


namespace compute_expr_l312_312823

theorem compute_expr : 5^2 - 3 * 4 + 3^2 = 22 := by
  sorry

end compute_expr_l312_312823


namespace problem_solution_l312_312348

noncomputable def question (x y z : ℝ) : Prop := 
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) → 
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) )

theorem problem_solution (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) →
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ) :=
sorry

end problem_solution_l312_312348


namespace compare_nums_l312_312931

def num_a : ℝ := 6 ^ 0.7
def num_b : ℝ := 0.7 ^ 0.8
def num_c : ℝ := 0.8 ^ 0.7

theorem compare_nums : num_a > num_c ∧ num_c > num_b :=
by
  sorry

end compare_nums_l312_312931


namespace part_I_part_II_l312_312405

-- Proving properties of the sequence {a_n}
variables {a : ℕ → ℝ} (h_pos : ∀ n, 0 < a n) 
  (h_seq : ∀ m n, a m * a n = 2^(m + n + 2))

-- S_n: Sum of first n terms of { log_2 a_n }
def S_n (n : ℕ) : ℝ := ∑ i in finset.range n.succ, real.logb 2 (a i)

-- Proving the closed form for S_n
theorem part_I (n : ℕ) : S_n h_pos h_seq n = (n * (n + 3) / 2) :=
sorry

-- Defining b_n as a_n * log_2(a_n)
def b_n (n : ℕ) : ℝ := a n * real.logb 2 (a n)

-- T_n: Sum of first n terms of { b_n }
def T_n (n : ℕ) : ℝ := ∑ i in finset.range n.succ, b_n a i

-- Proving the closed form for T_n (n * 2^(n+2))
theorem part_II (n : ℕ) : T_n h_pos h_seq n = n * 2^(n+2) :=
sorry

end part_I_part_II_l312_312405


namespace num_divisors_of_36_l312_312491

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l312_312491


namespace barrel_empty_time_l312_312296

def total_capacity : ℝ := 220
def rate1 : ℝ := 2
def rate2 : ℝ := 3
def rate3 : ℝ := 4

def combined_rate : ℝ := rate1 + rate2 + rate3

def total_time : ℝ := total_capacity / combined_rate

theorem barrel_empty_time :
  total_time ≈ 220 / 9 :=
by sorry

end barrel_empty_time_l312_312296


namespace distance_from_point_in_angle_l312_312130

theorem distance_from_point_in_angle 
  (α : ℝ) (a b : ℝ) : 
  ∃ OC : ℝ, OC = sin α * sqrt (a^2 - b^2) - b * cos α := 
by 
  sorry

end distance_from_point_in_angle_l312_312130


namespace arithmetic_seq_sum_l312_312558

theorem arithmetic_seq_sum (a : ℕ → ℤ) (a1 a7 a3 a5 : ℤ) (S7 : ℤ)
  (h1 : a1 = a 1) (h7 : a7 = a 7) (h3 : a3 = a 3) (h5 : a5 = a 5)
  (h_arith : ∀ n m, a (n + m) = a n + a m - a 0)
  (h_S7 : (7 * (a1 + a7)) / 2 = 14) :
  a3 + a5 = 4 :=
sorry

end arithmetic_seq_sum_l312_312558


namespace allocation_schemes_l312_312755

theorem allocation_schemes (n m : ℕ) (h1 : n = 8) (h2 : m = 3) 
  (h3 : ∀ s : Finset (Fin n), s.card = 2 → ∃ t : Finset (Fin m), t.card = 1) 
  (h4 : ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i)
  (h5 : ∀ i, i ∈ insert 2 (insert 3 (insert 3 (singleton 2)))) :
  nat.factorial m = 2940 :=
by 
  sorry

end allocation_schemes_l312_312755


namespace binomial_expansion_coeff_l312_312542

theorem binomial_expansion_coeff (n : ℕ) (h : 4 * Nat.choose n 2 = 14 * n) : n = 8 :=
by
  sorry

end binomial_expansion_coeff_l312_312542


namespace gratuity_calculation_l312_312614

-- Define the prices
def price1 := 21
def price2 := 15
def price3 := 26
def price4 := 13
def price5 := 20

-- Define the discount rate, tax rate, and tip rate
def discount_rate := 0.15
def tax_rate := 0.08
def tip_rate := 0.18

-- Define the final calculated tip amount (expected answer)
def expected_tip := 15.70

-- Main theorem statement
theorem gratuity_calculation :
  let total_before_discount := price1 + price2 + price3 + price4 + price5 in
  let discount_amount := discount_rate * total_before_discount in
  let discounted_total := total_before_discount - discount_amount in
  let tax_amount := tax_rate * discounted_total in
  let final_amount_before_tip := discounted_total + tax_amount in
  let tip_amount := tip_rate * final_amount_before_tip in
  tip_amount = expected_tip := by
  sorry

end gratuity_calculation_l312_312614


namespace parallelogram_sides_l312_312208

theorem parallelogram_sides (perimeter : ℝ) (acute_angle : Real.Angle.o) (obtuse_angle_ratio : ℝ) 
(h_perimeter : perimeter = 90) 
(h_acute_angle : acute_angle = Real.Angle.of_deg 60)
(h_obtuse_angle_ratio : obtuse_angle_ratio = 1 / 3) : 
∃ AB AD : ℝ, AB = 15 ∧ AD = 30 :=
by
  sorry

end parallelogram_sides_l312_312208


namespace evaluate_magnitude_l312_312005

noncomputable def mag1 : ℂ := 3 * Real.sqrt 2 - 3 * Complex.I
noncomputable def mag2 : ℂ := Real.sqrt 5 + 5 * Complex.I
noncomputable def mag3 : ℂ := 2 - 2 * Complex.I

theorem evaluate_magnitude :
  Complex.abs (mag1 * mag2 * mag3) = 18 * Real.sqrt 10 :=
by
  sorry

end evaluate_magnitude_l312_312005


namespace drunk_drivers_traffic_class_l312_312970

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end drunk_drivers_traffic_class_l312_312970


namespace perpendicularity_proof_l312_312887

variables (l m : Line) (α β : Plane)

def parallel (x y : Line) : Prop := ∃ p, p ∈ x ∧ p ∉ y
def perp (x : Line) (y : Plane) : Prop := ∀ p ∈ x, p ∉ y
def parallel_planes (x y : Plane) : Prop := ∃ p, p ∈ x ∧ p ∉ y

theorem perpendicularity_proof
  (cond1 : parallel l m)
  (cond2 : parallel_planes α β)
  (cond3 : perp m α)
  (cond4 : perp l β) :
  (cond1 ∧ cond3) ∨ (cond2 ∧ cond4) → perp l α :=
begin
  sorry
end

end perpendicularity_proof_l312_312887


namespace stop_signs_per_mile_l312_312162

-- Define the conditions
def miles_traveled := 5 + 2
def stop_signs_encountered := 17 - 3

-- Define the proof statement
theorem stop_signs_per_mile : (stop_signs_encountered / miles_traveled) = 2 := by
  -- Proof goes here
  sorry

end stop_signs_per_mile_l312_312162


namespace smallest_perimeter_even_integer_triangl_l312_312255

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l312_312255


namespace beads_problem_l312_312156

theorem beads_problem :
  ∀ (total_beads blue_beads : ℕ) (red_beads white_beads green_beads silver_beads : ℕ), 
  total_beads = 200 →
  blue_beads = 12 →
  red_beads = 3 * blue_beads →
  white_beads = 1.5 * (blue_beads + red_beads) →
  green_beads = 0.5 * (blue_beads + red_beads + white_beads) →
  silver_beads = total_beads - (blue_beads + red_beads + white_beads + green_beads) →
  silver_beads = 20 :=
by sorry

end beads_problem_l312_312156


namespace area_of_equilateral_triangle_l312_312672

noncomputable def radius : ℝ := 24
noncomputable def side_length : ℝ := radius * 3.sqrt

theorem area_of_equilateral_triangle (a b : ℝ) (ha : a = 108) (hb : b = 324) :
  let A := sqrt a + sqrt b in 
  A = sqrt (3) * 432.sqrt := 
by 
  let side := side_length
  let area := (sqrt 3 * (side ^ 2) / 4)
  have h_area : A = area := by sorry
  rw [ha, hb]
  exact h_area

end area_of_equilateral_triangle_l312_312672


namespace clock_angle_820_l312_312715

variable (h m : ℕ) (θ_m θ_h : ℝ)

-- Given the time is 8:20
def time_is_820 : Prop := h = 8 ∧ m = 20

-- Define the angles moved by minute hand and hour hand
def angle_minute_hand (m : ℕ) : ℝ := m / 60 * 360
def angle_hour_hand (h : ℕ) (m : ℕ) : ℝ := (h + m / 60) / 12 * 360

-- Define the absolute difference between two angles
def angle_difference (θ1 θ2 : ℝ) : ℝ := abs (θ1 - θ2)

-- The goal is to prove that at 8:20, the angle between hour hand and minute hand is 130 degrees
theorem clock_angle_820 (h_eq: h = 8) (m_eq: m = 20) : 
  angle_difference (angle_hour_hand 8 20) (angle_minute_hand 20) = 130 :=
by
  -- Proof goes here
  sorry

end clock_angle_820_l312_312715


namespace sin_A_correct_l312_312102

theorem sin_A_correct (A B C : Type) [triangle : Triangle A B C]
  (hC : ∠ C = 90)
  (BC : length B C = 3)
  (AB : length A B = 4) :
  sin (angle A) = 3 / 4 := 
sorry

end sin_A_correct_l312_312102


namespace BE_eq_CF_l312_312460

-- Define the geometric setup and conditions.
variables {A B C P E F : Type} [linear_ordered_ring E] 
variables {coord : Type} [field coord] [add_group coord] [module E coord]

structure Triangle (A B C : coord) : Prop :=
(is_bisector : ∀ {P : coord}, P ∈ bisector_of_angle A B C)

structure Parallel (A B : coord) : Prop :=
(is_parallel : Parallel >=> A >=> B)

-- Main statement
theorem BE_eq_CF
  (A B C E F P : coord)
  (h_triangle : Triangle A B C)
  (h_parallel1 : Parallel E P)
  (h_parallel2 : Parallel F P) :
  BE = CF :=
begin
  sorry
end

end BE_eq_CF_l312_312460


namespace inscribe_square_possible_l312_312569

-- Definitions of the conditions in Lean 4
variable (O P Q : Point) (angle_POQ : ℝ)

-- Assumptions based on the given conditions
variable (h_OPQ_sector : isSector O P Q)
variable (h_arc_PQ : ∀ (A B : Point), A ∈ arc P Q → B ∈ arc P Q)
variable (h_op : ∀ A: Point, A ∈ segment O P)
variable (h_oq : ∀ B: Point, B ∈ segment O Q)
variable (h_angle_constraint : angle_POQ ≤ 180)

-- The statement to be proved
theorem inscribe_square_possible : 
  (∃ (A B C D : Point), 
     A ∈ segment O P ∧ 
     B ∈ segment O Q ∧ 
     C ∈ arc P Q ∧ 
     D ∈ arc P Q ∧ 
     isSquare A B C D) ↔ angle_POQ ≤ 180 :=
sorry

end inscribe_square_possible_l312_312569


namespace gcd_lcm_sum_l312_312598

def GCD (a b c : ℕ) := Nat.gcd (Nat.gcd a b) c
def LCM (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

theorem gcd_lcm_sum (C D : ℕ) (hC : C = GCD 6 15 30) (hD : D = LCM 6 15 30) : C + D = 33 :=
by
  have hC : C = 3 := by simp [hC, GCD]
  have hD : D = 30 := by simp [hD, LCM]
  simp [hC, hD]
  sorry

end gcd_lcm_sum_l312_312598


namespace helga_usual_daily_work_hours_l312_312462

theorem helga_usual_daily_work_hours :
  (∃ h : ℕ, let articles_per_hour := 10 in
            let extra_hours := 2 + 3 in
            let total_hours := 250 / articles_per_hour in
            let usual_weekly_hours := total_hours - extra_hours in
            h = usual_weekly_hours / 5) →
  ∃ h : ℕ, h = 4 :=
begin
  sorry
end

end helga_usual_daily_work_hours_l312_312462


namespace angle_BC₁_plane_BBD₁D_l312_312567

-- Define all the necessary components of the cube and its geometry
variables {A B C D A₁ B₁ C₁ D₁ : ℝ} -- placeholders for points, represented by real coordinates

def is_cube (A B C D A₁ B₁ C₁ D₁ : ℝ) : Prop := sorry -- Define the cube property (this would need a proper definition)

def space_diagonal (B C₁ : ℝ) : Prop := sorry -- Define the property of being a space diagonal

def plane (B B₁ D₁ D : ℝ) : Prop := sorry -- Define a plane through these points (again needs a definition)

-- Define the angle between a line and a plane
def angle_between_line_and_plane (BC₁ B B₁ D₁ D : ℝ) : ℝ := sorry -- Define angle calculation (requires more context)

-- The proof statement, which is currently not proven (contains 'sorry')
theorem angle_BC₁_plane_BBD₁D (s : ℝ):
  is_cube A B C D A₁ B₁ C₁ D₁ →
  space_diagonal B C₁ →
  plane B B₁ D₁ D →
  angle_between_line_and_plane B C₁ B₁ D₁ D = π / 6 :=
sorry

end angle_BC₁_plane_BBD₁D_l312_312567


namespace iris_jackets_l312_312131

theorem iris_jackets (J : ℕ) (h : 10 * J + 12 + 48 = 90) : J = 3 :=
by
  sorry

end iris_jackets_l312_312131


namespace trapezoid_base_ratio_l312_312664

variable {A B C D K M P Q L N : Type}
variable [LinearOrderedField A]
variable [LinearOrderedField B]
variable [LinearOrderedField C]
variable [LinearOrderedField D]

def is_trapezoid (A B C D : Type) (a b : ℝ) : Prop :=
  is_parallel (AD BC) ∧ AD.length = a ∧ BC.length = b ∧ a > b

def area (A B C D : Type) (a b : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

def quadrilateral_area (K L M N : Type) (a b h : ℝ) : ℝ :=
  (1 / 4) * (a - b) * h

theorem trapezoid_base_ratio (A B C D K M P Q L N : Type) (a b : ℝ) (h : ℝ) :
  is_trapezoid A B C D a b →
  quadrilateral_area K L M N a b h = (1 / 4) * area A B C D a b ↔ a / b = 3 :=
by 
  sorry

end trapezoid_base_ratio_l312_312664


namespace average_inside_time_l312_312137

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end average_inside_time_l312_312137


namespace tan_A_of_triangle_l312_312425

theorem tan_A_of_triangle (A B C : Type)
  [Triangle A B C] 
  (perimeter_ABC : Perimeter A B C = 20)
  (radius_incircle : InscribedCircleRadius A B C = sqrt 3)
  (BC_eq_seven : SideLength B C = 7) :
  tan (Angle A) = sqrt 3 := 
sorry

end tan_A_of_triangle_l312_312425


namespace find_sin_2alpha_l312_312901

variable (α : ℝ)

-- Condition given in the problem
def cos_condition : Prop := cos (α + π / 4) = 3 * sqrt 2 / 5

theorem find_sin_2alpha (h : cos_condition α) : sin (2 * α) = -11 / 25 := by
  sorry

end find_sin_2alpha_l312_312901


namespace chord_length_of_line_on_curve_l312_312927

noncomputable def parametric_curve_to_polar (α : ℝ) : ℝ × ℝ :=
  let x := 3 + sqrt 10 * real.cos α
  let y := 1 + sqrt 10 * real.sin α
  (x, y)

noncomputable def polar_eq_of_curve : ℝ → ℝ :=
  λ θ, 6 * real.cos θ + 2 * real.sin θ

theorem chord_length_of_line_on_curve :
  let l_eq := (λ θ, real.sin θ - real.cos θ = 2)
  let distance_center_to_line := 2 * sqrt 2
  let curve_radius := sqrt 10
  2 * sqrt (curve_radius^2 - distance_center_to_line^2) = 2 * sqrt 2 :=
by
  sorry

end chord_length_of_line_on_curve_l312_312927


namespace average_inside_time_l312_312134

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end average_inside_time_l312_312134


namespace solve_for_y_l312_312182

theorem solve_for_y
  (y : ℤ)
  (h : 5^(y+1) = 625) :
  y = 3 :=
sorry

end solve_for_y_l312_312182


namespace trapezoid_base_ratio_l312_312657

theorem trapezoid_base_ratio
  (a b h : ℝ)  -- lengths of the bases and height
  (ha_gt_hb : a > b)
  (trapezoid_area : ℝ) 
  (quad_area : ℝ) 
  (h1 : trapezoid_area = (1/2) * (a + b) * h) 
  (h2 : quad_area = (1/2) * (a - b) * h / 4)
  (h3 : quad_area = trapezoid_area / 4) :
  a / b = 3 :=
by {
  sorry,
}

end trapezoid_base_ratio_l312_312657


namespace calculateDistance_l312_312787

variable (RingThickness : ℕ := 1)
variable (TopRingOutsideDiameter : ℕ := 20)
variable (BottomRingOutsideDiameter : ℕ := 3)

def insideDiameter (outsideDiameter : ℕ) : ℕ :=
  outsideDiameter - 2 * RingThickness

def numberOfRings : ℕ :=
  (TopRingOutsideDiameter - BottomRingOutsideDiameter) + 1

def sumOfInsideDiameters : ℕ :=
  let n := numberOfRings
  let a := insideDiameter TopRingOutsideDiameter
  let l := insideDiameter BottomRingOutsideDiameter
  n * (a + l) / 2

def totalDistance : ℕ :=
  sumOfInsideDiameters + 2 * RingThickness

theorem calculateDistance : totalDistance = 173 := by
  sorry

end calculateDistance_l312_312787


namespace alvin_earns_14_dollars_l312_312320

noncomputable def total_earnings (total_marbles : ℕ) (percent_white percent_black : ℚ)
  (price_white price_black price_colored : ℚ) : ℚ :=
  let white_marbles := percent_white * total_marbles
  let black_marbles := percent_black * total_marbles
  let colored_marbles := total_marbles - white_marbles - black_marbles
  (white_marbles * price_white) + (black_marbles * price_black) + (colored_marbles * price_colored)

theorem alvin_earns_14_dollars :
  total_earnings 100 (20/100) (30/100) 0.05 0.10 0.20 = 14 := by
  sorry

end alvin_earns_14_dollars_l312_312320


namespace number_of_divisors_of_36_l312_312505

/-- The number of integers (positive and negative) that are divisors of 36 is 18. -/
theorem number_of_divisors_of_36 : 
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  in 2 * positive_divisors.card = 18 :=
by
  let positive_divisors := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  have h : positive_divisors.card = 9 := sorry
  show 2 * positive_divisors.card = 18
  by rw [h]; norm_num
  sorry

end number_of_divisors_of_36_l312_312505


namespace OA1_plus_OB1_plus_OC1_lt_a_l312_312590

theorem OA1_plus_OB1_plus_OC1_lt_a (a : ℝ) (A B C O A1 B1 C1 : Point)
  (h_equilateral : equilateral_triangle A B C a)
  (h_interior : interior_point O A B C)
  (h_A1 : intersection_point A O B C A1)
  (h_B1 : intersection_point B O C A B1)
  (h_C1 : intersection_point C O A B C1) :
  distance O A1 + distance O B1 + distance O C1 < a := sorry

end OA1_plus_OB1_plus_OC1_lt_a_l312_312590


namespace trapezoid_area_l312_312210

theorem trapezoid_area 
  (a b : ℝ) 
  (h : ℝ := 3 * (a - b))
  (product_midline_segments: ((a + b) / 2) * ((a - b) / 2) = 25) : 
  (S : ℝ := (1 / 2) * (a + b) * h) :
  S = 150 :=
by
  -- Conditions
  have ha_b_sq : a^2 - b^2 = 100, 
  from calc
    (a + b) / 2 * (a - b) / 2 = 25 : product_midline_segments
    (a^2 - b^2) / 4 = 25 : by ring
    a^2 - b^2 = 100 : by linarith,
  -- Height Definition
  have height_def : h = 3 * (a - b) := rfl,
  -- Area Calculation
  have area_calc : (S : ℝ) = (1 / 2) * (a + b) * h := rfl,
  -- Simplify and Show the Area is 150
  calc
    S = (1 / 2) * (a + b) * h : rfl
    _ = (1 / 2) * (a + b) * (3 * (a - b)) : by rw [height_def]
    _ = (1 / 2) * 3 * (a + b) * (a - b) : by ring
    _ = (3 / 2) * (a^2 - b^2) : by ring
    _ = (3 / 2) * 100 : by rw [ha_b_sq]
    _ = 150 : by norm_num

end trapezoid_area_l312_312210


namespace cheapest_solution_l312_312973

noncomputable def minimum_cost (m n : ℕ) : ℝ :=
  7.03 * m + 30 * n

theorem cheapest_solution (m n : ℕ)
  (h1 : 7 * m + 30 * n ≥ 1096)
  (h2 : minimum_cost m n = 1401.34) : Prop :=
  (m = 178 ∧ n = 5) :=
begin
  sorry
end

end cheapest_solution_l312_312973


namespace heptagon_diagonals_l312_312467

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l312_312467


namespace mean_score_40_l312_312193

theorem mean_score_40 (mean : ℝ) (std_dev : ℝ) (h_std_dev : std_dev = 10)
  (h_within_2_std_dev : ∀ (score : ℝ), score ≥ mean - 2 * std_dev)
  (h_lowest_score : ∀ (score : ℝ), score = 20 → score = mean - 20) :
  mean = 40 :=
by
  -- Placeholder for the proof
  sorry

end mean_score_40_l312_312193


namespace maximum_value_l312_312605

theorem maximum_value (x y : ℝ) (h : x + y = 5) :
  let f := x^5 * y + x^4 * y + x^3 * y + x * y + x * y^2 + x * y^3 + x * y^5 in f ≤ 72.25 :=
by
  sorry

end maximum_value_l312_312605


namespace find_ABC_l312_312677

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  ∀ (A B C : ℝ),
  (∀ (x : ℝ), x > 2 → g x A B C > 0.3) →
  (∃ (A : ℤ), A = 4) →
  (∃ (B : ℤ), ∃ (C : ℤ), A = 4 ∧ B = 8 ∧ C = -12) →
  A + B + C = 0 :=
by
  intros A B C h1 h2 h3
  rcases h2 with ⟨intA, h2'⟩
  rcases h3 with ⟨intB, ⟨intC, h3'⟩⟩
  simp [h2', h3']
  sorry -- proof skipped

end find_ABC_l312_312677


namespace vector_problem_l312_312040

noncomputable theory

def vector_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_problem
  (m n : ℝ × ℝ)
  (h_perp : dot_product m n = 0)
  (h_mn_eq : (m.1 - 2 * n.1, m.2 - 2 * n.2) = (11, -2))
  (h_m_length : vector_length m = 5)
  : vector_length n = 5 :=
by
  sorry

end vector_problem_l312_312040


namespace quotient_of_m_and_n_l312_312058

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem quotient_of_m_and_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) :
  n / m = Real.exp 2 :=
by
  sorry

end quotient_of_m_and_n_l312_312058


namespace min_value_of_a_plus_4b_l312_312022

theorem min_value_of_a_plus_4b (a b : ℝ) (h₁ : log a b = -1) : a + 4*b = 4 :=
by 
  sorry

end min_value_of_a_plus_4b_l312_312022


namespace divisors_of_36_count_l312_312510

theorem divisors_of_36_count : 
  {n : ℤ | n ∣ 36}.to_finset.card = 18 := 
sorry

end divisors_of_36_count_l312_312510


namespace num_divisors_of_36_l312_312525

theorem num_divisors_of_36 : 
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  let total_divisors := 2 * List.length positive_divisors in
  total_divisors = 18 :=
by
  let positive_divisors : List Int := [1, 2, 3, 4, 6, 9, 12, 18, 36]
  let total_divisors := 2 * List.length positive_divisors
  show total_divisors = 18
  sorry

end num_divisors_of_36_l312_312525


namespace intervals_of_monotonicity_max_min_value_ln_inequality_l312_312050

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x - log x

theorem intervals_of_monotonicity :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f' x > 0) ∧ (∀ x : ℝ, x > 1 → f' x < 0) :=
sorry

theorem max_min_value :
  ∃ M m : ℝ, (∀ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → m ≤ f x ∧ f x ≤ M) ∧
  (M = 0) ∧ (m = 2 - Real.exp 1) :=
sorry

theorem ln_inequality (x : ℝ) (hx : 0 < x) : log (Real.exp 2 / x) ≤ (1 + x) / x :=
sorry

end intervals_of_monotonicity_max_min_value_ln_inequality_l312_312050


namespace count_negatives_in_given_set_l312_312803

/-- The given set of real numbers -/
def given_set : List ℝ := [-5, abs(-1), -Real.pi / 2, -(-2019), 0, (-2018 : ℝ) ^ 2019]

/-- Prove that there are exactly 3 negative numbers in the given set -/
theorem count_negatives_in_given_set : given_set.count (λ x => x < 0) = 3 := by
  sorry

end count_negatives_in_given_set_l312_312803


namespace part_I_part_II_l312_312891

noncomputable def sequence_a (n : ℕ) : ℤ :=
  if n = 0 then 0 else (-1)^n

noncomputable def sequence_S (n : ℕ) : ℤ :=
  (finset.range n).sum sequence_a

noncomputable def condition_a (n : ℕ) : Prop :=
  sequence_a n = 2 * sequence_S n + 1

noncomputable def sequence_b (n : ℕ) : ℤ :=
  (2 * n - 1) * sequence_a n

noncomputable def sequence_T (n : ℕ) : ℤ :=
  (finset.range n).sum sequence_b

theorem part_I (n : ℕ) (h1 : ∀ n > 0, condition_a n) : sequence_a n = (-1)^n :=
by
  sorry

theorem part_II (n : ℕ) (h1 : ∀ n > 0, condition_a n) : sequence_T n = (-1)^n * n :=
by
  sorry

end part_I_part_II_l312_312891


namespace solve_for_y_l312_312185

theorem solve_for_y (y : ℤ) (h : 5^(y + 1) = 625) : y = 3 := by
  sorry

end solve_for_y_l312_312185


namespace part1_l312_312143

-- Define the concept of a naughty subset
def is_naughty (S : Set ℕ) (T : Set ℕ) : Prop :=
  ∀ u v ∈ T, u + v ∉ T

-- Part 1: Prove that every naughty subset of {1, 2, ..., 2006} has at most 1003 elements
theorem part1 (S : Set ℕ) (hS : S = { n | n ≠ 0 ∧ n ≤ 2006 }) (T : Set ℕ) 
  (hT : is_naughty S T) : T.card ≤ 1003 :=
sorry

-- Part 2: Prove there exists a naughty subset of S with at least 669 elements
noncomputable def part2 (S : Set ℕ) (hS : S.card = 2006) :
  ∃ T : Set ℕ, is_naughty S T ∧ T.card ≥ 669 :=
sorry

end part1_l312_312143


namespace find_lisa_speed_l312_312617

theorem find_lisa_speed (Distance : ℕ) (Time : ℕ) (h1 : Distance = 256) (h2 : Time = 8) : Distance / Time = 32 := 
by {
  sorry
}

end find_lisa_speed_l312_312617


namespace percentage_income_spent_on_clothes_l312_312799

-- Define the assumptions
def monthly_income : ℝ := 90000
def household_expenses : ℝ := 0.5 * monthly_income
def medicine_expenses : ℝ := 0.15 * monthly_income
def savings : ℝ := 9000

-- Define the proof statement
theorem percentage_income_spent_on_clothes :
  ∃ (clothes_expenses : ℝ),
    clothes_expenses = monthly_income - household_expenses - medicine_expenses - savings ∧
    (clothes_expenses / monthly_income) * 100 = 25 := 
sorry

end percentage_income_spent_on_clothes_l312_312799


namespace monotonicity_of_f_inequality_ln_l312_312914

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x - (x + a) / x

theorem monotonicity_of_f (a x : ℝ) (h : x > 0) : 
  (a ≥ 0 → (∀ x > 0, f x a > f (x-ε) a )) ∧
  (a < 0 → ((∀ x ∈ (0, -a), f x a < f (x-ε) a) ∧ 
            (∀ x ∈ (-a, +∞), f x a > f (x-ε) a))) :=
sorry

theorem inequality_ln (x : ℝ) (h : x > 0) : 
  (1 / (x + 1) < ln (x + 1) / x) ∧ (ln (x + 1) / x < 1) :=
sorry

end monotonicity_of_f_inequality_ln_l312_312914


namespace simplify_expression_l312_312160

-- Given condition: a and b are real numbers.
variables (a b : ℝ)

-- Statement: Prove that the given expression simplifies to the specified form.
theorem simplify_expression : 
  (a ≠ 0) → (b ≠ 0) → 
  (a^{-2} * b^{-1}) / (a^{-4} - b^{-3}) = (a^3 * b^2) / (b^3 - a^4) :=
by sorry

end simplify_expression_l312_312160


namespace smallest_positive_period_and_monotonic_increase_min_max_values_interval_l312_312921

noncomputable def f (x : ℝ) := sqrt 2 * cos (2 * x - π / 4)

theorem smallest_positive_period_and_monotonic_increase :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧
  (∀ k : ℤ, 
    (∀ x : ℝ, -3 * π / 8 + k * π ≤ x ∧ x ≤ π / 8 + k * π → 
      f (x - ε) ≤ f x ∧ f x ≤ f (x + ε) ∨ sorry)) := 
sorry

theorem min_max_values_interval :
  ∃ x_min x_max, 
    x_min ∈ [-π / 8, π / 2] ∧ x_max ∈ [-π / 8, π / 2] ∧
    f x_min = -1 ∧ f x_max = sqrt 2 ∧ 
    (∀ x ∈ [-π / 8, π / 2], f x_min ≤ f x ∧ f x ≤ f x_max) :=
sorry

end smallest_positive_period_and_monotonic_increase_min_max_values_interval_l312_312921


namespace exists_zero_density_set_cover_l312_312139

open Set Filter MeasureTheory

variable {A : ℕ → Set ℝ}

-- Conditions
def covers_almost_every_point_infinitely_often (A : ℕ → Set ℝ) : Prop :=
  ∀ᶠ x in cofinite, ∃ n, x ∈ A n

def zero_density (B : Set ℕ) : Prop :=
  tendsto (λ n, (B ∩ Ico 0 n).card / n) atTop (𝓝 0)

-- Theorem
theorem exists_zero_density_set_cover :
  covers_almost_every_point_infinitely_often A →
  ∃ B : Set ℕ, zero_density B ∧ covers_almost_every_point_infinitely_often (λ n, A (B.find_index (λ k, k = n))) :=
sorry

end exists_zero_density_set_cover_l312_312139


namespace national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l312_312718

-- Question 5
theorem national_currency_depreciation (term : String) : term = "Devaluation" := 
sorry

-- Question 6
theorem bond_annual_coupon_income 
  (purchase_price face_value annual_yield annual_coupon : ℝ) 
  (h_price : purchase_price = 900)
  (h_face : face_value = 1000)
  (h_yield : annual_yield = 0.15) 
  (h_coupon : annual_coupon = 135) : 
  annual_coupon = annual_yield * purchase_price := 
sorry

-- Question 7
theorem dividend_yield 
  (num_shares price_per_share total_dividends dividend_yield : ℝ)
  (h_shares : num_shares = 1000000)
  (h_price : price_per_share = 400)
  (h_dividends : total_dividends = 60000000)
  (h_yield : dividend_yield = 15) : 
  dividend_yield = (total_dividends / num_shares / price_per_share) * 100 :=
sorry

-- Question 8
theorem tax_deduction 
  (insurance_premium annual_salary tax_return : ℝ)
  (h_premium : insurance_premium = 120000)
  (h_salary : annual_salary = 110000)
  (h_return : tax_return = 14300) : 
  tax_return = 0.13 * min insurance_premium annual_salary := 
sorry

end national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l312_312718


namespace function_increasing_symmetry_about_point_l312_312444

def f (x : ℝ) : ℝ := 2^x / (2^x + 1)

theorem function_increasing : ∀ x y : ℝ, x < y → f(x) < f(y) := 
by sorry

theorem symmetry_about_point : ∀ x : ℝ, f(x) + f(-x) = 1 := 
by sorry

end function_increasing_symmetry_about_point_l312_312444


namespace count_divisors_36_l312_312479

def is_divisor (n d : Int) : Prop := d ≠ 0 ∧ ∃ k : Int, n = d * k

theorem count_divisors_36 : 
  (Finset.filter (λ d, is_divisor 36 d) (Finset.range 37)).card 
    + (Finset.filter (λ d, is_divisor 36 (-d)) (Finset.range 37)).card
  = 18 :=
sorry

end count_divisors_36_l312_312479


namespace sin_double_angle_l312_312028

theorem sin_double_angle (α : ℝ) (h1 : -π / 2 < α) (h2 : α < π / 2)
  (h3 : cos (α + π/2) = 3/5) : sin (2 * α) = -24 / 25 :=
sorry

end sin_double_angle_l312_312028


namespace largest_lcm_l312_312714

theorem largest_lcm :
  max (max (max (max (Nat.lcm 18 4) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 14)) (Nat.lcm 18 18) = 126 :=
by
  sorry

end largest_lcm_l312_312714


namespace ceiling_of_neg_3_7_l312_312368

theorem ceiling_of_neg_3_7 : Int.ceil (-3.7) = -3 := by
  sorry

end ceiling_of_neg_3_7_l312_312368


namespace solvable_torsion_group_is_finite_l312_312589

-- Let's define a group G that satisfies the conditions described
variables (G : Type*) [group G]

-- Defining the conditions
def is_solvable_group (G : Type*) [group G] : Prop :=
sorry -- The detailed definition is omitted for brevity

def is_torsion_group (G : Type*) [group G] : Prop :=
∀ g : G, ∃ n : ℕ, n > 0 ∧ g ^ n = 1

def abelian_subgroup_finitely_generated (G : Type*) [group G] : Prop :=
∀ H : subgroup G, is_abelian H → H.fg

-- The condition that G is a solvable torsion group with every Abelian subgroup finitely generated
variable [is_solvable_group G]
variable [is_torsion_group G]
variable [abelian_subgroup_finitely_generated G]

-- The theorem to be proven
theorem solvable_torsion_group_is_finite (G : Type*) [group G] [is_solvable_group G] [is_torsion_group G] [abelian_subgroup_finitely_generated G] : 
  finite G :=
sorry

end solvable_torsion_group_is_finite_l312_312589


namespace original_deadline_l312_312760

theorem original_deadline (initial_people : ℕ) (days_worked : ℕ) (work_done : ℝ)
  (additional_people : ℕ) (total_people : ℕ) (remaining_days : ℕ) (D : ℕ) :
  initial_people = 20 →
  days_worked = 25 →
  work_done = 0.4 →
  additional_people = 30 →
  total_people = initial_people + additional_people →
  (days_worked / work_done) * (1 - work_done) / (total_people / initial_people) = remaining_days →
  days_worked + remaining_days = D →
  D = 40 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  sorry
end

end original_deadline_l312_312760


namespace max_odd_terms_in_sequence_l312_312352

def largest_digit (n : ℕ) : ℕ :=
  n.digits.max

def next_term (n : ℕ) : ℕ :=
  n + largest_digit n

def odd (n : ℕ) : Prop :=
  n % 2 = 1

def sequence (a : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := next_term (sequence n)

def max_successive_odd_terms (n : ℕ) : ℕ :=
nat.find_greatest (λ k, ∀ i < k, odd (sequence n i)) 6

theorem max_odd_terms_in_sequence :
  ∀ a : ℕ, odd a → max_successive_odd_terms a = 5 := sorry

end max_odd_terms_in_sequence_l312_312352


namespace apples_in_box_l312_312225

theorem apples_in_box :
  (∀ (o p a : ℕ), 
    (o = 1 / 4 * 56) ∧ 
    (p = 1 / 2 * o) ∧ 
    (a = 5 * p) → 
    a = 35) :=
  by sorry

end apples_in_box_l312_312225


namespace integral_abs_eq_8_l312_312006

theorem integral_abs_eq_8 :
  ∫ x in -2..2, abs (x^2 - 2 * x) = 8 :=
by
  sorry

end integral_abs_eq_8_l312_312006


namespace ratio_smaller_triangle_to_trapezoid_area_l312_312302

theorem ratio_smaller_triangle_to_trapezoid_area (a b : ℕ) (sqrt_three : ℝ) 
  (h_a : a = 10) (h_b : b = 2) (h_sqrt_three : sqrt_three = Real.sqrt 3) :
  ( ( (sqrt_three / 4 * (b ^ 2)) / 
      ( (sqrt_three / 4 * (a ^ 2)) - 
         (sqrt_three / 4 * (b ^ 2)))) = 1 / 24 ) := 
by
  -- conditions from the problem
  have h1: a = 10 := by exact h_a
  have h2: b = 2 := by exact h_b
  have h3: sqrt_three = Real.sqrt 3 := by exact h_sqrt_three
  sorry

end ratio_smaller_triangle_to_trapezoid_area_l312_312302


namespace unique_number_l312_312378

theorem unique_number (a : ℕ) (h1 : 1 < a) 
  (h2 : ∀ p : ℕ, Prime p → p ∣ a^6 - 1 → p ∣ a^3 - 1 ∨ p ∣ a^2 - 1) : a = 2 :=
by
  sorry

end unique_number_l312_312378


namespace largest_divisor_of_462_and_231_l312_312713

def is_factor (a b : ℕ) : Prop := a ∣ b

def largest_common_divisor (a b c : ℕ) : Prop :=
  is_factor c a ∧ is_factor c b ∧ (∀ d, (is_factor d a ∧ is_factor d b) → d ≤ c)

theorem largest_divisor_of_462_and_231 :
  largest_common_divisor 462 231 231 :=
by
  sorry

end largest_divisor_of_462_and_231_l312_312713


namespace min_children_to_ensure_three_same_combination_l312_312635

def min_children : ℕ := 25

theorem min_children_to_ensure_three_same_combination :
  ∀ (children : ℕ) (purchase_combination : ℕ × ℕ → ℕ), 
  (∀ c, purchase_combination.all (λ (a, b), 3 * a + 5 * b ≤ 15) ∧ ∃ a b, a ≥ 1 ∨ b ≥ 1) 
  ∧ (children ≥ min_children) → 
  ∃ (c1 c2 c3 : ℕ × ℕ),
  c1 = c2 ∧ c2 = c3 ∧ c1 ≠ (0, 0) :=
sorry

end min_children_to_ensure_three_same_combination_l312_312635
