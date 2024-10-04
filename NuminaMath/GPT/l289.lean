import Mathlib
import Mathlib.Algebra.ArithmeticSeq
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Field.Power
import Mathlib.Algebra.Group.Power.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Circle
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Circle
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialDefinition
import Mathlib.Combinatorics.Graph.Tournament
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.Coeff
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean
import Mathlib.Probability.Basic
import Mathlib.Probability.ContinuousDistributions
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Probability_Mass_Function
import Mathlib.SetTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Real
import data.finset.basic
import data.rat.basic

namespace initial_conditions_l289_289135

noncomputable def compound_interest (P : ℕ) (r : ℕ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / (100 * n)) ^ (n * t)

theorem initial_conditions :
  let P := 5000
  let r1 := 7
  let r2 := 8.5
  let n := 4
  let t := 0.25
  let additional_deposit := 2000
  let withdrawal := 1000
  let amount_after_first_quarter := compound_interest P r1 n t
  let new_principal := amount_after_first_quarter + additional_deposit
  let amount_after_second_quarter := compound_interest new_principal r2 n t
  let final_amount := amount_after_second_quarter - withdrawal
  final_amount = 6239.06 :=
by 
  sorry

end initial_conditions_l289_289135


namespace sin_eq_half_of_sin_sub_cos_eq_one_l289_289579

theorem sin_eq_half_of_sin_sub_cos_eq_one (α : ℝ) (h : sin α - sqrt 3 * cos α = 1) : 
  sin (7 * π / 6 - 2 * α) = 1 / 2 :=
sorry

end sin_eq_half_of_sin_sub_cos_eq_one_l289_289579


namespace min_value_of_g_l289_289018

variable (x : ℝ) (φ : ℝ)

def f (x : ℝ) (φ : ℝ) : ℝ := sin (x + φ) + (√3) * cos (x + φ)

def g (x : ℝ) (φ : ℝ) : ℝ := cos (x + φ)

theorem min_value_of_g :
  (0 < φ ∧ φ < π) →
  (symm_shift_right_by f φ ((π / 2), 0) (π / 6)) →
  Inf (set.image (λ x, g x φ) (set.Icc (-π) (π / 6))) = -1 / 2 :=
sorry

end min_value_of_g_l289_289018


namespace harriet_ran_48_miles_l289_289575

def total_distance : ℕ := 195
def katarina_distance : ℕ := 51
def equal_distance (n : ℕ) : Prop := (total_distance - katarina_distance) = 3 * n
def harriet_distance : ℕ := 48

theorem harriet_ran_48_miles
  (total_eq : total_distance = 195)
  (kat_eq : katarina_distance = 51)
  (equal_dist_eq : equal_distance harriet_distance) :
  harriet_distance = 48 :=
by
  sorry

end harriet_ran_48_miles_l289_289575


namespace miles_driven_before_gas_stop_l289_289375

def total_distance : ℕ := 78
def distance_left : ℕ := 46

theorem miles_driven_before_gas_stop : total_distance - distance_left = 32 := by
  sorry

end miles_driven_before_gas_stop_l289_289375


namespace infinite_product_value_l289_289551

-- We define the infinite product
def infinite_product : ℝ :=
  ∏' n : ℕ, (3^((n+1) / 2^n))

-- The main theorem stating the value of the product
theorem infinite_product_value : infinite_product = 9 :=
  sorry

end infinite_product_value_l289_289551


namespace am_gm_inequality_l289_289691

theorem am_gm_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by 
  sorry

end am_gm_inequality_l289_289691


namespace distance_A_to_BC_l289_289941

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def coord_diff (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def dot_product (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y + p1.z * p2.z

def magnitude (p : Point) : ℝ :=
  sqrt (p.x^2 + p.y^2 + p.z^2)

def distance_from_point_to_line (A B C : Point) : ℝ :=
  let AB := coord_diff A B
  let BC := coord_diff B C
  let mag_AB := magnitude AB
  let mag_BC := magnitude BC
  let dot_AB_BC := dot_product AB BC
  let cos_theta := dot_AB_BC / (mag_AB * mag_BC)
  mag_AB * sqrt (1 - cos_theta^2)

theorem distance_A_to_BC :
  let A := Point.mk 0 0 2
  let B := Point.mk 1 0 2
  let C := Point.mk 0 2 0
  distance_from_point_to_line A B C = 2 * sqrt 2 / 3 := by
  sorry

end distance_A_to_BC_l289_289941


namespace product_series_l289_289923

theorem product_series : (∏ n in (Finset.range 99).map (λ k, k + 2), (1 - (1 / n : ℚ))) = (1 / 100) := by
  -- Mathematical statement translated into Lean 4
  sorry

end product_series_l289_289923


namespace ConeCannotHaveSquarePlanView_l289_289425

def PlanViewIsSquare (solid : Type) : Prop :=
  -- Placeholder to denote the property that the plan view of a solid is a square
  sorry

def IsCone (solid : Type) : Prop :=
  -- Placeholder to denote the property that the solid is a cone
  sorry

theorem ConeCannotHaveSquarePlanView (solid : Type) :
  (PlanViewIsSquare solid) → ¬ (IsCone solid) :=
sorry

end ConeCannotHaveSquarePlanView_l289_289425


namespace circle_radius_equivalence_l289_289050

def radius_of_circle_proof_problem : Prop :=
  ∃ (R : ℝ),
    let ADE_area := 1 + Real.sqrt 3 in
    let angle_COD := 60 in
    let angle_COD_rad := Real.pi / 3 in
    ADE_area = (R^2 * (Real.sqrt 3 + 1) / 4) /\
    60 * Real.pi / 180 = angle_COD_rad /\
    R = 2

theorem circle_radius_equivalence :
  radius_of_circle_proof_problem :=
begin
  -- proof goes here
  sorry
end

end circle_radius_equivalence_l289_289050


namespace no_such_sequence_exists_l289_289440

theorem no_such_sequence_exists (a : ℕ → ℝ) :
  (∀ i, 1 ≤ i ∧ i ≤ 13 → a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i, 1 ≤ i ∧ i ≤ 12 → a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by
  sorry

end no_such_sequence_exists_l289_289440


namespace washing_water_use_l289_289428

variable (gallons_collected : Nat)
variable (gallons_per_car : Nat)
variable (num_cars : Nat)
variable (less_gallons_plants : Nat)

-- Here are the conditions provided in the problem
def initial_gallons_collected := gallons_collected = 65
def water_used_per_car := gallons_per_car = 7
def number_of_cars := num_cars = 2
def less_gallons_for_plants := less_gallons_plants = 11

-- Calculate total water used for cars
def total_water_cars := num_cars * gallons_per_car
-- Calculate water used for plants
def water_plants := total_water_cars - less_gallons_plants
-- Calculate total used for cars and plants
def total_water_cars_plants := total_water_cars + water_plants
-- Calculate remaining water
def remaining_water := gallons_collected - total_water_cars_plants
-- Calculate water used to wash plates and clothes
def water_plates_clothes := remaining_water / 2

-- The theorem to prove the problem statement
theorem washing_water_use (hg : initial_gallons_collected) (hwc : water_used_per_car) (hnc : number_of_cars) (hlp : less_gallons_for_plants) :
  water_plates_clothes = 24 :=
by
  -- Given conditions from the problem
  unfold initial_gallons_collected water_used_per_car number_of_cars less_gallons_for_plants at hg hwc hnc hlp
  -- Definition lies outside the immediate scope of main code, thus accuracy over proof is ensured
  sorry

end washing_water_use_l289_289428


namespace linear_coefficient_l289_289410

def polynomial := (x : ℝ) -> x^2 - 2*x - 3

theorem linear_coefficient (x : ℝ) : (polynomial x) = x^2 - 2*x - 3 → -2 :=
by
  sorry

end linear_coefficient_l289_289410


namespace minimum_photos_l289_289284

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289284


namespace data_transmission_time_l289_289921
noncomputable theory

def time_to_send_data (blocks chunks_per_block chunks_per_second : ℕ) : ℝ :=
  let total_chunks := blocks * chunks_per_block
  let time_seconds := total_chunks / chunks_per_second
  let time_minutes := time_seconds / 60
  time_minutes / 60

theorem data_transmission_time :
  time_to_send_data 100 450 200 = 0.0625 := by
  sorry

end data_transmission_time_l289_289921


namespace cooking_ways_l289_289713

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem cooking_ways : comb 5 2 = 10 :=
  by
  sorry

end cooking_ways_l289_289713


namespace problem_equiv_l289_289131

-- Definitions to match the conditions
def is_monomial (v : List ℤ) : Prop :=
  ∀ i ∈ v, True  -- Simplified; typically this would involve more specific definitions

def degree (e : String) : ℕ :=
  if e = "xy" then 2 else 0

noncomputable def coefficient (v : String) : ℤ :=
  if v = "m" then 1 else 0

-- Main fact to be proven
theorem problem_equiv :
  is_monomial [-3, 1, 5] :=
sorry

end problem_equiv_l289_289131


namespace fraction_unclaimed_stickers_l289_289879

theorem fraction_unclaimed_stickers
    (x : ℝ)
    (hx : 0 < x)
    (al_share := (4/9) * x)
    (bert_share := (3/10) * x)
    (carl_share := (2/9) * x) :
    1 - (al_share / x + bert_share / x + carl_share / x) = (1/30) := 
by 
  have h : al_share = (4/9) * x := rfl
  have h_bert : bert_share = (3/10) * x := rfl
  have h_carl : carl_share = (2/9) * x := rfl
  calc
    1 - ((al_share + bert_share + carl_share) / x)
        = 1 - (((4/9) * x + (3/10) * x + (2/9) * x) / x) : by rw [h, h_bert, h_carl]
    ... = 1 - ((4/9 + 3/10 + 2/9)) : by ring
    ... = 1 - (87/90) : by norm_num
    ... = 1/30 : by norm_num

end fraction_unclaimed_stickers_l289_289879


namespace age_of_sisters_l289_289783

theorem age_of_sisters (a b : ℕ) (h1 : 10 * a - 9 * b = 89) 
  (h2 : 10 = 10) : a = 17 ∧ b = 9 :=
by sorry

end age_of_sisters_l289_289783


namespace adjacent_angles_l289_289823

variable (θ : ℝ)

theorem adjacent_angles (h : θ + 3 * θ = 180) : θ = 45 ∧ 3 * θ = 135 :=
by 
  -- This is the place where the proof would go
  -- Here we only declare the statement, not the proof
  sorry

end adjacent_angles_l289_289823


namespace mark_cells_in_trominos_l289_289830

theorem mark_cells_in_trominos :
  ∃ f : ℕ × ℕ → bool,
  (∀ i j : ℕ, i < 2010 → j < 2010 →
    ∃ (k l m : ℕ), 
      k < 2010 ∧ l < 2010 ∧ m < 2010 ∧
      ((f (i, j) = tt ∨ f (k, l) = tt ∨ f (m, j) = tt) ∧
       (f (i + 1, j) = ℤ ∨ f (i + 1, l) = ℤ ∨ f (i + 1, m) = ℤ → f (i + 2, j) = f (i + 2, l) ∨ f (i + 2, m)))) ∧ 
    (∃ n : ℕ, ∀ i : ℕ, i < 2010 → (Σ j, f (i, j) = tt) = n) ∧
    (exists n : ℕ, ∀ j : ℕ, j < 2010 → (Σ i, f.def (i, j) = tt) = n) :=
sorry

end mark_cells_in_trominos_l289_289830


namespace minimum_photos_l289_289277

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289277


namespace vampire_daily_needed_people_l289_289873

-- Define the conditions as constants
def gallons_needed_per_week : ℕ := 7
def pints_per_person : ℕ := 2
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7

-- Define the proof statement
theorem vampire_daily_needed_people (gallons_needed_per_week = 7) 
                                    (pints_per_person = 2) 
                                    (pints_per_gallon = 8) 
                                    (days_per_week = 7) : 
                                    7 * 8 / 7 / 2 = 4 :=
by
    -- The proof is expected here
    sorry

end vampire_daily_needed_people_l289_289873


namespace largest_num_with_diff_digits_sum_17_l289_289804
-- Import the necessary library

-- Define the conditions and the answer
def digits_all_different (n : Nat) : Prop := 
  (n.toDigits).nodup

def digits_sum_17 (n : Nat) : Prop :=
  (n.toDigits).sum = 17

def largest_number := 6543210

-- The main statement to be proved
theorem largest_num_with_diff_digits_sum_17 : ∃ n, digits_all_different n ∧ digits_sum_17 n ∧ n = largest_number :=
by
  sorry

end largest_num_with_diff_digits_sum_17_l289_289804


namespace circum_ratio_gt_four_l289_289016

theorem circum_ratio_gt_four {ABC : Triangle} (h : abs (ABC.angleB - ABC.angleC) > 90) :
    ABC.circumradius > 4 * ABC.inradius :=
sorry

end circum_ratio_gt_four_l289_289016


namespace imaginary_part_of_complex_number_l289_289955

def complex_imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_complex_number : complex_imaginary_part (10 * (complex.I^2) + (real.sqrt 2) * (complex.I^3)) = -real.sqrt 2 :=
by
  sorry

end imaginary_part_of_complex_number_l289_289955


namespace monochromatic_triangle_of_large_points_l289_289194

open_locale big_operators

theorem monochromatic_triangle_of_large_points {N k : ℕ} (hN : N > nat.factorial k * real.exp 1) :
  (set.univ : set (fin N)).pairwise_disjoint (λ i j, (i ≠ j)) →
  (∀ i j : fin N, i ≠ j → true) →
  (∃ (i j l : fin N), i ≠ j ∧ j ≠ l ∧ i ≠ l ∧ true) :=
by
  sorry

end monochromatic_triangle_of_large_points_l289_289194


namespace total_dots_initially_marked_l289_289441

theorem total_dots_initially_marked 
  (dice_count : ℕ)
  (faces : Fin 6 → ℕ)
  (sum_opposite_faces : ∀ (i : Fin 3), faces i + faces ⟨5 - i, sorry⟩ = 7)
  (pair_glue_visible_faces : list (Fin 6))
  (dots_on_glued_faces : ℕ)
  (additional_faces_dots : ℕ) :
  dice_count = 7 ∧ 
  (∀ i, 1 ≤ faces i ∧ faces i ≤ 6) ∧
  dots_on_glued_faces = 54 ∧
  additional_faces_dots = 21 →
  ∑ i in (pair_glue_visible_faces), faces i = 75 :=
by
  sorry

end total_dots_initially_marked_l289_289441


namespace traveler_statement_false_l289_289768

structure Carpet where
  length : ℝ
  width : ℝ

def initial_large (c : Carpet) : Prop := c.length > 1 ∧ c.width > 1

def exchange1 (c : Carpet) : Carpet := {
  length := 1 / c.length, 
  width := 1 / c.width 
}

def exchange2 (c : Carpet) (x : ℝ) : Carpet × Carpet := ({
  length := x, 
  width := c.width 
}, {
  length := c.length / x, 
  width := c.width 
})

def equivalence (c1 c2 : Carpet) : Prop := 
  (c1.length = c2.length ∧ c1.width = c2.width) ∨ (c1.length = c2.width ∧ c1.width = c2.length)

def one_side_longer (c : Carpet) : Prop := 
  (c.length > 1 ∧ c.width < 1) ∨ (c.length < 1 ∧ c.width > 1)

theorem traveler_statement_false : 
  ∀ (initial : Carpet), initial_large initial →
  (∀ (transforms : list (Carpet → Carpet) × (Carpet → ℝ → Carpet × Carpet)), 
    transforms = ([exchange1], exchange2)) →
  (∀ (exchanged : list Carpet), 
    (∀ c ∈ exchanged, one_side_longer c) → false) :=
by
  intros
  sorry

end traveler_statement_false_l289_289768


namespace average_speed_l289_289340

variables (d1 d2 s1 s2 : ℝ) (t1 t2 d_total t_total : ℝ)
axiom d1_val : d1 = 180
axiom d2_val : d2 = 120
axiom s1_val : s1 = 60
axiom s2_val : s2 = 40
axiom t1_def : t1 = d1 / s1
axiom t2_def : t2 = d2 / s2
axiom d_total_def : d_total = d1 + d2
axiom t_total_def : t_total = t1 + t2

theorem average_speed : d_total / t_total = 50 := by
  rw [d1_val, d2_val, s1_val, s2_val, t1_def, t2_def, d_total_def, t_total_def]
  norm_num
  sorry

end average_speed_l289_289340


namespace smallest_a_l289_289424

theorem smallest_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 96 * a^2 = b^3) : a = 12 :=
by
  sorry

end smallest_a_l289_289424


namespace paint_two_faces_red_l289_289133

theorem paint_two_faces_red (f : Fin 8 → ℕ) (H : ∀ i, 1 ≤ f i ∧ f i ≤ 8) : 
  (∃ pair_count : ℕ, pair_count = 9 ∧
    ∀ i j, i < j → f i + f j ≤ 7 → true) :=
sorry

end paint_two_faces_red_l289_289133


namespace notebook_pen_cost_correct_l289_289754

noncomputable def notebook_pen_cost : Prop :=
  ∃ (x y : ℝ), 
  3 * x + 2 * y = 7.40 ∧ 
  2 * x + 5 * y = 9.75 ∧ 
  (x + 3 * y) = 5.53

theorem notebook_pen_cost_correct : notebook_pen_cost :=
sorry

end notebook_pen_cost_correct_l289_289754


namespace surface_area_of_solid_structure_l289_289925

-- Definition of the conditions
def num_cubes : ℕ := 15
def height : ℕ := 4
def length : ℕ := 5
def width : ℕ := 3

-- Statement of the proof problem
theorem surface_area_of_solid_structure : 
  (num_cubes = 15) → 
  (height = 4) → 
  (length = 5) → 
  (width = 3) → 
  ∃ surface_area : ℕ, surface_area = 84 :=
by
  sorry

end surface_area_of_solid_structure_l289_289925


namespace distance_of_intersections_l289_289978

theorem distance_of_intersections 
  (t : ℝ)
  (x := (2 - t) * (Real.sin (Real.pi / 6)))
  (y := (-1 + t) * (Real.sin (Real.pi / 6)))
  (curve : x = y)
  (circle : x^2 + y^2 = 8) :
  ∃ (B C : ℝ × ℝ), dist B C = Real.sqrt 30 := 
by
  sorry

end distance_of_intersections_l289_289978


namespace speed_conversion_l289_289837

-- Define the given condition
def kmph_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Speed in kmph
def speed_kmph : ℕ := 216

-- The proof statement
theorem speed_conversion : kmph_to_mps speed_kmph = 60 :=
by
  sorry

end speed_conversion_l289_289837


namespace distance_from_A_to_line_BC_l289_289945

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def distance_from_point_to_line (A B C : Point3D) : ℝ :=
  let AB := (B.x - A.x, B.y - A.y, B.z - A.z)
  let BC := (C.x - B.x, C.y - B.y, C.z - B.z)
  let magnitude := λ (v : ℝ × ℝ × ℝ), Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let dot_product := λ (v w : ℝ × ℝ × ℝ), v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let cos_theta := dot_product AB BC / ((magnitude AB) * (magnitude BC))
  magnitude AB * Real.sqrt (1 - cos_theta ^ 2)

theorem distance_from_A_to_line_BC (A B C : Point3D) (hA : A = ⟨0, 0, 2⟩) 
(hB : B = ⟨1, 0, 2⟩) (hC : C = ⟨0, 2, 0⟩) : 
distance_from_point_to_line A B C = 2 * Real.sqrt 2 / 3 := 
by
  sorry

end distance_from_A_to_line_BC_l289_289945


namespace equal_segments_IJ_JB_JC_l289_289689

-- Definitions and conditions
variable {A B C I J : Type}

-- Triangle ABC with I as the incenter and J as the intersection of AI with the circumcircle of ABC
axiom triangle_incenter_and_circumcircle_intersect
  (A B C I J : Type)
  [triangle A B C]
  (incenter : incenter_of A B C = I)
  (circumcircle_intersect : intersect_of AI (circumcircle A B C) = J)
  : true

-- The statement to be proven
theorem equal_segments_IJ_JB_JC
  (A B C I J : Type)
  [triangle A B C]
  (incenter : incenter_of A B C = I)
  (circumcircle_intersect : intersect_of AI (circumcircle A B C) = J)
  : segment_length J B = segment_length J C ∧ segment_length J C = segment_length J I ∧ segment_length J I = segment_length J B := by
  sorry

end equal_segments_IJ_JB_JC_l289_289689


namespace polynomial_subtraction_simplify_l289_289738

open Polynomial

noncomputable def p : Polynomial ℚ := 3 * X^2 + 9 * X - 5
noncomputable def q : Polynomial ℚ := 2 * X^2 + 3 * X - 10
noncomputable def result : Polynomial ℚ := X^2 + 6 * X + 5

theorem polynomial_subtraction_simplify : 
  p - q = result :=
by
  sorry

end polynomial_subtraction_simplify_l289_289738


namespace perimeters_equal_rectangle_area_correct_l289_289857

-- Define the lengths of the sides of the triangle.
def side1 : ℝ := 5
def side2 : ℝ := 7
def side3 : ℝ := 10

-- Define the width and length of the rectangle.
def width : ℝ := 11 / 3
def length : ℝ := 2 * (11 / 3)

-- Calculate the perimeter and area.
def triangle_perimeter : ℝ := side1 + side2 + side3
def rectangle_perimeter : ℝ := 2 * (width + length)
def rectangle_area : ℝ := width * length

-- Theorems to prove equality of perimeters and the area.
theorem perimeters_equal : triangle_perimeter = rectangle_perimeter := by sorry
theorem rectangle_area_correct : rectangle_area = 242 / 9 := by sorry

end perimeters_equal_rectangle_area_correct_l289_289857


namespace probability_valid_choices_l289_289682

def is_divisor (m n : ℕ) : Prop := ∃ k, n = m * k

def set_S : set ℕ := {d | d > 0 ∧ d ∣ (20 ^ 12)}

def valid_choices 
  (a1 a2 a3 : ℕ) : Prop := 
  a1 ∈ set_S ∧ 
  a2 ∈ set_S ∧ 
  a3 ∈ set_S ∧ 
  is_divisor a1 a2 ∧ 
  is_divisor a2 a3 ∧ 
  (a2 / a1 + a3 / a2 ≤ 800)

theorem probability_valid_choices :
  (∑ (a1 a2 a3 : ℕ) in set.univ.filter (λ a1 a2 a3, valid_choices a1 a2 a3), 1)
  / (∑ (a1 a2 a3 : ℕ) in set.univ, 1) = 1064100 / 34328125 :=
sorry

end probability_valid_choices_l289_289682


namespace triangle_area_of_tangent_line_l289_289012

theorem triangle_area_of_tangent_line : 
  let y (x : ℝ) := 3 * x * Real.log x + x
  let tangent_line_at (x₀ : ℝ) (y₀ : ℝ) :=
    let m := (deriv (λ x, y x)) x₀
    y₀ - m * x₀
  tangent_area : Real :=
    let x_intercept := 3 / 4
    let y_intercept := -3
    (1 / 2) * x_intercept * y_intercept
  in y 1 = 1 → 
  tangent_line_at 1 (y 1) = 4 * x - 3 →
  triangle_area = 9 / 8 :=
sorry

end triangle_area_of_tangent_line_l289_289012


namespace heat_capacity_at_100K_l289_289011

noncomputable def heat_capacity (t : ℝ) : ℝ :=
  0.1054 + 0.000004 * t

theorem heat_capacity_at_100K :
  heat_capacity 100 = 0.1058 := 
by
  sorry

end heat_capacity_at_100K_l289_289011


namespace benny_january_savings_l289_289142

theorem benny_january_savings :
  ∃ x : ℕ, x + x + 8 = 46 ∧ x = 19 :=
by
  sorry

end benny_january_savings_l289_289142


namespace part_a_part_b_l289_289816

-- Part (a)
theorem part_a (s : Finset ℕ) (h : s = Finset.range 556): 
  ∃ (a b c : Finset ℕ), a ∪ b ∪ c = s ∧ 
  a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅ ∧ 
  (∑ x in a, x) = (∑ x in b, x) = (∑ x in c, x) := 
  sorry

-- Part (b)
theorem part_b (squares : Finset ℕ) (h : squares = Finset.image (λ x, x^2) (Finset.range 82)): 
  ∃ (a b c : Finset ℕ), a ∪ b ∪ c = squares ∧ 
  a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅ ∧ 
  (∑ x in a, x) = (∑ x in b, x) = (∑ x in c, x) := 
  sorry

end part_a_part_b_l289_289816


namespace original_intensity_45_percent_l289_289746

variable (I : ℝ) -- Intensity of the original red paint in percentage.

-- Conditions
variable (h1 : 25 * 0.25 + 0.75 * I = 40) -- Given conditions about the intensities and the new solution.
variable (h2 : ∀ I : ℝ, 0.75 * I + 25 * 0.25 = 40) -- Rewriting the given condition to look specifically for I.

theorem original_intensity_45_percent (I : ℝ) (h1 : 25 * 0.25 + 0.75 * I = 40) : I = 45 := by
  -- We only need the statement. Proof is not required.
  sorry

end original_intensity_45_percent_l289_289746


namespace totalInterest_l289_289891

-- Definitions for the amounts and interest rates
def totalInvestment : ℝ := 22000
def investedAt18 : ℝ := 7000
def rate18 : ℝ := 0.18
def rate14 : ℝ := 0.14

-- Calculations as conditions
def interestFrom18 (p r : ℝ) : ℝ := p * r
def investedAt14 (total inv18 : ℝ) : ℝ := total - inv18
def interestFrom14 (p r : ℝ) : ℝ := p * r

-- Proof statement
theorem totalInterest : interestFrom18 investedAt18 rate18 + interestFrom14 (investedAt14 totalInvestment investedAt18) rate14 = 3360 :=
by
  sorry

end totalInterest_l289_289891


namespace beautiful_fold_through_F_l289_289374

noncomputable def beautiful_fold_probability := 
  let ABCD : set (ℝ × ℝ) := { p | (0 ≤ p.fst ∧ p.fst ≤ 1) ∧ (0 ≤ p.snd ∧ p.snd ≤ 1) } in
  let diagonals : set (ℝ × ℝ) := { p | p.fst = p.snd ∨ p.fst + p.snd = 1 } in
  calc
    _ = (measure_theory.measure (diagonals ∩ ABCD)) / 
          (measure_theory.measure ABCD) : sorry

theorem beautiful_fold_through_F :
  beautiful_fold_probability = 1/2 := sorry

end beautiful_fold_through_F_l289_289374


namespace awards_distribution_l289_289000

theorem awards_distribution :
  (∃ (f : Fin 6 → Fin 4), injective f ∧ ∀ i : Fin 4, ∃ a : Fin 6, f a = i) → 
  (finset.card {f : Function.Injective (Fin 6) (Fin 4) // ∀ i : Fin 4, ∃ a : Fin 6, f a = i} = 3720) :=
sorry

end awards_distribution_l289_289000


namespace find_initial_population_l289_289023

noncomputable def population_first_year (P : ℝ) : ℝ :=
  let P1 := 0.90 * P    -- population after 1st year
  let P2 := 0.99 * P    -- population after 2nd year
  let P3 := 0.891 * P   -- population after 3rd year
  P3

theorem find_initial_population (h : population_first_year P = 4455) : P = 4455 / 0.891 :=
by
  sorry

end find_initial_population_l289_289023


namespace find_a_l289_289318

theorem find_a (A B C : Type) [triangle A B C] 
  (angle_A : A = 60) (b : B = 8) (S_ABC : S_tr ABC = 12 * sqrt 3) : 
  ∃ a, a = 2 * sqrt 13 := by
  -- Definitions and conditions
  assume A B C : Type,
  assume [triangle A B C],
  assume angle_A : A = 60,
  assume b : B = 8,
  assume S_ABC : S_tr ABC = 12 * sqrt 3, 
  
  -- Proof to be filled
  sorry

end find_a_l289_289318


namespace flagpole_breaking_height_l289_289094

theorem flagpole_breaking_height (x : ℝ) (h_pos : 0 < x) (h_ineq : x < 6)
    (h_pythagoras : (x^2 + 2^2 = 6^2)) : x = Real.sqrt 10 :=
by sorry

end flagpole_breaking_height_l289_289094


namespace height_of_inscribed_cylinder_l289_289112

theorem height_of_inscribed_cylinder {R r : ℝ} (hR : R = 7) (hr : r = 3) :
  let h := 4 * Real.sqrt 10 in
  ∃ h_cylinder, h_cylinder = h :=
by
  let h := 4 * Real.sqrt 10
  have R_pos : 0 < R, by linarith
  have r_pos : 0 < r, by linarith
  use h
  sorry

end height_of_inscribed_cylinder_l289_289112


namespace smaller_circle_radius_l289_289265

theorem smaller_circle_radius
  (R : ℝ) (r : ℝ)
  (h1 : R = 12)
  (h2 : 7 = 7) -- This is trivial and just emphasizes the arrangement of seven congruent smaller circles
  (h3 : 4 * (2 * r) = 2 * R) : r = 3 := by
  sorry

end smaller_circle_radius_l289_289265


namespace find_distance_between_A_and_B_l289_289470

noncomputable def distance_between_cities (time_to_B A_to_B time_to_A B_to_A : ℝ) (reduced_time round_trip_speed: ℝ) : ℝ :=
  (round_trip_speed * reduced_time) / 2

theorem find_distance_between_A_and_B :
  let D := distance_between_cities 6 4.5 9.5 90 in D = 427.5 :=
by
  sorry

end find_distance_between_A_and_B_l289_289470


namespace maximize_angle_l289_289658

structure Point where
  x : ℝ
  y : ℝ

def A (a : ℝ) : Point := ⟨0, a⟩
def B (b : ℝ) : Point := ⟨0, b⟩

theorem maximize_angle
  (a b : ℝ)
  (h : a > b)
  (h₁ : b > 0)
  : ∃ (C : Point), C = ⟨Real.sqrt (a * b), 0⟩ :=
sorry

end maximize_angle_l289_289658


namespace apples_pie_calculation_l289_289669

-- Defining the conditions
def total_apples : ℕ := 34
def non_ripe_apples : ℕ := 6
def apples_per_pie : ℕ := 4 

-- Stating the problem
theorem apples_pie_calculation : (total_apples - non_ripe_apples) / apples_per_pie = 7 := by
  -- Proof would go here. For the structure of the task, we use sorry.
  sorry

end apples_pie_calculation_l289_289669


namespace pyramid_has_at_least_four_faces_l289_289109

def pyramid := sorry -- Placeholder for the definition of a pyramid

def triangular_pyramid_has_fewest_faces : Prop :=
  ∀ (P : pyramid), ∃ (F : ℕ), F ≥ 4

theorem pyramid_has_at_least_four_faces :
  ∀ (P : pyramid), 4 ≤ triangular_pyramid_has_fewest_faces P :=
  sorry

end pyramid_has_at_least_four_faces_l289_289109


namespace sum_probability_is_correct_l289_289084

noncomputable def student1_gen := (1 / 4) * Polynomial.C (2 ^ 2) + (1 / 2) * Polynomial.C (2 ^ 3) + (1 / 4) * Polynomial.C (2 ^ 4)
noncomputable def student2_gen := (1 / 2) * Polynomial.C (2 ^ 1) + (1 / 2) * Polynomial.C (2 ^ 2)
noncomputable def student3_gen := Polynomial.C (2 ^ 5)
noncomputable def student4_gen := (1 / 2) * Polynomial.C (2 ^ 4) + (1 / 2) * Polynomial.C (2 ^ 5)
noncomputable def student5_gen := (1 / 4) * Polynomial.C (2 ^ 1) + (1 / 2) * Polynomial.C (2 ^ 2) + (1 / 4) * Polynomial.C (2 ^ 3)
noncomputable def student6_gen := (1 / 4) * Polynomial.C (2 ^ 0) + (1 / 2) * Polynomial.C (2 ^ 1) + (1 / 4) * Polynomial.C (2 ^ 2)
noncomputable def student7_gen := (1 / 4) * Polynomial.C (2 ^ 3) + (1 / 2) * Polynomial.C (2 ^ 4) + (1 / 4) * Polynomial.C (2 ^ 5)

noncomputable def total_gen := student1_gen * student2_gen * student3_gen * student4_gen * student5_gen * student6_gen * student7_gen

theorem sum_probability_is_correct : (Polynomial.coeff total_gen 20) = (105 / 512) := by
  sorry

end sum_probability_is_correct_l289_289084


namespace fixed_point_of_function_l289_289761

theorem fixed_point_of_function (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ∃ x y : ℝ, (y = a^(2 * x - 1) + 1) ∧ x = 1/2 ∧ y = 2 :=
by
  use [1/2, 2]
  split
  · rw [←@Real.exp, Real.exp_eq]
    sorry
  · rfl

end fixed_point_of_function_l289_289761


namespace george_boxes_l289_289185

-- Define the problem conditions and the question's expected outcome.
def total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def expected_num_boxes : ℕ := 2

-- The proof statement that needs to be proved: George has the expected number of boxes.
theorem george_boxes : total_blocks / blocks_per_box = expected_num_boxes := 
  sorry

end george_boxes_l289_289185


namespace collinear_vectors_lambda_l289_289623

theorem collinear_vectors_lambda :
  ∀ (λ : ℝ), (∃ k : ℝ, k > 0 ∧ (2 - λ) = k * 2 ∧ (-4) = k * λ) → λ = -2 :=
by
  intros λ h
  cases h with k hk
  sorry

end collinear_vectors_lambda_l289_289623


namespace annual_interest_rate_l289_289853

-- Definitions based on conditions
def initial_amount : ℝ := 1000
def spent_amount : ℝ := 440
def final_amount : ℝ := 624

-- The main theorem
theorem annual_interest_rate (x : ℝ) : 
  (initial_amount * (1 + x) - spent_amount) * (1 + x) = final_amount →
  x = 0.04 :=
by
  intro h
  sorry

end annual_interest_rate_l289_289853


namespace range_of_f_l289_289220

-- Define the function f
def f (x : ℕ) : ℤ := x^2 - 2 * x

-- Define the domain
def domain : Finset ℕ := {0, 1, 2, 3}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 0, 3}

-- State the theorem
theorem range_of_f : (domain.image f) = expected_range := by
  sorry

end range_of_f_l289_289220


namespace locus_of_midpoints_KL_ge_PQ_l289_289434

-- Let's define our problem with given conditions and what needs to be proved.

theorem locus_of_midpoints (k l : Line) (A B P Q : Point) (PA PB KA LB KL : LineSegment) : 
  (k ⊥ l) →                      -- condition that lines k and l intersect at right angles
  (P ∈ k) →                      -- P lies on k
  (Q ∈ l) →                      -- Q lies on l
  (A ∈ k) →                      -- A lies on k
  (B ∈ l) →                      -- B lies on l
  (PA ∈ k) →                     -- line segment PA lies on k
  (PB ∈ l) →                     -- line segment PB lies on l
  (PQ ∠ 45° AB) →                -- PQ makes an angle of 45° with AB
  (midpoint_locus P Q = y = 0) := -- the locus of midpoints of PQ is y = 0
sorry

theorem KL_ge_PQ (k l : Line) (A B P Q K L : Point) (PA PB KA LB KL PQ : LineSegment) : 
  (k ⊥ l) →                      -- condition that lines k and l intersect at right angles
  (P ∈ k) →                      -- P lies on k
  (Q ∈ l) →                      -- Q lies on l
  (A ∈ k) →                      -- A lies on k
  (B ∈ l) →                      -- B lies on l
  (PA ∈ k) →                     -- line segment PA lies on k
  (PB ∈ l) →                     -- line segment PB lies on l
  (PQ ∠ 45° AB) →                -- PQ makes an angle of 45° with AB
  (perpendicular_bisector PQ = KL) → -- KL is the perpendicular bisector of PQ
  (KL.length ≥ PQ.length) :=     -- prove KL ≥ PQ
sorry

end locus_of_midpoints_KL_ge_PQ_l289_289434


namespace B_can_do_job_in_30_days_l289_289491

def work_rate_a : ℝ := 1 / 15
def work_rate_b (x : ℝ) : ℝ := 1 / x
def combined_work_rate (x : ℝ) : ℝ := work_rate_a + work_rate_b x
def work_done_in_4_days (x : ℝ) : ℝ := 4 * combined_work_rate x
def fraction_left_after_4_days := 0.6
def fraction_done := 1 - fraction_left_after_4_days

theorem B_can_do_job_in_30_days (x : ℝ) (h : work_done_in_4_days x = fraction_done) : x = 30 :=
by
  sorry

end B_can_do_job_in_30_days_l289_289491


namespace ratio_is_1_to_3_l289_289058

-- Definitions based on the conditions
def washed_on_wednesday : ℕ := 6
def washed_on_thursday : ℕ := 2 * washed_on_wednesday
def washed_on_friday : ℕ := washed_on_thursday / 2
def total_washed : ℕ := 26
def washed_on_saturday : ℕ := total_washed - washed_on_wednesday - washed_on_thursday - washed_on_friday

-- The ratio calculation
def ratio_saturday_to_wednesday : ℚ := washed_on_saturday / washed_on_wednesday

-- The theorem to prove
theorem ratio_is_1_to_3 : ratio_saturday_to_wednesday = 1 / 3 :=
by
  -- Insert proof here
  sorry

end ratio_is_1_to_3_l289_289058


namespace sum_s_h_e_base_three_l289_289236

def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧ S < 3 ∧ H < 3 ∧ E < 3 ∧ S ≠ H ∧ H ≠ E ∧ S ≠ E

def base_three_addition (S H E : ℕ) :=
  (S + H * 3 + E * 9) + (H + E * 3) == (H * 3 + S * 9 + S*27)

theorem sum_s_h_e_base_three (S H E : ℕ) (h1 : distinct_non_zero_digits S H E) (h2 : base_three_addition S H E) :
  (S + H + E = 5) := by sorry

end sum_s_h_e_base_three_l289_289236


namespace license_plate_count_l289_289096

/--
A license plate in a certain state consists of 4 digits, not necessarily distinct, and 2 letters, 
also not necessarily distinct. These six characters may appear in any order, except that the 
two letters must appear next to each other. 
-/
def distinct_license_plates : ℕ :=
  let digit_choices := 10^4 in
  let letter_choices := 26^2 in
  let positions_for_letters := 5 in
  positions_for_letters * digit_choices * letter_choices

theorem license_plate_count : distinct_license_plates = 33800000 := 
by 
  sorry

end license_plate_count_l289_289096


namespace inequality_proof_l289_289592

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1/2) :
  (1 - a + c) / (sqrt c * (sqrt a + 2 * sqrt b)) ≥ 2 :=
by sorry

end inequality_proof_l289_289592


namespace compare_production_values_minimum_average_cost_l289_289499

-- Definition of sequences with given conditions
def a_seq : ℕ → ℝ := λ n, a₁ + (n - 1) * d  -- Arithmetic sequence
def b_seq : ℕ → ℝ := λ n, b₁ * r ^ (n - 1)  -- Geometric sequence

-- Given conditions from the problem
variables (a₁ b₁ : ℝ) (d : ℝ) (r : ℝ) (h_a1_b1 : a₁ = b₁) (h_a13_b13 : a_seq 12 = b_seq 12)

-- Problem 1
theorem compare_production_values : a_seq 6 > b_seq 6 :=
by {
  sorry
}

-- Definition of average daily costs and conditions from the problem
def P (n : ℕ) := (32000000 + (∑ i in finRange n, (i + 49) / 10)) / n 

-- Problem 2
theorem minimum_average_cost : P 800 <= P n :=
by {
  sorry
}

end compare_production_values_minimum_average_cost_l289_289499


namespace problem_part1_problem_part2_l289_289536

-- Statement part (1)
theorem problem_part1 : ( (2 / 3) - (1 / 4) - (1 / 6) ) * 24 = 6 :=
sorry

-- Statement part (2)
theorem problem_part2 : (-2)^3 + (-9 + (-3)^2 * (1 / 3)) = -14 :=
sorry

end problem_part1_problem_part2_l289_289536


namespace volume_ratio_of_cubes_l289_289639

theorem volume_ratio_of_cubes (s2 : ℝ) : 
  let s1 := s2 * (Real.sqrt 3)
  let V1 := s1^3
  let V2 := s2^3
  V1 / V2 = 3 * (Real.sqrt 3) :=
by
  admit -- si



end volume_ratio_of_cubes_l289_289639


namespace min_photos_l289_289301

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289301


namespace andrew_start_age_l289_289136

-- Define the conditions
def annual_donation : ℕ := 7
def current_age : ℕ := 29
def total_donation : ℕ := 133

-- The theorem to prove
theorem andrew_start_age : (total_donation / annual_donation) = (current_age - 10) :=
by
  sorry

end andrew_start_age_l289_289136


namespace minimum_photos_taken_l289_289311

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289311


namespace jamie_peeled_potatoes_l289_289750

theorem jamie_peeled_potatoes :
  (∀ (total_potatoes sylvia_rate sylvia_time jamie_rate : ℕ), 
    total_potatoes = 60 → 
    sylvia_rate = 4 → 
    sylvia_time = 5 →
    jamie_rate = 6 →
    total_potatoes - sylvia_time * sylvia_rate = (total_potatoes - sylvia_time * sylvia_rate) →
    jamie_rate * (total_potatoes - sylvia_time * sylvia_rate) / (sylvia_rate + jamie_rate) = 4 * jamie_rate →
    jamie_rate * 4 = 24) := 
begin
  intros total_potatoes sylvia_rate sylvia_time jamie_rate h_total h_sylvia_rate h_sylvia_time h_jamie_rate,
  have sylvia_potatoes := sylvia_time * sylvia_rate,
  have remaining_potatoes := total_potatoes - sylvia_potatoes,
  have combined_rate := sylvia_rate + jamie_rate,
  have total_time := remaining_potatoes / combined_rate,
  have jamie_potatoes := jamie_rate * total_time,
  have h_jamie_potatoes := h_jamie_rate,
  rw h_jamie_potatoes,
  exact jamie_rate * total_time,
  sorry
end

end jamie_peeled_potatoes_l289_289750


namespace max_min_values_l289_289177

noncomputable def y (x : ℝ) : ℝ :=
  3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem max_min_values :
  (∀ k : ℤ, y (- (Real.pi/2) + 2 * k * Real.pi) = 7) ∧
  (∀ k : ℤ, y (Real.pi/6 + 2 * k * Real.pi) = -2) ∧
  (∀ k : ℤ, y (5 * Real.pi/6 + 2 * k * Real.pi) = -2) := by
  sorry

end max_min_values_l289_289177


namespace angle_subtended_by_shortest_side_l289_289747

theorem angle_subtended_by_shortest_side {A B C O : Type} 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (h_inscribed : ∃ (r : ℝ), CircleOfRadius r ∋ A ∧ CircleOfRadius r ∋ B ∧ CircleOfRadius r ∋ C)
  (h_sides : dist A B = 5 * sqrt 3 ∧ dist B C = 10 * sqrt 3 ∧ dist C A = 15 ∧
             is_center_of_circle_of_radius O (circumcircle A B C)) :
  angle_at_center_by_side AB O = 60 :=
sorry

end angle_subtended_by_shortest_side_l289_289747


namespace least_subtraction_to_div_by_10_l289_289062

theorem least_subtraction_to_div_by_10 :
  ∃ x, 427751 - x % 10 = 0 ∧ x = 1 :=
by
  have h : 427751 % 10 = 1 := sorry
  use 1
  split
  { rw [← nat.sub_mod, h]
    exact nat.zero_mod 10 }
  { refl }

end least_subtraction_to_div_by_10_l289_289062


namespace jam_fraction_left_l289_289334

theorem jam_fraction_left:
  let jam_total := 1 in
  let lunch_fraction := 1 / 3 in
  let after_lunch := jam_total - lunch_fraction in
  let dinner_fraction := 1 / 7 * after_lunch in
  let after_dinner := after_lunch - dinner_fraction in
  after_dinner = 4 / 7 := 
by {
  sorry
}

end jam_fraction_left_l289_289334


namespace fraction_area_below_line_square_l289_289422

theorem fraction_area_below_line_square : 
  let square := {P : ℝ × ℝ | 2 ≤ P.1 ∧ P.1 ≤ 5 ∧ 1 ≤ P.2 ∧ P.2 ≤ 4},
      line := {P : ℝ × ℝ | P.2 = (-2 / 3) * (P.1 - 2) + 3} in
  let area_square := (5 - 2) * (4 - 1),
      area_triangle := 1 / 2 * (5 - 2) * (3 - 1) in
  area_triangle / area_square = 1 / 3 := 
sorry

end fraction_area_below_line_square_l289_289422


namespace hundreds_digit_odd_of_n_squared_count_l289_289936

theorem hundreds_digit_odd_of_n_squared_count :
  (finset.filter (λ n : ℕ, (n^2 / 100) % 10 % 2 = 1) (finset.Icc 1 500)).card = 200 :=
by {
  sorry
}

end hundreds_digit_odd_of_n_squared_count_l289_289936


namespace angle_AMB_is_70_l289_289321

variables {A B C M : Type} [EuclideanGeometry A B C M]

open EuclideanGeometry

def isIsosceles (A B C : Type) [EuclideanGeometry A B C] :=
  dist A B = dist B C

def angleAtVertex (A B C : Type) [EuclideanGeometry A B C] (θ : ℝ) :=
  ∠ A B C = θ

theorem angle_AMB_is_70
  {A B C M : Type} [EuclideanGeometry A B C M]
  (isosceles_ABC : isIsosceles A B C)
  (angle_ABC_80 : angleAtVertex A B C (80 : ℝ))
  (angle_MAC_10 : ∠ M A C = 10)
  (angle_MCA_30 : ∠ M C A = 30) :
  ∠ A M B = 70 :=
by
  sorry

end angle_AMB_is_70_l289_289321


namespace sum_reciprocal_triangular_numbers_l289_289357

def triangular_number (n : ℕ) : ℚ := n * (n + 1) / 2

theorem sum_reciprocal_triangular_numbers :
  (∑ n in Finset.range 1001 + 1, 1 / triangular_number n) = 1001 / 501 :=
by
  sorry

end sum_reciprocal_triangular_numbers_l289_289357


namespace hyperbola_eccentricity_l289_289225

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (F : ℝ × ℝ)
  (hF : F = (2 * real.sqrt 2, 0))
  (A B : ℝ × ℝ)
  (hA : true) -- Placeholder for condition on A
  (hB : true) -- Placeholder for condition on B
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (hC : (F.1 - O.1) * (F.1 - O.1) + (F.2 - O.2) * (F.2 - O.2) = 8)
  (hArea : 4 = 4) -- Placeholder for area condition
  : real.sqrt 2 = abs (a / b) := 
sorry

end hyperbola_eccentricity_l289_289225


namespace music_festival_attendance_l289_289098

-- Definition of the conditions given in the problem
variables {x : ℝ}
variables (attendance_four_days : ℝ)
variables (first_day attendance_second attendance_third attendance_fourth : ℝ)

-- Conditions
def conditions :=
  attendance_four_days = 3600 ∧
  attendance_second = x / 2 ∧
  attendance_third = 3 * x ∧
  attendance_fourth = x

-- Proposition
theorem music_festival_attendance (x : ℝ) (attendance_four_days : ℝ)
  (attendance_second attendance_third attendance_fourth : ℝ) :
  conditions x attendance_four_days attendance_second attendance_third attendance_fourth →
  x + x / 2 + 3 * x + x = 3600 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  simp [h1, h2, h3, h4]
  sorry

end music_festival_attendance_l289_289098


namespace age_difference_l289_289478

theorem age_difference (J W : ℕ) (h1 : 0 < W ∧ W < 10) 
  (h2 : 10 * J = W + 21) :
  let john_age := 10 * J + W
  let wilson_age := W
  john_age - wilson_age = 30 :=
by
  let john_age := 10 * J + W
  let wilson_age := W
  have : (john_age + 21 = 2 * (wilson_age + 21)) := by linarith,
  sorry

end age_difference_l289_289478


namespace base3_to_base10_conversion_l289_289150

theorem base3_to_base10_conversion : 
  (1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3^1 + 1 * 3^0 = 100) :=
by 
  sorry

end base3_to_base10_conversion_l289_289150


namespace arrangement_6_people_l289_289489

theorem arrangement_6_people (A B : Type) : 
  (factorial 6) - 2 * (factorial 4) - 2 * (4 * (factorial 4)) = 504 :=
by
  sorry

end arrangement_6_people_l289_289489


namespace quadratic_roots_given_intersection_l289_289638

theorem quadratic_roots_given_intersection
  (a b c : ℝ)
  (h1 : a * 1^2 + b * 1 + c = 0)
  (h2 : a * 4^2 + b * 4 + c = 0) :
  (1 * a + b + c = 0) → (4 * a + b + c = 0) → ∀ (x : ℝ), (a * x^2 + b * x + c = 0) → (x = 1 ∨ x = 4) :=
by
  intros assumption1 assumption2 x h_x
  have root1 : a * 1 * 1 + b * 1 + c = 0 := h1
  have root2 : a * 4 * 4 + b * 4 + c = 0 := h2
  rw ← assumption1 at root1
  rw ← assumption2 at root2
  sorry

end quadratic_roots_given_intersection_l289_289638


namespace find_distance_from_A_to_line_BC_l289_289949

open Real EuclideanGeometry

noncomputable def distance_from_point_to_line (A B C : Point) : Real :=
  let AB := B - A
  let BC := C - B
  let cross_product := AB × BC -- This is a 3D vector cross product
  let magnitude_BC := |BC|
  |cross_product| / magnitude_BC

theorem find_distance_from_A_to_line_BC :
  let A := (0, 0, 2) : Point
  let B := (1, 0, 2) : Point
  let C := (0, 2, 0) : Point
  distance_from_point_to_line A B C = 2 * sqrt 2 / 3 :=
by
  sorry

end find_distance_from_A_to_line_BC_l289_289949


namespace tangent_line_equation_at_point_l289_289759

theorem tangent_line_equation_at_point 
  (x y : ℝ) (h_curve : y = x^3 - 2 * x) (h_point : (x, y) = (1, -1)) : 
  (x - y - 2 = 0) := 
sorry

end tangent_line_equation_at_point_l289_289759


namespace pure_imaginary_denom_rationalization_l289_289703

theorem pure_imaginary_denom_rationalization (a : ℝ) : 
  (∃ b : ℝ, 1 - a * Complex.I * Complex.I = b * Complex.I) → a = 0 :=
by
  sorry

end pure_imaginary_denom_rationalization_l289_289703


namespace wrong_observation_value_l289_289021

theorem wrong_observation_value :
  ∃ x : ℕ, let orig_mean := 41
             let num_obs := 50
             let sum_obs := num_obs * orig_mean
             let correct_obs := 48
             let new_mean := 41.5
             let new_sum := num_obs * new_mean
             let corrected_sum := sum_obs - x + correct_obs
           in orig_mean = 41 ∧ 
              num_obs = 50 ∧ 
              sum_obs = 2050 ∧ 
              correct_obs = 48 ∧ 
              new_mean = 41.5 ∧ 
              new_sum = 2075 ∧ 
              corrected_sum = new_sum ∧ 
              x = 23 := 
    begin
      use 23,
      unfold orig_mean num_obs sum_obs correct_obs new_mean new_sum corrected_sum,
      repeat { split }; 
      norm_num,
      sorry
    end

end wrong_observation_value_l289_289021


namespace triangle_area_of_ellipse_l289_289218

noncomputable def ellipse_area : ℝ :=
  let a : ℝ := 2
  let b : ℝ := sqrt 3
  let c : ℝ := sqrt (a * a - b * b)
  let F1F2 : ℝ := 2 * c
  let angle : ℝ := π / 3 -- 60 degrees in radians
  let PF1_length : ℝ := 4
  let PF1_PF2_length : ℝ := PF1_length / 2 * 2
  let sin_angle : ℝ := real.sin angle in
  1 / 2 * PF1_length * PF1_length / 2 * sin_angle

theorem triangle_area_of_ellipse :
  ∀ F1 F2 P : EuclideanGeometry.Point 2,
    ∃ a b : ℝ, 
    a = 2 ∧ b = sqrt 3 ∧ 
    c = sqrt (a * a - b * b) ∧
    Ellipse ((0, 0), 4, 3) P ∧
    F1F2 = 2 * c ∧ 
    angle = π / 3 ->
    area_of_triangle F1 F2 P = sqrt 3 :=
by
  sorry

end triangle_area_of_ellipse_l289_289218


namespace quadratic_inequality_real_roots_l289_289912

theorem quadratic_inequality_real_roots (c : ℝ) (h_pos : 0 < c) (h_ineq : c < 25) :
  ∃ x : ℝ, x^2 - 10 * x + c < 0 :=
sorry

end quadratic_inequality_real_roots_l289_289912


namespace shower_frequency_l289_289343

theorem shower_frequency
  (t_d: ℝ)   -- duration of each shower in minutes
  (w_r: ℝ)   -- water usage rate in gallons per minute
  (total_w: ℝ)  -- total water usage in 4 weeks
  (weeks: ℝ)   -- duration in weeks
  (showers_per_week: ℝ): showers_per_week = total_w / (t_d * w_r * weeks) →
    showers_per_week = 3.5 :=
begin
  intros h,
  rw h,
  sorry
end

end shower_frequency_l289_289343


namespace max_mass_sand_is_correct_l289_289504

-- Define the conditions
def platform_length : ℝ := 8
def platform_width : ℝ := 4
def max_angle : ℝ := Real.pi / 4  -- 45 degrees in radians
def sand_density : ℝ := 1500  -- kg/m³

-- Define the maximum mass calculation
def max_sand_mass (L W ρ : ℝ) : ℝ :=
  let h := W / 2
  let volume_prism := L * W * h
  let volume_pyramids := 2 * (1 / 3 * (W * (W / 2)) * h)
  let total_volume := volume_prism + volume_pyramids
  total_volume * ρ

-- Proof statement
theorem max_mass_sand_is_correct :
  max_sand_mass platform_length platform_width sand_density = 112000 := by
  sorry

end max_mass_sand_is_correct_l289_289504


namespace freshmen_and_sophomores_without_pet_l289_289036

theorem freshmen_and_sophomores_without_pet (total_students : ℕ) 
                                             (freshmen_sophomores_percent : ℕ)
                                             (pet_ownership_fraction : ℕ)
                                             (h_total : total_students = 400)
                                             (h_percent : freshmen_sophomores_percent = 50)
                                             (h_fraction : pet_ownership_fraction = 5) : 
                                             (total_students * freshmen_sophomores_percent / 100 - 
                                             total_students * freshmen_sophomores_percent / 100 / pet_ownership_fraction) = 160 :=
by
  sorry

end freshmen_and_sophomores_without_pet_l289_289036


namespace min_photographs_42_tourists_3_monuments_l289_289272

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289272


namespace parallel_lines_at_distance_perpendicular_lines_near_point_l289_289743

section LineEquations

  -- Definitions for the first problem
  def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
  
  -- Helper function to calculate the distance from a point to a line
  def distance_from_line (a b c x y : ℝ) : ℝ :=
    abs (a * x + b * y + c) / sqrt (a^2 + b^2)
    
  -- Define the new parallel lines at a distance of 1
  def line_parallel_1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0
  def line_parallel_2 (x y : ℝ) : Prop := 3 * x + 4 * y - 7 = 0
  
  -- Theorem for problem 1
  theorem parallel_lines_at_distance (x y : ℝ) :
    (line1 x y → distance_from_line 3 4 3 x y = 1 ∨ distance_from_line 3 4 (-7) x y = 1) :=
  sorry

  -- Definitions for the second problem
  def line2 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
  def point_P (x y : ℝ) : Prop := x = -1 ∧ y = 0
  
  -- Define the new perpendicular lines at a distance of √2 from point P
  def line_perpendicular_1 (x y : ℝ) : Prop := 3 * x - y + 9 = 0
  def line_perpendicular_2 (x y : ℝ) : Prop := 3 * x - y - 3 = 0
  
  -- Theorem for problem 2
  theorem perpendicular_lines_near_point (x y : ℝ) :
    (line2 x y ∧ point_P x y → distance_from_line 3 (-1) 9 x y = sqrt 2 ∨ distance_from_line 3 (-1) (-3) x y = sqrt 2) :=
  sorry

end LineEquations

end parallel_lines_at_distance_perpendicular_lines_near_point_l289_289743


namespace school_total_cost_l289_289026

theorem school_total_cost
  (num_sweaters : ℕ) (cost_per_sweater : ℕ)
  (num_sports_shirts : ℕ) (cost_per_sports_shirt : ℕ) :
  (num_sweaters * cost_per_sweater + num_sports_shirts * cost_per_sports_shirt) = 5400 :=
by
  have h1 : num_sweaters = 25 := rfl
  have h2 : cost_per_sweater = 98 := rfl
  have h3 : num_sports_shirts = 50 := rfl
  have h4 : cost_per_sports_shirt = 59 := rfl
  sorry -- The actual proof goes here, but is not required for the statement.

end school_total_cost_l289_289026


namespace sum_first_8_terms_l289_289201

def is_arithmetic_sequence (a : ℕ → ℤ) := ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) := ∑ i in Finset.range n, a i

theorem sum_first_8_terms (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h1 : is_arithmetic_sequence a)
    (h2 : S n = sum_first_n_terms a n)
    (h3 : a 2 = 18 - a 7) :
    S 8 = 72 := 
sorry

end sum_first_8_terms_l289_289201


namespace correct_statement_l289_289067

-- Definitions based on Conditions
def is_like_terms (a b : expr) : Prop := sorry -- Condition 1

def coeff (e : expr) : ℝ := sorry -- Condition 2

def degree (e : expr) : ℕ := sorry -- Condition 3

def polynomial_degree (poly : list expr) : ℕ :=
  list.foldr max 0 (list.map degree poly)

-- Expression for the given polynomial
def term1 := (2 : ℝ) * x ^ 2 * y
def term2 := (-3 : ℝ) * y ^ 2
def term3 := (-1 : ℝ)

def poly := [term1, term2, term3]

-- Proof statement
theorem correct_statement : polynomial_degree poly = 3 ∧ poly.length = 3 := by
  sorry

end correct_statement_l289_289067


namespace number_of_solutions_sine_exponential_l289_289566

theorem number_of_solutions_sine_exponential :
  let f := λ x => Real.sin x
  let g := λ x => (1 / 3) ^ x
  ∃ n, n = 150 ∧ ∀ k ∈ Set.Icc (0 : ℝ) (150 * Real.pi), f k = g k → (k : ℝ) ∈ {n : ℝ | n ∈ Set.Icc (0 : ℝ) (150 * Real.pi)} :=
sorry

end number_of_solutions_sine_exponential_l289_289566


namespace minimum_photos_taken_l289_289306

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289306


namespace trains_meet_at_9am_l289_289479

-- Definitions of conditions
def distance_AB : ℝ := 65
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def start_time_train_A : ℝ := 7
def start_time_train_B : ℝ := 8

-- This function calculates the meeting time of the two trains
noncomputable def meeting_time (distance_AB : ℝ) (speed_train_A : ℝ) (speed_train_B : ℝ) 
    (start_time_train_A : ℝ) (start_time_train_B : ℝ) : ℝ :=
  let distance_train_A := speed_train_A * (start_time_train_B - start_time_train_A)
  let remaining_distance := distance_AB - distance_train_A
  let relative_speed := speed_train_A + speed_train_B
  start_time_train_B + remaining_distance / relative_speed

-- Theorem stating the time when the two trains meet
theorem trains_meet_at_9am :
    meeting_time distance_AB speed_train_A speed_train_B start_time_train_A start_time_train_B = 9 := sorry

end trains_meet_at_9am_l289_289479


namespace right_angle_triangle_sets_l289_289526

def is_right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angle_triangle_sets :
  ¬ is_right_angle_triangle (2 / 3) 2 (5 / 4) :=
by {
  sorry
}

end right_angle_triangle_sets_l289_289526


namespace solve_inequality_system_l289_289745

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 2 < 3 * (x + 1)) ∧ (3x - 2) / 3 ≥ x + (x - 2) / 2 → x ≤ 2 / 3 :=
by
  sorry

end solve_inequality_system_l289_289745


namespace unique_intersection_point_l289_289900

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.log 3 / Real.log x)
noncomputable def h (x : ℝ) : ℝ := 3 - (1 / Real.sqrt (Real.log 3 / Real.log x))

theorem unique_intersection_point : (∃! (x : ℝ), (x > 0) ∧ (f x = g x ∨ f x = h x ∨ g x = h x)) :=
sorry

end unique_intersection_point_l289_289900


namespace seq_is_arith_compute_Sn_compute_an_bound_sum_l289_289584

noncomputable def S : ℕ → ℚ
| 0     => 0 -- Technically unnecessary but for total function
| (n+1) => if n = 0 then 1/2 else S (n+1) / (2 * (n+1) : ℚ)

noncomputable def a : ℕ → ℚ
| 0 => 0 -- Technically unnecessary but for total function
| (n+1) => if n = 0 then 1/2 else -2 * S (n+1) * S n

theorem seq_is_arith (n : ℕ) (h : n ≥ 1) : 
  ∃ d : ℚ, ∀ m : ℕ, m ≥ 1 → ((1 / S (m+1)) - (1 / S m) = d) := sorry

theorem compute_Sn (n : ℕ) (h : n ≥ 1) : S n = (1 / (2 * n : ℚ)) := sorry

theorem compute_an (n : ℕ) (h1 : n = 1 ∨ n ≥ 2) : 
  a n = if n = 1 then 1 / 2 else - 1 / (2 * n * (n - 1) : ℚ) := sorry

theorem bound_sum (n : ℕ) (h : n ≥ 1) :
  (∑ i in finset.range n, (S (i+1) ^ 2)) < (1 / 2) - (1 / (4 * n : ℚ)) := sorry

end seq_is_arith_compute_Sn_compute_an_bound_sum_l289_289584


namespace total_players_l289_289316

def num_teams : Nat := 35
def players_per_team : Nat := 23

theorem total_players :
  num_teams * players_per_team = 805 :=
by
  sorry

end total_players_l289_289316


namespace possible_measures_of_angle_X_l289_289767

theorem possible_measures_of_angle_X :
  ∃ (n : ℕ), n = 17 ∧ ∀ (X Y : ℕ), 
    (X > 0) → 
    (Y > 0) → 
    (∃ k : ℕ, k ≥ 1 ∧ X = k * Y) → 
    X + Y = 180 → 
    ∃ d : ℕ, d ∈ {d | d ∣ 180 } ∧ d ≥ 2 :=
by
  sorry

end possible_measures_of_angle_X_l289_289767


namespace trajectory_of_P_l289_289589

open Real

-- Definitions of points F1 and F2
def F1 : (ℝ × ℝ) := (-4, 0)
def F2 : (ℝ × ℝ) := (4, 0)

-- Definition of the condition on moving point P
def satisfies_condition (P : (ℝ × ℝ)) : Prop :=
  abs (dist P F2 - dist P F1) = 4

-- Definition of the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1 ∧ x ≤ -2

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, satisfies_condition P → ∃ x y : ℝ, P = (x, y) ∧ hyperbola_equation x y :=
by
  sorry

end trajectory_of_P_l289_289589


namespace turban_price_l289_289234

theorem turban_price (T : ℝ) (total_salary : ℝ) (received_salary : ℝ)
  (cond1 : total_salary = 90 + T)
  (cond2 : received_salary = 65 + T)
  (cond3 : received_salary = (3 / 4) * total_salary) :
  T = 10 :=
by
  sorry

end turban_price_l289_289234


namespace milkman_profit_calculation_l289_289471

def milkman_profit (total_milk pure_milk water_mixed cost_per_liter : ℕ) : ℕ :=
  let total_cost := total_milk * cost_per_liter in
  let mixture_vol := pure_milk + water_mixed in
  let cost_of_pure_milk_used := pure_milk * cost_per_liter in
  let revenue := mixture_vol * cost_per_liter in
  revenue - cost_of_pure_milk_used
  
theorem milkman_profit_calculation :
  milkman_profit 30 20 5 18 = 90 := by
    sorry

end milkman_profit_calculation_l289_289471


namespace equilateral_triangle_GHK_l289_289196

noncomputable theory

open Complex

def midpoint (z1 z2 : ℂ) : ℂ := (z1 + z2) / 2

theorem equilateral_triangle_GHK {r : ℝ} 
  (A B C D E F G H K : ℂ) 
  (h_inscribed : abs A = r ∧ abs B = r ∧ abs C = r ∧ abs D = r ∧ abs E = r ∧ abs F = r)
  (h_eq_sides : dist A B = r ∧ dist C D = r ∧ dist E F = r)
  (h_midpoints : G = midpoint B C ∧ H = midpoint D E ∧ K = midpoint F A) :
  dist G H = dist H K ∧ dist H K = dist K G :=
begin
  sorry
end

end equilateral_triangle_GHK_l289_289196


namespace pages_left_to_read_l289_289377

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l289_289377


namespace proof_angleA_proof_trig_identity_l289_289211

variable {a b c A B C : ℝ}
variable (ABC : ∀ ε > 0, ∃ x, x ≠ 0 ∧ abs ((a^2 + b^2 - c^2) / (2 * a * b)) < ε)

noncomputable def angleA_solution (h1: c = sqrt 3 * a * sin C - c * cos A) (h2 : A = π / 3) : Prop :=
  A = π / 3

theorem proof_angleA {a b c A B C : ℝ} 
  (h1: c = sqrt 3 * a * sin C - c * cos A)
  (h2 : A = π / 3) : angleA_solution h1 h2 :=
  by sorry

noncomputable def trigonometric_identity_proof : Prop :=
  ∀ a b A B, a ≠ 0 → b ≠ 0 → (cos (2 * A) / a^2 - cos (2 * B) / b^2 = 1 / a^2 - 1 / b^2)

theorem proof_trig_identity : trigonometric_identity_proof :=
  by sorry

end proof_angleA_proof_trig_identity_l289_289211


namespace pages_left_to_read_l289_289379

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l289_289379


namespace no_positive_ints_cube_l289_289627

theorem no_positive_ints_cube (n : ℕ) : ¬ ∃ y : ℕ, 3 * n^2 + 3 * n + 7 = y^3 := 
sorry

end no_positive_ints_cube_l289_289627


namespace shapes_can_be_divided_into_four_equal_parts_l289_289905

-- Definitions for the shapes and their properties
def Shape : Type := ℕ
def square : Shape := 1
def circle : Shape := 2
def triangle : Shape := 3
def rectangle : Shape := 4

-- Function to check if a shape is divided into 4 equal parts
def is_divided_into_four_equal_parts (s : Shape) : Prop :=
  s = square ∨ s = circle ∨ s = triangle ∨ s = rectangle

-- Theorem to prove that each shape can be divided into 4 equal parts
theorem shapes_can_be_divided_into_four_equal_parts
  (s : Shape) (h : s = square ∨ s = circle ∨ s = triangle ∨ s = rectangle) :
  is_divided_into_four_equal_parts s :=
by
  unfold is_divided_into_four_equal_parts
  exact h

end shapes_can_be_divided_into_four_equal_parts_l289_289905


namespace greatest_integer_c_not_in_range_l289_289457

theorem greatest_integer_c_not_in_range :
  ∃ c : ℤ, (¬ ∃ x : ℝ, x^2 + (c:ℝ)*x + 18 = -6) ∧ (∀ c' : ℤ, c' > c → (∃ x : ℝ, x^2 + (c':ℝ)*x + 18 = -6)) :=
sorry

end greatest_integer_c_not_in_range_l289_289457


namespace problem_statement_l289_289533

noncomputable def calculateValue (n : ℕ) : ℕ :=
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n)

theorem problem_statement : calculateValue 10 = 466 := by
  sorry

end problem_statement_l289_289533


namespace community_pantry_fraction_l289_289707

variables (annual_donation crisis_fund livelihood_fund contingency_fund community_pantry_donation : ℝ)

-- Conditions:
-- 1. Lyn donates $240 each year.
def donation := 240

-- 2. 1/2 of the donation goes to the local crisis fund.
def crisis_fund_fraction := 1 / 2
def crisis_fund := crisis_fund_fraction * donation

-- 3. 1/4 of the remaining after the local crisis fund goes to livelihood project funds.
def remaining_after_crisis := donation - crisis_fund
def livelihood_fund_fraction := 1 / 4
def livelihood_fund := livelihood_fund_fraction * remaining_after_crisis

-- 4. The rest of the remaining funds is designated for contingency funds.
def remaining_after_livelihood := remaining_after_crisis - livelihood_fund
-- 5. $30 goes to the contingency fund.
def contingency_fund := 30

-- Question: What fraction of the donation goes to the community pantry project?
def community_pantry_donation := remaining_after_livelihood - contingency_fund

theorem community_pantry_fraction :
  community_pantry_donation / donation = 1 / 4 :=
by sorry

end community_pantry_fraction_l289_289707


namespace number_of_yellow_carnations_at_greene_nursery_l289_289528

def greeneNursery (total_flowers red_roses white_roses yellow_carnations : ℕ) : Prop :=
  total_flowers = 6284 ∧
  red_roses = 1491 ∧ 
  white_roses = 1768 ∧
  yellow_carnations = total_flowers - (red_roses + white_roses)

theorem number_of_yellow_carnations_at_greene_nursery : 
  ∃ yellow_carnations : ℕ, greeneNursery 6284 1491 1768 yellow_carnations ∧ yellow_carnations = 3025 :=
begin
  have h : greeneNursery 6284 1491 1768 3025,
  { unfold greeneNursery,
    simp,
  },
  exact ⟨3025, h, rfl⟩,
end

end number_of_yellow_carnations_at_greene_nursery_l289_289528


namespace prize_purchasing_plans_l289_289836

theorem prize_purchasing_plans :
    { (x, y) : ℕ × ℕ // (4 * x + 3 * y = 48) ∧ (x > 0) ∧ (y > 0) }.to_finset.card = 3 := 
by
  sorry

end prize_purchasing_plans_l289_289836


namespace min_magnitude_eq_min_magnitude_value_l289_289186

def vector_a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 1 - t, t)
def vector_b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)

def vector_sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (b.1 - a.1, b.2 - a.2, b.3 - a.3)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem min_magnitude_eq : 
  ∀ (t : ℝ), magnitude (vector_sub (vector_a t) (vector_b t)) = 
    Real.sqrt (5 * t^2 - 2 * t + 2) :=
by
  sorry

theorem min_magnitude_value :
  ∃ (t : ℝ), (t = 1/5) ∧ magnitude (vector_sub (vector_a t) (vector_b t)) = 3 * Real.sqrt 5 / 5 :=
by
  sorry

end min_magnitude_eq_min_magnitude_value_l289_289186


namespace g_range_l289_289583

-- Define the quadratic function f and the conditions it satisfies
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Condition 1: f(x+1) - f(x) = 2x for all x in ℝ
lemma f_diff (x : ℝ) : f(x + 1) - f(x) = 2 * x := 
by sorry

-- Condition 2: f(0) = 1
lemma f_at_0 : f(0) = 1 := 
by sorry

-- Define the function g(x)
def g (x : ℝ) : ℝ := f(x) - 2 * x

-- Proving the range of function g over the interval [-1, 1]
theorem g_range : set.range (λ x, g x) ∩ set.Icc (-1) 5 = set.Icc (-1) 5 :=
by sorry

end g_range_l289_289583


namespace quadratic_inequality_solution_set_l289_289462

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) : 
  { x : ℝ | ax^2 + (a - 2)*x - 2 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 2/a } := 
by
  sorry

end quadratic_inequality_solution_set_l289_289462


namespace largest_num_with_diff_digits_sum_17_l289_289806
-- Import the necessary library

-- Define the conditions and the answer
def digits_all_different (n : Nat) : Prop := 
  (n.toDigits).nodup

def digits_sum_17 (n : Nat) : Prop :=
  (n.toDigits).sum = 17

def largest_number := 6543210

-- The main statement to be proved
theorem largest_num_with_diff_digits_sum_17 : ∃ n, digits_all_different n ∧ digits_sum_17 n ∧ n = largest_number :=
by
  sorry

end largest_num_with_diff_digits_sum_17_l289_289806


namespace weight_of_13_ingots_l289_289770

/-- There are 13 gold bars with weights C₁, C₂, ..., C₁₃. 
    Prove that the total weight of all 13 ingots can be determined using 8 weighings,
    given that the scales can measure the combined weight of any two gold bars. -/
theorem weight_of_13_ingots (C : Fin 13 → ℝ) 
  (w : Fin 8 → (Fin 13 × Fin 13)) (H : ∀ i, C (w i).1 + C (w i).2):
  ∃ total_weight : ℝ, total_weight = ∑ i in (Finset.fin_range 13), C i := by
  sorry

end weight_of_13_ingots_l289_289770


namespace part1_part2_l289_289586

variable {a : ℕ → ℝ}

-- Given conditions
def strictly_increasing (a : ℕ → ℝ) := ∀ n m, n < m → a n < a m
def unbounded (a : ℕ → ℝ) := ∀ M : ℝ, ∃ n : ℕ, a n > M

-- Questions restated in Lean 4:

-- Part 1
theorem part1 (h1 : strictly_increasing a) (h2 : unbounded a) :
  ∃ k0 : ℕ, ∀ k : ℕ, k ≥ k0 → (∑ i in Finset.range k, a i / a (i + 1)) < ↑k - 1 :=
by sorry

-- Part 2
theorem part2 (h1 : strictly_increasing a) (h2 : unbounded a) :
  ∃ k0 : ℕ, ∀ k : ℕ, k ≥ k0 → (∑ i in Finset.range k, a i / a (i + 1)) < ↑k - 1985 :=
by sorry

end part1_part2_l289_289586


namespace max_mass_sand_is_correct_l289_289505

-- Define the conditions
def platform_length : ℝ := 8
def platform_width : ℝ := 4
def max_angle : ℝ := Real.pi / 4  -- 45 degrees in radians
def sand_density : ℝ := 1500  -- kg/m³

-- Define the maximum mass calculation
def max_sand_mass (L W ρ : ℝ) : ℝ :=
  let h := W / 2
  let volume_prism := L * W * h
  let volume_pyramids := 2 * (1 / 3 * (W * (W / 2)) * h)
  let total_volume := volume_prism + volume_pyramids
  total_volume * ρ

-- Proof statement
theorem max_mass_sand_is_correct :
  max_sand_mass platform_length platform_width sand_density = 112000 := by
  sorry

end max_mass_sand_is_correct_l289_289505


namespace xiao_ming_min_correct_answers_l289_289452

theorem xiao_ming_min_correct_answers (x : ℕ) : (10 * x - 5 * (20 - x) > 100) → (x ≥ 14) := by
  sorry

end xiao_ming_min_correct_answers_l289_289452


namespace domain_of_f_eq_l289_289017

def domain_of_fractional_function : Set ℝ := 
  { x : ℝ | x > -1 }

theorem domain_of_f_eq : 
  ∀ x : ℝ, x ∈ domain_of_fractional_function ↔ x > -1 :=
by
  sorry -- Proof this part in Lean 4. The domain of f(x) is (-1, +∞)

end domain_of_f_eq_l289_289017


namespace freshmen_and_sophomores_without_pets_l289_289037

theorem freshmen_and_sophomores_without_pets :
  let total_students := 400
  let freshmen_sophomores_fraction := 0.50
  let pet_owners_fraction := 1 / 5
  let freshmen_sophomores := total_students * freshmen_sophomores_fraction
  let pet_owners := freshmen_sophomores * pet_owners_fraction
  let non_pet_owners := freshmen_sophomores - pet_owners
  non_pet_owners = 160 :=
by
  Sorry

end freshmen_and_sophomores_without_pets_l289_289037


namespace complex_conjugate_of_fraction_l289_289015

-- Define the conjugate of a complex number
noncomputable def conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Define the given complex numbers
def z1 : ℂ := 1 + I
def z2 : ℂ := 1 - 2 * I
def z := z1 / z2

-- State the theorem
theorem complex_conjugate_of_fraction : 
  conjugate (z) = - (1 / 5) - (3 / 5) * I := 
by 
  sorry

end complex_conjugate_of_fraction_l289_289015


namespace problem1_correct_problem2_correct_l289_289145

noncomputable def problem1 : ℝ :=
  (9 / 4) ^ (3 / 2) + (1 / 5) ^ (-2) - real.pi ^ 0 + (1 / 27) ^ (-1 / 3)

theorem problem1_correct : problem1 = 243 / 8 :=
  sorry

noncomputable def problem2 : ℝ :=
  real.logb 3 (9 * 27 ^ 2) + real.logb 2 6 - real.logb 2 3 + real.logb 4 3 * real.logb 3 16

theorem problem2_correct : problem2 = 11 :=
  sorry

end problem1_correct_problem2_correct_l289_289145


namespace height_of_pile_of_pipes_l289_289933

theorem height_of_pile_of_pipes :
  ∀ (r : ℝ), (r = 5) →
  (h = 3 * r + (r * sqrt 3)) →
  h = 15 + 5 * sqrt 3 :=
by
  intros r hr hh
  sorry

end height_of_pile_of_pipes_l289_289933


namespace convex_hexagon_segment_length_l289_289092

open Real

theorem convex_hexagon_segment_length (A₁ A₂ A₃ A₄ A₅ A₆ : Point) (M₁ M₂ M₃ M₄ M₅ M₆ : Point) (ω : Circle) :
  (is_convex_hexagon A₁ A₂ A₃ A₄ A₅ A₆) →
  (circumscribed_circle ω A₁ A₂ A₃ A₄ A₅ A₆)
  (radius ω = 1) →
  (midpoint (segment A₁ A₂) = M₁) →
  (midpoint (segment A₂ A₃) = M₂) →
  (midpoint (segment A₃ A₄) = M₃) →
  (midpoint (segment A₄ A₅) = M₄) →
  (midpoint (segment A₅ A₆) = M₅) →
  (midpoint (segment A₆ A₁) = M₆) →
  ∃ r, r = sqrt 3 ∧ (segment_length (segment M₁ M₄) ≥ r ∨ segment_length (segment M₂ M₅) ≥ r ∨ segment_length (segment M₃ M₆) ≥ r) :=
sorry

end convex_hexagon_segment_length_l289_289092


namespace Ron_book_picking_times_l289_289394

theorem Ron_book_picking_times (couples members : ℕ) (weeks people : ℕ) (Ron wife picks_per_year : ℕ) 
  (h1 : couples = 3) 
  (h2 : members = 5) 
  (h3 : Ron = 1) 
  (h4 : wife = 1) 
  (h5 : weeks = 52) 
  (h6 : people = 2 * couples + members + Ron + wife) 
  (h7 : picks_per_year = weeks / people) 
  : picks_per_year = 4 :=
by
  -- Definition steps can be added here if needed, currently immediate from conditions h1 to h7
  sorry

end Ron_book_picking_times_l289_289394


namespace irrational_lattice_point_exists_l289_289226

theorem irrational_lattice_point_exists (k : ℝ) (h_irrational : ¬ ∃ q r : ℚ, q / r = k)
  (ε : ℝ) (h_pos : ε > 0) : ∃ m n : ℤ, |m * k - n| < ε :=
by
  sorry

end irrational_lattice_point_exists_l289_289226


namespace triangle_perimeter_l289_289159

def distance (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem triangle_perimeter :
  let A := (2, 3)
  let B := (2, -4)
  let C := (9, 1)
  distance A B + distance B C + distance C A = 7 + real.sqrt 74 + real.sqrt 53 :=
by
  sorry

end triangle_perimeter_l289_289159


namespace problem_BC_l289_289985

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ :=
  sin (2 * x + φ)

theorem problem_BC {φ : ℝ} (hφ1 : 0 < φ) (hφ2 : φ < π / 2)
  (h : ∀ x : ℝ, f φ (π / 12) = f φ (π / 4)) :
  (∀ x, -π / 12 < x ∧ x < π / 6 → 2 * x + φ ∈ (Ioo 0 (π / 2)) → monotone_on (f φ) (Ioo (-π / 12) (π / 6)))
  ∧ (f φ (-π / 12) = 0) :=
sorry

end problem_BC_l289_289985


namespace area_triangle_CEF_l289_289319

-- Definitions for points and triangle
variables (A B C E F : Type) [RealInnerProductSpace ℝ Type]
variables (triangle_ABC : Triangle A B C)
variables (is_midpoint_E : IsMidpoint E A C)
variables (is_midpoint_F : IsMidpoint F A B)
variables (area_ABC : area triangle_ABC = 24)

-- Statement of the theorem
theorem area_triangle_CEF :
  area (Triangle C E F) = 6 :=
sorry

end area_triangle_CEF_l289_289319


namespace infinite_sum_equals_two_thirds_l289_289543

theorem infinite_sum_equals_two_thirds :
  ∑' n : ℕ, (n ≠ 0) → (3 * n + 2) / (n * (n + 1) * (n + 2)) = 2 / 3 :=
by 
  sorry

end infinite_sum_equals_two_thirds_l289_289543


namespace find_PB_l289_289312

variables (A B C D P : Point)
variables (AB AC AD BC BD CD BP : ℝ)
variables (theta phi : ℝ)

-- Conditions
variables [convex_quadrilateral A B C D]
variables [is_perpendicular CD AB]
variables [is_perpendicular BC AD]
variables (CD_val : CD = 52)
variables (BC_val : BC = 34)
variables [is_perpendicular (line_through C) BD]
variables (AP_val : distance A P = 5)

-- The conclusion to prove:
theorem find_PB (h1 : convex_quadrilateral A B C D)
                (h2 : is_perpendicular CD AB)
                (h3 : is_perpendicular BC AD)
                (h4 : CD = 52)
                (h5 : BC = 34)
                (h6 : is_perpendicular (line_through C) BD)
                (h7 : distance A P = 5) :
    distance P B = 80 :=
sorry

end find_PB_l289_289312


namespace sum_of_mean_median_mode_l289_289063

def numbers : List ℕ := [1, 2, 0, 2, 1, 3, 0, 2, 1, 3]

def sorted_numbers : List ℕ := [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

def mean (lst : List ℕ) : ℕ := lst.sum / lst.length
def median (lst : List ℕ) : ℚ := (lst.get! (lst.length/2 - 1) + lst.get! (lst.length/2)) / 2
def mode (lst : List ℕ) : ℕ := lst.mode

theorem sum_of_mean_median_mode : mean sorted_numbers + median sorted_numbers + mode sorted_numbers = 5 := 
by
  sorry

end sum_of_mean_median_mode_l289_289063


namespace computation_problems_count_l289_289119

theorem computation_problems_count
    (C W : ℕ)
    (h1 : 3 * C + 5 * W = 110)
    (h2 : C + W = 30) :
    C = 20 :=
by
  sorry

end computation_problems_count_l289_289119


namespace trigonometric_identity_proof_l289_289702

noncomputable def a : ℝ := -35 / 6 * Real.pi

theorem trigonometric_identity_proof :
  (2 * Real.sin (Real.pi + a) * Real.cos (Real.pi - a) - Real.cos (Real.pi + a)) / 
  (1 + Real.sin a ^ 2 + Real.sin (Real.pi - a) - Real.cos (Real.pi + a) ^ 2) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_proof_l289_289702


namespace equal_sum_seq_example_l289_289908

def EqualSumSeq (a : ℕ → ℕ) (c : ℕ) : Prop := ∀ n, a n + a (n + 1) = c

theorem equal_sum_seq_example (a : ℕ → ℕ) 
  (h1 : EqualSumSeq a 5) 
  (h2 : a 1 = 2) : a 6 = 3 :=
by 
  sorry

end equal_sum_seq_example_l289_289908


namespace range_of_x0_l289_289605

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 2)^x else log 2 (x + 2)

theorem range_of_x0 (x0 : ℝ) (h : f x0 ≥ 2) : x0 ≤ -1 ∨ x0 ≥ 2 :=
sorry

end range_of_x0_l289_289605


namespace average_marks_l289_289116

theorem average_marks (P C M : ℝ) (h1 : P = 95) (h2 : (P + M) / 2 = 90) (h3 : (P + C) / 2 = 70) :
  (P + C + M) / 3 = 75 := 
by
  sorry

end average_marks_l289_289116


namespace initial_distance_between_walkers_l289_289086

theorem initial_distance_between_walkers (D : ℝ) 
  (start_time meet_time : ℝ) 
  (A_speed B_speed : ℝ) 
  (duration : ℝ) 
  (A : Type) (B : Type)
  (walks_towards_each_other : Π (A B : Type), Prop) :
  start_time = 18 ∧ meet_time = 23 ∧ A_speed = 6 ∧ B_speed = 4 ∧ duration = 5 ∧ walks_towards_each_other A B →
  D = 50 :=
by
  intros h_conditions,
  have h1 : duration = meet_time - start_time, from sorry,
  have h2 : duration * (A_speed + B_speed) = duration * 6 + duration * 4, from sorry,
  have h3 : D = 5 * (6 + 4), from sorry,
  exact sorry

end initial_distance_between_walkers_l289_289086


namespace monotonic_intervals_when_a_eq_1_range_of_a_for_minimum_value_range_of_a_for_inequality_l289_289611

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - (a + 2) * x + real.log x

-- Problem (1): Monotonic intervals when a = 1
theorem monotonic_intervals_when_a_eq_1 : 
  ∀ (x : ℝ), 
    (0 < x ∧ x < 1/2) ∨ 
    (1 < x ∧ x < ∞) ∨ 
    (1/2 < x ∧ x < 1) → 
    (deriv (λ x, f 1 x) x > 0) := 
sorry 

-- Problem (2): Range of a for f(x) minimum value when a > 0 and f(1) = -2
theorem range_of_a_for_minimum_value : 
  ∀ (a : ℝ), a > 0 → 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ real.exp 1 → f a x ≥ f a 1) → 
    1 ≤ a := 
sorry 

-- Problem (3): Range of a for f(x1) + 2 x1 < f(x2) + 2 x2 for x1 < x2 in (0, ∞)
theorem range_of_a_for_inequality : 
  ∀ (a : ℝ), 
    (∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) → 
    (0 < a ∧ a ≤ 8) := 
sorry

end monotonic_intervals_when_a_eq_1_range_of_a_for_minimum_value_range_of_a_for_inequality_l289_289611


namespace sin_sum_to_product_l289_289165

theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin_sum_to_product_l289_289165


namespace polar_coordinates_intersection_points_l289_289655

/-- Convert given parametric equations and circle equation to polar form,
and prove the intersection points in polar coordinates --/
theorem polar_coordinates_intersection_points :
  let line_parametric (t : ℝ) := (2 - real.sqrt 3 * t, t)
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let line_cartesian (x y : ℝ) := x + real.sqrt 3 * y - 2 = 0
  let polar_line (rho theta : ℝ) := rho * real.cos (theta - real.pi / 3) = 1
  let polar_circle (rho : ℝ) := rho = 2 in
  ∃ theta_1 theta_2,
    theta_1 ∈ [0, 2 * real.pi) ∧ theta_2 ∈ [0, 2 * real.pi) ∧
    ((2, theta_1) ∨ (2, theta_2)) ∧
    (polar_line 2 theta_1 ∧ polar_line 2 theta_2) :=
  sorry

end polar_coordinates_intersection_points_l289_289655


namespace computation_problems_count_l289_289117

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end computation_problems_count_l289_289117


namespace range_of_a_correct_l289_289601

-- Define the functions
def f (x : ℝ) := Real.log (-x)
def g (a : ℝ) (x : ℝ) := Real.exp x - (Real.exp 1 - 1) * x - a

-- Define the range of a for which the functions are point symmetric about the y-axis
noncomputable def range_of_a := Set.Ici (1 : ℝ)

-- The main theorem statement
theorem range_of_a_correct (a : ℝ) :
  (∃ x : ℝ, f x = g a x ∧ f (-x) = g a (-x)) → a ∈ range_of_a :=
sorry

end range_of_a_correct_l289_289601


namespace black_squares_in_20th_row_l289_289852

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def squares_in_row (n : ℕ) : ℕ := 1 + sum_natural (n - 2)

noncomputable def black_squares_in_row (n : ℕ) : ℕ := 
  if squares_in_row n % 2 = 1 then (squares_in_row n - 1) / 2 else squares_in_row n / 2

theorem black_squares_in_20th_row : black_squares_in_row 20 = 85 := 
by
  sorry

end black_squares_in_20th_row_l289_289852


namespace probability_of_prime_sum_is_five_over_twelve_l289_289056

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

noncomputable def probability_prime_sum_dice : ℚ :=
  let outcomes := (Finset.product (Finset.range 6) (Finset.range 6)).filter (λ p, is_prime (p.1 + 1 + (p.2 + 1))) in
  outcomes.card / (6 * 6 : ℚ)

theorem probability_of_prime_sum_is_five_over_twelve :
  probability_prime_sum_dice = 5 / 12 :=
by
  sorry

end probability_of_prime_sum_is_five_over_twelve_l289_289056


namespace system_solutions_l289_289435

theorem system_solutions (x y : ℝ) : 
  (x - y) * (x^2 - y^2) = 160 ∧ (x + y) * (x^2 + y^2) = 580 ∧ (3, 7) = (x, y) →
  (7, 3) = (y, x) :=
by
  intro h
  cases h with h1 h2
  sorry

end system_solutions_l289_289435


namespace jan_uses_24_gallons_for_plates_and_clothes_l289_289430

theorem jan_uses_24_gallons_for_plates_and_clothes :
  (65 - (2 * 7 + (2 * 7 - 11))) / 2 = 24 :=
by sorry

end jan_uses_24_gallons_for_plates_and_clothes_l289_289430


namespace count_unique_sums_l289_289910

def a : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := a (n + 1) + a n + 1

theorem count_unique_sums :
  (Finset.card
    ((Finset.univ.product Finset.univ).filter
      (λ p : ℕ × ℕ, p.1 ≠ p.2 ∧ a p.1 + a p.2 ≤ 2004)).image (λ p, a p.1 + a p.2))
  = 120 := 
sorry

end count_unique_sums_l289_289910


namespace min_fraction_sum_l289_289696

theorem min_fraction_sum (b : Fin 10 → ℝ) (h₀ : ∀ i, 0 < b i) (h₁ : (∑ i, b i) = 2) :
  (∑ i, 1 / b i) ≥ 50 :=
sorry

end min_fraction_sum_l289_289696


namespace average_of_ages_is_correct_l289_289061

theorem average_of_ages_is_correct :
  let ages := [18, 27, 35, 46] in
  (list.sum ages : ℝ) / (list.length ages) = 31.5 :=
by
  let ages := [18, 27, 35, 46]
  have h_sum : (list.sum ages : ℝ) = 126, by norm_num
  have h_len : (list.length ages : ℝ) = 4, by norm_num
  rw [h_sum, h_len]
  norm_num

end average_of_ages_is_correct_l289_289061


namespace symmetry_center_coords_l289_289979

theorem symmetry_center_coords :
  (∀ k ∈ Int, (kπ / 2 + 5 * π / 12, 1)) →
  ({(-7 * π / 12, 1), (17 * π / 12, 1)} ⊆
    { (kπ / 2 + 5 * π / 12, 1) | k ∈ Int }) :=
by sorry

end symmetry_center_coords_l289_289979


namespace number_of_nonzero_terms_combined_polynomial_l289_289556

open Polynomial

-- Define the polynomials
def p1 := (Polynomial.C 2 * Polynomial.X + Polynomial.C 3) * (Polynomial.X ^ 2 + Polynomial.C 2 * Polynomial.X + Polynomial.C 4)
def p2 := Polynomial.C (-2) * (Polynomial.X ^ 3 + Polynomial.X ^ 2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1)
def p3 := (Polynomial.X - Polynomial.C 2) * (Polynomial.X + Polynomial.C 5)

-- Define the combined polynomial
def combined_polynomial := p1 + p2 + p3

-- Statement to prove that the number of nonzero terms in combined_polynomial is 2
theorem number_of_nonzero_terms_combined_polynomial : 
  (combined_polynomial.support.card = 2) :=
by sorry

end number_of_nonzero_terms_combined_polynomial_l289_289556


namespace distribute_coins_l289_289654

/-- The number of ways to distribute 25 identical coins among 4 schoolchildren -/
theorem distribute_coins :
  (Nat.choose 28 3) = 3276 :=
by
  sorry

end distribute_coins_l289_289654


namespace pages_left_to_read_l289_289378

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l289_289378


namespace range_of_a_l289_289083

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, x ≤ 4 → deriv (λ x, x^2 + 2*(a-1)*x + 2) x ≤ 0) → 
  a ≤ -3 :=
by
  intro h
  have h_deriv : deriv (λ x, x^2 + 2*(a-1)*x + 2) = λ x, 2*x + 2*(a-1)
  sorry

end range_of_a_l289_289083


namespace increasing_intervals_of_f_l289_289764

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 15 * x ^ 2 - 33 * x + 6

noncomputable def f_prime (x : ℝ) : ℝ := 3 * x ^ 2 - 30 * x - 33

theorem increasing_intervals_of_f :
  (∀ x, f_prime x > 0 → (x < -1 ∨ x > 11)) ∧
  (f_prime (-1) = 0) ∧
  (f_prime 11 = 0) →
  (∀ x, (-∞ < x ∧ x < -1) ∨ (11 < x ∧ x < ∞) → f_prime x > 0) :=
sorry

end increasing_intervals_of_f_l289_289764


namespace cube_distance_l289_289028

open Real

theorem cube_distance (A B C D : ℝ × ℝ × ℝ) (A1 B1 C1 D1 : ℝ × ℝ × ℝ) 
  (hA1 : A1 = (1, 0, 1)) (hB : B = (0, 0, 0)) (hC1 : C1 = (0, 1, 1)) (hD1 : D1 = (1, 1, 1)) : 
  let 
    d₁ := sqrt((1-0)^2 + (1-0)^2 + (1-0)^2),
    d₂ := sqrt((0-1)^2 + (1-0)^2 + (1-1)^2),
    d₃ := sqrt((1-0)^2 + (1-0)^2 + (1-0)^2),
    cross_prod := (1, 1, -2),
    A1_C1 := (-1, 1, 0), 
    BD1 := (1, 1, 1),
    A1_D1 := (0, 1, 0),
    dot_product := (A1_D1.1 * cross_prod.1 + A1_D1.2 * cross_prod.2 + A1_D1.3 * cross_prod.3),
    dist := abs(dot_product) / sqrt((cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2))
  in
    dist = sqrt(6) / 6:= 
begin
  sorry
end

end cube_distance_l289_289028


namespace minimum_towns_for_22_free_routes_l289_289437

-- Definition of the problem conditions and the proof goal
theorem minimum_towns_for_22_free_routes :
  ∃ n : ℕ, (∀ A B (towns : fin n → fin n → Prop) (routes_free : (fin n) → (fin n) → bool),
  ( (∀ x y : fin n, towns x y → x ≠ y) ∧
    (∀ x y : fin n, routes_free x y = true → towns x y ∨ towns y x) ∧ 
    (card {r : (fin n) × (fin n) // r.fst ≠ r.snd ∧ routes_free r.fst r.snd = true ∧ towns r.fst r.snd ∨ towns r.snd r.fst} = 22) )
   ) → n = 7 := sorry

end minimum_towns_for_22_free_routes_l289_289437


namespace section_area_proof_l289_289515

noncomputable def section_area (sphere_radius : ℝ) (plane_cut: ℝ) : ℝ := 
  let r := Real.sqrt (sphere_radius^2 - plane_cut^2)
  π * r^2

theorem section_area_proof :
  let sphere_radius := 1   -- sphere radius given car is tangent to cube faces of edge length 2
  let plane_cut := Real.sqrt (6) / 3 
  (section_area sphere_radius plane_cut) = π / 3 := 
by 
  sorry

end section_area_proof_l289_289515


namespace max_third_side_l289_289405

open Real

variables {A B C : ℝ} {a b c : ℝ} 

theorem max_third_side (h : cos (4 * A) + cos (4 * B) + cos (4 * C) = 1) 
                       (ha : a = 8) (hb : b = 15) : c = 17 :=
 by
  sorry 

end max_third_side_l289_289405


namespace prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l289_289784

-- Define events and their probabilities.
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8

-- Given P(A and B) = P(A) * P(B)
def prob_AB : ℝ := prob_A * prob_B

-- Statements to prove
theorem prob_both_hit : prob_AB = 0.64 :=
by
  -- P(A and B) = 0.8 * 0.8 = 0.64
  exact sorry

theorem prob_exactly_one_hit : (prob_A * (1 - prob_B) + (1 - prob_A) * prob_B) = 0.32 :=
by
  -- P(A and not B) + P(not A and B) = 0.8 * 0.2 + 0.2 * 0.8 = 0.32
  exact sorry

theorem prob_at_least_one_hit : (1 - (1 - prob_A) * (1 - prob_B)) = 0.96 :=
by
  -- 1 - P(not A and not B) = 1 - 0.04 = 0.96
  exact sorry

end prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l289_289784


namespace max_notebooks_l289_289249

theorem max_notebooks (price_per_notebook : ℕ) (total_amount : ℕ) (h1 : price_per_notebook = 45) (h2 : total_amount = 4050) :
  ∃ n : ℕ, n = 90 ∧ 45 * n ≤ 4050 := 
by
  use 90
  simp [h1, h2]
  sorry

end max_notebooks_l289_289249


namespace three_p_plus_two_q_l289_289782

noncomputable def num_prime_factors (n : ℕ) : ℕ :=
  (Nat.factorization n).values.sum

def check_log_eq (a b : ℕ) : Prop :=
  log 10 a + 2 * log 10 (Nat.gcd a b) = 12 ∧
  log 10 b + 2 * log 10 (Nat.lcm a b) = 42

theorem three_p_plus_two_q (a b p q : ℕ) (h1 : log 10 a + 2 * log 10 (Nat.gcd a b) = 12)
  (h2 : log 10 b + 2 * log 10 (Nat.lcm a b) = 42)
  (hp : p = num_prime_factors a) (hq : q = num_prime_factors b) : 3 * p + 2 * q = 80 :=
begin
  sorry
end

end three_p_plus_two_q_l289_289782


namespace infinite_congruent_partition_l289_289895

def partition (f : ℕ → ℕ) : Prop :=
  (∀ n, ∃ (m : ℕ), m ≥ n ∧
    ∃ (S : ℕ → set ℕ), 
      (∀ i : ℕ, S i ⊆ ℕ) ∧ 
      (∀ i j : ℕ, i ≠ j → S i ∩ S j = ∅) ∧ 
      (⋃ i, S i = set.univ))

theorem infinite_congruent_partition :
  ∃ f : ℕ → ℕ,
    (f 1 = 1) ∧
    (∀ k, ∀ y ∈ finset.range (2^k) \ (finset.range (2^(k-1))), 
        if (even k) then (f y = f (y - 2^(k-1))) 
        else (f y = f (y - 2^(k-1)) + 2^(k/2))) ∧
    partition f :=
begin
  sorry
end

end infinite_congruent_partition_l289_289895


namespace minimum_photos_l289_289279

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289279


namespace period1_period2_multiple_l289_289751

theorem period1_period2_multiple
  (students_period1 : ℕ)
  (students_period2 : ℕ)
  (h_students_period1 : students_period1 = 11)
  (h_students_period2 : students_period2 = 8)
  (M : ℕ)
  (h_condition : students_period1 = M * students_period2 - 5) :
  M = 2 :=
by
  sorry

end period1_period2_multiple_l289_289751


namespace current_speed_is_one_l289_289850

noncomputable def motorboat_rate_of_current (b h t : ℝ) : ℝ :=
  let eq1 := (b + 1 - h) * 4
  let eq2 := (b - 1 + t) * 6
  if eq1 = 24 ∧ eq2 = 24 then 1 else sorry

theorem current_speed_is_one (b h t : ℝ) : motorboat_rate_of_current b h t = 1 :=
by
  sorry

end current_speed_is_one_l289_289850


namespace carton_height_is_60_l289_289095

-- Definitions
def carton_length : ℕ := 30
def carton_width : ℕ := 42
def soap_length : ℕ := 7
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 360

-- Theorem Statement
theorem carton_height_is_60 (h : ℕ) (H : ∀ (layers : ℕ), layers = max_soap_boxes / ((carton_length / soap_length) * (carton_width / soap_width)) → h = layers * soap_height) : h = 60 :=
  sorry

end carton_height_is_60_l289_289095


namespace sin_function_unique_l289_289988

theorem sin_function_unique (ω : ℝ) (φ : ℝ) (hω_pos : ω > 0) (hφ_range : φ ∈ Ioo (-π/2) (π/2)) :
  (∀ x ∈ ℝ, sin (ω * x + φ) = sin (ω * (x + 2/ω) + φ)) ∧ (sin (ω * (1/3) + φ) = 1) →
  (ω = π ∧ φ = π/6) :=
by
  sorry

end sin_function_unique_l289_289988


namespace linear_arith_seq_l289_289597

variable {α : Type*}
variable (k b d : α)
          
def linear_function (f : α → α) : Prop :=
  ∃ k b, ∀ x, f(x) = k * x + b

def is_arith_seq (x : ℕ → α) : Prop :=
  ∃ d, ∀ n, x(n + 1) = x(n) + d

theorem linear_arith_seq (f : α → α) (x : ℕ → α) 
    (h_lin : linear_function f)
    (h_arith : is_arith_seq x) :
    is_arith_seq (λ n, f (x n)) :=
  sorry

end linear_arith_seq_l289_289597


namespace x_eight_equals_zero_l289_289926

-- Define the given condition
def condition (x : ℝ) : Prop := (Real.sqrt (Real.sqrt (1 - x^4)) + Real.sqrt (Real.sqrt (1 + x^4)) = 2)

-- Statement to prove
theorem x_eight_equals_zero (x : ℝ) (h : condition x) : x^8 = 0 := by
  sorry

end x_eight_equals_zero_l289_289926


namespace f_le_g_l289_289190

noncomputable def f (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ k, if k > 0 then (1 : ℚ) / (k ^ 3) else 0)

noncomputable def g (n : ℕ) : ℚ := 3 / 2 - (1 : ℚ) / (2 * n ^ 2)

theorem f_le_g (n : ℕ) (hn : n > 0) : f n ≤ g n :=
by sorry

end f_le_g_l289_289190


namespace planet_colonization_configurations_l289_289128

theorem planet_colonization_configurations :
  let total_planets := 12
  let earth_like_planets := 6
  let mars_like_planets := 6
  let earth_units := 3
  let mars_units := 1
  let total_units := 15
  -- number of Earth-like planets colonized
  exists (a b : ℕ), 3 * a + b = 15 ∧ 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 6 ∧
  -- calculate combinations for each valid (a,b) pair
  let combinations := (if a = 5 ∧ b = 0 then nat.choose earth_like_planets a * nat.choose mars_like_planets b
                      else if a = 4 ∧ b = 3 then nat.choose earth_like_planets a * nat.choose mars_like_planets b
                      else if a = 3 ∧ b = 6 then nat.choose earth_like_planets a * nat.choose mars_like_planets b
                      else 0) 
  -- sum of all valid combinations
  combinations + combinations + combinations = 326 := by
  sorry

end planet_colonization_configurations_l289_289128


namespace stella_paint_cans_l289_289404

theorem stella_paint_cans (initial_rooms : ℤ) (lost_cans : ℤ) (remaining_rooms : ℤ) (rooms_per_can : ℤ) :
  initial_rooms = 40 → lost_cans = 3 → remaining_rooms = 31 → rooms_per_can = 3 →
  ∃ cans_used : ℤ, cans_used = 11 ∧ cans_used * rooms_per_can ≥ remaining_rooms :=
by
  intro h1 h2 h3 h4
  use 11
  split
  · rfl
  · calc
      11 * 3 = 33 : by norm_num
      ... ≥ 31 : by norm_num
  sorry

end stella_paint_cans_l289_289404


namespace binomial_coeff_ratio_l289_289614

/-- The problem is to prove that the ratio of the binomial coefficients
  when a = 3/2 and a = 1/2 for k = 100 equals 3^100. -/

def generalized_binomial (a : ℚ) (k : ℕ) : ℚ :=
  (List.prod (List.iota k).map (λ i, a - i) / (Nat.factorial k))

theorem binomial_coeff_ratio :
  (generalized_binomial 3/2 100) / (generalized_binomial 1/2 100) = 3^100 :=
  sorry

end binomial_coeff_ratio_l289_289614


namespace kamal_english_marks_l289_289344

theorem kamal_english_marks 
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (total_marks : ℕ := average_marks * num_subjects)
  (sum_known_marks : ℕ := math_marks + physics_marks + chemistry_marks + biology_marks) :
  math_marks = 65 → 
  physics_marks = 82 → 
  chemistry_marks = 67 → 
  biology_marks = 85 → 
  average_marks = 75 → 
  num_subjects = 5 → 
  total_marks - sum_known_marks = 76 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  refl

end kamal_english_marks_l289_289344


namespace calculate_value_l289_289210

theorem calculate_value 
  (a : Int) (b : Int) (c : Real) (d : Real)
  (h1 : a = -1)
  (h2 : b = 2)
  (h3 : c * d = 1) :
  a + b - c * d = 0 := 
by
  sorry

end calculate_value_l289_289210


namespace number_of_ways_to_match_cups_and_lids_l289_289039

theorem number_of_ways_to_match_cups_and_lids : 
  let n := 5 in 
  let matches := 
    (1:ℕ) + 
    (nat.choose n 3 * 1) + 
    (nat.choose n 2 * 2) 
  in
  matches = 31 :=
by sorry

end number_of_ways_to_match_cups_and_lids_l289_289039


namespace largest_4_digit_div_by_5_smallest_primes_l289_289930

noncomputable def LCM_5_smallest_primes : ℕ := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))

theorem largest_4_digit_div_by_5_smallest_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 9240 := by
  sorry

end largest_4_digit_div_by_5_smallest_primes_l289_289930


namespace circle_eqn_l289_289827

noncomputable def P1 := (4 : ℝ, 9 : ℝ)
noncomputable def P2 := (6 : ℝ, 3 : ℝ)
noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
noncomputable def radius (M : ℝ × ℝ) (P : ℝ × ℝ) : ℝ := distance M P

theorem circle_eqn :
  let M := midpoint P1 P2 in
  let r := radius M P1 in
  (x y : ℝ) → (x - M.1)^2 + (y - M.2)^2 = r^2 → 
  (x - 5)^2 + (y - 6)^2 = 10 :=
sorry

end circle_eqn_l289_289827


namespace minimum_photos_l289_289294

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289294


namespace hammer_nail_cost_l289_289517

variable (h n : ℝ)

theorem hammer_nail_cost (h n : ℝ)
    (h1 : 4 * h + 5 * n = 10.45)
    (h2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 :=
sorry

end hammer_nail_cost_l289_289517


namespace max_value_g20_l289_289355

noncomputable def polynomial_real_nonnegative (f : ℝ → ℝ) :=
∀ x : ℝ, 0 ≤ f x

theorem max_value_g20 (g : ℝ → ℝ)
    (poly_g : ∀ (x : ℝ), g x = ∑ i in finset.range (n + 1), (b i) * x^i)
    (nonneg : polynomial_real_nonnegative g)
    (h10: g 10 = 100)
    (h30: g 30 = 2700) :
  g 20 ≤ 519.615 :=
sorry

end max_value_g20_l289_289355


namespace intersection_A_B_l289_289704

-- Conditions
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

-- Proof of the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l289_289704


namespace number_of_uncracked_seashells_l289_289454

theorem number_of_uncracked_seashells (toms_seashells freds_seashells cracked_seashells : ℕ) 
  (h_tom : toms_seashells = 15) 
  (h_fred : freds_seashells = 43) 
  (h_cracked : cracked_seashells = 29) : 
  toms_seashells + freds_seashells - cracked_seashells = 29 :=
by
  sorry

end number_of_uncracked_seashells_l289_289454


namespace total_cans_in_display_l289_289765

theorem total_cans_in_display :
  ∃ n S : ℕ,
  let a1 := 30,
      d := -4,
      an := 1 
  in 
  (an = a1 + (n-1) * d)
  ∧ (S = n * (a1 + an) / 2)
  ∧ (S = 128) :=
sorry

end total_cans_in_display_l289_289765


namespace loss_per_meter_calculation_l289_289866

/-- Define the given constants and parameters. --/
def total_meters : ℕ := 600
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 35

/-- Now we define the total cost price, total loss and loss per meter --/
def total_cost_price : ℕ := cost_price_per_meter * total_meters
def total_loss : ℕ := total_cost_price - selling_price
def loss_per_meter : ℕ := total_loss / total_meters

/-- State the theorem we need to prove. --/
theorem loss_per_meter_calculation : loss_per_meter = 5 :=
by
  sorry

end loss_per_meter_calculation_l289_289866


namespace mancino_garden_width_l289_289368

theorem mancino_garden_width :
  ∃ w : ℝ, (3 * 16 * w + 2 * 8 * 4 = 304) ∧ w = 5 :=
begin
  use 5,
  split,
  { calc
    3 * 16 * 5 + 2 * 8 * 4 = 240 + 64 : by norm_num
                     ... = 304 : by norm_num },
  { refl }
end

end mancino_garden_width_l289_289368


namespace problem1_problem2_problem3_l289_289588

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x - 4) = -f x)
variable (h_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ 2 → x ≤ y → y ≤ 2 → f x ≤ f y)

-- Problem statements
theorem problem1 : f 2012 = 0 := sorry

theorem problem2 : ∀ x, f (4 - x) = -f (4 + x) := sorry

theorem problem3 : f (-25) < f 80 ∧ f 80 < f 11 := sorry

end problem1_problem2_problem3_l289_289588


namespace infinite_product_root_form_l289_289548

theorem infinite_product_root_form :
  (∏ n, (3 ^ (n / 2^n))) = real.sqrt (9: ℝ) :=
sorry

end infinite_product_root_form_l289_289548


namespace minimum_photos_l289_289288

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289288


namespace find_m_l289_289657

theorem find_m (m : ℝ) : (∀ x y : ℝ, (x, y) = (m + 1, 2 * m - 4) → (x, y + 2) = (x, 0) → m = 1) :=
by
  intro x y h1 h2
  cases h2
  sorry

end find_m_l289_289657


namespace max_product_2015_l289_289345

theorem max_product_2015 : 
  let digits := [2, 0, 1, 5]
  ∃ a b c d : ℕ, 
  (a, b, c, d) ∈ digits.permutations ∧
  max (a * (100 * b + 10 * c + d)) -- (2 * 015, 20 * 15, etc.)
      (max ((10 * a + b) * (10 * c + d)) -- (21 * 50, 52 * 10, etc.)
           ((100 * a + 10 * b + c) * d)) -- (201 * 5, etc.)
  = 1050 :=
by
  sorry

end max_product_2015_l289_289345


namespace largest_number_with_unique_digits_summing_to_17_is_98_l289_289799

theorem largest_number_with_unique_digits_summing_to_17_is_98 :
  ∀ n : ℕ, (all_different n) → (digits_sum n = 17) → n = 98 :=
by
  sorry

def all_different (n : ℕ) : Prop :=
  let digits := n.to_digits
  digits.nodup

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.to_digits
  digits.sum

end largest_number_with_unique_digits_summing_to_17_is_98_l289_289799


namespace angle_in_second_quadrant_l289_289826

theorem angle_in_second_quadrant (n : ℤ) : (460 : ℝ) = 360 * n + 100 := by
  sorry

end angle_in_second_quadrant_l289_289826


namespace minimum_photos_l289_289291

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289291


namespace amplitude_of_cosine_l289_289990

theorem amplitude_of_cosine (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (max_val : a * cos b + d = 5) (min_val : -a * cos b + d = -3) :
  a = 4 := 
by
  sorry

end amplitude_of_cosine_l289_289990


namespace find_distance_from_A_to_line_BC_l289_289947

open Real EuclideanGeometry

noncomputable def distance_from_point_to_line (A B C : Point) : Real :=
  let AB := B - A
  let BC := C - B
  let cross_product := AB × BC -- This is a 3D vector cross product
  let magnitude_BC := |BC|
  |cross_product| / magnitude_BC

theorem find_distance_from_A_to_line_BC :
  let A := (0, 0, 2) : Point
  let B := (1, 0, 2) : Point
  let C := (0, 2, 0) : Point
  distance_from_point_to_line A B C = 2 * sqrt 2 / 3 :=
by
  sorry

end find_distance_from_A_to_line_BC_l289_289947


namespace trapezoid_proof_l289_289847

-- Define the trapezoid and its properties
structure Trapezoid (A B C D P O : Type*) :=
(AD_parallel_BC : AD ∥ BC)
(line_through_C_parallel_AB : line_through(C, P) ∥ AB)
(segment_DP_equal_OB : segment_DP = OB)
(O_diagonal_intersection : intersection_of(hypotenuse(AD,BC), O) = diagonals_intersect(A, B, C, D))

-- Define the necessary points and segments
variables (A B C D P O : Type*)

-- Define the hypothesis that given these conditions in the trapezoid,
-- prove that AD^2 = BC^2 + AD * BC.

theorem trapezoid_proof (h : Trapezoid A B C D P O) : 
  AD^2 = BC^2 + AD * BC :=
sorry

end trapezoid_proof_l289_289847


namespace stream_speed_l289_289490

-- Definitions based on conditions
def speed_in_still_water : ℝ := 5
def distance_downstream : ℝ := 100
def time_downstream : ℝ := 10

-- The required speed of the stream
def speed_of_stream (v : ℝ) : Prop :=
  distance_downstream = (speed_in_still_water + v) * time_downstream

-- Proof statement: the speed of the stream is 5 km/hr
theorem stream_speed : ∃ v, speed_of_stream v ∧ v = 5 := 
by
  use 5
  unfold speed_of_stream
  sorry

end stream_speed_l289_289490


namespace savannah_rolls_l289_289735

-- Definitions and conditions
def total_gifts := 12
def gifts_per_roll_1 := 3
def gifts_per_roll_2 := 5
def gifts_per_roll_3 := 4

-- Prove the number of rolls
theorem savannah_rolls :
  gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3 = total_gifts →
  3 + 5 + 4 = 12 →
  3 = total_gifts / (gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3) :=
by
  intros h1 h2
  sorry

end savannah_rolls_l289_289735


namespace average_temperature_bucyrus_l289_289451

theorem average_temperature_bucyrus : 
  let t1 := -14
  let t2 := -8
  let t3 := 1 
  (t1 + t2 + t3) / 3 = -7 := 
  by
  rw [show t1 + t2 + t3 = -21 by norm_num]
  rw [show -21 / 3 = -7 by norm_num]
  sorry

end average_temperature_bucyrus_l289_289451


namespace equiangular_star_inner_perimeter_l289_289494

noncomputable def perimeter_inner_pentagon (total_length : ℝ) (inner_angle : ℝ) : ℝ :=
  total_length - total_length / (1 + Real.sin inner_angle)

theorem equiangular_star_inner_perimeter :
  ∀ total_length : ℝ, total_length = 1 → 
  ∀ inner_angle : ℝ, inner_angle = Real.pi / 10 → 
  perimeter_inner_pentagon 1 (Real.pi / 10) = 1 - 1 / (1 + Real.sin (Real.pi / 10)) :=
by
  intros total_length h1 inner_angle h2
  rw [h1, h2]
  exact sorry

end equiangular_star_inner_perimeter_l289_289494


namespace num_rented_cars_at_3600_max_monthly_revenue_l289_289828

def total_cars : ℕ := 100
def initial_rent : ℕ := 3000
def rent_increment : ℕ := 50
def maintenance_rented : ℕ := 150
def maintenance_unrented : ℕ := 50

def rented_cars (rent : ℕ) : ℕ :=
  total_cars - ((rent - initial_rent) / rent_increment)

def monthly_revenue (rent : ℕ) : ℕ :=
  rent * rented_cars(rent) - (maintenance_rented * rented_cars(rent) + maintenance_unrented * (total_cars - rented_cars(rent)))

theorem num_rented_cars_at_3600 :
  rented_cars 3600 = 88 :=
sorry

theorem max_monthly_revenue :
  ∃ rent, rent = 4050 ∧ monthly_revenue rent = 307050 :=
sorry

end num_rented_cars_at_3600_max_monthly_revenue_l289_289828


namespace max_possible_M_l289_289078

noncomputable def max_M (A : Finset ℕ) (f : A → A) [bijection : Bijective f]
  (f_pow : ℕ → A → A) : ℕ :=
  ∃ M, 
    (∀ m, m < M → ∀ i, 1 ≤ i ∧ i ≤ 16 →
      (f_pow m f (i + 1) - f_pow m f i) % 17 ≠ 1 ∧ 
      (f_pow m f (i + 1) - f_pow m f i) % 17 ≠ 16) ∧ 
    ((f_pow m f 1 - f_pow m f 17) % 17 ≠ 1 ∧ 
    (f_pow m f 1 - f_pow m f 17) % 17 ≠ 16) ∧
    (∀ i, (1 ≤ i ∧ i ≤ 16) → 
      (f_pow M f (i + 1) - f_pow M f i) % 17 = 1 ∨ 
      (f_pow M f (i + 1) - f_pow M f i) % 17 = 16) ∧
    ((f_pow M f 1 - f_pow M f 17) % 17 = 1 ∨
    (f_pow M f 1 - f_pow M f 17) % 17 = 16)

theorem max_possible_M : max_M {1, 2, 3, ..., 17} f = 8 := 
sorry

end max_possible_M_l289_289078


namespace sum_of_radii_l289_289842

noncomputable def radius_sum : ℝ :=
  let r1 := 8 + 4 * Real.sqrt 3 in
  let r2 := 8 - 4 * Real.sqrt 3 in
  r1 + r2

theorem sum_of_radii :
  let C_coords := (fun r => (r, r)) in
  let tangent_condition (r : ℝ) := (r - 5)^2 + r^2 = (r + 3)^2 in
  (∀ r, tangent_condition r → r = 8 + 4 * Real.sqrt 3 ∨ r = 8 - 4 * Real.sqrt 3) →
  radius_sum = 16 :=
by
  sorry

end sum_of_radii_l289_289842


namespace range_of_a_if_f_has_max_l289_289245

-- Defining the function f
def f (x a : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * exp x else -2 * x - 1

-- Defining the hypothesis that f has a maximum value
def has_maximum_value (a : ℝ) : Prop :=
  ∃ M, ∀ x, f x a ≤ M

-- The theorem statement
theorem range_of_a_if_f_has_max : ∀ a : ℝ, has_maximum_value a → a ≥ - (1 / 2) - (1 / (2 * exp 2)) :=
by { sorry }

end range_of_a_if_f_has_max_l289_289245


namespace perfect_square_from_swapped_digits_l289_289939

def is_consecutive_digits (n : ℕ) : Prop :=
  n = 2134 ∨ n = 3245 ∨ n = 4356 ∨ n = 5467 ∨ n = 6578 ∨ n = 7689

def swap_first_two_digits (n : ℕ) : ℕ :=
  let d₁ := n / 1000,
      d₂ := (n % 1000) / 100,
      rest := n % 100 in
  d₂ * 1000 + d₁ * 100 + rest

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem perfect_square_from_swapped_digits :
  ∃ (n : ℕ), is_consecutive_digits n ∧ is_perfect_square (swap_first_two_digits n) ∧ n = 4356 :=
by
  sorry

end perfect_square_from_swapped_digits_l289_289939


namespace minimum_photos_l289_289293

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289293


namespace smallest_n_l289_289917

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3^n = k^4) (h2 : ∃ l : ℕ, 2^n = l^6) : n = 12 :=
by
  sorry

end smallest_n_l289_289917


namespace find_m_l289_289983

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log x + x^2 - 5 * x
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := m / x + 2 * x - 5

theorem find_m (m : ℝ) (h : f'(m, 1) = -1) : m = 2 :=
by 
  -- Given that f'(1) = m - 3 and it's known that it equals -1
  -- Therefore, m - 3 = -1, solving for m gives m = 2
  have : m - 3 = -1 := h
  linarith

end find_m_l289_289983


namespace conjugate_quadrant_l289_289603

open Complex

-- Define the given complex number z
noncomputable def z : ℂ := -2 * I + (3 - I) / I

-- State the proof problem as a hypothesis in Lean
theorem conjugate_quadrant (z := -2 * I + (3 - I) / I) :
  let z_conjugate := conj z in
  z_conjugate.re < 0 ∧ z_conjugate.im > 0 :=
by
  sorry  -- Proof goes here

end conjugate_quadrant_l289_289603


namespace sqrt_expr_eval_l289_289080

theorem sqrt_expr_eval : (Real.sqrt 72 / Real.sqrt 8 - | -2 |) = 1 := by
  sorry

end sqrt_expr_eval_l289_289080


namespace min_photos_l289_289300

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289300


namespace find_a1_l289_289957

-- Define the geometric sequence {a_n}
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the sum of the first n terms of the sequence
def sum_seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in range n, a i

-- Given conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ)
variable (q : ℝ)

-- Assuming the sequence {a_n} is geometric
axiom geom_seq_condition : geom_seq a q

-- Given sum conditions
axiom sum_condition1 : S (m - 2) = 1
axiom sum_condition2 : S m = 3
axiom sum_condition3 : S (m + 2) = 5

-- Define the goal: Prove that the first term a_1 = 1
theorem find_a1 : a 1 = 1 :=
  sorry

end find_a1_l289_289957


namespace average_temperature_brixton_l289_289755

theorem average_temperature_brixton (temps : List ℕ) (h : temps = [55, 68, 63, 60, 50]) :
  (List.sum temps : ℚ) / List.length temps = 59.2 := 
by {
  have temps_sum : List.sum temps = 55 + 68 + 63 + 60 + 50,
  { rw h },
  have temps_sum_calc : List.sum temps = 296,
  { rw [temps_sum], norm_num },
  
  have temps_length : List.length temps = 5,
  { rw h, norm_num },
  
  rw [temps_sum_calc, temps_length],
  norm_num,
}

end average_temperature_brixton_l289_289755


namespace number_of_spiders_l289_289070

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) 
  (h1 : total_legs = 40) (h2 : legs_per_spider = 8) : 
  (total_legs / legs_per_spider = 5) :=
by
  -- Placeholder for the actual proof
  sorry

end number_of_spiders_l289_289070


namespace speed_ratio_l289_289079

-- Define the problem conditions
variables (A B : ℝ) -- speeds of A and B
variables (d : ℝ) -- initial distance B from O (750 yards)

-- Define the equidistance conditions
theorem speed_ratio (A B : ℝ) :
  let d := 750 in
  (3 * A = abs (-d + 3 * B)) ∧ (9 * A = abs (-d + 9 * B)) → A / B = 2 / 7 :=
begin
  sorry
end

end speed_ratio_l289_289079


namespace SunshinePumpkinsCount_l289_289140

def MoonglowPumpkins := 14
def SunshinePumpkins := 3 * MoonglowPumpkins + 12

theorem SunshinePumpkinsCount : SunshinePumpkins = 54 :=
by
  -- proof goes here
  sorry

end SunshinePumpkinsCount_l289_289140


namespace distance_to_cheaper_gas_station_l289_289124

-- Define the conditions
def miles_per_gallon : ℕ := 3
def initial_gallons : ℕ := 12
def additional_gallons : ℕ := 18

-- Define the question and proof statement
theorem distance_to_cheaper_gas_station : 
  (initial_gallons + additional_gallons) * miles_per_gallon = 90 := by
  sorry

end distance_to_cheaper_gas_station_l289_289124


namespace sin_double_angle_l289_289213

noncomputable def alpha : ℝ := sorry

theorem sin_double_angle (h1 : alpha ∈ (π / 2, π)) (h2 : Real.sin alpha = sqrt 5 / 5) : 
    Real.sin (2 * alpha) = -4 / 5 := 
sorry

end sin_double_angle_l289_289213


namespace new_person_weight_l289_289818

theorem new_person_weight (avg_increase : ℝ) (num_people : ℕ) (initial_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 1.5 ∧ num_people = 6 ∧ initial_weight = 65 →
  new_weight = initial_weight + (num_people * avg_increase) →
  new_weight = 74 := by
  intro h1 h2
  cases h1 with ha1 h1
  cases h1 with ha2 ha3
  simp at h2
  rw [ha1, ha2, ha3] at h2
  exact h2

end new_person_weight_l289_289818


namespace odd_checkerboard_cannot_be_covered_by_dominoes_l289_289843

theorem odd_checkerboard_cannot_be_covered_by_dominoes 
    (m n : ℕ) (h : (m * n) % 2 = 1) :
    ¬ ∃ (dominos : Finset (Fin 2 × Fin 2)),
    ∀ {i j : Fin 2}, (i, j) ∈ dominos → 
    ((i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0)) ∧ 
    dominos.card = (m * n) / 2 := sorry

end odd_checkerboard_cannot_be_covered_by_dominoes_l289_289843


namespace purchase_price_l289_289734

theorem purchase_price (P : ℝ) (repair_cost trans_cost profit_percent : ℝ) (selling_price : ℝ) 
  (h1 : repair_cost = 5000)
  (h2 : trans_cost = 1000)
  (h3 : profit_percent = 0.5)
  (h4 : selling_price = 22500)
  (h5 : selling_price = (1 + profit_percent) * (P + repair_cost + trans_cost)) : P = 9000 :=
by
  have repair_cost_and_trans := h1 + h2
  have prof_ratio := 1 + profit_percent
  have p_plus_addition := P + (repair_cost + trans_cost)
  have prof_formula := (prof_ratio) * p_plus_addition
  have eqn := (1 + profit_percent) * (P + repair_cost + trans_cost) 
  exact sorry

end purchase_price_l289_289734


namespace trajectory_of_point_B_is_hyperbola_l289_289197

open Real

-- Define the given conditions
variables (l : Line) (α : Plane) (P : Point)

-- Hypotheses
-- Line l is parallel to plane α
axiom line_parallel_plane : l ∥ α

-- Point P is a fixed point on line l
axiom point_on_line : P ∈ l

-- Point B is a moving point in plane α
variable (B : Point)
axiom point_in_plane : B ∈ α

-- The line PB forms a 30 degree angle with line l
axiom angle_condition : angle (line_through P B) l = 30

-- Proof statement 
theorem trajectory_of_point_B_is_hyperbola : trajectory B = hyperbola :=
sorry

end trajectory_of_point_B_is_hyperbola_l289_289197


namespace angle_in_quadrant_IV_l289_289633

-- Definitions of quadrants for better readability
inductive Quadrant
| I
| II
| III
| IV

-- Problem Statement
theorem angle_in_quadrant_IV 
  (α : ℝ) 
  (h1 : cos (π - α) < 0) 
  (h2 : tan α < 0) : 
  Quadrant :=
begin
  -- The proof would go here.
  -- Solution steps have shown that α must lie in Quadrant IV.
  exact Quadrant.IV, 
end

end angle_in_quadrant_IV_l289_289633


namespace senior_discount_is_20_percent_l289_289537

-- Definitions based on conditions
def original_cost : ℝ := 7.50
def coupon_discount : ℝ := 2.50
def final_cost : ℝ := 4.00

-- Definition for computed values
def cost_after_coupon : ℝ := original_cost - coupon_discount

-- Senior discount amount calculation
def senior_discount_amount : ℝ := cost_after_coupon - final_cost

-- Senior discount percentage calculation
def senior_discount_percentage : ℝ := (senior_discount_amount / cost_after_coupon) * 100

-- Theorem to be proved
theorem senior_discount_is_20_percent : senior_discount_percentage = 20 := 
by 
  sorry

end senior_discount_is_20_percent_l289_289537


namespace intersection_A_B_l289_289966

def A : Set ℝ := { x | abs x < 3 }
def B : Set ℝ := { x | 2 - x > 0 }

theorem intersection_A_B : A ∩ B = { x : ℝ | -3 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l289_289966


namespace arithmetic_sequence_a5_l289_289661

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : 
  a 5 = 7 :=
by
  -- proof to be filled later
  sorry

end arithmetic_sequence_a5_l289_289661


namespace general_formula_find_n_l289_289964

-- Condition Definitions
def is_geometric_sequence {α : Type*} [DiscreteField α] (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- The given conditions:
def a1 : ℝ := 1 / 2
def a2 : ℝ := a1 * 1 / 2  -- since the common ratio q is 1/2

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i

-- Conditions
def cond1 (a : ℕ → ℝ) : Prop := is_geometric_sequence a (1 / 2)
def cond2 (a : ℕ → ℝ) : Prop := Sn a 1 = a1
def cond3 (a : ℕ → ℝ) (n : ℕ) : Prop := Sn a n = a1 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)
def cond4 (a : ℕ → ℝ) : Prop := 7 * (a 1) = 2 * (Sn a 3)

-- Part 1: Proving general formula a_n
theorem general_formula (a : ℕ → ℝ) (h1 : cond1 a) (h2 : cond2 a) (h3 : cond3 a 2) (h4 : cond4 a) :
  ∀ n, a n = (1 / 2)^n := sorry

-- Part 2: Proving n = 10 for the given series condition involving b_n
def bn (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ := log (2, 1 - S (n + 1))

def series_condition (b : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ k in finset.range n, 1 / (b (2 * k + 1) * b (2 * k + 3))) = 5 / 21

theorem find_n (a : ℕ → ℝ) (h1 : cond1 a) (h4 : cond4 a) (S : ℕ → ℝ) (b : ℕ → ℝ)
  (hb : ∀ n, b n = bn a S n) :
  series_condition b 10 := sorry

end general_formula_find_n_l289_289964


namespace average_temperature_bucyrus_l289_289450

theorem average_temperature_bucyrus : 
  let t1 := -14
  let t2 := -8
  let t3 := 1 
  (t1 + t2 + t3) / 3 = -7 := 
  by
  rw [show t1 + t2 + t3 = -21 by norm_num]
  rw [show -21 / 3 = -7 by norm_num]
  sorry

end average_temperature_bucyrus_l289_289450


namespace simplify_expression_l289_289740

variable {x : ℝ}

theorem simplify_expression (h1 : x ≠ -1) (h2 : x ≠ 1) :
  ( 
    ( ((x + 1)^3 * (x^2 - x + 1)^3) / (x^3 + 1)^3 )^3 *
    ( ((x - 1)^3 * (x^2 + x + 1)^3) / (x^3 - 1)^3 )^3 
  ) = 1 := by
  sorry

end simplify_expression_l289_289740


namespace over_budget_l289_289108

theorem over_budget (total_budget : ℕ) (months : ℕ) (spent : ℕ) (expected_months : ℕ) :
  spent = 6580 → total_budget = 12600 → months = 12 → expected_months = 6 →
  spent - ((total_budget / months) * expected_months) = 280 := by
  intros h_spent h_budget h_months h_expected_months
  rw [h_spent, h_budget, h_months, h_expected_months]
  sorry

end over_budget_l289_289108


namespace non_white_homes_without_fireplace_basement_garden_l289_289436

theorem non_white_homes_without_fireplace_basement_garden :
  ∀ (total_homes : ℕ) (white_fraction non_white_fireplace_fraction non_white_basement_fraction non_white_no_fireplace_garden_fraction : ℚ),
  total_homes = 400 →
  white_fraction = 1 / 4 →
  non_white_fireplace_fraction = 1 / 5 →
  non_white_basement_fraction = 1 / 3 →
  non_white_no_fireplace_garden_fraction = 1 / 2 →
  let white_homes := total_homes * white_fraction in
  let non_white_homes := total_homes - white_homes in
  let non_white_fireplace := non_white_homes * non_white_fireplace_fraction in
  let non_white_fireplace_and_basement := non_white_fireplace * non_white_basement_fraction in
  let non_white_no_fireplace := non_white_homes - non_white_fireplace in
  let non_white_no_fireplace_with_garden := non_white_no_fireplace * non_white_no_fireplace_garden_fraction in
  let non_white_no_fireplace_no_basement_no_garden := non_white_no_fireplace - non_white_no_fireplace_with_garden in
  non_white_no_fireplace_no_basement_no_garden = 120 :=
by
  intros
  sorry

end non_white_homes_without_fireplace_basement_garden_l289_289436


namespace sqrt_div_defined_l289_289914

theorem sqrt_div_defined (x : ℝ) : x ≥ 2 → ∃ y z : ℝ, y = sqrt (x - 2) ∧ z = sqrt (x - 1) :=
by
  sorry

end sqrt_div_defined_l289_289914


namespace sufficient_but_not_necessary_condition_l289_289188

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b > 1) :
  a - b < a^2 - b^2 ∧ (∀ a b, a - b < a^2 - b^2 → a > b > 1) = false :=
by
  sorry

end sufficient_but_not_necessary_condition_l289_289188


namespace problem_proof_l289_289585

noncomputable def a : ℕ → ℚ 
| 1   => 1/2
| (n + 1) => (n + a n) / 2

def b (n : ℕ) : ℚ := a (n + 1) - a n - 1

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ k, b (k + 1))

def is_arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) - s n = s 1

theorem problem_proof :
  a 2 = 3/4 ∧ a 3 = 11/8 ∧ a 4 = 35/16 ∧
  (∃ r, ∀ n, b (n + 1) = r * b n ∧ b 1 = -3/4 ∧ r = 1/2) ∧
  (∃ λ : ℚ, λ = 2 ∧ is_arithmetic_sequence (λ n, (S n + λ * T n) / n)) :=
by 
  sorry

end problem_proof_l289_289585


namespace integral_ln_two_coefficient_term_binomial_dist_variance_sequence_sum_value_l289_289485

-- Problem (1)
theorem integral_ln_two :
  ∫ (x : ℝ) in 1..2, 1 / x = Real.log 2 :=
by sorry

-- Problem (2)
def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 5

theorem coefficient_term :
  let coeff := Nat.choose 5 4 + Nat.choose 6 4 + Nat.choose 7 4
  ∃ (n : ℕ), coeff = arithmetic_sequence n ∧ n = 20 :=
by sorry

-- Problem (3)
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_dist_variance :
  ∃ (p : ℝ), p = 1 / 2 ∧ binomial_variance 4 p = 1 :=
by sorry

-- Problem (4)
def sequence_sum (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, Real.sqrt (1 + 1 / (k + 1)^2 + 1 / (k + 2)^2))

theorem sequence_sum_value :
  sequence_sum 2018 = 2018 :=
by sorry

end integral_ln_two_coefficient_term_binomial_dist_variance_sequence_sum_value_l289_289485


namespace distance_between_stations_l289_289057

-- Definitions based on conditions in step a):
def speed_train1 : ℝ := 20  -- speed of the first train in km/hr
def speed_train2 : ℝ := 25  -- speed of the second train in km/hr
def extra_distance : ℝ := 55  -- one train has traveled 55 km more

-- Definition of the proof problem
theorem distance_between_stations :
  ∃ D1 D2 T : ℝ, D1 = speed_train1 * T ∧ D2 = speed_train2 * T ∧ D2 = D1 + extra_distance ∧ D1 + D2 = 495 :=
by
  sorry

end distance_between_stations_l289_289057


namespace linear_function_third_quadrant_l289_289993

theorem linear_function_third_quadrant (a : ℝ) (h : a ≠ 0)
  (inv_prop : ∀ x : ℝ, x > 0 → y = a / x → y decreases as x increases) :
  ¬ (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = -a * x + a) :=
by
  sorry

end linear_function_third_quadrant_l289_289993


namespace discount_rate_on_pony_l289_289183

variable (F P : ℝ)
variable (price_Fox price_Pony discount_sum total_savings : ℝ)
variable (num_Fox num_Pony : ℕ)

-- Conditions
def fox_price : price_Fox = 15 := rfl
def pony_price : price_Pony = 18 := rfl
def discount_rate_sum : discount_sum = 22 := rfl
def total_savings_condition : total_savings = 8.64 := rfl
def num_Fox_jeans : num_Fox = 3 := rfl
def num_Pony_jeans : num_Pony = 2 := rfl

-- Define the discount equations
def discount_condition1 : F + P = discount_sum := by
  dsimp [discount_sum]; sorry

def discount_condition2 : (num_Fox * price_Fox * F / 100) + (num_Pony * price_Pony * P / 100) = total_savings := by
  dsimp [num_Fox, price_Fox, num_Pony, price_Pony, total_savings]; sorry

-- Question: Prove that the discount rate on Pony jeans is 14%
theorem discount_rate_on_pony : P = 14 := by
  have h1 : F + P = discount_sum := discount_condition1
  have h2 : (num_Fox * price_Fox * F / 100) + (num_Pony * price_Pony * P / 100) = total_savings := discount_condition2
  dsimp [num_Fox, price_Fox, num_Pony, price_Pony, total_savings, discount_sum] at *
  sorry

end discount_rate_on_pony_l289_289183


namespace minimum_photos_taken_l289_289307

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289307


namespace circles_are_separate_l289_289024

-- Define circle C1 as x^2 + y^2 = 1
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define circle C2 as (x-3)^2 + (y-4)^2 = 9
def circle2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 9

-- The Euclidean distance function
def euclidean_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Prove the positional relationship
theorem circles_are_separate : Separate :=
begin
  sorry
end

end circles_are_separate_l289_289024


namespace geometric_progression_common_ratio_l289_289328

-- Define the terms of the geometric progression
def a1 (x : ℝ) : ℝ := x / 2
def a2 (x : ℝ) : ℝ := 2 * x - 3
def a3 (x : ℝ) : ℝ := 18 / x + 1

-- State the theorem to prove the common ratio
theorem geometric_progression_common_ratio (x : ℝ) (hx : x ≠ 0) (h : a1(x) * a3(x) = a2(x) ^ 2) :
  (a2(x) / a1(x)) = 52 / 25 :=
by 
  sorry

end geometric_progression_common_ratio_l289_289328


namespace bisect_angle_PK_AP_BP_l289_289407

open EuclideanGeometry

-- Define the square and its properties
variables {A B C D K P : Point}

def is_center_of_square (K : Point) (A B C D : Point) := 
  is_square A B C D ∧ center A B C D = K

theorem bisect_angle_PK_AP_BP 
  (h_square : is_square A B C D)
  (h_center : is_center_of_square K A B C D)
  (h_P_ne_K : P ≠ K)
  (h_right_angle : ∠ A P B = 90) : 
  bisects (line_through P K) (angle AP BP) :=
sorry

end bisect_angle_PK_AP_BP_l289_289407


namespace sufficient_not_necessary_perpendicular_l289_289412

theorem sufficient_not_necessary_perpendicular (a : ℝ) :
  (∀ x y : ℝ, (a + 2) * x + 3 * a * y + 1 = 0 ∧
              (a - 2) * x + (a + 2) * y - 3 = 0 → false) ↔ a = -2 :=
sorry

end sufficient_not_necessary_perpendicular_l289_289412


namespace union_of_A_and_B_l289_289206

noncomputable def a : ℤ := 1 -- based on derived solution
noncomputable def b : ℤ := 2 -- based on derived solution

def A := ({5, b / a, a - b} : Set ℤ)
def B := ({b, a + b, -1} : Set ℤ)

theorem union_of_A_and_B :
  ({5, b / a, a - b} ∩ {b, a + b, -1} = {2, -1}) →
  ({5, b / a, a - b} ∪ {b, a + b, -1} = {-1, 2, 3, 5}) :=
by
  intro h
  sorry

end union_of_A_and_B_l289_289206


namespace reflection_A_BC_is_2_neg1_l289_289203

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 3⟩
def B : Point := ⟨0, 1⟩
def C : Point := ⟨3, 1⟩

-- Function to reflect a point across a horizontal line
def reflect_across_horizontal (p : Point) (y_line : ℝ) : Point :=
  ⟨p.x, 2 * y_line - p.y⟩

theorem reflection_A_BC_is_2_neg1 :
  let line_y := B.y in
  reflect_across_horizontal A line_y = ⟨2, -1⟩ :=
by
  sorry

end reflection_A_BC_is_2_neg1_l289_289203


namespace vehicles_involved_in_accidents_l289_289476

-- Define the constants for the conditions
def hundred_million_vehicles : ℕ := 100_000_000
def vehicles_per_accident : ℕ := 80
def total_vehicles : ℕ := 4_000_000_000

-- Define the theorem based on conditions
theorem vehicles_involved_in_accidents :
  total_vehicles / hundred_million_vehicles * vehicles_per_accident = 3200 :=
by
  sorry

end vehicles_involved_in_accidents_l289_289476


namespace rectangle_to_square_l289_289511

theorem rectangle_to_square :
  ∃ (a b : ℕ), a = 16 ∧ b = 9 ∧ (∃ (s : ℕ), s = 12 ∧ a * b = s * s) :=
by
  use 16
  use 9
  split
  exact rfl
  split
  exact rfl
  use 12
  split
  exact rfl
  sorry

end rectangle_to_square_l289_289511


namespace correct_propositions_l289_289634

variables {α β γ : Type} {l : α → Prop}

theorem correct_propositions (α_perp_γ : ∀ x : α, γ x → ¬ α x)
                            (β_perp_γ : ∀ y : β, γ y → ¬ β y)
                            (β_parallel_γ : ∀ z : β, γ z → β z)
                            (l_parallel_α : ∀ w : l, α w → l w)
                            (l_perp_β : ∀ u : l, β u → ¬ l u) :
  (∀ x, γ x → (α x → ¬ β x)) ∧                     -- Proposition 1
  (∀ x, (γ x → ¬ α x) ∧ (γ x → β x) → ¬ β x) ∧     -- Proposition 2
  (∀ x, (α x → l x) ∧ (l x → ¬ β x) → ¬ β x) ∧     -- Proposition 3
  (∀ x, (α x → l x) → l x) →             -- Proposition 4
  (∀ x, (γ x → ¬ α x) ∧ (γ x → β x) → ¬ β x) ∧    -- Correct proposition 2
  (∀ x, (α x → l x) ∧ (l x → ¬ β x) → ¬ β x)       -- Correct proposition 3 :=
by
  -- Correct proposition 2
  intros
  exact sorry
  -- Correct proposition 3
  intros
  exact sorry

end correct_propositions_l289_289634


namespace apple_pies_count_l289_289091

def total_pies := 13
def pecan_pies := 4
def pumpkin_pies := 7
def apple_pies := total_pies - pecan_pies - pumpkin_pies

theorem apple_pies_count : apple_pies = 2 := by
  sorry

end apple_pies_count_l289_289091


namespace probability_bernardo_larger_l289_289890

def set := {1, 2, 3, 4, 5, 6, 7, 8}

def is_even_3_digit_number (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ ∃ u, u ∈ s ∧ u % 2 = 0

noncomputable def bernardo_formed_even_number : Finset ℕ := 
  {s | s ⊆ set ∧ s.card = 3 ∧ is_even_3_digit_number s}

noncomputable def silvia_formed_even_number : Finset ℕ := 
  {s | s ⊆ set ∧ s.card = 3 ∧ is_even_3_digit_number s}

theorem probability_bernardo_larger :
  ∀ (b s : Finset ℕ),
    b ∈ bernardo_formed_even_number →
    s ∈ silvia_formed_even_number →
    (1 / 2 : ℝ) = 1 / 2 :=
by
  intros b s hb hs
  sorry

end probability_bernardo_larger_l289_289890


namespace maximum_F_l289_289223

noncomputable def f (x : ℝ) : ℝ := log x / log 2 + log (x + 1) / log 2
noncomputable def g (x : ℝ) : ℝ := 2 * log (x + 2) / log 2
noncomputable def F (x : ℝ) : ℝ := f x - g x

theorem maximum_F :
  (∀ x : ℝ, x > -1 → F x ≤ -2) ∧ (∃ x : ℝ, x > -1 ∧ F x = -2) := 
sorry

end maximum_F_l289_289223


namespace pq_difference_l289_289072

theorem pq_difference (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end pq_difference_l289_289072


namespace problem_1_problem_2_l289_289192

open Real InnerProductSpace

variables {E : Type*} [inner_product_space ℝ E]

-- Given conditions
variables (a b : E) (lambda : ℝ)
axiom norm_a : ∥a∥ = sqrt 3
axiom norm_b : ∥b∥ = 2
axiom angle_ab : ∠ a b = 150

-- Proof problems
theorem problem_1 : ∥a - (2 : ℝ) • b∥ = sqrt 31 := sorry

theorem problem_2 (h : inner (a + (3 : ℝ) • lambda • b) (a + lambda • b) = 0) : lambda = 1 / 2 := sorry

end problem_1_problem_2_l289_289192


namespace largest_non_sum_l289_289459

theorem largest_non_sum (n : ℕ) : 
  ¬ (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ∣ 2 ∧ n = 36 * a + b) ↔ n = 104 :=
by
  sorry

end largest_non_sum_l289_289459


namespace infinite_solutions_xyz_l289_289697

theorem infinite_solutions_xyz :
  ∃ (f : ℕ → ℕ × ℕ × ℕ), 
    (∀ t > 1, 
      let (x, y, z) := f t in (x^2 + x + 1) * (y^2 + y + 1) = (z^2 + z + 1)) 
    ∧ (∀ t > 1, let (x, y, z) := f t in x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

end infinite_solutions_xyz_l289_289697


namespace jordan_rectangle_width_l289_289897

noncomputable def carol_length : ℝ := 4.5
noncomputable def carol_width : ℝ := 19.25
noncomputable def jordan_length : ℝ := 3.75

noncomputable def carol_area : ℝ := carol_length * carol_width
noncomputable def jordan_width : ℝ := carol_area / jordan_length

theorem jordan_rectangle_width : jordan_width = 23.1 := by
  -- proof will go here
  sorry

end jordan_rectangle_width_l289_289897


namespace pretzels_count_l289_289710

-- Define the number of pretzels
def pretzels : ℕ := 64

-- Given conditions
def goldfish (P : ℕ) : ℕ := 4 * P
def suckers : ℕ := 32
def kids : ℕ := 16
def items_per_kid : ℕ := 22
def total_items (P : ℕ) : ℕ := P + goldfish P + suckers

-- The theorem to prove
theorem pretzels_count : total_items pretzels = kids * items_per_kid := by
  sorry

end pretzels_count_l289_289710


namespace inscribed_square_area_l289_289115

noncomputable def ellipse := { p : ℝ × ℝ // (p.1 ^ 2) / 3 + (p.2 ^ 2) / 6 = 1 }

theorem inscribed_square_area :
  ∃ (t : ℝ), t = √2 ∧
  (let side_length := 2 * t in side_length * side_length = 8) :=
by
  use √2
  split
  · rfl
  · let side_length := 2 * √2
    show side_length * side_length = 8
    sorry

end inscribed_square_area_l289_289115


namespace sugar_left_l289_289510

variable (full_amount_needed fraction : ℝ)

-- Given conditions as variables
def recipe_sugar := 2 : ℝ
def recipe_fraction := 0.165 : ℝ

theorem sugar_left (h1 : full_amount_needed = recipe_sugar) (h2 : fraction = recipe_fraction) :
  full_amount_needed * fraction = 0.33 := by
  sorry

end sugar_left_l289_289510


namespace range_of_f_l289_289360

noncomputable theory

open Real

def f (x : ℝ) : ℝ := (1 / 2) * (cos x)^2 + (sqrt 3 / 2) * (sin x) * (cos x) + 2

theorem range_of_f : 
  let I := Icc (-π / 6) (π / 4) in
  (set.range (λ x : I, f x) = set.Icc (2 : ℝ) (2 + 3 / 4 : ℝ)) := by
    sorry

end range_of_f_l289_289360


namespace tangency_point_l289_289931

noncomputable def point_of_tangency : ℝ × ℝ :=
  (-9/2, -35/2)

def parabola1 (x : ℝ) : ℝ :=
  x^2 + 10 * x + 19

def parabola2 (y : ℝ) : ℝ :=
  y^2 + 36 * y + 325

theorem tangency_point : 
  let p := point_of_tangency in 
  parabola1 p.1 = p.2 ∧ parabola2 p.2 = p.1 :=
by
  sorry

end tangency_point_l289_289931


namespace increase_in_average_commission_l289_289709

theorem increase_in_average_commission :
  ∀ (new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 : ℕ),
    new_avg = 400 → 
    n1 = 6 → 
    n2 = n1 - 1 → 
    big_sale = 1300 →
    total_earnings = new_avg * n1 →
    commission = total_earnings - big_sale →
    old_avg = commission / n2 →
    new_avg - old_avg = 180 :=
by 
  intros new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end increase_in_average_commission_l289_289709


namespace saline_solution_problem_l289_289041

open Real

noncomputable def solveSalineSolution (x y z : ℝ) : ℝ :=
  if z = 47 then 49 else 35

theorem saline_solution_problem :
  ∑ (solveSalineSolution 60 60 47) (solveSalineSolution 60 60 40) = 84 :=
begin
  sorry
end

end saline_solution_problem_l289_289041


namespace jesse_remaining_pages_l289_289339

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end jesse_remaining_pages_l289_289339


namespace power_series_expansion_and_region_of_validity_l289_289558

noncomputable def f (z : ℂ) : ℂ := (z + 1) / ((z - 1) ^ 2 * (z + 2))

theorem power_series_expansion_and_region_of_validity :
  f = λ z, ∑' n, (1 / 9 : ℂ) * (6 * n + 5 + (-1) ^ (n + 1) / (2 ^ (n+1))) * z ^ n ∧ for all (z : ℂ), abs z < 1 :=
by
  sorry

end power_series_expansion_and_region_of_validity_l289_289558


namespace time_to_be_d_miles_apart_l289_289372

def mary_walk_rate := 4 -- Mary's walking rate in miles per hour
def sharon_walk_rate := 6 -- Sharon's walking rate in miles per hour
def time_to_be_3_miles_apart := 0.3 -- Time in hours to be 3 miles apart
def initial_distance := 3 -- They are 3 miles apart after 0.3 hours

theorem time_to_be_d_miles_apart (d: ℝ) : ∀ t: ℝ,
  (mary_walk_rate + sharon_walk_rate) * t = d ↔ 
  t = d / (mary_walk_rate + sharon_walk_rate) :=
by
  intros
  sorry

end time_to_be_d_miles_apart_l289_289372


namespace rectangular_box_inscribed_in_sphere_l289_289110

noncomputable def problem_statement : Prop :=
  ∃ (a b c s : ℝ), (4 * (a + b + c) = 72) ∧ (2 * (a * b + b * c + c * a) = 216) ∧
  (a^2 + b^2 + c^2 = 108) ∧ (4 * s^2 = 108) ∧ (s = 3 * Real.sqrt 3)

theorem rectangular_box_inscribed_in_sphere : problem_statement := 
  sorry

end rectangular_box_inscribed_in_sphere_l289_289110


namespace parallelogram_BP1Q_l289_289997

noncomputable def configuration (outerCircle innerCircle : Circle) (N : Point) (B A : Point) (K M : Point)
  (Q P : Point) (B1 : Point) : Prop :=
  touches innerCircle outerCircle N ∧
  tangentChord outerCircle innerCircle B A K ∧
  tangentChord outerCircle innerCircle B C M ∧
  midpointArc outerCircle A B N Q ∧
  midpointArc outerCircle B C N P ∧
  secondIntersectionCircumcircle outerCircle B Q K innerCircle B P M B1

theorem parallelogram_BP1Q (outerCircle innerCircle : Circle) (N : Point) (B A K M Q P B1 : Point)
  (h : configuration outerCircle innerCircle N B A K M Q P B1) : parallelogram B P B1 Q :=
begin
  sorry
end

end parallelogram_BP1Q_l289_289997


namespace range_of_x_l289_289600

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)
noncomputable def is_monotone (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f(x) ≤ f(y)

theorem range_of_x (f : ℝ → ℝ) (h1 : is_even f) (h2 : is_monotone f) (h3 : ∀ x, 0 ≤ x → f x = f (-x)) :
  ∀ x, x ∈ set.Ioo (1 / 3) (2 / 3) → f (2 * x - 1) < f (1 / 3) :=
sorry

end range_of_x_l289_289600


namespace inscribed_circle_radii_rel_l289_289263

theorem inscribed_circle_radii_rel {a b c r r1 r2 : ℝ} :
  (a^2 + b^2 = c^2) ∧
  (r1 = (a / c) * r) ∧
  (r2 = (b / c) * r) →
  r^2 = r1^2 + r2^2 :=
by 
  sorry

end inscribed_circle_radii_rel_l289_289263


namespace bridge_length_correct_l289_289122

-- Define the given conditions
def train_length : ℝ := 250             -- The length of the train in meters
def train_speed_km_hr : ℝ := 60         -- The speed of the train in km/hr
def time_seconds : ℝ := 20              -- The time to cross the bridge in seconds
def train_speed_m_s : ℝ := (train_speed_km_hr * 1000) / 3600  -- Conversion from km/hr to m/s

-- Define the distance covered by the train in the given time
def distance_covered : ℝ := train_speed_m_s * time_seconds

-- Define the length of the bridge as the distance covered minus the length of the train
def bridge_length : ℝ := distance_covered - train_length

-- The proof statement
theorem bridge_length_correct : bridge_length = 83.4 := sorry

end bridge_length_correct_l289_289122


namespace floor_sum_geq_l289_289077

-- Definitions based on the problem
def floor_sum (x : ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), (⌊(i:ℝ) * x⌋ / i)

theorem floor_sum_geq (x : ℝ) (n : ℕ) (hx : 0 < x) (hn : 0 < n) :
  ⌊(n:ℝ) * x⌋ ≥ floor_sum x n :=
sorry

end floor_sum_geq_l289_289077


namespace product_of_possible_values_of_P_l289_289139

theorem product_of_possible_values_of_P :
  ∀ (C D P : ℝ),
  (C = D + P) →
  (C - 4 = D + P - 4) →
  (D + 2 = D + 2) →
  (|C - 4 - (D + 2)| = 6) →
  (P = 12 ∨ P = 0) →
  (0 : ℝ) := sorry

end product_of_possible_values_of_P_l289_289139


namespace largest_divisor_of_Q_l289_289125

def Q (hidden : ℕ) : ℕ := (∏ x in (Finset.range 12).erase hidden, x + 1)

theorem largest_divisor_of_Q :
  ∀ hidden ∈ Finset.range 12, (Q hidden) % 138600 = 0 :=
by
  sorry

end largest_divisor_of_Q_l289_289125


namespace simplify_expression_l289_289741

noncomputable def algebraic_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) : ℚ :=
(1 - 3 / (a + 2)) / ((a^2 - 2 * a + 1) / (a^2 - 4))

theorem simplify_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) :
  algebraic_expression a h1 h2 h3 = (a - 2) / (a - 1) :=
by
  sorry

end simplify_expression_l289_289741


namespace collinear_after_construction_l289_289620

variables {A B C A* B* C' : Point}

-- Definition of the condition: equilateral triangles constructed
-- outwardly on side BC with vertex A*, and inwardly on side CA with vertex B*.
def is_equilateral_triangle_outward (A B C A* : Point) : Prop :=
-- define it here

def is_equilateral_triangle_inward (B C A B* : Point) : Prop :=
-- define it here

-- Definition of reflection of point C over line AB to get C'
def reflect_over_line (C A B : Point) : Point := 
-- define it here

-- Proof statement
theorem collinear_after_construction 
  (ABC_is_triangle : Triangle A B C)
  (equilateral_triangle_outward : is_equilateral_triangle_outward A B C A*)
  (equilateral_triangle_inward : is_equilateral_triangle_inward B C A B*)
  (C'_reflected_over_AB  : C' = reflect_over_line C A B) :
  collinear A* B* C' :=
begin
  sorry  -- Proof to be added here
end

end collinear_after_construction_l289_289620


namespace dirichlet_solution_in_sphere_l289_289003

-- Define the Dirichlet problem for the Laplace equation in a sphere
def LaplaceSolution (u : ℝ → ℝ → ℝ) : Prop :=
  (∀ (r : ℝ) (θ : ℝ), r < 1 → ∆ u r θ = 0) ∧ (∀ θ, u 1 θ = 3 * cos (θ) ^ 2)

-- Correct answer
noncomputable def solution (r θ : ℝ) : ℝ :=
  1 + r^2 * (3 * cos(θ)^2 - 1)

-- The theorem to be proven
theorem dirichlet_solution_in_sphere :
  ∃ u : ℝ → ℝ → ℝ, LaplaceSolution u ∧ ∀ r θ, u r θ = solution r θ := 
sorry

end dirichlet_solution_in_sphere_l289_289003


namespace iceCreamCombo_l289_289093

open Finset

def iceCreamFlavors : ℕ := 5
def toppings : ℕ := 7
def chooseRequiredToppings : ℕ := 3

-- Theorem statement
theorem iceCreamCombo :
  iceCreamFlavors * (choose toppings chooseRequiredToppings) = 175 := by
  sorry

end iceCreamCombo_l289_289093


namespace min_photographs_42_tourists_3_monuments_l289_289273

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289273


namespace year_with_max_increase_after_1995_l289_289427

noncomputable def sales_increase_per_year : List ℕ :=
  [1000, 1500, 2000, 3000, 3500, 5000, 5500, 6000, 5750, 4750]

noncomputable def calculate_increases (sales : List ℕ) : List ℤ :=
  List.map2 (λ (p n : ℕ) => (n : ℤ) - ↑p) (List.dropLast 1 sales) (List.drop 1 sales)

theorem year_with_max_increase_after_1995 :
  let increases := calculate_increases sales_increase_per_year
  List.indexOf increases (List.maximum increases) + 1996 = 2000 :=
by
  sorry

end year_with_max_increase_after_1995_l289_289427


namespace max_mass_sand_is_correct_l289_289506

def platform_length := 8 -- Length of the platform in meters
def platform_width := 4 -- Width of the platform in meters
def angle_max := Real.pi / 4 -- Maximum angle in radians (45 degrees)
def sand_density := 1500 -- Density of sand in kg/m³

noncomputable def max_mass_sand : ℝ :=
  let height := platform_width / 2
  let volume_prism := platform_length * platform_width * height
  let volume_pyramid := (2 / 3) * (platform_width / 2) * platform_length * height
  (volume_prism + volume_pyramid) * sand_density

theorem max_mass_sand_is_correct : max_mass_sand = 112000 := by
  sorry

end max_mass_sand_is_correct_l289_289506


namespace anya_smallest_number_divisible_by_eleven_l289_289885

theorem anya_smallest_number_divisible_by_eleven :
  ∃ (n: ℕ), n = 909090909 ∧ (∀ (d_pos: ℕ), d_pos < n.num_digits → 
  ∀ (new_digit: ℕ), abs (new_digit - (n.digits.get d_pos)) = 1 → 
  (digits_to_nat (n.digits.update d_pos new_digit)) % 11 = 0) :=
sorry

end anya_smallest_number_divisible_by_eleven_l289_289885


namespace average_temperature_bucyrus_l289_289449

theorem average_temperature_bucyrus : 
  let t1 := -14
  let t2 := -8
  let t3 := 1 
  (t1 + t2 + t3) / 3 = -7 := 
  by
  rw [show t1 + t2 + t3 = -21 by norm_num]
  rw [show -21 / 3 = -7 by norm_num]
  sorry

end average_temperature_bucyrus_l289_289449


namespace count_terms_in_expression_l289_289417

theorem count_terms_in_expression :
  let terms := { (a, b, c) | a + b + c = 2010 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a % 2 = 0 ∧ a > b } in
  terms.card = 1010030 :=
by
  sorry

end count_terms_in_expression_l289_289417


namespace non_self_intersecting_polygon_count_l289_289780

theorem non_self_intersecting_polygon_count (n : ℕ) (h : n > 1) : 
  number_of_ways n = n * 2^(n - 2) := 
sorry

end non_self_intersecting_polygon_count_l289_289780


namespace largest_number_is_763210_l289_289795

-- Define the set of digits
def distinct_digits_sum_17 : Set ℕ := {n | (∃ (digits : List ℕ),
  n = digits.foldl (λ acc d, acc + d) 0 ∧
  digits.nodup ∧
  digits.sum = 17)}

-- Define the goal to show that 763210 is the largest such number
theorem largest_number_is_763210 
  (h : 763210 ∈ distinct_digits_sum_17): 
  ∀ n ∈ distinct_digits_sum_17, n ≤ 763210 :=
sorry

end largest_number_is_763210_l289_289795


namespace price_calculations_l289_289047

noncomputable def initial_cost : ℝ := 9 * 4.50

def discount1 : ℝ := 3/7
def discount2 : ℝ := 0.25
def discount3 : ℝ := 0.15

def best_discount1 : ℝ := discount1
def best_discount2 : ℝ := discount2

noncomputable def first_discount_applied : ℝ := initial_cost * best_discount1
noncomputable def subtotal_after_first_discount : ℝ := initial_cost - first_discount_applied
noncomputable def second_discount_applied : ℝ := subtotal_after_first_discount * best_discount2
noncomputable def price_before_tax : ℝ := subtotal_after_first_discount - second_discount_applied
noncomputable def sales_tax : ℝ := 0.07 * price_before_tax
noncomputable def final_price : ℝ := price_before_tax + sales_tax

theorem price_calculations : 
  final_price = 18.56 ∧ 
  (first_discount_applied + second_discount_applied) = 23.15 ∧ 
  sales_tax = 1.21 :=
by
  sorry

end price_calculations_l289_289047


namespace reward_model_meets_conditions_l289_289875

theorem reward_model_meets_conditions : 
  ∀ (x : ℝ), 10 ≤ x ∧ x ≤ 100 → 
    (λ x, log x / log 2 - 2) x ≤ 5 ∧ 
    (λ x, log x / log 2 - 2) x ≤ x / 5 :=
by 
  intro x 
  assume h : 10 ≤ x ∧ x ≤ 100
  sorry

end reward_model_meets_conditions_l289_289875


namespace x_squared_plus_y_squared_l289_289975

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y) ^ 2 = 9) : 
  x ^ 2 + y ^ 2 = 15 := sorry

end x_squared_plus_y_squared_l289_289975


namespace calculate_p_p1_neg1_p_neg5_neg2_l289_289701

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then
    x + y
  else if x < 0 ∧ y < 0 then
    x - 2 * y
  else
    3 * x + y

theorem calculate_p_p1_neg1_p_neg5_neg2 :
  p (p 1 (-1)) (p (-5) (-2)) = 5 :=
by
  sorry

end calculate_p_p1_neg1_p_neg5_neg2_l289_289701


namespace sales_profit_nth_year_no_profit_2014_l289_289641

-- Question 1: Proving the sales profit in the nth year
theorem sales_profit_nth_year (n : ℕ) : 
  let a_n := 10000 + 10000 * n
  let b_n := 2 * (0.9 ^ (n - 1))
  (a_n * b_n) = (10000 + 10000 * n) * (2 * (0.9 ^ (n - 1))) := 
by 
  intros 
  rw [mul_assoc, ← mul_assoc 10000, mul_comm 10000, ← mul_assoc]
  sorry

-- Question 2: Proving lack of profitability by end of 2014
theorem no_profit_2014 : 
  let total_profit := (20000 * 2 + 30000 * 2 * (0.9) + 40000 * 2 * (0.9^2) + 50000 * 2 * (0.9^3) + 60000 * 2 * (0.9^4))
  (total_profit < 3800*10^6) := 
by 
  have total_profit_calculated : total_profit = 312 := by 
    sorry
  rw total_profit_calculated
  norm_num
  sorry

end sales_profit_nth_year_no_profit_2014_l289_289641


namespace annual_income_correct_l289_289474

def investment (amount : ℕ) := 6800
def dividend_rate (rate : ℕ) := 20
def stock_price (price : ℕ) := 136
def face_value : ℕ := 100
def calculate_annual_income (amount rate price value : ℕ) : ℕ := 
  let shares := amount / price
  let annual_income_per_share := value * rate / 100
  shares * annual_income_per_share

theorem annual_income_correct : calculate_annual_income (investment 6800) (dividend_rate 20) (stock_price 136) face_value = 1000 :=
by
  sorry

end annual_income_correct_l289_289474


namespace personBCatchesPersonAAtB_l289_289727

-- Definitions based on the given problem's conditions
def personADepartsTime : ℕ := 8 * 60  -- Person A departs at 8:00 AM, given in minutes
def personBDepartsTime : ℕ := 9 * 60  -- Person B departs at 9:00 AM, given in minutes
def catchUpTime : ℕ := 11 * 60        -- Persons meet at 11:00 AM, given in minutes
def returnMultiplier : ℕ := 2         -- Person B returns at double the speed
def chaseMultiplier : ℕ := 2          -- After returning, Person B doubles their speed again

-- Exact question we want to prove
def meetAtBTime : ℕ := 12 * 60 + 48   -- Time when Person B catches up with Person A at point B

-- Statement to be proven
theorem personBCatchesPersonAAtB :
  ∀ (VA VB : ℕ) (x : ℕ),
    VA = 2 * x ∧ VB = 3 * x →
    ∃ t : ℕ, t = meetAtBTime := by
  sorry

end personBCatchesPersonAAtB_l289_289727


namespace joe_lowest_dropped_score_l289_289341

theorem joe_lowest_dropped_score (A B C D : ℕ) 
  (hmean_before : (A + B + C + D) / 4 = 35)
  (hmean_after : (A + B + C) / 3 = 40)
  (hdrop : D = min A (min B (min C D))) :
  D = 20 :=
by sorry

end joe_lowest_dropped_score_l289_289341


namespace no_max_min_value_l289_289555

noncomputable def expression (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 4)

theorem no_max_min_value (x : ℝ) (h1 : x > -1) (h2 : x < 5) :
  ¬(∃ y : ℝ, ∀ z : ℝ, (h1 < z ∧ z < h2) → (expression z ≥ y)) ∧
  ¬(∃ y : ℝ, ∀ z : ℝ, (h1 < z ∧ z < h2) → (expression z ≤ y)) := 
sorry

end no_max_min_value_l289_289555


namespace largest_r_s_sum_l289_289051

variable (r s : ℝ)
def D : ℝ × ℝ := (10, 15)
def E : ℝ × ℝ := (18, 17)
def F : ℝ × ℝ := (r, s)
def area_DEF := 36.0
def slope_median_DE := -3.0

theorem largest_r_s_sum
  (h1 : ∃ (r s : ℝ), (1 / 2) * abs (r * (D.2 - E.2) + D.1 * (E.2 - s) + E.1 * (s - D.2)) = area_DEF)
  (h2 : s = slope_median_DE * (r - 14) + 58) :
  ∃ (r s : ℝ), (r + s = 35.54) :=
by
  sorry

end largest_r_s_sum_l289_289051


namespace equivalent_expression_l289_289461

theorem equivalent_expression : 
  \frac{((2304 + 88) - 2400)^2}{121} = \frac{64}{121} :=
by
  -- We need to prove step by step according to the conditions given
  sorry

end equivalent_expression_l289_289461


namespace max_OA_div_OB_value_l289_289664

open Real

noncomputable def parametric_curve_C (α : ℝ) : ℝ × ℝ := (1 + cos α, sin α)
noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ := (1 - t, 3 + t)

def polar_ray_m (θ β : ℝ) : Prop := θ = β ∧ ∀ ρ, ρ > 0

def ρ₁ (β : ℝ) : ℝ := 2 * cos β
def ρ₂ (β : ℝ) : ℝ := 2 * sqrt 2 / (sin (β + π/4))

def max_OA_div_OB : ℝ := (sqrt 2 + 1)/4

theorem max_OA_div_OB_value (β : ℝ) (hβ1 : β ∈ Ioo (-π / 4) (π / 4)) : 
  ∃ (A B : ℝ × ℝ), polar_ray_m A.1 β ∧ polar_ray_m B.1 β ∧ 
  (parametric_curve_C β = A) ∧ (parametric_line_l β = B) ∧
  (abs (1 / ρ₂ β)) = max_OA_div_OB :=
sorry

end max_OA_div_OB_value_l289_289664


namespace simplify_and_evaluate_l289_289398

theorem simplify_and_evaluate : 
    ∀ (a b : ℤ), a = 1 → b = -1 → 
    ((2 * a^2 * b - 2 * a * b^2 - b^3) / b - (a + b) * (a - b) = 3) := 
by
  intros a b ha hb
  sorry

end simplify_and_evaluate_l289_289398


namespace minimum_photos_taken_l289_289305

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289305


namespace teagan_leather_jackets_l289_289711

def reduced_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
    original_price - original_price * discount_rate

variable [orig_shirt_price : ℝ := 60]
variable [orig_jacket_price : ℝ := 90]
variable [discount_rate : ℝ := 0.20]
variable [num_shirts : Nat := 5]
variable [total_paid : ℝ := 960]

def shirt_reduced_price : ℝ := reduced_price orig_shirt_price discount_rate
def jacket_reduced_price : ℝ := reduced_price orig_jacket_price discount_rate
def total_shirt_cost : ℝ := num_shirts * shirt_reduced_price

theorem teagan_leather_jackets : 
  let total_jacket_cost := total_paid - total_shirt_cost,
      num_jackets := total_jacket_cost / jacket_reduced_price
  in num_jackets = 10 := by
  sorry

end teagan_leather_jackets_l289_289711


namespace range_of_g_l289_289915

noncomputable def g (x : ℝ) : ℝ := (3 * x + 8 - 2 * x ^ 2) / (x + 4)

theorem range_of_g : 
  (∀ y : ℝ, ∃ x : ℝ, x ≠ -4 ∧ y = (3 * x + 8 - 2 * x^2) / (x + 4)) :=
by
  sorry

end range_of_g_l289_289915


namespace exactly_one_divisible_by_5_l289_289205

def a (n : ℕ) : ℕ := 2^(2*n + 1) - 2^(n + 1) + 1
def b (n : ℕ) : ℕ := 2^(2*n + 1) + 2^(n + 1) + 1

theorem exactly_one_divisible_by_5 (n : ℕ) (hn : 0 < n) : (a n % 5 = 0 ∧ b n % 5 ≠ 0) ∨ (a n % 5 ≠ 0 ∧ b n % 5 = 0) :=
  sorry

end exactly_one_divisible_by_5_l289_289205


namespace linear_eq_one_variable_l289_289466

def isLinearEquationWithOneVariable (eq : String) : Prop :=
  (eq = "y + 3 = 0") ∧ (eq.contains "y" ∧ ¬ eq.contains "x" ∧ eq.powMax 1)

theorem linear_eq_one_variable : isLinearEquationWithOneVariable "y + 3 = 0" :=
by
  sorry

end linear_eq_one_variable_l289_289466


namespace randy_blocks_l289_289388

theorem randy_blocks (total_blocks house_blocks diff_blocks tower_blocks : ℕ) 
  (h_total : total_blocks = 90)
  (h_house : house_blocks = 89)
  (h_diff : house_blocks = tower_blocks + diff_blocks)
  (h_diff_value : diff_blocks = 26) :
  tower_blocks = 63 :=
by
  -- sorry is placed here to skip the proof.
  sorry

end randy_blocks_l289_289388


namespace probability_at_least_one_vowel_l289_289396

noncomputable def set1 : finset char := {'a', 'b', 'c', 'd', 'e'}
noncomputable def set2 : finset char := {'k', 'l', 'm', 'n', 'o', 'p'}
noncomputable def set3 : finset char := {'r', 's', 't', 'u', 'v'}
noncomputable def set4 : finset char := {'w', 'x', 'y', 'z', 'i'}

def vowels : set char := {'a', 'e', 'i', 'o', 'u'}

def probability_of_picking_at_least_one_vowel : ℚ :=
  let p_set1 := (3 / 5 : ℚ)
  let p_set2 := (1 : ℚ)
  let p_set3 := (1 : ℚ)
  let p_set4 := (4 / 5 : ℚ)
  let p_no_vowel := p_set1 * p_set2 * p_set3 * p_set4
  (1 - p_no_vowel)

theorem probability_at_least_one_vowel :
  probability_of_picking_at_least_one_vowel = 13 / 25 :=
sorry

end probability_at_least_one_vowel_l289_289396


namespace binom_eq_one_binom_320_l289_289542

theorem binom_eq_one (n : ℕ) : (n.choose n) = 1 :=
  by sorry

theorem binom_320 : Nat.choose 320 320 = 1 :=
  by exact binom_eq_one 320

end binom_eq_one_binom_320_l289_289542


namespace hyeji_total_water_intake_l289_289631

variable (daily_intake_ml : ℕ) (extra_intake_ml : ℕ)

def total_intake(daily_intake_ml extra_intake_ml : ℕ) : ℕ :=
  daily_intake_ml + extra_intake_ml

theorem hyeji_total_water_intake :
  daily_intake_ml = 2000 ∧ extra_intake_ml = 460 → total_intake daily_intake_ml extra_intake_ml = 2460 :=
by
  intro h
  cases h with h_daily h_extra
  rw [h_daily, h_extra]
  exact rfl

end hyeji_total_water_intake_l289_289631


namespace cooking_ways_l289_289712

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem cooking_ways : comb 5 2 = 10 :=
  by
  sorry

end cooking_ways_l289_289712


namespace mindmaster_secret_codes_l289_289647

theorem mindmaster_secret_codes (slots colors : ℕ) (h1 : slots = 6) (h2 : colors = 5) : (colors ^ slots = 15625) :=
by
  rw [h1, h2]
  simp
  exact dec_trivial

end mindmaster_secret_codes_l289_289647


namespace sequence_converges_l289_289545

theorem sequence_converges (x : ℕ → ℝ) (h1 : x 1 > 1) (h2 : ∀ n, ∑ i in finset.range (n + 1), x (i + 1) = ∏ i in finset.range (n + 1), x (i + 1)) :
  ∃ l, real.is_limit (x n) l ∧ l = 1 := 
sorry

end sequence_converges_l289_289545


namespace probability_odd_product_three_integers_l289_289045

theorem probability_odd_product_three_integers :
  let p := (500/1000) * (499/999) * (498/998)
  in p < (1/8) :=
by
  let p := (500/1000) * (499/999) * (498/998)
  show p < (1/8)
  sorry

end probability_odd_product_three_integers_l289_289045


namespace min_photos_l289_289299

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289299


namespace total_pizzas_served_l289_289513

def lunch_pizzas : ℚ := 12.5
def dinner_pizzas : ℚ := 8.25

theorem total_pizzas_served : lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end total_pizzas_served_l289_289513


namespace average_temperature_l289_289446

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end average_temperature_l289_289446


namespace find_a_n_find_T_n_l289_289648

-- Definitions for Problem 1 
def arithmetic_seq (a d : ℕ → ℚ) (n : ℕ) : Prop :=
  ∀ n, a n = a 1 + ((n - 1) * d)

def geo_condition (a : ℕ → ℚ) (d : ℚ) : Prop :=
  a 1 * a 8 = (a 4)^2

def sum_condition (a : ℕ → ℚ) (S : ℚ) : Prop :=
  (∑ k in range 10, a k + 1) = S

-- Proof Problem 1: Proving the general term of the arithmetic sequence
theorem find_a_n (a : ℕ → ℚ) (d : ℚ) (S : ℚ) (h1 : geo_condition a d) (h2 : sum_condition a S) :
  ∀ n, a n = (n + 8) / 3 := sorry

-- Definitions for Problem 2
def b_n (a : ℕ → ℚ) (n : ℕ) := 1 / (a n * a (n + 1))

def b_sum (b : ℕ → ℚ) (T : ℚ) : Prop :=
  ∀ n, ∑ k in range n, b k = T

-- Proof Problem 2: Proving T_n for the series b_n
theorem find_T_n (a : ℕ → ℚ) (T : ℚ) (ha : ∀ n, a n = (n + 8) / 3) :
  ∀ n, ∑ k in range n, b_n a k = n / (n + 9) := sorry

end find_a_n_find_T_n_l289_289648


namespace probability_of_four_of_same_value_l289_289163

-- Define the conditions
def total_ways_to_draw_6_cards : ℕ := Nat.factorial 52 / (Nat.factorial 6 * Nat.factorial (52 - 6))
def ways_to_choose_4_of_same_value : ℕ := 13 * (Nat.factorial 48 / (Nat.factorial 2 * Nat.factorial (48 - 2)))

-- Define the probability calculation
def probability : ℚ :=
  (ways_to_choose_4_of_same_value : ℚ) / (total_ways_to_draw_6_cards : ℚ)

-- Prove that the probability equals 3/4165
theorem probability_of_four_of_same_value :
  probability = 3 / 4165 :=
sorry

end probability_of_four_of_same_value_l289_289163


namespace positional_relationship_l289_289619

-- Assume AB, BC, and CD are line segments in space
variables {AB BC CD : Type} [is_line_segment AB] [is_line_segment BC] [is_line_segment CD]

-- Angles between segments are equal
axiom angle_ABC_eq_angle_BCD : angle ABC BC = angle BCD CD 

-- Positional relationships: either parallel, skew, or intersect
theorem positional_relationship (AB BC CD : Type) [is_line_segment AB] [is_line_segment BC] [is_line_segment CD]
  (h : angle ABC BC = angle BCD CD) : 
  (is_parallel AB CD) ∨ (is_skew AB CD) ∨ (intersects AB CD) :=
sorry

end positional_relationship_l289_289619


namespace net_rate_25_dollars_per_hour_l289_289870

noncomputable def net_rate_of_pay (hours : ℕ) (speed : ℕ) (mileage : ℕ) (rate_per_mile : ℚ) (diesel_cost_per_gallon : ℚ) : ℚ :=
  let distance := hours * speed
  let diesel_used := distance / mileage
  let earnings := rate_per_mile * distance
  let diesel_cost := diesel_cost_per_gallon * diesel_used
  let net_earnings := earnings - diesel_cost
  net_earnings / hours

theorem net_rate_25_dollars_per_hour :
  net_rate_of_pay 4 45 15 (0.75 : ℚ) (3.00 : ℚ) = 25 :=
by
  -- Proof is omitted
  sorry

end net_rate_25_dollars_per_hour_l289_289870


namespace min_cost_to_form_closed_chain_l289_289854

/-- Definition for the cost model -/
def cost_separate_link : ℕ := 1
def cost_attach_link : ℕ := 2
def total_cost (n : ℕ) : ℕ := n * (cost_separate_link + cost_attach_link)

-- Number of pieces of gold chain and links in each chain
def num_pieces : ℕ := 13

/-- Minimum cost calculation proof statement -/
theorem min_cost_to_form_closed_chain : total_cost (num_pieces - 1) = 36 := 
by
  sorry

end min_cost_to_form_closed_chain_l289_289854


namespace downstream_speed_l289_289503

variable (Vu Vs Vd Vc : ℝ)

theorem downstream_speed
  (h1 : Vu = 25)
  (h2 : Vs = 32)
  (h3 : Vu = Vs - Vc)
  (h4 : Vd = Vs + Vc) :
  Vd = 39 := by
  sorry

end downstream_speed_l289_289503


namespace ln_of_gt_of_pos_l289_289238

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end ln_of_gt_of_pos_l289_289238


namespace max_pieces_after_cuts_l289_289906

theorem max_pieces_after_cuts (n : ℕ) : n = 5 → (1 + n * (n + 1) / 2) = 16 := by
  intros h
  rw h
  norm_num
  sorry

end max_pieces_after_cuts_l289_289906


namespace divisibility_in_base_8_l289_289666

theorem divisibility_in_base_8 : 
  ∃ (b : ℕ), (b ≥ 8) ∧ ((7 * b + 2) % (2 * b ^ 2 + 7 * b + 5) = 0) ∧ (b = 8) :=
by {
  -- Definitions based on conditions
  let b := 8,
  -- Prove the inequalities and divisibility
  have h1 : b ≥ 8 := by norm_num,
  have h2 : (7 * b + 2) % (2 * b ^ 2 + 7 * b + 5) = 0 := by norm_num [b],
  exact ⟨b, ⟨h1, h2, rfl⟩⟩,
  sorry
}

end divisibility_in_base_8_l289_289666


namespace sum_of_mean_and_median_l289_289029

noncomputable def median (s : List ℚ) : ℚ :=
  let n := s.length
  if n % 2 = 1 then
    s[(n / 2)]!
  else
    (s[(n / 2 - 1)]! + s[(n / 2)]!) / 2

noncomputable def mean (s : List ℚ) : ℚ :=
  s.sum / s.length

theorem sum_of_mean_and_median : 
  let s := [1, 2, 3, 0, 1].qsort (≤) in
  (median s) + (mean s) = 12/5 := by
  sorry

end sum_of_mean_and_median_l289_289029


namespace trig_identity_l289_289353

theorem trig_identity (θ : ℝ) (h : cos (2 * θ) = 1 / 5) : sin θ ^ 6 + cos θ ^ 6 = 7 / 25 :=
by
  sorry

end trig_identity_l289_289353


namespace largest_num_with_diff_digits_sum_17_l289_289805
-- Import the necessary library

-- Define the conditions and the answer
def digits_all_different (n : Nat) : Prop := 
  (n.toDigits).nodup

def digits_sum_17 (n : Nat) : Prop :=
  (n.toDigits).sum = 17

def largest_number := 6543210

-- The main statement to be proved
theorem largest_num_with_diff_digits_sum_17 : ∃ n, digits_all_different n ∧ digits_sum_17 n ∧ n = largest_number :=
by
  sorry

end largest_num_with_diff_digits_sum_17_l289_289805


namespace unique_solution_of_equation_l289_289240

theorem unique_solution_of_equation (x y : ℝ) (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
by
  sorry

end unique_solution_of_equation_l289_289240


namespace factorize_polynomial_l289_289559

noncomputable def P (a b x : ℂ) : ℂ := a + (a + b) * x + (a + 2 * b) * x^2 + (a + 3 * b) * x^3 + 3 * b * x^4 + 2 * b * x^5 + b * x^6

theorem factorize_polynomial (a b : ℂ) : 
  ∃ (Q : ℂ → ℂ) (hQ : Q = λ x, a + b * x + b * x^2 + b * x^3), 
    ∀ x : ℂ, P a b x = (1 + x) * (1 + x^2) * Q x :=
sorry

end factorize_polynomial_l289_289559


namespace chinese_chess_competition_l289_289678

theorem chinese_chess_competition (n : ℕ) (h : n ≥ 2) :
  ∃ P : Fin n × Fin n → Fin 2n^2,
    (∀ {i j i' j' : Fin n}, i < i' → P (i, j) < P (i', j')) ∧
    (∀ (A B C : Fin 2n^2), (A < B → B < C → A < C)) ∧
    (∃ k : ℕ, k ≤ n^3 / 16 ∧ ∀ (A B : Fin 2n^2), k = (A = B)) :=
  sorry

end chinese_chess_competition_l289_289678


namespace median_and_mode_of_written_scores_composite_score_percentages_find_top_two_candidates_l289_289838

-- Question 1: Median and Mode calculation for given written test scores.
theorem median_and_mode_of_written_scores : 
  ∀ (scores : List ℕ), 
  scores = [85, 92, 84, 90, 84, 80] →
  ∃ (median mode : ℝ),
    median = (84 + 85) / 2 ∧ mode = 84 := 
by
  intro scores h_scores
  have h : scores.sorted = [80, 84, 84, 85, 90, 92] := sorry
  use ((84 + 85) / 2), 84
  exact sorry

-- Question 2: Finding the composition percentages.
theorem composite_score_percentages :
  ∃ (x y : ℝ), 
    x + y = 1 ∧ 
    85 * x + 90 * y = 88 ∧ x = 0.4 ∧ y = 0.6 :=  
by
  have h_eqns : ∀ x y : ℝ, x + y = 1 → 85 * x + 90 * y = 88 → x = 0.4 ∧ y = 0.6 := sorry
  exact ⟨0.4, 0.6, sorry, sorry, rfl, rfl⟩

-- Question 3: Compute composite scores and find top two candidates.
theorem find_top_two_candidates :
  ∀ (candidates : List (ℕ × ℕ)),
  candidates = [(92, 88), (84, 86), (90, 90), (84, 80), (80, 85)] →
  ∃ (top1 top2 : (ℕ × ℕ)),
    top1 = (90, 90) ∧ top2 = (92, 88) :=
by
  intro candidates h_candidates
  have h_comp_scores : ∀ (w : ℕ) (i : ℕ), (0.4 * w + 0.6 * i) ∈ [89.6, 85.2, 90, 81.6, 83] := sorry
  use ((90, 90)), ((92, 88))
  exact sorry

end median_and_mode_of_written_scores_composite_score_percentages_find_top_two_candidates_l289_289838


namespace washing_water_use_l289_289429

variable (gallons_collected : Nat)
variable (gallons_per_car : Nat)
variable (num_cars : Nat)
variable (less_gallons_plants : Nat)

-- Here are the conditions provided in the problem
def initial_gallons_collected := gallons_collected = 65
def water_used_per_car := gallons_per_car = 7
def number_of_cars := num_cars = 2
def less_gallons_for_plants := less_gallons_plants = 11

-- Calculate total water used for cars
def total_water_cars := num_cars * gallons_per_car
-- Calculate water used for plants
def water_plants := total_water_cars - less_gallons_plants
-- Calculate total used for cars and plants
def total_water_cars_plants := total_water_cars + water_plants
-- Calculate remaining water
def remaining_water := gallons_collected - total_water_cars_plants
-- Calculate water used to wash plates and clothes
def water_plates_clothes := remaining_water / 2

-- The theorem to prove the problem statement
theorem washing_water_use (hg : initial_gallons_collected) (hwc : water_used_per_car) (hnc : number_of_cars) (hlp : less_gallons_for_plants) :
  water_plates_clothes = 24 :=
by
  -- Given conditions from the problem
  unfold initial_gallons_collected water_used_per_car number_of_cars less_gallons_for_plants at hg hwc hnc hlp
  -- Definition lies outside the immediate scope of main code, thus accuracy over proof is ensured
  sorry

end washing_water_use_l289_289429


namespace minimum_photos_l289_289289

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289289


namespace martin_failed_by_200_marks_l289_289708

theorem martin_failed_by_200_marks (max_marks : ℝ) (required_percentage : ℝ) (marks_scored : ℝ) :
  max_marks = 500 → required_percentage = 0.8 → marks_scored = 200 →
  (required_percentage * max_marks - marks_scored) = 200 :=
by
  intros h_max h_req h_scored
  rw [h_max, h_req, h_scored]
  norm_num
  exact rfl

end martin_failed_by_200_marks_l289_289708


namespace can_color_all_cells_8x9_cannot_color_all_cells_8x10_l289_289385

-- Define the general setup for the problem
structure Rectangle where
  width : ℕ
  height : ℕ

def initial_condition (r : Rectangle) : Prop :=
  r.width > 0 ∧ r.height > 0

def colorable (r : Rectangle) : Prop :=
  ∃ (initially_colored: ℕ) (rule: (ℕ -> ℕ -> Prop)),
  initially_colored < (r.width * r.height) ∧
  (∀ i j, 0 ≤ i ∧ i < r.width ∧ 0 ≤ j ∧ j < r.height ->
          (i, j) ≠ (initially_colored / r.height, initially_colored % r.height) ->
          (rule initially_colored (i * r.height + j) ↔ odd (neighbors r.initially_colored (i * r.height + j))))

-- Specific rectangles
def r8x9 : Rectangle := { width := 8, height := 9 }
def r8x10 : Rectangle := { width := 8, height := 10 }

-- Theorems to prove
theorem can_color_all_cells_8x9 : colorable r8x9 := sorry
theorem cannot_color_all_cells_8x10 : ¬ colorable r8x10 := sorry

end can_color_all_cells_8x9_cannot_color_all_cells_8x10_l289_289385


namespace sum_of_dimensions_l289_289111

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 50) (h2 : A * C = 90) (h3 : B * C = 100) : A + B + C = 24 :=
  sorry

end sum_of_dimensions_l289_289111


namespace p_plus_q_eq_10_l289_289022

theorem p_plus_q_eq_10 (p q : ℕ) (hp : p > q) (hpq1 : p < 10) (hpq2 : q < 10)
  (h : p.factorial / q.factorial = 840) : p + q = 10 :=
by
  sorry

end p_plus_q_eq_10_l289_289022


namespace total_weight_of_wood_l289_289121

theorem total_weight_of_wood (m1 m_circle : ℕ) (s1 s2 d_circle : ℕ) 
  (h1 : m1 = 18) (h2 : s1 = 4) (h3 : s2 = 6) (h4 : m_circle = 15): 
  (m1 * (s2 ^ 2) / (s1 ^ 2) + m_circle = 55.5) :=
    by
  have area_ratio : (s2 ^ 2) / (s1 ^ 2) = 9 / 4 := by sorry
  have m2 : (m1 * (s2 ^ 2)) / (s1 ^ 2) = m1 * 9 / 4 := by sorry
  have second_triangle_weight : m1 * 9 / 4 = 40.5 := by sorry
  have total_weight : (40.5 + m_circle) = 55.5 := by
    rw [h1, h4]
    sorry
  exact total_weight

end total_weight_of_wood_l289_289121


namespace sum_first_100_odd_integers_l289_289460

theorem sum_first_100_odd_integers : 
  (∑ k in Finset.range 100, (2 * (k + 1) - 1)) = 10000 := 
  by sorry

end sum_first_100_odd_integers_l289_289460


namespace correct_statements_l289_289066

-- Define the conditions as functions/types

-- Statement 1: A right triangle has three altitudes
def right_triangle_altitudes (T : Triangle) : Prop :=
  ∃ a b c : Altitude, a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Statement 2: Correct formula for number of diagonals in an n-sided polygon
def correct_diagonal_formula (n : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (diagonals n = n * (n - 3)) / 2 -- pseudo-function for diagonal calculation

-- Statement 3: Two circles with equal radii are congruent
def equal_radii_circles_congruent : Prop :=
  ∀ c1 c2 : Circle, c1.radius = c2.radius → c1.congruent c2

-- Statement 4: Polygon with all sides equal is regular
def equilateral_polygon_regular (P : Polygon) : Prop :=
  P.equilateral → P.regular

-- Statement 5: Definition of a circle
def circle_definition (C : Circle) : Prop :=
  ∀ (p : Point), dist C.center p = C.radius → p ∈ C.points

-- Theorem combining correct and incorrect statements
theorem correct_statements :
  ¬right_triangle_altitudes ∧
  ¬correct_diagonal_formula ∧
  equal_radii_circles_congruent ∧
  ¬equilateral_polygon_regular ∧
  circle_definition :=
by
  sorry

end correct_statements_l289_289066


namespace shortest_chord_length_l289_289766

def line_eq (m x y : ℝ) : ℝ := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4

def circle_eq (x y : ℝ) : ℝ := (x - 1)^2 + (y - 2)^2 - 25

theorem shortest_chord_length (m : ℝ) :
  ∃ (L : ℝ), (L = sqrt 20 * 4) → 
  let d := abs ((2 * m + 1) * 1 + (m + 1) * 2 - (7 * m + 4)) / sqrt ((2 * m + 1)^2 + (m + 1)^2) in
  let chord_length := 2 * sqrt (25 - d^2) in
  chord_length = 4 * sqrt 5 :=
begin
  sorry
end

end shortest_chord_length_l289_289766


namespace problem_l289_289733

theorem problem (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := by
  sorry

end problem_l289_289733


namespace incorrect_operation_l289_289467

theorem incorrect_operation 
    (x y : ℝ) :
    (x - y) / (x + y) = (y - x) / (y + x) ↔ False := 
by 
  sorry

end incorrect_operation_l289_289467


namespace pyramid_top_plus_ways_l289_289264

-- Define the signs and the pyramid rules
@[derive DecidableEq]
inductive Sign
| plus  : Sign
| minus : Sign

open Sign

def sign_mult (s1 : Sign) (s2 : Sign) : Sign :=
  match s1, s2 with
  | plus, plus => plus
  | minus, minus => plus
  | _, _ => minus

-- Define the condition when the top of the pyramid is "+"
def pyramid_top (a b c d e : Sign) : Prop :=
  let s2 := λ x y => sign_mult x y
  let ab := s2 a b
  let bc := s2 b c
  let cd := s2 c d
  let de := s2 d e
  let abc := s2 ab bc
  let bcd := s2 bc cd
  let cde := s2 cd de
  let abcd := s2 abc bcd
  let bcde := s2 bcd cde
  sign_mult abcd bcde = plus

-- Calculate the number of valid combinations
def valid_combinations : ℕ :=
  let values := [plus, minus]
  values.product values.product values.product values.product values
  .filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩ => pyramid_top a b c d e)
  .length

theorem pyramid_top_plus_ways : valid_combinations = 16 := 
sorry

end pyramid_top_plus_ways_l289_289264


namespace find_value_of_a2004_b2004_l289_289214

-- Given Definitions and Conditions
def a : ℝ := sorry
def b : ℝ := sorry
def A : Set ℝ := {a, a^2, a * b}
def B : Set ℝ := {1, a, b}

-- The theorem statement
theorem find_value_of_a2004_b2004 (h : A = B) : a ^ 2004 + b ^ 2004 = 1 :=
sorry

end find_value_of_a2004_b2004_l289_289214


namespace max_gcd_lcm_condition_l289_289719

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l289_289719


namespace mason_internet_speed_l289_289373

-- Definitions based on the conditions
def total_data : ℕ := 880
def downloaded_data : ℕ := 310
def remaining_time : ℕ := 190

-- Statement: The speed of Mason's Internet connection after it slows down
theorem mason_internet_speed :
  (total_data - downloaded_data) / remaining_time = 3 :=
by
  sorry

end mason_internet_speed_l289_289373


namespace parallelogram_area_591_92_l289_289175

noncomputable def parallelogram_area (base slant_height : ℝ) (angle : ℝ) : ℝ :=
  let height := base / Real.cos (angle * Real.pi / 180) in
  base * height

theorem parallelogram_area_591_92 :
  parallelogram_area 22 18 35 ≈ 591.92 :=
by
  sorry

end parallelogram_area_591_92_l289_289175


namespace pencils_to_sell_for_desired_profit_l289_289516

/-- Definitions based on the conditions provided in the problem. -/
def total_pencils : ℕ := 2000
def cost_per_pencil : ℝ := 0.20
def sell_price_per_pencil : ℝ := 0.40
def desired_profit : ℝ := 160
def total_cost : ℝ := total_pencils * cost_per_pencil

/-- The theorem considers all the conditions and asks to prove the number of pencils to sell -/
theorem pencils_to_sell_for_desired_profit : 
  (desired_profit + total_cost) / sell_price_per_pencil = 1400 :=
by 
  sorry

end pencils_to_sell_for_desired_profit_l289_289516


namespace intersection_eq_l289_289996

open set

variable (x : ℝ)

def A : set ℝ := { x | 3 * x - x^2 > 0 }
def B : set ℝ := { x | 0 ≤ sqrt (1 - x) }

theorem intersection_eq :
  A ∩ B = { x | 0 < x ∧ x ≤ 1 } := sorry

end intersection_eq_l289_289996


namespace pq_sum_eq_21_l289_289679

variable {α : Type _}

def M (p : ℕ) : Set ℕ := {x | x^2 - p * x + 6 = 0}
def N (q : ℕ) : Set ℕ := {x | x^2 + 6 * x - q = 0}

theorem pq_sum_eq_21 (p q : ℕ) (h : M p ∩ N q = {2}) : p + q = 21 := by
  have hp : 2 ∈ M p := by rw [Set.mem_inter_iff] at h; exact h.1.1
  have hq : 2 ∈ N q := by rw [Set.mem_inter_iff] at h; exact h.1.2
  have hp_eq : 2^2 - 2 * p + 6 = 0 := by exact hp
  have hq_eq : 2^2 + 6 * 2 - q = 0 := by exact hq
  sorry

end pq_sum_eq_21_l289_289679


namespace max_gcd_lcm_l289_289715

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l289_289715


namespace profit_percentage_correct_l289_289508

noncomputable def CP : ℝ := 460
noncomputable def SP : ℝ := 542.8
noncomputable def profit : ℝ := SP - CP
noncomputable def profit_percentage : ℝ := (profit / CP) * 100

theorem profit_percentage_correct :
  profit_percentage = 18 := by
  sorry

end profit_percentage_correct_l289_289508


namespace minimum_photos_l289_289286

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289286


namespace find_K_find_t_l289_289636

-- Proof Problem for G9.2
theorem find_K (x : ℚ) (K : ℚ) (h1 : x = 1.9898989) (h2 : x - 1 = K / 99) : K = 98 :=
sorry

-- Proof Problem for G9.3
theorem find_t (p q r t : ℚ)
  (h_avg1 : (p + q + r) / 3 = 18)
  (h_avg2 : ((p + 1) + (q - 2) + (r + 3) + t) / 4 = 19) : t = 20 :=
sorry

end find_K_find_t_l289_289636


namespace average_people_per_hour_l289_289663

theorem average_people_per_hour :
  let people := 5000
  let days := 5
  let hours_per_day := 24
  let total_hours := days * hours_per_day 
  let average := people / total_hours.toFloat
  float.round average = 42 := 
by 
  sorry

end average_people_per_hour_l289_289663


namespace correct_statements_l289_289468

variable (population_size : ℕ)
variable (sample_size : ℕ)
variable (individual_selected_probability : ℝ)
variable (data_set_1 : List ℝ)
variable (data_set_2 : List ℝ)
variable (data_set_average : ℝ)
variable (data_set_variance : ℝ)
variable (standard_deviation_original : ℝ)
variable (standard_deviation_transformed : ℝ)

def statement_A_correct := 
  (population_size = 50) → 
  (sample_size = 10) → 
  (individual_selected_probability = 0.2)

def statement_B_correct := 
  (data_set_1 = [1, 2, 4, 6, 7]) → 
  (data_set_average = 4) → 
  (data_set_variance = 5)

def statement_C_correct := 
  (data_set_2 = [27, 12, 14, 30, 15, 17, 19, 23]) → 
  (data_set_variance = 17)

def statement_D_correct := 
  (standard_deviation_original = 8) → 
  (standard_deviation_transformed = 16)

theorem correct_statements :
  (statement_A_correct → statement_D_correct) :=
sorry

end correct_statements_l289_289468


namespace problem_1_problem_2_l289_289969

noncomputable def S (n : ℕ) (a : ℕ → ℚ) := (4 : ℚ)/3 - (1 : ℚ)/3 * a n

theorem problem_1 (a : ℕ → ℚ) (S_n : ℕ → ℚ) 
  (h : ∀ n, S_n n = 4/3 - 1/3 * a n) :
  ∀ n, a n = (1/4) ^ (n-1) :=
sorry

noncomputable def b (a : ℕ → ℚ) (n : ℕ) := Real.logb (1/2) (a (n + 1))

theorem problem_2 (a : ℕ → ℚ) (S_n : ℕ → ℚ) (b_n : ℕ → ℚ)
  (h1 : ∀ n, S_n n = 4/3 - 1/3 * a n)
  (h2 : ∀ n, b_n n = Real.logb (1/2) (a (n + 1))) :
  ∀ n, ∑ i in Finset.range n, (1 / (b_n i * b_n (i + 1))) = n / (b_n 0 * b_n n) :=
sorry

end problem_1_problem_2_l289_289969


namespace integral_BC_values_count_l289_289693

noncomputable def triangle_side_counts (AB BC AC : ℝ) (h1 : AB = 7) (h2 : AC = 2 * AB) (h3 : 7 < BC) (h4 : BC < 21) : ℕ :=
  let eligible_BC_values := {n : ℕ | 7 < n ∧ n < 21}.card
  eligible_BC_values

theorem integral_BC_values_count : ∀ AB BC AC, 
  AB = 7 →
  AC = 2 * AB →
  7 < BC →
  BC < 21 →
  triangle_side_counts AB BC AC 7 (2 * 7) bc bc = 13 :=
by
  intros AB BC AC h1 h2 h3 h4
  rw [triangle_side_counts]
  sorry

end integral_BC_values_count_l289_289693


namespace problem_common_sum_l289_289020

theorem problem_common_sum :
  ∃ (matrix : Matrix (Fin 6) (Fin 6) ℤ), 
    let elements := filter (λ n, -12 ≤ n ∧ n ≤ 18) (List.range (31)) in
    ∀ i : Fin 6, 
      (Finset.univ.sum (λ j, matrix i j) = 15.5) ∧ -- Sum of each row
      (Finset.univ.sum (λ j, matrix j i) = 15.5) ∧ -- Sum of each column
      (Finset.univ.sum (λ k, matrix k k) = 15.5) ∧ -- Main diagonal 1
      (Finset.univ.sum (λ k, matrix k (5 - k)) = 15.5)  -- Main diagonal 2
:= sorry

end problem_common_sum_l289_289020


namespace nate_pages_left_to_read_l289_289382

-- Define the constants and conditions
def total_pages : ℕ := 400
def percentage_read : ℕ := 20

-- Calculate the number of pages already read
def pages_read := total_pages * percentage_read / 100

-- Calculate the number of pages left
def pages_left := total_pages - pages_read

-- Statement to prove
theorem nate_pages_left_to_read : pages_left = 320 :=
by {
  unfold pages_read,
  unfold pages_left,
  simp,
  sorry -- The proof will be filled in based on the calculations in the solution.
}

end nate_pages_left_to_read_l289_289382


namespace find_b_l289_289269

theorem find_b (a b : ℝ) (h₁ : ∀ x y, y = 0.75 * x + 1 → (4, b) = (x, y))
                (h₂ : k = 0.75) : b = 4 :=
by sorry

end find_b_l289_289269


namespace complex_quadrant_example_l289_289977

open Complex

def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_quadrant_example (z : ℂ) (h : (1 - I) * z = (1 + I) ^ 2) : in_second_quadrant z :=
by
  sorry

end complex_quadrant_example_l289_289977


namespace distance_from_focus_to_directrix_l289_289564

-- Parabola definitions
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Standard form relation
axiom standard_form_relation (x : ℝ) : ∃ (p : ℝ), parabola(x) = 4 * p * x^2

-- Distance from focus to directrix
theorem distance_from_focus_to_directrix : 
  ∃ (p : ℝ), p = 1 / 8 ∧ ∀ (x : ℝ), parabola(x) = 4 * p * x^2 :=
by
  sorry

end distance_from_focus_to_directrix_l289_289564


namespace point_in_shaded_region_l289_289383

open Set

-- Define the circles
def Circle1 : set ℝ := {p : ℝ × ℝ | (p.1 - x1)^2 + (p.2 - y1)^2 = r1^2}
def Circle2 : set ℝ := {p : ℝ × ℝ | (p.1 - x2)^2 + (p.2 - y2)^2 = r2^2}

-- Define the external tangents to both circles
def ext_tangents (C1 C2 : set ℝ) : set (set ℝ) := sorry 

-- Define the shaded regions determined by external tangents but excluding the tangent lines themselves
def shaded_region (C1 C2 : set ℝ) : set ℝ := { p : ℝ × ℝ | ∃ l ∈ ext_tangents C1 C2, p ∉ l }

-- The point \( M \) that satisfies the given condition
def locus_of_M (C1 C2 : set ℝ) : set ℝ := shaded_region C1 C2

-- Main theorem statement
theorem point_in_shaded_region (C1 C2 : set ℝ) (C1_non_overlap_C2 : disjoint C1 C2) :
  ∀ M : ℝ × ℝ, M ∈ locus_of_M C1 C2 → 
  (∀ l : set ℝ, l ∈ {l : set ℝ | ∃ p ∈ shaded_region C1 C2, p ∉ l} → ∃ p : ℝ × ℝ, p ∈ Circle1 ∨ p ∈ Circle2) := 
sorry

end point_in_shaded_region_l289_289383


namespace distinct_values_of_d_l289_289688

theorem distinct_values_of_d (d a b c : ℂ) (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_eq: ∀ z : ℂ, (z - a) * (z - b) * (z - c) = (z - d^2 * a) * (z - d^2 * b) * (z - d^2 * c)) :
  {d : ℂ | ∃ a b c : ℂ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ∀ z : ℂ, (z - a) * (z - b) * (z - c) = 
  (z - d^2 * a) * (z - d^2 * b) * (z - d^2 * c)}.to_finset.card = 6 :=
sorry

end distinct_values_of_d_l289_289688


namespace minimum_photos_l289_289285

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289285


namespace sum_of_coordinates_l289_289972

noncomputable def g : ℝ → ℝ := sorry

theorem sum_of_coordinates : g 2 = 3 → let x := 1 in let y := 2 * g (2 * x) + 4 in x + y = 11 :=
by
  intro h
  simp [h]
  let x := 1
  let y := 2 * g (2 * x) + 4
  have h2 : g 2 = 3 := h
  rw [h2, mul_add, mul_add]
  norm_num
  sorry

end sum_of_coordinates_l289_289972


namespace interest_rate_for_lending_l289_289104

def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ :=
  (P * R * T) / 100

theorem interest_rate_for_lending :
  ∀ (P T R_b G R_l : ℕ),
  P = 20000 →
  T = 6 →
  R_b = 8 →
  G = 200 →
  simple_interest P R_b T + G * T = simple_interest P R_l T →
  R_l = 9 :=
by
  intros P T R_b G R_l
  sorry

end interest_rate_for_lending_l289_289104


namespace equation_of_reflected_light_ray_l289_289097

-- Define points P and Q
def P : ℝ × ℝ := (2, 3)
def P' : ℝ × ℝ := (2, -3)
def Q : ℝ × ℝ := (1, 1)

-- Define the line equation for reflected ray
theorem equation_of_reflected_light_ray :
  ∃ (a b c : ℝ), a * (P'.fst - Q.fst) + b * (P'.snd - Q.snd) = c ∧
                 4 = a ∧ 1 = b ∧ -5 = c :=
by
  use [4, 1, -5]
  -- Prove the equation is consistent with points P' and Q
  sorry

end equation_of_reflected_light_ray_l289_289097


namespace rons_pick_times_l289_289392

def total_members(couples single_people : ℕ) : ℕ := couples * 2 + single_people + 2

def times_rons_pick(total_members : ℕ) : ℕ := 52 / total_members

theorem rons_pick_times
    (couples single_people : ℕ)
    (h_couples : couples = 3)
    (h_single_people : single_people = 5) :
    times_rons_pick (total_members couples single_people) = 4 :=
by
  have h_total_members : total_members couples single_people = 13 := by
    simp [total_members, h_couples, h_single_people]
  simp [times_rons_pick, h_total_members]
  sorry

end rons_pick_times_l289_289392


namespace y_power_x_equals_49_l289_289241

theorem y_power_x_equals_49 (x y : ℝ) (h : |x - 2| = -(y + 7)^2) : y ^ x = 49 := by
  sorry

end y_power_x_equals_49_l289_289241


namespace shaded_rectangles_area_l289_289481

def sum_areas_of_shaded_rectangles (area_of_square : ℝ) (num_rectangles : ℕ) : ℝ :=
  2 * (area_of_square / 16) * 2

theorem shaded_rectangles_area (area_of_square : ℝ) (num_rectangles : ℕ) (h1 : area_of_square = 64) (h2 : num_rectangles = 2) : 
   sum_areas_of_shaded_rectangles area_of_square num_rectangles = 16 := 
by
  -- stubbed proof
  sorry

end shaded_rectangles_area_l289_289481


namespace sugar_percentage_l289_289831

theorem sugar_percentage 
  (initial_volume : ℝ) (initial_water_perc : ℝ) (initial_kola_perc: ℝ) (added_sugar : ℝ) (added_water : ℝ) (added_kola : ℝ)
  (initial_solution: initial_volume = 340) 
  (perc_water : initial_water_perc = 0.75) 
  (perc_kola: initial_kola_perc = 0.05)
  (added_sugar_amt : added_sugar = 3.2) 
  (added_water_amt : added_water = 12) 
  (added_kola_amt : added_kola = 6.8) : 
  (71.2 / 362) * 100 = 19.67 := 
by 
  sorry

end sugar_percentage_l289_289831


namespace inequality_proof_l289_289581

theorem inequality_proof
  (a b c A α : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α)
  (h_sum : a + b + c = A)
  (h_A : A ≤ 1) :
  (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ≥ 3 * (3 / A - A / 3) ^ α :=
by
  sorry

end inequality_proof_l289_289581


namespace solve_quadratic_equation_l289_289402

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end solve_quadratic_equation_l289_289402


namespace ratio_knapsack_to_cloth_bag_l289_289008

def dozen := 12

def total_peaches : ℕ := 5 * dozen

def peaches_in_knapsack : ℕ := 12

def peaches_in_each_cloth_bag : ℕ := (total_peaches - peaches_in_knapsack) / 2

theorem ratio_knapsack_to_cloth_bag : (peaches_in_knapsack  : peaches_in_each_cloth_bag) = (1 : 2) :=
by {
  sorry
}

end ratio_knapsack_to_cloth_bag_l289_289008


namespace probability_white_marble_l289_289833

theorem probability_white_marble :
  ∀ (p_blue p_green p_white : ℝ),
    p_blue = 0.25 →
    p_green = 0.4 →
    p_blue + p_green + p_white = 1 →
    p_white = 0.35 :=
by
  intros p_blue p_green p_white h_blue h_green h_total
  sorry

end probability_white_marble_l289_289833


namespace ratio_cereal_A_to_B_l289_289473

-- Definitions translated from conditions
def sugar_percentage_A : ℕ := 10
def sugar_percentage_B : ℕ := 2
def desired_sugar_percentage : ℕ := 6

-- The theorem based on the question and correct answer
theorem ratio_cereal_A_to_B :
  let difference_A := sugar_percentage_A - desired_sugar_percentage
  let difference_B := desired_sugar_percentage - sugar_percentage_B
  difference_A = 4 ∧ difference_B = 4 → 
  difference_B / difference_A = 1 :=
by
  intros
  sorry

end ratio_cereal_A_to_B_l289_289473


namespace eval_expression_l289_289557

theorem eval_expression : (825 * 825) - (824 * 826) = 1 := by
  sorry

end eval_expression_l289_289557


namespace hyperbola_foci_distance_l289_289013

theorem hyperbola_foci_distance :
  (∃ (h : ℝ → ℝ) (c : ℝ), (∀ x, h x = 2 * x + 3 ∨ h x = 1 - 2 * x)
    ∧ (h 4 = 5)
    ∧ 2 * Real.sqrt (20.25 + 4.444) = 2 * Real.sqrt 24.694) := 
  sorry

end hyperbola_foci_distance_l289_289013


namespace minimum_photos_l289_289282

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289282


namespace larger_number_is_eight_l289_289073

theorem larger_number_is_eight (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l289_289073


namespace polynomial_over_Q_of_division_l289_289822

theorem polynomial_over_Q_of_division (A B : Polynomial ℚ) (Q : Polynomial ℝ) (h : A = B * Q) : Q ∈ Polynomial ℚ :=
  sorry

end polynomial_over_Q_of_division_l289_289822


namespace tan_inequality_l289_289811

theorem tan_inequality : tan 4 > tan 3 := by
  sorry

end tan_inequality_l289_289811


namespace computation_problems_count_l289_289118

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end computation_problems_count_l289_289118


namespace liam_more_than_200_paperclips_on_thursday_l289_289706

def sequence (n : ℕ) : ℕ := 5 * 3 ^ n

theorem liam_more_than_200_paperclips_on_thursday :
  ∃ n : ℕ, sequence n > 200 ∧ n = 4 :=
by {
  use 4,
  split,
  sorry, -- This is to indicate proving 5 * 3 ^ 4 > 200
  rfl
}

end liam_more_than_200_paperclips_on_thursday_l289_289706


namespace exists_unique_k_l289_289577

namespace SequenceProof

def is_seq (n : ℕ) (s : list (fin n → bool)) : Prop :=
  s.length = 2 ^ (n - 1)

def satisfies_condition (s : fin n → bool) (t : fin n → bool) (u : fin n → bool) : Prop :=
  ∃ m : fin n, s m = tt ∧ t m = tt ∧ u m = tt

theorem exists_unique_k
  (n : ℕ)
  (s : list (fin n → bool))
  (h_s : is_seq n s)
  (seq_condition : ∀ (x y z : fin n → bool), x ∈ s → y ∈ s → z ∈ s → satisfies_condition x y z) :
  ∃! k : fin n, ∀ x : fin n → bool, x ∈ s → x k = tt :=
sorry

end SequenceProof

end exists_unique_k_l289_289577


namespace bananas_proof_l289_289865

noncomputable def number_of_bananas (total_oranges : ℕ) (total_fruits_percent_good : ℝ) 
  (percent_rotten_oranges : ℝ) (percent_rotten_bananas : ℝ) : ℕ := 448

theorem bananas_proof :
  let total_oranges := 600
  let percent_rotten_oranges := 0.15
  let percent_rotten_bananas := 0.08
  let total_fruits_percent_good := 0.878
  
  number_of_bananas total_oranges total_fruits_percent_good percent_rotten_oranges percent_rotten_bananas = 448 :=
by
  sorry

end bananas_proof_l289_289865


namespace find_digits_l289_289574

-- Define the digits range
def is_digit (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 9

-- Define the five-digit numbers
def num_abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 111 * c
def num_abbbb (a b : ℕ) : ℕ := 10000 * a + 1111 * b

-- Problem statement
theorem find_digits (a b c : ℕ) (h_da : is_digit a) (h_db : is_digit b) (h_dc : is_digit c) :
  (num_abccc a b c) + 1 = (num_abbbb a b) ↔
  (a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0) :=
sorry

end find_digits_l289_289574


namespace minimum_photos_taken_l289_289309

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289309


namespace positive_difference_between_max_and_min_enrollment_l289_289790

theorem positive_difference_between_max_and_min_enrollment : 
  let varsity_enrollment := 1500 
  let northwest_enrollment := 1800 
  let central_enrollment := 2400 
  let greenbriar_enrollment := 2150 
  let maplewood_enrollment := 1000 in
  max varsity_enrollment (max northwest_enrollment (max central_enrollment (max greenbriar_enrollment maplewood_enrollment))) - 
  min varsity_enrollment (min northwest_enrollment (min central_enrollment (min greenbriar_enrollment maplewood_enrollment))) = 1400 :=
by
  simp [max, min]
  sorry

end positive_difference_between_max_and_min_enrollment_l289_289790


namespace latest_time_for_60_degrees_l289_289255

def temperature_at_time (t : ℝ) : ℝ :=
  -2 * t^2 + 16 * t + 40

theorem latest_time_for_60_degrees (t : ℝ) :
  temperature_at_time t = 60 → t = 5 :=
sorry

end latest_time_for_60_degrees_l289_289255


namespace max_mass_sand_is_correct_l289_289507

def platform_length := 8 -- Length of the platform in meters
def platform_width := 4 -- Width of the platform in meters
def angle_max := Real.pi / 4 -- Maximum angle in radians (45 degrees)
def sand_density := 1500 -- Density of sand in kg/m³

noncomputable def max_mass_sand : ℝ :=
  let height := platform_width / 2
  let volume_prism := platform_length * platform_width * height
  let volume_pyramid := (2 / 3) * (platform_width / 2) * platform_length * height
  (volume_prism + volume_pyramid) * sand_density

theorem max_mass_sand_is_correct : max_mass_sand = 112000 := by
  sorry

end max_mass_sand_is_correct_l289_289507


namespace diagonal_of_square_with_perimeter_40_l289_289565

theorem diagonal_of_square_with_perimeter_40 :
  ∀ (s : ℝ), 4 * s = 40 → real.sqrt (s^2 + s^2) = 10 * real.sqrt 2 :=
by
  intros s h
  sorry

end diagonal_of_square_with_perimeter_40_l289_289565


namespace work_completion_time_l289_289099

theorem work_completion_time {M W B : ℝ} (hM : M = 1/6) (hW : W = 1/36) (hB : B = 1/18) :
  1 / (M + W + B) = 4 := 
by 
  calc
    1 / (1/6 + 1/36 + 1/18) = 1 / (6/36 + 1/36 + 2/36) : by rw [hM, hW, hB]
    ... = 1 / (9/36) : sorry
    ... = 1 / (1/4) : sorry
    ... = 4 : sorry

end work_completion_time_l289_289099


namespace geometric_progression_common_ratio_l289_289326

variable (x : ℝ)

noncomputable def a1 := x / 2
noncomputable def a2 := 2 * x - 3
noncomputable def a3 := 18 / x + 1

theorem geometric_progression_common_ratio :
  a1 * a3 = a2^2 → x = 25 / 8 → a2 / a1 = 2.08 :=
by sorry

end geometric_progression_common_ratio_l289_289326


namespace coefficient_of_linear_term_l289_289408

def polynomial (x : ℝ) := x^2 - 2 * x - 3

theorem coefficient_of_linear_term : (∀ x : ℝ, polynomial x = x^2 - 2 * x - 3) → -2 = -2 := by
  intro h
  sorry

end coefficient_of_linear_term_l289_289408


namespace similar_triangles_length_KL_l289_289484

variable {G H I J K L : Type} [MetricSpace G] [MetricSpace H] [MetricSpace I] [MetricSpace J] [MetricSpace K] [MetricSpace L]

-- Given a similarity relationship between the triangles
variable (similar : ∀ {a b c d e f : Type} [MetricSpace a] [MetricSpace b] [MetricSpace c] 
                    [MetricSpace d] [MetricSpace e] [MetricSpace f], 
                    Triangle a b c → Triangle d e f → Prop)

-- Define the specific triangle side lengths in cm
variable (GH : ℝ) (HI : ℝ) (JK : ℝ) (KL : ℝ)
variable (GH_val : GH = 10) (HI_val : HI = 7) (JK_val : JK = 4)

-- We want to prove that KL = 2.8 cm given the similarity of the triangles
theorem similar_triangles_length_KL : 
  similar (Triangle G H I) (Triangle J K L) →
  GH = 10 → HI = 7 → JK = 4 → 
  KL = 2.8 :=
by
  sorry

end similar_triangles_length_KL_l289_289484


namespace marie_lost_erasers_l289_289370

def initialErasers : ℕ := 95
def finalErasers : ℕ := 53

theorem marie_lost_erasers : initialErasers - finalErasers = 42 := by
  sorry

end marie_lost_erasers_l289_289370


namespace last_digit_is_two_l289_289724

-- Define the initial conditions and the rules of the game
def initial_board : list ℕ := list.repeat 1 10 ++ list.repeat 2 10

def move (board : list ℕ) (i j : ℕ) : list ℕ :=
  if board.nth i = board.nth j then
    board.erase_nth i.erase_nth j ++ [2]
  else
    board.erase_nth i.erase_nth j ++ [1]

-- The game invariant is that the parity of the number of ones remains even
def parity_invariant : list ℕ → Prop
| []     => true
| [_]    => true
| (h :: t) => (h = 1 → (t.filter (λ x, x = 1)).length % 2 = 0) ∧ parity_invariant t

lemma parity_invariance (board : list ℕ) (i j : ℕ) (h : parity_invariant board) :
  parity_invariant (move board i j) := sorry

theorem last_digit_is_two :
  ∃ (d : ℕ), (d = 1 → false) ∧ (d = 2 → true) :=
begin
  -- Base our proof on the parity invariant of ones being even
  have h : parity_invariant initial_board,
  { sorry }, -- Initial parity invariant proof

  -- Using induction or other proof techniques
  apply exists.intro 2,
  split,
  { intro h1,
    unfold initial_board at h1,
    -- Contradiction by parity invariant
    sorry },
  { intro h2,
    -- This is trivially true
    trivial }
end

end last_digit_is_two_l289_289724


namespace complement_event_l289_289105

-- Definitions based on conditions
variables (shoot1 shoot2 : Prop) -- shoots the target on the first and second attempt

-- Definition based on the question and answer
def hits_at_least_once : Prop := shoot1 ∨ shoot2
def misses_both_times : Prop := ¬shoot1 ∧ ¬shoot2

-- Theorem statement based on the mathematical translation
theorem complement_event :
  misses_both_times shoot1 shoot2 = ¬hits_at_least_once shoot1 shoot2 :=
by sorry

end complement_event_l289_289105


namespace analytic_expression_f_exists_b_inequality_l289_289956

theorem analytic_expression_f (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x, 2 * f (x + 2) = f x) 
  (h₂ : ∀ x, (0 < x ∧ x < 2) → f x = ln x + a * x ∧ a < -1 / 2) 
  (h₃ : f (-4) = -4 ∧ ∀ x, (-4 < x ∧ x < -2) → f x ≤ f (-4)) :
  ∀ x, (0 < x ∧ x < 2) → f x = ln x - x :=
by
  sorry

theorem exists_b_inequality (f : ℝ → ℝ) (h_f : ∀ x, (0 < x ∧ x < 2) → f x = ln x - x) :
  ∃ b : ℝ, (b = 1) ∧ (∀ x, ((0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2)) → (x - b) / (f x + x) > sqrt x) :=
by
  sorry

end analytic_expression_f_exists_b_inequality_l289_289956


namespace max_triangles_area_1_l289_289960

def count_triangles (S : set (ℝ × ℝ)) : ℕ :=
  -- This function is used as a placeholder definition.
  sorry

theorem max_triangles_area_1 (S : set (ℝ × ℝ)) (h_points: S.finite) (h_no_collinear : ∀ p1 p2 p3 ∈ S, p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1 → ¬ collinear ℝ (λ p, (p.1, p.2)) p1 p2 p3) : 
  count_triangles S ≤ (2 * (S.to_finset.card) * (S.to_finset.card - 1) / 3) := 
  sorry

end max_triangles_area_1_l289_289960


namespace min_photographs_42_tourists_3_monuments_l289_289271

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289271


namespace average_temperature_l289_289448

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end average_temperature_l289_289448


namespace min_value_fraction_l289_289233

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  ∀ x, (x = (1 / m + 8 / n)) → x ≥ 18 :=
by
  sorry

end min_value_fraction_l289_289233


namespace min_photographs_42_tourists_3_monuments_l289_289275

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289275


namespace solve_sqrt_eqn_l289_289173

theorem solve_sqrt_eqn (z : ℝ) (h : sqrt (9 + 3 * z) = 12) : z = 45 :=
by
  sorry

end solve_sqrt_eqn_l289_289173


namespace harriet_ran_48_miles_l289_289576

def total_distance : ℕ := 195
def katarina_distance : ℕ := 51
def equal_distance (n : ℕ) : Prop := (total_distance - katarina_distance) = 3 * n
def harriet_distance : ℕ := 48

theorem harriet_ran_48_miles
  (total_eq : total_distance = 195)
  (kat_eq : katarina_distance = 51)
  (equal_dist_eq : equal_distance harriet_distance) :
  harriet_distance = 48 :=
by
  sorry

end harriet_ran_48_miles_l289_289576


namespace relationship_a_b_c_l289_289160

noncomputable def a : ℝ := 6 ^ 0.7
noncomputable def b : ℝ := 0.7 ^ 6
noncomputable def c : ℝ := Real.log 6 / Real.log 0.7

theorem relationship_a_b_c : c < b ∧ b < a := 
by {
  sorry
}

end relationship_a_b_c_l289_289160


namespace total_cost_proof_l289_289753

-- Define the cost of items
def cost_of_1kg_of_mango (M : ℚ) : Prop := sorry
def cost_of_1kg_of_rice (R : ℚ) : Prop := sorry
def cost_of_1kg_of_flour (F : ℚ) : Prop := F = 23

-- Condition 1: cost of some kg of mangos is equal to the cost of 24 kg of rice
def condition1 (M R : ℚ) (x : ℚ) : Prop := M * x = R * 24

-- Condition 2: cost of 6 kg of flour equals to the cost of 2 kg of rice
def condition2 (R : ℚ) : Prop := 23 * 6 = R * 2

-- Final proof problem
theorem total_cost_proof (M R F : ℚ) (x : ℚ) 
  (h1: condition1 M R x) 
  (h2: condition2 R) 
  (h3: cost_of_1kg_of_flour F) :
  4 * (69 * 24 / x) + 3 * R + 5 * 23 = 1978 :=
sorry

end total_cost_proof_l289_289753


namespace calculate_present_value_l289_289913

-- Define the parameters and constants
def future_value : ℝ := 750000
def annual_rate : ℝ := 0.045
def periods_per_year : ℤ := 2
def years : ℤ := 15
def expected_present_value : ℝ := 392946.77

-- The calculation for present value
noncomputable def present_value (FV r : ℝ) (n t : ℤ) : ℝ :=
  FV / (1 + r / n.to_real) ^ (n * t)

-- The proof goal statement
theorem calculate_present_value :
  present_value future_value annual_rate periods_per_year years = expected_present_value :=
by
  sorry

end calculate_present_value_l289_289913


namespace Xiao_Tian_hat_l289_289934

theorem Xiao_Tian_hat:
  ∃ (hats : Finset ℕ), 
    (∀ h ∈ hats, h ∈ {1, 2, 3, 4, 5}) ∧
    hats.card = 5 ∧
    ∀ (persons : Finset ℕ),
      persons = {Xiao Wang, Xiao Kong, Xiao Tian, Xiao Yan, Xiao Wei} ∧
      Xiao_Wang ∉ persons  ∧
	  persons = {1, 2, 3, 4, 5} ∧
	  (∀ w ∈ persons, w ≠ Xiao_Wang)∧
      ∃ (h8 : ℕ → ℕ), 
      function.bijective h8 ∧ 
      (h8 Xiao_Wang = 4) ∧
      (h8 Xiao_Kong ∈ {1, 2, 3, 4, 5}) ∧
      (h8 Xiao_Tian ≠ 3) ∧ (h8 Xiao_Tian ∈ {1, 2, 3, 4, 5}) ∧
      (h8 Xiao_Yan ∈ {1, 2, 3, 4, 5}) ∧ (h8 Xiao_Yan ≠ 3)∧
      (h8 Xiao_Wei ∈ {1, 2, 3, 4, 5}) ∧ (h8 Xiao_Wei = 3 ∨ h8 Xiao_Wei = 2)∧
      Xiao_Wei ≠ Xiao_Tian ∧ 
      h8 Xiao_Wang ≠ h8 Xiao_Tian,
      h8 Xiao_Tian = 2 
      := sorry

end Xiao_Tian_hat_l289_289934


namespace number_of_correct_statements_l289_289525

def statement1 (m : ℝ) : Prop := ¬ (m ∈ ℚ ∧ m ∈ ℝ)
def statement2 (a b : ℝ) : Prop := ¬ ((a > b) ↔ (a^2 > b^2))
def statement3 (x : ℝ) : Prop := ¬ ((x = 3) ↔ (x^2 - 2*x - 3 = 0))
def statement4 (A B : Set ℕ) [DecidableEq ℕ] : Prop := ¬ ((A ∩ B = B) ↔ (A = ∅))

theorem number_of_correct_statements : 
  ∀ (m : ℝ) (a b : ℝ) (x : ℝ) (A B : Set ℕ) [DecidableEq ℕ],
    (statement1 m) ∧ (statement2 a b) ∧ (statement3 x) ∧ (statement4 A B) -> 0 = 0 := 
by
  intros
  sorry

end number_of_correct_statements_l289_289525


namespace largest_square_not_divisible_by_100_l289_289158

theorem largest_square_not_divisible_by_100
  (n : ℕ) (h1 : ∃ a : ℕ, a^2 = n) 
  (h2 : n % 100 ≠ 0)
  (h3 : ∃ m : ℕ, m * 100 + n % 100 = n ∧ ∃ b : ℕ, b^2 = m) :
  n = 1681 := sorry

end largest_square_not_divisible_by_100_l289_289158


namespace length_more_than_breadth_l289_289421

theorem length_more_than_breadth (b x : ℝ) (h1 : b + x = 61) (h2 : 26.50 * (4 * b + 2 * x) = 5300) : x = 22 :=
by
  sorry

end length_more_than_breadth_l289_289421


namespace simplify_expression_l289_289144

theorem simplify_expression (w : ℝ) :
  3 * w + 4 - 2 * w - 5 + 6 * w + 7 - 3 * w - 9 = 4 * w - 3 :=
by 
  sorry

end simplify_expression_l289_289144


namespace relay_race_total_time_l289_289938

theorem relay_race_total_time :
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  athlete1 + athlete2 + athlete3 + athlete4 = 200 :=
by
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  show athlete1 + athlete2 + athlete3 + athlete4 = 200
  sorry

end relay_race_total_time_l289_289938


namespace number_of_red_balls_l289_289267

-- Initial conditions
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 5
def freq_red_ball : ℝ := 0.4

-- Proving the number of red balls
theorem number_of_red_balls (total_balls : ℕ) (num_red_balls : ℕ) :
  total_balls = num_black_balls + num_white_balls + num_red_balls ∧
  (num_red_balls : ℝ) / total_balls = freq_red_ball →
  num_red_balls = 8 :=
by
  sorry

end number_of_red_balls_l289_289267


namespace polynomial_identity_and_sum_of_squares_l289_289239

theorem polynomial_identity_and_sum_of_squares :
  ∃ (p q r s t u : ℤ), (∀ (x : ℤ), 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧
    p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 :=
sorry

end polynomial_identity_and_sum_of_squares_l289_289239


namespace find_y_l289_289774

noncomputable def y_solution : ℝ :=
  let k := 36
  in 15 * real.root 15 (1 / 5)

theorem find_y (x y : ℝ) (h1 : ∀ x y, x^2 * y^(1/3) = 36)
  (h2 : x * y = 90) : y = 15 * real.root 15 (1 / 5) :=
by
  sorry

end find_y_l289_289774


namespace max_S_is_9_l289_289967

-- Definitions based on the conditions
def a (n : ℕ) : ℤ := 28 - 3 * n
def S (n : ℕ) : ℤ := n * (25 + a n) / 2

-- The theorem to be proved
theorem max_S_is_9 : ∃ n : ℕ, n = 9 ∧ S n = 117 :=
by
  sorry

end max_S_is_9_l289_289967


namespace complex_number_in_first_quadrant_l289_289191

def i : ℂ := complex.I  -- defining the imaginary unit
def z : ℂ := 1 + i      -- defining the given complex number

-- Proving the complex number lies in the first quadrant
theorem complex_number_in_first_quadrant : 
  let w := (2 / z) + z ^ 2 in 
  w = 1 + i ∧ (w.re > 0 ∧ w.im > 0) :=
by 
  sorry

end complex_number_in_first_quadrant_l289_289191


namespace clown_balloons_l289_289014

theorem clown_balloons 
  (initial_balloons : ℕ := 123) 
  (additional_balloons : ℕ := 53) 
  (given_away_balloons : ℕ := 27) : 
  initial_balloons + additional_balloons - given_away_balloons = 149 := 
by 
  sorry

end clown_balloons_l289_289014


namespace min_photographs_42_tourists_3_monuments_l289_289276

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289276


namespace sum_F_1_to_1000_l289_289346

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def F (n : ℕ) : ℕ :=
  Nat.find (λ m, m > n ∧ sum_of_digits m = sum_of_digits n)

theorem sum_F_1_to_1000 : 
  (Finset.range 1000).sum (λ n, F (n + 1)) = 535501 :=
by
  sorry

end sum_F_1_to_1000_l289_289346


namespace min_photos_l289_289303

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289303


namespace graphs_equivalence_l289_289465

def f1 (x : ℝ) : ℝ := x^2 - 2
def f2 (x : ℝ) : ℝ := if x ≠ 2 then (x^3 - 8) / (x - 2) else 0 -- simplified later in proof
def f3 (x : ℝ) : ℝ := if x ≠ 2 then (x^3 - 8) / (x - 2) else 0 -- includes the behavior at x = 2

theorem graphs_equivalence : 
  (∀ x, f2 x = f3 x) ∧ (∀ x, f1 x ≠ f2 x) :=
by
  sorry -- proof is omitted as requested

end graphs_equivalence_l289_289465


namespace total_profit_l289_289856

noncomputable def profit_x (P : ℕ) : ℕ := 3 * P
noncomputable def profit_y (P : ℕ) : ℕ := 2 * P

theorem total_profit
  (P_x P_y : ℕ)
  (h_ratio : P_x = 3 * (P_y / 2))
  (h_diff : P_x - P_y = 100) :
  P_x + P_y = 500 :=
by
  sorry

end total_profit_l289_289856


namespace similar_triangle_with_equal_areas_l289_289538

theorem similar_triangle_with_equal_areas (S : ℝ) : 
  ∃ X' : set (ℝ × ℝ), 
    (∀ v ∈ X', ∃ (i j : ℤ), v = (i, j)) ∧ 
    (area X' = S) ∧ 
    (white_area X' = S) ∧ 
    (black_area X' = S) :=
sorry

end similar_triangle_with_equal_areas_l289_289538


namespace find_diagonal_length_l289_289176

theorem find_diagonal_length :
  ∀ (d h1 h2 A : ℝ),
  A = 240 →
  h1 = 10 →
  h2 = 6 →
  A = (1 / 2) * d * (h1 + h2) →
  d = 30 :=
begin
  intros d h1 h2 A h3 h4 h5,
  -- Theorem not yet proved
  sorry
end

end find_diagonal_length_l289_289176


namespace theta_solutions_count_l289_289629

theorem theta_solutions_count :
  (∃ (count : ℕ), count = 4 ∧ ∀ θ, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ) = 0 ↔ count = 4) :=
sorry

end theta_solutions_count_l289_289629


namespace two_pow_gt_twice_n_plus_one_l289_289789

theorem two_pow_gt_twice_n_plus_one (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
sorry

end two_pow_gt_twice_n_plus_one_l289_289789


namespace max_mondays_l289_289458

-- Definitions from conditions
def days_in_week := 7
def total_days := 45
def mondays_in_week := 1

-- Statement of mathematical proof problem
theorem max_mondays (days_in_week total_days mondays_in_week : ℕ) :
  days_in_week = 7 →
  total_days = 45 →
  mondays_in_week = 1 →
  ∃ max_mondays : ℕ, max_mondays = 7 :=
by 
  intros h1 h2 h3 
  use 7 
  sorry

end max_mondays_l289_289458


namespace a_4_value_l289_289995

def a : ℕ → ℕ
| 0     := 2
| (n+1) := 3 * a n

theorem a_4_value : a 3 = 54 :=
by sorry

end a_4_value_l289_289995


namespace Jesse_pages_left_to_read_l289_289337

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end Jesse_pages_left_to_read_l289_289337


namespace max_gcd_lcm_eq_10_l289_289722

open Nat -- Opening the namespace for natural numbers

theorem max_gcd_lcm_eq_10
  (a b c : ℕ) 
  (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) :
  gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_eq_10_l289_289722


namespace PS_div_QR_eq_sqrt3_l289_289662

theorem PS_div_QR_eq_sqrt3 (t : ℝ) (P Q R S : Type) 
  (hPQR : ∀ {X Y Z : Type}, X = P → Y = Q → Z = R → (equilateral_triangle X Y Z))
  (hQRS : ∀ {X Y Z : Type}, X = Q → Y = R → Z = S → (equilateral_triangle X Y Z)) :
  PS ∈ ([P, S]) →
  QR = t →
  PS / QR = √3 :=
sorry

end PS_div_QR_eq_sqrt3_l289_289662


namespace find_calories_per_slice_l289_289088

/-- Defining the number of slices and their respective calories. -/
def slices_in_cake : ℕ := 8
def calories_per_brownie : ℕ := 375
def brownies_in_pan : ℕ := 6
def extra_calories_in_cake : ℕ := 526

/-- Defining the total calories in cake and brownies -/
def total_calories_in_brownies : ℕ := brownies_in_pan * calories_per_brownie
def total_calories_in_cake (c : ℕ) : ℕ := slices_in_cake * c

/-- The equation from the given problem -/
theorem find_calories_per_slice (c : ℕ) :
  total_calories_in_cake c = total_calories_in_brownies + extra_calories_in_cake → c = 347 :=
by
  sorry

end find_calories_per_slice_l289_289088


namespace cone_rotations_ratio_l289_289863

theorem cone_rotations_ratio (r h : ℝ) 
  (h_pos : h > 0)
  (r_pos : r > 0)
  (rotations : 21 * (2 * r * Real.pi) = 2 * Real.pi * Real.sqrt (r ^ 2 + h ^ 2)):
    ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (∃ n_prime_square_free : ¬ ∃ p : ℕ, Prime p ∧ p * p ∣ n) ∧ (h / r = m * Real.sqrt n) ∧ (m + n = 31) :=
sorry

end cone_rotations_ratio_l289_289863


namespace problem_statement_l289_289189

noncomputable def a : ℝ := 1.2 ^ 0.3
noncomputable def b : ℝ := logBase 0.3 1.2
noncomputable def c : ℝ := logBase 1.2 3

theorem problem_statement : (b < a ∧ a < c) :=
by
  sorry

end problem_statement_l289_289189


namespace equilateral_and_counterclockwise_l289_289698

open Complex

def equilateral_triangle (A B C : ℂ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A 

def vertices_counterclockwise (A B C : ℂ) : Prop :=
  (B - A).arg < (C - A).arg

theorem equilateral_and_counterclockwise (A B C : ℂ) (P₀ : ℂ := 0) :
  (∀ n, ∃ P_n : ℂ, Pₙ = (P₀ rotation by 60° left n) and dist P₀ P₁ and ... for n=1986 Pₙ = 0 → 
  equilateral_triangle A B C ∧ vertices_counterclockwise A B C :=
sorry

end equilateral_and_counterclockwise_l289_289698


namespace cartesian_product_cardinality_l289_289998

def isCartesianProduct (P Q : Set ℕ) (R : Set (ℕ × ℕ)) : Prop :=
  ∀ a b, (a ∈ P) → (b ∈ Q) → ((a, b) ∈ R)

theorem cartesian_product_cardinality (P Q : Set ℕ) (hP : P = {0, 1, 2}) (hQ : Q = {1, 2}) :
  ∃ R, isCartesianProduct P Q R ∧ R.toFinset.card = 6 := by
  sorry

end cartesian_product_cardinality_l289_289998


namespace window_area_properties_l289_289498

theorem window_area_properties
  (AB : ℝ) (AD : ℝ) (ratio : ℝ)
  (h1 : ratio = 3 / 1)
  (h2 : AB = 40)
  (h3 : AD = 3 * AB) :
  (AD * AB / (π * (AB / 2) ^ 2) = 12 / π) ∧
  (AD * AB + π * (AB / 2) ^ 2 = 4800 + 400 * π) :=
by
  -- Proof will go here
  sorry

end window_area_properties_l289_289498


namespace correct_articles_l289_289560

-- Definitions based on conditions provided in the problem
def sentence := "Traveling in ____ outer space is quite ____ exciting experience."
def first_blank_article := "no article"
def second_blank_article := "an"

-- Statement of the proof problem
theorem correct_articles : 
  (first_blank_article = "no article" ∧ second_blank_article = "an") :=
by
  sorry

end correct_articles_l289_289560


namespace anya_smallest_number_divisible_by_eleven_l289_289886

theorem anya_smallest_number_divisible_by_eleven :
  ∃ (n: ℕ), n = 909090909 ∧ (∀ (d_pos: ℕ), d_pos < n.num_digits → 
  ∀ (new_digit: ℕ), abs (new_digit - (n.digits.get d_pos)) = 1 → 
  (digits_to_nat (n.digits.update d_pos new_digit)) % 11 = 0) :=
sorry

end anya_smallest_number_divisible_by_eleven_l289_289886


namespace tetrahedron_inequality_l289_289317

variables {A B C D : Type*}
variables {distance : A → A → ℝ}    -- distance function

-- Assuming points A, B, C, D form a tetrahedron
def is_tetrahedron (A B C D : A) : Prop := 
  ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D

-- Given ∠BDC = 90°
def angle_BDC_90 (A B C D : A) : Prop := 
  orthogonal (B - D) (C - D)

-- The foot of the perpendicular from D to ABC is the orthocenter of ABC
def orthocenter_condition (A B C D : A) : Prop :=
  let H := orthocenter A B C in
  is_perpendicular (D, H, A) ∧ is_perpendicular (D, H, B) ∧ is_perpendicular (D, H, C)

-- Main theorem statement
theorem tetrahedron_inequality 
  (h1 : is_tetrahedron A B C D) 
  (h2 : angle_BDC_90 A B C D) 
  (h3 : orthocenter_condition A B C D) : 
  (distance A B + distance B C + distance C A)^2 ≤ 6 * (distance A D^2 + distance B D^2 + distance C D^2) :=
sorry

end tetrahedron_inequality_l289_289317


namespace shoe_store_sales_l289_289920

theorem shoe_store_sales (sneakers sandals boots : ℕ) (h_sneakers : sneakers = 2) (h_sandals : sandals = 4) (h_boots : boots = 11) :
  sneakers + sandals + boots = 17 :=
by {
  rw [h_sneakers, h_sandals, h_boots],
  exact rfl,
}

end shoe_store_sales_l289_289920


namespace problem1_problem2_l289_289656

open Real

-- Definition of the Circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Coordinates of point M
def M := (2 : ℝ, -3 : ℝ)

-- Definition of point P (intersection with positive x-axis)
def P := (2 : ℝ, 0 : ℝ)

-- Tangency condition
def is_tangent (l : ℝ → ℝ → Prop) : Prop :=
∀ x y r₀ (h : circle_C x y), l x y → sqrt (x^2 + (y - r₀)^2) = 2

-- Line x = 2 and its tangency check
def line1 := λ x y : ℝ, x = 2

-- Line 5x + 12y + 26 = 0 and its tangency check
def line2 := λ x y : ℝ, 5 * x + 12 * y + 26 = 0

theorem problem1 : 
    (is_tangent line1 ∧ line1 (2, -3)) ∨ (is_tangent line2 ∧ line2 (2, -3)) :=
by sorry

-- Definition of slope function 
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Coordinates of points A and B on circle C, intersecting a line through M(2, -3)
def A (x1 : ℝ) := (x1, sqrt (4 - x1^2))
def B (x2 : ℝ) := (x2, -sqrt (4 - x2^2))

-- Theorem stating that the sum of slopes from P to A and P to B is constant (4/3)
theorem problem2 :
    ∀ x1 x2 : ℝ, circle_C x1 (sqrt (4 - x1^2)) →
                 circle_C x2 (-sqrt (4 - x2^2)) →
                 slope P (A x1) + slope P (B x2) = 4 / 3 :=
by sorry

end problem1_problem2_l289_289656


namespace rons_pick_times_l289_289391

def total_members(couples single_people : ℕ) : ℕ := couples * 2 + single_people + 2

def times_rons_pick(total_members : ℕ) : ℕ := 52 / total_members

theorem rons_pick_times
    (couples single_people : ℕ)
    (h_couples : couples = 3)
    (h_single_people : single_people = 5) :
    times_rons_pick (total_members couples single_people) = 4 :=
by
  have h_total_members : total_members couples single_people = 13 := by
    simp [total_members, h_couples, h_single_people]
  simp [times_rons_pick, h_total_members]
  sorry

end rons_pick_times_l289_289391


namespace vampire_daily_blood_suction_l289_289872

-- Conditions from the problem
def vampire_bl_need_per_week : ℕ := 7  -- gallons of blood per week
def blood_per_person_in_pints : ℕ := 2  -- pints of blood per person
def pints_per_gallon : ℕ := 8            -- pints in 1 gallon

-- Theorem statement to prove
theorem vampire_daily_blood_suction :
  let daily_requirement_in_gallons : ℕ := vampire_bl_need_per_week / 7   -- gallons per day
  let daily_requirement_in_pints : ℕ := daily_requirement_in_gallons * pints_per_gallon
  let num_people_needed_per_day : ℕ := daily_requirement_in_pints / blood_per_person_in_pints
  num_people_needed_per_day = 4 :=
by
  sorry

end vampire_daily_blood_suction_l289_289872


namespace zero_intercept_and_distinct_roots_l289_289919

noncomputable def Q (x a' b' c' d' : ℝ) : ℝ := x^4 + a' * x^3 + b' * x^2 + c' * x + d'

theorem zero_intercept_and_distinct_roots (a' b' c' d' : ℝ) (u v w : ℝ) (h_distinct : u ≠ v ∧ v ≠ w ∧ u ≠ w) (h_intercept_at_zero : d' = 0)
(h_Q_form : ∀ x, Q x a' b' c' d' = x * (x - u) * (x - v) * (x - w)) : c' ≠ 0 :=
by
  sorry

end zero_intercept_and_distinct_roots_l289_289919


namespace find_m_plus_n_l289_289777

open Real

-- Define the problem conditions
def squares_shared_center {O : Point} {A B: Point} : O ∈ center_of_square A B := sorry
def square_side_length_one {s₁ s₂: ℝ} (h₁ : s₁ = 1) (h₂ : s₂ = 1) := sorry
def length_AB (h : dist A B = 43 / 99) := sorry

-- Define the problem statement in Lean 4 using the conditions
theorem find_m_plus_n {m n : ℕ} (h₁ : coprime m n) 
    (h₂ : ∃ (area : ℚ), area = m / n ∧
    squares_shared_center ∧ square_side_length_one 1 1 ∧ length_AB) : 
    m + n = 185 := 
sorry

end find_m_plus_n_l289_289777


namespace incenter_distance_sum_l289_289690

theorem incenter_distance_sum
  (A B C O : Point)
  (a b c : ℝ)
  (hO : is_incenter O A B C)
  (OA OB OC : ℝ)
  (hOA : dist O A = OA)
  (hOB : dist O B = OB)
  (hOC : dist O C = OC) :
  OA^2 / (b * c) + OB^2 / (a * c) + OC^2 / (a * b) = 1 := 
sorry

end incenter_distance_sum_l289_289690


namespace number_of_ways_to_choose_lines_l289_289742

theorem number_of_ways_to_choose_lines :
  let hor_lines := 6
  let ver_lines := 7
  ∃ restricted_pairs : list (ℕ × ℕ),
  restricted_pairs = [(2, 5), (3, 6)] →
  number_of_rectangles hor_lines ver_lines restricted_pairs = 280 :=
by 
  let hor_lines := 6
  let ver_lines := 7
  let restricted_pairs := [(2, 5), (3, 6)]
  exist restricted_pairs
  sorry

noncomputable def number_of_rectangles (hor_lines ver_lines : ℕ) 
  (restricted_pairs : list (ℕ × ℕ)) : ℕ :=
sorry

end number_of_ways_to_choose_lines_l289_289742


namespace angela_sleep_difference_l289_289137

def days_in_month := 31
def december_weekend_days := 8
def december_weekdays := days_in_month - december_weekend_days
def january_weekend_days := 8
def january_weekdays := days_in_month - january_weekend_days
def napping_hours_on_sundays := 2
def new_year_nap_january := 3

def angela_sleep_december : ℕ :=
  (6.5 * december_weekdays) + (7.5 * december_weekend_days) + (napping_hours_on_sundays * (december_weekend_days / 2))

def angela_sleep_january : ℕ :=
  (8.5 * january_weekdays) + (8.5 * january_weekend_days) + (napping_hours_on_sundays * (january_weekend_days / 2)) + new_year_nap_january

theorem angela_sleep_difference : angela_sleep_january - angela_sleep_december = 57 := by
  sorry

end angela_sleep_difference_l289_289137


namespace largest_number_is_763210_l289_289797

-- Define the set of digits
def distinct_digits_sum_17 : Set ℕ := {n | (∃ (digits : List ℕ),
  n = digits.foldl (λ acc d, acc + d) 0 ∧
  digits.nodup ∧
  digits.sum = 17)}

-- Define the goal to show that 763210 is the largest such number
theorem largest_number_is_763210 
  (h : 763210 ∈ distinct_digits_sum_17): 
  ∀ n ∈ distinct_digits_sum_17, n ≤ 763210 :=
sorry

end largest_number_is_763210_l289_289797


namespace part1_part2_l289_289685

open Real EuclideanGeometry

variables {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Conditions for Part 1
variables (a b : V) (k : ℝ)
variable h_non_collinear : ¬Collinear ℝ ({0, a, b} : Set V)

theorem part1 (h : Collinear ℝ ({0, 8 • a + k • b, k • a + 2 • b} : Set V)) :
  k = 4 ∨ k = -4 :=
sorry

-- Conditions for Part 2
variable ha : ∥a∥ = 4
variable hb : ∥b∥ = 3
variable h_dot : (2 • a - 3 • b) ⬝ (2 • a + b) = 61

theorem part2 : ∥a + b∥ = Real.sqrt 13 :=
sorry

end part1_part2_l289_289685


namespace max_value_expression_l289_289178

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by
  exact sorry

end max_value_expression_l289_289178


namespace simplify_radicals_l289_289739

theorem simplify_radicals :
  (512 : ℝ)^(1/3) * (343 : ℝ)^(1/2) = 56 * real.sqrt 7 :=
by
  have h1 : (512 : ℝ) = real.of_nat 2 ^ 9 := by norm_num
  have h2 : (343 : ℝ) = real.of_nat 7 ^ 3 := by norm_num
  rw [h1, h2]
  rw [real.rpow_mul, real.rpow_mul]
  norm_num
  sorry

end simplify_radicals_l289_289739


namespace basketball_shots_l289_289644

theorem basketball_shots (total_points total_3pt_shots: ℕ) 
  (h1: total_points = 26) 
  (h2: total_3pt_shots = 4) 
  (h3: ∀ points_from_3pt_shots, points_from_3pt_shots = 3 * total_3pt_shots) :
  let points_from_3pt_shots := 3 * total_3pt_shots
  let points_from_2pt_shots := total_points - points_from_3pt_shots
  let total_2pt_shots := points_from_2pt_shots / 2
  total_2pt_shots + total_3pt_shots = 11 :=
by
  sorry

end basketball_shots_l289_289644


namespace probability_red_ball_l289_289268

variable (num_white_balls : ℕ) (num_red_balls : ℕ)
variable (total_balls : ℕ := num_white_balls + num_red_balls)
variable (favorable_outcomes : ℕ := num_red_balls)
variable (probability_of_red : ℚ := favorable_outcomes / total_balls)

theorem probability_red_ball
  (h1 : num_white_balls = 3)
  (h2 : num_red_balls = 7) :
  probability_of_red = 7 / 10 := by
  sorry

end probability_red_ball_l289_289268


namespace probability_exactly_four_even_out_of_eight_l289_289553

theorem probability_exactly_four_even_out_of_eight :
  (nat.choose 8 4 * (2/3)^4 * (1/3)^4) = (1120 / 6561) := sorry

end probability_exactly_four_even_out_of_eight_l289_289553


namespace batsman_total_score_l289_289832

theorem batsman_total_score 
  (T : ℝ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (running_percentage : ℝ)
  (H1 : boundaries = 5)
  (H2 : sixes = 5)
  (H3 : running_percentage = 66.67 / 100)
  (H4 : T = (2 / 3) * T + 50) :
  T = 150 :=
by
  have H1 : (5 * 4 : ℝ) = 20 := by norm_num
  have H2 : (5 * 6 : ℝ) = 30 := by norm_num
  have H3 : (20 + 30 : ℝ) = 50 := by norm_num
  have H4 : (2 / 3 : ℝ) = 66.67 / 100 := by norm_num
  sorry

end batsman_total_score_l289_289832


namespace minimum_photos_l289_289283

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289283


namespace median_and_mode_of_written_scores_composite_score_percentages_find_top_two_candidates_l289_289839

-- Question 1: Median and Mode calculation for given written test scores.
theorem median_and_mode_of_written_scores : 
  ∀ (scores : List ℕ), 
  scores = [85, 92, 84, 90, 84, 80] →
  ∃ (median mode : ℝ),
    median = (84 + 85) / 2 ∧ mode = 84 := 
by
  intro scores h_scores
  have h : scores.sorted = [80, 84, 84, 85, 90, 92] := sorry
  use ((84 + 85) / 2), 84
  exact sorry

-- Question 2: Finding the composition percentages.
theorem composite_score_percentages :
  ∃ (x y : ℝ), 
    x + y = 1 ∧ 
    85 * x + 90 * y = 88 ∧ x = 0.4 ∧ y = 0.6 :=  
by
  have h_eqns : ∀ x y : ℝ, x + y = 1 → 85 * x + 90 * y = 88 → x = 0.4 ∧ y = 0.6 := sorry
  exact ⟨0.4, 0.6, sorry, sorry, rfl, rfl⟩

-- Question 3: Compute composite scores and find top two candidates.
theorem find_top_two_candidates :
  ∀ (candidates : List (ℕ × ℕ)),
  candidates = [(92, 88), (84, 86), (90, 90), (84, 80), (80, 85)] →
  ∃ (top1 top2 : (ℕ × ℕ)),
    top1 = (90, 90) ∧ top2 = (92, 88) :=
by
  intro candidates h_candidates
  have h_comp_scores : ∀ (w : ℕ) (i : ℕ), (0.4 * w + 0.6 * i) ∈ [89.6, 85.2, 90, 81.6, 83] := sorry
  use ((90, 90)), ((92, 88))
  exact sorry

end median_and_mode_of_written_scores_composite_score_percentages_find_top_two_candidates_l289_289839


namespace bottle_caps_left_l289_289033

/-- 
Proof that if there are 16 bottle caps and Marvin takes away 6, there will be 10 bottle caps left.
-/
theorem bottle_caps_left (initial_caps : ℕ) (taken_caps : ℕ) (left_caps : ℕ) 
  (h_initial : initial_caps = 16) (h_taken : taken_caps = 6) : 
  left_caps = initial_caps - taken_caps :=
by sorry

-- Applying the given conditions
example : bottle_caps_left 16 6 10  (by rfl) (by rfl) = 10 :=
by simp [bottle_caps_left]

end bottle_caps_left_l289_289033


namespace max_sum_counts_l289_289487

theorem max_sum_counts (R C : ℕ) (num_pluses num_minuses : ℕ) (row_count col_count : Fin 30 → Fin 18) :
  R = 30 → C = 30 → num_pluses = 162 → num_minuses = 144 → 
  (∀ i, (row_count i : ℕ) ≤ 17) → (∀ j, (col_count j : ℕ) ≤ 17) →
  ∃ max_count : ℝ, max_count = 1296 :=
by
  intros
  exists 1296
  admit

end max_sum_counts_l289_289487


namespace exists_integers_satisfy_condition_l289_289005

theorem exists_integers_satisfy_condition :
  ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c :=
by
  use [-1, -2, -3]
  repeat {split; norm_num}

end exists_integers_satisfy_condition_l289_289005


namespace system_solve_unique_solution_for_positive_numbers_l289_289744

theorem system_solve_unique_solution_for_positive_numbers :
  ∀ x y z : ℝ,
  0 < x ∧ 0 < y ∧ 0 < z ∧
  (x^y = z ∧ y^z = x ∧ z^x = y) →
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  intro x y z
  intro h_pos h_eqns
  sorry  -- Proof goes here

end system_solve_unique_solution_for_positive_numbers_l289_289744


namespace arithmetic_expression_equality_l289_289532

theorem arithmetic_expression_equality :
  15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := 
by 
  sorry

end arithmetic_expression_equality_l289_289532


namespace find_k_range_l289_289705

section 
variables {y f : ℝ → ℝ} {k : ℝ} {x : ℝ} 

def has_k_order_linear_approx (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : Prop :=
  ∀ (λ : ℝ) (x₁ x₂ : ℝ),
  x = λ * x₁ + (1 - λ) * x₂ → 
  abs (f x - (λ * f x₁ + (1 - λ) * f x₂)) ≤ k

theorem find_k_range : 
  ∀ (k : ℝ), 
  (∀ (λ : ℝ) (x₁ x₂ : ℝ),
  (0 ≤ λ ∧ λ ≤ 1) ∧ (x₁ = 0 ∧ x₂ = 1) → 
  (abs ((sqrt x - x) - (λ * (sqrt 0 - 0) + (1 - λ) * (sqrt 1 - 1))) ≤ k) ∧
  (abs ((3 * x - x) - (λ * (3 * 0 - 0) + (1 - λ) * (3 * 1 - 1))) ≤ k)) 
  ↔ k ∈ set.Icc (1 / 4) ((2 * real.sqrt 3) / 9) := 
sorry
end

end find_k_range_l289_289705


namespace inequality_solution_l289_289812

theorem inequality_solution :
  { x : ℝ | 3 * x^2 - 2 * x - 8 < 0 } = { x : ℝ | -4 / 3 < x ∧ x < 2 } :=
begin
  sorry
end

end inequality_solution_l289_289812


namespace tulips_after_addition_l289_289643

theorem tulips_after_addition
  (initial_sunflowers : ℕ)
  (additional_sunflowers : ℕ)
  (tulips_to_sunflowers_ratio_num : ℕ)
  (tulips_to_sunflowers_ratio_den : ℕ)
  (h1 : initial_sunflowers = 42)
  (h2 : additional_sunflowers = 28)
  (h3 : tulips_to_sunflowers_ratio_num = 3)
  (h4 : tulips_to_sunflowers_ratio_den = 7) :
  let total_sunflowers := initial_sunflowers + additional_sunflowers
  in total_sunflowers * tulips_to_sunflowers_ratio_num / tulips_to_sunflowers_ratio_den = 30 := by
  sorry

end tulips_after_addition_l289_289643


namespace parallel_condition_l289_289951

variables (α β : Plane) (m n : Line)
-- α and β are different planes
-- m and n are different lines
-- α ∩ β = m
-- n ⊄ α
-- n ⊄ β

theorem parallel_condition (h1 : α ≠ β)
                           (h2 : m ≠ n)
                           (h3 : α ∩ β = m)
                           (h4 : ¬ n ⊆ α)
                           (h5 : ¬ n ⊆ β) :
  (is_parallel n m) ↔ (is_parallel n α ∧ is_parallel n β) :=
sorry

end parallel_condition_l289_289951


namespace problem1_problem2_l289_289221

-- Define the function f(x) = 2(x-1)e^x
def f (x : ℝ) : ℝ := 2 * (x - 1) * Real.exp x

-- Problem 1
theorem problem1 (a : ℝ) (h : a ≥ 0) : f a ≥ -2 := sorry

-- Define the function g(x) = e^x - x + p
def g (x p : ℝ) : ℝ := Real.exp x - x + p

-- Problem 2
theorem problem2 (p : ℝ) (h : ∃ (x0 : ℝ), 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ g x0 p ≥ f x0 - x0) : p ≥ -Real.exp 1 := sorry

end problem1_problem2_l289_289221


namespace product_of_odd_primes_mod_sixteen_l289_289680

-- Define the set of odd primes less than 16
def odd_primes_less_than_sixteen : List ℕ := [3, 5, 7, 11, 13]

-- Define the product of all odd primes less than 16
def N : ℕ := odd_primes_less_than_sixteen.foldl (· * ·) 1

-- Proposition to prove: N ≡ 7 (mod 16)
theorem product_of_odd_primes_mod_sixteen :
  (N % 16) = 7 :=
  sorry

end product_of_odd_primes_mod_sixteen_l289_289680


namespace product_of_abc_l289_289776

variable (a b c m : ℚ)

-- Conditions
def condition1 : Prop := a + b + c = 200
def condition2 : Prop := 8 * a = m
def condition3 : Prop := m = b - 10
def condition4 : Prop := m = c + 10

-- The theorem to prove
theorem product_of_abc :
  a + b + c = 200 ∧ 8 * a = m ∧ m = b - 10 ∧ m = c + 10 →
  a * b * c = 505860000 / 4913 :=
by
  sorry

end product_of_abc_l289_289776


namespace product_of_f_at_minus1_and_1_equals_zero_l289_289076

theorem product_of_f_at_minus1_and_1_equals_zero
  (a b c : ℝ)
  (h₀ : ∀ x, f x = a * x^2 + b * x + c)
  (h₁ : f ((a - b - c) / (2 * a)) = 0)
  (h₂ : f ((c - a - b) / (2 * a)) = 0) :
  f (-1) * f 1 = 0 :=
sorry

end product_of_f_at_minus1_and_1_equals_zero_l289_289076


namespace trapezoid_parallelogram_condition_l289_289650

-- Define trapezoid and circles with given diameters
variables {A B C D O₁ O₂ O₃ O₄ : Type} [trapezoid A B C D] (AD_parallel_BC : AD ∥ BC)

-- Define the main theorem
theorem trapezoid_parallelogram_condition :
  (∃ (O : Type), center_inside_trapezoid O A B C D ∧ tangent_to_all_circles O O₁ O₂ O₃ O₄) ↔ parallelogram A B C D := 
sorry

end trapezoid_parallelogram_condition_l289_289650


namespace cryptarithm_solution_l289_289660

-- Define the notion of Chinese characters representing unique digits
def distinct_digits (春 萧 杯 : ℕ) : Prop :=
  春 ≠ 萧 ∧ 萧 ≠ 杯 ∧ 春 ≠ 杯

theorem cryptarithm_solution 春 萧 杯 : (春 * 100 + 萧 * 10 + 杯 = 958) →
  (春 = 9 ∧ 萧 = 5 ∧ 杯 = 8) :=
begin
  -- Using the given conditions that different Chinese characters represent different digits
  assume h1 : (春 * 100 + 萧 * 10 + 杯 = 958),
  sorry
end

end cryptarithm_solution_l289_289660


namespace radius_and_area_of_circle_l289_289884

theorem radius_and_area_of_circle
  (triangle_isosceles : ∀ {A B C : Type} (AB BC CA : ℝ), AB = 4 ∧ BC = 3 ∧ CA = 4)
  (triangle_inscribed : ∀ {A B C O : Type} (center : ℝ) (radius : ℝ), radius = 3.5 ∧ (area : ℝ), area = 12.25 * Real.pi) :
  ∃ r a, r = 3.5 ∧ a = 12.25 * Real.pi :=
by
  sorry

end radius_and_area_of_circle_l289_289884


namespace measure_of_angle_A_l289_289254

theorem measure_of_angle_A (a b c : ℝ) (A B C : ℝ) (ha2 : a^2 = 2 * b * c + b^2) (hSinC : sin C = 3 * sin B) :
  A = 60 :=
by
  sorry

end measure_of_angle_A_l289_289254


namespace find_t_l289_289918

-- Define points
structure Point where
  x : ℝ
  y : ℝ

-- Define the input conditions
def p1 : Point := { x := 2, y := 1 }
def p2 : Point := { x := 10, y := 5 }
def target_point (t : ℝ) : Point := { x := t, y := 7 }

-- Define the slope function
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Define the equation of line through two points and passing through the target point
def onLine (p1 p2 : Point) (pt : Point) : Prop :=
  (pt.y - p1.y) / (pt.x - p1.x) = slope p1 p2

-- The main statement to prove
theorem find_t : ∃ t : ℝ, onLine p1 p2 (target_point t) ∧ target_point t = { x := 14, y := 7 } :=
by
  use 14
  -- Placeholder for the proof
  sorry

end find_t_l289_289918


namespace ratio_areas_EFC_BCD_l289_289106

-- Defining the given conditions formally
variable {A B C D : Point}
variable {B' C' D' E F : Point}

-- Given the conditions about the intersections
axiom intersects_AB_3_1 : divides A B B' 3
axiom intersects_AC_3_1 : divides A C C' 3
axiom intersects_AD_3_1 : divides A D D' 3
axiom intersects_BC_E : lies_on_line E B C
axiom intersects_CD_F : lies_on_line F C D

-- The theorem to be proven
theorem ratio_areas_EFC_BCD :
  area_ratio (triangle E F C) (triangle B C D) = 1 / 64 ∨ area_ratio (triangle E F C) (triangle B C D) = 81 / 64 :=
by
  sorry

end ratio_areas_EFC_BCD_l289_289106


namespace incorrect_expression_l289_289810

theorem incorrect_expression : ¬ (5 = (Real.sqrt (-5))^2) :=
by
  sorry

end incorrect_expression_l289_289810


namespace solve_sqrt_eqn_l289_289172

theorem solve_sqrt_eqn (z : ℝ) (h : sqrt (9 + 3 * z) = 12) : z = 45 :=
by
  sorry

end solve_sqrt_eqn_l289_289172


namespace range_of_a_l289_289246

theorem range_of_a (a : ℝ) : (∀ x > 0, ln x - a * x + 2 = 0 → a ∈ set.Ioo 0 Real.exp 1) :=
begin
  sorry,
end

end range_of_a_l289_289246


namespace gardener_b_time_l289_289940

theorem gardener_b_time :
  ∃ x : ℝ, (1 / 3 + 1 / x = 1 / 1.875) → (x = 5) := by
  sorry

end gardener_b_time_l289_289940


namespace geraldo_drank_l289_289053

def total_gallons : ℝ := 20
def total_containers : ℝ := 80
def containers_drank : ℝ := 3.5
def pints_per_gallon : ℝ := 8

theorem geraldo_drank :
  let tea_per_container : ℝ := total_gallons / total_containers in
  let pints_per_container : ℝ := tea_per_container * pints_per_gallon in
  let total_pints_drank : ℝ := containers_drank * pints_per_container in
  total_pints_drank = 7 :=
by
  sorry

end geraldo_drank_l289_289053


namespace intersection_complements_eq_l289_289438

-- Define the universal set
def U := Set.univ

-- Define set A
def A := {x : ℝ | |x| ≥ 1 }

-- Define set B
def B := {x : ℝ | x^2 - 2 * x - 3 > 0 }

-- Define complement of A in U
def complement_A : Set ℝ := {x | -1 < x ∧ x < 1 }

-- Define complement of B in U
def complement_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3 }

-- Define the intersection of the complements
def intersection_complements : Set ℝ := complement_A ∩ complement_B

-- State the theorem to be proved
theorem intersection_complements_eq : intersection_complements = {x | -1 < x ∧ x < 1} :=
by
  sorry

end intersection_complements_eq_l289_289438


namespace find_integer_n_l289_289927

theorem find_integer_n (n : ℤ) : ∃ x : ℤ, 2^n + 1 = x^2 ↔ n = 3 := 
by
  sorry

end find_integer_n_l289_289927


namespace proof_problem_l289_289699

noncomputable def sum_of_values (α β : ℝ) : ℝ :=
  (sin β)^2 / (sin α) + (cos β)^2 / (cos α)

theorem proof_problem (α β : ℝ) (h₁ : (cos α)^2 / cos β + (sin α)^2 / sin β = 2) :
  sum_of_values α β = real.sqrt 2 := 
sorry

end proof_problem_l289_289699


namespace recursion_relation_l289_289675

-- Define the function f
def f (k : ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), if (i ≤ n ∧ k ∣ (n - 2 * i)) then nat.choose n i else 0

-- Given that k is a fixed odd number greater than 1
axiom k_gt_one (k : ℕ) : k > 1
axiom k_odd (k : ℕ) : k % 2 = 1

-- Prove the recursion relation for f
theorem recursion_relation {k n : ℕ} (hk1 : k > 1) (hk2 : k % 2 = 1) : 
  (f k n) ^ 2 = ∑ i in finset.range (n + 1), nat.choose n i * (f k i) * (f k (n - i)) := 
sorry

end recursion_relation_l289_289675


namespace sum_difference_of_consecutive_odd_sets_l289_289456

theorem sum_difference_of_consecutive_odd_sets
  (C : ℕ)
  (hC_pos : C > 0)
  (hC_odd : C % 2 = 1) :
  let set1 := [C - 2, C, C + 2],
      set2 := [C - 4, C - 2, C] in
  (set1.sum - set2.sum = 6) :=
begin
  sorry
end

end sum_difference_of_consecutive_odd_sets_l289_289456


namespace sqrt_sub_eq_zero_l289_289535

theorem sqrt_sub_eq_zero : sqrt 12 - 2 * sqrt 3 = 0 := by
  sorry

end sqrt_sub_eq_zero_l289_289535


namespace largest_number_is_763210_l289_289796

-- Define the set of digits
def distinct_digits_sum_17 : Set ℕ := {n | (∃ (digits : List ℕ),
  n = digits.foldl (λ acc d, acc + d) 0 ∧
  digits.nodup ∧
  digits.sum = 17)}

-- Define the goal to show that 763210 is the largest such number
theorem largest_number_is_763210 
  (h : 763210 ∈ distinct_digits_sum_17): 
  ∀ n ∈ distinct_digits_sum_17, n ≤ 763210 :=
sorry

end largest_number_is_763210_l289_289796


namespace find_h_parallel_line_l289_289502

theorem find_h_parallel_line:
  ∃ h : ℚ, (3 * (h : ℚ) - 2 * (24 : ℚ) = 7) → (h = 47 / 3) :=
by
  sorry

end find_h_parallel_line_l289_289502


namespace phi_value_l289_289610

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h : |φ| < π / 2) :
  (∀ x : ℝ, f (x + π / 3) φ = f (-(x + π / 3)) φ) → φ = -(π / 6) :=
by
  intro h'
  sorry

end phi_value_l289_289610


namespace christmassy_sequence_contains_perfect_square_l289_289677

theorem christmassy_sequence_contains_perfect_square (n : ℕ) (M : Finset ℕ) (hM : M.card = n) 
    (seq : Fin n.succ → ℕ) (hseq : ∀ i, seq i ∈ M) : 
    ∃ i j, i < j ∧ ∃ k, (list.prod (list.map seq (list.range (j - i)))) = k^2 := 
begin
  sorry -- Proof is omitted as per instructions
end

end christmassy_sequence_contains_perfect_square_l289_289677


namespace parallel_or_coincides_midpoint_line_l289_289673

variables {A B C D X Y P : Type} [point A] [point B] [point C] [point D] [point X] [point Y] [point P]

structure Parallelogram (A B C D : Type) :=
(mk :: (parallelogram : true)) -- Placeholder for actual parallelogram definition

structure Midpoint (M : Type) (P Q : Type) :=
(mk :: (midpoint : true)) -- Placeholder for actual midpoint definition

structure Line (M N : Type) :=
(mk :: (line : true)) -- Placeholder for actual line definition

variables (ABCD : Parallelogram A B C D)
variables (midpointM : Midpoint M B D)
variables (midpointN : Midpoint N X Y)
variables (APLine : Line AP)
variables (MNLine : Line M N)

theorem parallel_or_coincides_midpoint_line
  (h1 : ∃ X Y : point, X ∈ segment B C ∧ Y ∈ segment C D)
  (h2 : ∃ P : point, BY ∩ DX = P)
  (h3 : Parallelogram ABCD)
  (h4 : Midpoint M B D)
  (h5 : Midpoint N X Y)
  (h6 : Line AP)
  (h7 : Line M N) :
  Line M N = Line AP ∨ Line M N ∥ Line AP :=
sorry

end parallel_or_coincides_midpoint_line_l289_289673


namespace evaluation_problem_l289_289164

theorem evaluation_problem :
  (⌈23 / 9 - ⌈35 / 21⌉⌉ / ⌈36 / 9 + ⌈9 * 23 / 36⌉⌉) = (1 / 10) :=
by 
  sorry

end evaluation_problem_l289_289164


namespace prime_factors_sum_l289_289570

theorem prime_factors_sum :
  let p := 2^23 * 3^19 * 5^17 * 7^13 * 11^11 * 13^9 * 17^7 * 19^5 * 23^3 * 29^2
  in 23 + 19 + 17 + 13 + 11 + 9 + 7 + 5 + 3 + 2 = 109 := 
by 
  have h : 23 + 19 + 17 + 13 + 11 + 9 + 7 + 5 + 3 + 2 = 109 := sorry
  exact h

end prime_factors_sum_l289_289570


namespace find_q_l289_289687

theorem find_q (a b m p q : ℚ) 
  (h1 : ∀ x, x^2 - m * x + 3 = (x - a) * (x - b)) 
  (h2 : a * b = 3) 
  (h3 : (x^2 - p * x + q) = (x - (a + 1/b)) * (x - (b + 1/a))) : 
  q = 16 / 3 := 
by sorry

end find_q_l289_289687


namespace cake_and_icing_sum_l289_289846

-- Definitions according to the problem conditions
noncomputable def cube_edge_length : ℝ := 3
noncomputable def cubic_cake_piece (edge_length : ℝ) : ℝ := 
  let height := edge_length in
  let area_top_face := (sqrt 20.25) in -- Heron's or some geometry for triangular area calc (simplified)
  (area_top_face * height)

noncomputable def icing_area (edge_length : ℝ) : ℝ :=
  let area_rectangle := edge_length * edge_length in -- Assuming one specific face is iced completely
  let area_triangle := sqrt 20.25 in -- Triangle area calc
  area_triangle + area_rectangle

theorem cake_and_icing_sum : cubic_cake_piece cube_edge_length + icing_area cube_edge_length = 21 :=
sorry

end cake_and_icing_sum_l289_289846


namespace distinguishable_cube_colorings_l289_289162

theorem distinguishable_cube_colorings :
  let colors := {red, white, blue, green}.to_finset in
  let faces := finset.univ.fin 6 in
  ∃ (f : fin 6 → colors), (∀ (g : fin 6 → colors), (rotate_equiv f g ↔ f = g)) ∧
  (∃ (count : nat), count = 37) :=
by
  let colors := { red, white, blue, green }.to_finset
  let faces := finset.univ.fin 6
  let colorings := faces → colors
  let rotations := finset.univ.fin 24  -- All possible rotations of the cube
  have h : ∃ (f : colorings), (∀ (g : colorings), (rotate_equiv f g ↔ f = g)) :=
    sorry  -- We are given the number and do not prove it here
  use h
  exact 37

noncomputable def rotate_equiv (f g : fin 6 → {red, white, blue, green}.to_finset) : Prop :=
  ∃ rot : fin 24, rotate_cube f = g

end distinguishable_cube_colorings_l289_289162


namespace larger_of_two_numbers_l289_289420

-- Define necessary conditions
def hcf : ℕ := 23
def factor1 : ℕ := 11
def factor2 : ℕ := 12
def lcm : ℕ := hcf * factor1 * factor2

-- Define the problem statement in Lean
theorem larger_of_two_numbers : ∃ (a b : ℕ), a = hcf * factor1 ∧ b = hcf * factor2 ∧ max a b = 276 := by
  sorry

end larger_of_two_numbers_l289_289420


namespace maria_money_left_l289_289369

def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def transportation_cost : ℕ := 80
def num_days : ℕ := 5
def avg_meal_cost_per_day : ℕ := 40
def tourist_tax_rate : ℚ := 0.10
def starting_amount : ℕ := 760

def total_meal_cost : ℕ := num_days * avg_meal_cost_per_day
def expenses_subject_to_tax := hotel_cost + transportation_cost
def tourist_tax := tourist_tax_rate * expenses_subject_to_tax
def total_expenses := ticket_cost + hotel_cost + transportation_cost + total_meal_cost + tourist_tax
def money_left := starting_amount - total_expenses

theorem maria_money_left : money_left = 7 := by
  sorry

end maria_money_left_l289_289369


namespace intersection_point_of_lines_l289_289621

theorem intersection_point_of_lines :
  let k1 := Real.tan (Real.pi / 4)
  let k2 := Real.tan (Real.pi / 3)
  ∃ (x y : ℝ), (y = k1 * x + 2) ∧ (y = k2 * x + sqrt 3 + 1) ∧ (x = -1) ∧ (y = 1) :=
by
  let k1 := Real.tan (Real.pi / 4)
  let k2 := Real.tan (Real.pi / 3)
  use -1, 1
  sorry

end intersection_point_of_lines_l289_289621


namespace min_power_for_84_to_divide_336_l289_289243

theorem min_power_for_84_to_divide_336 : 
  ∃ n : ℕ, (∀ m : ℕ, 84^m % 336 = 0 → m ≥ n) ∧ n = 2 := 
sorry

end min_power_for_84_to_divide_336_l289_289243


namespace max_gcd_lcm_eq_10_l289_289720

open Nat -- Opening the namespace for natural numbers

theorem max_gcd_lcm_eq_10
  (a b c : ℕ) 
  (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) :
  gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_eq_10_l289_289720


namespace cyclic_AFDP_l289_289365

theorem cyclic_AFDP
  (A B C D O F P : Type)
  (h_cyclic : ∀ (A B C D : Type) (O : Type), same_circle A B C D O)
  (h_intersect : ∀ (A B O C D : Type), second_intersect (circumcircle_triangle_ABO A B O) 
                                                           (circumcircle_triangle_CDO C D O) = F)
  (h_point_intersection : ∀ (A C B D : Type), intersection_point (diag_AC A C) (diag_BD B D) = P)  :
  same_circle A F D P :=
by sorry

end cyclic_AFDP_l289_289365


namespace find_expression_value_l289_289187

theorem find_expression_value (a b : ℝ)
  (h1 : a^2 - a - 3 = 0)
  (h2 : b^2 - b - 3 = 0) :
  2 * a^3 + b^2 + 3 * a^2 - 11 * a - b + 5 = 23 :=
  sorry

end find_expression_value_l289_289187


namespace total_spaces_in_game_l289_289007

-- Conditions
def first_turn : ℕ := 8
def second_turn_forward : ℕ := 2
def second_turn_backward : ℕ := 5
def third_turn : ℕ := 6
def total_to_end : ℕ := 37

-- Theorem stating the total number of spaces in the game
theorem total_spaces_in_game : first_turn + second_turn_forward - second_turn_backward + third_turn + (total_to_end - (first_turn + second_turn_forward - second_turn_backward + third_turn)) = total_to_end :=
by sorry

end total_spaces_in_game_l289_289007


namespace contact_point_at_vertices_l289_289992

noncomputable def hyperbola_problem (F1 F2 M N P G : ℝ) : Prop :=
  is_hyperbola F1 F2 M N ∧
  lies_on_hyperbola P F1 F2 ∧
  contact_point_incircle_triangle G P F1 F2 = G ∧
  (G = M ∨ G = N)

theorem contact_point_at_vertices (F1 F2 M N P G : ℝ) 
  (h1 : is_hyperbola F1 F2 M N) 
  (h2 : lies_on_hyperbola P F1 F2) 
  (h3 : contact_point_incircle_triangle G P F1 F2 = G) :
  G = M ∨ G = N :=
sorry

end contact_point_at_vertices_l289_289992


namespace simplified_expression_has_correct_number_of_terms_l289_289903

noncomputable def numberOfTermsInSimplifiedExpression : ℕ :=
  let total_exponent := 2008
  let even_values := List.range' 0 (total_exponent + 1) 2
  even_values.sum'.map (λ a => total_exponent + 1 - a).sum

theorem simplified_expression_has_correct_number_of_terms :
  numberOfTermsInSimplifiedExpression = 1_010_025 :=
sorry

end simplified_expression_has_correct_number_of_terms_l289_289903


namespace system_solutions_l289_289169

noncomputable def sols : set (ℝ × ℝ × ℝ × ℝ) :=
  { (1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1) }

theorem system_solutions :
  { (x, y, u, v) : ℝ × ℝ × ℝ × ℝ |
    x^2 + y^2 + u^2 + v^2 = 4 ∧
    x * y * u + y * u * v + u * v * x + v * x * y = -2 ∧
    x * y * u * v = -1 } = sols :=
sorry

end system_solutions_l289_289169


namespace long_sleeve_shirts_eq_7_l289_289723

-- Define the given conditions
def short_sleeve_shirts : Nat := 39
def washed_shirts : Nat := 20
def not_washed_total : Nat := 66

-- Define the number of long sleeve shirts Oliver has to wash
def long_sleeve_shirts : Nat := 
  let total_shirts := short_sleeve_shirts + L
  let shirts_not_washed := not_washed_total - washed_shirts
  L sorry

-- Prove that the number of long sleeve shirts is 7
theorem long_sleeve_shirts_eq_7 : long_sleeve_shirts = 7 := 
  let total_shirts := short_sleeve_shirts + 7
  let shirts_not_washed := not_washed_total - washed_shirts
  sorry

end long_sleeve_shirts_eq_7_l289_289723


namespace arithmetic_sequence_term_l289_289653

noncomputable def a_n (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem arithmetic_sequence_term
  (a_5 a_45 : ℤ)
  (h1 : a_5 = 33)
  (h2 : a_45 = 153) :
  let d := ((a_45 - a_5) / 40)
      a_1 := (a_5 - 4 * d) 
  in a_n a_1 d 61 = 201 :=
by
  sorry

end arithmetic_sequence_term_l289_289653


namespace fractions_product_l289_289894

theorem fractions_product :
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end fractions_product_l289_289894


namespace exists_constant_c_l289_289730

theorem exists_constant_c (c : ℝ) : 
  ∃ c, ∀ x y : ℝ, 
  ∃ m n : ℤ, Nat.gcd m.nat_abs n.nat_abs = 1 ∧ 
  real.sqrt((x - m) ^ 2 + (y - n) ^ 2) < c * real.log (x ^ 2 + y ^ 2 + 2) :=
sorry

end exists_constant_c_l289_289730


namespace log_a_sub_b_eq_one_half_range_of_a_l289_289982

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b / x - 2 * a + 2

def g (a : ℝ) (x : ℝ) := f a (a - 2) x - 2 * Real.log x

-- Problem (I)
theorem log_a_sub_b_eq_one_half (a b : ℝ) (h0 : 0 < a) (h1 : f'(1) = a - b ∧ a - b = 2) :
  Real.log 4 (a - b) = 1 / 2 :=
sorry

-- Problem (II)
theorem range_of_a (a : ℝ) (h0 : 0 < a) (h1 : ∀ x, x ∈ Set.Ici (1 : ℝ) → f a (a - 2) x - 2 * Real.log x ≥ 0) :
  1 ≤ a :=
sorry

end log_a_sub_b_eq_one_half_range_of_a_l289_289982


namespace max_n_lt_121_div_81_l289_289994

def seq_a (n : ℕ) : ℤ := -3 * n + 4

def common_ratio : ℤ := 
  have h : 1 < 2 := by norm_num
  seq_a 2 - seq_a 1

noncomputable def seq_b (n : ℕ) : ℤ :=
  seq_a 1 * common_ratio ^ (n - 1)

noncomputable def sum_reciprocal (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, 1 / (|seq_b (i+1)| : ℚ)

theorem max_n_lt_121_div_81 : ∃ n ≤ 4, sum_reciprocal n < 121 / 81 := sorry

end max_n_lt_121_div_81_l289_289994


namespace width_of_foil_covered_prism_l289_289860

noncomputable def foil_covered_prism_width : ℕ :=
  let (l, w, h) := (4, 8, 4)
  let inner_width := 2 * l
  let increased_width := w + 2
  increased_width

theorem width_of_foil_covered_prism : foil_covered_prism_width = 10 := 
by
  let l := 4
  let w := 2 * l
  let h := w / 2
  have volume : l * w * h = 128 := by
    sorry
  have width_foil_covered := w + 2
  have : foil_covered_prism_width = width_foil_covered := by
    sorry
  sorry

end width_of_foil_covered_prism_l289_289860


namespace sample_space_and_events_relationships_between_events_event_representations_l289_289390

open Set

namespace DiceEvents

def Ω := {1, 2, 3, 4, 5, 6}

def A := {1}
def B := {2, 4, 6}
def C := {1, 2}
def D := {3, 4, 5, 6}
def E := {3, 6}

theorem sample_space_and_events :
  Ω = {1, 2, 3, 4, 5, 6} ∧
  A = {1} ∧
  B = {2, 4, 6} ∧
  C = {1, 2} ∧
  D = {3, 4, 5, 6} ∧
  E = {3, 6} :=
sorry

theorem relationships_between_events :
  A ⊆ C ∧
  C ∪ D = Ω ∧
  E ⊆ D :=
sorry

theorem event_representations :
  Ω \ D = {1, 2} ∧
  (Ω \ A) ∩ C = {2} ∧
  (Ω \ B) ∪ C = {1, 2, 3} ∧
  (Ω \ D) ∪ (Ω \ E) = {1, 2, 4, 5} :=
sorry

end DiceEvents

end sample_space_and_events_relationships_between_events_event_representations_l289_289390


namespace minimum_photos_l289_289281

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289281


namespace equation_has_two_real_roots_l289_289400

noncomputable def equation : ℝ → ℝ :=
  λ x => 2 * x + real.sqrt (3 * x - 1)

theorem equation_has_two_real_roots :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 = 5 ∧ equation x2 = 5 :=
by
  sorry

end equation_has_two_real_roots_l289_289400


namespace max_gcd_lcm_l289_289716

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l289_289716


namespace PQRS_value_l289_289596

theorem PQRS_value :
  let P := (Real.sqrt 2011 + Real.sqrt 2010)
  let Q := (-Real.sqrt 2011 - Real.sqrt 2010)
  let R := (Real.sqrt 2011 - Real.sqrt 2010)
  let S := (Real.sqrt 2010 - Real.sqrt 2011)
  P * Q * R * S = -1 :=
by
  sorry

end PQRS_value_l289_289596


namespace infinite_product_root_form_l289_289549

theorem infinite_product_root_form :
  (∏ n, (3 ^ (n / 2^n))) = real.sqrt (9: ℝ) :=
sorry

end infinite_product_root_form_l289_289549


namespace product_even_probability_l289_289748

def spinnerA : List ℕ := [1, 1, 2, 2, 3, 4]
def spinnerB : List ℕ := [1, 3, 5, 6]
def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def probability_of_even_product : ℚ := 
  let totalA := spinnerA.length 
  let totalB := spinnerB.length 
  let oddA := spinnerA.count (λ n => ¬is_even n)
  let oddB := spinnerB.count (λ n => ¬is_even n)
  1 - (oddA.toRat / totalA.toRat) * (oddB.toRat / totalB.toRat)

theorem product_even_probability :
  probability_of_even_product = 5 / 8 := by
  sorry

end product_even_probability_l289_289748


namespace consecutive_numbers_100_l289_289442

theorem consecutive_numbers_100 (n : ℕ) (a : ℕ) 
  (h : ∀ sums : finset ℕ, 
        (sums = {s | ∃ (x y z w : ℕ), x < y ∧ y < z ∧ z < w ∧ 
                x ≥ a ∧ w < a + n ∧ s = x + y + z + w} →
        sums.card = 385)) : 
  n = 100 :=
sorry

end consecutive_numbers_100_l289_289442


namespace probability_of_selecting_two_co_captains_l289_289040

theorem probability_of_selecting_two_co_captains : 
  let sizes := [4, 5, 6, 7]
  ∃ (p : ℚ), 
    (p = ∑ s in sizes, (1 / 4) * (6 / (s * (s - 1))) ) ∧ 
    p = 2 / 7 :=
by
  let sizes := [4, 5, 6, 7]
  have h : ∑ s in sizes, (1 / 4) * (6 / (s * (s - 1))) = 2 / 7 := sorry
  use 2 / 7
  exact ⟨h, rfl⟩

end probability_of_selecting_two_co_captains_l289_289040


namespace quadratic_pure_imaginary_roots_l289_289025

-- Defining the given quadratic equation
def quadratic_eq (λ : ℝ) (x : ℂ) : ℂ :=
  (1 - complex.i) * x^2 + (λ + complex.i) * x + (1 + complex.i * λ)

-- The main statement that needs to be proven.
theorem quadratic_pure_imaginary_roots (λ : ℝ) :
  (∀ x : ℂ, quadratic_eq λ x = 0 → x.im = 0) → λ ≠ 2 :=
begin
  sorry
end

end quadratic_pure_imaginary_roots_l289_289025


namespace max_value_of_f_l289_289179

noncomputable def f (x : ℝ) : ℝ := x + √2 * Real.cos x

theorem max_value_of_f :
  ∃ x ∈ Icc (0 : ℝ) (Real.pi / 2), (∀ y ∈ Icc (0 : ℝ) (Real.pi / 2), f y ≤ f x) ∧ f x = Real.pi / 4 + 1 :=
by
  sorry

end max_value_of_f_l289_289179


namespace exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l289_289161

open Nat

theorem exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power :
  ∃ (a : ℕ → ℕ), (∀ k : ℕ, (∃ b : ℕ, a k = b ^ 2)) ∧ (StrictMono a) ∧ (∀ k : ℕ, 13^k ∣ (a k + 1)) :=
sorry

end exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l289_289161


namespace dollars_left_to_spend_l289_289342

/-
Given:
  - John initially has 80 dollars.
  - John spends 15 dollars on snacks.
  - John spends three times the amount spent on snacks on amusement rides.
  - John spends 10 dollars on games.
  
Prove that John has 10 dollars left.
-/

def initial_amount : ℕ := 80
def amount_spent_on_snacks : ℕ := 15
def amount_spent_on_rides : ℕ := 3 * amount_spent_on_snacks
def amount_spent_on_games : ℕ := 10

def total_spent : ℕ := amount_spent_on_snacks + amount_spent_on_rides + amount_spent_on_games

theorem dollars_left_to_spend : initial_amount - total_spent = 10 :=
by
  have total_spent : ℕ := 15 + 45 + 10  -- according to the conditions
  have initial_amount : ℕ := 80
  have h : initial_amount - total_spent = 80 - 70 := by rfl
  exact h

end dollars_left_to_spend_l289_289342


namespace part1_part2_part3_l289_289937

def four_digit_number (digits : Finset ℕ) (n : ℕ) : Prop := 
  digits.card = 6 ∧ 0 ∈ digits ∧ 1 ∈ digits ∧ 2 ∈ digits ∧ 3 ∈ digits ∧ 4 ∈ digits ∧ 5 ∈ digits

def valid_four_digit_numbers (digits : Finset ℕ) : Finset ℕ :=
  (Finset.range 10000).filter (λ n, n ≥ 1000 ∧ (n.digits 10).nodup ∧ (n.digits 10).to_finset ⊆ digits)

def tens_digit_largest (n : ℕ) : Prop :=
  let d := n.digits 10 in
  d.length = 4 ∧ (d!1 > d!0 ∧ d!1 > d!2)

noncomputable def four_digit_count (digits : Finset ℕ) : ℕ := 
  (valid_four_digit_numbers digits).card

noncomputable def tens_largest_count (digits : Finset ℕ) : ℕ := 
  (valid_four_digit_numbers digits).filter tens_digit_largest).card

noncomputable def nth_ascend_number (digits : Finset ℕ) (n : ℕ) : ℕ := 
  (valid_four_digit_numbers digits).sort(≤).nth(n).get_or_else 0

theorem part1 :
  four_digit_number {0, 1, 2, 3, 4, 5} ∧
  four_digit_count {0, 1, 2, 3, 4, 5} = 300 :=
begin
  sorry
end

theorem part2 :
  four_digit_number {0, 1, 2, 3, 4, 5} ∧
  tens_largest_count {0, 1, 2, 3, 4, 5} = 100 :=
begin
  sorry
end
  
theorem part3 :
  four_digit_number {0, 1, 2, 3, 4, 5} ∧
  nth_ascend_number {0, 1, 2, 3, 4, 5} 84 = 2301 :=
begin
  sorry
end

end part1_part2_part3_l289_289937


namespace weighted_average_score_l289_289907

def weight (subject_mark : Float) (weight_percentage : Float) : Float :=
    subject_mark * weight_percentage

theorem weighted_average_score :
    (weight 61 0.2) + (weight 65 0.25) + (weight 82 0.3) + (weight 67 0.15) + (weight 85 0.1) = 71.6 := by
    sorry

end weighted_average_score_l289_289907


namespace num_integral_values_BC_l289_289695

-- Given data
variables (A B C D E F : Type)
variables [IsTriangle A B C] [IsAngleBisector A B C D] [OnLineInter BC D] [OnLineInter AC E] [OnLineInter BC F]
variables [Parallel AD EF] [DividesTriangleIntoThreeEqualParts AD EF]

theorem num_integral_values_BC (BC : ℝ) (h1 : AB = 7) (h2 : AC = 2 * AB) :
  ∃ n : ℕ, n = 13 ∧ ∀ BC, (7 < BC ∧ BC < 21) → ∃! m, m = BC ∧ BC ∈ ℕ :=
by
  sorry

end num_integral_values_BC_l289_289695


namespace tangent_line_second_quadrant_l289_289971

theorem tangent_line_second_quadrant :
  ∃ (P : ℝ × ℝ), 
  P.1 < 0 ∧ P.2 > 0 ∧
  P.2 = P.1^3 - 10 * P.1 + 3 ∧
  (deriv (λ x : ℝ, x^3 - 10 * x + 3) P.1 = 2) ∧
  (∃ m b : ℝ, y = 2 * x + 19) :=
begin
  sorry
end

end tangent_line_second_quadrant_l289_289971


namespace find_real_numbers_calculate_conjugate_product_l289_289953

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex numbers z1, z2, z3
def z1 : ℂ := 2 * i
def z2 : ℂ := 1 - 3 * i
def z3 : ℂ := 1 - 2 * i

-- Problem (1): Find real numbers x and y
theorem find_real_numbers:
  ∃ (x y: ℝ), (x / z1 - 5 / z2 = y / z3) ∧ x = -1 ∧ y = -(5 / 2) :=
by
  sorry

-- Problem (2): Calculate the product of the conjugates of z1 and z2
theorem calculate_conjugate_product:
  conj z1 * conj z2 = 6 - 2 * i :=
by
  sorry

end find_real_numbers_calculate_conjugate_product_l289_289953


namespace shaded_squares_7x7_shaded_squares_101x101_unshaded_squares_given_shaded_41_total_squares_given_unshaded_196_l289_289085

-- Statement for Part (a)
theorem shaded_squares_7x7 : ∀ (n : Nat), n = 7 → (let shaded := 2 * (n - 1) + 1 in shaded = 13) :=
by intro n hn; sorry

-- Statement for Part (b)
theorem shaded_squares_101x101 : ∀ (n : Nat), n = 101 → (let shaded := n + (n - 1) in shaded = 201) :=
by intro n hn; sorry

-- Statement for Part (c)
theorem unshaded_squares_given_shaded_41 : ∀ (n shaded : Nat), shaded = 41 → (let total := n * n in let unshaded := total - shaded in (2 * n - 1 = shaded) → unshaded = 400) :=
by intro n shaded hs ht; sorry

-- Statement for Part (d)
theorem total_squares_given_unshaded_196 : ∀ (m : Nat), (let unshaded := m * m - (2 * m - 1) in unshaded = 196) → m * m = 225 :=
by intro m hu; sorry

end shaded_squares_7x7_shaded_squares_101x101_unshaded_squares_given_shaded_41_total_squares_given_unshaded_196_l289_289085


namespace height_difference_percentage_l289_289127

variable (a b : ℝ) (h : b = 1.25 * a)

theorem height_difference_percentage : (b - a) / b * 100 = 20 :=
by
  have h₁ := calc
    b - a = 1.25 * a - a : by rw [h]
    ... = 0.25 * a : by ring
  have h₂ := calc
    (b - a) / b * 100 = (0.25 * a) / (1.25 * a) * 100 : by rw [h, h₁]
    ... = 0.25 / 1.25 * 100 : by field_simp [ne_of_gt (show a > 0 from sorry)]
    ... = 0.20 * 100 : by norm_num
    ... = 20 : by norm_num
  exact h₂

end height_difference_percentage_l289_289127


namespace find_D_of_symmetric_circle_l289_289841

theorem find_D_of_symmetric_circle (D E F : ℝ) :
  (∃ (x y : ℝ), x - y + 4 = 0 ∧ x + 3y = 0 ∧ x^2 + y^2 + D * x + E * y + F = 0 ∧ 
  (-D / 2, -E / 2) = (-3 / 2, 1 / 2)) → D = 3 := 
by 
  sorry

end find_D_of_symmetric_circle_l289_289841


namespace restaurant_revenue_l289_289888

theorem restaurant_revenue (x y z : ℝ) : 
  let kids_meals := 12
  let adult_meals := (2/3 : ℝ) * kids_meals
  let senior_meals := (1/3 : ℝ) * kids_meals
  (kids_meals * x) + (adult_meals * y) + (senior_meals * z) = 12 * x + 8 * y + 4 * z :=
by
  let kids_meals := 12
  let adult_meals := (2/3 : ℝ) * kids_meals
  let senior_meals := (1/3 : ℝ) * kids_meals
  have h1 : (kids_meals * x) + (adult_meals * y) + (senior_meals * z) = 12 * x + ((2/3 : ℝ) * 12 * y) + ((1/3 : ℝ) * 12 * z)
  calc
    (kids_meals * x) + (adult_meals * y) + (senior_meals * z) = 12 * x + ((2/3 : ℝ) * 12 * y) + ((1/3 : ℝ) * 12 * z) : by sorry
  have h2 : ((2/3 : ℝ) * 12 * y) = 8 * y := by sorry
  have h3 : ((1/3 : ℝ) * 12 * z) = 4 * z := by sorry
  rw [h2, h3] at h1
  exact h1

end restaurant_revenue_l289_289888


namespace find_angle_B_max_perimeter_l289_289231

variables (A B C a b c : ℝ)

-- Conditions
def conditions (A B C a b c : ℝ) :=
  b = 3 ∧
  (cos A / cos B + sin A / sin B = 2 * c / b)

-- To Prove: (I) Angle B is 60 degrees
theorem find_angle_B (A B C a b c : ℝ) (h : conditions A B C a b c) : B = π / 3 :=
sorry

-- To Prove: (II) Maximum value of the perimeter L of triangle ABC is 9
theorem max_perimeter (A B C a b c : ℝ) (h : conditions A B C a b c) : a + b + c ≤ 9 :=
sorry

end find_angle_B_max_perimeter_l289_289231


namespace largest_in_list_l289_289848

theorem largest_in_list (l : List ℕ) (h1 : l.length = 5)
  (h2 : ∀ x ∈ l, 0 < x)
  (h3 : ∀ x ∈ l, l.count x > 1 → x = 9)
  (h4 : l.nth 2 = some 10)
  (h5 : l.sum = 55) :
  16 ∈ l :=
sorry

end largest_in_list_l289_289848


namespace not_monotonic_in_interval_l289_289760

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a * x - 5

theorem not_monotonic_in_interval (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f a x ≠ (1/3) * x^3 - x^2 + a * x - 5) → a ≥ 1 ∨ a ≤ -3 :=
sorry

end not_monotonic_in_interval_l289_289760


namespace max_mondays_in_45_days_l289_289019

theorem max_mondays_in_45_days (n : ℕ) (h : n = 45) : 
  ∃ m, m ≤ 7 ∧ (∀ x < n, (x % 7 = 0) → x / 7 < 7) :=
by
  existsi 7
  split
  . exact nat.le_refl 7
  . intros x hx
    sorry

end max_mondays_in_45_days_l289_289019


namespace find_angle_phi_l289_289509

-- Definitions for the conditions given in the problem
def folded_paper_angle (φ : ℝ) : Prop := 0 < φ ∧ φ < 90

def angle_XOY := 144

-- The main statement to be proven
theorem find_angle_phi (φ : ℝ) (h1 : folded_paper_angle φ) : φ = 81 :=
sorry

end find_angle_phi_l289_289509


namespace largest_number_is_763210_l289_289798

-- Define the set of digits
def distinct_digits_sum_17 : Set ℕ := {n | (∃ (digits : List ℕ),
  n = digits.foldl (λ acc d, acc + d) 0 ∧
  digits.nodup ∧
  digits.sum = 17)}

-- Define the goal to show that 763210 is the largest such number
theorem largest_number_is_763210 
  (h : 763210 ∈ distinct_digits_sum_17): 
  ∀ n ∈ distinct_digits_sum_17, n ≤ 763210 :=
sorry

end largest_number_is_763210_l289_289798


namespace kite_area_l289_289786

theorem kite_area (c d : ℝ) (H1 : ∀ x : ℝ, (cx^2 + 3 = 0 → x = sqrt (-3/c) ∨ x = - sqrt (-3/c)) ∧ (7 - dx^2 = 0 → x = sqrt (7/d) ∨ x = - sqrt (7/d)))
(H2 : c * d = -7/3)
(H3 : 0 < -3/c) 
(H4 : 20 = (1/2) * (2 * sqrt (-3/c)) * 4):
  c + d = 18/25 := 
sorry

end kite_area_l289_289786


namespace S8_value_l289_289973

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

def condition_a3_a6 (a : ℕ → ℝ) : Prop :=
  a 3 = 9 - a 6

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_formula : sum_of_first_n_terms S a)
  (h_condition : condition_a3_a6 a) :
  S 8 = 72 :=
by
  sorry

end S8_value_l289_289973


namespace total_seats_taken_l289_289771

def students_per_bus : ℝ := 14.0
def number_of_buses : ℝ := 2.0

theorem total_seats_taken :
  students_per_bus * number_of_buses = 28.0 :=
by
  sorry

end total_seats_taken_l289_289771


namespace no_nonzero_solution_l289_289824

theorem no_nonzero_solution (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
by 
  sorry

end no_nonzero_solution_l289_289824


namespace small_cubes_with_painted_faces_l289_289496

-- Definitions based on conditions
def large_cube_edge : ℕ := 8
def small_cube_edge : ℕ := 2
def division_factor : ℕ := large_cube_edge / small_cube_edge
def total_small_cubes : ℕ := division_factor ^ 3

-- Proving the number of cubes with specific painted faces.
theorem small_cubes_with_painted_faces :
  (8 : ℤ) = 8 ∧ -- 8 smaller cubes with three painted faces
  (24 : ℤ) = 24 ∧ -- 24 smaller cubes with two painted faces
  (24 : ℤ) = 24 := -- 24 smaller cubes with one painted face
by
  sorry

end small_cubes_with_painted_faces_l289_289496


namespace smallest_number_divisible_by_conditions_l289_289819

theorem smallest_number_divisible_by_conditions:
  ∃ n : ℕ, (∀ d ∈ [8, 12, 22, 24], d ∣ (n - 12)) ∧ (n = 252) :=
by
  sorry

end smallest_number_divisible_by_conditions_l289_289819


namespace find_a_b_prime_l289_289683

theorem find_a_b_prime (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hrel_prime : nat.coprime a b):
  let T := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50} in
  let y_eq_mx := {p : ℕ × ℕ | p.2 ≤ (p.1 * a / b)} in
  (card (T ∩ y_eq_mx) = 500) → (1 ≤ a ∧ 1 ≤ b ∧ a + b = 6) :=
begin
  intros,
  sorry
end

end find_a_b_prime_l289_289683


namespace strip_covers_cube_l289_289861

   -- Define the given conditions
   def strip_length := 12
   def strip_width := 1
   def cube_edge := 1
   def layers := 2

   -- Define the main statement to be proved
   theorem strip_covers_cube : 
     (strip_length >= 6 * cube_edge / layers) ∧ 
     (strip_width >= cube_edge) ∧ 
     (layers == 2) → 
     true :=
   by
     intro h
     sorry
   
end strip_covers_cube_l289_289861


namespace astronauts_experiment_l289_289527

-- Define the programs
inductive Program
| A | B | C | D | E | F

open Program

-- Define the condition: B and C cannot be adjacent to D.
def no_adjacent_bc_d (seq : List Program) : Prop :=
  ∀ i j, (seq.nth i = some B ∨ seq.nth i = some C) →
         (seq.nth j = some D) →
         abs (i - j) ≠ 1

-- Define the sequence generator
def possible_sequences : List (List Program) :=
  List.permutations [A, B, C, D, E, F]

-- Condition: B and C cannot be adjacent to D
def valid_sequences : Nat :=
  (possible_sequences.filter no_adjacent_bc_d).length

theorem astronauts_experiment : valid_sequences = 288 := by
  sorry

end astronauts_experiment_l289_289527


namespace rectangle_length_l289_289001

theorem rectangle_length :
  ∃ y : ℝ, y.round = 37 ∧
           6 * (2/3 * y^2) = 5400 :=
begin
  sorry
end

end rectangle_length_l289_289001


namespace find_S17_l289_289587

-- Definitions based on the conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (a1 : ℝ) (d : ℝ)

-- Conditions from the problem restated in Lean
axiom arithmetic_sequence : ∀ n, a n = a1 + (n - 1) * d
axiom sum_of_n_terms : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)
axiom arithmetic_subseq : 2 * a 7 = a 5 + 3

-- Theorem to prove
theorem find_S17 : S 17 = 51 :=
by sorry

end find_S17_l289_289587


namespace cos_sum_identity_l289_289729

theorem cos_sum_identity {α β γ : ℝ} (hαβγ : α + β + γ = 180) : 
  cos(α * (pi / 180))^2 + cos(β * (pi / 180))^2 + cos(γ * (pi / 180))^2 + 2 * cos(α * (pi / 180)) * cos(β * (pi / 180)) * cos(γ * (pi / 180)) = 1 :=
sorry

end cos_sum_identity_l289_289729


namespace exists_naughty_set_l289_289266

def isCycle (n : ℕ) (k : ℕ) (a : Fin k → Fin (n*n)) : Prop :=
  k ≥ 4 ∧ ∀ i : Fin k, shares_edge_with n (a i) (a ((i + 1) % k))

def isNaughtySet (n : ℕ) (X : Set (Fin (n*n))) : Prop :=
  ∀ (k : ℕ) (a : Fin k → Fin (n*n)), isCycle n k a → ∃ i : Fin k, a i ∈ X

theorem exists_naughty_set (C : ℝ) : 
  (∀ (n : ℕ), n ≥ 2 → ∃ X : Set (Fin (n*n)), isNaughtySet n X ∧ X.card ≤ C * n^2) ↔ C = 0.5 := 
by
  sorry

end exists_naughty_set_l289_289266


namespace minimum_photos_l289_289295

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289295


namespace distance_A_to_BC_l289_289943

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def coord_diff (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def dot_product (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y + p1.z * p2.z

def magnitude (p : Point) : ℝ :=
  sqrt (p.x^2 + p.y^2 + p.z^2)

def distance_from_point_to_line (A B C : Point) : ℝ :=
  let AB := coord_diff A B
  let BC := coord_diff B C
  let mag_AB := magnitude AB
  let mag_BC := magnitude BC
  let dot_AB_BC := dot_product AB BC
  let cos_theta := dot_AB_BC / (mag_AB * mag_BC)
  mag_AB * sqrt (1 - cos_theta^2)

theorem distance_A_to_BC :
  let A := Point.mk 0 0 2
  let B := Point.mk 1 0 2
  let C := Point.mk 0 2 0
  distance_from_point_to_line A B C = 2 * sqrt 2 / 3 := by
  sorry

end distance_A_to_BC_l289_289943


namespace average_age_of_team_l289_289074

theorem average_age_of_team 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (remaining_avg : ℕ → ℕ) 
  (h1 : n = 11)
  (h2 : captain_age = 27)
  (h3 : wicket_keeper_age = 28)
  (h4 : ∀ A, remaining_avg A = A - 1)
  (h5 : ∀ A, 11 * A = 9 * (remaining_avg A) + captain_age + wicket_keeper_age) : 
  ∃ A, A = 32 :=
by
  sorry

end average_age_of_team_l289_289074


namespace no_full_circle_by_any_bead_l289_289034

/--
There are 2009 beads freely placed on a ring. In one move, any bead can be moved so that it ends up exactly in the middle between two neighboring beads.
There do not exist any initial configurations or sequence of moves that allow any bead to complete a full circle on the ring.
-/
theorem no_full_circle_by_any_bead (n : ℕ) (h : n = 2009) :
  ∀ (initial_positions : Fin n → ℝ), 
  ¬∃ sequence_of_moves : ℕ → Fin n, 
    ∃ k : ℕ, 
    let bead_trajectory := 
      (λ i : ℕ, 
        if i = 0 
        then initial_positions (sequence_of_moves i) 
        else (initial_positions (sequence_of_moves i) + initial_positions (sequence_of_moves (i-1))) / 2)
    in bead_trajectory k ≥ 2 * Math.pi :=
by
  sorry

end no_full_circle_by_any_bead_l289_289034


namespace concrete_needed_l289_289518

-- Define the dimensions in yards
def width : ℝ := 4 / 3
def length : ℝ := 80 / 3
def thickness : ℝ := 1 / 9

-- Volume in cubic yards
def volume : ℝ := width * length * thickness

-- Goal: the number of cubic yards of concrete needed (rounded up to nearest whole number)
theorem concrete_needed : Int.ceil volume = 4 :=
by
  sorry

end concrete_needed_l289_289518


namespace shadow_boundary_l289_289199

theorem shadow_boundary (x y : ℝ) : 
  let O := (0, 0, 1)
      radius := 2
      P := (0, -2, 3)
  in y = sqrt 2 * x + 1 :=
sorry

end shadow_boundary_l289_289199


namespace distance_between_cities_l289_289414

theorem distance_between_cities:
    ∃ (x y : ℝ),
    (x = 135) ∧
    (y = 175) ∧
    (7 / 9 * x = 105) ∧
    (x + 7 / 9 * x + y = 415) ∧
    (x = 27 / 35 * y) :=
by
  sorry

end distance_between_cities_l289_289414


namespace equation_of_tangent_line_through_1_1_l289_289156

open Function

noncomputable def tangent_line_equation (p : ℝ × ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (tangent_slope : ℝ) : Prop :=
  ∃ (m b : ℝ), (p = (1,1) ∧ f x = x ^ 2 ∧ f' x = 2 * x ∧ m = tangent_slope) ∧ 
  ((tangent_slope = 2 ∧ m = 2 ∧ (2 * x - y - 1 = 0))

theorem equation_of_tangent_line_through_1_1 : 
  tangent_line_equation (1, 1) (λ x, x^2) (λ x, 2 * x) 2 :=
  sorry

end equation_of_tangent_line_through_1_1_l289_289156


namespace money_left_after_shopping_l289_289059

-- Definitions based on conditions
def initial_amount : ℝ := 5000
def percentage_spent : ℝ := 0.30
def amount_spent : ℝ := percentage_spent * initial_amount
def remaining_amount : ℝ := initial_amount - amount_spent

-- Theorem statement based on the question and correct answer
theorem money_left_after_shopping : remaining_amount = 3500 :=
by
  sorry

end money_left_after_shopping_l289_289059


namespace cost_of_gas_per_gallon_l289_289335

-- Definitions based on the conditions
def hours_driven_1 : ℕ := 2
def speed_1 : ℕ := 60
def hours_driven_2 : ℕ := 3
def speed_2 : ℕ := 50
def mileage_per_gallon : ℕ := 30
def total_gas_cost : ℕ := 18

-- An assumption to simplify handling dollars and gallons
noncomputable def cost_per_gallon : ℕ := total_gas_cost / (speed_1 * hours_driven_1 + speed_2 * hours_driven_2) * mileage_per_gallon

theorem cost_of_gas_per_gallon :
  cost_per_gallon = 2 := by
sorry

end cost_of_gas_per_gallon_l289_289335


namespace cards_divisible_by_100_l289_289193

open Nat

-- Define the problem statement
theorem cards_divisible_by_100 :
  let cards := Finset.range 5000
  let valid_pairs := cards.filter (λ n, ∃ m ∈ cards, (n + m) % 100 = 0)
  valid_pairs.card = 124950 :=
by
  sorry

end cards_divisible_by_100_l289_289193


namespace decrease_interval_f_l289_289929

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem decrease_interval_f :
  ∀ x : ℝ, differentiable ℝ f → 0 < x ∧ x < 1 → deriv f x < 0 :=
by
  sorry

end decrease_interval_f_l289_289929


namespace speed_of_jogger_l289_289500

noncomputable def jogger_speed_problem (jogger_distance_ahead train_length train_speed_kmh time_to_pass : ℕ) :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := jogger_distance_ahead + train_length
  let relative_speed := total_distance / time_to_pass
  let jogger_speed_ms := train_speed_ms - relative_speed
  let jogger_speed_kmh := jogger_speed_ms * 3600 / 1000
  jogger_speed_kmh

theorem speed_of_jogger :
  jogger_speed_problem 240 210 45 45 = 9 :=
by
  sorry

end speed_of_jogger_l289_289500


namespace iris_total_spending_l289_289323

theorem iris_total_spending :
  ∀ (price_jacket price_shorts price_pants : ℕ), 
  price_jacket = 10 → 
  price_shorts = 6 → 
  price_pants = 12 → 
  (3 * price_jacket + 2 * price_shorts + 4 * price_pants) = 90 :=
by
  intros price_jacket price_shorts price_pants
  sorry

end iris_total_spending_l289_289323


namespace probability_of_purple_probability_of_blue_or_purple_l289_289087

def total_jelly_beans : ℕ := 60
def purple_jelly_beans : ℕ := 5
def blue_jelly_beans : ℕ := 18

theorem probability_of_purple :
  (purple_jelly_beans : ℚ) / total_jelly_beans = 1 / 12 :=
by
  sorry
  
theorem probability_of_blue_or_purple :
  (blue_jelly_beans + purple_jelly_beans : ℚ) / total_jelly_beans = 23 / 60 :=
by
  sorry

end probability_of_purple_probability_of_blue_or_purple_l289_289087


namespace find_angle_C_l289_289595

noncomputable def angle_C : Real :=
  let G : EucSpace := centroid of triangle ABC
  let a : Real := side opposite to angle A
  let b : Real := side opposite to angle B
  let c : Real := side opposite to angle C
  assume h1 : G is centroid of triangle ABC
  assume h2 : a \overrightarrow{GA} + (3/5)b \overrightarrow{GB} + (3/7)c \overrightarrow{GC} = 0
  sorry

theorem find_angle_C
  (G : EucSpace) 
  (a b c : Real)
  (h1 : is_centroid G ABC)
  (h2 : a * (GA) + (3 / 5) * b * (GB) + (3 / 7) * c * (GC) = 0) :
  angle_C = (2 * Real.pi / 3) :=
sorry

end find_angle_C_l289_289595


namespace sum_of_products_l289_289775

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  ab + bc + ac = 72 := 
by 
  sorry

end sum_of_products_l289_289775


namespace area_difference_is_30_point_5_l289_289858

-- Define the conditions of the problem
def rectangle_diagonal := 10 -- inches
def rectangle_short_side := 6 -- inches
def circle_diameter := 10 -- inches
def pi_approx := 3.14159

-- Calculate the values based on conditions
def longer_side := real.sqrt (rectangle_diagonal^2 - rectangle_short_side^2)
def rectangle_area := rectangle_short_side * longer_side
def circle_radius := circle_diameter / 2
def circle_area := pi * (circle_radius^2)
def area_difference := circle_area - rectangle_area

-- Prove the difference in area
theorem area_difference_is_30_point_5 : (area_difference : real) ≈ 30.5 :=
by
  let l := real.sqrt (rectangle_diagonal^2 - rectangle_short_side^2)
  have h1 : l = 8 := by sorry
  have h2 : rectangle_area = 48 := by
    rw [h1]
    sorry
  let r := circle_diameter / 2
  have h3 : r = 5 := by
    sorry
  have h4 : circle_area ≈ 78.54 := by
    sorry
  show area_difference ≈ 30.5 := by
    sorry

end area_difference_is_30_point_5_l289_289858


namespace equal_intersection_angle_l289_289749

variables {A B C D A1 B1 C1 D1 O O1 : Type}
variables [EuclideanGeometry.Point O] [EuclideanGeometry.Point O1]
variables [EuclideanGeometry.Line (O, O1)] [EuclideanGeometry.Line (A1, B)] [EuclideanGeometry.Line (C1, D)]

/--
Given two squares ABCD and A1B1C1D1 in the same plane coinciding at vertices C and B1, and centers O and O1 
of the respective squares, prove that the line OO1 intersects segments A1B and C1D at equal angles.
-/
theorem equal_intersection_angle 
  (h_squares : ∀ {p : Type}, EuclideanGeometry.Point p)
  (h_coincide : C = B1) 
  (h_center_O : EuclideanGeometry.Center O ABCD)
  (h_center_O1 : EuclideanGeometry.Center O1 A1B1C1D1) :
  EuclideanGeometry.Angles_Equal (EuclideanGeometry.IntersectAngle (O, O1) (A1, B))
                                 (EuclideanGeometry.IntersectAngle (O, O1) (C1, D)) :=
sorry

end equal_intersection_angle_l289_289749


namespace smallest_degree_horizontal_asymptote_l289_289149

theorem smallest_degree_horizontal_asymptote (p : Polynomial ℝ) :
  (∃ d : ℕ, degree p = d ∧ d >= 7) ↔ (∃ h : ℝ, horizontal_asymptote (fun x => (3 * x^7 + 4 * x^3 - 2 * x - 5) / p x) h) :=
sorry

end smallest_degree_horizontal_asymptote_l289_289149


namespace distance_walked_by_man_l289_289849

theorem distance_walked_by_man (x t : ℝ) (h1 : d = (x + 0.5) * (4 / 5) * t) (h2 : d = (x - 0.5) * (t + 2.5)) : d = 15 :=
by
  sorry

end distance_walked_by_man_l289_289849


namespace problem1_eval_problem2_eval_1_problem2_eval_2_l289_289483

-- Problem 1
theorem problem1_eval : (-2021) ^ 0 + 3 * Real.sqrt 27 + (1 - 3^(-2) * 18) = 9 * Real.sqrt 3 :=
by sorry

-- Problem 2
variables (x y : ℝ)

-- Intersection Points
def intersection_points : List (ℝ × ℝ) :=
  [(1, 2), (-1, -2)]

theorem problem2_eval_1 (hx : x = 1) (hy : y = 2) : 
  (x^2 - y^2) / (x^2 - 2*x*y + y^2) * ((x - y) * (2*x + 3*y) / (x + y)) - x*y * (2/x + 3/y) = 1 :=
by sorry

theorem problem2_eval_2 (hx : x = -1) (hy : y = -2) : 
  (x^2 - y^2) / (x^2 - 2*x*y + y^2) * ((x - y) * (2*x + 3*y) / (x + y)) - x*y * (2/x + 3/y) = -1 :=
by sorry

end problem1_eval_problem2_eval_1_problem2_eval_2_l289_289483


namespace Ron_book_picking_times_l289_289393

theorem Ron_book_picking_times (couples members : ℕ) (weeks people : ℕ) (Ron wife picks_per_year : ℕ) 
  (h1 : couples = 3) 
  (h2 : members = 5) 
  (h3 : Ron = 1) 
  (h4 : wife = 1) 
  (h5 : weeks = 52) 
  (h6 : people = 2 * couples + members + Ron + wife) 
  (h7 : picks_per_year = weeks / people) 
  : picks_per_year = 4 :=
by
  -- Definition steps can be added here if needed, currently immediate from conditions h1 to h7
  sorry

end Ron_book_picking_times_l289_289393


namespace max_two_scoop_sundaes_l289_289883

theorem max_two_scoop_sundaes : 
  let n := 6 in 
  let r := 2 in 
  nat.choose n r = 15 :=
by
  sorry

end max_two_scoop_sundaes_l289_289883


namespace tangent_line_eqn_zero_diff_bound_l289_289607

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.exp x) - x^2 + Real.pi * x

-- Statement for Part 1: Tangent Line Equation
theorem tangent_line_eqn :
  let y := (1 + Real.pi) * (0 : ℝ) in
  y = 0 :=
by
  sorry

-- Statement for Part 2: Zero Difference Bound
theorem zero_diff_bound (m : ℝ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x2 ≤ Real.pi)
    (h3 : f x1 = m) (h4 : f x2 = m) :
    |x2 - x1| ≤ Real.pi - (2 * m) / (Real.pi + 1) :=
by
  sorry

end tangent_line_eqn_zero_diff_bound_l289_289607


namespace range_of_m_l289_289232

-- Definitions according to the problem conditions
def p (x : ℝ) : Prop := (-2 ≤ x ∧ x ≤ 10)
def q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m) ∧ m > 0

-- Rephrasing the problem statement in Lean
theorem range_of_m (x : ℝ) (m : ℝ) :
  (∀ x, p x → q x m) → m ≥ 9 :=
sorry

end range_of_m_l289_289232


namespace jason_initial_cards_l289_289332

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end jason_initial_cards_l289_289332


namespace coins_problem_l289_289671

theorem coins_problem : 
  ∀ (total coins heads_percentage tails_turn_over : ℕ), 
    total = 150 → 
    heads_percentage = 40 → 
    tails_turn_over = 15 →
    (0.4 * total + tails_turn_over = total / 2) :=
by {
  intros total _ _ _ total_eq heads_eq tails_turn_eq,
  sorry
}

end coins_problem_l289_289671


namespace problem_statement_l289_289590

theorem problem_statement (a b : ℝ) (h1 : 2^a = 10) (h2 : 5^b = 10) : (1 / a) + (1 / b) = 1 := 
by
  sorry

end problem_statement_l289_289590


namespace function_increasing_interval_l289_289546

theorem function_increasing_interval :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi),
  (2 * Real.sin ((Real.pi / 6) - 2 * x) : ℝ)
  ≤ 2 * Real.sin ((Real.pi / 6) - 2 * x + 1)) ↔ (x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6)) :=
sorry

end function_increasing_interval_l289_289546


namespace inequality_solution_l289_289004

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x) : 
  2021 * (x ^ 10) - 1 ≥ 2020 * x ↔ x = 1 := 
by
  sorry

end inequality_solution_l289_289004


namespace integral_BC_values_count_l289_289692

noncomputable def triangle_side_counts (AB BC AC : ℝ) (h1 : AB = 7) (h2 : AC = 2 * AB) (h3 : 7 < BC) (h4 : BC < 21) : ℕ :=
  let eligible_BC_values := {n : ℕ | 7 < n ∧ n < 21}.card
  eligible_BC_values

theorem integral_BC_values_count : ∀ AB BC AC, 
  AB = 7 →
  AC = 2 * AB →
  7 < BC →
  BC < 21 →
  triangle_side_counts AB BC AC 7 (2 * 7) bc bc = 13 :=
by
  intros AB BC AC h1 h2 h3 h4
  rw [triangle_side_counts]
  sorry

end integral_BC_values_count_l289_289692


namespace correct_calculation_l289_289464

theorem correct_calculation (m n : ℝ) :
  3 * m^2 * n - 3 * m^2 * n = 0 ∧
  ¬ (3 * m^2 - 2 * m^2 = 1) ∧
  ¬ (3 * m^2 + 2 * m^2 = 5 * m^4) ∧
  ¬ (3 * m + 2 * n = 5 * m * n) := by
  sorry

end correct_calculation_l289_289464


namespace largest_number_with_unique_digits_summing_to_17_is_98_l289_289801

theorem largest_number_with_unique_digits_summing_to_17_is_98 :
  ∀ n : ℕ, (all_different n) → (digits_sum n = 17) → n = 98 :=
by
  sorry

def all_different (n : ℕ) : Prop :=
  let digits := n.to_digits
  digits.nodup

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.to_digits
  digits.sum

end largest_number_with_unique_digits_summing_to_17_is_98_l289_289801


namespace relationship_y1_y2_y3_l289_289202

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 2

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-1, quadratic_function (-1)⟩
def B : Point := ⟨1, quadratic_function 1⟩
def C : Point := ⟨2, quadratic_function 2⟩

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 :
  A.y = B.y ∧ A.y > C.y :=
by
  sorry

end relationship_y1_y2_y3_l289_289202


namespace slope_of_line_eq_l289_289916

theorem slope_of_line_eq (x y : ℝ) (h : x / 4 + y / 5 = 1) : 
  let m := (-5 / 4) in 
  ∃ b : ℝ, y = m * x + b :=
by
  sorry

end slope_of_line_eq_l289_289916


namespace polynomial_zero_l289_289143

theorem polynomial_zero (P : Polynomial ℤ) (h : ∀ n : ℕ, n > 0 → (n : ℤ) ∣ P.eval (2^n)) : P = 0 := 
sorry

end polynomial_zero_l289_289143


namespace largest_number_with_unique_digits_summing_to_17_l289_289794

theorem largest_number_with_unique_digits_summing_to_17 : ∃ n : ℕ, 
  (∀ i j : ℕ, i ≠ j → (List.toDigits n).nth i ≠ (List.toDigits n).nth j) ∧
  (List.sum (List.filterMap (fun x => if x ∈ (List.toDigits n) then some (x : ℕ) else none) (finset.range 10))) = 17 ∧
  n = 7543210 :=
begin
  sorry
end

end largest_number_with_unique_digits_summing_to_17_l289_289794


namespace find_MN_l289_289257

open Real

-- Define the vertices of the triangle and their coordinates
variable {A B C D M N : Point}
variable {a b c : Line}
variable (h : IsTriangle A B C)
variable (AB BC AC : ℝ)
variable (hAB : length A B = 9)
variable (hBC : length B C = 8)
variable (hAC : length A C = 7)

-- Define the bisector of angle A
variable (h_angle_bisector : IsAngleBisector A B C D)

-- Define the circle passing through A and touching BC at D
variable (Circle : Circle)
variable (has_A : IsOnCircle Circle A)
variable (touches_D : IsTangent Circle BC D)

-- Define points M and N on AB and AC respectively
variable (cuts_AB : on_line M AB)
variable (cuts_AC : on_line N AC)

-- Deduced conditions from the problem setup
variable (H1 : the_circle_passes-through A M N D)

-- Main statement: proof that MN equals 6
theorem find_MN :
  length M N = 6 := 
sorry

end find_MN_l289_289257


namespace smallest_number_of_people_l289_289808

open Nat

theorem smallest_number_of_people (x : ℕ) :
  (∃ x, x % 18 = 0 ∧ x % 50 = 0 ∧
  (∀ y, y % 18 = 0 ∧ y % 50 = 0 → x ≤ y)) → x = 450 :=
by
  sorry

end smallest_number_of_people_l289_289808


namespace shoveling_problem_l289_289480

variable (S : ℝ) -- Wayne's son's shoveling rate (driveways per hour)
variable (W : ℝ) -- Wayne's shoveling rate (driveways per hour)
variable (T : ℝ) -- Time it takes for Wayne's son to shovel the driveway alone (hours)

theorem shoveling_problem 
  (h1 : W = 6 * S)
  (h2 : (S + W) * 3 = 1) : T = 21 := 
by
  sorry

end shoveling_problem_l289_289480


namespace simplify_and_evaluate_l289_289397

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ 0) (h₂ : x = 2) : 
  (1 + 1 / x) / ((x^2 - 1) / x) = 1 := 
by 
  sorry

end simplify_and_evaluate_l289_289397


namespace find_divisor_l289_289646

theorem find_divisor (q r D : ℕ) (hq : q = 120) (hr : r = 333) (hD : 55053 = D * q + r) : D = 456 :=
by
  sorry

end find_divisor_l289_289646


namespace largest_number_with_unique_digits_summing_to_17_is_98_l289_289802

theorem largest_number_with_unique_digits_summing_to_17_is_98 :
  ∀ n : ℕ, (all_different n) → (digits_sum n = 17) → n = 98 :=
by
  sorry

def all_different (n : ℕ) : Prop :=
  let digits := n.to_digits
  digits.nodup

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.to_digits
  digits.sum

end largest_number_with_unique_digits_summing_to_17_is_98_l289_289802


namespace find_exponent_l289_289180

theorem find_exponent (x : ℝ) : (14^x * 5^3 / 568 = 43.13380281690141) → x = 2 :=
by
  sorry

end find_exponent_l289_289180


namespace book_donations_l289_289049

-- Define the conditions
def initial_books := 300
def borrowed_books := 140
def remaining_books := 210
def total_people := 10
def total_books := initial_books + borrowed_books - total_books_borrowed 

-- Define the statement we want to prove
theorem book_donations (d : Nat) :
  initial_books + d - borrowed_books = remaining_books →
  d = 50 ∧ d / total_people = 5 :=
begin
  sorry
end

end book_donations_l289_289049


namespace funcD_minimum_value_l289_289882

open Real

-- Conditions as definitions
def funcA (x : ℝ) : ℝ := x + 4 / x
def funcB (x : ℝ) : ℝ := sin x + 4 / sin x  -- \( 0 < x < \pi \)
def funcC (x : ℝ) : ℝ := 4 * log (3) x + log x 3
def funcD (x : ℝ) : ℝ := 4 * exp x + exp (-x)

-- Proof problem statement
theorem funcD_minimum_value : ∃ x : ℝ, ∀ y : ℝ, (y = 4 * exp x + exp (-x)) → (y ≥ 4) ∧ (y = 4 ↔ x = -ln 2) :=
by
  sorry

end funcD_minimum_value_l289_289882


namespace solve_z_sqrt_equation_l289_289170

theorem solve_z_sqrt_equation : 
  ∃ z : ℤ, sqrt (9 + 3 * (z : ℝ)) = 12 ∧ z = 45 :=
by 
  sorry

end solve_z_sqrt_equation_l289_289170


namespace problem_statement_l289_289363

noncomputable def maximal_value (x y z v w : ℝ) :=
  x * z + 3 * y * z + 4 * z * v + 6 * z * w

theorem problem_statement:
  ∀ (x y z v w : ℝ),
    0 < x → 0 < y → 0 < z → 0 < v → 0 < w →
    x^2 + y^2 + z^2 + v^2 + w^2 = 2023 →
    let M := maximal_value x y z v w in
    M + x + y + z + v + w = 59 + 63 * 1011.5 :=
sorry

end problem_statement_l289_289363


namespace necessary_nor_sufficient_for_condition_l289_289352

noncomputable def vectors_assumptions
  (a b : V) [normedGroup V] [normedSpace ℝ V] : Prop :=
  ∥a∥ = ∥b∥ ∧ ∥a + b∥ = ∥a - b∥

theorem necessary_nor_sufficient_for_condition 
  (a b : V) [normedGroup V] [normedSpace ℝ V] :
  vectors_assumptions a b ↔ (∥a∥ = ∥b∥ ∧ ∥a + b∥ ≠ ∥a - b∥) ∨ 
                       (∥a∥ ≠ ∥b∥ ∧ ∥a + b∥ = ∥a - b∥) ∨ 
                       (∥a∥ ≠ ∥b∥ ∧ ∥a + b∥ ≠ ∥a - b∥) :=
sorry

end necessary_nor_sufficient_for_condition_l289_289352


namespace correct_statements_l289_289356

variables {l m n : Type} {a : Type}
variables [Line l] [Line m] [Line n] [Plane a]

-- Define a function to state that a line is perpendicular to a plane.
def perp (l : Line) (a : Plane) : Prop := sorry
def parallel (l m : Line) : Prop := sorry
def subset_line_plane (l : Line) (a : Plane) : Prop := sorry
def intersect (l : Line) (a : Plane) : Prop := sorry

theorem correct_statements
  (h1 : perp l a → intersect l a)
  (h_m_in_a : subset_line_plane m a)
  (h_n_in_a : subset_line_plane n a)
  (h2 : ¬ (subset_line_plane m a ∧ subset_line_plane n a ∧ perp l m ∧ perp l n → perp l a))
  (h3 : parallel l m ∧ parallel m n ∧ perp l a → perp n a)
  (h4 : parallel l m ∧ perp m a ∧ perp n a → parallel l n) :
  True := sorry

end correct_statements_l289_289356


namespace average_remaining_two_l289_289817

theorem average_remaining_two (a b c d e : ℝ) 
  (h1 : (a + b + c + d + e) / 5 = 12) 
  (h2 : (a + b + c) / 3 = 4) : 
  (d + e) / 2 = 24 :=
by 
  sorry

end average_remaining_two_l289_289817


namespace S_2023_value_l289_289974

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom sum_of_first_n_terms (n : ℕ) : S n = ∑ i in Finset.range (n + 1), a i
axiom recurrence_relation (n : ℕ) : S n + 1 = a (n + 1)
axiom first_term : a 0 = 2

-- Theorem statement
theorem S_2023_value : S 2023 = 3 * 2^2022 - 1 :=
by sorry

end S_2023_value_l289_289974


namespace geraldo_drank_7_pints_l289_289054

-- Conditions
def total_gallons : ℝ := 20
def num_containers : ℕ := 80
def gallons_to_pints : ℝ := 8
def containers_drank : ℝ := 3.5

-- Problem statement
theorem geraldo_drank_7_pints :
  let total_pints : ℝ := total_gallons * gallons_to_pints
  let pints_per_container : ℝ := total_pints / num_containers
  let pints_drank : ℝ := containers_drank * pints_per_container
  pints_drank = 7 :=
by
  sorry

end geraldo_drank_7_pints_l289_289054


namespace school_should_purchase_bookshelves_l289_289401

theorem school_should_purchase_bookshelves
  (x : ℕ)
  (h₁ : x ≥ 20)
  (cost_A : ℕ := 20 * 300 + 100 * (x - 20))
  (cost_B : ℕ := (20 * 300 + 100 * x) * 80 / 100)
  (h₂ : cost_A = cost_B) : x = 40 :=
by sorry

end school_should_purchase_bookshelves_l289_289401


namespace total_songs_is_nine_l289_289932

variables (a t e : ℕ)

-- Definition of the conditions
def is_in_range (n : ℕ) := n ≥ 6 ∧ n ≤ 9

-- The main theorem statement
theorem total_songs_is_nine :
  (is_in_range a) → (is_in_range t) → (is_in_range e) →
  (10 + 5 + a + t + e) % 4 = 0 →
  (10 + 5 + a + t + e) / 4 = 9 :=
by {
  intro h_a h_t h_e h_div,
  sorry
}

end total_songs_is_nine_l289_289932


namespace jackson_hermit_crabs_l289_289331

theorem jackson_hermit_crabs (H : ℕ) (total_souvenirs : ℕ) 
  (h1 : total_souvenirs = H + 3 * H + 6 * H) 
  (h2 : total_souvenirs = 450) : H = 45 :=
by {
  sorry
}

end jackson_hermit_crabs_l289_289331


namespace inverse_cube_root_variation_l289_289031

theorem inverse_cube_root_variation (k : ℝ) (h₁ : ∀ x y : ℝ, y * x^(1/3) = k)
    (h₂ : h₁ 8 2) :
    ∀ y : ℝ, y = 8 → ∃ x : ℝ, x = 1 / 8 := by
sorry

end inverse_cube_root_variation_l289_289031


namespace smallest_positive_integer_un_l289_289672

theorem smallest_positive_integer_un (n : ℕ) (h_pos : 0 < n) : 
  ∃ (u_n : ℕ), u_n = 2 * n - 1 ∧ 
  ∀ (d : ℕ), 0 < d → 
  ∀ (m : ℕ), 0 < m → odd m → 
  let N_m_uk := (list.range (2 * (u_n))) |>.filter odd |> list.map (λ i, m + 2 * i) in
  let N_1_n_d := (list.range (2 * n) |>.filter odd).filter (λ x, d ∣ x) in
  (N_m_uk.filter (λ x, d ∣ x)).length ≥ N_1_n_d.length :=
begin
  use (2 * n - 1),
  split,
  { refl },
  intros d hd_pos m hm_pos h_odd_m,
  let N_m_uk := (list.range (2 * (2 * n - 1))) |>.filter odd |> list.map (λ i, m + 2 * i),
  let N_1_n_d := (list.range (2 * n) |>.filter odd).filter (λ x, d ∣ x),
  suffices : (N_m_uk.filter (λ x, d ∣ x)).length ≥ N_1_n_d.length,
  from this,
  sorry
end

end smallest_positive_integer_un_l289_289672


namespace must_be_moment_with_10_roses_watering_never_moment_with_11_roses_watering_l289_289752

noncomputable section

-- Define the conditions
def num_roses : ℕ := 100
def num_sectors : ℕ := 11
def central_angle_per_sector : ℝ := 2 * Real.pi / num_sectors
variable (rotate_uniformly : Prop)

-- Definitions provided in the problem
def calc_roses_in_each_sector (roses : ℕ) : Prop :=
  let min_roses_per_sector := roses / num_sectors
  min_roses_per_sector * num_sectors = roses

-- Part 1: Prove that there must be a moment when exactly 10 roses are watered simultaneously
theorem must_be_moment_with_10_roses_watering 
  (h_rotate : rotate_uniformly)
  (h_roses : num_roses = 100)
  (h_sectors : num_sectors = 11)
  (h_roses_distr : calc_roses_in_each_sector num_roses) :
  ∃ t : ℝ, watering_roses_at_time t = 10 := by sorry

-- Part 2: Determine whether there must be a moment when exactly 11 roses are watered simultaneously
theorem never_moment_with_11_roses_watering
  (h_rotate : rotate_uniformly)
  (h_roses : num_roses = 100)
  (h_sectors : num_sectors = 11)
  (h_roses_distr : calc_roses_in_each_sector num_roses) :
  ¬ ∃ t : ℝ, watering_roses_at_time t = 11 := by sorry

end must_be_moment_with_10_roses_watering_never_moment_with_11_roses_watering_l289_289752


namespace sedans_in_anytown_l289_289100

def trucks_sedans_motorcycles_ratio (T S M : ℕ) : Prop :=
  T * 7 = S * 3 ∧ S * 2 = M * 7

theorem sedans_in_anytown (S : ℕ) (h : trucks_sedans_motorcycles_ratio 3 S 2600) : S = 9100 :=
by
  have h2 : S * 2 = 2600 * 7 := h.right
  have h3 : S * 2 = 18200 := by rw [h2]
  have h4 : S = 18200 / 2 := by norm_num at h3; exact eq.div_eq_of_eq_mul_right _ h3
  norm_num at h4
  exact h4

end sedans_in_anytown_l289_289100


namespace angle_between_vectors_l289_289563

noncomputable def vector1 : EuclideanSpace ℝ (Fin 2) := ![3, 4]
noncomputable def vector2 : EuclideanSpace ℝ (Fin 2) := ![4, -3]

theorem angle_between_vectors :
  let θ := real.arccos ((vector1 ⬝ vector2) / (∥vector1∥ * ∥vector2∥))
  θ = 90 :=
by sorry

end angle_between_vectors_l289_289563


namespace linear_coefficient_l289_289411

def polynomial := (x : ℝ) -> x^2 - 2*x - 3

theorem linear_coefficient (x : ℝ) : (polynomial x) = x^2 - 2*x - 3 → -2 :=
by
  sorry

end linear_coefficient_l289_289411


namespace eric_boxes_l289_289554

def numberOfBoxes (totalPencils : Nat) (pencilsPerBox : Nat) : Nat :=
  totalPencils / pencilsPerBox

theorem eric_boxes :
  numberOfBoxes 27 9 = 3 := by
  sorry

end eric_boxes_l289_289554


namespace placing_pencils_l289_289924

theorem placing_pencils (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
    (h1 : total_pencils = 6) (h2 : num_rows = 2) : pencils_per_row = 3 :=
by
  sorry

end placing_pencils_l289_289924


namespace general_inequality_harmonic_sum_l289_289981

theorem general_inequality_harmonic_sum (n : ℕ) (hn : n > 0) :
  (∑ i in finset.range (2^n), 1 / (i+1)) > (n / 2 : ℝ) :=
by
  sorry

end general_inequality_harmonic_sum_l289_289981


namespace final_selling_price_l289_289878

def actual_price : ℝ := 9941.52
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

noncomputable def final_price (P : ℝ) : ℝ :=
  P * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem final_selling_price :
  final_price actual_price = 6800.00 :=
by
  sorry

end final_selling_price_l289_289878


namespace trigonometric_identity_proof_l289_289541

theorem trigonometric_identity_proof :
  (1 / real.cos (80 * real.pi / 180)) - (real.sqrt 3 / real.sin (80 * real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_identity_proof_l289_289541


namespace compare_slopes_l289_289571

noncomputable def f (p q r x : ℝ) := x^3 + p * x^2 + q * x + r

noncomputable def s (p q x : ℝ) := 3 * x^2 + 2 * p * x + q

theorem compare_slopes (p q r a b c : ℝ) (hb : b ≠ 0) (ha : a ≠ c) 
  (hfa : f p q r a = 0) (hfc : f p q r c = 0) : a > c → s p q a > s p q c := 
by
  sorry

end compare_slopes_l289_289571


namespace find_xyz_area_proof_l289_289151

-- Conditions given in the problem
variable (x y z : ℝ)
-- Side lengths derived from condition of inscribed circle
def conditions :=
  (x + y = 5) ∧
  (x + z = 6) ∧
  (y + z = 8)

-- The proof problem: Show the relationships between x, y, and z given the side lengths
theorem find_xyz_area_proof (h : conditions x y z) :
  (z - y = 1) ∧ (z - x = 3) ∧ (z = 4.5) ∧ (x = 1.5) ∧ (y = 3.5) :=
by
  sorry

end find_xyz_area_proof_l289_289151


namespace sum_divisors_eq_product_divisors_eq_l289_289101

def sum_of_divisors (a : ℕ) : ℕ :=
  ∑ d in (finset.range (a + 1)).filter (λ x => x ∣ a), d

def product_of_divisors (a : ℕ) : ℕ :=
  ∏ d in (finset.range (a + 1)).filter (λ x => x ∣ a), d

theorem sum_divisors_eq (a : ℕ) (h : (finset.range (a + 1)).filter (λ x => x ∣ a) = 107) :
  ∃ p : ℕ, (∀ n : ℕ, a = p ^ 106 → sum_of_divisors a = (p ^ 107 - 1) / (p - 1)) :=
sorry

theorem product_divisors_eq (a : ℕ) (h : (finset.range (a + 1)).filter (λ x => x ∣ a) = 107) :
  ∃ p : ℕ, (∀ n : ℕ, a = p ^ 106 → product_of_divisors a = p ^ 11321) :=
sorry

end sum_divisors_eq_product_divisors_eq_l289_289101


namespace geom_seq_a4_a5_a6_value_l289_289261

theorem geom_seq_a4_a5_a6_value (a : ℕ → ℝ) (h_geom : ∃ r, 0 < r ∧ ∀ n, a (n + 1) = r * a n)
  (h_roots : ∃ x y, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 9 = y) :
  a 4 * a 5 * a 6 = 64 :=
by
  sorry

end geom_seq_a4_a5_a6_value_l289_289261


namespace freshmen_and_sophomores_without_pet_l289_289035

theorem freshmen_and_sophomores_without_pet (total_students : ℕ) 
                                             (freshmen_sophomores_percent : ℕ)
                                             (pet_ownership_fraction : ℕ)
                                             (h_total : total_students = 400)
                                             (h_percent : freshmen_sophomores_percent = 50)
                                             (h_fraction : pet_ownership_fraction = 5) : 
                                             (total_students * freshmen_sophomores_percent / 100 - 
                                             total_students * freshmen_sophomores_percent / 100 / pet_ownership_fraction) = 160 :=
by
  sorry

end freshmen_and_sophomores_without_pet_l289_289035


namespace range_of_x_satisfying_inequality_l289_289215

noncomputable def f : ℝ → ℝ := sorry -- f is some even and monotonically increasing function

theorem range_of_x_satisfying_inequality :
  (∀ x, f (-x) = f x) ∧ (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) → {x : ℝ | f x < f 1} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  intro h
  sorry

end range_of_x_satisfying_inequality_l289_289215


namespace tommy_expected_value_score_l289_289455

/-- Tommy's 25-question true-false test setup -/
def tommy_test : {n : ℕ} → distr (vector bool n) :=
λ n, distr.uniform (vector bool n) 

/-- Streak points calculation -/
def streak_points (qs : vector bool 25) : ℕ :=
qs.to_list.foldl (λ ⟨streak, total⟩ q, 
  if q then (streak + 1, total + (streak + 1)) 
  else (0, total)) (0, 0)).2

/-- Expected value of Tommy's score -/
def expected_value_of_tommy_score : ℕ :=
25 * 2

/-- Prove that the expected value of Tommy’s score is 50 -/
theorem tommy_expected_value_score:
  ∑ (q : vector bool 25), (tommy_test 25).pdf q * (streak_points q) = expected_value_of_tommy_score :=
sorry

end tommy_expected_value_score_l289_289455


namespace train_speed_proof_l289_289123

-- Define the constants used in the problem
def train_length : ℝ := 250
def time_to_cross_pole : ℝ := 5.5551111466638226

-- Define the function to calculate speed in m/s
def speed_m_per_s (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Define the conversion factor from m/s to km/hr
def m_per_s_to_km_per_hr (speed : ℝ) : ℝ := speed * 3.6

-- State the expected speed of the train in km/hr
def expected_speed_km_per_hr : ℝ := 161.9784

-- The main theorem to be proved
theorem train_speed_proof :
  m_per_s_to_km_per_hr (speed_m_per_s train_length time_to_cross_pole) = expected_speed_km_per_hr :=
by
  sorry

end train_speed_proof_l289_289123


namespace tangent_line_f1_add_fprime1_l289_289613

variable {ℝ : Type} [LinearOrderedField ℝ]

theorem tangent_line_f1_add_fprime1 :
  (∃ (f : ℝ → ℝ), ∀ (x : ℝ), has_tangent_at f 1 (λ x, (1 / 2) * x + 2) (1, f 1)) →
  (∃ f'(1) = (1 / 2 : ℝ), f(1) + f'(1) = 3) := 
by
  sorry

end tangent_line_f1_add_fprime1_l289_289613


namespace triangle_range_k_l289_289251

theorem triangle_range_k (α : ℝ) (a b c : ℝ) 
  (h1 : α = 60) 
  (h2 : a = 12) 
  (h3 : b = k) 
  (h4 : k > 0) 
  (h5 : k ≤ 12) 
  (h6 : k = 8*sqrt 3 ∨ (0 < k ∧ k ≤ 12)) 
  : (0 < k ∧ k ≤ 12) ∨ (k = 8*sqrt 3) :=
sorry

end triangle_range_k_l289_289251


namespace sin_double_angle_l289_289968

theorem sin_double_angle (α : ℝ) (h : sin α - cos α = 4 / 3) :
  sin (2 * α) = -7 / 9 :=
by
  sorry

end sin_double_angle_l289_289968


namespace find_OQ_l289_289935
-- Import the required math libarary

-- Define points on a line with the given distances
def O := 0
def A (a : ℝ) := 2 * a
def B (b : ℝ) := 4 * b
def C (c : ℝ) := 5 * c
def D (d : ℝ) := 7 * d

-- Given P between B and C such that ratio condition holds
def P (a b c d x : ℝ) := 
  B b ≤ x ∧ x ≤ C c ∧ 
  (A a - x) * (x - C c) = (B b - x) * (x - D d)

-- Calculate Q based on given ratio condition
def Q (b c d y : ℝ) := 
  C c ≤ y ∧ y ≤ D d ∧ 
  (C c - y) * (y - D d) = (B b - C c) * (C c - D d)

-- Main Proof Statement to prove OQ
theorem find_OQ (a b c d y : ℝ) 
  (hP : ∃ x, P a b c d x)
  (hQ : ∃ y, Q b c d y) :
  y = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) := by
  sorry

end find_OQ_l289_289935


namespace arithmetic_sequence_sum_l289_289962

theorem arithmetic_sequence_sum (a b c : ℤ) (d : ℤ) (n : ℕ) 
  (h_arith : a + (a + d) + (a + 2 * d) = -3)
  (h_prod : a * (a + d) * (a + 2 * d) = 8)
  (h_geom : (a + 2 * d)^2 = a * (a + d)) :
  ∑ i in finset.range n, (3 * i + 7) = (3 * n * (n + 1)) / 2 :=
by
  sorry

end arithmetic_sequence_sum_l289_289962


namespace find_ordered_pair_l289_289567

noncomputable def a : ℝ := 9
noncomputable def b : ℝ := -4

def cos_60 : ℝ := 1 / 2
def sec_60 : ℝ := 2

theorem find_ordered_pair :
  sqrt (25 - 16 * cos_60) = a - b * sec_60 :=
by
  sorry

end find_ordered_pair_l289_289567


namespace integer_sum_l289_289006

theorem integer_sum {p q r s : ℤ} 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 22 := 
sorry

end integer_sum_l289_289006


namespace admission_price_for_adults_l289_289522

-- Constants and assumptions
def children_ticket_price : ℕ := 25
def total_persons : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80

-- Definitions based on the conditions
def adults_attended : ℕ := total_persons - children_attended
def total_amount_from_children : ℕ := children_attended * children_ticket_price
def total_amount_from_adults (A : ℕ) : ℕ := total_collected_cents - total_amount_from_children
def adult_ticket_price := (total_collected_cents - total_amount_from_children) / adults_attended

-- Theorem statement to be proved
theorem admission_price_for_adults : adult_ticket_price = 60 := by
  sorry

end admission_price_for_adults_l289_289522


namespace volleyball_team_selection_count_l289_289384

theorem volleyball_team_selection_count :
  let total_players := 17
  let quadruplets := 4
  let starters := 6
  let chosen_quadruplets := 2
  let remaining_players := total_players - quadruplets
  let needed_from_remaining := starters - chosen_quadruplets
  (nat.choose quadruplets chosen_quadruplets) * (nat.choose remaining_players needed_from_remaining) = 4290 := 
by
  sorry

end volleyball_team_selection_count_l289_289384


namespace ln_of_gt_of_pos_l289_289237

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end ln_of_gt_of_pos_l289_289237


namespace log_base_243_l289_289531

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_base_243 : log_base 3 243 = 5 := by
  -- this is the statement, proof is omitted
  sorry

end log_base_243_l289_289531


namespace parallelogram_equality_l289_289314

open_locale classical

variables {A B C D F H G : Type*} [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ D] [normed_space ℝ F] [normed_space ℝ H] [normed_space ℝ G]

structure parallelogram (A B C D F H G : Type*) :=
(AB : ℝ)
(AF : ℝ)
(AD : ℝ)
(AH : ℝ)
(AC : ℝ)
(AG : ℝ)
(parallelogram_prop : True) -- Placeholder for the definition that ABCD is a parallelogram
(circle_passes_through_A : True) -- Placeholder for the circle passing through A and intersecting extended lines

theorem parallelogram_equality (p : parallelogram A B C D F H G) :
  p.AB * p.AF + p.AD * p.AH = p.AC * p.AG :=
sorry

end parallelogram_equality_l289_289314


namespace apple_pie_theorem_l289_289668

theorem apple_pie_theorem (total_apples : ℕ) (not_ripe_apples : ℕ) (apples_per_pie : ℕ) (total_ripe_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 34)
  (h2 : not_ripe_apples = 6)
  (h3 : apples_per_pie = 4)
  (h4 : total_ripe_apples = total_apples - not_ripe_apples)
  (h5 : number_of_pies = total_ripe_apples / apples_per_pie) :
  number_of_pies = 7 :=
  by
  have h6 : total_apples - not_ripe_apples = 28 := by rw [h1, h2]; norm_num
  have h7 : total_ripe_apples = 28 := by rw [h4, h6]
  have h8 : 28 / apples_per_pie = 7 := by rw [h3]; norm_num
  rw [h7, h5, h8]
  sorry

end apple_pie_theorem_l289_289668


namespace tyson_three_pointers_l289_289788

theorem tyson_three_pointers (x : ℕ) 
  (h1 : (Tyson scored two points twelve times)) 
  (h2 : (Tyson scored one point six times)) 
  (h3 : (In total, Tyson scored 75 points)) 
  (h4 : (scored three points some number of times)) : 
  3 * x + 24 + 6 = 75 → x = 15 := 
by
  sorry

end tyson_three_pointers_l289_289788


namespace infinite_product_value_l289_289550

-- We define the infinite product
def infinite_product : ℝ :=
  ∏' n : ℕ, (3^((n+1) / 2^n))

-- The main theorem stating the value of the product
theorem infinite_product_value : infinite_product = 9 :=
  sorry

end infinite_product_value_l289_289550


namespace orthocenters_rectangle_l289_289348

open EuclideanGeometry

theorem orthocenters_rectangle 
    (Γ Γ': Circle) 
    (A A' B C D E : Point) 
    (hA : A ∈ Γ) 
    (hA' : A' ∈ Γ)
    (hA_Γ' : A ∈ Γ') 
    (hA'_Γ' : A' ∈ Γ') 
    (hT1B : B ∈ tangent_point(Γ, Γ'))
    (hT1C : C ∈ tangent_point(Γ, Γ'))
    (hT2D : D ∈ tangent_point(Γ, Γ'))
    (hT2E : E ∈ tangent_point(Γ, Γ')) 
    (hBC : is_common_tangent(B, C, Γ, Γ')) 
    (hDE : is_common_tangent(D, E, Γ, Γ')) : 
    is_rectangle (orthocenter(Δ A B C)) (orthocenter(Δ A' B C)) (orthocenter(Δ A D E)) (orthocenter(Δ A' D E)) := 
sorry

end orthocenters_rectangle_l289_289348


namespace probability_of_sum_16_with_duplicates_l289_289046

namespace DiceProbability

def is_valid_die_roll (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 6

def is_valid_combination (x y z : ℕ) : Prop :=
  x + y + z = 16 ∧ 
  is_valid_die_roll x ∧ 
  is_valid_die_roll y ∧ 
  is_valid_die_roll z ∧ 
  (x = y ∨ y = z ∨ z = x)

theorem probability_of_sum_16_with_duplicates (P : ℚ) :
  (∃ x y z : ℕ, is_valid_combination x y z) → 
  P = 1 / 36 :=
sorry

end DiceProbability

end probability_of_sum_16_with_duplicates_l289_289046


namespace relationship_m_n_l289_289445

theorem relationship_m_n (m n : ℕ) (h : 10 / (m + 10 + n) = (m + n) / (m + 10 + n)) : m + n = 10 := 
by sorry

end relationship_m_n_l289_289445


namespace optimize_transport_fleet_l289_289138
-- Lean 4 statement for the equivalent proof problem


axiom normal_distribution_X (X : ℝ) : ProbabilityDistribution ℝ := normal 800 50

-- Define p_0
def p_0 : ℝ := 0.9772

-- Vehicle properties
def capacity_A : ℕ := 36
def capacity_B : ℕ := 60
def cost_A : ℕ := 1600
def cost_B : ℕ := 2400
def max_vehicles : ℕ := 21

-- Constraints
def num_vehicles_A : ℕ := 5
def num_vehicles_B : ℕ := 12

theorem optimize_transport_fleet :
  (∀ X, X ∼ normal_distribution_X →
    (∫ x, (x <= 900) ∂normal_distribution_X = p_0)) ∧ 
  (num_vehicles_A + num_vehicles_B ≤ max_vehicles) ∧
  (num_vehicles_B ≤ num_vehicles_A + 7) ∧
  (num_vehicles_A * capacity_A + num_vehicles_B * capacity_B ≥ (normal_distribution_X.mean)) :=
by
  sorry

end optimize_transport_fleet_l289_289138


namespace max_gcd_lcm_condition_l289_289717

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l289_289717


namespace ball_box_distribution_l289_289630

theorem ball_box_distribution :
  ∃ (distinct_ways : ℕ), distinct_ways = 7 :=
by
  let num_balls := 5
  let num_boxes := 4
  sorry

end ball_box_distribution_l289_289630


namespace min_photos_l289_289304

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289304


namespace convexNGonDissectedIntoEqualTriangles_l289_289495

-- Defining a convex n-gon with properties and conditions
noncomputable def isConvexNGon (P : Type) [Polygon P] (n : ℕ) : Prop :=
  n > 3 ∧ isConvex P ∧ isNGon P n ∧ isCircumscribed P

-- Defining the dissection into equal triangles with non-intersecting diagonals
noncomputable def dissectedIntoEqualTriangles (P : Type) [Polygon P] : Prop :=
  ∃ (triangles : Set (Triangle P)), 
    EquallySizedTriangles triangles ∧
    NonIntersectingDiagonalsInside P triangles

-- The final theorem
theorem convexNGonDissectedIntoEqualTriangles 
  (P : Type) [Polygon P] (n : ℕ) :
  isConvexNGon P n → dissectedIntoEqualTriangles P → n = 4 :=
by
  intros h1 h2
  sorry

end convexNGonDissectedIntoEqualTriangles_l289_289495


namespace unique_positive_x_values_l289_289182

-- Definitions from conditions
def Inequality1 (x : ℕ) : Prop := 3 * x > 4 * x - 4
def Inequality2 (x : ℕ) (b : ℕ) : Prop := 2 * x - b > -3

-- Statement to prove
theorem unique_positive_x_values (x : ℕ) : 
  (3 * x > 4 * x - 4) → (2 * x - 7 > -3 ∧ 2 * x - 8 > -3) → x = 3 :=
sorry

end unique_positive_x_values_l289_289182


namespace function_passes_through_one_one_l289_289572

noncomputable def f (a x : ℝ) : ℝ := a^(x - 1)

theorem function_passes_through_one_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 := 
by
  sorry

end function_passes_through_one_one_l289_289572


namespace tangent_ellipse_hyperbola_l289_289756

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ↔ x^2 - n * (y - 1)^2 = 4) →
  n = 9 / 5 :=
by sorry

end tangent_ellipse_hyperbola_l289_289756


namespace num_solutions_ffx_eq_2_l289_289364

def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 4 else 3*x - 6

theorem num_solutions_ffx_eq_2 : 
  let fx_fx_eq_2 := (λ x, f (f x) = 2)
  ({ x : ℝ | fx_fx_eq_2 x }).size = 2 :=
sorry

end num_solutions_ffx_eq_2_l289_289364


namespace horses_for_camels_l289_289493

noncomputable def cost_of_one_elephant : ℕ := 11000
noncomputable def cost_of_one_ox : ℕ := 7333 -- approx.
noncomputable def cost_of_one_horse : ℕ := 1833 -- approx.
noncomputable def cost_of_one_camel : ℕ := 4400

theorem horses_for_camels (H : ℕ) :
  (H * cost_of_one_horse = cost_of_one_camel) → H = 2 :=
by
  -- skipping proof details
  sorry

end horses_for_camels_l289_289493


namespace N_3p_minus_2q_l289_289349

variable (N : Matrix (Fin 2) (Fin 2) ℝ) 
variable (p q : Vector (Fin 2) ℝ) 

-- Conditions
axiom hN_p : N.mulVec p = ![2, -3]
axiom hN_q : N.mulVec q = ![-4, 6]

-- Problem Statement
theorem N_3p_minus_2q : N.mulVec (3 • p - 2 • q) = ![14, -21] := by
  sorry

end N_3p_minus_2q_l289_289349


namespace addition_and_rounding_l289_289877

theorem addition_and_rounding :
  let sum := 92.345 + 47.129 in
  Int.round sum = 139 :=
by
  -- Assuming Lean definition of Int.round according to the described behavior.
  sorry

end addition_and_rounding_l289_289877


namespace min_value_expression_l289_289637

theorem min_value_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ (a b : ℝ), ((a, b) = (x, y)) → a^2 + b^2 ≤ 1 → abs(2*a + b - 2) + abs(6 - a - 3*b) ≥ m := 
sorry

end min_value_expression_l289_289637


namespace minimum_photos_taken_l289_289308

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289308


namespace dice_probability_l289_289552

noncomputable def dieA_faces := {1, 2, 3, 3, 4, 4}
noncomputable def dieB_faces := {1, 2, 5, 6, 7, 8}

def prob_sum_consecutive (sum_val : ℕ) (diceA : set ℕ) (diceB : set ℕ) : Prop :=
  ∃ a ∈ diceA, ∃ b ∈ diceB, (a + b = sum_val) ∧ (a = b + 1 ∨ b = a + 1)
  
theorem dice_probability :
  (finset.filter (λ x, prob_sum_consecutive x dieA_faces dieB_faces) {6, 8, 10}.to_finset).card / 36 = 5 / 36 :=
sorry

end dice_probability_l289_289552


namespace range_of_omega_l289_289432

theorem range_of_omega (ω : ℝ) (hω : ω > 0) (h_range : ∀ x, 0 ≤ x ∧ x ≤ π → -1 ≤ cos (ω * x + π / 4) ∧ cos (ω * x + π / 4) ≤ sqrt 2 / 2) :
  3 / 4 ≤ ω ∧ ω ≤ 3 / 2 := 
sorry

end range_of_omega_l289_289432


namespace ratio_HC_JE_l289_289387

noncomputable def A : ℝ := 0
noncomputable def B : ℝ := 1
noncomputable def C : ℝ := B + 2
noncomputable def D : ℝ := C + 1
noncomputable def E : ℝ := D + 1
noncomputable def F : ℝ := E + 2

variable (G H J K : ℝ × ℝ)
variable (parallel_AG_HC parallel_AG_JE parallel_AG_KB : Prop)

-- Conditions
axiom points_on_line : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F
axiom AB : B - A = 1
axiom BC : C - B = 2
axiom CD : D - C = 1
axiom DE : E - D = 1
axiom EF : F - E = 2
axiom G_off_AF : G.2 ≠ 0
axiom H_on_GD : H.1 = G.1 ∧ H.2 = D
axiom J_on_GF : J.1 = G.1 ∧ J.2 = F
axiom K_on_GB : K.1 = G.1 ∧ K.2 = B
axiom parallel_hc_je_kb_ag : parallel_AG_HC ∧ parallel_AG_JE ∧ parallel_AG_KB ∧ (G.2 / 1) = (K.2 / (K.1 - G.1))

-- Task: Prove the ratio HC/JE = 7/8
theorem ratio_HC_JE : (H.2 - C) / (J.2 - E) = 7 / 8 :=
sorry

end ratio_HC_JE_l289_289387


namespace minimum_photos_l289_289292

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289292


namespace decimal_0_0_1_7_eq_rational_l289_289134

noncomputable def infinite_loop_decimal_to_rational_series (a : ℚ) (r : ℚ) : ℚ :=
  a / (1 - r)

theorem decimal_0_0_1_7_eq_rational :
  infinite_loop_decimal_to_rational_series (17 / 1000) (1 / 100) = 17 / 990 :=
by
  sorry

end decimal_0_0_1_7_eq_rational_l289_289134


namespace range_of_a_l289_289758

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, sin x ^ 2 + cos x + a = 0) ↔ (-5 / 4 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l289_289758


namespace product_between_21st_and_24th_multiple_of_3_l289_289423

theorem product_between_21st_and_24th_multiple_of_3 : 
  (66 * 69 = 4554) :=
by
  sorry

end product_between_21st_and_24th_multiple_of_3_l289_289423


namespace intersection_point_of_line_l289_289568

noncomputable def line_intersection_with_xy_plane : Prop :=
  let (x1, y1, z1) := (2, 7, 3)
  let (x2, y2, z2) := (6, 3, 8)
  let direction := (x2 - x1, y2 - y1, z2 - z1)
  let parameter_line := fun t ↦ (x1 + direction.1 * t, 
                                 y1 + direction.2 * t, 
                                 z1 + direction.3 * t)
  exists t, parameter_line t = (-2 / 5, 47 / 5, 0)

theorem intersection_point_of_line : line_intersection_with_xy_plane :=
  sorry

end intersection_point_of_line_l289_289568


namespace number_of_nonempty_subsets_of_M_l289_289593

-- Define the imaginary unit i as a constant
noncomputable def i : ℂ := complex.I

-- Define the set M based on the given condition
noncomputable def M : set ℂ := {z | ∃ n : ℕ, n > 0 ∧ z = ( (i - 1) / (i + 1) ) ^ n}

-- Define a theorem to prove that the number of non-empty subsets of M is 15
theorem number_of_nonempty_subsets_of_M : 
  ∃ n : ℤ, n = 2 ^ (finite.to_finset M).card - 1 ∧ n = 15 := 
sorry

end number_of_nonempty_subsets_of_M_l289_289593


namespace min_photographs_42_tourists_3_monuments_l289_289274

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289274


namespace lemma_figure_m_area_l289_289347

noncomputable def fig_m_area (x y : ℝ) : ℝ :=
  let M := {p : ℝ × ℝ | ∃ a b : ℝ,
    (p.1 - a)^2 + (p.2 - b)^2 ≤ 5 ∧
    a^2 + b^2 ≤ min (4 * a - 2 * b) 5} in
  15 * Real.pi - 5 * Real.sqrt 3 / 2

theorem lemma_figure_m_area (x y : ℝ) : fig_m_area x y = 15 * Real.pi - 5 * Real.sqrt 3 / 2 :=
sorry

end lemma_figure_m_area_l289_289347


namespace method_I_l289_289230

theorem method_I (t : ℕ) (ht : t ≥ 2) : 
  let a₁ := 2,
      a₂ := 2 * t^2,
      a₃ := 2 * (t^2 + 1) in
  (a₁ ≠ a₂ ∧ a₂ ≠ a₃ ∧ a₃ ≠ a₁) ∧
  (¬∃ k : ℕ, k^2 = a₁) ∧
  (¬∃ k : ℕ, k^2 = a₂) ∧
  (¬∃ k : ℕ, k^2 = a₃) ∧
  (∃ A : ℚ, A = 1/2 * √a₁ * √a₂) :=
sorry

end method_I_l289_289230


namespace computation_problems_count_l289_289120

theorem computation_problems_count
    (C W : ℕ)
    (h1 : 3 * C + 5 * W = 110)
    (h2 : C + W = 30) :
    C = 20 :=
by
  sorry

end computation_problems_count_l289_289120


namespace find_value_of_expression_l289_289958

theorem find_value_of_expression (m : ℝ) (h_m : m^2 - 3 * m + 1 = 0) : 2 * m^2 - 6 * m - 2024 = -2026 := by
  sorry

end find_value_of_expression_l289_289958


namespace vampire_daily_needed_people_l289_289874

-- Define the conditions as constants
def gallons_needed_per_week : ℕ := 7
def pints_per_person : ℕ := 2
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7

-- Define the proof statement
theorem vampire_daily_needed_people (gallons_needed_per_week = 7) 
                                    (pints_per_person = 2) 
                                    (pints_per_gallon = 8) 
                                    (days_per_week = 7) : 
                                    7 * 8 / 7 / 2 = 4 :=
by
    -- The proof is expected here
    sorry

end vampire_daily_needed_people_l289_289874


namespace Bob_can_drive_100_km_l289_289529

theorem Bob_can_drive_100_km (k_per_gallon : ℕ) (gallons : ℕ) (h1 : k_per_gallon = 10) (h2 : gallons = 10) : k_per_gallon * gallons = 100 :=
by
  rw [h1, h2]
  exact Nat.mul_self 10

end Bob_can_drive_100_km_l289_289529


namespace train_speed_is_45_km_per_hr_l289_289868

noncomputable def train_speed_in_km_per_hr 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time * 3.6

theorem train_speed_is_45_km_per_hr 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ)
  (h1 : train_length = 170)
  (h2 : bridge_length = 205)
  (h3 : crossing_time = 30) : 
  train_speed_in_km_per_hr train_length bridge_length crossing_time = 45 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end train_speed_is_45_km_per_hr_l289_289868


namespace intersecting_chords_sin_minor_arc_product_l289_289010

theorem intersecting_chords_sin_minor_arc_product (r : ℝ) (BC AD m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) 
    (h_r : r = 5) (h_BC : BC = 6) (h_AD_bisected : ∃ E, E ∈ segment A D ∧ AD = 2 * dist A E ∧ E ∈ segment B C)
    (unique_AD_bisect : ∀ A' D', A' ≠ A ∧ D' ≠ D → ¬(∃ E', E' ∈ segment A' D' ∧ dist A' E' = dist E' D' ∧ E' ∈ segment B C))
    (h_sin_rational : ∃ k l : ℕ, sin_arc_fractions k l ∧ l = 25 ∧ m * n = 175): prod m n = 175 :=
by
  -- The actual proof isn't required; 'sorry' marks the place where a proof would go.
  sorry

end intersecting_chords_sin_minor_arc_product_l289_289010


namespace graph_comparison_at_y_intercept_l289_289901

theorem graph_comparison_at_y_intercept :
  let y₁ := (0:ℝ)^2 - 2 * (0:ℝ) + 5
  let y₂ := (0:ℝ)^2 + 2 * (0:ℝ) + 3
  y₁ > y₂ :=
by
  let y₁ := 0^2 - 2 * 0 + 5
  let y₂ := 0^2 + 2 * 0 + 3
  have h₁ : y₁ = 5 := by rfl
  have h₂ : y₂ = 3 := by rfl
  rw [h₁, h₂]
  exact by norm_num

end graph_comparison_at_y_intercept_l289_289901


namespace quadratic_repeated_root_l289_289602

open Real

theorem quadratic_repeated_root (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) 
    (h3 : (4 * cos θ) ^ 2 - 4 * (1) * (cos θ / sin θ) = 0) : 
    θ = π / 12 ∨ θ = 5 * π / 12 :=
by {
-- Proof steps would go here
sorry
}

end quadratic_repeated_root_l289_289602


namespace paint_left_after_two_coats_l289_289463

theorem paint_left_after_two_coats :
  let initial_paint := 3 -- liters
  let first_coat_paint := initial_paint / 2
  let paint_after_first_coat := initial_paint - first_coat_paint
  let second_coat_paint := (2 / 3) * paint_after_first_coat
  let paint_after_second_coat := paint_after_first_coat - second_coat_paint
  (paint_after_second_coat * 1000) = 500 := by
  sorry

end paint_left_after_two_coats_l289_289463


namespace sin_cos_expr_value_l289_289547

noncomputable def sin_cos_expr : ℝ :=
  sin (real.pi / 2) + 2 * cos 0 - 3 * sin (3 * real.pi / 2) + 10 * cos real.pi

theorem sin_cos_expr_value : sin_cos_expr = -4 := 
by
  -- Utilize known values for special angles:
  -- sin (pi / 2) = 1
  -- cos 0 = 1
  -- sin (3 * pi / 2) = -1
  -- cos pi = -1
  sorry

end sin_cos_expr_value_l289_289547


namespace find_l2_equation_l289_289970

noncomputable def curve (x : ℝ) : ℝ := x^2 + x - 2

def tangent_slope (x : ℝ) : ℝ := 2*x + 1

def is_tangent_line (l : ℝ → ℝ) (x : ℝ) : Prop := 
  ∃ b, l = λ x, (tangent_slope x) * (x - b) + curve b

def perpendicular_tangent_lines (l1 l2 : ℝ → ℝ) (p q : ℝ × ℝ) : Prop :=
  is_tangent_line l1 p.1 ∧ is_tangent_line l2 q.1 ∧ tangent_slope p.1 * tangent_slope q.1 = -1

theorem find_l2_equation :
  ∃ (l2 : ℝ → ℝ), (∃ (b : ℝ), l2 = λ x, (2*b + 1) * (x - b) + (curve b)) ∧ (3 * (2*(-2/3) + 1) = -1) ∧ (∀ x : ℝ, 3*x + 9*(l2 x) + 22 = 0) :=
sorry

end find_l2_equation_l289_289970


namespace swim_tuesday_l289_289367

def days_of_week := {Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday : Type}

variable (sports : days_of_week → Prop)
variable (run : days_of_week → Prop)
variable (basketball : days_of_week → Prop)
variable (golf : days_of_week → Prop)
variable (tennis : days_of_week → Prop)
variable (swim : days_of_week → Prop)

-- conditions
axiom sport_each_day (d : days_of_week) : ∃! s, sports s d
axiom basketball_wednesday : basketball Wednesday
axiom golf_saturday : golf Saturday
axiom runs_three_days : (∃ d1 d2 d3 : days_of_week, run d1 ∧ run d2 ∧ run d3 ∧ 
                        d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ 
                        ¬ consecutive_days d1 d2 ∧ ¬ consecutive_days d1 d3 ∧ ¬ consecutive_days d2 d3)
axiom no_tennis_after_run_swim (d1 d2 : days_of_week) : (run d1 ∨ swim d1) → consecutive_days d1 d2 → ¬ tennis d2

-- goal
theorem swim_tuesday : swim Tuesday := sorry

end swim_tuesday_l289_289367


namespace profit_percentage_l289_289845

/-- 
  Given:
  - Selling Price (SP) is $850
  - Profit (P) is $215

  Prove:
  - The profit percentage is 33.86%
-/
theorem profit_percentage : 
  let SP := 850
      P := 215
      CP := SP - P
      profit_percentage := (P / CP.toFloat) * 100
  in profit_percentage = 33.86 :=
by
  sorry

end profit_percentage_l289_289845


namespace tangent_bisects_segment_l289_289725

open EuclideanGeometry

variables {A B C D E : Point} (ABC : Triangle A B C)
variables (isIsosceles : ABC.isIsosceles B C) (D_on_AC : D ∈ AC)
variables (ray_from_B : ∃ L, ray B L E ∧ L.parallel (lineThrough A C))

theorem tangent_bisects_segment 
  (tangent_bisects : TangentToCircumcircleBisectsSegment ABC E AC) : 
  tangent_bisects A B C D E : Prop :=
begin
  -- Proof goes here
  sorry
end

end tangent_bisects_segment_l289_289725


namespace sin_two_angle_BPC_l289_289728

theorem sin_two_angle_BPC
  (α β γ : ℝ)
  (h1 : cos α = 3/5)
  (h2 : cos β = 1/4)
  (hγ : γ = 180 - α - β) :
  sin (2 * γ) = (122 + 48 * Real.sqrt 15) / 200 :=
  by sorry

end sin_two_angle_BPC_l289_289728


namespace marie_lost_erasers_l289_289371

def initialErasers : ℕ := 95
def finalErasers : ℕ := 53

theorem marie_lost_erasers : initialErasers - finalErasers = 42 := by
  sorry

end marie_lost_erasers_l289_289371


namespace triangle_third_side_max_length_l289_289787

theorem triangle_third_side_max_length (a b : ℕ) (ha : a = 5) (hb : b = 11) : ∃ (c : ℕ), c = 15 ∧ (a + c > b ∧ b + c > a ∧ a + b > c) :=
by 
  sorry

end triangle_third_side_max_length_l289_289787


namespace sin_eq_one_fifth_l289_289244

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sin_eq_one_fifth (ϕ : ℝ)
  (h : binomial_coefficient 5 3 * (Real.cos ϕ)^2 = 4) :
  Real.sin (2 * ϕ - π / 2) = 1 / 5 := sorry

end sin_eq_one_fifth_l289_289244


namespace problem1_problem2_problem3_l289_289418

variable {R : Type*} [OrderedRing R] [Real R]

-- Condition: The function y = f(x) is differentiable on ℝ, its derivative f'(x) is increasing and f'(x) > 0
variables (f : R → R) (p : R)
variable [Differentiable R f]
variables (h1 : ∀ x : R, 0 < deriv f x)
variables (h2 : ∀ x₁ x₂ : R, x₁ ≤ x₂ → deriv f x₁ ≤ deriv f x₂)

-- The tangent line at (p, f(p))
def g (x : R) : R := deriv f p * (x - p) + f p

/-- 1. Prove that f(x) ≥ g(x) with equality if and only if x = p. -/
theorem problem1 (x : R) : f x ≥ g f p x ∧ (f x = g f p x ↔ x = p) :=
  sorry

-- Condition: g(a) = f(x₀)
variables (a x₀ : R)
variable (h3 : g f p a = f x₀)

/-- 2. Prove that if g(a) = f(x₀), then x₀ ≤ a. -/
theorem problem2 : x₀ ≤ a :=
  sorry

-- Condition: e^x > ln(x + m) for x ∈ ℝ, x > -m
variables (m : R)
variable (h4 : ∀ x : R, x > -m → exp x > log (x + m))

/-- 3. Prove that m < 5/2. -/
theorem problem3 : m < 5 / 2 :=
  sorry

end problem1_problem2_problem3_l289_289418


namespace area_of_convex_set_leq_4_l289_289844

variable (S : Set (ℝ × ℝ))
variable (is_convex : Convex ℝ S)
variable (contains_origin : (0, 0) ∈ S)
variable (no_other_lattice_points : ∀ p : ℤ × ℤ, p ≠ (0, 0) → p ∉ S)

theorem area_of_convex_set_leq_4
  (quadrant_area_eq : ∀ q ∈ ({1, 2, 3, 4} : Set ℕ), ((quadrant q) ∩ S).Area = a) 
  (all_quadrants_equal : ∀ q₁ q₂ ∈ ({1, 2, 3, 4} : Set ℕ), q₁ ≠ q₂ → ((quadrant q₁) ∩ S).Area = ((quadrant q₂) ∩ S).Area)
  (a_non_neg : 0 ≤ a) :
  S.Area ≤ 4 := by
  sorry

end area_of_convex_set_leq_4_l289_289844


namespace processing_sequences_l289_289090

-- Define the conditions
def num_processing_steps : ℕ := 6
def must_be_consecutive_steps : ℕ := 2
def cannot_be_consecutive_steps : ℕ := 2

-- Prove the number of possible processing sequences
theorem processing_sequences :
  ∃ p : ℕ, p = 144 ∧
    (num_processing_steps = 6) ∧
    (must_be_consecutive_steps = 2) ∧
    (cannot_be_consecutive_steps = 2) :=
by {
  existsi 144,
  split,
  { refl },     -- p = 144
  split,
  { exact rfl }, -- num_processing_steps = 6
  split,
  { exact rfl }, -- must_be_consecutive_steps = 2
  { exact rfl }  -- cannot_be_consecutive_steps = 2
}

end processing_sequences_l289_289090


namespace part1_solution_part2_solution_l289_289222

noncomputable def f (a x : ℝ) : ℝ := a * |x - 1| + |x - a|

theorem part1_solution (x : ℝ) : 
  ∀ x, f 2 x ≤ 4 ↔ (0 ≤ x ∧ x ≤ 8 / 3) := 
by 
  sorry

theorem part2_solution (a : ℝ) :
  (∀ x, f a x ≥ 1) → a ∈ set.Ici 2 :=
by 
  sorry

end part1_solution_part2_solution_l289_289222


namespace not_consecutive_sum_of_faces_l289_289726

theorem not_consecutive_sum_of_faces :
  let die_dots : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 2 else 3 in
  let total_dots := 8 * (1 + 1 + 2 + 2 + 3 + 3) in
  let consecutive_sum (n : ℕ) := n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) in
  (∑ i in finset.range 6, consecutive_sum i = 112) → False :=
by
  -- Let each die have exactly two faces:
  --    * 1 dot each on 2 faces,
  --    * 2 dots each on 2 faces,
  --    * 3 dots each on 2 faces 
  let die_dots := λ n, if n = 1 then 1 else if n = 2 then 2 else 3 

  -- Total dots in a single die
  let dots_per_die := die_dots 1 + die_dots 1 + die_dots 2 + die_dots 2 + die_dots 3 + die_dots 3
  
  -- Total dots for 8 dice forming the cube
  let total_dots := 8 * dots_per_die

  -- Defines the sum of a range of n to n+5
  let consecutive_sum := λ n, n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)

  -- Sum of six consecutive numbers cannot be equal to the total number of dots.
  assume H : ∃ (n : ℕ), ∑ i in finset.range 6, consecutive_sum (n + i) = total_dots
  sorry

end not_consecutive_sum_of_faces_l289_289726


namespace proof_problem_l289_289608

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 3

theorem proof_problem : f (Real.log 2) + f (Real.log (1 / 2)) = 6 := 
by 
  sorry

end proof_problem_l289_289608


namespace max_gcd_lcm_eq_10_l289_289721

open Nat -- Opening the namespace for natural numbers

theorem max_gcd_lcm_eq_10
  (a b c : ℕ) 
  (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) :
  gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_eq_10_l289_289721


namespace minimum_photos_l289_289297

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289297


namespace arithmetic_sequence_10th_term_l289_289976

theorem arithmetic_sequence_10th_term (a_1 : ℕ) (d : ℕ) (n : ℕ) 
  (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 10) : (a_1 + (n - 1) * d) = 28 := by 
  sorry

end arithmetic_sequence_10th_term_l289_289976


namespace eccentricity_of_hyperbola_l289_289615

-- Definition of hyperbola and its conditions
def hyperbola (a b : ℝ) (h_ab : a > b > 0) : Prop :=
  ∀ x y, y^2 / a^2 - x^2 / b^2 = 1

-- Condition for the asymptote being tangent to y = 1 + ln x + ln 2
def asymptote_tangent (a b : ℝ) (h_ab : a > b > 0) : Prop :=
  ∃ m n, n = (a / b) * m ∧ n = 1 + ln m + ln 2

-- The theorem we need to prove
theorem eccentricity_of_hyperbola (a b : ℝ) (h_ab : a > b > 0)
  (h_tangent : asymptote_tangent a b h_ab) : 
  ∃ e : ℝ, e = sqrt (1 + (b / a)^2) ∧ e = sqrt 5 / 2 :=
sorry

end eccentricity_of_hyperbola_l289_289615


namespace scout_troop_profit_l289_289864

theorem scout_troop_profit :
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let bars_per_dollar := 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let bars_per_three_dollars := 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  profit = 320 := by
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  sorry

end scout_troop_profit_l289_289864


namespace min_photographs_42_tourists_3_monuments_l289_289270

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l289_289270


namespace evaluate_expression_l289_289530

theorem evaluate_expression :
  (2/3)^6 * (5/6)^(-4) * (1/4)^2 = 1024 / 7290000 :=
by
  sorry

end evaluate_expression_l289_289530


namespace geometric_progression_common_ratio_l289_289329

-- Define the terms of the geometric progression
def a1 (x : ℝ) : ℝ := x / 2
def a2 (x : ℝ) : ℝ := 2 * x - 3
def a3 (x : ℝ) : ℝ := 18 / x + 1

-- State the theorem to prove the common ratio
theorem geometric_progression_common_ratio (x : ℝ) (hx : x ≠ 0) (h : a1(x) * a3(x) = a2(x) ^ 2) :
  (a2(x) / a1(x)) = 52 / 25 :=
by 
  sorry

end geometric_progression_common_ratio_l289_289329


namespace minimum_photos_l289_289278

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289278


namespace num_integral_values_BC_l289_289694

-- Given data
variables (A B C D E F : Type)
variables [IsTriangle A B C] [IsAngleBisector A B C D] [OnLineInter BC D] [OnLineInter AC E] [OnLineInter BC F]
variables [Parallel AD EF] [DividesTriangleIntoThreeEqualParts AD EF]

theorem num_integral_values_BC (BC : ℝ) (h1 : AB = 7) (h2 : AC = 2 * AB) :
  ∃ n : ℕ, n = 13 ∧ ∀ BC, (7 < BC ∧ BC < 21) → ∃! m, m = BC ∧ BC ∈ ℕ :=
by
  sorry

end num_integral_values_BC_l289_289694


namespace expected_balls_in_original_pos_after_two_transpositions_l289_289399

theorem expected_balls_in_original_pos_after_two_transpositions :
  ∃ (n : ℚ), n = 3.2 := 
sorry

end expected_balls_in_original_pos_after_two_transpositions_l289_289399


namespace triangle_ratio_l289_289320

variable {A B C D E : Type*}
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq E]
variables {triangle : Type*} [semiring triangle]
variables {cos : triangle → triangle}

-- Consider triangle ABC with vertices A, B, and C
-- BD and CE are the altitudes from vertices B and C to sides AC and AB respectively.
-- Find the value of DE/BC given the similarities and trigonometric properties involved.
theorem triangle_ratio (ABC : triangle) (BD CE : triangle) (h1 : is_altitude BD AC) (h2 : is_altitude CE AB) :
  ∃ DE BC : triangle, DE / BC = |cos A| := 
sorry

end triangle_ratio_l289_289320


namespace number_of_multiples_of_3_but_not_9_below_500_l289_289628

theorem number_of_multiples_of_3_but_not_9_below_500 :
  ∃ n : ℕ, n = 111 ∧ ∀ k : ℕ, (k < 500 ∧ k % 3 = 0 ∧ k % 9 ≠ 0) → k ∈ (finset.Ico 1 500) :=
begin
  sorry
end

end number_of_multiples_of_3_but_not_9_below_500_l289_289628


namespace eccentricity_of_given_hyperbola_l289_289991

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h₁ : 1 = 1) : ℝ :=
  let c := real.sqrt (a^2 + b^2)
  in c / a

theorem eccentricity_of_given_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_tangent : b = a / 2) :
  hyperbola_eccentricity a b ha hb (by trivial) = (real.sqrt 5) / 2 :=
by
  sorry

end eccentricity_of_given_hyperbola_l289_289991


namespace trapezoid_parallel_l289_289488

theorem trapezoid_parallel (A B C D O T Q P : Type*) [Trapezoid A B C D] 
  (h_concurrent : AngleBisectorsConcurrent A B C D O)
  (h_inter : IntersectionDiagonals A C B D T)
  (h_Q_on_CD : OnLine Q C D)
  (h_OQD : ∠(O, Q, D) = 90)
  (h_P_on_circ_OTQ : ∃ r : ℝ, IsOnCircumcircle Q O T P r)
  : Parallel T P A D :=
sorry

end trapezoid_parallel_l289_289488


namespace jan_uses_24_gallons_for_plates_and_clothes_l289_289431

theorem jan_uses_24_gallons_for_plates_and_clothes :
  (65 - (2 * 7 + (2 * 7 - 11))) / 2 = 24 :=
by sorry

end jan_uses_24_gallons_for_plates_and_clothes_l289_289431


namespace regular_rate_survey_l289_289521

theorem regular_rate_survey (R : ℝ) 
  (total_surveys : ℕ := 50)
  (rate_increase : ℝ := 0.30)
  (cellphone_surveys : ℕ := 35)
  (total_earnings : ℝ := 605) :
  35 * (1.30 * R) + 15 * R = 605 → R = 10 :=
by
  sorry

end regular_rate_survey_l289_289521


namespace chess_group_players_l289_289443

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by {
  sorry
}

end chess_group_players_l289_289443


namespace nate_pages_left_to_read_l289_289381

-- Define the constants and conditions
def total_pages : ℕ := 400
def percentage_read : ℕ := 20

-- Calculate the number of pages already read
def pages_read := total_pages * percentage_read / 100

-- Calculate the number of pages left
def pages_left := total_pages - pages_read

-- Statement to prove
theorem nate_pages_left_to_read : pages_left = 320 :=
by {
  unfold pages_read,
  unfold pages_left,
  simp,
  sorry -- The proof will be filled in based on the calculations in the solution.
}

end nate_pages_left_to_read_l289_289381


namespace Petrov_in_A_or_B_l289_289825

-- Definitions based on conditions
variable {City : Type} -- Cities are of some type

-- Conditions assumption
variable (connected : City → City → Prop) -- Connection between cities via direct routes
variable [h_conn : ∀ (c1 c2 : City), ∃ (c3 : List City), List.chain connected c3 c1 c2] -- Connectivity

variable (A B X : City) -- Starting city (A), Ivanov's destination (B), Petrov's end city (X)
variable (n : ℕ) -- Petrov buys n tickets for each route

-- Additional conditions for Ivanov
variable (Iv_routes : ∀ (c : City), c ≠ B → ∃ (c' : City), connected c c') -- Ivanov uses all tickets and ends at B

-- Additional condition for Petrov
variable (Pe_trapped : ∀ (c : City), c ≠ X → (∃ (c' : City), connected c c') ∧ n > 1) -- Petrov can only stay in X if he needs a new ticket

-- Statement to prove
theorem Petrov_in_A_or_B : X = A ∨ X = B := 
sorry

end Petrov_in_A_or_B_l289_289825


namespace warriors_can_watch_equidistributed_l289_289829

theorem warriors_can_watch_equidistributed :
  ∃ (schedule: Fin 33 → Fin 33 → Prop), 
    (∀ w : Fin 33, (Finset.univ.filter (λ d, schedule w d)).card = 17) ∧ -- each warrior watches 17 times
    (∀ d : Fin 33, (Finset.range (d + 1)).card = d + 1 ∧ -- each day the correct number of warriors watch
    ∀ w : Fin 33, Finset.mem w (Finset.range (d + 1)) → schedule w d = true) := sorry


end warriors_can_watch_equidistributed_l289_289829


namespace quadratic_has_real_root_for_any_t_l289_289618

theorem quadratic_has_real_root_for_any_t (s : ℝ) :
  (∀ t : ℝ, ∃ x : ℝ, s * x^2 + t * x + s - 1 = 0) ↔ (0 < s ∧ s ≤ 1) :=
by
  sorry

end quadratic_has_real_root_for_any_t_l289_289618


namespace lattice_points_in_region_sum_of_first_2n_terms_l289_289616

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def b (n : ℕ) : ℕ := 2^n + (-1)^n * n

theorem lattice_points_in_region (x y n : ℕ) (hn : n > 0) (hx : x > 0) (hy : y > 0) (hineq : y <= -n * x + 2 * n) :
  a n = n := 
sorry

theorem sum_of_first_2n_terms (n : ℕ) (hn : n > 0) : 
  (∑ k in range (2 * n), b k.succ) = 2^(2*n + 1) - 2 + n := 
sorry

end lattice_points_in_region_sum_of_first_2n_terms_l289_289616


namespace needle_endpoints_not_swapped_by_rotations_l289_289102

-- Define initial positions on the plane
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the transformation (weights)
def weight (p : ℚ × ℚ) : ℚ :=
  let (a, c) := p
  a + 2 * (sqrt 2 : ℝ).to_rat + c

-- Assertion of the problem
theorem needle_endpoints_not_swapped_by_rotations :
  ¬ ∃ (n : ℕ) (rotations : list(ℝ × ℝ → ℝ × ℝ)),
    (rotations.length = n) ∧
    (∃ k, k ∈ rotations ∧ ∀ t, rot(t) ∈ rotations ∧
    (k.to_fun (A) = B) ∧
    (k.to_fun (B) = A)) :=
  sorry

end needle_endpoints_not_swapped_by_rotations_l289_289102


namespace jason_initial_cards_l289_289333

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end jason_initial_cards_l289_289333


namespace find_n_l289_289224

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a + x - b^x

theorem find_n (a b x0 : ℝ) (n : ℤ)
  (h1 : 2019^a = 2020) 
  (h2 : 2020^b = 2019)
  (h3 : f x0 a b = 0)
  (h4 : x0 ∈ set.Ioo (n : ℝ) (n + 1 : ℝ)) : 
  n = -1 := 
sorry

end find_n_l289_289224


namespace range_of_polynomial_l289_289674

namespace RangeOfPolynomial

variables (f : ℝ → ℝ → ℝ)
  
def is_polynomial (f : ℝ → ℝ → ℝ) : Prop := 
  ∃ p : Polynomial ℝ, f = λ x y, p.eval₂ (algebraMap ℝ ℝ) x + p.eval₂ (algebraMap ℝ ℝ) y

theorem range_of_polynomial (f : ℝ → ℝ → ℝ) (h : is_polynomial f) :
  ∃ a : ℝ, 
    (range f = set.univ) ∨ 
    (range f = {y | ∃ b : ℝ, y = a + b ∧ b≥ 0}) ∨
    (range f = {y | ∃ b : ℝ, y = a - b ∧ b≤ 0}) ∨
    (range f = {y | ∃ b : ℝ, y = a + b ∧ b > 0}) ∨
    (range f = {y | ∃ b : ℝ, y = a - b ∧ b < 0}) ∨
    (range f = {a}) :=
sorry

end RangeOfPolynomial

end range_of_polynomial_l289_289674


namespace n_ge_2_pow_k_add_1_minus_1_find_n_good_l289_289472

open Nat

-- Define n-goodness

def n_good (k n : ℕ) : Prop :=
  ∃ (t : Tournament n), ∃ (v : Fin n), ∀ (u : Fin n), v ≠ u → u ∉ t.losses v

-- Statement 1: For a tournament with n players, prove that n >= 2^(k+1) - 1 for some player to have lost all k's matches
theorem n_ge_2_pow_k_add_1_minus_1 (k : ℕ) : ∃ n, n ≥ 2^(k+1) - 1 :=
  sorry

-- Statement 2: Find all n such that 2 is n-good and prove n >= 7
theorem find_n_good (n : ℕ) : (n_good 2 n ↔ n ≥ 7) :=
  sorry

end n_ge_2_pow_k_add_1_minus_1_find_n_good_l289_289472


namespace find_lambda_range_l289_289208

-- Define the vectors a and b and the condition that the angle between them is acute
def a (λ : ℝ) : ℝ × ℝ := (λ, 2)
def b : ℝ × ℝ := (3, 4)

def is_angle_acute (c : ℝ) : Prop := c > 0

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the norms of the vectors
def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Define the cosine of the angle
def cos_theta (λ : ℝ) : ℝ := dot_product (a λ) b / (norm (a λ) * norm b)

-- The theorem statement
theorem find_lambda_range (λ : ℝ) (h : is_angle_acute (cos_theta λ) ∧ cos_theta λ ≠ 1) : 
  λ > -8 / 3 ∧ λ ≠ 3 / 2 :=
sorry

end find_lambda_range_l289_289208


namespace original_selling_price_l289_289867

theorem original_selling_price (C : ℝ) (h : 1.60 * C = 2560) : 1.40 * C = 2240 :=
by
  sorry

end original_selling_price_l289_289867


namespace car_travel_distance_l289_289064

-- Definitions based on the problem
def arith_seq_sum (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1)) / 2

-- Main statement to prove
theorem car_travel_distance : arith_seq_sum 40 (-12) 5 = 88 :=
by sorry

end car_travel_distance_l289_289064


namespace fraction_of_dark_tiles_proof_l289_289148

noncomputable def fraction_of_dark_tiles
  (floor_repeating_pattern : ℕ)
  (section_size : ℕ)
  (dark_tiles_in_top_left : ℕ)
  (additional_dark_tiles_in_rest : ℕ)
  (configuration : section_size = 4 ∧ dark_tiles_in_top_left = 3 ∧ additional_dark_tiles_in_rest = 2 ∧ floor_repeating_pattern ≥ section_size)
  : ℚ :=
  let total_dark_tiles := dark_tiles_in_top_left + additional_dark_tiles_in_rest in
  let total_tiles := section_size * section_size in
  total_dark_tiles / total_tiles

theorem fraction_of_dark_tiles_proof
  (floor_repeating_pattern section_size dark_tiles_in_top_left additional_dark_tiles_in_rest : ℕ)
  (h : section_size = 4 ∧ dark_tiles_in_top_left = 3 ∧ additional_dark_tiles_in_rest = 2 ∧ floor_repeating_pattern ≥ section_size)
  : fraction_of_dark_tiles floor_repeating_pattern section_size dark_tiles_in_top_left additional_dark_tiles_in_rest h = 5 / 16 :=
by sorry

end fraction_of_dark_tiles_proof_l289_289148


namespace smaller_root_of_quadratic_l289_289902

theorem smaller_root_of_quadratic :
  let f : ℚ → ℚ := λ x, (x - 2 / 3) * (x - 2 / 3) + (x - 2 / 3) * (x - 1 / 3) - 1 / 9
  ∃ x : ℚ, f x = 0 ∧ ∀ y : ℚ, f y = 0 → x ≤ y :=
by
  sorry

end smaller_root_of_quadratic_l289_289902


namespace range_of_m_l289_289204

theorem range_of_m (m : ℝ) (x : ℝ) (h₁ : x^2 - 8*x - 20 ≤ 0) 
  (h₂ : (x - 1 - m) * (x - 1 + m) ≤ 0) (h₃ : 0 < m) : 
  m ≤ 3 := sorry

end range_of_m_l289_289204


namespace four_digit_palindrome_perfect_squares_l289_289103

theorem four_digit_palindrome_perfect_squares : 
  ∃ (count : ℕ), count = 2 ∧ 
  (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → 
            ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
            n = 1001 * a + 110 * b ∧ 
            ∃ k : ℕ, k * k = n) → count = 2 := by
  sorry

end four_digit_palindrome_perfect_squares_l289_289103


namespace find_prime_powers_l289_289928

open Nat

theorem find_prime_powers (p x y : ℕ) (hp : p.Prime) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 ↔
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end find_prime_powers_l289_289928


namespace ali_initial_money_l289_289129

theorem ali_initial_money (X : ℝ) (h1 : X / 2 - (1 / 3) * (X / 2) = 160) : X = 480 :=
by sorry

end ali_initial_money_l289_289129


namespace trigonometric_identity_solution_l289_289071

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.447 * tan (2 * x) * (tan (3 * x))^2 * (tan (5 * x))^2 = tan (2 * x) + (tan (3 * x))^2 - (tan (5 * x))^2 ↔ 
  (cos (2 * x) ≠ 0 ∧ cos (3 * x) ≠ 0 ∧ cos (5 * x) ≠ 0) → 
  (∃ k : ℤ, x = π * k ∨ x = (π / 32) * (4 * k + 1)) :=
by
  sorry

end trigonometric_identity_solution_l289_289071


namespace tangent_segments_right_triangle_iff_l289_289855

theorem tangent_segments_right_triangle_iff (ABC : Triangle) (P : Point)
  (hP : P ∈ incircle ABC) :
  (∃ (A B C : Segment) (hA : A ∈ tangents P (excircle ABC A)) (hB : B ∈ tangents P (excircle ABC B)) (hC : C ∈ tangents P (excircle ABC C)),
    is_right_triangle A B C) ↔
  (∃ (M1 M2 : Point), M1 ∈ midpoints ABC ∧ M2 ∈ midpoints ABC ∧ P ∈ line_through M1 M2) :=
sorry

end tangent_segments_right_triangle_iff_l289_289855


namespace min_photos_l289_289298

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289298


namespace elena_music_class_l289_289256

theorem elena_music_class (k : ℕ) (num_students num_girls num_boys : ℕ) 
    (ratio_girls_boys : 3 * k = num_girls ∧ 4 * k = num_boys)
    (total_students : num_girls + num_boys = num_students) :
  num_students = 35 → num_girls = 15 :=
by
  intro h_num_students
  have h7k : 7 * k = 35, from by linarith [← total_students, h_num_students, ratio_girls_boys.left, ratio_girls_boys.right]
  have hk : k = 5, by linarith [h7k]
  have hnum_girls : num_girls = 3 * 5, from by linarith [ratio_girls_boys.left, hk]
  linarith [hnum_girls]

end elena_music_class_l289_289256


namespace solve_quadratic_equation_l289_289403

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end solve_quadratic_equation_l289_289403


namespace simplify_expression_l289_289737

variables (x y : ℝ)

theorem simplify_expression :
  (3 * x)^4 + (4 * x) * (x^3) + (5 * y)^2 = 85 * x^4 + 25 * y^2 :=
by
  sorry

end simplify_expression_l289_289737


namespace cos_diff_proof_l289_289952

noncomputable def cos_diff (α β : ℝ) : ℝ := Real.cos (α - β)

theorem cos_diff_proof (α β : ℝ) 
  (h1 : Real.cos α - Real.cos β = 1 / 2)
  (h2 : Real.sin α - Real.sin β = 1 / 3) :
  cos_diff α β = 59 / 72 := by
  sorry

end cos_diff_proof_l289_289952


namespace min_positive_pairwise_sum_l289_289954

def pairwise_sum (a : Fin 95 → ℤ) : ℤ :=
  ∑ i j in Finset.offDiag (Finset.univ : Finset (Fin 95)), a i * a j

theorem min_positive_pairwise_sum (a : Fin 95 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ k, k > 0 ∧ k = pairwise_sum a := 
sorry

end min_positive_pairwise_sum_l289_289954


namespace dot_product_possible_values_l289_289351

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 2))]

theorem dot_product_possible_values (u v : euclidean_space ℝ (fin 2)) 
  (hu_norm : ∥u∥ = 5) (hv_norm : ∥v∥ = 13) :
  ∃ (cos_theta : ℝ), -1 ≤ cos_theta ∧ cos_theta ≤ 1 ∧
    (u ⬝ v) = 65 * cos_theta :=
by
  sorry

end dot_product_possible_values_l289_289351


namespace find_E_l289_289242

variable (x E x1 x2 : ℝ)

/-- Given conditions as assumptions: -/
axiom h1 : (x + 3)^2 / E = 2
axiom h2 : x1 - x2 = 14

/-- Prove the required expression for E in terms of x: -/
theorem find_E : E = (x + 3)^2 / 2 := sorry

end find_E_l289_289242


namespace calculate_determinant_l289_289217

open Real -- To bring in the real number operations and notation.

theorem calculate_determinant : 
  let integral := ∫ x in 1..2, x
  let determinant := integral * 2 - 3 
  determinant = 0 := 
by
  sorry

end calculate_determinant_l289_289217


namespace minimum_photos_taken_l289_289310

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l289_289310


namespace imaginary_part_z_times_z_plus_i_l289_289582

noncomputable def z : ℂ := 2 - complex.i
noncomputable def conj_z : ℂ := complex.conj z
noncomputable def z_plus_i : ℂ := conj_z + complex.i
noncomputable def z_times_z_plus_i : ℂ := z * z_plus_i

theorem imaginary_part_z_times_z_plus_i : (z_times_z_plus_i).im = 2 := 
by sorry

end imaginary_part_z_times_z_plus_i_l289_289582


namespace hall_marriage_theorem_l289_289366

-- Definitions using conditions from part (a)
variables {α : Type} (V : α → Prop) (E : α → α → Prop)
variables (X Y : set α)

-- Vertices X and Y are disjoint
axiom disjoint_sets : disjoint X Y

-- All edges connect vertices from X to vertices from Y
axiom bipartite_graph : ∀ (a b : α), E a b → (a ∈ X ∧ b ∈ Y) ∨ (a ∈ Y ∧ b ∈ X)

-- Hall's Marriage Theorem statement
theorem hall_marriage_theorem :
  (∃ M : set (α × α), ∀ x ∈ X, ∃ y ∈ Y, (x, y) ∈ M) ↔ 
  (∀ S ⊆ X, ∃ T ⊆ Y, |T| ≥ |S| ∧ ∀ x ∈ S, ∃ y ∈ T, E x y) :=
sorry

end hall_marriage_theorem_l289_289366


namespace amount_allocated_to_food_l289_289659

-- Define the conditions
def ratio_HFM : ℕ → ℕ → ℕ → Prop :=
  λ H F M, 5 * F = 4 * H ∧ H = 5 * M

def total_amount : ℕ → ℕ → ℕ → ℕ → Prop :=
  λ H F M T, H + F + M = T

-- Problem statement
theorem amount_allocated_to_food
  (H F M T : ℕ) 
  (h_ratio : ratio_HFM H F M)
  (h_total : total_amount H F M 1800) :
  F = 720 :=
by
  -- ... Proof steps would go here
  sorry

end amount_allocated_to_food_l289_289659


namespace find_lambda_l289_289999

-- Define the vectors a, b, and c
def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (1, 3)
def vector_c : ℝ × ℝ := (3, 2)

-- Define the parallel condition between (a + λb) and c
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = k * v

-- The formal statement to prove
theorem find_lambda (λ : ℝ) :
  parallel (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2) vector_c ↔ λ = -1 := by
  sorry

end find_lambda_l289_289999


namespace mark_more_hours_than_kate_l289_289477

-- Definitions for hours charged by Kate, Pat, and Mark
variables (K P M : ℕ)

-- Conditions given in the problem
def total_hours := K + P + M = 198
def pat_to_kate := P = 2 * K
def pat_to_mark := P = M / 3

-- The final statement we need to prove
theorem mark_more_hours_than_kate (h1 : total_hours)
                                  (h2 : pat_to_kate)
                                  (h3 : pat_to_mark) :
  M - K = 110 :=
sorry

end mark_more_hours_than_kate_l289_289477


namespace average_exercise_correct_l289_289376

axiom students_exercise : 
  (0:ℕ) * 0 + (2:ℕ) * 1 + (4:ℕ) * 2 + (3:ℕ) * 3 + (5:ℕ) * 4 + (7:ℕ) * 5 + (4:ℕ) * 6 = 98

axiom students_number :
  (0:ℕ) + (2:ℕ) + (4:ℕ) + (3:ℕ) + (5:ℕ) + (7:ℕ) + (4:ℕ) = 25

noncomputable def average_exercise : ℚ := (98 : ℚ) / (25 : ℚ)

theorem average_exercise_correct : (average_exercise ≈ 3.92) :=
by {
  sorry
}

end average_exercise_correct_l289_289376


namespace coefficient_of_linear_term_l289_289409

def polynomial (x : ℝ) := x^2 - 2 * x - 3

theorem coefficient_of_linear_term : (∀ x : ℝ, polynomial x = x^2 - 2 * x - 3) → -2 = -2 := by
  intro h
  sorry

end coefficient_of_linear_term_l289_289409


namespace rowing_problem_l289_289113

noncomputable def rowing_time_to_Big_Rock_and_back 
    (rower_speed : ℝ) 
    (river_speed : ℝ) 
    (distance : ℝ) : ℝ :=
let upstream_speed := rower_speed - river_speed in
let downstream_speed := rower_speed + river_speed in
let time_upstream := distance / upstream_speed in
let time_downstream := distance / downstream_speed in
time_upstream + time_downstream

theorem rowing_problem 
    (rower_speed : ℝ)
    (river_speed : ℝ)
    (distance : ℝ)
    (h1 : rower_speed = 6)
    (h2 : river_speed = 1)
    (h3 : distance = 2.916666666666667) : 
    rowing_time_to_Big_Rock_and_back rower_speed river_speed distance = 1 := by
  
sorry

end rowing_problem_l289_289113


namespace incorrect_games_leq_75_percent_l289_289840

theorem incorrect_games_leq_75_percent (N : ℕ) (win_points : ℕ) (draw_points : ℚ) (loss_points : ℕ) (incorrect : (ℕ × ℕ) → Prop) :
  (win_points = 1) → (draw_points = 1 / 2) → (loss_points = 0) →
  ∀ (g : ℕ × ℕ), incorrect g → 
  ∃ (total_games incorrect_games : ℕ), 
    total_games = N * (N - 1) / 2 ∧
    incorrect_games ≤ 3 / 4 * total_games := sorry

end incorrect_games_leq_75_percent_l289_289840


namespace shoppers_count_l289_289325

-- Define the amounts given in the problem conditions
def giselle_amount := 120
def isabella_more_than_giselle := 15
def isabella_more_than_sam := 45
def each_shopper_amount := 115

-- Calculate Isabella's and Sam's amounts based on conditions
noncomputable def isabella_amount := giselle_amount + isabella_more_than_giselle
noncomputable def sam_amount := isabella_amount - isabella_more_than_sam

-- Total money collected
noncomputable def total_amount := isabella_amount + sam_amount + giselle_amount

-- Number of shoppers (we want to prove this equals 3)
def number_of_shoppers := total_amount / each_shopper_amount

-- The main theorem to be proved
theorem shoppers_count : number_of_shoppers = 3 :=
by
  sorry

end shoppers_count_l289_289325


namespace junk_mail_total_l289_289781

theorem junk_mail_total (houses_per_block : ℕ) (blocks : ℕ) (junk_mail_per_house : ℕ)
    (h1 : houses_per_block = 50)
    (h2 : blocks = 3)
    (h3 : junk_mail_per_house = 45) :
    (houses_per_block * blocks * junk_mail_per_house) = 6750 :=
begin
  rw [h1, h2, h3],
  norm_num,
end

end junk_mail_total_l289_289781


namespace sum_three_digit_integers_from_200_to_900_l289_289809

theorem sum_three_digit_integers_from_200_to_900 : 
  let a := 200
  let l := 900
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 385550 := by
    let a := 200
    let l := 900
    let d := 1
    let n := (l - a) / d + 1
    let S := n / 2 * (a + l)
    sorry

end sum_three_digit_integers_from_200_to_900_l289_289809


namespace distinct_solutions_for_a_l289_289174

theorem distinct_solutions_for_a (a : ℝ) :
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ, 
    y1 = 0 ∧ x1 = -1 / real.sqrt a ∧
    y2 = 3 ∧ (x2 = 3 + 1 / real.sqrt (3 - a) ∨ x2 = 3 - 1 / real.sqrt (3 - a)) ∧
    y3 = 3 ∧ (x3 = 3 + 1 / real.sqrt (3 - a) ∨ x3 = 3 - 1 / real.sqrt (3 - a)) ∧
    y4 = 1 ∧ x4 = 1 + 1 / real.sqrt (1 - a)) ∨
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ, 
    y1 = 0 ∧ (x1 = 1 / real.sqrt a ∨ x1 = -1 / real.sqrt a) ∧
    y2 = 0 ∧ (x2 = 1 / real.sqrt a ∨ x2 = -1 / real.sqrt a) ∧
    y3 = 3 ∧ x3 = 3 + 1 / real.sqrt (3 - a) ∧
    y4 = 1 ∧ x4 = 1 - 1 / real.sqrt (a - 1)) ↔ 
  (0 < a ∧ a < 1) ∨ (11 / 4 ≤ a ∧ a < 3) := 
sorry

end distinct_solutions_for_a_l289_289174


namespace trip_duration_l289_289492

/--
Given:
1. The car averages 30 miles per hour for the first 5 hours of the trip.
2. The car averages 42 miles per hour for the rest of the trip.
3. The average speed for the entire trip is 34 miles per hour.

Prove: 
The total duration of the trip is 7.5 hours.
-/
theorem trip_duration (t T : ℝ) (h1 : 150 + 42 * t = 34 * T) (h2 : T = 5 + t) : T = 7.5 :=
by
  sorry

end trip_duration_l289_289492


namespace geraldo_drank_7_pints_l289_289055

-- Conditions
def total_gallons : ℝ := 20
def num_containers : ℕ := 80
def gallons_to_pints : ℝ := 8
def containers_drank : ℝ := 3.5

-- Problem statement
theorem geraldo_drank_7_pints :
  let total_pints : ℝ := total_gallons * gallons_to_pints
  let pints_per_container : ℝ := total_pints / num_containers
  let pints_drank : ℝ := containers_drank * pints_per_container
  pints_drank = 7 :=
by
  sorry

end geraldo_drank_7_pints_l289_289055


namespace perimeter_inequality_l289_289652

variable {A B C D E F : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

-- Acute-angled triangle ABC with altitudes AD, BE, and CF
variable (ABC_is_acute : Triangle A B C)
variable (AD_altitude : Altitude A D)
variable (BE_altitude : Altitude B E)
variable (CF_altitude : Altitude C F)

def perimeter_triangle1 (A B C : ℝ) : ℝ :=
  A + B + C

def perimeter_triangle2 (D E F : ℝ) : ℝ :=
  D + E + F

theorem perimeter_inequality
  (ABC_is_acute : Triangle A B C)
  (AD_altitude : Altitude A D)
  (BE_altitude : Altitude B E)
  (CF_altitude : Altitude C F)
  (AB BC AC DE EF FD : ℝ) :
  perimeter_triangle2 DE EF FD ≤ (1/2) * perimeter_triangle1 AB BC AC := by
  sorry

end perimeter_inequality_l289_289652


namespace Sandwiches_count_l289_289395

-- Define the number of toppings and the number of choices for the patty
def num_toppings : Nat := 10
def num_choices_per_topping : Nat := 2
def num_patties : Nat := 3

-- Define the theorem to prove the total number of sandwiches
theorem Sandwiches_count : (num_choices_per_topping ^ num_toppings) * num_patties = 3072 :=
by
  sorry

end Sandwiches_count_l289_289395


namespace geometric_progression_common_ratio_l289_289327

variable (x : ℝ)

noncomputable def a1 := x / 2
noncomputable def a2 := 2 * x - 3
noncomputable def a3 := 18 / x + 1

theorem geometric_progression_common_ratio :
  a1 * a3 = a2^2 → x = 25 / 8 → a2 / a1 = 2.08 :=
by sorry

end geometric_progression_common_ratio_l289_289327


namespace relatively_prime_fractions_not_integers_simultaneously_l289_289676

theorem relatively_prime_fractions_not_integers_simultaneously 
  (m n : ℕ) (h1 : Nat.coprime m n) (h2 : 0 < m) (h3 : 0 < n) :
  ¬ (∃ a b : ℤ, 
        (n^4 + m = a * (m^2 + n^2)) ∧ 
        (n^4 - m = b * (m^2 - n^2))) :=
sorry

end relatively_prime_fractions_not_integers_simultaneously_l289_289676


namespace geraldo_drank_l289_289052

def total_gallons : ℝ := 20
def total_containers : ℝ := 80
def containers_drank : ℝ := 3.5
def pints_per_gallon : ℝ := 8

theorem geraldo_drank :
  let tea_per_container : ℝ := total_gallons / total_containers in
  let pints_per_container : ℝ := tea_per_container * pints_per_gallon in
  let total_pints_drank : ℝ := containers_drank * pints_per_container in
  total_pints_drank = 7 :=
by
  sorry

end geraldo_drank_l289_289052


namespace gcd_1734_816_1343_l289_289763

theorem gcd_1734_816_1343 : Int.gcd (Int.gcd 1734 816) 1343 = 17 :=
by
  sorry

end gcd_1734_816_1343_l289_289763


namespace solve_z_sqrt_equation_l289_289171

theorem solve_z_sqrt_equation : 
  ∃ z : ℤ, sqrt (9 + 3 * (z : ℝ)) = 12 ∧ z = 45 :=
by 
  sorry

end solve_z_sqrt_equation_l289_289171


namespace sum_of_angles_at_BECFD_l289_289439

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk (-1) (-7)
def B := Point.mk 2 5
def C := Point.mk 3 (-8)
def D := Point.mk (-3) 4
def E := Point.mk 5 (-1)
def F := Point.mk (-4) (-2)
def G := Point.mk 6 4

theorem sum_of_angles_at_BECFD :
  angle_at B + angle_at E + angle_at C + angle_at F + angle_at D = 135 :=
  sorry

end sum_of_angles_at_BECFD_l289_289439


namespace polynomial_divisibility_l289_289821

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def a (n k : ℕ) (a0 : ℤ) : ℤ :=
  a0 * (-1)^(k - 1) * (factorial (2 * k - 2)) / (factorial (k - 1) * factorial k * 2^(2 * k - 1))

theorem polynomial_divisibility (n : ℕ) (a0 : ℤ) (h : a0 = 1 ∨ a0 = -1) :
  (1 + x) - ((a n 0 a0) + (a n 1 a0) * x + (a n 2 a0) * x^2 + ... + (a n n a0) * x^n) ^ 2 = k * x^(n + 1) :=
sorry

end polynomial_divisibility_l289_289821


namespace bus_driver_regular_rate_l289_289835

theorem bus_driver_regular_rate (R : ℝ) (h1 : 976 = (40 * R) + (14.32 * (1.75 * R))) : 
  R = 15 := 
by
  sorry

end bus_driver_regular_rate_l289_289835


namespace cone_sphere_ratio_l289_289862

-- Defining the conditions and proof goals
theorem cone_sphere_ratio (r h : ℝ) (h_cone_sphere_radius : r ≠ 0) 
  (h_cone_volume : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  -- All the assumptions / conditions given in the problem
  sorry -- Proof omitted

end cone_sphere_ratio_l289_289862


namespace approx_number_of_ants_l289_289512

noncomputable def number_of_ants (field_width_ft field_length_ft : ℝ) (concentration_ants : ℝ) (rock_percentage : ℝ) : ℝ :=
  let field_width_in := field_width_ft * 12
  let field_length_in := field_length_ft * 12
  let field_area_in² := field_width_in * field_length_in
  let usable_area_in² := field_area_in² * (1 - rock_percentage)
  concentration_ants * usable_area_in²

theorem approx_number_of_ants :
  number_of_ants 200 500 2 0.1 ≈ 26 * 10^6 :=
by
  sorry

end approx_number_of_ants_l289_289512


namespace find_a_extremum_l289_289216

-- Define the function f(x)
def f (x a : ℝ) := Real.exp x - a * x

-- Define the first derivative of f(x) with respect to x
def f_prime (x a : ℝ) := (Real.exp x) - a

-- State the theorem
theorem find_a_extremum (a : ℝ) : (f_prime 0 a = 0) ↔ (a = 1) :=
by
  sorry

end find_a_extremum_l289_289216


namespace total_worth_is_24_point7_l289_289153

def cost_of_taxable_items (x : ℝ) : Prop := 0.10 * x = 0.30
def cost_of_tax_free_items : ℝ := 21.7

def total_worth (x : ℝ) : ℝ := x + cost_of_tax_free_items

theorem total_worth_is_24_point7 (x : ℝ) : cost_of_taxable_items x → total_worth x = 24.7 :=
begin
  sorry
end

end total_worth_is_24_point7_l289_289153


namespace integral_abs_exp_l289_289030

open Real Set

theorem integral_abs_exp :
  ∫ x in -2..4, exp (abs x) = exp 4 + exp 2 - 2 :=
by
  sorry

end integral_abs_exp_l289_289030


namespace merchant_spent_for_belle_l289_289042

def dress_cost (S : ℤ) (H : ℤ) : ℤ := 6 * S + 3 * H
def hat_cost (S : ℤ) (H : ℤ) : ℤ := 3 * S + 5 * H
def belle_cost (S : ℤ) (H : ℤ) : ℤ := S + 2 * H

theorem merchant_spent_for_belle :
  ∃ (S H : ℤ), dress_cost S H = 105 ∧ hat_cost S H = 70 ∧ belle_cost S H = 25 :=
by
  sorry

end merchant_spent_for_belle_l289_289042


namespace vector_perpendicular_condition_l289_289594

variables {ℝ : Type} [inner_product_space ℝ (euclidean_space ℝ ℝ)]

theorem vector_perpendicular_condition (a b : ℝ) :
  ∥a + b∥ = ∥a - b∥ → inner_product_space.inner a b = 0 :=
by sorry

end vector_perpendicular_condition_l289_289594


namespace smallest_positive_period_and_range_sin_2x0_when_zero_point_l289_289984

noncomputable def f (x : ℝ) : ℝ := 
  sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + sin (x + π / 4) * sin (x - π / 4)

theorem smallest_positive_period_and_range :
  (∀ x : ℝ, f(x) = f(x + π)) ∧ (∀ y ∈ Set.Image f Set.Univ, y ∈ Set.Icc (-3/2 : ℝ) (5/2 : ℝ)) := sorry

theorem sin_2x0_when_zero_point (x0 : ℝ) (h0 : 0 ≤ x0 ∧ x0 ≤ π / 2) (h1 : f x0 = 0) :
  sin (2 * x0) = (sqrt 15 - sqrt 3) / 8 := sorry

end smallest_positive_period_and_range_sin_2x0_when_zero_point_l289_289984


namespace wheel_rotation_angle_l289_289126

-- Define the conditions
def radius : ℝ := 20
def arc_length : ℝ := 40

-- Define the theorem stating the desired proof problem
theorem wheel_rotation_angle (r : ℝ) (l : ℝ) (h_r : r = radius) (h_l : l = arc_length) :
  l / r = 2 := 
by sorry

end wheel_rotation_angle_l289_289126


namespace bracelet_beads_arrangement_l289_289313

theorem bracelet_beads_arrangement (n : ℕ) (hc : 8 = n) (ha : True) : 
  ∃ k, k = 315 := 
by
  use 315
  trivial

end bracelet_beads_arrangement_l289_289313


namespace find_m_given_ellipse_focus_l289_289963

theorem find_m_given_ellipse_focus
  (m : ℝ)
  (h1 : 0 < m)
  (h2 : ∃ (x y : ℝ), (x, y) = (0, 4))
  (h3 : ∀ x y : ℝ, (x^2 / 25) + (y^2 / m^2) = 1) :
  m = √41 :=
sorry

end find_m_given_ellipse_focus_l289_289963


namespace inverse_function_value_l289_289612

def g (x : ℝ) : ℝ := 4 * x ^ 3 - 5

theorem inverse_function_value (x : ℝ) : g x = -1 ↔ x = 1 :=
by
  sorry

end inverse_function_value_l289_289612


namespace apple_pie_theorem_l289_289667

theorem apple_pie_theorem (total_apples : ℕ) (not_ripe_apples : ℕ) (apples_per_pie : ℕ) (total_ripe_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 34)
  (h2 : not_ripe_apples = 6)
  (h3 : apples_per_pie = 4)
  (h4 : total_ripe_apples = total_apples - not_ripe_apples)
  (h5 : number_of_pies = total_ripe_apples / apples_per_pie) :
  number_of_pies = 7 :=
  by
  have h6 : total_apples - not_ripe_apples = 28 := by rw [h1, h2]; norm_num
  have h7 : total_ripe_apples = 28 := by rw [h4, h6]
  have h8 : 28 / apples_per_pie = 7 := by rw [h3]; norm_num
  rw [h7, h5, h8]
  sorry

end apple_pie_theorem_l289_289667


namespace sum_of_solutions_is_neg_20_l289_289253

def sum_of_solutions_abs_eq_10 : ℤ :=
  let solutions := {x | abs (x + 10) = 10} in
  solutions.sum

theorem sum_of_solutions_is_neg_20 : sum_of_solutions_abs_eq_10 = -20 :=
  sorry

end sum_of_solutions_is_neg_20_l289_289253


namespace no_such_A_l289_289544

theorem no_such_A : ∀ (A : ℕ), A > 0 ∧ A < 10 → ¬∃ x : ℕ, x*x - 2*A*x + (A+1)0 = 0 :=
by
  intros A hA hx
  cases hA with hApos hAdig
  obtain ⟨x, hxsol⟩ := hx
  sorry

end no_such_A_l289_289544


namespace rhombus_area_l289_289773

theorem rhombus_area (s d1 : ℝ) (h_s : s = 5) (h_d1 : d1 = 8) : 
  let d2 := 6 in
  (1/2) * d1 * d2 = 24 :=
by
  sorry

end rhombus_area_l289_289773


namespace midpoints_form_regular_dodecagon_l289_289322

noncomputable def square_side_length : ℝ := 2
noncomputable def origin : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (-1, -1)
noncomputable def B : ℝ × ℝ := (1, -1)
noncomputable def C : ℝ × ℝ := (1, 1)
noncomputable def D : ℝ × ℝ := (-1, 1)

noncomputable def K : ℝ × ℝ := (0, real.sqrt 3 - 1)
noncomputable def L : ℝ × ℝ := (1 - real.sqrt 3, 0)
noncomputable def M : ℝ × ℝ := (0, 1 - real.sqrt 3)
noncomputable def N : ℝ × ℝ := (real.sqrt 3 - 1, 0)

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

noncomputable def P1 : ℝ × ℝ := midpoint K L
noncomputable def P2 : ℝ × ℝ := midpoint L M
noncomputable def P3 : ℝ × ℝ := midpoint M N
noncomputable def P4 : ℝ × ℝ := midpoint N K

noncomputable def P5 : ℝ × ℝ := midpoint A K
noncomputable def P6 : ℝ × ℝ := midpoint B K
noncomputable def P7 : ℝ × ℝ := midpoint B L
noncomputable def P8 : ℝ × ℝ := midpoint C L
noncomputable def P9 : ℝ × ℝ := midpoint C M
noncomputable def P10 : ℝ × ℝ := midpoint D M
noncomputable def P11 : ℝ × ℝ := midpoint D N
noncomputable def P12 : ℝ × ℝ := midpoint A N

theorem midpoints_form_regular_dodecagon :
  set.to_finset {P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12}.elems.card = 12 ∧
  (∀ i j, (i ≠ j) → ∥P i - origin∥ = ∥P j - origin∥) :=
sorry

end midpoints_form_regular_dodecagon_l289_289322


namespace houses_after_boom_l289_289889

theorem houses_after_boom (h_pre_boom : ℕ) (h_built : ℕ) (h_count : ℕ)
  (H1 : h_pre_boom = 1426)
  (H2 : h_built = 574)
  (H3 : h_count = h_pre_boom + h_built) :
  h_count = 2000 :=
by {
  sorry
}

end houses_after_boom_l289_289889


namespace minimum_photos_l289_289296

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l289_289296


namespace integral_f_integral_g_l289_289922

noncomputable def f (x : ℝ) : ℝ := sqrt (9 - x^2)
noncomputable def g (x : ℝ) : ℝ := x + sin x

open Real

theorem integral_f : ∫ x in -3..3, f x = 9 * π / 2 := 
sorry

theorem integral_g : ∫ x in 0..π / 2, g x = π^2 / 8 + 1 := 
sorry

end integral_f_integral_g_l289_289922


namespace bookman_initial_total_copies_l289_289892

def initial_total_copies (hardback_count : ℕ) (hardback_price : ℕ) (paperback_price : ℕ) (sold_copies : ℕ) (remaining_value : ℕ) (initial_copies : ℕ) :=
  hardback_count = 10 ∧ 
  hardback_price = 20 ∧ 
  paperback_price = 10 ∧ 
  sold_copies = 14 ∧ 
  remaining_value = 360 ∧ 
  initial_copies = 50 

theorem bookman_initial_total_copies :
  ∀ (P : ℕ), initial_total_copies 10 20 10 14 360 (10 + 4 + (360 / 10)) →
  10 + P = 50 :=
by
  intro P,
  intro h,
  sorry

end bookman_initial_total_copies_l289_289892


namespace minimum_photos_l289_289290

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289290


namespace correct_statements_identification_l289_289132

theorem correct_statements_identification :
  (∀ k : ℤ, ¬∀ α : ℝ, α = k * π / 2) ∧
  (∀ k : ℤ, ∃ c : ℝ × ℝ, c = (π * k + 3 * π / 4, 0)) ∧
  ¬(∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x > tan (x + π)) ∧
  (∀ (x : ℝ), ∀ φ : ℝ, sin (2 * x - π / 3) = sin (2 * (x - π / 6))) :=
by
  split
  · intro k
    assume h : ∀ α : ℝ, α = k * π / 2
    sorry,
  split
  · intro k
    exists (k * π + 3 * π / 4, 0)
    rfl,
  split
  · assume h : ∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x > tan (x + π)
    sorry,
  · intro x
    intro φ
    calc
      sin (2 * x - π / 3) = sin (2 * (x - π / 6)) : by repeat { sorry }.

end correct_statements_identification_l289_289132


namespace desiree_age_l289_289155

-- Definitions of the given variables and conditions
variables (D C : ℝ)

-- Given conditions
def condition1 : Prop := D = 2 * C
def condition2 : Prop := D + 30 = 0.6666666 * (C + 30) + 14
def condition3 : Prop := D = 2.99999835

-- Main theorem to prove
theorem desiree_age : D = 2.99999835 :=
by
  { sorry }

end desiree_age_l289_289155


namespace f_monotonic_decreasing_l289_289769

def f (x : ℝ) : ℝ := x - Real.log x

theorem f_monotonic_decreasing : ∀ x, 0 < x ∧ x < 1 → ∃ δ > 0, ∀ ε > 0, f (x + ε) - f x ≤ -δ := 
by sorry

end f_monotonic_decreasing_l289_289769


namespace largest_number_with_unique_digits_summing_to_17_l289_289791

theorem largest_number_with_unique_digits_summing_to_17 : ∃ n : ℕ, 
  (∀ i j : ℕ, i ≠ j → (List.toDigits n).nth i ≠ (List.toDigits n).nth j) ∧
  (List.sum (List.filterMap (fun x => if x ∈ (List.toDigits n) then some (x : ℕ) else none) (finset.range 10))) = 17 ∧
  n = 7543210 :=
begin
  sorry
end

end largest_number_with_unique_digits_summing_to_17_l289_289791


namespace largest_number_with_unique_digits_summing_to_17_l289_289792

theorem largest_number_with_unique_digits_summing_to_17 : ∃ n : ℕ, 
  (∀ i j : ℕ, i ≠ j → (List.toDigits n).nth i ≠ (List.toDigits n).nth j) ∧
  (List.sum (List.filterMap (fun x => if x ∈ (List.toDigits n) then some (x : ℕ) else none) (finset.range 10))) = 17 ∧
  n = 7543210 :=
begin
  sorry
end

end largest_number_with_unique_digits_summing_to_17_l289_289792


namespace max_gcd_lcm_l289_289714

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l289_289714


namespace distance_from_A_to_line_BC_l289_289946

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def distance_from_point_to_line (A B C : Point3D) : ℝ :=
  let AB := (B.x - A.x, B.y - A.y, B.z - A.z)
  let BC := (C.x - B.x, C.y - B.y, C.z - B.z)
  let magnitude := λ (v : ℝ × ℝ × ℝ), Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let dot_product := λ (v w : ℝ × ℝ × ℝ), v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let cos_theta := dot_product AB BC / ((magnitude AB) * (magnitude BC))
  magnitude AB * Real.sqrt (1 - cos_theta ^ 2)

theorem distance_from_A_to_line_BC (A B C : Point3D) (hA : A = ⟨0, 0, 2⟩) 
(hB : B = ⟨1, 0, 2⟩) (hC : C = ⟨0, 2, 0⟩) : 
distance_from_point_to_line A B C = 2 * Real.sqrt 2 / 3 := 
by
  sorry

end distance_from_A_to_line_BC_l289_289946


namespace probability_six_draws_one_each_suit_l289_289635

-- Definitions for the problem conditions
def probability_of_suit (n : ℕ) : ℚ := (13 : ℚ) / 52 -- Probability of drawing a card from any given suit

def probability_at_least_one_each_suit (n : ℕ) : ℚ := 
  let prob_miss_one := (3 / 4) ^ n in
  let prob_miss_two := (1 / 2) ^ n in
  let prob_miss_three := (1 / 4) ^ n in
  let prob_miss_four := (0 : ℚ) ^ n in
  1 - (4 * prob_miss_one - 6 * prob_miss_two + 4 * prob_miss_three - prob_miss_four)

-- Main statement to prove
theorem probability_six_draws_one_each_suit : 
  probability_at_least_one_each_suit 6 = 1260 / 4096 :=
sorry

end probability_six_draws_one_each_suit_l289_289635


namespace point_distance_from_origin_l289_289107

theorem point_distance_from_origin (x y m : ℝ) (h1 : |y| = 15) (h2 : (x - 2)^2 + (y - 7)^2 = 169) (h3 : x > 2) :
  m = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end point_distance_from_origin_l289_289107


namespace digits_right_of_decimal_l289_289625

theorem digits_right_of_decimal : 
  ∃ n : ℕ, (3^6 : ℚ) / ((6^4 : ℚ) * 625) = 9 * 10^(-4 : ℤ) ∧ n = 4 := 
by 
  sorry

end digits_right_of_decimal_l289_289625


namespace square_of_integer_sequence_l289_289359

theorem square_of_integer_sequence (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 0) (h3 : a 3 = 1)
  (h_rec : ∀ n ≥ 1, a (n + 3) = ((n^2 + n + 1) * (n + 1) / n) * a (n + 2)
                     + (n^2 + n + 1) * a (n + 1) - ((n + 1) / n) * a n) :
  ∀ n ≥ 1, ∃ k : ℤ, a n = k^2 :=
begin
  sorry
end

end square_of_integer_sequence_l289_289359


namespace eventually_periodic_sequence_l289_289700

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h1 : ∀ n m : ℕ, 0 < n → 0 < m → a (n + 2 * m) ∣ (a n + a (n + m)))
  : ∃ N d : ℕ, 0 < N ∧ 0 < d ∧ ∀ n > N, a n = a (n + d) :=
sorry

end eventually_periodic_sequence_l289_289700


namespace complex_square_l289_289195

theorem complex_square (z : Complex) (hz : z = 1 + 2 * Complex.i) : z^2 = -3 + 4 * Complex.i :=
by {
  subst hz,
  simp,
  sorry
}

end complex_square_l289_289195


namespace num_odd_digit_palindromes_l289_289851

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

noncomputable def num_palindromes (k : ℕ) : ℕ :=
  if k = 2 then 9 else if k = 3 then 90 else 0 -- Simplified for input validation

theorem num_odd_digit_palindromes (n : ℕ) (h : n > 0) : 
  num_palindromes (2 * n + 1) = 9 * 10^n :=
sorry -- Proof omitted

end num_odd_digit_palindromes_l289_289851


namespace cost_of_18_pounds_l289_289141

-- Definitions based on the conditions
def rate (pounds : ℕ) : ℝ := (pounds / 3) * 3
def discount (cost : ℝ) : ℝ := if cost > 10 then cost * 0.10 else 0
def total_cost (pounds : ℕ) : ℝ := 
  let regular_cost := rate pounds
  regular_cost - discount regular_cost

-- Proof statement
theorem cost_of_18_pounds : total_cost 18 = 16.20 := by
  sorry

end cost_of_18_pounds_l289_289141


namespace min_photos_l289_289302

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l289_289302


namespace find_p_from_parabola_and_latus_rectum_l289_289227

-- Given definitions from the conditions
def parabola_property (p : ℝ) : Prop := ∀ y x : ℝ, y^2 = 2 * p * x
def latus_rectum_condition (p : ℝ) : Prop := ∀ x : ℝ, x = -2 → -p / 2 = -2

-- Assertion of the value of p
theorem find_p_from_parabola_and_latus_rectum :
  ∃ p : ℝ, parabola_property p ∧ latus_rectum_condition p ∧ p = 4 :=
by {
  use 4,
  split,
  -- Prove parabola property holds for p = 4 (normally would prove, skipping with sorry)
  { intros y x h, sorry },
  split,
  -- Prove latus rectum condition holds for p = 4 (normally would prove, skipping with sorry)
  { intros x h, sorry },
  -- Assert p = 4
  refl
}

end find_p_from_parabola_and_latus_rectum_l289_289227


namespace sine_of_angle_between_line_and_plane_is_correct_l289_289732

-- Define vectors and plane equation
def normal_vector : ℝ × ℝ × ℝ := (3, -5, 1)
def direction_vector : ℝ × ℝ × ℝ := (3, 1, -2)
def plane_eq (x y z : ℝ) : Prop := 3 * x - 5 * y + z - 7 = 0

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitude function
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Define the sine of the angle between a vector and a plane
def sine_angle_between_vector_and_plane (m n : ℝ × ℝ × ℝ) : ℝ :=
  (Real.abs (dot_product m n)) / (magnitude m * magnitude n)

theorem sine_of_angle_between_line_and_plane_is_correct :
  sine_angle_between_vector_and_plane direction_vector normal_vector = Real.sqrt 14 / 55 :=
by sorry

end sine_of_angle_between_line_and_plane_is_correct_l289_289732


namespace quadratic_solution_l289_289569

def q (x : ℝ) : ℝ := (17/15)*x^2 - (4/5)*x + 11/3

theorem quadratic_solution :
  q (-1) = 4 ∧ q 2 = 5 ∧ q 4 = 17 :=
by
  unfold q
  split
  { show (17/15)*(-1)^2 - (4/5)*(-1) + 11/3 = 4
    calc
      (17/15)*(-1)^2 - (4/5)*(-1) + 11/3
      = 17/15 + 4/5 + 11/3                 : by norm_num
      ... = 4                              : by norm_num }
  split
  { show (17/15)*(2)^2 - (4/5)*(2) + 11/3 = 5
    calc
      (17/15)*(2)^2 - (4/5)*(2) + 11/3
      = (17/15) * 4 - (4/5) * 2 + 11/3     : by norm_num
      ... = 5                              : by norm_num }
  { show (17/15)*(4)^2 - (4/5)*(4) + 11/3 = 17
    calc
      (17/15)*(4)^2 - (4/5)*(4) + 11/3
      = (17/15) * 16 - (4/5) * 4 + 11/3    : by norm_num
      ... = 17                             : by norm_num }

end quadratic_solution_l289_289569


namespace cos_sequence_negative_l289_289561

theorem cos_sequence_negative (α : ℝ) :
  (∀ n : ℕ, cos (2^n * α) < 0) → 
  (∃ k : ℤ, α = (2 * k:ℝ) * π + (2/3) * π ∨ α = (2 * k:ℝ) * π - (2/3) * π) :=
by
  intros h
  -- Sorry is used here to indicate that the proof is not provided.
  sorry

end cos_sequence_negative_l289_289561


namespace find_base_of_log_function_l289_289987

theorem find_base_of_log_function :
  (∀ (a : ℝ), (a > 1 → (log a 9 - log a 3 = 1 → a = 3)) ∧ (0 < a ∧ a < 1 → (log a 3 - log a 9 = 1 → a = 1 / 3))) :=
by
  intros a
  split
  { intros ha hlog
    sorry },
  { intros ha hlog
    sorry }

end find_base_of_log_function_l289_289987


namespace polynomial_divisibility_l289_289247

theorem polynomial_divisibility (P : ℤ[x]) (k a : ℤ) (h : P.eval a % k = 0) :
    ∀ t : ℤ, P.eval (a + k * t) % k = 0 :=
by sorry

end polynomial_divisibility_l289_289247


namespace minimum_photos_l289_289287

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l289_289287


namespace determine_f_1002_l289_289419

def arbitrary_function (f : ℕ → ℝ) : Prop :=
  ∀ n > 1, ∃ p : ℕ, nat.prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem determine_f_1002 (f : ℕ → ℝ)
  (h : arbitrary_function f)
  (h1001 : f 1001 = 1) :
  f 1002 = 1 :=
sorry

end determine_f_1002_l289_289419


namespace lottery_problem_l289_289262

theorem lottery_problem :
  let p : ℕ → ℚ := λ i, if i = 1 ∨ i = 3 then 1/2 else if i = 2 then 1/3 else 0,
      P : ℚ := 1/4,
      A : ℚ := (1/4 * (1/2 + 1/3 + 1/2)),

      -- Probability that host opens box 4
      P_B4 := A = 1/3,
      
      -- Conditional probabilities after host opens box 4
      P_A1_B4 := (P * p 1) / (1/3) = 3/8,
      P_A2_B4 := (P * p 2) / (1/3) = 1/4,
      P_A3_B4 := (P * p 3) / (1/3) = 3/8,

      -- Best strategy for Player A
      optimal_strategy : (P_A1_B4 > P_A2_B4 ∧ P_A3_B4 > P_A2_B4)
  in P_B4 ∧ optimal_strategy := by {
  sorry
}

end lottery_problem_l289_289262


namespace surface_area_of_circumscribed_sphere_l289_289665

theorem surface_area_of_circumscribed_sphere (SA AB BC : ℝ) 
  (h1 : SA = 1)
  (h2 : AB = 2)
  (h3 : BC = 3)
  (SA_perp_ABC : SA ⊥ ABC)
  (AB_perp_BC : AB ⊥ BC) :
  surface_area_of_circumscribed_sphere SA AB BC = 14 * real.pi := 
by 
  sorry

end surface_area_of_circumscribed_sphere_l289_289665


namespace min_shirts_to_save_l289_289876

def AcmeCost (x : ℕ) : ℕ := 40 + 10 * x
def BetaCost (x : ℕ) : ℕ := 15 * x
def GammaCost (x : ℕ) : ℕ := 20 + 12 * x

theorem min_shirts_to_save : ∃ x : ℕ, x = 11 ∧ GammaCost x < AcmeCost x ∧ GammaCost x < BetaCost x := by
  use 11
  rw [GammaCost, AcmeCost, BetaCost]
  norm_num
  split; linarith
  sorry

end min_shirts_to_save_l289_289876


namespace transportation_cost_l289_289009

theorem transportation_cost (weight_g : ℕ) (cost_per_kg : ℕ) (discount: ℝ) (weight_threshold_g : ℕ) : 
(weight_g = 400) → 
(cost_per_kg = 25000) → 
(discount = 0.1) → 
(weight_threshold_g = 500) → 
(weight_g < weight_threshold_g) → 
(weight_g / 1000.0) * cost_per_kg * (1 - discount) = 9000 := 
by 
  intros h_weight h_cost h_discount h_threshold h_cond
  rw [h_weight, h_cost, h_discount, h_threshold, h_cond]
  norm_num
  sorry

end transportation_cost_l289_289009


namespace initial_percentage_increase_l289_289880

theorem initial_percentage_increase (E P : ℝ) 
  (h1 : E * (1 + P / 100) = 678)
  (h2 : E * 1.15 = 683.95) :
  P ≈ 14 :=
by sorry

end initial_percentage_increase_l289_289880


namespace sum_x1_x2_l289_289212

noncomputable def x1 (a : ℝ) (x : ℝ) := log a x + x - 2016 = 0

noncomputable def x2 (a : ℝ) (x : ℝ) := a^x + x - 2016 = 0

theorem sum_x1_x2 (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (hx1 : log a x1 + x1 - 2016 = 0)
  (hx2 : a^x2 + x2 - 2016 = 0) : 
  x1 + x2 = 2016 := 
sorry

end sum_x1_x2_l289_289212


namespace find_number_l289_289807

theorem find_number
  (P : ℝ) (R : ℝ) (hP : P = 0.0002) (hR : R = 2.4712) :
  (12356 * P = R) := by
  sorry

end find_number_l289_289807


namespace simplify_cn_dn_l289_289181

def c_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), 1 / Nat.choose n k

def d_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), (k ^ 2 + 3 * k) / Nat.choose n k

theorem simplify_cn_dn (n : ℕ) (h : 0 < n) : c_n n / d_n n = 2 / (n * (n + 6)) :=
by
  sorry

end simplify_cn_dn_l289_289181


namespace max_value_max_value_achieved_l289_289358

theorem max_value (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) :
  a + 2 * real.sqrt (a * b) + 2 * real.cbrt (a * b * c) ≤ 7 :=
sorry

theorem max_value_achieved :
  a + 2 * real.sqrt (a * b) + 2 * real.cbrt (a * b * c) = 7 :=
sorry

end max_value_max_value_achieved_l289_289358


namespace counterexample_to_proposition_l289_289814

theorem counterexample_to_proposition (a b : ℝ) (ha : a = 1) (hb : b = -1) :
  a > b ∧ ¬ (1 / a < 1 / b) :=
by
  sorry

end counterexample_to_proposition_l289_289814


namespace horses_added_l289_289330

-- Define the problem parameters and conditions.
def horses_initial := 3
def water_per_horse_drinking_per_day := 5
def water_per_horse_bathing_per_day := 2
def days := 28
def total_water := 1568

-- Define the assumption based on the given problem.
def total_water_per_horse_per_day := water_per_horse_drinking_per_day + water_per_horse_bathing_per_day
def total_water_initial_horses := horses_initial * total_water_per_horse_per_day * days
def water_for_new_horses := total_water - total_water_initial_horses
def daily_water_consumption_new_horses := water_for_new_horses / days
def number_of_new_horses := daily_water_consumption_new_horses / total_water_per_horse_per_day

-- The theorem to prove number of horses added.
theorem horses_added : number_of_new_horses = 5 := 
  by {
    -- This is where you would put the proof steps.
    sorry -- skipping the proof for now
  }

end horses_added_l289_289330


namespace sum_of_paintable_integers_l289_289235
open Nat

def isPaintable (h t u : ℕ) : Prop :=
  (h > 1) ∧ (t > 1) ∧ (u > 1) ∧
  let lcm_ht : ℕ := lcm h t
  let lcm_htu : ℕ := lcm lcm_ht u
  ∀ n ∈ range lcm_htu, (n % h = 0) ∨ (n % t = 3) ∨ (n % u = 2)

theorem sum_of_paintable_integers : ∑ (h t u : ℕ) in { (h, t, u) | isPaintable h t u }, 100 * h + 10 * t + u = 465 :=
by
  sorry

end sum_of_paintable_integers_l289_289235


namespace can_surely_win_l289_289069

theorem can_surely_win (n : ℕ) (h : n > 1) (score_diff : ℕ) (h_lead : score_diff ≥ 2^(n-1)) : 
  ∃ strategy : (list (fin n) → ℕ) → (fin n → fin n → ℕ) → list (fin n) → fin n, -- Strategy function
    ∀ guesses : list (fin n), ∀ correct_answers : list (fin n), 
    strategy guesses correct_answers ██ -- Definition of winning condition
sorry

end can_surely_win_l289_289069


namespace min_value_of_expression_l289_289684

-- Definitions of the norms of the vectors
variables (u v w : ℝ^3)
hypothesis h1 : ‖u‖ = 2
hypothesis h2 : ‖v‖ = 3
hypothesis h3 : ‖w‖ = 4

theorem min_value_of_expression :
  ‖u - 3 • v‖^2 + ‖v - 3 • w‖^2 + ‖w - 3 • u‖^2 = 464 :=
by sorry

end min_value_of_expression_l289_289684


namespace triangle_area_l289_289475

theorem triangle_area :
  let (x1, y1, x2, y2, x3, y3) := (3, 0, 6, 3, 6, -3)
  let area := 1 / 2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|
  area = 9 :=
by
  let x1 := 3
  let y1 := 0
  let x2 := 6
  let y2 := 3
  let x3 := 6
  let y3 := -3
  let area := 1 / 2 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  have h : area = 1 / 2 * abs (3 * (3 - (-3)) + 6 * ((-3) - 0) + 6 * (0 - 3)) := by
    sorry
  have h2 : 1 / 2 * abs (3 * 6 - 18 - 18) = 1 / 2 * abs (-18) := by
    sorry
  have h3 : 1 / 2 * abs (-18) = 1 / 2 * 18 := by
    sorry
  have h4 : 1 / 2 * 18 = 9 := by
    sorry
  exact h.trans (h2.trans (h3.trans h4))

end triangle_area_l289_289475


namespace empty_is_proper_subset_of_singleton_zero_l289_289082

theorem empty_is_proper_subset_of_singleton_zero : ∅ ⊂ ({0} : Set Nat) :=
sorry

end empty_is_proper_subset_of_singleton_zero_l289_289082


namespace kafelnikov_served_in_first_game_l289_289068

theorem kafelnikov_served_in_first_game (games : ℕ) (kafelnikov_wins : ℕ) (becker_wins : ℕ)
  (server_victories : ℕ) (x y : ℕ) 
  (h1 : kafelnikov_wins = 6)
  (h2 : becker_wins = 3)
  (h3 : server_victories = 5)
  (h4 : games = 9)
  (h5 : kafelnikov_wins + becker_wins = games)
  (h6 : (5 - x) + y = 5) 
  (h7 : x + y = 6):
  x = 3 :=
by
  sorry

end kafelnikov_served_in_first_game_l289_289068


namespace probability_on_between_correct_l289_289482

noncomputable def probability_on_between (t : ℝ) : ℝ :=
  if 4 < t ∧ t < 5 then 7/20 else 0

theorem probability_on_between_correct :
    ∀ t, probability_on_between t = if 4 < t ∧ t < 5 then 7/20 else 0 := by
  intro t
  sorry

end probability_on_between_correct_l289_289482


namespace probability_X_eq_Y_l289_289881

theorem probability_X_eq_Y
  (x y : ℝ)
  (h1 : -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi)
  (h2 : -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi)
  (h3 : Real.cos (Real.cos x) = Real.cos (Real.cos y)) :
  (∃ N : ℕ, N = 100 ∧ ∃ M : ℕ, M = 11 ∧ M / N = (11 : ℝ) / 100) :=
by sorry

end probability_X_eq_Y_l289_289881


namespace largest_number_with_unique_digits_summing_to_17_is_98_l289_289800

theorem largest_number_with_unique_digits_summing_to_17_is_98 :
  ∀ n : ℕ, (all_different n) → (digits_sum n = 17) → n = 98 :=
by
  sorry

def all_different (n : ℕ) : Prop :=
  let digits := n.to_digits
  digits.nodup

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.to_digits
  digits.sum

end largest_number_with_unique_digits_summing_to_17_is_98_l289_289800


namespace car_speed_time_relation_l289_289834

noncomputable theory

-- Definitions
def original_speed (x : ℝ) : ℝ := x
def original_time (y : ℝ) : ℝ := y

def new_speed (x : ℝ) (a : ℝ) : ℝ := x * (1 + a / 100)
def new_time (y : ℝ) (b : ℝ) : ℝ := y * (1 - b / 100)

-- Statement of the problem
theorem car_speed_time_relation (x y a b : ℝ) (h : original_speed x * original_time y = new_speed x a * new_time y b) :
  b = 100 * a / (100 + a) :=
sorry

end car_speed_time_relation_l289_289834


namespace largest_number_with_unique_digits_summing_to_17_l289_289793

theorem largest_number_with_unique_digits_summing_to_17 : ∃ n : ℕ, 
  (∀ i j : ℕ, i ≠ j → (List.toDigits n).nth i ≠ (List.toDigits n).nth j) ∧
  (List.sum (List.filterMap (fun x => if x ∈ (List.toDigits n) then some (x : ℕ) else none) (finset.range 10))) = 17 ∧
  n = 7543210 :=
begin
  sorry
end

end largest_number_with_unique_digits_summing_to_17_l289_289793


namespace consecutive_interior_angles_are_supplementary_imply_angle_bisectors_perpendicular_l289_289469

theorem consecutive_interior_angles_are_supplementary_imply_angle_bisectors_perpendicular
    (l1 l2 l3 : line)
    (a1 a2 : angle)
    (h1 : consecutive_interior_angles l1 l2 l3 a1 a2)
    (h2 : supplementary a1 a2) :
  perpendicular (angle_bisector a1) (angle_bisector a2) :=
sorry

end consecutive_interior_angles_are_supplementary_imply_angle_bisectors_perpendicular_l289_289469


namespace polynomial_modulus_bound_l289_289617

theorem polynomial_modulus_bound {n : ℕ} (a : Fin n → ℂ) (a0 : ℂ) :
    ∃ (z : ℂ), |z| ≤ 1 ∧ |(z^n + ∑ i in Finset.range n, a i * z^i)| ≥ 1 + |a0| := by
    sorry

end polynomial_modulus_bound_l289_289617


namespace arithmetic_sequence_sum_zero_l289_289961

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n, a k

theorem arithmetic_sequence_sum_zero {a : ℕ → ℝ}
  (h_arith : is_arithmetic_sequence a)
  (h_cond : a 5 + a 11 = 3 * a 10) :
  S a 27 = 0 :=
sorry

end arithmetic_sequence_sum_zero_l289_289961


namespace trig_identity_l289_289354

theorem trig_identity (θ : ℝ) (h : cos (2 * θ) = 1 / 5) : sin θ ^ 6 + cos θ ^ 6 = 7 / 25 :=
by
  sorry

end trig_identity_l289_289354


namespace sum_of_30th_set_l289_289167

theorem sum_of_30th_set : 
  let start := 1
  let n := 30
  let first_element := start + (n-1) * n / 2 + 1
  let last_element := first_element + n - 1
  S_n (n : ℕ) = 15 * (first_element + last_element) 
  in S_30 = 13515 :=
by
  sorry

end sum_of_30th_set_l289_289167


namespace race_distance_l289_289649

theorem race_distance (dA dB dC : ℝ) (h1 : dA = 1000) (h2 : dB = 900) (h3 : dB = 800) (h4 : dC = 700) (d : ℝ) (h5 : d = dA + 127.5) :
  d = 600 :=
sorry

end race_distance_l289_289649


namespace max_gcd_lcm_condition_l289_289718

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l289_289718


namespace freshmen_and_sophomores_without_pets_l289_289038

theorem freshmen_and_sophomores_without_pets :
  let total_students := 400
  let freshmen_sophomores_fraction := 0.50
  let pet_owners_fraction := 1 / 5
  let freshmen_sophomores := total_students * freshmen_sophomores_fraction
  let pet_owners := freshmen_sophomores * pet_owners_fraction
  let non_pet_owners := freshmen_sophomores - pet_owners
  non_pet_owners = 160 :=
by
  Sorry

end freshmen_and_sophomores_without_pets_l289_289038


namespace general_form_of_aₙ_find_m_T_l289_289772

-- Definitions

def has_property_aₙ (a : ℕ → ℕ ) : Prop :=
  ∀ n : ℕ, 0 < n → (a 1 + ∑ i in Finset.range n, (1 / (i + 1)) * a (i + 2)) = a n + 1

def initial_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i + 1) * a (n - i)

def b (S: ℕ → ℕ) (n: ℕ) : ℕ :=
  1 / (3 * S n)

def T (b: ℕ → ℕ) (n: ℕ) : ℕ :=
  ∑ i in Finset.range n, b i

-- Proof goals

theorem general_form_of_aₙ (a : ℕ → ℕ) (n : ℕ) (h₁ : has_property_aₙ a) (h₂ : initial_condition a) : ∀ n, a n = n := 
  sorry

theorem find_m_T (a : ℕ → ℕ) (S: ℕ → ℕ) (b: ℕ → ℕ) (T : ℕ → ℕ) (h_a : ∀ n, a n = n) : ∃ m, ∀ n, T n < m / 2024 :=
  sorry

end general_form_of_aₙ_find_m_T_l289_289772


namespace minimum_photos_l289_289280

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l289_289280


namespace chuck_play_area_l289_289146

-- Definitions
def radius : ℝ := 4
def shed_width : ℝ := 4
def shed_height : ℝ := 3
def arc_fraction : ℝ := 3 / 4
def sector_fraction : ℝ := 1 / 4
def sector_radius : ℝ := 1

-- Area calculation
def full_circle_area (r : ℝ) : ℝ := π * r^2
def arc_area (r : ℝ) (fraction : ℝ) : ℝ := fraction * full_circle_area r
def sector_area (r : ℝ) (fraction : ℝ) : ℝ := fraction * full_circle_area r

-- Total playable area
def total_play_area : ℝ :=
  arc_area radius arc_fraction + sector_area sector_radius sector_fraction

theorem chuck_play_area : total_play_area = (49 / 4) * π := by
  sorry

end chuck_play_area_l289_289146


namespace vampire_daily_blood_suction_l289_289871

-- Conditions from the problem
def vampire_bl_need_per_week : ℕ := 7  -- gallons of blood per week
def blood_per_person_in_pints : ℕ := 2  -- pints of blood per person
def pints_per_gallon : ℕ := 8            -- pints in 1 gallon

-- Theorem statement to prove
theorem vampire_daily_blood_suction :
  let daily_requirement_in_gallons : ℕ := vampire_bl_need_per_week / 7   -- gallons per day
  let daily_requirement_in_pints : ℕ := daily_requirement_in_gallons * pints_per_gallon
  let num_people_needed_per_day : ℕ := daily_requirement_in_pints / blood_per_person_in_pints
  num_people_needed_per_day = 4 :=
by
  sorry

end vampire_daily_blood_suction_l289_289871


namespace log_base_computation_l289_289362

open Real

theorem log_base_computation (u v w : ℝ) (hu : 1 < u) (hv : 1 < v) (hw : 1 < w)
  (h1 : logBase u (v * w) + logBase v w = 5)
  (h2 : logBase v u + logBase w v = 3) :
  logBase w u = 4 / 5 :=
sorry

end log_base_computation_l289_289362


namespace probability_of_winning_pair_l289_289497

-- Define the deck and conditions
structure Card where
  color : String
  number : Nat

def deck : List Card :=
  [ {color := "blue", number := 1}, {color := "blue", number := 2},
    {color := "blue", number := 3}, {color := "blue", number := 4},
    {color := "yellow", number := 1}, {color := "yellow", number := 2},
    {color := "yellow", number := 3}, {color := "yellow", number := 4} ]

def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.number = c2.number

-- Define the proof problem
theorem probability_of_winning_pair : 
  (let total_ways := Nat.choose 8 2 in
   let same_number_ways := 4 in
   let same_color_ways := 2 * (Nat.choose 4 2) in
   let favorable_ways := same_number_ways + same_color_ways in
   (favorable_ways: ℚ) / (total_ways: ℚ)) = 4 / 7 := by
  sorry

end probability_of_winning_pair_l289_289497


namespace positive_rationals_once_in_seq_q_l289_289154

noncomputable def seq_n : ℕ → ℕ 
| 0 := 1
| 1 := 1
| (2 * k) := seq_n k + seq_n (k - 1)
| (2 * k + 1) := seq_n k

noncomputable def seq_q (k : ℕ) (h : k > 0) : ℚ := seq_n k / seq_n (k - 1)

theorem positive_rationals_once_in_seq_q :
  ∀ (q : ℚ) (h : 0 < q), ∃! (k : ℕ) (h : k > 0), seq_q k h = q :=
sorry

end positive_rationals_once_in_seq_q_l289_289154


namespace picnic_recyclable_collected_l289_289044

theorem picnic_recyclable_collected :
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  soda_drinkers + sparkling_water_drinkers + juice_consumed = 115 :=
by
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  show soda_drinkers + sparkling_water_drinkers + juice_consumed = 115
  sorry

end picnic_recyclable_collected_l289_289044


namespace incorrect_independence_test_conclusion_l289_289813

-- Definitions for each condition
def independence_test_principle_of_small_probability (A : Prop) : Prop :=
A  -- Statement A: The independence test is based on the principle of small probability.

def independence_test_conclusion_variability (C : Prop) : Prop :=
C  -- Statement C: Different samples may lead to different conclusions in the independence test.

def independence_test_not_the_only_method (D : Prop) : Prop :=
D  -- Statement D: The independence test is not the only method to determine whether two categorical variables are related.

-- Incorrect statement B
def independence_test_conclusion_always_correct (B : Prop) : Prop :=
B  -- Statement B: The conclusion drawn from the independence test is always correct.

-- Prove that statement B is incorrect given conditions A, C, and D
theorem incorrect_independence_test_conclusion (A B C D : Prop) 
  (hA : independence_test_principle_of_small_probability A)
  (hC : independence_test_conclusion_variability C)
  (hD : independence_test_not_the_only_method D) :
  ¬ independence_test_conclusion_always_correct B :=
sorry

end incorrect_independence_test_conclusion_l289_289813


namespace general_term_of_arithmetic_seq_sum_of_seq_reciprocal_l289_289200

-- Definitions and conditions

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a₀, ∀ n, a n = a₀ + n * d

def geometric_seq (a b c : ℝ) : Prop :=
  a * c = b * b

def first_term_of_seq (a : ℕ → ℝ) : ℝ :=
  a 1

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

def sum_of_seq_reciprocal (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in (finset.range n).image (λ i, i+1), 1 / S i

noncomputable def Sn (n : ℕ) : ℝ :=
  n * (n + 2)

noncomputable def Tn (n : ℕ) : ℝ :=
  3/4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))

-- Statements
theorem general_term_of_arithmetic_seq
  (a : ℕ → ℝ) (d : ℝ) (hnzero : d ≠ 0)
  (h1 : a 1 = 3)
  (h_geometric_seq : geometric_seq (a 1) (a 4) (a 13)) :
  ∀ n, a n = 2 * n + 1 :=
sorry

theorem sum_of_seq_reciprocal
  (a : ℕ → ℝ) (d : ℝ)
  (hnzero : d ≠ 0)
  (h1 : a 1 = 3)
  (h_geometric_seq : geometric_seq (a 1) (a 4) (a 13))
  (h_sn : ∀ n, sum_of_first_n_terms a n = Sn n) :
  ∀ n, sum_of_seq_reciprocal Sn n = Tn n :=
sorry

end general_term_of_arithmetic_seq_sum_of_seq_reciprocal_l289_289200


namespace set_M_listed_correctly_l289_289027

theorem set_M_listed_correctly :
  {a : ℕ+ | ∃ (n : ℤ), 4 = n * (1 - a)} = {2, 3, 4} := by
sorry

end set_M_listed_correctly_l289_289027


namespace problem_statement_l289_289534

theorem problem_statement : 
  - (1 : ℝ) ^ 2023 + real.sqrt ((-2)^2) - |2 - real.sqrt 3| + real.cbrt 27 = 2 + real.sqrt 3 := by
  sorry

end problem_statement_l289_289534


namespace apples_pie_calculation_l289_289670

-- Defining the conditions
def total_apples : ℕ := 34
def non_ripe_apples : ℕ := 6
def apples_per_pie : ℕ := 4 

-- Stating the problem
theorem apples_pie_calculation : (total_apples - non_ripe_apples) / apples_per_pie = 7 := by
  -- Proof would go here. For the structure of the task, we use sorry.
  sorry

end apples_pie_calculation_l289_289670


namespace intersection_point_l289_289157

theorem intersection_point : 
  ∃ (x y : ℚ), y = - (5/3 : ℚ) * x ∧ y + 3 = 15 * x - 6 ∧ x = 27 / 50 ∧ y = - 9 / 10 := 
by
  sorry

end intersection_point_l289_289157


namespace count_arrangements_l289_289386

theorem count_arrangements :
  ∃ (count : ℕ), count = 3960 ∧
    let grid := Matrix (Fin 4) (Fin 4) (Option Char) in
    let letters := ['a', 'a', 'b', 'b'] in
    ∀ (placements : List (Fin 4 × Fin 4)),
      placements.length = 4 →
      (∀ i, grid (placements.nthLe i (by sorry)).1 (placements.nthLe i (by sorry)).2 = some letters.nthLe i (by sorry)) →
      (∀ i j, i ≠ j → (placements.nthLe i (by sorry)).1 ≠ (placements.nthLe j (by sorry)).1 ∧ (placements.nthLe i (by sorry)).2 ≠ (placements.nthLe j (by sorry)).2) →
      List.length placements = 3960 := sorry

end count_arrangements_l289_289386


namespace PersonX_job_completed_time_l289_289152

-- Definitions for conditions
def Dan_job_time := 15 -- hours
def PersonX_job_time (x : ℝ) := x -- hours
def Dan_work_time := 3 -- hours
def PersonX_remaining_work_time := 8 -- hours

-- Given Dan's and Person X's work time, prove Person X's job completion time
theorem PersonX_job_completed_time (x : ℝ) (h1 : Dan_job_time > 0)
    (h2 : PersonX_job_time x > 0)
    (h3 : Dan_work_time > 0)
    (h4 : PersonX_remaining_work_time * (1 - Dan_work_time / Dan_job_time) = 1 / x * 8) :
    x = 10 :=
  sorry

end PersonX_job_completed_time_l289_289152


namespace number_of_players_taking_biology_l289_289887

-- Definitions from the conditions
def total_players : ℕ := 15
def players_taking_physics : ℕ := 8
def players_taking_chemistry : ℕ := 6
def players_taking_all_three : ℕ := 3
def players_taking_physics_chemistry : ℕ := 4

-- Theorem statement to prove the number of players taking biology
theorem number_of_players_taking_biology :
  ∃ (players_taking_biology : ℕ),
    players_taking_biology = total_players
    - (players_taking_physics - players_taking_physics_chemistry)
    - (players_taking_chemistry - players_taking_physics_chemistry) :=
begin
  use 9,
  sorry
end

end number_of_players_taking_biology_l289_289887


namespace average_minutes_heard_is_41_l289_289260

def totalAudience : ℕ := 200
def sessionTime : ℕ := 90
def fullListenPercentage : ℝ := 0.15
def missEntireTalkPercentage : ℝ := 0.08
def quarterListenPercentage : ℝ := 0.40
def halfListenPercentage : ℝ := 0.60

def fullListeners : ℕ := (fullListenPercentage * totalAudience).toNat
def missListeners : ℕ := (missEntireTalkPercentage * totalAudience).toNat
def remainingAudience : ℕ := totalAudience - fullListeners - missListeners
def quarterListeners : ℕ := (quarterListenPercentage * remainingAudience).toNat
def halfListeners : ℕ := (halfListenPercentage * remainingAudience).toNat

def fullListenMinutes : ℕ := fullListeners * sessionTime
def missListenMinutes : ℕ := missListeners * 0
def quarterListenMinutes : ℕ := quarterListeners * (sessionTime / 4)
def halfListenMinutes : ℕ := halfListeners * (sessionTime / 2)

def totalMinutesHeard : ℕ := fullListenMinutes + missListenMinutes + quarterListenMinutes + halfListenMinutes
def averageMinutesHeard : ℝ := totalMinutesHeard.toReal / totalAudience.toReal

theorem average_minutes_heard_is_41 : averageMinutesHeard = 41 := by
  sorry

end average_minutes_heard_is_41_l289_289260


namespace line_intersects_semicircle_twice_l289_289959

theorem line_intersects_semicircle_twice (k : ℝ) :
  ((k > -3/2 ∧ k < -sqrt 5 / 2) ∨ (k > sqrt 5 / 2 ∧ k ≤ 3/2)) ↔ 
  ∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 4 ∧ y ≥ 2 → 
  y = k * (x - 1) + 5 →
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∧ y₁ ≠ y₂) ∧ 
    (x₁ - 1) ^ 2 + (y₁ - 2) ^ 2 = 4 ∧ y₁ ≥ 2 ∧ y₁ = k * (x₁ - 1) + 5 ∧
    (x₂ - 1) ^ 2 + (y₂ - 2) ^ 2 = 4 ∧ y₂ ≥ 2 ∧ y₂ = k * (x₂ - 1) + 5) :=
sorry

end line_intersects_semicircle_twice_l289_289959


namespace part1_part2_l289_289578
noncomputable theory

def f (x a : ℝ) : ℝ := |2 * x - a| - |x - 2|

theorem part1 (x : ℝ) : (f x (-2) ≤ 4) ↔ (x ≥ -4 ∧ x ≤ 4) :=
sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 3 * a^2 - 3 * |2 - x|) ↔ (a ≥ -1 ∧ a ≤ 4/3) :=
sorry

end part1_part2_l289_289578


namespace production_days_l289_289573

-- Definitions of the conditions
variables (n : ℕ) (P : ℕ)
variable (H1 : P = n * 50)
variable (H2 : (P + 60) / (n + 1) = 55)

-- Theorem to prove that n = 1 given the conditions
theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 60) / (n + 1) = 55) : n = 1 :=
by
  sorry

end production_days_l289_289573


namespace tan_alpha_value_l289_289207

variable (α : Real)

theorem tan_alpha_value (h : (sin α - 2 * cos α) / (3 * sin α + 5 * cos α) = -5) : tan α = -23 / 16 := 
sorry

end tan_alpha_value_l289_289207


namespace cloud_computing_market_scale_in_2025_is_correct_l289_289540

def year_codes : List ℕ := [1, 2, 3, 4, 5]

def market_scales : List ℝ := [7.4, 11, 20, 36.6, 66.7]

def z_values : List ℝ := [2, 2.4, 3, 3.6, 4]

def empirical_regression (x : ℝ) (a : ℝ) : ℝ := 0.52 * x + a

noncomputable def estimated_market_scale_in_2025 (a : ℝ) : ℝ := 
  let x_2025 := 8
  let z_2025 := empirical_regression x_2025 a
  real.exp z_2025

theorem cloud_computing_market_scale_in_2025_is_correct :
  (let avg_x := (year_codes.foldr (· + ·) 0) / year_codes.length
   let avg_z := (z_values.foldr (· + ·) 0) / z_values.length
   let a := avg_z - 0.52 * avg_x
   estimated_market_scale_in_2025 a = real.exp 5.6) :=
by
  let avg_x := (year_codes.foldr (· + ·) 0) / year_codes.length
  let avg_z := (z_values.foldr (· + ·) 0) / z_values.length
  let a := avg_z - 0.52 * avg_x
  show estimated_market_scale_in_2025 a = real.exp 5.6
  sorry

end cloud_computing_market_scale_in_2025_is_correct_l289_289540


namespace game_ends_with_Chris_missing_l289_289539

-- Definitions and conditions
def initial_distance : Nat := 1
def distance_increment : Nat := 1
def throws_per_distance : Nat := 2
def total_throws : Nat := 29

-- Proof problem: Prove the distance and identify who misses the ball on the 29th throw
theorem game_ends_with_Chris_missing :
  let final_distance := initial_distance + distance_increment * ((total_throws - 1) / throws_per_distance)
  final_distance = 15 ∧ total_throws % 2 = 1 :=
by
  let final_distance := initial_distance + distance_increment * ((total_throws - 1) / throws_per_distance)
  have : final_distance = 15 := sorry
  have : total_throws % 2 = 1 := sorry
  exact ⟨this, this⟩

end game_ends_with_Chris_missing_l289_289539


namespace roots_distinct_and_real_l289_289899

variables (b d : ℝ)
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_distinct_and_real (h₁ : discriminant b (-3 * Real.sqrt 5) d = 25) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by 
  sorry

end roots_distinct_and_real_l289_289899


namespace smallest_piece_to_cut_l289_289869

theorem smallest_piece_to_cut (x : ℕ) 
  (h1 : 9 - x > 0) 
  (h2 : 16 - x > 0) 
  (h3 : 18 - x > 0) :
  7 ≤ x ∧ 9 - x + 16 - x ≤ 18 - x :=
by {
  sorry
}

end smallest_piece_to_cut_l289_289869


namespace increase_in_average_age_l289_289406

variable (A : ℝ)
variable (A_increase : ℝ)
variable (orig_age_sum : ℝ)
variable (new_age_sum : ℝ)

def original_total_age (A : ℝ) := 8 * A
def new_total_age (A : ℝ) := original_total_age A - 20 - 22 + 29 + 29

theorem increase_in_average_age (A : ℝ) (orig_age_sum := original_total_age A) (new_age_sum := new_total_age A) : 
  (new_age_sum / 8) = (A + 2) := 
by
  unfold new_total_age
  unfold original_total_age
  sorry

end increase_in_average_age_l289_289406


namespace tangent_line_to_cubic_at_P_l289_289416

theorem tangent_line_to_cubic_at_P :
  ∀ (P : ℝ × ℝ), P = (2, 8) → ∃ (m b : ℝ), tangent_line P (λ x, x^3) = (λ x y, 12 * x - y - 16 = 0) :=
by
  sorry

end tangent_line_to_cubic_at_P_l289_289416


namespace unit_prices_minimize_cost_l289_289089

theorem unit_prices (x y : ℕ) (h1 : x + 2 * y = 40) (h2 : 2 * x + 3 * y = 70) :
  x = 20 ∧ y = 10 :=
by {
  sorry -- proof would go here
}

theorem minimize_cost (total_pieces : ℕ) (cost_A cost_B : ℕ) 
  (total_cost : ℕ → ℕ)
  (h3 : total_pieces = 60) 
  (h4 : ∀ m, cost_A * m + cost_B * (total_pieces - m) = total_cost m) 
  (h5 : ∀ m, cost_A * m + cost_B * (total_pieces - m) ≥ 800) 
  (h6 : ∀ m, m ≥ (total_pieces - m) / 2) :
  total_cost 20 = 800 :=
by {
  sorry -- proof would go here
}

end unit_prices_minimize_cost_l289_289089


namespace olympiad_not_possible_l289_289324

theorem olympiad_not_possible (x : ℕ) (y : ℕ) (h1 : x + y = 1000) (h2 : y = x + 43) : false := by
  sorry

end olympiad_not_possible_l289_289324


namespace sculpture_plus_base_height_l289_289896

def height_sculpture_feet : Nat := 2
def height_sculpture_inches : Nat := 10
def height_base_inches : Nat := 4

def height_sculpture_total_inches : Nat := height_sculpture_feet * 12 + height_sculpture_inches
def height_total_inches : Nat := height_sculpture_total_inches + height_base_inches

theorem sculpture_plus_base_height :
  height_total_inches = 38 := by
  sorry

end sculpture_plus_base_height_l289_289896


namespace determine_a_l289_289911

noncomputable def lucas : ℕ → ℤ
| 0 := 2
| 1 := 1
| n + 2 := lucas (n + 1) + lucas n

theorem determine_a (a b : ℤ)
    (h1 : ∀ x : ℂ, (x^2 - 2*x - 1 = 0) → (a*x^19 + b*x^18 + 1 = 0))
    (L18 : lucas 18 = 5778)
    (L17 : lucas 17 = 3571) :
  a = 3571 := by
sorry

end determine_a_l289_289911


namespace probability_no_adjacent_standing_l289_289736

theorem probability_no_adjacent_standing (n : ℕ) (h_n : n = 7) : 
  let total_sequences := (2^n),
      valid_sequences := 34,
      p := 17,
      q := 64
  in p + q = 81 :=
by 
  sorry

end probability_no_adjacent_standing_l289_289736


namespace probability_of_even_sum_l289_289252

theorem probability_of_even_sum : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19] in
  let selections := (primes.combinations 3).filter (λ s, s.sum % 2 = 0) in
  (selections.length.to_rat / primes.combinations(3).length.to_rat) = (1 : rat) := 
by 
  sorry

end probability_of_even_sum_l289_289252


namespace normal_line_equation_l289_289820

noncomputable def curve (x : ℝ) : ℝ := (x^3 + 2) / (x^3 - 2)

def x0 : ℝ := 2

def normal_line (x : ℝ) : ℝ := (3 / 4) * x + (1 / 6)

theorem normal_line_equation : 
  ∃ (m b : ℝ), (m = 3 / 4) ∧ (b = 1 / 6) ∧
  (∀ (x : ℝ), normal_line x = m * x + b) ∧
  (∀ (x : ℝ), curve x0 = (normal_line x0)) :=
begin
  sorry
end

end normal_line_equation_l289_289820


namespace find_x_l289_289686

-- Define the operation a ⊘ b = (sqrt(3a + b))^3
def oslash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 3

-- Define the condition 6 ⊘ x = 64
def condition (x : ℝ) : Prop := oslash 6 x = 64

-- The statement to prove that x = -2
theorem find_x : ∃ x : ℝ, condition x ∧ x = -2 :=
by
  sorry

end find_x_l289_289686


namespace concentric_circle_area_ratio_l289_289433

theorem concentric_circle_area_ratio (r R : ℝ) (h_ratio : (π * R^2) / (π * r^2) = 16 / 3) :
  R - r = 1.309 * r :=
by
  sorry

end concentric_circle_area_ratio_l289_289433


namespace right_triangle_angle_bisector_length_l289_289198

noncomputable def angle_bisector_length (DE EF : ℝ) : ℝ :=
  let DF := Real.sqrt (DE^2 + EF^2)
  in (DF * Real.sqrt 2) / 2

theorem right_triangle_angle_bisector_length :
  ∀ (DE EF: ℝ), (DE = 5 ∧ EF = 12) → angle_bisector_length DE EF = 13 * (Real.sqrt 2) / 2 :=
by
  intro DE EF h
  cases h with hDE hEF
  rw [hDE, hEF]
  unfold angle_bisector_length
  have DF_calc : Real.sqrt (5^2 + 12^2) = 13 := by
    calc
      Real.sqrt (5^2 + 12^2) = Real.sqrt (25 + 144)   : by rw [pow_two, pow_two]
                        ... = Real.sqrt 169         : by simp
                        ... = 13                    : by norm_num
  rw DF_calc
  norm_num
  rw [mul_div_assoc, mul_comm _ (Real.sqrt 2), ← mul_assoc (13:ℝ), ← div_eq_mul_inv]
  simp
  norm_num
  sorry

end right_triangle_angle_bisector_length_l289_289198


namespace distinct_ratios_zero_l289_289361

theorem distinct_ratios_zero (p q r : ℝ) (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) 
  (h : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 :=
sorry

end distinct_ratios_zero_l289_289361


namespace isosceles_triangle_sides_l289_289514

theorem isosceles_triangle_sides (L B: ℝ) (h_total: L = 24) (h_base: B = 6) 
(h_isosceles: ∀ (a b: ℝ), a = (L - B) / 2 ∧ a = b): 
    (L = 24) → (B = 6) → (h_isosceles 9 9) → (B, 9, 9) = (6, 9, 9) := 
by 
  sorry

end isosceles_triangle_sides_l289_289514


namespace dot_product_equation_l289_289350

-- Given vectors a and b with specified norms and angle
variables (a b : EuclideanSpace ℝ (Fin 3)) (norm_a : ‖a‖ = 5) (norm_b : ‖b‖ = 2)
          (angle_ab : Real.angleBetween a b = Real.pi / 3)

-- Prove the given statement
theorem dot_product_equation :
  (a + 2 • b) ⋅ (a - 2 • b) = 21 :=
sorry

end dot_product_equation_l289_289350


namespace polynomial_factors_l289_289248

theorem polynomial_factors (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c = (x - 3) * (x - 2)) →
  a = 1 ∧ b = -5 ∧ c = 6 :=
by
  intro h
  have eq : ∀ x : ℝ, a * x^2 + b * x + c = x^2 - 5 * x + 6 :=
    by 
      intro x
      rw [h x, (x - 3) * (x - 2), mul_assoc, mul_comm (x - 3), ←sub_eq_add_neg, ←sub_eq_add_neg]
  sorry

end polynomial_factors_l289_289248


namespace probability_jqk_3_13_l289_289184

def probability_jack_queen_king (total_cards jacks queens kings : ℕ) : ℚ :=
  (jacks + queens + kings) / total_cards

theorem probability_jqk_3_13 :
  probability_jack_queen_king 52 4 4 4 = 3 / 13 := by
  sorry

end probability_jqk_3_13_l289_289184


namespace ice_cream_cost_l289_289166

variable {x F M : ℤ}

theorem ice_cream_cost (h1 : F = x - 7) (h2 : M = x - 1) (h3 : F + M < x) : x = 7 :=
by
  sorry

end ice_cream_cost_l289_289166


namespace symmetric_line_equation_l289_289415

variables {a : ℝ} 

def line1_passes_through_fixed_point (M : ℝ × ℝ) : Prop :=
  ∃ M, M = (-3, 1) ∧ (M.fst * a + M.snd + 3*a - 1 = 0)

def symmetric_line (M : ℝ × ℝ) : Prop :=
  ∃ c, c = 12 ∧ ∀ (x y : ℝ), |2*x + 3*y - 6| / (2^2 + 3^2).sqrt = |2*x + 3*y + c| / (2^2 + 3^2).sqrt

theorem symmetric_line_equation :
  ∃ M, line1_passes_through_fixed_point M → symmetric_line M :=
sorry

end symmetric_line_equation_l289_289415


namespace smallest_positive_phi_l289_289762

noncomputable def translated_function (φ : ℝ) := λ x: ℝ, 2 * sin (2 * (x - φ) + π / 4)
noncomputable def shortened_function (φ : ℝ) := λ x: ℝ, 2 * sin (4 * x - 2 * φ + π / 4)

theorem smallest_positive_phi :
  ∀ φ : ℝ,
    (φ > 0) →
    (∀ x : ℝ, shortened_function φ x = shortened_function φ (π / 4 - x)) → 
    φ = 3 * π / 8 :=
by
  intro φ h1 h2
  sorry

end smallest_positive_phi_l289_289762


namespace quotient_of_factorial_like_l289_289909

namespace FactorialQuotient

-- Define the generalized factorial-like function
def factorial_like (n a : ℕ) : ℕ :=
  if h : n = 0 then 1
  else if n < a then 1
  else (n * factorial_like (n - a) a)

-- Formalize the problem and state the proof
theorem quotient_of_factorial_like :
  factorial_like 72 8 / factorial_like 18 2 = 4 ^ 9 :=
by {
  sorry
}

end FactorialQuotient

end quotient_of_factorial_like_l289_289909


namespace bryden_receives_10_dollars_l289_289893

-- Define the face value of one quarter
def face_value_quarter : ℝ := 0.25

-- Define the number of quarters Bryden has
def num_quarters : ℕ := 8

-- Define the multiplier for 500%
def multiplier : ℝ := 5

-- Calculate the total face value of eight quarters
def total_face_value : ℝ := num_quarters * face_value_quarter

-- Calculate the amount Bryden will receive
def amount_received : ℝ := total_face_value * multiplier

-- The proof goal: Bryden will receive 10 dollars
theorem bryden_receives_10_dollars : amount_received = 10 :=
by
  sorry

end bryden_receives_10_dollars_l289_289893


namespace imaginary_part_of_z_l289_289219

theorem imaginary_part_of_z (z : ℂ) (h : 1 + z * complex.I = z - 2 * complex.I) : z.im = 3 / 2 := 
by
  sorry

end imaginary_part_of_z_l289_289219


namespace area_of_quadrilateral_WXYZ_l289_289315

/-- In quadrilateral WXYZ, where ∠WYZ is a right angle, WY = 15, YZ = 9, XZ = 17,
WX is not a straight line with any other side, and WX is perpendicular to YZ, 
prove that the area of the quadrilateral WXYZ is 135. -/
theorem area_of_quadrilateral_WXYZ :
  ∀ (W X Y Z : Type) [MetricSpace W] [MetricSpace X] [MetricSpace Y] [MetricSpace Z],
  (is_right_angle W Y Z) ∧ (distance W Y = 15) ∧ (distance Y Z = 9) ∧ 
  (distance X Z = 17) ∧ (not_straight_line W X Y Z) ∧ 
  (perpendicular W X Y Z) → 
  area_of_quadrilateral W X Y Z = 135 := by
  sorry

end area_of_quadrilateral_WXYZ_l289_289315


namespace jesse_remaining_pages_l289_289338

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end jesse_remaining_pages_l289_289338


namespace major_axis_of_ellipse_l289_289980

theorem major_axis_of_ellipse (m : ℝ) (h : m > 0) (h_dist : ∀ x ∈ {sqrt m}, ∃ y1 y2, 
  (x, y1) ∈ {p | ∃ x y, p = (x, y) ∧ x ^ 2 / (2 * m) + y ^ 2 / m = 1}
  ∧ (x, y2) ∈ {p | ∃ x y, p = (x, y) ∧ x ^ 2 / (2 * m) + y ^ 2 / m = 1}
  ∧ abs (y1 - y2) = 2) : 
  ∃ a : ℝ, 2 * a = 4 :=
by
  sorry

end major_axis_of_ellipse_l289_289980


namespace square_area_inscribed_in_parabola_l289_289114

def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

theorem square_area_inscribed_in_parabola :
  ∃ s : ℝ, s = (-1 + Real.sqrt 5) ∧ (2 * s)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end square_area_inscribed_in_parabola_l289_289114


namespace find_BF_l289_289757

-- Define ellipse parameters
def a : ℝ := 6
def b : ℝ := 4
def c : ℝ := Real.sqrt (a^2 - b^2)

-- Define the second focus F'
def F' : ℝ × ℝ := (-c, 0)

-- Point A satisfies the ellipse equation and passes 
def A (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 16 = 1 ∧ (x + c)^2 + y^2 = 4

-- Given distance AF' = 2
axiom AF' (x y : ℝ) (h : A x y) : (x + c)^2 + y^2 = 4

-- The distance formula
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The final goal: calculate BF'
theorem find_BF' : ∃ (x₁ y₁ x₂ y₂ : ℝ), A x₁ y₁ ∧ A x₂ y₂ ∧ x₁ ≠ x₂ ∧ distance (x₂, y₂) F' = result := sorry

end find_BF_l289_289757


namespace number_of_polynomials_l289_289898

theorem number_of_polynomials (count : ℕ) : 
  (∃ (a b c d e : ℕ), 
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧ e ≤ 9 ∧ 
    -a + b - c + d - e = -20) ∧ count = 12650 := 
sorry

end number_of_polynomials_l289_289898


namespace three_digit_numbers_l289_289562

theorem three_digit_numbers (n : ℕ) (a b c : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n = 100 * a + 10 * b + c)
  (h3 : b^2 = a * c)
  (h4 : (10 * b + c) % 4 = 0) :
  n = 124 ∨ n = 248 ∨ n = 444 ∨ n = 964 ∨ n = 888 :=
sorry

end three_digit_numbers_l289_289562


namespace converse_false_inverse_false_l289_289229

-- Definitions of the conditions
def is_rhombus (Q : Type) : Prop := -- definition of a rhombus
  sorry

def is_parallelogram (Q : Type) : Prop := -- definition of a parallelogram
  sorry

variable {Q : Type}

-- Initial statement: If a quadrilateral is a rhombus, then it is a parallelogram.
axiom initial_statement : is_rhombus Q → is_parallelogram Q

-- Goals: Prove both the converse and inverse are false
theorem converse_false : ¬ ((is_parallelogram Q) → (is_rhombus Q)) :=
sorry

theorem inverse_false : ¬ (¬ (is_rhombus Q) → ¬ (is_parallelogram Q)) :=
    sorry

end converse_false_inverse_false_l289_289229


namespace dining_preference_related_to_gender_xiao_lin_meal2_probability_maximize_binomial_probability_l289_289048

-- Proof 1: Independence test for gender and cafeteria dining preference
def chi_squared_value (a b c d n : ℕ) : ℚ :=
  let χ² := (n * ((a * d - b * c)^2).toRat) / ((a + b) * (c + d) * (a + c) * (b + d))
  χ²

theorem dining_preference_related_to_gender :
  let a := 40
  let b := 10 
  let c := 20 
  let d := 30 
  let n := 100 
  let alpha := 0.005
  let x_alpha := 7.879
  χ² := chi_squared_value a b c d n
  χ² > x_alpha :=
sorry

-- Proof 2: Probability that Xiao Lin chooses Meal 2 on Friday
def P_B_meal2 (P_A : ℚ) (P_B_given_A : ℚ) (P_B_given_not_A : ℚ) : ℚ :=
  P_A * P_B_given_A + (1 - P_A) * P_B_given_not_A

theorem xiao_lin_meal2_probability :
  let P_A := (1 : ℚ) / 2
  let P_B_given_A := (1 : ℚ) / 5
  let P_B_given_not_A := (2 : ℚ) / 3
  P_B := P_B_meal2 P_A P_B_given_A P_B_given_not_A
  P_B = 13 / 30 :=
sorry

-- Proof 3: Maximizing probability in binomial distribution
def binomial_P (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  Nat.choose n k * (p^k) * ((1 - p)^(n - k))

theorem maximize_binomial_probability :
  let n := 10
  let p := 0.6
  ∃ k : ℕ, (k ≤ n) ∧ (∀ j : ℕ, j ≤ n → binomial_P n p k ≥ binomial_P n p j) ∧ k = 6 :=
sorry

end dining_preference_related_to_gender_xiao_lin_meal2_probability_maximize_binomial_probability_l289_289048


namespace problem_1_problem_2_problem_3_l289_289640

theorem problem_1 (A B C a b c : ℝ) 
  (h1 : (sqrt (3) * sin B - cos B) * (sqrt (3) * sin C - cos C) = 4 * cos B * cos C) :
    A = π / 3 :=
sorry

theorem problem_2 (A B C a b c : ℝ) 
  (h1 : a = 2) 
  (h2 : (sqrt (3) * sin B - cos B) * (sqrt (3) * sin C - cos C) = 4 * cos B * cos C) :
    area_range : 0 < triangle_area a b c ∧ triangle_area a b c ≤ sqrt 3 :=
sorry

theorem problem_3 (A B C a b c p : ℝ)
  (h1 : sin B = p * sin C)
  (h2 : ∀ θ, θ = A ∨ θ = B ∨ θ = C → 0 < θ ∧ θ < π / 2)
  (h3 : (sqrt (3) * sin B - cos B) * (sqrt (3) * sin C - cos C) = 4 * cos B * cos C) :
    0.5 < p ∧ p < 2 :=
sorry

end problem_1_problem_2_problem_3_l289_289640


namespace trig_identity_example_l289_289778

theorem trig_identity_example : sin (105 * (Real.pi / 180)) * sin (15 * (Real.pi / 180)) - cos (105 * (Real.pi / 180)) * cos (15 * (Real.pi / 180)) = 1 / 2 := 
sorry

end trig_identity_example_l289_289778


namespace center_of_symmetry_sides_of_triangle_l289_289624

-- Given conditions
def vectors_m (x : ℝ) := (2 * (Real.cos x)^2, √3)
def vectors_n (x : ℝ) := (1, Real.sin (2 * x))
def f (x : ℝ) := (vectors_m x).1 * (vectors_n x).1 + (vectors_m x).2 * (vectors_n x).2

axiom C : ℝ
axiom a b : ℝ
axiom ab_is_2sqrt3 : a * b = 2 * √3
axiom c_is_1 : ∀ (x : ℝ), f C = 3
axiom sides_inequality : a > b

-- Proof problems
theorem center_of_symmetry : ∃ k : ℤ, ∃ (x : ℝ), x = (k * Real.pi) / 2 - Real.pi / 12 ∧ f x = 1 := sorry

theorem sides_of_triangle (C : ℝ) (a b : ℝ) (c : ℝ) (h1 : f C = 3) (h2 : c = 1) (h3 : a * b = 2 * √3) (h4 : a > b) : 
a = 2 ∧ b = √3 := sorry

end center_of_symmetry_sides_of_triangle_l289_289624


namespace total_spending_in_CAD_proof_l289_289523

-- Define Jayda's spending
def Jayda_spending_stall1 : ℤ := 400
def Jayda_spending_stall2 : ℤ := 120
def Jayda_spending_stall3 : ℤ := 250

-- Define the factor by which Aitana spends more
def Aitana_factor : ℚ := 2 / 5

-- Define the sales tax rate
def sales_tax_rate : ℚ := 0.10

-- Define the exchange rate from USD to CAD
def exchange_rate : ℚ := 1.25

-- Calculate Jayda's total spending in USD before tax
def Jayda_total_spending : ℤ := Jayda_spending_stall1 + Jayda_spending_stall2 + Jayda_spending_stall3

-- Calculate Aitana's spending at each stall
def Aitana_spending_stall1 : ℚ := Jayda_spending_stall1 + (Aitana_factor * Jayda_spending_stall1)
def Aitana_spending_stall2 : ℚ := Jayda_spending_stall2 + (Aitana_factor * Jayda_spending_stall2)
def Aitana_spending_stall3 : ℚ := Jayda_spending_stall3 + (Aitana_factor * Jayda_spending_stall3)

-- Calculate Aitana's total spending in USD before tax
def Aitana_total_spending : ℚ := Aitana_spending_stall1 + Aitana_spending_stall2 + Aitana_spending_stall3

-- Calculate the combined total spending in USD before tax
def combined_total_spending_before_tax : ℚ := Jayda_total_spending + Aitana_total_spending

-- Calculate the sales tax amount
def sales_tax : ℚ := sales_tax_rate * combined_total_spending_before_tax

-- Calculate the total spending including sales tax
def total_spending_including_tax : ℚ := combined_total_spending_before_tax + sales_tax

-- Convert the total spending to Canadian dollars
def total_spending_in_CAD : ℚ := total_spending_including_tax * exchange_rate

-- The theorem to be proven
theorem total_spending_in_CAD_proof : total_spending_in_CAD = 2541 := sorry

end total_spending_in_CAD_proof_l289_289523


namespace Jesse_pages_left_to_read_l289_289336

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end Jesse_pages_left_to_read_l289_289336


namespace last_card_in_box_l289_289779

-- Define the zigzag pattern
def card_position (n : Nat) : Nat :=
  let cycle_pos := n % 12
  if cycle_pos = 0 then
    12
  else
    cycle_pos

def box_for_card (pos : Nat) : Nat :=
  if pos ≤ 7 then
    pos
  else
    14 - pos

theorem last_card_in_box : box_for_card (card_position 2015) = 3 := by
  sorry

end last_card_in_box_l289_289779


namespace part_1_part_2_l289_289986

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part_1 (a : ℝ) (h : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 :=
sorry

theorem part_2 (a : ℝ) (h : a = 2) : ∀ m, (∀ x, f (3 * x) a + f (x + 3) a ≥ m) ↔ m ≤ 5 / 3 :=
sorry

end part_1_part_2_l289_289986


namespace consecutive_days_sum_l289_289250

theorem consecutive_days_sum (x : ℕ) (h : 3 * x + 3 = 33) : x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 :=
by {
  sorry
}

end consecutive_days_sum_l289_289250


namespace spherical_to_rectangular_l289_289904
noncomputable theory

-- Definitions for the conditions
def rho : ℝ := 3
def theta : ℝ := π / 2
def phi : ℝ := theta / 4

-- Prove the conversion to rectangular coordinates
theorem spherical_to_rectangular :
  let x := rho * sin(phi) * cos(theta),
      y := rho * sin(phi) * sin(theta),
      z := rho * cos(phi) in
  (x, y, z) = (0, (3 * sqrt (2 - sqrt 2)) / 2, (3 * sqrt (2 + sqrt 2)) / 2) :=
by
  sorry

end spherical_to_rectangular_l289_289904


namespace sum_of_exponents_of_1469_l289_289444

theorem sum_of_exponents_of_1469 :
  ∃ (r : ℕ) (m : fin r → ℕ) (b : fin r → ℤ), (∀ i j, i ≠ j → m i > m j) ∧ (∀ k, b k = 1 ∨ b k = -1) ∧ (finset.univ.sum (λ i, b i * 3 ^ (m i)) = 1469) → 
  finset.univ.sum m = 22 :=
sorry

end sum_of_exponents_of_1469_l289_289444


namespace nate_pages_left_to_read_l289_289380

-- Define the constants and conditions
def total_pages : ℕ := 400
def percentage_read : ℕ := 20

-- Calculate the number of pages already read
def pages_read := total_pages * percentage_read / 100

-- Calculate the number of pages left
def pages_left := total_pages - pages_read

-- Statement to prove
theorem nate_pages_left_to_read : pages_left = 320 :=
by {
  unfold pages_read,
  unfold pages_left,
  simp,
  sorry -- The proof will be filled in based on the calculations in the solution.
}

end nate_pages_left_to_read_l289_289380


namespace tangent_line_at_1_monotonicity_intervals_log_inequality_l289_289609

section PartI
variables {x : ℝ}
def f (x : ℝ) := log (1 + x) - x

theorem tangent_line_at_1 : (1 + 2 * (log 2 - 1) = 2 * log 2 - 1) :=
by sorry
end PartI

section PartII
variables (k : ℝ) (x : ℝ) (h : k ≠ 1)
def f_monotonicity (x : ℝ) := log (1 + x) - x + k / 2 * x ^ 2

theorem monotonicity_intervals :
  if k = 0 then (∀ x ∈ Ioo (-1:ℝ) 0, deriv (f_monotonicity k) x > 0) ∧ (∀ x ∈ Ioo 0 (⊤:ℝ), deriv (f_monotonicity k) x < 0)
  else if 0 < k ∧ k < 1 then 
    (∀ x ∈ Ioo (-1:ℝ) 0, deriv (f_monotonicity k) x > 0) ∧ (∀ x ∈ Ioo 0 (1 - k / k:ℝ), deriv (f_monotonicity k) x < 0) ∧
    (∀ x ∈ Ioo (1 - k / k:ℝ) (⊤:ℝ), deriv (f_monotonicity k) x > 0)
  else
    (∀ x ∈ Ioo (-1:ℝ) (1 - k / k:ℝ), deriv (f_monotonicity k) x > 0) ∧ 
    (∀ x ∈ Ioo (1 - k / k:ℝ) 0, deriv (f_monotonicity k) x < 0) ∧ 
    (∀ x ∈ Ioo 0 (⊤:ℝ), deriv (f_monotonicity k) x > 0) :=
by sorry
end PartII

section PartIII
variables {x : ℝ}
def g (x : ℝ) := log (x + 1) + 1 / (x + 1) - 1

theorem log_inequality (h : x > -1) : log (x + 1) ≥ 1 - 1 / (x + 1) :=
by sorry
end PartIII

end tangent_line_at_1_monotonicity_intervals_log_inequality_l289_289609


namespace reflection_transformations_l289_289989

theorem reflection_transformations (a b c : ℝ) (h : a ≠ 0) :
  let f := λ x : ℝ, a * x^2 + b * x + c,
      f1 := λ x : ℝ, a * x^2 - b * x + c,
      f2 := λ x : ℝ, -a * x^2 + b * x + 2 - c in
  ∀ x, f2 x = -a * x^2 + b * x + 2 - c :=
by
  intro x
  sorry

end reflection_transformations_l289_289989


namespace math_inequality_l289_289681

noncomputable def PA (z z1 : ℂ) : ℝ := complex.abs (z - z1)
noncomputable def PB (z z2 : ℂ) : ℝ := complex.abs (z - z2)
noncomputable def PC (z z3 : ℂ) : ℝ := complex.abs (z - z3)
noncomputable def a (z2 z3 : ℂ) : ℝ := complex.abs (z2 - z3)
noncomputable def b (z3 z1 : ℂ) : ℝ := complex.abs (z3 - z1)
noncomputable def c (z1 z2 : ℂ) : ℝ := complex.abs (z1 - z2)

theorem math_inequality (z z1 z2 z3 : ℂ) :
  let PA := PA z z1
      PB := PB z z2
      PC := PC z z3
      a := a z2 z3
      b := b z3 z1
      c := c z1 z2
  in
  a * PA ^ 2 + b * PB ^ 2 + c * PC ^ 2  ≥ a * b + b * c + c * a :=
by sorry

end math_inequality_l289_289681


namespace f2015_f2018_l289_289599

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x < 2 then Real.log2 (x + 1) else 
                    if x + 2 ≤ 0 ∧ x + 2 < 2 then -f (x + 2) else 
                    Real.log2 (x + 1) 

theorem f2015_f2018 : f 2015 + f 2018 = -1 := by
  sorry

end f2015_f2018_l289_289599


namespace polar_curve_and_chord_length_l289_289228

-- Define the parametric equation of curve C
def curve_C := {α : ℝ | (3 + real.sqrt 10 * real.cos α, 1 + real.sqrt 10 * real.sin α)}

-- Polar equation transformation
def cartesian_to_polar (x y : ℝ) : (ℝ × ℝ) := (real.sqrt(x^2 + y^2), real.atan2 y x)

-- Distance from a point to a line
def distance_point_to_line (x y : ℝ) : ℝ :=
  abs (x - y + 1) / real.sqrt 2

-- Length of the chord
def chord_length (r d : ℝ) : ℝ :=
  2 * real.sqrt (r^2 - d^2)

theorem polar_curve_and_chord_length :
  (∀ α : ℝ, curve_C α = (θ : ℝ, ρ : ℝ), ρ = 6 * real.cos θ + 2 * real.sin θ) ∧
  (sin θ - cos θ = 1/ρ → chord_length (real.sqrt 10) (distance_point_to_line 3 1) = real.sqrt 22) :=
by
  sorry

end polar_curve_and_chord_length_l289_289228


namespace probability_rain_at_most_3_days_l289_289642

-- Define the conditions
def probability_of_rain : ℚ := 1 / 5
def days_in_april : ℕ := 30

-- Define binomial probability function
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the probability of raining on at most 3 days
def probability_of_at_most_3_days (n : ℕ) (p : ℚ) : ℚ :=
  (binomial_probability n 0 p) +
  (binomial_probability n 1 p) +
  (binomial_probability n 2 p) +
  (binomial_probability n 3 p)

-- Define the theorem asserting the probability
theorem probability_rain_at_most_3_days :
  abs (probability_of_at_most_3_days days_in_april probability_of_rain - 0.502) < 0.001 :=
by
  -- sorry (Proof would go here)
  sorry

end probability_rain_at_most_3_days_l289_289642


namespace find_X_plus_Y_in_base_8_l289_289168

theorem find_X_plus_Y_in_base_8 (X Y : ℕ) (h1 : 3 * 8^2 + X * 8 + Y + 5 * 8 + 2 = 4 * 8^2 + X * 8 + 3) : X + Y = 1 :=
sorry

end find_X_plus_Y_in_base_8_l289_289168


namespace diff_baseball_soccer_l289_289389

variable (totalBalls soccerBalls basketballs tennisBalls baseballs volleyballs : ℕ)

axiom h1 : totalBalls = 145
axiom h2 : soccerBalls = 20
axiom h3 : basketballs = soccerBalls + 5
axiom h4 : tennisBalls = 2 * soccerBalls
axiom h5 : baseballs > soccerBalls
axiom h6 : volleyballs = 30

theorem diff_baseball_soccer : baseballs - soccerBalls = 10 :=
  by {
    sorry
  }

end diff_baseball_soccer_l289_289389


namespace distance_A_to_BC_l289_289942

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def coord_diff (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def dot_product (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y + p1.z * p2.z

def magnitude (p : Point) : ℝ :=
  sqrt (p.x^2 + p.y^2 + p.z^2)

def distance_from_point_to_line (A B C : Point) : ℝ :=
  let AB := coord_diff A B
  let BC := coord_diff B C
  let mag_AB := magnitude AB
  let mag_BC := magnitude BC
  let dot_AB_BC := dot_product AB BC
  let cos_theta := dot_AB_BC / (mag_AB * mag_BC)
  mag_AB * sqrt (1 - cos_theta^2)

theorem distance_A_to_BC :
  let A := Point.mk 0 0 2
  let B := Point.mk 1 0 2
  let C := Point.mk 0 2 0
  distance_from_point_to_line A B C = 2 * sqrt 2 / 3 := by
  sorry

end distance_A_to_BC_l289_289942


namespace correct_order_l289_289075

-- Define contestants as a set of strings
def contestants := {"A", "B", "C", "D", "E"}

-- Define the initial guess as a list
def initial_guess := ["A", "B", "C", "D", "E"]

-- Define the second guess as a list
def second_guess := ["D", "A", "E", "C", "B"]

-- Define the conditions
def valid_permutation (order : List String) : Prop :=
  ∀ (i : ℕ), i < 5 → (order.nth i).get_or_else "" ∈ contestants

def condition1 (order : List String) : Prop :=
  order ≠ initial_guess ∧
  ∀ i, i < 4 → (order.nth i).get_or_else "" ≠ initial_guess.nth i ∧
    ((order.nth i).get_or_else "", (order.nth (i + 1)).get_or_else "") ∉ [
      ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")
    ]

def condition2 (order : List String) : Prop :=
  (order.nth 0).get_or_else "" = "D" ∨ order.nth 1 = some "D" ∨
  (order.nth 2).get_or_else "" = "E" ∨ order.nth 3 = some "E" ∨
  order.nth 4 = some "B" ∧
  ∃ i j, i ≠ j ∧ order.nth i = second_guess.nth i ∧ 
    order.nth j = (second_guess.nth j).get_or_else "" ∧
    ∃ k, k < 4 ∧ list.is_adjacent order k k.succ

-- Prove the correct order based on conditions
theorem correct_order :
  ∃ order : List String, valid_permutation order ∧ 
  condition1 order ∧ condition2 order ∧ order = ["E", "D", "A", "C", "B"] :=
sorry

end correct_order_l289_289075


namespace problem_l289_289965

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (3 * x - Real.pi / 3)

theorem problem 
  (x₁ x₂ : ℝ)
  (hx₁x₂ : |f x₁ - f x₂| = 4)
  (x : ℝ)
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 6)
  (m : ℝ) : m ≥ 1 / 3 :=
sorry

end problem_l289_289965


namespace solution_set_f_x_le_5_l289_289606

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 + Real.log x / Real.log 2 else x^2 - x - 1

theorem solution_set_f_x_le_5 : {x : ℝ | f x ≤ 5} = Set.Icc (-2 : ℝ) 4 := by
  sorry

end solution_set_f_x_le_5_l289_289606


namespace find_two_alpha_plus_beta_l289_289591

noncomputable theory

open Real

variables {α β : ℝ}
hypothesis (hα : tan α = 1 / 3) (hβ : tan β = 1 / 7) 
(hα_acute : 0 < α ∧ α < π / 2)
(hβ_acute : 0 < β ∧ β < π / 2)

theorem find_two_alpha_plus_beta : 2 * α + β = π / 4 :=
sorry

end find_two_alpha_plus_beta_l289_289591


namespace number_of_groups_that_can_form_set_is_2_l289_289130

-- Definitions based on the conditions:
def numbers_close_to_zero_can_form_set : Prop := False
def small_positive_integers_can_form_set : Prop := False
def points_distance_1_from_O_can_form_set : Prop := True
def equilateral_triangles_can_form_set : Prop := True
def approximate_values_of_sqrt_2_can_form_set : Prop := False

-- Theorem statement to prove the number of groups that can form a set equals 2.
theorem number_of_groups_that_can_form_set_is_2 :
  (if numbers_close_to_zero_can_form_set then 1 else 0) +
  (if small_positive_integers_can_form_set then 1 else 0) +
  (if points_distance_1_from_O_can_form_set then 1 else 0) +
  (if equilateral_triangles_can_form_set then 1 else 0) +
  (if approximate_values_of_sqrt_2_can_form_set then 1 else 0) = 2 :=
by
  sorry

end number_of_groups_that_can_form_set_is_2_l289_289130


namespace expr_D_is_diff_of_squares_l289_289524

-- Definitions for the expressions
def expr_A (a b : ℤ) : ℤ := (a + 2 * b) * (-a - 2 * b)
def expr_B (m n : ℤ) : ℤ := (2 * m - 3 * n) * (3 * n - 2 * m)
def expr_C (x y : ℤ) : ℤ := (2 * x - 3 * y) * (3 * x + 2 * y)
def expr_D (a b : ℤ) : ℤ := (a - b) * (-b - a)

-- Theorem stating that Expression D can be calculated using the difference of squares formula
theorem expr_D_is_diff_of_squares (a b : ℤ) : expr_D a b = a^2 - b^2 :=
by sorry

end expr_D_is_diff_of_squares_l289_289524


namespace find_S_l289_289598

noncomputable theory
open_locale classical

variables {x A B : ℝ}

def problem_condition (A B : ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → x ≠ 1 → (2 * x - 3) / (x ^ 2 - x) = A / (x - 1) + B / x

def S (A B : ℝ) : ℝ := A ^ 2 + B ^ 2

theorem find_S (h : problem_condition A B) : S A B = 10 :=
sorry

end find_S_l289_289598


namespace balls_remaining_l289_289032

-- Define the initial number of balls in the box
def initial_balls := 10

-- Define the number of balls taken by Yoongi
def balls_taken := 3

-- Define the number of balls left after Yoongi took some balls
def balls_left := initial_balls - balls_taken

-- The theorem statement to be proven
theorem balls_remaining : balls_left = 7 :=
by
    -- Skipping the proof
    sorry

end balls_remaining_l289_289032


namespace average_temperature_l289_289447

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end average_temperature_l289_289447


namespace sum_of_center_coordinates_l289_289731

theorem sum_of_center_coordinates (A B C D : ℝ × ℝ)
  (h1 : A.1 = B.1) (h2 : C.1 = D.1)
  (h3 : A.2 = 0) (h4 : B.2 = 0) (h5 : C.2 = 0) (h6 : D.2 = 0)
  (h7 : A = (4, 0)) (h8 : B = (6, 0)) (h9 : C = (10, 0)) (h10 : D = (16, 0))
  (h11 : (B.1 - A.1) = (D.1 - C.1)) 
  (h12 : (D.2 - B.2) = (A.2 - C.2)) : 
  (let center := ((A.1 + C.1)/2, (A.2 + C.2)/2) in center.1 + center.2 = 13) :=
by 
  -- placeholder for the actual proof
  sorry

end sum_of_center_coordinates_l289_289731


namespace line_accuracy_l289_289501

theorem line_accuracy (P Q : ℝ × ℝ) : ∀ ε > 0, ∃ δ > 0, (dist P Q > δ) → (abs (angle (line_through P Q) - intended_angle) < ε) := sorry

end line_accuracy_l289_289501


namespace find_speed_of_bus_l289_289453

noncomputable def speed_of_bus (v : ℝ) : Prop :=
  (0.5 * v) + (2 * (v + 20)) = 140

theorem find_speed_of_bus : ∃ v, speed_of_bus v ∧ v = 40 :=
by {
  existsi 40,
  unfold speed_of_bus,
  norm_num
}

end find_speed_of_bus_l289_289453


namespace trains_fully_cross_time_l289_289519

-- Define the speeds and lengths of the trains
def speed_train1 := 100 -- in km/h
def length_train1 := 100 -- in meters
def speed_train2 := 120 -- in km/h
def length_train2 := 150 -- in meters

-- Convert speeds to m/s
def kmh_to_ms (kmh : ℕ) : ℝ := (kmh * 1000) / 3600

-- Define the relative speed and combined length
def relative_speed := kmh_to_ms (speed_train1 + speed_train2) -- in m/s
def combined_length := length_train1 + length_train2 -- in meters

-- Calculate the time to cross
def time_to_cross (length : ℕ) (speed : ℝ) : ℝ := length / speed

theorem trains_fully_cross_time : 
  time_to_cross combined_length relative_speed ≈ 4.09 :=
by
  sorry

end trains_fully_cross_time_l289_289519


namespace sqrt_inequality_l289_289950

theorem sqrt_inequality (C : ℝ) (hC : C > 1) :
  (sqrt (C + 1) - sqrt C) < (sqrt C - sqrt (C - 1)) :=
by sorry

end sqrt_inequality_l289_289950


namespace impossible_triangle_angle_sum_l289_289065

theorem impossible_triangle_angle_sum (x y z : ℝ) (h : x + y + z = 180) : x + y + z ≠ 360 :=
by
sorry

end impossible_triangle_angle_sum_l289_289065


namespace largest_num_with_diff_digits_sum_17_l289_289803
-- Import the necessary library

-- Define the conditions and the answer
def digits_all_different (n : Nat) : Prop := 
  (n.toDigits).nodup

def digits_sum_17 (n : Nat) : Prop :=
  (n.toDigits).sum = 17

def largest_number := 6543210

-- The main statement to be proved
theorem largest_num_with_diff_digits_sum_17 : ∃ n, digits_all_different n ∧ digits_sum_17 n ∧ n = largest_number :=
by
  sorry

end largest_num_with_diff_digits_sum_17_l289_289803


namespace problem1_problem2_l289_289486

-- For Problem 1
theorem problem1 :
  125 ^ (2 / 3 : ℝ) + (1 / 2 : ℝ) ^ (-2) - (1 / 27 : ℝ) ^ (-1 / 3) + 100 ^ (1 / 2) +
  ((Real.log 3 + (1 / 4) * Real.log 9 - Real.log (3 ^ (1 / 2))) / (Real.log 81 - Real.log 27)) = 37 :=
by sorry

-- For Problem 2
theorem problem2 (a : ℝ) (h₁ : a ≠ -1) (h₂ : a ≠ 0) :
  (a ^ (3 / 2) - 1) / (a + a ^ (1 / 2) + 1) - (a + a ^ (1 / 2)) / (a ^ (1 / 2) + 1) + (a - 1) / (a ^ (1 / 2) - 1) =
  a ^ (1 / 2) :=
by sorry

end problem1_problem2_l289_289486


namespace buns_left_l289_289043

theorem buns_left (buns_initial : ℕ) (h1 : buns_initial = 15)
                  (x : ℕ) (h2 : 13 * x ≤ buns_initial)
                  (buns_taken_by_bimbo : ℕ) (h3 : buns_taken_by_bimbo = x)
                  (buns_taken_by_little_boy : ℕ) (h4 : buns_taken_by_little_boy = 3 * x)
                  (buns_taken_by_karlsson : ℕ) (h5 : buns_taken_by_karlsson = 9 * x)
                  :
                  buns_initial - (buns_taken_by_bimbo + buns_taken_by_little_boy + buns_taken_by_karlsson) = 2 :=
by
  sorry

end buns_left_l289_289043


namespace maximum_triangle_area_l289_289622

open Real

noncomputable def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  1 / 2 * abs (x₁ * (y₂ - 3) + x₂ * (3 - y₁))

theorem maximum_triangle_area :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 = 4 * y₁) →
    (x₂^2 = 4 * y₂) →
    (y₁ + y₂ = 2) →
    (y₁ ≠ y₂) →
    triangle_area x₁ y₁ x₂ y₂ ≤ (16 * real.sqrt 6) / 9 :=
  sorry

end maximum_triangle_area_l289_289622


namespace arithmetic_expression_count_l289_289580

theorem arithmetic_expression_count (f : ℕ → ℤ) 
  (h1 : f 1 = 9)
  (h2 : f 2 = 99)
  (h_recur : ∀ n ≥ 2, f n = 9 * (f (n - 1)) + 36 * (f (n - 2))) :
  ∀ n, f n = (7 / 10 : ℚ) * 12^n - (1 / 5 : ℚ) * (-3)^n := sorry

end arithmetic_expression_count_l289_289580


namespace problem_arithmetic_sequence_l289_289209

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arithmetic_sequence (a : ℕ → ℝ) (d a2 a8 : ℝ) :
  arithmetic_sequence a d →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 450) →
  (a 1 + a 7 = 2 * a 4) →
  (a 2 + a 6 = 2 * a 4) →
  (a 2 + a 8 = 180) :=
by
  sorry

end problem_arithmetic_sequence_l289_289209


namespace find_distance_from_A_to_line_BC_l289_289948

open Real EuclideanGeometry

noncomputable def distance_from_point_to_line (A B C : Point) : Real :=
  let AB := B - A
  let BC := C - B
  let cross_product := AB × BC -- This is a 3D vector cross product
  let magnitude_BC := |BC|
  |cross_product| / magnitude_BC

theorem find_distance_from_A_to_line_BC :
  let A := (0, 0, 2) : Point
  let B := (1, 0, 2) : Point
  let C := (0, 2, 0) : Point
  distance_from_point_to_line A B C = 2 * sqrt 2 / 3 :=
by
  sorry

end find_distance_from_A_to_line_BC_l289_289948


namespace area_of_circle_above_y_equals_2_l289_289060

def circle_area_above_line (x y : ℝ) : Prop :=
  (x^2 + 4*x + y^2 - 8*y + 20 = 0) → (y ≥ 2)

theorem area_of_circle_above_y_equals_2 :
  ∀ (x y : ℝ), circle_area_above_line x y →
  circle_area_above_line x y = 8 * real.pi  :=
    sorry

end area_of_circle_above_y_equals_2_l289_289060


namespace angle_between_chord_and_origin_is_right_angle_l289_289604

theorem angle_between_chord_and_origin_is_right_angle (p : ℝ) (h : p > 0) :
  let A := (p, 0)
  let O := (0, 0)
  let P1 := (p, p)
  let P2 := (p, -p)
  let chord := (P1, P2)
  ∠ (P1, O, P2) = 90 :=
  sorry

end angle_between_chord_and_origin_is_right_angle_l289_289604


namespace solve_for_x_l289_289002

theorem solve_for_x : 
  (x : ℚ) (h : x = 45 / (7 - 3 / 4)) : x = 36 / 5 :=
sorry

end solve_for_x_l289_289002


namespace obtuse_angle_inequality_vector_magnitude_l289_289081

theorem obtuse_angle_inequality (λ : ℝ) :
    let a := (-2, -1)
    let b := (λ, 1)
    (a.1 * b.1 + a.2 * b.2 < 0) ∧ (-2 + λ ≠ 0) → λ > -1/2 ∧ λ ≠ 2 :=
by
  -- Definitions and assumptions go here
  sorry

theorem vector_magnitude (a b c : ℝ × ℝ × ℝ) 
    (a_magnitude : |a| = 2)
    (b_magnitude : |b| = 2)
    (c_magnitude : |c| = 1)
    (equal_angles : ∀ u v : ℝ × ℝ × ℝ, u ≠ v → |(u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / (|u| * |v|)| = cos (2 * pi / 3)) :
    |a + b + c| = 1 :=
by
  -- Definitions and assumptions go here
  sorry

end obtuse_angle_inequality_vector_magnitude_l289_289081


namespace area_triangle_proof_l289_289785

noncomputable def area_of_triangle : ℝ :=
  let intersection_A := (2, 6) in
  let intersection_B := (7, 1) in
  let intersection_C := (1, 3) in
  (1 / 2) * abs (2 * (1 - 3) + 7 * (3 - 6) + 1 * (6 - 1))

theorem area_triangle_proof : area_of_triangle = 10 :=
by
  -- Line equations
  have line1 : ∀ x y : ℝ, y = 3 * x ↔ y - 3 = 3 * (x - 1), by
    intros x y
    rw [sub_eq_iff_eq_add, eq_comm, sub_eq_iff_eq_add, eq_comm, mul_sub, mul_one]
    exact ⟨λ h, by rw h; simp, λ h, by linarith⟩
  
  have line2 : ∀ x y : ℝ, y = (-1 / 3) * x + 10 / 3 ↔ y - 3 = (-1 / 3) * (x - 1), by
    intros x y
    rw [sub_eq_iff_eq_add, eq_comm, sub_eq_iff_eq_add, eq_comm, div_mul, one_div, mul_neg]
    exact ⟨λ h, by rw h; ring, λ h, by ring⟩

  have line3 : ∀ x y : ℝ, x + y = 8 ↔ x + y = 8, from λ x y, Iff.rfl
  
  sorry

end area_triangle_proof_l289_289785


namespace rectangular_prism_dimensions_l289_289859

theorem rectangular_prism_dimensions (b l h : ℕ) 
  (h1 : l = 3 * b) 
  (h2 : l = 2 * h) 
  (h3 : l * b * h = 12168) :
  b = 14 ∧ l = 42 ∧ h = 21 :=
by
  -- The proof will go here
  sorry

end rectangular_prism_dimensions_l289_289859


namespace min_socks_no_conditions_l289_289259

theorem min_socks_no_conditions (m n : Nat) (h : (m * (m - 1) = 2 * (m + n) * (m + n - 1))) : 
  m + n ≥ 4 := sorry

end min_socks_no_conditions_l289_289259


namespace distance_from_A_to_line_BC_l289_289944

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def distance_from_point_to_line (A B C : Point3D) : ℝ :=
  let AB := (B.x - A.x, B.y - A.y, B.z - A.z)
  let BC := (C.x - B.x, C.y - B.y, C.z - B.z)
  let magnitude := λ (v : ℝ × ℝ × ℝ), Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let dot_product := λ (v w : ℝ × ℝ × ℝ), v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let cos_theta := dot_product AB BC / ((magnitude AB) * (magnitude BC))
  magnitude AB * Real.sqrt (1 - cos_theta ^ 2)

theorem distance_from_A_to_line_BC (A B C : Point3D) (hA : A = ⟨0, 0, 2⟩) 
(hB : B = ⟨1, 0, 2⟩) (hC : C = ⟨0, 2, 0⟩) : 
distance_from_point_to_line A B C = 2 * Real.sqrt 2 / 3 := 
by
  sorry

end distance_from_A_to_line_BC_l289_289944


namespace parabola_opens_upward_l289_289413

theorem parabola_opens_upward (a : ℝ) (b : ℝ) (h : a > 0) : ∀ x : ℝ, 3*x^2 + 2 = a*x^2 + b → a = 3 ∧ b = 2 → ∀ x : ℝ, 3 * x^2 + 2 ≤ a * x^2 + b := 
by
  sorry

end parabola_opens_upward_l289_289413


namespace last_three_digits_of_5_pow_9000_l289_289815

theorem last_three_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 800]) : 5^9000 ≡ 1 [MOD 800] :=
by
  -- The proof is omitted here according to the instruction
  sorry

end last_three_digits_of_5_pow_9000_l289_289815


namespace point_transformations_l289_289426

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, -p.3)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.2, -p.1, p.3)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

theorem point_transformations :
  let p₀ := (2, 2, 2) in
  let p₁ := rotate_y_180 p₀ in
  let p₂ := reflect_yz p₁ in
  let p₃ := rotate_z_90 p₂ in
  let p₄ := reflect_xz p₃ in
  let p₅ := reflect_xy p₄ in
  p₅ = (2, 2, 2) :=
by
  sorry

end point_transformations_l289_289426


namespace sock_pairs_count_l289_289632

theorem sock_pairs_count :
  let white := 4 in
  let brown := 4 in
  let blue := 2 in
  let gray := 5 in
  15 = white + brown + blue + gray →
  (white * brown + white * blue + white * gray + brown * blue + brown * gray + blue * gray) = 82 :=
by
  intros white brown blue gray h_sum
  sorry

end sock_pairs_count_l289_289632


namespace vacant_seats_l289_289645

open Nat

-- Define the conditions as Lean definitions
def num_tables : Nat := 5
def seats_per_table : Nat := 8
def occupied_tables : Nat := 2
def people_per_occupied_table : Nat := 3
def unusable_tables : Nat := 1

-- Calculate usable tables
def usable_tables : Nat := num_tables - unusable_tables

-- Calculate total occupied people
def total_occupied_people : Nat := occupied_tables * people_per_occupied_table

-- Calculate total seats for occupied tables
def total_seats_occupied_tables : Nat := occupied_tables * seats_per_table

-- Calculate vacant seats in occupied tables
def vacant_seats_occupied_tables : Nat := total_seats_occupied_tables - total_occupied_people

-- Calculate completely unoccupied tables
def unoccupied_tables : Nat := usable_tables - occupied_tables

-- Calculate total seats for unoccupied tables
def total_seats_unoccupied_tables : Nat := unoccupied_tables * seats_per_table

-- Calculate total vacant seats
def total_vacant_seats : Nat := vacant_seats_occupied_tables + total_seats_unoccupied_tables

-- Theorem statement to prove
theorem vacant_seats : total_vacant_seats = 26 := by
  sorry

end vacant_seats_l289_289645


namespace marbles_given_to_juan_l289_289147

def initial : ℕ := 776
def left : ℕ := 593

theorem marbles_given_to_juan : initial - left = 183 :=
by sorry

end marbles_given_to_juan_l289_289147


namespace trapezoid_equal_areas_l289_289520

open_locale classical
noncomputable theory

variables {A B C D M : Type*} [ordered_ring A]

structure Trapezoid (ABCD : set (A × A)) := 
(base1 : set (A × A))
(base2 : set (A × A))
(non_parallel_side1 : set (A × A))
(non_parallel_side2 : set (A × A))
(intersects_at_M : (A × A))

def is_trapezoid (ABCD : set (A × A)) (AD BC AB CD : set (A × A)) (M : A) : Prop :=
AD ∈ ABCD ∧ BC ∈ ABCD ∧ AB ∈ ABCD ∧ CD ∈ ABCD ∧ 
(intersection AD BC = {M}) ∧ (intersection AB CD = {M})

theorem trapezoid_equal_areas 
(ABCD : set (A × A)) (AD BC AB CD : set (A × A)) (M : A × A) 
(h_trapezoid : is_trapezoid ABCD AD BC AB CD M) :
(area (triangle AMB) = area (triangle CMD)) := 
sorry

end trapezoid_equal_areas_l289_289520


namespace integer_solution_inequality_l289_289626

theorem integer_solution_inequality (x : ℤ) : ((x - 1)^2 ≤ 4) → ([-1, 0, 1, 2, 3].count x = 5) :=
by
  sorry

end integer_solution_inequality_l289_289626


namespace AD_bisects_EDF_l289_289651

noncomputable def bisect_angle (A B C D P E F : Type) [triangle A B C] [altitude_foot A D B C] [point_on_segment P A D] [intersect_line_segment BP E AC] [intersect_line_segment CP F AB] : Prop :=
  AD E F = angle_bisector E D F

theorem AD_bisects_EDF (A B C D P E F : Type) [triangle A B C] [altitude_foot A D B C] [point_on_segment P A D] [intersect_line_segment BP E AC] [intersect_line_segment CP F AB]:  
  bisect_angle A B C D P E F :=
sorry

end AD_bisects_EDF_l289_289651


namespace find_area_of_triangle_l289_289258

noncomputable def triangle_area (a b: ℝ) (cosC: ℝ) : ℝ :=
  let sinC := Real.sqrt (1 - cosC^2)
  0.5 * a * b * sinC

theorem find_area_of_triangle :
  ∀ (a b cosC : ℝ), a = 3 * Real.sqrt 2 → b = 2 * Real.sqrt 3 → cosC = 1 / 3 →
  triangle_area a b cosC = 4 * Real.sqrt 3 :=
by
  intros a b cosC ha hb hcosC
  rw [ha, hb, hcosC]
  sorry

end find_area_of_triangle_l289_289258
