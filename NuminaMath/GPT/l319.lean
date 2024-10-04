import Data.Real.Basic
import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.Geometry.Ellipse
import Mathlib.Analysis.Geometry.Euclidean.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Matrix
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.EuclideanSpace.Basic
import analysis.special_functions.trigonometric
import data.real.basic

namespace some_pens_are_not_vens_l319_319495

variable (Pen Den Ven : Type)
variable (is_pen : Pen → Den)
variable (is_not_ven : ∃ d : Den, ¬ (d → Ven))

theorem some_pens_are_not_vens (h1 : ∀ p : Pen, is_pen p)
    (h2 : ∃ d : Den, ¬ d) :
  ∃ p : Pen, ¬ (is_pen p → Ven) :=
by sorry

end some_pens_are_not_vens_l319_319495


namespace correct_option_is_d_l319_319318

-- Define the conditions as propositions
def OptionA : Prop := 
  ¬ isCertainEvent ("It rains a lot during the Qingming Festival")

def OptionB : Prop :=
  ¬ (censusMethodNecessary (understandServiceLife lampTube))

def variance (S : Type) [Real] := S

def OptionC (S_A S_B : variance ℝ) : Prop :=
  S_A ^ 2 = 0.01 ∧ S_B ^ 2 = 0.02 ∧ S_B > S_A

def mode (l : List ℝ) : ℝ := 5 -- simplified calculation step
def median (l : List ℝ) : ℝ := 5 -- simplified calculation step
def mean (l : List ℝ) : ℝ := 5 -- simplified calculation step

def OptionD : Prop :=
  let l := [3, 5, 4, 5, 6, 7] in
  mode l = 5 ∧ median l = 5 ∧ mean l = 5

-- Define the theorem that given all conditions, the correct statement is D
theorem correct_option_is_d :
  ¬ OptionA ∧ ¬ OptionB ∧ ¬ OptionC 0.01 0.02 ∧ OptionD → 
  (OptionA ∨ OptionB ∨ OptionC 0.01 0.02 ∨ OptionD) :=
  sorry

end correct_option_is_d_l319_319318


namespace calculate_width_l319_319223

-- Define the conditions provided in the problem
variables (length_apartment : ℝ)
          (total_rooms : ℕ)
          (size_living_room : ℝ)
          (total_area : ℝ)

-- Define the known conditions as hypotheses
def apartment_conditions :=
  length_apartment = 16 ∧ total_rooms = 6 ∧
  ∃ (size_other_room : ℝ), size_living_room = 3 * size_other_room ∧ size_living_room = 60 ∧
  total_area = 5 * size_other_room + size_living_room

-- The width is the total area divided by the length of the apartment
def apartment_width (h : apartment_conditions) : Prop :=
  (total_area / length_apartment) = 10

-- The proof statement
theorem calculate_width (h : apartment_conditions) : apartment_width h :=
  sorry

end calculate_width_l319_319223


namespace bug_total_distance_l319_319812

def total_distance_bug (start : ℤ) (pos1 : ℤ) (pos2 : ℤ) (pos3 : ℤ) : ℤ :=
  abs (pos1 - start) + abs (pos2 - pos1) + abs (pos3 - pos2)

theorem bug_total_distance :
  total_distance_bug 3 (-4) 6 2 = 21 :=
by
  -- We insert a sorry here to indicate the proof is skipped.
  sorry

end bug_total_distance_l319_319812


namespace find_angle_C_l319_319470

noncomputable def measure_angle_C (a b : ℝ) (cosB : ℝ) : ℝ :=
  if h : a = 2 * Real.sqrt 6 ∧ b = 6 ∧ cosB = -1 / 2 then
    π - (π / 4) - (2 * π / 3)
  else
    sorry

theorem find_angle_C :
  measure_angle_C (2 * Real.sqrt 6) 6 (-1/2) = π / 12 :=
by
  unfold measure_angle_C
  have h : (2 * Real.sqrt 6 = 2 * Real.sqrt 6 ∧ 6 = 6 ∧ -1 / 2 = -1 / 2) := by
    simp
  simp [h]
  sorry

end find_angle_C_l319_319470


namespace assign_grades_l319_319008

def num_students : ℕ := 15
def options_per_student : ℕ := 4

theorem assign_grades:
  options_per_student ^ num_students = 1073741824 := by
  sorry

end assign_grades_l319_319008


namespace scientific_notation_correct_l319_319664

-- Define the number 0.0008
def num : ℝ := 0.0008

-- The condition given in the problem
def plays : ℝ := 10800000

-- Statement we need to prove
theorem scientific_notation_correct :
  num = 8 * 10^(-4) :=
sorry

end scientific_notation_correct_l319_319664


namespace proposition_④_l319_319016

-- Definitions and conditions
variables (line : Type) (plane : Type)
variables (a b : line) (α : plane)
variables (parallel : line → line → Prop) (parallel_plane : line → plane → Prop)
variables (subset_plane : line → plane → Prop) (not_subset_plane : Π (l : line) (p : plane), Prop)

-- Conditions
axiom parallel_a_b : parallel a b
axiom parallel_a_alpha : parallel_plane a α
axiom not_subset_b_alpha : not_subset_plane b α

-- Proposition ④ (our goal to prove)
theorem proposition_④ (parallel : line → line → Prop) (parallel_plane : line → plane → Prop)
  (subset_plane : line → plane → Prop) (not_subset_plane : Π (l : line) (p : plane), Prop)
  (a b : line) (α : plane) :
  parallel a b → parallel_plane a α → not_subset_plane b α → parallel_plane b α :=
by
  intros
  sorry

end proposition_④_l319_319016


namespace area_of_PDCE_l319_319549

/-- A theorem to prove the area of quadrilateral PDCE given conditions in triangle ABC. -/
theorem area_of_PDCE
  (ABC_area : ℝ)
  (BD_to_CD_ratio : ℝ)
  (E_is_midpoint : Prop)
  (AD_intersects_BE : Prop)
  (P : Prop)
  (area_PDCE : ℝ) :
  (ABC_area = 1) →
  (BD_to_CD_ratio = 2 / 1) →
  E_is_midpoint →
  AD_intersects_BE →
  ∃ P, P →
    area_PDCE = 7 / 30 :=
by sorry

end area_of_PDCE_l319_319549


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319747

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319747


namespace count_integers_between_square_bounds_l319_319144

theorem count_integers_between_square_bounds :
  (n : ℕ) (300 < n^2 ∧ n^2 < 1200) → 17 :=
sorry

end count_integers_between_square_bounds_l319_319144


namespace number_of_integers_l319_319143

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end number_of_integers_l319_319143


namespace triangle_AMC_is_isosceles_circumcenter_AMC_on_circle_l319_319451

-- Definitions and conditions
variables {O A B C M R : Type} [MetricSpace O]
variables [HasDist O]

def on_circle (A O : O) (R : ℝ) := dist O A = R
def is_tangent (e c : Set O) (A : O) := ∃ l, ∀ x ∈ e, dist A x = l → ∀ y ∈ c, y ≠ A → ∃ m, dist A y = m
def passes_through (d : Set O) (O B C : O) := ∃ x, x ∈ d ∧ dist O B = dist O C
def intersects (d e : Set O) (M : O) := ∀ x ∈ d, ∀ y ∈ e, x = y → M = x
def point_between (B O A : O) := dist O B < dist O A
def dist_eq (A M : O) (length : ℝ) := dist A M = length

-- Given conditions
axiom h1 : on_circle A O R
axiom h2 : ∃ e : Set O, is_tangent e (Metric.sphere O R) A
axiom h3 : ∃ d : Set O, (passes_through d O B C) ∧ intersects d (classical.some h2) M
axiom h4 : point_between B O A
axiom h5 : dist_eq A M (R * Real.sqrt 3)

-- Prove part (a): triangle AMC is isosceles
theorem triangle_AMC_is_isosceles : is_isosceles_triangle A M C :=
sorry

-- Prove part (b): circumcenter of triangle AMC lies on circle c
theorem circumcenter_AMC_on_circle : ∃ P, is_circumcenter P A M C ∧ on_circle P O R :=
sorry

end triangle_AMC_is_isosceles_circumcenter_AMC_on_circle_l319_319451


namespace solve_exponential_equation_l319_319787

theorem solve_exponential_equation (a x : ℝ) (h₁ : x ≠ 0) (h₂ : a ≠ -2) (h₃ : a ≠ -3) (h₄ : a ≠ 1 / 2) :
  (2 ^ ((a + 3) / (a + 2))) * (32 ^ (1 / (x * (a + 2)))) = 4 ^ (1 / x) ↔
  x = (2 * a - 1) / (a + 3) :=
by {
  -- proof steps would go here, but will be omitted (sorry)
  sorry
}

end solve_exponential_equation_l319_319787


namespace traceable_edges_l319_319600

-- Define the vertices of the rectangle
def vertex (x y : ℕ) : ℕ × ℕ := (x, y)

-- Define the edges of the rectangle
def edges : List (ℕ × ℕ) :=
  [vertex 0 0, vertex 0 1,    -- vertical edges
   vertex 1 0, vertex 1 1,
   vertex 2 0, vertex 2 1,
   vertex 0 0, vertex 1 0,    -- horizontal edges
   vertex 1 0, vertex 2 0,
   vertex 0 1, vertex 1 1,
   vertex 1 1, vertex 2 1]

-- Define the theorem to be proved
theorem traceable_edges :
  ∃ (count : ℕ), count = 61 :=
by
  sorry

end traceable_edges_l319_319600


namespace greatest_multiple_of_5_and_6_under_1000_l319_319738

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319738


namespace eccentricity_ellipse_isosceles_right_triangle_l319_319574

theorem eccentricity_ellipse_isosceles_right_triangle :
  ∃ e : Real,
    (∀ A B C : Point, ∃ AB AC BC : Real,
      -- right isosceles triangle property
      (BC = 4 * Real.sqrt 2) ∧
      (AB = AC) ∧
      -- ellipse focus property
      ∀ D : Point, (D ∈ LineSegment AB) →
        ∀ e : Ellipse, e.focus1 = C ∧ e.focus2 = D ∧ (e.contains A) ∧ (e.contains B) →
          -- equation for eccentricity
          e.eccentricity = Real.sqrt 6 - Real.sqrt 3) :=
sorry

end eccentricity_ellipse_isosceles_right_triangle_l319_319574


namespace incorrect_inequality_l319_319465

-- Given definitions
variables {a b : ℝ}
axiom h : a < b ∧ b < 0

-- Equivalent theorem statement
theorem incorrect_inequality (ha : a < b) (hb : b < 0) : (1 / (a - b)) < (1 / a) := 
sorry

end incorrect_inequality_l319_319465


namespace minimum_sum_of_dimensions_l319_319831

theorem minimum_sum_of_dimensions {a b c : ℕ} (h1 : a * b * c = 2310) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  a + b + c ≥ 42 := 
sorry

end minimum_sum_of_dimensions_l319_319831


namespace sum_product_digits_2017_l319_319403

def P (n : ℕ) : ℕ :=
  n.digits.map (λ d, if d = 0 then 1 else d).prod

theorem sum_product_digits_2017 :
  ∑ k in Finset.range 2018, P k = 188370 := by
  sorry

end sum_product_digits_2017_l319_319403


namespace probability_two_cards_sum_to_15_same_suit_l319_319302

theorem probability_two_cards_sum_to_15_same_suit :
  let n : ℚ := 5 / 663 in
  ∀ (deck : Finset (Fin 52)),
    deck.card = 52 →
    (∀ card ∈ deck, 2 ≤ card.1 + 1 ∧ card.1 + 1 ≤ 10) →
    (∃ (deck1 deck2 : Finset (Fin 52)) (sum : ℕ),
       deck1 ≠ deck2 ∧
       deck1.card = 2 ∧
       deck2.card = 2 ∧
       (sum = deck1.1 + deck2.1) ∧
       sum = 15 ∧
       (∀ card1 ∈ deck1, ∀ card2 ∈ deck2, card1 ≠ card2) ∧
       deck1.card + deck2.card = n) :=
begin
  sorry
end

end probability_two_cards_sum_to_15_same_suit_l319_319302


namespace odd_and_decreasing_on_0_1_neg_sin_and_inv_x_l319_319784

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x > f y

theorem odd_and_decreasing_on_0_1_neg_sin_and_inv_x :
  is_odd (λ x : ℝ, -sin x) ∧ is_decreasing (λ x : ℝ, -sin x) (set.Ioo 0 1) ∧
  is_odd (λ x : ℝ, x⁻¹) ∧ is_decreasing (λ x : ℝ, x⁻¹) (set.Ioo 0 1) :=
by
  sorry

end odd_and_decreasing_on_0_1_neg_sin_and_inv_x_l319_319784


namespace equation_of_ellipse_HN_passes_through_fixed_point_l319_319111

-- Definitions of points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Center and axes of the ellipse
def center : ℝ × ℝ := (0, 0)
def x_axis_symmetry := True
def y_axis_symmetry := True

-- The ellipse passes through A and B
def ellipse (x y : ℝ) : Prop := (3 * x^2 + 4 * y^2 = 12)

theorem equation_of_ellipse :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 :=
begin
  split;
  unfold ellipse;
  norm_num,
end

theorem HN_passes_through_fixed_point :
  ∃ K : ℝ × ℝ, (K = (0, -2)) ∧
  ∀ (M N T H : ℝ × ℝ), (ellipse M.1 M.2) ∧ (ellipse N.1 N.2) ∧ 
  (∃ (k : ℝ), N.1 = k * M.1 ∧ N.2 = k * M.2) ∧ -- line MN
  (T.1 = M.1) ∧ (∃ (yT : ℝ), (T.1, yT) = T) ∧ -- T on line AB
  (H.1 = 2 * M.1 - T.1) ∧ (H.2 = 2 * M.2 - T.2) -> -- H's coordinates
  ((N.1 - H.1) = 0∧ (N.2 - H.2) = -2) := -- checking HN passes through (0,-2)
sorry

end equation_of_ellipse_HN_passes_through_fixed_point_l319_319111


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319692

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319692


namespace solve_abs_inequality_l319_319243

theorem solve_abs_inequality (x : ℝ) :
  |x + 2| + |x - 2| < x + 7 ↔ -7 / 3 < x ∧ x < 7 :=
sorry

end solve_abs_inequality_l319_319243


namespace smallest_palindrome_not_six_digit_palindrome_l319_319898

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ n / 100 = n % 10

noncomputable def smallest_three_digit_palindrome := 404

theorem smallest_palindrome_not_six_digit_palindrome (a b : ℕ) (h1 : 0 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
    (h3 : three_digit_palindrome (100 * a + 10 * b + a)) :
    100 * a + 10 * b + a = smallest_three_digit_palindrome → 
    ¬ is_palindrome (102 * (100 * a + 10 * b + a)) :=
by
  sorry

end smallest_palindrome_not_six_digit_palindrome_l319_319898


namespace vertical_asymptote_l319_319436

-- Define the given function
def given_function (x : ℝ) : ℝ := (3 * x - 1) / (6 * x + 4)

-- Define the condition of the vertical asymptote where the denominator is zero
def denominator (x : ℝ) : ℝ := 6 * x + 4

-- Statement to prove that the vertical asymptote occurs at x = -2/3
theorem vertical_asymptote : ∃ x : ℝ, denominator x = 0 ∧ x = -2 / 3 :=
by 
  use -2 / 3
  simp [denominator]
  split
  . rfl
  . sorry

end vertical_asymptote_l319_319436


namespace books_total_l319_319195

theorem books_total (books_Keith books_Jason : ℕ) (h_K : books_Keith = 20) (h_J : books_Jason = 21) : books_Keith + books_Jason = 41 :=
by
  rw [h_K, h_J]
  exact rfl

end books_total_l319_319195


namespace point_closer_to_center_probability_l319_319817

-- Define the conditions as constants
def outerRadius := 3.0
def innerRadius := 1.5

-- Define the areas of the circles
def area (r : ℝ) : ℝ := Real.pi * r ^ 2

-- Theorem to prove the probability condition
theorem point_closer_to_center_probability : 
  (area innerRadius) / (area outerRadius) = 1 / 4 :=
by
  sorry

end point_closer_to_center_probability_l319_319817


namespace number_line_y_l319_319096

theorem number_line_y (step_length : ℕ) (steps_total : ℕ) (total_distance : ℕ) (y_step : ℕ) (y : ℕ) 
    (H1 : steps_total = 6) 
    (H2 : total_distance = 24) 
    (H3 : y_step = 4)
    (H4 : step_length = total_distance / steps_total) 
    (H5 : y = step_length * y_step) : 
  y = 16 := 
  by 
    sorry

end number_line_y_l319_319096


namespace marketing_percentage_l319_319364

-- Define the conditions
variable (monthly_budget : ℝ)
variable (rent : ℝ := monthly_budget / 5)
variable (remaining_after_rent : ℝ := monthly_budget - rent)
variable (food_beverages : ℝ := remaining_after_rent / 4)
variable (remaining_after_food_beverages : ℝ := remaining_after_rent - food_beverages)
variable (employee_salaries : ℝ := remaining_after_food_beverages / 3)
variable (remaining_after_employee_salaries : ℝ := remaining_after_food_beverages - employee_salaries)
variable (utilities : ℝ := remaining_after_employee_salaries / 7)
variable (remaining_after_utilities : ℝ := remaining_after_employee_salaries - utilities)
variable (marketing : ℝ := 0.15 * remaining_after_utilities)

-- Define the theorem we want to prove
theorem marketing_percentage : marketing / monthly_budget * 100 = 5.14 := by
  sorry

end marketing_percentage_l319_319364


namespace intersection_area_of_two_rectangles_l319_319681

noncomputable def rectangle_intersection_area :
  {r1 r2 : ℝ} → (h1 w1 : ℝ) → (angle_r2 : ℝ) → (h2 w2 : ℝ) → ℝ
  | r1, r2, h1, w1, angle_r2, h2, w2 := 24

theorem intersection_area_of_two_rectangles :
  rectangle_intersection_area (1 : ℝ) (1 : ℝ) (4 : ℝ) (12 : ℝ) (30 : ℝ) (5 : ℝ) (10 : ℝ) = 24 :=
by
  sorry

end intersection_area_of_two_rectangles_l319_319681


namespace students_not_in_bio_or_chem_l319_319384

def num_students_not_in_bio_or_chem (total_students : ℕ) (bio_percentage chem_percentage : ℝ) :=
  total_students - (total_students * bio_percentage + total_students * chem_percentage)

theorem students_not_in_bio_or_chem (t : ℕ) (b c : ℝ) (h1 : t = 1500) (h2 : b = 0.35) (h3 : c = 0.25) :
  num_students_not_in_bio_or_chem t b c = 600 := by
  rw [h1, h2, h3]
  have h4 : num_students_not_in_bio_or_chem 1500 0.35 0.25 = 1500 - (1500 * 0.35 + 1500 * 0.25) := rfl
  rw h4
  norm_num
  exact rfl

end students_not_in_bio_or_chem_l319_319384


namespace tan_B_eq_one_third_l319_319187

theorem tan_B_eq_one_third
  (A B : ℝ)
  (h1 : Real.cos A = 4 / 5)
  (h2 : Real.tan (A - B) = 1 / 3) :
  Real.tan B = 1 / 3 := by
  sorry

end tan_B_eq_one_third_l319_319187


namespace find_m_l319_319605

variables (m : ℝ)

def vec_a := (m, 1 : ℝ)
def vec_b := (1, 2 : ℝ)

noncomputable def dot_product (u v : ℝ × ℝ) :=
u.1 * v.1 + u.2 * v.2

noncomputable def norm_squared (u : ℝ × ℝ) :=
u.1 ^ 2 + u.2 ^ 2

theorem find_m (h : norm_squared (vec_a m + vec_b) = norm_squared (vec_a m) + norm_squared (vec_b)) :
  m = -2 :=
sorry

end find_m_l319_319605


namespace units_digit_13_times_41_l319_319072

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_13_times_41 :
  units_digit (13 * 41) = 3 :=
sorry

end units_digit_13_times_41_l319_319072


namespace range_of_m_l319_319156

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  ¬(∀ x : ℝ, (x < m - 1 ∨ x > m + 1) ↔ (x^2 - 2*x - 3 > 0)) 
  ↔ 0 ≤ m ∧ m ≤ 2 :=
by 
  sorry

end range_of_m_l319_319156


namespace sum_of_sequence_l319_319862

noncomputable theory
open_locale big_operators

lemma arithmetic_sequence_property :
  (∀ n : ℕ, n > 0 → 2 * n - 1 = 2 - 1 + 3 * (n - 1)) :=
sorry

theorem sum_of_sequence (n : ℕ) :
  let a : ℕ → ℕ := λ n, 2 * n - 1,
      b : ℕ → ℕ := λ n, 3 ^ n in
  ((∑ i in finset.range n, (i + 1) * 3^(i + 1))
    = (3 + (2 * n - 1) * 3^(n + 1)) / 4) :=
sorry

end sum_of_sequence_l319_319862


namespace number_of_integers_l319_319142

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end number_of_integers_l319_319142


namespace brownies_leftover_l319_319607

def initial_brownies : ℕ := 16
def percent_children_ate : ℝ := 0.25
def percent_family_ate : ℝ := 0.5
def lorraine_ate : ℕ := 1

theorem brownies_leftover :
  let children_ate := (percent_children_ate * initial_brownies).toNat,
      remaining_after_children := initial_brownies - children_ate,
      family_ate := (percent_family_ate * remaining_after_children).toNat,
      remaining_after_family := remaining_after_children - family_ate
  in remaining_after_family - lorraine_ate = 5 := sorry

end brownies_leftover_l319_319607


namespace calculate_rental_hours_l319_319852

theorem calculate_rental_hours (cost_first_hour : ℕ) (cost_additional_hour : ℕ) (total_cost : ℕ) :
  cost_first_hour = 25 → 
  cost_additional_hour = 10 → 
  total_cost = 125 → 
  ∃ number_of_hours_rented : ℕ, number_of_hours_rented = 11 :=
begin
  intros h1 h2 h3,
  have h : 10 * (total_cost - cost_first_hour) / cost_additional_hour + 1 = 11,
  {
    rw [h1, h2, h3],
    simp,
  },
  use 11,
  exact h,
end

end calculate_rental_hours_l319_319852


namespace determine_a_divides_polynomial_l319_319469

noncomputable def polynomial_division_condition (a : ℕ) : Prop :=
  ∃ q : ℕ → ℤ, (∀ x : ℕ, x.13 + x + 90 = (x ^ 2 - x + a) * q x)

theorem determine_a_divides_polynomial :
  ∃ a : ℕ, a ≠ 0 ∧ polynomial_division_condition a ∧ a = 2 :=
begin
  sorry
end

end determine_a_divides_polynomial_l319_319469


namespace max_n_permutation_l319_319586

theorem max_n_permutation :
  ∃ (n : ℕ), (∀ a : Fin 17 → ℕ, (Multiset.Equiv (Finset.univ : Finset (Fin 17)) ↑(Multiset.range 17.succ)) ∧
    (List.prod ((List.init (toList a) ++ [List.head (toList a)]).zipWith(λ a b, a - b) ((List.tail (toList a) ++ [List.head (toList a)])) = n ^ 17)) → n ≤ 6) :=
sorry

end max_n_permutation_l319_319586


namespace isosceles_triangle_base_angles_l319_319973

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end isosceles_triangle_base_angles_l319_319973


namespace smallest_n_l319_319453

-- Definitions given in the conditions
def S (n : ℕ) (a : ℕ → ℝ) := 3 * a n - 2
def T (n : ℕ) (a : ℕ → ℝ) := ∑ k in finset.range (n + 1), (k + 1 : ℕ) * a (k + 1)

-- Hypothesis that sequence {an} satisfies Sn = 3*an - 2 and Tn is the sum of {nan}
theorem smallest_n (a : ℕ → ℝ)
  (hS : ∀ n, S n a = 3 * a n - 2)
  (hT : ∀ n, T n a = ∑ k in finset.range (n + 1), (k + 1 : ℕ) * a (k + 1)) :
  ∃ n : ℕ, T n a > 100 ∧ ∀ m : ℕ, T m a > 100 → n ≤ m := 
begin
  sorry
end

end smallest_n_l319_319453


namespace num_not_factorial_tails_lt_5000_l319_319049

-- Definitions based on conditions from step a)
def f (m : ℕ) : ℕ :=
  ∑ k in (finset.range (nat.log 5 m).succ), m / 5^k

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, f m = n

def num_not_factorial_tails (bound : ℕ) : ℕ :=
  bound - (finset.range bound).filter (λ n, ∃ m, f m = n).card

-- The theorem as per conditions and the correct answer from steps a) and b)
theorem num_not_factorial_tails_lt_5000 : num_not_factorial_tails 5000 = 3751 :=
sorry

end num_not_factorial_tails_lt_5000_l319_319049


namespace evaluate_expression_l319_319885

theorem evaluate_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x ^ 2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x ^ 2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x ^ 2 + 3 * x - 5) / ((x + 2) * (x - 3)) :=
by
  sorry

end evaluate_expression_l319_319885


namespace find_y_given_area_l319_319277

-- Define the problem parameters and conditions
namespace RectangleArea

variables {y : ℝ} (y_pos : y > 0)

-- Define the vertices, they can be expressed but are not required in the statement
def vertices := [(-2, y), (8, y), (-2, 3), (8, 3)]

-- Define the area condition
def area_condition := 10 * (y - 3) = 90

-- Lean statement proving y = 12 given the conditions
theorem find_y_given_area (y_pos : y > 0) (h : 10 * (y - 3) = 90) : y = 12 :=
by
  sorry

end RectangleArea

end find_y_given_area_l319_319277


namespace factor_expression_l319_319057

theorem factor_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) :=
by
  sorry

end factor_expression_l319_319057


namespace sequence_property_l319_319282

theorem sequence_property : 
  (∀ (a : ℕ → ℝ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (2 * a n) / n) → a 200 = 40200) :=
by
  sorry

end sequence_property_l319_319282


namespace total_distance_between_foci_l319_319082

theorem total_distance_between_foci (a b : ℝ) (h_a : a = 4) (h_b : b = 3) : 2 * real.sqrt (a^2 - b^2) = 2 * real.sqrt 7 :=
by
  rw [h_a, h_b]
  sorry

end total_distance_between_foci_l319_319082


namespace eval_expr_at_neg_one_l319_319632

theorem eval_expr_at_neg_one : 
  (let x := -1 in
   (x-3)/(2*x-4) / ((5/(x-2)) - x - 2) = -1/4) :=
by
  sorry

end eval_expr_at_neg_one_l319_319632


namespace wang_payment_correct_l319_319367

noncomputable def first_trip_payment (x : ℝ) : ℝ := 0.9 * x
noncomputable def second_trip_payment (y : ℝ) : ℝ := 300 * 0.9 + (y - 300) * 0.8

theorem wang_payment_correct (x y: ℝ) 
  (cond1: 0.1 * x = 19)
  (cond2: (x + y) - (0.9 * x + ((y - 300) * 0.8 + 300 * 0.9)) = 67) :
  first_trip_payment x = 171 ∧ second_trip_payment y = 342 := 
by
  sorry

end wang_payment_correct_l319_319367


namespace find_exponent_l319_319339

theorem find_exponent (e : ℝ) : (1 / 5) ^ e * (1 / 4) ^ 18 = 1 / (2 * 10 ^ 35) → e = 35 :=
by {
  sorry
}

end find_exponent_l319_319339


namespace greatest_multiple_l319_319722

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319722


namespace minimum_number_of_rows_l319_319818

-- Define the conditions stated in the problem
variables (n : ℕ) (C : ℕ → ℕ)
-- Constraints for the problem
variable (hC : ∀ i, 1 ≤ C i ∧ C i ≤ 39)
variable (students_sum : (∑ i in range n, C i) = 1990)

-- Define what the problem asks to prove
def minimum_rows_required : Prop :=
  ∃ rows : ℕ, rows ≤ 12 ∧ ∀ (seating : ℕ → ℕ),
    (∀ i, seating i < rows) → 
    (∀ i j, seating i ≠ seating j → C i + C j > 199) → 
    ∑ i in range n, seating i / 199 < rows

theorem minimum_number_of_rows : minimum_rows_required n C hC students_sum := 
  sorry

end minimum_number_of_rows_l319_319818


namespace possible_values_ceil_square_l319_319510

noncomputable def num_possible_values (x : ℝ) (hx : ⌈x⌉ = 12) : ℕ := 23

theorem possible_values_ceil_square (x : ℝ) (hx : ⌈x⌉ = 12) :
  let n := num_possible_values x hx in n = 23 :=
by
  let n := num_possible_values x hx
  exact rfl

end possible_values_ceil_square_l319_319510


namespace parallel_and_orthogonal_dot_product_zero_l319_319118

variables {V : Type*} [inner_product_space ℝ V] 
variables (a b c : V)

theorem parallel_and_orthogonal_dot_product_zero (h1 : ∃ k : ℝ, a = k • b) (h2 : ⟪b, c⟫ = 0) :
  ⟪a + b, c⟫ = 0 :=
sorry

end parallel_and_orthogonal_dot_product_zero_l319_319118


namespace problem_l319_319779

noncomputable def x (a : ℝ) : ℝ := 5/2 / a * 5/2 / (5/2 * a / 5/2)

theorem problem : ∃ x : ℝ, (x (x) = 25) → (x = 1/2) := 
by
  sorry

end problem_l319_319779


namespace num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l319_319148

/-- The number of positive integers n such that 300 < n^2 < 1200 is 17. -/
theorem num_positive_integers_between_300_and_1200 (n : ℕ) :
  (300 < n^2 ∧ n^2 < 1200) ↔ n ∈ {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34} :=
by {
  sorry
}

/-- There are 17 positive integers n such that 300 < n^2 < 1200. -/
theorem count_positive_integers_between_300_and_1200 :
  fintype.card {n : ℕ // 300 < n^2 ∧ n^2 < 1200} = 17 :=
by {
  sorry
}

end num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l319_319148


namespace find_f_neg6_log2_5_l319_319488

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log2(2 - x)
  else Real.pow 2 x

theorem find_f_neg6_log2_5 : f (-6) + f (Real.log2 5) = 9 := by
  sorry

end find_f_neg6_log2_5_l319_319488


namespace problem1_tangent_line_problem2_inequality_l319_319124

-- Definition for problem 1: tangent line
def f1 (x : ℝ) : ℝ := (1 / 2) * x - Real.exp x

noncomputable def tangent_line_equation (x : ℝ) : String := 
  let slope := (1 / 2) - Real.exp 1
  let point := (1, (1 / 2) - Real.exp 1)
  s!"{slope} * (x - {point.1}) + {point.2 - slope * point.1}"

theorem problem1_tangent_line :
  tangent_line_equation 1 = "((1 / 2) - Real.exp 1) * (x - 1) + (1 / 2 - Real.exp 1)" := 
by
  sorry

-- Definition for problem 2: inequality proof
def f2 (a x : ℝ) := a * x - Real.exp x

theorem problem2_inequality (a : ℝ) (h : 1 ≤ a ∧ a ≤ Real.exp 1 + 1) :
  ∀ x : ℝ, f2 a x ≤ x := 
by
  sorry

end problem1_tangent_line_problem2_inequality_l319_319124


namespace number_of_inverses_mod_11_l319_319960

theorem number_of_inverses_mod_11 : 
  (Finset.filter (λ n : ℕ, Nat.coprime n 11) (Finset.range 11)).card = 10 := 
by
  sorry

end number_of_inverses_mod_11_l319_319960


namespace find_a8_expansion_l319_319087

noncomputable def binomial (n k : ℕ) : ℕ := 
(n.choose k)

theorem find_a8_expansion : 
  let a_8 := 4 * binomial 10 8 in
  a_8 = 180 := 
by {
  sorry
}

end find_a8_expansion_l319_319087


namespace count_inverses_modulo_11_l319_319958

theorem count_inverses_modulo_11 : 
  let prime := 11
  let set_of_integers := {x | 1 ≤ x ∧ x ≤ 10}
  card {x ∈ set_of_integers | gcd x prime = 1} = 10 := 
by
  sorry

end count_inverses_modulo_11_l319_319958


namespace set_D_is_empty_l319_319379

theorem set_D_is_empty :
  {x : ℝ | x > 6 ∧ x < 1} = ∅ :=
by
  sorry

end set_D_is_empty_l319_319379


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319710

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319710


namespace greatest_multiple_l319_319729

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319729


namespace largest_digit_for_divisibility_by_6_l319_319311

theorem largest_digit_for_divisibility_by_6 :
  ∃ N : ℕ,
    N ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (57890 + N) % 6 = 0 ∧
    (∀ M : ℕ, M ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 5789 * 10 + M % 6 = 0 → M ≤ N) :=
begin
  sorry
end

end largest_digit_for_divisibility_by_6_l319_319311


namespace relay_race_time_reduction_l319_319900

theorem relay_race_time_reduction
    (T T1 T2 T3 T4 T5 : ℝ)
    (h1 : T1 = 0.1 * T)
    (h2 : T2 = 0.2 * T)
    (h3 : T3 = 0.24 * T)
    (h4 : T4 = 0.3 * T)
    (h5 : T5 = 0.16 * T) :
    ((T1 + T2 + T3 + T4 + T5) - (T1 + T2 + T3 + T4 + T5 / 2)) / (T1 + T2 + T3 + T4 + T5) = 0.08 :=
by
  sorry

end relay_race_time_reduction_l319_319900


namespace jordan_rectangle_width_l319_319032

theorem jordan_rectangle_width
  (w : ℝ)
  (len_carol : ℝ := 5)
  (wid_carol : ℝ := 24)
  (len_jordan : ℝ := 12)
  (area_carol_eq_area_jordan : (len_carol * wid_carol) = (len_jordan * w)) :
  w = 10 := by
  sorry

end jordan_rectangle_width_l319_319032


namespace difference_abs_value_l319_319405

noncomputable def base6_sum_correct (C D : ℕ) : ℕ :=
  if h1 : C < 6 ∧ D < 6 then
    abs (C - D) 
  else 
    0

theorem difference_abs_value {C D : ℕ} (h1 : C < 6) (h2 : D < 6) 
  (h3 : ((D * 6^2 + D * 6^1 + C) + (3 * 6^2 + 2 * 6^1 + D) + (C * 6^2 + 2 * 6^1 + 4)) = (C * 6^2 + 2 * 6^1 + 4 * 6^1 + 3))
  : base6_sum_correct C D = 1 := 
sorry

end difference_abs_value_l319_319405


namespace find_a_l319_319777

theorem find_a (a : ℝ) (h : 1 / Real.log 5 / Real.log a + 1 / Real.log 6 / Real.log a + 1 / Real.log 10 / Real.log a = 1) : a = 300 :=
sorry

end find_a_l319_319777


namespace borris_grapes_in_a_year_l319_319389

-- Define the initial conditions and question
variables (initial_grapes : ℝ) (increase_rate : ℝ) (time_periods_per_year : ℝ) 

-- Given conditions
def uses_90kg_per_6mo : Prop := initial_grapes = 90
def increase_20_percent : Prop := increase_rate = 0.20
def two_6mo_periods_per_year : Prop := time_periods_per_year = 2

-- The main theorem statement
theorem borris_grapes_in_a_year
  (h1 : uses_90kg_per_6mo)
  (h2 : increase_20_percent)
  (h3 : two_6mo_periods_per_year) :
  let increased_grapes := initial_grapes * (1 + increase_rate) in
  let grapes_needed_per_year := increased_grapes * time_periods_per_year in
  grapes_needed_per_year = 216 :=
by
  sorry

end borris_grapes_in_a_year_l319_319389


namespace symmetric_graph_function_inverse_l319_319934

theorem symmetric_graph_function_inverse :
  (∀ x, f x = logBase 2 (x / 2)) ↔ (∀ y, y = 2^(y + 1)) :=
by sorry

end symmetric_graph_function_inverse_l319_319934


namespace ellipse_equation_l319_319474

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (0, -1)
def F2 : ℝ × ℝ := (0, 1)

-- Define the distance functions
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Hypotheses
def foci_dist : ℝ := dist F1 F2
def arithmetic_mean (P : ℝ × ℝ) : Prop := 2 * foci_dist = dist P F1 + dist P F2

-- Statement of the problem in Lean
theorem ellipse_equation (h : ∀ P : ℝ × ℝ, arithmetic_mean P) :
  ∃ a b : ℝ, a = 2 ∧ b^2 = 3 ∧
  ∀ x y : ℝ, (x^2 / 3 + y^2 / 4 = 1) := 
sorry

end ellipse_equation_l319_319474


namespace number_of_ordered_pairs_l319_319139

theorem number_of_ordered_pairs : 
  let ordered_pairs := { (a, b) : ℝ × ℕ | 0 < a ∧ 2 ≤ b ∧ b ≤ 100 ∧ (log b a)^4 = log b (a^4)} in
  set.card ordered_pairs = 297 :=
sorry

end number_of_ordered_pairs_l319_319139


namespace add_two_inequality_l319_319506

theorem add_two_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
sorry

end add_two_inequality_l319_319506


namespace find_least_positive_x_l319_319894

theorem find_least_positive_x :
  ∃ x : ℕ, 0 < x ∧ (x + 5713) % 15 = 1847 % 15 ∧ x = 4 :=
by
  sorry

end find_least_positive_x_l319_319894


namespace sequence_maximum_value_l319_319182
-- Import the necessary library

theorem sequence_maximum_value (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) :
  (∀ n : ℕ, a_n n = 13 / (2 * n) - 1) →
  (∀ n : ℕ, S_n n = ∑ i in range (n + 1), a_n i) →
  (∃ n : ℕ, n = 6 ∧ (∀ m : ℕ, S_n m ≤ S_n n)) :=
by
  sorry

end sequence_maximum_value_l319_319182


namespace k_is_square_l319_319461

theorem k_is_square (a b : ℕ) (h_a : a > 0) (h_b : b > 0) (k : ℕ) (h_k : k > 0)
    (h : (a^2 + b^2) = k * (a * b + 1)) : ∃ (n : ℕ), n^2 = k :=
sorry

end k_is_square_l319_319461


namespace concat_five_digit_not_square_l319_319617

/-!
  On cards are written all five-digit numbers from 11111 to 99999,
  and then these cards are arranged in any order in a string.
  Prove that the resulting 444445-digit number cannot be a square number.
-/

theorem concat_five_digit_not_square : 
  (∀ (x : ℕ),
    (∃ f : Fin 444445 → ℕ, 
      (∀ i, 11111 ≤ f i ∧ f i ≤ 99999) ∧ 
      (∀ i j, i ≠ j → f i ≠ f j) ∧ 
      x = Nat.iterate (λ n, n * 10 + f (Fin.ofNat i)) 444445 0)) → 
    ¬ ∃ y, y^2 = x :=
by
  intros x hx
  sorry

end concat_five_digit_not_square_l319_319617


namespace circle_theorem_l319_319333

noncomputable def circle : Type := sorry -- We define a circle type for simplicity

variables {A B C D : circle} -- Points on the circle
variables (l : line) -- Tangent line to the circle at point A
variables (hB_farther : distance_from_line B l > distance_from_line C l) -- B is farther from line l than C
variables (line_AC : line) (parallel_line_to_l_through_B : line) -- Define lines

-- Intersection of line_AC with parallel_line_to_l_through_B at point D
axiom AC_intersects_B_parallel_at_D : intersection line_AC parallel_line_to_l_through_B = D

-- The theorem to be proved
theorem circle_theorem (h1 : is_tangent l A) (h2 : is_on_circle A) (h3 : is_on_circle B) (h4 : is_on_circle C)
  (h5 : is_on_circle D) (h6 : parallel parallel_line_to_l_through_B l) : 
  distance A B ^ 2 = distance A C * distance A D :=
sorry

end circle_theorem_l319_319333


namespace angle_C_eq_pi_div_3_l319_319166

theorem angle_C_eq_pi_div_3 (A B C : ℝ) (h : tan A + tan B + sqrt 3 = sqrt 3 * tan A * tan B) :
  C = π / 3 :=
sorry

end angle_C_eq_pi_div_3_l319_319166


namespace entrance_ticket_cost_l319_319801

theorem entrance_ticket_cost (num_students num_teachers ticket_cost : ℕ) (h1 : num_students = 20) (h2 : num_teachers = 3) (h3 : ticket_cost = 5) :
  (num_students + num_teachers) * ticket_cost = 115 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end entrance_ticket_cost_l319_319801


namespace parabola_circle_intersect_directrix_l319_319207

theorem parabola_circle_intersect_directrix (x_0 y_0 : ℝ) (hM : x_0^2 = 8*y_0) :
  let F := (0 : ℝ, 2 : ℝ)
  let directrix := λ y : ℝ, y = -2
  let radius := (y_0 + 2 : ℝ)
  let dist_to_directrix := 4
  intersects_directrix (circle_eqn : ℝ → ℝ → Prop) :
  y_0 > 2 :=
by
  let F : (ℝ × ℝ) := (0, 2)
  let directrix : ℝ → Prop := λ y, y = -2
  have radius : ℝ := y_0 + 2
  have dist_to_directrix : ℝ := 4
  sorry

end parabola_circle_intersect_directrix_l319_319207


namespace g_range_excludes_zero_l319_319401

noncomputable def g (x : ℝ) : ℤ :=
if x > -1 then ⌈1 / (x + 1)⌉
else ⌊1 / (x + 1)⌋

theorem g_range_excludes_zero : ¬ ∃ x : ℝ, g x = 0 := 
by 
  sorry

end g_range_excludes_zero_l319_319401


namespace curve_equation_l319_319130

def A : Matrix (Fin 2) (Fin 2) ℝ :=
![
  ![1, 0],
  ![0, Real.sqrt 2]
]

def T (v : Vector 2 ℝ) : Vector 2 ℝ := A.mul_vec v

def C1 (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

theorem curve_equation (x y: ℝ) (P: Vector 2 ℝ) (h: P = ![x, y]):
  (C1 (T P) (T P)) ↔ (x^2 + y^2 = 4) :=
sorry

end curve_equation_l319_319130


namespace inequality_solution_l319_319636

theorem inequality_solution (x : ℝ) :
  (x ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo (-1) ∞) ↔ (2 * x / (x + 1) / (x + 2) ≥ 0) :=
sorry

end inequality_solution_l319_319636


namespace base_angle_isosceles_triangle_l319_319969

theorem base_angle_isosceles_triangle (T : Triangle) (h_iso : is_isosceles T) (h_angle : internal_angle T = 110) :
  base_angle T = 35 :=
sorry

end base_angle_isosceles_triangle_l319_319969


namespace find_second_term_geometric_sequence_l319_319667

noncomputable def second_term_geometric_sequence (a r : ℝ) : ℝ :=
  a * r

theorem find_second_term_geometric_sequence:
  ∀ (a r : ℝ),
    a * r^2 = 12 →
    a * r^3 = 18 →
    second_term_geometric_sequence a r = 8 :=
by
  intros a r h1 h2
  sorry

end find_second_term_geometric_sequence_l319_319667


namespace decreasing_interval_l319_319655

noncomputable def func (x : ℝ) := 2 * x^3 - 6 * x^2 + 11

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv func x < 0 :=
by
  sorry

end decreasing_interval_l319_319655


namespace find_a3_find_sum_three_l319_319523

noncomputable def expansion (x : ℝ) := x^6 = a_0 + a_1 * (2 + x) + a_2 * (2 + x)^2 + a_3 * (2 + x)^3 + a_4 * (2 + x)^4 + a_5 * (2 + x)^5 + a_6 * (2 + x)^6

theorem find_a3 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ) :
  (x^6 = a_0 + a_1 * (2 + x) + a_2 * (2 + x)^2 + a_3 * (2 + x)^3 + a_4 * (2 + x)^4 + a_5 * (2 + x)^5 + a_6 * (2 + x)^6) →
  a_3 = -160 :=
sorry

theorem find_sum_three (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x y : ℝ) :
  (x^6 = a_0 + a_1 * (2 + x) + a_2 * (2 + x)^2 + a_3 * (2 + x)^3 + a_4 * (2 + x)^4 + a_5 * (2 + x)^5 + a_6 * (2 + x)^6) →
  x = -1 →
  y = -3 →
  1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 →
  729 = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 →
  a_1 + a_3 + a_5 = -364 :=
sorry

end find_a3_find_sum_three_l319_319523


namespace sum_of_possible_y_l319_319863

noncomputable def mean (l: List ℝ) : ℝ :=
  l.sum / l.length

def median (l: List ℝ) : ℝ :=
  let sorted := l.qsort (≤)
  if l.length % 2 = 1 then sorted.get! (l.length / 2)
  else (sorted.get! (l.length / 2) + sorted.get! (l.length / 2 - 1)) / 2

def mode (l: List ℝ) : ℝ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) l.head!

theorem sum_of_possible_y :
  let xs := [15, 3, 7, 3, 6, 3, 0] -- Placeholder for the list including y
  let y_vals := {y : ℝ | let l := xs.map (λ x => if x = 0 then y else x)
                         let μ := mean l
                         let η := median l
                         let ν := mode l
                         (ν = 3) ∧ (η <= 6) ∧ (η ≠ μ → μ - η = η - ν)}
  ∀ (y ∈ y_vals), ∃ (y ∈ {(26 : ℝ), (58 / 13)})

sorry

end sum_of_possible_y_l319_319863


namespace sum_of_complex_numbers_l319_319854

-- Definition of our complex numbers
def z1 : ℂ := (3 : ℂ) + (5 : ℂ) * I
def z2 : ℂ := (4 : ℂ) + (-7 : ℂ) * I
def z3 : ℂ := (-2 : ℂ) + (3 : ℂ) * I

-- The theorem we want to prove
theorem sum_of_complex_numbers : z1 + z2 + z3 = (5 : ℂ) + (1 : ℂ) * I := by
  sorry -- Proof is skipped

end sum_of_complex_numbers_l319_319854


namespace ceil_pow_sq_cardinality_l319_319521

noncomputable def ceil_pow_sq_values (x : ℝ) (h : 11 < x ∧ x ≤ 12) : ℕ :=
  ((Real.ceil(x^2)) - (Real.ceil(121)) + 1)

theorem ceil_pow_sq_cardinality :
  ∀ (x : ℝ), (11 < x ∧ x ≤ 12) → ceil_pow_sq_values x _ = 23 :=
by
  intro x hx
  let attrs := (11 < x ∧ x ≤ 12)
  sorry

end ceil_pow_sq_cardinality_l319_319521


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319760

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319760


namespace complex_solution_correct_l319_319119

noncomputable def complex_solution : ℂ :=
  2 + complex.I * (√3)

theorem complex_solution_correct (z : ℂ)
  (h1 : z * conj z - z - conj z = 3)
  (h2 : complex.arg(z - 1) = real.pi / 3) :
  z = complex_solution := 
sorry

end complex_solution_correct_l319_319119


namespace range_half_diff_l319_319462

theorem range_half_diff (α β : ℝ) (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) : 
    -π/2 ≤ (α - β) / 2 ∧ (α - β) / 2 < 0 := 
    sorry

end range_half_diff_l319_319462


namespace rate_of_interest_first_year_l319_319062

-- Define the conditions
def principal : ℝ := 9000
def rate_second_year : ℝ := 0.05
def total_amount_after_2_years : ℝ := 9828

-- Define the problem statement which we need to prove
theorem rate_of_interest_first_year (R : ℝ) :
  (principal + (principal * R / 100)) + 
  ((principal + (principal * R / 100)) * rate_second_year) = 
  total_amount_after_2_years → 
  R = 4 := 
by
  sorry

end rate_of_interest_first_year_l319_319062


namespace part1_tangent_line_part2_inequality_l319_319128

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def tangent_line (x : ℝ) := x

theorem part1_tangent_line : 
  ∃ (m b : ℝ), (∀ x : ℝ, f(x) = m * x + b) ∧ m = 1 ∧ b = 0 := by
  sorry

theorem part2_inequality (a : ℝ) (h : a ≥ 1) (x : ℝ) (hx : x > -1) : 
  f(x) ≤ a^2 * Real.exp x - a := by
  sorry

end part1_tangent_line_part2_inequality_l319_319128


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319696

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319696


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319764

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319764


namespace root_in_interval_l319_319651

-- Define the function f(x)
def f (x : ℝ) : ℝ := log (3 * x) + x

-- State the main theorem with conditions and goal
theorem root_in_interval :
  (∃ x : ℝ, f(x) = 5) →
  (∀ x y : ℝ, 0 < x → x < y → f(x) < f(y)) →
  ∀ x : ℝ, 0 < x → ∃ y ∈ Ioo (3 : ℝ) 4, f(y) = 5 :=
begin
  sorry,
end

end root_in_interval_l319_319651


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319712

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319712


namespace domain_of_log_2011x_minus_1_l319_319262

def domain_of_log (x : ℝ) : Prop :=
  ∃ y, y = 2011^x - 1 ∧ y > 0

theorem domain_of_log_2011x_minus_1 :
  {x : ℝ | domain_of_log x} = set.Ioi 0 := sorry

end domain_of_log_2011x_minus_1_l319_319262


namespace sum_non_palindrome_seven_steps_l319_319080

def reverse (n : ℕ) : ℕ := 
-- Placeholder function that computes the reverse of a number
sorry

def is_palindrome (n : ℕ) : Prop := 
-- Placeholder function that checks if a number is a palindrome
sorry

def seven_steps_to_palindrome (n : ℕ) : Prop :=
-- Base case when n becomes a palindrome
(nat.rec_on n 
  (λ n, false)
  (λ m IH, reverse m + m = n ∧ is_palindrome (reverse m + m))
)

theorem sum_non_palindrome_seven_steps :
  ∃ S : ℕ, S = (finset.range 150).filter (λ n, n ≥ 10 ∧ ¬is_palindrome n ∧ seven_steps_to_palindrome n).sum =
    -- Correct sum S, placeholder
    sorry :=
sorry

end sum_non_palindrome_seven_steps_l319_319080


namespace trigonometric_identity_l319_319928

theorem trigonometric_identity
  (θ : ℝ)
  (h : (2 + (1 / (Real.sin θ) ^ 2)) / (1 + Real.sin θ) = 1) :
  (1 + Real.sin θ) * (2 + Real.cos θ) = 4 :=
sorry

end trigonometric_identity_l319_319928


namespace balance_four_squares_l319_319570

variable (Triangle Square Circle : Type) [HasEquiv Triangle] [HasEquiv Square] [HasEquiv Circle]

axiom first_balance : ∀ (t : Triangle) (s : Square), t ≈ 2 * s
axiom second_balance : ∀ (t1 t2 : Triangle) (c : Circle), (t1 + t2) ≈ 3 * c

theorem balance_four_squares : ∀ (s1 s2 s3 s4 : Square) (c1 c2 c3 : Circle),
  (4 * s1) ≈ 3 * c1 :=
by
  intros
  sorry

end balance_four_squares_l319_319570


namespace correct_statement_l319_319965

-- Definitions of the terms to use as conditions
variables {m n : Line}
variables {α β : Plane}

-- State the conditions
hypothesis h1 : m ≠ n
hypothesis h2 : α ≠ β
hypothesis h3 : m ∥ α
hypothesis h4 : m ⊆ β
hypothesis h5 : α ∩ β = n

-- The theorem to prove
theorem correct_statement : m ∥ n :=
sorry

end correct_statement_l319_319965


namespace sum_of_squares_l319_319964

theorem sum_of_squares (a b : ℝ) (h1 : (a + b)^2 = 11) (h2 : (a - b)^2 = 5) : a^2 + b^2 = 8 := 
sorry

end sum_of_squares_l319_319964


namespace center_number_is_five_l319_319375

-- Defining the grid and the given conditions
def grid := fin 2 × fin 3 → ℕ

-- Given conditions
def is_diagonally_adjacent (a b : fin 2 × fin 3) : Prop :=
  (a.1 + 1 = b.1 ∧ (a.2 + 1 = b.2 ∨ a.2 = b.2 - 1)) ∨
  (a.1 = b.1 + 1 ∧ (a.2 + 1 = b.2 ∨ a.2 = b.2 - 1))

def valid_grid (f : grid) : Prop := 
  (∀ i j, i ≠ j → is_diagonally_adjacent i j → abs (f i - f j) = 1) ∧
  ((f (0, 0) + f (0, 2) = 6) ∨ (f (1, 0) + f (1, 2) = 6))

-- Prove the number at the center is 5
theorem center_number_is_five (f : grid) (h : valid_grid f) : f (0, 1) = 5 ∨ f (1, 1) = 5 :=
sorry

end center_number_is_five_l319_319375


namespace cos_of_angle_l319_319480

-- Define the coordinates of point P
def P : ℝ × ℝ := (-1, -Real.sqrt 2)

-- Define the magnitude of the vector OP
noncomputable def OP_magnitude : ℝ := Real.sqrt ((-1)^2 + (-(Real.sqrt 2))^2)

-- Define the cosine of the angle α
noncomputable def cos_alpha : ℝ := (-1) / OP_magnitude

-- State the theorem
theorem cos_of_angle (α : ℝ) (h : α = Real.arccos (cos_alpha)) :
  cos α = - Real.sqrt 3 / 3 :=
by
  sorry

end cos_of_angle_l319_319480


namespace greatest_multiple_of_5_and_6_under_1000_l319_319734

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319734


namespace avg_age_of_a_and_c_l319_319254

variable (A B C : ℕ)

def avg_age3 := (A + B + C) / 3
def avg_age2 := (A + C) / 2

theorem avg_age_of_a_and_c:
  avg_age3 A B C = 28 → B = 26 → avg_age2 A C = 29 :=
by {
  intro h1,
  intro h2,
  sorry -- proof omitted
}

end avg_age_of_a_and_c_l319_319254


namespace logs_quadratic_sum_l319_319153

theorem logs_quadratic_sum (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_roots : ∀ x, 2 * x^2 + 4 * x + 1 = 0 → (x = Real.log a) ∨ (x = Real.log b)) :
  (Real.log a)^2 + Real.log (a^2) + a * b = 1 / Real.exp 2 - 1 / 2 :=
by
  sorry

end logs_quadratic_sum_l319_319153


namespace area_of_IXJY_l319_319599

-- Define the rectangle ABCD and the points I, J, X, and Y.
structure Rectangle where
  A B C D : Point
  area : ℝ
  area_eq : area = 4

structure Point where
  x y : ℝ

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

def line_intersection (l1 l2 : Line) : Option Point :=
  sorry  -- Assuming this is a function that calculates intersection

-- Definitions based on problem conditions
def I (rect : Rectangle) : Point := midpoint rect.A rect.D
def J (rect : Rectangle) : Point := midpoint rect.B rect.C
def X (rect : Rectangle) : Option Point := line_intersection (line rect.A (J rect)) (line rect.B (I rect))
def Y (rect : Rectangle) : Option Point := line_intersection (line rect.D (J rect)) (line rect.C (I rect))

-- Define the Lean 4 statement
theorem area_of_IXJY (rect : Rectangle) : ∀ (I J X Y : Point), 
  I = midpoint rect.A rect.D → 
  J = midpoint rect.B rect.C →
  X = line_intersection (line rect.A J) (line rect.B I) →
  Y = line_intersection (line rect.D J) (line rect.C I) →
  quadrilateral_area I X J Y = 1 :=
by
  sorry

end area_of_IXJY_l319_319599


namespace polynomial_is_quadratic_l319_319273

theorem polynomial_is_quadratic (m : ℤ) (h : (m - 2 ≠ 0) ∧ (|m| = 2)) : m = -2 :=
by sorry

end polynomial_is_quadratic_l319_319273


namespace proof_of_equation_solution_l319_319239

noncomputable def solve_equation (x : ℝ) : Prop :=
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = Real.cbrt (2 + Real.sqrt x)) ↔
  x = (2389 + 375 * Real.sqrt 17) / 4

-- The statement to be proved
theorem proof_of_equation_solution :
  solve_equation ((2389 + 375 * Real.sqrt 17) / 4) := sorry

end proof_of_equation_solution_l319_319239


namespace sin_cos_theta_l319_319532

-- Definitions based on the given conditions
def circle1_eq (a : ℝ) : (ℝ × ℝ) → ℝ := 
  λ p, (p.1)^2 + (p.2)^2 + a * (p.1)

def circle2_eq (a : ℝ) (θ : ℝ) : (ℝ × ℝ) → ℝ := 
  λ p, (p.1)^2 + (p.2)^2 + 2 * a * (p.1) + (p.2) * (Real.tan θ)

def is_symmetric (p : ℝ × ℝ) : Prop :=
  2 * p.1 - p.2 - 1 = 0

-- Main theorem statement
theorem sin_cos_theta (a θ : ℝ) 
  (h1 : ∀ p, circle1_eq a p = 0 → is_symmetric p)
  (h2 : ∀ p, circle2_eq a θ p = 0 → is_symmetric p) :
  Real.sin θ * Real.cos θ = -2 / 5 := 
sorry

end sin_cos_theta_l319_319532


namespace greatest_multiple_l319_319725

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319725


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319715

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319715


namespace optimal_washing_effect_l319_319027

noncomputable def optimal_laundry_addition (x y : ℝ) : Prop :=
  (5 + 0.02 * 2 + x + y = 20) ∧
  (0.02 * 2 + x = (20 - 5) * 0.004)

theorem optimal_washing_effect :
  ∃ x y : ℝ, optimal_laundry_addition x y ∧ x = 0.02 ∧ y = 14.94 :=
by
  sorry

end optimal_washing_effect_l319_319027


namespace incorrect_statement_is_D_l319_319614

theorem incorrect_statement_is_D :
  (∀ a b : ℝ, a > b → (∀ c : ℝ, c < 0 → a * c < b * c ∧ a / c < b / c)) ∧ 
  (∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b → 2 * a * b / (a + b) < Real.sqrt(a * b)) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = p → x + y ≠ 2 * Real.sqrt(p)) ∧
  (∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b → Real.sqrt(a^2 + b^2) ≤ a + b) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = s → x * y ≠ 0) → False := 
by 
  sorry

end incorrect_statement_is_D_l319_319614


namespace prime_a_pow_n_minus_one_imp_a_eq_2_and_n_prime_l319_319597

theorem prime_a_pow_n_minus_one_imp_a_eq_2_and_n_prime
  (a n : ℤ) (h1 : a ≥ 2) (h2 : n ≥ 2) (h3 : Prime (a^n - 1)) :
  a = 2 ∧ Prime n := 
by 
  sorry

end prime_a_pow_n_minus_one_imp_a_eq_2_and_n_prime_l319_319597


namespace discriminant_condition_l319_319362

theorem discriminant_condition (a b c : ℝ) (h : 2 * a ≠ 0) :
  let Δ := (3 * b) ^ 2 - 4 * (2 * a) * (4 * c) in
  Δ = 25 ↔ a * c = (9 * b ^ 2 - 25) / 32 :=
by
  sorry

end discriminant_condition_l319_319362


namespace henry_jill_age_ratio_l319_319289

theorem henry_jill_age_ratio :
  ∀ (H J : ℕ), (H + J = 48) → (H = 29) → (J = 19) → ((H - 9) / (J - 9) = 2) :=
by
  intros H J h_sum h_henry h_jill
  sorry

end henry_jill_age_ratio_l319_319289


namespace greatest_multiple_l319_319723

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319723


namespace teacher_weight_l319_319328

theorem teacher_weight :
  ∀ (avg_weight_students : ℤ) (num_students : ℤ) (num_teacher : ℤ) (avg_weight_increase : ℤ),
  avg_weight_students = 35 →
  num_students = 24 →
  num_teacher = 1 →
  avg_weight_increase = 0.4 →
  let total_weight_students := avg_weight_students * num_students in
  let total_weight_with_teacher := (avg_weight_students + avg_weight_increase) * (num_students + num_teacher) in
  total_weight_with_teacher - total_weight_students = 45 :=
by
  intros
  let total_weight_students := avg_weight_students * num_students
  let total_weight_with_teacher := (avg_weight_students + avg_weight_increase) * (num_students + num_teacher)
  sorry

end teacher_weight_l319_319328


namespace interval_of_monotonic_increase_l319_319245

variables (ω φ : ℝ) (k : ℤ)
variable (f : ℝ → ℝ)

def is_function_stretched_and_shifted (f : ℝ → ℝ) : Prop :=
  ∃ ω φ, ω > 0 ∧ (-π / 2) ≤ φ ∧ φ < π / 2 ∧ f = λ x, sin(2 * x - π / 3)

def is_graph_becomes_cos (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, g = λ x, cos x ∧
    ∀ x, f x = g (x - (5 * π / 6))

theorem interval_of_monotonic_increase :
  (∃ k : ℤ, (λ x, sin (2 * x - π / 3)) = λ x, cos x) →
  ∃ k : ℤ, 
    ∀ x, k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 :=
sorry

end interval_of_monotonic_increase_l319_319245


namespace perpendicular_pairs_l319_319504

/-- Define the equations of the lines --/
def eq_a1 (x y : ℝ) := 2 * x + 3 * y - 7 = 0
def eq_a2 (x y : ℝ) := 3 * x - 2 * y = 0
def eq_b1 (x y : ℝ) := 5 * x - 2 * y + 1 = 0
def eq_b2 (x y : ℝ) := 4 * x + 10 * y - 1 = 0
def eq_c1 (x y : ℝ) := 6 * x - 4 * y + 7 = 0
def eq_c2 (x y : ℝ) := 8 * x - 12 * y - 1 = 0

/-- Define the slopes of the lines --/
noncomputable def slope (A B : ℝ) := - A / B

/-- Define the product of slopes for each pair of lines --/
noncomputable def prod_slope (A₁ B₁ A₂ B₂ : ℝ) := slope A₁ B₁ * slope A₂ B₂

/-- Prove the perpendicularity of the given pairs of lines --/
theorem perpendicular_pairs:
  (prod_slope 2 3 3 (-2) = -1) ∧
  (prod_slope 5 (-2) 4 10 = -1) ∧
  (prod_slope 6 (-4) 8 (-12) ≠ -1) := 
by sorry

end perpendicular_pairs_l319_319504


namespace ratio_XM_NZ_l319_319186

namespace Problem

variables {X Y Z : Type} [triangle XYZ]
variables (a b c : ℝ)
variables (XM MN NZ : ℝ)
variables (YM YN : line)
variables (trisects_angle : trisection YM YN angle_Y)

def triangle_XYZ := triangle X Y Z
def condition_xm := XM = a
def condition_mn := MN = b
def condition_nz := NZ = c

theorem ratio_XM_NZ (htrisect : trisects_angle YM YN angle_Y) :
  XM / NZ = a / c :=
sorry

end Problem

end ratio_XM_NZ_l319_319186


namespace plane_ratio_l319_319359

section

variables (D B T P : ℕ)

-- Given conditions
axiom total_distance : D = 1800
axiom distance_by_bus : B = 720
axiom distance_by_train : T = (2 * B) / 3

-- Prove the ratio of the distance traveled by plane to the whole trip
theorem plane_ratio :
  D = 1800 →
  B = 720 →
  T = (2 * B) / 3 →
  P = D - (T + B) →
  P / D = 1 / 3 := by
  intros h1 h2 h3 h4
  sorry

end

end plane_ratio_l319_319359


namespace existence_of_x0_y0_l319_319921

theorem existence_of_x0_y0 (p : ℕ) [fact (nat.prime p)] :
  (∃ x0 : ℤ, p ∣ (x0^2 - x0 + 3)) ↔ (∃ y0 : ℤ, p ∣ (y0^2 - y0 + 25)) :=
sorry

end existence_of_x0_y0_l319_319921


namespace not_possible_linear_poly_conditions_l319_319598

theorem not_possible_linear_poly_conditions (a b : ℝ):
    ¬ (abs (b - 1) < 1 ∧ abs (a + b - 3) < 1 ∧ abs (2 * a + b - 9) < 1) := 
by
    sorry

end not_possible_linear_poly_conditions_l319_319598


namespace sphere_volume_inside_cone_l319_319841

-- Define the volume calculation for a sphere given its radius
def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the conditions
variables (d : ℝ) (h : d = 24) (r : ℝ) (hr : r = 12)

-- Theorem stating that given these conditions, the volume of the sphere is 2304π cubic inches
theorem sphere_volume_inside_cone : volume_of_sphere r = 2304 * Real.pi :=
by
  rw [volume_of_sphere, hr]
  have : r = 12 := hr
  -- Substitute radius r with 12 in the volume formula
  have volume := calc
    (4/3) * Real.pi * (12: ℝ) ^ 3 = (4/3) * Real.pi * 1728  := by norm_num
                                ... = 2304 * Real.pi := by norm_num
  exact volume

end sphere_volume_inside_cone_l319_319841


namespace number_of_people_in_group_l319_319260

theorem number_of_people_in_group :
  ∃ (N : ℕ), (∀ (avg_weight : ℝ), 
  ∃ (new_person_weight : ℝ) (replaced_person_weight : ℝ),
  new_person_weight = 85 ∧ replaced_person_weight = 65 ∧
  avg_weight + 2.5 = ((N * avg_weight + (new_person_weight - replaced_person_weight)) / N) ∧ 
  N = 8) :=
by
  sorry

end number_of_people_in_group_l319_319260


namespace equation_of_circle_passing_through_point_length_of_chord_intersecting_circle_l319_319180

-- Part I: Proving the equation of the circle
theorem equation_of_circle_passing_through_point :
  ∀ (O M : ℝ × ℝ), O = (0, 0) → M = (1, sqrt 3) → (dist O M) = 2 → 
  ∀ (x y : ℝ), x^2 + y^2 = 4 := 
by 
  intros O M hO hM hdist x y 
  sorry

-- Part II: Proving the length of the chord created by the intersection
theorem length_of_chord_intersecting_circle :
  ∀ (O : ℝ × ℝ) (r : ℝ) (x y : ℝ), O = (0, 0) → r = 2 → x^2 + y^2 = r^2 → 
  ∀ (a b : ℝ), (a = sqrt 3) → (b = 1) → 
  (a * x + b * y = 2) → (abs ((a * 0 + b * 0 - 2) / sqrt (a^2 + b^2)) = 1) → 
  ∀ (d : ℝ), d = 1 → ∀ (chord_len : ℝ), (chord_len = 2 * sqrt (r^2 - d^2)) → 
  chord_len = 2 * sqrt 3 := 
by 
  intros O r x y hO hr hcircle a b ha hb hline hcenter_distance d hd chord_len hchord_len 
  sorry

end equation_of_circle_passing_through_point_length_of_chord_intersecting_circle_l319_319180


namespace ceil_square_values_l319_319511

theorem ceil_square_values (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, (∀ m : ℕ, m = n ↔ (121 < x^2 ∧ x^2 ≤ 144) ∧ (⌈x^2⌉ = m)) ∧ n = 23 :=
by
  sorry

end ceil_square_values_l319_319511


namespace area_of_tangent_circles_l319_319595

noncomputable def radius_and_area_difference (a b c : ℝ) (r R : ℝ) : ℝ :=
  let S := π * R^2 - π * r^2 in S -- Defining the area difference

theorem area_of_tangent_circles (AB AC BC a b c r R : ℝ)
  (h_AB : AB = 2)
  (h_AC : AC = 2)
  (h_BC : BC = 3)
  (h_eq1 : a + b = 2)
  (h_eq2 : a + c = 2)
  (h_eq3 : b + c = 3)
  (ha : a = 1/2)
  (hb : b = 3/2)
  (hc : c = 3/2)
  (h_r : r = 1/2 * (2 * Real.sqrt 7 - 5))
  (h_R : R = 1/2 * (2 * Real.sqrt 7 + 5)) :
  radius_and_area_difference a b c r R = 10 * Real.sqrt 7 * π := 
by 
  sorry

end area_of_tangent_circles_l319_319595


namespace average_of_class_l319_319792

noncomputable theory
open_locale classical

theorem average_of_class (total_students : ℕ) (top_scorers : ℕ) (top_score_marks : ℕ)
  (zero_scorers : ℕ) (zero_score_marks : ℕ) (remaining_students_average : ℕ) :
  total_students = 27 ∧ top_scorers = 5 ∧ top_score_marks = 95 ∧ zero_scorers = 3 ∧ zero_score_marks = 0 ∧ remaining_students_average = 45
  → (5 * 95 + 3 * 0 + 19 * 45) / 27 = 49.26 :=
begin
  sorry
end

end average_of_class_l319_319792


namespace trig_identity_l319_319320

open Real

theorem trig_identity :
  3.4173 * sin (2 * pi / 17) + sin (4 * pi / 17) - sin (6 * pi / 17) - (1/2) * sin (8 * pi / 17) =
  8 * (sin (2 * pi / 17))^3 * (cos (pi / 17))^2 :=
by sorry

end trig_identity_l319_319320


namespace sequence_geometric_and_sum_l319_319181

theorem sequence_geometric_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 4) 
  (h2 : ∀ n : ℕ, S n = a (n + 1) + n)
  (h3 : ∀ n : ℕ, n ≥ 2 → (a (n + 1) - 1) = 2 * (a n - 1)) 
  (h4 : ∀ n : ℕ, n ≥ 2 → a n = 2^(n-1) + 1)
  (h5 : ∀ n : ℕ, b n = (n * a n / (2^(n-1) + 1)) * (1/3)^n) 
  (h6 : ∀ n : ℕ, T n = (∑ i in range (n+1), b i)) :
  T n = (13 / 12) - ((2 * n + 3) / 4) * (1 / 3^n) :=
sorry

end sequence_geometric_and_sum_l319_319181


namespace double_of_quarter_of_4_percent_as_decimal_l319_319325

theorem double_of_quarter_of_4_percent_as_decimal : 
  let percent := λ (x : ℝ), x / 100
  let quarter := λ (x : ℝ), x / 4
  let double := λ (x : ℝ), x * 2
  percent (double (quarter 4)) = 0.02 :=
by
  sorry

end double_of_quarter_of_4_percent_as_decimal_l319_319325


namespace prob_both_tell_truth_l319_319525

variable (P_A P_B P_A_and_B : ℝ)

-- Hypotheses based on the problem conditions
def A_truth_prob := P_A = 0.85
def B_truth_prob := P_B = 0.60

-- Statement to be proved
theorem prob_both_tell_truth (hA : A_truth_prob) (hB : B_truth_prob) : 
  P_A_and_B = 0.85 * 0.60 := 
sorry

end prob_both_tell_truth_l319_319525


namespace solve_system_of_inequalities_l319_319244

theorem solve_system_of_inequalities (x : ℝ) 
  (h1 : -3 * x^2 + 7 * x + 6 > 0) 
  (h2 : 4 * x - 4 * x^2 > -3) : 
  -1/2 < x ∧ x < 3/2 :=
sorry

end solve_system_of_inequalities_l319_319244


namespace sequence_a_200_l319_319281

theorem sequence_a_200 :
  let a : ℕ → ℕ := λ n, if n = 1 then 2 else a (n - 1) + 2 * a (n - 1) / (n - 1)
  in a 200 = 40200 := by
  let a : ℕ → ℕ
  sorry

end sequence_a_200_l319_319281


namespace length_BE_is_4_l319_319571

-- Conditions defined
structure Square where
  A B C D : ℝ × ℝ -- Coordinates of vertices
  side_length : ℝ -- Side length of the square
  is_square : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2 ∧
              (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2 ∧
              (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2 ∧
              (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2 ∧
              -- Angles are 90 degrees
              (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 = (2 * side_length) ^ 2

-- Define larger square ABCD with side length 4
def largeSquare : Square :=
{ A := (0, 0),
  B := (4, 0),
  C := (4, 4),
  D := (0, 4),
  side_length := 4,
  is_square := by {
    --Calculations checking properties of the square.
    simp [Real.sqrt, sq];
    sorry
} }

-- Define smaller square EFGH with side_length 2 sharing vertex A with largeSquare and inside it
def smallSquare : Square :=
{ A := largeSquare.A,
  B := (2, 2),
  C := (2, 4),
  D := (0, 2),
  side_length := 2,
  is_square := by {
    --Calculations checking properties of the square.
    simp [Real.sqrt, sq];
    sorry
} }

-- Point E lies on side GH of the smaller square (x, 4) where x ranges from 2 to 4
variable (E : ℝ × ℝ) (hE : 2 ≤ E.1 ∧ E.1 ≤ 4 ∧ E.2 = 4)

theorem length_BE_is_4 : (real.dist largeSquare.B E) = 4 := 
by {
  -- Proof showing the distance from B to E is 4
  have h₁ : largeSquare.B = (4, 0) := rfl,
  have h₂ : E = (4, 4) := by {
    simp [hE];
    sorry
  },
  simp [real.dist, h₁, h₂],
  sorry
}

end length_BE_is_4_l319_319571


namespace num_ordered_pairs_satisfying_equation_l319_319140

theorem num_ordered_pairs_satisfying_equation:
  { pairs : ℕ |
    ∃ x y : ℤ, x^4 + x^2 + y^2 = 2 * y + 1 
  }.card = 2 :=
sorry

end num_ordered_pairs_satisfying_equation_l319_319140


namespace part_a_part_b_l319_319219

open Matrix

-- Conditions
variables {R : Type*} [CommRing R]

-- Part (a)
theorem part_a (A B : Matrix (Fin 2) (Fin 2) R) (h : (A - B) * (A - B) = 0) :
  det (A * A - B * B) = (det A - det B) * (det A - det B) :=
sorry

-- Part (b)
theorem part_b (A B : Matrix (Fin 2) (Fin 2) R) (h : (A - B) * (A - B) = 0) :
  det (A * B - B * A) = 0 ↔ det A = det B :=
sorry

end part_a_part_b_l319_319219


namespace PR_length_right_triangle_l319_319565

theorem PR_length_right_triangle
  (P Q R : Type)
  (cos_R : ℝ)
  (PQ PR : ℝ)
  (h1 : cos_R = 5 * Real.sqrt 34 / 34)
  (h2 : PQ = Real.sqrt 34)
  (h3 : cos_R = PR / PQ) : PR = 5 := by
  sorry

end PR_length_right_triangle_l319_319565


namespace sequence_a_200_l319_319280

theorem sequence_a_200 :
  let a : ℕ → ℕ := λ n, if n = 1 then 2 else a (n - 1) + 2 * a (n - 1) / (n - 1)
  in a 200 = 40200 := by
  let a : ℕ → ℕ
  sorry

end sequence_a_200_l319_319280


namespace solve_sqrt_cubic_equation_l319_319242

theorem solve_sqrt_cubic_equation (x : ℝ) 
    (h : sqrt (2 + sqrt (3 + sqrt x)) = real.cbrt (2 + sqrt x)) :
    x = (real.cbrt (2 + sqrt x) - 2) ^ 2 :=
by
  sorry

end solve_sqrt_cubic_equation_l319_319242


namespace find_n_l319_319778

theorem find_n :
  ∃ n : ℤ, 10^n = 10^(-8) * (Real.sqrt (10^50 / (10^(-4)))) ∧ n = 19 :=
by
  sorry

end find_n_l319_319778


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319702

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319702


namespace infection_equation_correct_l319_319174

theorem infection_equation_correct (x : ℝ) :
  1 + x + x * (x + 1) = 196 :=
sorry

end infection_equation_correct_l319_319174


namespace chi_square_association_l319_319566

theorem chi_square_association (k : ℝ) :
  (k > 3.841 → (∃ A B, A ∧ B)) ∧ (k ≤ 2.076 → (∃ A B, ¬(A ∧ B))) :=
by
  sorry

end chi_square_association_l319_319566


namespace fraction_of_girls_at_concert_l319_319383

theorem fraction_of_girls_at_concert
  (b g : ℕ) 
  (h₁ : g = b)
  (h₂ : g > 0)
  (h₃ : b > 0)
  (fraction_girls : \(\frac{5}{6}\)) 
  (fraction_boys : \(\frac{3}{4}\))
  : (\(\frac{5}{6}\ g) / (\(\frac{5}{6}\ g + \(\frac{3}{4}\ b)) = \(\frac{10}{19}\)) :=
by
  sorry

end fraction_of_girls_at_concert_l319_319383


namespace maximum_friends_l319_319222

theorem maximum_friends 
    (sandwich_price roll_price pastry_price juice_pack_price small_soda_price large_soda_price : ℝ)
    (total_budget min_food_spend : ℝ)
    (validations : sandwich_price = 0.80 ∧ roll_price = 0.60 ∧ pastry_price = 1.00 ∧
                   juice_pack_price = 0.50 ∧ small_soda_price = 0.75 ∧ large_soda_price = 1.25 ∧
                   total_budget = 12.50 ∧ min_food_spend = 10) :
    ∃ max_friends, max_friends = 10 := 
by
  have cheapest_combination := roll_price + juice_pack_price
  have cost_10_people := 10 * cheapest_combination
  have _ : (cheapest_combination = 1.10 ∧ cost_10_people = 11) := sorry
  have _ : valid_budget := cost_10_people ≤ total_budget := sorry
  have _ : valid_food_spend := 10 * roll_price ≥ min_food_spend := sorry
  use 10
  have proof_max := (valid_budget ∧ valid_food_spend) := sorry
  exact proof_max

end maximum_friends_l319_319222


namespace find_c_plus_inv_b_l319_319247

variable (a b c : ℝ)

# Checking the positive conditions for a, b, and c
variable (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)

# Given conditions: abc = 1, a + 1/c = 8, b + 1/a = 20
variable (h1 : a * b * c = 1)
variable (h2 : a + 1 / c = 8)
variable (h3 : b + 1 / a = 20)

theorem find_c_plus_inv_b : c + 1 / b = 10 / 53 := by
  sorry

end find_c_plus_inv_b_l319_319247


namespace horse_food_per_day_l319_319385

-- Definitions of conditions in the problem
def ratio_sheep_horses : Nat := 7
def number_of_sheep : Nat := 8
def total_horse_food : Nat := 12880

-- Main theorem to prove
theorem horse_food_per_day :
  let number_of_horses := ratio_sheep_horses * number_of_sheep in
  total_horse_food / number_of_horses = 230 :=
sorry

end horse_food_per_day_l319_319385


namespace ellipse_equation_line_HN_fixed_point_l319_319112

open Real

-- Given conditions
def center (E : Type) : Point := (0, 0)
def axes_of_symmetry (E : Type) : Prop := true -- x-axis and y-axis are assumed
def passes_through (E : Type) (p q r : Point) := p ∈ E ∧ q ∈ E ∧ r ∈ E
def equation (E : Type) (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 4) = 1

def point := (ℝ × ℝ) -- Defining a type for point

-- Definitions for other points and line conditions
def A : point := (0, -2)
def B : point := (3/2, -1)
def P : point := (1, -2)

def intersects (E : Type) (P : point) : Point := sorry
def line_passes_through (P Q : point) : Prop := sorry
def M (E : Type) (P : point) : Point := intersects E P
def N (E : Type) (P : point) : Point := intersects E P

def parallel (P : point) : Prop := sorry -- line parallel to x-axis passing through P
def T (A B M: point) : Point := sorry -- intersection of line through M parallel to x-axis and line segment AB
def H (M T: point) : point := sorry -- point satisfying MT = TH

-- Proof Problem 1: Equation of the Ellipse
theorem ellipse_equation : ∀ E, 
  center E = (0, 0) →
  axes_of_symmetry E →
  passes_through E A B →
  ∃ x y, equation E x y :=
by
  intro E
  assume h1 h2 h3
  sorry

-- Proof Problem 2: Line HN passes through fixed point
theorem line_HN_fixed_point : 
  ∀ E, 
  center E = (0, 0) →
  axes_of_symmetry E →
  passes_through E A B →
  ∃ (x : ℝ) (y : ℝ), 
  ∀ (P : point), 
  intersects E P →
  let M := M E P, 
      N := N E P, 
      T := T A B M,
      H := H M T in 
  line_passes_through (1,-2) N
  →
  (HN_line : y = 2 + 2 * sqrt 6/3 * x - 2)
  ∧ line_passes_through (0, -2) H :=
by
  intro E
  assume h1 h2 h3
  sorry

end ellipse_equation_line_HN_fixed_point_l319_319112


namespace valid_values_of_l_l319_319584

def g (l x : ℝ) : ℝ := (3 * x + 4) / (l * x - 3)

theorem valid_values_of_l (l : ℝ) : 
  (∀ x, g (g l x) = x) ↔ (l ∈ set.Ioo ⊤ (-9 / 4) ∪ set.Ioo (-9 / 4) ⊤) :=
by
  sorry

end valid_values_of_l_l319_319584


namespace ellipse_standard_eq_AB_length_l319_319101

noncomputable def ellipse_equation : Prop :=
  let F1 := (-2 * Real.sqrt 2, 0)
  let F2 := (2 * Real.sqrt 2, 0)
  let major_axis_length := 6
  let a := 3
  let b := Real.sqrt (a^2 - (2 * Real.sqrt 2)^2)
  let ellipse := (x^2 / a^2) + (y^2 / b^2) = 1
  ellipse = (x^2 / 9) + (y^2 / 1) = 1

noncomputable def line_segment_length_AB : Prop :=
  let P := (0, 2)
  let slope := 1
  let line := fun x => x + 2
  let intersection_pts := [(x1, y1), (x2, y2)] -- points of intersection
  let AB_len := Real.sqrt (1 + slope^2) * (Real.sqrt ((18^2 / 5^2) - (4 * 27 / 10)))
  AB_len = 6 * Real.sqrt 3 / 5

theorem ellipse_standard_eq :
  ellipse_equation :=
sorry

theorem AB_length :
  line_segment_length_AB :=
sorry

end ellipse_standard_eq_AB_length_l319_319101


namespace sheets_per_class_per_day_l319_319055

theorem sheets_per_class_per_day
  (weekly_sheets : ℕ)
  (school_days_per_week : ℕ)
  (num_classes : ℕ)
  (h1 : weekly_sheets = 9000)
  (h2 : school_days_per_week = 5)
  (h3 : num_classes = 9) :
  (weekly_sheets / school_days_per_week) / num_classes = 200 :=
by
  sorry

end sheets_per_class_per_day_l319_319055


namespace greatest_multiple_l319_319721

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319721


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319719

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319719


namespace base_angle_isosceles_triangle_l319_319971

theorem base_angle_isosceles_triangle (T : Triangle) (h_iso : is_isosceles T) (h_angle : internal_angle T = 110) :
  base_angle T = 35 :=
sorry

end base_angle_isosceles_triangle_l319_319971


namespace count_x_squared_plus_6x_plus_9_in_interval_l319_319435

def count_valid_integers (f : ℤ → ℤ) (a b : ℤ) : ℤ :=
  (List.filter (λ x, a < f x ∧ f x < b) (List.range' 1 ((b : ℕ) + 1))).length

theorem count_x_squared_plus_6x_plus_9_in_interval :
  count_valid_integers (λ x, (x + 3) ^ 2) 50 200 = 7 := by
  sorry

end count_x_squared_plus_6x_plus_9_in_interval_l319_319435


namespace ellipse_equation_and_line_intersection_unique_l319_319456

-- Definitions from conditions
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1
def line (x0 y0 x y : ℝ) : Prop := 3*x0*x + 4*y0*y - 12 = 0
def on_ellipse (x0 y0 : ℝ) : Prop := ellipse x0 y0

theorem ellipse_equation_and_line_intersection_unique :
  ∀ (x0 y0 : ℝ), on_ellipse x0 y0 → ∀ (x y : ℝ), line x0 y0 x y → ellipse x y → x = x0 ∧ y = y0 :=
by
  sorry

end ellipse_equation_and_line_intersection_unique_l319_319456


namespace gain_percentage_is_20_l319_319000

-- Definitions for the conditions
def original_cost_price : ℝ := 51724.14
def loss_percentage : ℝ := 13 / 100
def selling_price_friend : ℝ := 54000

-- The statement to prove
theorem gain_percentage_is_20 :
  let cost_price_friend := original_cost_price * (1 - loss_percentage) in
  let gain := selling_price_friend - cost_price_friend in
  let gain_percentage := (gain / cost_price_friend) * 100 in
  gain_percentage = 20 :=
by
  sorry

end gain_percentage_is_20_l319_319000


namespace compatriots_sequence_sum_or_double_l319_319380

theorem compatriots_sequence_sum_or_double :
  ∃ (x ∈ (finset.range 1978) ∪ {1978}),
    (∃ (a b ∈ (finset.range 1978) ∪ {1978}), x = a + b) 
    ∨ (∃ (c ∈ (finset.range 1978) ∪ {1978}), x = 2 * c) :=
by
  sorry

end compatriots_sequence_sum_or_double_l319_319380


namespace window_width_is_thirty_l319_319618

noncomputable def window_width : ℕ :=
  let x := 6 in -- Assume x = width of each pane in inches.
  3 * x + 12   -- Total width with borders.

theorem window_width_is_thirty :
  let width_ratio := 3 in
  let panes_across := 3 in
  let border_width := 3 in
  let total_border_width := border_width * (panes_across + 1) in
  3 * 6 + total_border_width = 30 := 
by simp [mul_add, add_assoc]; rfl

end window_width_is_thirty_l319_319618


namespace age_difference_l319_319602

noncomputable def R (a c: ℤ): ℤ := a - c

theorem age_difference (a b c d: ℤ) : 
  (a + b + 10 = b + c + 20) →
  (c + d = a + d - 12) →
  R a c = 12 := 
by
  intros h1 h2
  have h3: a - c = 10, by
  {
    rw [add_assoc, add_right_comm _ 10, add_left_comm _ 10] at h1,
    linarith,
  }
  have h4: c = a - 12, by
  {
    linarith,
  }
  unfold R,
  linarith,

end age_difference_l319_319602


namespace rate_of_mangoes_l319_319675

theorem rate_of_mangoes :
  ∀ (apple_qty mango_qty : ℕ) (apple_rate total_paid : ℕ),
    apple_qty = 8 →
    apple_rate = 70 →
    mango_qty = 9 →
    total_paid = 1235 →
    let apple_cost := apple_qty * apple_rate in
    let mango_cost := total_paid - apple_cost in
    mango_cost / mango_qty = 75 :=
by
  intros apple_qty mango_qty apple_rate total_paid h₁ h₂ h₃ h₄
  let apple_cost := apple_qty * apple_rate
  let mango_cost := total_paid - apple_cost
  have h₅ : apple_cost = 560, by sorry
  have h₆ : mango_cost = 675, by sorry
  have h₇ : mango_qty = 9, by sorry
  have h₈ : mango_cost / mango_qty = 75, by sorry
  exact h₈

end rate_of_mangoes_l319_319675


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319754

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319754


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319705

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319705


namespace coefficient_x3_y7_expansion_l319_319305

theorem coefficient_x3_y7_expansion : 
  let n := 10
  let a := (2 : ℚ) / 3
  let b := -(3 : ℚ) / 5
  let k := 3
  let binom := Nat.choose n k
  let term := binom * (a ^ k) * (b ^ (n - k))
  term = -(256 : ℚ) / 257 := 
by
  -- Proof omitted
  sorry

end coefficient_x3_y7_expansion_l319_319305


namespace monotonically_increasing_on_interval_l319_319484

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem monotonically_increasing_on_interval :
  ∀ (ω ϕ : ℝ),
    ω > 0 →
    |ϕ| < Real.pi / 2 →
    let f := f ω ϕ in
    (∀ x, f (x + Real.pi / 12) = f (- (x + Real.pi / 12))) →
    ∀ x,
      3 * Real.pi / 4 ≤ x ∧ x ≤ Real.pi →
      monotone_on (λ x, f x) (Set.Icc (3 * Real.pi / 4) Real.pi) :=
by
  intros ω ϕ hω hϕ feven x hx
  sorry

end monotonically_increasing_on_interval_l319_319484


namespace value_of_x_l319_319366

-- Define the data set and the condition for the median
def dataSet := [1, 5, 6, x, 9, 19]

-- Median condition
axiom median_condition : ∀ x : ℕ, (6 + x) / 2 = 7

-- The theorem to prove that x is 8
theorem value_of_x (x : ℕ) : (median_condition x) → x = 8 :=
by
sorry

end value_of_x_l319_319366


namespace probability_equal_2s_after_4040_rounds_l319_319409

/-- 
Given three players Diana, Nathan, and Olivia each starting with $2, each player (with at least $1) 
simultaneously gives $1 to one of the other two players randomly every 20 seconds. 
Prove that the probability that after the bell has rung 4040 times, 
each player will have $2$ is $\frac{1}{4}$.
-/
theorem probability_equal_2s_after_4040_rounds 
  (n_rounds : ℕ) (start_money : ℕ) (probability_outcome : ℚ) :
  n_rounds = 4040 →
  start_money = 2 →
  probability_outcome = 1 / 4 :=
by
  sorry

end probability_equal_2s_after_4040_rounds_l319_319409


namespace greatest_multiple_of_5_and_6_under_1000_l319_319739

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319739


namespace labourer_saving_after_debt_clearance_l319_319255

variable (averageExpenditureFirst6Months : ℕ)
variable (monthlyIncome : ℕ)
variable (reducedMonthlyExpensesNext4Months : ℕ)

theorem labourer_saving_after_debt_clearance (h1 : averageExpenditureFirst6Months = 90)
                                              (h2 : monthlyIncome = 81)
                                              (h3 : reducedMonthlyExpensesNext4Months = 60) :
    (monthlyIncome * 4) - ((reducedMonthlyExpensesNext4Months * 4) + 
    ((averageExpenditureFirst6Months * 6) - (monthlyIncome * 6))) = 30 := by
  sorry

end labourer_saving_after_debt_clearance_l319_319255


namespace complex_magnitude_l319_319439

theorem complex_magnitude {z : ℂ} (h : (2 + I) * conj(z) = 5 + 5 * I) : |z| = Real.sqrt 10 := 
by
  sorry

end complex_magnitude_l319_319439


namespace arithmetic_sequence_sum_l319_319568

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℕ) (n : ℕ),
  (a 3 = 3) →
  (a 1 + a 7 = 8) →
  (∀ n, a (n + 1) = a n + (a 2 - a 1)) →
  ∑ k in finset.range 2018, (1 : ℚ) / ((a (k + 1)) * (a k)) = 2018 / 2019 :=
begin
  sorry
end

end arithmetic_sequence_sum_l319_319568


namespace exists_point_P_l319_319006

noncomputable def equilateral_triangle (A B C : ℂ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def rotated_scaled_triangle (A B C : ℂ) (θ : ℝ) (k : ℝ) : (ℂ × ℂ × ℂ) :=
  let O : ℂ := (A + B + C) / 3
  let A' := O + k * (A - O) * exp(θ * complex.I)
  let B' := O + k * (B - O) * exp(θ * complex.I)
  let C' := O + k * (C - O) * exp(θ * complex.I)
  (A', B', C')

theorem exists_point_P {A1 B1 C1 A2 B2 C2 : ℂ}
  (h₁ : equilateral_triangle A1 B1 C1)
  (h₂ : rotated_scaled_triangle A1 B1 C1 (23 * ℂ.pi / 180) (real.sqrt 5 / 2) = (A2, B2, C2)) :
  ∃ P : ℂ, equilateral_triangle P A1 A2 ∧ equilateral_triangle P B2 C1 :=
by
  sorry

end exists_point_P_l319_319006


namespace ceil_square_values_l319_319512

theorem ceil_square_values (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, (∀ m : ℕ, m = n ↔ (121 < x^2 ∧ x^2 ≤ 144) ∧ (⌈x^2⌉ = m)) ∧ n = 23 :=
by
  sorry

end ceil_square_values_l319_319512


namespace count_non_congruent_rectangles_l319_319829

def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def is_integer_length (n : ℕ) : Prop := True

def perimeter_condition (w h : ℕ) : Prop := 2 * (w + h) = 80

def non_congruent_rectangles_count (count : ℕ) : Prop :=
  ∃ (k : ℕ) (Hk1 : 1 ≤ k) (Hk2 : k ≤ 13),
  ∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 13 ∧ 1 ≤ j ∧ j ≤ 13 ∧ 3*i + 3*j = 40) → i = j

theorem count_non_congruent_rectangles :
  ∃ (count : ℕ), count = 13 ∧
  (∀ w h : ℕ, perimeter_condition w h → is_multiple_of_three w → is_integer_length w → is_integer_length h →
    non_congruent_rectangles_count count) :=
begin
  existsi 13,
  split,
  { refl, },
  { intros w h Hw Hm3 Hw_int Hh_int,
    sorry,
  }
end

end count_non_congruent_rectangles_l319_319829


namespace find_A_given_B_eq_pi_over_3_prove_S_given_largest_angle_conditions_prove_b_given_largest_angle_conditions_prove_a_c_squared_given_largest_angle_conditions_l319_319188

-- Given definitions for conditions
variable (a b c S : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : a*B + b*A = 2*a)

-- Problem Part 1
theorem find_A_given_B_eq_pi_over_3 : B = π / 3 → a*cos B + b*cos A = 2*a → A = π / 6 :=
  by 
    intro hB hCond
    sorry

-- Problem Part 2 Part (2)(i)
theorem prove_S_given_largest_angle_conditions : a^2 + c^2 + a*c = b^2 → b = sqrt 7 → B = largest_angle(triangle_ABC) → S = sqrt(3)/2 :=
  by 
    intro h1 h2 h3
    sorry

-- Problem Part 2 Part (2)(ii)
theorem prove_b_given_largest_angle_conditions : a^2 + c^2 + a*c = b^2 → S = sqrt(3)/2 → B = largest_angle(triangle_ABC) → b = sqrt 7 :=
  by 
    intro h1 h2 h3
    sorry

-- Problem Part 2 Part (2)(iii)
theorem prove_a_c_squared_given_largest_angle_conditions : b = sqrt 7 → S = sqrt(3)/2 → B = largest_angle(triangle_ABC) → a^2 + c^2 + a*c = b^2 :=
  by 
    intro h1 h2 h3
    sorry

end find_A_given_B_eq_pi_over_3_prove_S_given_largest_angle_conditions_prove_b_given_largest_angle_conditions_prove_a_c_squared_given_largest_angle_conditions_l319_319188


namespace pow_mod_eq_residue_l319_319407

theorem pow_mod_eq_residue :
  (3 : ℤ)^(2048) % 11 = 5 :=
sorry

end pow_mod_eq_residue_l319_319407


namespace find_ordered_pair_l319_319896

theorem find_ordered_pair (x y : ℝ) :
  (2 * x + 3 * y = (6 - x) + (6 - 3 * y)) ∧ (x - 2 * y = (x - 2) - (y + 2)) ↔ (x = -4) ∧ (y = 4) := by
  sorry

end find_ordered_pair_l319_319896


namespace intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l319_319209

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }

def B : Set ℝ := { x | -4 < x ∧ x < 0 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -4 < x ∧ x ≤ -3 } :=
by sorry

theorem union_of_A_and_B :
  A ∪ B = { x | x < 0 ∨ x ≥ 1 } :=
by sorry

theorem complement_of_A_with_respect_to_U :
  U \ A = { x | -3 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l319_319209


namespace each_last_two_videos_length_l319_319197

theorem each_last_two_videos_length (total_time : ℕ) (first_video : ℕ) (second_video : ℕ) 
  (total_last_two_videos : ℕ) 
  (total_time_condition : total_time = 510)
  (first_video_condition : first_video = 2 * 60)
  (second_video_condition : second_video = 4 * 60 + 30)
  (total_last_two_condition : total_last_two_videos = total_time - (first_video + second_video)) :
  (total_last_two_videos / 2) = 60 :=
begin
  rw total_time_condition,
  rw first_video_condition,
  rw second_video_condition,
  rw total_last_two_condition,
  sorry,
end

end each_last_two_videos_length_l319_319197


namespace rex_has_399_cards_left_l319_319228

def Nicole_cards := 700

def Cindy_cards := 3 * Nicole_cards + (40 / 100) * (3 * Nicole_cards)
def Tim_cards := (4 / 5) * Cindy_cards
def combined_total := Nicole_cards + Cindy_cards + Tim_cards
def Rex_and_Joe_cards := (60 / 100) * combined_total

def cards_per_person := Nat.floor (Rex_and_Joe_cards / 9)

theorem rex_has_399_cards_left : cards_per_person = 399 := by
  sorry

end rex_has_399_cards_left_l319_319228


namespace card_K_2004_2004_l319_319133

def K : ℕ → ℕ → set ℕ
| n 0 := ∅
| n (m + 1) := { k | 1 ≤ k ∧ k ≤ n ∧ K k m ∩ K (n - k) m = ∅ }

theorem card_K_2004_2004 : (K 2004 2004).card = 127 := 
by { sorry }

end card_K_2004_2004_l319_319133


namespace log_exp_simplification_l319_319238

theorem log_exp_simplification :
  log 6 4 + log 6 9 - 8^(2/3) = -2 := 
sorry

end log_exp_simplification_l319_319238


namespace pilot_speed_outbound_l319_319827

theorem pilot_speed_outbound (v : ℝ) (d : ℝ) (s_return : ℝ) (t_total : ℝ) 
    (return_time : ℝ := d / s_return) 
    (outbound_time : ℝ := t_total - return_time) 
    (speed_outbound : ℝ := d / outbound_time) :
  d = 1500 → s_return = 500 → t_total = 8 → speed_outbound = 300 :=
by
  intros hd hs ht
  sorry

end pilot_speed_outbound_l319_319827


namespace calculate_104_squared_l319_319035

theorem calculate_104_squared :
  let a := 100
  let b := 4
  104 * 104 = a^2 + 2 * a * b + b^2 := by
    let a := 100
    let b := 4
    have h1 : a^2 = 10000 := by rfl
    have h2 : 2 * a * b = 800 := by rfl
    have h3 : b^2 = 16 := by rfl
    calc
      104 * 104 = (100 + 4) * (100 + 4) : by rfl
             ... = 10000 + 2 * 100 * 4 + 16 : by rw [pow_add, mul_assoc, mul_assoc, pow_two, *h1, *h2, *h3]
             ... = 10816 : by rfl

end calculate_104_squared_l319_319035


namespace second_firm_can_hire_10_geniuses_l319_319679

structure Programmer where
  coord : Fin 11 → ℕ
  sum_eq_100 : (Fin.sum (λ i, coord i) = 100)

def Geniuses : Set Programmer := { p | ∃ i, ∀ j ≠ i, p.coord j = 0 ∧ p.coord i = 100 }

def Familiar (p q : Programmer) : Prop :=
  ∃ i j, i ≠ j ∧ (p.coord i = q.coord i + 1 ∨ p.coord i + 1 = q.coord i) ∧
           (p.coord j = q.coord j - 1 ∨ p.coord j - 1 = q.coord j) ∧
           (∀ k, k ≠ i ∧ k ≠ j -> p.coord k = q.coord k)

theorem second_firm_can_hire_10_geniuses (A B : Set Programmer) :
  (∀ p ∈ A, ∃ q ∈ B, Familiar p q) ∧ (∀ p ∈ B, ∃ q ∈ A, Familiar p q) ∧
  (B ⊆ Geniuses) ∧
  (∀ p : Programmer, p ∈ A ↔ p ∈ Geniuses) →
  ∃ S : Finset Programmer, (S.card = 10 ∧ S ⊆ B) :=
sorry

end second_firm_can_hire_10_geniuses_l319_319679


namespace probability_neither_red_nor_purple_l319_319347

theorem probability_neither_red_nor_purple (total_balls white_balls green_balls yellow_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_white : white_balls = 50)
  (h_green : green_balls = 30)
  (h_yellow : yellow_balls = 10)
  (h_red : red_balls = 7)
  (h_purple : purple_balls = 3) :
  (total_balls - (red_balls + purple_balls)) / total_balls = 0.9 :=
by
  sorry

end probability_neither_red_nor_purple_l319_319347


namespace pirate_treasure_chest_coins_l319_319024

theorem pirate_treasure_chest_coins:
  ∀ (gold_coins silver_coins bronze_coins: ℕ) (chests: ℕ),
    gold_coins = 3500 →
    silver_coins = 500 →
    bronze_coins = 2 * silver_coins →
    chests = 5 →
    (gold_coins / chests + silver_coins / chests + bronze_coins / chests = 1000) :=
by
  intros gold_coins silver_coins bronze_coins chests gold_eq silv_eq bron_eq chest_eq
  sorry

end pirate_treasure_chest_coins_l319_319024


namespace cos_sum_inequality_l319_319623

theorem cos_sum_inequality (α β γ : ℝ) (h1 : α + β + γ = π) :
    (cos α / (sin β * sin γ)) + (cos β / (sin γ * sin α)) + (cos γ / (sin α * sin β)) ≤ 3 := 
by 
  sorry

end cos_sum_inequality_l319_319623


namespace gallons_per_cubic_foot_l319_319580

theorem gallons_per_cubic_foot (mix_per_pound : ℝ) (capacity_cubic_feet : ℕ) (weight_per_gallon : ℝ)
    (price_per_tbs : ℝ) (total_cost : ℝ) (total_gallons : ℝ) :
  mix_per_pound = 1.5 →
  capacity_cubic_feet = 6 →
  weight_per_gallon = 8 →
  price_per_tbs = 0.5 →
  total_cost = 270 →
  total_gallons = total_cost / (price_per_tbs * mix_per_pound * weight_per_gallon) →
  total_gallons / capacity_cubic_feet = 7.5 :=
by
  intro h1 h2 h3 h4 h5 h6
  rw [h2, h6]
  sorry

end gallons_per_cubic_foot_l319_319580


namespace equation_of_ellipse_HN_passes_through_fixed_point_l319_319110

-- Definitions of points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Center and axes of the ellipse
def center : ℝ × ℝ := (0, 0)
def x_axis_symmetry := True
def y_axis_symmetry := True

-- The ellipse passes through A and B
def ellipse (x y : ℝ) : Prop := (3 * x^2 + 4 * y^2 = 12)

theorem equation_of_ellipse :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 :=
begin
  split;
  unfold ellipse;
  norm_num,
end

theorem HN_passes_through_fixed_point :
  ∃ K : ℝ × ℝ, (K = (0, -2)) ∧
  ∀ (M N T H : ℝ × ℝ), (ellipse M.1 M.2) ∧ (ellipse N.1 N.2) ∧ 
  (∃ (k : ℝ), N.1 = k * M.1 ∧ N.2 = k * M.2) ∧ -- line MN
  (T.1 = M.1) ∧ (∃ (yT : ℝ), (T.1, yT) = T) ∧ -- T on line AB
  (H.1 = 2 * M.1 - T.1) ∧ (H.2 = 2 * M.2 - T.2) -> -- H's coordinates
  ((N.1 - H.1) = 0∧ (N.2 - H.2) = -2) := -- checking HN passes through (0,-2)
sorry

end equation_of_ellipse_HN_passes_through_fixed_point_l319_319110


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319701

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319701


namespace question1_question2_question3_l319_319992

-- Question 1
def staircase_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (2 * n - 1) = a (2 * n) ∧ a (2 * n) < a (2 * n + 1)

def b_seq : ℕ → ℕ
| 0 := 1 -- technically unused in this problem, typically corrected in proper indexing
| (2 * n) := b_seq (2 * n - 2)
| (2 * n + 1) := 9 * b_seq (2 * n - 1)

theorem question1 (n : ℕ) (hn : n > 0) : b_seq 2016 = 3 ^ 2014 :=
  sorry

-- Question 2
def c_seq : ℕ → ℕ := sorry  -- Placeholder for sequence c_n
def S : ℕ → ℕ := sorry  -- Placeholder for sum of first n terms of c

theorem question2 : 
  (∃ k : ℕ, S (k + 2) - S (k + 1) = S (k + 1) - S k) ∧
  (¬ ∃ k : ℕ, (S (k + 3) - S (k + 2) = S (k + 2) - S (k + 1) ∧ S (k + 2) - S (k + 1) = S (k + 1) - S k)) :=
sorry

-- Question 3
def d_seq : ℕ → ℕ
| 0 := 1  -- technically, again, indexed for consistent sequence handling
| (2 * n) := d_seq (2 * n - 2)
| (2 * n + 1) := d_seq (2 * n - 1) + 2

def T (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of { 1 / (d_n * d_(n + 2)) }

theorem question3 (t : ℝ) : (∀ n : ℕ, (n > 0) → (t - T n) * (t + 1 / (T n)) < 0) ↔ (-1 : ℝ) ≤ t ∧ t < (1/3 : ℝ) :=
  sorry

end question1_question2_question3_l319_319992


namespace ratio_d_s_l319_319557

theorem ratio_d_s (s d : ℝ) 
  (h : (25 * 25 * s^2) / (25 * s + 50 * d)^2 = 0.81) :
  d / s = 1 / 18 :=
by
  sorry

end ratio_d_s_l319_319557


namespace log_mean_inequality_cauchy_inequality_generalized_cauchy_inequality_l319_319115

-- Problem 1: Prove the logarithmic mean inequality.
theorem log_mean_inequality (x : ℝ) (n : ℕ) (xs : Fin n → ℝ) (h : ∀ i, 1 ≤ xs i) :
  (∑ i in Finset.univ, Real.log (xs i)) / n < Real.log ((∑ i in Finset.univ, xs i) / n) :=
sorry

-- Problem 2: Prove the Cauchy inequality.
theorem cauchy_inequality (x : ℝ) (n : ℕ) (xs : Fin n → ℝ) (h : ∀ i, 0 < xs i) :
  (∏ i in Finset.univ, xs i)^(1/n) < (∑ i in Finset.univ, xs i) / n :=
sorry

-- Problem 3: Prove the generalized Cauchy inequality.
theorem generalized_cauchy_inequality (x : ℝ) (n : ℕ) (xs : Fin n → ℝ) (ms : Fin n → ℝ) (h : ∀ i, 0 < xs i ∧ 0 < ms i) :
  (∏ i in Finset.univ, xs i^(ms i))^((∑ i in Finset.univ, ms i)^(-1)) < (∑ i in Finset.univ, (ms i) * (xs i)) / (∑ i in Finset.univ, ms i) :=
sorry

end log_mean_inequality_cauchy_inequality_generalized_cauchy_inequality_l319_319115


namespace pentagon_square_ratio_l319_319834

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l319_319834


namespace p_or_q_then_p_and_q_is_false_l319_319963

theorem p_or_q_then_p_and_q_is_false (p q : Prop) (hpq : p ∨ q) : ¬(p ∧ q) :=
sorry

end p_or_q_then_p_and_q_is_false_l319_319963


namespace greatest_multiple_l319_319720

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319720


namespace table_height_solution_l319_319261

noncomputable def table_height_problem : Prop :=
  ∃ (t b c : ℝ), 
    (t + b = c + 150) ∧ 
    (t + c = b + 110) ∧ 
    t = 130

theorem table_height_solution : table_height_problem :=
begin
  let t : ℝ := 130,
  let b : ℝ := 0,
  let c : ℝ := 0,
  use [t, b, c],
  split,
  { simp, linarith },
  split,
  { simp, linarith },
  { refl }
end

end table_height_solution_l319_319261


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319745

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319745


namespace first_player_wins_with_initial_piles_l319_319669

theorem first_player_wins_with_initial_piles : ∀ (piles : List ℕ), piles = [2, 3, 4] →
  player_has_winning_strategy 1 piles :=
by
  sorry

end first_player_wins_with_initial_piles_l319_319669


namespace irrational_number_among_choices_l319_319848

-- Condition A
def A : Real := Real.cbrt 5

-- Condition B
def B : Real := Real.sqrt 9

-- Condition C
def C : Real := - (8 / 3 : Real)

-- Condition D
def D : Real := 60.25

-- Statement of the problem
theorem irrational_number_among_choices : Irrational A :=
by
  sorry

end irrational_number_among_choices_l319_319848


namespace range_of_abs_2z_minus_1_l319_319093

open Complex

theorem range_of_abs_2z_minus_1
  (z : ℂ)
  (h : abs (z + 2 - I) = 1) :
  abs (2 * z - 1) ∈ Set.Icc (Real.sqrt 29 - 2) (Real.sqrt 29 + 2) :=
sorry

end range_of_abs_2z_minus_1_l319_319093


namespace find_a_l319_319457

theorem find_a (a : ℝ) (h1 : ∀ x ∈ set.Icc 0 1, 0 < a) 
(h2 : (real.exp (real.log a * 1) + real.exp (real.log a * 0) = 3)) : 
  a = 2 :=
sorry

end find_a_l319_319457


namespace range_of_a_l319_319947

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * a * x^2 - 9 * a^2 * x + a^3

def f_prime (a x : ℝ) : ℝ := 3 * x^2 - 6 * a * x - 9 * a^2

theorem range_of_a (a : ℝ) (h1 : a > (1 / 4)) :
  (∀ x ∈ Icc 1 (4 * a), |f_prime a x| ≤ 12 * a) ↔ a ∈ Ioc (1 / 4) (4 / 5) := 
sorry

end range_of_a_l319_319947


namespace find_n_l319_319105

theorem find_n : (∃ n : ℕ, 2^3 * 8^3 = 2^(2 * n)) ↔ n = 6 :=
by
  sorry

end find_n_l319_319105


namespace min_val_of_3x_add_4y_l319_319978

theorem min_val_of_3x_add_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 
  (3 * x + 4 * y ≥ 5) ∧ (3 * x + 4 * y = 5 → x + 4 * y = 3) := 
by
  sorry

end min_val_of_3x_add_4y_l319_319978


namespace problem1_problem2_problem3_l319_319870

-- Definition of operation T
def T (x y m n : ℚ) := (m * x + n * y) * (x + 2 * y)

-- Problem 1: Given T(1, -1) = 0 and T(0, 2) = 8, prove m = 1 and n = 1
theorem problem1 (m n : ℚ) (h1 : T 1 (-1) m n = 0) (h2 : T 0 2 m n = 8) : m = 1 ∧ n = 1 := by
  sorry

-- Problem 2: Given the system of inequalities in terms of p and knowing T(x, y) = (mx + ny)(x + 2y) with m = 1 and n = 1
--            has exactly 3 integer solutions, prove the range of values for a is 42 ≤ a < 54
theorem problem2 (a : ℚ) 
  (h1 : ∃ p : ℚ, T (2 * p) (2 - p) 1 1 > 4 ∧ T (4 * p) (3 - 2 * p) 1 1 ≤ a)
  (h2 : ∃! p : ℤ, -1 < p ∧ p ≤ (a - 18) / 12) : 42 ≤ a ∧ a < 54 := by
  sorry

-- Problem 3: Given T(x, y) = T(y, x) when x^2 ≠ y^2, prove m = 2n
theorem problem3 (m n : ℚ) 
  (h : ∀ x y : ℚ, x^2 ≠ y^2 → T x y m n = T y x m n) : m = 2 * n := by
  sorry

end problem1_problem2_problem3_l319_319870


namespace intersection_of_domains_l319_319221

def A_domain : Set ℝ := { x : ℝ | 4 - x^2 ≥ 0 }
def B_domain : Set ℝ := { x : ℝ | 1 - x > 0 }

theorem intersection_of_domains :
  (A_domain ∩ B_domain) = { x : ℝ | -2 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_of_domains_l319_319221


namespace no_30_consecutive_zeros_in_last_100_digits_l319_319053

theorem no_30_consecutive_zeros_in_last_100_digits :
  ∀ k : ℕ, (∀ m : ℕ, k ≤ m → (5^k - 1) % 2^100 = 0) →
  ∀ n : ℕ, ¬ (∃ l : ℕ, l ≥ 30 ∧ Nat.digits 10 (5^n) % 10^100 = List.repeat 0 l) :=
by 
  sorry

end no_30_consecutive_zeros_in_last_100_digits_l319_319053


namespace ceil_pow_sq_cardinality_l319_319520

noncomputable def ceil_pow_sq_values (x : ℝ) (h : 11 < x ∧ x ≤ 12) : ℕ :=
  ((Real.ceil(x^2)) - (Real.ceil(121)) + 1)

theorem ceil_pow_sq_cardinality :
  ∀ (x : ℝ), (11 < x ∧ x ≤ 12) → ceil_pow_sq_values x _ = 23 :=
by
  intro x hx
  let attrs := (11 < x ∧ x ≤ 12)
  sorry

end ceil_pow_sq_cardinality_l319_319520


namespace leapYearCountIsFive_l319_319002

def isValidLeapYear (y : ℕ) : Prop :=
  (y % 1200 = 300 ∨ y % 1200 = 900) ∧ (2000 ≤ y ∧ y ≤ 5000)

def countValidLeapYears : ℕ :=
  List.countp isValidLeapYear (List.range' 2000 (5001 - 2000))

theorem leapYearCountIsFive : countValidLeapYears = 5 := by
  sorry

end leapYearCountIsFive_l319_319002


namespace books_total_l319_319196

theorem books_total (books_Keith books_Jason : ℕ) (h_K : books_Keith = 20) (h_J : books_Jason = 21) : books_Keith + books_Jason = 41 :=
by
  rw [h_K, h_J]
  exact rfl

end books_total_l319_319196


namespace largest_4_digit_congruent_15_mod_22_l319_319307

theorem largest_4_digit_congruent_15_mod_22 :
  ∃ (x : ℤ), x < 10000 ∧ x % 22 = 15 ∧ (∀ (y : ℤ), y < 10000 ∧ y % 22 = 15 → y ≤ x) → x = 9981 :=
sorry

end largest_4_digit_congruent_15_mod_22_l319_319307


namespace english_vocab_related_to_reading_level_l319_319674

theorem english_vocab_related_to_reading_level (N : ℕ) (K_squared : ℝ) (critical_value : ℝ) (p_value : ℝ)
  (hN : N = 100)
  (hK_squared : K_squared = 7)
  (h_critical_value : critical_value = 6.635)
  (h_p_value : p_value = 0.010) :
  p_value <= 0.01 → K_squared > critical_value → true :=
by
  intro h_p_value_le h_K_squared_gt
  sorry

end english_vocab_related_to_reading_level_l319_319674


namespace find_smallest_n_integer_l319_319038

noncomputable def sqrt4 : ℝ := real.rpow 4 (1/4)

def y : ℕ → ℝ
| 0     := sqrt4
| (n+1) := y n ^ sqrt4

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem find_smallest_n_integer:
  ∃ n : ℕ, is_integer (y n) ∧ ∀ m < n, ¬ is_integer (y m) ∧ n = 4 := by
  sorry

end find_smallest_n_integer_l319_319038


namespace chords_length_square_l319_319178

theorem chords_length_square 
  (r1 r2 d : ℝ)
  (h1 : r1 = 10)
  (h2 : r2 = 7)
  (h3 : d = 15)
  (h4 : ∃ P : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 = r1 ^ 2) ∧ (P.1 ^ 2 + (P.2 - d) ^ 2 = r2 ^ 2))
  (h5 : ∃ Q R : ℝ × ℝ, Q.1 ^ 2 + Q.2 ^ 2 = r1 ^ 2 ∧ R.1 ^ 2 + (R.2 - d) ^ 2 = r2 ^ 2 ∧ (Q.1 = P.1 ∧ Q.2 = P.2) ∧ (Q.1 = R.1 ∧ Q.2 = R.2)) :
  let x := (Q.1 - R.1) ^ 2 + (Q.2 - R.2) ^ 2 in x = 72.25 :=
sorry

end chords_length_square_l319_319178


namespace ratio_major_minor_is_15_4_l319_319189

-- Define the given conditions
def main_characters : ℕ := 5
def minor_characters : ℕ := 4
def minor_character_pay : ℕ := 15000
def total_payment : ℕ := 285000

-- Define the total pay to minor characters
def minor_total_pay : ℕ := minor_characters * minor_character_pay

-- Define the total pay to major characters
def major_total_pay : ℕ := total_payment - minor_total_pay

-- Define the ratio computation
def ratio_major_minor : ℕ × ℕ := (major_total_pay / 15000, minor_total_pay / 15000)

-- State the theorem
theorem ratio_major_minor_is_15_4 : ratio_major_minor = (15, 4) :=
by
  -- Proof goes here
  sorry

end ratio_major_minor_is_15_4_l319_319189


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319746

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319746


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319758

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319758


namespace negation_of_universal_proposition_l319_319270

-- Define the properties of a parallelogram
def isParallelogram (P : Type) : Prop :=
  ∃ (A B C D : P), P -- Should be a type to represent points or similar

-- State the conditions: diagonals of a parallelogram are equal and bisect each other
def diagonalsEqualAndBisect (P : Type) [isParallelogram P] : Prop :=
  ∀ (A B C D : P), -- Assuming A, B, C, D define a parallelogram
  -- Define equal diagonals and bisecting property

-- The negation we need to prove
def negatedProperty (P : Type) [isParallelogram P] : Prop :=
  ∃ (A B C D : P), -- There exists a parallelogram
  ¬(diagonalsEqualAndBisect P)

theorem negation_of_universal_proposition (P : Type) [isParallelogram P] :
  ¬(diagonalsEqualAndBisect P) ↔ negatedProperty P :=
begin
  sorry
end

end negation_of_universal_proposition_l319_319270


namespace regular_polygon_sides_l319_319531

theorem regular_polygon_sides (exterior_angle : ℝ) (h_exterior : exterior_angle = 45) : 
  ∃ n : ℕ, n = 8 :=
by
  have h_sum : 360 / exterior_angle = 8 := by sorry
  use 8
  exact h_sum

end regular_polygon_sides_l319_319531


namespace distance_between_center_and_point_l319_319688

theorem distance_between_center_and_point :
  let center := (1 : ℝ, 2 : ℝ)
      point := (13 : ℝ, 7 : ℝ)
      dist := Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2)
  in dist = 13 := sorry

end distance_between_center_and_point_l319_319688


namespace perpendicular_chords_of_parabola_l319_319496

theorem perpendicular_chords_of_parabola {A B C D : Point} (h_parabola : parabola y^2 = 4 * x)
    (h_perpendicular : perpendicular (chord AB) (chord CD))
    (h_through_focus : passes_through_focus (chord AB) ∧ passes_through_focus (chord CD))
    : (1 / |chord_length AB|) + (1 / |chord_length CD|) = 1 / 4 :=
sorry

end perpendicular_chords_of_parabola_l319_319496


namespace algebraic_comparison_l319_319857

theorem algebraic_comparison (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 / b + b^2 / a ≥ a + b) :=
by
  sorry

end algebraic_comparison_l319_319857


namespace complex_sum_lim_eq_l319_319249

noncomputable def complex_series_lim (a : ℕ → ℂ) (l : ℂ) : Prop :=
∀ ε > 0, ∃ N, ∀ n > N, abs (a n - l) < ε

theorem complex_sum_lim_eq (k : ℕ) (a : Finₓ k → ℂ) (c : ℂ)
  (h1 : ∀ i, 0 < i ∧ i <= k → abs (a i) = 1)
  (h2 : complex_series_lim (λ n, ∑ i in Finₓ.range k, a i ^ n) c) :
  c = k ∧ ∀ i, 0 < i ∧ i <= k → a i = 1 := 
sorry

end complex_sum_lim_eq_l319_319249


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319699

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319699


namespace calc_value_l319_319989

theorem calc_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := 
by 
  sorry

end calc_value_l319_319989


namespace area_of_isosceles_triangle_with_conditions_l319_319545

theorem area_of_isosceles_triangle_with_conditions
  (A B C D E : Point)
  (h_isosceles : dist A B = dist B C)
  (h_altitude : line_throughp A D ∧ line_throughp C D ∧ right_angle (angle A B D))
  (h_extension : collinear [A, C, E] ∧ dist B E = 10)
  (tan_gprog : geometric_progression (tan (angle C B E)) (tan (angle D B E)) (tan (angle A B E)))
  (cot_aprog : arithmetic_progression (cot (angle D B E)) (cot (angle C B E)) (cot (angle D B C))) :
  area_of_triangle A B C = 50 / 3 := 
sorry

end area_of_isosceles_triangle_with_conditions_l319_319545


namespace nat_set_eq_l319_319284

open Finset

noncomputable def nat_set := {x : ℕ | 8 < x ∧ x < 12}

theorem nat_set_eq : nat_set = {9, 10, 11} :=
by
  ext x
  simp only [mem_set_of, mem_insert, mem_singleton, mem_empty]
  constructor
  · rintro ⟨h₈, h₁₂⟩
    interval_cases x
    · left
      refl
    · right
      left
      refl
    · right
      right
      left
      refl
  · rintro (rfl | rfl | (rfl | h))
    · exact ⟨(by linarith), (by linarith)⟩
    · exact ⟨(by linarith), (by linarith)⟩
    · exact ⟨(by linarith), (by linarith)⟩
    · cases h

end nat_set_eq_l319_319284


namespace infinite_nested_radical_eq_three_l319_319042

theorem infinite_nested_radical_eq_three : 
  (∃ m : ℝ, m > 0 ∧ m = Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + ...))) ∧ m = 3) :=
sorry

end infinite_nested_radical_eq_three_l319_319042


namespace rate_of_change_area_at_t4_l319_319286

variable (t : ℝ)

def a (t : ℝ) : ℝ := 2 * t + 1

def b (t : ℝ) : ℝ := 3 * t + 2

def S (t : ℝ) : ℝ := a t * b t

theorem rate_of_change_area_at_t4 :
  (deriv S 4) = 55 := by
  sorry

end rate_of_change_area_at_t4_l319_319286


namespace combined_weight_after_removal_l319_319809

theorem combined_weight_after_removal (weight_sugar weight_salt weight_removed : ℕ) 
                                       (h_sugar : weight_sugar = 16)
                                       (h_salt : weight_salt = 30)
                                       (h_removed : weight_removed = 4) : 
                                       (weight_sugar + weight_salt) - weight_removed = 42 :=
by {
  sorry
}

end combined_weight_after_removal_l319_319809


namespace pentagon_square_ratio_l319_319838

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l319_319838


namespace ceil_square_range_count_l319_319516

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l319_319516


namespace area_bounded_by_curves_l319_319420

open Set Filter

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x

theorem area_bounded_by_curves : 
  abs (∫ x in 0..2, f x) = 4 := by
-- Proof goes here
  sorry

end area_bounded_by_curves_l319_319420


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319708

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319708


namespace smallest_perimeter_of_consecutive_even_triangle_l319_319773

theorem smallest_perimeter_of_consecutive_even_triangle (n : ℕ) :
  (2 * n + 2 * n + 2 > 2 * n + 4) ∧
  (2 * n + 2 * n + 4 > 2 * n + 2) ∧
  (2 * n + 2 + 2 * n + 4 > 2 * n) →
  2 * n + (2 * n + 2) + (2 * n + 4) = 18 :=
by 
  sorry

end smallest_perimeter_of_consecutive_even_triangle_l319_319773


namespace cucumbers_for_20_apples_l319_319539

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end cucumbers_for_20_apples_l319_319539


namespace option_d_correct_l319_319494

variables {Point : Type} [EuclideanGeometry Point]

-- Define lines and planes as sets of points
def Line (l : set Point) : Prop := ∃ p q : Point, p ≠ q ∧ ∀ r : Point, r ∈ l ↔ r = p ∨ r = q
def Plane (α : set Point) : Prop := ∃ p q r : Point, ¬ Collinear p q r ∧ ∀ s : Point, s ∈ α ↔ ∃ u v w : ℝ, u + v + w = 1 ∧ u * p + v * q + w * r = s

noncomputable def parallel (l₁ l₂ : set Point) : Prop := ∃ v, direction l₁ = v ∧ direction l₂ = v
noncomputable def perpendicular (l : set Point) (α : set Point) : Prop := ∃ v₁ v₂, direction l = v₁ ∧ direction α = v₂ ∧ v₁ ⊥ v₂

theorem option_d_correct (m n : set Point) (α : set Point)
  (h1 : Line m) (h2 : Line n) (h3 : Plane α) 
  (h4 : perpendicular m α) (h5 : parallel m n) : perpendicular n α := 
sorry

end option_d_correct_l319_319494


namespace f_of_x_squared_domain_l319_319114

structure FunctionDomain (f : ℝ → ℝ) :=
  (domain : Set ℝ)
  (domain_eq : domain = Set.Icc 0 1)

theorem f_of_x_squared_domain (f : ℝ → ℝ) (h : FunctionDomain f) :
  FunctionDomain (fun x => f (x ^ 2)) :=
{
  domain := Set.Icc (-1) 1,
  domain_eq := sorry
}

end f_of_x_squared_domain_l319_319114


namespace profit_maximization_l319_319349

def yield_function (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then
    5*(x^2 + 3)
  else if 2 < x ∧ x ≤ 5 then
    50 - 50/(x + 1)
  else
    0

def profit_function (x : ℝ) : ℝ :=
  15 * yield_function x - 30 * x

theorem profit_maximization :
  profit_function 4 = 480 ∧ (∀ x ∈ Icc (0:ℝ) 2, profit_function x ≤ profit_function 2) ∧ (∀ x ∈ Ioc (2:ℝ) 5, profit_function x ≤ profit_function 4) :=
by sorry

end profit_maximization_l319_319349


namespace greatest_multiple_l319_319727

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319727


namespace asymptotes_hyperbola_l319_319926

noncomputable section

variable {a x y : ℝ}
variable {O P A B : ℝ × ℝ}

/-- Given that O is the coordinate origin, P is a point on the hyperbola x^2/a^2 - y^2 = 1,
two lines parallel to the asymptotes pass through P and intersect the asymptotes at points A and B,
if the area of the parallelogram OBPA is 1, then the equations of the asymptotes of the hyperbola
are y = ± (1/2) x. -/
theorem asymptotes_hyperbola (hO : O = (0, 0))
  (hP : ∃ (x y : ℝ), P = (x, y) ∧ x^2 / a^2 - y^2 = 1 ∧ a > 0)
  (hAB : ∃ (A B : ℝ × ℝ), (∃ m n, P = (m, n) ∧ A ≠ B ∧
    ∀ A B : ℝ × ℝ, A ∈ line_through (0, 0) (1, -1) ∧ B ∈ line_through (0, 0) (1, 1)))
  (hArea : parallelogram_area O B P A = 1) :
  ( ∀ (x : ℝ), y = (1/2) * x) ∨ ( ∀ (x : ℝ), y = -(1/2) * x) := sorry

end asymptotes_hyperbola_l319_319926


namespace shower_walls_l319_319368

theorem shower_walls (w : ℕ) 
  (h_wd : 8) 
  (h_ht : 20) 
  (h_total : 480) :
  8 * 20 * w = 480 → w = 3 := 
by 
  sorry

end shower_walls_l319_319368


namespace suff_cond_parallel_l319_319797

variables (α β : Plane) (a b : Line)

theorem suff_cond_parallel (h1 : α ∥ β) (h2 : a ⊆ β) : a ∥ α :=
sorry

end suff_cond_parallel_l319_319797


namespace sum_even_n_l319_319951

noncomputable def sequence := ℕ → ℚ

def a_n (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else a_n (n - 2) + 3

def sum_first_n_terms (n : ℕ) (a : sequence) : ℚ :=
  ∑ i in Finset.range n, a i

theorem sum_even_n (a : sequence) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) (h3 : ∀ k : ℕ, a (k + 2) = a k + 3) (hn_even : n % 2 = 0) : 
sum_first_n_terms n a = 3 * n^2 / 4 := 
by
  sorry

end sum_even_n_l319_319951


namespace dealer_can_determine_values_l319_319551

def card_value_determined (a : Fin 100 → Fin 100) : Prop :=
  (∀ i j : Fin 100, i > j → a i > a j) ∧ (a 0 > a 99) ∧
  (∀ k : Fin 100, a k = k + 1)

theorem dealer_can_determine_values :
  ∃ (messages : Fin 100 → Fin 100), card_value_determined messages :=
sorry

end dealer_can_determine_values_l319_319551


namespace greatest_multiple_of_5_and_6_under_1000_l319_319735

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319735


namespace find_k_l319_319266

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  if p1.1 = p2.1 then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_k (k : ℝ) :
  let A := (-4 : ℝ, 0 : ℝ) in
  let B := (0 : ℝ, -4 : ℝ) in
  let X := (0 : ℝ, 8 : ℝ) in
  let Y := (14 : ℝ, k) in
  slope A B = slope X Y → k = -6 :=
by
  -- Definitions of points A, B, X, Y
  let A := (-4 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, -4 : ℝ)
  let X := (0 : ℝ, 8 : ℝ)
  let Y := (14 : ℝ, k)
  -- Checking slopes and conditions
  sorry

end find_k_l319_319266


namespace find_matrix_N_l319_319404

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ w : Fin 2 → ℝ, (MulVec ![![N]] w) = (3 : ℝ) • w) →
  N = ![![3, 0], ![0, 3]] :=
by
  sorry

end find_matrix_N_l319_319404


namespace max_n_value_is_9_l319_319091

variable (a b c d n : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : c > d)
variable (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d)))

theorem max_n_value_is_9 (h1 : a > b) (h2 : b > c) (h3 : c > d)
    (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end max_n_value_is_9_l319_319091


namespace find_PF2_l319_319587

open Real

noncomputable def hyperbola_equation (x y : ℝ) := (x^2 / 16) - (y^2 / 20) = 1

noncomputable def distance (P F : ℝ × ℝ) : ℝ := 
  let (px, py) := P
  let (fx, fy) := F
  sqrt ((px - fx)^2 + (py - fy)^2)

theorem find_PF2
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (on_hyperbola : hyperbola_equation P.1 P.2)
  (foci_F1_F2 : F1 = (-6, 0) ∧ F2 = (6, 0))
  (distance_PF1 : distance P F1 = 9) : 
  distance P F2 = 17 := 
by
  sorry

end find_PF2_l319_319587


namespace length_of_AE_l319_319553

-- Definitions of the given conditions
variables (AB CD AC AE EC : ℝ)
variables (E : ℝ → Prop)
variables (ABCD_convex : Prop)
variables (triangles_similar : Prop)

-- Conditions as given in problem
def condition1 : Prop := ABCD_convex
def condition2 : AB = 10
def condition3 : CD = 15
def condition4 : AC = 18
def condition5 : E = E(AC)
def condition6 : triangles_similar

-- Proven length of segment AE
theorem length_of_AE (h1 : condition1) (h2 : condition2) (h3 : condition3)
                     (h4 : condition4) (h5 : condition5) (h6 : condition6) :
                     AE = 36 / 5 := by
  sorry

end length_of_AE_l319_319553


namespace total_cost_for_tickets_l319_319805

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end total_cost_for_tickets_l319_319805


namespace sin_le_zero_probability_l319_319360

theorem sin_le_zero_probability : let I := set.Icc (0:ℝ) (2 * Real.pi)
in let E := {x | x ∈ I ∧ Real.sin x ≤ 0}
in ∃ p, (∀ x, x ∈ I → p = (|E| / |I|)) ∧ p = 1/2 :=
sorry

end sin_le_zero_probability_l319_319360


namespace arithmetic_seq_sum_l319_319100

noncomputable def T (n : ℕ) : ℚ := (n : ℚ) / (2 * n + 1)

theorem arithmetic_seq_sum (a : ℕ → ℚ) (b : ℕ → ℚ) (T : ℕ → ℚ) : 
  (∀ n, a n = 2 * n + 1) → 
  (∀ n, b n = 1 / a n) → 
  (T n = ∑ i in finset.range n, b (i + 1)) → 
  T n = n / (2 * n + 1) := 
by 
  sorry

end arithmetic_seq_sum_l319_319100


namespace sum_max_min_f_l319_319649

def f (x : ℝ) : ℝ :=
  max (sin x) (max (cos x) ((sin x + cos x) / real.sqrt 2))

theorem sum_max_min_f : 
  (real.max (sin x) (real.max (cos x) ((sin x + cos x) / real.sqrt 2)).max + (real.min (sin x) (real.min (cos x) ((sin x + cos x) / real.sqrt 2)))) = 
  1 - real.sqrt 2 / 2 := by
  sorry

end sum_max_min_f_l319_319649


namespace tangent_length_and_equations_l319_319941

theorem tangent_length_and_equations (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) :
  P = (2, 3) → C = (1, 1) → r = 1 →
  ((x - 1)^2 + (y - 1)^2 = 1) →
  (∃ L : ℝ, L = 2) ∧ 
  (∃ eq1 eq2 : (ℝ × ℝ) → Prop, eq1 = λ p, 3 * p.1 - 4 * p.2 + 6 = 0 ∧ eq2 = λ p, p.1 = 2) :=
begin
  intros,
  split,
  { use 2,
    -- Proof that the length of the tangent line is 2, skipped
    sorry },
  { use (λ p, 3 * p.1 - 4 * p.2 + 6 = 0),
    use (λ p, p.1 = 2),
    -- Proof that these are the equations of the tangent lines, skipped
    sorry }
end

end tangent_length_and_equations_l319_319941


namespace compare_abc_l319_319449

noncomputable def a : ℝ := 5 ^ 0.2
noncomputable def b : ℝ := Real.logBase π 3
noncomputable def c : ℝ := Real.logBase 5 (Real.sin (sqrt 3 / 2 * Real.pi))

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l319_319449


namespace number_of_people_in_group_l319_319259

theorem number_of_people_in_group :
  ∃ (N : ℕ), (∀ (avg_weight : ℝ), 
  ∃ (new_person_weight : ℝ) (replaced_person_weight : ℝ),
  new_person_weight = 85 ∧ replaced_person_weight = 65 ∧
  avg_weight + 2.5 = ((N * avg_weight + (new_person_weight - replaced_person_weight)) / N) ∧ 
  N = 8) :=
by
  sorry

end number_of_people_in_group_l319_319259


namespace profit_percentage_approx_l319_319980

-- Define the cost price of the first item
def CP1 (S1 : ℚ) : ℚ := 0.81 * S1

-- Define the selling price of the second item as 10% less than the first
def S2 (S1 : ℚ) : ℚ := 0.90 * S1

-- Define the cost price of the second item as 81% of its selling price
def CP2 (S1 : ℚ) : ℚ := 0.81 * (S2 S1)

-- Define the total selling price before tax
def TSP (S1 : ℚ) : ℚ := S1 + S2 S1

-- Define the total amount received after a 5% tax
def TAR (S1 : ℚ) : ℚ := TSP S1 * 0.95

-- Define the total cost price of both items
def TCP (S1 : ℚ) : ℚ := CP1 S1 + CP2 S1

-- Define the profit
def P (S1 : ℚ) : ℚ := TAR S1 - TCP S1

-- Define the profit percentage
def ProfitPercentage (S1 : ℚ) : ℚ := (P S1 / TCP S1) * 100

-- Prove the profit percentage is approximately 17.28%
theorem profit_percentage_approx (S1 : ℚ) : abs (ProfitPercentage S1 - 17.28) < 0.01 :=
by
  sorry

end profit_percentage_approx_l319_319980


namespace line_problems_l319_319492

noncomputable def l1 : (ℝ → ℝ) := λ x => x - 1
noncomputable def l2 (k : ℝ) : (ℝ → ℝ) := λ x => -(k + 1) / k * x - 1

theorem line_problems (k : ℝ) :
  ∃ k, k = 0 → (l2 k 1) = 90 →      -- A
  (∀ k, (l1 1 = l2 k 1 → True)) →   -- B
  (∀ k, (l1 1 ≠ l2 k 1 → True)) →   -- C (negated conclusion from False in C)
  (∀ k, (l1 1 * l2 k 1 ≠ -1))       -- D
:=
sorry

end line_problems_l319_319492


namespace extra_time_due_to_leak_l319_319845

theorem extra_time_due_to_leak
  (time_to_fill_without_leak : ℝ)
  (time_to_empty_with_leak : ℝ)
  (H1 : time_to_fill_without_leak = 5)
  (H2 : time_to_empty_with_leak = 30) :
  (1 / (1 / time_to_fill_without_leak - 1 / time_to_empty_with_leak) - time_to_fill_without_leak = 1) :=
by
  rw [H1, H2]
  norm_num
  rw [sub_left_eq_add]
  norm_num
  exact sorry -- Proof steps can be completed here.

end extra_time_due_to_leak_l319_319845


namespace bernardo_larger_probability_l319_319029

theorem bernardo_larger_probability : 
  let S1 := {1, 2, 3, 4, 5, 6, 7, 8, 10}
  let S2 := {1, 2, 3, 4, 5, 6, 7, 9}
  prob_bernardo_larger S1 S2 = 37 / 56 := 
by
  sorry

end bernardo_larger_probability_l319_319029


namespace product_value_l319_319776

theorem product_value : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := 
by
  sorry

end product_value_l319_319776


namespace number_of_ways_to_choose_officers_l319_319564

open Nat

theorem number_of_ways_to_choose_officers (n : ℕ) (h : n = 8) : 
  n * (n - 1) * (n - 2) = 336 := by
  sorry

end number_of_ways_to_choose_officers_l319_319564


namespace books_not_read_l319_319668

theorem books_not_read (total_books read_books : ℕ) (h1 : total_books = 20) (h2 : read_books = 15) : total_books - read_books = 5 := by
  sorry

end books_not_read_l319_319668


namespace minimum_value_of_expression_l319_319895

noncomputable def min_value (a b : ℝ) (h : a > 0 ∧ b > 0) : ℝ :=
  if ∃ x₀ : ℝ, 2x₀ - 2 ≠ 0 ∧ -2x₀ + a ≠ 0 ∧
    (2x₀ - 2) * (-2x₀ + a) = -1 ∧ x₀^2 - 2*x₀ + 2 = -x₀^2 + a*x₀ + b then
    (1 / a + 4 / b)
  else
    0

theorem minimum_value_of_expression : ∀ a b : ℝ, 
  a > 0 ∧ b > 0 → 
  (∃ x₀ : ℝ, 2x₀ - 2 ≠ 0 ∧ -2x₀ + a ≠ 0 ∧
    (2x₀ - 2) * (-2x₀ + a) = -1 ∧ x₀^2 - 2*x₀ + 2 = -x₀^2 + a*x₀ + b) →
    min_value a b (a > 0 ∧ b > 0) = 18 / 5 := 
by
  intros a b h h_intersection
  sorry

end minimum_value_of_expression_l319_319895


namespace petya_wins_best_actions_l319_319231

/--
Petya and Vasya take turns breaking a stick. Petya goes first. 
They both try to create four parts whose lengths form an arithmetic progression.
An arithmetic progression is defined by four numbers \( a_1, a_2, a_3, a_4 \) such that
\( a_2 - a_1 = a_3 - a_2 = a_4 - a_3 \). Prove that Petya wins if both players take
their best possible actions.
-/
theorem petya_wins_best_actions (l p x : ℝ) (h : 2 * x = p) :
  (∃ a₁ a₂ a₃ a₄ : ℝ, a₁ = l - p ∧ a₂ = l - x ∧ a₃ = l + x ∧ a₄ = l + p ∧ 
    a₂ - a₁ = a₃ - a₂ ∧ a₄ - a₃ = a₂ - a₁) :=
begin
  use [l - p, l - x, l + x, l + p],
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { simp [h], },
  { simp [h], },
end

end petya_wins_best_actions_l319_319231


namespace polynomial_remainder_l319_319428

theorem polynomial_remainder (x : ℂ) : 
  polynomial.remainder ((polynomial.X) ^ 2021 + 1) 
                       ((polynomial.X) ^ 8 - (polynomial.X) ^ 6 + (polynomial.X) ^ 4 - (polynomial.X) ^ 2 + 1) 
  = (polynomial.X - 1) :=  
sorry

end polynomial_remainder_l319_319428


namespace length_of_plot_l319_319653

theorem length_of_plot 
  (b : ℝ)
  (H1 : 2 * (b + 20) + 2 * b = 5300 / 26.50)
  : (b + 20 = 60) :=
sorry

end length_of_plot_l319_319653


namespace nuts_eventually_zero_l319_319253

noncomputable def nut_transfer_condition
  (num_people : ℕ := 10)
  (total_nuts : ℕ := 100)
  (nuts : Fin num_people → ℕ)
  (transfer_rule : ∀ k, nuts k % 2 = 0 → nuts k / 2 ∧ nuts k / 2) -- even case rule
  (transfer_rule_odd : ∀ k, nuts k % 2 = 1 → (nuts k / 2 + 1 / 2 : ℤ) ∧ (nuts k / 2 - 1 / 2 : ℤ)) -- odd case rule
  : Prop :=
  -- This condition is used to outline the rules of nut transfer.

theorem nuts_eventually_zero (nuts : Fin 10 → ℕ) :
  sum (λ k, nuts k) = 100 →
  (∀ k, nuts k % 2 = 0 → nuts k / 2 ∧ nuts k / 2) →
  (∀ k, nuts k % 2 = 1 → (nuts k / 2 + 1 / 2 : ℤ) ∧ (nuts k / 2 - 1 / 2 : ℤ)) →
  ∃ n, ∀ m ≥ n, ∀ k, nuts k = 0 :=
begin
  sorry
end

end nuts_eventually_zero_l319_319253


namespace proof1_proof2_false_proof3_proof4_false_proof5_l319_319874

-- Condition 1
def condition1 (α β : ℝ) : Prop :=
  α + β = 7 * Real.pi / 4

theorem proof1 (α β : ℝ) (h : condition1 α β) : 
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := 
sorry

-- Condition 2
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (λ : ℝ) : ℝ × ℝ := (2, λ)

def condition2 (λ : ℝ) : Prop :=
  λ < 1

theorem proof2_false (λ : ℝ) : 
  (let a := vector_a 
   let b := vector_b λ 
   (a.1 * b.1 + a.2 * b.2) > 0) ↔ ¬ condition2 λ :=
sorry

-- Condition 3
structure Point (α : Type) :=
(x : α) (y : α)

def condition3 (O A B C : Point ℝ) (λ : ℝ) (P : Point ℝ) : Prop :=
  λ > 0 ∧ λ < Real.infinity ∧ 
  (P.x = O.x + λ * (B.x - A.x + C.x - A.x)) ∧ 
  (P.y = O.y + λ * (B.y - A.y + C.y - A.y))

theorem proof3 (O A B C P : Point ℝ) (λ : ℝ) (h : condition3 O A B C λ P) : 
  (let G := Point.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) in P = G) :=
sorry

-- Condition 4
def triangle (A B C : Point ℝ) : Prop := 
  A ≠ B ∧ B ≠ C ∧ A ≠ C

def condition4 (A B C : Point ℝ) (angle_A : ℝ) (a c : ℝ) : Prop :=
  angle_A = Real.pi / 3 ∧ a = 4 ∧ c = 3 * Real.sqrt 3 ∧ triangle A B C

theorem proof4_false (A B C : Point ℝ) (angle_A : ℝ) (a c : ℝ) (h : condition4 A B C angle_A a c) : 
  ¬ ∃ B C, triangle A B C :=
sorry

-- Condition 5
def condition5 (A B C : Point ℝ) (R : ℝ) : Prop :=
  ∃ a b, 2 * R * (Real.sin (A.x) ^ 2 - Real.sin (C.x) ^ 2) = (Real.sqrt 2 * a - b) * Real.sin (B.x)

theorem proof5 (A B C : Point ℝ) (R a b : ℝ) (h : condition5 A B C R) : 
  let max_area := (Real.sqrt 2 + 1) / 2 * R^2 in 
  ∃ S, S ≤ max_area :=
sorry

end proof1_proof2_false_proof3_proof4_false_proof5_l319_319874


namespace distance_travelled_l319_319322

variables (S D : ℝ)

-- conditions
def cond1 : Prop := D = S * 7
def cond2 : Prop := D = (S + 12) * 5

-- Define the main theorem
theorem distance_travelled (h1 : cond1 S D) (h2 : cond2 S D) : D = 210 :=
by {
  sorry
}

end distance_travelled_l319_319322


namespace perfect_number_29_perfect_expression_x_squared_minus_4x_plus_5_perfect_expression_x_y_squared_minus_2x_plus_4y_plus_5_eq_zero_l319_319268

-- Question 1 Lean 4 Statement
theorem perfect_number_29 : ∃ a b : ℤ, 29 = a^2 + b^2 :=
by {
  use [2, 5],
  norm_num,
  exact eq.refl 29
}

-- Question 2 Lean 4 Statement
theorem perfect_expression_x_squared_minus_4x_plus_5 : ∃ m n : ℤ, (x - m)^2 + n = x^2 - 4x + 5 :=
by {
  use [2, 1],
  norm_num,
  exact eq.refl ((x - m)^2 + n)
}

-- Exploring problem Lean 4 Statement
theorem perfect_expression_x_y_squared_minus_2x_plus_4y_plus_5_eq_zero (x y : ℝ) (h : x^2 + y^2 - 2x + 4y + 5 = 0) :
  x + y = -1 :=
by {
  have h1: (x - 1)^2 + (y + 2)^2 = 0,
  {  },
  have eq0_1: (x - 1)^2 = 0 := sorry,
  have eq0_2: (y + 2)^2 = 0 := sorry,
  have sol1: x = 1 := eq0_1.elim_left,
  have sol2: y = -2 := eq0_2.elim_left,
  rw [sol1, sol2],
  norm_num
}

end perfect_number_29_perfect_expression_x_squared_minus_4x_plus_5_perfect_expression_x_y_squared_minus_2x_plus_4y_plus_5_eq_zero_l319_319268


namespace find_b_l319_319154

variables {a b : ℝ}

theorem find_b (h1 : (x - 3) * (x - a) = x^2 - b * x - 10) : b = -1/3 :=
  sorry

end find_b_l319_319154


namespace largest_quantity_l319_319317

noncomputable def A := (2006 / 2005) + (2006 / 2007)
noncomputable def B := (2006 / 2007) + (2008 / 2007)
noncomputable def C := (2007 / 2006) + (2007 / 2008)

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l319_319317


namespace symmetry_axis_l319_319650

variable (A : ℝ) (ϕ : ℝ)
def f (x : ℝ) : ℝ := A * Real.cos x
def g (x : ℝ) : ℝ := A * Real.cos (x - ϕ)
axiom A_ne_zero : A ≠ 0
axiom phi_range : 0 < ϕ ∧ ϕ < Real.pi
axiom tangent_at_zero : ∀ x : ℝ, (Real.tangent_line g 0 = x + Real.sqrt 3)

theorem symmetry_axis : ∃ k : ℤ, g (x : ℝ) = A * Real.cos (x - ϕ) → (x = ϕ + k * Real.pi) := sorry

end symmetry_axis_l319_319650


namespace abs_a1_b1_eq_4_l319_319271

theorem abs_a1_b1_eq_4 :
  ∃ (a_1 a_2 ⋯ a_m b_1 b_2 ⋯ b_n : Nat),
    (2520 = (a_1.factorial * a_2.factorial * ⋯ * a_m.factorial) / (b_1.factorial * b_2.factorial * ⋯ * b_n.factorial)) ∧
    (a_1 ≥ a_2) ∧ ⋯ ∧ (a_m ≥ 1) ∧
    (b_1 ≥ b_2) ∧ ⋯ ∧ (b_n ≥ 1) ∧
    (a_1 + b_1 = min (a_1 + b_1)) ∧
    (abs (a_1 - b_1) = 4) :=
sorry

end abs_a1_b1_eq_4_l319_319271


namespace digit_150_fraction_l319_319686

theorem digit_150_fraction :
  let frac := (75 / 625 : Real)
  (Real.frac_to_dec frac).decimalNth 150 = 2 :=
by
  sorry

end digit_150_fraction_l319_319686


namespace probability_major_A_less_than_25_l319_319555

def total_students : ℕ := 100 -- assuming a total of 100 students for simplicity

def male_percent : ℝ := 0.40
def major_A_percent : ℝ := 0.50
def major_B_percent : ℝ := 0.30
def major_C_percent : ℝ := 0.20
def major_A_25_or_older_percent : ℝ := 0.60
def major_A_less_than_25_percent : ℝ := 1 - major_A_25_or_older_percent

theorem probability_major_A_less_than_25 :
  (major_A_percent * major_A_less_than_25_percent) = 0.20 :=
by
  sorry

end probability_major_A_less_than_25_l319_319555


namespace color_boxes_problem_l319_319879

theorem color_boxes_problem :
  ∀ (n : ℕ),
  (∃ bix : ℕ → fin 8 → fin 6 → fin n, (∀ i : fin 8, function.injective (bix i)) ∧
     (∀ i j : fin 8, i ≠ j → ∀ (a b : fin 6), bix i a ≠ bix j b))
  ↔ n ≥ 23 :=
by
  sorry

end color_boxes_problem_l319_319879


namespace arithmetic_sequence_sum_l319_319929

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ)
variable (h_arith : is_arithmetic_sequence a)
variable (h_sum : a 2 + a 3 + a 10 + a 11 = 48)

-- Goal
theorem arithmetic_sequence_sum : a 6 + a 7 = 24 :=
sorry

end arithmetic_sequence_sum_l319_319929


namespace max_value_is_27_l319_319159

noncomputable def max_value_of_expression (a b c : ℝ) : ℝ :=
  (a - b)^2 + (b - c)^2 + (c - a)^2

theorem max_value_is_27 (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 9) : max_value_of_expression a b c = 27 :=
by
  sorry

end max_value_is_27_l319_319159


namespace Mike_monthly_time_is_200_l319_319229

def tv_time (days : Nat) (hours_per_day : Nat) : Nat := days * hours_per_day

def video_game_time (total_tv_time_per_week : Nat) (num_days_playing : Nat) : Nat :=
  (total_tv_time_per_week / 7 / 2) * num_days_playing

def piano_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours * 5 + weekend_hours * 2

def weekly_time (tv_time : Nat) (video_game_time : Nat) (piano_time : Nat) : Nat :=
  tv_time + video_game_time + piano_time

def monthly_time (weekly_time : Nat) (weeks : Nat) : Nat :=
  weekly_time * weeks

theorem Mike_monthly_time_is_200 : monthly_time
  (weekly_time 
     (tv_time 3 4 + tv_time 2 3 + tv_time 2 5) 
     (video_game_time 28 3) 
     (piano_time 2 3))
  4 = 200 :=
  by
  sorry

end Mike_monthly_time_is_200_l319_319229


namespace ratio_of_sides_l319_319836

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l319_319836


namespace mixture_price_l319_319842

-- Define the problem that involves candies priced at 2 and 3 rubles per kilogram
variable (s : ℝ) 

-- The weight of candies priced at 2 rubles per kilogram
def weight_candy_2 := s / 2

-- The weight of candies priced at 3 rubles per kilogram
def weight_candy_3 := s / 3

-- The total weight of the mixture
def total_weight := weight_candy_2 s + weight_candy_3 s

-- The total cost of the mixture
def total_cost := 2 * weight_candy_2 s + 3 * weight_candy_3 s

-- The price per kilogram of the mixture
def price_per_kilogram := total_cost s / total_weight s

-- Assert that the price per kilogram of the mixture is 2.4 rubles
theorem mixture_price (s : ℝ) (h: s ≠ 0) : price_per_kilogram s = 2.4 :=
by {
  -- Proof pending
  sorry
}

end mixture_price_l319_319842


namespace smallest_non_representable_l319_319429

def isRepresentable (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ n : ℕ, 0 < n → ¬ isRepresentable 11 ∧ ∀ k : ℕ, 0 < k ∧ k < 11 → isRepresentable k :=
by sorry

end smallest_non_representable_l319_319429


namespace calculate_f_x_plus_1_sub_f_x_l319_319485

noncomputable def f (x : ℝ) : ℝ := 3^(2 * x)

theorem calculate_f_x_plus_1_sub_f_x (x : ℝ) : f(x + 1) - f(x) = 8 * f(x) :=
by
  have h1 : f(x + 1) = 3^(2 * (x + 1)) := rfl
  have h2 : 3^(2 * (x + 1)) = 3^(2 * x + 2) := by sorry
  have h3 : 3^(2 * x + 2) = 3^(2 * x) * 9 := by sorry
  have h4 : 3^(2 * x) * 9 - 3^(2 * x) = 3^(2 * x) * (9 - 1) := by sorry
  have h5 : 3^(2 * x) * (9 - 1) = 3^(2 * x) * 8 := by sorry
  have h6 : f(x + 1) - f(x) = 8 * f(x) := by sorry
  exact h6

end calculate_f_x_plus_1_sub_f_x_l319_319485


namespace equation_has_151_real_solutions_l319_319052

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 150).sum (λ n, (n+1 : ℝ) / (x - (n+1 : ℝ)))

def g (x : ℝ) : ℝ := x ^ 2

theorem equation_has_151_real_solutions :
  ∃ (s : Finset ℝ), s.card = 151 ∧ ∀ x ∈ s, f x = g x :=
sorry

end equation_has_151_real_solutions_l319_319052


namespace frog_path_problem_l319_319594

noncomputable def a : ℕ → ℕ
| 2 * n - 1 => 0
| 2 * n     => (2 + real.sqrt 2) ^ (n - 1) / real.sqrt 2 - (2 - real.sqrt 2) ^ (n - 1) / real.sqrt 2

theorem frog_path_problem (n : ℕ) (a : ℕ → ℕ) : 
  (a (2 * n - 1) = 0) ∧ (a (2 * n) = ((2 + real.sqrt 2) ^ (n - 1) / real.sqrt 2 - (2 - real.sqrt 2) ^ (n - 1) / real.sqrt 2)) := 
sorry

end frog_path_problem_l319_319594


namespace determinant_of_A_l319_319411

noncomputable def matrix_A (α β γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![cos α * cos β, cos α * sin β, -sin α * cos γ],
    ![-sin β, cos β, sin γ],
    ![sin α * cos β, sin α * sin β, cos α * cos γ]
  ]

theorem determinant_of_A (α β γ : ℝ) :
  det (matrix_A α β γ) = cos α * cos γ :=
  sorry

end determinant_of_A_l319_319411


namespace intersection_A_B_l319_319104

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_A_B : A ∩ B = {1} := 
by
  sorry

end intersection_A_B_l319_319104


namespace probability_of_selecting_standard_parts_l319_319995

theorem probability_of_selecting_standard_parts :
  (∃ (N M n m : ℕ), N = 12 ∧ M = 8 ∧ n = 5 ∧ m = 3 ∧ 
  (fact M / (fact m * fact (M - m)) * fact (N - M) / (fact (n - m) * fact (N - M - (n - m))) / (fact N / (fact n * fact (N - n))) = 14 / 33)) :=
begin
  use [12, 8, 5, 3],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  sorry,
end

end probability_of_selecting_standard_parts_l319_319995


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319765

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319765


namespace distinct_paintings_count_l319_319177

theorem distinct_paintings_count :
  let disks := 6
  let blue := 3
  let red := 2
  let green := 1
  let symmetries := rotations_and_reflections_of_hexagon
  distinct_paintings(disks, blue, red, green, symmetries) = 12 :=
sorry

end distinct_paintings_count_l319_319177


namespace pears_total_l319_319582

-- Conditions
def keith_initial_pears : ℕ := 47
def keith_given_pears : ℕ := 46
def mike_initial_pears : ℕ := 12

-- Define the remaining pears
def keith_remaining_pears : ℕ := keith_initial_pears - keith_given_pears
def mike_remaining_pears : ℕ := mike_initial_pears

-- Theorem statement
theorem pears_total :
  keith_remaining_pears + mike_remaining_pears = 13 :=
by
  sorry

end pears_total_l319_319582


namespace dot_product_ABC_l319_319167

open Real

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 6
noncomputable def angleC : ℝ := π / 6  -- 30 degrees in radians

theorem dot_product_ABC :
  let CB := a
  let CA := b
  let angle_between := π - angleC  -- 150 degrees in radians
  let cos_angle := - (sqrt 3) / 2  -- cos(150 degrees)
  ∃ (dot_product : ℝ), dot_product = CB * CA * cos_angle :=
by
  have CB := a
  have CA := b
  have angle_between := π - angleC
  have cos_angle := - (sqrt 3) / 2
  use CB * CA * cos_angle
  sorry

end dot_product_ABC_l319_319167


namespace smallest_positive_period_of_f_max_value_of_f_min_value_of_f_monotonically_increasing_intervals_of_f_l319_319954

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 1 - Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem smallest_positive_period_of_f : 
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ ∀ T' > 0, (T' < T → ∃ x : ℝ, f (x + T') ≠ f x) :=
sorry

theorem max_value_of_f : 
  ∃ x : ℝ, f x = √2 + 1 :=
sorry

theorem min_value_of_f : 
  ∃ x : ℝ, f x = -√2 + 1 :=
sorry

theorem monotonically_increasing_intervals_of_f : 
  ∀ k : ℤ, ∃ I : Set ℝ, 
    I = Set.Icc (k * Real.pi - Real.pi / 8) (k * Real.pi + 3 * Real.pi / 8) ∧ 
    ∀ x₁ x₂ ∈ I, x₁ ≤ x₂ → f x₁ ≤ f x₂ :=
sorry

end smallest_positive_period_of_f_max_value_of_f_min_value_of_f_monotonically_increasing_intervals_of_f_l319_319954


namespace smallest_tangent_line_l319_319064

noncomputable def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

def tangent_line (m x₀ y₀ : ℝ) : ℝ → ℝ := λ x, m * (x - x₀) + y₀

theorem smallest_tangent_line :
  ∃ x₀ y₀, y₀ = curve x₀ ∧ tangent_line 3 (-1) (-14) = λ x, 3 * (x + 1) + -14 :=
begin
  use [-1, -14],
  split,
  { simp [curve], },
  { funext,
    simp [tangent_line],
    linarith }
end

end smallest_tangent_line_l319_319064


namespace find_a_l319_319400

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem find_a : (a : ℝ) (E a 3 10 = E a 5 (-2)) → a = 3 / 4 :=
by {
    assume h,
    sorry
}

end find_a_l319_319400


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319740

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319740


namespace leakage_time_l319_319828

theorem leakage_time (a : ℝ) (h : a > 0) : 
  let l := (7 * a) / 6 
  in 1 / a - 1 / l = 1 / (7 * a) :=
by
  let l := (7 * a) / 6
  sorry

end leakage_time_l319_319828


namespace relationship_among_abc_l319_319441

theorem relationship_among_abc (a b c : ℝ)
  (ha : a = 0.3^2)
  (hb : b = Real.log 0.3 / Real.log 2)
  (hc : c = 2^0.3) :
  b < a ∧ a < c := 
by
  sorry

end relationship_among_abc_l319_319441


namespace distance_from_center_to_point_l319_319306

theorem distance_from_center_to_point :
  let circle_equation := λ (x y : ℝ), x^2 + y^2 = 6*x + 8*y + 9
  let point := (5 : ℝ, -3 : ℝ)
  let center := (3 : ℝ, 4 : ℝ)
  let distance := real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2)
  distance = real.sqrt 53 :=
by
  -- Definitions of the circle equation and point are given as conditions 
  let circle_equation := λ (x y : ℝ), x^2 + y^2 = 6*x + 8*y + 9
  let point := (5 : ℝ, -3 : ℝ)
  let center := (3 : ℝ, 4 : ℝ)
  let distance := real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2)
  show distance = real.sqrt 53
  sorry

end distance_from_center_to_point_l319_319306


namespace total_cost_for_tickets_l319_319804

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end total_cost_for_tickets_l319_319804


namespace find_counterfeit_two_weighings_l319_319561

theorem find_counterfeit_two_weighings (coins : Fin 23 → ℝ) 
  (num_counterfeit : Finset (Fin 23)) :
  num_counterfeit.card = 6 ∧ 
  ((∀ i j : Fin 23, i ∉ num_counterfeit → j ∉ num_counterfeit → coins i = coins j) ∧ 
  ∃ i j : Fin 23, i ∈ num_counterfeit ∧ j ∈ num_counterfeit ∧ coins i ≠ coins j) 
  → ∃ fake_coin : Fin 23, fake_coin ∈ num_counterfeit :=
begin
  sorry
end

end find_counterfeit_two_weighings_l319_319561


namespace john_final_price_l319_319581

theorem john_final_price : 
  let goodA_price := 2500
  let goodA_rebate := 0.06 * goodA_price
  let goodA_price_after_rebate := goodA_price - goodA_rebate
  let goodA_sales_tax := 0.10 * goodA_price_after_rebate
  let goodA_final_price := goodA_price_after_rebate + goodA_sales_tax
  
  let goodB_price := 3150
  let goodB_rebate := 0.08 * goodB_price
  let goodB_price_after_rebate := goodB_price - goodB_rebate
  let goodB_sales_tax := 0.12 * goodB_price_after_rebate
  let goodB_final_price := goodB_price_after_rebate + goodB_sales_tax

  let goodC_price := 1000
  let goodC_rebate := 0.05 * goodC_price
  let goodC_price_after_rebate := goodC_price - goodC_rebate
  let goodC_sales_tax := 0.07 * goodC_price_after_rebate
  let goodC_final_price := goodC_price_after_rebate + goodC_sales_tax

  let total_amount := goodA_final_price + goodB_final_price + goodC_final_price

  let special_voucher_discount := 0.03 * total_amount
  let final_price := total_amount - special_voucher_discount
  let rounded_final_price := Float.round final_price

  rounded_final_price = 6642 := by
  sorry

end john_final_price_l319_319581


namespace problem_M_value_l319_319036

-- Define the triplets sum that alternates addition and subtraction of squares
def triplet_sum (n : ℕ) : ℤ :=
  (2 * n + 2)^2 + (2 * n)^2 - (2 * n - 2)^2

-- Define the total sum for the given problem
def M : ℤ :=
  (List.range 25).sum (λ i, triplet_sum (25 - i))

theorem problem_M_value : M = 2600 :=
by
  -- Proof is omitted
  sorry

end problem_M_value_l319_319036


namespace proof_problem_l319_319923

theorem proof_problem 
  (α : ℝ) 
  (hα : α ∈ set.Ioo (π / 2) (3 * π / 4))
  (hAC_BC_perp : let A := (4 : ℝ, 0 : ℝ)
                  let B := (0 : ℝ, 4 : ℝ)
                  let C := (3 * real.cos α, 3 * real.sin α)
                  let AC := (C.1 - A.1, C.2 - A.2)
                  let BC := (C.1 - B.1, C.2 - B.2)
                  AC.1 * BC.1 + AC.2 * BC.2 = 0) :
  (2 * (real.sin α)^2 - real.sin (2 * α)) / (1 + real.tan α) = - (7 * real.sqrt 23) / 48 :=
by
  sorry

end proof_problem_l319_319923


namespace value_of_mn_l319_319200

theorem value_of_mn (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : m^4 - n^4 = 3439) : m * n = 90 := 
by sorry

end value_of_mn_l319_319200


namespace product_of_roots_eq_neg4_l319_319070

theorem product_of_roots_eq_neg4 :
  (∃ x : ℝ, x^2 + 3 * x - 4 = 0) → 
  (∀ a b c : ℝ, a ≠ 0 → (a * (x ^ 2) + b * x + c = 0 → ((x * x).all? () = -4))) :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  have h1 : x^2 + 3 * x - 4 = 0 := hx
  let a := 1
  let b := 3
  let c := -4
  have ha : a ≠ 0 := by norm_num
  have H : ∀ x y, (a - (x * x) + b * x + c = 0) → (x * x * c / a = (-4) / 1) := 
    by sorry
  exact H

end product_of_roots_eq_neg4_l319_319070


namespace greatest_multiple_of_5_and_6_under_1000_l319_319731

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319731


namespace asymptote_of_hyperbola_l319_319640

theorem asymptote_of_hyperbola : 
  (∀ x y : ℝ, 9 * x^2 - 16 * y^2 = 144 → y = 3 / 4 * x ∨ y = - (3 / 4) * x) :=
by
  unfold (* Definitions being tested *)
  sorry (* Proof steps are not required. *)

end asymptote_of_hyperbola_l319_319640


namespace intersect_on_f_l319_319460

-- Definitions of points and lines in a plane
variables {Point Line : Type}
variables {O P : Point} {e f g : Line}

-- Assuming reflections across lines e, f, and g
constant reflection : Point → Line → Point
axiom reflection_e : reflection P e = P1
axiom reflection_f : reflection P1 f = P2
axiom reflection_g : reflection P2 g = P3

-- Intersection points of the line PP3 with lines e and g
axiom intersection_e : ∃ Q1 : Point, ∃ Q2 : Point, (line_through P P3) ∩ e = Q1 ∧ (line_through P P3) ∩ g = Q2

-- Defining the lines formed by intersection
noncomputable def Q1_line := line_through Q1 P1
noncomputable def Q2_line := line_through Q2 P2

-- Prove the lines Q1P1 and Q2P2 intersect on f
theorem intersect_on_f : ∃ Q3 : Point, Q3 ∈ f ∧ Q3 ∈ Q1_line ∧ Q3 ∈ Q2_line := 
sorry

end intersect_on_f_l319_319460


namespace find_a_l319_319573

open Real

theorem find_a :
  (∃ (a : ℝ), a > 0 ∧ (∀ (θ : ℝ), (θ = real.arctan 2) →
                        (let eq1 := ρ^2 - 2 * ρ * real.sin θ + 1 - a^2 in
                         eq1 = 0 → (let ρ := 4 * real.cos θ in true)))) →
    a = 1 :=
by
sor

end find_a_l319_319573


namespace point_on_line_l319_319985

theorem point_on_line (x : ℝ) :
  ∀ (p₁ p₂ : ℝ × ℝ),
  p₁ = (0, 8) →
  p₂ = (-4, 0) →
  (x, -4) ∈ line_through p₁ p₂ →
  x = -6 :=
by
  intros p₁ p₂ hp₁ hp₂ hx
  rw [← hp₁, ← hp₂] at hx
  sorry

end point_on_line_l319_319985


namespace find_a_l319_319949

noncomputable def f (a : ℝ) (x : ℝ) := a * real.sqrt x

noncomputable def f' (a : ℝ) (x : ℝ) := (a / (2 * real.sqrt x))

theorem find_a (a : ℝ) (h : f' a 1 = 1) : a = 2 :=
by 
  sorry

end find_a_l319_319949


namespace black_to_white_area_ratio_l319_319907

theorem black_to_white_area_ratio (r1 r2 r3 r4 : ℝ) (h1 : r1 = 2) (h2 : r2 = 4) (h3 : r3 = 6) (h4 : r4 = 8) :
  let area := λ r : ℝ, Real.pi * r^2,
      black_area := (area r2 - area r1) + (area r4 - area r3),
      white_area := (area r1) + (area r3 - area r2)
  in black_area / white_area = 5 / 3 :=
by
  sorry

end black_to_white_area_ratio_l319_319907


namespace intersection_tangent_line_l319_319816

noncomputable def tangent_line_intersection
  (radius1 radius2 : ℝ) (center1 center2 : ℝ × ℝ) (x : ℝ) : Prop :=
  let intersection_point := (x, 0) in
  let hr1 := radius1 = 3 in
  let hc1 := center1 = (0, 0) in
  let hr2 := radius2 = 5 in
  let hc2 := center2 = (12, 0) in
  let tangent_condition := x = 18 in
  hr1 ∧ hr2 ∧ hc1 ∧ hc2 ∧ tangent_condition

theorem intersection_tangent_line :
  ∃ x : ℝ, tangent_line_intersection 3 5 (0, 0) (12, 0) x := 
begin
  use 18,
  dsimp [tangent_line_intersection],
  simp,
  sorry
end

end intersection_tangent_line_l319_319816


namespace side_length_of_square_l319_319814

noncomputable def area_of_circle : ℝ := 3848.4510006474966
noncomputable def pi : ℝ := Real.pi

theorem side_length_of_square :
  ∃ s : ℝ, (∃ r : ℝ, area_of_circle = pi * r * r ∧ 2 * r = s) ∧ s = 70 := 
by
  sorry

end side_length_of_square_l319_319814


namespace coefficient_x_in_expansion_l319_319643

theorem coefficient_x_in_expansion :
  let general_term (r : ℕ) := binom 6 r * (2 : ℂ)^(6 - r) * (-1 : ℂ)^r * x^(6 - 2 * r) in
  let constant_term := ∑ r in range 7, if 6 - 2 * r = 0 then general_term r else 0 in
  constant_term = -160 →
  (∀ r : ℤ, 6 - 2 * r ≠ 1) →
  let original_expr := (1 / 2 * x - 1) * polynomial.expand (-160 : ℂ) in
  coefficient original_expr 1 = (-80 : ℂ) :=
  sorry

end coefficient_x_in_expansion_l319_319643


namespace largest_independent_set_size_l319_319821

theorem largest_independent_set_size (n k : ℕ) (kn_pos : 0 < k) 
  (h : ∀ (A B : Finset (Fin n)), A ∩ B = ∅ → A ∪ B = Finset.univ → (∑ x in A, (∑ y in B, @has_edge _ _ x y)) ≤ kn) : 
    ∃ m, m = Int.ceil (n / (4 * k)) ∧ ∀ (S : Finset (Fin n)), 
    S.card = m → ∀ (x y : Fin n), x ∈ S → y ∈ S → ¬ @has_edge _ _ x y := sorry

end largest_independent_set_size_l319_319821


namespace find_circle_through_points_l319_319063

def is_circle (a b c d : ℝ) : Prop :=
  ∀ x y, x^2 + y^2 + a * x + b * y + c = d

theorem find_circle_through_points :
  ∃ a b c d : ℝ, is_circle a b c d ∧
  (∀ x y, (x, y) = (2, -2) → x^2 + y^2 - 5 * x = 0) ∧
  (∀ x y, (x, y) = (2, -2) → x^2 + y^2 = 2) ∧
  (∀ x y, (x, y) = (2, -2) → x^2 + y^2 - 5 * x + (1 / 3) * (x^2 + y^2 - 2) = 0) ∧
  (∀ x y, is_circle a b c d → x^2 + y^2 - (15 / 4) * x - (1 / 2) = 0) :=
begin
  sorry
end

end find_circle_through_points_l319_319063


namespace researcher_can_cross_desert_l319_319010

structure Condition :=
  (distance_to_oasis : ℕ)  -- total distance to be covered
  (travel_per_day : ℕ)     -- distance covered per day
  (carry_capacity : ℕ)     -- maximum days of supplies they can carry
  (ensure_return : Bool)   -- flag to ensure porters can return
  (cannot_store_food : Bool) -- flag indicating no food storage in desert

def condition_instance : Condition :=
{ distance_to_oasis := 380,
  travel_per_day := 60,
  carry_capacity := 4,
  ensure_return := true,
  cannot_store_food := true }

theorem researcher_can_cross_desert (cond : Condition) : cond.distance_to_oasis = 380 
  ∧ cond.travel_per_day = 60 
  ∧ cond.carry_capacity = 4 
  ∧ cond.ensure_return = true 
  ∧ cond.cannot_store_food = true 
  → true := 
by 
  sorry

end researcher_can_cross_desert_l319_319010


namespace horse_revolutions_l319_319356

-- Defining the problem conditions
def radius_outer : ℝ := 30
def radius_inner : ℝ := 10
def revolutions_outer : ℕ := 25

-- The question we need to prove
theorem horse_revolutions :
  (revolutions_outer : ℝ) * (radius_outer / radius_inner) = 75 := 
by
  sorry

end horse_revolutions_l319_319356


namespace a_n_formula_b_n_arithmetic_sum_c_n_l319_319575
noncomputable def a_seq (n : ℕ) : ℝ := (1 / 4) ^ n

def b_seq (n : ℕ) : ℝ := 3 * n - 2

def c_seq (n : ℕ) : ℝ := a_seq n + b_seq n

def sum_c_seq (n : ℕ) := ∑ i in Finset.range n, c_seq (i + 1)

theorem a_n_formula (n : ℕ) : a_seq n = (1 / 4) ^ n :=
  sorry  -- Proof of the general formula for the sequence a_n

theorem b_n_arithmetic (n : ℕ) : b_seq n = 3 * n - 2 :=
  sorry  -- Proof that the sequence b_n is arithmetic

theorem sum_c_n (n : ℕ) : sum_c_seq n = (3 * n^2 - n) / 2 + 1 / 3 - (1 / 3) * (1 / 4)^n :=
  sorry  -- Proof of the sum of the first n terms of c_n

end a_n_formula_b_n_arithmetic_sum_c_n_l319_319575


namespace volume_of_T_is_correct_l319_319663

noncomputable def volume_of_T : ℝ :=
  32 * real.sqrt 3 / 9

theorem volume_of_T_is_correct :
  ∀ (T : set (ℝ × ℝ × ℝ)),
    (∀ x y z, ((0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y ≤ 2) ∨
               (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + z ≤ 2) ∨
               (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ y + z ≤ 2)) ∧
              T = {(x, y, z) | |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2}) →
    ∃ V : ℝ, V = volume_of_T :=
  sorry

end volume_of_T_is_correct_l319_319663


namespace ratio_adult_women_to_men_event_l319_319644

theorem ratio_adult_women_to_men_event :
  ∀ (total_members men_ratio women_ratio children : ℕ), 
  total_members = 2000 →
  men_ratio = 30 →
  children = 200 →
  women_ratio = men_ratio →
  women_ratio / men_ratio = 1 / 1 := 
by
  intros total_members men_ratio women_ratio children
  sorry

end ratio_adult_women_to_men_event_l319_319644


namespace ten_pow_n_plus_eight_div_nine_is_integer_l319_319621

theorem ten_pow_n_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, 10^n + 8 = 9 * k := 
sorry

end ten_pow_n_plus_eight_div_nine_is_integer_l319_319621


namespace skew_lines_projection_l319_319274

theorem skew_lines_projection (L₁ L₂ : Line) (P : Plane) (h_skew : skew L₁ L₂) :
  ∃ (intersecting_lines parallel_lines line_and_point : Prop), 
    (projections_of_skew_lines L₁ L₂ P = intersecting_lines ∨
     projections_of_skew_lines L₁ L₂ P = parallel_lines ∨
     projections_of_skew_lines L₁ L₂ P = line_and_point)
  :=
by
  sorry

end skew_lines_projection_l319_319274


namespace solve_system_l319_319635

-- Define the system of equations
def eq1 (x y : ℚ) : Prop := 4 * x - 3 * y = -10
def eq2 (x y : ℚ) : Prop := 6 * x + 5 * y = -13

-- Define the solution
def solution (x y : ℚ) : Prop := x = -89 / 38 ∧ y = 0.21053

-- Prove that the given solution satisfies both equations
theorem solve_system : ∃ x y : ℚ, eq1 x y ∧ eq2 x y ∧ solution x y :=
by
  sorry

end solve_system_l319_319635


namespace positive_integer_pairs_count_l319_319427

theorem positive_integer_pairs_count :
  (∃ (n : ℕ), n = 31 ∧ 
  (∀ (a b : ℕ), a^2 + b^2 < 2013 ∧ a^2 * b ∣ b^3 - a^3 → a = b)) :=
by
  sorry

end positive_integer_pairs_count_l319_319427


namespace lifting_equivalence_l319_319612

theorem lifting_equivalence :
  ∃ n : ℕ, (n = 12 ∨ n = 13) ∧ (2 * 20 * n ≥ 2 * 25 * 10) :=
by
  have total_weight_25 := 2 * 25 * 10
  have n_12 := 2 * 20 * 12
  have n_13 := 2 * 20 * 13
  use [12, 13]
  split
  case inl =>
    left
    exact rfl
  case inr =>
    right
    exact rfl
  split
  case inl =>
    exact n_12
  case inr =>
    exact n_13
  sorry

end lifting_equivalence_l319_319612


namespace hannahs_savings_l319_319135

theorem hannahs_savings(
  savings_goal : ℕ := 80,
  first_week : ℕ := 4,
  second_week : ℕ := 4 * 2,
  third_week : ℕ := 8 * 2,
  fourth_week : ℕ := 16 * 2,
  total_four_weeks : ℕ := 4 + 8 + 16 + 32
  ) : 
  first_week + second_week + third_week + fourth_week ≤ savings_goal → 
  savings_goal - total_four_weeks = 20 :=
by
  intro h
  sorry

end hannahs_savings_l319_319135


namespace part1_l319_319917

def sequence (a : ℕ → ℤ) := a 1 = 1 ∧ a 2 = 3 ∧ ∀ n, a (n + 2) = 3 * a (n + 1) - a n

theorem part1 (a : ℕ → ℤ) (h : sequence a) (n : ℕ) :
  1 + a n * a (n + 2) = a (n + 1) * a (n + 1) := sorry

end part1_l319_319917


namespace sahil_percentage_profit_l319_319627

-- Define the constants for the conditions.
def purchase_price : ℝ := 14000
def repair_costs : ℝ := 5000
def transportation_charges : ℝ := 1000
def selling_price : ℝ := 30000

-- Define the total cost.
def total_cost : ℝ := purchase_price + repair_costs + transportation_charges

-- Define the profit.
def profit : ℝ := selling_price - total_cost

-- Define the percentage of profit.
def percentage_profit : ℝ := (profit / total_cost) * 100

-- The theorem to be proved.
theorem sahil_percentage_profit : percentage_profit = 50 := by
  -- this is where the proof would go
  sorry

end sahil_percentage_profit_l319_319627


namespace complex_abs_value_l319_319530

theorem complex_abs_value :
  ∀ z : ℂ, (1 - 2 * complex.I)^2 / z = 4 - 3 * complex.I → complex.abs z = 1 :=
by
  intros z h
  sorry

end complex_abs_value_l319_319530


namespace num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l319_319147

/-- The number of positive integers n such that 300 < n^2 < 1200 is 17. -/
theorem num_positive_integers_between_300_and_1200 (n : ℕ) :
  (300 < n^2 ∧ n^2 < 1200) ↔ n ∈ {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34} :=
by {
  sorry
}

/-- There are 17 positive integers n such that 300 < n^2 < 1200. -/
theorem count_positive_integers_between_300_and_1200 :
  fintype.card {n : ℕ // 300 < n^2 ∧ n^2 < 1200} = 17 :=
by {
  sorry
}

end num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l319_319147


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319703

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319703


namespace inscribed_circle_area_ratio_l319_319850

-- Define the setup and main theorem to prove
theorem inscribed_circle_area_ratio {R : ℝ} (hR : R > 0) :
  let s := sqrt 3 * R,
      A := (sqrt 3 / 4) * s^2,
      a := sqrt (3 / 2) * R,
      r := (a * sqrt 3) / 6 in
  let A_incircle := π * r^2,
      A_original := π * R^2 in
  (A_incircle / A_original) = 1 / 8 :=
by
  sorry

end inscribed_circle_area_ratio_l319_319850


namespace true_discount_is_55_l319_319329

-- Definitions based on the given conditions
def BG : ℝ := 6.6
def rate : ℝ := 12
def time : ℝ := 1

-- The problem is to prove that the true discount (TD) is Rs. 55
theorem true_discount_is_55 (TD : ℝ) : BG = (TD * rate * time) / 100 → TD = 55 :=
by
  assume h_BG : BG = (TD * rate * time) / 100,
  -- This is where the proof will go
  sorry

end true_discount_is_55_l319_319329


namespace region_upper_left_of_line_l319_319278

theorem region_upper_left_of_line (x y : ℝ) : (x - 2 * y + 6 < 0) → (x - 2 * y + 6 = 0) → (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1 - 2 * p.2 + 6 < 0) :=
by
  sorry

end region_upper_left_of_line_l319_319278


namespace largest_integer_log_sum_l319_319312

theorem largest_integer_log_sum : 
  let sum_of_logs := (log 3 (3 / 1)) + (log 3 (7 / 3)) + (log 3 (4039 / 4037)) + (log 3 (4041 / 4039))
  in ⌊log 3 (4041)⌋ = 7 := by
  sorry

end largest_integer_log_sum_l319_319312


namespace probability_of_same_class_selection_l319_319348

-- Defining the students and their classes
inductive Student : Type
| A | B | C | D | E

-- Defining a predicate to check if two students are from the same class
def same_class : Student → Student → Prop
| Student.A, Student.B => true
| Student.A, Student.C => true
| Student.B, Student.A => true
| Student.B, Student.C => true
| Student.C, Student.A => true
| Student.C, Student.B => true
| Student.D, Student.E => true
| Student.E, Student.D => true
| _, _ => false

-- Defining the event M
def event_M : Set (Student × Student) :=
  {p | same_class p.1 p.2}

-- Total number of pairs of students
def all_pairs : Finset (Student × Student) :=
  Finset.ofList [(Student.A, Student.B), (Student.A, Student.C), (Student.A, Student.D),
                 (Student.A, Student.E), (Student.B, Student.C), (Student.B, Student.D),
                 (Student.B, Student.E), (Student.C, Student.D), (Student.C, Student.E),
                 (Student.D, Student.E)]

-- Counting the number of favorable outcomes for event M
def favorable_outcomes : Finset (Student × Student) :=
  all_pairs.filter (λ p => event_M p)

-- The probability of event M
def probability_event_M : ℚ :=
  favorable_outcomes.card / all_pairs.card

-- The theorem to prove
theorem probability_of_same_class_selection :
  probability_event_M = 2 / 5 := by
  sorry

end probability_of_same_class_selection_l319_319348


namespace evaluate_expression_l319_319886

theorem evaluate_expression : 
  (√(25 * √(15 * √(45 * √9)))) = 15 * (√15)^(1/4) := sorry

end evaluate_expression_l319_319886


namespace gray_region_area_l319_319872

theorem gray_region_area (d_small : ℝ) (h_small : d_small = 6) 
    (ratio : ℝ) (h_ratio : ratio = 3) 
    (π : ℝ) (h_π : π = Real.pi) : 
    let r_small := d_small / 2
    let r_large := ratio * r_small
    let A_small := π * r_small^2
    let A_large := π * r_large^2
    let A_gray := A_large - A_small
  in A_gray = 72 * π :=
by
  sorry

end gray_region_area_l319_319872


namespace melanies_plums_l319_319611

variable (pickedPlums : ℕ)
variable (gavePlums : ℕ)

theorem melanies_plums (h1 : pickedPlums = 7) (h2 : gavePlums = 3) : (pickedPlums - gavePlums) = 4 :=
by
  sorry

end melanies_plums_l319_319611


namespace prism_lateral_edge_length_l319_319007

-- Define the problem's conditions
constant vertices : ℕ
constant total_lateral_edge_length : ℕ

-- Define the lateral edge length to be proved
constant lateral_edge_length : ℕ

-- Define the assertion that needs to be proved
theorem prism_lateral_edge_length :
  vertices = 12 →
  total_lateral_edge_length = 60 →
  lateral_edge_length = 10 :=
by
  sorry

end prism_lateral_edge_length_l319_319007


namespace books_bought_two_years_ago_l319_319901

/-- 
Five years ago, there were 500 old books in the library.
Two years ago, the librarian bought some books.
Last year, the librarian bought 100 more books than she had bought the previous year.
This year, the librarian donated 200 of the library's old books.
There are now 1000 books in the library.
We need to show that the number of books the librarian bought two years ago is 300.
-/
theorem books_bought_two_years_ago (x : ℕ) :
  500 + x + (x + 100) - 200 = 1000 → x = 300 :=
by
  intros h,
  sorry

end books_bought_two_years_ago_l319_319901


namespace number_form_divisibility_l319_319414

theorem number_form_divisibility (x y z : ℕ) (n : ℕ)
  (h1 : n = 1300000 + 10000 * x + 1000 * y + 450 + z)
  (h2 : n % 8 = 0)
  (h3 : (n.digits 10).sum % 9 = 0)
  (h4 : (n.digits 10).alternating_sum % 11 = 0) :
  n = 1380456 :=
by sorry

end number_form_divisibility_l319_319414


namespace count_integers_between_square_bounds_l319_319145

theorem count_integers_between_square_bounds :
  (n : ℕ) (300 < n^2 ∧ n^2 < 1200) → 17 :=
sorry

end count_integers_between_square_bounds_l319_319145


namespace max_product_of_altitudes_l319_319396

theorem max_product_of_altitudes (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] 
  {h : ℝ} (h_pos : h > 0) (fixed_base : ℝ) :
  (Π (ABC : Type*) [triangle ABC], (ABC.base = fixed_base) ∧ (ABC.altitude_from_C = h) → 
  (ABC.product_of_altitudes ≤ triangle.isosceles.base (ABC, AC = BC))) :=
sorry

end max_product_of_altitudes_l319_319396


namespace cedarwood_earnings_l319_319246

theorem cedarwood_earnings :
  let daily_wage := 1080 / (9 * 4 + 5 * 6 + 6 * 8),
      cedarwood_student_days := 6 * 8,
      cedarwood_earnings := daily_wage * cedarwood_student_days
  in cedarwood_earnings = 454.74 :=
by
  let total_student_days := 9 * 4 + 5 * 6 + 6 * 8
  let daily_wage := 1080 / total_student_days
  let cedarwood_student_days := 6 * 8
  let cedarwood_earnings := daily_wage * cedarwood_student_days
  have h : daily_wage = 1080 / 114 := rfl
  have h2 : 6 * 8 = 48 := rfl
  have h3 : cedarwood_earnings = 9.473684210526315789 * 48 := by rw [h, h2]
  have h4 : cedarwood_earnings ≈ 454.74 := sorry
  exact h4

end cedarwood_earnings_l319_319246


namespace passengers_in_california_l319_319019

theorem passengers_in_california (P : ℕ) (crew : ℕ) (texas_off : ℕ) (texas_on : ℕ) (nc_off : ℕ) (nc_on : ℕ) (landed_virginia : ℕ)
  (h1 : crew = 10)
  (h2 : texas_off = 58)
  (h3 : texas_on = 24)
  (h4 : nc_off = 47)
  (h5 : nc_on = 14)
  (h6 : landed_virginia = 67) :
  P = 124 :=
by
  have total_crew_plus_passengers := landed_virginia - crew
  have step1 := P - texas_off + texas_on
  have step2 := step1 - nc_off + nc_on
  have final_passengers := step2
  have h := final_passengers = total_crew_plus_passengers
  have final := h6 - h1
  have equation := final = P - 67
  have solve := equation = 57
  exact solve
  sorry

end passengers_in_california_l319_319019


namespace triangle_problems_l319_319459

-- Definitions of the conditions in triangle ABC
variables {A B C : ℝ} {a b c : ℝ}
variable h1 : sin B = sqrt 7 / 4
variable h2 : (cos A / sin A) + (cos C / sin C) = (4 * sqrt 7) / 7
variable h3 : (b^2) = a * c
variable h4 : (a^2 + c^2) - 2 * a * c * (cos B) = b^2
variable h5 : (overrightarrow BA) ⋅ (overrightarrow BC) = 3 / 2
noncomputable def proofPart1 : Prop :=
  0 < B ∧ B ≤ π / 3

noncomputable def proofPart2 : Prop :=
  |overrightarrow BC + overrightarrow BA| = 2 * sqrt 2

theorem triangle_problems
  (h1 : sin B = sqrt 7 / 4)
  (h2 : (cos A / sin A) + (cos C / sin C) = (4 * sqrt 7) / 7)
  (h3 : b^2 = a * c)
  (h4 : (a^2 + c^2) - 2 * a * c * cos B = b^2)
  (h5 : (overrightarrow BA) ⋅ (overrightarrow BC) = 3 / 2) :
  proofPart1 ∧ proofPart2 :=
by
  split
  -- proof for each part here
  sorry

end triangle_problems_l319_319459


namespace construct_circle_l319_319953

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2,
    y := (A.y + B.y) / 2 }

axiom non_collinear (A B C : Point) : Prop

def line_through (P Q : Point) : ℝ × ℝ :=
  let m := (Q.y - P.y) / (Q.x - P.x) in
  (m, P.y - m * P.x)

def parallel_lines (m1 m2 : ℝ) : Prop :=
  m1 = m2

def perpendicular_distance (C : Point) (l : ℝ × ℝ) : ℝ := sorry

def projection (C : Point) (l : ℝ × ℝ) : Point := sorry

theorem construct_circle (A B C : Point)
  (h_non_collinear : non_collinear A B C) :
  ∃ r : ℝ, ∃ P Q : Point, ∃ circle_center : Point,
    circle_center = C ∧
    perpendicular_distance C (line_through P C) = r ∧
    perpendicular_distance C (line_through Q C) = r ∧
    parallel_lines (line_through P A).fst (line_through Q B).fst :=
begin
  -- Proof omitted
  sorry
end

end construct_circle_l319_319953


namespace students_not_taking_courses_l319_319171

theorem students_not_taking_courses (total_students french_students german_students both_courses: ℕ) 
    (h1 : total_students = 79)
    (h2 : french_students = 41)
    (h3 : german_students = 22)
    (h4 : both_courses = 9) :
    (total_students - (french_students + german_students - both_courses)) = 25 := 
    by
    rw [h1, h2, h3, h4]
    sorry

end students_not_taking_courses_l319_319171


namespace cosine_of_tangent_line_at_e_l319_319950

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem cosine_of_tangent_line_at_e :
  let θ := Real.arctan 2
  Real.cos θ = Real.sqrt (1 / 5) := by
  sorry

end cosine_of_tangent_line_at_e_l319_319950


namespace cos_19_pi_over_4_l319_319031

theorem cos_19_pi_over_4 :
  cos (19 * Real.pi / 4) = - (Real.sqrt 2 / 2) :=
by sorry

end cos_19_pi_over_4_l319_319031


namespace trigonometric_equation_solution_l319_319214

theorem trigonometric_equation_solution (a b : ℝ) :
  (∃ x : ℝ, (a * sin x + b) / (b * cos x + a) = (a * cos x + b) / (b * sin x + a) ∧
   ((b ≠ sqrt 2 * a ∧ x = k * π + π / 4) ∨
    (b = sqrt 2 * a ∧ x = 2 * k * π + π / 4) ∨
    (b = -sqrt 2 * a ∧ x = (2 * k + 1) * π))) :=
sorry

end trigonometric_equation_solution_l319_319214


namespace smallest_pos_int_mult_4410_sq_l319_319790

noncomputable def smallest_y : ℤ := 10

theorem smallest_pos_int_mult_4410_sq (y : ℕ) (hy : y > 0) :
  (∃ z : ℕ, 4410 * y = z^2) ↔ y = smallest_y :=
sorry

end smallest_pos_int_mult_4410_sq_l319_319790


namespace number_of_integers_in_original_list_l319_319780

theorem number_of_integers_in_original_list :
  ∃ n m : ℕ, (m + 2) * (n + 1) = m * n + 15 ∧
             (m + 1) * (n + 2) = m * n + 16 ∧
             n = 4 :=
by {
  sorry
}

end number_of_integers_in_original_list_l319_319780


namespace standard_equation_of_parabola_l319_319981

theorem standard_equation_of_parabola (focus : ℝ × ℝ): 
  (focus.1 - 2 * focus.2 - 4 = 0) → 
  ((focus = (4, 0) → (∃ a : ℝ, ∀ x y : ℝ, y^2 = 4 * a * x)) ∨
   (focus = (0, -2) → (∃ b : ℝ, ∀ x y : ℝ, x^2 = 4 * b * y))) :=
by
  sorry

end standard_equation_of_parabola_l319_319981


namespace isosceles_triangle_base_angle_l319_319975

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (h1 : vertex_angle = 110) :
  ∃ base_angle : ℝ, base_angle = 35 :=
by
  use 35
  sorry

end isosceles_triangle_base_angle_l319_319975


namespace ratio_of_sides_l319_319835

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l319_319835


namespace sum_of_squares_of_roots_l319_319866

theorem sum_of_squares_of_roots : 
  let a : ℚ := 6
  let b : ℚ := 9
  let c : ℚ := -21
  let x₁ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let x₂ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  (x₁^2 + x₂^2) = 37 / 4 :=
by
  let a : ℚ := 6
  let b : ℚ := 9
  let c : ℚ := -21
  let x₁ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let x₂ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  have h₁ : x₁ + x₂ = -b / a := by sorry
  have h₂ : x₁ * x₂ = c / a := by sorry
  calc
    x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2 * (x₁ * x₂) : by sorry
            ... = (-3/2)^2 - 2 * (-7/2) : by sorry
            ... = 9/4 + 7 : by sorry
            ... = 9/4 + 28/4 : by sorry
            ... = 37/4 : by sorry

end sum_of_squares_of_roots_l319_319866


namespace problem_1_problem_2_l319_319129
open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem problem_1 {x : ℝ} (h : π / 4 ≤ x ∧ x ≤ π / 2)
  : MonotoneOn f (Icc (π / 4) (5 * π / 12)) := sorry

theorem problem_2 {x : ℝ} (h : π / 4 ≤ x ∧ x ≤ π / 2)
  {m : ℝ} (h₁ : ∀ x ∈ Icc (π / 4) (π / 2), |f x - m| < 2)
  : 0 < m ∧ m < 3 := sorry

end problem_1_problem_2_l319_319129


namespace count_solutions_l319_319150

theorem count_solutions :
  ∃ (n1 n2 : ℕ), 0 < n1 ∧ 0 < n2 ∧ n1 ≠ n2 ∧
  (∀ n, (n = n1 ∨ n = n2) ↔ (n > 0 ∧ (n + 500) / 50 = ⌊ real.sqrt n ⌋)) :=
by
  sorry

end count_solutions_l319_319150


namespace lower_darboux_le_upper_l319_319620

open Set

variable {a b : ℝ} (f : ℝ → ℝ) (P1 P2 : Finset ℝ) (s1 S2 : ℝ)

-- Define the conditions in Lean
def lowerDarbouxSum (f : ℝ → ℝ) (P : Finset ℝ) (a b : ℝ) : ℝ := 
  P.fold min (f a) * (b - a)

def upperDarbouxSum (f : ℝ → ℝ) (P : Finset ℝ) (a b : ℝ) : ℝ := 
  P.fold max (f a) * (b - a)

axiom lower_sum (hP1 : P1.IsPartition a b) : s1 = lowerDarbouxSum f P1 a b
axiom upper_sum (hP2 : P2.IsPartition a b) : S2 = upperDarbouxSum f P2 a b

-- Statement in Lean
theorem lower_darboux_le_upper (hP1 : P1.IsPartition a b) (hP2 : P2.IsPartition a b) (h1 : s1 = lowerDarbouxSum f P1 a b) (h2 : S2 = upperDarbouxSum f P2 a b) : 
  s1 ≤ S2 := 
by 
  sorry

end lower_darboux_le_upper_l319_319620


namespace find_x3_l319_319303

noncomputable def pointA := (1 : ℝ, Real.log (1^2))
noncomputable def pointB := (10 : ℝ, Real.log (10^2))
noncomputable def pointC := (x' : ℝ, Real.log 10)

theorem find_x3 (h : 0 < (1 : ℝ) ∧ (1 : ℝ) < (10 : ℝ)) (hx' : x' ≠ 10) : ∃ x3 : ℝ, x3 = Real.sqrt 10 :=
by
  use Real.sqrt 10
  apply sorry

end find_x3_l319_319303


namespace sum_reciprocals_l319_319592

open Real

-- Definitions for sequences
def a : ℕ → ℝ
| 0       := -2
| (n + 1) := a n + b n + sqrt ((a n) ^ 2 + (b n) ^ 2)

def b : ℕ → ℝ
| 0       := 1
| (n + 1) := a n + b n - sqrt ((a n) ^ 2 + (b n) ^ 2)

-- Proof that the sum of reciprocals is 1/2
theorem sum_reciprocals :
  (1 / a 2012) + (1 / b 2012) = 1 / 2 :=
sorry

end sum_reciprocals_l319_319592


namespace find_m_value_l319_319498

noncomputable def perpendicular_vectors (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem find_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (-2, 3),
      b : ℝ × ℝ := (3, m) in
  perpendicular_vectors a b → m = 2 := 
by sorry

end find_m_value_l319_319498


namespace solution_set_l319_319948

def f (x : ℝ) : ℝ := 2^|x - 3| - 1

theorem solution_set : { x : ℝ | f x < 1 } = { x : ℝ | 2 < x ∧ x < 4 } := by
  sorry

end solution_set_l319_319948


namespace max_distance_point_on_curve_l319_319935

theorem max_distance_point_on_curve 
  (x y : ℝ)
  (hC : x^2 / 3 + y^2 = 1)
  (hl : x + sqrt 3 * y - sqrt 3 = 0) :
  x = -sqrt 3 / 2 ∧ y = -sqrt 2 / 2 :=
sorry

end max_distance_point_on_curve_l319_319935


namespace num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l319_319149

/-- The number of positive integers n such that 300 < n^2 < 1200 is 17. -/
theorem num_positive_integers_between_300_and_1200 (n : ℕ) :
  (300 < n^2 ∧ n^2 < 1200) ↔ n ∈ {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34} :=
by {
  sorry
}

/-- There are 17 positive integers n such that 300 < n^2 < 1200. -/
theorem count_positive_integers_between_300_and_1200 :
  fintype.card {n : ℕ // 300 < n^2 ∧ n^2 < 1200} = 17 :=
by {
  sorry
}

end num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l319_319149


namespace num_elements_with_9_as_first_digit_l319_319588

noncomputable def digit_count (n : ℕ) : ℕ :=
⌊ log10 n ⌋ + 1

theorem num_elements_with_9_as_first_digit :
  let T := {k | 0 ≤ k ∧ k ≤ 4000}
  in digit_count (9^4000) = 3817 ∧ (9^4000).digit_span[0] = 9 →
  card {k ∈ T | (9^k).digit_span[0] = 9} = 184 :=
by
  sorry

end num_elements_with_9_as_first_digit_l319_319588


namespace concyclic_points_l319_319590

noncomputable def areConcyclic (A N F P : Point) : Prop := sorry

theorem concyclic_points (A B C M N P D E F : Point)
  (h1 : Triangle A B C)
  (h2 : AcuteScalene A B C)
  (h3 : Midpoint M B C)
  (h4 : Midpoint N C A)
  (h5 : Midpoint P A B)
  (h6 : PerpendicularBisector D A B ∧ OnLine D (Line A M))
  (h7 : PerpendicularBisector E A C ∧ OnLine E (Line A M))
  (h8 : Intersects (Line B D) (Line C E) F)
  (h9 : InsideTriangle F A B C) :
  areConcyclic A N F P :=
sorry

end concyclic_points_l319_319590


namespace loss_percentage_on_first_book_l319_319503

theorem loss_percentage_on_first_book 
    (C1 C2 SP : ℝ) 
    (H1 : C1 = 210) 
    (H2 : C1 + C2 = 360) 
    (H3 : SP = 1.19 * C2) 
    (H4 : SP = 178.5) :
    ((C1 - SP) / C1) * 100 = 15 :=
by
  sorry

end loss_percentage_on_first_book_l319_319503


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319717

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319717


namespace derivative_of_f_l319_319125

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) :=
by
  intro x
  -- We skip the proof here
  sorry

end derivative_of_f_l319_319125


namespace fourth_quadrangle_area_l319_319578

theorem fourth_quadrangle_area (S1 S2 S3 S4 : ℝ) (h : S1 + S4 = S2 + S3) : S4 = S2 + S3 - S1 :=
by
  sorry

end fourth_quadrangle_area_l319_319578


namespace select_people_l319_319628

-- Define the binomial coefficient function (combination)
def binom (n k : ℕ) : ℕ := nat.choose n k

theorem select_people : binom 4 3 * binom 3 1 + binom 4 2 * binom 3 2 + binom 4 1 * binom 3 3 = 34 := by
  sorry

end select_people_l319_319628


namespace fraction_operation_correct_l319_319316

theorem fraction_operation_correct (a b : ℝ) (h : 0.2 * a + 0.5 * b ≠ 0) : 
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
sorry

end fraction_operation_correct_l319_319316


namespace tan_phi_proof_l319_319438

noncomputable def phi : ℝ := sorry

theorem tan_phi_proof (h1 : cos (π / 2 + φ) = sqrt 3 / 2) (h2 : |φ| < π / 2) : tan φ = -sqrt 3 :=
sorry

end tan_phi_proof_l319_319438


namespace region_traced_by_centroid_area_is_79_l319_319606

-- Define the circle with diameter
def is_diameter (A B : Point) (circle : Circle) : Prop := 
  distance A B = 30 ∧ (circle.center = midpoint A B)

-- Define point C on the circle
def on_circle (C : Point) (circle : Circle) : Prop := 
  distance C circle.center = circle.radius

-- Given diameter, circle, and moving point
variables {A B C : Point} {circle : Circle}

theorem region_traced_by_centroid_area_is_79
  (h1 : is_diameter A B circle)
  (h2 : on_circle C circle)
  (h3 : C ≠ A)
  (h4 : C ≠ B) :
  ∃ G : Point, area_of_traced_region G 79 :=
by sorry

end region_traced_by_centroid_area_is_79_l319_319606


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319691

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319691


namespace aaron_position_p2024_l319_319997

-- Definitions based on conditions
def spiral_steps (n : ℕ) : ℕ :=
  1 + (2 * (n - 1))

def total_steps_in_cycle (k : ℕ) : ℕ :=
  4 * k * k

def position_after_cycles (k : ℕ) : (ℤ × ℤ) :=
  (-k, -k)

-- Main statement for the position at p_{2024}
theorem aaron_position_p2024 :
  let steps_22_cycles := total_steps_in_cycle 22
  let remaining_steps := 2024 - steps_22_cycles
  let final_position := (13, 0)
  in remaining_steps = 88 ∧ final_position = (13, 0) := 
sorry

end aaron_position_p2024_l319_319997


namespace locus_of_intersection_of_lines_l319_319218

/-- 
Given an ellipse with foci F1(-c, 0) and F2(c, 0) where a > b > 0 and equation x^2/a^2 + y^2/b^2 = 1,
and given P1P2 is a chord perpendicular to the line segment F1F2,
prove that the locus of the intersection of lines P1F1 and P2F2
is given by the hyperbola x^2/(c^4/a^2) - y^2/(c^2*b^2/a^2) = 1.
-/
theorem locus_of_intersection_of_lines
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0) :
  ∃ (x y : ℝ), (x^2 / (c^4 / a^2) - y^2 / (c^2 * b^2 / a^2) = 1) :=
begin
  sorry
end

end locus_of_intersection_of_lines_l319_319218


namespace symmetric_point_product_l319_319158

theorem symmetric_point_product (x y : ℤ) (h1 : (2008, y) = (-x, -1)) : x * y = -2008 :=
by {
  sorry
}

end symmetric_point_product_l319_319158


namespace greatest_multiple_of_5_and_6_under_1000_l319_319737

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319737


namespace volume_of_prism_in_tetrahedron_l319_319363

def regular_tetrahedron (a : ℝ) := { x : ℝ // 0 < x } -- Edge length is positive

structure triangular_prism (a : ℝ) :=
(edge_length : ℝ)
(on_lateral_edges : Prop) -- Vertices on lateral edges
(in_plane_of_base : Prop) -- Vertices lie in the plane of base

noncomputable def prism_volume {a : ℝ} (t : regular_tetrahedron a) (p : triangular_prism a) : ℝ :=
  if p.on_lateral_edges ∧ p.in_plane_of_base then
    (a^3 * (27 * sqrt 2 - 22 * sqrt 3)) / 2
  else
    0

theorem volume_of_prism_in_tetrahedron (a : ℝ) (t : regular_tetrahedron a) (p : triangular_prism a)
  (h_on_edges : p.on_lateral_edges) (h_in_plane : p.in_plane_of_base) : 
  prism_volume t p = (a^3 * (27 * sqrt 2 - 22 * sqrt 3)) / 2 :=
sorry

end volume_of_prism_in_tetrahedron_l319_319363


namespace isosceles_triangle_base_angles_l319_319972

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end isosceles_triangle_base_angles_l319_319972


namespace max_a_value_l319_319155

noncomputable def f (x : ℝ) : ℝ := cos x - sin x

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Icc (-a) a, deriv f x ≤ 0) → a ≤ π / 4 := 
sorry

end max_a_value_l319_319155


namespace size_of_first_drink_approx_6_l319_319192

variable (x : ℝ)  -- size of the first drink in ounces

def caffeine_in_first_drink (x : ℝ) : ℝ := 250
def caffeine_per_ounce_in_first_drink (x : ℝ) : ℝ := 250 / x
def caffeine_per_ounce_in_second_drink (x : ℝ) : ℝ := 3 * (250 / x)
def caffeine_in_second_drink (x : ℝ) : ℝ := 2 * (3 * (250 / x))
def total_caffeine_from_drinks (x : ℝ) : ℝ :=
  caffeine_in_first_drink x + caffeine_in_second_drink x + caffeine_in_first_drink x + caffeine_in_second_drink x

theorem size_of_first_drink_approx_6 :
  total_caffeine_from_drinks x = 750 → x ≈ 6 := by
sorry

end size_of_first_drink_approx_6_l319_319192


namespace number_of_boys_l319_319342

theorem number_of_boys
  (M W B : Nat)
  (total_earnings wages_of_men earnings_of_men : Nat)
  (num_men_eq_women : 5 * M = W)
  (num_men_eq_boys : 5 * M = B)
  (earnings_eq_90 : total_earnings = 90)
  (men_wages_6 : wages_of_men = 6)
  (men_earnings_eq_30 : earnings_of_men = M * wages_of_men) : 
  B = 5 := 
by
  sorry

end number_of_boys_l319_319342


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319695

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319695


namespace units_digit_7_2083_l319_319775

theorem units_digit_7_2083 : (7^2083 % 10) = 3 := 
by
  -- Calculate the sequence of units digits of powers of 7
  let units_digits := [7, 9, 3, 1]
  -- Find the remainder of 2083 divided by 4
  have remainder := 2083 % 4
  -- Find the units digit corresponding to the remainder in the sequence
  have units_digit := units_digits.nth (rem remainder 4)
  -- Conclude the theorem
  show (7^2083 % 10) = units_digit.mk := sorry

end units_digit_7_2083_l319_319775


namespace find_n_value_l319_319437

theorem find_n_value : (∃ n : ℤ, 3^4 - 13 = 4^3 + n) → (∃ n : ℤ, n = 4) :=
by
  intro h
  cases h with n hn
  use 4
  have h1 : 3^4 = 81 := by norm_num
  have h2 : 4^3 = 64 := by norm_num
  calc
    3^4 - 13 = 81 - 13 := by rw h1
    ... = 68 := by norm_num
    ... = 64 + 4 := by norm_num
    ... = 4^3 + 4 := by rw h2
  contradiction

end find_n_value_l319_319437


namespace region_R_area_correct_l319_319684

noncomputable def area_of_region_R : ℝ :=
let side_length := 1 in
let middle_third_strip := (side_length / 3) * side_length in
let equilateral_triangle_height := (real.sqrt 3 / 2) * side_length in
let equilateral_triangle_area := (1 / 2) * side_length * equilateral_triangle_height in
let smaller_triangle_height := (real.sqrt 3 / 6) in
let smaller_triangle_area := (1 / 2) * (side_length / 3) * smaller_triangle_height in
let rectangle_area := (side_length / 3) * (equilateral_triangle_height - smaller_triangle_height) in
let pentagon_area := smaller_triangle_area + rectangle_area in
middle_third_strip - pentagon_area

theorem region_R_area_correct : area_of_region_R = (3 - real.sqrt 3) / 9 :=
by
  sorry

end region_R_area_correct_l319_319684


namespace determine_range_of_b_l319_319943

noncomputable def f (b x : ℝ) : ℝ := (Real.log x + (x - b) ^ 2) / x
noncomputable def f'' (b x : ℝ) : ℝ := (2 * Real.log x - 2) / x ^ 3

theorem determine_range_of_b (b : ℝ) (h : ∃ x ∈ Set.Icc (1 / 2) 2, f b x > -x * f'' b x) :
  b < 9 / 4 :=
by
  sorry

end determine_range_of_b_l319_319943


namespace home_runs_tied_in_may_l319_319263

theorem home_runs_tied_in_may :
  let aaron_runs := [2, 7, 14, 9, 11, 12, 18] in
  let bonds_runs := [2, 7, 7, 14, 9, 11, 12] in
  (aaron_runs.take 3).sum = (bonds_runs.take 3).sum :=
by {
  let aaron_mar_apr_may := aaron_runs.take 3
  let bonds_mar_apr_may := bonds_runs.take 3
  have h_aaron_may : aaron_mar_apr_may.sum = 2 + 7 + 14 := by sorry,
  have h_bonds_may : bonds_mar_apr_may.sum = 2 + 7 + 14 := by sorry,
  rw [h_aaron_may, h_bonds_may]
  exact rfl,
}

end home_runs_tied_in_may_l319_319263


namespace find_length_dm_l319_319098
-- Import the necessary libraries from Mathlib

-- Define the problem conditions
structure Pyramid :=
(a b c s : Type) (point : Type) (dist : point → point → ℝ)
(regular_tri_pyramid : Prop)
(apex : s) (side_ab : dist a b = 1) (side_as : dist a s = 2)
(median_bm : Prop) (angle_bisector_ad : Prop)

-- Define the theorem to prove
theorem find_length_dm (P : Pyramid) : ∀ (D M : P.point),
    P.regular_tri_pyramid →
    P.apex = s → 
    P.dist P.a P.b = 1 →
    P.dist P.a P.s = 2 →
    P.median_bm →
    P.angle_bisector_ad →
    P.dist D M = Real.sqrt 31 / 6 :=
sorry

end find_length_dm_l319_319098


namespace prime_square_sum_of_cubes_equals_three_l319_319415

open Nat

theorem prime_square_sum_of_cubes_equals_three (p : ℕ) (h_prime : p.Prime) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^3 + b^3) → (p = 3) :=
by
  sorry

end prime_square_sum_of_cubes_equals_three_l319_319415


namespace entrance_ticket_cost_l319_319800

theorem entrance_ticket_cost (num_students num_teachers ticket_cost : ℕ) (h1 : num_students = 20) (h2 : num_teachers = 3) (h3 : ticket_cost = 5) :
  (num_students + num_teachers) * ticket_cost = 115 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end entrance_ticket_cost_l319_319800


namespace min_odd_integers_l319_319680

theorem min_odd_integers (a b c d e f : ℤ)
  (h1 : a + b = 30)
  (h2 : c + d = 16)
  (h3 : e + f = 19) :
  ∃ (n : ℕ), (n = 1 ∧ ∃ (S : Finset ℤ), S.card = n ∧ ∀ x ∈ S, x ∈ {a, b, c, d, e, f} ∧ odd x) :=
sorry

end min_odd_integers_l319_319680


namespace circles_intersect_common_chord_properties_l319_319493

-- Definitions of the circles
def circle1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Prove that the circles intersect
theorem circles_intersect : ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

-- Define the equation of the line containing the common chord and length of the common chord
def common_chord_line : ℝ → ℝ → Prop := λ x y, 4*x + 3*y - 23 = 0
def full_sqrt (x : ℝ) : ℝ := real.sqrt x  -- Common chord length calculation

-- Prove the line equation and the common chord length
theorem common_chord_properties :
  (∀ (x y : ℝ), circle1 x y → circle2 x y → common_chord_line x y) ∧
  (full_sqrt (2 * real.sqrt 7) = 2 * real.sqrt 7) :=
sorry

end circles_intersect_common_chord_properties_l319_319493


namespace unique_real_root_range_l319_319264

theorem unique_real_root_range (a : ℝ) :
  (∃ x : ℝ, (√(a * x^2 + a * x + 2) = a * x + 2) ∧ (a * x + 2 ≥ 0)) ↔ (a = -8 ∨ a ≥ 1) :=
by
  sorry

end unique_real_root_range_l319_319264


namespace fifteen_pow_mn_eq_PnQm_l319_319250

-- Definitions
def P (m : ℕ) := 3^m
def Q (n : ℕ) := 5^n

-- Theorem statement
theorem fifteen_pow_mn_eq_PnQm (m n : ℕ) : 15^(m * n) = (P m)^n * (Q n)^m :=
by
  -- Placeholder for the proof, which isn't required
  sorry

end fifteen_pow_mn_eq_PnQm_l319_319250


namespace proof_of_equation_solution_l319_319240

noncomputable def solve_equation (x : ℝ) : Prop :=
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = Real.cbrt (2 + Real.sqrt x)) ↔
  x = (2389 + 375 * Real.sqrt 17) / 4

-- The statement to be proved
theorem proof_of_equation_solution :
  solve_equation ((2389 + 375 * Real.sqrt 17) / 4) := sorry

end proof_of_equation_solution_l319_319240


namespace sin_2alpha_plus_cos_2alpha_l319_319937

-- Given Conditions
def P : Point := ⟨4, 2⟩
def f (a x : ℝ) : ℝ := log a (x - 3) + 2
def α : ℝ -- α represents the angle whose terminal side passes through point P

-- Question: Proving the desired equality given the conditions
theorem sin_2alpha_plus_cos_2alpha (a : ℝ) (h : f a 4 = 2) (r : ℝ) (sinα : ℝ) (cosα : ℝ) :
  r = 2 * sqrt 5 →
  sinα = sqrt 5 / 5 →
  cosα = 2 * sqrt 5 / 5 →
  sin (2 * α) + cos (2 * α) = 7 / 5 :=
sorry

end sin_2alpha_plus_cos_2alpha_l319_319937


namespace standard_equation_of_circle_l319_319071

def parabola_focus : ℝ × ℝ := (1, 0)
def passing_point : ℝ × ℝ := (5, -2 * Real.sqrt 5)

theorem standard_equation_of_circle :
  ∃ R : ℝ, (x - 1)^2 + y^2 = R^2 ∧ R = 6 :=
by
  have h_focus : parabola_focus = (1, 0) := rfl
  have h_pass : passing_point = (5, -2 * Real.sqrt 5) := rfl
  sorry

end standard_equation_of_circle_l319_319071


namespace count_two_digit_prime_sum10_l319_319502

-- Define a predicate for a number being two-digit
def isTwoDigit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Define a predicate for the sum of digits being 10
def sumOfDigitsIs10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

-- Define a predicate for being a prime number
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Define a predicate for the number satisfying all conditions
def cond (n : ℕ) : Prop := isTwoDigit n ∧ sumOfDigitsIs10 n ∧ isPrime n

-- Theorem stating the number of such numbers is exactly 3
theorem count_two_digit_prime_sum10 : Finset.card (Finset.filter cond (Finset.range 100)) = 3 :=
by
  sorry

end count_two_digit_prime_sum10_l319_319502


namespace num_nonnegative_real_values_l319_319078

theorem num_nonnegative_real_values :
  ∃ n : ℕ, ∀ x : ℝ, (x ≥ 0) → (∃ k : ℕ, (169 - (x^(1/3))) = k^2) → n = 27 := 
sorry

end num_nonnegative_real_values_l319_319078


namespace ratio_accepted_to_rejected_l319_319996

-- Let n be the total number of eggs processed per day
def eggs_per_day := 400

-- Let accepted_per_batch be the number of accepted eggs per batch
def accepted_per_batch := 96

-- Let rejected_per_batch be the number of rejected eggs per batch
def rejected_per_batch := 4

-- On a particular day, 12 additional eggs were accepted
def additional_accepted_eggs := 12

-- Normalize definitions to make our statements clearer
def accepted_batches := eggs_per_day / (accepted_per_batch + rejected_per_batch)
def normally_accepted_eggs := accepted_per_batch * accepted_batches
def normally_rejected_eggs := rejected_per_batch * accepted_batches
def total_accepted_eggs := normally_accepted_eggs + additional_accepted_eggs
def total_rejected_eggs := eggs_per_day - total_accepted_eggs

theorem ratio_accepted_to_rejected :
  (total_accepted_eggs / gcd total_accepted_eggs total_rejected_eggs) = 99 ∧
  (total_rejected_eggs / gcd total_accepted_eggs total_rejected_eggs) = 1 :=
by
  sorry

end ratio_accepted_to_rejected_l319_319996


namespace problem_equivalence_l319_319076

-- Definitions needed from the conditions
def floor (x : ℝ) : ℤ := Int.floor x

-- Lean 4 statement representing the proof problem.
theorem problem_equivalence :
  (∀ (x : ℝ), floor (x + 1) = floor x + 1) ∧
  ¬(∀ (x y : ℝ) (k : ℤ), floor (x + y + k) = floor x + floor y + k) ∧
  ¬(∀ (x y : ℝ), floor (x * y) = floor x * floor y) :=
by
  sorry

end problem_equivalence_l319_319076


namespace min_distance_from_circle_l319_319447

noncomputable def min_value (z : ℂ) := complex.abs (z - 2 - 2 * complex.I)

theorem min_distance_from_circle :
  ∀ z : ℂ, complex.abs (z + 2 - 2 * complex.I) = 1 → min_value z = 3 :=
by
  intro z h
  sorry

end min_distance_from_circle_l319_319447


namespace find_g1_l319_319442

-- Define properties of f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f(x)

-- Define properties of g as an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g(x)

-- Define the given conditions
variables (f g : ℝ → ℝ)
variable h_odd : is_odd_function f
variable h_even : is_even_function g
variable h1 : f (-1) + g 1 = 2
variable h2 : f 1 + g (-1) = 4

-- The proof problem
theorem find_g1 : g 1 = 3 := by
sorry

end find_g1_l319_319442


namespace probability_of_heads_between_0_and_3_l319_319299

noncomputable def coinToss : Type := Finset (Fin 2)

def uniform_coin_toss (n : ℕ) : coinToss := 
  (@Finset.univ (Fin (2^n)) _)

def probability_X_between_0_and_3 (heads_toss : coinToss) : ℝ :=
  let total_outcomes := (heads_toss.card : ℝ)
  let non_zero_non_three := total_outcomes - 2
  non_zero_non_three / total_outcomes
  
theorem probability_of_heads_between_0_and_3 :
  probability_X_between_0_and_3 (uniform_coin_toss 3) = 0.75 :=
by
  sorry

end probability_of_heads_between_0_and_3_l319_319299


namespace true_propositions_l319_319466

-- Definitions for lines and planes
variables {Line Plane : Type} 
variable perp : Line → Plane → Prop
variable parallel : Plane → Plane → Prop
variable subset : Line → Plane → Prop
variable intersection : Plane → Plane → Line

-- Conditions propositions as Lean definitions
def proposition1 (m n : Line) (α β : Plane) : Prop := 
  intersection α β = m ∧ subset n α ∧ perp n m → perp α β

def proposition2 (m n : Line) (α β γ : Plane) : Prop := 
  perp α β ∧ intersection α γ = m ∧ intersection β γ = n → perp n m

def proposition3 (m : Line) (α β : Plane) : Prop := 
  perp m α ∧ perp m β → parallel α β

def proposition4 (m n : Line) (α β : Plane) : Prop := 
  perp m α ∧ perp n β ∧ perp m n → perp α β

-- The theorem statement
theorem true_propositions {Line Plane : Type} {perp : Line → Plane → Prop}
  {parallel : Plane → Plane → Prop} {subset : Line → Plane → Prop} {intersection : Plane → Plane → Line}
  (m n : Line) (α β γ : Plane) :
  {p | p = proposition3 m α β ∨ p = proposition4 m n α β} = {proposition3 m α β, proposition4 m n α β} := 
begin
  sorry 
end

end true_propositions_l319_319466


namespace nth_equation_l319_319613

theorem nth_equation (n : ℕ) (hn: n ≥ 1) : 
  (n+1) / ((n+1)^2 - 1) - 1 / (n * (n+1) * (n+2)) = 1 / (n+1) :=
by
  sorry

end nth_equation_l319_319613


namespace balls_in_original_positions_l319_319630

theorem balls_in_original_positions :
  let n := 7,
      prob_unswapped := (9 / 14) in
  let expected_number := n * prob_unswapped in
  expected_number = 4.5 :=
by
  sorry

end balls_in_original_positions_l319_319630


namespace necessarily_positive_l319_319626

-- Conditions
variables {x y z : ℝ}

-- Statement to prove
theorem necessarily_positive (h1 : 0 < x) (h2 : x < 1) (h3 : -2 < y) (h4 : y < 0) (h5 : 2 < z) (h6 : z < 3) :
  0 < y + 2 * z :=
sorry

end necessarily_positive_l319_319626


namespace PQ_eq_PR_l319_319067

noncomputable def P : ℝ × ℝ := (0, 0)
noncomputable def Q : ℝ × ℝ := (Real.sqrt 2, 0)
noncomputable def R : ℝ × ℝ := (0, Real.sqrt 2)

def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

def angle_PQR : ℝ := 45 * Real.pi / 180  -- in radians

theorem PQ_eq_PR (h1 : dist P R = 9) (h2 : angle_PQR = Real.pi / 4) : dist P Q = 9 :=
  sorry

end PQ_eq_PR_l319_319067


namespace bags_weight_after_removal_l319_319806

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end bags_weight_after_removal_l319_319806


namespace quadratic_nonneg_range_l319_319990

theorem quadratic_nonneg_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end quadratic_nonneg_range_l319_319990


namespace geometric_sequence_properties_l319_319179

theorem geometric_sequence_properties (a : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ (m k : ℕ), a (m + k) = a m * q ^ k) 
  (h_sum : a 1 + a n = 66) 
  (h_prod : a 3 * a (n - 2) = 128) 
  (h_s_n : (a 1 * (1 - q ^ n)) / (1 - q) = 126) : 
  n = 6 ∧ (q = 2 ∨ q = 1/2) :=
sorry

end geometric_sequence_properties_l319_319179


namespace chip_placement_count_l319_319293

/-- 
Given 3 red, 3 blue, and 3 green chips to be placed in a 3x3 grid such that no two chips of the same color are adjacent (vertically or horizontally), 
we prove that the total number of distinct ways to place the chips is 36.
-/
theorem chip_placement_count : 
  let colors := [red, blue, green],
      grid_size := (3, 3),
      adjacent (pos1 pos2) := (abs (pos1.1 - pos2.1) + abs (pos1.2 - pos2.2) = 1)
  in ∃ placements : list (list color), 
    (∀ row, row.length = 3 ∧ list.length placements = 3) ∧
    (∀ i j c, (i, j + 1) < grid_size → adjacent (i, j) (i, j + 1) → (placements.nth i).nth j ≠ (placements.nth i).nth (j + 1) ∧ 
              (i + 1, j) < grid_size → adjacent (i, j) (i + 1, j) → (placements.nth i).nth j ≠ (placements.nth i + 1).nth j) ∧
    list.length (filter (λ p, valid_placement p colors grid_size adjacent)) = 36 := 
sorry

end chip_placement_count_l319_319293


namespace sum_of_arguments_of_eighth_roots_correct_l319_319430

noncomputable def sum_of_arguments_of_eighth_roots (z : ℂ) (h : z^8 = -81 * complex.I) : ℝ :=
  sorry

theorem sum_of_arguments_of_eighth_roots_correct {z : ℂ} (h : z^8 = -81 * complex.I) :
  sum_of_arguments_of_eighth_roots z h = 1530 :=
sorry

end sum_of_arguments_of_eighth_roots_correct_l319_319430


namespace difference_of_sums_l319_319236

open List

theorem difference_of_sums :
  let A := filter (λ n, n % 2 = 0) (range' 22 (70 + 1))
  let B := filter (λ n, n % 2 = 0) (range' 62 (110 + 1))
  let sum_A := A.sum
  let sum_B := B.sum
  in sum_B - sum_A = 1000 :=
by 
  sorry

end difference_of_sums_l319_319236


namespace infinite_family_of_discs_l319_319685

theorem infinite_family_of_discs
  (D : set (set ℝ^2))
  (pairwise_disjoint_interior : ∀ d1 d2 ∈ D, d1 ≠ d2 → interior d1 ∩ interior d2 = ∅)
  (tangent_to_at_least_six : ∀ d ∈ D, ∃ six : set (set ℝ^2), six ⊆ D ∧ six ≠ ∅ ∧ card six ≥ 6 ∧ ∀ d' ∈ six, d ≠ d' ∧ is_tangent d d') :
  infinite D :=
begin
  sorry
end

end infinite_family_of_discs_l319_319685


namespace sandy_initial_money_l319_319374

theorem sandy_initial_money (spent left : ℕ) (h : spent = 6 ∧ left = 57) : 
  let initial_money := left + spent in
  initial_money = 63 :=
by
  let initial_money := left + spent 
  have h1 : initial_money = 63 := by
    rw [h.1, h.2]
    exact rfl
  exact h1


end sandy_initial_money_l319_319374


namespace count_integers_with_three_proper_divisors_l319_319137

open Nat

/-- The number of positive integers with exactly three proper divisors,
    each of which is less than 50, is 109. -/
theorem count_integers_with_three_proper_divisors : 
  {n : ℕ // 
    (∀ d : ℕ, d ∣ n → d < 50) ∧ 
    (∃! d : ℕ, d | n ∧ 1 < d ∧ d < n)} = 109 := 
sorry

end count_integers_with_three_proper_divisors_l319_319137


namespace incorrect_judgment_l319_319090

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- The incorrect judgment in Lean statement
theorem incorrect_judgment : ¬((p ∧ q) ∧ ¬p) :=
by
  sorry

end incorrect_judgment_l319_319090


namespace cardinality_S_l319_319206

open Set

noncomputable def C (s : Set ℝ) : ℕ := s.toFinset.card

def star (A B : Set ℝ) : ℕ :=
if C A ≥ C B then C A - C B else C B - C A

def A : Set ℝ := {1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | (x^2 + a * x) * (x^2 + a * x + 2) = 0}

theorem cardinality_S :
  let S := {a : ℝ | C (B a) = 1 ∨ C (B a) = 3} in
  C S = 3 :=
by {
  -- Proof body is omitted
  sorry
}

end cardinality_S_l319_319206


namespace train_length_l319_319371

theorem train_length (speed_km_per_hr : ℕ) (time_sec : ℕ) (conversion_factor : ℚ) (speed_in_m_per_s : ℚ) (length_of_train : ℚ) :
  speed_km_per_hr = 120 →
  time_sec = 42 →
  conversion_factor = 5 / 18 →
  speed_in_m_per_s = speed_km_per_hr * conversion_factor →
  length_of_train = speed_in_m_per_s * time_sec →
  length_of_train = 1400 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h4, ←h5]
  exact rfl

end train_length_l319_319371


namespace interest_received_l319_319252

theorem interest_received
  (total_investment : ℝ)
  (part_invested_6 : ℝ)
  (rate_6 : ℝ)
  (rate_9 : ℝ) :
  part_invested_6 = 7200 →
  rate_6 = 0.06 →
  rate_9 = 0.09 →
  total_investment = 10000 →
  (total_investment - part_invested_6) * rate_9 + part_invested_6 * rate_6 = 684 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end interest_received_l319_319252


namespace initial_machines_count_l319_319813

variables (N x : ℕ)

-- Given: N machines can produce x units in 5 days
def rate1 := x / 5

-- Given: 12 machines can produce 4x units in 10 days
def rate2 := 4 * x / 30

-- Prove: The number of machines N working initially is 2
theorem initial_machines_count (h : N * rate1 = 12 * rate2) : N = 2 :=
by
  -- Assuming the provided condition
  have h1 : N * (x / 5) = 12 * (4 * x / 30) := h
  -- Simplifying the equation
  sorry

end initial_machines_count_l319_319813


namespace problem1_problem2_problem3_l319_319888

-- Problem 1
theorem problem1 (x y : ℝ) : 4 * x^2 - y^4 = (2 * x + y^2) * (2 * x - y^2) :=
by
  -- proof omitted
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : 8 * x^2 - 24 * x * y + 18 * y^2 = 2 * (2 * x - 3 * y)^2 :=
by
  -- proof omitted
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) : (x - y) * (3 * x + 1) - 2 * (x^2 - y^2) - (y - x)^2 = (x - y) * (1 - y) :=
by
  -- proof omitted
  sorry

end problem1_problem2_problem3_l319_319888


namespace length_of_hypotenuse_l319_319560

theorem length_of_hypotenuse (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 2450) (h2 : c = b + 10) (h3 : a^2 + b^2 = c^2) : c = 35 :=
by
  sorry

end length_of_hypotenuse_l319_319560


namespace cost_price_of_toy_l319_319825

theorem cost_price_of_toy 
  (cost_price : ℝ)
  (SP : ℝ := 120000)
  (num_toys : ℕ := 40)
  (profit_per_toy : ℝ := 500)
  (gain_per_toy : ℝ := cost_price + profit_per_toy)
  (total_gain : ℝ := 8 * cost_price + profit_per_toy * num_toys)
  (total_cost_price : ℝ := num_toys * cost_price)
  (SP_eq_cost_plus_gain : SP = total_cost_price + total_gain) :
  cost_price = 2083.33 :=
by
  sorry

end cost_price_of_toy_l319_319825


namespace abs_diff_roots_eq_sqrt_13_l319_319473

theorem abs_diff_roots_eq_sqrt_13 {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  |x₁ - x₂| = Real.sqrt 13 :=
sorry

end abs_diff_roots_eq_sqrt_13_l319_319473


namespace perpendicular_planes_normal_vectors_l319_319542

-- Define the normal vectors pairs
def n1_1 : ℝ × ℝ × ℝ := (1, 2, 1)
def n1_2 : ℝ × ℝ × ℝ := (-3, 1, 1)

def n2_1 : ℝ × ℝ × ℝ := (1, 1, 2)
def n2_2 : ℝ × ℝ × ℝ := (-2, 1, 1)

def n3_1 : ℝ × ℝ × ℝ := (1, 1, 1)
def n3_2 : ℝ × ℝ × ℝ := (-1, 2, 1)

def n4_1 : ℝ × ℝ × ℝ := (1, 2, 1)
def n4_2 : ℝ × ℝ × ℝ := (0, -2, -2)

-- Dot product function
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Prove that the dot product of the correct pair is zero
theorem perpendicular_planes_normal_vectors :
  dot_product n1_1 n1_2 = 0 :=
by {
  -- calculation of dot product for the pair (1,2,1) and (-3,1,1)
  sorry
}

end perpendicular_planes_normal_vectors_l319_319542


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319751

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319751


namespace area_ratio_l319_319361

-- Define the conditions
variables {A B C D E F : Type*}
variables {alpha : set Type*}
variables (CA' : ℝ) (A'A : ℝ) (CB' : ℝ) (B'B : ℝ) (CD' : ℝ) (D'D : ℝ)

-- Conditions given in the problem
def condition1 (h1 : CA' / A'A = 4 / 1) : Prop := CA' = 4 * A'A
def condition2 (h2 : CB' / B'B = 4 / 1) : Prop := CB' = 4 * B'B
def condition3 (h3 : CD' / D'D = 4 / 1) : Prop := CD' = 4 * D'D

-- The result we need to prove
theorem area_ratio (h1 : condition1 CA' A'A) (h2 : condition2 CB' B'B) (h3 : condition3 CD' D'D) :
  ∃ (r : ℝ), r = 1 / 225 ∨ r = 256 / 225 :=
sorry

end area_ratio_l319_319361


namespace count_odd_three_digit_numbers_less_than_600_l319_319683

theorem count_odd_three_digit_numbers_less_than_600 : 
  ∃ n : ℕ, n = 75 ∧ 
  (∀ (hundreds tens units : ℕ),
   hundreds ∈ {1, 2, 3, 4, 5} ∧ 
   tens ∈ {1, 2, 3, 4, 5} ∧ 
   units ∈ {1, 3, 5} →
   let num := 100 * hundreds + 10 * tens + units in 
   num < 600 ∧ num % 2 = 1) :=
by
  use 75
  sorry

end count_odd_three_digit_numbers_less_than_600_l319_319683


namespace sum_of_A_and_B_l319_319345

theorem sum_of_A_and_B:
  ∃ A B : ℕ, (A = 2 + 4) ∧ (B - 3 = 1) ∧ (A < 10) ∧ (B < 10) ∧ (A + B = 10) :=
by 
  sorry

end sum_of_A_and_B_l319_319345


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319716

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319716


namespace greatest_multiple_of_5_and_6_under_1000_l319_319730

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319730


namespace isosceles_triangle_perimeter_l319_319658

theorem isosceles_triangle_perimeter (x y P : ℝ) (h₁ : x = 1 ∨ x = 2) (h₂ : y = 1 ∨ y = 2)
  (h₃ : x ≠ y) (h₄ : P = x + x + y ∨ P = x + y + y) : P = 5 :=
by
  cases h₁ with
  | inl x_eq_1 =>
    cases h₂ with
    | inl y_eq_1 =>
      rw [x_eq_1, y_eq_1] at h₄
      contradictions
    | inr y_eq_2 =>
      rw [x_eq_1, y_eq_2] at h₄
      cases h₄ with
      | inl double_base => contradiction
      | inr _ => exact rfl
  | inr x_eq_2 =>
    cases h₂ with
    | inl y_eq_1 =>
      rw [x_eq_2, y_eq_1] at h₄
      cases h₄ with
      | inl single_base => exact rfl
      | inr double_leg => contradiction
    | inr y_eq_2 =>
      rw [x_eq_2, y_eq_2] at h₄
      contradictions

end isosceles_triangle_perimeter_l319_319658


namespace find_MN_l319_319577

-- Define the geometry setup
variable {Point : Type}
variable [MetricSpace Point]

-- Points of the triangle
variables (A B C L K M N : Point)

-- Define distances and perpendiculars
noncomputable def distance_AB : ℝ := 130
noncomputable def distance_AC : ℝ := 123
noncomputable def distance_BC : ℝ := 110

-- Given Conditions
axiom cond1 : dist A B = distance_AB
axiom cond2 : dist A C = distance_AC
axiom cond3 : dist B C = distance_BC
axiom angle_bisector_A : angle_bisector A B C L
axiom angle_bisector_B : angle_bisector B A C K
axiom perpendicular_from_C_to_BK : is_foot_perpendicular C K B M
axiom perpendicular_from_C_to_AL : is_foot_perpendicular C L A N

-- Prove that MN = 51.5
theorem find_MN : dist M N = 51.5 := sorry

end find_MN_l319_319577


namespace graph_not_in_third_quadrant_l319_319081

-- Definition of the function y = 1 / x^2
def y (x : ℝ) : ℝ := 1 / (x ^ 2)

-- Problem statement to prove that the graph does not pass through the third quadrant
theorem graph_not_in_third_quadrant : ∀ (x : ℝ), (x < 0 ∧ y x < 0) → False := 
by
  intro x hx
  sorry

end graph_not_in_third_quadrant_l319_319081


namespace ellipse_point_angles_l319_319922

noncomputable theory

variables (A B C O : EuclideanSpace ℝ (Fin 2))

theorem ellipse_point_angles
  (h1 : (0:ℝ)^2 + 3^2 = (2:ℝ)^2)
  (h2 : ∀ t : ℝ, A = ⟨-t, √3 * t⟩)
  (h3 : ∀ p : EuclideanSpace ℝ (Fin 2), p ∈ {A, B, C} → (p.1 ^ 2) / 4 + p.2 ^ 2 = 1)
  (h4 : ∠ A O B = 2 * pi / 3)
  (h5 : ∠ B O C = 2 * pi / 3)
  (h6 : ∠ C O A = 2 * pi / 3) :
  (1 / (norm A) ^ 2) + (1 / (norm B) ^ 2) + (1 / (norm C) ^ 2) = 15 / 8 :=
sorry

end ellipse_point_angles_l319_319922


namespace entrance_ticket_cost_l319_319802

theorem entrance_ticket_cost (num_students num_teachers ticket_cost : ℕ) (h1 : num_students = 20) (h2 : num_teachers = 3) (h3 : ticket_cost = 5) :
  (num_students + num_teachers) * ticket_cost = 115 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end entrance_ticket_cost_l319_319802


namespace trigonometric_identity_l319_319399

theorem trigonometric_identity : 
  ∀ (α : ℝ), -sin α + sqrt 3 * cos α = 2 * sin (α + 2 * π / 3) :=
by
  intro α
  sorry

end trigonometric_identity_l319_319399


namespace water_depth_upright_l319_319365

def tank_is_right_cylindrical := true
def tank_height := 18.0
def tank_diameter := 6.0
def tank_initial_position_is_flat := true
def water_depth_flat := 4.0

theorem water_depth_upright : water_depth_flat = 4.0 :=
by
  sorry

end water_depth_upright_l319_319365


namespace find_p_and_a_n_find_T_n_l319_319455

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a(n + 1) = a(n) + d

noncomputable def Sn (p : ℕ) (n : ℕ) : ℕ := p * n ^ 2 + n

theorem find_p_and_a_n (d : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ):
  d = 2 →
  (∀ n : ℕ, S n = n * a(1) + (n * (n - 1) / 2) * d) →
  (∃ p : ℕ, ∀ n : ℕ, S n = p * n ^ 2 + n) →
  (p = 1 ∧ ∀ n : ℕ, a(n) = 2 * n) :=
sorry

theorem find_T_n (d : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) :
  d = 2 →
  (∀ n : ℕ, a(n) = 2 * n) →
  (∀ n : ℕ, b(n) = (-1) ^ n * a(n)) →
  (∀ n : ℕ, T n = ∑ i in range(n), b(i)) →
  (∀ n : ℕ, T n = if n % 2 = 0 then n else -n - 1) :=
sorry

end find_p_and_a_n_find_T_n_l319_319455


namespace zoo_total_revenue_l319_319615

theorem zoo_total_revenue :
  let monday_children := 7 * 3
  let monday_adults := 5 * 4
  let monday_seniors := 3 * 3
  let monday_total := monday_children + monday_adults + monday_seniors
  let tuesday_children := 9 * 4
  let tuesday_adults := 6 * 5
  let tuesday_seniors := 2 * 3
  let tuesday_total_before_discount := tuesday_children + tuesday_adults + tuesday_seniors
  let discount := tuesday_total_before_discount * 0.10
  let tuesday_total := tuesday_total_before_discount - discount
  monday_total + tuesday_total = 114.8 :=
by
  -- Definitions for Monday
  let monday_children := 7 * 3
  let monday_adults := 5 * 4
  let monday_seniors := 3 * 3
  let monday_total := monday_children + monday_adults + monday_seniors

  -- Definitions for Tuesday
  let tuesday_children := 9 * 4
  let tuesday_adults := 6 * 5
  let tuesday_seniors := 2 * 3
  let tuesday_total_before_discount := tuesday_children + tuesday_adults + tuesday_seniors
  let discount := tuesday_total_before_discount * 0.10
  let tuesday_total := tuesday_total_before_discount - discount

  -- The total revenue from both days
  have monday_total = 50 := by norm_num
  have tuesday_total = 64.8 := by norm_num
  have total_revenue := monday_total + tuesday_total
  have total_revenue = 114.8 := by norm_num

  exact total_revenue

end zoo_total_revenue_l319_319615


namespace net_effect_on_revenue_l319_319332

theorem net_effect_on_revenue (P S : ℝ) :
  let original_revenue := P * S
  let reduced_price := 0.6 * P
  let increased_sales := 1.8 * S
  let new_revenue := reduced_price * increased_sales
  net_effect := new_revenue - original_revenue
  net_effect = 0.08 * original_revenue :=
begin
  sorry
end

end net_effect_on_revenue_l319_319332


namespace position_at_2017_l319_319003

def a : ℕ → ℚ
def b : ℕ → ℚ

axiom a_0 : a 0 = 0
axiom b_0 : b 0 = 1
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 2 * b n
axiom b_recurrence : ∀ n : ℕ, b (n + 1) = a n + b n

theorem position_at_2017 : 
  a 2017 = (2/3) * 2^2016 - (2/3) 
  ∧ b 2017 = (1/3) * 2^2018 - (1/3) := 
sorry

end position_at_2017_l319_319003


namespace number_of_integers_l319_319141

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end number_of_integers_l319_319141


namespace find_other_number_l319_319330

open Nat

theorem find_other_number (A B lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 30) (h_A : A = 231) (h_eq : lcm * hcf = A * B) : 
  B = 300 :=
  sorry

end find_other_number_l319_319330


namespace pentagon_square_ratio_l319_319833

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l319_319833


namespace successive_increases_equiv_single_increase_l319_319326

theorem successive_increases_equiv_single_increase :
  ∀ (P : ℝ), (P * 1.06) * 1.06 = P * 1.1236 :=
by
  intro P
  calc     
    (P * 1.06) * 1.06 = P * (1.06 * 1.06) : by sorry
    ... = P * 1.1236 : by sorry 

end successive_increases_equiv_single_increase_l319_319326


namespace find_solutions_l319_319417

noncomputable def equation (x : ℝ) : ℝ :=
  (1 / (x^2 + 11*x - 8)) + (1 / (x^2 + 2*x - 8)) + (1 / (x^2 - 13*x - 8))

theorem find_solutions : 
  {x : ℝ | equation x = 0} = {1, -8, 8, -1} := by
  sorry

end find_solutions_l319_319417


namespace babysitting_earnings_l319_319034

theorem babysitting_earnings
  (cost_video_game : ℕ)
  (cost_candy : ℕ)
  (hours_worked : ℕ)
  (amount_left : ℕ)
  (total_earned : ℕ)
  (earnings_per_hour : ℕ) :
  cost_video_game = 60 →
  cost_candy = 5 →
  hours_worked = 9 →
  amount_left = 7 →
  total_earned = cost_video_game + cost_candy + amount_left →
  earnings_per_hour = total_earned / hours_worked →
  earnings_per_hour = 8 :=
by
  intros h_game h_candy h_hours h_left h_total_earned h_earn_per_hour
  rw [h_game, h_candy] at h_total_earned
  simp at h_total_earned
  have h_total_earned : total_earned = 72 := by linarith
  rw [h_total_earned, h_hours] at h_earn_per_hour
  simp at h_earn_per_hour
  assumption

end babysitting_earnings_l319_319034


namespace edward_initial_amount_l319_319882

theorem edward_initial_amount
  (spent1 spent2 remaining : ℕ)
  (h_spent1 : spent1 = 9)
  (h_spent2 : spent2 = 8)
  (h_remaining : remaining = 17) :
  let InitialAmount := spent1 + spent2 + remaining in
  InitialAmount = 34 :=
by
  sorry

end edward_initial_amount_l319_319882


namespace exponent_of_9_in_9_pow_7_l319_319689

theorem exponent_of_9_in_9_pow_7 : ∀ x : ℕ, (3 ^ x ∣ 9 ^ 7) ↔ x ≤ 14 := by
  sorry

end exponent_of_9_in_9_pow_7_l319_319689


namespace possible_values_ceil_square_l319_319507

noncomputable def num_possible_values (x : ℝ) (hx : ⌈x⌉ = 12) : ℕ := 23

theorem possible_values_ceil_square (x : ℝ) (hx : ⌈x⌉ = 12) :
  let n := num_possible_values x hx in n = 23 :=
by
  let n := num_possible_values x hx
  exact rfl

end possible_values_ceil_square_l319_319507


namespace greatest_multiple_l319_319726

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319726


namespace integral_result_l319_319884

noncomputable def integral_problem : ℝ :=
  ∫ x in (1:ℝ)..3, 2 * x - 1 / x^2

theorem integral_result : integral_problem = 8 := by
  sorry

end integral_result_l319_319884


namespace quadratic_minimization_l319_319074

theorem quadratic_minimization : 
  ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12 * x + 36 ≤ y^2 - 12 * y + 36) ∧ x^2 - 12 * x + 36 = 0 :=
by
  sorry

end quadratic_minimization_l319_319074


namespace envelope_area_l319_319022

-- Define the variables and conditions
def bottom_width : ℝ := 4
def top_width : ℝ := 6
def height : ℝ := 5

-- Define the formula for the area of a trapezoid
def trapezoid_area (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- The proof statement
theorem envelope_area :
  trapezoid_area bottom_width top_width height = 25 :=
by
  unfold trapezoid_area
  norm_num
  sorry

end envelope_area_l319_319022


namespace tire_wear_total_distance_l319_319811

-- Define the wear rates based on conditions
def wear_rate_front (k : ℝ) := k / 5000
def wear_rate_rear (k : ℝ) := k / 3000

-- Let k be a positive real number
variables (k x y : ℝ)
variables (hk : k > 0) (hx : x > 0) (hy : y > 0)

-- Conditions based on the total wear amount equations
def condition1 := (wear_rate_front k) * x + (wear_rate_rear k) * y = k
def condition2 := (wear_rate_rear k) * x + (wear_rate_front k) * y = k

-- Concluding the total distance travelled
theorem tire_wear_total_distance : x + y = 3750 :=
by
  let total_distance := x + y
  have h1 : total_distance / 5000 + total_distance / 3000 = 2 := sorry
  have h2 : total_distance = 2 / (1 / 5000 + 1 / 3000) := sorry
  exact h2.symm

end tire_wear_total_distance_l319_319811


namespace urn_gold_coins_percentage_l319_319381

theorem urn_gold_coins_percentage (obj_perc_beads : ℝ) (coins_perc_gold : ℝ) : 
    obj_perc_beads = 0.15 → coins_perc_gold = 0.65 → 
    (1 - obj_perc_beads) * coins_perc_gold = 0.5525 := 
by
  intros h_obj_perc_beads h_coins_perc_gold
  sorry

end urn_gold_coins_percentage_l319_319381


namespace cos_x_plus_5_sin_x_values_l319_319464

theorem cos_x_plus_5_sin_x_values :
  (∃ x : ℝ, sin x - 5 * cos x = 2 ∧ (cos x + 5 * sin x = sqrt 46 ∨ cos x + 5 * sin x = -sqrt 46)) :=
sorry

end cos_x_plus_5_sin_x_values_l319_319464


namespace minimum_value_l319_319925

theorem minimum_value (a b : ℝ) (h : a^2 * b^2 + 2 * a * b + 2 * a + 1 = 0) :
    ∃ c : ℝ, c = -3 / 4 ∧ (ab * (ab + 2) + (b + 1)^2 + 2 * a) ≥ c :=
begin
  sorry
end

end minimum_value_l319_319925


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319744

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319744


namespace concentric_circles_ratio_l319_319554

theorem concentric_circles_ratio
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : π * b^2 - π * a^2 = 4 * (π * a^2)) :
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end concentric_circles_ratio_l319_319554


namespace inverse_function_solution_l319_319533

variables {α β : Type*} [linear_order α] [linear_order β] {f : α → β} {a b x : α} {x0 : β}

theorem inverse_function_solution
  (h1 : ∀ x y ∈ set.Icc a b, x < y → f y < f x)
  (h2 : x0 ∈ set.Icc a b)
  (h3 : f x0 = 0)
  (h4 : ∀ y, f (f⁻¹ y) = y ∧ f⁻¹ (f y) = y) :
  f⁻¹ 0 ∈ set.Icc a b :=
sorry

end inverse_function_solution_l319_319533


namespace solution_set_inequality_l319_319287

theorem solution_set_inequality (x : ℝ) (h : x ≠ 1) : (frac (x - 2) (x - 1) ≥ 2) ↔ (0 ≤ x ∧ x < 1) :=
sorry

end solution_set_inequality_l319_319287


namespace fixed_point_through_line_shortest_chord_length_l319_319092

noncomputable def line (m : ℝ) : ℝ × ℝ → Prop :=
  λ (x y : ℝ), (2 * m + 1) * x + (m + 2) * y - 3 * m - 6 = 0

noncomputable def circle : ℝ × ℝ → Prop :=
  λ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 6

theorem fixed_point_through_line :
  ∀ (m : ℝ), line m (0, 3) := by
  sorry

theorem shortest_chord_length (m : ℝ) :
  m = 1 → (∀ (x1 y1 x2 y2 : ℝ), line m (x1, y1) ∧ circle (x1, y1) ∧ line m (x2, y2) ∧ circle (x2, y2)) → (x1 - x2)^2 + (y1 - y2)^2 = 16 := by
  sorry

end fixed_point_through_line_shortest_chord_length_l319_319092


namespace percent_growth_1991_to_2016_l319_319548

noncomputable def percent_growth_problem (p q r : ℕ) (P1991 P2006 P2016 : ℕ) : Prop :=
  let initial_population := P1991
  let final_population := P2016
  P1991 = p^2 ∧
  P2006 = p^2 + 120 ∧
  P2006 = q^2 - 1 ∧
  P2016 = p^2 + 120 + 180 ∧
  P2016 = r^2 ∧
  r^2 = final_population ∧
  100 * (final_population - initial_population) / initial_population = 5

theorem percent_growth_1991_to_2016 (p q r : ℕ) (P1991 P2006 P2016 : ℕ)
  (h1 : P1991 = p^2)
  (h2 : P2006 = p^2 + 120)
  (h3 : P2006 = q^2 - 1)
  (h4 : P2016 = p^2 + 120 + 180)
  (h5 : P2016 = r^2) :
  percent_growth_problem p q r P1991 P2006 P2016 :=
begin
  sorry
end

end percent_growth_1991_to_2016_l319_319548


namespace compare_M_N_sufficient_condition_range_l319_319337

-- Problem 1 Lean Statement
theorem compare_M_N (x : ℝ) : (2 * x^2 + 1) > (x^2 + 2 * x - 1) := by
  sorry

-- Problem 2 Lean Statement
theorem sufficient_condition_range 
    (m : ℝ) :
    (∀ x : ℝ, (2 * m ≤ x ∧ x ≤ m + 1) → (-1 ≤ x ∧ x ≤ 1)) ↔ (m ∈ set.Icc (-1/2 : ℝ) 0) := by
  sorry

end compare_M_N_sufficient_condition_range_l319_319337


namespace negation_of_exists_proposition_l319_319656

theorem negation_of_exists_proposition :
  ¬ (∃ x₀ : ℝ, x₀^2 - 1 < 0) ↔ ∀ x : ℝ, x^2 - 1 ≥ 0 :=
by
  sorry

end negation_of_exists_proposition_l319_319656


namespace volume_of_triangular_pyramid_l319_319039

noncomputable def pyramid_volume (s : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * s^2 in
  let height := sqrt (s^2 - (s / 2)^2) in
  (1 / 3) * base_area * height

theorem volume_of_triangular_pyramid :
  pyramid_volume (6 * sqrt 3) = 81 * sqrt 3 :=
by
  sorry

end volume_of_triangular_pyramid_l319_319039


namespace awards_distribution_correct_answer_awards_distribution_l319_319237

theorem awards_distribution :
  let awards := 6
  let students := 4
  ∃ f : Fin awards → Fin students, 
    (∀ s : Fin students, ∃ a : Fin awards, f a = s) ∧ 
    (∑ s, 1) = 6 :=
begin
  -- Here we define the number of ways to distribute the awards
  -- We assert the number of ways equals 1560
  sorry
end

theorem correct_answer_awards_distribution :
  let awards := 6
  let students := 4
  count_distributions(awards, students) = 1560 :=
begin
  sorry
end

end awards_distribution_correct_answer_awards_distribution_l319_319237


namespace bisects_angle_DAB_l319_319040

-- Define the points and the geometrical properties
variables {A B C D E F G : Type}
-- Assume A, B, C, D, E are points that form the given parallelogram and cyclic quadrilateral
-- Assume F and G are the points of intersection as described
-- Assume EF = EG = EC

axiom is_parallelogram (A B C D : Type) : Prop
axiom is_cyclic_quadrilateral (B C E D : Type) : Prop
axiom passes_through (ℓ : A) (P : Type) : Prop
axiom intersects_at (ℓ : Type) (P Q : Type) (R : Type) : Prop
axiom equal_lengths {U V W : Type} (EU EV EW : Type) : (EU = EV) ∧ (EV = EW)

variables (ℓ : Type)
variables (ABC_parallelogram : is_parallelogram A B C D)
variables (BCED_cyclic : is_cyclic_quadrilateral B C E D)
variables (ℓ_passes_A : passes_through ℓ A)
variables (ℓ_intersects_DC : intersects_at ℓ D C F)
variables (ℓ_intersects_BC : intersects_at ℓ B C G)
variables (lengths_equal : equal_lengths EF EG EC)

theorem bisects_angle_DAB : Prop :=
  -- The line ℓ bisects the angle DAB
  sorry

end bisects_angle_DAB_l319_319040


namespace find_number_of_people_l319_319257

-- Assuming the following conditions:
-- The average weight of a group increases by 2.5 kg when a new person replaces one weighing 65 kg
-- The weight of the new person is 85 kg

def avg_weight_increase (N : ℕ) (old_weight new_weight : ℚ) : Prop :=
  let weight_diff := new_weight - old_weight
  let total_increase := 2.5 * N
  weight_diff = total_increase

theorem find_number_of_people :
  avg_weight_increase N 65 85 → N = 8 :=
by
  intros h
  sorry -- complete proof is not required

end find_number_of_people_l319_319257


namespace max_k_l319_319425

open Complex

theorem max_k : ∃ k, k = 1010 ∧ ∀ (x : ℤ), 0 ≤ x ∧ x ≤ 2019 → |exp (2 * Real.pi * i * (x / 2019 : ℂ)) - 1| ≤ |exp (2 * Real.pi * i * (1010 / 2019 : ℂ)) - 1| := by
  sorry

end max_k_l319_319425


namespace leaves_per_frond_l319_319194

-- Definitions
def ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def total_leaves : ℕ := 1260

-- Statement to be proven
theorem leaves_per_frond : (total_leaves / (ferns * fronds_per_fern)) = 30 := 
by
  -- Here we provide the proof steps
  let total_fronds := ferns * fronds_per_fern  
  have h : total_fronds > 0 := by 
  -- Ensure total fronds is greater than zero to avoid division by zero
  sorry
  have total_leaves_correct : total_leaves = total_fronds * 30 := by
  -- Provide the validation of total leaves given the problem conditions
  sorry
  rw total_leaves_correct
  have division_non_zero : total_fronds * 30 / total_fronds = 30 := by 
  -- Simplify the division
  sorry
  rw division_non_zero
  rfl

-- Proving the conditions in Lean
lemma total_fronds_positive : ferns * fronds_per_fern > 0 := by
  -- Prove that the total number of fronds is positive
  sorry

end leaves_per_frond_l319_319194


namespace tangent_line_at_given_point_l319_319422

noncomputable def tangent_line_equation (x : ℝ) : ℝ := (3 / 2) * x - (19 / 4)

theorem tangent_line_at_given_point : 
  ∀ (x : ℝ), (x = 4) → 
  differentiable ℝ (λ x, (x^2 - 2*x - 3) / 4) ∧ 
  tangent_line_equation x = 
    ((1 / 2) * (x^2 - 2*x - 3)) / 4 := 
  sorry

end tangent_line_at_given_point_l319_319422


namespace maria_travel_fraction_l319_319877

variable (D : ℝ) (x : ℝ)

def condition1 := D = 280
def distance_first_stop := 280 * x
def remaining_distance_first_stop := 280 - 280 * x
def distance_second_stop := (1 / 4) * (280 - 280 * x)
def distance_after_second_stop := 105
def equation := distance_first_stop + distance_second_stop + distance_after_second_stop = 280

theorem maria_travel_fraction
  (h1 : condition1)
  (h2 : equation) :
  x = 1 / 2 := sorry

end maria_travel_fraction_l319_319877


namespace problem_ellipse_l319_319920

noncomputable def semiFocalDistance (b : ℝ) : ℝ := sqrt 2 * b

structure Ellipse where
  a b : ℝ
  h k : ℝ              -- Center of the ellipse
  (a_pos : a > b)
  (b_pos : b > 0)
  (passes_through : ℝ × ℝ)
  (semifocal_distance : ℝ)

def is_on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  ((x - e.h)^2 / e.a^2) + ((y - e.k)^2 / e.b^2) = 1

theorem problem_ellipse {a b : ℝ} (hb : b > 0) (ha : a > b)
  (h_ellipse : is_on_ellipse {a := a, b := b, h := 0, k := 0, a_pos := ha, b_pos := hb, passes_through := (-1, -1), semifocal_distance := semiFocalDistance b} (-1) (-1))
  (h_c : semiFocalDistance b = sqrt 2 * b) :
  (is_on_ellipse {a := 2, b := 2 / sqrt 3, h := 0, k := 0, a_pos := by sorry, b_pos := by sorry, passes_through := (-1, -1), semifocal_distance := sqrt 2 * (2 / sqrt 3)} (-1) (-1) ∧
  (TODO - Define and prove area of triangle ∆ PMN given slope l1 == -1 and midpoint condition)
  (TODO - Define and prove line MN equation)) := sorry

end problem_ellipse_l319_319920


namespace find_g_neg1_l319_319108

variable {α : Type*}

-- Given conditions
def is_odd_function (f : α → α) : Prop := ∀ x, f (-x) = -f x
def y (f : α → α) (x : α) : α := f x + x^2

variable (f : ℝ → ℝ) (g : ℝ → ℝ)
variable (h1 : is_odd_function (λ x, f x + x^2))
variable (h2 : f 1 = 1)
variable (h3 : ∀ x, g x = f x + 2)

-- Proof statement
theorem find_g_neg1 : g (-1) = 1 :=
sorry

end find_g_neg1_l319_319108


namespace solve_tan_equation_l319_319321

theorem solve_tan_equation (x : ℝ) (k : ℤ) :
  8.456 * (Real.tan x)^2 * (Real.tan (3 * x))^2 * Real.tan (4 * x) = 
  (Real.tan x)^2 - (Real.tan (3 * x))^2 + Real.tan (4 * x) ->
  x = π * k ∨ x = π / 4 * (2 * k + 1) := sorry

end solve_tan_equation_l319_319321


namespace tan_alpha_in_second_quadrant_l319_319927

variable (α : ℝ)
variable (cos_alpha : ℝ)
variable (sin_alpha : ℝ)

-- Conditions
variable (h_α_quadrant_2 : cos_alpha = -3 / 5 ∧ (2 * π / 2 < α ∧ α < π))
variable (h_cosα : cos_alpha = -3 / 5)

-- Question: Prove that tan(α) = -4 / 3
theorem tan_alpha_in_second_quadrant : tan α = -4 / 3 :=
by
  sorry

end tan_alpha_in_second_quadrant_l319_319927


namespace range_of_g_l319_319583

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_g :
  (set.range g) = set.Icc (π^4 / 32) (2 * π^4 / 5) :=
sorry

end range_of_g_l319_319583


namespace rtl_to_conventional_notation_l319_319563

theorem rtl_to_conventional_notation (a b c d e : ℚ) :
  (a / (b - (c * (d + e)))) = a / (b - c * (d + e)) := by
  sorry

end rtl_to_conventional_notation_l319_319563


namespace perimeter_of_semicircle_is_33_41_cm_l319_319793

def semicircle_perimeter (radius : ℝ) : ℝ :=
  π * radius + 2 * radius

theorem perimeter_of_semicircle_is_33_41_cm :
  semicircle_perimeter 6.5 ≈ 33.41 :=
by
  sorry

end perimeter_of_semicircle_is_33_41_cm_l319_319793


namespace radius_of_sphere_tetrahedron_l319_319012

/-- Radius of a sphere touching all edges of a tetrahedron with specific edge lengths. -/
theorem radius_of_sphere_tetrahedron
  (a b : ℝ)
  (h : ∀ {T : Type} [is_tetrahedron T], T.edge_lengths = [a, a, b, b, x, x]) :
  (radius : ℝ) := 
  R = (sqrt (2 * a * b)) / 4 :=
sorry

end radius_of_sphere_tetrahedron_l319_319012


namespace greatest_multiple_l319_319728

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319728


namespace candy_bar_sales_ratio_l319_319224

theorem candy_bar_sales_ratio
    (candy_bar_cost : ℕ := 2)
    (marvin_candy_sold : ℕ := 35)
    (tina_extra_earnings : ℕ := 140)
    (marvin_earnings := marvin_candy_sold * candy_bar_cost)
    (tina_earnings := marvin_earnings + tina_extra_earnings)
    (tina_candy_sold := tina_earnings / candy_bar_cost):
  tina_candy_sold / marvin_candy_sold = 3 :=
by
  sorry

end candy_bar_sales_ratio_l319_319224


namespace T_description_l319_319213

def is_single_point {x y : ℝ} : Prop := (x = 2) ∧ (y = 11)

theorem T_description :
  ∀ (T : Set (ℝ × ℝ)),
  (∀ x y : ℝ, 
    (T (x, y) ↔ 
    ((5 = x + 3 ∧ 5 = y - 6) ∨ 
     (5 = x + 3 ∧ x + 3 = y - 6) ∨ 
     (5 = y - 6 ∧ x + 3 = y - 6)) ∧ 
    ((x = 2) ∧ (y = 11))
    )
  ) →
  (T = { (2, 11) }) :=
by
  sorry

end T_description_l319_319213


namespace tangent_line_at_point_zero_two_is_correct_l319_319648

noncomputable def tangent_line_equation (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ × ℝ :=
let m := (deriv f a) in
let c := b - m * a in
fun x => (x, m * x + c)

theorem tangent_line_at_point_zero_two_is_correct :
  ∃ f : ℝ → ℝ, f = λ x, Real.exp x + Real.cos x ∧
  ∀ x y : ℝ, (x, y) ∈ (tangent_line_equation f 0 2 '' univ) ↔ x - y + 2 = 0 :=
by
  let f := λ x, Real.exp x + Real.cos x
  use f
  split
  { dsimp, rfl },
  { intros x y,
    split,
    { intro h,
      rcases h with ⟨x_val, ⟨hx1, hx2⟩⟩,
      dsimp [tangent_line_equation] at hx1,
      rw [hx1] at hx2,
      simpa using hx2 },
    { intro h,
      use x,
      dsimp [tangent_line_equation],
      split,
      { refl },
      { simp [h] } } }

end tangent_line_at_point_zero_two_is_correct_l319_319648


namespace unique_solution_exists_l319_319232

theorem unique_solution_exists (n : ℕ) :
  ∃! (x : Fin n.succ → ℝ), 
    (∑ (i : Fin n.succ), (x i - x i.succ)^2) + (x (Fin.last n))^2 = 1 / (n + 1) ∧ 
    (∀ i, x i = 1 - (i + 1) / (n + 1 : ℝ)) :=
sorry

end unique_solution_exists_l319_319232


namespace A_worked_alone_for_correct_days_l319_319346

-- Define the rates based on the conditions
def combined_rate (W : ℝ) : ℝ := W / 40
def A_rate (W : ℝ) : ℝ := W / 28
def B_rate (W : ℝ) : ℝ := combined_rate W - A_rate W

-- Define the total work done in 10 days by both A and B
def work_done_together_in_10_days (W : ℝ) : ℝ := 10 * combined_rate W

-- Define the remaining work after 10 days
def remaining_work (W : ℝ) : ℝ := W - work_done_together_in_10_days W

-- Define the number of days A worked alone to finish the remaining work
def days_A_worked_alone (W : ℝ) : ℝ := (remaining_work W) / A_rate W

theorem A_worked_alone_for_correct_days:
  ∀ (W : ℝ), days_A_worked_alone W = 21 :=
by
  intro W
  have h1 : combined_rate W = W / 40 := rfl
  have h2 : A_rate W = W / 28 := rfl
  have h3 : B_rate W =  W / 40 - W / 28,
    rw [B_rate, combined_rate, A_rate],
  have h4 : work_done_together_in_10_days W = 10 * (W / 40) := rfl,
  have h5 : remaining_work W = 3 * W / 4,
    rw [remaining_work, work_done_together_in_10_days],
  have h6 : days_A_worked_alone W = (3 * W / 4) / (W / 28),
    rw [days_A_worked_alone, remaining_work, A_rate],
  have h7 : (3 * W / 4) / (W / 28) = 21,
    ring, field_simp,
  exact h7

end A_worked_alone_for_correct_days_l319_319346


namespace student_claim_is_impossible_l319_319014

theorem student_claim_is_impossible (m n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 100) (h₃ : (m : ℚ) / n = 0.167 + ∑ i in finset.range n, (0.1 : ℚ) ^ (i + 3)) : false :=
by
  sorry

end student_claim_is_impossible_l319_319014


namespace rationalize_denominator_l319_319625

-- Define the expression as given in conditions
def expr : ℝ := sqrt (5 / (3 - sqrt 2))

-- Define the rationalized form as given in the correct answer
def rationalized_form : ℝ := sqrt (105 + 35 * sqrt 2) / 7

-- The theorem statement to be proved
theorem rationalize_denominator : 
  expr = rationalized_form :=
by sorry

end rationalize_denominator_l319_319625


namespace minimum_value_inequality_l319_319909

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : x + 2 * y + 3 * z = 1) :
  (16 / x^3 + 81 / (8 * y^3) + 1 / (27 * z^3)) ≥ 1296 := sorry

end minimum_value_inequality_l319_319909


namespace ceil_square_values_l319_319514

theorem ceil_square_values (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, (∀ m : ℕ, m = n ↔ (121 < x^2 ∧ x^2 ≤ 144) ∧ (⌈x^2⌉ = m)) ∧ n = 23 :=
by
  sorry

end ceil_square_values_l319_319514


namespace ceil_pow_sq_cardinality_l319_319519

noncomputable def ceil_pow_sq_values (x : ℝ) (h : 11 < x ∧ x ≤ 12) : ℕ :=
  ((Real.ceil(x^2)) - (Real.ceil(121)) + 1)

theorem ceil_pow_sq_cardinality :
  ∀ (x : ℝ), (11 < x ∧ x ≤ 12) → ceil_pow_sq_values x _ = 23 :=
by
  intro x hx
  let attrs := (11 < x ∧ x ≤ 12)
  sorry

end ceil_pow_sq_cardinality_l319_319519


namespace statement_B_not_true_l319_319869

def op_star (x y : ℝ) := x^2 - 2*x*y + y^2

theorem statement_B_not_true (x y : ℝ) : 3 * (op_star x y) ≠ op_star (3 * x) (3 * y) :=
by
  have h1 : 3 * (op_star x y) = 3 * (x^2 - 2 * x * y + y^2) := rfl
  have h2 : op_star (3 * x) (3 * y) = (3 * x)^2 - 2 * (3 * x) * (3 * y) + (3 * y)^2 := rfl
  sorry

end statement_B_not_true_l319_319869


namespace CN_Tower_taller_than_Space_Needle_l319_319235

theorem CN_Tower_taller_than_Space_Needle : 
  ∀ (CN_Tower_height Space_Needle_height : ℕ), 
  CN_Tower_height = 553 → Space_Needle_height = 184 → 
  (CN_Tower_height - Space_Needle_height = 369) :=
by
  intros CN_Tower_height Space_Needle_height h1 h2
  rw [h1, h2]
  norm_num
  sorry

end CN_Tower_taller_than_Space_Needle_l319_319235


namespace probability_of_sum_30_l319_319033

def primes_upto_30 : list ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
def count_ways_to_choose_2_from_10 : ℕ := 45
def favorable_pairs : list (ℕ × ℕ) := [(7, 23), (11, 19), (13, 17)]
def count_favorable_pairs : ℕ := favorable_pairs.length

theorem probability_of_sum_30 (primes : list ℕ := primes_upto_30)
  (n : ℕ := count_ways_to_choose_2_from_10)
  (f : list (ℕ × ℕ) := favorable_pairs) :
  (f.length : ℚ) / (n : ℚ) = 1 / 15 := 
sorry

end probability_of_sum_30_l319_319033


namespace varianceComparison_samplingProbability_l319_319662

noncomputable def classAScores : List ℕ := [4, 8, 9, 9, 10]
noncomputable def classBScores : List ℕ := [6, 7, 8, 9, 10]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x, (x : ℚ - m) ^ 2)).sum / l.length

def sampleSets (l : List ℕ) (n : ℕ) : List (List ℕ) :=
  if h : l.length < n then [] else List.filter (λ s, s.length = n) (l.powerset)

def probabilities (l : List ℕ) (n : ℕ) (m : ℚ) : List ℚ :=
  let samples := sampleSets l n
  samples.map (λ s, mean s)

theorem varianceComparison : variance classAScores > variance classBScores :=
by {
  sorry
}

theorem samplingProbability :
  let popMean := mean classBScores
  let sampleMeans := probabilities classBScores 2 popMean
  let validSamples := sampleMeans.filter (λ x, |x - popMean| < 1)
  validSamples.length / (sampleSets classBScores 2).length = (3 : ℚ) / 5 :=
by {
  sorry
}

end varianceComparison_samplingProbability_l319_319662


namespace total_week_cost_proof_l319_319386

-- Defining variables for costs and consumption
def cost_brand_a_biscuit : ℝ := 0.25
def cost_brand_b_biscuit : ℝ := 0.35
def cost_small_rawhide : ℝ := 1
def cost_large_rawhide : ℝ := 1.50

def odd_days_biscuits_brand_a : ℕ := 3
def odd_days_biscuits_brand_b : ℕ := 2
def odd_days_small_rawhide : ℕ := 1
def odd_days_large_rawhide : ℕ := 1

def even_days_biscuits_brand_a : ℕ := 4
def even_days_small_rawhide : ℕ := 2

def odd_day_cost : ℝ :=
  odd_days_biscuits_brand_a * cost_brand_a_biscuit +
  odd_days_biscuits_brand_b * cost_brand_b_biscuit +
  odd_days_small_rawhide * cost_small_rawhide +
  odd_days_large_rawhide * cost_large_rawhide

def even_day_cost : ℝ :=
  even_days_biscuits_brand_a * cost_brand_a_biscuit +
  even_days_small_rawhide * cost_small_rawhide

def total_cost_per_week : ℝ :=
  4 * odd_day_cost + 3 * even_day_cost

theorem total_week_cost_proof :
  total_cost_per_week = 24.80 :=
  by
    unfold total_cost_per_week
    unfold odd_day_cost
    unfold even_day_cost
    norm_num
    sorry

end total_week_cost_proof_l319_319386


namespace pocket_knife_worth_40_l319_319823

def value_of_pocket_knife (x : ℕ) (p : ℕ) (R : ℕ) : Prop :=
  p = 10 * x ∧
  R = 10 * x^2 ∧
  (∃ num_100_bills : ℕ, 2 * num_100_bills * 100 + 40 = R)

theorem pocket_knife_worth_40 (x : ℕ) (p : ℕ) (R : ℕ) :
  value_of_pocket_knife x p R → (∃ knife_value : ℕ, knife_value = 40) :=
by
  sorry

end pocket_knife_worth_40_l319_319823


namespace central_angle_of_sector_l319_319472

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def arc_length (r α : ℝ) : ℝ := r * α

theorem central_angle_of_sector :
  ∀ (r α : ℝ),
    circumference r = 2 * Real.pi + 2 →
    arc_length r α = 2 * Real.pi - 2 →
    α = Real.pi - 1 :=
by
  intros r α hcirc harc
  sorry

end central_angle_of_sector_l319_319472


namespace ceil_pow_sq_cardinality_l319_319522

noncomputable def ceil_pow_sq_values (x : ℝ) (h : 11 < x ∧ x ≤ 12) : ℕ :=
  ((Real.ceil(x^2)) - (Real.ceil(121)) + 1)

theorem ceil_pow_sq_cardinality :
  ∀ (x : ℝ), (11 < x ∧ x ≤ 12) → ceil_pow_sq_values x _ = 23 :=
by
  intro x hx
  let attrs := (11 < x ∧ x ≤ 12)
  sorry

end ceil_pow_sq_cardinality_l319_319522


namespace period_and_monotonic_intervals_l319_319051

def min_positive_period (f : ℝ → ℝ) : ℝ :=
  inf { T : ℝ | T > 0 ∧ ∀ x : ℝ, f (x + T) = f x }

noncomputable def is_monotonic_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

noncomputable def is_strictly_monotonic_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ interval → y ∈ interval → x < y → f x < f y

theorem period_and_monotonic_intervals :
  let f := fun x => Real.sin (2 * x - Real.pi / 6)
  min_positive_period f = Real.pi ∧
  (∀ k : ℤ, is_monotonic_interval f (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3))) ∧
  (∀ k : ℤ, is_strictly_monotonic_interval f (Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6))) :=
by
  sorry

end period_and_monotonic_intervals_l319_319051


namespace evaluate_expression_l319_319883

theorem evaluate_expression : (Complex.I ^ 3) + (Complex.I ^ 11) + (Complex.I ^ (-17)) + (2 * Complex.I) = -Complex.I :=
by 
  sorry

end evaluate_expression_l319_319883


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319706

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319706


namespace quadratic_fraction_equality_l319_319073

theorem quadratic_fraction_equality (r : ℝ) (h1 : r ≠ 4) (h2 : r ≠ 6) (h3 : r ≠ 5) 
(h4 : r ≠ -4) (h5 : r ≠ -3): 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) →
  r = -7/4 :=
by {
  sorry
}

end quadratic_fraction_equality_l319_319073


namespace find_original_speed_l319_319844

theorem find_original_speed :
  ∀ (v T : ℝ), 
    (300 = 212 + 88) →
    (T + 2/3 = 212 / v + 88 / (v - 50)) →
    v = 110 :=
by
  intro v T h_dist h_trip
  sorry

end find_original_speed_l319_319844


namespace polynomial_division_remainder_l319_319315

theorem polynomial_division_remainder :
  ∃ (m b : ℚ), 
    (m = 14 / 3 ∧ b = 79 / 9) ∧ 
    ∀ x : ℚ, 
      (x ^ 4 - 8 * x ^ 3 + 21 * x ^ 2 - 28 * x + 15) =
      (x ^ 2 - 3 * x + m) * (x ^ 2 + k * x + l) + (x + b) :=
begin 
  sorry 
end

end polynomial_division_remainder_l319_319315


namespace y_intercept_of_line_l319_319432

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  -- The proof steps will go here.
  sorry

end y_intercept_of_line_l319_319432


namespace number_of_possible_N_l319_319354

theorem number_of_possible_N : 
  let is_valid N := (748 * 10 + N) % 4 = 0 in
  Nat.card {N // is_valid N} = 3 :=
by
  let is_valid (N : Nat) := (748 * 10 + N) % 4 = 0
  have h : {N // is_valid N} = ({n // n = 0 ∨ n = 4 ∨ n = 8} : Set Nat), by sorry
  rw [h]
  exact Nat.card_fintype

end number_of_possible_N_l319_319354


namespace boat_license_count_l319_319370

theorem boat_license_count : 
  let letters := 3 in 
  let digits_per_slot := 10 in 
  let slots := 5 in 
  letters * digits_per_slot ^ slots = 300000 := 
by 
  let letters := 3
  let digits_per_slot := 10
  let slots := 5
  have calculation: letters * digits_per_slot ^ slots = 300000 := by sorry
  exact calculation

end boat_license_count_l319_319370


namespace value_of_expression_l319_319987

theorem value_of_expression 
  (a b : ℝ) 
  (h : b = 3 * a - 2) : 
  2 * b - 6 * a + 2 = -2 :=
by 
  intro h
  rw [h]
  linarith

end value_of_expression_l319_319987


namespace cylinder_volume_factor_l319_319541

theorem cylinder_volume_factor (h r : ℝ) :
  let V := π * r^2 * h,
      V' := π * (2.5 * r)^2 * (3 * h) in
  V' / V = 18.75 :=
by
  -- Definitions for the problem conditions
  let original_volume := π * r^2 * h
  let new_radius := 2.5 * r
  let new_height := 3 * h
  let new_volume := π * new_radius^2 * new_height
  -- Proof that the factor is 18.75
  have V := original_volume
  have V' := new_volume
  have factor := V' / V
  show factor = 18.75
  sorry

end cylinder_volume_factor_l319_319541


namespace max_area_quadrilateral_l319_319199

theorem max_area_quadrilateral (AB : ℝ) (radius : ℝ) (P : ℝ) (C D : ℝ) 
  (h1 : AB = 2 * radius)
  (h2 : radius = 1)
  (h3 : 0 ≤ P ∧ P ≤ AB)
  (h4 : 0 ≤ C ∧ C ≤ radius)
  (h5 : P = 0 ∨ P = AB ∨ (0 < P ∧ P < AB)) :
  ∃ (S : ℝ), S = 1 :=
by {
  use 1,
  sorry
}

end max_area_quadrilateral_l319_319199


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319713

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319713


namespace odd_function_behavior_l319_319107

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^3 + x + 1 else if x < 0 then x^3 + x - 1 else 0

theorem odd_function_behavior {f : ℝ → ℝ} (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  intro x hx
  have h_neg := h_pos (-x) (by linarith)
  rw [h_odd x, h_neg]
  linarith
  sorry  -- Placeholder for any additional steps

end odd_function_behavior_l319_319107


namespace vector_dot_product_l319_319497

open ScalarProduct

variables {V : Type*} [InnerProductSpace ℝ V]

-- Define the vectors a, b, c
variables (a b c : V)

-- Define the conditions
def parallel (a b : V) : Prop := ∃ λ : ℝ, b = λ • a
def perpendicular (a c : V) : Prop := ⟪a, c⟫ = 0

-- State the theorem
theorem vector_dot_product
  (h1 : parallel a b)
  (h2 : perpendicular a c) :
  ⟪c, a + 2 • b⟫ = 0 :=
sorry

end vector_dot_product_l319_319497


namespace university_students_count_l319_319026

noncomputable def total_students
  (foreign_students_percent : ℝ)
  (new_foreign_students : ℕ)
  (total_foreign_students_next_semester : ℕ) : ℕ :=
  let current_foreign_students := total_foreign_students_next_semester - new_foreign_students
  let S := current_foreign_students / foreign_students_percent
  S.to_nat

theorem university_students_count :
  total_students 0.30 200 740 = 1800 :=
by
  sorry

end university_students_count_l319_319026


namespace integer_points_between_A_and_C_l319_319357

def A : ℕ × ℕ := (1, 1)
def C : ℕ × ℕ := (120, 1181)
def line_eq (x : ℤ) : ℤ := 10 * x - 9

theorem integer_points_between_A_and_C :
  {p : ℤ × ℤ | p.1 > 1 ∧ p.1 < 120 ∧ p.2 = line_eq p.1}.finite.card = 118 := 
by sorry

end integer_points_between_A_and_C_l319_319357


namespace general_term_a_n_sum_of_sequence_b_n_l319_319915

theorem general_term_a_n (S_n : ℕ → ℚ) (S_def : ∀ n, S_n n = 2 * n^2 + 3 * n) : 
  ∀ n, let a_n := if n = 0 then S_n 0 else S_n n - S_n (n - 1) in a_n n = 4 * n + 1 := 
by
  sorry

theorem sum_of_sequence_b_n (a_n : ℕ → ℚ) (a_def : ∀ n, a_n n = 4 * n + 1) (b_n : ℕ → ℚ) 
  (b_def : ∀ n, b_n n = 1 / ((a_n n) * (a_n (n + 1)))) : 
  ∀ n, let T_n := ∑ i in finset.range n, b_n i in T_n n = n / (5 * (4 * n + 5)) := 
by
  sorry

end general_term_a_n_sum_of_sequence_b_n_l319_319915


namespace sacks_of_oranges_l319_319373

theorem sacks_of_oranges (sacks_per_day days : ℕ) (h1 : sacks_per_day = 76) (h2 : days = 63) :
  sacks_per_day * days = 4788 :=
by
  rw [h1, h2]
  -- provide the remaining proof term if needed
  sorry

end sacks_of_oranges_l319_319373


namespace banana_permutations_l319_319962

open Function

def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

def number_of_permutations (total items1 items2 : ℕ) : ℕ :=
  factorial total / (factorial items1 * factorial items2 * factorial (total - items1 - items2))

theorem banana_permutations : number_of_permutations 6 3 2 = 60 :=
by {
  unfold number_of_permutations,
  simp,
  sorry
}


end banana_permutations_l319_319962


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319697

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319697


namespace markup_calculation_l319_319659

theorem markup_calculation 
  (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) 
  (h1 : purchase_price = 48) 
  (h2 : overhead_percent = 0.30) 
  (h3 : net_profit = 12) : 
  (markup : ℝ) (h4 : markup = (purchase_price + overhead_percent * purchase_price + net_profit - purchase_price)) : 
  markup = 26.40 := 
sorry

end markup_calculation_l319_319659


namespace find_percentage_find_percentage_as_a_percentage_l319_319410

variable (P : ℝ)

theorem find_percentage (h : P / 2 = 0.02) : P = 0.04 :=
by
  sorry

theorem find_percentage_as_a_percentage (h : P / 2 = 0.02) : P = 4 :=
by
  sorry

end find_percentage_find_percentage_as_a_percentage_l319_319410


namespace number_of_roots_l319_319876

theorem number_of_roots (k : ℝ) :
  (∃ x : ℝ, x + |x^2 - 1| = k) → 
    (if k < -1 then 0 else
    if k = -1 then 1 else
    if -1 < k ∧ k < 1 then 2 else
    if k = 1 then 3 else
    if 1 < k ∧ k < 5/4 then 4 else
    if k = 5/4 then 3 else
    if k > 5/4 then 2 else 0) := 
sorry

end number_of_roots_l319_319876


namespace common_ratio_sum_arithmetic_seq_l319_319212

-- General Definitions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * (a n)

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), a i

-- Conditions from the problem
variables {a : ℕ → ℝ} {q : ℝ} (hgeom : is_geometric_sequence a q) (hq1 : q ≠ 1) (hq2 : q ≠ 0)

-- Question 1: Find the common ratio and prove it is -2.
theorem common_ratio : q = -2 := by
  sorry

-- Question 2: Prove that S_(k+2), S_k, and S_(k+1) form an arithmetic sequence.
theorem sum_arithmetic_seq (k : ℕ) (hk : 0 < k) :
  (sum_first_n_terms a (k + 2)) + (sum_first_n_terms a (k + 1)) - 2 * (sum_first_n_terms a k) = 0 := by
  sorry

end common_ratio_sum_arithmetic_seq_l319_319212


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319742

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319742


namespace max_dot_product_is_six_l319_319535

-- Define the ellipse equation
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  x^2 / 4 + y^2 / 3 = 1

-- Define the conditions
def conditions (P : ℝ × ℝ) : Prop :=
  (∃ x y, P = (x, y) ∧ is_on_ellipse P)

-- Define the dot product of OP and FP
def dot_product_OP_FP (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in
  x^2 + x + y^2

-- The theorem statement
theorem max_dot_product_is_six : ∀ P : ℝ × ℝ, conditions P → dot_product_OP_FP P ≤ 6 ∧ (∃ P' : ℝ × ℝ, conditions P' ∧ dot_product_OP_FP P' = 6) :=
by
  sorry

end max_dot_product_is_six_l319_319535


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319752

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319752


namespace number_of_5_letter_words_with_at_least_one_vowel_l319_319955

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let vowels := ['A', 'E']
  ∃ n : ℕ, n = 7^5 - 5^5 ∧ n = 13682 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l319_319955


namespace largest_value_n_under_100000_l319_319770

theorem largest_value_n_under_100000 :
  ∃ n : ℕ,
    0 ≤ n ∧
    n < 100000 ∧
    (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 ∧
    n = 99999 :=
sorry

end largest_value_n_under_100000_l319_319770


namespace sets_relation_l319_319334

def setM : set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def setN : set (ℝ × ℝ) := {p | sqrt((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + sqrt((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * sqrt(2)}

def setP : set (ℝ × ℝ) := {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem sets_relation : setM ⊆ setP ∧ setP ⊆ setN → setM ⊆ setN :=
by sorry

end sets_relation_l319_319334


namespace distinct_seven_digit_integers_l319_319501

theorem distinct_seven_digit_integers : 
  let n := 7 !
  let repeats := (2 ! * 3 ! * 2 !)
  n / repeats = 210 :=
by
  sorry

end distinct_seven_digit_integers_l319_319501


namespace sum_of_sequence_2017_l319_319099

def sequence (a : ℕ → ℚ) := (a 1 = 2) ∧ (∀ n : ℕ, a (n + 1) = 1 - 1 / (a n))
def S (a : ℕ → ℚ) (n : ℕ) := ∑ i in Finset.range n, a (i + 1)

theorem sum_of_sequence_2017 : ∀ (a : ℕ → ℚ),
  sequence a → S a 2017 = 1010 :=
by {
  sorry
}

end sum_of_sequence_2017_l319_319099


namespace blue_crayons_l319_319351

variables (B G : ℕ)

theorem blue_crayons (h1 : 24 = 8 + B + G + 6) (h2 : G = (2 / 3) * B) : B = 6 :=
by 
-- This is where the proof would go
sorry

end blue_crayons_l319_319351


namespace total_yellow_marbles_l319_319225

theorem total_yellow_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := 
by 
  sorry

end total_yellow_marbles_l319_319225


namespace probability_dot_product_l319_319097

noncomputable def probability_dot_product_gt_half (n : ℕ) [fact (n = 2017)] : ℝ :=
  let k := ⌊60/(360/n)⌋ in
  (2 * k) / (n - 1)

theorem probability_dot_product (n : ℕ) [fact (n = 2017)] :
  probability_dot_product_gt_half n = 1/3 :=
  sorry

end probability_dot_product_l319_319097


namespace range_of_f_l319_319932

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)/Real.exp x

theorem range_of_f''_over_f : 
  (∀ x : ℝ, deriv f x + f x = 2 * x * Real.exp (-x)) → 
  f 0 = 1 → 
  set.range (λ x : ℝ, (deriv^[2] f x) / f x) = Set.Icc (-2 : ℝ) 0 :=
by
  intro h_diff_eq h_init_cond
  sorry

end range_of_f_l319_319932


namespace ceil_square_range_count_l319_319517

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l319_319517


namespace greatest_multiple_of_5_and_6_under_1000_l319_319733

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319733


namespace fixed_point_of_log_function_l319_319486

theorem fixed_point_of_log_function
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (2017, 1) ∧ ∀ x : ℝ, y : ℝ, f(a, x) = y → y = log a (x - 2016) + 1 :=
begin
  sorry
end

def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2016) + 1

end fixed_point_of_log_function_l319_319486


namespace value_of_f_l319_319468

variables {α : Real}
-- Conditions
def in_second_quadrant (α : Real) : Prop := (π/2 < α ∧ α < π)
def cos_condition (α : Real) : Prop := cos (α - 3 * π / 2) = -1 / 3 

-- Function definition
def f (α : Real) : Real :=
  (sin (π - α) * tan (- α - π)) / (sin (π + α) * cos (2 * π - α) * tan (- α))

-- Theorem statement proving the required value of f(α)
theorem value_of_f (h1 : in_second_quadrant α) (h2 : cos_condition α) : 
  f α = 3 * sqrt 2 / 4 :=
sorry

end value_of_f_l319_319468


namespace sin_double_angle_value_l319_319544

theorem sin_double_angle_value 
  (α : ℝ) 
  (hα1 : π / 2 < α) 
  (hα2 : α < π)
  (h : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = - 17 / 18 := 
by
  sorry

end sin_double_angle_value_l319_319544


namespace opposite_sides_of_line_l319_319471

theorem opposite_sides_of_line 
  (x₀ y₀ : ℝ) 
  (h : (3 * x₀ + 2 * y₀ - 8) * (3 * 1 + 2 * 2 - 8) < 0) :
  3 * x₀ + 2 * y₀ > 8 :=
by
  sorry

end opposite_sides_of_line_l319_319471


namespace total_cost_for_tickets_l319_319803

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end total_cost_for_tickets_l319_319803


namespace tangent_line_ln_l319_319984

theorem tangent_line_ln (b : ℝ) : (∃ x₀ : ℝ, x₀ > 0 ∧ (1 / x₀ = 1 / real.exp 1) ∧ (b = real.log x₀ - 1)) → b = 0 := 
by 
  sorry 

end tangent_line_ln_l319_319984


namespace symmetry_center_of_sine_function_l319_319491

def f (w φ x : ℝ) : ℝ := Real.sin (w * x + φ)

theorem symmetry_center_of_sine_function :
  ∃ x : ℝ, x = -2 * Real.pi / 3 ∧ ∀ w φ : ℝ, w > 0 ∧ |φ| < Real.pi / 2 
  ∧ (∀ x : ℝ, f w φ x ≤  f w φ (Real.pi / 3)) 
  ∧ (∃ T : ℝ, T = 4 * Real.pi ∧ T = 2 * Real.pi / w) := sorry

end symmetry_center_of_sine_function_l319_319491


namespace goods_train_length_is_correct_l319_319001

def length_of_goods_train (speed_train_1 : ℕ) (speed_train_2 : ℕ) (time_s : ℕ) : ℕ :=
  let relative_speed := speed_train_1 + speed_train_2,
  let relative_speed_m_per_s := (relative_speed * 1000) / 3600,
  relative_speed_m_per_s * time_s

theorem goods_train_length_is_correct :
  length_of_goods_train 45 108 8 = 340 :=
by
  sorry

end goods_train_length_is_correct_l319_319001


namespace imaginary_part_of_fraction_l319_319168

-- Define the given complex number z using Euler's formula
def z := Complex.exp (Complex.I * Real.pi / 4)

-- Define the denominator complex number 1 - i
def denom := 1 - Complex.I

-- Define the fraction z / denom 
def fraction := z / denom

-- The theorem we want to prove
theorem imaginary_part_of_fraction : Complex.im fraction = Real.sqrt 2 / 2 :=
by sorry

end imaginary_part_of_fraction_l319_319168


namespace remainder_zero_l319_319406

theorem remainder_zero (x : ℤ) :
  (x^5 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end remainder_zero_l319_319406


namespace total_time_to_pump_540_gallons_l319_319637

-- Definitions for the conditions
def initial_rate : ℝ := 360  -- gallons per hour
def increased_rate : ℝ := 480 -- gallons per hour
def target_volume : ℝ := 540  -- total gallons
def first_interval : ℝ := 0.5 -- first 30 minutes as fraction of hour

-- Proof problem statement
theorem total_time_to_pump_540_gallons : 
  (first_interval * initial_rate) + ((target_volume - (first_interval * initial_rate)) / increased_rate) * 60 = 75 := by
  sorry

end total_time_to_pump_540_gallons_l319_319637


namespace parallelogram_ABCD_l319_319392

-- Definitions for circles, tangents, intersections, and parallel lines
variable (S1 S2 : Circle)
variable (A P B C D : Point)

def circles_intersect (S1 S2 : Circle) (A P : Point) : Prop :=
  S1.contains A ∧ S1.contains P ∧ S2.contains A ∧ S2.contains P

def tangent_at_point (S : Circle) (A B : Point) : Prop :=
  S.contains A ∧ Line.through A B ∧ perpendicular (Line.through A B) (tangent_line S A)

def line_through_parallel (P : Point) (l1 l2 : Line) : Prop :=
  point_on_line P l1 ∧ parallel l1 l2

def points_on_circles (B C : Point) (S2 : Circle) : Prop :=
  S2.contains B ∧ S2.contains C

def points_on_circles_and_line (D : Point) (S1 : Circle) : Prop :=
  S1.contains D

-- Problem statement
theorem parallelogram_ABCD
  (h1 : circles_intersect S1 S2 A P)
  (h2 : tangent_at_point S1 A B)
  (h3 : line_through_parallel P (line_through C D) (line_through A B))
  (h4 : points_on_circles B C S2)
  (h5 : points_on_circles_and_line D S1) :
  parallelogram (quadrilateral A B C D) := 
sorry

end parallelogram_ABCD_l319_319392


namespace glued_paper_length_l319_319295

theorem glued_paper_length (n : ℕ) (l : ℕ) (o : ℕ) (L : ℝ) :
  n = 15 ∧ l = 25 ∧ o = 0.5 ∧ L = l - o → (l + (n - 1) * L) / 100 = 3.68 :=
by
  sorry

end glued_paper_length_l319_319295


namespace time_taken_l319_319967

-- Define the function T which takes the number of cats, the number of rats, and returns the time in minutes
def T (n m : ℕ) : ℕ := if n = m then 4 else sorry

-- The theorem states that, given n cats and n rats, the time taken is 4 minutes
theorem time_taken (n : ℕ) : T n n = 4 :=
by simp [T]

end time_taken_l319_319967


namespace find_k_l319_319023

theorem find_k (a b k : ℝ) (h1 : a ≠ b ∨ a = b)
    (h2 : a^2 - 12 * a + k + 2 = 0)
    (h3 : b^2 - 12 * b + k + 2 = 0)
    (h4 : 4^2 - 12 * 4 + k + 2 = 0) :
    k = 34 ∨ k = 30 :=
by
  sorry

end find_k_l319_319023


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319757

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319757


namespace quadratic_inequality_roots_l319_319478

theorem quadratic_inequality_roots (a b : ℝ)
  (h_sol_set : ∀ x : ℝ, (ax^2 + bx + 1 > 0) ↔ x ∈ (-1, 1 / 3))
  (h_neg_coeff : a < 0)
  (h_sum_roots : -1 + 1/3 = -b / a)
  (h_prod_roots : -1 * (1/3) = 1 / a) :
  a - b = -1 :=
sorry

end quadratic_inequality_roots_l319_319478


namespace vector_magnitude_l319_319482

variables (a b : ℝ^3) -- assume a and b are vectors in three-dimensional space
open Real

-- hypotheses
def unit_vectors (u : ℝ^3) : Prop := u.norm = 1
def angle (u v : ℝ^3) : Prop := u ⬝ v = -1/2

-- main statement
theorem vector_magnitude : 
  unit_vectors a → unit_vectors b → angle a b → (norm (a - 2 • b) = sqrt 7) :=
by sorry

end vector_magnitude_l319_319482


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319714

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319714


namespace sixth_root_of_special_number_l319_319050

theorem sixth_root_of_special_number :
  (∃ x : ℕ, x = 24414062515625 ∧ ∛x = 51) := by
  sorry

end sixth_root_of_special_number_l319_319050


namespace airplane_altitude_l319_319390

def distance_CD : ℝ := 15
def angle_C : ℝ := 20
def angle_D : ℝ := 40
def tan_20 : ℝ := Real.tan (Real.pi / 9) / 9 -- approximation for tan(20°)
def tan_40 : ℝ := Real.tan (Real.pi / 4.5) / 4.5 -- approximation for tan(40°)
def altitude (h : ℝ) : Prop := abs (h - 3.805) < 0.005

theorem airplane_altitude : altitude (
  let DP := distance_CD * tan_20 / (tan_40 + tan_20)
  in DP * tan_40
) := sorry 

end airplane_altitude_l319_319390


namespace cost_per_load_is_25_cents_l319_319083

-- Define the given conditions
def loads_per_bottle : ℕ := 80
def usual_price_per_bottle : ℕ := 2500 -- in cents
def sale_price_per_bottle : ℕ := 2000 -- in cents
def bottles_bought : ℕ := 2

-- Defining the total cost and total loads
def total_cost : ℕ := bottles_bought * sale_price_per_bottle
def total_loads : ℕ := bottles_bought * loads_per_bottle

-- Define the cost per load in cents
def cost_per_load_in_cents : ℕ := (total_cost * 100) / total_loads

-- Formal proof statement
theorem cost_per_load_is_25_cents 
    (h1 : loads_per_bottle = 80)
    (h2 : usual_price_per_bottle = 2500)
    (h3 : sale_price_per_bottle = 2000)
    (h4 : bottles_bought = 2)
    (h5 : total_cost = bottles_bought * sale_price_per_bottle)
    (h6 : total_loads = bottles_bought * loads_per_bottle)
    (h7 : cost_per_load_in_cents = (total_cost * 100) / total_loads):
  cost_per_load_in_cents = 25 := by
  sorry

end cost_per_load_is_25_cents_l319_319083


namespace pentagon_square_ratio_l319_319839

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l319_319839


namespace value_of_a_l319_319966

theorem value_of_a (a x : ℝ) (h : x = 4) (h_eq : x^2 - 3 * x = a^2) : a = 2 ∨ a = -2 :=
by
  -- The proof is omitted, but the theorem statement adheres to the problem conditions and expected result.
  sorry

end value_of_a_l319_319966


namespace range_of_a_l319_319397

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a * (1 - x)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → f x a ≤ 2 * a - 2) : 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l319_319397


namespace largest_digit_N_divisible_by_6_l319_319309

theorem largest_digit_N_divisible_by_6 (N : ℕ) (h1 : Nat.even N) (h2 : (29 + N) % 3 = 0) (h3 : N < 10) :
  N = 4 :=
by
  sorry

end largest_digit_N_divisible_by_6_l319_319309


namespace fred_games_this_year_l319_319084

theorem fred_games_this_year (games_last_year : ℕ) (games_less_this_year : ℕ) (h1 : games_last_year = 36) (h2 : games_less_this_year = 11) : games_last_year - games_less_this_year = 25 :=
by {
    rw [h1, h2];
    rfl;
}

end fred_games_this_year_l319_319084


namespace total_fish_count_l319_319556

noncomputable def blue_fish_spots_percent := 0.50
noncomputable def yellow_fish_stripes_percent := 0.45
noncomputable def blue_fish_distribution_percent := 0.35
noncomputable def yellow_fish_distribution_percent := 0.25
noncomputable def blue_spotted_fish_count := 28
noncomputable def yellow_striped_fish_count := 15

theorem total_fish_count (T : ℝ) : 
  (0.35 * T = blue_spotted_fish_count / blue_fish_spots_percent) →
  (0.25 * T = 15 / 0.45 .approx_floor) → 
  T = 160 := 
  by sorry

end total_fish_count_l319_319556


namespace third_side_length_not_12_l319_319173

theorem third_side_length_not_12 (x : ℕ) (h1 : x % 2 = 0) (h2 : 5 < x) (h3 : x < 11) : x ≠ 12 := 
sorry

end third_side_length_not_12_l319_319173


namespace min_distance_from_circle_to_line_l319_319269

-- Definitions for the conditions.
def circle (p : ℝ × ℝ) : Prop :=
  p.fst ^ 2 + p.snd ^ 2 = 1

def line (p : ℝ × ℝ) : Prop :=
  3 * p.fst - 4 * p.snd - 10 = 0

-- Main theorem statement.
theorem min_distance_from_circle_to_line : 
  ∀ point : ℝ × ℝ, circle point → exists d : ℝ, d = 1 ∧ (∃ p : ℝ × ℝ, circle p ∧ (3 * p.fst - 4 * p.snd - 10 = d ∨ 3 * p.fst - 4 * p.snd - 10 = -d)) :=
by 
  sorry

end min_distance_from_circle_to_line_l319_319269


namespace blue_red_area_equal_l319_319434

theorem blue_red_area_equal (n : ℕ) (h : n ≥ 2) : 
  (∃ blue_area red_area, blue_area = red_area) ↔ (n % 2 = 1) :=
by
  sorry

end blue_red_area_equal_l319_319434


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319767

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319767


namespace find_numbers_l319_319665

theorem find_numbers (x y z u n : ℤ)
  (h1 : x + y + z + u = 36)
  (h2 : x + n = y - n)
  (h3 : x + n = z * n)
  (h4 : x + n = u / n) :
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
sorry

end find_numbers_l319_319665


namespace median_proof_l319_319175

-- Define the given problem and conditions
variable {A B C D E F P O : Point}
variable (triangleABC : Triangle A B C)
variable [AcuteTriangle triangleABC]
variable (circleO : Circle O)
variable [Diameter circleO D]

axiom intersect_AC : Intersects circleO (Segment A C) E
axiom intersect_AB : Intersects circleO (Segment A B) F
axiom tangents_intersect : TangentsIntersect circleO E F P

-- State the theorem
theorem median_proof : CoincidesWithOneMedian (Line A P) (Triangle A B C) := 
sorry

end median_proof_l319_319175


namespace julia_played_more_kids_l319_319193

theorem julia_played_more_kids :
  ∀ (kids_monday kids_tuesday : ℕ), 
  kids_monday = 18 → 
  kids_tuesday = 10 → 
  kids_monday - kids_tuesday = 8 :=
by
  intros kids_monday kids_tuesday h_monday h_tuesday
  rw [h_monday, h_tuesday]
  sorry

end julia_played_more_kids_l319_319193


namespace rock_splash_width_l319_319300

def total_splash_width : ℝ := 7
def pebble_splash_width : ℝ := 1 / 4
def boulder_splash_width : ℝ := 2
def num_pebbles : ℝ := 6
def num_rocks : ℝ := 3
def num_boulders : ℝ := 2

theorem rock_splash_width :
  let total_pebble_splash := num_pebbles * pebble_splash_width,
      total_boulder_splash := num_boulders * boulder_splash_width in
  (total_splash_width - total_pebble_splash - total_boulder_splash) / num_rocks = 0.5 := by
  sorry

end rock_splash_width_l319_319300


namespace smallest_munificence_monic_cubic_polynomial_l319_319433

theorem smallest_munificence_monic_cubic_polynomial :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = x^3 + a * x^2 + b * x + c) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ M) → M ≥ 1) :=
by
  sorry

end smallest_munificence_monic_cubic_polynomial_l319_319433


namespace number_of_bricks_needed_l319_319956

theorem number_of_bricks_needed :
  let V_wall := 600 * 400 * 2050
  let V_brick := 30 * 12 * 10
  V_wall / V_brick = 136667 :=
by
  let V_wall := 600 * 400 * 2050
  let V_brick := 30 * 12 * 10
  calc
    V_wall / V_brick = 492000000 / 3600   : by rfl
    ...             = 136666.6667        : by rfl
    ...             = 136667             : by sorry

end number_of_bricks_needed_l319_319956


namespace pink_rabbit_time_l319_319005

theorem pink_rabbit_time (t : ℝ) : 
  (∀ t : ℝ, 
    (∃ v_p v_w : ℝ, 
      v_p = 15 ∧ 
      v_w = 10 ∧ 
      15 * t = 10 * (t + 0.5)) 
    → t = 1) 
:= 
by
  intros _ H
  rcases H with ⟨v_p, v_w, hp, hw, h⟩
  rw [hp, hw] at h
  sorry -- proof omitted

end pink_rabbit_time_l319_319005


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319762

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319762


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319755

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319755


namespace main_theorem_l319_319505

variable {a : ℝ}

-- Define the function y = a^x
def f (a : ℝ) (x : ℝ) := a^x 

-- Define the function y = (2 - a) * x^3
def g (a : ℝ) (x : ℝ) := (2 - a) * x^3

-- State the property of g being increasing on ℝ
def is_increasing (a : ℝ) : Prop := ∀ x1 x2, x1 < x2 → g a x1 < g a x2

-- State the property of f being decreasing on ℝ
def is_decreasing (a : ℝ) : Prop := ∀ x1 x2, x1 < x2 → f a x2 < f a x1

-- The main statement to prove
theorem main_theorem (h1 : a > 0) (h2 : a ≠ 1) :
  (is_decreasing a → is_increasing (2 - a)) ∧
  (is_increasing (2 - a) → is_decreasing a → false) :=
by
  sorry

end main_theorem_l319_319505


namespace largest_prime_factor_of_expr_is_139_l319_319065

noncomputable def expr := 15^2 + 10^3 + 5^6

theorem largest_prime_factor_of_expr_is_139 :
  ∃ p : ℕ, prime p ∧ (p ∣ expr) ∧ (∀ q : ℕ, prime q ∧ (q ∣ expr) → q ≤ p) ∧ p = 139 :=
by
  sorry

end largest_prime_factor_of_expr_is_139_l319_319065


namespace volume_of_prism_l319_319230

variables {H α β : ℝ}

theorem volume_of_prism (H_pos : 0 < H) (α_pos : 0 < α) (β_pos : 0 < β) :
  let volume := (H^3 * real.cos β) / (2 * (real.sin α)^2) * 
                real.sqrt (real.sin(β + α) * real.sin(β - α)) in 
  volume = (H^3 * real.cos β) / (2 * (real.sin α)^2) * 
           real.sqrt (real.sin (β + α) * real.sin (β - α)) :=
by sorry

end volume_of_prism_l319_319230


namespace train_crossing_time_l319_319654

-- Definitions
def length_train : ℕ := 600 -- length of the train in meters
def length_platform : ℕ := 600 -- length of the platform in meters
def speed_train_km_hr : ℕ := 72 -- speed of train in km/hr

-- Conversion from km/hr to m/s
def speed_train_m_s : ℕ := (speed_train_km_hr * 1000) / 3600 -- speed of train in m/s

-- Total distance to cross the platform
def total_distance : ℕ := length_train + length_platform -- total distance to cross the platform

-- Time to cross the platform
def time_to_cross : ℕ := total_distance / speed_train_m_s -- time in seconds

-- Problem statement
theorem train_crossing_time : time_to_cross = 60 :=
by
  have speed_m_s : ℕ := (speed_train_km_hr * 1000) / 3600
  have total_dist : ℕ := length_train + length_platform
  have time_s : ℕ := total_dist / speed_m_s
  show time_s = 60
  from sorry

end train_crossing_time_l319_319654


namespace area_of_overlap_l319_319994

open Real

noncomputable def point := (ℝ × ℝ)

structure triangle :=
(vertices : point × point × point)

def area_of_triangle (T : triangle) : ℝ :=
  let ⟨(x1, y1), (x2, y2), (x3, y3)⟩ := T.vertices in
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

def tri_A : triangle := ⟨((0, 0), (2, 0), (2, 2))⟩
def tri_B : triangle := ⟨((0, 2), (2, 2), (0, 0))⟩

def overlapping_triangle : triangle := ⟨((0, 0), (2, 0), (2, 2))⟩

theorem area_of_overlap :
  area_of_triangle overlapping_triangle = 2 := by
  sorry

end area_of_overlap_l319_319994


namespace triangle_area_l319_319463

noncomputable def parabola_focus := (1, 0)

def parabola : (ℝ × ℝ) → Prop :=
  λ p, p.2 ^ 2 = 4 * p.1

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def area_of_triangle (A B F : ℝ × ℝ) : ℝ :=
  abs ((A.1 - F.1) * (B.2 - F.2) - (A.2 - F.2) * (B.1 - F.1)) / 2

theorem triangle_area {A B : ℝ × ℝ} (hA : parabola A) (hB : parabola B) (hM : midpoint A B (2, 2)) :
  area_of_triangle A B parabola_focus = 2 :=
sorry

end triangle_area_l319_319463


namespace problem_inequality_l319_319215

noncomputable theory

open Real

-- The main theorem to be proved
theorem problem_inequality
  (n : ℕ) -- Define n as a natural number
  (h : n ≥ 2) -- Condition that n is at least 2
  (a : Fin n → ℝ) -- Define 'a' as a function from Finite n to ℝ
  (h_pos : ∀ i, 0 < a i) -- Condition that all elements of a are positive
  : ∏ i in Finset.range n, (a i ^ 3 + 1) ≥ ∏ i in Finset.range n, (a i ^ 2 * a ((i + 1) % n) + 1) :=
sorry  -- Proof goes here

end problem_inequality_l319_319215


namespace complex_purely_imaginary_l319_319540

theorem complex_purely_imaginary (a : ℂ) (h1 : a^2 - 3 * a + 2 = 0) (h2 : a - 1 ≠ 0) : a = 2 :=
sorry

end complex_purely_imaginary_l319_319540


namespace weston_academy_geography_players_l319_319025

theorem weston_academy_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_players : ℕ) :
  total_players = 18 →
  history_players = 10 →
  both_players = 6 →
  ∃ (geo_players : ℕ), geo_players = 14 := 
by 
  intros h1 h2 h3
  use 18 - (10 - 6) + 6
  sorry

end weston_academy_geography_players_l319_319025


namespace thousands_digit_of_factorial_difference_l319_319774

theorem thousands_digit_of_factorial_difference :
  (30!.toNat - 25!.toNat) % 10000 / 1000 % 10 = 0 := 
sorry

end thousands_digit_of_factorial_difference_l319_319774


namespace ceil_square_values_l319_319513

theorem ceil_square_values (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, (∀ m : ℕ, m = n ↔ (121 < x^2 ∧ x^2 ≤ 144) ∧ (⌈x^2⌉ = m)) ∧ n = 23 :=
by
  sorry

end ceil_square_values_l319_319513


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319769

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319769


namespace angle_sum_l319_319176

theorem angle_sum (P Q R S : Type) (angle_PQR angle_PQS angle_SQR: ℝ) (x y : ℝ)
  (h1: angle_PQR = 90) 
  (h2: angle_PQS = 2 * y)
  (h3: angle_SQR = 2 * x) :
  x + y = 45 := 
by
  -- Use the sum of angles property
  have h := h1,
  have h4: angle_PQS + angle_SQR = 90, from sorry,
  rw [h2, h3, ←h] at h4,
  norm_num at h4,
  linarith,
  -- Final simplification steps skipped for simplicity
  sorry

end angle_sum_l319_319176


namespace cucumbers_for_20_apples_l319_319536

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end cucumbers_for_20_apples_l319_319536


namespace number_of_inverses_mod_11_l319_319959

theorem number_of_inverses_mod_11 : 
  (Finset.filter (λ n : ℕ, Nat.coprime n 11) (Finset.range 11)).card = 10 := 
by
  sorry

end number_of_inverses_mod_11_l319_319959


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319756

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319756


namespace area_circle_with_points_P_Q_l319_319687

def point := (ℝ × ℝ)

def distance (p q : point) : ℝ := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def area_of_circle (center : point) (point_on_circumference : point) : ℝ :=
  let radius := distance center point_on_circumference
  pi * radius^2

theorem area_circle_with_points_P_Q :
  let P : point := (-5, 3)
  let Q : point := (7, -2)
  area_of_circle P Q = 169 * pi :=
by
  -- Proof goes here
  sorry

end area_circle_with_points_P_Q_l319_319687


namespace determine_value_l319_319479

-- Define the conditions
variable (a : Real) -- Define the angle a as a real number.
def P : Real × Real := (1, -2) -- Define the point P with coordinates (1, -2).

-- Define a function that represents the terminal side passing through P
def terminal_side_through (a : Real) : Prop := (cos a, sin a) = (1 / sqrt (1^2 + (-2)^2), -2 / sqrt (1^2 + (-2)^2))

-- State the theorem
theorem determine_value (h : terminal_side_through a) : 2 * (sin a / cos a) = 4 := 
by 
  sorry -- Proof goes here

end determine_value_l319_319479


namespace tetrahedron_equal_reciprocal_squares_l319_319233

noncomputable def tet_condition_heights (h_1 h_2 h_3 h_4 : ℝ) : Prop :=
True

noncomputable def tet_condition_distances (d_1 d_2 d_3 : ℝ) : Prop :=
True

theorem tetrahedron_equal_reciprocal_squares
  (h_1 h_2 h_3 h_4 d_1 d_2 d_3 : ℝ)
  (hc_hts : tet_condition_heights h_1 h_2 h_3 h_4)
  (hc_dsts : tet_condition_distances d_1 d_2 d_3) :
  1 / (h_1 ^ 2) + 1 / (h_2 ^ 2) + 1 / (h_3 ^ 2) + 1 / (h_4 ^ 2) =
  1 / (d_1 ^ 2) + 1 / (d_2 ^ 2) + 1 / (d_3 ^ 2) :=
sorry

end tetrahedron_equal_reciprocal_squares_l319_319233


namespace time_to_fill_bucket_completely_l319_319161

-- Define the conditions given in the problem
def time_to_fill_two_thirds (time_filled: ℕ) : ℕ := 90

-- Define what we need to prove
theorem time_to_fill_bucket_completely (time_filled: ℕ) : 
  time_to_fill_two_thirds time_filled = 90 → time_filled = 135 :=
by
  sorry

end time_to_fill_bucket_completely_l319_319161


namespace ways_to_sum_to_21_with_six_rolls_l319_319353

theorem ways_to_sum_to_21_with_six_rolls : 
  (Finset.card {s : (Fin 6) → Fin 6 | s.sum (λ i, s i) = 21}) = 15504 := 
sorry

end ways_to_sum_to_21_with_six_rolls_l319_319353


namespace number_of_elements_in_set_A_l319_319285

open Set

def floor_div (n: ℤ) (d: ℤ) : ℤ := Int.floor (n / d)

def A : Set ℤ := { x | ∃ k : ℤ, 100 ≤ k ∧ k ≤ 999 ∧ x = floor_div (5 * k) 6 }

theorem number_of_elements_in_set_A : Finite.card A = 750 := by
  sorry

end number_of_elements_in_set_A_l319_319285


namespace garden_radius_l319_319323

theorem garden_radius :
  ∃ (r : ℝ), (2 * Real.pi * r = (1 / 5) * Real.pi * r^2) → r = 10 :=
begin
  use 10,
  intro h,
  sorry,
end

end garden_radius_l319_319323


namespace function_ordering_l319_319102

-- Definitions for the function and conditions
variable (f : ℝ → ℝ)

-- Assuming properties of the function
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodicity : ∀ x, f (x + 4) = -f x
axiom increasing_on : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 2 → f x < f y

-- Main theorem statement
theorem function_ordering : f (-25) < f 80 ∧ f 80 < f 11 :=
by 
  sorry

end function_ordering_l319_319102


namespace min_marked_cells_l319_319771

theorem min_marked_cells (marking : ℕ → ℕ → Prop) :
    (∀ i j, (i ≤ 3 ∨ i ≥ -3 ∨ j ≤ 3 ∨ j ≥ -3) → marking i j)
    ∧ ∀ x y, (∀ x y, x ∈ range 8 → y ∈ range 9 →
    (∃ i, ((0 ≤ i ∧ i < 5) →
    (marking (x + i) y ∨ marking x (y + i) ∨ 
    (marking (x + i) (y + i) ∨ marking (x + i) (y - i))))) 
    → (x ≤ 14))) :=
sorry

end min_marked_cells_l319_319771


namespace equilateral_triangle_on_three_concentric_circles_l319_319015

/-- 
Given three concentric circles with radii r₁, r₂, and r₃, prove that there exists
an equilateral triangle ABC with each vertex on one of the three circles.
-/
theorem equilateral_triangle_on_three_concentric_circles
  (O : Type*)
  [euclidean_space O]
  (r1 r2 r3 : ℝ)
  (h_radii : 0 < r1 ∧ r1 < r2 ∧ r2 < r3) :
  ∃ (A B C : O),
  distance O A = r1 ∧
  distance O B = r2 ∧
  distance O C = r3 ∧
  distance A B = distance B C ∧
  distance B C = distance C A :=
sorry

end equilateral_triangle_on_three_concentric_circles_l319_319015


namespace circumference_of_circle_l319_319009

def rectangleInscribedInCircle (width height : ℝ) : Prop :=
  ∃ (r : ℝ), r = (real.sqrt (width^2 + height^2)) / 2 ∧ ∀ (p : ℝ), p = width / 2 ∨ p = height / 2 → p ≤ r

theorem circumference_of_circle (r : ℝ) (h1 : r = (real.sqrt (9^2 + 12^2)) / 2) :
  2 * real.pi * r = 15 * real.pi :=
by sorry

end circumference_of_circle_l319_319009


namespace laura_has_435_dollars_l319_319227

-- Define the monetary values and relationships
def darwin_money := 45
def mia_money := 2 * darwin_money + 20
def combined_money := mia_money + darwin_money
def laura_money := 3 * combined_money - 30

-- The theorem to prove: Laura's money is $435
theorem laura_has_435_dollars : laura_money = 435 := by
  sorry

end laura_has_435_dollars_l319_319227


namespace monotonic_intervals_range_of_a_l319_319106

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 2 * (-1) * Real.exp x - x * 0

-- Monotonic intervals
theorem monotonic_intervals :
  (∀ x < 0, f' x < 0) ∧ (∀ x > 0, f' x > 0) :=
sorry

-- Range of values for a
theorem range_of_a (a : ℝ) :
  (∀ x > 0, a * f x < Real.exp x - x) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end monotonic_intervals_range_of_a_l319_319106


namespace area_identity_l319_319372

open Real

variables {A B C D E F : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E] [LinearOrder F]

-- Define the rectangle ABCD
variable (rectangle_ABCD : ∀ (A B C D : Type), Prop)

-- Define points E and F
variable (point_on_BC : ∀ (E : Type), Prop)
variable (point_on_CD : ∀ (F : Type), Prop)

-- Define an equilateral triangle AEF
variable (equilateral_AEF : ∀ (A E F : Type), Prop)

noncomputable def area_of_triangle {T: Type} : Real := sorry

-- Specific function to compute the area of triangles ABE, AFD and CEF given the conditions
def triangle_ABE_area {A B E : Type} [triangle_AEF : equilateral_AEF A B C] (theta : Real) : Real :=
  1/2 * sin(theta)

def triangle_AFD_area {A F D : Type} [triangle_AEF : equilateral_AEF A E F] (theta : Real) : Real :=
  1/2 * sin(theta - 60)

def triangle_CEF_area {C E F : Type} [triangle_AEF : equilateral_AEF A E F] (theta : Real) : Real :=
  1/2 * sin(theta - 30)

theorem area_identity :
  ∀ (rectangle_ABCD : ∀ (A B C D : Type), Prop)
    (point_on_BC : ∀ (E : Type), Prop)
    (point_on_CD : ∀ (F : Type), Prop)
    (equilateral_AEF : ∀ (A E F : Type), Prop)
    (θ : Real),
  rectangle_ABCD A B C D →
  point_on_BC E →
  point_on_CD F →
  equilateral_AEF A E F →
  triangle_CEF_area θ = triangle_ABE_area θ + triangle_AFD_area θ :=
by
  intros h1 h2 h3
  sorry

end area_identity_l319_319372


namespace greatest_multiple_of_5_and_6_under_1000_l319_319732

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319732


namespace sqrt_sum_iff_divisible_l319_319201

theorem sqrt_sum_iff_divisible (n : ℕ) (hn : n > 0) : 
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (nat.sqrt x + nat.sqrt y = nat.sqrt n)) ↔ 
  (∃ k : ℕ, k > 1 ∧ k^2 ∣ n) :=
sorry

end sqrt_sum_iff_divisible_l319_319201


namespace exactly_one_defective_l319_319298

theorem exactly_one_defective (p_A p_B : ℝ) (hA : p_A = 0.04) (hB : p_B = 0.05) :
  ((p_A * (1 - p_B)) + ((1 - p_A) * p_B)) = 0.086 :=
by
  sorry

end exactly_one_defective_l319_319298


namespace probability_units_digit_l319_319824

theorem probability_units_digit :
  let favorable := {0, 1, 2, 7} in
  let total := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  (favorable.count.toRational / total.count.toRational) = (2 / 5) :=
by
  sorry

end probability_units_digit_l319_319824


namespace prove_billy_age_l319_319030

-- Define B and J as real numbers representing the ages of Billy and Joe respectively
variables (B J : ℝ)

-- State the conditions
def billy_triple_of_joe : Prop := B = 3 * J
def sum_of_ages : Prop := B + J = 63

-- State the proposition to prove
def billy_age_proof : Prop := B = 47.25

-- Main theorem combining the conditions and the proof statement
theorem prove_billy_age (h1 : billy_triple_of_joe B J) (h2 : sum_of_ages B J) : billy_age_proof B :=
by
  sorry

end prove_billy_age_l319_319030


namespace ratio_of_colored_sheets_l319_319671

theorem ratio_of_colored_sheets
    (total_sheets : ℕ)
    (num_binders : ℕ)
    (sheets_colored_by_justine : ℕ)
    (sheets_per_binder : ℕ)
    (h1 : total_sheets = 2450)
    (h2 : num_binders = 5)
    (h3 : sheets_colored_by_justine = 245)
    (h4 : sheets_per_binder = total_sheets / num_binders) :
    (sheets_colored_by_justine / Nat.gcd sheets_colored_by_justine sheets_per_binder) /
    (sheets_per_binder / Nat.gcd sheets_colored_by_justine sheets_per_binder) = 1 / 2 := by
  sorry

end ratio_of_colored_sheets_l319_319671


namespace cylindrical_to_rectangular_coordinates_l319_319044

theorem cylindrical_to_rectangular_coordinates :
  ∀ (r θ z : ℝ), r = 10 → θ = real.pi / 6 → z = 2 →
  (r * real.cos θ, r * real.sin θ, z) = (5 * real.sqrt 3, 5, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  sorry

end cylindrical_to_rectangular_coordinates_l319_319044


namespace ab_divides_a_squared_plus_b_squared_l319_319601

theorem ab_divides_a_squared_plus_b_squared (a b : ℕ) (hab : a ≠ 1 ∨ b ≠ 1) (hpos : 0 < a ∧ 0 < b) (hdiv : (ab - 1) ∣ (a^2 + b^2)) :
  a^2 + b^2 = 5 * a * b - 5 := 
by
  sorry

end ab_divides_a_squared_plus_b_squared_l319_319601


namespace volume_increase_by_eight_l319_319991

theorem volume_increase_by_eight (r : ℝ) (A V : ℝ) 
  (hA : A = 4 * π * r^2) 
  (hV : V = (4 / 3) * π * r^3) : 
  let A' := 4 * A in 
  let r' := 2 * r in
  let V' := (4 / 3) * π * r'^3 in
  V' = 8 * V :=
by
  let A' := 4 * A
  let r' := 2 * r
  let V' := (4 / 3) * π * (r')^3
  sorry

end volume_increase_by_eight_l319_319991


namespace monotonicity_inequality_derivative_midpoint_roots_l319_319123

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 + (1 - a) * x - a * Real.log x

-- 1. Prove monotonicity
theorem monotonicity (a : ℝ) : 
  (a <= 0 → ∀ x > 0, deriv (λ x, f x a) x > 0) ∧ 
  (a > 0 → (∀ x, 0 < x ∧ x < a → deriv (λ x, f x a) x < 0) ∧ ∀ x, x > a → deriv (λ x, f x a) x > 0) :=
sorry

-- 2. Prove inequality f(x+a) < f(a-x)
theorem inequality (a x : ℝ) (h₀ : a > 0) (h₁ : 0 < x) (h₂ : x < a) : 
  f (x + a) a < f (a - x) a :=
sorry

-- 3. Prove the derivative at the midpoint of roots
theorem derivative_midpoint_roots (a x₁ x₂ : ℝ) (h₀ : f x₁ a = 0) (h₁ : f x₂ a = 0) (h₂ : 0 < x₁) (h₃ : x₁ < x₂) : 
  deriv (λ x, f x a) ((x₁ + x₂) / 2) > 0 :=
sorry

end monotonicity_inequality_derivative_midpoint_roots_l319_319123


namespace trigonometric_identity_l319_319292

theorem trigonometric_identity : 
  let sin := Real.sin
  let cos := Real.cos
  sin 18 * cos 63 - sin 72 * sin 117 = - (Real.sqrt 2 / 2) :=
by
  -- The proof would go here
  sorry

end trigonometric_identity_l319_319292


namespace taxi_fare_problem_l319_319290

theorem taxi_fare_problem :
  ∃ y : ℝ, 
    let initial_fare := 3.50,
        additional_fare_per_0_1_mile := 0.25,
        first_segment := 0.75,
        total_fare_available := 12.0,
        additional_miles := y - first_segment in
    initial_fare + (additional_fare_per_0_1_mile * (additional_miles / 0.1)) = total_fare_available 
      ∧ y = 4.15 :=
by
  sorry

end taxi_fare_problem_l319_319290


namespace max_binomial_coefficient_max_coefficient_terms_l319_319061

-- Define the binomial expansion conditions
def binomial_expansion (x : ℂ) : ℂ := (sqrt x + 2 / x^2)^8

-- Theorem for the term with the maximum binomial coefficient
theorem max_binomial_coefficient (x : ℂ) :
  ∃ (t : ℂ), t = binomial_expansion x ∧ t = 1120 / x^6 := 
by
  sorry

-- Theorem for the term with the maximum coefficient
theorem max_coefficient_terms (x : ℂ) :
  ∃ (t1 t2 : ℂ), t1 = binomial_expansion x ∧ t1 = 1792 * x^(-17/2) ∧
                  t2 = binomial_expansion x ∧ t2 = 1792 * x^(-11) :=
by
  sorry

end max_binomial_coefficient_max_coefficient_terms_l319_319061


namespace infinite_double_perfect_squares_l319_319157

-- Definition of a double number
def is_double_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (d : ℕ), d ≠ 0 ∧ 10^k * d + d = n ∧ 10^k ≤ d ∧ d < 10^(k+1)

-- The theorem statement
theorem infinite_double_perfect_squares :
  ∃ (S : Set ℕ), (∀ n ∈ S, is_double_number n ∧ ∃ m, m * m = n) ∧
  Set.Infinite S :=
sorry

end infinite_double_perfect_squares_l319_319157


namespace bags_weight_after_removal_l319_319807

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end bags_weight_after_removal_l319_319807


namespace h_is_odd_l319_319443

def f (x : ℝ) : ℝ := sqrt (4 - x^2)
def g (x : ℝ) : ℝ := abs (x - 2)
def h (x : ℝ) : ℝ := f x / (2 - g x)

theorem h_is_odd : h (-x) = -h (x) := 
sorry

end h_is_odd_l319_319443


namespace symmetry_axis_of_cos_function_l319_319444

theorem symmetry_axis_of_cos_function (ω : ℝ) 
  (h : ω = 2) : 
  f(x) = Real.cos (2*x + (Real.pi / 3)) := 
by
  -- Function definition 
  let y := λ x : ℝ, Real.exp x - Real.exp (2*x)
  have extremum_point : ∃ x, y' x = 0 := 
     -- Find the derivative y'
     let y' := λ x : ℝ, Real.exp x - 2 * Real.exp (2 * x)
     -- Setup the equation y' = 0
     sorry
  -- Given condition
  have w_value : ω = 2 :=
     -- Verify ω from extremum point
     sorry
  
  -- Function transformation to f(x)
  let f := λ x : ℝ, Real.cos (ω * x + Real.pi / 3)
  have f_transformed : f = λ x : ℝ, Real.cos (2*x + (Real.pi / 3)) := 
     -- Substitute ω with 2 in f(x)
     sorry
  
  -- Symmetry axis proof
  show x = (Real.pi / 3)

end symmetry_axis_of_cos_function_l319_319444


namespace light_intensity_after_3_glasses_l319_319314

theorem light_intensity_after_3_glasses (a : ℝ) :
  let reduction_factor := 0.9 in
  let final_intensity := a * (reduction_factor ^ 3) in
  final_intensity = 0.729 * a :=
by
  sorry

end light_intensity_after_3_glasses_l319_319314


namespace pentagon_square_ratio_l319_319840

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l319_319840


namespace choose_points_on_circle_l319_319815

theorem choose_points_on_circle :
  ∃ (S : Finset ℕ), S.card = 8 ∧
  ∀ (x y ∈ S), (x - y) % 24 ≠ 3 ∧ (x - y) % 24 ≠ 8 ∧ (x ≠ y) :=
  sorry

end choose_points_on_circle_l319_319815


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319693

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319693


namespace percentage_food_per_cat_correct_l319_319388

noncomputable def percentage_food_per_cat (D C : ℝ) : ℝ :=
  if h : 4 * C = D then 
    let total_food := 7 * D + 4 * C in
    let percentage := (C / total_food) * 100 in
    percentage
  else 
    0 -- This handles the case where 4 * C ≠ D, which should not happen given the conditions.

theorem percentage_food_per_cat_correct (D C : ℝ) (h : 4 * C = D) (total_food := 7 * D + 4 * C) :
  percentage_food_per_cat D C = 100 / 32 :=
by
  simp only [percentage_food_per_cat, h]
  sorry

end percentage_food_per_cat_correct_l319_319388


namespace sum_of_reciprocals_of_roots_l319_319899

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 * r2 = 7) (h2 : r1 + r2 = 16) :
  (1 / r1) + (1 / r2) = 16 / 7 :=
by
  sorry

end sum_of_reciprocals_of_roots_l319_319899


namespace final_sale_price_l319_319352

def initial_price : ℝ := 4000
def discount1 : ℝ := 0.15
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.08
def flat_discount : ℝ := 300

theorem final_sale_price :
  let price_after_first_discount := initial_price * (1 - discount1) in
  let price_after_second_discount := price_after_first_discount * (1 - discount2) in
  let price_after_third_discount := price_after_second_discount * (1 - discount3) in
  let final_price := price_after_third_discount - flat_discount in
  final_price = 2515.20 :=
by
  sorry

end final_sale_price_l319_319352


namespace solve_sqrt_cubic_equation_l319_319241

theorem solve_sqrt_cubic_equation (x : ℝ) 
    (h : sqrt (2 + sqrt (3 + sqrt x)) = real.cbrt (2 + sqrt x)) :
    x = (real.cbrt (2 + sqrt x) - 2) ^ 2 :=
by
  sorry

end solve_sqrt_cubic_equation_l319_319241


namespace avg_weight_increase_l319_319641

-- Conditions
def avg_weight_students : ℕ → ℕ → ℤ := λ total_weight num_students, (total_weight : ℤ) / (num_students : ℤ)
def total_weight_students := (24 * 35 : ℤ)
def weight_teacher := (45 : ℤ)
def new_total_weight := total_weight_students + weight_teacher
def num_students_with_teacher := (25 : ℕ)
def avg_weight_with_teacher := new_total_weight / (num_students_with_teacher : ℤ)

-- Question and answer
def increase_in_avg_weight := avg_weight_with_teacher - avg_weight_students 840 24
def increase_in_avg_weight_grams := increase_in_avg_weight * 1000

theorem avg_weight_increase :
  increase_in_avg_weight_grams = 400 := by
  sorry

end avg_weight_increase_l319_319641


namespace sum_less_than_addends_then_both_negative_l319_319783

theorem sum_less_than_addends_then_both_negative {a b : ℝ} (h : a + b < a ∧ a + b < b) : a < 0 ∧ b < 0 := 
sorry

end sum_less_than_addends_then_both_negative_l319_319783


namespace problem_I_problem_II_problem_III_l319_319450

-- (I) If \( k = 3 \) and \( a_1 = 2 \), find the sequence.
theorem problem_I (a_2 a_3 : ℝ) (h_seq_valid : 0 < a_2 ∧ 0 < a_3)
  (h1 : a_1 = 2) (h3 : a_3 = a_1) 
  (h_eq1 : a_1 + 2 / a_1 = 2 * a_2 + 1 / a_2)
  (h_eq2 : a_2 + 2 / a_2 = 2 * a_3 + 1 / a_3) :
  (a_1, a_2, a_3) = (2, 1/2, 2) :=
sorry

-- (II) If \( k = 4 \), find the set of all possible values of \( a_1 \).
theorem problem_II (a_2 a_3 a_4 : ℝ) (h_seq_valid : 0 < a_2 ∧ 0 < a_3 ∧ 0 < a_4)
  (h4 : a_4 = a_1) 
  (h_eq1 : a_1 + 2 / a_1 = 2 * a_2 + 1 / a_2)
  (h_eq2 : a_2 + 2 / a_2 = 2 * a_3 + 1 / a_3)
  (h_eq3 : a_3 + 2 / a_3 = 2 * a_4 + 1 / a_4) :
  a_1 ∈ {1/2, 1, 2} :=
sorry

-- (III) If \( k \) is even, find the maximum value of \( a_1 \).
theorem problem_III (a_1: ℝ) (k : ℕ) (hk: even k) (hk_ge3 : k ≥ 2) 
  (h_seq : ∀ n, 1 ≤ n ∧ n < k → 0 < a_n ∧ 0 < a_{n+1} ∧ a_n + 2 / a_n = 2 * a_{n+1} + 1 / a_{n+1}) :
  a_1 ≤ 2^(k / 2 - 1) :=
sorry

end problem_I_problem_II_problem_III_l319_319450


namespace count_board_configurations_l319_319562

-- Define the 3x3 board as a type with 9 positions
inductive Position 
| top_left | top_center | top_right
| middle_left | center | middle_right
| bottom_left | bottom_center | bottom_right

-- Define an enum for players' moves
inductive Mark
| X | O | Empty

-- Define a board as a mapping from positions to marks
def Board : Type := Position → Mark

-- Define the win condition for Carl
def win_condition (b : Board) : Prop := 
(b Position.center = Mark.O) ∧ 
((b Position.top_left = Mark.O ∧ b Position.top_center = Mark.O) ∨ 
(b Position.middle_left = Mark.O ∧ b Position.middle_right = Mark.O) ∨ 
(b Position.bottom_left = Mark.O ∧ b Position.bottom_center = Mark.O))

-- Define the condition for a filled board
def filled_board (b : Board) : Prop :=
∀ p : Position, b p ≠ Mark.Empty

-- The proof problem to show the total number of configurations is 30
theorem count_board_configurations : 
  ∃ (n : ℕ), n = 30 ∧
  (∃ b : Board, win_condition b ∧ filled_board b) := 
sorry

end count_board_configurations_l319_319562


namespace cost_of_toys_target_weekly_price_l319_319054

-- First proof problem: Cost of Plush Toy and Metal Ornament
theorem cost_of_toys (x : ℝ) (hx : 6400 / x = 2 * (4000 / (x + 20))) : 
  x = 80 :=
by sorry

-- Second proof problem: Price to achieve target weekly profit
theorem target_weekly_price (y : ℝ) (hy : (y - 80) * (10 + (150 - y) / 5) = 720) :
  y = 140 :=
by sorry

end cost_of_toys_target_weekly_price_l319_319054


namespace cucumbers_for_20_apples_l319_319538

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end cucumbers_for_20_apples_l319_319538


namespace angle_greater_difference_l319_319279

theorem angle_greater_difference (A B C : ℕ) (h1 : B = 5 * A) (h2 : A + B + C = 180) (h3 : A = 24) 
: C - A = 12 := 
by
  -- Proof omitted
  sorry

end angle_greater_difference_l319_319279


namespace problem_statement_l319_319489

noncomputable def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x + π / 3) + cos (ω * x - π / 3) - 1
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 6) - 1 

theorem problem_statement (ω : ℝ) (x : ℝ) (h₁ : ω > 0) (h₂ : ∃ T > 0, ∀ x, f ω (x + T) = f ω x) (h₃ : T = π) :
  (f ω x = 2 * sin (2 * x + π / 6) - 1) ∧ 
  (∀ x ∈ Icc 0 (π / 2), -2 ≤ g x ∧ g x ≤ 1) :=
sorry

end problem_statement_l319_319489


namespace fox_catch_hares_min_speed_l319_319820

theorem fox_catch_hares_min_speed :
  ∀ (v : ℝ), v ≥ 1 + Real.sqrt 2 →
    ∃ (A B C D : ℝ × ℝ),
      A = (0, 0) ∧
      B = (1, 0) ∧
      C = (1, 1) ∧
      D = (0, 1) ∧
      ∃ F : ℝ × ℝ, F = (0.5, 0.5) ∧
      (distance F A ≤ 1 ∧ distance F C ≤ 1) :=
by
  sorry

end fox_catch_hares_min_speed_l319_319820


namespace find_function_l319_319931

noncomputable def collinear_vectors (OA OB OC : Vector) (y : ℝ) (f'1 : ℝ) (x : ℝ) : Prop :=
  OA - (y + 2 * f'1) • OB + real.log (x+1) • OC = 0

theorem find_function (OA OB OC : Vector) (y : ℝ) (f'1 : ℝ) (x : ℝ) (h : collinear_vectors OA OB OC y f'1 x) :
  ∃ f : ℝ → ℝ, f = real.log ∘ (+1) :=
sorry

end find_function_l319_319931


namespace angle_cosine_relationship_l319_319164

theorem angle_cosine_relationship {A B : ℝ} (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180) :
  (A > B ↔ real.cos (A * real.pi / 180) < real.cos (B * real.pi / 180)) :=
by sorry

end angle_cosine_relationship_l319_319164


namespace find_m_l319_319160

theorem find_m (m : ℝ) : (∀ x : ℝ, x^2 - 4 * x + m = 0) → m = 4 :=
by
  intro h
  sorry

end find_m_l319_319160


namespace smallest_number_of_stamps_l319_319889

theorem smallest_number_of_stamps :
  ∃ m : ℕ, (∀ V : ℕ, V > 1 ∧ V < m → m % V = 0) ∧ (∀ W : ℕ, W > 1 ∧ W < m → least_factors W) ∧ (m=2304) := sorry

end smallest_number_of_stamps_l319_319889


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319690

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319690


namespace sum_f_1_2048_eq_22_l319_319903

def f (n : ℕ) : ℚ :=
if (∃ k : ℤ, real.log n / real.log 8 = k) then (real.log n / real.log 8 : ℚ) else 0

theorem sum_f_1_2048_eq_22 : (finset.sum (finset.range 2048) f) = 22 := 
sorry

end sum_f_1_2048_eq_22_l319_319903


namespace angle_BFP_half_angle_B_l319_319546

theorem angle_BFP_half_angle_B (A B C I F P : Point)
  (h_triangle : Triangle A B C)
  (h_angle_A : ∠A = 60)
  (h_incenter : IsIncenter I A B C)
  (h_parallel : Parallel (Line_through I parallel_to A C) AC)
  (h_intersect : Intersect (Line_through I parallel_to A C) AB F)
  (h_point_P : OnLine P BC)
  (h_BP_BC : 3 * dist B P = dist B C) :
  ∠BFP = (1/2) * ∠B :=
sorry

end angle_BFP_half_angle_B_l319_319546


namespace closest_point_l319_319069

-- Definitions
def line (x : ℝ) : ℝ := -2 * x + 3
def point : Prod ℝ ℝ := (3, -1)

-- Theorem stating the closest point on the line to (3, -1)
theorem closest_point : ∃ x y, (y = -2 * x + 3) ∧ (x, y) = (11 / 5, -7 / 5) :=
by
  use 11/5
  use -7/5
  split
  . -- Proof that the point satisfies the line equation
    sorry 
  . -- Proof that this point is indeed (11/5, -7/5)
    sorry

end closest_point_l319_319069


namespace correct_statements_l319_319591

structure Line := (name : String)
structure Plane := (name : String)

variable (a b : Line)
variable (α β γ : Plane)

axiom line_diff : a ≠ b
axiom plane_diff_1 : α ≠ β
axiom plane_diff_2 : α ≠ γ
axiom plane_diff_3 : β ≠ γ

axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom lines_parallel (l1 l2 : Line) : Prop
axiom planes_parallel (p1 p2 : Plane) : Prop

theorem correct_statements :
  ¬((line_in_plane b α ∧ lines_parallel a b) → lines_parallel a α) ∧
  ((lines_parallel a α ∧ (∃ b : Line, (line_in_plane b α ∧ line_in_plane b β))) ∧ line_in_plane a β → lines_parallel a b) ∧
  ((line_in_plane a α ∧ line_in_plane b α ∧ (∃ p : Point, a ∩ b = p) ∧ lines_parallel a β ∧ lines_parallel b β) → planes_parallel α β) ∧
  ((planes_parallel α β ∧ ∃ l1 : Line, line_in_plane l1 γ ∧ l1 = a ∧ line_in_plane l1 β ∧ l1 = b) → lines_parallel a b) :=
sorry

end correct_statements_l319_319591


namespace count_valid_m_for_fraction_l319_319906

theorem count_valid_m_for_fraction (m : ℕ) (h : (m > 0) ∧ (∃ k : ℕ, 3432 = k * (m^3 - 2))) :
  {m : ℕ | (m > 0) ∧ ∃ k : ℕ, 3432 = k * (m^3 - 2)}.card = 2 :=
sorry

end count_valid_m_for_fraction_l319_319906


namespace max_stickers_one_student_l319_319843

def total_students : ℕ := 25
def mean_stickers : ℕ := 4
def total_stickers := total_students * mean_stickers
def minimum_stickers_per_student : ℕ := 1
def minimum_stickers_taken_by_24_students := (total_students - 1) * minimum_stickers_per_student

theorem max_stickers_one_student : 
  total_stickers - minimum_stickers_taken_by_24_students = 76 := by
  sorry

end max_stickers_one_student_l319_319843


namespace modular_inverse_5_mod_29_l319_319426

theorem modular_inverse_5_mod_29 : ∃ a : ℤ, 0 ≤ a ∧ a < 29 ∧ (5 * a ≡ 1 [MOD 29]) ∧ a = 6 :=
by
  sorry

end modular_inverse_5_mod_29_l319_319426


namespace alexander_spends_total_amount_l319_319846

theorem alexander_spends_total_amount :
  (5 * 1) + (2 * 2) = 9 :=
by
  sorry

end alexander_spends_total_amount_l319_319846


namespace min_value_expression_l319_319068

theorem min_value_expression : ∀ x : ℝ, ∃ y : ℝ, y = (sin x)^4 + (cos x)^4 + 3 / (sin x)^2 + (cos x)^2 + 3 ∧ y = 3/8 := sorry

end min_value_expression_l319_319068


namespace polar_to_cartesian_l319_319868

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ * Real.cos θ = 1) :
  (∃ x, x = ρ * Real.cos θ ∧ x = 1) :=
by
  use ρ * Real.cos θ
  split
  · rfl
  · exact h
  sorry

end polar_to_cartesian_l319_319868


namespace find_b_c_d_l319_319864

def f (x : ℝ) := x^3 + 2 * x^2 + 3 * x + 4
def h (x : ℝ) := x^3 + 6 * x^2 - 8 * x + 16

theorem find_b_c_d :
  (∀ r : ℝ, f r = 0 → h (r^3) = 0) ∧ h (x : ℝ) = x^3 + 6 * x^2 - 8 * x + 16 :=
by 
  -- proof not required
  sorry

end find_b_c_d_l319_319864


namespace length_MN_l319_319616

-- Define points on a line
variables (A B C D M N : ℝ)

-- Define conditions
def is_midpoint (P Q R : ℝ) := P = (Q + R) / 2
def AD := A - D = 68
def BC := B - C = 20
def AM_mid := is_midpoint M A C
def BN_mid := is_midpoint N B D

-- The goal to prove
theorem length_MN (hAD : AD) (hBC : BC) (hAM : AM_mid) (hBN : BN_mid) : M - N = 24 :=
by
  sorry

end length_MN_l319_319616


namespace exists_integers_cd_iff_divides_l319_319248

theorem exists_integers_cd_iff_divides (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (a - b) ∣ (2 * a * b) := 
by
  sorry

end exists_integers_cd_iff_divides_l319_319248


namespace base9_to_base3_8723_eq_22210210_l319_319867

-- Given conditions
def digit_base_9_to_base_3 : ℕ → ℕ
| 0 := 0 | 1 := 1 | 2 := 2
| 3 := 10 | 4 := 11 | 5 := 12
| 6 := 20 | 7 := 21 | 8 := 22

def base9_to_base3 (n : ℕ) : ℕ :=
let d3 := digit_base_9_to_base_3 (n / 1000 % 10),
    d2 := digit_base_9_to_base_3 (n / 100 % 10),
    d1 := digit_base_9_to_base_3 (n / 10 % 10),
    d0 := digit_base_9_to_base_3 (n % 10)
in d3 * 100000000 + d2 * 10000 + d1 * 100 + d0

-- Correct answer
theorem base9_to_base3_8723_eq_22210210 :
  base9_to_base3 8723 = 22210210 :=
sorry

end base9_to_base3_8723_eq_22210210_l319_319867


namespace miles_left_l319_319678

theorem miles_left (total_miles hiked_miles : ℕ) (total_weight weight_difference : ℕ)
  (tripp_weight charlotte_weight : ℕ) :
  total_miles = 36 →
  hiked_miles = 9 →
  total_weight = 25 →
  weight_difference = 7 →
  tripp_weight = total_weight →
  charlotte_weight = total_weight - weight_difference →
  total_miles - hiked_miles = 27 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2]
  exact rfl

end miles_left_l319_319678


namespace rectangle_area_relation_l319_319830

theorem rectangle_area_relation (x y : ℝ) (h : x * y = 4) (hx : x > 0) : y = 4 / x := 
sorry

end rectangle_area_relation_l319_319830


namespace range_of_m_l319_319272

/-- The point (m^2, m) is within the planar region defined by x - 3y + 2 > 0. 
    Find the range of m. -/
theorem range_of_m {m : ℝ} : (m^2 - 3 * m + 2 > 0) ↔ (m < 1 ∨ m > 2) := 
by 
  sorry

end range_of_m_l319_319272


namespace problem1_problem2_problem3_l319_319475

noncomputable theory

-- Given conditions
def quadratic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g(x) = g(1-x) ∧ (∃! x', g(x') = -9/8) ∧ g(1) = -1

def defined_f (g : ℝ → ℝ) (m : ℝ) := λ x : ℝ, g(x + 1/2) + m * Real.log x + 9/8

-- Problem 1
theorem problem1 {g : ℝ → ℝ} (hg : quadratic_function g) : 
  ∀ x, g(x) = 1/2 * x^2 - 1/2 * x - 1 := 
sorry

-- Problem 2
theorem problem2 {g : ℝ → ℝ} (hg : quadratic_function g) {m : ℝ} 
  (hf : ∃ x > 0, defined_f g m x ≤ 0) : 
  m ∈ set.Ioo -e 0 ∨ m ∈ set.Ioc 0 ⊤ := 
sorry

-- Problem 3
theorem problem3 {g : ℝ → ℝ} (hg : quadratic_function g) {m : ℝ} 
  (hm : 1 < m ∧ m ≤ Real.exp 1) : 
  ∀ x1 x2 ∈ set.Icc 1 m, 
  let H := λ x, defined_f g m x - (m + 1) * x in 
  abs (H x1 - H x2) < 1 := 
sorry

end problem1_problem2_problem3_l319_319475


namespace largest_three_digit_in_pascal_triangle_l319_319313

-- Define Pascal's triangle and binomial coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem about the first appearance of the number 999 in Pascal's triangle
theorem largest_three_digit_in_pascal_triangle :
  ∃ (n : ℕ), n = 1000 ∧ ∃ (k : ℕ), pascal n k = 999 :=
sorry

end largest_three_digit_in_pascal_triangle_l319_319313


namespace sqrt_defined_l319_319413

-- Defining the problem
theorem sqrt_defined (x : ℝ) : x >= 4 -> ∃ y : ℝ, y = sqrt (x - 4) :=
by
  intro h
  use sqrt (x - 4)
  sorry

end sqrt_defined_l319_319413


namespace find_two_digit_numbers_l319_319682

theorem find_two_digit_numbers : 
  ∃ (AB CD : ℕ), 
  10 ≤ AB ∧ AB ≤ 99 ∧ 
  10 ≤ CD ∧ CD ≤ 99 ∧ 
  AB * CD = 1365 ∧ 
  let A := AB / 10, B := AB % 10, C := CD / 10, D := CD % 10 in 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D :=
sorry

end find_two_digit_numbers_l319_319682


namespace num_friends_received_pebbles_l319_319190

theorem num_friends_received_pebbles:
  ∀ (dozens_of_pebbles friends_pebbles: ℕ), dozens_of_pebbles = 3 → friends_pebbles = 4 → (dozens_of_pebbles * 12) / friends_pebbles = 9 :=
by 
  intros dozens_of_pebbles friends_pebbles h1 h2
  rw [h1, h2]
  calc
    3 * 12 / 4 = 36 / 4 : rfl
    ... = 9             : rfl

end num_friends_received_pebbles_l319_319190


namespace count_integers_between_square_bounds_l319_319146

theorem count_integers_between_square_bounds :
  (n : ℕ) (300 < n^2 ∧ n^2 < 1200) → 17 :=
sorry

end count_integers_between_square_bounds_l319_319146


namespace license_plate_combinations_l319_319028

-- Definitions based on the conditions
def num_letters := 26
def num_digits := 10
def num_positions := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Main theorem statement
theorem license_plate_combinations :
  choose num_letters 2 * (num_letters - 2) * choose num_positions 2 * choose (num_positions - 2) 2 * num_digits * (num_digits - 1) * (num_digits - 2) = 7776000 :=
by
  sorry

end license_plate_combinations_l319_319028


namespace honey_nectar_relationship_l319_319810

-- Definitions representing the conditions
def nectarA_water_content (x : ℝ) := 0.7 * x
def nectarB_water_content (y : ℝ) := 0.5 * y
def final_honey_water_content := 0.3
def evaporation_loss (initial_content : ℝ) := 0.15 * initial_content

-- The system of equations to prove
theorem honey_nectar_relationship (x y : ℝ) :
  (x + y = 1) ∧ (0.595 * x + 0.425 * y = 0.3) :=
sorry

end honey_nectar_relationship_l319_319810


namespace parabola_equation_minimum_FA_FB_product_l319_319910

open Real

theorem parabola_equation :
  ∀ p : ℝ, 0 < p ∧ p ≤ 8 →
  let focus := (p / 2, 0)
  let C_center := (3, 0)
  let radius := 1
  let tangent_length := sqrt 3
  (3 - p / 2) ^ 2 = radius ^ 2 + tangent_length ^ 2 →
  p = 2 →
  (∃ x y : ℝ, y ^ 2 = 4 * x) :=
begin
  intros p h_range focus C_center radius tangent_length h_eq h_p,
  use [1, 2],
  sorry, -- Proof omitted
end

theorem minimum_FA_FB_product :
  ∀ p : ℝ, 0 < p ∧ p ≤ 8 →
  let focus := (p / 2, 0)
  let C_center := (3, 0)
  let radius := 1
  let tangent_line := (m : ℝ) (n : ℝ) → x = n * y + m
  (x - 3) ^ 2 + y ^ 2 = radius ^ 2 →
  x = n * y + m → -- tangent line
  let A := (x1, y1) -- Intersection points
  let B := (x2, y2)
  (x1, y1) ∈ parabola ∧ (x2, y2) ∈ parabola ∧
  (A, B) ∈ tangent_line →
  (m - 3) ^ 2 = 1 + n ^ 2 →
  ∃ minimum_product : ℝ, minimum_product = 9 :=
begin
  intros p h_range focus C_center radius tangent_line h_circle h_tangent_line A B intersections tangent_condition,
  use 9,
  sorry, -- Proof omitted
end

end parabola_equation_minimum_FA_FB_product_l319_319910


namespace morning_snowfall_l319_319169

theorem morning_snowfall (afternoon_snowfall total_snowfall : ℝ) (h₀ : afternoon_snowfall = 0.5) (h₁ : total_snowfall = 0.63):
  total_snowfall - afternoon_snowfall = 0.13 :=
by 
  sorry

end morning_snowfall_l319_319169


namespace find_abc_sum_l319_319296

theorem find_abc_sum (a b c : ℕ) 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 3 * (Real.sqrt (Real.cbrt 5 - Real.cbrt 4)) = Real.cbrt a + Real.cbrt b - Real.cbrt c) :
  a + b + c = 47 := by
sorry

end find_abc_sum_l319_319296


namespace krishan_money_l319_319331

noncomputable def ram_money := 735
def ratio_rg := (7, 17)
def ratio_gk := (7, 17)

theorem krishan_money :
  ∃ K : ℝ, (735 / (17 / 7) = 735 * 17 / 7) ∧ (735 * 17 / 7 = K) :=
begin
  sorry
end

end krishan_money_l319_319331


namespace equation_of_hyperbola_l319_319892

noncomputable def hyperbola_equation (a b e : ℝ) : Prop :=
  ( ∃ (e : ℝ), e = Real.sqrt (1 + b^2 / a^2)
    ∧ ∀ x y, 
        (x = 2 ∧ y = 1 → (x^2 / a^2 - y^2 / b^2 = 1)) 
        ∧ (∀ d, d = ( | (2:ℝ) * b - a | / Real.sqrt(a^2 + b^2) ) →
            d = 1 / e ) )

theorem equation_of_hyperbola :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (hyperbola_equation a b (Real.sqrt (1 + b^2 / a^2)) →
     (a = Real.sqrt 3 ∧ b = Real.sqrt 3 →
      ∀ x y, x^2 / 3 - y^2 / 3 = 1)) := sorry

end equation_of_hyperbola_l319_319892


namespace isosceles_triangle_base_angles_l319_319974

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end isosceles_triangle_base_angles_l319_319974


namespace calculate_total_meters_examined_l319_319021

-- Define the proportion rejected
def proportion_rejected : ℝ := 0.05 / 100

-- Define the number of meters rejected
def meters_rejected : ℝ := 4

-- Define the total meters examined
def total_meters_examined (x : ℝ) : Prop :=
  proportion_rejected * x = meters_rejected

-- State the theorem
theorem calculate_total_meters_examined : ∃ x : ℝ, total_meters_examined x ∧ x = 8000 :=
by 
  sorry

end calculate_total_meters_examined_l319_319021


namespace radical_axis_l319_319589

structure Circle where
  center : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def power (P : Point) (C : Circle) : ℝ :=
  let d := dist P C.center
  d^2 - C.radius^2

theorem radical_axis
  (Γ1 Γ2 : Circle) (A B : Point)
  (h1 : dist A Γ1.center = Γ1.radius) (h2 : dist B Γ1.center = Γ1.radius)
  (h3 : dist A Γ2.center = Γ2.radius) (h4 : dist B Γ2.center = Γ2.radius) : 
  ∃ l : Line, ∀ P : Point,
    (power P Γ1 = power P Γ2) ↔ (P ∈ l ∧ lies_on_line P A B) :=
sorry

end radical_axis_l319_319589


namespace isosceles_triangle_base_angle_l319_319976

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (h1 : vertex_angle = 110) :
  ∃ base_angle : ℝ, base_angle = 35 :=
by
  use 35
  sorry

end isosceles_triangle_base_angle_l319_319976


namespace problem_statement_l319_319424

theorem problem_statement : 
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 27 ∧ n ≡ -3456 [MOD 28] := 
begin
  use 12,
  split,
  {
    exact lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one
     (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one (lt_add_one
     (lt_add_one (lt_add_one zero_lt_one))))))))))))))))))))))))),
  },
  split,
  {
    exact zero_lt_one_add (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add
      (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add (zero_lt_one_add zero_lt_one))))))))))),
  },
  {
    exact modeq_sub (by1 trivial) (by 1 trivial) (by1 trivial) -1 trivial (by1 trivial) (modeq_trans_cong (modeq_add_neg1) trivial) (by1 trivial),
  },
sorry,
end

end problem_statement_l319_319424


namespace smallest_three_digit_palindromic_prime_with_hundreds_digit_2_l319_319657

-- A natural number n is a palindrome if its digit representation reads the same backward.
def is_palindrome (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- A function to check if a given number is prime.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Three-digit numbers have to be in the range 100 to 999.
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- The hundreds digit of the number is 2.
def has_hundreds_digit_2 (n : ℕ) : Prop := n / 100 = 2

/-- Prove that 232 is the smallest three-digit palindromic prime with hundreds digit 2. -/
theorem smallest_three_digit_palindromic_prime_with_hundreds_digit_2 : 
  ∀ n, 
    is_prime n → 
    is_three_digit n →
    is_palindrome n →
    has_hundreds_digit_2 n →
    232 ≤ n :=
by
  sorry

end smallest_three_digit_palindromic_prime_with_hundreds_digit_2_l319_319657


namespace symmetric_point_l319_319576

def point_M : ℝ × ℝ × ℝ := (-2, 4, -3)
def projection_xOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, 0, p.3)
def symmetric_with_respect_to_origin (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, -p.2, -p.3)

theorem symmetric_point :
  symmetric_with_respect_to_origin (projection_xOz point_M) = (2, 0, 3) :=
by 
  sorry

end symmetric_point_l319_319576


namespace cos_150_eq_neg_sqrt3_div_2_l319_319858

theorem cos_150_eq_neg_sqrt3_div_2 : 
  cos (150 * (real.pi / 180)) = - (real.sqrt 3) / 2 := 
by sorry

end cos_150_eq_neg_sqrt3_div_2_l319_319858


namespace find_T_div_30_l319_319045

def is_pretty_30 (m : ℕ) : Prop :=
  (m > 0 ∧ ∃ d : ℕ, d = 30 ∧ ∃ divisors : List ℕ, (∀ x ∈ divisors, x > 0 ∧ m % x = 0) ∧ divisors.length = 30)

def is_divisible_by_30 (m : ℕ) : Prop :=
  m % 30 = 0

def is_less_than_2500 (m : ℕ) : Prop :=
  m < 2500

def T : ℕ :=
  ∑ m in Finset.range 2500, if is_pretty_30 m ∧ is_divisible_by_30 m then m else 0

theorem find_T_div_30 : T / 30 = 0 := by
  sorry

end find_T_div_30_l319_319045


namespace min_value_frac_l319_319924

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) : 
  (2 / x) + (1 / y) ≥ 9 / 2 :=
by
  sorry

end min_value_frac_l319_319924


namespace mod_pow_difference_l319_319393

theorem mod_pow_difference (a b n : ℕ) (h1 : a ≡ 47 [MOD n]) (h2 : b ≡ 22 [MOD n]) (h3 : n = 8) : (a ^ 2023 - b ^ 2023) % n = 1 :=
by
  sorry

end mod_pow_difference_l319_319393


namespace second_solid_not_cube_l319_319822

noncomputable def cube_area (L : ℝ) : ℝ := 6 * L^2

noncomputable def new_solid_area (L : ℝ) : ℝ := 6 * (2 * L)^2

theorem second_solid_not_cube (L : ℝ) (h : cube_area L / new_solid_area L = 0.6) :
  ∀ B : Prop, (B = false → false) → B := 
begin
  sorry
end

end second_solid_not_cube_l319_319822


namespace shoes_problem_proof_l319_319301

def shoes_problem : Prop :=
  let adults := 12 in
  let pairs := list (fin (2 * adults) × fin (2 * adults)) in
  let valid_pairings := 
    ( ∀ k, 1 ≤ k ∧ k < 6 →
    ∀ (pair_col: list (fin (2 * adults) × fin (2 * adults))),
    pair_col.length = k →
    ¬ (∃ (left_shoes right_shoes: finset (fin (2 * adults))),
      left_shoes.card = k ∧ right_shoes.card = k ∧
      ∀ (i ∈ finset.range adults), (left_shoes i, right_shoes i) ∈ pair_col)) in
  let prob := 1 / 12 in
  ∃ m n : ℕ, nat.coprime m n ∧ rat.mk m n = prob ∧ m + n = 13

theorem shoes_problem_proof : shoes_problem :=
sorry

end shoes_problem_proof_l319_319301


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319709

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319709


namespace ellipse_equation_line_HN_fixed_point_l319_319113

open Real

-- Given conditions
def center (E : Type) : Point := (0, 0)
def axes_of_symmetry (E : Type) : Prop := true -- x-axis and y-axis are assumed
def passes_through (E : Type) (p q r : Point) := p ∈ E ∧ q ∈ E ∧ r ∈ E
def equation (E : Type) (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 4) = 1

def point := (ℝ × ℝ) -- Defining a type for point

-- Definitions for other points and line conditions
def A : point := (0, -2)
def B : point := (3/2, -1)
def P : point := (1, -2)

def intersects (E : Type) (P : point) : Point := sorry
def line_passes_through (P Q : point) : Prop := sorry
def M (E : Type) (P : point) : Point := intersects E P
def N (E : Type) (P : point) : Point := intersects E P

def parallel (P : point) : Prop := sorry -- line parallel to x-axis passing through P
def T (A B M: point) : Point := sorry -- intersection of line through M parallel to x-axis and line segment AB
def H (M T: point) : point := sorry -- point satisfying MT = TH

-- Proof Problem 1: Equation of the Ellipse
theorem ellipse_equation : ∀ E, 
  center E = (0, 0) →
  axes_of_symmetry E →
  passes_through E A B →
  ∃ x y, equation E x y :=
by
  intro E
  assume h1 h2 h3
  sorry

-- Proof Problem 2: Line HN passes through fixed point
theorem line_HN_fixed_point : 
  ∀ E, 
  center E = (0, 0) →
  axes_of_symmetry E →
  passes_through E A B →
  ∃ (x : ℝ) (y : ℝ), 
  ∀ (P : point), 
  intersects E P →
  let M := M E P, 
      N := N E P, 
      T := T A B M,
      H := H M T in 
  line_passes_through (1,-2) N
  →
  (HN_line : y = 2 + 2 * sqrt 6/3 * x - 2)
  ∧ line_passes_through (0, -2) H :=
by
  intro E
  assume h1 h2 h3
  sorry

end ellipse_equation_line_HN_fixed_point_l319_319113


namespace derivative_at_zero_l319_319982

def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_zero :
  deriv f 0 = 1 := sorry

end derivative_at_zero_l319_319982


namespace parabola_focus_coordinates_l319_319645

theorem parabola_focus_coordinates :
  ∀ (x : ℝ), 3 * x^2 = y → focus_coordinates (parabola y) = (0, 1 / 12) :=
by
  sorry

end parabola_focus_coordinates_l319_319645


namespace pentagon_square_ratio_l319_319832

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l319_319832


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319766

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319766


namespace top_leftmost_rectangle_is_B_l319_319633

structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

def RectangleA := Rectangle.mk 5 2 8 11
def RectangleB := Rectangle.mk 2 1 4 7
def RectangleC := Rectangle.mk 4 9 6 3
def RectangleD := Rectangle.mk 8 6 5 9
def RectangleE := Rectangle.mk 10 3 9 1
def RectangleF := Rectangle.mk 11 4 10 2

theorem top_leftmost_rectangle_is_B : (TopLeftmostRectangle () = RectangleB) :=
by
  sorry

end top_leftmost_rectangle_is_B_l319_319633


namespace minimum_matches_to_remove_l319_319085

-- Definitions and conditions
def matchsticks := ℕ
def grid_size := 3
def total_matches := 24
def unit_square_matches := 4
def num_unit_squares := grid_size^2

-- Statement of the problem
theorem minimum_matches_to_remove (M : matchsticks) 
  (G : grid_size) (T : total_matches) (U : unit_square_matches) (N : num_unit_squares) :
  G = 3 → T = 24 → U = 4 → N = 9 → 
  (∃ m : matchsticks, m ≤ 5 ∧ ∀ (i : ℕ) (h : i < N), ¬(matchstick_composition i m)) :=
sorry

end minimum_matches_to_remove_l319_319085


namespace G_8_value_l319_319596

-- The conditions as hypotheses in Lean
noncomputable def G : ℝ → ℝ := sorry

axiom G_is_polynomial : ∃ (f : ℝ → ℝ), ∀ x, G(x) = f(x)
axiom G_4_eq_35 : G(4) = 35
axiom main_equation : ∀ x : ℝ, 
  x^2 + 4x + 4 ≠ 0 → 
  G(x+2) ≠ 0 → 
  G(4x) / G(x+2) = 4 - (20x + 24) / (x^2 + 4x + 4)

-- The final goal
theorem G_8_value : G(8) = 163 + 1/3 := by
  sorry

end G_8_value_l319_319596


namespace sin_60_eq_sqrt3_div_2_l319_319798

-- Problem statement translated to Lean
theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l319_319798


namespace base_angle_isosceles_triangle_l319_319970

theorem base_angle_isosceles_triangle (T : Triangle) (h_iso : is_isosceles T) (h_angle : internal_angle T = 110) :
  base_angle T = 35 :=
sorry

end base_angle_isosceles_triangle_l319_319970


namespace count_inverses_modulo_11_l319_319957

theorem count_inverses_modulo_11 : 
  let prime := 11
  let set_of_integers := {x | 1 ≤ x ∧ x ≤ 10}
  card {x ∈ set_of_integers | gcd x prime = 1} = 10 := 
by
  sorry

end count_inverses_modulo_11_l319_319957


namespace cos_eight_arccos_one_fourth_l319_319412

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1 / 4)) = 172546 / 1048576 :=
sorry

end cos_eight_arccos_one_fourth_l319_319412


namespace prove_solution_l319_319670

noncomputable def solution : ℕ :=
  let a := 3
  let b := 1
  let c := 7
  let d := 1
  a + b + c + d

theorem prove_solution : 
  ∃ (x y : ℝ), (x + y = 6) ∧ (3 * x * y = 6) ∧ 
               (∃ a b c d : ℕ, x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d ∧ 
               a + b + c + d = 12) :=
by {
  use [3 + Real.sqrt 7, 3 - Real.sqrt 7],
  split,
  { rw add_comm,
    exact add_self_div_two x,
  },
  use [3, 1, 7, 1],
  ring,
  sorry
}

end prove_solution_l319_319670


namespace numValidPairs_l319_319481

open Finset

-- Defining the universal set
def universalSet : Finset ℕ := {1, 2, 3}

-- Condition: Elements from the universal set
variable {A B : Finset ℕ}
variable (AneqB : A ≠ B)
variable (unionCond : A ∪ B = universalSet)

-- Defining the set of all possible pairs
def possiblePairs (U : Finset ℕ) : Finset (Finset ℕ × Finset ℕ) :=
  U.powerset.product U.powerset

-- Filtering pairs satisfying conditions
def validPairs (U : Finset ℕ) : Finset (Finset ℕ × Finset ℕ) :=
  (possiblePairs U).filter (λ p, p.fst ∪ p.snd = U ∧ p.fst ≠ p.snd)

-- The theorem statement
theorem numValidPairs : (validPairs universalSet).card = 26 := 
  sorry

end numValidPairs_l319_319481


namespace maximum_members_in_dance_troupe_l319_319567

theorem maximum_members_in_dance_troupe (m : ℕ) (h1 : 25 * m % 31 = 7) (h2 : 25 * m < 1300) : 25 * m = 875 :=
by {
  sorry
}

end maximum_members_in_dance_troupe_l319_319567


namespace cos_difference_of_distinct_zeros_l319_319445

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 3)
noncomputable def g (x : ℝ) : ℝ := f x - 1/3

theorem cos_difference_of_distinct_zeros 
  {x₁ x₂ : ℝ} 
  (hx₁ : g x₁ = 0) 
  (hx₂ : g x₂ = 0) 
  (h_distinct : x₁ ≠ x₂) 
  (h_in_interval₁ : 0 ≤ x₁ ∧ x₁ ≤ π) 
  (h_in_interval₂ : 0 ≤ x₂ ∧ x₂ ≤ π) : 
  cos (x₁ - x₂) = 1/3 := 
by 
  sorry

end cos_difference_of_distinct_zeros_l319_319445


namespace unique_k_exists_l319_319961

theorem unique_k_exists (k : ℕ) (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (a^2 + b^2 = k * a * b) ↔ k = 2 := sorry

end unique_k_exists_l319_319961


namespace count_S_equals_3_l319_319203

def number_of_elements (A : Set ℕ) := if A = ∅ then 0 else A.card

def star (A B : Set ℕ) : ℕ :=
  if A.card ≥ B.card then A.card - B.card else B.card - A.card

theorem count_S_equals_3 :
  let A := {1, 2}
  let B (a : ℝ) := {x | (x^2 + a*x)*(x^2 + a*x + 2) = 0}
  (star A (B a) = 1) → (number_of_elements {a | (star A (B a) = 1)} = 3) :=
by
  sorry

end count_S_equals_3_l319_319203


namespace discrim_of_quadratic_eqn_l319_319782

theorem discrim_of_quadratic_eqn : 
  let a := 3
  let b := -2
  let c := -1
  b^2 - 4 * a * c = 16 := 
by
  sorry

end discrim_of_quadratic_eqn_l319_319782


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319698

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319698


namespace largest_digit_for_divisibility_by_6_l319_319310

theorem largest_digit_for_divisibility_by_6 :
  ∃ N : ℕ,
    N ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (57890 + N) % 6 = 0 ∧
    (∀ M : ℕ, M ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 5789 * 10 + M % 6 = 0 → M ≤ N) :=
begin
  sorry
end

end largest_digit_for_divisibility_by_6_l319_319310


namespace f_is_periodic_with_period_4a_l319_319448

variable (f : ℝ → ℝ) (a : ℝ)

theorem f_is_periodic_with_period_4a (h : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_is_periodic_with_period_4a_l319_319448


namespace students_present_l319_319294

variable (X : ℕ) (Y : ℝ)

theorem students_present (h1 : 0 ≤ Y) (h2 : Y ≤ 100) : 
    let present_students := (1 - Y / 100) * X
    present_students = (1 - Y / 100) * X :=
by
  unfold present_students
  sorry

end students_present_l319_319294


namespace pet_store_earnings_l319_319004

theorem pet_store_earnings :
  let kitten_price := 6
  let puppy_price := 5
  let kittens_sold := 2
  let puppies_sold := 1 
  let total_earnings := kittens_sold * kitten_price + puppies_sold * puppy_price
  total_earnings = 17 :=
by
  sorry

end pet_store_earnings_l319_319004


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319700

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319700


namespace ratio_c_d_l319_319398

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x + 5 * y = c) (h2 : 8 * y - 10 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = 1 / 2 :=
by
  sorry

end ratio_c_d_l319_319398


namespace sum_q_p_evaluation_l319_319217

def p (x : Int) : Int := x^2 - 3
def q (x : Int) : Int := x - 2

def T : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

noncomputable def f (x : Int) : Int := q (p x)

noncomputable def sum_f_T : Int := List.sum (List.map f T)

theorem sum_q_p_evaluation :
  sum_f_T = 15 :=
by
  sorry

end sum_q_p_evaluation_l319_319217


namespace deceased_member_income_l319_319256

theorem deceased_member_income (a b c d : ℝ)
    (h1 : a = 735) 
    (h2 : b = 650)
    (h3 : c = 4 * 735)
    (h4 : d = 3 * 650) :
    c - d = 990 := by
  sorry

end deceased_member_income_l319_319256


namespace last_person_remaining_l319_319552

def is_eliminated (count : Nat) : Bool :=
  count % 8 = 0 \/ Nat.digits 10 count |> List.any (λ d => d = 8)

theorem last_person_remaining :
  ∃ last_person : String, last_person = "Leo" ∧
    let initial_positions := ["Gary", "Hank", "Ida", "June", "Kim", "Leo", "Moe"]
    ( ∀ (elimination_order : List String), elimination_order ⊆ initial_positions ∧
      elimination_order.erase_dup.length = 6 ) ∧
    let remaining_person := initial_positions.diff elimination_order.erase_dup
    remaining_person = ["Leo"] :=
sorry

end last_person_remaining_l319_319552


namespace Harkamal_purchased_mangoes_l319_319500

variable (m : ℕ)

theorem Harkamal_purchased_mangoes :
  let cost_of_grapes := 8 * 80 in
  let total_amount := 1135 in
  let total_cost := cost_of_grapes + 55 * m in
  total_cost = total_amount → m = 9 :=
by
  intros
  have cost_of_grapes_eq : cost_of_grapes = 640 := by norm_num
  have total_cost_eq : total_cost = 640 + 55 * m := rfl
  have eq1 : 640 + 55 * m = total_amount := by rw [total_cost_eq, cost_of_grapes_eq]; exact ‹_›
  have eq2 : 55 * m = 495 := by linarith
  have eq3 : m = 9 := by {
    have h_div : 495 / 55 = 9 := by norm_num,
    exact nat.eq_of_mul_eq_mul_right (by norm_num) ‹_›,
  }
  exact eq3

end Harkamal_purchased_mangoes_l319_319500


namespace rook_placement_5x5_l319_319527

theorem rook_placement_5x5 :
  ∀ (board : Fin 5 → Fin 5) (distinct : Function.Injective board),
  ∃ (ways : Nat), ways = 120 := by
  sorry

end rook_placement_5x5_l319_319527


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319741

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319741


namespace negative_values_count_l319_319904

theorem negative_values_count (n : ℕ) : (n < 13) → (n^2 < 150) → ∃ (k : ℕ), k = 12 :=
by
  sorry

end negative_values_count_l319_319904


namespace probability_two_boys_l319_319572

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_pairs : ℕ := Nat.choose number_of_students 2
def boys_pairs : ℕ := Nat.choose number_of_boys 2

theorem probability_two_boys :
  number_of_students = 5 →
  number_of_boys = 2 →
  number_of_girls = 3 →
  (boys_pairs : ℝ) / (total_pairs : ℝ) = 1 / 10 :=
by
  sorry

end probability_two_boys_l319_319572


namespace point_on_x_axis_l319_319849

theorem point_on_x_axis 
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) (p4 : ℝ × ℝ)
  (h1 : p1 = (-2, -3))
  (h2 : p2 = (-3, 0))
  (h3 : p3 = (-1, 2))
  (h4 : p4 = (0, 3)) :
  (∃ (p : ℝ × ℝ), (p = p2) ∧ (p.snd = 0)) := 
begin
  use p2,
  split,
  { exact h2 },
  { exact h2.symm ▸ rfl }
end

end point_on_x_axis_l319_319849


namespace non_intersecting_quadrilaterals_exists_l319_319559

theorem non_intersecting_quadrilaterals_exists :
  ∃ (quads : Finset (Finset (Fin 4000))), 
    quads.card = 1000 ∧ 
    ∀ q ∈ quads, q.card = 4 ∧ 
    ∀ p1 p2 p3 ∈ q, ¬Collinear ℝ ({p1, p2, p3} : Set ℝ) :=
begin
  sorry
end

end non_intersecting_quadrilaterals_exists_l319_319559


namespace tangent_line_at_origin_l319_319647

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp x + 2 * x - 1

def tangent_line (x₀ y₀ : ℝ) (k : ℝ) (x : ℝ) := y₀ + k * (x - x₀)

theorem tangent_line_at_origin : 
  tangent_line 0 (-1) 3 = λ x => 3 * x - 1 :=
by
  sorry

end tangent_line_at_origin_l319_319647


namespace smallest_integer_with_odd_and_even_factors_l319_319604

theorem smallest_integer_with_odd_and_even_factors (n : ℕ) :
  ∃ (n : ℕ), 
  n = 900 ∧ 
  (∃ (a A : ℕ), n = 2^a * A ∧ 
                       (∀ x : ℕ, (x ∣ n) → (odd x) ↔ (x ∣ A)) ∧
                       (∃ A_divisors : ℕ, A_divisors = 9 ∧ 
                                          (∀ d : ℕ, (d ∣ A) ↔ (∃ b : ℕ, d = p^b ∧ p.prime))) ∧ 
                       (∃ n_divisors : ℕ, n_divisors = 27 ∧ 
                                          (∃ e : ℕ, e = 18))) :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_factors_l319_319604


namespace solve_for_x_l319_319942

theorem solve_for_x :
  ∃ x : ℤ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27)) = 113 ∧ x = 40 := 
by
  sorry

end solve_for_x_l319_319942


namespace evaluate_func_l319_319944

def func (x: ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^(x-1)
  else func (x-1)

theorem evaluate_func : func (2 + Real.log2 3) = 8 / 3 := 
  sorry

end evaluate_func_l319_319944


namespace part_a_part_b_l319_319335

theorem part_a (a : ℚ) (x : ℝ) (h1 : a > 1 ∨ a < 0) (h2 : x > 0) (h3 : x ≠ 1) : 
  x^a - a*x + a - 1 > 0 := 
sorry

theorem part_b (a : ℚ) (x : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : x > 0) (h3 : x ≠ 1) : 
  x^a - a*x + a - 1 < 0 := 
sorry

end part_a_part_b_l319_319335


namespace min_value_AG_l319_319930

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def is_centroid (G A B C : V) : Prop :=
  G = (A + B + C) / 3

noncomputable def angle_A (A B C : V) : ℝ :=
  real.arccos ((B - A) • (C - A) / ((B - A).norm * (C - A).norm))

theorem min_value_AG (A B C G : V)
  (h1 : is_centroid G A B C)
  (h2 : angle_A A B C = 2 * real.pi / 3)
  (h3 : (B - A) • (C - A) = -2) :
  (G - A).norm = 2 / 3 :=
  sorry

end min_value_AG_l319_319930


namespace range_of_logarithmic_function_l319_319660

theorem range_of_logarithmic_function :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 2 ∧ y = 1 + real.log x / real.log 2) ↔ y ∈ set.Ici 2 := by
  sorry

end range_of_logarithmic_function_l319_319660


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319763

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319763


namespace problem_statement_l319_319911

noncomputable theory

-- Define the parabola
def parabola (p : ℝ) : Prop :=
  p = 4

-- Define the line equation and fixed point
def line_eq (t k x y : ℝ) : Prop :=
  y = k * x + t

-- Define the vector dot product condition
def vector_dot_product (x₁ x₂ y₁ y₂ t : ℝ) : Prop :=
  (x₁ + x₂ = 8) ∧ (x₁ * x₂ = -8 * t) ∧ (y₁ * y₂ = (x₁ + y₁) * (x₂ + y₂)) ∧ ( (x₁ * x₂) + (y₁ * y₂) = t^2 - 8 * t )

-- Define the minimum length condition
def min_length_AB (λ x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  (x₁ = -λ * x₂) ∧ ((1 - y₁) = λ * (y₂ - 1)) ∧ 
  ( λ > 0 ) ∧ ( abs (( x₁ - x₂)^2 + (y₁ - y₂)^2 )) = 4 * real.sqrt 2

-- Main problem statement
theorem problem_statement (p k t λ x₁ x₂ y₁ y₂ : ℝ) : parabola p → line_eq t k x₁ y₁ →
  vector_dot_product x₁ x₂ y₁ y₂ t → (t = 1) →
  min_length_AB λ x₁ x₂ y₁ y₂ := 
sorry

end problem_statement_l319_319911


namespace symmetry_center_l319_319936

/-- 
Given:
1. The symmetry center of the graph of the function y = 1/x is (0, 0).
2. The symmetry center of the graph of the function y = 1/x + 1/(x+1) is (-1/2, 0).
3. The symmetry center of the graph of the function y = 1/x + 1/(x+1) + 1/(x+2) is (-1, 0).

Prove:
The symmetry center of the graph of the function y = 1/x + 1/(x+1) + ... + 1/(x+n) is (-n/2, 0).
-/
theorem symmetry_center (n : ℕ) : 
  symmetry_center (λ x : ℝ, ∑ i in finset.range (n + 1), 1 / (x + i)) = (-n / 2, 0) := 
sorry

end symmetry_center_l319_319936


namespace evaluate_polynomial_l319_319795

def polynomial := 
  (√2017 * x - √2027)^2017 = a₁ * x^2017 + a₂ * x^2016 + ⋯ + a₂₀₁₇ * x + a₂₀₁₈

theorem evaluate_polynomial :
  (a₁ + a₃ + ⋯ + a₂₀₁₇)^2 - (a₂ + a₄ + ⋯ + a₂₀₁₈)^2 = -10^2017 := by
  sorry

end evaluate_polynomial_l319_319795


namespace ceil_square_range_count_l319_319515

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l319_319515


namespace measure_of_angle_Q_in_hexagon_l319_319873

theorem measure_of_angle_Q_in_hexagon :
  ∀ (Q : ℝ),
    (∃ (angles : List ℝ),
      angles = [134, 108, 122, 99, 87] ∧ angles.sum = 550) →
    180 * (6 - 2) - (134 + 108 + 122 + 99 + 87) = 170 → Q = 170 := by
  sorry

end measure_of_angle_Q_in_hexagon_l319_319873


namespace geometric_series_sum_l319_319391

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 7
  (a * (1 - r^n) / (1 - r)) = (16383 / 49152) :=
by
  sorry

end geometric_series_sum_l319_319391


namespace zachary_more_pushups_l319_319786

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := 44

theorem zachary_more_pushups : zachary_pushups - david_pushups = 7 := by
  sorry

end zachary_more_pushups_l319_319786


namespace find_x_given_total_area_l319_319037

theorem find_x_given_total_area :
  ∃ x : ℝ, (16 * x^2 + 36 * x^2 + 6 * x^2 + 3 * x^2 = 1100) ∧ (x = Real.sqrt (1100 / 61)) :=
sorry

end find_x_given_total_area_l319_319037


namespace determine_a_l319_319408

theorem determine_a (a p q : ℚ) (h1 : p^2 = a) (h2 : 2 * p * q = 28) (h3 : q^2 = 9) : a = 196 / 9 :=
by
  sorry

end determine_a_l319_319408


namespace prime_equals_two_l319_319452

theorem prime_equals_two 
  (p : ℕ) (hp : p.prime) 
  (n u v : ℕ) (huv_pos : 0 < u ∧ 0 < v ∧ 0 < n) 
  (h_divisors : (∀ d ∈ divisors n, d > 0) ∧ (divisors_count n = p ^ u)) 
  (h_sum_divisors : (∑ d in divisors n, d) = p ^ v) :
  p = 2 :=
sorry

end prime_equals_two_l319_319452


namespace problem_statement_l319_319477

variable {a x y : ℝ}

theorem problem_statement (hx : 0 < a) (ha : a < 1) (h : a^x < a^y) : x^3 > y^3 :=
sorry

end problem_statement_l319_319477


namespace trig_identity_l319_319121

-- Define the conditions
variable (α : ℝ)

-- State the required condition for α
def conditions (α : ℝ) : Prop := 
  ∀ k : ℤ, α ≠ (π / 2) + k * π

-- Define the trigonometric equation
def trig_expression (α : ℝ) :=
  ((tan α + sec α) * (cos α - cot α)) / ((cos α + cot α) * (tan α - sec α))

-- Prove the equation is equal to 1 under the given conditions
theorem trig_identity : conditions α → trig_expression α = 1 :=
by
  intro h
  sorry

end trig_identity_l319_319121


namespace min_abs_a_sub_b_l319_319528

theorem min_abs_a_sub_b (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b - 4 * a + 5 * b = 170) :
  ∃ a b, abs (a - b) = 14 :=
by
  sorry

end min_abs_a_sub_b_l319_319528


namespace part1_part2_l319_319134

open Set

namespace ProofProblem

variable (m : ℝ)

def A (m : ℝ) := {x : ℝ | 0 < x - m ∧ x - m < 3}
def B := {x : ℝ | x ≤ 0 ∨ x ≥ 3}

theorem part1 : (A 1 ∩ B) = {x : ℝ | 3 ≤ x ∧ x < 4} := by
  sorry

theorem part2 : (∀ m, (A m ∪ B) = B ↔ (m ≥ 3 ∨ m ≤ -3)) := by
  sorry

end ProofProblem

end part1_part2_l319_319134


namespace probability_sum_odd_l319_319881

section

variable {α : Type} {β : Type}

-- Define the probability measure
noncomputable def prob {Ω : Type} (ω: Ω → ℕ) (p : Ω): ℚ :=
(p.card (set_of ω) / (finset.card p)).to_rat

-- Define the sets of events
def wheel_A : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def even_num (n : ℕ) : Prop := n % 2 = 0
def odd_num (n : ℕ) : Prop := ¬ even_num n

def wheel_B : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def even_numbers_B : finset ℕ := {2, 4, 6, 8, 10, 12}
def odd_numbers_B : finset ℕ := {1, 3, 5, 7}

-- Proving the probability calculation
theorem probability_sum_odd :
  prob (even_num) wheel_A = 1 / 2 →
  prob (odd_num) wheel_A = 1 / 2 →
  prob (even_num) wheel_B = 3 / 4 →
  prob (odd_num) wheel_B = 1 / 4 →
  prob (λ n, even_num n ∨ odd_num n) (wheel_A ×ˢ wheel_B) = 1 / 2 :=
by
  sorry

end

end probability_sum_odd_l319_319881


namespace mary_bought_48_cards_l319_319609

variable (M T F C B : ℕ)

theorem mary_bought_48_cards
  (h1 : M = 18)
  (h2 : T = 8)
  (h3 : F = 26)
  (h4 : C = 84) :
  B = C - (M - T + F) :=
by
  -- Proof would go here
  sorry

end mary_bought_48_cards_l319_319609


namespace count_odd_functions_l319_319483

theorem count_odd_functions :
  let f1 := λ x : ℝ, 2 * x^3 + x^(1/3)
  let f2 := λ x : ℝ, if x < 0 then 2 / x else 0
  let f3 := λ x : ℝ, x + 3
  let f4 := λ x : ℝ, if x = 0 then 0 else (x^2 - 2) / x
  (∃ d, 
    d ∈ {f1, f2, f3, f4} ∧ 
    (∀ x : ℝ, d(-x) = -d(x)) ∧ 
    (∀ x : ℝ, ∃ y : ℝ, d(y) ≠ 0 ∨ y = 0)) →
  (2 = {f1, f2, f3, f4}.filter (λ f, ∀ x : ℝ, f(-x) = -f(x))).length := by
  sorry

end count_odd_functions_l319_319483


namespace factorial_tail_count_l319_319047

def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125

theorem factorial_tail_count : 
  let count_non_factorial_tails := 5000 - (19975 / 5) 
  in count_non_factorial_tails = 1005 := 
by {
  -- Formalizing the solution to show the equivalence 
  -- between the theoretical result and the given answer.
  sorry 
}

end factorial_tail_count_l319_319047


namespace simplify_expression_l319_319902

theorem simplify_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by
  sorry

end simplify_expression_l319_319902


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319704

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319704


namespace smallest_diff_EF_DE_l319_319676

theorem smallest_diff_EF_DE : 
  ∃ (DE EF DF : ℕ), 
  DE < EF ∧ EF ≤ DF ∧ DE + EF + DF = 3005 ∧ (EF - DE) = 1 :=
begin
  -- Finding the integers DE, EF, DF ensures the conditions of the problem
  let DE := 1001,
  let EF := 1002,
  let DF := 1002,
  use [DE, EF, DF],
  split,
  { exact nat.lt_succ_self DE },
  split,
  { exact le_refl EF },
  split,
  { norm_num },
  { norm_num }
end

end smallest_diff_EF_DE_l319_319676


namespace origin_moves_distance_under_dilation_l319_319355

noncomputable def dilation_distance_move (O A A' : ℝ × ℝ) (r1 r2 : ℝ) (k : ℝ) :=
  let dA := (A.1 - A'.1)^2 + (A.2 - A'.2)^2
  let dA2 := (dA : ℝ^(1/2))
  let dO := (O.1 + O.2)^2
  let dO2 := (dO : ℝ^(1/2))
  (dO2 * k) - dO2

theorem origin_moves_distance_under_dilation :
  let O := (0, 0)
  let A := (2, 2)
  let A' := (5, 6)
  let r1 := 2
  let r2 := 3
  let k := r2 / r1
  dilation_distance_move O A A' r1 r2 k = sqrt 13 :=
by {
  sorry
}

end origin_moves_distance_under_dilation_l319_319355


namespace reflect_function_l319_319983

theorem reflect_function :
  let f : ℝ → ℝ := λ x, -2 * x + 1
  let reflected_f : ℝ → ℝ := λ x, 2 * x + 9
  ∀ x : ℝ, reflected_f (2 * (-2) - x) = f x
:= sorry

end reflect_function_l319_319983


namespace greatest_multiple_of_5_and_6_under_1000_l319_319736

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l319_319736


namespace smallest_positive_period_intervals_of_monotonic_decrease_translated_function_max_value_in_interval_min_value_in_interval_l319_319126

def f (x : ℝ) : ℝ := (Real.cos x)^2 + (Real.cos (x - π / 3))^2 - 1

theorem smallest_positive_period : ∀ x ∈ Set.Icc (-π/4) (π/3), f x = f (x + π) := sorry

theorem intervals_of_monotonic_decrease : 
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + π / 6) (k * π + 2 * π / 3), 
  f x ≤ f x := sorry

def g (x : ℝ) : ℝ := - (1 / 2) * Real.cos (2 * x)

theorem translated_function :
  ∀ x : ℝ, g x = f (x - π / 3) := sorry

theorem max_value_in_interval :
  ∀ x ∈ Set.Icc (-π/4) (π/3), f x ≤ 1 / 2 := sorry

theorem min_value_in_interval :
  ∀ x ∈ Set.Icc (-π/4) (π/3), f x ≥ - (Real.sqrt 3) / 4 := sorry

end smallest_positive_period_intervals_of_monotonic_decrease_translated_function_max_value_in_interval_min_value_in_interval_l319_319126


namespace prime_sum_probability_l319_319324

theorem prime_sum_probability :
  let sides := 6 in
  let outcomes := sides * sides in
  let prime_sum_combinations := 15 in
  let probability := prime_sum_combinations / outcomes in
  probability = 5 / 12 := by
  sorry

end prime_sum_probability_l319_319324


namespace num_not_factorial_tails_lt_5000_l319_319048

-- Definitions based on conditions from step a)
def f (m : ℕ) : ℕ :=
  ∑ k in (finset.range (nat.log 5 m).succ), m / 5^k

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, f m = n

def num_not_factorial_tails (bound : ℕ) : ℕ :=
  bound - (finset.range bound).filter (λ n, ∃ m, f m = n).card

-- The theorem as per conditions and the correct answer from steps a) and b)
theorem num_not_factorial_tails_lt_5000 : num_not_factorial_tails 5000 = 3751 :=
sorry

end num_not_factorial_tails_lt_5000_l319_319048


namespace theta_positive_l319_319603

noncomputable
def xi : Type := sorry -- Assuming xi as a random variable type
variables {E : xi → ℝ} {θ : ℝ}

-- Defining conditions as hypotheses
axiom E_xi_neg : E xi < 0
axiom MGF_theta : E (λ x, Real.exp (θ * x)) = 1
axiom theta_nonzero : θ ≠ 0

theorem theta_positive : θ > 0 :=
sorry

end theta_positive_l319_319603


namespace solution_set_of_inequality_l319_319288

-- We define the inequality condition
def inequality (x : ℝ) : Prop := (x - 3) * (x + 2) < 0

-- We need to state that for all real numbers x, iff x satisfies the inequality,
-- then x must be within the interval (-2, 3).
theorem solution_set_of_inequality :
  ∀ x : ℝ, inequality x ↔ -2 < x ∧ x < 3 :=
by {
   sorry
}

end solution_set_of_inequality_l319_319288


namespace integer_part_one_minus_sqrt_ten_l319_319402

def intPart (x : ℝ) : ℤ := Int.floor x

theorem integer_part_one_minus_sqrt_ten : intPart (1 - Real.sqrt 10) = -3 := by
  have h1 : 9 < 10 := by norm_num
  have h2 : 10 < 16 := by norm_num
  have h3 : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by 
    split
    all_goals apply Real.sqrt_lt; norm_num
  have h4 : -4 < -Real.sqrt 10 ∧ -Real.sqrt 10 < -3 := by
    split
    all_goals linarith [h3]
  have h5 : -3 < 1 - Real.sqrt 10 ∧ 1 - Real.sqrt 10 < -2 := by
    split
    all_goals linarith [h4]
  sorry

end integer_part_one_minus_sqrt_ten_l319_319402


namespace problem_1_problem_2_problem_3_l319_319887

def α_smallest_angle (α : ℝ) (t : Triangle) : Prop := t.smallestAngle = α
def is_not_greater_than_60 (α : ℝ) : Prop := ¬ (α > 60)
def is_isosceles_right_triangle (t : Triangle) : Prop := t.has_angle 90 ∧ t.has_angle 45 ∧ t.is_isosceles
def has_angle_90_and_45 (t : Triangle) : Prop := t.has_angle 90 ∧ t.has_angle 45 ∧ t.is_right_triangle
def is_equilateral_or_right_triangle (t : Triangle) : Prop := t.has_angle 60 → (t.is_equilateral ∨ t.is_right_triangle)

theorem problem_1 (α : ℝ) (t : Triangle) (h : α_smallest_angle α t) : is_not_greater_than_60 α := by 
  sorry

theorem problem_2 (t : Triangle) (h1 : is_isosceles_right_triangle t) : has_angle_90_and_45 t := by
  sorry

theorem problem_3 (t : Triangle) (h : t.has_angle 60) : is_equilateral_or_right_triangle t := by
  sorry

end problem_1_problem_2_problem_3_l319_319887


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319753

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319753


namespace value_of_y_l319_319162

theorem value_of_y (x y z : ℕ) (h_positive_x : 0 < x) (h_positive_y : 0 < y) (h_positive_z : 0 < z)
    (h_sum : x + y + z = 37) (h_eq : 4 * x = 6 * z) : y = 32 :=
sorry

end value_of_y_l319_319162


namespace probability_adjacent_l319_319526

open Finset

-- Given condition
def total_permutations (s : Finset (Fin 3)) : ℕ :=
  card (s.permutations)

-- Given condition
def adjacent_permutations (s : Finset (Fin 2)) (s' : Finset (Fin 1)) : ℕ :=
  card (s.permutations) * card (s'.permutations)

-- Theorem statement
theorem probability_adjacent :
  let s := univ : Finset (Fin 3),
      s_ab := univ : Finset (Fin 2),
      s_c := singleton ⟨0⟩ : Finset (Fin 1) in
  (adjacent_permutations s_ab s_c : ℚ) / (total_permutations s : ℚ) = 2 / 3 :=
by
  sorry

end probability_adjacent_l319_319526


namespace part_I_part_II_l319_319127

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 - 3 * x + 1

theorem part_I (x : ℝ) (h : x = -1 ∨ x = 3) :
  (x = -1 → f x = 8 / 3) ∧ (x = 3 → f x = -8) := sorry

theorem part_II (x₀ : ℝ) (h : x₀ = 0 ∨ x₀ = 3 / 2) :
  (x₀ = 0 → ∀ x y : ℝ, y = f 0 + (f' 0) * (x - 0) → 3 * x + y - 1 = 0) ∧ 
  (x₀ = 3 / 2 → ∀ x y : ℝ, y = f (3 / 2) + (f' (3 / 2)) * (x - 3 / 2) → 15 * x + 4 * y - 4 = 0) := sorry

noncomputable def f' (x : ℝ) : ℝ := x^2 - 2 * x - 3

end part_I_part_II_l319_319127


namespace jack_more_emails_in_morning_l319_319579

theorem jack_more_emails_in_morning : ∀ (morning_emails afternoon_emails : ℕ), 
  morning_emails = 6 → 
  afternoon_emails = 2 → 
  morning_emails - afternoon_emails = 4 :=
begin
  intros morning_emails afternoon_emails h_morning h_afternoon,
  rw [h_morning, h_afternoon],
  exact nat.sub_self 2,
end

end jack_more_emails_in_morning_l319_319579


namespace calculate_expression_value_l319_319856

theorem calculate_expression_value (x : ℕ) (h : x = 3) :
  (∏ k in Finset.filter (fun k => k % 2 = 1) (Finset.range 18), x^k) / 
  (∏ k in Finset.filter (fun k => k % 2 = 0) (Finset.range 19), x^k) = 3^(-9) :=
by {
  rw h,
  sorry
}

end calculate_expression_value_l319_319856


namespace minimum_value_xyz_l319_319593

theorem minimum_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  ∃ m : ℝ, m = 16 ∧ ∀ w, w = (x + y) / (x * y * z) → w ≥ m :=
by
  sorry

end minimum_value_xyz_l319_319593


namespace find_carbon_atoms_l319_319819

variable (n : ℕ)
variable (molecular_weight : ℝ := 124.0)
variable (weight_Cu : ℝ := 63.55)
variable (weight_C : ℝ := 12.01)
variable (weight_O : ℝ := 16.00)
variable (num_Cu : ℕ := 1)
variable (num_O : ℕ := 3)

theorem find_carbon_atoms 
  (h : molecular_weight = (num_Cu * weight_Cu) + (n * weight_C) + (num_O * weight_O)) : 
  n = 1 :=
sorry

end find_carbon_atoms_l319_319819


namespace triangle_angle_y_value_l319_319184

theorem triangle_angle_y_value :
  ∀ (A B C D E : Type)
    [triangle ABC]
    [right_triangle CDE]
    (angle_ABC angle_BAC : ℝ),
    angle_ABC = 70 ∧ angle_BAC = 50 →
    y = 30 :=
by
  -- Assume the vertices of the triangles and the angles
  intros A B C D E hABC hCDE angle_ABC angle_BAC h,
  have h1 : ∠ ABC = 70 := h.1,
  have h2 : ∠ BAC = 50 := h.2,
  sorry

end triangle_angle_y_value_l319_319184


namespace first_day_exceed_2000_paperclips_l319_319191

def geometric {α : Type*} [OrderedSemiring α] (a r : α) (k : ℕ) : α :=
  a * r ^ k

theorem first_day_exceed_2000_paperclips :
  ∃ k : ℕ, geometric 4 3 k > 2000 ∧ k = 6 :=
by
  have lemma1 : ∀ (a r : ℕ), a * r = a * r := λ a r, rfl
  use 6
  apply And.intro
  · apply lemma1 4 ((3 : ℕ) ^ 6)
  · sorry

end first_day_exceed_2000_paperclips_l319_319191


namespace find_p_Y_ge_2_l319_319131

noncomputable def binomial_probability (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
(nat.choose n k) * p ^ k * (1 - p) ^ (n - k)

noncomputable def binomial_cdf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
∑ i in finset.range (k + 1), binomial_probability n p i

noncomputable def binomial_ge_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
1 - binomial_cdf n p (k - 1)

-- Definition of given parameters
def X_distribution := 2
def Y_distribution := 4
def p_prob : ℚ := sorry  -- Placeholder for \( p \) which will be verified during proof

-- Given condition
def condition := (binomial_ge_prob X_distribution p_prob 1) = (5 / 9)

-- Proof target
theorem find_p_Y_ge_2 : condition → binomial_ge_prob Y_distribution p_prob 2 = 11 / 27 := by
  sorry

end find_p_Y_ge_2_l319_319131


namespace possible_values_ceil_square_l319_319509

noncomputable def num_possible_values (x : ℝ) (hx : ⌈x⌉ = 12) : ℕ := 23

theorem possible_values_ceil_square (x : ℝ) (hx : ⌈x⌉ = 12) :
  let n := num_possible_values x hx in n = 23 :=
by
  let n := num_possible_values x hx
  exact rfl

end possible_values_ceil_square_l319_319509


namespace min_run_distance_l319_319558

theorem min_run_distance : 
  let A := (0 : ℝ, 200 : ℝ)
  let B := (800 : ℝ, 400 : ℝ)
  ∃ C : ℝ × ℝ, C.1 ≥ 0 ∧ C.1 ≤ 800 ∧ C.2 = 0 ∧
  cdist A C + cdist C B = 1000 :=
by 
  sorry

end min_run_distance_l319_319558


namespace sum_of_abcd_is_1_l319_319079

theorem sum_of_abcd_is_1
  (a b c d : ℤ)
  (h1 : (x^2 + a*x + b)*(x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) :
  a + b + c + d = 1 := by
  sorry

end sum_of_abcd_is_1_l319_319079


namespace num_cubes_with_two_icing_sides_l319_319394

-- Definition of the problem conditions
def cube_3x3x3 : set (ℕ × ℕ × ℕ) := 
  { (x, y, z) | 0 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 3 ∧ 0 ≤ z ∧ z < 3 }

def top_face : set (ℕ × ℕ × ℕ) := 
  { (x, y, z) | 0 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 3 ∧ z = 2 }

def front_face : set (ℕ × ℕ × ℕ) := 
  { (x, y, z) | x = 2 ∧ 0 ≤ y ∧ y < 3 ∧ 0 ≤ z ∧ z < 3 }

-- Definition of cubes with icing on exactly two sides
def icing_on_two_sides : set (ℕ × ℕ × ℕ) :=
  { (x, y, z) | (x = 2 ∧ y = 0 ∧ 1 ≤ z ∧ z < 3) ∨ 
               (x = 2 ∧ y = 1 ∧ 1 ≤ z ∧ z < 3) }

-- Main theorem statement
theorem num_cubes_with_two_icing_sides : 
  ∃ (count : ℕ), count = 2 ∧ count = finset.card (finset.filter (λ c, c ∈ icing_on_two_sides) (finset.univ : finset (ℕ × ℕ × ℕ))) :=
begin
  sorry
end

end num_cubes_with_two_icing_sides_l319_319394


namespace haley_seeds_l319_319499

theorem haley_seeds (total_seeds seeds_big_garden total_small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : seeds_big_garden = 35)
  (h3 : total_small_gardens = 7)
  (h4 : total_seeds - seeds_big_garden = 21)
  (h5 : 21 / total_small_gardens = seeds_per_small_garden) :
  seeds_per_small_garden = 3 :=
by sorry

end haley_seeds_l319_319499


namespace quadratic_equation_unique_solution_l319_319875

theorem quadratic_equation_unique_solution (p : ℚ) :
  (∃ x : ℚ, 3 * x^2 - 7 * x + p = 0) ∧ 
  ∀ y : ℚ, 3 * y^2 -7 * y + p ≠ 0 → ∀ z : ℚ, 3 * z^2 - 7 * z + p = 0 → y = z ↔ 
  p = 49 / 12 :=
by
  sorry

end quadratic_equation_unique_solution_l319_319875


namespace sequence_is_arithmetic_l319_319916

-- Define a_n as a sequence in terms of n, where the formula is given.
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Theorem stating that the sequence is arithmetic with a common difference of 2.
theorem sequence_is_arithmetic : ∀ (n : ℕ), n > 0 → (a_n n) - (a_n (n - 1)) = 2 :=
by
  sorry

end sequence_is_arithmetic_l319_319916


namespace no_savings_if_purchased_together_l319_319013

def window_price : ℕ := 120

def free_windows (purchased_windows : ℕ) : ℕ :=
  (purchased_windows / 10) * 2

def total_cost (windows_needed : ℕ) : ℕ :=
  (windows_needed - free_windows windows_needed) * window_price

def separate_cost : ℕ :=
  total_cost 9 + total_cost 11 + total_cost 10

def joint_cost : ℕ :=
  total_cost 30

theorem no_savings_if_purchased_together :
  separate_cost = joint_cost :=
by
  -- Proof will be provided here, currently skipped.
  sorry

end no_savings_if_purchased_together_l319_319013


namespace count_S_equals_3_l319_319204

def number_of_elements (A : Set ℕ) := if A = ∅ then 0 else A.card

def star (A B : Set ℕ) : ℕ :=
  if A.card ≥ B.card then A.card - B.card else B.card - A.card

theorem count_S_equals_3 :
  let A := {1, 2}
  let B (a : ℝ) := {x | (x^2 + a*x)*(x^2 + a*x + 2) = 0}
  (star A (B a) = 1) → (number_of_elements {a | (star A (B a) = 1)} = 3) :=
by
  sorry

end count_S_equals_3_l319_319204


namespace scientific_notation_example_l319_319878

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 218000000 = a * 10 ^ n ∧ a = 2.18 ∧ n = 8 :=
by {
  -- statement of the problem conditions
  sorry
}

end scientific_notation_example_l319_319878


namespace positive_diff_in_x_coords_at_y_10_y_coord_diff_at_x_5_l319_319086

-- Definitions
def line_p := {x : ℝ // true}
def line_q := {x : ℝ // true }

-- Points through which lines pass
def p1 := (0, 3 : ℝ)
def p2 := (4, 0 : ℝ)
def q1 := (0, 6 : ℝ)
def q2 := (6, 0 : ℝ)

-- Lean statement for the first proof problem
theorem positive_diff_in_x_coords_at_y_10 : 
  let y := 10 in
  let xp := 28 / 3 in -- x-coordinate for line p when y = 10
  let xq := -4 in   -- x-coordinate for line q when y = 10
  |xp - xq| = 40 / 3 := 
by
  sorry

-- Lean statement for the second proof problem
theorem y_coord_diff_at_x_5 : 
  let x := 5 in
  let y1 := -3 / 4 * x + 3 in -- y-coordinate for line p when x = 5
  let y2 := -x + 6 in         -- y-coordinate for line q when x = 5
  |y1 - y2| = 7 / 4 := 
by
  sorry

end positive_diff_in_x_coords_at_y_10_y_coord_diff_at_x_5_l319_319086


namespace diff_set_is_B_l319_319018

def A := { x : ℤ | x = 1 }
def B := { x : ℤ | x^2 = 1 }
def C := { 1 }
def D := { y : ℤ | (y - 1)^2 = 0 }

theorem diff_set_is_B : (A = {1}) ∧ (B = { -1, 1 }) ∧ (C = {1}) ∧ (D = {1}) → (B ≠ A ∧ B ≠ C ∧ B ≠ D) :=
by {
    intro h,
    sorry
}

end diff_set_is_B_l319_319018


namespace problem_proof_l319_319077

theorem problem_proof (p : ℕ) (hodd : p % 2 = 1) (hgt : p > 3):
  ((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 4) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p + 1) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 3) :=
by
  sorry

end problem_proof_l319_319077


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319707

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l319_319707


namespace expected_positions_after_swaps_l319_319629

theorem expected_positions_after_swaps:
  let n := 7
  let prob_not_swapped_by_Silva := (5/7: ℚ) * (5/7: ℚ)
  let prob_not_swapped_by_Chris := (5/7: ℚ)
  let prob_in_original_position := prob_not_swapped_by_Chris * prob_not_swapped_by_Silva + (2/7: ℚ) * prob_not_swapped_by_Silva
  let expected_number := n * prob_in_original_position
  in expected_number = (25/49: ℚ) * n := by
  sorry

end expected_positions_after_swaps_l319_319629


namespace min_value_of_ab_l319_319529

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_eq : ∀ (x y : ℝ), (x / a + y / b = 1) → (x^2 + y^2 = 1)) : a * b = 2 :=
by sorry

end min_value_of_ab_l319_319529


namespace distance_between_points_l319_319891

theorem distance_between_points (A B : ℝ × ℝ) (l1 l2 : ℝ × ℝ → Prop) (hA : A = (1, 7)) 
  (h_l1 : l1 = λ (p : ℝ × ℝ), p.1 - p.2 - 1 = 0) 
  (h_l2 : l2 = λ (p : ℝ × ℝ), p.1 + 3 * p.2 - 12 = 0) 
  (hB : ∃ B, l1 B ∧ l2 B) :
  let B := {B : ℝ × ℝ | l1 B ∧ l2 B}.some in 
  dist A B = (Real.sqrt 410) / 4 :=
sorry

end distance_between_points_l319_319891


namespace sum_seven_terms_of_arithmetic_seq_l319_319534

variable {α : Type}

open Nat

def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + n * d

def sum_first_seven_terms (a d : ℕ) : ℕ :=
  (1 + 2 + ··· + 7) * (a + d)

theorem sum_seven_terms_of_arithmetic_seq
  (a d : ℕ)
  (H : arithmetic_seq a d 2 + arithmetic_seq a d 3 + arithmetic_seq a d 4 = 12) :
  sum_first_seven_terms a d = 28 := by sorry

end sum_seven_terms_of_arithmetic_seq_l319_319534


namespace sugar_already_put_not_determinable_l319_319610

def Marys_baking_conditions (flour_in_recipe sugar_in_recipe flour_already_put flour_to_add : ℕ) : Prop :=
  flour_in_recipe = 9 ∧ sugar_in_recipe = 5 ∧ flour_already_put = 3 ∧ flour_to_add = 6

theorem sugar_already_put_not_determinable
  (flour_in_recipe sugar_in_recipe flour_already_put flour_to_add : ℕ)
  (h : Marys_baking_conditions flour_in_recipe sugar_in_recipe flour_already_put flour_to_add) :
  ¬(∃ sugar_already_put : ℕ, true) :=
by
  intro h_sugar_already_put
  cases h_sugar_already_put with s hs
  sorry

end sugar_already_put_not_determinable_l319_319610


namespace first_option_cost_l319_319198

-- Round trip distance
def round_trip_distance : ℕ := 300

-- First rental option cost per day
variable (a : ℕ)

-- Second rental option cost per day including gasoline
def second_option_cost : ℕ := 90

-- Gasoline coverage and cost
def kilometers_per_liter : ℕ := 15
def cost_per_liter : ℕ := 90 / 100  -- $0.90 per liter converted to ℕ for simplicity

-- Savings by choosing the first option
def savings : ℕ := 22

-- The main statement to prove
theorem first_option_cost :
  let gasoline_needed := round_trip_distance / kilometers_per_liter in
  let gasoline_cost := gasoline_needed * cost_per_liter in
  a = second_option_cost - savings - gasoline_cost / 10 :=
sorry

end first_option_cost_l319_319198


namespace smallest_positive_even_integer_l319_319897

noncomputable def smallest_even_integer (n : ℕ) : ℕ := 
  if 2 * n > 0 ∧ (3^(n * (n + 1) / 8)) > 500 then n else 0

theorem smallest_positive_even_integer :
  smallest_even_integer 6 = 6 :=
by
  -- Skipping the proofs
  sorry

end smallest_positive_even_integer_l319_319897


namespace hexagon_area_perimeter_ratio_l319_319772

theorem hexagon_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter := 6 * side_length
  let triangle_area := (Math.sqrt 3 / 4) * side_length^2
  let hexagon_area := 6 * triangle_area
  (hexagon_area / perimeter) = (5 * Math.sqrt 3 / 2) := 
by
  -- the actual proof would go here
  sorry

end hexagon_area_perimeter_ratio_l319_319772


namespace none_of_the_choices_sum_of_150_consecutive_integers_l319_319785

theorem none_of_the_choices_sum_of_150_consecutive_integers :
  ¬(∃ k : ℕ, 678900 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1136850 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1000000 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 2251200 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1876800 = 150 * k + 11325) :=
by
  sorry

end none_of_the_choices_sum_of_150_consecutive_integers_l319_319785


namespace matching_pair_probability_l319_319673

theorem matching_pair_probability :
  let gray_socks := 12
  let white_socks := 10
  let black_socks := 6
  let total_socks := gray_socks + white_socks + black_socks
  let total_ways := total_socks.choose 2
  let gray_matching := gray_socks.choose 2
  let white_matching := white_socks.choose 2
  let black_matching := black_socks.choose 2
  let matching_ways := gray_matching + white_matching + black_matching
  let probability := matching_ways / total_ways
  probability = 1 / 3 :=
by sorry

end matching_pair_probability_l319_319673


namespace ratio_of_sides_l319_319837

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l319_319837


namespace fourteenth_number_sum_digits_13_l319_319847

theorem fourteenth_number_sum_digits_13 : (list.filter (λ n : ℕ, (n.digits 10).sum = 13) (list.range 1000)).nth 13 = some 193 :=
by
  sorry

end fourteenth_number_sum_digits_13_l319_319847


namespace prove_solution_l319_319059

noncomputable def solve_problem : ℝ :=
  let x := Real.log 32 / Real.log 2 in
  32

theorem prove_solution (x y : ℝ) (h1 : 2^x * 3^y = 18) (h2 : 2^(x+3) + 3^(y+2) = 243) : 
  2^x ≈ 32 :=
sorry

end prove_solution_l319_319059


namespace sum_divisible_by_18_l319_319631

theorem sum_divisible_by_18 (n : ℕ) (a : Fin n → ℤ) (h : n = 35) :
  ∃ (S : Finset (Fin n)), S.card = 18 ∧ (∑ i in S, a i) % 18 = 0 := 
sorry

end sum_divisible_by_18_l319_319631


namespace initial_apples_correct_l319_319642

-- Define the conditions
def apples_handout : Nat := 5
def pies_made : Nat := 9
def apples_per_pie : Nat := 5

-- Calculate the number of apples used for pies
def apples_for_pies := pies_made * apples_per_pie

-- Define the total number of apples initially
def apples_initial := apples_for_pies + apples_handout

-- State the theorem to prove
theorem initial_apples_correct : apples_initial = 50 :=
by
  sorry

end initial_apples_correct_l319_319642


namespace complex_modulus_l319_319467

theorem complex_modulus (x y : ℝ) (h1 : (x : ℂ) * complex.i + (x : ℂ) = 4 + 2 * y * complex.i) :
  complex.abs ((x + 4 * y * complex.i) / (1 + complex.i)) = real.sqrt 10 :=
sorry

end complex_modulus_l319_319467


namespace board_of_ten_persons_exists_l319_319369

variable {α : Type}  -- Type for members of society
variable {β : Type}  -- Type for candidates

-- Definitions based on conditions
variable (S : set α)  -- Set of members of the society
variable (A : α → set β)  -- Each member chooses a set of 10 candidates

-- Condition: For each subset of 6 members, there exists a set of 2 persons making all happy
variable (B_condition : ∀ t : finset α, t.card = 6 → ∃ (B : set β), B.card = 2 ∧ ∀ i ∈ t, (B ∩ A i).nonempty)

-- We need to prove the existence of a board of 10 persons that makes every member happy.
theorem board_of_ten_persons_exists (h : ∀ (i : α), (A i).finite ∧ (A i).card = 10) :
  ∃ (B : set β), B.card = 10 ∧ ∀ i ∈ S, (B ∩ A i).nonempty :=
by
  sorry

end board_of_ten_persons_exists_l319_319369


namespace problem_statement_l319_319132

def S : ℕ → ℕ
| 0       := 1
| (n + 1) := if n < 2011 then 1 else S (n - 2011) + S (n - 2012)

theorem problem_statement (a : ℕ) : 2011 ∣ (S (2011 * a) - S a) :=
by 
  sorry

end problem_statement_l319_319132


namespace limit_proof_eq_5_l319_319796

open Classical

variables (ε : ℝ) (hε : ε > 0)

theorem limit_proof_eq_5 :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1/3| ∧ |x - 1/3| < δ → |(6 * x^2 + x - 1) / (x - 1/3) - 5| < ε :=
by
  let δ := ε / 6
  have hδ : δ > 0 := by linarith
  use δ, hδ
  intros x h
  have : (6 * x^2 + x - 1) / (x - 1/3) - 5 = 6 * x + 3 - 5 := by sorry
  sorry

end limit_proof_eq_5_l319_319796


namespace find_y_l319_319781

theorem find_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hrem : x % y = 11.52) (hdiv : x / y = 96.12) : y = 96 := 
sorry

end find_y_l319_319781


namespace yellow_yarns_count_l319_319226

theorem yellow_yarns_count (total_scarves red_yarn_count blue_yarn_count yellow_yarns scarves_per_yarn : ℕ) 
  (h1 : 3 = scarves_per_yarn)
  (h2 : red_yarn_count = 2)
  (h3 : blue_yarn_count = 6)
  (h4 : total_scarves = 36)
  :
  yellow_yarns = 4 :=
by 
  sorry

end yellow_yarns_count_l319_319226


namespace max_visitable_points_l319_319851

/-- Define the graph representing the paths between sites. 
This step would include details from the actual map, hypothetically represented here. -/
def pathGraph : Type -- Representation of path graph
-- details of pathGraph implementation are omitted for simplicity

/-- Define a function representing André's ability to visit points in alphabetical order without retracing steps. -/
def canVisitUpTo (n : ℕ) : Prop := -- implementation omitted for simplicity

theorem max_visitable_points : canVisitUpTo 10 :=
sorry -- proof omitted

end max_visitable_points_l319_319851


namespace cos_identity_l319_319089

theorem cos_identity 
  (x : ℝ) 
  (h : Real.sin (x - π / 3) = 3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := 
by 
  sorry

end cos_identity_l319_319089


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319749

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319749


namespace calculate_total_notebooks_given_to_tom_l319_319387

noncomputable def total_notebooks_given_to_tom : ℝ :=
  let initial_red := 15
  let initial_blue := 17
  let initial_white := 19
  let red_given_day1 := 4.5
  let blue_given_day1 := initial_blue / 3
  let remaining_red_day1 := initial_red - red_given_day1
  let remaining_blue_day1 := initial_blue - blue_given_day1
  let white_given_day2 := initial_white / 2
  let blue_given_day2 := remaining_blue_day1 * 0.25
  let remaining_white_day2 := initial_white - white_given_day2
  let remaining_blue_day2 := remaining_blue_day1 - blue_given_day2
  let red_given_day3 := 3.5
  let blue_given_day3 := (remaining_blue_day2 * 2) / 5
  let remaining_red_day3 := remaining_red_day1 - red_given_day3
  let remaining_blue_day3 := remaining_blue_day2 - blue_given_day3
  let white_kept_day3 := remaining_white_day2 / 4
  let remaining_white_day3 := initial_white - white_kept_day3
  let remaining_notebooks_day3 := remaining_red_day3 + remaining_blue_day3 + remaining_white_day3
  let notebooks_total_day3 := initial_red + initial_blue + initial_white - red_given_day1 - blue_given_day1 - white_given_day2 - blue_given_day2 - red_given_day3 - blue_given_day3 - white_kept_day3
  let tom_notebooks := red_given_day1 + blue_given_day1
  notebooks_total_day3

theorem calculate_total_notebooks_given_to_tom : total_notebooks_given_to_tom = 10.17 :=
  sorry

end calculate_total_notebooks_given_to_tom_l319_319387


namespace next_light_up_and_ring_time_l319_319020

def lcm (a b : ℕ) : ℕ :=
if a = 0 ∨ b = 0 then 0 else (a * b) / gcd a b

def light_up_interval : ℕ := 9
def ring_interval : ℕ := 60

theorem next_light_up_and_ring_time :
  lcm light_up_interval ring_interval = 180 :=
by sorry

end next_light_up_and_ring_time_l319_319020


namespace variance_inequality_l319_319202

-- Definitions based on the conditions
variable (x1 x2 x3 x4 x5 : ℝ)
variable (ξ1 ξ2 : List ℝ)

-- Conditions
def condition1 : Prop := 10 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 ≤ 10^4
def condition2 : Prop := x5 = 10^5
def condition3 : Prop := ξ1 = [x1, x2, x3, x4, x5] ∧ (∀ i, i ∈ ξ1 → i = 0.2)
def condition4 : Prop := ξ2 = [ (x1 + x2)/2, (x2 + x3)/2, (x3 + x4)/2, (x4 + x5)/2, (x5 + x1)/2 ] ∧ (∀ i, i ∈ ξ2 → i = 0.2)

-- Given variances Dξ1 and Dξ2
noncomputable def var (lst : List ℝ) : ℝ :=
  let μ := (∑ i in lst, i) / lst.length
  (∑ i in lst, (i - μ) ^ 2) / lst.length

def Dξ1 := var ξ1
def Dξ2 := var ξ2

-- The proof problem statement : Dξ1 > Dξ2.
theorem variance_inequality :
  condition1 x1 x2 x3 x4 → condition2 x5 →
  condition3 x1 x2 x3 x4 x5 ξ1 → condition4 x1 x2 x3 x4 x5 ξ2 →
  Dξ1 > Dξ2 :=
by 
  sorry

end variance_inequality_l319_319202


namespace angle_AOD_value_l319_319569

-- Define points and vectors
variables {O A B C D : Type} [inner_product_space ℝ O]
variables [normed_space ℝ O] [normed_group O] [complete_space O]

-- Define perpendicular conditions
def perp_OA_OC (O A C : O) : Prop := inner_product O A C = 0
def perp_OB_OD (O B D : O) : Prop := inner_product O B D = 0

-- Define angles in radians for generality, but will use degrees in problem context
noncomputable def angle_AOD (A O D : O) : ℝ := (2 * π / 360) * 128.57

-- Main theorem statement
theorem angle_AOD_value
  (O A B C D : O)
  (h1 : perp_OA_OC O A C)
  (h2 : perp_OB_OD O B D)
  (h3 : ∀(x : ℝ), angle_AOD A O D = 2.5 * (x * 2 * π / 360)) :
  angle_AOD A O D = (128.57 * 2 * π / 360) :=
sorry

end angle_AOD_value_l319_319569


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319748

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319748


namespace volume_of_water_in_cup_l319_319094

-- Define the frustum parameters
def frustum_lower_radius := 2
def frustum_upper_radius := 7
def frustum_height := 12

-- Define the calculation of frustum volume
def frustum_volume (r1 r2 h : ℝ) : ℝ := 
  (1 / 3) * Math.pi * h * (r1^2 + r1 * r2 + r2^2)

-- Calculate the volume of the frustum with the given parameters
def frustum_vol : ℝ := frustum_volume frustum_lower_radius frustum_upper_radius frustum_height

-- Define the radius and volume of sphere A
def sphere_a_radius := frustum_lower_radius
def sphere_volume (r : ℝ) : ℝ := 
  (4 / 3) * Math.pi * r^3

def sphere_a_vol : ℝ := sphere_volume sphere_a_radius

-- Define the volume of the segment created by sphere B
def sphere_b_radius := 3 -- Calculated from problem constraints
def frustum_segment_volume (r : ℝ) : ℝ := 
  (1 / 3) * Math.pi * r^2 * (frustum_height - r)

def sphere_b_vol : ℝ := frustum_segment_volume sphere_b_radius

-- Define the volume of water in the glass by subtracting the volumes of the spheres from the frustum
def water_volume : ℝ := frustum_vol - sphere_a_vol - sphere_b_vol

theorem volume_of_water_in_cup : water_volume = 61 * Math.pi := by
  sorry

end volume_of_water_in_cup_l319_319094


namespace hyperbola_focus_coordinates_l319_319421

theorem hyperbola_focus_coordinates :
  ∃ c : ℝ, (0, c) ∈ {(0, y) | ∃ x : ℝ, ((y^2 / 3) - (x^2 / 6) = 1)} ∧ c = 3 :=
by
  sorry

end hyperbola_focus_coordinates_l319_319421


namespace whole_numbers_between_cuberoots_l319_319151

theorem whole_numbers_between_cuberoots :
  let a := 10^(1 / 3)
  let b := 300^(1 / 3)
  (2 < a) → (a < 3) → (6 < b) → (b < 7) → ∃ n, n = 4 :=
by
  intro a b h1 h2 h3 h4
  have n_sub1 : {m // 3 ≤ m ∧ m ≤ 6} = {3, 4, 5, 6} :=
    by sorry
  exact ⟨4, by sorry⟩

end whole_numbers_between_cuberoots_l319_319151


namespace irrational_of_sqrt_3_l319_319376

theorem irrational_of_sqrt_3 :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ ↑a / ↑b = Real.sqrt 3) :=
sorry

end irrational_of_sqrt_3_l319_319376


namespace compare_a_and_fraction_correct_inequality_solution_correct_l319_319440

noncomputable def compare_a_and_fraction (a : ℝ) (h : a ≠ 0) : Prop :=
if a > 0 then 
  if a > sqrt 3 then a > 3 / a
  else if a = sqrt 3 then a = 3 / a
  else a < 3 / a 
else false

noncomputable def inequality_solution (a : ℝ) (h : a ≠ 0) : Set ℝ :=
if a > sqrt 2 then 
  {x : ℝ | x ≤ 2 / a} ∪ {x : ℝ | a ≤ x}
else if a = sqrt 2 then 
  Set.univ
else if 0 < a ∧ a < sqrt 2 then
  {x : ℝ | x ≤ a} ∪ {x : ℝ | 2 / a ≤ x}
else if -sqrt 2 < a ∧ a < 0 then
  {x : ℝ | 2 / a ≤ x ∧ x ≤ a}
else if a = -sqrt 2 then 
  {x : ℝ | x = -sqrt 2}
else
  {x : ℝ | a ≤ x ∧ x ≤ 2 / a}

theorem compare_a_and_fraction_correct (a : ℝ) (h : a ≠ 0) : {a > 0} → compare_a_and_fraction a h := 
sorry

theorem inequality_solution_correct (a : ℝ) (h : a ≠ 0) : (ax ^ 2 - (a ^ 2 + 2) * x + 2 * a ≥ 0) → (x ∈ (inequality_solution a h)) :=
sorry

end compare_a_and_fraction_correct_inequality_solution_correct_l319_319440


namespace incircle_distance_l319_319211

def dist_between_incircles (A B C : ℝ) (r1 : ℝ) (D E F G : ℝ) : Prop :=
  ∃ m, sqrt (10 * m) = dist (140 - 7, 7) (52.5, 24.5) ∧ m = 678.65

theorem incircle_distance :
  let A := 0 in
  let B := 105 in
  let C := 140 in
  let BC := 175 in
  let r1 := 35 in -- inferred from steps
  let D := 70 in
  let E := 70 in
  let F := 52.5 in
  let G := 122.5 in
  dist_between_incircles A B C r1 D E F G :=
sorry

end incircle_distance_l319_319211


namespace profit_percentage_l319_319979

-- definitions
variables (C S : ℝ)
variable (h : 40 * C = 25 * S)

-- statement
theorem profit_percentage (h : 40 * C = 25 * S) : ((S - C) / C) * 100 = 60 :=
by {
  have h1 : S = (8/5) * C,
  { linarith },

  have h2 : (S - C) = (3/5) * C,
  { rw h1, linarith },

  linarith
}

end profit_percentage_l319_319979


namespace quadratic_root_sum_l319_319120

theorem quadratic_root_sum (k : ℝ) (h : k ≤ 1 / 2) : 
  ∃ (α β : ℝ), (α + β = 2 - 2 * k) ∧ (α^2 - 2 * (1 - k) * α + k^2 = 0) ∧ (β^2 - 2 * (1 - k) * β + k^2 = 0) ∧ (α + β ≥ 1) :=
sorry

end quadratic_root_sum_l319_319120


namespace min_poly_degree_l319_319251

theorem min_poly_degree (p : Polynomial ℚ) :
  (p.isRoot(3 - Real.sqrt 8) ∧
   p.isRoot(5 + Real.sqrt 12) ∧
   p.isRoot(12 - 2 * Real.sqrt 11) ∧
   p.isRoot(-Real.sqrt 3)) →
   (∀ q : Polynomial ℚ, q.isRoot(3 - Real.sqrt 8) ∧
                         q.isRoot(5 + Real.sqrt 12) ∧
                         q.isRoot(12 - 2 * Real.sqrt 11) ∧
                         q.isRoot(-Real.sqrt 3) →
                         q.degree ≥ 8) →
  p.degree = 8 :=
by
  -- The proof is to be provided
  sorry

end min_poly_degree_l319_319251


namespace find_positive_integers_l319_319416

-- Define the required strictly positive integer variables
variables {a n p q r : ℕ}

-- Define the criteria for the variables to be strictly positive
def strictly_positive (x : ℕ) : Prop := x > 0

-- Define the main theorem
theorem find_positive_integers (ha : strictly_positive a)
  (hn : strictly_positive n) (hp : strictly_positive p)
  (hq : strictly_positive q) (hr : strictly_positive r) :
  a ^ n - 1 = (a ^ p - 1) * (a ^ q - 1) * (a ^ r - 1) ↔
  (a = 2 ∧ (p, q, r) ∈ { (1, 1, n), (1, n, 1), (n, 1, 1) }) ∨
  (a = 3 ∧ n = 2 ∧ p = 1 ∧ q = 1 ∧ r = 1) ∨
  (a = 2 ∧ n = 6 ∧ (p, q, r) ∈ { (2, 2, 3), (2, 3, 2), (3, 2, 2) }) :=
sorry

end find_positive_integers_l319_319416


namespace cucumbers_for_20_apples_l319_319537

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end cucumbers_for_20_apples_l319_319537


namespace sequence_property_l319_319283

theorem sequence_property : 
  (∀ (a : ℕ → ℝ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (2 * a n) / n) → a 200 = 40200) :=
by
  sorry

end sequence_property_l319_319283


namespace correct_sets_l319_319058

def positive_num_set : Set ℝ := {8, Real.pi / 2, 0.7, 3 / 4}
def integer_set : Set ℤ := {0, 8, -2}
def irrational_num_set : Set ℝ := {Real.pi / 2, -1.121121112...}

theorem correct_sets :
  positive_num_set = {8, Real.pi / 2, 0.7, 3 / 4} ∧
  integer_set = {0, 8, -2} ∧
  irrational_num_set = {Real.pi / 2, -1.121121112...} :=
by
  -- Each of the equivalencies needs proof of belonging to each set
  sorry

end correct_sets_l319_319058


namespace distance_between_4th_and_26th_blue_light_l319_319634

theorem distance_between_4th_and_26th_blue_light
  (sequence_cycle : List String)
  (repeat_pattern : sequence_cycle = ["blue", "blue", "yellow", "yellow", "yellow"])
  (light_spacing : ℕ)
  (spacing_8_inches : light_spacing = 8) :
  let distance_in_feet := (λ (pos1 pos2 : ℕ), ((pos2 - pos1) * light_spacing) / 12)
  in sequence_cycle = ["blue", "blue", "yellow", "yellow", "yellow"] →
     distance_in_feet 11 62 ≈ 33.333 := 
by
  sorry

end distance_between_4th_and_26th_blue_light_l319_319634


namespace third_smallest_number_sum_to_78_l319_319666

theorem third_smallest_number_sum_to_78 :
  ∃ n: ℕ, (∑ i in Finset.range (n + 1), i) = 78 ∧ n = 3 :=
by
  sorry

end third_smallest_number_sum_to_78_l319_319666


namespace minimum_teachers_required_l319_319011

def num_subjects : ℕ := 12
def num_teachers : ℕ := 40
def num_math_subjects : ℕ := 3
def num_science_subjects : ℕ := 4
def num_social_studies_subjects : ℕ := 3
def num_arts_subjects : ℕ := 2
def max_subjects_per_teacher : ℕ := 2
def min_math_teachers_per_subject : ℕ := 4
def min_science_teachers_per_subject : ℕ := 4
def min_social_studies_teachers_per_subject : ℕ := 2
def min_arts_teachers_per_subject : ℕ := 3

theorem minimum_teachers_required 
  (num_subjects = 12)
  (num_teachers = 40)
  (num_math_subjects = 3)
  (num_science_subjects = 4)
  (num_social_studies_subjects = 3)
  (num_arts_subjects = 2)
  (max_subjects_per_teacher = 2)
  (min_math_teachers_per_subject = 4)
  (min_science_teachers_per_subject = 4)
  (min_social_studies_teachers_per_subject = 2)
  (min_arts_teachers_per_subject  = 3) : 
  num_teachers =  40 :=
sorry 

end minimum_teachers_required_l319_319011


namespace subtract_angles_l319_319338

theorem subtract_angles :
  (90 * 60 * 60 - (78 * 60 * 60 + 28 * 60 + 56)) = (11 * 60 * 60 + 31 * 60 + 4) :=
by
  sorry

end subtract_angles_l319_319338


namespace range_of_a_l319_319275

theorem range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) ↔ a ≤ 1 :=
sorry

end range_of_a_l319_319275


namespace max_tan_A_correct_l319_319999

noncomputable def max_tan_A {A B C : ℝ} (AB BC : ℝ) (hAB : AB = 25) (hBC : BC = 20) : Prop :=
  ∀ C, ∃ (A B: ℝ), AC = (Real.sqrt (AB^2 - BC^2)) ∧ (AC = 15) → Real.tan (A) = 4 / 3

theorem max_tan_A_correct {A B C : ℝ} : max_tan_A 25 20 :=
by { sorry }

end max_tan_A_correct_l319_319999


namespace not_satisfied_ineq_count_l319_319905

theorem not_satisfied_ineq_count : 
  {x : ℤ | ¬ (10 * x^2 + 17 * x + 21 > 25)}.to_finset.card = 2 := 
begin
  sorry
end

end not_satisfied_ineq_count_l319_319905


namespace cyclic_quadrilateral_constraints_l319_319041

theorem cyclic_quadrilateral_constraints
  (α β : ℝ)
  (h1 : 0 < β ∧ β ≤ α ∧ α < 90)
  (h2 : α + β < 90)
  (A B C D : ℝ)
  (AB BC CD DA : ℝ) 
  (h3 : is_cyclic_quadrilateral A B C D)
  (h4 : ∠ BAD = α ∧ ∠ ABC = β) :
  sin (α + β) > AB ∧ AB > sin α ∧ sin β > DA ∧ DA > 0 ∧ 0 < CD ∧ CD < sin β ∧ sin α > BC ∧ BC > sin (α - β) :=
begin
  sorry,
end

end cyclic_quadrilateral_constraints_l319_319041


namespace find_x_l319_319431

noncomputable def a : ℝ := 0.889
noncomputable def b : ℝ := 55
noncomputable def c : ℝ := 9.97
noncomputable def d : ℝ := 1.23
noncomputable def e : ℝ := 2.71

noncomputable def x : ℝ := (a * Real.sqrt (b^2 - 4 * a * c) - b) / (2 * a) + Real.sin (d * e^(-c))

theorem find_x : x ≈ -5.1 :=
by
  sorry

end find_x_l319_319431


namespace find_QE_l319_319208

noncomputable def QE (QD DE : ℝ) : ℝ :=
  QD + DE

theorem find_QE :
  ∀ (Q C R D E : Type) (QR QD DE QE : ℝ), 
  QD = 5 →
  QE = QD + DE →
  QR = DE - QD →
  QR^2 = QD * QE →
  QE = (QD + 5 + 5 * Real.sqrt 5) / 2 :=
by
  intros
  sorry

end find_QE_l319_319208


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319718

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319718


namespace power_cycle_of_i_l319_319799

noncomputable def i : ℂ := complex.I

theorem power_cycle_of_i (n : ℕ) : 
  (∀ n : ℕ, i^(4*n + 1) = i) →
  (∀ n : ℕ, i^(4*n + 2) = -1) →
  (∀ n : ℕ, i^(4*n + 3) = -i) →
  (∀ n : ℕ, i^(4*n) = 1) →
  i^(2010) = -1 :=
by
  intro h1 h2 h3 h4
  sorry

end power_cycle_of_i_l319_319799


namespace angle_sum_triangle_l319_319165

theorem angle_sum_triangle (A B C : Type) (angle_A angle_B angle_C : ℝ) 
(h1 : angle_A = 45) (h2 : angle_B = 25) 
(h3 : angle_A + angle_B + angle_C = 180) : 
angle_C = 110 := 
sorry

end angle_sum_triangle_l319_319165


namespace german_russian_overlap_difference_l319_319880

def students := 2500
def G_lower := 1750
def G_upper := 1875
def R_lower := 625
def R_upper := 875

theorem german_russian_overlap_difference :
  (m M : ℕ) (G R : ℕ) 
  (H1 : students = 2500)
  (H2 : G ≥ G_lower ∧ G ≤ G_upper)
  (H3 : R ≥ R_lower ∧ R ≤ R_upper)
  (H4 : G + R - m = 2500)
  (H5 : G + R - M = 2500)
  (m_min : m = G + R - students)
  (M_max : M = G + R - students)
  : M - m = 375 := 
sorry

end german_russian_overlap_difference_l319_319880


namespace find_number_of_people_l319_319258

-- Assuming the following conditions:
-- The average weight of a group increases by 2.5 kg when a new person replaces one weighing 65 kg
-- The weight of the new person is 85 kg

def avg_weight_increase (N : ℕ) (old_weight new_weight : ℚ) : Prop :=
  let weight_diff := new_weight - old_weight
  let total_increase := 2.5 * N
  weight_diff = total_increase

theorem find_number_of_people :
  avg_weight_increase N 65 85 → N = 8 :=
by
  intros h
  sorry -- complete proof is not required

end find_number_of_people_l319_319258


namespace hyperbola_eccentricity_l319_319933

theorem hyperbola_eccentricity (p : Real) (hp : p > 0) :
  let a := Real.sqrt 3,
      b := p / (2 * Real.sqrt 2),
      c := Real.sqrt (3 + p^2 / 8) in
  c = p / 2 → (c / a = Real.sqrt 2) :=
by
  intros
  sorry

end hyperbola_eccentricity_l319_319933


namespace age_of_b_at_present_l319_319291

-- Variables representing the present age
variables (a b c d : ℕ)

-- Hypothesis
def conditions (A : ℕ) : Prop :=
  10 * A + 40 = 120 ∧ b = 2 * A + 10

theorem age_of_b_at_present (A : ℕ) (h : conditions A) : b = 26 :=
by {
  cases h with h_eq h_b,
  rw_nat_eq at h_eq,
  have ha8 : A = 8 := nat.equations._eqn_1.mp h_eq,
  rw [ha8, h_b],
  exact rfl,
}

end age_of_b_at_present_l319_319291


namespace Riemann_Inequality_l319_319638

def RiemannFunction (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 ∨ ¬(∃ p q : ℤ, 0 < q ∧ q < p ∧ Int.gcd p q = 1 ∧ x = q / p) 
  then 0 
  else 1 / ((classical.some_spec (classical.some_spec (classical.some x))).left)

theorem Riemann_Inequality (a b : ℝ) (ha : a ∈ Icc 0 1) (hb : b ∈ Icc 0 1) : 
  RiemannFunction (a * b) ≥ RiemannFunction a * RiemannFunction b :=
by
  sorry

end Riemann_Inequality_l319_319638


namespace cardinality_S_l319_319205

open Set

noncomputable def C (s : Set ℝ) : ℕ := s.toFinset.card

def star (A B : Set ℝ) : ℕ :=
if C A ≥ C B then C A - C B else C B - C A

def A : Set ℝ := {1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | (x^2 + a * x) * (x^2 + a * x + 2) = 0}

theorem cardinality_S :
  let S := {a : ℝ | C (B a) = 1 ∨ C (B a) = 3} in
  C S = 3 :=
by {
  -- Proof body is omitted
  sorry
}

end cardinality_S_l319_319205


namespace find_common_ratio_l319_319117

-- Define geometric sequence and sum of first n terms
def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

noncomputable def sum_of_geometric_sequence (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

-- Given conditions
axiom S1 (a q : ℝ) : ℝ := sum_of_geometric_sequence a q 1
axiom S2 (a q : ℝ) : ℝ := sum_of_geometric_sequence a q 2
axiom S3 (a q : ℝ) : ℝ := sum_of_geometric_sequence a q 3

-- S1, 2S2, 3S3 form an arithmetic sequence
axiom arithmetic_sequence (a q : ℝ) :
  2 * S2 a q - S1 a q = 3 * S3 a q - 2 * S2 a q

-- Proof of the common ratio
theorem find_common_ratio (a : ℝ) (q : ℝ) (hq : arithmetic_sequence a q) : q = 1 / 3 :=
by {
  sorry
}

end find_common_ratio_l319_319117


namespace playerAChampionshipProbability_l319_319550

-- Define the match format and winning probabilities
def playerAWins (n : Nat) : Bool :=
  n ≤ 5

-- List of generated random numbers
def randomNumbers : List Nat :=
  [192, 907, 966, 925, 271, 932, 812, 458, 569, 682,
   267, 393, 127, 556, 488, 730, 113, 537, 989, 431]

-- Count the number of wins for player A
def countAWins : Nat :=
  randomNumbers.filter playerAWins |>.length

-- Total number of games
def totalGames : Nat := 20

-- Estimated probability of player A winning the championship
def estimatedProbability : Float :=
  countAWins.toFloat / totalGames

theorem playerAChampionshipProbability : estimatedProbability = 0.65 := by
  sorry

end playerAChampionshipProbability_l319_319550


namespace monotonicity_intervals_l319_319095

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^3 - (a + b) * x^2 + b * x + c

noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 - 2 * (a + b) * x + b

theorem monotonicity_intervals (a b c : ℝ) (h₁ : a > 0) (h₂ : b ∈ ℝ) (h₃ : c ∈ ℝ) (h₄ : f' a b c (1/3) = 0) :
  (∀ x, x < 1/3 → (f' a a c x > 0)) ∧
  (∀ x, x > 1 → (f' a a c x > 0)) ∧
  (∀ x, 1/3 < x ∧ x < 1 → (f' a a c x < 0)) :=
sorry

end monotonicity_intervals_l319_319095


namespace three_real_roots_roots_as_triangle_sides_isosceles_triangle_l319_319940

theorem three_real_roots (m : ℝ) : (∀ x : ℝ, (x - 2) * (x^2 - 4 * x + m) = 0 → m ≤ 4) := sorry

theorem roots_as_triangle_sides (m : ℝ) : (∀ x : ℝ, (x - 2) * (x^2 - 4 * x + m) = 0 → 3 < m ∧ m ≤ 4) := sorry

theorem isosceles_triangle (m : ℝ) : (∀ x : ℝ, (x - 2) * (x^2 - 4 * x + m) = 0 → m = 4 ∧ (let s := 2 in (side_1 s * side_2 s / 2 = sqrt 3))) := sorry

end three_real_roots_roots_as_triangle_sides_isosceles_triangle_l319_319940


namespace g_prime_at_e_zero_range_of_a_range_of_m_l319_319487

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def e : ℝ := Real.exp 1
noncomputable def g (x : ℝ) := f(x) + x^2 - 2 * (e + 1) * x + 6

-- Problem 1.1 - Prove g'(e) = 0
theorem g_prime_at_e_zero : 
  (derivative g e) = 0 := sorry

-- Problem 1.2 - Prove the range of a
theorem range_of_a : 
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = a ∧ g x₂ = a) ↔ a ∈ Set.Ioo (6 - e^2 - e) 6 := sorry

-- Problem 2 - Prove the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x₀ ∈ Set.Icc 1 e, m * (f x₀ - 1) > x₀^2 + 1) ↔ 
  m < -2 ∨ m > (e^2 + 1) / (e - 1) := sorry

end g_prime_at_e_zero_range_of_a_range_of_m_l319_319487


namespace range_of_x_l319_319276

noncomputable def is_valid_x (x : ℝ) : Prop :=
  x ≥ 0 ∧ x ≠ 4

theorem range_of_x (x : ℝ) : 
  is_valid_x x ↔ x ≥ 0 ∧ x ≠ 4 :=
by sorry

end range_of_x_l319_319276


namespace parallelogram_height_l319_319423

theorem parallelogram_height (b A : ℝ) (h : ℝ) (h_base : b = 28) (h_area : A = 896) : h = A / b := by
  simp [h_base, h_area]
  norm_num
  sorry

end parallelogram_height_l319_319423


namespace combined_weight_after_removal_l319_319808

theorem combined_weight_after_removal (weight_sugar weight_salt weight_removed : ℕ) 
                                       (h_sugar : weight_sugar = 16)
                                       (h_salt : weight_salt = 30)
                                       (h_removed : weight_removed = 4) : 
                                       (weight_sugar + weight_salt) - weight_removed = 42 :=
by {
  sorry
}

end combined_weight_after_removal_l319_319808


namespace max_even_numbers_in_product_l319_319297

theorem max_even_numbers_in_product 
  (a : Fin 100 → ℕ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_32_even_sums : ∃ f : Fin 99 → Bool, (∑ i in Finset.univ.filter f, 1) = 32 ∧ 
    (∀ j, f j = true ↔ is_even ((\lam x, (Finset.univ.filter (λ y, y ≠ j)).prod a))) := 
  ∃ (n : ℕ), 
  n ≤ 33 ∧ 
  (∀ m, m ≤ 100 ∧ is_even (∑ i in Finset.range m, a i) → m ≤ n) := 
sorry

end max_even_numbers_in_product_l319_319297


namespace PA_perpendicular_BC_l319_319382

-- Define the geometric entities and their relationships
variables {A B C O1 O2 P : Point}
variables {E F G H : Point}
variables (tangent_1 : tangency_relation O1 A B C)
variables (tangent_2 : tangency_relation O2 A B C)
variables (tangency_points : tangency_points E F G H O1 O2)
variables (intersection : intersects (line_through_points E G) (line_through_points F H) P)

-- The main theorem to prove
theorem PA_perpendicular_BC (h: tangent_1 ∧ tangent_2 ∧ tangency_points ∧ intersection) : perp (line_through_points P A) (line_through_lines B C) :=
by { sorry }

end PA_perpendicular_BC_l319_319382


namespace clevercat_total_l319_319853

noncomputable def clevercat_center_total (J F S : Set Cat) : Nat :=
  let only_jump := (60 - (20 - 10 + 25 - 10 + 10 - 10))
  let only_fetch := (35 - (20 - 10 + 15 - 10 + 10 - 10))
  let only_spin := (40 - (25 - 10 + 15 - 10 + 10 - 10))
  let jump_fetch_only :=  (20 - 10)
  let fetch_spin_only := (15 - 10)
  let jump_spin_only := (25 - 10)
  let all_three := 10
  let none := 8
  in only_jump + only_fetch + only_spin + jump_fetch_only + fetch_spin_only + jump_spin_only + all_three + none

theorem clevercat_total:
  let J : Set Cat := {c | c.can_jump}
  let F : Set Cat := {c | c.can_fetch}
  let S : Set Cat := {c | c.can_spin}
  clevercat_center_total J F S = 93 :=
by
  sorry

end clevercat_total_l319_319853


namespace angle_B_measure_minimum_AB_dot_CB_l319_319185

-- Definitions for the sides and condition
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (BC BA CA CB AB : ℝ → ℝ → ℝ)

-- Conditions
def triangle_condition : Prop := 
  (2 * a + c) * (BC b c) * (BA c a) + c * (CA a c) * (CB c b) = 0

def side_length_b : Prop := 
  b = 2 * Real.sqrt 3

-- Theorem statements
theorem angle_B_measure (h : triangle_condition a b c BC BA CA CB AB) : 
  B = 2 * Real.pi / 3 := sorry

theorem minimum_AB_dot_CB (h1 : triangle_condition a b c BC BA CA CB AB) (h2 : side_length_b b) : 
  ∃(ac : ℝ), (\vec{AB} \cdot \vec{CB}) ac = -2 := sorry

end angle_B_measure_minimum_AB_dot_CB_l319_319185


namespace rectangle_side_greater_than_twelve_l319_319794

theorem rectangle_side_greater_than_twelve (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 :=
sorry

end rectangle_side_greater_than_twelve_l319_319794


namespace binary_conversion_correct_l319_319043

/-- Definition of the problem: convert decimal 51 to binary. -/
def decimal_number : ℕ := 51

/-- The expected binary representation of 51 is 110011 -/
def binary_representation : string := "110011"

/-- The theorem stating that the binary representation of decimal_number is binary_representation. -/
theorem binary_conversion_correct : nat.toBinaryString decimal_number = binary_representation :=
  sorry

end binary_conversion_correct_l319_319043


namespace greatest_prime_factor_14_12_l319_319791

-- Define the double factorial for even numbers
def double_factorial (n : ℕ) : ℕ :=
  if n % 2 = 1 then 1
  else List.prod (List.range' 2 (n / 2) * 2 + 1)

-- Specific definitions for {14} and {12}
def prod_14 : ℕ := double_factorial 14
def prod_12 : ℕ := double_factorial 12

-- Theorem: The greatest prime factor of {14} + {12} is 5
theorem greatest_prime_factor_14_12 : 
  Nat.greatestPrimeFactor (prod_14 + prod_12) = 5 :=
by
  sorry

end greatest_prime_factor_14_12_l319_319791


namespace largest_black_cells_5x100_l319_319343

theorem largest_black_cells_5x100 :
  ∀ (n : ℕ) (grid : Matrix (Fin 5) (Fin 100) Bool),
  (∀ i j, grid i j = true → (card { ij' | adj (i,j) ij' ∧ grid ij'.1 ij'.2 = true} ≤ 2)) →
  (∃ (colors : Fin 5 → Fin 100 → Bool), (∑ i j, if colors i j = true then 1 else 0) = n ∧ 
  (∀ i j, colors i j = true → (card { ij' | adj (i,j) ij' ∧ colors ij'.1 ij'.2 = true} ≤ 2))) →
  n ≤ 302 :=
by
  intros n grid h bounded_colors
  sorry

def adj {m n : Type} (pos1 pos2 : m × n) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2 = pos2.2 + 1 ∨ pos1.2 = pos2.2 - 1)) ∨
  (pos1.2 = pos2.2 ∧ (pos1.1 = pos2.1 + 1 ∨ pos1.1 = pos2.1 - 1))


end largest_black_cells_5x100_l319_319343


namespace find_all_k_l319_319060

theorem find_all_k :
  ∃ (k : ℝ), ∃ (v : ℝ × ℝ), v ≠ 0 ∧ (∃ (v₀ v₁ : ℝ), v = (v₀, v₁) 
  ∧ (3 * v₀ + 6 * v₁) = k * v₀ ∧ (4 * v₀ + 3 * v₁) = k * v₁) 
  ↔ k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6 :=
by
  -- here goes the proof
  sorry

end find_all_k_l319_319060


namespace sum_partial_fractions_eq_l319_319860

noncomputable def sum_partial_fractions : ℚ :=
  (∑ n in finset.range 499, (1 : ℚ) / (n + 2) / (n + 1))

theorem sum_partial_fractions_eq :
  sum_partial_fractions = 499 / 1000 := by
  sorry

end sum_partial_fractions_eq_l319_319860


namespace tangent_line_at_one_minimum_value_exists_l319_319946

-- Definition of the function f(x) = ax^2 - ln(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x

-- The equation of the tangent line to f(x) at x = 1
theorem tangent_line_at_one (a : ℝ) (x : ℝ) :
  (a = 1) → (f a 1 = 1) → (x - x = 0) :=
by sorry

-- Proving that there exists a real number such that the minimum value of the function on (0, e] is 3/2
theorem minimum_value_exists (a : ℝ) (x : ℝ) :
  (a = Real.exp 2 / 2) ∧ (∀ x ∈ Icc (0 : ℝ) (Real.exp 1), (f a x = 3/2)) :=
by sorry

end tangent_line_at_one_minimum_value_exists_l319_319946


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319750

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319750


namespace perimeter_of_DEF_l319_319913

noncomputable def hypotenuse := 4 + 4 * Real.sqrt 3
noncomputable def angle_A := 60
noncomputable def side_AC := (4 + 4 * Real.sqrt 3) / 2
noncomputable def side_BC := Math.sqrt ((4 + 4 * Real.sqrt 3)^2 - ((4 + 4 * Real.sqrt 3)/2)^2)

def side_D := hypotenuse
def side_E := side_BC

noncomputable def perimeter_DEF := 18 + 10 * Real.sqrt 3 + 4 * Real.sqrt 6 + 6 * Real.sqrt 2

theorem perimeter_of_DEF :
  let ABC_right_triangle := True in
  let angle_60_deg := angle_A = 60 in
  let hypotenuse_AB := hypotenuse in
  let line_p_parallel_to_AC := True in
  let D_on_line_p := side_D = hypotenuse in
  let E_on_line_p := side_E = side_BC in
  18 + 10 * Real.sqrt 3 + 4 * Real.sqrt 6 + 6 * Real.sqrt 2 = perimeter_DEF := by
  sorry

end perimeter_of_DEF_l319_319913


namespace inscribed_quadrilateral_circle_eq_radius_l319_319912

noncomputable def inscribed_circle_condition (AB CD AD BC : ℝ) : Prop :=
  AB + CD = AD + BC

noncomputable def equal_radius_condition (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  r₁ = r₃ ∨ r₄ = r₂

theorem inscribed_quadrilateral_circle_eq_radius 
  (AB CD AD BC r₁ r₂ r₃ r₄ : ℝ)
  (h_inscribed_circle: inscribed_circle_condition AB CD AD BC)
  (h_four_circles: ∀ i, (i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4) → ∃ (r : ℝ), r = rᵢ): 
  equal_radius_condition r₁ r₂ r₃ r₄ :=
by {
  sorry
}

end inscribed_quadrilateral_circle_eq_radius_l319_319912


namespace divisible_by_coprime_with_10_l319_319458

theorem divisible_by_coprime_with_10 
  (digits : ℕ → ℕ) 
  (n : ℕ) 
  (h_coprime : Nat.coprime n 10) 
  : ∃ (sub_seq : ℕ → ℕ) (k m : ℕ), 0 < k ∧ 0 < m 
      ∧ (∀ i, i < k → sub_seq i = digits (m + i)) 
      ∧ (Nat.ofDigits sub_seq) % n = 0 :=
sorry

end divisible_by_coprime_with_10_l319_319458


namespace domain_of_f_parity_of_f_l319_319340

noncomputable def f (x : ℝ) := log (3 + x) + log (3 - x)

theorem domain_of_f :
  ∀ x : ℝ, (-3 < x ∧ x < 3) ↔ f x = log (3 + x) + log (3 - x) :=
by
  sorry

theorem parity_of_f :
  ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end domain_of_f_parity_of_f_l319_319340


namespace α_minus_β_squared_l319_319865

-- Define the polynomial equation and its roots
def polynomial (a b c x : ℝ) := a * x^2 + b * x + c

-- The given polynomial
def given_polynomial := polynomial 1 (-3) 1

-- Define the roots α and β as roots of the polynomial
def is_root (p : ℝ → ℝ) (r : ℝ) := p r = 0

-- Define α and β
def α := (3 + Real.sqrt 5) / 2
def β := (3 - Real.sqrt 5) / 2

-- The main statement
theorem α_minus_β_squared :
  is_root given_polynomial α ∧ is_root given_polynomial β → (α - β) ^ 2 = 5 := by
sorry

end α_minus_β_squared_l319_319865


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319761

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319761


namespace function_maximum_at_1_l319_319490

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem function_maximum_at_1 :
  ∀ x > 0, (f x ≤ f 1) :=
by
  intro x hx
  have hx_pos : 0 < x := hx
  sorry

end function_maximum_at_1_l319_319490


namespace unique_positive_integer_l319_319861

theorem unique_positive_integer (n : ℕ) (h : 3 * 3^2 + 4 * 3^3 + 5 * 3^4 + ∑ i in finset.range (n - 2), (i + 6) * 3^(i + 3) = 3^(n + 11)) : n = 59049 := 
sorry

end unique_positive_integer_l319_319861


namespace value_of_last_installment_l319_319344

noncomputable def total_amount_paid_without_processing_fee : ℝ :=
  36 * 2300

noncomputable def total_interest_paid : ℝ :=
  total_amount_paid_without_processing_fee - 35000

noncomputable def last_installment_value : ℝ :=
  2300 + 1000

theorem value_of_last_installment :
  last_installment_value = 3300 :=
  by
    sorry

end value_of_last_installment_l319_319344


namespace algebraic_expression_independent_and_polynomial_value_l319_319938

variable (a b : ℤ)

def expression_independent_of_x (a b : ℤ) : Prop :=
  2 - 2 * b = 0 ∧ a + 3 = 0

def polynomial_value (a b : ℤ) : ℤ :=
  3 * (a^2 - 2 * a * b - b^2) - (4 * a^2 + a * b + b^2)

theorem algebraic_expression_independent_and_polynomial_value :
  expression_independent_of_x (-3) 1 →
  polynomial_value (-3) 1 = 8 :=
by
  intro h
  simp only [expression_independent_of_x, polynomial_value] at h
  rw [h.1, h.2]
  sorry

end algebraic_expression_independent_and_polynomial_value_l319_319938


namespace sum_of_squares_of_roots_l319_319859

theorem sum_of_squares_of_roots :
  ∀ (p q r : ℚ), (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) ∧
                 (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) ∧
                 (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
                 p^2 + q^2 + r^2 = 34 / 9 := 
by
  sorry

end sum_of_squares_of_roots_l319_319859


namespace j_is_odd_and_decreasing_l319_319017

-- Definitions of the functions
def f (x : ℝ) : ℝ := x ^ (1/3 : ℝ)
def g (x : ℝ) : ℝ := log ((1/3 : ℝ)) (|x|)
def h (x : ℝ) : ℝ := x + (2 / x)
def j (x : ℝ) : ℝ := 2 ^ -x - 2 ^ x

-- Predicate definitions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f(x) ≥ f(y)

theorem j_is_odd_and_decreasing :
  odd_function j ∧ decreasing_function j := by
  sorry

end j_is_odd_and_decreasing_l319_319017


namespace diamond_value_l319_319968

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem diamond_value : diamond 3 4 = 36 := by
  -- Given condition: x ♢ y = 4x + 6y
  -- To prove: (diamond 3 4) = 36
  sorry

end diamond_value_l319_319968


namespace largest_digit_N_divisible_by_6_l319_319308

theorem largest_digit_N_divisible_by_6 (N : ℕ) (h1 : Nat.even N) (h2 : (29 + N) % 3 = 0) (h3 : N < 10) :
  N = 4 :=
by
  sorry

end largest_digit_N_divisible_by_6_l319_319308


namespace fill_parentheses_correct_l319_319152

theorem fill_parentheses_correct (a b : ℝ) :
  (3 * b + a) * (3 * b - a) = 9 * b^2 - a^2 :=
by 
  sorry

end fill_parentheses_correct_l319_319152


namespace volume_BEG_CFH_l319_319998

/--
Given a regular quadrilateral pyramid P-ABCD with a volume of 1,
and points E, F, G, and H are the midpoints of segments AB, CD, PB, and PC respectively,
prove that the volume of the polyhedron BEG-CFH is equal to 5/16.
-/
theorem volume_BEG_CFH (P A B C D E F G H: Point)
    (h1: regular_pyramid P A B C D) 
    (h2: volume (pyramid P A B C D) = 1) 
    (h3: is_midpoint E A B) 
    (h4: is_midpoint F C D) 
    (h5: is_midpoint G P B) 
    (h6: is_midpoint H P C):
    volume (polyhedron BEG CFH) = 5 / 16 := 
  sorry

end volume_BEG_CFH_l319_319998


namespace tangent_line_eq_a1_max_a_for_nonneg_f_l319_319945

def f (x : ℝ) (a : ℝ) := Real.sin x - a * x

theorem tangent_line_eq_a1 :
  let x := (Real.pi / 2),
      tangent_slope := -1,
      tangent_pt := (x, f x 1)
  in
  tangent_pt = (Real.pi / 2, 1 - Real.pi / 2) ∧ tangent_slope = -1 ∧
  ∀ x y, y - (1 - Real.pi / 2) = -1 * (x - Real.pi / 2) ↔ y = -x + 1 := by
  sorry

theorem max_a_for_nonneg_f :
  ∀ x, (0 < x ∧ x ≤ Real.pi / 2) → (Real.sin x / x) ≥ Real.sin (Real.pi / 2) / (Real.pi / 2) → f x (2 / Real.pi) ≥ 0 := by
  sorry

end tangent_line_eq_a1_max_a_for_nonneg_f_l319_319945


namespace calc_value_l319_319988

theorem calc_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := 
by 
  sorry

end calc_value_l319_319988


namespace num_integers_j_l319_319543

def sum_of_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).sum

theorem num_integers_j (h : ∀ j, 1 ≤ j ∧ j ≤ 5041 → sum_of_divisors j = 1 + Nat.sqrt j + j) : 
  ∃ n, n = 20 :=
sorry

end num_integers_j_l319_319543


namespace quadrilateral_area_ratio_l319_319234

theorem quadrilateral_area_ratio 
  (r : ℝ) 
  (hPQRS_in_circle: ∀ (P Q R S: ℝ), (P^2 + Q^2 = r^2) ∧ (R^2 + S^2 = r^2))
  (hPR_diameter: ∃ PR: ℝ, PR = 2 * r)
  (h_angle_RPS: ∀ (P R S: ℝ), angle R P S = 60)
  (h_angle_QPR: ∀ (Q P R: ℝ), angle Q P R = 30):
  let a := 2 
  let b := 3 
  let c := 2 
  ((a + b + c) = 7) :=
sorry

end quadrilateral_area_ratio_l319_319234


namespace mahdi_bird_watches_on_Saturday_l319_319608

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq, Repr

open Day

-- Define Mahdi's activities
inductive Activity : Type
| Paint | Cooking | Yoga | BirdWatching | Cycling
deriving DecidableEq, Repr

open Activity

-- Mahdi's schedule conditions
def not_consecutive (d1 d2 : Day) : Prop :=
  abs (d1.to_nat - d2.to_nat) ≠ 1

def schedule : Day → Activity → Prop
| Monday, Paint := true
| Wednesday, Cooking := true
| d, Yoga := d ≠ Monday ∧ d ≠ Wednesday
| _ , _ := false

-- Mahdi's constraints
def yoga_days : list Day := [Tuesday, Friday, Sunday]
def possible_yoga day := schedule day Yoga
def bird_watching_day := Saturday

-- Prove that Mahdi goes bird watching on Saturday given the conditions
theorem mahdi_bird_watches_on_Saturday :
  (∃ day1 day2 day3, schedule day1 Yoga ∧ schedule day2 Yoga ∧ schedule day3 Yoga ∧
   not_consecutive day1 day2 ∧ not_consecutive day2 day3 ∧ not_consecutive day1 day3) →
  (∃ day, schedule day BirdWatching ∧ day = Saturday) :=
sorry

end mahdi_bird_watches_on_Saturday_l319_319608


namespace reconstruct_n_l319_319336

variables {R : Type*} [LinearOrderedField R]
variables (a b c : R) (ab_pos : a * b * (a + b) ≠ 0)

-- Define the lines
def k := λ (x : R), a * x
def l := λ (x : R), b * x
def m := λ (x : R), c + 2 * (a * b) / (a + b) * x
def n := λ (x : R), -a * x + c

theorem reconstruct_n :
  ∃ (a b c : R), ∀ x : R, (ab_pos → m x = c + 2 * (a * b) / (a + b) * x ∧ k x = a * x ∧ l x = b * x) → n x = -a * x + c :=
sorry

end reconstruct_n_l319_319336


namespace cost_per_serving_in_cents_after_coupon_l319_319136

def oz_per_serving : ℝ := 1
def price_per_bag : ℝ := 25
def bag_weight : ℝ := 40
def coupon : ℝ := 5
def dollars_to_cents (d : ℝ) : ℝ := d * 100

theorem cost_per_serving_in_cents_after_coupon : 
  dollars_to_cents ((price_per_bag - coupon) / bag_weight) = 50 := by
  sorry

end cost_per_serving_in_cents_after_coupon_l319_319136


namespace count_primes_with_g_equal_to_three_l319_319871

def F : ℕ → Polynomial ℤ 
| 0       := Polynomial.zero
| 1       := Polynomial.X - 1
| n@(m+2) := 2 * Polynomial.X * F (n-1) - F (n-2) + 2 * F 1

def g (n : ℕ) : ℕ :=
  (F n).factorization.toList.length

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem count_primes_with_g_equal_to_three :
  (Finset.range 101).filter (λ k, 2 < k ∧ g k = 3 ∧ is_prime k).card = 24 :=
by 
suppress sorry

end count_primes_with_g_equal_to_three_l319_319871


namespace line_equation_l319_319265

theorem line_equation {a b c : ℝ} (x : ℝ) (y : ℝ)
  (point : ∃ p: ℝ × ℝ, p = (-1, 0))
  (perpendicular : ∀ k: ℝ, k = 1 → 
    ∀ m: ℝ, m = -1 → 
      ∀ b1: ℝ, b1 = 0 → 
        ∀ x1: ℝ, x1 = -1 →
          ∀ y1: ℝ, y1 = 0 →
            ∀ l: ℝ, l = b1 + k * (x1 - (-1)) + m * (y1 - 0) → 
              x - y + 1 = 0) :
  x - y + 1 = 0 :=
sorry

end line_equation_l319_319265


namespace closest_integer_to_10_minus_sqrt_13_l319_319377

theorem closest_integer_to_10_minus_sqrt_13 :
  ∀ (x ∈ ({7, 6, 5, 4} : Set ℤ)), x = 6 :=
by
  sorry

end closest_integer_to_10_minus_sqrt_13_l319_319377


namespace HarryWorked33Hours_l319_319789

variable (x H : ℝ)

-- Conditions
def HarryPayFirst15Hours : ℝ := 15 * x
def HarryPayAdditionalHours : ℝ := 1.5 * x * (H - 15)
def HarryTotalPay : ℝ := HarryPayFirst15Hours + HarryPayAdditionalHours

def JamesPayFirst40Hours : ℝ := 40 * x
def JamesPayAdditionalHour : ℝ := 2 * x
def JamesTotalPay : ℝ := JamesPayFirst40Hours + JamesPayAdditionalHour

theorem HarryWorked33Hours 
  (h_equal_pay : HarryTotalPay x H = JamesTotalPay x) : 
  H = 33 := by
  sorry

end HarryWorked33Hours_l319_319789


namespace sets_equal_l319_319378

-- Defining the sets and proving their equality
theorem sets_equal : { x : ℝ | x^2 + 1 = 0 } = (∅ : Set ℝ) :=
  sorry

end sets_equal_l319_319378


namespace cylinder_radius_l319_319163

open Real

theorem cylinder_radius (r : ℝ) 
  (h₁ : ∀(V₁ : ℝ), V₁ = π * (r + 4)^2 * 3)
  (h₂ : ∀(V₂ : ℝ), V₂ = π * r^2 * 9)
  (h₃ : ∀(V₁ V₂ : ℝ), V₁ = V₂) :
  r = 2 + 2 * sqrt 3 :=
by
  sorry

end cylinder_radius_l319_319163


namespace largest_divisor_of_expression_l319_319267

theorem largest_divisor_of_expression (n : ℤ) : ∃ k, ∀ n : ℤ, n^4 - n^2 = k * 12 :=
by sorry

end largest_divisor_of_expression_l319_319267


namespace midpoint_on_radical_axis_l319_319585

variables {A B C D O G H : Point}

-- Define cyclic quadrilateral with center O
axiom cyclic_quadrilateral_center : cyclic_quadrilateral A B C D O

-- Define existence of G and H
axiom meet_G : meet_points (circumcircle A O B) (circumcircle C O D) G
axiom meet_H : meet_points (circumcircle A O D) (circumcircle B O C) H

-- Define circles ω1 and ω2
axiom circle_omega1 : circle_through G (foot_perpendicular G (line A B)) (foot_perpendicular G (line C D)) ω1
axiom circle_omega2 : circle_through H (foot_perpendicular H (line B C)) (foot_perpendicular H (line D A)) ω2

-- Goal statement: prove that the midpoint of GH lies on the radical axis of ω1 and ω2
theorem midpoint_on_radical_axis (midpoint: Point) :
  midpoint = (midpoint G H) →
  lies_on_radical_axis midpoint ω1 ω2 :=
sorry

end midpoint_on_radical_axis_l319_319585


namespace price_of_necklace_l319_319672

-- Define the necessary conditions.
def num_charms_per_necklace : ℕ := 10
def cost_per_charm : ℕ := 15
def num_necklaces_sold : ℕ := 30
def total_profit : ℕ := 1500

-- Calculation of selling price per necklace
def cost_per_necklace := num_charms_per_necklace * cost_per_charm
def total_cost := cost_per_necklace * num_necklaces_sold
def total_revenue := total_cost + total_profit
def selling_price_per_necklace := total_revenue / num_necklaces_sold

-- Statement of the problem in Lean 4
theorem price_of_necklace : selling_price_per_necklace = 200 := by
  sorry

end price_of_necklace_l319_319672


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319759

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319759


namespace expansion_properties_l319_319122

theorem expansion_properties (x : ℝ) (n : ℕ) :
  (let binom_expansion := (sqrt x + 1 / (2 * fourth_root x)) ^ n in
   let first_three_coefficients := 
     binom_expansion.coeff 0,
     1/2 * (binom_expansion.coeff 1),
     1/4 * (binom_expansion.coeff 2) in
   -- Condition: Coefficients form an arithmetic sequence
   (2 * first_three_coefficients.2 = first_three_coefficients.1 + first_three_coefficients.3) →
   -- Question 1: Finding n
   (n = 8)) ∧
  -- Rational Terms 
  (let rational_terms := [binom_expansion.term 0, binom_expansion.term 4, binom_expansion.term 8] in
   rational_terms = [x^4, (35/8)*x, (1/256)*x^(-2)]) ∧
  -- Terms with Maximum Coefficient
  (let max_coeff_terms := 
    let term3 := binom_expansion.term 2, term4 := binom_expansion.term 3 in
    [term3, term4] in
   max_coeff_terms = [7 * x^(5/2), 7 * x^(7/4)]))
  := sorry

end expansion_properties_l319_319122


namespace triangle_A_eqdist_excircle_centers_l319_319419

theorem triangle_A_eqdist_excircle_centers (A B C O1 O2 : Type) [MetricSpace A]
  (dist_A_O1 : Metric.dist A O1 = Metric.dist A O2)
  (angle_O1_A_O2 : ∀ (P Q R : Type), ∠PQR = 90) :
  ∠C = 90 :=
by sorry

end triangle_A_eqdist_excircle_centers_l319_319419


namespace ray_inequality_l319_319418

theorem ray_inequality (a : ℝ) :
  (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ 1)
  ∨ (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ -1) :=
sorry

end ray_inequality_l319_319418


namespace like_terms_satisfy_conditions_l319_319918

theorem like_terms_satisfy_conditions (m n : ℤ) (h1 : m - 1 = n) (h2 : m + n = 3) :
  m = 2 ∧ n = 1 := by
  sorry

end like_terms_satisfy_conditions_l319_319918


namespace speed_in_still_water_l319_319327

/-- Conditions -/
def upstream_speed : ℝ := 30
def downstream_speed : ℝ := 40

/-- Theorem: The speed of the man in still water is 35 kmph. -/
theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 35 := 
by 
  sorry

end speed_in_still_water_l319_319327


namespace compare_log_exp_l319_319446

theorem compare_log_exp (x y z : ℝ) 
  (hx : x = Real.log 2 / Real.log 5) 
  (hy : y = Real.log 2) 
  (hz : z = Real.sqrt 2) : 
  x < y ∧ y < z := 
sorry

end compare_log_exp_l319_319446


namespace isosceles_triangle_base_angle_l319_319977

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (h1 : vertex_angle = 110) :
  ∃ base_angle : ℝ, base_angle = 35 :=
by
  use 35
  sorry

end isosceles_triangle_base_angle_l319_319977


namespace greatest_multiple_l319_319724

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l319_319724


namespace a5_value_l319_319454

variable {a : ℕ → ℝ} (q : ℝ) (a2 a3 : ℝ)

-- Assume the conditions: geometric sequence, a_2 = 2, a_3 = -4
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ q, ∀ n, a (n + 1) = a n * q

-- Given conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 2 = 2
axiom h3 : a 3 = -4

-- Theorem to prove
theorem a5_value : a 5 = -16 :=
by
  -- Here you would provide the proof based on the conditions
  sorry

end a5_value_l319_319454


namespace Carlos_earnings_l319_319183

theorem Carlos_earnings :
  ∃ (wage : ℝ), 
  (18 * wage) = (12 * wage + 36) ∧ 
  wage = 36 / 6 ∧ 
  (12 * wage + 18 * wage) = 180 :=
by
  sorry

end Carlos_earnings_l319_319183


namespace euler_formula_not_true_l319_319547

theorem euler_formula_not_true : 
  (∀ x : ℝ, (Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x)) →
  (Complex.exp (Complex.I * Real.pi) + 1 = 0) ∧
  ((1/2 + Complex.I * (Real.sqrt 3)/2) ^ 2022 = 1) ∧
  (∀ x : ℝ, abs (Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x)) ≤ 2) ∧
  ¬(∀ x : ℝ, -2 ≤ Complex.exp (Complex.I * x) - Complex.exp (-Complex.I * x) ∧ Complex.exp (Complex.I * x) - Complex.exp (-Complex.I * x) ≤ 2) :=
sorry

end euler_formula_not_true_l319_319547


namespace rectangle_area_l319_319652

-- Definitions:
variables (l w : ℝ)

-- Conditions:
def condition1 : Prop := l = 4 * w
def condition2 : Prop := 2 * l + 2 * w = 200

-- Theorem statement:
theorem rectangle_area (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 1600 :=
sorry

end rectangle_area_l319_319652


namespace fraction_of_salary_spent_on_house_rent_l319_319358

theorem fraction_of_salary_spent_on_house_rent
    (S : ℕ) (H : ℚ)
    (cond1 : S = 180000)
    (cond2 : S / 5 + H * S + 3 * S / 5 + 18000 = S) :
    H = 1 / 10 := by
  sorry

end fraction_of_salary_spent_on_house_rent_l319_319358


namespace length_of_faster_train_is_70_meters_l319_319304

-- Define the speeds in kmph
def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36

-- Convert speeds to m/s
def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

-- Relate speeds
def speed_faster_train_mps := kmph_to_mps speed_faster_train_kmph
def speed_slower_train_mps := kmph_to_mps speed_slower_train_kmph

-- Define the time to cross in seconds
def time_to_cross_seconds : ℝ := 7

-- Define the relative speed in m/s
def relative_speed_mps : ℝ := speed_faster_train_mps - speed_slower_train_mps

-- The distance covered by the faster train in the given time
def length_faster_train : ℝ := relative_speed_mps * time_to_cross_seconds

-- The theorem to prove
theorem length_of_faster_train_is_70_meters : length_faster_train = 70 :=
by
  -- Skipping the steps for now
  sorry

end length_of_faster_train_is_70_meters_l319_319304


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319768

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l319_319768


namespace find_a_plus_b_l319_319088

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h_imag : i = complex.I) (h_eq : (a - 2 * i) * i = b + i) :
  a + b = 3 :=
by {
  sorry
}

end find_a_plus_b_l319_319088


namespace value_of_x_minus_y_l319_319524

theorem value_of_x_minus_y (x y : ℝ) 
  (h1 : |x| = 2) 
  (h2 : y^2 = 9) 
  (h3 : x + y < 0) : 
  x - y = 1 ∨ x - y = 5 := 
by 
  sorry

end value_of_x_minus_y_l319_319524


namespace P_plus_Q_plus_R_eq_x_plus_y_plus_z_l319_319220

variables (a b c x y z : ℝ)
hypothesis h : a + b + c = 1
hypothesis ha : 0 < a
hypothesis hb : 0 < b
hypothesis hc : 0 < c

def P := a * x + b * y + c * z
def Q := b * x + c * y + a * z
def R := c * x + a * y + b * z

theorem P_plus_Q_plus_R_eq_x_plus_y_plus_z : P + Q + R = x + y + z :=
by sorry

end P_plus_Q_plus_R_eq_x_plus_y_plus_z_l319_319220


namespace combined_basketballs_l319_319639

-- Conditions as definitions
def spursPlayers := 22
def rocketsPlayers := 18
def basketballsPerPlayer := 11

-- Math Proof Problem statement
theorem combined_basketballs : 
  (spursPlayers * basketballsPerPlayer) + (rocketsPlayers * basketballsPerPlayer) = 440 :=
by
  sorry

end combined_basketballs_l319_319639


namespace sum_series_l319_319855

theorem sum_series (n : ℕ) (h : n > 0) : 
  (∑ k in Finset.range n, 1 / (k + 1) / (k + 2 : ℝ)) = n / (n + 1 : ℝ) := by
  sorry

end sum_series_l319_319855


namespace nagel_point_concurrency_l319_319624

noncomputable def nagel_point (a b c : ℝ) : Prop :=
  let p := (a + b + c) / 2 in
  let A' := p - c in
  let A'' := p - b in
  let B' := p - a in
  let B'' := p - c in
  let C' := p - b in
  let C'' := p - a in
  (A' / A'') * (B' / B'') * (C' / C'') = 1

/-- 
 Prove that the segments connecting the vertices of a triangle with the points of 
 tangency of the opposite sides with the corresponding excircles intersect at a 
 single point (the Nagel point).
-/
theorem nagel_point_concurrency 
  (a b c : ℝ) (h : nagel_point a b c) : 
  ∃ P : ℝ, true := 
begin
  sorry
end

end nagel_point_concurrency_l319_319624


namespace fraction_of_women_married_l319_319788

-- Definitions for the conditions
def total_employees : ℕ := 100
def women_percent : ℚ := 61 / 100
def married_percent : ℚ := 60 / 100
def men_single_fraction : ℚ := 2 / 3

-- Calculate number of employees corresponding to conditions
def women : ℕ := (women_percent * total_employees).natAbs
def men : ℕ := total_employees - women
def married : ℕ := (married_percent * total_employees).natAbs

def men_married : ℕ := ((1 - men_single_fraction) * men).natAbs
def women_married : ℕ := married - men_married

-- Final fraction to prove
def fraction_women_married : ℚ := women_married / women

-- Lean theorem statement, ensuring conditions and goal
theorem fraction_of_women_married : fraction_women_married = 47 / 61 :=
by
  sorry

end fraction_of_women_married_l319_319788


namespace parallelism_condition_l319_319103

variable (l m : Line)
variable (α : Plane)
variable (h₁ : m ⊂ α)

-- Define parallelism statements
def parallel_to_plane (l : Line) (α : Plane) : Prop := sorry -- definition placeholder
def parallel_to_line (l m : Line) : Prop := sorry -- definition placeholder

theorem parallelism_condition (h₁ : m ⊂ α) :
  ¬( (parallel_to_line l m) -> (parallel_to_plane l α) ) ∧ 
  ¬( (parallel_to_plane l α) -> (parallel_to_line l m) ) :=
sorry

end parallelism_condition_l319_319103


namespace sample_size_of_stratified_sampling_l319_319350

theorem sample_size_of_stratified_sampling (total_employees young_employees middle_aged_employees elderly_employees sample_young_employees : ℕ)
    (h_total : total_employees = 600)
    (h_young : young_employees = 250)
    (h_middle_aged : middle_aged_employees = 200)
    (h_elderly : elderly_employees = 150)
    (h_sample_young : sample_young_employees = 5):
    ∃ (S : ℕ), S = 12 := 
  by
    have py := young_employees / total_employees
    have sample_young_proportion := sample_young_employees = S * py
    have h_sample_young_proportion : 5 = S * (250 / 600)
    sorry

end sample_size_of_stratified_sampling_l319_319350


namespace parabola_directrix_l319_319646

theorem parabola_directrix :
  ∀ (p : ℝ), (y^2 = 6 * x) → (x = -3/2) :=
by
  sorry

end parabola_directrix_l319_319646


namespace sequence_s5_l319_319914

-- Definitions based on the conditions provided in part a)
def sequence (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | n+1 => 2 * (sequence n + 1) - sequence (n - 1)

-- Theorem statement
theorem sequence_s5 : sequence 5 = 21 :=
sorry

end sequence_s5_l319_319914


namespace value_of_expression_l319_319986

theorem value_of_expression 
  (a b : ℝ) 
  (h : b = 3 * a - 2) : 
  2 * b - 6 * a + 2 = -2 :=
by 
  intro h
  rw [h]
  linarith

end value_of_expression_l319_319986


namespace compute_result_l319_319075

def star (a b : ℝ) := (a + b) / (a - b)

theorem compute_result :
  ((star 3 4) ⋆ 5) = 1 / 6 :=
by
  sorry

end compute_result_l319_319075


namespace factorial_tail_count_l319_319046

def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125

theorem factorial_tail_count : 
  let count_non_factorial_tails := 5000 - (19975 / 5) 
  in count_non_factorial_tails = 1005 := 
by {
  -- Formalizing the solution to show the equivalence 
  -- between the theoretical result and the given answer.
  sorry 
}

end factorial_tail_count_l319_319046


namespace gcd_binomials_l319_319622

open Nat

theorem gcd_binomials (n k : ℕ) (h : n ≥ k) : 
  Nat.gcd (list.foldl (λ acc i, Nat.gcd acc (binomial (n + i) k)) (binomial n k) (list.range k)) = 1 := 
sorry

end gcd_binomials_l319_319622


namespace minimum_floor_sum_l319_319908

theorem minimum_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊a^2 + b^2 / c⌋ + ⌊b^2 + c^2 / a⌋ + ⌊c^2 + a^2 / b⌋ = 34 :=
sorry

end minimum_floor_sum_l319_319908


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319694

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l319_319694


namespace range_of_m_l319_319116

theorem range_of_m (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : a * b = a + b + 3) (h_ineq : a * b ≥ m) : m ≤ 9 :=
sorry

end range_of_m_l319_319116


namespace tangential_quadrilateral_radius_l319_319395

noncomputable def radius_of_incircle (PQ QR RS SP : ℝ) (h_tangential : PQ + RS = QR + SP) : ℝ :=
  let s := (PQ + QR + RS + SP) / 2
      in let A := real.sqrt ((s - PQ) * (s - QR) * (s - RS) * (s - SP))
      in A / s

theorem tangential_quadrilateral_radius :
  radius_of_incircle 10 13 13 15 (by linarith) = r :=
sorry

end tangential_quadrilateral_radius_l319_319395


namespace proposition_choice_l319_319210

-- Definitions for planes and lines
variable (α β : Set Point) (l m : Set Point)

-- Propositions
def p := α ∥ β ∧ l ⊆ α ∧ m ⊆ β → l ∥ m
def q := l ∥ α ∧ m ⊥ l ∧ m ⊆ β → α ⊥ β

-- Theorem to be proved
theorem proposition_choice : ¬ p ∨ q :=
sorry

end proposition_choice_l319_319210


namespace sector_angle_radian_measure_l319_319476

theorem sector_angle_radian_measure (r l : ℝ) (h1 : r = 1) (h2 : l = 2) : l / r = 2 := by
  sorry

end sector_angle_radian_measure_l319_319476


namespace stable_triangle_sum_exists_unique_l319_319216

variable {n : ℕ}
variable {s : Fin n → ℕ}
variable (stable : (i j k : Fin n) → (0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n) →
             (a i j + a j k ≤ a i k) ∧ (a i k ≤ a i j + a j k + 1))

theorem stable_triangle_sum_exists_unique (h1 : 0 < n)
                                          (h2 : ∀ i, i < n → s i ≤ s (i + 1)) :
  ∃! (a : Fin n → Fin n → ℕ),
    (∀ i j k, 0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n →
      (a i j + a j k ≤ a i k) ∧ (a i k ≤ a i j + a j k + 1)) ∧
    (∀ k, k < n → (∑ j in Finset.range k, a k j) = s k) :=
sorry

end stable_triangle_sum_exists_unique_l319_319216


namespace greatest_multiple_of_5_and_6_less_than_1000_l319_319711

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l319_319711


namespace proof_A_B_relation_l319_319341

-- Define the sets A and B
def B (n : ℕ) : set (list char) :=
  { l | l.length = n ∧ ∀ i < n-1, ¬(l.nth i = l.nth (i+1) ∧ (l.nth i = some 'a' ∨ l.nth i = some 'b'))  }

def A (n : ℕ) : set (list char) :=
  { l | l.length = n ∧ ∀ i < n-2, l.nth i ≠ l.nth (i+1) ∨ l.nth (i+1) ≠ l.nth (i+2) }

-- The theorem to be proven
theorem proof_A_B_relation (n : ℕ) : ∃ (A B : ℕ), A = 3 * B :=
sorry

end proof_A_B_relation_l319_319341


namespace A_can_give_B_start_l319_319172

noncomputable def start_A_to_B : ℝ :=
let start_A_to_C := 200 in
let start_B_to_C := 130.43478260869563 in
let speed_ratio_B_to_C := (1000 - start_B_to_C) / 1000 in
let B_distance_when_A_runs_1000 := 800 / speed_ratio_B_to_C in
1000 - B_distance_when_A_runs_1000

theorem A_can_give_B_start :
  start_A_to_B = 80 :=
by
  -- skipped proof
  sorry

end A_can_give_B_start_l319_319172


namespace A_is_incenter_of_triangle_BCD_l319_319109

-- Given the conditions about the geometric setup
variables {O₁ O₂ A B C D : Type}
variables [circle O₁] [circle O₂]
variables (intersectAB : intersects O₁ O₂ A B)
variables (extensionO₁A_C : extends O₁ A C O₂)
variables (extensionO₂A_D : extends O₂ A D O₁)

-- Define the geometry relationships
def concyclicity : Prop :=
  ∃ circle α, α.contains O₁ ∧ α.contains B ∧ α.contains O₂ ∧ α.contains C ∧ α.contains D

def angle_bisectors : Prop :=
  is_angle_bisector O₁ C (angle B C D) ∧ is_angle_bisector O₂ D (angle B D C)

def incenter (A : Type) (Δ : triangle B C D) : Prop :=
  is_incenter A Δ

-- The theorem statement
theorem A_is_incenter_of_triangle_BCD (intersection_conditions : concyclicity ∧ angle_bisectors) : incenter A (triangle B C D) :=
by
  sorry

end A_is_incenter_of_triangle_BCD_l319_319109


namespace matt_books_second_year_l319_319619

-- Definitions based on the conditions
variables (M : ℕ) -- number of books Matt read last year
variables (P : ℕ) -- number of books Pete read last year

-- Pete read twice as many books as Matt last year
def pete_read_last_year (M : ℕ) : ℕ := 2 * M

-- This year, Pete doubles the number of books he read last year
def pete_read_this_year (M : ℕ) : ℕ := 2 * (2 * M)

-- Matt reads 50% more books this year than he did last year
def matt_read_this_year (M : ℕ) : ℕ := M + M / 2

-- Pete read 300 books across both years
def total_books_pete_read_last_and_this_year (M : ℕ) : ℕ :=
  pete_read_last_year M + pete_read_this_year M

-- Prove that Matt read 75 books in his second year
theorem matt_books_second_year (M : ℕ) (h : total_books_pete_read_last_and_this_year M = 300) :
  matt_read_this_year M = 75 :=
by sorry

end matt_books_second_year_l319_319619


namespace greatest_multiple_of_5_and_6_lt_1000_l319_319743

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l319_319743


namespace ceil_square_range_count_l319_319518

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l319_319518


namespace isosceles_right_triangle_area_l319_319677

-- Definitions for the conditions
structure Triangle where
  P Q R : Type
  P_angle : ℝ
  Q_angle : ℝ
  R_angle : ℝ
  PQ_length : ℝ
  PR_length : ℝ
  QR_length : ℝ

noncomputable def isosceles_triangle_PQR : Triangle :=
{ P := unit,
  Q := unit,
  R := unit,
  P_angle := 45,
  Q_angle := 90,
  R_angle := 45,
  PQ_length := 8,
  PR_length := 8,
  QR_length := 8 * Real.sqrt 2 -- Hypotenuse of a right triangle with legs of length 8
}

-- Proof statement
theorem isosceles_right_triangle_area :
  ∀ (T : Triangle), T = isosceles_triangle_PQR → (1/2) * T.PQ_length * T.PR_length = 32 := 
by
  intro T h
  rw [h]
  -- Proof steps skipped with sorry
  sorry

end isosceles_right_triangle_area_l319_319677


namespace sin_A_range_l319_319919

-- Definitions and conditions for the problem
variables {α β γ : Type} [Preorder α] [Preorder β] [Preorder γ]

-- Lean 4 statement
theorem sin_A_range (A B C : ℝ) (a b c : ℝ)
  (h_c_minus_a : c - a = 2 * a * cos B)
  (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) :
  (∃ (x : ℝ), x = (sin (B - A) / sin A) ∧ x ∈ (1/2 : ℝ, sqrt (2)/2 : ℝ)) :=
sorry

end sin_A_range_l319_319919


namespace complex_conjugate_z_in_first_quadrant_l319_319890

def z : ℂ := 1 / (1 - (1:ℂ).im * complex.i)

theorem complex_conjugate (z := 1 / (1 - (1:ℂ).im * complex.i)) :
  complex.conj z = (1 / 2) - ((1 / 2) : ℂ).re * complex.i := 
by sorry

theorem z_in_first_quadrant (z := 1 / (1 - (1:ℂ).im * complex.i)) :
  (z.re > 0) ∧ (z.im > 0) := 
by sorry

end complex_conjugate_z_in_first_quadrant_l319_319890


namespace least_mul_2520_l319_319066

theorem least_mul_2520 (k : ℕ) (h1 : k >= 5) (h2 : k <= 10) :
  Nat.lcm_list (List.range (k - 4) * 5) = 2520 :=
sorry

end least_mul_2520_l319_319066


namespace count_numbers_with_digit_5_in_1_to_600_l319_319138

theorem count_numbers_with_digit_5_in_1_to_600 :
  {n : ℕ | 1 ≤ n ∧ n ≤ 600 ∧ (∃ d, d ∈ nat.digits 10 n ∧ d = 5 ) }.card = 195 := 
sorry

end count_numbers_with_digit_5_in_1_to_600_l319_319138


namespace polar_coords_of_M_l319_319661

-- Define the rectangular coordinates of point M
def M_rect_coords : (ℝ × ℝ) := (-1, -sqrt 3)

-- Define the polar coordinates function
noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if x < 0 then Real.atan (y / x) + π else Real.atan (y / x)
  (r, θ)

-- Prove that the polar coordinates of the given point M are (2, 4π/3)
theorem polar_coords_of_M : polar_coordinates (-1) (-sqrt 3) = (2, 4 * π / 3) :=
by
  sorry

end polar_coords_of_M_l319_319661


namespace find_a_l319_319952

noncomputable def A : Set ℝ := {1, 2, 3, 4}
noncomputable def B (a : ℝ) : Set ℝ := { x | x ≤ a }

theorem find_a (a : ℝ) (h_union : A ∪ B a = Set.Iic 5) : a = 5 := by
  sorry

end find_a_l319_319952


namespace space_left_each_side_l319_319826

theorem space_left_each_side (wall_width : ℕ) (picture_width : ℕ)
  (picture_centered : wall_width = 2 * ((wall_width - picture_width) / 2) + picture_width) :
  (wall_width - picture_width) / 2 = 9 :=
by
  have h : wall_width = 25 := sorry
  have h2 : picture_width = 7 := sorry
  exact sorry

end space_left_each_side_l319_319826


namespace imaginary_part_of_square_l319_319893

def complex_square (z : ℂ) : ℂ :=
  z * z

theorem imaginary_part_of_square :
  let z := (1 + 2 * complex.I : ℂ) in
  let w := complex_square z in
  complex.im w = 4 := by
  sorry

end imaginary_part_of_square_l319_319893


namespace total_wheels_l319_319319

-- Definitions of given conditions
def bicycles : ℕ := 50
def tricycles : ℕ := 20
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Theorem stating the total number of wheels for bicycles and tricycles combined
theorem total_wheels : bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160 :=
by
  sorry

end total_wheels_l319_319319


namespace sin_alpha_beta_l319_319939

theorem sin_alpha_beta (a b c α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
    (h3 : a * Real.cos α + b * Real.sin α + c = 0) (h4 : a * Real.cos β + b * Real.sin β + c = 0) 
    (h5 : α ≠ β) : Real.sin (α + β) = (2 * a * b) / (a ^ 2 + b ^ 2) := 
sorry

end sin_alpha_beta_l319_319939


namespace cube_root_expression_l319_319056

theorem cube_root_expression (x : ℝ) (hx : x ≥ 0) : (x * Real.sqrt (x * x^(1/3)))^(1/3) = x^(5/9) :=
by
  sorry

end cube_root_expression_l319_319056


namespace possible_values_ceil_square_l319_319508

noncomputable def num_possible_values (x : ℝ) (hx : ⌈x⌉ = 12) : ℕ := 23

theorem possible_values_ceil_square (x : ℝ) (hx : ⌈x⌉ = 12) :
  let n := num_possible_values x hx in n = 23 :=
by
  let n := num_possible_values x hx
  exact rfl

end possible_values_ceil_square_l319_319508


namespace wage_increase_l319_319170

variables (L : ℝ) (w_dwarves w_elves : ℝ)

def labor_supply_dwarves := 1 + L / 3
def labor_supply_elves := 3 + L
def labor_demand_dwarves_inv := 10 - 2 * L / 3
def labor_demand_elves_inv := 18 - 2 * L

theorem wage_increase :
  (∃ L_dwarves : ℝ, labor_supply_dwarves L_dwarves = labor_demand_dwarves_inv L_dwarves ∧ w_dwarves = 4) ∧
  (∃ L_elves : ℝ, labor_supply_elves L_elves = labor_demand_elves_inv L_elves ∧ w_elves = 8) →
  (∃ w : ℝ, 
      (∃ L_total : ℝ, 4 * w - 6 = L_total) ∧ 
      (∃ L_total' : ℝ, 21.5 - 1.5 * w = L_total') ∧ 
      w = 5 ∧ 
      w / 4 = 1.25) :=
sorry

end wage_increase_l319_319170


namespace proof_statement_l319_319993

open Real

noncomputable def statement : Prop :=
  exists (X Y Z P Q R : Type),
  (X ≠ Y) ∧ (X ≠ Z) ∧ (Y ≠ Z) ∧
  (P ∈ line_segment Y Z) ∧
  (Q ∈ line_segment X Z) ∧
  (is_intersecting_point (line X P) (line Y Q) R) ∧
  (dist X R / dist R P = 2) ∧
  (dist Y R / dist R Q = 3) ∧
  (dist Y P / dist Y Z = 17 / 29)

theorem proof_statement : statement := sorry

end proof_statement_l319_319993
