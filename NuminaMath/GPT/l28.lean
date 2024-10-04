import Mathlib

namespace evaluate_difference_of_squares_l28_28194

theorem evaluate_difference_of_squares : 81^2 - 49^2 = 4160 := by
  sorry

end evaluate_difference_of_squares_l28_28194


namespace hyperbola_properties_l28_28272

def hyperbola (x y : ℝ) : Prop := x^2 - 4 * y^2 = 1

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → (x + 2 * y = 0 ∨ x - 2 * y = 0)) ∧
  (2 * (1 / 2) = 1) := 
by
  sorry

end hyperbola_properties_l28_28272


namespace triangle_at_most_one_obtuse_l28_28892

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180) (h3 : 0 < C ∧ C < 180) (h4 : A + B + C = 180) : A ≤ 90 ∨ B ≤ 90 ∨ C ≤ 90 :=
by
  sorry

end triangle_at_most_one_obtuse_l28_28892


namespace true_propositions_count_is_one_l28_28147

-- Define the given propositions
def P_1 (a b : ℝ) : Prop := a^2 * b^2 = (a * b)^2
def P_2 (a b : ℝ) : Prop := |a + b| > |a - b|
def P_3 (a b : ℝ) : Prop := |a + b|^2 = (a + b)^2
def P_4 (a b : ℝ) [vector_space ℝ] : Prop := a ∥ b → a • b = |a| * |b|

-- Prove that the number of true propositions is 1
theorem true_propositions_count_is_one (a b : ℝ) [vector_space ℝ] : 
  ((if P_1 a b then 1 else 0) 
  + (if P_2 a b then 1 else 0) 
  + (if P_3 a b then 1 else 0) 
  + (if P_4 a b then 1 else 0)) = 1 :=
by
  sorry

end true_propositions_count_is_one_l28_28147


namespace infinity_non_almost_square_l28_28580

def is_almost_square (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≤ b ∧ b ≤ 101 * a / 100 ∧ n = a * b

theorem infinity_non_almost_square :
  ∃ᶠ m in at_top, ∀ k : ℕ, k ≤ 198 → ¬ is_almost_square (m + k) :=
by sorry

end infinity_non_almost_square_l28_28580


namespace probability_of_black_ball_l28_28751

/-- Let the probability of drawing a red ball be 0.42, and the probability of drawing a white ball be 0.28. Prove that the probability of drawing a black ball is 0.3. -/
theorem probability_of_black_ball (p_red p_white p_black : ℝ) (h1 : p_red = 0.42) (h2 : p_white = 0.28) (h3 : p_red + p_white + p_black = 1) : p_black = 0.3 :=
by
  sorry

end probability_of_black_ball_l28_28751


namespace average_age_of_contestants_l28_28760

theorem average_age_of_contestants :
  let numFemales := 12
  let avgAgeFemales := 25
  let numMales := 18
  let avgAgeMales := 40
  let sumAgesFemales := avgAgeFemales * numFemales
  let sumAgesMales := avgAgeMales * numMales
  let totalSumAges := sumAgesFemales + sumAgesMales
  let totalContestants := numFemales + numMales
  (totalSumAges / totalContestants) = 34 := by
  sorry

end average_age_of_contestants_l28_28760


namespace no_naturals_satisfy_divisibility_condition_l28_28191

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end no_naturals_satisfy_divisibility_condition_l28_28191


namespace sin2_cos3_tan4_lt_zero_l28_28993

theorem sin2_cos3_tan4_lt_zero (h1 : Real.sin 2 > 0) (h2 : Real.cos 3 < 0) (h3 : Real.tan 4 > 0) : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 :=
sorry

end sin2_cos3_tan4_lt_zero_l28_28993


namespace average_of_six_l28_28853

theorem average_of_six (A : ℝ) 
  (h1 : 4 * 5 = 20) 
  (h2 : 2 * 14 = 28) 
  (h3 : 6 * A = 20 + 28) : 
  A = 8 :=
begin
  sorry
end

end average_of_six_l28_28853


namespace calculate_shot_cost_l28_28621

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l28_28621


namespace find_salary_l28_28570

noncomputable def salary (S : ℝ) : Prop :=
  S - (S / 4 + S / 6 + S / 8 + S / 12 + S / 24) = 45000

theorem find_salary : ∃ (S : ℝ), salary S ∧ S = 540000 :=
by
  use 540000
  dsimp [salary]
  have h₁ : 540000 / 4 = 135000 := by norm_num
  have h₂ : 540000 / 6 = 90000 := by norm_num
  have h₃ : 540000 / 8 = 67500 := by norm_num
  have h₄ : 540000 / 12 = 45000 := by norm_num
  have h₅ : 540000 / 24 = 22500 := by norm_num
  rw [h₁, h₂, h₃, h₄, h₅]
  norm_num
  sorry

end find_salary_l28_28570


namespace decreasing_function_range_of_a_l28_28245

noncomputable def f (a x : ℝ) : ℝ := log a (2 - a^x)

theorem decreasing_function_range_of_a:
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → f a y < f a x) →
  0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2 :=
by
  -- Proof to be filled in
  sorry

end decreasing_function_range_of_a_l28_28245


namespace central_angle_of_region_l28_28109

theorem central_angle_of_region (prob : ℚ) (h : prob = 1 / 8) : 
  ∃ (x : ℚ), x = 45 :=
by
  use 45
  exact h

end central_angle_of_region_l28_28109


namespace correct_propositions_l28_28668

-- Define the parameters involved in the propositions
variable (A B C D : Type) [DistinctPoints A B C D] -- Assume A, B, C, D are distinct points
variable (line : Type → Type) -- Type of lines in space
variable [Coplanar : ∀ {X Y : Type}, Prop] -- Predicate for coplanarity
variable [Skew : ∀ {X Y : Type}, Prop] -- Predicate for skew lines
variable [Perpendicular : ∀ {X Y : Type}, Prop] -- Predicate for perpendicular lines
variable [Equal : ∀ {X Y : Type}, Prop] -- Predicate for equality of segments
variable [RegularTetrahedron : ∀ {W X Y Z : Type}, Prop] -- Predicate for a regular tetrahedron

-- Define the propositions as variables
variable P1 : Coplanar (line A B) (line C D) → Coplanar (line A C) (line B D)
variable P2 : Skew (line A B) (line C D) → Skew (line A C) (line B D)
variable P3 : Equal (line A B) (line A C) ∧ Equal (line D B) (line D C) → Perpendicular (line A D) (line B C)
variable P4 : Perpendicular (line A B) (line C D) ∧ Perpendicular (line A C) (line B D) → Perpendicular (line A D) (line B C)
variable P5 : Equal (line A B) (line A C) ∧ Equal (line A B) (line A D) ∧ Equal (line B C) (line C D) ∧ Equal (line C D) (line D B) → RegularTetrahedron A B C D

-- The main theorem
theorem correct_propositions : (P1 ∧ P2 ∧ P3 ∧ P4) ∧ ¬P5 :=
by
  sorry

end correct_propositions_l28_28668


namespace largest_k_l28_28114

noncomputable def S : ℕ → ℕ → ℕ
| 1, n := n
| m, 1 := 1
| m, n := if h₁ : m ≥ 2 ∧ n ≥ 2 then S (m - 1) n * S m (n - 1) else 0

theorem largest_k (k : ℕ) : (∀ k₀ : ℕ, 2 ^ k₀ ∣ S 7 7 → k₀ ≤ k) ∧ (2 ^ k ∣ S 7 7) :=
begin
  let k := 63,
  have h₁ : 2 ^ k ∣ S 7 7,
  { 
    sorry 
  },
  have h₂ : ∀ k₀ : ℕ, 2 ^ k₀ ∣ S 7 7 → k₀ ≤ k,
  { 
    sorry 
  },
  exact ⟨h₂, h₁⟩
end

end largest_k_l28_28114


namespace solve_for_x_l28_28001

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l28_28001


namespace area_triangle_AEF_l28_28156

-- Define the properties of the rectangle and points
variables (ABCD : set (fin 4 → ℝ)) -- Rectangle
variables (BE : ℝ) -- Segment BE
variables (DF : ℝ) -- Segment DF

-- Given Conditions:
axiom area_ABCD : 56 = ∑ (x : fin 2), (finset.univ.sum (λ y, (ABCD x y))) -- Area of rectangle ABCD is 56 cm^2
axiom seg_BE : BE = 3 -- BE = 3 cm
axiom seg_DF : DF = 2 -- DF = 2 cm

-- Goal:
theorem area_triangle_AEF : ∃ (AEF : set (fin 3 → ℝ)), ∑ (x : fin 2), (finset.univ.sum (λ y, (AEF x y))) = 25 :=
begin
  sorry
end

end area_triangle_AEF_l28_28156


namespace no_solution_l28_28876

def fibonacci : ℕ → ℕ
| 0        := 1
| 1        := 1
| (n + 2)  := fibonacci (n + 1) + fibonacci n

theorem no_solution (n : ℕ) (h : n ≥ 1) : n * fibonacci n * fibonacci (n + 1) ≠ (fibonacci (n + 2) - 1) ^ 2 :=
sorry

end no_solution_l28_28876


namespace gamma_max_success_ratio_l28_28761

theorem gamma_max_success_ratio (x y z w : ℕ) (h_yw : y + w = 500)
    (h_gamma_first_day : 0 < x ∧ x < 170 * y / 280)
    (h_gamma_second_day : 0 < z ∧ z < 150 * w / 220)
    (h_less_than_500 : (28 * x + 22 * z) / 17 < 500) :
    (x + z) ≤ 170 := 
sorry

end gamma_max_success_ratio_l28_28761


namespace product_of_equal_numbers_l28_28852

theorem product_of_equal_numbers (a b : ℕ) (equal_pair : ℕ) (h1 : (10 + 18 + equal_pair + equal_pair) / 4 = 15) :
  equal_pair * equal_pair = 256 :=
by
  have h_sum : 10 + 18 + equal_pair + equal_pair = 60 :=
    by linarith
  have h_equal : 2 * equal_pair = 32 :=
    by linarith
  have h_equal_value : equal_pair = 16 :=
    by linarith
  show equal_pair * equal_pair = 256
    by sorry

end product_of_equal_numbers_l28_28852


namespace probability_of_selecting_specific_cubes_l28_28112

theorem probability_of_selecting_specific_cubes :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 88
  let total_pairs := (total_cubes * (total_cubes - 1)) / 2
  let successful_pairs := cubes_with_3_faces * cubes_with_no_faces
  by exact fractional.successful_pairs / total_pairs = 44 / 3875 := sorry

end probability_of_selecting_specific_cubes_l28_28112


namespace solve_problem_l28_28779

def ellipse_has_common_focus (C1 C2: Point → Prop) (F: Point) : Prop := 
  ∃ x1 x2 y1 y2, 
    C1 (x1, y1) ∧ C1 (x2, y2) ∧ 
    C2 (x1, y1) ∧ C2 (x2, y2) ∧ 
    focus_C1 = F ∧ focus_C2 = F

def find_line_equation (A B M: Point) (C2: Point → Prop) : Prop :=
  ∃ l: Line, 
    l passes_through M ∧
    ∃ A B, A ∈ l ∧ B ∈ l ∧ A ∈ C2 ∧ B ∈ C2 ∧ |M - B| = 4 * |A - M|

def find_min_major_axis (C1 C2: Point → Prop) (l: Line) : Prop :=
  ∃ a,  
    (∀ P, symmetric P (0,0) l → P ∈ C2) ∧
    (∃ P Q, P ∈ l ∧ Q ∈ l ∧ P ∈ C1 ∧ Q ∈ C1) ∧
    a = sqrt(34)

theorem solve_problem :
  ∀ C1 C2: Point → Prop,
  ∀ l: Line,
  ∀ F: Point,
  ∀ M A B: Point,
  ellipse_has_common_focus C1 C2 F ∧
  center_C1 = (0,0) ∧
  vertex_C2 = (0,0) ∧
  l passes_through M ∧
  A ∈ l ∧ B ∈ l ∧ A ∈ C2 ∧ B ∈ C2 ∧
  symmetric (0, 0) P l ∧
  (|M - B| = 4 * |A - M|) →
  find_line_equation A B M C2  ∧
  find_min_major_axis C1 C2 l.
by 
  sorry

end solve_problem_l28_28779


namespace volume_ratio_l28_28829

-- Definitions of points and tetrahedron.
def Point : Type := ℝ × ℝ × ℝ

structure Tetrahedron :=
(A B C D : Point)

structure Ratios :=
(BK_KC AM_MD CN_ND : ℝ)

-- Theorem statement
theorem volume_ratio {A B C D K M N : Point} (t : Tetrahedron) (r : Ratios) :
  r.BK_KC = 1/3 ∧ r.AM_MD = 3/1 ∧ r.CN_ND = 1/2 →
  (volume_divided_by_plane t K M N = 15 / 61) :=
sorry

end volume_ratio_l28_28829


namespace collinear_given_t_eq_2_perpendicular_given_s_eq_2_l28_28717

open Function

def pointA := (-2, 0, 2)
def pointB (t : ℝ) := (t - 2, 4, 3)
def pointC (s : ℝ) := (-4, s, 1)

def vectorA (t : ℝ) := (t, 4, 1)
def vectorB (s : ℝ) := (-2, s, -1)

theorem collinear_given_t_eq_2 (s : ℝ) : 
    let t := 2 in
    ∃ λ : ℝ, vectorA t = (λ • vectorB s) → s = -4 :=
by sorry

theorem perpendicular_given_s_eq_2 (t : ℝ) :
    let s := 2 in
    dot_product (λ (p1 p2 : (ℝ × ℝ × ℝ)), p1.1 * p2.1 + p1.2 * p2.2 + p1.3 * p2.3) 
      (vectorA (t - 2) + vectorB s) (vectorB s) = 0
      → t = 8 :=
by sorry

end collinear_given_t_eq_2_perpendicular_given_s_eq_2_l28_28717


namespace highest_percent_decrease_l28_28522

def CompanyA_initial_ratio : ℝ := 3 / 15
def CompanyA_subsequent_ratio : ℝ := 9 / 150
def CompanyB_initial_ratio : ℝ := 4 / 16
def CompanyB_subsequent_ratio : ℝ := 6.4 / 64
def CompanyC_initial_ratio : ℝ := 2.5 / 10
def CompanyC_subsequent_ratio : ℝ := 15 / 200

def percent_decrease (initial_ratio : ℝ) (subsequent_ratio : ℝ) : ℝ :=
  ((initial_ratio - subsequent_ratio) / initial_ratio) * 100

def CompanyA_percent_decrease : ℝ := percent_decrease CompanyA_initial_ratio CompanyA_subsequent_ratio
def CompanyB_percent_decrease : ℝ := percent_decrease CompanyB_initial_ratio CompanyB_subsequent_ratio
def CompanyC_percent_decrease : ℝ := percent_decrease CompanyC_initial_ratio CompanyC_subsequent_ratio

theorem highest_percent_decrease :
  (CompanyA_percent_decrease = 70 ∧ CompanyC_percent_decrease = 70) ∧ 
  (CompanyA_percent_decrease ≥ CompanyB_percent_decrease ∧ CompanyC_percent_decrease ≥ CompanyB_percent_decrease) := 
sorry

end highest_percent_decrease_l28_28522


namespace Sophie_donuts_l28_28473

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end Sophie_donuts_l28_28473


namespace max_intersection_points_l28_28910

theorem max_intersection_points (C L : Type) [fintype C] [fintype L] (hC : fintype.card C = 3) (hL : fintype.card L = 2) :
  (∑ c1 in C, ∑ c2 in (C \ {c1}), 2) + (∑ l in L, ∑ c in C, 2) + 1 = 19 :=
by
  sorry

end max_intersection_points_l28_28910


namespace sequence_general_term_l28_28715

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n : ℕ, S n = n^2 + 1) : 
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2 * n - 1) :=
by
  have a₁ : a 1 = S 1 := by sorry
  have SnSm : ∀ n, S n = n^2 + 1 := by exact hS
  have a_geq_2 : ∀ n ≥ 2, a n = S n - S (n - 1) := by sorry
  exact ⟨a₁, a_geq_2⟩

end sequence_general_term_l28_28715


namespace same_date_after_k_days_l28_28480

theorem same_date_after_k_days (k : ℕ) : (k > 0 ∧ (∀ y : ℕ, ∀ d : ℕ, d = 15 ∧ ¬(is_leap_year y) ∧ month_days y 2 = 28 ∧ (d - k = 1) ∧ (d + k - 28 = 1)) → k = 14) :=
begin
  sorry
end

end same_date_after_k_days_l28_28480


namespace donuts_left_for_sophie_l28_28477

def initial_boxes := 4
def donuts_per_box := 12
def boxes_given_to_mom := 1
def donuts_given_to_sister := 6

theorem donuts_left_for_sophie :
  let initial_donuts := initial_boxes * donuts_per_box in
  let remaining_boxes := initial_boxes - boxes_given_to_mom in
  let remaining_donuts := remaining_boxes * donuts_per_box in
  let donuts_left := remaining_donuts - donuts_given_to_sister in
  donuts_left = 30 := 
by 
  have initial_donuts := initial_boxes * donuts_per_box
  have remaining_boxes := initial_boxes - boxes_given_to_mom
  have remaining_donuts := remaining_boxes * donuts_per_box
  have donuts_left := remaining_donuts - donuts_given_to_sister
  show donuts_left = 30
  calc
  donuts_per_box * (initial_boxes - boxes_given_to_mom) - donuts_given_to_sister = donuts_per_box * 3 - donuts_given_to_sister := by sorry
  donuts_per_box * 3 - donuts_given_to_sister = 30 := by sorry

end donuts_left_for_sophie_l28_28477


namespace shots_cost_l28_28627

theorem shots_cost (n_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) :
  n_dogs = 3 →
  puppies_per_dog = 4 →
  shots_per_puppy = 2 →
  cost_per_shot = 5 →
  n_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    3 * 4 * 2 * 5 = 12 * 2 * 5 : by rfl
    ... = 24 * 5 : by rfl
    ... = 120 : by rfl

end shots_cost_l28_28627


namespace time_to_walk_without_walkway_l28_28081

theorem time_to_walk_without_walkway 
  (vp vw : ℝ) 
  (h1 : (vp + vw) * 40 = 80) 
  (h2 : (vp - vw) * 120 = 80) : 
  80 / vp = 60 :=
by
  sorry

end time_to_walk_without_walkway_l28_28081


namespace min_height_box_l28_28427

noncomputable def min_height (x : ℝ) : ℝ :=
  if h : x ≥ (5 : ℝ) then x + 5 else 0

theorem min_height_box (x : ℝ) (hx : 3*x^2 + 10*x - 65 ≥ 0) : min_height x = 10 :=
by
  sorry

end min_height_box_l28_28427


namespace rob_total_cards_l28_28837

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end rob_total_cards_l28_28837


namespace sum_of_roots_eq_14_l28_28912

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) → 11 + 3 = 14 :=
by
  intro h
  have x1 : 11 = 11 := rfl
  have x2 : 3 = 3 := rfl
  exact rfl

end sum_of_roots_eq_14_l28_28912


namespace square_area_l28_28207

theorem square_area (perimeter : ℝ) (h : perimeter = 40) : ∃ A : ℝ, A = 100 := by
  have h1 : ∃ s : ℝ, 4 * s = perimeter := by
    use perimeter / 4
    linarith

  cases h1 with s hs
  use s^2
  rw [hs, h]
  norm_num
  sorry

end square_area_l28_28207


namespace find_f_neg2_l28_28713

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem find_f_neg2 : f (-2) = -8 := by
  sorry

end find_f_neg2_l28_28713


namespace west_of_1km_l28_28012

def east_direction (d : Int) : Int :=
  d

def west_direction (d : Int) : Int :=
  -d

theorem west_of_1km :
  east_direction (2) = 2 →
  west_direction (1) = -1 := by
  sorry

end west_of_1km_l28_28012


namespace cricket_initial_avg_runs_l28_28486

theorem cricket_initial_avg_runs (A : ℝ) (h : 11 * (A + 4) = 10 * A + 86) : A = 42 :=
sorry

end cricket_initial_avg_runs_l28_28486


namespace regular_tetrahedron_volume_ratio_l28_28360

-- Defining the ratio of edge lengths and volume relationship.
def ratio_of_edge_lengths_is (a b : ℝ) (r : ℝ) : Prop := a / b = r

def volume_ratio (a b : ℝ) : ℝ := (a^3) / (b^3)

-- The main theorem to be proven.
theorem regular_tetrahedron_volume_ratio : 
  ∀ (a b : ℝ), ratio_of_edge_lengths_is a b 2 → volume_ratio a b = 1 / 8 :=
by
  intros a b h
  unfold ratio_of_edge_lengths_is at h
  unfold volume_ratio
  sorry

end regular_tetrahedron_volume_ratio_l28_28360


namespace sophie_donuts_left_l28_28475

theorem sophie_donuts_left :
  ∀ (boxes_initial : ℕ) (donuts_per_box : ℕ) (boxes_given_away : ℕ) (dozen : ℕ),
  boxes_initial = 4 →
  donuts_per_box = 12 →
  boxes_given_away = 1 →
  dozen = 12 →
  (boxes_initial - boxes_given_away) * donuts_per_box - (dozen / 2) = 30 :=
by 
  intros boxes_initial donuts_per_box boxes_given_away dozen 
  assume h1 h2 h3 h4
  sorry

end sophie_donuts_left_l28_28475


namespace exist_vectors_sum_of_squares_l28_28688

theorem exist_vectors_sum_of_squares 
  (a b c : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a * b ≥ c^2) :
  ∃ (n : ℕ) (x y : fin n → ℤ),
    (∑ i, x i ^ 2 = a) ∧
    (∑ i, y i ^ 2 = b) ∧
    (∑ i, x i * y i = c) :=
sorry

end exist_vectors_sum_of_squares_l28_28688


namespace f_3_eq_3_l28_28644

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then log 2 (8 - x) else sorry -- Recursive case will use f(x-1)

theorem f_3_eq_3 : f 3 = 3 := sorry

end f_3_eq_3_l28_28644


namespace root_in_interval_sum_eq_three_l28_28320

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end root_in_interval_sum_eq_three_l28_28320


namespace singer_arrangement_count_l28_28914

theorem singer_arrangement_count (S : Fin 5) (S₁ S₅ : Fin 5) (hS₁ : S₁ ≠ S 0) (hS₅ : S₅ ≠ S 4) :
  {p : Perm (Fin 5) | p 0 ≠ S₁ ∧ p 4 ≠ S₅}.card = 5! - 4! - 4! := 
by 
  sorry

end singer_arrangement_count_l28_28914


namespace event_type_random_event_type_certain_event_type_impossible_probability_even_l28_28879

def card_set := {1, 2, 3, 4, 5, 6}
def drawn_card := 3
def less_than_seven (n : ℕ) := n < 7
def less_than_zero (n : ℕ) := n < 0
def even_number (n : ℕ) := n % 2 = 0

theorem event_type_random :
  drawn_card ∈ card_set :=
by
  sorry

theorem event_type_certain :
  ∀ (n : ℕ), n ∈ card_set → less_than_seven n :=
by
  sorry

theorem event_type_impossible :
  ∀ (n : ℕ), n ∈ card_set → ¬less_than_zero n :=
by
  sorry

theorem probability_even :
  ∃ (p : ℚ), p = 1/2 :=
by
  sorry

end event_type_random_event_type_certain_event_type_impossible_probability_even_l28_28879


namespace triangle_side_length_l28_28388

theorem triangle_side_length :
  ∀ (a b c : ℝ) (B : ℝ),
  b = 3 ∧ c = sqrt 6 ∧ B = π / 3 →
  a = (sqrt 6 + 3 * sqrt 2) / 2 :=
by
  intros a b c B h,
  cases h with hb hcB,
  cases hcB with hc hB,
  sorry

end triangle_side_length_l28_28388


namespace alan_spent_total_amount_l28_28594

-- Conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation 
def cost_of_eggs : ℕ := eggs_bought * price_per_egg
def cost_of_chickens : ℕ := chickens_bought * price_per_chicken
def total_cost : ℕ := cost_of_eggs + cost_of_chickens

-- Proof statement
theorem alan_spent_total_amount : total_cost = 88 :=
by
  unfold total_cost cost_of_eggs cost_of_chickens
  simp [eggs_bought, price_per_egg, chickens_bought, price_per_chicken]
  sorry

end alan_spent_total_amount_l28_28594


namespace tiling_problem_l28_28844

theorem tiling_problem (b : ℕ) (hb : Even b) : ∃ M, ∀ n, n > M → (∃ tiling : Tiling, is_1_b_tileable tiling (2 * b) n) :=
by
  sorry

end tiling_problem_l28_28844


namespace cows_eat_all_grass_in_96_days_l28_28291

theorem cows_eat_all_grass_in_96_days :
  (∀ (cows days : ℕ) (growth_rate : ℝ), (cows = 70 ∧ days = 24) → (growth_rate = 1 / 480) → 
  (1 + days * growth_rate = cows * (days / 24))) → 
  (∀ (cows days : ℕ), (cows = 30 ∧ days = 60) → (1 + days * (1 / 480) = cows * (days / 60))) → 
  (∃ (cows : ℕ), cows * 96 * (1 / 1600) = 1 + 96 * (1 / 480) → cows = 20) :=
begin
  intros h1 h2,
  sorry
end

end cows_eat_all_grass_in_96_days_l28_28291


namespace max_intersection_points_circles_lines_l28_28906

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l28_28906


namespace total_sugar_needed_l28_28096

theorem total_sugar_needed (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
    sugar_frosting + sugar_cake = 0.8 :=
by
    rw [h1, h2]
    norm_num
    sorry

end total_sugar_needed_l28_28096


namespace min_value_expression_l28_28212

theorem min_value_expression (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
  ∃ y, y = (tan x + cot x)^2 + (sin x + csc x)^2 + (cos x + sec x)^2 ∧ y = 9 :=
by
  sorry

end min_value_expression_l28_28212


namespace b_arithmetic_sequence_minimum_lambda_l28_28722

section ArithmeticSequenceProblem

-- Definitions for sequences
def a : ℕ → ℚ
| 1 := 1/4
| (n + 1) := (1/4) * a n + (2/4^(n + 1))

def b (n : ℕ) : ℚ := 4^n * a n

-- Problem 1: Arithmetic nature of sequence {b_n}
theorem b_arithmetic_sequence : ∀ n : ℕ, b (n + 1) = b n + 2 :=
sorry -- Proof goes here

-- Definitions for sums
def S (n : ℕ) : ℚ := finset.sum (finset.range n) (λ k, a (k + 1))

-- Problem 2: Minimum value of λ
theorem minimum_lambda (λ : ℚ) : (∀ n : ℕ, n > 0 → S n + λ * n * a n ≥ 5 / 9) → λ ≥ 11 / 9 :=
sorry -- Proof goes here

end ArithmeticSequenceProblem

end b_arithmetic_sequence_minimum_lambda_l28_28722


namespace inequality_and_equality_condition_l28_28418

variable (a b c t : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_and_equality_condition :
  abc * (a^t + b^t + c^t) ≥ a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧ 
  (abc * (a^t + b^t + c^t) = a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ a = b ∧ b = c) :=
sorry

end inequality_and_equality_condition_l28_28418


namespace students_remaining_l28_28311

theorem students_remaining (initial_students : ℕ) (stops : ℕ)
    (get_off_fraction : ℚ) (h_initial : initial_students = 48)
    (h_stops : stops = 3) (h_fraction : get_off_fraction = 1 / 2) : 
    let remaining := initial_students * get_off_fraction^stops 
    in remaining = 6 :=
by
  sorry

end students_remaining_l28_28311


namespace no_equal_column_sums_l28_28777

def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def equal_column_sums_possible (numbers : List ℕ) (rows cols : ℕ) : Prop :=
  let total_sum := numbers.sum
  let target_sum := total_sum / cols
  total_sum % cols = 0 ∧ ∀ col_sum : ℕ, col_sum = target_sum

theorem no_equal_column_sums (numbers : List ℕ) (rows cols : ℕ) :
  numbers = List.range 30 ∧ rows = 5 ∧ cols = 6 →
  ¬ equal_column_sums_possible numbers rows cols :=
by
  intros h
  obtain ⟨hn, hr, hc⟩ := h
  have total_sum : (List.range 30).sum = 465 := by simp [sum_natural_numbers, List.range]
  have impossible_column_sum : ¬ 465 % 6 = 0 := by
    norm_num
  sorry

end no_equal_column_sums_l28_28777


namespace greatest_ratio_AB_CD_on_circle_l28_28186

/-- The statement proving the greatest possible value of the ratio AB/CD for points A, B, C, D lying on the 
circle x^2 + y^2 = 16 with integer coordinates and unequal distances AB and CD is sqrt 10 / 3. -/
theorem greatest_ratio_AB_CD_on_circle :
  ∀ (A B C D : ℤ × ℤ), A ≠ B → C ≠ D → 
  A.1^2 + A.2^2 = 16 → B.1^2 + B.2^2 = 16 → 
  C.1^2 + C.2^2 = 16 → D.1^2 + D.2^2 = 16 → 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let ratio := AB / CD
  AB ≠ CD →
  ratio ≤ Real.sqrt 10 / 3 :=
sorry

end greatest_ratio_AB_CD_on_circle_l28_28186


namespace max_rectangle_area_l28_28131

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l28_28131


namespace rob_total_cards_l28_28838

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end rob_total_cards_l28_28838


namespace commute_time_variance_l28_28577

theorem commute_time_variance (x y : ℝ) (h₁ : (x + y + 10 + 11 + 9) / 5 = 10) 
  (h₂ : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) : 
  x^2 + y^2 = 208 :=
begin
  sorry
end

end commute_time_variance_l28_28577


namespace find_perimeter_of_trapezoid_l28_28772

-- Definitions of the properties and lengths of trapezoid ABCD.
def is_trapezoid (A B C D : Type) [HasEquiv A] [HasEquiv B] [HasEquiv C] [HasEquiv D] :=
AD ≡ BC

def side_lengths (A B D : Type) [HasEquiv A] [HasEquiv B] [HasEquiv D] :=
(AB = 6) ∧ (AD = 9) ∧ (BD = 12)

def angle_equality (A B D C : Type) [HasEquiv A] [HasEquiv B] [HasEquiv D] [HasEquiv C] :=
(∠ABD = ∠DCB)

-- Statement of the proof problem
theorem find_perimeter_of_trapezoid :
  ∀ (A B C D : Type) [HasEquiv A] [HasEquiv B] [HasEquiv C] [HasEquiv D],
    is_trapezoid A B C D → side_lengths A B D → angle_equality A B D C → 
    (perimeter ABCD = 39) :=
by
  intros A B C D h_trapezoid h_lengths h_angles
  sorry

end find_perimeter_of_trapezoid_l28_28772


namespace frog_final_position_probability_l28_28957

noncomputable def probability_frog_final_position_exactly_one_meter :=
  let m := 1
  let jumps := 4
  let directions := random ℝ -- This represents the random directions
  let frog_final_position := λ n, sum (λ i, direction.random_vector (m, n), i from 1 to jumps)

theorem frog_final_position_probability :
  (probability (frog_final_position (directions).norm = 1)) = 1 / 8 :=
by
  sorry

end frog_final_position_probability_l28_28957


namespace ratio_side_lengths_l28_28113

-- Define the context for the problem.
variables {s_1 s_2 : ℝ} -- Side lengths of the two cubes
variable weight_1 : ℝ -- Weight of the first cube
variable weight_2 : ℝ -- Weight of the second cube
variable same_density : ℝ -- Assume same density for both cubes

-- Given conditions transformed to Lean definitions.
def first_cube_weight : Prop := weight_1 = 6
def second_cube_weight : Prop := weight_2 = 48
def cubes_same_density : Prop := same_density ≠ 0 ∧ (weight_1 = s_1^3 * same_density ∧ weight_2 = s_2^3 * same_density)

-- The mathematical statement to prove.
theorem ratio_side_lengths (h1 : first_cube_weight) (h2 : second_cube_weight) (h3 : cubes_same_density) : s_2 / s_1 = 2 :=
by
  sorry -- Proof omitted

end ratio_side_lengths_l28_28113


namespace PA_perpendicular_plane_ABCD_volume_of_pyramid_l28_28606

section PyramidProof

variables (A B C D P : Type) [Point A] [Point B] [Point C] [Point D] [Point P]

-- Define the conditions
def square_base : Prop :=
  side_length A B = 1 ∧ side_length A C = 1 ∧ side_length A D = 1 ∧ side_length B C = 1 ∧ side_length B D = 1 ∧ side_length C D = 1

def PA_perpendicular_CD : Prop := perpendicular PA CD

def PA_equals_1 : Prop := length PA = 1

-- To prove
theorem PA_perpendicular_plane_ABCD (h1 : square_base A B C D) (h2 : PA_perpendicular_CD A D P) (h3 : PA_equals_1 P A) : 
  perpendicular_to_plane PA A B C D :=
sorry

theorem volume_of_pyramid (h1 : square_base A B C D) (h2 : PA_perpendicular_CD A D P) (h3 : PA_equals_1 P A) : 
  volume P A B C D = 1/3 :=
sorry

end PyramidProof

end PA_perpendicular_plane_ABCD_volume_of_pyramid_l28_28606


namespace angle_C_eq_pi_over_4_l28_28328

variables {ℝ : Type*} [field ℝ] [ordered_ring ℝ]

/-- In triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively, M is a point on side AB. The conditions on the triangle and vectors are given, and we prove the value of angle C is π/4. -/
theorem angle_C_eq_pi_over_4
  (a b c : ℝ)
  (A B C : ℝ)
  (M P : ℝ × ℝ)
  (h1 : M = P + λ • P)
  (h2 : P = (A / abs A / cos A) + (B / abs B / cos B))
  (h3 : abs(CM) = c / 2)
  (h4 : a^2 + b^2 = 2 * sqrt 2 * a * b)
  (λ : ℝ) :
  C = π / 4 :=
by
  sorry

end angle_C_eq_pi_over_4_l28_28328


namespace probability_of_green_ball_l28_28530

theorem probability_of_green_ball :
  let P_A := 1 / 2
  let P_B := 3 / 10
  let P_C := 0
  let P_D := 3 / 5
  let P_container := 1 / 4
  P_container * P_A + P_container * P_B + P_container * P_C + P_container * P_D = 19 / 40 :=
by
  -- translate conditions into let bindings
  let P_A := 1 / 2
  let P_B := 3 / 10
  let P_C := 0
  let P_D := 3 / 5
  let P_container := 1 / 4
  have h : P_container * P_A + P_container * P_B + P_container * P_C + P_container * P_D = (1 / 4) * (1 / 2) + (1 / 4) * (3 / 10) + (1 / 4) * 0 + (1 / 4) * (3 / 5),
  sorry -- actual arithmetic transformations skipped

end probability_of_green_ball_l28_28530


namespace area_of_region_B_l28_28633

def region_B (z : ℂ) : Prop :=
  let z_div_50 := z / 50
  let fifty_div_conj_z := 50 / conj z
  let real_part_cond := 0 ≤ z_div_50.re ∧ z_div_50.re ≤ 1
  let imag_part_cond := 0 ≤ z_div_50.im ∧ z_div_50.im ≤ 1
  let real_part_div_cond := 0 ≤ (fifty_div_conj_z.re) ∧ (fifty_div_conj_z.re) ≤ 1
  let imag_part_div_cond := 0 ≤ (fifty_div_conj_z.im) ∧ (fifty_div_conj_z.im) ≤ 1
  real_part_cond ∧ imag_part_cond ∧ real_part_div_cond ∧ imag_part_div_cond

def area_of_B : ℝ := 3125 - 625 * Real.pi

theorem area_of_region_B : 
  (∃ z : ℂ, region_B z) → area_of_B = 3125 - 625 * Real.pi :=
sorry

end area_of_region_B_l28_28633


namespace exponentiation_rule_proof_l28_28992

-- Definitions based on conditions
def x : ℕ := 3
def a : ℕ := 4
def b : ℕ := 2

-- The rule that relates the exponents
def rule (x a b : ℕ) : ℕ := x^(a * b)

-- Proposition that we need to prove
theorem exponentiation_rule_proof : rule x a b = 6561 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end exponentiation_rule_proof_l28_28992


namespace angle_in_third_quadrant_l28_28554

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2014) : 180 < θ % 360 ∧ θ % 360 < 270 :=
by
  sorry

end angle_in_third_quadrant_l28_28554


namespace find_m_l28_28185

theorem find_m (m : ℝ) : 
  (10 ^ m = 10 ^ (-3) * real.sqrt (10 ^ 85 / 0.0001)) ↔ m = 41.5 :=
by sorry

end find_m_l28_28185


namespace al_wins_probability_l28_28379

variable (rock paper scissors : Type) [Fintype rock] [Fintype paper] [Fintype scissors]

def probability_of_al_winning (al_choice : rock) (bob_choice : paper) [decidable (bob_choice = rock ∨ bob_choice = paper ∨ bob_choice = scissors)] : ℚ :=
  if bob_choice = scissors then 1/3 else 0

theorem al_wins_probability (al_choice : rock) (bob_choice : paper) [decidable (bob_choice = rock ∨ bob_choice = paper ∨ bob_choice = scissors)] :
  probability_of_al_winning al_choice bob_choice = 1/3 :=
sorry

end al_wins_probability_l28_28379


namespace slower_time_l28_28811

-- Definitions for the problem conditions
def num_stories : ℕ := 50
def lola_time_per_story : ℕ := 12
def tara_time_per_story : ℕ := 10
def tara_stop_time : ℕ := 4
def tara_num_stops : ℕ := num_stories - 2 -- Stops on each floor except the first and last

-- Calculations based on the conditions
def lola_total_time : ℕ := num_stories * lola_time_per_story
def tara_total_time : ℕ := num_stories * tara_time_per_story + tara_num_stops * tara_stop_time

-- Target statement to be proven
theorem slower_time : tara_total_time = 692 := by
  sorry  -- Proof goes here (excluded as per instructions)

end slower_time_l28_28811


namespace find_second_discount_percentage_l28_28028

def initial_price : ℝ := 298.0
def first_discount_percentage : ℝ := 12.0
def final_sale_price : ℝ := 222.904

theorem find_second_discount_percentage :
  (∃ second_discount_percentage : ℝ,
  let first_discount := (first_discount_percentage / 100) * initial_price,
      price_after_first_discount := initial_price - first_discount,
      second_discount := price_after_first_discount - final_sale_price,
      second_discount_percentage_calculated := (second_discount / price_after_first_discount) * 100
  in second_discount_percentage_calculated = 15) :=
sorry

end find_second_discount_percentage_l28_28028


namespace problem_part_I_problem_part_II_l28_28329
open Real

noncomputable def Triangle := {a b c A B C : ℝ // 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ a > 0 ∧ b > 0 ∧ c > 0}

theorem problem_part_I (T : Triangle) (h : T.2.b^2 + T.2.c^2 = T.2.a^2 + T.2.b * T.2.c) : T.2.A = π / 3 := sorry

theorem problem_part_II (T : Triangle) (hcosB : T.2.B = acos (sqrt 6 / 3)) (hb : T.2.b = 2) 
  (h : T.2.b^2 + T.2.c^2 = T.2.a^2 + T.2.b * T.2.c) : 
  real.sqrt (1 - (cos T.2.B)^2) * T.2.b * T.2.c / 2 = (3*real.sqrt 2 + sqrt 3) / 2 := sorry

end problem_part_I_problem_part_II_l28_28329


namespace angle_between_vectors_is_pi_over_6_l28_28677

noncomputable def vector_a : ℝ × ℝ := (real.sqrt 3, 1)
noncomputable def vector_b : ℝ × ℝ := sorry -- Vector b is undetermined, only its magnitude and dot product is known
noncomputable def dot_product_ab : ℝ := real.sqrt 3
noncomputable def magnitude_b : ℝ := 1

theorem angle_between_vectors_is_pi_over_6
  (h1 : |vector_b| = magnitude_b)
  (h2 : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = dot_product_ab)
  : ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ real.pi ∧ θ = real.pi / 6 :=
sorry

end angle_between_vectors_is_pi_over_6_l28_28677


namespace regular_tetrahedron_volume_ratio_l28_28361

-- Defining the ratio of edge lengths and volume relationship.
def ratio_of_edge_lengths_is (a b : ℝ) (r : ℝ) : Prop := a / b = r

def volume_ratio (a b : ℝ) : ℝ := (a^3) / (b^3)

-- The main theorem to be proven.
theorem regular_tetrahedron_volume_ratio : 
  ∀ (a b : ℝ), ratio_of_edge_lengths_is a b 2 → volume_ratio a b = 1 / 8 :=
by
  intros a b h
  unfold ratio_of_edge_lengths_is at h
  unfold volume_ratio
  sorry

end regular_tetrahedron_volume_ratio_l28_28361


namespace angle_between_a_b_l28_28246

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
axiom norm_a : ∥a∥ = 1
axiom norm_a_plus_b : ∥a + b∥ = Real.sqrt 7
axiom dot_a_b_minus_a : inner a (b - a) = -4

-- Statement
theorem angle_between_a_b : angle a b = (5 * Real.pi) / 6 :=
by 
  -- Skipping the actual proof here
  sorry

end angle_between_a_b_l28_28246


namespace percent_reduction_hours_l28_28576

theorem percent_reduction_hours (W H : ℝ) (hW : W > 0) (hH : H > 0) :
  let new_wage := 1.10 * W
  let new_hours := H / 1.10
  let percent_reduction := (1 - new_hours / H) * 100
  percent_reduction = 9.09 :=
by
  let new_wage := 1.10 * W
  let new_hours := H / 1.10
  let percent_reduction := (1 - new_hours / H) * 100
  have h1 : 1 - (new_hours / H) = 1 - (H / 1.10 / H), by sorry
  have h2 : 1 - (H / 1.10 / H) = 1 - 1/1.10, by sorry
  have h3 : 1 - 1/1.10 ≈ (0.0909090909 : ℝ), by sorry
  have h4 : percent_reduction ≈ 9.09, by sorry
  exact h4

end percent_reduction_hours_l28_28576


namespace ratio_of_men_to_women_l28_28442
open Nat

theorem ratio_of_men_to_women 
  (total_players : ℕ) 
  (players_per_group : ℕ) 
  (extra_women_per_group : ℕ) 
  (H_total_players : total_players = 20) 
  (H_players_per_group : players_per_group = 3) 
  (H_extra_women_per_group : extra_women_per_group = 1) 
  : (7 / 13 : ℝ) = 7 / 13 :=
by
  -- Conditions
  have H1 : total_players = 20 := H_total_players
  have H2 : players_per_group = 3 := H_players_per_group
  have H3 : extra_women_per_group = 1 := H_extra_women_per_group
  -- The correct answer
  sorry

end ratio_of_men_to_women_l28_28442


namespace quotient_of_division_l28_28489

theorem quotient_of_division (a b : ℕ) (r q : ℕ) (h1 : a = 1637) (h2 : b + 1365 = a) (h3 : a = b * q + r) (h4 : r = 5) : q = 6 :=
by
  -- Placeholder for proof
  sorry

end quotient_of_division_l28_28489


namespace towel_area_decrease_l28_28971

theorem towel_area_decrease (L B : ℝ) (hL : 0 < L) (hB : 0 < B) :
  let A := L * B
  let L' := 0.80 * L
  let B' := 0.90 * B
  let A' := L' * B'
  A' = 0.72 * A →
  ((A - A') / A) * 100 = 28 :=
by
  intros _ _ _ _
  sorry

end towel_area_decrease_l28_28971


namespace curve_ne_parabola_l28_28736

theorem curve_ne_parabola (k : ℝ) : ¬ (∃ x y : ℝ, x^2 + k * y^2 = 1 ∧ is_parabola (x, y)) := 
sorry

end curve_ne_parabola_l28_28736


namespace min_distance_between_circles_l28_28016

-- Define the two circles with their radii and the distance between their centers.
variables (r R a : ℝ)
variable (h : a > r + R)

-- Statement: The minimum distance between any point on the first circle and any point on the second circle.
theorem min_distance_between_circles : 
  ∃ X Y, X ∈ circle (0, r) ∧ Y ∈ circle (0, R) ∧ dist X Y = a - r - R :=
sorry

end min_distance_between_circles_l28_28016


namespace probability_phone_not_answered_l28_28333

noncomputable def P_first_ring : ℝ := 0.1
noncomputable def P_second_ring : ℝ := 0.3
noncomputable def P_third_ring : ℝ := 0.4
noncomputable def P_fourth_ring : ℝ := 0.1

theorem probability_phone_not_answered : 
  1 - P_first_ring - P_second_ring - P_third_ring - P_fourth_ring = 0.1 := 
by
  sorry

end probability_phone_not_answered_l28_28333


namespace sphere_radius_is_half_l28_28193

-- Let r be the radius of each sphere
variable (r : ℝ)

-- Conditions:
-- 1. Eight congruent spheres arranged inside a cube of side 2 units
axiom sphere_configuration : (∀ i, 1 ≤ i ∧ i ≤ 8 → ∃ (x y z : ℝ), 
  (x = 0 ∨ x = 2) ∧ (y = 0 ∨ y = 2) ∧ (z = 0 ∨ z = 2) ∧ 
  (∀ j, 1 ≤ j ∧ j ≤ 8 → dist (x, y, z) (x', y', z') = 2 * r))

-- 2. Each sphere is tangent to 3 faces of the cube
axiom tangency_to_faces : (∀ i, 1 ≤ i ∧ i ≤ 8 → ∃ (x y z : ℝ), 
  (x = r ∨ x = 2 - r) ∧ (y = r ∨ y = 2 - r) ∧ (z = r ∨ z = 2 - r))

-- 3. Each sphere is tangent to its neighboring spheres
axiom tangency_to_neighbors : (∀ i j, (1 ≤ i ∧ i ≤ 8) → (1 ≤ j ∧ j ≤ 8) → 
  dist (x₁ i, y₁ i, z₁ i) (x₁ j, y₁ j, z₁ j) = 2 * r)

-- Problem Statement:
theorem sphere_radius_is_half : 
  r = 1/2 :=
sorry  -- Proof is not required

end sphere_radius_is_half_l28_28193


namespace product_in_1999th_day_l28_28888

/-- Define arithmetic mean of two numbers. -/
def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

/-- Define harmonic mean of two numbers. -/
def harmonic_mean (a b : ℝ) : ℝ := 2 / (1 / a + 1 / b)

/-- Define the transformation of the two numbers each day. -/
def transform (x y : ℝ) : ℝ × ℝ :=
  (arithmetic_mean x y, harmonic_mean x y)

/-- Initial numbers on the first day. -/
def initial_numbers : ℝ × ℝ := (1, 2)

/-- The main theorem stating the product of numbers in the evening of the 1999th day is 2. -/
theorem product_in_1999th_day : 
  let (x, y) := initial_numbers in
  (transform^[1999] (x, y)).1 * (transform^[1999] (x, y)).2 = 2 :=
sorry

end product_in_1999th_day_l28_28888


namespace total_shots_cost_l28_28626

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l28_28626


namespace max_PA_minus_PB_l28_28866

-- Define the points A, B, and B'
noncomputable def A : ℝ × ℝ := (4, 0)
noncomputable def B : ℝ × ℝ := (0, 3)
noncomputable def B' : ℝ × ℝ := (0, -3)

-- Define the line equations involved
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := y = x + 1

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem in Lean 4 style
theorem max_PA_minus_PB (P : ℝ × ℝ) (hP : line2 P.1 P.2) :
  ∃ L, L = distance P A - distance P B ∧ L ≤ 5 :=
sorry

end max_PA_minus_PB_l28_28866


namespace winning_player_analysis_l28_28391

/-- Define the game dynamics for Irene and Laura according to the rules given -/
inductive Player
| Irene
| Laura
deriving DecidableEq, Repr

def winning_player (n : ℕ) : Player :=
  if n % 6 = 3 then Player.Irene else Player.Laura

/-- Theorem: Determine the winning player for the game starting with 2015 or 2016 coins -/
theorem winning_player_analysis (n : ℕ) :
  (n = 2015 → winning_player n = Player.Irene) ∧
  (n = 2016 → winning_player n = Player.Laura) :=
by
  split
  { intro h
    rw h
    simp [winning_player, Nat.mod_eq_of_lt]
    apply Nat.mod_eq_of_lt
    norm_num }
  { intro h
    rw h
    simp [winning_player, Nat.mod_eq_of_lt]
    apply Nat.mod_eq_of_lt
    norm_num }

end winning_player_analysis_l28_28391


namespace max_sequence_sum_le_253009_l28_28801

noncomputable def sequenceSumMaximum (a : ℕ → ℝ) : ℝ :=
  if h : ∑ i in range 2013, a i = 0 ∧ a 1 = 0 ∧ a 2013 = 0 ∧ (∀ i, 1 ≤ i ∧ i ≤ 2012 → |a i - a (i + 1)| < 1) then
    (finset.range 2013).sup (λ m, ∑ i in finset.range m, a i)
  else
    0

theorem max_sequence_sum_le_253009 (a : ℕ → ℝ) (h_sum : ∑ i in range 2013, a i = 0)
  (h_start_end : a 1 = 0 ∧ a 2013 = 0)
  (h_diff : ∀ i, 1 ≤ i ∧ i ≤ 2012 → |a i - a (i + 1)| < 1) :
  sequenceSumMaximum a ≤ 253009 :=
sorry

end max_sequence_sum_le_253009_l28_28801


namespace fill_dominoes_odd_l28_28514

def upper_half_plane : Type := ℕ × ℤ  -- Define the upper half-plane

def is_domino (c1 c2 : upper_half_plane) : Prop :=
  ((c1.1 = c2.1) ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  ((c1.2 = c2.2) ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

def non_overlapping (dominoes : set (upper_half_plane × upper_half_plane)) : Prop :=
  ∀ d1 d2 ∈ dominoes, d1 ≠ d2 → d1.1 ≠ d2.1 ∧ d2.1 ≠ d1.2  -- No two dominoes overlap

def odd_filled (dominoes : set (upper_half_plane × upper_half_plane)) : Prop :=
  ∀ i : ℕ, (set.count (λ c → ∃ d ∈ dominoes, c = d.1 ∨ c = d.2) ({(i, j) | j ∈ ℤ})) % 2 = 1 ∧
  ∀ j : ℤ, (set.count (λ c → ∃ d ∈ dominoes, c = d.1 ∨ c = d.2) ({(i, j) | i ∈ ℕ})) % 2 = 1

theorem fill_dominoes_odd (dominoes : set (upper_half_plane × upper_half_plane)) :
  (∃ dom, non_overlapping dom ∧ odd_filled dom) ↔ true := sorry

end fill_dominoes_odd_l28_28514


namespace area_of_region_B_l28_28636

theorem area_of_region_B : 
  let B := { z : ℂ | (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1) ∧ 
                   (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
                    0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1) } in
  ∫ (z in B), 1 = 2500 - 312.5 * Real.pi := 
by
  sorry

end area_of_region_B_l28_28636


namespace combined_average_score_l28_28142

/-- 
Given: 
1. The average scores for 3 classes U, B, and C are 65, 80, and 77 respectively.
2. The ratio of the number of students in each class who took the test was 4 to 6 to 5.

Prove: 
The average score for the 3 classes combined is 75.
-/
theorem combined_average_score (x : ℕ) : 
  let students_U := 4 * x,
      students_B := 6 * x,
      students_C := 5 * x,
      total_students := students_U + students_B + students_C,
      total_score_U := 65 * students_U,
      total_score_B := 80 * students_B,
      total_score_C := 77 * students_C,
      combined_score := total_score_U + total_score_B + total_score_C
  in (combined_score / total_students) = 75 :=
by
  sorry

end combined_average_score_l28_28142


namespace midpoint_trajectory_is_quarter_circle_l28_28119

noncomputable def trajectory_midpoint_of_segment (L : ℝ) : set (ℝ × ℝ) :=
  { M : ℝ × ℝ | ∃ (A B : ℝ × ℝ), (A.1 = 0 ∧ B.2 = 0) ∧ (dist A B = L) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) }

theorem midpoint_trajectory_is_quarter_circle (L : ℝ) :
  ∀ M ∈ trajectory_midpoint_of_segment L, dist M (0, 0) = L / 2 :=
by
  sorry

end midpoint_trajectory_is_quarter_circle_l28_28119


namespace complex_quadrant_problem_l28_28653

-- Given conditions as Lean definitions
noncomputable def z1 : ℂ := 2 * complex.exp (complex.I * (real.pi / 3))
noncomputable def z2 : ℂ := complex.exp (complex.I * (real.pi / 2))

-- The proposition to prove that z lies in the fourth quadrant
theorem complex_quadrant_problem :
  let z := z1 / z2 in 
  z.im < 0 ∧ z.re > 0 :=
by
  -- The statement here ensures that we are in the fourth quadrant:
  -- z has a negative imaginary part and positive real part
  sorry

end complex_quadrant_problem_l28_28653


namespace simplify_fraction_multiplication_l28_28845

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end simplify_fraction_multiplication_l28_28845


namespace ratio_books_l28_28920

-- Definitions based on the conditions
def william_books_last_month : ℕ := 6
def brad_books_last_month : ℕ := 3 * william_books_last_month
def brad_books_this_month : ℕ := 8
def brad_books_two_months : ℕ := brad_books_last_month + brad_books_this_month
def william_books_two_months : ℕ := brad_books_two_months + 4
def william_books_this_month : ℕ := william_books_two_months - william_books_last_month

-- The final proof that the ratio is 3:1
theorem ratio_books :
  (william_books_this_month : brad_books_this_month) = 3 :=
by
  sorry

end ratio_books_l28_28920


namespace remaining_amount_proof_l28_28827

def hourly_wage_main_job := 15.75
def hours_worked_main_job := 45
def hourly_wage_part_time_job := 13.25
def hours_worked_part_time_job := 20
def tax_rate_main_job_first_40 := 0.22
def tax_rate_main_job_above_40 := 0.25
def tax_rate_part_time_job := 0.18
def misc_fee_rate_main_job := 0.10
def spending_rate_gummy_bears := 0.20
def spending_rate_phone_bill := 0.05
def saving_rate_vacation := 0.30
def expected_remaining_amount := 365.33

theorem remaining_amount_proof :
  let main_job_earnings := hourly_wage_main_job * hours_worked_main_job in
  let tax_main_job_first_40 := (min hours_worked_main_job 40) * hourly_wage_main_job * tax_rate_main_job_first_40 in
  let tax_main_job_above_40 := (hours_worked_main_job - min hours_worked_main_job 40) * hourly_wage_main_job * tax_rate_main_job_above_40 in
  let total_tax_main_job := tax_main_job_first_40 + tax_main_job_above_40 in
  let net_earnings_main_job := main_job_earnings - total_tax_main_job in
  let fee_main_job := main_job_earnings * misc_fee_rate_main_job in
  let net_after_fee_main_job := net_earnings_main_job - fee_main_job in
  let part_time_job_earnings := hourly_wage_part_time_job * hours_worked_part_time_job in
  let tax_part_time_job := part_time_job_earnings * tax_rate_part_time_job in
  let net_earnings_part_time_job := part_time_job_earnings - tax_part_time_job in
  let total_net_earnings := net_after_fee_main_job + net_earnings_part_time_job in
  let spending_gummy_bears := total_net_earnings * spending_rate_gummy_bears in
  let spending_phone_bill := total_net_earnings * spending_rate_phone_bill in
  let saving_vacation := (total_net_earnings - spending_gummy_bears - spending_phone_bill) * saving_rate_vacation in
  let remaining_amount := total_net_earnings - spending_gummy_bears - spending_phone_bill - saving_vacation in
  remaining_amount = expected_remaining_amount :=
sorry

end remaining_amount_proof_l28_28827


namespace num_five_digit_integers_l28_28048

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end num_five_digit_integers_l28_28048


namespace intersection_M_N_l28_28029

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_M_N : M ∩ N = { (1, 0) } := by
  sorry

end intersection_M_N_l28_28029


namespace probability_at_least_one_B_l28_28036

-- Definitions of conditions
constant total_questions : ℕ := 5
constant type_A : ℕ := 2
constant type_B : ℕ := 3
constant selected_questions : ℕ := 2

-- The proof goal statement
theorem probability_at_least_one_B :
  (↑(type_B * (total_questions - type_A)) / (total_questions * (total_questions - 1) / 2)) = 9 / 10 :=
by
  sorry

end probability_at_least_one_B_l28_28036


namespace four_is_integer_l28_28145

-- Math problem: Prove that 4 is an integer given conditions.

theorem four_is_integer (hnat_is_int: ∀ n: ℕ, ∃ i: ℤ, i = n) (h : ∃ n: ℕ, n = 4) : ∃ i: ℤ, i = 4 :=
by
  -- Definitions from conditions can be used directly in the Lean code here.
  cases h with n hn,
  rw hn,
  apply hnat_is_int,
  -- We can now skip the proof for simplification
  sorry

end four_is_integer_l28_28145


namespace skew_lines_angle_equal_dihedral_angle_l28_28260

-- Define the plane angle of the dihedral angle
def dihedral_angle_plane (α β : ℝ) : Prop := α = 45 ∧ β = 45

-- Define the perpendicularity of the lines to the planes
def perpendicular_to_plane (a b : ℝ) : Prop := 
∀ α β : ℝ, dihedral_angle_plane α β → angle a b = α

-- State the theorem
theorem skew_lines_angle_equal_dihedral_angle (a b : ℝ) (α β : ℝ) 
  (h1 : dihedral_angle_plane α β)
  (h2 : perpendicular_to_plane a b) : 
  angle a b = 45 :=
by
  sorry

end skew_lines_angle_equal_dihedral_angle_l28_28260


namespace hcf_of_two_numbers_l28_28889

theorem hcf_of_two_numbers (A B H L : ℕ) (h1 : A * B = 1800) (h2 : L = 200) (h3 : A * B = H * L) : H = 9 :=
by
  sorry

end hcf_of_two_numbers_l28_28889


namespace probability_mixed_committee_l28_28010

theorem probability_mixed_committee :
  let total_ways := Nat.choose 24 6
  let all_boys := Nat.choose 12 6
  let all_girls := Nat.choose 12 6
  let excl_ways := 2 * all_boys
  let mixed_prob := 1 - (excl_ways.to_rat / total_ways.to_rat)
  mixed_prob = (33187 / 33649 : ℚ) :=
by
  sorry

end probability_mixed_committee_l28_28010


namespace pirates_total_coins_l28_28444

theorem pirates_total_coins :
  ∀ (x : ℕ), (x * (x + 1)) / 2 = 5 * x → 6 * x = 54 :=
by
  intro x
  intro h
  -- proof omitted
  sorry

end pirates_total_coins_l28_28444


namespace triangle_inequality_real_l28_28938

theorem triangle_inequality_real:
  ∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
  sqrt (a / (b + c)) + sqrt (b / (a + c)) + sqrt (c / (a + b)) ≥ 2 :=
by
  sorry

end triangle_inequality_real_l28_28938


namespace fraction_eq_zero_implies_x_eq_one_l28_28323

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end fraction_eq_zero_implies_x_eq_one_l28_28323


namespace william_earnings_correct_l28_28921

def william_earnings : ℝ := 164.50

def charges : {norm_car_charge : ℝ, suv_charge : ℝ, minivan_charge : ℝ} :=
  { norm_car_charge := 15, suv_charge := 25, minivan_charge := 20 }

def promotion_discount : ℝ := 0.10

def vehicles_to_wash : {norm_car_count : ℕ, suv_count : ℕ, minivan_count : ℕ} :=
  { norm_car_count := 3, suv_count := 2, minivan_count := 1 }

def vehicle_washing_times : {norm_car_time : ℝ, suv_time_factor : ℝ, minivan_time_factor : ℝ} :=
  { norm_car_time := 1, suv_time_factor := 2, minivan_time_factor := 1.5 }

def customer_vehicles : {norm_car_count : ℕ, suv_count : ℕ} :=
  { norm_car_count := 2, suv_count := 1 }

theorem william_earnings_correct :
  let
    total_without_promotion := vehicles_to_wash.norm_car_count * charges.norm_car_charge +
                               vehicles_to_wash.suv_count * charges.suv_charge +
                               vehicles_to_wash.minivan_count * charges.minivan_charge,
    customer_total_before_discount := customer_vehicles.norm_car_count * charges.norm_car_charge +
                                      customer_vehicles.suv_count * charges.suv_charge,
    discount := promotion_discount * customer_total_before_discount,
    customer_total_after_discount := customer_total_before_discount - discount,
    total_amount := total_without_promotion + customer_total_after_discount
  in
    total_amount = william_earnings := 
  by
    sorry

end william_earnings_correct_l28_28921


namespace hcf_fractions_l28_28052

theorem hcf_fractions (hcf_all : HCF [4/9, 6/18, 0.1111111111111111] = 0.1111111111111111) : 
  HCF [4/9, 6/18] = 1/9 :=
by
  have hcf_simplified_fractions : 0.1111111111111111 = 1/9 := by norm_num
  rw ← hcf_simplified_fractions at hcf_all
  exact sorry

end hcf_fractions_l28_28052


namespace expression_equals_8_l28_28613

-- Define the expression we are interested in.
def expression : ℚ :=
  (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7)

-- Statement we need to prove
theorem expression_equals_8 : expression = 8 := by
  sorry

end expression_equals_8_l28_28613


namespace circle_radius_of_inscribed_rhombus_l28_28662

theorem circle_radius_of_inscribed_rhombus (d1 d2 a_square : ℝ) (h₁ : d1 = 12) (h₂ : d2 = 16) (h₃ : a_square = 20) :
  let a_rhombus := (sqrt ((d1/2)^2 + (d2/2)^2)) in
  let area_rhombus := (d1 * d2) / 2 in
  let side_rhombus := a_rhombus in
  let r := area_rhombus / (2 * side_rhombus) in
  r = 4.8 :=
by
  sorry

end circle_radius_of_inscribed_rhombus_l28_28662


namespace Sophie_donuts_l28_28471

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end Sophie_donuts_l28_28471


namespace symmetric_circle_eqn_l28_28858

theorem symmetric_circle_eqn (x y : ℝ) :
  (∃ (x0 y0 : ℝ), (x - 2)^2 + (y - 2)^2 = 7 ∧ x + y = 2) → x^2 + y^2 = 7 :=
by
  sorry

end symmetric_circle_eqn_l28_28858


namespace problem_statement_l28_28605

def approx_digit_place (num : ℕ) : ℕ :=
if num = 3020000 then 0 else sorry

theorem problem_statement :
  approx_digit_place (3 * 10^6 + 2 * 10^4) = 0 :=
by
  sorry

end problem_statement_l28_28605


namespace least_deletions_to_square_l28_28461

theorem least_deletions_to_square (l : List ℕ) (h : l = [10, 20, 30, 40, 50, 60, 70, 80, 90]) : 
  ∃ d, d.card ≤ 2 ∧ ∀ (lp : List ℕ), lp = l.diff d → 
  ∃ k, lp.prod = k^2 :=
by
  sorry

end least_deletions_to_square_l28_28461


namespace sum_of_angles_of_circumscribed_quadrilateral_l28_28108

theorem sum_of_angles_of_circumscribed_quadrilateral
  (EF GH : ℝ)
  (EF_central_angle : EF = 100)
  (GH_central_angle : GH = 120) :
  (EF / 2 + GH / 2) = 70 :=
by
  sorry

end sum_of_angles_of_circumscribed_quadrilateral_l28_28108


namespace trapezoid_longer_side_length_l28_28141

/-- Given a square of side length 2, divided into two congruent trapezoids and a pentagon
    such that each shape has equal area, prove that the length of the longer parallel side of each trapezoid is 5/3. -/
theorem trapezoid_longer_side_length :
  ∀ (a : ℝ) (P Q R : ℝ), 
  let square_area := 4 in
  let trapezoid_area := square_area / 3 in
  let side_length := 2 in
  let half_side := side_length / 2 in
  a = side_length ∧
  P = half_side ∧
  Q = half_side ∧
  R = half_side ∧
  let trapezoid_side := λ y : ℝ, (1 / 2) * (y + 1) * 1 in
  trapezoid_side y = trapezoid_area →
  y = 5 / 3 :=
begin 
  intros a P Q R square_area trapezoid_area side_length half_side,
  simp,
  intros h₁ h₂ h₃ h₄ trapezoid_side h_area,
  sorry -- proof goes here
end

end trapezoid_longer_side_length_l28_28141


namespace nate_matches_left_l28_28437

def initial_matches : ℕ := 70
def matches_dropped : ℕ := 10
def matches_eaten : ℕ := 2 * matches_dropped
def total_matches_lost : ℕ := matches_dropped + matches_eaten
def remaining_matches : ℕ := initial_matches - total_matches_lost

theorem nate_matches_left : remaining_matches = 40 := by
  sorry

end nate_matches_left_l28_28437


namespace max_intersection_points_three_circles_two_lines_l28_28900

theorem max_intersection_points_three_circles_two_lines : 
  ∀ (C1 C2 C3 L1 L2 : set ℝ × ℝ) (hC1 : is_circle C1) (hC2 : is_circle C2) (hC3 : is_circle C3) (hL1 : is_line L1) (hL2 : is_line L2),
  ∃ P : ℕ, P = 19 ∧
  (∀ (P : ℝ × ℝ), P ∈ C1 ∧ P ∈ C2 ∨ P ∈ C1 ∧ P ∈ C3 ∨ P ∈ C2 ∧ P ∈ C3 ∨ P ∈ C1 ∧ P ∈ L1 ∨ P ∈ C2 ∧ P ∈ L1 ∨ P ∈ C3 ∧ P ∈ L1 ∨ P ∈ C1 ∧ P ∈ L2 ∨ P ∈ C2 ∧ P ∈ L2 ∨ P ∈ C3 ∧ P ∈ L2 ∨ P ∈ L1 ∧ P ∈ L2) ↔ P = 19 :=
sorry

end max_intersection_points_three_circles_two_lines_l28_28900


namespace negation_of_square_positive_l28_28868

open Real

-- Define the original proposition
def prop_square_positive : Prop :=
  ∀ x : ℝ, x^2 > 0

-- Define the negation of the original proposition
def prop_square_not_positive : Prop :=
  ∃ x : ℝ, ¬ (x^2 > 0)

-- The theorem that asserts the logical equivalence for the negation
theorem negation_of_square_positive :
  ¬ prop_square_positive ↔ prop_square_not_positive :=
by sorry

end negation_of_square_positive_l28_28868


namespace terminal_side_second_quadrant_l28_28690

theorem terminal_side_second_quadrant (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end terminal_side_second_quadrant_l28_28690


namespace deepak_present_age_l28_28931

/-- Let Rahul and Deepak's current ages be 4x and 3x respectively
  Given that:
  1. The ratio between Rahul and Deepak's ages is 4:3
  2. After 6 years, Rahul's age will be 26 years
  Prove that Deepak's present age is 15 years.
-/
theorem deepak_present_age (x : ℕ) (hx : 4 * x + 6 = 26) : 3 * x = 15 :=
by
  sorry

end deepak_present_age_l28_28931


namespace exists_infinitely_many_non_clean_l28_28790

def is_clean (S : Set ℕ) (n : ℕ) : Prop :=
  ∃! (A : Finset ℕ), (A ⊆ S ∧ A.card % 2 = 1 ∧ A.sum = n)

theorem exists_infinitely_many_non_clean (S : Set ℕ) (hS : S.Nonempty) :
  ∃ᶠ n in at_top, ¬ is_clean S n :=
sorry

end exists_infinitely_many_non_clean_l28_28790


namespace shaded_percentage_seven_by_seven_grid_l28_28072

theorem shaded_percentage_seven_by_seven_grid :
  let total_squares := 49
  let shaded_squares := 7
  let shaded_fraction := shaded_squares / total_squares
  let shaded_percentage := shaded_fraction * 100
  shaded_percentage = 14.29 := by
  sorry

end shaded_percentage_seven_by_seven_grid_l28_28072


namespace ann_needs_42_more_toothpicks_l28_28982

def toothpicks_required_for_steps (n : Nat) (initial_steps : Nat) (initial_toothpicks : Nat) (step_increase : Nat → Nat → Nat) : Nat :=
  ∑ i in Finset.range (n - initial_steps), step_increase i (initial_toothpicks + i * 2)

def step_increase_rule (i : Nat) (base_increase : Nat) : Nat :=
  base_increase + 2 * i

theorem ann_needs_42_more_toothpicks :
  toothpicks_required_for_steps 7 4 (12) step_increase_rule = 42 :=
by
  sorry

end ann_needs_42_more_toothpicks_l28_28982


namespace circumcircle_through_fixed_point_l28_28126

theorem circumcircle_through_fixed_point
  (A B C D K L Q : Type)
  [is_triangle A B C]
  [D_on_AC : lies_on D (segment A C)]
  [ray_l_intersects_AC_at_K : ray B intersects_segment A C at K]
  [ray_l_intersects_circumcircle_at_L : ray B intersects_circumcircle (triangle A B C) at L]
  [circumcircle_DKL_through_Q : circumscribes (triangle D K L) through Q] :
  ∃ Q, (lies_on Q (circumcircle_of_triangle D K L)) ∧ Q ≠ D ∧ ∀ l, independent_of_ray l := 
sorry

end circumcircle_through_fixed_point_l28_28126


namespace area_of_region_B_l28_28632

def region_B (z : ℂ) : Prop :=
  let z_div_50 := z / 50
  let fifty_div_conj_z := 50 / conj z
  let real_part_cond := 0 ≤ z_div_50.re ∧ z_div_50.re ≤ 1
  let imag_part_cond := 0 ≤ z_div_50.im ∧ z_div_50.im ≤ 1
  let real_part_div_cond := 0 ≤ (fifty_div_conj_z.re) ∧ (fifty_div_conj_z.re) ≤ 1
  let imag_part_div_cond := 0 ≤ (fifty_div_conj_z.im) ∧ (fifty_div_conj_z.im) ≤ 1
  real_part_cond ∧ imag_part_cond ∧ real_part_div_cond ∧ imag_part_div_cond

def area_of_B : ℝ := 3125 - 625 * Real.pi

theorem area_of_region_B : 
  (∃ z : ℂ, region_B z) → area_of_B = 3125 - 625 * Real.pi :=
sorry

end area_of_region_B_l28_28632


namespace overall_percentage_increase_l28_28500

-- Define the monthly profit factors
def profit_factor (factors: List Real) : Real :=
  factors.foldl (*) 1

-- Define the conditions as factors
def march_to_april := 1 + 35 / 100
def april_to_may := 1 - 20 / 100
def may_to_june := 1 + 50 / 100
def june_to_july := 1 - 25 / 100
def july_to_august := 1 + 45 / 100

-- List all factors
def monthly_factors := [march_to_april, april_to_may, may_to_june, june_to_july, july_to_august]

-- Calculate the overall factor
def overall_factor := profit_factor monthly_factors

-- Define the expected overall increase
def expected_increase := 21.95 / 100

-- Proof statement for Lean
theorem overall_percentage_increase :
  overall_factor - 1 = expected_increase :=
by
  sorry

end overall_percentage_increase_l28_28500


namespace calc_expr_result_l28_28617

noncomputable def calc_expr : ℝ :=
  real.sqrt (real.sqrt 3) - 2 +
  real.sqrt ((-3) ^ 2).to_real +
  real.sqrt 3

theorem calc_expr_result : calc_expr = 2 * real.sqrt 3 - 1 := by
  sorry

end calc_expr_result_l28_28617


namespace inequality_transformation_l28_28225

theorem inequality_transformation (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by {
  sorry
}

end inequality_transformation_l28_28225


namespace price_reduction_l28_28099

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l28_28099


namespace solve_for_x_l28_28005

-- Define the equation as a predicate
def equation (x : ℝ) : Prop := (0.05 * x + 0.07 * (30 + x) = 15.4)

-- The proof statement
theorem solve_for_x :
  ∃ x : ℝ, equation x ∧ x = 110.8333 :=
by
  existsi (110.8333 : ℝ)
  split
  sorry -- Proof of the equation
  rfl -- Equality proof

end solve_for_x_l28_28005


namespace min_max_ineq_l28_28453

theorem min_max_ineq
  {n : ℕ} {a b : Fin n → ℝ}
  (hapos : ∀ k, 0 < a k)
  (hbpos : ∀ k, 0 < b k)
  : (Finset.univ.inf (λ k, a k / b k) : ℝ) ≤ (Fin.sum Finset.univ a) / (Fin.sum Finset.univ b) ∧ (Fin.sum Finset.univ a) / (Fin.sum Finset.univ b) ≤ (Finset.univ.sup (λ k, a k / b k) : ℝ) := 
sorry

end min_max_ineq_l28_28453


namespace probability_neither_l28_28024

variable (P : Set ℕ → ℝ) -- Use ℕ as a placeholder for the event space
variables (A B : Set ℕ)
variables (hA : P A = 0.25) (hB : P B = 0.35) (hAB : P (A ∩ B) = 0.15)

theorem probability_neither :
  P (Aᶜ ∩ Bᶜ) = 0.55 :=
by
  sorry

end probability_neither_l28_28024


namespace find_all_a_l28_28657

def digit_sum_base_4038 (n : ℕ) : ℕ :=
  n.digits 4038 |>.sum

def is_good (n : ℕ) : Prop :=
  2019 ∣ digit_sum_base_4038 n

def is_bad (n : ℕ) : Prop :=
  ¬ is_good n

def satisfies_condition (seq : ℕ → ℕ) (a : ℝ) : Prop :=
  (∀ n, seq n ≤ a * n) ∧ ∀ n, seq n = seq (n + 1) + 1

theorem find_all_a (a : ℝ) (h1 : 1 ≤ a) :
  (∀ seq, (∀ n m, n ≠ m → seq n ≠ seq m) → satisfies_condition seq a →
    ∃ n_infinitely, is_bad (seq n_infinitely)) ↔ a < 2019 := sorry

end find_all_a_l28_28657


namespace represent_2015_l28_28460

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def in_interval (n : ℕ) : Prop := 400 < n ∧ n < 500

def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem represent_2015 :
  ∃ a b c : ℕ,
  a + b + c = 2015 ∧
  is_prime a ∧
  is_divisible_by_3 b ∧
  in_interval c ∧
  not_divisible_by_3 c :=
by {
  use 7,
  use 1605,
  use 403,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
}

end represent_2015_l28_28460


namespace Paige_homework_problems_left_l28_28826

theorem Paige_homework_problems_left (math_problems : ℕ) (science_problems : ℕ) 
(history_problems : ℕ) (language_arts_problems : ℕ) 
(finished_problems : ℕ) (unfinished_math : ℕ) :
  math_problems = 43 → 
  science_problems = 12 →
  history_problems = 10 →
  language_arts_problems = 5 →
  finished_problems = 44 →
  unfinished_math = 3 →
  (math_problems + science_problems + history_problems + language_arts_problems) - finished_problems + unfinished_math = 29 := 
by 
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  simp
  sorry

end Paige_homework_problems_left_l28_28826


namespace exists_c_infinitely_many_n_and_unrepresentable_integers_l28_28451

theorem exists_c_infinitely_many_n_and_unrepresentable_integers :
  ∃ c > 0, ∃ᶠ n in Filter.atTop, ∃ᶠ k in Filter.atTop, 
    ∀ m, m < c * n * Real.log n → ¬ IsCoprimeSum k n m := 
sorry

end exists_c_infinitely_many_n_and_unrepresentable_integers_l28_28451


namespace masha_can_ensure_cherry_in_2_bites_l28_28818

-- Define the types for pies
inductive PieType
| Rice
| Cabbage
| Cherry

-- Given the initial conditions
def pies : List PieType := [PieType.Rice, PieType.Rice, PieType.Rice, PieType.Cabbage, PieType.Cabbage, PieType.Cabbage, PieType.Cherry]

-- Define what it means to ensure eating the cherry pie
def ensure_cherry (tries: Nat) : Prop :=
  ∃ i : Nat, i < tries ∧ pies.rotate i == PieType.Cherry :: _

-- Define the main theorem
theorem masha_can_ensure_cherry_in_2_bites : ensure_cherry 2 :=
sorry

end masha_can_ensure_cherry_in_2_bites_l28_28818


namespace percent_decrease_correct_l28_28401

namespace OrangeJuice

-- Define the conditions
def last_month_price_per_bottle := 8 / 6
def this_month_price_per_bottle := 6 / 8

-- Define the main statement for the percent decrease
def percent_decrease (old_price new_price : ℝ) : ℝ :=
  (old_price - new_price) / old_price * 100

-- Define the statement we want to prove
theorem percent_decrease_correct :
  percent_decrease last_month_price_per_bottle this_month_price_per_bottle = 58.33 :=
by
  sorry

end OrangeJuice

end percent_decrease_correct_l28_28401


namespace can_increase_average_l28_28337

def student_grades := List (String × Nat)

def group1 : student_grades := 
    [("Andreev", 5), ("Borisova", 3), ("Vasilieva", 5), ("Georgiev", 3),
     ("Dmitriev", 5), ("Evstigneeva", 4), ("Ignatov", 3), ("Kondratiev", 4),
     ("Leontieva", 3), ("Mironov", 4), ("Nikolaeva", 5), ("Ostapov", 5)]
     
def group2 : student_grades := 
    [("Alexeeva", 3), ("Bogdanov", 4), ("Vladimirov", 5), ("Grigorieva", 2),
     ("Davydova", 3), ("Evstahiev", 2), ("Ilina", 5), ("Klimova", 4),
     ("Lavrentiev", 5), ("Mikhailova", 3)]

def average_grade (grp : student_grades) : ℚ :=
    (grp.map (λ x => x.snd)).sum / grp.length

def updated_group (grp : List (String × Nat)) (student : String × Nat) : List (String × Nat) :=
    grp.erase student

def transfer_student (g1 g2 : student_grades) (student : String × Nat) : student_grades × student_grades :=
    (updated_group g1 student, student :: g2)

theorem can_increase_average (s : String × Nat) 
    (h1 : s ∈ group1)
    (h2 : s.snd = 4)
    (h3 : average_grade group1 < s.snd)
    (h4 : average_grade group2 > s.snd)
    : 
    let (new_group1, new_group2) := transfer_student group1 group2 s in 
    average_grade new_group1 > average_grade group1 ∧ 
    average_grade new_group2 > average_grade group2 :=
sorry

#eval can_increase_average ("Evstigneeva", 4)

end can_increase_average_l28_28337


namespace can_increase_averages_l28_28350

def grades_group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def grades_group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : Float :=
  (grades.map Nat.toFloat).sum / grades.length

def new_average (grades : List ℕ) (grade_to_remove_or_add : ℕ) (adding : Bool) : Float :=
  if adding
  then (grades.map Nat.toFloat).sum + grade_to_remove_or_add / (grades.length + 1)
  else (grades.map Nat.toFloat).sum - grade_to_remove_or_add / (grades.length - 1)

theorem can_increase_averages :
  ∃ grade,
    grade ∈ grades_group1 ∧
    average grades_group1 < new_average grades_group1 grade false ∧
    average grades_group2 < new_average grades_group2 grade true :=
  sorry

end can_increase_averages_l28_28350


namespace proof_sequences_of_length_21_is_114_l28_28724

   def valid_sequences (n : ℕ) : ℕ := 
   if n = 3 then 1
   else if n = 4 then 1
   else if n = 5 then 1
   else if n = 6 then 2
   else valid_sequences (n - 4) + 2 * valid_sequences (n - 5) + valid_sequences (n - 6)

   noncomputable def sequences_of_length_21_is_114 : Prop :=
   valid_sequences 21 = 114

   theorem proof_sequences_of_length_21_is_114 : sequences_of_length_21_is_114 := by
     sorry
   
end proof_sequences_of_length_21_is_114_l28_28724


namespace gain_percentage_l28_28989

theorem gain_percentage (SP Gain : ℝ) (h1 : SP = 225) (h2 : Gain = 75) :
  (Gain / (SP - Gain) * 100) = 50 :=
by
  have CP := SP - Gain
  rw [h1, h2]
  have h3 : CP = 225 - 75 := by rw [h1, h2]
  rw h3
  have h4 : CP = 150 := by norm_num
  rw h4
  have h5 : (75 / 150 * 100) = 50 := by norm_num
  exact h5

end gain_percentage_l28_28989


namespace jessica_exam_time_l28_28782

theorem jessica_exam_time (total_questions : ℕ) (answered_questions : ℕ) (used_minutes : ℕ)
    (total_time : ℕ) (remaining_time : ℕ) (rate : ℚ) :
    total_questions = 80 ∧ answered_questions = 16 ∧ used_minutes = 12 ∧ total_time = 60 ∧ rate = (answered_questions : ℚ) / used_minutes →
    remaining_time = total_time - used_minutes →
    remaining_time = 48 :=
by
  -- Proof will be filled in here
  sorry

end jessica_exam_time_l28_28782


namespace sum_of_angles_l28_28757

-- Declare variables and conditions
variables (A B E F : Type) [plane_geometry A B E F]

-- Define regular pentagon and regular triangle properties
def is_regular_pentagon (P : polygon) : Prop :=
  P.interior_angle = 108

def is_regular_triangle (T : polygon) : Prop :=
  T.interior_angle = 60

-- Define the problem statement
theorem sum_of_angles (P : polygon) (T : polygon)
  (hP : is_regular_pentagon P) (hT : is_regular_triangle T) :
  P.interior_angle + T.interior_angle = 168 :=
  by sorry

end sum_of_angles_l28_28757


namespace percentage_dried_fruit_of_combined_mix_l28_28482

theorem percentage_dried_fruit_of_combined_mix :
  ∀ (weight_sue weight_jane : ℝ),
  (weight_sue * 0.3 + weight_jane * 0.6) / (weight_sue + weight_jane) = 0.45 →
  100 * (weight_sue * 0.7) / (weight_sue + weight_jane) = 35 :=
by
  intros weight_sue weight_jane H
  sorry

end percentage_dried_fruit_of_combined_mix_l28_28482


namespace fraction_eq_zero_implies_x_eq_one_l28_28324

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end fraction_eq_zero_implies_x_eq_one_l28_28324


namespace area_of_square_with_perimeter_40_l28_28201

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l28_28201


namespace area_of_square_l28_28137

theorem area_of_square (A : ℝ) (x : ℝ)
  (h1 : x > 0)
  (h2 : 3 * x^2 = 14) :
  A = (4 * x)^2 :=
by
  have h3 : x^2 = 14 / 3 := by sorry
  have h4 : (4 * x)^2 = 16 * x^2 := by sorry
  rw [h3] at h4
  have h5 : 16 * (14 / 3) = 224 / 3 := by sorry
  rw [h5] at h4
  exact h4

end area_of_square_l28_28137


namespace minimum_xy_l28_28942

noncomputable def f (x y : ℝ) := 2 * x + y + 6

theorem minimum_xy (x y : ℝ) (h : 0 < x ∧ 0 < y) (h1 : f x y = x * y) : x * y = 18 :=
by
  sorry

end minimum_xy_l28_28942


namespace can_increase_average_l28_28336

def student_grades := List (String × Nat)

def group1 : student_grades := 
    [("Andreev", 5), ("Borisova", 3), ("Vasilieva", 5), ("Georgiev", 3),
     ("Dmitriev", 5), ("Evstigneeva", 4), ("Ignatov", 3), ("Kondratiev", 4),
     ("Leontieva", 3), ("Mironov", 4), ("Nikolaeva", 5), ("Ostapov", 5)]
     
def group2 : student_grades := 
    [("Alexeeva", 3), ("Bogdanov", 4), ("Vladimirov", 5), ("Grigorieva", 2),
     ("Davydova", 3), ("Evstahiev", 2), ("Ilina", 5), ("Klimova", 4),
     ("Lavrentiev", 5), ("Mikhailova", 3)]

def average_grade (grp : student_grades) : ℚ :=
    (grp.map (λ x => x.snd)).sum / grp.length

def updated_group (grp : List (String × Nat)) (student : String × Nat) : List (String × Nat) :=
    grp.erase student

def transfer_student (g1 g2 : student_grades) (student : String × Nat) : student_grades × student_grades :=
    (updated_group g1 student, student :: g2)

theorem can_increase_average (s : String × Nat) 
    (h1 : s ∈ group1)
    (h2 : s.snd = 4)
    (h3 : average_grade group1 < s.snd)
    (h4 : average_grade group2 > s.snd)
    : 
    let (new_group1, new_group2) := transfer_student group1 group2 s in 
    average_grade new_group1 > average_grade group1 ∧ 
    average_grade new_group2 > average_grade group2 :=
sorry

#eval can_increase_average ("Evstigneeva", 4)

end can_increase_average_l28_28336


namespace broken_seashells_count_l28_28431

def total_seashells : Nat := 6
def unbroken_seashells : Nat := 2
def broken_seashells : Nat := total_seashells - unbroken_seashells

theorem broken_seashells_count :
  broken_seashells = 4 :=
by
  -- The proof would go here, but for now, we use 'sorry' to denote it.
  sorry

end broken_seashells_count_l28_28431


namespace find_abc_l28_28851

theorem find_abc (a b c : ℝ) (h1 : a * (b + c) = 198) (h2 : b * (c + a) = 210) (h3 : c * (a + b) = 222) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a * b * c = 1069 :=
by
  sorry

end find_abc_l28_28851


namespace correct_weights_swapped_l28_28034

theorem correct_weights_swapped 
  (W X Y Z : ℝ) 
  (h1 : Z > Y) 
  (h2 : X > W) 
  (h3 : Y + Z > W + X) :
  (W, Z) = (Z, W) :=
sorry

end correct_weights_swapped_l28_28034


namespace max_intersection_points_l28_28907

theorem max_intersection_points (C L : Type) [fintype C] [fintype L] (hC : fintype.card C = 3) (hL : fintype.card L = 2) :
  (∑ c1 in C, ∑ c2 in (C \ {c1}), 2) + (∑ l in L, ∑ c in C, 2) + 1 = 19 :=
by
  sorry

end max_intersection_points_l28_28907


namespace golden_ratio_in_range_l28_28503

theorem golden_ratio_in_range :
  let phi := (Real.sqrt 5 - 1) / 2
  in 0.6 < phi ∧ phi < 0.7 :=
by
  let phi := (Real.sqrt 5 - 1) / 2
  sorry

end golden_ratio_in_range_l28_28503


namespace probability_of_hitting_target_at_least_once_l28_28584

noncomputable def prob_hit_target_once : ℚ := 2/3

noncomputable def prob_miss_target_once : ℚ := 1 - prob_hit_target_once

noncomputable def prob_miss_target_three_times : ℚ := prob_miss_target_once ^ 3

noncomputable def prob_hit_target_at_least_once : ℚ := 1 - prob_miss_target_three_times

theorem probability_of_hitting_target_at_least_once :
  prob_hit_target_at_least_once = 26 / 27 := 
sorry

end probability_of_hitting_target_at_least_once_l28_28584


namespace repeating_decimal_23_eq_23_div_99_l28_28774

theorem repeating_decimal_23_eq_23_div_99 :
  (0.\overline{23} = 23 / 99) :=
begin
  sorry
end

end repeating_decimal_23_eq_23_div_99_l28_28774


namespace A_winning_strategy_B_winning_strategy_neither_player_can_force_win_l28_28850

noncomputable def player_A_win (n_0 : ℕ) : Prop := 
  ∃ n : ℕ, n_0 <= n ∧ n <= n_0 ^ 2 ∧ n = 1990

noncomputable def player_B_win (n_0 : ℕ) : Prop := 
  ∃ n : ℕ, n_0 = n

theorem A_winning_strategy: ∀ n_0 : ℕ, n_0 ≥ 45 → player_A_win(n_0) :=
begin
  sorry
end

theorem B_winning_strategy: ∀ n_0 : ℕ, 2 ≤ n_0 ∧ n_0 ≤ 5 → player_B_win(n_0) :=
begin
  sorry
end

theorem neither_player_can_force_win: ∀ n_0 : ℕ, n_0 = 6 ∨ n_0 = 7 → ¬player_A_win(n_0) ∧ ¬player_B_win(n_0) :=
begin
  sorry
end

end A_winning_strategy_B_winning_strategy_neither_player_can_force_win_l28_28850


namespace harmonic_sum_not_int_l28_28843

theorem harmonic_sum_not_int (n : ℕ) (h : n > 1) : ¬ ∃ m : ℤ, (∑ i in Finset.range (n + 1), 1 / (i : ℚ)) = m := 
sorry

end harmonic_sum_not_int_l28_28843


namespace vector_parallel_dot_product_l28_28721

theorem vector_parallel_dot_product (m n : ℝ) (a b : ℝ × ℝ × ℝ)
  (ha : a = (1, 3 * m - 1, n - 2))
  (hb : b = (2, 3 * m + 1, 3 * n - 4))
  (h_parallel : ∃ λ : ℝ, a = (λ * (2, 3 * m + 1, 3 * n - 4)) ) :
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 18 :=
by
  sorry

end vector_parallel_dot_product_l28_28721


namespace term_containing_x10_is_5th_l28_28378

def binomial_general_term (n r : ℕ) (x : ℚ) : ℚ :=
  (n.choose r) * ((1 / real.sqrt x)^r) * ((-1)^(n - r) * x^(2*(n-r)))

theorem term_containing_x10_is_5th (x : ℚ) (hx : x > 0) :
  ∃ r, binomial_general_term 10 r x = (10.choose r) * ((1 / real.sqrt x)^r) * ((-1)^(10 - r) * x^(20 - (5 * r / 2))) ∧ 20 - (5 * r / 2) = 10 ∧ r + 1 = 5 :=
by
  sorry

end term_containing_x10_is_5th_l28_28378


namespace A_subset_B_l28_28424

def A : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 4 * k + 1 }
def B : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 }

theorem A_subset_B : A ⊆ B :=
  sorry

end A_subset_B_l28_28424


namespace represent_2015_l28_28458

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def in_interval (n : ℕ) : Prop := 400 < n ∧ n < 500

def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem represent_2015 :
  ∃ a b c : ℕ,
  a + b + c = 2015 ∧
  is_prime a ∧
  is_divisible_by_3 b ∧
  in_interval c ∧
  not_divisible_by_3 c :=
by {
  use 7,
  use 1605,
  use 403,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
}

end represent_2015_l28_28458


namespace circle_and_line_cartesian_equations_l28_28758

noncomputable def circle_in_cartesian : (ℝ × ℝ) × ℝ → (ℝ × ℝ) → Prop
| ((r, θ), R), (x, y) => (x - r * cos(θ))^2 + (y - r * sin(θ))^2 = R^2

noncomputable def line_in_cartesian : ℝ → (ℝ × ℝ) → Prop
| θ, (x, y) => y = x * tan(θ)

theorem circle_and_line_cartesian_equations :
    (circle_in_cartesian ((sqrt 2, π / 4), sqrt 3) (1, 1)) ∧
    (line_in_cartesian (π / 4) (x, y)) ↔
    ((x - 1)^2 + (y - 1)^2 = 3 ∧ y = x) :=
by
  sorry

end circle_and_line_cartesian_equations_l28_28758


namespace garrison_initial_men_l28_28958

def initial_men_in_garrison (M : ℕ) (provisions_days : ℕ) (reinforcement : ℕ) (remaining_days_after_reinforcement : ℕ) :=
  provisions_days = 54 ∧
  reinforcement = 1600 ∧
  remaining_days_after_reinforcement = 20 ∧
  M * 36 = (M + 1600) * 20

theorem garrison_initial_men : ∃ M : ℕ, initial_men_in_garrison M 54 1600 20 ∧ M = 2000 :=
by
  use 2000
  unfold initial_men_in_garrison
  simp
  sorry

end garrison_initial_men_l28_28958


namespace problem_statement_l28_28830

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 - a * b = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a - c) * (b - c) ≤ 0 :=
by sorry

end problem_statement_l28_28830


namespace can_increase_average_l28_28338

def student_grades := List (String × Nat)

def group1 : student_grades := 
    [("Andreev", 5), ("Borisova", 3), ("Vasilieva", 5), ("Georgiev", 3),
     ("Dmitriev", 5), ("Evstigneeva", 4), ("Ignatov", 3), ("Kondratiev", 4),
     ("Leontieva", 3), ("Mironov", 4), ("Nikolaeva", 5), ("Ostapov", 5)]
     
def group2 : student_grades := 
    [("Alexeeva", 3), ("Bogdanov", 4), ("Vladimirov", 5), ("Grigorieva", 2),
     ("Davydova", 3), ("Evstahiev", 2), ("Ilina", 5), ("Klimova", 4),
     ("Lavrentiev", 5), ("Mikhailova", 3)]

def average_grade (grp : student_grades) : ℚ :=
    (grp.map (λ x => x.snd)).sum / grp.length

def updated_group (grp : List (String × Nat)) (student : String × Nat) : List (String × Nat) :=
    grp.erase student

def transfer_student (g1 g2 : student_grades) (student : String × Nat) : student_grades × student_grades :=
    (updated_group g1 student, student :: g2)

theorem can_increase_average (s : String × Nat) 
    (h1 : s ∈ group1)
    (h2 : s.snd = 4)
    (h3 : average_grade group1 < s.snd)
    (h4 : average_grade group2 > s.snd)
    : 
    let (new_group1, new_group2) := transfer_student group1 group2 s in 
    average_grade new_group1 > average_grade group1 ∧ 
    average_grade new_group2 > average_grade group2 :=
sorry

#eval can_increase_average ("Evstigneeva", 4)

end can_increase_average_l28_28338


namespace find_current_speed_l28_28121

noncomputable def speed_of_current (rowing_speed := 6.5 : ℝ) 
                                   (distance := 0.09 : ℝ) 
                                   (time_seconds := 35.99712023038157 : ℝ) 
                                   (current_speed : ℝ) : Prop :=
  rowing_speed + current_speed = distance / (time_seconds / 3600)

theorem find_current_speed : ∃ c : ℝ, speed_of_current 6.5 0.09 35.99712023038157 c :=
begin
  use 2.500071993040956,
  sorry
end

end find_current_speed_l28_28121


namespace largest_mersenne_prime_lt_1000_l28_28556

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_mersenne_prime (n : ℕ) : Prop :=
  is_prime n ∧ ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_lt_1000 : ∃ (n : ℕ), is_mersenne_prime n ∧ n < 1000 ∧ ∀ (m : ℕ), is_mersenne_prime m ∧ m < 1000 → m ≤ n :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_1000_l28_28556


namespace bianca_tulips_l28_28158

theorem bianca_tulips :
  ∃ T : ℕ, (T + 49 = 88) ∧ (T = 39) :=
begin
  use 39,
  split,
  { rw add_comm,
    exact rfl },
  { exact rfl }
end

end bianca_tulips_l28_28158


namespace seedling_sales_problem_l28_28949

variable (x y z : ℕ)

theorem seedling_sales_problem
  (h1 : 3 * x + 2 * y + z = 29000)
  (h2 : x = (1 / 2) * y)
  (h3 : y = (3 / 4) * z) :
  x + y + z = 17000 :=
sorry

end seedling_sales_problem_l28_28949


namespace distance_of_point_to_line_l28_28381

noncomputable theory
open Real

def point_polar := (2: ℝ, π/3)

def line_polar (ρ θ : ℝ) := ρ * (cos θ + sqrt 3 * sin θ) = 6

def distance_from_point_to_line (point_polar : ℝ × ℝ) (line_polar : ℝ → ℝ → Prop) : ℝ :=
  let (ρ, θ) := point_polar
  let x := ρ * cos θ
  let y := ρ * sin θ
  let A := 1
  let B := sqrt 3
  let C := -6
  let num := abs (A * x + B * y + C)
  let denom := sqrt (A^2 + B^2)
  num / denom

theorem distance_of_point_to_line :
  distance_from_point_to_line point_polar line_polar = 1 :=
by
  sorry

end distance_of_point_to_line_l28_28381


namespace parallelepiped_volume_l28_28857

theorem parallelepiped_volume (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (perpendicular : ∀ (a b : ℝ), a * b = 0)
  (angle_ca : ∀ (c a : ℝ), ∃θ (hθ : θ = 60), cos θ = 1 / 2)
  (angle_cb : ∀ (c b : ℝ), ∃θ (hθ : θ = 60), cos θ = 1 / 2) :
  (volume : ℝ) = (a * b * c * Real.sqrt 3) / 2 := by
  sorry

end parallelepiped_volume_l28_28857


namespace find_n_l28_28196

def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

theorem find_n (n : ℕ) : 3 * n * factorial n + 2 * factorial n = 40320 → n = 8 :=
by
  sorry

end find_n_l28_28196


namespace complex_quadrant_proof_l28_28223

noncomputable def Z1 : ℂ := 1 + complex.I
noncomputable def Z2 : ℂ := -2 - 3 * complex.I

-- Define the subtraction of complex numbers Z1 and Z2
noncomputable def Z_diff := Z1 - Z2

-- Define the quadrant determination function
def quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "Origin or on the axes"

-- Prove the problem statement
theorem complex_quadrant_proof : quadrant Z_diff = "First quadrant" := by
  sorry

end complex_quadrant_proof_l28_28223


namespace John_remaining_money_l28_28399

theorem John_remaining_money (initial_amount : ℝ) (fraction_given : ℝ) (groceries_cost : ℝ) (gift_cost : ℝ) :
  initial_amount = 100 →
  fraction_given = 1/3 →
  groceries_cost = 40 →
  gift_cost = 15 →
  let remaining_amount := initial_amount - fraction_given * initial_amount - groceries_cost - gift_cost in
  remaining_amount = 11.67 :=
by
  intros h_init h_fraction h_groceries h_gift
  simp [h_init, h_fraction, h_groceries, h_gift]
  sorry

end John_remaining_money_l28_28399


namespace unique_sum_of_squares_l28_28450

theorem unique_sum_of_squares (p : ℕ) (k : ℕ) (x y a b : ℤ) 
  (hp : Prime p) (h1 : p = 4 * k + 1) (hx : x^2 + y^2 = p) (ha : a^2 + b^2 = p) :
  (x = a ∨ x = -a) ∧ (y = b ∨ y = -b) ∨ (x = b ∨ x = -b) ∧ (y = a ∨ y = -a) :=
sorry

end unique_sum_of_squares_l28_28450


namespace alan_spent_total_amount_l28_28595

-- Conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation 
def cost_of_eggs : ℕ := eggs_bought * price_per_egg
def cost_of_chickens : ℕ := chickens_bought * price_per_chicken
def total_cost : ℕ := cost_of_eggs + cost_of_chickens

-- Proof statement
theorem alan_spent_total_amount : total_cost = 88 :=
by
  unfold total_cost cost_of_eggs cost_of_chickens
  simp [eggs_bought, price_per_egg, chickens_bought, price_per_chicken]
  sorry

end alan_spent_total_amount_l28_28595


namespace lemons_at_opening_l28_28157

-- Definitions based on the conditions
def initial_oranges := 60
def final_oranges := 40
def final_lemons := 20
def ratio_decrease := 0.4

-- The theorem asserting our desired proof problem
theorem lemons_at_opening (L : ℕ)
  (H : L / initial_oranges.toFloat = (0.5 + (0.5 * ratio_decrease))) :
  L = 42 :=
sorry

end lemons_at_opening_l28_28157


namespace present_worth_proof_l28_28872

-- Define the conditions
def banker's_gain (BG : ℝ) : Prop := BG = 16
def true_discount (TD : ℝ) : Prop := TD = 96

-- Define the relationship from the problem
def relationship (BG TD PW : ℝ) : Prop := BG = TD - PW

-- Define the present worth of the sum
def present_worth : ℝ := 80

-- Theorem stating that the present worth of the sum is Rs. 80 given the conditions
theorem present_worth_proof (BG TD PW : ℝ)
  (hBG : banker's_gain BG)
  (hTD : true_discount TD)
  (hRelation : relationship BG TD PW) :
  PW = present_worth := by
  sorry

end present_worth_proof_l28_28872


namespace Mark_paid_total_cost_l28_28812

def length_of_deck : ℝ := 30
def width_of_deck : ℝ := 40
def cost_per_sq_ft_without_sealant : ℝ := 3
def additional_cost_per_sq_ft_sealant : ℝ := 1

def area (length width : ℝ) : ℝ := length * width
def total_cost (area cost_without_sealant cost_sealant : ℝ) : ℝ := 
  area * cost_without_sealant + area * cost_sealant

theorem Mark_paid_total_cost :
  total_cost (area length_of_deck width_of_deck) cost_per_sq_ft_without_sealant additional_cost_per_sq_ft_sealant = 4800 := 
by
  -- Placeholder for proof
  sorry

end Mark_paid_total_cost_l28_28812


namespace terminal_side_quadrant_l28_28740

theorem terminal_side_quadrant
  (k : ℤ) :
  let α := (2 * k * Real.pi / 3) + (Real.pi / 6) in
  (∃ n : ℤ, α = 2 * n * Real.pi + Real.pi / 6 ∨
             α = 2 * n * Real.pi + 5 * Real.pi / 6 ∨
             α = 2 * n * Real.pi + 3 * Real.pi / 2) :=
sorry

end terminal_side_quadrant_l28_28740


namespace part_a_part_b_l28_28497

def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem part_a : ∃ n : ℕ, is_multiple_of_9 n ∧ digit_sum n = 81 ∧ (n / 9) = 111111111 := 
sorry

theorem part_b : ∃ n1 n2 n3 n4 : ℕ,
  is_multiple_of_9 n1 ∧
  is_multiple_of_9 n2 ∧
  is_multiple_of_9 n3 ∧
  is_multiple_of_9 n4 ∧
  digit_sum n1 = 27 ∧ digit_sum n2 = 27 ∧ digit_sum n3 = 27 ∧ digit_sum n4 = 27 ∧
  (n1 / 9) + 1 = (n2 / 9) ∧ 
  (n2 / 9) + 1 = (n3 / 9) ∧ 
  (n3 / 9) + 1 = (n4 / 9) ∧ 
  (n4 / 9) < 1111 := 
sorry

end part_a_part_b_l28_28497


namespace zoe_remaining_pictures_l28_28542

-- Definitions for the problem conditions
def monday_pictures := 24
def tuesday_pictures := 37
def wednesday_pictures := 50
def thursday_pictures := 33
def friday_pictures := 44

def rate_first := 4
def rate_second := 5
def rate_third := 6
def rate_fourth := 3
def rate_fifth := 7

def days_colored (start_day : ℕ) (end_day := 6) := end_day - start_day

def remaining_pictures (total_pictures : ℕ) (rate_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pictures - (rate_per_day * days)

-- Main theorem statement
theorem zoe_remaining_pictures : 
  remaining_pictures monday_pictures rate_first (days_colored 1) +
  remaining_pictures tuesday_pictures rate_second (days_colored 2) +
  remaining_pictures wednesday_pictures rate_third (days_colored 3) +
  remaining_pictures thursday_pictures rate_fourth (days_colored 4) +
  remaining_pictures friday_pictures rate_fifth (days_colored 5) = 117 :=
  sorry

end zoe_remaining_pictures_l28_28542


namespace quadratic_inequality_solution_l28_28509

theorem quadratic_inequality_solution:
  ∀ x : ℝ, -x^2 + 3 * x - 2 ≥ 0 ↔ (1 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l28_28509


namespace arithmetic_sequence_middle_term_l28_28053

theorem arithmetic_sequence_middle_term :
  let a₁ := 3^2 in
  let a₃ := 3^3 in
  let y := (a₁ + a₃) / 2 in
  y = 18 :=
by
  let a₁ := 3^2
  let a₃ := 3^3
  let y := (a₁ + a₃) / 2
  have h₁ : a₁ = 9 := rfl
  have h₂ : a₃ = 27 := rfl
  have h₃ : y = (9 + 27) / 2 := by rw [h₁, h₂]
  have h₄ : 9 + 27 = 36 := rfl
  have h₅ : y = 36 / 2 := by rw [h₃, h₄]
  have h₆ : 36 / 2 = 18 := rfl
  show y = 18 from by
    rw [h₅, h₆]

end arithmetic_sequence_middle_term_l28_28053


namespace minimum_list_length_for_third_best_team_l28_28368

theorem minimum_list_length_for_third_best_team (n : ℕ) (h1 : n = 2^9) :
    ∃ (l : Finset ℕ), l.card = 45 ∧
                      ∀ (team : ℕ), (team ≤ 511) → (team ∈ l ∨
                      ∃ t1 t2 t3 : ℕ, t1 ∈ l ∧ t2 ∈ l ∧ t3 ∈ l ∧
                      team ≠ t1 ∧ team ≠ t2 ∧ team ≠ t3 ∧ 
                      (team beats t1 ∨ team beats t2 ∨ team beats t3))
  := sorry

end minimum_list_length_for_third_best_team_l28_28368


namespace polar_coords_and_max_distance_l28_28766

/-
  The parametric forms of the curves C1 and C2, and the transformation applied to the coordinates.
-/
def C1_parametric (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, Real.sqrt 3 * Real.sin α)
def C2_parametric (α : ℝ) : ℝ × ℝ := (1/2 + Real.cos α, Real.sin α)

/-
  The polar equation of curve C2 derived from its parametric form.
-/
def C2_polar_equation (ρ θ : ℝ) : Prop := ρ^2 - ρ * Real.cos θ - 3/4 = 0

/-
  The polar equation of line l.
-/
def line_polar_equation (ρ θ : ℝ) : Prop := 4 * ρ * Real.sin (θ + Real.pi / 3) + Real.sqrt 3 = 0

/-
  The polar coordinates of the intersection points between the line and curve.
-/
def intersection_points : List (ℝ × ℝ) :=
  [ (1/2, Real.pi), (Real.sqrt 3 / 2, 3 * Real.pi / 2) ]

/-
  The point on curve C1.
-/
def point_on_C1 (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, Real.sqrt 3 * Real.sin α)

/-
  The maximum distance from a point on curve C1 to the line l.
-/
def max_distance_from_C1_to_l : ℝ := (2 * Real.sqrt 15 + 3 * Real.sqrt 3) / 4

/-
  Statement to be proved using Lean 4.
-/
theorem polar_coords_and_max_distance :
  (∀ ρ θ, C2_polar_equation ρ θ) ∧
  (∀ ρ θ, line_polar_equation ρ θ) ∧
  (∀ (p ∈ intersection_points), 
    ∃ ρ θ, (ρ, θ) = p) ∧
  (∀ α, 
    let p := point_on_C1 α in 
    ∃ d, d = max_distance_from_C1_to_l)
:=
  sorry

end polar_coords_and_max_distance_l28_28766


namespace problem_I_problem_II_l28_28674

noncomputable def f1 (x a : ℝ) := x^2 + a * x
noncomputable def g1 (x b : ℝ) := x + b
noncomputable def h1 (x : ℝ) := 2 * x^2 + 3 * x + 1

theorem problem_I (a b : ℝ) (hb : b ∈ set.Icc (1 / 2) 1) :
  (∃ m n : ℝ, h1 x = (m * f1 x a + n * g1 x b)) →
  a + 2 * b ∈ set.Icc (3 / 2) 3 :=
by sorry

noncomputable def f2 (x : ℝ) := Real.log (4^(x : ℝ) + 1) / Real.log 4
noncomputable def g2 (x : ℝ) := x - 1

theorem problem_II : 
  (∃ (n : ℝ), ∀ (x : ℝ), h2 x = f2 x + n * (g2 x)) ∧ 
  Function.Even h2 ∧ 
  ∀ x, h2 x ≥ 1 →
  h2 x = f2 x - (1 / 2) * g2 x :=
by sorry

end problem_I_problem_II_l28_28674


namespace dominos_white_square_equality_l28_28563

theorem dominos_white_square_equality {chessboard : fin 8 × fin 8 → Prop} (dominoes : set (fin 8 × fin 8) -> Prop) :
  (∀ (x y : fin 8 × fin 8), dominoes {x, y} → adj x y) →
  (∀ (x : fin 8 × fin 8), ∃ (y : fin 8 × fin 8), dominoes {x, y}) →
  (card (dominoes)) = 32 →
  (∃ h_dominoes, ∀ dom : set (fin 8 × fin 8), dominoes dom →
    ((∃ a b : fin 8 × fin 8, a.2 = b.2 + 1 ∧ ′white_square_left′ a b) ↔ (∃ c d : fin 8 × fin 8, c.2 = d.2 + 1 ∧ ′white_square_right′ c d))) :=
sorry

end dominos_white_square_equality_l28_28563


namespace winning_candidate_votes_l28_28521

theorem winning_candidate_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 336): 0.62 * V = 868 :=
by
  sorry

end winning_candidate_votes_l28_28521


namespace train_passes_jogger_in_39_seconds_l28_28116

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_head_start : ℝ := 270
noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45

noncomputable def to_meters_per_second (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

noncomputable def jogger_speed_mps : ℝ :=
  to_meters_per_second jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  to_meters_per_second train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance : ℝ :=
  jogger_head_start + train_length

noncomputable def time_to_pass_jogger : ℝ :=
  total_distance / relative_speed_mps

theorem train_passes_jogger_in_39_seconds :
  time_to_pass_jogger = 39 := by
  sorry

end train_passes_jogger_in_39_seconds_l28_28116


namespace product_of_two_numbers_l28_28033

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : x - y = 16) : 
  x * y = 836 := 
by
  sorry

end product_of_two_numbers_l28_28033


namespace radius_of_sphere_phi_pi_over_3_l28_28648

noncomputable def radius_of_circle (ρ θ φ : ℝ) : ℝ :=
  abs (ρ * sin φ)

theorem radius_of_sphere_phi_pi_over_3 {θ : ℝ} :
  radius_of_circle 3 θ (π / 3) = 3 * sqrt 3 / 2 :=
by
  sorry

end radius_of_sphere_phi_pi_over_3_l28_28648


namespace calculate_shot_cost_l28_28622

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l28_28622


namespace inequality_for_positive_real_numbers_l28_28832

theorem inequality_for_positive_real_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := 
by 
  sorry

end inequality_for_positive_real_numbers_l28_28832


namespace bisectors_angles_condition_l28_28383

noncomputable def angle_bisectors_property
  (A B C D : Type)
  [Tetrahedron A B C D]
  [HasAngleBisectors A B C D D] : Prop :=
∃ (a b c : Angle), 
  angle_bisector A D B = a ∧ 
  angle_bisector B D C = b ∧ 
  angle_bisector C D A = c ∧ 
  ((acute a ∧ acute b ∧ acute c) ∨ 
   (obtuse a ∧ obtuse b ∧ obtuse c) ∨ 
   (right_angle a ∧ right_angle b ∧ right_angle c))

theorem bisectors_angles_condition
  (A B C D : Type)
  [Tetrahedron A B C D]
  [HasAngleBisectors A B C D D] :
  angle_bisectors_property A B C D :=
sorry

end bisectors_angles_condition_l28_28383


namespace Sophie_donuts_l28_28472

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end Sophie_donuts_l28_28472


namespace hives_needed_for_candles_l28_28394

theorem hives_needed_for_candles (h : (3 : ℕ) * c = 12) : (96 : ℕ) / c = 24 :=
by
  sorry

end hives_needed_for_candles_l28_28394


namespace range_of_m_l28_28687

theorem range_of_m (C : set (ℝ × ℝ)) (m : ℝ) 
  (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  C = {p | (p.1 - 3) ^ 2 + (p.2 - 4) ^ 2 = 1} ->
  A = (-m, 0) -> 
  B = (m, 0) -> 
  m > 0 -> 
  P ∈ C -> 
  (P.1 + m, P.2) = (P.1 - m, P.2) <-> 0 -> 
  4 <= m ∧ m <= 6 :=
by
  sorry

end range_of_m_l28_28687


namespace solve_laundry_problem_l28_28607

def laundry_problem : Prop :=
  let total_weight := 20
  let clothes_weight := 5
  let detergent_per_scoop := 0.02
  let initial_detergent := 2 * detergent_per_scoop
  let optimal_ratio := 0.004
  let additional_detergent := 0.02
  let additional_water := 14.94
  let total_detergent := initial_detergent + additional_detergent
  let final_amount := clothes_weight + initial_detergent + additional_detergent + additional_water
  final_amount = total_weight ∧ total_detergent / (total_weight - clothes_weight) = optimal_ratio

theorem solve_laundry_problem : laundry_problem :=
by 
  -- the proof would go here
  sorry

end solve_laundry_problem_l28_28607


namespace crank_slider_equations_of_motion_l28_28165

noncomputable def equations_of_motion
  (omega : ℝ) (OA AB AL : ℝ) (t : ℝ) :
  ℝ × ℝ × (ℝ × ℝ) :=
let x_L := OA * cos(omega * t) - AL * sin(omega * t),
    y_L := OA * sin(omega * t) + AL * cos(omega * t),
    v_Lx := -omega * (OA * sin(omega * t) + AL * cos(omega * t)),
    v_Ly := omega * (OA * cos(omega * t) - AL * sin(omega * t))
in (x_L, y_L, (v_Lx, v_Ly))

theorem crank_slider_equations_of_motion :
  let omega := 10,
      OA := 90,
      AB := 90,
      AL := 30 in 
  ∀ t : ℝ,
    equations_of_motion omega OA AB AL t =
    (90 * cos(10 * t) - 30 * sin(10 * t), 
     90 * sin(10 * t) + 30 * cos(10 * t), 
     (-10 * (90 * sin(10 * t) + 30 * cos(10 * t)),
      10 * (90 * cos(10 * t) - 30 * sin(10 * t))) :=
by
  intros,
  unfold equations_of_motion,
  sorry

end crank_slider_equations_of_motion_l28_28165


namespace smallest_integer_divisibility_l28_28452

noncomputable def u : ℝ := 3 + real.sqrt 5
noncomputable def v : ℝ := 3 - real.sqrt 5
def T (n : ℕ) : ℝ := u^n + v^n

theorem smallest_integer_divisibility (n : ℕ) (hn : 0 < n) : 
  ∃ (k : ℤ), k > (u^2n : ℝ).floor ∧ 2^(n+1) ∣ k :=
sorry

end smallest_integer_divisibility_l28_28452


namespace find_positive_integer_triples_l28_28199

-- Define the condition for the integer divisibility problem
def is_integer_division (t a b : ℕ) : Prop :=
  (t ^ (a + b) + 1) % (t ^ a + t ^ b + 1) = 0

-- Statement of the theorem
theorem find_positive_integer_triples :
  ∀ (t a b : ℕ), t > 0 → a > 0 → b > 0 → is_integer_division t a b → (t, a, b) = (2, 1, 1) :=
by
  intros t a b t_pos a_pos b_pos h
  sorry

end find_positive_integer_triples_l28_28199


namespace rectangle_dimensions_l28_28819

-- Define the known shapes and their dimensions
def square (s : ℝ) : ℝ := s^2
def rectangle1 : ℝ := 10 * 24
def rectangle2 (a b : ℝ) : ℝ := a * b

-- The total area must match the area of a square of side length 24 cm
def total_area (s a b : ℝ) : ℝ := (2 * square s) + rectangle1 + rectangle2 a b

-- The problem statement
theorem rectangle_dimensions
  (s a b : ℝ)
  (h0 : a ∈ [2, 19, 34, 34, 14, 14, 24])
  (h1 : b ∈ [24, 17.68, 10, 44, 24, 17, 38])
  : (total_area s a b = 24^2) :=
by
  sorry

end rectangle_dimensions_l28_28819


namespace can_transfer_increase_average_l28_28354

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l28_28354


namespace maximum_value_of_f_l28_28019

noncomputable def f (x : ℝ) : ℝ := -x + (1 / x)

theorem maximum_value_of_f :
  ∃ (x ∈ set.Icc (-2 : ℝ) (-1/9)), ∀ (y ∈ set.Icc (-2 : ℝ) (-1/9)), f y ≤ f x ∧ f x = (3 / 2) := 
by
  sorry

end maximum_value_of_f_l28_28019


namespace number_above_210_is_190_l28_28071

def triangular_number (k : ℕ) : ℕ := k * (k + 1) / 2

def start_of_row (k : ℕ) : ℕ := if k = 1 then 1 else triangular_number (k - 1) + 1

theorem number_above_210_is_190 :
  ∃ k : ℕ, k = 20 ∧ triangular_number 20 = 210 ∧ 
  (let start_20th = start_of_row 20 in 
   let start_19th = start_of_row 19 in 
   210 = start_20th + 19 ∧ 190 = start_19th + 18) := 
by
  sorry

end number_above_210_is_190_l28_28071


namespace sum_of_digits_of_N_l28_28593

-- The total number of coins
def total_coins : ℕ := 3081

-- Setting up the equation N^2 = 3081
def N : ℕ := 55 -- Since 55^2 is closest to 3081 and sqrt(3081) ≈ 55

-- Proving the sum of the digits of N is 10
theorem sum_of_digits_of_N : (5 + 5) = 10 :=
by
  sorry

end sum_of_digits_of_N_l28_28593


namespace mean_median_difference_l28_28331

/-- Definition of scores and their percentages -/
def student_scores : List ℕ := 
  List.replicate 4 60 ++ 
  List.replicate 5 75 ++ 
  List.replicate 8 85 ++ 
  List.replicate 5 90 ++ 
  List.replicate 3 100

/-- Check if the list is correctly defined -/
example : List.length student_scores = 25 := by
  simp [student_scores]

/-- Function to calculate the mean of a list of natural numbers -/
def mean (scores : List ℕ) : Float := 
  (Float.ofInt (scores.foldl (· + ·) 0)) / (Float.ofNat scores.length)

/-- Function to calculate the median of a list of natural numbers -/
def median (scores : List ℕ) : ℕ := 
  let sorted_scores := scores.qsort (· < ·)
  sorted_scores.get! (scores.length / 2)

/-- The mathematically equivalent proof problem. -/
theorem mean_median_difference :
  mean student_scores - Float.ofNat (median student_scores) = 0.8 :=
sorry

end mean_median_difference_l28_28331


namespace single_train_car_passenger_count_l28_28585

theorem single_train_car_passenger_count (P : ℕ) 
  (h1 : ∀ (plane_capacity train_capacity : ℕ), plane_capacity = 366 →
    train_capacity = 16 * P →
      (train_capacity = (2 * plane_capacity) + 228)) : 
  P = 60 :=
by
  sorry

end single_train_car_passenger_count_l28_28585


namespace annual_interest_income_l28_28590

variables (totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate : ℝ)
           (firstInterest secondInterest totalInterest : ℝ)

def investment_conditions : Prop :=
  totalInvestment = 32000 ∧
  firstRate = 0.0575 ∧
  secondRate = 0.0625 ∧
  firstBondPrincipal = 20000 ∧
  secondBondPrincipal = totalInvestment - firstBondPrincipal

def calculate_interest (principal rate : ℝ) : ℝ := principal * rate

def total_annual_interest (firstInterest secondInterest : ℝ) : ℝ :=
  firstInterest + secondInterest

theorem annual_interest_income
  (hc : investment_conditions totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate) :
  total_annual_interest (calculate_interest firstBondPrincipal firstRate)
    (calculate_interest secondBondPrincipal secondRate) = 1900 :=
by {
  sorry
}

end annual_interest_income_l28_28590


namespace angle_BKC_gt_90_l28_28762

noncomputable theory

open EuclideanGeometry

variable {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]
variables {A B C P Q H K : α}

-- Assume triangle is acute
variables 
  (h₁ : ∠A < π/2)
  (h₂ : ∠B > π/3)
  (h₃ : ∠C > π/3)
  
-- Conditions on points and orthocenter
variables 
  [circumcircle : circle ({A, P, Q, H})]
  (hP : on_side P A B)
  (hQ : on_side Q A C)

-- Midpoint condition
variable (hK : midpoint K P Q)

theorem angle_BKC_gt_90 : ∠BKC > π/2 :=
by
  sorry

end angle_BKC_gt_90_l28_28762


namespace problem1_problem2_problem3_l28_28676

-- Definitions and conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def even_digits : Finset ℕ := digits.filter (λ x => x % 2 = 0)
def odd_digits : Finset ℕ := digits.filter (λ x => x % 2 = 1)

-- Lean statements equivalent to the given problems
theorem problem1 : 
  let num_even_chosen := Nat.choose 4 3,
      num_odd_chosen := Nat.choose 5 4,
      arrangements := Nat.factorial 7 in
  num_even_chosen * num_odd_chosen * arrangements = 100800 := by sorry

theorem problem2 : 
  let num_even_chosen := Nat.choose 4 3,
      num_odd_chosen := Nat.choose 5 4,
      grouped_arrangement := Nat.factorial 5,
      arrangements_of_group := Nat.factorial 3 in
  num_even_chosen * num_odd_chosen * grouped_arrangement * arrangements_of_group = 14400 := by sorry

theorem problem3 : 
  let num_odd_chosen := Nat.choose 5 4,
      positions_choosen := Nat.choose 5 3,
      odd_arrangements := Nat.factorial 4,
      even_arrangements := Nat.factorial 3 in
  num_odd_chosen * positions_choosen * odd_arrangements * even_arrangements = 28800 := by sorry

end problem1_problem2_problem3_l28_28676


namespace child_l28_28545

/-
This theorem states that given the arrangement of four cubes displaying certain letters, 
we can determine the child's name.
-/
noncomputable def child's_name (cubes : List (List Char)) : String := 
sorry

/-
Conditions:
1. Each cube displays one of the letters visible to the child.
2. The letters visible spell out a name.
-/
axiom cube1 : List Char := ['Н', 'И']
axiom cube2 : List Char := ['К', 'Т']
axiom cube3 : List Char := ['Н']
axiom cube4 : List Char := ['А']

/-
Theorem: Given the conditions specified, the child's name is Ника.
-/
theorem child's_name_is_Nika (h : [cube1.head!, cube3.head!, cube2.head!, cube4.head!] = ['Н', 'И', 'К', 'А']) : 
  child's_name [cube1, cube2, cube3, cube4] = "Ника" :=
sorry

end child_l28_28545


namespace max_value_ab_bc_cd_de_ea_l28_28496

-- Define the integers set
def S := {1, 2, 3, 4, 5}

-- Define the function to compute the value
def f (a b c d e : ℕ) : ℕ := a * b + b * c + c * d + d * e + e * a

-- Statement of the theorem
theorem max_value_ab_bc_cd_de_ea : ∃ (a b c d e : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
  f a b c d e = 42 :=
sorry

end max_value_ab_bc_cd_de_ea_l28_28496


namespace XY_passes_through_H_l28_28447

open EuclideanGeometry

variables {A B C M H Q X Y : Point} -- assuming appropriate definitions of Point
variables [Triangle ABC] 

theorem XY_passes_through_H (M_midpoint : M = midpoint B C)
  (H_orthocenter : H = orthocenter A B C)
  (MH_Aangle_bisector_intersection : ∃ Q, line_through M H ∩ Aangle_bisector A B C = {Q})
  (X_projection : ∃ X, is_projection Q A B X)
  (Y_projection : ∃ Y, is_projection Q A C Y) :
  collinear X Y H :=
sorry

end XY_passes_through_H_l28_28447


namespace price_reduction_l28_28101

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l28_28101


namespace sum_of_polynomials_l28_28419

-- Define the given polynomials f, g, and h
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- Prove that the sum of f(x), g(x), and h(x) is a specific polynomial
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := 
by {
  -- Proof is omitted
  sorry
}

end sum_of_polynomials_l28_28419


namespace infinite_sum_equiv_l28_28162

theorem infinite_sum_equiv 
  (h1 : ∀ n ≥ 3, (5 * n^3 - 2 * n^2 - n + 3) / (n^6 - 2 * n^5 + 2 * n^4 - 2 * n^3 + 2 * n^2 - 2 * n + 1) = 
                  (3 / (n-1)^2) + ((2 * n - 1) / (n^2 + 1)) + ((n - 2) / (n^2 + 1)^2)) : 
  ∑' n : ℕ, if h : n ≥ 3 then (5 * n^3 - 2 * n^2 - n + 3) / (n^6 - 2 * n^5 + 2 * n^4 - 2 * n^3 + 2 * n^2 - 2 * n + 1) else 0 = 1 :=
by
  sorry

end infinite_sum_equiv_l28_28162


namespace correct_option_C_l28_28979

def geometric_body : Type := 
| TriangularPrism 
| QuadrangularPyramid 
| Sphere 
| Cone 
| Cube 
| Frustum 
| HexagonalPyramid
| Hemisphere

def is_polyhedron (g : geometric_body) : Prop :=
match g with
| geometric_body.TriangularPrism => true
| geometric_body.QuadrangularPyramid => true
| geometric_body.Cube => true
| geometric_body.HexagonalPyramid => true
| _ => false
end

def is_solids_of_revolution (g : geometric_body) : Prop :=
match g with
| geometric_body.Sphere => true
| geometric_body.Cone => true
| geometric_body.Frustum => true
| geometric_body.Hemisphere => true
| _ => false
end

theorem correct_option_C :
  ∀ (A B C D : List geometric_body),
  A = [geometric_body.TriangularPrism, geometric_body.QuadrangularPyramid, geometric_body.Sphere, geometric_body.Cone] →
  B = [geometric_body.TriangularPrism, geometric_body.QuadrangularPyramid, geometric_body.Cube, geometric_body.Frustum] →
  C = [geometric_body.TriangularPrism, geometric_body.QuadrangularPyramid, geometric_body.Cube, geometric_body.HexagonalPyramid] →
  D = [geometric_body.Cone, geometric_body.Frustum, geometric_body.Sphere, geometric_body.Hemisphere] →
  (∀ g ∈ A, is_polyhedron g) = false ∧ 
  (∀ g ∈ B, is_polyhedron g) = false ∧ 
  (∀ g ∈ D, is_polyhedron g) = false ∧ 
  (∀ g ∈ C, is_polyhedron g) = true :=
by
  intros A B C D hA hB hC hD
  sorry

end correct_option_C_l28_28979


namespace percentage_second_question_correct_l28_28307

-- Define the given percentages
def P_A : ℝ := 75 / 100
def P_A_and_B : ℝ := 65 / 100
def P_neither : ℝ := 20 / 100
def P_union_A_B : ℝ := 1 - P_neither

-- Define the percentage we need to prove
def x : ℝ := 70 / 100

theorem percentage_second_question_correct :
  P_union_A_B = P_A + x - P_A_and_B → x = (5 : ℝ) / 100 + (65 : ℝ) / 100 := by
  -- This would be the proof of the statement
  sorry

end percentage_second_question_correct_l28_28307


namespace lateral_surface_area_of_cylinder_l28_28954

-- Define the conditions
def length : ℝ := 4
def width : ℝ := 2

-- Define the lateral surface area calculation
def lateral_surface_area_cylinder (radius height : ℝ) : ℝ := 2 * Real.pi * radius * height

-- Prove the lateral surface area given the conditions
theorem lateral_surface_area_of_cylinder : 
  (lateral_surface_area_cylinder width length = 16 * Real.pi) ∧ 
  (lateral_surface_area_cylinder length width = 16 * Real.pi) :=
by
  sorry

end lateral_surface_area_of_cylinder_l28_28954


namespace find_a_value_l28_28253

theorem find_a_value (a : ℝ) :
  (∀ x ∈ set.Icc (2 : ℝ) 3, f(x) = x^2 - 2*x + a) ∧ (let min_val := f 2, max_val := f 3 in min_val + max_val = 5) → a = 1 :=
by
  sorry

end find_a_value_l28_28253


namespace solve_for_y_l28_28006

theorem solve_for_y : ∃ y : ℤ, (1/8)^(3*y + 12) = 64^(y + 4) ∧ y = -4 :=
by
  sorry

end solve_for_y_l28_28006


namespace line_slope_intercept_l28_28507

theorem line_slope_intercept (x y : ℝ) (k b : ℝ) (h : 3 * x + 4 * y + 5 = 0) :
  k = -3 / 4 ∧ b = -5 / 4 :=
by sorry

end line_slope_intercept_l28_28507


namespace second_month_sales_l28_28568

def sales_first_month : ℝ := 7435
def sales_third_month : ℝ := 7855
def sales_fourth_month : ℝ := 8230
def sales_fifth_month : ℝ := 7562
def sales_sixth_month : ℝ := 5991
def average_sales : ℝ := 7500

theorem second_month_sales : 
  ∃ (second_month_sale : ℝ), 
    (sales_first_month + second_month_sale + sales_third_month + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = average_sales ∧
    second_month_sale = 7927 := by
  sorry

end second_month_sales_l28_28568


namespace arithmetic_sequence_20th_term_l28_28008

theorem arithmetic_sequence_20th_term :
  let a1 := 8
  let d := -3
  let a_n (n : ℕ) := a1 + (n - 1) * d
  a_n 20 = -49 :=
by
  let a1 := 8
  let d := -3
  have formula_20th_term : a1 + (20 - 1) * d = -49 := sorry
  exact formula_20th_term

end arithmetic_sequence_20th_term_l28_28008


namespace sophie_donuts_left_l28_28476

theorem sophie_donuts_left :
  ∀ (boxes_initial : ℕ) (donuts_per_box : ℕ) (boxes_given_away : ℕ) (dozen : ℕ),
  boxes_initial = 4 →
  donuts_per_box = 12 →
  boxes_given_away = 1 →
  dozen = 12 →
  (boxes_initial - boxes_given_away) * donuts_per_box - (dozen / 2) = 30 :=
by 
  intros boxes_initial donuts_per_box boxes_given_away dozen 
  assume h1 h2 h3 h4
  sorry

end sophie_donuts_left_l28_28476


namespace expression_equals_answer_l28_28164

noncomputable def verify_expression : ℚ :=
  15 * (1 / 17) * 34 - (1 / 2)

theorem expression_equals_answer :
  verify_expression = 59 / 2 :=
by
  sorry

end expression_equals_answer_l28_28164


namespace minimum_value_frac_l28_28227

theorem minimum_value_frac (x y z : ℝ) (h : 2 * x * y + y * z > 0) : 
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 :=
sorry

end minimum_value_frac_l28_28227


namespace area_of_triangle_BXC_l28_28042

-- Define a trapezoid ABCD with given conditions
structure Trapezoid :=
  (A B C D X : Type)
  (AB CD : ℝ)
  (area_ABCD : ℝ)
  (intersect_at_X : Prop)

theorem area_of_triangle_BXC (t : Trapezoid) (h1 : t.AB = 24) (h2 : t.CD = 40)
  (h3 : t.area_ABCD = 480) (h4 : t.intersect_at_X) : 
  ∃ (area_BXC : ℝ), area_BXC = 120 :=
by {
  -- skip the proof here by using sorry
  sorry
}

end area_of_triangle_BXC_l28_28042


namespace hulk_first_jump_more_than_500_l28_28009

def hulk_jumping_threshold : Prop :=
  ∃ n : ℕ, (3^n > 500) ∧ (∀ m < n, 3^m ≤ 500)

theorem hulk_first_jump_more_than_500 : ∃ n : ℕ, n = 6 ∧ hulk_jumping_threshold :=
  sorry

end hulk_first_jump_more_than_500_l28_28009


namespace perpendicular_lines_m_value_l28_28318

theorem perpendicular_lines_m_value
  (l1 : ∀ (x y : ℝ), x - 2 * y + 1 = 0)
  (l2 : ∀ (x y : ℝ), m * x + y - 3 = 0)
  (perpendicular : ∀ (m : ℝ) (l1_slope l2_slope : ℝ), l1_slope * l2_slope = -1) : 
  m = 2 :=
by
  sorry

end perpendicular_lines_m_value_l28_28318


namespace vincent_spent_total_cost_l28_28893

-- Definitions
def books_animal := 10
def books_outer_space := 1
def books_trains := 3
def cost_per_book := 16

-- Question
theorem vincent_spent_total_cost : 
  let total_books := books_animal + books_outer_space + books_trains in
  let total_cost := total_books * cost_per_book in
  total_cost = 224 := 
by sorry

end vincent_spent_total_cost_l28_28893


namespace composite_product_division_l28_28190

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end composite_product_division_l28_28190


namespace find_larger_number_l28_28856

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1365) (h2 : y = 4 * x + 15) : y = 1815 :=
sorry

end find_larger_number_l28_28856


namespace part_a_part_b_l28_28402

-- Define the geometric setup with points on a semicircle and a tangent circle
variables {A B C D E S T : Point} -- Points on the plane
variables {h : Semicircle A B} -- AB is the diameter of the semicircle h
variables (k : Circle) -- Circle k touches the semicircle h and sides AB, and CD

-- Given conditions
axiom h_distinct_points    : A ≠ B ∧ B ≠ C ∧ C ≠ A
axiom C_on_h               : OnSemicircle C h
axiom D_foot_perpendicular : FootPerpendicular C A B D
axiom k_touches_h          : Touches k h T
axiom k_touches_AB         : Touches k (Line A B) E
axiom k_touches_CD         : Touches k (Line C D) S

-- Prove part (a): points A, S, T are collinear
theorem part_a : Collinear A S T :=
by 
  sorry

-- Prove part (b): AC = AE
theorem part_b : Distance A C = Distance A E :=
by 
  sorry

end part_a_part_b_l28_28402


namespace squares_not_all_congruent_l28_28539

theorem squares_not_all_congruent :
  (∀ (s : Square), isEquiangular s) ∧
  (∀ (s : Square), isRectangle s) ∧
  (∀ (s : Square), isRegularPolygon s) ∧
  ¬(∀ (s t : Square), congruent s t) ∧
  (∀ (s t : Square), similar s t) :=
by
  /* a comprehensive definition of Square and some accompanying definitions will be needed here. */
  -- requires formal definitions of the types and predicates used for conditions.
  sorry

end squares_not_all_congruent_l28_28539


namespace max_area_rectangle_l28_28133

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l28_28133


namespace playground_perimeter_km_l28_28038

def playground_length : ℕ := 360
def playground_width : ℕ := 480

def perimeter_in_meters (length width : ℕ) : ℕ := 2 * (length + width)

def perimeter_in_kilometers (perimeter_m : ℕ) : ℕ := perimeter_m / 1000

theorem playground_perimeter_km :
  perimeter_in_kilometers (perimeter_in_meters playground_length playground_width) = 168 :=
by
  sorry

end playground_perimeter_km_l28_28038


namespace total_price_l28_28541

theorem total_price (r w : ℕ) (hr : r = 4275) (hw : w = r - 1490) : r + w = 7060 :=
by
  sorry

end total_price_l28_28541


namespace final_graph_vertices_degree_one_l28_28753

-- Defining the initial conditions and notation for the problem
variables (V : Type*) [Fintype V]
variables (G : SimpleGraph V) (n : ℕ) [DecidableRel G.adj] [Fintype G.vertex]
variables (c : Cycle G) (k : ℕ)

-- Original graph G has 2002 vertices and is 2-vertex-connected
axiom init_graph_2002_vertices : Fintype.card V = 2002
axiom init_graph_2_vertex_connected : ∀ v, Connected (G - (G.neighbor_set v).toFinset)

-- Transformation operation: removing a cycle and adding a new vertex
axiom transformation_operation : Π (G : SimpleGraph V) (c : Cycle G), 
  ∃ (G' : SimpleGraph (V ⊕ Unit)), ∀ v ∈ c.support, G.degree v > 1 → 
  (G'.degree (Sum.inl v) = G.degree v - (c.support.card - 1) ∧ G'.degree (Sum.inr ()) = c.support.card)

-- Theorem statement: final graph contains at least 2002 vertices of degree 1
theorem final_graph_vertices_degree_one : 
  (∃ (G' : SimpleGraph (V ⊕ Unit)), ∀ v ∈ V, G.degree v > 1 → G'.degree (Sum.inl v) = 1) → 
  Fintype.card {v : (V ⊕ Unit) // (G.degree (Sum.inl v) = 1)} ≥ 2002 :=
sorry

end final_graph_vertices_degree_one_l28_28753


namespace problem_statement_l28_28681

theorem problem_statement (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^3 + (1 / (y + 2016)) = y^3 + (1 / (z + 2016))) 
  (h5 : y^3 + (1 / (z + 2016)) = z^3 + (1 / (x + 2016))) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end problem_statement_l28_28681


namespace fibonacci_property_l28_28404

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_property (a b : ℝ) :
  (∀ n : ℕ, ∃ k : ℕ, a * (fibonacci n) + b * (fibonacci (n + 1)) = fibonacci k) ↔ 
  ((a, b) = (0, 1) ∨ (a, b) = (1, 0) ∨ 
   (∃ k : ℕ, a = fibonacci k ∧ b = fibonacci (k + 1))) :=
by
  sorry

end fibonacci_property_l28_28404


namespace no_real_pairs_ab_arithmetic_progression_l28_28178

theorem no_real_pairs_ab_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (a = (15 + b) / 2) ∧ (a + a * b = 2 * b) :=
by {
  intro hab,
  rcases hab with ⟨a, b, ha1, ha2⟩,
  have h1 : a = (15 + b) / 2 := ha1,
  have h2 : (a + a * b = 2 * b),
  {
    rw ha1,
    rw ha2,
  },
  let f : ℝ → ℝ := λ b, b^2 + 9 * b + 30,
  have h_quad_eq : ∀ b, f b = 0 → false,
  {
    intro b,
    rw f,
    rw ha1,
    rw ha2,
    have discriminant : (9^2 - 4 * 1 * 30 < 0), by linarith,
    rw (ring_eq_of_se_zero_iff discriminant),
    have h : b^2 + 9b + 30 = 0 := sorry,
    sorry,
  },
  exact h_quad_eq (15, b, ha1, ha2),
}

end no_real_pairs_ab_arithmetic_progression_l28_28178


namespace percentage_not_silver_of_new_shipment_l28_28948

theorem percentage_not_silver_of_new_shipment 
  (original_cars : ℤ)
  (percent_silver_original : ℝ)
  (new_shipment_cars : ℤ)
  (percent_silver_total : ℝ) :
  original_cars = 40 →
  percent_silver_original = 0.2 →
  new_shipment_cars = 80 →
  percent_silver_total = 0.3 →
  let total_cars := original_cars + new_shipment_cars in
  let silver_original := percent_silver_original * original_cars in
  let silver_total := percent_silver_total * total_cars in
  let silver_new_shipment := silver_total - silver_original in
  let percent_silver_new_shipment := (silver_new_shipment / new_shipment_cars) * 100 in
  let percent_not_silver_new_shipment := 100 - percent_silver_new_shipment in
  percent_not_silver_new_shipment = 65 :=
sorry

end percentage_not_silver_of_new_shipment_l28_28948


namespace area_of_square_with_perimeter_40_l28_28204

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l28_28204


namespace midpoint_square_sum_l28_28407

theorem midpoint_square_sum (x y : ℝ) :
  (4, 1) = ((2 + x) / 2, (6 + y) / 2) → x^2 + y^2 = 52 :=
by
  sorry

end midpoint_square_sum_l28_28407


namespace solve_for_x_l28_28002

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l28_28002


namespace path_count_outside_boundary_l28_28095

def valid_paths_steps {x y : ℤ} (steps : ℕ) :=
  (steps = 20) ∧ 
  (x, y) = (-6, -6) ∧ 
  (x, y) = (6, 6) ∧ 
  ∀ i, i ∈ [0 .. steps] →
    let (xi, yi) := (x i, y i) in 
      -6 ≤ xi ∧ xi ≤ 6 ∧ 
      -6 ≤ yi ∧ yi ≤ 6 ∧ 
      ((xi + 1, yi) = (x (i+1), y (i+1)) ∨ 
       (xi, yi + 1) = (x (i+1), y (i+1))) ∧ 
      ((xi ≥ -3 ∧ xi ≤ 3 ∧ yi ≥ -3 ∧ yi ≤ 3) → (xi = -3 ∨ xi = 3 ∨ yi = -3 ∨ yi = 3))

theorem path_count_outside_boundary : 
  ∃ paths, 
    valid_paths_steps paths ∧ 
    (number_of_paths = 2882) :=
sorry

end path_count_outside_boundary_l28_28095


namespace jason_quarters_l28_28396

def quarters_original := 49
def quarters_added := 25
def quarters_total := 74

theorem jason_quarters : quarters_original + quarters_added = quarters_total :=
by
  sorry

end jason_quarters_l28_28396


namespace find_number_l28_28091

theorem find_number (x : ℝ) (h : 0.5 * x = 0.25 * x + 2) : x = 8 :=
by
  sorry

end find_number_l28_28091


namespace algebra_expression_eq_l28_28737

theorem algebra_expression_eq (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 := by
  sorry

end algebra_expression_eq_l28_28737


namespace can_increase_averages_by_transfer_l28_28341

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ := grades.sum / grades.length

theorem can_increase_averages_by_transfer :
    ∃ (student : ℝ) (new_group1_grades new_group2_grades : List ℝ),
      student ∈ group1_grades ∧
      new_group1_grades = (group1_grades.erase student) ∧
      new_group2_grades = student :: group2_grades ∧
      average new_group1_grades > average group1_grades ∧ 
      average new_group2_grades > average group2_grades :=
by
  sorry

end can_increase_averages_by_transfer_l28_28341


namespace compute_difference_l28_28802

noncomputable def f (n : ℝ) : ℝ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem compute_difference (r : ℝ) : f r - f (r - 1) = r * (r + 1) * (r + 2) := by
  sorry

end compute_difference_l28_28802


namespace min_value_a2_b2_l28_28218

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end min_value_a2_b2_l28_28218


namespace unique_polynomial_solution_l28_28647

theorem unique_polynomial_solution :
  ∃ P : ℝ[X], P 2017 = 2016 ∧ (∀ x : ℝ, (P.eval x + 1)^2 = P.eval (x^2 + 1)) ∧ (∀ Q : ℝ[X], (Q 2017 = 2016 ∧ (∀ x : ℝ, (Q.eval x + 1)^2 = Q.eval (x^2 + 1))) → Q = P) :=
by
  let P := (X - 1 : ℝ[X])
  use P
  sorry

end unique_polynomial_solution_l28_28647


namespace income_in_scientific_notation_l28_28974

theorem income_in_scientific_notation :
  10870 = 1.087 * 10^4 := 
sorry

end income_in_scientific_notation_l28_28974


namespace answer_choices_for_quiz_l28_28969

theorem answer_choices_for_quiz (n : ℕ) : 
  (∃ (n : ℕ), n^2 = 16 ∧ 6 * n^2 = 96) := 
begin
  use 4,
  split,
  { 
    norm_num,
  },
  { 
    norm_num,
  },
end

end answer_choices_for_quiz_l28_28969


namespace new_base_radius_l28_28682

noncomputable def original_cone_volume : ℝ :=
  (1/3) * π * 25 * 4

noncomputable def original_cylinder_volume : ℝ :=
  π * 2^2 * 8

noncomputable def total_original_volume : ℝ :=
  original_cone_volume + original_cylinder_volume

noncomputable def new_cone_volume (r : ℝ) : ℝ :=
  (1/3) * π * r^2 * 4

noncomputable def new_cylinder_volume (r : ℝ) : ℝ :=
  π * r^2 * 8

noncomputable def total_new_volume (r : ℝ) : ℝ :=
  new_cone_volume r + new_cylinder_volume r

theorem new_base_radius :
  (total_new_volume (real.sqrt 7) = total_original_volume) :=
by
  sorry

end new_base_radius_l28_28682


namespace matrix_scalar_multiplication_l28_28163

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 2]]
def C : Matrix (Fin 2) (Fin 2) ℤ := 2 • (A ⬝ B)
def D : Matrix (Fin 2) (Fin 2) ℤ := ![![34, -14], ![32, -32]]

theorem matrix_scalar_multiplication : C = D :=
by
  sorry

end matrix_scalar_multiplication_l28_28163


namespace no_integer_n_geq_1_divides_9_l28_28154

theorem no_integer_n_geq_1_divides_9 (n : ℤ) (hn : n ≥ 1) : ¬ (9 ∣ (7^n + n^3)) :=
by sorry

end no_integer_n_geq_1_divides_9_l28_28154


namespace fly_distance_from_ceiling_l28_28047

/-- 
Assume a room where two walls and the ceiling meet at right angles at point P.
Let point P be the origin (0, 0, 0). 
Let the fly's position be (2, 7, z), where z is the distance from the ceiling.
Given the fly is 2 meters from one wall, 7 meters from the other wall, 
and 10 meters from point P, prove that the fly is at a distance sqrt(47) from the ceiling.
-/
theorem fly_distance_from_ceiling : 
  ∀ (z : ℝ), 
  (2^2 + 7^2 + z^2 = 10^2) → 
  z = Real.sqrt 47 :=
by 
  intro z h
  sorry

end fly_distance_from_ceiling_l28_28047


namespace avg_percentage_reduction_correct_option_one_more_favorable_l28_28951

noncomputable def avg_percentage_reduction (initial_price final_price : ℝ) : ℝ :=
  let x := 1 - real.sqrt (final_price / initial_price)
  x

theorem avg_percentage_reduction_correct (initial_price final_price reduction : ℝ) 
  (h1 : initial_price = 10) (h2 : final_price = 6.4) (h3 : reduction = 0.2) :
  avg_percentage_reduction initial_price final_price = reduction :=
by
  sorry

noncomputable def cost_option_one (price_per_kg : ℝ) (discount : ℝ) (quantity_kg : ℝ) : ℝ :=
  price_per_kg * (1 - discount) * quantity_kg

noncomputable def cost_option_two (price_per_kg : ℝ) (cash_discount_per_ton : ℝ) (quantity_ton : ℝ) : ℝ :=
  (price_per_kg * quantity_ton * 1000) - (cash_discount_per_ton * quantity_ton)

theorem option_one_more_favorable
  (price_per_kg : ℝ) (discount : ℝ) (cash_discount_per_ton : ℝ) (quantity_kg : ℝ) (quantity_ton : ℝ) 
  (h1 : price_per_kg = 6.4) (h2 : discount = 0.2) (h3 : cash_discount_per_ton = 1000) (h4 : quantity_kg = 2000) (h5 : quantity_ton = 2) :
  cost_option_one price_per_kg discount quantity_kg < cost_option_two price_per_kg cash_discount_per_ton quantity_ton :=
by
  sorry

end avg_percentage_reduction_correct_option_one_more_favorable_l28_28951


namespace sum_of_solutions_sum_of_solutions_is_minus_twelve_l28_28663

theorem sum_of_solutions (x : ℝ) (hx : (x + 6) ^ 2 = 49) : x = -1 ∨ x = -13 := sorry

theorem sum_of_solutions_is_minus_twelve
  (S : set ℝ) (hS : ∀ x, x ∈ S ↔ (x + 6) ^ 2 = 49) : ∑ x in S, x = -12 := sorry 

end sum_of_solutions_sum_of_solutions_is_minus_twelve_l28_28663


namespace find_N_l28_28612

theorem find_N : ∃ N : ℕ, 36^2 * 72^2 = 12^2 * N^2 ∧ N = 216 :=
by
  sorry

end find_N_l28_28612


namespace find_remainder_l28_28823

def dividend : ℝ := 17698
def divisor : ℝ := 198.69662921348313
def quotient : ℝ := 89
def remainder : ℝ := 14

theorem find_remainder :
  dividend = (divisor * quotient) + remainder :=
by 
  -- Placeholder proof
  sorry

end find_remainder_l28_28823


namespace sqrt_approx_neg_neg_sqrt_approx_l28_28550

theorem sqrt_approx_neg (h1 : Real.sqrt 5.217 ≈ 2.284) (h2 : Real.sqrt 52.17 ≈ 7.223) : Real.sqrt 0.05217 ≈ 0.2284 :=
by
  -- Placeholder for actual proof
  sorry

theorem neg_sqrt_approx (h1 : Real.sqrt 5.217 ≈ 2.284) (h2 : Real.sqrt 52.17 ≈ 7.223) : -Real.sqrt 0.05217 ≈ -0.2284 :=
by
  have h := sqrt_approx_neg h1 h2
  -- Placeholder for actual proof
  sorry

end sqrt_approx_neg_neg_sqrt_approx_l28_28550


namespace decreasing_interval_of_log_composite_l28_28855

noncomputable def log_base_half (x : ℝ) : ℝ := real.log x / real.log (1 / 2)

theorem decreasing_interval_of_log_composite :
  (∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) →
  (∀ t : ℝ, t = 2 * x^2 - 3 * x + 4 → ∀ x ∈ [3/4, ∞), ∀ t : ℝ, log_base_half t = log_base_half(2 * x^2 - 3 * x + 4 )) →
  ∀ x ∈ [3/4, ∞), log_base_half (2 * x^2 - 3 * x + 4) = (log_base_half  .

-- The proof is omitted.
sorry

end decreasing_interval_of_log_composite_l28_28855


namespace tan_alpha_value_l28_28224

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 3 / 5) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : Real.tan α = -3 / 4 := 
sorry

end tan_alpha_value_l28_28224


namespace hyperbola_eccentricity_range_l28_28242

noncomputable def eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : Set ℝ :=
  {e | e = (Real.sqrt (a^2 + b^2)) / a}

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
    (h_acuteness : ∃ x y c : ℝ, c > 0 ∧ 
      let M := (c / 2, - (b * c) / (2 * a)) in 
      let F1 := (-c, 0) in 
      let F2 := (c, 0) in 
      let MF1 := (-3 * c / 2, b * c / (2 * a)) in  
      let MF2 := (c / 2, b * c / (2 * a)) in 
      (MF1.1 * MF2.1 + MF1.2 * MF2.2) > 0) :
  eccentricity_range a b h_a h_b = { e | e > 2 } :=
sorry

end hyperbola_eccentricity_range_l28_28242


namespace five_digit_numbers_div_by_3_count_l28_28049

theorem five_digit_numbers_div_by_3_count {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 3) :
    (∃ l : list ℕ, l.length = 5 ∧ 
    (∀ n ∈ l, n = d1 ∨ n = d2 ∨ n = d3) ∧ 
    list.count l d1 > 0 ∧ 
    list.count l d2 > 0 ∧ 
    list.count l d3 > 0 ∧ 
    (list.sum l) % 3 = 0) → 
    (list.filter (λ l, l.length = 5 ∧ 
    (∀ n ∈ l, n = d1 ∨ n = d2 ∨ n = d3) ∧ 
    list.count l d1 > 0 ∧ 
    list.count l d2 > 0 ∧ 
    list.count l d3 > 0 ∧ 
    (list.sum l) % 3 = 0) 
    (list.bind [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].erase_dup (λ x, permutations (x :: (permute_replicate 4 [d1, d2, d3]))))).length = 50 :=
by
  sorry

end five_digit_numbers_div_by_3_count_l28_28049


namespace hyperbola_focused_asymptotes_l28_28506

theorem hyperbola_focused_asymptotes (a b c : ℝ) (h : 0 < a) (h0 : 0 < b) (h1 : a = b) (h2 : c = 2) :
  (∃ a b : ℝ, ( a > 0 ∧ b > 0 ∧ (frac (x*x) (a*a) - frac (y*y) (b*b) = 1) ) ∧ (abs x = abs y) ∧ (c = sqrt 2 * a)) →
  (calc frac (x*x) (2) - frac (y*y) (2) = 1 ∧ e = sqrt 2) :=
by { sorry }

end hyperbola_focused_asymptotes_l28_28506


namespace num_valid_digits_l28_28669

def is_divisible (a b : ℕ) : Prop := b % a = 0

def valid_digit (A : ℕ) : Prop :=
  A ≤ 9 ∧
  is_divisible A 72 ∧
  (let last_two_digits := 10 * A + 2 in is_divisible 4 last_two_digits)

-- The main theorem statement we need to prove
theorem num_valid_digits : {A : ℕ | valid_digit A}.to_finset.card = 3 :=
  by sorry

end num_valid_digits_l28_28669


namespace sum_of_roots_l28_28244

theorem sum_of_roots {a b c d : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (h1 : c + d = -a) (h2 : c * d = b) (h3 : a + b = -c) (h4 : a * b = d) : 
    a + b + c + d = -2 := 
by
  sorry

end sum_of_roots_l28_28244


namespace acute_triangle_sum_ratios_l28_28144

-- Definitions of points, angles and conditions
variable {A B C A' B' C' A'' B'' C'' : Point}
variable {α β γ : ℝ} -- angles A, B, C

-- Conditions: 
-- Triangle ABC is acute.
-- B' is on the perpendicular bisector of AC such that ∠AB'C = 2∠A
-- A' and C' are defined similarly.
-- AA' and B'C' intersect at A'', BB' and C'A' intersect at B'', CC' and A'B' intersect at C''.

-- Question:
-- Prove that the sum of ratios is 4.

theorem acute_triangle_sum_ratios
  (acute_triangle_ABC : Triangle ABC)
  (on_perp_bisector_B' : PerpendicularBisector B' A C)
  (angle_condition_1 : Angle A B' C = 2 * α)
  (angle_condition_2: Angle C A' B = 2 * γ)
  (angle_condition_3: Angle B C' A = 2 * β)
  (intersect_points_A'' : Intersect AA' B'C' A'')
  (intersect_points_B'' : Intersect BB' C'A' B'')
  (intersect_points_C'' : Intersect CC' A'B' C'') :
  AA' / A''A' + BB' / B''B' + CC' / C''C' = 4 := 
sorry

end acute_triangle_sum_ratios_l28_28144


namespace function_value_range_l28_28516

noncomputable def f (x : ℝ) : ℝ := Real.log (3^x + 1) / Real.log 2

theorem function_value_range :
  ∀ x : ℝ, 0 < f x :=
begin
  sorry,
end

end function_value_range_l28_28516


namespace check_amounts_l28_28562

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem check_amounts (x y : ℕ) (hx : is_two_digit(x)) (hy : is_two_digit(y))
  (h1 : 100 * y + x - (100 * x + y) = 2556)
  (h2 : (x + y) % 11 = 0)
  (h3 : y = x + 26) : x = 9 := 
sorry

end check_amounts_l28_28562


namespace Pablo_is_70_cm_taller_than_Charlene_l28_28463

variable (Ruby Pablo Charlene Janet : ℕ)

-- Conditions
axiom h1 : Ruby + 2 = Pablo
axiom h2 : Charlene = 2 * Janet
axiom h3 : Janet = 62
axiom h4 : Ruby = 192

-- The statement to prove
theorem Pablo_is_70_cm_taller_than_Charlene : Pablo - Charlene = 70 :=
by
  -- Formalizing the proof
  sorry

end Pablo_is_70_cm_taller_than_Charlene_l28_28463


namespace dominos_balanced_l28_28564

-- Define the main theorem
theorem dominos_balanced (chessboard : Chessboard) (Hcover : covers_with_dominos chessboard 32):
  num_horizontal_dominos_with_white_left chessboard = num_horizontal_dominos_with_white_right chessboard := 
  sorry

-- Definitions
structure Chessboard :=
  (squares : array (8 × 8) bool) -- true for white, false for black

-- Predicate to check if a chessboard is covered with dominos
def covers_with_dominos (board : Chessboard) (N : ℕ) : Prop := 
  sorry -- definition of dominos covering the chessboard

-- Function to count the number of horizontal dominos with a white square on the left
def num_horizontal_dominos_with_white_left (board : Chessboard) : ℕ := 
  sorry -- count function

-- Function to count the number of horizontal dominos with a white square on the right
def num_horizontal_dominos_with_white_right (board : Chessboard) : ℕ := 
  sorry -- count function 

end dominos_balanced_l28_28564


namespace set_intersection_cardinality_l28_28936

theorem set_intersection_cardinality :
  let A := {x : ℝ | ∃ n : ℤ, x = (3 * n - 4) / (5 * n - 3)}
  let B := {x : ℝ | ∃ k : ℤ, x = (4 * k - 3) / (7 * k - 6)}
  let common_elements := {x | x ∈ A ∧ x ∈ B}
  common_elements.to_finset.card = 8 :=
by {
  let A := {x : ℝ | ∃ n : ℤ, x = (3 * n - 4) / (5 * n - 3)}
  let B := {x : ℝ | ∃ k : ℤ, x = (4 * k - 3) / (7 * k - 6)}
  let common_elements := {x | x ∈ A ∧ x ∈ B}
  show common_elements.to_finset.card = 8,
  sorry
}

end set_intersection_cardinality_l28_28936


namespace sqrt_16_eq_pm_4_l28_28510

theorem sqrt_16_eq_pm_4 (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := by
  sorry

end sqrt_16_eq_pm_4_l28_28510


namespace num_ways_paperboy_l28_28124

-- Define the recurrence relation and initial conditions
def D : ℕ → ℕ
| 1     := 1
| 2     := 1
| 3     := 1
| (n+1) := D n + D (n-1) + D (n-2)

-- The statement we want to prove
theorem num_ways_paperboy (n : ℕ) : D 12 = 355 :=
sorry

end num_ways_paperboy_l28_28124


namespace part_1_part_2_l28_28691

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∧ (x^2 - (a + 2) * x + 2 * a = 0)

-- Proposition q: x₁ and x₂ are two real roots of the equation x^2 - 2mx - 3 = 0
def proposition_q (m x₁ x₂ : ℝ) : Prop :=
  x₁ ^ 2 - 2 * m * x₁ - 3 = 0 ∧ x₂ ^ 2 - 2 * m * x₂ - 3 = 0

-- Inequality condition
def inequality_condition (a m x₁ x₂ : ℝ) : Prop :=
  a ^ 2 - 3 * a ≥ abs (x₁ - x₂)

-- Part 1: If proposition p is true, find the range of the real number a
theorem part_1 (a : ℝ) (h_p : proposition_p a) : -1 < a ∧ a < 1 :=
  sorry

-- Part 2: If exactly one of propositions p or q is true, find the range of the real number a
theorem part_2 (a m x₁ x₂ : ℝ) (h_p_or_q : (proposition_p a ∧ ¬(proposition_q m x₁ x₂)) ∨ (¬(proposition_p a) ∧ (proposition_q m x₁ x₂))) : (a < 1) ∨ (a ≥ 4) :=
  sorry

end part_1_part_2_l28_28691


namespace inscribed_circle_radius_eq_l28_28067

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  area / s

theorem inscribed_circle_radius_eq :
  radius_of_inscribed_circle 8 8 10 8 8 10 = (15 * Real.sqrt 13) / 13 :=
  sorry

end inscribed_circle_radius_eq_l28_28067


namespace Jake_weight_loss_l28_28739

variable (J S: ℕ) (x : ℕ)

theorem Jake_weight_loss:
  J = 93 -> J + S = 132 -> J - x = 2 * S -> x = 15 :=
by
  intros hJ hJS hCondition
  sorry

end Jake_weight_loss_l28_28739


namespace distance_from_point_to_line_P_l28_28382

noncomputable def distance_from_point_to_line (P : ℝ × ℝ) (l : ℝ → Prop) : ℝ :=
  let (x, y) := P in
  if l y then abs (y - 4 * real.sqrt 2) else 0

theorem distance_from_point_to_line_P (P : ℝ × ℝ) (l : ℝ → Prop) (h1 : P = (2 * real.sqrt 2, 2 * real.sqrt 2)) (h2 : ∀ y, l y ↔ y = 4 * real.sqrt 2) :
  distance_from_point_to_line P l = 2 * real.sqrt 2 :=
by
  rw [distance_from_point_to_line, h1, h2]
  simp
  sorry

end distance_from_point_to_line_P_l28_28382


namespace sqrt_log_sum_l28_28918

theorem sqrt_log_sum (h1 : ∀ x, log x 6 = (Real.log 6) / (Real.log x)) : 
  Real.sqrt (Real.log 6 / (Real.log 2) + Real.log 6 / (Real.log 3)) = Real.sqrt (Real.log 3 / (Real.log 2)) + Real.sqrt (Real.log 2 / (Real.log 3)) :=
by 
  sorry

end sqrt_log_sum_l28_28918


namespace transfer_student_increases_averages_l28_28348

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ :=
  if grades.length > 0 then (grades.sum / grades.length) else 0

def can_transfer_to_increase_averages (group1_grades group2_grades : List ℝ) : Prop :=
  ∃ x ∈ group1_grades, average (x :: group2_grades) > average group2_grades ∧
                       average (group1_grades.erase x) > average group1_grades

theorem transfer_student_increases_averages :
  can_transfer_to_increase_averages group1_grades group2_grades :=
by {
  sorry -- The proof is skipped as per instructions
}

end transfer_student_increases_averages_l28_28348


namespace sum_of_remainders_mod_15_l28_28074

theorem sum_of_remainders_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) :
  (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_mod_15_l28_28074


namespace G₁G₂_parallel_planes_l28_28234

variables (S A B C : Point) (G₁ G₂ : Point)
variables (T_SAB : Triangle S A B) (T_SAC : Triangle S A C) (T_SBC : Triangle S B C) (T_ABC : Triangle A B C)

-- Assume G₁ and G₂ are the centroids of triangles SAB and SAC, respectively.
axiom G₁_centroid_of_SAB : centroid T_SAB = G₁
axiom G₂_centroid_of_SAC : centroid T_SAC = G₂

-- Prove that line G₁G₂ is parallel to the planes of triangles SBC and ABC.
theorem G₁G₂_parallel_planes (T_SAB : Triangle S A B) (T_SAC : Triangle S A C) (T_SBC : Triangle S B C) (T_ABC : Triangle A B C)
(G₁_centroid_of_SAB : centroid T_SAB = G₁) (G₂_centroid_of_SAC : centroid T_SAC = G₂) :
parallel (line G₁ G₂) (plane T_SBC) ∧ parallel (line G₁ G₂) (plane T_ABC) :=
sorry

end G₁G₂_parallel_planes_l28_28234


namespace exists_constant_sum_inequality_l28_28835

theorem exists_constant_sum_inequality (m n : ℕ) (x : ℝ) 
  (hm : m > 1) (hn : n ≥ m) (hx : x > 1) : 
  ∃ C > 0, ∑ k in Finset.range (n + 1) \ Finset.range m, x^(1 / k : ℝ) ≤ 
    C * (m^2 * x^(1 / (m - 1) : ℝ) / Real.log x + n) :=
sorry

end exists_constant_sum_inequality_l28_28835


namespace P_eval_at_6_l28_28805

theorem P_eval_at_6 
  (a b c d e f : ℝ)
  (P : ℝ → ℝ := λ x, (2 * x ^ 4 - 26 * x ^ 3 + a * x ^ 2 + b * x + c) * 
                   (5 * x ^ 4 - 80 * x ^ 3 + d * x ^ 2 + e * x + f))
  (h_roots : {1, 2, 3, 4, 5} ⊆ { r : ℂ | P r = 0 }) :
  P 6 = 2400 :=
  sorry

end P_eval_at_6_l28_28805


namespace fraction_value_l28_28614

theorem fraction_value :
  (2015^2 : ℤ) / (2014^2 + 2016^2 - 2) = (1 : ℚ) / 2 :=
by
  sorry

end fraction_value_l28_28614


namespace geometric_sequence_find_a_n_l28_28808

variable {n m p : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom h1 : ∀ n, 2 * S (n + 1) - 3 * S n = 2 * a 1
axiom h2 : a 1 ≠ 0
axiom h3 : ∀ n, S (n + 1) = S n + a (n + 1)

-- Part (1)
theorem geometric_sequence : ∃ r, ∀ n, a (n + 1) = r * a n :=
sorry

-- Part (2)
axiom p_geq_3 : 3 ≤ p
axiom a1_pos : 0 < a 1
axiom a_p_pos : 0 < a p
axiom constraint1 : a 1 ≥ m ^ (p - 1)
axiom constraint2 : a p ≤ (m + 1) ^ (p - 1)

theorem find_a_n : ∀ n, a n = 2 ^ (p - 1) * (3 / 2) ^ (n - 1) :=
sorry

end geometric_sequence_find_a_n_l28_28808


namespace perimeter_of_rectangle_EFGH_l28_28836

noncomputable def rectangle_ellipse_problem (u v c d : ℝ) : Prop :=
  (u * v = 3000) ∧
  (3000 = c * d) ∧
  ((u + v) = 2 * c) ∧
  ((u^2 + v^2).sqrt = 2 * (c^2 - d^2).sqrt) ∧
  (d = 3000 / c) ∧
  (4 * c = 8 * (1500).sqrt)

theorem perimeter_of_rectangle_EFGH :
  ∃ (u v c d : ℝ), rectangle_ellipse_problem u v c d ∧ 2 * (u + v) = 8 * (1500).sqrt := sorry

end perimeter_of_rectangle_EFGH_l28_28836


namespace hyperbola_center_l28_28210

theorem hyperbola_center (x y : ℝ) :
  9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 900 = 0 →
  (x, y) = (3, 5) :=
sorry

end hyperbola_center_l28_28210


namespace square_area_l28_28205

theorem square_area (perimeter : ℝ) (h : perimeter = 40) : ∃ A : ℝ, A = 100 := by
  have h1 : ∃ s : ℝ, 4 * s = perimeter := by
    use perimeter / 4
    linarith

  cases h1 with s hs
  use s^2
  rw [hs, h]
  norm_num
  sorry

end square_area_l28_28205


namespace evaluate_expression_l28_28654

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end evaluate_expression_l28_28654


namespace comparison_of_logs_l28_28693

noncomputable def a : ℝ := Real.logb 4 6
noncomputable def b : ℝ := Real.logb 4 0.2
noncomputable def c : ℝ := Real.logb 2 3

theorem comparison_of_logs : c > a ∧ a > b := by
  sorry

end comparison_of_logs_l28_28693


namespace max_m_plus_n_l28_28945

def grid : Type := matrix (fin 7) (fin 7) bool

noncomputable def m (g : grid) : ℕ :=
  finset.card {j : fin 7 | finset.card {i : fin 7 | g i j = tt} < 4}

noncomputable def n (g : grid) : ℕ :=
  finset.card {i : fin 7 | finset.card {j : fin 7 | g i j = tt} > 3}

theorem max_m_plus_n (g : grid) : 
  ∃ m n : ℕ, (m = (finset.card {j : fin 7 | finset.card {i : fin 7 | g i j = tt} < 4}) ∧ 
              n = (finset.card {i : fin 7 | finset.card {j : fin 7 | g i j = tt} > 3}) ∧ 
              m + n = 12) :=
sorry

end max_m_plus_n_l28_28945


namespace cereal_ratio_l28_28620

variable (A B : ℝ)

axiom cereal_a_sugar : 0.10 * A
axiom cereal_b_sugar : 0.02 * B
axiom total_weight : A + B
axiom desired_sugar_content : 0.06 * (A + B)

theorem cereal_ratio (h: 0.10 * A + 0.02 * B = 0.06 * (A + B)) : A = B :=
  by
  -- Proceed with proving the theorem
  sorry

end cereal_ratio_l28_28620


namespace sin_B_value_a_range_two_solutions_l28_28748

-- Definitions and conditions of the triangle
def side_a : ℝ := 3
def side_b : ℝ := 2
def angle_A : ℝ := real.pi / 3  -- 60 degrees in radians

-- First part: Calculating sin B
theorem sin_B_value (h₁ : side_a = 3) (h₂ : side_b = 2) (h₃ : angle_A = real.pi / 3) : 
  real.sin B = real.sqrt 3 / 3 :=
sorry

-- Second part: Range for a when the triangle has two solutions
theorem a_range_two_solutions (h₂ : side_b = 2) (h₃ : angle_A = real.pi / 3) : 
  ∀ a : ℝ, (real.sqrt 3 / 2 < real.sin (real.arcsin (side_b * real.sin (angle_A) / a)) < 1) ↔ (real.sqrt 3 < a ∧ a < 2) :=
sorry

end sin_B_value_a_range_two_solutions_l28_28748


namespace max_intersection_points_l28_28908

theorem max_intersection_points (C L : Type) [fintype C] [fintype L] (hC : fintype.card C = 3) (hL : fintype.card L = 2) :
  (∑ c1 in C, ∑ c2 in (C \ {c1}), 2) + (∑ l in L, ∑ c in C, 2) + 1 = 19 :=
by
  sorry

end max_intersection_points_l28_28908


namespace sum_of_distinct_divisors_l28_28803

theorem sum_of_distinct_divisors (n : ℕ) (a : ℕ) (h_n : 0 < n) (h_a : a ≥ nat.factorial n) :
  ∃ (d : list ℕ), (∀ x ∈ d, x ∣ nat.factorial n) ∧ d.nodup ∧ d.length ≥ n ∧ a = d.sum :=
sorry

end sum_of_distinct_divisors_l28_28803


namespace number_of_small_triangles_needed_l28_28117

-- Definitions using given conditions
def side_length_large := 15
def side_length_small := 3
def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

-- Lean statement for the proof problem
theorem number_of_small_triangles_needed :
  (area_equilateral_triangle side_length_large) / (area_equilateral_triangle side_length_small) = 25 := by
  sorry

end number_of_small_triangles_needed_l28_28117


namespace simplify_expression_l28_28465

theorem simplify_expression (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 3) :
  (2 * x^(-1 / 3) / (x^(2 / 3) - 3 * x^(-1 / 3)) - x^(2 / 3) / (x^(5 / 3) - x^(2 / 3)) - (x + 1) / (x^2 - 4 * x + 3)) = 0 :=
by
  sorry

end simplify_expression_l28_28465


namespace can_increase_averages_l28_28351

def grades_group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def grades_group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : Float :=
  (grades.map Nat.toFloat).sum / grades.length

def new_average (grades : List ℕ) (grade_to_remove_or_add : ℕ) (adding : Bool) : Float :=
  if adding
  then (grades.map Nat.toFloat).sum + grade_to_remove_or_add / (grades.length + 1)
  else (grades.map Nat.toFloat).sum - grade_to_remove_or_add / (grades.length - 1)

theorem can_increase_averages :
  ∃ grade,
    grade ∈ grades_group1 ∧
    average grades_group1 < new_average grades_group1 grade false ∧
    average grades_group2 < new_average grades_group2 grade true :=
  sorry

end can_increase_averages_l28_28351


namespace length_DF_l28_28705

noncomputable def focus_of_ellipse (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)

theorem length_DF (a b : ℝ)
  (h1 : a = 6)
  (h2 : b = 4)
  (C F D : ℝ × ℝ)
  (hF : F = (focus_of_ellipse a b, 0))
  (hC1 : F = (C.1 + 2 * real.sqrt 5, C.2))
  (hC2 : C.2^2 = 16 - (4/9) * C.1^2)
  (hCF : dist C F = 2) 
  : dist D F = 2 :=
sorry

end length_DF_l28_28705


namespace eval_simplified_expression_l28_28848

theorem eval_simplified_expression : 
  (λ x : ℝ, ( (x - 1) / (x - 2) * (x^2 - 4) / (x^2 - 2*x + 1) - 2 / (x - 1) ) ) 2 = 2 :=
by 
  sorry

end eval_simplified_expression_l28_28848


namespace kimberly_peanuts_per_visit_l28_28786

theorem kimberly_peanuts_per_visit 
  (trips : ℕ) (total_peanuts : ℕ) 
  (h1 : trips = 3) 
  (h2 : total_peanuts = 21) : 
  total_peanuts / trips = 7 :=
by
  sorry

end kimberly_peanuts_per_visit_l28_28786


namespace farthest_vertex_l28_28775

structure ConvexQuadrilateral (A B C D E : Type) where
  convex : Prop

def dividesEqualArea {A B C D E : Type} [ConvexQuadrilateral A B C D E] : Prop := sorry

theorem farthest_vertex
  {A B C D E : Type}
  [ConvexQuadrilateral A B C D E]
  (h_divides : dividesEqualArea) :
  farthestVertex == B :=
sorry

end farthest_vertex_l28_28775


namespace volume_of_prism_l28_28583

-- Define the conditions as properties of the regular triangular prism and the sphere
axiom regular_triang_prism_inscribed_in_sphere_with_radius
  (ABC A1 B1 C1 : Type)
  (R : ℝ) (D : ℝ) (AD : ℝ)
  (CD : D = 2 * R)
  (AD_val : AD = 2 * sqrt 6)
  (R_val : R = 3)
  : ∃ V : ℝ, V = 6 * sqrt 15

-- The theorem states that given the conditions, the volume V of the prism is 6√15
theorem volume_of_prism
  (ABC A1 B1 C1 : Type)
  (R : ℝ) (D : ℝ) (AD : ℝ)
  (CD : D = 2 * R)
  (AD_val : AD = 2 * sqrt 6)
  (R_val : R = 3)
  : ∃ V : ℝ, V = 6 * sqrt 15 :=
by {
  apply regular_triang_prism_inscribed_in_sphere_with_radius;
  assumption,
}

end volume_of_prism_l28_28583


namespace factorization_correctness_l28_28916

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end factorization_correctness_l28_28916


namespace problem_statement_l28_28720

variables (a b : Vector ℝ 3)
variable (θ : ℝ) -- angle between a and b in radians

-- Conditions of the problem
def angle_condition : Prop := θ = π / 6
def magnitude_a : Prop := ‖a‖ = sqrt 3
def magnitude_b : Prop := ‖b‖ = 1

-- First part of the problem
def first_part : Prop := ‖a - 2 • b‖ = 1

-- Definitions for the second part
def p := a + 2 • b
def q := a - 2 • b

def projection_condition : Prop :=
  ‖q‖ = 1 ∧ (p ⬝ q) / ‖q‖ = -1

-- Theorem combining both parts
theorem problem_statement (h : angle_condition) (ha : magnitude_a) (hb : magnitude_b) : first_part ∧ projection_condition :=
by sorry

end problem_statement_l28_28720


namespace towel_area_decrease_28_percent_l28_28973

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := L * 0.80

def new_breadth (B : ℝ) : ℝ := B * 0.90

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

def percentage_decrease_in_area (L B : ℝ) : ℝ :=
  ((original_area L B - new_area L B) / original_area L B) * 100

theorem towel_area_decrease_28_percent (L B : ℝ) :
  percentage_decrease_in_area L B = 28 := by
  sorry

end towel_area_decrease_28_percent_l28_28973


namespace sampling_method_correctness_l28_28553

theorem sampling_method_correctness :
  let num_urban_students := 150 
  let num_rural_students := 150
  let num_products := 100
  let num_student_sample := 100
  let num_product_sample := 10
  (num_urban_students + num_rural_students == 300) →
  (num_student_sample == 100) →
  (num_products == 100) →
  (num_product_sample == 10) →
  stratified_sampling (num_urban_students, num_rural_students) num_student_sample ∧
  simple_random_sampling num_products num_product_sample :=
by
  sorry

end sampling_method_correctness_l28_28553


namespace max_points_of_intersection_l28_28895

theorem max_points_of_intersection (n m : ℕ) : n = 3 → m = 2 → 
  (let circle_intersections := (n * (n - 1)) / 2 * 2;
       line_circle_intersections := m * n * 2;
       line_intersections := (m * (m - 1)) / 2;
       total_intersections := circle_intersections + line_circle_intersections + line_intersections
   in total_intersections = 19) :=
by
  intros hn hm
  let circle_intersections := (n * (n - 1)) / 2 * 2
  let line_circle_intersections := m * n * 2
  let line_intersections := (m * (m - 1)) / 2
  let total_intersections := circle_intersections + line_circle_intersections + line_intersections
  rw [hn, hm] at *
  calc total_intersections
      = ((3 * 2) * 2) + (2 * 3 * 2) + 1 : by sorry
      ... = 6 + 12 + 1  : by sorry
      ... = 19         : by sorry

end max_points_of_intersection_l28_28895


namespace inradius_is_2sqrt69_l28_28410

-- Definitions based on conditions
variables {A B C I : Point} {AB AC : ℝ}
variables (BC : ℝ) (IC : ℝ)

-- Conditions
def is_isosceles_triangle : Prop := AB = AC
def base_length : Prop := BC = 40
def incenter_distance : Prop := IC = 26

-- Main statement
theorem inradius_is_2sqrt69 (h_tri : is_isosceles_triangle AB AC)
  (h_base : base_length BC) (h_incenter : incenter_distance IC) :
  exists (r : ℝ), r = 2 * Real.sqrt 69 :=
sorry

end inradius_is_2sqrt69_l28_28410


namespace continuous_inequality_to_convexity_l28_28040

variables {f : ℝ → ℝ} {x1 x2 : ℝ}

theorem continuous_inequality_to_convexity 
  (h_cont : Continuous f) 
  (h_ineq : ∀ x1 x2, f ((x1 + x2) / 2) ≤ (f x1 + f x2) / 2) :
  ∀ (x1 x2 : ℝ) (λ : ℝ), 0 < λ ∧ λ < 1 → f (λ * x1 + (1 - λ) * x2) ≤ λ * f x1 + (1 - λ) * f x2 :=
by
  sorry

end continuous_inequality_to_convexity_l28_28040


namespace determine_pairs_of_positive_integers_l28_28646

open Nat

theorem determine_pairs_of_positive_integers (n p : ℕ) (hp : Nat.Prime p) (hn_le_2p : n ≤ 2 * p)
    (hdiv : (p - 1)^n + 1 ∣ n^(p - 1)) : (n = 1) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
  sorry

end determine_pairs_of_positive_integers_l28_28646


namespace volume_of_prism_l28_28859

theorem volume_of_prism (x y z : ℕ) 
    (h1 : 2 * x + 2 * y = 38) 
    (h2 : y + z = 14) 
    (h3 : x + z = 11) : 
    x * y * z = 264 := 
by 
  -- simplifying condition 1
  have h4 : x + y = 19, from Nat.mul_right_injective (by norm_num) (by assumption),
  -- deriving the sum of all dimensions
  have h5 : x + y + z = 22,
  {
    rw [←h2, ←h3] at h4,
    linarith,
  },
  -- deriving individual dimensions
  have x_val : x = 22 - 14, from by linarith,
  have y_val : y = 22 - 11, from by linarith,
  have z_val : z = 22 - 19, from by linarith,
  -- calculating the volume
  rw [x_val, y_val, z_val],
  norm_num,
  apply x * y * z

end volume_of_prism_l28_28859


namespace find_beta_l28_28300

theorem find_beta 
  (α β : ℝ)
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) :
  β = Real.pi / 3 := 
sorry

end find_beta_l28_28300


namespace find_X_is_K_l28_28756

variable (d : ℕ)

def date_D : ℕ := d + 5
def date_E : ℕ := d + 7
def date_F : ℕ := d + 20

def sum_DF := date_D d + date_F d
def sum_EX (date_X : ℕ) := date_E d + date_X

theorem find_X_is_K : sum_DF d = sum_EX d (d + 18) :=
by
suffices : date_D d + date_F d = date_E d + (d + 18)
· exact this
· calc
    date_D d + date_F d
      = (d + 5) + (d + 20) : rfl
  ... = 2d + 25           : by ring
  ... = (d + 7) + (d + 18) : by ring
  ... = date_E d + (d + 18) : rfl

end find_X_is_K_l28_28756


namespace students_remaining_after_three_stops_l28_28308

theorem students_remaining_after_three_stops (initial_students : ℕ) (h_initial : initial_students = 48) :
  let after_first_stop := initial_students / 2,
      after_second_stop := after_first_stop / 2,
      after_third_stop := after_second_stop / 2
  in after_third_stop = 6 :=
by
  simp [h_initial]
  sorry

end students_remaining_after_three_stops_l28_28308


namespace inequality_solution_l28_28468

theorem inequality_solution (x : ℝ) : 
  (x = 0 ∨ (1 < x ∧ x ≤ 3/2)) → (x / (x - 1) ≥ 2 * x) := 
by
  intro hx
  cases hx
  . rw hx
    linarith
  . have : x ≠ 1 := by linarith
    have H : (x * (-2 * x + 3)) / (x - 1) ≥ 0 := by sorry
    have : x / (x - 1) - 2 * x = (x * (-2 * x + 3)) / (x - 1) := by sorry
    linarith
  sorry

end inequality_solution_l28_28468


namespace minimum_visible_sum_of_large_cube_l28_28555

theorem minimum_visible_sum_of_large_cube : 
  let corners := 8 * 6 in
  let edges := 24 * 3 in
  let face_centers := 24 * 1 in
  corners + edges + face_centers = 144 :=
by
  sorry

end minimum_visible_sum_of_large_cube_l28_28555


namespace count_integers_between_with_remainder_3_l28_28297

theorem count_integers_between_with_remainder_3 :
  {n : ℤ // 30 ≤ n ∧ n <= 90 ∧ n % 7 = 3}.card = 9 := sorry

end count_integers_between_with_remainder_3_l28_28297


namespace rectangle_area_1024_l28_28768

-- Define the given conditions about the rectangle
def rectangle_split_into_squares (ABCD : Type) (A B C D : ABCD)
  (side_length : ℝ) : Prop :=
  side_length > 0 ∧ 
  (B.dist A = 4 * side_length) ∧
  (D.dist A = side_length) ∧ 
  (C.dist B = D.dist A) ∧ 
  (C.dist B = 4 * side_length) ∧
  (C.dist D = D.dist A)

-- Define the perimeter condition 
def perimeter_condition (ABCD : Type) (A B C D : ABCD)
  (side_length : ℝ) : Prop :=
  (10 * side_length) = 160

-- Prove that the area of rectangle ABCD is 1024 square cm
theorem rectangle_area_1024 {ABCD : Type} (A B C D : ABCD) (side_length : ℝ)
  (h_squares : rectangle_split_into_squares ABCD A B C D side_length)
  (h_perimeter : perimeter_condition ABCD A B C D side_length) : 
  (4 * side_length^2) = 1024 :=
by
  sorry

end rectangle_area_1024_l28_28768


namespace common_area_l28_28044

structure Triangle (α : Type) := 
(A B C : α)

variables {α : Type} [LinearOrder α] [Field α]

def area (t : Triangle α) : α → α := sorry

variables (A B C D E F : α) {AD DB : α} (ABC ADE DBEF : Triangle α)
variables h_areaABC : area ABC 20 
variables h_AD : AD = 3
variables h_DB : DB = 4
variables h_point_conditions : D ∈ segment A B ∧ E ∈ segment B C ∧ F ∈ segment C A

theorem common_area : 
  area ADE = 180 / 49 ∧ area DBEF = 180 / 49 :=
by
  sorry

end common_area_l28_28044


namespace ellipse_eccentricity_roots_l28_28467

theorem ellipse_eccentricity_roots
  (h k : ℝ)
  (a : ℝ)
  (ellipse : ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / (2 * a)^2 = 1)
  (roots : ∀ (z : ℂ), (z-3) * (z^2+6*z+13) * (z^2+8*z+20) = 0 → 
    (∃ x y : ℝ, z = x + y * im ∧ ellipse x y)) :
  ∃ m n : ℕ, nat.coprime m n ∧ (sqrt (3 : ℝ)) = (real.sqrt (m / (n : ℝ)) : ℝ) ∧ m + n = 3 := 
sorry

end ellipse_eccentricity_roots_l28_28467


namespace gnuff_tutoring_minutes_l28_28288

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l28_28288


namespace max_intersection_points_circles_lines_l28_28903

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l28_28903


namespace probability_log_condition_l28_28454

noncomputable def select_from_set : List ℕ := [2, 3, 4]

def log_condition (x y : ℕ) : Prop := Real.log y / Real.log x ≤ 1 / 2

def count_favorable_cases : ℕ :=
  select_from_set.foldl (λ acc x => acc + select_from_set.count (λ y => log_condition x y)) 0

def total_cases : ℕ := select_from_set.length ^ 2

theorem probability_log_condition :
  (count_favorable_cases : ℝ) / total_cases = 1 / 6 := sorry

end probability_log_condition_l28_28454


namespace percentage_increase_in_area_l28_28085

-- Defining the lengths and widths in terms of real numbers
variables (L W : ℝ)

-- Defining the new lengths and widths
def new_length := 1.2 * L
def new_width := 1.2 * W

-- Original area of the rectangle
def original_area := L * W

-- New area of the rectangle
def new_area := new_length L * new_width W

-- Proof statement for the percentage increase
theorem percentage_increase_in_area : 
  ((new_area L W - original_area L W) / original_area L W) * 100 = 44 :=
by
  sorry

end percentage_increase_in_area_l28_28085


namespace max_distance_l28_28314

noncomputable def pointOnEllipse (θ : ℝ) : ℝ × ℝ :=
(√2 * Real.cos θ, Real.sin θ)

def distanceFromPointToLine (P : ℝ × ℝ) : ℝ :=
(abs (P.1 - P.2 + 1)) / Real.sqrt 2

theorem max_distance (θ : ℝ) : 
  ∃ (θ : ℝ), distanceFromPointToLine (pointOnEllipse θ) = (Real.sqrt 6 + Real.sqrt 2) / 2 := sorry

end max_distance_l28_28314


namespace water_remaining_l28_28784

variable (initial_amount : ℝ) (leaked_amount : ℝ)

theorem water_remaining (h1 : initial_amount = 0.75)
                       (h2 : leaked_amount = 0.25) :
  initial_amount - leaked_amount = 0.50 :=
by
  sorry

end water_remaining_l28_28784


namespace age_of_15th_student_l28_28933

noncomputable def total_age_of_class (n : ℕ) (average_age : ℕ) :=
  n * average_age

noncomputable def group_age (num_students : ℕ) (average_age : ℕ) :=
  num_students * average_age

theorem age_of_15th_student :
  let total_age := total_age_of_class 15 15 in
  let group1_age := group_age 7 14 in
  let group2_age := group_age 7 16 in
  let combined_group_age := group1_age + group2_age in
  total_age - combined_group_age = 15 :=
by
  intros
  sorry

end age_of_15th_student_l28_28933


namespace cave_depth_l28_28749

/-- 
  Problem statement:
  Given the total time t = 25 seconds, the speed of sound in air c, 
  and the acceleration due to gravity g, prove that the depth 
  of the cave x is 1867 meters.
--/
theorem cave_depth (c g : ℝ) (h_c : 343 ≤ c) (h_g : g > 0) : 
  ∃ x : ℝ, x = 1867 ∧ (let t := 25 in t = x / c + sqrt (2 * x / g)) :=
sorry

end cave_depth_l28_28749


namespace count_rational_numbers_l28_28146

def is_rational (x : ℝ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

theorem count_rational_numbers :
  let numbers := [23 / 7, Real.sqrt 16, 0, Real.cbrt 9, -Real.pi / 2, 2.2023010010001] in
  list.countp is_rational numbers = 4 :=
by
  sorry

end count_rational_numbers_l28_28146


namespace simplify_ratio_l28_28667

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def a_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), 1 / (binomial n k)

noncomputable def c_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), k^2 / (binomial n k)

theorem simplify_ratio (n : ℕ) (hn : 0 < n) : a_n n / c_n n = 1 / n^2 := by
  sorry

end simplify_ratio_l28_28667


namespace perfect_square_trinomial_k_l28_28299

theorem perfect_square_trinomial_k (k : ℤ) :
  (∀ x : ℝ, 9 * x^2 + 6 * x + k = (3 * x + 1) ^ 2) → (k = 1) :=
by
  sorry

end perfect_square_trinomial_k_l28_28299


namespace original_average_marks_l28_28011

theorem original_average_marks 
  (pupils : ℕ) 
  (scores_4 : list ℕ) 
  (average_new : ℕ) 
  (total_pupils : pupils = 21) 
  (specific_scores : scores_4 = [25, 12, 15, 19]) 
  (new_average_17 : average_new = 44) :
  let total_marks_17 := 17 * average_new in
  let total_marks := (21 * 39 : ℕ) in
  let removed_4_sum := list.sum scores_4 in
  total_marks = total_marks_17 + removed_4_sum :=
by 
  sorry

end original_average_marks_l28_28011


namespace equilateral_triangle_area_l28_28602

theorem equilateral_triangle_area (h : ℝ) (p : ℝ) (a : ℝ) :
  h = 9 → p = 36 → a = (sqrt 3 / 4) * (2 * sqrt(p / 3 / sqrt 3)) ^ 2 → a = 54 := by
  intros h_eq p_eq a_eq
  rw [h_eq, p_eq, a_eq]
  sorry

end equilateral_triangle_area_l28_28602


namespace find_C_l28_28251

theorem find_C (C : ℝ) (h : ∃ (d : ℝ) (A B C1 C2 : ℝ), d = Real.sqrt 10 ∧ A = 3 ∧ B = -1 ∧ C1 = 3 ∧ C2 = C ∧ d = |C2 - C1| / Real.sqrt (A^2 + B^2)) : C = 13 ∨ C = -7 :=
by
  cases h with d hd,
  rcases hd with ⟨h_dist_eq, hA, hB, hC1, hC2, h_dist_formula⟩,
  have h1 : 3 = hA := hA.symm,
  have h2 : -1 = hB := hB.symm,
  have h3 : 3 = hC1 := hC1.symm,
  have h4 : C = hC2 := hC2.symm,
  rw [h1, h2, h3, h4] at h_dist_formula,
  sorry

end find_C_l28_28251


namespace equal_share_compensation_l28_28526

theorem equal_share_compensation (x y z : ℕ) 
  (h1 : x * x = 10 * y + z) 
  (h2 : 0 < z) 
  (h3 : z < 10) 
  (h4 : z = 6) 
  (h5 : y % 2 = 1) :
  let compensation := (10 - z) / 2 in
  compensation = 2 :=
by
  sorry

end equal_share_compensation_l28_28526


namespace first_number_is_twenty_l28_28484

theorem first_number_is_twenty (x : ℕ) : 
  (x + 40 + 60) / 3 = ((10 + 70 + 16) / 3) + 8 → x = 20 := 
by 
  sorry

end first_number_is_twenty_l28_28484


namespace squaring_key_presses_l28_28559

theorem squaring_key_presses : 
  ∃ n : ℕ, (n ≥ 0) ∧ (∀ k, k = 3 → (square_iter k n > 1000)) := 
sorry

def square_iter (x : ℕ) (n : ℕ) : ℕ := 
  if n = 0 then x 
  else square_iter (x * x) (n - 1)

end squaring_key_presses_l28_28559


namespace golden_ratio_in_range_l28_28504

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_ratio_in_range : 0.6 < golden_ratio ∧ golden_ratio < 0.7 :=
by
  sorry

end golden_ratio_in_range_l28_28504


namespace triangle_area_l28_28698

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry

theorem triangle_area (h1 : ∥z1∥ = 4)
                     (h2 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0) :
    (1/2) * (abs z1) * (abs (z2 - z1)) = 8 * real.sqrt 3 := 
sorry

end triangle_area_l28_28698


namespace C_is_necessary_but_not_sufficient_for_A_l28_28247

-- Define C, B, A to be logical propositions
variables (A B C : Prop)

-- The conditions given
axiom h1 : A → B
axiom h2 : ¬ (B → A)
axiom h3 : B ↔ C

-- The conclusion: Prove that C is a necessary but not sufficient condition for A
theorem C_is_necessary_but_not_sufficient_for_A : (A → C) ∧ ¬ (C → A) :=
by
  sorry

end C_is_necessary_but_not_sufficient_for_A_l28_28247


namespace can_increase_averages_by_transfer_l28_28339

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ := grades.sum / grades.length

theorem can_increase_averages_by_transfer :
    ∃ (student : ℝ) (new_group1_grades new_group2_grades : List ℝ),
      student ∈ group1_grades ∧
      new_group1_grades = (group1_grades.erase student) ∧
      new_group2_grades = student :: group2_grades ∧
      average new_group1_grades > average group1_grades ∧ 
      average new_group2_grades > average group2_grades :=
by
  sorry

end can_increase_averages_by_transfer_l28_28339


namespace find_angle_A_find_range_bc_l28_28369

-- Definitions related to the acute triangle and the given equation
variables {a b c : ℝ}
variables {A B C : ℝ}
hypothesis (h₁ : 0 < A ∧ A < π / 2)
hypothesis (h₂ : 0 < B ∧ B < π / 2)
hypothesis (h₃ : 0 < C ∧ C < π / 2)
hypothesis (h₄ : A + B + C = π)
hypothesis (h₅ : b^2 - a^2 - c^2 = ac * (cos (A + C)) / (sin A * cos A))

-- Part 1: Proving the value of angle A
theorem find_angle_A : A = π / 4 :=
sorry

-- Part 2: Finding the range of values for bc given a = sqrt(2)
variables (sqrt_two : a = real.sqrt 2)
theorem find_range_bc (h₆ : b^2 - (real.sqrt 2)^2 - c^2 = (real.sqrt 2) * c * (cos (π / 4 + C)) / (sin (π / 4) * cos (π / 4))) : 
∃ v : set ℝ, v = set.Icc (-2 * real.sqrt 2) (2 * real.sqrt 2) ∧ bc ∈ v :=
sorry

end find_angle_A_find_range_bc_l28_28369


namespace no_real_pairs_in_arithmetic_progression_l28_28179

noncomputable def arithmetic_progression_pairs_count : ℕ :=
let pairs := {p : ℝ × ℝ | ∃ a b : ℝ, p = (a, b) ∧ a = (15 + b) / 2 ∧ a + a * b = 2 * b} in
  if h : pairs = ∅ then 0 else sorry

theorem no_real_pairs_in_arithmetic_progression :
  @arithmetic_progression_pairs_count = 0 :=
sorry

end no_real_pairs_in_arithmetic_progression_l28_28179


namespace truck_travel_yards_in_3_minutes_l28_28143

-- Definitions
def feet_travelled_per_t_seconds (b t : ℝ) := b / 6
def feet_in_a_yard := 3
def minutes_to_seconds (minutes : ℝ) := 60 * minutes
def seconds_in_3_minutes := minutes_to_seconds 3
def yards_travelled_in_seconds (b t seconds : ℝ) :=
  (feet_travelled_per_t_seconds b t / t) * seconds

-- Problem Statement
theorem truck_travel_yards_in_3_minutes (b t : ℝ) :
  yards_travelled_in_seconds b t seconds_in_3_minutes / feet_in_a_yard = 10 * b / t :=
by
  sorry

end truck_travel_yards_in_3_minutes_l28_28143


namespace angles_in_triangle_l28_28254

theorem angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 2 * B = 3 * A) (h3 : 5 * A = 2 * C) :
  B = 54 ∧ C = 90 :=
by
  sorry

end angles_in_triangle_l28_28254


namespace graph_not_in_first_quadrant_l28_28298

theorem graph_not_in_first_quadrant (a b : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : b < -1) :
  ∀ x : ℝ, ¬(f x > 0 ∧ f x < 0) :=
by
  let f := λ x, a^x + b
  sorry

end graph_not_in_first_quadrant_l28_28298


namespace red_black_ball_ratio_l28_28752

theorem red_black_ball_ratio (R B x : ℕ) (h1 : 3 * R = B + x) (h2 : 2 * R + x = B) :
  R / B = 2 / 5 := by
  sorry

end red_black_ball_ratio_l28_28752


namespace number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l28_28565

-- Definitions for the problem
def n : Nat := 12

-- Statement 1: Number of diagonals in a dodecagon
theorem number_of_diagonals_dodecagon (n : Nat) (h : n = 12) : (n * (n - 3)) / 2 = 54 := by
  sorry

-- Statement 2: Sum of interior angles in a dodecagon
theorem sum_of_interior_angles_dodecagon (n : Nat) (h : n = 12) : 180 * (n - 2) = 1800 := by
  sorry

end number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l28_28565


namespace pool_capacity_l28_28923

theorem pool_capacity (C : ℝ) (h1 : C * 0.70 = C * 0.40 + 300)
  (h2 : 300 = C * 0.30) : C = 1000 :=
sorry

end pool_capacity_l28_28923


namespace traffic_accident_emergency_number_l28_28021

theorem traffic_accident_emergency_number (A B C D : ℕ) (h1 : A = 122) (h2 : B = 110) (h3 : C = 120) (h4 : D = 114) : 
  A = 122 := 
by
  exact h1

end traffic_accident_emergency_number_l28_28021


namespace count_valid_sequences_l28_28725

def has_opposite_parity (a b : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)

def valid_sequence (s : Fin 7 → ℕ) : Prop :=
  ∀ i : Fin 6, has_opposite_parity (s i.cast_succ) (s i.succ)

theorem count_valid_sequences : 
  (∃ s : Fin 7 → ℕ, valid_sequence s) → 
  ∃ n : ℕ, n = 156250 :=
begin
  sorry
end

end count_valid_sequences_l28_28725


namespace tutoring_minutes_l28_28285

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l28_28285


namespace least_possible_value_of_smallest_integer_l28_28932

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), 
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B + C + D) / 4 = 68 →
    D = 90 →
    A = 5 :=
by
  intros A B C D h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end least_possible_value_of_smallest_integer_l28_28932


namespace angle_BIC_125_l28_28769

theorem angle_BIC_125 (A B C I : Type) (a b c : ℝ)
  (hB_bisector : bisector (∠ B I A))
  (hC_bisector : bisector (∠ C I A))
  (h_angle_A : ∠ A = 70) :
∠ B I C = 125 :=
by 
  sorry

end angle_BIC_125_l28_28769


namespace find_original_amount_l28_28996

-- Let X be the original amount of money in Christina's account.
variable (X : ℝ)

-- Condition 1: Remaining balance after transferring 20% is $30,000.
def initial_transfer (X : ℝ) : Prop :=
  0.80 * X = 30000

-- Prove that the original amount before the initial transfer was $37,500.
theorem find_original_amount (h : initial_transfer X) : X = 37500 :=
  sorry

end find_original_amount_l28_28996


namespace axis_of_symmetry_l28_28735

-- Define the condition that f(x) = f(3 - x) for all x.
variable {f : ℝ → ℝ}
axiom symmetry_property : ∀ x : ℝ, f(x) = f(3 - x)

-- The theorem statement that the graph of y = f(x) is symmetric with respect to the line x = 1.5.
theorem axis_of_symmetry : ∀ x : ℝ, f x = f (3 - x) → x = 1.5 :=
by
  intro x hx
  -- proof is omitted
  sorry

end axis_of_symmetry_l28_28735


namespace problem_statement_l28_28214

theorem problem_statement :
  ∃ (n : ℕ), n = 101 ∧
  (∀ (x : ℕ), x < 4032 → ((x^2 - 20) % 16 = 0) ∧ ((x^2 - 16) % 20 = 0) ↔ (∃ k1 k2 : ℕ, (x = 80 * k1 + 6 ∨ x = 80 * k2 + 74) ∧ k1 + k2 + 1 = n)) :=
by sorry

end problem_statement_l28_28214


namespace find_f1_plus_fneg1_l28_28267

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then 2^x else x

theorem find_f1_plus_fneg1 : f 1 + f (-1) = 1 :=
by
  -- here is a hint on our logic, but the actual proof is omitted
  -- f(1) = 2 since 1 > 0
  -- f(-1) = -1 since -1 <= 0
  -- then, f(1) + f(-1) = 2 + (-1) = 1
  sorry

end find_f1_plus_fneg1_l28_28267


namespace parabola_focus_tangent_properties_l28_28274

-- Definitions and conditions based on the problem
def parabola_C (x y : ℝ) : Prop := x^2 = 4 * y
def focus_F : ℝ × ℝ := (0, 1)
def vertex_O : ℝ × ℝ := (0, 0)
def moving_point_P (x y : ℝ) : Prop := parabola_C x y ∧ (x ≠ 0 ∨ y ≠ 0)

-- Given a point on the parabola, this defines the tangent at that point.
def tangent_line_at_P (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop := 
  ∃ (x0 y0 : ℝ), P = (x0, y0) ∧ y0 = (1/4) * x0^2 ∧
  (∀ x, l x = (1/2) * x0 * x - (1/4) * x0^2)

-- Tangent intersects the x-axis at T. 
def intersects_x_axis_at_T (l : ℝ → ℝ) (T : ℝ × ℝ) : Prop := 
  ∃ x_T : ℝ, T = (x_T, 0) ∧ x_T = 1/2 * (classical.some (classical.some_spec (tangent_line_at_P)))

-- O perpendicular to line l and meets at M
def perpendicular_from_O_to_Tangent (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop := 
  ∃ x_M : ℝ, M = (classical.some (classical.some_spec))) (l x_M) ∧ x_M = 0 ∧ l 0 = M.snd

-- L intersects line PF at N
def intersects_PF_at_N (P F M : ℝ × ℝ) (N : ℝ × ℝ) : Prop := 
  ∃ xN yN : ℝ, N = (xN, yN) ∧ 
    (yN = -2 / (classical.some_spec) * xN) ∧ 
    yN = (P.2 - F.2) / (P.1 - F.1) * xN + 1

-- The probelem statement
theorem parabola_focus_tangent_properties (P T F M N : ℝ × ℝ) :
  moving_point_P P.1 P.2 →
  intersects_x_axis_at_T (by classical.some_spec) T →
  perpendicular_from_O_to_Tangent (by classical.some_spec) M →
  intersects_PF_at_N P F M N →
  F = (0, 1) ∧ 
  (let T_slope := -2 / P.1 in
   let MN_slope := -2 / P.1 in
   T_slope = MN_slope) ∧ 
  (sqrt ((N.1 - F.1)^2 + (N.2 - F.2)^2) = 1) :=
by
  intros hP hT hM hN
  sorry

end parabola_focus_tangent_properties_l28_28274


namespace real_m_for_purely_imaginary_complex_l28_28745

theorem real_m_for_purely_imaginary_complex (m : ℝ) : (∃ z : ℂ, z = (m + complex.i) / (1 + m * complex.i) ∧ ∀ (z_re : ℝ), z_re = (complex.re z) → z_re = 0) → m = 0 :=
by
  intros h
  cases h with z h1
  cases h1 with hz z_re_zero
  sorry

end real_m_for_purely_imaginary_complex_l28_28745


namespace can_increase_average_l28_28335

def student_grades := List (String × Nat)

def group1 : student_grades := 
    [("Andreev", 5), ("Borisova", 3), ("Vasilieva", 5), ("Georgiev", 3),
     ("Dmitriev", 5), ("Evstigneeva", 4), ("Ignatov", 3), ("Kondratiev", 4),
     ("Leontieva", 3), ("Mironov", 4), ("Nikolaeva", 5), ("Ostapov", 5)]
     
def group2 : student_grades := 
    [("Alexeeva", 3), ("Bogdanov", 4), ("Vladimirov", 5), ("Grigorieva", 2),
     ("Davydova", 3), ("Evstahiev", 2), ("Ilina", 5), ("Klimova", 4),
     ("Lavrentiev", 5), ("Mikhailova", 3)]

def average_grade (grp : student_grades) : ℚ :=
    (grp.map (λ x => x.snd)).sum / grp.length

def updated_group (grp : List (String × Nat)) (student : String × Nat) : List (String × Nat) :=
    grp.erase student

def transfer_student (g1 g2 : student_grades) (student : String × Nat) : student_grades × student_grades :=
    (updated_group g1 student, student :: g2)

theorem can_increase_average (s : String × Nat) 
    (h1 : s ∈ group1)
    (h2 : s.snd = 4)
    (h3 : average_grade group1 < s.snd)
    (h4 : average_grade group2 > s.snd)
    : 
    let (new_group1, new_group2) := transfer_student group1 group2 s in 
    average_grade new_group1 > average_grade group1 ∧ 
    average_grade new_group2 > average_grade group2 :=
sorry

#eval can_increase_average ("Evstigneeva", 4)

end can_increase_average_l28_28335


namespace calculate_votes_cast_l28_28097

noncomputable def total_votes_cast (votes_candidate : ℕ) (margin : ℕ) (percentage_candidate : ℝ) (percentage_rival : ℝ) : ℝ :=
  let V := (votes_candidate + margin) / percentage_rival in V

theorem calculate_votes_cast (percentage_candidate percentage_rival : ℝ) (votes_candidate votes_rival : ℕ) (margin total_votes : ℕ) :
  (percentage_candidate = 0.35) →
  (percentage_rival = 0.65) →
  (votes_candidate = total_votes * percentage_candidate) →
  (votes_rival = total_votes * percentage_rival) →
  (votes_rival = votes_candidate + margin) →
  (margin = 2250) →
  (total_votes = 7500) :=
by
  sorry

end calculate_votes_cast_l28_28097


namespace problem_statements_l28_28978

open Real

theorem problem_statements :
  ¬(∀ x ∈ Ioo (-π / 3) (π / 6), (2 * cos (2 * x + π / 3)) > 0) ∧
  ((∀ x, cos (x + π / 3) = cos (2 * (π / 6 - x))) ∧
  (¬ (∀ x, tan (x + π / 3) = tan (π / 6 - x))) ∧
  (∀ x, 3 * sin (2 * (x - π / 6) + π / 3) = 3 * sin (2 * x)) :=
by
  sorry

end problem_statements_l28_28978


namespace cos_theta_zero_l28_28966

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (2, 3, -1)

def diagonal1 := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2, vector_a.3 + vector_b.3)
def diagonal2 := (vector_b.1 - vector_a.1, vector_b.2 - vector_a.2, vector_b.3 - vector_a.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem cos_theta_zero :
  dot_product diagonal1 diagonal2 = 0 ∧
  norm diagonal1 ≠ 0 ∧
  norm diagonal1 ≠ 0 →
  (0 : ℝ) = (dot_product diagonal1 diagonal2) / (norm diagonal1 * norm diagonal2) := 
by
  sorry

end cos_theta_zero_l28_28966


namespace max_area_rectangle_l28_28135

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l28_28135


namespace price_per_kilo_of_peanuts_l28_28961

namespace NutPrice

-- Define the price per kilo of cashew nuts
def priceCashewNutPerKilo := 210

-- Define the weight of cashew nuts bought
def weightCashewNut := 3

-- Define the weight of peanuts bought
def weightPeanut := 2

-- Define the total weight of nuts bought
def totalWeightNut := 5

-- Define the total price per kilo of nuts
def priceTotalNutPerKilo := 178

theorem price_per_kilo_of_peanuts :
  let totalCostCashew := weightCashewNut * priceCashewNutPerKilo in
  let totalCostNut := totalWeightNut * priceTotalNutPerKilo in
  let totalCostPeanut := totalCostNut - totalCostCashew in
  let pricePeanutPerKilo := totalCostPeanut / weightPeanut in
  pricePeanutPerKilo = 130 :=
by
  sorry

end NutPrice

end price_per_kilo_of_peanuts_l28_28961


namespace proposition_A_l28_28150

def tan (x : ℝ) : ℝ := Math.tan x

def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem proposition_A : ∃ α : ℝ, tan (degrees_to_radians (90 - α)) = 1 :=
by
  sorry

end proposition_A_l28_28150


namespace lattice_point_triangle_area_l28_28750

-- Define the overall constraints and the main problem theorem
theorem lattice_point_triangle_area :
  ∀ (P : Fin 6 → (ℤ × ℤ)),
    (∀ i : Fin 6, abs (P i).1 ≤ 2 ∧ abs (P i).2 ≤ 2) →
    ¬ (∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ collinear (P i) (P j) (P k)) →
    ∃ (i j k : Fin 6),
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
        triangle_area (P i) (P j) (P k) ≤ 2 :=
sorry

-- Helper function to determine if 3 points are collinear
def collinear (A B C : ℤ × ℤ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

-- Helper function to compute the area of a triangle given three points
def triangle_area (A B C : ℤ × ℤ) : ℚ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

end lattice_point_triangle_area_l28_28750


namespace angle_R_measure_l28_28372

-- Definitions for internal angles and congruence in the heptagon
variables (G E O M T R Y : ℝ)
variables (heptagon_sum : G + E + O + M + T + R + Y = 900)
variables (congruent_angles : G = E ∧ E = R)
variables (supplementary_angles : O + Y = 180)

theorem angle_R_measure : R = 144 :=
by
  -- translating assumptions into Lean
  have hr1 : G = R, from congruent_angles.left,
  have hr2 : E = R, from congruent_angles.right,
  -- sum of angles
  have heq : R + R + R + O + M + T + Y = 900, from
    calc R + E + G + O + M + T + Y
           = G + E + R + O + M + T + Y : by rw [hr1, hr2]
           ... = 900 : heptagon_sum,
  
  -- supplementary angle condition
  have hy : O + Y = 180 := supplementary_angles,
  sorry

end angle_R_measure_l28_28372


namespace golden_ratio_in_range_l28_28505

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_ratio_in_range : 0.6 < golden_ratio ∧ golden_ratio < 0.7 :=
by
  sorry

end golden_ratio_in_range_l28_28505


namespace price_reduction_equation_l28_28102

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l28_28102


namespace can_transfer_increase_average_l28_28358

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l28_28358


namespace three_digit_numbers_with_specific_digit_sums_l28_28513

theorem three_digit_numbers_with_specific_digit_sums : 
  (∃ N : ℕ, N >= 100 ∧ N < 1000 ∧ ∀ (N_minus_3 := N - 3) 
    (N_add_4 := N + 4) (N_div_5 := N / 5) (N_mul_6 := N * 6),
    N_minus_3.integral? ∧ N_add_4.integral? ∧ N_div_5.integral? ∧ N_mul_6.integral? ∧ 
    (∃ a : ℕ, (sum_digits N_minus_3 = a ∧ sum_digits N_add_4 = a + 1 ∧ 
                sum_digits N_div_5 = a + 2 ∧ sum_digits N_mul_6 = a + 3))) 
  = 4 :=
sorry

end three_digit_numbers_with_specific_digit_sums_l28_28513


namespace intersection_of_A_and_B_l28_28716

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | 1 ≤ x ∧ x < 4}

-- The theorem stating the problem
theorem intersection_of_A_and_B : A ∩ B = {1} :=
by
  sorry

end intersection_of_A_and_B_l28_28716


namespace spiders_hired_l28_28172

theorem spiders_hired (total_workers beavers : ℕ) (h_total : total_workers = 862) (h_beavers : beavers = 318) : (total_workers - beavers) = 544 := by
  sorry

end spiders_hired_l28_28172


namespace sin_neg_225_eq_sqrt2_div2_l28_28213

theorem sin_neg_225_eq_sqrt2_div2 :
  Real.sin (-225 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_225_eq_sqrt2_div2_l28_28213


namespace find_x_l28_28515

/-- 
Prove that the value of x is 25 degrees, given the following conditions:
1. The sum of the angles in triangle BAC: angle_BAC + 50° + 55° = 180°
2. The angles forming a straight line DAE: 80° + angle_BAC + x = 180°
-/
theorem find_x (angle_BAC : ℝ) (x : ℝ)
  (h1 : angle_BAC + 50 + 55 = 180)
  (h2 : 80 + angle_BAC + x = 180) :
  x = 25 :=
  sorry

end find_x_l28_28515


namespace find_x_l28_28764

theorem find_x :
  ∃ x : ℝ, ((56 ^ 2 + 56 ^ 2) / x ^ 2 = 8) ∧ x = 28 :=
by
  use 28
  split
  case left =>
    calc
      (56 ^ 2 + 56 ^ 2) / 28 ^ 2
        = (2 * 56 ^ 2) / 28 ^ 2 : by rw not-proved
        = 784 / 784 : by rw/ reduce
        = 1 : by norm_num
      end
  case right => rfl

axiom not-proved : 2 * 56 ^ 2 / 28 ^ 2 = 8 / 7

end find_x_l28_28764


namespace vasya_min_distinct_coeffs_l28_28558

theorem vasya_min_distinct_coeffs (P : ℝ[X]) (h_deg : P.degree = 9) (h_nonzero : ∀ (n : ℕ), P.coeff n ≠ 0) :
  ∃ (coeff_set : set ℝ), coeff_set.finite ∧ coeff_set.card = 9 :=
by
  sorry

end vasya_min_distinct_coeffs_l28_28558


namespace monotonic_increasing_interval_l28_28020

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 < x → (1 / 2 < x → (f (x + 0.1) > f x)) :=
by
  intro x hx h
  sorry

end monotonic_increasing_interval_l28_28020


namespace necessary_but_not_sufficient_l28_28013

theorem necessary_but_not_sufficient (M N : ℝ) (h1 : M > N) : 
  (M > N ↔ log 2 M > log 2 N) ↔ False :=
by
  sorry

end necessary_but_not_sufficient_l28_28013


namespace numbers_divisible_by_3_or_5_l28_28980

theorem numbers_divisible_by_3_or_5 (n : ℕ) (h : n = 120) : 
  (Finset.filter (λ x : ℕ, x % 3 = 0 ∨ x % 5 = 0) (Finset.range (n + 1))).card = 56 :=
by 
  sorry

end numbers_divisible_by_3_or_5_l28_28980


namespace horner_rule_example_l28_28994

def poly (x : ℕ) : ℕ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem horner_rule_example : poly 3 = 1642 :=
by {
  unfold poly,
  calc
    5 * 3^5 + 4 * 3^4 + 3 * 3^3 + 2 * 3^2 + 3 + 1
      = 5 * 243 + 4 * 81 + 3 * 27 + 2 * 9 + 3 + 1 : by simp
  ... = 1215 + 324 + 81 + 18 + 3 + 1 : by simp
  ... = 1642 : by simp
}

end horner_rule_example_l28_28994


namespace max_points_of_intersection_l28_28897

theorem max_points_of_intersection (n m : ℕ) : n = 3 → m = 2 → 
  (let circle_intersections := (n * (n - 1)) / 2 * 2;
       line_circle_intersections := m * n * 2;
       line_intersections := (m * (m - 1)) / 2;
       total_intersections := circle_intersections + line_circle_intersections + line_intersections
   in total_intersections = 19) :=
by
  intros hn hm
  let circle_intersections := (n * (n - 1)) / 2 * 2
  let line_circle_intersections := m * n * 2
  let line_intersections := (m * (m - 1)) / 2
  let total_intersections := circle_intersections + line_circle_intersections + line_intersections
  rw [hn, hm] at *
  calc total_intersections
      = ((3 * 2) * 2) + (2 * 3 * 2) + 1 : by sorry
      ... = 6 + 12 + 1  : by sorry
      ... = 19         : by sorry

end max_points_of_intersection_l28_28897


namespace game_necessarily_terminates_second_player_wins_l28_28943

-- Part (a)
theorem game_necessarily_terminates
  (n : ℕ) (h : n = 2009) :
  -- Initial condition: all cards start blue side up
  (∀ (i : ℕ), i < n → (cards₀ i = blue)) →
  -- Conditions for a move
  (∀ (k : ℕ), 0 < k ∧ k ≤ n - 50 → 
   (cards₀ k = blue) →
   (∀ (j : ℕ), j < 50 → flip (cards (k + j))) a :
  -- The game necessarily terminates
  sorry

-- Part (b)
theorem second_player_wins
  (n : ℕ) (h : n = 2009) :
  -- Initial condition: all cards start blue side up
  (∀ (i : ℕ), i < n → (cards₀ i = blue)) →
  -- Conditions for a move
  (∀ (k : ℕ), 0 < k ∧ k ≤ n - 50 → 
   (cards₀ k = blue) →
   (∀ (j : ℕ), j < 50 → flip (cards (k + j)) → 
   sorry

end game_necessarily_terminates_second_player_wins_l28_28943


namespace expression_not_prime_l28_28778
namespace MyNamespace

open Nat

-- Define the numbers involved.
def num1 : ℕ := 2011
def num2 : ℕ := 2111
def num3 : ℕ := 2500

-- Define the expression we are working with.
def expression : ℕ := num1 * num2 + num3

-- State the theorem: The expression is not a prime number.
theorem expression_not_prime : ¬ Prime expression := by
  -- Lean will skip the proof
  sorry

end MyNamespace

end expression_not_prime_l28_28778


namespace sum_of_powers_of_minus_one_l28_28991

theorem sum_of_powers_of_minus_one : (-1) ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 + (-1) ^ 2014 = -1 := by
  sorry

end sum_of_powers_of_minus_one_l28_28991


namespace xor_probability_l28_28086

theorem xor_probability (P_A P_B : ℝ) (hA : P_A = 0.5) (hB : P_B = 0.5) :
  P_A + P_B - 2 * P_A * P_B = 0.5 :=
by
  rw [hA, hB]
  norm_num
  sorry

end xor_probability_l28_28086


namespace max_points_of_intersection_l28_28896

theorem max_points_of_intersection (n m : ℕ) : n = 3 → m = 2 → 
  (let circle_intersections := (n * (n - 1)) / 2 * 2;
       line_circle_intersections := m * n * 2;
       line_intersections := (m * (m - 1)) / 2;
       total_intersections := circle_intersections + line_circle_intersections + line_intersections
   in total_intersections = 19) :=
by
  intros hn hm
  let circle_intersections := (n * (n - 1)) / 2 * 2
  let line_circle_intersections := m * n * 2
  let line_intersections := (m * (m - 1)) / 2
  let total_intersections := circle_intersections + line_circle_intersections + line_intersections
  rw [hn, hm] at *
  calc total_intersections
      = ((3 * 2) * 2) + (2 * 3 * 2) + 1 : by sorry
      ... = 6 + 12 + 1  : by sorry
      ... = 19         : by sorry

end max_points_of_intersection_l28_28896


namespace bug_returns_to_A_after_4_meters_l28_28364

-- Define the initial conditions and setup
def P : ℕ → ℝ
def Q : ℕ → ℝ

-- Initial conditions
axiom P_0 : P 0 = 1
axiom Q_0 : Q 0 = 0

-- Recurrence relations
axiom recurrence1 (n : ℕ) : P (n + 1) = 0
axiom recurrence2 (n : ℕ) : P (n + 2) = (1 / 3) * Q n
axiom recurrence3 (n : ℕ) : Q (n + 2) = (2 / 3) * Q n + P n

-- The target to prove
theorem bug_returns_to_A_after_4_meters : P 4 = 1 / 3 :=
by
  -- We put "sorry" here as a placeholder for the proof.
  sorry

end bug_returns_to_A_after_4_meters_l28_28364


namespace total_shots_cost_l28_28624

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l28_28624


namespace nadia_flower_shop_l28_28820

theorem nadia_flower_shop (roses lilies cost_per_rose cost_per_lily cost_roses cost_lilies total_cost : ℕ)
  (h1 : roses = 20)
  (h2 : lilies = 3 * roses / 4)
  (h3 : cost_per_rose = 5)
  (h4 : cost_per_lily = 2 * cost_per_rose)
  (h5 : cost_roses = roses * cost_per_rose)
  (h6 : cost_lilies = lilies * cost_per_lily)
  (h7 : total_cost = cost_roses + cost_lilies) :
  total_cost = 250 :=
by
  sorry

end nadia_flower_shop_l28_28820


namespace greatest_possible_piece_length_l28_28430

-- Conditions
def rope1_length : ℕ := 48
def rope2_length : ℕ := 72
def rope3_length : ℕ := 120
def max_piece_length : ℕ := 24

-- Statement
theorem greatest_possible_piece_length : (∃ n : ℕ, n = max_piece_length ∧
  ∀ k : ℕ, k > max_piece_length → k ∣ rope1_length ∧ k ∣ rope2_length ∧ k ∣ rope3_length →
  k > max_piece_length ∧ k ≤ max_piece_length) :=
begin
  sorry
end

end greatest_possible_piece_length_l28_28430


namespace union_of_M_and_N_l28_28807

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} :=
by
  sorry

end union_of_M_and_N_l28_28807


namespace ending_number_l28_28032

/-- The sum of the first n consecutive odd integers is n^2. 
Given that the sum of all odd integers between 13 and a certain number inclusive is 288,
prove that the ending number is 43. -/
theorem ending_number (n : ℕ) (a sum : ℤ) :
  (∀ k : ℕ, ∑ i in finset.range k, (2 * i + 1) = k ^ 2) →
  (a = 13) →
  (sum = 288) →
  (∃ end_num : ℤ, sum = ∑ i in finset.range (n + 1), (a + 2 * i) ∧ end_num = a + (n - 1) * 2 ∧ end_num = 43) :=
by
  sorry

end ending_number_l28_28032


namespace area_of_B_l28_28640

theorem area_of_B :
  let B := {z : ℂ | let x := z.re, y := z.im in 
                    (0 ≤ x ∧ x ≤ 50) ∧ 
                    (0 ≤ y ∧ y ≤ 50) ∧ 
                    (50 * x / (x^2 + y^2) ∈ Icc 0 1) ∧ 
                    (50 * y / (x^2 + y^2) ∈ Icc 0 1) }
  ∃ (area : ℝ), ∀ z ∈ B, area = 1875 - (625 * Real.pi) / 2 :=
by {
  sorry
}

end area_of_B_l28_28640


namespace vertex_of_quadratic_fn_l28_28014

def quadratic_fn (x : ℝ) : ℝ := (x - 2)^2 + 1

theorem vertex_of_quadratic_fn : ∃ v : ℝ × ℝ, v = (2, 1) ∧ ∀ x : ℝ, quadratic_fn x = (x - 2)^2 + 1 := 
begin
  use (2, 1),
  split,
  { refl },   -- v = (2, 1)
  { sorry }   -- verification of the quadratic function properties
end

end vertex_of_quadratic_fn_l28_28014


namespace solve_for_x_l28_28003

-- Define the equation as a predicate
def equation (x : ℝ) : Prop := (0.05 * x + 0.07 * (30 + x) = 15.4)

-- The proof statement
theorem solve_for_x :
  ∃ x : ℝ, equation x ∧ x = 110.8333 :=
by
  existsi (110.8333 : ℝ)
  split
  sorry -- Proof of the equation
  rfl -- Equality proof

end solve_for_x_l28_28003


namespace baker_initial_cakes_l28_28608

-- Definitions based on the given conditions
def cakes_bought : ℕ := 139
def cakes_sold : ℕ := 145
def extra_cakes_sold : ℕ := 6

-- The mathematically equivalent proof problem
theorem baker_initial_cakes :
  cakes_sold = cakes_bought + extra_cakes_sold → cakes_sold - cakes_bought = 6 :=
by
  intro h
  calc
    cakes_sold - cakes_bought = 145 - 139 : by rw [h, cakes_bought]
                          ... = 6         : by norm_num

end baker_initial_cakes_l28_28608


namespace standard_equation_of_ellipse_trajectory_equation_of_N_trajectory_equation_of_M_of_PA_l28_28236

-- Problem 1: Standard equation of the ellipse
theorem standard_equation_of_ellipse :
  (forall (x y : ℝ), (y^2 + (x^2 / 4) = 1)) ↔ 
  (∀ (x y : ℝ), (0 < x ∧ x < 2 ∧ y = 0 ∧ (sqrt 3)^2 + y^2 = a^2)) :=
sorry

-- Problem 2: Trajectory equation of point N
theorem trajectory_equation_of_N (x y : ℝ)  :
  (∀ P : ℝ × ℝ, (P ∈ set_of (λ pt, (pt.1^2 / 4) + y^2 = 1)) ∧
  ∀ M N : ℝ × ℝ, (M.1 = P.1 ∧ M.2 = 0 ∧ N = (P.1, 2 * P.2) ) -> 
  (∀ N : ℝ × ℝ, N.1^2 + N.2^2 = 4)) :=
sorry

-- Problem 3: Trajectory equation of the midpoint M of PA
theorem trajectory_equation_of_M_of_PA (x y : ℝ) :
  (∀ (x0 y0 : ℝ), ((2 * x - 1, 2 * y - 1 / 2) ∈ set_of (λ pt, pt.1^2 / 4 + pt.2^2 = 1)) ->
  ((x - 1/2)^2 + 4*(y - 1/4)^2 = 1)) :=
sorry

end standard_equation_of_ellipse_trajectory_equation_of_N_trajectory_equation_of_M_of_PA_l28_28236


namespace option_B_correct_l28_28601

-- Definitions for option B
def p := ∀ T : Type, isosceles_triangle T -> acute_triangle T
def q := ∀ T : Type, is_equilateral_triangle T -> similar_triangles T

-- Conditions
axiom h1 : ¬ p
axiom h2 : q
axiom h3 : p ∨ q
axiom h4 : ¬ (p ∧ q)

-- The goal to prove that Option B satisfies given conditions.
theorem option_B_correct : (¬ p ∧ q ∧ (p ∨ q) ∧ ¬ (p ∧ q)) :=
by
  sorry

end option_B_correct_l28_28601


namespace a_2_pow_100_l28_28796

noncomputable def a : ℕ → ℕ
| 1 := 3
| (2 * n) := n * a n
| _ := 0  -- placeholder for other cases, though it's not needed for the specific proof

theorem a_2_pow_100 : a (2^100) = 3 * 2^4950 := by
  sorry

end a_2_pow_100_l28_28796


namespace max_min_value_when_a_is_minus1_range_of_a_for_non_monotonicity_max_value_for_non_monotonic_a_l28_28266

def f (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2

theorem max_min_value_when_a_is_minus1 :
  (∀ x ∈ Set.Icc (-5) 5, f (-1) x ≤ 37) ∧ (∃ x ∈ Set.Icc (-5) 5, f (-1) x = 37) ∧
  (∀ x ∈ Set.Icc (-5) 5, f (-1) x ≥ 1) ∧ (∃ x ∈ Set.Icc (-5) 5, f (-1) x = 1) :=
sorry

theorem range_of_a_for_non_monotonicity (a : ℝ) :
  ( -5 < a ∧ a < 5) →  (∀ x₁ x₂ ∈ Set.Icc (-5) 5, x₁ < x₂ → ¬(f a x₁ < f a x₂ ∨ f a x₁ > f a x₂) ) :=
sorry

theorem max_value_for_non_monotonic_a (a : ℝ) (h : -5 < a ∧ a < 5) : 
  ∀ x ∈ Set.Icc (-5) 5, 
  (if h : -5 < a ∧ a < 0 then f a x ≤ 27 - 10 * a 
  else if h : 0 ≤ a ∧ a < 5 then f a x ≤ 27 + 10 * a 
  else True):=
sorry

end max_min_value_when_a_is_minus1_range_of_a_for_non_monotonicity_max_value_for_non_monotonic_a_l28_28266


namespace find_m_l28_28861

-- Define the function f
def f (m : ℝ) (x : ℝ) := (m^2 - m - 1) * x^(m^2 + m - 1)

-- Define the condition that f(x) needs to be decreasing on (0, +∞)
def is_decreasing_on_pos_real (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < y → f x > f y

-- Define the power function condition
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^b

-- The main theorem.
theorem find_m (m : ℝ) :
  is_power_function (f m) → is_decreasing_on_pos_real (f m) → m = -1 := by
  sorry

end find_m_l28_28861


namespace pump_out_time_l28_28582

theorem pump_out_time (length width depth : ℝ) (num_pumps pump_rate cft_to_gallons : ℝ) :
  length = 20 ∧ width = 40 ∧ depth = 2 ∧ num_pumps = 5 ∧ pump_rate = 10 ∧ cft_to_gallons = 7.5 →
  let volume := length * width * depth in
  let volume_gallons := volume * cft_to_gallons in
  let total_pump_rate := num_pumps * pump_rate in
  let time_required := volume_gallons / total_pump_rate in
  time_required = 240 := by
  intros h,
  cases h with hl hrest1,
  cases hrest1 with hw hrest2,
  cases hrest2 with hd hrest3,
  cases hrest3 with hnp hrest4,
  cases hrest4 with hpr hcg,
  simp [hl, hw, hd, hnp, hpr, hcg],
  have volume := 20 * 40 * 2,
  have volume_gallons := volume * 7.5,
  have total_pump_rate := 5 * 10,
  have time_required := volume_gallons / total_pump_rate,
  norm_num,
  sorry

end pump_out_time_l28_28582


namespace triangle_containment_l28_28232

variable {k : ℝ}
variable (A_0 B_0 C_0 : Type) 
variable [inhabited A_0] [inhabited B_0] [inhabited C_0]

def sequence_of_triangles (A_n B_n C_n: ℕ → Type) : Prop :=
  ∀ n : ℕ,
  (∃ (k : ℝ),
    (A_0 B_1 B_0 == k * C_0 B_0 B_1) ∧
    (B_0 C_1 A_0 == k * A_0 C_1 C_0) ∧
    (C_0 A_1 B_0 == k * B_0 A_1 A_0) ∧
    (A_1 B_2 C_1 == (1/(k*k)) * C_1 A_1 A_0) ∧
    (B_1 C_2 A_1 == (1/(k*k)) * A_1 B_1 B_0) ∧
    (C_1 A_2 B_1 == (1/(k*k)) * B_1 C_1 C_0))

theorem triangle_containment (A_1 B_1 C_1 A_2 B_2 C_2: Type) : 
  (∀ (n : ℕ), 
    sequence_of_triangles A_n B_n C_n → 
    (triangle ABC ⊆ triangle A_n B_n C_n))
sorry

end triangle_containment_l28_28232


namespace circle_area_difference_l28_28727

-- Definitions for conditions
def radius1 : ℝ := 30
def diameter2 : ℝ := 30
def radius2 : ℝ := diameter2 / 2

-- Definitions for areas
def area1 : ℝ := Real.pi * radius1^2
def area2 : ℝ := Real.pi * radius2^2

-- Theorem statement
theorem circle_area_difference :
  (area1 - area2) = 675 * Real.pi :=
by
  sorry

end circle_area_difference_l28_28727


namespace max_rectangle_area_l28_28130

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l28_28130


namespace curve_equation_angle_equality_l28_28228

theorem curve_equation (P : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) (F : ℝ × ℝ) (x y k : ℝ) :
  (∀ (x y : ℝ), C x y ↔ P x y) →
  (x - 1) ^ 2 + y ^ 2 = ((x + 1) ^ 2) →
  ∃ (x y : ℝ), y ^ 2 = 4 * x :=
sorry

theorem angle_equality (C : ℝ → ℝ → Prop) (M N P A B : ℝ × ℝ) (m n x1 y1 x2 y2 λ : ℝ) :
  (∀ (x y : ℝ), C x y ↔ y^2 = 4 * x) →
  (m > 0) →
  (n = -m) →
  (M = (m, 0)) →
  (N = (n, 0)) →
  (A = (x1, y1)) →
  (B = (x2, y2)) →
  (x1 = λ * y1 + m) →
  (x2 = λ * y2 + m) →
  ∠ANM = ∠BNM :=
sorry

end curve_equation_angle_equality_l28_28228


namespace square_area_l28_28206

theorem square_area (perimeter : ℝ) (h : perimeter = 40) : ∃ A : ℝ, A = 100 := by
  have h1 : ∃ s : ℝ, 4 * s = perimeter := by
    use perimeter / 4
    linarith

  cases h1 with s hs
  use s^2
  rw [hs, h]
  norm_num
  sorry

end square_area_l28_28206


namespace infinite_series_sum_correct_l28_28631

noncomputable def infinite_series_sum : ℝ :=
  ∑' n in (Finset.Icc 3 ∞), (n^4 + 4*n^3 + 10*n^2 + 10*n + 10) / (2^n * (n^4 + 9))

theorem infinite_series_sum_correct :
  infinite_series_sum = 7 / 36 :=
by
  sorry

end infinite_series_sum_correct_l28_28631


namespace speed_of_train_l28_28591

-- Conditions
def length_of_train : ℝ := 100
def time_to_cross : ℝ := 12

-- Question and answer
theorem speed_of_train : length_of_train / time_to_cross = 8.33 := 
by 
  sorry

end speed_of_train_l28_28591


namespace rob_baseball_cards_l28_28840

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end rob_baseball_cards_l28_28840


namespace shaded_cells_product_l28_28015

def product_eq (a b c : ℕ) (p : ℕ) : Prop := a * b * c = p

theorem shaded_cells_product :
  ∃ (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ : ℕ),
    product_eq a₁₁ a₁₂ a₁₃ 12 ∧
    product_eq a₂₁ a₂₂ a₂₃ 112 ∧
    product_eq a₃₁ a₃₂ a₃₃ 216 ∧
    product_eq a₁₁ a₂₁ a₃₁ 12 ∧
    product_eq a₁₂ a₂₂ a₃₂ 12 ∧
    (a₁₁ * a₂₂ * a₃₃ = 3 * 2 * 5) :=
sorry

end shaded_cells_product_l28_28015


namespace divisibility_of_polynomial_evaluations_l28_28230

theorem divisibility_of_polynomial_evaluations (a b c : ℤ) (r s : ℤ) (h : r * s = r + s ∧ r + s + r * s = -a) :
  let P := λ x : ℤ, x^3 + a * x^2 + b * x + c in
  let P1 := P 1 in
  let Pm1 := P (-1) in
  let P0 := P 0 in
  2 * Pm1 ∣ P1 + Pm1 - 2 * (1 + P0) := sorry

end divisibility_of_polynomial_evaluations_l28_28230


namespace area_BMND_l28_28376

-- Define the problem conditions
def square_side_length : ℝ := 2
def M_is_midpoint_BC (B C M : Point) : Prop := dist B M = dist C M ∧ dist B M = square_side_length / 2
def N_is_midpoint_CD (C D N : Point) : Prop := dist C N = dist D N ∧ dist C N = square_side_length / 2

-- Prove the area of the shaded region BMND is 3/2
theorem area_BMND (A B C D M N : Point) 
  (hSquare : square A B C D)
  (hM_mid : M_is_midpoint_BC B C M)
  (hN_mid : N_is_midpoint_CD C D N) :
  area (polygon B M N D) = 3 / 2 :=
sorry

end area_BMND_l28_28376


namespace two_isosceles_triangles_l28_28825

theorem two_isosceles_triangles (α : ℝ) (A B C D : Type) [linear_ordered_field α] 
[h1 : triangle_ABC_angle_BAC_eq_alpha A B C α]
[h2 : triangle_ABC_angle_BCA_eq_3alpha A B C 3α]
(h3 : construct_point_D_on_AB_eq_AD_CD A B C D):
    is_isosceles_triangle A C D ∧ is_isosceles_triangle B C D := sorry

end two_isosceles_triangles_l28_28825


namespace can_transfer_increase_average_l28_28356

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l28_28356


namespace fraction_of_color_films_is_20_over_21_l28_28111

variables (x y : ℕ) (h_positive_x : 0 < x) (h_positive_y : 0 < y)

def selected_black_and_white := (y / x : ℚ) * 20 * x / 100
def selected_color := 4 * y
def total_selected := selected_black_and_white + selected_color

theorem fraction_of_color_films_is_20_over_21
  (h_selected_black_and_white : selected_black_and_white = y / 5)
  (h_total_selected : total_selected = y / 5 + 4 * y) :
  (selected_color / total_selected = 20 / 21) :=
by sorry

end fraction_of_color_films_is_20_over_21_l28_28111


namespace man_walking_time_l28_28928

noncomputable def usual_drive_time : ℝ := sorry
noncomputable def early_arrival_time : ℝ := 60 -- man arrives an hour early
noncomputable def time_saved : ℝ := 12 -- arrived 12 minutes earlier than usual

theorem man_walking_time :
  ∃ T : ℝ, (T + time_saved = early_arrival_time - usual_drive_time) ∧ (T = 12) :=
begin
  use 12,
  split,
  { sorry, },
  { exact rfl, },
end

end man_walking_time_l28_28928


namespace purchase_price_is_correct_l28_28443

noncomputable def purchase_price : ℝ :=
let down_payment := 27 in
let monthly_payment := 10 in
let number_of_payments := 12 in
let interest_rate := 0.2126 in
let total_paid := down_payment + monthly_payment * number_of_payments in
let interest := interest_rate * purchase_price in
(total_paid = purchase_price + interest) → purchase_price = 147 / 1.2126

theorem purchase_price_is_correct :
    let down_payment := 27 in
    let monthly_payment := 10 in
    let number_of_payments := 12 in
    let interest_rate := 0.2126 in
    let total_paid := 27 + 10 * 12 in
    let interest := interest_rate * purchase_price in
    total_paid = purchase_price + interest → purchase_price = 147 / 1.2126 :=
sorry

end purchase_price_is_correct_l28_28443


namespace intersecting_lines_sum_l28_28046

-- Given conditions
def line1 (m : ℝ) : ℝ → ℝ := λ x, m * x + 6
def line2 (b : ℝ) : ℝ → ℝ := λ x, 4 * x + b
def point_of_intersection : ℝ × ℝ := (8, 14)

-- Proof problem statement
theorem intersecting_lines_sum (m b : ℝ) (h1 : line1 m 8 = 14) (h2 : line2 b 8 = 14) : b + m = -17 :=
by
  sorry

end intersecting_lines_sum_l28_28046


namespace radius_of_inscribed_circle_l28_28059

-- Define the triangle side lengths
def DE : ℝ := 8
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- The statement to be proved
theorem radius_of_inscribed_circle : r = (5 * Real.sqrt 39) / 13 := by
  sorry

end radius_of_inscribed_circle_l28_28059


namespace weight_of_dry_grapes_l28_28929

def fresh_grapes : ℝ := 10 -- weight of fresh grapes in kg
def fresh_water_content : ℝ := 0.90 -- fresh grapes contain 90% water by weight
def dried_water_content : ℝ := 0.20 -- dried grapes contain 20% water by weight

theorem weight_of_dry_grapes : 
  (fresh_grapes * (1 - fresh_water_content)) / (1 - dried_water_content) = 1.25 := 
by 
  sorry

end weight_of_dry_grapes_l28_28929


namespace area_of_region_B_l28_28635

theorem area_of_region_B : 
  let B := { z : ℂ | (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1) ∧ 
                   (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
                    0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1) } in
  ∫ (z in B), 1 = 2500 - 312.5 * Real.pi := 
by
  sorry

end area_of_region_B_l28_28635


namespace original_square_area_is_144_square_centimeters_l28_28967

noncomputable def area_of_original_square (x : ℝ) : ℝ :=
  x^2 - (x - 3) * (x - 5)

theorem original_square_area_is_144_square_centimeters (x : ℝ) (h : area_of_original_square x = 81) :
  (x = 12) → (x^2 = 144) :=
by
  sorry

end original_square_area_is_144_square_centimeters_l28_28967


namespace no_solutions_2_pow_2x_minus_3_pow_2y_eq_91_l28_28181

theorem no_solutions_2_pow_2x_minus_3_pow_2y_eq_91 :
  ∀ x y : ℤ, 2^(2*x) - 3^(2*y) ≠ 91 :=
by
  sorry

end no_solutions_2_pow_2x_minus_3_pow_2y_eq_91_l28_28181


namespace towel_area_decrease_l28_28970

theorem towel_area_decrease (L B : ℝ) (hL : 0 < L) (hB : 0 < B) :
  let A := L * B
  let L' := 0.80 * L
  let B' := 0.90 * B
  let A' := L' * B'
  A' = 0.72 * A →
  ((A - A') / A) * 100 = 28 :=
by
  intros _ _ _ _
  sorry

end towel_area_decrease_l28_28970


namespace not_sophomores_75_percent_l28_28366

noncomputable def total_students : ℕ := 800
noncomputable def juniors_percentage : ℚ := 23 / 100
noncomputable def juniors : ℕ := (juniors_percentage * total_students).toNat
noncomputable def seniors : ℕ := 160
noncomputable def freshmen_offset : ℕ := 56

def sophomores (total students juniors seniors freshmen_offset : ℕ) : ℕ :=
  (total - juniors - seniors + freshmen_offset) / 2

def freshmen (sophomores freshmen_offset : ℕ) : ℕ := sophomores + freshmen_offset

def not_sophomores_percentage (total sophomores : ℕ) : ℚ :=
  ((total - sophomores):ℚ / total) * 100

theorem not_sophomores_75_percent :
  let S := sophomores total_students juniors seniors freshmen_offset in
  not_sophomores_percentage total_students S = 75 := by
  sorry

end not_sophomores_75_percent_l28_28366


namespace convert_speed_l28_28139

theorem convert_speed (v_mps : ℝ) (h : v_mps = 20.0016) : v_mps * 3.6 = 72.00576 :=
by {
  rw h,
  norm_num,
  sorry
}

end convert_speed_l28_28139


namespace max_intersection_points_three_circles_two_lines_l28_28899

theorem max_intersection_points_three_circles_two_lines : 
  ∀ (C1 C2 C3 L1 L2 : set ℝ × ℝ) (hC1 : is_circle C1) (hC2 : is_circle C2) (hC3 : is_circle C3) (hL1 : is_line L1) (hL2 : is_line L2),
  ∃ P : ℕ, P = 19 ∧
  (∀ (P : ℝ × ℝ), P ∈ C1 ∧ P ∈ C2 ∨ P ∈ C1 ∧ P ∈ C3 ∨ P ∈ C2 ∧ P ∈ C3 ∨ P ∈ C1 ∧ P ∈ L1 ∨ P ∈ C2 ∧ P ∈ L1 ∨ P ∈ C3 ∧ P ∈ L1 ∨ P ∈ C1 ∧ P ∈ L2 ∨ P ∈ C2 ∧ P ∈ L2 ∨ P ∈ C3 ∧ P ∈ L2 ∨ P ∈ L1 ∧ P ∈ L2) ↔ P = 19 :=
sorry

end max_intersection_points_three_circles_two_lines_l28_28899


namespace g_increasing_g_multiplicative_g_special_case_g_18_value_l28_28412

def g (n : ℕ) : ℕ :=
sorry

theorem g_increasing : ∀ n : ℕ, n > 0 → g (n + 1) > g n :=
sorry

theorem g_multiplicative : ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n :=
sorry

theorem g_special_case : ∀ m n : ℕ, m > 0 → n > 0 → m ≠ n → m ^ n = n ^ m → g m = n ∨ g n = m :=
sorry

theorem g_18_value : g 18 = 324 :=
sorry

end g_increasing_g_multiplicative_g_special_case_g_18_value_l28_28412


namespace vector_combination_l28_28767

noncomputable def m : ℝ := (0.5 * Real.sqrt 3 * Real.sqrt 5 + 9) / (2 * Real.sqrt 3 * Real.sqrt 5 + 12)
noncomputable def n : ℝ := (-9 * (3 / 5) + 13.5 * Real.sqrt 3) / 35

def vec_length {α : Type*} [inner_product_space ℝ α] (v : α) : ℝ :=
  real.sqrt (inner_product_space.inner v v)

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_combination (OA OB OC : V)
  (hOA_len : vec_length OA = 2)
  (hOB_len : vec_length OB = 3)
  (hOC_len : vec_length OC = 3)
  (hTanAOC : real.tan (real.angle_of_vectors OA OC) = 4)
  (hAngleBOC : real.angle_of_vectors OB OC = real.pi / 3) :
  OC = m • OA + n • OB :=
sorry

end vector_combination_l28_28767


namespace composite_product_division_l28_28189

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end composite_product_division_l28_28189


namespace find_interest_rate_l28_28050

noncomputable def interest_rate (total_investment remaining_investment interest_earned part_interest : ℝ) : ℝ :=
  (interest_earned - part_interest) / remaining_investment

theorem find_interest_rate :
  let total_investment := 9000
  let invested_at_8_percent := 4000
  let total_interest := 770
  let interest_at_8_percent := invested_at_8_percent * 0.08
  let remaining_investment := total_investment - invested_at_8_percent
  let interest_from_remaining := total_interest - interest_at_8_percent
  interest_rate total_investment remaining_investment total_interest interest_at_8_percent = 0.09 :=
by
  sorry

end find_interest_rate_l28_28050


namespace proof_statement_l28_28313

variable {Point : Type}
variable [MetricSpace Point]
variable {Line Plane : Type}
variable [Subspace Point Line]
variable [Subspace Point Plane]
variable [LinearMap Line Plane]

@[hott]
def perpendicular {X : Type} [MetricSpace X] (a b : X) : Prop := 
  ∃ c : X, c ≠ a ∧ c ≠ b ∧ dist a c = dist b c

def problem_statement 
  (a b : Line) 
  (α : Plane)
  (h1 : perpendicular a b) 
  (h2 : perpendicular b α) : Prop :=
  (a = α) ∨ (a ∥ α)

theorem proof_statement 
  (a b : Line) 
  (α : Plane)
  (h1 : perpendicular a b) 
  (h2 : perpendicular b α) : problem_statement a b α h1 h2 :=
sorry

end proof_statement_l28_28313


namespace percentage_k_equal_125_percent_j_l28_28738

theorem percentage_k_equal_125_percent_j
  (j k l m : ℝ)
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := 
sorry

end percentage_k_equal_125_percent_j_l28_28738


namespace optimal_radius_l28_28428

noncomputable def minimize_material (r : ℝ) : Prop :=
  let h := 27 / (r^2)
  let surface_area := π * (r^2) + (2 * π * r * h)
  ∀ r > 0, surface_area ≥ (π * (3^2) + 2 * π * 3 * (27 / (3^2)))

theorem optimal_radius : minimize_material 3 :=
sorry

end optimal_radius_l28_28428


namespace sum_of_80th_equation_l28_28367

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end sum_of_80th_equation_l28_28367


namespace circle_equation_proof_l28_28249

noncomputable def circle_equation : Prop := 
  ∃ (t : ℝ), 
  (3 * t, t) ∈ { p : ℝ × ℝ | 3 * p.1 - p.2 = 0 } ∧ 
  ∀ (r : ℝ), r = 3 * t →
  ∀ (d : ℝ), d = abs (t - 3 * t) / sqrt 2 →
  r^2 = d^2 + (sqrt 7)^2 →
  ( ∀ (x y : ℝ), (x + 1)^2 + (y + 3)^2 = 9 ∨
    (x - 1)^2 + (y - 3)^2 = 9 )

theorem circle_equation_proof : circle_equation := sorry

end circle_equation_proof_l28_28249


namespace count_negative_numbers_l28_28975

theorem count_negative_numbers : 
  (List.filter (λ x => x < (0:ℚ)) [-14, 7, 0, -2/3, -5/16]).length = 3 := 
by
  sorry

end count_negative_numbers_l28_28975


namespace max_value_on_interval_l28_28493

noncomputable def function_y (x : ℝ) : ℝ :=
  x^4 - 8 * x^2 + 2

theorem max_value_on_interval :
  ∃ x ∈ set.Icc (-1 : ℝ) (3 : ℝ), function_y x = 11 :=
by
  sorry

end max_value_on_interval_l28_28493


namespace find_angle_B_g_increasing_intervals_l28_28747

variable {a b c A B C : ℝ}
variable (f g : ℝ → ℝ)

-- Conditions
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) -- triangle sides are positive
variable (h4 : A + B + C = π) -- angles in a triangle sum to π
variable (h5 : (2 * a - c) * Real.cos B - b * Real.cos C = 0) -- given equation

-- Definitions for Part (2)
noncomputable def f (x : ℝ) : ℝ := -2 * Real.cos (2 * x + B)
noncomputable def g (x : ℝ) : ℝ := f (x - π / 12)

-- Part (1) proof
theorem find_angle_B : B = π / 3 :=
by 
  sorry

-- Part (2) proof
theorem g_increasing_intervals : 
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 4) (k * π + π / 4), 
    g.diff x > 0 :=
by
  sorry

end find_angle_B_g_increasing_intervals_l28_28747


namespace sequence_not_generated_l28_28316

theorem sequence_not_generated (a : ℕ → ℝ) :
  (a 1 = 2) ∧ (a 2 = 0) ∧ (a 3 = 2) ∧ (a 4 = 0) → 
  (∀ n, a n ≠ (1 - Real.cos (n * Real.pi)) + (n - 1) * (n - 2)) :=
by sorry

end sequence_not_generated_l28_28316


namespace first_number_in_100th_group_l28_28875

theorem first_number_in_100th_group :
  let seq := (λ n : ℕ, 3^(n-1))
  ∃ a : ℕ, a = 100 → seq (1 + 2 + 3 + ... + (a - 1) + 1) = 3^4950 := by
  sorry

end first_number_in_100th_group_l28_28875


namespace irrational_number_in_list_l28_28600

theorem irrational_number_in_list :
  let l := [Real.pi / 2, 22 / 7, Real.sqrt 4, 0.101001000]
  ∃ x ∈ l, irrational x ∧ ∀ y ∈ l, y ≠ x → ¬ irrational y :=
by
  sorry

end irrational_number_in_list_l28_28600


namespace g_is_even_l28_28392

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.cos x + Real.sqrt (1 + Real.sin x ^ 2))

theorem g_is_even : ∀ x : ℝ, g (-x) = g (x) :=
by
  intro x
  sorry

end g_is_even_l28_28392


namespace no_distinct_sums_2020x2020_l28_28393

theorem no_distinct_sums_2020x2020 :
  ¬ ∃ (f : ℕ × ℕ → ℤ),
    (∀ i, i < 2020 → ∀ j, j < 2020 → f (i, j) ∈ {-1, 0, 1}) ∧
    (∀ i j, i < 2020 ∧ j < 2020 ∧ i ≠ j → 
     (∑ k in Finset.range 2020, f (i, k)) ≠ 
     (∑ k in Finset.range 2020, f (j, k)) ∧
     (∑ k in Finset.range 2020, f (k, i)) ≠ 
     (∑ k in Finset.range 2020, f (k, j)) ∧ 
     (∑ k in Finset.range 2020, f (k, k)) ≠ 
     (∑ k in Finset.range 2020, f (k, 2020 - k - 1))) 
  := sorry

end no_distinct_sums_2020x2020_l28_28393


namespace bryan_total_books_magazines_l28_28160

-- Conditions as definitions
def novels : ℕ := 90
def comics : ℕ := 160
def rooms : ℕ := 12
def x := (3 / 4 : ℚ) * novels
def y := (6 / 5 : ℚ) * comics
def z := (1 / 2 : ℚ) * rooms

-- Calculations based on conditions
def books_per_shelf := 27 * x
def magazines_per_shelf := 80 * y
def total_shelves := 23 * z
def total_books := books_per_shelf * total_shelves
def total_magazines := magazines_per_shelf * total_shelves
def grand_total := total_books + total_magazines

-- Theorem to prove
theorem bryan_total_books_magazines :
  grand_total = 2371275 := by
  sorry

end bryan_total_books_magazines_l28_28160


namespace mixed_number_difference_l28_28815

theorem mixed_number_difference :
  ∃ a b c d e f : ℕ,
  {a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5} ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ d ≠ e ∧ e ≠ f ∧ f ≠ d ∧
  (a + (b / c) : ℚ) = 5 + 3/4 ∧
  (d + (e / f) : ℚ) = 1 + 2/5 ∧
  (5 + 3/4 - (1 + 2/5) : ℚ) = 4 + 7/20 :=
sorry

end mixed_number_difference_l28_28815


namespace fraction_zero_iff_x_one_l28_28321

theorem fraction_zero_iff_x_one (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x - 5 ≠ 0) : x = 1 := by
  sorry

end fraction_zero_iff_x_one_l28_28321


namespace cartesian_eq_of_conic_focus_coords_maximum_distance_point_l28_28499

-- Definitions based on conditions
def polar_eq_conic (ρ θ : ℝ) : Prop := ρ^2 * (1 + sin(θ)^2) = 2
def cart_eq_conic (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def polar_eq_line (θ : ℝ) : Prop := θ = π / 3

-- Definitions based on solutions
def focus_cartesian_1 : (ℝ × ℝ) := (-1, 0)
def focus_cartesian_2 : (ℝ × ℝ) := (1, 0)

def focus_polar_1 : (ℝ × ℝ) := (1, π)
def focus_polar_2 : (ℝ × ℝ) := (1, 0)

-- Definition of maximum distance point
def max_dist_point_1 : (ℝ × ℝ) := (2 * sqrt(21) / 7, -sqrt(7) / 7)
def max_dist_point_2 : (ℝ × ℝ) := (-2 * sqrt(21) / 7, sqrt(7) / 7)

-- Theorems to be proven in Lean 4
theorem cartesian_eq_of_conic (ρ θ : ℝ) (x y : ℝ):
  polar_eq_conic ρ θ →
  cart_eq_conic x y :=
sorry

theorem focus_coords (polar focus_cart focus_polar : (ℝ × ℝ)):
  focus_cart = focus_cartesian_1 ∨ focus_cart = focus_cartesian_2 ∧
  focus_polar = focus_polar_1 ∨ focus_polar = focus_polar_2 :=
sorry

theorem maximum_distance_point (x y : ℝ):
  (x, y) = max_dist_point_1 ∨ (x, y) = max_dist_point_2 :=
sorry

end cartesian_eq_of_conic_focus_coords_maximum_distance_point_l28_28499


namespace tangent_lines_to_reflected_circles_l28_28088

noncomputable def tangent_problem (A B : Circle) (m : Line) (on_same_side : same_side A B m) : Nat :=
4

theorem tangent_lines_to_reflected_circles (A B : Circle) (m : Line) 
  (side_cond : same_side A B m) : 
  ∃ T : List Line, 
  (∀ t ∈ T, tangent_to A t ∧ tangent_to B (reflect_over t m)) ∧
  T.length = tangent_problem A B m side_cond :=
sorry

end tangent_lines_to_reflected_circles_l28_28088


namespace eccentricity_of_hyperbola_l28_28683

theorem eccentricity_of_hyperbola
  (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c = a * e)
  (h4 : c^2 = a^2 + b^2)
  (h5 : ∀ B : ℝ × ℝ, B = (0, b))
  (h6 : ∀ F : ℝ × ℝ, F = (c, 0))
  (h7 : ∀ m_FB m_asymptote : ℝ, m_FB * m_asymptote = -1 → (m_FB = -b / c) ∧ (m_asymptote = b / a)) :
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l28_28683


namespace problem_statement_l28_28788

def f (a b : ℝ) :=
  if a + b ≤ 3 then (a * b - a + 2) / (2 * a)
  else (a * b - b - 2) / (-2 * b)

theorem problem_statement : f 2 1 + f 2 4 = 1 / 4 := by
  sorry

end problem_statement_l28_28788


namespace angle_relation_l28_28386

open_locale real

variables {A B C L P : Type} [real B C L P] 

-- Condition 1: LA is the angle bisector of ∠BLP
def angle_bisector (a b c : Type) [real a b c] := sorry -- (Placeholder definition for the angle bisector concept)

-- Given conditions:
axiom condition1 : angle_bisector L B P
axiom condition2 : dist B L = dist C P

-- Prove ∠ABC = 2 * ∠BCA
theorem angle_relation (A B C L P : Type) [real B C L P] 
  (h1 : angle_bisector L B P) 
  (h2 : dist B L = dist C P) : angle B A C = 2 * angle C A B :=
sorry

end angle_relation_l28_28386


namespace P_plus_Q_calc_l28_28795
noncomputable def P : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
noncomputable def Q : Set ℝ := {x | ∃ y, y = Real.log2 (x^2 - 2*x - 15)}
noncomputable def P_plus_Q : Set ℝ := {x | x ∈ P ∨ (x ∈ Q ∧ x ∉ {y | y ∈ P ∧ y ∈ Q})}
theorem P_plus_Q_calc : P_plus_Q = {x : ℝ | x ≤ -3 ∨ (x ≥ -1 ∧ x ≤ 4) ∨ x > 5} := by
  sorry

end P_plus_Q_calc_l28_28795


namespace painted_cubes_visible_from_corner_l28_28094

def cubes_per_face (n : ℕ) := n * n
def faces_visible_from_corner := 3

theorem painted_cubes_visible_from_corner : cubes_per_face 12 * faces_visible_from_corner - faces_visible_from_corner * 11 + 1 = 400 :=
by
  have face_cubes := cubes_per_face 12,
  calc
    face_cubes * faces_visible_from_corner - faces_visible_from_corner * 11 + 1
        = 144 * 3 - 33 + 1 : by rw [face_cubes]
    ... = 400 : by norm_num

end painted_cubes_visible_from_corner_l28_28094


namespace tangent_hyperbola_dot_product_l28_28248

theorem tangent_hyperbola_dot_product :
  ∀ (P M N : Point) (l : Line),
  (tangent l (hyperbola { x // x^2 / 4 - y^2 = 1 }) P) ∧
  (intersects l (asymptote (hyperbola { x // x^2 / 4 - y^2 = 1 })) M) ∧
  (intersects l (asymptote (hyperbola { x // x^2 / 4 - y^2 = 1 })) N) →
  (dot OM ON = 3) := 
sorry -- Proof is not provided as it is not required.

end tangent_hyperbola_dot_product_l28_28248


namespace planning_committee_selection_l28_28968

theorem planning_committee_selection (x : ℕ) (h1 : (x.choose 3) = 35) : (x.choose 4) = 35 :=
by
  have h_x_eq_7 : x = 7 :=
    by sorry
  rw h_x_eq_7
  norm_num
  sorry

end planning_committee_selection_l28_28968


namespace compare_exponential_compare_logarithms_l28_28999

-- Given the increasing nature of the exponential function base 2 and the inputs
theorem compare_exponential (h : 0.6 > 0.5) : (2:ℝ)^(0.6) > (2:ℝ)^(0.5) :=
sorry

-- Given the increasing nature of logarithm function base 2 and the inputs
theorem compare_logarithms (h : 3.4 < 3.8) : Real.log(3.4) / Real.log(2) < Real.log(3.8) / Real.log(2) :=
sorry

end compare_exponential_compare_logarithms_l28_28999


namespace value_of_f_at_9_l28_28711

noncomputable def f (x : ℝ) : ℝ :=
  x ^ (-1 / 2)

theorem value_of_f_at_9 : f 9 = 1 / 3 :=
by
  sorry

end value_of_f_at_9_l28_28711


namespace grazing_months_correct_l28_28944

theorem grazing_months_correct :
  ∃ x : ℕ, sorry := -- specification that we need to find such an x

begin
  let C := 1440 / (24 * x), -- Define the cost per cow per month
  have eq1: 24 * x * C = 1440, by sorry, -- Given A's share of 1440 Rs.
  have eq2: (24 * x * C) + (10 * 5 * C) + (35 * 4 * C) + (21 * 3 * C) = 6500, by sorry, -- Total rent condition
  let eq1_simplified := by calc 
    24 * x * (1440 / (24 * x)) = 1440 : by sorry,
  let eq2_simplified := by calc 
    1440 + (1440 / (24 * x)) * 253 = 6500 : by sorry,

  -- Prove the final value of x
  have final_eq: x = 3, by sorry, 
  exact final_eq
end

end grazing_months_correct_l28_28944


namespace min_value_a2_b2_l28_28219

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end min_value_a2_b2_l28_28219


namespace p_or_q_represents_l28_28986

def p : Prop :=
  "Athlete A's trial jump exceeded 2 meters"

def q : Prop :=
  "Athlete B's trial jump exceeded 2 meters"

theorem p_or_q_represents :
  p ∨ q ↔ "At least one of Athlete A or B exceeded 2 meters in their trial jump" :=
begin
  sorry
end

end p_or_q_represents_l28_28986


namespace prob_first_question_correct_is_4_5_distribution_of_X_l28_28561

-- Assume probabilities for member A and member B answering correctly.
def prob_A_correct : ℚ := 2 / 5
def prob_B_correct : ℚ := 2 / 3

def prob_A_incorrect : ℚ := 1 - prob_A_correct
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- Given that A answers first, followed by B.
-- Calculate the probability that the first team answers the first question correctly.
def prob_first_question_correct : ℚ :=
  prob_A_correct + (prob_A_incorrect * prob_B_correct)

-- Assert that the calculated probability is equal to 4/5
theorem prob_first_question_correct_is_4_5 :
  prob_first_question_correct = 4 / 5 := by
  sorry

-- Define the possible scores and their probabilities
def prob_X_eq_0 : ℚ := prob_A_incorrect * prob_B_incorrect
def prob_X_eq_10 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_20 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 2 * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_30 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 3

-- Assert the distribution probabilities for the random variable X
theorem distribution_of_X :
  prob_X_eq_0 = 1 / 5 ∧
  prob_X_eq_10 = 4 / 25 ∧
  prob_X_eq_20 = 16 / 125 ∧
  prob_X_eq_30 = 64 / 125 := by
  sorry

end prob_first_question_correct_is_4_5_distribution_of_X_l28_28561


namespace composite_quotient_is_one_over_49_l28_28188

/-- Define the first twelve positive composite integers --/
def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def next_six_composites : List ℕ := [14, 15, 16, 18, 20, 21]

/-- Define the product of a list of integers --/
def product (l : List ℕ) : ℕ := l.foldl (λ acc x => acc * x) 1

/-- This defines the quotient of the products of the first six and the next six composite numbers --/
def composite_quotient : ℚ := (↑(product first_six_composites)) / (↑(product next_six_composites))

/-- Main theorem stating that the composite quotient equals to 1/49 --/
theorem composite_quotient_is_one_over_49 : composite_quotient = 1 / 49 := 
  by
    sorry

end composite_quotient_is_one_over_49_l28_28188


namespace number_of_cows_l28_28389

-- Definitions
variables (a g e c : ℕ)
variables (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g)

-- Theorem statement
theorem number_of_cows (a g e : ℕ) (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g) :
  ∃ c : ℕ, c * e * 6 = 6 * a + 36 * g ∧ c = 5 :=
by
  sorry

end number_of_cows_l28_28389


namespace num_small_triangles_needed_l28_28960

theorem num_small_triangles_needed
  (A B C : Type)
  [IsEquilateralTriangle A]
  [HasSideLength A (20 : ℝ)]
  [IsEquilateralTriangle B]
  [HasSideLength B (2 : ℝ)]
  [NonoverlappingCovering A B]
  [RotatedRowsCovering A B 180] :
  num_triangles_needed A B = 100 :=
sorry

end num_small_triangles_needed_l28_28960


namespace geometric_sequence_q_sum_of_bn_l28_28229
noncomputable theory

def a (n : ℕ) : ℕ := -- defining the geometric sequence
  if n = 1 then 2 else 2 * a (n - 1)

def S (n : ℕ) : ℕ := -- sum of the first n terms of {a_n}
  ∑ i in range n, a (i + 1)

def b (n : ℕ) : ℕ := -- defining the sequence {b_n}
  n + a n

def T (n : ℕ) : ℕ := -- sum of the first n terms of {b_n}
  ∑ i in range n, b (i + 1)

theorem geometric_sequence_q (q : ℕ) (n : ℕ) -- Proving common ratio of the geometric sequence
  (h : (4 * S 1, 3 * S 2, 2 * S 3).is_arithmetic_sequence) : q = 2 :=
sorry

theorem sum_of_bn (n : ℕ) : -- Proving the sum of first n terms of {b_n}
  T n = n * (n + 1) / 2 + 2 ^ (n + 1) - 2 :=
sorry

end geometric_sequence_q_sum_of_bn_l28_28229


namespace inscribed_circle_radius_of_DEF_l28_28062

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l28_28062


namespace exists_permutation_operations_l28_28262

theorem exists_permutation_operations (a b c d : ℤ) (ops : List (ℤ → ℤ → ℤ))
  (h : ops = [Int.add, Int.sub, Int.mul]) :
  ∃ (ops' : List (ℤ → ℤ → ℤ)), Permutation.ops ops' (ops) ∧
  (ops'.head a (ops'.tail.head b (ops'.tail.tail.head c d)) = 17) :=
by
  use [Int.add, Int.mul, Int.sub]
  split
  · apply Perm.perm.refl
  · sorry

end exists_permutation_operations_l28_28262


namespace donuts_left_for_sophie_l28_28478

def initial_boxes := 4
def donuts_per_box := 12
def boxes_given_to_mom := 1
def donuts_given_to_sister := 6

theorem donuts_left_for_sophie :
  let initial_donuts := initial_boxes * donuts_per_box in
  let remaining_boxes := initial_boxes - boxes_given_to_mom in
  let remaining_donuts := remaining_boxes * donuts_per_box in
  let donuts_left := remaining_donuts - donuts_given_to_sister in
  donuts_left = 30 := 
by 
  have initial_donuts := initial_boxes * donuts_per_box
  have remaining_boxes := initial_boxes - boxes_given_to_mom
  have remaining_donuts := remaining_boxes * donuts_per_box
  have donuts_left := remaining_donuts - donuts_given_to_sister
  show donuts_left = 30
  calc
  donuts_per_box * (initial_boxes - boxes_given_to_mom) - donuts_given_to_sister = donuts_per_box * 3 - donuts_given_to_sister := by sorry
  donuts_per_box * 3 - donuts_given_to_sister = 30 := by sorry

end donuts_left_for_sophie_l28_28478


namespace axis_of_symmetry_l28_28732

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f(x) = f(3 - x)) : ∀ x, f(x) = f(1.5 + (x - 1.5)) :=
by
  sorry

end axis_of_symmetry_l28_28732


namespace sum_of_squares_15_l28_28512

theorem sum_of_squares_15 :
  let sum_sq_first_n (n : ℕ) := (n * (n + 1) * (2 * n + 1)) / 6 in
  sum_sq_first_n 30 - sum_sq_first_n 15 = 8175 →
  sum_sq_first_n 15 = 1200 :=
by
  intros sum_sq_first_n h
  dsimp [sum_sq_first_n] at h 
  sorry

end sum_of_squares_15_l28_28512


namespace min_sqrt_distance_to_line_l28_28659

theorem min_sqrt_distance_to_line : 
  (∃ x y : ℝ, 10 * x + 24 * y = 120 ∧ sqrt (x^2 + y^2) = 60 / 13) :=
sorry

end min_sqrt_distance_to_line_l28_28659


namespace passengersInScientificNotation_l28_28439

-- defining the concept of 'million'
def million := 10^6

-- condition given in the problem
def totalPassengers : Real := 1.446 * million

-- proving the equivalent scientific notation
theorem passengersInScientificNotation : totalPassengers = 1.446 * 10^6 := by
  have h : 1.446 * million = 1.446 * 10^6 := by
    rw [million]
    sorry
  exact h

end passengersInScientificNotation_l28_28439


namespace explicit_formula_l28_28685

variable (f : ℝ → ℝ)
variable (is_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
variable (max_value : ∀ x, f x ≤ 13)
variable (value_at_3 : f 3 = 5)
variable (value_at_neg1 : f (-1) = 5)

theorem explicit_formula :
  (∀ x, f x = -2 * x^2 + 4 * x + 11) :=
by
  sorry

end explicit_formula_l28_28685


namespace distinct_lines_among_points_l28_28833

theorem distinct_lines_among_points (n : ℕ) (h : n ≥ 3) (points : set (Real × Real)) (hnl : ¬linear points) :
  ∃ lines : set (set (Real × Real)), lines.card ≥ n ∧ 
  ∀ p1 p2 ∈ points, p1 ≠ p2 → ∃ l ∈ lines, p1 ∈ l ∧ p2 ∈ l := 
sorry

end distinct_lines_among_points_l28_28833


namespace sum_of_inverse_geometric_sequence_l28_28860

theorem sum_of_inverse_geometric_sequence (a_n : ℕ → ℝ) (S_n q : ℝ) (n : ℕ) 
  (h1 : a_n 1 = 1) 
  (h2 : ∀ i, a_n (i + 1) / a_n i = q)
  (h3 : S_n = (finset.range n).sum (λ k, a_n (k + 1))) :
  (finset.range n).sum (λ k, 1 / a_n (k + 1)) = S_n * q^(1 - n) :=
sorry

end sum_of_inverse_geometric_sequence_l28_28860


namespace sum_of_roots_of_f_l28_28238

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - f x

noncomputable def f_increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y

theorem sum_of_roots_of_f (f : ℝ → ℝ) (m : ℝ) (x1 x2 x3 x4 : ℝ)
  (h1 : odd_function f)
  (h2 : ∀ x, f (x - 4) = - f x)
  (h3 : f_increasing_on f 0 2)
  (h4 : m > 0)
  (h5 : f x1 = m)
  (h6 : f x2 = m)
  (h7 : f x3 = m)
  (h8 : f x4 = m)
  (h9 : x1 ≠ x2)
  (h10 : x1 ≠ x3)
  (h11 : x1 ≠ x4)
  (h12 : x2 ≠ x3)
  (h13 : x2 ≠ x4)
  (h14 : x3 ≠ x4)
  (h15 : ∀ x, -8 ≤ x ∧ x ≤ 8 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_of_f_l28_28238


namespace radius_of_inscribed_circle_l28_28058

-- Define the triangle side lengths
def DE : ℝ := 8
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- The statement to be proved
theorem radius_of_inscribed_circle : r = (5 * Real.sqrt 39) / 13 := by
  sorry

end radius_of_inscribed_circle_l28_28058


namespace find_remainder_l28_28409

-- Definition of the sequence T as numbers with exactly 6 ones in binary representation.
def is_valid_number (n : ℕ) : Prop := 
  (nat.bit_count n = 6)

def T : ℕ → ℕ
| 0 := 0
| (n+1) := nat.find (λ x, is_valid_number (T n + x))

-- Define M as the 1500th number in T
def M : ℕ := T 1500

-- The theorem statement
theorem find_remainder : M % 500 = 16 := 
sorry

end find_remainder_l28_28409


namespace gumdrops_blue_count_l28_28959

theorem gumdrops_blue_count :
  (total_gumdrops blue brown red yellow green : ℕ)         -- Declare the numbers as natural numbers
  (initial_total_gumdrops : total_gumdrops = 150)         -- Total gumdrops is 150
  (blue_gumdrops_initial : blue = 37.5 ∨ blue = 38)     -- Initially 25% gumdrops are blue; rounded due to integer concerns
  (brown_gumdrops_initial : brown = 37.5 ∨ brown = 38)   -- Initially 25% gumdrops are brown; rounded due to integer concerns
  (red_gumdrops_initial : red = 30)                     -- Initially 20% gumdrops are red
  (yellow_gumdrops_initial : yellow = 15)               -- Initially 10% gumdrops are yellow
  (green_gumdrops_initial : green = total_gumdrops - (blue + brown + red + yellow))
  (replaced_red_to_blue : blue_final = blue + 3 * red / 4)
  : blue_final = 60 :=
by
  sorry

end gumdrops_blue_count_l28_28959


namespace sphere_shadow_boundary_l28_28140

theorem sphere_shadow_boundary (x : ℝ) :
  let O := (0, 0, 2)
  let P := (0, -4, 3)

  -- Conditions
  sphere_center_x_zero : (0 = 0) (sphere_X_zero : (0 = 0 (sphere_radius := 2)
  light_source_X_zero : (0 = 0) (light_source_Y := (0 = -4) light_source_Z := (0 = 3)
  -- Answer
  boundary_condition : (y = -19/4) :
  y = -19/4 := 
  sorry
  -- Proof is omitted

end sphere_shadow_boundary_l28_28140


namespace max_area_rectangle_l28_28134

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l28_28134


namespace find_m_l28_28252

def ellipse (m : ℝ) : Prop := ∃ e, (e = 1/2) ∧ (3 > m ∧ e = (Real.sqrt (3 - m) / Real.sqrt 3) ∨ 3 < m ∧ e = (Real.sqrt (m - 3) / Real.sqrt m))

theorem find_m : ∀ m, ellipse m → (m = 4 ∨ m = 9/4) :=
by {
  intros m h,
  sorry -- The proof would go here, but is not required for the statement generation.
}

end find_m_l28_28252


namespace af_de_equals_r_squared_l28_28441

-- Define the geometric configuration
variables {R : ℝ} (O A D B C F E : Point)

-- Conditions
def is_circle (O A D : Point) (R : ℝ) : Prop := distance O A = distance O D ∧ distance O A = R

def on_one_side_of_diameter (B C : Point) (AD : ℝ) : Prop := -- express the positioning relative to the diameter

def circumcircle_intersection (O A D B C F E : Point) : Prop :=
  circumsircle_abo_intersects_bc_at_f ∧ circumsircle_cdo_intersects_bc_at_e

-- Proof goal
theorem af_de_equals_r_squared (h1: is_circle O A D R) (h2: on_one_side_of_diameter B C (distance A D)) (h3: circumcircle_intersection O A D B C F E) :
  distance A F * distance D E = R ^ 2 :=
sorry

end af_de_equals_r_squared_l28_28441


namespace area_of_triangle_ABC_l28_28773

-- Defining a function to validate the calculation of the area of the triangle
def triangle_area (a b c B : ℝ) : ℝ :=
  (1 / 2) * a * c * (Real.sin B)

theorem area_of_triangle_ABC :
  ∀ (a b c : ℝ), b = 6 → a = 2 * c → B = Real.pi / 3 → 
  triangle_area a b c B = 6 * Real.sqrt 3 :=
by
  -- The goal is to prove the above statement given the conditions
  sorry

end area_of_triangle_ABC_l28_28773


namespace students_exceed_goldfish_l28_28651

theorem students_exceed_goldfish 
    (num_classrooms : ℕ) 
    (students_per_classroom : ℕ) 
    (goldfish_per_classroom : ℕ) 
    (h1 : num_classrooms = 5) 
    (h2 : students_per_classroom = 20) 
    (h3 : goldfish_per_classroom = 3) 
    : (students_per_classroom * num_classrooms) - (goldfish_per_classroom * num_classrooms) = 85 := by
  sorry

end students_exceed_goldfish_l28_28651


namespace can_increase_averages_by_transfer_l28_28342

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ := grades.sum / grades.length

theorem can_increase_averages_by_transfer :
    ∃ (student : ℝ) (new_group1_grades new_group2_grades : List ℝ),
      student ∈ group1_grades ∧
      new_group1_grades = (group1_grades.erase student) ∧
      new_group2_grades = student :: group2_grades ∧
      average new_group1_grades > average group1_grades ∧ 
      average new_group2_grades > average group2_grades :=
by
  sorry

end can_increase_averages_by_transfer_l28_28342


namespace donuts_left_for_sophie_l28_28479

def initial_boxes := 4
def donuts_per_box := 12
def boxes_given_to_mom := 1
def donuts_given_to_sister := 6

theorem donuts_left_for_sophie :
  let initial_donuts := initial_boxes * donuts_per_box in
  let remaining_boxes := initial_boxes - boxes_given_to_mom in
  let remaining_donuts := remaining_boxes * donuts_per_box in
  let donuts_left := remaining_donuts - donuts_given_to_sister in
  donuts_left = 30 := 
by 
  have initial_donuts := initial_boxes * donuts_per_box
  have remaining_boxes := initial_boxes - boxes_given_to_mom
  have remaining_donuts := remaining_boxes * donuts_per_box
  have donuts_left := remaining_donuts - donuts_given_to_sister
  show donuts_left = 30
  calc
  donuts_per_box * (initial_boxes - boxes_given_to_mom) - donuts_given_to_sister = donuts_per_box * 3 - donuts_given_to_sister := by sorry
  donuts_per_box * 3 - donuts_given_to_sister = 30 := by sorry

end donuts_left_for_sophie_l28_28479


namespace interval_contains_zero_l28_28268

noncomputable def f (x : ℝ) : ℝ := (6 / x) - Real.log2 x

theorem interval_contains_zero : ∃ c ∈ Ioo 2 4, f c = 0 :=
by
  have h_cont := continuous_on_div (continuous_const.mul continuous_id').continuous_on (continuous_at_log_id.continuous_on.comp' continuous_on_id) (λ x, ne_of_gt (by linarith [(show 2 > 0, by linarith), (show x > 0, by linarith)])) ;
  have h_pos  : f 2 > 0 := by norm_num [f, Real.log2] ;
  have h_neg  : f 4 < 0 := by norm_num [f, Real.log2] ;
  apply intermediate_value' h_cont;
  { interval },
  { exact lt_of_lt_of_le h_neg ((mem_Icc_of_Ioc (by linarith)).1 _) },
  { exact le_of_lt h_pos }
sorry

end interval_contains_zero_l28_28268


namespace problem1_problem2_problem3_l28_28153

variable {m n p x : ℝ}

-- Problem 1
theorem problem1 : m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := 
sorry

-- Problem 2
theorem problem2 : (p - 3) * (p - 1) + 1 = (p - 2) ^ 2 := 
sorry

-- Problem 3
theorem problem3 (hx : x^2 + x + 1 / 4 = 0) : (2 * x + 1) / (x + 1) + (x - 1) / 1 / (x + 2) / (x^2 + 2 * x + 1) = -1 / 4 :=
sorry

end problem1_problem2_problem3_l28_28153


namespace counterexample_to_multiple_of_8_l28_28599

theorem counterexample_to_multiple_of_8 : ∃ x, (x = 4) ∧ (x % 2 = 0) ∧ (x % 8 ≠ 0) :=
by {
  use 4,
  split,
  { exact rfl },
  split,
  { exact rfl },
  { sorry }
}

end counterexample_to_multiple_of_8_l28_28599


namespace tutoring_minutes_l28_28287

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l28_28287


namespace find_a_l28_28373

def curve1 (a t : ℝ) : ℝ × ℝ :=
  (a - (Real.sqrt 2) / 2 * t, 1 + (Real.sqrt 2) / 2 * t)

def curve2 (theta rho : ℝ) : Prop :=
  rho * (Real.cos theta)^2 + 2 * Real.cos theta - rho = 0

noncomputable def is_in_between (A P B : ℝ) : Prop := A < P ∧ P < B

theorem find_a (a t1 t2 : ℝ) (ha : t1 = -2 * t2)
  (h1 : t1 + t2 = -4 * Real.sqrt 2)
  (h2 : t1 * t2 = 2 * (1 - 2 * a)) :
  a = 33 / 2 :=
  sorry

end find_a_l28_28373


namespace solve_for_x_l28_28004

-- Define the equation as a predicate
def equation (x : ℝ) : Prop := (0.05 * x + 0.07 * (30 + x) = 15.4)

-- The proof statement
theorem solve_for_x :
  ∃ x : ℝ, equation x ∧ x = 110.8333 :=
by
  existsi (110.8333 : ℝ)
  split
  sorry -- Proof of the equation
  rfl -- Equality proof

end solve_for_x_l28_28004


namespace iced_cubes_count_l28_28947

theorem iced_cubes_count {n : ℕ} (n = 5) :
  ∀ (cake : ℕ → ℕ → ℕ → ℕ),
    (∀ x y z, 0 ≤ x ∧ x < n ∧ 0 ≤ y ∧ y < n ∧ 0 ≤ z ∧ z < n → cake x y z = if (x = n - 1 ∨ y = n - 1 ∨ z = 0) then 1 else 0) →
∞ (∀ (f : ℕ → ℕ → ℕ),
  (f x y z = ( if (z = 0 ∧ ∨ x = n - 1 ∧ y = n - 1) 
      + if (y = 0 ∨ ∧ x = n - 1 ∧ z = 0)
      + if (z = 0 ∨ y = n - 1) then 
      ∑ (x, y, z), f(x,y,z) = 32
  ) → sorry


end iced_cubes_count_l28_28947


namespace sequence_product_l28_28771

theorem sequence_product {n : ℕ} (h : 1 < n) (a : ℕ → ℕ) (h₀ : ∀ n, a n = 2^n) : 
  a (n-1) * a (n+1) = 4^n :=
by sorry

end sequence_product_l28_28771


namespace smaller_angle_in_parallelogram_l28_28930

theorem smaller_angle_in_parallelogram (a b : ℝ) (h1 : a + b = 180)
  (h2 : b = a + 70) : a = 55 :=
by sorry

end smaller_angle_in_parallelogram_l28_28930


namespace given_problem_l28_28255

theorem given_problem :
  3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end given_problem_l28_28255


namespace arithmetic_sequence_l28_28327

theorem arithmetic_sequence
  (a b c : ℝ)
  (A B C : ℝ) 
  (triangle_ABC : A + B + C = π)  -- Assumption of triangle angles sum
  (h1 : a * cos^2 (C / 2) + cos^2 (A / 2) = 3 * b / 2):
  ∃ d : ℝ, a = b - d ∧ c = b + d := by 
  sorry

end arithmetic_sequence_l28_28327


namespace lewis_age_l28_28426

-- Defining the ages of the brothers
def ages := {4, 6, 8, 10, 12}

-- Defining the conditions
def went_to_movies (a b : ℕ) : Prop := a + b = 18
def went_to_soccer (a b : ℕ) : Prop := a > 5 ∧ a < 11 ∧ b > 5 ∧ b < 11
def stayed_home (a : ℕ) : Prop := a = 6 ∨ (a ∈ ages ∧ a ≠ 6)

-- Proving that Lewis' age is 4
theorem lewis_age : ∃ age : ℕ, age ∈ ages ∧ ∀ (x y z : ℕ), 
  (went_to_movies x y → x ≠ age ∧ y ≠ age) →
  (went_to_soccer z (12 - z) → z ≠ age ∧ (12 - z) ≠ age) →
  stayed_home age :=
sorry

end lewis_age_l28_28426


namespace determine_x_l28_28696

theorem determine_x (x : ℤ) (h1 : (sqrt 27 : ℝ) < (x : ℝ)) (h2 : (x : ℝ) < 7) : x = 6 := 
sorry

end determine_x_l28_28696


namespace circus_balloons_l28_28487

theorem circus_balloons:
  -- Conditions
  (n_red : ℕ) (h1 : n_red = 40)
  (missing_yellow : ℕ) (h2 : missing_yellow = 3)
  -- Conclusion
  (n_yellow : ℕ) (h_yellow : n_yellow = n_red - 1 + missing_yellow)
  (n_blue : ℕ) (h_blue : n_blue = (n_red + n_yellow) - 1)
  -- Expected values
  (h3 : n_yellow = 42)
  (h4 : n_blue = 81)
  : n_yellow = 42 ∧ n_blue = 81 := 
by {
  cases h1, 
  cases h2, 
  cases h3, 
  cases h4,
  split,
  exact h3,
  exact h4,
  sorry -- Placeholder to indicate that this is the statement only and the proof is omitted.
}

end circus_balloons_l28_28487


namespace circle_C_equation_line_OP_parallel_to_AB_l28_28697

noncomputable def circle_M_radius := sorry

def Point {α : Type*} (x y : α) := (x, y)

def Circle (center : Point ℝ) (radius : ℝ) : Set (Point ℝ) :=
  { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def symmetric_point (p : Point ℝ) (line_slope : ℝ) (line_y_intercept : ℝ) : Point ℝ :=
  let t := (p.1 + p.2 + 2) / 2
  in (2 * (-2) - p.1, 2 * (-2) - p.2)

def point_O : Point ℝ := (0, 0)

def point_P : Point ℝ := (1, 1)

def circle_M := Circle (Point (-2) (-2)) circle_M_radius

def circle_C : Circle (symmetric_point (Point (-2) (-2)) 1 2) 2 := 
  Circle (symmetric_point (Point (-2) (-2)) 1 2) 2

theorem circle_C_equation : ∃ r, ∀ p, p ∈ circle_C ↔ p.1^2 + p.2^2 = r :=
by sorry

theorem line_OP_parallel_to_AB : 
   (line_slope (0, 0) (1, 1) = line_slope (A, B)) :=
by sorry

end circle_C_equation_line_OP_parallel_to_AB_l28_28697


namespace tadd_3000th_num_solution_l28_28365

noncomputable def tadd_3000th_num : ℕ :=
  let block_size n := n*(n+1) // 2 in -- Sum of the first n blocks
  let block_start n := block_size n + 1 in
  let nth_tadd_num n := block_start n + (3000 - block_size n) in
  if 3000 ≤ block_size 21 then block_start 21 - block_start 20 + 6297 else block_start 22 + 1592

theorem tadd_3000th_num_solution : tadd_3000th_num = 6297 :=
  by sorry

end tadd_3000th_num_solution_l28_28365


namespace can_transfer_increase_average_l28_28357

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l28_28357


namespace max_intersection_points_circles_lines_l28_28905

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l28_28905


namespace product_not_equal_to_N_l28_28983

noncomputable def is_possible_product_equal (digits : Fin 100 → Nat) : Bool :=
  let N := ∑ i, digits i * 10^i
  let P := ∏ i : Fin 100, ∏ j : Fin 100, if i ≠ j then digits i + digits j else 1
  N = P

theorem product_not_equal_to_N (digits : Fin 100 → Nat) (h1 : digits 99 ≠ 0) :
  ¬ is_possible_product_equal digits :=
by
  sorry

end product_not_equal_to_N_l28_28983


namespace value_of_m_l28_28317

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem value_of_m (a b m : ℝ) (h₀ : m ≠ 0)
  (h₁ : 3 * m^2 + 2 * a * m + b = 0)
  (h₂ : m^2 + a * m + b = 0)
  (h₃ : ∃ x, f x a b = 1/2) :
  m = 3/2 :=
by
  sorry

end value_of_m_l28_28317


namespace ratio_red_to_yellow_l28_28292

structure MugCollection where
  total_mugs : ℕ
  red_mugs : ℕ
  blue_mugs : ℕ
  yellow_mugs : ℕ
  other_mugs : ℕ
  colors : ℕ

def HannahCollection : MugCollection :=
  { total_mugs := 40,
    red_mugs := 6,
    blue_mugs := 6 * 3,
    yellow_mugs := 12,
    other_mugs := 4,
    colors := 4 }

theorem ratio_red_to_yellow
  (hc : MugCollection)
  (h_total : hc.total_mugs = 40)
  (h_blue : hc.blue_mugs = 3 * hc.red_mugs)
  (h_yellow : hc.yellow_mugs = 12)
  (h_other : hc.other_mugs = 4)
  (h_colors : hc.colors = 4) :
  hc.red_mugs / hc.yellow_mugs = 1 / 2 := by
  sorry

end ratio_red_to_yellow_l28_28292


namespace simply_connected_polyhedron_inequality_l28_28449

-- Define the necessary conditions
variables {σ3 σ4 σ5 : ℕ} -- Number of various types of faces
variables {B3 B4 B5 : ℕ} -- Number of vertex figures of various types
variables {E V σ : ℕ} -- Total number of edges, vertices, and faces

-- Conditions from the problem
def total_faces := σ = σ3 + σ4 + σ5 -- Definition of total faces
def total_edges := E = (3 * σ3 + 4 * σ4 + 5 * σ5) / 2 -- Definition of total unique edges
def total_angles := 3 * σ3 + 4 * σ4 + 5 * σ5 = 2 * E -- Each angle counted in exactly one face
def total_vertex_figures := B3 + B4 + B5 -- Total number of vertex figures
def vertex_angles := 3 * B3 + 4 * B4 + 5 * B5 = 2 * E
def eulers_formula := V - E + σ = 2 -- Euler's formula for simply connected polyhedron

-- Theorem representing the main question
theorem simply_connected_polyhedron_inequality
    (h1 : total_faces)
    (h2 : total_edges)
    (h3 : total_angles)
    (h4 : vertex_angles)
    (h5 : eulers_formula) :
    B3 + σ3 ≥ 8 :=
by
  sorry -- skipping the proof

end simply_connected_polyhedron_inequality_l28_28449


namespace sonny_cookies_given_to_sister_l28_28470

theorem sonny_cookies_given_to_sister:
  ∀ (total_boxes : ℕ) (given_to_brother : ℕ) (given_to_cousin : ℕ) (boxes_left : ℕ),
  total_boxes = 45 →
  given_to_brother = 12 →
  given_to_cousin = 7 →
  boxes_left = 17 →
  (total_boxes - given_to_brother - given_to_cousin - boxes_left) = 9 :=
by
  intros total_boxes given_to_brother given_to_cousin boxes_left
  assume h_total_boxes h_given_to_brother h_given_to_cousin h_boxes_left
  rw [h_total_boxes, h_given_to_brother, h_given_to_cousin, h_boxes_left]
  calc
    45 - 12 - 7 - 17 = 45 - (12 + 7) - 17 : by rw Nat.sub_sub
    ... = 45 - 19 - 17 : by rw add_comm
    ... = 9 : by norm_num

end sonny_cookies_given_to_sister_l28_28470


namespace totalWheelsInStorageArea_l28_28519

def numberOfBicycles := 24
def numberOfTricycles := 14
def wheelsPerBicycle := 2
def wheelsPerTricycle := 3

theorem totalWheelsInStorageArea :
  numberOfBicycles * wheelsPerBicycle + numberOfTricycles * wheelsPerTricycle = 90 :=
by
  sorry

end totalWheelsInStorageArea_l28_28519


namespace derivative_has_root_in_interval_l28_28874

noncomputable def polynomial (a b c : ℝ) : ℝ → ℝ :=
  λ x => (x - a) * (x - b) * (x - c)

noncomputable def derivative (a b c : ℝ) : ℝ → ℝ :=
  λ x => (x - b) * (x - c) + (x - a) * (x - c) + (x - a) * (x - b)

theorem derivative_has_root_in_interval (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) :
  ∃ x ∈ Icc ((b + c) / 2) ((b + 2 * c) / 3), derivative a b c x = 0 :=
  sorry

end derivative_has_root_in_interval_l28_28874


namespace initial_action_figures_l28_28397

theorem initial_action_figures (x : ℕ) (h : x + 4 - 1 = 6) : x = 3 :=
by {
  sorry
}

end initial_action_figures_l28_28397


namespace plane_equation_rewriting_l28_28498

theorem plane_equation_rewriting (A B C D x y z p q r : ℝ)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (eq1 : A * x + B * y + C * z + D = 0)
  (hp : p = -D / A) (hq : q = -D / B) (hr : r = -D / C) :
  x / p + y / q + z / r = 1 :=
by
  sorry

end plane_equation_rewriting_l28_28498


namespace distance_A_to_B_l28_28877

theorem distance_A_to_B (perimeter_sm : 8 = 4 * s) (area_lg : 25 = L^2) : 
    let horizontal_side := s + L
    let vertical_side := L - s
    d = Real.sqrt (horizontal_side^2 + vertical_side^2) →
    d = Real.sqrt 58 :=
by {
  intro hside,
  intro vside,
  intro dist,
  rw [hside, vside],
  have side_s : s = 2,
  { linarith },
  have side_L : L = 5,
  { linarith },
  rw [side_s, side_L],
  have horiz : horizontal_side = 7,
  { linarith },
  have vert : vertical_side = 3,
  { linarith },
  rw [horiz, vert, Real.sqrt_eq_iff_sq_eq, pow_two, pow_two, add_comm],
  sorry -- Proof step: Real.sqrt (7^2 + 3^2) = Real.sqrt 58
}

end distance_A_to_B_l28_28877


namespace sum_units_digits_3a_l28_28800

theorem sum_units_digits_3a (a : ℕ) (h_pos : 0 < a) (h_units : (2 * a) % 10 = 4) : 
  ((3 * (a % 10) = (6 : ℕ) ∨ (3 * (a % 10) = (21 : ℕ))) → 6 + 1 = 7) := 
by
  sorry

end sum_units_digits_3a_l28_28800


namespace number_of_houses_with_pool_l28_28755

theorem number_of_houses_with_pool :
  ∀ (total houses_with_garage houses_with_both houses_with_neither : ℕ),
    total = 90 →
    houses_with_garage = 50 →
    houses_with_both = 35 →
    houses_with_neither = 35 →
    let houses_with_pool := 90 - houses_with_neither + houses_with_both - houses_with_garage in
      houses_with_pool = 40 :=
by
  intros total houses_with_garage houses_with_both houses_with_neither h_total h_g h_b h_n
  have : houses_with_pool = 40 := sorry
  exact this

end number_of_houses_with_pool_l28_28755


namespace inscribed_triangle_area_l28_28630

open Real

theorem inscribed_triangle_area (r : ℝ) (h_r : r = 8) : 
  let d := 2 * r in
  let base := d in
  let height := r in
  let area := (1 / 2) * base * height in
  area = 64 := 
by 
  have d_eq : 2 * r = 16 :=
    by rw [h_r]; norm_num,
  have base_eq : d = 16 :=
    by exact d_eq,
  have height_eq : r = 8 :=
    by exact h_r,
  have area_eq : (1 / 2) * base * height = 64 :=
    by rw [base_eq, height_eq]; norm_num,
  exact area_eq

end inscribed_triangle_area_l28_28630


namespace second_to_last_digit_of_special_number_l28_28575

theorem second_to_last_digit_of_special_number :
  ∀ (N : ℕ), (N % 10 = 0) ∧ (∃ k : ℕ, k > 0 ∧ N = 2 * 5^k) →
  (N / 10) % 10 = 5 :=
by
  sorry

end second_to_last_digit_of_special_number_l28_28575


namespace complement_union_l28_28279

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4, 5}

theorem complement_union :
  (U \ A) ∪ (U \ B) = {1, 2, 3, 6} := 
by 
  sorry

end complement_union_l28_28279


namespace number_of_correct_propositions_l28_28977

-- Define the propositions as Booleans
def proposition1 : Prop := ∀ (b a : ℝ) (x y : ℝ), y = b * x + a → (x = (x + y) / 2 ∧ y = (x + y) / 2)
def proposition2 : Prop := ∀ (x : ℝ), (λ x, 12 - 0.2 * x) x = 12 - 0.2 * (x + 1)
def proposition3 : Prop := ∀ (p : ℝ), p = 0.99 → ¬(p = 0.99 → ∀ (x y : ℝ), x = y)
def proposition4 : Prop := ∃ (χ2 : ℝ), χ2 = 13.709 ∧ (P (χ2 ≥ 10.828) = 0.001)

-- The main theorem
theorem number_of_correct_propositions : (proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4) → 2 = 2 :=
by sorry

end number_of_correct_propositions_l28_28977


namespace composite_product_quotient_l28_28650

def first_seven_composite := [4, 6, 8, 9, 10, 12, 14]
def next_eight_composite := [15, 16, 18, 20, 21, 22, 24, 25]

noncomputable def product {α : Type*} [Monoid α] (l : List α) : α :=
  l.foldl (· * ·) 1

theorem composite_product_quotient : 
  (product first_seven_composite : ℚ) / (product next_eight_composite : ℚ) = 1 / 2475 := 
by 
  sorry

end composite_product_quotient_l28_28650


namespace age_of_15th_student_l28_28485

theorem age_of_15th_student
  (avg_age_15_students : ℕ) (avg_age_6_students : ℕ) (avg_age_8_students : ℕ)
  (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (H_avg15 : avg_age_15_students = 15) (H_avg6 : avg_age_6_students = 14) (H_avg8 : avg_age_8_students = 16)
  (H_total : total_students = 15) (H_group1 : group1_students = 6) (H_group2 : group2_students = 8) :
  ∃ (age_15th_student : ℕ), age_15th_student = 13 :=
by
  let total_age := total_students * avg_age_15_students
  let total_age_group1 := group1_students * avg_age_6_students
  let total_age_group2 := group2_students * avg_age_8_students
  let age_15th_student := total_age - (total_age_group1 + total_age_group2)
  have H1 : total_age = 225, by rw [H_total, H_avg15]; norm_num
  have H2 : total_age_group1 = 84, by rw [H_group1, H_avg6]; norm_num
  have H3 : total_age_group2 = 128, by rw [H_group2, H_avg8]; norm_num
  have H4 : age_15th_student = 225 - (84 + 128), by rw [H1, H2, H3]; norm_num
  use 13
  rw [H4]; norm_num
  sorry

end age_of_15th_student_l28_28485


namespace triangle_perimeter_l28_28661

theorem triangle_perimeter (a b c : ℕ) (ha : a = 7) (hb : b = 10) (hc : c = 15) :
  a + b + c = 32 :=
by
  -- Given the lengths of the sides
  have H1 : a = 7 := ha
  have H2 : b = 10 := hb
  have H3 : c = 15 := hc
  
  -- Therefore, we need to prove the sum
  sorry

end triangle_perimeter_l28_28661


namespace lcm_color_sum_not_2016_l28_28937

theorem lcm_color_sum_not_2016 (a : ℕ) (red blue : fin 10 → ℕ) (h1 : ∀ i, red i = a + i ∨ blue i = a + i)
  (h2 : ∃ i j, i ≠ j ∧ (red i = a + i ∧ blue j = a + j)) :
  ¬(∃ s : ℕ, s = Nat.lcm (List.of_fn red).foldr Nat.lcm 1 + Nat.lcm (List.of_fn blue).foldr Nat.lcm 1 ∧ s % 10000 = 2016) :=
by
  sorry

end lcm_color_sum_not_2016_l28_28937


namespace max_area_rect_l28_28128

/--
A rectangle has a perimeter of 40 units and its dimensions are whole numbers.
The maximum possible area of the rectangle is 100 square units.
-/
theorem max_area_rect {l w : ℕ} (hlw : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h_sum : l + w = 20 := by
    rw [two_mul, two_mul, add_assoc, add_assoc] at hlw
    exact Nat.div_eq_of_eq_mul_left (by norm_num : 2 ≠ 0) hlw

  have into_parabola : ∀ l w, l + w = 20 → l * w ≤ 100 := λ l w h_eq =>
  by
    let expr := l * w
    let w_def := 20 - l
    let expr' := l * (20 - l)
    have key_expr: l * w = l * (20 - l) := by
      rw h_eq
    rw key_expr
    let f (l: ℕ) := l * (20 - l)
    have step_expr: l * (20 - l) = 20*l - l^2 := by
      ring

    have boundary : (0 ≤ l * (20 - l)) := mul_nonneg (by apply l.zero_le) (by linarith)
    have max_ex : ((20 / 2)^2 ≤ 100) := by norm_num
    let sq_bound:= 100 - (l - 10)^2
    have complete_sq : 20 * l - l^2 = -(l-10)^2 + 100  := by
      have q_expr: 20 * l - l^2 = - (l-10)^2 + 100 := by linarith
      exact q_expr

    show l * (20 - l) ≤ 100,
    from Nat.le_of_pred_lt (by linarith)


  exact into_parabola l w h_sum

end max_area_rect_l28_28128


namespace geometric_series_remainder_l28_28913

theorem geometric_series_remainder :
  let S := (9^2024 - 1) / 8 in
  S % 500 = 45 :=
by
  sorry

end geometric_series_remainder_l28_28913


namespace bags_of_hammers_to_load_l28_28643

noncomputable def total_crate_capacity := 15 * 20
noncomputable def weight_of_nails := 4 * 5
noncomputable def weight_of_planks := 10 * 30
noncomputable def weight_to_be_left_out := 80
noncomputable def effective_capacity := total_crate_capacity - weight_to_be_left_out
noncomputable def weight_of_loaded_planks := 220

theorem bags_of_hammers_to_load : (effective_capacity - weight_of_nails - weight_of_loaded_planks = 0) :=
by
  sorry

end bags_of_hammers_to_load_l28_28643


namespace bridge_length_l28_28864

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (cross_time_seconds : ℝ)
  (train_length_eq : train_length = 150)
  (train_speed_kmph_eq : train_speed_kmph = 45)
  (cross_time_seconds_eq : cross_time_seconds = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 225 := 
  by
  sorry

end bridge_length_l28_28864


namespace problem1_problem2_problem3_problem4_problem5_problem6_l28_28090

-- Problem 1
theorem problem1 (x : ℝ) : |8 - 3 * x| > 0 ↔ x ≠ 8 / 3 := sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 - 3 * |x| + 2 ≤ 0 ↔ (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2) := sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h1 : 0 < a ∧ 0 < b) (h2 : a * b = 2) : (1 + 2 * a) * (1 + b) ≥ 9 := sorry

-- Problem 4
theorem problem4 (x y : ℝ) (h : (x - 1) * 4 + 2 * y = 0) : 9 ^ x + 3 ^ y ≥ 6 := sorry

-- Problem 5
theorem problem5 (a : ℝ) (m n : ℝ) (h1 : 0 < m ∧ 0 < n) (h2 : m + n = 1) (h3 : a > 0 ∧ a ≠ 1) : (y = a^(1 - x) ∧ y = 1) → (m * 1 + n * 1 - 1 = 0) → (1 / m + 1 / n ≥ 4) := sorry -- Note: h3 is added for constraints on 'a'

-- Problem 6
theorem problem6 (x a : ℝ) : (x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l28_28090


namespace inscribed_circle_radius_eq_l28_28066

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  area / s

theorem inscribed_circle_radius_eq :
  radius_of_inscribed_circle 8 8 10 8 8 10 = (15 * Real.sqrt 13) / 13 :=
  sorry

end inscribed_circle_radius_eq_l28_28066


namespace axis_of_symmetry_l28_28734

-- Define the condition that f(x) = f(3 - x) for all x.
variable {f : ℝ → ℝ}
axiom symmetry_property : ∀ x : ℝ, f(x) = f(3 - x)

-- The theorem statement that the graph of y = f(x) is symmetric with respect to the line x = 1.5.
theorem axis_of_symmetry : ∀ x : ℝ, f x = f (3 - x) → x = 1.5 :=
by
  intro x hx
  -- proof is omitted
  sorry

end axis_of_symmetry_l28_28734


namespace magnitude_of_vector_diff_l28_28281

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, -2)
def vector_diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vector_diff : magnitude vector_diff = 5 := by
  sorry

end magnitude_of_vector_diff_l28_28281


namespace correct_statements_l28_28258

def valid_statements (a b : ℝ) : Prop :=
  let C := (x y : ℝ) => (x * |x|) / (a^2) + (y * |y|) / (b^2) = 1
  let statement1 := ∀ x : ℝ, ∃! y : ℝ, C x y
  let statement2 := ∀ k m : ℝ, ∀ (p q r s : ℝ), (C p (k * p + m) ∧ C q (k * q + m) ∧ C r (k * r + m) ∧ C s (k * s + m)) → False
  let statement3 := ¬ (∀ (x y : ℝ), C x y ↔ C y x)
  let statement4 := ∀ (x1 y1 x2 y2 : ℝ), C x1 y1 ∧ C x2 y2 → (y1 - y2) / (x1 - x2) < 0
  statement1 ∧ statement2 ∧ statement4

theorem correct_statements (a b : ℝ) : valid_statements a b :=
  by
    sorry

end correct_statements_l28_28258


namespace representation_of_2015_l28_28457

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l28_28457


namespace p_sufficient_but_not_necessary_for_q_l28_28729

-- Definitions
variable {p q : Prop}

-- The condition: ¬p is a necessary but not sufficient condition for ¬q
def necessary_but_not_sufficient (p q : Prop) : Prop :=
  (∀ q, ¬q → ¬p) ∧ (∃ q, ¬q ∧ p)

-- The theorem stating the problem
theorem p_sufficient_but_not_necessary_for_q 
  (h : necessary_but_not_sufficient (¬p) (¬q)) : 
  (∀ p, p → q) ∧ (∃ p, p ∧ ¬q) :=
sorry

end p_sufficient_but_not_necessary_for_q_l28_28729


namespace find_x_angle_l28_28377

theorem find_x_angle (ABC ACB CDE : ℝ) (h1 : ABC = 70) (h2 : ACB = 90) (h3 : CDE = 42) : 
  ∃ x : ℝ, x = 158 :=
by
  sorry

end find_x_angle_l28_28377


namespace boys_girls_ratio_l28_28110

-- Definitions used as conditions
variable (B G : ℕ)

-- Conditions
def condition1 : Prop := B + G = 32
def condition2 : Prop := B = 2 * (G - 8)

-- Proof that the ratio of boys to girls initially is 1:1
theorem boys_girls_ratio (h1 : condition1 B G) (h2 : condition2 B G) : (B : ℚ) / G = 1 := by
  sorry

end boys_girls_ratio_l28_28110


namespace area_of_overlap_l28_28887

-- Define the properties of the 45-45-90 triangles, including the hypotenuse length
def hypotenuse_length : ℝ := 10

-- Define the leg length using the 45-45-90 triangle property
def leg_length := hypotenuse_length / Real.sqrt 2

-- State the theorem to be proven: The area of the overlapping region of the two triangles
theorem area_of_overlap :
  let A := (1 / 2) * (leg_length * leg_length) in
  A = 25 := by
  sorry

end area_of_overlap_l28_28887


namespace ones_digit_8_power_32_l28_28056

theorem ones_digit_8_power_32 : (8^32) % 10 = 6 :=
by sorry

end ones_digit_8_power_32_l28_28056


namespace find_vector_b_l28_28881

theorem find_vector_b :
  ∃ (b : ℝ × ℝ × ℝ), 
  (∃ a t : ℝ × ℝ × ℝ, a = t • (1, 1, 1) ∧ a + b = (6, -3, -6) ∧ dot_product b (1, 1, 1) = 0) → 
  b = (7, -2, -5) :=
by
  sorry

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

end find_vector_b_l28_28881


namespace square_area_l28_28208

theorem square_area (perimeter : ℝ) (h : perimeter = 40) : ∃ A : ℝ, A = 100 := by
  have h1 : ∃ s : ℝ, 4 * s = perimeter := by
    use perimeter / 4
    linarith

  cases h1 with s hs
  use s^2
  rw [hs, h]
  norm_num
  sorry

end square_area_l28_28208


namespace transfer_student_increases_averages_l28_28346

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ :=
  if grades.length > 0 then (grades.sum / grades.length) else 0

def can_transfer_to_increase_averages (group1_grades group2_grades : List ℝ) : Prop :=
  ∃ x ∈ group1_grades, average (x :: group2_grades) > average group2_grades ∧
                       average (group1_grades.erase x) > average group1_grades

theorem transfer_student_increases_averages :
  can_transfer_to_increase_averages group1_grades group2_grades :=
by {
  sorry -- The proof is skipped as per instructions
}

end transfer_student_increases_averages_l28_28346


namespace find_ff_neg_one_l28_28264

def f (x: ℝ) : ℝ :=
  if x < 0 then 2^(x + 2) else x^3

theorem find_ff_neg_one : f (f (-1)) = 8 := by
  sorry

end find_ff_neg_one_l28_28264


namespace radius_of_inscribed_circle_l28_28057

-- Define the triangle side lengths
def DE : ℝ := 8
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- The statement to be proved
theorem radius_of_inscribed_circle : r = (5 * Real.sqrt 39) / 13 := by
  sorry

end radius_of_inscribed_circle_l28_28057


namespace count_total_legs_l28_28882

theorem count_total_legs :
  let tables4 := 4 * 4
  let sofa := 1 * 4
  let chairs4 := 2 * 4
  let tables3 := 3 * 3
  let table1 := 1 * 1
  let rocking_chair := 1 * 2
  let total_legs := tables4 + sofa + chairs4 + tables3 + table1 + rocking_chair
  total_legs = 40 :=
by
  sorry

end count_total_legs_l28_28882


namespace smallest_integer_larger_than_x_pow8_l28_28534

def sqrt5 : ℝ := Real.sqrt 5
def sqrt3 : ℝ := Real.sqrt 3
def x : ℝ := sqrt5 + sqrt3

theorem smallest_integer_larger_than_x_pow8 : 
  ∃ n : ℕ, n = 1631 ∧ (n : ℝ) > x^8 := by
  sorry

end smallest_integer_larger_than_x_pow8_l28_28534


namespace asymptotes_of_hyperbola_l28_28273

theorem asymptotes_of_hyperbola (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (c : ℝ) (h₃ : ∀ x y : ℝ, x + 3 * y - 2 * b = 0 → x = c) :
  (c = 2 * b) → (a = real.sqrt (c^2 - b^2)) → (a = real.sqrt 3 * b) → 
  (∀ x, y = (real.sqrt 3 / 3) * x ∨ y = - (real.sqrt 3 / 3) * x) :=
sorry

end asymptotes_of_hyperbola_l28_28273


namespace gcd_base5_representation_l28_28170

theorem gcd_base5_representation (a b : ℕ) (h1 : a = 4034) (h2 : b = 10085) :
  (nat_to_base 5 (Nat.gcd a b)) = [3, 1, 0, 3, 2] :=
by 
  sorry

end gcd_base5_representation_l28_28170


namespace constant_term_expansion_l28_28488

theorem constant_term_expansion :
  let f := (x - 1) ^ 4 * (1 + 1 / x) ^ 4 in
  ∃ c : ℚ, is_constant_term f c ∧ c = 6 := 
by
  sorry

end constant_term_expansion_l28_28488


namespace cats_to_dogs_l28_28026

theorem cats_to_dogs (c d : ℕ) (h1 : c = 24) (h2 : 4 * d = 5 * c) : d = 30 :=
by
  sorry

end cats_to_dogs_l28_28026


namespace quadrilaterals_area_and_perimeter_l28_28168

-- Define the vertices of the quadrilaterals
def quadrilateral_I_vertices := [(0, 0), (3, 0), (3, 3), (0, 2)]
def quadrilateral_I_perimeter := 3 + 3 + Real.sqrt(2^2 + 3^2) + 3 -- Calculated as 12.16 units

def quadrilateral_II_vertices := [(0, 0), (3, 0), (3, 2), (0, 3)]
def quadrilateral_II_perimeter := 3 + 2 + 3 + Real.sqrt(3^2 + 2^2) -- Calculated as 11.6 units

-- Area of both quadrilaterals
def quadrilateral_area := 7.5 -- Both areas are calculated to be 7.5

-- The proof statement
theorem quadrilaterals_area_and_perimeter :
  (quadrilateral_I_vertices.area = quadrilateral_II_vertices.area) ∧ 
  (quadrilateral_I_perimeter > quadrilateral_II_perimeter) :=
by
  exact (quadrilateral_area = 7.5) ∧ (quadrilateral_I_perimeter > quadrilateral_II_perimeter)

end quadrilaterals_area_and_perimeter_l28_28168


namespace jean_gives_480_per_year_l28_28781

/-
  Given:
  - Jean has 3 grandchildren.
  - She buys each grandkid 2 cards a year.
  - She puts $80 in each card.

  Prove:
  - Jean gives away $480 to her grandchildren each year.
-/

theorem jean_gives_480_per_year 
  (grandchildren : ℕ := 3) 
  (cards_per_grandchild : ℕ := 2) 
  (amount_per_card : ℕ := 80) : 
  let amount_per_grandchild := cards_per_grandchild * amount_per_card,
      total_amount := grandchildren * amount_per_grandchild
  in total_amount = 480 := 
by
  sorry

end jean_gives_480_per_year_l28_28781


namespace maximum_u_achieved_when_a_eq_neg3_l28_28871

theorem maximum_u_achieved_when_a_eq_neg3:
  ∃ (r s t : ℕ), (r ≥ 1) ∧ (s ≥ 1) ∧ (t ≥ 1) ∧ (r + s + t = 4) ∧
  let u := (-1)^(r+s) * 2^s * 3^t + 3 in
  u = 21 :=
by
  sorry

end maximum_u_achieved_when_a_eq_neg3_l28_28871


namespace range_of_b_l28_28270

noncomputable def f (x : ℝ) (a : ℝ) := log x - a * x + (1 - a) / x - 1

noncomputable def g (x : ℝ) (b : ℝ) := x^2 - 2 * b * x + 4

theorem range_of_b (b : ℝ) : 
  (∀ x1 ∈ Ioo 0 2, ∃ x2 ∈ Icc 1 2, f x1 (1/4) ≥ g x2 b) → b ≥ 17/8 :=
by
  sorry

end range_of_b_l28_28270


namespace inscribed_circle_radius_of_DEF_l28_28063

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l28_28063


namespace inscribed_circle_radius_of_DEF_l28_28061

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l28_28061


namespace shots_cost_l28_28628

theorem shots_cost (n_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) :
  n_dogs = 3 →
  puppies_per_dog = 4 →
  shots_per_puppy = 2 →
  cost_per_shot = 5 →
  n_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    3 * 4 * 2 * 5 = 12 * 2 * 5 : by rfl
    ... = 24 * 5 : by rfl
    ... = 120 : by rfl

end shots_cost_l28_28628


namespace equal_distance_on_perpendicular_bisector_l28_28824

theorem equal_distance_on_perpendicular_bisector
  (A B P : Type)
  [metric_space P]
  (h_perpendicular_bisector : ∀ x : P, dist x A = dist x B)
  (h_PA : dist P A = 5) :
  dist P B = 5 := 
by
  sorry

end equal_distance_on_perpendicular_bisector_l28_28824


namespace factorization_correctness_l28_28917

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end factorization_correctness_l28_28917


namespace area_of_region_B_l28_28634

def region_B (z : ℂ) : Prop :=
  let z_div_50 := z / 50
  let fifty_div_conj_z := 50 / conj z
  let real_part_cond := 0 ≤ z_div_50.re ∧ z_div_50.re ≤ 1
  let imag_part_cond := 0 ≤ z_div_50.im ∧ z_div_50.im ≤ 1
  let real_part_div_cond := 0 ≤ (fifty_div_conj_z.re) ∧ (fifty_div_conj_z.re) ≤ 1
  let imag_part_div_cond := 0 ≤ (fifty_div_conj_z.im) ∧ (fifty_div_conj_z.im) ≤ 1
  real_part_cond ∧ imag_part_cond ∧ real_part_div_cond ∧ imag_part_div_cond

def area_of_B : ℝ := 3125 - 625 * Real.pi

theorem area_of_region_B : 
  (∃ z : ℂ, region_B z) → area_of_B = 3125 - 625 * Real.pi :=
sorry

end area_of_region_B_l28_28634


namespace maria_remaining_towels_l28_28548

def total_towels_initial := 40 + 44
def towels_given_away := 65

theorem maria_remaining_towels : (total_towels_initial - towels_given_away) = 19 := by
  sorry

end maria_remaining_towels_l28_28548


namespace fish_weight_after_5_years_fish_weight_decreases_after_5_years_l28_28955

-- Definition of fish weight growth without environmental pollution
def fish_weight_growth (a0 : ℝ) (year : ℕ) : ℝ :=
  if year = 0 then a0
  else if year = 1 then 3 * a0
  else fish_weight_growth a0 (year - 1) * (1 + (1 / 2)^(year - 2))

-- Proof statement for Part 1
theorem fish_weight_after_5_years (a0 : ℝ) :
  fish_weight_growth a0 5 = (405 / 32) * a0 := sorry

-- Definition of fish weight growth with environmental pollution
def fish_weight_with_pollution (a0 : ℝ) (year : ℕ) : ℝ :=
  if year = 0 then a0
  else fish_weight_with_pollution a0 (year - 1) * (1 + (1 / 2)^(year - 1)) * (9 / 10)

-- Proof statement for Part 2
theorem fish_weight_decreases_after_5_years (a0 : ℝ) :
  ∃ n, n ≥ 5 ∧ fish_weight_with_pollution a0 n < fish_weight_with_pollution a0 (n - 1) := sorry

end fish_weight_after_5_years_fish_weight_decreases_after_5_years_l28_28955


namespace modulus_of_z_l28_28256

noncomputable def z (w : ℂ) := 2 * complex.I / (1 - complex.I)

theorem modulus_of_z : |z (2 * complex.I)| = sqrt 2 := 
sorry

end modulus_of_z_l28_28256


namespace simplify_fraction_multiplication_l28_28846

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end simplify_fraction_multiplication_l28_28846


namespace max_min_difference_l28_28422

open Real

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 2 * y = 4) :
  ∃(max min : ℝ), (∀z, z = (|2 * x - y| / (|x| + |y|)) → z ≤ max) ∧ 
                  (∀z, z = (|2 * x - y| / (|x| + |y|)) → min ≤ z) ∧ 
                  (max - min = 5) :=
by
  sorry

end max_min_difference_l28_28422


namespace problem_two_probability_l28_28037

-- Definitions based on conditions
def physics_problems : List ℕ := [1, 2, 3, 4, 5]
def chemistry_problems : List ℕ := [6, 7, 8, 9]
def all_problems : List ℕ := physics_problems ++ chemistry_problems

def all_combinations : List (ℕ × ℕ) :=
  (all_problems.product all_problems).filter (λ (p : ℕ × ℕ), p.fst < p.snd)

def valid_combinations := all_combinations.filter (λ (p : ℕ × ℕ), 11 ≤ p.fst + p.snd ∧ p.fst + p.snd < 17)

-- Lean 4 statement
theorem problem_two_probability :
  (all_combinations.length = 36) ∧
  (valid_combinations.length = 15) →
  (valid_combinations.length.to_real / all_combinations.length.to_real = 5/12) := by
  sorry

end problem_two_probability_l28_28037


namespace explicit_expression_for_P_l28_28138

noncomputable def expression_for_P (n r s : ℕ) : ℕ :=
  6^(s + 2) * (Finset.sum (Finset.range (s - r + 1).map (λ k, binomial n (r + k) * 6^(n - (r + k)))))

theorem explicit_expression_for_P (n r s : ℕ) (hnr : n < 2 * r) (hrs : r < s) :
  let P := expression_for_P n r s in
  P = 6^(s + 2) * (Finset.sum (Finset.range (s - r + 1).map (λ k, binomial n (r + k) * 6^(n - (r + k))))) := 
sorry

end explicit_expression_for_P_l28_28138


namespace rob_baseball_cards_l28_28839

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end rob_baseball_cards_l28_28839


namespace length_of_chord_l28_28714

theorem length_of_chord {x1 x2 : ℝ} (h1 : ∃ (y : ℝ), y^2 = 8 * x1)
                                   (h2 : ∃ (y : ℝ), y^2 = 8 * x2)
                                   (h_midpoint : (x1 + x2) / 2 = 3) :
  x1 + x2 + 4 = 10 :=
sorry

end length_of_chord_l28_28714


namespace fraction_not_covered_l28_28087

-- Definitions of the diameters of the frames
def diameterX : ℝ := 16
def diameterY : ℝ := 12

-- Radii of the frames
def radiusX : ℝ := diameterX / 2
def radiusY : ℝ := diameterY / 2

-- Areas of the frames
def areaX : ℝ := Real.pi * radiusX^2
def areaY : ℝ := Real.pi * radiusY^2

-- Fraction of surface of frame X not covered by frame Y
theorem fraction_not_covered : (areaX - areaY) / areaX = 7 / 16 :=
by
  sorry

end fraction_not_covered_l28_28087


namespace annual_population_increase_rounded_l28_28380

noncomputable def births_per_day : ℝ := 24 / 6
noncomputable def deaths_per_day : ℝ := 1 / 2
noncomputable def immigrants_per_day : ℝ := 1 / 3

noncomputable def net_change_per_day : ℝ := births_per_day + immigrants_per_day - deaths_per_day

noncomputable def annual_population_increase : ℝ := net_change_per_day * 365

theorem annual_population_increase_rounded : 
  Float.to_nearest 100 annual_population_increase = 1400 := 
by { unfold annual_population_increase, unfold net_change_per_day, 
     unfold births_per_day, unfold deaths_per_day, unfold immigrants_per_day,
     sorry }

end annual_population_increase_rounded_l28_28380


namespace area_of_square_with_perimeter_40_l28_28202

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l28_28202


namespace angle_between_vectors_l28_28719

-- Lean code for the problem

variables {a b : Vector3} (ha : a.norm = 1) (hb : b.norm = 1)

theorem angle_between_vectors (h : ∥a - b∥ = √3 * ∥a + b∥) : 
  real.angle_between a b = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_l28_28719


namespace maximize_sector_area_l28_28741

noncomputable def max_area_sector_angle (r : ℝ) (l := 36 - 2 * r) (α := l / r) : ℝ :=
  α

theorem maximize_sector_area (h : ∀ r : ℝ, 2 * r + 36 - 2 * r = 36) :
  max_area_sector_angle 9 = 2 :=
by
  sorry

end maximize_sector_area_l28_28741


namespace hollow_iron_ball_diameter_l28_28115

theorem hollow_iron_ball_diameter (R r : ℝ) (s : ℝ) (thickness : ℝ) 
  (h1 : thickness = 1) (h2 : s = 7.5) 
  (h3 : R - r = thickness) 
  (h4 : 4 / 3 * π * R^3 = 4 / 3 * π * s * (R^3 - r^3)) : 
  2 * R = 44.44 := 
sorry

end hollow_iron_ball_diameter_l28_28115


namespace area_of_region_l28_28209

noncomputable def regionArea : set (ℝ × ℝ) := 
  {p | ∀ t ∈ Icc 0 1, |p.1 + p.2 * t + t^2| ≤ 1}

theorem area_of_region :
  measure_theory.measure.measure (regionArea) = (8/3 : ℝ) :=
sorry

end area_of_region_l28_28209


namespace find_initial_amounts_l28_28492

-- Define the players and their final amount after all rounds
def players : Type := sorry -- represent the type for players

-- Define the final amount of money each player has
def final_amount : players → ℝ := λ _ => 1.28

-- Define the initial amount each player had
def initial_amount : players → ℝ
| A => 4.49
| B => 2.25
| C => 1.13
| D => 0.57
| E => 0.29
| F => 0.15
| G => 0.08
| _ => sorry

-- The main theorem to prove the initial amount based on game rules
theorem find_initial_amounts :
  ∀ (p : players), 
  (if p = A then initial_amount p = 4.49 else
   if p = B then initial_amount p = 2.25 else
   if p = C then initial_amount p = 1.13 else
   if p = D then initial_amount p = 0.57 else
   if p = E then initial_amount p = 0.29 else
   if p = F then initial_amount p = 0.15 else
   if p = G then initial_amount p = 0.08 else
   false) :=
by
(\* The proof must be constructed here *)
sorry

end find_initial_amounts_l28_28492


namespace triangle_inequality_l28_28834

-- Define the problem conditions and the theorem
noncomputable def triangle := Type*

structure Circumcircle (T : triangle) :=
(A B C A1 B1 C1 : T)
(is_A1_angle_bisector : A1 ∈ circle_formed_by A B C)
(is_B1_angle_bisector : B1 ∈ circle_formed_by A B C)
(is_C1_angle_bisector : C1 ∈ circle_formed_by A B C)

theorem triangle_inequality (T : triangle) (circumcircle : Circumcircle T) :
  let A, B, C := circumcircle.A, circumcircle.B, circumcircle.C
  let A1, B1, C1 := circumcircle.A1, circumcircle.B1, circumcircle.C1
  AA1 + BB1 + CC1 > AB + BC + CA :=
sorry

end triangle_inequality_l28_28834


namespace transfer_student_increases_averages_l28_28344

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ :=
  if grades.length > 0 then (grades.sum / grades.length) else 0

def can_transfer_to_increase_averages (group1_grades group2_grades : List ℝ) : Prop :=
  ∃ x ∈ group1_grades, average (x :: group2_grades) > average group2_grades ∧
                       average (group1_grades.erase x) > average group1_grades

theorem transfer_student_increases_averages :
  can_transfer_to_increase_averages group1_grades group2_grades :=
by {
  sorry -- The proof is skipped as per instructions
}

end transfer_student_increases_averages_l28_28344


namespace rectangle_area_l28_28136

theorem rectangle_area {H W : ℝ} (h_height : H = 24) (ratio : W / H = 0.875) :
  H * W = 504 :=
by 
  sorry

end rectangle_area_l28_28136


namespace pond_fish_count_l28_28754

theorem pond_fish_count :
  (∃ (N : ℕ), (2 / 50 : ℚ) = (40 / N : ℚ)) → N = 1000 :=
by
  sorry

end pond_fish_count_l28_28754


namespace quadrilateral_is_square_and_center_l28_28390

-- Define the geometric conditions of the problem
variables {A B C D O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
variables (S : ℝ) (area : ℝ) [convex A B C D]
variable (AO BO CO DO : A → ℝ)

-- Define point O and its property
def condition (O : A) (S : ℝ) : Prop :=
AO O ^ 2 + BO O ^ 2 + CO O ^ 2 + DO O ^ 2 = 2 * S

-- Goal: Prove that the quadrilateral is a square and O is its center
theorem quadrilateral_is_square_and_center {A B C D O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
  (S : ℝ) [convex A B C D] 
  (h1 : condition O S) :
  is_square A B C D ∧ is_center O A B C D :=
  by
    sorry

end quadrilateral_is_square_and_center_l28_28390


namespace maximum_disjoint_regions_l28_28413

theorem maximum_disjoint_regions (p : ℕ) (h_prime : Nat.Prime p) (h_geq : p ≥ 3) : 
    let ABC := Triangle.create points_A B C in
    let cev_segments := create_cevians (ABC, p) in
    let disjoint_regions := count_disjoint_regions(cev_segments) in
    disjoint_regions = 3 * p^2 - 3 * p + 1 :=
begin
  sorry
end

end maximum_disjoint_regions_l28_28413


namespace divisors_count_30_l28_28295

theorem divisors_count_30 : 
  (∃ n : ℤ, n > 1 ∧ 30 % n = 0) 
  → 
  (∃ k : ℕ, k = 14) :=
by
  sorry

end divisors_count_30_l28_28295


namespace AE_plus_FC_eq_AC_l28_28828

-- Declare the given geometric elements and conditions.
variables (A B C D E F : Type) [OrderedRing A] [AffineSpace B A]

-- Assume a geometry context with points and segments.
variables (AC AB BC : Segment A B) (isosceles : ∀ P Q R : Point B, AC.contains P → AB.contains Q → BC.contains R → P = Q → Q = R)
(hD: AC.contains D) (hE : AB.contains E) (hF : BC.contains F)
(hDEDF : dist D E = dist D F) 
(hAngle : ∀ (P Q R : Point B), AC.contains P → AB.contains Q → BC.contains R → ∠P A Q = ∠F D E) 

-- Prove the required relation between segments.
theorem AE_plus_FC_eq_AC: dist A E + dist F C = dist A C :=
sorry

end AE_plus_FC_eq_AC_l28_28828


namespace range_of_a_for_inequality_l28_28215

theorem range_of_a_for_inequality : 
  ∃ a : ℝ, (∀ x : ℤ, (a * x - 1) ^ 2 < x ^ 2) ↔ 
    (a > -3 / 2 ∧ a ≤ -4 / 3) ∨ (4 / 3 ≤ a ∧ a < 3 / 2) :=
by
  sorry

end range_of_a_for_inequality_l28_28215


namespace simplify_fraction_l28_28847

theorem simplify_fraction (a b c d : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 5) (h4 : d = 4) : 
  (a + b * complex.I) / (c - d * complex.I) = (-2 / 41) + (64 / 41) * complex.I := by
  sorry

end simplify_fraction_l28_28847


namespace number_of_correct_statements_l28_28176

def statement_1_correct : Prop :=
  ∀ (A B C D : Set Point), Trapezoid A B C D → Coplanar {A, B, C, D}

def statement_2_correct : Prop :=
  ∀ (l1 l2 l3 : Line), (Parallel l1 l2 ∧ Parallel l2 l3 ∧ Parallel l1 l3) → Coplanar {l1, l2, l3}

def statement_3_correct : Prop :=
  ∀ (P Q R: Point) (π1 π2 : Plane), (Collinear P Q R) ∧ (On P π1 ∧ On Q π1 ∧ On R π1) ∧ (On P π2 ∧ On Q π2 ∧ On R π2) → (π1 = π2)

theorem number_of_correct_statements :
  (statement_1_correct = true) ∧ (statement_2_correct = false) ∧ (statement_3_correct = false) → (1 = 1) :=
by sorry

end number_of_correct_statements_l28_28176


namespace product_of_number_subtracting_7_equals_9_l28_28481

theorem product_of_number_subtracting_7_equals_9 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end product_of_number_subtracting_7_equals_9_l28_28481


namespace quadratic_property_l28_28271

noncomputable def f (x c : ℝ) : ℝ := x^2 + 4*x + c

theorem quadratic_property (c : ℝ) : 
  let f0 := f 0 c,
      f1 := f 1 c,
      f_neg2 := f (-2) c
  in f1 > f0 ∧ f0 > f_neg2 :=
by
  archimedes sorry

end quadratic_property_l28_28271


namespace problem_statement_l28_28728

variable (n : ℕ)
variable (op : ℕ → ℕ → ℕ)
variable (h1 : op 1 1 = 1)
variable (h2 : ∀ n, op (n+1) 1 = 3 * op n 1)

theorem problem_statement : op 5 1 - op 2 1 = 78 := by
  sorry

end problem_statement_l28_28728


namespace find_increase_in_inches_l28_28953

-- Conditions
def original_radius : ℝ := 10
def original_height : ℝ := 5
def volume_original := Real.pi * original_radius^2 * original_height

-- Increase in inches is x
def new_radius (x : ℝ) := original_radius + x
def new_height (x : ℝ) := original_height + x
def volume_new (x : ℝ) := Real.pi * (new_radius x)^2 * (new_height x)

-- Given volume_new = 4 * volume_original
def satisfies_condition (x : ℝ) : Prop :=
  volume_new x = 4 * volume_original

-- Prove that the increase x is 10 inches
theorem find_increase_in_inches (x : ℝ) (h : satisfies_condition x) : x = 10 := by
  sorry

end find_increase_in_inches_l28_28953


namespace can_increase_average_l28_28334

def student_grades := List (String × Nat)

def group1 : student_grades := 
    [("Andreev", 5), ("Borisova", 3), ("Vasilieva", 5), ("Georgiev", 3),
     ("Dmitriev", 5), ("Evstigneeva", 4), ("Ignatov", 3), ("Kondratiev", 4),
     ("Leontieva", 3), ("Mironov", 4), ("Nikolaeva", 5), ("Ostapov", 5)]
     
def group2 : student_grades := 
    [("Alexeeva", 3), ("Bogdanov", 4), ("Vladimirov", 5), ("Grigorieva", 2),
     ("Davydova", 3), ("Evstahiev", 2), ("Ilina", 5), ("Klimova", 4),
     ("Lavrentiev", 5), ("Mikhailova", 3)]

def average_grade (grp : student_grades) : ℚ :=
    (grp.map (λ x => x.snd)).sum / grp.length

def updated_group (grp : List (String × Nat)) (student : String × Nat) : List (String × Nat) :=
    grp.erase student

def transfer_student (g1 g2 : student_grades) (student : String × Nat) : student_grades × student_grades :=
    (updated_group g1 student, student :: g2)

theorem can_increase_average (s : String × Nat) 
    (h1 : s ∈ group1)
    (h2 : s.snd = 4)
    (h3 : average_grade group1 < s.snd)
    (h4 : average_grade group2 > s.snd)
    : 
    let (new_group1, new_group2) := transfer_student group1 group2 s in 
    average_grade new_group1 > average_grade group1 ∧ 
    average_grade new_group2 > average_grade group2 :=
sorry

#eval can_increase_average ("Evstigneeva", 4)

end can_increase_average_l28_28334


namespace ellipse_eccentricity_range_l28_28239

theorem ellipse_eccentricity_range (a b c e : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : c = a * e)
  (h4 : ∀ x y₁ y₂, (x, y₁) ∈ set_of (λ (p : ℝ × ℝ), (p.1 / a) ^ 2 + (p.2 / b) ^ 2 = 1) → (x, y₂) ∈ set_of (λ (p : ℝ × ℝ), (p.1 / a) ^ 2 + (p.2 / b) ^ 2 = 1) → y₁ ≠ -y₂ → y₁ * y₂ = - (b^2 / a^2))
  (h5 : ∀ a b c, 0 < b ∧ b^2 < 2 * a * c → 0 < e ∧ e < 1) :
  (sqrt 2 - 1 < e ∧ e < 1) :=
by sorry

end ellipse_eccentricity_range_l28_28239


namespace benny_number_of_kids_l28_28609

-- Define the conditions
def benny_has_dollars (d: ℕ): Prop := d = 360
def cost_per_apple (c: ℕ): Prop := c = 4
def apples_shared (num_kids num_apples: ℕ): Prop := num_apples = 5 * num_kids

-- State the main theorem
theorem benny_number_of_kids : 
  ∀ (d c k a : ℕ), benny_has_dollars d → cost_per_apple c → apples_shared k a → k = 18 :=
by
  intros d c k a hd hc ha
  -- The goal is to prove k = 18; use the provided conditions
  sorry

end benny_number_of_kids_l28_28609


namespace find_integers_l28_28200

theorem find_integers (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * Real.sqrt (Real.cbrt 7 - Real.cbrt 6) = Real.cbrt x + Real.cbrt y - Real.cbrt z :=
sorry

end find_integers_l28_28200


namespace merchant_marked_price_l28_28964

theorem merchant_marked_price (L : ℝ) (purchase_price : ℝ) (marked_price : ℝ) (selling_price : ℝ) :
  (purchase_price = L * 0.7) ∧ (selling_price = marked_price * 0.8) ∧ (selling_price = purchase_price * 1.3) →
  (marked_price = L * 1.25) :=
by
  intros h,
  obtain ⟨hp, hs, hsp⟩ := h,
  -- Solution steps proving this theorem are omitted as this is only a statement
  sorry

end merchant_marked_price_l28_28964


namespace equilateral_triangle_inscribed_circle_radius_l28_28603

theorem equilateral_triangle_inscribed_circle_radius (s : ℝ) (h : ℝ) (r : ℝ) 
  (equilateral : ∀ {A B C : ℝ}, is_equilateral_triangle A B C s) 
  (side_length : s = 18) 
  (altitude : h = 9 * Real.sqrt 3)
  (area : (1/2) * s * h = 81 * Real.sqrt 3)
  (semiperimeter : (3 * s) / 2 = 27)
  (area_formula : 81 * Real.sqrt 3 = r * 27) :
  r = 3 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_inscribed_circle_radius_l28_28603


namespace choosing_students_l28_28675

theorem choosing_students
  (harvard_students : Finset ℕ) (mit_students : Finset ℕ)
  (h₁ : harvard_students.card = 4) (mit_students.card = 5)
  (jack : ℕ) (h_jack : jack ∈ harvard_students)
  (jill : ℕ) (h_jill : jill ∈ mit_students)
  (total_tickets : ℕ := 5)
  (total_students := harvard_students ∪ mit_students) :
  (Finset.choose total_students total_tickets).card
  - (Finset.choose mit_students total_tickets).card
  - (Finset.choose (total_students.erase jack).erase jill total_tickets).card
  = 104 :=
by sorry

end choosing_students_l28_28675


namespace sum_of_digits_of_large_product_l28_28869

theorem sum_of_digits_of_large_product : 
  (M : ℕ) 
  (M_value : M = 2^122 * 3^86) 
  (S : ℕ) 
  (S_value : S = 75) :
  (36^50 * 144^36 = M^2) → 
  (sum_of_digits M = S) :=
begin
  intro h,
  rw M_value at *,
  sorry
end

end sum_of_digits_of_large_product_l28_28869


namespace symmetry_axes_not_in_interval_l28_28265

noncomputable def f (ω x : ℝ) := sin (ω * x) - cos (ω * x)

theorem symmetry_axes_not_in_interval (ω : ℝ) (hω : ω > 1/4) :
    (∀ x, f ω x ≠ 0 ∨ (2 * π < x ∧ x < 3 * π)) ↔ (3 / 8 ≤ ω ∧ ω ≤ 7 / 12) ∨ (7 / 8 ≤ ω ∧ ω ≤ 11 / 12) := 
sorry

end symmetry_axes_not_in_interval_l28_28265


namespace pete_walked_miles_closest_to_3600_l28_28831

theorem pete_walked_miles_closest_to_3600
    (steps_per_cycle : ℕ := 90000)
    (resets_per_year : ℕ := 60)
    (steps_on_last_day : ℕ := 30000)
    (steps_per_mile : ℕ := 1500) :
  let total_steps := steps_per_cycle * resets_per_year + steps_on_last_day
  let walked_miles := total_steps / steps_per_mile
  abs (walked_miles - 3600) < abs (walked_miles - x) ∀ x ∈ {3200, 3600, 4000, 4500, 5000}  :=
by
  sorry

end pete_walked_miles_closest_to_3600_l28_28831


namespace f_f_2012_eq_neg1_l28_28709

def f (x : ℝ) : ℝ := 
  if x ≤ 2000 then 2 * Real.cos (π / 3 * x) 
  else x - 12

theorem f_f_2012_eq_neg1 : f (f 2012) = -1 :=
  sorry

end f_f_2012_eq_neg1_l28_28709


namespace find_speed_train2_l28_28946

/-- Let's define the necessary conditions and translates the math problem to Lean. -/

def length_train1 : ℝ := 200
def speed_train1 : ℝ := 120
def length_train2 : ℝ := 300.04
def time_cross : ℝ := 9
def speed_train2 : ℝ := 80.016

theorem find_speed_train2 :
  let V_rel := (length_train1 + length_train2) / (9 / 3600) in 
  V_rel = speed_train1 + speed_train2 :=
sorry

end find_speed_train2_l28_28946


namespace roots_polynomial_squares_l28_28414

/-- Let p, q, and r be the roots of the polynomial x^3 - 24x^2 + 50x - 8 = 0.
Compute (p + q)^2 + (q + r)^2 + (r + p)^2. -/
theorem roots_polynomial_squares (p q r : ℝ)
  (h1 : p + q + r = 24)
  (h2 : p * q + q * r + r * p = 50)
  (h3 : p * q * r = 8) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 1052 :=
by
  calc
    (p + q)^2 + (q + r)^2 + (r + p)^2 
      = 2 * (p^2 + q^2 + r^2) + 2 * (p * q + q * r + r * p) : by sorry
    ... = 2 * ((p + q + r)^2 - 2 * (p * q + q * r + r * p)) + 2 * (p * q + q * r + r * p) : by sorry
    ... = 2 * 24^2 - 2 * 50 : by { rw [h1, h2] }
    ... = 1052 : by { norm_num }

end roots_polynomial_squares_l28_28414


namespace number_of_n_tuples_l28_28791

theorem number_of_n_tuples (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_n : 2 ≤ n)
  (h_S0 : S 0 = 1)
  (h_S : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → S k = ∑ s in finset.powerset_len k (finset.range n), (∏ x in s, a (x+1))) :
  (∑ k in finset.range((n+1)/2), (-1)^k * S (n-2*k)) ^ 2 + (∑ k in finset.range(n/2), (-1)^k * S (n-1-2*k)) ^ 2 = 2^n * S n →
  ∃ N : ℕ, N = 2^(n-1) :=
sorry

end number_of_n_tuples_l28_28791


namespace isosceles_triangle_perimeter_l28_28870

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4 ∨ a = 7) (h2 : b = 4 ∨ b = 7) (h3 : a ≠ b) :
  (a + a + b = 15 ∨ a + a + b = 18) ∨ (a + b + b = 15 ∨ a + b + b = 18) :=
sorry

end isosceles_triangle_perimeter_l28_28870


namespace calculate_shot_cost_l28_28623

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l28_28623


namespace ratio_of_albums_l28_28817

variable (M K B A : ℕ)
variable (s : ℕ)

-- Conditions
def adele_albums := (A = 30)
def bridget_albums := (B = A - 15)
def katrina_albums := (K = 6 * B)
def miriam_albums := (M = s * K)
def total_albums := (M + K + B + A = 585)

-- Proof statement
theorem ratio_of_albums (h1 : adele_albums A) (h2 : bridget_albums B A) (h3 : katrina_albums K B) 
(h4 : miriam_albums M s K) (h5 : total_albums M K B A) :
  s = 5 :=
by
  sorry

end ratio_of_albums_l28_28817


namespace count_valid_4digit_integers_l28_28294

theorem count_valid_4digit_integers: 
  let first_two_digits := {1, 4, 5, 6}
  let last_two_digits := {3, 5, 8}
  (∃ (n : ℕ), 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (n / 1000 ∈ first_two_digits ∧ (n / 100 % 10) ∈ first_two_digits) ∧ 
    (n % 10 ≠ n / 10 % 10) ∧ 
    (n / 10 % 10 ∈ last_two_digits ∧ n % 10 ∈ last_two_digits) ∧ 
    count_valid_4digit_integers n = 96) := 
sorry

end count_valid_4digit_integers_l28_28294


namespace inscribed_circle_radius_eq_l28_28065

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  area / s

theorem inscribed_circle_radius_eq :
  radius_of_inscribed_circle 8 8 10 8 8 10 = (15 * Real.sqrt 13) / 13 :=
  sorry

end inscribed_circle_radius_eq_l28_28065


namespace clock_hands_angle_120_l28_28051

noncomputable def degree_hour (t : ℝ) : ℝ := 30 * t
noncomputable def degree_minute (t : ℝ) : ℝ := 360 * t
def angle_between (t : ℝ) : ℝ := abs (degree_hour t - degree_minute t)

theorem clock_hands_angle_120 : 
  let minute : ℝ := 7 + m / 60 in
  (7 * 60 + 5 = minute * 60 ∧ 7 * 60 + 16 = minute * 60) →
  (angle_between (m : ℝ) = 120) :=
by
  sorry

end clock_hands_angle_120_l28_28051


namespace consecutive_odd_integers_expressions_l28_28483

theorem consecutive_odd_integers_expressions
  {p q : ℤ} (hpq : p + 2 = q ∨ p - 2 = q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) :
  (2 * p + 5 * q) % 2 = 1 ∧ (5 * p - 2 * q) % 2 = 1 ∧ (2 * p * q + 5) % 2 = 1 :=
  sorry

end consecutive_odd_integers_expressions_l28_28483


namespace solution_set_l28_28694

noncomputable def f : ℝ → ℝ := sorry
variable (x : ℝ)

axiom f_diff : ∀ x > 0, differentiable_at ℝ f x
axiom f_ineq : ∀ x > 0, f(x) > x * deriv f x

theorem solution_set :
  ∀ x > 0, (x^2 * f (1 / x) - f x < 0) ↔ (0 < x ∧ x < 1) :=
by
  intros x hx
  sorry

end solution_set_l28_28694


namespace triangle_area_k_plus_p_l28_28045

theorem triangle_area_k_plus_p (A B C D E F G : Type) [IsTriangle A B C] 
(BC : dist B C = 24) (incircle_trisects_median : ∃ D, median_splits_A_to_D_into_thirds A D B) :
  ∃ (k p : ℕ), area A B C = k * sqrt p ∧ (p_is_not_square_prime : ∀ (q : ℕ), prime q → ¬(q ^ 2 ∣ p)) ∧ k + p = 51 := 
sorry

end triangle_area_k_plus_p_l28_28045


namespace maximize_output_l28_28941

-- Define the conditions as parameters
variables (total_workers : ℕ) (workers_A : ℕ) (workers_B : ℕ)
          (electricity_A : ℕ) (electricity_B : ℕ) (max_electricity : ℕ)
          (output_A : ℕ) (output_B : ℕ)

-- Set the specific values for this problem
def problem_conditions : Prop :=
  total_workers = 12 ∧ workers_A = 2 ∧ workers_B = 3 ∧
  electricity_A = 30 ∧ electricity_B = 20 ∧ max_electricity = 130 ∧
  output_A = 4 ∧ output_B = 3

-- Main theorem: Given the conditions, prove the maximum output is 18 ten thousand yuan
theorem maximize_output (h : problem_conditions) : 
  ∃ (num_A num_B : ℕ), 
    (num_A * workers_A + num_B * workers_B ≤ total_workers) ∧
    (num_A * electricity_A + num_B * electricity_B ≤ max_electricity) ∧
    (num_A * output_A + num_B * output_B = 18) :=
sorry

end maximize_output_l28_28941


namespace Billys_age_l28_28159

variable (B J : ℕ)

theorem Billys_age :
  B = 2 * J ∧ B + J = 45 → B = 30 :=
by
  sorry

end Billys_age_l28_28159


namespace max_intersection_points_three_circles_two_lines_l28_28901

theorem max_intersection_points_three_circles_two_lines : 
  ∀ (C1 C2 C3 L1 L2 : set ℝ × ℝ) (hC1 : is_circle C1) (hC2 : is_circle C2) (hC3 : is_circle C3) (hL1 : is_line L1) (hL2 : is_line L2),
  ∃ P : ℕ, P = 19 ∧
  (∀ (P : ℝ × ℝ), P ∈ C1 ∧ P ∈ C2 ∨ P ∈ C1 ∧ P ∈ C3 ∨ P ∈ C2 ∧ P ∈ C3 ∨ P ∈ C1 ∧ P ∈ L1 ∨ P ∈ C2 ∧ P ∈ L1 ∨ P ∈ C3 ∧ P ∈ L1 ∨ P ∈ C1 ∧ P ∈ L2 ∨ P ∈ C2 ∧ P ∈ L2 ∨ P ∈ C3 ∧ P ∈ L2 ∨ P ∈ L1 ∧ P ∈ L2) ↔ P = 19 :=
sorry

end max_intersection_points_three_circles_two_lines_l28_28901


namespace tan_double_angle_l28_28680

theorem tan_double_angle (α : ℝ) (h : Real.tan (α + π / 4) = √3 - 2) : Real.tan (2 * α) = √3 :=
by
  sorry

end tan_double_angle_l28_28680


namespace ana_average_speed_l28_28604

def steps_per_floor := 14
def floors := 50
def speed_up := 3 -- steps per second
def speed_down := 5 -- steps per second
def height_per_step := 0.583 -- feet

theorem ana_average_speed :
  let steps_total := steps_per_floor * floors
  let time_up := steps_total / speed_up
  let time_down := steps_total / speed_down
  let total_time := time_up + time_down
  let building_height := steps_per_floor * floors * height_per_step
  let distance := 2 * building_height
  let distance_miles := distance / 5280
  let time_hours := total_time / 3600
  let avg_speed := distance_miles / time_hours
  abs (avg_speed - 1.486) < 0.001 :=
by
  sorry

end ana_average_speed_l28_28604


namespace unique_polynomial_function_l28_28660

theorem unique_polynomial_function :
  ∃! (f : ℝ[X]), (∀ x : ℝ, f.map(λ x, x^2) = f.eval x * f.eval x ∧ f.eval 0 = 0 ∧ f.eval(f.eval x) = f.eval x * f.eval x) ∧ (0 < f.degree) :=
sorry

end unique_polynomial_function_l28_28660


namespace sum_of_inscribed_circle_radii_in_polygon_l28_28763

theorem sum_of_inscribed_circle_radii_in_polygon (n : ℕ) (hn : n ≥ 3) (P : list (ℝ × ℝ)) 
  (Hinscribed : ∀ (A B : ℝ × ℝ), (A ∈ P) → (B ∈ P) → dist A B ≤ diam (circumcircle P))
  (diags : list (ℝ × ℝ)) (Hdiags_non_intersecting : true) -- non-intersecting diagonals condition should be defined
  (triangles : list (list (ℝ × ℝ))) (Htriangles : true) -- condition that diagonals divide polygon into triangles
  : ∃ k, ∀ (d : list (ℝ × ℝ)) (Hd : d ⊆ diags ∧ non_intersecting d),
    sum (λ t, radius (incircle t)) (triangulate P d) = k :=
  sorry

end sum_of_inscribed_circle_radii_in_polygon_l28_28763


namespace total_savings_is_correct_l28_28884

-- Define the prices
def ticket_price := 10
def large_popcorn_and_drink_cost := 10
def medium_nachos_and_drink_cost := 8
def hotdog_and_soft_drink_cost := 6

-- Define the discounts
def ticket_discount := 0.20
def large_popcorn_discount := 0.50
def medium_nachos_discount := 0.30
def hotdog_discount := 0.20

-- Define number of items purchased
def tickets_count := 3
def large_popcorn_and_drink_count := 1
def medium_nachos_and_drink_count := 1
def hotdog_and_soft_drink_count := 1

-- Define the savings calculations
def ticket_savings := tickets_count * ticket_price * ticket_discount
def large_popcorn_savings := large_popcorn_and_drink_count * large_popcorn_and_drink_cost * large_popcorn_discount
def medium_nachos_savings := medium_nachos_and_drink_count * medium_nachos_and_drink_cost * medium_nachos_discount
def hotdog_savings := hotdog_and_soft_drink_count * hotdog_and_soft_drink_cost * hotdog_discount

def total_savings := ticket_savings + large_popcorn_savings + medium_nachos_savings + hotdog_savings

theorem total_savings_is_correct : total_savings = 14.60 := by
  -- This is where the proof would go
  sorry

end total_savings_is_correct_l28_28884


namespace gnuff_tutoring_minutes_l28_28289

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l28_28289


namespace distinct_colorings_of_polygon_l28_28089

-- Given conditions
variables (p a : ℕ)
variables [hp : Fact (Nat.Prime p)]

-- Theorem statement
theorem distinct_colorings_of_polygon : 
  (∃ p a, Nat.Prime p ∧ ∀ (a : ℕ), a ≥ 1 → (a^p - a) / p + a = (a^p - a) / p + a) := 
sorry

end distinct_colorings_of_polygon_l28_28089


namespace percentage_increase_l28_28398

theorem percentage_increase (original new : ℝ) (h_original : original = 60) (h_new : new = 68) : 
  ((new - original) / original) * 100 ≈ 13.33 :=
by
  simp [h_original, h_new]
  norm_num
  sorry

end percentage_increase_l28_28398


namespace diameter_bounds_l28_28925

-- Define the conditions and parameters
variables {e f d : ℝ} (α : ℝ)
def quadrilateral_area :=
  (1:ℝ)

def diagonal_condition (e f d : ℝ) : Prop :=
  e ≤ d ∧ f ≤ d

def sin_alpha_condition (α : ℝ) : Prop :=
  0 ≤ real.sin α ∧ real.sin α ≤ 1

-- The main proposition to prove
theorem diameter_bounds (α : ℝ) (h_sin : sin_alpha_condition α) (h_diag : diagonal_condition e f d)
  (h_area : quadrilateral_area = (1:ℝ)) (h_eq : e * f * real.sin α = 2) : 
  sqrt 2 ≤ d ∧ d < ∞ :=
by sorry

end diameter_bounds_l28_28925


namespace necessary_but_not_sufficient_condition_for_ellipse_l28_28241

theorem necessary_but_not_sufficient_condition_for_ellipse (m : ℝ) :
  (2 < m ∧ m < 6) ↔ ((∃ m, 2 < m ∧ m < 6 ∧ m ≠ 4) ∧ (∀ m, (2 < m ∧ m < 6) → ¬(m = 4))) := 
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l28_28241


namespace molecular_weight_l28_28533

noncomputable def molecularWeightCompound : ℝ :=
  let weight_H := 2 * 1.01 in
  let weight_C := 3 * 12.01 in
  let weight_N := 1 * 14.01 in
  let weight_Cl := 1 * 35.45 in
  let weight_O := 3 * 16.00 in
  weight_H + weight_C + weight_N + weight_Cl + weight_O

theorem molecular_weight : 
  molecularWeightCompound = 135.51 :=
by
  -- Proof goes here
  sorry

end molecular_weight_l28_28533


namespace blue_string_length_l28_28865

def length_red := 8
def length_white := 5 * length_red
def length_blue := length_white / 8

theorem blue_string_length : length_blue = 5 := by
  sorry

end blue_string_length_l28_28865


namespace total_length_correct_l28_28106

noncomputable def length_of_shorter_piece : ℝ := 8.000028571387755
noncomputable def ratio : ℝ := 2.00001 / 5

-- Define the length of the longer piece L based on the given ratio and shorter piece S
noncomputable def length_of_longer_piece (S : ℝ) (r : ℝ) : ℝ := S / r

-- Define the total length of the wire as the sum of the lengths of the shorter and longer pieces
noncomputable def total_length_of_wire (S : ℝ) (L : ℝ) : ℝ := S + L

-- The main theorem that asserts the total length of the wire is approximately 28.0001 cm
theorem total_length_correct : 
  total_length_of_wire length_of_shorter_piece (length_of_longer_piece length_of_shorter_piece ratio) ≈ 28.0001 := by
  sorry

end total_length_correct_l28_28106


namespace excellent_sequences_count_l28_28022

def isExcellentSequence (a : Fin 6 → ℕ) : Prop :=
  (∑ i, abs (a i - (i + 1))) ≤ 4

def countExcellentSequences : ℕ :=
  Finset.card {a : Fin 6 → ℕ | isExcellentSequence a}

theorem excellent_sequences_count : countExcellentSequences = 18 :=
  sorry

end excellent_sequences_count_l28_28022


namespace no_solution_for_g_eq_19_l28_28806

def g (a : ℤ) : ℤ :=
  if a % 2 = 0 then a^2 -- even case
  else if Nat.prime a then a + 5 -- odd and prime case
  else if a % 4 = 1 then a * (a - 1) -- odd, not prime, and a ≡ 1 (mod 4)
  else a - 3 -- odd, not prime, and a ≡ 3 (mod 4)

theorem no_solution_for_g_eq_19 (a : ℤ) : g (g (g (g (g a)))) ≠ 19 :=
by sorry

end no_solution_for_g_eq_19_l28_28806


namespace leah_wins_probability_l28_28787

-- Define the conditions: probabilities for Leah and Ben
def leah_prob_head := 1 / 4
def ben_prob_head := 1 / 3

-- Define the probabilities calculated within the solution
def first_turn_leah_wins := leah_prob_head
def subsequent_turn_leah_wins := ben_prob_head * (1 - leah_prob_head) + (1 - leah_prob_head) * (1 - ben_prob_head) * leah_prob_head

-- Express the infinite series as a geometric series sum
def geom_series_sum (a r : ℝ) : ℝ := a / (1 - r)

-- Calculate the probability that Leah's result is different first
noncomputable def leah_result_different_first := 
  let a := first_turn_leah_wins 
  let r := subsequent_turn_leah_wins
  geom_series_sum a r

-- The final proof state that the probability is 3/5
theorem leah_wins_probability : leah_result_different_first = 3 / 5 := by
  sorry

end leah_wins_probability_l28_28787


namespace log_eq_l28_28301

-- Define the main problem statement indicating the conditions and required result.
theorem log_eq (x : ℝ) (h : log 7 (x + 4) = 4) :
  log 13 x ≈ 3.034 :=
sorry

end log_eq_l28_28301


namespace range_of_f_l28_28873

def f (x : ℝ) : ℝ := (3 + 5 * Real.sin x) / Real.sqrt (5 + 4 * Real.cos x + 3 * Real.sin x)

theorem range_of_f :
  ∀ x, f x ∈ Set.Ioo (- (4 / 5) * Real.sqrt 10) (Real.sqrt 10) ∪ {(Real.sqrt 10)} :=
sorry

end range_of_f_l28_28873


namespace incorrect_statements_l28_28919

theorem incorrect_statements (a b : ℝ×ℝ) (e1 e2 : ℝ×ℝ) :
  (a = (1, 2)) ∧ (b = (1, 1)) ∧ 
  (e1 = (2, -3)) ∧ (e2 = (1/2, -3/4)) →
  (¬(∃ λ : ℝ, λ > -5/3 ∧ λ ≠ 0)) ∧
  (¬(∃ a b : ℝ×ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (∥a∥ > ∥b∥) ∧ (a*b > 0))) :=
by
  sorry

end incorrect_statements_l28_28919


namespace area_of_B_l28_28638

theorem area_of_B :
  let B := {z : ℂ | let x := z.re, y := z.im in 
                    (0 ≤ x ∧ x ≤ 50) ∧ 
                    (0 ≤ y ∧ y ≤ 50) ∧ 
                    (50 * x / (x^2 + y^2) ∈ Icc 0 1) ∧ 
                    (50 * y / (x^2 + y^2) ∈ Icc 0 1) }
  ∃ (area : ℝ), ∀ z ∈ B, area = 1875 - (625 * Real.pi) / 2 :=
by {
  sorry
}

end area_of_B_l28_28638


namespace skate_time_correct_l28_28589

noncomputable def skate_time (path_length miles_length : ℝ) (skating_speed : ℝ) : ℝ :=
  let time_taken := (1.58 * Real.pi) / skating_speed
  time_taken

theorem skate_time_correct :
  skate_time 1 1 4 = 1.58 * Real.pi / 4 :=
by
  sorry

end skate_time_correct_l28_28589


namespace find_angle_C_find_perimeter_l28_28718

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Given conditions
def condition1 : Prop := 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c
def condition2 : Prop := c = Real.sqrt 7
def condition3 : Prop := 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 

-- Statements to prove
theorem find_angle_C (h : condition1) : C = Real.pi / 3 :=
sorry

theorem find_perimeter (h1 : condition1) (h2 : condition2) (h3 : condition3) : a + b + c = 5 + Real.sqrt 7 :=
sorry

end find_angle_C_find_perimeter_l28_28718


namespace axis_of_symmetry_l28_28733

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f(x) = f(3 - x)) : ∀ x, f(x) = f(1.5 + (x - 1.5)) :=
by
  sorry

end axis_of_symmetry_l28_28733


namespace integrate_diff_eq_l28_28776

noncomputable def particular_solution (x y : ℝ) : Prop :=
  (y^2 - x^2) / 2 + Real.exp y - Real.log ((x + Real.sqrt (1 + x^2)) / (2 + Real.sqrt 5)) = Real.exp 1 - 3 / 2

theorem integrate_diff_eq (x y : ℝ) :
  (∀ x y : ℝ, y' = (x * Real.sqrt (1 + x^2) + 1) / (Real.sqrt (1 + x^2) * (y + Real.exp y))) → 
  (∃ x0 y0 : ℝ, x0 = 2 ∧ y0 = 1) → 
  particular_solution x y :=
sorry

end integrate_diff_eq_l28_28776


namespace area_of_region_l28_28641

theorem area_of_region (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (x - 2 * a * y) ^ 2 = 9 * a ^ 2 → (2 * a * x + y) ^ 2 = 4 * a ^ 2) →
  ∀ x y : ℝ, (x - 2 * a * y) ^ 2 = 9 * a ^ 2 → 
    ∀ (hx hy : ℝ), (hx - 2 * a * hy) ^ 2 = 9 * a ^ 2 → (2 * a * hx + hy) ^ 2 = 4 * a ^ 2 →
  let A := (√((hx - 2 * a * hy) ^ 2)) * (√((2 * a * hx + hy) ^ 2)) in
  A = 24 * a ^ 2 / (1 + 4 * a ^ 2) := by
sorry

end area_of_region_l28_28641


namespace line_passes_through_fixed_point_l28_28222

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : x = 6) 
  (h3 : y = 4) : 
  a * x + b * y - 10 = 0 := 
by
  sorry

end line_passes_through_fixed_point_l28_28222


namespace solve_for_x_l28_28000

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l28_28000


namespace nate_matches_left_l28_28438

def initial_matches : ℕ := 70
def matches_dropped : ℕ := 10
def matches_eaten : ℕ := 2 * matches_dropped
def total_matches_lost : ℕ := matches_dropped + matches_eaten
def remaining_matches : ℕ := initial_matches - total_matches_lost

theorem nate_matches_left : remaining_matches = 40 := by
  sorry

end nate_matches_left_l28_28438


namespace simplify_expression_l28_28466

theorem simplify_expression (y : ℝ) :
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l28_28466


namespace students_remaining_after_three_stops_l28_28309

theorem students_remaining_after_three_stops (initial_students : ℕ) (h_initial : initial_students = 48) :
  let after_first_stop := initial_students / 2,
      after_second_stop := after_first_stop / 2,
      after_third_stop := after_second_stop / 2
  in after_third_stop = 6 :=
by
  simp [h_initial]
  sorry

end students_remaining_after_three_stops_l28_28309


namespace son_age_l28_28962

theorem son_age {M S : ℕ} 
  (h1 : M = S + 18) 
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 16 := 
by
  sorry

end son_age_l28_28962


namespace max_area_rect_l28_28127

/--
A rectangle has a perimeter of 40 units and its dimensions are whole numbers.
The maximum possible area of the rectangle is 100 square units.
-/
theorem max_area_rect {l w : ℕ} (hlw : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h_sum : l + w = 20 := by
    rw [two_mul, two_mul, add_assoc, add_assoc] at hlw
    exact Nat.div_eq_of_eq_mul_left (by norm_num : 2 ≠ 0) hlw

  have into_parabola : ∀ l w, l + w = 20 → l * w ≤ 100 := λ l w h_eq =>
  by
    let expr := l * w
    let w_def := 20 - l
    let expr' := l * (20 - l)
    have key_expr: l * w = l * (20 - l) := by
      rw h_eq
    rw key_expr
    let f (l: ℕ) := l * (20 - l)
    have step_expr: l * (20 - l) = 20*l - l^2 := by
      ring

    have boundary : (0 ≤ l * (20 - l)) := mul_nonneg (by apply l.zero_le) (by linarith)
    have max_ex : ((20 / 2)^2 ≤ 100) := by norm_num
    let sq_bound:= 100 - (l - 10)^2
    have complete_sq : 20 * l - l^2 = -(l-10)^2 + 100  := by
      have q_expr: 20 * l - l^2 = - (l-10)^2 + 100 := by linarith
      exact q_expr

    show l * (20 - l) ≤ 100,
    from Nat.le_of_pred_lt (by linarith)


  exact into_parabola l w h_sum

end max_area_rect_l28_28127


namespace arcsine_identity_l28_28080

theorem arcsine_identity (α : ℝ) (h : -1 ≤ 2 * α ∧ 2 * α ≤ 1) 
  (h1 : -1 ≤ 6 * α ∧ 6 * α ≤ 1) : 
  arcsin (2 * α) * arcsin (real.pi / 3 - 2 * α) * arcsin (real.pi / 3 + 2 * α) = 4 * arcsin (6 * α) :=
sorry

end arcsine_identity_l28_28080


namespace income_effects_l28_28704

noncomputable def income_data_ordinary : List ℝ := sorry -- Placeholder for 100 incomes
def income_data_ordinary_max : ℝ := 20000
def jack_ma_income := 100000000000
def median_ordinary := median income_data_ordinary
def mean_ordinary := mean income_data_ordinary
def variance_ordinary := variance income_data_ordinary

noncomputable def income_data_with_jack_ma : List ℝ := income_data_ordinary ++ [jack_ma_income]

def mean_income_with_jack_ma := mean income_data_with_jack_ma
def median_income_with_jack_ma := median income_data_with_jack_ma
def variance_income_with_jack_ma := variance income_data_with_jack_ma

theorem income_effects :
  mean_income_with_jack_ma > mean_ordinary ∧
  (median_income_with_jack_ma = median_ordinary ∨ median_income_with_jack_ma ≠ median_ordinary) ∧
  variance_income_with_jack_ma > variance_ordinary := sorry

end income_effects_l28_28704


namespace ratio_of_volumes_l28_28250

-- Conditions
variables (r : ℝ)
def V_sphere : ℝ := (4 / 3) * π * r^3
def V_cylinder : ℝ := π * r^2 * (2 * r)

-- Mathematical statement to prove
theorem ratio_of_volumes (r : ℝ) (h : r > 0) :
  (V_cylinder r) / (V_sphere r) = 3 / 2 :=
sorry

end ratio_of_volumes_l28_28250


namespace maximize_sector_area_l28_28743

theorem maximize_sector_area :
  (∀ (r l : ℝ), 2 * r + l = 36 ∧ S = 1 / 2 * l * r ∧ α = l / r → α = 2) :=
by
  intros r l h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end maximize_sector_area_l28_28743


namespace intersection_A_B_l28_28277

-- Defining the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Stating the theorem that A ∩ B equals (1, 2)
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l28_28277


namespace composite_quotient_is_one_over_49_l28_28187

/-- Define the first twelve positive composite integers --/
def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def next_six_composites : List ℕ := [14, 15, 16, 18, 20, 21]

/-- Define the product of a list of integers --/
def product (l : List ℕ) : ℕ := l.foldl (λ acc x => acc * x) 1

/-- This defines the quotient of the products of the first six and the next six composite numbers --/
def composite_quotient : ℚ := (↑(product first_six_composites)) / (↑(product next_six_composites))

/-- Main theorem stating that the composite quotient equals to 1/49 --/
theorem composite_quotient_is_one_over_49 : composite_quotient = 1 / 49 := 
  by
    sorry

end composite_quotient_is_one_over_49_l28_28187


namespace irrational_neg_pi_lt_neg_two_l28_28446

theorem irrational_neg_pi_lt_neg_two (h1 : Irrational π) (h2 : π > 2) : Irrational (-π) ∧ -π < -2 := by
  sorry

end irrational_neg_pi_lt_neg_two_l28_28446


namespace function_range_l28_28649

def f (x : ℝ) : ℝ := 3 * (x - 4) * (if x = -8 then 0 else 1)

theorem function_range :
  (∀ y ∈ Set.range f, y ≠ -36) ∧ (∀ y : ℝ, y ≠ -36 → ∃ x : ℝ, f x = y) :=
by
  sorry

end function_range_l28_28649


namespace general_term_formula_l28_28375

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (n : ℕ)
variable (a1 d : ℤ)

-- Given conditions
axiom a2_eq : a 2 = 8
axiom S10_eq : S 10 = 185
axiom S_def : ∀ n, S n = n * (a 1 + a n) / 2
axiom a_def : ∀ n, a (n + 1) = a 1 + n * d

-- Prove the general term formula
theorem general_term_formula : a n = 3 * n + 2 := sorry

end general_term_formula_l28_28375


namespace domino_arrangements_l28_28813

theorem domino_arrangements (grid_rows grid_columns number_of_dominoes : ℕ)
    (domino_squares_per_domino : ℕ) (path_start path_end : ℕ × ℕ)
    (cover_exact_squares touch_at_sides no_diagonal : Prop) 
    (h1 : grid_rows = 6) (h2 : grid_columns = 5)
    (h3 : number_of_dominoes = 6) (h4 : domino_squares_per_domino = 2) 
    (h5 : path_start = (0, 0)) (h6 : path_end = (5, 4))
    (h7 : cover_exact_squares) (h8 : touch_at_sides)
    (h9 : no_diagonal) : 
    ∃ (distinct_arrangements : ℕ), distinct_arrangements = 126 :=
by
  use 126
  sorry

end domino_arrangements_l28_28813


namespace find_f_neg_2_l28_28423

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 3 * x + 4 else 7 - 3 * x

theorem find_f_neg_2 : f (-2) = 13 := by
  sorry

end find_f_neg_2_l28_28423


namespace probability_third_ball_white_l28_28981

theorem probability_third_ball_white :
  let urn : List String := (List.repeat "white" 6) ++ (List.repeat "black" 5) 
  (∀ (draws : List String), draws.length = 3 → ∀ (without_replacement : Bool), 
   without_replacement → probability_white_at_third (urn, draws, without_replacement) = 6 / 11) := 
sorry

def probability_white_at_third (urn_draws: List String × List String × Bool) : ℚ := 
let (urn, draws, without_replacement) := urn_draws 
in calculate_probability urn draws without_replacement

noncomputable def calculate_probability (urn draws: List String) (without_replacement: Bool) : ℚ := 
  if without_replacement ∧ draws.length = 3 then
    if draws.last = "white" then 
      let w := (urn.count "white").toRat
          b := (urn.count "black").toRat
          total := (urn.length).toRat 
      in w / total
    else 0 
  else 0


end probability_third_ball_white_l28_28981


namespace number_of_true_propositions_is_two_l28_28707

def zero_vector_has_no_direction : Prop := false
def zero_vector_equals_zero_vector : Prop := true
def zero_vector_collinear_with_any : Prop := true
def unit_vectors_equal : Prop := false
def collinear_unit_vectors_equal : Prop := false

theorem number_of_true_propositions_is_two :
  (if zero_vector_has_no_direction then 1 else 0) +
  (if zero_vector_equals_zero_vector then 1 else 0) +
  (if zero_vector_collinear_with_any then 1 else 0) +
  (if unit_vectors_equal then 1 else 0) +
  (if collinear_unit_vectors_equal then 1 else 0) = 2 :=
by
  sorry

end number_of_true_propositions_is_two_l28_28707


namespace number_of_quasiperiodic_sums_l28_28403

open Finset

variables {p : ℕ} [Fact p.Prime]

def is_quasiperiodic (f : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b : ℕ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f (x + a) (y + b) = f x y

theorem number_of_quasiperiodic_sums (p : ℕ) [Fact p.Prime] :
  (∃ (f : (ℕ → ℕ → ℕ)), 
    (∀ x y, f x y < p ) ∧ 
     (is_quasiperiodic f)) 
     → 
       (finset.card (Σ f, is_quasiperiodic f) 
         = p ^ ((p^2 + p) / 2)) :=
sorry

end number_of_quasiperiodic_sums_l28_28403


namespace small_bottles_in_storage_l28_28586

theorem small_bottles_in_storage (initial_small_bottles : ℕ) (big_bottles : ℕ) (remaining_bottles : ℕ) :
  big_bottles = 15000 →
  (remaining_bottles = 18180) →
  0.88 * initial_small_bottles + 0.86 * big_bottles = remaining_bottles →
  initial_small_bottles = 6000 :=
by
  intros h1 h2 h3
  sorry

end small_bottles_in_storage_l28_28586


namespace G_at_8_l28_28794

noncomputable def G (x : ℝ) : ℝ := sorry

theorem G_at_8 :
  (G 4 = 8) →
  (∀ x : ℝ, (x^2 + 3 * x + 2 ≠ 0) →
    G (2 * x) / G (x + 2) = 4 - (16 * x + 8) / (x^2 + 3 * x + 2)) →
  G 8 = 112 / 3 :=
by
  intros h1 h2
  sorry

end G_at_8_l28_28794


namespace balls_placed_in_boxes_l28_28445

theorem balls_placed_in_boxes :
  ∃ n : ℕ, n = 10 ∧ (∀ (balls boxes : ℕ), balls = 6 ∧ boxes = 3 →
    let ways := (choose (balls - 1) (boxes - 1))
    ways = n) :=
sorry

end balls_placed_in_boxes_l28_28445


namespace k_divides_99_l28_28571

-- Define what it means for a number to reverse its digits
def reverse_digits (n : ℕ) : ℕ :=
  n.digits.reverse.foldl (λ acc d, acc * 10 + d) 0

-- The main statement to prove:
theorem k_divides_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
by
  sorry

end k_divides_99_l28_28571


namespace find_min_n_l28_28702

variable (a : Nat → Int)
variable (S : Nat → Int)
variable (d : Nat)
variable (n : Nat)

-- Definitions based on given conditions
def arithmetic_sequence (a : Nat → Int) (d : Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a1_eq_neg3 (a : Nat → Int) : Prop :=
  a 1 = -3

def condition (a : Nat → Int) (d : Nat) : Prop :=
  11 * a 5 = 5 * a 8

-- Correct answer condition
def minimized_sum_condition (a : Nat → Int) (S : Nat → Int) (d : Nat) (n : Nat) : Prop :=
  S n ≤ S (n + 1)

theorem find_min_n (a : Nat → Int) (S : Nat → Int) (d : Nat) :
  arithmetic_sequence a d ->
  a1_eq_neg3 a ->
  condition a 2 ->
  minimized_sum_condition a S 2 2 :=
by
  sorry

end find_min_n_l28_28702


namespace bill_sunday_miles_l28_28440

variables (B J M S : ℝ)

-- Conditions
def condition_1 := B + 4
def condition_2 := 2 * (B + 4)
def condition_3 := J = 0 ∧ M = 5 ∧ (M + 2 = 7)
def condition_4 := (B + 5) + (B + 4) + 2 * (B + 4) + 7 = 50

-- The main theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (h1 : S = B + 4) (h2 : ∀ B, J = 0 → M = 5 → S + 2 = 7 → (B + 5) + S + 2 * S + 7 = 50) : S = 10.5 :=
by {
  sorry
}

end bill_sunday_miles_l28_28440


namespace pow_increasing_log_increasing_l28_28998

open Real
open Nat

-- Given the conditions
variable {a b c d : ℝ}

-- Definitions
def is_increasing (f : ℝ → ℝ) : Prop := ∀ (x y : ℝ), x < y → f x < f y

theorem pow_increasing (a b : ℝ) (h : 2 > 1) (h₀ : 0.6 > 0.5) :
  2 ^ 0.6 > 2 ^ 0.5 :=
by {
  -- Statement only (proof omitted)
  sorry
}

theorem log_increasing (a b : ℝ) (h : 2 > 1) (h₀ : 3.4 < 3.8) :
  log 2 3.4 < log 2 3.8 :=
by {
  -- Statement only (proof omitted)
  sorry
}

end pow_increasing_log_increasing_l28_28998


namespace problem_comparison_l28_28797

-- Definitions based on conditions
def y₁ : ℝ := 4 ^ 0.9
def y₂ : ℝ := 8 ^ 0.44
def y₃ : ℝ := (1 / 2) ^ (-1.5)

-- Theorem statement based on the problem
theorem problem_comparison : y₁ > y₃ ∧ y₃ > y₂ :=
by
  -- The proof is omitted
  sorry

end problem_comparison_l28_28797


namespace find_c_l28_28025

variables {R : Type*} [LinearOrderedField R]
variables (u v : EuclideanSpace R 2) (c : R)

theorem find_c (c : R)
  (h1 : u = ![5, c])
  (h2 : v = ![-3, 2])
  (h3 : ∥v∥^2 = 13)
  (h4 : ⟪u, v⟫ / 13 * v = 5 • v) :
  c = 40 :=
by sorry

end find_c_l28_28025


namespace kaleb_ate_cherries_l28_28400

theorem kaleb_ate_cherries (original_cherries remaining_cherries eaten_cherries : ℕ) 
  (h1 : original_cherries = 67) (h2 : remaining_cherries = 42) :
  eaten_cherries = original_cherries - remaining_cherries → eaten_cherries = 25 :=
by
  intros
  rw [h1, h2]
  sorry

end kaleb_ate_cherries_l28_28400


namespace sum_of_all_distinct_x_for_g_g_g_x_eq_neg1_l28_28712

def g (x : ℝ) : ℝ := x^2 / 4 - x - 2

theorem sum_of_all_distinct_x_for_g_g_g_x_eq_neg1 : 
  (∑ x in {x | g(g(g(x))) = -1}.to_finset, x) = -5 := by
  sorry

end sum_of_all_distinct_x_for_g_g_g_x_eq_neg1_l28_28712


namespace root_exists_in_interval_l28_28075

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_exists_in_interval : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := 
by
  sorry

end root_exists_in_interval_l28_28075


namespace correct_function_l28_28151

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

def is_monotonically_increasing_on_pos_inf (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f(x) ≤ f(y)

def funcA : ℝ → ℝ := λ x, - (1 / x)
def funcB : ℝ → ℝ := λ x, -x^2
def funcC : ℝ → ℝ := λ x, exp(-x) + exp(x)
def funcD : ℝ → ℝ := λ x, |x + 1|

theorem correct_function :
  is_even_function funcC ∧ is_monotonically_increasing_on_pos_inf funcC ∧
  (¬ is_even_function funcA ∨ ¬ is_monotonically_increasing_on_pos_inf funcA) ∧
  (¬ is_even_function funcB ∨ ¬ is_monotonically_increasing_on_pos_inf funcB) ∧
  (¬ is_even_function funcD ∨ ¬ is_monotonically_increasing_on_pos_inf funcD) :=
by
  sorry

end correct_function_l28_28151


namespace can_increase_averages_by_transfer_l28_28340

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ := grades.sum / grades.length

theorem can_increase_averages_by_transfer :
    ∃ (student : ℝ) (new_group1_grades new_group2_grades : List ℝ),
      student ∈ group1_grades ∧
      new_group1_grades = (group1_grades.erase student) ∧
      new_group2_grades = student :: group2_grades ∧
      average new_group1_grades > average group1_grades ∧ 
      average new_group2_grades > average group2_grades :=
by
  sorry

end can_increase_averages_by_transfer_l28_28340


namespace transfer_student_increases_averages_l28_28345

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ :=
  if grades.length > 0 then (grades.sum / grades.length) else 0

def can_transfer_to_increase_averages (group1_grades group2_grades : List ℝ) : Prop :=
  ∃ x ∈ group1_grades, average (x :: group2_grades) > average group2_grades ∧
                       average (group1_grades.erase x) > average group1_grades

theorem transfer_student_increases_averages :
  can_transfer_to_increase_averages group1_grades group2_grades :=
by {
  sorry -- The proof is skipped as per instructions
}

end transfer_student_increases_averages_l28_28345


namespace combined_cost_price_l28_28588

theorem combined_cost_price (CP_A CP_B CP_C : ℝ) 
  (hA: (120 - CP_A) = (CP_A - 80))
  (hB: 150 - CP_B = 0.25 * CP_B)
  (hC: (CP_C - 200) = 0.20 * CP_C) :
  CP_A + CP_B + CP_C = 470 :=
by
  have h1 : CP_A = 100, from sorry,
  have h2 : CP_B = 120, from sorry,
  have h3 : CP_C = 250, from sorry,
  rw [h1, h2, h3],
  norm_num,
  exact dec_trivial

end combined_cost_price_l28_28588


namespace jade_transactions_more_than_cal_l28_28821

theorem jade_transactions_more_than_cal :
  let Mabel := 90 in
  let Anthony := Mabel + 10 / 100 * Mabel in
  let Cal := 2 / 3 * Anthony in
  let Jade := 80 in
  Jade - Cal = 14 := by
{
  sorry
}

end jade_transactions_more_than_cal_l28_28821


namespace find_a_l28_28730

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 5) (h3 : c = 3) : a = 1 := by
  sorry

end find_a_l28_28730


namespace slope_of_tangent_at_pi_over_4_l28_28508

noncomputable def f (x : ℝ) : ℝ := (sin x) / (sin x + cos x) - (1 / 2)

theorem slope_of_tangent_at_pi_over_4 :
  deriv f (π / 4) = 1 / 2 :=
sorry

end slope_of_tangent_at_pi_over_4_l28_28508


namespace expected_value_of_winning_is_2550_l28_28566

-- Definitions based on the conditions
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℚ := 1 / 8
def winnings (n : ℕ) : ℕ := n^2

-- Expected value calculation based on the conditions
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ n => probability n * winnings n)).sum

-- Proposition stating that the expected value is 25.50
theorem expected_value_of_winning_is_2550 : expected_value = 25.50 :=
by
  sorry

end expected_value_of_winning_is_2550_l28_28566


namespace distinct_ordered_pairs_count_l28_28842

theorem distinct_ordered_pairs_count (n_people : ℕ) (w m : ℕ) (w_condition : w ≥ 0) (m_condition : m ≥ 0) :
  n_people = 7 → (∀ (w, m) ∈ {(0, 7), (2, 7), (4, 7), (5, 7), (6, 7)}, (w, m) ∈ {((0, 7) : ℕ × ℕ), ((2, 7) : ℕ × ℕ), ((4, 7): ℕ × ℕ), ((5, 7) : ℕ × ℕ), ((6, 7) : ℕ × ℕ)}) →
  ∃ (pairs : set (ℕ × ℕ)), pairs = {(0, 7), (2, 7), (4, 7), (5, 7), (6, 7)} ∧ pairs.card = 5 :=
by {
  sorry
}

end distinct_ordered_pairs_count_l28_28842


namespace prob_abs_diff_gt_one_third_l28_28965

noncomputable theory

-- Definitions corresponding to given conditions
def coin_flip_outcome : Type := ℕ
def coin_flip : ℕ → coin_flip_outcome := λ _, if (rand.uniform 0 1) < 0.5 then 0 else 1
def generate_number : list coin_flip_outcome → ℝ
| [0, 0, 0]        := 0
| [1, 1, 1]        := 1
| _                := rand.uniform 0 1

def choose_number : ℝ :=
  let results := list.ret <| repeat (coin_flip 3) 3 in
  generate_number results

-- Theorem statement
theorem prob_abs_diff_gt_one_third :
  let x := choose_number
  let y := choose_number
  P (|x - y| > 1 / 3) = 3 / 32 :=
sorry

end prob_abs_diff_gt_one_third_l28_28965


namespace a4_minus_1_divisible_5_l28_28448

theorem a4_minus_1_divisible_5 (a : ℤ) (h : ¬ (∃ k : ℤ, a = 5 * k)) : 
  (a^4 - 1) % 5 = 0 :=
by
  sorry

end a4_minus_1_divisible_5_l28_28448


namespace toy_box_cost_l28_28783

theorem toy_box_cost
  (puppy : ℝ)
  (dog_food : ℝ)
  (treats_unit_cost : ℝ)
  (num_treats : ℝ)
  (crate : ℝ)
  (bed : ℝ)
  (collar_leash : ℝ)
  (discount_rate : ℝ)
  (total_spent_after_discount : ℝ)
  (unknown_toy_cost : ℝ)
  (total_spent_before_discount: ℝ) 
  (subtotal : ℝ):
  puppy + dog_food + num_treats * treats_unit_cost + crate + bed + collar_leash + unknown_toy_cost == 120 →
  subtotal = puppy + dog_food + num_treats * treats_unit_cost + crate + bed + collar_leash →
  total_spent_after_discount == total_spent_before_discount * (1 - discount_rate) →
  total_spent_before_discount == subtotal + unknown_toy_cost →
  subtotal == 80.00 →
  discount_rate == 0.20 →
  total_spent_after_discount == 96.00 →
  unknown_toy_cost == 40.00 :=
begin 
  intros,
  sorry,
end

end toy_box_cost_l28_28783


namespace maximum_ab_l28_28305

theorem maximum_ab (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3 * a + 2 * b - c = 0) : 
  ab <= 1/3 := 
by 
  sorry

end maximum_ab_l28_28305


namespace maximize_sector_area_l28_28742

noncomputable def max_area_sector_angle (r : ℝ) (l := 36 - 2 * r) (α := l / r) : ℝ :=
  α

theorem maximize_sector_area (h : ∀ r : ℝ, 2 * r + 36 - 2 * r = 36) :
  max_area_sector_angle 9 = 2 :=
by
  sorry

end maximize_sector_area_l28_28742


namespace arith_seq_contradiction_decreasing_geometric_prop_find_general_term_l28_28231

-- Condition Definitions
def sequence (a : ℕ → ℝ) : Prop :=
  | ∀ n : ℕ, n > 0 → abs (a n.succ - a n) = (2:ℝ)^n
def increasing (a : ℕ → ℝ) : Prop :=
  | ∀ n : ℕ, n > 0 → a n.succ > a n
def decreasing_geometric (a : ℕ → ℝ) : Prop :=
  | ∃ q : ℝ, 0 < q ∧ q < 1 ∧ (∀ n : ℕ, n > 0 → a n = q^(n-1))

-- (I) Statement
theorem arith_seq_contradiction (a : ℕ → ℝ) (p : ℝ) (h_seq : sequence a) (h_incr : increasing a) (hp : p ≠ 1) : ¬ ∃ d : ℝ, ∀ n : ℕ, a n = a 0 + n * d :=
sorry

-- (II) Statement
theorem decreasing_geometric_prop (a : ℕ → ℝ) (m : ℕ) (h_seq : sequence a) (h_decr : decreasing_geometric a) : ∀ n : ℕ, n > 0 → a n > (finset.sum (finset.range m) (λ k, a (n + k.succ))) :=
sorry

-- (III) Statement
noncomputable def general_term_formula (n : ℕ) (p := 2) : ℝ :=
  if (odd n) then (1 + 2 * 4^((n-1)/2) / 3) else (13 - 4^(n/2) / 3)

theorem find_general_term (a : ℕ → ℝ) (h_seq : sequence a) (h_incr_odd : ∀ n : ℕ, odd (2 * n - 1) ∧ increasing (λ n, a (2 * n - 1))) (h_decr_even : ∀ n : ℕ, even (2 * n) ∧ decreasing (λ n, a (2 * n))) (p_eq_2 : p = 2) : ∀ n : ℕ, a n = general_term_formula n :=
sorry

end arith_seq_contradiction_decreasing_geometric_prop_find_general_term_l28_28231


namespace pineapple_total_cost_correct_l28_28078

-- Define the conditions
def pineapple_cost : ℝ := 1.25
def num_pineapples : ℕ := 12
def shipping_cost : ℝ := 21.00

-- Calculate total cost
noncomputable def total_pineapple_cost : ℝ := pineapple_cost * num_pineapples
noncomputable def total_cost : ℝ := total_pineapple_cost + shipping_cost
noncomputable def cost_per_pineapple : ℝ := total_cost / num_pineapples

-- The proof problem
theorem pineapple_total_cost_correct : cost_per_pineapple = 3 := by
  -- The proof will be filled in here
  sorry

end pineapple_total_cost_correct_l28_28078


namespace simplify_expression_l28_28849

theorem simplify_expression (x : ℝ) : 
  sin (x + real.pi / 3) + 2 * sin (x - real.pi / 3) - sqrt 3 * cos (2 * real.pi / 3 - x) = 0 :=
by
  sorry

end simplify_expression_l28_28849


namespace quadrilateral_not_necessarily_parallelogram_l28_28789

structure Quadrilateral (α : Type) [Add α] :=
  (A B C D : α)

structure MidPoints (α : Type) :=
  (K L M N : α)

def areMidpoints {α : Type} [Field α] [Add α] [Mul α] [Sub α] [Div α] [One α] 
  (A B C D K L M N : α) :=
  (K = (B + C) / 2) ∧ (L = (C + D) / 2) ∧ (M = (D + A) / 2) ∧ (N = (A + B) / 2)

theorem quadrilateral_not_necessarily_parallelogram {α : Type} [Field α] [Add α] [Mul α] [Sub α] [Div α] [One α]
  (A B C D K L M N : α) (H1 : areMidpoints A B C D K L M N) 
  (H2 : ∃ λ : α, 
    ∀ (X Y Z W : α), 
    (X = (A + K) / 3 ∧ Y = (B + L) / 3 ∧ Z = (C + M) / 3 ∧ W = (D + N) / 3) → 
    (λ = (X - W) / (A - K) ∧ λ = (Y - Z) / (B - L) ∧ λ = (Z - Y) / (C - M) ∧ λ = (W - X) / (D - N))) :
  ¬ (A + C = B + D ∧ A + D = B + C) :=
  sorry

end quadrilateral_not_necessarily_parallelogram_l28_28789


namespace sum_of_squares_eq_eight_l28_28664

theorem sum_of_squares_eq_eight :
  (∑ x in ({y : ℝ | y ^ 64 = 64 ^ 16 ∧ y ^ 2}.to_finset), x) = 8 :=
by
  sorry

end sum_of_squares_eq_eight_l28_28664


namespace simplest_radical_x_eq_11_l28_28303

noncomputable def isSimplestQuadraticRadical (r : ℝ) : Prop :=
  ∀ (a b : ℕ), a > 1 → b > 1 → r ≠ a * (b : ℝ)^(1/2)

theorem simplest_radical_x_eq_11 (x : ℕ) :
  (isSimplestQuadraticRadical (sqrt (x - 5)) → x = 11) := by
sorry

end simplest_radical_x_eq_11_l28_28303


namespace convert_vector_eq_to_slope_intercept_form_l28_28684

noncomputable def vector_eq_to_slope_intercept (x y : ℝ) : ℝ × ℝ :=
  let eq := λ (x y : ℝ),  2 * (x - 4) - (y + 6) = 0 in
  if eq x y then (2, -14) else (0, 0) -- Fallback pair if condition is not met

theorem convert_vector_eq_to_slope_intercept_form :
  ∀ (x y : ℝ), (2 * (x - 4) - (y + 6) = 0) → (∃ (m b : ℝ), (m, b) = (2, -14)) :=
by
  intros x y h
  use (2, -14)
  sorry

end convert_vector_eq_to_slope_intercept_form_l28_28684


namespace percentage_increase_l28_28567

/-- Given:
    * retailer_price = 1.40C
    * customer_price = 1.6100000000000001C
    Prove that the percentage increase from the retailer's price to the customer's price is 15%.
-/
theorem percentage_increase (C : ℝ) (h₁ : retailer_price = 1.40 * C) (h₂ : customer_price = 1.6100000000000001 * C) :
  let percentage_increase := ((customer_price - retailer_price) / retailer_price) * 100 in
  percentage_increase = 15 :=
by
  sorry

end percentage_increase_l28_28567


namespace linear_system_k_value_l28_28278

theorem linear_system_k_value (x y k : ℝ) (h1 : x + 3 * y = 2 * k + 1) (h2 : x - y = 1) (h3 : x = -y) : k = -1 :=
sorry

end linear_system_k_value_l28_28278


namespace tan_x_over_tan_y_plus_tan_y_over_tan_x_l28_28415

open Real

theorem tan_x_over_tan_y_plus_tan_y_over_tan_x (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 5) :
  tan x / tan y + tan y / tan x = 10 := 
by
  sorry

end tan_x_over_tan_y_plus_tan_y_over_tan_x_l28_28415


namespace remainder_division_Q_x_10_x_50_l28_28408

-- Define the conditions
variables {Q : ℝ → ℝ}
hypothesis h1 : polynomial Q
hypothesis h2 : Q 10 = 5
hypothesis h3 : Q 50 = 15

-- Define the target statement to prove
theorem remainder_division_Q_x_10_x_50 : 
  ∃ (a b : ℝ), (Q = λ x, (x - 10) * (x - 50) * (R x) + a * x + b) ∧
  (a = 1/4 ∧ b = 2.5) :=
sorry

end remainder_division_Q_x_10_x_50_l28_28408


namespace smallest_positive_integer_l28_28911

theorem smallest_positive_integer (n : ℕ) (h : 629 * n ≡ 1181 * n [MOD 35]) : n = 35 :=
sorry

end smallest_positive_integer_l28_28911


namespace num_valid_digits_l28_28670

def is_divisible (a b : ℕ) : Prop := b % a = 0

def valid_digit (A : ℕ) : Prop :=
  A ≤ 9 ∧
  is_divisible A 72 ∧
  (let last_two_digits := 10 * A + 2 in is_divisible 4 last_two_digits)

-- The main theorem statement we need to prove
theorem num_valid_digits : {A : ℕ | valid_digit A}.to_finset.card = 3 :=
  by sorry

end num_valid_digits_l28_28670


namespace functions_satisfy_inequality_l28_28655

variable {ℝ : Type} [LinearOrderedField ℝ]

def p (x : ℝ) := sorry
def q (x : ℝ) := sorry

def f (x : ℝ) := 1 / 2 * (sin x + cos x + p x - q x)
def g (x : ℝ) := 1 / 2 * (sin x - cos x + p x + q x)

theorem functions_satisfy_inequality (f g : ℝ → ℝ) (p q : ℝ → ℝ) :
  (∀ x y : ℝ, f x + f y + g x - g y ≥ sin x + cos y) ↔
  (∀ x y : ℝ, p x ≥ q y) → 
  (f x = 1 / 2 * (sin x + cos x + p x - q x) ∧ g x = 1 / 2 * (sin x - cos x + p x + q x)) := sorry

end functions_satisfy_inequality_l28_28655


namespace domain_of_u_l28_28531

noncomputable def u (x : ℝ) : ℝ := 1 / Real.sqrt x 

theorem domain_of_u : {x : ℝ | 0 < x} = set.Ioi 0 :=
by
  sorry

end domain_of_u_l28_28531


namespace radius_of_inscribed_circle_l28_28060

-- Define the triangle side lengths
def DE : ℝ := 8
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- The statement to be proved
theorem radius_of_inscribed_circle : r = (5 * Real.sqrt 39) / 13 := by
  sorry

end radius_of_inscribed_circle_l28_28060


namespace min_value_of_squares_l28_28216

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end min_value_of_squares_l28_28216


namespace alan_total_spending_l28_28596

-- Define the conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation
def cost_eggs : ℕ := eggs_bought * price_per_egg
def cost_chickens : ℕ := chickens_bought * price_per_chicken
def total_amount_spent : ℕ := cost_eggs + cost_chickens

-- Prove the total amount spent
theorem alan_total_spending : total_amount_spent = 88 := by
  show cost_eggs + cost_chickens = 88
  sorry

end alan_total_spending_l28_28596


namespace nate_matches_final_count_l28_28435

theorem nate_matches_final_count :
  ∀ (initial: ℕ) (dropped: ℕ),
    initial = 70 →
    dropped = 10 →
    ∃ final: ℕ, final = initial - dropped - 2*dropped ∧ final = 40 :=
by
  intros initial dropped h_initial h_dropped
  use initial - dropped - 2*dropped
  have h_final_eq := calc
    initial - dropped - 2*dropped = 70 - 10 - 2*10 : by rw [h_initial, h_dropped]
    ... = 40 : by norm_num
  exact ⟨h_final_eq, h_final_eq.symm⟩

end nate_matches_final_count_l28_28435


namespace simplify_expression_l28_28017

theorem simplify_expression :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 :=
by
  sorry

end simplify_expression_l28_28017


namespace tan_interval_of_increase_l28_28174

noncomputable def interval_of_monotonic_increase (k : ℤ) : set ℝ :=
{kπ - π/4 < x < kπ + 3π/4 | x ∈ ℝ}

theorem tan_interval_of_increase {k : ℤ} :
  ∀ x : ℝ, x ∈ interval_of_monotonic_increase k ↔ (kπ - π/4 < x ∧ x < kπ + 3π/4) :=
by
  sorry

end tan_interval_of_increase_l28_28174


namespace problem1_solution_problem2_solution_l28_28549

-- Define the terms involved in Problem 1
def sqrt_sixteen : ℝ := real.sqrt 16
def cuberoot_one_twentyfive : ℝ := real.cbrt 125
def abs_sqrt3_minus_two : ℝ := abs (real.sqrt 3 - 2)

-- Define the final expression for Problem (1)
def problem1_expr : ℝ := sqrt_sixteen - cuberoot_one_twentyfive + abs_sqrt3_minus_two

-- Problem (1) statement: prove the expression equals 1 - sqrt(3)
theorem problem1_solution : problem1_expr = 1 - real.sqrt 3 := 
by sorry

-- Define the given equation for Problem (2)
def given_eqn (x : ℝ) : Prop := (x - 1)^3 + 27 = 0

-- Problem (2) statement: prove the solution to the equation
theorem problem2_solution : ∃ x, given_eqn x ∧ x = -2 :=
by sorry

end problem1_solution_problem2_solution_l28_28549


namespace mountain_climbers_average_speed_l28_28039

def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

noncomputable def average_climb_time (elizabeth_time : ℝ) (tom_time : ℝ) : ℝ :=
  (elizabeth_time + tom_time) / 2

noncomputable def average_speed (total_distance : ℝ) (average_time_hours : ℝ) : ℝ :=
  total_distance / average_time_hours

theorem mountain_climbers_average_speed :
  let elizabeth_rocky : ℝ := 30
  let elizabeth_forest : ℝ := 45
  let elizabeth_snowy : ℝ := 60
  let tom_rocky := 4 * elizabeth_rocky
  let tom_forest := 4 * elizabeth_forest
  let tom_snowy := 4 * elizabeth_snowy
  let total_distance : ℝ := 15
  let elizabeth_total := elizabeth_rocky + elizabeth_forest + elizabeth_snowy
  let tom_total := tom_rocky + tom_forest + tom_snowy
  let total_average_minutes := average_climb_time elizabeth_total tom_total
  let total_average_hours := minutes_to_hours total_average_minutes
  average_speed total_distance total_average_hours = 2.67 :=
begin
  let elizabeth_rocky : ℝ := 30,
  let elizabeth_forest : ℝ := 45,
  let elizabeth_snowy : ℝ := 60,
  let tom_rocky := 4 * elizabeth_rocky,
  let tom_forest := 4 * elizabeth_forest,
  let tom_snowy := 4 * elizabeth_snowy,
  let total_distance : ℝ := 15,
  let elizabeth_total := elizabeth_rocky + elizabeth_forest + elizabeth_snowy,
  let tom_total := tom_rocky + tom_forest + tom_snowy,
  let total_average_minutes := average_climb_time elizabeth_total tom_total,
  let total_average_hours := minutes_to_hours total_average_minutes,
  show average_speed total_distance total_average_hours = 2.67, from sorry
end

end mountain_climbers_average_speed_l28_28039


namespace find_solution_l28_28656

-- Definitions for the problem
def is_solution (x y z t : ℕ) : Prop := (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ (2^y + 2^z * 5^t - 5^x = 1))

-- Statement of the theorem
theorem find_solution : ∀ x y z t : ℕ, is_solution x y z t → (x, y, z, t) = (2, 4, 1, 1) := by
  sorry

end find_solution_l28_28656


namespace sum_of_first_fifteen_multiples_of_11_excluding_multiples_of_5_l28_28538

theorem sum_of_first_fifteen_multiples_of_11_excluding_multiples_of_5 :
  let multiples_11_sum := 11 * (15 * 16 / 2) in
  let multiples_5_11_sum := 55 + 110 + 165 in
  multiples_11_sum - multiples_5_11_sum = 990 := sorry

end sum_of_first_fifteen_multiples_of_11_excluding_multiples_of_5_l28_28538


namespace hyperbola_eccentricity_proof_l28_28490

noncomputable  -- needed because we will use square roots

def hyperbola_eccentricity (a b : ℝ) (ha : a = 2) (hb : b = sqrt 5) : Prop :=
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = 3 / 2

theorem hyperbola_eccentricity_proof : hyperbola_eccentricity 2 (sqrt 5)
    (by rfl) (by rfl) :=
  sorry

end hyperbola_eccentricity_proof_l28_28490


namespace smallest_number_of_rectangles_needed_l28_28536

-- Define the dimensions of the rectangle
def rectangle_area (length width : ℕ) : ℕ := length * width

-- Define the side length of the square
def square_side_length : ℕ := 12

-- Define the number of rectangles needed to cover the square horizontally
def num_rectangles_to_cover_square : ℕ := (square_side_length / 3) * (square_side_length / 4)

-- The theorem must state the total number of rectangles required
theorem smallest_number_of_rectangles_needed : num_rectangles_to_cover_square = 16 := 
by
  -- Proof details are skipped using sorry
  sorry

end smallest_number_of_rectangles_needed_l28_28536


namespace none_of_these_l28_28118

theorem none_of_these (a T : ℝ) : 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y - 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y - 4 * a * T = 0) :=
sorry

end none_of_these_l28_28118


namespace area_ratio_of_circles_l28_28886

variable (O P X : Type)

-- Define circle radii and areas
def circle_radius (OX OP : ℝ) : ℝ :=
  OX / OP

theorem area_ratio_of_circles
  (OX OP : ℝ)
  (h1 : 2 * OX = OP) :
  ((π * OX ^ 2) / (π * OP ^ 2)) = (1 / 4) :=
by
  have h2 : OX = 1 / 2 * OP := by linarith
  sorry

end area_ratio_of_circles_l28_28886


namespace house_colors_revert_to_initial_state_l28_28885

theorem house_colors_revert_to_initial_state :
  ∀ (initial_colors : Fin 12 → Bool), 
    (∃ i : Fin 12, initial_colors i = true) →
    (∀ (painter : Fin 12) (current_colors : Fin 12 → Bool) (i : Fin 12), 
      current_colors i = if i.val < ((painter.val + 1) % 12) 
                        then not (initial_colors ((i + painter) % 12)) else initial_colors ((i + painter) % 12)
    →
    (current_colors 0 % 12).val = initial_colors (0 % 12).val) :=
begin
  sorry
end

end house_colors_revert_to_initial_state_l28_28885


namespace problem_statement_l28_28411

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ∈ set.Ico 0 1 then real.log (0.5) (1 - x) 
  else if x ∈ set.Ico (-1) 0 then real.log (0.5) (1 + x)
  else if ∃ k : ℤ, x = k + x' ∧ (x' ∈ set.Ico 0 1 ∨ x' ∈ set.Ico (-1) 0) 
    then let ⟨k, x', hx⟩ := exists_el_of_mem_set_Ioo.elim_right (set.Icc.eventually_mem_Ioo hx.2) in f x'
  else f x
  
theorem problem_statement :
  (∀ x : ℝ, f(x + 1) = f(x - 1)) ∧
  (∀ x : ℝ, f(-x) = f(x)) ∧
  (∀ x : ℝ, x ∈ set.Ico 0 1 → f(x) = real.log (0.5) (1 - x)) →
  (periodic f 2) ∧
  (∀ x ∈ (set.Icc 1 2 \ {2}).1, (differentiable_at ℝ f x) ∧ f' x > 0) = false ∧
  (∀ x ∈ (set.Icc 2 3 \ {3}).1, (differentiable_at ℝ f x) ∧ f' x < 0) = false ∧
  (∀ x ∈ set.Icc 3 4, f(x) = real.log (0.5) (x - 3)) := sorry

end problem_statement_l28_28411


namespace sum_ineq_l28_28793

theorem sum_ineq (n : ℕ) (h : n ≥ 2) (a : ℕ → ℝ)
  (hpos : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0)
  (hsum : (Finset.range n).sum a = 1) :
  (Finset.range n).sum (λ k, (a (k + 1) / (1 - a (k + 1))) *
    ((Finset.range k).sum a)^2) < 1 / 3 :=
sorry

end sum_ineq_l28_28793


namespace problem_1_problem_2_l28_28092

open Finset

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem problem_1 (n : ℕ) :
  ∑ k in range (n + 1), (-1) ^ k * (binom n k : ℤ) * ((x - k : ℕ) ^ n) = n.fact :=
sorry

theorem problem_2 (x : ℝ) (m n : ℕ) (h : m ≤ n) :
  ∑ k in range (n + 2), (-1) ^ k * (binom (n + 1) k : ℤ) * ((x - k : ℝ) ^ m) = 0 :=
sorry

end problem_1_problem_2_l28_28092


namespace ratio_of_original_to_doubled_l28_28123

theorem ratio_of_original_to_doubled (x : ℕ) (h : x + 5 = 17) : (x / Nat.gcd x (2 * x)) = 1 ∧ ((2 * x) / Nat.gcd x (2 * x)) = 2 := 
by
  sorry

end ratio_of_original_to_doubled_l28_28123


namespace no_prime_sum_seventeen_l28_28726

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_sum_seventeen :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 17 := by
  sorry

end no_prime_sum_seventeen_l28_28726


namespace range_of_a_l28_28420

theorem range_of_a (x y a : ℝ) :
  (2 * x + y ≥ 4) → 
  (x - y ≥ 1) → 
  (x - 2 * y ≤ 2) → 
  (x = 2) → 
  (y = 0) → 
  (z = a * x + y) → 
  (Ax = 2) → 
  (Ay = 0) → 
  (-1/2 < a ∧ a < 2) := sorry

end range_of_a_l28_28420


namespace cyclic_quad_harmonic_set_collinear_l28_28804

open Set

/-- Let ABCD be a cyclic quadrilateral, E = AD ∩ BC with C between B and E, 
 and the diagonals intersect at F. Consider the midpoint M of [CD] and N on 
 the circumcircle of ABM such that AN / BN = AM / CM. 
 Show that E, F, and N are collinear. -/
theorem cyclic_quad_harmonic_set_collinear
  {A B C D E F M N : Point}
  (h1: is_cyclic_quad ABCD)
  (h2: E = AD ∩ BC)
  (h3: C_between_B_and_E : betw C B E)
  (h4: F = AC ∩ BD)
  (h5: M_midpoint_CD : M = midpoint C D)
  (h6: N_on_circum_ABM : on_circumcircle ABM N)
  (h7: harmonic_set_ratio : AN / BN = AM / CM) :
  collinear E F N :=
  sorry

end cyclic_quad_harmonic_set_collinear_l28_28804


namespace calculate_sum_l28_28990

theorem calculate_sum : (-2) + 1 = -1 :=
by 
  sorry

end calculate_sum_l28_28990


namespace calculate_expression_l28_28616

theorem calculate_expression :
  (pi - 2023)^0 + abs (1 - sqrt 3) + sqrt 8 - real.tan (real.pi / 3) = 2 * sqrt 2 :=
by
  sorry

end calculate_expression_l28_28616


namespace largest_circle_area_l28_28578

noncomputable def side_length_of_square (area : ℕ) : ℝ :=
  real.sqrt area

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

def radius_of_circle (circumference : ℝ) : ℝ :=
  circumference / (2 * real.pi)

def area_of_circle (radius : ℝ) : ℝ :=
  real.pi * radius ^ 2

theorem largest_circle_area (square_area : ℕ) :
  let side_length := side_length_of_square square_area,
      perimeter := perimeter_of_square side_length,
      radius := radius_of_circle perimeter,
      circle_area := area_of_circle radius
  in round circle_area = 287 :=
by
  sorry

end largest_circle_area_l28_28578


namespace solid_has_identical_views_is_sphere_or_cube_l28_28587

-- Define the conditions for orthographic projections being identical
def identical_views_in_orthographic_projections (solid : Type) : Prop :=
  sorry -- Assume the logic for checking identical orthographic projections is defined

-- Define the types for sphere and cube
structure Sphere : Type := 
  (radius : ℝ)

structure Cube : Type := 
  (side_length : ℝ)

-- The main statement to prove
theorem solid_has_identical_views_is_sphere_or_cube (solid : Type) 
  (h : identical_views_in_orthographic_projections solid) : 
  solid = Sphere ∨ solid = Cube :=
by 
  sorry -- The detailed proof is omitted

end solid_has_identical_views_is_sphere_or_cube_l28_28587


namespace smallest_possible_area_l28_28524

noncomputable def smallest_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) then l * w else 0

theorem smallest_possible_area : ∃ l w : ℕ, 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) ∧ smallest_area l w = 2100 := by
  sorry

end smallest_possible_area_l28_28524


namespace pascal_triangle_row10_sum_l28_28330

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2 ^ n

theorem pascal_triangle_row10_sum : pascal_triangle_row_sum 10 = 1024 :=
by
  -- Proof will demonstrate that 2^10 = 1024
  sorry

end pascal_triangle_row10_sum_l28_28330


namespace smaller_angle_clock_1245_l28_28532

theorem smaller_angle_clock_1245 
  (minute_rate : ℕ → ℝ) 
  (hour_rate : ℕ → ℝ) 
  (time : ℕ) 
  (minute_angle : ℝ) 
  (hour_angle : ℝ) 
  (larger_angle : ℝ) 
  (smaller_angle : ℝ) :
  (minute_rate 1 = 6) →
  (hour_rate 1 = 0.5) →
  (time = 45) →
  (minute_angle = minute_rate 45 * 45) →
  (hour_angle = hour_rate 45 * 45) →
  (larger_angle = |minute_angle - hour_angle|) →
  (smaller_angle = 360 - larger_angle) →
  smaller_angle = 112.5 :=
by
  intros
  sorry

end smaller_angle_clock_1245_l28_28532


namespace valid_triples_l28_28173

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ∣ (y + 1)) (hyz : y ∣ (z + 1)) (hzx : z ∣ (x + 1)) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
  (x = 1 ∧ y = 2 ∧ z = 3) :=
sorry

end valid_triples_l28_28173


namespace total_students_l28_28780

-- Define n as total number of students
variable (n : ℕ)

-- Define conditions
variable (h1 : 550 ≤ n)
variable (h2 : (n / 10) + 10 ≤ n)

-- Define the proof statement
theorem total_students (h : (550 * 10 + 5) = n ∧ 
                        550 * 10 / n + 10 = 45 + n) : 
                        n = 1000 := by
  sorry

end total_students_l28_28780


namespace all_propositions_correct_l28_28280

-- Definitions of planes, lines, and perpendicularity
variable (Plane : Type)
variable (Line : Type)
variable (Point : Type)

variable α β γ : Plane
variable a b c : Line
variable P : Point

open Classical

axiom non_coincident_planes : α ≠ β ∧ α ≠ γ ∧ β ≠ γ
axiom line_intersections :
  (α ∩ β = a) ∧ (α ∩ γ = b) ∧ (β ∩ γ = c)

-- Propositions
def proposition1 (a b c : Line) : Prop :=
  (a ⊥ b ∧ a ⊥ c) → b ⊥ c

def proposition2 (a b : Line) (P : Point) : Prop :=
  (a ∩ b = P) → (a ∩ c = P)

def proposition3 (a b : Line) (α γ : Plane) : Prop :=
  (a ⊥ b ∧ a ⊥ c) → α ⊥ γ

def proposition4 (a b : Line) : Prop :=
  (a ∥ b) → (a ∥ c)

-- Theorem to prove that all propositions are correct
theorem all_propositions_correct :
  proposition1 a b c ∧ proposition2 a b P ∧ proposition3 a b α γ ∧ proposition4 a b :=
by
  sorry

end all_propositions_correct_l28_28280


namespace probability_slope_geq_2_over_5_l28_28915

theorem probability_slope_geq_2_over_5 :
  let events := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]],
      favorable_events := [(a, b) ∈ events | (b: ℚ) / a ≤ 2 / 5] in
  (favorable_events.length : ℚ) / events.length = 1 / 6 :=
by
  let events := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]],
      favorable_events := events.filter (λ (ab : ℕ × ℕ), (ab.2 : ℚ) / ab.1 ≤ 2 / 5)
  have h1 : events.length = 36 := sorry
  have h2 : favorable_events.length = 6 := sorry
  rw [h1, h2]
  norm_num
  simp

end probability_slope_geq_2_over_5_l28_28915


namespace total_shots_cost_l28_28625

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l28_28625


namespace maxRegions100Parabolas_l28_28374

-- Define the number of parabolas of each type
def numberOfParabolas1 := 50
def numberOfParabolas2 := 50

-- Define the function that counts the number of regions formed by n parabolas intersecting at most m times
def maxRegions (n m : Nat) : Nat :=
  (List.range (m+1)).foldl (λ acc k => acc + Nat.choose n k) 0

-- Specify the intersection properties for each type of parabolas
def intersectionsParabolas1 := 2
def intersectionsParabolas2 := 2
def intersectionsBetweenSets := 4

-- Calculate the number of regions formed by each set of 50 parabolas
def regionsSet1 := maxRegions numberOfParabolas1 intersectionsParabolas1
def regionsSet2 := maxRegions numberOfParabolas2 intersectionsParabolas2

-- Calculate the additional regions created by intersections between the sets
def additionalIntersections := numberOfParabolas1 * numberOfParabolas2 * intersectionsBetweenSets

-- Combine the regions
def totalRegions := regionsSet1 + regionsSet2 + additionalIntersections + 1

-- Prove the final result
theorem maxRegions100Parabolas : totalRegions = 15053 :=
  sorry

end maxRegions100Parabolas_l28_28374


namespace minimum_value_exists_l28_28302

noncomputable def minimum_value_2x_plus_y (x y : ℝ) : ℝ :=
  2 * x + y

theorem minimum_value_exists (x y : ℝ) (h1 : log 2 x + log 2 y = 3) (h2 : x > 0) (h3 : y > 0) :
  ∃ (z : ℝ), z = minimum_value_2x_plus_y x y ∧ z = 8 :=
by
  sorry

end minimum_value_exists_l28_28302


namespace max_intersection_points_three_circles_two_lines_l28_28902

theorem max_intersection_points_three_circles_two_lines : 
  ∀ (C1 C2 C3 L1 L2 : set ℝ × ℝ) (hC1 : is_circle C1) (hC2 : is_circle C2) (hC3 : is_circle C3) (hL1 : is_line L1) (hL2 : is_line L2),
  ∃ P : ℕ, P = 19 ∧
  (∀ (P : ℝ × ℝ), P ∈ C1 ∧ P ∈ C2 ∨ P ∈ C1 ∧ P ∈ C3 ∨ P ∈ C2 ∧ P ∈ C3 ∨ P ∈ C1 ∧ P ∈ L1 ∨ P ∈ C2 ∧ P ∈ L1 ∨ P ∈ C3 ∧ P ∈ L1 ∨ P ∈ C1 ∧ P ∈ L2 ∨ P ∈ C2 ∧ P ∈ L2 ∨ P ∈ C3 ∧ P ∈ L2 ∨ P ∈ L1 ∧ P ∈ L2) ↔ P = 19 :=
sorry

end max_intersection_points_three_circles_two_lines_l28_28902


namespace transfer_student_increases_averages_l28_28347

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ :=
  if grades.length > 0 then (grades.sum / grades.length) else 0

def can_transfer_to_increase_averages (group1_grades group2_grades : List ℝ) : Prop :=
  ∃ x ∈ group1_grades, average (x :: group2_grades) > average group2_grades ∧
                       average (group1_grades.erase x) > average group1_grades

theorem transfer_student_increases_averages :
  can_transfer_to_increase_averages group1_grades group2_grades :=
by {
  sorry -- The proof is skipped as per instructions
}

end transfer_student_increases_averages_l28_28347


namespace total_interest_l28_28544

variable (P R : ℝ)

-- Given condition: Simple interest on sum of money is Rs. 700 after 10 years
def interest_10_years (P R : ℝ) : Prop := (P * R * 10) / 100 = 700

-- Principal is trebled after 5 years
def interest_5_years_treble (P R : ℝ) : Prop := (15 * P * R) / 100 = 105

-- The final interest is the sum of interest for the first 10 years and next 5 years post trebling the principal
theorem total_interest (P R : ℝ) (h1: interest_10_years P R) (h2: interest_5_years_treble P R) : 
  (700 + 105 = 805) := 
  by 
  sorry

end total_interest_l28_28544


namespace plane_equation_proof_l28_28579

-- Define the parametric representation of the plane
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 1 + 2 * s, 4 - s + 3 * t)

-- Define the plane equation form
def plane_equation (x y z : ℝ) (A B C D : ℤ) : Prop :=
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

-- Define the normal vector derived from the cross product
def normal_vector : ℝ × ℝ × ℝ := (6, -5, 2)

-- Define the initial point used to calculate D
def initial_point : ℝ × ℝ × ℝ := (2, 1, 4)

-- Proposition to prove the equation of the plane
theorem plane_equation_proof :
  ∃ (A B C D : ℤ), A = 6 ∧ B = -5 ∧ C = 2 ∧ D = -15 ∧
    ∀ x y z : ℝ, plane_equation x y z A B C D ↔
      ∃ s t : ℝ, plane_parametric s t = (x, y, z) :=
by
  sorry

end plane_equation_proof_l28_28579


namespace max_rectangle_area_l28_28132

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l28_28132


namespace pentagon_area_triangle_percentage_l28_28125

-- Definitions related to the problem conditions
def longer_leg (x : ℝ) := x
def shorter_leg (x : ℝ) := x / 2
def hypotenuse (x : ℝ) := x * Real.sqrt 3
def rectangle_area (a b : ℝ) := a * b
def triangle_area (a b : ℝ) := (1 / 2) * a * b
def pentagon_area (x : ℝ) :=
  triangle_area (longer_leg x) (shorter_leg x) + rectangle_area (hypotenuse x) (2 * x)

-- The target proof statement
theorem pentagon_area_triangle_percentage (x : ℝ) (h : x > 0) :
  (triangle_area (longer_leg x) (shorter_leg x) / pentagon_area x) * 100 = 4.188 := by
  sorry

end pentagon_area_triangle_percentage_l28_28125


namespace proof_l28_28770

structure Point where
  x : ℝ
  y : ℝ

def line_rect_eq (x y : ℝ) : Prop := 2 * x - y - 6 = 0

def curve_eq (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 4) = 1

def point_on_curve (P : Point) (θ : ℝ) : Prop :=
  P.x = sqrt 3 * cos θ ∧ P.y = 2 * sin θ

def distance (P : Point) : ℝ :=
  abs (2 * P.x - P.y - 6) / sqrt (2^2 + (-1)^2)

noncomputable def max_distance (C₁_eq : ∀ x y, curve_eq x y) : ℝ :=
  2 * sqrt 5

theorem proof (θ : ℝ) :
    (∀ (P : Point), point_on_curve P θ → C₁_eq P.x P.y) → 
    (∀ (x y : ℝ), line_rect_eq x y → 2 * x - y - 6 = 0) → 
    (∀ P, point_on_curve P θ → 
      max_distance curve_eq = 2 * sqrt 5) :=
by
  intros P H1 H2
  sorry

end proof_l28_28770


namespace num_valid_A_values_l28_28672

theorem num_valid_A_values : ∃ A_vals : Finset ℕ, 
  (∀ A ∈ A_vals, 72 % A = 0 ∧ (273100 + 10 * A + 2) % 4 = 0) ∧ A_vals.card = 3 :=
by
  let A_vals := {1, 3, 9} -- Set of valid A values
  use A_vals
  -- Prove the properties and cardinality
  simp [A_vals, Finset.mem_singleton, Finset.mem_insert]
  sorry

end num_valid_A_values_l28_28672


namespace peanut_ratio_correct_l28_28469

-- Define the conditions
def ratio_in_last_batch := 4 / 16
def total_weight := 20
def oil_used := 4
def peanut_weight := total_weight - oil_used

-- Define the wanted ratio to prove
def wanted_ratio := 2 / 8

-- Define the theorem to prove
theorem peanut_ratio_correct : ratio_in_last_batch = wanted_ratio :=
by
  unfold ratio_in_last_batch
  unfold wanted_ratio
  unfold total_weight
  unfold oil_used
  unfold peanut_weight
  sorry

end peanut_ratio_correct_l28_28469


namespace k_divides_99_l28_28572

-- Define what it means for a number to reverse its digits
def reverse_digits (n : ℕ) : ℕ :=
  n.digits.reverse.foldl (λ acc d, acc * 10 + d) 0

-- The main statement to prove:
theorem k_divides_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
by
  sorry

end k_divides_99_l28_28572


namespace num_valid_A_values_l28_28671

theorem num_valid_A_values : ∃ A_vals : Finset ℕ, 
  (∀ A ∈ A_vals, 72 % A = 0 ∧ (273100 + 10 * A + 2) % 4 = 0) ∧ A_vals.card = 3 :=
by
  let A_vals := {1, 3, 9} -- Set of valid A values
  use A_vals
  -- Prove the properties and cardinality
  simp [A_vals, Finset.mem_singleton, Finset.mem_insert]
  sorry

end num_valid_A_values_l28_28671


namespace smallest_integer_condition_l28_28535

def is_not_prime (n : Nat) : Prop := ¬ Nat.Prime n

def is_not_square (n : Nat) : Prop :=
  ∀ m : Nat, m * m ≠ n

def has_no_prime_factor_less_than (n k : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p < k → ¬ (p ∣ n)

theorem smallest_integer_condition :
  ∃ n : Nat, n > 0 ∧ is_not_prime n ∧ is_not_square n ∧ has_no_prime_factor_less_than n 70 ∧ n = 5183 :=
by {
  sorry
}

end smallest_integer_condition_l28_28535


namespace line_passes_through_center_l28_28706

-- Define the equation of the circle as given in the problem.
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the center of the circle.
def center_of_circle (x y : ℝ) : Prop := x = 1 ∧ y = -3

-- Define the equation of the line.
def line_equation (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- The theorem to prove.
theorem line_passes_through_center :
  (∃ x y, circle_equation x y ∧ center_of_circle x y) →
  (∃ x y, center_of_circle x y ∧ line_equation x y) :=
by
  sorry

end line_passes_through_center_l28_28706


namespace drop_volume_l28_28018

theorem drop_volume :
  let leak_rate := 3 -- drops per minute
  let pot_volume := 3 * 1000 -- volume in milliliters
  let time := 50 -- minutes
  let total_drops := leak_rate * time -- total number of drops
  (pot_volume / total_drops) = 20 := 
by
  let leak_rate : ℕ := 3
  let pot_volume : ℕ := 3 * 1000
  let time : ℕ := 50
  let total_drops := leak_rate * time
  have h : (pot_volume / total_drops) = 20 := by sorry
  exact h

end drop_volume_l28_28018


namespace find_line_l_l28_28703

open Real

-- Define the curve y = 4 / x
def curve (x : ℝ) : ℝ := 4 / x

-- Define the distance formula between two parallel lines
def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / sqrt (a^2 + b^2)

-- Define the tangent line at (1, 4) given the curve
def tangent_line_at_P (x y k : ℝ) : Prop :=
  y = curve x ∧ k = -4 ∧ y - 4 = k * (x - 1) -- tangent line properties

-- Define the general form of line l parallel to the tangent line
def line_l (a b c : ℝ) : Prop :=
  a = 4 ∧ b = 1 -- Since it is parallel, coefficients are same

noncomputable def line_equation (a b c : ℝ) : ℝ × ℝ × ℝ :=
  if distance_between_parallel_lines a b c (-8) = sqrt 17 then (a, b, c) else (a, b, -25)

theorem find_line_l :
  ∃ (a b c : ℝ), 
    tangent_line_at_P 1 4 (-4) ∧ 
    line_l a b c ∧ 
    distance_between_parallel_lines a b c (-8) = sqrt 17 ∧ 
    (a = 4 ∧ b = 1 ∧ (c = 9 ∨ c = -25)) :=
by
  -- The proof steps will be filled here.
  sorry

end find_line_l_l28_28703


namespace price_reduction_equation_l28_28103

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l28_28103


namespace students_remaining_l28_28310

theorem students_remaining (initial_students : ℕ) (stops : ℕ)
    (get_off_fraction : ℚ) (h_initial : initial_students = 48)
    (h_stops : stops = 3) (h_fraction : get_off_fraction = 1 / 2) : 
    let remaining := initial_students * get_off_fraction^stops 
    in remaining = 6 :=
by
  sorry

end students_remaining_l28_28310


namespace solution_set_l28_28416

noncomputable def f : ℝ → ℝ := sorry
noncomputable def derivative (f : ℝ → ℝ) (x : ℝ) := sorry

variable {x : ℝ}

-- Given conditions
axiom h1 : ∀ (x : ℝ), derivative f x = f' x
axiom h2 : f 3 = real.exp 3
axiom h3 : ∀ (x : ℝ), f' x - f x > 0

-- Result to be proved
theorem solution_set (hf : ∀ (x : ℝ), derivative f x = f' x) (hf3 : f 3 = real.exp 3) (hf_cond : ∀ (x : ℝ), f' x - f x > 0) : 
  { x : ℝ | f x - real.exp x > 0 } = set.Ioo 3 ⊤ :=
sorry

end solution_set_l28_28416


namespace inscribed_circle_radius_eq_l28_28068

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  area / s

theorem inscribed_circle_radius_eq :
  radius_of_inscribed_circle 8 8 10 8 8 10 = (15 * Real.sqrt 13) / 13 :=
  sorry

end inscribed_circle_radius_eq_l28_28068


namespace change_first_digit_results_in_largest_number_l28_28259

theorem change_first_digit_results_in_largest_number :
  ∀ (d1 d2 d3 d4 d5 d6 : ℕ), 
  let n := 0.123456,
      n1 := 0.823456,
      n2 := 0.183456,
      n3 := 0.128456,
      n4 := 0.123856,
      n5 := 0.123486,
      n6 := 0.123458 in
  0 ≤ d1 ∧ d1 < 10 → 
  0 ≤ d2 ∧ d2 < 10 → 
  0 ≤ d3 ∧ d3 < 10 → 
  0 ≤ d4 ∧ d4 < 10 → 
  0 ≤ d5 ∧ d5 < 10 → 
  0 ≤ d6 ∧ d6 < 10 → 
  (Max (List.map (λ (x : ℝ), x) [n1, n2, n3, n4, n5, n6])) = n1 :=
by
  sorry

end change_first_digit_results_in_largest_number_l28_28259


namespace width_of_each_brick_l28_28107

theorem width_of_each_brick (width : ℝ) : 
  let wall_volume := 800 * 600 * 22.5
  let brick_volume := 125 * width * 6
  1280 * brick_volume = wall_volume 
  ⟹ width = 11.25 :=
by
  sorry

end width_of_each_brick_l28_28107


namespace total_amount_is_4000_l28_28098

-- Define the amount put at a 3% interest rate
def amount_at_3_percent : ℝ := 2800

-- Define the total annual interest from both investments
def total_annual_interest : ℝ := 144

-- Define the interest rate for the amount put at 3% and 5%
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

-- Define the total amount to be proved
def total_amount_divided (T : ℝ) : Prop :=
  interest_rate_3_percent * amount_at_3_percent + 
  interest_rate_5_percent * (T - amount_at_3_percent) = total_annual_interest

-- The theorem that states the total amount divided is Rs. 4000
theorem total_amount_is_4000 : ∃ T : ℝ, total_amount_divided T ∧ T = 4000 :=
by
  use 4000
  unfold total_amount_divided
  simp
  sorry

end total_amount_is_4000_l28_28098


namespace sawyer_joined_coaching_l28_28841

variable (daily_fees total_fees : ℕ)
variable (year_not_leap : Prop)
variable (discontinue_day : ℕ)

theorem sawyer_joined_coaching :
  daily_fees = 39 → 
  total_fees = 11895 → 
  year_not_leap → 
  discontinue_day = 307 → 
  ∃ start_day, start_day = 30 := 
by
  intros h_daily_fees h_total_fees h_year_not_leap h_discontinue_day
  sorry

end sawyer_joined_coaching_l28_28841


namespace therese_older_than_aivo_l28_28221

-- Definitions based on given conditions
variables {Aivo Jolyn Leon Therese : ℝ}
variables (h1 : Jolyn = Therese + 2)
variables (h2 : Leon = Aivo + 2)
variables (h3 : Jolyn = Leon + 5)

-- Statement to prove
theorem therese_older_than_aivo :
  Therese = Aivo + 5 :=
by
  sorry

end therese_older_than_aivo_l28_28221


namespace maximize_q_l28_28798

noncomputable def maximum_q (X Y Z : ℕ) : ℕ :=
X * Y * Z + X * Y + Y * Z + Z * X

theorem maximize_q : ∃ (X Y Z : ℕ), X + Y + Z = 15 ∧ (∀ (A B C : ℕ), A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) ∧ maximum_q X Y Z = 200 :=
by
  sorry

end maximize_q_l28_28798


namespace find_triple_row_matrix_l28_28658

theorem find_triple_row_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (∀ a b c d : ℝ, (M ⬝ (Matrix.of ![![a, b], ![c, d]])) = (Matrix.of ![![a, b], ![3 * c, 3 * d]])) : 
  M = (Matrix.of ![![1, 0], ![0, 3]]) :=
by
  sorry

end find_triple_row_matrix_l28_28658


namespace problem_statement_l28_28148

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def has_minimum_value_at (f : ℝ → ℝ) (a : ℝ) := ∀ x : ℝ, f a ≤ f x
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_statement : is_even_function f4 ∧ has_minimum_value_at f4 0 :=
by
  sorry

end problem_statement_l28_28148


namespace find_slope_of_q_l28_28810

theorem find_slope_of_q (j : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + 3 → y = j * x + 1 → x = 1 → y = 5) → j = 4 := 
by
  intro h
  sorry

end find_slope_of_q_l28_28810


namespace average_non_prime_squares_approx_l28_28665

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of non-prime numbers between 50 and 100
def non_prime_numbers : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70,
   72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91,
   92, 93, 94, 95, 96, 98, 99]

-- Define the sum of squares of the elements in a list
def sum_of_squares (l : List ℕ) : ℕ :=
  l.foldr (λ x acc => x * x + acc) 0

-- Define the count of non-prime numbers
def count_non_prime : ℕ :=
  non_prime_numbers.length

-- Calculate the average
def average_non_prime_squares : ℚ :=
  sum_of_squares non_prime_numbers / count_non_prime

-- Theorem to state that the average of the sum of squares of non-prime numbers
-- between 50 and 100 is approximately 6417.67
theorem average_non_prime_squares_approx :
  abs ((average_non_prime_squares : ℝ) - 6417.67) < 0.01 := 
  sorry

end average_non_prime_squares_approx_l28_28665


namespace households_surveyed_l28_28963

theorem households_surveyed (
  neither_brands : ℕ := 80,
  only_brand_A : ℕ := 60,
  both_brands_ratio : ℕ := 3,
  households_both_brands : ℕ := 40
) : 
  let only_brand_B := both_brands_ratio * households_both_brands,
      total_households := neither_brands + only_brand_A + only_brand_B + households_both_brands
  in 
  total_households = 300 := 
by
  have h_only_brand_B : ℕ := both_brands_ratio * households_both_brands
  have total_households : ℕ := neither_brands + only_brand_A + h_only_brand_B + households_both_brands
  show total_households = 300 from sorry

end households_surveyed_l28_28963


namespace pinwheel_area_eq_six_l28_28166

open Set

/-- Define the pinwheel in a 6x6 grid -/
def is_midpoint (x y : ℤ) : Prop :=
  (x = 3 ∧ (y = 1 ∨ y = 5)) ∨ (y = 3 ∧ (x = 1 ∨ x = 5))

def is_center (x y : ℤ) : Prop :=
  x = 3 ∧ y = 3

def is_triangle_vertex (x y : ℤ) : Prop :=
  is_center x y ∨ is_midpoint x y

-- Main theorem statement
theorem pinwheel_area_eq_six :
  let pinwheel : Set (ℤ × ℤ) := {p | is_triangle_vertex p.1 p.2}
  ∀ A : ℝ, A = 6 :=
by sorry

end pinwheel_area_eq_six_l28_28166


namespace log_b_over_a_range_l28_28701

theorem log_b_over_a_range (a b c : ℝ) 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c)
  (h1 : 1 / real.exp 1 ≤ c / a) 
  (h2 : c / a ≤ 2) 
  (h3 : c * real.log b = a + c * real.log c) :
  1 ≤ real.log (b / a) ∧ real.log (b / a) ≤ real.exp 1 - 1 :=
by
  sorry

end log_b_over_a_range_l28_28701


namespace negation_of_some_triangles_have_three_equal_medians_l28_28495

theorem negation_of_some_triangles_have_three_equal_medians :
  ¬ (∃ t : Triangle, t.has_three_equal_medians) ↔ ∀ t : Triangle, ¬ t.has_three_equal_medians :=
by
  sorry

end negation_of_some_triangles_have_three_equal_medians_l28_28495


namespace min_value_of_squares_l28_28217

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end min_value_of_squares_l28_28217


namespace can_color_pairs_l28_28167

-- Definition of the coloring rule
def is_black (a b : ℕ) : Prop := ∃ k, b = a + 2^k * d ∧ odd k
def is_white (a b : ℕ) : Prop := ∃ k, b = a + 2^k * d ∧ even k

-- Main theorem
theorem can_color_pairs (a d : ℕ) (h : a < a + d ∧ a < a + 2 * d ∧ a + d < a + 2 * d) :
  ∃ (a d : ℕ), (is_black a (a + d) ∧ is_white a (a + 2 * d) ∧ is_black (a + d) (a + 2 * d)) 
  ∨ (is_white a (a + d) ∧ is_black a (a + 2 * d) ∧ is_white (a + d) (a + 2 * d)) :=
begin
  sorry
end

end can_color_pairs_l28_28167


namespace fill_cost_canN_l28_28619

-- Definitions based on conditions
def right_circular_cylinder (r h : ℝ) := r > 0 ∧ h > 0

variables (r h : ℝ) (k : ℝ)

-- Can B and Can N definitions based on conditions
def CanB := right_circular_cylinder r h
def CanN := right_circular_cylinder (2 * r) (k * h)

-- Given cost to fill half of Can B
def cost_half_B : ℝ := 4.00
def cost_full_B : ℝ := 2 * cost_half_B

-- Statement to prove: Cost to fill Can N
def cost_fill_CanN : ℝ := 32.00 * k

theorem fill_cost_canN
  (hb : CanB)
  (hn : CanN)
  (H_cost_B : cost_full_B = 8.00) :
  cost_fill_CanN = 32.00 * k := 
sorry

end fill_cost_canN_l28_28619


namespace polygon_bisected_by_any_line_l28_28192

-- Given definitions from the conditions
variable (P : Type) [polygon P]
variable (O : point P) -- on the boundary of P

-- The final proof statement
theorem polygon_bisected_by_any_line (P : polygon) (O : point P) 
  (h_boundary : O ∈ boundary P)
  (h_bisection : ∀ l : line, O ∈ l → divides_area_into_equal_halves P l) : --condition
  ∃ P : polygon, ∀ l : line, O ∈ l → divides_area_into_equal_halves P l := --prove

begin
  sorry -- Proof is deliberately omitted as requested
end

end polygon_bisected_by_any_line_l28_28192


namespace moles_diatomic_approx_l28_28546

-- Given conditions
def moles_monoatomic := 1.5 -- moles of monoatomic gas
def heat_capacity := 120 -- J/K
def gas_constant := 8.31 -- J/(K·mol)

-- Prove the number of moles of diatomic gas
theorem moles_diatomic_approx : 
  ∃ v' : Real, (v' ≈ 3.8) ∧ 
             (2 * moles_monoatomic * gas_constant + 3 * v' * gas_constant = heat_capacity) :=
by
  sorry

end moles_diatomic_approx_l28_28546


namespace zero_polynomial_l28_28610

theorem zero_polynomial (P : Polynomial ℤ)
  (h : ∀ n : ℕ, n > 0 → n ∣ P.eval (2^n)) : P = 0 :=
sorry

end zero_polynomial_l28_28610


namespace range_of_m_l28_28809

def vector_a (m : ℝ) : ℝ × ℝ := (m - 2, m + 3)
def vector_b : ℝ × ℝ := (3, 2)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem range_of_m (m : ℝ) (h_dot_product : dot_product (vector_a m) vector_b < 0) :
  m ∈ set.Ioo (-∞) (-13) ∪ set.Ioo (-13) 0 :=
sorry

end range_of_m_l28_28809


namespace parabola_translation_l28_28041

theorem parabola_translation (x : ℝ) :
  let y := 3*x^2 in 
  y + 3 = 3*x^2 + 3 :=
by
  sorry

end parabola_translation_l28_28041


namespace max_alpha_l28_28233

theorem max_alpha (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hSum : A + B + C = π)
  (hmin : ∀ alpha, alpha = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A))) :
  ∃ alpha, alpha = 2 * π / 9 := 
sorry

end max_alpha_l28_28233


namespace derivative_of_fraction_l28_28940

theorem derivative_of_fraction (x : ℝ) (h : x ≠ 0) :
  (deriv (λ x, (3 * x^2 - x * sqrt x + 5 * sqrt x - 9) / sqrt x)) x = 
  (9 / 2) * x ^ (1 / 2) - 1 + (9 / 2) * x ^ (- (3 / 2)) :=
sorry

end derivative_of_fraction_l28_28940


namespace ellipse_equation_l28_28237

theorem ellipse_equation (foci_on_axes : ∀ p : ℝ × ℝ, (p.1 = 4 ∧ p.2 = 0) ∨ (p.1 = -4 ∧ p.2 = 0) ∨ (p.1 = 0 ∧ p.2 = 4) ∨ (p.1 = 0 ∧ p.2 = -4))
  (midpoint : (0 : ℝ) ∈ ball (0 : ℝ × ℝ) 4)
  (focal_distance : real.dist (4 : ℝ) (-4 : ℝ) = 8)
  (sum_distances : ∀ p : ℝ × ℝ, ∃ q : ℝ × ℝ, real.dist p q = 12) :
  ∃ (a b : ℝ), (a = 6) ∧ (b^2 = 36 - 16) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ 
  (∀ x y : ℝ, (x^2 / 36 + y^2 / 20 = 1) ∨ (y^2 / 36 + x^2 / 20 = 1)) :=
by 
  sorry

end ellipse_equation_l28_28237


namespace horizontal_distance_approximation_l28_28890

def slope_inclination : ℝ := 24 + 36 / 60 -- 24 degrees 36 minutes represented in degrees
def slope_distance : ℝ := 4
def cosine_value : ℝ := 0.909

theorem horizontal_distance_approximation :
  let horizontal_distance := cosine_value * slope_distance in 
  abs (horizontal_distance - 3.6) < 0.1 := 
by
  sorry

end horizontal_distance_approximation_l28_28890


namespace gnuff_tutor_minutes_l28_28282

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l28_28282


namespace original_ribbon_length_l28_28525

theorem original_ribbon_length :
  ∃ x : ℝ, 
    (∀ a b : ℝ, 
       a = x - 18 ∧ 
       b = x - 12 ∧ 
       b = 2 * a → x = 24) :=
by
  sorry

end original_ribbon_length_l28_28525


namespace pow_increasing_log_increasing_l28_28997

open Real
open Nat

-- Given the conditions
variable {a b c d : ℝ}

-- Definitions
def is_increasing (f : ℝ → ℝ) : Prop := ∀ (x y : ℝ), x < y → f x < f y

theorem pow_increasing (a b : ℝ) (h : 2 > 1) (h₀ : 0.6 > 0.5) :
  2 ^ 0.6 > 2 ^ 0.5 :=
by {
  -- Statement only (proof omitted)
  sorry
}

theorem log_increasing (a b : ℝ) (h : 2 > 1) (h₀ : 3.4 < 3.8) :
  log 2 3.4 < log 2 3.8 :=
by {
  -- Statement only (proof omitted)
  sorry
}

end pow_increasing_log_increasing_l28_28997


namespace turtle_ratio_l28_28043

theorem turtle_ratio (K : ℕ) (T : ℕ) (kris_turtles kristen_turtles total_turtles : ℕ) 
  (h1 : T = 5 * K)
  (h2 : kristen_turtles = 12)
  (h3 : total_turtles = 30)
  (h4 : total_turtles = K + 5 * K + kristen_turtles) :
  (K / nat.gcd K kristen_turtles, kristen_turtles / nat.gcd K kristen_turtles) = (1, 4) :=
by
  have h5 : 6 * K + kristen_turtles = total_turtles := h4
  sorry

end turtle_ratio_l28_28043


namespace dividend_percentage_is_5_percent_l28_28122

-- Define the conditions as variables and hypotheses
variables (investment : ℝ) (share_price : ℝ) (premium_rate : ℝ)
variables (total_dividend : ℝ) (face_value : ℝ)

-- Given conditions
def cost_per_share := share_price * (1 + premium_rate)
def number_of_shares := investment / cost_per_share
def dividend_per_share := total_dividend / number_of_shares
def dividend_percentage := (dividend_per_share / face_value) * 100 

-- The theorem stating the dividend percentage calculation
theorem dividend_percentage_is_5_percent 
  (h1 : investment = 14400)
  (h2 : share_price = 100)
  (h3 : premium_rate = 0.2)
  (h4 : total_dividend = 600)
  (h5 : face_value = 100) : 
  dividend_percentage investment share_price premium_rate total_dividend face_value = 5 :=
by
  sorry

end dividend_percentage_is_5_percent_l28_28122


namespace sufficient_condition_for_inequality_l28_28511

theorem sufficient_condition_for_inequality (a x : ℝ) (h1 : -2 < x) (h2 : x < -1) :
  (a + x) * (1 + x) < 0 → a > 2 :=
sorry

end sufficient_condition_for_inequality_l28_28511


namespace sophie_donuts_left_l28_28474

theorem sophie_donuts_left :
  ∀ (boxes_initial : ℕ) (donuts_per_box : ℕ) (boxes_given_away : ℕ) (dozen : ℕ),
  boxes_initial = 4 →
  donuts_per_box = 12 →
  boxes_given_away = 1 →
  dozen = 12 →
  (boxes_initial - boxes_given_away) * donuts_per_box - (dozen / 2) = 30 :=
by 
  intros boxes_initial donuts_per_box boxes_given_away dozen 
  assume h1 h2 h3 h4
  sorry

end sophie_donuts_left_l28_28474


namespace find_a_if_F_leq_four_fifths_l28_28263

noncomputable def F (x a : ℝ) : ℝ := (x - a) ^ 2 + (log (x ^ 2) - 2 * a) ^ 2

theorem find_a_if_F_leq_four_fifths (x0 a : ℝ) (h1 : x0 > 0) (h2 : F x0 a ≤ 4 / 5) : a = 1 / 5 :=
sorry

end find_a_if_F_leq_four_fifths_l28_28263


namespace zero_lies_in_interval_l28_28325

def f (x : ℝ) : ℝ := -|x - 5| + 2 * x - 1

theorem zero_lies_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 2 := 
sorry

end zero_lies_in_interval_l28_28325


namespace flowers_wilted_l28_28195

theorem flowers_wilted 
  (initial_flowers : ℕ) 
  (flowers_per_bouquet : ℕ) 
  (bouquets_made_after_wilt : ℕ) :
  initial_flowers = 88 ∧ 
  flowers_per_bouquet = 5 ∧ 
  bouquets_made_after_wilt = 8 → 
  initial_flowers - (bouquets_made_after_wilt * flowers_per_bouquet) = 48 :=
by
  intro h
  cases h with h_initial h_rest
  cases h_rest with h_per_bouquet h_bouquets
  rw [h_initial, h_per_bouquet, h_bouquets]
  simp
  sorry

end flowers_wilted_l28_28195


namespace alan_total_spending_l28_28597

-- Define the conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation
def cost_eggs : ℕ := eggs_bought * price_per_egg
def cost_chickens : ℕ := chickens_bought * price_per_chicken
def total_amount_spent : ℕ := cost_eggs + cost_chickens

-- Prove the total amount spent
theorem alan_total_spending : total_amount_spent = 88 := by
  show cost_eggs + cost_chickens = 88
  sorry

end alan_total_spending_l28_28597


namespace parabola_directrix_l28_28275

variable (a : ℝ)

theorem parabola_directrix (h1 : ∀ x : ℝ, y = a * x^2) (h2 : y = -1/4) : a = 1 :=
sorry

end parabola_directrix_l28_28275


namespace find_x_value_l28_28678

open Real

theorem find_x_value (a : ℝ) (x : ℝ) (h : a > 0) (h_eq : 10^x = log (10 * a) + log (a⁻¹)) : x = 0 :=
by
  sorry

end find_x_value_l28_28678


namespace tan_neg_225_is_neg_1_l28_28878

def tan_neg_225_eq_neg_1 : Prop :=
  Real.tan (-225 * Real.pi / 180) = -1

theorem tan_neg_225_is_neg_1 : tan_neg_225_eq_neg_1 :=
  by
    sorry

end tan_neg_225_is_neg_1_l28_28878


namespace sine_product_value_l28_28183

noncomputable def evaluate_sine_product : ℝ :=
  real.sin (real.pi / 18) * real.sin (real.pi / 6) * real.sin (5 * real.pi / 18) * real.sin (7 * real.pi / 18)

theorem sine_product_value : evaluate_sine_product = 1 / 16 := by
  sorry

end sine_product_value_l28_28183


namespace perpendicular_condition_l28_28384

-- Definitions for the vectors corresponding to the edges VA, VB, and VC
variables {V A B C : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]
variables (VA VB VC : V)

-- Conditions
def condition1 : Prop := VC ⊥ VA
def condition2 : Prop := VC ⊥ VB

-- Prove that VC ⊥ AB under the given conditions.
theorem perpendicular_condition (h1 : condition1 VA VC) (h2 : condition2 VB VC) : 
  (⟪VC, A - B⟫ = 0) :=
by sorry

end perpendicular_condition_l28_28384


namespace intersection_empty_l28_28240

noncomputable def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
noncomputable def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem intersection_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_empty_l28_28240


namespace exists_common_tangent_l28_28169

-- Define the two circles with centers O and O1, radii R and r
variables {R r : ℝ} (O O1 : ℝ×ℝ) [fact (R > r)]

-- Define the existence of common tangents of two circles depending on cases mentioned
theorem exists_common_tangent :
  ∃ (T₁ T₂ : ℝ × ℝ), (is_tangent O R T₁ ∧ is_tangent O1 r T₁) ∨
                      (is_tangent O R T₂ ∧ is_tangent O1 r T₂) :=
by 
  sorry

end exists_common_tangent_l28_28169


namespace solve_for_x_l28_28031

theorem solve_for_x (x : ℝ) (h : (3 * x + 15)^2 = 3 * (4 * x + 40)) :
  x = -5 / 3 ∨ x = -7 :=
sorry

end solve_for_x_l28_28031


namespace dream_star_games_l28_28027

theorem dream_star_games (x y : ℕ) 
  (h1 : x + y + 2 = 9)
  (h2 : 3 * x + y = 17) : 
  x = 5 ∧ y = 2 := 
by 
  sorry

end dream_star_games_l28_28027


namespace no_real_pairs_ab_arithmetic_progression_l28_28177

theorem no_real_pairs_ab_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (a = (15 + b) / 2) ∧ (a + a * b = 2 * b) :=
by {
  intro hab,
  rcases hab with ⟨a, b, ha1, ha2⟩,
  have h1 : a = (15 + b) / 2 := ha1,
  have h2 : (a + a * b = 2 * b),
  {
    rw ha1,
    rw ha2,
  },
  let f : ℝ → ℝ := λ b, b^2 + 9 * b + 30,
  have h_quad_eq : ∀ b, f b = 0 → false,
  {
    intro b,
    rw f,
    rw ha1,
    rw ha2,
    have discriminant : (9^2 - 4 * 1 * 30 < 0), by linarith,
    rw (ring_eq_of_se_zero_iff discriminant),
    have h : b^2 + 9b + 30 = 0 := sorry,
    sorry,
  },
  exact h_quad_eq (15, b, ha1, ha2),
}

end no_real_pairs_ab_arithmetic_progression_l28_28177


namespace c_share_of_rent_l28_28922

theorem c_share_of_rent :
  let oxen_months_a := 10 * 7 in
  let oxen_months_b := 12 * 5 in
  let oxen_months_c := 15 * 3 in
  let total_oxen_months := oxen_months_a + oxen_months_b + oxen_months_c in
  let total_rent := 280 in
  let c_fraction := oxen_months_c / total_oxen_months in
  let c_payment := c_fraction * total_rent in
  c_payment = 72 :=
by
  sorry

end c_share_of_rent_l28_28922


namespace convex_quadrilateral_diameters_proof_l28_28079

variable (A B C D M K L : Point)
variable [ABCD_convex : ConvexQuadrilateral A B C D]
variable [touches_externally_at_M : CirclesOnDiametersTouchExternally A B M C D]
variable [circle_AMC_intersects_at_K : CircleIntersectsLine A M C M (Midpoint A B) at K]
variable [circle_BMD_intersects_at_L : CircleIntersectsLine B M D M (Midpoint A B) at L]

theorem convex_quadrilateral_diameters_proof :
  abs (distance M K - distance M L) = abs (distance A B - distance C D) :=
sorry

end convex_quadrilateral_diameters_proof_l28_28079


namespace max_points_of_intersection_l28_28898

theorem max_points_of_intersection (n m : ℕ) : n = 3 → m = 2 → 
  (let circle_intersections := (n * (n - 1)) / 2 * 2;
       line_circle_intersections := m * n * 2;
       line_intersections := (m * (m - 1)) / 2;
       total_intersections := circle_intersections + line_circle_intersections + line_intersections
   in total_intersections = 19) :=
by
  intros hn hm
  let circle_intersections := (n * (n - 1)) / 2 * 2
  let line_circle_intersections := m * n * 2
  let line_intersections := (m * (m - 1)) / 2
  let total_intersections := circle_intersections + line_circle_intersections + line_intersections
  rw [hn, hm] at *
  calc total_intersections
      = ((3 * 2) * 2) + (2 * 3 * 2) + 1 : by sorry
      ... = 6 + 12 + 1  : by sorry
      ... = 19         : by sorry

end max_points_of_intersection_l28_28898


namespace find_t_l28_28326

theorem find_t (t : ℝ) : 
  (∀ x y : ℝ, (x = 1 - 3 * t) → (y = 2 * t - 3) → (x = y) → (t = 4 / 5)) :=
begin
  intros x y h1 h2 h3,
  sorry,
end

end find_t_l28_28326


namespace tangent_fraction_15_degrees_l28_28615

theorem tangent_fraction_15_degrees : (1 + Real.tan (Real.pi / 12 )) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tangent_fraction_15_degrees_l28_28615


namespace C1_parametric_eqn_C2_polar_eqn_max_PM_PN_value_l28_28257

-- Define the parametric equations for curve C1
def C1_parametric (θ : ℝ) : ℝ × ℝ := (2 * cos θ, sqrt 3 * sin θ)

-- Define the ordinary equation of curve C1 
def C1_ordinary (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define a condition that the parametric form of C1 satisfies the ordinary equation
theorem C1_parametric_eqn (θ : ℝ) : ∃ (x y : ℝ), C1_parametric θ = (x, y) ∧ C1_ordinary x y :=
by
  let (x, y) := C1_parametric θ
  use x, y
  sorry

-- Define the polar equation for curve C2
def C2_polar (ρ : ℝ) : Prop := ρ = 2

-- Define the rectangular coordinate equation for curve C2
def C2_rectangular (x y : ℝ) : Prop := (x^2 + y^2 = 4)

-- Define a condition that the polar form of C2 satisfies the rectangular coordinate equation
theorem C2_polar_eqn (α : ℝ) : ∀ x y, (x = 2 * cos α ∧ y = 2 * sin α) → C2_rectangular x y :=
by
  intro x y h
  cases h with hx hy
  rw [hx, hy]
  sorry

-- Define function to find maximum value of |PM| + |PN|
def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def max_PM_PN (α : ℝ) : ℝ :=
  let M : ℝ × ℝ := (0, sqrt 3)
  let N : ℝ × ℝ := (0, -sqrt 3)
  let P : ℝ × ℝ := (2 * cos α, 2 * sin α)
  distance P M + distance P N

theorem max_PM_PN_value : ∃ α : ℝ, max_PM_PN α = 2 * sqrt 7 :=
by
  use 0 -- α = 0 should give us the maximum value, we assume α is 0 for simplicity
  sorry

end C1_parametric_eqn_C2_polar_eqn_max_PM_PN_value_l28_28257


namespace no_real_pairs_in_arithmetic_progression_l28_28180

noncomputable def arithmetic_progression_pairs_count : ℕ :=
let pairs := {p : ℝ × ℝ | ∃ a b : ℝ, p = (a, b) ∧ a = (15 + b) / 2 ∧ a + a * b = 2 * b} in
  if h : pairs = ∅ then 0 else sorry

theorem no_real_pairs_in_arithmetic_progression :
  @arithmetic_progression_pairs_count = 0 :=
sorry

end no_real_pairs_in_arithmetic_progression_l28_28180


namespace transformation_matrix_correct_l28_28175
noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 3],
  ![-3, 0]
]

theorem transformation_matrix_correct :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![0, 1],
    ![-1, 0]
  ];
  let S : ℝ := 3;
  M = S • R :=
by
  sorry

end transformation_matrix_correct_l28_28175


namespace time_to_pass_bridge_l28_28082

-- Define the conditions as given in the problem
def train_length : ℕ := 360 -- in meters
def bridge_length : ℕ := 140 -- in meters
def speed_kmh : ℝ := 30 -- speed in km/h

-- Conversion factor for converting speed from km/h to m/s
def kmh_to_ms (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Calculate the total distance to be covered
def total_distance (train_length bridge_length: ℕ) : ℕ := train_length + bridge_length

-- Speed of the train in meters per second
def speed_ms : ℝ := kmh_to_ms speed_kmh

-- Proving the time taken to pass the bridge
theorem time_to_pass_bridge : (total_distance train_length bridge_length) / speed_ms = 60 := by
  sorry

end time_to_pass_bridge_l28_28082


namespace cos_identity_l28_28226

theorem cos_identity (α : ℝ) (h : cos (π / 3 + α) = 2 / 3) : cos (2 * π / 3 - α) = - (2 / 3) := 
by
  sorry

end cos_identity_l28_28226


namespace can_increase_averages_l28_28352

def grades_group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def grades_group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : Float :=
  (grades.map Nat.toFloat).sum / grades.length

def new_average (grades : List ℕ) (grade_to_remove_or_add : ℕ) (adding : Bool) : Float :=
  if adding
  then (grades.map Nat.toFloat).sum + grade_to_remove_or_add / (grades.length + 1)
  else (grades.map Nat.toFloat).sum - grade_to_remove_or_add / (grades.length - 1)

theorem can_increase_averages :
  ∃ grade,
    grade ∈ grades_group1 ∧
    average grades_group1 < new_average grades_group1 grade false ∧
    average grades_group2 < new_average grades_group2 grade true :=
  sorry

end can_increase_averages_l28_28352


namespace correct_calculation_B_l28_28076

theorem correct_calculation_B : 
  ∀ (a b x : ℝ),
    (a + 1) ^ 2 ≠ a ^ 2 + 1 ∧
    (x ^ 2 + x) / x = x + 1 ∧
    sqrt 12 + sqrt 3 ≠ sqrt 15 ∧
    (a - 4 * b) * (a + 4 * b) ≠ a ^ 2 - 4 * b ^ 2 :=
by
  intros
  split
  {
    assume a : ℝ
    calc
      (a + 1) ^ 2 = a ^ 2 + 2 * a + 1 : by ring
      ... ≠ a ^ 2 + 1 : by linarith,
  }
  split
  {
    assume x : ℝ
    calc
      (x ^ 2 + x) / x = x + 1 := by field_simp,
  }
  split
  {
    calc
      sqrt 12 + sqrt 3 ≠ sqrt 15 : by linarith,
  }
  {
    assume a b : ℝ
    calc
      (a - 4 * b) * (a + 4 * b) = a ^ 2 - 16 * b ^ 2 : by ring
      ... ≠ a ^ 2 - 4 * b ^ 2 : by linarith,
  }

end correct_calculation_B_l28_28076


namespace proof_problem_l28_28405

-- Definitions for the given conditions
variables (Ω : Type) [probability_space Ω] (A B : event Ω)

-- Condition: A and B are mutually exclusive events
def mutually_exclusive : Prop := A ∩ B = ∅

-- Condition: P(A) > 0
def P_A_positive : Prop := 0 < P(A)

-- Condition: P(B) > 0
def P_B_positive : Prop := 0 < P(B)

-- Lean proof statement to check which statements are correct
theorem proof_problem (h_exclusive : mutually_exclusive A B) 
                      (h_P_A_positive : P_A_positive A)
                      (h_P_B_positive : P_B_positive B) :
                       (P(A ∩ B) = 0) ∧ 
                       (P(A ∩ B) ≠ P(A) * P(B)) ∧
                       (P(A ∪ B) = P(A) + P(B)) ∧
                       (P(Aᶜ ∪ Bᶜ) = 1) :=
by sorry

end proof_problem_l28_28405


namespace parallel_sufficient_not_necessary_l28_28673

noncomputable theory

variable (a b : Vector ℝ) -- Assume a and b are vectors in a vector space over ℝ, ℝ-denoting real numbers.
variable (a_ne_zero : a ≠ 0) -- Assume a is non-zero.
variable (b_ne_zero : b ≠ 0) -- Assume b is non-zero.
variable (a_plus_2b_zero : a + 2 • b = 0) -- Assume a + 2b = 0.

theorem parallel_sufficient_not_necessary : a + 2 • b = 0 → (∃ (k : ℝ), a = k • b) ∧ ¬(∀ (k : ℝ), a = k • b) :=
sorry

end parallel_sufficient_not_necessary_l28_28673


namespace marbles_left_l28_28995

variables (Chris Ryan Alex : ℕ)

def total_marbles := Chris + Ryan + Alex

def share_Chris (total : ℕ) := total / 4
def share_Ryan (total : ℕ) := total / 4
def share_Alex (total : ℕ) := total / 3

def marbles_remaining (total : ℕ) := total - (share_Chris total + share_Ryan total + share_Alex total)

theorem marbles_left (hChris : Chris = 12) (hRyan : Ryan = 28) (hAlex : Alex = 18) :
  marbles_remaining (total_marbles Chris Ryan Alex) = 11 :=
by
  simp [total_marbles, marbles_remaining, share_Chris, share_Ryan, share_Alex, hChris, hRyan, hAlex]
  sorry

end marbles_left_l28_28995


namespace integral_of_quadratic_l28_28491

noncomputable def compute_integral : ℝ :=
  ∫ x in (0:ℝ)..(2:ℝ), 3*x^2 - 2*x + 1

theorem integral_of_quadratic (result : compute_integral = 6) :
  (∫ x in (0:ℝ)..(2:ℝ), 3*x^2 - 2*x + 1) = 6 :=
begin
  have h1 : ∫ x in (0:ℝ)..(2:ℝ), 3*x^2 - 2*x + 1 = result, 
  { refl, },
  rw h1,
  exact result,
end

end integral_of_quadratic_l28_28491


namespace marco_ice_cream_cones_needed_l28_28429
-- Given conditions
def price_per_cone : ℝ := 5
def expense_ratio : ℝ := 0.8
def profit_goal : ℝ := 200

-- Proof goal
theorem marco_ice_cream_cones_needed (
    price_per_cone_eq : price_per_cone = 5,
    expense_ratio_eq : expense_ratio = 0.8,
    profit_goal_eq : profit_goal = 200
) : let profit_ratio := 1 - expense_ratio,
        total_sales := profit_goal / profit_ratio,
        num_cones := total_sales / price_per_cone
    in num_cones = 200 :=
by
  -- Placeholder for the proof
  sorry

end marco_ice_cream_cones_needed_l28_28429


namespace min_value_of_function_l28_28494

theorem min_value_of_function : 
  ∃ m : ℝ, m = Inf (set.image (λ x : ℝ, 4^x - 2^(x + 2)) (set.Icc (-1:ℝ) (2:ℝ))) ∧ m = -4 := 
by 
  sorry

end min_value_of_function_l28_28494


namespace square_of_fourth_power_of_fourth_smallest_prime_l28_28537

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the square of the fourth power of that number
def square_of_fourth_power (n : ℕ) : ℕ := (n^4)^2

-- Prove the main statement
theorem square_of_fourth_power_of_fourth_smallest_prime : square_of_fourth_power fourth_smallest_prime = 5764801 :=
by
  sorry

end square_of_fourth_power_of_fourth_smallest_prime_l28_28537


namespace gnuff_tutor_minutes_l28_28284

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l28_28284


namespace tan_range_l28_28501

theorem tan_range :
  ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ Real.pi / 4 → -1 ≤ Real.tan x ∧ Real.tan x < 0 ∨ 0 < Real.tan x ∧ Real.tan x ≤ 1 :=
by
  sorry

end tan_range_l28_28501


namespace simplify_expression_l28_28464

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2*x - 2)/(x + 1)) / ((x^2 - x) / (2*x + 2)) = 2 - Real.sqrt 2 := 
by
  -- Here we should include the proof steps, but we skip it with "sorry"
  sorry

end simplify_expression_l28_28464


namespace regular_tetrahedrons_volume_ratio_l28_28363

theorem regular_tetrahedrons_volume_ratio (T1 T2 : SemiTopoSpace) (V1 V2 : ℝ) (e1 e2 : ℝ)
  (h1 : IsRegularTetrahedron T1) (h2 : IsRegularTetrahedron T2)
  (h3 : e1 / e2 = 1 / 2) : V1 / V2 = 1 / 8 :=
sorry

end regular_tetrahedrons_volume_ratio_l28_28363


namespace Megan_account_increase_l28_28814

theorem Megan_account_increase (P : ℝ) : 
  let initial_amount := 125
  let final_after_babysitting := initial_amount + (initial_amount * P / 100)
  let final_after_shoes := final_after_babysitting * 0.8
  final_after_shoes = initial_amount -> P = 25 :=
begin
  intros,
  let initial_amnt := 125,
  let final_after_babysitting := initial_amnt + (initial_amnt * P / 100),
  let final_after_shoes := final_after_babysitting * 0.8,
  have h1 : final_after_shoes = initial_amnt,
  by assumption,
  have h2 : final_after_shoes = (initial_amnt + (initial_amnt * P / 100)) * 0.8,
  by refl,
  rw h2 at h1,
  have h3 : initial_amnt = 125,
  by refl,
  rw h3 at h1,
  norm_num at h1,
  linarith,
  exact sorry
end

end Megan_account_increase_l28_28814


namespace find_f_7_l28_28695

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_7 (h_odd : ∀ x, f (-x) = -f x)
                 (h_periodic : ∀ x, f (x + 4) = f x)
                 (h_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x ^ 2) :
  f 7 = -2 := 
sorry

end find_f_7_l28_28695


namespace compute_mean_xy_l28_28306

theorem compute_mean_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : log y x + log x y = 11 / 3) (h4 : x * y = 128) :
  (x + y) / 2 = real.root 5 4 + 8 * real.root 5 16 := sorry

end compute_mean_xy_l28_28306


namespace find_limit_of_hours_l28_28105

def regular_rate : ℝ := 16
def overtime_rate (r : ℝ) : ℝ := r * 1.75
def total_compensation : ℝ := 920
def total_hours : ℝ := 50

theorem find_limit_of_hours : 
  ∃ (L : ℝ), 
    total_compensation = (regular_rate * L) + ((overtime_rate regular_rate) * (total_hours - L)) →
    L = 40 :=
by
  sorry

end find_limit_of_hours_l28_28105


namespace team_wins_seven_consecutive_matches_l28_28370

theorem team_wins_seven_consecutive_matches (teams : ℕ) (matches : ℕ → ℕ) (rest_days : ℕ → ℕ) 
    (victory : ℕ → ℕ → Prop) (h_teams : teams = 65) 
    (h_match_division : ∀ d, matches d = 32) 
    (h_rest_day : ∀ d, rest_days d = 1) 
    (h_victory : ∀ t₁ t₂, victory t₁ t₂ ∨ victory t₂ t₁) :
    ∃ t, ∃ d, (∀ i, 0 ≤ i ∧ i < 7 → victory t (matches (d + i))) :=
sorry

end team_wins_seven_consecutive_matches_l28_28370


namespace distance_from_dormitory_to_city_l28_28385

theorem distance_from_dormitory_to_city (D : ℝ) (h : (1/2) * D + (1/4) * D + 6 = D) : D = 24 :=
by
  sorry

end distance_from_dormitory_to_city_l28_28385


namespace millet_more_than_half_on_sixth_day_l28_28432

-- Definitions based on the conditions
def initial_seeds : ℝ := 1 / 2
def initial_millet_ratio : ℝ := 0.4
def daily_doubling (day : ℕ) : ℝ := 2^day

-- Birds eat 50% of millet and all of other seeds daily
def consumed_millet_ratio : ℝ := 0.5
def consumed_other_ratio : ℝ := 1.0

-- Statement: Prove that on the 6th day, more than half of the seeds are millet
theorem millet_more_than_half_on_sixth_day : 
  ∃ days_since_initial : ℕ, days_since_initial = 6 ∧ 
  let total_seeds := ∑ i in finset.range (days_since_initial + 1), daily_doubling i * initial_seeds,
  let millet_seeds := initial_seeds * initial_millet_ratio * (1 - consumed_millet_ratio ^ (days_since_initial + 1)) * (2 ^ days_since_initial),
  millet_seeds > total_seeds / 2 := 
by
  -- Definition processing and assertions
  sorry

end millet_more_than_half_on_sixth_day_l28_28432


namespace range_of_f_x_minus_2_l28_28319

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + 1 else if x > 0 then -(x + 1) else 0

theorem range_of_f_x_minus_2 :
  ∀ x : ℝ, f (x - 2) < 0 ↔ x ∈ Set.union (Set.Iio 1) (Set.Ioo 2 3) := by
sorry

end range_of_f_x_minus_2_l28_28319


namespace area_of_region_B_l28_28637

theorem area_of_region_B : 
  let B := { z : ℂ | (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1) ∧ 
                   (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
                    0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1) } in
  ∫ (z in B), 1 = 2500 - 312.5 * Real.pi := 
by
  sorry

end area_of_region_B_l28_28637


namespace max_intersection_points_circles_lines_l28_28904

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l28_28904


namespace max_happy_family_intervals_l28_28792

theorem max_happy_family_intervals {n : ℕ} (h : 0 < n) :
  ∃ (wp : set (ℕ × ℕ)), (∀ (i j : ℕ), (0 ≤ i) → (i < j) → (j ≤ n) → ((i, j) ∈ wp)) ∧
    (∀ (I₁ I₂ : (ℕ × ℕ)), (I₁ ∈ wp) → (I₂ ∈ wp) → ((fst I₁ ≠ fst I₂ ∨ snd I₁ ≠ snd I₂) → ((fst I₁ + snd I₁) ≠ (fst I₂ + snd I₂))) → 
    (I₁ ⊆ I₂ → (fst I₁ = fst I₂ ∨ snd I₁ = snd I₂))) →
  wp.card = Catalan n :=
sorry

end max_happy_family_intervals_l28_28792


namespace log_3_m_minus_log_1_div_3_n_lt_zero_l28_28710

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

theorem log_3_m_minus_log_1_div_3_n_lt_zero
  (b c m n : ℝ)
  (hf_sym : ∀ x, f (1 - x) b c = f (1 + x) b c)
  (hf_pos : f 0 b c > 0)
  (hf_roots : f m b c = 0 ∧ f n b c = 0)
  (h_neq : m ≠ n)
  (h_pos : m > 0 ∧ n > 0) :
  log 3 m - log (1/3) n < 0 :=
by
  sorry

end log_3_m_minus_log_1_div_3_n_lt_zero_l28_28710


namespace investment_amount_a_l28_28084

theorem investment_amount_a (b c : ℕ) (investment_period : ℕ) (total_profit c_share x : ℕ) 
  (h_b_investment : b = 1000) (h_c_investment : c = 1200) 
  (h_investment_period : investment_period = 2) 
  (h_total_profit : total_profit = 1000) 
  (h_c_share : c_share = 400) 
  (h_profit_eq : 2 * x + 2 * b + 2 * c = total_profit * (2 * c) / c_share) :
  x = 800 :=
by
  rw [h_b_investment, h_c_investment, h_investment_period] at h_profit_eq
  simp at h_profit_eq
  sorry

end investment_amount_a_l28_28084


namespace number_of_integer_solutions_eq_6_l28_28023

theorem number_of_integer_solutions_eq_6 :
  ∃ (solutions : Finset (ℤ × ℤ)), 
    (∀ p ∈ solutions, 
       let (x, y) := p in (|x| + 1) * (|y| - 3) = 5) ∧ 
    solutions.card = 6 := 
sorry

end number_of_integer_solutions_eq_6_l28_28023


namespace fraction_zero_iff_x_one_l28_28322

theorem fraction_zero_iff_x_one (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x - 5 ≠ 0) : x = 1 := by
  sorry

end fraction_zero_iff_x_one_l28_28322


namespace pitcher_percentage_l28_28883

theorem pitcher_percentage (C : ℝ) (hC : 0 < C) :
  let juice_in_pitcher := (3 / 4) * C in
  let juice_per_cup := juice_in_pitcher / 8 in
  100 * (juice_per_cup / C) = 9 :=
by
  let juice_in_pitcher := (3 / 4) * C
  let juice_per_cup := juice_in_pitcher / 8
  sorry

end pitcher_percentage_l28_28883


namespace problem_statement_inequality_bounds_l28_28171

noncomputable def a_n (n : ℕ) : ℝ :=
  (n^2 + 1) / real.sqrt (n^4 + 4)

noncomputable def b_n (n : ℕ) : ℝ :=
  ∏ i in finset.range(n) + 1, a_n (i + 1)

theorem problem_statement (n : ℕ) (hn : n ≥ 1) :
  b_n n = real.sqrt(2) * real.sqrt(n^2 + 1) / real.sqrt(n^2 + 2 * n + 2) :=
sorry

theorem inequality_bounds (n : ℕ) (hn : n ≥ 1) :
  (1 / (n + 1)^3 < (b_n n / real.sqrt (2) - n / (n + 1)) ∧ (b_n n / real.sqrt (2) - n / (n + 1)) < (1 / n^3)) :=
sorry

end problem_statement_inequality_bounds_l28_28171


namespace slope_best_fit_l28_28540

variable (x1 x2 x3 y1 y2 y3 : ℝ)
variable (h_order : x1 < x2 ∧ x2 < x3)
variable (h_spacing : x3 - x2 = x2 - x1)

theorem slope_best_fit (x1 x2 x3 y1 y2 y3 : ℝ) (h_order : x1 < x2 ∧ x2 < x3) (h_spacing : x3 - x2 = x2 - x1) :
  (y3 - y1) / (x3 - x1) = (∑ (i : Fin 3), (y3 - y2 - y1 + y2)) / (∑ (i : Fin 3), (x3 - x2 - x1 + x2)) :=
sorry

end slope_best_fit_l28_28540


namespace total_arrangements_l28_28939

theorem total_arrangements (students communities : ℕ) 
  (h_students : students = 5) 
  (h_communities : communities = 3)
  (h_conditions :
    ∀(student : Fin students) (community : Fin communities), 
      true 
  ) : 150 = 150 :=
by sorry

end total_arrangements_l28_28939


namespace part1_part2_l28_28935

variables {a b c r r1 r2 r3 R t s s1 s2 s3 : ℝ}

-- Core conditions. Assumption of existence and properties of variables.
def conditions : Prop := 
  (r = t / s) ∧ 
  (r1 = t / s1) ∧ 
  (r2 = t / s2) ∧ 
  (r3 = t / s3) ∧ 
  (s = (a + b + c) / 2) ∧ 
  (s1 = s - a) ∧ 
  (s2 = s - b) ∧ 
  (s3 = s - c) ∧ 
  (16 * R^2 = (r1 + r2 + r3 - r)^2)

-- Problem Part 1
theorem part1 (h : conditions) : a^2 + b^2 + c^2 + r^2 + r1^2 + r2^2 + r3^2 = 16 * R^2 :=
by sorry

-- Problem Part 2
theorem part2 (h : conditions) : (1 / r^2) + (1 / r1^2) + (1 / r2^2) + (1 / r3^2) = (a^2 + b^2 + c^2) / t^2 :=
by sorry

end part1_part2_l28_28935


namespace geometric_sequence_a9_l28_28759

theorem geometric_sequence_a9 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 2) 
  (h2 : a 4 = 8 * a 7) 
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (hq : q > 0) 
  : a 9 = 1 / 32 := 
by sorry

end geometric_sequence_a9_l28_28759


namespace find_angle_A_in_triangle_ABC_l28_28387

noncomputable def sin := Real.sin
noncomputable def sqrt := Real.sqrt

theorem find_angle_A_in_triangle_ABC :
  ∀ (a b c : ℝ) (A : ℝ),
    b = 8 →
    c = 8 * sqrt 3 →
    (1 / 2) * b * c * sin A = 16 * sqrt 3 →
    A = π / 6 :=
by
  intros a b c A hb hc harea
  sorry

end find_angle_A_in_triangle_ABC_l28_28387


namespace symmDiff_A_B_l28_28692

open Set

-- Define the sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | abs x < 2}

-- Define the set operations
def diff (U V : Set ℤ) : Set ℤ := {x | x ∈ U ∧ x ∉ V}
def symmDiff (U V : Set ℤ) : Set ℤ := diff U V ∪ diff V U

-- The theorem to be proved
theorem symmDiff_A_B : symmDiff A B = {-1, 0, 2} := by
  sorry

end symmDiff_A_B_l28_28692


namespace equilateral_triangle_fill_l28_28569

theorem equilateral_triangle_fill :
  let A (s: ℝ) := (real.sqrt 3 / 4) * s ^ 2 in
  let s_large := 12 in
  let s_small := 2 in
  A s_large / A s_small = 36 :=
by
  sorry

end equilateral_triangle_fill_l28_28569


namespace nate_matches_final_count_l28_28436

theorem nate_matches_final_count :
  ∀ (initial: ℕ) (dropped: ℕ),
    initial = 70 →
    dropped = 10 →
    ∃ final: ℕ, final = initial - dropped - 2*dropped ∧ final = 40 :=
by
  intros initial dropped h_initial h_dropped
  use initial - dropped - 2*dropped
  have h_final_eq := calc
    initial - dropped - 2*dropped = 70 - 10 - 2*10 : by rw [h_initial, h_dropped]
    ... = 40 : by norm_num
  exact ⟨h_final_eq, h_final_eq.symm⟩

end nate_matches_final_count_l28_28436


namespace can_increase_averages_l28_28353

def grades_group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def grades_group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : Float :=
  (grades.map Nat.toFloat).sum / grades.length

def new_average (grades : List ℕ) (grade_to_remove_or_add : ℕ) (adding : Bool) : Float :=
  if adding
  then (grades.map Nat.toFloat).sum + grade_to_remove_or_add / (grades.length + 1)
  else (grades.map Nat.toFloat).sum - grade_to_remove_or_add / (grades.length - 1)

theorem can_increase_averages :
  ∃ grade,
    grade ∈ grades_group1 ∧
    average grades_group1 < new_average grades_group1 grade false ∧
    average grades_group2 < new_average grades_group2 grade true :=
  sorry

end can_increase_averages_l28_28353


namespace find_fake_coin_l28_28529

theorem find_fake_coin (k : ℕ) : 
  ∃ (n : ℕ), n = 3k + 1 ∧ ∀ (coins : Fin (3^(2 * k)) → ℕ) (balances : Fin 3 → ℕ → ℕ → Prop), 
  (∀ i, ∃ (j₁ j₂ : Fin (3^(2 * k))), j₁ < j₂ ∧ 
       ((i ≠ 2 → balances i (coins j₁) (coins j₂) = (1 : ℤ)) ∨ 
       (i ≠ 1 → balances i (coins j₁) (coins j₂) ≠ (0 : ℤ)))
  → ∃ j, (coins j) = (0 : ℤ) :=
begin
  sorry
end

end find_fake_coin_l28_28529


namespace problem1_problem2_l28_28618

-- First Problem
theorem problem1 : 
  Real.cos (Real.pi / 3) + Real.sin (Real.pi / 4) - Real.tan (Real.pi / 4) = (-1 + Real.sqrt 2) / 2 :=
by
  sorry

-- Second Problem
theorem problem2 : 
  6 * (Real.tan (Real.pi / 6))^2 - Real.sqrt 3 * Real.sin (Real.pi / 3) - 2 * Real.cos (Real.pi / 4) = 1 / 2 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l28_28618


namespace coffee_bread_combinations_l28_28035

theorem coffee_bread_combinations : ∀ (num_coffees num_breads : ℕ), (num_coffees = 2) → (num_breads = 3) → num_coffees * num_breads = 6 :=
by
  intros num_coffees num_breads h_coffee h_bread
  rw [h_coffee, h_bread]
  exact Nat.mul_comm 2 3

end coffee_bread_combinations_l28_28035


namespace area_of_B_l28_28639

theorem area_of_B :
  let B := {z : ℂ | let x := z.re, y := z.im in 
                    (0 ≤ x ∧ x ≤ 50) ∧ 
                    (0 ≤ y ∧ y ≤ 50) ∧ 
                    (50 * x / (x^2 + y^2) ∈ Icc 0 1) ∧ 
                    (50 * y / (x^2 + y^2) ∈ Icc 0 1) }
  ∃ (area : ℝ), ∀ z ∈ B, area = 1875 - (625 * Real.pi) / 2 :=
by {
  sorry
}

end area_of_B_l28_28639


namespace permutation_complete_residue_system_iff_coprime_6_l28_28198

open Nat

theorem permutation_complete_residue_system_iff_coprime_6 (n : ℕ) :
  (∃ (p : Fin n → Fin n), 
    (∀ i : Fin n, ∃ j : Fin n, (p i + i) % n = j) ∧
    (∀ i : Fin n, ∃ k : Fin n, (p i - i + n) % n = k)) ↔ gcd n 6 = 1 := 
  sorry

end permutation_complete_residue_system_iff_coprime_6_l28_28198


namespace max_intersections_l28_28952

-- Definitions for the problem setting
variables {P1 P2 : Type} [convex_polygon P1] [convex_polygon P2]

noncomputable def sides (P : Type) [convex_polygon P] : ℕ := sorry
noncomputable def contained_within (P1 P2 : Type) [convex_polygon P1] [convex_polygon P2] : Prop := sorry
noncomputable def max_intersection_points (P1 P2 : Type) [convex_polygon P1] [convex_polygon P2] : ℕ := sorry

-- Conditions from the problem
variables (n1 n2 : ℕ) (h1 : sides P1 = n1) (h2 : sides P2 = n2) (h_le : n1 ≤ n2)
(h_contained : contained_within P1 P2)

-- The theorem to be proven
theorem max_intersections (P1 P2 : Type) [convex_polygon P1] [convex_polygon P2]
  (h1 : sides P1 = n1) (h2 : sides P2 = n2) (h_le : n1 ≤ n2)
  (h_contained : contained_within P1 P2) :
  max_intersection_points P1 P2 = n1 * n2 :=
by sorry

end max_intersections_l28_28952


namespace price_reduction_l28_28100

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l28_28100


namespace find_b_coordinates_find_angle_ac_l28_28243

variables (a b c : EuclideanSpace ℝ 2)

-- Assuming the given conditions:
noncomputable def a : EuclideanSpace ℝ 2 := ![1, 2]
axiom norm_b : ∥b∥ = 3 * real.sqrt 5
axiom parallel_ab : ∃ k : ℝ, b = k • a
axiom norm_c : ∥c∥ = real.sqrt 10
axiom perpendicular : (2 • a + c) ⬝ (4 • a - 3 • c) = 0

-- 1. Finding the coordinates of vector b
theorem find_b_coordinates : 
  b = ![3, 6] ∨ b = ![-3, -6] :=
sorry

-- 2. Finding the angle between vectors a and c
theorem find_angle_ac :
  real.arccos ((a ⬝ c) / (∥a∥ * ∥c∥)) = real.pi / 4 :=
sorry

end find_b_coordinates_find_angle_ac_l28_28243


namespace no_xy_term_l28_28073

theorem no_xy_term (k : ℝ) : (P : ℝ → ℝ → ℝ := λ x y, x^2 + (k-1)*x*y - 3*y^2 - 2*x*y - 5) → 
  (∀ x y, P x y = x^2 - 3 * y^2 - 5) → 
  k = 3 := by
  sorry

end no_xy_term_l28_28073


namespace determine_b_value_l28_28184

noncomputable def b_satisfies_condition (b : ℝ) : Prop :=
  (1 / real.log b / real.log 3) + (1 / real.log b / real.log 5) + (1 / real.log b / real.log 6) = 1

theorem determine_b_value : ∃ b : ℝ, b_satisfies_condition b ∧ b = 90 :=
begin
  use 90,
  split,
  { 
    -- Here you would prove the condition
    sorry
  },
  { 
    -- Here you simply prove b = 90
    refl
  }
end

end determine_b_value_l28_28184


namespace point_coordinates_l28_28315

-- Defining the point structure and a condition that fits our problem setting
structure Point where
  x : Int
  y : Int

-- The conditions as definitions in Lean
def distance_to_x_axis (P : Point) : Nat := abs P.y
def distance_to_y_axis (P : Point) : Nat := abs P.x

-- The theorem stating the equivalence problem
theorem point_coordinates (P : Point) (h1 : distance_to_x_axis P = 3) (h2 : distance_to_y_axis P = 4) : P = ⟨4, 3⟩ :=
by
  -- Proof steps will go here
  sorry

end point_coordinates_l28_28315


namespace smaller_cuboid_length_l28_28950

-- Definitions based on conditions
def original_cuboid_volume : ℝ := 18 * 15 * 2
def smaller_cuboid_volume (L : ℝ) : ℝ := 4 * 3 * L
def smaller_cuboids_total_volume (L : ℝ) : ℝ := 7.5 * smaller_cuboid_volume L

-- Theorem statement
theorem smaller_cuboid_length :
  ∃ L : ℝ, smaller_cuboids_total_volume L = original_cuboid_volume ∧ L = 6 := 
by
  sorry

end smaller_cuboid_length_l28_28950


namespace parallelogram_exists_l28_28686

noncomputable def construct_parallelogram (A B C : Point) : Point := sorry

theorem parallelogram_exists (A B C : Point) : 
  let D := construct_parallelogram A B C in
  is_parallelogram A B C D :=
sorry

end parallelogram_exists_l28_28686


namespace regular_tetrahedrons_volume_ratio_l28_28362

theorem regular_tetrahedrons_volume_ratio (T1 T2 : SemiTopoSpace) (V1 V2 : ℝ) (e1 e2 : ℝ)
  (h1 : IsRegularTetrahedron T1) (h2 : IsRegularTetrahedron T2)
  (h3 : e1 / e2 = 1 / 2) : V1 / V2 = 1 / 8 :=
sorry

end regular_tetrahedrons_volume_ratio_l28_28362


namespace log_comparison_l28_28679

theorem log_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < e) (h3 : 0 < b) (h4 : b < e) (h5 : a < b) :
  a * Real.log b > b * Real.log a := sorry

end log_comparison_l28_28679


namespace number_of_correct_propositions_l28_28645

noncomputable def directed_distance (A B C x₀ y₀ : ℝ) : ℝ := (A * x₀ + B * y₀ + C) / (Real.sqrt (A ^ 2 + B ^ 2))

variables {A B C x₀ y₀ x₁ y₁ x₂ y₂ : ℝ} (h : A ^ 2 + B ^ 2 ≠ 0)

def d1 := directed_distance A B C x₁ y₁
def d2 := directed_distance A B C x₂ y₂

theorem number_of_correct_propositions :
  let num_correct := (if d1 - d2 = 0 then 0 else 0) +
                     (if d1 + d2 = 0 then 0 else 0) +
                     (if (d1 + d2 = 0) then 0 else 0) +
                     (if d1 * d2 < 0 then 1 else 0)
  in num_correct = 1 :=
by
  sorry

end number_of_correct_propositions_l28_28645


namespace bike_ride_ratio_l28_28988

theorem bike_ride_ratio (J : ℕ) (B : ℕ) (M : ℕ) (hB : B = 17) (hM : M = J + 10) (hTotal : B + J + M = 95) :
  J / B = 2 :=
by
  sorry

end bike_ride_ratio_l28_28988


namespace evaluate_at_two_l28_28070

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem evaluate_at_two : f 2 = 15 :=
by
  sorry

end evaluate_at_two_l28_28070


namespace probability_of_interval_l28_28581

noncomputable def normal_probability (μ σ : ℝ) (x : ℝ) : ℝ :=
  1 / (2 * Real.pi * σ^2).sqrt * Real.exp (-(x - μ)^2 / (2 * σ^2))

theorem probability_of_interval (μ σ : ℝ) (hμ : μ = 40) (hP : ∫ x in -∞..30, normal_probability μ σ x = 0.2) :
  (∫ x in 30..50, normal_probability μ σ x) = 0.6 := 
by
  sorry

end probability_of_interval_l28_28581


namespace isosceles_triangle_perimeter_l28_28152

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6) (h₂ : b = 5) :
  ∃ p : ℝ, (p = a + a + b ∨ p = b + b + a) ∧ (p = 16 ∨ p = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l28_28152


namespace hyperbola_properties_l28_28699

-- Definitions based on the conditions given in the problem
def is_hyperbola_point (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 = 1

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def hyperbola_foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let c := real.sqrt (a^2 + b^2)
  in ((-c, 0), (c, 0))

def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- The theorem based on the conclusions derived in the solution
theorem hyperbola_properties (a b : ℝ) (h₁ : a^2 = 2) (h₂ : b^2 = 1) :
  let c := real.sqrt (a^2 + b^2) in
  distance (hyperbola_foci a b).fst (hyperbola_foci a b).snd = 2 * c ∧
  eccentricity a c = real.sqrt 6 / 2 :=
by
  sorry

end hyperbola_properties_l28_28699


namespace triangle_construction_possible_l28_28642

theorem triangle_construction_possible (BC AA_1 : ℝ) (D : Point) 
  (H1 : midpoint BC = D)
  (H2 : perpendicular at D to BC)
  (H3 : ∀ F G : Point, DF = AA_1 ∧ DG = AA_1)
  (H4 : ∀ circle : Circle, circle.passes_through B C F)
  (H5 : ∀ line : Line, line.passes_through G ∧ line.parallel_to BC)
  (H6 : ∃ A : Point, circle.intersection_with line = A)
  : BC ≥ 2 * AA_1 := 
sorry

end triangle_construction_possible_l28_28642


namespace parabola_focus_l28_28211

theorem parabola_focus (x y : ℝ) : 
  let a := 2
  let b := 8
  let c := -1 in
  (y = a * x^2 + b * x + c) →
  let h := -b / (2 * a)
  let k := a * (h^2) + c in
  ∃ f_x f_y, f_x = h ∧ f_y = k + 1 / (4 * a) ∧ (f_x, f_y) = (-2, -71 / 8) :=
begin
  sorry
end

end parabola_focus_l28_28211


namespace shots_cost_l28_28629

theorem shots_cost (n_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) :
  n_dogs = 3 →
  puppies_per_dog = 4 →
  shots_per_puppy = 2 →
  cost_per_shot = 5 →
  n_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    3 * 4 * 2 * 5 = 12 * 2 * 5 : by rfl
    ... = 24 * 5 : by rfl
    ... = 120 : by rfl

end shots_cost_l28_28629


namespace can_transfer_increase_average_l28_28355

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l28_28355


namespace representation_of_2015_l28_28456

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l28_28456


namespace tom_gets_correct_share_l28_28434

def total_savings : ℝ := 18500.0
def natalie_share : ℝ := 0.35 * total_savings
def remaining_after_natalie : ℝ := total_savings - natalie_share
def rick_share : ℝ := 0.30 * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share
def lucy_share : ℝ := 0.40 * remaining_after_rick
def remaining_after_lucy : ℝ := remaining_after_rick - lucy_share
def minimum_share : ℝ := 1000.0
def tom_share : ℝ := remaining_after_lucy

theorem tom_gets_correct_share :
  (natalie_share ≥ minimum_share) ∧ (rick_share ≥ minimum_share) ∧ (lucy_share ≥ minimum_share) →
  tom_share = 5050.50 :=
by
  sorry

end tom_gets_correct_share_l28_28434


namespace sum_eleven_to_zero_l28_28987

theorem sum_eleven_to_zero :
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n ∧ n ≤ 11 → (f n = n ∨ f n = -n)) ∧ (∑ n in Finset.range 12, f n) = 0 :=
sorry

end sum_eleven_to_zero_l28_28987


namespace percent_did_not_take_cars_l28_28822

variable (P : ℝ) -- Total number of passengers on the ship
variable (round_trip_car : ℝ) -- Percent of passengers who held round-trip tickets and took their cars aboard
variable (total_round_trip : ℝ) -- Percent of passengers who held round-trip tickets in total

-- Given Conditions:
def condition1 : Prop := round_trip_car = 0.25 * P
def condition2 : Prop := total_round_trip = 0.625 * P

-- Theorem to prove:
theorem percent_did_not_take_cars (h1 : condition1 round_trip_car) (h2 : condition2 total_round_trip) :
  0.375 * P = total_round_trip - round_trip_car :=
by
  sorry

end percent_did_not_take_cars_l28_28822


namespace contradiction_proof_l28_28527

theorem contradiction_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h1 : a + 1/b < 2) (h2 : b + 1/c < 2) (h3 : c + 1/a < 2) : 
  ¬ (a + 1/b ≥ 2 ∨ b + 1/c ≥ 2 ∨ c + 1/a ≥ 2) :=
by
  sorry

end contradiction_proof_l28_28527


namespace problem_l28_28799

def sequence (s : ℕ → ℝ) : Prop :=
  ∀ n, s n = 2 * s (n - 1) + s (n - 2)

def special_sequence (s : ℕ → ℝ) : ℝ :=
  s 0 + s 1 * x + s 2 * x^2 + s 3 * x^3 + s 4 * x^4 + s 5 * x^5 + s 6 * x^6 + ...

theorem problem (s : ℕ → ℝ) (h : sequence s) : 
  ∀ n, s n^2 + s (n + 1)^2 = s (2n + 2) :=
sorry

end problem_l28_28799


namespace carson_total_seed_fertilizer_l28_28161

-- Definitions based on the conditions
variable (F S : ℝ)
variable (h_seed : S = 45)
variable (h_relation : S = 3 * F)

-- Theorem stating the total amount of seed and fertilizer used
theorem carson_total_seed_fertilizer : S + F = 60 := by
  -- Use the given conditions to relate and calculate the total
  sorry

end carson_total_seed_fertilizer_l28_28161


namespace percentage_per_annum_is_12_l28_28854

noncomputable def percentage_per_annum (BG TD : ℝ) : ℝ :=
  let BD := BG + TD
  let F := (BD * 100) / R
  let equation := (F * R = 61.6 * 100) ∧ (F * R = 55 * (100 + R))
  if equation then 12 else sorry

theorem percentage_per_annum_is_12 :
  percentage_per_annum 6.6 55 = 12 := 
by { sorry }

end percentage_per_annum_is_12_l28_28854


namespace towel_area_decrease_28_percent_l28_28972

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := L * 0.80

def new_breadth (B : ℝ) : ℝ := B * 0.90

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

def percentage_decrease_in_area (L B : ℝ) : ℝ :=
  ((original_area L B - new_area L B) / original_area L B) * 100

theorem towel_area_decrease_28_percent (L B : ℝ) :
  percentage_decrease_in_area L B = 28 := by
  sorry

end towel_area_decrease_28_percent_l28_28972


namespace B_days_to_complete_work_l28_28560

theorem B_days_to_complete_work (B : ℕ) (hB : B ≠ 0)
  (A_work_days : ℕ := 9) (combined_days : ℕ := 6)
  (work_rate_A : ℚ := 1 / A_work_days) (work_rate_combined : ℚ := 1 / combined_days):
  (1 / B : ℚ) = work_rate_combined - work_rate_A → B = 18 :=
by
  intro h
  sorry

end B_days_to_complete_work_l28_28560


namespace price_reduction_equation_l28_28104

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l28_28104


namespace age_difference_proof_l28_28598

def AlexAge : ℝ := 16.9996700066
def AlexFatherAge (A : ℝ) (F : ℝ) : Prop := F = 2 * A + 4.9996700066
def FatherAgeSixYearsAgo (A : ℝ) (F : ℝ) : Prop := A - 6 = 1 / 3 * (F - 6)

theorem age_difference_proof :
  ∃ (A F : ℝ), A = 16.9996700066 ∧
  (AlexFatherAge A F) ∧
  (FatherAgeSixYearsAgo A F) :=
by
  sorry

end age_difference_proof_l28_28598


namespace calculate_expression_l28_28611

theorem calculate_expression : (3^18 / 27^3 * 9) = 177147 := by
  -- Assume 27 = 3 ^ 3 and 9 = 3 ^ 2
  let x := 27
  let y := 9
  have h1 : x = 3 ^ 3 := rfl
  have h2 : y = 3 ^ 2 := rfl
  
  -- Main proof step with exponent rules assumption
  have h3 : (3^18 / x^3 * y) = 177147 := sorry
  
  -- Conclude that (3^18 / 27^3 * 9) = 177147
  exact h3

end calculate_expression_l28_28611


namespace labeling_possible_iff_odd_l28_28517

theorem labeling_possible_iff_odd (n : ℕ) (h : 2 < n) : 
  (∃ (label : ℕ → ℕ), 
    (∀ i < 2 * n, label i + label ((i + 1) % (2 * n)) = label ((i + n) % (2 * n)) + label ((i + n + 1) % (2 * n)))) ↔ 
  (odd n) :=
sorry

end labeling_possible_iff_odd_l28_28517


namespace completing_the_square_l28_28528

theorem completing_the_square (x : ℝ) : (x^2 - 6*x + 7 = 0) → ((x - 3)^2 = 2) :=
by
  intro h
  sorry

end completing_the_square_l28_28528


namespace max_d_in_52d_55e_l28_28197

theorem max_d_in_52d_55e :
  ∃ (d : ℕ), (∀ (e : ℕ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ e ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
    (520000 + 10000 * d + 550 + 10 * e) % 13 = 0) ∧
  (∀ (d' ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 
    (∀ (e : ℕ), e ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
      (520000 + 10000 * d' + 550 + 10 * e) % 13 = 0) → d' ≤ d) :=
begin
  use 6,
  sorry
end

end max_d_in_52d_55e_l28_28197


namespace range_of_given_function_l28_28708

noncomputable def function_no_zeros_extremes (ω : ℝ) (hω : ω > 0) : Prop :=
  ∀ {x : ℝ}, (0 < x) ∧ (x < π) →
  (f x = 0 → False) ∧ ((3 * cos (π * ω) + 4 * sin (π * ω)) = 5)

noncomputable def range_of_function (ω : ℝ) (hω : ω > 0) (f : ℝ → ℝ) : Set ℝ :=
  {y : ℝ | ∃ x, f x = y}

theorem range_of_given_function :
  ∀ (ω : ℝ) (hω : ω > 0),
  function_no_zeros_extremes ω hω →
  range_of_function ω hω (λ x, 3 * cos (π * ω) + 4 * sin (π * ω)) = Set.Icc (7 / 5) 5 :=
sorry

end range_of_given_function_l28_28708


namespace probability_one_white_ball_initial_find_n_if_one_red_ball_l28_28557

-- Define the initial conditions: 5 red balls and 3 white balls
def initial_red_balls := 5
def initial_white_balls := 3
def total_initial_balls := initial_red_balls + initial_white_balls

-- Define the probability of drawing exactly one white ball initially
def prob_draw_one_white := initial_white_balls / total_initial_balls

-- Define the number of white balls added
variable (n : ℕ)

-- Define the total number of balls after adding n white balls
def total_balls_after_adding := total_initial_balls + n

-- Define the probability of drawing exactly one red ball after adding n white balls
def prob_draw_one_red := initial_red_balls / total_balls_after_adding

-- Prove that the probability of drawing one white ball initially is 3/8
theorem probability_one_white_ball_initial : prob_draw_one_white = 3 / 8 := by
  sorry

-- Prove that, if the probability of drawing one red ball after adding n white balls is 1/2, then n = 2
theorem find_n_if_one_red_ball : prob_draw_one_red = 1 / 2 -> n = 2 := by
  sorry

end probability_one_white_ball_initial_find_n_if_one_red_ball_l28_28557


namespace total_amount_paid_l28_28927

theorem total_amount_paid (grapes_kg mangoes_kg rate_grapes rate_mangoes : ℕ) 
    (h1 : grapes_kg = 8) (h2 : mangoes_kg = 8) 
    (h3 : rate_grapes = 70) (h4 : rate_mangoes = 55) : 
    (grapes_kg * rate_grapes + mangoes_kg * rate_mangoes) = 1000 :=
by
  sorry

end total_amount_paid_l28_28927


namespace same_functions_l28_28149

def f_A1 (x : ℝ) := 2 * log 2 x
def f_A2 (x : ℝ) := log 2 (x^2)

def f_B1 (x : ℝ) := sqrt (x^2)
def f_B2 (x : ℝ) := (sqrt x)^2

def f_C1 (x : ℝ) := x
def f_C2 (x : ℝ) := log 2 (2^x)

def f_D1 (x : ℝ) := sqrt (x^2 - 4)
def f_D2 (x : ℝ) := sqrt (x - 2) * sqrt (x + 2)

theorem same_functions :
    (∀ x, f_C1 x = f_C2 x) ∧ 
    (¬∀ x, f_A1 x = f_A2 x) ∧
    (¬∀ x, f_B1 x = f_B2 x) ∧
    (¬∀ x, f_D1 x = f_D2 x) :=
  by sorry

end same_functions_l28_28149


namespace divisor_of_99_l28_28574

def reverse_digits (n : ℕ) : ℕ :=
  -- We assume a placeholder definition for reversing the digits of a number
  sorry

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
  sorry

end divisor_of_99_l28_28574


namespace car_2_speed_proof_l28_28543

noncomputable def car_1_speed : ℝ := 30
noncomputable def car_1_start_time : ℝ := 9
noncomputable def car_2_start_delay : ℝ := 10 / 60
noncomputable def catch_up_time : ℝ := 10.5
noncomputable def car_2_start_time : ℝ := car_1_start_time + car_2_start_delay
noncomputable def travel_duration : ℝ := catch_up_time - car_2_start_time
noncomputable def car_1_head_start_distance : ℝ := car_1_speed * car_2_start_delay
noncomputable def car_1_travel_distance : ℝ := car_1_speed * travel_duration
noncomputable def total_distance : ℝ := car_1_head_start_distance + car_1_travel_distance
noncomputable def car_2_speed : ℝ := total_distance / travel_duration

theorem car_2_speed_proof : car_2_speed = 33.75 := 
by 
  sorry

end car_2_speed_proof_l28_28543


namespace num_paths_P_to_Q_l28_28984

-- Definitions based on given conditions
def board_size : ℕ := 8

inductive color
| black
| white

-- Function to determine the color of a square
def square_color (row col : ℕ) : color :=
  if (row + col) % 2 = 0 then color.black else color.white

-- Predicate to say column and row is white
def is_white (row col : ℕ) : Prop :=
  square_color row col = color.white

-- Function to calculate the number of ways to reach a given square on the top row from the bottom row
def num_paths_to_top (bottom_row_col top_row_col : ℕ) : ℕ :=
  -- Initialize array ensuring all positions follow the adjacency rule of white squares only
  let init : array board_size (array board_size ℕ) := 
    (mk_array board_size
      (mk_array board_size 0)).setd bottom_row_col 1 in
  let paths := list.foldl
    (λ paths row, 
      (paths.getd row).modify row.succ $
        λ col,
        if is_white (row + 1) col then
          (paths.getd (row + 1)).getd (col - 1).getd col +
          (paths.getd (row + 1)).getd (col + 1).getd col
        else 0)
    init 
    (list.range (board_size - 1))
  in paths.getd (board_size - 1) top_row_col

-- Theorem to show number of distinct paths from P in bottom row to Q in top row is 28
theorem num_paths_P_to_Q : num_paths_to_top 0 7 = 28 :=
  sorry

end num_paths_P_to_Q_l28_28984


namespace continuous_ineq_example_l28_28547

open Real

theorem continuous_ineq_example
  (f : ℝ → ℝ)
  (h_cont : ∀ ε > 0, ∃ δ > 0, ∀ x y ∈ Ioi 0, abs (x - y) < δ → abs (f x - f y) < ε)
  (h_ineq : ∀ x > 0, f (x^2) ≥ f x)
  (h_value : f 1 = 5) :
  ∀ x > 0, f x ≥ 5 := 
sorry

end continuous_ineq_example_l28_28547


namespace length_of_train_l28_28083

noncomputable def train_speed_kmph : ℝ := 54
noncomputable def crossing_time_sec : ℝ := 9
noncomputable def speed_conversion_factor : ℝ := 5 / 18

theorem length_of_train (speed : ℝ) (time : ℝ) (conversion_factor : ℝ) :
  speed = 54 → time = 9 → conversion_factor = 5 / 18 → (speed * conversion_factor * time) = 135 :=
by
  intros
  calc
    (speed * conversion_factor * time) = (54 * (5 / 18) * 9) : by { rw [←h, ←h_1, ←h_2] }
    ... = 135 : by sorry

end length_of_train_l28_28083


namespace num_factors_720_l28_28723

theorem num_factors_720 : 
  let factors_720 := 30 in
  factors_720 = ∏ (e : ℕ) in {4, 2, 1}.image (λ e => e + 1), e
:= sorry

end num_factors_720_l28_28723


namespace can_increase_averages_by_transfer_l28_28343

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ := grades.sum / grades.length

theorem can_increase_averages_by_transfer :
    ∃ (student : ℝ) (new_group1_grades new_group2_grades : List ℝ),
      student ∈ group1_grades ∧
      new_group1_grades = (group1_grades.erase student) ∧
      new_group2_grades = student :: group2_grades ∧
      average new_group1_grades > average group1_grades ∧ 
      average new_group2_grades > average group2_grades :=
by
  sorry

end can_increase_averages_by_transfer_l28_28343


namespace fewer_VIP_tickets_sold_l28_28924

variable (V G : ℕ)

-- Definitions: total number of tickets sold and the total revenue from tickets sold
def total_tickets : Prop := V + G = 320
def total_revenue : Prop := 45 * V + 20 * G = 7500

-- Definition of the number of fewer VIP tickets than general admission tickets
def fewer_VIP_tickets : Prop := G - V = 232

-- The theorem to be proven
theorem fewer_VIP_tickets_sold (h1 : total_tickets V G) (h2 : total_revenue V G) : fewer_VIP_tickets V G :=
sorry

end fewer_VIP_tickets_sold_l28_28924


namespace represent_2015_l28_28459

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def in_interval (n : ℕ) : Prop := 400 < n ∧ n < 500

def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem represent_2015 :
  ∃ a b c : ℕ,
  a + b + c = 2015 ∧
  is_prime a ∧
  is_divisible_by_3 b ∧
  in_interval c ∧
  not_divisible_by_3 c :=
by {
  use 7,
  use 1605,
  use 403,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
}

end represent_2015_l28_28459


namespace inscribed_circle_radius_of_DEF_l28_28064

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l28_28064


namespace maximize_profit_for_prism_base_design_l28_28894

theorem maximize_profit_for_prism_base_design :
  ∃x : ℝ, let side_length := 2 in
          let height := 2 in
          let construction_cost_per_m2 := 1000 in
          let storage_value_per_m3 := 2500 in
          let OA_1 := 1 + real.cot x in
          let triangle_area := (side_length * OA_1) / 2 in
          let volume := height * triangle_area in
          let surface_area := 2 * triangle_area + 4 / real.sin x in
          let profit := storage_value_per_m3 * volume - construction_cost_per_m2 * surface_area in
          ∀ (x ≠ 0) (x < π / 2) (cos x = 3 / 4),
          1416 = 4 * profit :=
begin
  sorry
end

end maximize_profit_for_prism_base_design_l28_28894


namespace sum_numbers_greater_than_4_l28_28523

theorem sum_numbers_greater_than_4 (cards : List ℕ) (h : cards = [7, 3, 5]) : List.sum (cards.filter (λ x, x > 4)) = 12 :=
  by
  -- Proof goes here
  sorry

end sum_numbers_greater_than_4_l28_28523


namespace maximize_sector_area_l28_28744

theorem maximize_sector_area :
  (∀ (r l : ℝ), 2 * r + l = 36 ∧ S = 1 / 2 * l * r ∧ α = l / r → α = 2) :=
by
  intros r l h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end maximize_sector_area_l28_28744


namespace bikes_in_parking_lot_l28_28359

theorem bikes_in_parking_lot (C : ℕ) (Total_Wheels : ℕ) (Wheels_per_car : ℕ) (Wheels_per_bike : ℕ) (h1 : C = 14) (h2 : Total_Wheels = 76) (h3 : Wheels_per_car = 4) (h4 : Wheels_per_bike = 2) : 
  ∃ B : ℕ, 4 * C + 2 * B = Total_Wheels ∧ B = 10 :=
by
  sorry

end bikes_in_parking_lot_l28_28359


namespace sqrt_interval_int_count_l28_28296

theorem sqrt_interval_int_count : 
  let lower_bound := Real.ceil (Real.sqrt 12)
  let upper_bound := Real.floor (Real.sqrt 75)
  (upper_bound - lower_bound + 1 = 5) :=
by
  let lower_bound := Real.ceil (Real.sqrt 12)
  let upper_bound := Real.floor (Real.sqrt 75)
  have h1 : Real.sqrt 12 ≈ 3.464 := by sorry
  have h2 : Real.sqrt 75 ≈ 8.660 := by sorry
  have h3 : lower_bound = 4 := by sorry
  have h4 : upper_bound = 8 := by sorry
  have h5 : upper_bound - lower_bound + 1 = 5 := by sorry
  exact h5

end sqrt_interval_int_count_l28_28296


namespace Rogers_age_more_than_twice_Jills_age_l28_28462

/--
Jill is 20 years old.
Finley is 40 years old.
Roger's age is more than twice Jill's age.
In 15 years, the age difference between Roger and Jill will be 30 years less than Finley's age.
Prove that Roger's age is 5 years more than twice Jill's age.
-/
theorem Rogers_age_more_than_twice_Jills_age 
  (J F : ℕ) (hJ : J = 20) (hF : F = 40) (R x : ℕ)
  (hR : R = 2 * J + x) 
  (age_diff_condition : (R + 15) - (J + 15) = (F + 15) - 30) :
  x = 5 := 
sorry

end Rogers_age_more_than_twice_Jills_age_l28_28462


namespace slips_with_number_3_l28_28007

theorem slips_with_number_3 (total_slips : ℕ) (number_on_slip : ℕ → ℝ) (expected_value : ℝ) (y : ℕ) :
  total_slips = 15 →
  (∀ n, n = 3 ∨ n = 8) →
  expected_value = 5 →
  ((number_on_slip 3) * (y / total_slips.to_real) + 
   (number_on_slip 8) * ((total_slips - y).to_real / total_slips.to_real) / expected_value) :=
begin
  sorry
end

end slips_with_number_3_l28_28007


namespace max_intersection_points_l28_28909

theorem max_intersection_points (C L : Type) [fintype C] [fintype L] (hC : fintype.card C = 3) (hL : fintype.card L = 2) :
  (∑ c1 in C, ∑ c2 in (C \ {c1}), 2) + (∑ l in L, ∑ c in C, 2) + 1 = 19 :=
by
  sorry

end max_intersection_points_l28_28909


namespace largest_possible_integer_in_list_l28_28120

-- Definitions of the conditions
def list_of_six (l : list ℕ) : Prop := l.length = 6
def contains_no_repeats_except_seven (l : list ℕ) : Prop := (∀ n : ℕ, n ≠ 7 → l.count n ≤ 1) ∧ l.count 7 = 2
def list_median_is_ten (l : list ℕ) : Prop := 
  ∀ a b c d e f : ℕ, (l = [a, b, c, d, e, f]) → (a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ f) → (c + d) = 20
def list_mean_is_twelve (l : list ℕ) : Prop := l.sum = 72

-- The theorem to prove
theorem largest_possible_integer_in_list : 
  ∀ (l : list ℕ), list_of_six l → contains_no_repeats_except_seven l →
  list_median_is_ten l → list_mean_is_twelve l →
  ∃ (x : ℕ), x ∈ l ∧ ∀ y ∈ l, y ≤ x ∧ x = 27 :=
by
  sorry

end largest_possible_integer_in_list_l28_28120


namespace cos_sum_identity_l28_28926

theorem cos_sum_identity (α x : ℝ) (n : ℕ) :
  (∑ k in Finset.range (n + 1), Real.cos (α + k * x)) =
  (Real.sin (α + (n + 0.5) * x) - Real.sin (α - 0.5 * x)) / (2 * Real.sin (0.5 * x)) :=
sorry

end cos_sum_identity_l28_28926


namespace largest_possible_median_of_extended_list_l28_28652

open List

theorem largest_possible_median_of_extended_list :
  ∃ l : List ℕ, (∀ x ∈ l, 0 < x) ∧ l = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] ++ (List.repeat 24 4) 
  → median l = 17 :=
by
  sorry

end largest_possible_median_of_extended_list_l28_28652


namespace augmented_matrix_solution_l28_28700

theorem augmented_matrix_solution (c1 c2 : ℚ) 
    (h1 : 2 * (3 : ℚ) + 3 * (5 : ℚ) = c1)
    (h2 : (5 : ℚ) = c2) : 
    c1 - c2 = 16 := 
by 
  sorry

end augmented_matrix_solution_l28_28700


namespace inequality_abc_l28_28421

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) + 
  Real.sqrt (c^2 - c*a + a^2) + 9 * Real.cbrt (a*b*c) ≤ 4 * (a + b + c) :=
by
  sorry

end inequality_abc_l28_28421


namespace equations_of_ellipse_and_circle_range_of_AB_l28_28235

noncomputable theory

def ellipse_equation (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def circle_equation (r : ℝ) := ∀ x y : ℝ, x^2 + y^2 = r^2
def tangent_line_intersection (l : ℝ) := ∀ x y : ℝ, -- TODO: Define tangent line intersection condition

theorem equations_of_ellipse_and_circle (a b r : ℝ) (h1 : a > b) (h2 : b > 0) 
(h3 : a > 0) (h4 : b = √3/2 * a) (h5 : r = √(a^2 - b^2)) :
ellipse_equation 2 √3 ∧ circle_equation 1 :=
by
  sorry

theorem range_of_AB (a b r : ℝ) (h1 : a = 2) (h2 : b = √3) (h3 : r = 1) (l : ℝ) :
3 ≤ tangent_line_intersection 3 ∧ tangent_line_intersection (4 √6 / 3) :=
by
  sorry

end equations_of_ellipse_and_circle_range_of_AB_l28_28235


namespace distributor_B_lower_avg_price_l28_28880

theorem distributor_B_lower_avg_price (p_1 p_2 : ℝ) (h : p_1 < p_2) :
  (p_1 + p_2) / 2 > (2 * p_1 * p_2) / (p_1 + p_2) :=
by {
  sorry
}

end distributor_B_lower_avg_price_l28_28880


namespace find_annual_interest_rate_l28_28433

-- Definitions based on given conditions
def investment_initial : ℝ := 3500
def investment_final : ℝ := 31500
def years : ℕ := 28
def tripling_constant : ℕ := 112

-- Main theorem to prove
theorem find_annual_interest_rate (r : ℝ) :
  investment_final = investment_initial * 3 ^ (years / (tripling_constant / r)) → r = 8 :=
by
  intros h,
  sorry  -- Proof steps are skipped as instructed.

end find_annual_interest_rate_l28_28433


namespace correct_propositions_count_is_3_l28_28293

/-- Conditions on the propositions -/
def prop_1 : Prop := ∀ (P₁ P₂ P₃ : ℝ^3), P₁ ⊥ P₃ ∧ P₂ ⊥ P₃ → (P₁ ∥ P₂) 
def prop_2 : Prop := ∀ (a b : ℝ^3), (¬ a ∥ b ∧ ¬ a ⊥ b) → (∀ plane, plane ∋ a → plane ∋ b → plane ⊥ b)
def prop_3 : Prop := ∀ (a b c : ℝ^3), a ⊥ c ∧ b ⊥ c → (a ∥ b)
def prop_4 : Prop := ∀ (P₁ P₂ l : ℝ^3), P₁ ⊥ l ∧ P₂ ⊥ l → (P₁ ∥ P₂)

/-- Number of correct propositions -/
def correct_propositions_count : ℕ :=
  if ¬ prop_1
     then if prop_2 ∧ prop_3 ∧ prop_4 then 3 else 0 -- Since we conclusively know prop_1 is false
     else if prop_2 ∧ prop_3 ∧ prop_4 then 4 else 0

/-- Proof of the exact count of correct propositions -/
theorem correct_propositions_count_is_3 : correct_propositions_count = 3 :=
  by sorry

end correct_propositions_count_is_3_l28_28293


namespace golden_ratio_in_range_l28_28502

theorem golden_ratio_in_range :
  let phi := (Real.sqrt 5 - 1) / 2
  in 0.6 < phi ∧ phi < 0.7 :=
by
  let phi := (Real.sqrt 5 - 1) / 2
  sorry

end golden_ratio_in_range_l28_28502


namespace ratio_FD_EF_l28_28155

-- definitions for points and the ratio condition
variables {A B C D E F : Type} 

-- Definitions of points and conditions
variables [right_triangle : is_right_triangle A B C]
variables [F_on_AB : point_on_segment F A B]
variables [ratio_AF_FB : ratio_of_segments AF FB = 2]
variables [parallelogram_EBCD : is_parallelogram E B C D]

-- The theorem we're proving
theorem ratio_FD_EF : ratio_of_segments FD EF = 2 :=
sorry

end ratio_FD_EF_l28_28155


namespace gnuff_tutor_minutes_l28_28283

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l28_28283


namespace area_of_square_with_perimeter_40_l28_28203

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l28_28203


namespace tangent_slopes_product_eq_one_l28_28689

-- Given Point P(2, 2) and Circle C: x^2 + y^2 = 1
def P : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ → Prop := λ q, q.1^2 + q.2^2 = 1

-- Defines the slopes of the tangents from point P to circle C
noncomputable def slopes_of_tangents (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) : set ℝ := 
  { k | ∃ (x : ℝ), (x - 2)^2 + (k * (x - 2) - 2)^2 = 1 }

noncomputable def k1k2 (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) : ℝ :=
  let s := slopes_of_tangents P C in 
  if h : s = {k | ∃ x, 3 * k^2 - 8 * k + 3 = 0}
  then let ⟨k1, k2⟩ := classical.some h in k1 * k2 else 0

-- Prove the value of k1 * k2 is 1
theorem tangent_slopes_product_eq_one :
  ∀ P C, k1k2 P C = 1 :=
sorry

end tangent_slopes_product_eq_one_l28_28689


namespace only_B_is_linear_system_l28_28077

def linear_equation (eq : String) : Prop := 
-- Placeholder for the actual definition
sorry 

def system_B_is_linear : Prop :=
  linear_equation "x + y = 2" ∧ linear_equation "x - y = 4"

theorem only_B_is_linear_system 
: (∀ (A B C D : Prop), 
       (A ↔ (linear_equation "3x + 4y = 6" ∧ linear_equation "5z - 6y = 4")) → 
       (B ↔ (linear_equation "x + y = 2" ∧ linear_equation "x - y = 4")) → 
       (C ↔ (linear_equation "x + y = 2" ∧ linear_equation "x^2 - y^2 = 8")) → 
       (D ↔ (linear_equation "x + y = 2" ∧ linear_equation "1/x - 1/y = 1/2")) → 
       (B ∧ ¬A ∧ ¬C ∧ ¬D))
:= 
sorry

end only_B_is_linear_system_l28_28077


namespace square_of_binomial_l28_28731

theorem square_of_binomial (c : ℝ) (h : ∃ a : ℝ, x^2 + 50 * x + c = (x + a)^2) : c = 625 :=
by
  sorry

end square_of_binomial_l28_28731


namespace sum_of_roots_l28_28069

theorem sum_of_roots (a b c : ℝ) (h_eq : a = 1) (h_b : b = -5) (h_c : c = 6) :
  (-b / a) = 5 := by
sorry

end sum_of_roots_l28_28069


namespace number_of_attendants_using_pen_l28_28985

-- Definition of the problem conditions
def conditions : Prop :=
  ∃ P Pe B : ℕ,
  B = 10 ∧
  P + B = 25 ∧
  P + Pe = 20

-- Statement of the proof goal
theorem number_of_attendants_using_pen : conditions → ∃ Pe, Pe + 10 = 15 :=
by
  intro h
  obtain ⟨P, Pe, B, hB, hPB, hPP⟩ := h
  use Pe
  rw [hB]
  sorry

end number_of_attendants_using_pen_l28_28985


namespace jake_has_fewer_peaches_than_steven_l28_28395

theorem jake_has_fewer_peaches_than_steven :
  ∀ (jillPeaches jakePeaches stevenPeaches : ℕ),
    jillPeaches = 12 →
    jakePeaches = jillPeaches - 1 →
    stevenPeaches = jillPeaches + 15 →
    stevenPeaches - jakePeaches = 16 :=
  by
    intros jillPeaches jakePeaches stevenPeaches
    intro h_jill
    intro h_jake
    intro h_steven
    sorry

end jake_has_fewer_peaches_than_steven_l28_28395


namespace can_increase_averages_l28_28349

def grades_group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def grades_group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : Float :=
  (grades.map Nat.toFloat).sum / grades.length

def new_average (grades : List ℕ) (grade_to_remove_or_add : ℕ) (adding : Bool) : Float :=
  if adding
  then (grades.map Nat.toFloat).sum + grade_to_remove_or_add / (grades.length + 1)
  else (grades.map Nat.toFloat).sum - grade_to_remove_or_add / (grades.length - 1)

theorem can_increase_averages :
  ∃ grade,
    grade ∈ grades_group1 ∧
    average grades_group1 < new_average grades_group1 grade false ∧
    average grades_group2 < new_average grades_group2 grade true :=
  sorry

end can_increase_averages_l28_28349


namespace bisection_method_second_step_l28_28891

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem bisection_method_second_step :
  (f 0 < 0) ∧ (f 0.5 > 0) ∧ (continuous_on f (set.Icc 0 0.5)) →
  (∃ x ∈ (set.Icc 0 0.5), f x = 0) ∧ (f 0.25 ∈ f '' (set.Icc 0 0.5)) :=
by
  sorry

end bisection_method_second_step_l28_28891


namespace find_total_employees_l28_28371

variables (E : ℝ)
variables (total_employees : ℝ)
variables (male_employees_below_50 : ℝ)

-- Conditions translated to Lean variables
def condition1 := 0.25 * E = total_employees -- 25% of total employees are males
def condition2 := 120 = male_employees_below_50 -- There are 120 males aged below 50 years

-- The verification equation based on the conditions
def verification_equation := 0.60 * total_employees = male_employees_below_50

theorem find_total_employees (h1 : condition1) (h2 : condition2) (h3 : verification_equation) : E = 800 :=
by
  -- The theorem and conditions will be used in the proof (proof omitted)
  sorry

end find_total_employees_l28_28371


namespace inradius_sum_l28_28816

-- Definitions for given conditions
variable (A B C D E F A' D' G : ℝ × ℝ)
variable {h1 : is_square_paper A B C D}
variable {h2 : is_folding_line EF A B E F}
variable {h3 : sends A to A' EF}
variable {h4 : sends D to D' EF}
variable {h5 : line_intersection A'D' DC G}
variable {h6 : distinct A' B C}

-- Problem statement in Lean 4
theorem inradius_sum (h1 : is_square_paper A B C D)
         (h2 : is_folding_line EF A B E F)
         (h3 : sends A to A' EF)
         (h4 : sends D to D' EF)
         (h5 : line_intersection A'D' DC G)
         (h6 : distinct A' B C):
         inradius (triangle G C A') = inradius (triangle D' G F) + inradius (triangle A' B E) := 
sorry

end inradius_sum_l28_28816


namespace set_equality_implies_a_value_l28_28276

theorem set_equality_implies_a_value (a : ℤ) : ({2, 3} : Set ℤ) = {2, 2 * a - 1} → a = 2 := 
by
  intro h
  sorry

end set_equality_implies_a_value_l28_28276


namespace problem_part_1_problem_part_2_l28_28746

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

noncomputable def g (x : ℝ) := (2 * x + 1) / (x - 1)

theorem problem_part_1 (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_inc : is_monotonically_increasing_on f (set.Ici 0)) :
  is_monotonically_increasing_on (λ x, f (g x)) (set.Ioi 1) :=
sorry

theorem problem_part_2 (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_inc : is_monotonically_increasing_on f (set.Ici 0))
  (a : ℝ)
  (h : f (a - 1) + f (1 - 2 * a) > 0) :
  a < 0 :=
sorry

end problem_part_1_problem_part_2_l28_28746


namespace number_of_students_liked_all_three_l28_28332

open Finset

-- Define the total number of students
def total_students := 50

-- Define the number of students who did not like any dessert
def students_not_like_any := 15

-- Define the number of students who liked each dessert
def liked_apple_pie := 22
def liked_chocolate_cake := 17
def liked_pumpkin_pie := 10

-- Define the number of students who liked at least one dessert
def students_liked_at_least_one := total_students - students_not_like_any

-- Define the number of students who liked all three desserts
def students_liked_all := 7

-- Using the inclusion-exclusion principle, prove the number of students who liked all three desserts is 7
theorem number_of_students_liked_all_three :
  ∃ (students_liked_all : ℕ),
    students_liked_all = 7 ∧ 
    liked_apple_pie + liked_chocolate_cake + liked_pumpkin_pie 
    - students_liked_at_least_one 
    = 2 * students_liked_all :=
by
  have h_students_liked_at_least_one : students_liked_at_least_one = 35 := by
    exact rfl
  have h_desserts_sum : liked_apple_pie + liked_chocolate_cake + liked_pumpkin_pie = 49 := by
    exact rfl
  use 7
  split
  exact rfl
  calc  22 + 17 + 10 - 35 = 49 - 35    : by rw [h_desserts_sum]
                                 ...   = 14                : by norm_num
                                 ...   = 2 * 7             : by norm_num

end number_of_students_liked_all_three_l28_28332


namespace keith_total_cost_l28_28785

noncomputable def total_cost (original_price: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount)
  discounted_price * (1 + tax)

def item_prices : List (ℝ × ℝ × ℝ) := [
  (6.51, 0.10, 0),      -- Toy with 10% discount
  (5.79, 0.12, 0),      -- Pet food with 12% discount
  (12.51, 0, 0.08),     -- Cage with 8% sales tax
  (4.99, 0.15, 0),      -- Water bottle with 15% discount
  (7.65, 0, 0.05),      -- Bedding with 5% sales tax
  (3.25, 0, 0),         -- Food bowl with no discounts or taxes
  (6.35, 0.05, 0.03)    -- Hay feeder with 5% discount and 3% sales tax
]

noncomputable def total_purchase_cost (items: List (ℝ × ℝ × ℝ)) (found_money: ℝ) : ℝ :=
  let total_cost := (items.map (λ x => total_cost x.1 x.2 x.3)).sum
  total_cost - found_money

theorem keith_total_cost :
  total_purchase_cost item_prices 1 = 44.20 :=
by
  sorry

end keith_total_cost_l28_28785


namespace tutoring_minutes_l28_28286

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l28_28286


namespace proof_problem_l28_28518

variable (a k : ℕ)

def b1 : ℕ := 2*a + 1
def b2 : ℕ := b1 + 2
def b3 : ℕ := b2 + 2
def term3 : ℕ := (a+2)^2
def sum_bi (k : ℕ) : ℕ := k*(2*a) + k^2

theorem proof_problem :
  b3 = 2*a + 5 ∧
  (a = 2 → term3 = 16) ∧
  (b1 + b2 + ∑ i in Finset.range k, (2*a + 2*(i+1) - 3) = 2*a*k + k^2) :=
by sorry

end proof_problem_l28_28518


namespace zero_in_interval_l28_28552

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) 
  (h2 : ∃ x ∈ set.Ioo a b, f x = 0)
  (h3 : b = a + 1) : a + b = 5 :=
by
  sorry

end zero_in_interval_l28_28552


namespace representation_of_2015_l28_28455

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l28_28455


namespace transform_to_100_l28_28862

theorem transform_to_100 (a b c : ℤ) (h : Int.gcd (Int.gcd a b) c = 1) :
  ∃ f : (ℤ × ℤ × ℤ → ℤ × ℤ × ℤ), (∀ p : ℤ × ℤ × ℤ,
    ∃ q : ℕ, q ≤ 5 ∧ f^[q] p = (1, 0, 0)) :=
sorry

end transform_to_100_l28_28862


namespace part1_min_value_part1_no_max_value_part2_extreme_points_l28_28269

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (1 + a * x) - 2 * x / (x + 2)

theorem part1_min_value (h : ∀ a > 0, a = 1/2) : f 1/2 2 = Real.log 2 - 1 :=
by
  sorry

theorem part1_no_max_value (h : ∀ a > 0, a = 1/2) :
  ∃ x : ℝ, ∀ y : ℝ, f 1/2 x ≥ f 1/2 y :=
by
  sorry

theorem part2_extreme_points (a : ℝ) (x1 x2 : ℝ) (hx : a ∈ Set.Ioo (1/2 : ℝ) 1)
  (h1 : ∀ x, Deriv (f a) x = 0 → (x = x1 ∨ x = x2))
  (h2 : Deriv (f a) x1 = 0)
  (h3 : Deriv (f a) x2 = 0) :
  f a x1 + f a x2 > f a 0 :=
by
  sorry

end part1_min_value_part1_no_max_value_part2_extreme_points_l28_28269


namespace time_to_fill_remaining_l28_28934

-- Define the rates at which pipes P and Q fill the cistern
def rate_P := 1 / 12
def rate_Q := 1 / 15

-- Define the time both pipes are open together
def time_both_open := 4

-- Calculate the combined rate when both pipes are open
def combined_rate := rate_P + rate_Q

-- Calculate the amount of the cistern filled in the time both pipes are open
def filled_amount_both_open := time_both_open * combined_rate

-- Calculate the remaining amount to fill after Pipe P is turned off
def remaining_amount := 1 - filled_amount_both_open

-- Calculate the time it will take for Pipe Q alone to fill the remaining amount
def time_Q_to_fill_remaining := remaining_amount / rate_Q

-- The final theorem
theorem time_to_fill_remaining : time_Q_to_fill_remaining = 6 := by
  sorry

end time_to_fill_remaining_l28_28934


namespace greatest_possible_percentage_of_airlines_both_services_l28_28093

noncomputable def maxPercentageOfAirlinesWithBothServices (percentageInternet percentageSnacks : ℝ) : ℝ :=
  if percentageInternet <= percentageSnacks then percentageInternet else percentageSnacks

theorem greatest_possible_percentage_of_airlines_both_services:
  let p_internet := 0.35
  let p_snacks := 0.70
  maxPercentageOfAirlinesWithBothServices p_internet p_snacks = 0.35 :=
by
  sorry

end greatest_possible_percentage_of_airlines_both_services_l28_28093


namespace grid_point_triangle_centroid_l28_28220

theorem grid_point_triangle_centroid (n : ℕ) (h₁ : n ≥ 9)
  (grid_points : Fin n → ℤ × ℤ)
  (no_three_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    let (x1, y1) := grid_points i in
    let (x2, y2) := grid_points j in
    let (x3, y3) := grid_points k in
    (y3 - y1) * (x2 - x1) ≠ (y2 - y1) * (x3 - x1)) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    let (x1, y1) := grid_points i in
    let (x2, y2) := grid_points j in
    let (x3, y3) := grid_points k in
    (x1 + x2 + x3) % 3 = 0 ∧ (y1 + y2 + y3) % 3 = 0 := sorry

end grid_point_triangle_centroid_l28_28220


namespace max_area_rect_l28_28129

/--
A rectangle has a perimeter of 40 units and its dimensions are whole numbers.
The maximum possible area of the rectangle is 100 square units.
-/
theorem max_area_rect {l w : ℕ} (hlw : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h_sum : l + w = 20 := by
    rw [two_mul, two_mul, add_assoc, add_assoc] at hlw
    exact Nat.div_eq_of_eq_mul_left (by norm_num : 2 ≠ 0) hlw

  have into_parabola : ∀ l w, l + w = 20 → l * w ≤ 100 := λ l w h_eq =>
  by
    let expr := l * w
    let w_def := 20 - l
    let expr' := l * (20 - l)
    have key_expr: l * w = l * (20 - l) := by
      rw h_eq
    rw key_expr
    let f (l: ℕ) := l * (20 - l)
    have step_expr: l * (20 - l) = 20*l - l^2 := by
      ring

    have boundary : (0 ≤ l * (20 - l)) := mul_nonneg (by apply l.zero_le) (by linarith)
    have max_ex : ((20 / 2)^2 ≤ 100) := by norm_num
    let sq_bound:= 100 - (l - 10)^2
    have complete_sq : 20 * l - l^2 = -(l-10)^2 + 100  := by
      have q_expr: 20 * l - l^2 = - (l-10)^2 + 100 := by linarith
      exact q_expr

    show l * (20 - l) ≤ 100,
    from Nat.le_of_pred_lt (by linarith)


  exact into_parabola l w h_sum

end max_area_rect_l28_28129


namespace mx_squared_l28_28765

theorem mx_squared (PQRS : square) 
                   (P Q R S L O M N X Y : Point)
                   (hPL_EQ_PO : dist P L = dist P O)
                   (hMX_perp_OL : is_perpendicular (MX : Line) (OL : Line))
                   (hNY_perp_OL : is_perpendicular (NY : Line) (OL : Line))
                   (hArea_PLO : area (triangle P L O) = 1)
                   (hArea_QMXL : area (quadrilateral Q M X L) = 1)
                   (hArea_RNYO : area (quadrilateral R N Y O) = 1)
                   (hArea_QNXMY : area (pentagon Q N X M Y) = 1) : 
  MX^2 = 4 * real.sqrt 2 - 4 := sorry

end mx_squared_l28_28765


namespace total_weight_l28_28055

def molecular_weight := 1280

def moles := 8

theorem total_weight (molecular_weight : ℕ) (moles : ℕ) : 
  molecular_weight = 1280 →
  moles = 8 →
  8 * 1280 = 10240 := by
  intros h₁ h₂
  rw [←h₁, ←h₂]
  exact rfl

end total_weight_l28_28055


namespace eccentricity_ellipse_equation_and_coords_of_P_l28_28261

-- Define the given ellipse with its properties
def ellipse (a b : ℝ) (h₀ : a < b) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Given conditions
variables {a b : ℝ} (h₀ : a < b) (h₁ : 0 < b)

-- Define the focus, point on ellipse, and the condition on distance from origin to line FA
def left_focus (e : ℝ) : ℝ × ℝ := (-a * e, 0)

def point_A : ℝ × ℝ := (0, b)

def distance_origin_to_FA (e : ℝ) : Prop :=
  a * sqrt(1 - e^2) / sqrt(2) = sqrt(2) / 2 * b

-- Proof problems
theorem eccentricity_ellipse : 
  ∃ e : ℝ, e = sqrt(2) / 2 ∧ distance_origin_to_FA e :=
sorry

theorem equation_and_coords_of_P :
  ∃ e : ℝ, e = sqrt(2) / 2 ∧ 
    ∃ a' b' : ℝ, a' = sqrt(8) ∧ b' = sqrt(4) ∧ 
    ∃ P : ℝ × ℝ, P = (6 / 5, 8 / 5) ∧ 
      ellipse a' b' ∧ 
      ∀ x y : ℝ, (x, y) ≠ P → 2 * x + y = 0 →
      x^2 + y^2 = 4 :=
sorry

end eccentricity_ellipse_equation_and_coords_of_P_l28_28261


namespace base5_sum_of_234_and_78_l28_28863

theorem base5_sum_of_234_and_78 : 
  let n1 := 234
  let n2 := 78
  let sum_base10 := n1 + n2
  let sum_base5 := "2222"
in 
sum_base5 = (312 : ℕ) to_base 5 := by
  sorry

end base5_sum_of_234_and_78_l28_28863


namespace tetrahedron_edge_CD_length_l28_28030

theorem tetrahedron_edge_CD_length (h₁: ∃ (A B C D : Type), 
  ∃ (distance : A × A → ℕ),
  distance (A, B) = 40 ∧ 
  {distance (A, B), distance (A, C), distance (A, D), distance (B, C), distance (B, D), distance (C, D)} = {8, 15, 17, 24, 31, 40}) : 
  ∃ distance : (Π (A : Type), A × A → ℕ), distance (C, D) = 15 :=
by 
  sorry

end tetrahedron_edge_CD_length_l28_28030


namespace fraction_proof_l28_28956

theorem fraction_proof (x y : ℕ) (h1 : y = 7) (h2 : x = 22) : 
  (y / (x - 1) = 1 / 3) ∧ ((y + 4) / x = 1 / 2) := by
  sorry

end fraction_proof_l28_28956


namespace arithmetic_seq_sum_l28_28976

def seq_1 (n : ℕ) : ℝ := sqrt (3 * n - 1)

noncomputable def seq_2 : ℕ → ℤ 
| 1 := 3
| 2 := 6
| (n + 2) := seq_2 (n + 1) - seq_2 n

noncomputable def seq_3 (a_n : ℕ → ℕ) (s : ℕ) := s = a_n 2 + a_n 8

variable (a_3 a_4 a_5 a_6 a_7 : ℕ)
variable (a_1 a_2 a_4 a_5 : ℕ)
variable (n : ℕ)

theorem arithmetic_seq_sum (s : ℕ) :
  a_3 + a_4 + a_5 + a_6 + a_7 = s →
  5 * a_4 = 450 → seq_3 (λ n => n + a_1 ) 180 := sorry

noncomputable def sum_five (d : ℕ) (S : ℕ) :=
(2 * a_1 + (5 - 1) * d) / 2 → S = 15  := sorry

end

end arithmetic_seq_sum_l28_28976


namespace monkeys_reach_top_unique_l28_28520

structure Ladder where
  rungs : Finset ℕ

inductive Rope where
  | attach : Ladder → ℕ → Ladder → ℕ → Rope

structure MonkeysAndLadders where
  ladders : Fin 5 → Ladder
  ropes : List Rope

noncomputable def unique_monkeys (ML : MonkeysAndLadders) : Prop :=
  ∀ i j : Fin 5, i ≠ j → (∃ n : ℕ, n ∈ ML.ladders i.rungs) ∧ (∃ m : ℕ, m ∈ ML.ladders j.rungs) ∧ n ≠ m

theorem monkeys_reach_top_unique (ML : MonkeysAndLadders) (H : ∀ i : Fin 5, ∃ n : ℕ, n ∈ ML.ladders i.rungs) :
  unique_monkeys ML :=
sorry

end monkeys_reach_top_unique_l28_28520


namespace least_number_with_remainder_l28_28054

variable (x : ℕ)

theorem least_number_with_remainder (x : ℕ) : 
  (x % 16 = 11) ∧ (x % 27 = 11) ∧ (x % 34 = 11) ∧ (x % 45 = 11) ∧ (x % 144 = 11) → x = 36731 := by
  sorry

end least_number_with_remainder_l28_28054


namespace subset_implies_range_of_m_l28_28406

theorem subset_implies_range_of_m (m : ℝ) (A B : set ℝ)
  (hA : ∀ x, x ∈ A ↔ 0 < x ∧ x < m)
  (hB : ∀ x, x ∈ B ↔ 0 < x ∧ x < 1)
  (h : B ⊆ A) : m ≥ 1 :=
by
  sorry

end subset_implies_range_of_m_l28_28406


namespace min_value_expression_l28_28666

theorem min_value_expression (n : ℕ) (n_gt_3 : n > 3) 
  (x : Fin (n + 2) → ℝ) (x_positive_and_increasing : ∀ i : Fin (n + 1), 0 < x i ∧ x i < x (i + 1)) :
  let a := fun (i : Fin n) => x (i + 1) / x i in
  let b := fun (j : Fin n) => x (j + 2) / x (j + 1) in
  let c := fun (k : Fin n) => x (k + 1) * x (k + 2) / (x (k + 1)^2 + x k * x (k + 2)) in
  let d := fun (l : Fin n) => (x (l + 1)^2 + x l * x (l + 2)) / (x l * x (l + 1)) in
  ∑ i, a i * ∑ j, b j / (∑ k, c k * ∑ l, d l) ≥ 1/4 :=
sorry

end min_value_expression_l28_28666


namespace quadratic_max_value_l28_28867

theorem quadratic_max_value :
  ∃ x, f(x) = 1 ∧ ∀ y, f(y) ≤ 1 :=
by
  let f : ℝ → ℝ := fun x => -x^2 + 2 * x
  use 1
  sorry

end quadratic_max_value_l28_28867


namespace correlation_coefficient_one_iff_perfect_linearity_l28_28312

variables {α β : Type*} [linear_ordered_field α] [topological_space α]
variables (points : list (α × α)) (m b : α) (x y : α)

theorem correlation_coefficient_one_iff_perfect_linearity (h_points_on_line : ∀ ⦃p⦄, p ∈ points → p.2 = m * p.1 + b)
    (h_nonzero_slope : m ≠ 0) : 
    let SSR := ∑ val in points, (val.2 - (m * val.1 + b))^2,
        SST := ∑ val in points, (val.2 - y)^2 in
    1 - (SSR / SST) = 1 :=
by 
  sorry

end correlation_coefficient_one_iff_perfect_linearity_l28_28312


namespace triangle_area_l28_28592

theorem triangle_area 
    (a b c : ℕ) 
    (h₀ : a = 9) 
    (h₁ : b = 40) 
    (h₂ : c = 41) 
    (h₃ : a^2 + b^2 = c^2) : 
    1 / 2 * a * b = 180 := 
by {
    rw [h₀, h₁],
    simp,
    norm_num,
}

end triangle_area_l28_28592


namespace star_7_3_l28_28304

def star (a b : ℤ) : ℤ := 4 * a + 3 * b - a * b

theorem star_7_3 : star 7 3 = 16 := 
by 
  sorry

end star_7_3_l28_28304


namespace range_g_is_closed_interval_l28_28182

def g (x : ℝ) : ℝ := (x + 1) / (x^2 + x + 2)

theorem range_g_is_closed_interval :
  set.range g = set.Icc (-1 / 7 : ℝ) 1 :=
by {
  sorry,
}

end range_g_is_closed_interval_l28_28182


namespace possible_values_of_f_l28_28417

theorem possible_values_of_f :
  let f : ℤ → ℤ := 
    λ n, if n > 100 then n - 10 else f (f (n + 11))
  in ∀ n, f n ∈ { n : ℤ | n ≥ 91 } :=
by
  let f : ℤ → ℤ :=
    λ n, if n > 100 then n - 10 else f (f (n + 11))
  intros n
  sorry

end possible_values_of_f_l28_28417


namespace gnuff_tutoring_minutes_l28_28290

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l28_28290


namespace divisor_of_99_l28_28573

def reverse_digits (n : ℕ) : ℕ :=
  -- We assume a placeholder definition for reversing the digits of a number
  sorry

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
  sorry

end divisor_of_99_l28_28573


namespace domain_of_function_range_of_function_l28_28551

-- Domain of the function statement
theorem domain_of_function (x : ℝ) : 
  (x^2 - 2*x > 0) ∧ (9 - x^2 > 0) → (-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 3) :=
sorry

-- Range of the function statement
theorem range_of_function (x : ℝ) : 
  (-x^2 - 6*x - 5 ≥ 0) → (0 ≤ sqrt(- x^2 - 6*x - 5) ∧ sqrt(- x^2 - 6*x - 5) ≤ 2) :=
sorry

end domain_of_function_range_of_function_l28_28551


namespace math_problem_l28_28425

theorem math_problem 
  (p q r : ℕ) 
  (h : 2 * Real.sqrt (Real.cbrt 7 - Real.cbrt 6) = Real.cbrt p + Real.cbrt q - Real.cbrt r) 
  (hp : p = 56) 
  (hq : q = 2) 
  (hr : r = 196) : 
  p + q + r = 254 := 
by sorry

end math_problem_l28_28425
