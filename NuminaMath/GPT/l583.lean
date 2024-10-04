import Mathlib

namespace circle_equation_tangent_to_y_equals_x_l583_583386

theorem circle_equation_tangent_to_y_equals_x :
  (∃ Cx : ℝ, ∃ Cy : ℝ, Cy = 0 ∧ (Cx-1)^2 + (Cy-1)^2 = 2 ∧ 
  ∃ R : ℝ, R^2 = 2 ∧ (Cx - 2)^2 + Cy^2 = R^2) →
  (x:ℝ → y:ℝ → (x-2)^2 + y^2 = 2) :=
by
  assume h,
  sorry

end circle_equation_tangent_to_y_equals_x_l583_583386


namespace num_diamonds_in_G6_l583_583116

noncomputable def triangular_number (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

noncomputable def total_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ k => triangular_number (k + 1)))

theorem num_diamonds_in_G6 :
  total_diamonds 6 = 141 := by
  -- This will be proven
  sorry

end num_diamonds_in_G6_l583_583116


namespace parallelogram_and_triangle_area_eq_l583_583071

noncomputable def parallelogram_area (AB AD : ℝ) : ℝ :=
  AB * AD

noncomputable def right_triangle_area (DG FG : ℝ) : ℝ :=
  (DG * FG) / 2

variables (AB AD DG FG : ℝ)
variables (angleDFG : ℝ)

def parallelogram_ABCD (AB : ℝ) (AD : ℝ) (angleDFG : ℝ) (DG : ℝ) : Prop :=
  parallelogram_area AB AD = 24 ∧ angleDFG = 90 ∧ DG = 6

theorem parallelogram_and_triangle_area_eq (h1 : parallelogram_ABCD AB AD angleDFG DG)
    (h2 : parallelogram_area AB AD = right_triangle_area DG FG) : FG = 8 :=
by
  sorry

end parallelogram_and_triangle_area_eq_l583_583071


namespace largest_number_with_unique_digits_sum_19_is_943210_l583_583773

theorem largest_number_with_unique_digits_sum_19_is_943210 :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∈ (digits 10 n) → (∀ (d' : ℕ), d ≠ d')) ∧
  (list.sum (digits 10 n) = 19) ∧
  (∀ (m : ℕ), (∀ (d : ℕ), d ∈ (digits 10 m) → (∀ (d' : ℕ), d ≠ d')) ∧
  (list.sum (digits 10 m) = 19) → m ≤ n) ∧
  n = 943210 :=
by
  sorry

end largest_number_with_unique_digits_sum_19_is_943210_l583_583773


namespace positive_difference_l583_583410

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end positive_difference_l583_583410


namespace determine_phi_l583_583215

theorem determine_phi
  (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0)
  (hφ : |φ| < π / 2)
  (hω_eq : ω = 2)
  (hshift : ∀ x, f x = sin (2 * x + φ) →
    g x = cos (2 * x) → 
    ∀ x, g x = sin (2 * (x - 2 * π / 3) + φ - 4 * π / 3)) :
  φ = - π / 6 :=
sorry

end determine_phi_l583_583215


namespace work_completion_days_l583_583447

theorem work_completion_days (Dx : ℕ) (Dy : ℕ) (days_y_worked : ℕ) (days_x_finished_remaining : ℕ)
  (work_rate_y : ℝ) (work_rate_x : ℝ) 
  (h1 : Dy = 24)
  (h2 : days_y_worked = 12)
  (h3 : days_x_finished_remaining = 18)
  (h4 : work_rate_y = 1 / Dy)
  (h5 : 12 * work_rate_y = 1 / 2)
  (h6 : work_rate_x = 1 / (2 * days_x_finished_remaining))
  (h7 : Dx * work_rate_x = 1) : Dx = 36 := sorry

end work_completion_days_l583_583447


namespace curve_is_two_rays_and_a_circle_l583_583741

theorem curve_is_two_rays_and_a_circle (x y : ℝ) :
  x * real.sqrt (2 * x^2 + 2 * y^2 - 3) = 0 ↔
  (x^2 + y^2 = 3 / 2) ∨
  (x = 0 ∧ (y ≥ real.sqrt 3 / real.sqrt 2 ∨ y ≤ - real.sqrt 3 / real.sqrt 2)) := 
sorry

end curve_is_two_rays_and_a_circle_l583_583741


namespace least_positive_integer_l583_583541

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l583_583541


namespace circle_radius_order_l583_583863

theorem circle_radius_order 
  (rA: ℝ) (rA_condition: rA = 2)
  (CB: ℝ) (CB_condition: CB = 10 * Real.pi)
  (AC: ℝ) (AC_condition: AC = 16 * Real.pi) :
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  rA < rC ∧ rC < rB :=
by 
  sorry

end circle_radius_order_l583_583863


namespace probability_alice_clara_next_to_each_other_bob_end_l583_583487

open Classical

theorem probability_alice_clara_next_to_each_other_bob_end : 
  let people := ["Alice", "Bob", "Clara", "Dave"],
      arrangements := (people.permutations),
      bob_on_end (arr : List String) := arr.head = "Bob" ∨ arr.reverse.head = "Bob",
      alice_clara_next_each_other (arr : List String) := (arr.zip arr.tail).any (λ t => t = ("Alice", "Clara") ∨ t = ("Clara", "Alice"))
  in (arrangements.filter (λ arr => bob_on_end arr ∧ alice_clara_next_each_other arr)).length = 
     (2 / 3) * arrangements.length :=
by sorry

end probability_alice_clara_next_to_each_other_bob_end_l583_583487


namespace num_quadrilaterals_with_circumcenter_two_l583_583563

-- Definitions for each type of quadrilateral
def is_square (Q : Type) : Prop := 
  ∀ a b c d : Q, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
  ∃ O : Q, dist O a = dist O b ∧ dist O b = dist O c ∧ dist O c = dist O d

def is_rectangle_not_square (Q : Type) : Prop := 
  (∀ a b c d : Q, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  ∃ O : Q, dist O a = dist O b ∧ dist O b = dist O c ∧ dist O c = dist O d) ∧ ¬is_square Q

def is_rhombus_not_square (Q : Type) : Prop :=
  (∀ a b c d : Q, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ ∀ O : Q, dist O a ≠ dist O b ∨ dist O b ≠ dist O c ∨ dist O c ≠ dist O d) ∧  ¬is_square Q

def is_parallelogram_not_rectangle_nor_rhombus (Q : Type) : Prop :=
  (∀ a b c d : Q, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ ∀ O : Q, dist O a ≠ dist O b ∨ dist O b ≠ dist O c ∨ dist O c ≠ dist O d)

def is_trapezoid_not_parallelogram (Q : Type) : Prop :=
  (∀ a b c d : Q, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ ∀ O : Q, dist O a ≠ dist O b ∨ dist O b ≠ dist O c ∨ dist O c ≠ dist O d)

theorem num_quadrilaterals_with_circumcenter_two :
  ∃ (Q : Type), (is_square Q ∨ is_rectangle_not_square Q) ∧ 
  ¬ (is_rhombus_not_square Q ∨ is_parallelogram_not_rectangle_nor_rhombus Q ∨ is_trapezoid_not_parallelogram Q) :=
sorry

end num_quadrilaterals_with_circumcenter_two_l583_583563


namespace novels_per_month_l583_583241

theorem novels_per_month (pages_per_novel : ℕ) (total_pages_per_year : ℕ) (months_in_year : ℕ) 
  (h1 : pages_per_novel = 200) (h2 : total_pages_per_year = 9600) (h3 : months_in_year = 12) : 
  (total_pages_per_year / pages_per_novel) / months_in_year = 4 :=
by
  have novels_per_year := total_pages_per_year / pages_per_novel
  have novels_per_month := novels_per_year / months_in_year
  sorry

end novels_per_month_l583_583241


namespace sufficient_not_necessary_l583_583174

theorem sufficient_not_necessary (x : ℝ) : (|x - 2| < 1 → x^2 + x - 2 > 0) ∧ ∃ x, (x^2 + x - 2 > 0 ∧ ¬(|x - 2| < 1)):=
by
  split
  { intro h1,
    sorry,
  },
  { use -3,
    split,
    { sorry },
    { sorry }
  }

end sufficient_not_necessary_l583_583174


namespace smallest_debt_exists_l583_583422

theorem smallest_debt_exists :
  ∃ (p g : ℤ), 50 = 200 * p + 150 * g := by
  sorry

end smallest_debt_exists_l583_583422


namespace infinitely_many_composites_l583_583353

open Nat

lemma composite_of_form (k : ℕ) : ∃ d : ℕ, d ∣ (2 ^ (3 * (2 * k + 1)) - 1) ∧ 1 < d ∧ d < (2 ^ (3 * (2 * k + 1)) - 1) :=
by sorry

theorem infinitely_many_composites : ∀ n : ℕ, Odd n → (2^n - 1) = composite_of_form n :=
by sorry

end infinitely_many_composites_l583_583353


namespace period_f_l583_583614

noncomputable def f (x : ℝ) : ℝ := (Real.tan x) / (1 - (Real.tan x)^2)

theorem period_f : (∀ x : ℝ, f(x) = f(x + (π / 2))) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f(x) = f(x + T)) → T ≥ (π / 2)) := 
by 
  sorry

end period_f_l583_583614


namespace magician_success_l583_583418

theorem magician_success (n : ℕ) (hn : 2 ≤ n) (a : Fin n → ℕ) 
  (h_digits : ∀ i, a i < 10) :
  ∃! x : ℕ, ∃ (perms : List (List (Fin n))), perms.length = (n - 1)! ∧ S = perms.sum (fun l => list_to_nat (l.map a)) := 
sorry

end magician_success_l583_583418


namespace number_of_n_l583_583558

theorem number_of_n (n : ℕ) (h1 : n ≤ 1000) (h2 : ∃ k : ℕ, 18 * n = k^2) : 
  ∃ K : ℕ, K = 7 :=
sorry

end number_of_n_l583_583558


namespace perimeter_of_equilateral_triangle_l583_583798

-- Define the conditions of the problem in Lean 4
def equilateral_triangle (side_length : ℕ) := 
  ∀ (a b c : ℕ), a = side_length ∧ b = side_length ∧ c = side_length

-- The statement needing proof
theorem perimeter_of_equilateral_triangle : 
  ∀ (side_length : ℕ), 
  side_length = 23 → 
  let P := 3 * side_length in 
  P = 69 :=
by
  intros,
  sorry

end perimeter_of_equilateral_triangle_l583_583798


namespace positive_correlation_l583_583005

-- Definitions based on the conditions
def phrase : Prop := 
  "A great teacher produces outstanding students"

def interpretation : Prop := 
  "The teaching level of the teacher is positively correlated with the level of the students."

-- The proof statement
theorem positive_correlation : 
  interpretation → positive_correlation := 
by 
  sorry

end positive_correlation_l583_583005


namespace cube_painting_equiv_1260_l583_583061

def num_distinguishable_paintings_of_cube : Nat :=
  1260

theorem cube_painting_equiv_1260 :
  ∀ (colors : Fin 8 → Color), -- assuming we have a type Color representing colors
    (∀ i j : Fin 6, i ≠ j → colors i ≠ colors j) →  -- each face has a different color
    ∃ f : Cube × Fin 8 → Cube × Fin 8, -- considering symmetry transformations (rotations)
      num_distinguishable_paintings_of_cube = 1260 :=
by
  -- Proof would go here
  sorry

end cube_painting_equiv_1260_l583_583061


namespace bn_formula_Tn_formula_l583_583929

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := n

-- Define the sum S_n of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 1 / (S n)

-- Define the sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℝ := ∑ k in Finset.range n, b (k + 1)

-- The theorem statements to prove
theorem bn_formula (n : ℕ) : b n = 2 / (n * (n + 1)) :=
by
  sorry

theorem Tn_formula (n : ℕ) : T n = 2 * n / (n + 1) :=
by
  sorry

end bn_formula_Tn_formula_l583_583929


namespace range_of_a1_l583_583928

theorem range_of_a1 (a : ℕ → ℝ) (d : ℝ) (k : ℤ) :
  (d ∈ Ioo (-1 : ℝ) 0) →
  (cos (2 * a 3) * cos (2 * a 5) - sin (2 * a 3) * sin (2 * a 5) - cos (2 * a 3) = sin (a 1 + a 7)) →
  (∀ k : ℤ, a 4 ≠ k * (π / 2)) →
  (∃ n0 : ℕ, n0 = 8 ∧ 
    ∀ n : ℕ, n ≠ 8 → 
    n * (a 1) + ((n * (n - 1) * d) / 2) < 8 * (a 1) + ((8 * (8 - 1) * d) / 2)) →
  (∃ a1 : ℝ, a1 ∈ Ioo (7 * (π / 4)) (2 * π)) :=
begin
  sorry
end

end range_of_a1_l583_583928


namespace sum_coefficients_l583_583913

theorem sum_coefficients : 
  ∀ (a : ℕ → ℝ), 
    (∀ x, (2 - real.sqrt 3 * x) ^ 100 = ∑ i in finset.range 101, (a i) * x ^ i) →
    a 0 = 2 ^ 100 →
    ∑ i in finset.range 101 \ {0}, a i = (2 - real.sqrt 3) ^ 100 - 2 ^ 100 :=
by
  sorry

end sum_coefficients_l583_583913


namespace least_positive_integer_l583_583538

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l583_583538


namespace three_points_one_circle_l583_583965

theorem three_points_one_circle {A B C : Type} [Inh : Nonempty A] [Inh : Nonempty B] [Inh : Nonempty C]
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (not_collinear : ∃ (a : A) (b : B) (c : C), ¬ collinear a b c ) :
  ∃! (circumcircle : Circumcircle), circumcircle.contains A ∧ circumcircle.contains B ∧ circumcircle.contains C :=
sorry

end three_points_one_circle_l583_583965


namespace four_cards_probability_l583_583240

theorem four_cards_probability :
  let deck_size := 52
  let suits_size := 13
  ∀ (C D H S : ℕ), 
  C = 1 ∧ D = 13 ∧ H = 13 ∧ S = 13 →
  (C / deck_size) *
  (D / (deck_size - 1)) *
  (H / (deck_size - 2)) *
  (S / (deck_size - 3)) = (2197 / 499800) :=
by
  intros deck_size suits_size C D H S h
  sorry

end four_cards_probability_l583_583240


namespace find_sum_of_p_q_r_s_l583_583318

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l583_583318


namespace functional_eq_implies_odd_l583_583636

variable {f : ℝ → ℝ}

def functional_eq (f : ℝ → ℝ) :=
∀ a b, f (a + b) + f (a - b) = 2 * f a * Real.cos b

theorem functional_eq_implies_odd (h : functional_eq f) (hf_non_zero : ¬∀ x, f x = 0) : 
  ∀ x, f (-x) = -f x := 
by
  sorry

end functional_eq_implies_odd_l583_583636


namespace peyton_total_yards_l583_583717

def distance_on_Saturday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def distance_on_Sunday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def total_distance (distance_Saturday: Nat) (distance_Sunday: Nat) : Nat :=
  distance_Saturday + distance_Sunday

theorem peyton_total_yards :
  let throws_Saturday := 20
  let yards_per_throw_Saturday := 20
  let throws_Sunday := 30
  let yards_per_throw_Sunday := 40
  distance_on_Saturday throws_Saturday yards_per_throw_Saturday +
  distance_on_Sunday throws_Sunday yards_per_throw_Sunday = 1600 :=
by
  sorry

end peyton_total_yards_l583_583717


namespace abs_inequalities_imply_linear_relationship_l583_583697

theorem abs_inequalities_imply_linear_relationship (a b c : ℝ)
(h1 : |a - b| ≥ |c|)
(h2 : |b - c| ≥ |a|)
(h3 : |c - a| ≥ |b|) :
a = b + c ∨ b = c + a ∨ c = a + b :=
sorry

end abs_inequalities_imply_linear_relationship_l583_583697


namespace correlational_relationships_l583_583850

-- Definitions of relationships
def learning_attitude_and_academic_performance := "The relationship between a student's learning attitude and their academic performance"
def teacher_quality_and_student_performance := "The relationship between a teacher's teaching quality and students' academic performance"
def student_height_and_academic_performance := "The relationship between a student's height and their academic performance"
def family_economic_conditions_and_performance := "The relationship between family economic conditions and students' academic performance"

-- Definition of a correlational relationship
def correlational_relationship (relation : String) : Prop :=
  relation = learning_attitude_and_academic_performance ∨
  relation = teacher_quality_and_student_performance

-- Problem statement to prove
theorem correlational_relationships :
  correlational_relationship learning_attitude_and_academic_performance ∧ 
  correlational_relationship teacher_quality_and_student_performance :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end correlational_relationships_l583_583850


namespace rectangle_ratio_l583_583568

theorem rectangle_ratio (s y x : ℝ) (hs : s > 0) (hy : y > 0) (hx : x > 0)
  (h1 : s + 2 * y = 3 * s)
  (h2 : x + y = 3 * s)
  (h3 : y = s)
  (h4 : x = 2 * s) :
  x / y = 2 := by
  sorry

end rectangle_ratio_l583_583568


namespace ellipse_equation_minimum_k_value_l583_583604

-- Definition of ellipse, focus, and eccentricity
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def focus_of_ellipse (a c : ℝ) (f : ℝ × ℝ) : Prop :=
  f.1 = c ∧ f.2 = 0

def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Conditions and solutions as Lean definitions
def e : ℝ := sqrt 3 / 2
def a := 2 * sqrt 3
def b := sqrt 3
def F2 : ℝ × ℝ := (3, 0)

-- Part (I) proof
theorem ellipse_equation :
  focus_of_ellipse a 3 F2 ∧ eccentricity a 3 = e →
  is_ellipse 2 (sqrt 3) (x y) :=
  by
    sorry

-- Part (II) proof
theorem minimum_k_value (k : ℝ) :
  (y = k * x) ∧ (F2 = (3, 0)) ∧ (sqrt 2 / 2 < e ∧ e ≤ sqrt 3 / 2) →
  (aimplies intersect_ellipse_line (k > 0)) ∧ (dot_product (k) = 0) →
  k = sqrt 2 / 4 :=
  by
    sorry

end ellipse_equation_minimum_k_value_l583_583604


namespace meaningful_sqrt_domain_l583_583012

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end meaningful_sqrt_domain_l583_583012


namespace nat_numbers_exist_l583_583129

theorem nat_numbers_exist :
  ∃ a b c : ℕ,
    (a > 10^10) ∧
    (b > 10^10) ∧
    (c > 10^10) ∧
    ((a * b * c) % (a + 2012) = 0) ∧
    ((a * b * c) % (b + 2012) = 0) ∧
    ((a * b) % (c + 2012) = 0) :=
begin
  -- Proof goes here
  sorry
end

end nat_numbers_exist_l583_583129


namespace find_m_l583_583173

structure Vector (α : Type) :=
(x : α) (y : α) (z : α)

variables {α : Type} [Add α] [Zero α] [Div α] [Mul α]

-- Definitions given in the problem
variables (A B C M : Vector α)
variables (m : α)
noncomputable def vec_add (u v: Vector α) : Vector α :=
  ⟨u.x + v.x, u.y + v.y, u.z + v.z⟩

noncomputable def vec_zero : Vector α := ⟨0, 0, 0⟩

-- The given conditions
axiom cond1 : vec_add (vec_add (vec_add (M)) (A)) (vec_add (vec_add (B)) (vec_add (C))) = vec_zero
axiom cond2 : (vec_add (A) (B)) = m * (vec_add (A) (M))

-- Proof goal
theorem find_m : m = 3 := sorry

end find_m_l583_583173


namespace theo_needs_84_eggs_l583_583095

def customers_hour1 := 5
def customers_hour2 := 7
def customers_hour3 := 3
def customers_hour4 := 8

def eggs_per_omelette_3 := 3
def eggs_per_omelette_4 := 4

def total_eggs_needed : Nat :=
  (customers_hour1 * eggs_per_omelette_3) +
  (customers_hour2 * eggs_per_omelette_4) +
  (customers_hour3 * eggs_per_omelette_3) +
  (customers_hour4 * eggs_per_omelette_4)

theorem theo_needs_84_eggs : total_eggs_needed = 84 :=
by
  sorry

end theo_needs_84_eggs_l583_583095


namespace range_of_m_l583_583918

noncomputable def quadratic_no_real_roots (m : ℝ) : Prop :=
  let Δ := (m - 7) ^ 2 - 4
  in Δ < 0

def cubic_has_extrema_in_interval (m : ℝ) : Prop :=
  m - 9 >= -2 ∧ 9 - m <= 2

theorem range_of_m (m : ℝ) (h1 : m < 9)
  (h2 : (quadratic_no_real_roots m ∨ cubic_has_extrema_in_interval m))
  (h3 : ¬ (quadratic_no_real_roots m ∧ cubic_has_extrema_in_interval m)) :
  5 < m ∧ m < 7 :=
sorry

end range_of_m_l583_583918


namespace fishing_boat_should_go_out_to_sea_l583_583820

def good_weather_profit : ℤ := 6000
def bad_weather_loss : ℤ := -8000
def stay_at_port_loss : ℤ := -1000

def prob_good_weather : ℚ := 0.6
def prob_bad_weather : ℚ := 0.4

def expected_profit_going : ℚ :=  prob_good_weather * good_weather_profit + prob_bad_weather * bad_weather_loss
def expected_profit_staying : ℚ := stay_at_port_loss

theorem fishing_boat_should_go_out_to_sea : 
  expected_profit_going > expected_profit_staying :=
  sorry

end fishing_boat_should_go_out_to_sea_l583_583820


namespace smallest_k_exists_l583_583902

theorem smallest_k_exists :
  ∃ (k : ℕ), (1^2 + 2^2 + 3^2 + ... + k^2) % 180 = 0 ∧ ∀ k', (1^2 + 2^2 + 3^2 + ... + k') % 180 = 0 → k ≤ k' :=
sorry

end smallest_k_exists_l583_583902


namespace neg_p_equivalent_to_forall_x2_ge_1_l583_583394

open Classical

variable {x : ℝ}

-- Definition of the original proposition p
def p : Prop := ∃ (x : ℝ), x^2 < 1

-- The negation of the proposition p
def not_p : Prop := ∀ (x : ℝ), x^2 ≥ 1

-- The theorem stating the equivalence
theorem neg_p_equivalent_to_forall_x2_ge_1 : ¬ p ↔ not_p := by
  sorry

end neg_p_equivalent_to_forall_x2_ge_1_l583_583394


namespace intersection_points_l583_583203

-- Definitions of the given conditions
def C1 (x y : ℝ) : Prop := (x-4)^2 + (y-5)^2 = 25
def C2_polar (rho theta : ℝ) : Prop := rho = 2 * sin theta

-- Convert C2 to rectangular coordinates
def C2_rect (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Prove the polar coordinates of the intersection points of C1 and C2
theorem intersection_points :
  ∃ (rho1 theta1 rho2 theta2 : ℝ), 
  C1 (rho1 * cos theta1) (rho1 * sin theta1) ∧ C2_polar rho1 theta1 ∧ 
  C1 (rho2 * cos theta2) (rho2 * sin theta2) ∧ C2_polar rho2 theta2 ∧ 
  rho1 = sqrt 2 ∧ theta1 = π / 4 ∧
  rho2 = 2 ∧ theta2 = π / 2 :=
by
  sorry

end intersection_points_l583_583203


namespace percentage_apples_sold_l583_583821

theorem percentage_apples_sold (A : ℝ) (P : ℝ) (h1 : A ≈ 700) (h2 : A * (1 - P / 100) = 420) : P = 40 := sorry

end percentage_apples_sold_l583_583821


namespace no_real_intersection_for_pair_C_l583_583016

theorem no_real_intersection_for_pair_C :
  ∀ x : ℝ, x^2 - 5 * x + 6 = 0 → (¬ ∃ r : ℝ, r ≥ 0 ∧ r = x ∧ ∀ x : ℝ, sqrt x = sqrt (x - 6) + 1) :=
begin
  intros x hx,
  rw [(by ring : sqrt x = sqrt (x - 6) + 1 → -6 - 2 * sqrt (x - 6) = 1)],
  sorry
end

end no_real_intersection_for_pair_C_l583_583016


namespace sequence_term_2023_l583_583908

def sequence_term (a : ℕ → ℤ) (n : ℕ) : Prop :=
  (∑ i in Finset.range (n+1), a i) / (n + 1) = n + 1

theorem sequence_term_2023 {a : ℕ → ℤ} :
  (∀ n : ℕ, sequence_term a n) → a 2022 = 4046 :=
by sorry

end sequence_term_2023_l583_583908


namespace find_f_inverse_l583_583591

-- Definitions as derived from conditions
def f : ℕ → ℕ
| 5 := 3
| n := 2 * f (n / 2)  -- This pseudo-definition requires a proper function definition accommodating all cases.

axiom f_cond (x : ℕ) : f (2 * x) = 2 * f x

-- The equivalence theorem we need to prove
theorem find_f_inverse : f 320 = 192 :=
by
  -- Proof is omitted with sorry
  sorry

end find_f_inverse_l583_583591


namespace range_of_t_l583_583340

def ellipse (x y t : ℝ) : Prop := (x^2) / 4 + (y^2) / t = 1

def distance_greater_than_one (x y t : ℝ) : Prop := 
  let a := if t > 4 then Real.sqrt t else 2
  let b := if t > 4 then 2 else Real.sqrt t
  let c := if t > 4 then Real.sqrt (t - 4) else Real.sqrt (4 - t)
  a - c > 1

theorem range_of_t (t : ℝ) : 
  (∀ x y, ellipse x y t → distance_greater_than_one x y t) ↔ 
  (3 < t ∧ t < 4) ∨ (4 < t ∧ t < 25 / 4) := 
sorry

end range_of_t_l583_583340


namespace miaCompletedAdditionalTasksOn6Days_l583_583879

def numDaysCompletingAdditionalTasks (n m : ℕ) : Prop :=
  n + m = 15 ∧ 4 * n + 7 * m = 78

theorem miaCompletedAdditionalTasksOn6Days (n m : ℕ): numDaysCompletingAdditionalTasks n m -> m = 6 :=
by
  intro h
  sorry

end miaCompletedAdditionalTasksOn6Days_l583_583879


namespace exists_integer_polynomial_Q_l583_583309

-- Predicate for integer coefficients polynomials
def int_coeff_poly (P : Polynomial ℤ) : Prop :=
  ∀ n : ℤ, is_square (P.eval n)

-- Main theorem statement
theorem exists_integer_polynomial_Q (P : Polynomial ℤ) (hP : P.leading_coeff = 1) (hQ : int_coeff_poly P) :
  ∃ Q : Polynomial ℤ, P = Q * Q :=
sorry

end exists_integer_polynomial_Q_l583_583309


namespace kamal_chemistry_marks_l583_583293

variables (english math physics biology average total numSubjects : ℕ)

theorem kamal_chemistry_marks 
  (marks_in_english : english = 66)
  (marks_in_math : math = 65)
  (marks_in_physics : physics = 77)
  (marks_in_biology : biology = 75)
  (avg_marks : average = 69)
  (number_of_subjects : numSubjects = 5)
  (total_marks_known : total = 283) :
  ∃ chemistry : ℕ, chemistry = 62 := 
by 
  sorry

end kamal_chemistry_marks_l583_583293


namespace part1_part2_l583_583342

def partsProcessedA : ℕ → ℕ
| 0 => 10
| (n + 1) => if n = 0 then 8 else partsProcessedA n - 2

def partsProcessedB : ℕ → ℕ
| 0 => 8
| (n + 1) => if n = 0 then 7 else partsProcessedB n - 1

def partsProcessedLineB_A (n : ℕ) := 7 * n
def partsProcessedLineB_B (n : ℕ) := 8 * n

def maxSetsIn14Days : ℕ := 
  let aLineA := 2 * (10 + 8 + 6) + (10 + 8)
  let aLineB := 2 * (8 + 7 + 6) + (8 + 8)
  min aLineA aLineB

theorem part1 :
  partsProcessedA 0 + partsProcessedA 1 + partsProcessedA 2 = 24 := 
by sorry

theorem part2 :
  maxSetsIn14Days = 106 :=
by sorry

end part1_part2_l583_583342


namespace simplify_expression_find_inverse_sum_l583_583520

section
variable (x : ℝ)

theorem simplify_expression :
  (1:ℝ) * (2.25) ^ (0.5) - (0.3) ^ (0) - (16) ^ (- 0.75) = (3 / 8) :=
by
  sorry

theorem find_inverse_sum (h : x^(1/2) + x^(-1/2) = 3) : x + 1/x = 7 :=
by
  sorry
end

end simplify_expression_find_inverse_sum_l583_583520


namespace number_of_triplets_l583_583396

theorem number_of_triplets : 
  (∃ (s : Finset (ℝ × ℝ × ℝ)), s.card = 4 ∧ 
    (∀ (p : ℝ × ℝ × ℝ), p ∈ s → 
      (p.1 ≠ 0 ∧ p.2 ≠ 0 ∧ p.3 ≠ 0 ∧ 
       p.1 = p.2 * p.3 ∧ 
       p.2 = p.1 * p.3 ∧ 
       p.3 = p.1 * p.2))) := sorry

end number_of_triplets_l583_583396


namespace general_term_arithmetic_seq_sum_seq_b_l583_583605

variables {a_n : ℕ → ℝ} {b_n S_n : ℕ → ℝ}

-- Define the arithmetic sequence with given conditions
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- State the problem of finding the arithmetic sequence
theorem general_term_arithmetic_seq (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : is_arithmetic_seq a) 
  (h3 : a 2 = (a 1 + a 3) / 2 ∧ a 3 = (a 2 + (a 4 + 1)) / 2) : 
  ∀ n, a n = 2 * n := 
sorry

-- Define the sequence {b_n}
def seq_b (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, b n = 2 / ((n + 3) * (a n + 2))

-- State the problem of summing the sequence {b_n}
theorem sum_seq_b (b : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, b n = 2 / ((n + 3) * (2 * n + 2)))
  (h2 : S n = Σ k in range n, b k) :
  S n = 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := 
sorry

end general_term_arithmetic_seq_sum_seq_b_l583_583605


namespace coeff_x3_of_product_l583_583896

def P (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 5 * x + 6
def Q (x : ℝ) : ℝ := 4 * x^3 + 7 * x^2 + 9 * x + 8

theorem coeff_x3_of_product (x : ℝ) :
  coeff (P x * Q x) 3 = 77 :=
by sorry

end coeff_x3_of_product_l583_583896


namespace points_above_line_l583_583590

theorem points_above_line {t : ℝ} (hP : 1 + t - 1 > 0) (hQ : t^2 + (t - 1) - 1 > 0) : t > 1 :=
by
  sorry

end points_above_line_l583_583590


namespace slope_angle_range_l583_583345

noncomputable def curve (x : ℝ) := sqrt 3 * Real.cos x + 1

theorem slope_angle_range : 
  ∀ (P : ℝ), 
    let α := Real.atan (-(sqrt 3) * Real.sin P) in
    0 ≤ α ∧ α < (Real.pi / 3) ∨ (2 * Real.pi / 3) ≤ α ∧ α < Real.pi := 
by
  sorry

end slope_angle_range_l583_583345


namespace coeff_x2_y_in_expansion_l583_583283

theorem coeff_x2_y_in_expansion : 
  let f (m n : ℕ) := (nat.choose 6 m) * (nat.choose 4 n) in
  f 2 1 = 60 :=
by
  -- Real outline of solving:
  -- f 2 1 = C_6^2 * C_4^1 = 15 * 4 = 60
  let f (m n : ℕ) := (nat.choose 6 m) * (nat.choose 4 n)
  have h1 : f 2 1 = (nat.choose 6 2) * (nat.choose 4 1) := rfl
  have h2 : nat.choose 6 2 = 15 := by sorry
  have h3 : nat.choose 4 1 = 4 := by sorry
  rw [h1, h2, h3]
  exact mul_comm 15 4 -- 15 * 4 = 60

end coeff_x2_y_in_expansion_l583_583283


namespace annual_average_growth_rate_l583_583986

theorem annual_average_growth_rate (x : ℝ) :
  7200 * (1 + x)^2 = 8450 :=
sorry

end annual_average_growth_rate_l583_583986


namespace most_probable_top_quality_products_in_batch_l583_583488

noncomputable def most_probable_top_quality_products (p : ℝ) (n : ℕ) : ℕ :=
  let q := 1 - p
  let np := n * p
  let sqrt_npq := Real.sqrt (n * p * q)
  Nat.floor np

-- Problem statement in Lean
theorem most_probable_top_quality_products_in_batch (p : ℝ) (n : ℕ) (h_p : p = 0.31) (h_n : n = 75) : most_probable_top_quality_products p n = 23 :=
by
  have h_q : p := 0.31
  have h_n75 : n = 75
  calc
    most_probable_top_quality_products p n = Nat.floor (75 * 0.31) := by
      rw [most_probable_top_quality_products, h_p, h_n]
      . sorry -- additional steps for exact calculations
    ... = 23 := by norm_num

end most_probable_top_quality_products_in_batch_l583_583488


namespace find_acute_triangle_angles_find_obtuse_triangle_angles_l583_583388

variable {p q r : ℝ}

-- Angles of the acute-angled triangle
def acute_triangle_angles (p q r : ℝ) : (ℝ × ℝ × ℝ) :=
  let α := (π / 2) * (q + r) / (p + q + r)
  let β := (π / 2) * q / (p + q + r)
  let γ := (π / 2) * r / (p + q + r)
  (α, β, γ)

-- Angles of the obtuse-angled triangle where ∠ BAC is the obtuse angle
def obtuse_triangle_angles (p q r : ℝ) : (ℝ × ℝ × ℝ) :=
  let α := (π / 2) * (1 + p / (p + q + r))
  let β := (π / 2) * q / (p + q + r)
  let γ := (π / 2) * r / (p + q + r)
  (α, β, γ)

theorem find_acute_triangle_angles :
  acute_triangle_angles p q r = 
    ((π / 2) * (q + r) / (p + q + r), (π / 2) * q / (p + q + r), (π / 2) * r / (p + q + r)) := sorry

theorem find_obtuse_triangle_angles :
  obtuse_triangle_angles p q r = 
    ((π / 2) * (1 + p / (p + q + r)), (π / 2) * q / (p + q + r), (π / 2) * r / (p + q + r)) := sorry

end find_acute_triangle_angles_find_obtuse_triangle_angles_l583_583388


namespace maximum_telephone_numbers_l583_583424

theorem maximum_telephone_numbers
  (n : ℕ) (d : ℕ) (H : 1 ≤ d ∧ d ≤ 9)
  (num_diff : ∀ (x y : Fin d → Fin 9), x ≠ y →
    (∃ i, x i ≠ y i) ∧ (∃ j, (x j : ℕ) > (y j : ℕ) + 1 ∨ (y j : ℕ) > (x j : ℕ) + 1)) :
  (finset.card { t : finset (Fin n → Fin 9) |
     ∀ (x y ∈ t), x ≠ y →
     (∃ i, x i ≠ y i) ∧ 
     (∃ j, (x j : ℕ) > (y j : ℕ) + 1 ∨ (y j : ℕ) > (x j : ℕ) + 1) } ≤ (9^n + 1) / 2) := 
sorry

end maximum_telephone_numbers_l583_583424


namespace bill_spots_39_l583_583102

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end bill_spots_39_l583_583102


namespace cos_A_area_triangle_l583_583652

-- Definition of the conditions for the first part
variables {a b c : ℝ} (A B C : ℝ)
variables (h1 : 2 * a * sin B - (sqrt 5) * b * cos A = 0)
variables (h_triang : a = sqrt 5 ∧ b = 2)

-- Theorem statement for the first part
theorem cos_A (h1 : 2 * a * sin B - (sqrt 5) * b * cos A = 0) : cos A = 2 / 3 :=
sorry

-- Theorem statement for the second part
theorem area_triangle (h1 : 2 * a * sin B - (sqrt 5) * b * cos A = 0) (h_triang : a = sqrt 5 ∧ b = 2) : 
  let A_cos := cos A,
      A_sin := sqrt 5 / 3,
      sin_B := 2 / 3,
      cos_B := sqrt 5 / 3 in
  (1 / 2) * a * b * sin C = sqrt 5 :=
sorry

end cos_A_area_triangle_l583_583652


namespace quadratic_points_relationship_l583_583249

theorem quadratic_points_relationship (c y1 y2 y3 : ℝ) 
  (hA : y1 = (-3)^2 + 2*(-3) + c)
  (hB : y2 = (1/2)^2 + 2*(1/2) + c)
  (hC : y3 = 2^2 + 2*2 + c) : y2 < y1 ∧ y1 < y3 := 
sorry

end quadratic_points_relationship_l583_583249


namespace theo_total_eggs_needed_l583_583093

theorem theo_total_eggs_needed:
  (num_3egg_hour1: ℕ) (num_4egg_hour2: ℕ) (num_3egg_hour3: ℕ) (num_4egg_hour4: ℕ)
  (eggs_per_3egg: ℕ) (eggs_per_4egg: ℕ) 
  (num_3egg_customers: ℕ := num_3egg_hour1 + num_3egg_hour3)
  (num_4egg_customers: ℕ := num_4egg_hour2 + num_4egg_hour4)
  (total_eggs_needed: ℕ := eggs_per_3egg * num_3egg_customers + eggs_per_4egg * num_4egg_customers):
  (num_3egg_hour1 = 5) → (num_4egg_hour2 = 7) → (num_3egg_hour3 = 3) → (num_4egg_hour4 = 8) →
  (eggs_per_3egg = 3) → (eggs_per_4egg = 4) → 
  total_eggs_needed = 84 :=
by
  intros
  sorry

end theo_total_eggs_needed_l583_583093


namespace chords_passing_point_or_parallel_l583_583841

open EuclideanGeometry

-- Definitions of points and circle
variables {M N A B C D : Point}
variables {circle : Circle}

-- Tangent condition at endpoint M of diameter MN
axiom tangent_at_M (tangent : Line) : is_tangent tangent circle M

-- MA and MB are segments on the tangent line with constant product
axiom constant_product (MA MB : Segment) : tangent.contains(MA.start) ∧ tangent.contains(MB.start) ∧
  MA.start ≠ M ∧ MB.start ≠ M ∧
  MA.end = M ∧ MB.end = M ∧
  MA.length * MB.length = constant

-- Lines through N intersect the circle at points A, B with a second intersection at C, D respectively
axiom intersection_N (lineA lineB : Line) :
  lineA.contains (N) ∧ lineA.contains (A) ∧ intersects_with circle lineA = {A, C} ∧
  lineB.contains (N) ∧ lineB.contains (B) ∧ intersects_with circle lineB = {B, D}

-- Prove that the chords CD either pass through a common point or are parallel
theorem chords_passing_point_or_parallel :
  ∃ P, ∀ line: Line, (intersects_with circle line).contains(C) ∧ (intersects_with circle line).contains(D) →
    (line.contains P ∨ line.is_parallel P) :=
sorry

end chords_passing_point_or_parallel_l583_583841


namespace man_gets_dividend_l583_583824

    -- Definitions based on conditions
    noncomputable def investment : ℝ := 14400
    noncomputable def premium_rate : ℝ := 0.20
    noncomputable def face_value : ℝ := 100
    noncomputable def dividend_rate : ℝ := 0.07

    -- Calculate the price per share with premium
    noncomputable def price_per_share : ℝ := face_value * (1 + premium_rate)

    -- Calculate the number of shares bought
    noncomputable def number_of_shares : ℝ := investment / price_per_share

    -- Calculate the dividend per share
    noncomputable def dividend_per_share : ℝ := face_value * dividend_rate

    -- Calculate the total dividend
    noncomputable def total_dividend : ℝ := dividend_per_share * number_of_shares

    -- The proof statement
    theorem man_gets_dividend : total_dividend = 840 := by
        sorry
    
end man_gets_dividend_l583_583824


namespace sum_of_coordinates_l583_583255

def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := (g x)^3

theorem sum_of_coordinates (hg : g 4 = 8) : 4 + h 4 = 516 :=
by
  sorry

end sum_of_coordinates_l583_583255


namespace number_of_twinning_integers_under_250_l583_583359

def is_prime (p : Nat) : Prop := p > 1 ∧ ∀ n : Nat, 2 ≤ n → n < p → p % n ≠ 0

def is_twinning (n : Nat) : Prop :=
  n % 2 = 1 ∧ n > 1 ∧ ∀ p, is_prime p → p ∣ n → (p - 2) ∣ n

def twinning_integers : Finset Nat := Finset.filter (λ n, is_twinning n) (Finset.range 250)

theorem number_of_twinning_integers_under_250 : twinning_integers.card = 13 := by
  sorry

end number_of_twinning_integers_under_250_l583_583359


namespace max_additional_payment_days_l583_583747

theorem max_additional_payment_days (n k : ℕ) (h1 : n > k) : 
  ∃ strategy : ℕ → Prop, (∀ sages : list (ℕ × bool), satsifies_strategy_sages sages strategy → count_correct_guesses sages ≥ n - k - 1) :=
begin
  sorry
end

end max_additional_payment_days_l583_583747


namespace circle_bisection_l583_583250

theorem circle_bisection 
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, (x-a)^2 + (y-b)^2 = b^2 + 1 → (x+1)^2 + (y+1)^2 = 4 → False)
  : a^2 + 2a + 2b + 5 = 0 :=
sorry

end circle_bisection_l583_583250


namespace positive_sum_inequality_l583_583350

theorem positive_sum_inequality {n : ℕ} (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  (∑ i : Fin n, x i ^ 3 / (x i ^ 2 + x i * x ((i + 1) % n) + x ((i + 1) % n) ^ 2)) ≥ (1 / 3) * (∑ i in Finset.range n, x i) :=
  by
sorry

end positive_sum_inequality_l583_583350


namespace slope_MN_l583_583932

theorem slope_MN
  (M N F : ℝ × ℝ)
  (on_parabola : M.2 ^ 2 = 6 * M.1)
  (on_directrix : N.1 = -3 / 2)
  (focus : F = (3 / 2, 0))
  (vector_equality : (F.1 - N.1, F.2 - N.2) = (M.1 - F.1, M.2 - F.2)) :
  let slope := (M.2 - N.2) / (M.1 - N.1) in slope = 1 ∨ slope = - 1 :=
by 
  sorry

end slope_MN_l583_583932


namespace sequence_equals_permutation_l583_583519

-- Definitions to setup the problem
def sequence_value : ℕ := 100 * 99 * 98 * 97 * 96 * 95 * 94 * 93 * 92 * 91 * 90 * 89 * 88 * 87 * 86 * 85
def permutation_value (n r : ℕ) : ℕ := n! / (n - r)!

-- Theorem statement
theorem sequence_equals_permutation :
  sequence_value = permutation_value 100 16 :=
sorry

end sequence_equals_permutation_l583_583519


namespace gcd_7_8_fact_l583_583158

-- Define factorial function in lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the GCD function
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define specific factorial values
def f7 := fact 7
def f8 := fact 8

-- Theorem stating the gcd of 7! and 8!
theorem gcd_7_8_fact : gcd f7 f8 = 5040 := by
  sorry

end gcd_7_8_fact_l583_583158


namespace sqrt_expression_equality_l583_583035

theorem sqrt_expression_equality :
  sqrt (13 + sqrt (28 + sqrt 281)) * sqrt (13 - sqrt (28 + sqrt 281)) * sqrt (141 + sqrt 281) = 140 :=
by
  sorry

end sqrt_expression_equality_l583_583035


namespace max_value_of_f_on_interval_l583_583557

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem max_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), ∀ y ∈ set.Icc (-1 : ℝ) (1 : ℝ), f y ≤ f x :=
by {
  use 0,
  split,
  { norm_num },
  intros y hy,
  interval_cases y,
  calc f y ≤ max (f (-1)) (f 1) : sorry
        ... ≤ f 0 : sorry 
}

end max_value_of_f_on_interval_l583_583557


namespace maria_strawberries_l583_583332

theorem maria_strawberries (S : ℕ) :
  (21 = 8 + 9 + S) → (S = 4) :=
by
  intro h
  sorry

end maria_strawberries_l583_583332


namespace max_possible_sum_l583_583809

def pointColor : Type := {red, blue, green}

def numArcs (p1 p2 : pointColor) : Nat :=
match (p1, p2) with
| (pointColor.red, pointColor.blue) => 1
| (pointColor.blue, pointColor.red) => 1
| (pointColor.red, pointColor.green) => 2
| (pointColor.green, pointColor.red) => 2
| (pointColor.blue, pointColor.green) => 3
| (pointColor.green, pointColor.blue) => 3
| (pointColor.red, pointColor.red) => 0
| (pointColor.blue, pointColor.blue) => 0
| (pointColor.green, pointColor.green) => 0

noncomputable def maxSumNumbers : Nat :=
  let redCount := 40
  let blueCount := 30
  let greenCount := 20
  2 * (redCount * 0 + blueCount * 1 + greenCount * 2)

theorem max_possible_sum : maxSumNumbers = 140 := by
  sorry

end max_possible_sum_l583_583809


namespace count_no_digit_one_l583_583628

-- Define the characteristic function that checks if a number does not contain the digit 1
def no_digit_one (n : ℕ) : Bool := 
  ∀ d in toDigits 10 n, d ≠ 1

-- Prove the statement
theorem count_no_digit_one : 
  (Finset.filter no_digit_one (Finset.range 1001)).card = 810 :=
begin
  sorry
end

end count_no_digit_one_l583_583628


namespace convert_base7_to_base10_l583_583858

theorem convert_base7_to_base10 (n : ℕ) (h : n = 4213) : 
  let base7_to_base10 := 3 * 7^0 + 1 * 7^1 + 2 * 7^2 + 4 * 7^3 in
  base7_to_base10 = 1480 :=
by {
  assume h,
  let base7_to_base10 := 3 * 7^0 + 1 * 7^1 + 2 * 7^2 + 4 * 7^3,
  exact rfl,
  sorry
}

end convert_base7_to_base10_l583_583858


namespace shelves_in_room_l583_583845

theorem shelves_in_room
  (n_action_figures_per_shelf : ℕ)
  (total_action_figures : ℕ)
  (h1 : n_action_figures_per_shelf = 10)
  (h2 : total_action_figures = 80) :
  total_action_figures / n_action_figures_per_shelf = 8 := by
  sorry

end shelves_in_room_l583_583845


namespace ryan_spends_7_hours_on_english_l583_583135

variable (C : ℕ)
variable (E : ℕ)

def hours_spent_on_english (C : ℕ) : ℕ := C + 2

theorem ryan_spends_7_hours_on_english :
  C = 5 → E = hours_spent_on_english C → E = 7 :=
by
  intro hC hE
  rw [hC] at hE
  exact hE

end ryan_spends_7_hours_on_english_l583_583135


namespace gcd_fac_7_and_8_equals_5040_l583_583155

theorem gcd_fac_7_and_8_equals_5040 : Nat.gcd 7! 8! = 5040 := 
by 
  sorry

end gcd_fac_7_and_8_equals_5040_l583_583155


namespace inequality_solution_set_l583_583572

def f (x : ℝ) : ℝ :=
if x ≥ 1 then 1 else if 0 < x ∧ x < 1 then -2 else 0

/- The main theorem stating the equivalence of the conditions and the solution set. -/
theorem inequality_solution_set :
  { x : ℝ | log 2 x - (log (1 / 4) (4 * x) - 1) * f (log 3 x + 1) ≤ 5 } =
  { x : ℝ | 1 / 3 < x ∧ x ≤ 4 } := by
  sorry

end inequality_solution_set_l583_583572


namespace find_constant_c_l583_583997

theorem find_constant_c : ∃ (c : ℝ), 
  (∃ (x0 : ℝ), (y x0 = exp x0) ∧ (l x0 = ln 2) ∧ (y x0 = 2)) ∧
  (∀ x y, (exp (ln 2) = 2) ∧ (x + 2 * y + c = 0)) ∧
  (l = -4 - ln 2) :=
by 
  sorry

-- Definitions of the functions y, l
noncomputable def y (x : ℝ) : ℝ := exp x
noncomputable def l (x : ℝ) : ℝ := x + 2 * 2 + (-4 - ln 2)


end find_constant_c_l583_583997


namespace meaningful_sqrt_domain_l583_583011

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end meaningful_sqrt_domain_l583_583011


namespace sequence_count_l583_583559

def sequences (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (1 / 2) * ((1 + Real.sqrt 2) ^ (n + 1) + (1 - Real.sqrt 2) ^ (n + 1))

theorem sequence_count (n : ℕ) : 
  sequences n = (1 / 2) * ((1 + Real.sqrt 2) ^ (n + 1) + (1 - Real.sqrt 2) ^ (n + 1)) :=
  sorry

end sequence_count_l583_583559


namespace max_value_90_l583_583399

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end max_value_90_l583_583399


namespace volume_ratio_of_scaled_tetrahedron_l583_583993

theorem volume_ratio_of_scaled_tetrahedron :
  let V0 := (1, 0, 0, 0)
  let V1 := (0, 1, 0, 0)
  let V2 := (0, 0, 1, 0)
  let V3 := (0, 0, 0, 1)
  let small_V0 := (1/2, 0, 0, 0)
  let small_V1 := (0, 1/2, 0, 0)
  let small_V2 := (0, 0, 1/2, 0)
  let small_V3 := (0, 0, 0, 1/2)
  in
  volume_ratio (V0, V1, V2, V3) (small_V0, small_V1, small_V2, small_V3) = 1/8 := 
sorry

end volume_ratio_of_scaled_tetrahedron_l583_583993


namespace sum_f_1_to_2013_l583_583126

noncomputable def f (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x < -1 then -(x+2)^2
else if -1 ≤ x ∧ x < 3 then x
else if 3 ≤ x then f (x - 6)
else f (x + 6)

theorem sum_f_1_to_2013 : (∑ i in finset.range 2013, f (i + 1)) = 337 :=
by
  sorry

end sum_f_1_to_2013_l583_583126


namespace min_value_of_max_abs_l583_583907

theorem min_value_of_max_abs (a b : ℝ) : 
  ∃ M, M = max (|a + b|) (max (|a - b|) (|1 - b|)) ∧ ∀ M₀, (∀ a b, M₀ ≥ max (|a + b|) (max (|a - b|) (|1 - b|))) → (M₀ ≥ 1 / 2) ∧ (∃ a b, M = 1 / 2) :=
sorry

end min_value_of_max_abs_l583_583907


namespace integer_solutions_count_l583_583227

theorem integer_solutions_count :
  #{(x, y) | x ∈ ℤ ∧ y ∈ ℤ ∧ x^3 + 2 * y^2 = 6 * y} = 2 :=
by
  sorry

end integer_solutions_count_l583_583227


namespace paper_width_is_179_928_l583_583771

noncomputable def cube_volume_in_feet : ℝ := 124.93242414866094
noncomputable def cube_volume_in_inches : ℝ := cube_volume_in_feet * 12^3
noncomputable def cube_side_length_in_inches : ℝ := real.cbrt(cube_volume_in_inches)
noncomputable def cube_surface_area_in_inches : ℝ := 6 * (cube_side_length_in_inches)^2
noncomputable def paper_length_in_inches : ℝ := 120

theorem paper_width_is_179_928 :
  let paper_width := cube_surface_area_in_inches / paper_length_in_inches in
  paper_width = 179.928 :=
by
  sorry

end paper_width_is_179_928_l583_583771


namespace max_value_xyz_l583_583323

theorem max_value_xyz 
  (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 3) : 
  ∃ M, M = 243 ∧ (x + y^4 + z^5) ≤ M := 
  by sorry

end max_value_xyz_l583_583323


namespace largest_x_solution_l583_583030

theorem largest_x_solution (x : ℝ) : (∃ x, (64*x^2 - 3 = 0) ∧ 0 ≤ x ∧ (∀ y, (64*y^2 - 3 = 0) → 0 ≤ y → y ≤ x)) → x = (√3)/8 := by
  sorry

end largest_x_solution_l583_583030


namespace no_solution_for_floor_x_plus_x_eq_15_point_3_l583_583529

theorem no_solution_for_floor_x_plus_x_eq_15_point_3 : ¬ ∃ (x : ℝ), (⌊x⌋ : ℝ) + x = 15.3 := by
  sorry

end no_solution_for_floor_x_plus_x_eq_15_point_3_l583_583529


namespace distribution_plans_l583_583489

theorem distribution_plans (teachers schools : ℕ) (h_teachers : teachers = 3) (h_schools : schools = 6) : 
  ∃ plans : ℕ, plans = 210 :=
by
  sorry

end distribution_plans_l583_583489


namespace probability_of_xi_ge_1_l583_583256

noncomputable def normalDistribution (mean : ℝ) (variance : ℝ) : Type :=
{ ξ : ℝ // probabilityDensityFunction ξ = exp(-((ξ - mean)^2 / (2 * variance))) / sqrt(2 * π * variance) }

def P (predicate : ℝ → Prop) (distribution : normalDistribution) : ℝ :=
sorry -- This should represent the probability of the predicate under the distribution

theorem probability_of_xi_ge_1 (σ : ℝ) :
  let dist := normalDistribution (-1) (σ^2)
  (P (λ ξ, -3 ≤ ξ ∧ ξ ≤ -1) dist = 0.4) →
  (P (λ ξ, ξ ≥ 1) dist = 0.1) :=
sorry

end probability_of_xi_ge_1_l583_583256


namespace exists_triangle_area_le_4_l583_583919

noncomputable def exists_enclosing_triangle (n : ℕ) (P : Fin n → (ℝ × ℝ)) : Prop :=
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → area (triangle (P i) (P j) (P k)) ≤ 1) →
  ∃ T : Triangle, area T ≤ 4 ∧ (∀ i : Fin n, point_in_triangle (P i) T)

theorem exists_triangle_area_le_4 (n : ℕ) (P : Fin n → (ℝ × ℝ)) :
  ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → area (triangle (P i) (P j) (P k)) ≤ 1 →
  ∃ T : Triangle, area T ≤ 4 ∧ (∀ i : Fin n, point_in_triangle (P i) T) :=
sorry

end exists_triangle_area_le_4_l583_583919


namespace find_m_value_l583_583198

-- Define the conditions
def quadratic_has_real_roots (m : ℝ) : Prop :=
  let Δ := (2 * m - 1)^2 - 4 * m^2 in Δ ≥ 0

def correct_m_value (m : ℝ) : Prop :=
  let quadratic_solution_product := (x1 + 1) * (x2 + 1) in
  quadratic_solution_product = 3 → m = -3

theorem find_m_value (m : ℝ) :
  quadratic_has_real_roots m →
  correct_m_value m :=
  sorry

end find_m_value_l583_583198


namespace find_green_pepper_weight_l583_583963

variable (weight_red_peppers : ℝ) (total_weight_peppers : ℝ)

theorem find_green_pepper_weight 
    (h1 : weight_red_peppers = 0.33) 
    (h2 : total_weight_peppers = 0.66) 
    : total_weight_peppers - weight_red_peppers = 0.33 := 
by sorry

end find_green_pepper_weight_l583_583963


namespace find_sum_of_p_q_r_s_l583_583317

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l583_583317


namespace cubic_sum_l583_583731

theorem cubic_sum (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + x * z + y * z = -5) (h3 : x * y * z = -6) :
  x^3 + y^3 + z^3 = 18 :=
by
  sorry

end cubic_sum_l583_583731


namespace banas_no_a_at_first_position_l583_583518

theorem banas_no_a_at_first_position (total_letters: Finset Char) :
  multiset.card total_letters = 7 ∧ multiset.count 'B' total_letters = 1 ∧
  multiset.count 'A' total_letters = 3 ∧ multiset.count 'N' total_letters = 1 ∧
  multiset.count 'S' total_letters = 2 →
  num_permutations total_letters (λ (s : string), s.head ≠ 'A') = 240 :=
begin
  sorry
end

end banas_no_a_at_first_position_l583_583518


namespace lean_statement_l583_583586

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) (eccentricity : ℝ) (semi_minor : ℝ) : Prop :=
  a^2 = 8 ∧ b^2 = 2 ∧ (frac (sqrt 3) 2 = eccentricity) ∧ (sqrt 2 = semi_minor) ∧ 
  (frac (x^2) 8 + frac (y^2) 2 = 1)

noncomputable def line_through_ellipse_vertex (h : a > b ∧ b > 0) : Prop :=
  let l := λ x, 0.5 * x + sqrt(2) in
  ∀x y, frac(x^2, 8) + frac(y^2, 2) = 1 ∧ y = 0.5 * x + sqrt(2) → ((x, y) = (0, sqrt(2)) ∨ (x, y) = (-2*sqrt(2), 0))

noncomputable def slopes_through_point (M : ℝ × ℝ) : Prop := 
  ∀x1 y1 x2 y2, (M = (2, 1)) ∧ 
  (frac(x1^2, 8) + frac(y1^2, 2) = 1) ∧ 
  (frac(x2^2, 8) + frac(y2^2, 2) = 1) ∧ 
  (y1 = 0.5*x1 + sqrt(2)) ∧ 
  (y2 = 0.5*x2 + sqrt(2)) → 
  let k1 := (y1 - 1)/(x1 - 2) in
  let k2 := (y2 - 1)/(x2 - 2) in
  (k1 + k2 = 0 ∧ k1 = -1 * (frac (sqrt 2 - 1) 2) ∧ k2 = frac (sqrt 2 - 1) 2)

theorem lean_statement (a b : ℝ) (h : a > b ∧ b > 0) (eccentricity : ℝ) (semi_minor : ℝ) :
  ellipse_equation a b h eccentricity semi_minor ∧
  line_through_ellipse_vertex h ∧
  slopes_through_point (2, 1) :=
by sorry

end lean_statement_l583_583586


namespace cosine_angle_range_l583_583571

theorem cosine_angle_range (α : ℝ) (h_cos : real.cos α = 3 / 4) (h_acute : 0 < α ∧ α < real.pi / 2) : 
  real.pi / 6 < α ∧ α < real.pi / 4 := 
sorry

end cosine_angle_range_l583_583571


namespace trajectory_of_circle_center_l583_583644

theorem trajectory_of_circle_center :
  ∀ (M : ℝ × ℝ), (∃ r : ℝ, (M.1 + r = 1 ∧ M.1 - r = -1) ∧ (M.1 - 1)^2 + (M.2 - 0)^2 = r^2) → M.2^2 = 4 * M.1 :=
by
  intros M h
  sorry

end trajectory_of_circle_center_l583_583644


namespace min_marbles_prevent_bob_win_l583_583810

theorem min_marbles_prevent_bob_win (n : ℕ) (h : n ≥ 59) : 
  ∃ (n : ℕ), n = 59 ∧ (∀ (m : ℕ), m ≤ 59 → ∀ (boxes : Fin 60 → ℕ), 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 59 ∧ 
  (∃ g1 g2 : Fin k → ℕ, ∃ g3 g4 : Fin (60 - k) → ℕ,
  (∀ i, ((boxes i) + 1 - 1) * (1 + (k - (60 - k))) ≥ 0) ∧
  (∀ i, ((boxes i) - 1 + 1) * (1 - (60 - k)) ≥ 0 ) )
) ∧ 
∀ (b1 b2 : Fin (k - 1) → ℕ),
  ( ∀ j  (boxes : j ≤ 59 → 60 - k ≤ boxes → boxes  ≥ 0 ) ∧ n = 59) :=
begin
  sorry,
end

end min_marbles_prevent_bob_win_l583_583810


namespace weather_conditions_may_15_l583_583490

variable (T : ℝ)

/-- If it is at least 70°F and partly cloudy, then the park will be popular for picnics,
     and on May 15 the park was not popular for picnics,
     then the temperature was less than 70°F or it was not partly cloudy. -/
theorem weather_conditions_may_15 (h1 : (T >= 70) ∧ partly_cloudy → popular_for_picnics)
                                  (h2 : ¬ popular_for_picnics) :
  T < 70 ∨ ¬ partly_cloudy :=
begin
  sorry
end

end weather_conditions_may_15_l583_583490


namespace valid_pairs_count_l583_583672

theorem valid_pairs_count :
  let Jane_age := 30
  ∃ n d : ℕ, d > Jane_age ∧
  (∀ (a b : ℕ), (b > a) ∧ (a + b).Prime ∧ (10a + b > Jane_age) → 
    (d = 10b + a) ∧ (a, b) ∈ {(2, 3), (3, 4), (3, 8), (4, 7), (5, 6), (4, 9), (5, 8), (6, 7), (8, 9)}) ∧
  (∀ (a b : ℕ), (b > a) ∧ (a + b).Prime ∧ (10a + b > Jane_age) → 
    (d = 10b + a) →
    True) ∧ 
  (∃ k : ℕ, k = 9) :=
by
  sorry

end valid_pairs_count_l583_583672


namespace crayons_in_drawer_l583_583413

theorem crayons_in_drawer (initial_crayons : ℝ) (benny_add : ℝ) (lucy_remove : ℝ) (sam_add : ℝ) :
  initial_crayons = 25 → benny_add = 15.5 → lucy_remove = 8.75 → sam_add = 12.25 → 
  initial_crayons + benny_add - lucy_remove + sam_add = 44 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end crayons_in_drawer_l583_583413


namespace largest_number_with_unique_digits_sum_19_is_943210_l583_583774

theorem largest_number_with_unique_digits_sum_19_is_943210 :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∈ (digits 10 n) → (∀ (d' : ℕ), d ≠ d')) ∧
  (list.sum (digits 10 n) = 19) ∧
  (∀ (m : ℕ), (∀ (d : ℕ), d ∈ (digits 10 m) → (∀ (d' : ℕ), d ≠ d')) ∧
  (list.sum (digits 10 m) = 19) → m ≤ n) ∧
  n = 943210 :=
by
  sorry

end largest_number_with_unique_digits_sum_19_is_943210_l583_583774


namespace gcd_fac_7_and_8_equals_5040_l583_583156

theorem gcd_fac_7_and_8_equals_5040 : Nat.gcd 7! 8! = 5040 := 
by 
  sorry

end gcd_fac_7_and_8_equals_5040_l583_583156


namespace value_of_y_when_x_is_neg2_l583_583178

noncomputable def linear_function (k x : ℝ) : ℝ := k * x + 3

theorem value_of_y_when_x_is_neg2 (k : ℝ) (h : k > 0) : ∃ y, y = linear_function k (-2) ∧ y = 1 :=
by {
  use linear_function 1 (-2),
  simp [linear_function],
  sorry,
}

end value_of_y_when_x_is_neg2_l583_583178


namespace area_of_trapezium_eq_336_l583_583042

-- Define the lengths of the parallel sides and the distance between them
def a := 30 -- length of one parallel side in cm
def b := 12 -- length of the other parallel side in cm
def h := 16 -- distance between the parallel sides (height) in cm

-- Define the expected area
def expectedArea := 336 -- area in square cm

-- State the theorem to prove
theorem area_of_trapezium_eq_336 : (1/2 : ℝ) * (a + b) * h = expectedArea := 
by 
  -- The proof is omitted
  sorry

end area_of_trapezium_eq_336_l583_583042


namespace suff_but_not_necessary_condition_l583_583049

theorem suff_but_not_necessary_condition (x y : ℝ) :
  (xy ≠ 6 → x ≠ 2 ∨ y ≠ 3) ∧ ¬ (x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) :=
by
  sorry

end suff_but_not_necessary_condition_l583_583049


namespace part1_part2_l583_583448

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the sequence a_n with given recurrence relations
def sequence (a : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => let an := sequence n
           an + 2 * (floor (Real.sqrt (an.toReal)).toNat : ℕ)

-- Part (1): Prove that for a = 8, n = 5 is the smallest n such that a_n is a perfect square.
theorem part1 :
  let a := 8 in
  let an := sequence a in
  ∃ n, (an n) = (m^2) ∧ (∀ k < n, ¬ isPerfectSquare (an k)) :=
sorry

-- Helper definition to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m^2 = n

-- Part (2): Prove that for a = 2017, n = 82 is the smallest n such that a_n is a perfect square.
theorem part2 :
  let a := 2017 in
  let an := sequence a in
  ∃ n, (an n) = (m^2) ∧ (∀ k < n, ¬ isPerfectSquare (an k)) :=
sorry

end part1_part2_l583_583448


namespace theo_total_eggs_needed_l583_583094

theorem theo_total_eggs_needed:
  (num_3egg_hour1: ℕ) (num_4egg_hour2: ℕ) (num_3egg_hour3: ℕ) (num_4egg_hour4: ℕ)
  (eggs_per_3egg: ℕ) (eggs_per_4egg: ℕ) 
  (num_3egg_customers: ℕ := num_3egg_hour1 + num_3egg_hour3)
  (num_4egg_customers: ℕ := num_4egg_hour2 + num_4egg_hour4)
  (total_eggs_needed: ℕ := eggs_per_3egg * num_3egg_customers + eggs_per_4egg * num_4egg_customers):
  (num_3egg_hour1 = 5) → (num_4egg_hour2 = 7) → (num_3egg_hour3 = 3) → (num_4egg_hour4 = 8) →
  (eggs_per_3egg = 3) → (eggs_per_4egg = 4) → 
  total_eggs_needed = 84 :=
by
  intros
  sorry

end theo_total_eggs_needed_l583_583094


namespace man_age_difference_l583_583826

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end man_age_difference_l583_583826


namespace hexagon_area_ge_half_triangle_l583_583297

theorem hexagon_area_ge_half_triangle
  (A B C P A₁ A₂ B₁ B₂ C₁ C₂ : Point)
  (hP_inside_ABC : P ∈ triangle ABC)
  (hA₁_B₂_parallel_AB : segment A₁ B₂ ∥ segment A B)
  (hB₁_C₂_parallel_BC : segment B₁ C₂ ∥ segment B C)
  (hC₁_A₂_parallel_CA : segment C₁ A₂ ∥ segment C A)
  (hA₁_on_BC : A₁ ∈ segment B C)
  (hA₂_on_BC : A₂ ∈ segment B C)
  (hB₁_on_CA : B₁ ∈ segment C A)
  (hB₂_on_CA : B₂ ∈ segment C A)
  (hC₁_on_AB : C₁ ∈ segment A B)
  (hC₂_on_AB : C₂ ∈ segment A B) :
  area (hexagon A₁ A₂ B₁ B₂ C₁ C₂) ≥ (1 / 2) * area (triangle A B C) :=
sorry

end hexagon_area_ge_half_triangle_l583_583297


namespace find_sequence_l583_583451

theorem find_sequence (c : ℝ) (h : 0 < c) : ∃ (n : ℕ) (a : ℕ → ℂ), c * (1 / 2^n) * (∑ ε in (finset.pi set.univ (λ j, {-1, 1})), ∥∑ j in (finset.range n), (epsilon j) * (a j)∥) < (∑ j in (finset.range n), ∥a j∥^(3/2))^(2/3) := 
by
  sorry

end find_sequence_l583_583451


namespace sum_of_possible_x_l583_583511

theorem sum_of_possible_x (x : ℝ) :
  let seq := [60, 10, 100, 150, 30, x]
  (∀ x, 6 * (350 + x) / 6 = 350 + x) ∧
  (∀ seq, list.median seq = (list.nth seq 2).iget + (list.nth seq 3).iget / 2) →
  (x = 130 ∨ x = 85 ∨ x = -80) →
  x = 130 + 85 + -80 :=
by sorry

end sum_of_possible_x_l583_583511


namespace buses_needed_for_trip_l583_583477

theorem buses_needed_for_trip :
  ∀ (total_students students_in_vans bus_capacity : ℕ),
  total_students = 500 →
  students_in_vans = 56 →
  bus_capacity = 45 →
  ⌈(total_students - students_in_vans : ℝ) / bus_capacity⌉ = 10 :=
by
  sorry

end buses_needed_for_trip_l583_583477


namespace constant_value_l583_583683

variable {A B C P Q H : Type}
variable {a b c R : ℝ}

def is_orthocenter (H : Type) (A B C : Type) : Prop := sorry
def is_circumcenter (O : Type) (A B C : Type) : Prop := sorry
def is_diametrically_opposite (P Q : Type) (O : Type) : Prop := sorry
def on_circumcircle (P : Type) (A B C : Type) (R : ℝ) : Prop := sorry
def distance_squared (X Y : Type) : ℝ := sorry
def side_length (A B : Type) : ℝ := sorry

theorem constant_value (H : Type) (A B C P Q : Type)
  (a b c R : ℝ)
  (H_orthocenter : is_orthocenter H A B C)
  (P_on_circle : on_circumcircle P A B C R)
  (Q_diam_opposite : is_diametrically_opposite P Q (is_circumcenter H A B C)) :
  distance_squared P A + distance_squared Q B + distance_squared P C - distance_squared P H = a^2 + b^2 + c^2 - 4 * R^2 :=
sorry

end constant_value_l583_583683


namespace B_completion_days_l583_583462

theorem B_completion_days :
  (∃ x : ℚ, (3/14 + 1/x) + (41/(14*x)) = 1) → x = 5 := 
begin
  intro h,
  rcases h with ⟨x, hx⟩,
  -- Sorry to skip the proof steps
  sorry
end

end B_completion_days_l583_583462


namespace solve_oplus_l583_583187

def rotate60cw (p : ℝ × ℝ) : ℝ × ℝ :=
(0.5 * p.1 + (sqrt 3 / 2) * p.2, - (sqrt 3 / 2) * p.1 + 0.5 * p.2)

def op (a b : ℝ × ℝ) : ℝ × ℝ :=
rotate60cw (b.1 - a.1, b.2 - a.2)

theorem solve_oplus :
  let x := ( (1 - sqrt 3) / 2, (3 - sqrt 3) / 2 )
  let zero := (0, 0)
  let one_one := (1, 1)
  let target := (1, -1)
  ((op x zero) = ((-sqrt 3, sqrt 3))) ∧
  (op (( (1 - sqrt 3) / 2, (3 - sqrt 3) / 2 ) op zero) one_one = target) := by
  sorry

end solve_oplus_l583_583187


namespace simplify_sqrt_multiplication_l583_583367

variables (x : ℝ)

theorem simplify_sqrt_multiplication (h : 0 ≤ x) :
  (sqrt (50 * x^3) * sqrt (18 * x^2) * sqrt (35 * x) = 30 * x^3 * sqrt 35) :=
sorry

end simplify_sqrt_multiplication_l583_583367


namespace three_digit_solution_count_l583_583630

theorem three_digit_solution_count :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ (4843 * x + 731) % 29 = 1647 % 29 :=
∃ unique_solution_count : ℕ,
  unique_solution_count = 27 ∧
  (∀ x, 100 ≤ x ∧ x < 1000 ∧ (4843 * x + 731) % 29 = 1647 % 29 → unique_solution_count = 27) :=
sorried_puzzle

end three_digit_solution_count_l583_583630


namespace angle_ABH_eq_angle_BFH_l583_583273

theorem angle_ABH_eq_angle_BFH (A B C D E F H : Point)
  (hABC_isosceles : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ dist A B = dist B C)
  (hABC_acute : ∀ {X Y Z : Point}, triangle X Y Z → 0 < (angle X Y Z : Real) ∧ (angle X Y Z : Real) < π / 2)
  (hAD_altitude : altitude A D B C)
  (hCE_altitude : altitude C E A B)
  (hH_orthocenter : is_orthocenter H A B C)
  (hCH_circumcircle : circumcircle A C H F B C) :
  angle A B H = angle B F H :=
by
  sorry

end angle_ABH_eq_angle_BFH_l583_583273


namespace gcd_7_8_fact_l583_583161

-- Define factorial function in lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the GCD function
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define specific factorial values
def f7 := fact 7
def f8 := fact 8

-- Theorem stating the gcd of 7! and 8!
theorem gcd_7_8_fact : gcd f7 f8 = 5040 := by
  sorry

end gcd_7_8_fact_l583_583161


namespace gcf_fact7_fact8_l583_583146

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the values 7! and 8!
def fact_7 : ℕ := factorial 7
def fact_8 : ℕ := factorial 8

-- Prove that the greatest common factor of 7! and 8! is 7!
theorem gcf_fact7_fact8 : Nat.gcd fact_7 fact_8 = fact_7 :=
by
  sorry

end gcf_fact7_fact8_l583_583146


namespace melanie_gave_mother_l583_583703

theorem melanie_gave_mother {initial_dimes dad_dimes final_dimes dimes_given : ℕ}
  (h₁ : initial_dimes = 7)
  (h₂ : dad_dimes = 8)
  (h₃ : final_dimes = 11)
  (h₄ : initial_dimes + dad_dimes - dimes_given = final_dimes) :
  dimes_given = 4 :=
by 
  sorry

end melanie_gave_mother_l583_583703


namespace alice_mowing_time_l583_583847

def effectiveSwathWidthInFeet := (30 - 6) / 12

def flowerbedArea := Float.pi * (20 / 2) ^ 2

def totalLawnArea := 100 * 160

def mowableArea := totalLawnArea - flowerbedArea

def numberOfStrips := 160 / effectiveSwathWidthInFeet

def totalMowingDistance := numberOfStrips * 100

def walkingSpeed := 4000

def mowingTime := totalMowingDistance / walkingSpeed

theorem alice_mowing_time : mowingTime = 2 :=
by sorry

end alice_mowing_time_l583_583847


namespace milk_required_for_flour_l583_583700

theorem milk_required_for_flour (flour_ratio milk_ratio total_flour : ℕ) : 
  (milk_ratio * (total_flour / flour_ratio)) = 160 :=
by
  let milk_ratio := 40
  let flour_ratio := 200
  let total_flour := 800
  exact sorry

end milk_required_for_flour_l583_583700


namespace angle_measure_of_P_l583_583361

noncomputable def measure_angle_P (ABCDE : EuclideanGeometry.ConvexPolygon 5) (AB DE : EuclideanGeometry.Line) (P : EuclideanGeometry.Point) 
  (h1 : ABCDE.IsRegular) (h2 : AB ∈ ABCDE.Sides) (h3 : DE ∈ ABCDE.Sides) (h4 : EuclideanGeometry.IsExtension AB P) (h5 : EuclideanGeometry.IsExtension DE P)
  : Real :=
  36

theorem angle_measure_of_P (ABCDE : EuclideanGeometry.ConvexPolygon 5) (AB DE : EuclideanGeometry.Line) (P : EuclideanGeometry.Point) 
  (h1 : ABCDE.IsRegular) (h2 : AB ∈ ABCDE.Sides) (h3 : DE ∈ ABCDE.Sides) (h4 : EuclideanGeometry.IsExtension AB P) (h5 : EuclideanGeometry.IsExtension DE P) 
  : measure_angle_P ABCDE AB DE P h1 h2 h3 h4 h5 = 36 :=
sorry

end angle_measure_of_P_l583_583361


namespace shaded_fraction_is_one_eighth_l583_583772

noncomputable def total_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

noncomputable def half_area (length : ℕ) (width : ℕ) : ℚ :=
  total_area length width / 2

noncomputable def shaded_area (length : ℕ) (width : ℕ) : ℚ :=
  half_area length width / 4

theorem shaded_fraction_is_one_eighth : 
  ∀ (length width : ℕ), length = 15 → width = 21 → shaded_area length width / total_area length width = 1 / 8 :=
by
  sorry

end shaded_fraction_is_one_eighth_l583_583772


namespace find_lambda_l583_583454

theorem find_lambda (λ : ℝ) (h1 : (λ > 0)) (h2 : ∃ e b, e = 4 / 5 ∧ b = 6 ∧ 0 < b ∧ 
  ∀ (a : ℝ), a = 2 * λ → e = Real.sqrt(1 - (b ^ 2 / a ^ 2))) : λ = 5 :=
sorry

end find_lambda_l583_583454


namespace f_f_neg2_eq_neg4_l583_583980

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then -x
  else x + (2 / x) - 7

theorem f_f_neg2_eq_neg4 : f (f (-2)) = -4 := by
  sorry

end f_f_neg2_eq_neg4_l583_583980


namespace gcf_7fact_8fact_l583_583149

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l583_583149


namespace complex_solution_l583_583575

theorem complex_solution (z : ℂ) (h : z^2 = -5 - 12 * Complex.I) :
  z = 2 - 3 * Complex.I ∨ z = -2 + 3 * Complex.I := 
sorry

end complex_solution_l583_583575


namespace cannot_form_set_l583_583786

-- Definitions for the groups based on their descriptions
def isTall (student : Type) : Prop := sorry  -- As 'tall' is not well-defined.
def isMale (student : Type) : Prop := sorry  -- Gender as a defined attribute.
def isEquilateralTriangle (shape : Type) : Prop := sorry  -- Defined geometrical concept.
def isNonNegative (num : ℝ) : Prop := num ≥ 0 -- Non-negative real numbers.

-- Group definitions
def GroupA := {student : Type | isTall student}
def GroupB := {student : Type | isMale student}
def GroupC := {shape : Type | isEquilateralTriangle shape}
def GroupD := {num : ℝ | isNonNegative num}

-- Math problem statement in Lean
theorem cannot_form_set (A B C : Type) : ¬(noncomputable (GroupA A)) ∧ (noncomputable (GroupB B)) ∧ (noncomputable (GroupC C)) ∧ (∃ x : ℝ, GroupD x) :=
by
  sorry

end cannot_form_set_l583_583786


namespace correct_calculation_l583_583785

variable (a b : ℝ)

theorem correct_calculation : (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l583_583785


namespace solve_for_2023_minus_a_minus_2b_l583_583236

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end solve_for_2023_minus_a_minus_2b_l583_583236


namespace find_coordinates_of_C_l583_583038

-- Define the points A and B with their coordinates.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, -1, 2⟩
def B : Point3D := ⟨7, -4, -1⟩

-- Define the ratio
def ratio : ℝ := 2 / 3

-- Define the vector subtraction for two 3D points
def vector_sub (P Q : Point3D) : Point3D :=
  ⟨P.x - Q.x, P.y - Q.y, P.z - Q.z⟩

-- Define the scalar multiplication for a vector and a scalar
def scalar_mul (k : ℝ) (P : Point3D) : Point3D :=
  ⟨k * P.x, k * P.y, k * P.z⟩

-- Define the vector addition for two 3D points
def vector_add (P Q : Point3D) : Point3D :=
  ⟨P.x + Q.x, P.y + Q.y, P.z + Q.z⟩

-- Define C as the point obtained by the given ratio
def C := vector_add A (scalar_mul ratio (vector_sub B A))

-- Prove that the coordinates of C are (5, -3, 0)
theorem find_coordinates_of_C : C = ⟨5, -3, 0⟩ := 
by sorry

end find_coordinates_of_C_l583_583038


namespace travel_paths_l583_583710

def num_points : Nat := 10
def num_blue_points : Nat := 8
def num_red_points : Nat := 2

def red_connected_to_all_blue : (blue_point : Fin num_blue_points) -> Prop := 
  sorry -- define the connectivity of red points to blue points

def blue_fully_connected : (p1 p2 : Fin num_blue_points) -> Prop := 
  sorry -- define the full connectivity condition for blue points

def distinct_paths : Nat := 645120

theorem travel_paths (num_points = 10) (num_blue_points = 8) (num_red_points = 2)
  (red_connected_to_all_blue) (blue_fully_connected) : 
  ∃ (paths : Nat), paths = distinct_paths :=
  sorry

end travel_paths_l583_583710


namespace S_13_eq_3510_l583_583867

def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

theorem S_13_eq_3510 : S 13 = 3510 :=
by
  sorry

end S_13_eq_3510_l583_583867


namespace sum_powers_of_ab_l583_583745

theorem sum_powers_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1)
  (h3 : a^2 + b^2 = 7) (h4 : a^3 + b^3 = 18) (h5 : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 :=
sorry

end sum_powers_of_ab_l583_583745


namespace player_A_wins_strategy_l583_583759

-- Define the conditions: there are cards numbered from 1 to 2018
def cards := Finset.range 1 2019

-- Define the main theorem stating that player A has a winning strategy
theorem player_A_wins_strategy : ∃ (strategy : strategyA), (strategy_wins strategy) :=
by
  -- Prove that A has a strategy that guarantees an even sum
  -- We are skipping the proof as per instructions
  sorry  -- Proof goes here

end player_A_wins_strategy_l583_583759


namespace rhombus_perimeter_is_52_l583_583501

-- Define the conditions as given in the problem
def rhombus_diagonals (d1 d2: ℕ) := d1 = 10 ∧ d2 = 24

-- Define what it means to calculate the perimeter of the rhombus given its diagonals
def calculate_perimeter (d1 d2: ℕ) (h : rhombus_diagonals d1 d2) : ℕ :=
  let a := d1 / 2
  let b := d2 / 2
  let s := Math.sqrt (a * a + b * b)
  4 * s

-- The theorem stating that the perimeter is indeed 52 inches.
theorem rhombus_perimeter_is_52 : ∀ (d1 d2 : ℕ), rhombus_diagonals d1 d2 → calculate_perimeter d1 d2 = 52 := by
  intros d1 d2 h
  rw [calculate_perimeter]
  sorry

end rhombus_perimeter_is_52_l583_583501


namespace steps_per_flight_l583_583671

-- Define the problem conditions
def jack_flights_up := 3
def jack_flights_down := 6
def steps_height_inches := 8
def jack_height_change_feet := 24

-- Convert the height change to inches
def jack_height_change_inches := jack_height_change_feet * 12

-- Calculate the net flights down
def net_flights_down := jack_flights_down - jack_flights_up

-- Calculate total height change in inches for net flights
def total_height_change_inches := net_flights_down * jack_height_change_inches

-- Calculate the number of steps in each flight
def number_of_steps_per_flight :=
  total_height_change_inches / (steps_height_inches * net_flights_down)

theorem steps_per_flight :
  number_of_steps_per_flight = 108 :=
sorry

end steps_per_flight_l583_583671


namespace positive_expression_l583_583721

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 * (b + c) + a * (b^2 + c^2 - b * c) > 0 :=
by sorry

end positive_expression_l583_583721


namespace engine_capacity_requires_l583_583811

variable (c1 c2 v1 v2 : ℕ)

def condition1 : Prop := 
  c1 = 800 ∧ v1 = 60 ∧ 600 = 600

def condition2 : Prop := 
  c2 * 120 = 800 * 60

theorem engine_capacity_requires 
  (c1_eq : condition1) 
  (c2_eq : condition2) : 
  c2 = 1600 := 
by
  sorry

end engine_capacity_requires_l583_583811


namespace length_of_platform_is_270_m_l583_583064

def speed_km_hr := 72
def time_to_cross_platform_sec := 26
def length_of_train_m := 250

def speed_m_s := speed_km_hr * (5/18 : ℝ) -- Convert speed to m/s

theorem length_of_platform_is_270_m :
  let speed_m_s := speed_m_s in
  let distance_covered_m := speed_m_s * time_to_cross_platform_sec in
  let length_of_platform_m := distance_covered_m - length_of_train_m in
  length_of_platform_m = 270 :=
by 
  -- Proof omitted
  sorry

end length_of_platform_is_270_m_l583_583064


namespace gcf_7fact_8fact_l583_583147

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l583_583147


namespace eccentricity_range_l583_583934

theorem eccentricity_range (a b c x₀ y₀ e : ℝ)
  (h0 : a > b) (h1 : b > 0) (h2 : a > 0) (h3 : x₀^2 / a^2 + y₀^2 / b^2 = 1)
  (h4 : y₀^2 = 3 * c^2 - x₀^2) (hb : b^2 = a^2 - c^2) :
  (1 / 2) ≤ e ∧ e ≤ sqrt 3 / 3 ↔ (a^2 - c^2) / a^2 ≤ 1 / e^2 ≤ 4 :=
begin
  sorry
end

end eccentricity_range_l583_583934


namespace problem_statement_l583_583188

-- Definitions of propositions p and q
def p : Prop := ∀ x, sin (2 * x) = sin (2 * (x + (π / 2)))
def q : Prop := ∀ x y, (cos x = y) → (cos (x + π) = y) 

-- The theorem we want to prove
theorem problem_statement : ¬ (p ∧ q) :=
by sorry

end problem_statement_l583_583188


namespace vasya_time_when_escalator_upward_is_324_seconds_l583_583225

-- Definitions based on the conditions
def vasya_downward_speed := 1 / 2 -- in units per minute
def escalator_speed := 1 / 6 -- in units per minute

noncomputable def vasya_time_up_down_escalator_upward : ℝ :=
  1 / (vasya_downward_speed - escalator_speed) + 1 / ((vasya_downward_speed / 2) + escalator_speed) -- in minutes

-- Convert the time to seconds
def vasya_time_up_down_escalator_upward_seconds : ℝ :=
  vasya_time_up_down_escalator_upward * 60

-- Statement to prove
theorem vasya_time_when_escalator_upward_is_324_seconds : 
  vasya_time_up_down_escalator_upward_seconds = 324 :=
by
  sorry

end vasya_time_when_escalator_upward_is_324_seconds_l583_583225


namespace probability_at_least_one_woman_l583_583982

theorem probability_at_least_one_woman (m w n k : ℕ) (h_m : m = 7) (h_w : w = 3) (h_n : n = 10) (h_k : k = 3) :
  let total_people := m + w in
  let prob_no_woman := (m / total_people : ℝ) * ((m - 1) / (total_people - 1) : ℝ) * ((m - 2) / (total_people - 2) : ℝ) in
  let prob_at_least_one_woman := 1 - prob_no_woman in
  prob_no_woman = 7 / 24 ∧ prob_at_least_one_woman = 17 / 24 :=
by {
  sorry
}

end probability_at_least_one_woman_l583_583982


namespace wicket_keeper_age_difference_l583_583265

-- Definitions based on conditions
def avg_team_age : ℕ := 24
def captain_age : ℕ := 26
def team_members : ℕ := 11
def reduced_avg_age : ℕ := avg_team_age - 1 -- 23
def remaining_players := team_members - 2 -- 9

-- We set up the proof to verify the required age difference
theorem wicket_keeper_age_difference (W : ℕ) (hW : W > captain_age) :
  let total_team_age := team_members * avg_team_age in
  let total_remaining_age := remaining_players * reduced_avg_age in
  total_team_age - captain_age - W = total_remaining_age →
  W - captain_age = 5 :=
by
  sorry

end wicket_keeper_age_difference_l583_583265


namespace Inept_Hands_club_masses_l583_583855

theorem Inept_Hands_club_masses 
  (k b : ℝ)
  (h1 : 3 * k + b = 5.5)
  (h2 : 5 * k + b = 10) :
  (3 * k + b = 5.5) ∧ (5 * k + b = 10) →
  (3 * k + b = 5.5 ∧ 5 * k + b = 10) →
  (3 * k + b = 5.5 ∧ 5 * k + b = 10) .

end Inept_Hands_club_masses_l583_583855


namespace sequence_1234_sum_l583_583046

def sequence_sum (n : ℕ) : ℕ := if n = 0 then 0 else (if n % 2 = 1 then 1 else 2)
def sequence := (λ (n : ℕ), sequence_sum(n))

theorem sequence_1234_sum : (list.sum (list.map sequence (list.finRange 1234))) = 2419 := sorry

end sequence_1234_sum_l583_583046


namespace find_original_number_l583_583976

def original_number_divide_multiply (x : ℝ) : Prop :=
  (x / 12) * 24 = x + 36

theorem find_original_number (x : ℝ) (h : original_number_divide_multiply x) : x = 36 :=
by
  sorry

end find_original_number_l583_583976


namespace equal_face_areas_of_triangular_pyramid_l583_583288

noncomputable def is_triangular_pyramid (S A B C : ℝ × ℝ × ℝ) : Prop := 
  ¬ collinear {S, A, B, C}

noncomputable def is_height (vertex A1 base_plane : ℝ × ℝ × ℝ → Prop) : Prop := 
  vertex ∘ A1 ∈ base_plane ∧ 
  ∀ a b c : ℝ × ℝ × ℝ, base_plane a ∧ base_plane b ∧ base_plane c → 
    (vertex ∈ line_through base_plane a b c) ↔ 
    (∀ h : ℝ × ℝ × ℝ, h ∈ base_plane → line_through vertex(h) ⊥ line_through A1 h)

noncomputable def are_parallel (A B A1 B1 : ℝ × ℝ × ℝ) : Prop := 
  let AB := (λ (x y : ℝ × ℝ × ℝ), ∀ λ : ℝ, y = x + scalar_multiplic λ ) A B
  let A1B1 := (λ (x y : ℝ × ℝ × ℝ), ∀ λ : ℝ, y = x + scalar_multiplic λ ) A1 B1
  (∀ p1 p2 : ℝ × ℝ × ℝ, p1 = AB ∧ p2 = A1B1) p1 ∥ p2

theorem equal_face_areas_of_triangular_pyramid 
  {S A B C : ℝ × ℝ × ℝ}
  (h_tetra : is_triangular_pyramid S A B C)
  (A1 B1 : ℝ × ℝ × ℝ)
  (h1 : is_height A A1 (λ x, x ∈ [S, B, C]))
  (h2 : is_height B B1 (λ x, x ∈ [S, A, C]))
  (h_parallel : are_parallel A B A1 B1) :
  let area (u v w : ℝ × ℝ × ℝ) : ℝ := 
    0.5 * (dist u v) * (dist v w) * (sin ∠_uv w)
  area S C B = area S C A :=
sorry

end equal_face_areas_of_triangular_pyramid_l583_583288


namespace sqrt_condition_l583_583647

theorem sqrt_condition (x : ℝ) : (x - 3 ≥ 0) ↔ (x = 3) :=
by sorry

end sqrt_condition_l583_583647


namespace value_C_plus_D_l583_583417

theorem value_C_plus_D
  (C D : ℚ)
  (h : ∀ x : ℚ, x ≠ 4 → x ≠ 5 → (Dx - 17) / (x^2 - 9x + 20) = C / (x - 4) + 5 / (x - 5)) :
  C + D = 3.8 :=
by
  sorry

end value_C_plus_D_l583_583417


namespace polynomial_remainder_l583_583885

theorem polynomial_remainder (x : ℤ) :
  let poly := x^5 + 3*x^3 + 1
  let divisor := (x + 1)^2
  let remainder := 5*x + 9
  ∃ q : ℤ, poly = divisor * q + remainder := by
  sorry

end polynomial_remainder_l583_583885


namespace count_prime_two_digit_numbers_l583_583970

-- Define the set of digits
def digits := {3, 5, 7, 9}

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Generate all possible two-digit numbers
def two_digit_numbers : Finset ℕ :=
Finset.filter (λ x => x / 10 ≠ x % 10) $
  Finset.image (λ p : ℕ × ℕ => p.1 * 10 + p.2) $
    Finset.product digits digits

-- Define the set of prime two-digit numbers
def prime_two_digit_numbers : Finset ℕ :=
Finset.filter is_prime two_digit_numbers

-- The main theorem to prove
theorem count_prime_two_digit_numbers :
  prime_two_digit_numbers.card = 6 := by
  sorry

end count_prime_two_digit_numbers_l583_583970


namespace right_triangle_area_le_perimeter_squared_div_23_l583_583351

variables (x y : ℝ)

theorem right_triangle_area_le_perimeter_squared_div_23 (h1 : 0 < x) (h2 : 0 < y) :
  let P := x + y + (x^2 + y^2).sqrt in
  let S := (1 / 2) * x * y in
  S ≤ (P^2) / 23 :=
sorry

end right_triangle_area_le_perimeter_squared_div_23_l583_583351


namespace complex_number_count_l583_583162

theorem complex_number_count (z : ℂ) (hz : |z| = 1) (h: |(z / conj(z)) + (conj(z) / z)| = √3) : 
  {z : ℂ | |z| = 1 ∧ |(z / conj(z)) + (conj(z) / z)| = √3}.to_finset.card = 4 := by
  sorry

end complex_number_count_l583_583162


namespace angle_compute_l583_583624

open Real

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b : ℝ × ℝ := (1, 2)

noncomputable def sub_vec := (b.1 - a.1, b.2 - a.2)
noncomputable def sum_vec := (a.1 + 2 * b.1, a.2 + 2 * b.2)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_between (v₁ v₂ : ℝ × ℝ) : ℝ :=
  arccos (dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂))

theorem angle_compute : angle_between sub_vec sum_vec = π / 4 :=
by {
  sorry
}

end angle_compute_l583_583624


namespace parallelogram_side_length_l583_583072

theorem parallelogram_side_length (s : ℝ) (h1 : 0 < s)
  (h2 : let area := s * 3 * (s / real.sqrt 3) / 2 in area = (27 * real.sqrt 3)) :
  s = 3 * real.sqrt 3 :=
by
  sorry

end parallelogram_side_length_l583_583072


namespace maximum_value_of_expression_l583_583230

-- Define the four-dimensional vectors and their unit-vector status
variables (a b c : ℝ^4)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)

-- Define the expression to be maximized
def target_expression : ℝ :=
  2 * ∥a - b∥^2 + 3 * ∥a - c∥^2 + ∥b - c∥^2

-- State the theorem about the maximum value
theorem maximum_value_of_expression : target_expression a b c ≤ 15 := 
sorry

end maximum_value_of_expression_l583_583230


namespace cd_value_l583_583684

-- Define the set of lattice points
def T := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20}

-- Define the property of a line with slope m
def points_on_or_below_line (m : ℚ) : set (ℕ × ℕ) :=
  {p : ℕ × ℕ | (p.2 : ℚ) ≤ m * (p.1 : ℚ)}

-- Given that there are exactly 100 points on or below the line y = mx
def exact_points_count (m : ℚ) : Prop :=
  (T ∩ points_on_or_below_line(m)).to_finset.card = 100

-- Prove that the interval in which m lies has length 1/4, thus c+d = 5
theorem cd_value : (∃ m1 m2 : ℚ, m1 < m2 ∧ exact_points_count m1 ∧ exact_points_count m2 ∧ (m2 - m1 = 1/4)) → 1 + 4 = 5 :=
by sorry

end cd_value_l583_583684


namespace remainder_when_divided_by_3x_minus_6_l583_583900

noncomputable def polynomial := (x : ℚ) → 5 * x^5 - 12 * x^4 + 3 * x^3 - x^2 + 4 * x - 30

theorem remainder_when_divided_by_3x_minus_6 :
  let p := polynomial
  let d := (x : ℚ) → 3 * x - 6 in
  eval 2 p = -34 :=
by
  sorry

end remainder_when_divided_by_3x_minus_6_l583_583900


namespace range_of_a_l583_583329

variable {α : Type*} [OrderedSemiring α]

def is_increasing {β : Type*} [LinearOrderedField β] (f : β → β) : Prop :=
  ∀ x y, x < y → f x < f y

theorem range_of_a (f : ℝ → ℝ) (h_increasing : is_increasing f) 
  (a : ℝ) (h1 : 0 < a) (h2 : f a > f real.pi) : a > real.pi := by
  sorry

end range_of_a_l583_583329


namespace units_digit_17_pow_17_l583_583433

theorem units_digit_17_pow_17 : (17^17 % 10) = 7 := by
  sorry

end units_digit_17_pow_17_l583_583433


namespace half_powers_sum_l583_583207

theorem half_powers_sum (x : ℝ) (h : x + x⁻¹ = 4) : x^(1/2) + x^(-1/2) = Real.sqrt 6 := 
sorry

end half_powers_sum_l583_583207


namespace find_leftmost_vertex_l583_583020

theorem find_leftmost_vertex {n : ℤ} (h_vertices_are_on_curve : 
  ∀ x ∈ {n, n + 0.5, n + 1, n + 1.5}, ∃ y, (y = Real.exp x)) 
  (h_area : 0.5 * abs ((Real.exp n) * (Real.exp (n + 0.5)) +
                       (Real.exp (n + 0.5)) * (Real.exp (n + 1)) +
                       (Real.exp (n + 1)) * (Real.exp (n + 1.5)) +
                       (Real.exp (n + 1.5)) * (Real.exp n) -
                      ((Real.exp (n + 0.5)) * (Real.exp n) +
                       (Real.exp (n + 1)) * (Real.exp (n + 0.5)) +
                       (Real.exp (n + 1.5)) * (Real.exp (n + 1)) +
                       (Real.exp n) * (Real.exp (n + 1.5)))) = 1) : 
  n = 0 :=
sorry

end find_leftmost_vertex_l583_583020


namespace correct_operation_l583_583787

theorem correct_operation : ∃ x, (x = (cbrt ((-1)^3))) ∧ (x = -1) :=
begin
  use cbrt ((-1)^3),
  split,
  { refl },
  { sorry }
end

end correct_operation_l583_583787


namespace fraction_subtraction_simplified_l583_583889

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end fraction_subtraction_simplified_l583_583889


namespace ladder_sliding_rate_l583_583823

variables (t : ℝ) (x y : ℝ → ℝ) (x' y' : ℝ)
variables (hx : ∀ t, x t = 6)
variables (hxt : ∀ t, x' t = 1)

theorem ladder_sliding_rate
  (hx : ∀ t, x t = 6)
  (hxt : ∀ t, deriv x t = 1)
  (hxy : ∀ t, (x t)^2 + (y t)^2 = 100) :
  deriv y (by simp[Set.range x]) =  -3/4 := 
sorry

end ladder_sliding_rate_l583_583823


namespace least_integer_l583_583533

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l583_583533


namespace problem_statement_l583_583125

noncomputable def f (x : ℝ) : ℝ :=
  if 1 < x ∧ x < 3 then 1 + (x - 2)^2 else f (x - 2)

theorem problem_statement : f (Real.sin (2 * Real.pi / 3)) > f (Real.sin (Real.pi / 6)) :=
sorry

end problem_statement_l583_583125


namespace find_remainder_l583_583952

theorem find_remainder (P Q R D D' Q' R' C : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  (P % (D * D')) = (D * R' + R + C) :=
sorry

end find_remainder_l583_583952


namespace distance_between_foci_of_ellipse_l583_583500

theorem distance_between_foci_of_ellipse :
  ∀ (x y : ℝ), (x^2 + 9 * y^2 = 8100) → 
  (let a := 90 in
   let b := 30 in
   let c := Real.sqrt (a^2 - b^2) in
   2 * c = 120 * Real.sqrt 2) :=
by
  intro x y h_eq
  let a := 90
  let b := 30
  let c := Real.sqrt (a^2 - b^2)
  have ha : a = 90 := rfl
  have hb : b = 30 := rfl
  have hc : c = Real.sqrt (a^2 - b^2) := rfl
  sorry

end distance_between_foci_of_ellipse_l583_583500


namespace range_of_x_l583_583577

-- Defining the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Given conditions in Lean
axiom f : ℝ → ℝ
axiom h_odd : odd_function f
axiom h_decreasing_pos : ∀ x y, 0 < x ∧ x < y → f y ≤ f x
axiom h_f4 : f 4 = 0

-- To prove the range of x for which f(x-3) ≤ 0
theorem range_of_x :
    {x : ℝ | f (x - 3) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x} :=
by
  sorry

end range_of_x_l583_583577


namespace length_of_AD_l583_583360

-- Define the sides and the sine of angle B
def AB : ℝ := 6
def BC : ℝ := 8
def CD : ℝ := 30
def sinB : ℝ := 4 / 5

-- Define the length of AD
def AD : ℝ := 31

-- The theorem to prove that the length AD is 31 given the conditions
theorem length_of_AD (AB BC CD : ℝ) (sinB : ℝ) (h₁ : AB = 6) (h₂ : BC = 8) (h₃ : CD = 30) (h₄ : sinB = 4 / 5) : 
  AD = 31 :=
sorry

end length_of_AD_l583_583360


namespace smallest_integer_with_2310_divisors_l583_583018

theorem smallest_integer_with_2310_divisors (m k : ℕ) 
  (h1 : ∃ n, n = m * 30^k ∧ (∀ p q : ℕ, p * q = n → (∀ r1 r2 : ℕ, r1 + 1 = p → r2 + 1 = q → is_prime r1 ∧ is_prime r2)) ∧ num_divisors n = 2310) 
  (h2 : 30 ∣ m) 
  : m + k = 18441 :=
sorry

end smallest_integer_with_2310_divisors_l583_583018


namespace teammates_score_is_correct_l583_583261

-- Definitions based on the given conditions
def Lizzie_score : ℕ := 4
def Nathalie_score : ℕ := Lizzie_score + 3
def Combined_score : ℕ := Lizzie_score + Nathalie_score
def Aimee_score : ℕ := 2 * Combined_score
def Total_score : ℕ := Lizzie_score + Nathalie_score + Aimee_score
def Whole_team_score : ℕ := 50
def Teammates_score : ℕ := Whole_team_score - Total_score

-- Proof statement
theorem teammates_score_is_correct : Teammates_score = 17 := by
  sorry

end teammates_score_is_correct_l583_583261


namespace largest_number_with_digits_sum_to_19_l583_583775

theorem largest_number_with_digits_sum_to_19 : ∃ (n : ℕ), (∀ (d : ℕ), d ∈ digits n → d ≠ 0) ∧ digit_sum n = 19 ∧ largest_number_with_properties n :=
sorry

def digits (n : ℕ) : list ℕ :=
sorry

def digit_sum (n : ℕ) : ℕ :=
(digits n).sum

def largest_number_with_properties (n : ℕ) : Prop :=
∀ (m : ℕ), (∀ (d : ℕ), d ∈ digits m → d ≠ 0) ∧ digit_sum m = 19 → m ≤ n

end largest_number_with_digits_sum_to_19_l583_583775


namespace angle_measure_of_P_l583_583362

noncomputable def measure_angle_P (ABCDE : EuclideanGeometry.ConvexPolygon 5) (AB DE : EuclideanGeometry.Line) (P : EuclideanGeometry.Point) 
  (h1 : ABCDE.IsRegular) (h2 : AB ∈ ABCDE.Sides) (h3 : DE ∈ ABCDE.Sides) (h4 : EuclideanGeometry.IsExtension AB P) (h5 : EuclideanGeometry.IsExtension DE P)
  : Real :=
  36

theorem angle_measure_of_P (ABCDE : EuclideanGeometry.ConvexPolygon 5) (AB DE : EuclideanGeometry.Line) (P : EuclideanGeometry.Point) 
  (h1 : ABCDE.IsRegular) (h2 : AB ∈ ABCDE.Sides) (h3 : DE ∈ ABCDE.Sides) (h4 : EuclideanGeometry.IsExtension AB P) (h5 : EuclideanGeometry.IsExtension DE P) 
  : measure_angle_P ABCDE AB DE P h1 h2 h3 h4 h5 = 36 :=
sorry

end angle_measure_of_P_l583_583362


namespace fraction_of_cars_with_permanent_passes_l583_583090

theorem fraction_of_cars_with_permanent_passes 
    (total_cars : ℕ)
    (non_paying_cars : ℕ)
    (valid_ticket_percentage : ℚ)
    (valid_ticket_cars := valid_ticket_percentage * total_cars)
    (permanent_pass_cars := valid_ticket_cars - non_paying_cars) : 
    total_cars = 300 ∧ non_paying_cars = 30 ∧ valid_ticket_percentage = 0.75 →
    permanent_pass_cars / valid_ticket_cars = 13 / 15 := 
begin
  -- Assumptions
  rintro ⟨htc, hnpc, hvtp⟩,
  sorry
end

end fraction_of_cars_with_permanent_passes_l583_583090


namespace inequality_solution_l583_583375

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end inequality_solution_l583_583375


namespace pqrs_sum_l583_583314

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l583_583314


namespace tangent_point_x_eq_e_a_geq_1_l583_583611

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

-- (1) Prove that the x-coordinate of the tangent point is e
theorem tangent_point_x_eq_e {a : ℝ} : 
  ∃ x₀ : ℝ, x₀ = Real.exp 1 ∧ 
  ∃ k : ℝ, k = a - 1 / x₀ ∧ 
  (λ x, k * (x - x₀) + (a * x₀ - Real.log x₀)) 0 = 0 := sorry

-- (2) Prove that a ≥ 1 given the inequality holds for all x in [1, ∞)
theorem a_geq_1 (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x → f a x ≥ a * (2 * x - x^2)) → 1 ≤ a := sorry

end tangent_point_x_eq_e_a_geq_1_l583_583611


namespace saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l583_583799

noncomputable def bread_saving (n_days : ℕ) : ℕ :=
  (1 / 2) * n_days

theorem saving_20_days :
  bread_saving 20 = 10 :=
by
  -- proof steps for bread_saving 20 = 10
  sorry

theorem cost_saving_20_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 20 * cost_per_loaf) = 350 :=
by
  -- proof steps for cost_saving_20_days
  sorry

theorem saving_60_days :
  bread_saving 60 = 30 :=
by
  -- proof steps for bread_saving 60 = 30
  sorry

theorem cost_saving_60_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 60 * cost_per_loaf) = 1050 :=
by
  -- proof steps for cost_saving_60_days
  sorry

end saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l583_583799


namespace unique_real_solution_between_consecutive_integers_l583_583455

theorem unique_real_solution_between_consecutive_integers (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k < x ∧ x < k + 1 ∧ (⌊x⌋ : ℝ) * (x^2 + 1) = x^3 := sorry

end unique_real_solution_between_consecutive_integers_l583_583455


namespace calculate_expression_l583_583110

theorem calculate_expression :
  (-2)^(4^2) + 2^(3^2) = 66048 := by sorry

end calculate_expression_l583_583110


namespace arrangement_of_students_l583_583457

theorem arrangement_of_students (n m : ℕ) (h_n : n = 6) (h_m : m = 3) :
  (nat.choose n m) * nat.perm m m = 720 :=
by
  rw [h_n, h_m]
  sorry

end arrangement_of_students_l583_583457


namespace polar_to_rectangular_conversion_l583_583120

theorem polar_to_rectangular_conversion:
  ∀ (r θ : ℝ), r = 5 → θ = (5 * π) / 4 →
    let x := r * cos θ, y := r * sin θ in
    (x, y) = (- (5 * real.sqrt 2) / 2, - (5 * real.sqrt 2) / 2) :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  simp [(5 : ℝ), real.cos, real.sin, real.sqrt]
  sorry

end polar_to_rectangular_conversion_l583_583120


namespace average_of_shortest_distances_l583_583068

def rectangle_area : Type := { width : ℕ, height : ℕ }

def lemming_simulation (rect : rectangle_area) (distance_diagonal : ℝ) (distance_upward : ℝ) :=
  let (final_x, final_y) := (10, 8)  -- Adjusted final coordinates after correct calculations
  in (final_x, final_y)

def distance_to_sides (coordinates : ℝ × ℝ) (rect : rectangle_area) : list ℝ :=
  let x := coordinates.1
  let y := coordinates.2
  in [x, y, rect.width - x, rect.height - y]

noncomputable def average_distance (distances : list ℝ) : ℝ :=
  (distances.sum) / (distances.length)

theorem average_of_shortest_distances
  (rect : rectangle_area)
  (h_width : rect.width = 15)
  (h_height : rect.height = 8)
  (distance_diagonal : ℝ)
  (h_diagonal : distance_diagonal = 11.3)
  (distance_upward : ℝ)
  (h_upward : distance_upward = 3)
  : average_distance (distance_to_sides (lemming_simulation rect distance_diagonal distance_upward) rect) = 5.75 :=
by sorry

end average_of_shortest_distances_l583_583068


namespace values_of_x_plus_y_l583_583639

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end values_of_x_plus_y_l583_583639


namespace amount_taken_from_petty_cash_l583_583445

variable (num_staff : ℕ) (days : ℕ) (rate_per_day : ℕ) (amount_given : ℕ)

theorem amount_taken_from_petty_cash (h1 : num_staff = 20) (h2 : days = 30) (h3 : rate_per_day = 100) (h4 : amount_given = 65000) :
  amount_given - (num_staff * days * rate_per_day) = 5000 :=
by
  subst h1
  subst h2
  subst h3
  subst h4
  have total_allowance : (20 * 30 * 100) = 60000 := by norm_num
  norm_num at total_allowance
  rw [total_allowance]
  norm_num
  sorry

end amount_taken_from_petty_cash_l583_583445


namespace pastry_distribution_probability_l583_583067

theorem pastry_distribution_probability :
  let guests := 3
  let pastry_types := 3
  let pastries := 9
  let distribution := 3
  -- Each guest receives exactly 3 pastries randomly distributed
  let prob_two_guests_receive_mixed := (3 * ((9 / 28) * (1 / 10)))

  -- Final result as a fraction where m and n are relatively prime
  probability_that_two_guests_receive_each_type (m n : ℕ) : 
  m / n = 27 / 280 := 
sorry

end pastry_distribution_probability_l583_583067


namespace minimum_words_recall_l583_583229

theorem minimum_words_recall (total_words : ℕ) (recall_rate : ℝ) (required_recall : ℝ) : ℕ :=
  let words_needed := ⌈(required_recall * ↑total_words) / recall_rate⌉ in
  words_needed

#eval minimum_words_recall 600 0.95 0.90 -- Should output 569

end minimum_words_recall_l583_583229


namespace distance_difference_l583_583735

variable (ted_speed : ℝ) (frank_speed : ℝ) (time : ℝ)

def ted_speed := 11.9999976
def time := 2
def frank_speed := 2 / 3 * ted_speed

theorem distance_difference : ted_speed * time - frank_speed * time = 8 :=
by
  sorry

end distance_difference_l583_583735


namespace max_integer_k_l583_583210

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.log (1 - x) else 2 / (x - 1)

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := k / (x * x)

theorem max_integer_k :
  ∃ k : ℤ, k = 7 ∧ ∀ p : ℝ, (1 < p) →
    ∃ m n : ℝ, m < 0 ∧ 0 < n ∧ n < p ∧ f p = f m ∧ f m = g n 7 :=
begin
  sorry
end

end max_integer_k_l583_583210


namespace parking_lot_full_sized_cars_l583_583830

theorem parking_lot_full_sized_cars 
  (total_spaces : ℕ) (ratio_full : ℕ) (ratio_compact : ℕ) 
  (h_total : total_spaces = 450) (h_ratio : ratio_full = 11 ∧ ratio_compact = 4) : 
  let parts := ratio_full + ratio_compact in
  (total_spaces / parts) * ratio_full = 330 :=
by
  sorry

end parking_lot_full_sized_cars_l583_583830


namespace experiment_procedures_count_l583_583521

theorem experiment_procedures_count :
  let P := ["A", "B", "C", "P1", "P2", "P3"],       -- the procedures
  let conditions := "A can only occur first or last" ∧ 
                    "B and C must be adjacent",
  count_sequences P conditions = 96 :=
by
  sorry

end experiment_procedures_count_l583_583521


namespace sum_of_series_l583_583905

theorem sum_of_series (n : ℕ) :
  (∑ k in Finset.range (n + 1), 1 / ((2 * k + 1) * (2 * k + 3))) = n / (2 * n + 1) := sorry

end sum_of_series_l583_583905


namespace gcd_fac_7_and_8_equals_5040_l583_583154

theorem gcd_fac_7_and_8_equals_5040 : Nat.gcd 7! 8! = 5040 := 
by 
  sorry

end gcd_fac_7_and_8_equals_5040_l583_583154


namespace max_value_a_l583_583252

def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1 / x|

theorem max_value_a : ∃ (a : ℝ), condition a ∧ (∀ b : ℝ, condition b → b ≤ 4) :=
  sorry

end max_value_a_l583_583252


namespace magnitude_of_difference_l583_583959

noncomputable def vector_a : ℝ × ℝ := (-1, Real.sqrt 7)
noncomputable def vector_b : ℝ × ℝ := (cos (Real.pi / 4), sin (Real.pi / 4)) -- since |b| = 1 and angle is pi/4

lemma vector_b_magnitude_one : Real.sqrt ((vector_b.1)^2 + (vector_b.2)^2) = 1 := by
  simp [vector_b]
  sorry

lemma angle_between_vector_a_and_vector_b :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 2 := by
  simp [vector_a, vector_b]
  sorry

theorem magnitude_of_difference :
  Real.sqrt ((vector_a.1 - 2 * vector_b.1)^2 + (vector_a.2 - 2 * vector_b.2)^2) = 2 := by
  simp [vector_a, vector_b]
  sorry

end magnitude_of_difference_l583_583959


namespace sqrt_fraction_sum_l583_583165

theorem sqrt_fraction_sum : sqrt((9 / 16 : ℝ) + (25 / 36 : ℝ)) = sqrt(181) / 12 :=
by
  sorry

end sqrt_fraction_sum_l583_583165


namespace at_least_5_limit_ups_needed_l583_583839

-- Let's denote the necessary conditions in Lean
variable (a : ℝ) -- the buying price of stock A

-- Initial price after 4 consecutive limit downs
def price_after_limit_downs (a : ℝ) : ℝ := a * (1 - 0.1) ^ 4

-- Condition of no loss after certain limit ups
def no_loss_after_limit_ups (a : ℝ) (x : ℕ) : Prop := 
  price_after_limit_downs a * (1 + 0.1)^x ≥ a
  
theorem at_least_5_limit_ups_needed (a : ℝ) : ∃ x, no_loss_after_limit_ups a x ∧ x ≥ 5 :=
by
  -- We are required to find such x and prove the condition, which has been shown in the mathematical solution
  sorry

end at_least_5_limit_ups_needed_l583_583839


namespace add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l583_583756

theorem add_neg_eq_neg_add (a b : Int) : a + -b = a - b := by
  sorry

theorem neg_ten_plus_neg_twelve : -10 + (-12) = -22 := by
  have h1 : -10 + (-12) = -10 - 12 := add_neg_eq_neg_add _ _
  have h2 : -10 - 12 = -(10 + 12) := by
    sorry -- This step corresponds to recognizing the arithmetic rule for subtraction.
  have h3 : -(10 + 12) = -22 := by
    sorry -- This step is the concrete calculation.
  exact Eq.trans h1 (Eq.trans h2 h3)

end add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l583_583756


namespace find_a_n_sum_b_n_l583_583274

variable {a : ℕ → ℕ}
variable (h₁ : a 2 = 4)
variable (h₂ : a 4 + a 7 = 15)

theorem find_a_n : a n = n + 2 :=
by
  sorry

def b (n : ℕ) : ℕ := 2 * (a n) - 2 + n

theorem sum_b_n : (Finset.range 10).sum (λ n, b (n + 1)) = 185 :=
by
  sorry

end find_a_n_sum_b_n_l583_583274


namespace saras_age_l583_583678

theorem saras_age (S : ℕ) (h1 : 22 = 22) (h2 : S = (1.5 * 22)) : S = 33 := 
by
  -- proof
  sorry

end saras_age_l583_583678


namespace number_of_terms_in_sequence_l583_583105

-- Define the arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℤ) (start d : ℤ) : Prop :=
  ∀ n, s n = start - d * n

-- Define the specific sequence in the problem
def specific_sequence : ℕ → ℤ :=
  λ n, 165 - 5 * n

-- Prove that the number of terms in the sequence is 27
theorem number_of_terms_in_sequence : ∃ n, specific_sequence n = 35 ∧ ( ∀ m, specific_sequence m = 35 -> m < 27) ∧ n = 27 :=
by
  sorry

end number_of_terms_in_sequence_l583_583105


namespace solve_for_m_l583_583795

theorem solve_for_m (m : ℤ) (h : (-2 : ℤ)^(2 * m) = (2 : ℤ)^(6 - m)) : m = 2 :=
sorry

end solve_for_m_l583_583795


namespace allocation_schemes_count_l583_583706

def distinct_quotas := {1, 2, 3, 4}

def villages := {v1, v2, v3, v4}

theorem allocation_schemes_count : (finset.univ.card (equiv.fin_fin_arrow 4)) = 24 := by
  sorry

end allocation_schemes_count_l583_583706


namespace W_nonzero_coefficients_l583_583868

-- Define the polynomial W(x)
noncomputable def W (k : ℕ) (a : ℝ) (Q : ℝ[X]) : ℝ[X] := (X - C a)^k * Q

-- Conditions: a ≠ 0, Q is a nonzero polynomial
variable (a : ℝ) (h_a : a ≠ 0) (Q : ℝ[X]) (hQ : Q ≠ 0)

/-- Prove that the polynomial W(x) = (x - a)^k * Q(x) has 
    at least k + 1 nonzero coefficients -/
theorem W_nonzero_coefficients (k : ℕ) : (W k a Q).nonzero_coeffs.card ≥ k + 1 :=
by 
  sorry  -- Proof to be completed.

end W_nonzero_coefficients_l583_583868


namespace part1_part2_l583_583212

theorem part1
    (a b : ℝ)
    (h₁: ∃ f : ℝ → ℝ, ∀ x, f x = 2 * x^3 + a * x^2 + b * x + 1)
    (h₂: ∀ x, f'(x) = 6 * x^2 + 2 * a * x + b)
    (symmetry_condition: -a / 6 = -1 / 2)
    (derivative_at_1: f'(1) = 0):
    a = 3 ∧ b = -12 := sorry

theorem part2
    (a b : ℝ)
    (h₁: a = 3)
    (h₂: b = -12):
    (let f := λ x, 2 * x^3 + 3 * x^2 - 12 * x + 1 in
    let f' := λ x, 6 * x^2 + 6 * x - 12 in
    f (-2) = 21 ∧ f (1) = -6) := sorry

end part1_part2_l583_583212


namespace inequality_solution_l583_583370

noncomputable def solve_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo (-3 : ℝ) 3

theorem inequality_solution (x : ℝ) (h : x ≠ -3) :
  (x^2 - 9) / (x + 3) < 0 ↔ solve_inequality x :=
by
  sorry

end inequality_solution_l583_583370


namespace round_robin_winner_loser_l583_583270

theorem round_robin_winner_loser {n : ℕ} (h8n : 8 * n > 0) 
  (no_ties : ∀ (V1 V2 : ℕ), V1 ≠ V2 → (beats V1 V2 ∨ beats V2 V1)) 
  (no_cycles : ∀ k (V : ℕ → ℕ), (∀ i, i < k → beats (V i) (V (i+1)) ) → ¬ beats (V k) (V 0)) 
  : ∃ V1 V8, (∀ V, V ≠ V1 → beats V1 V) ∧ (∀ V, V ≠ V8 → beats V V8) :=
by
  sorry

end round_robin_winner_loser_l583_583270


namespace length_AE_l583_583272

structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 4⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt (((Q.x - P.x : ℝ) ^ 2) + ((Q.y - P.y : ℝ) ^ 2))

noncomputable def AE_length : ℝ :=
  (5 * (dist A B)) / 9

theorem length_AE :
  ∃ E : Point, AE_length = (5 * Real.sqrt 65) / 9 := by
  sorry

end length_AE_l583_583272


namespace peyton_total_yards_l583_583718

def distance_on_Saturday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def distance_on_Sunday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def total_distance (distance_Saturday: Nat) (distance_Sunday: Nat) : Nat :=
  distance_Saturday + distance_Sunday

theorem peyton_total_yards :
  let throws_Saturday := 20
  let yards_per_throw_Saturday := 20
  let throws_Sunday := 30
  let yards_per_throw_Sunday := 40
  distance_on_Saturday throws_Saturday yards_per_throw_Saturday +
  distance_on_Sunday throws_Sunday yards_per_throw_Sunday = 1600 :=
by
  sorry

end peyton_total_yards_l583_583718


namespace scientific_notation_of_diameter_l583_583385

-- Definitions of the conditions
def diameter_of_moss_pollen : ℝ := 0.0000084

-- Prove the scientific notation of the given diameter
theorem scientific_notation_of_diameter : diameter_of_moss_pollen = 8.4 * 10^(-6) :=
by
  sorry

end scientific_notation_of_diameter_l583_583385


namespace min_value_a2_b2_l583_583609

theorem min_value_a2_b2 (a b : ℝ) 
  (h : ∀ x : ℝ, f x = x^2 + a*x + b - 3) 
  (h1 : f 2 = 0) : a^2 + b^2 = 4 :=
by
  sorry

end min_value_a2_b2_l583_583609


namespace solve_for_m_l583_583199

theorem solve_for_m (m : ℝ) (x1 x2 : ℝ)
    (h1 : x1^2 - (2 * m - 1) * x1 + m^2 = 0)
    (h2 : x2^2 - (2 * m - 1) * x2 + m^2 = 0)
    (h3 : (x1 + 1) * (x2 + 1) = 3)
    (h_reality : (2 * m - 1)^2 - 4 * m^2 ≥ 0) :
    m = -3 := by
  sorry

end solve_for_m_l583_583199


namespace log_decreasing_interval_proof_l583_583002

noncomputable def log_decreasing_interval (x : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (1 < x1 ∧ 1 < x2 ∧ x1 < x2 → log (1/2) (2 * x1^2 - 3 * x1 + 1) > log (1/2) (2 * x2^2 - 3 * x2 + 1))

theorem log_decreasing_interval_proof : ∀ x : ℝ, (1 < x) → log_decreasing_interval x :=
by
  sorry

end log_decreasing_interval_proof_l583_583002


namespace part1_extremum_at_0_part2_increasing_function_l583_583117

noncomputable def f (x : ℝ) (a : ℝ) := (Real.sin x + a) / Real.exp x

-- Part (1)
theorem part1_extremum_at_0 (a : ℝ) (h_extremum : ∃ ε, ∀ x ∈ (0 - ε, 0 + ε), f x a ≤ f 0 a) : 
  a = 1 := 
sorry

-- Part (2)
theorem part2_increasing_function (a : ℝ) (h_increasing : ∀ x y, x < y → f x a ≤ f y a) : 
  a ≤ -Real.sqrt 2 := 
sorry

end part1_extremum_at_0_part2_increasing_function_l583_583117


namespace diagonal_sum_l583_583468

-- Definitions based on conditions
variables (A B C D E F : Type) [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry E] [Geometry F]

variable (circle_inscribed_hexagon : InscribedHexagon A B C D E F)

variables (AB : Segment A B) [AB.length = 39]
variables (BC CD DE EF FA : Segment A B) [BC.length = 81] [CD.length = 81] [DE.length = 81] [EF.length = 81] [FA.length = 81]

-- Diagonal lengths definitions
variables (AC : Segment A C) (AD : Segment A D) (AE : Segment A E)
variables (x y z : ℝ) [AC.length = x] [AD.length = y] [AE.length = z]

-- The proof goal
theorem diagonal_sum : x + y + z = 331.038 :=
sorry

end diagonal_sum_l583_583468


namespace sum_of_solutions_l583_583781

theorem sum_of_solutions (S : Finset ℝ) (h : ∀ x ∈ S, |x^2 - 10 * x + 29| = 3) : S.sum id = 0 :=
sorry

end sum_of_solutions_l583_583781


namespace bug_visits_tiles_l583_583472

-- Define the conditions
def width : ℕ := 18
def length : ℕ := 24

-- Define the GCD computation
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the number of tiles crossed
def tiles_crossed (w l : ℕ) : ℕ :=
w + l - gcd w l

-- Prove that the number of tiles the bug visits is 36
theorem bug_visits_tiles (h_w : width = 18) (h_l : length = 24) : tiles_crossed width length = 36 :=
by
  sorry

end bug_visits_tiles_l583_583472


namespace smallest_sum_of_sequence_l583_583006

theorem smallest_sum_of_sequence :
  ∃ A B C D : ℕ, A > 0 ∧ B > 0 ∧ C > 0 ∧
  (C - B = B - A) ∧ (C = 7 * B / 4) ∧ (D = 49 * B / 16) ∧
  (A + B + C + D = 97) :=
begin
  sorry
end

end smallest_sum_of_sequence_l583_583006


namespace min_value_expression_l583_583596

theorem min_value_expression (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  ∃ x : ℝ, x = 1 ∧ x = (3 * a - 2 * b + c) / (b - a) := 
  sorry

end min_value_expression_l583_583596


namespace mixed_number_sum_in_range_l583_583857

-- Definitions for mixed numbers as improper fractions
def mixed_to_improper(m : ℚ, n : ℚ) (d : ℚ) : ℚ := m + n / d

-- Specific mixed number conversions given in the problem
def sum_mixed_numbers : ℚ :=
  mixed_to_improper 3 1 8 + mixed_to_improper 4 1 3 + mixed_to_improper 6 1 21

-- Prove that the sum is within the given range
theorem mixed_number_sum_in_range : 13 < sum_mixed_numbers ∧ sum_mixed_numbers ≤ 14 :=
by
  let a := mixed_to_improper 3 1 8
  let b := mixed_to_improper 4 1 3
  let c := mixed_to_improper 6 1 21
  have : 13 < a + b + c := sorry -- Use the derivation steps here
  have : a + b + c ≤ 14 := sorry -- Use the derivation steps here
  exact ⟨this, this⟩

end mixed_number_sum_in_range_l583_583857


namespace evaluate_f_at_2a_plus_2_l583_583937

open Real

def f (x : ℝ) : ℝ :=
if x < 2 then x - 2^n else log x (x - 2)

theorem evaluate_f_at_2a_plus_2 (a : ℝ) : f (2^a + 2) = a :=
sorry

end evaluate_f_at_2a_plus_2_l583_583937


namespace call_charge_ratio_l583_583524

def elvin_jan_total_bill : ℕ := 46
def elvin_feb_total_bill : ℕ := 76
def elvin_internet_charge : ℕ := 16
def elvin_call_charge_ratio : ℕ := 2

theorem call_charge_ratio : 
  (elvin_feb_total_bill - elvin_internet_charge) / (elvin_jan_total_bill - elvin_internet_charge) = elvin_call_charge_ratio := 
by
  sorry

end call_charge_ratio_l583_583524


namespace relationship_among_abc_l583_583201

variable {R : Type*} [OrderedRing R] [Differentiable R R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f(x)
  
theorem relationship_among_abc (f : R → R) (hf_odd : odd_function f) (hf_deriv : ∀ x < 0, f(x) + x * deriv f(x) < 0)
  (a : R) (b : R) (c : R) :
  a = 3^0.3 * f (3^0.3) →
  b = (Real.log π 3) * f (Real.log π 3) →
  c = (Real.log 3 (1 / 9)) * f (Real.log 3 (1 / 9)) →
  c > a ∧ a > b :=
by
  intros ha hb hc
  sorry

end relationship_among_abc_l583_583201


namespace quadratic_algebraic_expression_l583_583232

theorem quadratic_algebraic_expression (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
    a + b - a * b = 2 := by
  sorry

end quadratic_algebraic_expression_l583_583232


namespace P_inter_Q_eq_set_l583_583958

noncomputable def P : set ℝ := {x | -x^2 + 3 * x + 4 < 0}
noncomputable def Q : set ℝ := {x | 2 * x - 5 > 0}

theorem P_inter_Q_eq_set : P ∩ Q = {x | x > 4} :=
by { sorry }

end P_inter_Q_eq_set_l583_583958


namespace coin_flip_probability_l583_583794

theorem coin_flip_probability :
  let p := (1 / 2 : ℝ) in
  let event_probability := p * p * (1 - p) * (1 - p) * (1 - p) in
  event_probability = (1 / 32 : ℝ) :=
by
  sorry

end coin_flip_probability_l583_583794


namespace smallest_n_satisfying_equation_l583_583875

theorem smallest_n_satisfying_equation : ∃ (k : ℤ), (∃ (n : ℤ), n > 0 ∧ n % 2 = 1 ∧ (n ^ 3 + 2 * n ^ 2 = k ^ 2) ∧ ∀ m : ℤ, (m > 0 ∧ m < n ∧ m % 2 = 1) → ¬ (∃ j : ℤ, m ^ 3 + 2 * m ^ 2 = j ^ 2)) ∧ k % 2 = 1 :=
sorry

end smallest_n_satisfying_equation_l583_583875


namespace fraction_value_l583_583034

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end fraction_value_l583_583034


namespace sum_of_angles_in_triangles_l583_583626

theorem sum_of_angles_in_triangles : 
  (∀ (A B C : Type) (anglesA anglesB anglesC : A → ℝ), 
    (anglesA 1 + anglesA 2 + anglesA 3 = 180) ∧ 
    (anglesB 4 + anglesB 5 + anglesB 6 = 180) ∧ 
    (anglesC 7 + anglesC 8 + anglesC 9 = 180)) → 
  (Σ anglesA 1 + Σ anglesA 2 + Σ anglesA 3 + Σ anglesB 4 + Σ anglesB 5 + Σ anglesB 6 + Σ anglesC 7 + Σ anglesC 8 + Σ anglesC 9 = 540) :=
by
  sorry

end sum_of_angles_in_triangles_l583_583626


namespace monotonically_increasing_sequence_b_bounds_l583_583866

theorem monotonically_increasing_sequence_b_bounds (b : ℝ) :
  (∀ n : ℕ, 0 < n → (n + 1)^2 + b * (n + 1) > n^2 + b * n) ↔ b > -3 :=
by
  sorry

end monotonically_increasing_sequence_b_bounds_l583_583866


namespace gcf_7fact_8fact_l583_583150

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l583_583150


namespace share_of_A_correct_l583_583084

theorem share_of_A_correct :
  let investment_A1 := 20000
  let investment_A2 := 15000
  let investment_B1 := 20000
  let investment_B2 := 16000
  let investment_C1 := 20000
  let investment_C2 := 26000
  let total_months1 := 5
  let total_months2 := 7
  let total_profit := 69900

  let total_investment_A := (investment_A1 * total_months1) + (investment_A2 * total_months2)
  let total_investment_B := (investment_B1 * total_months1) + (investment_B2 * total_months2)
  let total_investment_C := (investment_C1 * total_months1) + (investment_C2 * total_months2)
  let total_investment := total_investment_A + total_investment_B + total_investment_C

  let share_A := (total_investment_A : ℝ) / (total_investment : ℝ)
  let profit_A := share_A * (total_profit : ℝ)

  profit_A = 20500.99 :=
by
  sorry

end share_of_A_correct_l583_583084


namespace smallest_n_satisfying_f_gt_15_l583_583974

def sum_of_digits_right_of_decimal (x : ℚ) : ℕ := 
  (x - x.floor).digits.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits_right_of_decimal (1 / 7^n)

theorem smallest_n_satisfying_f_gt_15 :
  ∃ n : ℕ, 0 < n ∧ f(n) > 15 ∧ ∀ m : ℕ, 0 < m ∧ m < n → f(m) ≤ 15 :=
begin
  use 7,
  split,
  { norm_num },
  split,
  { sorry },
  { intros m hm,
    sorry }
end

end smallest_n_satisfying_f_gt_15_l583_583974


namespace gcd_7_8_fact_l583_583157

-- Define factorial function in lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the GCD function
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define specific factorial values
def f7 := fact 7
def f8 := fact 8

-- Theorem stating the gcd of 7! and 8!
theorem gcd_7_8_fact : gcd f7 f8 = 5040 := by
  sorry

end gcd_7_8_fact_l583_583157


namespace graph_passes_through_point_l583_583391

theorem graph_passes_through_point :
  ∀ (a : ℝ), 0 < a ∧ a < 1 → (∃ (x y : ℝ), (x = 2) ∧ (y = -1) ∧ (y = 2 * a * x - 1)) :=
by
  sorry

end graph_passes_through_point_l583_583391


namespace solve_for_m_l583_583200

theorem solve_for_m (m : ℝ) (x1 x2 : ℝ)
    (h1 : x1^2 - (2 * m - 1) * x1 + m^2 = 0)
    (h2 : x2^2 - (2 * m - 1) * x2 + m^2 = 0)
    (h3 : (x1 + 1) * (x2 + 1) = 3)
    (h_reality : (2 * m - 1)^2 - 4 * m^2 ≥ 0) :
    m = -3 := by
  sorry

end solve_for_m_l583_583200


namespace sequence_property_l583_583404

theorem sequence_property : ∃ (b c d : ℤ), 
  b = 2 ∧ c = -1 ∧ d = 1 ∧ 
  (∀ n : ℕ, a n = b * (sqrt (n + c)).toNat + d ∧ 
  (a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 3 ∧ a 4 = 3 ∧ a 5 = 5 ∧ a 6 = 5 ∧ a 7 = 5 ∧ a 8 = 5 ∧ a 9 = 5 ∧ ...)) →
  b + c + d = 2 :=
by {
  -- Define the sequence according to the given problem
  -- a_n = b[\sqrt{n+c}] + d
  let a : ℕ → ℤ := λ n, 2 * (Int.ofNat (Nat.sqrt (n + (-1))).toNat) + 1,
  -- Execute the proof to check if the property holds
  -- This essentially means proving the supplied sequence satisfies all properties
  have bcd : (2, -1, 1) := by 
  {
    unfold,
    sorry
  },
  -- Summing up b + c + d
  have sum_values : 2 + (-1) + 1 = 2 := by 
  {
    unfold,
    trivial
  },
  exact sum_values
}

end sequence_property_l583_583404


namespace prime_pairs_condition_l583_583140

open Nat

def satisfies_congruence (p : ℕ) : ℕ → ℕ → Prop :=
  λ x y => (y^2) % p = (x^3 - x) % p

theorem prime_pairs_condition (p : ℕ) (hp : Prime p)
  (hpxy : ∃ n : ℕ, n = p ∧
    (∀ x y : ℕ, 0 ≤ x ∧ x ≤ p ∧ 0 ≤ y ∧ y ≤ p → satisfies_congruence p x y) ∧
    ∃ m : ℕ, m = p ∧
    (card {(x, y) : ℕ × ℕ | 0 ≤ x ∧ x ≤ p ∧ 0 ≤ y ∧ y ≤ p ∧ satisfies_congruence p x y} = p)) :
  p = 2 ∨ (p % 4 = 3) :=
by
  sorry

end prime_pairs_condition_l583_583140


namespace exists_unique_poly_odd_degree_l583_583720

open Polynomial

-- Statement of the theorem to be proven
theorem exists_unique_poly_odd_degree (n : ℕ) (hn : n % 2 = 1) :
  ∃! (P : Polynomial ℚ), P.degree = n ∧ ∀ x, P (x - (1 / x)) = x^n - (1 / x^n) := 
sorry

end exists_unique_poly_odd_degree_l583_583720


namespace gcd_fac_7_and_8_equals_5040_l583_583152

theorem gcd_fac_7_and_8_equals_5040 : Nat.gcd 7! 8! = 5040 := 
by 
  sorry

end gcd_fac_7_and_8_equals_5040_l583_583152


namespace min_distinct_b_100_l583_583047

theorem min_distinct_b_100 (a : Fin 100 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
    ∃ b : Fin 100 → ℕ, (∀ i, b i = a i + Nat.gcdList (Finset.erase (Finset.univ) i).1.toList.map a) ∧ (Finset.card (Finset.image b Finset.univ) = 99) :=
begin
  sorry,
end

end min_distinct_b_100_l583_583047


namespace next_thursday_Aug_15_2012_l583_583330

/-- Define a function that returns the day of the week for August 15 of a given year. -/
def day_of_week_Aug_15 (year : ℕ) : Nat :=
  let days_in_year (y : ℕ) := if leap_year y then 366 else 365
  let starting_day_2010 := 0  -- Monday is 0
  let days_passed : Nat := (List.range' 2010 (year - 2010)).foldl (λ acc y => acc + days_in_year y) 0
  (starting_day_2010 + days_passed) % 7

/-- Prove that the next year when August 15 is a Thursday is 2012. -/
theorem next_thursday_Aug_15_2012 : (day_of_week_Aug_15 2012) = 4 := by
  sorry

end next_thursday_Aug_15_2012_l583_583330


namespace price_adjustment_50_percent_l583_583079

theorem price_adjustment_50_percent (P : ℝ) (h : P * (1 - ((x:ℝ / 100) ^ 2)) = 0.75 * P) : x = 50 :=
by 
  sorry

end price_adjustment_50_percent_l583_583079


namespace triangle_perimeter_l583_583941

/-- Given that the area of triangle ABC is (sqrt 3 / 2), AC = sqrt 3, and angle ABC = π / 3,
    the perimeter of triangle ABC is 3 + sqrt 3. -/
theorem triangle_perimeter 
  (A B C : Type) 
  (area_ABC : ℝ) 
  (AC : ℝ) 
  (angle_ABC : ℝ) 
  (perimeter : ℝ) 
  (h1 : area_ABC = sqrt 3 / 2) 
  (h2 : AC = sqrt 3) 
  (h3 : angle_ABC = π / 3) :
  perimeter = 3 + sqrt 3 := 
sorry

end triangle_perimeter_l583_583941


namespace surface_area_of_cube_96_l583_583911

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)
  
def is_cube (A B C D : ℝ × ℝ × ℝ) : Prop :=
  distance A B = 6 ∧ distance A C = 4 ∧ distance A D = real.sqrt 52

theorem surface_area_of_cube_96 (A B C D : ℝ × ℝ × ℝ) 
  (hA : A = (1, 2, 3)) (hB : B = (1, 8, 3)) 
  (hC : C = (5, 2, 3)) (hD : D = (5, 8, 3))
  (hcube : is_cube A B C D) : 
  6 * 4^2 = 96 :=
by
  sorry

end surface_area_of_cube_96_l583_583911


namespace polar_to_rectangular_conversion_l583_583122

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular 5 (5 * Real.pi / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end polar_to_rectangular_conversion_l583_583122


namespace largest_element_CUA_inter_B_is_3_l583_583220

-- Conditions from a)
def U : Set ℝ := Set.univ
def A : Set ℤ := {x : ℤ | - (x * x) + 5 * x ≤ 0}
def B : Set ℝ := {x : ℝ | x < 4}
def CUA : Set ℝ := U \ (A : Set ℝ)

-- Statement to prove the question == answer given the conditions
theorem largest_element_CUA_inter_B_is_3 : ∃ x : ℝ, x ∈ (CUA ∩ B) ∧ ∀ y : ℝ, y ∈ (CUA ∩ B) → y ≤ x ∧ x = 3 := by
  sorry

end largest_element_CUA_inter_B_is_3_l583_583220


namespace quadratic_positive_difference_l583_583510

theorem quadratic_positive_difference :
  ∀ x : ℝ, x^2 - 5 * x + 15 = x + 55 → x = 10 ∨ x = -4 →
  |10 - (-4)| = 14 :=
by
  intro x h1 h2
  have h3 : x = 10 ∨ x = -4 := h2
  have h4 : |10 - (-4)| = 14 := by norm_num
  exact h4

end quadratic_positive_difference_l583_583510


namespace max_subset_cardinality_l583_583729

-- Define what it means for a number to be a product of two consecutive integers
def is_product_of_consecutive_integers (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * (m + 1)

-- Define the conditions that A must satisfy
def satisfies_conditions (A : set ℤ) : Prop :=
  ∀ (k : ℤ) (a b : ℤ), a ∈ A → b ∈ A → ¬is_product_of_consecutive_integers (a + b + 30 * k)

-- Define the problem statement with the maximum cardinality as a hypothesis
theorem max_subset_cardinality : ∃ (A : set ℤ), A ⊆ {0, 1, ..., 29} ∧ satisfies_conditions A ∧ (set.card A = 8) :=
sorry

end max_subset_cardinality_l583_583729


namespace polar_to_rectangular_conversion_l583_583123

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular 5 (5 * Real.pi / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end polar_to_rectangular_conversion_l583_583123


namespace number_of_parallelograms_divisible_by_three_l583_583837

-- Definitions:
-- 1. Regular hexagon
-- 2. Hexagon divided into congruent parallelograms
variables {Hexagon : Type} [regular_hexagon : regular_hexagon Hexagon]
variables (divide_into_parallelograms : Hexagon → list (Parallelogram))

-- Problem Statement:
theorem number_of_parallelograms_divisible_by_three
  (H : hexagon_is_regular Hexagon) 
  (D : hexagon_divided_into_congruent_parallelograms Hexagon) : 
  ∃ n, D.card = 3 * n := sorry

end number_of_parallelograms_divisible_by_three_l583_583837


namespace Frank_time_correct_l583_583114

def Dave_time := 10
def Chuck_time := 5 * Dave_time
def Erica_time := 13 * Chuck_time / 10
def Frank_time := 12 * Erica_time / 10

theorem Frank_time_correct : Frank_time = 78 :=
by
  sorry

end Frank_time_correct_l583_583114


namespace arithmetic_sequence_a2_a8_l583_583998

variable {a : ℕ → ℝ}

-- given condition
axiom h1 : a 4 + a 5 + a 6 = 450

-- problem statement
theorem arithmetic_sequence_a2_a8 : a 2 + a 8 = 300 :=
by
  sorry

end arithmetic_sequence_a2_a8_l583_583998


namespace find_m_find_range_l583_583692

-- Define the function
def f (x m : ℝ) : ℝ := (1/3) * x^3 + m * x^2 - 3 * x + 1

-- Part 1: Define the conditions and prove the value of m
theorem find_m (x₁ x₂ m : ℝ) (h1 : x₁ < x₂) (h2 : (x₁ + x₂) / (x₁ * x₂) = 2 / 3)
  (h3 : (deriv (λ x : ℝ, f x m) x₁ = 0) ∧ (deriv (λ x : ℝ, f x m) x₂ = 0)) :
  m = 1 :=
sorry

-- Part 2: Define the interval and prove the range of f(x)
theorem find_range (m : ℝ) (h : m = 1) :
  set.range (λ x, f x m) ∩ set.Icc (0 : ℝ) 3 = set.Icc (-2/3 : ℝ) 10 :=
sorry

end find_m_find_range_l583_583692


namespace ellipse_standard_eq_area_of_triangle_l583_583583

theorem ellipse_standard_eq (ellipse_hyperbola_shared_foci : ∃ c, c^2 = 2)
    (eccentricity_ellipse : ∀ a, a > 0 → c = a * ( √2 / 2 )) :
    ∃ a b, a = 2 ∧ b = √2 ∧ (∀ x y, ( x^2 / a^2 ) + ( y^2 / b^2 ) = 1 ∨ ellipse_hyperbola_shared_foci ∧ eccentricity_ellipse) := 
begin
  sorry
end

theorem area_of_triangle (ellipse_eq : ∀ x y, (x^2 / 4) + (y^2 / 2) = 1)
    (line_passing_through_point_P : ∀ x y k, y = k * x + 1)
    (AP_2PB : ∀ x1 y1 x2 y2, ∃ x y, x = 2 * x2 ∧ y = 2*(y2-1) ∧ -x1 = x ∧ 1-y1 = y ) :
    ∃ area,  area = ( √126 / 8 ) ∧ ellipse_eq ∧ line_passing_through_point_P ∧ AP_2PB :=
begin
  sorry
end

end ellipse_standard_eq_area_of_triangle_l583_583583


namespace g_range_l583_583307

def g (x y z : ℝ) : ℝ :=
  (x^2) / (x^2 + y^2) + (y^2) / (y^2 + z^2) + (z^2) / (z^2 + x^2)

theorem g_range (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  0 < g x y z ∧ g x y z < 3 := by
  sorry

end g_range_l583_583307


namespace compute_p_plus_2q_plus_3r_plus_4s_l583_583285

noncomputable def num_holes : ℕ := 0
noncomputable def num_vertical_asymptotes : ℕ := 3
noncomputable def num_horizontal_asymptotes : ℕ := 1
noncomputable def num_oblique_asymptotes : ℕ := 0

theorem compute_p_plus_2q_plus_3r_plus_4s :
  let p := num_holes
      q := num_vertical_asymptotes
      r := num_horizontal_asymptotes
      s := num_oblique_asymptotes
  in p + 2 * q + 3 * r + 4 * s = 9 :=
by
  sorry

end compute_p_plus_2q_plus_3r_plus_4s_l583_583285


namespace le_condition_l583_583719

-- Given positive numbers a, b, c
variables {a b c : ℝ}
-- Assume positive values for the numbers
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
-- Given condition a² + b² - ab = c²
axiom condition : a^2 + b^2 - a*b = c^2

-- We need to prove (a - c)(b - c) ≤ 0
theorem le_condition : (a - c) * (b - c) ≤ 0 :=
sorry

end le_condition_l583_583719


namespace triangle_area_of_parabola_intersection_l583_583000

theorem triangle_area_of_parabola_intersection
  (line_passes_through : ∃ (p : ℝ × ℝ), p = (0, -2))
  (parabola_intersection : ∃ (x1 y1 x2 y2 : ℝ),
    (x1, y1) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst} ∧
    (x2, y2) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst})
  (y_cond : ∃ (y1 y2 : ℝ), y1 ^ 2 - y2 ^ 2 = 1) :
  ∃ (area : ℝ), area = 1 / 16 :=
by
  sorry

end triangle_area_of_parabola_intersection_l583_583000


namespace math_problem_l583_583584

open Real

noncomputable def ellipse (a b : ℝ) : set ℝ × ℝ :=
{ p | p.1^2 / a^2 + p.2^2 / b^2 = 1 }

def circle (h k r : ℝ) : set ℝ × ℝ :=
{ p | (p.1 - h)^2 + (p.2 - k)^2 = r^2 }

theorem math_problem 
  (a b : ℝ) 
  (h k r : ℝ) 
  (e : ℝ) 
  (A B O : ℝ × ℝ) 
  (C : set (ℝ × ℝ)) 
  (M : set (ℝ × ℝ)) 
  (l : set (ℝ × ℝ)) :
  a > b → b > 0 → e = 1 / 2 → 
  C = ellipse a b → 
  M = circle 0 3 2 → 
  (∃ C', C' ⊆ C ∧ C' ⊆ M ∧ ∀ p₁ p₂ ∈ C', p₁ ≠ p₂ → (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = 16) →
  (A = (4, 0)) → 
  (l = {p | p.2 = 1 / 3 * (p.1 - 4)} ∨ l = {p | p.2 = -1 / 3 * (p.1 - 4)}) → 
  (circle 0 0 (sqrt (8 / 5)).^2) = {p | p ∈ l} → 
  (∃ B, B ∈ l ∧ B ∈ {p | p.1^2 + p.2^2 = 8 / 5}) → 
  (C = ellipse 4 (2 * sqrt(3))) :=
sorry

end math_problem_l583_583584


namespace Piglet_ate_one_l583_583789

theorem Piglet_ate_one (V S K P : ℕ) (h1 : V + S + K + P = 70)
  (h2 : S + K = 45) (h3 : V > S) (h4 : V > K) (h5 : V > P) 
  (h6 : V ≥ 1) (h7 : S ≥ 1) (h8 : K ≥ 1) (h9 : P ≥ 1) : P = 1 :=
sorry

end Piglet_ate_one_l583_583789


namespace problem_solution_l583_583641

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 8 := 
by 
  sorry

end problem_solution_l583_583641


namespace x_y_sum_vals_l583_583637

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end x_y_sum_vals_l583_583637


namespace bus_driver_earnings_l583_583464

-- Definitions of given conditions
def regular_rate := 16        -- $16 per hour
def regular_hours := 40       -- 40 hours per week
def total_hours_worked := 54  -- 54 hours in the week
def overtime_multiplier := 1.75 -- 75% higher rate (1 + 0.75)

-- Calculate overtime hours
def overtime_hours := total_hours_worked - regular_hours

-- Calculate regular and overtime pay rates
def overtime_rate := regular_rate * overtime_multiplier

-- Calculate regular and overtime pays
def regular_pay := regular_rate * regular_hours
def overtime_pay := overtime_rate * overtime_hours

-- Calculate total compensation
def total_compensation := regular_pay + overtime_pay

-- Lean 4 statement to prove total compensation is $1032
theorem bus_driver_earnings : total_compensation = 1032 := by sorry

end bus_driver_earnings_l583_583464


namespace winnie_the_pooh_wins_l583_583026

variable (cones : ℕ)

def can_guarantee_win (initial_cones : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ), 
    (strategy initial_cones = 4 ∨ strategy initial_cones = 1) ∧ 
    ∀ n, (strategy n = 1 → (n = 2012 - 4 ∨ n = 2007 - 1 ∨ n = 2005 - 1)) ∧
         (strategy n = 4 → n = 2012)

theorem winnie_the_pooh_wins : can_guarantee_win 2012 :=
sorry

end winnie_the_pooh_wins_l583_583026


namespace right_triangle_hypotenuse_l583_583247

def is_nat (n : ℕ) : Prop := n > 0

theorem right_triangle_hypotenuse (x : ℕ) (x_pos : is_nat x) (consec : x + 1 > x) (h : 11^2 + x^2 = (x + 1)^2) : x + 1 = 61 :=
by
  sorry

end right_triangle_hypotenuse_l583_583247


namespace possible_values_count_l583_583921

theorem possible_values_count {x y z : ℤ} (h₁ : x = 5) (h₂ : y = -3) (h₃ : z = -1) :
  ∃ v, v = x - y - z ∧ (v = 7 ∨ v = 8 ∨ v = 9) :=
by
  sorry

end possible_values_count_l583_583921


namespace smallest_positive_integer_adding_to_725_is_5_l583_583427

theorem smallest_positive_integer_adding_to_725_is_5 :
  ∃ n : ℕ, n > 0 ∧ (725 + n) % 5 = 0 ∧ (∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m) :=
begin
  use 5,
  split,
  { exact nat.succ_pos' 4 },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  {
    intros m hm_mod hm_le,
    by_contra h,
    have : m < 5 := lt_of_not_ge h,
    have hm_lt_5 : (725 + m) % 5 < 5 := (nat.mod_lt (725 + m) (nat.zero_lt_succ 4)),
    linarith,
  }
end

end smallest_positive_integer_adding_to_725_is_5_l583_583427


namespace sufficient_not_necessary_condition_l583_583696

theorem sufficient_not_necessary_condition 
  (x : ℝ) 
  (h1 : |x - 1| < 2) 
  (h2 : -1 < x ∧ x < 5) :
  x^2 - 4x - 5 < 0 :=
begin
  -- Placeholder for the proof
  sorry
end

end sufficient_not_necessary_condition_l583_583696


namespace equilateral_triangle_perimeter_l583_583999

theorem equilateral_triangle_perimeter (a : ℕ) (h : a = 8) : 3 * a = 24 :=
by
  rw [h]
  norm_num

end equilateral_triangle_perimeter_l583_583999


namespace random_error_in_linear_regression_l583_583663

-- Define the conditions of the problem
def linear_regression_model (a b e x : ℝ) : ℝ := b * x + a + e

def expected_error_is_zero (e : ℝ) : Prop := 
  ExpectedValue e = 0

def variance_of_error_affects_accuracy (e : ℝ) : Prop := 
  ∃ (σ² : ℝ), variance e = σ²

-- State the problem as a theorem with the given conditions
theorem random_error_in_linear_regression (a b e x : ℝ) :
  (expected_error_is_zero e) ∧ (variance_of_error_affects_accuracy e) →
  is_random_error e :=
sorry

end random_error_in_linear_regression_l583_583663


namespace polynomial_solution_l583_583139

theorem polynomial_solution (P : ℝ[X])
  (h_nonconstant : P ≠ 0 ∧ ∀ (n : ℕ), P ≠ C (0 : ℝ))
  (h_real_coeffs : ∀ n, P.coeff n ∈ ℝ)
  (h_real_zeros : ∀ z, P.eval z = 0 → z ∈ ℝ)
  (h_functional_eq : ∀ x : ℝ, P.eval (x + 1) * P.eval (x^2 - x + 1) = P.eval (x^3 + 1)) :
  ∃ k : ℕ, P = X^k :=
begin
  sorry
end

end polynomial_solution_l583_583139


namespace integer_solutions_l583_583979

theorem integer_solutions (m : ℤ) :
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) → 
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + 1 / 2 = 0) →
  m = 3 :=
by
  sorry

end integer_solutions_l583_583979


namespace pqrs_sum_l583_583313

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l583_583313


namespace interest_calculation_correct_l583_583791

-- Define the principal amounts and their respective interest rates
def principal1 : ℝ := 3000
def rate1 : ℝ := 0.08
def principal2 : ℝ := 8000 - principal1
def rate2 : ℝ := 0.05

-- Calculate interest for one year
def interest1 : ℝ := principal1 * rate1 * 1
def interest2 : ℝ := principal2 * rate2 * 1

-- Define the total interest
def total_interest : ℝ := interest1 + interest2

-- Prove that the total interest calculated is $490
theorem interest_calculation_correct : total_interest = 490 := by
  sorry

end interest_calculation_correct_l583_583791


namespace smallest_int_proof_l583_583311
noncomputable def smallest_int(N n p : ℕ) : ℕ :=
  2^n - 2^(n-p) + (∑ k in finset.range (n / p), (-1 : ℕ)^k * 2^(k * p))

theorem smallest_int_proof (n p : ℕ) (hn : Composite n) (hp : ProperDivisor p n) : 
  ∃ N : ℕ, (1 + 2^p + 2^(n - p)) * N = 1 %[ (2^n) ] :=
begin
  let N := smallest_int 0 n p,
  use N,
  sorry
end

end smallest_int_proof_l583_583311


namespace problem_solution_l583_583585

noncomputable def problem_statement : Prop :=
  ∃ (C : ℝ × ℝ → Prop) (M A B : ℝ × ℝ) (k1 k2 a b : ℝ),
    (0 < b) ∧ (b < a) ∧ (C = λ p, (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1) ∧
    (a^2 = 2) ∧ (b = 1) ∧
    (∃ c, (c / a = sqrt 2 / 2) ∧ (b^2 + c^2 = a^2)) ∧
    (C (0,1)) ∧
    (∀ A', C A' → ∃ k, (A'.2 = k * A'.1 + 1) ∧ (k = k1 ∨ k = k2)) ∧
    (∀ B', C B' → ∃ k, (B'.2 = k * B'.1 + 1) ∧ (k = k1 ∨ k = k2)) ∧
    (k1 + k2 = 4) ∧
    ∀ p, (p = (-1 / 2, -1)) → ((A.1 - p.1) * (B.2 - p.2) - (A.2 - p.2) * (B.1 - p.1) = 0)

theorem problem_solution : problem_statement := sorry

end problem_solution_l583_583585


namespace calculate_value_of_x_l583_583764

noncomputable def machine_working_times := 
  let x : ℝ := 2
  let p_time := x + 4
  let q_time := x + 2
  let r_time := 2 * x + 6
  let total_combined := 
    1 / p_time + 1 / q_time + 1 / r_time = 1 / x
  total_combined

theorem calculate_value_of_x : 
  ∃ x : ℝ, 
    (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 6) = 1 / x) → x = 2 := 
by
  apply exists.intro
  use 2
  unfold machine_working_times
  sorry

end calculate_value_of_x_l583_583764


namespace largest_integer_N_l583_583873

def Table (m n : Nat) := Matrix (Fin m) (Fin n) Nat

def is_permutation (c : Vector Nat) : Prop :=
  ∀ (x : Nat), x ∈ {1, 2, 3, 4, 5, 6} → count x c = 1

def valid_table (T : Table 6 N) : Prop :=
  (∀ j, is_permutation (T.getColumn j)) ∧
  (∀ i j, i ≠ j → ∃ r, T.getFin r i = T.getFin r j) ∧
  (∀ i j, i ≠ j → ∃ s, T.getFin s i ≠ T.getFin s j)

theorem largest_integer_N :
  ∃ (N : Nat), (∀ (T : Table 6 N), valid_table T) ∧ (N = 120) :=
sorry

end largest_integer_N_l583_583873


namespace tourism_income_exceeds_investment_at_least_five_years_l583_583765

def initial_investment : ℝ := 8
def initial_income : ℝ := 4
def investment_rate : ℝ := 1 - (1 / 5)
def income_rate : ℝ := 1 + (1 / 4)

noncomputable def total_investment (n: ℕ) : ℝ := 
  initial_investment * (1 - investment_rate ^ n) / (1 - investment_rate) * 
    (investment_rate ≠ 1)

noncomputable def total_income (n: ℕ) : ℝ := 
  initial_income * (income_rate ^ n - 1) / (income_rate - 1) * 
    (income_rate ≠ 1)

theorem tourism_income_exceeds_investment_at_least_five_years :
  ∃ n : ℕ, n ≥ 5 ∧ total_income n > total_investment n := 
begin
  sorry
end

end tourism_income_exceeds_investment_at_least_five_years_l583_583765


namespace expected_lonely_cars_l583_583336

-- Define the conditions as a parameter
def are_cars_on_highway_driving_in_random_order (n : ℕ) : Prop := sorry

-- Define the main theorem about the expected number of lonely cars
theorem expected_lonely_cars (n : ℕ) (h : are_cars_on_highway_driving_in_random_order n) : 
  (∑ k in finset.range n, if k = n-1 then (1/n : ℚ) else (1/((k+1)*(k+1)) : ℚ)) = 1 :=
by sorry

end expected_lonely_cars_l583_583336


namespace exists_even_floor_l583_583326

theorem exists_even_floor (n : ℕ) : 
  ∃ k ∈ {0, 1, 2, ..., n}, even (int.floor (2^(n + k) * real.sqrt 2)) :=
sorry

end exists_even_floor_l583_583326


namespace locus_perpendicular_bisector_of_OM_l583_583574

noncomputable def locus_of_points_X
  (S : Type) [metric_space S] (center_S : S) (radius_S : ℝ)
  (M : S) (outside_S : M ∉ metric.closed_ball center_S radius_S) :
  set S :=
{ X : S | dist(X, center_of_S) = dist(X, M) }

theorem locus_perpendicular_bisector_of_OM
  (S : Type) [metric_space S] (center_S : S) (radius_S : ℝ)
  (M : S) (outside_S : M ∉ metric.closed_ball center_S radius_S)
  (S₁ : set S) (through_M : M ∈ S₁) (intersect_S : ∃ A B : S, A ∈ S ∧ B ∈ S ∧ A ∈ S₁ ∧ B ∈ S₁):
  ∀ X : S, X ∈ locus_of_points_X S center_S radius_S M outside_S ↔ 
            ∃ P : S, is_perp_bisector P (line_through center_S M) :=
sorry

end locus_perpendicular_bisector_of_OM_l583_583574


namespace gcf_fact7_fact8_l583_583142

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the values 7! and 8!
def fact_7 : ℕ := factorial 7
def fact_8 : ℕ := factorial 8

-- Prove that the greatest common factor of 7! and 8! is 7!
theorem gcf_fact7_fact8 : Nat.gcd fact_7 fact_8 = fact_7 :=
by
  sorry

end gcf_fact7_fact8_l583_583142


namespace vasya_max_pencils_l583_583705

theorem vasya_max_pencils (money_for_pencils : ℕ) (rebate_20 : ℕ) (rebate_5 : ℕ) :
  money_for_pencils = 30 → rebate_20 = 25 → rebate_5 = 10 → ∃ max_pencils, max_pencils = 36 :=
by
  intros h_money h_r20 h_r5
  sorry

end vasya_max_pencils_l583_583705


namespace square_side_length_l583_583058

noncomputable def segment_length : ℝ := real.sqrt 2 + real.sqrt (2 - real.sqrt 2)

theorem square_side_length
  (circle_touches_extensions : ∀ (AB AD : ℝ), true)
  (tangency_segment : (segment_length : ℝ))
  (angle_tangents_C : ∠ACB = 45)
  (sin_22_5: real.sin 22.5 = real.sqrt (2 - real.sqrt 2) / 2):
  true :=
begin
  sorry
end

end square_side_length_l583_583058


namespace range_of_a_l583_583649

theorem range_of_a (a : ℝ) (h : ∃ x0 ∈ set.Icc 0 1, x0 + (Real.exp(2) - 1) * Real.log a ≥ (2 * a) / Real.exp(x0) + Real.exp(2) * x0 - 2) : 1 ≤ a ∧ a ≤ Real.exp(3) :=
by
  -- Proof is omitted
  sorry

end range_of_a_l583_583649


namespace chord_inequality_l583_583276

theorem chord_inequality (O : Type*) [normed_field O] 
  (r : ℝ) (h_r : r = 1)
  (CD EF : set O)
  (P Q : O)
  (h_parallel : is_parallel CD EF)
  (h_angle : ∀ (AB : set O), intersects_at_angle AB CD 45 ∧ intersects_at_angle AB EF 45)
  (h_intersect_CD : ∃ d p : O, d ∈ CD ∧ p ∈ CD ∧ (diameter_ab_intersection p P) ∧ (diameter_ab_intersection d D))
  (h_intersect_EF : ∃ e q : O, e ∈ EF ∧ q ∈ EF ∧ (diameter_ab_intersection q Q) ∧ (diameter_ab_intersection e F)) :
  PC * QE + PD * QF < 2 :=
begin
  have h_r_pos : r > 0 := by linarith,
  sorry
end

end chord_inequality_l583_583276


namespace inv_mod_997_l583_583504

theorem inv_mod_997 : ∃ x : ℤ, 0 ≤ x ∧ x < 997 ∧ (10 * x) % 997 = 1 := 
sorry

end inv_mod_997_l583_583504


namespace find_sum_of_distinct_numbers_l583_583319

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l583_583319


namespace average_age_of_women_is_37_33_l583_583381

noncomputable def women_average_age (A : ℝ) : ℝ :=
  let total_age_men := 12 * A
  let removed_men_age := (25 : ℝ) + 15 + 30
  let new_average := A + 3.5
  let total_age_with_women := 12 * new_average
  let total_age_women := total_age_with_women -  (total_age_men - removed_men_age)
  total_age_women / 3

theorem average_age_of_women_is_37_33 (A : ℝ) (h_avg : women_average_age A = 37.33) :
  true :=
by
  sorry

end average_age_of_women_is_37_33_l583_583381


namespace no_polynomial_exists_l583_583354

noncomputable def bezout_theorem (P : ℕ → ℤ) : Prop := 
  ∀ a b : ℕ, a ≠ b → (P a - P b) % (a - b) = 0

theorem no_polynomial_exists (P : ℕ → ℤ) (hP : bezout_theorem P) :
  ¬ (P 6 = 5 ∧ P 14 = 9) :=
by
  intro h
  obtain ⟨h6, h14⟩ := h
  have hdiff : P 14 - P 6 = 4 := by linarith
  have hmod : (P 14 - P 6) % (14 - 6) = 4 % 8 := by rw hdiff
  simp at hmod
  have h4_div_8 : 4 % 8 = 4 := by norm_num
  rw h4_div_8 at hmod
  exact absurd hmod (by norm_num)

end no_polynomial_exists_l583_583354


namespace trajectory_equation_l583_583599

def pointA := (0, 0, 4)
def d (P A : ℝ × ℝ × ℝ) : ℝ := sorry  -- Distance formula between points
def trajectory (P : ℝ × ℝ × ℝ) : Prop := (d P pointA = 5)

theorem trajectory_equation (x y : ℝ) : 
  trajectory (x, y, 0) → (x^2 + y^2 = 9) := 
sorry

end trajectory_equation_l583_583599


namespace ice_cream_weekend_total_l583_583846

theorem ice_cream_weekend_total 
  (f : ℝ) (r : ℝ) (n : ℕ)
  (h_friday : f = 3.25)
  (h_saturday_reduction : r = 0.25)
  (h_num_people : n = 4)
  (h_saturday : (f - r * n) = 2.25)
  (h_sunday : 2 * ((f - r * n) / n) * n = 4.5) :
  f + (f - r * n) + (2 * ((f - r * n) / n) * n) = 10 := sorry

end ice_cream_weekend_total_l583_583846


namespace not_collinear_C_vector_decomposition_l583_583221

namespace VectorProof

open Function

structure Vector2 where
  x : ℝ
  y : ℝ

def add (v1 v2 : Vector2) : Vector2 := ⟨v1.x + v2.x, v1.y + v2.y⟩
def scale (c : ℝ) (v : Vector2) : Vector2 := ⟨c * v.x, c * v.y⟩

def collinear (v1 v2 : Vector2) : Prop :=
  ∃ k : ℝ, v2 = scale k v1

def vector_a : Vector2 := ⟨3, 4⟩
def e₁_C : Vector2 := ⟨-1, 2⟩
def e₂_C : Vector2 := ⟨3, -1⟩

theorem not_collinear_C :
  ¬ collinear e₁_C e₂_C :=
sorry

theorem vector_decomposition :
  ∃ (x y : ℝ), vector_a = add (scale x e₁_C) (scale y e₂_C) :=
sorry

end VectorProof

end not_collinear_C_vector_decomposition_l583_583221


namespace hexagon_vertices_trace_lines_and_intersect_at_common_point_l583_583836

theorem hexagon_vertices_trace_lines_and_intersect_at_common_point
  (A B C D E F O : Type)
  [regular_hexagon A B C D E F O]
  (line_o : line)
  (O_traces_line_o : O ∈ line_o) :
  (∀ (P : Type), ∃ (line_b line_c line_d line_e line_f : line), 
    B ∈ line_b ∧ C ∈ line_c ∧ D ∈ line_d ∧ E ∈ line_e ∧ F ∈ line_f ∧ 
    fixed_point P ∧ 
    P ∈ line_b ∧ P ∈ line_c ∧ P ∈ line_d ∧ P ∈ line_e ∧ P ∈ line_f) :=
begin
  sorry
end

end hexagon_vertices_trace_lines_and_intersect_at_common_point_l583_583836


namespace difference_c_and_d_l583_583107

theorem difference_c_and_d :
  let C := (List.range 20).map (λ n, (2 * n + 1) * (2 * n + 2)).sum + 41
  let D := 1 + (List.range 19).map (λ n, (2 * n + 2) * (2 * n + 3)).sum + 40 * 41
  C - D = -800 := by
  let C := (List.range 20).map (λ n, (2 * n + 1) * (2 * n + 2)).sum + 41
  let D := 1 + (List.range 19).map (λ n, (2 * n + 2) * (2 * n + 3)).sum + 40 * 41
  sorry

end difference_c_and_d_l583_583107


namespace bug_total_distance_l583_583461

theorem bug_total_distance :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let final_pos := 0
  let distance1 := |pos1 - pos2|
  let distance2 := |pos2 - pos3|
  let distance3 := |pos3 - final_pos|
  let total_distance := distance1 + distance2 + distance3
  total_distance = 29 := by
    sorry

end bug_total_distance_l583_583461


namespace pq_sum_expression_value_l583_583695

variable {a_n b_n : ℕ → ℝ}

-- Define the conditions given in the problem
def condition1 (n : ℕ) : Prop :=
  (sqrt 2 + sqrt 3) ^ (2 * n - 1) = a_n n * sqrt 2 + b_n n * sqrt 3

def an_bn_relation (n : ℕ) (p q : ℝ) : Prop :=
  a_n (n + 1) = p * a_n n + q * b_n n

-- Prove that p + q = 11 given the conditions
theorem pq_sum (n : ℕ) (p q : ℝ) (h1 : condition1 n) (h2 : an_bn_relation n p q) : 
  p + q = 11 :=
sorry

-- Prove that 2a_n^2 - 3b_n^2 = -1 given the conditions
theorem expression_value (h3 : ∀ n, 2 * (a_n n) ^ 2 - 3 * (b_n n) ^ 2 = -1) : 
  2 * (a_n n) ^ 2 - 3 * (b_n n) ^ 2 = -1 :=
sorry

end pq_sum_expression_value_l583_583695


namespace total_revenue_is_correct_l583_583467

def craftsman_jars : ℕ := 35
def marbles_in_jar : ℕ := 5
def price_jar : ℕ := 10
def price_pot : ℕ := 15

def pots : ℕ := (craftsman_jars : ℕ) / 2.5
def marbles_in_pot : ℕ := 4 * marbles_in_jar + 3

def jars_sold : ℕ := (0.75 * (craftsman_jars : ℕ)).to_nat
def pots_sold : ℕ := (0.60 * (pots : ℕ)).to_nat

def revenue_jars : ℕ := jars_sold * price_jar
def revenue_pots : ℕ := pots_sold * price_pot

def total_revenue : ℕ := revenue_jars + revenue_pots

theorem total_revenue_is_correct : total_revenue = 380 := by
  sorry

end total_revenue_is_correct_l583_583467


namespace line_intersection_and_conditions_l583_583222

theorem line_intersection_and_conditions :
  let l1 := (3 * x + 4 * y - 2 = 0) 
  let l2 := (2 * x + y + 2 = 0) 
  let P := (-2, 2)
  let d := (| 4 * -2 - 3 * 2 - 6 | / sqrt (4^2 + (-3)^2) = 4)
  let line_parallel := (3 * x - y + 8 = 0)
  let line_perpendicular := (x + 3 * y - 4 = 0)
  P ∈ l1 ∧ P ∈ l2 ∧ d ∧ line_parallel ∧ line_perpendicular :=
  by sorry

end line_intersection_and_conditions_l583_583222


namespace sum_of_inverse_poly_roots_l583_583508

noncomputable def cubic_roots_sum_inverse (p q r : ℝ) (h1 : (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧ (0 < p ∧ p < 2) ∧ (0 < q ∧ q < 2) ∧ (0 < r ∧ r < 2)) 
(h2 : (60 * p ^ 3 - 70 * p ^ 2 + 24 * p - 2 = 0) ∧ (60 * q ^ 3 - 70 * q ^ 2 + 24 * q - 2 = 0) ∧ (60 * r ^ 3 - 70 * r ^ 2 + 24 * r - 2 = 0)) : ℝ :=
  (1 / (2 - p)) + (1 / (2 - q)) + (1 / (2 - r))

theorem sum_of_inverse_poly_roots (p q r : ℝ) (h1 : (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧ (0 < p ∧ p < 2) ∧ (0 < q ∧ q < 2) ∧ (0 < r ∧ r < 2)) 
(h2 : (60 * p ^ 3 - 70 * p ^ 2 + 24 * p - 2 = 0) ∧ (60 * q ^ 3 - 70 * q ^ 2 + 24 * q - 2 = 0) ∧ (60 * r ^ 3 - 70 * r ^ 2 + 24 * r - 2 = 0)): 
  cubic_roots_sum_inverse p q r h1 h2 = 116 / 15 := 
  sorry

end sum_of_inverse_poly_roots_l583_583508


namespace tangent_circle_equation_l583_583218

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop := p.1 + p.2 = 0

-- Specify the symmetric point condition
def symmetric_point (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) (symmetric : ℝ × ℝ) : Prop :=
  l ((p.1 + symmetric.1) / 2, (p.2 + symmetric.2) / 2) ∧
  (p.2 - symmetric.2) / (p.1 + symmetric.1) = -1

-- Specify the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ :=
  |p.1 + p.2| / Real.sqrt 2

-- Definition of a circle with a center and radius
def circle_eq (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem tangent_circle_equation :
  ∃ (C : ℝ × ℝ), symmetric_point (-2, 0) line_l C → circle_eq C (distance_point_to_line C line_l) = λ p, x^2 + (y - 2)^2 = 2 :=
sorry

end tangent_circle_equation_l583_583218


namespace max_students_l583_583778

theorem max_students (pens pencils : ℕ) (h1 : pens = 1001) (h2 : pencils = 910) : Nat.gcd pens pencils = 91 := 
by
  rw [h1, h2]
  exact Nat.gcd_eq_right 91 -- Explains as gcd(1001, 910) = 91
  sorry -- Proof can continue from here

end max_students_l583_583778


namespace product_multiple_of_4_probability_l583_583677

/-- 
  Given:
  1. Juan rolls a fair regular octahedral die marked with the numbers 1 through 8.
  2. Amal rolls a fair twelve-sided die marked with the numbers 1 through 12.
  Prove:
  The probability that the product of the two rolls is a multiple of 4 is 7/16.
-/
theorem product_multiple_of_4_probability :
  let juan_roll := {1, 2, 3, 4, 5, 6, 7, 8}
  let amal_roll := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  -- calculating the probability of product being multiple of 4
  let event_multiple_of_4 := 
    (juan_roll.filter (λ x, x % 4 = 0)).card.to_rat / juan_roll.card +
    (juan_roll.filter (λ x, x % 4 ≠ 0)).card.to_rat / juan_roll.card *
    (amal_roll.filter (λ y, y % 4 = 0)).card.to_rat / amal_roll.card
  in event_multiple_of_4 = 7/16 := by sorry

end product_multiple_of_4_probability_l583_583677


namespace coefficient_of_x_in_sum_of_expansions_l583_583662

theorem coefficient_of_x_in_sum_of_expansions :
  (coeff (1 + x)^3 x + coeff (1 + x)^4 x + coeff (1 + x)^5 x + coeff (1 + x)^6 x + coeff (1 + x)^7 x) = 25 := by sorry

end coefficient_of_x_in_sum_of_expansions_l583_583662


namespace mode_and_median_correct_l583_583480

/-- Mode and Median of a set -/
def mode_and_median (s : Multiset ℕ) : ℕ × ℚ :=
  let mode := s.filter (λ x, Multiset.count x s = (Multiset.count s).values.max') |>.head
  let median := if (s.card % 2 = 0)
                then ((s.sort s).nth_le (s.card / 2) sorry + (s.sort s).nth_le (s.card / 2 - 1) sorry) / 2
                else (s.sort s).nth_le (s.card / 2) sorry
  (mode, median)

/-- Given a set {2, 7, 6, 3, 4, 7}, the mode is 7 and the median is 5 -/
theorem mode_and_median_correct :
  mode_and_median {2, 7, 6, 3, 4, 7} = (7, 5) :=
sorry

end mode_and_median_correct_l583_583480


namespace cos_difference_l583_583224

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem cos_difference (α β : ℝ) 
  (h₁ : let a := (real.cos α, real.sin α) in let b := (real.cos β, real.sin β) in vector_magnitude (a.1 - b.1, a.2 - b.2) = (2 * real.sqrt 5) / 5) :
  real.cos (α - β) = 3 / 5 :=
sorry

end cos_difference_l583_583224


namespace determine_k_l583_583280

noncomputable theory

-- Given conditions
variables {α : Type*} [linear_ordered_field α] [ne_zero d : α]
variables (a : ℕ → α) (k : ℕ)

-- Conditions of the problem
def arithmetic_sequence : Prop := 
  ∀ (n : ℕ), a n = a 0 + n * d

def geometric_mean_condition : Prop := 
  a k * a k = a 6 * a (k + 6)

-- The theorem to prove
theorem determine_k 
  (h_sequence : arithmetic_sequence a) 
  (h_a3 : a 3 = 0)
  (h_geometric : geometric_mean_condition a k)
  (h_d_nonzero : d ≠ 0) :
  k = 9 :=
begin
  sorry,
end

end determine_k_l583_583280


namespace solve_for_2023_minus_a_minus_2b_l583_583237

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end solve_for_2023_minus_a_minus_2b_l583_583237


namespace john_spent_on_wigs_l583_583676

def total_wig_cost (n_plays n_acts : ℕ) (wigs_per_act : ℕ)
                   (costs : ℕ → ℝ)
                   (dropout_play : ℕ) (discount : ℝ)
                   (refund_play : ℕ) (refund_rate : ℝ) : ℝ :=
  let cost_play (p : ℕ) := n_acts * wigs_per_act * costs p
  let adjusted_cost := cost_play dropout_play - (cost_play dropout_play * discount)
  let adjusted_refund := cost_play refund_play - (cost_play refund_play * refund_rate)
  (∑ p in ({1, 2, 3, 4} \ {dropout_play, refund_play}).to_finset, cost_play p) + adjusted_cost + adjusted_refund

theorem john_spent_on_wigs :
  total_wig_cost 4 8 3 (λ p, [0, 5, 6, 7, 8].get p) 2 0.25 4 0.10 = 352.80 :=
by
  sorry

end john_spent_on_wigs_l583_583676


namespace number_of_hens_l583_583792

theorem number_of_hens (H C : Nat) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := 
by
  sorry

end number_of_hens_l583_583792


namespace find_k_l583_583893

open Real

def vector_norm (v : ℝ × ℝ) : ℝ :=
  sqrt ((v.1) ^ 2 + (v.2) ^ 2)

theorem find_k (k : ℝ)
  (h : vector_norm (k * (3, 1) - (-5, 6)) = 5 * sqrt 5) :
  k = (-9 + sqrt 721) / 10 ∨ k = (-9 - sqrt 721) / 10 :=
sorry

end find_k_l583_583893


namespace positive_difference_of_two_numbers_l583_583411

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_of_two_numbers_l583_583411


namespace find_least_positive_integer_l583_583551

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l583_583551


namespace soup_feeding_problem_l583_583055

theorem soup_feeding_problem (C A : ℕ) (h1 : C = 6) (h2 : A = 4) (total_cans : ℕ) (initial_cans : total_cans = 10) (children_fed : ℕ) (children : children_fed = 30) :
  ∃ adults_fed : ℕ, adults_fed = 20 := 
by
  let used_cans := children_fed / C
  have rem_cans : total_cans - used_cans = 5 := sorry
  have adults_fed := rem_cans * A
  exact ⟨adults_fed, by sorry⟩

end soup_feeding_problem_l583_583055


namespace value_of_expression_l583_583234

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end value_of_expression_l583_583234


namespace perimeter_of_congruent_rectangle_l583_583019

theorem perimeter_of_congruent_rectangle (s : ℝ) : 
  (4 * s = 144) → (2 * (s + 0.25 * s)) = 90 :=
by
  assume h : 4 * s = 144
  sorry

end perimeter_of_congruent_rectangle_l583_583019


namespace group_C_forms_triangle_l583_583392

theorem group_C_forms_triangle :
  ∀ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) ↔ ((a, b, c) = (2, 3, 4)) :=
by
  -- we'll prove the forward and backward directions separately
  sorry

end group_C_forms_triangle_l583_583392


namespace log_inequality_solution_pairs_l583_583872

theorem log_inequality_solution_pairs (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha_ne_one : a ≠ 1) : 
  (log a b < log (a + 1) (b + 1)) ↔ 
  (b = 1 ∧ 0 < a ∧ a ≠ 1) ∨
  (1 < b ∧ b < a) ∨
  (a < 1 ∧ 1 < b) := 
sorry

end log_inequality_solution_pairs_l583_583872


namespace nancy_hours_to_work_l583_583335

def tuition := 22000
def scholarship := 3000
def hourly_wage := 10
def parents_contribution := tuition / 2
def student_loan := 2 * scholarship
def total_financial_aid := scholarship + student_loan
def remaining_tuition := tuition - parents_contribution - total_financial_aid
def hours_to_work := remaining_tuition / hourly_wage

theorem nancy_hours_to_work : hours_to_work = 200 := by
  -- This by block demonstrates that a proof would go here
  sorry

end nancy_hours_to_work_l583_583335


namespace max_u_plus_2v_l583_583691

theorem max_u_plus_2v (u v : ℝ) (h1 : 2 * u + 3 * v ≤ 10) (h2 : 4 * u + v ≤ 9) : u + 2 * v ≤ 6.1 :=
sorry

end max_u_plus_2v_l583_583691


namespace modulus_of_conjugate_z_l583_583944

def z : ℂ := (2 * Complex.I) / ((1 + Complex.I)^3)

theorem modulus_of_conjugate_z : Complex.abs (Complex.conj z) = Real.sqrt 2 / 2 :=
by
  sorry

end modulus_of_conjugate_z_l583_583944


namespace angle_P_of_extended_sides_l583_583364

noncomputable def regular_pentagon_angle_sum : ℕ := 540

noncomputable def internal_angle_regular_pentagon (n : ℕ) (h : 5 = n) : ℕ :=
  regular_pentagon_angle_sum / n

def interior_angle_pentagon : ℕ := 108

theorem angle_P_of_extended_sides (ABCDE : Prop) (h1 : interior_angle_pentagon = 108)
  (P : Prop) (h3 : 72 + 72 = 144) : 180 - 144 = 36 := by 
  sorry

end angle_P_of_extended_sides_l583_583364


namespace find_special_numbers_l583_583891

/-- A natural number N has the form 1000*x + 196 and removing the digits "196" from N
yields a number 1000 times smaller than N. The valid solutions for N are provided -/
theorem find_special_numbers:
  ∃ x : ℕ, N = 1000 * x + 196 ∧ (N - 196) / 1000 = x ↔ 
            N ∈ {1196, 2196, 4196, 7196, 14196, 49196, 98196} :=
sorry

end find_special_numbers_l583_583891


namespace second_derivative_value_l583_583946

def f (x : ℝ) := Real.cos x - Real.sin x

theorem second_derivative_value :
  (derivative^[2] f) (Real.pi / 6) = -(1 - Real.sqrt 3) / 2 := by
  sorry

end second_derivative_value_l583_583946


namespace current_intensity_leq_3_2_l583_583469

theorem current_intensity_leq_3_2 (P : ℝ) (U : ℝ) (R : ℝ) (hP : P = 800) (hU : U = 200) (hR : R ≥ 62.5) :
  (U / R) ≤ 3.2 :=
by
  have hI : (U / R) = 200 / R := by
    rw [hU]
  have hineq : 200 / R ≤ 200 / 62.5 := by
    apply (div_le_div_left _ _ _).mpr; linarith
  rw [hI] at hineq
  norm_num at hineq
  exact hineq

end current_intensity_leq_3_2_l583_583469


namespace complement_union_eq_l583_583619

noncomputable def A : set ℝ := {x | 1 < x ∧ x < 2}
noncomputable def B : set ℝ := {x | x^2 ≥ 2 }

theorem complement_union_eq :
  (A ∪ B)ᶜ = {x | -real.sqrt 2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end complement_union_eq_l583_583619


namespace paperboy_delivery_sequences_l583_583471

noncomputable def D : ℕ → ℕ
| 0       => 1  -- D_0 is a dummy value to facilitate indexing
| 1       => 2
| 2       => 4
| 3       => 7
| (n + 4) => D (n + 3) + D (n + 2) + D (n + 1)

theorem paperboy_delivery_sequences : D 11 = 927 := by
  sorry

end paperboy_delivery_sequences_l583_583471


namespace minimal_d1_l583_583754

theorem minimal_d1 :
  (∃ (S3 S6 : ℕ), 
    ∃ (d1 : ℚ), 
      S3 = d1 + (d1 + 1) + (d1 + 2) ∧ 
      S6 = d1 + (d1 + 1) + (d1 + 2) + (d1 + 3) + (d1 + 4) + (d1 + 5) ∧ 
      d1 = (5 * S3 - S6) / 9 ∧ 
      d1 ≥ 1 / 2) → 
  ∃ (d1 : ℚ), d1 = 5 / 9 := 
by 
  sorry

end minimal_d1_l583_583754


namespace brian_traveled_correct_distance_l583_583499

def miles_per_gallon : Nat := 20
def gallons_used : Nat := 3
def expected_miles : Nat := 60

theorem brian_traveled_correct_distance : (miles_per_gallon * gallons_used) = expected_miles := by
  sorry

end brian_traveled_correct_distance_l583_583499


namespace triangle_internal_region_l583_583484

-- Define the three lines forming the triangle
def line1 (x y : ℝ) : Prop := x + 2 * y = 2
def line2 (x y : ℝ) : Prop := 2 * x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the inequalities representing the internal region of the triangle
def region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2 * y < 2 ∧ 2 * x + y > 2

-- State that the internal region excluding the boundary is given by the inequalities
theorem triangle_internal_region (x y : ℝ) :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 x y) → region x y :=
  sorry

end triangle_internal_region_l583_583484


namespace min_value_expression_specific_a_value_range_of_k_l583_583209

def f (a x : ℝ) : ℝ := 3 - 2 * a - 2 * a * Real.cos x - 2 * (Real.sin x)^2

def g : ℝ → ℝ
| a if a ≤ -2       := 3
| a if -2 < a < 2  := -1/2 * a^2 - 2 * a + 1
| a if a ≥ 2       := 3 - 4*a

theorem min_value_expression :
  ∀ (a : ℝ), 
    g a = 
    if a ≤ -2 then 3
    else if -2 < a ∧ a < 2 then -1/2 * a^2 - 2 * a + 1
    else 3 - 4 * a := 
  sorry

theorem specific_a_value (a : ℝ) :
  g a = 5/2 ↔ a = -1 :=
  sorry

theorem range_of_k (k : ℝ) :
  ∀ (x : ℝ), 
    f (-1) x ≤ k ↔ k ≥ 7 :=
  sorry

end min_value_expression_specific_a_value_range_of_k_l583_583209


namespace A_inter_B_eq_A_A_union_B_l583_583698

-- Definitions for sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 3 * a = (a + 3) * x}
def B : Set ℝ := {x | x^2 + 3 = 4 * x}

-- Proof problem for part (1)
theorem A_inter_B_eq_A (a : ℝ) : (A a ∩ B = A a) ↔ (a = 1 ∨ a = 3) :=
by
  sorry

-- Proof problem for part (2)
theorem A_union_B (a : ℝ) : A a ∪ B = if a = 1 then {1, 3} else if a = 3 then {1, 3} else {a, 1, 3} :=
by
  sorry

end A_inter_B_eq_A_A_union_B_l583_583698


namespace find_a_b_and_perimeter_l583_583915

def is_isosceles_triangle (a b c : ℕ) : Prop :=
a = b ∨ b = c ∨ c = a

theorem find_a_b_and_perimeter :
  ∃ (a b : ℕ), (4 * a - 3 * b = 22) ∧ (2 * a + b = 16) ∧
  let perimeter := if a = b then 2 * a + b else 2 * b + a in
  perimeter = 16 :=
by
  sorry

end find_a_b_and_perimeter_l583_583915


namespace remainder_when_divided_by_7_l583_583436

theorem remainder_when_divided_by_7 (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ p = 5) : n % 7 = 5 :=
by
  sorry

end remainder_when_divided_by_7_l583_583436


namespace trapezoid_area_l583_583271

variables (P Q R S T : Type) 

def trapezoid (P Q R S : Type) := (P Q R S : Type) ∧ (∀ (pq : P Q) (rs : R S), pq.parallel rs)
def intersect_at (P Q R S T : Type) (d1 : P R) (d2 : Q S) := ∃ T, d1 ∩ d2 = T

variables [PQRS : trapezoid P Q R S] [PR_QS : intersect_at P Q R S T]

theorem trapezoid_area (h1 : area (triangle P Q T) = 70) (h2 : area (triangle P R T) = 30) :
  ∀ trapezoid_area : real, trapezoid_area = 142.857 :=
sorry

end trapezoid_area_l583_583271


namespace number_of_distinct_rationals_l583_583517

theorem number_of_distinct_rationals (L : ℕ) :
  L = 26 ↔
  (∃ (k : ℚ), |k| < 100 ∧ (∃ (x : ℤ), 7 * x^2 + k * x + 20 = 0)) :=
sorry

end number_of_distinct_rationals_l583_583517


namespace sqrt_sqrt_36_eq_pm_sqrt_6_l583_583408

theorem sqrt_sqrt_36_eq_pm_sqrt_6 : sqrt (sqrt 36) = ± sqrt 6 :=
by
  sorry

end sqrt_sqrt_36_eq_pm_sqrt_6_l583_583408


namespace ellipse_eccentricity_l583_583743

def c (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)

def e (a c : ℝ) : ℝ := c / a

theorem ellipse_eccentricity : 
  let a := 2
  let b := 1
  e a (c a b) = real.sqrt 3 / 2 := 
by
  sorry

end ellipse_eccentricity_l583_583743


namespace remainder_of_polynomial_division_l583_583874

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), 
  (Polynomial.X ^ 2 + Polynomial.X ^ 4 + 1) ∣ (Polynomial.X ^ 6 - 1) →
  ∃ r, r = 1 ∧ (Polynomial.X ^ 6 - 1) * (Polynomial.X ^ 4 - 1) ≡ r [MOD ((Polynomial.X ^ 2 + Polynomial.X ^ 4 + 1) : Polynomial ℂ)] :=
by
  sorry

end remainder_of_polynomial_division_l583_583874


namespace missed_field_goals_l583_583770

theorem missed_field_goals (TotalAttempts MissedFraction WideRightPercentage : ℕ) 
  (TotalAttempts_eq : TotalAttempts = 60)
  (MissedFraction_eq : MissedFraction = 15)
  (WideRightPercentage_eq : WideRightPercentage = 3) : 
  (TotalAttempts * (1 / 4) * (20 / 100) = 3) :=
  by
    sorry

end missed_field_goals_l583_583770


namespace band_song_average_l583_583749

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end band_song_average_l583_583749


namespace regression_line_l583_583615

theorem regression_line (x y : ℝ) (m : ℝ) (x1 y1 : ℝ)
  (h_slope : m = 6.5)
  (h_point : (x1, y1) = (2, 3)) :
  (y - y1) = m * (x - x1) ↔ y = 6.5 * x - 10 :=
by
  sorry

end regression_line_l583_583615


namespace separation_into_groups_l583_583762

theorem separation_into_groups :
  ∃ (partition_count : ℕ), 
  let lst := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      total_sum := 45,
      valid_divisors := [3, 5],
      target_partition_sums := λ n, total_sum / n,
      valid_partitions := (λ n, n ∈ valid_divisors) in
  partition_count = 32 ∧
  (∀ n, valid_partitions n → ∃ (groups : list (list ℕ)), 
    (∀ g ∈ groups, g.sum = target_partition_sums n) ∧ 
    ((groups.bind id).perm lst) ∧ 
    n = groups.length) :=
begin
  sorry
end

end separation_into_groups_l583_583762


namespace equivalent_single_discount_calculation_l583_583108

-- Definitions for the successive discounts
def discount10 (x : ℝ) : ℝ := 0.90 * x
def discount15 (x : ℝ) : ℝ := 0.85 * x
def discount25 (x : ℝ) : ℝ := 0.75 * x

-- Final price after applying all discounts
def final_price (x : ℝ) : ℝ := discount25 (discount15 (discount10 x))

-- Equivalent single discount fraction
def equivalent_discount (x : ℝ) : ℝ := 0.57375 * x

theorem equivalent_single_discount_calculation (x : ℝ) : 
  final_price x = equivalent_discount x :=
sorry

end equivalent_single_discount_calculation_l583_583108


namespace MN_passes_through_fixed_point_max_area_S_AMN_l583_583279

-- Define the conditions
variables (O : Circle) (B C : Point) (A : Point) (D E I H M N F : Point)
variables (h1 : B ∈ O) (h2 : C ∈ O) (h3 : B ≠ C) (h4 : ¬(BC is_diameter O))
variables (h5 : A ∈ O) (h6 : A ≠ B) (h7 : A ≠ C) (h8 : AB ≠ AC)
variables (h9 : D = lineBC ∩ (internal_bisector ∠BAC))
variables (h10 : E = lineBC ∩ (external_bisector ∠BAC))
variables (h11 : I = midpoint D E)
variables (h12 : H = orthocenter_triangle A B C)
variables (h13 : lineH_perpendicular_A_I ∩ AD = M)
variables (h14 : lineH_perpendicular_A_I ∩ AE = N)

-- Part 1: Prove that MN passes through a fixed point
theorem MN_passes_through_fixed_point :
  ∃ F : Point, is_fixed F ∧ MN ∋ F :=
sorry

-- Part 2: Determine the position of A such that S_AMN has maximum value
theorem max_area_S_AMN :
  ∃ A_optimal : A, maximizes_height A_optimal_I H F :=
sorry

end MN_passes_through_fixed_point_max_area_S_AMN_l583_583279


namespace last_fish_in_swamp_l583_583341

noncomputable def final_fish (perches pikes sudaks : ℕ) : String :=
  let p := perches
  let pi := pikes
  let s := sudaks
  if p = 6 ∧ pi = 7 ∧ s = 8 then "Sudak" else "Unknown"

theorem last_fish_in_swamp : final_fish 6 7 8 = "Sudak" := by
  sorry

end last_fish_in_swamp_l583_583341


namespace number_of_incorrect_statements_l583_583566

-- Definitions of each statement for clarity
def stmt1 := ({0} ∈ ({2, 3, 4} : Set ℕ)) = false
def stmt2 := (∅ ⊆ ({0} : Set ℕ)) = true
def stmt3 := ({-1, 0, 1} : Set ℤ) = ({0, -1, 1} : Set ℤ)
def stmt4 := (0 ∈ (∅ : Set ℕ)) = false

-- The theorem to be proven
theorem number_of_incorrect_statements : 
  (¬stmt1) ∧ stmt2 ∧ stmt3 ∧ (¬stmt4) → (2 = 2) :=
by
  sorry

end number_of_incorrect_statements_l583_583566


namespace s4_value_l583_583268

-- Definitions for the problem
noncomputable def a_n (a1 r : ℝ) (n : ℕ) := a1 * r^(n-1)

noncomputable def s (a1 r : ℝ) (n : ℕ) := ∑ k in finset.range n, a_n a1 r (k + 1)

-- Conditions stated in the problem
axiom s2_eq_7 (a1 r : ℝ) : s a1 r 2 = 7
axiom s6_eq_91 (a1 r : ℝ) : s a1 r 6 = 91

-- The theorem to prove
theorem s4_value (a1 r : ℝ) (pos_seq : ∀ n, a_n a1 r n > 0) : s a1 r 4 = 49 :=
by
  intros
  sorry

end s4_value_l583_583268


namespace negation_of_forall_ln_neq_l583_583003

theorem negation_of_forall_ln_neq (h : ∀ x ∈ set.Ioi (0 : ℝ), Real.log x ≠ x - 1) :
  ¬ (∃ x0 ∈ set.Ioi (0 : ℝ), Real.log x0 = x0 - 1) :=
sorry

end negation_of_forall_ln_neq_l583_583003


namespace number_of_students_taking_statistics_l583_583989

theorem number_of_students_taking_statistics
  (total_students : ℕ)
  (history_students : ℕ)
  (history_or_statistics : ℕ)
  (history_only : ℕ)
  (history_and_statistics : ℕ := history_students - history_only)
  (statistics_only : ℕ := history_or_statistics - history_and_statistics - history_only)
  (statistics_students : ℕ := history_and_statistics + statistics_only) :
  total_students = 90 → history_students = 36 → history_or_statistics = 59 → history_only = 29 →
    statistics_students = 30 :=
by
  intros
  -- Proof goes here but is omitted.
  sorry

end number_of_students_taking_statistics_l583_583989


namespace limsup_borel_set_lim_gt_borel_set_l583_583048

open Filter

variable {x : ℕ → ℝ}
variable {a : ℝ}

-- First statement: Proving the set where limsup is less than or equal to 'a' is in the Borel sigma-algebra
theorem limsup_borel_set (a : ℝ) :
  {x : ∀ n : ℕ, ℝ | limsup (fun n => x n) at_top ≤ a} ∈ borel (ℝ^∞) := 
sorry

-- Second statement: Proving the set where lim is greater than 'a' is in the Borel sigma-algebra
theorem lim_gt_borel_set (a : ℝ) :
  {x : ∀ n : ℕ, ℝ | tendsto (fun n => x n) at_top (𝓝 a) ∧ a > a} ∈ borel (ℝ^∞) :=
sorry

end limsup_borel_set_lim_gt_borel_set_l583_583048


namespace find_n_l583_583682

theorem find_n (n : ℕ) (h1 : n > 0) 
  (h2 : x = (sqrt (n+2) - sqrt n) / (sqrt (n+2) + sqrt n))
  (h3 : y = (sqrt (n+2) + sqrt n) / (sqrt (n+2) - sqrt n))
  (h4 : 14 * x^2 + 26 * x * y + 14 * y^2 = 2014) : 
  n = 5 := 
by
  sorry

end find_n_l583_583682


namespace least_positive_integer_l583_583539

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l583_583539


namespace smallest_positive_integer_x_l583_583903

theorem smallest_positive_integer_x :
  ∃ x : ℕ, 42 * x + 14 ≡ 4 [MOD 26] ∧ x ≡ 3 [MOD 5] ∧ x = 38 := 
by
  sorry

end smallest_positive_integer_x_l583_583903


namespace consecutive_primes_min_distinct_prime_factors_l583_583804

theorem consecutive_primes_min_distinct_prime_factors :
  ∀ p q : ℕ, prime p ∧ prime q ∧ p > 2 ∧ q > 2 ∧ ( ∀ r, prime r → r > p → r ≥ q ) → 
             (∃ m : ℕ, m = 1 ∧ (∀ n, prime n → n ∣ (p + q) → n = 2)) :=
by 
  intro p q h
  have pf1 : prime p := h.1
  have pf2 : prime q := h.2.1
  have hp2 : p > 2 := h.2.2.left
  have hq2 : q > 2 := h.2.2.right
  have hc : ∀ r, prime r → r > p → r ≥ q := h.2.2.right

  sorry

end consecutive_primes_min_distinct_prime_factors_l583_583804


namespace inequality_solution_l583_583372

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x + 3) < 0 ↔ x ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo (-3) 3 :=
by
  sorry

end inequality_solution_l583_583372


namespace student_d_not_top_student_l583_583977

structure Rankings :=
  (exam1 : ℕ)
  (exam2 : ℕ)
  (exam3 : ℕ)

def is_top_student (r : Rankings) : Prop :=
  r.exam1 ≤ 3 ∧ r.exam2 ≤ 3 ∧ r.exam3 ≤ 3

def mode_is_2 (r : Rankings) : Prop :=
  (r.exam1 = 2 ∧ r.exam2 = 2) ∨ (r.exam1 = 2 ∧ r.exam3 = 2) ∨ (r.exam2 = 2 ∧ r.exam3 = 2)

def variance_exceeds_1 (r : Rankings) : Prop :=
  let avg := (r.exam1 + r.exam2 + r.exam3) / 3 in
  let var := ((r.exam1 - avg) ^ 2 + (r.exam2 - avg) ^ 2 + (r.exam3 - avg) ^ 2) / 3 in
  var > 1

theorem student_d_not_top_student (r : Rankings) :
  mode_is_2 r ∧ variance_exceeds_1 r → ¬ is_top_student r :=
sorry

end student_d_not_top_student_l583_583977


namespace difference_of_numbers_l583_583755

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 22500) (h2 : b = 10 * a + 5) : b - a = 18410 :=
by
  sorry

end difference_of_numbers_l583_583755


namespace find_B_set_l583_583956

variable (a b : ℚ) -- Assuming a and b are rational for clear handling of division.

theorem find_B_set (h0 : b ≠ 0) (h1 : {1, a / b, b} = {0, a + b, b^2}) : 
  {0, -1, 1} = {0, a + b, b^2} :=
by 
  sorry

end find_B_set_l583_583956


namespace find_first_number_l583_583975

theorem find_first_number (n : ℝ) (h1 : n / 14.5 = 175) :
  n = 2537.5 :=
by 
  sorry

end find_first_number_l583_583975


namespace convert_250_to_base_12_l583_583513

def natToBase (n : ℕ) (b : ℕ) : List ℕ :=
  if b <= 1 then [n] else
    let rec convert (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n == 0 then acc else convert (n / b) ((n % b) :: acc)
    convert n []

theorem convert_250_to_base_12 : natToBase 250 12 = [1, 10] := by
  sorry

end convert_250_to_base_12_l583_583513


namespace inequality_reversal_l583_583231

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by
  sorry

end inequality_reversal_l583_583231


namespace longest_side_of_similar_triangle_l583_583753

-- Define the sides of the original triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 12

-- Define the perimeter of the similar triangle
def perimeter_similar_triangle : ℕ := 150

-- Formalize the problem using Lean statement
theorem longest_side_of_similar_triangle :
  ∃ x : ℕ, 8 * x + 10 * x + 12 * x = 150 ∧ 12 * x = 60 :=
by
  sorry

end longest_side_of_similar_triangle_l583_583753


namespace not_divisible_by_5_count_l583_583441

-- Define the total number of four-digit numbers using the digits 0, 1, 2, 3, 4, 5 without repetition
def total_four_digit_numbers : ℕ := 300

-- Define the number of four-digit numbers ending with 0
def numbers_ending_with_0 : ℕ := 60

-- Define the number of four-digit numbers ending with 5
def numbers_ending_with_5 : ℕ := 48

-- Theorem stating the number of four-digit numbers that cannot be divided by 5
theorem not_divisible_by_5_count : total_four_digit_numbers - numbers_ending_with_0 - numbers_ending_with_5 = 192 :=
by
  -- Proof skipped
  sorry

end not_divisible_by_5_count_l583_583441


namespace basketball_team_heights_l583_583708

theorem basketball_team_heights :
  ∃ (second tallest third fourth shortest : ℝ),
  (tallest = 80.5 ∧
   second = tallest - 6.25 ∧
   third = second - 3.75 ∧
   fourth = third - 5.5 ∧
   shortest = fourth - 4.8 ∧
   second = 74.25 ∧
   third = 70.5 ∧
   fourth = 65 ∧
   shortest = 60.2) := sorry

end basketball_team_heights_l583_583708


namespace det_sine_matrix_is_zero_l583_583506

theorem det_sine_matrix_is_zero : 
  ∀ A : Matrix (Fin 3) (Fin 3) ℝ, 
  (A = ![![sin 1, sin 2, sin 3], ![sin 4, sin 5, sin 6], ![sin 7, sin 8, sin 9]]) → 
  Matrix.det A = 0 :=
by
  sorry

end det_sine_matrix_is_zero_l583_583506


namespace unit_digit_23_pow_100000_l583_583033

theorem unit_digit_23_pow_100000 : (23^100000) % 10 = 1 := 
by
  -- Import necessary submodules and definitions

sorry

end unit_digit_23_pow_100000_l583_583033


namespace expected_value_balls_l583_583192

-- Define the initial contents of Bag A and Bag B
def bagA_initial := (red := 2, white := 3)
def bagB_initial := (red := 3, white := 3)

-- Define the random drawing and transferring operations
def transfer_ball (bagA bagB : {red : ℕ, white : ℕ}) : {red : ℕ, white : ℕ} :=
  let bagA' := { red := bagA.red - 1, white := bagA.white }
  let bagB' := { red := bagB.red + 1, white := bagB.white }
  { red := bagA'.red + 1, white := bagA'.white } -- Simplified operation for demonstration

-- Define the expected value calculation for the number of white balls in Bag A
def expected_white_balls_in_bagA : ℚ :=
  (3 * (1 - (2 / 5 * 3 / 7 + 3 / 5 * 3 / 7))
   + 4 * (2 / 5 * 3 / 7)
   + 2 * (3 / 5 * 3 / 7))

-- Problem statement in Lean
theorem expected_value_balls :
  expected_white_balls_in_bagA = 102 / 35 :=
by
  unfold expected_white_balls_in_bagA
  unfold transfer_ball
  sorry

end expected_value_balls_l583_583192


namespace unnamed_in_seat_3_l583_583131

def Person := {Name : String}

def Ella := {Name := "Ella"}
def Finn := {Name := "Finn"}
def Gabe := {Name := "Gabe"}
def Holly := {Name := "Holly"}
def Unnamed := {Name := "Unnamed"}

def Seat := {Number : Nat, Person : Person}

def seats : List Seat := [
    {Number := 1, Person := Holly},
    {Number := 2, Person := Finn},
    {Number := 3, Person := Unnamed},
    {Number := 4, Person := Gabe},
    {Number := 5, Person := Ella}
]

theorem unnamed_in_seat_3 :
  (seats.find? (λ s => s.Number = 3)).get! = {Number := 3, Person := Unnamed} :=
sorry

end unnamed_in_seat_3_l583_583131


namespace students_take_neither_l583_583656

variable (Total Mathematic Physics Both MathPhysics ChemistryNeither Neither : ℕ)

axiom Total_students : Total = 80
axiom students_mathematics : Mathematic = 50
axiom students_physics : Physics = 40
axiom students_both : Both = 25
axiom students_chemistry_neither : ChemistryNeither = 10

theorem students_take_neither :
  Neither = Total - (Mathematic - Both + Physics - Both + Both + ChemistryNeither) :=
  by
  have Total_students := Total_students
  have students_mathematics := students_mathematics
  have students_physics := students_physics
  have students_both := students_both
  have students_chemistry_neither := students_chemistry_neither
  sorry

end students_take_neither_l583_583656


namespace sum_of_arithmetic_sequence_l583_583660

variable {a : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a_8_eq_8 : a 8 = 8

-- Proof problem statement
theorem sum_of_arithmetic_sequence :
  is_arithmetic_sequence a →
  a 8 = 8 →
  (∑ i in Finset.range 15, a (i + 1)) = 120 :=
by
  intros h_as h_a8
  sorry

end sum_of_arithmetic_sequence_l583_583660


namespace imaginary_part_of_fraction_l583_583453

open Complex

theorem imaginary_part_of_fraction :
  let i := Complex.I in
  imagPart ((-25 * i) / (3 + 4 * i)) = -3 :=
by
  sorry

end imaginary_part_of_fraction_l583_583453


namespace complement_of_exponential_set_l583_583955

theorem complement_of_exponential_set :
  (set.univ \ {x : ℝ | 2^x < 1}) = {x : ℝ | 0 <= x} :=
by
  sorry

end complement_of_exponential_set_l583_583955


namespace derivative_cos_2x_correct_l583_583383

noncomputable def derivative_cos_2x : Prop :=
  ∀ (x : ℝ), has_deriv_at (λ x, Real.cos (2 * x)) (-2 * Real.sin (2 * x)) x

theorem derivative_cos_2x_correct : derivative_cos_2x :=
sorry

end derivative_cos_2x_correct_l583_583383


namespace nearest_integer_power_l583_583779

noncomputable def power_expression := (3 + Real.sqrt 2)^6

theorem nearest_integer_power :
  Int.floor power_expression = 7414 :=
sorry

end nearest_integer_power_l583_583779


namespace coefficient_of_middle_term_l583_583284

noncomputable def middle_coefficient (n : ℕ) (x : ℕ) := (choose n (n / 2)) * (-2)^(n/2)

theorem coefficient_of_middle_term (n : ℕ) (sum_even_coeff : ℕ) (h : sum_even_coeff = 128) :
  ∃ (m : ℕ), middle_coefficient 8 m = 1120 := 
by 
  have H : 2^n = 256 := sorry,
  have n_eq_8 : n = 8 := by {
    -- Based on the condition provided
    sorry 
  },
  show middle_coefficient 8 1 = 1120,
  sorry

end coefficient_of_middle_term_l583_583284


namespace least_positive_integer_l583_583546

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l583_583546


namespace books_not_sold_l583_583818

theorem books_not_sold (B : ℕ)
  (h1 : (2 / 3 : ℚ) * B * 4.25 = 255)
  : (1 / 3 : ℚ) * B = 30 :=
begin
  sorry
end

end books_not_sold_l583_583818


namespace work_done_example_l583_583195

variable (F : ℝ → ℝ)
variable (M : Type)

def work_done_by_force (F : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, F x

theorem work_done_example :
  (F x = x^2 + 1) →
  work_done_by_force F 0 6 = 78 :=
by
  intro hF
  rw [work_done_by_force, hF]
  sorry

end work_done_example_l583_583195


namespace coordinates_of_W_l583_583886

theorem coordinates_of_W
  (O : Point := (0, 0))
  (S : Point := (3, 3))
  (U : Point := (3, 0))
  (V : Point := (0, 3))
  (W : Point := (0, -9)) :
  let side_length := 3
  let area_square := side_length * side_length
  let area_triangle := 2 * area_square
  ∃ W : Point, 
    let base := (U.1 - V.1).abs
    let height := (W.2 - V.2).abs
    area_triangle = (1/2 : ℝ) * base * height :=
begin
  sorry
end

end coordinates_of_W_l583_583886


namespace jerry_needs_to_complete_the_collection_l583_583674

theorem jerry_needs_to_complete_the_collection :
  let current_action_figures := 7
  let total_action_figures_needed := 25
  let cost_per_action_figure := 12
  let action_figures_needed := total_action_figures_needed - current_action_figures
  let total_cost := action_figures_needed * cost_per_action_figure
  total_cost = 216 :=
by
  let current_action_figures := 7
  let total_action_figures_needed := 25
  let cost_per_action_figure := 12
  let action_figures_needed := total_action_figures_needed - current_action_figures
  let total_cost := action_figures_needed * cost_per_action_figure
  show total_cost = 216 from sorry

end jerry_needs_to_complete_the_collection_l583_583674


namespace trivia_team_points_l583_583843

theorem trivia_team_points (total_members absent_members total_points : ℕ) 
    (h1 : total_members = 5) 
    (h2 : absent_members = 2) 
    (h3 : total_points = 18) 
    (h4 : total_members - absent_members = present_members) 
    (h5 : total_points = present_members * points_per_member) : 
    points_per_member = 6 :=
  sorry

end trivia_team_points_l583_583843


namespace sin_of_fourth_quadrant_l583_583245

theorem sin_of_fourth_quadrant (α : ℝ) 
  (h1 : α ∈ Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.cos α = 12 / 13) :
  Real.sin α = -5 / 13 := 
by 
  sorry

end sin_of_fourth_quadrant_l583_583245


namespace probability_sum_five_eq_one_third_l583_583010

def set_numbers : Finset ℕ := {1, 2, 3, 4}

def favorable_pairs : Finset (ℕ × ℕ) :=
  { (1, 4), (2, 3) }

def all_possible_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2) (set_numbers.product set_numbers)

def probability_sum_five : ℚ :=
  favorable_pairs.card.to_rat / all_possible_pairs.card.to_rat

theorem probability_sum_five_eq_one_third :
  probability_sum_five = 1 / 3 := by
    sorry

end probability_sum_five_eq_one_third_l583_583010


namespace solve_for_x_l583_583945

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 - 1 else 3 * x

theorem solve_for_x (x : ℝ) : f x = 15 ↔ x = -4 ∨ x = 5 := by
  sorry

end solve_for_x_l583_583945


namespace cos_pi_third_minus_2theta_l583_583936

-- Given condition
def theta : ℝ
axiom sin_theta_minus_pi_six : Real.sin (theta - Real.pi / 6) = Real.sqrt 3 / 3

-- Prove that cos (π/3 - 2θ) = 1/3
theorem cos_pi_third_minus_2theta : Real.cos (Real.pi / 3 - 2 * theta) = 1 / 3 :=
by
  -- The proof is omitted, only the statement is required
  sorry

end cos_pi_third_minus_2theta_l583_583936


namespace least_positive_integer_l583_583543

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l583_583543


namespace sidney_monday_jumping_jacks_l583_583365

/-- Sidney's total jumping jacks including Monday, 36 on Tuesday, 40 on Wednesday, and 50 on Thursday. -/
def sidney_total_jumping_jacks (M : ℕ) : ℕ := M + 36 + 40 + 50

/-- Brooke's jumping jacks are three times of Sidney's total jumping jacks. -/
def brooke_jumping_jacks (S : ℕ) : ℕ := 3 * S

/-- Sidney's total jumping jacks from Monday to Thursday matches Brooke's given total. -/
theorem sidney_monday_jumping_jacks :
  ∃ (M : ℕ), sidney_total_jumping_jacks M = 438 / 3 → M = 20 :=
by
  intro M
  intro h
  have h1 : sidney_total_jumping_jacks M = sidney_total_jumping_jacks (438 / 3 - 126) := sorry
  have h2 : M = 20 := sorry
  exact h2
  sorry

end sidney_monday_jumping_jacks_l583_583365


namespace sequence_general_term_series_sum_correct_l583_583927

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions: a_n are positive, S_n is the sum of first n terms, √(S_n) is mean(1, a_n)
variable (h1 : ∀ n : ℕ, 0 < a n)
variable (h2 : ∀ n : ℕ, S n = ∑ i in Finset.range n, a i)
variable (h3 : ∀ n : ℕ, Real.sqrt (S n) = (1 + a n) / 2)

-- Question (i): General term formula of sequence {a_n}
def general_term (n : ℕ) : ℝ := 2 * n - 1

theorem sequence_general_term (n : ℕ) : a n = 2 * n - 1 :=
  sorry

-- Define series sum term
def series_term (n : ℕ) : ℝ :=
  2 / (a n * a (n + 1))

-- Question (ii): Sum of first n terms of the series {2 / (a_n * a_{n+1})}
def series_sum (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, series_term a i

def expected_sum (n : ℕ) : ℝ :=
  2 * n / (2 * n + 1)

theorem series_sum_correct (n : ℕ) : series_sum a n = expected_sum n :=
  sorry

end sequence_general_term_series_sum_correct_l583_583927


namespace simplify_trig_expression_l583_583366

theorem simplify_trig_expression (x : ℝ) :
  (1 - sin x - cos x) / (1 - sin x + cos x) = -tan (x / 2) :=
sorry

end simplify_trig_expression_l583_583366


namespace smallest_integer_to_make_multiple_of_five_l583_583430

/-- The smallest positive integer that can be added to 725 to make it a multiple of 5 is 5. -/
theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k : ℕ, k > 0 ∧ (725 + k) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → k ≤ m :=
sorry

end smallest_integer_to_make_multiple_of_five_l583_583430


namespace minimum_f_l583_583949

def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

theorem minimum_f : ∃ x, f x = 3 :=
by
  use 3
  unfold f
  sorry

end minimum_f_l583_583949


namespace arbitrarily_large_circle_no_intersection_any_circle_intersects_countable_lines_l583_583923

-- Definition: A finite set of lines in the plane
variable {L : set (set (ℝ × ℝ))} [finite L]

-- Theorem: An arbitrarily large circle can be drawn which does not intersect any of the lines in a finite set of lines in the plane.
theorem arbitrarily_large_circle_no_intersection (L : set (set (ℝ × ℝ))) [finite L] :
  ∃ (R : ℝ), ∃ (x₀ y₀ : ℝ), 0 < R ∧ ∀ line ∈ L, ¬ ∃ p ∈ line, (p.1 - x₀)^2 + (p.2 - y₀)^2 = R^2 :=
sorry

-- Definition: A countable set of lines in the plane (lines through rational points parallel to coordinate axes)
def countable_lines_through_rationals : set (set (ℝ × ℝ)) :=
  { line | ∃ r : ℚ, line = { p | p.1 = r } ∨ line = { p | p.2 = r } }

-- Theorem: Any circle of positive radius intersects at least one line from a given countable set of lines in the plane.
theorem any_circle_intersects_countable_lines :
  ∀ (R : ℝ) (x₀ y₀ : ℝ), 0 < R → ∃ line ∈ countable_lines_through_rationals,
    ∃ p ∈ line, (p.1 - x₀)^2 + (p.2 - y₀)^2 = R^2 :=
sorry

end arbitrarily_large_circle_no_intersection_any_circle_intersects_countable_lines_l583_583923


namespace b50_value_l583_583926

def sequence (n : ℕ) : ℝ := if n = 1 then 3 else (3 * (T n)^3) / (3 * (T n) - 2)

def T (n : ℕ) : ℝ := if n = 0 then 0 else (∑ k in finset.range n, sequence (k + 1))

theorem b50_value : sequence 50 = -9 / 21460 :=
sorry

end b50_value_l583_583926


namespace find_wall_width_l583_583057

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.1325
def brick_height : ℝ := 0.08

-- Define the dimensions of the wall in meters
def wall_length : ℝ := 7
def wall_height : ℝ := 15.5
def number_of_bricks : ℝ := 4094.3396226415093

-- Volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Total volume of bricks used
def total_brick_volume : ℝ := number_of_bricks * brick_volume

-- Wall volume in terms of width W
def wall_volume (W : ℝ) : ℝ := wall_length * W * wall_height

-- The theorem we want to prove
theorem find_wall_width (W : ℝ) (h : wall_volume W = total_brick_volume) : W = 0.08 := by
  sorry

end find_wall_width_l583_583057


namespace cards_can_be_arranged_in_any_order_in_5_operations_cards_cannot_be_arranged_in_reverse_order_in_4_operations_l583_583860

theorem cards_can_be_arranged_in_any_order_in_5_operations :
  ∀ (cards : List ℕ), (sorted (>) cards → length cards = 32 → (∀ target : List ℕ, sorted (≤) target → length target = 32 → 
  ∃ (operations : ℕ), operations ≤ 5 ∧ (∀ (k : ℕ), k < operations → move_top_part cards target))) :=
begin
  sorry
end

theorem cards_cannot_be_arranged_in_reverse_order_in_4_operations :
  ∀ (cards : List ℕ), (sorted (≤) cards → length cards = 32 → ∃ (target : List ℕ), sorted (≥) target → length target = 32 → 
  ∃ (operations : ℕ), operations ≤ 4 ∧ (∀ (k : ℕ), k < operations → move_top_part cards target) → (disorder target > 0))) :=
begin
  sorry
end

-- Definition of move_top_part: a function to model the allowed operation.
def move_top_part (cards target : List ℕ) : Prop :=
sorry

-- Definition of disorder: a function to count disorders in a list.
def disorder (l : List ℕ) : ℕ :=
sorry

end cards_can_be_arranged_in_any_order_in_5_operations_cards_cannot_be_arranged_in_reverse_order_in_4_operations_l583_583860


namespace expression_evaluation_l583_583167

def floor (x : ℝ) : ℤ := Int.floor x

theorem expression_evaluation (y : ℝ) (hy : y = 8.4) :
    (floor 6.5 : ℝ) * floor (2 / 3) + (floor 2 : ℝ) * 7.2 + floor y - 6.2 = 16.2 := by
  have h1 : (floor 6.5 : ℝ) = 6 := by sorry
  have h2 : (floor (2 / 3) : ℝ) = 0 := by sorry
  have h3 : (floor 2 : ℝ) = 2 := by sorry
  have h4 : floor y = 8 := by rw [hy]; sorry
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end expression_evaluation_l583_583167


namespace least_positive_integer_l583_583537

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l583_583537


namespace tory_earns_more_than_bert_l583_583494

-- Define the initial prices of the toys
def initial_price_phones : ℝ := 18
def initial_price_guns : ℝ := 20

-- Define the quantities sold by Bert and Tory
def quantity_phones : ℕ := 10
def quantity_guns : ℕ := 15

-- Define the discounts
def discount_phones : ℝ := 0.15
def discounted_phones_quantity : ℕ := 3

def discount_guns : ℝ := 0.10
def discounted_guns_quantity : ℕ := 7

-- Define the tax
def tax_rate : ℝ := 0.05

noncomputable def bert_initial_earnings : ℝ := initial_price_phones * quantity_phones

noncomputable def tory_initial_earnings : ℝ := initial_price_guns * quantity_guns

noncomputable def bert_discount : ℝ := discount_phones * initial_price_phones * discounted_phones_quantity

noncomputable def tory_discount : ℝ := discount_guns * initial_price_guns * discounted_guns_quantity

noncomputable def bert_earnings_after_discount : ℝ := bert_initial_earnings - bert_discount

noncomputable def tory_earnings_after_discount : ℝ := tory_initial_earnings - tory_discount

noncomputable def bert_tax : ℝ := tax_rate * bert_earnings_after_discount

noncomputable def tory_tax : ℝ := tax_rate * tory_earnings_after_discount

noncomputable def bert_final_earnings : ℝ := bert_earnings_after_discount + bert_tax

noncomputable def tory_final_earnings : ℝ := tory_earnings_after_discount + tory_tax

noncomputable def earning_difference : ℝ := tory_final_earnings - bert_final_earnings

theorem tory_earns_more_than_bert : earning_difference = 119.805 := by
  sorry

end tory_earns_more_than_bert_l583_583494


namespace find_circle_equation_l583_583196

-- Define the conditions on the circle
def passes_through_points (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (c = center ∧ r = radius) ∧ 
  dist (0, 2) c = r ∧ dist (0, 4) c = r

def lies_on_line (center : ℝ × ℝ) : Prop :=
  2 * center.1 - center.2 - 1 = 0

-- Define the problem
theorem find_circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  passes_through_points center radius ∧ lies_on_line center ∧ 
  (∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 
  ↔ (x - 2)^2 + (y - 3)^2 = 5) :=
sorry

end find_circle_equation_l583_583196


namespace smallest_number_of_distinct_pairwise_sums_and_products_l583_583909

theorem smallest_number_of_distinct_pairwise_sums_and_products (a b c d : ℤ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  6 ≤ cardinal.mk (set.image2 (+) {a, b, c, d} {a, b, c, d} ∪ set.image2 (*) {a, b, c, d} {a, b, c, d}) :=
by {
  sorry
}

end smallest_number_of_distinct_pairwise_sums_and_products_l583_583909


namespace disc_completely_covers_squares_l583_583816

theorem disc_completely_covers_squares (D : ℝ) (hD : 0 < D) :
  let diameter := 2 * D,
      radius := D,
      side_length := D,
      board_size := 10 * D in
  let disc_area := π * radius^2,
      total_squares := 10 * 10 in
  let covered_squares := 16 in
  covered_squares = 16 :=
by
  -- Here, we assume the necessary conditions and the problem details provided.
  sorry

end disc_completely_covers_squares_l583_583816


namespace incorrect_statement_C_l583_583214

theorem incorrect_statement_C (ω φ : ℝ) (h1 : ω > 0) (h2 : φ > 0)
    (h3 : ∀ x, sin (ω * x + φ) = sin (ω * (x + π)))
    (h4 : ∀ x, sin (ω * x + φ) ≤ sin (ω * (π / 8) + φ)) :
    ¬∀ x, (x > 3 * π / 8) ∧ (x < 5 * π / 8) → (2 * x + φ) > (2 * (3 * π /8) + φ) :=
by sorry

end incorrect_statement_C_l583_583214


namespace tetrahedron_edge_length_l583_583475

theorem tetrahedron_edge_length (s : ℕ) :
  let n := s^3 / 6         -- number of smaller tetrahedra
  let total_faces := 4 * n -- total number of faces on smaller tetrahedra
  let blue_faces := 4 * s * (s - 1) / 2 -- faces becoming blue after cutting
  (blue_faces : ℝ) / (total_faces : ℝ) = 1 / 4 := 
  s = 4 :=
sorry

end tetrahedron_edge_length_l583_583475


namespace shortest_distance_is_altitude_l583_583406

-- Define a triangle in an abstract sense
structure Triangle :=
  (A B C : ℝ × ℝ) -- points in a plane

-- Definition of an altitude from a vertex to the opposite side in a triangle
def altitude (T : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  let (v1, v2) := if vertex = T.A then (T.B, T.C)
                  else if vertex = T.B then (T.A, T.C)
                  else (T.A, T.B) in
  -- Here we would compute the distance from 'vertex' to the line through v1 and v2
  sorry -- We skip the actual implementation for simplicity

-- Theorem stating the shortest distance from an angle to the opposite side is the altitude
theorem shortest_distance_is_altitude (T : Triangle) (vertex : ℝ × ℝ)
  (vertex_is_angle : vertex = T.A ∨ vertex = T.B ∨ vertex = T.C) :
  ∃ alt, alt = altitude T vertex :=
by {
  -- We are expected to provide a proof here, but we use sorry for now.
  sorry
}

end shortest_distance_is_altitude_l583_583406


namespace compute_100m_plus_n_l583_583092

theorem compute_100m_plus_n :
  let m := Int.cbrt 61629875
  let n := Int.root 170859375 7
  m ^ 3 = 61629875 →
  n ^ 7 = 170859375 →
  100 * m + n = 39515 :=
by
  intros m n h_m h_n
  sorry

end compute_100m_plus_n_l583_583092


namespace range_of_a_l583_583377

noncomputable def condition_p (x a : ℝ) : Prop :=
  (x - 3 * a) * (x - a) < 0

noncomputable def condition_q (x : ℝ) : Prop :=
  (x^2 - 3 * x ≤ 0) ∧ (x^2 - x - 2 > 0)

noncomputable def not_q (x : ℝ) : Prop :=
  ¬condition_q x

theorem range_of_a (a : ℝ) : (0 < a ∧ a ≤ 2 / 3) ∨ a ≥ 3 :=
begin
  sorry
end

end range_of_a_l583_583377


namespace subsets_XY_count_l583_583560

theorem subsets_XY_count :
  let S := {1, 2, ..., 2001};
  ∃ (X Y : set ℕ), (X ≠ ∅ ∧ Y ≠ ∅ ∧ X ⊆ S ∧ Y ⊆ S ∧ 
                   |Y| = 1001 ∧ ∃ m ∈ Y, ∀ x ∈ X, x ≤ m ∧ ∀ y ∈ Y, y ≥ m) → 
  (number of such pairs (X, Y)) = 2^2000 := by
  sorry

end subsets_XY_count_l583_583560


namespace selection_plans_l583_583172

-- Definitions for the students
inductive Student
| A | B | C | D | E | F

open Student

-- Definitions for the subjects
inductive Subject
| Mathematics | Physics | Chemistry | Biology

open Subject

-- A function to count the number of valid selections such that A and B do not participate in Biology.
def countValidSelections : Nat :=
  let totalWays := Nat.factorial 6 / Nat.factorial 2 / Nat.factorial (6 - 4)
  let forbiddenWays := 2 * (Nat.factorial 5 / Nat.factorial 2 / Nat.factorial (5 - 3))
  totalWays - forbiddenWays

theorem selection_plans :
  countValidSelections = 240 :=
by
  sorry

end selection_plans_l583_583172


namespace shortest_rope_length_l583_583796

theorem shortest_rope_length (x : ℕ) (h : 4 * x + 6 * x = 5 * x + 100) : 4 * x = 80 :=
by {
  have : 10 * x = 5 * x + 100 := by linarith,
  have : 5 * x = 100 := by linarith,
  have x_val : x = 20 := by linarith,
  rw x_val,
  linarith,
}. sorry

end shortest_rope_length_l583_583796


namespace sequence_fifth_term_l583_583017

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 0 else 4 * sequence (n - 1) + 3

theorem sequence_fifth_term : sequence 5 = 255 :=
by
  sorry

end sequence_fifth_term_l583_583017


namespace find_sum_of_distinct_numbers_l583_583321

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l583_583321


namespace median_price_l583_583403

-- Definitions from conditions
def price1 : ℝ := 10
def price2 : ℝ := 12
def price3 : ℝ := 15

def sales1 : ℝ := 0.50
def sales2 : ℝ := 0.30
def sales3 : ℝ := 0.20

-- Statement of the problem
theorem median_price : (price1 * sales1 + price2 * sales2 + price3 * sales3) / 2 = 11 := by
  sorry

end median_price_l583_583403


namespace triangle_area_l583_583184

theorem triangle_area (a b C : ℝ) (h₁ : a = 45) (h₂ : b = 60) (h₃ : C = 37 * real.pi / 180) :
  1 / 2 * a * b * real.sin(C) ≈ 812.45 :=
by
  have area := 1 / 2 * a * b * real.sin(C)
  rw [h₁, h₂, h₃] at area
  sorry

end triangle_area_l583_583184


namespace eight_n_plus_nine_is_perfect_square_l583_583395

theorem eight_n_plus_nine_is_perfect_square 
  (n : ℕ) (N : ℤ) 
  (hN : N = 2 ^ (4 * n + 1) - 4 ^ n - 1)
  (hdiv : 9 ∣ N) :
  ∃ k : ℤ, 8 * N + 9 = k ^ 2 :=
by
  sorry

end eight_n_plus_nine_is_perfect_square_l583_583395


namespace probability_factor_24_l583_583032

theorem probability_factor_24 : 
  (∃ (k : ℚ), k = 1 / 3 ∧ 
  ∀ (n : ℕ), n ≤ 24 ∧ n > 0 → 
  (∃ (m : ℕ), 24 = m * n)) := sorry

end probability_factor_24_l583_583032


namespace product_of_roots_quadratic_l583_583163

noncomputable def product_of_roots (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  x1 * x2

theorem product_of_roots_quadratic :
  (product_of_roots 1 3 (-5)) = -5 :=
by
  sorry

end product_of_roots_quadratic_l583_583163


namespace value_of_5_S_3_l583_583127

def S (a b : ℕ) : ℕ := 4 * a + 6 * b + 1

theorem value_of_5_S_3 : S 5 3 = 39 := by
  sorry

end value_of_5_S_3_l583_583127


namespace cos_product_eq_one_over_2pow_l583_583349

theorem cos_product_eq_one_over_2pow (n : ℕ) (h : 0 < n) :
  (∏ k in Finset.range n, Real.cos ((k + 1 : ℕ) * Real.pi / (2 * n + 1))) = 1 / (2 ^ n) :=
sorry

end cos_product_eq_one_over_2pow_l583_583349


namespace nth_equation_pattern_specific_calculation_2023_sum_consecutive_odds_l583_583707

theorem nth_equation_pattern (n : ℕ) : (n+1)^2 - n^2 = 2 * n + 1 := by
  calc
    (n+1)^2 - n^2 = (n^2 + 2 * n + 1) - n^2 : by sorry
              ... = 2 * n + 1          : by sorry

theorem specific_calculation_2023 : 2023^2 - 2022^2 = 4045 := by
  calc
    2023^2 - 2022^2 = 2 * 2022 + 1 : by sorry
                 ... = 4045 : by sorry

theorem sum_consecutive_odds : (∑ k in finset.range 99, (k+2)^2 - (k+1)^2) = 9800 := by
  calc
    (∑ k in finset.range 99, (k+2)^2 - (k+1)^2) = 99^2 - 1^2 : by sorry
                                             ... = 9800 : by sorry


end nth_equation_pattern_specific_calculation_2023_sum_consecutive_odds_l583_583707


namespace greatest_number_to_miss_l583_583854

theorem greatest_number_to_miss (total_problems : ℕ) (required_percentage : ℚ) (passing_score : ℕ) 
    (h1 : total_problems = 40) (h2 : required_percentage = 75/100) : (total_problems * (1 - required_percentage) = passing_score) → 
    passing_score = 10 :=
by
  intros h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end greatest_number_to_miss_l583_583854


namespace power_function_value_at_4_l583_583951

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_value_at_4 :
  ∃ a : ℝ, power_function a 2 = (Real.sqrt 2) / 2 → power_function a 4 = 1 / 2 :=
by
  sorry

end power_function_value_at_4_l583_583951


namespace lattice_points_on_hyperbola_l583_583226

theorem lattice_points_on_hyperbola (N : ℕ) (h : N = (2^6 * 3^4)^2) :
    ∃ (lattice_points : ℕ), lattice_points = 198 ∧
    (∀ (x y : ℤ), x^2 - y^2 = N → (x,y) ∈ set_of_lattice_points) :=
by
  let N := (2^6 * 3^4)^2
  use 198
  sorry

end lattice_points_on_hyperbola_l583_583226


namespace min_value_frac_l583_583954
open BigOperators

def sequence (n : ℕ) : ℕ → ℤ
| 0     := 18
| (k+1) := sequence k + 3 * k

theorem min_value_frac (n : ℕ) (hn : n ≥ 1) : (∃ (k : ℕ) (hk : k ≥ 1), (sequence k) / k = 9) :=
  sorry

end min_value_frac_l583_583954


namespace angle_of_inclination_proof_l583_583128

noncomputable def angle_of_inclination (x y k : ℝ) : ℝ :=
  let m := -1 / (Real.sqrt 3)
  Real.pi - Real.arctan (Real.abs m)

theorem angle_of_inclination_proof (x y k : ℝ) :
  angle_of_inclination x y k = 5 * Real.pi / 6 := 
sorry

end angle_of_inclination_proof_l583_583128


namespace band_song_average_l583_583750

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end band_song_average_l583_583750


namespace decreasing_on_interval_max_min_values_l583_583211

def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem decreasing_on_interval (x1 x2 : ℝ) (h1 : 2 ≤ x1) (h2 : x1 ≤ 6) (h3 : 2 ≤ x2) (h4 : x2 ≤ 6) (h5 : x1 < x2) :
  f x1 > f x2 :=
by sorry

theorem max_min_values :
  (∀ x ∈ set.Icc (2:ℝ) (6:ℝ), f x ≤ 2) ∧ (∃ x ∈ set.Icc (2:ℝ) (6:ℝ), f x = 2) ∧
  (∀ x ∈ set.Icc (2:ℝ) (6:ℝ), f x ≥ 2/5) ∧ (∃ x ∈ set.Icc (2:ℝ) (6:ℝ), f x = 2/5) :=
by sorry

end decreasing_on_interval_max_min_values_l583_583211


namespace smallest_a_l583_583687

-- Define the conditions and the proof goal
theorem smallest_a (a b : ℝ) (h₁ : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (15 * (x : ℝ))) (h₂ : 0 ≤ a) (h₃ : 0 ≤ b) :
  a = 15 :=
sorry

end smallest_a_l583_583687


namespace range_of_a_l583_583305

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * x + a^2 + 3 * a - 3
def C : set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1 }
def p (a : ℝ) : Prop := ∃ x, f(x, a) < 0
def q (a : ℝ) : Prop := let M := (a^2 / 4, a) in let F := (1, 0) in dist M F > 2

theorem range_of_a : {a : ℝ | ¬ p a ∧ ¬ (p a ∧ q a)} = [-2, 1) :=
by
  sorry

end range_of_a_l583_583305


namespace fraction_pregnant_eq_one_eighth_l583_583817

def initial_population : ℕ := 300000
def immigration : ℕ := 50000
def emigration : ℕ := 30000
def final_population : ℕ := 370000

theorem fraction_pregnant_eq_one_eighth
  (n_initial : ℕ)
  (n_immigration : ℕ)
  (n_emigration : ℕ)
  (n_final : ℕ)
  (n_initial = initial_population)
  (n_immigration = immigration)
  (n_emigration = emigration)
  (n_final = final_population)
  : ∃ f : ℚ, f = 1 / 8 ∧
    let n_pop_after_immigration := n_initial + n_immigration,
        n_pop_after_emigration := n_pop_after_immigration - n_emigration,
        n_births := n_final - n_pop_after_emigration,
        n_births_expected := (5 / 4) * f * n_pop_after_emigration
    in n_births = n_births_expected :=
by
  sorry

end fraction_pregnant_eq_one_eighth_l583_583817


namespace bureaucrats_total_l583_583022

-- Define the parameters and conditions as stated in the problem
variables (a b c : ℕ)

-- Conditions stated in the problem
def condition_1 : Prop :=
  ∀ (i j : ℕ) (h1 : i ≠ j), 
    (10 * a * b = 10 * a * c ∧ 10 * b * c = 10 * a * b)

-- The main goal: proving the total number of bureaucrats
theorem bureaucrats_total (h1 : a = b) (h2 : b = c) (h3 : condition_1 a b c) : 
  3 * a = 120 :=
by sorry

end bureaucrats_total_l583_583022


namespace find_two_digit_numbers_l583_583416

def first_two_digit_number (x y : ℕ) : ℕ := 10 * x + y
def second_two_digit_number (x y : ℕ) : ℕ := 10 * (x + 5) + y

theorem find_two_digit_numbers :
  ∃ (x_2 y : ℕ), 
  (first_two_digit_number x_2 y = x_2^2 + x_2 * y + y^2) ∧ 
  (second_two_digit_number x_2 y = (x_2 + 5)^2 + (x_2 + 5) * y + y^2) ∧ 
  (second_two_digit_number x_2 y - first_two_digit_number x_2 y = 50) ∧ 
  (y = 1 ∨ y = 3) := 
sorry

end find_two_digit_numbers_l583_583416


namespace find_digits_l583_583337

/-- 
  Four different digits are on the cards, one of which is zero.
  Vojta forms the largest possible four-digit number, three-digit number, two-digit number,
  and single-digit number from the cards.
  Martin forms the smallest possible four-digit number, three-digit number, two-digit number,
  and single-digit number from the cards.
  Adam writes down the differences between Vojta's and Martin's numbers at each step and
  sums these differences to get 9090.
  
  Prove: the four digits on the cards are 0, 1, 2, and 9.
-/
theorem find_digits (a b c : ℕ) (h : a < b ∧ b < c ∧ c < 10) : 
  let d₄ := (999 * c + 90 * b - 990 * a),
      d₃ := (100 * c + 9 * b - 99 * a),
      d₂ := (10 * c + b - 10 * a),
      d₁ := (c - a)
  in d₄ + d₃ + d₂ + d₁ = 9090 → (a, b, c) = (0, 1, 2, 9) := 
by
  intros
  sorry

end find_digits_l583_583337


namespace cost_of_each_scoop_l583_583344

theorem cost_of_each_scoop (x : ℝ) 
  (pierre_scoops : ℝ := 3)
  (mom_scoops : ℝ := 4)
  (total_bill : ℝ := 14) 
  (h : 7 * x = total_bill) :
  x = 2 :=
by 
  sorry

end cost_of_each_scoop_l583_583344


namespace least_perimeter_of_triangle_l583_583805

theorem least_perimeter_of_triangle (x: ℕ) (integral_x : x ∈ ℕ) (h1 : 35 + x > 43.5) (h2 : 43.5 + x > 35) (h3 : 35 + 43.5 > x) : 
  35 + 43.5 + x >= 87.5 :=
by
  sorry

end least_perimeter_of_triangle_l583_583805


namespace elaine_rent_percentage_l583_583679

variable (E : ℝ)

def spent_last_year : ℝ := 0.20 * E
def earnings_this_year : ℝ := 1.25 * E
def rent_this_year : ℝ := 1.875 * spent_last_year

theorem elaine_rent_percentage :
  (rent_this_year / earnings_this_year) * 100 = 30 := by
  sorry

end elaine_rent_percentage_l583_583679


namespace stratified_sampling_l583_583654

theorem stratified_sampling :
  ∀ (first_year second_year third_year total_students sample_size : ℕ),
  first_year = 540 → 
  second_year = 440 → 
  third_year = 420 → 
  total_students = first_year + second_year + third_year →
  sample_size = 70 →
  let prop_1 := (540 / 1400) * 70,
      prop_2 := (440 / 1400) * 70,
      prop_3 := (420 / 1400) * 70 in
  (prop_1 = 27 ∧ prop_2 = 22 ∧ prop_3 = 21) := 
begin
  sorry
end

end stratified_sampling_l583_583654


namespace rationalized_denominator_l583_583766

theorem rationalized_denominator (a b : ℝ) (h₁ : a = 3) (h₂ : b = 2) : 
    (sqrt a - sqrt b) / sqrt a = 1 / (sqrt a * (sqrt a + sqrt b)) :=
by sorry

end rationalized_denominator_l583_583766


namespace problem_l583_583763

theorem problem (s t : ℕ) (hs : Nat.coprime s t) (h : s > 0) (h : t > 0) :
  ∑ n in Finset.range 99 \ λ n, n + 2, (n / (n^2 - 1) - 1 / n) = (↑s : ℚ) / ↑t →
  s + t = 127 :=
by
  sorry

end problem_l583_583763


namespace no_solution_pos_integers_l583_583892

theorem no_solution_pos_integers (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a + b + c + d - 3 ≠ a * b + c * d := 
by
  sorry

end no_solution_pos_integers_l583_583892


namespace total_distance_l583_583124

-- Defining the conditions
def darius_outward := 679
def detour := 120
def julia_outward := 998
def thomas_one_way := 1205
def miles_to_km := 1.60934

-- Total miles driven by Darius (round trip including detour)
def darius_total_miles := 2 * darius_outward + detour

-- Total miles driven by Julia (round trip including detour)
def julia_total_miles := 2 * julia_outward + detour

-- Total miles driven by Thomas (round trip)
def thomas_total_miles := 2 * thomas_one_way

-- Sum of all miles
def total_miles := darius_total_miles + julia_total_miles + thomas_total_miles

-- Convert miles to kilometers
def total_kilometers := total_miles * miles_to_km

-- Theorem to prove the total number of miles and kilometers
theorem total_distance (d_actual : darius_total_miles = 1478) (j_actual : julia_total_miles = 2116) (t_actual : thomas_total_miles = 2410) :
  total_miles = 6004 ∧ total_kilometers = 9665.73616 := by
  have h1 : darius_total_miles = 2 * darius_outward + detour := rfl
  have h2 : julia_total_miles = 2 * julia_outward + detour := rfl
  have h3 : thomas_total_miles = 2 * thomas_one_way := rfl
  have h4 : total_miles = darius_total_miles + julia_total_miles + thomas_total_miles := rfl
  have h5 : total_kilometers = total_miles * miles_to_km := rfl
  split
  . exact eq.trans d_actual h1.symm
  . exact eq.trans j_actual h2.symm
  . exact eq.trans t_actual h3.symm
  . sorry  -- needs the numerical checks

end total_distance_l583_583124


namespace susans_coins_worth_l583_583734

theorem susans_coins_worth :
  ∃ n d : ℕ, n + d = 40 ∧ (5 * n + 10 * d) = 230 ∧ (10 * n + 5 * d) = 370 :=
sorry

end susans_coins_worth_l583_583734


namespace sufficient_but_not_necessary_condition_l583_583324

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a > 1) : (1/a < 1) ∧ (¬ ∀ a : ℝ, (1/a < 1) → (a > 1)) :=
by
    -- Prove sufficiency
    have : 1/a < 1 := by sorry
    split
    -- Prove a > 1 implies 1/a < 1
    exact this
    -- Prove 1/a < 1 does not imply a > 1
    intro h'
    exact ⟨-1, by linarith⟩

end sufficient_but_not_necessary_condition_l583_583324


namespace range_of_a_in_ellipse_l583_583587

theorem range_of_a_in_ellipse (a : ℝ) (h1 : a > 1) 
    (h2 : ∀ x y, (x^2 / a^2 + y^2 = 1) → 
    (∃ A C : ℝ × ℝ, 
      A ≠ B ∧ C ≠ B ∧ 
      (∃ k : ℝ, k > 0 ∧ 
       (∃ m : ℝ, m > 0 ∧ 
        (∀ x1 y1 : ℝ, (x^2 / a^2 + y^2 = 1) → 
          (right_angle_isosceles_triangle (0, 1) (x1, y1) (0, 1 + k))))))) : 
    1 < a ∧ a ≤ real.sqrt 3 := 
sorry

end range_of_a_in_ellipse_l583_583587


namespace length_of_chord_l583_583815

theorem length_of_chord 
  (r : ℝ) (d : ℝ) (CD : ℝ) 
  (h1 : r = 3) 
  (h2 : d = 2) 
  (h3 : ∀ O E D : ℝ, (O = r) ∧ (E = d) ∧ (CD = 2 * (sqrt (O^2 - E^2)))) 
  : CD = 2 * sqrt 5 := by
  sorry

end length_of_chord_l583_583815


namespace unpainted_face_area_l583_583074

noncomputable def area_of_unpainted_face (r h : ℝ) (angle : ℝ) : ℝ :=
  let un_center_dist := r * Real.sin (angle / 2)
  let ou := h / 2
  let on := Real.sqrt (ou ^ 2 + un_center_dist ^ 2)
  let sector_area := (angle / 360) * π * r ^ 2
  let triangle_area := 0.5 * r ^ 2 * Real.sin angle
  let unpainted_area := (on / un_center_dist) * (sector_area + triangle_area)
  unpainted_area

theorem unpainted_face_area :
  area_of_unpainted_face 8 10 150 = 32 * π + 18 * Real.sqrt 3 :=
begin
  sorry
end

end unpainted_face_area_l583_583074


namespace limit_sum_l583_583310

noncomputable def a_n (n : ℕ) : ℕ :=
  (choose n 2) * 3 ^ (n - 2)

theorem limit_sum : 
  (filterlim (λ n : ℕ, (finset.range n).sum (λ k, if k ≥ 2 then (3^k) / ((a_n k).toReal) else 0)) 
  filter.at_top (nhds 18)) :=
begin
  sorry
end

end limit_sum_l583_583310


namespace evaluate_abs_expression_l583_583865

theorem evaluate_abs_expression : |2 * Real.pi - |2 * Real.pi - 9|| = 4 * Real.pi - 9 := by
  sorry

end evaluate_abs_expression_l583_583865


namespace find_principal_amount_l583_583039

-- Define the conditions
variables (R T SI : ℝ)
def simple_interest (P : ℝ) : ℝ := (P * R * T) / 100

-- Given values
noncomputable def R_val : ℝ := 12
noncomputable def T_val : ℝ := 10
noncomputable def SI_val : ℝ := 1500

-- The theorem stating the principal amount calculation
theorem find_principal_amount : ∃ P : ℝ, simple_interest R_val T_val P = SI_val :=
by
  let P := 1250
  use P
  have h : simple_interest 12 10 P = 1500 := by sorry
  exact h

end find_principal_amount_l583_583039


namespace det_A4_l583_583632

variable {A : Matrix ℝ ℝ}

theorem det_A4 (h : det A = 3) : det (A^4) = 81 :=
sorry

end det_A4_l583_583632


namespace segment_order_l583_583831

def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def order_segments (angles_ABC angles_XYZ angles_ZWX : ℝ → ℝ → ℝ) : Prop :=
  let A := angles_ABC 55 60
  let B := angles_XYZ 95 70
  ∀ (XY YZ ZX WX WZ: ℝ), 
    YZ < ZX ∧ ZX < XY ∧ ZX < WZ ∧ WZ < WX

theorem segment_order:
  ∀ (A B C X Y Z W : Type)
  (XYZ_ang ZWX_ang : ℝ), 
  angle_sum_triangle 55 60 65 →
  angle_sum_triangle 95 70 15 →
  order_segments (angles_ABC) (angles_XYZ) (angles_ZWX)
:= sorry

end segment_order_l583_583831


namespace hexagon_diagonals_l583_583782

theorem hexagon_diagonals : 
  let n : ℕ := 6 in
  n * (n - 3) / 2 = 9 :=
by
  let n : ℕ := 6
  calc
    n * (n - 3) / 2 = 6 * (6 - 3) / 2 : by rfl
    ... = 6 * 3 / 2         : by rfl
    ... = 18 / 2            : by rfl
    ... = 9                 : by rfl

end hexagon_diagonals_l583_583782


namespace integral_abs_x_minus_1_eq_half_l583_583133

theorem integral_abs_x_minus_1_eq_half : ∫ x in 0..1, |x - 1| = 1 / 2 := 
  sorry

end integral_abs_x_minus_1_eq_half_l583_583133


namespace alcohol_percentage_l583_583726

variable {P : ℝ} -- P is the percentage of alcohol in solution y

theorem alcohol_percentage (hx : 10 / 100 * 250 = 25) (hy_volume : 750) (hx_volume : 250) (desired_percent : 25) (total_volume : hx_volume + hy_volume = 1000) (desired_alcohol_volume : (desired_percent / 100) * 1000 = 250) 
: (P / 100) * hy_volume = 225 → P = 30 := sorry

end alcohol_percentage_l583_583726


namespace cubic_expression_value_l583_583592

theorem cubic_expression_value (m : ℝ) (h : m^2 + 3 * m - 2023 = 0) :
  m^3 + 2 * m^2 - 2026 * m - 2023 = -4046 :=
by
  sorry

end cubic_expression_value_l583_583592


namespace handshake_problem_l583_583491

noncomputable def total_handshakes (n : ℕ) : ℕ :=
  let men_handshakes := n * (2 * n - 1) / 2
  let franklin_handshakes := n
  men_handshakes + franklin_handshakes

theorem handshake_problem : total_handshakes 15 = 225 := 
  by 
    unfold total_handshakes
    have h_men := 15 * (2 * 15 - 1) / 2  -- Evaluate men handshakes part
    have h_franklin := 15  -- Handshakes by Franklin
    have total := h_men + h_franklin  -- Total handshakes
    unfold Nat.mul Nat.sub Nat.add Nat.div at h_men
    -- Applying numerical simplification
    have h_eval : 15 * 29 / 2 = 210 := sorry
    rw h_eval at total
    -- Adding the handshakes involving Franklin
    have h_total: 210 + 15 = 225 := by norm_num
    rw [h_total]
    exact congr_arg Nat.succ rfl

end handshake_problem_l583_583491


namespace lattice_point_count_l583_583051

noncomputable def countLatticePoints (N : ℤ) : ℤ :=
  2 * N * (N + 1) + 1

theorem lattice_point_count (N : ℤ) (hN : 71 * N > 0) :
    ∃ P, P = countLatticePoints N := sorry

end lattice_point_count_l583_583051


namespace exchange_rate_l583_583459

theorem exchange_rate (a b : ℕ) (h : 5000 = 60 * a) : b = 75 * a → b = 6250 := by
  sorry

end exchange_rate_l583_583459


namespace det_A4_l583_583631

variable {A : Matrix ℝ ℝ}

theorem det_A4 (h : det A = 3) : det (A^4) = 81 :=
sorry

end det_A4_l583_583631


namespace a_2012_value_l583_583694

noncomputable def a_seq : ℕ → ℝ
| 0     := 2
| (n+1) := a_seq n / (1 + a_seq n)

theorem a_2012_value : a_seq 2012 = 2 / (2 * 2012 + 1) := by
  sorry

end a_2012_value_l583_583694


namespace values_of_x_plus_y_l583_583640

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end values_of_x_plus_y_l583_583640


namespace triangle_problem_l583_583651

theorem triangle_problem
  (A B C : Real)
  (R : Real)
  (h1 : cos (2 * A) - 3 * cos (B + C) - 1 = 0)
  (h2 : R = 1) :
  A = π / 3 ∧ (let a := 2 * R * sin A in 
               let b := some b in 
               let c := some c in 
               a = sqrt 3 ∧ S := (1/2) * b * c * sin A ∧ S ≤ (3 * sqrt 3) / 4) :=
sorry

end triangle_problem_l583_583651


namespace determinant_of_cross_product_matrix_l583_583304

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def determinant_equality (u v w : V) (E : ℝ) : Prop :=
let E' := det ![(u × v), (v × w), (w × u)] in
E = det ![u, v, w] ∧ E' = E ^ 2

theorem determinant_of_cross_product_matrix (u v w : V) (E : ℝ) :
  E = det ![u, v, w] → det ![(u × v), (v × w), (w × u)] = E ^ 2 :=
sorry

end determinant_of_cross_product_matrix_l583_583304


namespace inequality_solution_l583_583371

noncomputable def solve_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo (-3 : ℝ) 3

theorem inequality_solution (x : ℝ) (h : x ≠ -3) :
  (x^2 - 9) / (x + 3) < 0 ↔ solve_inequality x :=
by
  sorry

end inequality_solution_l583_583371


namespace log10_ge_one_iff_x_ge_ten_x_ge_one_necessary_not_sufficient_for_log10_ge_one_l583_583036

variable (x : ℝ)

theorem log10_ge_one_iff_x_ge_ten :
  log 10 x ≥ 1 ↔ x ≥ 10 := by sorry

theorem x_ge_one_necessary_not_sufficient_for_log10_ge_one :
  (x : ℝ) → (log 10 x ≥ 1 → x ≥ 1)
  ∧ (x : ℝ) → (x ≥ 1) → ¬(log 10 x ≥ 1) :=
by sorry

end log10_ge_one_iff_x_ge_ten_x_ge_one_necessary_not_sufficient_for_log10_ge_one_l583_583036


namespace eggplant_weight_l583_583393

-- Define the conditions
def number_of_cucumbers : ℕ := 25
def weight_per_cucumber_basket : ℕ := 30
def number_of_eggplants : ℕ := 32
def total_weight : ℕ := 1870

-- Define the statement to be proved
theorem eggplant_weight :
  (total_weight - (number_of_cucumbers * weight_per_cucumber_basket)) / number_of_eggplants =
  (1870 - (25 * 30)) / 32 := 
by sorry

end eggplant_weight_l583_583393


namespace greatest_x_value_l583_583898

theorem greatest_x_value :
  ∃ x : ℝ, (x ≠ 2 ∧ (x^2 - 5 * x - 14) / (x - 2) = 4 / (x + 4)) ∧ x = -2 ∧ 
           ∀ y, (y ≠ 2 ∧ (y^2 - 5 * y - 14) / (y - 2) = 4 / (y + 4)) → y ≤ x :=
by
  sorry

end greatest_x_value_l583_583898


namespace multiple_parenthesis_possibilities_l583_583688

def exp (a b : ℕ) : ℕ := a ^ b

theorem multiple_parenthesis_possibilities :
  ∃ p1 p2, 
    p1 ≠ p2 ∧
    (exp (exp (exp (exp 7 7) 7) 7) 7) = (exp 7 (exp 7 (exp 7 (exp 7 7)))) ∧
    (exp 7 (exp 7 (exp 7 (exp 7 7)))) = (exp 7 (exp 7 (exp 7 (exp 7 7)))) := 
sorry

end multiple_parenthesis_possibilities_l583_583688


namespace smallest_integer_condition_l583_583553

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l583_583553


namespace rational_combination_zero_eqn_l583_583973

theorem rational_combination_zero_eqn (a b c : ℚ) (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end rational_combination_zero_eqn_l583_583973


namespace volleyball_team_points_l583_583263

theorem volleyball_team_points (lizzie_points nathalie_points aimee_points teammate_points total_points : ℕ)
  (h1 : lizzie_points = 4)
  (h2 : nathalie_points = lizzie_points + 3)
  (h3 : aimee_points = 2 * (lizzie_points + nathalie_points))
  (h4 : total_points = 50)
  (h5 : total_points = lizzie_points + nathalie_points + aimee_points + teammate_points) :
  teammate_points = 17 :=
begin
  sorry
end

end volleyball_team_points_l583_583263


namespace max_value_expression_l583_583398

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end max_value_expression_l583_583398


namespace count_approximately_equal_sums_l583_583967

def approximately_equal (a b : ℤ) : Prop := abs (a - b) ≤ 1

def sum_of_terms (n K : ℤ) (terms : list ℤ) : Prop :=
  terms.sum = 2004 ∧
  ∀ t ∈ terms, abs (t - n) ≤ 1 ∧ t > 0 ∧ terms.length = K

theorem count_approximately_equal_sums :
  ∃ (result : ℤ), result = 2004 ∧
  ∀ (n K : ℤ) (terms : list ℤ),
    sum_of_terms n K terms → (1 ≤ K ∧ K ≤ 2004) :=
sorry

end count_approximately_equal_sums_l583_583967


namespace minimum_distance_l583_583248

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line x - y - 2 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the distance from a point (x1, y1) to a line Ax + By + C = 0
noncomputable def distance_to_line (x1 y1 A B C : ℝ) : ℝ := 
  abs (A * x1 + B * y1 + C) / Real.sqrt (A^2 + B^2)

-- The target statement to prove
theorem minimum_distance : 
  ∀ x y, (x = 1 ∧ y = f x ∧ y = 1) → 
  distance_to_line x y 1 (-1) (-2) = Real.sqrt 2 :=
begin
  intros x y h,
  rcases h with ⟨hx, hy1, hy2⟩,
  subst hx,
  subst hy2,
  unfold distance_to_line,
  norm_num,
end

end minimum_distance_l583_583248


namespace cube_root_of_nine_irrational_l583_583085

theorem cube_root_of_nine_irrational : ¬ ∃ (r : ℚ), r^3 = 9 :=
by sorry

end cube_root_of_nine_irrational_l583_583085


namespace sum_of_exponents_l583_583981

-- Definition of Like Terms
def like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 2

-- Theorem statement
theorem sum_of_exponents (m n : ℕ) (h : like_terms m n) : m + n = 5 :=
sorry

end sum_of_exponents_l583_583981


namespace perpendicular_vectors_l583_583914

def vec (x y : ℝ) : ℝ × ℝ := (x, y)

def dot (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def λ : ℝ := -6 / 13

theorem perpendicular_vectors:
  let a := vec 4 (-1)
  let b := vec (-2) 0
  let v1 := vec (4 * λ + 2) (-λ)
  let v2 := vec 6 (-2)
  dot v1 v2 = 0 := 
sorry

end perpendicular_vectors_l583_583914


namespace candidate_lost_votes_rounding_l583_583813

theorem candidate_lost_votes_rounding (votes_cast : ℝ) (candidate_percent : ℝ) (rival_percent : ℝ) 
  (h_votes : votes_cast = 7499.999999999999)
  (h_candidate_percent : candidate_percent = 0.35)
  (h_rival_percent : rival_percent = 0.65) :
  let votes_cast_rounded := (votes_cast + 0.5).floor
  let candidate_votes := candidate_percent * votes_cast_rounded
  let rival_votes := rival_percent * votes_cast_rounded
  rival_votes - candidate_votes = 2250 :=
by
  let votes_cast_rounded := (votes_cast + 0.5).floor
  let candidate_votes := candidate_percent * votes_cast_rounded
  let rival_votes := rival_percent * votes_cast_rounded
  sorry

end candidate_lost_votes_rounding_l583_583813


namespace rationalize_frac_l583_583723

-- Define the problem
def frac : ℝ := 1 / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7)

-- Define the desired form
def desired_form : ℝ := (4 * Real.sqrt 2 + 3 * Real.sqrt 3 - Real.sqrt 7 - Real.sqrt 42) / 10

-- Define A, B, C, D, E, F corresponding to the terms in the desired form
def A : ℤ := 4
def B : ℤ := 3
def C : ℤ := -1
def D : ℤ := -1
def E : ℤ := 42
def F : ℤ := 10

-- State the sum of A, B, C, D, E, F as a constant
def sum_ABCDEF : ℤ := A + B + C + D + E + F

-- Lean theorem statement
theorem rationalize_frac :
    frac = desired_form ∧ sum_ABCDEF = 57 :=
by
  sorry -- Proof to be filled in

end rationalize_frac_l583_583723


namespace value_of_x_squared_plus_y_squared_l583_583643

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x^2 = 8 * x + y) (h2 : y^2 = x + 8 * y) (h3 : x ≠ y) : 
  x^2 + y^2 = 63 := sorry

end value_of_x_squared_plus_y_squared_l583_583643


namespace even_three_digit_number_count_is_48_l583_583567

-- Definitions from conditions
def is_digit (d : ℕ) : Prop := d ∈ {0, 1, 2, 3, 4, 5}
def is_even_digit (d : ℕ) : Prop := d ∈ {0, 2, 4}

-- Function to count the number of even three-digit numbers without repeating digits
def even_three_digit_numbers_count : ℕ :=
  let even_digits := [0, 2, 4]
  let remaining_digits (d : ℕ) := [0, 1, 2, 3, 4, 5].filter (≠ d)
  even_digits.sum (λ u, remaining_digits u |>.filter (≠ 0) |>.length * (remaining_digits u).length)

-- Statement to prove
theorem even_three_digit_number_count_is_48 : even_three_digit_numbers_count = 48 := by
  sorry

end even_three_digit_number_count_is_48_l583_583567


namespace angle_A_is_not_angle_BCD_l583_583289

variable (A B C D F : Point)
variable (ω : Circle)
variable (ABC_triangle_is_isosceles : AB = BC)
variable (circle_ω_passes_through_B : ω.passes_through B)
variable (D_on_AC : D ∈ AC)
variable (D_is_intersection : ω ∩ AC = {D})
variable (F_on_tangent : F ∈ tangent_line_at D ω)
variable (F_on_AB : F ∈ AB)

theorem angle_A_is_not_angle_BCD
    (angle_A : angle A B C = 60)
    (angle_B : angle B = 60)
    (angle_BDA : angle B D A = 90)
    (angleB_DF : angle B D F = 90)
    (angle_DFB : angle D F B = 90) :
    ¬ (angle A = angle B C D) := 
sorry

end angle_A_is_not_angle_BCD_l583_583289


namespace triangle_similarity_l583_583834

variables {A B C D P E F G : Type}
variables [EuclideanGeometry A B C D]
variables (cyclic_ABCD : CyclicQuadrilateral A B C D)
variables (P_interior : InteriorPoint P A B C D)
variables (angle_condition : ∠BPC = ∠BAP + ∠PDC)
variables (foot_E : PerpendicularFoot P E A B)
variables (foot_F : PerpendicularFoot P F A D)
variables (foot_G : PerpendicularFoot P G D C)

theorem triangle_similarity (cyclic_ABCD : CyclicQuadrilateral A B C D) (P_interior : InteriorPoint P A B C D)
    (angle_condition : ∠BPC = ∠BAP + ∠PDC) (foot_E : PerpendicularFoot P E A B)
    (foot_F : PerpendicularFoot P F A D) (foot_G : PerpendicularFoot P G D C) :
    Similar (Triangle F E G) (Triangle P B C) :=
sorry

end triangle_similarity_l583_583834


namespace incorrect_average_initially_calculated_l583_583738

theorem incorrect_average_initially_calculated :
  ∀ (S' S : ℕ) (n : ℕ) (incorrect_correct_difference : ℕ),
  n = 10 →
  incorrect_correct_difference = 30 →
  S = 200 →
  S' = S - incorrect_correct_difference →
  (S' / n) = 17 :=
by
  intros S' S n incorrect_correct_difference h_n h_diff h_S h_S' 
  sorry

end incorrect_average_initially_calculated_l583_583738


namespace least_integer_l583_583534

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l583_583534


namespace type_B_completion_time_l583_583844

theorem type_B_completion_time :
  ∃ (T_B : ℝ), (1 / T_B ≈ 1 / 7) ∧ 
  2 * (1 / 5) + 3 * (1 / T_B) = 1 / 1.2068965517241381 :=
begin
  sorry,
end

end type_B_completion_time_l583_583844


namespace smallest_integer_condition_l583_583554

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l583_583554


namespace eccentricity_range_l583_583217

variables {a b x1 x2 y1 y2 : ℝ}

-- Definitions and assumptions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

noncomputable def slopes_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  slope x1 y1 x2 y2 * slope x1 (-y1) x2 (-y2)

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + b^2 / a^2)

-- The proof statement
theorem eccentricity_range (a b : ℝ) (P : ℝ × ℝ) (h1 : a > 0) (h2 : b > 0)
  (hP : hyperbola a b P.1 P.2)
  (hA : hyperbola a b x1 y1) (hPA : x1 ≠ P.1 ∧ y1 ≠ P.2)
  (hAB_slope : slope x1 y1 (-x1) (-y1) = y1 / x1)
  (h_k1k2 : slopes_product x1 y1 P.1 P.2 > slope x1 y1 (-x1) (-y1)) :
  eccentricity a b ≥ sqrt 2 :=
sorry

end eccentricity_range_l583_583217


namespace calculate_expected_value_of_S_l583_583732

-- Define the problem context
variables (boys girls : ℕ)
variable (boy_girl_pair_at_start : Bool)

-- Define the expected value function
def expected_S (boys girls : ℕ) (boy_girl_pair_at_start : Bool) : ℕ :=
  if boy_girl_pair_at_start then 10 else sorry  -- we only consider the given scenario

-- The theorem to prove
theorem calculate_expected_value_of_S :
  expected_S 5 15 true = 10 :=
by
  -- proof needs to be filled in
  sorry

end calculate_expected_value_of_S_l583_583732


namespace non_neg_int_count_l583_583390

def is_non_neg_int (n : ℝ) : Prop := n ≥ 0 ∧ ∃ m : ℤ, n = m

theorem non_neg_int_count : (|-3, -2.5, 2.25, 0, 0.1, 3.2, 10, -4.25|).count is_non_neg_int = 2 :=
sorry

end non_neg_int_count_l583_583390


namespace find_altitude_to_hypotenuse_l583_583838

-- define the conditions
def area : ℝ := 540
def hypotenuse : ℝ := 36
def altitude : ℝ := 30

-- define the problem statement
theorem find_altitude_to_hypotenuse (A : ℝ) (c : ℝ) (h : ℝ) 
  (h_area : A = 540) (h_hypotenuse : c = 36) : h = 30 :=
by
  -- skipping the proof
  sorry

end find_altitude_to_hypotenuse_l583_583838


namespace find_least_positive_integer_l583_583548

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l583_583548


namespace probability_of_consecutive_numbers_l583_583426

noncomputable def binomialCoefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_consecutive_numbers :
  let total_sets := binomialCoefficient 90 5
  let num_adjacent_sets := 9122966
  let probability := num_adjacent_sets.toRat / total_sets.toRat
  probability ≈ 0.2075 :=
  by
    let total_sets := binomialCoefficient 90 5
    have : total_sets = 43949268 := by sorry
    let num_adjacent_sets := 9122966
    let probability := num_adjacent_sets.toRat / total_sets.toRat
    have : probability ≈ 0.2075 := by sorry
    exact this

end probability_of_consecutive_numbers_l583_583426


namespace find_sum_of_distinct_numbers_l583_583320

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l583_583320


namespace cost_of_3000_pencils_is_760_l583_583460

-- Define given conditions
def cost_of_box (num_pencils : ℕ) (total_cost : ℝ) : ℝ :=
  total_cost / num_pencils

def discount_price (base_price : ℝ) (discount : ℝ) : ℝ :=
  base_price * (1 - discount)

-- Prove that the total cost is equal to 760 dollars given the conditions
theorem cost_of_3000_pencils_is_760 :
  let base_price := cost_of_box 150 40
  let discounted_price := discount_price base_price 0.05
  let total_pencils := 3000
  in total_pencils * discounted_price = 760 := 
  sorry

end cost_of_3000_pencils_is_760_l583_583460


namespace percentage_chromium_first_alloy_l583_583278

theorem percentage_chromium_first_alloy
  (x : ℝ) (h : (x / 100) * 15 + (8 / 100) * 35 = (9.2 / 100) * 50) : x = 12 :=
sorry

end percentage_chromium_first_alloy_l583_583278


namespace least_integer_l583_583532

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l583_583532


namespace eunji_total_money_l583_583884

-- Define initial total money X, and money spent on toy and book
variables (X : ℕ)

-- Conditions as definitions
def toy_cost := (1 / 4) * X
def remaining_after_toy := (3 / 4) * X
def book_cost := (1 / 3) * remaining_after_toy
def final_remaining := 1600

-- Statement to prove
theorem eunji_total_money : X - (toy_cost + book_cost) = final_remaining -> X = 3200 := 
begin
  sorry
end

end eunji_total_money_l583_583884


namespace number_of_initials_with_vowels_l583_583966

def letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
def vowels := ['A', 'E', 'I']
def consonants := ['B', 'C', 'D', 'F', 'G', 'H', 'J']

theorem number_of_initials_with_vowels : 
    (∃ (s : finset (list char)), (∀ x ∈ s, x ∈ list.replicate 3 'A' ∩ list.replicate 3 'B' ∩ list.replicate 3 'C' ∩ list.replicate 3 'D' ∩ list.replicate 3 'E' ∩ list.replicate 3 'F' ∩ list.replicate 3 'G' ∩ list.replicate 3 'H' ∩ list.replicate 3 'I' ∩ list.replicate 3 'J') ∧ (∃ y ∈ s, y ∈ vowels)) → 
    ∃ (t : finset (list char)), (∀ x ∈ t, x ∈ list.replicate 3 'B' ∩ list.replicate 3 'C' ∩ list.replicate 3 'D' ∩ list.replicate 3 'F' ∩ list.replicate 3 'G' ∩ list.replicate 3 'H' ∩ list.replicate 3 'J') ∨ t.card = 657 := 
sorry

end number_of_initials_with_vowels_l583_583966


namespace vertex_of_parabola_l583_583597

theorem vertex_of_parabola 
  (a b c : ℝ) 
  (h1 : a * 2^2 + b * 2 + c = 5)
  (h2 : -b / (2 * a) = 2) : 
  (2, 4 * a + 2 * b + c) = (2, 5) :=
by
  sorry

end vertex_of_parabola_l583_583597


namespace minimum_blue_cells_to_reach_end_l583_583299

theorem minimum_blue_cells_to_reach_end (n : ℕ) (h : n > 2) : 
  ∃ k, (∀ coloring : Fin n → bool, 
    (initial_position: Fin n := 0) 
        → (steps : list (Fin n)) :=
        -- Use the movement rules to define the steps        
        sorry)
    →
    (steps.last == some (Fin.last n)) 
    →
    (count (finset.filter (λ i, coloring i) (finset.range n)) = k) 
    ∧
     k = (n + 1 + 1)/2 :=
sorry

end minimum_blue_cells_to_reach_end_l583_583299


namespace fred_remaining_cards_l583_583912

section
variable (initial_cards : ℕ) (fraction_keith : ℚ) (fraction_linda : ℚ)
variable (cards_after_keith : ℕ) (cards_after_linda : ℕ)

def condition_1 : initial_cards = 40 := sorry
def condition_2 : fraction_keith = 1/4 := sorry
def condition_3 : fraction_linda = 1/3 := sorry
def condition_4 : cards_after_keith = initial_cards - initial_cards * fraction_keith := sorry
def condition_5 : cards_after_linda = cards_after_keith - cards_after_keith * fraction_linda := sorry

theorem fred_remaining_cards (initial_cards fraction_keith fraction_linda cards_after_keith cards_after_linda : ℕ) :
  condition_1 initial_cards →
  condition_2 fraction_keith →
  condition_3 fraction_linda →
  condition_4 initial_cards fraction_keith cards_after_keith →
  condition_5 cards_after_keith fraction_linda cards_after_linda →
  cards_after_linda = 20 := sorry
end

end fred_remaining_cards_l583_583912


namespace arithmetic_mean_of_heads_gt_1988_l583_583655

variable {n : ℕ}
variable {a : ℕ → ℝ}

def is_dragon (k l : ℕ) : Prop :=
  ∃ l, l ≥ 0 ∧ (∑ i in finset.range (l + 1), a (k + i)) / (l + 1) > 1988

def is_head_of_dragon (k : ℕ) : Prop :=
  ∃ l, is_dragon k l

theorem arithmetic_mean_of_heads_gt_1988 (h : ∃ k, is_head_of_dragon k) :
  ∑ i in (finset.range n).filter is_head_of_dragon, a i / 
    (((finset.range n).filter is_head_of_dragon).card : ℝ) > 1988 :=
sorry

end arithmetic_mean_of_heads_gt_1988_l583_583655


namespace year_2023_ad_is_written_as_positive_2023_l583_583862

theorem year_2023_ad_is_written_as_positive_2023 :
  (∀ (year : Int), year = -500 → year = -500) → -- This represents the given condition that year 500 BC is -500
  (∀ (year : Int), year > 0) → -- This represents the condition that AD years are postive
  2023 = 2023 := -- The problem conclusion

by
  intros
  trivial -- The solution is quite trivial due to the conditions.

end year_2023_ad_is_written_as_positive_2023_l583_583862


namespace f_increasing_intervals_triangle_area_proof_l583_583916

noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (Real.pi + 2 * x)

theorem f_increasing_intervals :
  ∀ (k : ℤ), increasing_on f  (Icc (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi)) :=
sorry

variables {a b c : ℝ}
variables {A B C : ℝ}

def triangle_area (a b c : ℝ) (C : ℝ) : ℝ := (1 / 2) * a * b * Real.sin C

theorem triangle_area_proof (C : ℝ) (h1 : f C = 1) (h2 : c = Real.sqrt 3) (h3 : a + b = 2 * Real.sqrt 3) : 
  triangle_area a b c C = (3 * Real.sqrt 3) / 4 :=
sorry

end f_increasing_intervals_triangle_area_proof_l583_583916


namespace sqrt_defined_range_l583_583013

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end sqrt_defined_range_l583_583013


namespace smallest_sum_of_sequence_l583_583007

theorem smallest_sum_of_sequence :
  ∃ A B C D : ℕ, A > 0 ∧ B > 0 ∧ C > 0 ∧
  (C - B = B - A) ∧ (C = 7 * B / 4) ∧ (D = 49 * B / 16) ∧
  (A + B + C + D = 97) :=
begin
  sorry
end

end smallest_sum_of_sequence_l583_583007


namespace least_positive_integer_l583_583545

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l583_583545


namespace prob_at_least_two_girls_l583_583465

def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem prob_at_least_two_girls :
  let nB := 5  -- number of boys
  let nG := 3  -- number of girls
  let n := nB + nG  -- total students
  let k := 3  -- number of students selected
  let X2 := (comb nG 2 * comb nB 1) / comb n 3  -- P(X = 2)
  let X3 := (comb nG 3 * comb nB 0) / comb n 3  -- P(X = 3)
  P(X ≥ 2) = X2 + X3 := by {
  sorry
}

end prob_at_least_two_girls_l583_583465


namespace exists_positive_integers_x_y_l583_583290

theorem exists_positive_integers_x_y (x y : ℕ) : 0 < x ∧ 0 < y ∧ x^2 = y^2 + 2023 :=
  sorry

end exists_positive_integers_x_y_l583_583290


namespace part_a_part_b_l583_583449

variables {A B C D K P Q R : Type}
variables [Incircle A B C D] [TangentsIntersectAt K B D] [KOnLineAC K A C]
variables [LineParallelToKBIntersectsAt PQ KB BA BD BC]

-- Part (a)
theorem part_a (hb : is_tangent K B) (hd : is_tangent K D) (arta : is_on_line K A C) : 
  AB * CD = BC * AD := 
sorry

-- Part (b)
theorem part_b (hpar : is_parallel PQ KB) (hinter1 : intersects PQ BA at P) (hinter2 : intersects PQ BD at Q) (hinter3 : intersects PQ BC at R) : 
  PQ = QR := 
sorry

end part_a_part_b_l583_583449


namespace bill_spots_l583_583101

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end bill_spots_l583_583101


namespace tan_45_eq_one_l583_583505

theorem tan_45_eq_one 
  (h1 : ∀ A B : ℝ, tan (A - B) = (tan A - tan B) / (1 + tan A * tan B)) 
  (h2 : ∀ x : ℝ, tan x = if x = 60 then real.sqrt 3 else if x = 15 then 2 - real.sqrt 3 else 0) : 
  tan 45 = 1 := 
sorry

end tan_45_eq_one_l583_583505


namespace right_triangle_locus_area_l583_583578

theorem right_triangle_locus_area (DE DF : ℝ) (h1 : DE = 45) (h2 : DF = 60) (angleE : ∠E = 90) : ∃ (q r s : ℕ), 
    let area := q * real.pi - r * real.sqrt s in 
    s ≠ 0 ∧ (∀ p, prime p → ¬(p * p ∣ s)) ∧ q = 703 ∧ r = 338 ∧ s = 1 ∧ q + r + s = 1042 :=
by
  sorry

end right_triangle_locus_area_l583_583578


namespace inequality_example_l583_583730

theorem inequality_example (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) (h4 : b < 0) : a + b < b + c := 
by sorry

end inequality_example_l583_583730


namespace paul_score_higher_by_26_l583_583485

variable {R : Type} [LinearOrderedField R]

variables (A1 A2 A3 P1 P2 P3 : R)

-- hypotheses
variable (h1 : A1 = P1 + 10)
variable (h2 : A2 = P2 + 4)
variable (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)

-- goal
theorem paul_score_higher_by_26 : P3 - A3 = 26 := by
  sorry

end paul_score_higher_by_26_l583_583485


namespace train_crossing_time_l583_583969

theorem train_crossing_time
  (length_train : ℝ)
  (length_platform : ℝ)
  (speed_kmph : ℝ) :
  length_train = 250 →
  length_platform = 200 →
  speed_kmph = 90 →
  (length_train + length_platform) / (speed_kmph * (1000 / 3600)) = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end train_crossing_time_l583_583969


namespace max_f_prime_range_of_a_l583_583612

-- Definitions for the given problem
noncomputable def f (a x : ℝ) := a * x^2 - Real.exp x
noncomputable def f' (a x : ℝ) := 2 * a * x - Real.exp x
noncomputable def h (a x : ℝ) := f a x + x * (2 - 2 * Real.log 2)
noncomputable def h' (a x : ℝ) := 2 * a * x + (2 - 2 * Real.log 2) - Real.exp x

-- Statement 1: Maximum value of f'(x) is 0 given f'(1) = 0
theorem max_f_prime (a : ℝ) (hf'_1 : f' a 1 = 0) : ∀ x, f' a x ≤ 0 := 
begin
  sorry
end

-- Statement 2: Range of a under the condition on h(x)
theorem range_of_a (a : ℝ) (hdec : ∀ x1 x2, 0 ≤ x1 → x1 < x2 → h a x2 < h a x1) : a ≤ 1 := 
begin
  sorry
end

end max_f_prime_range_of_a_l583_583612


namespace find_num_of_boys_l583_583414

-- Define the constants for number of girls and total number of kids
def num_of_girls : ℕ := 3
def total_kids : ℕ := 9

-- The theorem stating the number of boys based on the given conditions
theorem find_num_of_boys (g t : ℕ) (h1 : g = num_of_girls) (h2 : t = total_kids) :
  t - g = 6 :=
by
  sorry

end find_num_of_boys_l583_583414


namespace find_third_root_l583_583509

noncomputable def P (a b x : ℝ) : ℝ := a * x^3 + (a + 4 * b) * x^2 + (b - 5 * a) * x + (10 - a)

theorem find_third_root (a b : ℝ) (h1 : P a b (-1) = 0) (h2 : P a b 4 = 0) : 
 ∃ c : ℝ, c ≠ -1 ∧ c ≠ 4 ∧ P a b c = 0 ∧ c = 8 / 3 :=
 sorry

end find_third_root_l583_583509


namespace probability_sum_of_roots_is_five_l583_583737

noncomputable def prob_sum_of_roots_is_five (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4.5 ∧ ((2.5 ≤ x ∧ x < 3) ∨ (3 ≤ x ∧ x < 3.5))

theorem probability_sum_of_roots_is_five : 
  (Set.Icc 0 4.5).measure (Set.Of (prob_sum_of_roots_is_five x)) = 2 / 9 := 
sorry

end probability_sum_of_roots_is_five_l583_583737


namespace distinct_real_roots_iff_l583_583565

noncomputable def operation (a b : ℝ) : ℝ := a * b^2 - b 

theorem distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation 1 x1 = k ∧ operation 1 x2 = k) ↔ k > -1/4 :=
by
  sorry

end distinct_real_roots_iff_l583_583565


namespace exists_circle_passing_through_A_B_cutting_chord_l583_583512

-- Definitions for given points A and B
variables (A B : Point)

-- Definition for given circle O
variable (O : Circle)

-- Definition for the length of the chord
variable (chord_length : ℝ)

-- Statement: There exists a circle passing through points A and B intersecting given circle O such that a chord of specified length is cut off from circle O
theorem exists_circle_passing_through_A_B_cutting_chord (A B : Point) (O : Circle) (chord_length : ℝ) :
  ∃ (C : Circle), passes_through C A ∧ passes_through C B ∧ intersects_with_chord_of_length C O chord_length :=
begin
  sorry
end

end exists_circle_passing_through_A_B_cutting_chord_l583_583512


namespace det_matrix_power_l583_583633

theorem det_matrix_power (A : Matrix ℕ ℕ ℝ) (h : det A = 3) : det (A ^ 4) = 81 :=
by
  sorry

end det_matrix_power_l583_583633


namespace cos_alpha_plus_pi_six_cos_two_alpha_plus_seven_pi_twelve_l583_583685

theorem cos_alpha_plus_pi_six 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < pi / 3) 
  (h3 : sqrt 3 * sin α + cos α = sqrt 6 / 2) : 
  cos (α + pi / 6) = sqrt 10 / 4 := 
sorry

theorem cos_two_alpha_plus_seven_pi_twelve 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < pi / 3) 
  (h3 : sqrt 3 * sin α + cos α = sqrt 6 / 2) : 
  cos (2 * α + 7 * pi / 12) = (sqrt 2 - sqrt 30) / 8 := 
sorry

end cos_alpha_plus_pi_six_cos_two_alpha_plus_seven_pi_twelve_l583_583685


namespace find_polynomial_l583_583528

open Nat

-- defining the problem parameters and conditions
def polynomial (P : ℕ → ℤ) (d : ℕ) : Prop :=
  (odd d) ∧ 
  (∀ n : ℕ, ∃ (x : Fin n → ℕ), ∀ i j : Fin n,
  (1 / 2 < (P x i) / (P x j) < 2) ∧ 
  ∃ r : ℚ, (P x j) / (P x i) = r^d)

-- statement to prove the form of these polynomials
theorem find_polynomial (P : ℕ → ℤ) (d : ℕ) : 
  polynomial P d →
  ∃ (a r s : ℤ), a ≠ 0 ∧ 1 ≤ r ∧ coprime r s ∧ P = λ x, a * (r * x + s)^d :=
sorry -- proof will be provided here

end find_polynomial_l583_583528


namespace payroll_amount_l583_583076

theorem payroll_amount (P : ℝ) 
  (h1 : P > 500000) 
  (h2 : 0.004 * (P - 500000) - 1000 = 600) :
  P = 900000 :=
by
  sorry

end payroll_amount_l583_583076


namespace integer_solutions_count_l583_583968

theorem integer_solutions_count :
  let cond1 (x : ℤ) := -4 * x ≥ 2 * x + 9
  let cond2 (x : ℤ) := -3 * x ≤ 15
  let cond3 (x : ℤ) := -5 * x ≥ x + 22
  ∃ s : Finset ℤ, 
    (∀ x ∈ s, cond1 x ∧ cond2 x ∧ cond3 x) ∧
    (∀ x, cond1 x ∧ cond2 x ∧ cond3 x → x ∈ s) ∧
    s.card = 2 :=
sorry

end integer_solutions_count_l583_583968


namespace all_sets_form_right_angled_triangle_l583_583086

theorem all_sets_form_right_angled_triangle :
    (6 * 6 + 8 * 8 = 10 * 10) ∧
    (7 * 7 + 24 * 24 = 25 * 25) ∧
    (3 * 3 + 4 * 4 = 5 * 5) ∧
    (Real.sqrt 2 * Real.sqrt 2 + Real.sqrt 3 * Real.sqrt 3 = Real.sqrt 5 * Real.sqrt 5) :=
by {
  sorry
}

end all_sets_form_right_angled_triangle_l583_583086


namespace maximum_people_shaked_hands_l583_583267

-- Given conditions
variables (N : ℕ) (hN : N > 4)
def has_not_shaken_hands_with (a b : ℕ) : Prop := sorry -- This should define the shaking hand condition

-- Main statement
theorem maximum_people_shaked_hands (h : ∃ i, has_not_shaken_hands_with i 2) :
  ∃ k, k = N - 3 := 
sorry

end maximum_people_shaked_hands_l583_583267


namespace sum_of_x_in_given_range_l583_583904

def is_integer (x : ℕ) : Prop :=
  (x : ℝ).denom = 1

theorem sum_of_x_in_given_range :
  (sum (filter (λ x, is_integer $ ∑ n in range (x - 1), real.log (real.exp ((n + 2) * real.log (n + 1)) / (n * real.log n))) (Icc 2 1000))) = 739 :=
sorry

end sum_of_x_in_given_range_l583_583904


namespace goblet_competition_points_difference_l583_583446

theorem goblet_competition_points_difference :
  let teams := 6
  let matches := teams * (teams - 1) / 2  -- each team plays every other team
  let points_for_win := 3
  let points_for_tie := 1
  let max_points := matches * points_for_win
  let min_points := matches * (2 * points_for_tie)
  max_points - min_points = 30 :=
by
  let teams := 6
  let matches := teams * (teams - 1) / 2  -- each team plays every other team
  let points_for_win := 3
  let points_for_tie := 1
  let max_points := matches * points_for_win
  let min_points := matches * (2 * points_for_tie)
  have h_matches: matches = 15 := by sorry
  have h_max_points: max_points = 45 := by sorry
  have h_min_points: min_points = 15 := by sorry
  show 45 - 15 = 30 from by sorry

end goblet_competition_points_difference_l583_583446


namespace seq_no_consecutive_ones_prob_l583_583478

theorem seq_no_consecutive_ones_prob :
  ∃ m n : ℕ, (∀ k : ℕ, k > m + n → (int.coe_nat n = 4096) → (376.gcd 4096 = 1) → 
    (0 < n → ∃ a b, n = a + b * 4096 → m+n = 4473)) :=
sorry

end seq_no_consecutive_ones_prob_l583_583478


namespace sum_of_first_15_terms_of_geometric_sequence_l583_583258

theorem sum_of_first_15_terms_of_geometric_sequence (a r : ℝ) 
  (h₁ : (a * (1 - r^5)) / (1 - r) = 10) 
  (h₂ : (a * (1 - r^10)) / (1 - r) = 50) : 
  (a * (1 - r^15)) / (1 - r) = 210 := 
by 
  sorry

end sum_of_first_15_terms_of_geometric_sequence_l583_583258


namespace number_of_red_squares_in_19th_row_l583_583458

-- Define the number of squares in the n-th row
def number_of_squares (n : ℕ) : ℕ := 3 * n - 1

-- Define the number of red squares in the n-th row
def red_squares (n : ℕ) : ℕ := (number_of_squares n) / 2

-- The theorem stating the problem
theorem number_of_red_squares_in_19th_row : red_squares 19 = 28 := by
  -- Proof goes here
  sorry

end number_of_red_squares_in_19th_row_l583_583458


namespace stone_radius_l583_583078

theorem stone_radius (hole_diameter hole_depth : ℝ) (r : ℝ) :
  hole_diameter = 30 → hole_depth = 10 → (r - 10)^2 + 15^2 = r^2 → r = 16.25 :=
by
  intros h_diam h_depth hyp_eq
  sorry

end stone_radius_l583_583078


namespace no_four_distinct_real_roots_l583_583130

theorem no_four_distinct_real_roots (a b : ℝ) : ¬ (∃ (x1 x2 x3 x4 : ℝ), 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧ 
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧ 
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧ 
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0)) :=
by
  sorry

end no_four_distinct_real_roots_l583_583130


namespace pyramid_base_cyclic_and_height_passes_center_l583_583722

theorem pyramid_base_cyclic_and_height_passes_center 
  (V : Type*)
  [InnerProductSpace ℝ V]
  (S : V) (A : ℕ → V) (n : ℕ)
  (h_angles_equal : ∀ i j : ℕ, i ≠ j → i < n → j < n → angle S (A i) (A j) = angle S (A j) (A i)) :
  ∃ O : V, (∀ i : ℕ, i < n → dist O (A i) = dist O (A (i+1) % n)) ∧ 
           (∀ i : ℕ, i < n → (O + S) ∈ line [O, S]) :=
sorry

end pyramid_base_cyclic_and_height_passes_center_l583_583722


namespace simplify_fraction_l583_583783

theorem simplify_fraction : (3 ^ 100 + 3 ^ 98) / (3 ^ 100 - 3 ^ 98) = 5 / 4 := 
by sorry

end simplify_fraction_l583_583783


namespace angle_BDC_15_l583_583733

-- Let A, B, C, D be points in the plane.
noncomputable def point (α : Type) := α

variables (A B C D : Type) [point A] [point B] [point C] [point D]

-- AB, AC, AD are segments where AB = AC and AC = AD.
axiom AB_AC : A → B → Prop
axiom AC_AD : A → C → Prop
axiom AB_AD : A → D → Prop

-- Define the equalities.
axiom eq_AB_AC : ∀ (A B C : Type) [AB_AC A B] [AB_AC A C], true
axiom eq_AC_AD : ∀ (A C D : Type) [AC_AD A C] [AC_AD C D], true

-- Given angle BAC is 30 degrees.
axiom angle_BAC_30 : ∀ (A B C : Type), Prop

-- Define the measure of the angle BAC.
axiom measure_bac : angle_BAC_30 A B C → 30 = 30

-- Proof of the problem statement: ∠BDC = 15 degrees.
theorem angle_BDC_15 :
  ∀  (A B C D : Type)
  [point A] [point B] [point C] [point D]
  [AB_AC A B] [AC_AD A C] [AB_AD A D]
  [angle_BAC_30 A B C],
  15 = 15 :=
by
  intros,
  sorry

end angle_BDC_15_l583_583733


namespace large_container_price_l583_583062

-- Define the volume function for a cylinder
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

-- Given conditions
def small_diameter := 4
def small_radius : ℝ := small_diameter / 2
def small_height := 5
def small_price := 0.80

def large_diameter := 8
def large_radius : ℝ := large_diameter / 2
def large_height := 10

-- Calculating the volumes
def small_volume := cylinder_volume small_radius small_height
def large_volume := cylinder_volume large_radius large_height

-- Volume ratio
def volume_ratio := large_volume / small_volume

-- Expected result for the larger container
def expected_large_price := volume_ratio * small_price

-- The proof statement assuming the price is proportional to the volume
theorem large_container_price : expected_large_price = 6.40 := by
  -- Proof omitted
  sorry

end large_container_price_l583_583062


namespace cobbler_charged_for_mold_l583_583498

noncomputable def cost_to_make_the_mold (total_payment : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) (discount : ℚ) : ℕ :=
  let cost_of_work := (hourly_rate * hours_worked : ℕ) in
  let discounted_work := (discount * (cost_of_work : ℚ)).to_nat in
  total_payment - discounted_work

theorem cobbler_charged_for_mold :
  cost_to_make_the_mold 730 75 8 0.80 = 250 :=
by
  -- Proof will be filled in.
  sorry

end cobbler_charged_for_mold_l583_583498


namespace mowing_problem_l583_583099

noncomputable def area (x : ℝ) (f : ℝ → ℝ) (input : ℝ) : ℝ := f input

noncomputable def rate (y : ℝ) (g : ℝ → ℝ) (input : ℝ) : ℝ := g input

theorem mowing_problem (x y : ℝ)
  (b_area : area x (λ _ , x) x)
  (s_area : area x (λ _ , x / 3) x)
  (t_area : area x (λ _ , x / 4) x)
  (b_rate : rate y (λ _ , y) y)
  (s_rate : rate y (λ _ , y / 4) y)
  (t_rate : rate y (λ _ , y / 8) y) :
  (x / y) < (4 * x / (3 * y)) ∧ (x / y) < (2 * x / y) := sorry

end mowing_problem_l583_583099


namespace gcd_poly_l583_583938

theorem gcd_poly (k : ℕ) : Nat.gcd ((4500 * k)^2 + 11 * (4500 * k) + 40) (4500 * k + 8) = 3 := by
  sorry

end gcd_poly_l583_583938


namespace triangle_possible_second_group_l583_583570

-- Definitions and conditions from the problem
variables {a_1 b_1 c_1 a_2 b_2 c_2 : ℝ}
variables (swap_first swap_second : ℝ)
variables (sum_eq_2_1 : a_1 + b_1 + c_1 = 2)
variables (sum_eq_2_2 : a_2 + b_2 + c_2 = 2)
variables (not_triangle_1 : ¬(a_2 < b_1 + c_1 ∧ b_1 < a_2 + c_1 ∧ c_1 < a_2 + b_1))

-- Prove the statement
theorem triangle_possible_second_group (
  hswap1 : swap_first ∈ {a_1, b_1, c_1},
  hswap2 : swap_second ∈ {a_2, b_2, c_2},
  hswap : (hswap1, hswap2) = (a_1, a_2) ∨ (hswap1, hswap2) = (b_1, b_2) ∨ (hswap1, hswap2) = (c_1, c_2)
) :
  a_1 < b_2 + c_2 ∧ b_2 < a_1 + c_2 ∧ c_2 < a_1 + b_2 :=
by
  sorry

end triangle_possible_second_group_l583_583570


namespace geom_seq_sum_eqn_l583_583991

theorem geom_seq_sum_eqn (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 2 + 2 * a 3 = a 1)
  (h2 : a 1 * a 4 = a 6)
  (h3 : ∀ n, a (n + 1) = a 1 * (1 / 2) ^ n)
  (h4 : ∀ n, S n = 2 * ((1 - (1 / 2) ^ n) / (1 - (1 / 2)))) :
  a n + S n = 4 :=
sorry

end geom_seq_sum_eqn_l583_583991


namespace max_value_90_l583_583400

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end max_value_90_l583_583400


namespace fluffy_striped_or_spotted_cats_l583_583666

theorem fluffy_striped_or_spotted_cats (total_cats : ℕ) (striped_fraction : ℚ) (spotted_fraction : ℚ)
    (fluffy_striped_fraction : ℚ) (fluffy_spotted_fraction : ℚ) (striped_spotted_fraction : ℚ) :
    total_cats = 180 ∧ striped_fraction = 1/2 ∧ spotted_fraction = 1/3 ∧
    fluffy_striped_fraction = 1/8 ∧ fluffy_spotted_fraction = 3/7 →
    striped_spotted_fraction = 36 :=
by
    sorry

end fluffy_striped_or_spotted_cats_l583_583666


namespace determine_all_counterfeit_pile_l583_583060

/--
  There are three piles of coins, with sizes 15, 19, and 25.
  One of these piles contains a real coin among counterfeit coins,
  and the other two piles contain only counterfeit coins.
  All counterfeit coins weigh the same, but the real coin has a different weight.
  The problem is to prove that the counterfeiter can determine the pile with all counterfeit coins using one weighing on a balance scale.
-/
theorem determine_all_counterfeit_pile (w15 w19 w25 : ℕ) (real_weight counterfeit_weight : ℕ) :
  (w15 = 15 ∧ w19 = 19 ∧ w25 = 25 ∧ (∃ pile, pile ∈ {15, 19, 25} ∧ (w15 = pile ∨ w19 = pile ∨ w25 = pile) ∧ (w15 ≠ w19 ∨ w15 ≠ w25 ∨ w19 ≠ w25) ∧ real_weight ≠ counterfeit_weight)) →
  ∃ pile, pile ∈ {15, 19, 25} ∧ (w15 = pile ∨ w19 = pile ∨ w25 = pile) ∧ (pile = 15 ∨ pile = 19 ∨ pile = 25)
:=
sorry

end determine_all_counterfeit_pile_l583_583060


namespace find_point_P_on_parabola_l583_583206

theorem find_point_P_on_parabola :
  let P := (1/2, (1/2)^2) in
  ∀ (x : ℝ), y = x^2 → (1 - P.1) * (x - P.1) = 0 → x = P.1 →
  (∃ y, x = 1/2 ∧ y = 1/4) :=
by
  sorry

end find_point_P_on_parabola_l583_583206


namespace original_price_of_petrol_l583_583473

theorem original_price_of_petrol (P : ℝ): 
  (∀ P, 200 / (0.9 * P) - 200 / P = 5 → P ≈ 2.11) :=
by
  sorry

end original_price_of_petrol_l583_583473


namespace arithmetic_sequence_fifth_term_l583_583582

theorem arithmetic_sequence_fifth_term (a1 d : ℕ) (a_n : ℕ → ℕ) 
  (h_a1 : a1 = 2) (h_d : d = 1) (h_a_n : ∀ n : ℕ, a_n n = a1 + (n-1) * d) : 
  a_n 5 = 6 := 
    by
    -- Given the conditions, we need to prove a_n evaluated at 5 is equal to 6.
    sorry

end arithmetic_sequence_fifth_term_l583_583582


namespace coin_stacking_count_l583_583790

theorem coin_stacking_count : 
  ∀(coins : list ℕ), 
  (coins.length = 4) → 
  (∀ i, i < coins.length - 1 → coins[i] ≠ coins[i + 1]) → 
  (set.image (λ c, c :: coins) {0, 1}).to_finset.card = 5 :=
sorry

end coin_stacking_count_l583_583790


namespace quarters_to_dollars_l583_583629

theorem quarters_to_dollars (total_quarters : ℕ) (quarters_per_dollar : ℕ) (h1 : total_quarters = 8) (h2 : quarters_per_dollar = 4) : total_quarters / quarters_per_dollar = 2 :=
by {
  sorry
}

end quarters_to_dollars_l583_583629


namespace gcd_fac_7_and_8_equals_5040_l583_583153

theorem gcd_fac_7_and_8_equals_5040 : Nat.gcd 7! 8! = 5040 := 
by 
  sorry

end gcd_fac_7_and_8_equals_5040_l583_583153


namespace number_of_functions_number_of_injections_number_of_surjections_l583_583052

-- Part (i)
theorem number_of_functions (m n : ℕ) : n^m = n^m :=
sorry

-- Part (ii)
theorem number_of_injections (m n : ℕ) (h : m ≤ n) : 
  (finset.perm m).card = n! / (n - m)! :=
sorry

-- Part (iii)
theorem number_of_surjections (m n : ℕ) (h : m ≥ n) : 
  (finset.surj m n).card = ∑ k in finset.range (n + 1), (-1)^k * n.choose k * (n - k)^m :=
sorry

end number_of_functions_number_of_injections_number_of_surjections_l583_583052


namespace distance_AB_is_5_l583_583589

def point (x y : ℝ) : Type := ℝ

def A : point := point.mk (-1) 2
def B : point := point.mk (-4) 6

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_AB_is_5 :
  distance A B = 5 := by
  sorry

end distance_AB_is_5_l583_583589


namespace distance_between_A_B_l583_583712

theorem distance_between_A_B : 
    let A := -2006
    let B := +17
    |B - A| = 2023 :=
by
  let A := -2006
  let B := +17
  sorry

end distance_between_A_B_l583_583712


namespace bill_spots_l583_583100

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end bill_spots_l583_583100


namespace squares_not_all_congruent_l583_583788

/-- Proof that the statement "all squares are congruent to each other" is false. -/
theorem squares_not_all_congruent : ¬(∀ (a b : ℝ), a = b ↔ a = b) :=
by 
  sorry

end squares_not_all_congruent_l583_583788


namespace greatest_possible_number_l583_583423

theorem greatest_possible_number :
  exists (n : ℕ),
    (∃ (d : ℕ), n = 98_6310 ∧ d < 10^(6)) ∧
    Nat.digits 10 n = [9, 8, 6, 3, 1, 0] ∧
    ∀ m x, Nat.digits 10 m = [1, 3, 4, 6, 8, 9] → Nat.digits 10 x = [0] → x = d → n > m :=
sorry

end greatest_possible_number_l583_583423


namespace min_value_EP_dot_QP_is_6_l583_583339

noncomputable def min_value_dot_product : ℝ :=
    let E : ℝ × ℝ := (3, 0) in
    let ellipse (x y : ℝ) := x^2 / 36 + y^2 / 9 = 1 in
    let EP (P : ℝ × ℝ) := (P.1 - E.1, P.2 - E.2) in
    let QP (P Q : ℝ × ℝ) := (Q.1 - P.1, Q.2 - P.2) in
    let dot (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 in
    infi (λ P, if ellipse P.1 P.2 ∧ ∃ Q, dot (EP P) (QP P Q) = 0
          then dot (EP P) (QP P E)
          else ⊤)

theorem min_value_EP_dot_QP_is_6 :
  min_value_dot_product = 6 :=
begin
  sorry -- Proof not required
end

end min_value_EP_dot_QP_is_6_l583_583339


namespace chess_total_games_played_l583_583760
noncomputable theory

def total_games_played (n : ℕ) : ℕ :=
  n.choose 2

theorem chess_total_games_played : total_games_played 30 = 435 := by
  sorry

end chess_total_games_played_l583_583760


namespace alcohol_solution_problem_l583_583369

theorem alcohol_solution_problem (x_vol y_vol : ℚ) (x_alcohol y_alcohol target_alcohol : ℚ) (target_vol : ℚ) :
  x_vol = 250 ∧ x_alcohol = 10/100 ∧ y_alcohol = 30/100 ∧ target_alcohol = 25/100 ∧ target_vol = 250 + y_vol →
  (x_alcohol * x_vol + y_alcohol * y_vol = target_alcohol * target_vol) →
  y_vol = 750 :=
by
  sorry

end alcohol_solution_problem_l583_583369


namespace scheduling_courses_l583_583415

theorem scheduling_courses :
  ∃ slots : Finset (Finset ℕ), 
    (∀ s ∈ slots, ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≠ y → abs (x - y) > 1) → 
    slots.card * 6 = 24 :=
begin
  sorry
end

end scheduling_courses_l583_583415


namespace smallest_odd_n_3_product_gt_5000_l583_583431

theorem smallest_odd_n_3_product_gt_5000 :
  ∃ n : ℕ, (∃ k : ℤ, n = 2 * k + 1 ∧ n > 0) ∧ (3 ^ ((n + 1)^2 / 8)) > 5000 ∧ n = 8 :=
by
  sorry

end smallest_odd_n_3_product_gt_5000_l583_583431


namespace sum_binomial_coefficients_l583_583306

theorem sum_binomial_coefficients (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range n, (k + 1) * Nat.choose n (k + 1)) = n * 2^(n-1) :=
by
  sorry

end sum_binomial_coefficients_l583_583306


namespace spherical_to_cartesian_l583_583606

theorem spherical_to_cartesian :
  ∀ (r θ φ : ℝ),
    r = 1 → θ = π / 3 → φ = π / 6 →
    let x := r * sin θ * cos φ in
    let y := r * sin θ * sin φ in
    let z := r * cos θ in
    (x, y, z) = (3 / 4, sqrt 3 / 4, 1 / 2) :=
by
  intros r θ φ hr hθ hφ
  simp [hr, hθ, hφ]
  sorry

end spherical_to_cartesian_l583_583606


namespace trajectory_is_ellipse_l583_583622

variables {P F1 F2 : ℝ × ℝ}
def d1 : ℝ := dist P F1
def d2 : ℝ := dist P F2
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def dist (P F : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

theorem trajectory_is_ellipse (P : ℝ × ℝ) (d1 d2 : ℝ)
  (H1 : d1 = dist P F1) (H2 : d2 = dist P F2) 
  (H3 : dist F1 F2 = (d1 + d2) / 2) : 
  (d1 + d2 = 4) ↔ trajectory_is_ellipse :=
sorry

end trajectory_is_ellipse_l583_583622


namespace sum_c_d_l583_583233

theorem sum_c_d (c d : ℝ) (h : ∀ x, (x - 2) * (x + 3) = x^2 + c * x + d) :
  c + d = -5 :=
sorry

end sum_c_d_l583_583233


namespace arithmetic_geometric_sequence_min_sum_l583_583009

theorem arithmetic_geometric_sequence_min_sum :
  ∃ (A B C D : ℕ), 
    (C - B = B - A) ∧ 
    (C * 4 = B * 7) ∧ 
    (D * 4 = C * 7) ∧ 
    (16 ∣ B) ∧ 
    (A + B + C + D = 97) :=
by sorry

end arithmetic_geometric_sequence_min_sum_l583_583009


namespace pages_remaining_total_l583_583113

-- Define the conditions
def total_pages_book1 : ℕ := 563
def read_pages_book1 : ℕ := 147

def total_pages_book2 : ℕ := 849
def read_pages_book2 : ℕ := 389

def total_pages_book3 : ℕ := 700
def read_pages_book3 : ℕ := 134

-- The theorem to be proved
theorem pages_remaining_total :
  (total_pages_book1 - read_pages_book1) + 
  (total_pages_book2 - read_pages_book2) + 
  (total_pages_book3 - read_pages_book3) = 1442 := 
by
  sorry

end pages_remaining_total_l583_583113


namespace varphi_value_max_y_value_in_interval_l583_583213

-- Definitions of f and g
def f (x varphi : ℝ) := cos (x + varphi)
def f' (x varphi : ℝ) := -sin (x + varphi)
def g (x varphi : ℝ) := f x varphi + f' x varphi

-- Given conditions
variable {varphi : ℝ}
variable h1 : -π < varphi ∧ varphi < 0
variable h2 : ∀ x : ℝ, g x varphi = g (-x) varphi

-- Proof for the value of varphi
theorem varphi_value : varphi = -π / 4 := sorry

-- Maximum value of y over [0, π/4]
def y (x varphi : ℝ) := f x varphi * g x varphi

theorem max_y_value_in_interval : 
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π / 4 ∧ y x (-π/4) = (sqrt 2 + 1) / 2 := sorry

end varphi_value_max_y_value_in_interval_l583_583213


namespace number_of_correct_statements_l583_583849

theorem number_of_correct_statements :
  let s1 := ¬ (∀ x : ℝ, x^2 - 3*x - 2 ≥ 0) = (∃ x : ℝ, x^2 - 3*x - 2 < 0),
      s2 := ∀ P Q : Prop, (P ∨ Q) → P ∧ Q → false,
      s3 := ¬ (∃ m : ℝ, ∀ x : ℝ, (m ≠ 1 ∨ f x = mx^(m^2 + 2m))),
      s4 := ∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) → (¬ (∀ x y : ℝ, x = a ∧ y = b → x/a + y/b = 1)) 
  in s1 + s2 + s3 + s4 = 2 := sorry

end number_of_correct_statements_l583_583849


namespace prime_count_at_least_two_l583_583564

theorem prime_count_at_least_two :
  ∃ (n1 n2 : ℕ), n1 ≥ 2 ∧ n2 ≥ 2 ∧ (n1 ≠ n2) ∧ Prime (n1^3 + n1^2 + 1) ∧ Prime (n2^3 + n2^2 + 1) := 
by
  sorry

end prime_count_at_least_two_l583_583564


namespace flagpole_perpendicular_to_ground_l583_583287

variables (l : Type) (π : Type) [line l] [plane π]

theorem flagpole_perpendicular_to_ground : is_perpendicular l π :=
sorry

end flagpole_perpendicular_to_ground_l583_583287


namespace find_number_l583_583807

theorem find_number (x : ℝ) (h : 0.30 * x = 90 + 120) : x = 700 :=
by 
  sorry

end find_number_l583_583807


namespace proof_problem_l583_583942

theorem proof_problem
  {f : ℝ → ℝ}
  (domain : ∀ x, 0 < x ∧ x < (π / 2) → f x ≠ 0)
  (derivative : ∀ x, DifferentiableAt ℝ f x)
  (condition : ∀ x, 0 < x ∧ x < (π / 2) → HasDerivAt f (f' x) x)
  (ineq : ∀ x, 0 < x ∧ x < (π / 2) → (f' x) * sin x < (f x) * cos x) :
  sqrt 3 * f (π / 4) > sqrt 2 * f (π / 3) :=
sorry

end proof_problem_l583_583942


namespace pentagon_perimeter_l583_583031

/-- Given the side lengths and the fact that triangle AED is a right triangle, we want to prove
    that the perimeter of the pentagon ABCDE is 14 + 6 * sqrt 2. -/
theorem pentagon_perimeter (AB BC CD DE : ℝ) (AED_right : ∃ AE ED AD, AE = AB + BC ∧ ED = DE ∧ AD^2 = AE^2 + ED^2)
  (h_AB : AB = 4) (h_BC : BC = 2) (h_CD : CD = 2) (h_DE : DE = 6) :
  (AB + BC + CD + DE + sqrt (AB + BC)^2 + DE^2) = 14 + 6 * Real.sqrt 2 :=
by
  sorry

end pentagon_perimeter_l583_583031


namespace fraction_of_q_age_l583_583098

theorem fraction_of_q_age (P Q : ℕ) (h1 : P / Q = 3 / 4) (h2 : P + Q = 28) : (P - 0) / (Q - 0) = 3 / 4 :=
by
  sorry

end fraction_of_q_age_l583_583098


namespace center_and_radius_tangent_lines_eq_intercepts_trajectory_point_P_l583_583922

open Real

def circle (x y : ℝ) := x^2 + y^2 + 2*x - 4*y + 3 = 0

theorem center_and_radius :
  ∃ x y r : ℝ, circle x y → (x, y) = (-1, 2) ∧ r = sqrt 2 := 
sorry

theorem tangent_lines_eq_intercepts :
  ∃ a : ℝ, (a ≠ 0) ∧
  (∀ x y : ℝ, (circle x y ∧ x + y = a) →
  (x + y + 1 = 0 ∨ x + y - 3 = 0)) :=
sorry

theorem trajectory_point_P :
  ∀ (x y : ℝ), (∀ O P : ℝ,  |O - P| = |O - (2, 4)|) →
  (circle x y → 2*x - 4*y + 3 = 0) :=
sorry

end center_and_radius_tangent_lines_eq_intercepts_trajectory_point_P_l583_583922


namespace inequality_solution_l583_583374

noncomputable def problem_statement (x : ℝ) : Prop :=
  (0 ≤ x / ((x + 1) * (x + 2))) ↔ (x ∈ Set.Icc 0 ∞ ∨ x ∈ Set.Ioo (-2 : ℝ) (-1))

theorem inequality_solution :
  ∀ x : ℝ, problem_statement x := by
  sorry

end inequality_solution_l583_583374


namespace tire_circumference_l583_583444

theorem tire_circumference 
  (rev_per_min : ℝ) -- revolutions per minute
  (car_speed_kmh : ℝ) -- car speed in km/h
  (conversion_factor : ℝ) -- conversion factor for speed from km/h to m/min
  (min_to_meter : ℝ) -- multiplier to convert minutes to meters
  (C : ℝ) -- circumference of the tire in meters
  : rev_per_min = 400 ∧ car_speed_kmh = 120 ∧ conversion_factor = 1000 / 60 ∧ min_to_meter = 1000 / 60 ∧ (C * rev_per_min = car_speed_kmh * min_to_meter) → C = 5 :=
by
  sorry

end tire_circumference_l583_583444


namespace trigonometric_identity_l583_583607

theorem trigonometric_identity (x y : ℝ) (h1 : x = -5) (h2 : y = 12) :
  let r := real.sqrt (x^2 + y^2)
  in r = 13 → sin (real.arctan2 y x) + 2 * cos (real.arctan2 y x) = 2 / 13 := 
by 
  intros 
  sorry

end trigonometric_identity_l583_583607


namespace find_m_n_l583_583972

theorem find_m_n :
  ∀ (m n : ℤ), (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m * x + n) → 
  (m = 4 ∧ n = -32) :=
by
  intros m n h
  let x := 0
  sorry

end find_m_n_l583_583972


namespace min_reciprocal_sum_l583_583598

theorem min_reciprocal_sum (m n : ℝ) 
  (h1 : m * 1 + n * 1 - 2 = 0) 
  (h2 : m * n > 0) : 
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ m, n, (mn > 0) ∧ (m + n = 2) → (1/m + 1/n ≥ 2) :=
by
  sorry

end min_reciprocal_sum_l583_583598


namespace prob_within_region_l583_583191

theorem prob_within_region :
  let points := {p ∈ ( {-1, 1} : set ℤ ) × ( {-2, 0, 2} : set ℤ ) | p.1 + 2 * p.2 >= 1} in
  (set.card points : ℚ) / 6 = 1 / 2 :=
by
  sorry

end prob_within_region_l583_583191


namespace speed_of_water_l583_583073

theorem speed_of_water (v : ℝ) (swim_speed_still_water : ℝ)
  (distance : ℝ) (time : ℝ)
  (h1 : swim_speed_still_water = 4) 
  (h2 : distance = 14) 
  (h3 : time = 7) 
  (h4 : 4 - v = distance / time) : 
  v = 2 := 
sorry

end speed_of_water_l583_583073


namespace CHIEF_arrangement_l583_583277

theorem CHIEF_arrangement : 
  let letters := ['C', 'H', 'I', 'E', 'F'],
      total_arrangements := (list.permutations letters).length,
      valid_arrangements := total_arrangements / 2
  in valid_arrangements = 60 :=
by
  -- sorry allows us to skip the actual proof steps
  sorry

end CHIEF_arrangement_l583_583277


namespace number_of_solutions_l583_583228

theorem number_of_solutions :
  ∃ sols: Finset (ℕ × ℕ), (∀ (x y : ℕ), (x, y) ∈ sols ↔ x^2 + y^2 + 2*x*y - 1988*x - 1988*y = 1989 ∧ x > 0 ∧ y > 0)
  ∧ sols.card = 1988 :=
by
  sorry

end number_of_solutions_l583_583228


namespace log2_arith_seq_sum_l583_583186

-- Define an arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: (a 5 = 2)
variables (a : ℕ → ℝ) (h_arith : arithmetic_seq a) (h_a5 : a 5 = 2)

-- Statement to prove: log2(a 4 + a 6) = 2
theorem log2_arith_seq_sum (a : ℕ → ℝ) (h_arith : arithmetic_seq a) (h_a5 : a 5 = 2) : 
  Real.log2 (a 4 + a 6) = 2 :=
sorry

end log2_arith_seq_sum_l583_583186


namespace isosceles_right_triangle_angle_l583_583275

-- Hypotenuse and angle definitions for the isosceles right triangle
variables (a : ℝ) (θ : ℝ)

-- Definition of hypotenuse in an isosceles right triangle
def hypotenuse (a : ℝ) := a * (Real.sqrt 2)

-- The given condition about the square of the hypotenuse
def given_condition (a : ℝ) (θ : ℝ) :=
  (hypotenuse a) ^ 2 = 4 * a * Real.cos θ

-- The theorem to prove
theorem isosceles_right_triangle_angle (a : ℝ) (h : ℝ) (θ : ℝ)
  (H: h = hypotenuse a) 
  (C: given_condition a θ) :
  θ = Real.pi / 3 :=
sorry

end isosceles_right_triangle_angle_l583_583275


namespace find_alpha_l583_583189

noncomputable def angle_in_interval (α : ℝ) : Prop :=
  370 < α ∧ α < 520 

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = 1 / 2) (h_interval: angle_in_interval α) : α = 420 :=
sorry

end find_alpha_l583_583189


namespace binomial_calc_l583_583104

theorem binomial_calc : (Nat.choose 10 3) * (Nat.choose 8 3) * (Nat.fact 7 / Nat.fact 4) = 235200 := 
by
  sorry

end binomial_calc_l583_583104


namespace tangent_lengths_inequality_l583_583177

noncomputable def circle (k : Type) := sorry
noncomputable def segment (A B : Type) := sorry
noncomputable def tangent_length (P : Type) (c : circle) : ℝ := sorry
noncomputable def intersects (seg : segment) (c : circle) : Prop := sorry
noncomputable def length (seg : segment) : ℝ := sorry

theorem tangent_lengths_inequality (k : Type) (A B : Type)
  (a b : ℝ) (seg : segment A B) (c : circle k) :
  tangent_length A c = a → 
  tangent_length B c = b →
  (a + b > length seg ↔ ¬ intersects seg c) :=
sorry

end tangent_lengths_inequality_l583_583177


namespace line_through_P_and_origin_line_through_P_and_perpendicular_to_l3_l583_583608

variable (x y : ℝ)
variable (P : ℝ × ℝ) (l1 l2 l3 : ℝ → ℝ → Prop)

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the point P as the intersection of l1 and l2
def P : ℝ × ℝ := (-2, 2)

-- Prove the line passing through point P and origin has the equation x + y = 0
theorem line_through_P_and_origin :
  ∀ (x y : ℝ), (y = x * (2 / -2)) → (x + y = 0) := by
  sorry

-- Define the perpendicular line through point P to l3
def perpendicular_line := λ x y : ℝ, 2 * y - x - 6 = 0

-- Prove the line passing through point P and perpendicular to l3 has the equation x - 2y + 6 = 0 
theorem line_through_P_and_perpendicular_to_l3 :
  ∀ (x y : ℝ), (2 * y - x + 6 = 0) → (x - 2 * y + 6 = 0) := by
  sorry

end line_through_P_and_origin_line_through_P_and_perpendicular_to_l3_l583_583608


namespace count_three_digit_x_l583_583303

def heartsuit (x : ℕ) : ℕ :=
  (x.toString.data.map (λ c, c.toNat - '0'.toNat)).sum

def heartsuit_heartsuit_condition (x : ℕ) : Prop :=
  heartsuit (heartsuit x) = 5

theorem count_three_digit_x :
  (finset.range 900).filter (λ x, 100 ≤ x ∧ heartsuit_heartsuit_condition x)).card = 60 := 
sorry

end count_three_digit_x_l583_583303


namespace negation_of_universal_proposition_l583_583748

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0 :=
by sorry

end negation_of_universal_proposition_l583_583748


namespace man_age_difference_l583_583828

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end man_age_difference_l583_583828


namespace usual_time_is_20_l583_583045

-- Define the problem
variables (T T': ℕ)

-- Conditions
axiom condition1 : T' = T + 5
axiom condition2 : T' = 5 * T / 4

-- Proof statement
theorem usual_time_is_20 : T = 20 :=
  sorry

end usual_time_is_20_l583_583045


namespace binary_ternary_equality_l583_583645

theorem binary_ternary_equality (a b : Nat) (h_a : a = 1) (h_b : b = 1) (ha_condition : a ∈ {0, 1, 2}) (hb_condition : b ∈ {0, 1}) :
  let binary := 1 * 2^3 + 0 * 2^2 + b * 2^1 + 1
  let ternary := a * 3^2 + 0 * 3^1 + 2
  binary = ternary :=
by {
  sorry
}

end binary_ternary_equality_l583_583645


namespace range_of_m_l583_583216

theorem range_of_m (m : Real) :
  (∀ x y : Real, 0 < x ∧ x < y ∧ y < (π / 2) → 
    (m - 2 * Real.sin x) / Real.cos x > (m - 2 * Real.sin y) / Real.cos y) →
  m ≤ 2 := 
sorry

end range_of_m_l583_583216


namespace triangle_area_l583_583269

structure Point where
  x : ℝ
  y : ℝ

def area_triangle (A B C : Point) : ℝ := 
  0.5 * (B.x - A.x) * (C.y - A.y)

theorem triangle_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨8, 15⟩
  let C : Point := ⟨8, 0⟩
  area_triangle A B C = 60 :=
by
  sorry

end triangle_area_l583_583269


namespace index_card_area_l583_583486

theorem index_card_area (a b : ℕ) (new_area : ℕ) (reduce_length reduce_width : ℕ)
  (original_length : a = 3) (original_width : b = 7)
  (reduced_area_condition : a * (b - reduce_width) = new_area)
  (reduce_width_2 : reduce_width = 2) 
  (new_area_correct : new_area = 15) :
  (a - reduce_length) * b = 7 := by
  sorry

end index_card_area_l583_583486


namespace trigonometric_identity_trigonometric_expression_equals_two_l583_583757

-- Setting up the conditions and identities as theorems or lemmas
theorem trigonometric_identity (α : ℝ) :
  sin(π + α) = -sin(α) ∧ cos(π + α) = -cos(α) ∧ cos(-α) = cos(α) :=
by sorry

-- Proving the main problem using the identities
theorem trigonometric_expression_equals_two (α : ℝ) :
  sin(π + α)^2 - cos(π + α) * cos(-α) + 1 = 2 :=
by
  have h := trigonometric_identity α,
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  rw [pow_two, pow_two],
  ring
  sorry


end trigonometric_identity_trigonometric_expression_equals_two_l583_583757


namespace peyton_manning_total_yards_l583_583715

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end peyton_manning_total_yards_l583_583715


namespace sector_arc_length_l583_583380

theorem sector_arc_length (n : ℝ) (r : ℝ) (l : ℝ) (h1 : n = 90) (h2 : r = 3) (h3 : l = (n * Real.pi * r) / 180) :
  l = (3 / 2) * Real.pi := by
  rw [h1, h2] at h3
  sorry

end sector_arc_length_l583_583380


namespace exists_odd_point_l583_583680

-- Let P be a sequence of distinct points in the xy-plane.
variable (P : ℕ → ℤ × ℤ)

-- Conditions:
-- 1. Distinct points P_0, P_1, ..., P_1992 in the xy-plane with integer coordinates
def distinct_points (P : ℕ → ℤ × ℤ) : Prop :=
  ∀ (i j : ℕ), (i < 1993) → (j < 1993) → (i ≠ j) → P i ≠ P j

-- 2. Coordinates of all P_i are integers (implicit in using ℤ × ℤ)
-- 3. No other integer coordinate points on line segment P_i to P_(i+1)
def no_other_integer_points (P : ℕ → ℤ × ℤ) : Prop :=
  ∀ (i : ℕ), (i < 1992) →
  ¬∃ (a b : ℤ), (a, b) ∈ { t • (P (i + 1)).1 + (1-t) • (P i).1,
                             t • (P (i + 1)).2 + (1-t) • (P i).2 | t ∈ Ico 0 1 }
          ∧ (a, b) ≠ P i ∧ (a, b) ≠ P (i + 1)

-- Prove that for some i, 0 ≤ i ≤ 1992, there is a point Q on the segment joining P_i and P_(i+1)
-- such that both 2q_x and 2q_y are odd integers.
theorem exists_odd_point (P : ℕ → ℤ × ℤ)
  (h_distinct : distinct_points P)
  (h_no_other : no_other_integer_points P):
  ∃ (i : ℕ) (Q : ℤ × ℤ), (0 ≤ i ∧ i ≤ 1992) ∧ 
  (Q ∈ { t • (P (i + 1)).1 + (1-t) • (P i).1, t • (P (i + 1)).2 + (1-t) • (P i).2 | t ∈ Ico 0 1 }) ∧
  (2 * Q.1 % 2 = 1) ∧ (2 * Q.2 % 2 = 1) :=
by sorry

end exists_odd_point_l583_583680


namespace blocks_tower_l583_583356

theorem blocks_tower (T H Total : ℕ) (h1 : H = 53) (h2 : Total = 80) (h3 : T + H = Total) : T = 27 :=
by
  -- proof goes here
  sorry

end blocks_tower_l583_583356


namespace smallest_positive_integer_adding_to_725_is_5_l583_583428

theorem smallest_positive_integer_adding_to_725_is_5 :
  ∃ n : ℕ, n > 0 ∧ (725 + n) % 5 = 0 ∧ (∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m) :=
begin
  use 5,
  split,
  { exact nat.succ_pos' 4 },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  {
    intros m hm_mod hm_le,
    by_contra h,
    have : m < 5 := lt_of_not_ge h,
    have hm_lt_5 : (725 + m) % 5 < 5 := (nat.mod_lt (725 + m) (nat.zero_lt_succ 4)),
    linarith,
  }
end

end smallest_positive_integer_adding_to_725_is_5_l583_583428


namespace max_gcd_of_sequence_l583_583751

theorem max_gcd_of_sequence : 
  ∀ n : ℕ, let a_n := 100 + n^2 in
           let a_n1 := 100 + (n+1)^2 in
           let d_n := Nat.gcd a_n a_n1 in
           d_n ≤ 401 :=
by
  sorry

end max_gcd_of_sequence_l583_583751


namespace part1_expression_value_l583_583803

theorem part1_expression_value :
  2 * Real.cos (Real.pi / 6) - Real.tan (Real.pi / 3) + Real.sin (Real.pi / 6) + Real.abs (-1 / 2) = 1 := 
by
  sorry

end part1_expression_value_l583_583803


namespace bob_average_speed_l583_583496

theorem bob_average_speed
  (lap_distance : ℕ) (lap1_time lap2_time lap3_time total_laps : ℕ)
  (h_lap_distance : lap_distance = 400)
  (h_lap1_time : lap1_time = 70)
  (h_lap2_time : lap2_time = 85)
  (h_lap3_time : lap3_time = 85)
  (h_total_laps : total_laps = 3) : 
  (lap_distance * total_laps) / (lap1_time + lap2_time + lap3_time) = 5 := by
    sorry

end bob_average_speed_l583_583496


namespace tigers_win_probability_l583_583736

/-- The probability of the Tigers winning the World Series given the conditions -/
theorem tigers_win_probability :
  let p_tigers_win_game := 2 / 3
      p_marlins_win_game := 1 / 3
      series_win_probability := (∑ k in finset.range(4), (nat.choose (3 + k) k) * (p_tigers_win_game^4) * (p_marlins_win_game^k)) in
  series_win_probability = 0.66 :=
by
  sorry

end tigers_win_probability_l583_583736


namespace bill_spots_39_l583_583103

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end bill_spots_39_l583_583103


namespace segment_length_PQ_l583_583994

-- Defining a right triangle PQR with a right angle at R
structure RightTriangle where
  P Q R : Type
  (isRightAngle : ∃ (A B C : P Q R), ∠(A B C) = 90)

-- Describing the medians
structure Medians where
  P Q R : Type
  S T : Type
  lengthPS : Real := 5
  lengthQT : Real := 3 * Real.sqrt (5)
  midpointS : S = midpoint Q R
  midpointT : T = midpoint P R

-- The proof problem based on given conditions
theorem segment_length_PQ (P Q R S T : Type) [RightTriangle P Q R] [Medians P Q R S T] :
  ∃ (PQ : Real), PQ = 2 * Real.sqrt (14) :=
sorry

end segment_length_PQ_l583_583994


namespace flower_order_ways_l583_583322

def flower_arrangements (X : ℕ) (crimson scarlet vermillion : ℕ) (adjacent : bool) : ℕ :=
  if X > 0 ∧ adjacent = false then 30 else 0

theorem flower_order_ways (X : ℕ) (crimson : ℕ) (scarlet : ℕ) (vermillion : ℕ) (adjacent : bool) :
  crimson = X ∧ scarlet = X ∧ vermillion = X ∧ adjacent = false →
  flower_arrangements X crimson scarlet vermillion adjacent = 30 := by 
  intro h
  cases h
  sorry

end flower_order_ways_l583_583322


namespace polar_to_rectangular_conversion_l583_583121

theorem polar_to_rectangular_conversion:
  ∀ (r θ : ℝ), r = 5 → θ = (5 * π) / 4 →
    let x := r * cos θ, y := r * sin θ in
    (x, y) = (- (5 * real.sqrt 2) / 2, - (5 * real.sqrt 2) / 2) :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  simp [(5 : ℝ), real.cos, real.sin, real.sqrt]
  sorry

end polar_to_rectangular_conversion_l583_583121


namespace time_for_A_to_complete_race_l583_583990

open Real

theorem time_for_A_to_complete_race (V_A V_B : ℝ) (T_A : ℝ) :
  (V_B = 4) →
  (V_B = 960 / T_A) →
  T_A = 1000 / V_A →
  T_A = 240 := by
  sorry

end time_for_A_to_complete_race_l583_583990


namespace arithmetic_sequence_50th_term_l583_583029

theorem arithmetic_sequence_50th_term :
  ∀ (a₁ d n : ℕ), a₁ = 2 → d = 6 → n = 50 → a₁ + (n - 1) * d = 296 :=
by
  intros a₁ d n h₁ h₂ h₃
  rw [h₁, h₂, h³]
  sorry

end arithmetic_sequence_50th_term_l583_583029


namespace no_special_set_of_size_1000_l583_583300

def special_set (A : Set ℝ) : Prop :=
  (∀ a ∈ A, a ≠ 0 ∧ a ≠ 1) ∧
  (∀ a ∈ A, (1 / a) ∈ A) ∧
  (∀ a ∈ A, (1 / (1 - a)) ∈ A)

theorem no_special_set_of_size_1000 :
  ¬∃ (A : Set ℝ), special_set A ∧ A.size = 1000 :=
sorry

end no_special_set_of_size_1000_l583_583300


namespace max_value_expression_l583_583397

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end max_value_expression_l583_583397


namespace installation_service_cost_l583_583883

theorem installation_service_cost :
  let curtains_cost := 2 * 30 in
  let prints_cost := 9 * 15 in
  let total_order_cost := 245 in
  let items_cost := curtains_cost + prints_cost in
  let service_cost := total_order_cost - items_cost in
  service_cost = 50 :=
by
  -- Definitions and conditions are specified above
  sorry

end installation_service_cost_l583_583883


namespace teammates_score_is_correct_l583_583262

-- Definitions based on the given conditions
def Lizzie_score : ℕ := 4
def Nathalie_score : ℕ := Lizzie_score + 3
def Combined_score : ℕ := Lizzie_score + Nathalie_score
def Aimee_score : ℕ := 2 * Combined_score
def Total_score : ℕ := Lizzie_score + Nathalie_score + Aimee_score
def Whole_team_score : ℕ := 50
def Teammates_score : ℕ := Whole_team_score - Total_score

-- Proof statement
theorem teammates_score_is_correct : Teammates_score = 17 := by
  sorry

end teammates_score_is_correct_l583_583262


namespace dot_product_zero_not_implies_zero_vector_l583_583087

variables (a b : ℝ^3) -- Assuming 3-dimensional real vectors

theorem dot_product_zero_not_implies_zero_vector (h : a.dot b = 0) : a = 0 ∨ b = 0 :=
sorry

end dot_product_zero_not_implies_zero_vector_l583_583087


namespace percentage_increases_and_total_l583_583492

/-- Assume four visual ranges: initial, after first telescope,
    after second telescope, and after third telescope. -/
def initial_range : ℕ := 50
def first_telescope_range : ℕ := 150
def second_telescope_range : ℕ := 400
def third_telescope_range : ℕ := 750

/-- Define the formula for percentage increase. -/
def percentage_increase (original new : ℕ) : ℕ :=
  ((new - original) * 100) / original

noncomputable def percentage_increase_first : ℕ :=
  percentage_increase initial_range first_telescope_range

noncomputable def percentage_increase_second : ℕ :=
  percentage_increase first_telescope_range second_telescope_range

noncomputable def percentage_increase_third : ℕ :=
  percentage_increase second_telescope_range third_telescope_range

noncomputable def total_percentage_increase : ℕ :=
  percentage_increase initial_range third_telescope_range

theorem percentage_increases_and_total :
  percentage_increase_first = 200 ∧
  percentage_increase_second = 166.67 ∧
  percentage_increase_third = 87.5 ∧
  total_percentage_increase = 1400 :=
by
  sorry

end percentage_increases_and_total_l583_583492


namespace hyperbola_equation_l583_583530

-- Definitions based on the conditions:
def hyperbola (x y a b : ℝ) : Prop := (y^2 / a^2) - (x^2 / b^2) = 1

def point_on_hyperbola (a b : ℝ) : Prop := hyperbola 2 (-2) a b

def asymptotes (a b : ℝ) : Prop := a / b = (Real.sqrt 2) / 2

-- Prove the equation of the hyperbola
theorem hyperbola_equation :
  ∃ a b, a = Real.sqrt 2 ∧ b = 2 ∧ hyperbola y x (Real.sqrt 2) 2 :=
by
  -- Placeholder for the actual proof
  sorry

end hyperbola_equation_l583_583530


namespace crease_length_l583_583070

-- Definitions of points and triangle sides
variables (A B C M : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space M]

-- Conditions based on the problem statement
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Specific right triangle with sides 5, 12, and 13
def triangle_ABC : Prop := is_right_triangle 5 12 13

-- Assume coordinates for the points (for simplicity)
def point_A := (0 : ℝ, 0 : ℝ)
def point_B := (12 : ℝ, 0 : ℝ)
def point_C := (0 : ℝ, 5 : ℝ)
def point_M := (6 : ℝ, 0 : ℝ)

-- Distance function
def distance (p1 p2 : A) : ℝ := real.sqrt ((p1.fst - p2.fst) ^ 2 + (p1.snd - p2.snd) ^ 2)

-- Main theorem statement: prove the crease length is √61
theorem crease_length : triangle_ABC → distance point_C point_M = real.sqrt 61 :=
sorry

end crease_length_l583_583070


namespace find_n_in_arithmetic_sequence_l583_583281

noncomputable def arithmetic_sequence_n : ℕ :=
  sorry

theorem find_n_in_arithmetic_sequence (a : ℕ → ℕ) (d n : ℕ) :
  (a 3) + (a 4) = 10 → (a (n-3) + a (n-2)) = 30 → n * (a 1 + a n) / 2 = 100 → n = 10 :=
  sorry

end find_n_in_arithmetic_sequence_l583_583281


namespace rope_segment_equation_l583_583877

theorem rope_segment_equation (x : ℝ) (h1 : 2 - x > 0) :
  x^2 = 2 * (2 - x) :=
by
  sorry

end rope_segment_equation_l583_583877


namespace fraction_field_planted_l583_583526

-- Define the problem conditions
structure RightTriangle (leg1 leg2 hypotenuse : ℝ) : Prop :=
  (right_angle : ∃ (A B C : ℝ), A = 5 ∧ B = 12 ∧ hypotenuse = 13 ∧ A^2 + B^2 = hypotenuse^2)

structure SquarePatch (shortest_distance : ℝ) : Prop :=
  (distance_to_hypotenuse : shortest_distance = 3)

-- Define the statement
theorem fraction_field_planted (T : RightTriangle 5 12 13) (P : SquarePatch 3) : 
  ∃ (fraction : ℚ), fraction = 7 / 10 :=
by
  sorry

end fraction_field_planted_l583_583526


namespace problem_amc12a_2002_p12_l583_583971

theorem problem_amc12a_2002_p12 
  (a b : ℚ)
  (h : ∀ (x : ℚ), 0 < x → 
    (a / (2^x - 1) + b / (2^x + 3) = (3 * 2^x + 1) / ((2^x - 1) * (2^x + 3)))) :
  a - b = -1 :=
by
  sorry

end problem_amc12a_2002_p12_l583_583971


namespace proof_of_diagonals_and_angles_l583_583562

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def sum_of_internal_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem proof_of_diagonals_and_angles :
  let p_diagonals := number_of_diagonals 5
  let o_diagonals := number_of_diagonals 8
  let total_diagonals := p_diagonals + o_diagonals
  let p_internal_angles := sum_of_internal_angles 5
  let o_internal_angles := sum_of_internal_angles 8
  let total_internal_angles := p_internal_angles + o_internal_angles
  total_diagonals = 25 ∧ total_internal_angles = 1620 :=
by
  sorry

end proof_of_diagonals_and_angles_l583_583562


namespace derivative_of_f_l583_583897

noncomputable def f : ℝ → ℝ := λ x, 1 / x

theorem derivative_of_f : ∀ x : ℝ, x ≠ 0 → derivative f x = - (1 / (x^2)) :=
by
  sorry

end derivative_of_f_l583_583897


namespace remainder_x2023_l583_583901

theorem remainder_x2023 (x : ℤ) : 
  let dividend := x^2023 + 1
  let divisor := x^6 - x^4 + x^2 - 1
  let remainder := -x^7 + 1
  dividend % divisor = remainder :=
by
  sorry

end remainder_x2023_l583_583901


namespace ratio_of_Marleys_oranges_to_Louiss_oranges_l583_583331

variable (L_O : ℕ := 5) (S_A : ℕ := 7)

def Marley_has_three_times_as_many_apples_as_Samantha (M_A : ℕ) : Prop :=
  M_A = 3 * S_A

def Marley_has_a_total_of_thirty_one_fruits (M_O M_A : ℕ) : Prop :=
  M_O + M_A = 31

theorem ratio_of_Marleys_oranges_to_Louiss_oranges
  (M_O M_A : ℕ)
  (h1 : Marley_has_three_times_as_many_apples_as_Samantha M_A)
  (h2 : Marley_has_a_total_of_thirty_one_fruits M_O M_A) :
  M_O / L_O = 2 := 
sorry

end ratio_of_Marleys_oranges_to_Louiss_oranges_l583_583331


namespace correct_statements_count_l583_583325

variables (a b : Line) (α β : Plane)

def statement1 := (a ⊥ b ∧ a ∥ α) → b ∥ α
def statement2 := (a ∥ α ∧ α ⊥ β) → a ⊥ β
def statement3 := (a ⊥ β ∧ α ⊥ β) → a ∥ α
def statement4 := (a ⊥ b ∧ a ⊥ α ∧ b ⊥ β) → α ⊥ β

theorem correct_statements_count : 
  (¬ statement1 → true) ∧ (¬ statement2 → true) ∧ (¬ statement3 → true) ∧ statement4 → true := 
sorry

end correct_statements_count_l583_583325


namespace can_choose_skew_edges_to_cover_tetrahedron_l583_583581

noncomputable def skew_edge_spheres_cover_tetrahedron (A B C D : Point) : Prop :=
  ∃ (e1 e2 : Edge), 
    skew e1 e2 ∧ -- e1 and e2 are skew edges
    ∀ (p : Point), p ∈ tetrahedron_span {A, B, C, D} → 
      p ∈ sphere_diameter (edge_length e1) ∨ p ∈ sphere_diameter (edge_length e2)

theorem can_choose_skew_edges_to_cover_tetrahedron (A B C D : Point) : 
  skew_edge_spheres_cover_tetrahedron A B C D := sorry

end can_choose_skew_edges_to_cover_tetrahedron_l583_583581


namespace find_sum_of_p_q_r_s_l583_583316

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l583_583316


namespace concave_number_count_is_eight_l583_583842

def concave_number_count (s : Finset ℕ) := 
  ∑ a in s, ∑ b in s \ {a}, ∑ c in s \ {a, b}, if a > b ∧ b < c then 1 else 0

theorem concave_number_count_is_eight :
  concave_number_count ({1, 2, 3, 4} : Finset ℕ) = 8 :=
by
  sorry

end concave_number_count_is_eight_l583_583842


namespace part1_part2_part3_l583_583302

noncomputable def a_sequence (a1 d : ℕ) (n : ℕ) := a1 + d * (n - 1)
noncomputable def S (a1 d : ℕ) (n : ℕ) := n * (2 * a1 + (n - 1) * d) / 2

axiom a1 : ℕ
axiom d : ℕ
axiom h_d_nonzero : d ≠ 0
axiom h1 : a_sequence a1 d 3 + 3 * a_sequence a1 d 4 = S a1 d 5
axiom h2 : a1 * a_sequence a1 d 5 = S a1 d 4

noncomputable def b_sequence (b1 : ℕ) : ℕ → ℕ
| 1 => b1
| 2 => 3 * (b_sequence 1) + 2
| (n + 3) => 3 * (b_sequence (n + 2)) + 2^n

axiom b1 : ℕ
axiom b1_def : b1 = a1 - 1

theorem part1 : ∀ n : ℕ, a_sequence 2 2 n = 2 * n :=
sorry

theorem part2 (n : ℕ) : ∃ r : ℝ, ∀ k : ℕ, b_sequence b1 k / 2 ^ k + 1 = (3/2) ^ k :=
sorry

theorem part3 (n : ℕ) : (∑ k in finset.range n, (1 : ℝ) / b_sequence b1 (k + 1)) < 77 / 60 :=
sorry

end part1_part2_part3_l583_583302


namespace quadratic_eqn_vertex_point_l583_583527

noncomputable def quadratic_eqn (p q r x : ℝ) : ℝ := 
  p * x^2 + q * x + r

theorem quadratic_eqn_vertex_point 
  (p q r : ℝ) 
  (H1 : ∃ (a b : ℝ), quadratic_eqn p q r x = a * (x - 3) ^ 2 + b 
                       ∧ b = 4) 
  (H2 : quadratic_eqn p q r 1 = 2) : 
  p + q + r = 3 := 
begin 
  sorry 
end

end quadratic_eqn_vertex_point_l583_583527


namespace min_value_x_plus_one_over_x_plus_two_l583_583920

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1 / (x + 2) ∧ y ≥ 0 := 
sorry

end min_value_x_plus_one_over_x_plus_two_l583_583920


namespace last_third_speed_l583_583463

-- Definitions based on the conditions in the problem statement
def first_third_speed : ℝ := 80
def second_third_speed : ℝ := 30
def average_speed : ℝ := 45

-- Definition of the distance covered variable (non-zero to avoid division by zero)
variable (D : ℝ) (hD : D ≠ 0)

-- The unknown speed during the last third of the distance
noncomputable def V : ℝ := 
  D / ((D / 3 / first_third_speed) + (D / 3 / second_third_speed) + (D / 3 / average_speed))

-- The theorem to prove
theorem last_third_speed : V = 48 :=
by
  sorry

end last_third_speed_l583_583463


namespace num_true_propositions_l583_583617

-- Definitions of the vectors a and b and their properties as given in the condition
variables (a b : Type) [normed_group a] [normed_group b]
variables (v w : a) (x y : b)

-- The proposition p
def prop_p := v = w → ∥v∥ = ∥w∥

-- The contrapositive of the proposition p
def contrapositive_p := ∥v∥ ≠ ∥w∥ → v ≠ w

-- The converse of the proposition p
def converse_p := ∥v∥ = ∥w∥ → v = w

-- The inverse of the proposition p
def inverse_p := v ≠ w → ∥v∥ ≠ ∥w∥

-- The statement: Among prop_p and its contrapositive, converse, and inverse, the number of true propositions is 2.
theorem num_true_propositions : (prop_p v w → true) ∧ (contrapositive_p v w → true) ∧
  ¬(converse_p v w → true) ∧ ¬(inverse_p v w → true) :=
sorry

end num_true_propositions_l583_583617


namespace alphanumeric_puzzle_l583_583053

/-- Alphanumeric puzzle proof problem -/
theorem alphanumeric_puzzle
  (A B C D E F H J K L : Nat)
  (h1 : A * B = B)
  (h2 : B * C = 10 * A + C)
  (h3 : C * D = 10 * B + C)
  (h4 : D * E = 100 * C + H)
  (h5 : E * F = 10 * D + K)
  (h6 : F * H = 100 * C + J)
  (h7 : H * J = 10 * K + J)
  (h8 : J * K = E)
  (h9 : K * L = L)
  (h10 : A * L = L) :
  A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0 :=
sorry

end alphanumeric_puzzle_l583_583053


namespace x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l583_583802

theorem x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero :
  ∃ (x : ℝ), (x = 1) → (x^2 + x - 2 = 0) ∧ (¬ (∀ (y : ℝ), y^2 + y - 2 = 0 → y = 1)) := by
  sorry

end x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l583_583802


namespace math_problem_proof_l583_583616

noncomputable def parabola_through_A (p : ℝ) : Prop :=
  let C := λ (x y : ℝ), y^2 = -2 * p * x in
  C (-1) (-2)

noncomputable def equation_axis_symmetry : Prop :=
  let p := -2 in
  let eq_parabola := ∀ (x y : ℝ), y^2 = -4 * x in
  let axis_symmetry := λ x, x = -1 in
  eq_parabola ∧ axis_symmetry

noncomputable def length_segment_AB : Prop :=
  let focus := (-1, 0) in
  let slope := - real.sqrt 3 in
  let line_through_focus := λ x y, y = - real.sqrt 3 * (x + 1) in
  let parabola := λ (x y : ℝ), y^2 = -4 * x in
  let AB_length := 16 / 3 in
  let system := system_of_eq line_through_focus parabola in
  length_of_AB_segment system = AB_length

theorem math_problem_proof :
  parabola_through_A ∧ equation_axis_symmetry ∧ length_segment_AB :=
sorry

end math_problem_proof_l583_583616


namespace line_intersects_circle_l583_583402

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (x^2 + y^2 - 2*y = 0) ∧ (y - 1 = k * (x - 1)) :=
sorry

end line_intersects_circle_l583_583402


namespace find_angle_and_area_l583_583205

variables (A B C : ℝ) (a b c : ℝ)
hypothesis (h₁ : 2 * a * cos A = c * cos B + b * cos C)
hypothesis (h₂ : b ^ 2 + c ^ 2 = 7)
hypothesis (R : ℝ)
hypothesis (hR : R = 1)

theorem find_angle_and_area :
  (A = π / 3) ∧ (1 / 2 * b * c * sin A = sqrt 3) :=
by
  sorry

end find_angle_and_area_l583_583205


namespace angle_O1AO2_eq_half_diff_beta_gamma_l583_583670

theorem angle_O1AO2_eq_half_diff_beta_gamma 
    (A B C : Type) [inst : nonempty A] [inst' : nonempty B] [inst'' : nonempty C]
    (alpha beta gamma : ℝ) 
    (incenter circumcenter : A)
    (O1 O2 : A) 
    (angle_A : A)
    (h_triangle : angle_A = α ∧ angle_A = β ∧ angle_A = γ)
    (h_incenter : O1 = incenter)
    (h_circumcenter : O2 = circumcenter) :
    ∃ θ : ℝ, θ = (β - γ) / 2 ∨ θ = (γ - β) / 2 := 
by
  sorry

end angle_O1AO2_eq_half_diff_beta_gamma_l583_583670


namespace price_of_each_cupcake_l583_583739

variable (x : ℝ)

theorem price_of_each_cupcake (h : 50 * x + 40 * 0.5 = 2 * 40 + 20 * 2) : x = 2 := 
by 
  sorry

end price_of_each_cupcake_l583_583739


namespace each_dog_eats_per_day_l583_583523

theorem each_dog_eats_per_day : 
  (∀ (x : ℝ), 2 * x = 0.25 → x = 0.125) :=
by 
  intro x h,
  have h1 : x = 0.125 := by sorry,
  exact h1

end each_dog_eats_per_day_l583_583523


namespace image_of_center_l583_583115

-- Define the initial coordinates
def initial_coordinate : ℝ × ℝ := (-3, 4)

-- Function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to translate a point up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Definition of the final coordinate
noncomputable def final_coordinate : ℝ × ℝ :=
  translate_up (reflect_x initial_coordinate) 5

-- Theorem stating the final coordinate after transformations
theorem image_of_center : final_coordinate = (-3, 1) := by
  -- Proof is omitted
  sorry

end image_of_center_l583_583115


namespace solve_for_y_l583_583728

theorem solve_for_y (y : ℤ) : 5 * 5^y = 3125 → y = 4 :=
by 
  sorry

end solve_for_y_l583_583728


namespace exists_continuous_function_iff_nonneg_k_l583_583910

noncomputable def solve_k : Set ℝ := { k | ∃ (f : ℝ → ℝ), continuous f ∧ (∀ x : ℝ, f (f x) = k * x^9) }

theorem exists_continuous_function_iff_nonneg_k (k : ℝ) : 
  (∃ (f : ℝ → ℝ), continuous f ∧ (∀ x : ℝ, f (f x) = k * x^9)) ↔ k ≥ 0 :=
sorry

end exists_continuous_function_iff_nonneg_k_l583_583910


namespace problem_1_parallel_problem_2_range_l583_583625

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (1, 2 - x)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (1 + x, 2)

theorem problem_1_parallel (x : ℝ) : 
  vec_a x = (1, 2 - x) → vec_b x = (1 + x, 2) → 
  (∀ x, det ![
    [1, 2 - x],
    [1 + x, 2]
  ] = 0 → (x = 0 ∨ x = 1)) :=
  sorry

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
  
theorem problem_2_range : ∀ x ∈ set.Icc 0 2, 
    vec_a x = (1, 2 - x) →
    vec_b x = (1 + x, 2) →
    ∃ y ∈ set.Icc (-9 / 4) 0, 
      dot_product (vec_a x) (vec_a x - vec_b x) = y :=
  sorry

#align problem_1_parallel problem_1_parallel
#align problem_2_range problem_2_range

end problem_1_parallel_problem_2_range_l583_583625


namespace subtraction_is_addition_of_negatives_l583_583109

theorem subtraction_is_addition_of_negatives : (-1) - 3 = -4 := by
  sorry

end subtraction_is_addition_of_negatives_l583_583109


namespace area_of_shaded_region_l583_583882

theorem area_of_shaded_region (s : ℝ) (h1 : s = 3) :
  let area_octagon := 2 * (1 + real.sqrt 2) * s^2,
      r := s / 2,
      area_semicircle  := (1 / 2) * real.pi * r^2,
      total_area_semicircles := 8 * area_semicircle,
      area_shaded_region := area_octagon - total_area_semicircles
  in area_shaded_region = 18 * (1 + real.sqrt 2) - 9 * real.pi :=
by
  unfold_projs,
  sorry

end area_of_shaded_region_l583_583882


namespace candy_distribution_l583_583761

-- Define the problem conditions and theorem.

theorem candy_distribution (X : ℕ) (total_pieces : ℕ) (portions : ℕ) 
  (subsequent_more : ℕ) (h_total : total_pieces = 40) 
  (h_portions : portions = 4) 
  (h_subsequent : subsequent_more = 2) 
  (h_eq : X + (X + subsequent_more) + (X + subsequent_more * 2) + (X + subsequent_more * 3) = total_pieces) : 
  X = 7 := 
sorry

end candy_distribution_l583_583761


namespace angle_P_of_extended_sides_l583_583363

noncomputable def regular_pentagon_angle_sum : ℕ := 540

noncomputable def internal_angle_regular_pentagon (n : ℕ) (h : 5 = n) : ℕ :=
  regular_pentagon_angle_sum / n

def interior_angle_pentagon : ℕ := 108

theorem angle_P_of_extended_sides (ABCDE : Prop) (h1 : interior_angle_pentagon = 108)
  (P : Prop) (h3 : 72 + 72 = 144) : 180 - 144 = 36 := by 
  sorry

end angle_P_of_extended_sides_l583_583363


namespace gadget_no_reach_2011_l583_583063

theorem gadget_no_reach_2011 :
  let initial_state := [2, 0, 1, 0]
  let target_state := [2, 0, 1, 1]
  let valid_operations := λ s : List ℕ, 
    ([s.take 1, [1, 1], s.drop 3].join = s ∨ [s.take 1, [2, 2], s.drop 3].join = s ∨ [s.take 1, [0, 0], s.drop 3].join = s)
  let sum_list := λ l : List ℕ, list.foldl (+) 0 l 
  sum_list initial_state % 3 = 0 ∧ sum_list target_state % 3 ≠ 0 →
  ¬ ∃ (seq : List (List ℕ)), seq.head = initial_state ∧ seq.tail.seq.foldl valid_operations = target_state := 
by
  intros initial_state target_state valid_operations sum_list h
  have h := list.take 1 initial_state 
  have h := list.drop 3 initial_state   
  problem := sorry -- Proof goes here

end gadget_no_reach_2011_l583_583063


namespace forest_green_initial_yellow_l583_583097

-- Define the ratios and the required amount to add for conversion
def forest_green_ratio_blue_yellow : ℝ := 4 / 3
def verdant_green_ratio_yellow_blue : ℝ := 4 / 3
def yellow_paint_to_add : ℝ := 2.333333333333333

-- Theorem to prove the amount of yellow paint in original forest green mixture
theorem forest_green_initial_yellow :
  ∃ Y : ℝ,
    (∃ B : ℝ, B / Y = forest_green_ratio_blue_yellow) ∧
    ((Y + yellow_paint_to_add) / (4 / 3 * Y) = verdant_green_ratio_yellow_blue) ∧
    Y = 3 :=
begin
  sorry
end

end forest_green_initial_yellow_l583_583097


namespace sum_of_squares_of_midsegments_l583_583384

def sum_of_squares_of_midsegment_lengths (AC BD : ℝ) (angle : ℝ) : ℝ :=
  let a := AC / 2
  let b := BD / 2
  let cos_angle := Math.cos (Real.toRadians angle)
  in a^2 + b^2 - 2 * a * b * cos_angle

theorem sum_of_squares_of_midsegments :
  let AC := 3
  let BD := 4
  let angle := 75
  sum_of_squares_of_midsegment_lengths AC BD angle = 12.5 :=
by
  unfold sum_of_squares_of_midsegment_lengths
  sorry

end sum_of_squares_of_midsegments_l583_583384


namespace smallest_integer_condition_l583_583555

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l583_583555


namespace derivative_inequality_l583_583601

theorem derivative_inequality (f : ℝ → ℝ) (h : ∀ x : ℝ, f'' x > f x) (a : ℝ) (h_a : a > 0) : f a > exp a * f 0 :=
sorry

end derivative_inequality_l583_583601


namespace theo_needs_84_eggs_l583_583096

def customers_hour1 := 5
def customers_hour2 := 7
def customers_hour3 := 3
def customers_hour4 := 8

def eggs_per_omelette_3 := 3
def eggs_per_omelette_4 := 4

def total_eggs_needed : Nat :=
  (customers_hour1 * eggs_per_omelette_3) +
  (customers_hour2 * eggs_per_omelette_4) +
  (customers_hour3 * eggs_per_omelette_3) +
  (customers_hour4 * eggs_per_omelette_4)

theorem theo_needs_84_eggs : total_eggs_needed = 84 :=
by
  sorry

end theo_needs_84_eggs_l583_583096


namespace minimize_absolute_error_l583_583835

theorem minimize_absolute_error 
  (n : ℕ := 120)
  (d1 d2 : ℕ := 10)
  (d3 d4 : ℕ := 6)
  (small_diameter big_diameter : ℝ := 6.01)
  (another_diameter : ℝ := 6.11)
  (threshold diameter_a diameter_b : ℝ) : 
  (∃ (d a b : ℝ), 
    d = 6.09 ∧ 
    a = 6.06 ∧ 
    b = 6.15 ∧ 
    ∀ i < 60, ball_diameters[i] = small_diameter ∧ 
    ∀ 60 ≤ i < 120, ball_diameters[i] = another_diameter ∧ 
    (threshold < d) ∨ (threshold > d) ∨ 
    a = (small_diameter * d1 + big_diameter * (n - d1 - d3))/60 ∧ 
    b = (another_diameter * d4 + big_diameter * (n - d4 - 60))/60) := 
sorry

end minimize_absolute_error_l583_583835


namespace Sn_min_values_l583_583657

def Sn (d : ℝ) (n : ℕ) : ℝ := d * n * (n / 2 - 4.5)

theorem Sn_min_values (d : ℝ) (h : d > 0) : 
  ∃ n : ℕ, (Sn d n = Sn d 4 ∨ Sn d n = Sn d 5) ∧ (∀ m : ℕ, Sn d m ≥ Sn d n) :=
begin
  sorry
end

end Sn_min_values_l583_583657


namespace batsman_average_increase_l583_583812

theorem batsman_average_increase (A : ℕ) 
    (h1 : 15 * A + 64 = 19 * 16) 
    (h2 : 19 - A = 3) : 
    19 - A = 3 := 
sorry

end batsman_average_increase_l583_583812


namespace stone_general_path_count_l583_583481

def stone_general_moves (start : ℕ × ℕ) (end : ℕ × ℕ) (moves : ℕ) : ℕ :=
  let (sx, sy) := start
  let (ex, ey) := end
  if ey - sy ≠ moves then 0
  else
    let dx := ex - sx
    let nw_moves := (moves + dx) / 2
    let ne_moves := moves - nw_moves
    if (nw_moves + ne_moves ≠ moves) ∨ (nw_moves - ne_moves ≠ dx) ∨ (nw_moves < 0) ∨ (ne_moves < 0)
    then 0
    else (nat.choose moves nw_moves)

theorem stone_general_path_count :
  stone_general_moves (5, 1) (4, 8) 7 = 35 :=
by {
  -- using the conditions and definitions provided, we assert the expected result
  sorry
}

end stone_general_path_count_l583_583481


namespace triangle_area_ab_l583_583642

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : ∀ (x y : ℝ), a * x + b * y = 6) (harea : (1/2) * (6 / a) * (6 / b) = 6) : 
  a * b = 3 := 
by sorry

end triangle_area_ab_l583_583642


namespace max_intersection_points_l583_583425

theorem max_intersection_points : 
  ∀ (quadrilateral_sides hexagon_sides : ℕ), quadrilateral_sides = 4 → hexagon_sides = 6 → (quadrilateral_sides * hexagon_sides = 24) :=
by
  intros quadrilateral_sides hexagon_sides hq hh
  rw [hq, hh]
  exact eq.refl 24

end max_intersection_points_l583_583425


namespace question_1_question_2_l583_583450

open Real

variables (n : ℕ) (a b : Fin n → ℝ)

noncomputable def S := ∑ i, a i

axiom h_pos_n : 0 < n
axiom h_pos_a : ∀ i, 0 < a i
axiom h_pos_b : ∀ i, 0 < b i
axiom h_sum_eq : ∑ i, a i = ∑ i, b i

theorem question_1 :
  ∑ i, (a i)^2 / (b i + a i) = ∑ i, (b i)^2 / (b i + a i) :=
sorry

theorem question_2 :
  ∑ i, (a i)^2 / (b i + a i) ≥ S / 2 :=
sorry

end question_1_question_2_l583_583450


namespace john_total_beats_l583_583675

noncomputable def minutes_in_hour : ℕ := 60
noncomputable def hours_per_day : ℕ := 2
noncomputable def days_played : ℕ := 3
noncomputable def beats_per_minute : ℕ := 200

theorem john_total_beats :
  (beats_per_minute * hours_per_day * minutes_in_hour * days_played) = 72000 :=
by
  -- we will implement the proof here
  sorry

end john_total_beats_l583_583675


namespace inequality_proof_l583_583594

open Real

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^2 / (b + c + d) + b^2 / (c + d + a) +
   c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2 / 3) :=
by
  sorry

end inequality_proof_l583_583594


namespace find_m_if_f_is_odd_l583_583573

-- Define the given function f
def f (m : ℝ) (x : ℝ) : ℝ := m + 2 / (2^x + 1)

-- Define the property for f to be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem to prove that m = -1 if f is an odd function
theorem find_m_if_f_is_odd (m : ℝ) :
  is_odd_function (f m) → m = -1 :=
by
  sorry  -- Proof to be provided

end find_m_if_f_is_odd_l583_583573


namespace find_d_share_l583_583482

def money_distribution (a b c d : ℕ) (x : ℕ) := 
  a = 5 * x ∧ 
  b = 2 * x ∧ 
  c = 4 * x ∧ 
  d = 3 * x ∧ 
  (c = d + 500)

theorem find_d_share (a b c d x : ℕ) (h : money_distribution a b c d x) : d = 1500 :=
by
  --proof would go here
  sorry

end find_d_share_l583_583482


namespace area_triangle_l583_583933

noncomputable def A := (1, 3)
noncomputable def B := (3, 1)
noncomputable def l1 (x y : ℝ) := 3 * x - 2 * y + 3 = 0
noncomputable def l2 (x y : ℝ) := 2 * x - y + 2 = 0

def intersection : ℝ × ℝ := 
let (x, y) := (-1, 0) in
(x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def line (p1 p2 : ℝ × ℝ) (x y : ℝ) : ℝ := 
  (y - p1.2) / (p2.2 - p1.2) = (x - p1.1) / (p2.1 - p1.1)

def distance_from_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
abs (a * P.1 + b * P.2 + c) / real.sqrt (a^2 + b^2)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ := 
1 / 2 * distance A B * distance_from_point_to_line C 1 1 (-4)

theorem area_triangle : area_of_triangle A B intersection = 5 := by
sorry

end area_triangle_l583_583933


namespace box_area_ratio_l583_583514

theorem box_area_ratio 
  (l w h : ℝ)
  (V : l * w * h = 5184)
  (A1 : w * h = (1/2) * l * w)
  (A2 : l * h = 288):
  (l * w) / (l * h) = 3 / 2 := 
by
  sorry

end box_area_ratio_l583_583514


namespace det_matrix_power_l583_583634

theorem det_matrix_power (A : Matrix ℕ ℕ ℝ) (h : det A = 3) : det (A ^ 4) = 81 :=
by
  sorry

end det_matrix_power_l583_583634


namespace transform_ellipse_to_circle_l583_583767

theorem transform_ellipse_to_circle :
  (∀ x y : ℝ, (x'^{} = (sqrt 10 / 5) * x) ∧ (y'^{} = (sqrt 2 / 2) * y) → 
  (∀ x' y' : ℝ, (x^2 / 10 + y^2 / 8 = 1) → (x'^2 + y'^2 = 4))) := by
  sorry

end transform_ellipse_to_circle_l583_583767


namespace inequality_solution_l583_583373

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x + 3) < 0 ↔ x ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo (-3) 3 :=
by
  sorry

end inequality_solution_l583_583373


namespace power_division_l583_583028

theorem power_division (a b c : ℕ) (h : a = 2 ∧ b = 24 ∧ c = 8^3) : 2^24 / 8^3 = 32768 := by 
  have h_base_eq : 8 = 2^3 := rfl
  have h_power_eq : 8^3 = (2^3)^3 := by rw h_base_eq
  have h_simplify_power : (2^3)^3 = 2^9 := by rw pow_mul
  have h_replacement : 8^3 = 2^9 := by rw [h_power_eq, h_simplify_power]
  have h_divide : 2^24 / 2^9 = 2^(24 - 9) := by rw nat.div_pow
  have h_result : 2^(24 - 9) = 2^15 := by norm_num
  have h_final : 2^15 = 32768 := rfl
  sorry

end power_division_l583_583028


namespace intersection_of_sets_l583_583620

-- Define set M
def M : set ℝ := {x | x^2 - 3 * x = 0}

-- Define set N
def N : set ℝ := {x | x > -1}

-- Prove that M ∩ N = {0, 3}
theorem intersection_of_sets : M ∩ N = {0, 3} :=
sorry

end intersection_of_sets_l583_583620


namespace pipes_in_second_scenario_l583_583023

theorem pipes_in_second_scenario (C T : ℝ) (h1 : 3 * C * 8 = T) (h2 : ∀ x : ℝ, x * C * 12 = T → x = 2) :
  ∃ x : ℝ, (x * C * 12 = T) ∧ (x = 2) :=
by
  use 2
  split
  { -- proof of (2 * C * 12 = T)
    rw ←h1,
    have : 2 * C * 12 = 24 * C := by linarith,
    rw this,
    norm_num },
  { -- proof of (2 = 2)
    refl }

end pipes_in_second_scenario_l583_583023


namespace mean_transformation_l583_583183

theorem mean_transformation (x1 x2 x3 x4 : ℝ)
                            (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4)
                            (s2 : ℝ)
                            (h_var : s2 = (1 / 4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16)) :
                            (x1 + 2 + x2 + 2 + x3 + 2 + x4 + 2) / 4 = 4 :=
by
  sorry

end mean_transformation_l583_583183


namespace least_integer_l583_583535

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l583_583535


namespace crate_height_difference_l583_583769

open Real

-- Definitions from the conditions
def diameter : ℝ := 8
def rowsA : ℕ := 25
def rowsB : ℕ := 24

def height_crateA : ℝ := rowsA * diameter

def staggered_distance : ℝ := (sqrt 3 / 2) * diameter
def height_crateB : ℝ := rowsB * staggered_distance

noncomputable def difference_in_heights : ℝ := height_crateA - height_crateB

-- Assert the difference is the given value
theorem crate_height_difference :
  difference_in_heights = 200 - 96 * sqrt 3 := 
sorry

end crate_height_difference_l583_583769


namespace clearance_sale_total_earnings_l583_583819

-- Define the variables used in the problem
def total_jackets := 214
def price_before_noon := 31.95
def price_after_noon := 18.95
def jackets_sold_after_noon := 133

-- Calculate the total earnings
def total_earnings_from_clearance_sale : Prop :=
  (133 * 18.95 + (214 - 133) * 31.95) = 5107.30

-- State the theorem to be proven
theorem clearance_sale_total_earnings : total_earnings_from_clearance_sale :=
  by sorry

end clearance_sale_total_earnings_l583_583819


namespace black_squares_covered_by_trominoes_l583_583185

theorem black_squares_covered_by_trominoes (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (k : ℕ), k * k = (n + 1) / 2 ∧ n ≥ 7) ↔ n ≥ 7 :=
by
  sorry

end black_squares_covered_by_trominoes_l583_583185


namespace num_planes_in_rectangular_prism_l583_583106

theorem num_planes_in_rectangular_prism (a b c : ℕ) (ha : a = 4) (hb : b = 2) (hc : c = 3) : 
  let num_edges := 12 in
  let lengths := {4, 2, 3} in
  let num_possible_pairs := (num_edges * (num_edges - 1)) / 2 in
  let num_parallel_pairs := 3 * (nat.choose 4 2) in
  let num_intersecting_pairs := num_possible_pairs - num_parallel_pairs in
  num_parallel_pairs + num_intersecting_pairs = 42 := 
by
  intros
  sorry

end num_planes_in_rectangular_prism_l583_583106


namespace unique_positive_integer_solution_l583_583437

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 2652 := sorry

end unique_positive_integer_solution_l583_583437


namespace ant_distance_l583_583088

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem ant_distance :
  let A := (0, 0) in 
  let B := (4, 0) in
  distance A B = 4 :=
by
  sorry

end ant_distance_l583_583088


namespace distinct_polynomials_do_not_intersect_l583_583618

-- Definition of S
def S : set (ℝ → ℝ) := {f | 
  f = id ∨ (∃ g ∈ S, f = λ x, x * g x ∨ f = λ x, x + (1 - x) * g x)}

-- Property to prove: Any two distinct polynomials in S do not intersect in the interval 0 < x < 1.
theorem distinct_polynomials_do_not_intersect :
  ∀ f g ∈ S, f ≠ g → ∀ x ∈ Ioo 0 1, f x ≠ g x :=
by sorry

end distinct_polynomials_do_not_intersect_l583_583618


namespace seq_2007_l583_583665

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

theorem seq_2007 (a : ℕ → ℤ) (h : seq a) : a 2007 = 4 :=
by
  cases h with h1 h1_rest
  cases h1_rest with h2 hrec
  sorry

end seq_2007_l583_583665


namespace part_a_part_b_set_exists_part_c_set_minimum_l583_583181

variable (n : ℕ)
variable (T : Finset (Finset (ℕ × ℕ × ℕ)))

def is_positive_integer (x : ℕ) : Prop := x > 0

def all_distinct (x y z : ℕ) : Prop :=
  x ≠ y ∧ x ≠ z ∧ y ≠ z

def within_range (x y z : ℕ) (n : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 2 * n ∧
  1 ≤ y ∧ y ≤ 2 * n ∧
  1 ≤ z ∧ z ≤ 2 * n

theorem part_a (h1 : n > 1)
               (h2 : ∀ (x y z : ℕ), (x, y, z) ∈ T → all_distinct x y z ∧ within_range x y z n) :
  T.card = 2 * n * (2 * n - 1) * (2 * n - 2) :=
sorry

def connected_with (A : Finset (ℕ × ℕ)) (T : Finset (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (x y z : ℕ), (x, y, z) ∈ T → ({(x, y), (x, z), (y, z)} ∩ A).nonempty

theorem part_b_set_exists (h1 : n > 1)
                            (h2 : ∀ (x y z : ℕ), (x, y, z) ∈ T → all_distinct x y z ∧ within_range x y z n) :
  ∃ A : Finset (ℕ × ℕ), connected_with A T ∧ A.card = 2 * n * (n - 1) :=
sorry

theorem part_c_set_minimum (A : Finset (ℕ × ℕ))
                            (h1 : n > 1)
                            (h2 : ∀ (x y z : ℕ), (x, y, z) ∈ T → all_distinct x y z ∧ within_range x y z n)
                            (h3 : connected_with A T) :
  A.card ≥ 2 * n * (n - 1) :=
sorry

end part_a_part_b_set_exists_part_c_set_minimum_l583_583181


namespace range_of_g_number_of_zeros_exists_mu_for_lambda_l583_583961

def a (x : ℝ) : ℝ × ℝ := (Real.cos x + Real.sin x, 1)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x + Real.sin x, -1)
def g (x : ℝ) : ℝ := 4 * ((a x).1 * (b x).1 + (a x).2 * (b x).2)

theorem range_of_g : 
  ∃ (a b : ℝ), a = (4 : ℝ) ∧ b = (2 : ℝ) ∧ 
  (∀ x, x ∈ Set.Icc (Real.pi / 12) (Real.pi / 3) → g x ∈ Set.Icc b a) :=
sorry

theorem number_of_zeros : 
  ∃ (n : ℕ), n = 4033 ∧ 
  (∀ x, x ∈ Set.Icc 0 (2016 * Real.pi) → (g x = 0 ↔ ∃ k : ℤ, x = k * (Real.pi / 2))) :=
sorry

theorem exists_mu_for_lambda (λ : ℝ) (hλ : 0 < λ) : 
  ∃ (μ : ℝ), 0 < μ ∧ ∀ x, x < λ * μ → g x + x - 4 < 0 :=
sorry

end range_of_g_number_of_zeros_exists_mu_for_lambda_l583_583961


namespace pairwise_distance_sum_le_n_sq_l583_583711

variables {n : ℕ} (A : Fin n → ℝ^3)
def norm_sq (v : ℝ^3) : ℝ := v.dot v

noncomputable def pairwise_distance_sum (A : Fin n → ℝ^3) : ℝ :=
∑ i j in Finset.univ.filter (λ p : Fin n × Fin n, p.1 < p.2), (A i - A j).norm_sq

axiom radius_one {i : Fin n} : norm_sq (A i) = 1

theorem pairwise_distance_sum_le_n_sq (A : Fin n → ℝ^3) [∀ i, radius_one (A i)] :
  pairwise_distance_sum A ≤ n ^ 2 :=
sorry

end pairwise_distance_sum_le_n_sq_l583_583711


namespace quadratic_function_axis_of_symmetry_l583_583953

theorem quadratic_function_axis_of_symmetry :
  (∀ x : ℝ, let y := x^2 + 5*x + 4 in
   (∃ y : ℝ, y = 4 ∧ x = 0) ∧ 
   (∃ y : ℝ, y = -2 ∧ x = -3) ∧ 
   (∃ y : ℝ, y = 0 ∧ (x = -4 ∨ x = -1)) ∧ 
   (∃ y : ℝ, y = 4 ∧ x = 0)) →
  let a := 1 in
  let b := 5 in
  let c := 4 in
  let axis_of_symmetry := -b / (2 * a) in
  axis_of_symmetry = -5 / 2 :=
sorry

end quadratic_function_axis_of_symmetry_l583_583953


namespace find_ordinate_of_A_l583_583338

def parabola (x : ℝ) : ℝ := x^2

noncomputable def point_A_ord : ℝ :=
  let A := (0 : ℝ, a)
  sorry

theorem find_ordinate_of_A
  (a : ℝ)
  (A : ℝ × ℝ := (0, a))
  (h1 : A.2 > 0)
  (h2 : ∀ k > 0, ∃ M N : (ℝ × ℝ), 
    (parabola M.1 = M.2) ∧ (parabola N.1 = N.2) ∧
    ((M.2 = k * M.1 + a) ∧ (N.2 = k * N.1 + a)))
  (h3 : ∀ k1 k2 > 0, 
    let M1 := classical.some (h2 k1 k1.2),
        N1 := classical.some (h2 k1 k1.2),
        M2 := classical.some (h2 k2 k2.2),
        N2 := classical.some (h2 k2 k2.2)
    in angle M1 O N1 = angle M2 O N2) : 
    a = 1 := 
sorry

end find_ordinate_of_A_l583_583338


namespace initial_percent_is_5_l583_583054

theorem initial_percent_is_5 :
  ∃ x : ℝ, 0.03 = 0.60 * x ∧ x = 0.05 :=
begin
  use 0.05,
  split,
  {
    -- proving 0.03 = 0.60 * 0.05
    have h : 0.03 = 0.60 * 0.05,
    {
      norm_num,
    },
    exact h,
  },
  {
    -- trivial part, since x = 0.05 is already given
    norm_num,
  }
end

end initial_percent_is_5_l583_583054


namespace at_least_one_genuine_l583_583569

/-- Given 12 products, of which 10 are genuine and 2 are defective.
    If 3 products are randomly selected, then at least one of the selected products is a genuine product. -/
theorem at_least_one_genuine : 
  ∀ (products : Fin 12 → Prop), 
  (∃ n₁ n₂ : Fin 12, (n₁ ≠ n₂) ∧ 
                   (products n₁ = true) ∧ 
                   (products n₂ = true) ∧ 
                   (∃ n₁' n₂' : Fin 12, (n₁ ≠ n₁' ∧ n₂ ≠ n₂') ∧
                                         products n₁' = products n₂' = true ∧
                                         ∀ j : Fin 3, products j = true)) → 
  (∃ m : Fin 3, products m = true) :=
sorry

end at_least_one_genuine_l583_583569


namespace percentage_increase_in_savings_l583_583713

theorem percentage_increase_in_savings (I : ℝ) (hI : 0 < I) :
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  ((S_new - S) / S) * 100 = 50 :=
by
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  sorry

end percentage_increase_in_savings_l583_583713


namespace max_value_x_plus_y_l583_583924

theorem max_value_x_plus_y (x y : ℝ) (h : (x-2)^2 / 4 + (y-1)^2 = 1) : 
  x + y ≤ 3 + Real.sqrt 5 :=
begin
  sorry
end

end max_value_x_plus_y_l583_583924


namespace median_is_91_l583_583405

noncomputable def median_of_set (S : Set ℝ) (card_S : S.card = 9) (mean_S : (S.toList.sum) / 9 = 202)
  (mean_smallest_five : ∀ l : List ℝ, l.length = 5 → l.to_finset ⊆ S → (l.sum) / 5 = 100)
  (mean_largest_five : ∀ l : List ℝ, l.length = 5 → l.to_finset ⊆ S → (l.sum) / 5 = 300) : ℝ :=
  let sorted_S := S.toList.sorted (≤)
  in sorted_S.nth_le 4 sorry

theorem median_is_91 (S : Set ℝ) (h1 : S.card = 9) (h2 : (S.toList.sum) / 9 = 202)
  (h3 : ∀ l : List ℝ, l.length = 5 → l.to_finset ⊆ S → (l.sum) / 5 = 100)
  (h4 : ∀ l : List ℝ, l.length = 5 → l.to_finset ⊆ S → (l.sum) / 5 = 300) :
  median_of_set S h1 h2 h3 h4 = 91 :=
  sorry

end median_is_91_l583_583405


namespace quadratic_inequality_solution_l583_583170

theorem quadratic_inequality_solution (x : ℝ) : 
  (2 * x^2 - 6 * x - 56 > 0) ↔ (x ∈ Set.Ioo (−∞ : ℝ) (-4 : ℝ) ∪ Set.Ioo (7 : ℝ) (∞ : ℝ)) :=
by
  sorry

end quadratic_inequality_solution_l583_583170


namespace find_least_positive_integer_l583_583547

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l583_583547


namespace trigonometric_inequality_l583_583308

theorem trigonometric_inequality (a b : ℝ) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  5 / Real.cos a ^ 2 + 5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2) ≥ 27 * Real.cos a + 36 * Real.sin a :=
sorry

end trigonometric_inequality_l583_583308


namespace ceil_floor_subtraction_l583_583525

theorem ceil_floor_subtraction :
  ⌈(7:ℝ) / 3⌉ + ⌊- (7:ℝ) / 3⌋ - 3 = -3 := 
by
  sorry   -- Placeholder for the proof

end ceil_floor_subtraction_l583_583525


namespace volleyball_team_points_l583_583264

theorem volleyball_team_points (lizzie_points nathalie_points aimee_points teammate_points total_points : ℕ)
  (h1 : lizzie_points = 4)
  (h2 : nathalie_points = lizzie_points + 3)
  (h3 : aimee_points = 2 * (lizzie_points + nathalie_points))
  (h4 : total_points = 50)
  (h5 : total_points = lizzie_points + nathalie_points + aimee_points + teammate_points) :
  teammate_points = 17 :=
begin
  sorry
end

end volleyball_team_points_l583_583264


namespace geometric_sequence_seventh_term_l583_583822

theorem geometric_sequence_seventh_term (a r : ℕ) (h₁ : a = 6) (h₂ : a * r^4 = 486) : a * r^6 = 4374 :=
by
  -- The proof is not required, hence we use sorry.
  sorry

end geometric_sequence_seventh_term_l583_583822


namespace grasshopper_return_to_origin_l583_583065

def grasshopper_can_return : Prop :=
  ∃ n m : ℕ, (1 + 2 + ... + n) * (-2) = 0 ∧ (1 + 2 + ... + m) * (-2) = 0

theorem grasshopper_return_to_origin : grasshopper_can_return :=
sorry

end grasshopper_return_to_origin_l583_583065


namespace inradius_of_right_triangle_l583_583579

/-- Theorem: Calculate the inradius of a right triangle -/
theorem inradius_of_right_triangle 
  (a b c : ℝ) (h_tri : a = 9 ∧ b = 40 ∧ c = 41) 
  (h_right : a^2 + b^2 = c^2) : 
  ∃ r : ℝ, r = 4 :=
by
  let A := (1 / 2) * a * b
  let s := (a + b + c) / 2
  have hA : A = 180 := sorry
  have hs : s = 45 := sorry
  let r := A / s
  have hr : r = 4 := by 
    calc r = 180 / 45 : by sorry
      ... = 4 : by norm_num
  use r
  exact hr

end inradius_of_right_triangle_l583_583579


namespace all_nat_as_sum_of_types_l583_583906

-- Definitions of the geometric progressions
def prog (n : ℕ) : List ℕ :=
  List.map (λ i => (n+2)^i) (List.range 100)  -- Assume 100 terms for simplicity

noncomputable def nth_type (n : ℕ) : Set ℕ :=
  {0} ∪ {a | ∃ k ∈ prog (n-1), a = k} ∪ {a | ∃ (k ∈ prog (n-1)) (l ∈ prog (n-1)), k ≠ l ∧ a = k + l}

def isSumOfTypes (x : ℕ) : Prop :=
  ∃ a b c, a ∈ nth_type 1 ∧ b ∈ nth_type 2 ∧ c ∈ nth_type 3 ∧ x = a + b + c

theorem all_nat_as_sum_of_types :
  ∀ n : ℕ, isSumOfTypes n :=
by
  intro n
  sorry

end all_nat_as_sum_of_types_l583_583906


namespace fractional_g_asymptotic_l583_583690

-- Definition of g(n) as given in the problem
def g (n : ℕ) : ℝ := Real.log10 (Nat.choose (2 * n) n)

-- The main theorem statement, defining what needs to be proved
theorem fractional_g_asymptotic (n : ℕ) (h : n > 0) :
  g(n) / Real.log10 3 = 2 * n * (Real.log10 2 / Real.log10 3) :=
by
  sorry

end fractional_g_asymptotic_l583_583690


namespace maximum_regular_hours_is_40_l583_583829

-- Definitions based on conditions
def regular_pay_per_hour := 3
def overtime_pay_per_hour := 6
def total_payment_received := 168
def overtime_hours := 8
def overtime_earnings := overtime_hours * overtime_pay_per_hour
def regular_earnings := total_payment_received - overtime_earnings
def maximum_regular_hours := regular_earnings / regular_pay_per_hour

-- Lean theorem statement corresponding to the proof problem
theorem maximum_regular_hours_is_40 : maximum_regular_hours = 40 := by
  sorry

end maximum_regular_hours_is_40_l583_583829


namespace range_of_a_l583_583257

theorem range_of_a (x y a : ℝ) (h1 : 3 * x + y = a + 1) (h2 : x + 3 * y = 3) (h3 : x + y > 5) : a > 16 := 
sorry 

end range_of_a_l583_583257


namespace math_proof_problem_l583_583930

def ellipse_equation (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2) / (a^2) + (y^2) / (b^2) = 1

def right_focus_distance_condition (c : ℝ) : Prop :=
  |c + 3 * sqrt 2| / sqrt 2 = 5

def major_minor_endpoint_distance_condition (a b : ℝ) : Prop :=
  a^2 + b^2 = 10 ∧ a^2 - b^2 = 8

def parametric_line_condition (m t α : ℝ) (x y : ℝ) : Prop :=
  x = m + t * cos α ∧ y = t * sin α

def intersection_condition (m : ℝ) : Prop :=
  let t := ∀{α x y}, parametric_line_condition m t α x y →
  ∃ t1 t2 : ℝ, t^2 * (cos α)^2 + 9 * (sin α)^2 + 2 * m * cos α * t + m^2 - 9 = 0

theorem math_proof_problem :
  ∃ a b : ℝ, ellipse_equation a b x y ∧
  right_focus_distance_condition (2 * sqrt 2) ∧ 
  major_minor_endpoint_distance_condition a b ∧
  (a = 3 ∧ b = 1) ∧ 
  ∃ Qx : ℝ, ∃ t1 t2 : ℕ, 
  intersection_condition Qx ∧ 
  Qx = 6 * sqrt 5 / 5 ∧  
  (1 / (Qx + t1)^2 + 1 / (Qx + t2)^2 = 10) :=
by
  sorry

end math_proof_problem_l583_583930


namespace zoey_holidays_in_a_year_l583_583440

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end zoey_holidays_in_a_year_l583_583440


namespace max_ice_cream_servings_l583_583260

def days_in_february : ℕ := 28

def ice_cream_servings_per_day (date : ℕ) (day_of_week : String) : ℕ :=
  if (date % 2 = 0 ∧ (day_of_week = "Wednesday" ∨ day_of_week = "Thursday")) then 7
  else if (date % 2 = 1 ∧ (day_of_week = "Monday" ∨ day_of_week = "Tuesday")) then 3
  else if (day_of_week = "Friday") then date
  else 0

theorem max_ice_cream_servings :
  ∀ (days_in_feb : ℕ) (servings_func : ℕ → String → ℕ),
    days_in_feb = 28 →
    servings_func = ice_cream_servings_per_day →
    (Σ (date : ℕ) (h : 1 ≤ date ∧ date ≤ days_in_feb), servings_func date (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].nth! ((date - 1) % 7))) = 110 :=
by
  intros days_in_feb servings_func h1 h2
  sorry

end max_ice_cream_servings_l583_583260


namespace proof_problem_l583_583610

variable {𝕜 : Type*} [LinearOrderedField 𝕜]
variable {f : 𝕜 → 𝕜}
variable {k : 𝕜}

-- Conditions
hypothesis h1 : f 0 = -1
hypothesis h2 : ∀ x, deriv f x > k
hypothesis h3 : k > 1

-- Statements to be proved
theorem proof_problem :
  f (1 / k) > (1 / k) - 1 ∧ 
  f (1 / (k - 1)) > 1 / (k - 1) ∧ 
  f (1 / k) < f (1 / (k - 1)) :=
sorry

end proof_problem_l583_583610


namespace jason_fish_count_ninth_day_l583_583673

def fish_growth_day1 := 8 * 3
def fish_growth_day2 := fish_growth_day1 * 3
def fish_growth_day3 := fish_growth_day2 * 3
def fish_day4_removed := 2 / 5 * fish_growth_day3
def fish_after_day4 := fish_growth_day3 - fish_day4_removed
def fish_growth_day5 := fish_after_day4 * 3
def fish_growth_day6 := fish_growth_day5 * 3
def fish_day6_removed := 3 / 7 * fish_growth_day6
def fish_after_day6 := fish_growth_day6 - fish_day6_removed
def fish_growth_day7 := fish_after_day6 * 3
def fish_growth_day8 := fish_growth_day7 * 3
def fish_growth_day9 := fish_growth_day8 * 3
def fish_final := fish_growth_day9 + 20

theorem jason_fish_count_ninth_day : fish_final = 18083 :=
by
  -- proof steps will go here
  sorry

end jason_fish_count_ninth_day_l583_583673


namespace no_solution_ineq_system_l583_583438

def inequality_system (x : ℝ) : Prop :=
  (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
  (x + 9 / 2 > x / 8) ∧
  (11 / 3 - x / 6 < (34 - 3 * x) / 5)

theorem no_solution_ineq_system : ¬ ∃ x : ℝ, inequality_system x :=
  sorry

end no_solution_ineq_system_l583_583438


namespace ratio_of_average_speeds_l583_583443

theorem ratio_of_average_speeds
    (time_eddy : ℝ) (distance_eddy : ℝ)
    (time_freddy : ℝ) (distance_freddy : ℝ) :
  time_eddy = 3 ∧ distance_eddy = 600 ∧ time_freddy = 4 ∧ distance_freddy = 460 →
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 :=
by
  sorry

end ratio_of_average_speeds_l583_583443


namespace correct_answers_l583_583669

noncomputable def given_problem_conditions (A B C D : Point) : Prop :=
  let angle_B := 50
  let side_BC := 3
  let angle_ADC := 130
  let side_AD := real.sqrt 3
  let is_orthocenter := Orthocenter A B C D
  angle B A C = angle_B ∧
  distance B C = side_BC ∧
  angle A D C = angle_ADC ∧
  distance A D = side_AD ∧ 
  is_orthocenter

theorem correct_answers (A B C D : Point) (H : Altitude A B C) :
  given_problem_conditions A B C D →
  (angle D B C = 90) ∧ (angle C B H = 20) := 
by 
  intros
  sorry

end correct_answers_l583_583669


namespace cody_candy_total_l583_583503

theorem cody_candy_total
  (C_c : ℕ) (C_m : ℕ) (P_b : ℕ)
  (h1 : C_c = 7) (h2 : C_m = 3) (h3 : P_b = 8) :
  (C_c + C_m) * P_b = 80 :=
by
  sorry

end cody_candy_total_l583_583503


namespace jared_total_distance_l583_583292

-- Define the conditions with appropriate types and constraints
def average_speed_initial : ℝ := 22
def distance_fraction_initial : ℝ := 2 / 3
def reduced_speed : ℝ := 10
def distance_fraction_reduced : ℝ := 1 / 3
def total_time_minutes : ℝ := 36
def total_time_hours : ℝ := total_time_minutes / 60

-- Define the problem statement to prove
theorem jared_total_distance :
  ∀ (d : ℝ), 
  ((distance_fraction_initial * d) / average_speed_initial +
   (distance_fraction_reduced * d) / reduced_speed = total_time_hours) →
  abs (d - 4.6) ≤ 0.05 :=
by
  sorry

end jared_total_distance_l583_583292


namespace shaded_areas_equal_l583_583880

def square_area (side_length : ℝ) : ℝ := side_length ^ 2
def shaded_area_I (side_length : ℝ) : ℝ := (1 / 4) * square_area side_length
def shaded_area_II (side_length : ℝ) : ℝ := (1 / 4) * square_area side_length
def shaded_area_III (side_length : ℝ) : ℝ := (1 / 4) * square_area side_length

theorem shaded_areas_equal (side_length : ℝ) (h_pos : side_length > 0) :
  shaded_area_I side_length = shaded_area_II side_length ∧
  shaded_area_II side_length = shaded_area_III side_length :=
by
  sorry

end shaded_areas_equal_l583_583880


namespace general_term_and_minimum_value_of_sum_sequence_l583_583943

theorem general_term_and_minimum_value_of_sum_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 4 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (∀ n, n > 1 → a n = 2 * n - 5) ∧ (a 1 = -3) ∧ (∀ m, m > 1 → S m ≥ -4) :=
by {
  intros hS ha,
  split,
  { intros n hn,
    calc a n = S n - S (n - 1) : ha n
    ... = (n^2 - 4 * n) - ((n - 1)^2 - 4 * (n - 1)) : by { rw hS, rw hS (n - 1) }
    ... = 2 * n - 5 : by ring },
  split,
  { calc a 1 = S 1 - S 0 : ha 1
    ... = (1^2 - 4 * 1) - (0^2 - 4 * 0) : by { rw hS, rw hS 0 }
    ... = -3 : by ring },
  { intros m hm,
    calc S m = (m - 2)^2 - 4 : by { rw hS, ring }
    ... ≥ -4 : by { apply sub_nonneg_of_le, norm_num } }
}

end general_term_and_minimum_value_of_sum_sequence_l583_583943


namespace vertices_of_equilateral_triangle_l583_583401

noncomputable def a : ℝ := 52 / 3
noncomputable def b : ℝ := -13 / 3 - 15 * Real.sqrt 3 / 2

theorem vertices_of_equilateral_triangle (a b : ℝ)
  (h₀ : (0, 0) = (0, 0))
  (h₁ : (a, 15) = (52 / 3, 15))
  (h₂ : (b, 41) = (-13 / 3 - 15 * Real.sqrt 3 / 2, 41)) :
  a * b = -676 / 9 := 
by
  sorry

end vertices_of_equilateral_triangle_l583_583401


namespace gcd_7_8_fact_l583_583160

-- Define factorial function in lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the GCD function
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define specific factorial values
def f7 := fact 7
def f8 := fact 8

-- Theorem stating the gcd of 7! and 8!
theorem gcd_7_8_fact : gcd f7 f8 = 5040 := by
  sorry

end gcd_7_8_fact_l583_583160


namespace find_x_when_water_added_l583_583420

variable (m x : ℝ)

theorem find_x_when_water_added 
  (h1 : m > 25)
  (h2 : (m * m / 100) = ((m - 15) / 100) * (m + x)) :
  x = 15 * m / (m - 15) :=
sorry

end find_x_when_water_added_l583_583420


namespace payment_for_extra_chore_is_two_l583_583881

variables (chores_per_week : ℕ) (chores_per_day : ℕ) (days : ℕ)
variables (earnings : ℝ) (total_chores : ℕ)

-- Conditions from part (a)
def edmund_chores_per_week : ℕ := 12
def edmund_chores_per_day : ℕ := 4
def duration_in_days : ℕ := 14
def total_earnings : ℝ := 64

-- The actual number of extra chores
def total_chores_done : ℕ := edmund_chores_per_day * duration_in_days
def normal_chores_in_two_weeks : ℕ := edmund_chores_per_week * (duration_in_days / 7)
def extra_chores : ℕ := total_chores_done - normal_chores_in_two_weeks
def payment_per_extra_chore : ℝ := total_earnings / extra_chores

-- Proof statement: Edmund's parents pay him $2 per extra chore
theorem payment_for_extra_chore_is_two :
  edmund_chores_per_week = chores_per_week ∧
  edmund_chores_per_day = chores_per_day ∧
  duration_in_days = days ∧
  total_earnings = earnings ∧
  total_chores_done = total_chores →
  payment_per_extra_chore = 2 := by sorry

end payment_for_extra_chore_is_two_l583_583881


namespace find_n_l583_583190

open Real

theorem find_n (x n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -1) 
  (h2 : log 10 (sin x - cos x) = (1/2) * (log 10 n - 1)) : 
  n = 8 := 
by 
  sorry

end find_n_l583_583190


namespace prime_power_inequality_l583_583050

theorem prime_power_inequality (n k : ℕ) (p_2k : ℕ) (prime_2k : nat.prime p_2k) :
  (∀ x, x = 2 * k → x.th_prime = p_2k) → n < p_2k ^ k := 
by
  intro h
  sorry

end prime_power_inequality_l583_583050


namespace range_of_a_l583_583251

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (exp x1 - 2*x1 - a = 0) ∧ (exp x2 - 2*x2 - a = 0)) ↔ (2 - 2 * Real.log 2 < a) :=
sorry

end range_of_a_l583_583251


namespace cone_volume_l583_583253

noncomputable def volume_of_cone_from_lateral_surface (radius_semicircle : ℝ) 
  (circumference_base : ℝ := 2 * radius_semicircle * Real.pi) 
  (radius_base : ℝ := circumference_base / (2 * Real.pi)) 
  (height_cone : ℝ := Real.sqrt ((radius_semicircle:ℝ) ^ 2 - (radius_base:ℝ) ^ 2)) : ℝ := 
  (1 / 3) * Real.pi * (radius_base ^ 2) * height_cone

theorem cone_volume (h_semicircle : 2 = 2) : volume_of_cone_from_lateral_surface 2 = (Real.sqrt 3) / 3 * Real.pi := 
by
  -- Importing Real.sqrt and Real.pi to bring them into scope
  sorry

end cone_volume_l583_583253


namespace ratio_of_cats_to_dogs_sold_l583_583493

theorem ratio_of_cats_to_dogs_sold (cats dogs : ℕ) (h1 : cats = 16) (h2 : dogs = 8) :
  (cats : ℚ) / dogs = 2 / 1 :=
by
  sorry

end ratio_of_cats_to_dogs_sold_l583_583493


namespace positional_relationship_l583_583246

-- Define parallelism between a line and another line
def line_parallel_line (m n : Prop) : Prop := m ∥ n

-- Define parallelism between a line and a plane
def line_parallel_plane (m : Prop) (α : Prop) : Prop := m ∥ α

-- Define when a line is a subset of a plane
def line_in_plane (n : Prop) (α : Prop) : Prop := n ⊂ α

-- The given conditions
variable (m n : Prop) (α : Prop)
variable (h1 : line_parallel_line m n)
variable (h2 : line_parallel_plane m α)

-- The positional relationship to be proven
theorem positional_relationship : (line_parallel_plane n α) ∨ (line_in_plane n α) :=
sorry

end positional_relationship_l583_583246


namespace unique_eigenvalue_of_matrix_l583_583141

theorem unique_eigenvalue_of_matrix :
  ∃ (w : ℝ × ℝ), w ≠ (0, 0) ∧
  (∀ k : ℝ, (∃ (w : ℝ × ℝ), w ≠ (0, 0) ∧
   (\begin{pmatrix} 2 & 5 \\ 3 & 4 \end{pmatrix} • w.1, \begin{pmatrix} 2 & 5 \\ 3 & 4 \end{pmatrix} • w.2) = (k * w.1, k * w.2)) →
   k = 7) :=
sorry

end unique_eigenvalue_of_matrix_l583_583141


namespace combined_average_score_l583_583334

variable (M1 M2 : ℝ)
variable (m1 m2 : ℝ)
variable (ratio : ℝ)

theorem combined_average_score :
  M1 = 88 ∧ M2 = 75 ∧ ratio = 2/3 ∧ m1 = ratio * m2 →
  (M1 * (m1 / m2) + M2) / (1 + (m1 / m2)) = 80 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  rw [h1, h2, h3, h4]
  sorry

end combined_average_score_l583_583334


namespace smallest_integer_condition_l583_583556

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l583_583556


namespace min_y_value_l583_583435

-- Noncomputable to denote that we are not computing a specific value here
noncomputable def y (x : ℝ) : ℝ :=
  ∑ i in finset.range 2010, |x - (i + 1)|

theorem min_y_value (x : ℝ) : 1005 ≤ x ∧ x ≤ 1006 → y x = ∑ i in finset.range 2010, |(1005 : ℝ) - (i + 1)| := by
  sorry

end min_y_value_l583_583435


namespace incorrect_option_A_incorrect_option_D_l583_583244

variable (x y : ℝ)
axiom h : x / y = 2 / 5

-- Prove that invalid option A: x + 3y / 2y != 13 / 10
theorem incorrect_option_A : (x + 3 * y) / (2 * y) ≠ 13 / 10 :=
by
  intro h1
  have h2 : (x/y) + 3 = 16 / 5 := by
    calc
      (x / y) + 3 = (2 / 5) + 3       : by rw [h]
      ...     = 13 / 10           : sorry
  have h3 : 16 / 5 = 13 / 10 := sorry
  contradiction

-- Prove that invalid option D: 2y - x / 3y != 7 / 15
theorem incorrect_option_D : (2 * y - x) / (3 * y) ≠ 7 / 15 :=
by
  intro h4
  have h5 : (2 - (x / y)) / 3 = 8 / 15 := by
    calc
      (2 - (x / y)) / 3 = (2 - (2 / 5)) / 3 : by rw [h]
      ...          = (10 / 5 - 2 / 5) / 3 : sorry
  have h6 : 8 / 15 = 7 / 15 := sorry
  contradiction

end incorrect_option_A_incorrect_option_D_l583_583244


namespace fraction_of_shaded_area_l583_583389

theorem fraction_of_shaded_area
  (total_smaller_rectangles : ℕ)
  (shaded_smaller_rectangles : ℕ)
  (h1 : total_smaller_rectangles = 18)
  (h2 : shaded_smaller_rectangles = 4) :
  (shaded_smaller_rectangles : ℚ) / total_smaller_rectangles = 1 / 4 := 
sorry

end fraction_of_shaded_area_l583_583389


namespace find_eccentricity_of_ellipse_l583_583935

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  sorry

theorem find_eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0)
  (C : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (F1 F2 : ℝ) (A : ℝ × ℝ) (P : ℝ × ℝ)
  (hA : A = (-a, 0)) 
  (k : ℝ := √3 / 6)
  (line_eq : ∃ x, P.2 = k * (x + a))
  (isosceles : ∃ c, P.1 = 2 * c ∧ P.2 = √3 * c ∧ c = a / 4)
  (angle : ∃ θ : ℝ, θ = 120 ∧ ∠F1 F2 P = θ) :
  eccentricity_of_ellipse a b h = 1/4 := 
begin
  sorry
end

end find_eccentricity_of_ellipse_l583_583935


namespace minimal_number_of_weights_l583_583343

theorem minimal_number_of_weights (N : ℕ) :
  (∀ (weights : fin N → ℝ), 
    (∀ i j, abs (weights i - weights j) ≤ 1.25 * min (weights i) (weights j)) ∧
    (∃ (groups_pete : fin 10 → finset (fin N)), ∀ i, (finset.sum (groups_pete i) (λ x, weights x)) = k / 10) ∧
    (∃ (groups_vasya : fin 11 → finset (fin N)), ∀ i, (finset.sum (groups_vasya i) (λ x, weights x)) = k / 11)
  ) → N = 50 := by
  sorry

end minimal_number_of_weights_l583_583343


namespace product_floor_ceil_l583_583134

theorem product_floor_ceil :
  (Int.floor (-3.5) * Int.ceil (3.5) *
   Int.floor (-2.5) * Int.ceil (2.5) *
   Int.floor (-1.5) * Int.ceil (1.5)) = -576 := 
by
  sorry

end product_floor_ceil_l583_583134


namespace profit_correct_l583_583358

-- Conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def sets : ℕ := 500

-- Definitions used in the problem
def manufacturing_cost : ℕ := initial_outlay + (sets * cost_per_set)
def revenue : ℕ := sets * selling_price_per_set
def profit : ℕ := revenue - manufacturing_cost

-- The theorem statement
theorem profit_correct : profit = 5000 := by
  sorry

end profit_correct_l583_583358


namespace solution_set_l583_583208

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume f is some real function (The existence of f is granted by the problem statement)

-- Given conditions
axiom f_at_1 : f 1 = 1
axiom f'_lt_half : ∀ x : ℝ, deriv f x < 1 / 2

-- Problem statement
theorem solution_set (x : ℝ) : f (x ^ 2) < 1 / 2 * x ^ 2 + 1 / 2 ↔ x < -1 ∨ 1 < x :=
by sorry

end solution_set_l583_583208


namespace total_seashells_l583_583333

theorem total_seashells (mary_seashells : ℕ) (jessica_seashells : ℕ)
  (kevin_seashells : ℕ) (laura_seashells : ℕ)
  (h1 : mary_seashells = 18)
  (h2 : jessica_seashells = 41)
  (h3 : kevin_seashells = 3 * mary_seashells)
  (h4 : laura_seashells = (jessica_seashells / 2).toNat) :
  mary_seashells + jessica_seashells + kevin_seashells + laura_seashells = 134 :=
by
  -- Proof will go here
  sorry

end total_seashells_l583_583333


namespace axis_of_symmetry_center_of_symmetry_decreasing_intervals_range_of_function_l583_583948
open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos (2 * x - π / 4)

theorem axis_of_symmetry : ∀ (k : ℤ), ∃ x : ℝ, x = π / 8 + k * π / 2 :=
sorry

theorem center_of_symmetry : ∀ (k : ℤ), ∃ x : ℝ, (x, 0) = (3 * π / 8 + k * π / 2, 0) :=
sorry

theorem decreasing_intervals (k : ℤ) :
  ∀ x : ℝ, k * π + π / 8 ≤ x ∧ x ≤ k * π + 5 * π / 8 → f x ≤ f (k * π + π / 8) ∧ f x ≥ f (k * π + 5 * π / 8) :=
sorry

theorem range_of_function :
  ∀ x : ℝ, -π / 8 ≤ x ∧ x ≤ π / 2 → -√2 ≤ f x ∧ f x ≤ 2 :=
sorry

end axis_of_symmetry_center_of_symmetry_decreasing_intervals_range_of_function_l583_583948


namespace miles_driven_each_day_l583_583379

-- Definition of the given conditions
def total_miles : ℝ := 1250
def number_of_days : ℝ := 5.0

-- The statement to be proved
theorem miles_driven_each_day :
  total_miles / number_of_days = 250 :=
by
  sorry

end miles_driven_each_day_l583_583379


namespace fill_time_is_13_seconds_l583_583456

-- Define the given conditions as constants
def flow_rate_in (t : ℝ) : ℝ := 24 * t -- 24 gallons/second
def leak_rate (t : ℝ) : ℝ := 4 * t -- 4 gallons/second
def basin_capacity : ℝ := 260 -- 260 gallons

-- Main theorem to be proven
theorem fill_time_is_13_seconds : 
  ∀ t : ℝ, (flow_rate_in t - leak_rate t) * (13) = basin_capacity := 
sorry

end fill_time_is_13_seconds_l583_583456


namespace roots_ratio_quadratic_l583_583169

theorem roots_ratio_quadratic (p : ℤ) (h : (∃ x1 x2 : ℤ, x1*x2 = -16 ∧ x1 + x2 = -p ∧ x2 = -4 * x1)) :
  p = 6 ∨ p = -6 :=
sorry

end roots_ratio_quadratic_l583_583169


namespace parabola_chord_constant_t_l583_583059

theorem parabola_chord_constant_t (c : ℝ) (h : c = 1 / 4) :
  ∀ A B : ℝ × ℝ, 
  y = 2 * x ^ 2 →
  (line_through C passes_through (0,c)) →
  t = 1 / (A.distance (0, c)) ^ 2 + 1 / (B.distance (0, c)) ^ 2 →
  t = 8 :=
begin
  sorry
end

end parabola_chord_constant_t_l583_583059


namespace divisors_of_24516_divisor_count_24516_l583_583627

def is_divisor (n d : ℕ) : Prop := d ∣ n

def divisors (n : ℕ) (divisors : List ℕ) : List ℕ :=
  divisors.filter (is_divisor n)

theorem divisors_of_24516 : divisors 24516 [1, 2, 3, 4, 5, 6, 7, 8, 9] = [1, 2, 3, 4, 6, 9] :=
by 
  sorry

theorem divisor_count_24516 : (divisors_of_24516.length) = 6 :=
by 
  sorry

end divisors_of_24516_divisor_count_24516_l583_583627


namespace floor_equality_solution_l583_583137

theorem floor_equality_solution (x : ℝ) :
  (∃ x, ⌊⌊3 * x⌋ + 1 / 2⌋ = ⌊x + 3⌋) ↔ (x ∈ set.Ico (4 / 3) 2) :=
begin
  sorry
end

end floor_equality_solution_l583_583137


namespace derivative_y_l583_583800

noncomputable def u (x : ℝ) := 4 * x - 1 + Real.sqrt (16 * x ^ 2 - 8 * x + 2)
noncomputable def v (x : ℝ) := Real.sqrt (16 * x ^ 2 - 8 * x + 2) * Real.arctan (4 * x - 1)

noncomputable def y (x : ℝ) := Real.log (u x) - v x

theorem derivative_y (x : ℝ) :
  deriv y x = (4 * (1 - 4 * x)) / (Real.sqrt (16 * x ^ 2 - 8 * x + 2)) * Real.arctan (4 * x - 1) :=
by
  sorry

end derivative_y_l583_583800


namespace find_least_positive_integer_l583_583549

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l583_583549


namespace total_profit_l583_583083

-- Define the capital shares of each partner
def A_share (X : ℝ) := (1/3) * X
def B_share (X : ℝ) := (1/4) * X
def C_share (X : ℝ) := (1/5) * X
def D_share (X : ℝ) := X - (A_share X + B_share X + C_share X)

-- Define the total profit without proof
theorem total_profit (X : ℝ) (A_profit : ℝ) (hA : A_profit = 810) : 
  let P := A_profit * 3 in
  P = 2430 :=
by
  sorry

end total_profit_l583_583083


namespace distance_traveled_by_P_l583_583259

noncomputable def triangleABC : Triangle := { a := 6, b := 8, c := 10, rightAngle := true }
def radius : ℝ := 2

theorem distance_traveled_by_P (t : Triangle) (r : ℝ) (h_t : t = triangleABC) (h_r : r = radius) :
  ∀ (P : Point), (P is the center of a circle with radius r that rolls inside t) → distance_traveled P = 12 := 
sorry

end distance_traveled_by_P_l583_583259


namespace find_m_l583_583962

theorem find_m (m : ℝ) : 
  let a := (m, real.sqrt 3);
      b := (real.sqrt 3, -3)
  in a.1 / b.1 = a.2 / b.2 → m = -1 :=
by
  intros h
  sorry

end find_m_l583_583962


namespace range_g_a_values_l583_583917

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem range_g : ∀ x : ℝ, -1 ≤ g x ∧ g x ≤ 1 :=
sorry

theorem a_values (a : ℝ) : (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) :=
sorry

end range_g_a_values_l583_583917


namespace midpoints_of_ratios_and_collinearity_l583_583995

variables {A B C D M N : Point}
variables (S_ABD S_BCD S_ABC : ℝ)
variables (r : ℝ)

-- Given conditions
def is_collinear (B M N : Point) : Prop := ∃ (l : Line), B ∈ l ∧ M ∈ l ∧ N ∈ l
def area_ratio_condition (S_ABD S_BCD S_ABC : ℝ) : Prop := S_ABD / S_BCD = 3 / 4 ∧ S_ABD / S_ABC = 3

-- Main theorem to prove M and N are midpoints
theorem midpoints_of_ratios_and_collinearity
    (h1 : S_ABD / S_BCD = 3 / 4)
    (h2 : S_ABD / S_ABC = 3)
    (h3 : is_collinear B M N)
    (h4 : AM / AC = CN / CD := r) :
    AM = AC / 2 ∧ CN = CD / 2 :=
sorry

end midpoints_of_ratios_and_collinearity_l583_583995


namespace g_243_l583_583015

noncomputable def g : ℕ → ℝ := sorry

theorem g_243 : g 243 = 125 :=
by {
  -- Define the given condition as an assumption
  have h : ∀ (x y m : ℕ), x + y = 3 ^ m → g x + g y = m ^ 3, from sorry,
  sorry
}

end g_243_l583_583015


namespace find_k_of_collinear_points_l583_583470

theorem find_k_of_collinear_points :
  ∃ k : ℚ, ∀ (x1 y1 x2 y2 x3 y3 : ℚ), (x1, y1) = (4, 10) → (x2, y2) = (-3, k) → (x3, y3) = (-8, 5) → 
  ((y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)) → k = 85 / 12 :=
by
  sorry

end find_k_of_collinear_points_l583_583470


namespace train_ride_length_l583_583714

noncomputable def totalMinutesUntil0900 (leaveTime : Nat) (arrivalTime : Nat) : Nat :=
  arrivalTime - leaveTime

noncomputable def walkTime : Nat := 10

noncomputable def rideTime (totalTime : Nat) (walkTime : Nat) : Nat :=
  totalTime - walkTime

theorem train_ride_length (leaveTime : Nat) (arrivalTime : Nat) :
  leaveTime = 450 → arrivalTime = 540 → rideTime (totalMinutesUntil0900 leaveTime arrivalTime) walkTime = 80 :=
by
  intros h_leaveTime h_arrivalTime
  rw [h_leaveTime, h_arrivalTime]
  unfold totalMinutesUntil0900
  unfold rideTime
  unfold walkTime
  sorry

end train_ride_length_l583_583714


namespace area_transformation_l583_583853

variables {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x in a..b, g x = 12) :
  ∫ x in c..d, 4 * g (2 * x + 3) = 48 :=
by
  sorry

end area_transformation_l583_583853


namespace find_six_y_minus_four_squared_l583_583004

theorem find_six_y_minus_four_squared (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) :
  (6 * y - 4)^2 = 134 :=
by
  sorry

end find_six_y_minus_four_squared_l583_583004


namespace pentagon_proof_l583_583179

theorem pentagon_proof (ABCDE : Type) [circle ABCDE]
  (O : center ABCDE)
  (h1 : arc_length O A B = arc_length O B C = arc_length O C D)
  (h2 : arc_length O A E = arc_length O D E)
  (BD BE AC : Type) [intersect AC BD F] [intersect AC BE G] [intersect AD BE H]
  (h_AG : AG = 3)
  (h_BF : BF = 4) :
  (AG^2 = BG * EH) ∧ (BE = 6 * sqrt 2) := 
sorry

end pentagon_proof_l583_583179


namespace solution_set_x2_f_x_positive_l583_583940

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_at_2 : f 2 = 0
axiom derivative_condition : ∀ x, x > 0 → ((x * (deriv f x) - f x) / x^2) > 0

theorem solution_set_x2_f_x_positive :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_x2_f_x_positive_l583_583940


namespace x_y_sum_vals_l583_583638

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end x_y_sum_vals_l583_583638


namespace find_k_l583_583681

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 50 < f a b c 7)
  (h3 : f a b c 7 < 60)
  (h4 : 70 < f a b c 8)
  (h5 : f a b c 8 < 80)
  (h6 : 5000 * k < f a b c 100)
  (h7 : f a b c 100 < 5000 * (k + 1)) :
  k = 3 :=
sorry

end find_k_l583_583681


namespace factorize_expr1_factorize_expr2_factorize_expr3_factorize_expr4_l583_583136

variable {R : Type} [CommRing R]
variables {m n x a b : R}

-- 1. Factorization of 3mn - 6m^2n^2
theorem factorize_expr1 : 3 * m * n - 6 * m^2 * n^2 = 3 * m * n * (1 - 2 * m * n) :=
by
  sorry

-- 2. Factorization of m^2 - 4mn + 4n^2
theorem factorize_expr2 : m^2 - 4 * m * n + 4 * n^2 = (m - 2 * n)^2 :=
by
  sorry

-- 3. Factorization of x^3 - 9x
theorem factorize_expr3 : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

-- 4. Factorization of 6ab^2 - 9a^2b - b^3
theorem factorize_expr4 : 6 * a * b^2 - 9 * a^2 * b - b^3 = -b * (b - 3 * a)^2 :=
by
  sorry

end factorize_expr1_factorize_expr2_factorize_expr3_factorize_expr4_l583_583136


namespace sum_of_all_paintable_numbers_l583_583964

def paintable (h t u : ℕ) : Prop :=
  ¬(h = 1 ∨ t + 1 = 1 ∨ u + 2 = 1) ∧
  ∀ n : ℕ, n ≠ 0 → n < 1000 → (n % h = 0 ∨ (n - 3) % (t + 1) = 0 ∨ (n - 4) % (u + 2) = 0) ∧
  ∀ n m : ℕ, n ≠ m → (n % h = 0 ∨ (n - 3) % (t + 1) = 0 ∨ (n - 4) % (u + 2) = 0) →
  ((n % h = 0 ∨ (n - 3) % (t + 1) = 0 ∨ (n - 4) % (u + 2) = 0) → ¬(m % h = 0 ∨ (m - 3) % (t + 1) = 0 ∨ (m - 4) % (u + 2) = 0)) 

theorem sum_of_all_paintable_numbers : 
  (∑ h t u in finset.range 5, if paintable h t u then 200 * h + 20 * t + 2 * u else 0) = 1510 := 
  by sorry

end sum_of_all_paintable_numbers_l583_583964


namespace distance_from_sphere_center_to_plane_is_4_l583_583077

noncomputable def distance_center_to_plane (O : Point) (radius : ℝ) (triangle : Triangle) (legs : ℝ × ℝ) : ℝ :=
    let (leg1, leg2) := legs
    let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
    let area := (1 / 2) * leg1 * leg2
    let semiperimeter := (leg1 + leg2 + hypotenuse) / 2
    let inradius := area / semiperimeter
    let radius_distance := radius^2 - inradius^2
    Real.sqrt radius_distance

theorem distance_from_sphere_center_to_plane_is_4 (O : Point) (radius : ℝ) (triangle : Triangle) (leg1 leg2 : ℝ) (tangent : Triangle → Sphere → Prop) 
    (tangent_to_sphere : tangent triangle (Sphere.mk O radius)) 
    (legs_tangent_to_sphere : leg1 = 8 ∧ leg2 = 15) :
  distance_center_to_plane O radius triangle (8, 15) = 4 := 
sorry

end distance_from_sphere_center_to_plane_is_4_l583_583077


namespace zero_clever_numbers_l583_583746

def isZeroClever (n : Nat) : Prop :=
  ∃ a b c : Nat, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  n = 1000 * a + 10 * b + c ∧
  n = 9 * (100 * a + 10 * b + c)

theorem zero_clever_numbers :
  ∀ n : Nat, isZeroClever n → n = 2025 ∨ n = 4050 ∨ n = 6075 :=
by
  -- Proof to be provided
  sorry

end zero_clever_numbers_l583_583746


namespace value_of_expression_l583_583235

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end value_of_expression_l583_583235


namespace measure_exterior_angle_BAC_l583_583840

-- Define the interior angle of a regular nonagon
def nonagon_interior_angle := (180 * (9 - 2)) / 9

-- Define the exterior angle of the nonagon
def nonagon_exterior_angle := 360 - nonagon_interior_angle

-- The square's interior angle
def square_interior_angle := 90

-- The question to be proven
theorem measure_exterior_angle_BAC :
  nonagon_exterior_angle - square_interior_angle = 130 :=
  by
  sorry

end measure_exterior_angle_BAC_l583_583840


namespace quadrilateral_area_proof_l583_583387

noncomputable def area_of_quadrilateral : ℚ :=
  let f1 : ℚ × ℚ → Prop := λ p, p.snd = -p.fst + 7
  let f2 : ℚ × ℚ → Prop := λ p, p.snd = p.fst / 2 + 1
  let f3 : ℚ × ℚ → Prop := λ p, p.snd = -3 * p.fst / 2 + 2
  let f4 : ℚ × ℚ → Prop := λ p, p.snd = 7 * p.fst / 4 + 3 / 2
  let A : ℚ × ℚ := (2, 5)
  let B : ℚ × ℚ := (4, 3)
  let C : ℚ × ℚ := (1 / 2, 5 / 4)
  let D : ℚ × ℚ := (2 / 13, 23 / 13)
  let det := λ (p1 p2 p3 : ℚ × ℚ), p1.fst * p2.snd + p2.fst * p3.snd + p3.fst * p1.snd - (p1.snd * p2.fst + p2.snd * p3.fst + p3.snd * p1.fst)
  (1 / 2 : ℚ) * |det A B C + det A C D|

theorem quadrilateral_area_proof (h1 : ∀ p, f1 p → f4 p) (h2 : ∀ p, f1 p → f2 p) (h3 : ∀ p, f2 p → f3 p) (h4 : ∀ p, f3 p → f4 p) :
  area_of_quadrilateral = 327 / 52 := by
  sorry

end quadrilateral_area_proof_l583_583387


namespace eliminate_uv_and_w_l583_583522

variables {u v w a b c d : ℝ}

def a_cond : Prop := a = Real.cos u + Real.cos v + Real.cos w
def b_cond : Prop := b = Real.sin u + Real.sin v + Real.sin w
def c_cond : Prop := c = Real.cos (2 * u) + Real.cos (2 * v) + Real.cos w
def d_cond : Prop := d = Real.sin (2 * u) + Real.sin (2 * v) + Real.sin w

theorem eliminate_uv_and_w (h1 : a_cond) (h2 : b_cond) (h3 : c_cond) (h4 : d_cond) : 
  (a^2 - b^2 - c)^2 + (2 * a * b - d)^2 = 4 * (a^2 + b^2) := 
sorry

end eliminate_uv_and_w_l583_583522


namespace smallest_integer_condition_l583_583552

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l583_583552


namespace simplify_expression_evaluate_expression_l583_583368

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 0) (h₂ : a ≠ 1) (h₃ : a ≠ -1) (h₄ : -3 < a) (h₅ : a < 3) :
  (a - (2 * a - 1) / a) / ((a ^ 2 - 1) / a) = (a - 1) / (a + 1) :=
sorry

theorem evaluate_expression :
  let a := 2 in
  (2 - (2 * 2 - 1) / 2) / ((2 ^ 2 - 1) / 2) = 1 / 3 :=
sorry

end simplify_expression_evaluate_expression_l583_583368


namespace largest_number_with_digits_sum_to_19_l583_583776

theorem largest_number_with_digits_sum_to_19 : ∃ (n : ℕ), (∀ (d : ℕ), d ∈ digits n → d ≠ 0) ∧ digit_sum n = 19 ∧ largest_number_with_properties n :=
sorry

def digits (n : ℕ) : list ℕ :=
sorry

def digit_sum (n : ℕ) : ℕ :=
(digits n).sum

def largest_number_with_properties (n : ℕ) : Prop :=
∀ (m : ℕ), (∀ (d : ℕ), d ∈ digits m → d ≠ 0) ∧ digit_sum m = 19 → m ≤ n

end largest_number_with_digits_sum_to_19_l583_583776


namespace instantaneous_rate_of_change_at_point_l583_583180

-- Define the function y = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the specific point (1, 2)
def point : ℝ × ℝ := (1, 2)

-- Define the limit of the difference quotient as δx approaches 0
noncomputable def limit_diff_quotient (δx : ℝ) : ℝ := (f (1 + δx) - f 1) / δx

-- State the theorem to prove the question == answer given conditions
theorem instantaneous_rate_of_change_at_point :
  tendsto limit_diff_quotient (𝓝 0) (𝓝 2) :=
sorry

end instantaneous_rate_of_change_at_point_l583_583180


namespace price_reduction_daily_profit_l583_583814

theorem price_reduction_daily_profit
    (profit_per_item : ℕ)
    (avg_daily_sales : ℕ)
    (item_increase_per_unit_price_reduction : ℕ)
    (target_daily_profit : ℕ)
    (x : ℕ) :
    profit_per_item = 40 →
    avg_daily_sales = 20 →
    item_increase_per_unit_price_reduction = 2 →
    target_daily_profit = 1200 →

    ((profit_per_item - x) * (avg_daily_sales + item_increase_per_unit_price_reduction * x) = target_daily_profit) →
    x = 20 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_reduction_daily_profit_l583_583814


namespace gcf_7fact_8fact_l583_583151

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l583_583151


namespace find_n_l583_583168

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_n (n : ℕ) (hn : 10 ≤ n ∧ n < 100)
  (A : n % 16 = 0) (B : n % 27 = 0) (C : n % 36 = 0) (D : sum_of_digits n = 15)
  (two_of_four : exactly_two [A, B, C, D]) :
  n = 96 :=
by
  sorry

/-- Helper function to check exactly two of the four conditions are true. -/
def exactly_two (conds : list Prop) : Prop :=
  conds.count true = 2

end find_n_l583_583168


namespace count_nice_subsets_l583_583182

def S := finset.range 201

def is_nice_subset (A : finset ℕ) : Prop :=
  A.card = 3 ∧ ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + c = 2 * b)

theorem count_nice_subsets : 
  (finset.filter is_nice_subset (S.powerset.filter (λ t, t.card = 3))).card = 9900 :=
sorry

end count_nice_subsets_l583_583182


namespace folded_triangle_length_squared_l583_583833

-- Define the problem conditions
def is_equilateral_triangle (A B C : Point) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Main theorem statement
theorem folded_triangle_length_squared
  (A B C : Point)
  (h_equilateral : is_equilateral_triangle A B C 15)
  (h_fold : ∃ P : Point, dist P C = 4 ∧ on_line P B C) :
  ∃ PA' : ℝ, PA' ^ 2 = 18964 / 2401 :=
sorry

end folded_triangle_length_squared_l583_583833


namespace smallest_integer_to_make_multiple_of_five_l583_583429

/-- The smallest positive integer that can be added to 725 to make it a multiple of 5 is 5. -/
theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k : ℕ, k > 0 ∧ (725 + k) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → k ≤ m :=
sorry

end smallest_integer_to_make_multiple_of_five_l583_583429


namespace f_2020_minus_f_2018_l583_583243

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 5) = f x
axiom f_seven : f 7 = 9

theorem f_2020_minus_f_2018 : f 2020 - f 2018 = 9 := by
  sorry

end f_2020_minus_f_2018_l583_583243


namespace sum_distances_correct_l583_583474

-- Define a regular hexagon with side length 1 and center O
structure Hexagon :=
  (A B C D E F O : ℝ)
  (side_length : ℝ)
  (center : ℝ)
  (regular : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = F ∧
             side_length = 1 ∧ center = O)

-- Define the parabolas P1, P2, ..., P6 with common focus O and directrices AB, BC, CD, DE, EF, FA
structure Parabola :=
  (focus : ℝ)
  (directrix : ℝ)
  (locus : ∀ (X : ℝ), X = (X - focus) ∧ X = (X - directrix))

-- Define the set χ as the set of all distinct points on the plane that lie on at least two of the six parabolas
def χ (hex : Hexagon) :=
  {X : ℝ // ∃ i j, i ≠ j ∧ X ∈ (Parabola.locus i hex) ∧ X ∈ (Parabola.locus j hex)}

-- The question is to compute the sum of distances from O to the points in set χ
def sum_of_distances (hex : Hexagon) : ℝ :=
  ∑ (X : χ hex), |X - hex.center|

-- The statement of the theorem:
theorem sum_distances_correct (hex : Hexagon) : sum_of_distances hex = 35 * sqrt 3 :=
sorry

end sum_distances_correct_l583_583474


namespace tammy_speed_second_day_l583_583043

theorem tammy_speed_second_day:
  ∃ (v t: ℝ), 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    (v + 0.5) = 4 := sorry

end tammy_speed_second_day_l583_583043


namespace find_ab_l583_583347

theorem find_ab 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : a^4 + b^4 = 15 / 4) : 
  a * b = real.sqrt(42) / 4 := 
begin
  sorry
end

end find_ab_l583_583347


namespace arithmetic_geometric_sequence_min_sum_l583_583008

theorem arithmetic_geometric_sequence_min_sum :
  ∃ (A B C D : ℕ), 
    (C - B = B - A) ∧ 
    (C * 4 = B * 7) ∧ 
    (D * 4 = C * 7) ∧ 
    (16 ∣ B) ∧ 
    (A + B + C + D = 97) :=
by sorry

end arithmetic_geometric_sequence_min_sum_l583_583008


namespace valid_inequalities_l583_583346

theorem valid_inequalities (a b c : ℝ) (h : 0 < c) 
  (h1 : b > c - b)
  (h2 : c > a)
  (h3 : c > b - a) :
  a < c / 2 ∧ b < a + c / 2 :=
by
  sorry

end valid_inequalities_l583_583346


namespace option_d_must_be_true_l583_583635

theorem option_d_must_be_true (a b : ℝ) (h : a > b) : 
  (a + 2 > b + 2) ∧ ¬(a^2 > b^2) ∧ ¬(-3a > -3b) ∧ ¬(a/4 < b/4) :=
by
  split; try {linarith}
  all_goals
    linarith

end option_d_must_be_true_l583_583635


namespace odd_function_properties_l583_583254

variables {f : ℝ → ℝ} [odd : ∀ x, f (-x) = -f x] [inc : ∀ x y ∈ Icc 3 7, x < y → f x < f y]
variables {min_val : f 3 = 1}

theorem odd_function_properties :
  (∀ x y ∈ Icc (-7 : ℝ) (-3 : ℝ), x < y → f x < f y) ∧ ∃ x ∈ Icc (-7 : ℝ) (-3 : ℝ), f x = -1 :=
by {
  sorry
}

end odd_function_properties_l583_583254


namespace solve_z_l583_583238

theorem solve_z (z : ℂ) (h : z * (1 + complex.I) = 2 * complex.I) : z = 1 + complex.I :=
by
  sorry

end solve_z_l583_583238


namespace points_in_plane_l583_583118
noncomputable def myFunction (x : ℝ) : ℝ := sin x / |sin x|

theorem points_in_plane (x : ℝ) (n : ℤ) : 
  (y = myFunction x) → 
  ((x ∈ set.Ioo (π * (2 * n - 1)) (2 * n * π) → y = 1) ∧ 
   (x ∈ set.Ioo (2 * n * π) (π * (2 * n + 1)) → y = -1) ∧ 
   (x ∈ ⋃ k : ℤ, {k * π} → false)) := sorry

end points_in_plane_l583_583118


namespace math_problem_l583_583193

noncomputable def proof_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : Prop :=
  let n1 := a + 1/b
  let n2 := b + 1/c
  let n3 := c + 1/a
  (n1 ≤ -2) ∨ (n2 ≤ -2) ∨ (n3 ≤ -2)

theorem math_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : proof_problem a b c h₀ h₁ h₂ :=
sorry

end math_problem_l583_583193


namespace probability_even_sum_l583_583768

def total_balls := List.range' 1 13  -- List of numbers 1 to 12

noncomputable def total_outcomes := (total_balls.length) * (total_balls.length - 1)

def is_even (n : ℕ) : Bool := n % 2 = 0

def favourable_outcomes (balls : List ℕ) : ℕ :=
  (balls.filter is_even).length * (balls.filter is_even).length +
  (balls.filter (λ n, ¬is_even n)).length * (balls.filter (λ n, ¬is_even n)).length

theorem probability_even_sum : 
  (ℚ.of_nat (favourable_outcomes total_balls) / ℚ.of_nat total_outcomes) = 5 / 11 :=
by
  -- Solution would go here
  sorry

end probability_even_sum_l583_583768


namespace find_m_l583_583202

def ellipse_major_axis_x (m : ℝ) : Prop :=
  m - 2 > 10 - m

def focal_length_four (m : ℝ) : Prop :=
  let c := 4 in c * c = (m - 2) - (10 - m)

theorem find_m (m : ℝ) (h1 : 6 < m) (h2 : m < 10) (h3 : ellipse_major_axis_x m) (h4 : focal_length_four m) : m = 8 :=
sorry

end find_m_l583_583202


namespace total_rent_l583_583040

/-- Given the usage of the pasture by A, B, and C, and knowing the amount paid by B,
    prove that the total rent for the pasture is Rs. 870. -/
theorem total_rent (A_horses : ℕ) (A_months : ℕ) (B_horses : ℕ) (B_months : ℕ) (C_horses : ℕ) (C_months : ℕ)
  (B_payment : ℕ) (B_payment_amount : ℕ)
  (hA : A_horses = 12) (hA_months : A_months = 8)
  (hB : B_horses = 16) (hB_months : B_months = 9) (hB_payment : B_payment_amount = 360)
  (hC : C_horses = 18) (hC_months : C_months = 6)
  (hB_payment_months : B_payment = B_horses * B_months) :
  let A_horse_months := A_horses * A_months,
      B_horse_months := B_horses * B_months,
      C_horse_months := C_horses * C_months,
      total_horse_months := A_horse_months + B_horse_months + C_horse_months,
      cost_per_horse_month := B_payment_amount / B_horse_months,
      total_rent := cost_per_horse_month * total_horse_months
  in total_rent = 870 :=
by {
  -- Proof goes here
  sorry
}

end total_rent_l583_583040


namespace value_of_b_l583_583983

theorem value_of_b (y b : ℝ) (hy : y > 0) (h : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 :=
by
  -- Proof omitted for brevity
  sorry

end value_of_b_l583_583983


namespace fraction_subtraction_equivalence_l583_583887

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end fraction_subtraction_equivalence_l583_583887


namespace man_age_difference_l583_583825

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end man_age_difference_l583_583825


namespace ratio_of_boys_to_girls_l583_583985

theorem ratio_of_boys_to_girls (B G : ℕ) (h1 : B = 80) (h2 : G = B + 128) : B / Nat.gcd B G = 5 ∧ G / Nat.gcd B G = 13 :=
by
  -- Definitions based on conditions
  have hB : B = 80 := h1
  -- Substituting B into the equation for G
  have hG : G = 208 := by rwa [hB, add_comm]
  -- Numerically finding GCD(B, G)
  have gcd_def : Nat.gcd 80 208 = 16 := by norm_num
  -- Simplifying the ratio using the GCD
  have ratio_boys : 80 / 16 = 5 := by norm_num
  have ratio_girls : 208 / 16 = 13 := by norm_num
  
  -- Combining the simplified results
  exact ⟨ratio_boys, ratio_girls⟩

end ratio_of_boys_to_girls_l583_583985


namespace find_least_positive_integer_l583_583550

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l583_583550


namespace probability_inequality_l583_583355

theorem probability_inequality:
  let lower_bound := -2
  let upper_bound := 3
  let condition := ∀ (x : ℝ), lower_bound ≤ x ∧ x ≤ upper_bound → (x + 1) * (x - 3) ≤ 0
  let probability := ((3 - (-1)) / (3 - (-2)))
  in probability = 4 / 5 :=
by
  sorry

end probability_inequality_l583_583355


namespace solve_sqrt_equation_l583_583727

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt (9 + sqrt (25 + 5 * x)) + sqrt (3 + sqrt (5 + x)) = 3 + 3 * sqrt 3) → x = 0.2 :=
by
  intro x h
  sorry

end solve_sqrt_equation_l583_583727


namespace sum_x_coordinates_l583_583864

theorem sum_x_coordinates (x1 x2 : ℝ) :
  let center := (3 : ℝ, -4 : ℝ)
  let radius := 7
  let equation := λ x y : ℝ, (x - 3) ^ 2 + (y + 4) ^ 2 = radius ^ 2
  ((equation x1 0) ∧ (equation x2 0)) →
  x1 + x2 = 6 :=
by
  sorry

end sum_x_coordinates_l583_583864


namespace mom_younger_than_grandmom_l583_583112

def cara_age : ℕ := 40
def cara_younger_mom : ℕ := 20
def grandmom_age : ℕ := 75

def mom_age : ℕ := cara_age + cara_younger_mom
def age_difference : ℕ := grandmom_age - mom_age

theorem mom_younger_than_grandmom : age_difference = 15 := by
  sorry

end mom_younger_than_grandmom_l583_583112


namespace trays_from_second_table_l583_583027

variable (capacity_per_trip : ℕ) (total_trips : ℕ) (first_table_trays : ℕ)

theorem trays_from_second_table (h_capacity : capacity_per_trip = 7) (h_trips : total_trips = 4) (h_first_table : first_table_trays = 23) :
  (capacity_per_trip * total_trips - first_table_trays) = 5 :=
by
  rw [h_capacity, h_trips, h_first_table]
  simp
  norm_num
  sorry

end trays_from_second_table_l583_583027


namespace parabola_intersection_l583_583024

theorem parabola_intersection :
  let p1 := λ x : ℝ => 3*x^2 - 9*x - 5,
      p2 := λ x : ℝ => x^2 - 6*x + 10,
      x1 := (3 + Real.sqrt 129) / 4,
      x2 := (3 - Real.sqrt 129) / 4,
      y1 := p1 x1,
      y2 := p1 x2
  in (p1 x1 = p2 x1) ∧ (p1 x2 = p2 x2) :=
by
  sorry

end parabola_intersection_l583_583024


namespace minimize_power_line_correct_l583_583852

noncomputable def minimize_power_line (AB AC BC : ℝ) (h1 : AB = 0.6) (h2 : AC = 0.5) (h3 : BC = 0.5) (h4 : ∀ D : ℝ, 0 < D ∧ D < AB) : ℝ :=
  let x := (real.sqrt 3) / 10
  x

theorem minimize_power_line_correct :
  minimize_power_line 0.6 0.5 0.5 = (real.sqrt 3) / 10 :=
by
  -- Proof goes here
  sorry

end minimize_power_line_correct_l583_583852


namespace product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l583_583175

variable {x y : ℝ}

-- The formal statement in Lean
theorem product_pos_implies_pos_or_neg (h : x * y > 0) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
sorry

theorem pos_pair_implies_product_pos (hx : x > 0) (hy : y > 0) : x * y > 0 :=
sorry

theorem product_pos_necessary_for_pos (h : x > 0 ∧ y > 0) : x * y > 0 :=
pos_pair_implies_product_pos h.1 h.2

theorem product_pos_not_sufficient_for_pos (h : x * y > 0) : ¬ (x > 0 ∧ y > 0) :=
sorry

end product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l583_583175


namespace initial_ratio_of_milk_to_water_l583_583653

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + 20 = 3 * W) (h2 : M + W = 40) :
  (M : ℚ) / W = 5 / 3 := by
sorry

end initial_ratio_of_milk_to_water_l583_583653


namespace remainder_division_1000_l583_583784

theorem remainder_division_1000 (x : ℕ) (hx : x > 0) (h : 100 % x = 10) : 1000 % x = 10 :=
  sorry

end remainder_division_1000_l583_583784


namespace least_positive_integer_l583_583544

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l583_583544


namespace triangle_side_length_b_l583_583686

theorem triangle_side_length_b
  (a b c : ℝ)
  (h_area : 0.5 * a * c * (Real.sin (Real.pi / 3)) = Real.sqrt 3)
  (h_angle_B : Real.angle 60 = Real.pi / 3)
  (h_relation : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 := 
sorry

end triangle_side_length_b_l583_583686


namespace sum_f_zero_l583_583646

-- Definition of the function f
def f (n : ℕ) : ℝ := Real.tan ((n : ℝ) * Real.pi / 2 + Real.pi / 4)

-- Main theorem statement of the proof problem
theorem sum_f_zero : ∑ n in Finset.range 2016, f n = 0 := 
by 
  sorry

end sum_f_zero_l583_583646


namespace sum_of_exponents_l583_583502

theorem sum_of_exponents : 
  (-1)^(2010) + (-1)^(2013) + 1^(2014) + (-1)^(2016) = 0 := 
by
  sorry

end sum_of_exponents_l583_583502


namespace can_guess_number_of_numbers_cannot_guess_number_of_numbers_l583_583724

-- Definitions
def Sasha_circle (n : ℕ) (numbers : List ℕ) : Prop :=
  n ≤ 100 ∧ numbers.length = n ∧ ∀ x, x ∈ numbers → x ∈ Finset.range (n + 1)

def Dima_indices (indices : List ℕ) : Prop :=
  indices.length = 17 ∧ ∀ x, x ∈ indices → x ∈ Finset.range 100

-- Hypotheses
variables (n : ℕ) (numbers : List ℕ) (indices : List ℕ)
variable (hSasha : Sasha_circle n numbers)
variable (hDima : Dima_indices indices)

-- Theorem Statement
theorem can_guess_number_of_numbers : ∃ k, k ∈ Finset.range n → kth_position indices numbers = k :=
sorry

-- Define the second part for fewer than 16 indices
def Dima_indices_fewer (indices : List ℕ) : Prop :=
  indices.length < 16 ∧ ∀ x, x ∈ indices → x ∈ Finset.range 100

-- Hypotheses for the second part
variable (hDima_fewer : Dima_indices_fewer indices)

-- Theorem Statement for fewer than 16 indices
theorem cannot_guess_number_of_numbers : ¬ ∃ k, k ∈ Finset.range n → kth_position indices numbers = k :=
sorry

end can_guess_number_of_numbers_cannot_guess_number_of_numbers_l583_583724


namespace proof_AM_length_l583_583667

noncomputable def length_AM (AB BC CA : ℕ) (p q : ℕ) : Prop :=
  ∃ A B C : Type, ∃ (triangle : (A × B × C)) (ω1 ω2 : Type × Type × Type) (M : Type),
  (AB = 13) ∧ (BC = 14) ∧ (CA = 15) ∧
  ((ω1 through C and is tangent to line AB at A) ∧ 
   (ω2 through B and is tangent to line AC at A)) ∧
  (M is the intersection of ω1 and ω2 not equal to A) ∧
   15 = p ∧ 2 = q ∧ AM = 15/2

theorem proof_AM_length :
  (AB BC CA : ℕ) (p q : ℕ), length_AM AB BC CA p q → p + q = 17 := sorry

end proof_AM_length_l583_583667


namespace circle_center_correct_two_distinct_intersections_shortest_chord_k_value_shortest_chord_length_l583_583588

noncomputable def circle_center : ℝ × ℝ :=
  let c := (3, 4) in
  if (c.1 - 3)^2 + (c.2 - 4)^2 = 4 then c
  else sorry

theorem circle_center_correct :
  (x y: ℝ) (h : (x - 3)^2 + (y - 4)^2 = 4) : (x, y) = (3, 4) :=
sorry

theorem two_distinct_intersections (k : ℝ) :
  ∃ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 4 ∧ k x - y - 4 * k + 3 = 0 :=
sorry

theorem shortest_chord_k_value :
  ∃ (d q : ℝ), d = sqrt 2 ∧ q = 1 ∧
  d = abs (1 + q) / sqrt (1 + q^2) :=
sorry

theorem shortest_chord_length :
  q = 1 →
  2 * sqrt ((2^2) - (sqrt 2)^2) = 2 * sqrt 2 :=
sorry

end circle_center_correct_two_distinct_intersections_shortest_chord_k_value_shortest_chord_length_l583_583588


namespace positive_solutions_l583_583516

def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib n + fib (n+1)

theorem positive_solutions (x : ℝ) (hx : 0 < x) :
  (1 / (1 + 1 / (1 + 1 / x))) = x ↔ x = 1 ∨ ∃ (n : ℕ), 0 < n ∧ x = real.sqrt (fib n / fib (n + 2)) :=
by sorry

end positive_solutions_l583_583516


namespace integer_solutions_l583_583531

def system_of_equations (x y z t : ℤ) : Prop :=
  x * z - 2 * y * t = 3 ∧ x * t + y * z = 1

theorem integer_solutions :
  { (x, y, z, t) : ℤ × ℤ × ℤ × ℤ // system_of_equations x y z t } =
  {⟨(1, 0, 3, 1), sorry⟩, ⟨(-1, 0, -3, -1), sorry⟩, ⟨(3, 1, 1, 0), sorry⟩, ⟨(-3, -1, -1, 0), sorry⟩} :=
sorry

end integer_solutions_l583_583531


namespace positive_difference_of_two_numbers_l583_583412

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_of_two_numbers_l583_583412


namespace Pn_converges_l583_583693

universe u

structure Point (α : Type u) :=
  (x : α)
  (y : α)

structure Triangle (α : Type u) :=
  (A B C : Point α)
  (eq_sides : dist A B = dist B C ∧ dist B C = dist C A)

noncomputable def dist {α : Type u} [Field α] [MetricSpace (Point α)] (p1 p2 : Point α) : α :=
  sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

noncomputable def perpendicular_foot {α : Type u} [Field α] (p1 p2 p3 : Point α) : Point α := sorry

noncomputable def sequence_P (α : Type u) [Field α] (T : Triangle α) : ℕ → Point α
| 0     := sorry -- Let P1 be any initial point on AB
| (n+1) := let Q := perpendicular_foot (sequence_P α T n) T.B T.C in
           let R := perpendicular_foot Q T.C T.A in
           perpendicular_foot R T.A T.B

theorem Pn_converges {α : Type u} [Field α] [MetricSpace (Point α)] (T : Triangle α) (P : ℕ → Point α) :
  (∀ n, P n = sequence_P α T n) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, dist (P n) (Point.mk ((2 / 3 : α) * (T.B.x) + (1 / 3) * (T.A.x)) ((2 / 3 : α) * (T.B.y) + (1 / 3) * (T.A.y))) < ε) :=
begin
  sorry
end

end Pn_converges_l583_583693


namespace repeating_decimal_denominators_l583_583378

theorem repeating_decimal_denominators (a b c : ℕ) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) (h_not_all_nine : ¬(a = 9 ∧ b = 9 ∧ c = 9)) : 
  ∃ denominators : Finset ℕ, denominators.card = 7 ∧ (∀ d ∈ denominators, d ∣ 999) ∧ ¬ 1 ∈ denominators :=
sorry

end repeating_decimal_denominators_l583_583378


namespace carolyn_marbles_l583_583861

theorem carolyn_marbles (initial_marbles : ℕ) (shared_items : ℕ) (end_marbles: ℕ) : 
  initial_marbles = 47 → shared_items = 42 → end_marbles = initial_marbles - shared_items → end_marbles = 5 :=
by
  intros h₀ h₁ h₂
  rw [h₀, h₁] at h₂
  exact h₂

end carolyn_marbles_l583_583861


namespace max_volume_at_sixty_degrees_l583_583407

-- Define the conditions
variable (a : ℝ) (α : ℝ) (V : ℝ)

-- Define the volume as per the derived formula
def volume_of_solid_of_revolution : ℝ :=
  (π / 3 ) * a^3 * (sin α + sqrt 3 / 2 * cos α + 1 / 2 * sin α) *
  (cos α * (sqrt 3 / 2 * cos α + 1 / 2 * sin α) + (1 / 2 * cos α - sqrt 3 / 2 * sin α) * sin α)

-- Prove that the volume is maximized at α = π / 3 (60 degrees)
theorem max_volume_at_sixty_degrees 
    (h₀: α = π / 3) : V = (π * a^3) / 2 :=
by
  rw [h₀, volume_of_solid_of_revolution]
  sorry

end max_volume_at_sixty_degrees_l583_583407


namespace perspective_drawing_area_equilateral_triangle_l583_583595

-- Declare the side length and the original area of the equilateral triangle
def side_length : ℝ := 4
def original_area (a : ℝ) : ℝ := (sqrt 3 / 4) * a^2

-- Original area with side length 4
def S : ℝ := original_area side_length

-- Perspective drawing area using the oblique projection method
def perspective_area (S : ℝ) : ℝ := (sqrt 2 / 4) * S

-- Statement of the theorem
theorem perspective_drawing_area_equilateral_triangle :
  perspective_area S = sqrt 6 :=
by
  sorry

end perspective_drawing_area_equilateral_triangle_l583_583595


namespace dot_product_eq_three_l583_583623

variables (a b : ℝ → ℝ)
variables (θ : ℝ) (norm_a norm_b : ℝ)

-- Assume conditions
def angle_between (a b : ℝ → ℝ) : ℝ := 30 * Real.pi / 180
def norm_a := Real.sqrt 3
def norm_b := 2

-- Prove that the dot product equals 3
theorem dot_product_eq_three (h1 : θ = angle_between a b)
                             (h2 : norm_a = Real.sqrt 3)
                             (h3 : norm_b = 2) :
  (norm_a * norm_b * Real.cos θ) = 3 :=
sorry

end dot_product_eq_three_l583_583623


namespace flagpole_perpendicular_to_ground_l583_583286

variables (l : Type) (π : Type) [line l] [plane π]

theorem flagpole_perpendicular_to_ground : is_perpendicular l π :=
sorry

end flagpole_perpendicular_to_ground_l583_583286


namespace radical_combination_possible_l583_583037

noncomputable def can_combine (x y : ℝ) : Prop := x = y

theorem radical_combination_possible :
  can_combine (sqrt 3) (sqrt (1 / 3)) :=
by
  sorry

end radical_combination_possible_l583_583037


namespace positive_difference_l583_583409

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end positive_difference_l583_583409


namespace least_positive_integer_l583_583540

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l583_583540


namespace calc_3a2b_times_neg_a_squared_l583_583111

variables {a b : ℝ}

theorem calc_3a2b_times_neg_a_squared : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b :=
by
  sorry

end calc_3a2b_times_neg_a_squared_l583_583111


namespace find_number_l583_583808

theorem find_number (x : ℝ) (h : 0.36 * x = 129.6) : x = 360 :=
by sorry

end find_number_l583_583808


namespace sum_odd_numbers_less_than_20_l583_583164

theorem sum_odd_numbers_less_than_20 : 
  let a_n := λ n : ℕ, 2 * n - 1 in
  ∑ i in finset.range 10, a_n (i + 1) = 100 :=
by
  sorry

end sum_odd_numbers_less_than_20_l583_583164


namespace initial_chocolates_l583_583701

theorem initial_chocolates (y: ℕ) 
    (H1: y % 4 = 0) 
    (H2: 9 ≤ (y / 2) - 40) 
    (H3: (y / 2) - 40 ≤ 14)
    : y = 104 :=
begin
    sorry
end

end initial_chocolates_l583_583701


namespace pqrs_sum_l583_583315

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l583_583315


namespace phase_shift_right_by_pi_div_3_l583_583421

noncomputable def graph_shift_right_by_pi_div_3 
  (A : ℝ := 1) 
  (ω : ℝ := 1) 
  (φ : ℝ := - (Real.pi / 3)) 
  (y : ℝ → ℝ := fun x => Real.sin (x - Real.pi / 3)) : 
  Prop :=
  y = fun x => Real.sin (x - (Real.pi / 3))

theorem phase_shift_right_by_pi_div_3 (A : ℝ := 1) (ω : ℝ := 1) (φ : ℝ := - (Real.pi / 3)) :
  graph_shift_right_by_pi_div_3 A ω φ (fun x => Real.sin (x - Real.pi / 3)) :=
sorry

end phase_shift_right_by_pi_div_3_l583_583421


namespace remainder_of_poly_division_l583_583561

theorem remainder_of_poly_division :
  ∀ (x : ℝ), (x^2023 + x + 1) % (x^6 - x^4 + x^2 - 1) = x^7 + x + 1 :=
by
  sorry

end remainder_of_poly_division_l583_583561


namespace boysFromANotStudyingScience_is_56_l583_583984

noncomputable def numberOfBoysFromSchoolA (totalBoys : ℕ) (percentA : ℝ) : ℕ :=
  percentA * totalBoys

noncomputable def numberOfBoysFromSchoolAStudyingScience (boysFromA : ℕ) (percentStudyScience : ℝ) : ℕ :=
  percentStudyScience * boysFromA

noncomputable def boysFromAScience (totalBoys : ℕ) (percentA percentScience : ℝ) : ℕ :=
  let fromA := numberOfBoysFromSchoolA totalBoys percentA
  let fromAScience := numberOfBoysFromSchoolAStudyingScience fromA percentScience
  fromA - fromAScience

theorem boysFromANotStudyingScience_is_56 :
  let totalBoys := 400
  let percentA := 0.20
  let percentScience := 0.30
  boysFromAScience totalBoys percentA percentScience = 56 :=
by
  -- Omitted proof
  sorry

end boysFromANotStudyingScience_is_56_l583_583984


namespace ball_selection_l583_583758

open Nat

theorem ball_selection (h_10 : Finset.Fin 10)
  (exists_balls : ∀(b1 b2 b3 : Finset Fintype (Fin 10)), 
    (b1 + b2 + b3 = 10) ∧ b1.val < 5 ∧ b2.val > 5 ∧ b3.val = 5):
  C(5, 1) * C(4, 1) = 20 :=
by
  sorry

end ball_selection_l583_583758


namespace part1_part2_l583_583801

noncomputable def f (x : ℝ) : ℝ := x - log x - 2
noncomputable def g (x : ℝ) : ℝ := x * log x + x

theorem part1 : ∃ x0, 3 < x0 ∧ x0 < 4 ∧ f x0 = 0 := by
  sorry

theorem part2 : ∃ k : ℤ, (∀ x > 1, g x > k * (x - 1)) ∧ 
                 (∀ k', (∀ x > 1, g x > k' * (x - 1)) → k' ≤ 3) := by
  sorry

end part1_part2_l583_583801


namespace bob_average_speed_l583_583497

theorem bob_average_speed
  (lap_distance : ℕ) (lap1_time lap2_time lap3_time total_laps : ℕ)
  (h_lap_distance : lap_distance = 400)
  (h_lap1_time : lap1_time = 70)
  (h_lap2_time : lap2_time = 85)
  (h_lap3_time : lap3_time = 85)
  (h_total_laps : total_laps = 3) : 
  (lap_distance * total_laps) / (lap1_time + lap2_time + lap3_time) = 5 := by
    sorry

end bob_average_speed_l583_583497


namespace solve_system_l583_583376

def eq1 (x y : ℝ) := x^2 - 6 * (sqrt (3 - 2 * x)) - y + 11 = 0
def eq2 (x y : ℝ) := y^2 - 4 * (sqrt (3 * y - 2)) + 4 * x + 16 = 0

theorem solve_system : eq1 (-3) 2 ∧ eq2 (-3) 2 := sorry

end solve_system_l583_583376


namespace linear_regression_not_through_6_l583_583204

noncomputable def linear_regression_line_through_point 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (hx1 : x1 = 2) (hy1 : y1 = 3) 
  (hx2 : x2 = 5) (hy2 : y2 = 7) 
  (hx3 : x3 = 8) (hy3 : y3 = 9) 
  (hx4 : x4 = 11) (hy4 : y4 = 13) 
  : Prop :=
  ¬(let regression_line := -- define the linear regression line here
    regression_line.contains (6.5, 8))

theorem linear_regression_not_through_6.5_8 :
  linear_regression_line_through_point 2 3 5 7 8 9 11 13 :=
by {
  sorry 
}

end linear_regression_not_through_6_l583_583204


namespace melissa_trips_per_month_l583_583704

theorem melissa_trips_per_month (hours_per_trip : ℕ) (hours_per_year : ℕ) : 
  hours_per_trip = 3 → hours_per_year = 72 → (hours_per_year / hours_per_trip) / 12 = 2 :=
by
  intros h_trip h_year
  have h_trips_per_year : hours_per_year / hours_per_trip = 24, by
    rw [h_trip, h_year]
    norm_num
  have h_trips_per_month : (hours_per_year / hours_per_trip) / 12 = 2, by
    rw h_trips_per_year
    norm_num
  exact h_trips_per_month

end melissa_trips_per_month_l583_583704


namespace smallest_degree_polynomial_l583_583091

theorem smallest_degree_polynomial :
  ∃ (p : Polynomial ℚ), p ≠ 0 ∧
  p.eval (2 - Real.sqrt 3) = 0 ∧
  p.eval (-2 - Real.sqrt 3) = 0 ∧
  p.eval (Real.sqrt 5 - 2) = 0 ∧
  p.eval (2 - Real.sqrt 5) = 0 ∧
  p.degree = 6 := 
sorry

end smallest_degree_polynomial_l583_583091


namespace probability_3a_3b_event_l583_583194

open MeasureTheory

noncomputable def probability_event (μ : Measure (ℝ × ℝ)) : ℝ :=
  μ {p : ℝ × ℝ | p.1 > 1/3 ∧ p.2 > 1/3} / μ univ

theorem probability_3a_3b_event (a b : ℝ) (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) :
  probability_event (volume.restrict (Ioo (0 : ℝ) (1 : ℝ) ×ˢ Ioo (0 : ℝ) (1 : ℝ))) = 4 / 9 :=
  sorry

end probability_3a_3b_event_l583_583194


namespace derivative_at_1_l583_583327

def f (x : ℝ) : ℝ := (1 - 2 * x^3) ^ 10

theorem derivative_at_1 : deriv f 1 = 60 :=
by
  sorry

end derivative_at_1_l583_583327


namespace find_f_l583_583613

-- Define the context of the function and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (a : ℝ)

-- Given condition for the function
def func (x : ℝ) : ℝ := x^2 * a + Real.sin x

-- Its derivative
def func' (x : ℝ) : ℝ := 2 * x * a + Real.cos x

-- Theorem statement that needs to be proven
theorem find_f'_pi_over_three (h : func' (π / 3) = f' (π / 3)) : 
  f' (π / 3) = 3 / (6 - 4 * π) :=
by
  sorry

end find_f_l583_583613


namespace find_nat_number_l583_583138

theorem find_nat_number (N : ℕ) (d : ℕ) (hd : d < 10) (h : N = 5 * d + d) : N = 25 :=
by
  sorry

end find_nat_number_l583_583138


namespace find_x_in_proportion_l583_583793

theorem find_x_in_proportion :
  ∀ (x : ℝ), (0.6 / x = 5 / 8) → x = 0.96 :=
by
  intro x
  intro h
  have h1: 0.6 * 8 = 5 * x,
  { sorry }
  have h2: 4.8 = 5 * x,
  { sorry }
  have h3: 4.8 / 5 = x,
  { sorry }
  exact h3

end find_x_in_proportion_l583_583793


namespace arc_length_correct_l583_583466

-- Define the radius and central angle
def radius : ℝ := 10
def central_angle : ℝ := (2 * Real.pi) / 3

-- Define the circumference of the circle
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define the proportion of the circumference
def proportion_of_circle (angle : ℝ) : ℝ := angle / (2 * Real.pi)

-- Define the length of the arc given the radius and central angle
def arc_length (r : ℝ) (angle : ℝ) : ℝ :=
  (proportion_of_circle angle) * (circumference r)

-- The statement to be proved
theorem arc_length_correct : arc_length radius central_angle = (20 * Real.pi) / 3 := by
  sorry

end arc_length_correct_l583_583466


namespace lines_are_parallel_l583_583223

def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_are_parallel : ∀ x y : ℝ, line1 x = y → line2 x = y → false :=
by
  sorry

end lines_are_parallel_l583_583223


namespace find_n_l583_583699

-- Definitions of the problem conditions
def sum_coefficients (n : ℕ) : ℕ := 4^n
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- The main theorem to be proved
theorem find_n (n : ℕ) (P S : ℕ) (hP : P = sum_coefficients n) (hS : S = sum_binomial_coefficients n) (h : P + S = 272) : n = 4 :=
by
  sorry

end find_n_l583_583699


namespace math_problem_l583_583939

-- Define the conditions
variable {f : ℝ → ℝ}

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)

-- Define the function for non-negative reals
def f_nonneg (x : ℝ) := x^2 + 2 * x

-- Given conditions in Lean
axiom odd_function : is_odd_function f
axiom f_def : ∀ x, 0 ≤ x → f x = f_nonneg x

-- Questions as hypotheses
def f_expression :=
  ∀ x, f x = if x ≥ 0 then x^2 + 2 * x else -x^2 + 2 * x

def inequality_sol_set :=
  {x : ℝ | f x ≥ x + 2} = {x | x ≥ 1}

-- Lean statement to prove
theorem math_problem :
  f_expression ∧ inequality_sol_set :=
by sorry -- proof not required for the statement

end math_problem_l583_583939


namespace equation_has_real_roots_l583_583744

theorem equation_has_real_roots (k : ℝ) : ∀ (x : ℝ), 
  ∃ x, x = k^2 * (x - 1) * (x - 2) :=
by {
  sorry
}

end equation_has_real_roots_l583_583744


namespace solve_g_eq_two_l583_583950

noncomputable def f (x : ℝ) := 2^(-x) + 1

noncomputable def g : ℝ → ℝ
| x := if x ≥ 0 then Real.log (x + 1) / Real.log 2 else f (-x)

theorem solve_g_eq_two : ∃ x : ℝ, g x = 2 ∧ x = 3 := by
  sorry

end solve_g_eq_two_l583_583950


namespace b1f_hex_to_dec_l583_583119

/-- 
  Convert the given hexadecimal digit to its corresponding decimal value.
  -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | _ => 0

/-- 
  Convert a hexadecimal string to a decimal number.
  -/
def hex_string_to_dec (s : String) : Nat :=
  s.foldl (λ acc c => acc * 16 + hex_to_dec c) 0

theorem b1f_hex_to_dec : hex_string_to_dec "B1F" = 2847 :=
by
  sorry

end b1f_hex_to_dec_l583_583119


namespace tangency_points_and_median_concyclic_l583_583668

open EuclideanGeometry

noncomputable def TriangleRightAngle (A B C : Point) : Prop :=
∃ (h : IsRightTriangle ABC), h.angleA = 90

noncomputable def IncircleTangencyPoints (A B C D E : Point) : Prop :=
IsTangencyPoint A ABC D ∧ IsTangencyPoint A ABC E

noncomputable def MedianIntersectsCircumcircle (A B C P Q : Point) : Prop :=
∃ (M : Point), IsMidpoint M B C ∧ LineThroughPoints M (Circumcenter ABC) ∧ 
  (OnCircumcircle ABC P ∧ OnCircumcircle ABC Q)

noncomputable def PointsAreConcyclic (X Y Z W : Point) : Prop :=
∃ (Circle : Circle), OnCircle X Circle ∧ OnCircle Y Circle ∧ OnCircle Z Circle ∧ OnCircle W Circle

theorem tangency_points_and_median_concyclic
  (A B C D E P Q : Point) :
  TriangleRightAngle A B C →
  IncircleTangencyPoints A B C D E →
  MedianIntersectsCircumcircle A B C P Q →
  PointsAreConcyclic D E P Q :=
by
  intros h_tr h_in h_median
  sorry

end tangency_points_and_median_concyclic_l583_583668


namespace option_D_correct_l583_583689

theorem option_D_correct (f : ℕ+ → ℕ) (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (hf : f 4 ≥ 25) : ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
by
  sorry

end option_D_correct_l583_583689


namespace james_found_bills_l583_583291

def initial_money : ℝ := 75
def final_money : ℝ := 135
def bill_value : ℝ := 20

theorem james_found_bills :
  (final_money - initial_money) / bill_value = 3 :=
by
  sorry

end james_found_bills_l583_583291


namespace gcf_fact7_fact8_l583_583143

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the values 7! and 8!
def fact_7 : ℕ := factorial 7
def fact_8 : ℕ := factorial 8

-- Prove that the greatest common factor of 7! and 8! is 7!
theorem gcf_fact7_fact8 : Nat.gcd fact_7 fact_8 = fact_7 :=
by
  sorry

end gcf_fact7_fact8_l583_583143


namespace uniqueness_and_value_of_f_l583_583298

-- Define the function f and given conditions
variables {f : ℝ → ℝ}
axiom f_integer (m : ℤ) : f m = m
axiom f_rational (a b c d : ℤ) (h_ad_bc : abs (a * d - b * c) = 1) (h_c : c > 0) (h_d : d > 0) :
  f ((a + b) / (c + d)) = (f (a / c) + f (b / d)) / 2
axiom f_monotone : ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Uniqueness and value for Golden Ratio inverse
theorem uniqueness_and_value_of_f :
  (∀ g : ℝ → ℝ, (∀ m : ℤ, g m = m) → 
    (∀ a b c d : ℤ, abs (a * d - b * c) = 1 → c > 0 → d > 0 → g ((a + b) / (c + d)) = (g (a / c) + g (b / d)) / 2) → 
    (∀ x y : ℝ, x ≤ y → g x ≤ g y) → g = f) ∧
  (f ((Real.sqrt 5 - 1) / 2) = 
    let a_0 := Int.ofNat ⌊(Real.sqrt 5 - 1) / 2⌋ in
    a_0 - 2 * (∑' (n : ℕ) in Nat.filter odd (range (∘ λ k : ℕ, Int.ofNat ∘ (1 - bit0 1))),
              (λ n, (-1) ^ n / (2 ^ n)) sorry)) :=
sorry

end uniqueness_and_value_of_f_l583_583298


namespace least_integer_l583_583536

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l583_583536


namespace problem_inequality_l583_583925

noncomputable def f (x : ℝ) : ℝ := 1/4 * (x + 1)^2

theorem problem_inequality (n : ℕ) : 
  (∑ k in finset.range n, 1 / f k) > 2 * n / (n + 2) :=
by
  sorry

end problem_inequality_l583_583925


namespace solve_z_l583_583239

theorem solve_z (z : ℂ) (h : z * (1 + complex.I) = 2 * complex.I) : z = 1 + complex.I :=
by
  sorry

end solve_z_l583_583239


namespace ratio_and_fraction_l583_583987

-- Given definitions and conditions
def exists_larger_and_smaller_circle (a b : ℝ) : Prop :=
  -- condition larger circle and smaller circle centered within it
  a > 0 ∧ b > a

def gray_area_twice_smaller_circle (a b : ℝ) : Prop :=
  -- condition: the gray area is equal to twice the area of the smaller circle
  π * b ^ 2 - π * a ^ 2 = 2 * π * a ^ 2

-- To prove the required properties
theorem ratio_and_fraction {a b : ℝ} :
  exists_larger_and_smaller_circle a b →
  gray_area_twice_smaller_circle a b →
  (a / b = 1 / Real.sqrt 3) ∧ (π * a ^ 2 / (π * b ^ 2) = 1 / 3) :=
by
  intro h1 h2
  sorry

end ratio_and_fraction_l583_583987


namespace magnitude_of_vector_cosine_of_angle_l583_583621

-- Problem 1: Magnitude of the vector 2*AB + AC
theorem magnitude_of_vector (A B C : ℝ × ℝ) (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (2, 5))
: let AB := (B.1 - A.1, B.2 - A.2) in
  let AC := (C.1 - A.1, C.2 - A.2) in
  let v := (2 * AB.1 + AC.1, 2 * AB.2 + AC.2) in
  real.sqrt(v.1 ^ 2 + v.2 ^ 2) = 5 * real.sqrt 2 := 
by
  sorry

-- Problem 2: Cosine of the angle between AB and AC
theorem cosine_of_angle (A B C : ℝ × ℝ) (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (2, 5))
: let AB := (B.1 - A.1, B.2 - A.2) in
  let AC := (C.1 - A.1, C.2 - A.2) in
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2 in
  let magnitude_AB := real.sqrt (AB.1 ^ 2 + AB.2 ^ 2) in
  let magnitude_AC := real.sqrt (AC.1 ^ 2 + AC.2 ^ 2) in
  dot_product / (magnitude_AB * magnitude_AC) = 2 * real.sqrt 13 / 13 :=
by
  sorry

end magnitude_of_vector_cosine_of_angle_l583_583621


namespace binomial_expansion_coefficient_x_l583_583661

theorem binomial_expansion_coefficient_x :
  (∃ (c : ℕ), (x : ℝ) → (x + 1/x^(1/2))^7 = c * x + (rest)) ∧ c = 35 := by
  sorry

end binomial_expansion_coefficient_x_l583_583661


namespace white_area_is_69_l583_583021

def area_of_sign : ℕ := 6 * 20

def area_of_M : ℕ := 2 * (6 * 1) + 2 * 2

def area_of_A : ℕ := 2 * 4 + 1 * 2

def area_of_T : ℕ := 1 * 4 + 6 * 1

def area_of_H : ℕ := 2 * (6 * 1) + 1 * 3

def total_black_area : ℕ := area_of_M + area_of_A + area_of_T + area_of_H

def white_area (sign_area black_area : ℕ) : ℕ := sign_area - black_area

theorem white_area_is_69 : white_area area_of_sign total_black_area = 69 := by
  sorry

end white_area_is_69_l583_583021


namespace sum_of_integers_between_neg20_5_and_10_5_l583_583432

theorem sum_of_integers_between_neg20_5_and_10_5 :
  let a := -20
  let l := 10
  let n := (l - a) / 1 + 1
  let S := n / 2 * (a + l)
  S = -155 := by
{
  sorry
}

end sum_of_integers_between_neg20_5_and_10_5_l583_583432


namespace exists_even_a_odd_b_l583_583878

theorem exists_even_a_odd_b (a b : ℕ) (hapos : 0 < a) (hbpos : 0 < b) :
  (∀ n : ℕ, Nat.coprime (a^n + n^b) (b^n + n^a)) ↔ (Even a ∧ Odd b) := 
sorry

end exists_even_a_odd_b_l583_583878


namespace fraction_checked_by_worker_y_l583_583041

variable (P : ℝ) -- Total number of products
variable (f_X f_Y : ℝ) -- Fraction of products checked by worker X and Y
variable (dx : ℝ) -- Defective rate for worker X
variable (dy : ℝ) -- Defective rate for worker Y
variable (dt : ℝ) -- Total defective rate

-- Conditions
axiom f_sum : f_X + f_Y = 1
axiom dx_val : dx = 0.005
axiom dy_val : dy = 0.008
axiom dt_val : dt = 0.0065

-- Proof
theorem fraction_checked_by_worker_y : f_Y = 1 / 2 :=
by
  sorry

end fraction_checked_by_worker_y_l583_583041


namespace hyperbola_eccentricity_range_l583_583069

theorem hyperbola_eccentricity_range
  (a b t : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_condition : a > b) :
  ∃ e : ℝ, e = Real.sqrt (1 + (b / a)^2) ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_range_l583_583069


namespace part1_part2_l583_583176

-- Definition of sets M and N
def M : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = x^2}
def N (a : ℝ) : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 + (y - a)^2 = 1}

-- Intersection of M and N
def A (a : ℝ) : set (ℝ × ℝ) := M ∩ N a

-- Lean statements for the proof problems

theorem part1 (a : ℝ) : (∃ p1 p2 p3, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p1 ∈ A a ∧ p2 ∈ A a ∧ p3 ∈ A a) → a = 1 :=
by
  sorry

theorem part2 (a : ℝ) : A a = ∅ → (a < -1 ∨ a > 5 / 4) :=
by
  sorry

end part1_part2_l583_583176


namespace hypotenuse_length_l583_583001

theorem hypotenuse_length (x y : ℝ) (h1 : y^2 + (x/2)^2 = 13) (h2 : x^2 + (y/2)^2 = 9) : 
  2 * real.sqrt (x^2 + y^2) = 8.4 :=
by
  sorry

end hypotenuse_length_l583_583001


namespace zero_a_and_b_l583_583806

theorem zero_a_and_b (a b : ℝ) (h : a^2 + |b| = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_a_and_b_l583_583806


namespace F_is_ln_l583_583348

noncomputable def F (z : ℝ) : ℝ := sorry -- we state that F is some undefined function

axiom F_property (z : ℝ) (α : ℝ) : F (z^α) = α * F z
axiom ln_definition (z : ℝ) : Real.log z = ln z

theorem F_is_ln (z : ℝ) : F z = Real.log z := sorry

end F_is_ln_l583_583348


namespace exists_one_integer_is_32_l583_583166

theorem exists_one_integer_is_32 (a b c d e : ℕ) :
  (a + b + c + d) / 4 + e = 44 →
  (a + b + c + e) / 4 + d = 38 →
  (a + b + d + e) / 4 + c = 35 →
  (a + c + d + e) / 4 + b = 30 →
  (b + c + d + e) / 4 + a = 29 →
  a = 32 ∨ b = 32 ∨ c = 32 ∨ d = 32 ∨ e = 32 :=
begin
  sorry
end

end exists_one_integer_is_32_l583_583166


namespace right_triangle_perimeter_l583_583476

theorem right_triangle_perimeter (area : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (h_area : area = 120)
  (h_a : a = 24)
  (h_area_eq : area = (1/2) * a * b)
  (h_c : c^2 = a^2 + b^2) :
  a + b + c = 60 :=
by
  sorry

end right_triangle_perimeter_l583_583476


namespace least_positive_integer_l583_583542

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l583_583542


namespace projection_inner_product_l583_583602

-- Definitions of the vectors and their properties
variables {V : Type*} [inner_product_space ℝ V] (a b : V)
variables (h1 : inner a b = 2 * ∥a∥) (h2 : ∥a∥ = 1)

-- Main theorem statement
theorem projection_inner_product : inner a b = 2 :=
by
  -- The proof is omitted
  sorry

end projection_inner_product_l583_583602


namespace man_age_difference_l583_583827

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end man_age_difference_l583_583827


namespace find_value_X_l583_583507

def arithmetic_sequence_vertical (a₂ a₃ a₄ : ℕ) : Prop :=
  a₂ - 4 = a₃ ∧ a₃ - 4 = a₄

def arithmetic_sequence_horizontal (a₁ a₄ a₇ : ℕ) : Prop :=
  a₄ = a₁ - 4 ∧ a₇ = a₁ - 4 * 6

noncomputable def value_X : ℤ :=
  -4

theorem find_value_X : 
  ∀ (a₁ a₂ a₃ a₄ a₇ a₄₂ : ℤ),
    a₁ = 32 ∧ a₂ = 20 ∧ a₃ = 16 ∧ a₄ = 12 ∧ a₄₂ = -8 ∧ 
    arithmetic_sequence_vertical a₂ a₃ a₄ ∧ 
    arithmetic_sequence_horizontal a₁ a₄ a₇ →
    value_X = a₇ := 
by
  intros a₁ a₂ a₃ a₄ a₇ a₄₂ h
  sorry

end find_value_X_l583_583507


namespace proof_problem_l583_583869

open EuclideanGeometry

noncomputable def problem : Prop :=
  ∃ (A B C P D E : Point) (r : ℝ), 
  (∠ BCA > 90) ∧ 
  (PB = PC) ∧
  (PA = r) ∧
  (circumradius (triangle ABC) = r) ∧
  (∃ (circumcircle : Circle), circumcircle.radius = r ∧ circumcircle.center = O ∧ 
   Context.inCircumcircle circumcircle B A C) ∧
  (∃ (PB_perp_bisector : Line), PB_perp_bisector.is_perpendicular_bisector PB ∧
   PB_perp_bisector.intersects_circumcircle circumcircle D E) ∧
  (is_incenter P (triangle CDE))

theorem proof_problem : problem := sorry

end proof_problem_l583_583869


namespace transform_sequence_zero_l583_583580

theorem transform_sequence_zero 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_nonempty : n > 0) :
  ∃ k : ℕ, k ≤ n ∧ ∀ k' ≤ k, ∃ α : ℝ, (∀ i, i < n → |a i - α| = 0) := 
sorry

end transform_sequence_zero_l583_583580


namespace fraction_subtraction_simplified_l583_583890

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end fraction_subtraction_simplified_l583_583890


namespace salary_increase_l583_583242

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 1.16 * S = 406) (h2 : 350 + 350 * P = 420) : P * 100 = 20 := 
by
  sorry

end salary_increase_l583_583242


namespace arithmetic_sequence_a6_l583_583282

theorem arithmetic_sequence_a6 (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, ∃ d, a (n+1) = a n + d)
  (h_sum : a 4 + a 8 = 16) : a 6 = 8 :=
sorry

end arithmetic_sequence_a6_l583_583282


namespace trapezoid_properties_l583_583752

theorem trapezoid_properties
  (a b : ℝ) (h₁ : a = 6) (h₂ : b = 6.25) :
  ∃ d area, d ≈ 10.423 ∧ area = 32 := 
by
  sorry

end trapezoid_properties_l583_583752


namespace smallest_n_l583_583066

-- Definitions related to the problem
def height := 165 -- prism height in cm
def sideLength := 30 -- hexagon side length in cm

-- Distance flown by the fly
def flyDistance : ℝ := Real.sqrt ((2 * sideLength)^2 + height^2)

-- Distance crawled by the ant winding around (n + 1/2) times
def antDistance (n : ℕ) : ℝ := Real.sqrt (((6 * n + 3) * sideLength)^2 + height^2)

-- Condition for the ant's distance being more than 20 times the fly's distance
def condition (n : ℕ) : Prop := antDistance n > 20 * flyDistance

-- The goal is to find the minimum n satisfying the condition
theorem smallest_n : ∃ n : ℕ, condition n ∧ ∀ m : ℕ, m < n → ¬ condition m :=
begin
  use 19,
  -- Here you'd provide the detailed proof that n = 19 is the smallest satisfying the condition
  sorry -- Proof omitted
end

end smallest_n_l583_583066


namespace length_CF_area_triangle_ACF_l583_583442

noncomputable def circle (O : Point) (R : ℝ) : Set Point := {P | dist O P = R}

variables (A B C D F : Point)
variable (R : ℝ) (hR : R = 10)
variable (circle1 : Set Point) (hC1 : circle1 = circle A R)
variable (circle2 : Set Point) (hC2 : circle2 = circle B R)
variable (hAB : A ∈ circle2)
variable (C_on_circle1 : C ∈ circle1)
variable (D_on_circle2 : D ∈ circle2)
variable (hB_on_CD : B ∈ segment C D)
variable (hCAD : ∠ C A D = real.pi / 2)
variable (F_on_perpendicular_B : on_perpendicular F B (line C D))
variable (hBF_eq_BD : dist B F = dist B D)
variable (hBC : dist B C = 12)

-- To prove the length of segment CF is 20
theorem length_CF : dist C F = 20 := sorry

-- To prove the area of triangle ACF is 196
theorem area_triangle_ACF : area A C F = 196 := sorry

end length_CF_area_triangle_ACF_l583_583442


namespace angle_between_vectors_is_pi_div_3_l583_583219

-- Definitions based on the problem conditions
def a : ℝ × ℝ := (1, Real.sqrt 3)
def b (x : ℝ) : ℝ × ℝ := (x, 2 * Real.sqrt 3)
def projection_length (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

-- Main theorem statement
theorem angle_between_vectors_is_pi_div_3 (x : ℝ) (h : projection_length b x a = 2) : 
  ∀ (θ : ℝ), Real.cos θ = (a.1 * (b x).1 + a.2 * (b x).2) / ((Real.sqrt (a.1 ^ 2 + a.2 ^ 2)) * (Real.sqrt ((b x).1 ^ 2 + (b x).2 ^ 2))) → θ = Real.pi / 3 :=
by
  sorry

end angle_between_vectors_is_pi_div_3_l583_583219


namespace triangle_inequality_l583_583296

theorem triangle_inequality
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (BAC : A)
  (BAC_angle : BAC = 45)
  (ACB_angle : (A ∠ C B) > 90)
  (BC : ℝ)
  (CA : ℝ)
  (AB : ℝ) : 
  BC + (Real.sqrt 2 - 1) * CA < AB :=
sorry

end triangle_inequality_l583_583296


namespace number_of_valid_5_element_subsets_l583_583957

def set_A : Finset ℕ := Finset.range 104
def is_isolated_point (S : Finset ℕ) (x : ℕ) : Prop :=
  x ∈ S ∧ ¬ (x - 1 ∈ S) ∧ ¬ (x + 1 ∈ S)

def no_isolated_points (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, ¬ is_isolated_point S x

def count_valid_5_element_subsets : ℕ :=
  (Finset.powersetLen 5 set_A).filter (λ S, no_isolated_points S).card

theorem number_of_valid_5_element_subsets :
  count_valid_5_element_subsets = 10000 :=
sorry

end number_of_valid_5_element_subsets_l583_583957


namespace convert_decimal_to_base5_l583_583870

theorem convert_decimal_to_base5 :
  ∃ n : ℕ, n = 1357 ∧ nat.to_digits 5 n = [2, 0, 4, 1, 2] :=
by sorry

end convert_decimal_to_base5_l583_583870


namespace coeff_x3_of_product_l583_583895

def P (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 5 * x + 6
def Q (x : ℝ) : ℝ := 4 * x^3 + 7 * x^2 + 9 * x + 8

theorem coeff_x3_of_product (x : ℝ) :
  coeff (P x * Q x) 3 = 77 :=
by sorry

end coeff_x3_of_product_l583_583895


namespace find_a_l583_583659

-- Definition of the line l in the parametric form
def line_l (t a : ℝ) : ℝ × ℝ := (t, t - a)

-- Definition of the ellipse C in the parametric form
def ellipse_C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

-- The right vertex of the ellipse C is the point (3, 0)
def right_vertex_of_ellipse_C := (3, 0)

-- The line l passes through the right vertex of the ellipse C
theorem find_a (a : ℝ) : ∃ t : ℝ, line_l t a = right_vertex_of_ellipse_C → a = 3 :=
by
  sorry

end find_a_l583_583659


namespace smallest_integral_length_piece_hypotenuse_l583_583081

theorem smallest_integral_length_piece_hypotenuse :
  (∀ (x : ℕ), (x < 8) → (12 - x) + (20 - x) > (24 - x)) ∧
  (∀ (x : ℕ), (x < 8) → (12 - x) + (24 - x) > (20 - x)) ∧
  (∀ (x : ℕ), (x < 8) → (20 - x) + (24 - x) > (12 - x)) ∧
  (12 - 8) + (20 - 8) ≤ (24 - 8) := 
begin
  sorry
end

end smallest_integral_length_piece_hypotenuse_l583_583081


namespace extremum_point_of_f_no_such_a_b_exist_l583_583576

def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * (Real.log x + b)

theorem extremum_point_of_f (a : ℝ) :
  (∀ b, ∃! x > 0, (f'(a b x) = 0)) ↔ a ∈ Iio 0 :=
sorry

theorem no_such_a_b_exist (a b : ℝ) (x₀ : ℝ) :
  (∃ a b > 0, ∃ x₀ > 0, (f'(a b x₀) = 0) ∧ (f(a b x₀) ∈ Icc (-exp 1 : ℝ) 0)) →
  False :=
sorry

end extremum_point_of_f_no_such_a_b_exist_l583_583576


namespace monomial_exponents_l583_583648

theorem monomial_exponents (m n : ℕ) 
  (h1 : m + 1 = 3)
  (h2 : n - 1 = 3) : 
  m^n = 16 := by
  sorry

end monomial_exponents_l583_583648


namespace wealth_ratio_l583_583871

theorem wealth_ratio 
  (P W : ℝ)
  (hP_pos : 0 < P)
  (hW_pos : 0 < W)
  (pop_A : ℝ := 0.30 * P)
  (wealth_A : ℝ := 0.40 * W)
  (pop_B : ℝ := 0.20 * P)
  (wealth_B : ℝ := 0.25 * W)
  (avg_wealth_A : ℝ := wealth_A / pop_A)
  (avg_wealth_B : ℝ := wealth_B / pop_B) :
  avg_wealth_A / avg_wealth_B = 16 / 15 :=
by
  sorry

end wealth_ratio_l583_583871


namespace find_min_value_l583_583960

open Real

def vec_a (x : ℝ) : ℝ × ℝ :=
  (cos (3 * x / 2), sin (3 * x / 2))

def vec_b (x : ℝ) : ℝ × ℝ :=
  (cos (x / 2), -sin (x / 2))

def f (x : ℝ) : ℝ :=
  let (ax, ay) := vec_a x
  let (bx, by) := vec_b x
  sqrt ((ax + bx)^2 + (ay + by)^2)

theorem find_min_value :
  ∃ x ∈ Icc (-(π / 6)) (π / 4), f x = sqrt 2 := sorry

end find_min_value_l583_583960


namespace part_a_part_b_part_c_l583_583992

noncomputable def rect := {x : ℝ × ℝ // (0 ≤ x.1 ∧ x.1 ≤ 1 ) ∧ (0 ≤ x.2 ∧ x.2 ≤ 1)}
noncomputable def shapes : Type := rect → Prop

variable (shape1 shape2 shape3 shape4 shape5 : shapes)
variable (h1 : ∀ x, shape1 x → x ∈ rect)
variable (h2 : ∀ x, shape2 x → x ∈ rect)
variable (h3 : ∀ x, shape3 x → x ∈ rect)
variable (h4 : ∀ x, shape4 x → x ∈ rect)
variable (h5 : ∀ x, shape5 x → x ∈ rect)
variable (A1 : ∀ x, shape1 x → (1 / 2 : ℝ))
variable (A2 : ∀ x, shape2 x → (1 / 2 : ℝ))
variable (A3 : ∀ x, shape3 x → (1 / 2 : ℝ))
variable (A4 : ∀ x, shape4 x → (1 / 2 : ℝ))
variable (A5 : ∀ x, shape5 x → (1 / 2 : ℝ))

theorem part_a : ∃ (S1 S2 : shapes), S1 ≠ S2 ∧ (∀ x, S1 x ∧ S2 x → (3 / 20 : ℝ)) :=
sorry

theorem part_b : ∃ (S1 S2 : shapes), S1 ≠ S2 ∧ (∀ x, S1 x ∧ S2 x → (1 / 5 : ℝ)) :=
sorry

theorem part_c : ∃ (S1 S2 S3 : shapes), S1 ≠ S2 ∧ S2 ≠ S3 ∧ S1 ≠ S3 ∧ (∀ x, S1 x ∧ S2 x ∧ S3 x → (1 / 20 : ℝ)) :=
sorry

end part_a_part_b_part_c_l583_583992


namespace problem_2_l583_583357

theorem problem_2 (x y : ℚ) (h : x^2 - 2 * y - real.sqrt 2 * y = 17 - 4 * real.sqrt 2) : 
  2 * x + y = 14 ∨ 2 * x + y = -6 :=
sorry

end problem_2_l583_583357


namespace proposition_A_proposition_B_proposition_C_proposition_D_false_propositions_l583_583848

theorem proposition_A (a b : ℝ) : ¬ (a + b ≥ 2 * Real.sqrt (a * b)) :=
sorry

theorem proposition_B (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : ¬ (x + y = 12) :=
sorry

theorem proposition_C (x : ℝ) : x^2 + 4 / x^2 ≥ 4 :=
begin
  calc
    x^2 + 4 / x^2 ≥ 2 * Real.sqrt(x^2 * (4 / x^2)) : by apply add_le_add
    ...              = 4                            : by simp
end

theorem proposition_D (a b : ℝ) (hab : 0 < a * b) : b / a + a / b ≥ 2 :=
begin
  calc 
    b / a + a / b ≥ 2 * Real.sqrt((b / a) * (a / b)) : by apply add_le_add
    ...              = 2                            : by simp
end

theorem false_propositions : 
  (∀ a b : ℝ, ¬ (a + b ≥ 2 * Real.sqrt (a * b))) ∧ 
  (∀ x y : ℝ, (0 < x) → (0 < y) → (1 / x + 9 / y = 1) → ¬(x + y = 12)) ∧ 
  (∀ x : ℝ, x^2 + 4 / x^2 ≥ 4) ∧
  (∀ a b : ℝ, (0 < a * b) → (b / a + a / b ≥ 2)) :=
by 
  refine ⟨proposition_A, proposition_B, proposition_C, proposition_D⟩

end proposition_A_proposition_B_proposition_C_proposition_D_false_propositions_l583_583848


namespace decimal_difference_l583_583797

theorem decimal_difference : (0.650 : ℝ) - (1 / 8 : ℝ) = 0.525 := by
  sorry

end decimal_difference_l583_583797


namespace Sierra_pies_proper_l583_583419

variable (Bill Adam Sierra : ℕ)

-- Conditions
def condition1 : Prop := Adam = Bill + 3
def condition2 : Prop := Sierra = 2 * Bill
def condition3 : Prop := Bill + Adam + Sierra = 27

-- The conjecture we want to prove
theorem Sierra_pies_proper : condition1 Bill Adam Sierra ∧ condition2 Bill Adam Sierra ∧ condition3 Bill Adam Sierra → Sierra = 12 := by
  sorry

end Sierra_pies_proper_l583_583419


namespace largest_of_12_consecutive_with_avg_25_point_5_l583_583382

theorem largest_of_12_consecutive_with_avg_25_point_5
  (a : ℕ)
  (h_avg : (a + (a + 1) + … + (a + 11)) / 12 = 25.5) :
  a + 11 = 31 :=
sorry

end largest_of_12_consecutive_with_avg_25_point_5_l583_583382


namespace candidate_lost_by_l583_583056

noncomputable def candidate_votes (total_votes : ℝ) := 0.35 * total_votes
noncomputable def rival_votes (total_votes : ℝ) := 0.65 * total_votes

theorem candidate_lost_by (total_votes : ℝ) (h : total_votes = 7899.999999999999) :
  rival_votes total_votes - candidate_votes total_votes = 2370 :=
by
  sorry

end candidate_lost_by_l583_583056


namespace passenger_on_train_B_sees_train_A_in_7_l583_583025

/-- Definition of the lengths and the times based on the problem statement --/
def len_A : ℕ := 150
def len_B : ℕ := 200
def time_B : ℕ := 10
def time_A := 7.5

/-- The proof problem stating the original question and answer -/
theorem passenger_on_train_B_sees_train_A_in_7.5_seconds :
  ∀ (lenA lenB timeB : ℕ) (timeA : ℝ), 
    lenA = 150 → lenB = 200 → timeB = 10 → timeA = (lenA * timeB / lenB) →
      timeA = 7.5 :=
by
  intros lenA lenB timeB timeA hlenA hlenB htimeB htimeA
  sorry

end passenger_on_train_B_sees_train_A_in_7_l583_583025


namespace circle_area_greater_than_hexagon_area_l583_583352

theorem circle_area_greater_than_hexagon_area (h : ℝ) (r : ℝ) (π : ℝ) (sqrt3 : ℝ) (ratio : ℝ) : 
  (h = 1) →
  (r = sqrt3 / 2) →
  (π > 3) →
  (sqrt3 > 1.7) →
  (ratio = (π * sqrt3) / 6) →
  ratio > 0.9 :=
by
  intros h_eq r_eq pi_gt sqrt3_gt ratio_eq
  -- Proof omitted
  sorry

end circle_area_greater_than_hexagon_area_l583_583352


namespace thm_1_thm_2_thm_3_l583_583515

variables (a b c : ℝ) (k : ℕ) (h_pos : k > 0)

def op_⊕ (a b : ℝ) : ℝ := k * max a b
def op_⊖ (a b : ℝ) : ℝ := min (a^2) (b^2)

theorem thm_1 : op_⊕ a b k = op_⊕ b a k := by sorry

theorem thm_2 : op_⊕ a (op_⊕ b c k) k = op_⊕ (op_⊕ a b k) c k := by sorry

theorem thm_3 : op_⊖ a (op_⊕ b c k) ≠ op_⊕ (op_⊖ a b) (op_⊖ a c) k := by sorry

end thm_1_thm_2_thm_3_l583_583515


namespace sufficient_but_not_necessary_condition_l583_583996

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≤ 0) : 
  a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_l583_583996


namespace ratio_of_areas_l583_583266

variable (s : ℝ)

def area_square := s^2
def area_rectangle := 1.2 * s * (0.85 * s)
def ratio := area_rectangle s / area_square s

theorem ratio_of_areas (hs : s > 0) : ratio s = 51 / 50 := by
  -- Summary of the conditions that were given.
  -- Prove that the ratio of areas is 51 / 50.
  sorry

end ratio_of_areas_l583_583266


namespace slope_of_CD_l583_583740

-- Given circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the line whose slope needs to be found
def line (x y : ℝ) : Prop := 22*x - 12*y - 33 = 0

-- State the proof problem
theorem slope_of_CD : ∀ x y : ℝ, circle1 x y → circle2 x y → line x y ∧ (∃ m : ℝ, m = 11/6) :=
by sorry

end slope_of_CD_l583_583740


namespace speed_of_train_l583_583483

theorem speed_of_train 
  (length_of_train : ℕ)
  (length_of_bridge : ℕ)
  (time_to_cross : ℕ)
  (h1 : length_of_train = 145)
  (h2 : length_of_bridge = 230)
  (h3 : time_to_cross = 30) :
  let total_distance := length_of_train + length_of_bridge in
  let speed_m_per_s := total_distance / time_to_cross in
  let speed_km_per_hr := speed_m_per_s * 3.6 in
  speed_km_per_hr = 45 := 
by
  -- The proof will go here
  sorry

end speed_of_train_l583_583483


namespace intersection_A_B_l583_583301

-- Define set A
def A : Set ℤ := {-1, 1, 2, 3, 4}

-- Define set B with the given condition
def B : Set ℤ := {x : ℤ | 1 ≤ x ∧ x < 3}

-- The main theorem statement showing the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} :=
    sorry -- Placeholder for the proof

end intersection_A_B_l583_583301


namespace no_intersection_of_q_ax_q_b_l583_583312

theorem no_intersection_of_q_ax_q_b
  (p : Polynomial ℝ)
  (hdeg : p.degree = 3)
  (a b : ℝ)
  (ha_ne_hb : a ≠ b)
  (qa qb : Polynomial ℝ)
  (hqa : ∀ a, ∃! qa, (qa.degree ≤ 2 ∧ ∃ g : Polynomial ℝ, p = qa + (X - C a)^3 * g)) :
  ¬ ∃ x : ℝ, qa.eval x = qb.eval x := sorry

end no_intersection_of_q_ax_q_b_l583_583312


namespace num_points_where_segments_intersect_l583_583709

noncomputable def circleSegmentsIntersectAtLeastNPoints (n : ℕ) : Prop :=
  ∀ (points : Finset (ℕ × ℕ))
    (redPairs bluePairs : Finset (ℕ × ℕ)),
    (points.card = 4 * n) ∧
    (n > 0) ∧
    (∀ (i : ℕ), i ∈ points → (i % 2 = 0) ∨ (i % 2 = 1)) ∧
    (∀ (p : ℕ × ℕ), p ∈ redPairs ∧ p ∈ bluePairs → p.1 ≠ p.2) ∧
    (∀ (x y z : ℕ × ℕ), x ≠ y → y ≠ z → x ≠ z →
      ¬ (x.1 = y.1 ∧ y.1 = z.1 ∧ x.2 = y.2 ∧ y.2 = z.2))
    → ∃ (intersections : ℕ), intersections ≥ n

-- Proof is omitted
theorem num_points_where_segments_intersect (n : ℕ) :
  circleSegmentsIntersectAtLeastNPoints n :=
sorry

end num_points_where_segments_intersect_l583_583709


namespace SetC_not_right_angled_triangle_l583_583851

theorem SetC_not_right_angled_triangle :
  ¬ (7^2 + 24^2 = 26^2) :=
by 
  have h : 7^2 + 24^2 ≠ 26^2 := by decide
  exact h

end SetC_not_right_angled_triangle_l583_583851


namespace problem1_problem2_l583_583859

-- Problem 1
theorem problem1 (α : ℝ) (h : 2 * Real.sin α - Real.cos α = 0) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10 / 3 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) :
  (Real.sin x ^ 3 + Real.sin x * Real.cos x ^ 2) / (1 - Real.tan x) = 7 * Real.sqrt 2 / 60 :=
sorry

end problem1_problem2_l583_583859


namespace sqrt_defined_range_l583_583014

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end sqrt_defined_range_l583_583014


namespace fred_earnings_l583_583295

-- Conditions as definitions
def initial_amount : ℕ := 23
def final_amount : ℕ := 86

-- Theorem to prove
theorem fred_earnings : final_amount - initial_amount = 63 := by
  sorry

end fred_earnings_l583_583295


namespace f_monotonic_decreasing_l583_583328

noncomputable def f (x : ℝ) := sqrt 2 * cos (2 * x)

theorem f_monotonic_decreasing : ∀ x y : ℝ, 0 < x → x < y → y < π / 2 → f x > f y :=
by
  sorry

end f_monotonic_decreasing_l583_583328


namespace sequence_factorial_divides_product_l583_583452

theorem sequence_factorial_divides_product (k : ℤ) (h_k : k > 2)
  (a : ℕ → ℤ) (h_a0 : a 0 = 0) (h_recur : ∀ n, a (n + 1) = k * a n - a (n - 1)) :
  ∀ (m : ℕ), 0 < m → (2 * m)! ∣ (List.prod (List.map a (List.range (3 * m + 1)))) :=
by
  sorry

end sequence_factorial_divides_product_l583_583452


namespace sequence_b2_values_l583_583075

-- Definitions of the sequence conditions.
def sequence_condition (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → b (n + 2) = abs (b (n + 1) - b n)

-- Definitions used in the problem.
def b1 := 1001

-- Lean proof problem statement.
theorem sequence_b2_values : 
  ∃ b : ℕ → ℕ, b 1 = b1 ∧ b 2 < b1 ∧ b 2004 = 1 ∧ sequence_condition b ∧ (∃ n : ℕ, n = 360) :=
sorry

end sequence_b2_values_l583_583075


namespace students_in_miss_evans_class_l583_583658

theorem students_in_miss_evans_class
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (contribution_per_student : ℕ)
  (remaining_contribution : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : contribution_per_student = 4)
  (h4 : remaining_contribution = total_contribution - class_funds)
  (h5 : num_students = remaining_contribution / contribution_per_student)
  : num_students = 19 :=
sorry

end students_in_miss_evans_class_l583_583658


namespace gcd_7_8_fact_l583_583159

-- Define factorial function in lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the GCD function
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define specific factorial values
def f7 := fact 7
def f8 := fact 8

-- Theorem stating the gcd of 7! and 8!
theorem gcd_7_8_fact : gcd f7 f8 = 5040 := by
  sorry

end gcd_7_8_fact_l583_583159


namespace slope_of_line_l583_583080

-- Define the points
def point1 : (ℝ × ℝ) := (4, -2)
def point2 : (ℝ × ℝ) := (-3, 4)

-- Define the slope function between two points
def slope (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Statement to prove
theorem slope_of_line : slope point1 point2 = -6 / 7 := by
  sorry

end slope_of_line_l583_583080


namespace biff_break_even_night_hours_l583_583495

-- Define the constants and conditions
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def lunch_cost : ℝ := 8
def dinner_cost : ℝ := 10
def accommodation_cost : ℝ := 35

def total_expenses_without_wifi : ℝ := ticket_cost + snacks_cost + headphones_cost + lunch_cost + dinner_cost + accommodation_cost

def earnings_per_hour : ℝ := 12
def wifi_cost_day : ℝ := 2
def wifi_cost_night : ℝ := 1

-- Define the total expenses with wifi cost variable
def total_expenses (D N : ℝ) : ℝ := total_expenses_without_wifi + (wifi_cost_day * D) + (wifi_cost_night * N)

-- Define the total earnings
def total_earnings (D N : ℝ) : ℝ := earnings_per_hour * (D + N)

-- Prove that the minimum number of hours Biff needs to work at night to break even is 8 hours
theorem biff_break_even_night_hours :
  ∃ N : ℕ, N = 8 ∧ total_earnings 0 N ≥ total_expenses 0 N := 
by 
  sorry

end biff_break_even_night_hours_l583_583495


namespace keith_remaining_cards_l583_583294

-- Definitions and conditions
def initial_cards := 0
def new_cards := 8
def total_cards_after_purchase := initial_cards + new_cards
def remaining_cards := total_cards_after_purchase / 2

-- Proof statement (in Lean, the following would be a theorem)
theorem keith_remaining_cards : remaining_cards = 4 := sorry

end keith_remaining_cards_l583_583294


namespace total_marbles_correct_l583_583702

-- Define the number of marbles Mary has
def MaryYellowMarbles := 9
def MaryBlueMarbles := 7
def MaryGreenMarbles := 6

-- Define the number of marbles Joan has
def JoanYellowMarbles := 3
def JoanBlueMarbles := 5
def JoanGreenMarbles := 4

-- Define the total number of marbles for Mary and Joan combined
def TotalMarbles := MaryYellowMarbles + MaryBlueMarbles + MaryGreenMarbles + JoanYellowMarbles + JoanBlueMarbles + JoanGreenMarbles

-- We want to prove that the total number of marbles is 34
theorem total_marbles_correct : TotalMarbles = 34 := by
  -- The proof is skipped with sorry
  sorry

end total_marbles_correct_l583_583702


namespace problem_statement_l583_583876

-- Define the alternating sum with sign change at perfect squares
def alternating_sum_square_sign : ℕ → ℤ
| 0 => 0
| 1 => -1
| (n + 1) =>
  let k := nat.floor (real.sqrt (n + 1).nat_abs)
  if (k * k = n + 1) then
    (-1) * alternating_sum_square_sign n
  else
    if (k * k < n + 1) then
      alternating_sum_square_sign n + n + 1
    else
      alternating_sum_square_sign n - n - 1

-- Proof statement
theorem problem_statement :
  alternating_sum_square_sign 729 = 729 := sorry

end problem_statement_l583_583876


namespace abs_floor_value_l583_583132

theorem abs_floor_value : (Int.floor (|(-56.3: Real)|)) = 56 := 
by
  sorry

end abs_floor_value_l583_583132


namespace probability_two_tails_one_head_l583_583650

theorem probability_two_tails_one_head :
  let outcomes := {x | x = [1,1,0] ∨ x = [1,0,1] ∨ x = [0,1,1]} in
  let total_outcomes := 3 / 8 in
  let possible_outcomes := 8 in
  (card outcomes) / possible_outcomes = total_outcomes :=
  sorry

end probability_two_tails_one_head_l583_583650


namespace pizza_consumption_fraction_l583_583439

theorem pizza_consumption_fraction : 
  let first_trip := 2 * (1 / 4)
  let second_trip := 2 * (1 / 8)
  let third_trip := 2 * (1 / 16)
  let fourth_trip := 2 * (1 / 32)
  in first_trip + second_trip + third_trip + fourth_trip = 15 / 16 :=
by
  let first_trip := 2 * (1 / 4)
  let second_trip := 2 * (1 / 8)
  let third_trip := 2 * (1 / 16)
  let fourth_trip := 2 * (1 / 32)
  have h_sum : first_trip + second_trip + third_trip + fourth_trip = 15 / 16 := sorry
  exact h_sum

end pizza_consumption_fraction_l583_583439


namespace partial_fraction_decomposition_product_l583_583899

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℚ,
    (A + 2) * (A - 3) *
    (B - 2) * (B - 3) *
    (C - 2) * (C + 2) = x^2 - 12 ∧
    (A = -2) ∧
    (B = 2/5) ∧
    (C = 3/5) ∧
    (A * B * C = -12/25) :=
  sorry

end partial_fraction_decomposition_product_l583_583899


namespace f_zero_t_f_t_nonneg_f_is_identity_l583_583742

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, f x ∈ ℝ
axiom f_not_identically_zero : ∃ x : ℝ, f x ≠ 0
axiom f_condition : ∀ (m n : ℝ), f m * f n = m * f (n / 2) + n * f (m / 2)

theorem f_zero : f 0 = 0 := sorry

theorem t_f_t_nonneg (t : ℝ) : t * f t ≥ 0 := sorry

theorem f_is_identity (x : ℝ) : f x = x := sorry

end f_zero_t_f_t_nonneg_f_is_identity_l583_583742


namespace find_omega_correct_omega_finder_l583_583947

noncomputable def omega_finder (ω : ℝ) :=
∀ (f : ℝ → ℝ) (α β : ℝ),
  (∀ x : ℝ, f x = (Real.sin (ω * x - π / 6) + 1 / 2)) ∧
  (f α = -1 / 2) ∧
  (f β = 1 / 2) ∧
  (abs (α - β) = 3 * π / 4) →
  ω = 2 / 3

theorem find_omega_correct_omega_finder : ∃ ω : ℝ, omega_finder ω :=
begin
  use 2 / 3,
  unfold omega_finder,
  intros f α β h,
  sorry -- Proof will be provided here
end

end find_omega_correct_omega_finder_l583_583947


namespace quadratic_real_roots_l583_583603

theorem quadratic_real_roots (k : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + 4 * x + k - 1 = 0) → ∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → 
  k ≤ 3 :=
by
  intro h
  have h_discriminant : 16 - 8 * k >= 0 := sorry
  linarith

end quadratic_real_roots_l583_583603


namespace gcf_fact7_fact8_l583_583144

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the values 7! and 8!
def fact_7 : ℕ := factorial 7
def fact_8 : ℕ := factorial 8

-- Prove that the greatest common factor of 7! and 8! is 7!
theorem gcf_fact7_fact8 : Nat.gcd fact_7 fact_8 = fact_7 :=
by
  sorry

end gcf_fact7_fact8_l583_583144


namespace smallest_prime_with_digit_sum_18_l583_583780

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_18 : ∃ p : ℕ, Prime p ∧ 18 = sum_of_digits p ∧ (∀ q : ℕ, (Prime q ∧ 18 = sum_of_digits q) → p ≤ q) :=
by
  sorry

end smallest_prime_with_digit_sum_18_l583_583780


namespace least_positive_integer_factorial_4725_l583_583777

noncomputable def least_n_for_factorial (k : ℤ) : ℕ :=
  if h : ∃ m : ℕ, k ∣ (nat.factorial m) then 
    nat.find h 
  else 0

theorem least_positive_integer_factorial_4725 (n : ℕ) :
  least_n_for_factorial 4725 = 15 :=
by sorry

end least_positive_integer_factorial_4725_l583_583777


namespace simplify_fraction_l583_583434

theorem simplify_fraction : 
  1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_fraction_l583_583434


namespace group_selection_l583_583988

theorem group_selection (m f : ℕ) (h1 : m + f = 8) (h2 : (m * (m - 1) / 2) * f = 30) : f = 3 :=
sorry

end group_selection_l583_583988


namespace ana_multiplied_numbers_l583_583089

theorem ana_multiplied_numbers (x : ℕ) (y : ℕ) 
    (h_diff : y = x + 202) 
    (h_mistake : x * y - 1000 = 288 * x + 67) :
    x = 97 ∧ y = 299 :=
sorry

end ana_multiplied_numbers_l583_583089


namespace inequality_solution_l583_583894

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x - 3)^2 ≥ 0} = (Set.Iic (-1) ∪ Set.Ici 1) :=
by
  sorry

end inequality_solution_l583_583894


namespace gcf_7fact_8fact_l583_583148

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l583_583148


namespace gcf_fact7_fact8_l583_583145

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the values 7! and 8!
def fact_7 : ℕ := factorial 7
def fact_8 : ℕ := factorial 8

-- Prove that the greatest common factor of 7! and 8! is 7!
theorem gcf_fact7_fact8 : Nat.gcd fact_7 fact_8 = fact_7 :=
by
  sorry

end gcf_fact7_fact8_l583_583145


namespace peyton_manning_total_yards_l583_583716

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end peyton_manning_total_yards_l583_583716


namespace find_ellipse_equation_find_t_range_l583_583931

noncomputable def ellipse_c (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem find_ellipse_equation (a b : ℝ) (h_ab : a > b ∧ b > 0) 
  (h_focal_distance : 2 * b^2 = a^2 - b^2) :
  ellipse_c x y (Real.sqrt (2)) 1 =
  (x^2 / 2) + y^2 = 1 :=
sorry

theorem find_t_range (t : ℝ) (h_ab : a > b ∧ b > 0) 
  (h_focal_distance : 2 * b^2 = a^2 - b^2) 
  (h_line_ellipse_intersection : ∃ A B : ℝ × ℝ,
      A.1 ≠ B.1 ∧ ellipse_c A.1 A.2 (Real.sqrt (2)) 1 ∧ ellipse_c B.1 B.2 (Real.sqrt (2)) 1 ∧ 
      A.1 - A.2 + t = 0 ∧ B.1 - B.2 + t = 0)
  (h_mid_point_condition : ∀ A B : ℝ × ℝ, 
      A.1 ≠ B.1 ∧ ellipse_c A.1 A.2 (Real.sqrt (2)) 1 ∧ ellipse_c B.1 B.2 (Real.sqrt (2)) 1 ∧ 
      A.1 - A.2 + t = 0 ∧ B.1 - B.2 + t = 0 → 
      ((-(A.1 + B.1) / 3)^2 + ((A.2 + B.2) / 3)^2) ≥ 10/9) :
  (-Real.sqrt 3 < t ∧ t ≤ -Real.sqrt 2) ∨ (Real.sqrt 2 ≤ t ∧ t < Real.sqrt 3) :=
sorry

end find_ellipse_equation_find_t_range_l583_583931


namespace average_snack_sales_per_ticket_l583_583044

theorem average_snack_sales_per_ticket :
  let cracker_price := 2.25
  let beverage_price := 1.50
  let chocolate_price := 1.00
  let cracker_count := 3
  let beverage_count := 4
  let chocolate_count := 4
  let total_tickets := 6
  let total_sales := cracker_count * cracker_price + beverage_count * beverage_price + chocolate_count * chocolate_price
  let average_sales := total_sales / total_tickets
  average_sales ≈ 2.79 :=
by
  -- Proof omitted
  sorry

end average_snack_sales_per_ticket_l583_583044


namespace symmetry_M3_M_wrt_DA_l583_583664

variables {A B C D M M1 M2 M3 : Type}
variables [has_coords A] [has_coords B] [has_coords C] [has_coords D] [has_coords M] [has_coords M1]
          [has_coords M2] [has_coords M3]
variables (xA yA xB yB xC yC xD yD x y xD1 yD1 : ℝ)

def midpoint (P Q : Type) [has_coords P] [has_coords Q] : Type :=
  (xP + xQ) / 2, (yP + yQ) / 2

def symmetric_point (M : Type) (mid_pt : Type) [has_coords M] [has_coords mid_pt] : Type :=
  2 * mid_pt - M

-- Conditions:
def midpoint_AB := midpoint A B
def midpoint_BC := midpoint B C
def midpoint_CD := midpoint C D
def midpoint_DA := midpoint D A

def M1 := symmetric_point M midpoint_AB
def M2 := symmetric_point M1 midpoint_BC
def M3 := symmetric_point M2 midpoint_CD

-- Proof Statement:
theorem symmetry_M3_M_wrt_DA :
  symmetric_point M midpoint_DA = M3 :=
sorry

end symmetry_M3_M_wrt_DA_l583_583664


namespace terminal_sides_positions_l583_583978

def in_third_quadrant (θ : ℝ) (k : ℤ) : Prop :=
  (180 + k * 360 : ℝ) < θ ∧ θ < (270 + k * 360 : ℝ)

theorem terminal_sides_positions (θ : ℝ) (k : ℤ) :
  in_third_quadrant θ k →
  ((2 * θ > 360 + 2 * k * 360 ∧ 2 * θ < 540 + 2 * k * 360) ∨
   (90 + k * 180 < θ / 2 ∧ θ / 2 < 135 + k * 180) ∨
   (2 * θ = 360 + 2 * k * 360) ∨ (2 * θ = 540 + 2 * k * 360) ∨ 
   (θ / 2 = 90 + k * 180) ∨ (θ / 2 = 135 + k * 180)) :=
by
  intro h
  sorry

end terminal_sides_positions_l583_583978


namespace find_m_value_l583_583197

-- Define the conditions
def quadratic_has_real_roots (m : ℝ) : Prop :=
  let Δ := (2 * m - 1)^2 - 4 * m^2 in Δ ≥ 0

def correct_m_value (m : ℝ) : Prop :=
  let quadratic_solution_product := (x1 + 1) * (x2 + 1) in
  quadratic_solution_product = 3 → m = -3

theorem find_m_value (m : ℝ) :
  quadratic_has_real_roots m →
  correct_m_value m :=
  sorry

end find_m_value_l583_583197


namespace largest_integer_l583_583171

variable (x y z w : ℤ)

# Check and set conditions:
def cond1 : Prop := x + y + z = 163
def cond2 : Prop := x + y + w = 178
def cond3 : Prop := x + z + w = 184
def cond4 : Prop := y + z + w = 194

theorem largest_integer 
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3)
  (h4 : cond4) :
  w = 77 :=
sorry

end largest_integer_l583_583171


namespace probability_cos_between_zero_and_half_l583_583832

noncomputable def probability_cos_interval : ℝ :=
  let interval_length := (π/2) - (-π/2)
  let valid_length := (π/2 - π/3 + π/3 - (-π/2))
  valid_length / interval_length

theorem probability_cos_between_zero_and_half :
  probability_cos_interval = 1 / 3 :=
sorry

end probability_cos_between_zero_and_half_l583_583832


namespace min_value_a_plus_b_l583_583856

theorem min_value_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a^2 ≥ 8 * b) (h4 : b^2 ≥ a) : a + b ≥ 6 := by
  sorry

end min_value_a_plus_b_l583_583856


namespace wheel_radius_l583_583082
noncomputable def radius_of_wheel 
  (distance_miles : ℝ) 
  (revolutions : ℕ) 
  (mile_to_feet : ℝ)
  (feet_to_inches : ℝ) : ℝ :=
  let total_distance_in_inches := distance_miles * mile_to_feet * feet_to_inches
  let circumference := total_distance_in_inches / (revolutions : ℝ)
  let radius := circumference / (2 * Real.pi)
  radius

theorem wheel_radius (h : radius_of_wheel 120 2300 5280 12 ≈ 526.348) : 
  radius_of_wheel 120 2300 5280 12 ≈ 526.348 := 
  by sorry

end wheel_radius_l583_583082


namespace projection_vector_l583_583600

variables {V : Type*} [inner_product_space ℝ V] -- Assume V is a real inner product space
variables (a b : V)  -- a and b are vectors in this space

-- Conditions
variables (angle_ab : real.angle a b = 2 * real.pi / 3)
variables (orthogonal_condition : (a + b) ⬝ a = 0)

theorem projection_vector (h : (a + b) ⬝ a = 0) (ha : a ≠ 0) (hb : b ≠ 0) :
  (proj b a) = - (1 / 4 : ℝ) • b :=
sorry  -- Provide the proof here. Proof steps are not necessary for this exercise.

end projection_vector_l583_583600


namespace fraction_subtraction_equivalence_l583_583888

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end fraction_subtraction_equivalence_l583_583888


namespace card_average_value_l583_583479

theorem card_average_value (n : ℕ) (h : (2 * n + 1) / 3 = 2023) : n = 3034 :=
sorry

end card_average_value_l583_583479


namespace dave_hits_seven_l583_583725

-- Define the friends competing in the contest
inductive Friend
| Alice | Ben | Cindy | Dave | Ellen | Frank
deriving DecidableEq

-- Define the scores they achieved
def scores : Friend → ℕ
| Friend.Alice := 18
| Friend.Ben := 9
| Friend.Cindy := 13
| Friend.Dave := 8
| Friend.Ellen := 12
| Friend.Frank := 17

-- Define the regions scores range
def region_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Prove that Dave hits the region worth 7 points
theorem dave_hits_seven :
  ∃ (r1 r2 : ℕ), r1 ≠ r2 ∧ r1 ∈ region_scores ∧ r2 ∈ region_scores ∧
  r1 + r2 = scores Friend.Dave ∧ (r1 = 7 ∨ r2 = 7) :=
sorry

end dave_hits_seven_l583_583725


namespace min_value_f_eq_neg4_l583_583593

def f (x b : ℝ) := (x^2 - 2 * x + 1 / b) / (2 * x^2 + 2 * x + 1)

theorem min_value_f_eq_neg4 (b : ℝ) : 
  ∃ x : ℝ, f x b = -4 :=
by 
  sorry

end min_value_f_eq_neg4_l583_583593
