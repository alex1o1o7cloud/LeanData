import Mathlib

namespace work_completion_time_x_l309_309834

open Real

noncomputable theory

def time_taken_x (R_x R_y : ℝ) := 1 / R_x

def combined_rate (R_x R_y : ℝ) := R_x + R_y

theorem work_completion_time_x :
  ∀ (W : ℝ), 
    let R_y := W / 45 in
    let R_xy := W / 11.25 in
    let R_x := R_xy - R_y in
    time_taken_x R_x R_y = 15 :=
by
  intros
  simp [time_taken_x, combined_rate, div_eq_inv_mul]
  sorry

end work_completion_time_x_l309_309834


namespace total_employees_l309_309043

def part_time_employees : ℕ := 2047
def full_time_employees : ℕ := 63109
def contractors : ℕ := 1500
def interns : ℕ := 333
def consultants : ℕ := 918

theorem total_employees : 
  part_time_employees + full_time_employees + contractors + interns + consultants = 66907 := 
by
  -- proof goes here
  sorry

end total_employees_l309_309043


namespace find_n_when_x_and_y_are_given_l309_309267

theorem find_n_when_x_and_y_are_given :
  ∃ n, (x = 3) ∧ (y = -4) ∧ (n = x - y^(x - y)) → n = 16387 :=
by
  sorry

end find_n_when_x_and_y_are_given_l309_309267


namespace trigonometric_inequality_l309_309720

theorem trigonometric_inequality (n : ℕ) (α : Fin n → ℝ) 
(α_pos : ∀ i, 0 < α i ∧ α i < π / 2) :
  ( ∑ i in Finset.univ, 1 / Real.sin (α i) ) *
  ( ∑ i in Finset.univ, 1 / Real.cos (α i) ) ≤ 
  2 * ( ∑ i in Finset.univ, 1 / Real.sin (2 * α i) )^2 :=
by
  sorry

end trigonometric_inequality_l309_309720


namespace sphere_surface_area_of_cube_vertices_on_sphere_l309_309849

theorem sphere_surface_area_of_cube_vertices_on_sphere (a : ℝ) (ha : a = 1) :
  let diameter := a * Real.sqrt 3 in
  let radius := diameter / 2 in
  let surface_area := 4 * Real.pi * radius^2 in
  surface_area = 3 * Real.pi :=
by
  sorry

end sphere_surface_area_of_cube_vertices_on_sphere_l309_309849


namespace midpoint_parallel_l309_309677

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309677


namespace tan_prob_correct_l309_309439

noncomputable def tan_probability : ℝ := sorry

theorem tan_prob_correct :
  (∃ p : ℝ, p = 1 / 3 ∧
            ∀ (x : ℝ), 
            (-π / 2 < x ∧ x < π / 2) →
            (p = ((π / 2 - π / 6) / π))) :=
begin
  use 1 / 3,
  split,
  { refl },
  { intro x,
    intro h,
    sorry
  }
end

end tan_prob_correct_l309_309439


namespace right_triangle_medians_l309_309706

theorem right_triangle_medians (A B C M N : Point)
    (hM : M = midpoint B C) (hN : N = midpoint A C)
    (hA : right_triangle A B C)
    (hAM : distance A M = 5)
    (hBN : distance B N = 3 * sqrt 5) :
    distance A B = 8 :=
by
  sorry

end right_triangle_medians_l309_309706


namespace ocean_depth_at_base_of_cone_l309_309848

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

noncomputable def submerged_height_fraction (total_height volume_fraction : ℝ) : ℝ :=
  total_height * (volume_fraction)^(1/3)

theorem ocean_depth_at_base_of_cone (total_height radius : ℝ) 
  (above_water_volume_fraction : ℝ) : ℝ :=
  let above_water_height := submerged_height_fraction total_height above_water_volume_fraction
  total_height - above_water_height

example : ocean_depth_at_base_of_cone 10000 2000 (3 / 5) = 1566 := by
  sorry

end ocean_depth_at_base_of_cone_l309_309848


namespace part1_part2_case1_part2_case2_l309_309969

-- Part 1: Prove that if f(x) is increasing on x ∈ [1, +∞), then a ≤ 3/2
theorem part1 (a : ℝ) :
  (∀ x, x ≥ 1 → (3*x^2 - 4*a*x + 3) ≥ 0) → a ≤ 3 / 2 :=
sorry

-- Part 2: Given a = sqrt(3), find min and max values of f(x) on [-2, sqrt(3)]
theorem part2_case1 :
  let a := Real.sqrt 3 in
  let f : ℝ → ℝ := λ x, x^3 - 2*a*x^2 + 3*x in
  f (-2) = -14 - 8 * Real.sqrt 3 ∧
  f (Real.sqrt 3) = 0 ∧
  f (Real.sqrt 3 / 3) = 4 * Real.sqrt 3 / 9 :=
sorry

-- Part 3: Given a = -sqrt(3), find min and max values of f(x) on [-2, -sqrt(3)]
theorem part2_case2 :
  let a := -Real.sqrt 3 in
  let f : ℝ → ℝ := λ x, x^3 - 2*a*x^2 + 3*x in
  f (-2) = -14 + 8 * Real.sqrt 3 ∧
  f (-Real.sqrt 3) = 0 :=
sorry

end part1_part2_case1_part2_case2_l309_309969


namespace minutes_in_year_correct_l309_309055

-- Define the conditions
def days_in_year : ℕ := 360
def hours_in_day : ℕ := 24
def minutes_in_hour : ℕ := 60

-- Define the target total minutes in scientific notation with three significant figures
def total_minutes := days_in_year * hours_in_day * minutes_in_hour
def target_minutes_sf := 5.18 * 10^5

theorem minutes_in_year_correct : (total_minutes : ℝ) = target_minutes_sf :=
by
  -- Placeholder for the proof
  sorry

end minutes_in_year_correct_l309_309055


namespace sphere_surface_area_ratio_l309_309541

theorem sphere_surface_area_ratio (a r R : ℝ) 
  (h1 : R ^ 2 = (a * sqrt 3 / 3) ^ 2 + r ^ 2)
  (h2 : r = a * sqrt 3 / 6)
  (h3 : R > 0 ∧ r > 0):
  4 * π * R ^ 2 / (4 * π * r ^ 2) = 5 :=
by
  sorry

end sphere_surface_area_ratio_l309_309541


namespace total_earnings_l309_309365

variables (g : ℝ) (V1 : ℝ := 100) (E_min : ℝ := 0.01) (E_max : ℝ := 0.05)

def Vn (n : ℕ) : ℝ :=
  V1 * (1 + g / 100) ^ (n - 1)

def S6 : ℝ :=
  (Finset.range 6).sum (λ n, Vn g (n + 1))

def V7 : ℝ :=
  2 * S6 g

def T : ℝ :=
  S6 g + V7 g

def E_avg : ℝ :=
  (E_min + E_max) / 2

def E_total : ℝ :=
  T g * E_avg E_min E_max

theorem total_earnings (g : ℝ) :
  E_total g = T g * 0.03 :=
by
  sorry

end total_earnings_l309_309365


namespace solve_for_x_l309_309404

theorem solve_for_x (x : ℝ) (h : 0.4 * x = (1 / 3) * x + 110) : x = 1650 :=
by sorry

end solve_for_x_l309_309404


namespace midpoint_parallel_l309_309671

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309671


namespace train_cost_difference_l309_309867

-- Setting up the constants
def T_train : ℝ := by sorry
def T_bus : ℝ := 1.75
def total_cost : ℝ := 9.85

-- The main statement to prove
theorem train_cost_difference :
  T_train - T_bus = 6.35 :=
by
  have h1 : T_train + T_bus = total_cost, from sorry,
  have h2 : T_bus = 1.75, from sorry,
  have h3 : total_cost = 9.85, from sorry,
  sorry

end train_cost_difference_l309_309867


namespace quadratic_negative_roots_prob_eq_two_thirds_l309_309306

noncomputable def quadratic_negative_roots_probability : ℝ :=
∫ p in set.Icc (0 : ℝ) 5, if (p ∈ set.Icc (2/3) 1 ∪ set.Ici 2) then 1 else 0 / (5 - 0)

theorem quadratic_negative_roots_prob_eq_two_thirds :
  quadratic_negative_roots_probability = 2 / 3 := 
sorry

end quadratic_negative_roots_prob_eq_two_thirds_l309_309306


namespace circle_meets_parabola_exactly_three_points_l309_309901

noncomputable def circle_parabola_intersection (b : ℝ) : Prop :=
  let f := λ x : ℝ, x^4 - (4 * b - 1) * x^2 + 3 * b^2
  let x_vals := {x : ℝ | x = 0 ∨ x^2 = 4 * b - 1}
  x_vals.card = 3 ∧ ∀ x ∈ x_vals, f x = 0

theorem circle_meets_parabola_exactly_three_points (b : ℝ) : circle_parabola_intersection b ↔ b > 1 / 4 :=
  sorry

end circle_meets_parabola_exactly_three_points_l309_309901


namespace right_triangle_exists_l309_309389

theorem right_triangle_exists (a b c : Nat) : 
  (a = 5 ∧ b = 12 ∧ c = 13) →
  (a^2 + b^2 = c^2) :=
by
  intro h
  cases h with
  | intro ha hb hc =>
      rw [ha, hb, hc]
      calc
        5^2 + 12^2 
        _ = 25 + 144 := by rfl
        _ = 169 := by rfl
        _ = 13^2 := by rfl
      sorry

end right_triangle_exists_l309_309389


namespace power_of_point_l309_309067

variable {Q M A B : Point}
variable {R m : ℝ}
variable {a b : Vec}

-- Definitions based on given conditions
def is_center (Q : Point) (s : Sphere) : Prop := s.center = Q
def has_radius (s : Sphere) (R : ℝ) : Prop := s.radius = R
def distance_from_point (M Q : Point) (m : ℝ) : Prop := dist M Q = m
def is_diameter (A B : Point) (s : Sphere) : Prop := dist A B = 2 * s.radius

-- Vectors representing \(\overrightarrow{MA}\) and \(\overrightarrow{MB}\)
def vec_a (M A : Point) : Vec := \overrightarrow{M A}
def vec_b (M B : Point) : Vec := \overrightarrow{M B}

-- Main theorem statement
theorem power_of_point {Q M A B : Point} {R m : ℝ}
  (s : Sphere)
  (h_center : is_center Q s)
  (h_radius : has_radius s R)
  (h_dist : distance_from_point M Q m)
  (h_diam : is_diameter A B s)
  (h_vec_a : vec_a M A = a)
  (h_vec_b : vec_b M B = b) :
  (a • b) = (m^2 - R^2) :=
sorry

end power_of_point_l309_309067


namespace integer_solutions_eq_l309_309911

theorem integer_solutions_eq (x y z : ℤ) :
  (x + y + z) ^ 5 = 80 * x * y * z * (x ^ 2 + y ^ 2 + z ^ 2) ↔
  ∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨ (x = -a ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = -a) ∨ (x = -a ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = -a) ∨ (x = 0 ∧ y = -a ∧ z = a) :=
by sorry

end integer_solutions_eq_l309_309911


namespace triangle_area_eq_twelve_l309_309814

theorem triangle_area_eq_twelve (a b c : ℝ)
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 6) : 
  let s := (a + b + c) / 2 in
  √(s * (s - a) * (s - b) * (s - c)) = 12 :=
by sorry

end triangle_area_eq_twelve_l309_309814


namespace solve_inequalities_l309_309313

-- Define the inequalities and their expected solution sets
def inequality1 := ∀ x : ℝ, (2 * x - 4) * (x - 5) < 0 ↔ 2 < x ∧ x < 5

def inequality2 := ∀ x : ℝ, 3 * x^2 + 5 * x + 1 > 0 ↔ (x < (-5 - real.sqrt 13) / 6 ∨ x > (-5 + real.sqrt 13) / 6)

def inequality3 := ∀ x : ℝ, -x^2 + x < 2 ↔ true

def inequality4 := ∀ x : ℝ, 7 * x^2 + 5 * x + 1 ≤ 0 ↔ false

def inequality5 := ∀ x : ℝ, 4 * x ≥ 4 * x^2 + 1 ↔ x = 1 / 2

-- Include a summary to bundle all inequalities into one definition
def all_inequalities := 
  inequality1 ∧ 
  inequality2 ∧ 
  inequality3 ∧ 
  inequality4 ∧ 
  inequality5

-- Statement to be proved
theorem solve_inequalities : all_inequalities :=
  by
  sorry

end solve_inequalities_l309_309313


namespace bottles_left_on_shelf_l309_309350

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l309_309350


namespace number_of_children_l309_309523

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l309_309523


namespace boat_speed_l309_309346

theorem boat_speed (v : ℝ) : 
  let rate_current := 7
  let distance := 35.93
  let time := 44 / 60
  (v + rate_current) * time = distance → v = 42 :=
by
  intro h
  sorry

end boat_speed_l309_309346


namespace baron_munchausen_truth_l309_309064

def sum_of_digits_squared (n : ℕ) : ℕ :=
  (n.digits 10).sum (λ d, d ^ 2)

theorem baron_munchausen_truth : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a.digits.length = 10 ∧ 
    b.digits.length = 10 ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    (a - sum_of_digits_squared a) = 
    (b - sum_of_digits_squared b) :=
begin
  use 10^9 + 8,
  use 10^9 + 9,
  split,
  { exact ne_of_lt (nat.lt_succ_self (10^9 + 8)) }, -- proof of a ≠ b
  split,
  { exact rfl }, -- proof of a is 10 digits long
  split,
  { exact rfl }, -- proof of b is 10 digits long
  split,
  { norm_num }, -- proof a % 10 ≠ 0
  split,
  { norm_num }, -- proof b % 10 ≠ 0
  { sorry },
end

end baron_munchausen_truth_l309_309064


namespace num_comics_bought_l309_309303

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end num_comics_bought_l309_309303


namespace problem_part1_problem_part2_l309_309183

open Real

def f (x : ℝ) (a : ℝ) : ℝ := log (2*x + 1) / log 2 + a*x
def g (x : ℝ) (a : ℝ) : ℝ := f x a + x
def h (x : ℝ) (m : ℝ) : ℝ := x^2 - 2*x + m
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem problem_part1 (a : ℝ) : 
  (∀ x : ℝ, f x a = f (-x a)) → a = -1/2 :=
by
  sorry

theorem problem_part2 (a : ℝ) (m : ℝ) :
  (∀ x1 ∈ set.Icc 0 4, ∃ x2 ∈ set.Icc 0 5, g x1 a ≥ h x2 m) → m ≤ 2 :=
by
  sorry

end problem_part1_problem_part2_l309_309183


namespace positive_real_solution_count_l309_309096

open Real

noncomputable def f (x : ℝ) : ℝ := x^8 + 5 * x^7 + 10 * x^6 + 729 * x^5 - 379 * x^4
noncomputable def g (x : ℝ) : ℝ := x^4 + 5 * x^3 + 10 * x^2 + 729 * x - 379

theorem positive_real_solution_count : ∃! x > 0, f x = 0 :=
by {
  -- proof required here,
  sorry
}

end positive_real_solution_count_l309_309096


namespace fourth_grade_students_l309_309882

theorem fourth_grade_students:
  (initial_students = 35) →
  (first_semester_left = 6) →
  (first_semester_joined = 4) →
  (first_semester_transfers = 2) →
  (second_semester_left = 3) →
  (second_semester_joined = 7) →
  (second_semester_transfers = 2) →
  final_students = initial_students - first_semester_left + first_semester_joined - second_semester_left + second_semester_joined :=
  sorry

end fourth_grade_students_l309_309882


namespace prove_x_value_l309_309089

theorem prove_x_value :
  let S₁ := ∑' n : ℕ, (1/3)^n,
      S₂ := ∑' n : ℕ, (-1/3)^n,
      S₃ := 1 / (1 - 1/x)
  in S₁ * S₂ = S₃ → x = 9 := by
  let S₁ := ∑' n : ℕ, (1/3)^n
  have h₁ : S₁ = 3/2 :=
    sorry
  let S₂ := ∑' n : ℕ, (-1/3)^n
  have h₂ : S₂ = 3/4 :=
    sorry
  let S₃ := 1 / (1 - 1/x)
  have h₃ : S₁ * S₂ = S₃ :=
    sorry
  exact (by
    sorry : S₁ * S₂ = 9/8 → x = 9)
    {subst h₁, subst h₂, subst h₃, 
     sorry}

end prove_x_value_l309_309089


namespace percentage_of_green_eyed_brunettes_l309_309462

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end percentage_of_green_eyed_brunettes_l309_309462


namespace plane_representation_is_correct_l309_309330

-- Definitions based on the problem description
def is_plane_representation (plane: String) (notation: String) : Prop :=
  notation = plane

-- Given conditions
def option_A : String := "AC"
def option_B : String := "Plane AC"
def option_C : String := "AB"
def option_D : String := "Plane AB"

-- The correct answer
def correct_option : String := "Plane AC"

-- Statement we need to prove
theorem plane_representation_is_correct : 
  is_plane_representation correct_option option_B :=
by
  sorry

end plane_representation_is_correct_l309_309330


namespace compute_x_l309_309087

theorem compute_x :
  (∑ n : ℕ, (1 / 3 : ℝ) ^ n) * (∑ n : ℕ, (-1 / 3 : ℝ) ^ n) = ∑ n : ℕ, (1 / (9 : ℝ)) ^ n :=
begin
  sorry
end

end compute_x_l309_309087


namespace bottles_left_l309_309357

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l309_309357


namespace number_of_children_l309_309521

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l309_309521


namespace interval_of_increase_find_side_c_l309_309965

noncomputable def f (x : ℝ) : ℝ := 
  √3 * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

def interval_increasing (k : ℤ) : Set ℝ := 
  { x | k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 }

theorem interval_of_increase (k : ℤ) : 
  ∀ x : ℝ, (interval_increasing k) x ↔ f x > f (x - Real.pi / 3) ∧ f x < f (x + Real.pi / 6) :=
sorry

theorem find_side_c (A : ℝ) (a b : ℝ) (hA : f A = 1 / 2) (ha : a = √17) (hb : b = 4) : 
  ∃ c : ℝ, c = 2 + √5 :=
sorry

end interval_of_increase_find_side_c_l309_309965


namespace muffins_total_l309_309248

theorem muffins_total (num_friends : ℕ) (num_muffins_per_person : ℕ) (total_people : ℕ) :
  total_people = num_friends + 1 → num_muffins_per_person = 4 → num_friends = 4 → total_people * num_muffins_per_person = 20 :=
by 
  intro h1 h2 h3
  rw [h3] at h1
  rw [h1, h2]
  norm_num
  sorry

end muffins_total_l309_309248


namespace closest_point_on_line_is_correct_l309_309914

-- Define vector structure and required operations
structure Vector3 :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def point_on_line (t : ℝ) : Vector3 :=
  ⟨3 - t, 2 + 4 * t, 2 - 2 * t⟩

def target_point : Vector3 := ⟨5, 1, 6⟩

def closest_point : Vector3 := ⟨47 / 19, 78 / 19, 18 / 19⟩

theorem closest_point_on_line_is_correct : ∃ t : ℝ, point_on_line t = closest_point :=
by
  sorry

end closest_point_on_line_is_correct_l309_309914


namespace bottles_left_on_shelf_l309_309354

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l309_309354


namespace no_increase_in_probability_l309_309036

def hunter_two_dogs_probability (p : ℝ) : ℝ :=
  let both_correct := p * p
  let one_correct := 2 * p * (1 - p) / 2
  both_correct + one_correct

theorem no_increase_in_probability (p : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ 1) :
  hunter_two_dogs_probability p = p :=
by
  sorry

end no_increase_in_probability_l309_309036


namespace side_length_of_square_l309_309759

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l309_309759


namespace max_intersection_points_three_circles_two_lines_l309_309820

theorem max_intersection_points_three_circles_two_lines :
  let circles := 3
  let lines := 2
  (max_intersection_points circles lines = 19) :=
by
  let circles := 3
  let lines := 2
  exact sorry

end max_intersection_points_three_circles_two_lines_l309_309820


namespace Euclid1976_PartA_Problem8_l309_309314

theorem  Euclid1976_PartA_Problem8 (a b c m n : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h4 : Polynomial.eval (-a) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0)
  (h5 : Polynomial.eval (-b) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0) :
  n = -1 :=
sorry

end Euclid1976_PartA_Problem8_l309_309314


namespace problem_statement_l309_309550

-- Define the necessary components
variables {A B C O : Point}
variable {x y : ℝ}

-- Given conditions
axiom circumcenter_O (h : O = circumcenter A B C) : 
  AB = 8 ∧ AC = 12 ∧ ∠ A = π / 3 ∧ (AO = x • (AB : Vector) + y • (AC : Vector))

theorem problem_statement : 
  6 * x + 9 * y = 5 := 
sorry

end problem_statement_l309_309550


namespace side_length_of_square_l309_309752

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309752


namespace projection_of_v_on_w_l309_309922

open Real EuclideanSpace

def v : ℝ^3 := ![-5, 10, -15]
def w : ℝ^3 := ![10, -20, 30]

theorem projection_of_v_on_w : (proj w v) = v := 
by
  sorry

end projection_of_v_on_w_l309_309922


namespace hunter_strategy_does_not_increase_probability_l309_309029

theorem hunter_strategy_does_not_increase_probability (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1) :
  let p2 := p * p in
  let split_prob := p * (1 - p) in
  let combined_prob := p2 + split_prob in
  combined_prob = p :=
by
  let p := p
  let p2 := p * p
  let split_prob := p * (1 - p)
  let combined_prob := p2 + split_prob
  have : combined_prob = p := sorry
  exact this

end hunter_strategy_does_not_increase_probability_l309_309029


namespace certain_number_is_50_l309_309836

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end certain_number_is_50_l309_309836


namespace find_x_from_log_condition_l309_309984

theorem find_x_from_log_condition (x : ℝ) (h : 5^(Real.log x / Real.log 10) = 25) : x = 100 :=
by
  sorry

end find_x_from_log_condition_l309_309984


namespace unique_y_for_diamond_l309_309494

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond :
  ∃! y : ℝ, diamond 4 y = 21 :=
by
  sorry

end unique_y_for_diamond_l309_309494


namespace main_theorem_l309_309238

variable (A B C D E F T S O P Q R : Type)
variable [metric_space A] [metric_space B] [metric_space C]
variable [metric_space D] [metric_space E] [metric_space F]
variable [metric_space T] [metric_space S] [metric_space O]
variable [metric_space P] [metric_space Q] [metric_space R]

axiom in_quad (ABCD : quadrilateral A B C D) :
  E ∈ segment AD ∧ F ∈ segment BC

axiom ratio_condition :
  ∀ a b c d : ℝ, a / b = c / d ↔ a * d = b * c

axiom intersections (CD_EF : ∃ T, T ∈ extension_line CD ∧ T ∈ extension_line EF) :
  ∃ S, S ∈ extension_line BA ∧ S ∈ extension_line EF

axiom circumcircles (triangles : ∀ O P Q R,
  O ∈ circumcircle (triangle AES) ∧
  P ∈ circumcircle (triangle BFS) ∧
  Q ∈ circumcircle (triangle CFT) ∧
  R ∈ circumcircle (triangle DET))

theorem main_theorem :
  ∃ K, K ∈ circumcircle (triangle AES) ∧
  K ∈ circumcircle (triangle BFS) ∧
  K ∈ circumcircle (triangle CFT) ∧
  K ∈ circumcircle (triangle DET) ∧
  quadrilateral_similar OPQR ABCD := sorry

end main_theorem_l309_309238


namespace product_bn_eq_fraction_l309_309927

noncomputable def b (n : ℕ) : ℚ :=
  (↑(n + 1)^3 - 1) / (↑n * (↑n^3 - 1))

theorem product_bn_eq_fraction :
  ∀ n ≥ 5, ∏ k in finset.range (96) (∑ m in (finset.range 96).map nat.succ_pred,  b (m + 5)) = 24847144 / (100!) :=
  by sorry

end product_bn_eq_fraction_l309_309927


namespace angle_between_vectors_l309_309192

open Real

noncomputable def vec_angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.acos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2)))

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : (b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 1))
  (h2 : Real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 1) : 
  vec_angle_between a b = π / 2 :=
begin
  sorry
end

end angle_between_vectors_l309_309192


namespace hunting_dog_strategy_l309_309032

theorem hunting_dog_strategy (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let single_dog_success_prob := p,
      both_dogs_same_path_prob := p^2,
      one_dog_correct_prob := p * (1 - p),
      combined_prob := both_dogs_same_path_prob + one_dog_correct_prob
  in combined_prob = single_dog_success_prob := 
by
  sorry

end hunting_dog_strategy_l309_309032


namespace simplify_expression_l309_309733

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end simplify_expression_l309_309733


namespace side_length_of_square_l309_309745

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l309_309745


namespace maximum_value_sine_squares_l309_309124

theorem maximum_value_sine_squares (n : ℕ) (θ : Fin n → ℝ)
  (h1 : ∀ i, 0 ≤ θ i ∧ θ i ≤ π)
  (h2 : ∑ i, θ i = π) :
  ∃ S, S = (∑ i in Finset.univ, (sin (θ i)) ^ 2) ∧ S ≤ 9 / 4 :=
sorry

end maximum_value_sine_squares_l309_309124


namespace kevin_number_of_digits_written_l309_309400

theorem kevin_number_of_digits_written :
  let multiples := list.filter (λ n, n % 3 = 0) (list.range 101)
  let single_digit_multiples := list.filter (λ n, n < 10) multiples
  let two_digit_multiples := list.filter (λ n, n >= 10) multiples
  let single_digit_count := list.length single_digit_multiples
  let two_digit_count := list.length two_digit_multiples
  single_digit_count + 2 * two_digit_count = 63 := by
  sorry

end kevin_number_of_digits_written_l309_309400


namespace area_rhombus_is_168_l309_309803

-- Definition of the rhombus' area given diagonals d1 and d2
def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- Given conditions: diagonals of the rhombus
def d1 : ℝ := 14
def d2 : ℝ := 24

-- Proof goal: the area of the rhombus is 168 square meters.
theorem area_rhombus_is_168 : area_of_rhombus d1 d2 = 168 :=
by sorry

end area_rhombus_is_168_l309_309803


namespace max_elements_in_T_l309_309090

def T : Set ℤ := {x | x > 0} -- Placeholder for the actual set T, needs further refinement

axiom T_cond1 (y : ℤ) (hy : y ∈ T) : 
  ∃ m : ℤ, ∃ M : ℤ, (|T| - 1) > 0 ∧ (M - y) % ((|T| - 1).toNat) = 0

axiom T_contains_1 : 1 ∈ T

axiom T_max_element : ∃ t ∈ T, t = 3003 ∧ ∀ x ∈ T, x ≤ 3003

theorem max_elements_in_T : (|T| = 23) :=
by
  sorry

end max_elements_in_T_l309_309090


namespace Pythagorean_triple_proof_l309_309698

variable (a b c x y : ℤ)

-- Conditions: a, b, c form a Pythagorean triple.
def is_pythagorean_triple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Show that the given expressions are perfect squares.
def are_perfect_squares (a b c : ℤ) : Prop :=
  ∃ s1 s2 s3 s4 : ℤ,
    2 * (c - a) * (c - b) = s1^2 ∧
    2 * (c - a) * (c + b) = s2^2 ∧
    2 * (c + a) * (c - b) = s3^2 ∧
    2 * (c + a) * (c + b) = s4^2

-- The equations are solvable in integers.
def equations_solvable (c : ℤ) : Prop :=
  ∃ x y : ℤ,
    x + y + Int.sqrt (2 * x * y) = c ∧
    x + y - Int.sqrt (2 * x * y) = c

-- Specific solutions for c = 13.
def specific_solution_c13 : Prop :=
  (2 * 13^2 + 2 * 5 * 13 + 2 * 12 * 13 + 2 * 5 * 12) = (5 + 12 + 13)^2 ∧
  (2 * 13^2 - 2 * 5 * 13 - 2 * 12 * 13 + 2 * 5 * 12) = (13 - 5 - 12)^2 ∧
  (2 * 13^2 + 13^2 - 2 * 12 * 13 - 2 * 5 * 12) = (13 + 5 - 12)^2 ∧
  (x + y + Int.sqrt(2 * x * y) = 13 ∧ x + y - Int.sqrt(2 * x * y) = 13)

-- Specific solutions for c = 50.
def specific_solution_c50 : Prop :=
  (2 * 50^2 + 2 * 14 * 50 + 2 * 48 * 50 + 2 * 14 * 48) = (14 + 48 + 50)^2 ∧
  (2 * 50^2 - 2 * 14 * 50 - 2 * 48 * 50 + 2 * 14 * 48) = (50 - 14 - 48)^2 ∧
  (2 * 50^2 + 50^2 - 2 * 48 * 50 - 2 * 14 * 48) = (50 + 14 - 48)^2 ∧
  (x + y + Int.sqrt(2 * x * y) = 50 ∧ x + y - Int.sqrt(2 * x * y) = 50)

-- A Lean theorem that encapsulates all conditions and statements.
theorem Pythagorean_triple_proof : ∀ a b c : ℤ,
  is_pythagorean_triple a b c →
  are_perfect_squares a b c ∧
  equations_solvable c ∧
  (c = 13 → specific_solution_c13) ∧
  (c = 50 → specific_solution_c50) :=
by sorry

end Pythagorean_triple_proof_l309_309698


namespace laurissa_height_mode_median_mean_l309_309335

theorem laurissa_height_mode_median_mean :
  ∃ L : ℕ, (list.mode [135, 160, 170, 175, L] = list.median [135, 160, 170, 175, L] ∧
            list.median [135, 160, 170, 175, L] = (list.sum [135, 160, 170, 175, L] / 5)) ↔ L = 160 :=
by
  sorry

end laurissa_height_mode_median_mean_l309_309335


namespace midpoint_parallel_l309_309674

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309674


namespace max_ratio_1099_l309_309928

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_ratio_1099 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (sum_of_digits n : ℚ) / n ≤ (sum_of_digits 1099 : ℚ) / 1099 :=
by
  intros n hn
  sorry

end max_ratio_1099_l309_309928


namespace cyclic_sum_inequality_l309_309642

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (h_product : a * b * c = 1) :
  (a^6 / ((a - b) * (a - c)) + b^6 / ((b - c) * (b - a)) + c^6 / ((c - a) * (c - b)) > 15) := 
by sorry

end cyclic_sum_inequality_l309_309642


namespace negation_of_universal_l309_309551

theorem negation_of_universal (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, 2^x > 0) ↔ (∃ x : ℝ, 2^x ≤ 0) :=
by { sorry }

end negation_of_universal_l309_309551


namespace rotate90CW_correct_l309_309429

/-- Define the shapes involved --/
inductive Shape
| square : Shape
| circle : Shape
| pentagon : Shape

/-- Define the large circle and its evenly spaced shapes --/
structure LargeCircle where
  shape_positions: List Shape
  h_length: shape_positions.length = 3

/-- Define the 90 degree clockwise rotation on a large circle --/
def rotate90CW (circle: LargeCircle) : LargeCircle :=
  LargeCircle.mk
    ([circle.shape_positions.getLast, circle.shape_positions.head!, circle.shape_positions.getLast(1)])
    sorry  -- proof of length being 3 is skipped (1-based index last but one)

/-- The theorem statement: After 90 degree clockwise rotation, shapes move one position in a closed cycle --/
theorem rotate90CW_correct (circle: LargeCircle)
  (h_shapes : circle.shape_positions = [Shape.square, Shape.circle, Shape.pentagon]) :
  rotate90CW circle = LargeCircle.mk [Shape.pentagon, Shape.square, Shape.circle] :=
begin
  sorry
end

end rotate90CW_correct_l309_309429


namespace side_length_of_square_l309_309753

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309753


namespace people_lineup_l309_309230

/-- Theorem: Given five people, where the youngest person cannot be on the first or last position, 
we want to prove that there are exactly 72 ways to arrange them in a straight line. -/
theorem people_lineup (p : Fin 5 → ℕ) 
  (hy : ∃ i : Fin 5, ∀ j : Fin 5, i ≠ j → p i < p j) 
  (h_pos : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4)
  : (∑ x in ({1, 2, 3, 4} : Finset (Fin 5)), 4 * 3 * 2 * 1) = 72 := by
  -- The proof is omitted.
  sorry

end people_lineup_l309_309230


namespace danny_initial_bottle_caps_l309_309895

theorem danny_initial_bottle_caps 
  (bottle_caps_lost : ℕ) 
  (current_bottle_caps : ℕ) :
  bottle_caps_lost = 66 →
  current_bottle_caps = 25 →
  current_bottle_caps + bottle_caps_lost = 91 := 
by
  intros bl cb hl hc
  rw [hc, hl]
  exact rfl

end danny_initial_bottle_caps_l309_309895


namespace polynomial_value_l309_309985

-- Define the conditions as Lean definitions
def condition (x : ℝ) : Prop := x^2 + 2 * x + 1 = 4

-- State the theorem to be proved
theorem polynomial_value (x : ℝ) (h : condition x) : 2 * x^2 + 4 * x + 5 = 11 :=
by
  -- Proof goes here
  sorry

end polynomial_value_l309_309985


namespace ratio_of_b_l309_309167

variables {a1 b1 a2 b2 c : ℝ}
variables {P F1 F2 : ℝ → ℝ → Prop}

-- Assuming the given conditions
def is_ellipse (a1 b1 : ℝ) (P : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, P x y ∧ (x^2 / a1^2 + y^2 / b1^2 = 1)

def is_hyperbola (a2 b2 : ℝ) (P : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, P x y ∧ (x^2 / a2^2 - y^2 / b2^2 = 1)

def common_foci (F1 F2 : ℝ → ℝ → Prop) (a1 b1 a2 c : ℝ) : Prop :=
  ∃ F1_x F1_y F2_x F2_y : ℝ, 
    F1 F1_x F1_y ∧ F2 F2_x F2_y ∧ 
    (F1_x^2 + F1_y^2 = c^2) ∧ (F2_x^2 + F2_y^2 = c^2) ∧ 
    a1^2 - b1^2 = c^2 ∧ a2^2 + b2^2 = c^2

def angle (P F1 F2 : ℝ → ℝ → Prop) : Prop :=
  ∃ θ : ℝ, θ = π / 3

-- Prove the desired ratio
theorem ratio_of_b (h1 : a1 > b1) (h2 : b1 > 0) 
                   (h3 : a2 > 0) (h4 : b2 > 0)
                   (h5 : is_ellipse a1 b1 P)
                   (h6 : is_hyperbola a2 b2 P)
                   (h7 : common_foci F1 F2 a1 b1 a2 c)
                   (h8 : angle P F1 F2) :
  b1 / b2 = sqrt 3 :=
sorry

end ratio_of_b_l309_309167


namespace daniel_age_is_13_l309_309883

-- Define Aunt Emily's age
def aunt_emily_age : ℕ := 48

-- Define Brianna's age as a third of Aunt Emily's age
def brianna_age : ℕ := aunt_emily_age / 3

-- Define that Daniel's age is 3 years less than Brianna's age
def daniel_age : ℕ := brianna_age - 3

-- Theorem to prove Daniel's age is 13 given the conditions
theorem daniel_age_is_13 :
  brianna_age = aunt_emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age = 13 :=
  sorry

end daniel_age_is_13_l309_309883


namespace count_three_digit_perfect_square_palindromes_l309_309433

-- Define a three-digit number as a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

-- Define a three-digit number as a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The set of three-digit numbers
def three_digit_numbers : set ℕ :=
  {n | 100 ≤ n ∧ n < 1000}

-- The set of three-digit perfect square palindromes
def three_digit_perfect_square_palindromes :=
  {n | three_digit_numbers n ∧ is_perfect_square n ∧ is_palindrome n}

-- The statement to prove
theorem count_three_digit_perfect_square_palindromes :
  {n | three_digit_perfect_square_palindromes n}.finite.to_finset.card = 3 :=
begin
  sorry
end

end count_three_digit_perfect_square_palindromes_l309_309433


namespace fraction_shaded_semicircle_l309_309331

-- Definitions for the problem
variables (r R : ℝ) -- radius of the smaller and larger semicircles
variable (O : Type) -- center of the semicircles
variable rectangle_OABC : r > 0 ∧ R > 0 -- rectangle having sides of lengths r and R
variable OC_eq : (r^2 + R^2 = (sqrt(r^2 + R^2)) ^ 2) -- OC as diagonal of rectangle
variable CD_eq_CE : (∀ OC, CD = CE) -- lengths of CD and CE are equal

-- Proof to find the fraction of the large semicircle that is shaded
theorem fraction_shaded_semicircle : 
  r = (3 * R) / 5 →
  (π * (r^2) / 2) / (π * (R^2) / 2) = 9 / 25 :=
sorry

end fraction_shaded_semicircle_l309_309331


namespace music_class_uncool_parents_l309_309359

theorem music_class_uncool_parents:
  ∀ (total students coolDads coolMoms bothCool : ℕ),
  total = 40 →
  coolDads = 25 →
  coolMoms = 19 →
  bothCool = 8 →
  (total - (bothCool + (coolDads - bothCool) + (coolMoms - bothCool))) = 4 :=
by
  intros total coolDads coolMoms bothCool h_total h_dads h_moms h_both
  sorry

end music_class_uncool_parents_l309_309359


namespace overlap_area_rhombus_l309_309811

theorem overlap_area_rhombus (alpha : ℝ) (h_alpha_pos : 0 < sin alpha) :
    ∃ A : ℝ, A = 1 / sin alpha :=
sorry

end overlap_area_rhombus_l309_309811


namespace f_is_odd_f_max_value_on_interval_g_inequality_k_range_l309_309403

-- Define f(x) as given
def f (x : ℝ) : ℝ := -x^3 + x

-- Prove that f is an odd function
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) := by
  sorry

-- Find the maximum value of f(x) on [-1, m] where m > -1
theorem f_max_value_on_interval (m : ℝ) (hm : m > -1) : 
  ∃ max_value, ∀ x ∈ set.Icc (-1) m, f(x) ≤ max_value := by
  sorry

-- Define g(x) as given and find the range of k for the inequality
def g (x : ℝ) : ℝ := -- Need specific definition here, omit for abstraction since not provided
theorem g_inequality (k : ℝ) (hk : k > 0) : 
  ∀ x ∈ set.Ioo 0 (2*k), g(x) * g(2*k - x) ≥ (-k)^2 := by
  sorry

-- Prove range of k
theorem k_range : ∃ (k : ℝ), k > 0 := by
  sorry

end f_is_odd_f_max_value_on_interval_g_inequality_k_range_l309_309403


namespace half_ears_kernels_l309_309195

theorem half_ears_kernels (stalks ears_per_stalk total_kernels : ℕ) (X : ℕ)
  (half_ears : ℕ := stalks * ears_per_stalk / 2)
  (total_ears : ℕ := stalks * ears_per_stalk)
  (condition_e1 : stalks = 108)
  (condition_e2 : ears_per_stalk = 4)
  (condition_e3 : total_kernels = 237600)
  (condition_kernel_sum : total_kernels = 216 * X + 216 * (X + 100)) :
  X = 500 := by
  have condition_eq : 432 * X + 21600 = 237600 := by sorry
  have X_value : X = 216000 / 432 := by sorry
  have X_result : X = 500 := by sorry
  exact X_result

end half_ears_kernels_l309_309195


namespace common_difference_of_arithmetic_sequence_is_2_l309_309945

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d
def increasing (a : ℕ → ℝ) := ∀ n, a n < a (n + 1)
def geometric_sequence (a : ℕ → ℝ) := ∃ r, ∀ n, a (n + 1) = a n * r

-- Problem statement
theorem common_difference_of_arithmetic_sequence_is_2 
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_increasing : increasing a)
  (h_sum : a 0 + a 1 + a 2 = 12)
  (h_geom : (a 0 + d, a 1 + d, a 2 + 1) ∈ set_of (λ x, x.2 = x.1 * x.0)) :
  d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_is_2_l309_309945


namespace percentage_is_4_l309_309021

-- Define the problem conditions
def percentage_condition (p : ℝ) : Prop := p * 50 = 200

-- State the theorem with the given conditions and the correct answer
theorem percentage_is_4 (p : ℝ) (h : percentage_condition p) : p = 4 := sorry

end percentage_is_4_l309_309021


namespace prob_negative_product_of_three_elems_from_set_l309_309363

open Finset

theorem prob_negative_product_of_three_elems_from_set : 
  let s := ({-3, -1, 0, 2, 4, 5} : Finset ℤ)
  let subsets_of_three := s.powerset.filter (λ t, t.card = 3)
  let total_subsets := subsets_of_three.card
  let negative_product_subsets := subsets_of_three.filter (λ t, (t.prod id < 0)).card
  (negative_product_subsets : ℚ) / total_subsets = 3 / 10 :=
by
  sorry

end prob_negative_product_of_three_elems_from_set_l309_309363


namespace integers_square_less_than_three_times_l309_309377

theorem integers_square_less_than_three_times (x : ℤ) : x^2 < 3 * x ↔ x = 1 ∨ x = 2 :=
by
  sorry

end integers_square_less_than_three_times_l309_309377


namespace smallest_possible_n_l309_309692

theorem smallest_possible_n (n : ℕ) (x : fin n → ℝ) (hx_nonneg : ∀ i, 0 ≤ x i) (hx_sum : ∑ i, x i = 1) (hx_sq_sum : ∑ i, (x i)^2 ≤ 1 / 64) : n ≥ 64 :=
by sorry

end smallest_possible_n_l309_309692


namespace f_decreasing_interval_perimeter_range_l309_309154

noncomputable section

-- Define the given function f(x)
def f (x : ℝ) : ℝ := sin (x / 4) ^ 2 - 2 * cos (x / 4) ^ 2 + sqrt 3 * sin (x / 4) * cos (x / 4)

-- Prove that the function f(x) is decreasing on the given intervals
theorem f_decreasing_interval (k : ℤ) : 
  ∃ I : Set ℝ, I = Set.Icc (4 * k * π + 5 * π / 3) (4 * k * π + 11 * π / 3) ∧ 
  ∀ x ∈ I, ∃ y : ℝ, y = f x ∧ sorry := sorry

-- Define the conditions in triangle ABC
variables {a b c : ℝ} {A B C : ℝ}

-- Given conditions for the triangle problem
def f_B_eq_neg_half (B : ℝ) : Prop := f B = -1 / 2
def b_sq_eq_3 := b = sqrt 3

-- Prove the range of the perimeter of triangle ABC
theorem perimeter_range (B : ℝ) (b := sqrt 3) (h₁ : f_B_eq_neg_half B) (h₂ : 0 < B ∧ B < π)
  (h₃ : b_sq_eq_3) : 
  ∃ P : Set ℝ, P = Set.Ioo (2 * sqrt 3) (2 + sqrt 3] ∧ 
  ∃ a c : ℝ, P = {a + b + c | sorry} := sorry

end f_decreasing_interval_perimeter_range_l309_309154


namespace min_value_f_l309_309129

noncomputable def f (x : ℝ) : ℝ := (Real.tan x)^2 - 4 * (Real.tan x) - 12 * (Real.cot x) + 9 * (Real.cot x)^2 - 3

theorem min_value_f : ∃ x ∈ Ioo (-Real.pi / 2) 0, ∀ y ∈ Ioo (-Real.pi / 2) 0, f(x) ≤ f(y) ∧ f(x) = 3 + 8 * Real.sqrt(3) := by
  sorry

end min_value_f_l309_309129


namespace product_of_areas_square_of_volume_l309_309317

-- Declare the original dimensions and volume
variables (a b c : ℝ)
def V := a * b * c

-- Declare the areas of the new box
def area_bottom := (a + 2) * (b + 2)
def area_side := (b + 2) * (c + 2)
def area_front := (c + 2) * (a + 2)

-- Final theorem to prove
theorem product_of_areas_square_of_volume :
  (area_bottom a b) * (area_side b c) * (area_front c a) = V a b c ^ 2 :=
sorry

end product_of_areas_square_of_volume_l309_309317


namespace ratio_a_to_b_zero_l309_309489

variables (a y b : ℝ)
-- Conditions for the arithmetic sequence
axiom arithmetic_sequence : 
  (b - y) = (y - a) ∧ (3 * y - b) = (b - y)

theorem ratio_a_to_b_zero : (a / b) = 0 :=
by {
  have h1 : (b - y) = (y - a),
  have h2 : (3 * y - b) = (b - y),
  rw [h1, h2],
  sorry
}

end ratio_a_to_b_zero_l309_309489


namespace mn_parallel_pq_l309_309647

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309647


namespace tangent_circles_locus_l309_309326

noncomputable def locus_condition (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

theorem tangent_circles_locus (a b : ℝ) (r : ℝ) (h : locus_condition a b r) :
  a^2 + 7 * b^2 - 34 * a - 57 = 0 :=
sorry

end tangent_circles_locus_l309_309326


namespace gas_pressure_inversely_proportional_l309_309880

variable {T : Type} [Nonempty T]

theorem gas_pressure_inversely_proportional
  (P : T → ℝ) (V : T → ℝ)
  (h_inv : ∀ t, P t * V t = 24) -- Given that pressure * volume = k where k = 24
  (t₀ t₁ : T)
  (hV₀ : V t₀ = 3) (hP₀ : P t₀ = 8) -- Initial condition: volume = 3 liters, pressure = 8 kPa
  (hV₁ : V t₁ = 6) -- New condition: volume = 6 liters
  : P t₁ = 4 := -- We need to prove that the new pressure is 4 kPa
by 
  sorry

end gas_pressure_inversely_proportional_l309_309880


namespace environmental_select_groups_l309_309527

theorem environmental_select_groups :
  let group1 := 3,
      group2 := 3,
      group3 := 4 in
  (nat.choose 3 2) * (nat.choose 3 1) * (nat.choose 4 1) +  -- case 1
  (nat.choose 3 1) * (nat.choose 3 2) * (nat.choose 4 1) +  -- case 2
  (nat.choose 3 1) * (nat.choose 3 1) * (nat.choose 4 2) = 126 :=  -- case 3
by
  sorry

end environmental_select_groups_l309_309527


namespace degree_of_k_l309_309204

open Polynomial

theorem degree_of_k (h k : Polynomial ℝ) 
  (h_def : h = -5 * X^5 + 4 * X^3 - 2 * X^2 + C 8)
  (deg_sum : (h + k).degree = 2) : k.degree = 5 :=
sorry

end degree_of_k_l309_309204


namespace percentage_of_green_eyed_brunettes_l309_309463

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end percentage_of_green_eyed_brunettes_l309_309463


namespace find_number_l309_309342

theorem find_number (n : ℝ) :
  (n + 2 * 1.5)^5 = (1 + 3 * 1.5)^4 → n = 0.72 :=
sorry

end find_number_l309_309342


namespace sin_C_value_triangle_area_l309_309630

variables (A B C a b c : ℝ)

-- Given conditions
def triangle_conditions : Prop :=
  B = Real.pi / 3 ∧
  Real.cos A = 4 / 5 ∧
  b = Real.sqrt 3

-- Prove the value of sin(C)
theorem sin_C_value (h : triangle_conditions A B C a b c) :
  ∃ C, Real.sin C = (3 + 4 * Real.sqrt 3) / 10 :=
by sorry

-- Prove the area of the triangle
theorem triangle_area (h : triangle_conditions A B C a b c) :
  ∃ S, (1 / 2) * (6 / 5) * Real.sqrt 3 * ((3 + 4 * Real.sqrt 3) / 10) = S :=
by sorry

end sin_C_value_triangle_area_l309_309630


namespace determine_alpha_l309_309528

theorem determine_alpha (α : ℝ) (h : α ∈ Set.Ioo 0 (2 * Real.pi)) 
  (coord : (Real.sin (Real.pi / 6), Real.cos (5 * Real.pi / 6))) : 
  α = 5 * Real.pi / 3 := 
sorry

end determine_alpha_l309_309528


namespace evaluate_fraction_l309_309112

theorem evaluate_fraction (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 1) : 6 / (a + b + c) = 1 := 
by 
  rw [ha, hb, hc]
  norm_num

end evaluate_fraction_l309_309112


namespace exists_j_k_l309_309146

noncomputable def P (m : ℕ) : ℕ :=
  if m > 1 then m.factors.filter (λ p, p.prime).prod else 1

def a_seq : ℕ → ℕ
| 0     => 2
| (n+1) => a_seq n + P (a_seq n)

theorem exists_j_k : ∃ (j k : ℕ), (∏ i in (Finset.range k).map Nat.prime, i) = a_seq j :=
sorry

end exists_j_k_l309_309146


namespace goldfish_problem_solution_l309_309715

variables (initial remaining vanished heat deaths disease strays raccoons birds : ℕ) 
variables (vanished_correct : initial - remaining = vanished)
variables (heat_correct : heat = vanished * 20 / 100)
variables (disease_correct : disease = 30)
variables (birds_correct : birds = 15)
variables (total_eaten : ℕ)
variables (total_eaten_correct : total_eaten = vanished - heat - disease - birds)
variables (raccoons_correct : raccoons + 2 * raccoons + birds = total_eaten)
variables (raccoons_val : raccoons = 32)
variables (strays_val : strays = 2 * raccoons)

theorem goldfish_problem_solution :
  raccoons = 32 ∧ strays = 64 ∧ birds = 15 ∧ heat = 39 ∧ disease = 30 :=
by {
  -- condition keeping the context for automatic assumption
  rename_i vanished_correct heat_correct disease_correct birds_correct total_eaten_correct raccoons_correct raccoons_val strays_val,
  split,
  -- proof of each partial assertion
  { exact raccoons_val },
  { exact strays_val },
  { exact birds_correct.symm },
  { exact heat_correct },
  { exact disease_correct }
}

end goldfish_problem_solution_l309_309715


namespace ellipse_equation_quadrilateral_OP_A2Q_rhombus_l309_309546

-- Define the ellipse and its properties
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the properties and conditions
variables (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
variables (c : ℝ) (h4 : c = a / 2)
variables (area_max : ℝ) (h5 : area_max = sqrt 3) (triangle_area : ℝ) 
(h6 : ∃ P F1 F2 : ℝ × ℝ, triangle_area = (1/2) * a * b)
(h7 : ∃ P F1 F2 : ℝ × ℝ, triangle_area_max = sqrt 3)

-- Part (1): Prove the equation of the ellipse
theorem ellipse_equation : ellipse 0 0 2 sqrt 3 := by
  sorry

-- Part (2): Prove that OPA2Q is a rhombus under the given conditions
variables (l : ℝ → ℝ) (F2 A1 A2 : ℝ × ℝ)
variables (h8 : (F2 = (1,0)) ∧ (A1 = (-2,0)) ∧ (A2 = (2,0)))
variables (M N P Q : ℝ × ℝ)
variables (h9 : ∃ m, m ≠ 0 ∧ ∀ y, l y = m * y + 1 ∧ 
  (interactive_lemma M intersecting points) ∧
  (interactive_lemma N intersecting points) ∧ ... )
  
theorem quadrilateral_OP_A2Q_rhombus : quadrilateral_OPA2Q_is_rhombus := by
  sorry

end ellipse_equation_quadrilateral_OP_A2Q_rhombus_l309_309546


namespace hamburgers_leftover_l309_309441

-- Define the number of hamburgers made and served
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove the number of leftover hamburgers
theorem hamburgers_leftover : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_leftover_l309_309441


namespace sec_pi_over_six_l309_309908

theorem sec_pi_over_six : Real.sec (Real.pi / 6) = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end sec_pi_over_six_l309_309908


namespace find_second_number_l309_309319

variable (A B : ℕ)

def is_LCM (a b lcm : ℕ) := Nat.lcm a b = lcm
def is_HCF (a b hcf : ℕ) := Nat.gcd a b = hcf

theorem find_second_number (h_lcm : is_LCM 330 B 2310) (h_hcf : is_HCF 330 B 30) : B = 210 := by
  sorry

end find_second_number_l309_309319


namespace expected_heads_after_turning_l309_309247

open ProbabilityTheory

-- Let initial_heads be the initial number of heads which is 30.
def initial_heads : ℕ := 30

-- Let initial_tails be the initial number of tails which is 70.
def initial_tails : ℕ := 70

-- The total number of pennies.
def total_pennies : ℕ := 100

-- The number of pennies chosen at random to be turned over.
def chosen_pennies : ℕ := 40

-- The expected number of heads in the chosen 40 pennies.
def expected_heads_in_chosen : ℝ := chosen_pennies * (initial_heads / total_pennies)

-- The expected number of tails in the chosen 40 pennies.
def expected_tails_in_chosen : ℝ := chosen_pennies * (initial_tails / total_pennies)

-- The expected final number of heads.
def expected_final_heads : ℝ :=
  initial_heads - expected_heads_in_chosen + expected_tails_in_chosen

theorem expected_heads_after_turning :
  expected_final_heads = 46 := by
  sorry  -- Proof is omitted

end expected_heads_after_turning_l309_309247


namespace num_special_integers_l309_309524

theorem num_special_integers : ∃ n : ℕ, 
  (∀ m : ℕ, m ≤ 100 → (∃ k : ℕ, 10 * m = k ∧ divisors_count k = 3 * divisors_count m)) → n = 28 := 
sorry

end num_special_integers_l309_309524


namespace find_m_l309_309384

noncomputable def m_value (m : ℝ) : Prop :=
  ∃ a : ℝ, (x^2 - 20 * x + m) = (x - a)^2

theorem find_m : m_value 100 :=
begin
  sorry,
end

end find_m_l309_309384


namespace average_score_l309_309865

theorem average_score (n : ℕ) (h_n : n = 30) : 
  ∀ (p3 p2 p1 p0 : ℝ),
  p3 = 0.3 ∧ p2 = 0.4 ∧ p1 = 0.2 ∧ p0 = 0.1 →
  (3 * (p3 * n) + 2 * (p2 * n) + 1 * (p1 * n) + 0 * (p0 * n)) / n = 1.9 :=
by
  intros p3 p2 p1 p0 h,
  sorry

end average_score_l309_309865


namespace side_length_of_square_l309_309757

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l309_309757


namespace dz_dt_correct_l309_309907

noncomputable def z (t : ℝ) := Real.arcsin ((2*t)^2 + (4*t*t)^2 + t*t)
noncomputable def x (t : ℝ) := 2 * t
noncomputable def y (t : ℝ) := 4 * t * t
noncomputable def dz_dt (t : ℝ) := (2 * t * (1 + 4 * t + 32 * t * t)) / Real.sqrt(1 - ((x t)^2 + (y t)^2 + t*t)^2)

theorem dz_dt_correct (t : ℝ) : 
    has_deriv_at z (dz_dt t) t := by
  sorry

end dz_dt_correct_l309_309907


namespace maximum_product_l309_309547

open Real

theorem maximum_product (n : ℕ) (x : Fin n → ℝ) (h1 : 3 ≤ n)
    (h2 : ∑ k, x k / (1 + x k) = 1) : 
    ∏ k, x k ≤ (1 / (n - 1)) ^ n :=
sorry

end maximum_product_l309_309547


namespace product_and_sum_of_base8_digits_l309_309813

theorem product_and_sum_of_base8_digits (n : ℕ) (digits : Fin 5 → ℕ)
  (h : n = 7927)
  (h_base8 : digits 0 = 1 ∧ digits 1 = 7 ∧ digits 2 = 7 ∧ digits 3 = 5 ∧ digits 4 = 7) :
  (∏ i in Finset.range 5, digits i) = 1715 ∧ (∑ i in Finset.range 5, digits i) = 27 :=
by
  sorry

end product_and_sum_of_base8_digits_l309_309813


namespace part_a_part_b_l309_309737

-- Define the grid size
def grid_size : ℕ := 11

-- Positions on the grid
structure Position :=
(x : ℕ)
(y : ℕ)
(hx : x < grid_size)
(hy : y < grid_size)

inductive FrogColor
| Red
| Green

-- Define the possible jumps for red and green frogs
def red_jump (p : Position) : Position → Prop
| ⟨x, y, hx, hy⟩ =>
  (x = p.x + 1 ∨ x = p.x - 1 ∨ x = p.x + 2 ∨ x = p.x - 2)
  ∧ (y = p.y + 2 ∨ y = p.y - 2) ∨
  (y = p.y + 1 ∨ y = p.y - 1 ∨ y = p.y + 2 ∨ y = p.y - 2)
  ∧ (x = p.x + 2 ∨ x = p.x - 2)

def green_jump (p : Position) : Position → Prop
| ⟨x, y, hx, hy⟩ =>
  (x = p.x + 1 ∨ x = p.x - 1)
  ∧ (y = p.y + 1 ∨ y = p.y - 1)

-- Frogs can meet at a position if both can reach it in one or more jumps
def can_meet (frog1 frog2 : Position → Prop) : Position → Prop :=
  λ p, frog1 p ∧ frog2 p

-- Part (a)
theorem part_a (frogs : Fin 6 → Position) (colors : Fin 6 → FrogColor) :
  ∃ i j : Fin 6, i ≠ j ∧ ∃ p : Position, 
    (match colors i with
     | FrogColor.Red => red_jump (frogs i)
     | FrogColor.Green => green_jump (frogs i)) p ∧
    (match colors j with
     | FrogColor.Red => red_jump (frogs j)
     | FrogColor.Green => green_jump (frogs j)) p :=
  sorry

-- Part (b)
theorem part_b (p1 p2 : Position) :
  ∀ n : ℕ, 
   n ∈ {k : ℕ | ∃ p : Position, can_meet (red_jump p1) (green_jump p2) p = true} ↔
   n ≤ 4 :=
  sorry

end part_a_part_b_l309_309737


namespace train_length_l309_309447

theorem train_length
  (speed : ℝ) (time : ℝ)
  (h_speed : speed = 56.8)
  (h_time : time = 18) :
  speed * time = 1022.4 :=
by
  rw [h_speed, h_time]
  norm_num
  sorry

end train_length_l309_309447


namespace maintain_chromosome_stability_and_continuity_l309_309625

-- Condition definitions
def is_stability_and_continuity_maintained (p: Type → Prop) : Prop := 
  p = (λ x, x = "Meiosis" ∨ x = "Fertilization")

-- Proof problem
theorem maintain_chromosome_stability_and_continuity (p: Type → Prop):
  is_stability_and_continuity_maintained p :=
by
  sorry

end maintain_chromosome_stability_and_continuity_l309_309625


namespace present_age_of_son_l309_309851

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 28
def condition2 := M + 2 = 2 * (S + 2)

-- Theorem to be proven
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 26 := by
  sorry

end present_age_of_son_l309_309851


namespace ellipse_equation_l309_309566

theorem ellipse_equation (a b : ℝ) (x y : ℝ) (M : ℝ × ℝ)
  (h1 : 2 * a = 4)
  (h2 : 2 * b = 2 * a / 2)
  (h3 : M = (2, 1))
  (line_eq : ∀ k : ℝ, (y = 1 + k * (x - 2))) :
  (a = 2) ∧ (b = 1) ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (x^2 / 4 + y^2 = 1)) ∧
  (∃ k : ℝ, (k = -1/2) ∧ (∀ x y : ℝ, (y - 1 = k * (x - 2)) → (x + 2*y - 4 = 0))) :=
by
  sorry

end ellipse_equation_l309_309566


namespace locus_of_intersections_l309_309535

-- Define the given circle
def given_circle {α : Type*} [metric_space α] (c : α) (r : ℝ) := 
  { p : α | dist p c = r }

-- Define the point P inside the circle, distinct from the center
variables {α : Type*} [metric_space α] (c : α) (r : ℝ) (P : α)
  (hP : P ≠ c) (hP_inside : dist P c < r)

-- Define the pairs of circles tangent to the given circle from the inside and to each other at point P
def tangent_circles (c1 : α) (r1 : ℝ) (c2 : α) (r2 : ℝ) :=
  given_circle c1 r1 ∧ given_circle c2 r2 ∧ 
  ∀ p ∈ given_circle c r, p = P → dist p c1 = r1 → dist p c2 = r2 → 
  (dist c1 c2 = r1 + r2)

-- Define the Radical Axis
def radical_axis (c : α) (r : ℝ) (P : α) : set α := 
  { X : α | dist X c = dist X P + r }

-- Problem statement to prove
theorem locus_of_intersections 
  (c : α) (r : ℝ) (P : α) (hP : P ≠ c) (hP_inside : dist P c < r) :
  ∀ (c1 c2 : α) (r1 r2 : ℝ),
  tangent_circles c1 r1 c2 r2 →
  to_external_tangent_intersection = if dist X P = r  // Here, you may want to define what 'to_external_tangent_intersection' equals based on your geometric properties
→ to_external_tangent_intersection ∈ radical_axis c r :=
sorry


end locus_of_intersections_l309_309535


namespace parametric_to_standard_l309_309092

theorem parametric_to_standard (t : ℝ) (x y: ℝ) :
  (x = 3 + (1/2) * t) ∧ (y = (√3 / 2) * t) → y = √3 * x - 3 * √3 :=
by
  intros h
  sorry

end parametric_to_standard_l309_309092


namespace avg_width_is_3_5_l309_309894

def book_widths : List ℚ := [4, (3/4), 1.25, 3, 2, 7, 5.5]

noncomputable def average (l : List ℚ) : ℚ :=
  l.sum / l.length

theorem avg_width_is_3_5 : average book_widths = 23.5 / 7 :=
by
  sorry

end avg_width_is_3_5_l309_309894


namespace polygon_sides_eight_l309_309217

theorem polygon_sides_eight (n : ℕ) (h : 180 * (n - 2) = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l309_309217


namespace number_of_americans_l309_309875

theorem number_of_americans 
  (total_people : ℕ)
  (num_chinese : ℕ)
  (num_australians : ℕ)
  (h1 : total_people = 49)
  (h2 : num_chinese = 22)
  (h3 : num_australians = 11) :
  total_people - num_chinese - num_australians = 16 := 
by
  skip

end number_of_americans_l309_309875


namespace max_children_apples_l309_309501

theorem max_children_apples : 
    ∀ n : ℕ, (∑ i in finset.range (n + 1), i + 1 ≤ 100) → n ≤ 13 := 
sorry

end max_children_apples_l309_309501


namespace length_of_train_l309_309336

theorem length_of_train (v : ℝ) (t : ℝ) (L : ℝ) 
  (h₁ : v = 36) 
  (h₂ : t = 1) 
  (h_eq_lengths : true) -- assuming the equality of lengths tacitly without naming
  : L = 300 := 
by 
  -- proof steps would go here
  sorry

end length_of_train_l309_309336


namespace julien_swims_50_meters_per_day_l309_309246

-- Definitions based on given conditions
def distance_julien_swims_per_day : ℕ := 50
def distance_sarah_swims_per_day (J : ℕ) : ℕ := 2 * J
def distance_jamir_swims_per_day (J : ℕ) : ℕ := distance_sarah_swims_per_day J + 20
def combined_distance_per_day (J : ℕ) : ℕ := J + distance_sarah_swims_per_day J + distance_jamir_swims_per_day J
def combined_distance_per_week (J : ℕ) : ℕ := 7 * combined_distance_per_day J

-- Proof statement 
theorem julien_swims_50_meters_per_day :
  combined_distance_per_week distance_julien_swims_per_day = 1890 :=
by
  -- We are formulating the proof without solving it, to be proven formally in Lean
  sorry

end julien_swims_50_meters_per_day_l309_309246


namespace Alec_needs_5_more_votes_l309_309872

theorem Alec_needs_5_more_votes (
    total_students : ℕ,
    initial_votes : ℕ,
    thinking_votes : ℕ,
    goal_votes : ℕ,
    round1_fraction : ℕ,
    round2_fraction : ℕ
) : total_students = 100 →
    initial_votes = 50 →
    thinking_votes = 10 →
    goal_votes = 75 →
    round1_fraction = 4 →
    round2_fraction = 3 →
    ∃ (more_votes_needed : ℕ), more_votes_needed = 75 - (50 + (40 / round1_fraction) + (30 / round2_fraction)) :=
by
    sorry

end Alec_needs_5_more_votes_l309_309872


namespace island_population_percentage_l309_309461

theorem island_population_percentage :
  -- Defining conditions
  (∀ a b : ℕ, (a + b ≠ 0) → (a.toRat / (a + b).toRat = 65 / 100) →
   ∀ b c : ℕ, (b + c ≠ 0) → (b.toRat / (b + c).toRat = 70 / 100) →
   ∀ c d : ℕ, (c + d ≠ 0) → (c.toRat / (c + d).toRat = 10 / 100) →
  
  -- Correct answer based on conditions
  ∃ a b c d : ℕ, 
    let total := a + b + c + d in 
    total ≠ 0 ∧ 
    (d.toRat / total.toRat = 54 / 100)) := 
sorry

end island_population_percentage_l309_309461


namespace angle_is_120_deg_l309_309977

variables (a b : EuclideanSpace ℝ (Fin 3))

noncomputable def vector_length (v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.sqrt (v.dot v)

def angle_cos (v1 v2 : EuclideanSpace ℝ (Fin 3)) : ℝ :=
v1.dot v2 / (vector_length v1 * vector_length v2)

theorem angle_is_120_deg (ha : vector_length a = 1)
                         (hb : vector_length b = 2)
                         (hab : a.dot (a + b) = 0) :
  real.angle a b = real.angle_of_cos (-1 / 2) :=
sorry

end angle_is_120_deg_l309_309977


namespace expected_value_8_sided_die_l309_309413

-- Define the roll outcomes and their associated probabilities
def roll_outcome (n : ℕ) : ℕ := 2 * n^2

-- Define the expected value calculation
def expected_value (sides : ℕ) : ℚ := ∑ i in range (1, sides+1), (1 / sides) * roll_outcome i

-- Prove the expected value calculation for an 8-sided fair die
theorem expected_value_8_sided_die : expected_value 8 = 51 := by
  sorry

end expected_value_8_sided_die_l309_309413


namespace greatest_sum_of_squares_l309_309645

variable (a b c d : ℝ)

def conditions := (a + b = 12) ∧ (ab + c + d = 47) ∧ (ad + bc = 88) ∧ (cd = 54)

theorem greatest_sum_of_squares :
  conditions a b c d →
  a^2 + b^2 + c^2 + d^2 ≤ 254 :=
by
  sorry

end greatest_sum_of_squares_l309_309645


namespace problem_1_problem_2_l309_309579

noncomputable def isogonal_conjugate (P Q : Triangle.Point) (ABC : Triangle) : Prop :=
sorry

def on_circle (M N P Q : Triangle.Point) : Prop :=
sorry

def radial_axis (P Q : Triangle.Point) : Triangle.Point :=
sorry

def antipode (A B C : Triangle.Point) : Triangle.Point :=
sorry

variables {A B C P Q M N J I : Triangle.Point}

theorem problem_1 (ABC : Triangle) (P Q : Triangle.Point)
(h1 : isogonal_conjugate P Q ABC)
(h2 : ∃ M, line_through (A, P) ∩ circle_through B C Q = M)
(h3 : ∃ N, line_through (A, Q) ∩ circle_through B C P = N) :
on_circle M N P Q :=
sorry

theorem problem_2 (ABC : Triangle) (P Q : Triangle.Point)
(h1 : isogonal_conjugate P Q ABC)
(h2 : ∃ M, line_through (A, P) ∩ circle_through B C Q = M)
(h3 : ∃ N, line_through (A, Q) ∩ circle_through B C P = N)
(J : ∃ J, line_through (M, N) ∩ line_through (P, Q) = J):
line_through I J = fixed_line :=
sorry

end problem_1_problem_2_l309_309579


namespace concurrency_of_medial_reflections_l309_309297

theorem concurrency_of_medial_reflections
  (A B C P : Point)
  (D : Point := midpoint B C)
  (E : Point := midpoint C A)
  (F : Point := midpoint A B)
  (K : Point := reflection P D)
  (L : Point := reflection P E)
  (M : Point := reflection P F) :
  are_concurrent (line_through A K) (line_through B L) (line_through C M) :=
sorry

end concurrency_of_medial_reflections_l309_309297


namespace find_f_find_range_a_l309_309569

def f_condition (x : ℝ) : Prop :=
  f (1 + 1/x) = 1 / (x ^ 2) - 1

def g (x a : ℝ) : ℝ :=
  (a * x ^ 2 + x) / (x ^ 2 - 2 * x)

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem find_f (x : ℝ) (h : x ≠ 0) : f (1 + 1/x) = 1/(x^2) - 1 :=
by sorry

theorem find_range_a (a : ℝ) :
  (∀ x : ℝ, x > 2 → deriv (λ x, g x a) x > 0) → a < -1/2 :=
by sorry

end find_f_find_range_a_l309_309569


namespace no_factors_l309_309505

-- Define the polynomial p(x) = x^4 + 4x^2 + 8
def p (x : ℝ) : ℝ := x^4 + 4*x^2 + 8

-- Define possible factors
def A (x : ℝ) : ℝ := x^2 + 4
def B (x : ℝ) : ℝ := x + 2
def C (x : ℝ) : ℝ := x^2 - 4
def D (x : ℝ) : ℝ := x^2 - 2*x - 2

-- Prove that none of these are factors of p(x)
theorem no_factors (x : ℝ) : ¬(A(x) ∣ p x ∨ B(x) ∣ p x ∨ C(x) ∣ p x ∨ D(x) ∣ p x) :=
by
  sorry

end no_factors_l309_309505


namespace number_of_grids_l309_309488

/-
Mathematically Equivalent Proof Problem:
Given a 4x4 grid with unique numbers from 1 to 16,
each row and column are in increasing order,
1 is at position (0,0) and 16 is at (3,3),
prove the number of such grids equals 14400.
-/

noncomputable def count_4x4_grids : ℕ := 14400

theorem number_of_grids :
  ∃ count : ℕ,
  (count = count_4x4_grids) ∧ 
  (∃ grid : Array (Array ℕ),
  (grid.size = 4 ∧ 
   ∀ i < grid.size, (grid[i].size = 4) ∧
   (∀ j < grid[i].size, (1 ≤ grid[i][j] ∧ grid[i][j] ≤ 16)) ∧
   (∀ i < 4, ∀ j < 4, i > 0 → grid[i][0] > grid[i-1][0]) ∧
   (∀ i < 4, ∀ j < 4, j > 0 → grid[i][j] > grid[i][j-1]) ∧
   (grid[0][0] = 1) ∧
   (grid[3][3] = 16))) := 
begin
  use 14400, 
  split,
  {
    -- Definition of count_4x4_grids is 14400
    refl,
  },
  {
    -- Skipping detailed construction and verification of the grid
    sorry,
  }
end

end number_of_grids_l309_309488


namespace minimum_n_value_l309_309274

def sequence_property (n : ℕ) (S : finset ℕ) (a : ℕ → ℕ) : Prop :=
∀ B : finset ℕ, B ≠ ∅ → B ⊆ S → ∃ i : ℕ, B = finset.image (λ j, a (i + j)) (finset.range B.card)

theorem minimum_n_value :
  ∀ (S : finset ℕ), S = {1, 2, 3, 4} →
  ∃ n : ℕ, (∀ a : ℕ → ℕ, sequence_property n S a) ∧ n = 8 :=
by
  intros S hS
  use 8
  split
  · intros a hs hB hBs
    -- proof steps skipped
    sorry
  · trivial

end minimum_n_value_l309_309274


namespace smallest_positive_period_monotonically_increasing_intervals_max_min_value_on_interval_l309_309961

noncomputable def f (x : ℝ) : ℝ := 4 * (Math.sin x)^3 * (Math.cos x) - 2 * (Math.sin x) * (Math.cos x) - 1/2 * (Math.cos (4 * x))

theorem smallest_positive_period :
  ∃ (T > 0), T = π / 2 ∧ ∀ x : ℝ, f(x + T) = f(x) := 
sorry

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc ((k : ℝ) * π / 2 + π / 16) ((k : ℝ) * π / 2 + 5 * π / 16) →
  (f x).deriv x > 0 := 
sorry

theorem max_min_value_on_interval :
  ∀ x : ℝ, x ∈ Set.Icc 0 (π / 4) → 
  f x ≤ 1/2 ∧ f x ≥ -√2/2 := 
sorry

end smallest_positive_period_monotonically_increasing_intervals_max_min_value_on_interval_l309_309961


namespace manager_salary_is_3800_l309_309741

-- Definitions of the given conditions
constant average_salary_20_employees : ℝ := 1700
constant number_of_employees : ℤ := 20
constant salary_manager_added : ℝ := 1800

-- Sum of salaries of the 20 employees
def total_salary_20_employees : ℝ := number_of_employees * average_salary_20_employees

-- Sum of salaries of the 21 people after manager is added
def total_salary_21_people : ℝ := (number_of_employees + 1) * salary_manager_added

-- Definition of the manager's salary
def manager_salary : ℝ := total_salary_21_people - total_salary_20_employees

-- Proof statement that needs to be proven
theorem manager_salary_is_3800 : manager_salary = 3800 := by sorry

end manager_salary_is_3800_l309_309741


namespace min_f_value_l309_309126

noncomputable def f : ℝ → ℝ := λ x, (Real.tan x)^2 - 4 * Real.tan x - 12 * Real.cot x + 9 * (Real.cot x)^2 - 3

theorem min_f_value : ∃ x ∈ Ioo (-Real.pi / 2) 0, f x = 3 + 8 * Real.sqrt 3 := 
sorry

end min_f_value_l309_309126


namespace abel_arrival_earlier_l309_309871

variable (distance : ℕ) (speed_abel : ℕ) (speed_alice : ℕ) (start_delay_alice : ℕ)

theorem abel_arrival_earlier (h_dist : distance = 1000) 
                             (h_speed_abel : speed_abel = 50) 
                             (h_speed_alice : speed_alice = 40) 
                             (h_start_delay : start_delay_alice = 1) : 
                             (start_delay_alice + distance / speed_alice) * 60 - (distance / speed_abel) * 60 = 360 :=
by
  sorry

end abel_arrival_earlier_l309_309871


namespace product_of_repeating_decimal_l309_309130

theorem product_of_repeating_decimal (x : ℚ) (h : x = 1 / 3) : (x * 8) = 8 / 3 := by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l309_309130


namespace circles_tangent_tangent_product_radii_l309_309485

theorem circles_tangent_tangent_product_radii (a b c : ℕ) (m k : ℝ) (h_pos_m : m > 0) (h_pos_k : k > 0)
    (h_intersect : intersect (circle (7, 9)  r1) (circle (7, 9) r2))
    (h_prod_radii : r1 * r2 = 65)
    (h_tangent_xaxis : tangent_to_xaxis (circle C₁ r1))
    (h_tangent_line : tangent_to_line (circle C₁ r1) (line m k))
    (h_rel_prime : nat.coprime a c)
    (h_expr_m : m = a * (sqrt b) / c)
    (h_prime_b : ¬ prime_squares_div (b)) :
  a + b + c = 259 :=
by
  sorry

end circles_tangent_tangent_product_radii_l309_309485


namespace perfect_cube_factors_count_l309_309343

-- Define the given prime factorization
def prime_factorization_8820 : Prop :=
  ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧
  (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) = 8820

-- Prove the statement about positive integer factors that are perfect cubes
theorem perfect_cube_factors_count : prime_factorization_8820 → (∃ n : ℕ, n = 1) :=
by
  sorry

end perfect_cube_factors_count_l309_309343


namespace smallest_digit_to_make_divisible_by_9_l309_309140

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l309_309140


namespace tangent_parabola_line_l309_309897

theorem tangent_parabola_line (a b : ℝ) (h : a ≥ 0) :
  (∀ x : ℝ, (ax^2 + bx + 12 = 2x + 3) → ∃ x0 : ℝ, (a * x0^2 + (b - 2) * x0 + 9) = 0) ↔
  b = 2 + 6 * real.sqrt a ∨ b = 2 - 6 * real.sqrt a :=
sorry

end tangent_parabola_line_l309_309897


namespace max_sum_inverses_OA_OB_l309_309402

-- Definitions based on the conditions
def line_rectangular_coords (t a : ℝ) : ℝ × ℝ := (t * cos a, t * sin a)
def circle_rectangular_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Polar coordinate equations derived from conditions
def line_polar_eq (a : ℝ) : ℝ → ℝ → Prop := λ ρ θ, θ = a
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ^2 - (2 * cos θ + 4 * sin θ) * ρ + 1 = 0

-- Theorem to prove
theorem max_sum_inverses_OA_OB {a : ℝ} (h : 0 < a ∧ a < π / 2) :
  ∃ ρ1 ρ2 : ℝ, 
    (ρ1 + ρ2 = 2 * cos a + 4 * sin a) ∧ 
    (ρ1 * ρ2 = 1) ∧ 
    (∀ ρ1 ρ2, (ρ1 + ρ2 = 2 * cos a + 4 * sin a) → (ρ1 * ρ2 = 1) → (1 / ρ1 + 1 / ρ2 ≤ 2 * sqrt 5)) ∧
    (1 / ρ1 + 1 / ρ2 = 2 * sqrt 5) :=
begin
  sorry
end

end max_sum_inverses_OA_OB_l309_309402


namespace period_sec2_add_csc2_l309_309821

theorem period_sec2_add_csc2 (x : ℝ) : 
  (∀ x, sec x = sec (x + π)) → 
  (∀ x, csc x = csc (x + π)) → 
  (∀ x, sec x ^ 2 = sec (x + π) ^ 2) → 
  (∀ x, csc x ^ 2 = csc (x + π) ^ 2) → 
  (∀ x, (sec x ^ 2 + csc x ^ 2) = (sec (x + π) ^ 2 + csc (x + π) ^ 2)) :=
by 
  intro h1 h2 h3 h4
  sorry

end period_sec2_add_csc2_l309_309821


namespace fraction_used_first_day_l309_309050

theorem fraction_used_first_day (x : ℝ) :
  let initial_supplies := 400
  let supplies_remaining_after_first_day := initial_supplies * (1 - x)
  let supplies_remaining_after_three_days := (2/5 : ℝ) * supplies_remaining_after_first_day
  supplies_remaining_after_three_days = 96 → 
  x = (2/5 : ℝ) :=
by
  intros
  sorry

end fraction_used_first_day_l309_309050


namespace line_through_A_and_negative_reciprocal_intercepts_l309_309913

theorem line_through_A_and_negative_reciprocal_intercepts
    (a b : ℝ) (h1: A : (5, 2)) 
    (h2 : ∃ a b : ℝ, l = line_through (5, 2) ∧ 
        (a ≠ 0 ∧ b ≠ 0 ∧ a * b = -1)) :
  (2 * x - 5 * y = -8) ∨ (x - y = 3) :=
sorry

end line_through_A_and_negative_reciprocal_intercepts_l309_309913


namespace mn_parallel_pq_l309_309648

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309648


namespace sum_of_squares_CE_correct_l309_309904

-- Given conditions
def s : ℝ := Real.sqrt 200
def r : ℝ := Real.sqrt 20

-- Definitions based on the problem
-- In an equilateral triangle, the summed square of CE_k's for the given congruent triangles
noncomputable def sum_of_squares_CE (s r : ℝ) : ℝ :=
  4 * (2 * s^2 - (2 * s^2 - r^2))

-- The theorem we need to prove
theorem sum_of_squares_CE_correct (s r : ℝ) (h_s : s = Real.sqrt 200) (h_r : r = Real.sqrt 20) :
    sum_of_squares_CE s r = 1220 := by
  rw [h_s, h_r]
  sorry  -- Proof Details Here

end sum_of_squares_CE_correct_l309_309904


namespace minimum_b_value_l309_309169

theorem minimum_b_value (k : ℕ) (x y z b : ℕ) (h1 : x = 3 * k) (h2 : y = 4 * k)
  (h3 : z = 7 * k) (h4 : y = 15 * b - 5) (h5 : ∀ n : ℕ, n = 4 * k + 5 → n % 15 = 0) : 
  b = 3 :=
by
  sorry

end minimum_b_value_l309_309169


namespace area_ratio_proof_l309_309295

noncomputable theory

-- Definitions based on the problem conditions
def Triangle (V : Type) := (V × V × V)
def Pentagon (V : Type) := (V × V × V × V × V)

variables {V : Type} [EuclideanGeometry V] 
variables (A H I J K L M N : V) 

-- Conditions as per the problem
def is_equilateral_triangle (T : Triangle V) : Prop := 
  let ⟨A, H, I⟩ := T in
  dist A H = dist H I ∧ dist H I = dist I A ∧ dist I A = dist A H 

def parallel_segments (A H I J K L M N : V) : Prop :=
  Parallel (line_through AJ KJ) (line_through AH HI) ∧
  Parallel (line_through LM MN) (line_through HI HI) ∧
  Parallel (line_through MN MN) (line_through HI HI)

def equal_segments (A H J L M N : V) : Prop := 
  dist A J = dist J L ∧ dist J L = dist L M ∧ dist L M = dist M N

-- Given definitions
def pentagon_JKLMN_in_triangle_AHI := 
  ∃ (T : Triangle V), is_equilateral_triangle T ∧
    parallel_segments A H I J K L M N ∧
    equal_segments A H J L M N

-- Our proof goal
def area_ratio_pentagon_to_triangle := 
  pentagon_JKLMN_in_triangle_AHI A H I J K L M N → 
  area (pentagon J K L M N) / area (triangle A H I) = 24 / 25

-- Proof statement
theorem area_ratio_proof : area_ratio_pentagon_to_triangle := sorry

end area_ratio_proof_l309_309295


namespace urn_gold_coins_percentage_l309_309876

noncomputable def percentage_gold_coins_in_urn
  (total_objects : ℕ)
  (beads_percentage : ℝ)
  (rings_percentage : ℝ)
  (coins_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  : ℝ := 
  let gold_coins_percentage := 100 - silver_coins_percentage
  let coins_total_percentage := total_objects * coins_percentage / 100
  coins_total_percentage * gold_coins_percentage / 100

theorem urn_gold_coins_percentage 
  (total_objects : ℕ)
  (beads_percentage rings_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  (h1 : beads_percentage = 15)
  (h2 : rings_percentage = 15)
  (h3 : beads_percentage + rings_percentage = 30)
  (h4 : coins_percentage = 100 - 30)
  (h5 : silver_coins_percentage = 35)
  : percentage_gold_coins_in_urn total_objects beads_percentage rings_percentage (100 - 30) 35 = 45.5 :=
sorry

end urn_gold_coins_percentage_l309_309876


namespace probability_A_occurs_l309_309396

variables (A B : Prop) [ProbA : probability A] [ProbB : probability B]

noncomputable def P (a : Prop) [probability a] : ℝ := classical.some (ProbA a)

theorem probability_A_occurs : 
  ∀ P_A P_B : ℝ, 
    P_A > 0 ∧ 
    P_A = 2 * P_B ∧ 
    (P_A + P_B - P_A * P_B) = 12 * (P_A * P_B) ∧ 
    (independent A B) → 
    P_A = 3 / 13 :=
by
  intros P_A P_B h_pos h_eq h_lhs_rhs h_indep
  sorry

end probability_A_occurs_l309_309396


namespace volleyball_cyclic_relation_exists_l309_309017

theorem volleyball_cyclic_relation_exists :
  ∀ (team : Type) (teams : Finset team) (defeated_by : team → team → Prop),
    (teams.card = 12) →
    (∀ t, ∑ u in teams, if defeated_by t u then 1 else 0 ≠ 7) →
    ∃ (a b c : team), a ∈ teams ∧ b ∈ teams ∧ c ∈ teams ∧
                      defeated_by a b ∧ defeated_by b c ∧ defeated_by c a :=
by {
  sorry
}

end volleyball_cyclic_relation_exists_l309_309017


namespace midpoint_parallel_l309_309667

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309667


namespace discount_price_l309_309826

theorem discount_price (P : ℝ) (h : P > 0) (discount : ℝ) (h_discount : discount = 0.80) : 
  (P - P * discount) = P * 0.20 :=
by
  sorry

end discount_price_l309_309826


namespace proof_order_values_l309_309925

variable {f : ℝ → ℝ}

-- f is an even function
def even_function (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)

-- f is strictly decreasing on (0, +∞)
def strictly_decreasing_on_pos (f : ℝ → ℝ) := ∀ x, (0 < x) → f'(x) < 0

theorem proof_order_values 
  (hf_even : even_function f)
  (hf_decreasing : strictly_decreasing_on_pos f) 
  : f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end proof_order_values_l309_309925


namespace inequality_solution_l309_309345

theorem inequality_solution (x : ℝ) : (1 - 3 * (x - 1) < x) ↔ (x > 1) :=
by sorry

end inequality_solution_l309_309345


namespace polygon_sides_eight_l309_309216

theorem polygon_sides_eight (n : ℕ) (h : 180 * (n - 2) = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l309_309216


namespace sum_harmonious_numbers_le_2016_l309_309209

def is_harmonious (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2 * k + 1)^3 - (2 * k - 1)^3

theorem sum_harmonious_numbers_le_2016 : 
  (Finset.filter is_harmonious (Finset.range 2017)).sum id = 6860 :=
sorry

end sum_harmonious_numbers_le_2016_l309_309209


namespace parallel_vectors_ratio_l309_309578

variable {θ : ℝ}

def a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (1, 3)
def are_parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_ratio (h : are_parallel a b) : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 := by
  sorry

end parallel_vectors_ratio_l309_309578


namespace negation_of_universal_l309_309339

theorem negation_of_universal (h : ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) : ∃ x : ℝ, x^2 + 2 * x + 5 = 0 :=
sorry

end negation_of_universal_l309_309339


namespace smallest_digit_to_make_divisible_by_9_l309_309139

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l309_309139


namespace average_talk_minutes_heard_is_correct_l309_309863

-- Definitions for the conditions based on the given problem
def talk_duration : ℝ := 80
def audience_count : ℝ := 100
def heard_entire_percentage : ℝ := 0.25
def slept_entire_percentage : ℝ := 0.15
def half_talk_percentage : ℝ := 0.40

-- Define the number of people for each category based on the conditions
def people_heard_entire : ℝ := heard_entire_percentage * audience_count
def people_slept_entire : ℝ := slept_entire_percentage * audience_count
def remaining_audience : ℝ := audience_count - people_heard_entire - people_slept_entire
def people_heard_half : ℝ := half_talk_percentage * remaining_audience
def people_heard_three_quarters : ℝ := remaining_audience - people_heard_half

-- Define the total minutes heard by the audience
def total_minutes_heard : ℝ :=
  (people_heard_entire * talk_duration) + 
  (people_slept_entire * 0) +
  (people_heard_half * (talk_duration / 2)) +
  (people_heard_three_quarters * (talk_duration * (3/4)))

-- Calculate the average minutes heard
def average_minutes_heard : ℝ := total_minutes_heard / audience_count

-- Theorem to prove the average minutes heard is 51.2 minutes
theorem average_talk_minutes_heard_is_correct : average_minutes_heard = 51.2 := 
by 
  sorry

end average_talk_minutes_heard_is_correct_l309_309863


namespace probability_greater_than_l309_309702

/-- Let X be a normal random variable such that X ~ N(3, σ^2).
    Given P(X > m) = 0.3, prove that P(X > 6 - m) = 0.7. -/
theorem probability_greater_than (X : ℝ → ℝ) (σ m : ℝ) :
  (∀ x, X x = real.norm_dist 3 σ^2) →
  (∫ x in set.Ioi m, X x = 0.3) →
  (∫ x in set.Ioi (6 - m), X x = 0.7) :=
by
  sorry

end probability_greater_than_l309_309702


namespace angle_bisector_length_l309_309628

theorem angle_bisector_length (A B C D : Type) [Triangle A B C]
  (h1 : AB = 4) (h2 : AC = 5) (h3 : cos (angle A) = 1 / 10) :
  length (segment AD) = sorry :=
sorry

end angle_bisector_length_l309_309628


namespace sports_parade_children_l309_309519

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l309_309519


namespace system_of_equations_solution_l309_309510

theorem system_of_equations_solution (x y z : ℝ) :
  (4 * x^2 / (1 + 4 * x^2) = y ∧
   4 * y^2 / (1 + 4 * y^2) = z ∧
   4 * z^2 / (1 + 4 * z^2) = x) →
  ((x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) :=
by
  sorry

end system_of_equations_solution_l309_309510


namespace number_of_children_l309_309522

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l309_309522


namespace total_pencils_correct_l309_309483

noncomputable def Cindi_pencils : ℕ := 75

noncomputable def Marcia_pencils : ℕ := 131

noncomputable def Donna_pencils : ℕ := 589

noncomputable def Bob_pencils : ℕ := 93

noncomputable def Ellen_pencils : ℕ := 392

theorem total_pencils_correct :
  let total_pencils := Donna_pencils + Marcia_pencils + Bob_pencils + Ellen_pencils 
  in total_pencils = 1205 :=
by
  let cindi_pencils := (18.75 / 0.25).toNat
  let marcia_pencils := (1.75 * cindi_pencils).toNat
  let donna_pencils := (4.5 * marcia_pencils).toNat
  let bob_pencils := (cindi_pencils + (0.25 * cindi_pencils)).toNat
  let ellen_pencils := ((2/3:Real) * donna_pencils).toNat
  let total_pencils := donna_pencils + marcia_pencils + bob_pencils + ellen_pencils
  have h1 : cindi_pencils = 75 := by sorry
  have h2 : marcia_pencils = 131 := by sorry
  have h3 : donna_pencils = 589 := by sorry
  have h4 : bob_pencils = 93 := by sorry
  have h5 : ellen_pencils = ((2/3:Real) * donna_pencils).toNat := by sorry
  have h6 : total_pencils = 1205 := by sorry
  exact h6

end total_pencils_correct_l309_309483


namespace primes_sum_factorial_div_l309_309095

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_pairs_to_n_sum_prod (n : ℕ) : ℕ :=
  ∏ p in (Finset.filter is_prime (Finset.range n.succ)), ∏ q in (Finset.filter is_prime (Finset.Ico (p + 1) n.succ)), (p + q)

theorem primes_sum_factorial_div {n : ℕ} (h : n ≥ 3) :
  (∃ n, n = 7) ↔ n! ∣ prime_pairs_to_n_sum_prod n := 
by
  sorry

end primes_sum_factorial_div_l309_309095


namespace hcf_of_abc_l309_309599

-- Given conditions
variables (a b c : ℕ)
def lcm_abc := Nat.lcm (Nat.lcm a b) c
def product_abc := a * b * c

-- Statement to prove
theorem hcf_of_abc (H1 : lcm_abc a b c = 1200) (H2 : product_abc a b c = 108000) : 
  Nat.gcd (Nat.gcd a b) c = 90 :=
by
  sorry

end hcf_of_abc_l309_309599


namespace part_I_solution_set_part_II_range_of_a_l309_309572

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 6

-- (I) Prove the solution set of f(x) < 0 when a = 5
theorem part_I_solution_set : 
  (∀ x : ℝ, f x 5 < 0 ↔ (-3 < x ∧ x < -2)) := by
  sorry

-- (II) Prove the range of a such that f(x) > 0 for all x ∈ ℝ 
theorem part_II_range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f x a > 0) ↔ (-2*Real.sqrt 6 < a ∧ a < 2*Real.sqrt 6)) := by
  sorry

end part_I_solution_set_part_II_range_of_a_l309_309572


namespace direction_same_l309_309272

variables {V : Type*} [inner_product_space ℝ V]

-- Let a and b be non-zero vectors
variables (a b : V)
(h₁ : a ≠ 0) (h₂ : b ≠ 0)
(h₃ : ‖a + b‖ = ‖a‖ + ‖b‖)

-- Prove: The direction of a is the same as the direction of b
theorem direction_same (h₃ : ‖a + b‖ = ‖a‖ + ‖b‖) : ∃ k : ℝ, k > 0 ∧ a = k • b :=
sorry

end direction_same_l309_309272


namespace math_problem_l309_309691

noncomputable theory

variables (x y z : ℂ)

-- Conditions
def conditions : Prop :=
  (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) ∧
  (x + y + z = 30) ∧
  ((x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z)

-- Proof statement
theorem math_problem
  (h : conditions x y z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 33 :=
sorry

end math_problem_l309_309691


namespace no_increase_in_probability_l309_309037

def hunter_two_dogs_probability (p : ℝ) : ℝ :=
  let both_correct := p * p
  let one_correct := 2 * p * (1 - p) / 2
  both_correct + one_correct

theorem no_increase_in_probability (p : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ 1) :
  hunter_two_dogs_probability p = p :=
by
  sorry

end no_increase_in_probability_l309_309037


namespace sum_of_x_coordinates_of_A_l309_309809

theorem sum_of_x_coordinates_of_A :
  let B := (0, 0) in
  let C := (400, 0) in
  let D := (1000, 500) in
  let E := (1010, 515) in
  let area_ABC := 4000 in
  let area_ADE := 12000 in
  ∃ (A : ℝ × ℝ), 
    let a := A.1 in
    let b := A.2 in
    let h := 20 in -- From the area of triangle ABC
    (b = 20 ∨ b = -20) ∧
    |a - 2*b - 850| = 2400 ∧
    (a - 40 - 850 = 2400 ∨ a - 40 - 850 = -2400 ∨ a + 40 - 850 = 2400 ∨ a + 40 - 850 = -2400) ∧
    a ∈ {-1550, 3290, -1990, 3210} ∧
    ∑ x in {3290, -1550, 3210, -1990}, x = 2960 :=
sorry

end sum_of_x_coordinates_of_A_l309_309809


namespace part_one_part_two_l309_309610

-- Definition of the acute triangle and the given conditions
variables {A B C : ℝ} {a b c : ℝ}
variables (h₁ : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
          (h₂ : A + B + C = π)
          (h₃ : tan A + tan B + tan C = sqrt 3 * tan B * tan C)
          (h₄ : b * (c - b) ≤ λ * a^2)

-- Part (1): Prove that angle A = π / 3
theorem part_one : A = π / 3 := sorry

-- Part (2): Prove the range of λ
theorem part_two : λ ≥ 1 / 3 := sorry

end part_one_part_two_l309_309610


namespace tangent_circles_locus_l309_309327

noncomputable def locus_condition (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

theorem tangent_circles_locus (a b : ℝ) (r : ℝ) (h : locus_condition a b r) :
  a^2 + 7 * b^2 - 34 * a - 57 = 0 :=
sorry

end tangent_circles_locus_l309_309327


namespace sequence_last_number_is_one_l309_309361

theorem sequence_last_number_is_one :
  ∃ (a : ℕ → ℤ), (a 1 = 1) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → a (n + 1) = a n + a (n + 2)) ∧ (a 1999 = 1) := sorry

end sequence_last_number_is_one_l309_309361


namespace cos_approximation_l309_309701

noncomputable def nth_derivative_of_cos (n : ℕ) : ℝ → ℝ :=
  if n % 4 = 0 then cos
  else if n % 4 = 1 then λ x, -sin x
  else if n % 4 = 2 then λ x, -cos x
  else λ x, sin x

theorem cos_approximation : 
  let f := cos in
  ∀ (x : ℝ), 
  f 2 ≈ f 0 + (nth_derivative_of_cos 1 0 / 1!) * 2 + 
             (nth_derivative_of_cos 2 0 / 2!) * (2^2) + 
             (nth_derivative_of_cos 3 0 / 3!) * (2^3) + 
             (nth_derivative_of_cos 4 0 / 4!) * (2^4) :=
by
  sorry

end cos_approximation_l309_309701


namespace g_passes_through_fixed_point_l309_309576

theorem g_passes_through_fixed_point {b : ℝ} (hb : b > 1) (a : ℝ) 
  (h : ∀ x > 0, (a^2 - a - 1) * x^a > 0) : 
  ∃ x, x = -2 ∧ g x = 0 :=
by
  let f (x : ℝ) := (a^2 - a - 1) * x^a
  let g (x : ℝ) := b^(x + a) - 1
  have ha : a = 2 := sorry  -- justification for a = 2 based on f(x) being increasing
  have gx0 : g (-2) = 0 :=
  by
    rw [ha]
    simp [g, b, hb]
  exact ⟨-2, gx0⟩

end g_passes_through_fixed_point_l309_309576


namespace folding_hexagon_quadrilateral_folding_hexagon_pentagon_l309_309920

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem folding_hexagon_quadrilateral :
  (sum_of_interior_angles 4 = 360) :=
by
  sorry

theorem folding_hexagon_pentagon :
  (sum_of_interior_angles 5 = 540) :=
by
  sorry

end folding_hexagon_quadrilateral_folding_hexagon_pentagon_l309_309920


namespace tan_2x_abs_properties_l309_309334

open Real

theorem tan_2x_abs_properties :
  (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (-x))|) ∧ (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (x + π / 2))|) :=
by
  sorry

end tan_2x_abs_properties_l309_309334


namespace inequality_holds_for_negative_a_l309_309915

theorem inequality_holds_for_negative_a (a : ℝ) (h : a ≤ -2) :
  ∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x :=
sorry

end inequality_holds_for_negative_a_l309_309915


namespace angle_bisector_l309_309484

-- Definitions
variables {ℝ : Type*} [linear_ordered_field ℝ] [topological_space ℝ]
variables (k1 k2 : set (ℝ × ℝ))
variables (A D B C : ℝ × ℝ)

-- Conditions
-- Circle definitions, tangency, and intersection
def circle (k : set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ (P : ℝ × ℝ), P ∈ k ↔ (dist P center = radius)

-- Define internal tangency
def internally_tangent_at (k1 k2 : set (ℝ × ℝ)) (A : ℝ × ℝ) : Prop :=
  circle k1 A.radius ∧ circle k2 A.radius ∧ subset k1 k2

-- Tangent and intersection definitions
def is_tangent (k : set (ℝ × ℝ)) (D : ℝ × ℝ) : Prop :=
  ∃ (T : set (ℝ × ℝ)), is_line T ∧ ∀ (P : ℝ × ℝ), P ∈ T → tangent_point P k = D

def tangent_intersects (T : set (ℝ × ℝ)) (k : set (ℝ × ℝ)) (B C : ℝ × ℝ) : Prop :=
  ∀ (P : ℝ × ℝ), P ∈ T → P ∈ k ↔ (P = B ∨ P = C)

-- Main theorem statement
theorem angle_bisector (h1 : internally_tangent_at k1 k2 A)
  (h2 : is_tangent k2 D)
  (h3 : tangent_intersects (tangent_line k2 D) k1 B C)
  (h4 : D ≠ A) : 
  angle A D B = angle A D C :=
sorry

end angle_bisector_l309_309484


namespace possible_values_for_n_l309_309428

theorem possible_values_for_n :
  let n_values : Finset ℝ := {n | 
    let numbers := Finset.cons 12 (Finset.cons 15 (Finset.cons 18 (Finset.cons 21 ∅))),
    let sorted_numbers := (numbers ∪ {n}).erase_dup.sort (≤),
    let mean := (∑ x in numbers, x + n) / 5,
    let median := sorted_numbers.to_list.nth (2),
    mean = median 
  } in n_values.card = 3 := sorry

end possible_values_for_n_l309_309428


namespace complement_intersection_l309_309188

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2 * x > 0}

-- Define complement of A in U
def C_U_A : Set ℝ := U \ A

-- Define set B
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_intersection (x : ℝ) : x ∈ C_U_A ∩ B ↔ 1 < x ∧ x ≤ 2 :=
by
   sorry

end complement_intersection_l309_309188


namespace peculiar_looking_less_than_500_l309_309496

def is_composite (n : ℕ) : Prop :=
  1 < n ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def peculiar_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (n % 2 = 0 ∨ n % 3 = 0 ∨ n % 7 = 0 ∨ n % 11 = 0)

theorem peculiar_looking_less_than_500 :
  ∃ n, n = 33 ∧ ∀ k, k < 500 → peculiar_looking k → k = n :=
sorry

end peculiar_looking_less_than_500_l309_309496


namespace domain_range_identify_a_f_decreasing_and_nonpositive_l309_309568

-- Given function
def f (x a : ℝ) := x^2 - 2 * a * x + 5

-- Part (1)
theorem domain_range_identify_a (a : ℝ) (h1 : 1 < a)
  (h2 : ∀ x ∈ (set.Icc 1 a), f x a ∈ set.Icc 1 a) : a = 2 :=
by sorry

-- Part (2)
theorem f_decreasing_and_nonpositive (a : ℝ) (h1 : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≥ f x₂ a)
  (h2 : ∀ x ∈ (set.Icc 1 2), f x a ≤ 0) : 3 ≤ a :=
by sorry

end domain_range_identify_a_f_decreasing_and_nonpositive_l309_309568


namespace original_price_l309_309019

theorem original_price (SP : ℝ) (P_pct : ℝ) (h1 : SP = 624) (h2 : P_pct = 4) : ∃ CP : ℝ, CP = 600 :=
by
  let CP := SP / (1 + P_pct / 100)
  use CP
  have h : CP = 600 := by
    rw [h1, h2]
    norm_num
  exact h

end original_price_l309_309019


namespace seed_cost_calc_l309_309878

-- Definition of initial conditions
def cost_two_pounds : ℝ := 44.68
def pounds_needed : ℝ := 6

-- Calculation based on conditions
def cost_per_pound : ℝ := cost_two_pounds / 2
def total_cost : ℝ := pounds_needed * cost_per_pound

-- Theorem statement
theorem seed_cost_calc : total_cost = 134.04 := by
  sorry -- Proof is required here, but it is skipped as per instructions

end seed_cost_calc_l309_309878


namespace smallest_m_value_l309_309537

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := real.sin (ω * x + real.pi / 2)

theorem smallest_m_value (ω : ℝ) (hω : ω > 0) (T_dist : (2 * real.pi) = (4 * real.pi)) :
  ∃ m > 0, is_even_function (λ x, f ω (x + m)) ∧ m = real.pi / (2 * ω) :=
sorry

end smallest_m_value_l309_309537


namespace domain_transformation_l309_309954

-- Definitions of conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

def domain_g (x : ℝ) : Prop := 1 < x ∧ x ≤ 3

-- Theorem stating the proof problem
theorem domain_transformation : 
  (∀ x, domain_f x → 0 ≤ x+1 ∧ x+1 ≤ 4) →
  (∀ x, (0 ≤ x+1 ∧ x+1 ≤ 4) → (x-1 > 0) → domain_g x) :=
by
  intros h1 x hx
  sorry

end domain_transformation_l309_309954


namespace range_of_a_l309_309687

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x^2
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem range_of_a (a : ℝ) (h : ∀ s t : ℝ, (1/2 ≤ s ∧ s ≤ 2) → (1/2 ≤ t ∧ t ≤ 2) → f a s ≥ g t) : a ≥ 1 :=
sorry

end range_of_a_l309_309687


namespace length_of_second_offset_l309_309119

theorem length_of_second_offset (d₁ d₂ h₁ A : ℝ) (h_d₁ : d₁ = 30) (h_h₁ : h₁ = 9) (h_A : A = 225):
  ∃ h₂, (A = (1/2) * d₁ * h₁ + (1/2) * d₁ * h₂) → h₂ = 6 := by
  sorry

end length_of_second_offset_l309_309119


namespace ad_parallel_mp_l309_309611

open EuclideanGeometry Real

theorem ad_parallel_mp
  (ABC : Triangle)
  (hABC_acute : ABC.isAcute)
  (hAB_lt_AC : ABC.side3 < ABC.side2)
  (I : Point := ABC.incenter)
  (D : Point := ABC.incirclePointBC)
  (E : Point := lineThrough AD intersectsCircleAt ABC.circumcircle at anotherPoint)
  (M : Point := midPoint of ABC.side1)
  (N : Point := midPoint of arc(ABC.angle))
  (P : Point := lineThrough (N, E) intersectsCircleAtBIC ABC.circumcircle anotherPoint) :
  Parallel AD MP := sorry

end ad_parallel_mp_l309_309611


namespace double_rooms_booked_l309_309061

theorem double_rooms_booked (S D : ℕ) 
(rooms_booked : S + D = 260) 
(single_room_cost : 35 * S + 60 * D = 14000) : 
D = 196 := 
sorry

end double_rooms_booked_l309_309061


namespace triangle_with_ratio_is_right_l309_309795

theorem triangle_with_ratio_is_right (x : ℝ) (hx : 0 < x) : 
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  a ^ 2 + b ^ 2 = c ^ 2 :=
by
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  have ha2_plus_b2_eq_c2 : a^2 + b^2 = c^2 := by
    calc
      a^2 + b^2 = (3 * x)^2 + (4 * x)^2 : by rfl
          ... = 9 * x^2 + 16 * x^2 : by sorry -- omitted for brevity
          ... = 25 * x^2 : by sorry -- omitted for brevity
          ... = (5 * x)^2 : by rfl
          ... = c^2 : by rfl
  exact ha2_plus_b2_eq_c2

end triangle_with_ratio_is_right_l309_309795


namespace hansel_salary_l309_309982

variable (H : ℝ)

def hansel_raise : ℝ := 1.10 * H
def gretel_raise : ℝ := 1.15 * H
def salary_difference : ℝ := 1.15 * H - 1.10 * H

theorem hansel_salary : H = 30000 := 
by
  have condition : salary_difference H = 1500 :=
    sorry
  rw [salary_difference, sub_eq_add_neg, ← add_assoc, add_comm, mul_assoc] at condition
  sorry -- complete the proof here

end hansel_salary_l309_309982


namespace PF_bisects_angle_BFC_l309_309045

noncomputable def midpoint (A B : Point) : Point := sorry

noncomputable def angle_bisector_internal (A B C : Point) : Line := sorry

noncomputable def angle_bisector_external (A B C : Point) : Line := sorry

noncomputable def rectangle (A B C D : Point) : Prop := sorry

theorem PF_bisects_angle_BFC
  (A B C P D E F : Point)
  (hP : P ∈ (angle_bisector_internal A B C))
  (hD : D = midpoint B C)
  (hE : E ∈ (angle_bisector_external A B C))
  (h_intersect : P, D, E collinear)
  (h_rect : rectangle P A E F) :
  (PF bisects (angle B F C) internally or externally) :=
sorry

end PF_bisects_angle_BFC_l309_309045


namespace mn_parallel_pq_l309_309660

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309660


namespace green_eyed_brunettes_percentage_l309_309458

noncomputable def green_eyed_brunettes_proportion (a b c d : ℕ) 
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7) 
  (h3 : c / (c + d) = 0.1) : Prop :=
  d / (a + b + c + d) = 0.54

-- The main theorem to be proved
theorem green_eyed_brunettes_percentage (a b c d : ℕ)
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7)
  (h3 : c / (c + d) = 0.1) : 
  green_eyed_brunettes_proportion a b c d h1 h2 h3 := 
sorry

end green_eyed_brunettes_percentage_l309_309458


namespace illegal_simplification_works_for_specific_values_l309_309864

-- Definitions for the variables
def a : ℕ := 43
def b : ℕ := 17
def c : ℕ := 26

-- Define the sum of cubes
def sum_of_cubes (x y : ℕ) : ℕ := x ^ 3 + y ^ 3

-- Define the illegal simplification fraction
def illegal_simplification_fraction_correct (a b c : ℕ) : Prop :=
  (a^3 + b^3) / (a^3 + c^3) = (a + b) / (a + c)

-- The theorem to prove
theorem illegal_simplification_works_for_specific_values :
  illegal_simplification_fraction_correct a b c :=
by
  -- Proof will reside here
  sorry

end illegal_simplification_works_for_specific_values_l309_309864


namespace rank_values_l309_309264

noncomputable def a : ℝ := Real.log 7 / Real.log 3
noncomputable def b : ℝ := 2^1.1
noncomputable def c : ℝ := 0.8^3.1

theorem rank_values : c < a ∧ a < b := 
by
  have ha: 1 < a := sorry
  have hb: a < 2 := sorry
  have hc: b > 2 := sorry
  have hd: c < 1 := sorry
  exact ⟨sorry, sorry⟩

end rank_values_l309_309264


namespace triangle_inequality_tangents_l309_309298

theorem triangle_inequality_tangents (α β γ R A : ℝ) 
  (hαβγ: α + β + γ = π) 
  (hA: A = (R * R * sin(α) * sin(β) * sin(γ)) / (a * b * c)) :
  tan (α / 2) + tan (β / 2) + tan (γ / 2) ≤ (9 * R ^ 2) / (4 * A) := 
sorry

end triangle_inequality_tangents_l309_309298


namespace find_standard_equation_of_ellipse_find_maximum_area_of_triangle_PMN_l309_309944

-- Conditions
variable (a b c L : ℝ) (e : ℝ) -- L represents 4√3
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity : Prop := c / a = sqrt 6 / 3 
def perimeter_triangle : Prop := 2 * a * 2 = L -- 4√3

-- Problem 1: Find the standard equation of ellipse C
theorem find_standard_equation_of_ellipse :
  (exists a b : ℝ, a > 0 ∧ b > 0 ∧ b < a ∧ a = sqrt 3 ∧ b = 1) →
  (exists x y : ℝ, ellipse x y ↔ (x^2 / 3 + y^2 = 1)) := by
  sorry

-- Additional condition for Problem 2
variable (O_radius : ℝ := 2)
def circle (x y : ℝ) : Prop := x^2 + y^2 = O_radius^2

-- Problem 2: Find the maximum area of △PMN
theorem find_maximum_area_of_triangle_PMN :
  (forall P : ℝ × ℝ, circle P.1 P.2) →
  (∃ M N : ℝ × ℝ, M ≠ P ∧ N ≠ P ∧ 
    (* M and N are points of tangency with ellipse C, and form triangle PMN *))
  (∃ max_area : ℝ, max_area = 4) := by
  sorry

end find_standard_equation_of_ellipse_find_maximum_area_of_triangle_PMN_l309_309944


namespace no_prime_divides_all_euclid_l309_309624

noncomputable def e : ℕ → ℕ
| 0     := 2
| 1     := 3
| (n+2) := e (n+1) * e n + 1

-- Assume we have a property P that states whether a number is prime.
def P (p : ℕ) : Prop := Nat.Prime p

theorem no_prime_divides_all_euclid : ¬ ∀ p, P p → ∃ n, p ∣ e n := by
  sorry

end no_prime_divides_all_euclid_l309_309624


namespace expected_value_of_win_l309_309416

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l309_309416


namespace expected_value_of_win_is_51_l309_309422

noncomputable def expected_value_of_win : ℝ :=
  (∑ n in (finset.range 8).map (λ x, x + 1), (1/8) * 2 * (n : ℝ)^2)

theorem expected_value_of_win_is_51 : expected_value_of_win = 51 := 
by 
  sorry

end expected_value_of_win_is_51_l309_309422


namespace sum_of_gcd_values_l309_309470

open Nat

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) :
  (Finset.sum (Finset.map ⟨λ n, gcd (5 * n + 6) (2 * n + 3), λ _ _, Finset.mem_univ _⟩ (Finset.range (n + 1))) = 4) :=
by
  sorry

end sum_of_gcd_values_l309_309470


namespace conjecture_a_n_l309_309283

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

noncomputable def S_n (n : ℕ) : ℚ := 2 * n - a_n n

theorem conjecture_a_n (n : ℕ) (h : n > 0) : a_n n = (2^n - 1) / 2^(n-1) :=
by 
  sorry

end conjecture_a_n_l309_309283


namespace sum_of_fourth_powers_l309_309604

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 :=
by sorry

end sum_of_fourth_powers_l309_309604


namespace root_relationship_l309_309177

theorem root_relationship (x1 x2 : ℝ) :
  (log 4 x1 - (1 / 4)^x1 = 0) →
  (log (1 / 4) x2 - (1 / 4)^x2 = 0) →
  (0 < x1 * x2 ∧ x1 * x2 < 1) :=
sorry

end root_relationship_l309_309177


namespace bottles_left_l309_309358

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l309_309358


namespace circle_symmetric_to_line_l309_309959

theorem circle_symmetric_to_line (m : ℝ) :
  (∃ (x y : ℝ), (x^2 + y^2 - m * x + 3 * y + 3 = 0) ∧ (m * x + y - m = 0))
  → m = 3 :=
by
  sorry

end circle_symmetric_to_line_l309_309959


namespace insufficient_data_l309_309052

variable (M P O : ℝ)

theorem insufficient_data
  (h1 : M < P)
  (h2 : O > M) :
  ¬(P < O) ∧ ¬(O < P) ∧ ¬(P = O) := 
sorry

end insufficient_data_l309_309052


namespace abs_five_minus_e_l309_309905

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_e : |5 - e| = 2.282 := 
by 
    -- Proof is omitted 
    sorry

end abs_five_minus_e_l309_309905


namespace fixed_point_F_l309_309252

/--
L is a line that does not intersect a circle with center O.
E is the foot of the perpendicular from O to L.
M is a variable point on L (not E).
The tangents from O to the circle meet it at points A and B.
The feet of the perpendiculars from E to MA and MB are C and D respectively.
The lines CD and OE intersect at F.
Show that F is fixed.
-/
theorem fixed_point_F (L : Line) (circle : Circle) (O : Point) (E : Point) (M : Point)
  (hL : ¬ (intersect L circle))
  (hE : foot_perpendicular O E L)
  (hM : M ∈ L ∧ M ≠ E)
  (A B : Point) (hA : tangent_from O A circle) (hB : tangent_from O B circle)
  (C D : Point) (hC : foot_perpendicular E C (line_through M A)) (hD : foot_perpendicular E D (line_through M B))
  (F : Point) (hF : intersect (line_through C D) (line_through O E) = F)
  : fixed F :=
sorry

end fixed_point_F_l309_309252


namespace smallest_rel_prime_l309_309822

theorem smallest_rel_prime (n : ℕ) (h : n > 1) (rel_prime : ∀ p ∈ [2, 3, 5, 7], ¬ p ∣ n) : n = 11 :=
by sorry

end smallest_rel_prime_l309_309822


namespace find_m_l309_309385

noncomputable def m_value (m : ℝ) : Prop :=
  ∃ a : ℝ, (x^2 - 20 * x + m) = (x - a)^2

theorem find_m : m_value 100 :=
begin
  sorry,
end

end find_m_l309_309385


namespace mutually_exclusive_not_complementary_l309_309109

-- Definitions based on the conditions
def Card := {hearts, spades, diamonds, clubs}
def Person := {A, B, C, D}

variable (receives_card : Person → Card)

-- Definitions for the events
def A_receives_club := receives_card A = 'clubs
def B_receives_club := receives_card B = 'clubs

-- Theorem statement
theorem mutually_exclusive_not_complementary :
  (A_receives_club ∧ B_receives_club) = false ∧ 
  ∃ C : Card, receives_card B = C ∧ C ≠ 'clubs ∧ ∃ D : Card, receives_card A = D ∧ D ≠ 'clubs :=
sorry

end mutually_exclusive_not_complementary_l309_309109


namespace sufficient_condition_not_necessary_condition_l309_309689

variables (α β : Type) [plane α] [plane β]

variables (m n : line α) (l1 l2 : line β)

-- Conditions
axiom ax1 : m ≠ n -- m and n are different lines in plane α
axiom ax2 : ∃ P, incident l1 P ∧ incident l2 P -- l1 and l2 are intersecting lines in plane β

-- Sufficient but not necessary condition
theorem sufficient_condition (h1 : perpendicular m l1) (h2 : perpendicular m l2) : 
  perpendicular α β :=
  sorry

-- Define what it means for α to be perpendicular to β
axiom perpendicular_planes : α ⊥ β

-- Direction that it's not necessary condition
theorem not_necessary_condition (h : perpendicular α β) : 
  ¬ (perpendicular m l1 ∧ perpendicular m l2) :=
  sorry

end sufficient_condition_not_necessary_condition_l309_309689


namespace larger_number_is_299_l309_309832

theorem larger_number_is_299 {a b : ℕ} (hcf : Nat.gcd a b = 23) (lcm_factors : ∃ k1 k2 : ℕ, Nat.lcm a b = 23 * k1 * k2 ∧ k1 = 12 ∧ k2 = 13) :
  max a b = 299 :=
by
  sorry

end larger_number_is_299_l309_309832


namespace solve_p_q_sum_l309_309278

noncomputable def p_q_sum : ℕ :=
  let c_interval := set.Icc (-20 : ℝ) 20
  let discriminant (c : ℝ) : ℝ := (5*c^2 - 15*c)^2 - 4*36*c^2
  let valid_c (c : ℝ) : Prop := c ∈ c_interval ∧ discriminant c > 0
  let valid_interval := {c : ℝ | valid_c c}
  let interval_length (a b : ℝ) : ℝ := b - a
  let effective_length := interval_length 0 (54 / 25) + interval_length (-20) (-81 / 25)
  let total_interval_length := interval_length (-20) 20
  let p_q_fraction := effective_length / total_interval_length
  let (p, q) := (58, 125)  -- As relative prime integers resulting from the fraction simplification
  p + q

theorem solve_p_q_sum : p_q_sum = 183 := 
  by 
  sorry

end solve_p_q_sum_l309_309278


namespace ann_age_is_26_l309_309060

theorem ann_age_is_26
  (a b : ℕ)
  (h1 : a + b = 50)
  (h2 : b = 2 * a / 3 + 2 * (a - b)) :
  a = 26 :=
by
  sorry

end ann_age_is_26_l309_309060


namespace calculate_expression_l309_309888

theorem calculate_expression : 
  - 2 ^ 2 + 2 * (Real.sin (Real.pi / 3)) + (sqrt 3 - Real.pi) ^ 0 - abs (1 - sqrt 3) = -2 :=
by
  have h1 : Real.sin (Real.pi / 3) = sqrt 3 / 2 := Real.sin_pi_div_three
  rw [h1]
  have h2 : (sqrt 3 - Real.pi) ^ 0 = 1 := pow_zero _
  simp only [h2, abs_sub_comm, abs_of_neg, pow_zero, abs_sub_lt_iff]
  linarith

end calculate_expression_l309_309888


namespace general_eq_line_BC_std_eq_circumscribed_circle_ABC_l309_309235

-- Define the points A, B, and C
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-4, 1)

-- Prove the general equation of line BC is x + 1 = 0
theorem general_eq_line_BC : ∀ x y : ℝ, (x = -1) → y = 2 ∧ (x = -4) → y = 1 → x + 1 = 0 :=
by
  sorry

-- Prove the standard equation of the circumscribed circle of triangle ABC is (x + 5/2)^2 + (y - 3/2)^2 = 5/2
theorem std_eq_circumscribed_circle_ABC :
  ∀ x y : ℝ,
  (x, y) = (A : ℝ × ℝ) ∨ (x, y) = (B : ℝ × ℝ) ∨ (x, y) = (C : ℝ × ℝ) →
  (x + 5/2)^2 + (y - 3/2)^2 = 5/2 :=
by
  sorry

end general_eq_line_BC_std_eq_circumscribed_circle_ABC_l309_309235


namespace four_digit_number_sum_of_digits_2023_l309_309594

theorem four_digit_number_sum_of_digits_2023 (a b c d : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  1000 * a + 100 * b + 10 * c + d = a + b + c + d + 2023 → 
  (1000 * a + 100 * b + 10 * c + d = 1997 ∨ 1000 * a + 100 * b + 10 * c + d = 2015) :=
by
  sorry

end four_digit_number_sum_of_digits_2023_l309_309594


namespace super_ball_total_distance_l309_309051

theorem super_ball_total_distance :
  let initial_height : ℝ := 150
  let rebound_factor : ℝ := 1 / 3
  let height_after_bounce (n : ℕ) := initial_height * rebound_factor ^ n
  let descent_distance (n : ℕ) := height_after_bounce n
  let ascent_distance (n : ℕ) := height_after_bounce (n - 1)
  let total_distance_after_n_bounces := sum (range 5) (λ n, (descent_distance n) + (ascent_distance n))
  total_distance_after_n_bounces 5 ≈ 251.85 :=
  by
  let initial_height : ℝ := 150
  let rebound_factor : ℝ := 1 / 3
  let height_after_bounce (n : ℕ) := initial_height * rebound_factor ^ n
  let descent_distance (n : ℕ) := height_after_bounce n
  let ascent_distance (n : ℕ) := height_after_bounce (n - 1)
  let total_distance :
    ℝ := initial_height + 
          ascent_distance 1 + descent_distance 1 +
          ascent_distance 2 + descent_distance 2 +
          ascent_distance 3 + descent_distance 3 +
          ascent_distance 4 + descent_distance 4 +
          ascent_distance 5
  show total_distance ≈ 251.85,
  sorry

end super_ball_total_distance_l309_309051


namespace original_expenditure_l309_309804

theorem original_expenditure 
(average_expenditure_per_head : ℕ → ℝ)
(original_students new_students : ℕ)
(expense_increase : ℝ)
(average_expenditure_diminished : ℝ)
(h1 : original_students = 35)
(h2 : new_students = 7)
(h3 : expense_increase = 42)
(h4 : average_expenditure_diminished = 1)
(h5 : ∀ x, average_expenditure_per_head (original_students + new_students) = average_expenditure_per_head original_students - average_expenditure_diminished ∧ 
    new_students * (average_expenditure_per_head original_students - average_expenditure_diminished) - original_students * average_expenditure_per_head original_students = expense_increase) :
original_students * average_expenditure_per_head original_students = 420 :=
by
  sorry

end original_expenditure_l309_309804


namespace most_accurate_approximation_l309_309618

variable (V : ℝ)

def d_true (V : ℝ) : ℝ :=
  (6 * V / Real.pi)^(1 / 3)

def d_A (V : ℝ) : ℝ :=
  (16 * V / 9)^(1 / 3)

def d_B (V : ℝ) : ℝ :=
  (2 * V)^(1 / 3)

def d_C (V : ℝ) : ℝ :=
  (300 * V / 157)^(1 / 3)

def d_D (V : ℝ) : ℝ :=
  (21 * V / 11)^(1 / 3)

theorem most_accurate_approximation :
  abs (d_D V - d_true V) < abs (d_A V - d_true V)
  ∧ abs (d_D V - d_true V) < abs (d_B V - d_true V)
  ∧ abs (d_D V - d_true V) < abs (d_C V - d_true V) :=
sorry

end most_accurate_approximation_l309_309618


namespace gcd_of_2535_5929_11629_l309_309513

theorem gcd_of_2535_5929_11629 : Nat.gcd (Nat.gcd 2535 5929) 11629 = 1 := by
  sorry

end gcd_of_2535_5929_11629_l309_309513


namespace rotation_of_isosceles_triangle_is_cone_l309_309846

-- Definitions of conditions
def isosceles_triangle := Type -- with appropriate properties
def altitude (T : isosceles_triangle) : Type := sorry -- definition of altitude in context of triangle
def rotate_180 (shape : Type) (line : Type) : Type := sorry -- rotation by 180°

-- Definition of cone based on geometric properties
def cone := Type -- predefined or detailed definition of a cone

-- The theorem we need to prove
theorem rotation_of_isosceles_triangle_is_cone
  (T : isosceles_triangle)
  (L : altitude T)
  (S : Type := rotate_180 T L) :
  S = cone := sorry

end rotation_of_isosceles_triangle_is_cone_l309_309846


namespace complex_conjugate_l309_309175

noncomputable def conjugate_of_z (z : ℂ) : ℂ := conj z

theorem complex_conjugate :
  (∃ z : ℂ, (1 + complex.i) * z = (1 - complex.i) ^ 2) →
  ∃ z : ℂ, conjugate_of_z z = -1 + complex.i :=
by {
  sorry
}

end complex_conjugate_l309_309175


namespace book_arrangement_count_l309_309228

-- Definitions based on conditions
variable (M : Finset ℕ) (H : Finset ℕ)
variable (hc1 : M.card = 4) (hc2 : H.card = 6)

-- The theorem to be proven
theorem book_arrangement_count (M H : Finset ℕ) (hc1 : M.card = 4) (hc2 : H.card = 6) :
  (4 * 3 * Nat.factorial 8) = 145_152 :=
  by {
    -- Proof would go here
    sorry
  }

end book_arrangement_count_l309_309228


namespace real_part_of_z_is_negative_three_l309_309328

def imaginary_unit : ℂ := complex.i
def z : ℂ := (1 + 2 * imaginary_unit)^2

theorem real_part_of_z_is_negative_three : complex.re z = -3 :=
by
  sorry

end real_part_of_z_is_negative_three_l309_309328


namespace triangle_condition_l309_309277

theorem triangle_condition (A B C T M : Point) (angleA angleATB angleBTC angleCTA : ℝ)
  (h₀ : angleA = 60)
  (h₁ : angleATB = 120)
  (h₂ : angleBTC = 120)
  (h₃ : angleCTA = 120)
  (h₄ : is_midpoint M B C) :
  dist A T + dist B T + dist C T = 2 * dist A M :=
sorry

end triangle_condition_l309_309277


namespace pat_interest_rate_l309_309211

noncomputable def interest_rate (t : ℝ) : ℝ := 70 / t

theorem pat_interest_rate (r : ℝ) (t : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (years : ℝ) : 
  initial_amount * 2^((years / t)) = final_amount ∧ 
  years = 18 ∧ 
  final_amount = 28000 ∧ 
  initial_amount = 7000 →    
  r = interest_rate 9 := 
by
  sorry

end pat_interest_rate_l309_309211


namespace math_problem_proof_l309_309530

noncomputable def a := 2 ^ (1 / 3 : ℝ)
noncomputable def b := Real.log 3 (2 / 3)
noncomputable def c := Real.log (1 / 2 : ℝ) (1 / 3)

theorem math_problem_proof : c > a ∧ a > b :=
by
  sorry

end math_problem_proof_l309_309530


namespace side_length_of_square_l309_309764

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l309_309764


namespace mopping_time_is_30_l309_309981

def vacuuming_time := 45
def dusting_time := 60
def brushing_time_per_cat := 5
def number_of_cats := 3
def total_free_time := 180
def free_time_left := 30

def total_cleaning_time := total_free_time - free_time_left
def brushing_time := brushing_time_per_cat * number_of_cats
def time_other_tasks := vacuuming_time + dusting_time + brushing_time

theorem mopping_time_is_30 : total_cleaning_time - time_other_tasks = 30 := by
  -- Calculation proof would go here
  sorry

end mopping_time_is_30_l309_309981


namespace impossible_arrangement_l309_309011

theorem impossible_arrangement : 
  ∀ (a : Fin 111 → ℕ), (∀ i, a i ≤ 500) → (∀ i j, i ≠ j → a i ≠ a j) → 
  ¬ ∀ i : Fin 111, (a i % 10 = ((Finset.univ.sum (λ j => if j = i then 0 else a j)) % 10)) :=
by 
  sorry

end impossible_arrangement_l309_309011


namespace midpoint_parallel_l309_309669

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309669


namespace max_sum_b_n_l309_309940

noncomputable def a_n (n : ℕ) : ℝ := 11 - 2 * n
noncomputable def S_n (n : ℕ) : ℝ := n * (20 - 2 * n) / 2 -- Sum of the first n terms of a_n

theorem max_sum_b_n : 
  (∀ n, S_n n ≤ S_n 5) → 
  (∀ n, 0 < a_n n) → 
  (∀ n, a_2 ∈ ℤ) → 
  (∃ m, m > 0 ∧ ∀ k, k ≤ m → b_sum := 
    ∑ i in range k, (1 / ((11 - 2 * i) * (9 - 2 * i))) ≤ 4 / 9) :=
begin
  sorry
end

end max_sum_b_n_l309_309940


namespace midpoint_parallel_l309_309678

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309678


namespace choosing_three_different_positions_l309_309613

theorem choosing_three_different_positions (n : ℕ) (h : n = 6) :
  let num_ways := n * (n - 1) * (n - 2)
  in num_ways = 120 := by
sorry

end choosing_three_different_positions_l309_309613


namespace parabola_hyperbola_vertex_focus_parabola_chord_midpoint_l309_309174

theorem parabola_hyperbola_vertex_focus :
  (∃ (a b : ℝ), x^2 - y^2 = 1 ∧ a = 1 ∧ b = 1 ∧
   (∀ p, focus_of_parabola p = (1, 0))) → 
   (parabola_eq : ∀ y x : ℝ, y^2 = 4 * x) :=
sorry

theorem parabola_chord_midpoint (l :ℝ → Prop) :
  (∃ (C M N : ℝ × ℝ), 
    C = (2, 1) ∧
    l = λ C : ℝ × ℝ, 2 * C.fst - C.snd - 3 = 0 ∧
    (∃ (M N : ℝ × ℝ), (C.fst = (M.fst + N.fst) / 2) ∧ 
                       (C.snd = (M.snd + N.snd) / 2) ∧
                       (y_coordinates_on_parabola M N))) →
  (∃ l, l = 2 * x - y - 3 = 0) :=
sorry

end parabola_hyperbola_vertex_focus_parabola_chord_midpoint_l309_309174


namespace people_lineup_l309_309229

/-- Theorem: Given five people, where the youngest person cannot be on the first or last position, 
we want to prove that there are exactly 72 ways to arrange them in a straight line. -/
theorem people_lineup (p : Fin 5 → ℕ) 
  (hy : ∃ i : Fin 5, ∀ j : Fin 5, i ≠ j → p i < p j) 
  (h_pos : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4)
  : (∑ x in ({1, 2, 3, 4} : Finset (Fin 5)), 4 * 3 * 2 * 1) = 72 := by
  -- The proof is omitted.
  sorry

end people_lineup_l309_309229


namespace compute_inverse_l309_309534

-- Define the conditions
def x : ℂ := (1 + complex.I * real.sqrt 3) / 2

-- Define the goal to prove
theorem compute_inverse (h : x = (1 + complex.I * real.sqrt 3) / 2) : (1 / (x^2 - x)) = -1 :=
sorry

end compute_inverse_l309_309534


namespace expected_value_equals_51_l309_309425

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l309_309425


namespace star_evaluation_l309_309145

-- Define the operation
def star (a b : ℝ) : ℝ := (a + b) / (a - b)

-- State the theorem
theorem star_evaluation : (star (star 2 3) 5) = 0 :=
by
  sorry

end star_evaluation_l309_309145


namespace quadratic_real_roots_condition_l309_309539

theorem quadratic_real_roots_condition (a b c : ℝ) (q : b^2 - 4 * a * c ≥ 0) (h : a ≠ 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ a ≠ 0) ↔ ((∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) ∨ (∃ x : ℝ, a * x ^ 2 + b * x + c = 0)) :=
by
  sorry

end quadratic_real_roots_condition_l309_309539


namespace solve_equation_l309_309736

theorem solve_equation : ∃ x : ℝ, 2021 * x = 2022 * (x ^ (2021 / 2022)) - 1 ∧ x ≥ 0 :=
by {
  use 1,
  split,
  simp,
  ring,
  linarith,
  sorry
}

end solve_equation_l309_309736


namespace bottles_left_on_shelf_l309_309352

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l309_309352


namespace shortest_chord_intercept_l309_309992

theorem shortest_chord_intercept (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 3 → x + m * y - m - 1 = 0 → m = 1) :=
sorry

end shortest_chord_intercept_l309_309992


namespace solve_phi_eq_l309_309312

noncomputable def φ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat := (1 - Real.sqrt 5) / 2
noncomputable def F : ℕ → ℤ
| n =>
  if n = 0 then 0
  else if n = 1 then 1
  else F (n - 1) + F (n - 2)

theorem solve_phi_eq (n : ℕ) :
  ∃ x y : ℤ, x * φ ^ (n + 1) + y * φ^n = 1 ∧ 
    x = (-1 : ℤ)^(n+1) * F n ∧ y = (-1 : ℤ)^n * F (n + 1) := by
  sorry

end solve_phi_eq_l309_309312


namespace number_of_boxes_l309_309840

-- Definitions based on conditions
def pieces_per_box := 500
def total_pieces := 3000

-- Theorem statement, we need to prove that the number of boxes is 6
theorem number_of_boxes : total_pieces / pieces_per_box = 6 :=
by {
  sorry
}

end number_of_boxes_l309_309840


namespace projection_of_b_onto_a_l309_309556

open Real

noncomputable def e1 : ℝ × ℝ := (1, 0)
noncomputable def e2 : ℝ × ℝ := (0, 1)

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude u)

theorem projection_of_b_onto_a : projection b a = 2 * sqrt 5 / 5 := by
  sorry

end projection_of_b_onto_a_l309_309556


namespace commission_rate_l309_309884

theorem commission_rate (old_salary new_base_salary sale_amount : ℝ) (required_sales : ℕ) (condition: (old_salary = 75000) ∧ (new_base_salary = 45000) ∧ (sale_amount = 750) ∧ (required_sales = 267)) :
  ∃ commission_rate : ℝ, abs (commission_rate - 0.14981) < 0.0001 :=
by
  sorry

end commission_rate_l309_309884


namespace meaningful_fraction_l309_309212

theorem meaningful_fraction (x : ℝ) : (x ≠ -2) ↔ (∃ y : ℝ, y = 1 / (x + 2)) :=
by sorry

end meaningful_fraction_l309_309212


namespace cube_root_l309_309157

theorem cube_root (x : ℝ) (h : x^3 = 4) : x = real.cbrt 4 :=
by sorry

end cube_root_l309_309157


namespace counterexamples_count_l309_309099

def sum_of_digits (n : Nat) : Nat :=
  -- Function to calculate the sum of digits of n
  sorry

def no_zeros (n : Nat) : Prop :=
  -- Function to check that there are no zeros in the digits of n
  sorry

def is_prime (n : Nat) : Prop :=
  -- Function to check if a number is prime
  sorry

theorem counterexamples_count : 
  ∃ (M : List Nat), 
  (∀ m ∈ M, sum_of_digits m = 5 ∧ no_zeros m) ∧ 
  (∀ m ∈ M, ¬ is_prime m) ∧
  M.length = 9 := 
sorry

end counterexamples_count_l309_309099


namespace find_expression_value_l309_309144

theorem find_expression_value : 
  let x := 2;
  let y := 3 in
  x^3 + y^2 * (x^2 * y) = 116 :=
by
  sorry

end find_expression_value_l309_309144


namespace unique_intersection_point_of_line_and_parabola_l309_309779

theorem unique_intersection_point_of_line_and_parabola :
  ∃ k : ℝ, (∀ y : ℝ, -3*y^2 + 2*y + 7 = k ↔ y = (1 + real.sqrt 1 - 4*(-3)*(k-7))/(2*(-3)) ∨ y = (1 - real.sqrt 1 - 4*(-3)*(k-7))/(2*(-3))) ∧ k = 22/3 :=
sorry

end unique_intersection_point_of_line_and_parabola_l309_309779


namespace minimum_monkeys_required_l309_309056

theorem minimum_monkeys_required (total_weight : ℕ) (weapon_max_weight : ℕ) (monkey_max_capacity : ℕ) 
  (num_monkeys : ℕ) (total_weapons : ℕ) 
  (H1 : total_weight = 600) 
  (H2 : weapon_max_weight = 30) 
  (H3 : monkey_max_capacity = 50) 
  (H4 : total_weapons = 600 / 30) 
  (H5 : num_monkeys = 23) : 
  num_monkeys ≤ (total_weapons * weapon_max_weight) / monkey_max_capacity :=
sorry

end minimum_monkeys_required_l309_309056


namespace expression_meaningful_l309_309790

theorem expression_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by
  sorry

end expression_meaningful_l309_309790


namespace prime_transformation_l309_309810

theorem prime_transformation (p : ℕ) (prime_p : Nat.Prime p) (h : p = 3) : ∃ q : ℕ, q = 13 * p + 2 ∧ Nat.Prime q :=
by
  use 41
  sorry

end prime_transformation_l309_309810


namespace minimum_balls_ensure_20_single_color_l309_309020

def num_balls_to_guarantee_color (r g y b w k : ℕ) : ℕ :=
  let max_without_20 := 19 + 19 + 19 + 18 + 15 + 12
  max_without_20 + 1

theorem minimum_balls_ensure_20_single_color :
  num_balls_to_guarantee_color 30 25 25 18 15 12 = 103 := by
  sorry

end minimum_balls_ensure_20_single_color_l309_309020


namespace probability_blue_prime_and_yellow_divisible_by_3_l309_309368

-- Define the set of outcomes for each die roll
def outcomes := Finset.range 8

-- Define the set of prime numbers ≤ 8 for the blue die
def blue_prime_outcomes := {2, 3, 5, 7}

-- Define the set of numbers divisible by 3 ≤ 8 for the yellow die
def yellow_divisible_by_3_outcomes := {3, 6}

-- Define the total number of outcomes when two 8-sided dice are rolled
def total_outcomes := (outcomes.card) * (outcomes.card)

-- Define the number of successful outcomes for our condition
def successful_outcomes := (blue_prime_outcomes.card) * (yellow_divisible_by_3_outcomes.card)

-- Define the probability calculation
def probability := (successful_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem to be proved
theorem probability_blue_prime_and_yellow_divisible_by_3 :
  probability = 1 / 8 :=
by
  -- Proof is not required, so we use sorry
  sorry

end probability_blue_prime_and_yellow_divisible_by_3_l309_309368


namespace unordered_samples_ordered_sequences_place_balls_non_decreasing_paths_subsets_l309_309003

-- Define the binomial coefficient
def binomial_coefficient (N n : ℕ) : ℕ :=
  Nat.factorial N / (Nat.factorial n * Nat.factorial (N - n))

-- Conditions
variables (N n : ℕ) (A : set ℕ)
variable h : N > 0
variable h1 : n ≥ 0
variable h2 : n ≤ N

-- Proof that the number of unordered samples of size n from a set A with |A| = N is C_N^n
theorem unordered_samples (hA : A.card = N) :
  ∑ s in powerset_len n A, 1 = binomial_coefficient N n :=
sorry

-- Proof that the number of ordered sequences of length N with n ones and (N - n) zeros is C_N^n
theorem ordered_sequences :
  (Nat.choose N n) = binomial_coefficient N n :=
sorry

-- Proof that the number of ways to place n balls into N boxes with no more than one ball per box is C_N^n
theorem place_balls :
  (finset.range N).choose n = binomial_coefficient N n :=
sorry

-- Proof that the number of non-decreasing paths on a grid from (0, 0) to (n, N - n) is C_N^n
theorem non_decreasing_paths (hn : 0 ≤ n) (hN : n ≤ N) :
  (Nat.choose N n) = binomial_coefficient N n :=
sorry

-- Proof that the number of different subsets D of set A with |D| = n is C_N^n
theorem subsets (hA : A.card = N) (D : finset ℕ) (hD : D.card = n) :
  D = binomial_coefficient N n :=
sorry

end unordered_samples_ordered_sequences_place_balls_non_decreasing_paths_subsets_l309_309003


namespace monotonic_decreasing_interval_l309_309338

def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x ∈ Set.Ici Real.exp 1 → (∀ y : ℝ, y ∈ Set.Ioi x → f y ≤ f x) :=
sorry

end monotonic_decreasing_interval_l309_309338


namespace bars_sold_this_week_l309_309634

-- Definitions based on conditions
def total_bars : Nat := 18
def bars_sold_last_week : Nat := 5
def bars_needed_to_sell : Nat := 6

-- Statement of the proof problem
theorem bars_sold_this_week : (total_bars - (bars_needed_to_sell + bars_sold_last_week)) = 2 := by
  -- proof goes here
  sorry

end bars_sold_this_week_l309_309634


namespace slope_angle_of_line_l309_309184

noncomputable def line := λ x : ℝ, (√3) * x - 4

theorem slope_angle_of_line :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < π ∧ tan θ = √3 ∧ θ = π / 3 :=
by
  use π / 3
  split
  exact Real.le_of_lt Real.pi_div_three_pos
  split
  exact Real.pi_div_three_lt_pi
  split
  exact Real.tan_pi_div_three
  rfl

end slope_angle_of_line_l309_309184


namespace div_expression_calc_l309_309479

theorem div_expression_calc :
  (3752 / (39 * 2) + 5030 / (39 * 10) = 61) :=
by
  sorry -- Proof of the theorem

end div_expression_calc_l309_309479


namespace Shiela_drawings_l309_309307

theorem Shiela_drawings (n_neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
    (h1 : n_neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
by 
  sorry

end Shiela_drawings_l309_309307


namespace no_such_arrangement_l309_309010

theorem no_such_arrangement :
  ¬∃ (a : Fin 111 → ℕ), (∀ i : Fin 111, a i ≤ 500 ∧ (∀ j k : Fin 111, j ≠ k → a j ≠ a k)) ∧ (∀ i : Fin 111, (a i % 10) = ((Finset.univ.sum (λ j, if j = i then 0 else a j)) % 10)) :=
by
  sorry

end no_such_arrangement_l309_309010


namespace sum_of_positive_integers_less_than_10_l309_309917

theorem sum_of_positive_integers_less_than_10 : (Finset.range 10).sum = 45 :=
by
  -- This is a statement where you would introduce the proof.
  -- I will use sorry to skip the actual proof.
  sorry

end sum_of_positive_integers_less_than_10_l309_309917


namespace find_principal_sum_l309_309830

theorem find_principal_sum
  (R : ℝ := 0.10) (T : ℝ := 2) (diff : ℝ := 150) :
  let SI := (P : ℝ) * R * T / 100,
      CI := (P : ℝ) * (1 + R)^T - P in
  ∃ (P : ℝ), CI - SI = diff ∧ P = 15000 := 
by
  let SI := (P : ℝ) * R * T / 100
  let CI := (P : ℝ) * (1 + R)^T - P
  use (150 / 0.01)
  sorry

end find_principal_sum_l309_309830


namespace even_perfect_square_factors_count_l309_309584

theorem even_perfect_square_factors_count :
  let a_choices := [2, 4, 6]
  let b_choices := [0, 2, 4]
  let c_choices := [0, 2, 4]
  (2^6 * 3^4 * 7^5).perfect_square_factors_count = 27 := by
  sorry

end even_perfect_square_factors_count_l309_309584


namespace find_a_and_b_l309_309554

theorem find_a_and_b (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a * b - 9) : a = 3 ∧ b = 3 :=
by
  sorry

end find_a_and_b_l309_309554


namespace expected_value_of_win_l309_309418

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l309_309418


namespace parallel_vectors_l309_309193

theorem parallel_vectors (a b : ℝ × ℝ) (x : ℝ) (h₁ : a = (2, x)) (h₂ : b = (x - 1, 1)) (h₃ : ∃ k : ℝ, b = k • a) :
  x = 2 ∨ x = -1 :=
by
  intro a b x h₁ h₂ h₃
  -- Proof omitted.
  sorry

end parallel_vectors_l309_309193


namespace angle_CHY_l309_309466

theorem angle_CHY (A B C X Y H : Type) [orthocenter_triangle A B C H X Y]
  (h1 : altitude A X)
  (h2 : altitude B Y)
  (h3 : intersect H X Y)
  (h4 : acute_triangle A B C)
  (hBAC : angle A B C = 53)
  (hABC : angle B A C = 82) :
  angle C H Y = 45 :=
by sorry

end angle_CHY_l309_309466


namespace inequality_afb_leq_bfa_l309_309266

theorem inequality_afb_leq_bfa
  (f : ℝ → ℝ)
  (differentiable_on_f : ∀ x > 0, differentiable_at ℝ f x)
  (nonneg_f : ∀ x, 0 < x → 0 ≤ f x)
  (ineq : ∀ x > 0, x * (deriv f x) + f x ≤ 0)
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : a < b):
  a * f b ≤ b * f a := 
begin
  sorry
end

end inequality_afb_leq_bfa_l309_309266


namespace trigonometric_inequality_l309_309938

theorem trigonometric_inequality
  {n : ℕ}
  (h_n : n > 1)
  (α β : Fin n → ℝ)
  (hα : ∀ i, 0 < α i)
  (hβ : ∀ i, 0 < β i)
  (sum_α : ∑ i in Finset.univ, α i = π)
  (sum_β : ∑ i in Finset.univ, β i = π) :
  (∑ i in Finset.univ, (cos (β i) / sin (α i))) ≤ (∑ i in Finset.univ, (cos (α i) / sin (α i))) := 
sorry

end trigonometric_inequality_l309_309938


namespace negation_equiv_l309_309190

variable (teachers : Type)
variable (is_excellent : teachers → Prop)
variable (is_poor : teachers → Prop)

-- Statement (6): All teachers are excellent at math
def statement_6 : Prop := ∀ t : teachers, is_excellent t

-- Statement (5): At least one teacher is poor at math
def statement_5 : Prop := ∃ t : teachers, is_poor t

-- Negation relation
theorem negation_equiv (exc_to_poor : ∀ t, (is_poor t ↔ ¬ is_excellent t)) :
  statement_6 ↔ ¬ statement_5 :=
by
  sorry

end negation_equiv_l309_309190


namespace product_of_repeating_decimal_l309_309132

-- Define the repeating decimal 0.3
def repeating_decimal : ℚ := 1 / 3
-- Define the question
def product (a b : ℚ) := a * b

-- State the theorem to be proved
theorem product_of_repeating_decimal :
  product repeating_decimal 8 = 8 / 3 :=
sorry

end product_of_repeating_decimal_l309_309132


namespace chocolate_chip_cookie_count_l309_309705

-- Let cookies_per_bag be the number of cookies in each bag
def cookies_per_bag : ℕ := 5

-- Let oatmeal_cookies be the number of oatmeal cookies
def oatmeal_cookies : ℕ := 2

-- Let num_baggies be the number of baggies
def num_baggies : ℕ := 7

-- Define the total number of cookies as num_baggies * cookies_per_bag
def total_cookies : ℕ := num_baggies * cookies_per_bag

-- Define the number of chocolate chip cookies as total_cookies - oatmeal_cookies
def chocolate_chip_cookies : ℕ := total_cookies - oatmeal_cookies

-- Prove that the number of chocolate chip cookies is 33
theorem chocolate_chip_cookie_count : chocolate_chip_cookies = 33 := by
  sorry

end chocolate_chip_cookie_count_l309_309705


namespace proof1_proof2_l309_309077

noncomputable def problem1 : ℝ :=
  2 * real.log 2 / real.log 3 
  - real.log (32 / 9) / real.log 3 
  + real.log 8 / real.log 3 
  - 5^(real.log 3 / real.log 5) 
  + (real.log 5 / real.log 10)^2 
  + (real.log 2 / real.log 10) * (real.log 50 / real.log 10)

theorem proof1 : problem1 = 0 := by
  sorry

noncomputable def problem2 : ℝ :=
  2^(-1/2) 
  + 1 / (0.064^(1/3)) 
  - (-4)^0 / real.sqrt 2 
  + 2^(-2) * real.sqrt(4 / 9)

theorem proof2 : problem2 = 8 / 3 := by
  sorry

end proof1_proof2_l309_309077


namespace remainder_when_divided_by_60_l309_309990

theorem remainder_when_divided_by_60 (k : ℤ) :
  let n := 60 * k - 1 in (n^2 + 2 * n + 3) % 60 = 2 := 
by
  sorry

end remainder_when_divided_by_60_l309_309990


namespace coefficient_x5_l309_309097

theorem coefficient_x5 (x : ℝ) :
  coefficient_of_x5 (expand ((x + 1) * (x^2 - x - 2)^3)) = -6 :=
by 
  sorry

end coefficient_x5_l309_309097


namespace complex_expression_evaluation_l309_309788

theorem complex_expression_evaluation :
  ( (1 + complex.I) / real.sqrt 2 ) ^ 2016 = 1 := 
by
  -- inserting some general placeholder for proof until actual solving
  sorry

end complex_expression_evaluation_l309_309788


namespace curve_equation_l309_309960

theorem curve_equation
  (a b : ℝ)
  (h1 : a * 0 ^ 2 + b * (5 / 3) ^ 2 = 2)
  (h2 : a * 1 ^ 2 + b * 1 ^ 2 = 2) :
  (16 / 25) * x^2 + (9 / 25) * y^2 = 1 := 
by {
  sorry
}

end curve_equation_l309_309960


namespace plane_covered_by_2014_circles_l309_309481

-- Definitions based on identified conditions
def radius := 1007
def circle (center : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2
def integral_y_line (y : ℝ) : Prop := ∃ n : ℤ, y = n

-- Problem statement
theorem plane_covered_by_2014_circles :
  ∀ (p : ℝ × ℝ), 
    (∃ centers : set (ℝ × ℝ), 
      (∀ c ∈ centers, integral_y_line c.2) ∧
      (∀ c ∈ centers, circle c radius p) ∧ 
      (centers.card = 2014)) :=
sorry

end plane_covered_by_2014_circles_l309_309481


namespace domain_of_g_eq_l309_309380

noncomputable def g (x : ℝ) : ℝ := (x + 2) / (Real.sqrt (x^2 - 5 * x + 6))

theorem domain_of_g_eq : 
  {x : ℝ | 0 < x^2 - 5 * x + 6} = {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_g_eq_l309_309380


namespace complement_of_A_in_U_l309_309285

open Set

def U := { x : ℕ | x ≥ 2 }
def A := { x : ℕ | x * x ≥ 5 }

theorem complement_of_A_in_U : compl U A = {2} :=
by
  sorry

end complement_of_A_in_U_l309_309285


namespace garden_table_ratio_l309_309430

theorem garden_table_ratio (x y : ℝ) (h₁ : x + y = 750) (h₂ : y = 250) : x / y = 2 :=
by
  -- Proof omitted
  sorry

end garden_table_ratio_l309_309430


namespace polygon_sides_eight_l309_309219

theorem polygon_sides_eight (n : ℕ) 
  (h₀ : ∑ (exterior_angles : 360)) 
  (h₁ : ∑ (interior_angles = 180 * (n - 2)) = 3 * ∑ (exterior_angles)) 
  : n = 8 := 
by 
  sorry

end polygon_sides_eight_l309_309219


namespace first_number_in_a10_is_91_l309_309289

theorem first_number_in_a10_is_91 :
  let a : ℕ → ℕ 
  := λ n, 1 + 2 * (n-1) * (n-1 + 1) / 2
  in a 10 = 91 := by
  sorry

end first_number_in_a10_is_91_l309_309289


namespace integral_1_integral_2_integral_3_l309_309122

-- 1st Integral Proof Problem
theorem integral_1 :
  ∫ (x^2 - 2 * sin x + 3 * exp x) dx = (x^3 / 3) + 2 * cos x + 3 * exp x + C :=
sorry

-- 2nd Integral Proof Problem
theorem integral_2 :
  ∫ (sec^2 x - 3 * cos x + 1) dx = (tan x - 3 * sin x + x + C) :=
sorry

-- 3rd Integral Proof Problem
theorem integral_3 :
  ∫ (csc^2 x + 7 * sin x - 2) dx = (-cot x - 7 * cos x - 2 * x + C) :=
sorry

end integral_1_integral_2_integral_3_l309_309122


namespace count_valid_numbers_l309_309391

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ a b => a + b) 0

def is_multiple (n : ℕ) (k : ℕ) : Prop :=
  n % k = 0

def satisfies_condition (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 100 ∧ is_multiple n (sum_of_digits n)

theorem count_valid_numbers : 
  ∃ (count : ℕ), count = {n | satisfies_condition n}.to_finset.card ∧ count = 24 := by
  sorry

end count_valid_numbers_l309_309391


namespace side_length_of_square_l309_309763

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l309_309763


namespace pr_eq_sq_l309_309322

variables {A B C D P Q R S : ℝ}

namespace trapezoid_problem

-- Assume the existence of a trapezoid with base AD and specific points on its legs.
def trapezoid_conditions (A B C D P Q R S : ℝ) : Prop :=
  A ≠ B ∧ 
  A ≠ C ∧ 
  A ≠ D ∧ 
  B ≠ C ∧ 
  B ≠ D ∧ 
  C ≠ D ∧
  P ∈ Ioo A D ∧
  Q ∈ Ioo C B ∧ 
  ∃ k, k > 0 ∧ AP / PD = k ∧ CQ / QB = k ∧
  (∃ R, R = line_intersection_of PQ AC) ∧
  (∃ S, S = line_intersection_of PQ BD)

-- The main theorem to be proved
theorem pr_eq_sq {A B C D P Q R S : ℝ} (h : trapezoid_conditions A B C D P Q R S) : PR = SQ :=
by sorry

end trapezoid_problem

end pr_eq_sq_l309_309322


namespace semicircle_circumference_is_correct_l309_309397

noncomputable def problem_statement : ℝ :=
  let length := 36
  let breadth := 20
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let diameter_semicircle := side_square
  let circumference_semicircle := (Real.pi * diameter_semicircle) / 2 + diameter_semicircle
  circumference_semicircle

theorem semicircle_circumference_is_correct :
  Real.round (problem_statement * 100) / 100 = 71.96 :=
sorry

end semicircle_circumference_is_correct_l309_309397


namespace find_x_eq_zero_l309_309509

theorem find_x_eq_zero (x : ℝ) : 
  (9^x + 8^x) / (6^x + 12^x) = 5 / 4 ↔ x = 0 :=
sorry

end find_x_eq_zero_l309_309509


namespace find_incircle_radius_l309_309294

-- Definitions of conditions
variables {A B C D O : Type}
variables [euclidean_space V] [isometric_space V]

-- Parameters given by problem
def midpoint (O : V) (A C : V) : Prop := 2 * O = A + C
def is_diameter (diam : line ℝ) (circ : circle ℝ) : Prop := 
  diam.length = 2 * circ.radius
def intersection (D : V) (line1 line2 : line ℝ) : Prop := D ∈ line1 ∧ D ∈ line2
def center_on_segment (circumcenter : V) (seg : segment ℝ) : Prop :=
  circumcenter ∈ seg

def radius_cond (OD DC : ℝ) : Prop := OD = 4 ∧ DC = 3 

-- Main statement
theorem find_incircle_radius
  {A B C D O : V} 
  (h_midpoint : midpoint O A C)
  (h_diameter : is_diameter (line_through A C) (circle_with_center O))
  (h_intersect : intersection D (line_through B C) (circle_with_center O))
  (h_center_on_seg : center_on_segment (circumcenter A B C) (segment_through B O))
  (h_radius_cond : radius_cond (dist O D) (dist D C)) :
  incircle_radius A B C = 4 * real.sqrt 55 / 11 := sorry

end find_incircle_radius_l309_309294


namespace coeff_m6n6_in_mn_12_l309_309816

open BigOperators

theorem coeff_m6n6_in_mn_12 (m n : ℕ) : 
  (∑ k in finset.range (13), (nat.choose 12 k) * m^k * n^(12 - k)) = 
  (nat.choose 12 6) * m^6 * n^6 :=
by sorry

end coeff_m6n6_in_mn_12_l309_309816


namespace no_increase_in_probability_l309_309035

def hunter_two_dogs_probability (p : ℝ) : ℝ :=
  let both_correct := p * p
  let one_correct := 2 * p * (1 - p) / 2
  both_correct + one_correct

theorem no_increase_in_probability (p : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ 1) :
  hunter_two_dogs_probability p = p :=
by
  sorry

end no_increase_in_probability_l309_309035


namespace power_mod_equiv_l309_309697

-- Define the main theorem
theorem power_mod_equiv {a n k : ℕ} (h₁ : a ≥ 2) (h₂ : n ≥ 1) :
  (a^k ≡ 1 [MOD (a^n - 1)]) ↔ (k % n = 0) :=
by sorry

end power_mod_equiv_l309_309697


namespace smallest_digit_divisible_by_9_l309_309141

theorem smallest_digit_divisible_by_9 : 
  ∃ (d : ℕ), 0 ≤ d ∧ d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ d = 2 :=
by
  use 2
  split
  { exact nat.zero_le _ }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end smallest_digit_divisible_by_9_l309_309141


namespace mn_parallel_pq_l309_309663

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309663


namespace opposite_of_grey_is_violet_l309_309773

def color: Type := {Y, G, O, V, B, K}

structure CubeView :=
  (top : color)
  (front : color)
  (right : color)

def view1 : CubeView := { top := Y, front := B, right := K }
def view2 : CubeView := { top := O, front := Y, right := K }
def view3 : CubeView := { top := O, front := V, right := K }

theorem opposite_of_grey_is_violet :
  (view1.top = Y ∧ view1.front = B ∧ view1.right = K) ∧
  (view2.top = O ∧ view2.front = Y ∧ view2.right = K) ∧
  (view3.top = O ∧ view3.front = V ∧ view3.right = K) →
  opposite_color G = V :=
by
    sorry

end opposite_of_grey_is_violet_l309_309773


namespace part1_part2_l309_309571

open Real

-- Part 1: Prove that f(x) > 0 for a = 1
theorem part1 (x : ℝ) : (exp x - 2 * x) > 0 :=
sorry

-- Part 2: Minimum and Maximum values of f(x) when a > 1/2
theorem part2 (a : ℝ) (h : a > 1/2) :
  ∃ x₁ x₂ ∈ set.Icc (0 : ℝ) (2 * a), 
    (∀ x ∈ set.Icc (0 : ℝ) (2 * a), f a x ≥ f a x₁) ∧ 
    (∀ x ∈ set.Icc (0 : ℝ) (2 * a), f a x ≤ f a x₂) ∧
    f a x₁ = 2 * a * (1 - log (2 * a)) ∧ 
    f a x₂ = exp (2 * a) - 4 * a^2  :=
sorry

-- Define the function from the problem
def f (a x : ℝ) : ℝ := exp x - 2 * a * x

end part1_part2_l309_309571


namespace find_n_values_l309_309511

theorem find_n_values :
  ∀ (n : ℕ), 
  (∃ (x : Fin n → ℝ), (∀ i, 0 < x i) ∧ (∑ i, x i = 9) ∧ (∑ i, (x i)⁻¹ = 1)) ↔ (n = 2 ∨ n = 3) :=
by
  intros n
  sorry

end find_n_values_l309_309511


namespace hunting_dog_strategy_l309_309033

theorem hunting_dog_strategy (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let single_dog_success_prob := p,
      both_dogs_same_path_prob := p^2,
      one_dog_correct_prob := p * (1 - p),
      combined_prob := both_dogs_same_path_prob + one_dog_correct_prob
  in combined_prob = single_dog_success_prob := 
by
  sorry

end hunting_dog_strategy_l309_309033


namespace functional_eq_solution_l309_309497

noncomputable def functional_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

theorem functional_eq_solution (f : ℝ → ℝ) (h : functional_solution f) :
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 :=
sorry

end functional_eq_solution_l309_309497


namespace john_dials_correct_number_probability_l309_309637

theorem john_dials_correct_number_probability :
  (297 ∈ {297, 298, 299} ∧ 298 ∈ {297, 298, 299} ∧ 299 ∈ {297, 298, 299}) →
  (0 ∈ {0, 1, 2, 6, 7} ∧ 1 ∈ {0, 1, 2, 6, 7} ∧ 2 ∈ {0, 1, 2, 6, 7} ∧ 6 ∈ {0, 1, 2, 6, 7} ∧ 7 ∈ {0, 1, 2, 6, 7}) →
  ∃ p : ℝ, p = 1 / 360 :=
by
  sorry

end john_dials_correct_number_probability_l309_309637


namespace inequality_holds_for_all_x_l309_309560

variable (a x : ℝ)

theorem inequality_holds_for_all_x (h : a ∈ Set.Ioc (-2 : ℝ) 4): ∀ x : ℝ, (x^2 - a*x + 9 > 0) :=
sorry

end inequality_holds_for_all_x_l309_309560


namespace value_of_expression_l309_309205

theorem value_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) : 
  real.sqrt (x^2 - 4 * x + 4) + real.sqrt (x * y - 3 * x + y - 3) = 3 :=
sorry

end value_of_expression_l309_309205


namespace triangle_area_approx_l309_309602

theorem triangle_area_approx {a b c : ℝ} (h₁ : a = 36) (h₂ : b = 34) (h₃ : c = 20) : 
  let s := (a + b + c) / 2,
      area := Real.sqrt ( s * (s - a) * (s - b) * (s - c)) in
  abs (area - 333.73) < 0.01 :=
by
  sorry

end triangle_area_approx_l309_309602


namespace eval_expression_l309_309401

theorem eval_expression : (3 - real.pi)^0 - (3:ℝ)⁻¹ = (2:ℝ) / 3 := by
  sorry

end eval_expression_l309_309401


namespace planes_intersect_necessary_for_skew_lines_l309_309580

-- Conditions: planes α and β are non-intersecting, lines m and n are perpendicular to planes α and β respectively.
variables {α β : Plane} {m n : Line}
axiom non_intersecting_planes : ¬(α ∩ β)
axiom perpendicular_line_m : m ⊥ α
axiom perpendicular_line_n : n ⊥ β

-- Statement: "planes α and β intersect" is a necessary but not sufficient condition for "lines m and n are skew lines".
theorem planes_intersect_necessary_for_skew_lines (h : α ∩ β) : ¬(α ∩ β) ∧ skew_lines m n := sorry

end planes_intersect_necessary_for_skew_lines_l309_309580


namespace child_tickets_sold_l309_309042

theorem child_tickets_sold
  (A C : ℕ) 
  (h1 : A + C = 900)
  (h2 : 7 * A + 4 * C = 5100) :
  C = 400 :=
by
  sorry

end child_tickets_sold_l309_309042


namespace parallel_MN_PQ_l309_309654

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309654


namespace cos_b_eq_one_div_sqrt_two_l309_309542

variable {a b c : ℝ} -- Side lengths
variable {A B C : ℝ} -- Angles in radians

-- Conditions of the problem
variables (h1 : c = 2 * a) 
          (h2 : b^2 = a * c) 
          (h3 : a^2 + b^2 = c^2 - 2 * a * b * Real.cos C)
          (h4 : A + B + C = Real.pi)

theorem cos_b_eq_one_div_sqrt_two
    (h1 : c = 2 * a)
    (h2 : b = a * Real.sqrt 2)
    (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
    (h4 : A + B + C = Real.pi )
    : Real.cos B = 1 / Real.sqrt 2 := 
sorry

end cos_b_eq_one_div_sqrt_two_l309_309542


namespace intersection_condition_l309_309186

theorem intersection_condition (m : ℝ) : 
  let A := ({2, 7, -4 * m + (m + 2) * complex.i} : set ℂ)
  let B := ({8, 3} : set ℝ)
  A ∩ ↑B ≠ ∅ 
  → m = -2 := 
by
  sorry

end intersection_condition_l309_309186


namespace exist_tangent_circles_l309_309325

noncomputable def locus_centers_of_tangent_circles (a b : ℝ) : Prop :=
  40 * a ^ 2 + 49 * b ^ 2 - 48 * a - 64 = 0

theorem exist_tangent_circles (a b : ℝ) :
  (∀ r ≥ 0, ∃ C, (C.center = ⟨a, b⟩ ∧ C.radius = r ∧ externally_tangent C C1 ∧ internally_tangent C C3)) → 
  locus_centers_of_tangent_circles a b := 
sorry

end exist_tangent_circles_l309_309325


namespace problem_l309_309694

noncomputable def F (x : ℝ) : ℝ :=
  (1 + x^2 - x^3) / (2 * x * (1 - x))

theorem problem (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  F x + F ((x - 1) / x) = 1 + x :=
by
  sorry

end problem_l309_309694


namespace side_length_of_square_l309_309755

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l309_309755


namespace no_solutions_for_equation_l309_309909

theorem no_solutions_for_equation :
  ∀ (x y z : ℤ), ¬(x^3 - 5 * x = 1728^y * 1733^z - 17) :=
by
  intro x y z
  sorry

end no_solutions_for_equation_l309_309909


namespace limit_a_x_to_ln_a_l309_309719

theorem limit_a_x_to_ln_a (a : ℝ) (ha : 0 < a) :
  tendsto (fun x => (a^x - 1) / x) (nhds 0) (nhds (Real.log a)) := 
by
  sorry

end limit_a_x_to_ln_a_l309_309719


namespace relationship_between_a_and_b_l309_309152

def a : ℤ := (-12) * (-23) * (-34) * (-45)
def b : ℤ := (-123) * (-234) * (-345)

theorem relationship_between_a_and_b : a > b := by
  sorry

end relationship_between_a_and_b_l309_309152


namespace expected_value_equals_51_l309_309426

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l309_309426


namespace smallest_n_satisfies_l309_309791

def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1.5 else 1 / (n^2 - 1)

def sequence_sum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, sequence (k + 1)

theorem smallest_n_satisfies (n : ℕ) (h : n = 100) : 
  | sequence_sum n - 2.25 | < 0.01 :=
by
  sorry

end smallest_n_satisfies_l309_309791


namespace eccentricity_of_ellipse_l309_309565

theorem eccentricity_of_ellipse
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (ellipse_equation : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 → Prop)
  (tangent_condition : ∀ x y : ℝ, bx - ay + 2ab = 0 → Prop)
  (distance_condition : ∀ x y : ℝ, (2ab / sqrt (a^2 + b^2)) = a → Prop) :
  (sqrt (1 - b^2 / a^2)) = sqrt 6 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l309_309565


namespace max_segments_edges_without_triangles_or_quadrangles_l309_309151

noncomputable def max_segments_no_triangles_quadrangles : ℕ :=
  let points := fin 10 in
  let edges := (points.product points).filter (λ e, e.1 ≠ e.2) in
  25

theorem max_segments_edges_without_triangles_or_quadrangles :
  max_segments_no_triangles_quadrangles = 25 := 
sorry

end max_segments_edges_without_triangles_or_quadrangles_l309_309151


namespace image_of_rectangle_l309_309681

open Real
open Set

noncomputable def transformation (x y : ℝ) : ℝ × ℝ :=
  (x^2 - y^2, x * y)

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (0, 3)

def u_O : ℝ × ℝ := transformation O.1 O.2
def u_A : ℝ × ℝ := transformation A.1 A.2
def u_P : ℝ × ℝ := transformation P.1 P.2
def u_B : ℝ × ℝ := transformation B.1 B.2

theorem image_of_rectangle :
  u_O = (0, 0) ∧
  u_A = (4, 0) ∧
  u_P = (-5, 6) ∧
  u_B = (-9, 0) ∧
  -- Here we need to describe the boundary transformations
  ∃ (curves : set (ℝ × ℝ)),
    curves = {p | ∃ x y, p = transformation x y ∧
                    ((0 ≤ x ∧ x ≤ 2 ∧ y = 0) ∨
                     (0 ≤ y ∧ y ≤ 3 ∧ x = 2) ∨
                     (0 ≤ x ∧ x ≤ 2 ∧ y = 3) ∨
                     (0 ≤ y ∧ y ≤ 3 ∧ x = 0))} ∧
    (0, 0) ∈ curves ∧
    (4, 0) ∈ curves ∧
    (-5, 6) ∈ curves ∧
    (-9, 0) ∈ curves := 
sorry

end image_of_rectangle_l309_309681


namespace highest_salary_grade_l309_309473

theorem highest_salary_grade 
  (s : ℕ) 
  (h1 : ∀ s, 1 ≤ s ∧ s ≤ s_max) 
  (h2 : ∀ s, p s = 7.50 + 0.25 * (s - 1))
  (h3 : p s_max = p 1 + 1.25) 
  : s_max = 6 := 
  by sorry

end highest_salary_grade_l309_309473


namespace A_gt_B_and_C_lt_A_l309_309375

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end A_gt_B_and_C_lt_A_l309_309375


namespace p_sufficient_but_not_necessary_for_q_l309_309155

variables (x y : ℕ)

def p : Prop := x + y ≠ 4
def q : Prop := x ≠ 1 ∨ y ≠ 3

theorem p_sufficient_but_not_necessary_for_q : (p → q) ∧ ¬(q → p) :=
by {
  sorry
}


end p_sufficient_but_not_necessary_for_q_l309_309155


namespace possible_values_of_ab_plus_ac_plus_bc_l309_309261

-- Definitions and conditions
variables {a b c : ℝ} 

-- The main theorem statement
theorem possible_values_of_ab_plus_ac_plus_bc (h : a + b + c = 1) : 
  ∃ (S : set ℝ), S = (-∞, 1/2] ∧ (ab + ac + bc) ∈ S := 
sorry

end possible_values_of_ab_plus_ac_plus_bc_l309_309261


namespace system_of_equations_unique_solution_l309_309103

theorem system_of_equations_unique_solution :
  (∃ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7) →
  (∀ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7 →
    x = 26 / 5 ∧ y = 9 / 5) := 
by {
  -- Proof to be provided
  sorry
}

end system_of_equations_unique_solution_l309_309103


namespace product_of_fractions_l309_309076

theorem product_of_fractions :
  (2 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) * (5 / 6 : ℚ) * (6 / 7 : ℚ) * (7 / 8 : ℚ) = 1 / 4 :=
by
  sorry

end product_of_fractions_l309_309076


namespace hendricks_payment_l309_309194

variable (Hendricks Gerald : ℝ)
variable (less_percent : ℝ) (amount_paid : ℝ)

theorem hendricks_payment (h g : ℝ) (h_less_g : h = g * (1 - less_percent)) (g_val : g = amount_paid) (less_percent_val : less_percent = 0.2) (amount_paid_val: amount_paid = 250) :
h = 200 :=
by
  sorry

end hendricks_payment_l309_309194


namespace area_of_triangle_ABC_l309_309738

-- Define the triangle and conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (BC AD : ℝ) (triangle_ABC : ∃ Δ : Triangle A B C, True)

-- Hypothetical foot of altitude condition
axiom foot_of_altitude_condition : D = foot_of_altitude A B C

-- Hypothetical 45-45-90 triangle condition
axiom forty_five_ninety_triangles : divides_into_454590 A B C D

-- Define the distance from A to D
axiom AD_distance : dist A D = Real.sqrt 2

theorem area_of_triangle_ABC (h1 : foot_of_altitude_condition) (h2 : forty_five_ninety_triangles) (h3 : AD_distance) :
  area of_triangle_ABC = 1 :=
by
  sorry

end area_of_triangle_ABC_l309_309738


namespace store_discount_discrepancy_l309_309443

theorem store_discount_discrepancy :
  (original_price : ℝ) → 
  let discount1 := 0.30 * original_price in
  let first_reduced_price := original_price - discount1 in
  let discount2 := 0.15 * first_reduced_price in
  let final_price := first_reduced_price - discount2 in
  -- actual discount percentage
  let actual_discount_percentage := 100.0 - (final_price / original_price * 100.0) in
  actual_discount_percentage = 40.5 ∧ 
  -- claimed discount percentage
  let claimed_discount_percentage := 45.0 in
  let difference := claimed_discount_percentage - actual_discount_percentage in
  difference = 4.5 :=
by
  sorry

end store_discount_discrepancy_l309_309443


namespace flag_length_equals_ten_l309_309475

def fabric1 := (8, 5)
def fabric2 := (10, 7)
def fabric3 := (5, 5)

def flag_height := 9

theorem flag_length_equals_ten : 
  ∃ (length : ℕ), 
  (fabric2.snd > flag_height ∨ (fabric2.snd + fabric3.snd > flag_height)) ∧ 
  length = fabric2.fst :=
begin
  sorry
end

end flag_length_equals_ten_l309_309475


namespace part1_part2_l309_309966

noncomputable def f (a θ x : ℝ) : ℝ := (a + 2 * (cos x) ^ 2) * cos (2 * x + θ)

theorem part1 (a θ : ℝ) (h1 : f a θ (π / 4) = 0) (h2 : ∀ x, f a θ x = -f a θ (-x))
  (h_cond1 : θ ∈ Set.Ioo 0 π) :
  a = -1 ∧ θ = π / 2 :=
by
  sorry

theorem part2 (α : ℝ)
  (h3 : f (-1) (π / 2) (α / 4) = -2 / 5)
  (h_cond2 : α ∈ Set.Ioo (π / 2) π) :
  sin (α + π / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  let f := fun x => - (1 / 2) * sin (4 * x)
  sorry

end part1_part2_l309_309966


namespace directrix_of_parabola_l309_309121

def parabola (x : ℝ) : ℝ := (x^2 - 10 * x + 21) / 14

def directrix_eq : ℝ := -53 / 14

theorem directrix_of_parabola :
  ∀ x : ℝ, directrix_eq = -7 / 2 - 2 / 7 := 
sorry

end directrix_of_parabola_l309_309121


namespace sum_f_from_1_to_2008_l309_309939

noncomputable def f : ℝ → ℝ := sorry

axiom symmetry_about_point : ∀ x, f(x) = f(-3/4 - (x + 3/4))
axiom periodicity_3_2 : ∀ x, f(x) = -f(x + 3 / 2)
axiom f_neg1_eq_1 : f(-1) = 1
axiom f_zero_eq_neg2 : f(0) = -2

theorem sum_f_from_1_to_2008 : (Finset.range 2008).sum (λ i, f(i+1)) = 1 :=
by
  sorry

end sum_f_from_1_to_2008_l309_309939


namespace jessica_can_mail_letter_l309_309636

-- Define the constants
def paper_weight := 1/5 -- each piece of paper weighs 1/5 ounce
def envelope_weight := 2/5 -- envelope weighs 2/5 ounce
def num_papers := 8

-- Calculate the total weight
def total_weight := num_papers * paper_weight + envelope_weight

-- Define stamping rates
def international_rate := 2 -- $2 per ounce internationally

-- Calculate the required postage
def required_postage := total_weight * international_rate

-- Define the available stamp values
inductive Stamp
| one_dollar : Stamp
| fifty_cents : Stamp

-- Function to calculate the total value of a given stamp combination
def stamp_value : List Stamp → ℝ
| [] => 0
| (Stamp.one_dollar :: rest) => 1 + stamp_value rest
| (Stamp.fifty_cents :: rest) => 0.5 + stamp_value rest

-- State the theorem to be proved
theorem jessica_can_mail_letter :
  ∃ stamps : List Stamp, stamp_value stamps = required_postage := by
sorry

end jessica_can_mail_letter_l309_309636


namespace impossible_arrangement_l309_309012

theorem impossible_arrangement : 
  ∀ (a : Fin 111 → ℕ), (∀ i, a i ≤ 500) → (∀ i j, i ≠ j → a i ≠ a j) → 
  ¬ ∀ i : Fin 111, (a i % 10 = ((Finset.univ.sum (λ j => if j = i then 0 else a j)) % 10)) :=
by 
  sorry

end impossible_arrangement_l309_309012


namespace infinitely_many_perfect_squares_of_form_l309_309255

theorem infinitely_many_perfect_squares_of_form (k : ℕ) (h : k > 0) : 
  ∃ (n : ℕ), ∃ m : ℕ, n * 2^k - 7 = m^2 :=
by
  sorry

end infinitely_many_perfect_squares_of_form_l309_309255


namespace minimum_value_of_2a5_a4_l309_309225

variable {a : ℕ → ℝ} {q : ℝ}

-- Defining that the given sequence is geometric, i.e., a_{n+1} = a_n * q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

-- The condition given in the problem is
def condition (a : ℕ → ℝ) : Prop :=
2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

-- The sequence is positive
def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

theorem minimum_value_of_2a5_a4 (h_geom : is_geometric_sequence a q) (h_cond : condition a) (h_pos : positive_sequence a) (h_q : q > 0) :
  2 * a 5 + a 4 = 12 * Real.sqrt 3 :=
sorry

end minimum_value_of_2a5_a4_l309_309225


namespace infinite_n_divisible_by_3_l309_309722

-- Define the concept of distinct odd prime divisors
def odd_prime_divisors (n : ℕ) : Finset ℕ :=
  (Finset.filter (λ p, p ∣ n ∧ Nat.Prime p ∧ p % 2 = 1) (Finset.range (n + 1)))

-- Define the number of distinct odd prime divisors
def a (n : ℕ) : ℕ :=
  (odd_prime_divisors (n * (n + 3))).card

-- The main theorem stating there are infinitely many n such that a(n) is divisible by 3
theorem infinite_n_divisible_by_3 : ∃ᶠ (n : ℕ), a n % 3 = 0 :=
sorry

end infinite_n_divisible_by_3_l309_309722


namespace probability_greater_l309_309451

-- Define the conditions
def uniformly_distributed (a b : ℝ) (x : ℝ) : Prop :=
  a ≤ x ∧ x ≤ b -- specifies that x is uniformly distributed in the interval [a, b]

variables {x y : ℝ}

-- Specific conditions for the problem
def alice_statement (x : ℝ) : Prop := uniformly_distributed (1/2) 1 x
def bob_statement (y : ℝ) : Prop := uniformly_distributed (3/4) 1 y
def alice_concede (x : ℝ) : Prop := uniformly_distributed (1/2) (7/8) x

-- The main statement to prove
theorem probability_greater (hx : alice_concede x) (hy : bob_statement y) : 
  let area_region := (7/8 - 1/2) * (1 - 3/4) in
  let area_triangle := (1/2) * (1/8) * (1/8) in
  ((area_region - area_triangle) / area_region) = 11 / 12 :=
by
  sorry

end probability_greater_l309_309451


namespace john_avg_increase_l309_309250

theorem john_avg_increase (a b c d : ℝ) (h₁ : a = 90) (h₂ : b = 85) (h₃ : c = 92) (h₄ : d = 95) :
    let initial_avg := (a + b + c) / 3
    let new_avg := (a + b + c + d) / 4
    new_avg - initial_avg = 1.5 :=
by
  sorry

end john_avg_increase_l309_309250


namespace sampling_methods_correct_pairing_l309_309301

theorem sampling_methods_correct_pairing :
  let number_of_students := 10000 in
  let high_school_students := 2000 in
  let middle_school_students := 4500 in
  let elementary_school_students := 3500 in
  let student_sample_size := 200 in
  let total_products := 1002 in
  let inspected_products := 20 in
  let sampling_method_1 := "Stratified" in
  let sampling_method_2 := "Systematic" in
  (number_of_students = high_school_students + middle_school_students + elementary_school_students) →
  student_sample_size ≤ number_of_students →
  inspected_products ≤ total_products →
  (sampling_method_1 = "Stratified" ∧ sampling_method_2 = "Systematic") :=
begin
  sorry
end

end sampling_methods_correct_pairing_l309_309301


namespace strictly_decreasing_function_exists_l309_309116

theorem strictly_decreasing_function_exists (k : ℝ) (h : ∀ x : ℝ, 0 < x) :
  k > 0 → (∃ g : ℝ → ℝ, (∀ x > 0, g(x) > 0) ∧ (∀ x > 0, g(x) > g(x + g(x))) ∧ (∀ x > 0, g(x) ≥ k * g(x + g(x)))) ↔ k ≤ 1 :=
by
  sorry

end strictly_decreasing_function_exists_l309_309116


namespace expected_value_of_win_is_51_l309_309421

noncomputable def expected_value_of_win : ℝ :=
  (∑ n in (finset.range 8).map (λ x, x + 1), (1/8) * 2 * (n : ℝ)^2)

theorem expected_value_of_win_is_51 : expected_value_of_win = 51 := 
by 
  sorry

end expected_value_of_win_is_51_l309_309421


namespace side_length_of_square_l309_309768

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309768


namespace beth_cookie_cost_l309_309932

theorem beth_cookie_cost
  (shared_dough : ℕ)
  (alex_length : ℕ)
  (alex_width : ℕ)
  (alex_cookies : ℕ)
  (alex_price_each : ℕ)
  (beth_cookies : ℕ)
  (alex_total_dough : shared_dough = alex_length * alex_width * alex_cookies)
  (alex_total_earnings : alex_price_each * alex_cookies = 500) :
  (500 / beth_cookies = 31.25) :=
sorry

end beth_cookie_cost_l309_309932


namespace sum_five_consecutive_l309_309316

theorem sum_five_consecutive (n : ℤ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) = 5 * n + 10 := by
  sorry

end sum_five_consecutive_l309_309316


namespace hyperbola_center_l309_309898

theorem hyperbola_center (x y : ℝ) :
  ( ∃ (h k : ℝ), ∀ (x y : ℝ), (4 * x - 8)^2 / 9^2 - (5 * y - 15)^2 / 7^2 = 1 → (h, k) = (2, 3) ) :=
by
  existsi 2
  existsi 3
  intros x y h
  sorry

end hyperbola_center_l309_309898


namespace polynomial_expansion_l309_309113

noncomputable def poly1 (z : ℝ) : ℝ := 3 * z ^ 3 + 2 * z ^ 2 - 4 * z + 1
noncomputable def poly2 (z : ℝ) : ℝ := 2 * z ^ 4 - 3 * z ^ 2 + z - 5
noncomputable def expanded_poly (z : ℝ) : ℝ := 6 * z ^ 7 + 4 * z ^ 6 - 4 * z ^ 5 - 9 * z ^ 3 + 7 * z ^ 2 + z - 5

theorem polynomial_expansion (z : ℝ) : poly1 z * poly2 z = expanded_poly z := by
  sorry

end polynomial_expansion_l309_309113


namespace min_f_value_l309_309127

noncomputable def f : ℝ → ℝ := λ x, (Real.tan x)^2 - 4 * Real.tan x - 12 * Real.cot x + 9 * (Real.cot x)^2 - 3

theorem min_f_value : ∃ x ∈ Ioo (-Real.pi / 2) 0, f x = 3 + 8 * Real.sqrt 3 := 
sorry

end min_f_value_l309_309127


namespace max_reciprocals_l309_309962

noncomputable def f (a x : ℝ) : ℝ := log a (x + 3) - 1

theorem max_reciprocals (a : ℝ) (m n : ℝ) (hm : m < 0) (hn : n < 0) :
  (f a (-2) = -1) → (m * (-2) + n * (-1) = 1) →
  (∃ x : ℝ, (x = -3 - 2 * real.sqrt 2) ∧ 
    (∀ y : ℝ, (y = (1 / m) + (1 / n) → y ≤ -3 - 2 * real.sqrt 2))) :=
begin
  sorry
end

end max_reciprocals_l309_309962


namespace sum_of_squares_of_solutions_l309_309918

   theorem sum_of_squares_of_solutions :
     (∑ x in ({x : ℝ | ∣x^2 - x + 1 / 2024∣ = 1 / 2024}).to_finset, x^2) = 1011 / 506 :=
   by
     sorry
   
end sum_of_squares_of_solutions_l309_309918


namespace exists_sum_of_quintessentially_special_l309_309890

def quintessentially_special (x : ℝ) : Prop :=
  ∀ d ∈ x.to_digits, d = 5 ∨ d = 0

theorem exists_sum_of_quintessentially_special :
  ∃ (n : ℕ) (xs : Fin n → ℝ), (∀ i, quintessentially_special (xs i)) ∧ (∑ i, xs i = 1) ∧ n = 5 := 
by 
  sorry

end exists_sum_of_quintessentially_special_l309_309890


namespace range_of_x_l309_309926

theorem range_of_x :
  (∀ t : ℝ, |t - 3| + |2 * t + 1| ≥ |2 * x - 1| + |x + 2|) →
  (-1/2 ≤ x ∧ x ≤ 5/6) :=
by
  intro h 
  sorry

end range_of_x_l309_309926


namespace coeff_m6n6_in_m_plus_n_pow_12_l309_309818

theorem coeff_m6n6_in_m_plus_n_pow_12 : 
  (∃ c : ℕ, (m + n)^12 = c * m^6 * n^6 + ...) → c = 924 := by
sorry

end coeff_m6n6_in_m_plus_n_pow_12_l309_309818


namespace no_triangle_satisfies_condition_l309_309903

theorem no_triangle_satisfies_condition (x y z : ℝ) (h_tri : x + y > z ∧ x + z > y ∧ y + z > x) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by
  sorry

end no_triangle_satisfies_condition_l309_309903


namespace vector_subtraction_identity_l309_309072

variables (a b : ℝ)

theorem vector_subtraction_identity (a b : ℝ) :
  ((1 / 2) * a - b) - ((3 / 2) * a - 2 * b) = b - a :=
by
  sorry

end vector_subtraction_identity_l309_309072


namespace impossible_arrangement_l309_309015

-- Definitions for the problem
def within_range (n : ℕ) : Prop := n > 0 ∧ n ≤ 500
def distinct (l : List ℕ) : Prop := l.Nodup

-- The main problem statement
theorem impossible_arrangement :
  ∀ (l : List ℕ),
  l.length = 111 →
  l.All within_range →
  distinct l →
  ¬(∀ (k : ℕ) (h : k < l.length), (l.get ⟨k, h⟩) % 10 = (l.sum - l.get ⟨k, h⟩) % 10) :=
by
  intros l length_cond within_range_cond distinct_cond condition
  sorry

end impossible_arrangement_l309_309015


namespace count_consecutive_integers_in_list_l309_309703

/-- We are given that List K consists of consecutive integers. 
    -4 is the least integer in list K.
    The range of the positive integers in list K is 4.
    We need to prove that the number of consecutive integers in list K is 10. -/
theorem count_consecutive_integers_in_list 
  (K : List ℤ) 
  (h1 : (∀ (n : ℤ), n ∈ K ↔ -4 ≤ n ∧ n ≤ 5))
  (h2 : (∃ m : ℤ, m ∈ K ∧ m = -4))
  (h3 : (∃ p1 p2 : ℤ, p1 ∈ K ∧ p2 ∈ K ∧ 1 ≤ p1 ∧ p2 ≤ 5 ∧ p2 - p1 = 4)) : 
  K.length = 10 := 
begin
  sorry
end

end count_consecutive_integers_in_list_l309_309703


namespace min_positive_period_abs_tan_2x_l309_309774

theorem min_positive_period_abs_tan_2x : 
  ∃ T > 0, (∀ x : ℝ, |Real.tan (2 * x) + T| = |Real.tan (2 * x)|)
  ∧ (∀ T' > 0, (∀ x : ℝ, |Real.tan (2 * (x + T'))| = |Real.tan (2 * x) → T' ≥ T)) :=
sorry

end min_positive_period_abs_tan_2x_l309_309774


namespace num_comics_bought_l309_309302

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end num_comics_bought_l309_309302


namespace total_toothpicks_l309_309808

def toothpick_grid (height width : ℕ) : ℕ :=
  let horizontal := (height + 1) * width
  let vertical := (width + 1) * height
  let diagonal := height * width
  horizontal + vertical + diagonal

theorem total_toothpicks (height width : ℕ) (h_height : height = 25) (h_width : width = 15) :
  toothpick_grid height width = 1165 :=
by
  subst h_height
  subst h_width
  unfold toothpick_grid
  simp
  sorry

end total_toothpicks_l309_309808


namespace negation_of_one_odd_l309_309743

-- Given a, b, c are natural numbers
def exactly_one_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 1)

def not_exactly_one_odd (a b c : ℕ) : Prop :=
  ¬ exactly_one_odd a b c

def at_least_two_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1) ∨
  (a % 2 = 1 ∧ c % 2 = 1) ∨
  (b % 2 = 1 ∧ c % 2 = 1)

def all_even (a b c : ℕ) : Prop :=
  (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0)

theorem negation_of_one_odd (a b c : ℕ) : ¬ exactly_one_odd a b c ↔ all_even a b c ∨ at_least_two_odd a b c := by
  sorry

end negation_of_one_odd_l309_309743


namespace star_three_and_four_l309_309896

def star (a b : ℝ) : ℝ := 4 * a + 5 * b - 2 * a * b

theorem star_three_and_four : star 3 4 = 8 :=
by
  sorry

end star_three_and_four_l309_309896


namespace sports_parade_children_l309_309518

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l309_309518


namespace mean_value_theorem_for_integrals_l309_309699

variable {a b : ℝ} (f : ℝ → ℝ)

theorem mean_value_theorem_for_integrals (h_cont : ContinuousOn f (Set.Icc a b)) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) :=
sorry

end mean_value_theorem_for_integrals_l309_309699


namespace B_time_to_complete_work_l309_309827

-- Declare the variables.
variables (A B C : ℝ)

-- Conditions as Lean statements.
def condition1 : Prop := A = (3/8) * B
def condition2 : Prop := C = (15/28) * B
def together_work : Prop := 15 * (A + B + C) = 1  -- Assuming "1" is the total work.

-- The theorem that encapsulates the problem.
theorem B_time_to_complete_work (h1 : condition1) (h2 : condition2) (h3 : together_work) : 
  ∃ t : ℝ, t = 141 := 
sorry

end B_time_to_complete_work_l309_309827


namespace TV_price_net_change_l309_309831

theorem TV_price_net_change (P : ℝ) : 
  let decreased_price := 0.80 * P
  let final_price := 1.55 * decreased_price
  in final_price - P = 0.24 * P :=
by
  sorry

end TV_price_net_change_l309_309831


namespace percent_increase_large_small_semicircles_l309_309048

theorem percent_increase_large_small_semicircles (length width : ℝ) (h_length : length = 10) (h_width : width = 6) : 
  let area_large_semicircles := 2 * (1/2) * Real.pi * (length / 2)^2,
      area_small_semicircles := 2 * (1/2) * Real.pi * (width / 2)^2,
      percent_increase := ((area_large_semicircles / area_small_semicircles) - 1) * 100 in
  percent_increase = 178 := 
by 
  sorry

end percent_increase_large_small_semicircles_l309_309048


namespace side_length_of_square_l309_309754

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309754


namespace rainfall_on_wednesday_l309_309476

theorem rainfall_on_wednesday 
  (rain_on_monday : ℝ)
  (rain_on_tuesday : ℝ)
  (total_rain : ℝ) 
  (hmonday : rain_on_monday = 0.16666666666666666) 
  (htuesday : rain_on_tuesday = 0.4166666666666667) 
  (htotal : total_rain = 0.6666666666666666) :
  total_rain - (rain_on_monday + rain_on_tuesday) = 0.0833333333333333 :=
by
  -- Proof would go here
  sorry

end rainfall_on_wednesday_l309_309476


namespace ratio_CQ_QA_l309_309240

open Function

variable {A B C D N Q : Point}

-- Assuming points on the plane and required lengths and properties
variables (AB AC : ℝ)
variables (hAB : AB = 24) (hAC : AC = 15)

-- BD/DC ratio according to the Angle Bisector Theorem
axiom hBD_DC_ratio : ∃ BD DC, BD / DC = AB / AC

-- N is the midpoint of AD
axiom hMidpoint_N : midpoint N A D

-- Q is the point of intersection of AC and BN
axiom hQ_intersection : ∃ Q, lies_on Q AC ∧ lies_on Q BN

-- Prove the ratio CQ/QA is 1/1, hence x+y = 2
theorem ratio_CQ_QA (hAB_AC : AB = 24 ∧ AC = 15 
  ∧ midpoint N A D
  ∧ ∃ Q, lies_on Q AC ∧ lies_on Q BN):
  (∃ x y, (x + y = 2) ∧ x.gcd y = 1 
  ∧ CQ / QA = (x:ℝ) / (y:ℝ)) :=
sorry

end ratio_CQ_QA_l309_309240


namespace infinite_n_divisible_by_3_l309_309721

-- Define the concept of distinct odd prime divisors
def odd_prime_divisors (n : ℕ) : Finset ℕ :=
  (Finset.filter (λ p, p ∣ n ∧ Nat.Prime p ∧ p % 2 = 1) (Finset.range (n + 1)))

-- Define the number of distinct odd prime divisors
def a (n : ℕ) : ℕ :=
  (odd_prime_divisors (n * (n + 3))).card

-- The main theorem stating there are infinitely many n such that a(n) is divisible by 3
theorem infinite_n_divisible_by_3 : ∃ᶠ (n : ℕ), a n % 3 = 0 :=
sorry

end infinite_n_divisible_by_3_l309_309721


namespace side_length_of_square_l309_309751

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309751


namespace product_of_bc_l309_309800

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end product_of_bc_l309_309800


namespace series_sum_l309_309105

theorem series_sum (n : Nat) : 
  (∑ k in Finset.range n, 1 / ((3 * (k + 1) - 2) * (3 * (k + 1) + 1))) = n / (3 * n + 1) :=
by
  sorry

end series_sum_l309_309105


namespace coeff_m6n6_in_mn_12_l309_309815

open BigOperators

theorem coeff_m6n6_in_mn_12 (m n : ℕ) : 
  (∑ k in finset.range (13), (nat.choose 12 k) * m^k * n^(12 - k)) = 
  (nat.choose 12 6) * m^6 * n^6 :=
by sorry

end coeff_m6n6_in_mn_12_l309_309815


namespace product_of_repeating_decimal_l309_309131

theorem product_of_repeating_decimal (x : ℚ) (h : x = 1 / 3) : (x * 8) = 8 / 3 := by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l309_309131


namespace isosceles_triangle_angles_l309_309612

theorem isosceles_triangle_angles (a b : ℝ) (h₁ : a = 80 ∨ b = 80) (h₂ : a + b + c = 180) (h_iso : a = b ∨ a = c ∨ b = c) :
  (a = 80 ∧ b = 20 ∧ c = 80)
  ∨ (a = 80 ∧ b = 80 ∧ c = 20)
  ∨ (a = 50 ∧ b = 50 ∧ c = 80) :=
by sorry

end isosceles_triangle_angles_l309_309612


namespace grooming_time_l309_309435

theorem grooming_time
  (poodle_time : ℕ := 30)
  (terrier_time : ℕ := poodle_time / 2)
  (num_poodles : ℕ := 3)
  (num_terriers : ℕ := 8)
  (num_employees : ℕ := 4) :
  (num_poodles * poodle_time + num_terriers * terrier_time) / num_employees = 52.5 :=
by
  sorry

end grooming_time_l309_309435


namespace concurrency_of_lines_l309_309000

open Set

theorem concurrency_of_lines {O₁ O₂ O₃ : Point} {A B C D P Q : Point}
  (h₁ : intersect (circle O₁) (circle O₂) = {A, B})
  (h₂ : parallel (line_through_points O₁ O₂) (line_through_points A C))
  (h₃ : parallel (line_through_points O₁ O₂) (line_through_points A D))
  (h₄ : intersect (line_through_points A C) (circle O₁) = {A, C})
  (h₅ : intersect (line_through_points A D) (circle O₂) = {A, D})
  (h₆ : diameter_of_circle O₃ C D)
  (h₇ : intersect (circle O₃) (circle O₁) = {P})
  (h₈ : intersect (circle O₃) (circle O₂) = {Q}) :
  concurrency (line_through_points C P) (line_through_points D Q) (line_through_points A B) :=
sorry

end concurrency_of_lines_l309_309000


namespace transformation_count_l309_309585

theorem transformation_count (ell : Type) (pattern : ell → Prop) 
  (T1 T2 T3 T4 T5 : ell → ell) :
  (∀ p, ¬(T1 p = p) → ¬ (pattern (T1 p) ↔ pattern p)) →  -- Condition for $90^\circ$ rotation
  (∃ p, ¬(T2 p = p) ∧ (pattern (T2 p) ↔ pattern p)) ∧  -- Condition for $180^\circ$ rotation
  (∃ p, (pattern (T3 p) ↔ pattern p)) ∧                 -- Condition for translation parallel to line $\ell$
  (∃ p, ¬ (pattern (T4 p) ↔ pattern p)) ∧               -- Condition for translation perpendicular to line $\ell$
  (∃ p, ¬(T5 p = p) ∧ (pattern (T5 p) ↔ pattern p))     -- Condition for reflection across line $\ell$
→ 3 = (if (∃ p, ¬(T1 p = p) → ¬ (pattern (T1 p) ↔ pattern p)) then 1 else 0) + 
      (if (∃ p, ¬(T2 p = p) ∧ (pattern (T2 p) ↔ pattern p)) then 1 else 0) +
      (if (∃ p, (pattern (T3 p) ↔ pattern p)) then 1 else 0) +
      (if (∃ p, ¬ (pattern (T4 p) ↔ pattern p)) then 0 else 1) +
      (if (∃ p, ¬(T5 p = p) ∧ (pattern (T5 p) ↔ pattern p)) then 1 else 0) :=
by sorry

end transformation_count_l309_309585


namespace sequence_contains_all_except_one_l309_309164

noncomputable def sequence (a1 : ℕ) : ℕ → ℕ
| 1 := a1
| (n+1) := Nat.find (λ k, k > 0 ∧ ¬ Nat.coprime k (sequence n) ∧ ¬ (k ∈ (Finset.range n).map (λ i, sequence (i+1))))

theorem sequence_contains_all_except_one (a1 : ℕ) (h : a1 ≥ 2) :
  ∀ (m : ℕ), m > 1 → ∃ n, sequence a1 n = m :=
sorry

end sequence_contains_all_except_one_l309_309164


namespace lowest_price_per_component_l309_309411

def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 6
def fixed_monthly_costs : ℝ := 16500
def components_per_month : ℕ := 150

theorem lowest_price_per_component (price_per_component : ℝ) :
  let total_cost_per_component := production_cost_per_component + shipping_cost_per_component
  let total_production_and_shipping_cost := total_cost_per_component * components_per_month
  let total_cost := total_production_and_shipping_cost + fixed_monthly_costs
  price_per_component = total_cost / components_per_month → price_per_component = 196 :=
by
  sorry

end lowest_price_per_component_l309_309411


namespace factorization_correct_l309_309506

noncomputable def p (x : ℝ) : ℝ := 32 * x^4 - 48 * x^7 + 16 * x^2

theorem factorization_correct (x : ℝ) : p(x) = 16 * x^2 * (2 * x^2 - 3 * x^5 + 1) := by
  sorry

end factorization_correct_l309_309506


namespace exist_tangent_circles_l309_309324

noncomputable def locus_centers_of_tangent_circles (a b : ℝ) : Prop :=
  40 * a ^ 2 + 49 * b ^ 2 - 48 * a - 64 = 0

theorem exist_tangent_circles (a b : ℝ) :
  (∀ r ≥ 0, ∃ C, (C.center = ⟨a, b⟩ ∧ C.radius = r ∧ externally_tangent C C1 ∧ internally_tangent C C3)) → 
  locus_centers_of_tangent_circles a b := 
sorry

end exist_tangent_circles_l309_309324


namespace angle_FCA_is_correct_l309_309620

noncomputable def ellipse : Type := sorry

variables {F C A B : Point}
variable {Γ : ellipse}
variable {l : Line}
variable (E : ∀ (x y : ℝ), (x^2 / 2019 + y^2 / 2018) = 1 → x y ∈ Γ)
variable (Fleft: Focus E Γ F)
variable (l_intersect : ∀ (p1 p2 : Point), (Line l intersects Focus E Γ p1) → (Line l intersects Directrix E Γ p2) → p1 p2 intersect l)
variables (angle_FAB : ∀ (α : ℝ), α = 40)
variables (angle_FBA : ∀ (β : ℝ), β = 10)

theorem angle_FCA_is_correct : ∃ (θ : ℝ), θ = 15 := 
by 
  sorry

end angle_FCA_is_correct_l309_309620


namespace parallel_MN_PQ_l309_309655

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309655


namespace cot_alpha_value_l309_309173

theorem cot_alpha_value 
  (α : Real) 
  (h1 : ∃ θ : Real, cos θ = - (Real.sqrt 3) / 2 ∧ sin θ = 1 / 2 ∧ θ = α) : 
  Real.cot α = - Real.sqrt 3 := 
  sorry

end cot_alpha_value_l309_309173


namespace cellos_in_store_l309_309841

theorem cellos_in_store (C : ℕ) (violins : ℕ) (matching_pairs : ℕ) (prob : ℚ) 
  (h1 : violins = 600) 
  (h2 : matching_pairs = 70) 
  (h3 : prob = 0.00014583333333333335) : 
  70 / (C * 600) = 0.00014583333333333335 -> C = 800 :=
by
  intro h4
  sorry

end cellos_in_store_l309_309841


namespace water_consumed_l309_309446

theorem water_consumed (traveler_water : ℕ) (camel_multiplier : ℕ) (ounces_in_gallon : ℕ) (total_water : ℕ)
  (h_traveler : traveler_water = 32)
  (h_camel : camel_multiplier = 7)
  (h_ounces_in_gallon : ounces_in_gallon = 128)
  (h_total : total_water = traveler_water + camel_multiplier * traveler_water) :
  total_water / ounces_in_gallon = 2 :=
by
  sorry

end water_consumed_l309_309446


namespace no_increase_in_probability_l309_309034

def hunter_two_dogs_probability (p : ℝ) : ℝ :=
  let both_correct := p * p
  let one_correct := 2 * p * (1 - p) / 2
  both_correct + one_correct

theorem no_increase_in_probability (p : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ 1) :
  hunter_two_dogs_probability p = p :=
by
  sorry

end no_increase_in_probability_l309_309034


namespace part_I_part_II_l309_309282

section problem_1

def f (x : ℝ) (a : ℝ) := |x - 3| - |x + a|

theorem part_I (x : ℝ) (hx : f x 2 < 1) : 0 < x :=
by
  sorry

theorem part_II (a : ℝ) (h : ∀ (x : ℝ), f x a ≤ 2 * a) : 3 ≤ a :=
by
  sorry

end problem_1

end part_I_part_II_l309_309282


namespace correlation_proof_l309_309976

variable (x y z : ℝ)
variable (k : ℝ)

-- Define the conditions
def condition1 : Prop := ∀ x y, y = -0.1 * x + 1
def condition2 : Prop := ∀ y z, (y = k * z ∧ k > 0) → (y positively correlated with z)

-- Define the conclusion to be proven
def conclusion : Prop :=
  (z negatively correlated with y) ∧ (x negatively correlated with z)

-- The final theorem statement with given conditions and conclusion
theorem correlation_proof :
  (condition1) ∧ (condition2) → (conclusion) :=
by
  sorry   -- Proof to be filled in

end correlation_proof_l309_309976


namespace max_two_factors_Jk_l309_309921

def J (k : ℕ) : ℕ := 10^(k+2) + 648

def M (k : ℕ) : ℕ := (J(k)).factors.count 2

theorem max_two_factors_Jk {k : ℕ} (h : 0 < k) : 
  ∀ n : ℕ, M(k) ≤ 4 :=
by
  intro n
  sorry

end max_two_factors_Jk_l309_309921


namespace ball_arrangement_problem_l309_309469

-- Defining the problem statement and conditions
theorem ball_arrangement_problem : 
  (∃ (A : ℕ), 
    (∀ (b : Fin 6 → ℕ), 
      (b 0 = 1 ∨ b 1 = 1) ∧ (b 0 = 2 ∨ b 1 = 2) ∧ -- 1 adjacent to 2
      b 4 ≠ 5 ∧ b 4 ≠ 6 ∧                 -- 5 not adjacent to 6 condition
      b 5 ≠ 5 ∧ b 5 ≠ 6     -- Add all other necessary conditions for arrangement
    ) →
    A = 144)
:= sorry

end ball_arrangement_problem_l309_309469


namespace min_value_magnitude_l309_309234

variables (t : ℝ) (x y : ℝ)
def OA := (0, 4)
def OB := (0, 2)
def OC := (x, y)
def C_condition (OC : ℝ × ℝ) : Prop :=
  let OA := (0, 4)
  let OB := (0, 2)
  (2 * OC.1 - OA.1) * (OC.1 - OB.1) + (2 * OC.2 - OA.2) * (OC.2 - OB.2) = 0

noncomputable def magnitude (t : ℝ) (OC : ℝ × ℝ) : ℝ :=
  let OA := (0, 4)
  let OB := (0, 2)
  let diff := (OC.1, OC.2 - (1/4)*t*OA.2 - (1/2)*(Real.log (-t) - 1)*OB.2)
  Real.abs diff.2

theorem min_value_magnitude : ∀ t < 0, ∀ OC, C_condition OC → magnitude t OC ≥ 4 := by
  sorry

end min_value_magnitude_l309_309234


namespace sides_of_second_polygon_l309_309370

theorem sides_of_second_polygon (s : ℝ) (n : ℕ) 
  (perimeter1_is_perimeter2 : 38 * (2 * s) = n * s) : 
  n = 76 := by
  sorry

end sides_of_second_polygon_l309_309370


namespace selection_probability_l309_309605

-- Let's define the conditions
def num_products : ℕ := 2003
def num_selected : ℕ := 50
def removed_products : ℕ := 3
def remaining_products : ℕ := num_products - removed_products

theorem selection_probability :
  ∀ (p : ℕ), p < num_products →
  let probability_of_selection := (num_selected : ℝ) / (num_products : ℝ) in
  (p < removed_products → (0 : ℝ) = probability_of_selection) ∧
  (p ≥ removed_products → (1 : ℝ) = probability_of_selection) :=
by
  sorry

end selection_probability_l309_309605


namespace bent_strips_odd_l309_309772

theorem bent_strips_odd (cube_dim : ℕ) (strip_dim : ℕ) 
  (H_cube : cube_dim = 9) 
  (H_strip : strip_dim = 2) 
  : ∃ n : ℕ, is_odd n ∧ n = number_of_bent_strips cube_dim strip_dim := 
sorry

end bent_strips_odd_l309_309772


namespace single_zero_point_range_l309_309598

def f (x m : ℝ) := exp x * (2 * x - 1) - m * x + m

theorem single_zero_point_range (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ (m = 1 ∨ m = 4 * exp (3 / 2) ∨ m ≤ 0) :=
by
  sorry

end single_zero_point_range_l309_309598


namespace candy_distribution_l309_309644

theorem candy_distribution (n : ℕ) (boys girls : List ℕ) (N : ℕ) 
  (h_boy_count : boys.length = n) 
  (h_girl_count : girls.length = n)
  (h_candy_condition : ∀ (X : ℕ) (m : ℕ), 
    (∃ l r : List ℕ, 
    (l ++ [X] ++ r = boys ++ girls 
    ∧ l.length + r.length + 1 = boys.length + girls.length) 
    ∧ l.length = m ∧ r.length = n - m - 1)
    → X receives m candies) :
  N ≤ (1 / 3 : ℚ) * n * (n^2 - 1) := 
sorry

end candy_distribution_l309_309644


namespace sum_arith_geo_seq_l309_309958

theorem sum_arith_geo_seq (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (i + 3) * (1 / 2 ^ (i + 1))) = 4 - (n + 4) / 2^n := 
by
  sorry

end sum_arith_geo_seq_l309_309958


namespace expected_value_of_win_is_51_l309_309423

noncomputable def expected_value_of_win : ℝ :=
  (∑ n in (finset.range 8).map (λ x, x + 1), (1/8) * 2 * (n : ℝ)^2)

theorem expected_value_of_win_is_51 : expected_value_of_win = 51 := 
by 
  sorry

end expected_value_of_win_is_51_l309_309423


namespace infinite_perpendicular_lines_l309_309950

-- Definitions for the problem setup
variables {α : Type} [Plane α] {a : Line} {P : α}
-- Assume line 'a' intersects plane 'α' at point P and is not perpendicular
axiom intersects (a_intersects_α : ∃ P: α, P ∈ a ∧ P ∈ α)
axiom not_perpendicular (a_not_perpendicular : ¬(a ⊥ α))

-- Goal to prove: there are infinitely many lines in plane "α" that are perpendicular to line "a"
theorem infinite_perpendicular_lines (a_intersects_α : ∃ P : α, P ∈ a ∧ P ∈ α) (a_not_perpendicular : ¬ (a ⊥ α)) :
  ∃ (f : ℕ → Line), ∀ n, f n ⊂ α ∧ (f n ⊥ a) := 
sorry

end infinite_perpendicular_lines_l309_309950


namespace tan_sum_of_tan_roots_l309_309593

theorem tan_sum_of_tan_roots : 
  ∀ α β : ℝ, 
  (∀ x : ℝ, x^2 - 3 * real.sqrt 3 * x + 4 = 0 → (x = real.tan α ∨ x = real.tan β)) 
  → real.tan (α + β) = -real.sqrt 3 :=
sorry

end tan_sum_of_tan_roots_l309_309593


namespace train_crossing_time_l309_309448

theorem train_crossing_time
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_train_kmh : ℕ)
  (speed_train_ms : ℝ := speed_train_kmh * 5 / 18)
  (total_distance : ℕ := length_train + length_bridge)
  (time : ℝ := total_distance / speed_train_ms) :
  length_train = 240 ∧ length_bridge = 150 ∧ speed_train_kmh = 70.2 → time = 20 :=
begin
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  unfold total_distance,
  unfold speed_train_ms,
  unfold time,
  norm_num,
end

end train_crossing_time_l309_309448


namespace circumcircle_diameter_l309_309999

theorem circumcircle_diameter (a B S : ℝ) (h1 : a = 1)
  (h2 : B = real.to_radians 45) (h3 : S = 2) :
  (2 / (real.sin B) = 5 * real.sqrt 2) :=
by
  sorry

end circumcircle_diameter_l309_309999


namespace arithmetic_sequence_range_of_lambda_l309_309162

-- Define the sequence {a_n}
def sequence_a : ℕ → ℝ := sorry

-- Define the sum of the first n terms as Sn
def S (n : ℕ) : ℝ := 2 * sequence_a n - 2^(n + 1)

-- Condition: Sn definition
axiom S_def (n : ℕ) : S n = 2 * sequence_a n - 2^(n + 1)

-- Question (1): Prove that {a_n / 2^n} is an arithmetic sequence.
theorem arithmetic_sequence (n m : ℕ) (h : m = n + 1) : (sequence_a m / 2^m) - (sequence_a n / 2^n) = 1 := sorry

-- Condition: Inequality for all n ∈ ℕ*
axiom inequality (n : ℕ) (hn : n > 0) (λ : ℝ) : 2 * n^2 - n - 3 < (5 - λ) * sequence_a n

-- Question (2): Range of λ
theorem range_of_lambda (λ : ℝ) : λ < 37/8 := sorry

end arithmetic_sequence_range_of_lambda_l309_309162


namespace number_of_unlocked_cells_l309_309437

-- Establish the conditions from the problem description.
def total_cells : ℕ := 2004

-- Helper function to determine if a number is a perfect square.
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- Counting the number of perfect squares in the range from 1 to total_cells.
def perfect_squares_up_to (n : ℕ) : ℕ :=
  (Nat.sqrt n)

-- The theorem that needs to be proved.
theorem number_of_unlocked_cells : perfect_squares_up_to total_cells = 44 :=
by
  sorry

end number_of_unlocked_cells_l309_309437


namespace complete_consoles_production_rate_l309_309044

-- Define the production rates of each chip
def production_rate_A := 467
def production_rate_B := 413
def production_rate_C := 532
def production_rate_D := 356
def production_rate_E := 494

-- Define the maximum number of consoles that can be produced per day
def max_complete_consoles (A B C D E : ℕ) := min (min (min (min A B) C) D) E

-- Statement
theorem complete_consoles_production_rate :
  max_complete_consoles production_rate_A production_rate_B production_rate_C production_rate_D production_rate_E = 356 :=
by
  sorry

end complete_consoles_production_rate_l309_309044


namespace max_pens_with_10_dollars_l309_309713

theorem max_pens_with_10_dollars : 
  ∀ (cost_individual : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ) (budget : ℕ),
    cost_individual = 1 → 
    cost_pack4 = 3 →
    cost_pack7 = 4 →
    budget = 10 →
    (max_pens cost_individual cost_pack4 cost_pack7 budget = 16) 
    :=
begin
  -- Define the number of pens you can buy with certain options
  sorry
end

noncomputable def max_pens (cost_individual cost_pack4 cost_pack7 budget : ℕ) : ℕ := 
  (budget / 4) * 7 + min ((budget % 4) / 1) * 1

end max_pens_with_10_dollars_l309_309713


namespace island_population_percentage_l309_309459

theorem island_population_percentage :
  -- Defining conditions
  (∀ a b : ℕ, (a + b ≠ 0) → (a.toRat / (a + b).toRat = 65 / 100) →
   ∀ b c : ℕ, (b + c ≠ 0) → (b.toRat / (b + c).toRat = 70 / 100) →
   ∀ c d : ℕ, (c + d ≠ 0) → (c.toRat / (c + d).toRat = 10 / 100) →
  
  -- Correct answer based on conditions
  ∃ a b c d : ℕ, 
    let total := a + b + c + d in 
    total ≠ 0 ∧ 
    (d.toRat / total.toRat = 54 / 100)) := 
sorry

end island_population_percentage_l309_309459


namespace prove_x_value_l309_309088

theorem prove_x_value :
  let S₁ := ∑' n : ℕ, (1/3)^n,
      S₂ := ∑' n : ℕ, (-1/3)^n,
      S₃ := 1 / (1 - 1/x)
  in S₁ * S₂ = S₃ → x = 9 := by
  let S₁ := ∑' n : ℕ, (1/3)^n
  have h₁ : S₁ = 3/2 :=
    sorry
  let S₂ := ∑' n : ℕ, (-1/3)^n
  have h₂ : S₂ = 3/4 :=
    sorry
  let S₃ := 1 / (1 - 1/x)
  have h₃ : S₁ * S₂ = S₃ :=
    sorry
  exact (by
    sorry : S₁ * S₂ = 9/8 → x = 9)
    {subst h₁, subst h₂, subst h₃, 
     sorry}

end prove_x_value_l309_309088


namespace find_m_for_binomial_square_l309_309387

theorem find_m_for_binomial_square (m : ℝ) : (∃ a : ℝ, (λ x : ℝ, x^2 - 20 * x + m) = (λ x : ℝ, (x + a)^2)) → m = 100 :=
by
  intro h
  sorry

end find_m_for_binomial_square_l309_309387


namespace tan_double_angle_identity_l309_309533

theorem tan_double_angle_identity (theta : ℝ) (h1 : 0 < theta ∧ theta < Real.pi / 2)
  (h2 : Real.sin theta - Real.cos theta = Real.sqrt 5 / 5) :
  Real.tan (2 * theta) = -(4 / 3) := 
by
  sorry

end tan_double_angle_identity_l309_309533


namespace compare_combined_sums_l309_309084

def numeral1 := 7524258
def numeral2 := 523625072

def place_value_2_numeral1 := 200000 + 20
def place_value_5_numeral1 := 50000 + 500
def combined_sum_numeral1 := place_value_2_numeral1 + place_value_5_numeral1

def place_value_2_numeral2 := 200000000 + 20
def place_value_5_numeral2 := 500000 + 50
def combined_sum_numeral2 := place_value_2_numeral2 + place_value_5_numeral2

def difference := combined_sum_numeral2 - combined_sum_numeral1

theorem compare_combined_sums :
  difference = 200249550 := by
  sorry

end compare_combined_sums_l309_309084


namespace sum_of_1001_terms_l309_309858

def sequence (b : ℕ → ℤ) : Prop :=
  ∀ n ≥ 3, b n = b (n - 1) - b (n - 2) + 2

def sum_first_n_terms (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, b i

theorem sum_of_1001_terms (b : ℕ → ℤ)  
  (h1 : sequence b) 
  (h2 : sum_first_n_terms b 801 = 997) 
  (h3 : sum_first_n_terms b 1000 = 1003) :
  sum_first_n_terms b 1001 = 1003 :=
by 
  sorry

end sum_of_1001_terms_l309_309858


namespace n_is_power_of_2_l309_309191

-- Definition and conditions in Lean 4
variables {n : ℕ}
variables {a b : Fin n → ℚ}

/-- Given two distinct n-element subsets a and b of Q,
    such that the pairwise sums of elements are equal,
    and for any such sum, the number of pairs (i, j) producing that sum in a
    equals the number of pairs (i, j) producing that sum in b,
    prove that n is a power of 2. -/
theorem n_is_power_of_2 (h_distinct : a ≠ b)
  (h_sums_eq : {r | ∃ (i j : Fin n), i < j ∧ r = a i + a j} = {r | ∃ (i j : Fin n), i < j ∧ r = b i + b j})
  (h_counts_eq : ∀ r, (∃ i j, i < j ∧ r = a i + a j) → (finset.univ.filter (λ ij, r = a ij.1 + a ij.2)).card = (finset.univ.filter (λ ij, r = b ij.1 + b ij.2)).card) :
  ∃ k : ℕ, n = 2^k :=
sorry

end n_is_power_of_2_l309_309191


namespace rope_length_l309_309857

-- Definitions and assumptions directly derived from conditions
variable (total_length : ℕ)
variable (part_length : ℕ)
variable (sub_part_length : ℕ)

-- Conditions
def condition1 : Prop := total_length / 4 = part_length
def condition2 : Prop := (part_length / 2) * 2 = part_length
def condition3 : Prop := part_length / 2 = sub_part_length
def condition4 : Prop := sub_part_length = 25

-- Proof problem statement
theorem rope_length (h1 : condition1 total_length part_length)
                    (h2 : condition2 part_length)
                    (h3 : condition3 part_length sub_part_length)
                    (h4 : condition4 sub_part_length) :
                    total_length = 100 := 
sorry

end rope_length_l309_309857


namespace arithmetic_geometric_sequence_a1_l309_309545

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ)
  (h1 : a 1 + a 6 = 11)
  (h2 : a 3 * a 4 = 32 / 9) :
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 :=
sorry

end arithmetic_geometric_sequence_a1_l309_309545


namespace ellipse_equation_proof_l309_309942

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := sqrt (a^2 - 1)
noncomputable def c : ℝ := 1
def e : ℝ := c / a

def ellipse_equation (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def M : ℝ × ℝ := (-3, 0)
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)

def line_l (k m x y : ℝ) : Prop := y = k * x + m

def A (k m : ℝ) : ℝ × ℝ := (2, 2 * k + m)
def B (k m : ℝ) : ℝ × ℝ := (-2, -2 * k + m)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_equation_proof :
  e = 1 / 2 ∧
  ellipse_equation = λ x y, (x^2 / 4) + (y^2 / 3) = 1 ∧
  ∀ k m, 
    dot_product (F2.1 - A k m.1, F2.2 - A k m.2) (F2.1 - B k m.1, F2.2 - B k m.2) = 0 →
    ∃ x y, ellipse_equation x y ∧ line_l k m x y ∧
      (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 12 = 0 ∧ (64 * k^2 * m^2 - 4 * (3 + 4 * k^2) * (4 * m^2 - 12)) = 0 :=
sorry

end ellipse_equation_proof_l309_309942


namespace sum_first_five_terms_l309_309946

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 1, ∀ n, a (n + 1) = a n * q

theorem sum_first_five_terms (h₁ : is_geometric_sequence a) 
  (h₂ : a 1 > 0) 
  (h₃ : a 1 * a 7 = 64) 
  (h₄ : a 3 + a 5 = 20) : 
  a 1 * (1 - (2 : ℝ) ^ 5) / (1 - 2) = 31 := 
by
  sorry

end sum_first_five_terms_l309_309946


namespace infinitely_many_n_l309_309723

def num_odd_prime_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ p, p.prime ∧ ¬ even p) (n.factors : Finset ℕ)).card

def a_n (n : ℕ) : ℕ := num_odd_prime_divisors (n * (n + 3))

theorem infinitely_many_n (N : ℕ → Prop) : 
  ∃ᶠ n in at_top, a_n n % 3 = 0 := 
sorry

end infinitely_many_n_l309_309723


namespace condition_sufficient_not_necessary_l309_309329

theorem condition_sufficient_not_necessary (x : ℝ) :
  (0 < x ∧ x < 2) → (x < 2) ∧ ¬((x < 2) → (0 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l309_309329


namespace positive_area_triangles_in_6x6_grid_l309_309197

theorem positive_area_triangles_in_6x6_grid : 
  let points := (fin 6) × (fin 6)
  -- function to check if three points are collinear
  let collinear (p1 p2 p3 : points) : Prop :=
    (p1.1.to_nat * (p2.2.to_nat - p3.2.to_nat) + p2.1.to_nat * (p3.2.to_nat - p1.2.to_nat) + p3.1.to_nat * (p1.2.to_nat - p2.2.to_nat) = 0)
  -- number of subsets of three non-collinear points
  let non_collinear_triples : finset (points × points × points) := 
    (finset.univ : finset points).to_list.combinations 3
    |>.filter (λ l, ¬ collinear l.head (l.tail.head) l.tail.tail.head)
   let total_non_collinear_triples := non_collinear_triples.to_finset.card
  in total_non_collinear_triples = 6744 :=
sorry

end positive_area_triangles_in_6x6_grid_l309_309197


namespace midpoint_parallel_l309_309676

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309676


namespace fred_gave_balloons_to_sandy_l309_309525

-- Define the number of balloons Fred originally had
def original_balloons : ℕ := 709

-- Define the number of balloons Fred has now
def current_balloons : ℕ := 488

-- Define the number of balloons Fred gave to Sandy
def balloons_given := original_balloons - current_balloons

-- Theorem: The number of balloons given to Sandy is 221
theorem fred_gave_balloons_to_sandy : balloons_given = 221 :=
by
  sorry

end fred_gave_balloons_to_sandy_l309_309525


namespace max_sum_of_cubes_l309_309700

theorem max_sum_of_cubes (p q r s t : ℝ) (h : p^2 + q^2 + r^2 + s^2 + t^2 = 5) :
  p^3 + q^3 + r^3 + s^3 + t^3 ≤ 5 * sqrt 5 :=
sorry

end max_sum_of_cubes_l309_309700


namespace length_exceeds_breadth_by_24_l309_309778

noncomputable def breadth (b : ℝ) : ℝ :=
  let x := 62 - b in
  let P := 4 * b + 2 * x in
  let cost := 26.5 * P in
  if cost = 5300 then x else 0

theorem length_exceeds_breadth_by_24 (b : ℝ) (h1 : 26.5 * (4 * b + 2 * (62 - b)) = 5300) :
  62 - b = 24 := by
  sorry

end length_exceeds_breadth_by_24_l309_309778


namespace sum_c_d_l309_309930

theorem sum_c_d (c d : ℕ) (h1 : 0 < c ∧ 0 < d) 
  (h2 : ∏ i in finset.range (d - c), log (i+c) (i+c+1) = 3)
  (h3 : (d - 1) - c + 1 = 930) : c + d = 1010 := 
by
  sorry

end sum_c_d_l309_309930


namespace set_values_of_a_l309_309214

theorem set_values_of_a (a : ℝ) : 
  (∀ x, (ax^2 + 2*x + 4*a = 0) → a = 0 ∨ a = 1/2 ∨ a = -1/2) → 
  (∃ A : set ℝ, (∀ x, x ∈ A ↔ ax^2 + 2*x + 4*a = 0) ∧ (#A = 1) → (a = 0 ∨ a = 1/2 ∨ a = -1/2)) :=
sorry

end set_values_of_a_l309_309214


namespace f_neg_pi_over_3_f_neg_pi_over_4_f_pi_over_4_f_pi_over_3_f_sign_neg_f_zero_f_sign_pos_l309_309182

noncomputable def f (x : ℝ) : ℝ := tan x - sin x

theorem f_neg_pi_over_3 : f (-π / 3) < 0 :=
sorry

theorem f_neg_pi_over_4 : f (-π / 4) < 0 :=
sorry

theorem f_pi_over_4 : f (π / 4) > 0 :=
sorry

theorem f_pi_over_3 : f (π / 3) > 0 :=
sorry

theorem f_sign_neg (x : ℝ) (h : x ∈ Ioo (-π / 2) 0) : f x < 0 :=
sorry

theorem f_zero : f 0 = 0 :=
sorry

theorem f_sign_pos (x : ℝ) (h : x ∈ Ioo 0 (π / 2)) : f x > 0 :=
sorry

end f_neg_pi_over_3_f_neg_pi_over_4_f_pi_over_4_f_pi_over_3_f_sign_neg_f_zero_f_sign_pos_l309_309182


namespace find_x_l309_309964

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then -3^x else 1 - x^2

theorem find_x (x : ℝ) : f(x) = -3 ↔ x = 1 ∨ x = -2 := by
  unfold f
  split_ifs
  . sorry  -- Case when x > 0
  . sorry  -- Case when x ≤ 0

end find_x_l309_309964


namespace opposite_of_two_l309_309787

theorem opposite_of_two : ∃ x : ℤ, 2 + x = 0 ∧ x = -2 :=
by
  exists  -2
  split
  . simp
  . refl

end opposite_of_two_l309_309787


namespace trivia_team_members_l309_309869

theorem trivia_team_members (x : ℕ) (h : 3 * (x - 6) = 27) : x = 15 := 
by
  sorry

end trivia_team_members_l309_309869


namespace find_angle_between_vectors_l309_309912

open Real

def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  real.arccos ((u.1 * v.1 + u.2 * v.2) / (sqrt (u.1^2 + u.2^2) * sqrt (v.1^2 + v.2^2))) * 180 / π

theorem find_angle_between_vectors :
  angle_between_vectors (4, -2) (5, 9) = 85.916 := 
by
  sorry

end find_angle_between_vectors_l309_309912


namespace speed_of_man_walking_l309_309837

theorem speed_of_man_walking {L t : ℝ} {Vt_km_hr : ℝ} (h₁ : L = 550) (h₂ : t = 32.997) (h₃ : Vt_km_hr = 63) :
  let Vt := Vt_km_hr * (1000 / 3600)
  let Vr := L / t
  Vt - Vr ≈ 0.831 :=
by 
  let Vt := Vt_km_hr * (1000 / 3600)
  let Vr := L / t
  sorry

end speed_of_man_walking_l309_309837


namespace sum_of_coeffs_eq_one_l309_309588

theorem sum_of_coeffs_eq_one (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) (x : ℝ) :
  (1 - 2 * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + 
                    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1 :=
  sorry

end sum_of_coeffs_eq_one_l309_309588


namespace sin_graph_shift_l309_309574

def graph_transformation_proof : Prop :=
  ∀ (x : ℝ), 
    let y1 := sin (x + (π / 5)) in
    let y2 := sin (x + (2 * π / 5)) in
    (∀ (x' : ℝ), y2 (x' - (π / 5)) = y1 x')

theorem sin_graph_shift : graph_transformation_proof :=
by
  sorry

end sin_graph_shift_l309_309574


namespace shaded_region_area_l309_309005

theorem shaded_region_area (ABCD: Type) (D B: Type) (AD CD: ℝ) 
  (h1: (AD = 5)) (h2: (CD = 12)):
  let radiusD := Real.sqrt (AD^2 + CD^2)
  let quarter_circle_area := Real.pi * radiusD^2 / 4
  let radiusC := CD / 2
  let semicircle_area := Real.pi * radiusC^2 / 2
  quarter_circle_area - semicircle_area = 97 * Real.pi / 4 :=
by sorry

end shaded_region_area_l309_309005


namespace compare_a_b_c_l309_309531

noncomputable def a := 1 / 2023
noncomputable def b := Real.log (2024 / 2023)
noncomputable def c := Real.logBase 5 (2024 / 2023)

theorem compare_a_b_c : c < b ∧ b < a := by
  sorry

end compare_a_b_c_l309_309531


namespace eccentricity_range_l309_309943

-- Ellipse properties and conditions
variables (a b : ℝ) (ha : a > b > 0)
variables (α : ℝ) (hα : π / 6 ≤ α ∧ α ≤ π / 4)

noncomputable def e := 1 / (Real.sin α + Real.cos α)

theorem eccentricity_range :
  ∀ (a b : ℝ) (ha : a > b > 0) (α : ℝ) (hα : π / 6 ≤ α ∧ α ≤ π / 4),
  ∃ (e : ℝ), (√2 / 2 ≤ e ∧ e ≤ √3 - 1) :=
by
  intros a b ha α hα
  use 1 / (Real.sin α + Real.cos α)
  sorry

end eccentricity_range_l309_309943


namespace count_squares_parallel_to_r_or_s_count_isosceles_right_triangles_parallel_to_r_or_s_l309_309236

-- Definitions
def square_grid : finset (ℕ × ℕ) := 
  {(i, j) | i ∈ range 4 ∧ j ∈ range 4 }.to_finset

def is_parallel_to_lines (v1 v2 : (ℕ × ℕ)) (r s : (ℕ × ℕ) → bool) : Prop := 
  r v1 || s v2

-- Theorem statements
theorem count_squares_parallel_to_r_or_s :
  (∀ r s, ∃ count, count = 6 ∧
    ∀ (sq : finset (ℕ × ℕ)), sq ⊆ square_grid → 
    (∃ v1 v2 v3 v4 ∈ sq, 
      (¬ is_parallel_to_lines v1 v2 r s) ∧ 
      (¬ is_parallel_to_lines v2 v3 r s) ∧ 
      (¬ is_parallel_to_lines v3 v4 r s) ∧ 
      (¬ is_parallel_to_lines v4 v1 r s))) := sorry

theorem count_isosceles_right_triangles_parallel_to_r_or_s :
  (∀ r s, ∃ count, count = 16 ∧ 
    ∀ (tri : finset (ℕ × ℕ)), tri ⊆ square_grid →
    (∃ v1 v2 v3 ∈ tri, 
      (¬ is_parallel_to_lines v1 v2 r s) ∧ 
      (¬ is_parallel_to_lines v2 v3 r s) ∧ 
      (∃ hyp, hyp = (v1 - v3) /\ |hyp| = |v1 - v2| * sqrt2))) := sorry

end count_squares_parallel_to_r_or_s_count_isosceles_right_triangles_parallel_to_r_or_s_l309_309236


namespace initial_money_in_wallet_l309_309805

theorem initial_money_in_wallet (x : ℝ) 
  (h1 : x = 78 + 16) : 
  x = 94 :=
by
  sorry

end initial_money_in_wallet_l309_309805


namespace not_always_ac_gt_bc_l309_309980

variable (a b c : ℝ)

theorem not_always_ac_gt_bc (ha : a > 0) (hb : b > 0) (hab : a > b) (hcn : c ≠ 0) : 
  ∃ c : ℝ, a > b ∧ ¬ (a * c > b * c) :=
by
  use (some_negative_value) -- Replace with an appropriate negative value to satisfy the condition
  sorry

end not_always_ac_gt_bc_l309_309980


namespace ratio_of_areas_of_triangles_l309_309555

variables {a b c : ℝ}
variables {F1 F2 Q P R O : ℝ × ℝ}

-- elliptic condition
def ellipse (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- foci of the ellipse
def is_focus (F1 F2 : ℝ × ℝ) := F1 = (-c, 0) ∧ F2 = (c, 0)

-- point on ellipse such that forms equilateral triangle with origin and foci
def equilateral_triangle (Q F1 O : ℝ × ℝ) := (Q = (-c/2, sqrt 3 / 2 * c)) ∧ (O = (0,0))

-- intersecting points from rays
def intersection_points (Q F1 QO : ℝ × ℝ) := true -- Unspecified but necessary for completeness

-- The given problem transformed into a proof statement in Lean
theorem ratio_of_areas_of_triangles :
  a > b → b > 0 → is_focus F1 F2 → ellipse a b (-c/2) (sqrt 3 / 2 * c) →
  intersecting_points Q F1 QO →
  equilateral_triangle Q F1 O →
  let area_△QF1O := (sqrt 3 / 4) * c^2 in
  let area_△QPR := (3 - sqrt 3) * c^2 / 2 in
  area_△QF1O / area_△QPR = (sqrt 3 + 1) / 8 :=
sorry

end ratio_of_areas_of_triangles_l309_309555


namespace spiral_homothety_center_l309_309160

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C D O Q : V}
variables (parallelogram : parallelogram A B C D)
variables (center_O : is_center O A B C D)
variables (sym_lines_intersect_Q : symmetric_lines_intersect Q A B C D)

theorem spiral_homothety_center :
  is_spiral_homothety_center Q A O O D := 
sorry

end spiral_homothety_center_l309_309160


namespace average_speed_correct_l309_309853

noncomputable def total_distance : ℝ := 300
noncomputable def regular_distance : ℝ := 50
noncomputable def regular_speed : ℝ := 100
noncomputable def muddy_distance : ℝ := 70
noncomputable def muddy_speed : ℝ := 40
noncomputable def hilly_distance : ℝ := 80
noncomputable def hilly_speed : ℝ := 60
noncomputable def dusty_distance : ℝ := 50
noncomputable def dusty_speed : ℝ := 30
noncomputable def gravel_distance : ℝ := 50
noncomputable def gravel_speed : ℝ := 70

noncomputable def time_regular : ℝ := regular_distance / regular_speed
noncomputable def time_muddy : ℝ := muddy_distance / muddy_speed
noncomputable def time_hilly : ℝ := hilly_distance / hilly_speed
noncomputable def time_dusty : ℝ := dusty_distance / dusty_speed
noncomputable def time_gravel : ℝ := gravel_distance / gravel_speed

noncomputable def total_time : ℝ := time_regular + time_muddy + time_hilly + time_dusty + time_gravel
noncomputable def average_speed : ℝ := total_distance / total_time

theorem average_speed_correct : average_speed ≈ 50.306 := by
  sorry

end average_speed_correct_l309_309853


namespace tan_alpha_eq_one_third_l309_309202

theorem tan_alpha_eq_one_third
  (α : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : cos(2 * α) = (2 * real.sqrt 5 / 5) * sin(α + π / 4)) :
  tan α = 1 / 3 := 
by sorry

end tan_alpha_eq_one_third_l309_309202


namespace triangle_area_ratio_l309_309256

theorem triangle_area_ratio 
  (A B C M P D: Type)
  [triangle A B C]
  [point_on_side M A B]
  [point_on_side P A B]
  [parallel M D P C]
  (h1: dist A M = 2 * dist M B)
  (h2: dist A P = 3 * dist P B)
  (MD_intersects_ AC : segment A C)
  : area_ratio (triangle B P D) (triangle A B C) = 1 / 72 := 
sorry

end triangle_area_ratio_l309_309256


namespace tb_tc_eq_ti_b_sq_l309_309004

variable {A B C I_b M T : Point}
variable {ω : Circle}

-- Definitions based on given problem
def is_b_excenter (I_b : Point) (A B C : Point) : Prop := sorry
def is_circumcircle (ω : Circle) (A B C : Point) : Prop := sorry
def is_midpoint_of_arc (M : Point) (B C : Point) (ω : Circle) (A : Point) : Prop := sorry
def intersects_at (M I_b T : Point) (ω : Circle) (A : Point) : Prop := sorry

-- The proof problem statement
theorem tb_tc_eq_ti_b_sq
  (h_b_excenter : is_b_excenter I_b A B C)
  (h_circumcircle : is_circumcircle ω A B C)
  (h_midpoint_arc : is_midpoint_of_arc M B C ω A)
  (h_intersects : intersects_at M I_b T ω A)
  (T_ne_M : T ≠ M) :
  TB * TC = (TI_b)^2 := 
sorry

end tb_tc_eq_ti_b_sq_l309_309004


namespace sufficient_but_not_necessary_condition_l309_309201

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : ∀ x : ℝ, x > a → x > 2 ∧ ¬(x > 2 → x > a)) : a > 2 :=
sorry

end sufficient_but_not_necessary_condition_l309_309201


namespace midpoint_parallel_l309_309679

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309679


namespace correct_conclusions_l309_309221

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
A + B + C = π ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + b^2 = c^2 + 2 * c^2 * (cos (A - B + C))) ∧  
  (a^2 + c^2 = b^2 + 2 * b^2 * (cos (A + B - C))) ∧
  (b^2 + c^2 = a^2 + 2 * a^2 * (cos (C + B - A)))

theorem correct_conclusions 
  (A B C a b c : ℝ) 
  (h : triangle_ABC A B C a b c) 
  (hA_gt_B : A > B) : 
  a > b ∧ sin A > sin B ∧ cos A < cos B :=
by
  sorry

end correct_conclusions_l309_309221


namespace base_729_base8_l309_309094

theorem base_729_base8 (b : ℕ) (X Y : ℕ) (h_distinct : X ≠ Y)
  (h_range : b^3 ≤ 729 ∧ 729 < b^4)
  (h_form : 729 = X * b^3 + Y * b^2 + X * b + Y) : b = 8 :=
sorry

end base_729_base8_l309_309094


namespace frog_hops_ratio_l309_309807

theorem frog_hops_ratio :
  ∀ (F1 F2 F3 : ℕ),
    F1 = 4 * F2 →
    F1 + F2 + F3 = 99 →
    F2 = 18 →
    (F2 : ℚ) / (F3 : ℚ) = 2 :=
by
  intros F1 F2 F3 h1 h2 h3
  -- algebraic manipulations and proof to be filled here
  sorry

end frog_hops_ratio_l309_309807


namespace lowest_possible_price_l309_309393

theorem lowest_possible_price
  (regular_discount_rate : ℚ)
  (sale_discount_rate : ℚ)
  (manufacturer_price : ℚ)
  (H1 : regular_discount_rate = 0.30)
  (H2 : sale_discount_rate = 0.20)
  (H3 : manufacturer_price = 35) :
  (manufacturer_price * (1 - regular_discount_rate) * (1 - sale_discount_rate)) = 19.60 := by
  sorry

end lowest_possible_price_l309_309393


namespace total_votes_750_l309_309877

theorem total_votes_750 : ∃ (y : ℕ), let N := 75
  ∧ 0.55 * y = 55% * y
  ∧ 0.45 * y = 45% * y
  ∧ 0.55 * y - 0.45 * y = 0.10 * y
  ∧ 0.10 * y = N
  → y = 750 := 
sorry

end total_votes_750_l309_309877


namespace opposite_of_two_l309_309786

theorem opposite_of_two : ∃ x : ℤ, 2 + x = 0 ∧ x = -2 :=
by
  exists  -2
  split
  . simp
  . refl

end opposite_of_two_l309_309786


namespace num_triangles_with_longest_side_6_l309_309600

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem num_triangles_with_longest_side_6 : 
  ∃ (count : ℕ), (∃ (sides : finset (ℕ × ℕ × ℕ)), 
  ∀ (a b c ∈ sides), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ max a (max b c) = 6 ∧ (is_triangle a b c)) ∧ 
  count = 4 := 
sorry

end num_triangles_with_longest_side_6_l309_309600


namespace tangent_line_equation_monotonic_decreasing_range_of_a_l309_309963

-- Definition for Part I
def f (x a : ℝ) := Real.exp x * (x^2 - a)

-- Part I: The equation of the tangent line at (0, f(0))
theorem tangent_line_equation (a : ℝ) (h_a : a = 1) : 
  let f := f x a,
      df := fun x => (Real.exp x * (x^2 + 2 * x - a)),
      tangent_line := (x : ℝ) * 1 + (y : ℝ) * 1 + 1 = 0
  in
  f 0 = -1 ∧ df 0 = -1 → tangent_line := 
by
  sorry

-- Part II: The range of values for a such that f(x) is monotonically decreasing on (-3, 0)
theorem monotonic_decreasing_range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo (-3 : ℝ) 0, f x a ≤ 0) ↔ a ≥ 3 :=
by
  sorry

end tangent_line_equation_monotonic_decreasing_range_of_a_l309_309963


namespace work_rate_problem_l309_309839

theorem work_rate_problem
  (A_days : ℕ)
  (B_days : ℕ)
  (combined_days : ℕ)
  (hA : A_days = 45)
  (hB : B_days = 30)
  (hC : combined_days = 72) :
  (combined_days * (1 / A_days + 1 / B_days) = 4) :=
by {
  rw [hA, hB, hC],
  -- Work rates: A = 1/45, B = 1/30
  have A_rate : ℚ := 1 / 45,
  have B_rate : ℚ := 1 / 30,
  -- Combined work rate
  have combined_rate : ℚ := A_rate + B_rate,
  -- Simplify combined work rate
  have simplified_combined_rate : ℚ := 1 / 45 + 1 / 30,
  rw [show 1 / 45 + 1 / 30 = 1 / 18, by norm_num] at simplified_combined_rate,
  -- Calculate work done in 72 days
  have result := combined_days * (simplified_combined_rate),
  have simplified_result : ℚ := 72 * (1 / 18),
  rw [show 72 * (1 / 18) = 4, by norm_num] at simplified_result,
  exact simplified_result,
}

end work_rate_problem_l309_309839


namespace smallest_n_proof_l309_309893

open BigOperators

def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

noncomputable def lhs_sum (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), log3 (1 + 1 / (3 ^ 3^k : ℝ))

noncomputable def rhs_val : ℝ := 2 + log3 (8 / 9)

theorem smallest_n_proof (n : ℕ) (h : lhs_sum n ≥ rhs_val) : n = 0 := 
by sorry

end smallest_n_proof_l309_309893


namespace janous_inequality_l309_309254

variables {a b c d : ℝ}

theorem janous_inequality (h : a^2 + b^2 + c^2 + d^2 = 4) : (a + 2) * (b + 2) ≥ c * d :=
by
  sorry

example : (a = -2) ∧ (b = -2) ∧ (c = 1) ∧ (d = 1) :=
by
  simp

end janous_inequality_l309_309254


namespace solution_correct_l309_309467

noncomputable def probability_three_primes_and_one_eight : ℚ :=
  let primes := {2, 3, 5, 7}
  let die_sides := finset.range 1 (8 + 1) -- {1, 2, 3, 4, 5, 6, 7, 8}
  let num_primes := finset.card primes
  
  let prob_prime := num_primes.to_rat / die_sides.card.to_rat
  let prob_not_prime := (die_sides.card - num_primes).to_rat / die_sides.card.to_rat

  let prob_three_primes := (finset.card (finset.range (6 + 1) : ℚ)
    * (prob_prime ^ 3) * (prob_not_prime ^ 3)).to_rat

  let prob_not_eight := ((die_sides.card - 1).to_rat / die_sides.card.to_rat)^6
  let prob_at_least_one_eight := 1 - prob_not_eight

  prob_three_primes * prob_at_least_one_eight

def proof_target : Prop :=
  probability_three_primes_and_one_eight = (2899900 / 16777216 : ℚ)

theorem solution_correct : proof_target :=
by
  sorry

end solution_correct_l309_309467


namespace length_of_second_train_correct_l309_309371

noncomputable def length_of_second_train 
  (speed_train1_kmh : ℝ) (speed_train2_kmh : ℝ) 
  (time_clearance_sec : ℝ) 
  (length_train1_m : ℝ) : ℝ :=
let speed_train1_ms := speed_train1_kmh * 1000 / 3600 in
let speed_train2_ms := speed_train2_kmh * 1000 / 3600 in
let relative_speed_ms := speed_train1_ms + speed_train2_ms in
let total_distance := relative_speed_ms * time_clearance_sec in
total_distance - length_train1_m

theorem length_of_second_train_correct 
  (speed_train1_kmh : ℝ) (speed_train2_kmh : ℝ) 
  (time_clearance_sec : ℝ) 
  (length_train1_m : ℝ) 
  (h_speed_train1 : speed_train1_kmh = 80) 
  (h_speed_train2 : speed_train2_kmh = 65)
  (h_time_clearance : time_clearance_sec = 7.596633648618456) 
  (h_length_train1 : length_train1_m = 141) : 
  length_of_second_train speed_train1_kmh speed_train2_kmh time_clearance_sec length_train1_m ≈ 165.1224489795918 :=
by 
  rw [h_speed_train1, h_speed_train2, h_time_clearance, h_length_train1]
  dsimp [length_of_second_train]
  norm_num
  sorry

end length_of_second_train_correct_l309_309371


namespace find_a_l309_309956

theorem find_a (a : ℝ) (h : ∀ x y : ℝ, ax + y - 4 = 0 → x + (a + 3/2) * y + 2 = 0 → True) : a = 1/2 :=
sorry

end find_a_l309_309956


namespace impossible_arrangement_l309_309016

-- Definitions for the problem
def within_range (n : ℕ) : Prop := n > 0 ∧ n ≤ 500
def distinct (l : List ℕ) : Prop := l.Nodup

-- The main problem statement
theorem impossible_arrangement :
  ∀ (l : List ℕ),
  l.length = 111 →
  l.All within_range →
  distinct l →
  ¬(∀ (k : ℕ) (h : k < l.length), (l.get ⟨k, h⟩) % 10 = (l.sum - l.get ⟨k, h⟩) % 10) :=
by
  intros l length_cond within_range_cond distinct_cond condition
  sorry

end impossible_arrangement_l309_309016


namespace min_k_sin4_cos4_l309_309125

theorem min_k_sin4_cos4 (k : ℝ) (h : k ≥ 0) : ∀ x : ℝ, k * (sin x)^4 + (cos x)^4 ≥ 0 :=
by
  sorry

end min_k_sin4_cos4_l309_309125


namespace customers_in_other_countries_l309_309407

-- Definitions for conditions
def total_customers : ℕ := 7422
def us_customers : ℕ := 723

-- Statement to prove
theorem customers_in_other_countries : total_customers - us_customers = 6699 :=
by
  sorry

end customers_in_other_countries_l309_309407


namespace intersection_A_B_l309_309007

def interval_A : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def interval_B : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem intersection_A_B :
  interval_A ∩ interval_B = { x | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
sorry

end intersection_A_B_l309_309007


namespace solve_sqrt_eq_l309_309118

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 3) ↔ (x = 2 ∨ x = -2) := 
by sorry

end solve_sqrt_eq_l309_309118


namespace monomial_sum_l309_309603

theorem monomial_sum (m n : ℤ) (h1 : n = 2) (h2 : m + 2 = 1) : m + n = 1 := by
  sorry

end monomial_sum_l309_309603


namespace find_widgets_l309_309406

theorem find_widgets (a b c d e f : ℕ) : 
  (3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255) →
  (3 ^ a * 11 ^ b * 5 ^ c * 7 ^ d * 13 ^ e * 17 ^ f = 351125648000) →
  c = 3 :=
by
  sorry

end find_widgets_l309_309406


namespace side_length_of_square_l309_309750

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309750


namespace circumscribed_circle_radius_square_part_l309_309563

theorem circumscribed_circle_radius_square_part:
  (a b c d : ℕ) (h_a : a = 4) (h_b : b = 5) (h_c : c = 6) (h_d : d = 7) :
  ⌊(radius_circumscribed_circle_sq a b c d : ℝ)⌋ = 15 :=
by
  have radius_circumscribed_circle_sq : ℕ → ℕ → ℕ → ℕ → ℝ := sorry
  sorry

end circumscribed_circle_radius_square_part_l309_309563


namespace mn_parallel_pq_l309_309651

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309651


namespace product_of_roots_cubicEq_l309_309487

noncomputable def cubicEq : Polynomial ℝ := Polynomial.Cubic 1 (-12) 48 28

theorem product_of_roots_cubicEq : cubicEq.roots.prod = -28 := 
sorry

end product_of_roots_cubicEq_l309_309487


namespace color_elements_of_X_l309_309253

variable {X : Type} [LinearOrder X]

theorem color_elements_of_X (X : Type) [LinearOrder X] :
  ∃ (color : X → bool), ∀ (x y : X), x < y → color x = color y → 
  ∃ z : X, x < z ∧ z < y ∧ color z ≠ color x :=
sorry

end color_elements_of_X_l309_309253


namespace cocktail_cost_per_litre_l309_309337

theorem cocktail_cost_per_litre :
  let mixed_fruit_cost := 262.85
  let acai_berry_cost := 3104.35
  let mixed_fruit_volume := 37
  let acai_berry_volume := 24.666666666666668
  let total_cost := mixed_fruit_volume * mixed_fruit_cost + acai_berry_volume * acai_berry_cost
  let total_volume := mixed_fruit_volume + acai_berry_volume
  total_cost / total_volume = 1400 :=
by
  sorry

end cocktail_cost_per_litre_l309_309337


namespace number_of_valid_five_digit_numbers_l309_309149

def valid_five_digit_numbers (digits : Finset ℕ) : Finset (Finset ℕ) :=
  digits.powerset.filter (λ s, s.card = 5 ∧ ∀ (d : ℕ) (h : d ∈ s), d ≠ 1 ∨ (∃ e ∈ s, e ≠ d ∧ e ≠ 2))

theorem number_of_valid_five_digit_numbers : 
  valid_five_digit_numbers ({1, 2, 3, 4, 5}).card = 72 := 
sorry

end number_of_valid_five_digit_numbers_l309_309149


namespace sector_area_l309_309171

-- Define the given parameters
def central_angle : ℝ := 2
def radius : ℝ := 3

-- Define the statement about the area of the sector
theorem sector_area (α r : ℝ) (hα : α = 2) (hr : r = 3) :
  let l := α * r
  let A := 0.5 * l * r
  A = 9 :=
by
  -- The proof is not required
  sorry

end sector_area_l309_309171


namespace side_length_of_square_l309_309756

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l309_309756


namespace max_pieces_with_3_cuts_l309_309819

theorem max_pieces_with_3_cuts (cake : Type) : 
  (∀ (cuts : ℕ), cuts = 3 → (∃ (max_pieces : ℕ), max_pieces = 8)) := by
  sorry

end max_pieces_with_3_cuts_l309_309819


namespace num_solutions_pos_integers_l309_309100

theorem num_solutions_pos_integers (k x : ℕ) :
  ∃ k_values : finset ℕ, (∀ k ∈ k_values, ∃ x : ℤ, k * x - 18 = 5 * k) ∧ k_values.card = 6 :=
by
  let k_values := {1, 2, 3, 6, 9, 18}
  use k_values
  split
  { intro k
    intro hk
    use (18 / k + 5)
    rw [sub_eq_add_neg, ← eq_sub_iff_add_eq', add_zero, add_comm, mul_comm _ 18, ← mul_assoc, ← nat.div_add_mod 18 k,
      nat.mod_eq_zero_of_dvd _ (nat.dvd_of_mem_finset k_values hk)]
    ring_nf },
  { norm_num }

end num_solutions_pos_integers_l309_309100


namespace parabola_vertex_properties_l309_309499

theorem parabola_vertex_properties :
  let y := λ x : ℝ, -3 * x^2 - 6 * x + 2 in
  ∃ x_max : ℝ, ∃ y_max : ℝ,
    (y_max = y x_max) ∧ (x_max = 1) ∧ (y_max = -7) ∧ (∀ x : ℝ, y x ≤ y x_max) ∧ (∃ c : ℝ, c = 1 ∧ ∀ x : ℝ, x = c → y x = y x_max) :=
by {
  sorry
}

end parabola_vertex_properties_l309_309499


namespace time_between_ticks_at_6_oclock_l309_309471

theorem time_between_ticks_at_6_oclock (ticks6 ticks12 intervals6 intervals12 total_time12: ℕ) (time_per_tick : ℕ) :
  ticks6 = 6 →
  ticks12 = 12 →
  total_time12 = 66 →
  intervals12 = ticks12 - 1 →
  time_per_tick = total_time12 / intervals12 →
  intervals6 = ticks6 - 1 →
  (time_per_tick * intervals6) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end time_between_ticks_at_6_oclock_l309_309471


namespace twice_as_many_juniors_as_seniors_l309_309472

theorem twice_as_many_juniors_as_seniors (j s : ℕ) (h : (1/3 : ℝ) * j = (2/3 : ℝ) * s) : j = 2 * s :=
by
  --proof steps here
  sorry

end twice_as_many_juniors_as_seniors_l309_309472


namespace analytical_expression_of_f_solve_inequality_l309_309570

-- Define the function f
def f (x : ℝ) (a b : ℝ) := (a * x + b) / (1 + x^2)

-- Given conditions
axiom odd_f : ∀ x : ℝ, f (-x) a b = -f x a b
axiom f_half : f (1/2) a b = 2 / 5

-- Prove part 1
theorem analytical_expression_of_f (a b : ℝ) (h0 : a = 1) (h1 : b = 0) : 
  ∀ x ∈ Ioo (-1 : ℝ) (1 : ℝ), f x a b = x / (1 + x^2) := by
  sorry

-- Given the analytical expression, prove the inequality solution
theorem solve_inequality (h : ∀ x : ℝ, f x 1 0 = x / (1 + x^2)) : 
  ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → f (x-1) 1 0 + f x 1 0 < 0 := by
  sorry


end analytical_expression_of_f_solve_inequality_l309_309570


namespace chord_length_of_line_on_ellipse_l309_309323

noncomputable def chord_length_intercepted_by_line_on_ellipse (m b a c : ℝ) :=
  let k := -m in
  let x1 := (b * b - 4 * a * c) in
  let x2 := a in
  let delta := b ^ 2 - 4 * a * c in
  if delta < 0 then
    0
  else
    let sum_roots := -b / (2 * a) in
    let prod_roots := c / a in
    let root_term := sum_roots ^ 2 - 4 * prod_roots in
    let chord_length := Real.sqrt (1 + k^2) * Real.sqrt root_term in
    chord_length

theorem chord_length_of_line_on_ellipse :
  chord_length_intercepted_by_line_on_ellipse (-2) 1 6 (-1) = 5 * Real.sqrt 2 / 3 :=
by
  sorry

end chord_length_of_line_on_ellipse_l309_309323


namespace arc_length_of_path_l309_309493

def correspondence_rule (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) : ℝ × ℝ :=
  (Real.sqrt m, Real.sqrt n)

def point_A : ℝ × ℝ := (2, 6)
def point_B : ℝ × ℝ := (6, 2)

def line_AB (x y : ℝ) : Prop := x + y = 8

theorem arc_length_of_path :
  let A' := correspondence_rule 2 6 (by norm_num) (by norm_num)
  let B' := correspondence_rule 6 2 (by norm_num) (by norm_num) in
  ∃ θ : ℝ,
  cos θ = (sqrt 2 / sqrt 8) ∧ sin θ = (sqrt 6 / sqrt 8) ∧
  cos (-θ) = (sqrt 6 / sqrt 8) ∧ sin (-θ) = (sqrt 2 / sqrt 8) ∧
  (π / 3 ≤ θ ∧ θ ≤ (2 * π) / 3) ∧
  (2 * sqrt 8 * abs θ / 2) = arc_length_A'B' :=
sorry

end arc_length_of_path_l309_309493


namespace hunting_dog_strategy_l309_309030

theorem hunting_dog_strategy (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let single_dog_success_prob := p,
      both_dogs_same_path_prob := p^2,
      one_dog_correct_prob := p * (1 - p),
      combined_prob := both_dogs_same_path_prob + one_dog_correct_prob
  in combined_prob = single_dog_success_prob := 
by
  sorry

end hunting_dog_strategy_l309_309030


namespace count_div_by_nine_l309_309983

theorem count_div_by_nine (start : ℕ) (count : ℕ) (N : ℕ) 
  (hstart : start = 10) (hcount : count = 1110) :
  (∃ first multiple, first multiple > start ∧ first multiple % 9 = 0 ∧
  (N = first multiple + (hcount - 1) * 9)) → N = 9989 :=
by simp [hstart, hcount]; sorry

end count_div_by_nine_l309_309983


namespace find_x_l309_309979

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x + 1, x)
noncomputable def angle : ℝ := 45 -- in degrees

theorem find_x (x : ℝ) :
  (∃ x : ℝ, let a := (1, -1)
            let b := (x + 1, x)
            let dot_product := a.1 * b.1 + a.2 * b.2
            let magnitude_a := real.sqrt (a.1 ^ 2 + a.2 ^ 2)
            let magnitude_b := real.sqrt ((x + 1) ^ 2 + x ^ 2)
            dot_product = magnitude_a * magnitude_b * real.cos (real.pi / 4))
  → (x = 0 ∨ x = -1) :=
sorry

end find_x_l309_309979


namespace solve_for_n_l309_309856

noncomputable def regular_polygon_n (A : ℝ) (R : ℝ) : ℕ :=
  Nat.find (λ n, A = n * (R^2 / 2) * Real.sin (2 * Real.pi / n))

theorem solve_for_n :
  ∀ (R : ℝ) (A : ℝ), R = Real.sqrt 2 → A = 6 → regular_polygon_n A R = 12 :=
by
  intros R A hr ha
  -- This is where the proof would go
  sorry

end solve_for_n_l309_309856


namespace eight_natives_possible_arrangement_l309_309502

-- Define tribes as an inductive type
inductive Tribe
| tribe1
| tribe2
| tribe3

-- Eight natives represented by their tribes
def natives : list Tribe := [Tribe.tribe1, Tribe.tribe1, Tribe.tribe2, Tribe.tribe2,
                             Tribe.tribe1, Tribe.tribe1, Tribe.tribe3, Tribe.tribe3]

-- Truth-telling and lying conditions
def tells_truth (a b : Tribe) : Prop :=
  a = b

def lies (a b : Tribe) : Prop :=
  a ≠ b

-- Statement condition for the given problem
def statement_condition (natives : list Tribe) (i : ℕ) [Inhabited (list Tribe)] : Prop :=
  let left := natives[(i + natives.length - 1) % natives.length]
  let right := natives[(i + 1) % natives.length]
  if tells_truth (natives[i]) right then
    lies (natives[i]) left
  else
    tells_truth (natives[i]) left

-- Proof problem statement in Lean 4
theorem eight_natives_possible_arrangement :
  ∃ arrangement : list Tribe, (list.length arrangement = 8) ∧ 
  ∀ i : fin 8, statement_condition arrangement i := by
  sorry

end eight_natives_possible_arrangement_l309_309502


namespace range_of_m_l309_309553

-- Definitions based on conditions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → f (m) x1 > f (m) x2
def f (m : ℝ) (x : ℝ) : ℝ := -(5 - 2 * m) ^ x

theorem range_of_m (m : ℝ) : (¬ p m ∧ q m) ↔ 1 ≤ m ∧ m < 2 :=
by
  sorry

end range_of_m_l309_309553


namespace number_of_four_digit_integers_divisible_by_15_and_8_l309_309586

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def is_divisible_by_15_and_8 (n : ℕ) : Prop :=
  n % 15 = 0 ∧ n % 8 = 0

theorem number_of_four_digit_integers_divisible_by_15_and_8 : 
  {n : ℕ // is_four_digit n ∧ is_divisible_by_15_and_8 n}.to_finset.card = 75 := 
by
  sorry

end number_of_four_digit_integers_divisible_by_15_and_8_l309_309586


namespace passengers_on_board_l309_309986

/-- 
Given the fractions of passengers from different continents and remaining 42 passengers,
show that the total number of passengers P is 240.
-/
theorem passengers_on_board :
  ∃ P : ℕ,
    (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) ∧ P = 240 :=
by
  let P := 240
  have h : (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) := sorry
  exact ⟨P, h, rfl⟩

end passengers_on_board_l309_309986


namespace sum_of_digits_M_l309_309258

-- Definition of M
def lcm_upto (n : ℕ) : ℕ :=
  Nat.foldl (λ x y, Nat.lcm x y) 1 (List.range (n + 1))

def M : ℕ :=
  2 * lcm_upto 7

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Theorem statement
theorem sum_of_digits_M : sum_of_digits M = 12 := 
  sorry

end sum_of_digits_M_l309_309258


namespace expected_value_of_win_l309_309417

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l309_309417


namespace probability_product_divisible_by_8_l309_309812

-- Definitions based on conditions
def rollDie : Set ℕ := {1, 2, 3, 4, 5, 6}
def numberOfDice : Nat := 8

-- Statement of the problem
theorem probability_product_divisible_by_8 :
  let totalOutcomes := (6^numberOfDice : ℚ)
  let outcomesNotDivisibleBy2 := (3^numberOfDice : ℚ)
  let outcomesDivisibleBy2NotBy4 := (8 * ((1 : ℚ) / 6) * ((3^7 : ℚ) * (1/2^7)))
  let outcomesDivisibleBy4NotBy8 := (binomial numberOfDice 2 * ((1 : ℚ) / 6)^2 * ((3^6 : ℚ) * (1/2^6))
                                     + 8 * ((1 : ℚ) / 6) * ((3^7 : ℚ) * (1/2^7)))
  let outcomesNotDivisibleBy8 := outcomesNotDivisibleBy2 + outcomesDivisibleBy2NotBy4 + outcomesDivisibleBy4NotBy8
  let outcomesDivisibleBy8 := totalOutcomes - outcomesNotDivisibleBy8
  let probabilityDivisibleBy8 := outcomesDivisibleBy8 / totalOutcomes
  probabilityDivisibleBy8 = 1651/1728 :=
by 
  sorry

end probability_product_divisible_by_8_l309_309812


namespace bobby_initial_blocks_l309_309068

variable (b : ℕ)

theorem bobby_initial_blocks
  (h : b + 6 = 8) : b = 2 := by
  sorry

end bobby_initial_blocks_l309_309068


namespace min_value_a_squared_plus_b_squared_l309_309239

theorem min_value_a_squared_plus_b_squared :
  ∃ (a b : ℝ), (b = 3 * a - 6) → (a^2 + b^2 = 18 / 5) :=
by
  sorry

end min_value_a_squared_plus_b_squared_l309_309239


namespace expected_value_equals_51_l309_309424

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l309_309424


namespace sum_of_sequence_l309_309796

-- Sequence definition
def seq_term (k : ℕ) : ℕ := (2^k) - 1

-- Sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
  (∑ k in Finset.range n, seq_term (k + 1))

-- Statement of the problem to prove in Lean
theorem sum_of_sequence (n : ℕ) :
  sum_seq n = 2^(n + 1) - n - 2 := sorry

end sum_of_sequence_l309_309796


namespace incorrect_sampling_statement_l309_309058

/-- 
Given the following conditions about selecting a sample:
1. The selected sample must be large enough.
2. The selected sample should be universally representative.
3. The selected sample can be drawn according to one's own preferences.
4. Merely increasing the number of respondents does not necessarily improve the quality of the survey.

Prove that the statement "The selected sample can be drawn according to one's own preferences" is incorrect.
-/
theorem incorrect_sampling_statement 
  (H1 : ∀ sample, large_enough sample)
  (H2 : ∀ sample, universally_representative sample)
  (H3 : ∀ sample, can_be_drawn_according_to_own_preferences sample)
  (H4 : ∀ sample, ¬ necessarily_improve_quality_by_increasing_respondents sample) :
  ¬ can_be_drawn_according_to_own_preferences sample :=
sorry

end incorrect_sampling_statement_l309_309058


namespace Mira_trips_to_fill_tank_l309_309287

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cube (a : ℝ) : ℝ :=
  a^3

noncomputable def number_of_trips (cube_side : ℝ) (sphere_diameter : ℝ) : ℕ :=
  let r := sphere_diameter / 2
  let sphere_volume := volume_of_sphere r
  let cube_volume := volume_of_cube cube_side
  Nat.ceil (cube_volume / sphere_volume)

theorem Mira_trips_to_fill_tank : number_of_trips 8 6 = 5 :=
by
  sorry

end Mira_trips_to_fill_tank_l309_309287


namespace height_equation_from_A_to_BC_median_equation_from_A_to_BC_l309_309543

-- Define the vertices A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 3 0
def B := Point.mk 4 5
def C := Point.mk 0 7

-- Definition of line equation
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Conditions for the height and median equations
theorem height_equation_from_A_to_BC :
  ∃ l : Line, l.a = 2 ∧ l.b = -1 ∧ l.c = -6 :=
begin
  sorry
end

theorem median_equation_from_A_to_BC :
  ∃ l : Line, l.a = 6 ∧ l.b = 1 ∧ l.c = -18 :=
begin
  sorry
end

end height_equation_from_A_to_BC_median_equation_from_A_to_BC_l309_309543


namespace gabor_can_always_return_to_station_l309_309582

-- Definitions based on the problem conditions
def initial_position := "station"
def can_travel_odd_times (streets : list (string × string)) (intersection : string) : Prop :=
  ∃ paths, (∀ path ∈ paths, (fst path = intersection ∨ snd path = intersection) ∧ 
                                                      count_occ paths path % 2 ≠ 0)

-- Theorem statement
theorem gabor_can_always_return_to_station (streets : list (string × string)) :
  ∀ (current_pos : string),
    current_pos = initial_position →
    can_travel_odd_times streets initial_position →
    can_travel_odd_times streets current_pos →  
    ∃ final_pos : string, final_pos = initial_position :=
  sorry

end gabor_can_always_return_to_station_l309_309582


namespace seven_points_triangle_angle_gt_120_seven_points_all_triangle_angles_lt_varphi_eight_points_extension_l309_309394

-- Part (a)
theorem seven_points_triangle_angle_gt_120 :
  ∀ (points : Fin 7 → EuclideanSpace ℝ 2), 
  ∃ (i j k : Fin 7), 
  angle (points i) (points j) (points k) > 120 :=
sorry

-- Part (b)
theorem seven_points_all_triangle_angles_lt_varphi (φ : ℝ) (hφ : φ > 120) :
  ∃ (points : Fin 7 → EuclideanSpace ℝ 2),
  ∀ (i j k : Fin 7), 
  angle (points i) (points j) (points k) < φ :=
sorry

-- Part (c)
theorem eight_points_extension :
  (∀ (points : Fin 7 → EuclideanSpace ℝ 2), 
   ∃ (i j k : Fin 7), 
   angle (points i) (points j) (points k) > 120) ∧
  (∀ (φ : ℝ) (hφ : φ > 120), 
  ∃ (points : Fin 7 → EuclideanSpace ℝ 2),
  ∀ (i j k : Fin 7), 
  angle (points i) (points j) (points k) < φ) →
  (∀ (points : Fin 8 → EuclideanSpace ℝ 2), 
  ∃ (i j k : Fin 8), 
  angle (points i) (points j) (points k) > 120) ∧
  (∀ (φ : ℝ) (hφ : φ > 120), 
  ∃ (points : Fin 8 → EuclideanSpace ℝ 2),
  ∀ (i j k : Fin 8), 
  angle (points i) (points j) (points k) < φ) :=
sorry

end seven_points_triangle_angle_gt_120_seven_points_all_triangle_angles_lt_varphi_eight_points_extension_l309_309394


namespace additional_discount_wednesday_l309_309147

theorem additional_discount_wednesday 
  (original_price : ℝ) 
  (final_price : ℝ) 
  (summer_discount : ℝ) 
  (post_summer_discount_price : ℝ)
  (wednesday_discount_percentage : ℝ)
  (h1 : original_price = 49) 
  (h2 : summer_discount = 0.5) 
  (h3 : post_summer_discount_price = original_price * summer_discount) 
  (h4 : final_price = 14.50) 
  (h5 : post_summer_discount_price - final_price = (wednesday_discount_percentage / 100) * post_summer_discount_price)
  : wednesday_discount_percentage ≈ 40.82 :=
by
  sorry

end additional_discount_wednesday_l309_309147


namespace parallel_MN_PQ_l309_309659

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309659


namespace basketball_match_scores_l309_309223

theorem basketball_match_scores :
  ∃ (a r b d : ℝ), (a = b) ∧ (a * (1 + r + r^2 + r^3) < 120) ∧
  (4 * b + 6 * d < 120) ∧ ((a * (1 + r + r^2 + r^3) - (4 * b + 6 * d)) = 3) ∧
  a + b + (a * r + (b + d)) = 35.5 :=
sorry

end basketball_match_scores_l309_309223


namespace group1_calculation_group2_calculation_l309_309074

theorem group1_calculation : 9 / 3 * (9 - 1) = 24 := by
  sorry

theorem group2_calculation : 7 * (3 + 3 / 7) = 24 := by
  sorry

end group1_calculation_group2_calculation_l309_309074


namespace cube_root_of_square_root_l309_309073

theorem cube_root_of_square_root (h : Real.sqrt 0.0081 = (9/100 : ℝ)) : 
  Real.sqrt (Real.cbrt (0.0081)) = 0.34 :=
by
  sorry

end cube_root_of_square_root_l309_309073


namespace bacon_calories_percentage_l309_309885

-- Mathematical statement based on the problem
theorem bacon_calories_percentage :
  ∀ (total_sandwich_calories : ℕ) (number_of_bacon_strips : ℕ) (calories_per_strip : ℕ),
    total_sandwich_calories = 1250 →
    number_of_bacon_strips = 2 →
    calories_per_strip = 125 →
    (number_of_bacon_strips * calories_per_strip) * 100 / total_sandwich_calories = 20 :=
by
  intros total_sandwich_calories number_of_bacon_strips calories_per_strip h1 h2 h3 
  sorry

end bacon_calories_percentage_l309_309885


namespace pentagon_length_DE_l309_309233

/-- Pentagon $ABCDE$ where:
    - $AB=BC=CD=2$ units,
    - $\angle A$ is a right angle (90°),
    - $\angle B = \angle C = \angle D = 120°$,
   leads to the conclusion that the length of segment $DE$ can be expressed in simplest radical form as $0 + \sqrt{12}$,
   hence $a + b = 12$. -/
theorem pentagon_length_DE (AB BC CD : ℝ) (angleA : ℝ) (angleBCD : ℝ) (angleCDE : ℝ) 
(D E : ℂ) 
(h_AB : AB = 2) (h_BC : BC = 2) (h_CD : CD = 2) (h_angleA : angleA = 90) 
(h_angleBCD : angleBCD = 120) (h_angleCDE : angleCDE = 120) :
  let DE := complex.abs (D - E) in
  ∃ a b : ℝ, DE = a + real.sqrt b ∧ a + b = 12 :=
begin
  sorry
end

end pentagon_length_DE_l309_309233


namespace candy_cost_l309_309714

-- Definitions and assumptions from problem conditions
def cents_per_page := 1
def pages_per_book := 150
def books_read := 12
def leftover_cents := 300  -- $3 in cents

-- Total pages read
def total_pages_read := pages_per_book * books_read

-- Total earnings in cents
def total_cents_earned := total_pages_read * cents_per_page

-- Cost of the candy in cents
def candy_cost_cents := total_cents_earned - leftover_cents

-- Theorem statement
theorem candy_cost : candy_cost_cents = 1500 := 
  by 
    -- proof goes here
    sorry

end candy_cost_l309_309714


namespace total_cakes_served_l309_309049

def weekday_cakes_lunch : Nat := 6 + 8 + 10
def weekday_cakes_dinner : Nat := 9 + 7 + 5 + 13
def weekday_cakes_total : Nat := weekday_cakes_lunch + weekday_cakes_dinner

def weekend_cakes_lunch : Nat := 2 * (6 + 8 + 10)
def weekend_cakes_dinner : Nat := 2 * (9 + 7 + 5 + 13)
def weekend_cakes_total : Nat := weekend_cakes_lunch + weekend_cakes_dinner

def total_weekday_cakes : Nat := 5 * weekday_cakes_total
def total_weekend_cakes : Nat := 2 * weekend_cakes_total

def total_week_cakes : Nat := total_weekday_cakes + total_weekend_cakes

theorem total_cakes_served : total_week_cakes = 522 := by
  sorry

end total_cakes_served_l309_309049


namespace trisha_annual_take_home_pay_l309_309367

noncomputable def hourly_rate : ℝ := 15
noncomputable def weekly_hours : ℝ := 40
noncomputable def weeks_per_year : ℝ := 52
noncomputable def overtime_multiplier : ℝ := 1.5
noncomputable def overtime_hours_per_quarter : ℝ := 20
noncomputable def quarters_per_year : ℝ := 4

noncomputable def federal_tax_bracket1_rate : ℝ := 0.10
noncomputable def federal_tax_bracket1_limit : ℝ := 10000
noncomputable def federal_tax_bracket2_rate : ℝ := 0.12
noncomputable def federal_tax_bracket2_limit : ℝ := 20000

noncomputable def state_tax_rate : ℝ := 0.05
noncomputable def unemployment_rate : ℝ := 0.01
noncomputable def unemployment_limit : ℝ := 7000
noncomputable def social_security_rate : ℝ := 0.062
noncomputable def social_security_limit : ℝ := 142800

theorem trisha_annual_take_home_pay : 
  let annual_gross_income := hourly_rate * weekly_hours * weeks_per_year
  let annual_overtime_pay := hourly_rate * overtime_multiplier * overtime_hours_per_quarter * quarters_per_year
  let total_annual_gross_income := annual_gross_income + annual_overtime_pay
  let federal_tax := federal_tax_bracket1_rate * min total_annual_gross_income federal_tax_bracket1_limit + federal_tax_bracket2_rate * min (total_annual_gross_income - federal_tax_bracket1_limit) federal_tax_bracket2_limit
  let state_tax := state_tax_rate * total_annual_gross_income
  let unemployment_tax := unemployment_rate * min total_annual_gross_income unemployment_limit
  let social_security_tax := social_security_rate * min total_annual_gross_income social_security_limit
  let total_deductions := federal_tax + state_tax + unemployment_tax + social_security_tax
  let annual_take_home_pay := total_annual_gross_income - total_deductions
  in annual_take_home_pay = 25474 :=
by sorry

end trisha_annual_take_home_pay_l309_309367


namespace bottles_left_on_shelf_l309_309353

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l309_309353


namespace eleventh_number_is_166_l309_309455

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def digits_sum_13 (n : ℕ) : Prop :=
  sum_of_digits n = 13

def increasing_list_numbers := list.Icc 1 9999 |>.filter digits_sum_13

def num_at_index_11 := increasing_list_numbers.nth 10  -- lists are 0-indexed

theorem eleventh_number_is_166 :
  num_at_index_11 = some 166 :=
by
  -- Skip proof
  sorry

end eleventh_number_is_166_l309_309455


namespace pairs_meet_once_impossible_l309_309850

theorem pairs_meet_once_impossible (n : ℕ) (h_n : n = 100) : ¬ ∃ (schedule : finset (finset (fin 100))), 
  (∀ d ∈ schedule, d.card = 3) ∧ (∀ x y : fin 100, x ≠ y → ∃! (d ∈ schedule), {x, y} ⊆ d) :=
by {
  -- Given 100 people
  have h1 : n = 100 := h_n,
  -- Assume we can find a schedule satisfying the conditions
  assume h2: ∃ (schedule : finset (finset (fin 100))),
    (∀ d ∈ schedule, d.card = 3) ∧ (∀ x y : fin 100, x ≠ y → ∃! (d ∈ schedule), {x, y} ⊆ d),
  -- Derive a contradiction
  sorry,
}

end pairs_meet_once_impossible_l309_309850


namespace find_angle_A_find_range_of_b_plus_c_l309_309557

-- Given conditions
variables {A B C : ℝ} {a b c : ℝ}
variables (h1 : 0 < A ∧ A < π)   -- $\triangle ABC$ is acute-angled
           (h2 : b^2 + c^2 = bc + a^2)  -- b^2 + c^2 = bc + a^2

-- Problem statement 1: Finding angle A
theorem find_angle_A : A = π / 3 :=
by
  -- Proof skipped
  sorry

-- Problem statement 2: Finding the range of b + c, given a = sqrt(3)
theorem find_range_of_b_plus_c (ha : a = sqrt 3) : 3 < b + c ∧ b + c ≤ 2 * sqrt 3 :=
by
  -- Proof skipped
  sorry

end find_angle_A_find_range_of_b_plus_c_l309_309557


namespace range_of_f_l309_309641

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x) ^ 3 + (Real.arcsin x) ^ 3

theorem range_of_f : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 
           ∃ y : ℝ, y = f x ∧ (y ≥ (Real.pi ^ 3) / 32) ∧ (y ≤ (7 * (Real.pi ^ 3)) / 8) :=
sorry

end range_of_f_l309_309641


namespace number_greater_than_213_l309_309220

theorem number_greater_than_213 (x : ℤ) (h1: 2.13 * 10 ^ x < 214) : x ≤ 2 := 
sorry

end number_greater_than_213_l309_309220


namespace smallest_digit_divisible_by_9_l309_309143

theorem smallest_digit_divisible_by_9 : 
  ∃ (d : ℕ), 0 ≤ d ∧ d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ d = 2 :=
by
  use 2
  split
  { exact nat.zero_le _ }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end smallest_digit_divisible_by_9_l309_309143


namespace collinear_intersections_l309_309163

-- Given conditions:
variable (A B C T : Point)
variable (t : Line)
variable (R1 R2 R3 : Point) -- Positions for reflections
variable (p q r : Line) -- Intersection lines

-- Definitions reflecting points:
def reflect (P : Point) (l : Line) : Point := sorry -- The reflection definition

noncomputable def P := intersection (p) (line B C)
noncomputable def Q := intersection (q) (line C A)
noncomputable def R := intersection (r) (line A B)

-- Problem statement:
theorem collinear_intersections 
  (h1 : intersects T t) 
  (hP : intersects p (line B C)) 
  (hQ : intersects q (line C A)) 
  (hR : intersects r (line A B)) :
  collinear [P, Q, R] :=
sorry

end collinear_intersections_l309_309163


namespace points_difference_is_90_l309_309638

theorem points_difference_is_90 (a b c : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  |(100 * a + 10 * b + c) - (100 * b + 10 * a + c)| = 90 * |a - b| :=
by
  sorry

end points_difference_is_90_l309_309638


namespace hunting_dog_strategy_l309_309031

theorem hunting_dog_strategy (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let single_dog_success_prob := p,
      both_dogs_same_path_prob := p^2,
      one_dog_correct_prob := p * (1 - p),
      combined_prob := both_dogs_same_path_prob + one_dog_correct_prob
  in combined_prob = single_dog_success_prob := 
by
  sorry

end hunting_dog_strategy_l309_309031


namespace simplify_and_rationalize_denominator_l309_309309

theorem simplify_and_rationalize_denominator :
  ( (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) = 3 * Real.sqrt 420 / 42 ) := 
by {
  sorry
}

end simplify_and_rationalize_denominator_l309_309309


namespace curve_C_eq_min_distance_M_to_l_l309_309082

noncomputable def P_rect_coords : ℝ × ℝ :=
  (3, real.sqrt 3)

theorem curve_C_eq : ∀ x y: ℝ, (x^2 + (y + real.sqrt 3)^2 = 4) ↔ 
  ∃ (ρ θ : ℝ), (ρ^2 + 2*real.sqrt 3 * ρ * real.sin θ = 1) ∧ 
  (x = ρ * real.cos θ) ∧ 
  (y = ρ * real.sin θ) :=
by {
  sorry
}

theorem min_distance_M_to_l : ∀ (t : ℝ), ∀ (Q : ℝ × ℝ), 
  ((Q.1)^2 + (Q.2 + real.sqrt 3)^2 = 4) → 
  let Mux := (3 + Q.1) / 2 in
  let Muy := (real.sqrt 3 + Q.2) / 2 in 
  dist (Mux, Muy) (3 + 2*t, -2 + t) = (11 * real.sqrt 5 / 10 - 1) :=
by {
  sorry
}

end curve_C_eq_min_distance_M_to_l_l309_309082


namespace remaining_grass_area_l309_309845

theorem remaining_grass_area (R r : ℝ) (path_width : ℝ)
  (hR : R = 10) (hr : r = 8) (h_path_width : path_width = 2)
  (h_tangent : r + path_width = R) :
  real.pi * r^2 = 64 * real.pi :=
by
  -- Definitions of the radii and dimensions
  rw [hR, hr, h_path_width, h_tangent]
  -- Area calculations
  sorry

end remaining_grass_area_l309_309845


namespace circumscribed_circle_intersection_l309_309629

noncomputable def hypotenuse : ℝ := 41

def length_XY : ℝ := 37
def length_YZ : ℝ := 20
def length_XZ : ℝ := hypotenuse

theorem circumscribed_circle_intersection (R : ℝ)
  (h1 : 2 * R ≥ 2) -- Ensuring R is positive
  (h2 : NE * (2 * R - NE) = (length_XZ / 2) ^ 2) :
  ∃ (p q : ℝ), q ≠ 0 ∧ ¬(∃ z : ℤ, z^2 ∣ q) ∧
  ∃ (E : EuclideanGeometry.Point ℝ), EuclideanGeometry.length ⟨37, 20⟩ E = p * Real.sqrt q ∧ 
  ⌊p + Real.sqrt q⌋ = 36 :=
begin
  -- The proof is omitted as required by the problem statement
  sorry
end

end circumscribed_circle_intersection_l309_309629


namespace bananas_in_collection_l309_309296

theorem bananas_in_collection
  (groups : ℕ)
  (bananas_per_group : ℕ)
  (h1 : groups = 11)
  (h2 : bananas_per_group = 37) :
  (groups * bananas_per_group) = 407 :=
by sorry

end bananas_in_collection_l309_309296


namespace yang_hui_problem_l309_309081

theorem yang_hui_problem (x : ℝ) :
  x * (x + 12) = 864 :=
sorry

end yang_hui_problem_l309_309081


namespace product_of_repeating_decimal_l309_309133

-- Define the repeating decimal 0.3
def repeating_decimal : ℚ := 1 / 3
-- Define the question
def product (a b : ℚ) := a * b

-- State the theorem to be proved
theorem product_of_repeating_decimal :
  product repeating_decimal 8 = 8 / 3 :=
sorry

end product_of_repeating_decimal_l309_309133


namespace no_such_arrangement_l309_309008

theorem no_such_arrangement :
  ¬∃ (a : Fin 111 → ℕ), (∀ i : Fin 111, a i ≤ 500 ∧ (∀ j k : Fin 111, j ≠ k → a j ≠ a k)) ∧ (∀ i : Fin 111, (a i % 10) = ((Finset.univ.sum (λ j, if j = i then 0 else a j)) % 10)) :=
by
  sorry

end no_such_arrangement_l309_309008


namespace parallel_MN_PQ_l309_309653

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309653


namespace baron_munchausen_truth_l309_309063

def sum_of_digits_squared (n : ℕ) : ℕ :=
  (n.digits 10).sum (λ d, d ^ 2)

theorem baron_munchausen_truth : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a.digits.length = 10 ∧ 
    b.digits.length = 10 ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    (a - sum_of_digits_squared a) = 
    (b - sum_of_digits_squared b) :=
begin
  use 10^9 + 8,
  use 10^9 + 9,
  split,
  { exact ne_of_lt (nat.lt_succ_self (10^9 + 8)) }, -- proof of a ≠ b
  split,
  { exact rfl }, -- proof of a is 10 digits long
  split,
  { exact rfl }, -- proof of b is 10 digits long
  split,
  { norm_num }, -- proof a % 10 ≠ 0
  split,
  { norm_num }, -- proof b % 10 ≠ 0
  { sorry },
end

end baron_munchausen_truth_l309_309063


namespace expected_value_of_win_is_51_l309_309420

noncomputable def expected_value_of_win : ℝ :=
  (∑ n in (finset.range 8).map (λ x, x + 1), (1/8) * 2 * (n : ℝ)^2)

theorem expected_value_of_win_is_51 : expected_value_of_win = 51 := 
by 
  sorry

end expected_value_of_win_is_51_l309_309420


namespace product_bc_l309_309802

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end product_bc_l309_309802


namespace arithmetic_seq_a3_value_l309_309619

-- Given the arithmetic sequence {a_n}, where
-- a_1 + a_2 + a_3 + a_4 + a_5 = 20
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_seq_a3_value {a : ℕ → ℝ}
    (h_seq : arithmetic_seq a)
    (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
by
  sorry

end arithmetic_seq_a3_value_l309_309619


namespace probability_at_least_six_points_distribution_and_expectation_l309_309842

section part_one

/-- Prove that the probability that Team A will score at least 6 points after 
answering one question each from Traffic Safety and Fire Safety is 5/6, given 
the correct rates. -/
theorem probability_at_least_six_points (p_traffic : ℝ) (p_fire : ℝ) :
  p_traffic = 2/3 → p_fire = 1/2 → 
  let P := p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire in
  P = 5/6 :=
by
  intro h1 h2
  let p := p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire
  sorry

end part_one

section part_two

/-- Prove that the score distribution for Team A after answering 3 distinct 
category questions (Traffic Safety, Fire Safety, Water Safety) is Y ∈ {3, 7, 11, 15} 
with probabilities specified and that the expected score is 9, given the correct rates. -/
theorem distribution_and_expectation (r_traffic r_fire r_water : ℝ) :
  r_traffic = 2/3 → r_fire = 1/2 → r_water = 1/3 → 
  let P := [((3, 1/9)), ((7, 7/18)), ((11, 7/18)), ((15, 1/9))]
  let E_Y := 3 * (1 / 9) + 7 * (7 / 18) + 11 * (7 / 18) + 15 * (1 / 9)
  P.map Prod.snd = [1/9, 7/18, 7/18, 1/9] ∧ E_Y = 9 :=
by
  intro h1 h2 h3
  sorry

end part_two

end probability_at_least_six_points_distribution_and_expectation_l309_309842


namespace arithmetic_sequence_condition_l309_309957

theorem arithmetic_sequence_condition {a : ℕ → ℤ} 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (m p q : ℕ) (hpq_pos : 0 < p) (hq_pos : 0 < q) (hm_pos : 0 < m) : 
  (p + q = 2 * m) → (a p + a q = 2 * a m) ∧ ¬((a p + a q = 2 * a m) → (p + q = 2 * m)) :=
by 
  sorry

end arithmetic_sequence_condition_l309_309957


namespace distance_AB_l309_309237

open real

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * cos theta, rho * sin theta)

def curve (theta : ℝ) : ℝ × ℝ :=
  polar_to_cartesian (-2 * sin theta) theta

def line : ℝ × ℝ → Prop :=
  λ p, p.snd = -1

theorem distance_AB :
  ∃ A B : ℝ × ℝ,
    (∃ θ₁ θ₂ : ℝ, curve θ₁ = A ∧ curve θ₂ = B ∧ line A ∧ line B) ∧
    dist A B = 2 :=
by
  sorry

end distance_AB_l309_309237


namespace log_identity_unique_n_l309_309900

theorem log_identity_unique_n :
  ∃ (n : ℕ), (n > 0 ∧ log 2 (log 8 (n ^ 4 : ℝ)) = log 4 (log 4 (n ^ 2 : ℝ))) ∧ n = 3 := by
sorry

end log_identity_unique_n_l309_309900


namespace statistical_measures_independence_l309_309340

def number_of_students := 30
def known_scores := [(26, 2), (27, 3), (28, 6), (29, 7), (30, 9)]
def missing_data_count := number_of_students - (2 + 3 + 6 + 7 + 9) -- 3
def missing_scores := [24, 25]

def mode := 30
def median := 29

theorem statistical_measures_independence :
  (∀ a b c d : Type, ∀ (x : a) (y : b) (z : c) (w : d), x = mode ∧ y = median) ∧
  ¬ (∀ m n : Type, ∀ (mean_val : m) (variance_val : n), true) :=
by sorry

end statistical_measures_independence_l309_309340


namespace figure_area_is_correct_l309_309695

def M_figure_area : ℝ :=
  6 * Real.pi - Real.sqrt 3

theorem figure_area_is_correct :
  ∀ (x y : ℝ),
  (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 ≤ 2 ∧ a^2 + b^2 ≤ min (2*a + 2*b) 2) →
  area_of_figure_is M_figure_area := sorry

end figure_area_is_correct_l309_309695


namespace bankers_discount_l309_309395

variable (T : ℕ) (R BG : ℝ)

def SimpleInterest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem bankers_discount (h1 : T = 3) (h2 : R = 10) (h3 : BG = 60) : 
  let TD := SimpleInterest BG 100 (R * T),
      BD := BG + TD in
  BD = 260 :=
by
  rw [h1, h2, h3]
  let TD := SimpleInterest 60 100 (10 * 3)
  let BD := 60 + TD
  exact calc
    BD = 260 := sorry

end bankers_discount_l309_309395


namespace find_x_y_sum_l309_309091

def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem find_x_y_sum (n x y : ℕ) (hn : n = 450) (hx : x > 0) (hy : y > 0)
  (hxsq : is_perfect_square (n * x))
  (hycube : is_perfect_cube (n * y)) :
  x + y = 62 :=
  sorry

end find_x_y_sum_l309_309091


namespace largest_possible_package_l309_309057

/-- Alice, Bob, and Carol bought certain numbers of markers and the goal is to find the greatest number of markers per package. -/
def alice_markers : Nat := 60
def bob_markers : Nat := 36
def carol_markers : Nat := 48

theorem largest_possible_package :
  Nat.gcd (Nat.gcd alice_markers bob_markers) carol_markers = 12 :=
sorry

end largest_possible_package_l309_309057


namespace projection_of_a_onto_b_l309_309935

def vector_projection (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (b.1 * b.1 + b.2 * b.2))

theorem projection_of_a_onto_b :
  vector_projection (2, 1) (3, 4) = 2 :=
by
  -- The proof is omitted
  sorry

end projection_of_a_onto_b_l309_309935


namespace count_ways_to_choose_l309_309284

open Finset

def I : Finset ℕ := {1, 2, 3, 4, 5}

noncomputable def count_ways : ℕ :=
  let subsets := powerset I in
  let valid_pairs := (subsets ×ˢ subsets).filter 
                      (λ (p : Finset ℕ × Finset ℕ), 
                        p.1.nonempty ∧ p.2.nonempty ∧ p.1.max' sorry < p.2.min' sorry) in
  valid_pairs.card

theorem count_ways_to_choose : count_ways = 49 := sorry

end count_ways_to_choose_l309_309284


namespace odd_number_red_faces_in_1_inch_cubes_l309_309870

theorem odd_number_red_faces_in_1_inch_cubes :
  let total_cubes := 25,
      corner_cubes := 4,
      edge_cubes := 12,
      center_cubes := 9
  in (corner_cubes + center_cubes = 13) :=
by
  let total_cubes := 25
  let corner_cubes := 4
  let edge_cubes := 12
  let center_cubes := 9
  have h1 : corner_cubes + center_cubes = 13 := by sorry
  exact h1

end odd_number_red_faces_in_1_inch_cubes_l309_309870


namespace range_of_a_l309_309955

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, (exp x - (a / x)) ≥ 0) → a ≤ exp 1 :=
by {
  sorry
}

end range_of_a_l309_309955


namespace num_prime_factors_30_fact_l309_309477

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def is_prime (n : ℕ) : Bool :=
  if h : n ≤ 1 then false else
    let divisors := List.range (n - 2) |>.map (· + 2)
    !divisors.any (· ∣ n)

def primes_upto (n : ℕ) : List ℕ :=
  List.range (n - 1) |>.map (· + 1) |>.filter is_prime

def count_primes_factorial_upto (n : ℕ) : ℕ :=
  (primes_upto n).length

theorem num_prime_factors_30_fact : count_primes_factorial_upto 30 = 10 := sorry

end num_prime_factors_30_fact_l309_309477


namespace eleventh_number_is_175_l309_309453

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum
-- The digits_sum function calculates the sum of digits of a given natural number n in base 10

def eleventh_number_with_digits_sum_13 : ℕ :=
175 -- This is the correct answer based on the problem solution

theorem eleventh_number_is_175 :
  ∃! n : ℕ, (digits_sum n = 13 ∧
             ∃ l : List ℕ, (l.nth 10 = some n ∧
                             (∀ m, digits_sum m = 13 → List.nthLe l (l.indexOf m) sorry = m) ∧
                             (∀ m1 m2, l.indexOf m1 < l.indexOf m2 ↔ m1 < m2))) →
             n = eleventh_number_with_digits_sum_13 :=
sorry

end eleventh_number_is_175_l309_309453


namespace green_eyed_brunettes_percentage_l309_309457

noncomputable def green_eyed_brunettes_proportion (a b c d : ℕ) 
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7) 
  (h3 : c / (c + d) = 0.1) : Prop :=
  d / (a + b + c + d) = 0.54

-- The main theorem to be proved
theorem green_eyed_brunettes_percentage (a b c d : ℕ)
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7)
  (h3 : c / (c + d) = 0.1) : 
  green_eyed_brunettes_proportion a b c d h1 h2 h3 := 
sorry

end green_eyed_brunettes_percentage_l309_309457


namespace find_b_l309_309947

noncomputable def a := Real.sqrt 10
noncomputable def c := 3
noncomputable def cosA := 1 / 4

-- Define the cosine theorem in Lean
def cosine_theorem (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)

-- Lean 4 statement
theorem find_b : a = Real.sqrt 10 → c = 3 → cosA = 1/4 → ∃ b, cosine_theorem a b c = cosA ∧ b = 2 :=
by
  intros ha hc hcosA
  use 2
  rw [cosine_theorem, ha, hc, hcosA]
  have hb : 2 * 2^2 - 3 * 2 - 2 = 0, by norm_num
  simp [hb]
  sorry

end find_b_l309_309947


namespace point_M_first_quadrant_distances_length_of_segment_MN_l309_309161

-- Proof problem 1
theorem point_M_first_quadrant_distances (m : ℝ) (h1 : 2 * m + 1 > 0) (h2 : m + 3 > 0) (h3 : m + 3 = 2 * (2 * m + 1)) :
  m = 1 / 3 :=
by
  sorry

-- Proof problem 2
theorem length_of_segment_MN (m : ℝ) (h4 : m + 3 = 1) :
  let Mx := 2 * m + 1
  let My := m + 3
  let Nx := 2
  let Ny := 1
  let distMN := abs (Nx - Mx)
  distMN = 5 :=
by
  sorry

end point_M_first_quadrant_distances_length_of_segment_MN_l309_309161


namespace boat_traveled_downstream_distance_l309_309018

theorem boat_traveled_downstream_distance
  (speed_boat_still_water : ℝ)
  (speed_stream : ℝ)
  (time_downstream : ℝ)
  (effective_speed_downstream : ℝ)
  (distance_downstream : ℝ) :
  speed_boat_still_water = 22 →
  speed_stream = 5 →
  time_downstream = 4 →
  effective_speed_downstream = speed_boat_still_water + speed_stream →
  distance_downstream = effective_speed_downstream * time_downstream →
  distance_downstream = 108 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end boat_traveled_downstream_distance_l309_309018


namespace domain_of_function_l309_309770

theorem domain_of_function (x : ℝ) :
  (x^2 - 5*x + 6 ≥ 0) → (x ≠ 2) → (x < 2 ∨ x ≥ 3) :=
by
  intros h1 h2
  sorry

end domain_of_function_l309_309770


namespace jared_sarah_same_color_prob_eq_l309_309039

noncomputable
def pick_same_color_probability : ℚ := 
  let total_candies := 24
  let jared_picks := 3
  let sarah_picks := 3
  let red_candies := 8
  let blue_candies := 8
  let green_candies := 8
  
  -- Probability Jared picks (r1, b1, g1) candies
  let P_J :
    (ℕ × ℕ × ℕ) → ℚ := 
    λ ⟨r1, b1, g1⟩, 
      if r1 + b1 + g1 = jared_picks 
      then (nat.choose red_candies r1 *
            nat.choose blue_candies b1 *
            nat.choose green_candies g1) /
           (nat.choose total_candies jared_picks : ℚ) 
      else 0

  -- Remaining candies after Jared's pick
  let remaining_candies (r1 b1 g1 : ℕ) := 
    (red_candies - r1, blue_candies - b1, green_candies - g1)
  
  -- Probability Sarah picks the same (r1, b1, g1) candies
  let P_S :
    (ℕ × ℕ × ℕ) → ℕ → ℚ := 
    λ ⟨r1, b1, g1⟩ (total_remaining : ℕ), 
      if r1 + b1 + g1 = sarah_picks
      then (nat.choose (red_candies - r1) r1 *
            nat.choose (blue_candies - b1) b1 *
            nat.choose (green_candies - g1) g1) /
           (nat.choose total_remaining sarah_picks : ℚ)
      else 0

  let total_prob : ℚ := 
    [ (0, 0, 3), (0, 1, 2), (0, 2, 1),
      (0, 3, 0), (1, 0, 2), (1, 1, 1),
      (1, 2, 0), (2, 0, 1), (2, 1, 0),
      (3, 0, 0) -- list all valid combinations
    ].foldl
      (λ acc ⟨r1, b1, g1⟩, 
        let rem := remaining_candies r1 b1 g1
        acc + P_J (r1, b1, g1) * P_S (r1, b1, g1) (total_candies - jared_picks)
      )
      0

  -- Assuming total_prob as a simplified fraction m/n
  let simplified := total_prob.num + total_prob.denom

  simplified

theorem jared_sarah_same_color_prob_eq (m n : ℕ) :
  ∑ i in finset.range 10, 
    let ⟨r1, b1, g1⟩ := finset.nth finset.range i
    pick_same_color_probability = m / n := 
  sorry -- proof is omitted

end jared_sarah_same_color_prob_eq_l309_309039


namespace parallel_MN_PQ_l309_309657

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309657


namespace find_divisor_l309_309608

theorem find_divisor (Q R D V : ℤ) (hQ : Q = 65) (hR : R = 5) (hV : V = 1565) (hEquation : V = D * Q + R) : D = 24 :=
by
  sorry

end find_divisor_l309_309608


namespace min_value_y_l309_309561

variable {R : Type*} [LinearOrderedField R]

def point (x y : R) := (x, y)
def line_through (A B : R × R) : R × R → Prop :=
  λ C, ∃ t : R, C.1 = A.1 + t * (B.1 - A.1) ∧ C.2 = A.2 + t * (B.2 - A.2)

theorem min_value_y (a b : R) (hC : line_through (1, 1) (-2, 4) (a, b)) :
  (∃ C, C = (a, b) ∧ a + b = 2) →
  ∀ y, y = (1 / a + 4 / b) →
  y ≥ 9 / 2 :=
sorry

end min_value_y_l309_309561


namespace fraction_enlarged_by_three_times_l309_309210

theorem fraction_enlarged_by_three_times (x y : ℝ) :
  let f := λ x y : ℝ, 2 * x * y / (x + y) in
  f (3 * x) (3 * y) = 3 * f x y :=
sorry

end fraction_enlarged_by_three_times_l309_309210


namespace height_pillar_D_l309_309874

-- Definitions based on conditions
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 0, y := 0, z := 15 }
def B : Point3D := { x := 10, y := 0, z := 11 }
def C : Point3D := { x := 5, y := 5*Real.sqrt 3, z := 13 }
def D : Point3D := { x := -10, y := 0, z := 0 } -- z is unknown and to be determined

-- The function to calculate the height at D
def height_at_D (d : Point3D) : ℝ :=
  let PQ := ⟨ B.x - A.x, B.y - A.y, B.z - A.z ⟩
  let PR := ⟨ C.x - A.x, C.y - A.y, C.z - A.z ⟩
  -- The normal vector
  let n := ⟨ PQ.2 * PR.3 - PQ.3 * PR.2,
              PQ.3 * PR.1 - PQ.1 * PR.3,
              PQ.1 * PR.2 - PQ.2 * PR.1 ⟩
  -- The plane equation Ax + By + Cz = D
  let D_const := n.1 * A.x + n.2 * A.y + n.3 * A.z
  -- Solve for z in the plane equation at point D
  (D_const - (n.1 * d.x + n.2 * d.y)) / n.3

-- Statement of the proof problem
theorem height_pillar_D : height_at_D D = 20 :=
by
  -- The steps of the solution go here (skipped for now)
  sorry

end height_pillar_D_l309_309874


namespace circle_radius_parallel_tangents_l309_309617

theorem circle_radius_parallel_tangents (AX BY : ℝ) (r : ℝ) (AX_eq : AX = 6) (BY_eq : BY = 15)
  (AX_parallel_BY : AX ∥ BY) (AX BY_tangent_circle : AX ⊥ r ∧ BY ⊥ r)
  (C_opposite_B : C = -B) :
  r = sqrt 261 :=
by
  sorry

end circle_radius_parallel_tangents_l309_309617


namespace expected_value_8_sided_die_l309_309415

-- Define the roll outcomes and their associated probabilities
def roll_outcome (n : ℕ) : ℕ := 2 * n^2

-- Define the expected value calculation
def expected_value (sides : ℕ) : ℚ := ∑ i in range (1, sides+1), (1 / sides) * roll_outcome i

-- Prove the expected value calculation for an 8-sided fair die
theorem expected_value_8_sided_die : expected_value 8 = 51 := by
  sorry

end expected_value_8_sided_die_l309_309415


namespace dilation_and_rotation_l309_309123

def dilation_matrix (factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![factor, 0], ![0, factor]]

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

def combined_transformation_matrix (factor : ℝ) (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (rotation_matrix θ) ⬝ (dilation_matrix factor)

theorem dilation_and_rotation :
  combined_transformation_matrix 2 (Real.pi / 2) = ![![0, -2], ![2, 0]] :=
by
  sorry

end dilation_and_rotation_l309_309123


namespace mr_thompson_third_score_is_78_l309_309288

theorem mr_thompson_third_score_is_78 :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
                   (a = 58 ∧ b = 65 ∧ c = 70 ∧ d = 78) ∧ 
                   (a + b + c + d) % 4 = 3 ∧ 
                   (∀ i j k, (a + i + j + k) % 4 = 0) ∧ -- This checks that average is integer
                   c = 78 := sorry

end mr_thompson_third_score_is_78_l309_309288


namespace sum_F_G_H_l309_309591

theorem sum_F_G_H : 
  ∀ (F G H : ℕ), 
    (F < 10 ∧ G < 10 ∧ H < 10) ∧ 
    ∃ k : ℤ, 
      (F - 8 + 6 - 1 + G - 2 - H - 11 * k = 0) → 
        F + G + H = 23 :=
by sorry

end sum_F_G_H_l309_309591


namespace range_of_y_over_x_l309_309997

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : 
  y / x ∈ set.Icc (-real.sqrt 3) (real.sqrt 3) :=
sorry

end range_of_y_over_x_l309_309997


namespace intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l309_309577

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (3 * a + 1)) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a ^ 2 + 1)) < 0 }

-- Theorem for question (1): Intersection of A and B when a = 2
theorem intersection_of_A_and_B_when_a_is_2 :
  setA 2 ∩ setB 2 = { x | 4 < x ∧ x < 5 } :=
sorry

-- Theorem for question (2): Range of a such that B ⊆ A
theorem range_of_a_such_that_B_subset_A :
  { a : ℝ | setB a ⊆ setA a } = { x | 1 < x ∧ x ≤ 3 } ∪ { -1 } :=
sorry

end intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l309_309577


namespace geometric_sequence_first_term_and_common_ratio_geometric_sequence_sum_of_bn_l309_309538

open Real

theorem geometric_sequence_first_term_and_common_ratio (S_n : ℕ → ℝ) (a : ℕ → ℝ) 
  (hSn : ∀ n, S_n n = 2^(n+1) - 2) :
  a 1 = 2 ∧ (a 2) / (a 1) = 2 :=
by 
  sorry

theorem geometric_sequence_sum_of_bn (S_n : ℕ → ℝ) (a b : ℕ → ℝ)
  (hSn : ∀ n, S_n n = 2^(n+1) - 2) 
  (ha1 : a 1 = 2) (hq : ∀ n, a (n+1) = (a n) * 2) :
  (∀ n, b n = Real.log2 (a n)) → 
  (∀ n, ∑ k in Finset.range n, b (k+1) = (n * (n + 1)) / 2) :=
by 
  sorry

end geometric_sequence_first_term_and_common_ratio_geometric_sequence_sum_of_bn_l309_309538


namespace inverse_of_P_implies_negation_l309_309213

-- Define a proposition P.
variable (P : Prop)

-- Define the condition: the inverse of proposition P is true.
def inverse_of_P_true : Prop := ¬P

-- The mathematically equivalent proof problem: if the inverse of a proposition P is true, then the negation of P is true.
theorem inverse_of_P_implies_negation (h : inverse_of_P_true P) : ¬P := 
by assume h
sorried.


end inverse_of_P_implies_negation_l309_309213


namespace inscribed_circle_radius_l309_309102

theorem inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 30) : 
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let area := (d1 * d2) / 2 in
  let r := area / (2 * a) in
  r = 30 / Real.sqrt 29 := 
by
  -- begin the proof
  sorry

end inscribed_circle_radius_l309_309102


namespace part1_part2_l309_309180

def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem part1 (x : ℝ) : f(x) ≤ 5 → x ∈ Set.Icc (-3.5) 1.5 :=
sorry

theorem part2 (t : ℝ) : (∃ x : ℝ, t^2 + 3 * t > f(x)) → t ∈ Set.Ioo (-∞) (-4) ∪ Set.Ioo 1 (∞) :=
sorry

end part1_part2_l309_309180


namespace sum_xyz_eq_two_l309_309532

-- Define the variables x, y, and z to be real numbers
variables (x y z : ℝ)

-- Given condition
def condition : Prop :=
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0

-- The theorem to prove
theorem sum_xyz_eq_two (h : condition x y z) : x + y + z = 2 :=
sorry

end sum_xyz_eq_two_l309_309532


namespace polygon_sides_eight_l309_309218

theorem polygon_sides_eight (n : ℕ) 
  (h₀ : ∑ (exterior_angles : 360)) 
  (h₁ : ∑ (interior_angles = 180 * (n - 2)) = 3 * ∑ (exterior_angles)) 
  : n = 8 := 
by 
  sorry

end polygon_sides_eight_l309_309218


namespace find_theta_l309_309120

theorem find_theta :
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ cos (15 * real.pi / 180) = sin (35 * real.pi / 180) + cos (θ * real.pi / 180) ↔ θ = 79 :=
by
  sorry

end find_theta_l309_309120


namespace grandmother_age_l309_309320

theorem grandmother_age 
  (avg_age : ℝ)
  (age1 age2 age3 grandma_age : ℝ)
  (h_avg_age : avg_age = 20)
  (h_ages : age1 = 5)
  (h_ages2 : age2 = 10)
  (h_ages3 : age3 = 13)
  (h_eq : (age1 + age2 + age3 + grandma_age) / 4 = avg_age) : 
  grandma_age = 52 := 
by
  sorry

end grandmother_age_l309_309320


namespace probability_at_least_one_female_is_five_sixths_l309_309150

-- Declare the total number of male and female students
def total_male_students := 6
def total_female_students := 4
def total_students := total_male_students + total_female_students
def selected_students := 3

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 3 students from 10 students
def total_ways_to_select_3 := binomial_coefficient total_students selected_students

-- Ways to select 3 male students from 6 male students
def ways_to_select_3_males := binomial_coefficient total_male_students selected_students

-- Probability of selecting at least one female student
def probability_of_at_least_one_female : ℚ := 1 - (ways_to_select_3_males / total_ways_to_select_3)

-- The theorem statement to be proved
theorem probability_at_least_one_female_is_five_sixths :
  probability_of_at_least_one_female = 5/6 := by
  sorry

end probability_at_least_one_female_is_five_sixths_l309_309150


namespace find_number_l309_309206

theorem find_number (x : ℤ) (h : 45 - (28 - (37 - (x - 18))) = 57) : x = 15 :=
by
  sorry

end find_number_l309_309206


namespace midpoint_parallel_l309_309680

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309680


namespace mn_parallel_pq_l309_309665

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309665


namespace find_n_ge_2018_l309_309172

noncomputable def seq_a (n : ℕ) : ℝ :=
  if n = 1 then 2 else 2 ^ n

noncomputable def seq_b (n : ℕ) : ℝ :=
  seq_a n * Real.cos (n * Real.pi)

noncomputable def sum_b (n : ℕ) : ℝ :=
  ∑ i in Finset.range (2 * n), seq_b (i + 1)

theorem find_n_ge_2018 : ∃ n : ℕ, 0 < n ∧ sum_b n ≥ 2018 := 
  sorry

end find_n_ge_2018_l309_309172


namespace problem_statement_l309_309972

theorem problem_statement (a : ℝ) :
  (∀ x : ℝ, (1/2 < x ∧ x < 2 → ax^2 + 5 * x - 2 > 0)) →
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) → ax^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end problem_statement_l309_309972


namespace possible_values_of_ab_plus_ac_plus_bc_l309_309263

-- Definitions and conditions
variables {a b c : ℝ} 

-- The main theorem statement
theorem possible_values_of_ab_plus_ac_plus_bc (h : a + b + c = 1) : 
  ∃ (S : set ℝ), S = (-∞, 1/2] ∧ (ab + ac + bc) ∈ S := 
sorry

end possible_values_of_ab_plus_ac_plus_bc_l309_309263


namespace largest_divisible_n_l309_309381

theorem largest_divisible_n (n : ℕ) :
  (n^3 + 2006) % (n + 26) = 0 → n = 15544 :=
sorry

end largest_divisible_n_l309_309381


namespace widely_spaced_subsets_15_l309_309079

-- Definition of "widely spaced" as a condition
def is_widely_spaced (s : set ℕ) : Prop :=
  ∀ (a b ∈ s), |a - b| ≥ 4

-- Define d recursively based on the problem's solution steps
def d : ℕ → ℕ
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 5
| n := d (n-1) + d (n-4)

-- State the theorem to be proved
theorem widely_spaced_subsets_15 : d 15 = 181 := 
sorry

end widely_spaced_subsets_15_l309_309079


namespace find_first_parallel_side_l309_309512

-- Definitions of the problem's conditions
def length_second_parallel_side : ℝ := 18
def distance_between_parallel_sides : ℝ := 13
def area_of_trapezium : ℝ := 247

-- The formula for the area of a trapezium
def trapezium_area (a b h : ℝ) : ℝ :=
  1/2 * (a + b) * h

-- The length of the first parallel side
def length_first_parallel_side : ℝ :=
  let x := 20
  x

-- The theorem statement of the equivalent proof problem
theorem find_first_parallel_side :
  trapezium_area length_first_parallel_side length_second_parallel_side distance_between_parallel_sides = area_of_trapezium :=
by
  sorry

end find_first_parallel_side_l309_309512


namespace tiffany_fastest_l309_309709

def average_speed (total_distance : ℕ) (total_time : ℕ) : ℚ := total_distance / total_time

def total_distance_tiffany : ℕ := 6 + 8 + 6
def total_time_tiffany : ℕ := 3 + 5 + 3
def average_speed_tiffany : ℚ := average_speed total_distance_tiffany total_time_tiffany

def total_distance_moses : ℕ := 5 + 10 + 5
def total_time_moses : ℕ := 5 + 10 + 4
def average_speed_moses : ℚ := average_speed total_distance_moses total_time_moses

def total_distance_morgan : ℕ := 7 + 9 + 4
def total_time_morgan : ℕ := 4 + 6 + 2
def average_speed_morgan : ℚ := average_speed total_distance_morgan total_time_morgan

theorem tiffany_fastest :
  average_speed_tiffany > average_speed_moses ∧
  average_speed_tiffany > average_speed_morgan :=
by
  unfold average_speed_tiffany average_speed_moses average_speed_morgan
  have tiffany_speed : average_speed 20 11 = 1.818 := sorry
  have moses_speed : average_speed 20 19 = 1.053 := sorry
  have morgan_speed : average_speed 20 12 = 1.667 := sorry
  have max_speed := max tiffany_speed (max moses_speed morgan_speed)
  have tiffany_fast : tiffany_speed = max_speed := sorry
  exact ⟨by linarith, by linarith⟩


end tiffany_fastest_l309_309709


namespace distance_from_F1_to_F2M_l309_309971

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2) / 6 - (y^2) / 3 = 1

-- Define the foci coordinates
def F1 : (ℝ × ℝ) := (-3, 0)
def F2 : (ℝ × ℝ) := (3, 0)

-- Define the condition that M lies on the hyperbola and MF1 is perpendicular to the x-axis
def onHyperbola (M : ℝ × ℝ) : Prop := hyperbola M.1 M.2
def perpendicularMF1 (M : ℝ × ℝ) : Prop := M.1 = 3

-- Define the distance function from a point to a line
def distanceFromPointToLine (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ :=
  sorry -- Distance computation implementation

-- Define line equation from points (F2 and M)
def line (F2 M : ℝ × ℝ) : (ℝ × ℝ) → Prop :=
  λ P, (P.2 - M.2) * (F2.1 - M.1) = (F2.2 - M.2) * (P.1 - M.1)

-- The Lean statement we want to prove
theorem distance_from_F1_to_F2M :
  ∀ (M : ℝ × ℝ), onHyperbola M → perpendicularMF1 M →
    distanceFromPointToLine F1 (line F2 M) = 6 / 5 :=
begin
  -- Proof goes here
  sorry
end

end distance_from_F1_to_F2M_l309_309971


namespace ways_to_put_5_balls_into_3_boxes_nonempty_l309_309587

def number_of_ways_to_put_balls_into_boxes (balls boxes : ℕ) (indistinguishable_balls indistinguishable_boxes nonempty_boxes : Prop) : ℕ :=
  if balls = 5 ∧ boxes = 3 ∧ indistinguishable_balls ∧ indistinguishable_boxes ∧ nonempty_boxes then 3 else sorry

theorem ways_to_put_5_balls_into_3_boxes_nonempty :
  number_of_ways_to_put_balls_into_boxes 5 3 true true true = 3 :=
by
  simp [number_of_ways_to_put_balls_into_boxes]
  sorry

end ways_to_put_5_balls_into_3_boxes_nonempty_l309_309587


namespace no_real_solutions_l309_309910

noncomputable def no_real_solution_equation (x : ℝ) : Prop :=
  let y := real.sqrt (real.sqrt x) in
  y = 15 / (8 - 2 * y)

theorem no_real_solutions :
  ∀ (x : ℝ), ¬ (no_real_solution_equation x) :=
by
  assume x,
  let y := real.sqrt (real.sqrt x),
  have h : y ^ 2 - 4 * y + 7.5 = 0,
  -- The quadratic discriminant part
  have d : (4) ^ 2 - 4 * (2) * (7.5) = -4 * 7.5,
  have neg := mul_neg_of_pos_of_neg (zero_lt_four) (neg_of_neg_pos _),
  -- Therefore, no real root exists for y, and consequently, for x
  sorry

end no_real_solutions_l309_309910


namespace tan_alpha_l309_309933

theorem tan_alpha (α β : ℝ) (h : 0 < α ∧ α < β ∧ β < π / 2) 
  (h1 : cos α * cos β + sin α * sin β = 4 / 5) 
  (h2 : tan β = 4 / 3) : 
  tan α = 7 / 24 :=
by
  sorry

end tan_alpha_l309_309933


namespace no_solutions_a_l309_309002

theorem no_solutions_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := 
sorry

end no_solutions_a_l309_309002


namespace smallest_digit_to_make_divisible_by_9_l309_309138

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l309_309138


namespace expected_value_equals_51_l309_309427

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l309_309427


namespace eccentricity_range_l309_309166

def ellipse (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a > b) :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def c_squared (a b : ℝ) : ℝ := a^2 - b^2

def perpendicular_condition (a b c x y : ℝ) : Prop :=
  (x - c, y).fst * (x + c, y).fst + (x - c, y).snd * (x + c, y).snd = 0

theorem eccentricity_range (a b e c x y : ℝ)
  (h : a > 0 ∧ b > 0 ∧ a > b)
  (h_ellipse : ellipse a b h)
  (h_perp : perpendicular_condition a b c x y) :
  (e = c / a ∧ 0 < e ∧ e < 1) →
  (√2 / 2 ≤ e ∧ e < 1) :=
by
  sorry

end eccentricity_range_l309_309166


namespace polygon_area_PQR_is_50_l309_309023

def point := (ℝ × ℝ × ℝ)

-- Define the vertices of the cube
def A : point := (0, 0, 0)
def B : point := (30, 0, 0)
def C : point := (30, 0, 30)
def D : point := (30, 30, 30)

-- Define the points on the edges of the cube
def P : point := (10, 0, 0)
def Q : point := (30, 0, 20)
def R : point := (30, 25, 30)

def plane (a b c d : ℝ) (x y z : ℝ) : Prop := 
  a * x + b * y + c * z = d

-- The actual statement related to the polygon intersection's area computation
theorem polygon_area_PQR_is_50 :
  let (x₀, y₀, z₀) := P in
  let (x₁, y₁, z₁) := Q in
  let (x₂, y₂, z₂) := R in

  ∃ a b c d : ℝ, 
  plane a b c d x₀ y₀ z₀ ∧
  plane a b c d x₁ y₁ z₁ ∧
  plane a b c d x₂ y₂ z₂ ∧
  
  -- Note: We assume the plane's intersection forming the polygon whose area we'll denote as 50
  ∃ A : ℝ, A = 50 := 
sorry

end polygon_area_PQR_is_50_l309_309023


namespace angle_AKB_eq_angle_DMC_l309_309347

theorem angle_AKB_eq_angle_DMC 
  (A B C D K M : Point)
  (tetrahedron : InscribedSphereInTetrahedron A B C D)
  (hK : TouchesSphereAtPoint tetrahedron.faceABD K)
  (hM : TouchesSphereAtPoint tetrahedron.faceDBC M) :
  angle A K B = angle D M C :=
sorry

end angle_AKB_eq_angle_DMC_l309_309347


namespace find_rate_of_interest_l309_309727

noncomputable def rate_of_interest (P SI : ℝ) : ℝ :=
  let R := real.sqrt (SI / (P / 100))
  R

theorem find_rate_of_interest (P : ℝ) (SI : ℝ) (T : ℝ) (R : ℝ) (h1 : P = 800) (h2 : SI = 632) (h3 : T = R) (h4 : SI = (P * R * T) / 100) :
  R = real.sqrt 79 :=
by
  sorry

end find_rate_of_interest_l309_309727


namespace vanya_exam_scores_l309_309399

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end vanya_exam_scores_l309_309399


namespace island_population_percentage_l309_309460

theorem island_population_percentage :
  -- Defining conditions
  (∀ a b : ℕ, (a + b ≠ 0) → (a.toRat / (a + b).toRat = 65 / 100) →
   ∀ b c : ℕ, (b + c ≠ 0) → (b.toRat / (b + c).toRat = 70 / 100) →
   ∀ c d : ℕ, (c + d ≠ 0) → (c.toRat / (c + d).toRat = 10 / 100) →
  
  -- Correct answer based on conditions
  ∃ a b c d : ℕ, 
    let total := a + b + c + d in 
    total ≠ 0 ∧ 
    (d.toRat / total.toRat = 54 / 100)) := 
sorry

end island_population_percentage_l309_309460


namespace pyramid_new_volume_l309_309046

theorem pyramid_new_volume (V : ℝ) (l w h : ℝ) 
  (hV : V = 60) 
  (hV_eq : V = (1/3) * l * w * h) :
  let l' := 3 * l,
      w' := 2 * w,
      h' := 2 * h,
      V' := (1/3) * l' * w' * h' in
  V' = 720 := 
by 
  sorry

end pyramid_new_volume_l309_309046


namespace ratio_of_volumes_l309_309480

-- Definitions based on given conditions
def V1 : ℝ := sorry -- Volume of the first vessel
def V2 : ℝ := sorry -- Volume of the second vessel

-- Given condition
def condition : Prop := (3 / 4) * V1 = (5 / 8) * V2

-- The theorem to prove the ratio V1 / V2 is 5 / 6
theorem ratio_of_volumes (h : condition) : V1 / V2 = 5 / 6 :=
sorry

end ratio_of_volumes_l309_309480


namespace loss_percentage_is_17_l309_309744

noncomputable def loss_percentage (CP SP : ℝ) := ((CP - SP) / CP) * 100

theorem loss_percentage_is_17 :
  let CP : ℝ := 1500
  let SP : ℝ := 1245
  loss_percentage CP SP = 17 :=
by
  sorry

end loss_percentage_is_17_l309_309744


namespace find_length_EC_l309_309558

-- Definitions of the conditions
def angle_A := 45 -- in degrees
def AC := 10 -- segment AC length in units
def BD_perpendicular_AC := true
def CE_perpendicular_AB := true
def angle_DBC_eq_2_angle_ECB (angle_ECB : ℝ) := 2 * angle_ECB

-- Final statement to prove
theorem find_length_EC :
  ∃ (a b c : ℕ) (EC : ℝ),
    m_angle_A = angle_A ∧
    segment_AC = AC ∧
    BD_perpendicular_AC ∧
    CE_perpendicular_AB ∧
    angle_DBC_eq_angle_ECB (m_angle_ECB) ∧
    EC = a * (sqrt b + sqrt c) ∧
    a + b + c = 11 :=
sorry

end find_length_EC_l309_309558


namespace triple_nesting_l309_309989

def f (x : ℕ) : ℕ := 3 * x^2 - 2 * x + 1

theorem triple_nesting (h : ∀ x, f x = 3 * x^2 - 2 * x + 1) : f (f (f 1)) = 226 := by
  rw [h 1, h (f 1), h (f (f 1))]
  -- The rw steps are checking the function evaluation steps at each level.
  sorry

end triple_nesting_l309_309989


namespace number_of_solutions_x_squared_equiv_x_l309_309268

theorem number_of_solutions_x_squared_equiv_x (n : ℕ) (h : n ≥ 2) :
  let k := (Nat.factors n).toFinset.card in
  {x ∈ Finset.range (n + 1) | x^2 % n = x % n}.card = 2 ^ k :=
begin
  sorry
end

end number_of_solutions_x_squared_equiv_x_l309_309268


namespace sum_arith_seq_S9_l309_309168

variable (a : ℕ → ℚ)
variable (d : ℚ)

-- Definition of an arithmetic sequence
def is_arith_seq : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def cond1 : is_arith_seq a d := sorry
def cond2 : a 2 + a 8 = 4 / 3 := sorry

-- Theorem to prove
theorem sum_arith_seq_S9 (h1 : is_arith_seq a d) (h2 : a 2 + a 8 = 4 / 3) :
  (∑ i in Finset.range 9, a i) = 6 := by
  sorry

end sum_arith_seq_S9_l309_309168


namespace packed_lunch_groups_l309_309053

theorem packed_lunch_groups 
  (students_per_group : ℕ)
  (sandwiches_per_student : ℕ)
  (total_pieces_bread : ℕ)
  (students_per_group = 6)
  (sandwiches_per_student = 2)
  (total_pieces_bread = 120) :
  (total_pieces_bread / (sandwiches_per_student * students_per_group)) = 5 :=
by
  sorry

end packed_lunch_groups_l309_309053


namespace roots_of_equation_l309_309916

-- Conditions
-- z is a complex number and satisfies the given equation
def is_root (z : ℂ) : Prop := z^3 + z^2 - z = 7 + 7i

-- The proof statement
theorem roots_of_equation : (is_root (4 + complex.i)) ∧ (is_root (-3 - complex.i)) :=
by sorry

end roots_of_equation_l309_309916


namespace parallel_MN_PQ_l309_309656

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309656


namespace min_value_f_l309_309128

noncomputable def f (x : ℝ) : ℝ := (Real.tan x)^2 - 4 * (Real.tan x) - 12 * (Real.cot x) + 9 * (Real.cot x)^2 - 3

theorem min_value_f : ∃ x ∈ Ioo (-Real.pi / 2) 0, ∀ y ∈ Ioo (-Real.pi / 2) 0, f(x) ≤ f(y) ∧ f(x) = 3 + 8 * Real.sqrt(3) := by
  sorry

end min_value_f_l309_309128


namespace pentagon_angle_proof_l309_309224

variable (A B C D E P : Type)
variables [IsConvexPentagon A B C D E] -- Assuming we have a type class for convex pentagon
variables (AE_parallel_BC : IsParallel AE BC)
variables (angle_equality_1 : MeasureAngle ADE = MeasureAngle BDC)
variables (intersection : IntersectsAt AC BE P)

theorem pentagon_angle_proof :
  MeasureAngle EAD = MeasureAngle BDP ∧ MeasureAngle CBD = MeasureAngle ADP := by
  sorry

end pentagon_angle_proof_l309_309224


namespace polar_eq_curve_C_length_segment_AB_l309_309615

noncomputable def parametric_eq_curve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 2 + 2 * Real.sin θ)

noncomputable def polar_eq_curve (θ : ℝ) : ℝ :=
  4 * Real.sin θ

def point_M_cartesian : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos (Real.pi / 4), Real.sqrt 2 * Real.sin (Real.pi / 4))

noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, 1 + t * Real.sin α)

-- For the polar equation of curve C
theorem polar_eq_curve_C (θ : ℝ) : 
  ∃ (ρ : ℝ), parametric_eq_curve θ = (ρ * Real.cos θ, ρ * Real.sin θ + 2) ∧ ρ = polar_eq_curve θ :=
sorry

-- For the length of segment AB, given point M and the parametric equation of curve C
theorem length_segment_AB (α : ℝ) (t1 t2 : ℝ)
    (h1 : parametric_eq_curve (Real.atan2 1 1) =
        (line_l t1 α).fst + (parametric_eq_curve (Real.atan2 1 1)).fst,
        (line_l t1 α).snd)
    (h2 : parametric_eq_curve (Real.atan2 1 1) =
        (line_l t2 α).fst + (parametric_eq_curve (Real.atan2 1 1)).fst,
        (line_l t2 α).snd)
    (ht : (t1 * t2 = -2) ∧ (t1 = -2 * t2 ∨ t1 = 2 * t2)) :
  |t1 - t2| = 3 :=
sorry

end polar_eq_curve_C_length_segment_AB_l309_309615


namespace book_pages_l309_309891

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
  pages_per_day = 8 → days = 12 → total_pages = pages_per_day * days → total_pages = 96 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end book_pages_l309_309891


namespace expected_value_8_sided_die_l309_309414

-- Define the roll outcomes and their associated probabilities
def roll_outcome (n : ℕ) : ℕ := 2 * n^2

-- Define the expected value calculation
def expected_value (sides : ℕ) : ℚ := ∑ i in range (1, sides+1), (1 / sides) * roll_outcome i

-- Prove the expected value calculation for an 8-sided fair die
theorem expected_value_8_sided_die : expected_value 8 = 51 := by
  sorry

end expected_value_8_sided_die_l309_309414


namespace max_correct_answers_l309_309226

-- Definitions based on the conditions
def total_problems : ℕ := 12
def points_per_correct : ℕ := 6
def points_per_incorrect : ℕ := 3
def max_score : ℤ := 37 -- Final score, using ℤ to handle potential negatives in deducting points

-- The statement to prove
theorem max_correct_answers :
  ∃ (c w : ℕ), c + w = total_problems ∧ points_per_correct * c - points_per_incorrect * (total_problems - c) = max_score ∧ c = 8 :=
by
  sorry

end max_correct_answers_l309_309226


namespace hyperbola_a_h_l309_309771

noncomputable theory
open Real

theorem hyperbola_a_h (a b h k : ℝ) 
  (ha : a = 6 * sqrt 2)
  (hb : b = sqrt (8 / 3))
  (hh : h = -2 / 3)
  (hk : k = 4)
  (pasymp1 : ∀ x y, y = 3 * x + 6 → y = -3 * x + 2 → x = -2 / 3 ∧ y = 4)
  (pcond : (1 - -2/3) ^ 2 = 8 / 3): 
  a + h = 6 * sqrt 2 - 2 / 3 :=
by {
  sorry
}

end hyperbola_a_h_l309_309771


namespace limit_of_hours_for_overtime_l309_309408

theorem limit_of_hours_for_overtime
  (R : Real) (O : Real) (total_compensation : Real) (total_hours_worked : Real) (L : Real)
  (hR : R = 14)
  (hO : O = 1.75 * R)
  (hTotalCompensation : total_compensation = 998)
  (hTotalHoursWorked : total_hours_worked = 57.88)
  (hEquation : (R * L) + ((total_hours_worked - L) * O) = total_compensation) :
  L = 40 := 
  sorry

end limit_of_hours_for_overtime_l309_309408


namespace new_outsiders_count_l309_309041

theorem new_outsiders_count (total_people: ℕ) (initial_snackers: ℕ)
  (first_group_outsiders: ℕ) (first_group_leave_half: ℕ) 
  (second_group_leave_count: ℕ) (half_remaining_leave: ℕ) (final_snackers: ℕ) 
  (total_snack_eaters: ℕ) 
  (initial_snackers_eq: total_people = 200) 
  (snackers_eq: initial_snackers = 100) 
  (first_group_outsiders_eq: first_group_outsiders = 20) 
  (first_group_leave_half_eq: first_group_leave_half = 60) 
  (second_group_leave_count_eq: second_group_leave_count = 30) 
  (half_remaining_leave_eq: half_remaining_leave = 15) 
  (final_snackers_eq: final_snackers = 20) 
  (total_snack_eaters_eq: total_snack_eaters = 120): 
  (60 - (second_group_leave_count + half_remaining_leave + final_snackers)) = 40 := 
by sorry

end new_outsiders_count_l309_309041


namespace isabel_bouquets_l309_309244

theorem isabel_bouquets (initial_flowers wilted_flowers flowers_per_bouquet : ℕ) 
  (initial_flowers = 66) (wilted_flowers = 10) (flowers_per_bouquet = 8) :
  (initial_flowers - wilted_flowers) / flowers_per_bouquet = 7 :=
by
  sorry

end isabel_bouquets_l309_309244


namespace find_f_neg_one_l309_309994

variable (f : ℝ → ℝ)

-- Given condition
axiom func_property : ∀ x : ℝ, f(x - 1) = x^2 + 1

-- Theorem to prove
theorem find_f_neg_one : f(-1) = 1 :=
by
  -- Proof goes here
  sorry

end find_f_neg_one_l309_309994


namespace side_length_of_square_l309_309762

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l309_309762


namespace david_cups_consumed_l309_309291

noncomputable def cups_of_water (time_in_minutes : ℕ) : ℝ :=
  time_in_minutes / 20

theorem david_cups_consumed : cups_of_water 225 = 11.25 := by
  sorry

end david_cups_consumed_l309_309291


namespace lottery_selection_count_l309_309711

theorem lottery_selection_count :
  let numbers := {n : ℕ | n ≤ 90 ∧ n % 2 = 0}
  let divisible_by_4 := {n ∈ numbers | n % 4 = 0}
  let divisible_by_8 := {n ∈ divisible_by_4 | n % 8 = 0}
  let divisible_by_16 := {n ∈ divisible_by_8 | n % 16 = 0}
  ∃ (count : ℕ), count = 15180 ∧
    ((numbers : Set ℕ).card = 5 ∧
     (divisible_by_16 : Set ℕ).card = 2 ∧
     (divisible_by_8 : Set ℕ).card = 3 ∧
     (divisible_by_4 : Set ℕ).card = 4) :=
by
  sorry

end lottery_selection_count_l309_309711


namespace time_worked_together_l309_309825

noncomputable def combined_rate (P_rate Q_rate : ℝ) : ℝ :=
  P_rate + Q_rate

theorem time_worked_together (P_rate Q_rate : ℝ) (t additional_time job_completed : ℝ) :
  P_rate = 1 / 4 ∧ Q_rate = 1 / 15 ∧ additional_time = 1 / 5 ∧ job_completed = (additional_time * P_rate) →
  (t * combined_rate P_rate Q_rate + job_completed = 1) → 
  t = 3 :=
sorry

end time_worked_together_l309_309825


namespace problem_part1_problem_part2_problem_part3_l309_309159

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem problem_part1 : f 1 = 5 / 2 ∧ f 2 = 17 / 4 := 
by
  sorry

theorem problem_part2 : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem problem_part3 : ∀ x1 x2 : ℝ, x1 < x2 → x1 < 0 → x2 < 0 → f x1 > f x2 :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l309_309159


namespace baseball_card_distribution_l309_309717

theorem baseball_card_distribution (total_cards : ℕ) (capacity_4 : ℕ) (capacity_6 : ℕ) (capacity_8 : ℕ) :
  total_cards = 137 →
  capacity_4 = 4 →
  capacity_6 = 6 →
  capacity_8 = 8 →
  (total_cards % capacity_4) % capacity_6 = 1 :=
by
  intros
  sorry

end baseball_card_distribution_l309_309717


namespace find_k_l309_309919

theorem find_k (k : ℚ) (h : ∀ x : ℚ, 4 * x^2 + 15 * x + k = 0 → x = (−15 - real.sqrt 165) / 8) : 
  k = 15 / 4 :=
by
  sorry

end find_k_l309_309919


namespace largest_number_1_4_5_l309_309372

/--
Given the digits 1, 4, and 5, the largest number you can create using each of the digits exactly once is 541.
-/
theorem largest_number_1_4_5 :
  ∃ (n : ℕ), (n = 541) ∧ (∃ l : List ℕ, l = [1, 4, 5] ∧ (∀ d ∈ l, (∃ m : ℕ, to_digits 10 n = [d]))) := sorry

end largest_number_1_4_5_l309_309372


namespace wire_length_ratio_l309_309069

open Real

noncomputable def bonnie_wire_length : ℝ := 12 * 8
noncomputable def bonnie_cube_volume : ℝ := 8^3
noncomputable def roark_unit_cube_volume : ℝ := 2^3
noncomputable def roark_number_of_cubes : ℝ := bonnie_cube_volume / roark_unit_cube_volume
noncomputable def roark_wire_length_per_cube : ℝ := 12 * 2
noncomputable def roark_total_wire_length : ℝ := roark_number_of_cubes * roark_wire_length_per_cube
noncomputable def bonnie_to_roark_wire_ratio := bonnie_wire_length / roark_total_wire_length

theorem wire_length_ratio : bonnie_to_roark_wire_ratio = (1 : ℝ) / 16 :=
by
  sorry

end wire_length_ratio_l309_309069


namespace intersection_complement_l309_309189

open Set

-- Definitions from the problem
def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {y | 0 < y}

-- The proof statement
theorem intersection_complement : A ∩ (compl B) = Ioc (-1 : ℝ) 0 := by
  sorry

end intersection_complement_l309_309189


namespace count_8_digit_integers_with_consecutive_ones_l309_309583

theorem count_8_digit_integers_with_consecutive_ones : 
  let total_8_digit := 2^8
  let a : ℕ → ℕ 
  let a 1 := 2
  let a 2 := 3
  let a 3 := a 2 + a 1
  let a 4 := a 3 + a 2
  let a 5 := a 4 + a 3
  let a 6 := a 5 + a 4
  let a 7 := a 6 + a 5
  let a 8 := a 7 + a 6
  total_8_digit - a 8 = 201 := 
by 
  sorry

end count_8_digit_integers_with_consecutive_ones_l309_309583


namespace sammy_math_homework_pages_l309_309781

theorem sammy_math_homework_pages (total_pages pages_remaining : ℕ) (percent_science_project : ℚ) :
  total_pages = 120 →
  pages_remaining = 80 →
  percent_science_project = 0.25 →
  let pages_science_project := percent_science_project * total_pages in
  let pages_before_math_homework := total_pages - nat.floor pages_science_project in
  pages_before_math_homework - pages_remaining = 10 :=
by
  intros h_total_pages h_pages_remaining h_percent_science_project
  let pages_science_project := h_percent_science_project * total_pages
  let pages_before_math_homework := total_pages - nat.floor pages_science_project
  have calc_science_project : pages_science_project = 30 := by sorry
  have pages_before_math := pages_before_math_homework = 90 := by sorry
  have num_pages_math_homework := pages_before_math_homework - pages_remaining
  show num_pages_math_homework = 10
  sorry

end sammy_math_homework_pages_l309_309781


namespace integral_eq_square_iff_t_eq_two_l309_309156

theorem integral_eq_square_iff_t_eq_two (t : ℝ) (h1 : t > 1) (h2 : ∫ x in 1..t, (2*x + 1) = t^2) : t = 2 :=
sorry

end integral_eq_square_iff_t_eq_two_l309_309156


namespace midpoint_parallel_l309_309675

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l309_309675


namespace sector_radius_of_circle_l309_309740

theorem sector_radius_of_circle :
  ∃ r : ℝ, let θ := 42 * Real.pi / 180 in
  let area := 82.5 in
  θ / (2 * Real.pi) * Real.pi * r^2 = area ∧ r = 15 :=
begin
  sorry, -- The proof is omitted.
end

end sector_radius_of_circle_l309_309740


namespace solution_set_inequality_l309_309104

theorem solution_set_inequality (x : ℝ) (h : x > 0) : 
  (|2 * x - log 2 x| < 2 * x + |log 2 x|) ↔ (x > 1) := 
sorry

end solution_set_inequality_l309_309104


namespace quadratic_range_l309_309789

theorem quadratic_range (x y : ℝ) 
    (h1 : y = (x - 1)^2 + 1)
    (h2 : 2 ≤ y ∧ y < 5) : 
    (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) :=
by
  sorry

end quadratic_range_l309_309789


namespace sum_of_lengths_of_parallel_segments_l309_309726

noncomputable def sum_segments (n : ℕ) (len_JK len_LM : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (len_JK^2 + len_LM^2)
  let segment_length (k : ℕ) := hypotenuse * ((n - k) / n)
  2 * (∑ k in Finset.range n, segment_length k) - hypotenuse

theorem sum_of_lengths_of_parallel_segments :
  sum_segments 200 5 12 = 1280 := by
  sorry

end sum_of_lengths_of_parallel_segments_l309_309726


namespace price_of_basketball_l309_309886

-- Problem definitions based on conditions
def price_of_soccer_ball (x : ℝ) : Prop :=
  let price_of_basketball := 2 * x
  x + price_of_basketball = 186

theorem price_of_basketball (x : ℝ) (h : price_of_soccer_ball x) : 2 * x = 124 :=
by
  sorry

end price_of_basketball_l309_309886


namespace side_length_of_square_l309_309748

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l309_309748


namespace maria_earnings_over_three_days_l309_309704

-- Define the earnings calculations
def day1_earnings : ℝ := 
  let tulips := 30 * 2 in
  let roses := 20 * 3 in
  let lilies := 15 * 4 in
  let sunflowers := 10 * 5 in
  tulips + roses + lilies + sunflowers

def day2_earnings : ℝ :=
  let tulips := 30 * 2 * 2 in
  let roses := 20 * 2 * 3 * 0.8 in
  let lilies := 15 * 4 * 1.25 in
  let sunflowers := 10 * 3 * 5 in
  tulips + roses + lilies + sunflowers

def day3_earnings : ℝ :=
  let tulips := (30 * 2 * 0.1) * 2 * 0.85 in
  let roses := (16 / 3) * 2 * 3 in
  let lilies := (15 / 2) * 4 * 1.1 in
  let sunflowers := 10 * 3 * 5 in
  tulips + roses + lilies + sunflowers

def total_earnings : ℝ :=
  day1_earnings + day2_earnings + day3_earnings

theorem maria_earnings_over_three_days :
  total_earnings = 896.20 :=
by
  unfold total_earnings day1_earnings day2_earnings day3_earnings
  norm_num
  sorry


end maria_earnings_over_three_days_l309_309704


namespace ratio_of_unit_prices_l309_309070

def volume_y (v : ℝ) : ℝ := v
def price_y (p : ℝ) : ℝ := p
def volume_x (v : ℝ) : ℝ := 1.3 * v
def price_x (p : ℝ) : ℝ := 0.8 * p

theorem ratio_of_unit_prices (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (0.8 * p / (1.3 * v)) / (p / v) = 8 / 13 :=
by 
  sorry

end ratio_of_unit_prices_l309_309070


namespace cos_difference_identity_l309_309936

theorem cos_difference_identity (α : ℝ)
  (h : Real.sin (α + π / 6) + Real.cos α = - (Real.sqrt 3) / 3) :
  Real.cos (π / 6 - α) = -1 / 3 := 
sorry

end cos_difference_identity_l309_309936


namespace smallest_digit_divisible_by_9_l309_309135

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l309_309135


namespace sqrt_repeating_digits_l309_309269

theorem sqrt_repeating_digits (n : ℕ) (hn : 0 < n) :
  let a := (10^(2 * n) - 1) / 9
  let b := 2 * (10^n - 1) / 9 in
  sqrt (a - b) = (10^n - 1) / 3 := 
sorry

end sqrt_repeating_digits_l309_309269


namespace isosceles_triangle_AXY_l309_309273

-- Define the problem setup
variables {A B C X Y : Type*}
variables [triangle : Triangle A B C]
variable (Gamma : Circle A B C)
variable (X : FootAngleBisector A B C)
variable (Y : IntersectionLineTangent BC (TangentAt A Gamma))

-- State the theorem to prove
theorem isosceles_triangle_AXY : distance A Y = distance X Y := 
sorry

end isosceles_triangle_AXY_l309_309273


namespace eleventh_number_is_166_l309_309454

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def digits_sum_13 (n : ℕ) : Prop :=
  sum_of_digits n = 13

def increasing_list_numbers := list.Icc 1 9999 |>.filter digits_sum_13

def num_at_index_11 := increasing_list_numbers.nth 10  -- lists are 0-indexed

theorem eleventh_number_is_166 :
  num_at_index_11 = some 166 :=
by
  -- Skip proof
  sorry

end eleventh_number_is_166_l309_309454


namespace find_m_for_binomial_square_l309_309386

theorem find_m_for_binomial_square (m : ℝ) : (∃ a : ℝ, (λ x : ℝ, x^2 - 20 * x + m) = (λ x : ℝ, (x + a)^2)) → m = 100 :=
by
  intro h
  sorry

end find_m_for_binomial_square_l309_309386


namespace least_addition_to_divisibility_l309_309988

theorem least_addition_to_divisibility (x : Nat) (h₁ : x = 821562) : 
  (∃ k : Nat, (x + k) % 5 = 0 ∧ (x + k) % 13 = 0 ∧ k = 8) :=
by
  let k := 8
  have h₃ : (x + k) % 5 = 0 := by sorry
  have h₄ : (x + k) % 13 = 0 := by sorry
  exists k
  constructor
  exact h₃
  constructor
  exact h₄
  rfl

end least_addition_to_divisibility_l309_309988


namespace new_girl_weight_l309_309321

variable (W : ℝ) -- total weight of the 20 girls before the new girl arrives
variable (avg_weight : W / 20)
variable (weight_dec : 40)
variable (new_avg_inc : 2)
variable (new_w : ℝ) -- weight of the new girl

theorem new_girl_weight :
  (W - weight_dec + new_w) / 20 = avg_weight + new_avg_inc →
  new_w = 80 :=
by
  intros h
  sorry

end new_girl_weight_l309_309321


namespace probability_other_side_red_l309_309405

theorem probability_other_side_red
  (cards : Finset (Finset (Fin 2)))
  (h1 : cards.card = 10)
  (black_black_cards : cards.filter (λ c, c = {0})) -- c = {0} means two sides black
  (black_red_cards : cards.filter (λ c, c = {0, 1})) -- c = {0, 1} means one side black, one side red
  (red_red_cards : cards.filter (λ c, c = {1})) -- c = {1} means two sides red
  (h2 : black_black_cards.card = 4)
  (h3 : black_red_cards.card = 3)
  (h4 : red_red_cards.card = 3) :
  (red_red_cards.card * 2 : ℚ) / (red_red_cards.card * 2 + black_red_cards.card) = 2 / 3 :=
by
  sorry

end probability_other_side_red_l309_309405


namespace max_integer_solutions_eq_six_l309_309436

def half_centered (p : ℤ[X]) : Prop :=
  p.coeff (nat.of_num 50) = 50

theorem max_integer_solutions_eq_six (p : ℤ[X]) (h : half_centered p) :
  ∃ n, (∀ k : ℤ, p.eval k = k^2 → k ∈ finset.range n) ∧ (n ≤ 6) :=
sorry

end max_integer_solutions_eq_six_l309_309436


namespace impossible_arrangement_l309_309013

theorem impossible_arrangement : 
  ∀ (a : Fin 111 → ℕ), (∀ i, a i ≤ 500) → (∀ i j, i ≠ j → a i ≠ a j) → 
  ¬ ∀ i : Fin 111, (a i % 10 = ((Finset.univ.sum (λ j => if j = i then 0 else a j)) % 10)) :=
by 
  sorry

end impossible_arrangement_l309_309013


namespace longest_side_length_l309_309868

open Real

theorem longest_side_length (A B C : ℝ × ℝ)
  (hA : A = (1, 1))
  (hB : B = (4, 5))
  (hC : C = (5, 1))
: ∃ (longest : ℝ), longest = 5 :=
by
  let d_AB := dist A B
  let d_A := dist A C
  let d_B := dist B C
  have h1 : d_AB = real.sqrt ((4 - 1) ^ 2 + (5 - 1) ^ 2) := by sorry
  have h2 : d_A = real.sqrt ((5 - 1) ^ 2 + (1 - 1) ^ 2)  := by sorry
  have h3 : d_B = real.sqrt ((5 - 4) ^ 2 + (1 - 5) ^ 2)  := by sorry
  have max_side := max d_AB (max d_A d_B)
  use max_side
  have h : max_side = 5 := by sorry
  assumption

end longest_side_length_l309_309868


namespace inequality_solution_l309_309108

theorem inequality_solution (x : ℝ) :
  (2 * x^2 - 4 * x - 70 > 0) ∧ (x ≠ -2) ∧ (x ≠ 0) ↔ (x < -5 ∨ x > 7) :=
by
  sorry

end inequality_solution_l309_309108


namespace fx_in_3_to_5_l309_309686

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the actual function definition.

-- The conditions of the problem
axiom condition1 : ∀ x : ℝ, f(x) + f(x + 2) = 0
axiom condition2 : ∀ x : ℝ, x ∈ set.Ioc (-1 : ℝ) 1 → f(x) = 2*x + 1

-- The statement we want to prove
theorem fx_in_3_to_5 : ∀ x : ℝ, x ∈ set.Ioc 3 5 → f(x) = 2*x - 7 :=
by
  intros x hx
  sorry -- Placeholder for the actual proof.

end fx_in_3_to_5_l309_309686


namespace car_truck_meet_l309_309881

theorem car_truck_meet (S x y : ℝ) (h₁ : 18 * (x + y) = 0.75 * x * (x + y)) : 
  k = 8 :=
by
  have h2 : (x * y) / (x + y) = 24, from sorry -- derived from h₁
  exact sorry -- complete the step to show k = 8 as derived from h2

end car_truck_meet_l309_309881


namespace smallest_n_for_sqrt_18n_integer_l309_309949

theorem smallest_n_for_sqrt_18n_integer :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (∃ k : ℕ, k^2 = 18 * m) → n <= m) ∧ (∃ k : ℕ, k^2 = 18 * n) :=
sorry

end smallest_n_for_sqrt_18n_integer_l309_309949


namespace mn_parallel_pq_l309_309650

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309650


namespace percentage_first_question_l309_309208

variable (P Q : Type)

theorem percentage_first_question
  (B : ℝ) (Neither : ℝ) (Both : ℝ) (HB : B = 0.55) (HNeither : Neither = 0.20) (HBoth : Both = 0.50) :
  ∃ A : ℝ, A = 0.75 :=
by
  -- Define the variables for the different percentages
  let AtLeastOne := (A + B - Both)
  have eq1 : AtLeastOne + Neither = 1 := by sorry -- equation: AtLeastOne + Neither = 100%
  have eq2 : AtLeastOne = A + B - Both := by sorry -- equation of inclusion-exclusion
  let A := 0.75
  use A
  sorry

end percentage_first_question_l309_309208


namespace systematic_sampling_correct_l309_309158

-- Define systematic_sampling
def systematic_sampling (n_products : ℕ) (n_selected : ℕ) (start : ℕ) (interval : ℕ) : list ℕ :=
(list.range n_selected).map (λ i, start + i * interval)

-- Define the problem parameters
def num_products : ℕ := 60
def num_selected : ℕ := 6
def initial_product : ℕ := 3
def sampling_interval : ℕ := 10
def expected_sequence : list ℕ := [3, 13, 23, 33, 43, 53]

-- State the theorem
theorem systematic_sampling_correct : 
  systematic_sampling num_products num_selected initial_product sampling_interval = expected_sequence :=
sorry

end systematic_sampling_correct_l309_309158


namespace trig_identity_proof_l309_309937

theorem trig_identity_proof (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) :
  Real.sin (2 * α - π / 6) + Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end trig_identity_proof_l309_309937


namespace number_increased_by_six_l309_309730

theorem number_increased_by_six :
  ∀ (S : Finset ℝ), S.card = 10 → (S.sum / 10 = 6.2) →
  (∃ x ∈ S, (S.erase x).sum + (x + 6) / 10 = 6.8) →
  6 = 6 :=
by
  intros S hc h_avg h_new_avg
  sorry

end number_increased_by_six_l309_309730


namespace opposite_of_2_is_minus_2_l309_309784

-- Define the opposite function
def opposite (x : ℤ) : ℤ := -x

-- Assert the theorem to prove that the opposite of 2 is -2
theorem opposite_of_2_is_minus_2 : opposite 2 = -2 := by
  sorry -- Placeholder for the proof

end opposite_of_2_is_minus_2_l309_309784


namespace mn_parallel_pq_l309_309661

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309661


namespace truncated_cone_sphere_radius_l309_309054

noncomputable def radius_of_sphere (r1 r2 h : ℝ) : ℝ := 
  (Real.sqrt (h^2 + (r1 - r2)^2)) / 2

theorem truncated_cone_sphere_radius : 
  ∀ (r1 r2 h : ℝ), r1 = 20 → r2 = 6 → h = 15 → radius_of_sphere r1 r2 h = Real.sqrt 421 / 2 :=
by
  intros r1 r2 h h1 h2 h3
  simp [radius_of_sphere]
  rw [h1, h2, h3]
  sorry

end truncated_cone_sphere_radius_l309_309054


namespace smallest_digit_divisible_by_9_l309_309136

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l309_309136


namespace simplify_expression_value_at_3_value_at_4_l309_309734

-- Define the original expression
def original_expr (x : ℕ) : ℚ := (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2 * x + 1))

-- Property 1: Simplify the expression
theorem simplify_expression (x : ℕ) (h1 : x ≠ 1) (h2 : x ≠ 2) : 
  original_expr x = (x - 1) / (x + 2) :=
sorry

-- Property 2: Evaluate the expression at x = 3
theorem value_at_3 : original_expr 3 = 2 / 5 :=
sorry

-- Property 3: Evaluate the expression at x = 4
theorem value_at_4 : original_expr 4 = 1 / 2 :=
sorry

end simplify_expression_value_at_3_value_at_4_l309_309734


namespace particle_10_steps_distance_4_l309_309434

def num_ways_10_steps_4_distance : ℕ := 240

theorem particle_10_steps_distance_4 :
  ∃ n : ℕ, n = 10 ∧ 
  ∃ k : ℕ, k = 4 ∧ 
  (∑ (i in {0, 4} : finset ℕ, nat.choose 10 (5 + i / 2)) = num_ways_10_steps_4_distance) :=
by {
  sorry
}

end particle_10_steps_distance_4_l309_309434


namespace midpoint_parallel_l309_309670

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309670


namespace day_of_week_after_100_days_l309_309482

theorem day_of_week_after_100_days
  (h : ∃ (n : ℕ), n % 7 = 2) 
  : "Chris' birthday is on a Tuesday" -> "The day of the week 100 days after will be Thursday" := 
sorry

end day_of_week_after_100_days_l309_309482


namespace min_log_div_is_zero_l309_309260

noncomputable def log_div (a b : ℝ) := |Real.log (a + b) / Real.log a| + |Real.log (b + a) / Real.log b|

theorem min_log_div_is_zero (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (∀ a b, 0 < a → 0 < b → a ≠ b → ∃ x, x = 0 ∧ ∀ y, log_div a b ≤ y :=
begin
  sorry
end

end min_log_div_is_zero_l309_309260


namespace product_of_a_l309_309101

theorem product_of_a : 
  (∃ a b : ℝ, (3 * a - 5)^2 + (a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2 ∧ 
    (a * b = -8.32)) :=
by 
  sorry

end product_of_a_l309_309101


namespace smallest_digit_divisible_by_9_l309_309142

theorem smallest_digit_divisible_by_9 : 
  ∃ (d : ℕ), 0 ≤ d ∧ d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ d = 2 :=
by
  use 2
  split
  { exact nat.zero_le _ }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end smallest_digit_divisible_by_9_l309_309142


namespace smallest_n_l309_309085

theorem smallest_n
    (f : ℕ → ℝ)
    (log2 : ℝ → ℝ)
    (hlog : ∀ x, log2 x = Real.log x / Real.log 2)
    (S : ℕ → ℝ)
    (hS : ∀ n, S n = ∑ k in Finset.range (n+1), log2 (1 + 1 / 2^(2^k))) :
  ∃ n, S n ≥ 1 + log2 (2014 / 2015) ∧
       (∀ m, S m ≥ 1 + log2 (2014 / 2015) → n ≤ m) :=
by
    sorry

end smallest_n_l309_309085


namespace smallest_digit_divisible_by_9_l309_309137

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l309_309137


namespace side_length_of_square_l309_309766

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309766


namespace sandy_bought_6_books_l309_309304

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_bought_6_books_l309_309304


namespace graph_of_abs_g_l309_309490

def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then -3 - x
  else if -1 ≤ x ∧ x ≤ 1 then real.sqrt (9 - (x + 1)^2) - 3
  else if 1 ≤ x ∧ x ≤ 2 then 3 * (x - 1)
  else 0

theorem graph_of_abs_g :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 2 → 
  (if x < -1 then  |g(x)| = 3 + x
   else if -1 ≤ x ∧ x ≤ 1 then |g(x)| = real.sqrt (9 - (x + 1)^2) - 3
   else if 1 ≤ x then |g(x)| = 3 * (x - 1)) :=
by {
  intro x,
  intros h,
  by_cases (-4 ≤ x ∧ x ≤ -1),
  { simp [g, h_1], sorry },
  by_cases (-1 ≤ x ∧ x ≤ 1),
  { simp [g, h_1], sorry },
  by_cases (1 ≤ x ∧ x ≤ 2),
  { simp [g, h_1], sorry },
  { simp [g, lt_irrefl] } 
}

end graph_of_abs_g_l309_309490


namespace side_length_of_square_l309_309767

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309767


namespace deposit_correct_l309_309731

-- Define the conditions
def monthly_income : ℝ := 10000
def deposit_percentage : ℝ := 0.25

-- Define the deposit calculation based on the conditions
def deposit_amount (income : ℝ) (percentage : ℝ) : ℝ :=
  percentage * income

-- Theorem: Prove that the deposit amount is Rs. 2500
theorem deposit_correct :
    deposit_amount monthly_income deposit_percentage = 2500 :=
  sorry

end deposit_correct_l309_309731


namespace mn_parallel_pq_l309_309666

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309666


namespace compute_x_l309_309086

theorem compute_x :
  (∑ n : ℕ, (1 / 3 : ℝ) ^ n) * (∑ n : ℕ, (-1 / 3 : ℝ) ^ n) = ∑ n : ℕ, (1 / (9 : ℝ)) ^ n :=
begin
  sorry
end

end compute_x_l309_309086


namespace mn_parallel_pq_l309_309662

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309662


namespace probability_transform_in_R_l309_309855

noncomputable def region_R : Set ℂ :=
  {z : ℂ | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

noncomputable def transform (z : ℂ) : ℂ :=
  (2/3) * (z.re - z.im) + (2/3) * complex.I * (z.re + z.im)

theorem probability_transform_in_R : 
  let z := uniform_sampling region_R in
  (∃ z ∈ region_R, transform z ∈ region_R) := 
begin
  -- proof outline:
  -- 1. show that for z in region_R, transform z is also in region_R
  -- 2. conclude the probability is 1
  sorry
end

end probability_transform_in_R_l309_309855


namespace find_angle_between_vectors_l309_309978

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def magnitude_a : ℝ := ‖a‖ = 1
def magnitude_b : ℝ := ‖b‖ = real.sqrt 2
def magnitude_a_minus_2b : ℝ := ‖a - (2 : ℝ) • b‖ = real.sqrt 13

-- The proof statement.
theorem find_angle_between_vectors :
  ∀ (a b : EuclideanSpace ℝ (Fin 3)), 
  magnitude_a → magnitude_b → magnitude_a_minus_2b → (∃ θ : ℝ, θ = (3 * real.pi) / 4) :=
by
  sorry

end find_angle_between_vectors_l309_309978


namespace log_function_fixed_point_l309_309573

theorem log_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  ∃ x y : ℝ, (x = 2) ∧ (y = 1) ∧ (y = 1 + log a (x - 1)) :=
by
  use [2, 1]
  split
  · rfl
  split
  · rfl
  sorry

end log_function_fixed_point_l309_309573


namespace factorize_expression_l309_309906

theorem factorize_expression (a x y : ℤ) : a * x - a * y = a * (x - y) :=
  sorry

end factorize_expression_l309_309906


namespace jill_total_tax_percentage_l309_309290

theorem jill_total_tax_percentage (spent_clothing_percent spent_food_percent spent_other_percent tax_clothing_percent tax_food_percent tax_other_percent : ℝ)
  (h1 : spent_clothing_percent = 0.5)
  (h2 : spent_food_percent = 0.25)
  (h3 : spent_other_percent = 0.25)
  (h4 : tax_clothing_percent = 0.1)
  (h5 : tax_food_percent = 0)
  (h6 : tax_other_percent = 0.2) :
  ((spent_clothing_percent * tax_clothing_percent + spent_food_percent * tax_food_percent + spent_other_percent * tax_other_percent) * 100) = 10 :=
by
  sorry

end jill_total_tax_percentage_l309_309290


namespace solve_real_numbers_l309_309498

theorem solve_real_numbers (x y : ℝ) :
  (x = 3 * x^2 * y - y^3) ∧ (y = x^3 - 3 * x * y^2) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 + Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = (Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 + Real.sqrt 2)) / 2)) :=
by
  sorry

end solve_real_numbers_l309_309498


namespace impossible_arrangement_l309_309014

-- Definitions for the problem
def within_range (n : ℕ) : Prop := n > 0 ∧ n ≤ 500
def distinct (l : List ℕ) : Prop := l.Nodup

-- The main problem statement
theorem impossible_arrangement :
  ∀ (l : List ℕ),
  l.length = 111 →
  l.All within_range →
  distinct l →
  ¬(∀ (k : ℕ) (h : k < l.length), (l.get ⟨k, h⟩) % 10 = (l.sum - l.get ⟨k, h⟩) % 10) :=
by
  intros l length_cond within_range_cond distinct_cond condition
  sorry

end impossible_arrangement_l309_309014


namespace total_clothing_given_l309_309507

theorem total_clothing_given (shirts trousers : ℕ) (h₁ : shirts = 589) (h₂ : trousers = 345) : shirts + trousers = 934 :=
by
  rw [h₁, h₂]
  exact Nat.add_comm (589) (345) ▸ rfl

end total_clothing_given_l309_309507


namespace number_of_planes_equals_32_l309_309549

-- Define the four non-coplanar points in space with distinct pairwise distances
variables (A B C D : Point) (not_coplanar : ¬ coplanar {A, B, C, D}) (distinct_distances : ∀ (P Q : Point), P ≠ Q → distance P Q ≠ distance P R)

-- Hypothesis concerning distances from plane α
constants (α : Plane) (d : ℝ)
(h1 : distance A α = d ∧ distance B α = d ∧ distance C α = d ∧ distance D α = 2 * d)
(h2 : distance A α = d ∧ distance B α = d ∧ distance D α = d ∧ distance C α = 2 * d)
(h3 : distance A α = d ∧ distance C α = d ∧ distance D α = d ∧ distance B α = 2 * d)
(h4 : distance B α = d ∧ distance C α = d ∧ distance D α = d ∧ distance A α = 2 * d)

-- Define the theorem that states the proof problem
theorem number_of_planes_equals_32 : ∃ α, count_planes_with_given_distances A B C D α d = 32 :=
sorry

end number_of_planes_equals_32_l309_309549


namespace complex_multiplication_l309_309688

-- Definition of the imaginary unit
def is_imaginary_unit (i : ℂ) : Prop := i * i = -1

theorem complex_multiplication (i : ℂ) (h : is_imaginary_unit i) : (1 + i) * (1 - i) = 2 :=
by
  -- Given that i is the imaginary unit satisfying i^2 = -1
  -- We need to show that (1 + i) * (1 - i) = 2
  sorry

end complex_multiplication_l309_309688


namespace fraction_addition_simplified_form_l309_309508

theorem fraction_addition_simplified_form :
  (7 / 8) + (3 / 5) = 59 / 40 := 
by sorry

end fraction_addition_simplified_form_l309_309508


namespace find_slope_l3_l309_309286

/-- Conditions --/
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def line2 (x y : ℝ) : Prop := y = 2
def A : Prod ℝ ℝ := (0, -3)
def area_ABC : ℝ := 5

noncomputable def B : Prod ℝ ℝ := (2, 2)  -- Simultaneous solution of line1 and line2

theorem find_slope_l3 (C : ℝ × ℝ) (slope_l3 : ℝ) :
  line2 C.1 C.2 ∧
  ((0 : ℝ), -3) ∈ {p : ℝ × ℝ | line1 p.1 p.2 → line2 p.1 p.2 } ∧
  C.2 = 2 ∧
  0 ≤ slope_l3 ∧
  area_ABC = 5 →
  slope_l3 = 5 / 4 :=
sorry

end find_slope_l3_l309_309286


namespace claire_white_roses_count_l309_309083

noncomputable def number_of_white_roses 
  (total_flowers : ℕ) 
  (tulips : ℕ) 
  (price_per_red_rose : ℝ) 
  (earnings_by_selling_half : ℝ) : ℕ :=
  let total_roses := total_flowers - tulips
  let number_of_red_roses_sold := 2 * (earnings_by_selling_half / price_per_red_rose).to_nat
  let red_roses := number_of_red_roses_sold
  let white_roses := total_roses - red_roses
  white_roses

theorem claire_white_roses_count : 
  number_of_white_roses 400 120 0.75 75 = 80 := 
by sorry

end claire_white_roses_count_l309_309083


namespace midpoint_parallel_l309_309672

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309672


namespace constant_term_is_24_l309_309601

noncomputable def constant_term_of_binomial_expansion 
  (a : ℝ) (hx : π * a^2 = 4 * π) : ℝ :=
  if ha : a = 2 then 24 else 0

theorem constant_term_is_24
  (a : ℝ) (hx : π * a^2 = 4 * π) :
  constant_term_of_binomial_expansion a hx = 24 :=
by
  sorry

end constant_term_is_24_l309_309601


namespace A_gt_B_and_C_lt_A_l309_309376

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end A_gt_B_and_C_lt_A_l309_309376


namespace compositeQuotientCorrect_l309_309902

namespace CompositeNumbersProof

def firstFiveCompositesProduct : ℕ :=
  21 * 22 * 24 * 25 * 26

def subsequentFiveCompositesProduct : ℕ :=
  27 * 28 * 30 * 32 * 33

def compositeQuotient : ℚ :=
  firstFiveCompositesProduct / subsequentFiveCompositesProduct

theorem compositeQuotientCorrect : compositeQuotient = 1 / 1964 := by sorry

end CompositeNumbersProof

end compositeQuotientCorrect_l309_309902


namespace remainder_is_neg_x_plus_5_l309_309388

open Polynomial

noncomputable def q : Polynomial ℝ := sorry
noncomputable def r : Polynomial ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Conditions
axiom h1 : eval 2 q = 3
axiom h2 : eval 3 q = 2

-- Definition of q(x) in terms of (x-2)(x-3), a, and b
def q_def : q = (Polynomial.C (1 : ℝ) * X^2 - Polynomial.C (5 : ℝ) * X + Polynomial.C (6 : ℝ)) * r + (Polynomial.C a * X + Polynomial.C b) := sorry

-- Goal
theorem remainder_is_neg_x_plus_5 :
  ∃ a b : ℝ, (q = (Polynomial.C (1 : ℝ) * X^2 - Polynomial.C (5 : ℝ) * X + Polynomial.C (6 : ℝ)) * r + (Polynomial.C a * X + Polynomial.C b)) ∧
   a = -1 ∧ b = 5 :=
by
  use [-1, 5]
  split; assumption

end remainder_is_neg_x_plus_5_l309_309388


namespace baron_munchausen_is_telling_truth_l309_309065

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_10_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def not_divisible_by_10 (n : ℕ) : Prop :=
  ¬(n % 10 = 0)

theorem baron_munchausen_is_telling_truth :
  ∃ a b : ℕ, a ≠ b ∧ is_10_digit a ∧ is_10_digit b ∧ not_divisible_by_10 a ∧ not_divisible_by_10 b ∧
  (a - digit_sum (a^2) = b - digit_sum (b^2)) := sorry

end baron_munchausen_is_telling_truth_l309_309065


namespace mn_parallel_pq_l309_309649

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309649


namespace josh_initial_wallet_l309_309639

noncomputable def initial_wallet_amount (investment final_wallet: ℕ) (stock_increase_percentage: ℕ): ℕ :=
  let investment_value_after_rise := investment + (investment * stock_increase_percentage / 100)
  final_wallet - investment_value_after_rise

theorem josh_initial_wallet : initial_wallet_amount 2000 2900 30 = 300 :=
by
  sorry

end josh_initial_wallet_l309_309639


namespace sum_series_value_l309_309991

theorem sum_series_value :
  let x := λ i : ℕ, (i : ℝ) / 101
  let S := ∑ i in Finset.range 102, (x i)^3 / (3 * (x i)^2 - 3 * (x i) + 1)
  S = 51 := 
by
  sorry

end sum_series_value_l309_309991


namespace baron_munchausen_is_telling_truth_l309_309066

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_10_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def not_divisible_by_10 (n : ℕ) : Prop :=
  ¬(n % 10 = 0)

theorem baron_munchausen_is_telling_truth :
  ∃ a b : ℕ, a ≠ b ∧ is_10_digit a ∧ is_10_digit b ∧ not_divisible_by_10 a ∧ not_divisible_by_10 b ∧
  (a - digit_sum (a^2) = b - digit_sum (b^2)) := sorry

end baron_munchausen_is_telling_truth_l309_309066


namespace circle_integer_points_count_l309_309996

theorem circle_integer_points_count : 
  (finset.univ.filter (λ p : ℤ × ℤ, p.1^2 + p.2^2 = 25)).card = 12 :=
sorry

end circle_integer_points_count_l309_309996


namespace side_length_of_square_l309_309760

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l309_309760


namespace children_count_125_l309_309515

def numberOfChildren (a : ℕ) : Prop :=
  a % 8 = 5 ∧ a % 10 = 7 ∧ 100 ≤ a ∧ a ≤ 150

theorem children_count_125 : ∃ a : ℕ, numberOfChildren a ∧ a = 125 := by
  use 125
  unfold numberOfChildren
  apply And.intro
  apply And.intro
  · norm_num
  · norm_num
  · split
  repeat {norm_num}
  sorry

end children_count_125_l309_309515


namespace mn_parallel_pq_l309_309646

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309646


namespace infinitely_many_n_l309_309724

def num_odd_prime_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ p, p.prime ∧ ¬ even p) (n.factors : Finset ℕ)).card

def a_n (n : ℕ) : ℕ := num_odd_prime_divisors (n * (n + 3))

theorem infinitely_many_n (N : ℕ → Prop) : 
  ∃ᶠ n in at_top, a_n n % 3 = 0 := 
sorry

end infinitely_many_n_l309_309724


namespace side_length_of_square_l309_309758

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l309_309758


namespace four_not_at_extreme_l309_309783

def valid_sequence (s : List ℕ) : Prop :=
  ∀ i, i + 2 < s.length → abs (s[i] - s[i + 2]) = 1

theorem four_not_at_extreme (s : List ℕ) (h : valid_sequence s) (h_len : s.length = 9) :
  ¬(s.head = 4 ∨ s.last! = 4) :=
sorry

end four_not_at_extreme_l309_309783


namespace customer_bought_29_eggs_l309_309114

-- Defining the conditions
def baskets : List ℕ := [4, 6, 12, 13, 22, 29]
def total_eggs : ℕ := 86
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Stating the problem
theorem customer_bought_29_eggs :
  ∃ eggs_in_basket,
    eggs_in_basket ∈ baskets ∧
    is_multiple_of_three (total_eggs - eggs_in_basket) ∧
    eggs_in_basket = 29 :=
by sorry

end customer_bought_29_eggs_l309_309114


namespace percentage_of_green_eyed_brunettes_l309_309464

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end percentage_of_green_eyed_brunettes_l309_309464


namespace figure_perimeter_diff_l309_309200

def perimeter_rect (l w : ℕ) : ℕ := 2 * (l + w)

def perimeter_L_shape (a b : ℕ) (missing : ℕ) : ℕ :=
  let total_perimeter := 2 * (a + b)
  let subtract_for_missing := 2 * missing
  let subtract_for_interior := 1
  total_perimeter - subtract_for_missing - subtract_for_interior

def Figure1_perimeter : ℕ := perimeter_rect 3 1 + perimeter_L_shape 2 2 1

def Figure2_perimeter : ℕ := perimeter_rect 6 2

def perimeter_difference : ℕ :=
  abs (Figure1_perimeter - Figure2_perimeter)

theorem figure_perimeter_diff : perimeter_difference = 3 :=
by
  unfold perimeter_difference 
  unfold Figure1_perimeter 
  unfold Figure2_perimeter 
  unfold perimeter_rect 
  unfold perimeter_L_shape 
  sorry

end figure_perimeter_diff_l309_309200


namespace find_f_1_7_l309_309279

noncomputable def f (x : ℝ) : ℝ := sorry
def a : ℝ := 1 / 2

axiom f_conditions (x y : ℝ) (hx : 0 ≤ x) (hy : x ≤ y) (hy' : y ≤ 1) : 
  f ((x + y) / 2) = (1 - a) * f x + a * f y

axiom f_endpoints : f 0 = 0 ∧ f 1 = 1

theorem find_f_1_7 : f (1/7) = 1/7 := sorry

end find_f_1_7_l309_309279


namespace correct_answer_is_C_l309_309860

def exactly_hits_n_times (n k : ℕ) : Prop :=
  n = k

def hits_no_more_than (n k : ℕ) : Prop :=
  n ≤ k

def hits_at_least (n k : ℕ) : Prop :=
  n ≥ k

def is_mutually_exclusive (P Q : Prop) : Prop :=
  ¬ (P ∧ Q)

def is_non_opposing (P Q : Prop) : Prop :=
  ¬ P ∧ ¬ Q

def events_are_mutually_exclusive_and_non_opposing (n : ℕ) : Prop :=
  let event1 := exactly_hits_n_times 5 3
  let event2 := exactly_hits_n_times 5 4
  is_mutually_exclusive event1 event2 ∧ is_non_opposing event1 event2

theorem correct_answer_is_C : events_are_mutually_exclusive_and_non_opposing 5 :=
by
  sorry

end correct_answer_is_C_l309_309860


namespace a_greater_than_b_c_less_than_a_l309_309374

-- Condition 1: Definition of box dimensions
def Box := (Nat × Nat × Nat)

-- Condition 2: Dimension comparisons
def le_box (a b : Box) : Prop :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  (a1 ≤ b1 ∨ a1 ≤ b2 ∨ a1 ≤ b3) ∧ (a2 ≤ b1 ∨ a2 ≤ b2 ∨ a2 ≤ b3) ∧ (a3 ≤ b1 ∨ a3 ≤ b2 ∨ a3 ≤ b3)

def lt_box (a b : Box) : Prop := le_box a b ∧ ¬(a = b)

-- Condition 3: Box dimensions
def A : Box := (6, 5, 3)
def B : Box := (5, 4, 1)
def C : Box := (3, 2, 2)

-- Equivalent Problem 1: Prove A > B
theorem a_greater_than_b : lt_box B A :=
by
  -- theorem proof here
  sorry

-- Equivalent Problem 2: Prove C < A
theorem c_less_than_a : lt_box C A :=
by
  -- theorem proof here
  sorry

end a_greater_than_b_c_less_than_a_l309_309374


namespace small_n_for_sum_l309_309793

def a : ℕ → ℝ
| 1     := 1.5
| (n+2) := 1 / (n + 2)^2 - 1

noncomputable def S (n : ℕ) : ℝ :=
1.5 + (∑ k in finset.range (n - 1 + 1), 1 / (k + 1)^2 - 1)

theorem small_n_for_sum :
  ∃ n : ℕ, |S n - 2.25| < 0.01 ∧ n = 100 :=
begin
  sorry
end

end small_n_for_sum_l309_309793


namespace periodic_sequence_l309_309280

-- Define the necessary parameters and conditions
variable (r : ℕ) (a : ℕ → ℝ)

-- Define the main theorem stating the periodicity of the sequence
theorem periodic_sequence : 
  (∀ (m s : ℕ), ∃ (n : ℕ), (m + 1 ≤ n ∧ n ≤ m + r) ∧ 
    (∑ i in finset.range (m + s + 1) \ (finset.range m), a i) = 
    (∑ i in finset.range (n + s + 1) \ (finset.range n), a i)) →
  ∃ p ≥ 1, ∀ n ≥ 0, a (n + p) = a n := 
sorry

end periodic_sequence_l309_309280


namespace ellipse_foci_l309_309332

noncomputable def focal_coordinates (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

-- Given the equation of the ellipse: x^2 / a^2 + y^2 / b^2 = 1
def ellipse_equation (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

-- Proposition stating that if the ellipse equation holds for a=√5 and b=2, then the foci are at (± c, 0)
theorem ellipse_foci (x y : ℝ) (h : ellipse_equation x y (Real.sqrt 5) 2) :
  y = 0 ∧ (x = 1 ∨ x = -1) :=
sorry

end ellipse_foci_l309_309332


namespace find_modulus_l309_309564

theorem find_modulus {z : ℂ} {a : ℝ} (h : (a-2) * z^2018 + a * z^2017 * complex.I + a * z * complex.I + 2 - a = 0) (ha : a < 1) : 
  complex.abs z = 1 := 
by 
  sorry

end find_modulus_l309_309564


namespace sandy_bought_6_books_l309_309305

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_bought_6_books_l309_309305


namespace circumcircle_intersection_on_BC_l309_309693

-- Definition of the problem's conditions
variables {A B C M N O R : Point}
variables (h1 : Triangle A B C)
variables (h2 : AcuteTriangle A B C)
variables (h3 : ¬IsoscelesTriangle A B C A)
variables (h4 : CircleDiameterOf BC)
variables (h5 : OnCircle M (CircleDiameterOf BC))
variables (h6 : OnCircle N (CircleDiameterOf BC))
variables (h7 : Midpoint O BC)
variables (h8 : AngleBisectorsIntersection R (∠BAC) (∠MON))

-- Statement of the theorem to prove
theorem circumcircle_intersection_on_BC :
  IntersectOnLine (Circumcircle (Triangle B M R)) (Circumcircle (Triangle C N R)) BC :=
sorry

end circumcircle_intersection_on_BC_l309_309693


namespace length_segment_AB_l309_309616

open Real

noncomputable def parametric_curve : ℝ → ℝ × ℝ := λ t => (t - 1, t^2 - 1)

def line_y_eq_x : ℝ → ℝ × ℝ := λ x => (x, x)

def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = parametric_curve t ∧ p = line_y_eq_x (p.fst)}

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2

theorem length_segment_AB :
  ∃ A B ∈ intersection_points, dist A B = 2 :=
sorry

end length_segment_AB_l309_309616


namespace product_bc_l309_309801

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end product_bc_l309_309801


namespace smallest_positive_multiple_of_45_with_digits_0_and_8_l309_309514

theorem smallest_positive_multiple_of_45_with_digits_0_and_8 :
  ∃ K : ℕ, (K > 0 ∧ K % 45 = 0 ∧ (∀ d, d ∈ (K.digits 10) → d = 0 ∨ d = 8) ∧ (last_digit K = 0) ∧ (digit_sum K % 9 = 0) ∧ K = 8888888880) := 
sorry

end smallest_positive_multiple_of_45_with_digits_0_and_8_l309_309514


namespace max_k_value_l309_309968

noncomputable def f (x : ℝ) := x^2 + x * Real.log x
def g (x : ℝ) := x^2 - x

theorem max_k_value :
  (∃ k : ℤ, ∀ x > 2, k * (x - 2) < f x - g x) → 4 := sorry

end max_k_value_l309_309968


namespace xy_fraction_l309_309592

theorem xy_fraction (x y : ℚ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) :
  x * y = -1 / 5 := 
by sorry

end xy_fraction_l309_309592


namespace cups_flip_impossible_l309_309349

theorem cups_flip_impossible : ¬(∃ n : ℕ, ∃ flips : fin n → fin 11, (∀ i, flips i.val < 11 ∧ flips i.val ≠ flips j.val ∧
  (initial_state : vector ℕ 11) (final_state = initial_state),
    flips i.val → final_state[i] = !initial_state[i] ∧
    (final_state.count = 6) → (initial_state[i], up))) :=
sorry

end cups_flip_impossible_l309_309349


namespace managers_meeting_l309_309432

/-- A meeting has to be conducted with 4 managers out of 7 managers.
Two specific managers (A and B) will not attend the meeting together.
Prove that the number of ways to select the managers for the meeting is 25. -/
theorem managers_meeting : 
  let total_managers := 7
  let choose_managers := 4
  let specific_managers_not_together := 2
  (∑ k in {0, 1}, Nat.choose (total_managers - specific_managers_not_together) (choose_managers - k) * Nat.choose specific_managers_not_together k) = 25 :=
by
  sorry

end managers_meeting_l309_309432


namespace mul_99_101_equals_9999_l309_309889

theorem mul_99_101_equals_9999 : 99 * 101 = 9999 := by
  sorry

end mul_99_101_equals_9999_l309_309889


namespace rectangle_side_ratio_l309_309431

theorem rectangle_side_ratio (s x y : ℝ) 
  (h1 : 8 * (x * y) = (9 - 1) * s^2) 
  (h2 : s + 4 * y = 3 * s) 
  (h3 : 2 * x + y = 3 * s) : 
  x / y = 2.5 :=
by
  sorry

end rectangle_side_ratio_l309_309431


namespace problem_1_problem_2_l309_309078

-- Problem 1 Statement in Lean 4
theorem problem_1 : sqrt 12 + 3 - 2^2 + abs (1 - sqrt 3) = 3 * sqrt 3 - 2 := 
by sorry

-- Problem 2 Statement in Lean 4
theorem problem_2 : (sqrt 3 - sqrt 2)^2 - (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 4 - 2 * sqrt 6 := 
by sorry

end problem_1_problem_2_l309_309078


namespace smallest_third_term_GP_l309_309859

theorem smallest_third_term_GP : 
  ∃ d : ℝ, 
    (11 + d) ^ 2 = 9 * (29 + 2 * d) ∧
    min (29 + 2 * 10) (29 + 2 * -14) = 1 :=
by
  sorry

end smallest_third_term_GP_l309_309859


namespace g_is_odd_function_l309_309243

noncomputable def g (x : ℝ) := 5 / (3 * x^5 - 7 * x)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  unfold g
  sorry

end g_is_odd_function_l309_309243


namespace quadratic_function_inequality_l309_309165

theorem quadratic_function_inequality
  (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hx1_pos : 0 < x1)
  (hx2_pos : x1 < x2)
  (hy1 : y1 = x1^2 - 1)
  (hy2 : y2 = x2^2 - 1) :
  y1 < y2 := 
sorry

end quadratic_function_inequality_l309_309165


namespace sum_of_first_100_terms_l309_309575

def f (x : ℝ) := x^2 * Real.cos (π * x / 2)

def a_n (n : ℕ) := f n + f (n + 1)

def S (N : ℕ) := ∑ n in Finset.range N, a_n n

theorem sum_of_first_100_terms :
  S 100 = 10200 :=
sorry

end sum_of_first_100_terms_l309_309575


namespace sin_cos_difference_l309_309106

theorem sin_cos_difference :
  sin (50 * Real.pi / 180) * cos (20 * Real.pi / 180) - 
  cos (50 * Real.pi / 180) * sin (20 * Real.pi / 180) = 1 / 2 :=
sorry

end sin_cos_difference_l309_309106


namespace simplify_fraction_eq_l309_309310

theorem simplify_fraction_eq :
  (∀ x, (x-3)*(x-1) = x^2 - 4x + 3) →
  (∀ x, (x-4)*(x-2) = x^2 - 6x + 8) →
  (∀ x, (x-5)*(x-1) = x^2 - 6x + 5) →
  (∀ x, (x-5)*(x-3) = x^2 - 8x + 15) →
  (∀ x, x ≠ 1 → x ≠ 2 → x ≠ 3 → x ≠ 4 → x ≠ 5 →
    (x^2 - 4x + 3) / (x^2 - 6x + 8) / ((x^2 - 6x + 5) / (x^2 - 8x + 15)) =
    ((x-3)*(x-3)) / ((x-4)*(x-2))) :=
by
  sorry

end simplify_fraction_eq_l309_309310


namespace num_partitions_l309_309257

open Finset

theorem num_partitions (s : Finset ℕ) (h : s = (range 15)) :
  let p := s.filter (λ x, x ≠ 7) in
  ∑ x in p, choose 12 (x-1) = 3172 := 
by
  let n := ∑ x in range 13, if x ≠ 7 then 1 else 0;
  let m := choose 12 6;
  have h1 : ∑ x in p, choose 12 (x-1) = 2 ^ 12 - m :=
    by simp [p, sum_filter, choose, h];
  rw [h1];
  simp [nat.choose, nat.sub, tsub, nat.pow];
  sorry

end num_partitions_l309_309257


namespace compound_interest_rate_l309_309062

theorem compound_interest_rate (P : ℝ) (r : ℝ) (t : ℕ) (A : ℝ) 
  (h1 : t = 15) (h2 : A = (9 / 5) * P) :
  (1 + r) ^ t = (9 / 5) → 
  r ≠ 0.05 ∧ r ≠ 0.06 ∧ r ≠ 0.07 ∧ r ≠ 0.08 :=
by
  -- Sorry could be placed here for now
  sorry

end compound_interest_rate_l309_309062


namespace plant_height_increase_l309_309776

theorem plant_height_increase (total_increase : ℕ) (century_in_years : ℕ) (decade_in_years : ℕ) (years_in_2_centuries : ℕ) (num_decades : ℕ) : 
  total_increase = 1800 →
  century_in_years = 100 →
  decade_in_years = 10 →
  years_in_2_centuries = 2 * century_in_years →
  num_decades = years_in_2_centuries / decade_in_years →
  total_increase / num_decades = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end plant_height_increase_l309_309776


namespace polynomial_integer_values_all_integers_l309_309828

open Polynomial

theorem polynomial_integer_values_all_integers (P : ℤ[X]) (n : ℕ) :
  (∀ i : ℕ, i ≤ n → is_integral (P.eval i)) →
  ∀ x : ℤ, is_integral (P.eval x) :=
sorry

end polynomial_integer_values_all_integers_l309_309828


namespace sticker_price_l309_309728

theorem sticker_price (y : ℝ) (h1 : ∀ (p : ℝ), p = 0.8 * y - 60 → p ≤ y)
  (h2 : ∀ (q : ℝ), q = 0.7 * y → q ≤ y)
  (h3 : (0.8 * y - 60) + 20 = 0.7 * y) :
  y = 400 :=
by
  sorry

end sticker_price_l309_309728


namespace a_greater_than_b_c_less_than_a_l309_309373

-- Condition 1: Definition of box dimensions
def Box := (Nat × Nat × Nat)

-- Condition 2: Dimension comparisons
def le_box (a b : Box) : Prop :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  (a1 ≤ b1 ∨ a1 ≤ b2 ∨ a1 ≤ b3) ∧ (a2 ≤ b1 ∨ a2 ≤ b2 ∨ a2 ≤ b3) ∧ (a3 ≤ b1 ∨ a3 ≤ b2 ∨ a3 ≤ b3)

def lt_box (a b : Box) : Prop := le_box a b ∧ ¬(a = b)

-- Condition 3: Box dimensions
def A : Box := (6, 5, 3)
def B : Box := (5, 4, 1)
def C : Box := (3, 2, 2)

-- Equivalent Problem 1: Prove A > B
theorem a_greater_than_b : lt_box B A :=
by
  -- theorem proof here
  sorry

-- Equivalent Problem 2: Prove C < A
theorem c_less_than_a : lt_box C A :=
by
  -- theorem proof here
  sorry

end a_greater_than_b_c_less_than_a_l309_309373


namespace hunter_strategy_does_not_increase_probability_l309_309028

theorem hunter_strategy_does_not_increase_probability (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1) :
  let p2 := p * p in
  let split_prob := p * (1 - p) in
  let combined_prob := p2 + split_prob in
  combined_prob = p :=
by
  let p := p
  let p2 := p * p
  let split_prob := p * (1 - p)
  let combined_prob := p2 + split_prob
  have : combined_prob = p := sorry
  exact this

end hunter_strategy_does_not_increase_probability_l309_309028


namespace missing_digit_is_two_l309_309478

def sequence := { x | ∃ n: ℕ, x = 8 * (10^n - 1) / 9 ∧ n < 9 }

def arithmetic_mean (s : Set ℕ) : ℕ :=
  (s.to_finset.sum id) / s.to_finset.card

noncomputable def N : ℕ := arithmetic_mean sequence

theorem missing_digit_is_two :
  ¬(2 ∈ Nat.digits 10 N) := by 
  sorry

end missing_digit_is_two_l309_309478


namespace length_of_EF_l309_309614

theorem length_of_EF (AB BC : ℝ) (hAB : AB = 4) (hBC : BC = 8) :
  let EF := 2 * Real.sqrt 5 in EF = 2 * Real.sqrt 5 :=
by
  sorry

end length_of_EF_l309_309614


namespace bottles_left_l309_309356

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l309_309356


namespace opposite_of_2_is_minus_2_l309_309785

-- Define the opposite function
def opposite (x : ℤ) : ℤ := -x

-- Assert the theorem to prove that the opposite of 2 is -2
theorem opposite_of_2_is_minus_2 : opposite 2 = -2 := by
  sorry -- Placeholder for the proof

end opposite_of_2_is_minus_2_l309_309785


namespace don_eats_80_pizzas_l309_309492

variable (D Daria : ℝ)

-- Condition 1: Daria consumes 2.5 times the amount of pizza that Don does.
def condition1 : Prop := Daria = 2.5 * D

-- Condition 2: Together, they eat 280 pizzas.
def condition2 : Prop := D + Daria = 280

-- Conclusion: The number of pizzas Don eats is 80.
theorem don_eats_80_pizzas (h1 : condition1 D Daria) (h2 : condition2 D Daria) : D = 80 :=
by
  sorry

end don_eats_80_pizzas_l309_309492


namespace average_values_in_interval_l309_309379

noncomputable def average_of_int_values : ℚ :=
  let values := { N : ℤ | 16 < N ∧ N < 27 }
  (values.sum id).toRat / values.card

theorem average_values_in_interval :
  average_of_int_values = 21.5 :=
by
  sorry

end average_values_in_interval_l309_309379


namespace non_congruence_of_isosceles_triangles_l309_309631

open Real

noncomputable def same_inscribed_circle_radius (A B C : Point) (r : ℝ) : Prop :=
  -- Assuming a way to calculate the radius of the inscribed circle
  radius_of_inscribed_circle A B C = r

def acute_angled (A B C : Point) :=
  -- Placeholder definition for an acute angle condition.
  ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90 

def isosceles_triangle (A B C : Point) : Prop :=
  -- Placeholder definition for an isosceles triangle with AB = AC
  dist A B = dist A C

theorem non_congruence_of_isosceles_triangles 
  (A1 B1 C1 A2 B2 C2 : Point) (r : ℝ) :
  isosceles_triangle A1 B1 C1 →
  isosceles_triangle A2 B2 C2 →
  acute_angled A1 B1 C1 →
  acute_angled A2 B2 C2 →
  same_inscribed_circle_radius A1 B1 C1 r →
  same_inscribed_circle_radius A2 B2 C2 r →
  ¬congruent A1 B1 C1 A2 B2 C2 :=
by 
  -- Placeholder proof
  sorry

end non_congruence_of_isosceles_triangles_l309_309631


namespace midpoint_parallel_l309_309668

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309668


namespace club_truncator_more_wins_than_losses_probability_club_truncator_answer_m_plus_n_l309_309486

noncomputable def prob_more_wins_than_losses : ℚ :=
  let total_outcomes := 3 ^ 8 in
  let equal_wins_losses := 70 + 560 + 420 + 28 + 1 in
  let unequal_wins_losses := total_outcomes - equal_wins_losses in
  let prob := (unequal_wins_losses / total_outcomes) / 2 in
  prob

theorem club_truncator_more_wins_than_losses_probability :
  prob_more_wins_than_losses = 2741 / 6561 :=
by
  sorry

theorem club_truncator_answer_m_plus_n :
  let m := 2741 in
  let n := 6561 in
  m + n = 9302 :=
by
  sorry

end club_truncator_more_wins_than_losses_probability_club_truncator_answer_m_plus_n_l309_309486


namespace square_computation_product_computation_l309_309712

-- Define the broken calculator operations and the propositions to prove
section BrokenCalculator

variables (a b c : ℝ)
local notation "recip" x := 1 / x

-- Definitions of necessary functions using only recip, add, sub
def broken_calculator_square (a c : ℝ) := 
  let A := recip a - recip (a + c)
  let A_rec := recip A
  c * (A_rec - a)

def broken_calculator_product (a b c : ℝ) := 
  let half_b := recip (2 * recip b)
  let square_sum := (a + half_b) ^ 2
  let a_square := broken_calculator_square a c
  let half_b_square := (half_b ^ 2)
  square_sum - a_square - half_b_square

-- Propositions to prove
theorem square_computation (a : ℝ) (c : ℝ) : 
  a ^ 2 = c * (recip (c / (a ^ 2 + a * c)) - a) := 
sorry

theorem product_computation (a : ℝ) (b : ℝ) : 
  a * b = (a + recip (2 * recip b)) ^ 2 - a ^ 2 - (recip (2 * recip b)) ^ 2 := 
sorry

end BrokenCalculator

end square_computation_product_computation_l309_309712


namespace no_such_arrangement_l309_309009

theorem no_such_arrangement :
  ¬∃ (a : Fin 111 → ℕ), (∀ i : Fin 111, a i ≤ 500 ∧ (∀ j k : Fin 111, j ≠ k → a j ≠ a k)) ∧ (∀ i : Fin 111, (a i % 10) = ((Finset.univ.sum (λ j, if j = i then 0 else a j)) % 10)) :=
by
  sorry

end no_such_arrangement_l309_309009


namespace largest_possible_product_is_3886_l309_309718

theorem largest_possible_product_is_3886 :
  ∃ a b c d : ℕ, 5 ≤ a ∧ a ≤ 8 ∧
               5 ≤ b ∧ b ≤ 8 ∧
               5 ≤ c ∧ c ≤ 8 ∧
               5 ≤ d ∧ d ≤ 8 ∧
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
               b ≠ c ∧ b ≠ d ∧
               c ≠ d ∧
               (max ((10 * a + b) * (10 * c + d))
                    ((10 * c + b) * (10 * a + d))) = 3886 :=
sorry

end largest_possible_product_is_3886_l309_309718


namespace expected_value_8_sided_die_l309_309412

-- Define the roll outcomes and their associated probabilities
def roll_outcome (n : ℕ) : ℕ := 2 * n^2

-- Define the expected value calculation
def expected_value (sides : ℕ) : ℚ := ∑ i in range (1, sides+1), (1 / sides) * roll_outcome i

-- Prove the expected value calculation for an 8-sided fair die
theorem expected_value_8_sided_die : expected_value 8 = 51 := by
  sorry

end expected_value_8_sided_die_l309_309412


namespace find_m_l309_309559

theorem find_m (x : ℝ) (m : ℝ) (h1 : x > 2) (h2 : x - 3 * m + 1 > 0) : m = 1 :=
sorry

end find_m_l309_309559


namespace ellipse_standard_form_l309_309952

def center_origin (x y : ℝ) := x = 0 ∧ y = 0
def focus_y_axis (f : ℝ × ℝ) := f.1 = 0 -- Only y-coordinate matters as x must be zero for focus on y-axis
def eccentricity (c a : ℝ) := c / a = 1 / 2
def focal_length (c : ℝ) := 2 * c = 8
def ellipse_eq (a b : ℝ) := (λ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1)

theorem ellipse_standard_form :
  ∃ a b c : ℝ, center_origin 0 0 ∧ focus_y_axis (0, c) ∧ eccentricity c a ∧ focal_length c ∧ a = 8 ∧ b^2 = a^2 - c^2 ∧ ellipse_eq a b = (λ x y, y^2 / 64 + x^2 / 48 = 1) := 
by {
  sorry
}

end ellipse_standard_form_l309_309952


namespace handshake_count_l309_309879

theorem handshake_count (couples : ℕ) (men_handshakes : ℕ) (men_women_handshakes : ℕ) (women_handshakes : ℕ)
  (h1 : couples = 15)
  (h2 : men_handshakes = (couples * (couples - 1)) / 2)
  (h3 : men_women_handshakes = couples * (couples - 1))
  (h4 : women_handshakes = (3 * (3 - 1)) / 2) :
  men_handshakes + men_women_handshakes + women_handshakes = 318 := 
by
  have h5 : men_handshakes = (15 * 14) / 2 := by rw h1; rw h2
  have h6 : men_women_handshakes = 15 * 14 := by rw h1; rw h3
  have h7 : women_handshakes = (3 * 2) / 2 := by rw h4
  calc
    men_handshakes + men_women_handshakes + women_handshakes
      = ((15 * 14) / 2) + (15 * 14) + ((3 * 2) / 2) : by rw [h5, h6, h7]
  ... = 105 + 210 + 3 : by norm_num
  ... = 318 : by norm_num

end handshake_count_l309_309879


namespace infinite_composite_numbers_l309_309308

theorem infinite_composite_numbers:
  ∃ᶠ n in (filter.at_top : filter ℕ), n % 6 = 4 ∧ ¬ nat.prime (n^n + (n + 1)^(n+1)) :=
sorry

end infinite_composite_numbers_l309_309308


namespace triangle_MNO_perimeter_l309_309862

-- Define the vertices of the prism
structure Prism :=
  (P Q R S T U : Vect3) -- vertices of the prism
  (height : ℝ)
  (side_length : ℝ)
  (equilateral_tri_base : equilateral_triangle P Q R)
  (right_prism_edges : right_prism P Q R S T U)

-- Define the midpoint points M, N, and O
structure Midpoints :=
  (M N O : Vect3)
  (midpoint_PR : M = midpoint_ P R)
  (midpoint_QR : N = midpoint_ Q R)
  (midpoint_RT : O = midpoint_ R T)

-- Main theorem statement
theorem triangle_MNO_perimeter (prsm : Prism) (mdpts : Midpoints) :
  perimeter (triangle mdpts.M mdpts.N mdpts.O) = 5 * (2 * sqrt 5 + 1) :=
by 
  sorry

end triangle_MNO_perimeter_l309_309862


namespace sin_theta_value_l309_309590

theorem sin_theta_value (θ : ℝ) (h1 : 5 * tan θ = 2 * cos θ) (h2 : 0 < θ ∧ θ < π) : sin θ = 1 / 2 :=
by
  sorry

end sin_theta_value_l309_309590


namespace fraction_of_b_equals_4_15_of_a_is_0_4_l309_309392

variable (A B : ℤ)
variable (X : ℚ)

def a_and_b_together_have_1210 : Prop := A + B = 1210
def b_has_484 : Prop := B = 484
def fraction_of_b_equals_4_15_of_a : Prop := (4 / 15 : ℚ) * A = X * B

theorem fraction_of_b_equals_4_15_of_a_is_0_4
  (h1 : a_and_b_together_have_1210 A B)
  (h2 : b_has_484 B)
  (h3 : fraction_of_b_equals_4_15_of_a A B X) :
  X = 0.4 := sorry

end fraction_of_b_equals_4_15_of_a_is_0_4_l309_309392


namespace num_points_P_on_ellipse_l309_309398

theorem num_points_P_on_ellipse :
  let line := (λ x y : ℝ, x / 4 + y / 3 = 1)
  let ellipse := (λ x y : ℝ, x^2 / 16 + y^2 / 9 = 1)
  let area_triangle_PAB := (λ P A B : ℝ × ℝ, ∃ (h : ℝ), h = 6 / 5 ∧ 
    let APB := (A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1)
    in abs APB / 2 = 3)
  ∃ (P : ℝ × ℝ) (A B : ℝ × ℝ), 
    line A.1 A.2 ∧ line B.1 B.2 ∧ ellipse P.1 P.2 ∧ ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
    area_triangle_PAB P A B → P = (4 * cos, 3 * sin) 
  :=
-- The correct answer is 2
  sorry

end num_points_P_on_ellipse_l309_309398


namespace child_ticket_cost_l309_309852

variable (adult_ticket_price : ℕ) 
variable (total_tickets_sold : ℕ)
variable (total_revenue : ℕ)
variable (adult_tickets_sold : ℕ)
variable (child_ticket_price : ℕ)

theorem child_ticket_cost (h₁ : adult_ticket_price = 7) 
                         (h₂ : total_tickets_sold = 900) 
                         (h₃ : total_revenue = 5100)
                         (h₄ : adult_tickets_sold = 500) :
  let child_tickets_sold := total_tickets_sold - adult_tickets_sold in
  let revenue_from_adults := adult_tickets_sold * adult_ticket_price in
  let revenue_from_children := total_revenue - revenue_from_adults in
  let child_ticket_cost := revenue_from_children / child_tickets_sold in
  child_ticket_cost = 4 :=
by {
  sorry
}

end child_ticket_cost_l309_309852


namespace polio_cases_in_1990_l309_309222

theorem polio_cases_in_1990 (c_1970 c_2000 : ℕ) (T : ℕ) (linear_decrease : ∀ t, c_1970 - (c_2000 * t) / T > 0):
  (c_1970 = 300000) → (c_2000 = 600) → (T = 30) → ∃ c_1990, c_1990 = 100400 :=
by
  intros
  sorry

end polio_cases_in_1990_l309_309222


namespace positive_area_triangles_in_6x6_grid_l309_309196

theorem positive_area_triangles_in_6x6_grid : 
  let points := (fin 6) × (fin 6)
  -- function to check if three points are collinear
  let collinear (p1 p2 p3 : points) : Prop :=
    (p1.1.to_nat * (p2.2.to_nat - p3.2.to_nat) + p2.1.to_nat * (p3.2.to_nat - p1.2.to_nat) + p3.1.to_nat * (p1.2.to_nat - p2.2.to_nat) = 0)
  -- number of subsets of three non-collinear points
  let non_collinear_triples : finset (points × points × points) := 
    (finset.univ : finset points).to_list.combinations 3
    |>.filter (λ l, ¬ collinear l.head (l.tail.head) l.tail.tail.head)
   let total_non_collinear_triples := non_collinear_triples.to_finset.card
  in total_non_collinear_triples = 6744 :=
sorry

end positive_area_triangles_in_6x6_grid_l309_309196


namespace sports_parade_children_l309_309520

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l309_309520


namespace domain_of_function_l309_309899

def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  ((-b + real.sqrt discriminant) / (2 * a), (-b - real.sqrt discriminant) / (2 * a))

def is_root (p q : ℝ) (r : ℝ) : Prop :=
  r = p ∨ r = q

theorem domain_of_function : 
  ∀ x : ℝ, ¬ is_root (9 + real.sqrt 21) / 6 (9 - real.sqrt 21) / 6 x → 
  3 * x^2 - 9 * x + 5 ≠ 0 := by
  sorry

end domain_of_function_l309_309899


namespace rhombus_area_l309_309833

-- Definition of a rhombus with given conditions
structure Rhombus where
  side : ℝ
  d1 : ℝ
  d2 : ℝ

noncomputable def Rhombus.area (r : Rhombus) : ℝ :=
  (r.d1 * r.d2) / 2

noncomputable example : Rhombus :=
{ side := 20,
  d1 := 16,
  d2 := 8 * Real.sqrt 21 }

theorem rhombus_area : 
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  Rhombus.area r = 64 * Real.sqrt 21 :=
by
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  sorry

end rhombus_area_l309_309833


namespace count_diff_2_ways_l309_309844

def Material := {plastic, wood}
def Size := {small, medium, large, xlarge}
def Color := {blue, green, red, yellow}
def Shape := {circle, hexagon, square, triangle}

constant blocks : List (Material × Size × Color × Shape)
constant distinct_blocks : blocks.toFinset.card = 128
constant target_block := (plastic, medium, red, circle)

theorem count_diff_2_ways :
  (blocks.filter (λ b, 
  (if b.1 ≠ plastic then 1 else 0) + 
  (if b.2 ≠ medium then 1 else 0) + 
  (if b.3 ≠ red then 1 else 0) + 
  (if b.4 ≠ circle then 1 else 0) = 2)).length = 30 := sorry

end count_diff_2_ways_l309_309844


namespace triangle_shape_and_maximum_tan_B_minus_C_l309_309242

open Real

variable (A B C : ℝ)
variable (sin cos tan : ℝ → ℝ)

-- Given conditions
axiom sin2A_plus_3sin2C_equals_3sin2B : sin A ^ 2 + 3 * sin C ^ 2 = 3 * sin B ^ 2
axiom sinB_cosC_equals_2div3 : sin B * cos C = 2 / 3

-- Prove
theorem triangle_shape_and_maximum_tan_B_minus_C :
  (A = π / 2) ∧ (∀ x y : ℝ, (x = B - C) → tan x ≤ sqrt 2 / 4) :=
by sorry

end triangle_shape_and_maximum_tan_B_minus_C_l309_309242


namespace inequality_for_positive_reals_l309_309299

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) :=
by
  sorry

end inequality_for_positive_reals_l309_309299


namespace money_left_l309_309491

def initial_money : ℕ := 7
def candy_cost : ℕ := 2
def chocolate_cost : ℕ := 3

theorem money_left : initial_money - (candy_cost + chocolate_cost) = 2 := 
by
  rw [initial_money, candy_cost, chocolate_cost]
  sorry

end money_left_l309_309491


namespace find_sum_l309_309281

noncomputable def a : ℕ → ℝ
| 0       := -3
| (n + 1) := 2 * a n + b n + Real.sqrt (a n ^ 2 + b n ^ 2)

noncomputable def b : ℕ → ℝ
| 0       := 2
| (n + 1) := b n + 3 * a n - Real.sqrt (a n ^ 2 + b n ^ 2)

theorem find_sum (n : ℕ) :
  (∀ n : ℕ, 1 / a n + 1 / b n = 1 / (-3) + 1 / 2) → 1 / a 2012 + 1 / b 2012 = 1 / 6 :=
by
  intro hn_constant
  exact hn_constant 2012


end find_sum_l309_309281


namespace garage_sale_radio_l309_309474

theorem garage_sale_radio (h_total : ∀ (prices : List ℕ), prices.length = 36)
 (h_15th_highest : ∀ (prices : List ℕ) (radio_price : ℕ), (prices.erase radio_price).length = 35 
 ∧ prices.sort.nth_le 14 (by sorry) = radio_price) :
 ∃ (n : ℕ), n = 22 := 
by
  sorry

end garage_sale_radio_l309_309474


namespace log_fixed_point_l309_309775

theorem log_fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
    ∃ (P : ℝ × ℝ), P = (2, 1) ∧ f(2) = 1 :=
by
  let f := λ x : ℝ, log a (2 * x - 3) + 1
  use (2, 1)
  split
  · rfl
  · simp [f]
  · sorry

end log_fixed_point_l309_309775


namespace min_m_for_perfect_function_l309_309595

noncomputable def g (x : ℝ) : ℝ := Real.exp x + x - Real.log x + 1

theorem min_m_for_perfect_function : ∃ m : ℕ, m = 3 ∧ 
  ∀ x ∈ set.Ici (3/2 : ℝ), (g x > 0) ∧ 
    (∀ y ∈ set.Ici (3/2 : ℝ), y ≤ x → g y <= g x) ∧ 
    (∀ y ∈ set.Ici (3/2 : ℝ), y ≤ x → g y / y <= g x / x) := by
  sorry

end min_m_for_perfect_function_l309_309595


namespace max_value_neg1_to_2_min_max_value_m_to_0_l309_309567

noncomputable def f (x : ℝ) : ℝ := 2^(2 * x) - 2^(x + 1) + 3

theorem max_value_neg1_to_2 : (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 2) : 
  2^(2 * x) - 2^(x + 1) + 3 ≤ 11 :=
sorry

theorem min_max_value_m_to_0 (m x : ℝ) (h1 : m ≤ 0) (h2 : m ≤ x) (h3 : x ≤ 0) :
  2^(2 * x) - 2^(x + 1) + 3 ≥ 2 ∧
  2^(2 * x) - 2^(x + 1) + 3 ≤ 2^(2 * m) - 2^(m + 1) + 3 :=
sorry

end max_value_neg1_to_2_min_max_value_m_to_0_l309_309567


namespace side_length_of_square_l309_309746

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l309_309746


namespace cistern_fill_time_l309_309410

theorem cistern_fill_time
  (T : ℝ)
  (H1 : 0 < T)
  (rate_first_tap : ℝ := 1 / T)
  (rate_second_tap : ℝ := 1 / 6)
  (net_rate : ℝ := 1 / 12)
  (H2 : rate_first_tap - rate_second_tap = net_rate) :
  T = 4 :=
sorry

end cistern_fill_time_l309_309410


namespace tan_quotient_sum_l309_309690

theorem tan_quotient_sum (x y : ℝ) 
  (h1 : (sin x / cos y) + (sin y / cos x) = 1)
  (h2 : (cos x / sin y) + (cos y / sin x) = 6) :
  (tan x / tan y) + (tan y / tan x) = 124 / 13 := 
by
  -- Proof goes here.
  sorry

end tan_quotient_sum_l309_309690


namespace data_set_unique_l309_309442

noncomputable def data_set := {s : Set ℕ // s.card = 4 ∧ ∀ x ∈ s, x > 0}

def is_ascending (s : List ℕ) : Prop := s = s.sorted (· ≤ ·)

def has_average (s : List ℕ) (avg : ℚ) : Prop :=
  (s.sum : ℚ) / s.length = avg

def has_median (s : List ℕ) (med : ℚ) : Prop :=
  let mid_idx := s.length / 2
  (s[mid_idx - 1] + s[mid_idx]) / 2 = med

def has_standard_deviation (s : List ℕ) (σ : ℚ) : Prop :=
  let mean := (s.sum : ℚ) / s.length
  let variance := (s.map (λ x, ((x : ℚ) - mean) ^ 2)).sum / s.length
  Real.sqrt variance = σ

theorem data_set_unique :
  ∀ (a b c d : ℕ), {a, b, c, d}.card = 4 →
  a > 0 → b > 0 → c > 0 → d > 0 →
  is_ascending [a, b, c, d] →
  has_average [a, b, c, d] 2 →
  has_median [a, b, c, d] 2 →
  has_standard_deviation [a, b, c, d] 1 →
  [a, b, c, d] = [1, 2, 2, 3] :=
by sorry

end data_set_unique_l309_309442


namespace symmetry_center_l309_309536

-- Function f definition
def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + π / 6)

-- Conditions
axiom ω_pos : ∀ ω : ℝ, ω > 0
axiom period_condition : ∀ (ω : ℝ), ω > 0 → (2 * π / ω = π)

-- Mathematical equivalent proof problem
theorem symmetry_center (ω : ℝ) (hω : ω > 0) (h : 2 * π / ω = π) : 
  ∃ x : ℝ, f ω x = 0 ∧ x = 5 * π / 12 :=
by
  sorry

end symmetry_center_l309_309536


namespace points_on_line_l309_309929

theorem points_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (2 * t + 2) / t,
      y := (2 * t - 2) / t
  in x + y = 4 :=
by
  simp only [gt, lt] -- Handle non-zero t
  sorry -- Proof

end points_on_line_l309_309929


namespace vietnam_olympiad_2007_l309_309006

theorem vietnam_olympiad_2007 (b : ℝ) (hb : 0 < b) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f(x + y) = f(x) * 3^(b + f(y) - 1) + b^x * 3^(b^3 + f(y) - 1) - b^(x + y)) :
  (f = λ x, -b^x) ∨ (f = λ x, 1 - b^x) :=
sorry

end vietnam_olympiad_2007_l309_309006


namespace sum_a_n_888_l309_309924

noncomputable def a_n (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 17 = 0 then 18
  else if n % 15 = 0 ∧ n % 17 = 0 then 21
  else 0

theorem sum_a_n_888 :
  (∑ n in Finset.range 3000, a_n n.succ) = 888 :=
by
  sorry

end sum_a_n_888_l309_309924


namespace inv_256_mod_89_l309_309948

-- Given definition
def sixteen_inv_mod_89 : ℤ := 28  -- given condition 16^-1 ≡ 28 (mod 89)

-- Math proof statement
theorem inv_256_mod_89 :
  let sixteen := 16
  let two_fifty_six := sixteen ^ 2
  (two_fifty_six : ℤ)⁻¹ ≡ 56 [MOD 89] :=
sorry

end inv_256_mod_89_l309_309948


namespace original_gross_profit_percentage_l309_309409

theorem original_gross_profit_percentage 
  (C : ℝ) -- Cost of the product
  (h1 : 1.15 * C = 92) -- New selling price equation implying 15% gross profit increase
  (h2 : 88 - C = 8) -- Original gross profit in dollar terms
  : ((88 - C) / C) * 100 = 10 := 
sorry

end original_gross_profit_percentage_l309_309409


namespace children_count_125_l309_309517

def numberOfChildren (a : ℕ) : Prop :=
  a % 8 = 5 ∧ a % 10 = 7 ∧ 100 ≤ a ∧ a ≤ 150

theorem children_count_125 : ∃ a : ℕ, numberOfChildren a ∧ a = 125 := by
  use 125
  unfold numberOfChildren
  apply And.intro
  apply And.intro
  · norm_num
  · norm_num
  · split
  repeat {norm_num}
  sorry

end children_count_125_l309_309517


namespace max_product_is_64_l309_309941

noncomputable def max_product (a : ℕ → ℝ) : ℕ :=
if h1 : (a 1 + a 3 = 10) ∧ (a 2 + a 4 = 5)
then max (a 1 * a 2 * a 3) (a 1 * a 2 * a 3 * a 4)
else 0

theorem max_product_is_64 (a : ℕ → ℝ) :
  (a 1 + a 3 = 10) ∧ (a 2 + a 4 = 5) → max_product a = 64 :=
by
  sorry

end max_product_is_64_l309_309941


namespace james_total_cost_l309_309245

-- Definitions based on conditions
def total_cost (T : ℝ) (insurance_coverage : ℝ) (out_of_pocket : ℝ) : Prop :=
  insurance_coverage = 0.80 ∧ 
  out_of_pocket = 60 ∧
  T = (out_of_pocket / (1 - insurance_coverage))

-- Lean statement of the problem
theorem james_total_cost (T : ℝ)
  (h_insurance_coverage : ∀ insurance_coverage, insurance_coverage = 0.80)
  (h_out_of_pocket : ∀ out_of_pocket, out_of_pocket = 60)
  : T = 300 :=
begin
  -- Placeholder proof
  sorry
end

end james_total_cost_l309_309245


namespace prob_X_greater_than_2_l309_309934

open ProbabilityTheory MeasureTheory

noncomputable def normalDist : Measure ℝ := measure_space.measure (continuous_probability_space.gaussian 0 σ^2)

theorem prob_X_greater_than_2 (σ : ℝ) (h1 : 0 < σ)
  (h2 : ∫ s in Ico (-2 : ℝ) 0, normalDist s = 0.4) : ∫ s in Ioi (2 : ℝ), normalDist s = 0.1 :=
sorry

end prob_X_greater_than_2_l309_309934


namespace quadrilateral_proof_l309_309047

theorem quadrilateral_proof (A B C D M N K L : Type) 
  (h_cyclic : ∀ (P Q R S : Type), inscribed P Q R S)
  (h_tangent_M : is_tangent A C M)
  (h_tangent_N : is_tangent B D N)
  (h_bisector_K : angle_bisector_intersection A C K)
  (h_bisector_L : angle_bisector_intersection B D L)
  (h_condition : |AB| * |CD| = |AD| * |BC|) :
  (M ∈ line_through B D) ↔ (N ∈ line_through A C) ↔ (K ∈ line_through B D) ↔ (L ∈ line_through A C) :=
sorry

end quadrilateral_proof_l309_309047


namespace find_m_if_orthogonal_l309_309581

theorem find_m_if_orthogonal:
  (let OA := (-1, 2) in let OB := (3, m) in OA.1 * OB.1 + OA.2 * OB.2 = 0) → m = (3 / 2) :=
by
  sorry

end find_m_if_orthogonal_l309_309581


namespace quadratic_equal_roots_k_value_l309_309975

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∃ x : ℝ, x^2 + k - 3 = 0 ∧ (0^2 - 4 * 1 * (k - 3) = 0)) → k = 3 := 
by
  sorry

end quadratic_equal_roots_k_value_l309_309975


namespace suraj_next_innings_runs_l309_309318

variable (A R : ℕ)

def suraj_average_eq (A : ℕ) : Prop :=
  A + 8 = 128

def total_runs_eq (A R : ℕ) : Prop :=
  9 * A + R = 10 * 128

theorem suraj_next_innings_runs :
  ∃ A : ℕ, suraj_average_eq A ∧ ∃ R : ℕ, total_runs_eq A R ∧ R = 200 := 
by
  sorry

end suraj_next_innings_runs_l309_309318


namespace find_number_l309_309806

-- Define the condition given in the problem
def condition (x : ℕ) : Prop :=
  x / 5 + 6 = 65

-- Prove that the solution satisfies the condition
theorem find_number : ∃ x : ℕ, condition x ∧ x = 295 :=
by
  -- Skip the actual proof steps
  sorry

end find_number_l309_309806


namespace vertical_slip_distance_l309_309040

-- Definitions
def ladder_length : ℝ := 14
def initial_horizontal_distance : ℝ := 5
def final_horizontal_distance : ℝ := 10.658966865741546

-- Theorem stating the vertical slip of the ladder's top
theorem vertical_slip_distance :
  let initial_vertical_distance := Real.sqrt (ladder_length^2 - initial_horizontal_distance^2)
  let final_vertical_distance := Real.sqrt (ladder_length^2 - final_horizontal_distance^2)
  let slip_distance := initial_vertical_distance - final_vertical_distance
  slip_distance ≈ 4.00392512594753 := 
by
  sorry

end vertical_slip_distance_l309_309040


namespace product_of_first_2016_terms_l309_309185

noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := (1 + a n) / (1 - a n)

theorem product_of_first_2016_terms : ∏ i in Finset.range 2016, a i = 1 :=
sorry

end product_of_first_2016_terms_l309_309185


namespace shape_factor_cylinder_minimize_shape_factor_l309_309366

-- Part 1: Shape factor for a cylindrical building
theorem shape_factor_cylinder (R H : ℝ) (hR : R > 0) (hH : H > 0) :
  let F0 := π * R^2 + 2 * π * R * H
  let V0 := π * R^2 * H
  S = F0 / V0
  S = (2 * H + R) / (H * R) := 
by sorry

-- Part 2: Minimizing shape factor for the dormitory building
theorem minimize_shape_factor (f T : ℝ) (hf : f = 18) (hT : T = 10000) :
  ∃ (n : ℝ), n ≈ 6 ∧ 
  let S := sqrt(f * n / T) + 1 / (3 * n)
  is_minimum (S) := 
by sorry

end shape_factor_cylinder_minimize_shape_factor_l309_309366


namespace decimal_75_to_base_4_digits_count_l309_309623

theorem decimal_75_to_base_4_digits_count :
  Nat.numDigits 4 75 = 4 :=
by
  sorry

end decimal_75_to_base_4_digits_count_l309_309623


namespace smallest_positive_sum_l309_309227

structure ArithmeticSequence :=
  (a_n : ℕ → ℤ)  -- The sequence is an integer sequence
  (d : ℤ)        -- The common difference of the sequence

def sum_of_first_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a_n 1 + seq.a_n n)) / 2  -- Sum of first n terms

def condition (seq : ArithmeticSequence) : Prop :=
  (seq.a_n 11 < -1 * seq.a_n 10)

theorem smallest_positive_sum (seq : ArithmeticSequence) (H : condition seq) :
  ∃ n, sum_of_first_n seq n > 0 ∧ ∀ m < n, sum_of_first_n seq m ≤ 0 → n = 19 :=
sorry

end smallest_positive_sum_l309_309227


namespace fraction_sum_5625_l309_309782

theorem fraction_sum_5625 : 
  ∃ (a b : ℕ), 0.5625 = (9 : ℚ) / 16 ∧ (a + b = 25) := 
by 
  sorry

end fraction_sum_5625_l309_309782


namespace timeTakenByBobIs30_l309_309632

-- Define the conditions
def timeTakenByAlice : ℕ := 40
def fractionOfTimeBobTakes : ℚ := 3 / 4

-- Define the statement to be proven
theorem timeTakenByBobIs30 : (fractionOfTimeBobTakes * timeTakenByAlice : ℚ) = 30 := 
by
  sorry

end timeTakenByBobIs30_l309_309632


namespace center_is_seven_l309_309465

-- Define the 3x3 grid with indices.
structure Grid3x3 where
  data : Fin 3 × Fin 3 → Fin 9

-- Conditions
def consecutive_adjacent (g : Grid3x3) : Prop :=
  ∀ i j, (i, j) ∈ Finset.Union (Finset.finRange 9) 
    → let n := g.data (i, j)
    in ∃ di dj, abs di + abs dj = 1 ∧ g.data (i + di, j + dj) = n + 1

def corners_sum_to_22 (g : Grid3x3) : Prop :=
  g.data (0, 0) + g.data (0, 2) + g.data (2, 0) + g.data (2, 2) = 22

-- Given conditions and the statement to prove
theorem center_is_seven (g : Grid3x3) 
  (h1 : consecutive_adjacent g)
  (h2 : corners_sum_to_22 g) : 
  g.data (1, 1) = 6 := -- Middle coordinate in a zero-based index
sorry

end center_is_seven_l309_309465


namespace essays_fill_pages_l309_309251

theorem essays_fill_pages :
  ∀ (johnny madeline timothy total_words pages_per_page : ℕ),
  johnny = 150 →
  madeline = 2 * johnny →
  timothy = madeline + 30 →
  pages_per_page = 260 →
  total_words = johnny + madeline + timothy →
  total_words / pages_per_page = 3 :=
by
  intros johnny madeline timothy total_words pages_per_page h_johnny h_madeline h_timothy h_pages h_total_words
  rw [h_johnny, h_madeline, h_timothy, h_pages, h_total_words]
  sorry

end essays_fill_pages_l309_309251


namespace max_discount_l309_309861

noncomputable def cost_price := 200
noncomputable def marked_price := 360
noncomputable def required_profit := 0.2 * cost_price

theorem max_discount : 
  ∃ (x : ℝ), (marked_price - x - cost_price ≥ required_profit) ∧ (marked_price - x = 120) :=
by
  sorry

end max_discount_l309_309861


namespace coeff_m6n6_in_m_plus_n_pow_12_l309_309817

theorem coeff_m6n6_in_m_plus_n_pow_12 : 
  (∃ c : ℕ, (m + n)^12 = c * m^6 * n^6 + ...) → c = 924 := by
sorry

end coeff_m6n6_in_m_plus_n_pow_12_l309_309817


namespace bottles_left_on_shelf_l309_309351

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l309_309351


namespace negation_of_existence_statement_l309_309315

theorem negation_of_existence_statement :
  ¬ (∃ x : ℝ, 2 ^ x ≥ 1) ↔ ∀ x : ℝ, 2 ^ x < 1 :=
by
  sorry

end negation_of_existence_statement_l309_309315


namespace small_n_for_sum_l309_309794

def a : ℕ → ℝ
| 1     := 1.5
| (n+2) := 1 / (n + 2)^2 - 1

noncomputable def S (n : ℕ) : ℝ :=
1.5 + (∑ k in finset.range (n - 1 + 1), 1 / (k + 1)^2 - 1)

theorem small_n_for_sum :
  ∃ n : ℕ, |S n - 2.25| < 0.01 ∧ n = 100 :=
begin
  sorry
end

end small_n_for_sum_l309_309794


namespace find_specific_number_l309_309438

theorem find_specific_number :
  ∃ x : ℕ, 
    x % 2 = 1 ∧
    x % 5 = 2 ∧
    x % 7 = 3 ∧
    x % 9 = 4 ∧
    ∀ y : ℕ,
    (y % 2 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3 ∧ y % 9 = 4) → y ≥ x :=
begin
  use 157,
  sorry
end

end find_specific_number_l309_309438


namespace solve_fraction_problem_l309_309504

noncomputable def x_value (a b c d : ℤ) : ℝ :=
  (a + b * Real.sqrt c) / d

theorem solve_fraction_problem (a b c d : ℤ) (h1 : x_value a b c d = (5 + 5 * Real.sqrt 5) / 4)
  (h2 : (4 * x_value a b c d) / 5 - 2 = 5 / x_value a b c d) :
  (a * c * d) / b = 20 := by
  sorry

end solve_fraction_problem_l309_309504


namespace green_eyed_brunettes_percentage_l309_309456

noncomputable def green_eyed_brunettes_proportion (a b c d : ℕ) 
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7) 
  (h3 : c / (c + d) = 0.1) : Prop :=
  d / (a + b + c + d) = 0.54

-- The main theorem to be proved
theorem green_eyed_brunettes_percentage (a b c d : ℕ)
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7)
  (h3 : c / (c + d) = 0.1) : 
  green_eyed_brunettes_proportion a b c d h1 h2 h3 := 
sorry

end green_eyed_brunettes_percentage_l309_309456


namespace eleventh_number_is_175_l309_309452

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum
-- The digits_sum function calculates the sum of digits of a given natural number n in base 10

def eleventh_number_with_digits_sum_13 : ℕ :=
175 -- This is the correct answer based on the problem solution

theorem eleventh_number_is_175 :
  ∃! n : ℕ, (digits_sum n = 13 ∧
             ∃ l : List ℕ, (l.nth 10 = some n ∧
                             (∀ m, digits_sum m = 13 → List.nthLe l (l.indexOf m) sorry = m) ∧
                             (∀ m1 m2, l.indexOf m1 < l.indexOf m2 ↔ m1 < m2))) →
             n = eleventh_number_with_digits_sum_13 :=
sorry

end eleventh_number_is_175_l309_309452


namespace find_X_l309_309973

variable (A : Matrix (Fin 2) (Fin 2) ℝ) 
variable (B : Matrix (Fin 2) (Fin 1) ℝ)
variable (X : Matrix (Fin 2) (Fin 1) ℝ)

theorem find_X (hA : A = !![ [2, 1], [3, 2] ]) (hB : B = !![ [4], [7] ]) 
  (hX : X = !![ [1], [2] ]) : A.mul X = B :=
by
  sorry

end find_X_l309_309973


namespace transformed_polynomial_l309_309684

theorem transformed_polynomial 
  {a b c : ℝ} 
  (h1 : Polynomial.eval a (Polynomial.C 1 * X^3 + Polynomial.C (-6) * X + Polynomial.C 5) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 * X^3 + Polynomial.C (-6) * X + Polynomial.C 5) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1 * X^3 + Polynomial.C (-6) * X + Polynomial.C 5) = 0) :
  ∃ q : Polynomial ℝ, 
    q.monic ∧
    q.eval (a - 3) = 0 ∧ 
    q.eval (b - 3) = 0 ∧ 
    q.eval (c - 3) = 0 ∧ 
    q = (Polynomial.C 1 * X^3 + Polynomial.C 9 * X^2 + Polynomial.C 21 * X + Polynomial.C 14) :=
by
  sorry

end transformed_polynomial_l309_309684


namespace difference_between_largest_shares_l309_309059

noncomputable def x : ℕ := 500
def V : ℕ := 3 * x
def A : ℕ := 4 * x
def R : ℕ := 7 * x
def Difference : ℕ := R - A

theorem difference_between_largest_shares :
  V = 1500 → A = 2000 → Difference = 1500 :=
by
  intros hV hA
  rw [V, A] at hV hA
  rw [Difference, R]
  exact Nat.sub_eq_of_eq_add
    (show 3500 = 2000 + 1500 by sorry)  -- Proof required for the exact value

end difference_between_largest_shares_l309_309059


namespace negation_of_proposition_l309_309390

theorem negation_of_proposition (a b c : ℝ) : (a = 0 ∨ b = 0 ∨ c = 0) → (a * b * c = 0) :=
by 
  intros h
  cases h with ha hb hc
  -- ha case
  simp [ha],
  -- hb case
  cases hb with hb hc,
  simp [hb],
  -- hc case
  simp [hc],
  sorry -- this line is just for demonstration purpose, you need to complete the proof

end negation_of_proposition_l309_309390


namespace five_person_lineup_l309_309231

theorem five_person_lineup : 
  let total_ways := Nat.factorial 5
  let invalid_first := Nat.factorial 4
  let invalid_last := Nat.factorial 4
  let valid_ways := total_ways - (invalid_first + invalid_last)
  valid_ways = 72 :=
by
  sorry

end five_person_lineup_l309_309231


namespace unique_function_property_l309_309117

theorem unique_function_property (f : ℕ → ℕ) (k : ℕ) (hk : k > 0) 
    (H1 : ∀ m n : ℕ, m > 0 → n > 0 → f (m * n) = f m * f n)
    (H2 : ∀ n : ℕ, n > 0 → (iterate (λ x, f x) (n^k) n) = n) :
  ∀ x : ℕ, x > 0 → f x = x := 
sorry

end unique_function_property_l309_309117


namespace decreasing_power_function_l309_309333

theorem decreasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(m^2 + m - 1) < (m^2 - m - 1) * (x + 1) ^ (m^2 + m - 1)) →
  m = -1 :=
sorry

end decreasing_power_function_l309_309333


namespace children_count_125_l309_309516

def numberOfChildren (a : ℕ) : Prop :=
  a % 8 = 5 ∧ a % 10 = 7 ∧ 100 ≤ a ∧ a ≤ 150

theorem children_count_125 : ∃ a : ℕ, numberOfChildren a ∧ a = 125 := by
  use 125
  unfold numberOfChildren
  apply And.intro
  apply And.intro
  · norm_num
  · norm_num
  · split
  repeat {norm_num}
  sorry

end children_count_125_l309_309516


namespace spadesuit_example_l309_309093

-- Define the operation spadesuit
def spadesuit (a b : ℤ) : ℤ := abs (a - b)

-- Define the specific instance to prove
theorem spadesuit_example : spadesuit 2 (spadesuit 4 7) = 1 :=
by
  sorry

end spadesuit_example_l309_309093


namespace max_profit_condition_max_selling_price_l309_309348

def avg_monthly_profit (a x : ℝ) : ℝ := 5 * a * (1 + 4 * x - x^2 - 4 * x^3)

theorem max_profit_condition (a : ℝ) (h1 : 0 < 1) : 
  ∃ x : ℝ, (0 < x ∧ x < 1) ∧ (avg_monthly_profit a x = avg_monthly_profit a (1 / 2)) := sorry

theorem max_selling_price (a : ℝ) : 
  (20 * (1 + (1/2)) = 30) := by norm_num

end max_profit_condition_max_selling_price_l309_309348


namespace perpendicular_distance_between_stripes_l309_309444

noncomputable def distance_between_stripes : ℝ :=
  let area : ℝ := 20 * 60 in
  let stripe_length : ℝ := 50 in
  area / stripe_length

theorem perpendicular_distance_between_stripes (curb_distance stripe_length curb_length : ℝ) 
  (h_curb : curb_distance = 60) 
  (h_stripe : stripe_length = 50) 
  (h_curb_stripe : curb_length = 20) : 
  distance_between_stripes = 24 :=
by
  rw [distance_between_stripes, h_curb_stripe, h_curb, mul_comm curb_length curb_distance, mul_comm stripe_length, ←mul_assoc, mul_div_cancel];
  simp only [stripe_length];
  norm_num;
  sorry

end perpendicular_distance_between_stripes_l309_309444


namespace sum_of_coefficients_l309_309589

theorem sum_of_coefficients :
  let p := (3 * x - 1)^7,
      coeffs := (a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 +
                a_2 * x^2 + a_1 * x + a_0)
  in coeffs.eval (λ (a: ℕ), 1) = 128 :=
by
  -- Begin proof here
  sorry

end sum_of_coefficients_l309_309589


namespace derivative_at_zero_l309_309685

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + x)

-- Statement of the problem: The derivative of f at 0 is 1
theorem derivative_at_zero : deriv f 0 = 1 := 
  sorry

end derivative_at_zero_l309_309685


namespace triangle_AHB_area_l309_309241

theorem triangle_AHB_area
  (ABC : Type)
  [triangle ABC]
  (G H : ABC → ABC)
  (is_midpoint_G : ∀ (A B : ABC), G A = midpoint A B)
  (is_midpoint_H : ∀ (B C : ABC), H B = midpoint B C)
  (area_ABC : has_area ABC 36)
  : has_area (triangle_vertices A H B) 9 :=
sorry

end triangle_AHB_area_l309_309241


namespace initial_volume_of_solution_l309_309843

theorem initial_volume_of_solution 
  (V : ℝ) 
  (h1 : V > 0)
  (h2 : 0.1 * V = 0.08 * (V + 14)) : 
  V = 56 :=
begin
  sorry
end

end initial_volume_of_solution_l309_309843


namespace combinatorial_proof_l309_309526

noncomputable def combinatorial_identity (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) : ℕ :=
  let summation_term (i : ℕ) := Nat.choose k i * Nat.choose n (m - i)
  List.sum (List.map summation_term (List.range (k + 1)))

theorem combinatorial_proof (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) :
  combinatorial_identity n m k h1 h2 h3 = Nat.choose (n + k) m :=
sorry

end combinatorial_proof_l309_309526


namespace probability_alex_paired_with_jordan_l309_309607

theorem probability_alex_paired_with_jordan :
  ∀ (students : Finset ℕ) (Alex Jordan : ℕ),
    (students.card = 40) →
    (Alex ∉ students) →
    (Jordan ∉ students) →
    (Alex ≠ Jordan) →
    (∀ s ∈ students, s ≠ Alex ∧ s ≠ Jordan) →
    (∃ pairs : Finset (Finset ℕ), ∀ pair ∈ pairs, pair.card = 2 ∧ (∀ p ∈ pair, p ∈ students ∪ {Alex, Jordan})) →
    (∃! pair ∈ pairs, Alex ∈ pair ∧ Jordan ∈ pair) →
    (∀ pair ∈ pairs, Alex ∈ pair → Jordan ∈ pair) →
    (students.card.choose 1 = 1 / 39) :=
begin
  sorry
end

end probability_alex_paired_with_jordan_l309_309607


namespace vova_cannot_prevent_pasha_from_winning_l309_309716

theorem vova_cannot_prevent_pasha_from_winning :
  ∀ (initial_mass : ℝ), 
  initial_mass > 0 → 
  (∃ pasha_strategy vova_strategy, (pasha_strategy.follows_rules ∧ vova_strategy.follows_rules) →
  pasha_wins pasha_strategy vova_strategy initial_mass) :=
by 
  intros initial_mass h_mass_pos
  have strategy_def : ∃ pasha_strategy vova_strategy,
    (pasha_strategy.follows_rules ∧ vova_strategy.follows_rules) → pasha_wins pasha_strategy vova_strategy initial_mass,
  {
    -- Assume pasha_strategy and vova_strategy are valid strategies under the rules.
    sorry
  }
  exact strategy_def

end vova_cannot_prevent_pasha_from_winning_l309_309716


namespace junior_score_proof_l309_309606

noncomputable def class_total_score (total_students : ℕ) (average_class_score : ℕ) : ℕ :=
total_students * average_class_score

noncomputable def number_of_juniors (total_students : ℕ) (percent_juniors : ℕ) : ℕ :=
percent_juniors * total_students / 100

noncomputable def number_of_seniors (total_students juniors : ℕ) : ℕ :=
total_students - juniors

noncomputable def total_senior_score (seniors average_senior_score : ℕ) : ℕ :=
seniors * average_senior_score

noncomputable def total_junior_score (total_score senior_score : ℕ) : ℕ :=
total_score - senior_score

noncomputable def junior_score (junior_total_score juniors : ℕ) : ℕ :=
junior_total_score / juniors

theorem junior_score_proof :
  ∀ (total_students: ℕ) (percent_juniors average_class_score average_senior_score : ℕ),
  total_students = 20 →
  percent_juniors = 15 →
  average_class_score = 85 →
  average_senior_score = 84 →
  (junior_score (total_junior_score (class_total_score total_students average_class_score)
                                    (total_senior_score (number_of_seniors total_students (number_of_juniors total_students percent_juniors))
                                                        average_senior_score))
                (number_of_juniors total_students percent_juniors)) = 91 :=
by
  intros
  sorry

end junior_score_proof_l309_309606


namespace not_a_function_l309_309824

theorem not_a_function (angle_sine : ℝ → ℝ) 
                       (side_length_area : ℝ → ℝ) 
                       (sides_sum_int_angles : ℕ → ℝ)
                       (person_age_height : ℕ → Set ℝ) :
  (∃ y₁ y₂, y₁ ∈ person_age_height 20 ∧ y₂ ∈ person_age_height 20 ∧ y₁ ≠ y₂) :=
by {
  sorry
}

end not_a_function_l309_309824


namespace f_increasing_f_range_l309_309181

-- Problem Statement I
theorem f_increasing (f : ℝ → ℝ) (h_f : ∀ x, f(x) = 1 - 2 / (3^x + 1)) :
  ∀ (x1 x2 : ℝ), x1 < x2 → f(x1) < f(x2) :=
sorry

-- Problem Statement II
theorem f_range (f : ℝ → ℝ) (h_f : ∀ x, f(x) = 1 - 2 / (3^x + 1)) :
  Set.Icc (f (-1)) (f 2) = Set.Icc (-1 / 2) (4 / 5) :=
sorry

end f_increasing_f_range_l309_309181


namespace range_f_l309_309344

def f (x : ℝ) : ℝ := cos (2 * x) + 2 * |sin x|

theorem range_f :
  set.range (λ x, f x) = {y | 1 ≤ y ∧ y ≤ (3 / 2)} :=
sorry

end range_f_l309_309344


namespace sqrt_product_l309_309892

theorem sqrt_product :
  real.sqrt 75 * real.sqrt 48 * real.sqrt 12 = 120 * real.sqrt 3 := 
sorry

end sqrt_product_l309_309892


namespace parallel_MN_PQ_l309_309658

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l309_309658


namespace percentage_of_cone_volume_filled_with_water_l309_309847

theorem percentage_of_cone_volume_filled_with_water
  (r h : ℝ) : 
  (let V := (1 / 3) * real.pi * r^2 * h in
   let V_water := (1 / 3) * real.pi * ((3 / 4) * r)^2 * ((3 / 4) * h) in
   (V_water / V) * 100 = 42.1875) :=
by
  sorry

end percentage_of_cone_volume_filled_with_water_l309_309847


namespace value_of_f2012_l309_309643

theorem value_of_f2012 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 → f(x) + 2 * f((x + 2010) / (x - 1)) = 4020 - x) : f(2012) = 2010 :=
sorry

end value_of_f2012_l309_309643


namespace als_initial_portion_l309_309450

theorem als_initial_portion (a b c : ℝ)
  (h1 : a + b + c = 1200)
  (h2 : a - 150 + 3 * b + 3 * c = 1800) :
  a = 825 :=
sorry

end als_initial_portion_l309_309450


namespace part_I_part_II_l309_309544

-- Given an arithmetic sequence \{a_n\} with the sum of the first n terms Sn = -a_n - (1/2)^(n-1) + 2
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- a_n = (n / 2^n)
theorem part_I (h : ∀ n, S n = -a n - (1 / 2)^(n-1) + 2) :
  (∀ n, a n = n / 2^n) := sorry

-- Compare T_n = ∑_{k=1}^{n} ((k+1) / k) a_k with 5n / (2n + 1)
variable (c : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- c_n = (n + 1) a_n / n
theorem part_II (h : ∀ n, S n = -a n - (1 / 2)^(n-1) + 2)
               (h1 : ∀ n, c n = ((n + 1) / n) * a n)
               (h2 : ∀ n, T n = ∑ i in finset.range (n + 1), c (i + 1)) :
  ∀ n, (T n) = 3 - (n + 3) / 2^n ∧ T n < 5 * n / (2 * n + 1) := sorry

end part_I_part_II_l309_309544


namespace height_comparison_cases_l309_309798

variable (p q : ℕ) (hp : p > 1) (hq : q > 1)
variable (heights : Matrix (Fin p) (Fin q) ℕ)

def shortest_in_row (i : Fin p) : ℕ := 
  Fintype.min (λ j => heights i j) sorry

def tallest_of_shortest_in_rows : ℕ := 
  Fintype.max (λ i => shortest_in_row p q heights i) sorry

def tallest_in_column (j : Fin q) : ℕ :=
  Fintype.max (λ i => heights i j) sorry

def shortest_of_tallest_in_columns : ℕ :=
  Fintype.min (λ j => tallest_in_column p q heights j) sorry

theorem height_comparison_cases :
  let a := tallest_of_shortest_in_rows p q heights
  let b := shortest_of_tallest_in_columns p q heights
  (a = b ∨ a < b) ∧ ¬ (a > b) :=
by
  sorry

end height_comparison_cases_l309_309798


namespace solve_system_l309_309187

theorem solve_system :
  ∃ (x y z : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y - 4 * z = 10 ∧ 2 * x + y + 3 * z = 1.25 :=
by
  sorry

end solve_system_l309_309187


namespace number_removed_to_achieve_average_l309_309866

theorem number_removed_to_achieve_average (s : Finset ℕ) (h₁ : s = Finset.range 16) (h₂ : 15 ∈ s) :
  (∑ x in s \ {15}, x) / (s.card - 1) = 7.5 :=
by
  sorry

end number_removed_to_achieve_average_l309_309866


namespace radius_of_inscribed_circle_l309_309729

theorem radius_of_inscribed_circle 
  (r O A B : Type) [metric_space r] [metric_space O] [metric_space A] [metric_space B]
  (radius : ℝ) (sector : set (point r)) :
  (∃ (C : point r), sector = metric_ball C 5 ∧
    tangent_at C O ∧ tangent_at C A ∧ tangent_at C B ∧
    radius = 5 ∧
    sector.is_sector ∧
    area sector = π * radius^2 / 3)
    → r = (5 * (sqrt 3 - 1)) / 2 := 
sorry

end radius_of_inscribed_circle_l309_309729


namespace part1_part2_l309_309311

theorem part1 :
  3 < real.cbrt 30 ∧ real.cbrt 30 < 4 →
  ⌊real.cbrt 30⌋ = 3 ∧ real.cbrt 30 - 3 < 1 := 
by
  assume h1 : 3 < real.cbrt 30 ∧ real.cbrt 30 < 4,
  sorry

theorem part2 (m : ℕ) (h1 : 2 < real.cbrt 20 ∧ real.cbrt 20 < 3) (h2: m = 4) : 
  (∃ x : ℤ, (x + 1)^2 = m) :=
by
  assume h,
  use [1, -3],
  sorry

end part1_part2_l309_309311


namespace dot_product_w_l309_309203

noncomputable def vec_norm (v : ℝ) : ℝ := ‖v‖

def v : ℝ := 3 -- Placeholder for vector, assuming it's in the real numbers for simplicity
noncomputable def w : ℝ := 2 * v

theorem dot_product_w : w * w = 36 := by
  have h₁ : vec_norm v = 3 := by sorry
  have h₂ : vec_norm w = 2 * vec_norm v := by sorry
  have h₃ : vec_norm w = 6 := by sorry
  have h₄ : w * w = (vec_norm w) ^ 2 := by sorry
  show w * w = 36 := by sorry

end dot_product_w_l309_309203


namespace distance_rowed_downstream_l309_309838

def speed_of_boat_still_water : ℝ := 70 -- km/h
def distance_upstream : ℝ := 240 -- km
def time_upstream : ℝ := 6 -- hours
def time_downstream : ℝ := 5 -- hours

theorem distance_rowed_downstream :
  let V_b := speed_of_boat_still_water
  let V_upstream := distance_upstream / time_upstream
  let V_s := V_b - V_upstream
  let V_downstream := V_b + V_s
  V_downstream * time_downstream = 500 :=
by
  sorry

end distance_rowed_downstream_l309_309838


namespace possible_values_of_ab_plus_ac_plus_bc_l309_309262

-- Definitions and conditions
variables {a b c : ℝ} 

-- The main theorem statement
theorem possible_values_of_ab_plus_ac_plus_bc (h : a + b + c = 1) : 
  ∃ (S : set ℝ), S = (-∞, 1/2] ∧ (ab + ac + bc) ∈ S := 
sorry

end possible_values_of_ab_plus_ac_plus_bc_l309_309262


namespace collinear_points_find_fx_expression_l309_309170

theorem collinear_points_find_fx_expression (A B C : Type)
    (l : Type) [collinear A B C l]
    (OA OB OC : ℝ → ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, OA x = (f x + 2 * (deriv f 1) * x) * OB x - (log x) * OC x) : 
    f = λ x, log x - (2 * x / 3) + 1 :=
by
  sorry

end collinear_points_find_fx_expression_l309_309170


namespace max_of_set_partition_l309_309024

theorem max_of_set_partition (K : Finset ℕ) (hK : 3 ≤ K.card)
  (A B : Finset ℕ) (hA : A ≠ ∅) (hB : B ≠ ∅)
  (hP : ∀ a ∈ A, ∀ b ∈ B, ∃ m : ℕ, a * b + 1 = m ^ 2) :
  ∃ m : ℕ, K.max' (by linarith [hK]) ≥ m :=
begin
  let min_AB := min A.card B.card,
  let bound := (2 + Real.sqrt 3) ^ (min_AB - 1),
  use ⌊bound⌋ + 1,
  sorry
end

end max_of_set_partition_l309_309024


namespace fraction_not_covered_correct_l309_309854

def area_floor : ℕ := 64
def width_rug : ℕ := 2
def length_rug : ℕ := 7
def area_rug := width_rug * length_rug
def area_not_covered := area_floor - area_rug
def fraction_not_covered := (area_not_covered : ℚ) / area_floor

theorem fraction_not_covered_correct :
  fraction_not_covered = 25 / 32 :=
by
  -- Proof goes here
  sorry

end fraction_not_covered_correct_l309_309854


namespace regression_decrease_l309_309540

theorem regression_decrease (x : ℝ) : let y = 2 - 1.5 * x 
                                      let y' = 2 - 1.5 * (x + 1) 
                                      (y' - y) = -1.5 := 
by 
  sorry

end regression_decrease_l309_309540


namespace sin_tan_relation_l309_309529

theorem sin_tan_relation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -(2 / 5) := 
sorry

end sin_tan_relation_l309_309529


namespace hyperbola_equation_l309_309038

-- Problem conditions as definitions
def center : ℝ × ℝ := (0, 0)

def focus : ℝ × ℝ := (0, Real.sqrt 3)

def vertex_distance : ℝ := Real.sqrt 3 - 1

-- Goal: Prove the equation of the hyperbola
theorem hyperbola_equation : 
  (∃ a b c : ℝ, c = Real.sqrt 3 ∧ a = 1 ∧ b^2 = c^2 - a^2 ∧ 
  ∀ x y : ℝ, y^2 - (x^2 / b) = 1) := 
sorry

end hyperbola_equation_l309_309038


namespace problem_part1_1_problem_part2_l309_309179

noncomputable def f (x : ℝ) : ℝ := Float.cos x

theorem problem_part1_1
    (A : ℝ) (ϕ : ℝ) (ω : ℝ) 
    (A_pos : A > 0) (ω_pos : ω > 0) (ϕ_range : 0 < ϕ ∧ ϕ < π)
    (f_max : A = 1) (f_period : ω = 1)
    (M : (ℝ × ℝ)) (M_def : M = (0, 1))
  : ∀ x ∈ set.Icc (-π / 3) (2 * π / 3), f x ∈ set.Icc (-1 / 2) 1 :=
sorry

theorem problem_part2
  (A B C : ℝ)
  (A_in_triangle : 0 < A ∧ A < π) 
  (B_in_triangle : 0 < B ∧ B < π) 
  (C_in_triangle : 0 < C ∧ C < π) 
  (triangle_condition : A + B + C = π)
  (cos_A : Float.cos A = 3 / 5)
  (cos_B : Float.cos B = 5 / 13)
  : Float.sin (A + B) = 56 / 65 :=
sorry

end problem_part1_1_problem_part2_l309_309179


namespace kate_bought_wands_l309_309640

theorem kate_bought_wands (price_per_wand : ℕ)
                           (additional_cost : ℕ)
                           (total_money_collected : ℕ)
                           (number_of_wands_sold : ℕ)
                           (total_wands_bought : ℕ) :
  price_per_wand = 60 → additional_cost = 5 → total_money_collected = 130 → 
  number_of_wands_sold = total_money_collected / (price_per_wand + additional_cost) →
  total_wands_bought = number_of_wands_sold + 1 →
  total_wands_bought = 3 := by
  sorry

end kate_bought_wands_l309_309640


namespace pentagon_area_ratio_l309_309364

noncomputable def side_length : ℝ := 1

-- Define points
def M := (side_length / 2, 0)
def N := (side_length / 2, side_length)
def P := (side_length / 2, side_length * 3)
def Q := (1, 0) -- Assuming Q is B
def C := (1, 1)

-- Define areas
def area_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  1/2 * abs((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

def area_MNC : ℝ := area_triangle M N C
def area_NCP : ℝ := area_triangle N C P

def area_pentagon : ℝ := area_MNC + area_NCP

def total_area_squares : ℝ := 3 * side_length ^ 2

theorem pentagon_area_ratio : area_pentagon / total_area_squares = 7 / 12 :=
by
  sorry

end pentagon_area_ratio_l309_309364


namespace hexagon_area_minimum_l309_309682

noncomputable def Q (z : ℂ) : ℂ := z^6 + (3 * real.sqrt 2 + 4) * z^3 + 3 * real.sqrt 2 + 5

theorem hexagon_area_minimum : 
  let roots := {z : ℂ // Q z = 0} in
  ∃ A, (is_min_area_of_hexagon A) ∧ (A = 9 * real.sqrt 3) :=
sorry

end hexagon_area_minimum_l309_309682


namespace rect_coord_eq_and_max_area_l309_309626

noncomputable def C1_rect_eq : (ℝ × ℝ) → Prop := 
  λ M, (∃ y_0 : ℝ, M = (4, y_0))

def on_segment (O M P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t • O.1 + (1 - t) • M.1, t • O.2 + (1 - t) • M.2)

theorem rect_coord_eq_and_max_area :
  (∀ M P : (ℝ × ℝ),
    C1_rect_eq M →
    on_segment (0, 0) M P →
    (∥M.1, M.2∥ * ∥P.1, P.2∥ = 16) →
    (M.1 ≠ 0) →
    (P.1 - 2)^2 + P.2^2 = 4) ∧
  (∃ A B : (ℝ × ℝ),
    A = (1, Real.sqrt 3) ∧
    ((B.1 - 2)^2 + B.2^2 = 4 ∧ B.1 ≠ 0) ∧
    (∃ t : ℝ, 1/2 * abs (0 * ((A.2 - B.2) + 1 * (B.2) + B.1 * (0 - A.2))) = 1 + Real.sqrt 3 + 2 * Real.sqrt 2)) :=
by
  -- Statements as suggested steps included.
  sorry

end rect_coord_eq_and_max_area_l309_309626


namespace total_books_sum_l309_309249

-- Given conditions
def Joan_books := 10
def Tom_books := 38
def Lisa_books := 27
def Steve_books := 45
def Kim_books := 14
def Alex_books := 48

-- Define the total number of books
def total_books := Joan_books + Tom_books + Lisa_books + Steve_books + Kim_books + Alex_books

-- Proof statement
theorem total_books_sum : total_books = 182 := by
  sorry

end total_books_sum_l309_309249


namespace hundred_div_point_two_five_eq_four_hundred_l309_309503

theorem hundred_div_point_two_five_eq_four_hundred : 100 / 0.25 = 400 := by
  sorry

end hundred_div_point_two_five_eq_four_hundred_l309_309503


namespace range_of_a_l309_309995

noncomputable def f (a x : ℝ) := (a - Real.sin x) / Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (π / 6 < x) → (x < π / 3) → (f a x) ≤ (f a (x + ε))) → 2 ≤ a :=
by
  sorry

end range_of_a_l309_309995


namespace nth_term_150_l309_309998

-- Conditions
def a : ℕ := 2
def d : ℕ := 5
def arithmetic_sequence (n : ℕ) : ℕ := a + (n - 1) * d

-- Question and corresponding answer proof
theorem nth_term_150 : arithmetic_sequence 150 = 747 := by
  sorry

end nth_term_150_l309_309998


namespace hunter_strategy_does_not_increase_probability_l309_309026

theorem hunter_strategy_does_not_increase_probability (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1) :
  let p2 := p * p in
  let split_prob := p * (1 - p) in
  let combined_prob := p2 + split_prob in
  combined_prob = p :=
by
  let p := p
  let p2 := p * p
  let split_prob := p * (1 - p)
  let combined_prob := p2 + split_prob
  have : combined_prob = p := sorry
  exact this

end hunter_strategy_does_not_increase_probability_l309_309026


namespace coeff_x5y2_in_expansion_l309_309621

theorem coeff_x5y2_in_expansion:
  let f := (x^2 - x + y)^6
  in coeff f (x^5 * y^2) = -60 :=
sorry

end coeff_x5y2_in_expansion_l309_309621


namespace max_checkers_on_chessboard_l309_309382

open Finset

variable (V : Finset (Fin 8 × Fin 8))
variable [hboard : card V ≤ 64]

def convex_polygon (S : Finset (Fin 8 × Fin 8)) : Prop :=
  ∀ p1 p2 p3 ∈ S, ∠ p1 p2 p3 ≤ 180 ∧
    ∀ p q ∈ S, segment p q ⊆ S

theorem max_checkers_on_chessboard : ∃ S : Finset (Fin 8 × Fin 8), convex_polygon S ∧ card S = 13 :=
by
  sorry

end max_checkers_on_chessboard_l309_309382


namespace expected_value_of_win_l309_309419

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l309_309419


namespace Anne_mom_toothpaste_usage_l309_309797

theorem Anne_mom_toothpaste_usage
  (total_toothpaste : ℕ)
  (dad_usage_per_brush : ℕ)
  (sibling_usage_per_brush : ℕ)
  (num_brushes_per_day : ℕ)
  (total_days : ℕ)
  (total_toothpaste_used : ℕ)
  (M : ℕ)
  (family_use_model : total_toothpaste = total_toothpaste_used + 3 * num_brushes_per_day * M)
  (total_toothpaste_used_def : total_toothpaste_used = 5 * (dad_usage_per_brush * num_brushes_per_day + 2 * sibling_usage_per_brush * num_brushes_per_day))
  (given_values : total_toothpaste = 105 ∧ dad_usage_per_brush = 3 ∧ sibling_usage_per_brush = 1 ∧ num_brushes_per_day = 3 ∧ total_days = 5)
  : M = 2 := by
  sorry

end Anne_mom_toothpaste_usage_l309_309797


namespace smallest_n_satisfies_l309_309792

def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1.5 else 1 / (n^2 - 1)

def sequence_sum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, sequence (k + 1)

theorem smallest_n_satisfies (n : ℕ) (h : n = 100) : 
  | sequence_sum n - 2.25 | < 0.01 :=
by
  sorry

end smallest_n_satisfies_l309_309792


namespace not_all_equal_arithmetic_mean_grid_all_equal_geometric_mean_grid_l309_309293

def arithmetic_mean_grid (f : ℤ × ℤ → ℤ) : Prop :=
∀ (i j : ℤ), f (i, j) = (f (i-1, j) + f (i+1, j) + f (i, j-1) + f (i, j+1)) / 4

theorem not_all_equal_arithmetic_mean_grid :
  ∃ (f : ℤ × ℤ → ℤ), arithmetic_mean_grid f ∧ ∃ (i j k l : ℤ), f (i, j) ≠ f (k, l) :=
sorry

def geometric_mean_grid (f : ℤ × ℤ → ℕ) : Prop :=
∀ (i j : ℤ), f (i, j) = nat.root 4 (f (i-1, j) * f (i+1, j) * f (i, j-1) * f (i, j+1))

theorem all_equal_geometric_mean_grid :
  ∀ (f : ℤ × ℤ → ℕ), geometric_mean_grid f → ∀ (i j k l : ℤ), f (i, j) = f (k, l) :=
sorry

end not_all_equal_arithmetic_mean_grid_all_equal_geometric_mean_grid_l309_309293


namespace sum_of_digits_l309_309275

def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits :
  (Finset.range 2013).sum S = 28077 :=
by 
  sorry

end sum_of_digits_l309_309275


namespace jellybean_removal_l309_309362

theorem jellybean_removal (x : ℕ) : 
  let initial := 37 in
  let final := 23 in
  (initial - x + 5 - 4 = final) → x = 15 := 
by
  intros h
  sorry

end jellybean_removal_l309_309362


namespace seaweed_fed_to_livestock_l309_309080

def total_seaweed : ℝ := 600
def unusable_percentage : ℝ := 0.10
def fire_percentage : ℝ := 0.40
def medicinal_percentage : ℝ := 0.20
def consumption_and_feed_percentage : ℝ := 0.40
def human_consumption_percentage : ℝ := 0.30
def livestock_feed_percentage : ℝ := 0.70
def weight_loss_percentage : ℝ := 0.05

theorem seaweed_fed_to_livestock :
  let unusable_seaweed := unusable_percentage * total_seaweed in
  let usable_seaweed := total_seaweed - unusable_seaweed in
  let for_fire := fire_percentage * usable_seaweed in
  let for_medicinal := medicinal_percentage * usable_seaweed in
  let for_consumption_and_feed := consumption_and_feed_percentage * usable_seaweed in
  let for_human_consumption := human_consumption_percentage * for_consumption_and_feed in
  let for_livestock_feed := livestock_feed_percentage * for_consumption_and_feed in
  let weight_loss := weight_loss_percentage * for_livestock_feed in
  let final_livestock_feed := for_livestock_feed - weight_loss in
  final_livestock_feed = 143.64 :=
by
  sorry

end seaweed_fed_to_livestock_l309_309080


namespace max_vouchers_with_680_l309_309110

def spend_to_voucher (spent : ℕ) : ℕ := (spent / 100) * 20

theorem max_vouchers_with_680 : spend_to_voucher 680 = 160 := by
  sorry

end max_vouchers_with_680_l309_309110


namespace hexagon_area_l309_309001

noncomputable def area_of_hexagon (A B C D E F A' B' C' D' E' F' : Point)
  (area_ABC' : ℝ) (area_BCD' : ℝ) (area_CDE' : ℝ) (area_DEF' : ℝ) (area_EFA' : ℝ) (area_FAB' : ℝ) : ℝ :=
  (2 / 3) * (area_ABC' + area_BCD' + area_CDE' + area_DEF' + area_EFA' + area_FAB')

theorem hexagon_area (A B C D E F A' B' C' D' E' F' : Point)
  (hA' : midpoint A B A')
  (hB' : midpoint B C B')
  (hC' : midpoint C D C')
  (hD' : midpoint D E D')
  (hE' : midpoint E F E')
  (hF' : midpoint F A F')
  (area_ABC' area_BCD' area_CDE' area_DEF' area_EFA' area_FAB': ℝ)
  (h_ABC' : area_of_triangle A B C' = area_ABC')
  (h_BCD' : area_of_triangle B C D' = area_BCD')
  (h_CDE' : area_of_triangle C D E' = area_CDE')
  (h_DEF' : area_of_triangle D E F' = area_DEF')
  (h_EFA' : area_of_triangle E F A' = area_EFA')
  (h_FAB' : area_of_triangle F A B' = area_FAB') :
  area_of_hexagon A B C D E F A' B' C' D' E' F' area_ABC' area_BCD' area_CDE' area_DEF' area_EFA' area_FAB' =
    (2 / 3) * (area_ABC' + area_BCD' + area_CDE' + area_DEF' + area_EFA' + area_FAB') :=
  sorry

end hexagon_area_l309_309001


namespace integer_value_of_K_l309_309098

theorem integer_value_of_K (K : ℤ) : 
  (1000 < K^4 ∧ K^4 < 5000) ∧ K > 1 → K = 6 ∨ K = 7 ∨ K = 8 :=
by sorry

end integer_value_of_K_l309_309098


namespace hunter_strategy_does_not_increase_probability_l309_309027

theorem hunter_strategy_does_not_increase_probability (p : ℝ) (hp : 0 < p) (hp1 : p ≤ 1) :
  let p2 := p * p in
  let split_prob := p * (1 - p) in
  let combined_prob := p2 + split_prob in
  combined_prob = p :=
by
  let p := p
  let p2 := p * p
  let split_prob := p * (1 - p)
  let combined_prob := p2 + split_prob
  have : combined_prob = p := sorry
  exact this

end hunter_strategy_does_not_increase_probability_l309_309027


namespace roots_of_poly_l309_309134

noncomputable def poly : Polynomial ℝ := 8 * (Polynomial.monomial 4 1) + 14 * (Polynomial.monomial 3 1) - 66 * (Polynomial.monomial 2 1) + 40 * (Polynomial.monomial 1 1)

theorem roots_of_poly : {0, 1 / 2, 2, -5} = {x : ℝ | poly.eval x = 0} :=
by {
  sorry
}

end roots_of_poly_l309_309134


namespace simplify_complex_expr_correct_l309_309732

noncomputable def simplify_complex_expr (i : ℂ) (h : i^2 = -1) : ℂ :=
  3 * (4 - 2 * i) - 2 * i * (3 - 2 * i) + (1 + i) * (2 + i)

theorem simplify_complex_expr_correct (i : ℂ) (h : i^2 = -1) : 
  simplify_complex_expr i h = 9 - 9 * i :=
by
  sorry

end simplify_complex_expr_correct_l309_309732


namespace negation_proof_l309_309974

open Classical

variable (ℤ : Type) [Int : Type]

def proposition_p : Prop := ∃ x : ℤ, x^2 ≥ x
def neg_p : Prop := ∀ x : ℤ, x^2 < x

theorem negation_proof : ¬ proposition_p ↔ neg_p :=
by sorry

end negation_proof_l309_309974


namespace sine_neg_periodic_value_l309_309823

theorem sine_neg_periodic_value :
  sin (-10 * Real.pi / 3) = sqrt 3 / 2 :=
by
  sorry

end sine_neg_periodic_value_l309_309823


namespace side_length_of_square_l309_309749

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l309_309749


namespace find_series_sum_l309_309271

noncomputable def series_sum (s : ℝ) : ℝ := ∑' n : ℕ, (n+1) * s^(4*n + 3)

theorem find_series_sum (s : ℝ) (h : s^4 - s - 1/2 = 0) : series_sum s = -4 := by
  sorry

end find_series_sum_l309_309271


namespace midpoint_parallel_l309_309673

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l309_309673


namespace side_length_of_square_l309_309761

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l309_309761


namespace find_cos_alpha_l309_309970
noncomputable def cos_alpha_question (A : ℝ) : Prop :=
  ∀ α : ℝ, 
  f α = -1/5 → 
  (π / 2 < α ∧ α < π) →
  cos α = -4 / 5

def f : ℝ → ℝ := sorry
axiom A_value : ∀ x ∈ ℝ, x = √2
axiom f_zero_one : f(0) = 1
axiom f_alpha_cond : ∀ α : ℝ, (π / 2 < α ∧ α < π) → f α = -1/5 → false

theorem find_cos_alpha : cos_alpha_question (√2) :=
begin
  sorry
end

end find_cos_alpha_l309_309970


namespace no_line_intersects_all_three_sides_l309_309300

theorem no_line_intersects_all_three_sides (A B C : Point) (l : Line) :
  ¬ (∃ p1 p2 p3 : Point,
        p1 ∈ (side A B) ∧ p2 ∈ (side B C) ∧ p3 ∈ (side C A) ∧
        p1 ≠ A ∧ p1 ≠ B ∧
        p2 ≠ B ∧ p2 ≠ C ∧
        p3 ≠ C ∧ p3 ≠ A ∧
        l.Intersects p1 ∧ l.Intersects p2 ∧ l.Intersects p3) :=
by
  sorry

end no_line_intersects_all_three_sides_l309_309300


namespace shanghai_expo_scientific_notation_l309_309341

theorem shanghai_expo_scientific_notation : 5_280_000 = 5.28 * 10^6 :=
by
  sorry

end shanghai_expo_scientific_notation_l309_309341


namespace min_days_to_double_l309_309635

noncomputable def daily_interest (principal : ℝ) (rate : ℝ) : ℝ := principal * rate

def total_payment (principal : ℝ) (rate : ℝ) (days : ℕ) : ℝ :=
  principal + daily_interest principal rate * days

theorem min_days_to_double (principal : ℝ) (rate : ℝ) (days : ℕ) :
  principal = 20 → rate = 0.20 →
  total_payment principal rate days ≥ 2 * principal ↔ days ≥ 5 :=
by
  intros
  sorry

end min_days_to_double_l309_309635


namespace incorrect_gcd_statement_l309_309873

theorem incorrect_gcd_statement :
  ¬(gcd 85 357 = 34) ∧ (gcd 16 12 = 4) ∧ (gcd 78 36 = 6) ∧ (gcd 105 315 = 105) :=
by
  sorry

end incorrect_gcd_statement_l309_309873


namespace hyperbola_focal_distance_solution_l309_309993

-- Definitions corresponding to the problem conditions
def hyperbola_equation (x y m : ℝ) :=
  x^2 / m - y^2 / 6 = 1

def focal_distance (c : ℝ) := 2 * c

-- Theorem statement to prove m = 3 based on given conditions
theorem hyperbola_focal_distance_solution (m : ℝ) (h_eq : ∀ x y : ℝ, hyperbola_equation x y m) (h_focal : focal_distance 3 = 6) :
  m = 3 :=
by {
  -- sorry is used here as a placeholder for the actual proof steps
  sorry
}

end hyperbola_focal_distance_solution_l309_309993


namespace distance_bottom_edge_l309_309292

theorem distance_bottom_edge (d : ℝ) (h₁ : d = 53) : 10 + d + x = 20 + 63 → x = 73 :=
by
  intro h₀
  have h : 10 + d + x = 73 + d := calc
    10 + d + x = 20 + 63 : by rw h₀
  rw [add_left_eq_self] at h
  assumption
  sorry

end distance_bottom_edge_l309_309292


namespace percent_of_l309_309378

theorem percent_of (num : ℕ) (percent : ℕ) (decimal_equiv : ℕ → ℕ) :
  decimal_equiv percent = 3 → num * 3 = 120 :=
begin
  intros h,
  rw h,
  refl,
end

end percent_of_l309_309378


namespace rational_roots_of_equation_l309_309683

theorem rational_roots_of_equation (x : ℚ) (h : 9 * x^2 - 8 * (floor x) = 1) : x = 1 ∨ x = 1 / 3 :=
by
  sorry

end rational_roots_of_equation_l309_309683


namespace count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l309_309835

-- Statement for Question 1
theorem count_numbers_with_digit_7 :
  ∃ n, n = 19 ∧ (∀ k, (k < 100 → (k / 10 = 7 ∨ k % 10 = 7) ↔ k ≠ 77)) :=
sorry

-- Statement for Question 2
theorem count_numbers_divisible_by_3_or_5 :
  ∃ n, n = 47 ∧ (∀ k, (k < 100 → (k % 3 = 0 ∨ k % 5 = 0)) ↔ (k % 15 = 0)) :=
sorry

end count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l309_309835


namespace triangles_with_positive_area_in_grid_l309_309199

theorem triangles_with_positive_area_in_grid : 
  let points := {(i, j) | 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6} in
  let total_combinations := Nat.choose 36 3 in
  let horizontal_vertical_combinations := 12 * Nat.choose 6 3 in
  let main_diagonal_combinations := 2 * Nat.choose 6 3 in
  total_combinations - horizontal_vertical_combinations - main_diagonal_combinations = 6860 :=
by
  let points := {(i, j) | 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6}
  let total_combinations := Nat.choose 36 3
  let horizontal_vertical_combinations := 12 * Nat.choose 6 3
  let main_diagonal_combinations := 2 * Nat.choose 6 3
  show total_combinations - horizontal_vertical_combinations - main_diagonal_combinations = 6860
  sorry

end triangles_with_positive_area_in_grid_l309_309199


namespace count_9_as_most_significant_digit_l309_309276

/-- Mathematically equivalent proof problem: -/
theorem count_9_as_most_significant_digit 
  (T : Set ℕ) 
  (hT : T = {k | ∃ k, 0 ≤ k ∧ k ≤ 4000 ∧ 9^k})
  (h_digits_9_4000 : (9:ℝ)^4000 < 10^(3817 + 1))
  (h_most_significant_digit_9_4000 : 9 * 10^(3816:ℝ) ≤ (9:ℝ)^(4000) ∧ (9:ℝ)^(4000) < 10 * 10^(3816:ℝ)): 
  ∃ (count : ℕ), count = 184 ∧ (∀ k ∈ T, has_digit_9 (9^k) ↔ k < count) := 
sorry

end count_9_as_most_significant_digit_l309_309276


namespace product_of_bc_l309_309799

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end product_of_bc_l309_309799


namespace complex_power_identity_l309_309115

theorem complex_power_identity (h : (1 + complex.I) / real.sqrt 2 ^ 2 = complex.I) : 
  ((1 + complex.I) / real.sqrt 2) ^ 30 = -complex.I :=
by sorry

end complex_power_identity_l309_309115


namespace sum_of_two_digit_reversible_primes_l309_309075

noncomputable def is_two_digit_prime (n : ℕ) : Prop :=
  (10 ≤ n) ∧ (n < 50) ∧ (Nat.Prime n)

noncomputable def is_reverse_prime (n : ℕ) : Prop :=
  let reversed_digits : ℕ := (n % 10) * 10 + (n / 10) in
  (reversed_digits < 50) ∧ (Nat.Prime reversed_digits)

theorem sum_of_two_digit_reversible_primes : 
  ∑ n in {n | is_two_digit_prime n ∧ is_reverse_prime n}, n = 55 := by
  sorry

end sum_of_two_digit_reversible_primes_l309_309075


namespace class_boys_count_l309_309360

theorem class_boys_count
    (x y : ℕ)
    (h1 : x + y = 20)
    (h2 : (1 / 3 : ℚ) * x = (1 / 2 : ℚ) * y) :
    x = 12 :=
by
  sorry

end class_boys_count_l309_309360


namespace problem_solution_l309_309548

noncomputable def point := (ℝ × ℝ)

noncomputable def distance (p q : point) : ℝ := 
  real.sqrt ((fst p - fst q)^2 + (snd p - snd q)^2)

noncomputable def curve_C (p : point) : Prop := 
  distance p (1, 0) = distance p (-1, snd p)

noncomputable def on_curve (p : point) : Prop := 
  curve_C p

noncomputable def line_l (p : point) : Prop := 
  snd p = 2 * (fst p) - 1

noncomputable def A : point := (1, 1)

noncomputable def midpoint (p q : point) : point := 
  ((fst p + fst q) / 2, (snd p + snd q) / 2)

noncomputable def area_triangle (O P Q : point) : ℝ := 
  (1 / 2) * real.abs ((fst P * snd Q - fst Q * snd P))

theorem problem_solution :
  (∀ p : point, on_curve p ↔ p.2^2 = 4 * p.1) ∧ 
  (∀ P Q : point, on_curve P → on_curve Q → line_l P → line_l Q → midpoint P Q = A → area_triangle (0, 0) P Q = real.sqrt 3 / 2) :=
sorry

end problem_solution_l309_309548


namespace water_volume_in_pool_excluding_column_l309_309887

-- Define the diameters and depth of the pool and the column
def pool_diameter : ℝ := 20
def pool_depth : ℝ := 6
def column_diameter : ℝ := 4

-- Define the radii based on the given diameters
def pool_radius : ℝ := pool_diameter / 2
def column_radius : ℝ := column_diameter / 2

-- Define the volumes of the pool and column using the volume formula for cylinders
def pool_volume : ℝ := π * pool_radius^2 * pool_depth
def column_volume : ℝ := π * column_radius^2 * pool_depth

-- Final volume of water in the pool
def water_volume : ℝ := pool_volume - column_volume

-- The theorem to prove
theorem water_volume_in_pool_excluding_column :
  water_volume = 576 * π :=
by
  sorry

end water_volume_in_pool_excluding_column_l309_309887


namespace triangle_inequalities_l309_309449

theorem triangle_inequalities (a b c h_a h_b h_c : ℝ) (ha_eq : h_a = b * Real.sin (arc_c)) (hb_eq : h_b = a * Real.sin (arc_c)) (hc_eq : h_c = a * Real.sin (arc_b)) (h : a > b) (h2 : b > c) :
  (a + h_a > b + h_b) ∧ (b + h_b > c + h_c) :=
by
  sorry

end triangle_inequalities_l309_309449


namespace measure_of_angle_A_maximum_value_of_f_l309_309627

theorem measure_of_angle_A
  (a b c : ℝ)
  (h : b^2 + c^2 = a^2 + b * c) :
  ∃ A, A = Real.pi / 3 ∧ A ∈ (0, Real.pi) → Real.cos A = 1 / 2 :=
sorry

theorem maximum_value_of_f
  (x : ℝ) :
  let A := Real.pi / 3 in
  let f := λ x, Real.sin (x - A) + Real.sqrt 3 * Real.cos x in
  ∃ x, f x ≤ 1 :=
sorry

end measure_of_angle_A_maximum_value_of_f_l309_309627


namespace expression_divisible_by_7_l309_309148

theorem expression_divisible_by_7 (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ m : ℤ, 3^(6*n-1) - k * 2^(3*n-2) + 1 = 7 * m) ↔ ∃ m' : ℤ, k = 7 * m' + 3 := 
by
  sorry

end expression_divisible_by_7_l309_309148


namespace white_bar_dimensions_l309_309707

theorem white_bar_dimensions (volume_cube : ℝ)
  (bars : fin 8 → ℝ)
  (gray_bars : fin 4 → ℝ)
  (white_bars : fin 4 → ℝ)
  (cube_edge_length : ℝ)
  (h_volume_edges : cube_edge_length = 1)
  (h_cube_volume : volume_cube = 1)
  (h_bars_volume : ∀ i, bars i = volume_cube / 8)
  (h_gray_bars_identical : ∀ i j, gray_bars i = gray_bars j)
  (h_white_bars_identical : ∀ i j, white_bars i = white_bars j) :
  white_bars 0 = (7/10, 1/2, 1/4) :=
by
  sorry

end white_bar_dimensions_l309_309707


namespace smallest_n_for_Tn_integer_l309_309259

-- Conditions
def L : ℚ := ∑ i in Finset.range 10 \ {0}, (1 : ℚ) / (i^2)
def D : ℕ := 2^6 * 3^4 * 5^2 * 7^2
def T_n (n : ℕ) : ℚ := (n * 10^(n - 1)) * L

-- Problem Statement
theorem smallest_n_for_Tn_integer (n : ℕ) : (T_n n).denom = 1 → n = 8 := 
sorry

end smallest_n_for_Tn_integer_l309_309259


namespace total_dog_food_amount_l309_309111

def initial_dog_food : ℝ := 15
def first_purchase : ℝ := 15
def second_purchase : ℝ := 10

theorem total_dog_food_amount : initial_dog_food + first_purchase + second_purchase = 40 := 
by 
  sorry

end total_dog_food_amount_l309_309111


namespace brenda_peaches_left_l309_309071

theorem brenda_peaches_left (total_peaches : ℕ) (fresh_percentage : ℚ) (too_small : ℕ) :
  total_peaches = 550 →
  fresh_percentage = 0.45 →
  too_small = 35 →
  let fresh_peaches := int.floor (fresh_percentage * total_peaches) in
  fresh_peaches - too_small = 212 :=
by
  intros h1 h2 h3
  let fresh_peaches := Int.floor (fresh_percentage * total_peaches)
  have h_fresh_peaches: fresh_peaches = 247 := sorry
  have h_total := fresh_peaches - too_small = 212
  exact h_total
  sorry

end brenda_peaches_left_l309_309071


namespace problem_equivalent_proof_l309_309153

noncomputable def f (α : ℝ) : ℝ := 
  (sin (α - 3 * Real.pi) * cos (2 * Real.pi - α) * sin (-α + 3 * Real.pi / 2)) / 
  (cos (-Real.pi - α) * sin (-Real.pi - α))

theorem problem_equivalent_proof (α : ℝ) 
  (h1 : π < α ∧ α < 3 * π / 2)
  (h2 : cos (α - 3 * π / 2) = 1 / 5) : 
  f α = 2 * sqrt 6 / 5 := 
sorry

end problem_equivalent_proof_l309_309153


namespace coefficient_x2_expansion_eq_21_l309_309597

theorem coefficient_x2_expansion_eq_21 (a : ℝ) :
  (∑ i in Finset.range 8, (∑ j in Finset.range (i + 1), (5.choose j) * (a^(5 - j)) * (2.choose (i - j)) * x^(i - j)) * (2.choose (i - j))) = 21 → a = 1 ∨ a = -2 := 
by
  sorry

end coefficient_x2_expansion_eq_21_l309_309597


namespace reconstruct_diagonals_l309_309022

theorem reconstruct_diagonals (n : ℕ) (polygon : convex_polygon n) (triangles_count : fin n → ℕ)
  (triangulation : non_intersecting_diagonals polygon triangles_count) :
  can_reconstruct_diagonals polygon triangles_count := 
by sorry

end reconstruct_diagonals_l309_309022


namespace proposition_1_proposition_3_l309_309967

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) - 2 * sqrt 3 * sin x * cos x

theorem proposition_1 (x1 x2 : ℝ) (h : x1 - x2 = π) : f x1 = f x2 := by
  have h₁ : 2 * x1 + π / 3 = 2 * x2 + π / 3 + 2 * π := by sorry
  rw [cos_add, cos_sub] at h₁
  sorry

theorem proposition_3 : f (π / 12) = 0 := by
  have h : f (π / 12) = 2 * cos (2 * π / 12 + π / 3) := by sorry
  rw [cos, sin, cos_add, sin_add] at h
  sorry

end proposition_1_proposition_3_l309_309967


namespace bottles_left_on_shelf_l309_309355

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l309_309355


namespace order_number_reversed_l309_309923

def is_positive (x : ℕ) : Prop := x > 0

def is_distinct {α : Type*} [DecidableEq α] (l : List α) : Prop :=
  l.Nodup

def order_number (l : List ℕ) : ℕ :=
  List.length {(i, j) | i < j ∧ l.nth i < l.nth j}.toFinset

theorem order_number_reversed (a1 a2 a3 a4 a5 : ℕ)
  (h1 : is_positive a1) (h2 : is_positive a2) (h3 : is_positive a3)
  (h4 : is_positive a4) (h5 : is_positive a5)
  (h_distinct : is_distinct [a1, a2, a3, a4, a5])
  (h_order : order_number [a1, a2, a3, a4, a5] = 4) :
  order_number [a5, a4, a3, a2, a1] = 6 :=
sorry

end order_number_reversed_l309_309923


namespace sum_of_positive_real_solutions_l309_309696

def f (x : ℝ) : ℝ := x^(3^(Real.sqrt 3))
def g (x : ℝ) : ℝ := 3^(3^x)
noncomputable def S : ℝ := 3

theorem sum_of_positive_real_solutions :
  (∀ x : ℝ, x > 0 → f x = g x → x = 3) → S = 3 :=
by
  intros h
  exact h 3 (by norm_num : 3 > 0) (by norm_num : f 3 = g 3)

end sum_of_positive_real_solutions_l309_309696


namespace sum_of_monomials_is_monomial_l309_309215

variable (a b : ℕ)

theorem sum_of_monomials_is_monomial (m n : ℕ) (h : ∃ k : ℕ, 2 * a^m * b^n + a * b^3 = k * a^1 * b^3) :
  m = 1 ∧ n = 3 :=
sorry

end sum_of_monomials_is_monomial_l309_309215


namespace remainder_when_3m_divided_by_5_l309_309829

theorem remainder_when_3m_divided_by_5 (m : ℤ) (hm : m % 5 = 2) : (3 * m) % 5 = 1 := 
sorry

end remainder_when_3m_divided_by_5_l309_309829


namespace mn_parallel_pq_l309_309652

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l309_309652


namespace train_probability_l309_309780

theorem train_probability
  (main_line_start : ℕ) (main_line_frequency : ℕ)
  (harbor_line_start : ℕ) (harbor_line_frequency : ℕ)
  (guy_arrival : ℕ) :
  main_line_start = 0 →
  main_line_frequency = 10 →
  harbor_line_start = 2 →
  harbor_line_frequency = 10 →
  (0 ≤ guy_arrival ∧ guy_arrival < main_line_frequency * harbor_line_frequency) →
  (∃ k : ℕ, guy_arrival = main_line_start + k * main_line_frequency) ∨
  (∃ k : ℕ, guy_arrival = harbor_line_start + k * harbor_line_frequency) →
  (∃ k : ℕ, guy_arrival = main_line_start + k * main_line_frequency) →
  probability_of_getting_main_line = 1 / 2 :=
begin
  intros,
  sorry
end

end train_probability_l309_309780


namespace least_number_of_cookies_l309_309708

theorem least_number_of_cookies (c : ℕ) :
  (c % 6 = 5) ∧ (c % 8 = 7) ∧ (c % 9 = 6) → c = 23 :=
by
  sorry

end least_number_of_cookies_l309_309708


namespace sum_of_roots_transformed_l309_309265

theorem sum_of_roots_transformed {R : Type*} [Field R] :
  let b : Fin 2017 → R := λ n, (Polynomial.roots (Polynomial.C (-1045) +
          Polynomial.X - Polynomial.X^((2017:ℕ) + 1))).nth n in
  ∑ i in Finset.range 2017, 1 / (1 - b i) = 2017 / (-1046 : R) :=
sorry

end sum_of_roots_transformed_l309_309265


namespace find_b_l309_309270

variable (p q : ℕ → ℕ) (b : ℤ)

def p_def : (ℕ → ℤ) := λ x, 3 * x - 5
def q_def : (ℕ → ℤ) := λ x, 4 * x - b

theorem find_b (h : p_def (q_def 4) = 23) : b = 20 / 3 := by
  sorry

end find_b_l309_309270


namespace find_k_l309_309951

noncomputable def tangents_to_circle (k : ℝ) (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
P.2 = -k * P.1 - 4 ∧
(C.1, C.2) = (0, 1) ∧
∀ t : ℝ, (x - 0)^2 + (y - 1)^2 ≤ 1 →

def minimum_area_quadrilateral (S : ℝ): ℝ := 2 * S

theorem find_k (k : ℝ) (P : ℝ × ℝ) (A B C : ℝ × ℝ) (S : ℝ) :
 tangents_to_circle k P A B C ∧ minimum_area_quadrilateral S = 2 →
 k = 2 :=
sorry

end find_k_l309_309951


namespace percentage_of_acid_in_original_mixture_l309_309369

theorem percentage_of_acid_in_original_mixture
  (a w : ℚ)
  (h1 : a / (a + w + 2) = 18 / 100)
  (h2 : (a + 2) / (a + w + 4) = 30 / 100) :
  (a / (a + w)) * 100 = 29 := 
sorry

end percentage_of_acid_in_original_mixture_l309_309369


namespace millet_majority_day_l309_309710

theorem millet_majority_day :
  let M₀ := 0.4 -- Initial millet on Monday
  let daily_addition := 0.4 -- Daily millet addition
  let consumption_rate := 0.7 -- 100% - 30%, remaining millet each day
  (have M₁ := M₀ + daily_addition * consumption_rate,
   have M₂ := daily_addition + consumption_rate * M₁,
   have M₃ := daily_addition + consumption_rate * M₂,
   have M₄ := daily_addition + consumption_rate * M₃,
   M₄ > 0.5) :=
begin
  sorry
end

end millet_majority_day_l309_309710


namespace range_of_m_l309_309178

noncomputable def f (m x : ℝ) : ℝ :=
  1 - m * (Real.exp x) / (x^2 + x + 1)

theorem range_of_m (m : ℝ) :
  (∃ x : ℕ, 0 < x ∧ f m x ≥ 0 ∧ (∀ y : ℕ, (0 < y ∧ y ≠ x) → f m y < 0)) →
  (∃ a b : ℝ, a = 7 / Real.exp 2 ∧ b = 3 / Real.exp 1 ∧ (a < m ∧ m ≤ b)) :=
sorry

end range_of_m_l309_309178


namespace side_length_of_square_l309_309747

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l309_309747


namespace determine_y_l309_309107

theorem determine_y : 
  ∀ y : ℝ, 
    (2 * Real.arctan (1 / 5) + Real.arctan (1 / 25) + Real.arctan (1 / y) = Real.pi / 4) -> 
    y = -121 / 60 :=
by
  sorry

end determine_y_l309_309107


namespace coefficient_of_fourth_term_in_expansion_coefficient_of_fourth_term_eq_neg7_l309_309622

theorem coefficient_of_fourth_term_in_expansion (x : ℝ) :
  (3*x - (1 / (2 * 3 * x))) ^ 8 = ∑ i in finset.range 9, (nat.choose 8 i) * (3*x)^(8 - i) * (-(1 / (2 * 3 * x)))^i :=
begin
  sorry
end

theorem coefficient_of_fourth_term_eq_neg7 (x : ℝ) :
  ∃ T₄, T₄ = nat.choose 8 3 * (3*x)^(8 - 3) * (-(1 / (2 * 3 * x)))^3 ∧ T₄ = -7 :=
begin
  sorry
end

end coefficient_of_fourth_term_in_expansion_coefficient_of_fourth_term_eq_neg7_l309_309622


namespace tiles_required_l309_309440

def area_of_room (length width : ℝ) : ℝ :=
  length * width

def area_of_tile (length width : ℝ) : ℝ :=
  length * width

def number_of_tiles (room_area tile_area : ℝ) : ℝ :=
  room_area / tile_area

theorem tiles_required : 
  let room_length := 15
  let room_width := 20
  let tile_length := (3 / 12)   -- Convert inches to feet
  let tile_width := (9 / 12)    -- Convert inches to feet
  number_of_tiles (area_of_room room_length room_width) (area_of_tile tile_length tile_width) = 1600 :=
by
  sorry

end tiles_required_l309_309440


namespace star_7_3_eq_neg_5_l309_309495

def star_operation (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem star_7_3_eq_neg_5 : star_operation 7 3 = -5 :=
by
  -- proof goes here
  sorry

end star_7_3_eq_neg_5_l309_309495


namespace mn_parallel_pq_l309_309664

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l309_309664


namespace five_person_lineup_l309_309232

theorem five_person_lineup : 
  let total_ways := Nat.factorial 5
  let invalid_first := Nat.factorial 4
  let invalid_last := Nat.factorial 4
  let valid_ways := total_ways - (invalid_first + invalid_last)
  valid_ways = 72 :=
by
  sorry

end five_person_lineup_l309_309232


namespace ratio_eq_l309_309562

variable (a b c d : ℚ)

theorem ratio_eq :
  (a / b = 5 / 2) →
  (c / d = 7 / 3) →
  (d / b = 5 / 4) →
  (a / c = 6 / 7) :=
by
  intros h1 h2 h3
  sorry

end ratio_eq_l309_309562


namespace range_of_f_lt_zero_l309_309739

noncomputable
def f : ℝ → ℝ := sorry

theorem range_of_f_lt_zero 
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_neg2_zero : f (-2) = 0) :
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end range_of_f_lt_zero_l309_309739


namespace side_length_of_square_l309_309765

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309765


namespace prove_hyperbola_prove_ellipse_prove_circle_prove_straight_lines_l309_309176

-- Define the context for the given curve
def curve (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 - n * y^2 = 1

-- Statement 1: Proving that if mn > 0, then C is a hyperbola
theorem prove_hyperbola (m n : ℝ) (h_mn : m * n > 0) : ∀ x y : ℝ, curve m n x y → mx^2 - ny^2 (x) - (y).prop sorry := sorry

-- Statement 2: Proving that if m > 0 and m + n < 0, then C is an ellipse with the foci on the x-axis
theorem prove_ellipse (m n : ℝ) (h_m : m > 0) (h_sum : m + n < 0) : ∀ x y : ℝ, curve m n x y → mx^2 - ny^2 (x) - (y).prop sorry := sorry

-- Statement 3: Proving that if m > 0 and n < 0, then C can represent a circle
theorem prove_circle (m n : ℝ) (h_m : m > 0) (h_n : n < 0) : ∃ x y : ℝ, curve m n x y := sorry

-- Statement 4: Proving that if m > 0 and n = 0, then C consists of two straight lines
theorem prove_straight_lines (m : ℝ) (h_m : m > 0) : ∃ x y : ℝ, curve m 0 x y := sorry

end prove_hyperbola_prove_ellipse_prove_circle_prove_straight_lines_l309_309176


namespace cyclist_wait_time_l309_309025

-- Define the given conditions
def hiker_speed := (1 : ℝ) / 15
def cyclist_speed := (1 : ℝ) / 4
def waiting_time := 13.75

-- State the proof problem
theorem cyclist_wait_time :
  ∃ t : ℝ, (cyclist_speed * t = hiker_speed * waiting_time) → t = 11 / 3 :=
by
  sorry

end cyclist_wait_time_l309_309025


namespace solve_for_y_l309_309735

theorem solve_for_y {y : ℚ} : y = 12 → (1 / 3 - 1 / 4 = 1 / y) :=
by
  intro hy
  rw hy
  norm_num

end solve_for_y_l309_309735


namespace find_x_value_l309_309931

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem find_x_value :
  ∃ α : ℝ, power_function α (-2) = -1 / 8 ∧ ∀ x : ℝ, power_function α x = 64 ↔ x = 1 / 4 :=
by {
  use -3,
  split,
  { -- show power_function -3 (-2) = -1 / 8
    sorry },
  { -- show ∀ x, power_function -3 x = 64 ↔ x = 1 / 4
    intro x,
    split;
    intro h,
    { -- power_function -3 x = 64 → x = 1 / 4
      sorry },
    { -- x = 1 / 4 → power_function -3 x = 64
      sorry } }
}

end find_x_value_l309_309931


namespace cube_root_two_irrational_l309_309725

theorem cube_root_two_irrational : ¬ ∃ (n m : ℕ), (n^3 = 2 * m^3) ∧ (Nat.coprime n m) := by
  sorry

end cube_root_two_irrational_l309_309725


namespace max_min_product_diff_l309_309777

theorem max_min_product_diff (f_range : Set ℝ) (g_range : Set ℝ)
    (hf : f_range = { x | -3 ≤ x ∧ x ≤ 9 })
    (hg : g_range = { y | -1 ≤ y ∧ y ≤ 6 }) :
    (let max_product := max { x * y | x ∈ f_range ∧ y ∈ g_range }
     let min_product := min { x * y | x ∈ f_range ∧ y ∈ g_range }
     in max_product - min_product = 72) :=
by {
  -- The proof will go here
  sorry
}

end max_min_product_diff_l309_309777


namespace range_of_a_l309_309552

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def proposition_P (a : ℝ) : Prop := ∀ x : ℝ, log_base2 (x^2 + x + a) > 0

def proposition_Q (a : ℝ) : Prop := ∃ x0 : ℝ, (-2 ≤ x0 ∧ x0 ≤ 2) ∧ 2^a ≤ 2^x0

theorem range_of_a (a : ℝ) : proposition_P a ∧ proposition_Q a → a > 5/4 ∧ a ≤ 2 :=
sorry

end range_of_a_l309_309552


namespace inequality_bounds_l309_309500

theorem inequality_bounds (x y : ℝ) : |y - 3 * x| < 2 * x ↔ x > 0 ∧ x < y ∧ y < 5 * x := by
  sorry

end inequality_bounds_l309_309500


namespace side_length_of_square_l309_309769

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l309_309769


namespace salary_january_l309_309742

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 :=
by 
  sorry

end salary_january_l309_309742


namespace find_b_if_longest_chord_l309_309596

noncomputable def chord_condition (l : ℝ → ℝ) (b : ℝ) : Prop :=
  ∃ C : ℝ × ℝ × ℝ, 
    C = (1, 0, 2) ∧ 
    (∀ P Q : ℝ × ℝ, 
      (P.1 - 1) ^ 2 + P.2 ^ 2 = 4 → 
      (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 4 → 
      P ≠ Q → 
      P.2 = P.1 + b → 
      Q.2 = Q.1 + b →
      (C.1 - P.1) * (C.1 - Q.1) + (C.2 - P.2) * (C.2 - Q.2) = -C.3)

theorem find_b_if_longest_chord :
  ∀ (b : ℝ), 
  (chord_condition (λ x, x + b) b) → 
  b = -1 :=
by
  assume b,
  assume h: chord_condition (λ x, x + b) b,
  sorry

end find_b_if_longest_chord_l309_309596


namespace extra_men_to_finish_in_time_l309_309468

noncomputable section
def initial_length := 10 -- km
def total_days := 150 -- days
def initial_men := 30 -- men
def completed_length := 2 -- km
def days_passed := 50 -- days
def remaining_length := initial_length - completed_length -- km
def days_remaining := total_days - days_passed -- days
def rate_of_work_per_day_initial := completed_length / days_passed -- km/day initial work rate

def required_rate_of_work := remaining_length / days_remaining -- km/day required work rate
def men_needed := initial_men * (required_rate_of_work / rate_of_work_per_day_initial)
def extra_men_needed := men_needed - initial_men

theorem extra_men_to_finish_in_time 
  (initial_length = 10) 
  (total_days = 150) 
  (initial_men = 30) 
  (completed_length = 2) 
  (remaining_length = 8) 
  (days_remaining = 100) :
  extra_men_needed = 30 := 
sorry

end extra_men_to_finish_in_time_l309_309468


namespace ellipse_standard_form_l309_309953

def center_origin (x y : ℝ) := x = 0 ∧ y = 0
def focus_y_axis (f : ℝ × ℝ) := f.1 = 0 -- Only y-coordinate matters as x must be zero for focus on y-axis
def eccentricity (c a : ℝ) := c / a = 1 / 2
def focal_length (c : ℝ) := 2 * c = 8
def ellipse_eq (a b : ℝ) := (λ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1)

theorem ellipse_standard_form :
  ∃ a b c : ℝ, center_origin 0 0 ∧ focus_y_axis (0, c) ∧ eccentricity c a ∧ focal_length c ∧ a = 8 ∧ b^2 = a^2 - c^2 ∧ ellipse_eq a b = (λ x y, y^2 / 64 + x^2 / 48 = 1) := 
by {
  sorry
}

end ellipse_standard_form_l309_309953


namespace compute_expression_l309_309207

theorem compute_expression (x : ℝ) (h : x + (1 / x) = 7) :
  (x - 3)^2 + (49 / (x - 3)^2) = 23 :=
by
  sorry

end compute_expression_l309_309207


namespace water_consumed_l309_309445

theorem water_consumed (traveler_water : ℕ) (camel_multiplier : ℕ) (ounces_in_gallon : ℕ) (total_water : ℕ)
  (h_traveler : traveler_water = 32)
  (h_camel : camel_multiplier = 7)
  (h_ounces_in_gallon : ounces_in_gallon = 128)
  (h_total : total_water = traveler_water + camel_multiplier * traveler_water) :
  total_water / ounces_in_gallon = 2 :=
by
  sorry

end water_consumed_l309_309445


namespace triangles_with_positive_area_in_grid_l309_309198

theorem triangles_with_positive_area_in_grid : 
  let points := {(i, j) | 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6} in
  let total_combinations := Nat.choose 36 3 in
  let horizontal_vertical_combinations := 12 * Nat.choose 6 3 in
  let main_diagonal_combinations := 2 * Nat.choose 6 3 in
  total_combinations - horizontal_vertical_combinations - main_diagonal_combinations = 6860 :=
by
  let points := {(i, j) | 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6}
  let total_combinations := Nat.choose 36 3
  let horizontal_vertical_combinations := 12 * Nat.choose 6 3
  let main_diagonal_combinations := 2 * Nat.choose 6 3
  show total_combinations - horizontal_vertical_combinations - main_diagonal_combinations = 6860
  sorry

end triangles_with_positive_area_in_grid_l309_309198


namespace find_weekly_allowance_l309_309633

-- Define the context for the problem
variables (A : ℝ) -- Jack's weekly allowance
constants (initial_amount final_amount weeks : ℝ)
constants (h1 : initial_amount = 43)
          (h2 : final_amount = 83)
          (h3 : weeks = 8)
          (h4 : final_amount = initial_amount + weeks * (A / 2))

theorem find_weekly_allowance : A = 10 :=
by sorry

end find_weekly_allowance_l309_309633


namespace length_of_each_piece_l309_309987

theorem length_of_each_piece :
  ∀ (ribbon_length remaining_length pieces : ℕ),
  ribbon_length = 51 →
  remaining_length = 36 →
  pieces = 100 →
  (ribbon_length - remaining_length) / pieces * 100 = 15 :=
by
  intros ribbon_length remaining_length pieces h1 h2 h3
  sorry

end length_of_each_piece_l309_309987


namespace average_salary_of_all_workers_l309_309609

-- Definitions of conditions
def T : ℕ := 7
def total_workers : ℕ := 56
def W : ℕ := total_workers - T
def A_T : ℕ := 12000
def A_W : ℕ := 6000

-- Definition of total salary and average salary
def total_salary : ℕ := (T * A_T) + (W * A_W)

theorem average_salary_of_all_workers : total_salary / total_workers = 6750 := 
  by sorry

end average_salary_of_all_workers_l309_309609


namespace isomer_molecular_weight_l309_309383

noncomputable def atomic_weight_C : Real := 12.01
noncomputable def atomic_weight_H : Real := 1.008
noncomputable def atomic_weight_O : Real := 16.00

def molecular_weight_C4H6O : Real :=
  4 * atomic_weight_C + 6 * atomic_weight_H + atomic_weight_O

theorem isomer_molecular_weight : molecular_weight_C4H6O = 70.088 :=
by
  sorry

end isomer_molecular_weight_l309_309383
