import Mathlib

namespace required_circle_through_intersections_and_center_on_line_l375_375097

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 3 = 0

noncomputable def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * y - 3 = 0

noncomputable def line (x y : ℝ) : Prop :=
  2 * x - y - 4 = 0

noncomputable def required_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 12 * x - 16 * y - 3 = 0

theorem required_circle_through_intersections_and_center_on_line :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y ∧ required_circle x y) ∧ 
  (∃ h k : ℝ, line h k ∧ (required_circle h k (k - 2 * h))) :=
sorry

end required_circle_through_intersections_and_center_on_line_l375_375097


namespace find_k_l375_375196

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l375_375196


namespace cuboid_surface_area_l375_375328

noncomputable def total_surface_area (x y z : ℝ) : ℝ :=
  2 * (x * y + y * z + z * x)

theorem cuboid_surface_area (x y z : ℝ) (h1 : x + y + z = 40) (h2 : x^2 + y^2 + z^2 = 625) :
  total_surface_area x y z = 975 :=
sorry

end cuboid_surface_area_l375_375328


namespace sulfuric_acid_reaction_l375_375978

theorem sulfuric_acid_reaction (SO₃ H₂O H₂SO₄ : ℕ) 
  (reaction : SO₃ + H₂O = H₂SO₄)
  (H₂O_eq : H₂O = 2)
  (H₂SO₄_eq : H₂SO₄ = 2) :
  SO₃ = 2 :=
by
  sorry

end sulfuric_acid_reaction_l375_375978


namespace triangle_area_l375_375973

-- Define points a, b, and c as vectors in ℝ³
def a := (2, -1, 1 : ℝ)
def b := (5, 1, 3 : ℝ)
def c := (10, 6, 5 : ℝ)

-- Definitions of the vectors b - a and c - a
def v1 := (b.1 - a.1, b.2 - a.2, b.3 - a.3 : ℝ)
def v2 := (c.1 - a.1, c.2 - a.2, c.3 - a.3 : ℝ)

-- Cross product of v1 and v2
def cross_prod (x y z : ℝ) (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (y * c - z * b, z * a - x * c, x * b - y * a)

-- Cross product of v1 and v2
def cp := cross_prod v1.1 v1.2 v1.3 v2.1 v2.2 v2.3

-- Magnitude of the cross product vector
noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

-- Proof statement
theorem triangle_area : (1/2) * magnitude cp = real.sqrt 77 / 2 :=
by
  sorry

end triangle_area_l375_375973


namespace length_XY_eq_27_72_l375_375257

noncomputable theory

section power_of_point

-- Define the variables and constants
variables (A B C D P Q X Y: Type)
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] 
variables (AP: ℕ) (CQ: ℕ) (AB: ℕ) (CD: ℕ) (PQ: ℕ) 
variables (XY: ℕ)

-- Set the conditions according to the problem.
variables (h1: AB = 13) (h2: CD = 17) (h3: AP = 7) (h4: CQ = 5) (h5: PQ = 25)

-- The final statement we aim to prove
theorem length_XY_eq_27_72 
  (hXY: XY = 27.72) : XY = 27.72 :=
sorry

end power_of_point

end length_XY_eq_27_72_l375_375257


namespace symmetric_parabola_l375_375696

def parabola1 (x : ℝ) : ℝ := (x - 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -(x + 2)^2 - 3

theorem symmetric_parabola : ∀ x y : ℝ,
  y = parabola1 x ↔ 
  (-y) = parabola2 (-x) ∧ y = -(x + 2)^2 - 3 :=
sorry

end symmetric_parabola_l375_375696


namespace root_in_interval_l375_375265

noncomputable def f (x : ℝ) := 3^x + 3 * x - 8

theorem root_in_interval :
  f 1 < 0 → f 1.25 < 0 → f 1.5 > 0 →
  ∃ x ∈ Ioo (1.25 : ℝ) (1.5 : ℝ), f x = 0 :=
by
  sorry

end root_in_interval_l375_375265


namespace slices_left_for_Phill_l375_375699

theorem slices_left_for_Phill  : 
  let initial_slices := 1
  let after_first_cut := initial_slices * 2
  let after_second_cut := after_first_cut * 2
  let after_third_cut := after_second_cut * 2
  let slices_given_to_first_three_friends := 3
  let slices_given_to_two_friends := 2 * 2
  let total_slices_given := slices_given_to_first_three_friends + slices_given_to_two_friends
  after_third_cut - total_slices_given = 1 :=
by
  -- Introducing the variables as mentioned in the conditions.
  let initial_slices := 1
  let after_first_cut := initial_slices * 2
  let after_second_cut := after_first_cut * 2
  let after_third_cut := after_second_cut * 2
  let slices_given_to_first_three_friends := 3
  let slices_given_to_two_friends := 2 * 2
  let total_slices_given := slices_given_to_first_three_friends + slices_given_to_two_friends
  
  -- Showing the final statement to be proven.
  show after_third_cut - total_slices_given = 1 from sorry

end slices_left_for_Phill_l375_375699


namespace inequality_holds_for_positive_integers_l375_375285

theorem inequality_holds_for_positive_integers (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) :
  (∑ i in finset.range (n + 1), a (i + 1))^2 ≤ ∑ i in finset.range (n + 1), a (i + 1)^3 :=
by
  sorry

end inequality_holds_for_positive_integers_l375_375285


namespace total_eyes_in_family_l375_375653

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end total_eyes_in_family_l375_375653


namespace total_eyes_in_family_l375_375655

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end total_eyes_in_family_l375_375655


namespace fraction_equals_decimal_l375_375074

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375074


namespace corrected_mean_l375_375416

theorem corrected_mean (mean original_observation wrong_observation corrected_observation : ℝ)
    (n : ℕ) (h_n : n = 40) (h_mean : mean = 36) (h_original_observation : wrong_observation = 20)
    (h_corrected_observation : corrected_observation = 34) :
    let total_sum := mean * n
    let difference := corrected_observation - wrong_observation
    let corrected_sum := total_sum + difference
    let new_mean := corrected_sum / n in
    new_mean = 36.35 :=
by
  intros
  dsimp at *
  rw [h_n, h_mean, h_original_observation, h_corrected_observation]
  sorry

end corrected_mean_l375_375416


namespace base8_subtraction_l375_375092

def subtract_base_8 (a b : Nat) : Nat :=
  sorry  -- This is a placeholder for the actual implementation.

theorem base8_subtraction :
  subtract_base_8 0o5374 0o2645 = 0o1527 :=
by
  sorry

end base8_subtraction_l375_375092


namespace probability_at_least_one_boy_and_one_girl_in_four_children_l375_375938

theorem probability_at_least_one_boy_and_one_girl_in_four_children :
  ∀ (n : ℕ), n = 4 → 
  (∀ (p : ℚ), p = 1 / 2 →
  ((1 : ℚ) - ((p ^ n) + (p ^ n)) = 7 / 8)) :=
by
  intro n hn p hp
  rw [hn, hp]
  norm_num
  sorry

end probability_at_least_one_boy_and_one_girl_in_four_children_l375_375938


namespace gcd_possible_values_count_l375_375393

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375393


namespace second_largest_number_l375_375330

theorem second_largest_number (Yoongi Jungkook Yuna : ℕ) (hY : Yoongi = 7) (hJ : Jungkook = 6) (hN : Yuna = 9) :
  ∃ x, x ∈ {Yoongi, Jungkook, Yuna} ∧
       (∀ y, y ∈ {Yoongi, Jungkook, Yuna} → y ≥ x ∨ y ≤ x) ∧
       (∃ z, z ∈ {Yoongi, Jungkook, Yuna} ∧ z ≠ x ∧ z ≤ x) :=
begin
  sorry
end

end second_largest_number_l375_375330


namespace casting_roles_l375_375856

-- Define the overall assumptions and conditions
def roles (M : Type) (W : Type) (R : Type) [Fintype M] [Fintype W] [Fintype R] :=
  ∃ (men : ℕ) (women : ℕ) (roles_male : ℕ) (roles_either : ℕ) (total_roles : ℕ),
  (roles_male + roles_either = total_roles) ∧
  (roles_male = 3) ∧ 
  (roles_either = 2) ∧
  (total_roles = 5) ∧ 
  (men = 7) ∧ 
  (women = 5)

-- Main theorem to prove the total number of ways to assign roles.
theorem casting_roles : 
  roles ℕ ℕ ℕ → ∃ (n : ℕ), n = 15120 :=
by
  intro h
  -- Using the assumed conditions
  rcases h with ⟨men, women, roles_male, roles_either, total_roles, h1, h2, h3, h4, h5, h6⟩
  -- Cast roles_male can be done in 210 ways, and roles_either in 72 ways, so total ways is 15120
  use 15120
  sorry

end casting_roles_l375_375856


namespace sine_shift_graph_l375_375779

theorem sine_shift_graph (x : ℝ) : 
  (shift_right_by (sin (3 * x)) (π / 9) = sin (3 * x - π / 3)) := 
sorry

end sine_shift_graph_l375_375779


namespace sum_x_coords_of_A_l375_375785

open Real

def area_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  abs ((1 / 2) * ((fst A - fst C) * (snd B - snd C) - (fst B - fst C) * (snd A - snd C)))

def line_eq (P Q : (ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  let a : ℝ := snd Q - snd P
  let b : ℝ := fst P - fst Q
  let c : ℝ := fst Q * snd P - fst P * snd Q
  (a, b, c)

def point_line_distance (A : (ℝ × ℝ)) (line_coeff : ℝ × ℝ × ℝ) : ℝ :=
  let ax, by, c := line_coeff
  abs (ax * fst A + by * snd A + c) / sqrt (ax^2 + by^2)

theorem sum_x_coords_of_A :
  let B := (0, 0)
  let C := (334, 0)
  let F := (1020, 570)
  let G := (1031, 581)
  let area_ABC := 3010
  let area_AFG := 9030
  let lines_eq_FG := line_eq F G
  ∃ A : (ℝ × ℝ),
    (area_triangle A B C = area_ABC) →
    (area_triangle A F G = area_AFG)
    → (∑ A, (fst A) = 2336) := sorry

end sum_x_coords_of_A_l375_375785


namespace time_after_2051_hours_l375_375768

theorem time_after_2051_hours (h₀ : 9 ≤ 11): 
  (9 + 2051 % 12) % 12 = 8 :=
by {
  -- proving the statement here
  sorry
}

end time_after_2051_hours_l375_375768


namespace convert_fraction_to_decimal_l375_375058

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375058


namespace highest_possible_score_l375_375123

def grid_value (i j : ℕ) : ℕ := i * j

def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 + 1 = y2 ∨ y1 = y2 + 1)) ∨
  (y1 = y2 ∧ (x1 + 1 = x2 ∨ x1 = x2 + 1))

def valid_path (path : List (ℕ × ℕ)) : Prop :=
  match path with
  | [] => False
  | [p] => p = (6, 1) -- Start at the bottom-left corner
  | _ => 
    path.head = (6, 1) ∧ -- Start at the bottom-left corner
    path.getLast! = (1, 6) ∧ -- End at the top-right corner
    (path.length = arraySize) ∧ -- No square is visited more than once
    -- Consecutive squares share an edge
    ∀ i, i < path.length - 1 → is_adjacent (path.get! i).1 (path.get! i).2 (path.get! (i + 1)).1 (path.get! (i + 1)).2

def highest_path_score (path : List (ℕ × ℕ)) : ℕ :=
  path.foldr (λ (coord : ℕ × ℕ) acc => acc + grid_value coord.1 coord.2) 0

theorem highest_possible_score : ∃ (path : List (ℕ × ℕ)), valid_path path ∧ highest_path_score path = 439 :=
sorry

end highest_possible_score_l375_375123


namespace doritos_in_each_pile_l375_375690

theorem doritos_in_each_pile:
  (total_chips : ℕ) (one_quarter_chips : ℕ) (piles : ℕ) 
  (h1 : total_chips = 80) 
  (h2 : one_quarter_chips = total_chips / 4) 
  (h3 : piles = 4) :
  one_quarter_chips / piles = 5  := 
sorry

end doritos_in_each_pile_l375_375690


namespace fraction_equals_decimal_l375_375075

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375075


namespace evaluate_polynomial_l375_375967

noncomputable def polynomial_evaluation : Prop :=
∀ (x : ℝ), x^2 - 3*x - 9 = 0 ∧ 0 < x → (x^4 - 3*x^3 - 9*x^2 + 27*x - 8) = (65 + 81*(Real.sqrt 5))/2

theorem evaluate_polynomial : polynomial_evaluation :=
sorry

end evaluate_polynomial_l375_375967


namespace solve_for_y_l375_375718

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l375_375718


namespace phase_shift_sin_l375_375979

theorem phase_shift_sin (x : ℝ) : 
  ∃ φ : ℝ, (sin (5 * (x - φ)) = sin (5 * x - π / 2)) ∧ (φ = π / 10 ∨ φ = -π / 10) :=
by
  sorry

end phase_shift_sin_l375_375979


namespace acute_triangles_from_cuboid_vertices_l375_375475

/-- Prove the number of acute triangles that can be formed
from the vertices of a rectangular cuboid is 0. -/
theorem acute_triangles_from_cuboid_vertices : 
  ∀ (cuboid_vertices : Finset (ℝ × ℝ × ℝ)), 
  cuboid_vertices.card = 8 → 
  ∑ (t : Finset ℝ × Finset ℝ × Finset ℝ) 
    in cuboid_vertices.powerset.filter (λ s, s.card = 3), 
    (if is_acute_triangle t then 1 else 0) = 0 :=
by 
  intro cuboid_vertices h_card 
  sorry

end acute_triangles_from_cuboid_vertices_l375_375475


namespace rosy_fish_count_l375_375687

theorem rosy_fish_count (lilly_fish total_fish : ℕ) (h1 : lilly_fish = 10) (h2 : total_fish = 21) : 
  ∃ rosy_fish : ℕ, rosy_fish = total_fish - lilly_fish ∧ rosy_fish = 11 :=
by
  have h3 : total_fish - lilly_fish = 11 := by
    rw [h1, h2]
    norm_num
  exact ⟨total_fish - lilly_fish, h3, h3⟩

end rosy_fish_count_l375_375687


namespace painting_time_for_rose_l375_375961

theorem painting_time_for_rose :
  ∃ R : ℕ, (17 * 5 + 10 * R + 6 * 3 + 20 * 2 = 213) ∧ (R = 7) :=
by {
  existsi 7,
  split,
  {
    -- Given equation: 17 * 5 + 10 * R + 6 * 3 + 20 * 2 = 213
    calc
      17 * 5 + 10 * 7 + 6 * 3 + 20 * 2
        = 85 + 10 * 7 + 6 * 3 + 20 * 2   : by norm_num
    ... = 85 + 70 + 6 * 3 + 20 * 2       : by norm_num
    ... = 85 + 70 + 18 + 20 * 2          : by norm_num
    ... = 85 + 70 + 18 + 40              : by norm_num
    ... = 213                            : by norm_num
  },
  {
    -- Prove R = 7
    refl
  }
}

end painting_time_for_rose_l375_375961


namespace find_a_b_l375_375150

noncomputable def f (a b x : ℝ) := b * a^x

def passes_through (a b : ℝ) : Prop :=
  f a b 1 = 27 ∧ f a b (-1) = 3

theorem find_a_b (a b : ℝ) (h : passes_through a b) : 
  a = 3 ∧ b = 9 :=
  sorry

end find_a_b_l375_375150


namespace dice_sum_probability_l375_375607

theorem dice_sum_probability : 
  let dice_faces := finset.range 1 7 in 
  let sum_17 (a b c : ℕ) := a + b + c = 17 in 
  let is_valid_dice (n : ℕ) := n ∈ dice_faces in 
  (finset.card (finset.filter (λ (t : ℕ × ℕ × ℕ), sum_17 t.1 t.2 t.3 ∧ is_valid_dice t.1 ∧ is_valid_dice t.2 ∧ is_valid_dice t.3) (finset.product (finset.product dice_faces dice_faces) dice_faces)) : ℚ) = 1/72 :=
sorry

end dice_sum_probability_l375_375607


namespace solve_x_values_l375_375515

theorem solve_x_values : ∀ (x : ℝ), (x + 45 / (x - 4) = -10) ↔ (x = -1 ∨ x = -5) :=
by
  intro x
  sorry

end solve_x_values_l375_375515


namespace milburg_population_l375_375333

/-- Number of grown-ups in Milburg --/
def grownUps : ℕ := 5256

/-- Number of children in Milburg --/
def children : ℕ := 2987

/-- Total number of people in Milburg --/
def totalPeople : ℕ := grownUps + children

theorem milburg_population : totalPeople = 8243 := by
  have h1 : grownUps = 5256 := rfl
  have h2 : children = 2987 := rfl
  have h3 : totalPeople = grownUps + children := rfl
  have h4 : grownUps + children = 8243 := by
    calc
      5256 + 2987 = 8243 := by sorry -- Proof step to be filled in
  exact h4

end milburg_population_l375_375333


namespace find_k_l375_375195

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l375_375195


namespace problem_solution_l375_375109

-- Definitions from the problem statement
def sum_of_digits (k : ℕ) : ℕ :=
  (k.digits 10).sum

def f1 (k : ℕ) : ℕ :=
  (sum_of_digits k) ^ 2

noncomputable def fn : ℕ → ℕ → ℕ
| 1, k => f1 k
| (n+1), k => f1 (fn n k)

-- The leaned statement
theorem problem_solution : fn 1988 11 = 169 :=
by
  sorry

end problem_solution_l375_375109


namespace quadratic_root_conditions_l375_375597

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k + 2
  let b := 4
  let c := 1
  (a ≠ 0) ∧ (b^2 - 4*a*c > 0)

theorem quadratic_root_conditions (k : ℝ) :
  quadratic_has_two_distinct_real_roots k ↔ k < 2 ∧ k ≠ -2 := 
by
  sorry

end quadratic_root_conditions_l375_375597


namespace probability_sum_is_square_l375_375305

theorem probability_sum_is_square (n : ℕ) (h1 : n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12) : 
  let favorable_outcomes := 7
  let total_possible_outcomes := 36
  (favorable_outcomes / total_possible_outcomes = 7 / 36) :=
by {
  have square_sums : list ℕ := [4, 9],
  have favorable_count : list (ℕ × ℕ) := [(1, 3), (2, 2), (3, 1), (3, 6), (4, 5), (5, 4), (6, 3)],
  exact sorry
}

end probability_sum_is_square_l375_375305


namespace monster_family_eyes_count_l375_375658

theorem monster_family_eyes_count :
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  (mom_eyes + dad_eyes) + (num_kids * kid_eyes) = 16 :=
by
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  have parents_eyes : mom_eyes + dad_eyes = 4 := by rfl
  have kids_eyes : num_kids * kid_eyes = 12 := by rfl
  show parents_eyes + kids_eyes = 16
  sorry

end monster_family_eyes_count_l375_375658


namespace mean_tasks_b_l375_375340

variable (a b : ℕ)
variable (m_a m_b : ℕ)
variable (h1 : a + b = 260)
variable (h2 : a = 3 * b / 10 + b)
variable (h3 : m_a = 80)
variable (h4 : m_b = 12 * m_a / 10)

theorem mean_tasks_b :
  m_b = 96 := by
  -- This is where the proof would go
  sorry

end mean_tasks_b_l375_375340


namespace fraction_to_decimal_l375_375063

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375063


namespace colin_speed_l375_375488

variable (B T Br C D : ℝ)

-- Given conditions
axiom cond1 : C = 6 * Br
axiom cond2 : Br = (1/3) * T^2
axiom cond3 : T = 2 * B
axiom cond4 : D = (1/4) * C
axiom cond5 : B = 1

-- Prove Colin's speed C is 8 mph
theorem colin_speed :
  C = 8 :=
by
  sorry

end colin_speed_l375_375488


namespace workers_contribution_l375_375412

theorem workers_contribution (N C : ℕ) 
(h1 : N * C = 300000) 
(h2 : N * (C + 50) = 360000) : 
N = 1200 :=
sorry

end workers_contribution_l375_375412


namespace gcd_values_count_l375_375368

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375368


namespace mixed_fraction_product_example_l375_375879

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375879


namespace ratio_of_volumes_l375_375860

theorem ratio_of_volumes (s : ℝ) (hs : s > 0) :
  let r_s := s / 2
  let r_c := s / 2
  let V_sphere := (4 / 3) * π * (r_s ^ 3)
  let V_cylinder := π * (r_c ^ 2) * s
  let V_total := V_sphere + V_cylinder
  let V_cube := s ^ 3
  V_total / V_cube = (5 * π) / 12 := by {
    -- Given the conditions and expressions
    sorry
  }

end ratio_of_volumes_l375_375860


namespace restore_fractions_l375_375891

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375891


namespace find_k_l375_375096

noncomputable def matrix_eigenvalues_nonzero_vector (k : ℝ) : Prop :=
  ∃ (v : ℝ × ℝ), v ≠ 0 ∧
    let (x, y) := v in
    (3 * x + 5 * y = k * x) ∧
    (4 * x + 3 * y = k * y)

theorem find_k : ∀ (k : ℝ), matrix_eigenvalues_nonzero_vector k ↔ (k = 3 + 2 * Real.sqrt 5 ∨ k = 3 - 2 * Real.sqrt 5) :=
sorry

end find_k_l375_375096


namespace fraction_to_decimal_l375_375050

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375050


namespace A_lies_on_line_T1T2_l375_375262

variables (O A T1 T2 : Point)
variables (r OA OA_star : ℝ)
variables (ω : Circle)

-- Definitions related to the problem
def inversion_radius := r
def inverse_point_relation := OA * OA_star = r ^ 2
def inside_circle := OA < r
def outside_circle := OA_star > r
def tangents_from_A_star := is_tangent (A_star, ω, T1) ∧ is_tangent (A_star, ω, T2)

-- Theorem Statement
theorem A_lies_on_line_T1T2
  (h1 : inside_circle A)
  (h2 : inverse_point_relation A A_star)
  (h3 : tangents_from_A_star ) :
  lies_on_line A T1 T2 :=
sorry

end A_lies_on_line_T1T2_l375_375262


namespace order_of_a_b_c_l375_375672

-- Definitions based on the given conditions
def a : ℝ := real.sqrt 0.6
def b : ℝ := real.sqrt 0.7
def c : ℝ := real.log 0.7

-- The proof problem statement
theorem order_of_a_b_c : c < a ∧ a < b :=
by
  -- The proof will go here, but is omitted in this task.
  sorry

end order_of_a_b_c_l375_375672


namespace equilateral_triangle_side_length_l375_375757

theorem equilateral_triangle_side_length (perimeter : ℝ) (h : perimeter = 2) : abs (perimeter / 3 - 0.67) < 0.01 :=
by
  -- The proof will go here.
  sorry

end equilateral_triangle_side_length_l375_375757


namespace solve_for_y_l375_375717

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l375_375717


namespace power_of_complex_l375_375104

theorem power_of_complex : (1 + complex.I * √3)^3 = -8 :=
by
  -- Proof is skipped
  sorry

end power_of_complex_l375_375104


namespace find_k_l375_375172

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l375_375172


namespace distinct_intersection_points_count_l375_375956

def intersect_points (f g : ℝ → ℝ → Prop) : ℕ :=
  {p : ℝ × ℝ | f p.1 p.2 ∧ g p.1 p.2}.to_finset.card

noncomputable def eq1 := λ x y : ℝ, (x - y + 3) * (3 * x + y - 7) = 0
noncomputable def eq2 := λ x y : ℝ, (x + y - 3) * (2 * x - 5 * y + 12) = 0

theorem distinct_intersection_points_count :
  intersect_points eq1 eq2 = 4 :=
sorry

end distinct_intersection_points_count_l375_375956


namespace number_of_paths_l375_375786

-- Definitions of coordinates
def start : ℕ × ℕ := (0, 0)
def end : ℕ × ℕ := (6, 6)
def center : ℕ × ℕ := (3, 3)

-- Movement constraints (right or top)
def valid_moves : ℕ × ℕ → list (ℕ × ℕ)
| (x, y) => if x < 6 ∧ y < 6 then [(x + 1, y), (x, y + 1)] else if x < 6 then [(x + 1, y)] else if y < 6 then [(x, y + 1)] else []

-- Main theorem
theorem number_of_paths (start end center : ℕ × ℕ) (valid_moves : ℕ × ℕ → list (ℕ × ℕ)) : ℕ :=
  let moves_to_center := nat.choose (3 + 3) 3
  let moves_from_center := nat.choose (3 + 3) 3
  moves_to_center * moves_from_center

#eval number_of_paths start end center valid_moves -- Expected output: 400

end number_of_paths_l375_375786


namespace percent_pepperjack_l375_375650

-- Define the total number of cheese sticks
def total_cheese_sticks : ℕ := 15 + 30 + 45

-- Define the number of pepperjack cheese sticks
def pepperjack_sticks : ℕ := 45

-- The percentage chance that a randomly picked cheese stick is pepperjack
theorem percent_pepperjack : (pepperjack_sticks * 100) / total_cheese_sticks = 50 := by
  calc
    (pepperjack_sticks * 100) / total_cheese_sticks = (45 * 100) / 90 : by rw [pepperjack_sticks, total_cheese_sticks]
    ... = 4500 / 90 : rfl
    ... = 50 : by norm_num

end percent_pepperjack_l375_375650


namespace series_convergence_l375_375762

theorem series_convergence (a : ℕ → ℝ) 
  (h_monotonic : ∀ n, a n ≥ a (n + 1)) 
  (h_sum_convergence : ∃ L : ℝ, has_sum (λ n, a n) L) : 
  ∃ M : ℝ, has_sum (λ n, n * (a n - a (n + 1))) M := 
sorry

end series_convergence_l375_375762


namespace total_lattice_points_in_region_l375_375283

def is_lattice_point (p : ℤ × ℤ) : Prop :=
  p.1 ≥ 0 ∧ p.1 ≤ 4 ∧ p.2 ≥ 0 ∧ p.2 ≤ p.1^2

theorem total_lattice_points_in_region : 
  (Finset.card (Finset.filter is_lattice_point (Finset.univ : Finset (ℤ × ℤ)))) = 35 :=
by
  sorry

end total_lattice_points_in_region_l375_375283


namespace minimum_fence_length_l375_375461

theorem minimum_fence_length {x y : ℝ} (hxy : x * y = 100) : 2 * (x + y) ≥ 40 :=
by
  sorry

end minimum_fence_length_l375_375461


namespace solve_for_x_l375_375715

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x - 4) / (x + 8)) : x = -32 / 13 :=
sorry

end solve_for_x_l375_375715


namespace XiaoMing_wins_when_drawing_4_XiaoMing_wins_when_drawing_6_l375_375770

-- Definitions for the conditions in a)
def total_slips : ℕ := 7
def remaining_slips (drawn : ℕ) : ℕ := total_slips - 1

-- Questions translated to Lean statements
theorem XiaoMing_wins_when_drawing_4 :
  (Π drawn, drawn = 4 → (remaining_slips drawn = 6) →
    (probability_of_XiaoMing_winning (drawn := 4) = 1 / 2) ∧ (probability_of_XiaoYing_winning (drawn := 4) = 1 / 2)) := sorry

theorem XiaoMing_wins_when_drawing_6 :
  (Π drawn, drawn = 6 → (remaining_slips drawn = 6) →
    (probability_of_XiaoMing_winning (drawn := 6) = 5 / 6) ∧ (probability_of_XiaoYing_winning (drawn := 6) = 1 / 6)) := sorry

-- Definitions for the probability outcomes
def probability_of_XiaoMing_winning {drawn : ℕ} : ℚ := 
  if drawn = 4 then 1 / 2 else 
  if drawn = 6 then 5 / 6 else 0

def probability_of_XiaoYing_winning {drawn : ℕ} : ℚ := 
  if drawn = 4 then 1 / 2 else 
  if drawn = 6 then 1 / 6 else 0

end XiaoMing_wins_when_drawing_4_XiaoMing_wins_when_drawing_6_l375_375770


namespace two_digit_even_factors_count_l375_375201

theorem two_digit_even_factors_count : ∃ n, n = 84 ∧
  (∀ k, 10 ≤ k ∧ k ≤ 99 → (¬ (∃ m, m^2 = k) → (k has_even_number_of_factors ↔ n = 84)))
:=
sorry

end two_digit_even_factors_count_l375_375201


namespace evaluate_expression_l375_375965

theorem evaluate_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ( (1 / a^2 + 1 / b^2)⁻¹ = a^2 * b^2 / (a^2 + b^2) ) :=
by
  sorry

end evaluate_expression_l375_375965


namespace bx_eq_cy_l375_375273

-- Define the geometric setting
variables {A B C D E F Q X Y : Point}
variable (acute_triangle : Triangle A B C)
variable (D_foot : AltitudeFoot A B C D)
variable (E_foot : AltitudeFoot B A C E)
variable (F_foot : AltitudeFoot C A B F)
variable (Q_on_AD : OnSegment Q A D)
variable (circle_QED_X : ∃ c : Circle, OnCircle Q D E c ∧ c.IntersectsLine BC X)
variable (circle_QFD_Y : ∃ c : Circle, OnCircle Q D F c ∧ c.IntersectsLine BC Y)

-- The theorem we aim to prove
theorem bx_eq_cy :
  BX = CY :=
sorry

end bx_eq_cy_l375_375273


namespace smallest_four_digit_equiv_mod_8_l375_375806

theorem smallest_four_digit_equiv_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 :=
by
  -- We state the assumptions and final goal
  use 1003
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  · refl
  sorry

end smallest_four_digit_equiv_mod_8_l375_375806


namespace find_root_floor_l375_375674

noncomputable def g (x : ℝ) := Real.sin x - Real.cos x + 4 * Real.tan x

theorem find_root_floor :
  ∃ s : ℝ, (g s = 0) ∧ (π / 2 < s) ∧ (s < 3 * π / 2) ∧ (Int.floor s = 3) :=
  sorry

end find_root_floor_l375_375674


namespace coloring_count_in_3x3_grid_l375_375950

theorem coloring_count_in_3x3_grid (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : 
  ∃ count : ℕ, count = 15 ∧ ∀ (cells : Finset (Fin n × Fin m)),
  (cells.card = 3 ∧ ∀ (c1 c2 : Fin n × Fin m), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 → 
  (c1.fst ≠ c2.fst ∧ c1.snd ≠ c2.snd)) → cells.card ∣ count :=
sorry

end coloring_count_in_3x3_grid_l375_375950


namespace solve_fractions_l375_375915

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375915


namespace min_lamps_to_all_off_l375_375005

theorem min_lamps_to_all_off (n : ℕ) (hn : 2 ≤ n) :
  ∃ (initial_on : set (ℕ × ℕ)), (∀ i j, (i = 1 ∨ i = n ∨ j = 1 ∨ j = n) → ((i, j) ∈ initial_on ↔ (i = 1 ∧ j = 1) ∨ (i = n ∧ j = n)) 
  ∧ (∀ (row_or_col : bool) (idx : ℕ), ∀ (i j), (i, j) ∈ initial_on → ((i = idx ∧ row_or_col = tt) ∨ (j = idx ∧ row_or_col = ff)) → (i, j) ∉ initial_on)
  ∧ (∃ (k : ℕ), k = 2 * n - 4 ∧ ∀ i j, i ≠ n → j ≠ n → (i, j) ∉ initial_on)) :=
sorry

end min_lamps_to_all_off_l375_375005


namespace dice_sum_probability_l375_375605

theorem dice_sum_probability :
  let probability_sum_17 (dice : list ℕ) := (dice.sum = 17 ∧ dice.length = 3) in
  let total_outcomes := 6^3 in
  let favorable_outcomes := 3 in -- (6,6,5), (6,5,6), and (5,6,6)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 72 :=
by
  sorry

end dice_sum_probability_l375_375605


namespace surface_area_comparison_l375_375799

noncomputable def gamma (x : ℝ) : ℝ := 
  sorry -- Define the Gamma function in a suitable way if necessary

noncomputable def volume_n_sphere (n : ℕ) : ℝ := 
  π^(n/2) / (gamma ((n/2) + 1))

noncomputable def surface_area_n_sphere (n : ℕ) : ℝ :=
  n * volume_n_sphere n

theorem surface_area_comparison :
  surface_area_n_sphere 2018 > surface_area_n_sphere 2017 :=
by 
  sorry -- placeholder for the actual proof

end surface_area_comparison_l375_375799


namespace max_points_condition_l375_375518

noncomputable def MaxPoints : ℕ :=
  18

theorem max_points_condition (n : ℕ)
  (points : fin n → (ℝ × ℝ))
  (color : fin n → ℕ) :
  (∀ i j k : fin n, i ≠ j ∧ j ≠ k ∧ k ≠ i -> ¬ collinear (points i) (points j) (points k)) →
  (∀ i j k : fin n, color i = 0 ∧ color j = 0 ∧ color k = 0 → 
    ∃ l : fin n, color l = 1 ∧ (convex_hull (points '' {i, j, k}) ∋ points l)) →
  (∀ i j k : fin n, color i = 1 ∧ color j = 1 ∧ color k = 1 → 
    ∃ l : fin n, color l = 2 ∧ (convex_hull (points '' {i, j, k}) ∋ points l)) →
  (∀ i j k : fin n, color i = 2 ∧ color j = 2 ∧ color k = 2 → 
    ∃ l : fin n, color l = 0 ∧ (convex_hull (points '' {i, j, k}) ∋ points l)) →
  n ≤ MaxPoints :=
sorry

end max_points_condition_l375_375518


namespace mixed_fraction_product_l375_375901

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375901


namespace angle_CMD_l375_375676

-- Definitions for points and quadrilateral
variables (A B C D M : Type) [point A] [point B] [point C] [point D] [point M]

-- Intersection of diagonals
def is_intersection_of_diagonals (M A B C D : Type) : Prop :=
  ∃ (P Q : Type), (diagonal A C intersects diagonal B D) ∧ (M = P ∩ Q)

-- Angle equality conditions
def angles_are_equal (A B D : Type) : Prop :=
  length (A B) = length (A D) ∧ length (B C) = length (A D) 

-- Segment equality conditions
def segments_are_equal (D M C : Type) : Prop :=
  length (D M) = length (M C)

-- Non equality of specified angles
def non_adjacent_angles_not_equal (A B C D : Type) : Prop :=
  ∠CAB ≠ ∠DBA

-- Main problem statement
theorem angle_CMD (M A B C D : Type) [is_intersection_of_diagonals M A B C D]
  (equal_sides : angles_are_equal A B D) (equal_segments : segments_are_equal D M C)
  (neq_angles : non_adjacent_angles_not_equal A B C D) :
  ∠CMD = 120 :=
sorry

end angle_CMD_l375_375676


namespace radius_range_of_circle_l375_375544

theorem radius_range_of_circle (r : ℝ) :
  (∃ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 ∧ 
    (∃ a b : ℝ, 4*a - 3*b - 2 = 0 ∧ ∃ c d : ℝ, 4*c - 3*d - 2 = 0 ∧ 
      (a - x)^2 + (b - y)^2 = 1 ∧ (c - x)^2 + (d - y)^2 = 1 ∧
       a ≠ c ∧ b ≠ d)) ↔ 4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l375_375544


namespace find_side_length_c_l375_375618

namespace TriangleProblem

variables {α β γ a b c : ℝ}
variables [fact (a > 0)] [fact (b > 0)] [fact (π / 180 > 0 : ℝ)]  -- Ensuring angles are in degrees and sides are positive

-- Conditions
axiom angle_sum (h1 : α + β + γ = 180) : true
axiom given_equation (h2 : 3 * α + 2 * β = 180) : true
axiom sides (h3 : a = 2) (h4 : b = 3) : true

-- The theorem to prove
theorem find_side_length_c (h1 : α + β + γ = 180) (h2 : 3 * α + 2 * β = 180) (h3 : a = 2) (h4 : b = 3) : c = 4 :=
sorry

end TriangleProblem

end find_side_length_c_l375_375618


namespace find_k_l375_375178

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l375_375178


namespace fraction_to_decimal_l375_375019

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375019


namespace num_biology_books_is_15_l375_375772

-- conditions
def num_chemistry_books : ℕ := 8
def total_ways : ℕ := 2940

-- main statement to prove
theorem num_biology_books_is_15 : ∃ B: ℕ, (B * (B - 1)) / 2 * (num_chemistry_books * (num_chemistry_books - 1)) / 2 = total_ways ∧ B = 15 :=
by
  sorry

end num_biology_books_is_15_l375_375772


namespace pyramid_volume_l375_375493

/-- Given the vertices of a triangle and its midpoints, calculate the volume of the folded triangular pyramid. -/
theorem pyramid_volume
  (A B C : ℝ × ℝ)
  (D E F : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (24, 0))
  (hC : C = (12, 16))
  (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hE : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (hF : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (area_ABC : ℝ)
  (h_area : area_ABC = 192)
  : (1 / 3) * area_ABC * 8 = 512 :=
by sorry

end pyramid_volume_l375_375493


namespace complex_square_real_iff_l375_375753

theorem complex_square_real_iff (a b : ℝ) : 
  (∀ z : ℂ, z = a + b * complex.I → (z * z).im = 0) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end complex_square_real_iff_l375_375753


namespace simplify_fraction_l375_375712

variable {x y : ℝ}

theorem simplify_fraction (hx : x = 3) (hy : y = 4) : (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end simplify_fraction_l375_375712


namespace extreme_values_of_f_l375_375570

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4 * x + 4

theorem extreme_values_of_f :
  (∃ x_max x_min, x_max = -2 ∧ f x_max = 28/3 ∧ x_min = 2 ∧ f x_min = -4/3) ∧
  (∀ x ∈ set.Icc 0 3, f x ≤ 4 ∧ f x ≥ -4/3) :=
by
  sorry

end extreme_values_of_f_l375_375570


namespace value_of_a_b_c_l375_375119

theorem value_of_a_b_c (a b c : ℚ) (h₁ : |a| = 2) (h₂ : |b| = 2) (h₃ : |c| = 3) (h₄ : b < 0) (h₅ : 0 < a) :
  a + b + c = 3 ∨ a + b + c = -3 :=
by
  sorry

end value_of_a_b_c_l375_375119


namespace milburg_population_l375_375334

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l375_375334


namespace solve_fractions_l375_375911

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375911


namespace compute_expression_l375_375670

theorem compute_expression : 
  let a := (4 : ℚ) / (7 : ℚ)
  let b := (5 : ℚ) / (6 : ℚ)
  (a^3 * b^(-2))^2 = (2304 ^ 2) / (8575 ^ 2) := by
  sorry

end compute_expression_l375_375670


namespace restore_original_problem_l375_375872

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375872


namespace range_of_x_f_inequality_l375_375139

noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then -Real.log (1 - x) else Real.log (1 + x)

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 else g x

theorem range_of_x_f_inequality :
  {x : ℝ | f (2 - x^2) > f x} = set.Ioo (-2 : ℝ) (1 : ℝ) :=
by
  sorry

end range_of_x_f_inequality_l375_375139


namespace triangle_is_isosceles_l375_375609

theorem triangle_is_isosceles {a b c : ℝ} {A B C : ℝ} (h1 : b * Real.cos A = a * Real.cos B) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end triangle_is_isosceles_l375_375609


namespace mixed_fraction_product_l375_375896

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375896


namespace intersection_in_fourth_quadrant_l375_375624

-- We define the points and required lines
def line_l (x : ℝ) : ℝ := - (5 / 3) * x - 5
def x_coord_of_line_l' : ℝ := 2

-- A function that identifies the quadrant of a point (x, y)
def quadrant (x y : ℝ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

-- Proving the intersection lies in the fourth quadrant
theorem intersection_in_fourth_quadrant :
  quadrant 2 (line_l 2) = "Fourth quadrant" :=
by
  sorry

end intersection_in_fourth_quadrant_l375_375624


namespace fraction_to_decimal_l375_375037

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375037


namespace no_injective_function_exists_l375_375502

theorem no_injective_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f (x^2) - (f x)^2 ≥ 1/4) ∧ (∀ x y, f x = f y → x = y) := 
sorry

end no_injective_function_exists_l375_375502


namespace gcd_values_count_l375_375369

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375369


namespace translate_cosine_function_l375_375782

theorem translate_cosine_function (x : ℝ) :
  let f := λ x, Real.cos (2 * x - π / 3)
  let g := λ x, Real.cos (2 * (x + π / 6) - π / 3)
  let h := λ x, Real.cos (2 * x)
  g x = h x :=
by
  sorry

end translate_cosine_function_l375_375782


namespace complex_product_l375_375245

theorem complex_product : ∀ (i : ℂ), i^2 = -1 → (3 + i) * (3 - i) = 10 :=
by 
  intros,
  sorry

end complex_product_l375_375245


namespace new_interest_rate_is_5_percent_l375_375319

-- Definitions from the given problem
def principal (interest : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  interest / (rate * time)

def newRate (new_interest : ℝ) (principal : ℝ) (time : ℝ) : ℝ :=
  new_interest / (principal * time)

-- Given conditions
axiom interest_1_year : ℝ := 405
axiom rate_1_year : ℝ := 0.045
axiom time_1_year : ℝ := 1
axiom additional_interest : ℝ := 45

-- Calculated from given conditions
def principal_amount : ℝ := principal interest_1_year rate_1_year time_1_year
def total_new_interest : ℝ := interest_1_year + additional_interest

-- Proof statement
theorem new_interest_rate_is_5_percent : newRate total_new_interest principal_amount time_1_year = 0.05 := by
  sorry

end new_interest_rate_is_5_percent_l375_375319


namespace variance_boys_greater_than_girls_l375_375612

noncomputable theory

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length : ℝ)

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x, (x - m)^2)).sum / (l.length : ℝ)

theorem variance_boys_greater_than_girls :
  variance boys_scores > variance girls_scores := by
  sorry

end variance_boys_greater_than_girls_l375_375612


namespace rachelle_needs_15_pounds_l375_375293
-- Importing the necessary library

-- Defining the conditions
def meatPerHamburger (totalMeat: ℝ) (totalHamburgers: ℕ) : ℝ :=
  totalMeat / totalHamburgers

def totalMeatNeeded (meatPerHamburger: ℝ) (hamburgers: ℕ) : ℝ :=
  meatPerHamburger * hamburgers

-- Stating the proof problem
theorem rachelle_needs_15_pounds :
  meatPerHamburger 5 10 * 30 = 15 :=
by
  sorry

end rachelle_needs_15_pounds_l375_375293


namespace trigonometric_expression_l375_375143

variables {a : ℝ} (h : a ≠ 0)
def P : ℝ × ℝ := (a, -2a)
def A : ℝ × ℝ := (a, 2a)
def Q : ℝ × ℝ := (2a, a)

-- Define trigonometric functions based on given points and angles
def sin_alpha : ℝ := -2a / Real.sqrt (a ^ 2 + (-2a) ^ 2)
def cos_alpha : ℝ := a / Real.sqrt (a ^2 + (-2a) ^ 2)
def tan_alpha : ℝ := -2

def sin_beta : ℝ := a / Real.sqrt ((2a) ^ 2 + a ^ 2)
def cos_beta : ℝ := 2a / Real.sqrt ((2a) ^ 2 + a ^ 2)
def tan_beta : ℝ := 1 / 2

-- Prove the required expression
theorem trigonometric_expression : 
  sin_alpha * cos_alpha + sin_beta * cos_beta + tan_alpha * tan_beta = -1 := 
by {
  sorry
}

end trigonometric_expression_l375_375143


namespace sin_alpha_eq_l375_375711

theorem sin_alpha_eq :
  ∀ (α : ℝ) (E : ℂ), 
  (cos α = 1 / (2 * sin (Real.pi / 5))) ∧ 
  (E = Complex.exp (2 * Complex.pi * Complex.I / 5)) → 
  sin α = (E^2 - E^3) / (Complex.I * Real.sqrt 5) :=
begin
  intros α E h,
  cases h with h_cos h_E,
  sorry, -- Proof goes here
end

end sin_alpha_eq_l375_375711


namespace solve_fractions_l375_375917

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375917


namespace minimize_cost_l375_375353

def C (x : ℝ) : ℝ := 40 / (3 * x + 5)

def f (x : ℝ) : ℝ := 20 * C(x) + 6 * x

theorem minimize_cost : 
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 10 ∧ 
  (∀ y ∈ (set.Icc 1 10), f(x) ≤ f(y)) ∧ 
  f(x) = 70 :=
by
  sorry

end minimize_cost_l375_375353


namespace fraction_traditionalists_l375_375316

noncomputable def traditionalist_to_pop_ratio := 
  let ratio_A := 1 / 7
  let ratio_B := 2 / 5
  let ratio_C := 1 / 4
  let ratio_D := 1 / 3
  let ratio_E := 1 / 9
  let ratio_F := 3 / 7
  let ratio_G := 3 / 8
  let ratio_H := 1 / 5
  let ratio_I := 2 / 7
  let ratio_J := 1 / 8

  let pop_A := 700000
  let pop_B := 350000
  let pop_C := 500000
  let pop_D := 600000
  let pop_E := 900000
  let pop_F := 200000
  let pop_G := 440000
  let pop_H := 750000
  let pop_I := 630000
  let pop_J := 720000

  let traditionalists := 
    pop_A * ratio_A +
    pop_B * ratio_B +
    pop_C * ratio_C +
    pop_D * ratio_D +
    pop_E * ratio_E +
    pop_F * ratio_F +
    pop_G * ratio_G +
    pop_H * ratio_H +
    pop_I * ratio_I +
    pop_J * ratio_J

  let total_population := 
    pop_A +
    pop_B +
    pop_C +
    pop_D +
    pop_E +
    pop_F +
    pop_G +
    pop_H +
    pop_I +
    pop_J

  traditionalists / total_population

theorem fraction_traditionalists (h : traditionalist_to_pop_ratio ≈ 0.2335) : true :=
  sorry

end fraction_traditionalists_l375_375316


namespace sufficient_but_not_necessary_l375_375929

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ (¬(a > b) → ¬(a > b + 1)) :=
by
  sorry

end sufficient_but_not_necessary_l375_375929


namespace fraction_to_decimal_l375_375042

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375042


namespace arithmetic_sequence_term_l375_375838

theorem arithmetic_sequence_term (a1 d n an : ℕ) (h1 : a1 = 1) (h2 : d = 3)
  (h3 : an = a1 + (n - 1) * d) (h4 : an = 2011) : n = 671 :=
by {
  subst h1,
  subst h2,
  subst h4,
  rw add_comm at h3,
  linarith,
  sorry
}

end arithmetic_sequence_term_l375_375838


namespace monster_family_eyes_count_l375_375657

theorem monster_family_eyes_count :
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  (mom_eyes + dad_eyes) + (num_kids * kid_eyes) = 16 :=
by
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  have parents_eyes : mom_eyes + dad_eyes = 4 := by rfl
  have kids_eyes : num_kids * kid_eyes = 12 := by rfl
  show parents_eyes + kids_eyes = 16
  sorry

end monster_family_eyes_count_l375_375657


namespace asymptotes_of_hyperbola_l375_375237

open Real

-- Definitions of the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola (p x y : ℝ) : Prop := x^2 = 2 * p * y

noncomputable def focus (p : ℝ) := (0, p / 2)

-- The main statement with all conditions
theorem asymptotes_of_hyperbola
  (a b p : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hp : p > 0)
  (intersects : ∃ (x_A y_A x_B y_B : ℝ), hyperbola a b x_A y_A ∧ hyperbola a b x_B y_B ∧ parabola p x_A y_A ∧ parabola p x_B y_B)
  (condition : let F := focus p in ∃ (A B : ℝ × ℝ), |(A.1 - F.1)^2 + (A.2 - F.2)^2|.sqrt + |(B.1 - F.1)^2 + (B.2 - F.2)^2|.sqrt = 4 * |(0 - F.1)^2 + (0 - F.2)^2|.sqrt) :
  ∀ x y, (y = (sqrt 2 / 2) * x ∨ y = -(sqrt 2 / 2) * x) :=
begin
  -- Proof is omitted
  sorry
end

end asymptotes_of_hyperbola_l375_375237


namespace smallest_n_l375_375361

theorem smallest_n (n : ℕ) : 634 * n ≡ 1275 * n [MOD 30] ↔ n = 30 :=
by
  sorry

end smallest_n_l375_375361


namespace maxWater_l375_375984

def initialBuckets : Vector ℕ 5 := ⟨[1, 2, 3, 4, 5], by decide⟩

def isValidState (buckets : Vector ℕ 5) : Prop :=
  ∀ (i : Fin 5), buckets i ≤ 15

def canTriple (buckets : Vector ℕ 5) : Prop :=
  ∃ (i j : Fin 5), i ≠ j ∧ 3 * buckets i ≤ buckets i + buckets j

def performTriple (buckets : Vector ℕ 5) (i j : Fin 5) (h : buckets i + 2 * buckets j ≤ 15) : Vector ℕ 5 :=
  buckets.updateNth i (buckets i + 2 * buckets j) >>= λ bkt, bkt.updateNth j 0

theorem maxWater : ∃ buckets : Vector ℕ 5, isValidState buckets ∧
  (∀ (i j : Fin 5) (h : buckets i + 2 * buckets j ≤ 15),
    isValidState (performTriple buckets i j h)) ∧
  (∀ (i : Fin 5), buckets i ≤ 12) ∧
  (∃ (i : Fin 5), buckets i = 9 ∨ buckets i = 12)
  := sorry

end maxWater_l375_375984


namespace total_population_l375_375337

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end total_population_l375_375337


namespace simplify_expression_l375_375001

theorem simplify_expression :
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 125 / 13 :=
by
  sorry

end simplify_expression_l375_375001


namespace largest_of_four_numbers_l375_375543

theorem largest_of_four_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max (a^2 + b^2) (2 * a * b)) a) (1 / 2) = a^2 + b^2 :=
by
  sorry

end largest_of_four_numbers_l375_375543


namespace sum_of_digits_a_plus_b_l375_375664

def is_pos_root (m : ℤ) (x : ℤ) := 
  x > 0 ∧ (x^2 + m * x + 2020 = 0)

def is_neg_root (n : ℤ) (x : ℤ) := 
  x < 0 ∧ (x^2 + 2020 * x + n = 0)

def A : Set ℤ := {m | ∃ p q : ℤ, is_pos_root m p ∧ is_pos_root m q ∧ p ≠ q}
def B : Set ℤ := {n | ∃ r s : ℤ, is_neg_root n r ∧ is_neg_root n s ∧ r ≠ s}

noncomputable def a : ℤ := Sup A -- largest element of A
noncomputable def b : ℤ := Inf B -- smallest element of B

def sum_of_digits (n : ℤ) : ℕ :=
  (n.toNat.digits 10).sum

theorem sum_of_digits_a_plus_b : sum_of_digits (a + b) = 27 :=
  sorry

end sum_of_digits_a_plus_b_l375_375664


namespace number_of_cans_on_15th_layer_number_of_cans_each_layer_of_prism_rearrange_pentagonal_to_triangular_prism_l375_375480

def pentagonal_number (n : ℕ) : ℕ :=
  (3 * n * n - n) / 2

def sum_of_pentagonal_numbers (n : ℕ) : ℕ :=
  (n * (n+1) * (2*n+1)) / 6 - (n * (n+1)) / 2

-- Part (a): Proving the number of cans on the bottom, 15th layer is 330
theorem number_of_cans_on_15th_layer : pentagonal_number 15 = 330 :=
  sorry

-- Part (b): Proving the number of cans on each layer of the prism after rearranging is 120
theorem number_of_cans_each_layer_of_prism : (sum_of_pentagonal_numbers 15) / 15 = 120 :=
  sorry

-- Part (c): Proving that any pentagonal pyramid with layers l >= 2 can be rearranged into a triangular prism with the same number of layers
theorem rearrange_pentagonal_to_triangular_prism (l : ℕ) (hl : l ≥ 2) :
  (∑ k in finset.range l, pentagonal_number (k+1)) = l * (l * (l + 1) / 2) :=
  sorry

end number_of_cans_on_15th_layer_number_of_cans_each_layer_of_prism_rearrange_pentagonal_to_triangular_prism_l375_375480


namespace derivative_of_y_l375_375974

noncomputable def y (x : ℝ) : ℝ :=
  cos (real.cot 2) - (1 / 16) * (cos (8 * x))^2 / sin (16 * x)

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (4 * (sin (8 * x))^2) :=
by
  sorry

end derivative_of_y_l375_375974


namespace orange_juice_percentage_l375_375848

theorem orange_juice_percentage 
  (V : ℝ) 
  (W : ℝ) 
  (G : ℝ)
  (hV : V = 300)
  (hW: W = 0.4 * V)
  (hG: G = 105) : 
  (V - W - G) / V * 100 = 25 := 
by 
  -- We will need to use sorry to skip the proof and focus just on the statement
  sorry

end orange_juice_percentage_l375_375848


namespace find_weight_of_fourth_cat_l375_375648

noncomputable def weight_cat_1 : ℝ := 12
noncomputable def weight_cat_2 : ℝ := 12
noncomputable def weight_cat_3 : ℝ := 14.7
noncomputable def number_of_cats : ℝ := 4
noncomputable def average_weight : ℝ := 12

theorem find_weight_of_fourth_cat : ∃ W : ℝ, (weight_cat_1 + weight_cat_2 + weight_cat_3 + W) / number_of_cats = average_weight ∧ W = 9.3 := 
by
  use 9.3
  have h1 : weight_cat_1 + weight_cat_2 + weight_cat_3 = 38.7 := by norm_num
  have h2 : 4 * 12 = 48 := by norm_num
  have h3 : 38.7 + 9.3 = 48 := by norm_num
  rw [h1, h2, h3]
  norm_num
  split
  { norm_num }
  { norm_num }

end find_weight_of_fourth_cat_l375_375648


namespace exist_odd_prime_and_k_l375_375529

noncomputable def dist_to_nearest_integer (x : ℝ) : ℝ :=
  min (x - (floor x)) ((ceil x) - x)

theorem exist_odd_prime_and_k
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (p : ℕ) (hp : prime p) (kp : p % 2 = 1) (k : ℕ), k > 0 ∧
    dist_to_nearest_integer (a / (p ^ k)) +
    dist_to_nearest_integer (b / (p ^ k)) +
    dist_to_nearest_integer ((a + b) / (p ^ k)) = 1 :=
sorry

end exist_odd_prime_and_k_l375_375529


namespace alice_has_largest_result_l375_375926

def initial_number : ℕ := 15

def alice_transformation (x : ℕ) : ℕ := (x * 3 - 2 + 4)
def bob_transformation (x : ℕ) : ℕ := (x * 2 + 3 - 5)
def charlie_transformation (x : ℕ) : ℕ := (x + 5) / 2 * 4

def alice_final := alice_transformation initial_number
def bob_final := bob_transformation initial_number
def charlie_final := charlie_transformation initial_number

theorem alice_has_largest_result :
  alice_final > bob_final ∧ alice_final > charlie_final := by
  sorry

end alice_has_largest_result_l375_375926


namespace ternary_to_decimal_l375_375603

theorem ternary_to_decimal (k : ℕ) (hk : k > 0) : (1 * 3^3 + k * 3^1 + 2 = 35) → k = 2 :=
by
  sorry

end ternary_to_decimal_l375_375603


namespace student_needs_to_lose_five_kg_l375_375469

def student_weight_initial := 79
def total_weight := 116

def sister_weight := total_weight - student_weight_initial
def weight_to_lose := student_weight_initial - 2 * sister_weight

theorem student_needs_to_lose_five_kg : weight_to_lose = 5 := by
  have h1: student_weight_initial = 79 := by rfl
  have h2: total_weight = 116 := by rfl
  have h3: sister_weight = 37 := by 
    unfold sister_weight
    rw [h2, h1]
    rfl
  have h4: weight_to_lose = 5 := by 
    unfold weight_to_lose
    rw [h1, h3]
    rfl
  exact h4

end student_needs_to_lose_five_kg_l375_375469


namespace coconut_grove_yield_l375_375821

theorem coconut_grove_yield (x : ℕ)
  (h1 : ∀ y, y = x + 3 → 60 * y = 60 * (x + 3))
  (h2 : ∀ z, z = x → 120 * z = 120 * x)
  (h3 : ∀ w, w = x - 3 → 180 * w = 180 * (x - 3))
  (avg_yield : 100 = 100)
  (total_trees : 3 * x = (x + 3) + x + (x - 3)) :
  60 * (x + 3) + 120 * x + 180 * (x - 3) = 300 * x →
  x = 6 :=
by
  sorry

end coconut_grove_yield_l375_375821


namespace fraction_to_decimal_l375_375070

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375070


namespace sequence_v_20_l375_375952

noncomputable def sequence_v : ℕ → ℝ → ℝ
| 0, b => b
| (n + 1), b => - (2 / (sequence_v n b + 2))

theorem sequence_v_20 (b : ℝ) (hb : 0 < b) : sequence_v 20 b = -(2 / (b + 2)) :=
by
  sorry

end sequence_v_20_l375_375952


namespace train_speed_is_correct_l375_375921

-- Define the conditions
def train_length : ℝ := 140
def platform_length : ℝ := 233.3632
def crossing_time : ℝ := 16

-- Helper definitions
def total_distance := train_length + platform_length -- Total distance covered by the train
def speed_m_per_s := total_distance / crossing_time  -- Speed of the train in m/s
def conversion_factor : ℝ := 3.6
def speed_km_per_h := speed_m_per_s * conversion_factor -- Speed of the train in km/h

-- The main theorem to prove the speed of the train
theorem train_speed_is_correct :
  speed_km_per_h = 84.00672 := 
by
  sorry

end train_speed_is_correct_l375_375921


namespace total_games_l375_375503

theorem total_games (N : ℕ) (p : ℕ)
  (hPetya : 2 ∣ N)
  (hKolya : 3 ∣ N)
  (hVasya : 5 ∣ N)
  (hGamesNotInvolving : 2 ≤ N - (N / 2 + N / 3 + N / 5)) :
  N = 30 :=
by
  sorry

end total_games_l375_375503


namespace leadership_ways_l375_375477

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

def numWaysLeadershipChosen (total_members president vp1 vp2 vp1_managers vp2_managers: ℕ) : ℕ :=
  total_members * vp1 * vp2 * combination vp1_managers 2 * combination vp2_managers 2

theorem leadership_ways : numWaysLeadershipChosen 12 11 10 8 6 = 554400 := by
  sorry

end leadership_ways_l375_375477


namespace number_of_subsets_of_M_l375_375594

open Set

/-- Definitions of sets and their intersection -/
def P : Set ℕ := {2, 3, 4, 5, 6}
def Q : Set ℕ := {3, 5, 7}
def M : Set ℕ := P ∩ Q

/-- Proof statement -/
theorem number_of_subsets_of_M : {s : Set ℕ | s ⊆ M}.toFinset.card = 4 := by
  sorry

end number_of_subsets_of_M_l375_375594


namespace expected_balls_original_positions_l375_375960

noncomputable def expected_original_positions : ℝ :=
  8 * ((3/4:ℝ)^3)

theorem expected_balls_original_positions :
  expected_original_positions = 3.375 := by
  sorry

end expected_balls_original_positions_l375_375960


namespace Reeya_average_score_l375_375296

theorem Reeya_average_score :
  let scores := [65, 67, 76, 82, 85]
  let total_score := scores.sum
  let number_of_subjects := scores.length
  total_score / number_of_subjects = 75 := by
  let scores := [65, 67, 76, 82, 85]
  have h1 : total_score = 65 + 67 + 76 + 82 + 85 := by rfl
  have h2 : total_score = 375 := by sorry
  have h3 : number_of_subjects = 5 := by rfl
  have h4 : 375 / 5 = 75 := by norm_num
  exact h4

end Reeya_average_score_l375_375296


namespace excircle_problem_l375_375259

open EuclideanGeometry

variables {A B C : Point ℝ} -- Points of the triangle
variables {O_a O_b O_c I : Point ℝ} -- Centers and incenter
variables {AB AC : Line ℝ}
variable {R : ℝ} -- Radius of the circumcircle
variables (P_1 : Point ℝ) -- Intersection point

-- Let us define the required conditions
def is_excircle_center (O : Point ℝ) (A B C : Point ℝ) : Prop := -- O is the center of the excircle of the triangle
  ∃ (r: ℝ), ∀ P: Point ℝ, distance O P = r

def is_incenter (I A B C: Point ℝ) : Prop := -- I is the actual incenter of the triangle
  ∃ rI: ℝ, ∀ P, on_incircle I A B C P = distance I P = rI

def is_circumradius (A B C O : Point ℝ) (R : ℝ) : Prop := -- R is the radius of the circumcircle
  ∃ O', circumscribed_circle A B C O' R

-- Define the problem statement in Lean
theorem excircle_problem (h1 : is_excircle_center O_a A B C) 
  (h2 : is_excircle_center O_b B C A)
  (h3 : is_excircle_center O_c C A B)
  (h4 : is_incenter I A B C)
  (h5 : is_circumradius A B C O_a R)
  (h6 : is_perpendicular P_1 O_b O_a AB)
  (h7 : is_perpendicular P_1 O_c O_a AC):
  distance P_1 I = 2 * R :=
begin
  sorry -- Proof to be implemented here
end

end excircle_problem_l375_375259


namespace number_of_zeros_l375_375563

-- Definitions based on the given conditions
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def periodic_10 (f : ℝ → ℝ) := ∀ x : ℝ, f (5 + x) = f (5 - x)
def only_zero_in_interval (f : ℝ → ℝ) := ∀ x : ℝ, (0 ≤ x ∧ x ≤ 5) → (f x = 0 → x = 1)

-- Main proof statement
theorem number_of_zeros (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_period : periodic_10 f) 
  (h_only_zero : only_zero_in_interval f):
  interval_integral_count f (-2012) 2012 = 806 :=
sorry

end number_of_zeros_l375_375563


namespace right_triangle_AC_length_l375_375243

theorem right_triangle_AC_length
  (A B C : Type)
  [has_distance A B C]
  (right_triangle_ABC : is_right_triangle A B C)
  (tan_B : tan B = 4 / 3)
  (AB_length : distance A B = 3) :
  distance A C = 5 := 
by
  sorry

end right_triangle_AC_length_l375_375243


namespace negation_of_existence_statement_l375_375754

theorem negation_of_existence_statement:
  ¬ (∃ x_0 : ℝ, (x_0^2 > real.exp x_0)) ↔ ∀ x : ℝ, (x^2 ≤ real.exp x) :=
by
  sorry

end negation_of_existence_statement_l375_375754


namespace calculate_total_tulips_l375_375478

def number_of_red_tulips_for_eyes := 8 * 2
def number_of_purple_tulips_for_eyebrows := 5 * 2
def number_of_red_tulips_for_nose := 12
def number_of_red_tulips_for_smile := 18
def number_of_yellow_tulips_for_background := 9 * number_of_red_tulips_for_smile

def total_number_of_tulips : ℕ :=
  number_of_red_tulips_for_eyes + 
  number_of_red_tulips_for_nose + 
  number_of_red_tulips_for_smile + 
  number_of_purple_tulips_for_eyebrows + 
  number_of_yellow_tulips_for_background

theorem calculate_total_tulips : total_number_of_tulips = 218 := by
  sorry

end calculate_total_tulips_l375_375478


namespace prob_each_ball_diff_color_half_l375_375504
noncomputable def prob_diff_color_equal_half_other_balls : ℚ :=
  let n := 8  -- Number of balls
  let k := 4  -- Half of (n-1)
  let P := (1/2 : ℚ)  -- Probability of each ball being either color
  let specific_prob := P^n  -- Probability of one specific arrangement (e.g., BBBBWWWW)
  let comb := Nat.choose n k  -- Number of combinations choosing k out of n
  comb * specific_prob  -- Total probability

theorem prob_each_ball_diff_color_half (h : prob_diff_color_equal_half_other_balls = 35/128) : h :=
by
  sorry

end prob_each_ball_diff_color_half_l375_375504


namespace probability_perfect_square_sum_is_7_div_36_l375_375306

-- Define the sample space for rolling two dice
def sample_space : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 1 7) (Finset.range 1 7)

-- Define the event of interest (sum is a perfect square)
def is_perfect_square_sum (x y : ℕ) : Prop :=
  let sum := x + y
  sum = 4 ∨ sum = 9

-- Calculate the number of successful outcomes
def num_successful_outcomes : ℕ := 
  Finset.card (sample_space.filter (λ (xy : ℕ × ℕ), is_perfect_square_sum xy.1 xy.2))

-- Calculate the total number of outcomes
def num_total_outcomes : ℕ := Finset.card sample_space

-- The probability is given by the ratio of successful outcomes to total outcomes
noncomputable def probability : ℚ := num_successful_outcomes / num_total_outcomes

-- Assert the probability
theorem probability_perfect_square_sum_is_7_div_36 : probability = 7 / 36 := by sorry

end probability_perfect_square_sum_is_7_div_36_l375_375306


namespace alpha_plus_3beta_is_zero_l375_375668

theorem alpha_plus_3beta_is_zero 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : 4 * cos α ^ 2 + 3 * cos β ^ 2 = 2) 
  (h4 : 4 * sin (2 * α) + 3 * sin (2 * β) = 0) : 
  α + 3 * β = 0 := 
sorry

end alpha_plus_3beta_is_zero_l375_375668


namespace mixed_fraction_product_l375_375898

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375898


namespace find_k_of_vectors_orthogonal_l375_375166

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l375_375166


namespace value_of_a_l375_375684

noncomputable def function_f (x a : ℝ) : ℝ := (x - a) ^ 2 + (Real.log x ^ 2 - 2 * a) ^ 2

theorem value_of_a (x0 : ℝ) (a : ℝ) (h1 : x0 > 0) (h2 : function_f x0 a ≤ 4 / 5) : a = 1 / 5 :=
sorry

end value_of_a_l375_375684


namespace find_a_l375_375727

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (4254253 % 53^1 - a) % 17 = 0): 
  a = 3 := 
sorry

end find_a_l375_375727


namespace no_unique_solution_for_c_l375_375993

theorem no_unique_solution_for_c (k : ℕ) (hk : k = 9) (c : ℕ) :
  (∀ x y : ℕ, 9 * x + c * y = 30 → 3 * x + 4 * y = 12) → c = 12 :=
by
  sorry

end no_unique_solution_for_c_l375_375993


namespace integral_f_l375_375536

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x < 1 then 1
  else 0

theorem integral_f : ∫ x in (-1:ℝ)..1, f x = 4/3 :=
by
  sorry

end integral_f_l375_375536


namespace average_price_per_book_l375_375297

theorem average_price_per_book (books_mexico : ℕ) (price_mexico_mxn : ℕ)
  (discount_mexico : ℚ) (exchange_rate_mxn_usd : ℚ) 
  (books_uk : ℕ) (price_uk_gbp : ℚ)
  (buy_2_get_1_free : ℚ) (vat_uk : ℚ) (exchange_rate_gbp_usd : ℚ)
  (total_price_usd : ℚ) (total_books : ℕ) (avg_price_usd : ℚ) :
  books_mexico = 65 →
  price_mexico_mxn = 7120 →
  discount_mexico = 0.10 →
  exchange_rate_mxn_usd = 20 →
  books_uk = 30 →
  price_uk_gbp = 420 →
  buy_2_get_1_free = 3 / 2 →
  vat_uk = 0.05 →
  exchange_rate_gbp_usd = 0.8 →
  total_price_usd = 871.65 →
  total_books = 95 →
  avg_price_usd = 9.175 →
  (7120 * (1 - 0.10) / 20 + 420 * 1.05 / 0.8) / (65 + 30) = avg_price_usd :=
by {
  intros,
  sorry
}

end average_price_per_book_l375_375297


namespace convert_fraction_to_decimal_l375_375055

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375055


namespace redistribute_l375_375994

-- Given conditions
variable (C : ℝ) -- Total weight of the cheese
variable (W : ℝ) -- Weight of White's slice
variable (G : ℝ) -- Weight of Gray's slice
variable (F : ℝ) -- Weight of Fat's slice
variable (T : ℝ) -- Weight of Thin's slice

-- Additional cheese redistributed
variable (extra_cheese : ℝ := 28)

-- Conditions
axiom h1 : T = F - 20
axiom h2 : W = G - 8
axiom h3 : W = C / 4
axiom h4 : G - 8 = C / 4  -- Gray cut 8 grams
axiom h5 : F - 20 = C / 4 -- Fat cut 20 grams

-- Target: each mouse should receive equal amount of cheese.
def equal_cheese (W G F T : ℝ) : Prop :=
  W = (C / 4) ∧ G = (C / 4) ∧ F + 14 = (C / 4) ∧ T + 14 = (C / 4)

theorem redistribute :
  equal_cheese W G F T :=
by
  sorry

end redistribute_l375_375994


namespace area_of_sinusoidal_figure_l375_375728

open Real

noncomputable def area_of_closed_figure (a : ℝ) (h : a = 2) : ℝ :=
  ∫ x in -a..a, sin x

theorem area_of_sinusoidal_figure :
  (∃ a : ℝ, a ∈ ℝ ∧ (real.norm ((-1)^9 * (1/a)^9 * (9.choose 9) * (2*a)^0) = 21 / 2)) →
  (area_of_closed_figure 2 rfl = 2 * cos 2 - 2) :=
by
  intro h
  sorry

end area_of_sinusoidal_figure_l375_375728


namespace transform_non_negative_sums_l375_375238

theorem transform_non_negative_sums (m n : ℕ) (table : Matrix ℤ) :
  ∃ (transform : Matrix ℤ), 
    (∀ i, 0 ≤ (Finset.sum (Finset.range n) (λ j, transform ⟨i, sorry⟩ ⟨j, sorry⟩ ))) ∧
    (∀ j, 0 ≤ (Finset.sum (Finset.range m) (λ i, transform ⟨i, sorry⟩ ⟨j, sorry⟩ ))) :=
sorry

end transform_non_negative_sums_l375_375238


namespace ticket_distribution_l375_375500

theorem ticket_distribution (tickets people : ℕ) (h_tickets : tickets = 5) (h_people : people = 4) :
  (Σ (f : fin tickets.succ → fin people.succ), (∀ (p i), f p = f (p + i) → i = 0)) = 96 :=
by
  rw [h_tickets, h_people]
  sorry

end ticket_distribution_l375_375500


namespace average_mark_is_correct_l375_375231

-- Define the maximum score in the exam
def max_score := 1100

-- Define the percentages scored by Amar, Bhavan, Chetan, and Deepak
def score_percentage_amar := 64 / 100
def score_percentage_bhavan := 36 / 100
def score_percentage_chetan := 44 / 100
def score_percentage_deepak := 52 / 100

-- Calculate the actual scores based on percentages
def score_amar := score_percentage_amar * max_score
def score_bhavan := score_percentage_bhavan * max_score
def score_chetan := score_percentage_chetan * max_score
def score_deepak := score_percentage_deepak * max_score

-- Define the total score
def total_score := score_amar + score_bhavan + score_chetan + score_deepak

-- Define the number of students
def number_of_students := 4

-- Define the average score
def average_score := total_score / number_of_students

-- The theorem to prove that the average score is 539
theorem average_mark_is_correct : average_score = 539 := by
  -- Proof skipped
  sorry

end average_mark_is_correct_l375_375231


namespace arithmetic_mean_minimizes_squared_differences_l375_375220

variable (n : ℕ) (x : Fin n → ℝ)

def arithmetic_mean (x : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, x i) / n

theorem arithmetic_mean_minimizes_squared_differences (a : ℝ) : 
  (∀ i, x i ≠ x j) → (a = arithmetic_mean x) ↔ (∀ b : ℝ, ∑ i, (x i - a) ^ 2 ≤ ∑ i, (x i - b) ^ 2) :=
by
  sorry

end arithmetic_mean_minimizes_squared_differences_l375_375220


namespace probability_shortest_diagonal_l375_375828

theorem probability_shortest_diagonal (n : ℕ) (h_n: n = 9) (h_regular: True) : 
  let D := n * (n - 3) / 2 in
  let shortest_diagonals := n in
  (shortest_diagonals : ℚ) / D = 1 / 3 :=
by
  sorry

end probability_shortest_diagonal_l375_375828


namespace cyclic_quadrilateral_XF_XG_value_l375_375291

-- Lean 4 statement
theorem cyclic_quadrilateral_XF_XG_value :
  ∀ (A B C D X Y E F G : Point) (O : Circle),
  ∃! (E : Point), ∃! (F : Point), ∃! (G : Point),
  (inscribed_quadrilateral (A B C D) O) ∧
  (length (A B) = 4) ∧ (length (B C) = 3) ∧ (length (C D) = 7) ∧ (length (D A) = 9) ∧
  (lying_on (X) (B D)) ∧ (lying_on (Y) (B D)) ∧ 
  (ratio (D X) (B D) = 1/3) ∧ (ratio (B Y) (B D) = 1/4) ∧
  (intersection (A X) (parallel_line_through (Y) (B C)) = E) ∧
  (intersection (C X) (parallel_line_through (E) (A B)) = F) ∧
  (other_intersection_of_line_circle (C X) O C = G) →
  XF * XG = 121 / 3 :=
sorry

end cyclic_quadrilateral_XF_XG_value_l375_375291


namespace graph_two_intersecting_lines_l375_375408

theorem graph_two_intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ x = 0 ∨ y = 0 :=
by
  -- Placeholder for the proof
  sorry

end graph_two_intersecting_lines_l375_375408


namespace find_dihedral_angle_l375_375934

-- Definitions of points and conditions in the 3D space
structure Cube (V : Type) [EuclideanSpace V] :=
  (A B C D A1 B1 C1 D1 : V)
  (edge_length : ℝ)

def is_midpoint {V : Type} [EuclideanSpace V] (E B C : V) : Prop :=
  dist B E = dist E C

def is_movable {V : Type} [EuclideanSpace V] (F C D : V) : Prop :=
  F ∈ line_segment ℝ C D

def is_perpendicular {V : Type} [EuclideanSpace V] (D1 E A B1 F : V) : Prop :=
  ∀u v, u ∈ plane ℝ A B1 F → v ∈ plane ℝ A B1 F → D1 -ᵥ E ⬝ u = 0

def dihedral_angle (cube : Cube ℝ) : ℝ :=
  let E := midpoint cube.B cube.C in
  let F := (segment cube.C cube.D).point_at (1/2) in
  if (is_midpoint E cube.B cube.C) ∧ (is_movable F cube.C cube.D) ∧
     (is_perpendicular cube.D1 E cube.A cube.B1 F) then
    π - arctan (2 * sqrt 2)
  else
    sorry

theorem find_dihedral_angle (cube : Cube ℝ) :
  dihedral_angle cube = π - arctan (2 * sqrt 2) :=
sorry

end find_dihedral_angle_l375_375934


namespace find_k_of_vectors_orthogonal_l375_375165

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l375_375165


namespace tan_of_diff_l375_375136

theorem tan_of_diff (θ : ℝ) (hθ : -π/2 + 2 * π < θ ∧ θ < 2 * π) 
  (h : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 :=
sorry

end tan_of_diff_l375_375136


namespace number_division_l375_375456

theorem number_division (n : ℕ) (h1 : 555 + 445 = 1000) (h2 : 555 - 445 = 110) 
  (h3 : n % 1000 = 80) (h4 : n / 1000 = 220) : n = 220080 :=
by {
  -- proof steps would go here
  sorry
}

end number_division_l375_375456


namespace convert_fraction_to_decimal_l375_375059

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375059


namespace correlation_coefficient_and_regression_model_l375_375280

noncomputable def t_i : List ℝ := [1, 2, 3, 4, 5]
noncomputable def y_i : List ℝ := [2.6, 3.1, 4.5, 6.8, 8.0]
noncomputable def sum_ti_yi : ℝ := 89.5
noncomputable def sqrt_sum_ti_diff_squared : ℝ := Real.sqrt 10
noncomputable def sqrt_sum_yi_diff_squared : ℝ := Real.sqrt 21.86

noncomputable def t_bar : ℝ := (t_i.sum / t_i.length)
noncomputable def y_bar : ℝ := (y_i.sum / y_i.length)
noncomputable def r : ℝ := 
  (sum_ti_yi - t_i.length * t_bar * y_bar) / 
    (sqrt_sum_ti_diff_squared * sqrt_sum_yi_diff_squared)

noncomputable def b_hat : ℝ := 
  (sum_ti_yi - t_i.length * t_bar * y_bar) / 
    (t_i.foldl (+) 0 (λ x, x^2) - t_i.length * t_bar^2)

noncomputable def a_hat : ℝ := y_bar - b_hat * t_bar

noncomputable def regression_eq (t : ℝ) : ℝ := b_hat * t + a_hat
noncomputable def predicted_profit_t_7 : ℝ := regression_eq 7

theorem correlation_coefficient_and_regression_model :
  r ≈ 0.98 ∧ regression_eq = λ t, 1.45 * t + 0.65 ∧ predicted_profit_t_7 = 10.8 :=
by
  sorry

end correlation_coefficient_and_regression_model_l375_375280


namespace hexagon_perimeter_l375_375627

-- Define the conditions
structure Hexagon :=
  (side_length : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (angle_D : ℝ)
  (angle_E : ℝ)
  (angle_F : ℝ)
  (area : ℝ)

-- A property stating if a hexagon is equilateral and has certain angles and area
def is_special_hexagon (hex : Hexagon) : Prop :=
  hex.angle_B = 45 ∧ hex.angle_D = 45 ∧ hex.angle_F = 45 ∧ hex.area = 12 * real.sqrt 2

-- The definition of the perimeter of the hexagon
def perimeter (hex : Hexagon) : ℝ :=
  6 * hex.side_length

-- The theorem we aim to prove
theorem hexagon_perimeter (hex : Hexagon) (h : is_special_hexagon hex) :
  perimeter hex = 24 * real.sqrt 2 := sorry

end hexagon_perimeter_l375_375627


namespace find_conjugate_z_l375_375683

noncomputable def z_conjugate (z : ℂ) : ℂ :=
  if (|z| = 5 ∧ (∃ a b : ℝ, z = a + b * I ∧ 3 * a - 4 * b = 0)) then
    if z.re = 4 ∧ z.im = 3 then
      4 - 3 * I
    else if z.re = -4 ∧ z.im = -3 then
      -4 + 3 * I
    else
      z -- default case (should not occur based on provided conditions)
  else
    z -- case when conditions are not satisfied (should not occur based on problem context)

theorem find_conjugate_z (z : ℂ) : |z| = 5 ∧ (∃ a b : ℝ, z = a + b * I ∧ 3 * a - 4 * b = 0) → (z_conjugate z = 4 - 3 * I ∨ z_conjugate z = -4 + 3 * I) :=
by
  intros h
  cases h with h_mag h_pure_im
  obtain ⟨a, b, hz, h_real_part⟩ := h_pure_im
  have h_squared : a^2 + b^2 = 25 := by
    rw [← hz] at h_mag
    simp [Complex.normSq_eq_abs, h_mag]
  have h_solutions : (a = 4 ∧ b = 3) ∨ (a = -4 ∧ b = -3) := by
    linear_combination using [h_squared, h_real_part]
  cases h_solutions with h_pos h_neg
  · left
    rw [hz]
    simp [h_pos]
  · right
    rw [hz]
    simp [h_neg]

end find_conjugate_z_l375_375683


namespace find_k_l375_375188

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l375_375188


namespace probability_close_in_heart_correct_l375_375356

def is_close_in_heart (a b : ℕ) : Prop := |a - b| ≤ 1

def total_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ :=
  let pairs := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
                (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3),
                (4, 5), (5, 4), (5, 6), (6, 5)]
  pairs.length

def probability_close_in_heart : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_close_in_heart_correct :
  probability_close_in_heart = 4 / 9 := by
  sorry

end probability_close_in_heart_correct_l375_375356


namespace min_value_expr_l375_375975

noncomputable def min_value (f : ℝ → ℝ) :=
  Inf {y : ℝ | ∃ x : ℝ, x > 0 ∧ f x = y}

theorem min_value_expr (x : ℝ) (h : x > 0) : 2 * real.sqrt x + x⁻¹ ≥ 3 :=
by
  sorry

#eval min_value (λ x, 2 * real.sqrt x + x⁻¹)

end min_value_expr_l375_375975


namespace probability_both_tell_truth_l375_375590

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_tell_truth (hA : P_A = 0.75) (hB : P_B = 0.60) : P_A * P_B = 0.45 :=
by
  rw [hA, hB]
  norm_num

end probability_both_tell_truth_l375_375590


namespace chefs_earn_less_than_manager_l375_375479

theorem chefs_earn_less_than_manager :
  let d1 := 6
  let d2 := 7
  let d3 := 8
  let w1 := d1 + 0.20 * d1
  let w2 := d2 + 0.25 * d2
  let w3 := d3 + 0.30 * d3
  let total_chefs_wage := w1 + w2 + w3
  let manager_wage := 12.50
  abs (manager_wage - total_chefs_wage) = 13.85 := 
by
  let d1 := 6
  let d2 := 7
  let d3 := 8
  let w1 := d1 + 0.20 * d1
  let w2 := d2 + 0.25 * d2
  let w3 := d3 + 0.30 * d3
  let total_chefs_wage := w1 + w2 + w3
  let manager_wage := 12.50
  sorry

end chefs_earn_less_than_manager_l375_375479


namespace part1_part2_part3_l375_375154

noncomputable def f (a c x : ℝ) : ℝ := a * x^2 - (1 / 2) * x + c
noncomputable def h (b x : ℝ) : ℝ := (3 / 4) * x^2 - b * x + b / 2 - 1 / 4
noncomputable def g (a c m x : ℝ) : ℝ := f a c x - m * x

-- Definitions of the conditions
def condition1 (a c : ℝ) := f a c 1 = 0
def condition2 (a c : ℝ) := ∀ x, f a c x ≥ 0

-- The statement for part (1): proving a and c
theorem part1 (a c : ℝ) (h1 : condition1 a c) (h2 : condition2 a c) : a = 1 / 4 ∧ c = 1 / 4 := 
sorry

-- The statement for part (2): solving the inequality
theorem part2 (b : ℝ) : ∀ x, f (1 / 4) (1 / 4) x + h b x < 0 ↔ 
  ((b < 1 / 2) ∧ (∀ x, b < x ∧ x < 1 / 2)) ∨ 
  ((b > 1 / 2) ∧ (∀ x, 1 / 2 < x ∧ x < b)) ∨ 
  ((b = 1 / 2) ∧ (∀ x, false)) := 
sorry

-- The statement for part (3): existence of m
theorem part3 (a c : ℝ) (m : ℝ) (h1 : condition1 a c) (h2 : condition2 a c) 
  (h3 : g a c m m = -5 ∧ g a c m (m + 2) = -5) :
  m = -3 ∨ m = -1 + 2 * sqrt 2 := 
sorry

end part1_part2_part3_l375_375154


namespace cone_from_sector_l375_375407

def cone_can_be_formed (θ : ℝ) (r_sector : ℝ) (r_cone_base : ℝ) (l_slant_height : ℝ) : Prop :=
  θ = 270 ∧ r_sector = 12 ∧ ∃ L, L = θ / 360 * (2 * Real.pi * r_sector) ∧ 2 * Real.pi * r_cone_base = L ∧ l_slant_height = r_sector

theorem cone_from_sector (base_radius slant_height : ℝ) :
  cone_can_be_formed 270 12 base_radius slant_height ↔ base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l375_375407


namespace gcd_possible_values_count_l375_375391

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375391


namespace circumscribed_triangle_max_side_length_l375_375423

-- Let's define K as a Δ-curve.
universe u
variable (K : Type u) [DeltaCurve K] -- Assuming DeltaCurve is a typeclass for Δ-curves

-- Let T be an equilateral triangle circumscribing K with points of tangency A, B, C.
def EquilateralTriangle (V : Type u) := {t : Triangle V // t.is_equilateral}

variable (V : Type u) [EuclideanSpaces V]
variable (T : EquilateralTriangle V)
variables (A B C : V)  -- Points where T touches K

-- Define the tangent condition
axiom tangency_condition (t : EquilateralTriangle V) (K : Type u) [DeltaCurve K] (A B C : V) : 
  touches_at A K ∧ touches_at B K ∧ touches_at C K

-- Main Statement
theorem circumscribed_triangle_max_side_length (T1 : EquilateralTriangle V) :
  side_length T1 ≤ side_length T := 
sorry

end circumscribed_triangle_max_side_length_l375_375423


namespace kims_total_points_l375_375750

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end kims_total_points_l375_375750


namespace find_k_l375_375193

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l375_375193


namespace factorial_mod_13_l375_375944

theorem factorial_mod_13 :
  10! % 13 = 6 := 
by
  sorry

end factorial_mod_13_l375_375944


namespace mixed_fractions_product_l375_375864

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375864


namespace kim_points_correct_l375_375747

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end kim_points_correct_l375_375747


namespace fraction_to_decimal_l375_375049

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375049


namespace remainder_of_polynomial_division_l375_375102

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), ∃ q r : ℂ[X], (x^{2023} + 1) = q * (x^{10} - x^8 + x^6 - x^4 + x^2 - 1) + r ∧ degree(r) < degree(x^{10} - x^8 + x^6 - x^4 + x^2 - 1) ∧ r = x^7 + 1 :=
by sorry

end remainder_of_polynomial_division_l375_375102


namespace p_is_necessary_not_sufficient_for_q_l375_375539

  variable (x : ℝ)

  def p := |x| ≤ 2
  def q := 0 ≤ x ∧ x ≤ 2

  theorem p_is_necessary_not_sufficient_for_q : (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) :=
  by
    sorry
  
end p_is_necessary_not_sufficient_for_q_l375_375539


namespace number_of_valid_digits_l375_375724

theorem number_of_valid_digits :
  (finset.card (finset.filter (λ d : ℕ, 2 + d * 0.01 + 0.0005 > 2.010)
                               (finset.range 10))) = 9 :=
by sorry

end number_of_valid_digits_l375_375724


namespace people_present_l375_375347

-- Number of parents, pupils, teachers, staff members, and volunteers
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_teachers : ℕ := 35
def num_staff_members : ℕ := 20
def num_volunteers : ℕ := 50

-- The total number of people present in the program
def total_people : ℕ := num_parents + num_pupils + num_teachers + num_staff_members + num_volunteers

-- Proof statement
theorem people_present : total_people = 908 := by
  -- Proof goes here, but adding sorry for now
  sorry

end people_present_l375_375347


namespace restore_fractions_l375_375889

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375889


namespace ellen_final_legos_count_is_correct_l375_375962

noncomputable def final_legos_count (initial_count : ℕ)
    (loss_fraction : ℚ) (gain_fraction : ℚ)
    (third_week_loss : ℕ) (final_gain_fraction : ℚ)
    (round_function : ℚ → ℕ) : ℕ :=
  let after_week_1 := initial_count - initial_count * loss_fraction.to_nat
  let after_week_2 := after_week_1 + (after_week_1 * gain_fraction.to_nat)
  let after_week_3 := after_week_2 - third_week_loss
  let final_gain := round_function (after_week_3 * final_gain_fraction)
  after_week_3 + final_gain

theorem ellen_final_legos_count_is_correct :
  final_legos_count 380 (1/5 : ℚ) (25/100 : ℚ) 57 (10/100 : ℚ) (λ x, x.to_nat) = 355 :=
by
  sorry

end ellen_final_legos_count_is_correct_l375_375962


namespace mixed_fraction_product_l375_375900

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375900


namespace john_has_enough_money_l375_375695

def first_weekend_earnings : ℝ := 20

def second_weekend_saturday_earnings : ℝ := 18
def second_weekend_sunday_earnings : ℝ := second_weekend_saturday_earnings / 2
def second_weekend_total_earnings : ℝ := second_weekend_saturday_earnings + second_weekend_sunday_earnings

def third_weekend_earnings : ℝ := second_weekend_total_earnings * 1.25

def fourth_weekend_earnings : ℝ := third_weekend_earnings * 1.15

def total_earnings : ℝ := first_weekend_earnings + second_weekend_total_earnings + third_weekend_earnings + fourth_weekend_earnings

def pogo_stick_cost : ℝ := 60

def extra_money : ℝ := total_earnings - pogo_stick_cost

theorem john_has_enough_money : extra_money = 59.5625 := by
  sorry

end john_has_enough_money_l375_375695


namespace find_roots_l375_375523

theorem find_roots : 
  (∃ x : ℝ, (x-1) * (x-2) * (x+1) * (x-5) = 0) ↔ 
  x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by sorry

end find_roots_l375_375523


namespace problem1_problem2_problem3_problem4_l375_375947

theorem problem1 : -20 - (-14) + (-18) - 13 = -37 := by
  sorry

theorem problem2 : (-3/4 + 1/6 - 5/8) / (-1/24) = 29 := by
  sorry

theorem problem3 : -3^2 + (-3)^2 + 3 * 2 + |(-4)| = 10 := by
  sorry

theorem problem4 : 16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3 := by
  sorry

end problem1_problem2_problem3_problem4_l375_375947


namespace solve_y_power_six_l375_375345

theorem solve_y_power_six :
  ∃ y : ℝ, y > 0 ∧ sin(arctan(y)) = y^3 → 
  ∃ z : ℝ, z > 0 ∧ y^2 = z ∧ z^3 + z^2 - 1 = 0 :=
by sorry

end solve_y_power_six_l375_375345


namespace least_product_two_primes_gt_10_l375_375790

open Nat

/-- Two distinct primes, each greater than 10, are multiplied. What is the least possible product of these two primes? -/
theorem least_product_two_primes_gt_10 : ∃ (p q : ℕ), p > 10 ∧ q > 10 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 143 :=
sorry

end least_product_two_primes_gt_10_l375_375790


namespace fraction_to_decimal_l375_375020

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375020


namespace equal_number_of_rectangles_with_given_perimeters_l375_375816

theorem equal_number_of_rectangles_with_given_perimeters :
  let count_rectangles (p : ℕ) := finset.card {ab : ℕ × ℕ | ab.1 + ab.2 = p / 2 ∧ ab.1 < ab.2}
  in count_rectangles 1996 = 499 ∧ count_rectangles 1998 = 499 :=
by
  sorry

end equal_number_of_rectangles_with_given_perimeters_l375_375816


namespace eighteenth_entry_l375_375986

def r_8 (n : ℕ) : ℕ := n % 8

theorem eighteenth_entry (n : ℕ) (h : r_8 (3 * n) ≤ 3) : n = 17 :=
sorry

end eighteenth_entry_l375_375986


namespace fraction_to_decimal_l375_375028

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375028


namespace magnitude_of_sum_l375_375583

def p : ℝ × ℝ := (1, 2)
def q (x : ℝ) : ℝ × ℝ := (x, 3)

theorem magnitude_of_sum (x : ℝ) (h : 1 * x + 2 * 3 = 0) :
  ‖(p.1 + q x.1, p.2 + q x.2)‖ = 5 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_sum_l375_375583


namespace distinct_intersection_points_l375_375977

theorem distinct_intersection_points :
  let S1 := { p : ℝ × ℝ | (p.1 + p.2 - 7) * (2 * p.1 - 3 * p.2 + 9) = 0 }
  let S2 := { p : ℝ × ℝ | (p.1 - p.2 - 2) * (4 * p.1 + 3 * p.2 - 18) = 0 }
  ∃! (p1 p2 p3 : ℝ × ℝ), p1 ∈ S1 ∧ p1 ∈ S2 ∧ p2 ∈ S1 ∧ p2 ∈ S2 ∧ p3 ∈ S1 ∧ p3 ∈ S2 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end distinct_intersection_points_l375_375977


namespace find_k_l375_375194

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l375_375194


namespace range_of_m_l375_375556

variables (f : ℝ → ℝ) (m : ℝ)

-- Assume f is a decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

-- Assume f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem stating the main condition and the implication
theorem range_of_m (h_decreasing : is_decreasing f) (h_odd : is_odd f) (h_condition : f (m - 1) + f (2 * m - 1) > 0) : m > 2 / 3 :=
sorry

end range_of_m_l375_375556


namespace shortest_chord_line_intersect_circle_l375_375125

-- Define the equation of the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (0, 1)

-- Define the center of the circle
def center : ℝ × ℝ := (1, 0)

-- Define the equation of the line l
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- The theorem that needs to be proven
theorem shortest_chord_line_intersect_circle :
  ∃ k : ℝ, ∀ x y : ℝ, (circle_eq x y ∧ y = k * x + 1) ↔ line_eq x y :=
by
  sorry

end shortest_chord_line_intersect_circle_l375_375125


namespace gcd_possible_values_count_l375_375399

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375399


namespace probability_other_side_yellow_l375_375431

noncomputable def yellow_side_probability : ℚ :=
let total_cards := 8
let blue_blue := 4
let blue_yellow := 2
let yellow_yellow := 2
let total_yellow_sides := 0 + 2 + 4 in
let favorable_yellow_sides := 4 in
favorable_yellow_sides / total_yellow_sides

theorem probability_other_side_yellow :
  yellow_side_probability = 2 / 3 :=
by
  sorry

end probability_other_side_yellow_l375_375431


namespace unique_solution_iff_prime_l375_375288

theorem unique_solution_iff_prime (n x y : ℕ) : 
  (∃ (x y : ℕ), (1 / x - 1 / y = 1 / n) ∧ (∀ x' y' ∈ ℕ, 1 / x' - 1 / y' = 1 / n → x = x' ∧ y = y')) ↔ Prime n :=
sorry

end unique_solution_iff_prime_l375_375288


namespace diameter_of_circumscribed_circle_l375_375617

noncomputable def right_triangle_circumcircle_diameter (a b : ℕ) : ℕ :=
  let hypotenuse := (a * a + b * b).sqrt
  if hypotenuse = max a b then hypotenuse else 2 * max a b

theorem diameter_of_circumscribed_circle
  (a b : ℕ)
  (h : a = 16 ∨ b = 16)
  (h1 : a = 12 ∨ b = 12) :
  right_triangle_circumcircle_diameter a b = 16 ∨ right_triangle_circumcircle_diameter a b = 20 :=
by
  -- The proof goes here.
  sorry

end diameter_of_circumscribed_circle_l375_375617


namespace planes_divide_space_into_parts_l375_375211

theorem planes_divide_space_into_parts :
  ∀ (P1 P2 P3 : Plane), 
  (∀ (l1 : Line), l1 ∈ (P1 ∩ P2) → ∃ (l2 : Line), l2 ∈ (P2 ∩ P3) ∧ l1 ∥ l2) →
  (∀ (l2 : Line), l2 ∈ (P2 ∩ P3) → ∃ (l3 : Line), l3 ∈ (P1 ∩ P3) ∧ l2 ∥ l3) →
  (∀ (l1 : Line), l1 ∈ (P1 ∩ P2) → ∃ (l3 : Line), l3 ∈ (P1 ∩ P3) ∧ l1 ∥ l3) →
  ∃ (n : ℕ), n = 7 :=
by
  intros P1 P2 P3 h1 h2 h3
  exists 7
  sorry

end planes_divide_space_into_parts_l375_375211


namespace problem1_problem2_l375_375637

variables {A B C : ℝ} {a b c : ℝ}

-- Definitions based on the conditions given in the problem
def triangle_condition := (2 * c^2 - 2 * a^2 = b^2)
def triangle_area (a b S : ℝ) := S = (1 / 2) * a * b

-- Statement of problem 1
theorem problem1 (A B C a b c : ℝ) (h : triangle_condition) : 
  (c * real.cos A - a * real.cos C) / b = 1 / 2 :=
sorry

-- Statement of problem 2
theorem problem2 (A : ℝ) (a b S : ℝ) (h1 : a = 1) (h2 : real.tan A = 1 / 3) :
  triangle_area a (real.sqrt (1 + real.tan A ^ 2)) S :=
sorry

end problem1_problem2_l375_375637


namespace unique_solution_of_system_l375_375514

theorem unique_solution_of_system :
  ∀ (x : Fin 2000 → ℝ),
    (∑ i, x i = 2000) →
    (∑ i, (x i)^4 = ∑ i, (x i)^3) →
    (∀ i, x i = 1) :=
by
  intro x hsum hpow
  sorry

end unique_solution_of_system_l375_375514


namespace trees_circular_path_distance_l375_375506

theorem trees_circular_path_distance 
  (num_trees : ℕ) (d_between_some_trees : ℕ) 
  (num_intervals : ℕ) 
  (tree1 : fin 8) (tree5 : fin 8) :
  num_trees = 8 →
  d_between_some_trees = 100 →
  num_intervals = 4 →
  (tree5 - tree1) = (4 : fin 8) →
  let interval_distance := d_between_some_trees / num_intervals in
  let full_circle_intervals := 8 in
  let total_distance := interval_distance * full_circle_intervals in
  total_distance = 200 :=
by
  intros h1 h2 h3 h4
  let interval_distance := d_between_some_trees / num_intervals
  let full_circle_intervals := 8
  let total_distance := interval_distance * full_circle_intervals 
  -- Proof part will use the given conditions to show total_distance = 200.
  sorry

end trees_circular_path_distance_l375_375506


namespace log_property_l375_375558

theorem log_property (x : ℝ) (h1 : x < 1) (h2 : (Real.log10 x)^3 - 3 * (Real.log10 x) = 522) : 
  (Real.log10 x)^4 - Real.log10 (x^4) = 6597 :=
sorry

end log_property_l375_375558


namespace find_m_l375_375630

def vector_sub (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem find_m (m : ℝ) (A B : ℝ × ℝ × ℝ) (hA : A = (-1, 2, m)) (hB : B = (3, -2, 2)) (hDist : vector_norm (vector_sub A B) = 4 * real.sqrt 2) : m = 2 :=
by
  sorry

end find_m_l375_375630


namespace fraction_to_decimal_l375_375035

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375035


namespace sin_alpha_minus_beta_value_l375_375134

noncomputable def sin_alpha_minus_beta (α β : ℝ) : ℝ :=
  sin (α - β)

-- Given conditions
variable (α β : ℝ) 
variable (h_sin_alpha : sin α = 2 * sqrt 3 / 3)
variable (h_cos_alpha_plus_beta : cos (α + β) = -1 / 3)
variable (h_α_range : 0 < α ∧ α < π / 2)
variable (h_β_range : 0 < β ∧ β < π / 2)

theorem sin_alpha_minus_beta_value :
  sin_alpha_minus_beta α β = 10 * sqrt 2 / 27 :=
by
  sorry

end sin_alpha_minus_beta_value_l375_375134


namespace min_value_C_l375_375930

open Real

noncomputable def y_A (x : ℝ) : ℝ := x / 2 + 2 / x
noncomputable def y_B (x : ℝ) : ℝ := log x + 1 / log x
noncomputable def y_C (x : ℝ) : ℝ := 3^x + 3^(-x)
noncomputable def y_D (x : ℝ) : ℝ := sin x + 1 / sin x

theorem min_value_C : (∃ x : ℝ, y_C x = 2) ∧ (∀ x, y_A x ≠ 2) ∧ (∀ x (h : 1 < x ∧ x < 10), y_B x ≠ 2) ∧ (∀ x (h : 0 < x ∧ x < π / 2), y_D x ≠ 2) :=
by {
  sorry
}

end min_value_C_l375_375930


namespace restore_fractions_l375_375888

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375888


namespace inequality_proof_l375_375122

theorem inequality_proof (a b t : ℝ) (h₀ : 0 < t) (h₁ : t < 1) (h₂ : a * b > 0) : 
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 :=
by
  sorry

end inequality_proof_l375_375122


namespace independent_set_cardinality_bound_l375_375258

open Finset

-- Define the graph structure
structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (symm : ∀ {x y : V}, adj x y → adj y x)
  (loopless : ∀ {x : V}, ¬ adj x x)

-- Define the degree of a vertex
noncomputable def degree {V : Type} (G : Graph V) (v : V) : ℕ := 
  (univ.filter (G.adj v)).card

-- Define the main theorem
theorem independent_set_cardinality_bound {V : Type} (G : Graph V) 
  (vertices : Finset V) (degrees : ∀ v ∈ vertices, ℕ) :
  ∃ I : Finset V, (∀ v w ∈ I, ¬ G.adj v w) ∧ I.card ≥ ∑ v in vertices, 1 / (degrees v + 1) := 
sorry

end independent_set_cardinality_bound_l375_375258


namespace find_k_l375_375173

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l375_375173


namespace equilateral_triangle_perimeter_with_inscribed_circle_l375_375230

theorem equilateral_triangle_perimeter_with_inscribed_circle (r : ℝ) (r_eq_two : r = 2) (a : ℝ) (h : r = a / (2 * real.sqrt 3)) : 
  3 * a = 12 * real.sqrt 3 :=
by
  -- Definitions based on given conditions
  have radius_eq := r_eq_two
  have side_eq := h
  
  -- Sorry statement to skip the proof
  sorry

end equilateral_triangle_perimeter_with_inscribed_circle_l375_375230


namespace sum_of_fractions_l375_375269

theorem sum_of_fractions (a b c : ℕ) (hc1 : c ∣ a + b) (hc2 : b ∣ a + c) (hc3 : a ∣ b + c) :
  let n1 := (a + b) / c,
      n2 := (a + c) / b,
      n3 := (b + c) / a in
  n1 + n2 + n3 ∈ {6, 7, 8} := 
by {
  sorry
}

end sum_of_fractions_l375_375269


namespace total_population_l375_375338

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end total_population_l375_375338


namespace fraction_to_decimal_l375_375043

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375043


namespace lily_pads_cover_entire_lake_l375_375224

/-- 
If a patch of lily pads doubles in size every day and takes 57 days to cover half the lake,
then it will take 58 days to cover the entire lake.
-/
theorem lily_pads_cover_entire_lake (days_to_half : ℕ) (h : days_to_half = 57) : (days_to_half + 1 = 58) := by
  sorry

end lily_pads_cover_entire_lake_l375_375224


namespace sum_pn_equals_target_l375_375110

-- Given p_n is the probability that all n people place their drink in a cup holder.
noncomputable def p : ℕ → ℝ 
| 0     := 1
| 1     := 1
| n + 2 := (1 / (2 * (n + 2))) * p (n + 1) + (1 / (2 * (n + 2))) * (Finset.sum (Finset.range (n + 1)) (λ k, p k * p (n - k)))

-- Define the generating function P(x)
noncomputable def P (x : ℝ) : ℝ := (Real.exp (x / 2)) / (2 - Real.exp (x / 2))

theorem sum_pn_equals_target : (∑' n : ℕ, p (n + 1)) = (2 * Real.sqrt Real.exp 1 - 2) / (2 - Real.sqrt Real.exp 1) :=
by
  sorry

end sum_pn_equals_target_l375_375110


namespace fraction_to_decimal_l375_375045

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375045


namespace order_of_numbers_l375_375535

def base_to_decimal (n : Nat) (base : Nat) : Nat :=
  let digits := n.digits base
  List.foldl (λ (sum : Nat) (d : Nat) => sum * base + d) 0 digits

theorem order_of_numbers :
  let a := base_to_decimal 12 16
  let b := base_to_decimal 25 7
  let c := base_to_decimal 33 4
  c < a ∧ a < b :=
by
  let a := base_to_decimal 12 16
  let b := base_to_decimal 25 7
  let c := base_to_decimal 33 4
  have h1 : a = 18 := by rfl
  have h2 : b = 19 := by rfl
  have h3 : c = 15 := by rfl
  sorry

end order_of_numbers_l375_375535


namespace not_sufficient_nor_necessary_l375_375671

theorem not_sufficient_nor_necessary (a b : ℝ) (hb : b ≠ 0) :
  ¬ ((a > b) ↔ (1 / a < 1 / b)) :=
by
  sorry

end not_sufficient_nor_necessary_l375_375671


namespace fraction_to_decimal_l375_375040

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375040


namespace fraction_meaningful_iff_nonzero_l375_375778

theorem fraction_meaningful_iff_nonzero (x : ℝ) : (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 :=
by sorry

end fraction_meaningful_iff_nonzero_l375_375778


namespace complex_magnitude_squared_l375_375682

open Complex Real

theorem complex_magnitude_squared :
  ∃ (z : ℂ), z + abs z = 3 + 7 * i ∧ abs z ^ 2 = 841 / 9 :=
by
  sorry

end complex_magnitude_squared_l375_375682


namespace gcd_values_count_l375_375406

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375406


namespace second_player_wins_with_optimal_play_l375_375832

theorem second_player_wins_with_optimal_play :
  ∀ (game_sequence : List Nat),
  (∀ n, (n ∈ game_sequence → (n = 1 ∨ (∃ m, m ∈ game_sequence ∧ (n = 2 * m ∨ n = m + 1)))) ∧
         list.Nodup game_sequence ∧
         1000 ∈ game_sequence) →
  ((list.head game_sequence = some 1) →
  ((list.index_of 1000 game_sequence) % 2 = 1)) :=
by
  sorry

end second_player_wins_with_optimal_play_l375_375832


namespace intersection_M_N_l375_375158

-- Define set M and N
def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5 * x + 6 > 0}

-- Problem statement to show their intersection
theorem intersection_M_N :
  M ∩ N = {x | x < 1} := 
sorry

end intersection_M_N_l375_375158


namespace sum_first_50_arithmetic_sequence_l375_375303

theorem sum_first_50_arithmetic_sequence : 
  let a : ℕ := 2
  let d : ℕ := 4
  let n : ℕ := 50
  let a_n (n : ℕ) : ℕ := a + (n - 1) * d
  let S_n (n : ℕ) : ℕ := n / 2 * (2 * a + (n - 1) * d)
  S_n n = 5000 :=
by
  sorry

end sum_first_50_arithmetic_sequence_l375_375303


namespace gcd_possible_values_count_l375_375372

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375372


namespace min_distance_sum_l375_375593

theorem min_distance_sum (x : ℝ) : 
  ∃ y, y = |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ∧ y = 45 / 8 :=
sorry

end min_distance_sum_l375_375593


namespace line_integral_eq_pi_R_sq_l375_375519

variable (R : ℝ) (t : ℝ)
def x (t : ℝ) := R * Real.cos t
def y (t : ℝ) := R * Real.sin t
def z (t : ℝ) := t / (2 * Real.pi)

noncomputable def line_integral :=
  ∫ t in 0..(2 * Real.pi), (z t) * (-R * Real.sin t) + (x t) * (R * Real.cos t) + (y t) * (1 / (2 * Real.pi))

theorem line_integral_eq_pi_R_sq : line_integral = Real.pi * R^2 :=
  sorry

end line_integral_eq_pi_R_sq_l375_375519


namespace solve_system_of_equations_l375_375720

theorem solve_system_of_equations : 
  ∀ x y : ℝ, 
    (2 * x^2 - 3 * x * y + y^2 = 3) ∧ 
    (x^2 + 2 * x * y - 2 * y^2 = 6) 
    ↔ (x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by
  sorry

end solve_system_of_equations_l375_375720


namespace hollow_circles_in_2001_triangle_number_difference_l375_375508

-- Conditions for problem (1)
def circlePattern (n : ℕ) : Bool :=
  match n % 9 with
  | 0 | 3 | 5 => false  -- solid circle
  | 1 | 4 | 8 => true   -- hollow circle
  | _ => false         -- rest are arbitrary to complete pattern

-- Theorem for problem (1)
theorem hollow_circles_in_2001 : ∑ i in Finset.range 2001, if circlePattern i then 1 else 0 = 667 :=
by sorry

-- Definition of triangle number
def triangleNumber (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem for problem (2)
theorem triangle_number_difference : triangleNumber 24 - triangleNumber 22 = 47 :=
by sorry

end hollow_circles_in_2001_triangle_number_difference_l375_375508


namespace cyclic_quadrilateral_count_l375_375111

def is_cyclic (q : Type) : Prop := 
  ∃ O : Point, ∀ V : Vertex, dist O V = r

def square : Type := sorry
def rectangle_not_square : Type := sorry
def rhombus_not_square : Type := sorry
def parallelogram_not_rectangle_or_rhombus : Type := sorry
def isosceles_trapezoid_not_parallelogram : Type:= sorry

theorem cyclic_quadrilateral_count : 
  ( {square, rectangle_not_square, isosceles_trapezoid_not_parallelogram}.count is_cyclic) = 3 :=
by sorry

end cyclic_quadrilateral_count_l375_375111


namespace problem_statement_l375_375120

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

variable (x : ℝ)

theorem problem_statement
  (h : f_iter 13 x = f_iter 31 x) :
  f_iter 16 x = (x - 1) / x :=
by
  sorry

end problem_statement_l375_375120


namespace fraction_to_decimal_l375_375044

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375044


namespace xiaoming_probability_l375_375229

-- Define the binomial probability function
def prob_exactly_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem xiaoming_probability :
  prob_exactly_k_successes 6 2 (1 / 3) = 240 / 729 :=
by sorry

end xiaoming_probability_l375_375229


namespace pie_eating_contest_l375_375614

theorem pie_eating_contest
  (consumed_first : ℚ)
  (consumed_second : ℚ)
  (h1 : consumed_first = 7 / 8)
  (h2 : consumed_second = 5 / 6) : 
  consumed_first - consumed_second = 1 / 24 :=
by {
  rw [h1, h2],
  norm_num
}

end pie_eating_contest_l375_375614


namespace C2_rect_eq_max_distance_C2_to_C1_l375_375576

-- Definitions and conditions
def C1_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi / 3) = -1

def C2_polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

-- Proof statement for part (1)
theorem C2_rect_eq :
  (∀ (ρ θ : ℝ), C2_polar_eq ρ θ → ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2) := by
  sorry

-- Proof statement for part (2)
theorem max_distance_C2_to_C1 :
  let C2_rect := λ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2
      C1_line := λ (x y : ℝ), x + Real.sqrt 3 * y + 2 = 0
  in ∀ (x y : ℝ) (h : C2_rect x y), 
      ∃ d : ℝ, d = (3 + Real.sqrt 3) / 2 + Real.sqrt 2 := by
  sorry

end C2_rect_eq_max_distance_C2_to_C1_l375_375576


namespace least_possible_product_of_two_distinct_primes_gt_10_l375_375788

-- Define a predicate to check if a number is a prime greater than 10
def is_prime_gt_10 (n : ℕ) : Prop := 
  nat.prime n ∧ n > 10

-- Lean theorem statement
theorem least_possible_product_of_two_distinct_primes_gt_10 :
  ∃ p q : ℕ, is_prime_gt_10 p ∧ is_prime_gt_10 q ∧ p ≠ q ∧ p * q = 143 :=
begin
  -- existence proof omitted
  sorry
end

end least_possible_product_of_two_distinct_primes_gt_10_l375_375788


namespace digits_concatenated_l375_375721

theorem digits_concatenated (x : ℕ) (y : ℕ) (hx : x = 2^2020) (hy : y = 5^2020) :
    (nat.digits 10 x).length + (nat.digits 10 y).length = 2021 := by
  sorry

end digits_concatenated_l375_375721


namespace value_of_v_star_star_l375_375107

noncomputable def v_star (v : ℝ) : ℝ :=
  v - v / 3
  
theorem value_of_v_star_star (v : ℝ) (h : v = 8.999999999999998) : v_star (v_star v) = 4.000000000000000 := by
  sorry

end value_of_v_star_star_l375_375107


namespace gcd_possible_values_count_l375_375387

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375387


namespace find_k_l375_375185

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l375_375185


namespace fraction_to_decimal_l375_375031

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375031


namespace determine_values_l375_375766

-- Define the main problem conditions
def A := 1.2
def B := 12

-- The theorem statement capturing the problem conditions and the solution
theorem determine_values (A B : ℝ) (h1 : A + B = 13.2) (h2 : B = 10 * A) : A = 1.2 ∧ B = 12 :=
  sorry

end determine_values_l375_375766


namespace restore_original_problem_l375_375877

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375877


namespace area_of_quadrilateral_PQRS_l375_375292

theorem area_of_quadrilateral_PQRS (P Q R S T: EuclideanSpace ℝ 2)
  (hPQ: dist P Q = 24)
  (hQS: dist Q S = 18)
  (hPT: dist P T = 6)
  (hQPS: ∠ Q P S = 90)
  (hQRS: ∠ Q R S = 90):
  area_of_quadrilateral P Q R S = 378 :=
by
  sorry

end area_of_quadrilateral_PQRS_l375_375292


namespace no_common_root_l375_375831

variables {R : Type*} [OrderedRing R]

def f (x m n : R) := x^2 + m*x + n
def p (x k l : R) := x^2 + k*x + l

theorem no_common_root (k m n l : R) (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬ ∃ x : R, (f x m n = 0 ∧ p x k l = 0) :=
by
  sorry

end no_common_root_l375_375831


namespace who_finished_5th_l375_375215

-- Defining the positions
variables {Sam Tony Chris Ana Ben Kim : ℕ}
axiom h1 : Kim = 7
axiom h2 : ∃ k Chris, Kim = Chris + 2
axiom h3 : ∃ k Ben, Chris = Ben + 1
axiom h4 : ∃ k Ana, Ben = Ana + 2
axiom h5 : ∃ k Tony, Ana = Tony + 3
axiom h6 : ∃ k Sam, Sam = Tony + 1

-- The main theorem to prove who finished in 5th place
theorem who_finished_5th : Chris = 5 :=
by sorry

end who_finished_5th_l375_375215


namespace derivative_of_y_l375_375829

noncomputable def ch (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
noncomputable def sh (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

theorem derivative_of_y (x : ℝ) :
  ∀ x, deriv (λ x, (1 + 8 * (ch x)^2 * Real.log (ch x)) / (2 * (ch x)^2)) x =
  (sh x * (4 * (ch x)^2 - 1)) / (ch x)^3 :=
by
  sorry

end derivative_of_y_l375_375829


namespace find_line_eq_l375_375160

structure Vector (α : Type*) := 
(x : α) 
(y : α)

def a : Vector ℝ := ⟨6, 2⟩
def b : Vector ℝ := ⟨-4, 0.5⟩

def pointA := (3, -1 : ℝ)

def dir_vector : Vector ℝ :=
{ x := a.x + 2 * b.x,
  y := a.y + 2 * b.y }

def is_perpendicular (v1 v2 : Vector ℝ) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

theorem find_line_eq : 
(is_perpendicular dir_vector ⟨1, k⟩) → 
(∃ (A B C : ℝ), A * 3 + B * (-1) = -C ∧ A * 3 + B * (-1) - C = 0) :=
by
  sorry

end find_line_eq_l375_375160


namespace restore_original_problem_l375_375873

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375873


namespace tiles_needed_to_cover_floor_l375_375847

/-- 
A floor 10 feet by 15 feet is to be tiled with 3-inch-by-9-inch tiles. 
This theorem verifies that the necessary number of tiles is 800. 
-/
theorem tiles_needed_to_cover_floor
  (floor_length : ℝ)
  (floor_width : ℝ)
  (tile_length_inch : ℝ)
  (tile_width_inch : ℝ)
  (conversion_factor : ℝ)
  (num_tiles : ℕ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 15)
  (h_tile_length_inch : tile_length_inch = 3)
  (h_tile_width_inch : tile_width_inch = 9)
  (h_conversion_factor : conversion_factor = 12)
  (h_num_tiles : num_tiles = 800) :
  (floor_length * floor_width) / ((tile_length_inch / conversion_factor) * (tile_width_inch / conversion_factor)) = num_tiles :=
by
  -- The proof is not included, using sorry to mark this part
  sorry

end tiles_needed_to_cover_floor_l375_375847


namespace pentagon_side_CD_eq_radius_l375_375419

theorem pentagon_side_CD_eq_radius (A B C D E : Type) (R : ℝ)
  (circle : ∀ P : Type, P ∈ {A, B, C, D, E} → dist P (center circle) = R)
  (angle_B : ∠B = 110°) (angle_E : ∠E = 100°) :
  dist C D = R :=
sorry

end pentagon_side_CD_eq_radius_l375_375419


namespace pages_per_day_l375_375660

def notebooks : Nat := 5
def pages_per_notebook : Nat := 40
def total_days : Nat := 50

theorem pages_per_day (H1 : notebooks = 5) (H2 : pages_per_notebook = 40) (H3 : total_days = 50) : 
  (notebooks * pages_per_notebook / total_days) = 4 := by
  sorry

end pages_per_day_l375_375660


namespace fraction_to_decimal_l375_375029

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375029


namespace Arun_crossing_time_l375_375795

def trainA_length : ℝ := 200  -- length of train A in meters
def trainA_speed_kmh : ℝ := 54  -- speed of train A in km/hr

def trainB_length : ℝ := 150  -- length of train B in meters
def trainB_speed_kmh : ℝ := 36  -- speed of train B in km/hr

def trainC_length : ℝ := 180  -- length of train C in meters
def trainC_speed_kmh : ℝ := 45  -- speed of train C in km/hr

def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 1) / (3600 / 1)  -- convert from km/hr to m/s

def trainA_speed : ℝ := speed_kmh_to_ms trainA_speed_kmh
def trainB_speed : ℝ := speed_kmh_to_ms trainB_speed_kmh
def trainC_speed : ℝ := speed_kmh_to_ms trainC_speed_kmh

def relative_speed_A_B : ℝ := trainA_speed + trainB_speed
def relative_speed_A_C : ℝ := trainA_speed - trainC_speed

def distance_to_cross_B : ℝ := trainA_length + trainB_length
def distance_to_cross_C : ℝ := trainA_length + trainC_length

def time_to_cross_B : ℝ := distance_to_cross_B / relative_speed_A_B
def time_to_cross_C : ℝ := distance_to_cross_C / relative_speed_A_C

def total_time_to_cross : ℝ := time_to_cross_B + time_to_cross_C

theorem Arun_crossing_time :
  total_time_to_cross = 166 := 
sorry

end Arun_crossing_time_l375_375795


namespace equal_good_cells_iff_odd_l375_375736

def good_cell (n : ℕ) (board : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  1 ≤ board i j ∧ board i j ≤ n ∧ board i j > j

def board_condition (n : ℕ) (board : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j1 j2, j1 ≠ j2 → board i j1 ≠ board i j2) ∧
  (∀ j i1 i2, i1 ≠ i2 → board i1 j ≠ board i2 j)

def row_contains_equal_good_cells (n : ℕ) (board : ℕ → ℕ → ℕ) : Prop :=
  ∀ i1 i2, 
    (finset.card (finset.filter (λ j, good_cell n board i1 j) (finset.range n + 1))) =
    (finset.card (finset.filter (λ j, good_cell n board i2 j) (finset.range n + 1)))

theorem equal_good_cells_iff_odd (n : ℕ) :
  (∃ (board : ℕ → ℕ → ℕ), board_condition n board ∧ row_contains_equal_good_cells n board) ↔ Odd n := 
sorry

end equal_good_cells_iff_odd_l375_375736


namespace convert_fraction_to_decimal_l375_375053

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375053


namespace geometric_seq_sum_odd_positions_eq_91_l375_375449

theorem geometric_seq_sum_odd_positions_eq_91
  (a : ℕ → ℕ)
  (hq : ∃ q : ℕ, a 1 = 1 ∧ a 2 = 1 * q ∧ a 3 = 1 * q^2 ∧ a 4 = 1 * q^3 ∧ a 5 = 1 * q^4)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 121)
  (h_pos_int : ∀ n, a n < 100) :
  let S := a 1 + a 3 + a 5 in
  S = 91 := 
by
  sorry

end geometric_seq_sum_odd_positions_eq_91_l375_375449


namespace multiplication_72515_9999_l375_375983

theorem multiplication_72515_9999 : 72515 * 9999 = 725077485 :=
by
  sorry

end multiplication_72515_9999_l375_375983


namespace tangential_quadrilateral_condition_incircle_radii_relation_l375_375351

-- Step d: Problem a in Lean
theorem tangential_quadrilateral_condition 
    (A B C D P Q : Type)
    (is_convex_quadrilateral : ∀ (X : Type), X = A ∨ X = B ∨ X = C ∨ X = D)
    (are_tangential_A_B_C: Prop)
    (radius_incircle : A → ℝ) :
    (are_tangential_A_B_C → ∃ (D : Type), D := is_convex_quadrilateral D) :=
begin
  sorry,
end

-- Step d: Problem b in Lean
theorem incircle_radii_relation
    (r_a r_b r_c r_d : ℝ)
    (cond: ∀ (X : ℝ), X = r_a ∨ X = r_b ∨ X = r_c ∨ X = r_d):
    (\frac{1}{r_a} + \frac{1}{r_c} = \frac{1}{r_b} + \frac{1}{r_d}) :=
begin
  sorry,
end

end tangential_quadrilateral_condition_incircle_radii_relation_l375_375351


namespace find_circle_radius_l375_375443

-- Definition of the problem conditions
variables {A B C O F E G : Type}
variable {a R γ : ℝ}
variable {triangle_ABC_isosceles : ∀ (A B C : Type), Prop}
variable {circle_tangent_F : ∀ (A C F : Type), Prop}
variable {circle_intersects_G : ∀ (B C G : Type), Prop}
variable {circle_center_O : ∀ (A B O : Type), Prop}
variable {circle_intersects_E : ∀ (A B E: Type), Prop}
variable {AE_eq_a : a = a}
variable {angle_BFG_eq_gamma : γ = γ}

-- The theorem statement
theorem find_circle_radius (A B C O F E G : Type) (a R γ : ℝ)
  (triangle_ABC_isosceles : ∀ (A B C : Type), Prop)
  (circle_tangent_F : ∀ (A C F : Type), Prop)
  (circle_intersects_G : ∀ (B C G : Type), Prop)
  (circle_center_O : ∀ (A B O : Type), Prop)
  (circle_intersects_E : ∀ (A B E: Type), Prop)
  (AE_eq_a : a = a)
  (angle_BFG_eq_gamma : γ = γ) :
  R = (a * real.sin (π / 4 + γ / 2)) / (1 - real.sin (π / 4 + γ / 2)) :=
  sorry

end find_circle_radius_l375_375443


namespace smallest_four_digit_equiv_to_3_mod_8_l375_375802

theorem smallest_four_digit_equiv_to_3_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 3 → m ≥ n) := 
begin
  use 1003,
  split,
  { 
    -- Prove that 1003 is a four-digit number
    linarith,
  },
  split,
  { 
    -- Prove that 1003 is less than 10000
    linarith,
  },
  split,
  {
    -- Prove that 1003 ≡ 3 (mod 8)
    exact nat.mod_eq_of_lt (show 3 < 8, by linarith),
  },
  {
    -- Prove that 1003 is the smallest such number
    intros m h1 h2 h3,
    have h_mod : m % 8 = 3 := h3,
    have trivial_ineq : 1003 ≤ m := sorry,
    exact trivial_ineq,
  },
end

end smallest_four_digit_equiv_to_3_mod_8_l375_375802


namespace sum_of_digits_greatest_divisor_l375_375680

def digits_sum (n: ℕ) : ℕ :=
  (n.toString.foldl (λ acc c, acc + (c.toNat - '0'.toNat)) 0)

theorem sum_of_digits_greatest_divisor :
  let d := Nat.gcd (1305 - 4665) (Nat.gcd (6905 - 4665) (6905 - 1305))
  digits_sum d = 4 :=
by
  let d := Nat.gcd (4665 - 1305) (Nat.gcd (6905 - 4665) (6905 - 1305))
  have h1 : digits_sum d = 4, from sorry
  exact h1

end sum_of_digits_greatest_divisor_l375_375680


namespace fraction_equals_decimal_l375_375079

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375079


namespace evaluate_fraction_subtraction_l375_375006

theorem evaluate_fraction_subtraction :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = (11 / 30) :=
by
  sorry

end evaluate_fraction_subtraction_l375_375006


namespace oyster_crab_ratio_l375_375769

theorem oyster_crab_ratio
  (O1 C1 : ℕ)
  (h1 : O1 = 50)
  (h2 : C1 = 72)
  (h3 : ∃ C2 : ℕ, C2 = (2 * C1) / 3)
  (h4 : ∃ O2 : ℕ, O1 + C1 + O2 + C2 = 195) :
  ∃ ratio : ℚ, ratio = O2 / O1 ∧ ratio = (1 : ℚ) / 2 := 
by 
  sorry

end oyster_crab_ratio_l375_375769


namespace dream_recall_impossibility_l375_375363

-- Conditions
def officer_returned_from_china : Prop := True
def officer_fell_asleep_in_church : Prop := True
def officer_dreamed_during_sleep : Prop := True
def wife_touched_officer_with_fan : Prop := True
def officer_died_immediately : Prop := True

-- Theorem to be proved
theorem dream_recall_impossibility
  (h1 : officer_returned_from_china)
  (h2 : officer_fell_asleep_in_church)
  (h3 : officer_dreamed_during_sleep)
  (h4 : wife_touched_officer_with_fan)
  (h5 : officer_died_immediately) :
  ¬ (∃ dream, officer_dreamed_during_sleep ∧ officer_recalled_dream) :=
sorry

end dream_recall_impossibility_l375_375363


namespace prime_factor_of_sum_of_four_consecutive_odd_numbers_is_two_l375_375346

theorem prime_factor_of_sum_of_four_consecutive_odd_numbers_is_two {n : ℤ} (hn : Odd n) :
    ∃ p : ℕ, Prime p ∧ ∀ a b c d : ℤ, (a = n - 3) → (b = n - 1) → (c = n + 1) → (d = n + 3) → p ∣ (a + b + c + d) ∧ p = 2 :=
by
  sorry

end prime_factor_of_sum_of_four_consecutive_odd_numbers_is_two_l375_375346


namespace pencils_ordered_l375_375859

theorem pencils_ordered (pencils_per_student : ℕ) (number_of_students : ℕ) (total_pencils : ℕ) :
  pencils_per_student = 3 →
  number_of_students = 65 →
  total_pencils = pencils_per_student * number_of_students →
  total_pencils = 195 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_ordered_l375_375859


namespace y_work_duration_l375_375417

theorem y_work_duration (x_rate y_rate : ℝ) (d : ℝ) :
  -- 1. x and y together can do the work in 20 days.
  (x_rate + y_rate = 1/20) →
  -- 2. x started the work alone and after 4 days y joined him till the work completed.
  -- 3. The total work lasted 10 days.
  (4 * x_rate + 6 * (x_rate + y_rate) = 1) →
  -- Prove: y can do the work alone in 12 days.
  y_rate = 1/12 :=
by {
  sorry
}

end y_work_duration_l375_375417


namespace find_k_l375_375170

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l375_375170


namespace triangle_angle_sum_cannot_exist_l375_375815

theorem triangle_angle_sum (A : Real) (B : Real) (C : Real) :
    A + B + C = 180 :=
sorry

theorem cannot_exist (right_two_60 : ¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) 
    (scalene_100 : ∃ A B C : Real, A = 100 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A + B + C = 180)
    (isosceles_two_70 : ∃ A B C : Real, A = B ∧ A = 70 ∧ C = 180 - 2 * A ∧ A + B + C = 180)
    (equilateral_60 : ∃ A B C : Real, A = 60 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180)
    (one_90_two_50 : ¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :
  (¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) ∧
  (¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :=
by
  sorry

end triangle_angle_sum_cannot_exist_l375_375815


namespace wholesale_prices_correct_max_profit_correct_l375_375959

namespace ZongziProblem

noncomputable def wholesale_prices : Prop :=
  ∃ (x y : ℕ), (x - y = 10) ∧ (x + 2 * y = 100) ∧ (x = 40 ∧ y = 30)

noncomputable def max_profit : Prop :=
  ∃ (a : ℕ), let w := (a - 40) * (100 - 2 * (a - 50)) in
  (w = 1800 ∧ a = 70)

theorem wholesale_prices_correct : wholesale_prices :=
  sorry

theorem max_profit_correct : max_profit :=
  sorry

end ZongziProblem

end wholesale_prices_correct_max_profit_correct_l375_375959


namespace sum_f_evaluations_l375_375679

def f (x : ℝ) : ℝ :=
  if x > 3 then x^2 - 4
  else if x < -3 then -2
  else 3*x + 1

theorem sum_f_evaluations : f (-4) + f 0 + f 4 = 11 := by
  sorry

end sum_f_evaluations_l375_375679


namespace largest_is_D_l375_375410

open Real

def A := 17231 + 1 / 3251
def B := 17231 - 1 / 3251
def C := 17231 * (1 / 3251)
def D := 17231 / (1 / 3251)
def E := 17231.3251

theorem largest_is_D : D > A ∧ D > B ∧ D > C ∧ D > E := by
  sorry

end largest_is_D_l375_375410


namespace simplify_fraction_l375_375276

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (b^(-2) * a^(-2)) / (b^(-4) - a^(-4)) = (b^2 * a^2) / (a^4 - b^4) :=
by
  sorry

end simplify_fraction_l375_375276


namespace triangle_incenter_BI_eq_fifteen_l375_375355

theorem triangle_incenter_BI_eq_fifteen
  (A B C I : Type)
  (AB AC BC : ℝ)
  (hAB : AB = 27)
  (hAC : AC = 26)
  (hBC : BC = 25)
  (hI : internal_angle_bisector_intersection I A B C) :
  BI = 15 :=
  sorry

end triangle_incenter_BI_eq_fifteen_l375_375355


namespace fraction_to_decimal_l375_375047

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375047


namespace skilled_in_exactly_two_areas_l375_375225

theorem skilled_in_exactly_two_areas (n x' y' z : ℕ) 
  (h1 : n = 150) 
  (h2 : x' = 75) 
  (h3 : y' = 90) 
  (h4 : z' = 45)
  (h5 : ∀ s (hs : s ∈ {painter, writer, musician}), ¬(painter s ∧ writer s ∧ musician s)) :
  ∃ k, k = 90 := 
by
  let x := n - x'
  let y := n - y'
  let z := n - z'
  let total := x + y + z
  let overlap := total - n
  have : k = overlap := by
    sorry
  use k
  exact this

end skilled_in_exactly_two_areas_l375_375225


namespace profit_without_discount_is_31_25_percent_l375_375466

-- Define the conditions used in the problem
def CP : ℝ := 100
def discount_percent : ℝ := 0.04
def profit_with_discount_percent : ℝ := 0.26

-- Define the selling price with the discount
def SP_with_discount : ℝ := CP * (1 + profit_with_discount_percent)

-- Define the marked price (MP) before discount
def MP_before_discount : ℝ := SP_with_discount / (1 - discount_percent)

-- Define the selling price without the discount (which is MP)
def SP_without_discount : ℝ := MP_before_discount

-- Define the expected profit percentage without discount
def expected_profit_percent_without_discount : ℝ := (SP_without_discount - CP) / CP * 100

theorem profit_without_discount_is_31_25_percent :
  expected_profit_percent_without_discount = 31.25 := 
sorry

end profit_without_discount_is_31_25_percent_l375_375466


namespace mixed_fraction_product_example_l375_375880

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375880


namespace probability_of_points_l375_375775

noncomputable def point_probability : ℝ :=
  ∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in x..min (4 * x) 1, ∫ (z : ℝ) in 0..y, 1

theorem probability_of_points : point_probability = 3 / 4 :=
sorry

end probability_of_points_l375_375775


namespace value_of_f_f_neg1_l375_375151

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 1 - 2^x else real.sqrt x

theorem value_of_f_f_neg1 : f(f(-1)) = real.sqrt 2 / 2 :=
by
  sorry

end value_of_f_f_neg1_l375_375151


namespace relationship_among_abc_l375_375117

def a : ℝ := Real.tan (-7 * Real.pi / 6)
def b : ℝ := Real.cos (23 * Real.pi / 4)
def c : ℝ := Real.sin (-33 * Real.pi / 4)

theorem relationship_among_abc : (b > a) ∧ (a > c) :=
by
  sorry

end relationship_among_abc_l375_375117


namespace cone_lateral_surface_area_ratio_l375_375209

/-- Let a be the side length of the equilateral triangle front view of a cone.
    The base area of the cone is (π * (a / 2)^2).
    The lateral surface area of the cone is (π * (a / 2) * a).
    We want to show that the ratio of the lateral surface area to the base area is 2.
 -/
theorem cone_lateral_surface_area_ratio 
  (a : ℝ) 
  (base_area : ℝ := π * (a / 2)^2) 
  (lateral_surface_area : ℝ := π * (a / 2) * a) 
  : lateral_surface_area / base_area = 2 :=
by
  sorry

end cone_lateral_surface_area_ratio_l375_375209


namespace find_k_l375_375183

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l375_375183


namespace tracy_initial_balloons_l375_375941

theorem tracy_initial_balloons (T : ℕ) : 
  (12 + 8 + (T + 24) / 2 = 35) → T = 6 :=
by
  sorry

end tracy_initial_balloons_l375_375941


namespace scaling_transformation_curve_l375_375239

theorem scaling_transformation_curve : 
  ∀ (x y x' y' : ℝ), 
  x' = 2 * x ∧ y' = 3 * y ∧ y = 3 * sin (2 * x) → y' = 9 * sin x' :=
by
  intros x y x' y' hx hy hcurve
  sorry

end scaling_transformation_curve_l375_375239


namespace restore_fractions_l375_375893

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375893


namespace tangent_lines_through_P_l375_375571

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem tangent_lines_through_P :
  (∃ k b, (∀ x: ℝ, f x = x^3 - 3 * x ) →
           (∀ (x₀ : ℝ),
              (f x₀ = x₀^3 - 3 * x₀) →
              (P : ℝ × ℝ) →
              P = (2, -6) →
              (y - (f x₀) = k * (x - x₀) ∧ y = k*x + b)) ∧
  ∀ y : ℝ, y = 3x + y = 0 ∨ y = 24x - y - 54 = 0) := 
  sorry

end tangent_lines_through_P_l375_375571


namespace stock_investment_amount_l375_375746

theorem stock_investment_amount :
  ∀ (I : ℝ) (MV : ℝ) (NR : ℝ) (BR : ℝ), 
    I = 756 →
    MV = 96.97222222222223 → 
    NR = 10.5 → 
    BR = 0.0025 → 
    let AMV := MV * (1 - BR) in
    let FV := (I * 100) / NR in
    let AI := (FV / 100) * AMV in
    AI = 6964 :=
by
  intros I MV NR BR I_eq MV_eq NR_eq BR_eq
  let AMV := MV * (1 - BR)
  let FV := (I * 100) / NR
  let AI := (FV / 100) * AMV
  sorry

end stock_investment_amount_l375_375746


namespace milburg_population_l375_375336

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l375_375336


namespace symmetry_center_and_sum_l375_375545

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * x^2 + 3 * x - (5 / 12)

theorem symmetry_center_and_sum :
  let x₀ : ℝ := 1 / 2 in
  let y₀ : ℝ := f x₀ in
  (f x₀ = 1) ∧ 
  ∑ k in (finset.range 2012), f ((k + 1) / 2013.0) = 2012 :=
by
  sorry

end symmetry_center_and_sum_l375_375545


namespace shaded_area_l375_375626

/-- Proof problem to find the area of the shaded region within the rectangle but outside the semicircles -/
theorem shaded_area (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) :
  let r3 := (r1 + r2) / 2
  let area_rect := (r1 + r2) * (2 * r2)
  let area_ADB := (1/2) * real.pi * r1 ^ 2
  let area_BEC := (1/2) * real.pi * r2 ^ 2
  let area_DFE := (1/2) * real.pi * r3 ^ 2
  let area_semicircles := area_ADB + area_BEC + area_DFE
  area_rect - area_semicircles = 30 - 14.625 * real.pi :=
by
  -- Begin by defining each area
  let r3 := (r1 + r2) / 2
  let area_rect := (r1 + r2) * (2 * r2)
  let area_ADB := (1/2) * real.pi * r1 ^ 2
  let area_BEC := (1/2) * real.pi * r2 ^ 2
  let area_DFE := (1/2) * real.pi * r3 ^ 2
  let area_semicircles := area_ADB + area_BEC + area_DFE
  -- Show
  have h3 : r3 = 2.5 := by linarith
  have h_area_rect : area_rect = 30 := by 
    rw [h1, h2]
    norm_num
  have h_area_ADB : area_ADB = 2 * real.pi := by 
    rw [h1]
    norm_num [real.pi]
  have h_area_BEC : area_BEC = 4.5 * real.pi := by 
    rw [h2]
    norm_num [real.pi]
  have h_area_DFE : area_DFE = 3.125 * real.pi := by 
    rw [h3]
    norm_num [real.pi]
  have h_area_semicircles : area_semicircles = 14.625 * real.pi := by 
    rw [h_area_ADB, h_area_BEC, h_area_DFE]
    norm_num [real.pi]
  have h_diff : area_rect - area_semicircles = 30 - 14.625 * real.pi := by 
    rw [h_area_rect, h_area_semicircles]
    norm_num [real.pi]
  exact h_diff

end shaded_area_l375_375626


namespace coefficient_of_term_degree_of_term_l375_375735

-- Definitions based on conditions
def term := - (4 / 3) * (a ^ 2) * b

-- Theorem statements
theorem coefficient_of_term : coefficient term = - (4 / 3) :=
sorry

theorem degree_of_term : degree term = 3 :=
sorry

end coefficient_of_term_degree_of_term_l375_375735


namespace find_k_of_vectors_orthogonal_l375_375163

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l375_375163


namespace fraction_to_decimal_l375_375048

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375048


namespace find_radius_l375_375843

def radius_of_circle (r : ℝ) : Prop :=
  let x := π * r^2
  let y := 2 * π * r
  x + y = 100 * π

theorem find_radius : radius_of_circle 9 :=
by
  unfold radius_of_circle
  sorry

end find_radius_l375_375843


namespace number_of_integers_with_gcd_24_4_l375_375530

theorem number_of_integers_with_gcd_24_4 : 
  (Finset.filter (λ n, Int.gcd 24 n = 4) (Finset.range 201)).card = 17 := by
  sorry

end number_of_integers_with_gcd_24_4_l375_375530


namespace sum_of_row_in_pascals_triangle_sum_of_row_10_in_pascals_triangle_sum_of_row_11_in_pascals_triangle_l375_375214

/-- In Pascal's Triangle, each number is the sum of the number above it to the left and the number above it to the right. -/
theorem sum_of_row_in_pascals_triangle (n : ℕ) : ∑ i in Finset.range (n + 1), Nat.choose n i = 2 ^ n := sorry

theorem sum_of_row_10_in_pascals_triangle : ∑ i in Finset.range (10 + 1), Nat.choose 10 i = 1024 :=
by
  have h := sum_of_row_in_pascals_triangle 10
  rw [pow_succ, pow_succ, pow_two] at h
  exact h

theorem sum_of_row_11_in_pascals_triangle : ∑ i in Finset.range (11 + 1), Nat.choose 11 i = 2048 :=
by
  have h := sum_of_row_in_pascals_triangle 11
  rw [pow_succ, pow_succ, pow_two] at h
  exact h

end sum_of_row_in_pascals_triangle_sum_of_row_10_in_pascals_triangle_sum_of_row_11_in_pascals_triangle_l375_375214


namespace find_k_l375_375179

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l375_375179


namespace fraction_to_decimal_l375_375068

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375068


namespace fraction_to_decimal_l375_375084

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375084


namespace gcd_values_count_l375_375402

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375402


namespace gcd_possible_values_count_l375_375374

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375374


namespace dice_sum_probability_l375_375606

theorem dice_sum_probability : 
  let dice_faces := finset.range 1 7 in 
  let sum_17 (a b c : ℕ) := a + b + c = 17 in 
  let is_valid_dice (n : ℕ) := n ∈ dice_faces in 
  (finset.card (finset.filter (λ (t : ℕ × ℕ × ℕ), sum_17 t.1 t.2 t.3 ∧ is_valid_dice t.1 ∧ is_valid_dice t.2 ∧ is_valid_dice t.3) (finset.product (finset.product dice_faces dice_faces) dice_faces)) : ℚ) = 1/72 :=
sorry

end dice_sum_probability_l375_375606


namespace ratio_XR_XU_l375_375242

theorem ratio_XR_XU (X Y Z P Q U R : Point) 
    (h1 : collinear X Y P)
    (h2 : collinear X Z Q)
    (h3 : is_angle_bisector X U Y Z)
    (h4 : intersects X U P Q R)
    (h5 : dist X P = 2)
    (h6 : dist P Y = 6)
    (h7 : dist X Q = 3)
    (h8 : dist Q Z = 3) :
  dist X R / dist X U = 1 / 3 := by
  sorry

end ratio_XR_XU_l375_375242


namespace find_k_l375_375175

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l375_375175


namespace prove_log_l375_375130

noncomputable def f : ℝ → ℝ := sorry -- You would define the function based on the given conditions.
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f (x)

theorem prove_log (h1 : is_odd_function f)
                  (h2 : is_periodic_function f 3)
                  (h3 : ∀ x, 0 < x ∧ x ≤ 1 → f(x) = 2^x - 1) :
  f (Real.logb (1/2) 9) = -1/8 :=
sorry

end prove_log_l375_375130


namespace identity_eq_a_minus_b_l375_375000

theorem identity_eq_a_minus_b (a b : ℚ) (x : ℚ) (h : ∀ x, x > 0 → 
  (a / (2^x - 2) + b / (2^x + 3) = (5 * 2^x + 4) / ((2^x - 2) * (2^x + 3)))) : 
  a - b = 3 / 5 := 
by 
  sorry

end identity_eq_a_minus_b_l375_375000


namespace range_of_a_l375_375157

variable (a : ℝ)

def p : Prop := ∃ (x₀ : ℝ), x₀ ∈ Icc (-1 : ℝ) 1 ∧ (x₀^2 + x₀ - a + 1 > 0)

def q : Prop := ∀ (t : ℝ), t ∈ Ioo 0 1 → (t^2 - (2*a + 2)*t + a^2 + 2*a + 1 > 1) ∧
  ∀ (x y : ℝ), x^2 + (y^2 / (t^2 - (2*a + 2)*t + a^2 + 2*a + 1)) = 1

theorem range_of_a : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2 : ℝ) 1 ∪ Set.Ici 3 :=
by
  sorry

end range_of_a_l375_375157


namespace solution_set_inequality_l375_375981

theorem solution_set_inequality : {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_inequality_l375_375981


namespace fraction_to_decimal_equiv_l375_375012

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375012


namespace fraction_to_decimal_l375_375082

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375082


namespace mixed_fraction_product_example_l375_375884

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375884


namespace find_a_for_parallel_lines_l375_375210

theorem find_a_for_parallel_lines 
  (a : ℝ)
  (l1 : ∀ x y, a * x + 2 * y + 6 = 0)
  (l2 : ∀ x y, x + (a - 1) * y + (a^2 - 1) = 0)
  (parallel : ∀ x y, l1 x y → l2 x y → -a / 2 = -1 / (a - 1)) :
  a = -1 :=
by
  sorry

end find_a_for_parallel_lines_l375_375210


namespace smallest_four_digit_integer_mod_8_eq_3_l375_375808

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end smallest_four_digit_integer_mod_8_eq_3_l375_375808


namespace fraction_equals_decimal_l375_375071

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375071


namespace BE_eq_CD_l375_375625

theorem BE_eq_CD (A B C O H D E : Type) [incidence_geometry A B C] [acute_triangle A B C (c AB AC)] 
  (circumcenter : is_circumcenter O A B C) (orthocenter : is_orthocenter H A B C)
  (AD_inter_BC : line_intersects_segment (line_through A O) (segment B C) D)
  (HE_parallel_AD : parallel (line_through H E) (line_through A D)) 
  : distance B E = distance C D := 
sorry

end BE_eq_CD_l375_375625


namespace tug_of_war_competition_l375_375613

theorem tug_of_war_competition (n : ℕ) (class : finset (fin n)) 
  (teams : finset (finset (fin n))) 
  (h1 : ∀ T ∈ teams, T ≠ ∅ ∧ T ≠ class) 
  (h2 : ∀ T ∈ teams, ∃ T' ∈ teams, T' = class \ T) : 
  ∀ T ∈ teams, ∃ T' ∈ teams, T' = class \ T := by
  sorry

end tug_of_war_competition_l375_375613


namespace largest_among_a_b_c_d_l375_375204

noncomputable def a : ℝ := Real.log 2022 / Real.log 2021
noncomputable def b : ℝ := Real.log 2023 / Real.log 2022
noncomputable def c : ℝ := 2022 / 2021
noncomputable def d : ℝ := 2023 / 2022

theorem largest_among_a_b_c_d : max a (max b (max c d)) = c := 
sorry

end largest_among_a_b_c_d_l375_375204


namespace milburg_population_l375_375331

/-- Number of grown-ups in Milburg --/
def grownUps : ℕ := 5256

/-- Number of children in Milburg --/
def children : ℕ := 2987

/-- Total number of people in Milburg --/
def totalPeople : ℕ := grownUps + children

theorem milburg_population : totalPeople = 8243 := by
  have h1 : grownUps = 5256 := rfl
  have h2 : children = 2987 := rfl
  have h3 : totalPeople = grownUps + children := rfl
  have h4 : grownUps + children = 8243 := by
    calc
      5256 + 2987 = 8243 := by sorry -- Proof step to be filled in
  exact h4

end milburg_population_l375_375331


namespace divisors_of_expression_l375_375302

theorem divisors_of_expression (a b : ℤ) (h : 4 * b = 10 - 3 * a) : 
  fintype.card ({d ∈ finset.range 10 | d > 0 ∧ d ∣ (3 * b + 18)} : set ℕ) = 3 := 
sorry

end divisors_of_expression_l375_375302


namespace people_sharing_pizzas_l375_375360

-- Definitions based on conditions
def number_of_pizzas : ℝ := 21.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

-- Theorem to prove the number of people
theorem people_sharing_pizzas : (number_of_pizzas * slices_per_pizza) / slices_per_person = 64 :=
by
  sorry

end people_sharing_pizzas_l375_375360


namespace sandy_has_32_fish_l375_375709

-- Define the initial number of pet fish Sandy has
def initial_fish : Nat := 26

-- Define the number of fish Sandy bought
def fish_bought : Nat := 6

-- Define the total number of pet fish Sandy has now
def total_fish : Nat := initial_fish + fish_bought

-- Prove that Sandy now has 32 pet fish
theorem sandy_has_32_fish : total_fish = 32 :=
by
  sorry

end sandy_has_32_fish_l375_375709


namespace find_positive_x_l375_375582

def vector_perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_positive_x (x : ℝ) (h_perpendicular : vector_perpendicular (x, 3) (x + 2, -1)) : x = 1 := 
begin
  sorry
end

end find_positive_x_l375_375582


namespace distinct_x_intercepts_l375_375198

-- Given conditions
def polynomial (x : ℝ) : ℝ := (x - 4) * (x^2 + 4 * x + 13)

-- Statement of the problem as a Lean theorem
theorem distinct_x_intercepts : 
  (∃ (x : ℝ), polynomial x = 0 ∧ 
    ∀ (y : ℝ), y ≠ x → polynomial y = 0 → False) :=
  sorry

end distinct_x_intercepts_l375_375198


namespace select_five_circles_l375_375127

theorem select_five_circles (P : Point) (circles : List Circle) (h_circles : ∀ c ∈ circles, P ∈ c) (h_length : circles.length = 1980) :
  ∃ (selected : List Circle), selected.length = 5 ∧ ∀ c ∈ circles, ∃ s ∈ selected, ∀ x, x ∈ c.center → x ∈ s.boundary ∨ x ∈ interior s := 
sorry

end select_five_circles_l375_375127


namespace interval_equivalence_l375_375090

theorem interval_equivalence :
  {x : ℝ | x < 0 ∨ x ≥ 1} = set.Iio 0 ∪ set.Ici 1 :=
begin
  sorry
end

end interval_equivalence_l375_375090


namespace length_of_AB_l375_375608

variables (A B C P : Type) [normed_add_comm_group A]
(vA vB vC : A)
(h_incircle: ∥vA - vB∥ = 1 ∧ ∥vA - vC∥ = 1 ∧ ∥vB - vC∥ = 1)
(h_vector_eq : 3 • vA + 4 • vB + 5 • vC = 0)

theorem length_of_AB : ∥ vA - vB ∥ = real.sqrt 2 :=
sorry

end length_of_AB_l375_375608


namespace smallest_four_digit_equiv_to_3_mod_8_l375_375804

theorem smallest_four_digit_equiv_to_3_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 3 → m ≥ n) := 
begin
  use 1003,
  split,
  { 
    -- Prove that 1003 is a four-digit number
    linarith,
  },
  split,
  { 
    -- Prove that 1003 is less than 10000
    linarith,
  },
  split,
  {
    -- Prove that 1003 ≡ 3 (mod 8)
    exact nat.mod_eq_of_lt (show 3 < 8, by linarith),
  },
  {
    -- Prove that 1003 is the smallest such number
    intros m h1 h2 h3,
    have h_mod : m % 8 = 3 := h3,
    have trivial_ineq : 1003 ≤ m := sorry,
    exact trivial_ineq,
  },
end

end smallest_four_digit_equiv_to_3_mod_8_l375_375804


namespace mixed_fraction_product_example_l375_375878

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375878


namespace smallest_four_digit_equiv_mod_8_l375_375807

theorem smallest_four_digit_equiv_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 :=
by
  -- We state the assumptions and final goal
  use 1003
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  · refl
  sorry

end smallest_four_digit_equiv_mod_8_l375_375807


namespace find_k_of_vectors_orthogonal_l375_375167

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l375_375167


namespace company_team_sizes_l375_375846

open Nat

theorem company_team_sizes (factors_in_range : List Nat) :
  factors_in_range.length = 4 :=
by
  sorry

def factors (n : Nat) : List Nat :=
  (List.range (n+1)).filter (λ k => k > 0 ∧ n % k = 0)

def valid_sizes (n : Nat) (low high : Nat) : List Nat :=
  (factors n).filter (λ k => low ≤ k ∧ k ≤ high)

noncomputable def company_team_sizes :
  valid_sizes 120 8 25 = [8, 15, 20, 24] :=
by
  sorry

end company_team_sizes_l375_375846


namespace exists_two_vertices_singular_degree_leq_four_l375_375949

structure Polyhedron :=
  (vertices : Type)
  (edges : vertices → vertices → Prop)
  (is_convex : Prop)

def isRedOrYellow (e : Prop) : Prop :=
  e = true ∨ e = false

def singular_angle (v1 v2 : Prop) : Prop :=
  (v1 = true ∧ v2 = false) ∨ (v1 = false ∧ v2 = true)

def singular_degree (P : Polyhedron) (A : P.vertices) : ℕ :=
  ∑ (B C : P.vertices), if singular_angle (P.edges A B) (P.edges A C) then 1 else 0

theorem exists_two_vertices_singular_degree_leq_four :
  ∀ (P : Polyhedron),
  P.is_convex →
  (∃ (B C : P.vertices), singular_degree P B + singular_degree P C ≤ 4) :=
by
  sorry

end exists_two_vertices_singular_degree_leq_four_l375_375949


namespace convert_fraction_to_decimal_l375_375060

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375060


namespace eve_walked_farther_l375_375968

def running_intervals : List ℝ := [0.75, 2/3, 0.95, 3/4, 7/8]
def walking_intervals : List ℝ := [1/2, 0.65, 5/6, 3/5, 0.8, 3/4]

def total_distance (intervals : List ℝ) : ℝ := intervals.sum

def total_running_distance : ℝ := total_distance running_intervals
def total_walking_distance : ℝ := total_distance walking_intervals

theorem eve_walked_farther :
  total_walking_distance - total_running_distance = 0.1416 :=
sorry

end eve_walked_farther_l375_375968


namespace restore_original_problem_l375_375875

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375875


namespace geometric_sequence_20_sum_is_2_pow_20_sub_1_l375_375546

def geometric_sequence_sum_condition (a : ℕ → ℕ) (q : ℕ) : Prop :=
  (a 1 * q + 2 * a 1 = 4) ∧ (a 1 ^ 2 * q ^ 4 = a 1 * q ^ 4)

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) : ℕ :=
  (a 1 * (1 - q ^ 20)) / (1 - q)

theorem geometric_sequence_20_sum_is_2_pow_20_sub_1 (a : ℕ → ℕ) (q : ℕ) 
  (h : geometric_sequence_sum_condition a q) : 
  geometric_sequence_sum a q =  2 ^ 20 - 1 := 
sorry

end geometric_sequence_20_sum_is_2_pow_20_sub_1_l375_375546


namespace triangle_side_length_l375_375641

-- Definitions for the conditions given in the problem
def angleB : ℝ := 45
def AB : ℝ := 100
def AC : ℝ := 100 * Real.sqrt 2

-- Statement of the theorem/problem
theorem triangle_side_length (ABC : Type) [triangle ABC] 
  (angleB : ∠B = 45)
  (ABeq : AB = 100)
  (ACeq : AC = 100 * Real.sqrt 2) :
  BC ≈ Real.sqrt (30000 + 5160 * Real.sqrt 2) :=
sorry

end triangle_side_length_l375_375641


namespace gcd_possible_values_count_l375_375396

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375396


namespace chance_of_picking_pepperjack_l375_375651

-- Defining the conditions of the problem
def cheddarSticks : ℕ := 15
def mozzarellaSticks : ℕ := 30
def pepperjackSticks : ℕ := 45

-- Defining the problem as a theorem in Lean
theorem chance_of_picking_pepperjack :
  let totalSticks := cheddarSticks + mozzarellaSticks + pepperjackSticks in
  (pepperjackSticks : ℚ) / totalSticks * 100 = 50 := by
  sorry

end chance_of_picking_pepperjack_l375_375651


namespace mixed_fraction_product_l375_375897

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375897


namespace geometric_sequence_l375_375634

noncomputable def S (a : ℕ → ℝ) : ℕ → ℝ
| 0       => 0
| (n + 1) => a (n + 1) + S a n

theorem geometric_sequence (a : ℕ → ℝ) (S_n_eq : ∀ {n}, S a n = ∑ i in Finset.range n, a (i + 1))
  (geo_seq : ∀ n, 2 ≤ n → ∃ r, a n = r * (S a n) ∧ (S a n) = r * (S a (n - 1)) ∧ (S a (n + 1)) = r * (2 * S a n)):
  (∀ n, a 1 = 2) ∧ (∀ n, 2 ≤ n → a n = - (2 / (n^2 - n))) :=
by
  sorry

end geometric_sequence_l375_375634


namespace new_integer_increases_average_l375_375415

theorem new_integer_increases_average (n : ℤ)
  (h : ∀ S : Finset ℤ, S = {8, 11, 12, 14, 15} → 
    let new_avg := (S.sum + n) / (S.card + 1) in 
    new_avg = 3 / 2 * (S.sum / S.card)) :
  n = 48 :=
sorry

end new_integer_increases_average_l375_375415


namespace student_not_in_front_l375_375342

theorem student_not_in_front (students : Finset ℕ) (spec_stud : ℕ) 
  (h1 : spec_stud ∈ students) (h2 : students.card = 5) :
  (∃ (arrangements : Finset (Finset ℕ)), arrangements.card = 96 
    ∧ (∀ a ∈ arrangements, ∃ perm : List ℕ, perm.length = 5 
      ∧ (spec_stud ∈ perm.tail ∧ spec_stud = perm.head))) :=
sorry

end student_not_in_front_l375_375342


namespace cost_per_candy_bar_is_correct_l375_375299

-- Define the conditions given in the problem
def first_day_sales : ℕ := 10
def increase_per_day : ℕ := 4
def days_per_week : ℕ := 6
def total_earnings : ℝ := 12.0

-- Use these conditions to define the number of candy bars sold each day
def daily_sales (day : ℕ) : ℕ :=
  if day = 1 then first_day_sales
  else daily_sales (day - 1) + increase_per_day

-- Calculate the total number of candy bars sold in the week
def total_candy_bars_sold : ℕ :=
  (List.range days_per_week).sum (λ day, daily_sales (day + 1))

-- Define the cost per candy bar
def cost_per_candy_bar : ℝ :=
  total_earnings / total_candy_bars_sold

-- The final theorem
theorem cost_per_candy_bar_is_correct : cost_per_candy_bar = 0.10 :=
by
  sorry

end cost_per_candy_bar_is_correct_l375_375299


namespace gcd_possible_values_count_l375_375388

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375388


namespace round_to_nearest_hundredth_l375_375706

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 3.447) : Real.round_nearest_hundreth x = 3.45 :=
by
  sorry

end round_to_nearest_hundredth_l375_375706


namespace angle_between_line_and_plane_l375_375516

-- Define the conditions including line and plane equations
def line_eq1 (x z : ℝ) : Prop := x - 2 * z + 3 = 0
def line_eq2 (y z : ℝ) : Prop := y + 3 * z - 1 = 0
def plane_eq (x y z : ℝ) : Prop := 2 * x - y + z + 3 = 0

-- Define the direction vector of the line
def line_direction_vector : ℝ × ℝ × ℝ := (2, -3, 1)

-- Define the normal vector to the plane
def plane_normal_vector : ℝ × ℝ × ℝ := (2, -1, 1)

-- Prove that the angle between the given line and plane is approximately 60° 49'
theorem angle_between_line_and_plane : 
  ∃ (ϕ : ℝ), 
  ϕ ≈ 60.8166667 ∧ 
  let m := (2: ℝ), n := (-3: ℝ), p := (1: ℝ) in
  let A := (2: ℝ), B := (-1: ℝ), C := (1: ℝ) in
  sin ϕ = (abs (A * m + B * n + C * p)) / (sqrt (A^2 + B^2 + C^2) * sqrt (m^2 + n^2 + p^2)) :=
by 
  sorry

end angle_between_line_and_plane_l375_375516


namespace largest_power_of_two_in_e_p_l375_375364

noncomputable def p : ℝ := ∑ k in finset.range 9, k * real.log k

theorem largest_power_of_two_in_e_p :
  ∃ k : ℕ, 2^k ∣ real.exp p ∧ ∀ l : ℕ, 2^l ∣ real.exp p → l ≤ 40 := 
sorry

end largest_power_of_two_in_e_p_l375_375364


namespace option_A_sufficient_not_necessary_l375_375997

variable (a b : ℝ)

def A : Set ℝ := { x | x^2 - x + a ≤ 0 }
def B : Set ℝ := { x | x^2 - x + b ≤ 0 }

theorem option_A_sufficient_not_necessary : (A = B → a = b) ∧ (a = b → A = B) :=
by
  sorry

end option_A_sufficient_not_necessary_l375_375997


namespace fraction_problem_l375_375437

theorem fraction_problem (x : ℝ) (h₁ : x * 180 = 18) (h₂ : x < 0.15) : x = 1/10 :=
by sorry

end fraction_problem_l375_375437


namespace tan_alpha_minus_pi_over_4_l375_375205

theorem tan_alpha_minus_pi_over_4
  (α : ℝ)
  (h : tan (α - π / 12) = sin (13 * π / 3)) :
  tan (α - π / 4) = sqrt 3 / 9 :=
by
  sorry

end tan_alpha_minus_pi_over_4_l375_375205


namespace smallest_four_digit_equiv_to_3_mod_8_l375_375803

theorem smallest_four_digit_equiv_to_3_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 3 → m ≥ n) := 
begin
  use 1003,
  split,
  { 
    -- Prove that 1003 is a four-digit number
    linarith,
  },
  split,
  { 
    -- Prove that 1003 is less than 10000
    linarith,
  },
  split,
  {
    -- Prove that 1003 ≡ 3 (mod 8)
    exact nat.mod_eq_of_lt (show 3 < 8, by linarith),
  },
  {
    -- Prove that 1003 is the smallest such number
    intros m h1 h2 h3,
    have h_mod : m % 8 = 3 := h3,
    have trivial_ineq : 1003 ≤ m := sorry,
    exact trivial_ineq,
  },
end

end smallest_four_digit_equiv_to_3_mod_8_l375_375803


namespace hyperbola_asymptotes_example_l375_375575

noncomputable def hyperbola_asymptotes (a b : ℝ) (h : a > 0 ∧ b > 0) : 
    ∀ (x y : ℝ), (x, y) ∈ {(x, y) | x^2 / a^2 - y^2 / b^2 = 1} ↔ y = b/a * x ∨ y = -b/a * x := sorry

theorem hyperbola_asymptotes_example :
    hyperbola_asymptotes 1 (sqrt 3) (by norm_num [sqrt_pos.mpr zero_lt_three]) :=
sorry

end hyperbola_asymptotes_example_l375_375575


namespace minimum_value_correct_l375_375121

noncomputable def minimum_value (n k : ℕ) (F : ℂ[X]) (h1 : F.degree = n) (h2 : F.leading_coeff = 1) : ℂ :=
  |F.eval 0|^2 + |F.eval 1|^2 + ... + |F.eval (n + k)|^2

theorem minimum_value_correct (n : ℕ) (k : ℕ) (F : ℂ[X]) (h1 : F.degree = n) (h2 : F.leading_coeff = 1) :
  minimum_value n k F h1 h2 = (n!^2 * (finset.choose (2*n + 1 + k) k)) / (finset.choose 2*n n) :=
sorry

end minimum_value_correct_l375_375121


namespace fraction_to_decimal_l375_375062

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375062


namespace no_infinite_pairs_l375_375141

def seq_a (a₁ : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 1 then a₁
  else Nat.gcd (2^(n - 1) - 1 + a (n-1)) (2^(n - 1) - 1 + a (n-1))

theorem no_infinite_pairs (a₁ : ℕ) (h₁ : a₁ > 0) (r : ℕ) (hr : r > 1) :
    ¬∃ (m n : ℕ), (∀ k : ℕ, k > 0 → 
    seq_a a₁ (seq_a a₁) m > seq_a a₁ (seq_a a₁) n ∧ seq_a a₁ (seq_a a₁) m = r * seq_a a₁ (seq_a a₁) n) :=
  sorry

end no_infinite_pairs_l375_375141


namespace smallest_positive_period_l375_375951

variable {R : Type} [LinearOrderedField R]
variable (f : R → R)

def functional_equation := ∀ x : R, f(x + 5) + f(x - 5) = f(x)

theorem smallest_positive_period (h : functional_equation f) :
  ∃ p > 0, (∀ x : R, f(x + p) = f(x)) ∧ (∀ q > 0, (∀ x : R, f(x + q) = f(x)) → q ≥ p) ∧ p = 30 :=
by
  sorry

end smallest_positive_period_l375_375951


namespace find_k_l375_375180

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l375_375180


namespace sequence_an_formula_sequence_an_product_l375_375329

theorem sequence_an_formula (a : ℕ → ℚ) :
  a 0 = 1 ∧
  (∀ n, a (n+1) = 1 - ((n+1) / ((n+2):ℚ))^2 * a n) →
  (∀ n, a n = (n + 2) / (2 * (n + 1))) :=
by
  sorry

theorem sequence_an_product (a : ℕ → ℚ) :
  a 0 = 1 ∧
  (∀ n, a (n+1) = 1 - ((n+1) / ((n+2):ℚ))^2 * a n) →
  (∀ n, (∏ i in Finset.range (n + 1), a i) = (n + 2) / (2 ^ (n + 1))) :=
by
  sorry

end sequence_an_formula_sequence_an_product_l375_375329


namespace proof_problem_l375_375573
noncomputable section

def f (x : ℝ) := (3 * x + 4) / (x + 3)
def S := {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ f x = y}
def M := 3
def m := 4 / 3

theorem proof_problem :
  m ∈ S ∧ M ∉ S :=
by
  sorry

end proof_problem_l375_375573


namespace yard_length_l375_375222

theorem yard_length (num_trees : ℕ) (dist_between_trees : ℕ) (num_trees = 26) (dist_between_trees = 20) : ℕ :=
  (num_trees - 1) * dist_between_trees

example : yard_length 26 20 = 500 := by
  unfold yard_length
  simp
  sorry

end yard_length_l375_375222


namespace local_minimum_f_when_k2_at_x1_l375_375555

def f (x : ℝ) (k : ℕ) :=
  (Real.exp x - 1) * (x - 1)^k

theorem local_minimum_f_when_k2_at_x1 :
  ∃ ε > 0, ∀ x, |x - 1| < ε → (f x 2 ≥ f 1 2) :=
by
  sorry

end local_minimum_f_when_k2_at_x1_l375_375555


namespace sandy_walks_30_meters_in_each_of_first_three_legs_l375_375710

def leg_distance (x : ℕ) : Prop :=
  let first_leg := x
  let second_leg := x
  let third_leg := x
  let fourth_leg := 10
  30 = first_leg + third_leg - second_leg

theorem sandy_walks_30_meters_in_each_of_first_three_legs :
  ∃ x : ℕ, leg_distance x ∧ x = 30 :=
begin
  sorry
end

end sandy_walks_30_meters_in_each_of_first_three_legs_l375_375710


namespace triangular_pyramid_surface_area_l375_375471

theorem triangular_pyramid_surface_area
  (base_area : ℝ)
  (side_area : ℝ) :
  base_area = 3 ∧ side_area = 6 → base_area + 3 * side_area = 21 :=
by
  sorry

end triangular_pyramid_surface_area_l375_375471


namespace expression_varies_l375_375966

variable (x : ℝ)

def expression (x : ℝ) : ℝ :=
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (x^2 + 4 * x + 4) / ((x - 3) * (x + 2))

theorem expression_varies (hx3 : x ≠ 3) (hx_neg2 : x ≠ -2) : ∃ x : ℝ, expression x ≠ expression (x + 1) :=
sorry

end expression_varies_l375_375966


namespace x_y_solution_l375_375964

variable (x y : ℕ)

noncomputable def x_wang_speed : ℕ := x - 6

theorem x_y_solution (hx : (5 : ℚ) / 6 * x = y) (hy : (2 : ℚ) / 3 * (x - 6) = y - 10) : x = 36 ∧ y = 30 :=
by {
  sorry
}

end x_y_solution_l375_375964


namespace volume_bounded_by_surfaces_l375_375946

noncomputable def volume_enclosed : ℝ := 
  6 * Real.pi

theorem volume_bounded_by_surfaces :
  let z1 (x y : ℝ) := 2*x^2 + 18*y^2,
      z2 := 6
  ∃ V, (V = volume_enclosed ∧ 
        ∀ (x y : ℝ), 0 ≤ z1 x y ∧ z1 x y ≤ z2 → 
        ∃ r θ, ∫ (0..2*Real.pi) (λ θ, ∫ (0..1) (λ r, 12*r^3) dr) θ = V) :=
begin
  sorry
end

end volume_bounded_by_surfaces_l375_375946


namespace correct_judgments_l375_375568

theorem correct_judgments (a : ℝ) :
  (¬(∀ x, x ≥ 1 → (differentiable ℝ (λ x, x^2 - 2 * a * x) ∧ deriv (λ x, x^2 - 2 * a * x) x ≥ 0)) ∧ 
   (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 2^x1 - x1^2 = 0 ∧ 2^x2 - x2^2 = 0 ∧ 2^x3 - x3^2 = 0) ∧ 
   (∀ x, 2^|x| ≥ 1 ∧ ∃ x, 2^|x| = 1) ∧ 
   (∀ x, 2^x = 2^(-x) → x = 0)) :=
begin
  sorry
end

end correct_judgments_l375_375568


namespace triangle_bc_value_l375_375639

theorem triangle_bc_value (A B C: Type) [euclidean_geometry.bary A B C]
  (angle_B : ∠ B = 45)
  (AB : B.dist A = 100)
  (AC : C.dist A = 100 * real.sqrt 2) :
  B.dist C = 100 * real.sqrt ((5 : ℝ) + real.sqrt 2 * (real.sqrt 6 - real.sqrt 2)) := 
by sorry

end triangle_bc_value_l375_375639


namespace handshake_problem_7_boys_21_l375_375589

theorem handshake_problem_7_boys_21 :
  let n := 7
  let total_handshakes := n * (n - 1) / 2
  total_handshakes = 21 → (n - 1) = 6 :=
by
  -- Let n be the number of boys (7 in this case)
  let n := 7
  
  -- Define the total number of handshakes equation
  let total_handshakes := n * (n - 1) / 2
  
  -- Assume the total number of handshakes is 21
  intro h
  -- Proof steps would go here
  sorry

end handshake_problem_7_boys_21_l375_375589


namespace find_m_l375_375133

theorem find_m
  (x : ℝ)
  (m : ℝ)
  (h1 : log 10 (sqrt (sin x)) + log 10 (sqrt (cos x)) = -0.5)
  (h2 : log 10 (sin x + cos x) = 0.5 * (log 10 m - 2)) :
  m = 120.226 := sorry

end find_m_l375_375133


namespace bases_same_color_l375_375798

-- Define the vertices of the two pentagons (not necessarily in Euclidean space)
inductive Vertex
| A (n : Nat) : Vertex  -- vertices for one pentagon
| B (n : Nat) : Vertex  -- vertices for the other pentagon

def edge (v1 v2 : Vertex) : Prop := sorry -- Relation representing an edge between two vertices

-- Define the coloration (red or blue) of edges
inductive Color
| Red
| Blue

-- Color assignment to edges
def color_of_edge (v1 v2 : Vertex) [edge v1 v2] : Color := sorry

-- Condition that states in every triangle with colored edges, there exists at least one red side and one blue side
def valid_coloring : Prop :=
  ∀ (v1 v2 v3 : Vertex) [edge v1 v2] [edge v2 v3] [edge v3 v1],
    (color_of_edge v1 v2 ≠ color_of_edge v2 v3) ∨
    (color_of_edge v2 v3 ≠ color_of_edge v3 v1) ∨
    (color_of_edge v3 v1 ≠ color_of_edge v1 v2)

-- Theorem to prove that all sides of the bases are the same color
theorem bases_same_color (valid_coloring : valid_coloring) : 
  (∀ i j, 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 →
    (color_of_edge (Vertex.A i) (Vertex.A j) = color_of_edge (Vertex.B i) (Vertex.B j))) := 
  sorry

end bases_same_color_l375_375798


namespace probability_at_least_one_boy_and_one_girl_in_four_children_l375_375937

theorem probability_at_least_one_boy_and_one_girl_in_four_children :
  ∀ (n : ℕ), n = 4 → 
  (∀ (p : ℚ), p = 1 / 2 →
  ((1 : ℚ) - ((p ^ n) + (p ^ n)) = 7 / 8)) :=
by
  intro n hn p hp
  rw [hn, hp]
  norm_num
  sorry

end probability_at_least_one_boy_and_one_girl_in_four_children_l375_375937


namespace bridge_length_l375_375920

theorem bridge_length
  (train_length : ℝ)
  (train_speed_km_hr : ℝ)
  (crossing_time_sec : ℝ)
  (train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600)
  (total_distance : ℝ := train_speed_m_s * crossing_time_sec)
  (bridge_length : ℝ := total_distance - train_length)
  (train_length_val : train_length = 110)
  (train_speed_km_hr_val : train_speed_km_hr = 36)
  (crossing_time_sec_val : crossing_time_sec = 24.198064154867613) :
  bridge_length = 131.98064154867613 :=
by
  sorry

end bridge_length_l375_375920


namespace mixed_fractions_product_l375_375869

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375869


namespace profit_percentage_is_20_l375_375918

open Real

-- Define the conditions
def tea_80_kg := 80
def cost_tea_80_kg := 15
def tea_20_kg := 20
def cost_tea_20_kg := 20
def sale_price_per_kg := 19.2

-- Calculate the total cost
def total_cost := tea_80_kg * cost_tea_80_kg + tea_20_kg * cost_tea_20_kg

-- Calculate the total weight
def total_weight := tea_80_kg + tea_20_kg

-- Calculate the cost price per kg
def cost_price_per_kg := total_cost / total_weight

-- Define the profit per kg
def profit_per_kg := sale_price_per_kg - cost_price_per_kg

-- Define the profit percentage
def profit_percentage := (profit_per_kg / cost_price_per_kg) * 100

-- Theorem: Prove that the profit percentage the trader wants to earn is 20%
theorem profit_percentage_is_20 :
  profit_percentage = 20 := by
  sorry

end profit_percentage_is_20_l375_375918


namespace cookies_in_each_bag_l375_375723

theorem cookies_in_each_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 75) (h2 : num_bags = 25) : total_cookies / num_bags = 3 :=
by
  rw [h1, h2]
  norm_num
  sorry

end cookies_in_each_bag_l375_375723


namespace tea_consumption_eq1_tea_consumption_eq2_l375_375756

theorem tea_consumption_eq1 (k : ℝ) (w_sunday t_sunday w_wednesday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : w_wednesday = 4) : 
  t_wednesday = 6 := 
  by sorry

theorem tea_consumption_eq2 (k : ℝ) (w_sunday t_sunday t_thursday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : t_thursday = 2) : 
  w_thursday = 12 := 
  by sorry

end tea_consumption_eq1_tea_consumption_eq2_l375_375756


namespace number_of_males_choosing_malt_l375_375223

-- Definitions of conditions as provided in the problem
def total_males : Nat := 10
def total_females : Nat := 16

def total_cheerleaders : Nat := total_males + total_females

def females_choosing_malt : Nat := 8
def females_choosing_coke : Nat := total_females - females_choosing_malt

noncomputable def cheerleaders_choosing_malt (M_males : Nat) : Nat :=
  females_choosing_malt + M_males

noncomputable def cheerleaders_choosing_coke (M_males : Nat) : Nat :=
  females_choosing_coke + (total_males - M_males)

theorem number_of_males_choosing_malt : ∃ (M_males : Nat), 
  cheerleaders_choosing_malt M_males = 2 * cheerleaders_choosing_coke M_males ∧
  cheerleaders_choosing_malt M_males + cheerleaders_choosing_coke M_males = total_cheerleaders ∧
  M_males = 9 := 
by
  sorry

end number_of_males_choosing_malt_l375_375223


namespace ramu_paid_for_old_car_l375_375703

theorem ramu_paid_for_old_car (repairs : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (P : ℝ) :
    repairs = 12000 ∧ selling_price = 64900 ∧ profit_percent = 20.185185185185187 → 
    selling_price = P + repairs + (P + repairs) * (profit_percent / 100) → 
    P = 42000 :=
by
  intros h1 h2
  sorry

end ramu_paid_for_old_car_l375_375703


namespace min_value_expression_l375_375520

theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, 2 * a^2 + 3 * a * b + 4 * b^2 + 5 ≥ 5) ∧ (2 * x^2 + 3 * x * y + 4 * y^2 + 5 = 5) := 
by 
sorry

end min_value_expression_l375_375520


namespace painting_time_l375_375729

theorem painting_time (taylor_time jennifer_time : ℝ) (h1 : taylor_time = 12) (h2 : jennifer_time = 10) : 
  (1 / (1 / taylor_time + 1 / jennifer_time)) = 60 / 11 :=
by
  have taylor_rate : ℝ := 1 / taylor_time
  have jennifer_rate : ℝ := 1 / jennifer_time
  have combined_rate : ℝ := taylor_rate + jennifer_rate
  have time_to_paint : ℝ := 1 / combined_rate
  calc 
    time_to_paint = 60 / 11 : sorry

end painting_time_l375_375729


namespace geometric_sequence_general_term_sum_of_fn_terms_l375_375137

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h_increasing : ∀ n m, n < m → a n < a m)
  (h_a2_a3 : a 1 + a 2 = 4) (h_a1_a4 : a 0 * a 3 = 3) :
  ∀ n, a n = 3^(n - 2) := sorry

theorem sum_of_fn_terms (b : ℕ → ℝ) (a : ℕ → ℝ) (h_b_def : ∀ n, b n = n * a n) 
  (h_a_n : ∀ n, a n = 3^(n - 2)) :
  ∀ n, (∑ i in Finset.range n, b i) = (1/4) * (2 * n - 1) * 3^(n - 1) + (1/12) := sorry

end geometric_sequence_general_term_sum_of_fn_terms_l375_375137


namespace fraction_to_decimal_l375_375024

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375024


namespace solution_count_valid_numbers_l375_375202

def is_valid_number (n : ℕ) : Prop :=
  200 ≤ n ∧ n < 500 ∧ (∃ d ∈ [n / 10^2, (n / 10) % 10, n % 10], d = 3) ∧ n % 10 ≠ 5

def count_valid_numbers : ℕ :=
  Nat.count is_valid_number (λ x, x) 200 499

theorem solution_count_valid_numbers : count_valid_numbers = 132 := sorry

end solution_count_valid_numbers_l375_375202


namespace consecutive_numbers_sum_l375_375825

theorem consecutive_numbers_sum (n : ℤ) (h1 : (n - 1) * n * (n + 1) = 210) (h2 : ∀ m, (m - 1) * m * (m + 1) = 210 → (m - 1)^2 + m^2 + (m + 1)^2 ≥ (n - 1)^2 + n^2 + (n + 1)^2) :
  (n - 1) + n = 11 :=
by 
  sorry

end consecutive_numbers_sum_l375_375825


namespace fraction_to_decimal_l375_375052

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375052


namespace sequence_explicit_formula_l375_375633

noncomputable def sequence_a : ℕ → ℝ
| 0     => 0  -- Not used, but needed for definition completeness
| 1     => 3
| (n+1) => n / (n + 1) * sequence_a n

theorem sequence_explicit_formula (n : ℕ) (h : n ≠ 0) :
  sequence_a n = 3 / n :=
by sorry

end sequence_explicit_formula_l375_375633


namespace gcd_possible_values_count_l375_375386

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375386


namespace length_and_width_of_prism_l375_375422

theorem length_and_width_of_prism (w l h d : ℝ) (h_cond : h = 12) (d_cond : d = 15) (length_cond : l = 3 * w) :
  (w = 3) ∧ (l = 9) :=
by
  -- The proof is omitted as instructed in the task description.
  sorry

end length_and_width_of_prism_l375_375422


namespace find_number_l375_375813

theorem find_number (x : ℝ) (h : (x / 4) + 3 = 5) : x = 8 :=
by
  sorry

end find_number_l375_375813


namespace fuel_tank_capacity_l375_375931

theorem fuel_tank_capacity :
  let C := 204 in
  0.12 * 66 + 0.16 * (C - 66) = 30 := by
  sorry

end fuel_tank_capacity_l375_375931


namespace gcd_values_count_l375_375367

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375367


namespace fraction_to_decimal_l375_375036

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375036


namespace lines_parallel_l375_375998

variables (α β γ : Plane) (l m n : Line)

-- Assuming that l is perpendicular to α and m is perpendicular to α
axiom perp_l_alpha : l ⟂ α
axiom perp_m_alpha : m ⟂ α

-- Conclusion: lines l and m are parallel
theorem lines_parallel (l ⟂ α) (m ⟂ α): l ∥ m :=
sorry

end lines_parallel_l375_375998


namespace solve_fractions_l375_375913

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375913


namespace min_time_to_return_to_start_l375_375458

-- Definitions for the conditions
def length_of_circular_track : ℕ := 400
def walking_speed (km_per_hour : ℕ) : ℕ := (km_per_hour * 1000) / 60  -- in meters per minute

def movements (speeds directions : ℕ) : ℕ :=
  let distance1 := 1 * speeds  -- 1 minute clockwise
  let distance2 := -3 * speeds -- 3 minutes counterclockwise
  let distance3 := 5 * speeds  -- 5 minutes clockwise
  distance1 + distance2 + distance3

def net_displacement_on_track (track_length displacement : ℕ) : ℕ :=
  let net_distance := displacement % track_length
  if net_distance > track_length / 2 then track_length - net_distance else net_distance

-- Lean 4 theorem statement
theorem min_time_to_return_to_start (v : ℕ) (l : ℕ) :
  v = walking_speed 6 →
  l = length_of_circular_track →
  let displacement := movements v 1 in  -- directions are encoded in the function movements
  net_displacement_on_track l displacement = v := by
  sorry

end min_time_to_return_to_start_l375_375458


namespace num_lines_through_four_points_in_3D_grid_l375_375585

/-- The count of lines passing through four distinct points in a 3D grid with points \((i, j, k)\) where \(1 ≤ i, j, k ≤ 5\) is 156. -/
theorem num_lines_through_four_points_in_3D_grid : 
  let points := { (i, j, k) : ℕ × ℕ × ℕ | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 ∧ 1 ≤ k ∧ k ≤ 5 } in
  let is_valid_line (line : (ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ)) := 
    let (p1, p2) := line in
    ∃ d : (ℤ × ℤ × ℤ), d ≠ (0, 0, 0) ∧ 
      (p2.1 : ℤ) = p1.1 + d.1 ∧ 
      (p2.2 : ℤ) = p1.2 + d.2 ∧ 
      (p2.3 : ℤ) = p1.3 + d.3 ∧ 
      (p1.1 + 3 * d.1 : ℤ) ∈ {1, 2, 3, 4, 5} ∧ 
      (p1.2 + 3 * d.2 : ℤ) ∈ {1, 2, 3, 4, 5} ∧ 
      (p1.3 + 3 * d.3 : ℤ) ∈ {1, 2, 3, 4, 5} in
  (finset.univ.product finset.univ).filter is_valid_line).card = 156 :=
by
  sorry

end num_lines_through_four_points_in_3D_grid_l375_375585


namespace find_k_l375_375171

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l375_375171


namespace final_projection_matrix_l375_375669

-- Definitions for the problem conditions
def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / (a^2 + b^2)) • Matrix.of_list [[a^2, a * b], [a * b, b^2]]

def matrix_multiplication {m n p : Type*} [Fintype m] [Fintype n] [Fintype p] (A : Matrix m n ℝ) (B : Matrix n p ℝ) : Matrix m p ℝ :=
  Matrix.mul A B

-- Proving the final matrix from v0 to v2
theorem final_projection_matrix :
  let P1 := projection_matrix 3 1 in
  let P2 := projection_matrix 1 1 in
  matrix_multiplication P2 P1 = Matrix.of_list [[3/5, 1/5], [3/5, 1/5]] :=
by
  sorry

end final_projection_matrix_l375_375669


namespace market_value_of_stock_l375_375427

-- Definitions based on the conditions
def face_value : ℝ := 100
def percentage_stock : ℝ := 14
def yield_stock : ℝ := 8

-- Equivalent proof problem in Lean
theorem market_value_of_stock :
  let dividend_per_share := (percentage_stock / 100) * face_value in
  let market_value := (dividend_per_share / (yield_stock / 100)) in
  market_value = 175 :=
by
  sorry

end market_value_of_stock_l375_375427


namespace average_marks_is_77_l375_375824

-- Define the marks scored in each subject
def marks_math : ℕ := 76
def marks_science : ℕ := 65
def marks_social_studies : ℕ := 82
def marks_english : ℕ := 67
def marks_biology : ℕ := 95

-- Define the number of subjects
def num_subjects : ℕ := 5

-- Define total marks as the sum of the scores in all subjects
def total_marks : ℕ := marks_math + marks_science + marks_social_studies + marks_english + marks_biology

-- Define the average marks
def average_marks : ℕ := total_marks / num_subjects

-- Prove that the average marks is 77
theorem average_marks_is_77 : average_marks = 77 :=
by
  have h_total : total_marks = 76 + 65 + 82 + 67 + 95 := rfl
  have h_sum : total_marks = 385 := by
    rw h_total
    norm_num
  have h_avg : average_marks = 385 / num_subjects := rfl
  have h_num : num_subjects = 5 := rfl
  rw [h_avg, h_num]
  norm_num
  sorry -- this is where the actual arithmetic proof would go

end average_marks_is_77_l375_375824


namespace percent_pepperjack_l375_375649

-- Define the total number of cheese sticks
def total_cheese_sticks : ℕ := 15 + 30 + 45

-- Define the number of pepperjack cheese sticks
def pepperjack_sticks : ℕ := 45

-- The percentage chance that a randomly picked cheese stick is pepperjack
theorem percent_pepperjack : (pepperjack_sticks * 100) / total_cheese_sticks = 50 := by
  calc
    (pepperjack_sticks * 100) / total_cheese_sticks = (45 * 100) / 90 : by rw [pepperjack_sticks, total_cheese_sticks]
    ... = 4500 / 90 : rfl
    ... = 50 : by norm_num

end percent_pepperjack_l375_375649


namespace cos_inequality_l375_375496

theorem cos_inequality (x y : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π / 2) (hy : 0 ≤ y) (hy' : y ≤ π) :
  cos (x + y) ≤ cos x * cos y :=
sorry

end cos_inequality_l375_375496


namespace local_minimum_f_when_k2_at_x1_l375_375554

def f (x : ℝ) (k : ℕ) :=
  (Real.exp x - 1) * (x - 1)^k

theorem local_minimum_f_when_k2_at_x1 :
  ∃ ε > 0, ∀ x, |x - 1| < ε → (f x 2 ≥ f 1 2) :=
by
  sorry

end local_minimum_f_when_k2_at_x1_l375_375554


namespace least_product_two_primes_gt_10_l375_375791

open Nat

/-- Two distinct primes, each greater than 10, are multiplied. What is the least possible product of these two primes? -/
theorem least_product_two_primes_gt_10 : ∃ (p q : ℕ), p > 10 ∧ q > 10 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 143 :=
sorry

end least_product_two_primes_gt_10_l375_375791


namespace gcd_lcm_product_360_l375_375382

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375382


namespace volume_ratio_l375_375463

def s : ℝ := arbitrary

def V_d (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) * s^3 / 4

noncomputable def V_i : ℝ :=
  (5 * (3 + Real.sqrt 5)) * (s * Real.sqrt 3 / 2) ^ 3 / 12

theorem volume_ratio (s : ℝ) :
  let Vd := V_d s in
  let Vi := V_i in
  (Vi / Vd) = ((45 * Real.sqrt 3 * (3 + Real.sqrt 5)) / (384 * (15 + 7 * Real.sqrt 5))) :=
by
  sorry

end volume_ratio_l375_375463


namespace doritos_in_each_pile_l375_375689

theorem doritos_in_each_pile (total_chips : ℕ) (quarter_of_chips : ℕ) (piles : ℕ) (chips_per_pile : ℕ) 
  (h1 : total_chips = 80) 
  (h2 : quarter_of_chips = total_chips / 4) 
  (h3 : piles = 4)
  (h4 : chips_per_pile = quarter_of_chips / piles) : 
  chips_per_pile = 5 :=
by 
  rw [h1, h2, h3, h4]
  rw nat.div_self;
  sorry

end doritos_in_each_pile_l375_375689


namespace fraction_to_decimal_l375_375030

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375030


namespace minimum_black_edges_of_cube_l375_375958

-- Defining the cube structure with conditions on edges.
structure Cube (E : Type) :=
(edges : fin 12 → E)
(color : E → Prop)

-- Defining the condition that each face has exactly 2 black edges.
def two_black_edges_per_face (C : Cube E) (is_black : E → Prop) : Prop :=
  ∀ (faces : fin 6), (∃ (e1 e2 e3 e4 : fin 12), 
    (face_contains C.faces e1 ∧ face_contains C.faces e2 ∧
     face_contains C.faces e3 ∧ face_contains C.faces e4 ∧
     is_black (C.edges e1) ∧ is_black (C.edges e2) ∧
     ¬is_black (C.edges e3) ∧ ¬is_black (C.edges e4)))

-- Main theorem statement in Lean.
theorem minimum_black_edges_of_cube : ∃ (C : Cube bool) (is_black : bool → Prop),
     two_black_edges_per_face C is_black ∧ 
     (∀ C' is_black', two_black_edges_per_face C' is_black' → (card (set_of is_black' ≤ 8))) :=
sorry

end minimum_black_edges_of_cube_l375_375958


namespace expression_evaluation_l375_375007

theorem expression_evaluation :
  (1 / 9 - 1 / 5 + 1 / 2)⁻¹ = (90 / 37) :=
by
  sorry

end expression_evaluation_l375_375007


namespace triangle_side_length_l375_375642

-- Definitions for the conditions given in the problem
def angleB : ℝ := 45
def AB : ℝ := 100
def AC : ℝ := 100 * Real.sqrt 2

-- Statement of the theorem/problem
theorem triangle_side_length (ABC : Type) [triangle ABC] 
  (angleB : ∠B = 45)
  (ABeq : AB = 100)
  (ACeq : AC = 100 * Real.sqrt 2) :
  BC ≈ Real.sqrt (30000 + 5160 * Real.sqrt 2) :=
sorry

end triangle_side_length_l375_375642


namespace length_of_AC_l375_375996

theorem length_of_AC
  (south : ℕ)
  (west : ℕ)
  (north : ℕ)
  (east : ℕ)
  (south_walked : south = 50)
  (west_walked : west = 30)
  (north_walked : north = 15)
  (east_walked : east = 10) :
  real.sqrt ((south - north)^2 + (west - east)^2) = 5 * real.sqrt 65 :=
by
  sorry

end length_of_AC_l375_375996


namespace imaginary_part_of_complex_number_l375_375100

open Complex

theorem imaginary_part_of_complex_number :
  ∀ (i : ℂ), i^2 = -1 → im ((2 * I) / (2 + I^3)) = 4 / 5 :=
by
  intro i hi
  sorry

end imaginary_part_of_complex_number_l375_375100


namespace river_depth_difference_l375_375235

theorem river_depth_difference
  (mid_may_depth : ℕ)
  (mid_july_depth : ℕ)
  (mid_june_depth : ℕ)
  (H1 : mid_july_depth = 45)
  (H2 : mid_may_depth = 5)
  (H3 : 3 * mid_june_depth = mid_july_depth) :
  mid_june_depth - mid_may_depth = 10 := 
sorry

end river_depth_difference_l375_375235


namespace fraction_to_decimal_l375_375086

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375086


namespace seat_three_is_Abby_l375_375924

-- Define seats
inductive seat
| one 
| two 
| three 
| four

open seat

-- People
inductive person
| Abby
| Bret
| Carl
| Dana

open person

-- Define the positions of individuals
def sits_at (p : person) (s : seat) : Prop :=
  match s with
  | one => p = Bret
  | two => p = Carl
  | three => p = Abby
  | four => p = Dana

-- Theorem statement
theorem seat_three_is_Abby :
  (sits_at Bret one) ∧ 
  ((sits_at Carl two) ∨ (sits_at Carl three)) ∧ 
  (¬ (sits_at Dana (if (sits_at Abby one) then two else (if (sits_at Abby two) then one else three)))) →
  sits_at Abby three :=
by
  sorry

end seat_three_is_Abby_l375_375924


namespace quadratic_roots_distinct_l375_375764

theorem quadratic_roots_distinct (k : ℝ) : 
  let Δ := (k - 5)^2 + 12 * k
  Δ > 0 :=
by
  let Δ := (k - 5)^2 + 12 * k
  have : (k + 1)^2 + 24 > 0 := by
    sorry
  exact this

end quadratic_roots_distinct_l375_375764


namespace fraction_to_decimal_l375_375032

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375032


namespace restore_original_problem_l375_375907

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375907


namespace correct_system_of_equations_l375_375836

variables (x y : ℝ)

-- Conditions based on the problem statement
def condition1 := 5 * x + 2 * y = 10
def condition2 := 2 * x + 5 * y = 8

theorem correct_system_of_equations : (condition1) ∧ (condition2) :=
by
  -- skipping the actual proof implementation
  sorry

end correct_system_of_equations_l375_375836


namespace cyclic_quadrilateral_A_X_O_Y_l375_375253

-- Define the points and the required conditions
variables {A B C I O X Y : Type*}

-- Define the incenter and circumcenter properties
variable [triangle : Triangle ABC] 
variable [Incenter I ABC]
variable [Circumcenter O ABC]

-- Define the line L's properties
variable {L : line}

-- Conditions for line L
noncomputable def LineL_is_parallel_to_BC_and_tangent_to_incircle :
  Parallel L (line_through B C) ∧ Tangent L (incircle I ABC) := sorry

-- Properties for point X on line L
variable (X_on_L: Point_on_line X L)
variable (L_intersections_IO_at_X: LineIntersection L (line_through I O) X)

-- Properties for point Y on line L
variable (Y_on_L: Point_on_line Y L)
variable (YI_perpendicular_to_IO: Perpendicular (line_through Y I) (line_through I O))

-- The theorem statement
theorem cyclic_quadrilateral_A_X_O_Y :
  Cyclic_quad A X O Y :=
sorry

end cyclic_quadrilateral_A_X_O_Y_l375_375253


namespace find_k_l375_375192

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l375_375192


namespace mass_percentage_O_in_N2O3_l375_375101

-- Given conditions
def molar_mass_N : ℝ := 14.01
def molar_mass_O : ℝ := 16.00
def N2O3_formula : ℕ × ℕ := (2, 3)

-- Proof that the mass percentage of oxygen in N2O3 is 63.15%
theorem mass_percentage_O_in_N2O3
  (mN : ℝ := molar_mass_N)
  (mO : ℝ := molar_mass_O)
  (n2o3 : ℕ × ℕ := N2O3_formula) :
  let M_N2O3 := 2 * mN + 3 * mO in
  let mass_O := 3 * mO in
  (mass_O / M_N2O3) * 100 = 63.15 :=
by
  -- Proof steps are omitted
  sorry

end mass_percentage_O_in_N2O3_l375_375101


namespace find_beta_l375_375560

theorem find_beta (α β : ℝ) (h : α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2)
(ha : cos α = 1/7) (hab : cos (α + β) = -11/14) : 
β = π / 3 :=
by
  sorry

end find_beta_l375_375560


namespace quadratic_trinomial_int_l375_375094

theorem quadratic_trinomial_int (a b c x : ℤ) (h : y = (x - a) * (x - 6) + 1) :
  ∃ (b c : ℤ), (x + b) * (x + c) = (x - 8) * (x - 6) + 1 :=
by
  sorry

end quadratic_trinomial_int_l375_375094


namespace closest_angle_l375_375314

/-- Prove the angle formed by the side edge and the base of a regular quadrilateral pyramid 
with base edge length 2017 and side edge length 2000 is closest to 40 degrees from the given options. -/
theorem closest_angle (a : ℝ) (l : ℝ) (θ : ℝ) (h : ℝ) :
  a = 2017 → l = 2000 →
  h = Real.sqrt (l^2 - (a/2)^2) →
  θ = Real.arccos ((a/2) / l) →
  θ < Real.pi / 4 →
  40 < θ * 180 / Real.pi * 1.0 ∧  θ * 180 / Real.pi < 50 -> θ * 180 / Real.pi = 40.0 :=
sorry

end closest_angle_l375_375314


namespace essential_infinity_n_l375_375526

def Q (x : ℕ) : ℕ := (Vector.ofFn x.digits).sum
def P (x : ℕ) : ℕ := (Vector.ofFn x.digits).prod

theorem essential_infinity_n (n : ℕ) : ∃ᶠ x in (Filter.atTop : Filter ℕ), Q(Q(x)) + P(Q(x)) + Q(P(x)) + P(P(x)) = n :=
sorry

end essential_infinity_n_l375_375526


namespace donuts_selection_l375_375281

theorem donuts_selection : ∀ (g c p s : ℕ), g + c + p + s = 4 →
  (nat.choose 7 3) = 35 :=
by
-- We assume the hypotheses to hold
intros g c p s h,
-- The proof of the theorem using the conditions and the correct answer
-- will be based on the stars and bars theorem,
-- but we'll annotate the placeholder for now
sorry

end donuts_selection_l375_375281


namespace solve_for_y_l375_375716

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l375_375716


namespace gcd_possible_values_count_l375_375376

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375376


namespace fraction_to_decimal_l375_375067

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375067


namespace least_perimeter_of_triangle_l375_375321

-- Define the sides of the triangle
def side1 : ℕ := 40
def side2 : ℕ := 48

-- Given condition for the third side
def valid_third_side (x : ℕ) : Prop :=
  8 < x ∧ x < 88

-- The least possible perimeter given the conditions
def least_possible_perimeter : ℕ :=
  side1 + side2 + 9

theorem least_perimeter_of_triangle (x : ℕ) (h : valid_third_side x) (hx : x = 9) : least_possible_perimeter = 97 :=
by
  rw [least_possible_perimeter]
  exact rfl

end least_perimeter_of_triangle_l375_375321


namespace points_lie_on_line_l375_375991

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
    let x := (t + 2) / t
    let y := (t - 2) / t
    x + y = 2 :=
by
  let x := (t + 2) / t
  let y := (t - 2) / t
  sorry

end points_lie_on_line_l375_375991


namespace detergent_usage_l375_375833

theorem detergent_usage (detergent_per_pound : ℕ) (pounds_of_clothes : ℕ) (detergent_needed : ℕ) :
  detergent_per_pound = 2 →
  pounds_of_clothes = 9 →
  detergent_needed = 18 →
  detergent_needed = detergent_per_pound * pounds_of_clothes :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end detergent_usage_l375_375833


namespace product_of_x_y_l375_375414

theorem product_of_x_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) : x * y = 72 :=
by
  sorry

end product_of_x_y_l375_375414


namespace inequality_reciprocal_l375_375264

theorem inequality_reciprocal (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : (1 / a < 1 / b) :=
by
  sorry

end inequality_reciprocal_l375_375264


namespace mixed_fraction_product_l375_375899

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375899


namespace minimize_sum_of_roots_l375_375106

noncomputable def f (a : ℝ) : ℝ := a^2 - real.sqrt 21 * a + 26
noncomputable def g (a : ℝ) : ℝ := (3 / 2) * a^2 - real.sqrt 21 * a + 27

def sum_of_roots_equation (a x : ℝ) : Prop :=
  (f a * x^2 + 1) / (x^2 + g a) = real.sqrt ((x * g a - 1) / (f a - x))

theorem minimize_sum_of_roots : ∃ a : ℝ, (∀ x : ℝ, sum_of_roots_equation a x) ∧
  a = real.sqrt 21 / 2 :=
sorry

end minimize_sum_of_roots_l375_375106


namespace trisector_length_correct_l375_375550

noncomputable def length_of_shorter_trisector {α : Type*} [LinearOrder α] 
[LinearOrderedField α] (DE EF: α) (h : DE = 5 ∧ EF = 12) : α :=
  let DF := Real.sqrt (DE^2 + EF^2) in
  let trisector_length := (5 * Real.sqrt 3) / 36 in
  trisector_length

theorem trisector_length_correct:
  ∀ {α : Type*} [LinearOrder α] [LinearOrderedField α],
    ∀ DE EF (h : DE = 5 ∧ EF = 12),
    length_of_shorter_trisector DE EF h = (5 * Real.sqrt 3) / 36 :=
begin
  intros,
  rw length_of_shorter_trisector,
  sorry
end

end trisector_length_correct_l375_375550


namespace triangle_incircle_tangent_l375_375643

theorem triangle_incircle_tangent (O X Y Z : Point) : 
  let A := ∠ XYZ;
  let B := ∠ YXZ;
  A = 70 ∧ B = 52 →
  let C := 180 - A - B;
  let θ := C / 2;
  θ = 29 :=
begin
  sorry
end

end triangle_incircle_tangent_l375_375643


namespace gcd_values_count_l375_375401

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375401


namespace average_production_last_5_days_l375_375218

theorem average_production_last_5_days (tv_per_day_25 : ℕ) (total_tv_30 : ℕ) :
  tv_per_day_25 = 63 →
  total_tv_30 = 58 * 30 →
  (total_tv_30 - tv_per_day_25 * 25) / 5 = 33 :=
by
  intros h1 h2
  sorry

end average_production_last_5_days_l375_375218


namespace probability_of_five_green_marbles_l375_375698

open ProbabilityTheory

-- Define the conditions
def total_marbles := 12
def green_marbles := 8
def purple_marbles := 4
def trials := 8

-- Define the event of interest
def probability_of_exactly_five_green : ℚ :=
  56 * ((2 / 3)^5) * ((1 / 3)^3)

theorem probability_of_five_green_marbles :
  (probability_of_exactly_five_green.to_real = 0.273) :=
  sorry

end probability_of_five_green_marbles_l375_375698


namespace probability_winning_1000_yuan_answering_A_first_higher_expected_prize_amount_answering_B_first_l375_375781

theorem probability_winning_1000_yuan_answering_A_first :
  (1 / 3) * (1 - (1 / 4)) = 1 / 4 :=
by
  sorry

theorem higher_expected_prize_amount_answering_B_first :
  let P1 := 1 / 3
  let P2 := 1 / 4
  let E_xi := 0 * (2 / 3) + 1000 * (1 / 4) + 3000 * (1 / 12)
  let E_eta := 0 * (3 / 4) + 2000 * (1 / 6) + 3000 * (1 / 12)
  E_eta > E_xi :=
by
  have h1 : 0 * (2 / 3) = 0 := by ring
  have h2 : 1000 * (1 / 4) = 250 := by norm_num
  have h3 : 3000 * (1 / 12) = 250 := by norm_num
  have h4 : E_xi = 500 := by rw [h1, h2, h3]; norm_num
  have h5 : 0 * (3 / 4) = 0 := by ring
  have h6 : 2000 * (1 / 6) = 333.3333 := by norm_num
  have h7 : 3000 * (1 / 12) = 250 := by norm_num
  have h8 : E_eta = 583.3333 := by rw [h5, h6, h7]; norm_num
  show E_eta > E_xi from by rw [h4, h8]; norm_num
  sorry

end probability_winning_1000_yuan_answering_A_first_higher_expected_prize_amount_answering_B_first_l375_375781


namespace log_sum_geometric_sequence_l375_375559

variable (a : ℕ → ℝ)
variable (h₀ : ∀ n, a n > 0)
variable (h₁ : a 5 * a 6 + a 4 * a 7 = 8)

theorem log_sum_geometric_sequence :
  (Real.log 2 (a 1) + Real.log 2 (a 2) + Real.log 2 (a 3) + Real.log 2 (a 4) +
   Real.log 2 (a 5) + Real.log 2 (a 6) + Real.log 2 (a 7) + Real.log 2 (a 8) +
   Real.log 2 (a 9) + Real.log 2 (a 10)) = 10 :=
by
  sorry

end log_sum_geometric_sequence_l375_375559


namespace manufacturer_profit_percentage_l375_375451

-- Definitions and conditions
def cost_price_manufacturer : ℝ := 17
def selling_price_retailer : ℝ := 30.09
def profit_percent_retailer : ℝ := 25
def profit_percent_wholesaler : ℝ := 20

-- Computed intermediate results (used in the equivalent proof problem, but not directly in the definitions for Lean)
def cost_price_retailer := selling_price_retailer / (1 + profit_percent_retailer / 100)
def cost_price_wholesaler := cost_price_retailer / (1 + profit_percent_wholesaler / 100)
def selling_price_manufacturer := cost_price_wholesaler
def profit_manufacturer := selling_price_manufacturer - cost_price_manufacturer

-- The profit percentage for the manufacturer we want to prove
def profit_percent_manufacturer := (profit_manufacturer / cost_price_manufacturer) * 100

-- The target statement to prove
theorem manufacturer_profit_percentage :
  profit_percent_manufacturer = 18 := by
  sorry

end manufacturer_profit_percentage_l375_375451


namespace min_value_of_3x_2y_l375_375549

noncomputable def min_value (x y: ℝ) : ℝ := 3 * x + 2 * y

theorem min_value_of_3x_2y (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y - x * y = 0) :
  min_value x y = 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_3x_2y_l375_375549


namespace calculate_length_l375_375844

theorem calculate_length (rent_per_acre_per_month : ℝ)
                         (total_rent_per_month : ℝ)
                         (width_of_plot : ℝ)
                         (square_feet_per_acre : ℝ) :
  rent_per_acre_per_month = 60 →
  total_rent_per_month = 600 →
  width_of_plot = 1210 →
  square_feet_per_acre = 43560 →
  (total_rent_per_month / rent_per_acre_per_month) * square_feet_per_acre / width_of_plot = 360 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    (600 / 60) * 43560 / 1210 = 10 * 43560 / 1210  : by rw div_eq_mul_inv
                            ... = 435600 / 1210    : by norm_num
                            ... = 360              : by norm_num
  -- Proof using calc block to show step-by-step result
  sorry

end calculate_length_l375_375844


namespace dealership_sedans_sales_l375_375745

theorem dealership_sedans_sales (sports_cars sedans : ℕ) 
  (h1 : 3 * sedans = 5 * sports_cars)
  (h2 : sports_cars = 45)
  (h3 : sedans ≥ sports_cars + 20) : 
  sedans = 75 := 
begin
  sorry
end

end dealership_sedans_sales_l375_375745


namespace circle_y_axis_intersection_y_coord_l375_375441

-- Define the endpoints of the diameter of the circle
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (10, 0)

-- Prove that the y-coordinate of the intersection with the y-axis is 0
theorem circle_y_axis_intersection_y_coord :
  ∃ y : ℝ, y = 0 ∧ (∃ x : ℝ, (x = 0 ∧ ((x - ((fst point1 + fst point2) / 2))^2 + y^2 = ((dist point1 point2) / 2)^2))) :=
by
  sorry

end circle_y_axis_intersection_y_coord_l375_375441


namespace cookies_distribution_probability_l375_375842

-- Define the total number of cookies and the conditions of distribution.
def num_cookies := 12
def num_students := 4
def num_types := 3
def cookies_per_student := 3

-- Define the problem as ensuring the probability p/q simplifies as described and leads to the correct sum.
theorem cookies_distribution_probability :
  let probability := (9 / 55) * (9 / 28) * (2 / 5)
  ∃ p q : ℕ, RelativelyPrime p q ∧ probability = p / q ∧ p + q = 3931 :=
by
  sorry

end cookies_distribution_probability_l375_375842


namespace find_k_of_vectors_orthogonal_l375_375162

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l375_375162


namespace find_range_last_year_l375_375252

def annual_yield_range_last_year 
    (range_this_year : ℝ) 
    (improvement_percentage : ℝ) 
    (range_last_year : ℝ) 
    (h : range_this_year = range_last_year * (1 + improvement_percentage)) : Prop :=
  range_last_year = 10000

theorem find_range_last_year :
  annual_yield_range_last_year 11500 0.15 10000 := 
by
  simp [annual_yield_range_last_year]
  sorry

end find_range_last_year_l375_375252


namespace gcd_possible_values_count_l375_375378

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375378


namespace number_of_recipes_needed_l375_375481

noncomputable def cookies_per_student : ℕ := 3
noncomputable def total_students : ℕ := 150
noncomputable def recipe_yield : ℕ := 20
noncomputable def attendance_drop_rate : ℝ := 0.30

theorem number_of_recipes_needed : 
  ⌈ (total_students * (1 - attendance_drop_rate) * cookies_per_student) / recipe_yield ⌉ = 16 := by
  sorry

end number_of_recipes_needed_l375_375481


namespace equation_of_line_l_l375_375322

theorem equation_of_line_l
  (a : ℝ)
  (l_intersects_circle : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + a = 0)
  (midpoint_chord : ∃ C : ℝ × ℝ, C = (-2, 3) ∧ ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + B.1) / 2 = C.1 ∧ (A.2 + B.2) / 2 = C.2) :
  a < 3 →
  ∃ l : ℝ × ℝ → Prop, (∀ x y : ℝ, l (x, y) ↔ x - y + 5 = 0) :=
by {
  sorry
}

end equation_of_line_l_l375_375322


namespace minimum_perimeter_of_octagon_at_zeros_l375_375677

noncomputable def P (z : ℂ) : ℂ :=
  z^8 + (6*real.sqrt 2 + 8) * z^4 - (6*real.sqrt 2 + 9)

theorem minimum_perimeter_of_octagon_at_zeros :
  let roots := {z : ℂ | P z = 0}
  ∃ polygon : {z : ℂ // P z = 0} → ℂ,
  (polygon.vertices.sorted_edges.sum edge_length = 4 * real.sqrt 2 + 4 * real.sqrt (3 + real.sqrt 2)) :=
sorry

end minimum_perimeter_of_octagon_at_zeros_l375_375677


namespace speed_of_B_is_8_l375_375428

noncomputable def speed_of_B (crosses : ℕ) (time : ℝ) (speed_A : ℝ) : ℝ :=
  let v_B := crosses / time - speed_A
  v_B

theorem speed_of_B_is_8 :
  let crosses := 10 in
  let time := 1 in
  let speed_A := 2 in
  speed_of_B crosses time speed_A = 8 :=
by
  sorry

end speed_of_B_is_8_l375_375428


namespace polygon_sides_l375_375464

theorem polygon_sides (R : ℝ) (n : ℕ) (h : R ≠ 0)
  (h_area : (1 / 2) * n * R^2 * Real.sin (360 / n * (Real.pi / 180)) = 4 * R^2) :
  n = 8 := 
by
  sorry

end polygon_sides_l375_375464


namespace sum_common_divisors_of_8_and_12_l375_375103

open Finset

theorem sum_common_divisors_of_8_and_12 : (∑ x in (filter (λ d, 8 % d = 0 ∧ 12 % d = 0) (range 13)), x) = 7 :=
by
  sorry

end sum_common_divisors_of_8_and_12_l375_375103


namespace ball_arrangements_l375_375425

theorem ball_arrangements :
  let redCount := 24
      whiteCount := 11
      totalBalls := 35
  in
  let satisfies_conditions (arrangement : List ℕ) : Prop :=
      (∀ i, i < totalBalls - 1 → arrangement[i] = 0 → arrangement[i + 1] ≠ 0) ∧
      (∀ i, i ≤ totalBalls - 7 → (arrangement.slice i 7).contains 0)
  in
  let arrangements := {arrangement : List ℕ // satisfies_conditions arrangement ∧ arrangement.length = totalBalls ∧ arrangement.count 0 = redCount ∧ arrangement.count 1 = whiteCount }
  in
  arrangements.toFinset.card = 31 := sorry

end ball_arrangements_l375_375425


namespace area_parallelogram_roots_l375_375567

noncomputable def area_parallelogram (z : ℂ) (w : ℂ) : ℂ := 
  Complex.abs (Complex.im (z * Complex.conj w))

theorem area_parallelogram_roots 
  (z1 z2 : ℂ) (h1 : z1 ^ 2 = 1 + 3 * Complex.sqrt 15 * Complex.I)
                   (h2 : z2 ^ 2 = 3 + 5 * Complex.sqrt 3 * Complex.I) : 
  area_parallelogram 
    (2 * (Complex.sqrt (17 + Complex.sqrt 544) + 
            Complex.sqrt (17 - Complex.sqrt 544) * Complex.I)) 
    (3 * Complex.sqrt 2 * (Complex.sqrt 3 + Complex.I)) = 
  40 * Complex.sqrt 17 - 24 * Complex.sqrt 3 := 
sorry

end area_parallelogram_roots_l375_375567


namespace percentage_is_0_point_3_l375_375837

theorem percentage_is_0_point_3 (P : ℝ) :
  (0.15 * P * 0.5 * 4800 = 108) → (P = 0.3) := 
by 
  intro h,
  sorry

end percentage_is_0_point_3_l375_375837


namespace length_of_first_two_CDs_l375_375647

theorem length_of_first_two_CDs
  (x : ℝ)
  (h1 : x + x + 2 * x = 6) :
  x = 1.5 := 
sorry

end length_of_first_two_CDs_l375_375647


namespace gcd_possible_values_count_l375_375389

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375389


namespace restore_original_problem_l375_375904

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375904


namespace batsman_average_proof_l375_375840

noncomputable def batsman_average_after_17th_inning (A : ℝ) : ℝ :=
  (A * 16 + 87) / 17

theorem batsman_average_proof (A : ℝ) (h1 : 16 * A + 87 = 17 * (A + 2)) : batsman_average_after_17th_inning 53 = 55 :=
by
  sorry

end batsman_average_proof_l375_375840


namespace clubs_with_common_member_l375_375834

theorem clubs_with_common_member (n : ℕ) (h : n ≥ 4)
  (clubs : Finset (Finset ℕ))
  (clubs_size : clubs.card = n + 1)
  (each_club_size : ∀ club ∈ clubs, club.card = 3)
  (distinct_clubs : ∀ (c1 c2 ∈ clubs), c1 ≠ c2 → c1 ≠ c2 → c1 ∩ c2 ≠ ∅ → c1 ∩ c2 ≠ ∅ → c1 ∩ c2).card = 1) :
  ∃ c1 c2 ∈ clubs, c1 ≠ c2 ∧ (c1 ∩ c2).card = 1 :=
sorry

end clubs_with_common_member_l375_375834


namespace snacks_sold_l375_375348

theorem snacks_sold (snacks_initial ramens_initial ramens_bought total_after : ℕ) 
  (h1 : snacks_initial = 1238)
  (h2 : ramens_initial = snacks_initial + 374)
  (h3 : ramens_bought = 276)
  (h4 : total_after = 2527)
  : 599 = snacks_initial + ramens_initial + ramens_bought - total_after :=
by
  have h5 : snacks_initial + ramens_initial + ramens_bought - total_after = 599 := sorry
  exact h5

end snacks_sold_l375_375348


namespace convert_fraction_to_decimal_l375_375057

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375057


namespace ellipse_parabola_common_point_l375_375148

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 := 
by 
  sorry

end ellipse_parabola_common_point_l375_375148


namespace find_value_of_a_l375_375434

theorem find_value_of_a :
  ∃ a, 
  let xs := [1, 2, 3, 4, 5],
      ys := [2, 3, 7, 8, a],
      mean_x := (xs.sum / 5),
      mean_y := (ys.sum / 5),
      y_hat := 2.1 * mean_x - 0.3
  in mean_y = y_hat ∧ a = 10 := 
by
  sorry

end find_value_of_a_l375_375434


namespace ratio_of_areas_l375_375615

-- Given conditions
variables (P : Type) [TruncatedQuadrilateralPyramid P]
variables (α : ℝ) -- the angle in radians between the intersecting planes
variables (A1 A2 : ℝ) -- areas of the sections

-- The theorem to prove
theorem ratio_of_areas (hα : α ≥ 0 ∧ α ≤ π / 2) :
  (A2 / A1) = 2 * cos α :=
sorry

end ratio_of_areas_l375_375615


namespace total_population_l375_375339

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end total_population_l375_375339


namespace factor_theorem_for_Q_l375_375811

variable (d : ℝ) -- d is a real number

def Q (x : ℝ) : ℝ := x^3 + 3 * x^2 + d * x + 20

theorem factor_theorem_for_Q :
  (x : ℝ) → (Q x = 0) → (x = 4) → d = -33 :=
by
  intro x Q4 hx
  sorry

end factor_theorem_for_Q_l375_375811


namespace sin_alpha_minus_beta_eq_cos_beta_eq_l375_375557

-- Given conditions
variables {α β : ℝ}
axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom sin_α_eq : sin α = 3 / 5
axiom tan_alpha_minus_beta_eq : tan (α - β) = -1 / 3

-- Desired results
theorem sin_alpha_minus_beta_eq : sin (α - β) = - (√10) / 10 :=
by {
  sorry
}

theorem cos_beta_eq : cos β = 9 * (√10) / 50 :=
by {
  sorry
}

end sin_alpha_minus_beta_eq_cos_beta_eq_l375_375557


namespace smallest_four_digit_equiv_mod_8_l375_375805

theorem smallest_four_digit_equiv_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 :=
by
  -- We state the assumptions and final goal
  use 1003
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  · refl
  sorry

end smallest_four_digit_equiv_mod_8_l375_375805


namespace unique_valid_placements_l375_375091

-- Define the numbers and operators to be used
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def operators : List (ℕ → ℕ → ℕ) := [(+), (-), (*), (/)]

-- Assume valid placements for numbers and operators forming a quadrilateral
-- such that the four resulting equations are valid.
def isValidPlacement (nums : List ℕ) (ops : List (ℕ → ℕ → ℕ)) : Prop :=
  -- This predicate checks if the placement of numbers and operators is valid.
  -- Note: The exact conditions for the quadrilateral to be valid need to be specified.
  sorry

-- The main statement that needs to be proved
theorem unique_valid_placements : ∃ n : ℕ, n = 4 ∧ 
  (∃ placements : List (List ℕ × List (ℕ → ℕ → ℕ)), 
    length placements = n ∧
    ∀ p ∈ placements, isValidPlacement (p.fst) (p.snd)) :=
begin
  sorry  -- Proof goes here
end

end unique_valid_placements_l375_375091


namespace strawberries_count_l375_375251

def strawberries_total (J M Z : ℕ) : ℕ :=
  J + M + Z

theorem strawberries_count (J M Z : ℕ) (h1 : J + M = 350) (h2 : M + Z = 250) (h3 : Z = 200) : 
  strawberries_total J M Z = 550 :=
by
  sorry

end strawberries_count_l375_375251


namespace gcd_lcm_product_360_l375_375384

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375384


namespace willam_percentage_taxable_land_l375_375970

def total_tax_collected : ℝ := 3840
def willam_tax_paid : ℝ := 500

theorem willam_percentage_taxable_land :
  (willam_tax_paid / total_tax_collected) * 100 ≈ 13.02 :=
by
  -- This will be left as sorry for now as no proof is needed in the statement
  sorry

end willam_percentage_taxable_land_l375_375970


namespace no_square_divisors_of_n_l375_375254

theorem no_square_divisors_of_n (n : ℕ) (f : ℕ → ℤ)
  (h1 : ∀ d, d ∣ n → f (n) = (-1)^d * d)
  (h2 : ∃ k : ℕ, f(n) = 2^k) :
  ∀ m : ℕ, m > 1 → ¬ (m^2 ∣ n) :=
by
  sorry

end no_square_divisors_of_n_l375_375254


namespace expected_draws_given_no_ugly_marble_l375_375839

-- Defining the conditions
def total_marbles : ℕ := 20
def blue_marbles : ℕ := 9
def ugly_marbles : ℕ := 10
def special_marble : ℕ := 1

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_ugly : ℚ := ugly_marbles / total_marbles
def probability_special : ℚ := special_marble / total_marbles

-- Expected value calculation given the conditions
theorem expected_draws_given_no_ugly_marble :
  (∃ (e : ℚ), e = 20 / 11) :=
begin
  use (20 / 11),
  sorry
end

end expected_draws_given_no_ugly_marble_l375_375839


namespace concurrent_LQ_MR_NP_l375_375439

variable (A B C D E F L M N P Q R : Type)
variable [HexagonInscribedCircle A B C D E F L M N P Q R]

theorem concurrent_LQ_MR_NP :
  concurrent (line_through L Q) (line_through M R) (line_through N P) :=
sorry

end concurrent_LQ_MR_NP_l375_375439


namespace circle_equation_line_intersection_l375_375217

theorem circle_equation (P : ℝ × ℝ) (C : set (ℝ × ℝ))
  (hF1 : P = (-1, 0))
  (hF2 : P = (1, 0))
  (hRatio : ∀ x y : ℝ, (real.sqrt ((x + 1)^2 + y^2) / real.sqrt ((x - 1)^2 + y^2)) = real.sqrt(2) / 2) :
  (∀ x y : ℝ, ((x + 3)^2 + y^2 = 8) ↔ (x, y) ∈ C) :=
by
  sorry

theorem line_intersection (C'' : set (ℝ × ℝ))
  (hC'' : ∀ x y : ℝ, (x^2 + (y + 3)^2 = 8) ↔ (x, y) ∈ C'')
  (hLine : ∀ m : ℝ, ∃ A B : ℝ × ℝ, (y = x + m - 3))
  (hArea : ∀ m : ℝ, 1 / 2 * (abs (m - 3)) / real.sqrt(2) * 2 * real.sqrt(8 - ((abs (m - 3)) / real.sqrt(2))^2) = real.sqrt(7)) :
  m = 3 ± real.sqrt(2) ∨ m = 3 ± real.sqrt(14) :=
by
  sorry

end circle_equation_line_intersection_l375_375217


namespace trapezoid_BC_squared_l375_375636

open Real

variables {A B C D : Point}
variables (a b c d : ℝ)
variables (AB : A.distanceTo B = sqrt 11) (AD : A.distanceTo D = sqrt 1001)
variables (BC_perp_AB : is_perpendicular (lineThrough B C) (lineThrough A B))
variables (BC_perp_CD : is_perpendicular (lineThrough B C) (lineThrough C D))
variables (AC_perp_BD : is_perpendicular (lineThrough A C) (lineThrough B D))

theorem trapezoid_BC_squared :
  A.distanceTo C = a → B.distanceTo C = b → C.distanceTo D = d →
  BC_perp_AB → BC_perp_CD → AC_perp_BD → 
  (a^2 + d^2 = 1001) → (a^2 + b^2 = 11) → 
  b^2 = 110 :=
sorry

end trapezoid_BC_squared_l375_375636


namespace gcd_lcm_product_360_l375_375381

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375381


namespace probability_point_in_external_spheres_l375_375858

noncomputable def inscribed_radius (R : ℝ) : ℝ := R / 3 
noncomputable def external_radius (r : ℝ) : ℝ := 1.5 * r
noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def volume_external_spheres (r : ℝ) : ℝ := 4 * volume_sphere (external_radius r)
noncomputable def volume_circumscribed_sphere (R : ℝ) : ℝ := volume_sphere R

theorem probability_point_in_external_spheres (R : ℝ) :
  let r := inscribed_radius R in
  volume_external_spheres r / volume_circumscribed_sphere R = 0.5 :=
by sorry

end probability_point_in_external_spheres_l375_375858


namespace sine_shift_l375_375780

theorem sine_shift (x : ℝ) : sin(2 * x - π / 2) = sin(2 * (x - π / 4)) :=
by sorry

end sine_shift_l375_375780


namespace survey_response_total_l375_375277

theorem survey_response_total
  (X Y Z : ℕ)
  (h_ratio : X / 4 = Y / 2 ∧ X / 4 = Z)
  (h_X : X = 200) :
  X + Y + Z = 350 :=
sorry

end survey_response_total_l375_375277


namespace option_d_correct_l375_375532

-- Definitions
variables {Point Line Plane : Type}
variables (a b : Line) (α β γ : Plane)

-- Conditions for proposition D
variables (h1 : α ∥ β)
variables (h2 : ∃ l : Line, α ∩ γ = l ∧ l = a)
variables (h3 : ∃ l : Line, β ∩ γ = l ∧ l = b)

theorem option_d_correct : a ∥ b :=
sorry

end option_d_correct_l375_375532


namespace modulus_of_i_mul_1_plus_i_l375_375140

def imaginary_unit : ℂ := complex.I 

def modulus_of_complex_number (z : ℂ) : ℝ :=
complex.abs z

theorem modulus_of_i_mul_1_plus_i : modulus_of_complex_number (imaginary_unit * (1 + imaginary_unit)) = 1 :=
by
  sorry

end modulus_of_i_mul_1_plus_i_l375_375140


namespace fraction_to_decimal_l375_375041

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375041


namespace proof_sunrise_sunset_speed_l375_375484

def degrees (d m: ℝ): ℝ := d + m/60

def latitude : ℝ := degrees 46 22
def declination : ℝ := degrees 19 52
def earth_radius : ℝ := 6.371 * 10^6

noncomputable def tan_deg (x: ℝ) : ℝ := Real.tan (x * Real.pi / 180)
noncomputable def cos_deg (x: ℝ) : ℝ := Real.cos (x * Real.pi / 180)

noncomputable def time_sunrise_sunset_given_latitude_declination 
  (latitude declination : ℝ) : (ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ) :=
  let tan_lat := tan_deg latitude
  let tan_dec := tan_deg declination
  let cos_omega := - tan_lat * tan_dec
  let omega := Real.arccos cos_omega * 180 / Real.pi
  let time_hour := Int.floor (omega / 15)
  let time_minute := Int.floor ((omega / 15 - time_hour) * 60)
  let time_second := Int.floor ((((omega / 15 - time_hour) * 60) - time_minute) * 60)
  let sunrise := (12 - time_hour - 1, 60 - time_minute, 60 - time_second)
  let sunset := (12 + time_hour, time_minute, time_second)
  ((sunrise.1.to_nat, sunrise.2.to_nat, sunrise.3.to_nat),
   (sunset.1.to_nat, sunset.2.to_nat, sunset.3.to_nat))

noncomputable def speed_due_to_earth_rotation :: (earth_radius latitude : ℝ) : ℝ :=
  2 * Real.pi * earth_radius * cos_deg latitude / 86400

theorem proof_sunrise_sunset_speed : 
  time_sunrise_sunset_given_latitude_declination latitude declination = 
    ((4, 30, 55), (19, 29, 5)) ∧ speed_due_to_earth_rotation earth_radius latitude = 319.5 :=
  by 
  sorry

end proof_sunrise_sunset_speed_l375_375484


namespace chromium_percentage_l375_375621

theorem chromium_percentage (x : ℝ) :
  let chromium1 := (12 / 100) * 20,
      chromium2 := (x / 100) * 35,
      total_chromium := (9.454545454545453 / 100) * 55,
      total_mass := 20 + 35 in
  chromium1 + chromium2 = total_chromium → x = 8 :=
by
  intro h
  sorry

end chromium_percentage_l375_375621


namespace sqrt_two_irrational_l375_375284

theorem sqrt_two_irrational :
  ¬ ∃ (p q : ℕ), p ≠ 0 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (↑q / ↑p) ^ 2 = (2:ℝ) :=
sorry

end sqrt_two_irrational_l375_375284


namespace fraction_goal_l375_375260

variable {a : ℕ → ℚ} (S : ℕ → ℚ)

-- Define the arithmetic sequence condition
definition arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n : ℕ, a (n+1) = a n + d

-- Define the sum of first n terms S_n
def S_n (n : ℕ) : ℚ := (finset.range n).sum a

-- Given conditions
variable (h1 : S_n 5 = 3 * (a 2 + a 8))
variable (h2 : arithmetic_sequence a)

-- Proof goal
theorem fraction_goal : ((a 5) / (a 3)) = 5 / 6 := by
  sorry

end fraction_goal_l375_375260


namespace cos_alpha_plus_beta_equals_neg_one_l375_375551

variables (α β : ℝ)

-- Conditions of the problem
axiom h1 : 0 < β ∧ β < π / 2 ∧ π / 2 < α ∧ α < π
axiom h2 : cos (α - β / 2) = - sqrt 2 / 2
axiom h3 : sin (α / 2 - β) = sqrt 2 / 2

-- Goal to prove
theorem cos_alpha_plus_beta_equals_neg_one : cos (α + β) = -1 :=
by
  sorry

end cos_alpha_plus_beta_equals_neg_one_l375_375551


namespace min_changes_to_unequal_sums_l375_375610

theorem min_changes_to_unequal_sums (n : ℕ) (table : ℕ → ℕ → ℤ)
    (equal_row_sum : ∀ i j, (∑ k in Finset.range n, table i k) = (∑ k in Finset.range n, table j k))
    (equal_col_sum : ∀ k l, (∑ i in Finset.range n, table i k) = (∑ i in Finset.range n, table i l))
    (rows : Fin)
    (cols : Fin)
    : n = 13 → 
      ∃ m ≥ 17, ∀ tab : ℕ → ℕ → ℤ, 
        ( ∀ i, ∑ k in Finset.range n, table i k ≠ ∑ k in Finset.range n, tab i k) ∧ 
        ( ∀ k, ∑ x in Finset.range n, table x k ≠ ∑ x in Finset.range n, tab x k) :=
by sorry

end min_changes_to_unequal_sums_l375_375610


namespace fraction_to_decimal_l375_375018

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375018


namespace bounded_jumps_l375_375849

/-- The initial conditions -/
variables (M : Point) (d : ℝ)
  (non_integer_coords : (¬ ∃ n : ℤ, M.x = n) ∧ (¬ ∃ n : ℤ, M.y = n))
  (initial_distance: distance(M, center(0, 0, 1, 1)) = d)
/-- The definition of jump_condition that describes the symmetrical jump -/
def jump (current : Point) : Point :=
  let nearest_corner := find_nearest_corner(current, set={(0,0), (0,1), (1,0), (1,1)}) in
  symmetric_point(current, nearest_corner)

/-- Prove that the distance from the center never exceeds 10d -/
theorem bounded_jumps :
  ∀ N, (distance ((jump^[N]) M, center(0, 0, 1, 1)) ≤ 10 * d) :=
  sorry

end bounded_jumps_l375_375849


namespace solveRealInequality_l375_375513

theorem solveRealInequality (x : ℝ) (hx : 0 < x) : x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry -- proof to be filled in

end solveRealInequality_l375_375513


namespace fraction_equals_decimal_l375_375077

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375077


namespace find_divisor_l375_375823

theorem find_divisor
  (n : ℕ) (h1 : n > 0)
  (h2 : (n + 1) % 6 = 4)
  (h3 : ∃ d : ℕ, n % d = 1) :
  ∃ d : ℕ, (n % d = 1) ∧ d = 2 :=
by
  sorry

end find_divisor_l375_375823


namespace probability_of_two_queens_or_at_least_one_jack_l375_375588

/-- There are 4 Jacks, 4 Queens, and 52 total cards in a standard deck. -/
namespace CardProbability

def total_cards := 52
def num_jacks := 4
def num_queens := 4

def prob_two_queens := (num_queens / total_cards) * ((num_queens - 1) / (total_cards - 1))
def prob_at_least_one_jack := 
  2 * (num_jacks / total_cards) * ((total_cards - num_jacks) / (total_cards - 1)) +
  (num_jacks / total_cards) * ((num_jacks - 1) / (total_cards - 1))

def prob_either_two_queens_or_at_least_one_jack := prob_two_queens + prob_at_least_one_jack

theorem probability_of_two_queens_or_at_least_one_jack : 
  prob_either_two_queens_or_at_least_one_jack = 2 / 13 :=
by
  sorry

end CardProbability

end probability_of_two_queens_or_at_least_one_jack_l375_375588


namespace expression_equals_three_l375_375115

theorem expression_equals_three (x : ℚ) :
  let a := 2010 * x + 2010
  let b := 2010 * x + 2011
  let c := 2010 * x + 2012
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 := 
by
  let a := 2010 * x + 2010
  let b := 2010 * x + 2011
  let c := 2010 * x + 2012
  calc
    a^2 + b^2 + c^2 - a * b - b * c - c * a = sorry

end expression_equals_three_l375_375115


namespace fraction_to_decimal_l375_375025

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375025


namespace inclination_l375_375236

-- Define the inclination angle alpha and the point M(-2, -4)
variable (α : ℝ) (M : ℝ × ℝ := (-2, -4))

-- Define the polar equation of the curve C
def polar_equation (θ ρ : ℝ) : Prop :=
  ρ * (sin θ) ^ 2 = 2 * cos θ

-- Define the Cartesian equation of the curve C
def cartesian_equation (x y : ℝ) : Prop :=
  y ^ 2 = 2 * x

-- Define the parametric equation of the line l 
def parametric_line (t : ℝ) : ℝ × ℝ :=
  let x := -2 + t * cos α
  let y := -4 + t * sin α
  (x, y)

-- Hypothesis that the product of distances |MA| * |MB| is 40
def distances_product (t1 t2 : ℝ) : Prop :=
  let MA := t1
  let MB := t2
  abs (MA * MB) = 40

-- The theorem to prove that α = π / 4
theorem inclination (HA HB : ℝ) : 
  polar_equation θ ρ ∧ cartesian_equation x y ∧ distances_product HA HB → α = π / 4 :=
by
  sorry

end inclination_l375_375236


namespace best_fit_model_l375_375501

def R_squared_values : Type :=
  ℝ × ℝ × ℝ × ℝ

def model_R_squared (model: ℕ) (values: R_squared_values) : ℝ :=
  match model with
  | 1 => values.1
  | 2 => values.2
  | 3 => values.3
  | 4 => values.4
  | _ => 0 -- Assume models are 1 to 4 only

theorem best_fit_model 
  (R_values : R_squared_values)
  (h1 : model_R_squared 1 R_values = 0.86)
  (h2 : model_R_squared 2 R_values = 0.68)
  (h3 : model_R_squared 3 R_values = 0.88)
  (h4 : model_R_squared 4 R_values = 0.66) :
  ∃ (m : ℕ), m = 3 ∧ ∀ (i : ℕ), model_R_squared i R_values ≤ model_R_squared m R_values :=
by
  sorry

end best_fit_model_l375_375501


namespace sum_of_prime_h_l375_375988

def h (n : ℕ) := n^4 - 380 * n^2 + 600

theorem sum_of_prime_h (S : Finset ℕ) (hS : S = { n | Nat.Prime (h n) }) :
  S.sum h = 0 :=
by
  sorry

end sum_of_prime_h_l375_375988


namespace symmetric_point_on_side_l375_375645

def triangle (A B C : Type) := ∃ (a b c : ℕ), a + b + c = 180

noncomputable def angle_bisector (A B C B1 C1 : Type) :=
  ∃ (α β : ℕ), α = 60 ∧ β = 90

theorem symmetric_point_on_side
    {A B C B1 C1 I : Type}
    (h1 : triangle A B C)
    (h2 : angle_bisector A B C B1 C1)
    (h3 : angle_bisector A C B C1 B1) :
    (∃ K : Type, ¬ collinear K A I) → on_side K BC :=
sorry

end symmetric_point_on_side_l375_375645


namespace restore_original_problem_l375_375908

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375908


namespace gcd_possible_values_count_l375_375397

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375397


namespace infinite_sequence_with_conditions_l375_375646

theorem infinite_sequence_with_conditions (a_0 : ℕ) (h1 : a_0 > 0) (h2 : a_0 ≤ 17526) (h3 : a_0 % 2 = 0) :
  ∃ (f : ℕ → ℕ), (∀ i, f i ∈ ℕ) ∧ set.infinite {i : ℕ | f i ∈ ℕ} :=
sorry

end infinite_sequence_with_conditions_l375_375646


namespace decreasing_power_function_l375_375742

theorem decreasing_power_function (n : ℝ) (f : ℝ → ℝ) 
    (h : ∀ x > 0, f x = (n^2 - n - 1) * x^n) 
    (h_decreasing : ∀ x > 0, f x > f (x + 1)) : n = -1 :=
sorry

end decreasing_power_function_l375_375742


namespace gcd_values_count_l375_375366

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375366


namespace tiles_needed_l375_375857

theorem tiles_needed (room_length room_width tile_length tile_width : ℝ) (h1 : room_length = 10) (h2 : room_width = 13) (h3 : tile_length = 0.5) (h4 : tile_width = 0.75) : 
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  let num_tiles_required := Real.ceil (room_area / tile_area)
  num_tiles_required = 347 :=
by
  -- Sorry placeholder for the proof
  sorry

end tiles_needed_l375_375857


namespace hyperbola_eccentricity_l375_375738

theorem hyperbola_eccentricity (a b e : ℝ) (heq : a = 1 ∧ b = 2 ∧ e = sqrt 5):
    x^2 - (y^2 / b^2) = 1 -> e = sqrt 5 := 
sorry

end hyperbola_eccentricity_l375_375738


namespace gcd_values_count_l375_375370

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375370


namespace probability_both_selected_l375_375827

/- 
Problem statement: Given that the probability of selection of Ram is 5/7 and that of Ravi is 1/5,
prove that the probability that both Ram and Ravi are selected is 1/7.
-/

theorem probability_both_selected (pRam : ℚ) (pRavi : ℚ) (hRam : pRam = 5 / 7) (hRavi : pRavi = 1 / 5) :
  (pRam * pRavi) = 1 / 7 :=
by
  sorry

end probability_both_selected_l375_375827


namespace b_n_formula_c_sequence_sum_l375_375147

-- Definitions
def a (n : ℕ) : ℕ := 2 * n + 1
def b (n : ℕ) : ℝ := (1 / 2) ^ (n - 1)
def c (n : ℕ) : ℝ := 1 / (a n * a (n + 1))
def S (n : ℕ) : ℝ := (Finset.range n).sum (fun k => c (k + 1))

-- Problem statements
theorem b_n_formula (n : ℕ) (h : 1 ≤ n) : a n * b (n + 1) - b (n + 1) = n * b n :=
sorry

theorem c_sequence_sum (n : ℕ) : S n = n / (6 * n + 9) :=
sorry

end b_n_formula_c_sequence_sum_l375_375147


namespace fraction_to_decimal_l375_375088

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375088


namespace doritos_in_each_pile_l375_375688

theorem doritos_in_each_pile (total_chips : ℕ) (quarter_of_chips : ℕ) (piles : ℕ) (chips_per_pile : ℕ) 
  (h1 : total_chips = 80) 
  (h2 : quarter_of_chips = total_chips / 4) 
  (h3 : piles = 4)
  (h4 : chips_per_pile = quarter_of_chips / piles) : 
  chips_per_pile = 5 :=
by 
  rw [h1, h2, h3, h4]
  rw nat.div_self;
  sorry

end doritos_in_each_pile_l375_375688


namespace angle_of_inclination_l375_375800

theorem angle_of_inclination (x y : ℝ) (α : ℝ) (line_eq : 2*x + y - 1 = 0) : α = π - arctan 2 :=
by
  sorry

end angle_of_inclination_l375_375800


namespace find_k_l375_375168

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l375_375168


namespace polynomial_root_omega_count_l375_375586

-- Definition of the problem as a Lean statement
theorem polynomial_root_omega_count :
  let ω := complex.exp (2 * real.pi * complex.I / 3) in
  let P := λ (a b c d : ℝ), (λ x : ℂ, (x^5 : ℂ) + (a : ℂ) * x^4 + (b : ℂ) * x^3 + (c : ℂ) * x^2 + (d : ℂ) * x + 2020) in 
  (∃ a b c d : ℝ, ∀ r : ℂ, P a b c d r = 0 → P a b c d (ω * r) = 0) →
  ∃! n : ℕ, n = 2 :=
sorry

end polynomial_root_omega_count_l375_375586


namespace find_pq_l375_375203

noncomputable def C (x : ℝ) : ℝ :=
  x^3 / (1 - x)

theorem find_pq : ∃ (p q : ℚ), r ∈ set.Icc (-3 : ℝ) (3 / 2) ∧ r ≠ 1 ∧ 
                            (root : ℝ) (H : is_root (2 * X^3 + 3 * X - 7) root) ∧ 
                            root = p * (inverse (λ y, C y − q)) q ∧ 
                            (p, q) = (7/3 : ℚ, 27/98 : ℚ) :=
by 
  sorry

end find_pq_l375_375203


namespace store_loss_14_yuan_l375_375468

noncomputable def suit_profit_or_loss : ℤ :=
  let p := 168 in
  let x := p / 1.2 in
  let y := p / 0.8 in
  let profit := p - x in
  let loss := y - p in
  loss - profit

theorem store_loss_14_yuan (p : ℕ) (h₁ : p = 168) :
  suit_profit_or_loss = 14 := by
  -- Provide the proof here
  sorry

end store_loss_14_yuan_l375_375468


namespace minimize_elements_in_set_l375_375352

theorem minimize_elements_in_set (k : ℝ) :
  (∀ x : ℤ, (k * ↑x - k^2 - 6) * (↑x - 4) > 0 → (↑x ∈ Set.Ioo (k + 6 / k) 4)) →
  Set.Ioo (-3) (-2) k :=
begin
  sorry
end

end minimize_elements_in_set_l375_375352


namespace learning_time_at_90_l375_375572

noncomputable def learning_curve (N : ℝ) : ℝ := -144 * Real.log10 (1 - N / 100)

theorem learning_time_at_90 :
  learning_curve 90 = 144 :=
by
  sorry

end learning_time_at_90_l375_375572


namespace find_p_q_sum_l375_375448

-- Define the number of trees
def pine_trees := 2
def cedar_trees := 3
def fir_trees := 4

-- Total number of trees
def total_trees := pine_trees + cedar_trees + fir_trees

-- Number of ways to arrange the 9 trees
def total_arrangements := Nat.choose total_trees fir_trees

-- Number of ways to place fir trees so no two are adjacent
def valid_arrangements := Nat.choose (pine_trees + cedar_trees + 1) fir_trees

-- Desired probability in its simplest form
def probability := valid_arrangements / total_arrangements

-- Denominator and numerator of the simplified fraction
def num := 5
def den := 42

-- Statement to prove that the probability is 5/42
theorem find_p_q_sum : (num + den) = 47 := by
  sorry

end find_p_q_sum_l375_375448


namespace triangle_BC_DC_ratio_l375_375244

noncomputable def triangle_ratio (A B C D P Q M N : Type) [inner_product_space ℝ Type]
  (angle_B : real.angle) (angle_C : real.angle)
  (h_angle_B : angle_B = real.angle.pi_div_two / 2)
  (h_angle_C : angle_C = real.angle.pi / 6) :
  real :=
  let BC : real := sorry in   -- Length of side BC
  let DC : real := sorry in   -- Length of side DC
  BC / DC

theorem triangle_BC_DC_ratio 
  (A B C D P Q M N : Type) [inner_product_space ℝ Type]
  (angle_B : real.angle) (angle_C : real.angle)
  (h_angle_B : angle_B = real.angle.pi_div_two / 2)
  (h_angle_C : angle_C = real.angle.pi / 6)
  (BM CN : line_segment A B) (BM, CN = sorry) (intersect : meet BM CN -> P Q)
  (chord_intersect : PQ.intersect BC = D) :
  triangle_ratio A B C D P Q M N angle_B angle_C h_angle_B h_angle_C = 1 / real.sqrt 3 :=
  sorry

end triangle_BC_DC_ratio_l375_375244


namespace power_function_inequality_l375_375577

theorem power_function_inequality (m : ℕ) (h : m > 0)
  (h_point : (2 : ℝ) ^ (1 / (m ^ 2 + m)) = Real.sqrt 2) :
  m = 1 ∧ ∀ a : ℝ, 1 ≤ a ∧ a < (3 / 2) → 
  (2 - a : ℝ) ^ (1 / (m ^ 2 + m)) > (a - 1 : ℝ) ^ (1 / (m ^ 2 + m)) :=
by
  sorry

end power_function_inequality_l375_375577


namespace find_value_of_a_l375_375433

theorem find_value_of_a :
  ∃ a, 
  let xs := [1, 2, 3, 4, 5],
      ys := [2, 3, 7, 8, a],
      mean_x := (xs.sum / 5),
      mean_y := (ys.sum / 5),
      y_hat := 2.1 * mean_x - 0.3
  in mean_y = y_hat ∧ a = 10 := 
by
  sorry

end find_value_of_a_l375_375433


namespace circle_equation_with_given_radius_and_point_l375_375631

-- Definitions
def point_on_circle (x y : ℝ) (r : ℝ) := x^2 + y^2 = r^2

-- Theorem statement
theorem circle_equation_with_given_radius_and_point :
  ∀ (x y : ℝ), point_on_circle (-4) 0 4 → point_on_circle x y 4 :=
by 
  intros x y h 
  have h1 : (-4)^2 + 0^2 = 4^2 := h
  let hx := eq.refl 16
  existsi (hx)
  sorry

end circle_equation_with_given_radius_and_point_l375_375631


namespace sequence_starts_with_one_l375_375270

theorem sequence_starts_with_one :
  ∃ (n : ℕ), n ≤ 2007 ∧ ∀ (a : Fin 2007 → ℕ), 
  (∀ i, a i ∈ finset.range 1 2008) → 
  (∀ i j, i ≠ j → a i ≠ a j) →
  (
    let seq_op (s : List ℕ) := 
        if s.head = n then (s.take n).reverse ++ s.drop n else s
    in a 0 = 1 ∨ ∃ m, m > 0 ∧ seq_op^[m] (List.ofFn a) = 
    (List.ofFn a).update_nth 0 1
  )
:= sorry

end sequence_starts_with_one_l375_375270


namespace restore_original_problem_l375_375876

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375876


namespace fill_time_with_A_and_B_l375_375505

variables (a b c : ℝ) (V : ℝ)

-- The conditions
def condition1 : Prop := 1 / (a + b + c) = 2
def condition2 : Prop := 1 / (a + c) = 3
def condition3 : Prop := 1 / (b + c) = 4

-- The question to be proved
theorem fill_time_with_A_and_B (ha : condition1) (hb : condition2) (hc : condition3) : 
  1 / (a + b) = 2.4 :=
sorry

end fill_time_with_A_and_B_l375_375505


namespace find_k_l375_375169

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l375_375169


namespace quadratic_condition_range_l375_375758

noncomputable def quadratic_satisfies :=
  ∀ (f : ℝ → ℝ), (∀ x, f(2 + x) = f(2 - x)) ∧ (∃ a b c, a > 0 ∧ ∀ x, f(x) = a * x^2 + b * x + c) →
  ∀ x, f(1 - 2 * x^2) < f(1 + 2 * x - x^2) → (-2 < x ∧ x < 0)

theorem quadratic_condition_range :
  quadratic_satisfies := sorry

end quadratic_condition_range_l375_375758


namespace gcd_possible_values_count_l375_375392

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375392


namespace total_cost_is_2819_l375_375098

-- Define the lengths of the sides of the triangular section
def side1 : ℝ := 100
def side2 : ℝ := 150
def side3 : ℝ := 50

-- Define the radius of the circular section
def radius : ℝ := 30

-- Define the cost per meter for both sections
def cost_per_meter_triangle : ℝ := 5
def cost_per_meter_circle : ℝ := 7

-- Define the perimeter of the triangular section
def perimeter_triangle : ℝ := side1 + side2 + side3

-- Define the circumference of the circular section
def perimeter_circle : ℝ := 2 * Real.pi * radius

-- Define the total cost of fencing
def total_cost : ℝ := (perimeter_triangle * cost_per_meter_triangle) + (perimeter_circle * cost_per_meter_circle)

-- Prove that the total cost is approximately Rs. 2819
theorem total_cost_is_2819 : Real.abs (total_cost - 2819) < 1 :=
by
  -- including sorry to denote the proof steps are not required
  sorry

end total_cost_is_2819_l375_375098


namespace fraction_to_decimal_l375_375046

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375046


namespace value_of_f_2007_l375_375987

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f (x + 2008) = f (x + 2007) + f (x + 2009)
axiom condition2 : f 1 = Real.log (3 / 2)
axiom condition3 : f 2 = Real.log 15

theorem value_of_f_2007 : f 2007 = 1 :=
begin
  sorry,
end

end value_of_f_2007_l375_375987


namespace bob_favorite_number_is_correct_l375_375940

def bob_favorite_number : ℕ :=
  99

theorem bob_favorite_number_is_correct :
  50 < bob_favorite_number ∧
  bob_favorite_number < 100 ∧
  bob_favorite_number % 11 = 0 ∧
  bob_favorite_number % 2 ≠ 0 ∧
  (bob_favorite_number / 10 + bob_favorite_number % 10) % 3 = 0 :=
by
  sorry

end bob_favorite_number_is_correct_l375_375940


namespace min_value_quadratic_l375_375976

theorem min_value_quadratic :
  ∃ (x y : ℝ), (∀ (a b : ℝ), (3*a^2 + 4*a*b + 2*b^2 - 6*a - 8*b + 6 ≥ 0)) ∧ 
  (3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6 = 0) := 
sorry

end min_value_quadratic_l375_375976


namespace percentage_of_students_owning_cats_l375_375228

theorem percentage_of_students_owning_cats (dogs cats total : ℕ) (h_dogs : dogs = 45) (h_cats : cats = 75) (h_total : total = 500) : 
  (cats / total) * 100 = 15 :=
by
  sorry

end percentage_of_students_owning_cats_l375_375228


namespace red_white_probability_l375_375219

theorem red_white_probability :
  (∀ (total red white : ℕ)
      (h_total: total = 5)
      (h_red: red = 2)
      (h_white: white = 3),
     ∃ (total_outcomes favorable_outcomes : ℕ),
       total_outcomes = (total * (total - 1)) / 2 ∧
       favorable_outcomes = red * white ∧
       (favorable_outcomes : ℚ) / total_outcomes = 3 / 5) :=
by {
  intros total red white h_total h_red h_white,
  use [10, 6],
  simp [h_total, h_red, h_white],
  sorry
}

end red_white_probability_l375_375219


namespace find_unique_positive_integer_pair_l375_375990

theorem find_unique_positive_integer_pair :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ c > b^2 ∧ b > c^2 :=
sorry

end find_unique_positive_integer_pair_l375_375990


namespace Balint_claim_impossible_l375_375942

-- Declare the lengths of the ladders and the vertical projection distance
def AC : ℝ := 3
def BD : ℝ := 2
def E_proj : ℝ := 1

-- State the problem conditions and what we need to prove
theorem Balint_claim_impossible (h1 : AC = 3) (h2 : BD = 2) (h3 : E_proj = 1) :
  False :=
  sorry

end Balint_claim_impossible_l375_375942


namespace initial_walking_speed_l375_375457

variable (v : ℝ)

theorem initial_walking_speed :
  (13.5 / v - 13.5 / 6 = 27 / 60) → v = 5 :=
by
  intro h
  sorry

end initial_walking_speed_l375_375457


namespace domain_of_f_l375_375317

-- Define the function and condition
def f (x : ℝ) : ℝ := real.sqrt ((x - 1)^2 * (x + 1) / (x - 2))

-- The statement we need to prove
theorem domain_of_f :
  {x : ℝ | (x - 1)^2 * (x + 1) / (x - 2) ≥ 0} = 
  {x : ℝ | x > 2} ∪ {x : ℝ | x ≤ -1} ∪ {1} :=
by
  sorry

end domain_of_f_l375_375317


namespace gcd_values_count_l375_375400

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375400


namespace part_a_part_b_l375_375256

variable {R : Type*} [Field R] [CharZero R] [NontriviallyOrderedRing R]

theorem part_a (x y z : R) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x * y ∈ ℚ) (h2 : y * z ∈ ℚ) (h3 : z * x ∈ ℚ) :
  (x^2 + y^2 + z^2) ∈ ℚ :=
sorry 

theorem part_b (x y z : R) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x * y ∈ ℚ) (h2 : y * z ∈ ℚ) (h3 : z * x ∈ ℚ) (h4 : (x^3 + y^3 + z^3) ∈ ℚ) :
  x ∈ ℚ ∧ y ∈ ℚ ∧ z ∈ ℚ := 
sorry

end part_a_part_b_l375_375256


namespace fraction_to_decimal_l375_375065

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375065


namespace cole_drive_time_to_work_l375_375819

theorem cole_drive_time_to_work :
  ∀ (D : ℝ),
    (D / 80 + D / 120 = 3) → (D / 80 * 60 = 108) :=
by
  intro D h
  sorry

end cole_drive_time_to_work_l375_375819


namespace manufacturer_profit_percentage_l375_375452

-- Definitions and conditions
def cost_price_manufacturer : ℝ := 17
def selling_price_retailer : ℝ := 30.09
def profit_percent_retailer : ℝ := 25
def profit_percent_wholesaler : ℝ := 20

-- Computed intermediate results (used in the equivalent proof problem, but not directly in the definitions for Lean)
def cost_price_retailer := selling_price_retailer / (1 + profit_percent_retailer / 100)
def cost_price_wholesaler := cost_price_retailer / (1 + profit_percent_wholesaler / 100)
def selling_price_manufacturer := cost_price_wholesaler
def profit_manufacturer := selling_price_manufacturer - cost_price_manufacturer

-- The profit percentage for the manufacturer we want to prove
def profit_percent_manufacturer := (profit_manufacturer / cost_price_manufacturer) * 100

-- The target statement to prove
theorem manufacturer_profit_percentage :
  profit_percent_manufacturer = 18 := by
  sorry

end manufacturer_profit_percentage_l375_375452


namespace concurrent_bisectors_l375_375318

theorem concurrent_bisectors
  (ABC : Triangle ℝ)
  (D E F : Point ℝ)
  (X Y Z : Point ℝ)
  (incircle_touches : incircleTouches ABC D E F)
  (X_incenter_ae : incenter (triangle.mk A E F))
  (Y_incenter_bd : incenter (triangle.mk B F D))
  (Z_incenter_ce : incenter (triangle.mk C D E)) :
  concurrent (line D X) (line E Y) (line F Z) :=
sorry

end concurrent_bisectors_l375_375318


namespace triangle_bc_value_l375_375640

theorem triangle_bc_value (A B C: Type) [euclidean_geometry.bary A B C]
  (angle_B : ∠ B = 45)
  (AB : B.dist A = 100)
  (AC : C.dist A = 100 * real.sqrt 2) :
  B.dist C = 100 * real.sqrt ((5 : ℝ) + real.sqrt 2 * (real.sqrt 6 - real.sqrt 2)) := 
by sorry

end triangle_bc_value_l375_375640


namespace dot_product_a_b_l375_375114

-- Definitions of the conditions provided in the problem
def a : ℝ × ℝ := (2 * Real.sin (16 * Real.pi / 180), 2 * Real.sin (74 * Real.pi / 180))
def b : ℝ × ℝ

def magnitude (v : ℝ × ℝ) := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def vec_sub (u v : ℝ × ℝ) := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2
def angle_between (u v : ℝ × ℝ) := acos (dot_product u v / (magnitude u * magnitude v))

-- Use the given conditions to set up the scenarios
axiom assumption1 : magnitude (vec_sub a b) = 1
axiom assumption2 : angle_between a (vec_sub a b) = Real.pi / 3

-- Goal to prove
theorem dot_product_a_b : dot_product a b = 3 :=
by {
  sorry  -- The detailed proof steps go here
}

end dot_product_a_b_l375_375114


namespace find_k_l375_375182

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l375_375182


namespace problem_proof_l375_375411

noncomputable def does_not_uniquely_determine_scalene_triangle : Prop :=
  ∃ (α β R : ℝ) (T₁ T₂ : Triangle),
    α + β < 180 ∧
    Triangle.is_circumscribed T₁ R ∧
    Triangle.is_circumscribed T₂ R ∧
    T₁.angles = (α, β, 180 - α - β) ∧
    T₂.angles = (α, β, 180 - α - β) ∧
    T₁ ≠ T₂

theorem problem_proof : does_not_uniquely_determine_scalene_triangle := sorry

end problem_proof_l375_375411


namespace gcd_values_count_l375_375403

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375403


namespace find_k_l375_375190

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l375_375190


namespace problem_l375_375591

theorem problem (a b c : ℝ) (Ha : a > 0) (Hb : b > 0) (Hc : c > 0) : 
  (|a| / a + |b| / b + |c| / c - (abc / |abc|) = 2 ∨ |a| / a + |b| / b + |c| / c - (abc / |abc|) = -2) :=
by
  sorry

end problem_l375_375591


namespace consecutive_ints_prod_square_l375_375289

theorem consecutive_ints_prod_square (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
sorry

end consecutive_ints_prod_square_l375_375289


namespace range_of_a_plus_b_l375_375118

noncomputable def f (x : ℝ) : ℝ := abs (2 - x^2)

theorem range_of_a_plus_b (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : f a = f b) : 
  2 < a + b ∧ a + b < 2 * real.sqrt 2 :=
by 
  sorry

end range_of_a_plus_b_l375_375118


namespace certain_number_value_l375_375206

theorem certain_number_value (x : ℝ) (certain_number : ℝ) 
  (h1 : x = 0.25) 
  (h2 : 625^(-x) + 25^(-2 * x) + certain_number^(-4 * x) = 11) : 
  certain_number = 5 / 53 := 
sorry

end certain_number_value_l375_375206


namespace fraction_to_decimal_l375_375051

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l375_375051


namespace find_k_l375_375174

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l375_375174


namespace maximum_pairs_l375_375113

theorem maximum_pairs:
  ∃ k pairs, 
  (∀ i j, i ≠ j → Pair.fst (pairs i) ≠ Pair.fst (pairs j) ∧ Pair.fst (pairs i) ≠ Pair.snd (pairs j) ∧ Pair.snd (pairs i) ≠ Pair.snd (pairs j)) ∧
  (∀ i, Pair.fst (pairs i) < Pair.snd (pairs i)) ∧
  (∀ i, Pair.fst (pairs i) + Pair.snd (pairs i) ≤ 2017) ∧
  (∀ i j, i ≠ j → Pair.fst (pairs i) + Pair.snd (pairs i) ≠ Pair.fst (pairs j) + Pair.snd (pairs j)) ∧
  (∀ i, Pair.fst (pairs i) ∈ (Set.finRange 2017) ∧ Pair.snd (pairs i) ∈ (Set.finRange 2017)) ∧
  k ≤ 806 :=
sorry

end maximum_pairs_l375_375113


namespace solve_fractions_l375_375916

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375916


namespace minimum_blocks_needed_l375_375953

-- Definitions of conditions
def wall_length := 120 -- in feet
def wall_height := 10 -- in feet
def block_a_length := 3 -- in feet (3-foot block)
def block_b_length := 2 -- in feet (2-foot block)
def block_height := 1 -- height of each block in feet
def number_of_rows := 10

-- Theorem stating the smallest number of blocks needed
theorem minimum_blocks_needed : 
  (∀ row1 row2 row3 row4 row5 row6 row7 row8 row9 row10 : ℕ, 
  (number_of_blocks_required row1 row2 row3 row4 row5 row6 row7 row8 row9 row10 = 466)) :=
sorry

-- Definition to calculate the number of blocks required for all rows
noncomputable def number_of_blocks_required 
  (row1 row2 row3 row4 row5 row6 row7 row8 row9 row10 : ℕ) : ℕ := 
    row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8 + row9 + row10

-- Sorry proof placeholder to satisfy compilation

end minimum_blocks_needed_l375_375953


namespace petya_candies_110_l375_375697

noncomputable def number_of_candies_Petya_received (candies_masha: ℕ) : ℕ :=
let n := Nat.ceil (Real.sqrt (candies_masha + 20.5) - 1) in
n * (n + 1)

theorem petya_candies_110 (h : 101 = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21) :
  number_of_candies_Petya_received 101 = 110 :=
by
  calc number_of_candies_Petya_received 101 = 10 * (10 + 1) : by sorry
                                              ... = 110     : by sorry

end petya_candies_110_l375_375697


namespace find_second_integer_l375_375327

-- Definitions of the conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  ∃ a b c : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2 ∧ (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1)

def sum_first_and_third_is_156 (n : ℤ) : Prop :=
  let a := n - 2 in
  let c := n + 2 in
  a + c = 156

-- The question formally stated as a proof problem
theorem find_second_integer :
  ∀ n : ℤ, consecutive_odd_integers n ∧ sum_first_and_third_is_156 n → n = 78 :=
by
  intros
  sorry

end find_second_integer_l375_375327


namespace parallelogram_product_l375_375622

noncomputable def EF := 58
noncomputable def GH (x : ℚ) := 3 * x + 5
noncomputable def FG (y : ℚ) := 4 * y^3
noncomputable def HE := 24

theorem parallelogram_product (x y : ℚ) (hx : GH x = EF) (hy : FG y = HE) :
  x * y = 53 / 3 * Real.cbrt 6 :=
by
  sorry

end parallelogram_product_l375_375622


namespace quadratic_inequality_solution_set_l375_375326

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 3 / 2} :=
sorry

end quadratic_inequality_solution_set_l375_375326


namespace find_n_conditions_l375_375972

theorem find_n_conditions {n : ℕ} :
  (∃ k : ℕ, n = k + 2 * ⌊real.sqrt k⌋ + 2) ↔ (∀ y : ℕ+, n ≠ y ^ 2 - 1 ∧ n ≠ y ^ 2) :=
sorry

end find_n_conditions_l375_375972


namespace proof_of_Tn_l375_375263

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Let a_n be an increasing arithmetic sequence with common difference d
-- Conditions for a_n and S_n
axiom increasing_arithmetic_sequence (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d ∧ d > 0)

-- Sum of first five terms
axiom sum_of_first_five_terms (h1 : 5 * (a 3) = 85) : 
  S 5 = 85

-- Value of a_6
axiom value_of_a6 (h2 : a 6 = 7 * a 1) : 
  a 6 = 7 * a 1

-- General form of a_n
noncomputable def a_general_form : ℕ → ℝ :=
  λ n, let a1 := 5; let d := 6 in a1 + (n - 1) * d

-- Sum of first n terms S_n
noncomputable def sum_n_terms (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), a k

-- Definition of b_n
noncomputable def sequence_b (n : ℕ) : ℝ :=
  5 / (a n * a (n + 1))

-- Sum of first n terms of sequence b
noncomputable def sum_of_sequence_b (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, b k

-- Finally proving T_n = (n / (6n + 5))
theorem proof_of_Tn :
  T n = n / (6 * n + 5) :=
sorry

end proof_of_Tn_l375_375263


namespace exists_k_leq_n_l375_375528

noncomputable def arithmetic_mean (s : List ℝ) : ℝ :=
s.sum / s.length

theorem exists_k_leq_n (n : ℕ) (a : Fin n → ℝ) :
  let c := arithmetic_mean (List.map a (List.finRange n)) in
  ∃ k ≤ n, ∀ m, 1 ≤ m → m ≤ k → (∑ i in Finset.range m, a ⟨k - i - 1, _⟩) / m ≤ c :=
by
  sorry

end exists_k_leq_n_l375_375528


namespace parabola_relationship_l375_375628

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem parabola_relationship (a b m n t : ℝ) (ha : a ≠ 0)
  (h1 : 3 * a + b > 0) (h2 : a + b < 0)
  (hm : parabola a b (-3) = m)
  (hn : parabola a b 2 = n)
  (ht : parabola a b 4 = t) :
  n < t ∧ t < m :=
by
  sorry

end parabola_relationship_l375_375628


namespace no_integers_divisible_by_all_l375_375199

-- Define the list of divisors
def divisors : List ℕ := [2, 3, 4, 5, 7, 11]

-- Define the LCM function
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Calculate the LCM of the given divisors
def lcm_divisors : ℕ := lcm_list divisors

-- Define a predicate to check divisibility by all divisors
def is_divisible_by_all (n : ℕ) (ds : List ℕ) : Prop :=
  ds.all (λ d => n % d = 0)

-- Define the theorem to prove the number of integers between 1 and 1000 divisible by the given divisors
theorem no_integers_divisible_by_all :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1000 ∧ is_divisible_by_all n divisors) → False := by
  sorry

end no_integers_divisible_by_all_l375_375199


namespace gcd_values_count_l375_375365

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375365


namespace mixed_fraction_product_example_l375_375883

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375883


namespace solve_fractions_l375_375912

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375912


namespace age_ratio_l375_375932

noncomputable def ratio_of_ages (A M : ℕ) : ℕ × ℕ :=
if A = 30 ∧ (A + 15 + (M + 15)) / 2 = 50 then
  (A / Nat.gcd A M, M / Nat.gcd A M)
else
  (0, 0)

theorem age_ratio :
  (45 + (40 + 15)) / 2 = 50 → 30 = 3 * 10 ∧ 40 = 4 * 10 →
  ratio_of_ages 30 40 = (3, 4) :=
by
  sorry

end age_ratio_l375_375932


namespace sandy_marks_total_l375_375298

/-- Sandy gets 3 marks for each correct sum and loses 2 marks for each incorrect sum.
    Sandy attempts 30 sums and gets 23 correct sums. 
    Prove that Sandy obtained 55 marks in total. -/
theorem sandy_marks_total : 
  let correct_marks := 3 * 23 in
  let incorrect_marks := 2 * (30 - 23) in
  correct_marks - incorrect_marks = 55 :=
by
  let correct_marks := 3 * 23
  let incorrect_marks := 2 * (30 - 23)
  have h : correct_marks - incorrect_marks = 55
  sorry

end sandy_marks_total_l375_375298


namespace num_five_dollar_coins_l375_375344

theorem num_five_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by
  sorry -- Proof to be completed

end num_five_dollar_coins_l375_375344


namespace parallel_vectors_dot_product_l375_375581

-- Definitions according to conditions in the problem
def vec_a : (ℝ × ℝ) := (1, 3)
def vec_b (λ : ℝ) : (ℝ × ℝ) := (λ, -1)

-- Mathematical condition for vectors to be parallel
def parallel (a b : (ℝ × ℝ)) : Prop :=
∃ k : ℝ, a = (k * b.1, k * b.2)

-- Proof statement
theorem parallel_vectors_dot_product (λ : ℝ) (h : parallel vec_a (vec_b λ)) :
    vec_a.1 * (vec_b λ).1 + vec_a.2 * (vec_b λ).2 = -10 / 3 :=
sorry

end parallel_vectors_dot_product_l375_375581


namespace sequence_periodic_a2014_l375_375240

theorem sequence_periodic_a2014 (a : ℕ → ℚ) 
  (h1 : a 1 = -1/4) 
  (h2 : ∀ n > 1, a n = 1 - (1 / (a (n - 1)))) : 
  a 2014 = -1/4 :=
sorry

end sequence_periodic_a2014_l375_375240


namespace tetrahedron_volume_l375_375665

noncomputable def volume_of_tetrahedron (a S : ℝ) : ℝ := (1 / 3) * S * a

theorem tetrahedron_volume (ABCD : Type*) (a S : ℝ)
  (length_AB : a = dist (point A) (point B))
  (projection_area : S = projection_area_of_tetrahedron ABCD ⟂ AB) :
  volume_of_tetrahedron a S = (1 / 3) * S * a :=
by
  sorry

end tetrahedron_volume_l375_375665


namespace total_books_l375_375773

def shelves : ℕ := 150
def books_per_shelf : ℕ := 15

theorem total_books (shelves books_per_shelf : ℕ) : shelves * books_per_shelf = 2250 := by
  sorry

end total_books_l375_375773


namespace average_speed_for_entire_trip_l375_375413

-- Define the conditions
def distance1 : ℝ := 10
def speed1 : ℝ := 12
def distance2 : ℝ := 12
def speed2 : ℝ := 10
def totalDistance : ℝ := distance1 + distance2

-- Define the time taken for each part
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2
def totalTime : ℝ := time1 + time2

-- Define the expected average speed
def correctAverageSpeed : ℝ := 660 / 61

-- The theorem to state the problem 
theorem average_speed_for_entire_trip :
  (distance1 + distance2) / (time1 + time2) = 660 / 61 := 
by
  -- The proof is omitted
  sorry

end average_speed_for_entire_trip_l375_375413


namespace smallest_area_of_triangle_l375_375666

noncomputable def vec3 := ℝ × ℝ × ℝ

def A : vec3 := (-1, 1, 2)
def B : vec3 := (2, 3, 4)
def C (s : ℝ) : vec3 := (s, 2, 3)

def sub (v1 v2 : vec3) : vec3 := (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)
def cross (v1 v2 : vec3) : vec3 :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)
def norm (v : vec3) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def area_of_triangle (a b c : vec3) : ℝ :=
  (1/2) * norm (cross (sub b a) (sub c a))

theorem smallest_area_of_triangle : 
  ∃ s, area_of_triangle A B (C s) = real.sqrt 2 / 4 := 
sorry

end smallest_area_of_triangle_l375_375666


namespace number_of_unique_triangle_areas_l375_375003

theorem number_of_unique_triangle_areas :
  ∀ (G H I J K L : ℝ) (d₁ d₂ d₃ d₄ : ℝ),
    G ≠ H → H ≠ I → I ≠ J → G ≠ I → G ≠ J →
    H ≠ J →
    G - H = 1 → H - I = 1 → I - J = 2 →
    K - L = 2 →
    d₄ = abs d₃ →
    (d₁ = abs (K - G)) ∨ (d₂ = abs (L - G)) ∨ (d₁ = d₂) →
    ∃ (areas : ℕ), 
    areas = 3 :=
by sorry

end number_of_unique_triangle_areas_l375_375003


namespace mixed_fractions_product_l375_375866

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375866


namespace white_roses_per_bouquet_l375_375693

/-- Mrs. Dunbar needs to make 5 bouquets and 7 table decorations. -/
def number_of_bouquets : ℕ := 5
def number_of_table_decorations : ℕ := 7
/-- She uses 12 white roses in each table decoration. -/
def white_roses_per_table_decoration : ℕ := 12
/-- She needs a total of 109 white roses to complete all bouquets and table decorations. -/
def total_white_roses_needed : ℕ := 109

/-- Prove that the number of white roses used in each bouquet is 5. -/
theorem white_roses_per_bouquet : ∃ (white_roses_per_bouquet : ℕ),
  number_of_bouquets * white_roses_per_bouquet + number_of_table_decorations * white_roses_per_table_decoration = total_white_roses_needed
  ∧ white_roses_per_bouquet = 5 := 
by
  sorry

end white_roses_per_bouquet_l375_375693


namespace number_of_distinct_real_roots_l375_375144

theorem number_of_distinct_real_roots (f : ℝ → ℝ) (h : ∀ x, f x = |x| - (4 / x) - (3 * |x| / x)) : ∃ k, k = 1 :=
by
  sorry

end number_of_distinct_real_roots_l375_375144


namespace eagles_win_probability_l375_375308

open ProbabilityTheory

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

noncomputable def prob_binomial (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1-p)^(n-k)

noncomputable def success_probability (n : ℕ) (p : ℚ) (min_successes : ℕ) : ℚ :=
  ∑ k in finset.range (n+1), if k ≥ min_successes then prob_binomial n k p else 0

theorem eagles_win_probability : success_probability 5 0.5 3 = 1 / 2 := by
  sorry

end eagles_win_probability_l375_375308


namespace kate_needs_more_money_for_trip_l375_375661

theorem kate_needs_more_money_for_trip:
  let kate_money_base6 := 3 * 6^3 + 2 * 6^2 + 4 * 6^1 + 2 * 6^0
  let ticket_cost := 1000
  kate_money_base6 - ticket_cost = -254 :=
by
  -- Proving the theorem, steps will go here.
  sorry

end kate_needs_more_money_for_trip_l375_375661


namespace minimum_f_on_interval_l375_375124

def f (x : ℝ) : ℝ := if x ∈ Icc (- 1) 1 then abs x - 1 else sorry

theorem minimum_f_on_interval :
  (∀ x, f (x + 2) = f x / 2) →
  (∀ x ∈ Icc (-6) (-4), f x ≥ -8) →
  (∃ x ∈ Icc (-6) (-4), f x = -8) :=
begin
  assume h1 h2,
  sorry
end

end minimum_f_on_interval_l375_375124


namespace find_a_l375_375436

theorem find_a (a : ℝ) (x_values y_values : List ℝ)
  (h_y : ∀ x, List.getD y_values x 0 = 2.1 * List.getD x_values x 1 - 0.3) :
  a = 10 :=
by
  have h_mean_x : (1 + 2 + 3 + 4 + 5) / 5 = 3 := by norm_num
  have h_sum_y : (2 + 3 + 7 + 8 + a) / 5 = (2.1 * 3 - 0.3) := by sorry
  sorry

end find_a_l375_375436


namespace slope_PA1_range_l375_375574

open Real

noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 5) - (y^2 / 3) = 1

def A1 : ℝ × ℝ := (-sqrt 5, 0)
def A2 : ℝ × ℝ := (sqrt 5, 0)

theorem slope_PA1_range (x0 y0 : ℝ) (hx0 : x0 ≠ sqrt 5) (hx0n : x0 ≠ -sqrt 5)
  (hP : hyperbola x0 y0)
  (hk2 : ∀ k2 : ℝ, k2 ∈ Icc (-4) (-2) → k2 = y0 / (x0 - sqrt 5)) :
  ∀ k1 : ℝ, k1 = y0 / (x0 + sqrt 5) → k1 ∈ Icc (-3/10) (-3/20) :=
begin
  sorry
end

end slope_PA1_range_l375_375574


namespace value_of_f_at_1_over_16_l375_375743

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem value_of_f_at_1_over_16 (α : ℝ) (h : f 4 α = 2) : f (1 / 16) α = 1 / 4 :=
by
  sorry

end value_of_f_at_1_over_16_l375_375743


namespace area_EFGH_trapezoid_l375_375801

-- Define the type of vertices
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the trapezoid
def E : Point := { x := 1, y := 1 }
def F : Point := { x := 1, y := -3 }
def G : Point := { x := 5, y := 1 }
def H : Point := { x := 5, y := 7 }

-- Define a function to calculate the distance between two points
def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

-- Define the bases of the trapezoid
def EF : ℝ := distance E F
def GH : ℝ := distance G H

-- Define the height of the trapezoid
def height : ℝ := (G.x - E.x).abs

-- Define the area of the trapezoid using the trapezoidal area formula
def area_trapezoid (EF GH height : ℝ) : ℝ :=
  0.5 * (EF + GH) * height

-- The theorem to prove
theorem area_EFGH_trapezoid : area_trapezoid EF GH height = 20 := by
  sorry

end area_EFGH_trapezoid_l375_375801


namespace number_of_lines_l375_375279

-- Definitions and conditions
variables {P : Type} [plane : MetricSpace P]
variable (A B C : P)

-- Point collinearity definition
def collinear (A B C : P) : Prop := ∃ l : Line P, A ∈ l ∧ B ∈ l ∧ C ∈ l

-- The theorem statement
theorem number_of_lines (A B C : P) : ∃ n : ℕ, (collinear A B C → n = 1) ∧ (¬ collinear A B C → n = 3) :=
sorry  -- Proof would be here

end number_of_lines_l375_375279


namespace Jenny_walked_distance_l375_375659

theorem Jenny_walked_distance
  (distance_ran : ℝ)
  (ran_further : ℝ) :
  distance_ran = 0.6 → ran_further = 0.2 → 
  ∃ walked_distance : ℝ, walked_distance = 0.4 := 
by {
  intros h1 h2,
  use 0.4,
  sorry
}

end Jenny_walked_distance_l375_375659


namespace family_four_children_includes_at_least_one_boy_one_girl_l375_375935

-- Specification of the probability function
def prob_event (n : ℕ) (event : fin n → bool) : ℚ := 
  (Real.to_rat (Real.exp (- (Real.nat_to_real (nat.log2 n)))) : ℚ)

-- Predicate that checks if there is at least one boy and one girl in the list
def has_boy_and_girl (children : fin 4 → bool) : Prop :=
  ∃ i j, children i ≠ children j

theorem family_four_children_includes_at_least_one_boy_one_girl : 
  (∑ event in (finset.univ : finset (fin 4 → bool)), 
     if has_boy_and_girl event then prob_event 4 event else 0) = 7 / 8 :=
by
  sorry

end family_four_children_includes_at_least_one_boy_one_girl_l375_375935


namespace sphere_volume_l375_375855

theorem sphere_volume (A : ℝ) (d : ℝ) (V : ℝ) : 
    (A = 2 * Real.pi) →  -- Cross-sectional area is 2π cm²
    (d = 1) →            -- Distance from center to cross-section is 1 cm
    (V = 4 * Real.sqrt 3 * Real.pi) :=  -- Volume of sphere is 4√3 π cm³
by 
  intros hA hd
  sorry

end sphere_volume_l375_375855


namespace students_still_in_school_l375_375465

def total_students := 5000
def students_to_beach := total_students / 2
def remaining_after_beach := total_students - students_to_beach
def students_to_art_museum := remaining_after_beach / 3
def remaining_after_art_museum := remaining_after_beach - students_to_art_museum
def students_to_science_fair := remaining_after_art_museum / 4
def remaining_after_science_fair := remaining_after_art_museum - students_to_science_fair
def students_to_music_workshop := 200
def remaining_students := remaining_after_science_fair - students_to_music_workshop

theorem students_still_in_school : remaining_students = 1051 := by
  sorry

end students_still_in_school_l375_375465


namespace fraction_to_decimal_equiv_l375_375008

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375008


namespace gcd_lcm_product_360_l375_375379

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375379


namespace total_cable_cost_neighborhood_l375_375485

-- Define the number of east-west streets and their length
def ew_streets : ℕ := 18
def ew_length_per_street : ℕ := 2

-- Define the number of north-south streets and their length
def ns_streets : ℕ := 10
def ns_length_per_street : ℕ := 4

-- Define the cable requirements and cost
def cable_per_mile_of_street : ℕ := 5
def cable_cost_per_mile : ℕ := 2000

-- Calculate total length of east-west streets
def ew_total_length : ℕ := ew_streets * ew_length_per_street

-- Calculate total length of north-south streets
def ns_total_length : ℕ := ns_streets * ns_length_per_street

-- Calculate total length of all streets
def total_street_length : ℕ := ew_total_length + ns_total_length

-- Calculate total length of cable required
def total_cable_length : ℕ := total_street_length * cable_per_mile_of_street

-- Calculate total cost of the cable
def total_cost : ℕ := total_cable_length * cable_cost_per_mile

-- The statement to prove
theorem total_cable_cost_neighborhood : total_cost = 760000 :=
by
  sorry

end total_cable_cost_neighborhood_l375_375485


namespace range_of_x_l375_375108

theorem range_of_x (a : ℝ) (h_a : a ∈ set.Icc (-1 : ℝ) 1) (x : ℝ) :
    (x ^ 2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) := 
sorry

end range_of_x_l375_375108


namespace fraction_equals_decimal_l375_375076

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375076


namespace coloring_problem_l375_375925

def condition (m n : ℕ) : Prop :=
  2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 ∧ m ≠ n ∧ m % n = 0

def color (f : ℕ → ℕ) : Prop :=
  ∀ m n, condition m n → f m ≠ f n

theorem coloring_problem :
  ∃ (k : ℕ) (f : ℕ → ℕ), (∀ n, 2 ≤ n ∧ n ≤ 31 → f n ≤ k) ∧ color f ∧ k = 4 :=
by
  sorry

end coloring_problem_l375_375925


namespace sum_of_years_with_digit_sum_10_l375_375349

def digits_sum_to (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_years_with_digit_sum_10 :
  let years := (2000:ℕ) :: (2001:ℕ) :: (2002:ℕ) :: (2003:ℕ) ::
               (2004:ℕ) :: (2005:ℕ) :: (2006:ℕ) :: (2007:ℕ) ::
               (2008:ℕ) :: (2009:ℕ) :: (2010:ℕ) :: (2011:ℕ) ::
               (2012:ℕ) :: (2013:ℕ) :: (2014:ℕ) :: (2015:ℕ) ::
               (2016:ℕ) :: (2017:ℕ) :: (2018:ℕ) :: (2019:ℕ) ::
               (2020:ℕ) :: (2021:ℕ) :: (2022:ℕ) :: (2023:ℕ) ::
               (2024:ℕ) :: (2025:ℕ) :: (2026:ℕ) :: (2027:ℕ) ::
               (2028:ℕ) :: (2029:ℕ) :: (2030:ℕ) :: (2031:ℕ) ::
               (2032:ℕ) :: (2033:ℕ) :: (2034:ℕ) :: (2035:ℕ) ::
               (2036:ℕ) :: (2037:ℕ) :: (2038:ℕ) :: (2039:ℕ) ::
               (2040:ℕ) :: (2041:ℕ) :: (2042:ℕ) :: (2043:ℕ) ::
               (2044:ℕ) :: (2045:ℕ) :: (2046:ℕ) :: (2047:ℕ) ::
               (2048:ℕ) :: (2049:ℕ) :: (2050:ℕ) :: (2051:ℕ) ::
               (2052:ℕ) :: (2053:ℕ) :: (2054:ℕ) :: (2055:ℕ) ::
               (2056:ℕ) :: (2057:ℕ) :: (2058:ℕ) :: (2059:ℕ) ::
               (2060:ℕ) :: (2061:ℕ) :: (2062:ℕ) :: (2063:ℕ) ::
               (2064:ℕ) :: (2065:ℕ) :: (2066:ℕ) :: (2067:ℕ) ::
               (2068:ℕ) :: (2069:ℕ) :: (2070:ℕ) :: (2071:ℕ) ::
               (2072:ℕ) :: (2073:ℕ) :: (2074:ℕ) :: (2075:ℕ) ::
               (2076:ℕ) :: (2077:ℕ) :: (2078:ℕ) :: (2079:ℕ) ::
               (2080:ℕ) :: (2081:ℕ) :: (2082:ℕ) :: (2083:ℕ) ::
               (2084:ℕ) :: (2085:ℕ) :: (2086:ℕ) :: (2087:ℕ) ::
               (2088:ℕ) :: (2089:ℕ) :: (2090:ℕ) :: (2091:ℕ) ::
               (2092:ℕ) :: (2093:ℕ) :: (2094:ℕ) :: (2095:ℕ) ::
               (2096:ℕ) :: (2097:ℕ) :: (2098:ℕ) :: (2099:ℕ) :: []
  ∑ y in years.filter (λ y, digits_sum_to y = 10), y = 18396 :=
by
  sorry

end sum_of_years_with_digit_sum_10_l375_375349


namespace fraction_vanya_ate_l375_375358

variables {x y T : ℕ}

-- Given conditions
def condition1 : Prop := T = x + y 
def condition2 : Prop := x = 5 * (y / 2)

-- Question: Prove the fraction of cookies Vanya ate by the time Tanya arrived is 5/7
theorem fraction_vanya_ate (h1 : condition1) (h2 : condition2) : (x : ℚ) / (T : ℚ) = 5 / 7 :=
by sorry

end fraction_vanya_ate_l375_375358


namespace kim_points_correct_l375_375748

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end kim_points_correct_l375_375748


namespace quadrilateral_inscribed_AC_l375_375290

open Real

/-- Quadrilateral ABCD is inscribed in a circle with,
    ∠ BAC = 60°, ∠ ADB = 50°, AD = 3, and BC = 5.
    Prove that AC ≈ 6.5
-/
theorem quadrilateral_inscribed_AC (α β : ℝ) (AD BC : ℝ) (AC : ℝ) : α = 60 → β = 50 → AD = 3 → BC = 5 →
  let γ := 180 - (α + β) in 
  AC ≈ 5 * (sin γ * sin⁻¹ β) :=
by
  sorry

end quadrilateral_inscribed_AC_l375_375290


namespace find_b_l375_375138

-- Define the given conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Given specific values for sides and area
def a_value : ℝ := 4
def c_value : ℝ := 6
def area : ℝ := 6 * Real.sqrt 3

-- Assuming the triangle is acute-angled
axiom acute_angle (A B C : ℝ) (a b c : ℝ) : a = 4 → c = 6 → (∀ x, x ∈ {A, B, C} → x < π / 2) → (1 / 2) * a * c * Real.sin B = area → b = 2 * Real.sqrt 7

-- Main theorem
theorem find_b (A B C : ℝ) (a b c : ℝ) (h_a : a = 4) (h_c : c = 6) (h_area : (1 / 2) * a * c * Real.sin B = area) :
  (∀ x, x ∈ {A, B, C} → x < π / 2) → b = 2 * Real.sqrt 7 :=
acute_angle A B C a b c h_a h_c

end find_b_l375_375138


namespace sin_double_angle_l375_375999

theorem sin_double_angle (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 / 3 := 
sorry

end sin_double_angle_l375_375999


namespace hexagon_perimeter_invariance_l375_375492

theorem hexagon_perimeter_invariance
  (T1 T2 : Type) [equilateral_triangle T1] [equilateral_triangle T2] 
  (H : Type) [hexagon H (T1 ∩ T2)] 
  (translate_T1_parallelly : Type → Type) :
  perimeter (H (T1 ∩ T2)) = perimeter (H (translate_T1_parallelly T1 ∩ T2)) :=
sorry

end hexagon_perimeter_invariance_l375_375492


namespace range_of_f_l375_375760

noncomputable def f (x : ℝ) : ℝ := 4^x + 2^(x + 1) + 1

theorem range_of_f : Set.range f = {y : ℝ | y > 1} :=
by
  sorry

end range_of_f_l375_375760


namespace slope_divides_L_shape_exactly_in_half_l375_375623

-- Define the vertices of the L-shaped region
def vertex_A := (0, 0)
def vertex_B := (0, 4)
def vertex_C := (4, 4)
def vertex_D := (4, 2)
def vertex_E := (7, 2)
def vertex_F := (7, 0)

-- Define the area function
def area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((p1.1 - p3.1) * (p2.2 - p1.2) - (p1.1 - p2.1) * (p3.2 - p1.2))

-- Define the areas for the two rectangles
def area_ABCD : ℝ := 16
def area_CDEF : ℝ := 6
def total_area : ℝ := 22
def half_area : ℝ := 11

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the intersection point
def intersection_point_G : ℝ × ℝ := (4, 4 - 32 / 9)

-- Lean statement to prove the slope
theorem slope_divides_L_shape_exactly_in_half :
  slope vertex_A intersection_point_G = 1 / 9 :=
by
  sorry

end slope_divides_L_shape_exactly_in_half_l375_375623


namespace sawing_time_incorrect_l375_375854

theorem sawing_time_incorrect (h1 : ∀ n : ℕ, sawing_time (n+1) = (n * single_sawing_time)) :
  ∃ t : ℕ, t ≠ 24 := 
by
  have h2 : single_sawing_time = 12 / 3, from sorry
  have h3 : sawing_time 8 = 7 * single_sawing_time, from sorry
  have h4 : 7 * (12 / 3) = 28, from sorry
  use 28
  sorry

end sawing_time_incorrect_l375_375854


namespace num_trained_in_all_three_restaurants_l375_375482

-- Definitions corresponding to the conditions
def num_employees := 39
def num_family_buffet := 17
def num_dining_room := 18
def num_snack_bar := 12
def num_exactly_two := 4

-- The proof problem to verify the number of employees trained in all three restaurants
theorem num_trained_in_all_three_restaurants : 
  ∃ x : ℕ, 17 + 18 + 12 - 4 - 2 * x + x = num_employees ∧ x = 8 :=
begin
  use 8,
  split,
  { -- Show that the equation holds with x = 8
    linarith,
  },
  { -- Show that x = 8
    refl,
  }
end

end num_trained_in_all_three_restaurants_l375_375482


namespace calculate_value_l375_375486

theorem calculate_value (a b c x : ℕ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) (h_x : x = 3) :
  x^(a * (b + c)) - (x^a + x^b + x^c) = 204 := by
  sorry

end calculate_value_l375_375486


namespace manufacturer_profit_percentage_l375_375454

theorem manufacturer_profit_percentage :
  (∀ (wholesaler_profit retailer_profit manufacturer_cost customer_price : ℝ),
    wholesaler_profit = 0.20 →
    retailer_profit = 0.25 →
    manufacturer_cost = 17 →
    customer_price = 30.09 →
    let retailer_cost = customer_price / (1 + retailer_profit);
        wholesaler_cost = retailer_cost / (1 + wholesaler_profit);
        manufacturer_price = wholesaler_cost;
        manufacturer_profit = manufacturer_price - manufacturer_cost;
        manufacturer_profit_percentage = (manufacturer_profit / manufacturer_cost) * 100
    in manufacturer_profit_percentage ≈ 18) :=
sorry

end manufacturer_profit_percentage_l375_375454


namespace arithmetic_sequence_k_l375_375740

variable (x k : ℤ)

def term1 := 2 * x - 3
def term2 := 3 * x + 1
def term3 := 5 * x + k

theorem arithmetic_sequence_k :
  (term2 - term1 = term3 - term2) ↔ k = 5 - x := by
  sorry

end arithmetic_sequence_k_l375_375740


namespace pirates_coins_l375_375850

theorem pirates_coins (x : ℕ) (h : (x * 9.succ.factorial) % 10^9 = 0) :
  (x * 9.succ.factorial / 10^9) = 1 :=
by
  sorry

end pirates_coins_l375_375850


namespace fraction_to_decimal_equiv_l375_375009

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375009


namespace mirella_page_difference_l375_375274

theorem mirella_page_difference :
  let purple_books := 8,
      purple_pages_per_book := 320,
      orange_books := 7,
      orange_pages_per_book := 640,
      total_purple_pages := purple_books * purple_pages_per_book,
      total_orange_pages := orange_books * orange_pages_per_book
  in total_orange_pages - total_purple_pages = 1920 :=
by
  sorry

end mirella_page_difference_l375_375274


namespace exists_a_i_l375_375426

theorem exists_a_i (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ i j k : Fin n, x i j + x l k + x k l = 0) : 
  ∃ a : Fin n → ℝ, ∀ i j : Fin n, x i j = a i - a j := 
sorry

end exists_a_i_l375_375426


namespace fraction_to_decimal_l375_375064

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375064


namespace vertical_asymptotes_sum_l375_375737

theorem vertical_asymptotes_sum (A B C : ℤ)
  (h : ∀ x : ℝ, x = -1 ∨ x = 2 ∨ x = 3 → x^3 + A * x^2 + B * x + C = 0)
  : A + B + C = -3 :=
sorry

end vertical_asymptotes_sum_l375_375737


namespace angle_bam_l375_375547

noncomputable def is_isosceles (A B C : Point) : Prop :=
  dist A B = dist A C

noncomputable def midpoint (C A K : Point) : Prop :=
  dist A C = dist C K

noncomputable def angle_abc (A B C : Point) : Real :=
  53

noncomputable def max_angle_mak (C A K M B : Point) : Prop :=
  dist K M = dist A B ∧
  (M = some_point_with_max_angle MAK)

theorem angle_bam
  (A B C K M : Point)
  (hAB_AC : is_isosceles A B C)
  (hABC_53 : angle_abc A B C = 53)
  (hMid_C : midpoint C A K)
  (hB_M_Side : same_side_line B M (line_through A C))
  (hKM_AB : dist K M = dist A B)
  (hMax_MAK : max_angle_mak C A K M B) :
  ∠BAM = 44 :=
sorry

end angle_bam_l375_375547


namespace correct_operation_div_sqrt_l375_375814

theorem correct_operation_div_sqrt (x y z w : ℝ) (h1 : 3 * real.sqrt 2 - real.sqrt 2 ≠ 3)
                                   (h2 : real.sqrt 2 * real.sqrt 3 ≠ real.sqrt 5)
                                   (h3 : (real.sqrt 2 - 1)^2 ≠ 2 - 1)
                                   (h4 : real.sqrt 27 / 3 = real.sqrt 3) : 
  ∃ (op : ℝ), op = real.sqrt 27 / 3 ∧ op = real.sqrt 3 :=
by {
  use real.sqrt 27 / 3,
  split,
  exact h4,
  exact h4,
}

end correct_operation_div_sqrt_l375_375814


namespace fraction_equals_decimal_l375_375073

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375073


namespace convert_fraction_to_decimal_l375_375054

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375054


namespace eggs_per_cheesecake_l375_375845

theorem eggs_per_cheesecake (E : ℕ) : 
  (∀ chocolate_cake_eggs cheesecakes : ℕ, chocolate_cake_eggs = 3 
    ∧ 57 + 5 * chocolate_cake_eggs = 9 * E 
    → E = 8) :=
begin
  intros chocolate_cake_eggs cheesecakes h,
  sorry
end

end eggs_per_cheesecake_l375_375845


namespace first_train_speed_is_80_kmph_l375_375794

noncomputable def speedOfFirstTrain
  (lenFirstTrain : ℝ)
  (lenSecondTrain : ℝ)
  (speedSecondTrain : ℝ)
  (clearTime : ℝ)
  (oppositeDirections : Bool) : ℝ :=
  if oppositeDirections then
    let totalDistance := (lenFirstTrain + lenSecondTrain) / 1000  -- convert meters to kilometers
    let timeHours := clearTime / 3600 -- convert seconds to hours
    let relativeSpeed := totalDistance / timeHours
    relativeSpeed - speedSecondTrain
  else
    0 -- This should not happen based on problem conditions

theorem first_train_speed_is_80_kmph :
  speedOfFirstTrain 151 165 65 7.844889650207294 true = 80 :=
by
  sorry

end first_train_speed_is_80_kmph_l375_375794


namespace find_k_l375_375181

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l375_375181


namespace simplify_and_evaluate_l375_375713

theorem simplify_and_evaluate :
  let x := -1 in
  ((x-2)^2 - (2*x+3)*(2*x-3) - 4*x*(x-1)) = 6 :=
by
  let x := -1
  sorry

end simplify_and_evaluate_l375_375713


namespace typing_efficiency_l375_375796

theorem typing_efficiency :
  ∃ (x y : ℚ), 
    (30 * x + 25 * y = 1 ∧ 
     24 * x + 30 * y = 1 ∧ 
     1 / x = 60 ∧ 
     1 / y = 50) :=
begin
  use [1/60, 1/50],
  split,
  { ring, norm_cast, field_simp, norm_num },
  split,
  { ring, norm_cast, field_simp, norm_num },
  split,
  { field_simp, norm_num },
  { field_simp, norm_num }
end

end typing_efficiency_l375_375796


namespace martha_found_blocks_l375_375705

variable (initial_blocks final_blocks found_blocks : ℕ)

theorem martha_found_blocks 
    (h_initial : initial_blocks = 4) 
    (h_final : final_blocks = 84) 
    (h_found : found_blocks = final_blocks - initial_blocks) : 
    found_blocks = 80 := by
  sorry

end martha_found_blocks_l375_375705


namespace manufacturer_profit_percentage_l375_375453

theorem manufacturer_profit_percentage :
  (∀ (wholesaler_profit retailer_profit manufacturer_cost customer_price : ℝ),
    wholesaler_profit = 0.20 →
    retailer_profit = 0.25 →
    manufacturer_cost = 17 →
    customer_price = 30.09 →
    let retailer_cost = customer_price / (1 + retailer_profit);
        wholesaler_cost = retailer_cost / (1 + wholesaler_profit);
        manufacturer_price = wholesaler_cost;
        manufacturer_profit = manufacturer_price - manufacturer_cost;
        manufacturer_profit_percentage = (manufacturer_profit / manufacturer_cost) * 100
    in manufacturer_profit_percentage ≈ 18) :=
sorry

end manufacturer_profit_percentage_l375_375453


namespace find_equation_of_line_l_l375_375142

-- Defining the existing conditions
def circle (x y : ℝ) (a : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + a = 0
def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

-- The proof problem in Lean statement
theorem find_equation_of_line_l (a : ℝ) (ha : a < 3) :
  (∃ A B : ℝ × ℝ, circle A.1 A.2 a ∧ circle B.1 B.2 a ∧ midpoint A.1 A.2 B.1 B.2 = (-2, 3)) →
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ x - y + 5 = 0) :=
by
  sorry

end find_equation_of_line_l_l375_375142


namespace keystone_arch_larger_interior_angle_l375_375320

theorem keystone_arch_larger_interior_angle :
  ∀ (n : ℕ) (α : ℝ),
    n = 10 ∧ (∀ i, 0 ≤ i → i < n → isosceles_trapezoid α) 
    ∧ (let angles := (360:ℝ) / n in 
          let interior_larger := (180:ℝ) - (α / 2) in
          α = 18 ∧ interior_larger = 99) :=
begin
  intros n α,
  sorry
end

end keystone_arch_larger_interior_angle_l375_375320


namespace find_x_when_areas_equal_l375_375234

-- Definitions based on the problem conditions
def glass_area : ℕ := 4 * (30 * 20)
def window_area (x : ℕ) : ℕ := (60 + 3 * x) * (40 + 3 * x)
def total_area_of_glass : ℕ := glass_area
def total_area_of_wood (x : ℕ) : ℕ := window_area x - glass_area

-- Proof problem, proving x == 20 / 3 when total area of glass equals total area of wood
theorem find_x_when_areas_equal : 
  ∃ x : ℕ, (total_area_of_glass = total_area_of_wood x) ∧ x = 20 / 3 :=
sorry

end find_x_when_areas_equal_l375_375234


namespace part1_part2_part3_l375_375153

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log(x + 1)) / (a * x + 1)

theorem part1 (x : ℝ) : 
  (∀ (a : ℝ), a = 1 → (f' x = (1 - Real.log(x + 1)) / (x + 1)^2) → 
  (f' 0 = 1) → 
  tangent_line_at x 0 (1 * x) = some (λ (x : ℝ), x)) :=
sorry

theorem part2 : 
  (∀ (f : ℝ → ℝ) (h : ∀ x ∈ (0, 1), deriv f x ≥ 0), 
  ∀ a, (a ≥ 0 ∨ (-1 ≤ a ∧ a < 0)) → 
  range_of_a a = [-1, 1/(2*ln(2)-1)]) :=
sorry

theorem part3 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  ∑ cyc (k : ℝ), (3 * k - 1) * (Real.log (k + 1) / (k - 1)) ≤ 0 :=
sorry

end part1_part2_part3_l375_375153


namespace fraction_to_decimal_l375_375081

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375081


namespace total_produced_first_three_days_least_production_and_difference_total_wage_for_week_l375_375841

def deviation (day: String) : Int :=
    match day with
    | "Monday" => 6
    | "Tuesday" => -2
    | "Wednesday" => -3
    | "Thursday" => 14
    | "Friday" => -10
    | "Saturday" => 8
    | "Sunday" => -9
    | _ => 0

def planned_daily_production := 300
def planned_weekly_production := 2100

theorem total_produced_first_three_days : 
  planned_daily_production * 3 + (deviation "Monday" + deviation "Tuesday" + deviation "Wednesday") = 901 :=
by
    sorry

theorem least_production_and_difference : 
  let least_production := planned_daily_production + min (min (min (min (min (deviation "Monday") (deviation "Tuesday")) (deviation "Wednesday")) (deviation "Thursday")) (min (deviation "Friday")(deviation "Saturday"))) (deviation "Sunday")
  let most_production := planned_daily_production + max (max (max (max (max (deviation "Monday") (deviation "Tuesday")) (deviation "Wednesday")) (deviation "Thursday")) (max (deviation "Friday")(deviation "Saturday"))) (deviation "Sunday") 
  least_production = 290 ∧ (most_production - least_production) = 24 :=
by
    sorry

theorem total_wage_for_week : 
  let total_actual_production := planned_weekly_production + (deviation "Monday" + deviation "Tuesday" + deviation "Wednesday" + deviation "Thursday" + deviation "Friday" + deviation "Saturday" + deviation "Sunday")
  let wage := total_actual_production * 60 + ((deviation "Monday" + deviation "Tuesday" + deviation "Wednesday" + deviation "Thursday" + deviation "Friday" + deviation "Saturday" + deviation "Sunday") * 15)
  wage = 126300 :=
by
    sorry

end total_produced_first_three_days_least_production_and_difference_total_wage_for_week_l375_375841


namespace max_neighbors_same_country_l375_375255

noncomputable def max_people_with_neighbors_same_country (n : ℕ) (h : n ≥ 2) : ℕ :=
  (n-2)*(n-1)/2

theorem max_neighbors_same_country (n : ℕ) (h : n ≥ 2) (h1 : ∀ i j, i ≠ j → 
  (∀ k, (people_from_country i).left_neighbor(k) ≠ (people_from_country i).left_neighbor(j))):
  ∃ m, m = max_people_with_neighbors_same_country n h :=
sorry

end max_neighbors_same_country_l375_375255


namespace find_a_l375_375435

theorem find_a (a : ℝ) (x_values y_values : List ℝ)
  (h_y : ∀ x, List.getD y_values x 0 = 2.1 * List.getD x_values x 1 - 0.3) :
  a = 10 :=
by
  have h_mean_x : (1 + 2 + 3 + 4 + 5) / 5 = 3 := by norm_num
  have h_sum_y : (2 + 3 + 7 + 8 + a) / 5 = (2.1 * 3 - 0.3) := by sorry
  sorry

end find_a_l375_375435


namespace sum_of_roots_l375_375362

-- Define the polynomial
def poly : Polynomial ℝ := 3 * Polynomial.X^3 + 7 * Polynomial.X^2 - 6 * Polynomial.X - 10

-- The goal is to prove that the sum of the roots is equal to -7/3
theorem sum_of_roots : Σ p : Polynomial ℝ, p = poly ∧ p.root_sum p.splits = -7/3 :=
begin
  use poly,
  split,
  {
    -- First, we confirm poly is equal to the given polynomial
    refl,
  },
  {
    -- Then, we prove that the sum of its roots equals -7/3
    sorry,  -- We skip the proof details
  }
end

end sum_of_roots_l375_375362


namespace kilometers_to_meters_l375_375135

theorem kilometers_to_meters
  (km_to_hm : 1 = 10)  -- This represents 1 km = 10 hm
  (hm_to_m : 1 = 100)  -- This represents 1 hm = 100 m
  : 1 * (10 * 100) = 1000 :=
by 
  -- Substitute each conversion directly into the expression
  have h1 : 1 * 10 = 10 := by norm_num,
  have h2 : 10 * 100 = 1000 := by norm_num,
  calc
    1 * (10 * 100) = 1 * 1000 : by rw h2
    ... = 1000 : by norm_num

end kilometers_to_meters_l375_375135


namespace fraction_to_decimal_l375_375085

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375085


namespace max_elements_non_multiple_subset_l375_375268

noncomputable def maxSubsetSize (T : Set ℕ) (S : Set ℕ) : ℕ :=
  if S ⊆ T ∧ (∀ a b ∈ S, a ∣ b → a = b ∨ b ∣ a → a = b)
  then S.toFinset.card
  else 0

theorem max_elements_non_multiple_subset :
  let T := { n | ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 200 ∧ 0 ≤ b ∧ b ≤ 100 ∧ 0 ≤ c ∧ c ≤ 100 ∧ n = 2^a * 3^b * 5^c } in
  ∃ S ⊆ T,
  (∀ a b ∈ S, a ∣ b → a = b) ∧ 
  maxSubsetSize T S = 10201 :=
by sorry

end max_elements_non_multiple_subset_l375_375268


namespace initial_percentage_water_l375_375455

theorem initial_percentage_water (P : ℝ) (H1 : 150 * P / 100 + 10 = 40) : P = 20 :=
by
  sorry

end initial_percentage_water_l375_375455


namespace robin_birdwatching_l375_375704

theorem robin_birdwatching (N : ℕ) (r : ℕ) (p : ℕ) 
    (h1 : 2 / 3 * N = r)
    (h2 : 1 / 8 * N = p)
    (h3 : p = 5)
    (h4 : (2 / 3 + 1 / 8 + 5 / N : ℚ) = 1) :
    r = 16 :=
by
  have hN : N = 24 := sorry
  have hr : r = (2 / 3 * 24).toNat := sorry
  exact hr

end robin_birdwatching_l375_375704


namespace terminating_decimal_representation_l375_375509

theorem terminating_decimal_representation : 
  (67 / (2^3 * 5^4) : ℝ) = 0.0134 :=
    sorry

end terminating_decimal_representation_l375_375509


namespace probability_symmetric_line_l375_375467

theorem probability_symmetric_line (P : (ℕ × ℕ) := (5, 5))
    (n : ℕ := 10) (total_points remaining_points symmetric_points : ℕ) 
    (probability : ℚ) :
  total_points = n * n →
  remaining_points = total_points - 1 →
  symmetric_points = 4 * (n - 1) →
  probability = (symmetric_points : ℚ) / (remaining_points : ℚ) →
  probability = 32 / 99 :=
by
  sorry

end probability_symmetric_line_l375_375467


namespace center_of_circle_l375_375521

theorem center_of_circle (x y : ℝ) : (x^2 + y^2 - 10 * x + 4 * y + 13 = 0) → (x - y = 7) :=
by
  -- Statement, proof omitted
  sorry

end center_of_circle_l375_375521


namespace number_of_pairs_l375_375499

theorem number_of_pairs (N : ℕ) (y_range : fin N) (x_range : fin N) : 
  (card {xy_pair : (fin N × fin N) // (xy_pair.1.val ^ 2 + xy_pair.2.val ^ 2) % 121 = 0}) = 8100 :=
sorry

end number_of_pairs_l375_375499


namespace express_EC_l375_375830

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (A B C D E : V) (a b : V)

-- Conditions
axiom DA_eq_a : D - A = a
axiom DB_eq_b : D - B = b
axiom points_on_circle (h : ∃ r : ℝ, ∀ p ∈ {A, B, C, D, E}, dist p D = r) : 
  list.evenly_distributes [A, B, C, D, E]

-- Question: Express EC in terms of a and b
theorem express_EC :
  C - E = (1 + real.sqrt 5) / 2 • b - (1 + real.sqrt 5) / 2 • a :=
sorry

end express_EC_l375_375830


namespace restore_original_problem_l375_375871

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375871


namespace find_k_l375_375191

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l375_375191


namespace ab_value_l375_375208

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l375_375208


namespace horner_method_correct_l375_375797

-- Define the polynomial function using Horner's method
def f (x : ℤ) : ℤ := (((((x - 8) * x + 60) * x + 16) * x + 96) * x + 240) * x + 64

-- Define the value to be plugged into the polynomial
def x_val : ℤ := 2

-- Compute v_0, v_1, and v_2 according to the Horner's method
def v0 : ℤ := 1
def v1 : ℤ := v0 * x_val - 8
def v2 : ℤ := v1 * x_val + 60

-- Formal statement of the proof problem
theorem horner_method_correct :
  v2 = 48 := by
  -- Insert proof here
  sorry

end horner_method_correct_l375_375797


namespace minimum_people_in_photographs_l375_375221

theorem minimum_people_in_photographs : 
  (∀ (photo : ℕ → ℕ → ℕ → Prop), (∃ S : fin 10 → Prop, (∀ i, S i ∧ ∃ son brother, photo i son brother)) → 
  ∃ (people : fin 10 → ℕ) (son brother : fin 10 → ℕ), 
    (∀ i, photo (people i) (son i) (brother i)) →
    (∃ distinct_people, (∀ i j, i ≠ j → people i ≠ people j) ∧ (∃ k, distinct_people k)) ∧ 
    (distinct_people.length ≥ 16)) :=
begin
  sorry
end

end minimum_people_in_photographs_l375_375221


namespace intersection_in_polar_l375_375629

-- Definitions for the circle and line in polar coordinates
def circle_polar (ρ θ : ℝ) := ρ = Real.cos θ + Real.sin θ
def line_polar (ρ θ : ℝ) := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

-- Equivalent Cartesian coordinate forms
def circle_cartesian (x y : ℝ) := x^2 + y^2 - x - y = 0
def line_cartesian (x y : ℝ) := x - y + 1 = 0

-- Proving the intersection in polar coordinates
theorem intersection_in_polar :
  (∃ θ, ∃ ρ ≥ 0, θ ∈ set.Ico 0 (2 * Real.pi) ∧ circle_polar ρ θ ∧ line_polar ρ θ) →
  ∃ ρ θ, ρ = 1 ∧ θ = Real.pi / 2 := 
begin
  intros h,
  -- Expanding the definition of the intersection
  rcases h with ⟨θ, ⟨ρ, ⟨h_ρ, ⟨h_θ, ⟨h_circle, h_line⟩⟩⟩⟩⟩,
  -- Verifying intersection
  use [1, Real.pi / 2],
  split,
  { refl },
  { refl },
  sorry,
end

end intersection_in_polar_l375_375629


namespace fraction_equals_decimal_l375_375072

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375072


namespace similar_triangles_division_l375_375004

theorem similar_triangles_division (A B C D E F A1 A2 B1 B2 : Type)
  (h1 : similar A B C D E F)
  (h2A : divides A B C A1 A2)
  (h2B : divides D E F B1 B2)
  (h3 : similar A1 B1) :
  ¬ similar A2 B2 :=
by sorry

end similar_triangles_division_l375_375004


namespace angle_between_tangents_l375_375517

/-- Given the function y = (x^2 * sqrt(3)) / 6, the angle between the tangents to this curve 
    passing through the point M(1, -sqrt(3)/2) is 90 degrees. -/
theorem angle_between_tangents :
  let f (x : ℝ) := (x ^ 2 * real.sqrt 3) / 6
  ∃ (M : ℝ × ℝ), M = (1, -(real.sqrt 3) / 2) →
  angle (tangent_slope f M) = π / 2 :=
by
  sorry

end angle_between_tangents_l375_375517


namespace total_books_read_l375_375438

-- Definitions based on the conditions
def books_per_month : ℕ := 4
def months_per_year : ℕ := 12
def books_per_year_per_student : ℕ := books_per_month * months_per_year

variables (c s : ℕ)

-- Main theorem statement
theorem total_books_read (c s : ℕ) : 
  (books_per_year_per_student * c * s) = 48 * c * s :=
by
  sorry

end total_books_read_l375_375438


namespace polar_distance_l375_375667

noncomputable def distance_point (r1 θ1 r2 θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 ^ 2) + (r2 ^ 2) - 2 * r1 * r2 * Real.cos (θ1 - θ2))

theorem polar_distance :
  ∀ (θ1 θ2 : ℝ), (θ1 - θ2 = Real.pi / 2) → distance_point 5 θ1 12 θ2 = 13 :=
by
  intros θ1 θ2 hθ
  rw [distance_point, hθ, Real.cos_pi_div_two]
  norm_num
  sorry

end polar_distance_l375_375667


namespace emma_average_speed_l375_375507

-- Define the given conditions
def distance1 : ℕ := 420     -- Distance traveled in the first segment
def time1 : ℕ := 7          -- Time taken in the first segment
def distance2 : ℕ := 480    -- Distance traveled in the second segment
def time2 : ℕ := 8          -- Time taken in the second segment

-- Define the total distance and total time
def total_distance : ℕ := distance1 + distance2
def total_time : ℕ := time1 + time2

-- Define the expected average speed
def expected_average_speed : ℕ := 60

-- Prove that the average speed is 60 miles per hour
theorem emma_average_speed : (total_distance / total_time) = expected_average_speed := by
  sorry

end emma_average_speed_l375_375507


namespace gcd_possible_values_count_l375_375377

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375377


namespace minimize_sum_of_roots_l375_375105

noncomputable def f (a : ℝ) : ℝ := a^2 - real.sqrt 21 * a + 26
noncomputable def g (a : ℝ) : ℝ := (3 / 2) * a^2 - real.sqrt 21 * a + 27

def sum_of_roots_equation (a x : ℝ) : Prop :=
  (f a * x^2 + 1) / (x^2 + g a) = real.sqrt ((x * g a - 1) / (f a - x))

theorem minimize_sum_of_roots : ∃ a : ℝ, (∀ x : ℝ, sum_of_roots_equation a x) ∧
  a = real.sqrt 21 / 2 :=
sorry

end minimize_sum_of_roots_l375_375105


namespace solve_fractions_l375_375910

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375910


namespace gcd_lcm_product_360_l375_375380

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375380


namespace sum_of_consecutive_powers_of_2_divisible_by_7_l375_375702

theorem sum_of_consecutive_powers_of_2_divisible_by_7 (n : ℤ) : 7 ∣ (2^n + 2^(n+1) + 2^(n+2)) := 
sorry

end sum_of_consecutive_powers_of_2_divisible_by_7_l375_375702


namespace miles_per_gallon_l375_375275

theorem miles_per_gallon (miles gallons : ℝ) (h : miles = 100 ∧ gallons = 5) : miles / gallons = 20 := by
  cases h with
  | intro miles_eq gallons_eq =>
    rw [miles_eq, gallons_eq]
    norm_num

end miles_per_gallon_l375_375275


namespace find_A_l375_375424

theorem find_A (A B : ℝ) (h1 : B = 10 * A) (h2 : 211.5 = B - A) : A = 23.5 :=
by {
  sorry
}

end find_A_l375_375424


namespace restore_fractions_l375_375892

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375892


namespace dealership_sales_l375_375989

theorem dealership_sales :
  (∀ (n : ℕ), 3 * n ≤ 36 → 5 * n ≤ x) →
  (36 / 3) * 5 = 60 :=
by
  sorry

end dealership_sales_l375_375989


namespace positive_m_unique_solution_l375_375992

theorem positive_m_unique_solution :
  (∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, 3 * x^2 + m * x + 16 = 0 → (m^2 - 192 = 0))) ↔ ∃ m : ℝ, m > 0 ∧ m = 8 * real.sqrt 3 := 
by sorry

end positive_m_unique_solution_l375_375992


namespace sum_of_solutions_l375_375271

theorem sum_of_solutions : 
  (∀ x y : ℝ, (|x - 5| = 2 * |y - 7|) ∧ (|x - 7| = 3 * |y - 5|) → 
    (x, y) = (13, 11) ∨ (x, y) = (25, -3)) →
  ∃ (n : ℕ) (x y : Fin n → ℝ), 
    (∀ i, (|x i - 5| = 2 * |y i - 7| ∧ |x i - 7| = 3 * |y i - 5|)) ∧ 
    (x 0 + y 0 + x 1 + y 1 + ... + x (n - 1) + y (n - 1) = 46) :=
by
  intros h
  sorry

end sum_of_solutions_l375_375271


namespace horse_max_carrots_zero_l375_375851

def is_knight_move (start finish : ℤ × ℤ) : Prop :=
  (abs (start.1 - finish.1) = 2 ∧ abs (start.2 - finish.2) = 1) ∨
  (abs (start.1 - finish.1) = 1 ∧ abs (start.2 - finish.2) = 2)

def is_alternate_color (current : ℤ × ℤ) (next : ℤ × ℤ) : Bool :=
  (current.1 + current.2) % 2 ≠ (next.1 + next.2) % 2

theorem horse_max_carrots_zero
  (path : List (ℤ × ℤ))
  (h_start : path.head = (0, 0))
  (h_moves : ∀i, is_knight_move (path.nth! i) (path.nth! (i + 1)))
  (h_unique : ∀loc, path.count loc ≤ 2)
  : ∑ i in path, if (is_alternate_color (path.nth! i) (path.nth! (i + 1))) then 2 else -1 = 0 :=
sorry

end horse_max_carrots_zero_l375_375851


namespace fraction_to_decimal_l375_375066

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375066


namespace fraction_to_decimal_l375_375087

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375087


namespace sum_max_min_f_l375_375767

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 - 6 * (Real.sin x) + 2

theorem sum_max_min_f : 
  (let max_f := (λ x, f x) (Real.pi / 2) in -- assume max_f occurs at x = π/2
  let min_f := (λ x, f x) (3 * Real.pi / 2) in -- assume min_f occurs at x = 3π/2
  max_f + min_f) = 8 :=
by
  sorry

end sum_max_min_f_l375_375767


namespace fraction_to_decimal_l375_375033

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375033


namespace fountain_distance_l375_375793

theorem fountain_distance (h_AD : ℕ) (h_BC : ℕ) (h_AB : ℕ) (h_AD_eq : h_AD = 30) (h_BC_eq : h_BC = 40) (h_AB_eq : h_AB = 50) :
  ∃ AE EB : ℕ, AE = 32 ∧ EB = 18 := by
  sorry

end fountain_distance_l375_375793


namespace local_minimum_f_at_1_for_k_2_l375_375553

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := 
  (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_at_1_for_k_2 : 
  (∀ k : ℕ, k = 2 → ∃ ε > 0, ∀ x : ℝ, abs(x - 1) < ε → f 2 x ≥ f 2 1) := 
sorry

end local_minimum_f_at_1_for_k_2_l375_375553


namespace pairs_count_eq_five_l375_375200

theorem pairs_count_eq_five : 
  (∃ (m n : ℕ), (3 / 2008 = 1 / m + 1 / n) ∧ (m < n)) ↔ 5 := 
sorry

end pairs_count_eq_five_l375_375200


namespace votes_of_third_candidate_l375_375774

open Real

def votes1 : ℝ := 4136
def votes2 : ℝ := 11628
def winning_percentage : ℝ := 49.69230769230769 / 100

theorem votes_of_third_candidate :
  ∃ V votes3, V = votes1 + votes2 + winning_percentage * V ∧ votes3 = V - votes1 - votes2 ∧ votes3 = 15584 :=
by
  sorry

end votes_of_third_candidate_l375_375774


namespace store_meter_longer_than_standard_l375_375861
open Real

variable (a : ℝ) (L : ℝ)
hypothesis h1 : L = 140 / 139
hypothesis h2 : (140 % a - L * a) / (L * a) = 0.39

theorem store_meter_longer_than_standard :
  L - 1 = 1 / 139 := sorry

end store_meter_longer_than_standard_l375_375861


namespace sequence_sum_l375_375632

theorem sequence_sum (a : ℕ+ → ℚ) (h : ∀ n : ℕ+, ∏ i in finset.range n, a i.succ = n^2) :
  a 3 + a 5 = 61 / 16 :=
begin
  sorry
end

end sequence_sum_l375_375632


namespace tan_A_in_triangle_ABC_l375_375213

theorem tan_A_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) (ha : 0 < A) (ha_90 : A < π / 2) 
(hb : b = 3 * a * Real.sin B) : Real.tan A = Real.sqrt 2 / 4 :=
sorry

end tan_A_in_triangle_ABC_l375_375213


namespace sum_bounds_l375_375495

noncomputable def sum_of_inverses_square (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (i + 1)^2

theorem sum_bounds (n : ℕ) (hn_pos : 0 < n) :
  (1 - 1 / (2 * n + 1)) * (1 - 2 / (2 * n + 1)) * (π^2 / 6) < 
  sum_of_inverses_square n ∧ 
  sum_of_inverses_square n <
  (1 - 1 / (2 * n + 1)) * (1 + 1 / (2 * n + 1)) * (π^2 / 6) :=
by
  sorry

end sum_bounds_l375_375495


namespace mixed_fractions_product_l375_375867

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375867


namespace parabola_focus_coordinates_l375_375315

theorem parabola_focus_coordinates : 
  ∀ x y : ℝ, (∀ x, x^2 = 4 * y) -> (0, 1) = (0, y/2) := 
by
  intros x y h_eq
  have h1 : 4 * y = 4 * (1) := by sorry
  have h2 : y = 2 := by sorry
  have h3 : (0, 2 / 2) = (0, 1) := by sorry
  exact h3

end parabola_focus_coordinates_l375_375315


namespace number_of_days_in_first_part_l375_375732

variable {x : ℕ}

-- Conditions
def avg_exp_first_part (x : ℕ) : ℕ := 350 * x
def avg_exp_next_four_days : ℕ := 420 * 4
def total_days (x : ℕ) : ℕ := x + 4
def avg_exp_whole_week (x : ℕ) : ℕ := 390 * total_days x

-- Equation based on the conditions
theorem number_of_days_in_first_part :
  avg_exp_first_part x + avg_exp_next_four_days = avg_exp_whole_week x →
  x = 3 :=
by
  sorry

end number_of_days_in_first_part_l375_375732


namespace select_model_and_predict_value_l375_375432

noncomputable def model_choice (R1 R2 : ℝ) : Prop :=
  R1 < R2

noncomputable def regression_equation (y : ℝ) (x : ℝ) : ℝ :=
  100 / x + 2
  
definition predicted_value (x : ℝ) : ℝ :=
  regression_equation y x

theorem select_model_and_predict_value :
  let R1 := 0.7891
  let R2 := 0.9485
  let regression_y := 6  
  model_choice R1 R2 → regression_equation y 25 = regression_y := by
    sorry

end select_model_and_predict_value_l375_375432


namespace restore_original_problem_l375_375902

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375902


namespace waiter_tips_l375_375939

theorem waiter_tips (total_customers : ℕ) (no_tip_customers : ℕ) (tip_per_customer : ℕ) (tipping_customers : ℕ) :
  total_customers = 10 → no_tip_customers = 5 → tip_per_customer = 3 → tipping_customers = total_customers - no_tip_customers → 
  tipping_customers * tip_per_customer = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry -- Proof omitted according to instructions

end waiter_tips_l375_375939


namespace inequality_max_value_l375_375512

theorem inequality_max_value :
  (∀ (C : ℝ), (∀ (α β x : ℝ), α^2 + β^2 ≤ 16 → |α * sin (2 * x) + β * cos (8 * x)| ≤ C) → C ≥ 4 * sqrt 2) :=
by
  sorry

end inequality_max_value_l375_375512


namespace chance_of_picking_pepperjack_l375_375652

-- Defining the conditions of the problem
def cheddarSticks : ℕ := 15
def mozzarellaSticks : ℕ := 30
def pepperjackSticks : ℕ := 45

-- Defining the problem as a theorem in Lean
theorem chance_of_picking_pepperjack :
  let totalSticks := cheddarSticks + mozzarellaSticks + pepperjackSticks in
  (pepperjackSticks : ℚ) / totalSticks * 100 = 50 := by
  sorry

end chance_of_picking_pepperjack_l375_375652


namespace scheduling_methods_count_l375_375761

def subjects : List String := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"]

def is_valid_schedule (schedule : List String): Bool :=
  schedule.length = 6 ∧ 
  schedule.head! ≠ "Chinese" ∧ 
  schedule[5]! = "Biology"

def valid_schedules_count : Nat :=
  (List.permutations subjects).count (λ schedule => is_valid_schedule schedule)

theorem scheduling_methods_count :
  valid_schedules_count = 96 :=
sorry

end scheduling_methods_count_l375_375761


namespace vector_parallel_k_value_l375_375161

theorem vector_parallel_k_value (k : ℝ) (h_parallel : (∃ λ : ℝ, (k, √2) = (λ * 2, λ * (-2)))) : k = -√2 :=
by
  sorry

end vector_parallel_k_value_l375_375161


namespace janelle_marbles_l375_375246

variable (initial_green : ℕ) (bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ)

def marbles_left (initial_green bags marbles_per_bag gift_green gift_blue : ℕ) : ℕ :=
  initial_green + (bags * marbles_per_bag) - (gift_green + gift_blue)

theorem janelle_marbles : marbles_left 26 6 10 6 8 = 72 := 
by 
  simp [marbles_left]
  sorry

end janelle_marbles_l375_375246


namespace average_salary_technicians_is_l375_375619

variable 
  (average_salary_all : ℝ) 
  (total_workers : ℕ) 
  (average_salary_non_technicians : ℝ) 
  (num_technicians : ℕ) 
  (num_non_technicians : ℕ) 
  (total_salary_all : ℝ) 
  (total_salary_non_technicians : ℝ) 
  (total_salary_technicians : ℝ)

-- Defining the conditions
def condition_1 : Prop := average_salary_all = 8000
def condition_2 : Prop := total_workers = 24
def condition_3 : Prop := average_salary_non_technicians = 6000
def condition_4 : Prop := num_technicians = 8
def condition_5 : Prop := num_non_technicians = total_workers - num_technicians
def condition_6 : Prop := total_salary_all = average_salary_all * total_workers
def condition_7 : Prop := total_salary_non_technicians = average_salary_non_technicians * num_non_technicians
def condition_8 : Prop := total_salary_technicians = total_salary_all - total_salary_non_technicians
def condition_9 : Prop := 
  total_salary_technicians = (average_salary_all * total_workers) - (average_salary_non_technicians * num_non_technicians)

-- Using the conditions to prove the average salary of the technicians
theorem average_salary_technicians_is 
  (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) 
  (h4 : condition_4) (h5 : condition_5) 
  (h6 : condition_6) (h7 : condition_7)
  (h8 : condition_8) :
  (total_salary_technicians / num_technicians) = 12000 := 
by
  sorry

end average_salary_technicians_is_l375_375619


namespace smallest_degree_polynomial_l375_375128

variable {p k n : ℕ}

theorem smallest_degree_polynomial (hp : Nat.Prime p) (hk : k > 0) (hn : n > 0) :
  ∃ (d : ℕ), 
    (∀ (f : (Fin n → ℕ) → ℤ), 
      (∀ (a : Fin n → ℕ), 
         (a i ∈ {0, 1} →
          (p | f a) ↔ (p^k | ∑ i, a i))) 
      → d = p^k) :=
sorry

end smallest_degree_polynomial_l375_375128


namespace grid_is_valid_l375_375971

-- Definitions for the grid and conditions
def is_valid_grid (grid : Array (Array Nat)) : Prop :=
  (∀ i, (grid[i].toList.nodup)) ∧       -- Each row contains unique elements
  (∀ j, (grid.map (fun row => row[j])).toList.nodup) ∧  -- Each column contains unique elements
  (grid[0][1] * grid[0][2] = 6) ∧       -- Top right corner product condition
  (grid[1][0] - grid[1][1] = 1) ∧       -- Center horizontal subtraction condition
  (grid[2][0] + grid[2][2] = 4)         -- Top center vertical addition condition

def example_grid : Array (Array Nat) :=
  #[ #[3, 2, 1],
     #[1, 3, 2],
     #[2, 1, 3] ]

theorem grid_is_valid : is_valid_grid example_grid :=
by
  sorry

end grid_is_valid_l375_375971


namespace major_axis_length_of_ellipse_l375_375562

noncomputable def major_axis_length (m : ℝ) : ℝ :=
  if 4 + 2^2 = m then 2 * real.sqrt m else 0

theorem major_axis_length_of_ellipse (m : ℝ) (cond : 4 + 2^2 = m) :
  major_axis_length m = 4 * real.sqrt 2 :=
by
  unfold major_axis_length
  rw [if_pos cond]
  have h : real.sqrt m = 2 * real.sqrt 2 := sorry,
  rw [h]
  norm_num
  sorry

end major_axis_length_of_ellipse_l375_375562


namespace row_number_sum_l375_375694

theorem row_number_sum (n : ℕ) (h : (2 * n - 1) ^ 2 = 2015 ^ 2) : n = 1008 :=
by
  sorry

end row_number_sum_l375_375694


namespace doritos_in_each_pile_l375_375691

theorem doritos_in_each_pile:
  (total_chips : ℕ) (one_quarter_chips : ℕ) (piles : ℕ) 
  (h1 : total_chips = 80) 
  (h2 : one_quarter_chips = total_chips / 4) 
  (h3 : piles = 4) :
  one_quarter_chips / piles = 5  := 
sorry

end doritos_in_each_pile_l375_375691


namespace strictly_increasing_interval_l375_375497

open Set

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (-x^2 + 4 * x)

theorem strictly_increasing_interval : ∀ x ∈ (Ici 2 : Set ℝ), StrictMono f :=
by
  intros x hx
  let t := -x^2 + 4 * x
  have decreasing_t : ∀ y, 2 ≤ y → t' deriving Use the properties of quadratic functions,
  apply StrictMonoOn_of_deriv_pos,
  ...
  sorry

end strictly_increasing_interval_l375_375497


namespace coloring_impossible_l375_375763

theorem coloring_impossible : 
  let initial_sequence := (List.repeat 0 49).intercalate [1] ++ [2]
  let target_sequence := (List.repeat 1 49).intercalate [0] ++ [2, 1]
  ∀ (sequence : List ℕ), 
    sequence.perm initial_sequence →
    (∀ i : ℕ, i < sequence.length - 1 → sequence.get i ≠ sequence.get (i + 1)) →
    sequence ≠ target_sequence :=
by {
  let initial_sequence := (List.repeat 0 49.intercalate [1]) ++ [2]
  let target_sequence := (List.repeat 1 49.intercalate [0]) ++ [2, 1]
  intro sequence
  intro perm
  intro adj_diff
  sorry
}

end coloring_impossible_l375_375763


namespace milburg_population_l375_375332

/-- Number of grown-ups in Milburg --/
def grownUps : ℕ := 5256

/-- Number of children in Milburg --/
def children : ℕ := 2987

/-- Total number of people in Milburg --/
def totalPeople : ℕ := grownUps + children

theorem milburg_population : totalPeople = 8243 := by
  have h1 : grownUps = 5256 := rfl
  have h2 : children = 2987 := rfl
  have h3 : totalPeople = grownUps + children := rfl
  have h4 : grownUps + children = 8243 := by
    calc
      5256 + 2987 = 8243 := by sorry -- Proof step to be filled in
  exact h4

end milburg_population_l375_375332


namespace PX_eq_2AM_l375_375272

variables (A B C P Q X Y M : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space P] [metric_space Q] [metric_space X] [metric_space Y] [metric_space M]

variable (triangle_ABC : A ≠ B ∧ B ≠ C ∧ C ≠ A)
variable (P_Q_X_Y_on_side_A_of_BC : same_side A B C P Q X Y)
variable (is_square_ABYX : is_square A B Y X)
variable (is_square_ACQP : is_square A C Q P)
variable (midpoint_M_BC : M = midpoint B C)

theorem PX_eq_2AM : dist P X = 2 * dist A M :=
sorry

end PX_eq_2AM_l375_375272


namespace units_digit_of_product_l375_375525

theorem units_digit_of_product (a b c : ℕ) (n m p : ℕ) (units_a : a ≡ 4 [MOD 10])
  (units_b : b ≡ 9 [MOD 10]) (units_c : c ≡ 16 [MOD 10])
  (exp_a : n = 150) (exp_b : m = 151) (exp_c : p = 152) :
  (a^n * b^m * c^p) % 10 = 4 :=
by
  sorry

end units_digit_of_product_l375_375525


namespace part1_part2_l375_375131

-- Part (1) Statement
theorem part1 (A B: Set ℝ) (R: Set ℝ) (m: ℝ) 
(hA: A = {x | -m + 2 ≤ x ∧ x ≤ m + 2})
(hB: B = {x | x ≤ -2 ∨ x ≥ 4})
(hUnion: A ∪ B = R):
  m ∈ Set.Ici 4 :=
sorry

-- Part (2) Statement
theorem part2 (A B: Set ℝ) (R: Set ℝ) (m: ℝ) 
(hA: A = {x | -m + 2 ≤ x ∧ x ≤ m + 2})
(hB: B = {x | x ≤ -2 ∨ x ≥ 4})
(hComplement: A = λ x, x ∈ R ∧ x ∉ B):
  m ∈ Set.Ioo 0 2 :=
sorry

end part1_part2_l375_375131


namespace length_segment_midpoints_bases_l375_375002

variable (A B C D E G F H O : Type)
variables [add_comm_group A] [vector_space ℝ A] [affine_space A E] [add_comm_group G] [vector_space ℝ G] [affine_space G B] [add_comm_group F] [vector_space ℝ F] [affine_space F C] [add_comm_group H] [vector_space ℝ H] [affine_space H D] [add_comm_group O] [vector_space ℝ O]

-- Define the conditions
variables {AB CD AC BD : set A}
variables {midpoint : A → A → A}
variables (AC_perp_BD : ∀ (O: A), (AC ∩ BD = {O}) → AC ⊥ BD)
variables (midline_eq_5 : ∀ {E G : A}, E ∈ midpoint A B → G ∈ midpoint C D → dist E G = 5)

-- Define the theorem to prove the length of the segment connecting the midpoints
theorem length_segment_midpoints_bases (ABCD_trapezium : parallelogram A B C D) :
  ∃ F H : A, 
    F ∈ midpoint A D → 
    H ∈ midpoint B C → 
    dist F H = 5 := 
sorry

end length_segment_midpoints_bases_l375_375002


namespace cos_105_degree_value_l375_375982

noncomputable def cos105 : ℝ := Real.cos (105 * Real.pi / 180)

theorem cos_105_degree_value :
  cos105 = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  sorry

end cos_105_degree_value_l375_375982


namespace only_one_solution_l375_375980

theorem only_one_solution (n : ℕ) (h : 0 < n ∧ ∃ a : ℕ, a * a = 5^n + 4) : n = 1 :=
sorry

end only_one_solution_l375_375980


namespace convert_fraction_to_decimal_l375_375056

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375056


namespace select_student_for_performance_and_stability_l375_375995

def average_score_A : ℝ := 6.2
def average_score_B : ℝ := 6.0
def average_score_C : ℝ := 5.8
def average_score_D : ℝ := 6.2

def variance_A : ℝ := 0.32
def variance_B : ℝ := 0.58
def variance_C : ℝ := 0.12
def variance_D : ℝ := 0.25

theorem select_student_for_performance_and_stability :
  (average_score_A ≤ average_score_D ∧ variance_D < variance_A) →
  (average_score_B < average_score_A ∧ average_score_B < average_score_D) →
  (average_score_C < average_score_A ∧ average_score_C < average_score_D) →
  "D" = "D" :=
by
  intros h₁ h₂ h₃
  exact rfl

end select_student_for_performance_and_stability_l375_375995


namespace gcd_possible_values_count_l375_375394

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375394


namespace range_of_m_plus_n_l375_375538

theorem range_of_m_plus_n (m n : ℝ) (h1 : ∀ x ∈ Icc m n, (log 2 (4 - |x|)) ∈ Icc (0 : ℝ) 2)
                         (h2 : ∃ t : ℝ, ((1 / 2) ^ |t| + m + 1 = 0)) :
  1 ≤ m + n ∧ m + n < 2 :=
by 
  sorry

end range_of_m_plus_n_l375_375538


namespace solve_fractions_l375_375914

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l375_375914


namespace simon_fraction_of_alvin_l375_375928

theorem simon_fraction_of_alvin (alvin_age simon_age : ℕ) (h_alvin : alvin_age = 30)
  (h_simon : simon_age = 10) (h_fraction : ∃ f : ℚ, simon_age + 5 = f * (alvin_age + 5)) :
  ∃ f : ℚ, f = 3 / 7 := by
  sorry

end simon_fraction_of_alvin_l375_375928


namespace f_neg1_equals_neg3_l375_375266

-- Define the odd function f(x).
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -(2^(-x) + 2*(-x) - 1)

-- Definition that f(x) is an odd function.
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Prove that f(-1) = -3 given the conditions.
theorem f_neg1_equals_neg3 :
  is_odd_function f →
  f (0) = 0 →
  (∀ x ≥ 0, f (x) = 2^x + 2*x - 1) →
  f (-1) = -3 :=
by
  intros h_odd h_f0 h_pos
  have hb : ∀ x ≥ 0, f x = 2^x + 2*x - 1 := h_pos
  have h_b : f 0 = 0 := h_f0
  have h_b_neg : f (-1) = - f (1) := (h_odd (-1))
  rw[<- h_b_neg,<- hb]
  exact sorry

end f_neg1_equals_neg3_l375_375266


namespace probability_winning_pair_is_5_over_11_l375_375445

def deck_card := (color : String, number : Nat)

def deck : List deck_card := List.concat [
  [deck_card ("red", 1), deck_card ("red", 2), deck_card ("red", 3), deck_card ("red", 4)],
  [deck_card ("green", 1), deck_card ("green", 2), deck_card ("green", 3), deck_card ("green", 4)],
  [deck_card ("yellow", 1), deck_card ("yellow", 2), deck_card ("yellow", 3), deck_card ("yellow", 4)]
]

def is_winning_pair (c1 c2 : deck_card) : Bool :=
  c1.color = c2.color ∨ c1.number = c2.number

noncomputable def probability_winning_pair : ℚ :=
  let total_ways := (Multiset.cons, List.combinations 2 deck).card
  let winning_ways := (Multiset.cons, (List.combinations 2 deck).filter (λ p => is_winning_pair p.1 p.2)).card
  winning_ways / total_ways

theorem probability_winning_pair_is_5_over_11 : probability_winning_pair = 5 / 11 := by
  sorry

end probability_winning_pair_is_5_over_11_l375_375445


namespace convert_fraction_to_decimal_l375_375061

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l375_375061


namespace problem_statement_l375_375784

noncomputable def lengthPS : ℝ := 9 * real.sqrt 129

theorem problem_statement : ∀ (P Q R S : Type) (PQ QR RP : ℝ)
  (circumference_radius : ℝ), 
  PQ = 39 → QR = 36 → RP = 15 → 
  circumference_radius = 15 * real.sqrt 129 / 13 →
  S lies on the circle circumscribed around △PQR and the perpendicular bisector of RP
  → (floor (9 + real.sqrt 129) = 21) :=
begin
  intros P Q R S PQ QR RP circumference_radius,
  assume hPQ hQR hRP hRad hS,
  rw hPQ at *,
  rw hQR at *,
  rw hRP at *,
  rw hRad at *,
  sorry, -- Here would be the rest of the proof
end

end problem_statement_l375_375784


namespace necessarily_positive_l375_375294

noncomputable def Real := ℝ

-- Problem domain: Real-valued variables with specific constraints
variables (a b c : Real)

-- Conditions given in the problem
axiom a_condition : 0 < a ∧ a < 2
axiom b_condition : -2 < b ∧ b < 0
axiom c_condition : 0 < c ∧ c < 1

-- The equivalent statement to prove
theorem necessarily_positive : b + 3 * b^2 > 0 :=
by
  sorry

end necessarily_positive_l375_375294


namespace angle_C_magnitude_max_perimeter_triangle_l375_375638

theorem angle_C_magnitude (a b c A B C : ℝ) (h₁ : a * real.sin A - c * real.sin C = (a - b) * real.sin B)
    (h₂ : 0 < C) (h₃ : C < real.pi) : C = real.pi / 3 :=
by
    sorry

theorem max_perimeter_triangle (a b c A B C : ℝ) (h₁ : a * real.sin A - c * real.sin C = (a - b) * real.sin B) 
    (h₂ : c = real.sqrt 3) (h₃ : 0 < C) (h₄ : C = real.pi / 3) : 
    let perimeter := a + b + c in
    perimeter = 3 * real.sqrt 3 :=
by
    sorry

end angle_C_magnitude_max_perimeter_triangle_l375_375638


namespace cone_height_l375_375759

noncomputable theory

def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_height :
  ∃ h : ℝ, cone_volume 10 h = 2199.114857512855 ∧ h ≈ 21 :=
by
  sorry

end cone_height_l375_375759


namespace scientific_notation_correct_l375_375473

theorem scientific_notation_correct {n : ℕ} (h : n = 680000000) : n = 6.8 * 10^8 := by
  sorry

end scientific_notation_correct_l375_375473


namespace neg_sin_le_proof_l375_375323

noncomputable theory

open Real

theorem neg_sin_le_proof : ¬(∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 := 
by
  sorry

end neg_sin_le_proof_l375_375323


namespace joan_dimes_l375_375249

theorem joan_dimes (initial_dimes spent_dimes remaining_dimes : ℕ) 
    (h1 : initial_dimes = 5) (h2 : spent_dimes = 2) 
    (h3 : remaining_dimes = initial_dimes - spent_dimes) : 
    remaining_dimes = 3 := 
sorry

end joan_dimes_l375_375249


namespace repairs_cost_l375_375708

def purchase_price : ℝ := 900
def selling_price : ℝ := 1500
def gain_percentage : ℝ := 25 / 100

theorem repairs_cost :
  ∃ R : ℝ, selling_price = 1.25 * (purchase_price + R) ∧ R = 300 :=
by
  sorry

end repairs_cost_l375_375708


namespace polynomial_integer_root_unique_l375_375531

theorem polynomial_integer_root_unique (a : ℕ) (h : a > 0) 
  (p : ℕ → ℕ → ℕ := λ a x, x^2 - a * x + a) :
  (∃ x : ℕ, p a x = 0) → a = 4 := 
sorry

end polynomial_integer_root_unique_l375_375531


namespace cube_root_last_three_digits_removed_l375_375678

theorem cube_root_last_three_digits_removed (n a : ℕ) (h1 : n > 0) (h2 : remove_last_three_digits n = a) (h3 : a^3 = n) : n = 32768 := sorry

end cube_root_last_three_digits_removed_l375_375678


namespace dot_product_sum_l375_375261

variables (u v w : EuclideanSpace ℝ (Fin 3))

theorem dot_product_sum
  (h1: ‖u‖ = 5)
  (h2: ‖v‖ = 12)
  (h3: ‖w‖ = 13)
  (h4: u + v + w = 0) :
  u ⬝ v + u ⬝ w + v ⬝ w = -169 :=
sorry

end dot_product_sum_l375_375261


namespace light_bulb_problem_l375_375343

theorem light_bulb_problem :
  let Bulbs := {0, 1, 2, 3, 4, 5, 6} -- 7 bulbs indexed from 0 to 6
  let AtLeast3 (s : Finset ℕ) := s.card ≥ 3 -- At least 3 bulbs lit
  let NoAdjacent (s : Finset ℕ) := ∀ x ∈ s, ∀ y ∈ s, x ≠ y → |x - y| ≠ 1 -- No adjacent bulbs lit
  ∃ s : Finset ℕ, AtLeast3 s ∧ NoAdjacent s ∧ s.card = 11 -- 11 ways to light up the bulbs
:= by
  sorry

end light_bulb_problem_l375_375343


namespace find_function_l375_375510

theorem find_function (f : ℝ → ℝ) :
  (∀ x : ℝ, f(2 * x + 1) = 4 * x ^ 2 + 14 * x + 7) →
  ∀ x : ℝ, f(x) = x^2 + 5 * x + 1 :=
by sorry

end find_function_l375_375510


namespace unique_integer_solution_range_l375_375602

theorem unique_integer_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x + 3 > 5) ∧ (x - a ≤ 0) → (x = 2)) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end unique_integer_solution_range_l375_375602


namespace a_not_96_l375_375350

-- Given data set
def dataSet : List ℕ := [80, 80, 81, 82, 83, 84, 84, 85, 86, 87, 89, 90, 93, 95, 95, 97, 100]

-- Including 'a' in the data set
def fullDataSet (a : ℕ) : List ℕ := a :: dataSet

-- Definition for upper quartile position
def upperQuartilePosition (n : ℕ) : ℕ := (3 * n + 3) / 4

-- Proof statement
theorem a_not_96 (a : ℕ) (h : a ∈ fullDataSet a) (hq : List.nthLe (fullDataSet a).sorted (upperQuartilePosition 18 - 1) (by sorry) = a) :
  ¬a = 96 := sorry


end a_not_96_l375_375350


namespace solution_set_inequality_range_of_t_l375_375156

noncomputable def f (x : ℝ) : ℝ := |x| - 2 * |x + 3|

-- Problem (1)
theorem solution_set_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | -4 ≤ x ∧ x ≤ - (8 / 3) } :=
by
  sorry

-- Problem (2)
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x - |3 * t - 2| ≥ 0) ↔ (- (1 / 3) ≤ t ∧ t ≤ 5 / 3) :=
by
  sorry

end solution_set_inequality_range_of_t_l375_375156


namespace find_b_l375_375592

theorem find_b (a b h : ℕ) (hb_pos : 0 < b) (hh_pos : 0 < h) (hb_lt_h : b < h)
               (cond : b^2 + h^2 = b * (a + h) + a * h) :
  b = 2 :=
begin
  sorry
end

end find_b_l375_375592


namespace rectangle_area_l375_375440

theorem rectangle_area (r : ℝ) (w l : ℝ) (h_radius : r = 7) 
  (h_ratio : l = 3 * w) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end rectangle_area_l375_375440


namespace fraction_to_decimal_l375_375027

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375027


namespace local_minimum_f_at_1_for_k_2_l375_375552

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := 
  (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_at_1_for_k_2 : 
  (∀ k : ℕ, k = 2 → ∃ ε > 0, ∀ x : ℝ, abs(x - 1) < ε → f 2 x ≥ f 2 1) := 
sorry

end local_minimum_f_at_1_for_k_2_l375_375552


namespace fraction_to_decimal_l375_375069

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l375_375069


namespace trajectory_of_P_line_satisfying_condition_l375_375126

noncomputable def fixed_points : (ℝ × ℝ) × (ℝ × ℝ) := ((-Real.sqrt 2, 0), (Real.sqrt 2, 0))

def slope_product_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  (y / (x + Real.sqrt 2)) * (y / (x - Real.sqrt 2)) = -1 / 2

def trajectory_equation (x y : ℝ) : Prop :=
  (x^2 / 2) + y^2 = 1

theorem trajectory_of_P (P : ℝ × ℝ) :
  slope_product_condition P → trajectory_equation P.1 P.2 := by
  sorry

def line_eq (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

def dist_MN_condition (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  Real.sqrt ((1 + (y2 - y1) / (x2 - x1))^2 * (x2 - x1)^2) = (4 * Real.sqrt 2) / 3

theorem line_satisfying_condition (k : ℝ) :
  ∃ M N : ℝ × ℝ, trajectory_equation M.1 M.2 ∧ trajectory_equation N.1 N.2 ∧
  line_eq k M.1 M.2 ∧ line_eq k N.1 N.2 ∧ dist_MN_condition M N →
  (line_eq 1 M.1 M.2 ∨ line_eq (-1) M.1 M.2) := by
  sorry

end trajectory_of_P_line_satisfying_condition_l375_375126


namespace boat_speed_in_still_water_l375_375822

theorem boat_speed_in_still_water  (b s : ℝ) (h1 : b + s = 13) (h2 : b - s = 9) : b = 11 :=
sorry

end boat_speed_in_still_water_l375_375822


namespace set_intersection_is_result_l375_375579

def set_A := {x : ℝ | 1 < x^2 ∧ x^2 < 4 }
def set_B := {x : ℝ | x ≥ 1}
def result_set := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_is_result : (set_A ∩ set_B) = result_set :=
by sorry

end set_intersection_is_result_l375_375579


namespace limit_of_difference_quotient_l375_375537

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

theorem limit_of_difference_quotient :
  (Real.Lim (fun h : ℝ => (f (2 + h) - f 2) / h) 0) = - 1 / 4 := by
  sorry

end limit_of_difference_quotient_l375_375537


namespace circle_circumference_l375_375311

theorem circle_circumference (A : ℝ) (hA : A = 196 * Real.pi) : ∃ C : ℝ, C = 2 * Real.pi * 14 := by
  have r_sq : ℝ := 196
  have r : ℝ := Real.sqrt r_sq
  have r_value : r = 14 := by
    simp [Real.sqrt_eq_rpow, Real.sqrt_sq_eq_abs]
    sorry
  use 2 * Real.pi * r
  simp [r_value]
  sorry

end circle_circumference_l375_375311


namespace avg_of_last_three_l375_375313

-- Define the conditions given in the problem
def avg_5 : Nat := 54
def avg_2 : Nat := 48
def num_list_length : Nat := 5
def first_two_length : Nat := 2

-- State the theorem
theorem avg_of_last_three
    (h_avg5 : 5 * avg_5 = 270)
    (h_avg2 : 2 * avg_2 = 96) :
  (270 - 96) / 3 = 58 :=
sorry

end avg_of_last_three_l375_375313


namespace maximum_correct_questions_maximum_correct_questions_correct_answer_l375_375226

theorem maximum_correct_questions (c w b : ℕ) 
  (h1 : c + w + b = 25) 
  (h2 : 4 * c - w = 65) : 
  c ≤ 18 :=
by {
  have h3 := calc
    4 * c - w = 65 : sorry,
  -- From equation 4c - w = 65
  -- w = 4c - 65
  have hw : w = 4 * c - 65 := sorry,

  -- From equation c + w + b = 25
  -- c + (4c - 65) + b = 25
  -- 5c - 65 + b = 25
  -- b = 90 - 5c
  have hb : b = 90 - 5 * c := sorry,

  -- ensure b >= 0
  have hb_nonneg : 90 - 5 * c ≥ 0 := sorry,
  -- 90 - 5c ≥ 0
  -- c ≤ 18
  show c ≤ 18,
end,

-- With the correct answer being with maximum value of c being 18
theorem maximum_correct_questions_correct_answer : 
  ∃ c w b : ℕ, c = 18 ∧ c + w + b = 25 ∧ 4 * c - w = 65 :=
by {
  use 18,
  use 7,
  use 0,
  split; [refl, split; [exact rfl, exact rfl]],

  have h3 : 4 * 18 - 7 = 65 := by norm_num,
  exact h3,
}

end maximum_correct_questions_maximum_correct_questions_correct_answer_l375_375226


namespace tetrahedron_vector_relation_l375_375112

theorem tetrahedron_vector_relation
  {A B C D O : Type*}
  [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ O]
  (V_O_BCD V_O_ACD V_O_ABD V_O_ABC : ℝ)
  (vec_OA : vector ℝ A) (vec_OB : vector ℝ B) (vec_OC : vector ℝ C) (vec_OD : vector ℝ D)
  (h_inside_tetrahedron : O ⊆ convex_hull ℝ (set.insert A (set.insert B (set.insert C (set.singleton D)))))
  : V_O_BCD • vec_OA + V_O_ACD • vec_OB + V_O_ABD • vec_OC + V_O_ABC • vec_OD = 0 := 
sorry

end tetrahedron_vector_relation_l375_375112


namespace interior_surface_area_is_812_l375_375722

/-- Surface area of the interior of the box formed by removing square corners of 7 units from a 
    28 by 36 unit rectangle and folding up the remaining flaps -/
noncomputable def surface_area_of_interior_of_box : ℕ :=
let length := 28
let width := 36
let cut := 7
let new_length := length - 2 * cut
let new_width := width - 2 * cut
let height := cut
let base_area := new_length * new_width
let side_area := 2 * (new_length * height + new_width * height)
in base_area + side_area

theorem interior_surface_area_is_812 :
  surface_area_of_interior_of_box = 812 := 
sorry

end interior_surface_area_is_812_l375_375722


namespace trigonometric_identity_eq_tan4alpha_l375_375817

open Real

noncomputable def cos4Alphaminus9div2pi (α : ℝ) : ℝ := cos(4 * α - 9 / 2 * π)
noncomputable def ctg5div4piPlus2alpha (α : ℝ) : ℝ := 1 / tan (5 / 4 * π + 2 * α)
noncomputable def cos5div2piPlus4alpha (α : ℝ) : ℝ := cos(5 / 2 * π + 4 * α)
noncomputable def tan4alpha (α : ℝ) : ℝ := tan(4 * α)

theorem trigonometric_identity_eq_tan4alpha (α : ℝ) :
  cos4Alphaminus9div2pi α / (ctg5div4piPlus2alpha α * (1 - cos5div2piPlus4alpha α)) = tan4alpha α := 
by
  sorry

end trigonometric_identity_eq_tan4alpha_l375_375817


namespace binom_8_2_eq_28_l375_375489

open Nat

theorem binom_8_2_eq_28 : Nat.choose 8 2 = 28 := by
  sorry

end binom_8_2_eq_28_l375_375489


namespace cyclic_quad_incircle_tangent_l375_375301

theorem cyclic_quad_incircle_tangent
  (A B C D : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  (cyclic_quad : cyclic_quad A B C D)
  (BC_eq_CD : B = C → C = D)
  (omega : circle C)
  (tangent_to_BD : tangent omega.to_circle B D)
  (I : center (incircle (triangle A B D))) :
  is_tangent (line.through I (parallel to A B)) omega :=
sorry

end cyclic_quad_incircle_tangent_l375_375301


namespace even_degree_poly_ge_deriv_l375_375093

noncomputable def exists_poly_n (n : ℕ) : Prop :=
  ∃ (f : ℝ → ℝ), polynomial.degree f = n ∧ ∀ x : ℝ, f x ≥ polynomial.derivative f x

theorem even_degree_poly_ge_deriv (n : ℕ) :
  (∃ f : ℝ → ℝ, polynomial.degree f = n ∧ ∀ x : ℝ, f x ≥ polynomial.derivative f x) ↔ even n := by
  sorry

end even_degree_poly_ge_deriv_l375_375093


namespace mixed_fractions_product_l375_375862

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375862


namespace ball_bounce_height_l375_375429

theorem ball_bounce_height (n : ℕ) (h₀ : ℝ) (r : ℝ) :
  (h₀ = 20) → (r = 2/3) → (h₀ * r^n < 2) ↔ (n = 6) :=
by
  assume h₀_eq r_eq
  sorry

end ball_bounce_height_l375_375429


namespace probability_perfect_square_sum_is_7_div_36_l375_375307

-- Define the sample space for rolling two dice
def sample_space : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 1 7) (Finset.range 1 7)

-- Define the event of interest (sum is a perfect square)
def is_perfect_square_sum (x y : ℕ) : Prop :=
  let sum := x + y
  sum = 4 ∨ sum = 9

-- Calculate the number of successful outcomes
def num_successful_outcomes : ℕ := 
  Finset.card (sample_space.filter (λ (xy : ℕ × ℕ), is_perfect_square_sum xy.1 xy.2))

-- Calculate the total number of outcomes
def num_total_outcomes : ℕ := Finset.card sample_space

-- The probability is given by the ratio of successful outcomes to total outcomes
noncomputable def probability : ℚ := num_successful_outcomes / num_total_outcomes

-- Assert the probability
theorem probability_perfect_square_sum_is_7_div_36 : probability = 7 / 36 := by sorry

end probability_perfect_square_sum_is_7_div_36_l375_375307


namespace find_k_l375_375189

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l375_375189


namespace a_n_formula_b_n_sum1_b_n_sum2_C_n_sum_l375_375534

noncomputable def seq_sum (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S_n n = (1 / 4) * (a_n n)^2 + (1 / 2) * (a_n n) + (1 / 4)

def a_n (n : ℕ) : ℝ := 2 * n - 1

theorem a_n_formula (S_n : ℕ → ℝ) (a_n_prop : seq_sum a_n S_n) : 
  ∀ (n : ℕ), n > 0 → a_n n = 2 * n - 1 :=
sorry

noncomputable def b_n (q : ℝ) (a_n b_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a_n n + b_n n = q^(n-1)

theorem b_n_sum1 (q : ℝ) (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) 
  (geom_seq : b_n q a_n b_n) : ∀ (n : ℕ), q = 1 → 
  ∑ i in Finset.range n, b_n (i+1) = -n^2 + n :=
sorry

theorem b_n_sum2 (q : ℝ) (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) 
  (geom_seq : b_n q a_n b_n) : ∀ (n : ℕ), q ≠ 1 → 
  ∑ i in Finset.range n, b_n (i+1) = -n^2 + (1 - q^n) / (1 - q) :=
sorry

noncomputable def f (a_n : ℕ → ℝ) : ℕ → ℝ
| n => if n % 2 = 1 then a_n n else f (n / 2)

noncomputable def C_n (a_n : ℕ → ℝ) : ℕ → ℝ :=
λ n, f a_n (2^n + 4)

theorem C_n_sum (a_n : ℕ → ℝ) : ∀ (n : ℕ), n > 0 → 
  (if n = 1 then ∑ i in Finset.range 1, λ i, C_n a_n i = 5 
   else ∑ i in Finset.range n, C_n a_n i = 2^n + n) :=
sorry

end a_n_formula_b_n_sum1_b_n_sum2_C_n_sum_l375_375534


namespace range_g_l375_375662

open Real

def g (x : ℝ) : ℝ := (arccos x)^4 + (arcsin x)^4

theorem range_g : 
  ∀ x, x ∈ Icc (-1:ℝ) 1 → 
  ∀ y, y = g x → 
  y ∈ Icc (π^4 / 16) (3 * π^4 / 32) :=
sorry

end range_g_l375_375662


namespace find_OP_length_l375_375663

-- Define the problem's parameters.
noncomputable def diameter_of_circle (A B O : Point) (radius : ℝ) : Prop :=
  -- A, B, and O form the diameter of the circle with the given radius.
  dist O A = radius ∧ dist O B = radius ∧ dist A B = 2 * radius

noncomputable def angle_relation (A C B D Q O : Point) (x : ℝ) :=
  -- C and D are points on the circle such that AC and BD intersect at Q inside the circle
  -- The angle relationship given in the problem.
  ∠ AQB = 2 * ∠ COD

-- Main theorem statement
theorem find_OP_length (A B O C D Q P : Point) (r : ℝ) :
  -- Given the conditions
  diameter_of_circle A B O r ∧
  (∃ Q, intersects AC BD Q) ∧
  angle_relation A C B D Q O (π / 3) ∧ -- Since x = 60 degrees or π/3 radians
  tangent_through C D P →
  -- Prove the length of OP
  dist O P = 2 * r / sqrt 3 :=
sorry

end find_OP_length_l375_375663


namespace four_digit_swap_square_l375_375212

theorem four_digit_swap_square (a b : ℤ) (N M : ℤ) : 
  N = 1111 * a + 123 ∧ 
  M = 1111 * a + 1023 ∧ 
  M = b ^ 2 → 
  N = 3456 := 
by sorry

end four_digit_swap_square_l375_375212


namespace mixed_fraction_product_l375_375895

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375895


namespace polynomial_inequality_l375_375287

/-- Given a polynomial P(x) with real coefficients and real, pairwise distinct roots,
    prove that a_i^2 > ((n - i + 1) / (n - i)) * ((i + 1) / i) * a_(i - 1) * a_(i + 1)
    for all i from 1 to n - 1. -/
theorem polynomial_inequality
  (n : ℕ)
  (a : ℕ → ℝ)
  (P : polynomial ℝ)
  (hP : P.coeff = λ i, a i)
  (roots_real : ∀ x ∈ P.roots, x ∈ ℝ)
  (roots_distinct : P.roots.nodup) :
  ∀ i : ℕ, 1 ≤ i ∧ i < n - 1 →
  a i ^ 2 > ((n - i + 1) / (n - i) * (i + 1) / i) * a (i - 1) * a (i + 1) :=
begin
  sorry
end

end polynomial_inequality_l375_375287


namespace functions_are_equal_l375_375476

theorem functions_are_equal (x : ℝ) (h : x ≠ 0) : (λ x, 1 / |x|) x = (λ x, 1 / real.sqrt (x^2)) x :=
by
  have h1 : |x| = real.sqrt (x^2) := sorry
  rw [h1]
  simp

end functions_are_equal_l375_375476


namespace fraction_to_decimal_l375_375080

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375080


namespace sum_of_inscribed_circle_radii_equal_l375_375286

-- Prove that the sum of the radii of circles inscribed in triangles ABC and ACD
-- is equal to the sum of the radii of circles inscribed in triangles BCD and BDA
theorem sum_of_inscribed_circle_radii_equal 
  (A B C D : Point)
  (h1 : InscribedQuadrilateral A B C D)
  (O1 : CenterOfInscribedCircle A B C)
  (O2 : CenterOfInscribedCircle B C D)
  (O3 : CenterOfInscribedCircle C D A)
  (O4 : CenterOfInscribedCircle D A B)
  (r1 r2 r3 r4 : ℝ)
  (r1_def : RadiusOfInscribedCircle A B C = r1)
  (r2_def : RadiusOfInscribedCircle B C D = r2)
  (r3_def : RadiusOfInscribedCircle C D A = r3)
  (r4_def : RadiusOfInscribedCircle D A B = r4) : 
  r1 + r3 = r2 + r4 :=
sorry

end sum_of_inscribed_circle_radii_equal_l375_375286


namespace diagonals_and_tangency_lines_intersect_at_a_point_l375_375734

theorem diagonals_and_tangency_lines_intersect_at_a_point
  (A B C D P Q R S : Point)
  (circumcircle : Circle)
  (incircle : Circle)
  (h1 : QuadrilateralInscribed ABCD circumcircle)
  (h2 : TangentToCircle incircle A P ∧ TangentToCircle incircle B P)
  (h3 : TangentToCircle incircle B Q ∧ TangentToCircle incircle C Q)
  (h4 : TangentToCircle incircle C R ∧ TangentToCircle incircle D R)
  (h5 : TangentToCircle incircle D S ∧ TangentToCircle incircle A S) :
  ConcurrentLines (LineThrough A C) (LineThrough B D) (LineThrough P Q) (LineThrough R S) :=
  sorry

end diagonals_and_tangency_lines_intersect_at_a_point_l375_375734


namespace fraction_meaningful_iff_nonzero_l375_375777

theorem fraction_meaningful_iff_nonzero (x : ℝ) : (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 :=
by sorry

end fraction_meaningful_iff_nonzero_l375_375777


namespace like_terms_exponents_l375_375587

variable {a b : ℝ}
variable (m n : ℕ)

def are_like_terms (term1 term2 : ℝ) : Prop :=
  term1 = (1/3) * (a ^ 2) * (b ^ m) ∧ term2 = (-1/2) * (a ^ n) * (b ^ 4)

theorem like_terms_exponents (h : are_like_terms ((1/3) * (a ^ 2) * (b ^ m)) ((-1/2) * (a ^ n) * (b ^ 4))) :
  m = 4 ∧ n = 2 :=
by
  cases h with h1 h2
  sorry

end like_terms_exponents_l375_375587


namespace general_formula_a_general_formula_b_sum_of_cn_l375_375129

def a (n : ℕ) : ℕ := 2 * n - 1
def S (n : ℕ) : ℕ := ∑ k in Finset.range n, a (k + 1) -- assuming sum of first n terms
def b : ℕ → ℕ 
| 1 := 1
| (n+1) := 2 * (b n) + 1
def c (n : ℕ) : ℕ := a n * (b n + 1)
def T (n : ℕ) : ℕ := ∑ k in Finset.range n, c (k + 1)

theorem general_formula_a (n : ℕ) (hn : 0 < n) : a n = 2 * n - 1 := sorry
theorem general_formula_b (n : ℕ) (hn : 0 < n) : b n = 2^n - 1 := sorry
theorem sum_of_cn (n : ℕ) (hn : 0 < n) : T n = (2 * n - 3) * 2^(n + 1) + 6 := sorry

end general_formula_a_general_formula_b_sum_of_cn_l375_375129


namespace sum_of_two_longest_altitudes_l375_375923

/- Define the triangle sides and the question -/
def sides := (6 : ℝ, 8 : ℝ, 10 : ℝ)

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

/- Define the altitudes to the sides -/
def altitude_to_side (a b c : ℝ) (h : is_right_triangle a b c) : ℝ :=
  if b = c then a else if a = c then b else 0

/- Prove the sum of the two longest altitudes -/
theorem sum_of_two_longest_altitudes :
  let tr := (6 : ℝ, 8 : ℝ, 10 : ℝ) in
  ∃ (a b c : ℝ), tr = (a, b, c) ∧ is_right_triangle a b c ∧
  altitude_to_side a b c sorry + altitude_to_side b a c sorry = 14 :=
begin
  sorry
end

end sum_of_two_longest_altitudes_l375_375923


namespace productOfCommonRoots_l375_375522

noncomputable def commonRootsProduct (C D : ℝ) (p q : ℝ) : ℝ :=
  if h1 : p^4 - 3*p^3 + C*p + 24 = 0 ∧ q^4 - 3*q^3 + C*q + 24 = 0
      ∧ p^4 - D*p^3 + 4*p^2 + 72 = 0 ∧ q^4 - D*q^3 + 4*q^2 + 72 = 0 then
    p * q
  else 0

theorem productOfCommonRoots 
  (C D : ℝ) (h1 : ∃ (p q : ℝ), p ≠ q ∧ p^4 - 3*p^3 + C*p + 24 = 0 ∧ q^4 - 3*q^3 + C*q + 24 = 0
  ∧ p^4 - D*p^3 + 4*p^2 + 72 = 0 ∧ q^4 - D*q^3 + 4*q^2 + 72 = 0) :
  ∃ (a b c : ℕ), a * real.root b c = 1 / 3 ∧ a + b + c = 5 :=
begin
  sorry
end

end productOfCommonRoots_l375_375522


namespace find_x_l375_375132

theorem find_x (x : ℝ) (h : 2 * x - 1 = -( -x + 5 )) : x = -6 :=
by
  sorry

end find_x_l375_375132


namespace dice_sum_probability_l375_375604

theorem dice_sum_probability :
  let probability_sum_17 (dice : list ℕ) := (dice.sum = 17 ∧ dice.length = 3) in
  let total_outcomes := 6^3 in
  let favorable_outcomes := 3 in -- (6,6,5), (6,5,6), and (5,6,6)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 72 :=
by
  sorry

end dice_sum_probability_l375_375604


namespace mixed_fraction_product_example_l375_375885

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375885


namespace trigonometric_identity_m_n_find_m_plus_n_l375_375542

/-- Given the condition \(\frac{1+\sin x}{\cos x} = \frac{22}{7}\), prove that \(\frac{1+\cos x}{\sin x} = \frac{29}{15}\) and hence \(m+n = 44\). -/
theorem trigonometric_identity_m_n :
  (∀ x : ℝ, (1 + Real.sin x) / Real.cos x = 22 / 7 → (1 + Real.cos x) / Real.sin x = 29 / 15) :=
by
  intros
  sorry

/-- With the given condition and the deduced equality, prove that \(m + n = 44\). -/
theorem find_m_plus_n :
  (∀ x : ℝ, (1 + Real.sin x) / Real.cos x = 22 / 7 → (∃ m n : ℕ, (m : ℝ) / (n : ℝ) = (1 + Real.cos x) / Real.sin x ∧ m + n = 44)) :=
by
  intros x hx
  use 29, 15
  split
  { -- Prove m / n = (1 + cos x) / sin x
    lib
    sorry 
  },
  { -- Prove m + n = 44
    lib
    norm_num,
    sorry 
  }

end trigonometric_identity_m_n_find_m_plus_n_l375_375542


namespace total_number_of_books_l375_375771

theorem total_number_of_books (history_books geography_books math_books : ℕ)
  (h1 : history_books = 32) (h2 : geography_books = 25) (h3 : math_books = 43) :
  history_books + geography_books + math_books = 100 :=
by
  -- the proof would go here but we use sorry to skip it
  sorry

end total_number_of_books_l375_375771


namespace compare_log_values_l375_375116

noncomputable def log3_2 : ℝ := Real.logBase 3 2
noncomputable def log5_1div2 : ℝ := Real.logBase 5 (1 / 2)
noncomputable def log2_3 : ℝ := Real.logBase 2 3

theorem compare_log_values : log2_3 > log3_2 ∧ log3_2 > log5_1div2 := by
  sorry

end compare_log_values_l375_375116


namespace circle_problem_l375_375787

theorem circle_problem (P : ℝ × ℝ) (S : ℝ × ℝ) (k : ℝ)
  (hP : P = (6, 8))
  (hS : S = (0, k))
  (radius_P : ∃ r : ℝ, r = real.sqrt ((P.1)^2 + (P.2)^2))
  (QR : ℝ)
  (hQR: QR = 5) : k = 5 :=
sorry

end circle_problem_l375_375787


namespace fraction_to_decimal_l375_375021

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375021


namespace restore_fractions_l375_375887

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375887


namespace watermelonJuicePercentage_l375_375447

-- Definitions based on the conditions
def totalDrink : ℝ := 140
def orangeJuicePercentage : ℝ := 0.15
def grapeJuiceAmount : ℝ := 35

-- The tuple (question, conditions, correct answer) transformed into a Lean 4 statement
theorem watermelonJuicePercentage :
  let orangeJuiceAmount := orangeJuicePercentage * totalDrink in
  let watermelonJuiceAmount := totalDrink - (orangeJuiceAmount + grapeJuiceAmount) in
  ((watermelonJuiceAmount / totalDrink) * 100) = 60 :=
by
  sorry

end watermelonJuicePercentage_l375_375447


namespace fraction_to_decimal_equiv_l375_375016

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375016


namespace color_change_probability_l375_375919

theorem color_change_probability :
  let cycle_duration := 45 + 5 + 35,
      total_favorable_duration := 4 + 4 + 4 in
  (total_favorable_duration : ℝ) / cycle_duration = (12 : ℝ) / (85 : ℝ) :=
by
  let cycle_duration := 45 + 5 + 35
  let total_favorable_duration := 4 + 4 + 4
  sorry

end color_change_probability_l375_375919


namespace triangle_inequality_l375_375241

theorem triangle_inequality (a b c R r : ℝ) (ha : a ≥ b) (hb : a ≥ c)
    (R_eq : R = (a * b * c) / (4 * (sqrt (s * (s - a) * (s - b) * (s - c)))))
    (r_eq : r = (sqrt (s * (s - a) * (s - b) * (s - c))) / s)
    (mu_a_eq : ∀ A B C : ℝ, μ_a = sqrt((2 * B^2 + 2 * C^2 - A^2) / 4))
    (h_a_eq : ∀ A Ha, h_a = (2 * sqrt (s * (s - a) * (s - b) * (s - c))) / A): 
    R / (2 * r) >= μ_a / h_a :=
  sorry

end triangle_inequality_l375_375241


namespace fraction_to_decimal_l375_375038

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375038


namespace restore_original_problem_l375_375870

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375870


namespace quadratic_two_distinct_real_roots_l375_375598

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end quadratic_two_distinct_real_roots_l375_375598


namespace max_rectangle_area_l375_375460

theorem max_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 120) : l * w ≤ 900 :=
by 
  sorry

end max_rectangle_area_l375_375460


namespace average_monthly_growth_rate_correct_l375_375611

theorem average_monthly_growth_rate_correct:
  (∃ x : ℝ, 30000 * (1 + x)^2 = 36300) ↔ 3 * (1 + x)^2 = 3.63 := 
by {
  sorry -- proof placeholder
}

end average_monthly_growth_rate_correct_l375_375611


namespace three_digit_with_five_is_divisible_by_five_l375_375470

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_with_five_is_divisible_by_five (M : ℕ) :
  is_three_digit M ∧ ends_in_five M → divisible_by_five M :=
by
  sorry

end three_digit_with_five_is_divisible_by_five_l375_375470


namespace parallelogram_area_gt_1_l375_375700

theorem parallelogram_area_gt_1
  (P : Type) [parallelogram P]
  (h_vertices_lattice : ∀ v ∈ vertices P, ∃ x y : ℤ, v = (x, y))
  (h_additional_lattice_point : ∃ p ∈ (interior P ∪ boundary P), p ∉ vertices P) :
  area P > 1 := 
sorry

end parallelogram_area_gt_1_l375_375700


namespace fraction_to_decimal_l375_375023

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375023


namespace find_k_l375_375186

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l375_375186


namespace fraction_equals_decimal_l375_375078

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l375_375078


namespace grid_number_sum_constant_l375_375491

theorem grid_number_sum_constant :
  ∀ (grid : Matrix ℕ 5 5), 
  (∀ (move : ℕ), is_valid_move grid move) → -- Assumes is_valid_move checks the validity of the move based on the problem's rules
  (∃ sum_val : ℕ, sum_val = 40 ∧ 
    ∑ i j, grid i j = sum_val) :=
sorry

end grid_number_sum_constant_l375_375491


namespace find_multiple_l375_375765

-- Given conditions as definitions
def smaller_number := 21
def sum_of_numbers := 84

-- Definition of larger number being a multiple of the smaller number
def is_multiple (k : ℤ) (a b : ℤ) : Prop := b = k * a

-- Given that one number is a multiple of the other and their sum
def problem (L S : ℤ) (k : ℤ) : Prop := 
  is_multiple k S L ∧ S + L = sum_of_numbers

theorem find_multiple (L S : ℤ) (k : ℤ) (h1 : problem L S k) : k = 3 := by
  -- Proof omitted
  sorry

end find_multiple_l375_375765


namespace incorrect_statements_B_D_l375_375409

-- Definitions from problem
def statementA : Prop :=
  ∀ (s : Sphere), ∃ (p : Point), (p ∈ s.center) → (crossSection p s).radius = s.radius

def statementB : Prop :=
  ∀ (solid : GeometricSolid), (solid.parallel_faces = 2 ∧ solid.other_faces.all (== parallelogram)) → solid.isPrism

def statementC : Prop :=
  ∀ (tetra : Tetrahedron), tetra.regular → tetra.lateral_faces.all (== equilateral_triangle)

def statementD : Prop :=
  ∀ (hex : Hexahedron), 
    (hex.parallel_faces = 2 ∧ hex.other_faces.all (== trapezoid)) → hex.isFrustum

-- Formalizing the proof problem
theorem incorrect_statements_B_D : ¬ statementB ∧ ¬ statementD := by
  sorry

end incorrect_statements_B_D_l375_375409


namespace y_eq_x_plus_2_l375_375675

variable (p : ℝ)

def x : ℝ := 1 + 2^p - 2^(-p)
def y : ℝ := 1 + 2^p + 2^(-p)

theorem y_eq_x_plus_2 : y p = x p + 2 :=
by sorry

end y_eq_x_plus_2_l375_375675


namespace mixed_fractions_product_l375_375868

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375868


namespace restore_original_problem_l375_375874

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l375_375874


namespace find_k_l375_375176

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l375_375176


namespace find_k_l375_375187

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l375_375187


namespace floor_is_greatest_integer_l375_375954

-- Define a function that returns the floor of a real number
def floor_function (x : ℝ) : ℤ := int.floor x

-- Define the statement to be proved
theorem floor_is_greatest_integer (x : ℝ) : ∃ (y : ℤ), y = floor_function x ∧ (y ≤ x ∧ ∀ z : ℤ, z ≤ x → z ≤ y) := by
  sorry

end floor_is_greatest_integer_l375_375954


namespace mixed_fractions_product_l375_375865

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375865


namespace translate_complex_number_l375_375922

/-- A translation in the complex plane shifts the point 1 - 3i to 4 + 2i. Find the complex number
    that the same translation takes 3 - i to is 6 + 4i. -/
theorem translate_complex_number :
  ∀ (w : ℂ), (w = (4 + 2 * complex.i) - (1 - 3 * complex.i)) →
  ((3 - complex.i) + w = 6 + 4 * complex.i) :=
by
  intros
  sorry

end translate_complex_number_l375_375922


namespace part1_tan_ratio_part2_cos_2alpha_and_sum_l375_375421

-- Part 1

theorem part1_tan_ratio
  (α β : Real)
  (h1 : sin (α + β) = 1 / 2)
  (h2 : sin (α - β) = 1 / 3) :
  tan α / tan β = 5 := 
sorry

-- Part 2

theorem part2_cos_2alpha_and_sum
  (α β : Real)
  (α_obtuse : π / 2 < α ∧ α < π)
  (h3 : sin α / cos α = 2)
  (h4 : cos β = -7 * sqrt 2 / 10)
  (h5 : 0 < β ∧ β < π) :
  cos (2 * α) = -3 / 5 ∧ 2 * α + β = 9 * π / 4 := 
sorry

end part1_tan_ratio_part2_cos_2alpha_and_sum_l375_375421


namespace strictly_increasing_interval_l375_375498

open Set

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (-x^2 + 4 * x)

theorem strictly_increasing_interval : ∀ x ∈ (Ici 2 : Set ℝ), StrictMono f :=
by
  intros x hx
  let t := -x^2 + 4 * x
  have decreasing_t : ∀ y, 2 ≤ y → t' deriving Use the properties of quadratic functions,
  apply StrictMonoOn_of_deriv_pos,
  ...
  sorry

end strictly_increasing_interval_l375_375498


namespace factorize_expression_l375_375969

theorem factorize_expression (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 :=
  sorry

end factorize_expression_l375_375969


namespace ratio_of_area_to_breadth_l375_375312

variable (l b : ℕ)

theorem ratio_of_area_to_breadth 
  (h1 : b = 14) 
  (h2 : l - b = 10) : 
  (l * b) / b = 24 := by
  sorry

end ratio_of_area_to_breadth_l375_375312


namespace gcd_lcm_product_360_l375_375383

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375383


namespace lions_win_championship_l375_375730

def probability_lions_win_series (p : ℚ) : ℚ :=
    ∑ k in finset.range 5, (nat.choose (4 + k) k) * p^5 * (1 - p)^k

theorem lions_win_championship : 
  probability_lions_win_series (4/7) ≈ 0.71 :=
by
    sorry

end lions_win_championship_l375_375730


namespace fraction_to_decimal_l375_375022

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375022


namespace restore_original_problem_l375_375906

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375906


namespace greatest_gcd_of_6Tn_and_n_plus_1_l375_375527

-- Define the nth triangular number
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Define the gcd function using Lean's library
axiom gcd : ℕ → ℕ → ℕ

-- The main proposition to prove
theorem greatest_gcd_of_6Tn_and_n_plus_1 (n : ℕ) (h : n > 0) :
  let T_n := triangular_number n
  let value := gcd (6 * T_n) (n + 1)
  value ≤ 3 :=
by
  sorry

end greatest_gcd_of_6Tn_and_n_plus_1_l375_375527


namespace initial_cows_l375_375446

theorem initial_cows (x : ℕ) (h : (3 / 4 : ℝ) * (x + 5) = 42) : x = 51 :=
by
  sorry

end initial_cows_l375_375446


namespace gcd_ropes_lengths_l375_375474

theorem gcd_ropes_lengths :
  ∀ (a b c d : ℕ), a = 48 → b = 60 → c = 72 → d = 120 → Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 12 :=
by
  intros a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  rfl
  sorry

end gcd_ropes_lengths_l375_375474


namespace quadratic_two_distinct_real_roots_l375_375599

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end quadratic_two_distinct_real_roots_l375_375599


namespace fraction_to_decimal_l375_375026

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375026


namespace flight_duration_l375_375250

theorem flight_duration (departure_time arrival_time : ℕ) (time_difference : ℕ) (h m : ℕ) (m_bound : 0 < m ∧ m < 60) 
  (h_val : h = 1) (m_val : m = 35)  : h + m = 36 := by
  sorry

end flight_duration_l375_375250


namespace find_k_l375_375197

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l375_375197


namespace probability_12th_roll_correct_l375_375207

noncomputable def probability_12th_roll_is_last : ℚ :=
  (7 / 8) ^ 10 * (1 / 8)

theorem probability_12th_roll_correct :
  probability_12th_roll_is_last = 282475249 / 8589934592 :=
by
  sorry

end probability_12th_roll_correct_l375_375207


namespace allocation_methods_count_l375_375309

theorem allocation_methods_count (A B C D E : Prop)
  (p : finset (fin 5)) (located : finset (finset (fin 5)))
  (condition1 : {A, B} ⊆ p) (condition2 : ∀ g ∈ located, 2 ≤ g.card) :
  located.card = 2 → (number_of_ways_to_allocate located = 8) :=
sorry

end allocation_methods_count_l375_375309


namespace find_k_of_vectors_orthogonal_l375_375164

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l375_375164


namespace average_of_first_5_numbers_l375_375733

-- Constants used in the conditions
constant A : ℝ
constant sum_first_5 : ℝ := 5 * A
constant sum_last_4 : ℝ := 4 * 200
constant middle_number : ℝ := 1100
constant total_sum : ℝ := 10 * 210

-- Main theorem to prove
theorem average_of_first_5_numbers :
  sum_first_5 + sum_last_4 + middle_number = total_sum → A = 40 :=
by
  intro h
  have : sum_first_5 + sum_last_4 + middle_number = 2100 := h
  sorry -- skipping proof steps

end average_of_first_5_numbers_l375_375733


namespace max_extra_time_matches_l375_375418

theorem max_extra_time_matches (number_teams : ℕ) 
    (points_win : ℕ) (points_lose : ℕ) 
    (points_win_extra : ℕ) (points_lose_extra : ℕ) 
    (total_matches_2016 : number_teams = 2016)
    (pts_win_3 : points_win = 3)
    (pts_lose_0 : points_lose = 0)
    (pts_win_extra_2 : points_win_extra = 2)
    (pts_lose_extra_1 : points_lose_extra = 1) :
    ∃ N, N = 1512 := 
by {
  sorry
}

end max_extra_time_matches_l375_375418


namespace abs_diff_solution_l375_375820

theorem abs_diff_solution (x y : ℕ) (h1 : x + y = 42) (h2 : x * y = 437) : |x - y| = 4 := 
sorry

end abs_diff_solution_l375_375820


namespace planting_point_6_planting_point_2016_l375_375853

noncomputable def T (a : ℕ) : ℕ :=
  a / 5

def x : ℕ → ℕ
| 1       := 1
| (k + 1) := x k + 1 - 5 * (T k - T (k - 1))

def y : ℕ → ℕ
| 1       := 1
| (k + 1) := y k + (T k - T (k - 1))

theorem planting_point_6 : (x 6, y 6) = (1, 2) :=
  sorry

theorem planting_point_2016 : (x 2016, y 2016) = (1, 404) :=
  sorry

end planting_point_6_planting_point_2016_l375_375853


namespace find_n_l375_375564

open Real BigOperators

noncomputable def min_sum_is_integer (n : ℕ) (a : Fin n → ℝ) : Prop :=
  ∑ i in Finset.range n, a i = 17 ∧
  (∑ i in Finset.range n, sqrt ((a i) ^ 2 + (2 * i + 1 : ℕ) ^ 2)).floor = 
  (sqrt (17 ^ 2 + n ^ 4)).floor

theorem find_n (n : ℕ) (a : Fin n → ℝ) (h : min_sum_is_integer n a) : n = 12 := 
sorry

end find_n_l375_375564


namespace gcd_lcm_product_360_l375_375385

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l375_375385


namespace least_common_denominator_sum_proof_l375_375943

theorem least_common_denominator_sum_proof : 
  let den1 := 3 
  let den2 := 4 
  let den3 := 5 
  let den4 := 6 
  let den5 := 8 
  let den6 := 9 
  let den7 := 10 
  lcm [den1, den2, den3, den4, den5, den6, den7] = 360 :=
by 
  rw lcm_eq_prod_factors
  repeat { split_ifs }
  rw prime_factors.2
  sorry

end least_common_denominator_sum_proof_l375_375943


namespace probability_sum_is_square_l375_375304

theorem probability_sum_is_square (n : ℕ) (h1 : n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12) : 
  let favorable_outcomes := 7
  let total_possible_outcomes := 36
  (favorable_outcomes / total_possible_outcomes = 7 / 36) :=
by {
  have square_sums : list ℕ := [4, 9],
  have favorable_count : list (ℕ × ℕ) := [(1, 3), (2, 2), (3, 1), (3, 6), (4, 5), (5, 4), (6, 3)],
  exact sorry
}

end probability_sum_is_square_l375_375304


namespace total_cost_of_water_l375_375714

-- Define conditions in Lean 4
def cost_per_liter : ℕ := 1
def liters_per_bottle : ℕ := 2
def number_of_bottles : ℕ := 6

-- Define the theorem to prove the total cost
theorem total_cost_of_water : (number_of_bottles * (liters_per_bottle * cost_per_liter)) = 12 :=
by
  sorry

end total_cost_of_water_l375_375714


namespace equality_of_a_b_c_l375_375533

theorem equality_of_a_b_c
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (eqn : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
by
  sorry

end equality_of_a_b_c_l375_375533


namespace restore_original_problem_l375_375903

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375903


namespace length_of_DE_l375_375548

theorem length_of_DE (A B C D E : Type)
  [triangle : Triangle A B C] (h1 : angle A B C = 90) (h2 : distance A B = distance A C) (h3 : distance B C = 10)
  (h4 : Perpendicular B D A C) (h5 : Perpendicular C E A B) (h6 : distance B D = 2 * distance D E) : 
  distance D E = 10 / 3 :=
  sorry

end length_of_DE_l375_375548


namespace min_sum_grid_l375_375620

open Function

theorem min_sum_grid :
  ∀ (n : ℕ) (grid : ℕ → ℕ → ℕ),
    (n = 100) →
    (∀ i j, i < n → j < n → Natural → grid i j → grid i j > 0) →
    (∀ i j, i < n → j < n →
      ((∀ (di dj : ℕ), (di, dj) ∈ {(-1, 0), (1, 0), (0, -1), (0, 1)} → 
        i + di < n → j + dj < n → 
        (grid i j < grid (i + di) (j + dj)) ∨ 
        (grid i j > grid (i + di) (j + dj))))) →
    (∑ i in range n, ∑ j in range n, grid i j) = 15000 :=
  sorry

end min_sum_grid_l375_375620


namespace min_value_of_quadratic_function_min_attained_at_negative_two_l375_375752

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

theorem min_value_of_quadratic_function : ∀ x : ℝ, quadratic_function x ≥ -5 :=
by
  sorry

theorem min_attained_at_negative_two : quadratic_function (-2) = -5 :=
by
  sorry

end min_value_of_quadratic_function_min_attained_at_negative_two_l375_375752


namespace restore_original_problem_l375_375909

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375909


namespace gcd_values_count_l375_375405

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375405


namespace kims_total_points_l375_375749

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end kims_total_points_l375_375749


namespace range_of_f_intervals_monotonically_increasing_l375_375569

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  let f := λ x : ℝ, 2 * sin x * cos x - 2 * sin x ^ 2 + 1
  ∃ y, y ∈ set.Icc (-1) (real.sqrt 2) ∧ f x = y :=
sorry

theorem intervals_monotonically_increasing (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
  let f := λ x : ℝ, 2 * sin x * cos x - 2 * sin x ^ 2 + 1
  (∀ x1 x2, 0 ≤ x1 → x1 ≤ π/8 → x1 ≤ x2 → x2 ≤ π/8 → f x1 ≤ f x2)
  ∧ (∀ x1 x2, 5*π/8 ≤ x1 → x1 ≤ π → x1 ≤ x2 → x2 ≤ π → f x1 ≤ f x2) :=
sorry

end range_of_f_intervals_monotonically_increasing_l375_375569


namespace mixed_fraction_product_example_l375_375882

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375882


namespace fraction_to_decimal_equiv_l375_375010

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375010


namespace football_team_won_game_score_l375_375635

theorem football_team_won_game_score
  (games : ℕ)
  (total_goals_scored total_goals_conceded : ℕ)
  (won drew lost : ℕ)
  (goals_won_game : ℕ)
  (goals_drew_game goals_lost_game_conceded goals_lost_game_scored : ℕ)
  (won_score conceded_draw lost_score won_conceded : bool)
  (H1 : games = 3)
  (H2 : total_goals_scored = 3)
  (H3 : total_goals_conceded = 1)
  (H4 : won = 1)
  (H5 : drew = 1)
  (H6 : lost = 1)
  (H7 : goals_lost_game_conceded = 1)
  (H8 : goals_lost_game_scored = 0)
  (H9 : goals_drew_game = 0)
  (H10 : won_score = true)
  (H11 : conceded_draw = true)
  (H12 : lost_score = true)
  (H13 : won_conceded = true) :
  goals_won_game = 3 := 
sorry

end football_team_won_game_score_l375_375635


namespace fraction_missing_l375_375963

-- Define the conditions
variables (y : ℕ) -- Assuming y is a natural number representing the number of coins
def lost_coins : ℕ := y / 3
def found_coins : ℕ := (y / 3) * 3 / 4
-- Total number of coins after finding some of the lost ones
def remaining_coins : ℕ := y - lost_coins + found_coins

-- The goal is to prove that the fraction of the coins she originally received that she is still missing is 1/12
theorem fraction_missing (y : ℕ) : (y - remaining_coins) / y = 1 / 12 := 
by
  sorry

end fraction_missing_l375_375963


namespace gcd_values_count_l375_375371

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l375_375371


namespace sum_of_all_possible_values_of_g7_l375_375673

def f (x : ℝ) : ℝ := x ^ 2 - 6 * x + 14
def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g7 :
  let x1 := 3 + Real.sqrt 2;
  let x2 := 3 - Real.sqrt 2;
  let g1 := g x1;
  let g2 := g x2;
  g (f 7) = g1 + g2 := by
  sorry

end sum_of_all_possible_values_of_g7_l375_375673


namespace sum_telescope_fraction_l375_375089

theorem sum_telescope_fraction : ∑ n in finset.range 14, (1 : ℚ) / (n + 1) / (n + 2) = 14 / 15 := by
  sorry

end sum_telescope_fraction_l375_375089


namespace extension_MN_bisects_EF_l375_375933

noncomputable def bisects_ef (A B C D E F M N : Point) : Prop :=
  let line_mn := line_through M N
  let intersection_ef := intersection_point_on_line EF line_mn
  let midpoint_ef := midpoint E F
  intersection_ef = midpoint_ef

theorem extension_MN_bisects_EF (A B C D E F M N : Point)
    (h1: convex_quadrilateral A B C D)
    (h2: ¬ parallel (line_through A B) 
                    (line_through D C))
    (h3: ¬ parallel (line_through D A) 
                    (line_through C B))
    (h4: extends_to_intersect_at (line_through A B) 
                                 (line_through D C) E)
    (h5: extends_to_intersect_at (line_through D A) 
                                 (line_through C B) F)
    (h6: midpoint M A C) 
    (h7: midpoint N B D)
      : bisects_ef A B C D E F M N :=
sorry

end extension_MN_bisects_EF_l375_375933


namespace examination_duration_in_hours_l375_375232

theorem examination_duration_in_hours 
  (total_questions : ℕ)
  (type_A_questions : ℕ)
  (time_for_A_problems : ℝ) 
  (time_ratio_A_to_B : ℝ)
  (total_time_for_A : ℝ) 
  (total_time : ℝ) :
  total_questions = 200 → 
  type_A_questions = 15 → 
  time_ratio_A_to_B = 2 → 
  total_time_for_A = 25.116279069767444 →
  total_time = (total_time_for_A + 185 * (25.116279069767444 / 15 / 2)) → 
  total_time / 60 = 3 :=
by sorry

end examination_duration_in_hours_l375_375232


namespace min_area_rectangle_l375_375459

theorem min_area_rectangle (P : ℕ) (hP : P = 60) :
  ∃ (l w : ℕ), 2 * l + 2 * w = P ∧ l * w = 29 :=
by
  sorry

end min_area_rectangle_l375_375459


namespace math_problem_l375_375267

-- Definitions of the notions
variable {m n : Line}
variable {α β : Plane}

-- Define the relevant propositions as terms 
def proposition_1 : Prop := m ⟂ α ∧ n ⟂ m → n ∥ α
def proposition_2 : Prop := α ∥ β ∧ n ⟂ α ∧ m ∥ β → n ⟂ m
def proposition_3 : Prop := m ∥ α ∧ n ⟂ β ∧ m ⟂ n → α ⟂ β
def proposition_4 : Prop := m ∥ α ∧ n ⟂ β ∧ m ∥ n → α ⟂ β

-- The main theorem statement that ② and ④ are true
theorem math_problem : (proposition_2 ∧ proposition_4) := 
by
  -- proof goes here. Currently, a placeholder for a non-trivial proof.
  sorry

end math_problem_l375_375267


namespace minimum_positive_period_is_pi_l375_375751

-- Define the function using Lean's syntax
def f (x : ℝ) : ℝ := sin (π / 3 - 2 * x) + sin (2 * x)

-- Define the problem statement in Lean
theorem minimum_positive_period_is_pi : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T' :=
by
  use π
  sorry

end minimum_positive_period_is_pi_l375_375751


namespace least_possible_product_of_two_distinct_primes_gt_10_l375_375789

-- Define a predicate to check if a number is a prime greater than 10
def is_prime_gt_10 (n : ℕ) : Prop := 
  nat.prime n ∧ n > 10

-- Lean theorem statement
theorem least_possible_product_of_two_distinct_primes_gt_10 :
  ∃ p q : ℕ, is_prime_gt_10 p ∧ is_prime_gt_10 q ∧ p ≠ q ∧ p * q = 143 :=
begin
  -- existence proof omitted
  sorry
end

end least_possible_product_of_two_distinct_primes_gt_10_l375_375789


namespace lcm_of_numbers_l375_375826

theorem lcm_of_numbers (x : Nat) (h_ratio : x ≠ 0) (h_hcf : Nat.gcd (5 * x) (Nat.gcd (7 * x) (9 * x)) = 11) :
    Nat.lcm (5 * x) (Nat.lcm (7 * x) (9 * x)) = 99 :=
by
  sorry

end lcm_of_numbers_l375_375826


namespace find_k_l375_375184

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l375_375184


namespace find_x_when_fx_is_11_min_and_max_values_on_interval_l375_375149

def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 3

theorem find_x_when_fx_is_11 :
  f 2 = 11 :=
sorry

theorem min_and_max_values_on_interval :
  ∀ x ∈ set.Icc (-2 : ℝ) 1,
    (x = 0 → f x = 2) ∧ (x = 1 → f x = 3) :=
sorry

end find_x_when_fx_is_11_min_and_max_values_on_interval_l375_375149


namespace fraction_to_decimal_l375_375039

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l375_375039


namespace gcd_possible_values_count_l375_375390

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l375_375390


namespace solve_for_y_l375_375719

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l375_375719


namespace max_non_overlapping_strips_min_number_of_strips_cover_l375_375818

-- Problem (a): Maximum number of non-overlapping $1 \times 3$ strips.
theorem max_non_overlapping_strips (n : ℕ) (black_cells : ℕ) (strip_size : ℕ) 
  (napkin_diagonally_colored : Prop) 
  (h1 : black_cells = 7) 
  (h2 : strip_size = 3) 
  (h3 : by → n * strip_size ≤ black_cells) : 
  n ≤ 7 :=
sorry

-- Problem (b): Minimum number of overlapping $1 \times 3$ strips to cover the entire napkin.
theorem min_number_of_strips_cover (n : ℕ) (black_cells : ℕ) (strip_size : ℕ) 
  (napkin_diagonally_colored : Prop) 
  (h1 : black_cells = 11) 
  (h2 : strip_size = 3) 
  (h3 : by → n * black_cells ≥ strip_size) : 
  n ≥ 11 :=
sorry

end max_non_overlapping_strips_min_number_of_strips_cover_l375_375818


namespace total_eyes_in_family_l375_375654

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end total_eyes_in_family_l375_375654


namespace angle_of_inclination_of_tangent_l375_375566

/-- Given the curve y = (1/2) * x^2 - 2 and a point P(1, -3/2) on it, 
the angle of inclination of the tangent line passing through point P is 45 degrees. -/
theorem angle_of_inclination_of_tangent :
  let curve := λ x : ℝ, (1 / 2) * x ^ 2 - 2,
      P := (1 : ℝ, -3 / 2),
      deriv := λ x : ℝ, x 
  in curve 1 = -3 / 2 ∧ P = (1, -3 / 2) ∧ ∀ θ : ℝ, tan θ = deriv 1 → θ = π / 4 :=
by 
  sorry

end angle_of_inclination_of_tangent_l375_375566


namespace mixed_fractions_product_l375_375863

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l375_375863


namespace area_of_triangle_ABC_l375_375681

variables (O A B C : ℝ) 
variables (angle_BAC : ℝ)
variables (OA : ℝ)

-- Definitions corresponding to conditions
def origin := O = 0
def point_A := A = 12 -- Since OA = 12
def point_B := B > 0 -- Positive y-axis
def point_C := C > 0 -- Positive z-axis
def angle_BAC_45_deg := angle_BAC = 45 * (Real.pi / 180)

-- The proposition to prove
theorem area_of_triangle_ABC : 
  origin ∧ point_A ∧ point_B ∧ point_C ∧ angle_BAC_45_deg → 
  (1 / 2) * A * B * Real.sin angle_BAC = 72 :=
by
  -- Conditions are assumed here
  intros h
  sorry -- Proof to be provided

end area_of_triangle_ABC_l375_375681


namespace vectors_coplanar_l375_375601

theorem vectors_coplanar (m : ℝ) 
  (a : ℝ × ℝ × ℝ := (1, 1, 1)) 
  (b : ℝ × ℝ × ℝ := (1, 2, 1)) 
  (c : ℝ × ℝ × ℝ := (1, 0, m)) : 
  (∃ λ μ : ℝ, c = (λ * 1 + μ * 1, λ * 1 + μ * 2, λ * 1 + μ * 1)) → 
  m = 1 :=
sorry

end vectors_coplanar_l375_375601


namespace length_of_BC_l375_375927

theorem length_of_BC {b : ℝ} (h_parabola : ∀ A B C : ℝ × ℝ, ((A = (0, 0)) → (∃ b, B = (-b, 2*b^2) ∧ C = (b, 2*b^2)))) 
  (h_horizontal : ∃ b, ((-b, 2*b^2), (b, 2*b^2)) → true) 
  (h_area : ∀ b : ℝ, 2 * b^3 = 72) : ∃ b : ℝ, 2 * b = 2 * (real.cbrt 36) :=
by
  sorry

end length_of_BC_l375_375927


namespace a_pow_m_minus_a_pow_n_divisible_by_30_l375_375701

theorem a_pow_m_minus_a_pow_n_divisible_by_30
  (a m n k : ℕ)
  (h_n_ge_two : n ≥ 2)
  (h_m_gt_n : m > n)
  (h_m_n_diff : m = n + 4 * k) :
  30 ∣ (a ^ m - a ^ n) :=
sorry

end a_pow_m_minus_a_pow_n_divisible_by_30_l375_375701


namespace square_area_l375_375300

theorem square_area (A : ℝ) (s : ℝ) (prob_not_in_B : ℝ)
  (h1 : s * 4 = 32)
  (h2 : prob_not_in_B = 0.20987654320987653)
  (h3 : A - s^2 = prob_not_in_B * A) :
  A = 81 :=
by
  sorry

end square_area_l375_375300


namespace shooting_match_orders_l375_375616

theorem shooting_match_orders : 
  let targets := ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'] in
  let count_A := 3 in
  let count_B := 3 in
  let count_C := 4 in
  multiset.cardα * finset.card ↥([↟multiset α⁼f, A, _↦30]:=count_A) = 10 →
  multiset.cardα * finset.card ↥([↟multiset α⁼f, B, _↦30]:=count_B) = 10 →
  multiset.cardα * finset.card ↥([↟multiset α⁼f, C, _↦30]:=count_C) = 10 →
  nat.factorial (targets.length) / 
  (nat.factorial count_A * nat.factorial count_B * nat.factorial count_C) = 4200 :=
begin
  sorry
end

end shooting_match_orders_l375_375616


namespace a8_div_b8_l375_375580

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given Conditions
axiom sum_a (n : ℕ) : S n = (n * (a 1 + (n - 1) * a 2)) / 2 -- Sum of first n terms of arithmetic sequence a_n
axiom sum_b (n : ℕ) : T n = (n * (b 1 + (n - 1) * b 2)) / 2 -- Sum of first n terms of arithmetic sequence b_n
axiom ratio (n : ℕ) : S n / T n = (7 * n + 3) / (n + 3)

-- Proof statement
theorem a8_div_b8 : a 8 / b 8 = 6 := by
  sorry

end a8_div_b8_l375_375580


namespace smallest_four_digit_integer_mod_8_eq_3_l375_375809

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end smallest_four_digit_integer_mod_8_eq_3_l375_375809


namespace fraction_to_decimal_equiv_l375_375011

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375011


namespace triangle_ABC_XY_squared_l375_375354

theorem triangle_ABC_XY_squared 
  (AB AC BC : ℝ)
  (h1 : AB = 25)
  (h2 : AC = 29)
  (h3 : BC = 36)
  (Omega : ∀ (P : Type) [inhabited P], ∃ (circle : P → Prop), ∀ E, circle E ↔ (E = P))
  (omega : ∀ (P : Type) [inhabited P], ∃ (circle : P → Prop), ∀ E, circle E ↔ (E = P))
  (D : Type)
  (h4 : Omega D)
  (AD_length : ∀ (A D : Type), AD_length A D = D) -- AD is a diameter of Omega
  (X Y : Type)
  (h5 : ∃ (points : X → Y → Prop), points X Y) -- AD intersects omega in two distinct points X and Y
  : XY^2 = 252 := 
sorry

end triangle_ABC_XY_squared_l375_375354


namespace range_of_m_and_n_l375_375578

theorem range_of_m_and_n (m n : ℝ) : 
  (2 * 2 - 3 + m > 0) → ¬ (2 + 3 - n ≤ 0) → (m > -1 ∧ n < 5) := by
  intros hA hB
  sorry

end range_of_m_and_n_l375_375578


namespace increasing_on_interval_of_m_l375_375595

def f (m x : ℝ) := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_on_interval_of_m (m : ℝ) :
  (∀ x : ℝ, 2 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0) → m ≤ 5 / 2 :=
sorry

end increasing_on_interval_of_m_l375_375595


namespace prob_two_days_ge_100_one_day_lt_50_xi_distribution_xi_mean_xi_variance_l375_375472

section CakeShop

def freqDist := [
  (0, 50, 15),
  (50, 100, 25),
  (100, 150, 30),
  (150, 200, 20),
  (200, 250, 10)
]

def p_sales_ge_100 : ℝ := (30 + 20 + 10) / 100
def p_sales_lt_50 : ℝ := 15 / 100
def X := binomial 3 0.3

theorem prob_two_days_ge_100_one_day_lt_50 : 
  (3 * (p_sales_ge_100 ^ 2) * p_sales_lt_50) = 0.162 := 
by
  simp [p_sales_ge_100, p_sales_lt_50]
  sorry

theorem xi_distribution :
  (∀ x, x ∈ X.support → 
    (X.prob x = match x with
      | 0 => 0.343
      | 1 => 0.441
      | 2 => 0.189
      | 3 => 0.027
      | _ => 0)
  ) := sorry

theorem xi_mean : X.mean = 0.9 := by
  simp [X.mean]
  sorry

theorem xi_variance : X.variance = 0.63 := by
  simp [X.variance]
  sorry

end CakeShop

end prob_two_days_ge_100_one_day_lt_50_xi_distribution_xi_mean_xi_variance_l375_375472


namespace jade_transactions_l375_375278

theorem jade_transactions (mabel anthony cal jade : ℕ) 
    (h1 : mabel = 90) 
    (h2 : anthony = mabel + (10 * mabel / 100)) 
    (h3 : cal = 2 * anthony / 3) 
    (h4 : jade = cal + 18) : 
    jade = 84 := by 
  -- Start with given conditions
  rw [h1] at h2 
  have h2a : anthony = 99 := by norm_num; exact h2 
  rw [h2a] at h3 
  have h3a : cal = 66 := by norm_num; exact h3 
  rw [h3a] at h4 
  norm_num at h4 
  exact h4

end jade_transactions_l375_375278


namespace proof_problem_l375_375494

def diamondsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem proof_problem :
  { (x, y) : ℝ × ℝ | diamondsuit x y = diamondsuit y x } =
  { (x, y) | x = 0 } ∪ { (x, y) | y = 0 } ∪ { (x, y) | x = y } ∪ { (x, y) | x = -y } :=
by
  sorry

end proof_problem_l375_375494


namespace small_cubes_with_two_faces_painted_red_l375_375462

theorem small_cubes_with_two_faces_painted_red (edge_length : ℕ) (small_cube_edge_length : ℕ)
  (h1 : edge_length = 4) (h2 : small_cube_edge_length = 1) :
  ∃ n, n = 24 :=
by
  -- Proof skipped
  sorry

end small_cubes_with_two_faces_painted_red_l375_375462


namespace restore_fractions_l375_375890

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375890


namespace fraction_to_decimal_equiv_l375_375015

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375015


namespace fraction_to_decimal_equiv_l375_375014

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375014


namespace marbles_left_calculation_l375_375852

/-- A magician starts with 20 red marbles and 30 blue marbles.
    He removes 3 red marbles and 12 blue marbles. We need to 
    prove that he has 35 marbles left in total. -/
theorem marbles_left_calculation (initial_red : ℕ) (initial_blue : ℕ) (removed_red : ℕ) 
    (removed_blue : ℕ) (H1 : initial_red = 20) (H2 : initial_blue = 30) 
    (H3 : removed_red = 3) (H4 : removed_blue = 4 * removed_red) :
    (initial_red - removed_red) + (initial_blue - removed_blue) = 35 :=
by
   -- sorry to skip the proof
   sorry

end marbles_left_calculation_l375_375852


namespace gcd_possible_values_count_l375_375373

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375373


namespace exists_subset_sum_equal_A_l375_375755

theorem exists_subset_sum_equal_A 
  (A : ℕ) (k : ℕ) (a : Fin k → ℕ) 
  (hA : ∀ n : ℕ, n ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → A % n = 0)
  (hSum : (∑ i, a i) = 2 * A)
  (ha : ∀ i, a i ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) :
  ∃ S : Finset (Fin k), (∑ i in S, a i) = A :=
by
  sorry

end exists_subset_sum_equal_A_l375_375755


namespace twelve_pretty_sum_div_12_eq_14_l375_375487

def is_twelve_pretty (n : ℕ) : Prop :=
  n > 0 ∧
  n % 12 = 0 ∧
  (∀ d : ℕ, d > 0 ∧ d ≤ n → n % d = 0 → 
  d = 1 ∨ d = n ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 9 ∨ d = 12 
  ∨ d = 18 ∨ d = 24 ∨ d = 36 ∨ d = 72)

def twelve_pretty_sum : ℕ :=
  ∑ n in finset.filter is_twelve_pretty (finset.Ico 1 500), n

theorem twelve_pretty_sum_div_12_eq_14 : twelve_pretty_sum / 12 = 14 :=
by sorry

end twelve_pretty_sum_div_12_eq_14_l375_375487


namespace mixed_fraction_product_l375_375894

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l375_375894


namespace carter_trip_stop_time_l375_375948

-- Define the key components and conditions of the problem.
def total_stops (road_trip_hours : ℕ) (stretch_interval : ℕ) (food_stops : ℕ) (gas_stops : ℕ) : ℕ :=
  road_trip_hours / stretch_interval + food_stops + gas_stops

def total_additional_time (initial_trip_hours : ℕ) (final_trip_hours : ℕ): ℕ :=
  (final_trip_hours - initial_trip_hours) * 60

def time_per_stop (additional_time : ℕ) (stops : ℕ) : ℕ := 
  additional_time / stops

-- Formal statement of the problem in Lean
theorem carter_trip_stop_time :
  ∀ (road_trip_hours stretch_interval food_stops gas_stops initial_trip_hours final_trip_hours : ℕ),
    road_trip_hours = 14 →
    stretch_interval = 2 →
    food_stops = 2 →
    gas_stops = 3 →
    initial_trip_hours = 14 →
    final_trip_hours = 18 →
    let stops := total_stops road_trip_hours stretch_interval food_stops gas_stops in
    let additional_time := total_additional_time initial_trip_hours final_trip_hours in
    time_per_stop additional_time stops = 20 :=
sorry

end carter_trip_stop_time_l375_375948


namespace train_speed_l375_375357

theorem train_speed (v : ℝ) :
  let speed_train1 := 80  -- speed of the first train in km/h
  let length_train1 := 150 / 1000 -- length of the first train in km
  let length_train2 := 100 / 1000 -- length of the second train in km
  let total_time := 5.999520038396928 / 3600 -- time in hours
  let total_length := length_train1 + length_train2 -- total length in km
  let relative_speed := total_length / total_time -- relative speed in km/h
  relative_speed = speed_train1 + v → v = 70 :=
by
  sorry

end train_speed_l375_375357


namespace gymnastics_people_count_l375_375812

theorem gymnastics_people_count (lines people_per_line : ℕ) (h1 : lines = 4) (h2 : people_per_line = 8) : lines * people_per_line = 32 := 
by
  rw [h1, h2]
  exact Nat.mul_comm 4 8

end gymnastics_people_count_l375_375812


namespace circles_common_tangents_l375_375324

def is_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  let dist := (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2
  (dist = (r1 + r2)^2) -- The circles are externally tangent

noncomputable def num_common_tangents (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℕ :=
  if r1 + r2 = real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) then 3 else sorry

theorem circles_common_tangents (x y : ℝ) :
  let c1 := (1, 1)
  let r1 := 2
  let c2 := (-3, 1)
  let r2 := 2
  num_common_tangents c1 c2 r1 r2 = 3 :=
by
  -- Here we will write the proof, but for now, we'll use sorry
  sorry

end circles_common_tangents_l375_375324


namespace number_of_buses_l375_375744

theorem number_of_buses (total_supervisors : ℕ) (supervisors_per_bus : ℕ) (h1 : total_supervisors = 21) (h2 : supervisors_per_bus = 3) : total_supervisors / supervisors_per_bus = 7 :=
by
  sorry

end number_of_buses_l375_375744


namespace mixed_fraction_product_example_l375_375881

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l375_375881


namespace part1_part2_l375_375152

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem part1 (x : ℝ) (hx : x ≠ 0) : f(x) + f(1/x) = 1 := 
by
  rw [f, f]
  unfold f
  field_simp [hx]
  norm_num

theorem part2 : 2 * (Finset.sum (Finset.range 2016) (λ k, f (k+2))) + (Finset.sum (Finset.range 2016) (λ k, f (1 / (k+2)))) + (Finset.sum (Finset.range 2016) (λ k, 1 / (k+2)^2 * f (k+2))) = 4032 := 
by
  sorry

end part1_part2_l375_375152


namespace circulation_zero_l375_375490

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
(y * Real.exp(x * y), x * Real.exp(x * y), x * y * z)

def cone (x y z : ℝ) : Prop :=
x * x + y * y = (z - 1) * (z - 1)

def path_L : Set (ℝ × ℝ × ℝ) :=
{p | ∃ x y z, p = (x, y, z) ∧ cone x y z}

theorem circulation_zero : 
  ∮ (vector_field (x y z) along path_L) = 0 :=
sorry

end circulation_zero_l375_375490


namespace f_monotonicity_max_ab_l375_375155

-- Conditions
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * (x - 1)

-- Theorem 1: Monotonicity
theorem f_monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x, (f' := Real.exp x - a) > 0) ∧ 
  (a > 0 → 
    (∀ x, x < Real.log a → (f' := Real.exp x - a) < 0) ∧ 
    (∀ x, x > Real.log a → (f' := Real.exp x - a) > 0)) :=
by sorry

-- Theorem 2: Maximum of ab
theorem max_ab (a b : ℝ) (h : ∀ x, f x a ≥ b) (h_pos : a > 0) :
  ab ≤ (2 * a^2 - a^2 * Real.log a) :=
by sorry

end f_monotonicity_max_ab_l375_375155


namespace platform_length_l375_375450

theorem platform_length 
  (train_speed_kmph : ℝ) (cross_time_sec : ℕ) (train_length_m : ℝ) : 
  train_speed_kmph = 72 → cross_time_sec = 26 → train_length_m = 280.04 → 
  (let train_speed_mps := (train_speed_kmph * 1000 / 3600) in
   let total_distance := (train_speed_mps * cross_time_sec) in
   let platform_length := total_distance - train_length_m in
   platform_length = 239.96) :=
by
  intros h_speed h_time h_length
  let train_speed_mps := (72 * 1000 / 3600) -- converting speed from kmph to m/s
  let total_distance := train_speed_mps * 26 -- calculating total distance
  let platform_length := total_distance - 280.04 -- calculating platform length
  show platform_length = 239.96
  sorry

end platform_length_l375_375450


namespace milburg_population_l375_375335

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l375_375335


namespace smallest_four_digit_integer_mod_8_eq_3_l375_375810

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end smallest_four_digit_integer_mod_8_eq_3_l375_375810


namespace Liza_percentage_is_90_l375_375216

def LizaExam (L : ℕ) : Prop :=
  let total_items := 60
  let Rose_correct := L + 2
  let Rose_incorrect := 4
  Rose_correct = total_items - Rose_incorrect

theorem Liza_percentage_is_90 (L : ℕ) (h : LizaExam L) : L / 60 = 9 / 10 :=
by
  unfold LizaExam at h
  have L_value : L = 54 := by sorry
  calc
    L / 60 = 54 / 60 : by rw [L_value]
    ...  = 9 / 10    : by norm_num

end Liza_percentage_is_90_l375_375216


namespace S_l375_375442

/-- Given a triangular pyramid SABC with an inscribed circle centered at I in face ABC that touches AB, BC, and CA at points D, E, and F respectively. Points A', B', and C' are on segments SA, SB, and SC respectively such that AA' = AD, BB' = BE, and CC' = CF. If S' is a point on the circumsphere of the pyramid diametrically opposite to S and SI is the height of the pyramid, then S' is equidistant from points A', B', and C'. -/
theorem S'_equidistant (S A B C I D E F A' B' C' SI R r : ℝ)
  (h1: SI > 0)
  (h2 : metric.dist A' S = metric.dist D A)
  (h3 : metric.dist B' S = metric.dist E B)
  (h4 : metric.dist C' S = metric.dist F C)
  (h5 : metric.dist S' I = 2 * R)
  (h6 : S'A'^2 = S'B'^2 = S'C'^2 = (2 * R)^2 - SI^2 - r^2) :
  metric.dist S' A' = metric.dist S' B' ∧ metric.dist S' B' = metric.dist S' C' :=
sorry

end S_l375_375442


namespace distance_DE_l375_375783

noncomputable def point := (ℝ × ℝ)

variables (A B C P D E : point)
variables (AB BC AC PC : ℝ)
variables (on_line : point → point → point → Prop)
variables (is_parallel : point → point → point → point → Prop)

axiom AB_length : AB = 13
axiom BC_length : BC = 14
axiom AC_length : AC = 15
axiom PC_length : PC = 10

axiom P_on_AC : on_line A C P
axiom D_on_BP : on_line B P D
axiom E_on_BP : on_line B P E

axiom AD_parallel_BC : is_parallel A D B C
axiom AB_parallel_CE : is_parallel A B C E

theorem distance_DE : ∀ (D E : point), 
  on_line B P D → on_line B P E → 
  is_parallel A D B C → is_parallel A B C E → 
  ∃ dist : ℝ, dist = 12 * Real.sqrt 2 :=
by
  sorry

end distance_DE_l375_375783


namespace fraction_to_decimal_l375_375017

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l375_375017


namespace handshake_count_is_correct_l375_375483

def num_people : ℕ := 16
def handshakes_per_person (num_people : ℕ) : ℕ := num_people - 1 - 1 - 2
def total_handshakes (num_people : ℕ) : ℕ := (num_people * handshakes_per_person num_people) / 2

theorem handshake_count_is_correct : total_handshakes num_people = 96 := by
  have h1 : handshakes_per_person num_people = 12 := by simp [handshakes_per_person, num_people]
  have h2 : num_people * 12 = 192 := by norm_num
  have h3 : 192 / 2 = 96 := by norm_num
  rw [total_handshakes, h1, h2, h3]
  exact rfl

end handshake_count_is_correct_l375_375483


namespace not_divisible_by_5_l375_375692

-- Define the conditions: children ages and the number '7773'
def ages := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def number := 7773

-- Check divisibility
def divisible_by (n : ℕ) (d : ℕ) : Prop := ∃ (k : ℕ), n = d * k

-- State the problem
theorem not_divisible_by_5 : ¬ divisible_by number 5 ∧ 
  (∀ age ∈ ages, age ≠ 5 → divisible_by number age) :=
by
  split
  {
    -- show not divisible by 5
    sorry
  }
  {
    -- show divisible by all ages except 5
    intros age h1 h2
    sorry
  }

end not_divisible_by_5_l375_375692


namespace vector_sum_after_scalar_multiplication_l375_375945

open Real

def v1 : (ℝ × ℝ) := (4, -9)
def v2 : (ℝ × ℝ) := (-3, 5)

theorem vector_sum_after_scalar_multiplication :
  let scaled_v2 := (2 * v2.1, 2 * v2.2)
  in (v1.1 + scaled_v2.1, v1.2 + scaled_v2.2) = (-2, 1) :=
by
  sorry

end vector_sum_after_scalar_multiplication_l375_375945


namespace transformed_root_poly_l375_375725

theorem transformed_root_poly :
  ∀ (p q r s : ℚ), 
  (root (X^4 + 4 * X^3 - 5 : Polynomial ℚ) p) ∧ 
  (root (X^4 + 4 * X^3 - 5 : Polynomial ℚ) q) ∧ 
  (root (X^4 + 4 * X^3 - 5 : Polynomial ℚ) r) ∧ 
  (root (X^4 + 4 * X^3 - 5 : Polynomial ℚ) s) →
  (root (5 * X^6 - X^2 + 4 * X : Polynomial ℚ) (-(p + q + r) / s^3)) ∧
  (root (5 * X^6 - X^2 + 4 * X : Polynomial ℚ) (-(p + q + s) / r^3)) ∧
  (root (5 * X^6 - X^2 + 4 * X : Polynomial ℚ) (-(p + r + s) / q^3)) ∧
  (root (5 * X^6 - X^2 + 4 * X : Polynomial ℚ) (-(q + r + s) / p^3)) :=
by 
  sorry

end transformed_root_poly_l375_375725


namespace triangle_base_length_l375_375731

theorem triangle_base_length (A h : ℝ) (b : ℝ) (h_pos : 0 < h) (base_formula : A = (b * h) / 2) :
  A = 25 ∧ h = 5 → b = 10 := by
  intro h₁
  rw [h₁.1, h₁.2] at base_formula
  rw [mul_comm b 5, mul_div_cancel_left 25 two_ne_zero] at base_formula
  linarith

end triangle_base_length_l375_375731


namespace comparison_l375_375541

-- Defining the function f and the necessary conditions
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log (Real.sin θ))

-- Given θ such that 0 < θ < π/2
variables (θ : ℝ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2)

-- Definitions of α, β, and γ
def α : ℝ := f θ ((Real.sin θ + Real.cos θ) / 2)
def β : ℝ := f θ (Real.sqrt (Real.sin θ * Real.cos θ))
def γ : ℝ := f θ (2 * (Real.sin θ * Real.cos θ) / (Real.sin θ + Real.cos θ))

-- Statement of the proof problem
theorem comparison (h : ∀ x y : ℝ, f θ x ≤ f θ y ↔ x ≥ y) :
  α θ ≤ β θ ∧ β θ ≤ γ θ :=
by
  sorry

end comparison_l375_375541


namespace remainder_of_expression_l375_375957

theorem remainder_of_expression :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by {
  -- Prove the expression step by step
  -- sorry
  sorry
}

end remainder_of_expression_l375_375957


namespace cubic_polynomial_solution_l375_375726

variables (x : ℂ)

theorem cubic_polynomial_solution :
  ∃ (p : ℂ → ℂ), 
    (∀ (p : ℂ → ℂ), (p(4 - 3 * Complex.i) = 0 ∧ p(4 + 3 * Complex.i) = 0 ∧ p(0) = -108)) →
    (p x = x^3 - (308 / 25) * x^2 + (1192 / 25) * x - 108) :=
begin
  sorry
end

end cubic_polynomial_solution_l375_375726


namespace smallest_x_l375_375524

theorem smallest_x : ∃ x : ℕ, x + 6721 ≡ 3458 [MOD 12] ∧ x % 5 = 0 ∧ x = 45 :=
by
  sorry

end smallest_x_l375_375524


namespace find_function_l375_375145

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
  ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := 
by
  sorry

end find_function_l375_375145


namespace ellipse_m_value_l375_375741

-- Definitions for conditions of the problem
def ellipse_on_y_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, (x^2 + (a * y)^2 = 1) ∧ (2 * a = 2 * b) ∧ (a = b) ∧ (b^2 = 1) ∧ (a^2 = 1 / m)

-- The statement we need to prove
theorem ellipse_m_value (m : ℝ) (h : ellipse_on_y_axis m) : m = 1 :=
sorry

end ellipse_m_value_l375_375741


namespace monster_family_eyes_count_l375_375656

theorem monster_family_eyes_count :
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  (mom_eyes + dad_eyes) + (num_kids * kid_eyes) = 16 :=
by
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  have parents_eyes : mom_eyes + dad_eyes = 4 := by rfl
  have kids_eyes : num_kids * kid_eyes = 12 := by rfl
  show parents_eyes + kids_eyes = 16
  sorry

end monster_family_eyes_count_l375_375656


namespace extreme_points_count_range_of_a_l375_375686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

-- Statement 1: Number of extreme points in (-1, +∞) based on the value of a
theorem extreme_points_count (a : ℝ) :
  let domain := {x : ℝ | -1 < x}
  (if a < 0 then
    ∃! x ∈ domain, (∃ v, (x, v) ∈ Real.extr (f a))
  else if 0 ≤ a ∧ a ≤ (8/9) then
    ¬∃ x ∈ domain, (∃ v, (x, v) ∈ Real.extr (f a))
  else if a > (8/9) then
    ∃ x1 x2 ∈ domain, x1 < x2 ∧ (∃ v1, (x1, v1) ∈ Real.extr (f a)) ∧ 
                                      (∃ v2, (x2, v2) ∈ Real.extr (f a))
  else true) := sorry

-- Statement 2: Range of a such that ∀ x > 0, f(a, x) ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x, x > 0 → f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := sorry

end extreme_points_count_range_of_a_l375_375686


namespace convert_8pi_over_5_to_degrees_l375_375955

noncomputable def radian_to_degree (rad : ℝ) : ℝ := rad * (180 / Real.pi)

theorem convert_8pi_over_5_to_degrees : radian_to_degree (8 * Real.pi / 5) = 288 := by
  sorry

end convert_8pi_over_5_to_degrees_l375_375955


namespace find_function_l375_375511

theorem find_function (f : ℝ → ℝ) :
  (∀ x : ℝ, f(2 * x + 1) = 4 * x ^ 2 + 14 * x + 7) →
  ∀ x : ℝ, f(x) = x^2 + 5 * x + 1 :=
by sorry

end find_function_l375_375511


namespace restore_fractions_l375_375886

theorem restore_fractions (X Y : ℕ) : 5 + 1 / X ∈ ℚ → Y + 1 / 2 ∈ ℚ → (5 + 1 / X) * (Y + 1 / 2) = 43 ↔ (X = 17 ∧ Y = 8) := by
  -- proof goes here
  sorry

end restore_fractions_l375_375886


namespace x_y_value_l375_375295

theorem x_y_value (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 30) : x + y = 2 :=
sorry

end x_y_value_l375_375295


namespace gcd_possible_values_count_l375_375398

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375398


namespace family_four_children_includes_at_least_one_boy_one_girl_l375_375936

-- Specification of the probability function
def prob_event (n : ℕ) (event : fin n → bool) : ℚ := 
  (Real.to_rat (Real.exp (- (Real.nat_to_real (nat.log2 n)))) : ℚ)

-- Predicate that checks if there is at least one boy and one girl in the list
def has_boy_and_girl (children : fin 4 → bool) : Prop :=
  ∃ i j, children i ≠ children j

theorem family_four_children_includes_at_least_one_boy_one_girl : 
  (∑ event in (finset.univ : finset (fin 4 → bool)), 
     if has_boy_and_girl event then prob_event 4 event else 0) = 7 / 8 :=
by
  sorry

end family_four_children_includes_at_least_one_boy_one_girl_l375_375936


namespace fraction_to_decimal_equiv_l375_375013

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l375_375013


namespace zero_in_intervals_l375_375685

-- Define the function f
def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

-- Prove that there are zeros in the specified intervals
theorem zero_in_intervals :
  ∃ x1 ∈ set.Ioo 0 3, f x1 = 0 ∧ ∃ x2 ∈ set.Ioo 3 ∞, f x2 = 0 := 
sorry

end zero_in_intervals_l375_375685


namespace more_red_than_white_red_not_less_than_white_l375_375341

open Nat

theorem more_red_than_white :
  let red := 4
  let white := 6
  (choose red 4) + (choose red 3) * (choose white 1) = 25 := 
by
  intros
  rw [choose_self, choose_succ_self, mul_comm]
  exact rfl

theorem red_not_less_than_white :
  let red := 4
  let white := 6
  (choose red 4) + (choose red 3) * (choose white 1) + (choose red 2) * (choose white 2) = 115 := 
by
  intros
  rw [choose_self, choose_succ_self, choose_two, mul_comm]
  exact rfl


end more_red_than_white_red_not_less_than_white_l375_375341


namespace quadratic_root_conditions_l375_375596

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k + 2
  let b := 4
  let c := 1
  (a ≠ 0) ∧ (b^2 - 4*a*c > 0)

theorem quadratic_root_conditions (k : ℝ) :
  quadratic_has_two_distinct_real_roots k ↔ k < 2 ∧ k ≠ -2 := 
by
  sorry

end quadratic_root_conditions_l375_375596


namespace number_of_routes_l375_375707

-- Definitions of the conditions
def conditions := 
  ∀ (blocksNorth_Home_NW : ℕ) (blocksEast_Home_NW : ℕ) (blocksSouth_Sch_SE : ℕ) (blocksEast_Sch_SE : ℕ),
  blocksNorth_Home_NW = 4 ∧ blocksEast_Home_NW = 3 ∧ 
  blocksSouth_Sch_SE = 2 ∧ blocksEast_Sch_SE = 5

-- Proof that given the conditions, the number of different routes is 735
theorem number_of_routes : 
  conditions →
  let house_to_nw_routes := (Nat.choose 7 3)
  let se_to_school_routes := (Nat.choose 7 2)
  let through_park_routes := 1
  house_to_nw_routes * through_park_routes * se_to_school_routes = 735
:= by
  intros h
  simp [conditions, Nat.choose]
  sorry

end number_of_routes_l375_375707


namespace routes_from_A_to_B_using_each_road_once_l375_375227

theorem routes_from_A_to_B_using_each_road_once 
  (A B D : Type) 
  (road_AD road_AD' road_AB road_BD road_BD' : A → D)
  (initial_condition : ∀ x : A, road_AD x = road_BD x)
  : ∃ routes : list (A → D), routes.length = 16 :=
by
  sorry

end routes_from_A_to_B_using_each_road_once_l375_375227


namespace gcd_values_count_l375_375404

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l375_375404


namespace order_of_magnitudes_l375_375159

theorem order_of_magnitudes (x : ℝ) (h : 0.9 < x ∧ x < 1.0) :
  x < x^(2 * x) ∧ x^(2 * x) < x^(x^x) :=
begin
  -- Proof omitted
  sorry,
end

end order_of_magnitudes_l375_375159


namespace max_gray_squares_no_gray_trimino_l375_375359

open Finset

theorem max_gray_squares_no_gray_trimino (B : matrix (fin 8) (fin 8) bool) :
  (∀ i j k : fin 8, ¬(B i j ∧ B i.succ j ∧ B i.succ.succ j) ∧
                   ¬(B i j ∧ B i j.succ ∧ B i j.succ.succ)) → 
  card { p : fin 8 × fin 8 | B p.1 p.2 } ≤ 43 :=
sorry

end max_gray_squares_no_gray_trimino_l375_375359


namespace find_k_l375_375177

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l375_375177


namespace probability_one_black_ball_l375_375233

/-- There are 10 balls in a box, among which 3 are black and 7 are white. Each person draws a ball,
records its color, and puts it back before the next draw. We need to prove that the probability
that exactly one out of three people will draw a black ball is 0.441. -/
theorem probability_one_black_ball (h : 0 < 10 ∧ 3 < 10):
Prob = (finset.choose 3 1) * (0.7 * 0.7 * 0.3) := 
sorry

end probability_one_black_ball_l375_375233


namespace probability_of_at_least_one_red_ball_l375_375146

theorem probability_of_at_least_one_red_ball
  (pA_red : ℝ) (pB_red : ℝ)
  (hA : pA_red = 1 / 3)
  (hB : pB_red = 1 / 2) :
  (1 - (1 - pA_red) * (1 - pB_red)) = 2 / 3 :=
by {
  -- We introduce the assumptions
  rw [hA, hB],
  -- Simplify the expression
  simp,
  -- By computation, the simplified terms should match
  exact dec_trivial,
}

end probability_of_at_least_one_red_ball_l375_375146


namespace sufficient_but_not_necessary_l375_375985

theorem sufficient_but_not_necessary (x: ℝ) (hx: 0 < x ∧ x < 1) : 0 < x^2 ∧ x^2 < 1 ∧ (∀ y, 0 < y^2 ∧ y^2 < 1 → (y > 0 ∧ y < 1 ∨ y < 0 ∧ y > -1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l375_375985


namespace gcd_possible_values_count_l375_375395

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l375_375395


namespace circle_eq_of_hyperbola_focus_eccentricity_l375_375739

theorem circle_eq_of_hyperbola_focus_eccentricity :
  ∀ (x y : ℝ), ((y^2 - (x^2 / 3) = 1) → (x^2 + (y-2)^2 = 4)) := by
  intro x y
  intro hyp_eq
  sorry

end circle_eq_of_hyperbola_focus_eccentricity_l375_375739


namespace words_per_page_eq_106_l375_375430

-- Definition of conditions as per the problem statement
def pages : ℕ := 224
def max_words_per_page : ℕ := 150
def total_words_congruence : ℕ := 156
def modulus : ℕ := 253

theorem words_per_page_eq_106 (p : ℕ) : 
  (224 * p % 253 = 156) ∧ (p ≤ 150) → p = 106 :=
by 
  sorry

end words_per_page_eq_106_l375_375430


namespace nine_digit_number_not_prime_l375_375776

/--
Given three three-digit prime numbers in arithmetic progression, 
the nine-digit number formed by writing them consecutively is not prime.
-/
theorem nine_digit_number_not_prime 
  (p1 p2 p3 : ℕ) 
  (hp1 : nat.prime p1) 
  (hp2 : nat.prime p2) 
  (hp3 : nat.prime p3) 
  (h_digit : 100 ≤ p1 ∧ p1 < 1000 ∧ 100 ≤ p2 ∧ p2 < 1000 ∧ 100 ≤ p3 ∧ p3 < 1000) 
  (h_ap : p2 - p1 = p3 - p2) 
  : ¬ nat.prime (p1 * 10^6 + p2 * 10^3 + p3) :=
sorry

end nine_digit_number_not_prime_l375_375776


namespace contrapositive_of_not_p_implies_q_l375_375600

variable (p q : Prop)

theorem contrapositive_of_not_p_implies_q :
  (¬p → q) → (¬q → p) := by
  sorry

end contrapositive_of_not_p_implies_q_l375_375600


namespace problem_solution_l375_375644

noncomputable def vertex_C_and_area (A B : ℝ × ℝ) (M N : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let midpoint_ac := (C.1 + A.1) / 2 = M.1 ∧ M.1 = 0
  let midpoint_bc := (C.2 + B.2) / 2 = N.2 ∧ N.2 = 0
  let length_ab := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let slope_ab := (B.2 - A.2) / (B.1 - A.1)
  let line_ab := λ x, A.2 + slope_ab * (x - A.1)
  let distance_c_to_ab := abs ((B.1 - A.1) * (A.2 - C.2) - (A.1 - C.1) * (B.2 - A.2)) / length_ab
  let triangle_area := (1 / 2) * length_ab * distance_c_to_ab
  C = (-5, -3) ∧ triangle_area = 24

theorem problem_solution :
  ∃ (C : ℝ × ℝ), vertex_C_and_area (5, -2) (7, 3) (0, _ ) (_, 0) C :=
begin
  use (-5, -3),
  sorry
end

end problem_solution_l375_375644


namespace janelle_marbles_l375_375247

variable (initial_green : ℕ) (bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ)

def marbles_left (initial_green bags marbles_per_bag gift_green gift_blue : ℕ) : ℕ :=
  initial_green + (bags * marbles_per_bag) - (gift_green + gift_blue)

theorem janelle_marbles : marbles_left 26 6 10 6 8 = 72 := 
by 
  simp [marbles_left]
  sorry

end janelle_marbles_l375_375247


namespace calories_per_pancake_l375_375248

theorem calories_per_pancake:
  ∃ x : ℕ, 
  (6 * x + 200 + 200 = 1120) ∧ 
  (x = 120) := by
  exists 120
  sorry

end calories_per_pancake_l375_375248


namespace collatz_inequality_l375_375310

def collatz (x : ℕ) : ℕ :=
  if x % 2 = 1 then 3 * x + 1 else x / 2

def collatz_iter (k : ℕ) (x : ℕ) : ℕ :=
  Nat.recOn k x (λ k ih, collatz ih)

theorem collatz_inequality : ∃ x : ℕ, collatz_iter 40 x > 2012 * x := 
  sorry

end collatz_inequality_l375_375310


namespace restore_original_problem_l375_375905

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l375_375905


namespace value_of_x_l375_375584

def vec_a : (ℝ × ℝ) := (1, -2)
def vec_b (x : ℝ) : (ℝ × ℝ) := (x, 1)
def vec_c : (ℝ × ℝ) := (1, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem value_of_x (x : ℝ) (h : dot_product (vec_a.1 + vec_b x.1, vec_a.2 + vec_b x.2) vec_c = 0) : x = 1 :=
by
  sorry

end value_of_x_l375_375584


namespace solution_set_of_f_le_1_maximum_value_of_a_l375_375561

section ProblemDefinitions

def f (x : ℝ) : ℝ := abs (3 * x + 2)

theorem solution_set_of_f_le_1 :
  {x : ℝ | f x ≤ 1} = set.Icc (-1) (-1 / 3) :=
by sorry

theorem maximum_value_of_a (a : ℝ) :
  (∀ x : ℝ, f (x^2) ≥ a * abs x) ↔ a ≤ 2 * real.sqrt 6 :=
by sorry

end ProblemDefinitions

end solution_set_of_f_le_1_maximum_value_of_a_l375_375561


namespace gcd_2183_1947_l375_375099

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := 
by 
  sorry

end gcd_2183_1947_l375_375099


namespace fraction_to_decimal_l375_375034

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l375_375034


namespace find_a5_from_geometric_sequence_l375_375565

def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  geo_seq a q ∧ 0 < a 1 ∧ 0 < q ∧ 
  (a 4 = (a 2) ^ 2) ∧ 
  (a 2 + a 4 = 5 / 16)

theorem find_a5_from_geometric_sequence :
  ∀ (a : ℕ → ℝ) (q : ℝ), geometric_sequence_property a q → 
  a 5 = 1 / 32 :=
by 
  sorry

end find_a5_from_geometric_sequence_l375_375565


namespace fraction_to_decimal_l375_375083

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l375_375083


namespace remove_150_red_balls_l375_375444

def balls_problem (total_balls : ℕ) (initial_red : ℕ) (desired_percentage : ℝ) : ℕ :=
  let initial_blue := total_balls - initial_red in
  let x := initial_red - desired_percentage * (total_balls - x) / (1 - desired_percentage) in
  x

theorem remove_150_red_balls :
  balls_problem 600 420 0.6 = 150 :=
by
  sorry

end remove_150_red_balls_l375_375444


namespace number_of_students_in_school_l375_375325

theorem number_of_students_in_school :
  ∀ (num_classrooms num_students_per_classroom : ℕ),
  num_classrooms = 24 →
  num_students_per_classroom = 5 →
  num_classrooms * num_students_per_classroom = 120 :=
by
  intros num_classrooms num_students_per_classroom h_classrooms h_students
  rw [h_classrooms, h_students]
  norm_num
  sorry

end number_of_students_in_school_l375_375325


namespace document_total_characters_l375_375792

theorem document_total_characters (T : ℕ) : 
  (∃ (t_1 t_2 t_3 : ℕ) (v_A v_B : ℕ),
      v_A = 100 ∧ v_B = 200 ∧
      t_1 = T / 600 ∧
      v_A * t_1 = T / 6 ∧
      v_B * t_1 = T / 3 ∧
      v_A * 3 * 5 = 1500 ∧
      t_2 = (T / 2 - 1500) / 500 ∧
      (v_A * 3 * t_2 + 1500 + v_A * t_1 = v_B * t_1 + v_B * t_2) ∧
      (v_A * 3 * (T - 3000) / 1000 + 1500 + v_A * T / 6 =
       v_B * 2 * (T - 3000) / 10 + v_B * T / 3)) →
  T = 18000 := by
  sorry

end document_total_characters_l375_375792


namespace time_to_clear_driveway_l375_375282

def rate_of_clearing (n : ℕ) : ℕ := 20 - n + 1

def volume_of_driveway : ℕ := 5 * 10 * 4

theorem time_to_clear_driveway : ∑ i in range 13, rate_of_clearing (i + 1) ≥ volume_of_driveway :=
by sorry

end time_to_clear_driveway_l375_375282


namespace only_natural_number_solution_l375_375095

theorem only_natural_number_solution (n : ℕ) :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = n * x * y * z) ↔ (n = 3) := 
sorry

end only_natural_number_solution_l375_375095


namespace necessary_but_not_sufficient_condition_condition_type_l375_375420

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 3 → x > 5) ↔ false  ∧ (x > 5 → x > 3) ↔ true :=
begin
  split,
  { intros h,
    have h1 : 4 > 3 := by norm_num,
    have h2 : ¬(4 > 5) := by norm_num,
    contradiction h h1 h2,
  },
  { intros h,
    exact λ h5, lt_of_lt_of_le (by norm_num) h5 },
end

theorem condition_type : (∀ x : ℝ, x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ ¬ (x > 5)) ↔ "necessary but not sufficient" :=
begin
  split,
  { intros h,
    split,
    { intros h5,
      exact h.1 h5 },
    { by_contra,
      push_neg at h,
      exact lt_irrefl 4 (h.2 4 (by norm_num)) } },
  { split,
    { exact λ x hx, hx.1 },
    { exact ⟨4, by norm_num, by norm_num⟩ } }
end

end necessary_but_not_sufficient_condition_condition_type_l375_375420


namespace inequality_proof_l375_375835

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem inequality_proof (a x : ℝ) (h : |x - a| < 1) : |f(x) - f(a)| < 2 * (|a| + 1) :=
by
  sorry

end inequality_proof_l375_375835


namespace max_value_inequality_max_value_equality_l375_375540

theorem max_value_inequality (x : ℝ) (hx : x < 0) : 
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 :=
sorry

theorem max_value_equality (x : ℝ) (hx : x = -2 * Real.sqrt 3 / 3) : 
  3 * x + 4 / x = -4 * Real.sqrt 3 :=
sorry

end max_value_inequality_max_value_equality_l375_375540


namespace gcd_possible_values_count_l375_375375

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l375_375375
