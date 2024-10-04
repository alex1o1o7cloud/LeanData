import Mathlib

namespace yoki_cans_l23_23608

-- Definitions of the conditions
def total_cans_collected : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_cans := avi_initial_cans / 2

-- Statement that needs to be proved
theorem yoki_cans : ∀ (total_cans_collected ladonna_cans : ℕ) 
  (prikya_cans : ℕ := 2 * ladonna_cans) 
  (avi_initial_cans : ℕ := 8) 
  (avi_cans : ℕ := avi_initial_cans / 2), 
  (total_cans_collected = 85) → 
  (ladonna_cans = 25) → 
  (prikya_cans = 2 * ladonna_cans) →
  (avi_initial_cans = 8) → 
  (avi_cans = avi_initial_cans / 2) → 
  total_cans_collected - (ladonna_cans + prikya_cans + avi_cans) = 6 :=
by
  intros total_cans_collected ladonna_cans prikya_cans avi_initial_cans avi_cans H1 H2 H3 H4 H5
  sorry

end yoki_cans_l23_23608


namespace rectangular_plot_dimensions_l23_23446

theorem rectangular_plot_dimensions (a b : ℝ) 
  (h_area : a * b = 800) 
  (h_perimeter_fencing : 2 * a + b = 100) :
  (a = 40 ∧ b = 20) ∨ (a = 10 ∧ b = 80) := 
sorry

end rectangular_plot_dimensions_l23_23446


namespace colors_of_clothes_l23_23761

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l23_23761


namespace constant_term_expansion_l23_23555

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end constant_term_expansion_l23_23555


namespace reciprocal_of_repeating_decimal_l23_23559

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l23_23559


namespace correct_outfits_l23_23777

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l23_23777


namespace min_colors_required_l23_23161

-- Define predicate for the conditions
def conditions (n : ℕ) (m : ℕ) (k : ℕ)(Paint : ℕ → Set ℕ) : Prop := 
  (∀ S : Finset ℕ, S.card = n → (∃ c ∈ ⋃ p ∈ S, Paint p, c ∈ S)) ∧ 
  (∀ c, ¬ (∀ i ∈ (Finset.range m).1, c ∈ Paint i))

-- The main theorem statement
theorem min_colors_required :
  ∀ (Paint : ℕ → Set ℕ), conditions 20 100 21 Paint → 
  ∃ k, conditions 20 100 k Paint ∧ k = 21 :=
sorry

end min_colors_required_l23_23161


namespace count_balanced_integers_l23_23781

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3) = d1 + (d2 + d3) ∧ (100 ≤ n) ∧ (n ≤ 999)

theorem count_balanced_integers : ∃ c, c = 330 ∧ ∀ n, 100 ≤ n ∧ n ≤ 999 → is_balanced n ↔ c = 330 :=
sorry

end count_balanced_integers_l23_23781


namespace prove_clothing_colors_l23_23759

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l23_23759


namespace symmetric_axis_of_quadratic_fn_l23_23408

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 9 

-- State the theorem that the axis of symmetry for the quadratic function y = x^2 + 8x + 9 is x = -4
theorem symmetric_axis_of_quadratic_fn : ∃ h : ℝ, h = -4 ∧ ∀ x, quadratic_function x = quadratic_function (2 * h - x) :=
by sorry

end symmetric_axis_of_quadratic_fn_l23_23408


namespace power_mod_remainder_l23_23125

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end power_mod_remainder_l23_23125


namespace determine_coefficients_l23_23278

theorem determine_coefficients (A B C : ℝ) 
  (h1 : 3 * A - 1 = 0)
  (h2 : 3 * A^2 + 3 * B = 0)
  (h3 : A^3 + 6 * A * B + 3 * C = 0) :
  A = 1 / 3 ∧ B = -1 / 9 ∧ C = 5 / 81 :=
by 
  sorry

end determine_coefficients_l23_23278


namespace find_correct_grades_l23_23432

structure StudentGrades := 
  (Volodya: ℕ) 
  (Sasha: ℕ) 
  (Petya: ℕ)

def isCorrectGrades (grades : StudentGrades) : Prop :=
  grades.Volodya = 5 ∧ grades.Sasha = 4 ∧ grades.Petya = 3

theorem find_correct_grades (grades : StudentGrades)
  (h1 : grades.Volodya = 5 ∨ grades.Volodya ≠ 5)
  (h2 : grades.Sasha = 3 ∨ grades.Sasha ≠ 3)
  (h3 : grades.Petya ≠ 5 ∨ grades.Petya = 5)
  (unique_h1: grades.Volodya = 5 ∨ grades.Sasha = 5 ∨ grades.Petya = 5) 
  (unique_h2: grades.Volodya = 4 ∨ grades.Sasha = 4 ∨ grades.Petya = 4)
  (unique_h3: grades.Volodya = 3 ∨ grades.Sasha = 3 ∨ grades.Petya = 3) 
  (lyingCount: (grades.Volodya ≠ 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya = 5)
              ∨ (grades.Volodya = 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya ≠ 5)
              ∨ (grades.Volodya ≠ 5 ∧ grades.Sasha = 3 ∧ grades.Petya ≠ 5)) :
  isCorrectGrades grades :=
sorry

end find_correct_grades_l23_23432


namespace correct_outfits_l23_23776

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l23_23776


namespace compute_expression_l23_23263

theorem compute_expression : 7^3 - 5 * (6^2) + 2^4 = 179 :=
by
  sorry

end compute_expression_l23_23263


namespace impossible_to_divide_into_three_similar_piles_l23_23356

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l23_23356


namespace fixed_point_of_f_l23_23534

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 4

theorem fixed_point_of_f (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  -- Skip the proof; it will be filled in the subsequent steps
  sorry

end fixed_point_of_f_l23_23534


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l23_23042

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l23_23042


namespace intersection_points_of_circle_and_vertical_line_l23_23107

theorem intersection_points_of_circle_and_vertical_line :
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (3, y1) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y1) ≠ (3, y2)) := 
by
  sorry

end intersection_points_of_circle_and_vertical_line_l23_23107


namespace green_peaches_sum_l23_23542

theorem green_peaches_sum (G1 G2 G3 : ℕ) : 
  (4 + G1) + (4 + G2) + (3 + G3) = 20 → G1 + G2 + G3 = 9 :=
by
  intro h
  sorry

end green_peaches_sum_l23_23542


namespace problem_f8_minus_f4_l23_23150

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 2

theorem problem_f8_minus_f4 : f 8 - f 4 = -1 :=
by sorry

end problem_f8_minus_f4_l23_23150


namespace intersection_sets_l23_23292

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l23_23292


namespace equivalent_problem_l23_23905

theorem equivalent_problem : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 := by
  sorry

end equivalent_problem_l23_23905


namespace identify_clothing_l23_23774

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l23_23774


namespace hyperbola_center_l23_23718

def is_midpoint (x1 y1 x2 y2 xc yc : ℝ) : Prop :=
  xc = (x1 + x2) / 2 ∧ yc = (y1 + y2) / 2

theorem hyperbola_center :
  is_midpoint 2 (-3) (-4) 5 (-1) 1 :=
by
  sorry

end hyperbola_center_l23_23718


namespace red_marbles_in_bag_l23_23243

theorem red_marbles_in_bag (T R : ℕ) (hT : T = 84)
    (probability_not_red : ((T - R : ℚ) / T)^2 = 36 / 49) : 
    R = 12 := 
sorry

end red_marbles_in_bag_l23_23243


namespace sequence_term_formula_l23_23537

open Real

def sequence_sum_condition (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → S n + a n = 4 - 1 / (2 ^ (n - 2))

theorem sequence_term_formula 
  (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h : sequence_sum_condition S a) :
  ∀ n : ℕ, n > 0 → a n = n / 2 ^ (n - 1) :=
sorry

end sequence_term_formula_l23_23537


namespace intersection_A_B_is_C_l23_23296

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l23_23296


namespace exists_triangle_with_prime_angles_l23_23850

-- Definition of prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Definition of being an angle of a triangle
def is_valid_angle (α : ℕ) : Prop := α > 0 ∧ α < 180

-- Main statement
theorem exists_triangle_with_prime_angles :
  ∃ (α β γ : ℕ), is_prime α ∧ is_prime β ∧ is_prime γ ∧ is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧ α + β + γ = 180 :=
by
  sorry

end exists_triangle_with_prime_angles_l23_23850


namespace minimize_fence_perimeter_l23_23074

-- Define the area of the pen
def area (L W : ℝ) : ℝ := L * W

-- Define that only three sides of the fence need to be fenced
def perimeter (L W : ℝ) : ℝ := 2 * W + L

-- Given conditions
def A : ℝ := 54450  -- Area in square meters

-- The proof statement
theorem minimize_fence_perimeter :
  ∃ (L W : ℝ), 
  area L W = A ∧ 
  ∀ (L' W' : ℝ), area L' W' = A → perimeter L W ≤ perimeter L' W' ∧ L = 330 ∧ W = 165 :=
sorry

end minimize_fence_perimeter_l23_23074


namespace complementary_angles_positive_difference_l23_23399

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l23_23399


namespace kolya_or_leva_wins_l23_23980

variable (k l : ℝ) -- Defining k and l as real numbers

-- Define a function to check the winner based on the lengths k and l
def determine_winner (k l : ℝ) : String :=
  if k > l then "Kolya" else "Leva"

-- The theorem to prove our solution statement
theorem kolya_or_leva_wins (k l : ℝ) (hk : 0 < k) (hl : 0 < l) :
  (determine_winner k l = "Kolya" ↔ k > l) ∧ (determine_winner k l = "Leva" ↔ k ≤ l) :=
by
  split
  · split
    · intro h
      simp [determine_winner] at h
      exact h
    · intro h
      simp [determine_winner]
      exact h
  · split
    · intro h
      simp [determine_winner] at h
      exact h
    · intro h
      simp [determine_winner]
      exact h

end kolya_or_leva_wins_l23_23980


namespace trains_clear_in_approx_6_85_seconds_l23_23898

noncomputable def length_first_train : ℝ := 111
noncomputable def length_second_train : ℝ := 165
noncomputable def speed_first_train : ℝ := 80 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def speed_second_train : ℝ := 65 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_in_approx_6_85_seconds : abs (time_to_clear - 6.85) < 0.01 := sorry

end trains_clear_in_approx_6_85_seconds_l23_23898


namespace alexis_total_sewing_time_l23_23751

-- Define the time to sew a skirt and a coat
def t_skirt : ℕ := 2
def t_coat : ℕ := 7

-- Define the numbers of skirts and coats
def n_skirts : ℕ := 6
def n_coats : ℕ := 4

-- Define the total time
def total_time : ℕ := t_skirt * n_skirts + t_coat * n_coats

-- State the theorem
theorem alexis_total_sewing_time : total_time = 40 :=
by
  -- the proof would go here; we're skipping the proof as per instructions
  sorry

end alexis_total_sewing_time_l23_23751


namespace mean_of_five_numbers_l23_23211

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l23_23211


namespace triangle_obtuse_l23_23655

theorem triangle_obtuse
  (A B : ℝ) 
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (h : Real.cos A > Real.sin B) : 
  π / 2 < π - (A + B) ∧ π - (A + B) < π :=
by
  sorry

end triangle_obtuse_l23_23655


namespace problem_statement_l23_23516

theorem problem_statement (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) (h₃ : x + y + z = 0) (h₄ : xy + xz + yz ≠ 0) : 
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z)) = -7 :=
by
  sorry

end problem_statement_l23_23516


namespace more_campers_afternoon_than_morning_l23_23579

def campers_morning : ℕ := 52
def campers_afternoon : ℕ := 61

theorem more_campers_afternoon_than_morning : campers_afternoon - campers_morning = 9 :=
by
  -- proof goes here
  sorry

end more_campers_afternoon_than_morning_l23_23579


namespace smallest_y_value_in_set_l23_23724

theorem smallest_y_value_in_set : ∀ y : ℕ, (0 < y) ∧ (y + 4 ≤ 8) → y = 4 :=
by
  intros y h
  have h1 : y + 4 ≤ 8 := h.2
  have h2 : 0 < y := h.1
  sorry

end smallest_y_value_in_set_l23_23724


namespace sum_remainder_l23_23726

theorem sum_remainder (a b c : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 14) (h3 : c % 53 = 9) : 
  (a + b + c) % 53 = 3 := 
by 
  sorry

end sum_remainder_l23_23726


namespace constant_term_expansion_l23_23553

theorem constant_term_expansion (x : ℝ) :
  (constant_term ((5 * x + 2 / (5 * x)) ^ 8) = 1120) := by
  sorry

end constant_term_expansion_l23_23553


namespace geometric_sequence_problem_l23_23287

variable {a : ℕ → ℝ}

-- Given conditions
def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q r, (∀ n, a (n + 1) = q * a n ∧ a 0 = r)

-- Define the conditions from the problem
def condition1 (a : ℕ → ℝ) :=
  a 3 + a 6 = 6

def condition2 (a : ℕ → ℝ) :=
  a 5 + a 8 = 9

-- Theorem to be proved
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (hgeom : geometric_sequence a)
  (h1 : condition1 a)
  (h2 : condition2 a) :
  a 7 + a 10 = 27 / 2 :=
sorry

end geometric_sequence_problem_l23_23287


namespace complex_equation_solution_l23_23621

variable (a b : ℝ)

theorem complex_equation_solution :
  (1 + 2 * complex.I) * a + b = 2 * complex.I → a = 1 ∧ b = -1 :=
by
  sorry

end complex_equation_solution_l23_23621


namespace parabola_point_distance_l23_23888

theorem parabola_point_distance (x y : ℝ) (h : y^2 = 2 * x) (d : ℝ) (focus_x : ℝ) (focus_y : ℝ) :
    focus_x = 1/2 → focus_y = 0 → d = 3 →
    (x + 1/2 = d) → x = 5/2 :=
by
  intros h_focus_x h_focus_y h_d h_dist
  sorry

end parabola_point_distance_l23_23888


namespace michelle_oranges_l23_23683

theorem michelle_oranges (x : ℕ) 
  (h1 : x - x / 3 - 5 = 7) : x = 18 :=
by
  -- We would normally provide the proof here, but it's omitted according to the instructions.
  sorry

end michelle_oranges_l23_23683


namespace max_a_for_integer_roots_l23_23488

theorem max_a_for_integer_roots (a : ℕ) :
  (∀ x : ℤ, x^2 - 2 * (a : ℤ) * x + 64 = 0 → (∃ y : ℤ, x = y)) →
  (∀ x1 x2 : ℤ, x1 * x2 = 64 ∧ x1 + x2 = 2 * (a : ℤ)) →
  a ≤ 17 := 
sorry

end max_a_for_integer_roots_l23_23488


namespace skateboard_total_distance_is_3720_l23_23077

noncomputable def skateboard_distance : ℕ :=
  let a1 := 10
  let d := 9
  let n := 20
  let flat_time := 10
  let a_n := a1 + (n - 1) * d
  let ramp_distance := n * (a1 + a_n) / 2
  let flat_distance := a_n * flat_time
  ramp_distance + flat_distance

theorem skateboard_total_distance_is_3720 : skateboard_distance = 3720 := 
by
  sorry

end skateboard_total_distance_is_3720_l23_23077


namespace rectangle_ratio_l23_23545

theorem rectangle_ratio (s : ℝ) (w h : ℝ) (h_cond : h = 3 * s) (w_cond : w = 2 * s) :
  h / w = 3 / 2 :=
by
  sorry

end rectangle_ratio_l23_23545


namespace parabola_find_c_l23_23073

theorem parabola_find_c (b c : ℝ) 
  (h1 : (1 : ℝ)^2 + b * 1 + c = 2)
  (h2 : (5 : ℝ)^2 + b * 5 + c = 2) : 
  c = 7 := by
  sorry

end parabola_find_c_l23_23073


namespace triangle_circle_distance_l23_23982

open Real

theorem triangle_circle_distance 
  (DE DF EF : ℝ)
  (hDE : DE = 12) (hDF : DF = 16) (hEF : EF = 20) :
  let s := (DE + DF + EF) / 2
  let K := sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let ra := K / (s - EF)
  let DP := s - DF
  let DQ := s
  let DI := sqrt (DP^2 + r^2)
  let DE := sqrt (DQ^2 + ra^2)
  let distance := DE - DI
  distance = 24 * sqrt 2 - 4 * sqrt 10 :=
by
  sorry

end triangle_circle_distance_l23_23982


namespace arithmetic_geometric_sequence_l23_23480

theorem arithmetic_geometric_sequence
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (f : ℕ → ℝ)
  (h₁ : a 1 = 3)
  (h₂ : b 1 = 1)
  (h₃ : b 2 * S 2 = 64)
  (h₄ : b 3 * S 3 = 960)
  : (∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 8^(n - 1)) ∧ 
    (∀ n, f n = (a n - 1) / (S n + 100)) ∧ 
    (∃ n, f n = 1 / 11 ∧ n = 10) := 
sorry

end arithmetic_geometric_sequence_l23_23480


namespace perpendicular_lines_a_value_l23_23844

theorem perpendicular_lines_a_value :
  ∀ a : ℝ, 
    (∀ x y : ℝ, 2*x + a*y - 7 = 0) → 
    (∀ x y : ℝ, (a-3)*x + y + 4 = 0) → a = 2 :=
by
  sorry

end perpendicular_lines_a_value_l23_23844


namespace jose_bottle_caps_l23_23172

def jose_start : ℕ := 7
def rebecca_gives : ℕ := 2
def final_bottle_caps : ℕ := 9

theorem jose_bottle_caps :
  jose_start + rebecca_gives = final_bottle_caps :=
by
  sorry

end jose_bottle_caps_l23_23172


namespace divide_pile_l23_23358

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l23_23358


namespace f_2007_2007_l23_23983

def f (n : ℕ) : ℕ :=
  n.digits 10 |>.map (fun d => d * d) |>.sum

def f_k : ℕ → ℕ → ℕ
| 0, n => n
| (k+1), n => f (f_k k n)

theorem f_2007_2007 : f_k 2007 2007 = 145 :=
by
  sorry -- Proof omitted

end f_2007_2007_l23_23983


namespace determine_clothes_l23_23769

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l23_23769


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l23_23040

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l23_23040


namespace min_value_f_l23_23403

noncomputable def f (x : ℝ) : ℝ := (8^x + 5) / (2^x + 1)

theorem min_value_f : ∃ x : ℝ, f x = 3 :=
sorry

end min_value_f_l23_23403


namespace twice_plus_eight_lt_five_times_x_l23_23609

theorem twice_plus_eight_lt_five_times_x (x : ℝ) : 2 * x + 8 < 5 * x := 
sorry

end twice_plus_eight_lt_five_times_x_l23_23609


namespace martha_total_clothes_l23_23374

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l23_23374


namespace sum_possible_b_quad_eq_rational_roots_l23_23133

theorem sum_possible_b_quad_eq_rational_roots :
  (∑ b in { b : ℕ | b > 0 ∧ (∃ k : ℕ, 7^2 - 4 * 3 * b = k^2) ∧ b ≤ 4 }, b) = 6 :=
by
  sorry

end sum_possible_b_quad_eq_rational_roots_l23_23133


namespace julian_initial_owing_l23_23979

theorem julian_initial_owing (jenny_owing_initial: ℕ) (borrow: ℕ) (total_owing: ℕ):
    borrow = 8 → total_owing = 28 → jenny_owing_initial + borrow = total_owing → jenny_owing_initial = 20 :=
by intros;
   exact sorry

end julian_initial_owing_l23_23979


namespace fraction_eval_l23_23269

theorem fraction_eval :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 :=
by
  sorry

end fraction_eval_l23_23269


namespace intersection_complement_eq_l23_23182

open Set

variable (U M N : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5} →
  M = {1, 4} →
  N = {1, 3, 5} →
  N ∩ (U \ M) = {3, 5} := by 
sorry

end intersection_complement_eq_l23_23182


namespace powers_of_three_two_digit_count_l23_23826

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l23_23826


namespace johns_total_spent_l23_23669

def total_spent (num_tshirts: Nat) (price_per_tshirt: Nat) (price_pants: Nat): Nat :=
  (num_tshirts * price_per_tshirt) + price_pants

theorem johns_total_spent : total_spent 3 20 50 = 110 := by
  sorry

end johns_total_spent_l23_23669


namespace segment_proportionality_l23_23635

variable (a b c x : ℝ)

theorem segment_proportionality (ha : a ≠ 0) (hc : c ≠ 0) 
  (h : x = a * (b / c)) : 
  (x / a) = (b / c) := 
by
  sorry

end segment_proportionality_l23_23635


namespace intersection_sets_l23_23293

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l23_23293


namespace sum_of_x_coordinates_mod13_intersection_l23_23863

theorem sum_of_x_coordinates_mod13_intersection :
  (∀ x y : ℕ, y ≡ 3 * x + 5 [MOD 13] → y ≡ 7 * x + 4 [MOD 13]) → (x ≡ 10 [MOD 13]) :=
by
  sorry

end sum_of_x_coordinates_mod13_intersection_l23_23863


namespace seats_required_l23_23242

def children := 58
def per_seat := 2
def seats_needed (children : ℕ) (per_seat : ℕ) := children / per_seat

theorem seats_required : seats_needed children per_seat = 29 := 
by
  sorry

end seats_required_l23_23242


namespace ice_cream_melt_l23_23748

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h : ℝ)
  (V_sphere : ℝ := (4 / 3) * Real.pi * r_sphere^3)
  (V_cylinder : ℝ := Real.pi * r_cylinder^2 * h)
  (H_equal_volumes : V_sphere = V_cylinder) :
  h = 4 / 9 := by
  sorry

end ice_cream_melt_l23_23748


namespace gcd_A_B_l23_23026

theorem gcd_A_B (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a > 0) (h3 : b > 0) : 
  Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) ≠ 1 → Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) = 7 :=
by
  sorry

end gcd_A_B_l23_23026


namespace arithmetic_sequence_sum_l23_23333

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h : ∀ n, a n = a 1 + (n - 1) * d) (h_6 : a 6 = 1) :
  a 2 + a 10 = 2 := 
sorry

end arithmetic_sequence_sum_l23_23333


namespace minimum_value_l23_23723

noncomputable def function_y (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1450

theorem minimum_value : ∀ x : ℝ, function_y x ≥ 1438 :=
by 
  intro x
  sorry

end minimum_value_l23_23723


namespace car_speed_l23_23573

variable (v : ℝ)
variable (Distance : ℝ := 1)  -- distance in kilometers
variable (Speed_120 : ℝ := 120)  -- speed in kilometers per hour
variable (Time_120 : ℝ := Distance / Speed_120)  -- time in hours to travel 1 km at 120 km/h
variable (Time_120_sec : ℝ := Time_120 * 3600)  -- time in seconds to travel 1 km at 120 km/h
variable (Additional_time : ℝ := 2)  -- additional time in seconds
variable (Time_v_sec : ℝ := Time_120_sec + Additional_time)  -- time in seconds for unknown speed
variable (Time_v : ℝ := Time_v_sec / 3600)  -- time in hours for unknown speed

theorem car_speed (h : v = Distance / Time_v) : v = 112.5 :=
by
  -- The given proof steps will go here
  sorry

end car_speed_l23_23573


namespace reciprocal_of_repeating_decimal_l23_23561

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l23_23561


namespace number_of_toddlers_l23_23268

-- Definitions based on the conditions provided in the problem
def total_children := 40
def newborns := 4
def toddlers (T : ℕ) := T
def teenagers (T : ℕ) := 5 * T

-- The theorem to prove
theorem number_of_toddlers : ∃ T : ℕ, newborns + toddlers T + teenagers T = total_children ∧ T = 6 :=
by
  sorry

end number_of_toddlers_l23_23268


namespace value_of_mn_squared_l23_23632

theorem value_of_mn_squared (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 3) (h3 : m - n < 0) : (m + n)^2 = 1 ∨ (m + n)^2 = 49 :=
by sorry

end value_of_mn_squared_l23_23632


namespace count_perfect_squares_l23_23494

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def E1 : ℕ := 1^3 + 2^3
def E2 : ℕ := 1^3 + 2^3 + 3^3
def E3 : ℕ := 1^3 + 2^3 + 3^3 + 4^3
def E4 : ℕ := 1^3 + 2^3 + 3^3 + 4^3 + 5^3

theorem count_perfect_squares :
  (is_perfect_square E1 → true) ∧
  (is_perfect_square E2 → true) ∧
  (is_perfect_square E3 → true) ∧
  (is_perfect_square E4 → true) →
  (∀ n : ℕ, (n = 4) ↔
    ∃ E1 E2 E3 E4, is_perfect_square E1 ∧ is_perfect_square E2 ∧ is_perfect_square E3 ∧ is_perfect_square E4) :=
by
  sorry

end count_perfect_squares_l23_23494


namespace chess_mixed_games_l23_23656

theorem chess_mixed_games (W M : ℕ) (hW : W * (W - 1) / 2 = 45) (hM : M * (M - 1) / 2 = 190) : M * W = 200 :=
by
  sorry

end chess_mixed_games_l23_23656


namespace min_primes_to_guarantee_win_l23_23897

theorem min_primes_to_guarantee_win : 
  ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧
    p1 < 100 ∧ p2 < 100 ∧ p3 < 100 ∧
    (p1 % 10 = p2 / 10 % 10) ∧ 
    (p2 % 10 = p3 / 10 % 10) ∧ 
    (p3 % 10 = p1 / 10 % 10) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
  by sorry

end min_primes_to_guarantee_win_l23_23897


namespace clean_per_hour_l23_23096

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end clean_per_hour_l23_23096


namespace price_per_yellow_stamp_l23_23990

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end price_per_yellow_stamp_l23_23990


namespace problem1_problem2_l23_23576

-- Problem 1
theorem problem1 (x: ℚ) (h: x + 1 / 4 = 7 / 4) : x = 3 / 2 :=
by sorry

-- Problem 2
theorem problem2 (x: ℚ) (h: 2 / 3 + x = 3 / 4) : x = 1 / 12 :=
by sorry

end problem1_problem2_l23_23576


namespace problem1_l23_23577

theorem problem1 : 1361 + 972 + 693 + 28 = 3000 :=
by
  sorry

end problem1_l23_23577


namespace total_shaded_area_l23_23078

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  1 * S ^ 2 + 8 * (T ^ 2) = 13.5 := by
  sorry

end total_shaded_area_l23_23078


namespace find_x_l23_23887

theorem find_x (x y z : ℕ) (h1 : x = y / 2) (h2 : y = z / 3) (h3 : z = 90) : x = 15 :=
by
  sorry

end find_x_l23_23887


namespace sum_possible_values_q_l23_23367

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end sum_possible_values_q_l23_23367


namespace solve_system_l23_23013

def system_of_equations (x y : ℝ) : Prop :=
  (4 * (x - y) = 8 - 3 * y) ∧ (x / 2 + y / 3 = 1)

theorem solve_system : ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 0 := 
  by
  sorry

end solve_system_l23_23013


namespace correct_outfits_l23_23779

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l23_23779


namespace sum_S9_l23_23848

variable (a : ℕ → ℤ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Given condition for the sum of specific terms
def condition_given (a : ℕ → ℤ) : Prop :=
  a 2 + a 5 + a 8 = 12

-- Sum of the first 9 terms
def sum_of_first_nine_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8

-- Problem statement: Prove that given the arithmetic sequence and the condition,
-- the sum of the first 9 terms is 36
theorem sum_S9 :
  arithmetic_sequence a → condition_given a → sum_of_first_nine_terms a = 36 :=
by
  intros
  sorry

end sum_S9_l23_23848


namespace find_first_4_hours_speed_l23_23436

noncomputable def average_speed_first_4_hours
  (total_avg_speed : ℝ)
  (first_4_hours_avg_speed : ℝ)
  (remaining_hours_avg_speed : ℝ)
  (total_time : ℕ)
  (first_4_hours : ℕ)
  (remaining_hours : ℕ) : Prop :=
  total_avg_speed * total_time = first_4_hours_avg_speed * first_4_hours + remaining_hours * remaining_hours_avg_speed

theorem find_first_4_hours_speed :
  average_speed_first_4_hours 50 35 53 24 4 20 :=
by
  sorry

end find_first_4_hours_speed_l23_23436


namespace complex_numbers_xyz_l23_23856

theorem complex_numbers_xyz (x y z : ℂ) (h1 : x * y + 5 * y = -20) (h2 : y * z + 5 * z = -20) (h3 : z * x + 5 * x = -20) :
  x * y * z = 100 :=
sorry

end complex_numbers_xyz_l23_23856


namespace people_in_room_l23_23473

/-- 
   Problem: Five-sixths of the people in a room are seated in five-sixths of the chairs.
   The rest of the people are standing. If there are 10 empty chairs, 
   prove that there are 60 people in the room.
-/
theorem people_in_room (people chairs : ℕ) 
  (h_condition1 : 5 / 6 * people = 5 / 6 * chairs) 
  (h_condition2 : chairs = 60) :
  people = 60 :=
by
  sorry

end people_in_room_l23_23473


namespace find_some_number_l23_23654

theorem find_some_number (some_number q x y : ℤ) 
  (h1 : x = some_number + 2 * q) 
  (h2 : y = 4 * q + 41) 
  (h3 : q = 7) 
  (h4 : x = y) : 
  some_number = 55 := 
by 
  sorry

end find_some_number_l23_23654


namespace steps_to_11th_floor_l23_23729

theorem steps_to_11th_floor 
  (steps_between_3_and_5 : ℕ) 
  (third_floor : ℕ := 3) 
  (fifth_floor : ℕ := 5) 
  (eleventh_floor : ℕ := 11) 
  (ground_floor : ℕ := 1) 
  (steps_per_floor : ℕ := steps_between_3_and_5 / (fifth_floor - third_floor)) :
  steps_between_3_and_5 = 42 →
  steps_between_3_and_5 / (fifth_floor - third_floor) = 21 →
  (eleventh_floor - ground_floor) = 10 →
  21 * 10 = 210 := 
by
  intros _ _ _
  exact rfl

end steps_to_11th_floor_l23_23729


namespace solution_set_of_inequality_l23_23876

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, true
axiom f_zero_eq : f 0 = 2
axiom f_derivative_ineq : ∀ x : ℝ, f x + (deriv f x) > 1

theorem solution_set_of_inequality : { x : ℝ | e^x * f x > e^x + 1 } = { x | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l23_23876


namespace solution_to_power_tower_l23_23019

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solution_to_power_tower : ∃ x : ℝ, infinite_power_tower x = 4 ∧ x = Real.sqrt 2 := sorry

end solution_to_power_tower_l23_23019


namespace max_intersections_circle_cos_curve_correct_answer_l23_23467

open Real

-- Definitions of circle and curve
def circle_eq (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 4
def cos_curve (x y : ℝ) := y = cos x

-- Main Statement: Proof that the maximum number of intersection points within the interval [0, 2π] is 4
theorem max_intersections_circle_cos_curve :
  ∃ (P : ℝ → ℝ → Prop), (P = circle_eq) ∧ 
    ∃ (Q : ℝ → ℝ → Prop), (Q = cos_curve) ∧ 
    ∀ (a b c d : ℝ), 
      (0 ≤ a ∧ a ≤ 2 * π) ∧ 
      (0 ≤ b ∧ b ≤ 2 * π) ∧ 
      (0 ≤ c ∧ c ≤ 2 * π) ∧ 
      (0 ≤ d ∧ d ≤ 2 * π) →
      (P a (cos a) ∧ P b (cos b) ∧ P c (cos c) ∧ P d (cos d)) →
      (cos_curve a (cos a) ∧ cos_curve b (cos b) ∧ cos_curve c (cos c) ∧ cos_curve d (cos d)) →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem correct_answer : max_intersections_circle_cos_curve = 4 := sorry

end max_intersections_circle_cos_curve_correct_answer_l23_23467


namespace sum_interior_angles_l23_23390

theorem sum_interior_angles (n : ℕ) (h : 180 * (n - 2) = 3240) : 180 * ((n + 3) - 2) = 3780 := by
  sorry

end sum_interior_angles_l23_23390


namespace volunteer_assignment_correct_l23_23800

def volunteerAssignment : ℕ := 5
def pavilions : ℕ := 4

def numberOfWays (volunteers pavilions : ℕ) : ℕ := 72 -- This is based on the provided correct answer.

theorem volunteer_assignment_correct : 
  numberOfWays volunteerAssignment pavilions = 72 := 
by
  sorry

end volunteer_assignment_correct_l23_23800


namespace find_a_plus_b_l23_23152

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hx : x = a + Real.sqrt b)
  (hxeq : x^2 + 5*x + 5/x + 1/(x^2) = 42) : a + b = 5 :=
sorry

end find_a_plus_b_l23_23152


namespace prime_for_all_k_l23_23672

theorem prime_for_all_k (n : ℕ) (h_n : n ≥ 2) (h_prime : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Prime (k^2 + k + n) :=
by
  intros
  sorry

end prime_for_all_k_l23_23672


namespace matrix_addition_l23_23288

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![-1, 2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 4], ![1, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, -1]]

theorem matrix_addition : A + B = C := by
    sorry

end matrix_addition_l23_23288


namespace problem_number_eq_7_5_l23_23190

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l23_23190


namespace period_started_at_7_am_l23_23946

-- Define the end time of the period
def end_time : ℕ := 16 -- 4 pm in 24-hour format

-- Define the total duration in hours
def duration : ℕ := 9

-- Define the start time of the period
def start_time : ℕ := end_time - duration

-- Prove that the start time is 7 am
theorem period_started_at_7_am : start_time = 7 := by
  sorry

end period_started_at_7_am_l23_23946


namespace monkey_height_37_minutes_l23_23843

noncomputable def monkey_climb (minutes : ℕ) : ℕ :=
if minutes = 37 then 60 else 0

theorem monkey_height_37_minutes : (monkey_climb 37) = 60 := 
by
  sorry

end monkey_height_37_minutes_l23_23843


namespace complementary_angles_positive_difference_l23_23395

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l23_23395


namespace positive_difference_of_b_l23_23343

def g (n : Int) : Int :=
  if n < 0 then n^2 + 3 else 2 * n - 25

theorem positive_difference_of_b :
  let s := g (-3) + g 3
  let t b := g b = -s
  ∃ a b, t a ∧ t b ∧ a ≠ b ∧ |a - b| = 18 :=
by
  sorry

end positive_difference_of_b_l23_23343


namespace Danica_additional_cars_l23_23105

theorem Danica_additional_cars (n : ℕ) (row_size : ℕ) (danica_cars : ℕ) (answer : ℕ) :
  row_size = 8 →
  danica_cars = 37 →
  answer = 3 →
  ∃ k : ℕ, (k + danica_cars) % row_size = 0 ∧ k = answer :=
by
  sorry

end Danica_additional_cars_l23_23105


namespace power_mod_remainder_l23_23126

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end power_mod_remainder_l23_23126


namespace quadratic_inequality_solution_l23_23468

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 3 * x - 18 > 0) ↔ (x < -6 ∨ x > 3) := 
sorry

end quadratic_inequality_solution_l23_23468


namespace total_fence_poles_l23_23919

def num_poles_per_side : ℕ := 27
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

theorem total_fence_poles : 
  (num_poles_per_side * sides_of_square) - corners_of_square = 104 :=
  sorry

end total_fence_poles_l23_23919


namespace distance_from_Martins_house_to_Lawrences_house_l23_23362

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end distance_from_Martins_house_to_Lawrences_house_l23_23362


namespace complementary_angles_positive_difference_l23_23398

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l23_23398


namespace time_to_finish_task_l23_23378

-- Define the conditions
def printerA_rate (total_pages : ℕ) (time_A_alone : ℕ) : ℚ := total_pages / time_A_alone
def printerB_rate (rate_A : ℚ) : ℚ := rate_A + 10

-- Define the combined rate of printers working together
def combined_rate (rate_A : ℚ) (rate_B : ℚ) : ℚ := rate_A + rate_B

-- Define the time taken to finish the task together
def time_to_finish (total_pages : ℕ) (combined_rate : ℚ) : ℚ := total_pages / combined_rate

-- Given conditions
def total_pages : ℕ := 35
def time_A_alone : ℕ := 60

-- Definitions derived from given conditions
def rate_A : ℚ := printerA_rate total_pages time_A_alone
def rate_B : ℚ := printerB_rate rate_A

-- Combined rate when both printers work together
def combined_rate_AB : ℚ := combined_rate rate_A rate_B

-- Lean theorem statement to prove time taken by both printers
theorem time_to_finish_task : time_to_finish total_pages combined_rate_AB = 210 / 67 := 
by
  sorry

end time_to_finish_task_l23_23378


namespace x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l23_23984

variable (x : ℝ)

theorem x_positive_implies_abs_positive (hx : x > 0) : |x| > 0 := sorry

theorem abs_positive_not_necessiarily_x_positive : (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := sorry

theorem x_positive_is_sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ 
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := 
  ⟨x_positive_implies_abs_positive, abs_positive_not_necessiarily_x_positive⟩

end x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l23_23984


namespace part1_solution_part2_solution_l23_23458

theorem part1_solution (x : ℝ) (h1 : (2 * x) / (x - 2) + 3 / (2 - x) = 1) : x = 1 := by
  sorry

theorem part2_solution (x : ℝ) 
  (h1 : 2 * x - 1 ≥ 3 * (x - 1)) 
  (h2 : (5 - x) / 2 < x + 3) : -1 / 3 < x ∧ x ≤ 2 := by
  sorry

end part1_solution_part2_solution_l23_23458


namespace find_k_l23_23153

theorem find_k (k m : ℝ) : (m^2 - 8*m) ∣ (m^3 - k*m^2 - 24*m + 16) → k = 8 := by
  sorry

end find_k_l23_23153


namespace laundry_per_hour_l23_23099

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end laundry_per_hour_l23_23099


namespace cards_ratio_l23_23601

theorem cards_ratio (b_c : ℕ) (m_c : ℕ) (m_l : ℕ) (m_g : ℕ) 
  (h1 : b_c = 20) 
  (h2 : m_c = b_c + 8) 
  (h3 : m_l = 14) 
  (h4 : m_g = m_c - m_l) : 
  m_g / m_c = 1 / 2 :=
by
  sorry

end cards_ratio_l23_23601


namespace complex_eq_solution_l23_23628

variable (a b : ℝ)

theorem complex_eq_solution (h : (1 + 2 * complex.I) * a + b = 2 * complex.I) : a = 1 ∧ b = -1 := 
by
  sorry

end complex_eq_solution_l23_23628


namespace best_selling_price_70_l23_23439

-- Definitions for the conditions in the problem
def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 50

-- The profit function
def profit (x : ℕ) : ℕ :=
(50 + x - purchase_price) * (initial_sales_volume - x)

-- The problem statement to be proved
theorem best_selling_price_70 :
  ∃ x : ℕ, 0 < x ∧ x < 50 ∧ profit x = 900 ∧ (initial_selling_price + x) = 70 :=
by
  sorry

end best_selling_price_70_l23_23439


namespace problem_statement_l23_23571

theorem problem_statement (a b : ℝ) (h : a ≠ b) : (a - b) ^ 2 > 0 := sorry

end problem_statement_l23_23571


namespace log_three_div_square_l23_23865

theorem log_three_div_square (x y : ℝ) (h₁ : x ≠ 1) (h₂ : y ≠ 1) (h₃ : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h₄ : x * y = 243) :
  (Real.log (x / y) / Real.log 3) ^ 2 = 9 := 
sorry

end log_three_div_square_l23_23865


namespace solve_pow_problem_l23_23092

theorem solve_pow_problem : (-2)^1999 + (-2)^2000 = 2^1999 := 
sorry

end solve_pow_problem_l23_23092


namespace remainder_of_largest_divided_by_second_smallest_l23_23893

theorem remainder_of_largest_divided_by_second_smallest 
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  c % b = 1 :=
by
  -- We assume and/or prove the necessary statements here.
  -- The core of the proof uses existing facts or assumptions.
  -- We insert the proof strategy or intermediate steps here.
  
  sorry

end remainder_of_largest_divided_by_second_smallest_l23_23893


namespace sum_of_possible_b_values_l23_23129

theorem sum_of_possible_b_values : 
  (∑ b in { b | b ∈ {1, 2, 3, 4} ∧ ∃ k : ℕ, 49 - 12 * b = k * k }, b) = 6 :=
by 
  sorry

end sum_of_possible_b_values_l23_23129


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l23_23045

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l23_23045


namespace number_of_routes_A_to_B_l23_23088

theorem number_of_routes_A_to_B :
  (∃ f : ℕ × ℕ → ℕ,
  (∀ n m, f (n + 1, m) = f (n, m) + f (n + 1, m - 1)) ∧
  f (0, 0) = 1 ∧ 
  (∀ i, f (i, 0) = 1) ∧ 
  (∀ j, f (0, j) = 1) ∧ 
  f (3, 5) = 23) :=
sorry

end number_of_routes_A_to_B_l23_23088


namespace additional_discount_during_sale_l23_23747

theorem additional_discount_during_sale:
  ∀ (list_price : ℝ) (max_typical_discount_pct : ℝ) (lowest_possible_sale_pct : ℝ),
  30 ≤ max_typical_discount_pct ∧ max_typical_discount_pct ≤ 50 ∧
  lowest_possible_sale_pct = 40 ∧ 
  list_price = 80 →
  ((max_typical_discount_pct * list_price / 100) - (lowest_possible_sale_pct * list_price / 100)) * 100 / 
    (max_typical_discount_pct * list_price / 100) = 20 :=
by
  sorry

end additional_discount_during_sale_l23_23747


namespace find_correct_value_l23_23059

theorem find_correct_value (k : ℕ) (h1 : 173 * 240 = 41520) (h2 : 41520 / 48 = 865) : k * 48 = 173 * 240 → k = 865 :=
by
  intros h
  sorry

end find_correct_value_l23_23059


namespace calculate_gain_percentage_l23_23914

theorem calculate_gain_percentage (CP SP : ℝ) (h1 : 0.9 * CP = 450) (h2 : SP = 550) : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end calculate_gain_percentage_l23_23914


namespace complementary_angles_positive_difference_l23_23394

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l23_23394


namespace product_evaluation_l23_23112

theorem product_evaluation :
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 :=
by sorry

end product_evaluation_l23_23112


namespace number_of_insects_l23_23600

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 30) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 5 :=
by
  sorry

end number_of_insects_l23_23600


namespace isosceles_triangle_perimeter_l23_23965

-- Definitions of the side lengths
def side1 : ℝ := 8
def side2 : ℝ := 4

-- Theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter (side1 side2 : ℝ) (h1 : side1 = 8 ∨ side2 = 8) (h2 : side1 = 4 ∨ side2 = 4) : ∃ p : ℝ, p = 20 :=
by
  -- We omit the proof using sorry
  sorry

end isosceles_triangle_perimeter_l23_23965


namespace price_per_yellow_stamp_l23_23991

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end price_per_yellow_stamp_l23_23991


namespace find_ab_l23_23953

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by 
  sorry

end find_ab_l23_23953


namespace intersection_A_B_is_C_l23_23297

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l23_23297


namespace fraction_of_p_l23_23574

theorem fraction_of_p (p q r f : ℝ) (hp : p = 49) (hqr : p = (2 * f * 49) + 35) : f = 1/7 :=
sorry

end fraction_of_p_l23_23574


namespace two_digit_numbers_of_form_3_pow_n_l23_23836

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l23_23836


namespace largest_shaded_area_of_figures_l23_23066

theorem largest_shaded_area_of_figures :
  let square_area (s : ℝ) := s * s
  let circle_area (r : ℝ) := real.pi * r * r
  
  let shaded_area_A := square_area 3 - circle_area (3 / 2)
  let shaded_area_B := (3 * 4) / 2 - (1 / 4) * real.pi * (2 * 2)
  let shaded_area_C := square_area 2 - circle_area 1
  shaded_area_B > shaded_area_A ∧ shaded_area_B > shaded_area_C := 
by
  -- Proof is not required, so we use sorry
  sorry

end largest_shaded_area_of_figures_l23_23066


namespace exponentiation_rule_l23_23787

theorem exponentiation_rule (b : ℝ) : (-2 * b) ^ 3 = -8 * b ^ 3 :=
by sorry

end exponentiation_rule_l23_23787


namespace geometric_sequence_q_cubed_l23_23633

noncomputable def S (a_1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a_1 else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_q_cubed (a_1 q : ℝ) (h1 : q ≠ 1) (h2 : a_1 ≠ 0)
  (h3 : S a_1 q 3 + S a_1 q 6 = 2 * S a_1 q 9) : q^3 = -1 / 2 :=
by
  sorry

end geometric_sequence_q_cubed_l23_23633


namespace prob_of_point_below_x_axis_l23_23285

noncomputable def probability_point_below_x_axis (a : ℝ) (ha : a ∈ set.Icc (-1 : ℝ) 2) : ℚ :=
  (set.Icc (-1 : ℝ) 0).measure / (set.Icc (-1 : ℝ) 2).measure

theorem prob_of_point_below_x_axis : probability_point_below_x_axis = (1 / 3 : ℚ) := by
  sorry

end prob_of_point_below_x_axis_l23_23285


namespace bouquet_count_l23_23583

theorem bouquet_count : ∃ n : ℕ, n = 9 ∧ ∀ (r c : ℕ), 3 * r + 2 * c = 50 → n = 9 :=
by
  sorry

end bouquet_count_l23_23583


namespace possible_value_of_b_l23_23330

-- Definition of the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Condition for the linear function to pass through the second, third, and fourth quadrants
def passes_second_third_fourth_quadrants (b : ℝ) : Prop :=
  b < 0

-- Lean 4 statement expressing the problem
theorem possible_value_of_b (b : ℝ) (h : passes_second_third_fourth_quadrants b) : b = -1 :=
  sorry

end possible_value_of_b_l23_23330


namespace maximize_profit_l23_23572

noncomputable def profit_function (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 40 then
  -2 * x^2 + 120 * x - 300
else if 40 < x ∧ x ≤ 100 then
  -x - 3600 / x + 1800
else
  0

theorem maximize_profit :
  profit_function 60 = 1680 ∧
  ∀ x, 0 < x ∧ x ≤ 100 → profit_function x ≤ 1680 := 
sorry

end maximize_profit_l23_23572


namespace intersection_A_B_is_C_l23_23295

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l23_23295


namespace earnings_per_day_correct_l23_23027

-- Given conditions
variable (total_earned : ℕ) (days : ℕ) (earnings_per_day : ℕ)

-- Specify the given values from the conditions
def given_conditions : Prop :=
  total_earned = 165 ∧ days = 5 ∧ total_earned = days * earnings_per_day

-- Statement of the problem: proving the earnings per day
theorem earnings_per_day_correct (h : given_conditions total_earned days earnings_per_day) : 
  earnings_per_day = 33 :=
by
  sorry

end earnings_per_day_correct_l23_23027


namespace simplify_expression_l23_23095

theorem simplify_expression (a : ℝ) : (2 * a - 3)^2 - (a + 5) * (a - 5) = 3 * a^2 - 12 * a + 34 :=
by
  sorry

end simplify_expression_l23_23095


namespace triangle_side_length_l23_23336

variables {BC AC : ℝ} {α β γ : ℝ}

theorem triangle_side_length :
  α = 45 ∧ β = 75 ∧ AC = 6 ∧ α + β + γ = 180 →
  BC = 6 * (Real.sqrt 3 - 1) :=
by
  intros h
  sorry

end triangle_side_length_l23_23336


namespace largest_part_of_proportional_division_l23_23495

theorem largest_part_of_proportional_division :
  ∀ (x y z : ℝ),
    x + y + z = 120 ∧
    x / (1 / 2) = y / (1 / 4) ∧
    x / (1 / 2) = z / (1 / 6) →
    max x (max y z) = 60 :=
by sorry

end largest_part_of_proportional_division_l23_23495


namespace triangle_prime_sides_l23_23276

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_prime_sides :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ 
  a + b + c = 25 ∧
  (a = b ∨ b = c ∨ a = c) ∧
  (∀ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 25 → (x, y, z) = (3, 11, 11) ∨ (x, y, z) = (7, 7, 11)) :=
by
  sorry

end triangle_prime_sides_l23_23276


namespace set_difference_A_B_l23_23519

-- Defining the sets A and B
def setA : Set ℝ := { x : ℝ | abs (4 * x - 1) > 9 }
def setB : Set ℝ := { x : ℝ | x >= 0 }

-- The theorem stating the result of set difference A - B
theorem set_difference_A_B : (setA \ setB) = { x : ℝ | x > 5/2 } :=
by
  -- Proof omitted
  sorry

end set_difference_A_B_l23_23519


namespace probability_even_sum_l23_23279

open Finset

theorem probability_even_sum :
  let cards := {1, 2, 3, 4, 5} in
  let card_combinations := cards.to_finset.subsets 2 in
  let even_sum_combinations := card_combinations.filter (λ s, (s.sum id) % 2 = 0) in
  (even_sum_combinations.card.to_nat / card_combinations.card.to_nat) = (2 / 5) := 
sorry

end probability_even_sum_l23_23279


namespace simplify_expression_l23_23008

-- Define the original expression and the simplified version
def original_expr (x y : ℤ) : ℤ := 7 * x + 3 - 2 * x + 15 + y
def simplified_expr (x y : ℤ) : ℤ := 5 * x + y + 18

-- The equivalence to be proved
theorem simplify_expression (x y : ℤ) : original_expr x y = simplified_expr x y :=
by sorry

end simplify_expression_l23_23008


namespace ball_color_probability_l23_23114

def balls := fin 8               -- Define the balls as elements of a finite type of size 8.

-- Define a function to calculate the probability that each ball is different from more than half of the other balls.
def prob_diff_color_from_half (p : rat) : Prop :=
  p = 7/32

theorem ball_color_probability (h : ∀ b : balls, fin 2) :
  prob_diff_color_from_half (1 / 2 ^ 8 * choose 8 5 + 1 / 2 ^ 8 * choose 8 3) :=
sorry

end ball_color_probability_l23_23114


namespace cost_of_rice_l23_23967

-- Define the cost variables
variables (E R K : ℝ)

-- State the conditions as assumptions
def conditions (E R K : ℝ) : Prop :=
  (E = R) ∧
  (K = (2 / 3) * E) ∧
  (2 * K = 48)

-- State the theorem to be proven
theorem cost_of_rice (E R K : ℝ) (h : conditions E R K) : R = 36 :=
by
  sorry

end cost_of_rice_l23_23967


namespace triangle_angle_sum_l23_23596

theorem triangle_angle_sum (CD CB : ℝ) 
    (isosceles_triangle: CD = CB)
    (interior_pentagon_angle: 108 = 180 * (5 - 2) / 5)
    (interior_triangle_angle: 60 = 180 / 3)
    (triangle_angle_sum: ∀ (a b c : ℝ), a + b + c = 180) :
    mangle_CDB = 6 :=
by
  have x : ℝ := 6
  sorry

end triangle_angle_sum_l23_23596


namespace triangle_sides_are_6_8_10_l23_23536

theorem triangle_sides_are_6_8_10 (a b c r r1 r2 r3 : ℕ) (hr_even : Even r) (hr1_even : Even r1) 
(hr2_even : Even r2) (hr3_even : Even r3) (relationship : r * r1 * r2 + r * r2 * r3 + r * r3 * r1 + r1 * r2 * r3 = r * r1 * r2 * r3) :
  (a, b, c) = (6, 8, 10) :=
sorry

end triangle_sides_are_6_8_10_l23_23536


namespace carl_garden_area_l23_23461

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end carl_garden_area_l23_23461


namespace cells_count_at_day_8_l23_23245

theorem cells_count_at_day_8 :
  let initial_cells := 3
  let common_ratio := 2
  let days := 8
  let interval := 2
  ∃ days_intervals, days_intervals = days / interval ∧ initial_cells * common_ratio ^ days_intervals = 48 :=
by
  sorry

end cells_count_at_day_8_l23_23245


namespace tangent_line_at_1_l23_23020

def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_1 (f' : ℝ → ℝ) (h1 : ∀ x, deriv f x = f' x) (h2 : ∀ y, 2 * 1 + y - 3 = 0) :
  f' 1 + f 1 = -1 :=
by
  sorry

end tangent_line_at_1_l23_23020


namespace ceil_floor_arith_l23_23259

theorem ceil_floor_arith :
  (Int.ceil (((15: ℚ) / 8)^2 * (-34 / 4)) - Int.floor ((15 / 8) * Int.floor (-34 / 4))) = -12 :=
by sorry

end ceil_floor_arith_l23_23259


namespace determinant_condition_l23_23599

theorem determinant_condition (a b c d : ℤ)
    (H : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by 
  sorry

end determinant_condition_l23_23599


namespace mean_of_five_numbers_l23_23218

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l23_23218


namespace find_a_l23_23637

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -f a x) ∧ a > 0 ∧ a ≠ 1 → a = 4 :=
by
  sorry

end find_a_l23_23637


namespace complex_eq_solution_l23_23629

variable (a b : ℝ)

theorem complex_eq_solution (h : (1 + 2 * complex.I) * a + b = 2 * complex.I) : a = 1 ∧ b = -1 := 
by
  sorry

end complex_eq_solution_l23_23629


namespace find_seven_m_squared_minus_one_l23_23487

theorem find_seven_m_squared_minus_one (m : ℝ)
  (h1 : ∃ x₁, 5 * m + 3 * x₁ = 1 + x₁)
  (h2 : ∃ x₂, 2 * x₂ + m = 3 * m)
  (h3 : ∀ x₁ x₂, (5 * m + 3 * x₁ = 1 + x₁) → (2 * x₂ + m = 3 * m) → x₁ = x₂ + 2) :
  7 * m^2 - 1 = 2 / 7 :=
by
  let m := -3/7
  sorry

end find_seven_m_squared_minus_one_l23_23487


namespace clothes_color_proof_l23_23755

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l23_23755


namespace continuity_f1_continuity_f2_l23_23234

variable {x : ℝ}

theorem continuity_f1 : Continuous (λ x, x^4 + 3 * x + 5) :=
  by continuity

theorem continuity_f2 : Continuous (λ x, x^2 * sin x - x^2 / (x^2 + 3)) :=
  by continuity

end continuity_f1_continuity_f2_l23_23234


namespace fish_population_estimate_l23_23548

theorem fish_population_estimate :
  (∀ (x : ℕ),
    ∃ (m n k : ℕ), 
      m = 30 ∧
      k = 2 ∧
      n = 30 ∧
      ((k : ℚ) / n = m / x) → x = 450) :=
by
  sorry

end fish_population_estimate_l23_23548


namespace complete_the_square_l23_23696

theorem complete_the_square (y : ℝ) : (y^2 + 12*y + 40) = (y + 6)^2 + 4 := by
  sorry

end complete_the_square_l23_23696


namespace billy_piles_l23_23089

theorem billy_piles (Q D : ℕ) (h : 2 * Q + 3 * D = 20) :
  Q = 4 ∧ D = 4 :=
sorry

end billy_piles_l23_23089


namespace martin_walk_distance_l23_23365

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end martin_walk_distance_l23_23365


namespace one_head_two_tails_probability_l23_23425

noncomputable def probability_of_one_head_two_tails :=
  let total_outcomes := 8
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem one_head_two_tails_probability :
  probability_of_one_head_two_tails = 3 / 8 :=
by
  -- Proof would go here
  sorry

end one_head_two_tails_probability_l23_23425


namespace impossible_divide_into_three_similar_parts_l23_23346

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l23_23346


namespace positive_difference_complementary_angles_l23_23401

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l23_23401


namespace reciprocal_of_repeating_decimal_l23_23565

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l23_23565


namespace rational_number_addition_l23_23050

theorem rational_number_addition :
  (-206 : ℚ) + (401 + 3 / 4) + (-(204 + 2 / 3)) + (-(1 + 1 / 2)) = -10 - 5 / 12 :=
by
  sorry

end rational_number_addition_l23_23050


namespace find_a_if_f_is_odd_l23_23639

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a_if_f_is_odd (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = - f a x) → a = 4 :=
by
  sorry

end find_a_if_f_is_odd_l23_23639


namespace estimate_total_fish_in_pond_l23_23228

theorem estimate_total_fish_in_pond :
  ∀ (total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample : ℕ),
  initial_sample_size = 100 →
  second_sample_size = 200 →
  tagged_in_second_sample = 10 →
  total_tagged_fish = 100 →
  (total_tagged_fish : ℚ) / (total_fish : ℚ) = tagged_in_second_sample / second_sample_size →
  total_fish = 2000 := by
  intros total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample
  intro h1 h2 h3 h4 h5
  sorry

end estimate_total_fish_in_pond_l23_23228


namespace M_gt_N_l23_23705

-- Define the variables and conditions
variables (x y : ℝ)
noncomputable def M := x^2 + y^2
noncomputable def N := 2*x + 6*y - 11

-- State the theorem
theorem M_gt_N : M x y > N x y := by
  sorry -- Placeholder for the proof

end M_gt_N_l23_23705


namespace Bret_catches_12_frogs_l23_23868

-- Conditions from the problem
def frogs_caught_by_Alster : Nat := 2
def frogs_caught_by_Quinn : Nat := 2 * frogs_caught_by_Alster
def frogs_caught_by_Bret : Nat := 3 * frogs_caught_by_Quinn

-- Statement of the theorem to be proved
theorem Bret_catches_12_frogs : frogs_caught_by_Bret = 12 :=
by
  sorry

end Bret_catches_12_frogs_l23_23868


namespace joe_anne_bill_difference_l23_23933

theorem joe_anne_bill_difference (m j a : ℝ) 
  (hm : (15 / 100) * m = 3) 
  (hj : (10 / 100) * j = 2) 
  (ha : (20 / 100) * a = 3) : 
  j - a = 5 := 
by {
  sorry
}

end joe_anne_bill_difference_l23_23933


namespace total_units_per_day_all_work_together_l23_23541

-- Conditions
def men := 250
def women := 150
def units_per_day_by_men := 15
def units_per_day_by_women := 3

-- Problem statement and proof
theorem total_units_per_day_all_work_together :
  units_per_day_by_men + units_per_day_by_women = 18 :=
sorry

end total_units_per_day_all_work_together_l23_23541


namespace find_m_l23_23616

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem find_m (y b : ℝ) (m : ℕ) 
  (h5 : binomial m 4 * y^(m-4) * b^4 = 210) 
  (h6 : binomial m 5 * y^(m-5) * b^5 = 462) 
  (h7 : binomial m 6 * y^(m-6) * b^6 = 792) : 
  m = 7 := 
sorry

end find_m_l23_23616


namespace mean_of_five_numbers_l23_23220

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l23_23220


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l23_23043

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l23_23043


namespace find_single_digit_number_l23_23444

-- Define the given conditions:
def single_digit (A : ℕ) := A < 10
def rounded_down_tens (x : ℕ) (result: ℕ) := (x / 10) * 10 = result

-- Lean statement of the problem:
theorem find_single_digit_number (A : ℕ) (H1 : single_digit A) (H2 : rounded_down_tens (A * 1000 + 567) 2560) : A = 2 :=
sorry

end find_single_digit_number_l23_23444


namespace mean_of_five_numbers_l23_23214

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l23_23214


namespace complex_equation_solution_l23_23623

theorem complex_equation_solution (a b : ℝ) : (1 + (2:ℂ) * complex.I) * a + b = 2 * complex.I → 
  a = 1 ∧ b = -1 :=
by
  intro h
  sorry

end complex_equation_solution_l23_23623


namespace tower_height_l23_23959

theorem tower_height (h : ℝ) (α : ℝ)
  (h_tan_α : tan α = h / 48)
  (h_tan_2α : tan (2 * α) = h / 18) : h = 24 :=
by
  sorry

end tower_height_l23_23959


namespace youngest_child_age_possible_l23_23072

theorem youngest_child_age_possible 
  (total_bill : ℝ) (mother_charge : ℝ) 
  (yearly_charge_per_child : ℝ) (minimum_charge_per_child : ℝ) 
  (num_children : ℤ) (children_total_bill : ℝ)
  (total_years : ℤ)
  (youngest_possible_age : ℤ) :
  total_bill = 15.30 →
  mother_charge = 6 →
  yearly_charge_per_child = 0.60 →
  minimum_charge_per_child = 0.90 →
  num_children = 3 →
  children_total_bill = total_bill - mother_charge →
  children_total_bill - num_children * minimum_charge_per_child = total_years * yearly_charge_per_child →
  total_years = 11 →
  youngest_possible_age = 1 :=
sorry

end youngest_child_age_possible_l23_23072


namespace nancy_rose_bracelets_l23_23861

-- Definitions based on conditions
def metal_beads_nancy : ℕ := 40
def pearl_beads_nancy : ℕ := metal_beads_nancy + 20
def total_beads_nancy : ℕ := metal_beads_nancy + pearl_beads_nancy

def crystal_beads_rose : ℕ := 20
def stone_beads_rose : ℕ := 2 * crystal_beads_rose
def total_beads_rose : ℕ := crystal_beads_rose + stone_beads_rose

def number_of_bracelets (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

-- Theorem to be proved
theorem nancy_rose_bracelets : number_of_bracelets (total_beads_nancy + total_beads_rose) 8 = 20 := 
by
  -- Definitions will be expanded here
  sorry

end nancy_rose_bracelets_l23_23861


namespace problem1_eval_problem2_eval_l23_23788

-- Problem 1 equivalent proof problem
theorem problem1_eval : |(-2 + 1/4)| - (-3/4) + 1 - |(1 - 1/2)| = 3 + 1/2 := 
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2_eval : -3^2 - (8 / (-2)^3 - 1) + 3 / 2 * (1 / 2) = -6 + 1/4 :=
by
  sorry

end problem1_eval_problem2_eval_l23_23788


namespace largest_difference_l23_23855

noncomputable def A : ℕ := 3 * 2010 ^ 2011
noncomputable def B : ℕ := 2010 ^ 2011
noncomputable def C : ℕ := 2009 * 2010 ^ 2010
noncomputable def D : ℕ := 3 * 2010 ^ 2010
noncomputable def E : ℕ := 2010 ^ 2010
noncomputable def F : ℕ := 2010 ^ 2009

theorem largest_difference :
  (A - B = 2 * 2010 ^ 2011) ∧ 
  (B - C = 2010 ^ 2010) ∧ 
  (C - D = 2006 * 2010 ^ 2010) ∧ 
  (D - E = 2 * 2010 ^ 2010) ∧ 
  (E - F = 2009 * 2010 ^ 2009) ∧ 
  (2 * 2010 ^ 2011 > 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2006 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2009 * 2010 ^ 2009) :=
by
  sorry

end largest_difference_l23_23855


namespace smallest_perfect_square_divisible_by_5_and_6_l23_23047

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l23_23047


namespace sum_of_valid_b_values_l23_23130

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l23_23130


namespace david_spent_difference_l23_23792

-- Define the initial amount, remaining amount, amount spent and the correct answer
def initial_amount : Real := 1800
def remaining_amount : Real := 500
def spent_amount : Real := initial_amount - remaining_amount
def correct_difference : Real := spent_amount - remaining_amount

-- Prove that the difference between the amount spent and the remaining amount is $800
theorem david_spent_difference : correct_difference = 800 := by
  sorry

end david_spent_difference_l23_23792


namespace edward_garage_sale_games_l23_23469

variables
  (G_total : ℕ) -- total number of games
  (G_good : ℕ) -- number of good games
  (G_bad : ℕ) -- number of bad games
  (G_friend : ℕ) -- number of games bought from a friend
  (G_garage : ℕ) -- number of games bought at the garage sale

-- The conditions
def total_games (G_total : ℕ) (G_good : ℕ) (G_bad : ℕ) : Prop :=
  G_total = G_good + G_bad

def garage_sale_games (G_total : ℕ) (G_friend : ℕ) (G_garage : ℕ) : Prop :=
  G_total = G_friend + G_garage

-- The theorem to be proved
theorem edward_garage_sale_games
  (G_total : ℕ) 
  (G_good : ℕ) 
  (G_bad : ℕ)
  (G_friend : ℕ) 
  (G_garage : ℕ) 
  (h1 : total_games G_total G_good G_bad)
  (h2 : G_good = 24)
  (h3 : G_bad = 31)
  (h4 : G_friend = 41) :
  G_garage = 14 :=
by
  sorry

end edward_garage_sale_games_l23_23469


namespace problem_l23_23144

theorem problem (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a^4 + b^4 = 228) :
  a * b = 8 :=
sorry

end problem_l23_23144


namespace find_sixth_term_of_geometric_sequence_l23_23275

noncomputable def common_ratio (a b : ℚ) : ℚ := b / a

noncomputable def geometric_sequence_term (a r : ℚ) (k : ℕ) : ℚ := a * (r ^ (k - 1))

theorem find_sixth_term_of_geometric_sequence :
  geometric_sequence_term 5 (common_ratio 5 1.25) 6 = 5 / 1024 :=
by
  sorry

end find_sixth_term_of_geometric_sequence_l23_23275


namespace find_real_numbers_l23_23630

theorem find_real_numbers (a b : ℝ) (h : (1 : ℂ) + 2 * complex.i) * (a : ℂ) + (b : ℂ) = 2 * complex.i) :
  a = 1 ∧ b = -1 := 
by {
  sorry
}

end find_real_numbers_l23_23630


namespace possible_remainders_of_a2_l23_23339

theorem possible_remainders_of_a2 (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 0 < k) 
  (hresidue : ∀ i : ℕ, i < p → ∃ j : ℕ, j < p ∧ ((j^k+j) % p = i)) :
  ∃ s : Finset ℕ, s = Finset.range p ∧ (2^k + 2) % p ∈ s := 
sorry

end possible_remainders_of_a2_l23_23339


namespace rectangular_garden_width_l23_23575

-- Define the problem conditions as Lean definitions
def rectangular_garden_length (w : ℝ) : ℝ := 3 * w
def rectangular_garden_area (w : ℝ) : ℝ := rectangular_garden_length w * w

-- This is the theorem we want to prove
theorem rectangular_garden_width : ∃ w : ℝ, rectangular_garden_area w = 432 ∧ w = 12 :=
by
  sorry

end rectangular_garden_width_l23_23575


namespace portion_of_pizza_eaten_l23_23740

-- Define the conditions
def total_slices : ℕ := 16
def slices_left : ℕ := 4
def slices_eaten : ℕ := total_slices - slices_left

-- Define the portion of pizza eaten
def portion_eaten := (slices_eaten : ℚ) / (total_slices : ℚ)

-- Statement to prove
theorem portion_of_pizza_eaten : portion_eaten = 3 / 4 :=
by sorry

end portion_of_pizza_eaten_l23_23740


namespace maria_compensation_l23_23185

def deposit_insurance (deposit : ℕ) : ℕ :=
  if deposit <= 1400000 then deposit else 1400000

def maria_deposit : ℕ := 1600000

theorem maria_compensation :
  deposit_insurance maria_deposit = 1400000 :=
by sorry

end maria_compensation_l23_23185


namespace kevin_total_miles_l23_23607

theorem kevin_total_miles : 
  ∃ (d1 d2 d3 d4 d5 : ℕ), 
  d1 = 60 / 6 ∧ 
  d2 = 60 / (6 + 6 * 1) ∧ 
  d3 = 60 / (6 + 6 * 2) ∧ 
  d4 = 60 / (6 + 6 * 3) ∧ 
  d5 = 60 / (6 + 6 * 4) ∧ 
  (d1 + d2 + d3 + d4 + d5) = 13 := 
by
  sorry

end kevin_total_miles_l23_23607


namespace draw_two_green_marbles_probability_l23_23069

theorem draw_two_green_marbles_probability :
  let red := 5
  let green := 3
  let white := 7
  let total := red + green + white
  (green / total) * ((green - 1) / (total - 1)) = 1 / 35 :=
by
  sorry

end draw_two_green_marbles_probability_l23_23069


namespace harry_weekly_earnings_l23_23324

def dogs_walked_MWF := 7
def dogs_walked_Tue := 12
def dogs_walked_Thu := 9
def pay_per_dog := 5

theorem harry_weekly_earnings : 
  dogs_walked_MWF * pay_per_dog * 3 + dogs_walked_Tue * pay_per_dog + dogs_walked_Thu * pay_per_dog = 210 :=
by
  sorry

end harry_weekly_earnings_l23_23324


namespace max_path_length_correct_l23_23247

noncomputable def maxFlyPathLength : ℝ :=
  2 * Real.sqrt 2 + Real.sqrt 6 + 6

theorem max_path_length_correct :
  ∀ (fly_path_length : ℝ), (fly_path_length = maxFlyPathLength) :=
by
  intro fly_path_length
  sorry

end max_path_length_correct_l23_23247


namespace intersection_of_A_and_B_l23_23309

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l23_23309


namespace mohan_cookies_l23_23521

theorem mohan_cookies :
  ∃ (a : ℕ), 
    (a % 6 = 5) ∧ 
    (a % 7 = 3) ∧ 
    (a % 9 = 7) ∧ 
    (a % 11 = 10) ∧ 
    (a = 1817) :=
sorry

end mohan_cookies_l23_23521


namespace geom_seq_product_l23_23975

theorem geom_seq_product {a : ℕ → ℝ} (h_geom : ∀ n, a (n + 1) = a n * r)
 (h_a1 : a 1 = 1 / 2) (h_a5 : a 5 = 8) : a 2 * a 3 * a 4 = 8 := 
sorry

end geom_seq_product_l23_23975


namespace prove_inequality_l23_23598

noncomputable def inequality_problem :=
  ∀ (x y z : ℝ),
    0 < x ∧ 0 < y ∧ 0 < z ∧ x^2 + y^2 + z^2 = 3 → 
      (x ^ 2009 - 2008 * (x - 1)) / (y + z) + 
      (y ^ 2009 - 2008 * (y - 1)) / (x + z) + 
      (z ^ 2009 - 2008 * (z - 1)) / (x + y) ≥ 
      (x + y + z) / 2

theorem prove_inequality : inequality_problem := 
  by 
    sorry

end prove_inequality_l23_23598


namespace find_integer_triples_l23_23471

theorem find_integer_triples (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ 
  (x = 668 ∧ y = 668 ∧ z = 667) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 667 ∧ y = 668 ∧ z = 668) :=
by sorry

end find_integer_triples_l23_23471


namespace count_multiples_5_or_7_not_35_l23_23149

def count_multiples_5 (n : ℕ) : ℕ := n / 5
def count_multiples_7 (n : ℕ) : ℕ := n / 7
def count_multiples_35 (n : ℕ) : ℕ := n / 35
def inclusion_exclusion (a b c : ℕ) : ℕ := a + b - c

theorem count_multiples_5_or_7_not_35 : 
  inclusion_exclusion (count_multiples_5 3000) (count_multiples_7 3000) (count_multiples_35 3000) = 943 :=
by
  sorry

end count_multiples_5_or_7_not_35_l23_23149


namespace arithmetic_sequence_problem_l23_23948

theorem arithmetic_sequence_problem (q a₁ a₂ a₃ : ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : q > 1)
  (h2 : a₁ + a₂ + a₃ = 7)
  (h3 : a₁ + 3 + a₃ + 4 = 6 * a₂) :
  (∀ n : ℕ, a n = 2^(n-1)) ∧ (∀ n : ℕ, T n = (3 * n - 5) * 2^n + 5) :=
by
  sorry

end arithmetic_sequence_problem_l23_23948


namespace reciprocal_of_repeating_decimal_l23_23558

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l23_23558


namespace alfred_saving_goal_l23_23595

theorem alfred_saving_goal (leftover : ℝ) (monthly_saving : ℝ) (months : ℕ) :
  leftover = 100 → monthly_saving = 75 → months = 12 → leftover + monthly_saving * months = 1000 :=
by
  sorry

end alfred_saving_goal_l23_23595


namespace retail_price_of_washing_machine_l23_23582

variable (a : ℝ)

theorem retail_price_of_washing_machine :
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price 
  retail_price = 1.04 * a :=
by
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price
  sorry -- Proof skipped

end retail_price_of_washing_machine_l23_23582


namespace fraction_increase_by_five_l23_23158

variable (x y : ℝ)

theorem fraction_increase_by_five :
  let f := fun x y => (x * y) / (2 * x - 3 * y)
  f (5 * x) (5 * y) = 5 * (f x y) :=
by
  sorry

end fraction_increase_by_five_l23_23158


namespace problem_divisible_by_1946_l23_23904

def F (n : ℕ) : ℤ := 1492 ^ n - 1770 ^ n - 1863 ^ n + 2141 ^ n

theorem problem_divisible_by_1946 
  (n : ℕ) 
  (hn : n ≤ 1945) : 
  1946 ∣ F n :=
sorry

end problem_divisible_by_1946_l23_23904


namespace factorize_a_cubed_minus_25a_l23_23120

variable {a : ℝ}

theorem factorize_a_cubed_minus_25a (a : ℝ) : a^3 - 25 * a = a * (a + 5) * (a - 5) := 
by sorry

end factorize_a_cubed_minus_25a_l23_23120


namespace find_f_of_2_l23_23478

theorem find_f_of_2 
  (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1/x) = x^2 + 1/x^2) : f 2 = 6 :=
sorry

end find_f_of_2_l23_23478


namespace max_vx_minus_yz_l23_23721

-- Define the set A
def A : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

-- Define the conditions
variables (v w x y z : ℤ)
#check v ∈ A -- v belongs to set A
#check w ∈ A -- w belongs to set A
#check x ∈ A -- x belongs to set A
#check y ∈ A -- y belongs to set A
#check z ∈ A -- z belongs to set A

-- vw = x
axiom vw_eq_x : v * w = x

-- w ≠ 0
axiom w_ne_zero : w ≠ 0

-- The target problem
theorem max_vx_minus_yz : ∃ v w x y z : ℤ, v ∈ A ∧ w ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ v * w = x ∧ w ≠ 0 ∧ (v * x - y * z) = 150 := by
  sorry

end max_vx_minus_yz_l23_23721


namespace race_winner_l23_23410

theorem race_winner
  (faster : String → String → Prop)
  (Minyoung Yoongi Jimin Yuna : String)
  (cond1 : faster Minyoung Yoongi)
  (cond2 : faster Yoongi Jimin)
  (cond3 : faster Yuna Jimin)
  (cond4 : faster Yuna Minyoung) :
  ∀ s, s ≠ Yuna → faster Yuna s :=
by
  sorry

end race_winner_l23_23410


namespace pieces_on_third_day_impossibility_of_2014_pieces_l23_23183

-- Define the process of dividing and eating chocolate pieces.
def chocolate_pieces (n : ℕ) : ℕ :=
  9 + 8 * n

-- The number of pieces after the third day.
theorem pieces_on_third_day : chocolate_pieces 3 = 25 :=
sorry

-- It's impossible for Maria to have exactly 2014 pieces on any given day.
theorem impossibility_of_2014_pieces : ∀ n : ℕ, chocolate_pieces n ≠ 2014 :=
sorry

end pieces_on_third_day_impossibility_of_2014_pieces_l23_23183


namespace distance_from_Martins_house_to_Lawrences_house_l23_23363

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end distance_from_Martins_house_to_Lawrences_house_l23_23363


namespace neg_of_exists_l23_23206

theorem neg_of_exists (P : ℝ → Prop) : 
  (¬ ∃ x: ℝ, x ≥ 3 ∧ x^2 - 2 * x + 3 < 0) ↔ (∀ x: ℝ, x ≥ 3 → x^2 - 2 * x + 3 ≥ 0) :=
by
  sorry

end neg_of_exists_l23_23206


namespace simplify_expression_l23_23007

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l23_23007


namespace minimum_value_of_expression_l23_23964

theorem minimum_value_of_expression {a c : ℝ} (h_pos : a > 0)
  (h_range : ∀ x, a * x ^ 2 - 4 * x + c ≥ 1) :
  ∃ a c, a > 0 ∧ (∀ x, a * x ^ 2 - 4 * x + c ≥ 1) ∧ (∃ a, a > 0 ∧ ∃ c, c - 1 = 4 / a ∧ (a / 4 + 9 / a = 3)) :=
by sorry

end minimum_value_of_expression_l23_23964


namespace jacket_cost_l23_23384

noncomputable def cost_of_shorts : ℝ := 13.99
noncomputable def cost_of_shirt : ℝ := 12.14
noncomputable def total_spent : ℝ := 33.56
noncomputable def cost_of_jacket : ℝ := total_spent - (cost_of_shorts + cost_of_shirt)

theorem jacket_cost : cost_of_jacket = 7.43 := by
  sorry

end jacket_cost_l23_23384


namespace mandarin_ducks_total_l23_23543

theorem mandarin_ducks_total : (3 * 2) = 6 := by
  sorry

end mandarin_ducks_total_l23_23543


namespace intersection_eq_l23_23302

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l23_23302


namespace picked_balls_correct_l23_23033

-- Conditions
def initial_balls := 6
def final_balls := 24

-- The task is to find the number of picked balls
def picked_balls : Nat := final_balls - initial_balls

-- The proof goal
theorem picked_balls_correct : picked_balls = 18 :=
by
  -- We declare, but the proof is not required
  sorry

end picked_balls_correct_l23_23033


namespace two_digit_numbers_of_form_3_pow_n_l23_23837

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l23_23837


namespace price_of_Microtron_stock_l23_23782

theorem price_of_Microtron_stock
  (n d : ℕ) (p_d p p_m : ℝ) 
  (h1 : n = 300) 
  (h2 : d = 150) 
  (h3 : p_d = 44) 
  (h4 : p = 40) 
  (h5 : p_m = 36) : 
  (d * p_d + (n - d) * p_m) / n = p := 
sorry

end price_of_Microtron_stock_l23_23782


namespace intersection_of_A_and_B_l23_23307

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l23_23307


namespace find_x_l23_23842

def hash_op (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) (h : hash_op x 6 = 48) : x = 3 :=
by
  sorry

end find_x_l23_23842


namespace angles_with_same_terminal_side_pi_div_3_l23_23029

noncomputable def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 2 * k * Real.pi

theorem angles_with_same_terminal_side_pi_div_3 :
  { α : ℝ | same_terminal_side α (Real.pi / 3) } =
  { α : ℝ | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3 } :=
by
  sorry

end angles_with_same_terminal_side_pi_div_3_l23_23029


namespace open_spots_level4_correct_l23_23744

noncomputable def open_spots_level_4 (total_levels : ℕ) (spots_per_level : ℕ) (open_spots_level1 : ℕ) (open_spots_level2 : ℕ) (open_spots_level3 : ℕ) (full_spots_total : ℕ) : ℕ := 
  let total_spots := total_levels * spots_per_level
  let open_spots_total := total_spots - full_spots_total 
  let open_spots_first_three := open_spots_level1 + open_spots_level2 + open_spots_level3
  open_spots_total - open_spots_first_three

theorem open_spots_level4_correct :
  open_spots_level_4 4 100 58 (58 + 2) (58 + 2 + 5) 186 = 31 :=
by
  sorry

end open_spots_level4_correct_l23_23744


namespace positive_difference_complementary_angles_l23_23400

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l23_23400


namespace factorization_correct_l23_23270

noncomputable def factorize_diff_of_squares (a b : ℝ) : ℝ :=
  36 * a * a - 4 * b * b

theorem factorization_correct (a b : ℝ) : factorize_diff_of_squares a b = 4 * (3 * a + b) * (3 * a - b) :=
by
  sorry

end factorization_correct_l23_23270


namespace exactly_one_absent_l23_23706

variables (B K Z : Prop)

theorem exactly_one_absent (h1 : B ∨ K) (h2 : K ∨ Z) (h3 : Z ∨ B)
    (h4 : ¬B ∨ ¬K ∨ ¬Z) : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
by
  sorry

end exactly_one_absent_l23_23706


namespace find_a_b_l23_23625

theorem find_a_b (a b : ℝ) (h : (⟨1, 2⟩ * (a : ℂ) + b = ⟨0, 2⟩)) : a = 1 ∧ b = -1 := 
by
  sorry

end find_a_b_l23_23625


namespace area_bounded_region_l23_23023

theorem area_bounded_region (x y : ℝ) (h : y^2 + 2*x*y + 30*|x| = 300) : 
  ∃ A, A = 900 := 
sorry

end area_bounded_region_l23_23023


namespace compute_expression_l23_23463

theorem compute_expression:
  let a := 3
  let b := 7
  (a + b) ^ 2 + Real.sqrt (a^2 + b^2) = 100 + Real.sqrt 58 :=
by
  sorry

end compute_expression_l23_23463


namespace min_soldiers_in_square_formations_l23_23409

theorem min_soldiers_in_square_formations : ∃ (a : ℕ), 
  ∃ (k : ℕ), 
    (a = k^2 ∧ 
    11 * a + 1 = (m : ℕ) ^ 2) ∧ 
    (∀ (b : ℕ), 
      (∃ (j : ℕ), b = j^2 ∧ 11 * b + 1 = (n : ℕ) ^ 2) → a ≤ b) ∧ 
    a = 9 := 
sorry

end min_soldiers_in_square_formations_l23_23409


namespace martha_total_clothes_l23_23373

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l23_23373


namespace simplify_cube_root_expression_l23_23006

-- statement of the problem
theorem simplify_cube_root_expression :
  ∛(8 + 27) * ∛(8 + ∛(27)) = ∛(385) :=
  sorry

end simplify_cube_root_expression_l23_23006


namespace solve_ab_eq_l23_23618

theorem solve_ab_eq:
  ∃ a b : ℝ, (1 + (2 : ℂ) * (Complex.I)) * (a : ℂ) + (b : ℂ) = (2 : ℂ) * (Complex.I) ∧ a = 1 ∧ b = -1 := by
  sorry

end solve_ab_eq_l23_23618


namespace real_solutions_to_system_l23_23122

theorem real_solutions_to_system (x y : ℝ) (h1 : x^3 + y^3 = 1) (h2 : x^4 + y^4 = 1) :
  (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end real_solutions_to_system_l23_23122


namespace tuples_and_triples_counts_are_equal_l23_23854

theorem tuples_and_triples_counts_are_equal (n : ℕ) (h : n > 0) :
  let countTuples := 8^n - 2 * 7^n + 6^n
  let countTriples := 8^n - 2 * 7^n + 6^n
  countTuples = countTriples :=
by
  sorry

end tuples_and_triples_counts_are_equal_l23_23854


namespace martha_clothes_total_l23_23370

-- Given conditions
def jackets_bought : Nat := 4
def t_shirts_bought : Nat := 9
def free_jacket_ratio : Nat := 2
def free_t_shirt_ratio : Nat := 3

-- Problem statement to prove
theorem martha_clothes_total :
  (jackets_bought + jackets_bought / free_jacket_ratio) + 
  (t_shirts_bought + t_shirts_bought / free_t_shirt_ratio) = 18 := 
by 
  sorry

end martha_clothes_total_l23_23370


namespace martha_clothes_total_l23_23369

-- Given conditions
def jackets_bought : Nat := 4
def t_shirts_bought : Nat := 9
def free_jacket_ratio : Nat := 2
def free_t_shirt_ratio : Nat := 3

-- Problem statement to prove
theorem martha_clothes_total :
  (jackets_bought + jackets_bought / free_jacket_ratio) + 
  (t_shirts_bought + t_shirts_bought / free_t_shirt_ratio) = 18 := 
by 
  sorry

end martha_clothes_total_l23_23369


namespace B_work_days_l23_23244

theorem B_work_days (A B C : ℕ) (hA : A = 15) (hC : C = 30) (H : (5 / 15) + ((10 * (1 / C + 1 / B)) / (1 / C + 1 / B)) = 1) : B = 30 := by
  sorry

end B_work_days_l23_23244


namespace gain_percentage_l23_23961

theorem gain_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 + C2 = 540) (h2 : C1 = 315)
    (h3 : SP1 = C1 - (0.15 * C1)) (h4 : SP1 = SP2) :
    ((SP2 - C2) / C2) * 100 = 19 :=
by
  sorry

end gain_percentage_l23_23961


namespace g_h_2_eq_2175_l23_23647

def g (x : ℝ) : ℝ := 2 * x^2 - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_2_eq_2175 : g (h 2) = 2175 := by
  sorry

end g_h_2_eq_2175_l23_23647


namespace solve_complex_eq_l23_23626

-- Defining the given condition equation with complex numbers and real variables
theorem solve_complex_eq (a b : ℝ) (h : (1 + 2 * complex.i) * a + b = 2 * complex.i) : 
  a = 1 ∧ b = -1 := 
sorry

end solve_complex_eq_l23_23626


namespace train_speed_correct_l23_23449

def length_of_train : ℕ := 700
def time_to_cross_pole : ℕ := 20
def expected_speed : ℕ := 35

theorem train_speed_correct : (length_of_train / time_to_cross_pole) = expected_speed := by
  sorry

end train_speed_correct_l23_23449


namespace scientists_from_usa_l23_23198

theorem scientists_from_usa (total_scientists : ℕ)
  (from_europe : ℕ)
  (from_canada : ℕ)
  (h1 : total_scientists = 70)
  (h2 : from_europe = total_scientists / 2)
  (h3 : from_canada = total_scientists / 5) :
  (total_scientists - from_europe - from_canada) = 21 :=
by
  sorry

end scientists_from_usa_l23_23198


namespace greatest_common_divisor_of_three_common_divisors_l23_23719

theorem greatest_common_divisor_of_three_common_divisors (m : ℕ) :
  (∀ d, d ∣ 126 ∧ d ∣ m → d = 1 ∨ d = 3 ∨ d = 9) →
  gcd 126 m = 9 := 
sorry

end greatest_common_divisor_of_three_common_divisors_l23_23719


namespace problem_number_eq_7_5_l23_23191

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l23_23191


namespace shop_owner_percentage_profit_l23_23053

theorem shop_owner_percentage_profit
  (cost_price_per_kg : ℝ)
  (selling_price_per_kg : ℝ)
  (buying_cheat_percentage : ℝ)
  (selling_cheat_percentage : ℝ)
  (h_buying_cheat : buying_cheat_percentage = 20)
  (h_selling_cheat : selling_cheat_percentage = 20)
  (h_cost_price_when_buying : cost_price_per_kg = 100 / 1.2)
  (h_selling_price_when_selling : selling_price_per_kg = 100 / 0.8) : 
  ((selling_price_per_kg - cost_price_per_kg) / cost_price_per_kg) * 100 = 50 := 
by 
  have cost_price_per_kg_value : cost_price_per_kg = 83.33 := by sorry
  have selling_price_per_kg_value : selling_price_per_kg = 125 := by sorry
  calc
    ((125 - 83.33) / 83.33) * 100 = 50 := by sorry

end shop_owner_percentage_profit_l23_23053


namespace distance_by_land_l23_23083

theorem distance_by_land (distance_by_sea total_distance distance_by_land : ℕ)
  (h1 : total_distance = 601)
  (h2 : distance_by_sea = 150)
  (h3 : total_distance = distance_by_land + distance_by_sea) : distance_by_land = 451 := by
  sorry

end distance_by_land_l23_23083


namespace martin_ring_fraction_l23_23681

theorem martin_ring_fraction (f : ℚ) :
  (36 + (36 * f + 4) = 52) → (f = 1 / 3) :=
by
  intro h
  -- Solution steps would go here
  sorry

end martin_ring_fraction_l23_23681


namespace impossible_to_divide_into_three_similar_piles_l23_23357

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l23_23357


namespace anna_score_below_90_no_A_l23_23859

def score_implies_grade (score : ℝ) : Prop :=
  score > 90 → true

theorem anna_score_below_90_no_A (score : ℝ) (A_grade : Prop) (h : score_implies_grade score) :
  score < 90 → ¬ A_grade :=
by sorry

end anna_score_below_90_no_A_l23_23859


namespace tan_alpha_minus_pi_div_4_l23_23493

open Real

theorem tan_alpha_minus_pi_div_4 (α : ℝ) (h : (cos α * 2 + (-1) * sin α = 0)) : 
  tan (α - π / 4) = 1 / 3 :=
sorry

end tan_alpha_minus_pi_div_4_l23_23493


namespace seashells_second_day_l23_23860

theorem seashells_second_day (x : ℕ) (h1 : 5 + x + 2 * (5 + x) = 36) : x = 7 :=
by
  sorry

end seashells_second_day_l23_23860


namespace tank_dimension_l23_23922

theorem tank_dimension (cost_per_sf : ℝ) (total_cost : ℝ) (length1 length3 : ℝ) (surface_area : ℝ) (dimension : ℝ) :
  cost_per_sf = 20 ∧ total_cost = 1520 ∧ 
  length1 = 4 ∧ length3 = 2 ∧ 
  surface_area = total_cost / cost_per_sf ∧
  12 * dimension + 16 = surface_area → dimension = 5 :=
by
  intro h
  obtain ⟨hcps, htac, hl1, hl3, hsa, heq⟩ := h
  sorry

end tank_dimension_l23_23922


namespace calculate_expression_l23_23903

theorem calculate_expression : (3072 - 2993) ^ 2 / 121 = 49 :=
by
  sorry

end calculate_expression_l23_23903


namespace sum_of_b_values_l23_23131

-- Definitions based on conditions from the problem statement
def quadratic_equation (b : ℕ) : Prop := ∃ x : ℚ, 3 * x^2 + 7 * x + b = 0

def has_rational_roots (b : ℕ) : Prop :=
  ∃ (m : ℤ), ∃ (k : ℤ), m^2 = 49 - 12 * b

def possible_b_values (b : ℕ) : Prop := b > 0 ∧ has_rational_roots b

-- The statement of the proof problem
theorem sum_of_b_values :
  (∑ b in { b | possible_b_values b }.to_finset, b) = 6 :=
sorry

end sum_of_b_values_l23_23131


namespace impossible_to_divide_into_three_similar_parts_l23_23348

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l23_23348


namespace daniel_fraction_l23_23454

theorem daniel_fraction (A B C D : Type) (money : A → ℝ) 
  (adriano bruno cesar daniel : A)
  (h1 : money daniel = 0)
  (given_amount : ℝ)
  (h2 : money adriano = 5 * given_amount)
  (h3 : money bruno = 4 * given_amount)
  (h4 : money cesar = 3 * given_amount)
  (h5 : money daniel = (1 / 5) * money adriano + (1 / 4) * money bruno + (1 / 3) * money cesar) :
  money daniel / (money adriano + money bruno + money cesar) = 1 / 4 := 
by
  sorry

end daniel_fraction_l23_23454


namespace payment_is_variable_l23_23207

variable (x y : ℕ)

def price_of_pen : ℕ := 3

theorem payment_is_variable (x y : ℕ) (h : y = price_of_pen * x) : 
  (price_of_pen = 3) ∧ (∃ n : ℕ, y = 3 * n) :=
by 
  sorry

end payment_is_variable_l23_23207


namespace intersect_graphs_exactly_four_l23_23680

theorem intersect_graphs_exactly_four (A : ℝ) (hA : 0 < A) :
  (∃ x y : ℝ, y = A * x^2 ∧ x^2 + 2 * y^2 = A + 3) ↔ (∀ x1 y1 x2 y2 : ℝ, (y1 = A * x1^2 ∧ x1^2 + 2 * y1^2 = A + 3) ∧ (y2 = A * x2^2 ∧ x2^2 + 2 * y2^2 = A + 3) → (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end intersect_graphs_exactly_four_l23_23680


namespace intersection_two_sets_l23_23320

theorem intersection_two_sets (M N : Set ℤ) (h1 : M = {1, 2, 3, 4}) (h2 : N = {-2, 2}) :
  M ∩ N = {2} := 
by
  sorry

end intersection_two_sets_l23_23320


namespace least_common_addition_of_primes_l23_23698

theorem least_common_addition_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < y) (h : 4 * x + y = 87) : x + y = 81 := 
sorry

end least_common_addition_of_primes_l23_23698


namespace cubic_polynomial_coefficients_l23_23518

theorem cubic_polynomial_coefficients (f g : Polynomial ℂ) (b c d : ℂ) :
  f = Polynomial.C 4 + Polynomial.X * (Polynomial.C 3 + Polynomial.X * (Polynomial.C 2 + Polynomial.X)) →
  (∀ x, Polynomial.eval x f = 0 → Polynomial.eval (x^2) g = 0) →
  g = Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)) →
  (b, c, d) = (4, -15, -32) :=
by
  intro h1 h2 h3
  sorry

end cubic_polynomial_coefficients_l23_23518


namespace minimum_ab_l23_23962

variable (a b : ℝ)

def is_collinear (a b : ℝ) : Prop :=
  (0 - b) * (-2 - 0) = (-2 - b) * (a - 0)

theorem minimum_ab (h1 : a * b > 0) (h2 : is_collinear a b) : a * b = 16 := by
  sorry

end minimum_ab_l23_23962


namespace percentage_decrease_l23_23208

variable (current_price original_price : ℝ)

theorem percentage_decrease (h1 : current_price = 760) (h2 : original_price = 1000) :
  (original_price - current_price) / original_price * 100 = 24 :=
by
  sorry

end percentage_decrease_l23_23208


namespace find_line_l_l23_23972

def line_equation (x y: ℤ) : Prop := x - 2 * y = 2

def scaling_transform_x (x: ℤ) : ℤ := x
def scaling_transform_y (y: ℤ) : ℤ := 2 * y

theorem find_line_l :
  ∀ (x y x' y': ℤ),
  x' = scaling_transform_x x →
  y' = scaling_transform_y y →
  line_equation x y →
  x' - y' = 2 := by
  sorry

end find_line_l_l23_23972


namespace infinite_chains_of_tangent_circles_exist_l23_23380

theorem infinite_chains_of_tangent_circles_exist
  (R₁ R₂ : Circle) 
  (h_disjoint : ¬ (R₁ ⋂ R₂).nonempty)
  (T₁ : Circle)
  (h_tangent_R₁ : T₁.TangentTo R₁)
  (h_tangent_R₂ : T₁.TangentTo R₂)
  : ∃ (T : ℕ → Circle), ∀ n, 
    (T n).TangentTo R₁ ∧ 
    (T n).TangentTo R₂ ∧ 
    (∀ m, T m.TangentTo (T (m + 1)) ∧ T m.TangentTo (T (m - 1))) :=
sorry

end infinite_chains_of_tangent_circles_exist_l23_23380


namespace distinct_collections_of_letters_in_bag_l23_23713

theorem distinct_collections_of_letters_in_bag : 
  let word := "STATISTICS".to_list
  let vowels := {'A', 'I'}
  let consonants := {'S', 'T', 'C'}
  (count_repeats word vowels 3) ∧ (count_repeats word consonants 4) → count_possible_collections word 30 := 
sorry

end distinct_collections_of_letters_in_bag_l23_23713


namespace tan_ineq_solution_l23_23653

theorem tan_ineq_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x, x = a * Real.pi → ¬ (Real.tan x = a * Real.pi)) :
    {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2}
    = {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2} := sorry

end tan_ineq_solution_l23_23653


namespace num_two_digit_powers_of_3_l23_23828

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l23_23828


namespace powers_of_three_two_digit_count_l23_23827

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l23_23827


namespace probability_y_gt_x_l23_23789

-- Define the uniform distribution and the problem setup
def uniform_distribution (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the variables
variables (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000)

-- Define the probability calculation function (assuming some proper definition for probability)
noncomputable def probability_event (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the event that Laurent's number is greater than Chloe's number
def event_y_gt_x : Set (ℝ × ℝ) := {p | p.2 > p.1}

-- State the theorem
theorem probability_y_gt_x (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000) :
  probability_event event_y_gt_x = 3/4 :=
sorry

end probability_y_gt_x_l23_23789


namespace hypotenuse_not_5_cm_l23_23031

theorem hypotenuse_not_5_cm (a b c : ℝ) (h₀ : a + b = 8) (h₁ : a^2 + b^2 = c^2) : c ≠ 5 := by
  sorry

end hypotenuse_not_5_cm_l23_23031


namespace find_a_b_l23_23624

theorem find_a_b (a b : ℝ) (h : (⟨1, 2⟩ * (a : ℂ) + b = ⟨0, 2⟩)) : a = 1 ∧ b = -1 := 
by
  sorry

end find_a_b_l23_23624


namespace fraction_of_credit_extended_l23_23257

noncomputable def C_total : ℝ := 342.857
noncomputable def P_auto : ℝ := 0.35
noncomputable def C_company : ℝ := 40

theorem fraction_of_credit_extended :
  (C_company / (C_total * P_auto)) = (1 / 3) :=
  by
    sorry

end fraction_of_credit_extended_l23_23257


namespace solve_for_y_l23_23612

theorem solve_for_y (y : ℝ) (h : y + 81 / (y - 3) = -12) : y = -6 ∨ y = -3 :=
sorry

end solve_for_y_l23_23612


namespace martha_total_clothes_l23_23376

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l23_23376


namespace tailor_trimming_l23_23749

theorem tailor_trimming (x : ℝ) (A B : ℝ)
  (h1 : ∃ (L : ℝ), L = 22) -- Original length of a side of the cloth is 22 feet
  (h2 : 6 = 6) -- Feet trimmed from two opposite edges
  (h3 : ∃ (remaining_area : ℝ), remaining_area = 120) -- 120 square feet of cloth remain after trimming
  (h4 : A = 22 - 2 * 6) -- New length of the side after trimming 6 feet from opposite edges
  (h5 : B = 22 - x) -- New length of the side after trimming x feet from the other two edges
  (h6 : remaining_area = A * B) -- Relationship of the remaining area
: x = 10 :=
by
  sorry

end tailor_trimming_l23_23749


namespace tangent_circle_line_l23_23652

theorem tangent_circle_line (a : ℝ) :
  (∀ x y : ℝ, (x - y + 3 = 0) → (x^2 + y^2 - 2 * x + 2 - a = 0)) →
  a = 9 :=
by
  sorry

end tangent_circle_line_l23_23652


namespace volleyball_team_selection_l23_23686

theorem volleyball_team_selection (total_players starting_players : ℕ) (libero : ℕ) : 
  total_players = 12 → 
  starting_players = 6 → 
  libero = 1 →
  (∃ (ways : ℕ), ways = 5544) :=
by
  intros h1 h2 h3
  sorry

end volleyball_team_selection_l23_23686


namespace average_speed_is_one_l23_23060

-- Definition of distance and time
def distance : ℕ := 1800
def time_in_minutes : ℕ := 30
def time_in_seconds : ℕ := time_in_minutes * 60

-- Definition of average speed as distance divided by time
def average_speed (distance : ℕ) (time : ℕ) : ℚ :=
  distance / time

-- Theorem: Given the distance and time, the average speed is 1 meter per second
theorem average_speed_is_one : average_speed distance time_in_seconds = 1 :=
  by
    sorry

end average_speed_is_one_l23_23060


namespace intersection_A_B_l23_23301

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l23_23301


namespace sufficientButNotNecessary_l23_23875

theorem sufficientButNotNecessary (x : ℝ) : ((x + 1) * (x - 3) < 0) → x < 3 ∧ ¬(x < 3 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficientButNotNecessary_l23_23875


namespace max_value_l23_23675

open Real

theorem max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 6 * y < 108) :
  (x^2 * y * (108 - 3 * x - 6 * y)) ≤ 7776 :=
sorry

end max_value_l23_23675


namespace exists_positive_integer_divisible_by_15_and_sqrt_in_range_l23_23795

theorem exists_positive_integer_divisible_by_15_and_sqrt_in_range :
  ∃ (n : ℕ), (n % 15 = 0) ∧ (28 < Real.sqrt n) ∧ (Real.sqrt n < 28.5) ∧ (n = 795) :=
by
  sorry

end exists_positive_integer_divisible_by_15_and_sqrt_in_range_l23_23795


namespace colors_of_clothes_l23_23763

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l23_23763


namespace constant_term_expansion_l23_23554

theorem constant_term_expansion : 
  (constant_term_of_expansion (5 * x + 2 / (5 * x)) 8) = 1120 := 
by sorry

end constant_term_expansion_l23_23554


namespace annie_passes_bonnie_first_l23_23087

def bonnie_speed (v : ℝ) := v
def annie_speed (v : ℝ) := 1.3 * v
def track_length := 500

theorem annie_passes_bonnie_first (v t : ℝ) (ht : 0.3 * v * t = track_length) : 
  (annie_speed v * t) / track_length = 4 + 1 / 3 :=
by 
  sorry

end annie_passes_bonnie_first_l23_23087


namespace max_value_of_expression_l23_23383

theorem max_value_of_expression :
  ∃ x : ℝ, ∀ y : ℝ, -x^2 + 4*x + 10 ≤ -y^2 + 4*y + 10 ∧ -x^2 + 4*x + 10 = 14 :=
sorry

end max_value_of_expression_l23_23383


namespace total_eggs_l23_23874

theorem total_eggs (eggs_today eggs_yesterday : ℕ) (h_today : eggs_today = 30) (h_yesterday : eggs_yesterday = 19) : eggs_today + eggs_yesterday = 49 :=
by
  sorry

end total_eggs_l23_23874


namespace five_x_ge_seven_y_iff_exists_abcd_l23_23381

theorem five_x_ge_seven_y_iff_exists_abcd (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔ ∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d :=
by sorry

end five_x_ge_seven_y_iff_exists_abcd_l23_23381


namespace symmetric_line_eq_x_axis_l23_23391

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * (-y) + 5 = 0) :=
by
  sorry

end symmetric_line_eq_x_axis_l23_23391


namespace mean_of_five_numbers_l23_23219

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l23_23219


namespace linear_function_quadrants_l23_23157

theorem linear_function_quadrants (m : ℝ) :
  (∀ (x : ℝ), y = -3 * x + m →
  (x < 0 ∧ y > 0 ∨ x > 0 ∧ y < 0 ∨ x < 0 ∧ y < 0)) → m < 0 :=
sorry

end linear_function_quadrants_l23_23157


namespace margie_change_l23_23989

theorem margie_change :
  let num_apples := 5
  let cost_per_apple := 0.30
  let discount := 0.10
  let amount_paid := 10.00
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount)
  let change_received := amount_paid - discounted_cost
  change_received = 8.65 := sorry

end margie_change_l23_23989


namespace running_time_l23_23851

variable (t : ℝ)
variable (v_j v_p d : ℝ)

-- Given conditions
variable (v_j : ℝ := 0.133333333333)  -- Joe's speed
variable (v_p : ℝ := 0.0666666666665) -- Pete's speed
variable (d : ℝ := 16)                -- Distance between them after t minutes

theorem running_time (h : v_j + v_p = 0.2 * t) : t = 80 :=
by
  -- Distance covered by Joe and Pete running in opposite directions
  have h1 : v_j * t + v_p * t = d := by sorry
  -- Given combined speeds
  have h2 : v_j + v_p = 0.2 := by sorry
  -- Using the equation to solve for time t
  exact sorry

end running_time_l23_23851


namespace no_division_into_three_similar_piles_l23_23344

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l23_23344


namespace lucy_bought_cakes_l23_23988

theorem lucy_bought_cakes (cookies chocolate total c : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) (h4 : c = total - (cookies + chocolate)) : c = 22 := by
  sorry

end lucy_bought_cakes_l23_23988


namespace garden_area_l23_23460

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end garden_area_l23_23460


namespace height_at_end_of_2_years_l23_23450

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end height_at_end_of_2_years_l23_23450


namespace simplify_expression_l23_23934

theorem simplify_expression (x y z : ℝ) : (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := 
by 
sorry

end simplify_expression_l23_23934


namespace largest_sum_ABC_l23_23164

theorem largest_sum_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 3003) : 
  A + B + C ≤ 105 :=
sorry

end largest_sum_ABC_l23_23164


namespace evaluate_expression_l23_23117

theorem evaluate_expression :
  3 + 2*Real.sqrt 3 + 1/(3 + 2*Real.sqrt 3) + 1/(2*Real.sqrt 3 - 3) = 3 + (16 * Real.sqrt 3) / 3 :=
by
  sorry

end evaluate_expression_l23_23117


namespace intersection_A_B_l23_23300

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l23_23300


namespace find_number_l23_23192

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l23_23192


namespace impossible_divide_into_three_similar_l23_23350

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l23_23350


namespace rate_of_current_l23_23921

theorem rate_of_current : 
  ∀ (v c : ℝ), v = 3.3 → (∀ d: ℝ, d > 0 → (d / (v - c) = 2 * (d / (v + c))) → c = 1.1) :=
by
  intros v c hv h
  sorry

end rate_of_current_l23_23921


namespace friends_in_group_l23_23443

theorem friends_in_group (n : ℕ) 
  (avg_before_increase : ℝ := 800) 
  (avg_after_increase : ℝ := 850) 
  (individual_rent_increase : ℝ := 800 * 0.25) 
  (original_rent : ℝ := 800) 
  (new_rent : ℝ := 1000)
  (original_total : ℝ := avg_before_increase * n) 
  (new_total : ℝ := original_total + individual_rent_increase):
  new_total = avg_after_increase * n → 
  n = 4 :=
by
  sorry

end friends_in_group_l23_23443


namespace fixed_point_of_invariant_line_l23_23994

theorem fixed_point_of_invariant_line :
  ∀ (m : ℝ) (x y : ℝ), (3 * m + 4) * x + (5 - 2 * m) * y + 7 * m - 6 = 0 →
  (x = -1 ∧ y = 2) :=
by
  intro m x y h
  sorry

end fixed_point_of_invariant_line_l23_23994


namespace square_area_l23_23079

theorem square_area (p : ℝ → ℝ) (a b : ℝ) (h₁ : ∀ x, p x = x^2 + 3 * x + 2) (h₂ : p a = 5) (h₃ : p b = 5) (h₄ : a ≠ b) : (b - a)^2 = 21 :=
by
  sorry

end square_area_l23_23079


namespace dividend_is_10_l23_23966

theorem dividend_is_10
  (q d r : ℕ)
  (hq : q = 3)
  (hd : d = 3)
  (hr : d = 3 * r) :
  (q * d + r = 10) :=
by
  sorry

end dividend_is_10_l23_23966


namespace reciprocal_of_repeating_decimal_l23_23564

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l23_23564


namespace max_value_of_f_l23_23942

def f (x : ℝ) : ℝ := 10 * x - 2 * x ^ 2

theorem max_value_of_f : ∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) :=
  ⟨12.5, sorry⟩

end max_value_of_f_l23_23942


namespace range_of_m_l23_23340

open Set

variable {α : Type*}

theorem range_of_m (A : Set ℝ) (n m : ℝ) (x : ℝ) : 
  (B ⊆ A)  → (B = { x | -m < x ∧ x < 2 }) → 
  (f x = n * (x + 1)) → 
  m ≤ (1 / 2) := sorry

end range_of_m_l23_23340


namespace odd_cube_difference_divisible_by_power_of_two_l23_23179

theorem odd_cube_difference_divisible_by_power_of_two {a b n : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) :
  (2^n ∣ (a^3 - b^3)) ↔ (2^n ∣ (a - b)) :=
by
  sorry

end odd_cube_difference_divisible_by_power_of_two_l23_23179


namespace intersection_sets_l23_23290

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l23_23290


namespace amelia_wins_probability_l23_23780

def amelia_prob_heads : ℚ := 1 / 4
def blaine_prob_heads : ℚ := 3 / 7

def probability_blaine_wins_first_turn : ℚ := blaine_prob_heads

def probability_amelia_wins_first_turn : ℚ :=
  (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_second_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_third_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * 
  (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins : ℚ :=
  probability_amelia_wins_first_turn + probability_amelia_wins_second_turn + probability_amelia_wins_third_turn

theorem amelia_wins_probability : probability_amelia_wins = 223 / 784 := by
  sorry

end amelia_wins_probability_l23_23780


namespace surface_area_three_dimensional_shape_l23_23550

-- Define the edge length of the largest cube
def edge_length_large : ℕ := 5

-- Define the condition for dividing the edge of the attachment face of the large cube into five equal parts
def divided_into_parts (edge_length : ℕ) (parts : ℕ) : Prop :=
  parts = 5

-- Define the condition that the edge lengths of all three blocks are different
def edge_lengths_different (e1 e2 e3 : ℕ) : Prop :=
  e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3

-- Define the surface area formula for a cube
def surface_area (s : ℕ) : ℕ :=
  6 * s^2

-- State the problem as a theorem
theorem surface_area_three_dimensional_shape (e1 e2 e3 : ℕ) (h1 : e1 = edge_length_large)
    (h2 : divided_into_parts e1 5) (h3 : edge_lengths_different e1 e2 e3) : 
    surface_area e1 + (surface_area e2 + surface_area e3 - 4 * (e2 * e3)) = 270 :=
sorry

end surface_area_three_dimensional_shape_l23_23550


namespace impossible_to_divide_into_three_similar_parts_l23_23349

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l23_23349


namespace problem_statement_l23_23314

variables (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def condition (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (2 - x)

theorem problem_statement (h_odd : is_odd f) (h_cond : condition f) : f 2010 = 0 := 
sorry

end problem_statement_l23_23314


namespace find_range_m_l23_23289

variables (m : ℝ)

def p (m : ℝ) : Prop :=
  (∀ x y : ℝ, (x^2 / (2 * m)) - (y^2 / (m - 1)) = 1) → false

def q (m : ℝ) : Prop :=
  (∀ e : ℝ, (1 < e ∧ e < 2) → (∀ x y : ℝ, (y^2 / 5) - (x^2 / m) = 1)) → false

noncomputable def range_m (m : ℝ) : Prop :=
  p m = false ∧ q m = false ∧ (p m ∨ q m) = true → (1/3 ≤ m ∧ m < 15)

theorem find_range_m : ∀ m : ℝ, range_m m :=
by
  intro m
  simp [range_m, p, q]
  sorry

end find_range_m_l23_23289


namespace solve_x_squared_eq_sixteen_l23_23708

theorem solve_x_squared_eq_sixteen : ∃ (x1 x2 : ℝ), (x1 = -4 ∧ x2 = 4) ∧ ∀ x : ℝ, x^2 = 16 → (x = x1 ∨ x = x2) :=
by
  sorry

end solve_x_squared_eq_sixteen_l23_23708


namespace Gina_makes_30_per_hour_l23_23281

variable (rose_cups_per_hour lily_cups_per_hour : ℕ)
variable (rose_cup_order lily_cup_order total_payment : ℕ)
variable (total_hours : ℕ)

def Gina_hourly_rate (rose_cups_per_hour: ℕ) (lily_cups_per_hour: ℕ) (rose_cup_order: ℕ) (lily_cup_order: ℕ) (total_payment: ℕ) : Prop :=
    let rose_time := rose_cup_order / rose_cups_per_hour
    let lily_time := lily_cup_order / lily_cups_per_hour
    let total_time := rose_time + lily_time
    total_payment / total_time = total_hours

theorem Gina_makes_30_per_hour :
    let rose_cups_per_hour := 6
    let lily_cups_per_hour := 7
    let rose_cup_order := 6
    let lily_cup_order := 14
    let total_payment := 90
    Gina_hourly_rate rose_cups_per_hour lily_cups_per_hour rose_cup_order lily_cup_order total_payment 30 :=
by
    sorry

end Gina_makes_30_per_hour_l23_23281


namespace rational_operation_example_l23_23141

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end rational_operation_example_l23_23141


namespace trees_died_proof_l23_23816

def treesDied (original : Nat) (remaining : Nat) : Nat := original - remaining

theorem trees_died_proof : treesDied 20 4 = 16 := by
  -- Here we put the steps needed to prove the theorem, which is essentially 20 - 4 = 16.
  sorry

end trees_died_proof_l23_23816


namespace simplest_radical_expression_l23_23108

theorem simplest_radical_expression :
  let A := Real.sqrt 3
  let B := Real.sqrt 4
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 2)
  B = 2 :=
by
  sorry

end simplest_radical_expression_l23_23108


namespace find_percentage_of_male_students_l23_23660

def percentage_of_male_students (M F : ℝ) : Prop :=
  M + F = 1 ∧ 0.40 * M + 0.60 * F = 0.52

theorem find_percentage_of_male_students (M F : ℝ) (h1 : M + F = 1) (h2 : 0.40 * M + 0.60 * F = 0.52) : M = 0.40 :=
by
  sorry

end find_percentage_of_male_students_l23_23660


namespace height_at_2_years_l23_23453

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end height_at_2_years_l23_23453


namespace geo_seq_ratio_l23_23361

theorem geo_seq_ratio (S : ℕ → ℝ) (r : ℝ) (hS : ∀ n, S n = (1 - r^(n+1)) / (1 - r))
  (hS_ratio : S 10 / S 5 = 1 / 2) : S 15 / S 5 = 3 / 4 := 
by
  sorry

end geo_seq_ratio_l23_23361


namespace line_through_origin_tangent_lines_line_through_tangents_l23_23949

section GeomProblem

variables {A : ℝ × ℝ} {C : ℝ × ℝ → Prop}

def is_circle (C : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
∀ (P : ℝ × ℝ), C P ↔ (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

theorem line_through_origin (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ m : ℝ, ∀ P : ℝ × ℝ, C P → abs ((m * P.1 - P.2) / Real.sqrt (m ^ 2 + 1)) = 1)
    ↔ m = 0 :=
sorry

theorem tangent_lines (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P : ℝ × ℝ, C P → (P.2 - 2 * Real.sqrt 3) = k * (P.1 - 1))
    ↔ (∀ P : ℝ × ℝ, C P → (Real.sqrt 3 * P.1 - 3 * P.2 + 5 * Real.sqrt 3 = 0 ∨ P.1 = 1)) :=
sorry

theorem line_through_tangents (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P D E : ℝ × ℝ, C P → (Real.sqrt 3 * D.1 - 3 * D.2 + 5 * Real.sqrt 3 = 0 ∧
                                      (E.1 - 1 = 0 ∨ Real.sqrt 3 * E.1 - 3 * E.2 + 5 * Real.sqrt 3 = 0)) →
    (D.1 + Real.sqrt 3 * D.2 - 1 = 0 ∧ E.1 + Real.sqrt 3 * E.2 - 1 = 0)) :=
sorry

end GeomProblem

end line_through_origin_tangent_lines_line_through_tangents_l23_23949


namespace shorter_piece_length_l23_23434

def wireLength := 150
def ratioLongerToShorter := 5 / 8

theorem shorter_piece_length : ∃ x : ℤ, x + (5 / 8) * x = wireLength ∧ x = 92 := by
  sorry

end shorter_piece_length_l23_23434


namespace total_strawberries_l23_23996

-- Define the number of original strawberries and the number of picked strawberries
def original_strawberries : ℕ := 42
def picked_strawberries : ℕ := 78

-- Prove the total number of strawberries
theorem total_strawberries : original_strawberries + picked_strawberries = 120 := by
  -- Proof goes here
  sorry

end total_strawberries_l23_23996


namespace mean_of_five_numbers_l23_23217

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l23_23217


namespace last_digit_of_3_to_2010_is_9_l23_23685

theorem last_digit_of_3_to_2010_is_9 : (3^2010 % 10) = 9 := by
  -- Given that the last digits of powers of 3 cycle through 3, 9, 7, 1
  -- We need to prove that the last digit of 3^2010 is 9
  sorry

end last_digit_of_3_to_2010_is_9_l23_23685


namespace find_x_l23_23252

theorem find_x (x : ℕ) (h : x * 5^4 = 75625) : x = 121 :=
by
  sorry

end find_x_l23_23252


namespace car_travel_distance_l23_23915

-- Definitions based on the conditions
def car_speed : ℕ := 60  -- The actual speed of the car
def faster_speed : ℕ := car_speed + 30  -- Speed if the car traveled 30 km/h faster
def time_difference : ℚ := 0.5  -- 30 minutes less in hours

-- The distance D we need to prove
def distance_traveled : ℚ := 90

-- Main statement to be proven
theorem car_travel_distance : ∀ (D : ℚ),
  (D / car_speed) = (D / faster_speed) + time_difference →
  D = distance_traveled :=
by
  intros D h
  sorry

end car_travel_distance_l23_23915


namespace find_a5_l23_23615

noncomputable def arithmetic_sequence (n : ℕ) (a d : ℤ) : ℤ :=
a + n * d

theorem find_a5 (a d : ℤ) (a_2_a_4_sum : arithmetic_sequence 1 a d + arithmetic_sequence 3 a d = 16)
  (a1 : arithmetic_sequence 0 a d = 1) :
  arithmetic_sequence 4 a d = 15 :=
by
  sorry

end find_a5_l23_23615


namespace part_time_job_pay_per_month_l23_23090

def tuition_fee : ℝ := 90
def scholarship_percent : ℝ := 0.30
def scholarship_amount := scholarship_percent * tuition_fee
def amount_after_scholarship := tuition_fee - scholarship_amount
def remaining_amount : ℝ := 18
def months_to_pay : ℝ := 3
def amount_paid_so_far := amount_after_scholarship - remaining_amount

theorem part_time_job_pay_per_month : amount_paid_so_far / months_to_pay = 15 := by
  sorry

end part_time_job_pay_per_month_l23_23090


namespace complementary_angles_positive_difference_l23_23397

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l23_23397


namespace abs_neg_ten_l23_23929

theorem abs_neg_ten : abs (-10) = 10 := 
by {
  sorry
}

end abs_neg_ten_l23_23929


namespace reciprocal_of_repeating_decimal_l23_23562

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l23_23562


namespace complete_the_square_l23_23697

theorem complete_the_square (y : ℝ) : (y^2 + 12*y + 40) = (y + 6)^2 + 4 := by
  sorry

end complete_the_square_l23_23697


namespace exponential_function_condition_l23_23879

theorem exponential_function_condition (a : ℝ) (x : ℝ) 
  (h1 : a^2 - 5 * a + 5 = 1) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) : 
  a = 4 := 
sorry

end exponential_function_condition_l23_23879


namespace total_weight_full_bucket_l23_23049

theorem total_weight_full_bucket (x y c d : ℝ) 
(h1 : x + 3/4 * y = c) 
(h2 : x + 1/3 * y = d) :
x + y = (8 * c - 3 * d) / 5 :=
sorry

end total_weight_full_bucket_l23_23049


namespace f_divisible_by_27_l23_23525

theorem f_divisible_by_27 (n : ℕ) : 27 ∣ (2^(2*n - 1) - 9 * n^2 + 21 * n - 14) :=
sorry

end f_divisible_by_27_l23_23525


namespace fill_question_mark_l23_23737

def sudoku_grid : Type := 
  List (List (Option ℕ))

def initial_grid : sudoku_grid := 
  [ [some 3, none, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ]

def valid_sudoku (grid : sudoku_grid) : Prop :=
  -- Ensure the grid is a valid 4x4 Sudoku grid
  -- Adding necessary constraints for rows, columns and 2x2 subgrids.
  sorry

def solve_sudoku (grid : sudoku_grid) : sudoku_grid :=
  -- Function that solves the Sudoku (not implemented for this proof statement)
  sorry

theorem fill_question_mark : solve_sudoku initial_grid = 
  [ [some 3, some 2, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ] :=
  sorry

end fill_question_mark_l23_23737


namespace last_score_is_87_l23_23993

-- Definitions based on conditions:
def scores : List ℕ := [73, 78, 82, 84, 87, 95]
def total_sum := 499
def final_median := 83

-- Prove that the last score is 87 under given conditions.
theorem last_score_is_87 (h1 : total_sum = 499)
                        (h2 : ∀ n ∈ scores, (499 - n) % 6 ≠ 0)
                        (h3 : final_median = 83) :
  87 ∈ scores := sorry

end last_score_is_87_l23_23993


namespace find_a_l23_23955

noncomputable def f (x : ℝ) : ℝ := x^2 + 10

noncomputable def g (x : ℝ) : ℝ := x^2 - 6

theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 12) :
    a = Real.sqrt (6 + Real.sqrt 2) ∨ a = Real.sqrt (6 - Real.sqrt 2) :=
sorry

end find_a_l23_23955


namespace find_t_l23_23693

variable {a b c r s t : ℝ}

-- Conditions from part a)
def first_polynomial_has_roots (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c)) : Prop :=
  ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = 0 → x = a ∨ x = b ∨ x = c

def second_polynomial_has_roots (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) : Prop :=
  ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = 0 → x = (a + b) ∨ x = (b + c) ∨ x = (c + a)

-- Translate problem (find t) with conditions
theorem find_t (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c))
    (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a)))
    (sum_roots : a + b + c = -3) 
    (prod_roots : a * b * c = -11):
  t = 23 := 
sorry

end find_t_l23_23693


namespace total_feet_in_garden_l23_23889

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end total_feet_in_garden_l23_23889


namespace sum_slopes_const_zero_l23_23878

-- Define variables and constants
variable (p : ℝ) (h : 0 < p)

-- Define parabola and circle equations
def parabola_C1 (x y : ℝ) : Prop := y^2 = 2 * p * x
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 = p^2

-- Condition: The line segment length from circle cut by directrix
def segment_length_condition : Prop := ∃ d : ℝ, d^2 + 3 = p^2

-- The main theorem to prove
theorem sum_slopes_const_zero
  (A : ℝ × ℝ)
  (F : ℝ × ℝ := (p / 2, 0))
  (M N : ℝ × ℝ)
  (line_n_through_A : ∀ x : ℝ, x = 1 / p - 1 + 1 / p → (1 / p - 1 + x) = 0)
  (intersection_prop: parabola_C1 p M.1 M.2 ∧ parabola_C1 p N.1 N.2) 
  (slope_MF : ℝ := (M.2 / (p / 2 - M.1)) ) 
  (slope_NF : ℝ := (N.2 / (p / 2 - N.1))) :
  slope_MF + slope_NF = 0 := 
sorry

end sum_slopes_const_zero_l23_23878


namespace find_num_carbon_atoms_l23_23584

def num_carbon_atoms (nH nO mH mC mO mol_weight : ℕ) : ℕ :=
  (mol_weight - (nH * mH + nO * mO)) / mC

theorem find_num_carbon_atoms :
  num_carbon_atoms 2 3 1 12 16 62 = 1 :=
by
  -- The proof is skipped
  sorry

end find_num_carbon_atoms_l23_23584


namespace num_two_digit_powers_of_3_l23_23821

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l23_23821


namespace geom_seq_11th_term_l23_23392

/-!
The fifth and eighth terms of a geometric sequence are -2 and -54, respectively. 
What is the 11th term of this progression?
-/
theorem geom_seq_11th_term {a : ℕ → ℤ} (r : ℤ) 
  (h1 : a 5 = -2) (h2 : a 8 = -54) 
  (h3 : ∀ n : ℕ, a (n + 3) = a n * r ^ 3) : 
  a 11 = -1458 :=
sorry

end geom_seq_11th_term_l23_23392


namespace pow_zero_eq_one_l23_23457

theorem pow_zero_eq_one : (-2023)^0 = 1 :=
by
  -- The proof of this theorem will go here.
  sorry

end pow_zero_eq_one_l23_23457


namespace max_tickets_jane_can_buy_l23_23793

def ticket_price : ℝ := 15.75
def processing_fee : ℝ := 1.25
def jane_money : ℝ := 150.00

theorem max_tickets_jane_can_buy : ⌊jane_money / (ticket_price + processing_fee)⌋ = 8 := 
by
  sorry

end max_tickets_jane_can_buy_l23_23793


namespace percentage_of_16_l23_23246

theorem percentage_of_16 (p : ℝ) (h : (p / 100) * 16 = 0.04) : p = 0.25 :=
by
  sorry

end percentage_of_16_l23_23246


namespace investment_time_period_l23_23063

variable (P : ℝ) (r15 r12 : ℝ) (T : ℝ)
variable (hP : P = 15000)
variable (hr15 : r15 = 0.15)
variable (hr12 : r12 = 0.12)
variable (diff : 2250 * T - 1800 * T = 900)

theorem investment_time_period :
  T = 2 := by
  sorry

end investment_time_period_l23_23063


namespace intersection_A_B_l23_23298

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l23_23298


namespace alpha_value_l23_23809

theorem alpha_value (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.logb 3 (x + 1)) (h2 : f α = 1) : α = 2 := by
  sorry

end alpha_value_l23_23809


namespace students_speaking_Gujarati_l23_23159

theorem students_speaking_Gujarati 
  (total_students : ℕ)
  (students_Hindi : ℕ)
  (students_Marathi : ℕ)
  (students_two_languages : ℕ)
  (students_all_three_languages : ℕ)
  (students_total_set: 22 = total_students)
  (students_H_set: 15 = students_Hindi)
  (students_M_set: 6 = students_Marathi)
  (students_two_set: 2 = students_two_languages)
  (students_all_three_set: 1 = students_all_three_languages) :
  ∃ (students_Gujarati : ℕ), 
  22 = students_Gujarati + 15 + 6 - 2 + 1 ∧ students_Gujarati = 2 :=
by
  sorry

end students_speaking_Gujarati_l23_23159


namespace second_smallest_five_digit_in_pascals_triangle_l23_23422

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem second_smallest_five_digit_in_pascals_triangle :
  (∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (10000 ≤ binomial n k) ∧ (binomial n k < 100000) ∧
    (∀ m l : ℕ, m > 0 ∧ l > 0 ∧ (10000 ≤ binomial m l) ∧ (binomial m l < 100000) →
    (binomial n k < binomial m l → binomial n k ≥ 31465)) ∧  binomial n k = 31465) :=
sorry

end second_smallest_five_digit_in_pascals_triangle_l23_23422


namespace calculate_expression_l23_23094

theorem calculate_expression : -Real.sqrt 9 - 4 * (-2) + 2 * Real.cos (Real.pi / 3) = 6 :=
by
  sorry

end calculate_expression_l23_23094


namespace suff_not_nec_l23_23735

theorem suff_not_nec (x : ℝ) : (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬(x ≤ 0)) :=
by
  sorry

end suff_not_nec_l23_23735


namespace math_problem_l23_23093

theorem math_problem :
  -50 * 3 - (-2.5) / 0.1 = -125 := by
sorry

end math_problem_l23_23093


namespace no_fib_right_triangle_l23_23798

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem no_fib_right_triangle (n : ℕ) : 
  ¬ (fibonacci n)^2 + (fibonacci (n+1))^2 = (fibonacci (n+2))^2 := 
by 
  sorry

end no_fib_right_triangle_l23_23798


namespace calculate_total_feet_in_garden_l23_23892

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end calculate_total_feet_in_garden_l23_23892


namespace cost_equation_l23_23389

def cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem cost_equation (W : ℕ) : cost W = 
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10 :=
by
  -- Proof goes here
  sorry

end cost_equation_l23_23389


namespace Mika_stickers_l23_23992

theorem Mika_stickers
  (initial_stickers : ℕ)
  (bought_stickers : ℕ)
  (received_stickers : ℕ)
  (given_stickers : ℕ)
  (used_stickers : ℕ)
  (final_stickers : ℕ) :
  initial_stickers = 45 →
  bought_stickers = 53 →
  received_stickers = 35 →
  given_stickers = 19 →
  used_stickers = 86 →
  final_stickers = initial_stickers + bought_stickers + received_stickers - given_stickers - used_stickers →
  final_stickers = 28 :=
by
  intros
  sorry

end Mika_stickers_l23_23992


namespace tim_books_l23_23547

def has_some_books (Tim Sam : ℕ) : Prop :=
  Sam = 52 ∧ Tim + Sam = 96

theorem tim_books (Tim : ℕ) :
  has_some_books Tim 52 → Tim = 44 := 
by
  intro h
  obtain ⟨hSam, hTogether⟩ := h
  sorry

end tim_books_l23_23547


namespace circles_intersect_and_common_chord_l23_23139

open Real

def circle1 (x y : ℝ) := x^2 + y^2 - 6 * x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6 = 0

theorem circles_intersect_and_common_chord :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) ∧ (∀ x y : ℝ, circle1 x y → circle2 x y → 3 * x - 2 * y = 0) :=
by
  sorry

end circles_intersect_and_common_chord_l23_23139


namespace prove_angle_C_prove_max_area_l23_23845

open Real

variables {A B C : ℝ} {a b c : ℝ} (abc_is_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (R : ℝ) (circumradius_is_sqrt2 : R = sqrt 2)
variables (H : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
variables (law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C)

-- Part 1: Prove that angle C = π / 3
theorem prove_angle_C : C = π / 3 :=
sorry

-- Part 2: Prove that the maximum value of the area S of triangle ABC is (3 * sqrt 3) / 2
theorem prove_max_area : (1 / 2) * a * b * sin C ≤ (3 * sqrt 3) / 2 :=
sorry

end prove_angle_C_prove_max_area_l23_23845


namespace intersection_eq_l23_23305

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l23_23305


namespace largest_share_of_partner_l23_23022

theorem largest_share_of_partner 
    (ratios : List ℕ := [2, 3, 4, 4, 6])
    (total_profit : ℕ := 38000) :
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    largest_share = 12000 :=
by
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    have h1 : total_parts = 19 := by
        sorry
    have h2 : part_value = 2000 := by
        sorry
    have h3 : List.maximum ratios = 6 := by
        sorry
    have h4 : largest_share = 12000 := by
        sorry
    exact h4


end largest_share_of_partner_l23_23022


namespace value_of_fraction_l23_23326

theorem value_of_fraction (a b c d e f : ℚ) (h1 : a / b = 1 / 3) (h2 : c / d = 1 / 3) (h3 : e / f = 1 / 3) :
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 :=
by
  sorry

end value_of_fraction_l23_23326


namespace find_f_of_neg_1_l23_23474

-- Define the conditions
variables (a b c : ℝ)
variables (g f : ℝ → ℝ)
axiom g_definition : ∀ x, g x = x^3 + a*x^2 + 2*x + 15
axiom f_definition : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c

axiom g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3)
axiom roots_of_g_are_roots_of_f : ∀ x, g x = 0 → f x = 0

-- Prove the value of f(-1) given the conditions
theorem find_f_of_neg_1 (a : ℝ) (b : ℝ) (c : ℝ) (g f : ℝ → ℝ)
  (h_g_def : ∀ x, g x = x^3 + a*x^2 + 2*x + 15)
  (h_f_def : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c)
  (h_g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3))
  (h_roots : ∀ x, g x = 0 → f x = 0) :
  f (-1) = 3733.25 := 
by {
  sorry
}

end find_f_of_neg_1_l23_23474


namespace sin_double_angle_15_eq_half_l23_23241

theorem sin_double_angle_15_eq_half : 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 := 
sorry

end sin_double_angle_15_eq_half_l23_23241


namespace five_b_value_l23_23496

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := 
by
  sorry

end five_b_value_l23_23496


namespace smallest_n_common_factor_l23_23902

theorem smallest_n_common_factor :
  ∃ n : ℤ, n > 0 ∧ (gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 10 :=
by
  sorry

end smallest_n_common_factor_l23_23902


namespace largest_value_l23_23267

theorem largest_value (A B C D E : ℕ)
  (hA : A = (3 + 5 + 2 + 8))
  (hB : B = (3 * 5 + 2 + 8))
  (hC : C = (3 + 5 * 2 + 8))
  (hD : D = (3 + 5 + 2 * 8))
  (hE : E = (3 * 5 * 2 * 8)) :
  max (max (max (max A B) C) D) E = E := 
sorry

end largest_value_l23_23267


namespace minimum_choir_members_l23_23064

theorem minimum_choir_members:
  ∃ n : ℕ, (n % 9 = 0) ∧ (n % 10 = 0) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, (m % 9 = 0) ∧ (m % 10 = 0) ∧ (m % 11 = 0) → n ≤ m) → n = 990 :=
by
  sorry

end minimum_choir_members_l23_23064


namespace Jim_time_to_fill_pool_l23_23668

-- Definitions for the work rates of Sue, Tony, and their combined work rate.
def Sue_work_rate : ℚ := 1 / 45
def Tony_work_rate : ℚ := 1 / 90
def Combined_work_rate : ℚ := 1 / 15

-- Proving the time it takes for Jim to fill the pool alone.
theorem Jim_time_to_fill_pool : ∃ J : ℚ, 1 / J + Sue_work_rate + Tony_work_rate = Combined_work_rate ∧ J = 30 :=
by {
  sorry
}

end Jim_time_to_fill_pool_l23_23668


namespace two_digit_powers_of_three_l23_23834

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l23_23834


namespace minimum_value_ineq_l23_23514

theorem minimum_value_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
    (1 / (a + 2 * b)) + (1 / (b + 2 * c)) + (1 / (c + 2 * a)) ≥ 3 := 
by
  sorry

end minimum_value_ineq_l23_23514


namespace smallest_n_l23_23700

theorem smallest_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n = 3 * k) (h3 : ∃ m : ℕ, 3 * n = 5 * m) : n = 15 :=
sorry

end smallest_n_l23_23700


namespace determine_clothes_l23_23770

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l23_23770


namespace impossibility_of_dividing_into_three_similar_piles_l23_23352

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l23_23352


namespace trig_identity_l23_23057

theorem trig_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 :=
sorry

end trig_identity_l23_23057


namespace cubes_divisible_by_nine_l23_23551

theorem cubes_divisible_by_nine (n : ℕ) (hn : n > 0) : 
    (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := by
  sorry

end cubes_divisible_by_nine_l23_23551


namespace identify_clothes_l23_23764

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l23_23764


namespace count_two_digit_powers_of_three_l23_23832

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l23_23832


namespace B_share_after_tax_l23_23082

noncomputable def B_share (x : ℝ) : ℝ := 3 * x
noncomputable def salary_proportion (A B C D : ℝ) (x : ℝ) :=
  A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ D = 6 * x
noncomputable def D_more_than_C (D C : ℝ) : Prop :=
  D - C = 700
noncomputable def meets_minimum_wage (B : ℝ) : Prop :=
  B ≥ 1000
noncomputable def tax_deduction (B : ℝ) : ℝ :=
  if B > 1500 then B - 0.15 * (B - 1500) else B

theorem B_share_after_tax (A B C D : ℝ) (x : ℝ) (h1 : salary_proportion A B C D x)
  (h2 : D_more_than_C D C) (h3 : meets_minimum_wage B) :
  tax_deduction B = 1050 :=
by
  sorry

end B_share_after_tax_l23_23082


namespace max_value_of_expression_l23_23940

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 := 
sorry

end max_value_of_expression_l23_23940


namespace contradiction_with_angles_l23_23426

-- Definitions of conditions
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

-- The proposition we want to prove by contradiction
def at_least_one_angle_not_greater_than_60 (α β γ : ℝ) : Prop := α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The assumption for contradiction
def all_angles_greater_than_60 (α β γ : ℝ) : Prop := α > 60 ∧ β > 60 ∧ γ > 60

-- The proof problem
theorem contradiction_with_angles (α β γ : ℝ) (h : triangle α β γ) :
  ¬ all_angles_greater_than_60 α β γ → at_least_one_angle_not_greater_than_60 α β γ :=
sorry

end contradiction_with_angles_l23_23426


namespace total_food_amount_l23_23943

-- Define constants for the given problem
def chicken : ℕ := 16
def hamburgers : ℕ := chicken / 2
def hot_dogs : ℕ := hamburgers + 2
def sides : ℕ := hot_dogs / 2

-- Prove the total amount of food Peter will buy is 39 pounds
theorem total_food_amount : chicken + hamburgers + hot_dogs + sides = 39 := by
  sorry

end total_food_amount_l23_23943


namespace length_of_arc_l23_23657

theorem length_of_arc (angle_SIT : ℝ) (radius_OS : ℝ) (h1 : angle_SIT = 45) (h2 : radius_OS = 15) :
  arc_length_SIT = 7.5 * Real.pi :=
by
  sorry

end length_of_arc_l23_23657


namespace number_is_280_l23_23730

theorem number_is_280 (x : ℝ) (h : x / 5 + 4 = x / 4 - 10) : x = 280 := 
by 
  sorry

end number_is_280_l23_23730


namespace necessary_not_sufficient_condition_l23_23742

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the condition for the problem
def condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the proof problem: Prove that the interval is a necessary but not sufficient condition for f(x) < 0
theorem necessary_not_sufficient_condition : 
  ∀ x : ℝ, condition x → ¬ (∀ y : ℝ, condition y → f y < 0) :=
sorry

end necessary_not_sufficient_condition_l23_23742


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l23_23830

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l23_23830


namespace count_two_digit_powers_of_three_l23_23833

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l23_23833


namespace min_value_frac_sum_l23_23485

theorem min_value_frac_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ m, ∀ x y, 0 < x → 0 < y → 2 * x + y = 1 → m ≤ (1/x + 1/y) ∧ (1/x + 1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_frac_sum_l23_23485


namespace calc_1_calc_2_l23_23604

-- Question 1
theorem calc_1 : (5 / 17 * -4 - 5 / 17 * 15 + -5 / 17 * -2) = -5 :=
by sorry

-- Question 2
theorem calc_2 : (-1^2 + 36 / ((-3)^2) - ((-3 + 3 / 7) * (-7 / 24))) = 2 :=
by sorry

end calc_1_calc_2_l23_23604


namespace annual_production_2010_l23_23872

-- Defining the parameters
variables (a x : ℝ)

-- Define the growth formula
def annual_growth (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate)^years

-- The statement we need to prove
theorem annual_production_2010 :
  annual_growth a x 5 = a * (1 + x) ^ 5 :=
by
  sorry

end annual_production_2010_l23_23872


namespace solve_quadratic_l23_23010

theorem solve_quadratic {x : ℝ} (h : 2 * (x - 1)^2 = x - 1) : x = 1 ∨ x = 3 / 2 :=
sorry

end solve_quadratic_l23_23010


namespace red_sequence_57_eq_103_l23_23503

-- Definitions based on conditions described in the problem
def red_sequence : Nat → Nat
| 0 => 1  -- First number is 1
| 1 => 2  -- Next even number
| 2 => 4  -- Next even number
-- Continue defining based on patterns from problem
| (n+3) => -- Each element recursively following the pattern
 sorry  -- Detailed pattern definition is skipped

-- Main theorem: the 57th number in the red subsequence is 103
theorem red_sequence_57_eq_103 : red_sequence 56 = 103 :=
 sorry

end red_sequence_57_eq_103_l23_23503


namespace find_value_of_a3_a6_a9_l23_23662

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

variables {a : ℕ → ℤ} (d : ℤ)

-- Given conditions
axiom cond1 : a 1 + a 4 + a 7 = 45
axiom cond2 : a 2 + a 5 + a 8 = 29

-- Lean 4 Statement
theorem find_value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 13 :=
sorry

end find_value_of_a3_a6_a9_l23_23662


namespace probability_at_least_one_female_l23_23201

open Nat

theorem probability_at_least_one_female :
  let males := 2
  let females := 3
  let total_students := males + females
  let select := 2
  let total_ways := choose total_students select
  let ways_at_least_one_female : ℕ := (choose females 1) * (choose males 1) + choose females 2
  (ways_at_least_one_female / total_ways : ℚ) = 9 / 10 := by
  sorry

end probability_at_least_one_female_l23_23201


namespace bond_selling_price_l23_23055

def bond_face_value : ℝ := 5000
def bond_interest_rate : ℝ := 0.06
def interest_approx : ℝ := bond_face_value * bond_interest_rate
def selling_price_interest_rate : ℝ := 0.065
def approximate_selling_price : ℝ := 4615.38

theorem bond_selling_price :
  interest_approx = selling_price_interest_rate * approximate_selling_price :=
sorry

end bond_selling_price_l23_23055


namespace not_always_possible_to_predict_winner_l23_23659

def football_championship (teams : Fin 16 → ℕ) : Prop :=
  ∃ i j : Fin 16, i ≠ j ∧ teams i = teams j ∧
  ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧
               teams (pairs k).fst ≠ teams (pairs k).snd) ∨
  ∃ k : Fin 8, (pairs k).fst = i ∧ (pairs k).snd = j

theorem not_always_possible_to_predict_winner :
  ∀ teams : Fin 16 → ℕ, (∃ i j : Fin 16, i ≠ j ∧ teams i = teams j) →
  ∃ pairs : Fin 16 → Fin 16 × Fin 16,
  (∃ k : Fin 8, teams (pairs k).fst = 15 ∧ teams (pairs k).snd = 15) ↔
  ¬ ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧ teams (pairs k).fst ≠ teams (pairs k).snd) :=
by
  sorry

end not_always_possible_to_predict_winner_l23_23659


namespace train_length_l23_23750

theorem train_length (L : ℝ) : (L + 200) / 15 = (L + 300) / 20 → L = 100 :=
by
  intro h
  -- Skipping the proof steps
  sorry

end train_length_l23_23750


namespace ratio_of_cards_lost_l23_23200

-- Definitions based on the conditions
def purchases_per_week : ℕ := 20
def weeks_per_year : ℕ := 52
def cards_left : ℕ := 520

-- Main statement to be proved
theorem ratio_of_cards_lost (total_cards : ℕ := purchases_per_week * weeks_per_year)
                            (cards_lost : ℕ := total_cards - cards_left) :
                            (cards_lost : ℚ) / total_cards = 1 / 2 :=
by
  sorry

end ratio_of_cards_lost_l23_23200


namespace center_of_circle_l23_23917

noncomputable def center_is_correct (x y : ℚ) : Prop :=
  (5 * x - 2 * y = -10) ∧ (3 * x + y = 0)

theorem center_of_circle : center_is_correct (-10 / 11) (30 / 11) :=
by
  sorry

end center_of_circle_l23_23917


namespace Faraway_not_possible_sum_l23_23973

theorem Faraway_not_possible_sum (h g : ℕ) : (74 ≠ 21 * h + 6 * g) ∧ (89 ≠ 21 * h + 6 * g) :=
by
  sorry

end Faraway_not_possible_sum_l23_23973


namespace sandy_saved_percentage_last_year_l23_23671

noncomputable def sandys_saved_percentage (S : ℝ) (P : ℝ) : ℝ :=
  (P / 100) * S

noncomputable def salary_with_10_percent_more (S : ℝ) : ℝ :=
  1.1 * S

noncomputable def amount_saved_this_year (S : ℝ) : ℝ :=
  0.15 * (salary_with_10_percent_more S)

noncomputable def amount_saved_this_year_compare_last_year (S : ℝ) (P : ℝ) : Prop :=
  amount_saved_this_year S = 1.65 * sandys_saved_percentage S P

theorem sandy_saved_percentage_last_year (S : ℝ) (P : ℝ) :
  amount_saved_this_year_compare_last_year S P → P = 10 :=
by
  sorry

end sandy_saved_percentage_last_year_l23_23671


namespace lowest_number_in_range_l23_23524

theorem lowest_number_in_range (y : ℕ) (h : ∀ x y : ℕ, 0 < x ∧ x < y) : ∃ x : ℕ, x = 999 :=
by
  existsi 999
  sorry

end lowest_number_in_range_l23_23524


namespace total_pencils_l23_23885

   variables (n p t : ℕ)

   -- Condition 1: number of students
   def students := 12

   -- Condition 2: pencils per student
   def pencils_per_student := 3

   -- Theorem statement: Given the conditions, the total number of pencils given by the teacher is 36
   theorem total_pencils : t = students * pencils_per_student :=
   by
   sorry
   
end total_pencils_l23_23885


namespace find_extreme_values_find_m_range_for_zeros_l23_23811

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x + 2

theorem find_extreme_values (m : ℝ) :
  (∀ x > 0, m ≤ 0 → (f x m ≠ 0 ∨ ∀ y > 0, f y m ≥ f x m ∨ f y m ≤ f x m)) ∧
  (∀ x > 0, m > 0 → ∃ x_max, x_max = 1 / m ∧ ∀ y > 0, f y m ≤ f x_max m) := 
sorry

theorem find_m_range_for_zeros (m : ℝ) :
  (∃ a b, a = 1 / Real.exp 2 ∧ b = Real.exp 1 ∧ (f a m = 0 ∧ f b m = 0)) ↔ 
  (m ≥ 3 / Real.exp 1 ∧ m < Real.exp 1) :=
sorry

end find_extreme_values_find_m_range_for_zeros_l23_23811


namespace reciprocal_of_repeating_decimal_l23_23557

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l23_23557


namespace part_one_part_two_l23_23634

noncomputable def a (n : ℕ) : ℚ := if n = 1 then 1 / 2 else 2 ^ (n - 1) / (1 + 2 ^ (n - 1))

noncomputable def b (n : ℕ) : ℚ := n / a n

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => b (i + 1))

/-Theorem:
1. Prove that for all n > 0, a(n) = 2^(n-1) / (1 + 2^(n-1)).
2. Prove that for all n ≥ 3, S(n) > n^2 / 2 + 4.
-/
theorem part_one (n : ℕ) (h : n > 0) : a n = 2 ^ (n - 1) / (1 + 2 ^ (n - 1)) := sorry

theorem part_two (n : ℕ) (h : n ≥ 3) : S n > n ^ 2 / 2 + 4 := sorry

end part_one_part_two_l23_23634


namespace find_natural_number_pairs_l23_23611

theorem find_natural_number_pairs (a b q : ℕ) : 
  (a ∣ b^2 ∧ b ∣ a^2 ∧ (a + 1) ∣ (b^2 + 1)) ↔ 
  ((a = q^2 ∧ b = q) ∨ 
   (a = q^2 ∧ b = q^3) ∨ 
   (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by
  sorry

end find_natural_number_pairs_l23_23611


namespace amy_money_left_l23_23906

-- Definitions for item prices
def stuffed_toy_price : ℝ := 2
def hot_dog_price : ℝ := 3.5
def candy_apple_price : ℝ := 1.5
def soda_price : ℝ := 1.75
def ferris_wheel_ticket_price : ℝ := 2.5

-- Tax rate
def tax_rate : ℝ := 0.1 

-- Initial amount Amy had
def initial_amount : ℝ := 15

-- Function to calculate price including tax
def price_with_tax (price : ℝ) (tax_rate : ℝ) : ℝ := price * (1 + tax_rate)

-- Prices including tax
def stuffed_toy_price_with_tax := price_with_tax stuffed_toy_price tax_rate
def hot_dog_price_with_tax := price_with_tax hot_dog_price tax_rate
def candy_apple_price_with_tax := price_with_tax candy_apple_price tax_rate
def soda_price_with_tax := price_with_tax soda_price tax_rate
def ferris_wheel_ticket_price_with_tax := price_with_tax ferris_wheel_ticket_price tax_rate

-- Discount rates
def discount_most_expensive : ℝ := 0.5
def discount_second_most_expensive : ℝ := 0.25

-- Applying discounts
def discounted_hot_dog_price := hot_dog_price_with_tax * (1 - discount_most_expensive)
def discounted_ferris_wheel_ticket_price := ferris_wheel_ticket_price_with_tax * (1 - discount_second_most_expensive)

-- Total cost with discounts
def total_cost_with_discounts : ℝ := 
  stuffed_toy_price_with_tax + discounted_hot_dog_price + candy_apple_price_with_tax +
  soda_price_with_tax + discounted_ferris_wheel_ticket_price

-- Amount left after purchases
def amount_left : ℝ := initial_amount - total_cost_with_discounts

theorem amy_money_left : amount_left = 5.23 := by
  -- Here the proof will be provided.
  sorry

end amy_money_left_l23_23906


namespace range_of_f_l23_23128

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * Real.sin x + 6

theorem range_of_f :
  ∀ (x : ℝ), Real.sin x ≠ 2 → 
  (1 ≤ f x ∧ f x ≤ 11) :=
by 
  sorry

end range_of_f_l23_23128


namespace smallest_positive_period_of_f_minimum_value_of_f_on_interval_l23_23492

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2) * (Real.sin (x / 2)) * (Real.cos (x / 2)) - (Real.sqrt 2) * (Real.sin (x / 2)) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

theorem minimum_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (-Real.pi) 0, 
  f x = -1 - Real.sqrt 2 / 2 :=
by sorry

end smallest_positive_period_of_f_minimum_value_of_f_on_interval_l23_23492


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l23_23041

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l23_23041


namespace find_value_of_a_minus_b_l23_23648

variable (a b : ℝ)

theorem find_value_of_a_minus_b (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := 
sorry

end find_value_of_a_minus_b_l23_23648


namespace integer_solutions_to_equation_l23_23796

theorem integer_solutions_to_equation :
  { p : ℤ × ℤ | (p.1 ^ 2 * p.2 + 1 = p.1 ^ 2 + 2 * p.1 * p.2 + 2 * p.1 + p.2) } =
  { (-1, -1), (0, 1), (1, -1), (2, -7), (3, 7) } :=
by
  sorry

end integer_solutions_to_equation_l23_23796


namespace triangle_altitude_length_l23_23138

-- Define the problem
theorem triangle_altitude_length (l w h : ℝ) (hl : l = 2 * w) 
  (h_triangle_area : 0.5 * l * h = 0.5 * (l * w)) : h = w := 
by 
  -- Use the provided conditions and the equation setup to continue the proof
  sorry

end triangle_altitude_length_l23_23138


namespace bob_overtime_pay_rate_l23_23784

theorem bob_overtime_pay_rate :
  let regular_pay_rate := 5
  let total_hours := (44, 48)
  let total_pay := 472
  let overtime_hours (hours : Nat) := max 0 (hours - 40)
  let regular_hours (hours : Nat) := min 40 hours
  let total_regular_hours := regular_hours 44 + regular_hours 48
  let total_regular_pay := total_regular_hours * regular_pay_rate
  let total_overtime_hours := overtime_hours 44 + overtime_hours 48
  let total_overtime_pay := total_pay - total_regular_pay
  let overtime_pay_rate := total_overtime_pay / total_overtime_hours
  overtime_pay_rate = 6 := by sorry

end bob_overtime_pay_rate_l23_23784


namespace general_term_sequence_l23_23507

theorem general_term_sequence (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3^n) :
  ∀ n, a n = (3^n - 1) / 2 := 
by
  sorry

end general_term_sequence_l23_23507


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l23_23044

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l23_23044


namespace lettuce_types_l23_23386

/-- Let L be the number of types of lettuce. 
    Given that Terry has 3 types of tomatoes, 4 types of olives, 
    and 2 types of soup. The total number of options for his lunch combo is 48. 
    Prove that L = 2. --/

theorem lettuce_types (L : ℕ) (H : 3 * 4 * 2 * L = 48) : L = 2 :=
by {
  -- beginning of the proof
  sorry
}

end lettuce_types_l23_23386


namespace min_value_of_3a_plus_2_l23_23644

theorem min_value_of_3a_plus_2 
  (a : ℝ) 
  (h : 4 * a^2 + 7 * a + 3 = 2)
  : 3 * a + 2 >= -1 :=
sorry

end min_value_of_3a_plus_2_l23_23644


namespace find_principal_l23_23908

-- Defining the conditions
def A : ℝ := 5292
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The theorem statement
theorem find_principal :
  ∃ (P : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ P = 4800 :=
by
  sorry

end find_principal_l23_23908


namespace clothes_color_proof_l23_23752

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l23_23752


namespace tie_to_shirt_ratio_l23_23617

-- Definitions for the conditions
def pants_cost : ℝ := 20
def shirt_cost : ℝ := 2 * pants_cost
def socks_cost : ℝ := 3
def r : ℝ := sorry -- This will be proved
def tie_cost : ℝ := r * shirt_cost
def uniform_cost : ℝ := pants_cost + shirt_cost + tie_cost + socks_cost

-- The total cost for five uniforms
def total_cost : ℝ := 5 * uniform_cost

-- The given total cost
def given_total_cost : ℝ := 355

-- The theorem to be proved
theorem tie_to_shirt_ratio :
  total_cost = given_total_cost → r = 1 / 5 := 
sorry

end tie_to_shirt_ratio_l23_23617


namespace solve_equation_solve_proportion_l23_23999

theorem solve_equation (x : ℚ) :
  (3 + x) * (30 / 100) = 4.8 → x = 13 :=
by sorry

theorem solve_proportion (x : ℚ) :
  (5 / x) = (9 / 2) / (8 / 5) → x = (16 / 9) :=
by sorry

end solve_equation_solve_proportion_l23_23999


namespace solve_arithmetic_sequence_problem_l23_23286

noncomputable def arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) : Prop :=
  (∀ n, a n = a 0 + n * (a 1 - a 0)) ∧  -- Condition: sequence is arithmetic
  (∀ n, S n = (n * (a 0 + a (n - 1))) / 2) ∧  -- Condition: sum of first n terms
  (m > 1) ∧  -- Condition: m > 1
  (a (m - 1) + a (m + 1) - a m ^ 2 = 0) ∧  -- Given condition
  (S (2 * m - 1) = 38)  -- Given that sum of first 2m-1 terms equals 38

-- The statement we need to prove
theorem solve_arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) :
  arithmetic_sequence_problem a S m → m = 10 :=
by
  sorry  -- Proof to be completed

end solve_arithmetic_sequence_problem_l23_23286


namespace unique_solution_l23_23931

noncomputable def pair_satisfying_equation (m n : ℕ) : Prop :=
  2^m - 1 = 3^n

theorem unique_solution : ∀ (m n : ℕ), m > 0 → n > 0 → pair_satisfying_equation m n → (m, n) = (2, 1) :=
by
  intros m n m_pos n_pos h
  sorry

end unique_solution_l23_23931


namespace max_regular_hours_l23_23248

/-- A man's regular pay is $3 per hour up to a certain number of hours, and his overtime pay rate
    is twice the regular pay rate. The man was paid $180 and worked 10 hours overtime.
    Prove that the maximum number of hours he can work at his regular pay rate is 40 hours.
-/
theorem max_regular_hours (P R OT : ℕ) (hP : P = 180) (hOT : OT = 10) (reg_rate overtime_rate : ℕ)
  (hreg_rate : reg_rate = 3) (hovertime_rate : overtime_rate = 2 * reg_rate) :
  P = reg_rate * R + overtime_rate * OT → R = 40 :=
by
  sorry

end max_regular_hours_l23_23248


namespace sector_area_eq_13pi_l23_23506

theorem sector_area_eq_13pi
    (O A B C : Type)
    (r : ℝ)
    (θ : ℝ)
    (h1 : θ = 130)
    (h2 : r = 6) :
    (θ / 360) * (π * r^2) = 13 * π := by
  sorry

end sector_area_eq_13pi_l23_23506


namespace number_is_seven_point_five_l23_23196

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l23_23196


namespace f_strictly_increasing_intervals_l23_23317

noncomputable def f (x : Real) : Real :=
  x * Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real :=
  x * Real.cos x

theorem f_strictly_increasing_intervals :
  ∀ (x : Real), (-π < x ∧ x < -π / 2 ∨ 0 < x ∧ x < π / 2) → f' x > 0 :=
by
  intros x h
  sorry

end f_strictly_increasing_intervals_l23_23317


namespace intersection_of_A_and_B_l23_23483

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, -1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_A_and_B_l23_23483


namespace mean_of_five_numbers_is_correct_l23_23221

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l23_23221


namespace inequality_one_inequality_two_l23_23256

variable (a b c : ℝ)

-- Conditions given in the problem
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom positive_c : 0 < c
axiom sum_eq_one : a + b + c = 1

-- Statements to prove
theorem inequality_one : ab + bc + ac ≤ 1 / 3 :=
sorry

theorem inequality_two : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end inequality_one_inequality_two_l23_23256


namespace two_digit_powers_of_3_count_l23_23823

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l23_23823


namespace min_edge_disjoint_cycles_l23_23910

noncomputable def minEdgesForDisjointCycles (n : ℕ) (h : n ≥ 6) : ℕ := 3 * (n - 2)

theorem min_edge_disjoint_cycles (n : ℕ) (h : n ≥ 6) : minEdgesForDisjointCycles n h = 3 * (n - 2) := 
by
  sorry

end min_edge_disjoint_cycles_l23_23910


namespace martin_walk_distance_l23_23364

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end martin_walk_distance_l23_23364


namespace toby_photo_shoot_l23_23717

theorem toby_photo_shoot (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_pictures : ℕ) (deleted_post_editing : ℕ) (final_photos : ℕ) (photo_shoot_photos : ℕ) :
  initial_photos = 63 →
  deleted_bad_shots = 7 →
  cat_pictures = 15 →
  deleted_post_editing = 3 →
  final_photos = 84 →
  final_photos = initial_photos - deleted_bad_shots + cat_pictures + photo_shoot_photos - deleted_post_editing →
  photo_shoot_photos = 16 :=
by
  intros
  sorry

end toby_photo_shoot_l23_23717


namespace prove_clothing_colors_l23_23756

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l23_23756


namespace appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l23_23937

-- Definitions for binomial coefficient and Pascal's triangle

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Check occurrences in Pascal's triangle more than three times
theorem appears_more_than_three_times_in_Pascal (n : ℕ) :
  n = 10 ∨ n = 15 ∨ n = 21 → ∃ a b c : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ 
    (binomial_coeff a 2 = n ∨ binomial_coeff a 3 = n) ∧
    (binomial_coeff b 2 = n ∨ binomial_coeff b 3 = n) ∧
    (binomial_coeff c 2 = n ∨ binomial_coeff c 3 = n) := 
by
  sorry

-- Check occurrences in Pascal's triangle more than four times
theorem appears_more_than_four_times_in_Pascal (n : ℕ) :
  n = 120 ∨ n = 210 ∨ n = 3003 → ∃ a b c d : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ (1 < d) ∧ 
    (binomial_coeff a 3 = n ∨ binomial_coeff a 4 = n) ∧
    (binomial_coeff b 3 = n ∨ binomial_coeff b 4 = n) ∧
    (binomial_coeff c 3 = n ∨ binomial_coeff c 4 = n) ∧
    (binomial_coeff d 3 = n ∨ binomial_coeff d 4 = n) := 
by
  sorry

end appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l23_23937


namespace total_students_sampled_l23_23504

theorem total_students_sampled (freq_ratio : ℕ → ℕ → ℕ) (second_group_freq : ℕ) 
  (ratio_condition : freq_ratio 2 1 = 2 ∧ freq_ratio 2 3 = 3) : 
  (6 + second_group_freq + 18) = 48 := 
by 
  sorry

end total_students_sampled_l23_23504


namespace probability_A8_l23_23199

/-- Define the probability of event A_n where the sum of die rolls equals n -/
def P (n : ℕ) : ℚ :=
  1/7 * (if n = 8 then 5/36 + 21/216 + 35/1296 + 35/7776 + 21/46656 +
    7/279936 + 1/1679616 else 0)

theorem probability_A8 : P 8 = (1/7) * (5/36 + 21/216 + 35/1296 + 35/7776 + 
  21/46656 + 7/279936 + 1/1679616) :=
by
  sorry

end probability_A8_l23_23199


namespace unique_solution_of_functional_eqn_l23_23271

theorem unique_solution_of_functional_eqn (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1) → (∀ x : ℝ, f x = x) :=
by
  intros h
  sorry

end unique_solution_of_functional_eqn_l23_23271


namespace fliers_left_l23_23240

theorem fliers_left (initial_fliers : ℕ) (fraction_morning : ℕ) (fraction_afternoon : ℕ) :
  initial_fliers = 2000 → 
  fraction_morning = 1 / 10 → 
  fraction_afternoon = 1 / 4 → 
  (initial_fliers - initial_fliers * fraction_morning - 
  (initial_fliers - initial_fliers * fraction_morning) * fraction_afternoon) = 1350 := by
  intros initial_fliers_eq fraction_morning_eq fraction_afternoon_eq
  sorry

end fliers_left_l23_23240


namespace math_problem_l23_23952

theorem math_problem (a b c d x : ℝ)
  (h1 : a = -(-b))
  (h2 : c = -1 / d)
  (h3 : |x| = 3) :
  x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36 :=
by sorry

end math_problem_l23_23952


namespace parallelogram_angle_bisector_l23_23160

theorem parallelogram_angle_bisector (a b S Q : ℝ) (α : ℝ) 
  (hS : S = a * b * Real.sin α)
  (hQ : Q = (1 / 2) * (a - b) ^ 2 * Real.sin α) :
  (2 * a * b) / (a - b) ^ 2 = (S + Q + Real.sqrt (Q ^ 2 + 2 * Q * S)) / S :=
by
  sorry

end parallelogram_angle_bisector_l23_23160


namespace calculateL_l23_23116

-- Defining the constants T, H, and C
def T : ℕ := 5
def H : ℕ := 10
def C : ℕ := 3

-- Definition of the formula for L
def crushingLoad (T H C : ℕ) : ℚ := (15 * T^3 : ℚ) / (H^2 + C)

-- The theorem to prove
theorem calculateL : crushingLoad T H C = 1875 / 103 := by
  -- Proof goes here
  sorry

end calculateL_l23_23116


namespace solve_ab_eq_l23_23619

theorem solve_ab_eq:
  ∃ a b : ℝ, (1 + (2 : ℂ) * (Complex.I)) * (a : ℂ) + (b : ℂ) = (2 : ℂ) * (Complex.I) ∧ a = 1 ∧ b = -1 := by
  sorry

end solve_ab_eq_l23_23619


namespace incorrect_option_C_l23_23808

theorem incorrect_option_C (a b d : ℝ) (h₁ : ∀ x : ℝ, x ≠ d → x^2 + a * x + b > 0) (h₂ : a > 0) :
  ¬∀ x₁ x₂ : ℝ, (x₁ * x₂ > 0) → ((x₁, x₂) ∈ {p : (ℝ × ℝ) | p.1^2 + a * p.1 - b < 0 ∧ p.2^2 + a * p.2 - b < 0}) :=
sorry

end incorrect_option_C_l23_23808


namespace question4_l23_23727

noncomputable def question1 (ξ : ℝ → MeasureTheory.ProbabilityMeasure ℝ) (σ : ℝ) (P1 : ξ 3 ≤ 1 = 0.23) : Prop :=
  ξ 3 ≤ 5 = 0.77

def data_set : List ℝ := [96, 90, 92, 92, 93, 93, 94, 95, 99, 100]

def percentile_80 (sorted_data : List ℝ) : ℝ :=
  let n := sorted_data.length
  let pos := (0.8 * n).floor
  (sorted_data.get pos.succ + sorted_data.get pos) / 2

theorem question4 : percentile_80 data_set.sorted = 97.5 :=
sorry

end question4_l23_23727


namespace locus_of_circle_center_l23_23678

theorem locus_of_circle_center (a : ℝ) :
  let C := λ (x y : ℝ), x^2 + y^2 - (2 * a^2 - 4) * x - 4 * a^2 * y + 5 * a^4 - 4 = 0 in
  ∃ (x y : ℝ), C x y ∧ -2 ≤ (a^2 - 2) ∧ (a^2 - 2) < 0 → 
    (2 * (a^2 - 2) - (2 * a^2) + 4 = 0) :=
by sorry

end locus_of_circle_center_l23_23678


namespace difference_between_numbers_l23_23532

noncomputable def L : ℕ := 1614
noncomputable def Q : ℕ := 6
noncomputable def R : ℕ := 15

theorem difference_between_numbers (S : ℕ) (h : L = Q * S + R) : L - S = 1348 :=
by {
  -- proof skipped
  sorry
}

end difference_between_numbers_l23_23532


namespace Christine_wandered_hours_l23_23100

theorem Christine_wandered_hours (distance speed : ℝ) (h_distance : distance = 20) (h_speed : speed = 4) : distance / speed = 5 := by
  sorry

end Christine_wandered_hours_l23_23100


namespace sqrt_five_gt_two_l23_23790

theorem sqrt_five_gt_two : Real.sqrt 5 > 2 :=
by
  -- Proof goes here
  sorry

end sqrt_five_gt_two_l23_23790


namespace function_is_increasing_l23_23687

theorem function_is_increasing : ∀ (x1 x2 : ℝ), x1 < x2 → (2 * x1 + 1) < (2 * x2 + 1) :=
by sorry

end function_is_increasing_l23_23687


namespace power_mod_remainder_l23_23124

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end power_mod_remainder_l23_23124


namespace min_Sn_l23_23663

variable {a : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (a₄ : ℤ) (d : ℤ) : Prop :=
  a 4 = a₄ ∧ ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

def Sn (a : ℕ → ℤ) (n : ℕ) :=
  n / 2 * (2 * a 1 + (n - 1) * 3)

theorem min_Sn (a : ℕ → ℤ) (h1 : arithmetic_sequence a (-15) 3) :
  ∃ n : ℕ, (Sn a n = -108) :=
sorry

end min_Sn_l23_23663


namespace part1_part2_part3_l23_23911

-- Define conditions
variables (n : ℕ) (h₁ : 5 ≤ n)

-- Problem part (1): Define p_n and prove its value
def p_n (n : ℕ) := (10 * n) / ((n + 5) * (n + 4))

-- Problem part (2): Define EX and prove its value for n = 5
def EX : ℚ := 5 / 3

-- Problem part (3): Prove n = 20 maximizes P
def P (n : ℕ) := 3 * ((p_n n) ^ 3 - 2 * (p_n n) ^ 2 + (p_n n))
def n_max := 20

-- Making the proof skeletons for clarity, filling in later
theorem part1 : p_n n = 10 * n / ((n + 5) * (n + 4)) :=
sorry

theorem part2 (h₂ : n = 5) : EX = 5 / 3 :=
sorry

theorem part3 : n_max = 20 :=
sorry

end part1_part2_part3_l23_23911


namespace travis_takes_home_money_l23_23231

-- Define the conditions
def total_apples : ℕ := 10000
def apples_per_box : ℕ := 50
def price_per_box : ℕ := 35

-- Define the main theorem to be proved
theorem travis_takes_home_money : (total_apples / apples_per_box) * price_per_box = 7000 := by
  sorry

end travis_takes_home_money_l23_23231


namespace amount_decreased_is_5_l23_23841

noncomputable def x : ℕ := 50
noncomputable def equation (x y : ℕ) : Prop := (1 / 5) * x - y = 5

theorem amount_decreased_is_5 : ∃ y : ℕ, equation x y ∧ y = 5 :=
by
  sorry

end amount_decreased_is_5_l23_23841


namespace total_fence_used_l23_23731

-- Definitions based on conditions
variables {L W : ℕ}
def area (L W : ℕ) := L * W

-- Provided conditions as Lean definitions
def unfenced_side := 40
def yard_area := 240

-- The proof problem statement
theorem total_fence_used (L_eq : L = unfenced_side) (A_eq : area L W = yard_area) : (2 * W + L) = 52 :=
sorry

end total_fence_used_l23_23731


namespace find_number_l23_23193

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l23_23193


namespace complete_square_l23_23695

theorem complete_square (y : ℝ) : y^2 + 12 * y + 40 = (y + 6)^2 + 4 :=
by {
  sorry
}

end complete_square_l23_23695


namespace ratio_of_group_average_l23_23529

theorem ratio_of_group_average
  (d l e : ℕ)
  (avg_group_age : ℕ := 45) 
  (avg_doctors_age : ℕ := 40) 
  (avg_lawyers_age : ℕ := 55) 
  (avg_engineers_age : ℕ := 35)
  (h : (40 * d + 55 * l + 35 * e) / (d + l + e) = avg_group_age)
  : d = 2 * l - e ∧ l = 2 * e :=
sorry

end ratio_of_group_average_l23_23529


namespace correct_outfits_l23_23778

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l23_23778


namespace parallelogram_area_l23_23054

theorem parallelogram_area (base height : ℝ) (h_base : base = 20) (h_height : height = 16) :
  base * height = 320 :=
by
  sorry

end parallelogram_area_l23_23054


namespace find_angle_BXY_l23_23335

noncomputable def angle_AXE (angle_CYX : ℝ) : ℝ := 3 * angle_CYX - 108

theorem find_angle_BXY
  (AB_parallel_CD : Prop)
  (h_parallel : ∀ (AXE CYX : ℝ), angle_AXE CYX = AXE)
  (x : ℝ) :
  (angle_AXE x = x) → x = 54 :=
by
  intro h₁
  unfold angle_AXE at h₁
  sorry

end find_angle_BXY_l23_23335


namespace smallest_perfect_square_divisible_by_5_and_6_l23_23048

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l23_23048


namespace chord_length_l23_23699

noncomputable def circle_eq (θ : ℝ) : ℝ × ℝ :=
  (2 + 5 * Real.cos θ, 1 + 5 * Real.sin θ)

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
  (-2 + 4 * t, -1 - 3 * t)

theorem chord_length :
  let center := (2, 1)
  let radius := 5
  let line_dist := |3 * center.1 + 4 * center.2 + 10| / Real.sqrt (3^2 + 4^2)
  let chord_len := 2 * Real.sqrt (radius^2 - line_dist^2)
  chord_len = 6 := 
by
  sorry

end chord_length_l23_23699


namespace smallest_number_with_exactly_eight_factors_l23_23419

theorem smallest_number_with_exactly_eight_factors
    (n : ℕ)
    (h1 : ∃ a b : ℕ, (a + 1) * (b + 1) = 8)
    (h2 : ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p^a * q^b) : 
    n = 24 := by
  sorry

end smallest_number_with_exactly_eight_factors_l23_23419


namespace total_cases_after_three_weeks_l23_23665

theorem total_cases_after_three_weeks (week1_cases week2_cases week3_cases : ℕ) 
  (h1 : week1_cases = 5000)
  (h2 : week2_cases = week1_cases + week1_cases / 10 * 3)
  (h3 : week3_cases = week2_cases - week2_cases / 10 * 2) :
  week1_cases + week2_cases + week3_cases = 16700 := 
by
  sorry

end total_cases_after_three_weeks_l23_23665


namespace probability_at_least_one_hit_l23_23702

-- Define probabilities of each shooter hitting the target
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complementary probabilities (each shooter misses the target)
def P_A_miss : ℚ := 1 - P_A
def P_B_miss : ℚ := 1 - P_B
def P_C_miss : ℚ := 1 - P_C

-- Calculate the probability of all shooters missing the target
def P_all_miss : ℚ := P_A_miss * P_B_miss * P_C_miss

-- Calculate the probability of at least one shooter hitting the target
def P_at_least_one_hit : ℚ := 1 - P_all_miss

-- The theorem to be proved
theorem probability_at_least_one_hit : 
  P_at_least_one_hit = 3 / 4 := 
by sorry

end probability_at_least_one_hit_l23_23702


namespace equivalent_operation_l23_23237

theorem equivalent_operation (x : ℚ) : (x * (2 / 5)) / (4 / 7) = x * (7 / 10) :=
by
  sorry

end equivalent_operation_l23_23237


namespace tax_diminished_percentage_l23_23538

theorem tax_diminished_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) (X : ℝ) 
  (h : T * (1 - X / 100) * C * 1.15 = T * C * 0.9315) : X = 19 :=
by 
  sorry

end tax_diminished_percentage_l23_23538


namespace certain_number_divisibility_l23_23817

theorem certain_number_divisibility {n : ℕ} (h : ∃ count : ℕ, count = 50 ∧ (count = (300 / (2 * n)))) : n = 3 :=
by
  sorry

end certain_number_divisibility_l23_23817


namespace find_x1_l23_23310

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 := 
sorry

end find_x1_l23_23310


namespace max_value_x_plus_y_l23_23178

theorem max_value_x_plus_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 48) (hx_mult_4 : x % 4 = 0) : x + y ≤ 49 :=
sorry

end max_value_x_plus_y_l23_23178


namespace no_valid_pairs_l23_23960

theorem no_valid_pairs : ∀ (a b : ℕ), (a > 0) → (b > 0) → (a ≥ b) → 
  a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b → 
  false := by
  sorry

end no_valid_pairs_l23_23960


namespace least_positive_integer_with_eight_factors_l23_23418

noncomputable def numDivisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d => d > 0 ∧ n % d = 0)

theorem least_positive_integer_with_eight_factors : ∃ n : ℕ, n > 0 ∧ numDivisors n = 8 ∧ (∀ m : ℕ, m > 0 → numDivisors m = 8 → n ≤ m) := 
  sorry

end least_positive_integer_with_eight_factors_l23_23418


namespace cos_alpha_sqrt_l23_23839

theorem cos_alpha_sqrt {α : ℝ} (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α ∧ α ≤ π) : 
  Real.cos α = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end cos_alpha_sqrt_l23_23839


namespace smaller_number_eq_l23_23405

variable (m n t s : ℝ)
variable (h_ratio : m / n = t)
variable (h_sum : m + n = s)
variable (h_t_gt_one : t > 1)

theorem smaller_number_eq : n = s / (1 + t) :=
by sorry

end smaller_number_eq_l23_23405


namespace inequality_proof_l23_23517

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a + b + c + a * b + b * c + c * a + a * b * c = 7)

theorem inequality_proof : 
  (Real.sqrt (a ^ 2 + b ^ 2 + 2) + Real.sqrt (b ^ 2 + c ^ 2 + 2) + Real.sqrt (c ^ 2 + a ^ 2 + 2)) ≥ 6 := by
  sorry

end inequality_proof_l23_23517


namespace minimum_quotient_value_l23_23421

-- Helper definition to represent the quotient 
def quotient (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d) / (a + b + c + d)

-- Conditions: digits are distinct and non-zero 
def distinct_and_nonzero (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem minimum_quotient_value :
  ∀ (a b c d : ℕ), distinct_and_nonzero a b c d → quotient a b c d = 71.9 :=
by sorry

end minimum_quotient_value_l23_23421


namespace Angle_Not_Equivalent_l23_23084

theorem Angle_Not_Equivalent (θ : ℤ) : (θ = -750) → (680 % 360 ≠ θ % 360) :=
by
  intro h
  have h1 : 680 % 360 = 320 := by norm_num
  have h2 : -750 % 360 = -30 % 360 := by norm_num
  have h3 : -30 % 360 = 330 := by norm_num
  rw [h, h2, h3]
  sorry

end Angle_Not_Equivalent_l23_23084


namespace dana_hours_sunday_l23_23466

-- Define the constants given in the problem
def hourly_rate : ℝ := 13
def hours_worked_friday : ℝ := 9
def hours_worked_saturday : ℝ := 10
def total_earnings : ℝ := 286

-- Define the function to compute total earnings from worked hours and hourly rate
def earnings (hours : ℝ) (rate : ℝ) : ℝ := hours * rate

-- Define the proof problem to show the number of hours worked on Sunday
theorem dana_hours_sunday (hours_sunday : ℝ) :
  earnings hours_worked_friday hourly_rate
  + earnings hours_worked_saturday hourly_rate
  + earnings hours_sunday hourly_rate = total_earnings ->
  hours_sunday = 3 :=
by
  sorry -- proof to be filled in

end dana_hours_sunday_l23_23466


namespace inequality_ab_l23_23513

theorem inequality_ab (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end inequality_ab_l23_23513


namespace necessary_and_sufficient_condition_for_f_to_be_odd_l23_23881

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f (a b x : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem necessary_and_sufficient_condition_for_f_to_be_odd (a b : ℝ) :
  is_odd_function (f a b) ↔ sorry :=
by
  -- This is where the proof would go.
  sorry

end necessary_and_sufficient_condition_for_f_to_be_odd_l23_23881


namespace sufficient_not_necessary_condition_l23_23954

theorem sufficient_not_necessary_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : (0 < x ∧ x < 2) → (x^2 - x - 2 < 0) :=
by
  intros h
  sorry

end sufficient_not_necessary_condition_l23_23954


namespace tree_planting_campaign_l23_23552

theorem tree_planting_campaign
  (P : ℝ)
  (h1 : 456 = P * (1 - 1/20))
  (h2 : P ≥ 0)
  : (P * (1 + 0.1)) = (456 / (1 - 1/20) * 1.1) :=
by
  sorry

end tree_planting_campaign_l23_23552


namespace solve_cubic_equation_l23_23121

theorem solve_cubic_equation :
  ∀ x : ℝ, (x^3 + 2 * (x + 1)^3 + (x + 2)^3 = (x + 4)^3) → x = 3 :=
by
  intro x
  sorry

end solve_cubic_equation_l23_23121


namespace min_sum_of_2x2_grid_l23_23791

theorem min_sum_of_2x2_grid (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum : a * b + c * d + a * c + b * d = 2015) : a + b + c + d = 88 :=
sorry

end min_sum_of_2x2_grid_l23_23791


namespace extreme_point_property_l23_23489

variables (f : ℝ → ℝ) (a b x x₀ x₁ : ℝ) 

-- Define the function f
def func (x : ℝ) := x^3 - a * x - b

-- The main theorem
theorem extreme_point_property (h₀ : ∃ x₀, ∃ x₁, (x₀ ≠ 0) ∧ (x₀^2 = a / 3) ∧ (x₁ ≠ x₀) ∧ (func a b x₀ = func a b x₁)) :
  x₁ + 2 * x₀ = 0 :=
sorry

end extreme_point_property_l23_23489


namespace parallel_lines_k_l23_23322

theorem parallel_lines_k (k : ℝ) 
  (h₁ : k ≠ 0)
  (h₂ : ∀ x y : ℝ, (x - k * y - k = 0) = (y = (1 / k) * x - 1))
  (h₃ : ∀ x : ℝ, (y = k * (x - 1))) :
  k = -1 :=
by
  sorry

end parallel_lines_k_l23_23322


namespace martha_clothes_total_l23_23368

-- Given conditions
def jackets_bought : Nat := 4
def t_shirts_bought : Nat := 9
def free_jacket_ratio : Nat := 2
def free_t_shirt_ratio : Nat := 3

-- Problem statement to prove
theorem martha_clothes_total :
  (jackets_bought + jackets_bought / free_jacket_ratio) + 
  (t_shirts_bought + t_shirts_bought / free_t_shirt_ratio) = 18 := 
by 
  sorry

end martha_clothes_total_l23_23368


namespace vertex_of_given_function_l23_23712

noncomputable def vertex_coordinates (f : ℝ → ℝ) : ℝ × ℝ := 
  (-2, 1)  -- Prescribed coordinates for this specific function form.

def function_vertex (x : ℝ) : ℝ :=
  -3 * (x + 2) ^ 2 + 1

theorem vertex_of_given_function : 
  vertex_coordinates function_vertex = (-2, 1) :=
by
  sorry

end vertex_of_given_function_l23_23712


namespace original_soldiers_eq_136_l23_23544

-- Conditions
def original_soldiers (n : ℕ) : ℕ := 8 * n
def after_adding_120 (n : ℕ) : ℕ := original_soldiers n + 120
def after_removing_120 (n : ℕ) : ℕ := original_soldiers n - 120

-- Given that both after_adding_120 n and after_removing_120 n are perfect squares.
def is_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- Theorem statement
theorem original_soldiers_eq_136 : ∃ n : ℕ, original_soldiers n = 136 ∧ 
                                   is_square (after_adding_120 n) ∧ 
                                   is_square (after_removing_120 n) :=
sorry

end original_soldiers_eq_136_l23_23544


namespace min_losses_max_loses_l23_23433

theorem min_losses_max_loses (n : ℕ) (h_n : n = 12)
  (h_players : ∀ (i j : ℕ), i ≠ j → ∃ (m: ℕ), m ≤ n - 1) :
  ∃ (LB : ℕ), LB ≥ ⌈((n - 1)/ 2: ℝ)⌉.to_nat := 
by
  have h_n_spec : n = 12 := rfl
  have h_matches : ∀ (i j : ℕ), i ≠ j → ∃ (m: ℕ), m ≤ n - 1 := λ _ _ _, sorry
  use(⌈((n - 1)/ 2: ℝ)⌉.to_nat)
  have h_lb : ⌈((n - 1)/ 2: ℝ)⌉.to_nat = 6 := sorry
  linarith

end min_losses_max_loses_l23_23433


namespace find_t_l23_23677

/-
Let points A and B be on the coordinate plane with coordinates (2t-3, 0) and (1, 2t+2), respectively. 
The square of the distance between the midpoint of AB and point A is equal to 2t^2 + 3t. 
What is the value of t?
-/
noncomputable def A (t : ℝ) : ℝ × ℝ := (2 * t - 3, 0)
noncomputable def B (t : ℝ) : ℝ × ℝ := (1, 2 * t + 2)
noncomputable def M (t : ℝ) : ℝ × ℝ := ((2 * t - 3 + 1) / 2, (0 + 2 * t + 2) / 2)
noncomputable def distance_square (x y : ℝ × ℝ) : ℝ := (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem find_t (t : ℝ) (h : distance_square (M t) (A t) = 2 * t^2 + 3 * t) : t = 10 / 7 :=
by
  -- Proof steps to be filled in later
  sorry

end find_t_l23_23677


namespace girls_to_boys_ratio_l23_23846

theorem girls_to_boys_ratio (g b : ℕ) (h1 : g = b + 5) (h2 : g + b = 35) : g / b = 4 / 3 :=
by
  sorry

end girls_to_boys_ratio_l23_23846


namespace geometric_series_sum_l23_23464

theorem geometric_series_sum :
  let a := -2
  let r := 4
  let n := 10
  let S := (a * (r^n - 1)) / (r - 1)
  S = -699050 :=
by
  sorry

end geometric_series_sum_l23_23464


namespace rectangle_x_value_l23_23589

theorem rectangle_x_value (x : ℝ) (h : (4 * x) * (x + 7) = 2 * (4 * x) + 2 * (x + 7)) : x = 0.675 := 
sorry

end rectangle_x_value_l23_23589


namespace greatest_two_digit_number_with_digit_product_16_l23_23416

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digit_product (n m : ℕ) : Prop :=
  n * m = 16

def from_digits (n m : ℕ) : ℕ :=
  10 * n + m

theorem greatest_two_digit_number_with_digit_product_16 :
  ∀ n m, is_two_digit_number (from_digits n m) → digit_product n m → (82 ≥ from_digits n m) :=
by
  intros n m h1 h2
  sorry

end greatest_two_digit_number_with_digit_product_16_l23_23416


namespace subtraction_correct_l23_23928

def x : ℝ := 5.75
def y : ℝ := 1.46
def result : ℝ := 4.29

theorem subtraction_correct : x - y = result := 
by
  sorry

end subtraction_correct_l23_23928


namespace probability_of_drawing_1_red_1_white_l23_23501

-- Definitions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Probabilities
def p_red_first_white_second : ℚ := (red_balls / total_balls : ℚ) * (white_balls / total_balls : ℚ)
def p_white_first_red_second : ℚ := (white_balls / total_balls : ℚ) * (red_balls / total_balls : ℚ)

-- Total probability
def total_probability : ℚ := p_red_first_white_second + p_white_first_red_second

theorem probability_of_drawing_1_red_1_white :
  total_probability = 12 / 25 := by
  sorry

end probability_of_drawing_1_red_1_white_l23_23501


namespace second_smallest_five_digit_in_pascals_triangle_l23_23423

theorem second_smallest_five_digit_in_pascals_triangle : ∃ n k, (10000 < binomial n k) ∧ (binomial n k < 100000) ∧ 
    (∀ m l, (m < n ∨ (m = n ∧ l < k)) → (10000 ≤ binomial m l → binomial m l < binomial n k)) ∧ 
    binomial n k = 10001 :=
begin
  sorry
end

end second_smallest_five_digit_in_pascals_triangle_l23_23423


namespace volunteer_hours_per_year_l23_23978

def volunteers_per_month : ℕ := 2
def hours_per_session : ℕ := 3
def months_per_year : ℕ := 12

theorem volunteer_hours_per_year :
  volunteers_per_month * months_per_year * hours_per_session = 72 :=
by
  -- Proof is omitted
  sorry

end volunteer_hours_per_year_l23_23978


namespace ellipse_focus_distance_l23_23481

theorem ellipse_focus_distance :
  ∀ {x y : ℝ},
    (x^2) / 25 + (y^2) / 16 = 1 →
    (dist (x, y) (3, 0) = 8) →
    dist (x, y) (-3, 0) = 2 :=
by
  intro x y h₁ h₂
  sorry

end ellipse_focus_distance_l23_23481


namespace probability_two_odds_l23_23137

open Finset

-- Define the set A = {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the set O = {1, 3, 5} (the set of odd numbers in A)
def O : Finset ℕ := {1, 3, 5}

-- Define total number of outcomes when choosing 2 from A without replacement
def total_outcomes : ℕ := choose 5 2

-- Define the number of outcomes where both chosen numbers are odd
def odd_outcomes : ℕ := choose 3 2

-- Calculate the probability p of choosing two odd numbers from A without replacement
def probability_odd : ℚ := odd_outcomes / total_outcomes

theorem probability_two_odds :
  probability_odd = 3 / 10 :=
by
  sorry

end probability_two_odds_l23_23137


namespace chebyshevs_inequality_two_dim_l23_23176

open ProbabilityTheory

variables {Ω : Type*} {P : ProbMeasure Ω}

-- Given the definitions of random variables ξ and η
variables (ξ η : Ω → ℝ)

-- Expected values and variances
variables [is_finite_measure P] [integrable ξ P] [integrable η P]

-- Correlation coefficient ρ
variable (ρ : ℝ)

-- Positive real number ε
variable (ε : ℝ) (hε : ε > 0)

-- The statement to be proved
theorem chebyshevs_inequality_two_dim :
  P {ω | |ξ ω - 𝔼[ξ]| ≥ ε * sqrt (var ξ P) ∨ |η ω - 𝔼[η]| ≥ ε * sqrt (var η P)} ≤ 
  1 / (ε^2) * (1 + sqrt (1 - ρ^2)) :=
sorry

end chebyshevs_inequality_two_dim_l23_23176


namespace savings_percentage_correct_l23_23068

variables (price_jacket : ℕ) (price_shirt : ℕ) (price_hat : ℕ)
          (discount_jacket : ℕ) (discount_shirt : ℕ) (discount_hat : ℕ)

def original_total_cost (price_jacket price_shirt price_hat : ℕ) : ℕ :=
  price_jacket + price_shirt + price_hat

def savings (price : ℕ) (discount : ℕ) : ℕ :=
  price * discount / 100

def total_savings (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  (savings price_jacket discount_jacket) + (savings price_shirt discount_shirt) + (savings price_hat discount_hat)

def total_savings_percentage (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  total_savings price_jacket price_shirt price_hat discount_jacket discount_shirt discount_hat * 100 /
  original_total_cost price_jacket price_shirt price_hat

theorem savings_percentage_correct :
  total_savings_percentage 100 50 30 30 60 50 = 4167 / 100 :=
sorry

end savings_percentage_correct_l23_23068


namespace basketball_player_scores_mode_median_l23_23438

theorem basketball_player_scores_mode_median :
  let scores := [20, 18, 23, 17, 20, 20, 18]
  let ordered_scores := List.sort scores
  let mode := 20
  let median := 20
  (mode = List.maximum (List.frequency ordered_scores)) ∧ 
  (median = List.nthLe ordered_scores (List.length ordered_scores / 2) sorry) :=
by
  sorry

end basketball_player_scores_mode_median_l23_23438


namespace solution_set_inequality_system_l23_23707

theorem solution_set_inequality_system (
  x : ℝ
) : (x + 1 ≥ 0 ∧ (x - 1) / 2 < 1) ↔ (-1 ≤ x ∧ x < 3) := by
  sorry

end solution_set_inequality_system_l23_23707


namespace crazy_silly_school_books_movies_correct_l23_23714

noncomputable def crazy_silly_school_books_movies (B M : ℕ) : Prop :=
  M = 61 ∧ M = B + 2 ∧ M = 10 ∧ B = 8

theorem crazy_silly_school_books_movies_correct {B M : ℕ} :
  crazy_silly_school_books_movies B M → B = 8 :=
by
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end crazy_silly_school_books_movies_correct_l23_23714


namespace sum_of_squares_l23_23804

variables (x y z w : ℝ)

def condition1 := (x^2 / (2^2 - 1^2)) + (y^2 / (2^2 - 3^2)) + (z^2 / (2^2 - 5^2)) + (w^2 / (2^2 - 7^2)) = 1
def condition2 := (x^2 / (4^2 - 1^2)) + (y^2 / (4^2 - 3^2)) + (z^2 / (4^2 - 5^2)) + (w^2 / (4^2 - 7^2)) = 1
def condition3 := (x^2 / (6^2 - 1^2)) + (y^2 / (6^2 - 3^2)) + (z^2 / (6^2 - 5^2)) + (w^2 / (6^2 - 7^2)) = 1
def condition4 := (x^2 / (8^2 - 1^2)) + (y^2 / (8^2 - 3^2)) + (z^2 / (8^2 - 5^2)) + (w^2 / (8^2 - 7^2)) = 1

theorem sum_of_squares : condition1 x y z w → condition2 x y z w → 
                          condition3 x y z w → condition4 x y z w →
                          (x^2 + y^2 + z^2 + w^2 = 36) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_squares_l23_23804


namespace remainder_div_modulo_l23_23743

theorem remainder_div_modulo (N : ℕ) (h1 : N % 19 = 7) : N % 20 = 6 :=
by
  sorry

end remainder_div_modulo_l23_23743


namespace ratio_future_age_l23_23404

variable (S M : ℕ)

theorem ratio_future_age (h1 : (S : ℝ) / M = 7 / 2) (h2 : S - 6 = 78) : 
  ((S + 16) : ℝ) / (M + 16) = 5 / 2 := 
by
  sorry

end ratio_future_age_l23_23404


namespace proof_problem_l23_23606

noncomputable def x_values_satisfy_equation (x : ℝ) : Prop :=
  let y := 3 * x
  4 * y^2 + y + 5 = 2 * (8 * x^2 + y + 3)

theorem proof_problem : 
  x_values_satisfy_equation (3 + Real.sqrt 89) / 40 ∧ x_values_satisfy_equation (3 - Real.sqrt 89) / 40 :=
sorry

end proof_problem_l23_23606


namespace first_term_exceeding_10000_l23_23021

theorem first_term_exceeding_10000 :
  ∃ (n : ℕ), (2^(n-1) > 10000) ∧ (2^(n-1) = 16384) :=
by
  sorry

end first_term_exceeding_10000_l23_23021


namespace sqrt_expression_evaluation_l23_23603

theorem sqrt_expression_evaluation (sqrt48 : Real) (sqrt1div3 : Real) 
  (h1 : sqrt48 = 4 * Real.sqrt 3) (h2 : sqrt1div3 = Real.sqrt (1 / 3)) :
  (-1 / 2) * sqrt48 * sqrt1div3 = -2 :=
by 
  rw [h1, h2]
  -- Continue with the simplification steps, however
  sorry

end sqrt_expression_evaluation_l23_23603


namespace find_tangent_circle_l23_23321

-- Define circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the problem statement as a theorem
theorem find_tangent_circle :
  ∃ (x0 y0 : ℝ), (x - x0)^2 + (y - y0)^2 = 5/4 ∧ (x0, y0) = (1/2, 1) ∧
                   ∀ (x y : ℝ), (circle1 x y → circle2 x y → line_l (x0 + x) (y0 + y) ) :=
sorry

end find_tangent_circle_l23_23321


namespace child_running_speed_on_still_sidewalk_l23_23440

theorem child_running_speed_on_still_sidewalk (c s : ℕ) 
  (h1 : c + s = 93) 
  (h2 : c - s = 55) : c = 74 :=
sorry

end child_running_speed_on_still_sidewalk_l23_23440


namespace find_n_values_l23_23710

-- Define a function to sum the first n consecutive natural numbers starting from k
def sum_consecutive_numbers (n k : ℕ) : ℕ :=
  n * k + (n * (n - 1)) / 2

-- Define a predicate to check if a number is a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the theorem statement
theorem find_n_values (n : ℕ) (k : ℕ) :
  is_prime (sum_consecutive_numbers n k) →
  n = 1 ∨ n = 2 :=
sorry

end find_n_values_l23_23710


namespace pyramid_volume_l23_23997

-- Define the given conditions
def regular_octagon (A B C D E F G H : Point) : Prop := sorry
def right_pyramid (P A B C D E F G H : Point) : Prop := sorry
def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := sorry

-- Define the specific pyramid problem with all the given conditions
noncomputable def volume_pyramid (P A B C D E F G H : Point) (height : ℝ) (base_area : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D E F G H P : Point) 
(h1 : regular_octagon A B C D E F G H)
(h2 : right_pyramid P A B C D E F G H)
(h3 : equilateral_triangle P A D 10) :
  volume_pyramid P A B C D E F G H (5 * Real.sqrt 3) (50 * Real.sqrt 3) = 250 := 
sorry

end pyramid_volume_l23_23997


namespace t_minus_s_equals_neg_17_25_l23_23592

noncomputable def t : ℝ := (60 + 30 + 20 + 5 + 5) / 5
noncomputable def s : ℝ := (60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 5 * (5 / 120) + 5 * (5 / 120))
noncomputable def t_minus_s : ℝ := t - s

theorem t_minus_s_equals_neg_17_25 : t_minus_s = -17.25 := by
  sorry

end t_minus_s_equals_neg_17_25_l23_23592


namespace tenth_term_of_arithmetic_sequence_l23_23146

theorem tenth_term_of_arithmetic_sequence :
  ∃ a : ℕ → ℤ, (∀ n : ℕ, a n + 1 - a n = 2) ∧ a 1 = 1 ∧ a 10 = 19 :=
sorry

end tenth_term_of_arithmetic_sequence_l23_23146


namespace range_of_a_l23_23950

variable {α : Type} [LinearOrderedField α]

def A (a : α) : Set α := {x | |x - a| ≤ 1}

def B : Set α := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : α) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l23_23950


namespace max_handshakes_l23_23058

theorem max_handshakes (total_people : ℕ) (groupA groupB groupC : ℕ)
  (constraint1 : ∀ {p1 p2 : ℕ}, p1 ≠ p2 → p1 < total_people ∧ p2 < total_people → true)
  (constraint2 : ∀ p : ℕ, ¬ (p < total_people ∧ p = p))
  (constraint3 : groupA + groupB + groupC = total_people ∧ groupA = 30 ∧ groupB = 35 ∧ groupC = 35) :
   ∑ x in finset.range(total_people), ∑ y in finset.Ico(x+1, total_people), (1 : ℕ) - (
      if x < groupA ∧ y < groupA then 1
      else if x ≥ groupA ∧ x < groupA + groupB ∧ y ≥ groupA ∧ y < groupA + groupB then 1
      else if x ≥ groupA + groupB ∧ y ≥ groupA + groupB then 1
      else 0
    ) = 3325 :=
by
  sorry

end max_handshakes_l23_23058


namespace remainder_when_dividing_by_2x_minus_4_l23_23901

def f (x : ℝ) := 4 * x^3 - 9 * x^2 + 12 * x - 14
def g (x : ℝ) := 2 * x - 4

theorem remainder_when_dividing_by_2x_minus_4 : f 2 = 6 := by
  sorry

end remainder_when_dividing_by_2x_minus_4_l23_23901


namespace olivia_grocery_cost_l23_23522

theorem olivia_grocery_cost :
  let cost_bananas := 12
  let cost_bread := 9
  let cost_milk := 7
  let cost_apples := 14
  cost_bananas + cost_bread + cost_milk + cost_apples = 42 :=
by
  rfl

end olivia_grocery_cost_l23_23522


namespace neg_p_exists_x_l23_23813

-- Let p be the proposition: For all x in ℝ, x^2 - 3x + 3 > 0
def p : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 > 0

-- Prove that the negation of p implies that there exists some x in ℝ such that x^2 - 3x + 3 ≤ 0
theorem neg_p_exists_x : ¬p ↔ ∃ x : ℝ, x^2 - 3 * x + 3 ≤ 0 :=
by {
  sorry
}

end neg_p_exists_x_l23_23813


namespace initial_distance_planes_l23_23377

theorem initial_distance_planes (speed_A speed_B : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_A distance_B : ℝ) (total_distance : ℝ) :
  speed_A = 240 ∧ speed_B = 360 ∧ time_seconds = 72000 ∧ time_hours = 20 ∧ 
  time_hours = time_seconds / 3600 ∧
  distance_A = speed_A * time_hours ∧ 
  distance_B = speed_B * time_hours ∧ 
  total_distance = distance_A + distance_B →
  total_distance = 12000 :=
by
  intros
  sorry

end initial_distance_planes_l23_23377


namespace closest_perfect_square_multiple_of_4_l23_23725

theorem closest_perfect_square_multiple_of_4 (n : ℕ) (h1 : ∃ k : ℕ, k^2 = n) (h2 : n % 4 = 0) : n = 324 := by
  -- Define 350 as the target
  let target := 350

  -- Conditions
  have cond1 : ∃ k : ℕ, k^2 = n := h1
  
  have cond2 : n % 4 = 0 := h2

  -- Check possible values meeting conditions
  by_cases h : n = 324
  { exact h }
  
  -- Exclude non-multiples of 4 and perfect squares further away from 350
  sorry

end closest_perfect_square_multiple_of_4_l23_23725


namespace product_price_interval_l23_23847

def is_too_high (price guess : ℕ) : Prop := guess > price
def is_too_low  (price guess : ℕ) : Prop := guess < price

theorem product_price_interval 
    (price : ℕ)
    (h1 : is_too_high price 2000)
    (h2 : is_too_low price 1000)
    (h3 : is_too_high price 1500)
    (h4 : is_too_low price 1250)
    (h5 : is_too_low price 1375) :
    1375 < price ∧ price < 1500 :=
    sorry

end product_price_interval_l23_23847


namespace intersection_sets_l23_23291

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l23_23291


namespace domain_of_f_l23_23017

-- The domain of the function is the set of all x such that the function is defined.
theorem domain_of_f:
  {x : ℝ | x > 3 ∧ x ≠ 4} = (Set.Ioo 3 4 ∪ Set.Ioi 4) := 
sorry

end domain_of_f_l23_23017


namespace cone_section_volume_ratio_l23_23447

theorem cone_section_volume_ratio :
  ∀ (r h : ℝ), (h > 0 ∧ r > 0) →
  let V1 := ((75 / 3) * π * r^2 * h - (64 / 3) * π * r^2 * h)
  let V2 := ((64 / 3) * π * r^2 * h - (27 / 3) * π * r^2 * h)
  V2 / V1 = 37 / 11 :=
by
  intros r h h_pos
  sorry

end cone_section_volume_ratio_l23_23447


namespace friend_initial_money_l23_23907

theorem friend_initial_money (F : ℕ) : 
    (160 + 25 * 7 = F + 25 * 5) → 
    (F = 210) :=
by
  sorry

end friend_initial_money_l23_23907


namespace range_of_independent_variable_l23_23883

theorem range_of_independent_variable (x : ℝ) : x ≠ -3 ↔ ∃ y : ℝ, y = 1 / (x + 3) :=
by 
  -- Proof is omitted
  sorry

end range_of_independent_variable_l23_23883


namespace rope_length_before_folding_l23_23202

theorem rope_length_before_folding (L : ℝ) (h : L / 4 = 10) : L = 40 :=
by
  sorry

end rope_length_before_folding_l23_23202


namespace sequence_equality_l23_23540

theorem sequence_equality (a : Fin 1973 → ℝ) (hpos : ∀ n, a n > 0)
  (heq : a 0 ^ a 0 = a 1 ^ a 2 ∧ a 1 ^ a 2 = a 2 ^ a 3 ∧ 
         a 2 ^ a 3 = a 3 ^ a 4 ∧ 
         -- etc., continued for all indices, 
         -- ensuring last index correctly refers back to a 0
         a 1971 ^ a 1972 = a 1972 ^ a 0) :
  a 0 = a 1972 :=
sorry

end sequence_equality_l23_23540


namespace find_x_l23_23810

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 1)

theorem find_x (x : ℝ) (h : deriv f x = x) : x = 0 ∨ x = 1 :=
by
  sorry

end find_x_l23_23810


namespace central_symmetry_preserves_distance_l23_23867

variables {Point : Type} [MetricSpace Point]

def central_symmetry (O A A' B B' : Point) : Prop :=
  dist O A = dist O A' ∧ dist O B = dist O B'

theorem central_symmetry_preserves_distance {O A A' B B' : Point}
  (h : central_symmetry O A A' B B') : dist A B = dist A' B' :=
sorry

end central_symmetry_preserves_distance_l23_23867


namespace D_coordinates_l23_23956

namespace Parallelogram

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 2 }
def C : Point := { x := 3, y := 1 }

theorem D_coordinates :
  ∃ D : Point, D = { x := 2, y := -1 } ∧ ∀ A B C D : Point, 
    (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) := by
  sorry

end Parallelogram

end D_coordinates_l23_23956


namespace mean_of_five_numbers_l23_23216

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l23_23216


namespace positive_difference_of_squares_l23_23407

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end positive_difference_of_squares_l23_23407


namespace identify_clothing_l23_23775

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l23_23775


namespace radius_of_cone_base_l23_23594

theorem radius_of_cone_base {R : ℝ} {theta : ℝ} (hR : R = 6) (htheta : theta = 120) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_cone_base_l23_23594


namespace probability_first_king_spades_second_spade_l23_23080

noncomputable def deck := fin 52

def is_king_of_spades (c: deck) : Prop := c = fin.mk 0 (by norm_num)
def is_spade (c: deck) : Prop := c.val / 13 = 0  -- first 13 cards are Spades

-- Define event that first card is King of Spades and second card is Spade
def event (first second : deck) : Prop :=
  is_king_of_spades first ∧ 
  is_spade second 

theorem probability_first_king_spades_second_spade :
  (∃ (first second : deck), event first second) →
  (Pr (event first second) = 1 / 221) :=
sorry

end probability_first_king_spades_second_spade_l23_23080


namespace steve_ate_bags_l23_23337

-- Given conditions
def total_macaroons : Nat := 12
def weight_per_macaroon : Nat := 5
def num_bags : Nat := 4
def total_weight_remaining : Nat := 45

-- Derived conditions
def total_weight_macaroons : Nat := total_macaroons * weight_per_macaroon
def macaroons_per_bag : Nat := total_macaroons / num_bags
def weight_per_bag : Nat := macaroons_per_bag * weight_per_macaroon
def bags_remaining : Nat := total_weight_remaining / weight_per_bag

-- Proof statement
theorem steve_ate_bags : num_bags - bags_remaining = 1 := by
  sorry

end steve_ate_bags_l23_23337


namespace reciprocal_of_36_recurring_decimal_l23_23567

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l23_23567


namespace smallest_value_3a_2_l23_23325

theorem smallest_value_3a_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 2 = - (5 / 2) := sorry

end smallest_value_3a_2_l23_23325


namespace height_at_end_of_2_years_l23_23451

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end height_at_end_of_2_years_l23_23451


namespace no_solution_system_iff_n_eq_neg_cbrt_four_l23_23151

variable (n : ℝ)

theorem no_solution_system_iff_n_eq_neg_cbrt_four :
    (∀ x y z : ℝ, ¬ (2 * n * x + 3 * y = 2 ∧ 3 * n * y + 4 * z = 3 ∧ 4 * x + 2 * n * z = 4)) ↔
    n = - (4 : ℝ)^(1/3) := 
by
  sorry

end no_solution_system_iff_n_eq_neg_cbrt_four_l23_23151


namespace additional_chicken_wings_l23_23586

theorem additional_chicken_wings (friends : ℕ) (wings_per_friend : ℕ) (initial_wings : ℕ) (H1 : friends = 9) (H2 : wings_per_friend = 3) (H3 : initial_wings = 2) : 
  friends * wings_per_friend - initial_wings = 25 := by
  sorry

end additional_chicken_wings_l23_23586


namespace translate_parabola_l23_23229

theorem translate_parabola (x : ℝ) :
  (x^2 + 3) = (x - 5)^2 + 3 :=
sorry

end translate_parabola_l23_23229


namespace quadratic_inequality_solution_l23_23030

theorem quadratic_inequality_solution (b c : ℝ) 
    (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + b * x + c < 0) :
    b + c = -1 :=
sorry

end quadratic_inequality_solution_l23_23030


namespace systematic_sampling_probabilities_l23_23661

-- Define the total number of students
def total_students : ℕ := 1005

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of individuals removed
def individuals_removed : ℕ := 5

-- Define the probability of an individual being removed
def probability_removed : ℚ := individuals_removed / total_students

-- Define the probability of an individual being selected in the sample
def probability_selected : ℚ := sample_size / total_students

-- The statement we need to prove
theorem systematic_sampling_probabilities :
  probability_removed = 5 / 1005 ∧ probability_selected = 50 / 1005 :=
sorry

end systematic_sampling_probabilities_l23_23661


namespace bread_cost_l23_23895

theorem bread_cost (H C B : ℕ) (h₁ : H = 150) (h₂ : C = 200) (h₃ : H + B = C) : B = 50 :=
by
  sorry

end bread_cost_l23_23895


namespace calculate_expression_l23_23262

theorem calculate_expression :
  -15 - 21 + 8 = -28 :=
by
  sorry

end calculate_expression_l23_23262


namespace no_integer_solutions_for_square_polynomial_l23_23674

theorem no_integer_solutions_for_square_polynomial :
  (∀ x : ℤ, ∃ k : ℤ, k^2 = x^4 + 5*x^3 + 10*x^2 + 5*x + 25 → false) :=
by
  sorry

end no_integer_solutions_for_square_polynomial_l23_23674


namespace num_customers_left_more_than_remaining_l23_23255

theorem num_customers_left_more_than_remaining (initial remaining : ℕ) (h : initial = 11 ∧ remaining = 3) : (initial - remaining) = (remaining + 5) :=
by sorry

end num_customers_left_more_than_remaining_l23_23255


namespace fruit_seller_apples_l23_23429

theorem fruit_seller_apples (original_apples : ℝ) (sold_percent : ℝ) (remaining_apples : ℝ)
  (h1 : sold_percent = 0.40)
  (h2 : remaining_apples = 420)
  (h3 : original_apples * (1 - sold_percent) = remaining_apples) :
  original_apples = 700 :=
by
  sorry

end fruit_seller_apples_l23_23429


namespace time_to_pass_jogger_l23_23070

noncomputable def jogger_speed_kmh := 9 -- in km/hr
noncomputable def train_speed_kmh := 45 -- in km/hr
noncomputable def jogger_headstart_m := 240 -- in meters
noncomputable def train_length_m := 100 -- in meters

noncomputable def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_mps := kmh_to_mps jogger_speed_kmh
noncomputable def train_speed_mps := kmh_to_mps train_speed_kmh
noncomputable def relative_speed := train_speed_mps - jogger_speed_mps
noncomputable def distance_to_be_covered := jogger_headstart_m + train_length_m

theorem time_to_pass_jogger : distance_to_be_covered / relative_speed = 34 := by
  sorry

end time_to_pass_jogger_l23_23070


namespace contractor_absent_days_l23_23428

-- Definition of problem conditions
def total_days : ℕ := 30
def daily_wage : ℝ := 25
def daily_fine : ℝ := 7.5
def total_amount_received : ℝ := 620

-- Function to define the constraint equations
def equation1 (x y : ℕ) : Prop := x + y = total_days
def equation2 (x y : ℕ) : Prop := (daily_wage * x - daily_fine * y) = total_amount_received

-- The proof problem translation as Lean 4 statement
theorem contractor_absent_days (x y : ℕ) (h1 : equation1 x y) (h2 : equation2 x y) : y = 8 :=
by
  sorry

end contractor_absent_days_l23_23428


namespace friends_count_is_four_l23_23186

def number_of_friends (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) : ℕ :=
  4

theorem friends_count_is_four (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) (h1 : total_cards = 12) :
  number_of_friends Melanie Benny Sally Jessica total_cards = 4 :=
by
  sorry

end friends_count_is_four_l23_23186


namespace factorization_correct_l23_23610

-- Define the polynomial we are working with
def polynomial := ∀ x : ℝ, x^3 - 6 * x^2 + 9 * x

-- Define the factorized form of the polynomial
def factorized_polynomial := ∀ x : ℝ, x * (x - 3)^2

-- State the theorem that proves the polynomial equals its factorized form
theorem factorization_correct (x : ℝ) : polynomial x = factorized_polynomial x :=
by
  sorry

end factorization_correct_l23_23610


namespace correct_calculation_result_l23_23539

theorem correct_calculation_result :
  ∀ (A B D : ℝ),
  C = 6 →
  E = 5 →
  (A * 10 + B) * 6 + D * E = 39.6 ∨ (A * 10 + B) * 6 * D * E = 36.9 →
  (A * 10 + B) * 6 + D * E = 26.1 :=
by
  intros A B D C_eq E_eq errors
  sorry

end correct_calculation_result_l23_23539


namespace constant_term_expansion_l23_23388

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) : 
  let term (r : ℕ) : ℝ := (1 / 2) ^ (9 - r) * (-1) ^ r * Nat.choose 9 r * x ^ (3 / 2 * r - 9)
  term 6 = 21 / 2 :=
by
  sorry

end constant_term_expansion_l23_23388


namespace num_two_digit_powers_of_3_l23_23829

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l23_23829


namespace boxes_total_is_correct_l23_23227

def initial_boxes : ℕ := 7
def additional_boxes_per_box : ℕ := 7
def final_non_empty_boxes : ℕ := 10
def total_boxes := 77

theorem boxes_total_is_correct
  (h1 : initial_boxes = 7)
  (h2 : additional_boxes_per_box = 7)
  (h3 : final_non_empty_boxes = 10)
  : total_boxes = 77 :=
by
  -- Proof goes here
  sorry

end boxes_total_is_correct_l23_23227


namespace correct_operations_result_l23_23995

/-
Pat intended to multiply a number by 8 but accidentally divided by 8.
Pat then meant to add 20 to the result but instead subtracted 20.
After these errors, the final outcome was 12.
Prove that if Pat had performed the correct operations, the final outcome would have been 2068.
-/

theorem correct_operations_result (n : ℕ) (h1 : n / 8 - 20 = 12) : 8 * n + 20 = 2068 :=
by
  sorry

end correct_operations_result_l23_23995


namespace cubic_sum_l23_23155

theorem cubic_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 14) : x ^ 3 + y ^ 3 = 580 :=
by 
  sorry

end cubic_sum_l23_23155


namespace least_positive_integer_with_eight_factors_l23_23417

theorem least_positive_integer_with_eight_factors :
  ∃ n : ℕ, n = 24 ∧ (8 = (nat.factors n).length) :=
sorry

end least_positive_integer_with_eight_factors_l23_23417


namespace fruit_prices_l23_23916

theorem fruit_prices :
  (∃ x y : ℝ, 60 * x + 40 * y = 1520 ∧ 30 * x + 50 * y = 1360 ∧ x = 12 ∧ y = 20) :=
sorry

end fruit_prices_l23_23916


namespace primary_college_employee_relation_l23_23969

theorem primary_college_employee_relation
  (P C N : ℕ)
  (hN : N = 20 + P + C)
  (h_illiterate_wages_before : 20 * 25 = 500)
  (h_illiterate_wages_after : 20 * 10 = 200)
  (h_primary_wages_before : P * 40 = P * 40)
  (h_primary_wages_after : P * 25 = P * 25)
  (h_college_wages_before : C * 50 = C * 50)
  (h_college_wages_after : C * 60 = C * 60)
  (h_avg_decrease : (500 + 40 * P + 50 * C) / N - (200 + 25 * P + 60 * C) / N = 10) :
  15 * P - 10 * C = 10 * N - 300 := 
by
  sorry

end primary_college_employee_relation_l23_23969


namespace two_digit_powers_of_3_count_l23_23822

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l23_23822


namespace max_value_expr_l23_23272

-- Define the expression
def expr (a b c d : ℝ) : ℝ :=
  a + b + c + d - a * b - b * c - c * d - d * a

-- The main theorem
theorem max_value_expr :
  (∀ (a b c d : ℝ), 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → expr a b c d ≤ 2) ∧
  (∃ (a b c d : ℝ), 0 ≤ a ∧ a = 1 ∧ 0 ≤ b ∧ b = 0 ∧ 0 ≤ c ∧ c = 1 ∧ 0 ≤ d ∧ d = 0 ∧ expr a b c d = 2) :=
  by
  sorry

end max_value_expr_l23_23272


namespace least_integer_with_exactly_eight_factors_l23_23420

theorem least_integer_with_exactly_eight_factors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d = 1 ∨ d = 2 ∨ d = 3 ∨ d
= 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) → m = n) :=
begin
  sorry
end

end least_integer_with_exactly_eight_factors_l23_23420


namespace garden_area_l23_23459

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end garden_area_l23_23459


namespace incorrect_expressions_l23_23981

-- Definitions for the conditions
def F : ℝ := sorry   -- F represents a repeating decimal
def X : ℝ := sorry   -- X represents the t digits of F that are non-repeating
def Y : ℝ := sorry   -- Y represents the u digits of F that repeat
def t : ℕ := sorry   -- t is the number of non-repeating digits
def u : ℕ := sorry   -- u is the number of repeating digits

-- Statement that expressions (C) and (D) are incorrect
theorem incorrect_expressions : 
  ¬ (10^(t + 2 * u) * F = X + Y / 10 ^ u) ∧ ¬ (10^t * (10^u - 1) * F = Y * (X - 1)) :=
sorry

end incorrect_expressions_l23_23981


namespace ratio_arithmetic_geometric_mean_l23_23110

/-- Let a and b be positive real numbers. Given the ratio of the arithmetic mean to
  the geometric mean of a and b is 25:24, prove that the ratio of a to b is 16:9 or 9:16. -/
theorem ratio_arithmetic_geometric_mean (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a + b) / (2 * real.sqrt (a * b)) = 25 / 24) : 
  a / b = 16 / 9 ∨ a / b  = 9 / 16 :=
by
  sorry

end ratio_arithmetic_geometric_mean_l23_23110


namespace geometric_sequence_ratios_l23_23802

theorem geometric_sequence_ratios {n : ℕ} {r : ℝ}
  (h1 : 85 = (1 - r^(2*n)) / (1 - r^2))
  (h2 : 170 = r * 85) :
  r = 2 ∧ 2*n = 8 :=
by
  sorry

end geometric_sequence_ratios_l23_23802


namespace find_prime_p_l23_23938

open Int

theorem find_prime_p (p k m n : ℕ) (hp : Nat.Prime p) 
  (hk : 0 < k) (hm : 0 < m)
  (h_eq : (mk^2 + 2 : ℤ) * p - (m^2 + 2 * k^2 : ℤ) = n^2 * (mp + 2 : ℤ)) :
  p = 3 ∨ p = 1 := sorry

end find_prime_p_l23_23938


namespace simplify_sqrt_product_l23_23870

theorem simplify_sqrt_product (y : ℝ) (hy : y > 0) : 
  (Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y)) :=
by
  sorry

end simplify_sqrt_product_l23_23870


namespace find_number_l23_23194

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l23_23194


namespace impossible_divide_into_three_similar_l23_23351

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l23_23351


namespace tan_theta_value_l23_23963

theorem tan_theta_value (θ : ℝ) (h1 : Real.sin θ = 3/5) (h2 : Real.cos θ = -4/5) : 
  Real.tan θ = -3/4 :=
  sorry

end tan_theta_value_l23_23963


namespace rahul_work_days_l23_23001

theorem rahul_work_days
  (R : ℕ)
  (Rajesh_days : ℕ := 2)
  (total_payment : ℕ := 170)
  (rahul_share : ℕ := 68)
  (combined_work_rate : ℚ := 1) :
  (∃ R : ℕ, (1 / (R : ℚ) + 1 / (Rajesh_days : ℚ) = combined_work_rate) ∧ (68 / (total_payment - rahul_share) = 2 / R) ∧ R = 3) :=
sorry

end rahul_work_days_l23_23001


namespace max_abs_cubic_at_least_one_fourth_l23_23180

def cubic_polynomial (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem max_abs_cubic_at_least_one_fourth (p q r : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |cubic_polynomial p q r x| ≥ 1 / 4 :=
by
  sorry

end max_abs_cubic_at_least_one_fourth_l23_23180


namespace interval_f_has_two_roots_l23_23316

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

theorem interval_f_has_two_roots (a : ℝ) : (∀ x : ℝ, f a x = 0 → ∃ u v : ℝ, u ≠ v ∧ f a u = 0 ∧ f a v = 0) ↔ 0 < a ∧ a < 1 / 8 := 
sorry

end interval_f_has_two_roots_l23_23316


namespace abs_sum_inequality_l23_23277

theorem abs_sum_inequality (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := 
sorry

end abs_sum_inequality_l23_23277


namespace identify_clothing_l23_23773

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l23_23773


namespace dawn_hours_l23_23232

-- Define the conditions
def pedestrian_walked_from_A_to_B (x : ℕ) : Prop :=
  x > 0

def pedestrian_walked_from_B_to_A (x : ℕ) : Prop :=
  x > 0

def met_at_noon (x : ℕ) : Prop :=
  x > 0

def arrived_at_B_at_4pm (x : ℕ) : Prop :=
  x > 0

def arrived_at_A_at_9pm (x : ℕ) : Prop :=
  x > 0

-- Define the theorem to prove
theorem dawn_hours (x : ℕ) :
  pedestrian_walked_from_A_to_B x ∧ 
  pedestrian_walked_from_B_to_A x ∧
  met_at_noon x ∧ 
  arrived_at_B_at_4pm x ∧ 
  arrived_at_A_at_9pm x → 
  x = 6 := 
sorry

end dawn_hours_l23_23232


namespace factorization_correct_l23_23118

theorem factorization_correct (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end factorization_correct_l23_23118


namespace salt_weight_l23_23912

theorem salt_weight {S : ℝ} (h1 : 16 + S = 46) : S = 30 :=
by
  sorry

end salt_weight_l23_23912


namespace height_at_2_years_l23_23452

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end height_at_2_years_l23_23452


namespace find_f_of_one_l23_23477

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_of_one : f 1 = 2 := 
by
  sorry

end find_f_of_one_l23_23477


namespace impossible_divide_into_three_similar_parts_l23_23347

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l23_23347


namespace cos_value_in_second_quadrant_l23_23311

variable (a : ℝ)
variables (h1 : π/2 < a ∧ a < π) (h2 : Real.sin a = 5/13)

theorem cos_value_in_second_quadrant : Real.cos a = -12/13 :=
  sorry

end cos_value_in_second_quadrant_l23_23311


namespace arc_length_l23_23016

theorem arc_length (circumference : ℝ) (angle_degrees : ℝ) (h : circumference = 90) (θ : angle_degrees = 45) :
  (angle_degrees / 360) * circumference = 11.25 := 
  by 
    sorry

end arc_length_l23_23016


namespace box_breadth_l23_23590

noncomputable def cm_to_m (cm : ℕ) : ℝ := cm / 100

theorem box_breadth :
  ∀ (length depth cm cubical_edge blocks : ℕ), 
    length = 160 →
    depth = 60 →
    cubical_edge = 20 →
    blocks = 120 →
    breadth = (blocks * (cubical_edge ^ 3)) / (length * depth) →
    breadth = 100 :=
by
  sorry

end box_breadth_l23_23590


namespace solve_quadratic_eq_l23_23732

theorem solve_quadratic_eq (x y z w d X Y Z W : ℤ) 
    (h1 : w % 2 = z % 2) 
    (h2 : x = 2 * d * (X * Z - Y * W))
    (h3 : y = 2 * d * (X * W + Y * Z))
    (h4 : z = d * (X^2 + Y^2 - Z^2 - W^2))
    (h5 : w = d * (X^2 + Y^2 + Z^2 + W^2)) :
    x^2 + y^2 + z^2 = w^2 :=
sorry

end solve_quadratic_eq_l23_23732


namespace discount_equation_l23_23113

theorem discount_equation (x : ℝ) : 280 * (1 - x) ^ 2 = 177 := 
by 
  sorry

end discount_equation_l23_23113


namespace initial_roses_l23_23441

theorem initial_roses (x : ℕ) (h1 : x - 3 + 34 = 36) : x = 5 :=
by 
  sorry

end initial_roses_l23_23441


namespace xy_product_l23_23546

theorem xy_product (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) :
  x = y * z ∨ y = x * z := 
by
  sorry

end xy_product_l23_23546


namespace find_natural_number_l23_23273

theorem find_natural_number (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 95 = k^2) : n = 5 ∨ n = 14 := by
  sorry

end find_natural_number_l23_23273


namespace circumcenter_distance_two_l23_23165

noncomputable def distance_between_circumcenter (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1)
  : ℝ :=
dist ( ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 ) ) ( ( (B.1 + C.1) / 2, (B.2 + C.2) / 2 )) 

theorem circumcenter_distance_two (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1) 
  : distance_between_circumcenter A B C M hAB hBC hAC hM_on_AC hCM = 2 :=
sorry

end circumcenter_distance_two_l23_23165


namespace power_difference_of_squares_l23_23071

theorem power_difference_of_squares : (((7^2 - 3^2) : ℤ)^4) = 2560000 := by
  sorry

end power_difference_of_squares_l23_23071


namespace no_arithmetic_sqrt_of_neg_real_l23_23204

theorem no_arithmetic_sqrt_of_neg_real (x : ℝ) (h : x < 0) : ¬ ∃ y : ℝ, y * y = x :=
by
  sorry

end no_arithmetic_sqrt_of_neg_real_l23_23204


namespace rosie_pies_from_apples_l23_23691

-- Given conditions
def piesPerDozen : ℕ := 3
def baseApples : ℕ := 12
def apples : ℕ := 36

-- Define the main theorem to prove the question == answer
theorem rosie_pies_from_apples 
  (h : piesPerDozen / baseApples * apples = 9) : 
  36 / 12 * 3 = 9 :=
by
  exact h
  sorry

end rosie_pies_from_apples_l23_23691


namespace find_expression_max_value_min_value_l23_23958

namespace MathProblem

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := a * x^2 + a^2 * x + 2 * b - a^3

-- Hypotheses based on problem conditions
lemma a_neg (a b : ℝ) : a < 0 := sorry
lemma root_neg2 (a b : ℝ) : f a b (-2) = 0 := sorry
lemma root_6 (a b : ℝ) : f a b 6 = 0 := sorry

-- Proving the explicit expression for f(x)
theorem find_expression (a b : ℝ) (x : ℝ) : 
  a = -4 → 
  b = -8 → 
  f a b x = -4 * x^2 + 16 * x + 48 :=
sorry

-- Maximum value of f(x) on the interval [1, 10]
theorem max_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 2 = 64 :=
sorry

-- Minimum value of f(x) on the interval [1, 10]
theorem min_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 10 = -192 :=
sorry

end MathProblem

end find_expression_max_value_min_value_l23_23958


namespace asymptote_problem_l23_23393

-- Definitions for the problem
def r (x : ℝ) : ℝ := -3 * (x + 2) * (x - 1)
def s (x : ℝ) : ℝ := (x + 2) * (x - 4)

-- Assertion to prove
theorem asymptote_problem : r (-1) / s (-1) = 6 / 5 :=
by {
  -- This is where the proof would be carried out
  sorry
}

end asymptote_problem_l23_23393


namespace calculate_uphill_distance_l23_23581

noncomputable def uphill_speed : ℝ := 30
noncomputable def downhill_speed : ℝ := 40
noncomputable def downhill_distance : ℝ := 50
noncomputable def average_speed : ℝ := 32.73

theorem calculate_uphill_distance : ∃ d : ℝ, d = 99.86 ∧ 
  32.73 = (d + downhill_distance) / (d / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end calculate_uphill_distance_l23_23581


namespace right_triangle_third_side_l23_23803

theorem right_triangle_third_side (a b : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :
  c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2) :=
by 
  sorry

end right_triangle_third_side_l23_23803


namespace minimum_shots_to_hit_ship_l23_23722

def is_ship_hit (shots : Finset (Fin 7 × Fin 7)) : Prop :=
  -- Assuming the ship can be represented by any 4 consecutive points in a row
  ∀ r : Fin 7, ∃ c1 c2 c3 c4 : Fin 7, 
    (0 ≤ c1.1 ∧ c1.1 ≤ 6 ∧ c1.1 + 3 = c4.1) ∧
    (0 ≤ c2.1 ∧ c2.1 ≤ 6 ∧ c2.1 = c1.1 + 1) ∧
    (0 ≤ c3.1 ∧ c3.1 ≤ 6 ∧ c3.1 = c1.1 + 2) ∧
    (r, c1) ∈ shots ∧ (r, c2) ∈ shots ∧ (r, c3) ∈ shots ∧ (r, c4) ∈ shots

theorem minimum_shots_to_hit_ship : ∃ shots : Finset (Fin 7 × Fin 7), 
  shots.card = 12 ∧ is_ship_hit shots :=
by 
  sorry

end minimum_shots_to_hit_ship_l23_23722


namespace find_geometric_ratio_l23_23334

-- Definitions for the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def geometric_sequence (a1 a3 a4 : ℝ) (q : ℝ) : Prop :=
  a3 * a3 = a1 * a4 ∧ a3 = a1 * q ∧ a4 = a3 * q

-- Definition for the proof statement
theorem find_geometric_ratio (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hnz : ∀ n, a n ≠ 0)
  (hq : ∃ (q : ℝ), geometric_sequence (a 0) (a 2) (a 3) q) :
  ∃ q, q = 1 ∨ q = 1 / 2 := sorry

end find_geometric_ratio_l23_23334


namespace prove_ordered_pair_l23_23174

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem prove_ordered_pair (h1 : p 0 = -24) (h2 : q 0 = 30) (h3 : ∀ x : ℝ, p (q x) = q (p x)) : (p 3, q 6) = (3, -24) := 
sorry

end prove_ordered_pair_l23_23174


namespace best_fitting_model_is_model_2_l23_23549

-- Variables representing the correlation coefficients of the four models
def R2_model_1 : ℝ := 0.86
def R2_model_2 : ℝ := 0.96
def R2_model_3 : ℝ := 0.73
def R2_model_4 : ℝ := 0.66

-- Statement asserting that Model 2 has the best fitting effect
theorem best_fitting_model_is_model_2 :
  R2_model_2 = 0.96 ∧ R2_model_2 > R2_model_1 ∧ R2_model_2 > R2_model_3 ∧ R2_model_2 > R2_model_4 :=
by {
  sorry
}

end best_fitting_model_is_model_2_l23_23549


namespace problem_1_problem_2_l23_23957

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem problem_1 (x : ℝ) : (g x ≥ abs (x - 1)) ↔ (x ≥ 2/3) :=
by
  sorry

theorem problem_2 (c : ℝ) : (∀ x, abs (g x) - c ≥ abs (x - 1)) → (c ≤ -1/2) :=
by
  sorry

end problem_1_problem_2_l23_23957


namespace expand_and_simplify_l23_23935

-- Define the two polynomials P and Q.
def P (x : ℝ) := 5 * x + 3
def Q (x : ℝ) := 2 * x^2 - x + 4

-- State the theorem we want to prove.
theorem expand_and_simplify (x : ℝ) : (P x * Q x) = 10 * x^3 + x^2 + 17 * x + 12 := 
by
  sorry

end expand_and_simplify_l23_23935


namespace f_2011_l23_23930

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_defined_segment : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_2011 : f 2011 = -2 := by
  sorry

end f_2011_l23_23930


namespace tournament_players_l23_23203

theorem tournament_players (n : ℕ) (h : n * (n - 1) / 2 = 56) : n = 14 :=
sorry

end tournament_players_l23_23203


namespace energy_difference_l23_23465

noncomputable def initial_energy (k q d : ℝ) : ℝ :=
  2 * (k * q^2 / d) + (k * q^2 / (2 * d))

noncomputable def new_energy (k q d : ℝ) : ℝ :=
  3 * (k * q^2 / d)

theorem energy_difference (k q d : ℝ) (h₁ : initial_energy k q d = 18) :
  new_energy k q d - initial_energy k q d = 3.6 :=
by
  sorry

end energy_difference_l23_23465


namespace overlap_area_rhombus_l23_23104

noncomputable def area_of_overlap (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  1 / (Real.sin (α / 2))

theorem overlap_area_rhombus (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  area_of_overlap α hα = 1 / (Real.sin (α / 2)) :=
sorry

end overlap_area_rhombus_l23_23104


namespace julie_initial_savings_l23_23670

-- Definition of the simple interest condition
def simple_interest_condition (P : ℝ) : Prop :=
  575 = P * 0.04 * 5

-- Definition of the compound interest condition
def compound_interest_condition (P : ℝ) : Prop :=
  635 = P * ((1 + 0.05) ^ 5 - 1)

-- The final proof problem
theorem julie_initial_savings (P : ℝ) :
  simple_interest_condition P →
  compound_interest_condition P →
  2 * P = 5750 :=
by sorry

end julie_initial_savings_l23_23670


namespace pyramid_base_length_of_tangent_hemisphere_l23_23067

noncomputable def pyramid_base_side_length (radius height : ℝ) (tangent : ℝ → ℝ → Prop) : ℝ := sorry

theorem pyramid_base_length_of_tangent_hemisphere 
(r h : ℝ) (tangent : ℝ → ℝ → Prop) (tangent_property : ∀ x y, tangent x y → y = 0) 
(h_radius : r = 3) (h_height : h = 9) 
(tangent_conditions : tangent r h → tangent r h) : 
  pyramid_base_side_length r h tangent = 9 :=
sorry

end pyramid_base_length_of_tangent_hemisphere_l23_23067


namespace maximize_lower_houses_l23_23505

theorem maximize_lower_houses (x y : ℕ) 
    (h1 : x + 2 * y = 30)
    (h2 : 0 < y)
    (h3 : (∃ k, k = 112)) :
  ∃ x y, (x + 2 * y = 30) ∧ ((x * y)) = 112 :=
by
  sorry

end maximize_lower_houses_l23_23505


namespace calculate_total_feet_in_garden_l23_23891

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end calculate_total_feet_in_garden_l23_23891


namespace find_tangent_m_value_l23_23134

-- Define the line equation
def line (x m : ℝ) := 2*x + m

-- Define the curve equation
def curve (x : ℝ) := x * Real.log x

-- The tangent line condition
def isTangentAt (x0 m : ℝ) := deriv curve x0 = 2 ∧ curve x0 = line x0 m

-- Prove that the value of m for which the line y = 2x + m is tangent to the curve y = x ln x is -e
theorem find_tangent_m_value : ∃ m, isTangentAt Real.exp m ∧ m = -Real.exp :=
by 
  sorry

end find_tangent_m_value_l23_23134


namespace total_time_to_braid_hair_l23_23168

constant dancers : ℕ := 8
constant braidsPerDancer : ℕ := 5
constant secondsPerBraid : ℕ := 30
constant secondsPerMinute : ℕ := 60

theorem total_time_to_braid_hair : 
  (dancers * braidsPerDancer * secondsPerBraid) / secondsPerMinute = 20 := 
by
  sorry

end total_time_to_braid_hair_l23_23168


namespace single_elimination_matches_l23_23032

theorem single_elimination_matches (players byes : ℕ)
  (h1 : players = 100)
  (h2 : byes = 28) :
  (players - 1) = 99 :=
by
  -- The proof would go here if it were needed
  sorry

end single_elimination_matches_l23_23032


namespace vector_computation_l23_23101

def v1 : ℤ × ℤ := (3, -5)
def v2 : ℤ × ℤ := (2, -10)
def s1 : ℤ := 4
def s2 : ℤ := 3

theorem vector_computation : s1 • v1 - s2 • v2 = (6, 10) :=
  sorry

end vector_computation_l23_23101


namespace positive_difference_complementary_angles_l23_23402

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l23_23402


namespace identify_clothes_l23_23767

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l23_23767


namespace balcony_height_l23_23435

-- Definitions for conditions given in the problem

def final_position := 0 -- y, since the ball hits the ground
def initial_velocity := 5 -- v₀ in m/s
def time_elapsed := 3 -- t in seconds
def gravity := 10 -- g in m/s²

theorem balcony_height : 
  ∃ h₀ : ℝ, final_position = h₀ + initial_velocity * time_elapsed - (1/2) * gravity * time_elapsed^2 ∧ h₀ = 30 := 
by 
  sorry

end balcony_height_l23_23435


namespace perpendicular_lines_k_value_l23_23498

theorem perpendicular_lines_k_value :
  ∀ (k : ℝ), (∀ (x y : ℝ), x + 4 * y - 1 = 0) →
             (∀ (x y : ℝ), k * x + y + 2 = 0) →
             (-1 / 4 * -k = -1) →
             k = -4 :=
by
  intros k h1 h2 h3
  sorry

end perpendicular_lines_k_value_l23_23498


namespace right_triangle_area_l23_23869

open Real

theorem right_triangle_area
  (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a < 24)
  (h₃ : 24^2 + a^2 = (48 - a)^2) : 
  1/2 * 24 * a = 216 :=
by
  -- This is just a statement, the proof is omitted
  sorry

end right_triangle_area_l23_23869


namespace solve_first_train_length_l23_23233

noncomputable def first_train_length (time: ℝ) (speed1_kmh: ℝ) (speed2_kmh: ℝ) (length2: ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * 1000 / 3600
  let speed2_ms := speed2_kmh * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem solve_first_train_length :
  first_train_length 7.0752960452818945 80 65 165 = 120.28 :=
by
  simp [first_train_length]
  norm_num
  sorry

end solve_first_train_length_l23_23233


namespace min_value_x2_sub_xy_add_y2_l23_23106

/-- Given positive real numbers x and y such that x^2 + xy + 3y^2 = 10, 
prove that the minimum value of x^2 - xy + y^2 is 2. -/
theorem min_value_x2_sub_xy_add_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + x * y + 3 * y^2 = 10) : 
  ∃ (value : ℝ), value = x^2 - x * y + y^2 ∧ value = 2 := 
by 
  sorry

end min_value_x2_sub_xy_add_y2_l23_23106


namespace minimum_value_of_expression_l23_23799

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x * y / z + z * x / y + y * z / x) * (x / (y * z) + y / (z * x) + z / (x * y))

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  expression x y z ≥ 9 :=
sorry

end minimum_value_of_expression_l23_23799


namespace impossible_to_divide_three_similar_parts_l23_23354

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l23_23354


namespace sample_size_calculation_l23_23746

-- Definitions based on the conditions
def num_classes : ℕ := 40
def num_representatives_per_class : ℕ := 3

-- Theorem statement we aim to prove
theorem sample_size_calculation : num_classes * num_representatives_per_class = 120 :=
by
  sorry

end sample_size_calculation_l23_23746


namespace number_of_straight_A_students_l23_23926

-- Define the initial conditions and numbers
variables {x y : ℕ}

-- Define the initial student count and conditions on percentages
def initial_student_count := 25
def new_student_count := 7
def total_student_count := initial_student_count + new_student_count
def initial_percentage (x : ℕ) := (x : ℚ) / initial_student_count * 100
def new_percentage (x y : ℕ) := ((x + y : ℚ) / total_student_count) * 100

theorem number_of_straight_A_students
  (x y : ℕ)
  (h : initial_percentage x + 10 = new_percentage x y) :
  (x + y = 16) :=
sorry

end number_of_straight_A_students_l23_23926


namespace ratio_of_longer_side_to_square_l23_23448

theorem ratio_of_longer_side_to_square (s a b : ℝ) (h1 : a * b = 2 * s^2) (h2 : a = 2 * b) : a / s = 2 :=
by
  sorry

end ratio_of_longer_side_to_square_l23_23448


namespace mean_of_five_numbers_l23_23210

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l23_23210


namespace solve_for_x_l23_23667

theorem solve_for_x :
  (48 = 5 * x + 3) → x = 9 :=
by
  sorry

end solve_for_x_l23_23667


namespace sum_of_distinct_elements_not_square_l23_23797

open Set

noncomputable def setS : Set ℕ := { n | ∃ k : ℕ, n = 2^(2*k+1) }

theorem sum_of_distinct_elements_not_square (s : Finset ℕ) (hs: ∀ x ∈ s, x ∈ setS) :
  ¬∃ k : ℕ, s.sum id = k^2 :=
sorry

end sum_of_distinct_elements_not_square_l23_23797


namespace maria_compensation_l23_23184

theorem maria_compensation (d : ℝ) (h1 : d ≥ 1600000) : ∃ c, c = 1400000 :=
by
  have h2 : c = 1400000, from sorry
  exact ⟨c, h2⟩

end maria_compensation_l23_23184


namespace domain_of_function_l23_23877

theorem domain_of_function : 
  {x : ℝ | 2 - x ≥ 0 ∧ x - 1 > 0} = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by {
  sorry
}

end domain_of_function_l23_23877


namespace total_highlighters_correct_l23_23500

variable (y p b : ℕ)
variable (total_highlighters : ℕ)

def num_yellow_highlighters := 7
def num_pink_highlighters := num_yellow_highlighters + 7
def num_blue_highlighters := num_pink_highlighters + 5
def total_highlighters_in_drawer := num_yellow_highlighters + num_pink_highlighters + num_blue_highlighters

theorem total_highlighters_correct : 
  total_highlighters_in_drawer = 40 :=
sorry

end total_highlighters_correct_l23_23500


namespace polynomial_solution_l23_23123

theorem polynomial_solution (p : ℝ → ℝ) (h : ∀ x, p (p x) = x * (p x) ^ 2 + x ^ 3) : 
  p = id :=
by {
    sorry
}

end polynomial_solution_l23_23123


namespace probability_all_dice_same_l23_23038

/--
Given four eight-sided dice, each numbered from 1 to 8 and each die landing independently,
prove that the probability of all four dice showing the same number is 1/512.
-/
theorem probability_all_dice_same :
  let n := 8 in       -- Number of sides on each dice
  let total_outcomes := n * n * n * n in  -- Total possible outcomes for four dice
  let favorable_outcomes := n in          -- Favorable outcomes (one same number for all dice)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 512 :=
by
  sorry

end probability_all_dice_same_l23_23038


namespace largest_number_of_hcf_lcm_l23_23056

theorem largest_number_of_hcf_lcm (HCF : ℕ) (factor1 factor2 : ℕ) (n1 n2 : ℕ) (largest : ℕ) 
  (h1 : HCF = 52) 
  (h2 : factor1 = 11) 
  (h3 : factor2 = 12) 
  (h4 : n1 = HCF * factor1) 
  (h5 : n2 = HCF * factor2) 
  (h6 : largest = max n1 n2) : 
  largest = 624 := 
by 
  sorry

end largest_number_of_hcf_lcm_l23_23056


namespace max_price_of_most_expensive_product_l23_23238

noncomputable def greatest_possible_price
  (num_products : ℕ)
  (avg_price : ℕ)
  (min_price : ℕ)
  (mid_price : ℕ)
  (higher_price_count : ℕ)
  (total_retail_price : ℕ)
  (least_expensive_total_price : ℕ)
  (remaining_price : ℕ)
  (less_expensive_total_price : ℕ) : ℕ :=
  total_retail_price - least_expensive_total_price - less_expensive_total_price

theorem max_price_of_most_expensive_product :
  greatest_possible_price 20 1200 400 1000 10 (20 * 1200) (10 * 400) (20 * 1200 - 10 * 400) (9 * 1000) = 11000 :=
by
  sorry

end max_price_of_most_expensive_product_l23_23238


namespace find_xyz_ratio_l23_23947

theorem find_xyz_ratio (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 2) 
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 :=
by sorry

end find_xyz_ratio_l23_23947


namespace smallest_sum_BB_b_l23_23645

theorem smallest_sum_BB_b (B b : ℕ) (hB : 1 ≤ B ∧ B ≤ 4) (hb : b > 6) (h : 31 * B = 4 * b + 4) : B + b = 8 :=
sorry

end smallest_sum_BB_b_l23_23645


namespace normal_level_shortage_l23_23455

variable (T : ℝ) (normal_capacity : ℝ) (end_of_month_reservoir : ℝ)
variable (h1 : end_of_month_reservoir = 6)
variable (h2 : end_of_month_reservoir = 2 * normal_capacity)
variable (h3 : end_of_month_reservoir = 0.60 * T)

theorem normal_level_shortage :
  normal_capacity = 7 :=
by
  sorry

end normal_level_shortage_l23_23455


namespace initial_kittens_count_l23_23510

-- Let's define the initial conditions first.
def kittens_given_away : ℕ := 2
def kittens_remaining : ℕ := 6

-- The main theorem to prove the initial number of kittens.
theorem initial_kittens_count : (kittens_given_away + kittens_remaining) = 8 :=
by sorry

end initial_kittens_count_l23_23510


namespace find_sum_zero_l23_23313

open Complex

noncomputable def complex_numbers_satisfy (a1 a2 a3 : ℂ) : Prop :=
  a1^2 + a2^2 + a3^2 = 0 ∧
  a1^3 + a2^3 + a3^3 = 0 ∧
  a1^4 + a2^4 + a3^4 = 0

theorem find_sum_zero (a1 a2 a3 : ℂ) (h : complex_numbers_satisfy a1 a2 a3) :
  a1 + a2 + a3 = 0 :=
by {
  sorry
}

end find_sum_zero_l23_23313


namespace triangle_equilateral_l23_23034

variable {a b c : ℝ}

theorem triangle_equilateral (h : a^2 + 2 * b^2 = 2 * b * (a + c) - c^2) : a = b ∧ b = c := by
  sorry

end triangle_equilateral_l23_23034


namespace steve_book_sales_l23_23527

theorem steve_book_sales
  (copies_price : ℝ)
  (agent_rate : ℝ)
  (total_earnings : ℝ)
  (net_per_copy : ℝ := copies_price * (1 - agent_rate))
  (total_copies_sold : ℝ := total_earnings / net_per_copy) :
  copies_price = 2 → agent_rate = 0.10 → total_earnings = 1620000 → total_copies_sold = 900000 :=
by
  intros
  sorry

end steve_book_sales_l23_23527


namespace range_of_k_for_intersecting_circles_l23_23315

/-- Given circle \( C \) with equation \( x^2 + y^2 - 8x + 15 = 0 \) and a line \( y = kx - 2 \),
    prove that if there exists at least one point on the line such that a circle with this point
    as the center and a radius of 1 intersects with circle \( C \), then \( 0 \leq k \leq \frac{4}{3} \). -/
theorem range_of_k_for_intersecting_circles (k : ℝ) :
  (∃ (x y : ℝ), y = k * x - 2 ∧ (x - 4) ^ 2 + y ^ 2 - 1 ≤ 1) → 0 ≤ k ∧ k ≤ 4 / 3 :=
by {
  sorry
}

end range_of_k_for_intersecting_circles_l23_23315


namespace king_arthur_actual_weight_l23_23682

theorem king_arthur_actual_weight (K H E : ℤ) 
  (h1 : K + E = 19) 
  (h2 : H + E = 101) 
  (h3 : K + H + E = 114) : K = 13 := 
by 
  -- Introduction for proof to be skipped
  sorry

end king_arthur_actual_weight_l23_23682


namespace solve_inequality_l23_23011

theorem solve_inequality (x : ℝ) : 2 * x ^ 2 - 7 * x - 30 < 0 ↔ - (5 / 2) < x ∧ x < 6 := 
sorry

end solve_inequality_l23_23011


namespace base8_subtraction_correct_l23_23899

-- Define what it means to subtract in base 8
def base8_sub (a b : ℕ) : ℕ :=
  let a_base10 := 8 * (a / 10) + (a % 10)
  let b_base10 := 8 * (b / 10) + (b % 10)
  let result_base10 := a_base10 - b_base10
  8 * (result_base10 / 8) + (result_base10 % 8)

-- The given numbers in base 8
def num1 : ℕ := 52
def num2 : ℕ := 31
def expected_result : ℕ := 21

-- The proof problem statement
theorem base8_subtraction_correct : base8_sub num1 num2 = expected_result := by
  sorry

end base8_subtraction_correct_l23_23899


namespace sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l23_23794

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

theorem sec_225_eq_neg_sqrt2 :
  sec (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

theorem csc_225_eq_neg_sqrt2 :
  csc (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

end sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l23_23794


namespace solve_fractional_equation_l23_23709

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x + 1 ≠ 0) :
  (1 / x = 2 / (x + 1)) → x = 1 := 
by
  sorry

end solve_fractional_equation_l23_23709


namespace factorization_correct_l23_23119

noncomputable def factor_polynomial (x : ℝ) : ℝ := 4 * x^3 - 4 * x^2 + x

theorem factorization_correct (x : ℝ) : 
  factor_polynomial x = x * (2 * x - 1)^2 :=
by
  sorry

end factorization_correct_l23_23119


namespace cross_product_u_v_l23_23786

-- Define the vectors u and v
def u : ℝ × ℝ × ℝ := (3, -4, 7)
def v : ℝ × ℝ × ℝ := (2, 5, -3)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

-- State the theorem to be proved
theorem cross_product_u_v : cross_product u v = (-23, 23, 23) :=
  sorry

end cross_product_u_v_l23_23786


namespace length_PD_l23_23587

theorem length_PD (PA PB PC PD : ℝ) (hPA : PA = 5) (hPB : PB = 3) (hPC : PC = 4) :
  PD = 4 * Real.sqrt 2 :=
by
  sorry

end length_PD_l23_23587


namespace identify_clothing_l23_23772

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l23_23772


namespace solve_for_xy_l23_23970

-- The conditions given in the problem
variables (x y : ℝ)
axiom cond1 : 1 / 2 * x - y = 5
axiom cond2 : y - 1 / 3 * x = 2

-- The theorem we need to prove
theorem solve_for_xy (x y : ℝ) (cond1 : 1 / 2 * x - y = 5) (cond2 : y - 1 / 3 * x = 2) : 
  x = 42 ∧ y = 16 := sorry

end solve_for_xy_l23_23970


namespace simplify_expression_l23_23004

-- Define the constants.
def a : ℚ := 8
def b : ℚ := 27

-- Assuming cube root function is available and behaves as expected for rationals.
def cube_root (x : ℚ) : ℚ := x^(1/3 : ℚ)

-- Assume the necessary property of cube root of 27.
axiom cube_root_27_is_3 : cube_root 27 = 3

-- The main statement to prove.
theorem simplify_expression : cube_root (a + b) * cube_root (a + cube_root b) = cube_root 385 :=
by
  sorry

end simplify_expression_l23_23004


namespace money_constraints_l23_23115

variable (a b : ℝ)

theorem money_constraints (h1 : 8 * a - b = 98) (h2 : 2 * a + b > 36) : a > 13.4 ∧ b > 9.2 :=
sorry

end money_constraints_l23_23115


namespace speed_of_current_l23_23249

variable (c : ℚ) -- Speed of the current in miles per hour
variable (d : ℚ) -- Distance to the certain point in miles

def boat_speed := 16 -- Boat's speed relative to water in mph
def upstream_time := (20:ℚ) / 60 -- Time upstream in hours 
def downstream_time := (15:ℚ) / 60 -- Time downstream in hours

theorem speed_of_current (h1 : d = (boat_speed - c) * upstream_time)
                         (h2 : d = (boat_speed + c) * downstream_time) :
    c = 16 / 7 :=
  by
  sorry

end speed_of_current_l23_23249


namespace necessary_not_sufficient_condition_l23_23283

variable {x : ℝ}

theorem necessary_not_sufficient_condition (h : x > 2) : x > 1 :=
by
  sorry

end necessary_not_sufficient_condition_l23_23283


namespace simplify_expression_l23_23526

variables (a b : ℝ)
noncomputable def x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))

theorem simplify_expression (ha : a > 0) (hb : b > 0) :
  (2 * a * Real.sqrt (1 + x a b ^ 2)) / (x a b + Real.sqrt (1 + x a b ^ 2)) = a + b :=
sorry

end simplify_expression_l23_23526


namespace find_k_l23_23154

theorem find_k (k x y : ℝ) (h1 : x = 2) (h2 : y = -3)
    (h3 : 2 * x^2 + k * x * y = 4) : k = 2 / 3 :=
by
  sorry

end find_k_l23_23154


namespace identify_clothes_l23_23766

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l23_23766


namespace martha_total_clothes_l23_23372

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l23_23372


namespace Bill_trips_l23_23783

theorem Bill_trips (total_trips : ℕ) (Jean_trips : ℕ) (Bill_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : Jean_trips = 23) 
  (h3 : Bill_trips + Jean_trips = total_trips) : 
  Bill_trips = 17 := 
by
  sorry

end Bill_trips_l23_23783


namespace line_through_points_eq_l23_23643

theorem line_through_points_eq
  (x1 y1 x2 y2 : ℝ)
  (h1 : 2 * x1 + 3 * y1 = 4)
  (h2 : 2 * x2 + 3 * y2 = 4) :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) ↔ (2 * x + 3 * y = 4)) :=
by
  sorry

end line_through_points_eq_l23_23643


namespace exists_equal_sum_disjoint_subsets_l23_23000

-- Define the set and conditions
def is_valid_set (S : Finset ℕ) : Prop :=
  S.card = 15 ∧ ∀ x ∈ S, x ≤ 2020

-- Define the problem statement
theorem exists_equal_sum_disjoint_subsets (S : Finset ℕ) (h : is_valid_set S) :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end exists_equal_sum_disjoint_subsets_l23_23000


namespace original_number_of_girls_l23_23923

theorem original_number_of_girls (b g : ℕ) (h1 : b = g)
                                (h2 : 3 * (g - 25) = b)
                                (h3 : 6 * (b - 60) = g - 25) :
  g = 67 :=
by sorry

end original_number_of_girls_l23_23923


namespace complete_square_l23_23694

theorem complete_square (y : ℝ) : y^2 + 12 * y + 40 = (y + 6)^2 + 4 :=
by {
  sorry
}

end complete_square_l23_23694


namespace correct_operation_C_l23_23570

theorem correct_operation_C (m : ℕ) : m^7 / m^3 = m^4 := by
  sorry

end correct_operation_C_l23_23570


namespace average_age_before_new_students_l23_23387

theorem average_age_before_new_students
  (A : ℝ) (N : ℕ)
  (h1 : N = 15)
  (h2 : 15 * 32 + N * A = (N + 15) * (A - 4)) :
  A = 40 :=
by {
  sorry
}

end average_age_before_new_students_l23_23387


namespace calculate_expression_l23_23261

theorem calculate_expression : 7 + 15 / 3 - 5 * 2 = 2 :=
by sorry

end calculate_expression_l23_23261


namespace total_area_of_figure_l23_23264

theorem total_area_of_figure :
  let h := 7
  let w1 := 6
  let h1 := 2
  let h2 := 3
  let h3 := 1
  let w2 := 5
  let a1 := h * w1
  let a2 := (h - h1) * (11 - 7)
  let a3 := (h - h1 - h2) * (11 - 7)
  let a4 := (15 - 11) * h3
  a1 + a2 + a3 + a4 = 74 :=
by
  sorry

end total_area_of_figure_l23_23264


namespace same_parity_iff_exists_c_d_l23_23688

theorem same_parity_iff_exists_c_d (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a % 2 = b % 2) ↔ ∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2 := 
by 
  sorry

end same_parity_iff_exists_c_d_l23_23688


namespace clothing_discounted_to_fraction_of_original_price_l23_23437

-- Given conditions
variable (P : ℝ) (f : ℝ)

-- Price during first sale is fP, price during second sale is 0.5P
-- Price decreased by 40% from first sale to second sale
def price_decrease_condition : Prop :=
  f * P - (1/2) * P = 0.4 * (f * P)

-- The main theorem to prove
theorem clothing_discounted_to_fraction_of_original_price (h : price_decrease_condition P f) :
  f = 5/6 :=
sorry

end clothing_discounted_to_fraction_of_original_price_l23_23437


namespace anna_has_2_fewer_toys_than_amanda_l23_23086

-- Define the variables for the number of toys each person has
variables (A B : ℕ)

-- Define the conditions
def conditions (M : ℕ) : Prop :=
  M = 20 ∧ A = 3 * M ∧ A + M + B = 142

-- The theorem to prove
theorem anna_has_2_fewer_toys_than_amanda (M : ℕ) (h : conditions A B M) : B - A = 2 :=
sorry

end anna_has_2_fewer_toys_than_amanda_l23_23086


namespace clothes_color_proof_l23_23754

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l23_23754


namespace acute_triangle_sin_sum_gt_2_l23_23379

open Real

theorem acute_triangle_sin_sum_gt_2 (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (h_sum : α + β + γ = π) :
  sin α + sin β + sin γ > 2 :=
sorry

end acute_triangle_sin_sum_gt_2_l23_23379


namespace no_solutions_988_1991_l23_23941

theorem no_solutions_988_1991 :
    ¬ ∃ (m n : ℤ),
      (988 ≤ m ∧ m ≤ 1991) ∧
      (988 ≤ n ∧ n ≤ 1991) ∧
      m ≠ n ∧
      ∃ (a b : ℤ), (mn + n = a^2 ∧ mn + m = b^2) := sorry

end no_solutions_988_1991_l23_23941


namespace find_K_l23_23035

theorem find_K 
  (Z K : ℤ) 
  (hZ_range : 1000 < Z ∧ Z < 2000)
  (hZ_eq : Z = K^4)
  (hK_pos : K > 0) :
  K = 6 :=
by {
  sorry -- Proof to be filled in
}

end find_K_l23_23035


namespace ratio_vegan_gluten_free_cupcakes_l23_23250

theorem ratio_vegan_gluten_free_cupcakes :
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  (vegan_gluten_free_cupcakes / vegan_cupcakes) = 1 / 2 :=
by {
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  have h : vegan_gluten_free_cupcakes = 12 := by norm_num
  have r : 12 / 24 = 1 / 2 := by norm_num
  exact r
}

end ratio_vegan_gluten_free_cupcakes_l23_23250


namespace sqrt_product_l23_23103

open Real

theorem sqrt_product :
  sqrt 54 * sqrt 48 * sqrt 6 = 72 * sqrt 3 := by
  sorry

end sqrt_product_l23_23103


namespace three_powers_in_two_digit_range_l23_23819

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l23_23819


namespace average_marks_five_subjects_l23_23081

theorem average_marks_five_subjects 
  (P total_marks : ℕ)
  (h1 : total_marks = P + 350) :
  (total_marks - P) / 5 = 70 :=
by
  sorry

end average_marks_five_subjects_l23_23081


namespace power_mod_remainder_l23_23127

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end power_mod_remainder_l23_23127


namespace solve_system_of_equations_l23_23012

def sys_eq1 (x y : ℝ) : Prop := 6 * (1 - x) ^ 2 = 1 / y
def sys_eq2 (x y : ℝ) : Prop := 6 * (1 - y) ^ 2 = 1 / x

theorem solve_system_of_equations (x y : ℝ) :
  sys_eq1 x y ∧ sys_eq2 x y ↔
  ((x = 3 / 2 ∧ y = 2 / 3) ∨
   (x = 2 / 3 ∧ y = 3 / 2) ∨
   (x = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)) ∧ y = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)))) :=
sorry

end solve_system_of_equations_l23_23012


namespace vector_operation_result_l23_23323

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end vector_operation_result_l23_23323


namespace find_a_if_f_is_odd_l23_23638

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a_if_f_is_odd (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = - f a x) → a = 4 :=
by
  sorry

end find_a_if_f_is_odd_l23_23638


namespace find_least_multiple_of_50_l23_23556

def digits (n : ℕ) : List ℕ := n.digits 10

def product_of_digits (n : ℕ) : ℕ := (digits n).prod

theorem find_least_multiple_of_50 :
  ∃ n, (n % 50 = 0) ∧ ((product_of_digits n) % 50 = 0) ∧ (∀ m, (m % 50 = 0) ∧ ((product_of_digits m) % 50 = 0) → n ≤ m) ↔ n = 5550 :=
by sorry

end find_least_multiple_of_50_l23_23556


namespace part_one_solution_set_part_two_range_of_a_l23_23319

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part_one_solution_set :
  { x : ℝ | f x ≤ 9 } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

theorem part_two_range_of_a (a : ℝ) (B := { x : ℝ | x^2 - 3 * x < 0 })
  (A := { x : ℝ | f x < 2 * x + a }) :
  B ⊆ A → 5 ≤ a :=
sorry

end part_one_solution_set_part_two_range_of_a_l23_23319


namespace vertex_on_x_axis_segment_cut_on_x_axis_l23_23815

-- Define the quadratic function
def quadratic_func (k x : ℝ) : ℝ :=
  (k + 2) * x^2 - 2 * k * x + 3 * k

-- The conditions to prove
theorem vertex_on_x_axis (k : ℝ) :
  (4 * k^2 - 4 * 3 * k * (k + 2) = 0) ↔ (k = 0 ∨ k = -3) :=
sorry

theorem segment_cut_on_x_axis (k : ℝ) :
  ((2 * k / (k + 2))^2 - 12 * k / (k + 2) = 16) ↔ (k = -8/3 ∨ k = -1) :=
sorry

end vertex_on_x_axis_segment_cut_on_x_axis_l23_23815


namespace mean_of_five_numbers_l23_23213

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l23_23213


namespace jacob_three_heads_probability_l23_23509

noncomputable section

def probability_three_heads_after_two_tails : ℚ := 1 / 96

theorem jacob_three_heads_probability :
  let p := (1 / 2) ^ 4 * (1 / 6)
  p = probability_three_heads_after_two_tails := by
sorry

end jacob_three_heads_probability_l23_23509


namespace simplify_and_evaluate_l23_23385

theorem simplify_and_evaluate (a : ℕ) (h : a = 2022) :
  (a - 1) / a / (a - 1 / a) = 1 / 2023 :=
by
  sorry

end simplify_and_evaluate_l23_23385


namespace total_feet_in_garden_l23_23890

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end total_feet_in_garden_l23_23890


namespace greatest_integer_x_l23_23720

theorem greatest_integer_x (x : ℤ) : (5 - 4 * x > 17) → x ≤ -4 :=
by
  sorry

end greatest_integer_x_l23_23720


namespace tangent_line_slope_at_one_l23_23711

variable {f : ℝ → ℝ}

theorem tangent_line_slope_at_one (h : ∀ x, f x = e * x - e) : deriv f 1 = e :=
by sorry

end tangent_line_slope_at_one_l23_23711


namespace equilateral_triangle_shares_side_with_regular_pentagon_l23_23597

theorem equilateral_triangle_shares_side_with_regular_pentagon :
  -- Definitions from the conditions:
  -- CD = CB (isosceles triangle, hence equal angles at B and D)
  let C := Point
  let D := Point
  let B := Point
  let CD := Segment C D
  let CB := Segment C B
  let angle_BCD := 108 -- regular pentagon interior angle
  let angle_DBC := 60 -- equilateral triangle interior angle
  -- Statement to prove:
  mangle_CDB (= CB CD) = 6 :=
  sorry

end equilateral_triangle_shares_side_with_regular_pentagon_l23_23597


namespace total_fencing_cost_l23_23205

-- Definitions of the given conditions
def length : ℝ := 57
def breadth : ℝ := length - 14
def cost_per_meter : ℝ := 26.50

-- Definition of the total cost calculation
def total_cost : ℝ := 2 * (length + breadth) * cost_per_meter

-- Statement of the theorem to be proved
theorem total_fencing_cost :
  total_cost = 5300 := by
  -- Proof is omitted
  sorry

end total_fencing_cost_l23_23205


namespace prove_clothing_colors_l23_23758

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l23_23758


namespace two_digit_powers_of_three_l23_23835

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l23_23835


namespace kayak_trip_friends_l23_23745

theorem kayak_trip_friends :
  ∀ (G : simple_graph (fin 450)), (∀ v : (fin 450), degree v ≥ 100) →
  (∀ s : finset (fin 450), s.card = 200 → ∃ u v ∈ s, G.adj u v) →
  ∃ S : finset (fin 450), S.card = 302 ∧
  ∃ pairing : finset (fin 450 × fin 450), pairing.card = 151 ∧
  (∀ p ∈ pairing, ∃ u v : (fin 450), u ∈ S ∧ v ∈ S ∧ G.adj u v) :=
begin
  sorry
end

end kayak_trip_friends_l23_23745


namespace trapezium_hole_perimeter_correct_l23_23715

variable (a b : ℝ)

def trapezium_hole_perimeter (a b : ℝ) : ℝ :=
  6 * a - 3 * b

theorem trapezium_hole_perimeter_correct (a b : ℝ) :
  trapezium_hole_perimeter a b = 6 * a - 3 * b :=
by
  sorry

end trapezium_hole_perimeter_correct_l23_23715


namespace boat_speed_still_water_l23_23684

theorem boat_speed_still_water (V_s : ℝ) (T_u T_d : ℝ) 
  (h1 : V_s = 24) 
  (h2 : T_u = 2 * T_d) 
  (h3 : (V_b - V_s) * T_u = (V_b + V_s) * T_d) : 
  V_b = 72 := 
sorry

end boat_speed_still_water_l23_23684


namespace two_digit_numbers_in_form_3_pow_n_l23_23824

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l23_23824


namespace candy_initial_amount_l23_23135

namespace CandyProblem

variable (initial_candy given_candy left_candy : ℕ)

theorem candy_initial_amount (h1 : given_candy = 10) (h2 : left_candy = 68) (h3 : left_candy = initial_candy - given_candy) : initial_candy = 78 := 
  sorry
end CandyProblem

end candy_initial_amount_l23_23135


namespace locus_of_center_of_circle_l23_23679

theorem locus_of_center_of_circle (x y a : ℝ)
  (hC : x^2 + y^2 - (2 * a^2 - 4) * x - 4 * a^2 * y + 5 * a^4 - 4 = 0) :
  2 * x - y + 4 = 0 ∧ -2 ≤ x ∧ x < 0 :=
sorry

end locus_of_center_of_circle_l23_23679


namespace intersection_eq_l23_23304

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l23_23304


namespace algebra_expr_eval_l23_23284

theorem algebra_expr_eval {x y : ℝ} (h : x - 2 * y = 3) : 5 - 2 * x + 4 * y = -1 :=
by sorry

end algebra_expr_eval_l23_23284


namespace distinguishable_squares_count_is_70_l23_23235

def count_distinguishable_squares : ℕ :=
  let total_colorings : ℕ := 2^9
  let rotation_90_270_fixed : ℕ := 2^3
  let rotation_180_fixed : ℕ := 2^5
  let average_fixed_colorings : ℕ :=
    (total_colorings + rotation_90_270_fixed + rotation_90_270_fixed + rotation_180_fixed) / 4
  let distinguishable_squares : ℕ := average_fixed_colorings / 2
  distinguishable_squares

theorem distinguishable_squares_count_is_70 :
  count_distinguishable_squares = 70 := by
  sorry

end distinguishable_squares_count_is_70_l23_23235


namespace max_difference_y_coords_intersection_l23_23265

def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

theorem max_difference_y_coords_intersection : ∀ x : ℝ, 
  (f x = g x) → 
  (∀ x₁ x₂ : ℝ, f x₁ = g x₁ ∧ f x₂ = g x₂ → |f x₁ - f x₂| = 0) := 
by
  sorry

end max_difference_y_coords_intersection_l23_23265


namespace students_in_survey_three_l23_23593

theorem students_in_survey_three: 
  (num_students total_students total_groups student_number start_3 end_3 selected_1 : ℤ)
  (systematic_sampling : total_students / total_groups = num_students)
  (student_number = num_students * (systematic_sampling * n - 1) - (total_groups - 7))
  (num_students = 90)
  (total_students = 1080) 
  (total_groups = 1080)
  (start_3 = 847) 
  (end_3 = 1080) 
  (selected_1 = 5)
  (71 < n ∧ n < 91)
: n = 19 :=
sorry

end students_in_survey_three_l23_23593


namespace points_on_ellipse_satisfying_dot_product_l23_23175

theorem points_on_ellipse_satisfying_dot_product :
  ∃ P1 P2 : ℝ × ℝ,
    P1 = (0, 3) ∧ P2 = (0, -3) ∧
    ∀ P : ℝ × ℝ, 
    (P ∈ ({p : ℝ × ℝ | (p.1 / 5)^2 + (p.2 / 3)^2 = 1}) → 
     ((P.1 - (-4)) * (P.1 - 4) + P.2^2 = -7) →
     (P = P1 ∨ P = P2))
:=
sorry

end points_on_ellipse_satisfying_dot_product_l23_23175


namespace min_phi_l23_23640

theorem min_phi
  (ϕ : ℝ) (hϕ : ϕ > 0)
  (h_symm : ∃ k : ℤ, 2 * (π / 6) - 2 * ϕ = k * π + π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end min_phi_l23_23640


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l23_23445

-- Define the basic conditions of the figures
def regular_pentagon (side_length : ℕ) : ℝ := 5 * side_length

-- Define ink length of a figure n
def ink_length (n : ℕ) : ℝ :=
  if n = 1 then regular_pentagon 1 else
  regular_pentagon (n-1) + (3 * (n - 1) + 2)

-- Part (a): Ink length of Figure 4
theorem ink_length_figure_4 : ink_length 4 = 38 := 
  by sorry

-- Part (b): Difference between ink length of Figure 9 and Figure 8
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 29 :=
  by sorry

-- Part (c): Ink length of Figure 100
theorem ink_length_figure_100 : ink_length 100 = 15350 :=
  by sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l23_23445


namespace variation_of_variables_l23_23327

variables (k j : ℝ) (x y z : ℝ)

theorem variation_of_variables (h1 : x = k * y^2) (h2 : y = j * z^3) : ∃ m : ℝ, x = m * z^6 :=
by
  -- Placeholder for the proof
  sorry

end variation_of_variables_l23_23327


namespace answer_one_answer_two_answer_three_l23_23689

def point_condition (A B : ℝ) (P : ℝ) (k : ℝ) : Prop := |A - P| = k * |B - P|

def question_one : Prop :=
  let A := -3
  let B := 6
  let k := 2
  let P := 3
  point_condition A B P k

def question_two : Prop :=
  ∀ x k : ℝ, |x + 2| + |x - 1| = 3 → point_condition (-3) 6 x k → (1 / 8 ≤ k ∧ k ≤ 4 / 5)

def question_three : Prop :=
  let A := -3
  let B := 6
  ∃ t : ℝ, t = 3 / 2 ∧ point_condition A (-3 + t) (6 - 2 * t) 3

theorem answer_one : question_one := by sorry

theorem answer_two : question_two := by sorry

theorem answer_three : question_three := by sorry

end answer_one_answer_two_answer_three_l23_23689


namespace trader_profit_percentage_l23_23909

-- Define the conditions.
variables (indicated_weight actual_weight_given claimed_weight : ℝ)
variable (profit_percentage : ℝ)

-- Given conditions
def conditions :=
  indicated_weight = 1000 ∧
  actual_weight_given = claimed_weight / 1.5 ∧
  claimed_weight = indicated_weight ∧
  profit_percentage = (claimed_weight - actual_weight_given) / actual_weight_given * 100

-- Prove that the profit percentage is 50%
theorem trader_profit_percentage : conditions indicated_weight actual_weight_given claimed_weight profit_percentage → profit_percentage = 50 :=
by
  sorry

end trader_profit_percentage_l23_23909


namespace R_depends_on_a_d_n_l23_23177

-- Definition of sum of an arithmetic progression
def sum_arithmetic_progression (n : ℕ) (a d : ℤ) : ℤ := 
  n * (2 * a + (n - 1) * d) / 2

-- Definitions for s1, s2, and s4
def s1 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression n a d
def s2 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (2 * n) a d
def s4 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (4 * n) a d

-- Definition of R
def R (n : ℕ) (a d : ℤ) : ℤ := s4 n a d - s2 n a d - s1 n a d

-- Theorem stating R depends on a, d, and n
theorem R_depends_on_a_d_n : 
  ∀ (n : ℕ) (a d : ℤ), ∃ (p q r : ℤ), R n a d = p * a + q * d + r := 
by
  sorry

end R_depends_on_a_d_n_l23_23177


namespace find_r_from_tan_cosine_tangent_l23_23497

theorem find_r_from_tan_cosine_tangent 
  (θ : ℝ) 
  (r : ℝ) 
  (htan : Real.tan θ = -7 / 24) 
  (hquadrant : π / 2 < θ ∧ θ < π) 
  (hr : 100 * Real.cos θ = r) : 
  r = -96 := 
sorry

end find_r_from_tan_cosine_tangent_l23_23497


namespace number_is_seven_point_five_l23_23197

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l23_23197


namespace solve_eq_simplify_expression_l23_23736

-- Part 1: Prove the solution to the given equation

theorem solve_eq (x : ℚ) : (1 / (x - 1) + 1 = 3 / (2 * x - 2)) → x = 3 / 2 :=
sorry

-- Part 2: Prove the simplified value of the given expression when x=1/2

theorem simplify_expression : (x = 1/2) →
  ((x^2 / (1 + x) - x) / ((x^2 - 1) / (x^2 + 2 * x + 1)) = 1) :=
sorry

end solve_eq_simplify_expression_l23_23736


namespace total_discount_is_58_percent_l23_23924

-- Definitions and conditions
def sale_discount : ℝ := 0.4
def coupon_discount : ℝ := 0.3

-- Given an original price, the sale discount price and coupon discount price
def sale_price (original_price : ℝ) : ℝ := (1 - sale_discount) * original_price
def final_price (original_price : ℝ) : ℝ := (1 - coupon_discount) * (sale_price original_price)

-- Theorem statement: final discount is 58%
theorem total_discount_is_58_percent (original_price : ℝ) : (original_price - final_price original_price) / original_price = 0.58 :=
by intros; sorry

end total_discount_is_58_percent_l23_23924


namespace two_digit_numbers_in_form_3_pow_n_l23_23825

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l23_23825


namespace powderman_distance_when_blast_heard_l23_23588

-- Define constants
def fuse_time : ℝ := 30  -- seconds
def run_rate : ℝ := 8    -- yards per second
def sound_rate : ℝ := 1080  -- feet per second
def yards_to_feet : ℝ := 3  -- conversion factor

-- Define the time at which the blast was heard
noncomputable def blast_heard_time : ℝ := 675 / 22

-- Define distance functions
def p (t : ℝ) : ℝ := run_rate * yards_to_feet * t  -- distance run by powderman in feet
def q (t : ℝ) : ℝ := sound_rate * (t - fuse_time)  -- distance sound has traveled in feet

-- Proof statement: given the conditions, the distance run by the powderman equals 245 yards
theorem powderman_distance_when_blast_heard :
  p (blast_heard_time) / yards_to_feet = 245 := by
  sorry

end powderman_distance_when_blast_heard_l23_23588


namespace find_a_l23_23636

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -f a x) ∧ a > 0 ∧ a ≠ 1 → a = 4 :=
by
  sorry

end find_a_l23_23636


namespace balance_balls_l23_23523

variable {R Y B W : ℕ}

theorem balance_balls (h1 : 4 * R = 8 * B) 
                      (h2 : 3 * Y = 9 * B) 
                      (h3 : 5 * B = 3 * W) : 
    (2 * R + 4 * Y + 3 * W) = 21 * B :=
by 
  sorry

end balance_balls_l23_23523


namespace clean_per_hour_l23_23097

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end clean_per_hour_l23_23097


namespace mean_of_five_numbers_l23_23215

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l23_23215


namespace side_length_a_l23_23508

theorem side_length_a (a b c : ℝ) (B : ℝ) (h1 : a = c - 2 * a * Real.cos B) (h2 : c = 5) (h3 : 3 * a = 2 * b) :
  a = 4 := by
  sorry

end side_length_a_l23_23508


namespace Rachel_picked_apples_l23_23382

theorem Rachel_picked_apples :
  let apples_from_first_tree := 8
  let apples_from_second_tree := 10
  let apples_from_third_tree := 12
  let apples_from_fifth_tree := 6
  apples_from_first_tree + apples_from_second_tree + apples_from_third_tree + apples_from_fifth_tree = 36 :=
by
  sorry

end Rachel_picked_apples_l23_23382


namespace area_ratio_correct_l23_23666

noncomputable def ratio_area_MNO_XYZ (s t u : ℝ) (S_XYZ : ℝ) : ℝ := 
  let S_XMO := s * (1 - u) * S_XYZ
  let S_YNM := t * (1 - s) * S_XYZ
  let S_OZN := u * (1 - t) * S_XYZ
  S_XYZ - S_XMO - S_YNM - S_OZN

theorem area_ratio_correct (s t u : ℝ) (h1 : s + t + u = 3 / 4) 
  (h2 : s^2 + t^2 + u^2 = 3 / 8) : 
  ratio_area_MNO_XYZ s t u 1 = 13 / 32 := 
by
  -- Proof omitted
  sorry

end area_ratio_correct_l23_23666


namespace no_solution_eqn_l23_23998

theorem no_solution_eqn : ∀ x : ℝ, x ≠ -11 ∧ x ≠ -8 ∧ x ≠ -12 ∧ x ≠ -7 →
  ¬ (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  intros x h
  sorry

end no_solution_eqn_l23_23998


namespace larger_number_of_product_and_sum_l23_23642

theorem larger_number_of_product_and_sum (x y : ℕ) (h_prod : x * y = 35) (h_sum : x + y = 12) : max x y = 7 :=
by {
  sorry
}

end larger_number_of_product_and_sum_l23_23642


namespace dot_product_is_one_l23_23499

variable (a : ℝ × ℝ := (1, 1))
variable (b : ℝ × ℝ := (-1, 2))

theorem dot_product_is_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_is_one_l23_23499


namespace will_total_clothes_l23_23728

theorem will_total_clothes (n1 n2 n3 : ℕ) (h1 : n1 = 32) (h2 : n2 = 9) (h3 : n3 = 3) : n1 + n2 * n3 = 59 := 
by
  sorry

end will_total_clothes_l23_23728


namespace no_valid_triples_l23_23580

theorem no_valid_triples (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : 6 * (a * b + b * c + c * a) = a * b * c) : false :=
by
  sorry

end no_valid_triples_l23_23580


namespace rectangle_to_square_l23_23076

-- Definitions based on conditions
def rectangle_width : ℕ := 12
def rectangle_height : ℕ := 3
def area : ℕ := rectangle_width * rectangle_height
def parts : ℕ := 3
def part_area : ℕ := area / parts
def square_side : ℕ := Nat.sqrt area

-- Theorem to restate the problem
theorem rectangle_to_square : (area = 36) ∧ (part_area = 12) ∧ (square_side = 6) ∧
  (rectangle_width / parts = 4) ∧ (rectangle_height = 3) ∧ 
  ((rectangle_width / parts * parts) = rectangle_width) ∧ (parts * rectangle_height = square_side ^ 2) := by
  -- Placeholder for proof
  sorry

end rectangle_to_square_l23_23076


namespace mean_of_five_numbers_is_correct_l23_23223

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l23_23223


namespace part1_monotonic_intervals_part2_range_of_a_l23_23515

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x + 0.5

theorem part1_monotonic_intervals (x : ℝ) : 
  (f 1 x < (f 1 (x + 1)) ↔ x < 1) ∧ 
  (f 1 x > (f 1 (x - 1)) ↔ x > 1) :=
by sorry

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 1 < x ∧ x ≤ Real.exp 1) 
  (h : (f a x / x) + (1 / (2 * x)) < 0) : 
  a < 1 - (1 / Real.exp 1) :=
by sorry

end part1_monotonic_intervals_part2_range_of_a_l23_23515


namespace neg_p_iff_exists_ge_zero_l23_23475

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + x + 1 < 0

theorem neg_p_iff_exists_ge_zero : ¬ p ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by 
   sorry

end neg_p_iff_exists_ge_zero_l23_23475


namespace arithmetic_mean_solution_l23_23015

theorem arithmetic_mean_solution (x : ℚ) :
  (x + 10 + 20 + 3*x + 18 + 3*x + 6) / 5 = 30 → x = 96 / 7 :=
by
  intros h
  sorry

end arithmetic_mean_solution_l23_23015


namespace determine_k_l23_23535

theorem determine_k (k : ℝ) (h1 : ∃ x y : ℝ, y = 4 * x + 3 ∧ y = -2 * x - 25 ∧ y = 3 * x + k) : k = -5 / 3 := by
  sorry

end determine_k_l23_23535


namespace parallelogram_area_l23_23900

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l23_23900


namespace ratio_a_to_c_l23_23704

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) : 
  a / c = 105 / 16 :=
by sorry

end ratio_a_to_c_l23_23704


namespace rectangle_area_l23_23254

theorem rectangle_area (c h x : ℝ) (h_pos : 0 < h) (c_pos : 0 < c) : 
  (A : ℝ) = (x * (c * x / h)) :=
by
  sorry

end rectangle_area_l23_23254


namespace sequence_property_l23_23479

theorem sequence_property (a : ℕ+ → ℚ)
  (h1 : ∀ p q : ℕ+, a p + a q = a (p + q))
  (h2 : a 1 = 1 / 9) :
  a 36 = 4 :=
sorry

end sequence_property_l23_23479


namespace solve_complex_eq_l23_23627

-- Defining the given condition equation with complex numbers and real variables
theorem solve_complex_eq (a b : ℝ) (h : (1 + 2 * complex.i) * a + b = 2 * complex.i) : 
  a = 1 ∧ b = -1 := 
sorry

end solve_complex_eq_l23_23627


namespace apple_consumption_l23_23085

-- Definitions for the portions of the apple above and below water
def portion_above_water := 1 / 5
def portion_below_water := 4 / 5

-- Rates of consumption by fish and bird
def fish_rate := 120  -- grams per minute
def bird_rate := 60  -- grams per minute

-- The question statements with the correct answers
theorem apple_consumption :
  (portion_below_water * (fish_rate / (fish_rate + bird_rate)) = 2 / 3) ∧ 
  (portion_above_water * (bird_rate / (fish_rate + bird_rate)) = 1 / 3) := 
sorry

end apple_consumption_l23_23085


namespace solve_problem_l23_23785

noncomputable def problem_statement : Prop :=
  let a := Real.arcsin (4/5)
  let b := Real.arccos (1/2)
  Real.sin (a + b) = (4 + 3 * Real.sqrt 3) / 10

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l23_23785


namespace distinct_convex_polygons_l23_23412

def twelve_points : Finset (Fin 12) := (Finset.univ : Finset (Fin 12))

noncomputable def polygon_count_with_vertices (n : ℕ) : ℕ :=
  2^n - 1 - n - (n * (n - 1)) / 2

theorem distinct_convex_polygons :
  polygon_count_with_vertices 12 = 4017 := 
by
  sorry

end distinct_convex_polygons_l23_23412


namespace mean_of_five_numbers_l23_23212

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l23_23212


namespace evaluate_expression_l23_23569

theorem evaluate_expression : 40 + 5 * 12 / (180 / 3) = 41 :=
by
  -- Proof goes here
  sorry

end evaluate_expression_l23_23569


namespace repeating_decimal_to_fraction_l23_23051

theorem repeating_decimal_to_fraction : (let a := (0.28282828 : ℚ); a = 28/99) := sorry

end repeating_decimal_to_fraction_l23_23051


namespace avg_height_correct_l23_23880

theorem avg_height_correct (h1 h2 h3 h4 : ℝ) (h_distinct: h1 ≠ h2 ∧ h2 ≠ h3 ∧ h3 ≠ h4 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h4)
  (h_tallest: h4 = 152) (h_shortest: h1 = 137) 
  (h4_largest: h4 > h3 ∧ h4 > h2 ∧ h4 > h1) (h1_smallest: h1 < h2 ∧ h1 < h3 ∧ h1 < h4) :
  ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg := 
sorry

end avg_height_correct_l23_23880


namespace sum_of_three_numbers_l23_23225

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 20 :=
sorry

end sum_of_three_numbers_l23_23225


namespace max_abs_z_l23_23329

open Complex

theorem max_abs_z (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) : abs z ≤ 1 :=
sorry

end max_abs_z_l23_23329


namespace michelle_initial_crayons_l23_23858

variable (m j : Nat)

axiom janet_crayons : j = 2
axiom michelle_has_after_gift : m + j = 4

theorem michelle_initial_crayons : m = 2 :=
by
  sorry

end michelle_initial_crayons_l23_23858


namespace one_fourth_more_equals_thirty_percent_less_l23_23894

theorem one_fourth_more_equals_thirty_percent_less :
  ∃ n : ℝ, 80 - 0.30 * 80 = (5 / 4) * n ∧ n = 44.8 :=
by
  sorry

end one_fourth_more_equals_thirty_percent_less_l23_23894


namespace max_marks_eq_300_l23_23531

-- Problem Statement in Lean 4

theorem max_marks_eq_300 (m_score p_score c_score : ℝ) 
    (m_percent p_percent c_percent : ℝ)
    (h1 : m_score = 285) (h2 : m_percent = 95) 
    (h3 : p_score = 270) (h4 : p_percent = 90) 
    (h5 : c_score = 255) (h6 : c_percent = 85) :
    (m_score / (m_percent / 100) = 300) ∧ 
    (p_score / (p_percent / 100) = 300) ∧ 
    (c_score / (c_percent / 100) = 300) :=
by
  sorry

end max_marks_eq_300_l23_23531


namespace point_in_second_quadrant_l23_23664

def in_second_quadrant (z : Complex) : Prop := 
  z.re < 0 ∧ z.im > 0

theorem point_in_second_quadrant : in_second_quadrant (Complex.ofReal (1) + 2 * Complex.I / (Complex.ofReal (1) - Complex.I)) :=
by sorry

end point_in_second_quadrant_l23_23664


namespace num_two_digit_powers_of_3_l23_23820

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l23_23820


namespace neg_p_is_correct_l23_23025

def is_positive_integer (x : ℕ) : Prop := x > 0

def proposition_p (x : ℕ) : Prop := (1 / 2 : ℝ) ^ x ≤ 1 / 2

def negation_of_p : Prop := ∃ x : ℕ, is_positive_integer x ∧ ¬ proposition_p x

theorem neg_p_is_correct : negation_of_p :=
sorry

end neg_p_is_correct_l23_23025


namespace number_is_seven_point_five_l23_23195

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l23_23195


namespace job_completion_time_l23_23692

theorem job_completion_time (x : ℤ) (hx : (4 : ℝ) / x + (2 : ℝ) / 3 = 1) : x = 12 := by
  sorry

end job_completion_time_l23_23692


namespace base_conversion_is_248_l23_23873

theorem base_conversion_is_248 (a b c n : ℕ) 
  (h1 : n = 49 * a + 7 * b + c) 
  (h2 : n = 81 * c + 9 * b + a) 
  (h3 : 0 ≤ a ∧ a ≤ 6) 
  (h4 : 0 ≤ b ∧ b ≤ 6) 
  (h5 : 0 ≤ c ∧ c ≤ 6)
  (h6 : 0 ≤ a ∧ a ≤ 8) 
  (h7 : 0 ≤ b ∧ b ≤ 8) 
  (h8 : 0 ≤ c ∧ c ≤ 8) 
  : n = 248 :=
by 
  sorry

end base_conversion_is_248_l23_23873


namespace braiding_time_l23_23167

variables (n_dancers : ℕ) (b_braids_per_dancer : ℕ) (t_seconds_per_braid : ℕ)

theorem braiding_time : n_dancers = 8 → b_braids_per_dancer = 5 → t_seconds_per_braid = 30 → 
  (n_dancers * b_braids_per_dancer * t_seconds_per_braid) / 60 = 20 :=
by
  intros
  sorry

end braiding_time_l23_23167


namespace seventh_term_value_l23_23585

open Nat

noncomputable def a : ℤ := sorry
noncomputable def d : ℤ := sorry
noncomputable def n : ℤ := sorry

-- Conditions as definitions
def sum_first_five : Prop := 5 * a + 10 * d = 34
def sum_last_five : Prop := 5 * a + 5 * (n - 1) * d = 146
def sum_all_terms : Prop := (n * (2 * a + (n - 1) * d)) / 2 = 234

-- Theorem statement
theorem seventh_term_value :
  sum_first_five ∧ sum_last_five ∧ sum_all_terms → a + 6 * d = 18 :=
by
  sorry

end seventh_term_value_l23_23585


namespace female_democrats_count_l23_23239

theorem female_democrats_count 
  (F M : ℕ) 
  (total_participants : F + M = 750)
  (female_democrats : ℕ := F / 2) 
  (male_democrats : ℕ := M / 4)
  (total_democrats : female_democrats + male_democrats = 250) :
  female_democrats = 125 := 
sorry

end female_democrats_count_l23_23239


namespace emma_uniform_number_correct_l23_23476

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

noncomputable def dan : ℕ := 11  -- Example value, but needs to satisfy all conditions
noncomputable def emma : ℕ := 19  -- This is what we need to prove
noncomputable def fiona : ℕ := 13  -- Example value, but needs to satisfy all conditions
noncomputable def george : ℕ := 11  -- Example value, but needs to satisfy all conditions

theorem emma_uniform_number_correct :
  is_two_digit_prime dan ∧
  is_two_digit_prime emma ∧
  is_two_digit_prime fiona ∧
  is_two_digit_prime george ∧
  dan ≠ emma ∧ dan ≠ fiona ∧ dan ≠ george ∧
  emma ≠ fiona ∧ emma ≠ george ∧
  fiona ≠ george ∧
  dan + fiona = 23 ∧
  george + emma = 9 ∧
  dan + fiona + george + emma = 32
  → emma = 19 :=
sorry

end emma_uniform_number_correct_l23_23476


namespace measure_of_angle_F_l23_23974

-- Definitions for the angles in triangle DEF
variables (D E F : ℝ)

-- Given conditions
def is_right_triangle (D : ℝ) : Prop := D = 90
def angle_relation (E F : ℝ) : Prop := E = 4 * F - 10
def angle_sum (D E F : ℝ) : Prop := D + E + F = 180

-- The proof problem statement
theorem measure_of_angle_F (h1 : is_right_triangle D) (h2 : angle_relation E F) (h3 : angle_sum D E F) : F = 20 :=
sorry

end measure_of_angle_F_l23_23974


namespace sum_first_five_special_l23_23251

def is_special (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

theorem sum_first_five_special :
  let special_numbers := [36, 100, 196, 484, 676]
  (∀ n ∈ special_numbers, is_special n) →
  special_numbers.sum = 1492 := by {
  sorry
}

end sum_first_five_special_l23_23251


namespace cos_theta_plus_5π_div_6_l23_23486

theorem cos_theta_plus_5π_div_6 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcond : Real.sin (θ / 2 + π / 6) = 3 / 5) :
  Real.cos (θ + 5 * π / 6) = -24 / 25 :=
by
  sorry -- Proof is skipped as instructed

end cos_theta_plus_5π_div_6_l23_23486


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l23_23831

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l23_23831


namespace clothes_color_proof_l23_23753

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l23_23753


namespace product_of_two_numbers_l23_23884

theorem product_of_two_numbers (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 6) : x * y = 616 :=
sorry

end product_of_two_numbers_l23_23884


namespace correct_product_of_a_b_l23_23857

theorem correct_product_of_a_b (a b : ℕ) (h1 : (a - (10 * (a / 10 % 10) + 1)) * b = 255)
                              (h2 : (a - (10 * (a / 100 % 10 * 10 + a % 10 - (a / 100 % 10 * 10 + 5 * 10)))) * b = 335) :
  a * b = 285 := sorry

end correct_product_of_a_b_l23_23857


namespace Scruffy_weight_l23_23456

variable {Muffy Puffy Scruffy : ℝ}

def Puffy_weight_condition (Muffy Puffy : ℝ) : Prop := Puffy = Muffy + 5
def Scruffy_weight_condition (Muffy Scruffy : ℝ) : Prop := Scruffy = Muffy + 3
def Combined_weight_condition (Muffy Puffy : ℝ) : Prop := Muffy + Puffy = 23

theorem Scruffy_weight (h1 : Puffy_weight_condition Muffy Puffy) (h2 : Scruffy_weight_condition Muffy Scruffy) (h3 : Combined_weight_condition Muffy Puffy) : Scruffy = 12 := by
  sorry

end Scruffy_weight_l23_23456


namespace center_of_tangent_circle_l23_23738

theorem center_of_tangent_circle (x y : ℝ) 
    (h1 : 3 * x - 4 * y = 20) 
    (h2 : 3 * x - 4 * y = -40) 
    (h3 : x - 3 * y = 0) : 
    (x, y) = (-6, -2) := 
by
    sorry

end center_of_tangent_circle_l23_23738


namespace part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l23_23318

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * (p - 1) * x^2 + q * x

theorem part_I_extreme_values : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (f 1 (-3) 3 = f 3 (-3) 3) := 
sorry

theorem part_II_three_distinct_real_roots : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (∀ g : ℝ → ℝ, g x = f x (-3) 3 - 1 → 
  (∀ x, g x ≠ 0) → 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

theorem part_III_compare_sizes (x1 x2 p a l q: ℝ) :
  f (x : ℝ) (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x → 
  x1 < x2 → 
  x2 - x1 > l → 
  x1 > a → 
  (a^2 + p * a + q) > x1 := 
sorry

end part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l23_23318


namespace complex_equation_solution_l23_23620

variable (a b : ℝ)

theorem complex_equation_solution :
  (1 + 2 * complex.I) * a + b = 2 * complex.I → a = 1 ∧ b = -1 :=
by
  sorry

end complex_equation_solution_l23_23620


namespace integer_solutions_of_equation_l23_23024

def satisfies_equation (x y : ℤ) : Prop :=
  x * y - 2 * x - 2 * y + 7 = 0

theorem integer_solutions_of_equation :
  { (x, y) : ℤ × ℤ | satisfies_equation x y } = { (5, 1), (-1, 3), (3, -1), (1, 5) } :=
by sorry

end integer_solutions_of_equation_l23_23024


namespace problem_1_problem_2_l23_23490

def f (a x : ℝ) : ℝ := |a - 3 * x| - |2 + x|

theorem problem_1 (x : ℝ) : f 2 x ≤ 3 ↔ -3 / 4 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

theorem problem_2 (a x : ℝ) : f a x ≥ 1 - a + 2 * |2 + x| → a ≥ -5 / 2 := by
  sorry

end problem_1_problem_2_l23_23490


namespace colors_of_clothes_l23_23762

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l23_23762


namespace determine_clothes_l23_23771

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l23_23771


namespace min_value_expression_l23_23342

theorem min_value_expression (x y z : ℝ) (h1 : -1/2 < x ∧ x < 1/2) (h2 : -1/2 < y ∧ y < 1/2) (h3 : -1/2 < z ∧ z < 1/2) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) + 1 / 2) ≥ 2.5 :=
by {
  sorry
}

end min_value_expression_l23_23342


namespace find_sum_x_y_l23_23676

variables (x y : ℝ)
def a := (x, 1 : ℝ × ℝ)
def b := (1, y : ℝ × ℝ)
def c := (2, -4 : ℝ × ℝ)

axiom a_perpendicular_c : a ⋅ c = 0  -- a ⊥ c
axiom b_parallel_c : ∃ k : ℝ, b = k • c  -- b ∥ c

theorem find_sum_x_y : x + y = 0 :=
sorry

end find_sum_x_y_l23_23676


namespace probability_of_green_tile_l23_23913

theorem probability_of_green_tile :
  let total_tiles := 100
  let green_tiles := 14
  let probability := green_tiles / total_tiles
  probability = 7 / 50 :=
by
  sorry

end probability_of_green_tile_l23_23913


namespace a1_geq_2_pow_k_l23_23338

-- Definitions of the problem conditions in Lean 4
def conditions (a : ℕ → ℕ) (n k : ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i < 2 * n) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → ¬(a i ∣ a j)) ∧
  (3^k < 2 * n ∧ 2 * n < 3^(k+1))

-- The main theorem to be proven
theorem a1_geq_2_pow_k (a : ℕ → ℕ) (n k : ℕ) (h : conditions a n k) : 
  a 1 ≥ 2^k :=
sorry

end a1_geq_2_pow_k_l23_23338


namespace reciprocal_of_repeating_decimal_l23_23563

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l23_23563


namespace percent_yield_hydrogen_gas_l23_23109

theorem percent_yield_hydrogen_gas
  (moles_fe : ℝ) (moles_h2so4 : ℝ) (actual_yield_h2 : ℝ) (theoretical_yield_h2 : ℝ) :
  moles_fe = 3 →
  moles_h2so4 = 4 →
  actual_yield_h2 = 1 →
  theoretical_yield_h2 = moles_fe →
  (actual_yield_h2 / theoretical_yield_h2) * 100 = 33.33 :=
by
  intros h_moles_fe h_moles_h2so4 h_actual_yield_h2 h_theoretical_yield_h2
  sorry

end percent_yield_hydrogen_gas_l23_23109


namespace intersection_eq_l23_23303

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l23_23303


namespace sum_of_sides_is_seven_l23_23613

def triangle_sides : ℕ := 3
def quadrilateral_sides : ℕ := 4
def sum_of_sides : ℕ := triangle_sides + quadrilateral_sides

theorem sum_of_sides_is_seven : sum_of_sides = 7 :=
by
  sorry

end sum_of_sides_is_seven_l23_23613


namespace area_of_region_below_and_left_l23_23266

theorem area_of_region_below_and_left (x y : ℝ) :
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 4^2) ∧ y ≤ 0 ∧ y ≤ x - 4 →
  π * 4^2 / 4 = 4 * π :=
by sorry

end area_of_region_below_and_left_l23_23266


namespace smallest_perfect_square_divisible_by_5_and_6_l23_23046

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l23_23046


namespace cost_of_running_tv_for_week_l23_23173

def powerUsage : ℕ := 125
def hoursPerDay : ℕ := 4
def costPerkWh : ℕ := 14

theorem cost_of_running_tv_for_week :
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  weeklyCost = 49 := by
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  sorry

end cost_of_running_tv_for_week_l23_23173


namespace equalize_expenses_l23_23853

/-- Problem Statement:
Given the amount paid by LeRoy (A), Bernardo (B), and Carlos (C),
prove that the amount LeRoy must adjust to share the costs equally is (B + C - 2A) / 3.
-/
theorem equalize_expenses (A B C : ℝ) : 
  (B+C-2*A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equalize_expenses_l23_23853


namespace adults_wearing_sunglasses_l23_23862

def total_adults : ℕ := 2400
def one_third_of_adults (total : ℕ) : ℕ := total / 3
def women_wearing_sunglasses (women : ℕ) : ℕ := (15 * women) / 100
def men_wearing_sunglasses (men : ℕ) : ℕ := (12 * men) / 100

theorem adults_wearing_sunglasses : 
  let women := one_third_of_adults total_adults
  let men := total_adults - women
  let women_in_sunglasses := women_wearing_sunglasses women
  let men_in_sunglasses := men_wearing_sunglasses men
  women_in_sunglasses + men_in_sunglasses = 312 :=
by
  sorry

end adults_wearing_sunglasses_l23_23862


namespace afternoon_sales_l23_23430

theorem afternoon_sales (x : ℕ) (h : 3 * x = 510) : 2 * x = 340 :=
by sorry

end afternoon_sales_l23_23430


namespace correct_operation_l23_23427

theorem correct_operation (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end correct_operation_l23_23427


namespace investigate_local_extrema_l23_23976

noncomputable def f (x1 x2 : ℝ) : ℝ :=
  3 * x1^2 * x2 - x1^3 - (4 / 3) * x2^3

def is_local_maximum (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∀ (x y : ℝ × ℝ), dist x c < ε → f x.1 x.2 ≤ f c.1 c.2

def is_saddle_point (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∃ (x1 y1 x2 y2 : ℝ × ℝ),
    dist x1 c < ε ∧ dist y1 c < ε ∧ dist x2 c < ε ∧ dist y2 c < ε ∧
    (f x1.1 x1.2 > f c.1 c.2 ∧ f y1.1 y1.2 < f c.1 c.2) ∧
    (f x2.1 x2.2 < f c.1 c.2 ∧ f y2.1 y2.2 > f c.1 c.2)

theorem investigate_local_extrema :
  is_local_maximum f (6, 3) ∧ is_saddle_point f (0, 0) :=
sorry

end investigate_local_extrema_l23_23976


namespace problem_number_eq_7_5_l23_23189

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l23_23189


namespace smaller_circle_radius_l23_23971

theorem smaller_circle_radius (r R : ℝ) (hR : R = 10) (h : 2 * r = 2 * R) : r = 10 :=
by
  sorry

end smaller_circle_radius_l23_23971


namespace ratio_of_means_l23_23111

theorem ratio_of_means (x y : ℝ) (h : (x + y) / (2 * Real.sqrt (x * y)) = 25 / 24) :
  (x / y = 16 / 9) ∨ (x / y = 9 / 16) :=
by
  sorry

end ratio_of_means_l23_23111


namespace solve_inequality_l23_23341

open Set

variable {f : ℝ → ℝ}
open Function

theorem solve_inequality (h_inc : ∀ x y, 0 < x → 0 < y → x < y → f x < f y)
  (h_func_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y)
  (h_f3 : f 3 = 1)
  (x : ℝ) (hx_pos : 0 < x)
  (hx_ge : x > 5)
  (h_ineq : f x - f (1 / (x - 5)) ≥ 2) :
  x ≥ (5 + Real.sqrt 61) / 2 := sorry

end solve_inequality_l23_23341


namespace seven_digit_number_subtraction_l23_23414

theorem seven_digit_number_subtraction 
  (n : ℕ)
  (d1 d2 d3 d4 d5 d6 d7 : ℕ)
  (h1 : n = d1 * 10^6 + d2 * 10^5 + d3 * 10^4 + d4 * 10^3 + d5 * 10^2 + d6 * 10 + d7)
  (h2 : d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ d5 < 10 ∧ d6 < 10 ∧ d7 < 10)
  (h3 : n - (d1 + d3 + d4 + d5 + d6 + d7) = 9875352) :
  n - (d1 + d3 + d4 + d5 + d6 + d7 - d2) = 9875357 :=
sorry

end seven_digit_number_subtraction_l23_23414


namespace problem_statement_l23_23484

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem problem_statement (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end problem_statement_l23_23484


namespace rosie_pie_count_l23_23690

-- Conditions and definitions
def apples_per_pie (total_apples pies : ℕ) : ℕ := total_apples / pies

-- Theorem statement (mathematical proof problem)
theorem rosie_pie_count :
  ∀ (a p : ℕ), a = 12 → p = 3 → (36 : ℕ) / (apples_per_pie a p) = 9 :=
by
  intros a p ha hp
  rw [ha, hp]
  -- Skipping the proof
  sorry

end rosie_pie_count_l23_23690


namespace tan_alpha_beta_l23_23805

noncomputable def tan_alpha := -1 / 3
noncomputable def cos_beta := (Real.sqrt 5) / 5
noncomputable def beta := (1:ℝ) -- Dummy representation for being in first quadrant

theorem tan_alpha_beta (h1 : tan_alpha = -1 / 3) 
                       (h2 : cos_beta = (Real.sqrt 5) / 5) 
                       (h3 : 0 < beta ∧ beta < Real.pi / 2) : 
  Real.tan (α + β) = 1 := 
sorry

end tan_alpha_beta_l23_23805


namespace distinct_convex_polygons_count_l23_23413

-- Twelve points on a circle
def twelve_points_on_circle := 12

-- Calculate the total number of subsets of twelve points
def total_subsets : ℕ := 2 ^ twelve_points_on_circle

-- Calculate the number of subsets with fewer than three members
def subsets_fewer_than_three : ℕ :=
  (Finset.card (Finset.powersetLen 0 (Finset.range twelve_points_on_circle)) +
   Finset.card (Finset.powersetLen 1 (Finset.range twelve_points_on_circle)) +
   Finset.card (Finset.powersetLen 2 (Finset.range twelve_points_on_circle)))

-- The number of convex polygons that can be formed using three or more points
def distinct_convex_polygons : ℕ := total_subsets - subsets_fewer_than_three

-- Lean theorem statement
theorem distinct_convex_polygons_count :
  distinct_convex_polygons = 4017 := by sorry

end distinct_convex_polygons_count_l23_23413


namespace tower_height_l23_23280

theorem tower_height (h d : ℝ) 
  (tan_30_eq : Real.tan (Real.pi / 6) = h / d)
  (tan_45_eq : Real.tan (Real.pi / 4) = h / (d - 20)) :
  h = 20 * Real.sqrt 3 :=
by
  sorry

end tower_height_l23_23280


namespace area_of_rectangle_l23_23075

noncomputable def rectangle_area : ℚ :=
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  length * width

theorem area_of_rectangle : rectangle_area = 392 / 9 :=
  by 
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  have : length * width = 392 / 9 := sorry
  exact this

end area_of_rectangle_l23_23075


namespace benjamin_skating_time_l23_23328

-- Defining the conditions
def distance : ℕ := 80 -- Distance in kilometers
def speed : ℕ := 10   -- Speed in kilometers per hour

-- The main theorem statement
theorem benjamin_skating_time : ∀ (T : ℕ), T = distance / speed → T = 8 := by
  sorry

end benjamin_skating_time_l23_23328


namespace cost_of_bananas_l23_23366

theorem cost_of_bananas
  (apple_cost : ℕ)
  (orange_cost : ℕ)
  (banana_cost : ℕ)
  (num_apples : ℕ)
  (num_oranges : ℕ)
  (num_bananas : ℕ)
  (total_paid : ℕ) 
  (discount_threshold : ℕ)
  (discount_amount : ℕ)
  (total_fruits : ℕ)
  (total_without_discount : ℕ) :
  apple_cost = 1 → 
  orange_cost = 2 → 
  num_apples = 5 → 
  num_oranges = 3 → 
  num_bananas = 2 → 
  total_paid = 15 → 
  discount_threshold = 5 → 
  discount_amount = 1 → 
  total_fruits = num_apples + num_oranges + num_bananas →
  total_without_discount = (num_apples * apple_cost) + (num_oranges * orange_cost) + (num_bananas * banana_cost) →
  (total_without_discount - (discount_amount * (total_fruits / discount_threshold))) = total_paid →
  banana_cost = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end cost_of_bananas_l23_23366


namespace certain_number_is_correct_l23_23649

theorem certain_number_is_correct (x : ℝ) (h : x / 1.45 = 17.5) : x = 25.375 :=
sorry

end certain_number_is_correct_l23_23649


namespace custom_operation_example_l23_23143

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end custom_operation_example_l23_23143


namespace apples_difference_l23_23170

def jimin_apples : ℕ := 7
def grandpa_apples : ℕ := 13
def younger_brother_apples : ℕ := 8
def younger_sister_apples : ℕ := 5

theorem apples_difference :
  grandpa_apples - younger_sister_apples = 8 :=
by
  sorry

end apples_difference_l23_23170


namespace octagon_area_difference_is_512_l23_23925

noncomputable def octagon_area_difference (side_length : ℝ) : ℝ :=
  let initial_octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let triangle_area := (1 / 2) * side_length^2
  let total_triangle_area := 8 * triangle_area
  let inner_octagon_area := initial_octagon_area - total_triangle_area
  initial_octagon_area - inner_octagon_area

theorem octagon_area_difference_is_512 :
  octagon_area_difference 16 = 512 :=
by
  -- This is where the proof would be filled in.
  sorry

end octagon_area_difference_is_512_l23_23925


namespace reciprocal_of_36_recurring_decimal_l23_23568

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l23_23568


namespace subtraction_proof_l23_23602

theorem subtraction_proof :
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 :=
by sorry

end subtraction_proof_l23_23602


namespace log_function_increasing_interval_l23_23806

theorem log_function_increasing_interval (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x < y → y ≤ 3 → 4 - ax > 0 ∧ (4 - ax < 4 - ay)) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end log_function_increasing_interval_l23_23806


namespace prove_clothing_colors_l23_23757

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l23_23757


namespace arithmetic_sequence_probability_l23_23260

def favorable_sequences : List (List ℕ) :=
  [[1, 2, 3], [1, 3, 5], [2, 3, 4], [2, 4, 6], [3, 4, 5], [4, 5, 6], 
   [3, 2, 1], [5, 3, 1], [4, 3, 2], [6, 4, 2], [5, 4, 3], [6, 5, 4], 
   [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := favorable_sequences.length

theorem arithmetic_sequence_probability : (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end arithmetic_sequence_probability_l23_23260


namespace colors_of_clothes_l23_23760

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l23_23760


namespace gcd_solution_l23_23840

theorem gcd_solution {m n : ℕ} (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := 
sorry

end gcd_solution_l23_23840


namespace max_sn_at_16_l23_23028

variable {a : ℕ → ℝ} -- the sequence a_n is represented by a

-- Conditions given in the problem
def isArithmetic (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def bn (a : ℕ → ℝ) (n : ℕ) : ℝ := a n * a (n + 1) * a (n + 2)

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (bn a)

-- Condition: a_{12} = 3/8 * a_5 and a_12 > 0
def specificCondition (a : ℕ → ℝ) : Prop := a 12 = (3 / 8) * a 5 ∧ a 12 > 0

-- The theorem to prove that for S n, the maximum value is reached at n = 16
theorem max_sn_at_16 (a : ℕ → ℝ) (h_arithmetic : isArithmetic a) (h_condition : specificCondition a) :
  ∀ n : ℕ, Sn a n ≤ Sn a 16 := sorry

end max_sn_at_16_l23_23028


namespace probability_same_number_on_four_dice_l23_23037

open Probability

/-- The probability that the same number will be facing up on each of four eight-sided dice that are tossed simultaneously. -/
def prob_same_number_each_four_dice : ℚ := 1 / 512

/-- Given that the dice are eight-sided, four dice are tossed,
and the results on the four dice are independent of each other.
Prove that the probability that the same number will be facing up on each of these four dice is 1/512. -/
theorem probability_same_number_on_four_dice (n : ℕ) (s : Fin n → Fin 8) :
  n = 4 → (∀ i, ∃ k, s i = k) → prob_same_number_each_four_dice = 1 / 512 :=
by intros; simp; sorry

end probability_same_number_on_four_dice_l23_23037


namespace number_notebooks_in_smaller_package_l23_23187

theorem number_notebooks_in_smaller_package 
  (total_notebooks : ℕ)
  (large_packs : ℕ)
  (notebooks_per_large_pack : ℕ)
  (condition_1 : total_notebooks = 69)
  (condition_2 : large_packs = 7)
  (condition_3 : notebooks_per_large_pack = 7)
  (condition_4 : ∃ x : ℕ, x < 7 ∧ (total_notebooks - (large_packs * notebooks_per_large_pack)) % x = 0) :
  ∃ x : ℕ, x < 7 ∧ x = 5 := 
by 
  sorry

end number_notebooks_in_smaller_package_l23_23187


namespace set_intersection_eq_l23_23986

def setA : Set ℝ := { x | x^2 - 3 * x - 4 > 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 5 }
def setC : Set ℝ := { x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) }

theorem set_intersection_eq : setA ∩ setB = setC := by
  sorry

end set_intersection_eq_l23_23986


namespace increased_cost_is_4_percent_l23_23968

-- Initial declarations
variables (initial_cost : ℕ) (price_change_eggs price_change_apples percentage_increase : ℕ)

-- Cost definitions based on initial conditions
def initial_cost_eggs := 100
def initial_cost_apples := 100

-- Price adjustments
def new_cost_eggs := initial_cost_eggs - (initial_cost_eggs * 2 / 100)
def new_cost_apples := initial_cost_apples + (initial_cost_apples * 10 / 100)

-- New combined cost
def new_combined_cost := new_cost_eggs + new_cost_apples

-- Old combined cost
def old_combined_cost := initial_cost_eggs + initial_cost_apples

-- Increase in cost
def increase_in_cost := new_combined_cost - old_combined_cost

-- Percentage increase
def calculated_percentage_increase := (increase_in_cost * 100) / old_combined_cost

-- The proof statement
theorem increased_cost_is_4_percent :
  initial_cost = 100 →
  price_change_eggs = 2 →
  price_change_apples = 10 →
  percentage_increase = 4 →
  calculated_percentage_increase = percentage_increase :=
sorry

end increased_cost_is_4_percent_l23_23968


namespace is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l23_23578

-- Problem 1: If \(2^{n} - 1\) is prime, then \(n\) is prime.
theorem is_prime_if_two_pow_n_minus_one_is_prime (n : ℕ) (hn : Prime (2^n - 1)) : Prime n :=
sorry

-- Problem 2: If \(2^{n} + 1\) is prime, then \(n\) is a power of 2.
theorem is_power_of_two_if_two_pow_n_plus_one_is_prime (n : ℕ) (hn : Prime (2^n + 1)) : ∃ k : ℕ, n = 2^k :=
sorry

end is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l23_23578


namespace determine_clothes_l23_23768

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l23_23768


namespace probability_diff_color_ball_l23_23162

variable (boxA : List String) (boxB : List String)
def P_A (boxA := ["white", "white", "red", "red", "black"]) (boxB := ["white", "white", "white", "white", "red", "red", "red", "black", "black"]) : ℚ := sorry

theorem probability_diff_color_ball :
  P_A boxA boxB = 29 / 50 :=
sorry

end probability_diff_color_ball_l23_23162


namespace reciprocal_of_repeating_decimal_l23_23560

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l23_23560


namespace translation_correct_l23_23927

theorem translation_correct : 
  ∀ (x y : ℝ), (y = -(x-1)^2 + 3) → (x, y) = (0, 0) ↔ (x - 1, y - 3) = (0, 0) :=
by 
  sorry

end translation_correct_l23_23927


namespace range_of_m_for_hyperbola_l23_23018

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ (x y : ℝ), (m+2) ≠ 0 ∧ (m-2) ≠ 0 ∧ (x^2)/(m+2) + (y^2)/(m-2) = 1) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end range_of_m_for_hyperbola_l23_23018


namespace gage_skating_time_l23_23801

theorem gage_skating_time :
  let gage_times_in_minutes1 := 1 * 60 + 15 -- 1 hour 15 minutes converted to minutes
  let gage_times_in_minutes2 := 2 * 60      -- 2 hours converted to minutes
  let total_skating_time_8_days := 5 * gage_times_in_minutes1 + 3 * gage_times_in_minutes2
  let required_total_time := 10 * 95       -- 10 days * 95 minutes per day
  required_total_time - total_skating_time_8_days = 215 :=
by
  sorry

end gage_skating_time_l23_23801


namespace find_x_l23_23701

theorem find_x (x : ℝ) : 17 + x + 2 * x + 13 = 60 → x = 10 :=
by
  sorry

end find_x_l23_23701


namespace greatest_integer_radius_l23_23236

theorem greatest_integer_radius (r : ℝ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
sorry

end greatest_integer_radius_l23_23236


namespace paul_is_19_years_old_l23_23864

theorem paul_is_19_years_old
  (mark_age : ℕ)
  (alice_age : ℕ)
  (paul_age : ℕ)
  (h1 : mark_age = 20)
  (h2 : alice_age = mark_age + 4)
  (h3 : paul_age = alice_age - 5) : 
  paul_age = 19 := by 
  sorry

end paul_is_19_years_old_l23_23864


namespace reciprocal_of_repeating_decimal_l23_23566

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l23_23566


namespace intersection_of_A_and_B_l23_23306

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l23_23306


namespace problem_solution_l23_23406

theorem problem_solution (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^11 - 7 * x^7 + x^3 = 0 := 
sorry

end problem_solution_l23_23406


namespace sum_of_b_for_rational_roots_l23_23132

theorem sum_of_b_for_rational_roots (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 4) (Δ : Nat) :
  (Δ = 49 - 12 * b ∧ (∃ k : Nat, Δ = k * k)) → b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 → 
  ∑ i in Finset.filter (λ b, (∃ (k : ℕ), 49 - 12 * b = k^2)) 
  (Finset.range' 1 5), b = 6 :=
by sorry

end sum_of_b_for_rational_roots_l23_23132


namespace total_boys_school_l23_23332

variable (B : ℕ)
variables (percMuslim percHindu percSikh boysOther : ℕ)

-- Defining the conditions
def condition1 : percMuslim = 44 := by sorry
def condition2 : percHindu = 28 := by sorry
def condition3 : percSikh = 10 := by sorry
def condition4 : boysOther = 54 := by sorry

-- Main theorem statement
theorem total_boys_school (h1 : percMuslim = 44) (h2 : percHindu = 28) (h3 : percSikh = 10) (h4 : boysOther = 54) : 
  B = 300 := by sorry

end total_boys_school_l23_23332


namespace smallest_integer_k_l23_23424

theorem smallest_integer_k (k : ℤ) : k > 2 ∧ k % 19 = 2 ∧ k % 7 = 2 ∧ k % 4 = 2 ↔ k = 534 :=
by
  sorry

end smallest_integer_k_l23_23424


namespace travis_revenue_l23_23230

-- Declare nonnegative integers for apples, apples per box, and price per box
variables (apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ)

-- Specify the conditions
def conditions := apples = 10000 ∧ apples_per_box = 50 ∧ price_per_box = 35

-- State the theorem to be proved
theorem travis_revenue (h : conditions) : (apples / apples_per_box) * price_per_box = 7000 :=
by
  cases h with 
  | intro h1 h2 h3 =>
  rw [h1, h2, h3]
  sorry -- Proof is not required as per the instructions

end travis_revenue_l23_23230


namespace polar_center_coordinates_l23_23849

-- Define polar coordinate system equation
def polar_circle (ρ θ : ℝ) := ρ = 2 * Real.sin θ

-- Define the theorem: Given the equation of a circle in polar coordinates, its center in polar coordinates.
theorem polar_center_coordinates :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi → ∃ ρ, polar_circle ρ θ) →
  (∀ ρ θ, polar_circle ρ θ → 0 ≤ θ ∧ θ < 2 * Real.pi → (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = -1 ∧ θ = 3 * Real.pi / 2)) :=
by {
  sorry 
}

end polar_center_coordinates_l23_23849


namespace parabola_translation_eq_l23_23411

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -x^2 + 2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := - (x - 2)^2 - 1

-- State the theorem to prove the translated function
theorem parabola_translation_eq :
  ∀ x : ℝ, translated_parabola x = - (x - 2)^2 - 1 :=
by
  sorry

end parabola_translation_eq_l23_23411


namespace number_of_women_in_preston_after_one_year_l23_23866

def preston_is_25_times_leesburg (preston leesburg : ℕ) : Prop := 
  preston = 25 * leesburg

def leesburg_population : ℕ := 58940

def women_percentage_leesburg : ℕ := 40

def women_percentage_preston : ℕ := 55

def growth_rate_leesburg : ℝ := 0.025

def growth_rate_preston : ℝ := 0.035

theorem number_of_women_in_preston_after_one_year : 
  ∀ (preston leesburg : ℕ), 
  preston_is_25_times_leesburg preston leesburg → 
  leesburg = 58940 → 
  (women_percentage_preston : ℝ) / 100 * (preston * (1 + growth_rate_preston) : ℝ) = 838788 :=
by 
  sorry

end number_of_women_in_preston_after_one_year_l23_23866


namespace rational_operation_example_l23_23140

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end rational_operation_example_l23_23140


namespace ab_sum_l23_23147

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -1 < x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a - 2 }
def complement_A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 5 }
def complement_B : Set ℝ := { x | x ≤ 2 ∨ x ≥ 8 }
def complement_A_and_C (a b : ℝ) : Set ℝ := { x | 6 ≤ x ∧ x ≤ b }

theorem ab_sum (a b: ℝ) (h: (complement_A ∩ C a) = complement_A_and_C a b) : a + b = 13 :=
by
  sorry

end ab_sum_l23_23147


namespace martha_total_clothes_l23_23375

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l23_23375


namespace proof_x_y_3_l23_23512

noncomputable def prime (n : ℤ) : Prop := 2 <= n ∧ ∀ m : ℤ, 1 ≤ m → m < n → n % m ≠ 0

theorem proof_x_y_3 (x y : ℝ) (p q r : ℤ) (h1 : x - y = p) (hp : prime p) 
  (h2 : x^2 - y^2 = q) (hq : prime q)
  (h3 : x^3 - y^3 = r) (hr : prime r) : p = 3 :=
sorry

end proof_x_y_3_l23_23512


namespace bus_driver_total_earnings_l23_23062

noncomputable def regular_rate : ℝ := 20
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours : ℝ := 45.714285714285715
noncomputable def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
noncomputable def overtime_hours : ℝ := total_hours - regular_hours
noncomputable def regular_pay : ℝ := regular_rate * regular_hours
noncomputable def overtime_pay : ℝ := overtime_rate * overtime_hours
noncomputable def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_total_earnings :
  total_compensation = 1000 :=
by
  sorry

end bus_driver_total_earnings_l23_23062


namespace find_real_numbers_l23_23274

theorem find_real_numbers (a b c : ℝ)    :
  (a + b + c = 3) → (a^2 + b^2 + c^2 = 35) → (a^3 + b^3 + c^3 = 99) → 
  (a = 1 ∧ b = -3 ∧ c = 5) ∨ (a = 1 ∧ b = 5 ∧ c = -3) ∨ 
  (a = -3 ∧ b = 1 ∧ c = 5) ∨ (a = -3 ∧ b = 5 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = -3) ∨ (a = 5 ∧ b = -3 ∧ c = 1) :=
by intros h1 h2 h3; sorry

end find_real_numbers_l23_23274


namespace jerry_trips_l23_23977

-- Define the conditions
def trays_per_trip : Nat := 8
def trays_table1 : Nat := 9
def trays_table2 : Nat := 7

-- Define the proof problem
theorem jerry_trips :
  trays_table1 + trays_table2 = 16 →
  (16 / trays_per_trip) = 2 :=
by
  sorry

end jerry_trips_l23_23977


namespace JuanitaDessertCost_l23_23920

-- Define costs as constants
def brownieCost : ℝ := 2.50
def regularScoopCost : ℝ := 1.00
def premiumScoopCost : ℝ := 1.25
def deluxeScoopCost : ℝ := 1.50
def syrupCost : ℝ := 0.50
def nutsCost : ℝ := 1.50
def whippedCreamCost : ℝ := 0.75
def cherryCost : ℝ := 0.25

-- Define the total cost calculation
def totalCost : ℝ := brownieCost + regularScoopCost + premiumScoopCost +
                     deluxeScoopCost + syrupCost + syrupCost + nutsCost + whippedCreamCost + cherryCost

-- The proof problem: Prove that total cost equals $9.75
theorem JuanitaDessertCost : totalCost = 9.75 :=
by
  -- Proof is omitted
  sorry

end JuanitaDessertCost_l23_23920


namespace custom_operation_example_l23_23142

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end custom_operation_example_l23_23142


namespace largest_class_students_l23_23502

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 115) : x = 27 := 
by 
  sorry

end largest_class_students_l23_23502


namespace identify_clothes_l23_23765

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l23_23765


namespace sample_size_calculation_l23_23918

/--
A factory produces three different models of products: A, B, and C. The ratio of their quantities is 2:3:5.
Using stratified sampling, a sample of size n is drawn, and it contains 16 units of model A.
We need to prove that the sample size n is 80.
-/
theorem sample_size_calculation
  (k : ℕ)
  (hk : 2 * k = 16)
  (n : ℕ)
  (hn : n = (2 + 3 + 5) * k) :
  n = 80 :=
by
  sorry

end sample_size_calculation_l23_23918


namespace intersection_A_B_is_C_l23_23294

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l23_23294


namespace packet_a_weight_l23_23733

theorem packet_a_weight (A B C D E : ℕ) :
  A + B + C = 252 →
  A + B + C + D = 320 →
  E = D + 3 →
  B + C + D + E = 316 →
  A = 75 := by
  sorry

end packet_a_weight_l23_23733


namespace laundry_per_hour_l23_23098

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end laundry_per_hour_l23_23098


namespace min_value_of_box_l23_23650

theorem min_value_of_box 
  (a b : ℤ) 
  (h_distinct : a ≠ b) 
  (h_eq : (a * x + b) * (b * x + a) = 34 * x^2 + Box * x + 34) 
  (h_prod : a * b = 34) :
  ∃ (Box : ℤ), Box = 293 :=
by
  sorry

end min_value_of_box_l23_23650


namespace octavio_can_reach_3_pow_2023_l23_23188

theorem octavio_can_reach_3_pow_2023 (n : ℤ) (hn : n ≥ 1) :
  ∃ (steps : ℕ → ℤ), steps 0 = n ∧ (∀ k, steps (k + 1) = 3 * (steps k)) ∧
  steps 2023 = 3 ^ 2023 :=
by
  sorry

end octavio_can_reach_3_pow_2023_l23_23188


namespace impossibility_of_dividing_into_three_similar_piles_l23_23353

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l23_23353


namespace inverse_of_square_l23_23646

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_square (h : A⁻¹ = ![
  ![3, -2],
  ![1, 1]
]) : 
  (A^2)⁻¹ = ![
  ![7, -8],
  ![4, -1]
] :=
sorry

end inverse_of_square_l23_23646


namespace original_gift_card_value_l23_23002

def gift_card_cost_per_pound : ℝ := 8.58
def coffee_pounds_bought : ℕ := 4
def remaining_balance_after_purchase : ℝ := 35.68

theorem original_gift_card_value :
  (remaining_balance_after_purchase + coffee_pounds_bought * gift_card_cost_per_pound) = 70.00 :=
by
  -- Proof goes here
  sorry

end original_gift_card_value_l23_23002


namespace find_k_l23_23282

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

def vec_add_2b (k : ℝ) : ℝ × ℝ := (2 + 2 * k, 7)
def vec_sub_b (k : ℝ) : ℝ × ℝ := (4 - k, -1)

def vectors_not_parallel (k : ℝ) : Prop :=
  (vec_add_2b k).fst * (vec_sub_b k).snd ≠ (vec_add_2b k).snd * (vec_sub_b k).fst

theorem find_k (k : ℝ) (h : vectors_not_parallel k) : k ≠ 6 :=
by
  sorry

end find_k_l23_23282


namespace complementary_angles_positive_difference_l23_23396

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l23_23396


namespace polynomial_equality_l23_23985

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 5 * x + 7
noncomputable def g (x : ℝ) : ℝ := 12 * x^2 - 19 * x + 25

theorem polynomial_equality :
  f 3 = g 3 ∧ f (3 - Real.sqrt 3) = g (3 - Real.sqrt 3) ∧ f (3 + Real.sqrt 3) = g (3 + Real.sqrt 3) :=
by
  sorry

end polynomial_equality_l23_23985


namespace lives_per_player_l23_23226

-- Definitions based on the conditions
def initial_players : Nat := 2
def joined_players : Nat := 2
def total_lives : Nat := 24

-- Derived condition
def total_players : Nat := initial_players + joined_players

-- Proof statement
theorem lives_per_player : total_lives / total_players = 6 :=
by
  sorry

end lives_per_player_l23_23226


namespace m_range_l23_23491

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) 
  - Real.cos x ^ 2 + 1

def valid_m (m : ℝ) : Prop := 
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), abs (f x - m) ≤ 1

theorem m_range : 
  ∀ m : ℝ, valid_m m ↔ (m ∈ Set.Icc (1 / 2) ((3 - Real.sqrt 3) / 2)) :=
by sorry

end m_range_l23_23491


namespace Q_at_one_is_zero_l23_23605

noncomputable def Q (x : ℚ) : ℚ := x^4 - 2 * x^2 + 1

theorem Q_at_one_is_zero :
  Q 1 = 0 :=
by
  -- Here we would put the formal proof in Lean language
  sorry

end Q_at_one_is_zero_l23_23605


namespace blue_pill_cost_l23_23166

theorem blue_pill_cost :
  ∀ (cost_yellow cost_blue : ℝ) (days : ℕ) (total_cost : ℝ),
    (days = 21) →
    (total_cost = 882) →
    (cost_blue = cost_yellow + 3) →
    (total_cost = days * (cost_blue + cost_yellow)) →
    cost_blue = 22.50 :=
by sorry

end blue_pill_cost_l23_23166


namespace rectangle_perimeter_l23_23939

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 4 * (a + b)) : 2 * (a + b) = 36 := by
  sorry

end rectangle_perimeter_l23_23939


namespace ways_to_write_2020_as_sum_of_twos_and_threes_l23_23838

def write_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2 + 1) else 0

theorem ways_to_write_2020_as_sum_of_twos_and_threes :
  write_as_sum_of_twos_and_threes 2020 = 337 :=
sorry

end ways_to_write_2020_as_sum_of_twos_and_threes_l23_23838


namespace slope_of_line_through_midpoints_l23_23039

theorem slope_of_line_through_midpoints (A B C D : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 4)) (hC : C = (4, 1)) (hD : D = (7, 4)) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  (N.2 - M.2) / (N.1 - M.1) = 0 := by
  sorry

end slope_of_line_through_midpoints_l23_23039


namespace mean_of_five_numbers_is_correct_l23_23224

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l23_23224


namespace ratio_proof_l23_23431

theorem ratio_proof (a b c d e : ℕ) (h1 : a * 4 = 3 * b) (h2 : b * 9 = 7 * c)
  (h3 : c * 7 = 5 * d) (h4 : d * 13 = 11 * e) : a * 468 = 165 * e :=
by
  sorry

end ratio_proof_l23_23431


namespace intersection_A_B_l23_23299

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l23_23299


namespace necessary_but_not_sufficient_l23_23312

variable (a : ℝ)

theorem necessary_but_not_sufficient : (a > 2) → (a > 1) ∧ ¬((a > 1) → (a > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l23_23312


namespace project_completion_time_l23_23052

def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 30
def total_project_days (x : ℚ) : Prop := (work_rate_A * (x - 10) + work_rate_B * x = 1)

theorem project_completion_time (x : ℚ) (h : total_project_days x) : x = 13 := 
sorry

end project_completion_time_l23_23052


namespace molecular_weight_of_compound_l23_23036

theorem molecular_weight_of_compound (C H O n : ℕ) 
    (atomic_weight_C : ℝ) (atomic_weight_H : ℝ) (atomic_weight_O : ℝ) 
    (total_weight : ℝ) 
    (h_C : C = 2) (h_H : H = 4) 
    (h_atomic_weight_C : atomic_weight_C = 12.01) 
    (h_atomic_weight_H : atomic_weight_H = 1.008) 
    (h_atomic_weight_O : atomic_weight_O = 16.00) 
    (h_total_weight : total_weight = 60) : 
    C * atomic_weight_C + H * atomic_weight_H + n * atomic_weight_O = total_weight → 
    n = 2 := 
sorry

end molecular_weight_of_compound_l23_23036


namespace tip_percentage_l23_23886

theorem tip_percentage
  (original_bill : ℝ)
  (shared_per_person : ℝ)
  (num_people : ℕ)
  (total_shared : ℝ)
  (tip_percent : ℝ)
  (h1 : original_bill = 139.0)
  (h2 : shared_per_person = 50.97)
  (h3 : num_people = 3)
  (h4 : total_shared = shared_per_person * num_people)
  (h5 : total_shared - original_bill = 13.91) :
  tip_percent = 13.91 / 139.0 * 100 := 
sorry

end tip_percentage_l23_23886


namespace original_cost_l23_23169

theorem original_cost (SP : ℝ) (C : ℝ) (h1 : SP = 540) (h2 : SP = C + 0.35 * C) : C = 400 :=
by {
  sorry
}

end original_cost_l23_23169


namespace simplify_cub_root_multiplication_l23_23005

theorem simplify_cub_root_multiplication (a b : ℝ) (ha : a = 8) (hb : b = 27) :
  (real.cbrt (a + b) * real.cbrt (a + real.cbrt b)) = real.cbrt ((a + b) * (a + real.cbrt b)) := 
by
  sorry

end simplify_cub_root_multiplication_l23_23005


namespace intersection_of_A_and_B_l23_23308

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l23_23308


namespace employee_b_payment_l23_23734

theorem employee_b_payment (total_payment : ℝ) (A_ratio : ℝ) (payment_B : ℝ) : 
  total_payment = 550 ∧ A_ratio = 1.2 ∧ total_payment = payment_B + A_ratio * payment_B → payment_B = 250 := 
by
  sorry

end employee_b_payment_l23_23734


namespace m_leq_neg_one_l23_23156

theorem m_leq_neg_one (m : ℝ) :
    (∀ x : ℝ, 2^(-x) + m > 0 → x ≤ 0) → m ≤ -1 :=
by
  sorry

end m_leq_neg_one_l23_23156


namespace fixed_point_exists_l23_23136

noncomputable def fixed_point : Prop := ∀ d : ℝ, ∃ (p q : ℝ), (p = -3) ∧ (q = 45) ∧ (q = 5 * p^2 + d * p + 3 * d)

theorem fixed_point_exists : fixed_point :=
by
  sorry

end fixed_point_exists_l23_23136


namespace complex_equation_solution_l23_23622

theorem complex_equation_solution (a b : ℝ) : (1 + (2:ℂ) * complex.I) * a + b = 2 * complex.I → 
  a = 1 ∧ b = -1 :=
by
  intro h
  sorry

end complex_equation_solution_l23_23622


namespace melanie_correct_coins_and_value_l23_23520

def melanie_coins_problem : Prop :=
let dimes_initial := 19
let dimes_dad := 39
let dimes_sister := 15
let dimes_mother := 25
let total_dimes := dimes_initial + dimes_dad + dimes_sister + dimes_mother

let nickels_initial := 12
let nickels_dad := 22
let nickels_sister := 7
let nickels_mother := 10
let nickels_grandmother := 30
let total_nickels := nickels_initial + nickels_dad + nickels_sister + nickels_mother + nickels_grandmother

let quarters_initial := 8
let quarters_dad := 15
let quarters_sister := 12
let quarters_grandmother := 3
let total_quarters := quarters_initial + quarters_dad + quarters_sister + quarters_grandmother

let dimes_value := total_dimes * 0.10
let nickels_value := total_nickels * 0.05
let quarters_value := total_quarters * 0.25
let total_value := dimes_value + nickels_value + quarters_value

total_dimes = 98 ∧ total_nickels = 81 ∧ total_quarters = 38 ∧ total_value = 23.35

theorem melanie_correct_coins_and_value : melanie_coins_problem :=
by sorry

end melanie_correct_coins_and_value_l23_23520


namespace rectangular_plot_breadth_l23_23528

theorem rectangular_plot_breadth:
  ∀ (b l : ℝ), (l = b + 10) → (24 * b = l * b) → b = 14 :=
by
  intros b l hl hs
  sorry

end rectangular_plot_breadth_l23_23528


namespace fox_can_eat_80_fox_cannot_eat_65_l23_23065
-- import the required library

-- Define the conditions for the problem.
def total_candies := 100
def piles := 3
def fox_eat_equalize (fox: ℕ) (pile1: ℕ) (pile2: ℕ): ℕ :=
  if pile1 = pile2 then fox + pile1 else fox + pile2 - pile1

-- Statement for part (a)
theorem fox_can_eat_80: ∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 80) ∨ 
              (fox_eat_equalize x c₁ c₂  = 80)) :=
sorry

-- Statement for part (b)
theorem fox_cannot_eat_65: ¬ (∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 65) ∨ 
              (fox_eat_equalize x c₁ c₂  = 65))) :=
sorry

end fox_can_eat_80_fox_cannot_eat_65_l23_23065


namespace A_left_after_3_days_l23_23061

def work_done_by_A_and_B_together (x : ℕ) : ℚ :=
  (1 / 21) * x + (1 / 28) * x

def work_done_by_B_alone (days : ℕ) : ℚ :=
  (1 / 28) * days

def total_work_done (x days_b_alone : ℕ) : ℚ :=
  work_done_by_A_and_B_together x + work_done_by_B_alone days_b_alone

theorem A_left_after_3_days :
  ∀ (x : ℕ), total_work_done x 21 = 1 ↔ x = 3 := by
  sorry

end A_left_after_3_days_l23_23061


namespace h_at_neg_one_l23_23181

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x ^ 3
def h (x : ℝ) : ℝ := f (g x)

-- The main statement to prove
theorem h_at_neg_one : h (-1) = 3 := by
  sorry

end h_at_neg_one_l23_23181


namespace three_powers_in_two_digit_range_l23_23818

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l23_23818


namespace find_required_school_year_hours_l23_23852

-- Define constants for the problem
def summer_hours_per_week : ℕ := 40
def summer_weeks : ℕ := 12
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 36
def school_year_earnings : ℕ := 9000

-- Calculate total summer hours, hourly rate, total school year hours, and required school year weekly hours
def total_summer_hours := summer_hours_per_week * summer_weeks
def hourly_rate := summer_earnings / total_summer_hours
def total_school_year_hours := school_year_earnings / hourly_rate
def required_school_year_hours_per_week := total_school_year_hours / school_year_weeks

-- Prove the required hours per week is 20
theorem find_required_school_year_hours : required_school_year_hours_per_week = 20 := by
  sorry

end find_required_school_year_hours_l23_23852


namespace min_total_penalty_l23_23741

noncomputable def min_penalty (B W R : ℕ) : ℕ :=
  min (B * W) (min (2 * W * R) (3 * R * B))

theorem min_total_penalty (B W R : ℕ) :
  min_penalty B W R = min (B * W) (min (2 * W * R) (3 * R * B)) := by
  sorry

end min_total_penalty_l23_23741


namespace find_real_numbers_l23_23631

theorem find_real_numbers (a b : ℝ) (h : (1 : ℂ) + 2 * complex.i) * (a : ℂ) + (b : ℂ) = 2 * complex.i) :
  a = 1 ∧ b = -1 := 
by {
  sorry
}

end find_real_numbers_l23_23631


namespace integral_cos8_0_2pi_l23_23091

noncomputable def definite_integral_cos8 (a b : ℝ) : ℝ :=
  ∫ x in a..b, (Real.cos (x / 4)) ^ 8

theorem integral_cos8_0_2pi :
  definite_integral_cos8 0 (2 * Real.pi) = (35 * Real.pi) / 64 :=
by
  sorry

end integral_cos8_0_2pi_l23_23091


namespace restaurant_vegetarian_dishes_l23_23253

theorem restaurant_vegetarian_dishes (n : ℕ) : 
    5 ≥ 2 → 200 < Nat.choose 5 2 * Nat.choose n 2 → n ≥ 7 :=
by
  intros h_combinations h_least
  sorry

end restaurant_vegetarian_dishes_l23_23253


namespace double_probability_correct_l23_23739

def is_double (a : ℕ × ℕ) : Prop := a.1 = a.2

def total_dominoes : ℕ := 13 * 13

def double_count : ℕ := 13

def double_probability := (double_count : ℚ) / total_dominoes

theorem double_probability_correct : double_probability = 13 / 169 := by
  sorry

end double_probability_correct_l23_23739


namespace minimal_primes_ensuring_first_player_win_l23_23896

-- Define primes less than or equal to 100
def primes_le_100 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Define function to get the last digit of a number
def last_digit (n : Nat) : Nat := n % 10

-- Define function to get the first digit of a number
def first_digit (n : Nat) : Nat :=
  let rec first_digit_aux (m : Nat) :=
    if m < 10 then m else first_digit_aux (m / 10)
  first_digit_aux n

-- Define a condition that checks if a prime number follows the game rule
def follows_rule (a b : Nat) : Bool :=
  last_digit a = first_digit b

theorem minimal_primes_ensuring_first_player_win :
  ∃ (p1 p2 p3 : Nat),
  p1 ∈ primes_le_100 ∧
  p2 ∈ primes_le_100 ∧
  p3 ∈ primes_le_100 ∧
  follows_rule p1 p2 ∧
  follows_rule p2 p3 ∧
  p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
sorry

end minimal_primes_ensuring_first_player_win_l23_23896


namespace october_profit_condition_l23_23148

noncomputable def calculate_profit (price_reduction : ℝ) : ℝ :=
  (50 - price_reduction) * (500 + 20 * price_reduction)

theorem october_profit_condition (x : ℝ) (h : calculate_profit x = 28000) : x = 10 ∨ x = 15 := 
by
  sorry

end october_profit_condition_l23_23148


namespace mean_of_five_numbers_l23_23209

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l23_23209


namespace functional_equation_solution_l23_23470

theorem functional_equation_solution {f : ℝ → ℝ}
  (h : ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)) :
  (f = fun x => 0) ∨ (f = id) ∨ (f = fun x => -x) :=
sorry

end functional_equation_solution_l23_23470


namespace impossible_to_divide_three_similar_parts_l23_23355

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l23_23355


namespace mean_of_five_numbers_is_correct_l23_23222

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l23_23222


namespace initial_average_mark_l23_23530

theorem initial_average_mark (A : ℝ) (n_total n_excluded remaining_students_avg : ℝ) 
  (h1 : n_total = 25) 
  (h2 : n_excluded = 5) 
  (h3 : remaining_students_avg = 90)
  (excluded_students_avg : ℝ)
  (h_excluded_avg : excluded_students_avg = 40)
  (A_def : (n_total * A) = (n_excluded * excluded_students_avg + (n_total - n_excluded) * remaining_students_avg)) :
  A = 80 := 
by
  sorry

end initial_average_mark_l23_23530


namespace pears_sold_in_a_day_l23_23591

-- Define the conditions
variable (morning_pears afternoon_pears : ℕ)
variable (h1 : afternoon_pears = 2 * morning_pears)
variable (h2 : afternoon_pears = 320)

-- Lean theorem statement to prove the question answer
theorem pears_sold_in_a_day :
  (morning_pears + afternoon_pears = 480) :=
by
  -- Insert proof here
  sorry

end pears_sold_in_a_day_l23_23591


namespace extreme_value_at_3_increasing_on_interval_l23_23360

def f (a : ℝ) (x : ℝ) : ℝ := 2*x^3 - 3*(a+1)*x^2 + 6*a*x + 8

theorem extreme_value_at_3 (a : ℝ) : (∃ x, x = 3 ∧ 6*x^2 - 6*(a+1)*x + 6*a = 0) → a = 3 :=
by
  sorry

theorem increasing_on_interval (a : ℝ) : (∀ x, x < 0 → 6*(x-a)*(x-1) > 0) → 0 ≤ a :=
by
  sorry

end extreme_value_at_3_increasing_on_interval_l23_23360


namespace work_done_is_halved_l23_23533

theorem work_done_is_halved
  (A₁₂ A₃₄ : ℝ)
  (isothermal_process : ∀ (p V₁₂ V₃₄ : ℝ), V₁₂ = 2 * V₃₄ → p * V₁₂ = A₁₂ → p * V₃₄ = A₃₄) :
  A₃₄ = (1 / 2) * A₁₂ :=
sorry

end work_done_is_halved_l23_23533


namespace carl_garden_area_l23_23462

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end carl_garden_area_l23_23462


namespace quadratic_real_equal_roots_l23_23614

theorem quadratic_real_equal_roots (m : ℝ) :
  (3*x^2 + (2 - m)*x + 5 = 0 → (3 : ℕ) * x^2 + ((2 : ℕ) - m) * x + (5 : ℕ) = 0) →
  ∃ m₁ m₂ : ℝ, m₁ = 2 - 2 * Real.sqrt 15 ∧ m₂ = 2 + 2 * Real.sqrt 15 ∧ 
    (∀ x : ℝ, (3 * x^2 + (2 - m₁) * x + 5 = 0) ∧ (3 * x^2 + (2 - m₂) * x + 5 = 0)) :=
sorry

end quadratic_real_equal_roots_l23_23614


namespace quadratic_inequality_l23_23807

variable (a b c A B C : ℝ)

theorem quadratic_inequality
  (h₁ : a ≠ 0)
  (h₂ : A ≠ 0)
  (h₃ : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end quadratic_inequality_l23_23807


namespace general_term_is_correct_l23_23163

variable (a : ℕ → ℤ)
variable (n : ℕ)

def is_arithmetic_sequence := ∃ d a₁, ∀ n, a n = a₁ + d * (n - 1)

axiom a_10_eq_30 : a 10 = 30
axiom a_20_eq_50 : a 20 = 50

noncomputable def general_term (n : ℕ) : ℤ := 2 * n + 10

theorem general_term_is_correct (a: ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 10 = 30)
  (h3 : a 20 = 50)
  : ∀ n, a n = general_term n :=
sorry

end general_term_is_correct_l23_23163


namespace divide_pile_l23_23359

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l23_23359


namespace solve_equation_l23_23871

theorem solve_equation (x : ℝ) :
  (4 * x + 1) * (3 * x + 1) * (2 * x + 1) * (x + 1) = 3 * x ^ 4  →
  x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 :=
by
  sorry

end solve_equation_l23_23871


namespace minimum_value_of_a_l23_23482

def is_prime (n : ℕ) : Prop := sorry  -- Provide the definition of a prime number

def is_perfect_square (n : ℕ) : Prop := sorry  -- Provide the definition of a perfect square

theorem minimum_value_of_a 
  (a b : ℕ) 
  (h1 : is_prime (a - b)) 
  (h2 : is_perfect_square (a * b)) 
  (h3 : a ≥ 2012) : 
  a = 2025 := 
sorry

end minimum_value_of_a_l23_23482


namespace gcd_a_b_eq_one_l23_23102

def a : ℕ := 47^5 + 1
def b : ℕ := 47^5 + 47^3 + 1

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l23_23102


namespace cards_probability_ratio_l23_23936

theorem cards_probability_ratio :
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  q / p = 44 :=
by
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  have : q / p = 44 := sorry
  exact this

end cards_probability_ratio_l23_23936


namespace football_championship_prediction_l23_23658

theorem football_championship_prediction
  (teams : Fin 16 → ℕ)
  (h_distinct: ∃ i j, i ≠ j ∧ teams i = teams j) :
  ∃ i_j_same : Fin 16, ∃ i_j_strongest : ∀ k, teams k ≤ teams i_j_same,
  ¬ ∀ (pairing : (Fin 16) → (Fin 2)) (round : ℕ), ∀ (p1 p2 : Fin 16), p1 ≠ p2 ∧ pairing p1 = pairing p2 → teams p1 ≠ teams p2 → 
  ∃ w, w ∈ {p1, p2} :=
sorry

end football_championship_prediction_l23_23658


namespace kite_cost_l23_23932

variable (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ)

theorem kite_cost (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ) (h_initial_amount : initial_amount = 78) (h_cost_frisbee : cost_frisbee = 9) (h_amount_left : amount_left = 61) : 
  initial_amount - amount_left - cost_frisbee = 8 :=
by
  -- Proof can be completed here
  sorry

end kite_cost_l23_23932


namespace ratio_diamond_brace_ring_l23_23511

theorem ratio_diamond_brace_ring
  (cost_ring : ℤ) (cost_car : ℤ) (total_worth : ℤ) (cost_diamond_brace : ℤ)
  (h1 : cost_ring = 4000) (h2 : cost_car = 2000) (h3 : total_worth = 14000)
  (h4 : cost_diamond_brace = total_worth - (cost_ring + cost_car)) :
  cost_diamond_brace / cost_ring = 2 :=
by
  sorry

end ratio_diamond_brace_ring_l23_23511


namespace sum_first_five_terms_eq_15_l23_23145

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d 

variable (a : ℕ → ℝ) (h_arith_seq : is_arithmetic_sequence a) (h_a3 : a 3 = 3)

theorem sum_first_five_terms_eq_15 : (a 1 + a 2 + a 3 + a 4 + a 5 = 15) :=
sorry

end sum_first_five_terms_eq_15_l23_23145


namespace quotient_product_larger_integer_l23_23703

theorem quotient_product_larger_integer
  (x y : ℕ)
  (h1 : y / x = 7 / 3)
  (h2 : x * y = 189)
  : y = 21 := 
sorry

end quotient_product_larger_integer_l23_23703


namespace fruit_store_initial_quantities_l23_23442

-- Definitions from conditions:
def total_fruit (a b c : ℕ) := a + b + c = 275
def sold_apples (a : ℕ) := a - 30
def added_peaches (b : ℕ) := b + 45
def sold_pears (c : ℕ) := c - c / 4
def final_ratio (a b c : ℕ) := (sold_apples a) / 4 = (added_peaches b) / 3 ∧ (added_peaches b) / 3 = (sold_pears c) / 2

-- The proof problem:
theorem fruit_store_initial_quantities (a b c : ℕ) (h1 : total_fruit a b c) 
  (h2 : final_ratio a b c) : a = 150 ∧ b = 45 ∧ c = 80 :=
sorry

end fruit_store_initial_quantities_l23_23442


namespace fraction_grades_C_l23_23331

def fraction_grades_A (students : ℕ) : ℕ := (1 / 5) * students
def fraction_grades_B (students : ℕ) : ℕ := (1 / 4) * students
def num_grades_D : ℕ := 5
def total_students : ℕ := 100

theorem fraction_grades_C :
  (total_students - (fraction_grades_A total_students + fraction_grades_B total_students + num_grades_D)) / total_students = 1 / 2 :=
by
  sorry

end fraction_grades_C_l23_23331


namespace most_representative_sample_l23_23716

/-- Options for the student sampling methods -/
inductive SamplingMethod
| NinthGradeStudents : SamplingMethod
| FemaleStudents : SamplingMethod
| BasketballStudents : SamplingMethod
| StudentsWithIDEnding5 : SamplingMethod

/-- Definition of representativeness for each SamplingMethod -/
def isMostRepresentative (method : SamplingMethod) : Prop :=
  method = SamplingMethod.StudentsWithIDEnding5

/-- Prove that the students with ID ending in 5 is the most representative sampling method -/
theorem most_representative_sample : isMostRepresentative SamplingMethod.StudentsWithIDEnding5 :=
  by
  sorry

end most_representative_sample_l23_23716


namespace simplify_expression_l23_23258

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end simplify_expression_l23_23258


namespace rosy_fish_count_l23_23987

theorem rosy_fish_count (L R T : ℕ) (hL : L = 10) (hT : T = 19) : R = T - L := by
  sorry

end rosy_fish_count_l23_23987


namespace double_inequality_solution_l23_23009

open Set

theorem double_inequality_solution (x : ℝ) :
  -1 < (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) ∧
  (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) < 1 ↔
  x ∈ Ioo (3 / 2) 4 ∪ Ioi 8 :=
by
  sorry

end double_inequality_solution_l23_23009


namespace ab_plus_cd_l23_23651

variable (a b c d : ℝ)

theorem ab_plus_cd (h1 : a + b + c = -4)
                  (h2 : a + b + d = 2)
                  (h3 : a + c + d = 15)
                  (h4 : b + c + d = 10) :
                  a * b + c * d = 485 / 9 :=
by
  sorry

end ab_plus_cd_l23_23651


namespace salary_increase_l23_23003

variable (S : ℝ) -- Robert's original salary
variable (P : ℝ) -- Percentage increase after decrease in decimal form

theorem salary_increase (h1 : 0.5 * S * (1 + P) = 0.75 * S) : P = 0.5 := 
by 
  sorry

end salary_increase_l23_23003


namespace martha_total_clothes_l23_23371

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l23_23371


namespace max_product_of_real_roots_quadratic_eq_l23_23814

theorem max_product_of_real_roots_quadratic_eq : ∀ (k : ℝ), (∃ x y : ℝ, 4 * x ^ 2 - 8 * x + k = 0 ∧ 4 * y ^ 2 - 8 * y + k = 0) 
    → k = 4 :=
sorry

end max_product_of_real_roots_quadratic_eq_l23_23814


namespace relation_among_a_b_c_l23_23951

open Real

theorem relation_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = log 3 / log 2)
  (h2 : b = log 7 / (2 * log 2))
  (h3 : c = 0.7 ^ 4) :
  a > b ∧ b > c :=
by
  -- we leave the proof as an exercise
  sorry

end relation_among_a_b_c_l23_23951


namespace solve_for_x_l23_23944

theorem solve_for_x (x : ℝ) : (3 : ℝ)^(4 * x^2 - 3 * x + 5) = (3 : ℝ)^(4 * x^2 + 9 * x - 6) ↔ x = 11 / 12 :=
by sorry

end solve_for_x_l23_23944


namespace product_of_functions_l23_23812

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem product_of_functions (x : ℝ) (hx : x ≠ -3) : f x * g x = x - 3 := by
  -- proof goes here
  sorry

end product_of_functions_l23_23812


namespace eq_satisfied_for_all_y_l23_23472

theorem eq_satisfied_for_all_y (x : ℝ) : 
  (∀ y: ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by
  sorry

end eq_satisfied_for_all_y_l23_23472


namespace joan_gave_27_apples_l23_23171

theorem joan_gave_27_apples (total_apples : ℕ) (current_apples : ℕ)
  (h1 : total_apples = 43) 
  (h2 : current_apples = 16) : 
  total_apples - current_apples = 27 := 
by
  sorry

end joan_gave_27_apples_l23_23171


namespace sheets_borrowed_l23_23641

theorem sheets_borrowed (pages sheets borrowed remaining_sheets : ℕ) 
  (h1 : pages = 70) 
  (h2 : sheets = 35)
  (h3 : remaining_sheets = sheets - borrowed)
  (h4 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> 2*i-1 <= pages) 
  (h5 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> i + 1 != borrowed ∧ i <= remaining_sheets)
  (avg : ℕ) (h6 : avg = 28)
  : borrowed = 17 := by
  sorry

end sheets_borrowed_l23_23641


namespace condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l23_23673

variables {a b : ℝ}

theorem condition_3_implies_at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

theorem condition_5_implies_at_least_one_gt_one (h : ab > 1) : a > 1 ∨ b > 1 :=
sorry

end condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l23_23673


namespace number_of_players_l23_23014
-- Importing the necessary library

-- Define the number of games formula for the tournament
def number_of_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- The theorem to prove the number of players given the conditions
theorem number_of_players (n : ℕ) (h : number_of_games n = 306) : n = 18 :=
by
  sorry

end number_of_players_l23_23014


namespace cost_of_each_green_hat_l23_23415

theorem cost_of_each_green_hat
  (total_hats : ℕ) (cost_blue_hat : ℕ) (total_price : ℕ) (green_hats : ℕ) (blue_hats : ℕ) (cost_green_hat : ℕ)
  (h1 : total_hats = 85) 
  (h2 : cost_blue_hat = 6) 
  (h3 : total_price = 550) 
  (h4 : green_hats = 40) 
  (h5 : blue_hats = 45) 
  (h6 : green_hats + blue_hats = total_hats) 
  (h7 : total_price = green_hats * cost_green_hat + blue_hats * cost_blue_hat) :
  cost_green_hat = 7 := 
sorry

end cost_of_each_green_hat_l23_23415


namespace fraction_covered_by_pepperoni_l23_23945

theorem fraction_covered_by_pepperoni 
  (d_pizza : ℝ) (n_pepperoni_diameter : ℕ) (n_pepperoni : ℕ) (diameter_pepperoni : ℝ) 
  (radius_pepperoni : ℝ) (radius_pizza : ℝ)
  (area_one_pepperoni : ℝ) (total_area_pepperoni : ℝ) (area_pizza : ℝ)
  (fraction_covered : ℝ)
  (h1 : d_pizza = 16)
  (h2 : n_pepperoni_diameter = 14)
  (h3 : n_pepperoni = 42)
  (h4 : diameter_pepperoni = d_pizza / n_pepperoni_diameter)
  (h5 : radius_pepperoni = diameter_pepperoni / 2)
  (h6 : radius_pizza = d_pizza / 2)
  (h7 : area_one_pepperoni = π * radius_pepperoni ^ 2)
  (h8 : total_area_pepperoni = n_pepperoni * area_one_pepperoni)
  (h9 : area_pizza = π * radius_pizza ^ 2)
  (h10 : fraction_covered = total_area_pepperoni / area_pizza) :
  fraction_covered = 3 / 7 :=
sorry

end fraction_covered_by_pepperoni_l23_23945


namespace no_division_into_three_similar_piles_l23_23345

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l23_23345


namespace sum_of_values_l23_23882

theorem sum_of_values (N : ℝ) (h : N * (N + 4) = 8) : N + (4 - N - 8 / N) = -4 := 
sorry

end sum_of_values_l23_23882
