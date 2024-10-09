import Mathlib

namespace mole_fractions_C4H8O2_l1763_176331

/-- 
Given:
- The molecular formula of C4H8O2,
- 4 moles of carbon (C) atoms,
- 8 moles of hydrogen (H) atoms,
- 2 moles of oxygen (O) atoms.

Prove that:
The mole fractions of each element in C4H8O2 are:
- Carbon (C): 2/7
- Hydrogen (H): 4/7
- Oxygen (O): 1/7
--/
theorem mole_fractions_C4H8O2 :
  let m_C := 4
  let m_H := 8
  let m_O := 2
  let total_moles := m_C + m_H + m_O
  let mole_fraction_C := m_C / total_moles
  let mole_fraction_H := m_H / total_moles
  let mole_fraction_O := m_O / total_moles
  mole_fraction_C = 2 / 7 ∧ mole_fraction_H = 4 / 7 ∧ mole_fraction_O = 1 / 7 := by
  sorry

end mole_fractions_C4H8O2_l1763_176331


namespace inequality_interval_l1763_176345

theorem inequality_interval : ∀ x : ℝ, (x^2 - 3 * x - 4 < 0) ↔ (-1 < x ∧ x < 4) :=
by
  intro x
  sorry

end inequality_interval_l1763_176345


namespace num_arithmetic_sequences_l1763_176316

theorem num_arithmetic_sequences (a d : ℕ) (n : ℕ) (h1 : n >= 3) (h2 : n * (2 * a + (n - 1) * d) = 2 * 97^2) :
  ∃ seqs : ℕ, seqs = 4 :=
by sorry

end num_arithmetic_sequences_l1763_176316


namespace proof_problem_l1763_176322

variable (a b c x y z : ℝ)

theorem proof_problem
  (h1 : x + y - z = a - b)
  (h2 : x - y + z = b - c)
  (h3 : - x + y + z = c - a) : 
  x + y + z = 0 := by
  sorry

end proof_problem_l1763_176322


namespace bruce_total_payment_l1763_176314

def grapes_quantity : ℕ := 8
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_grapes : ℕ := grapes_quantity * grapes_rate
def cost_mangoes : ℕ := mangoes_quantity * mangoes_rate
def total_cost : ℕ := cost_grapes + cost_mangoes

theorem bruce_total_payment : total_cost = 1055 := by
  sorry

end bruce_total_payment_l1763_176314


namespace projectile_height_35_l1763_176329

noncomputable def projectile_height (t : ℝ) : ℝ := -4.9 * t^2 + 30 * t

theorem projectile_height_35 (t : ℝ) :
  projectile_height t = 35 ↔ t = 10/7 :=
by {
  sorry
}

end projectile_height_35_l1763_176329


namespace tiffany_mile_fraction_l1763_176344

/-- Tiffany's daily running fraction (x) for Wednesday, Thursday, and Friday must be 1/3
    such that both Billy and Tiffany run the same total miles over a week. --/
theorem tiffany_mile_fraction :
  ∃ x : ℚ, (3 * 1 + 1) = 1 + (3 * 2 + 3 * x) → x = 1 / 3 :=
by
  sorry

end tiffany_mile_fraction_l1763_176344


namespace expand_polynomial_eq_l1763_176398

theorem expand_polynomial_eq :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) = 6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by
  sorry

end expand_polynomial_eq_l1763_176398


namespace sufficient_but_not_necessary_condition_l1763_176356

theorem sufficient_but_not_necessary_condition (a b : ℝ) : (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(∀ a b, a^2 + b ≥ 0 → b ≥ 0) := by
  sorry

end sufficient_but_not_necessary_condition_l1763_176356


namespace sally_balloon_count_l1763_176327

theorem sally_balloon_count 
  (joan_balloons : Nat)
  (jessica_balloons : Nat)
  (total_balloons : Nat)
  (sally_balloons : Nat)
  (h_joan : joan_balloons = 9)
  (h_jessica : jessica_balloons = 2)
  (h_total : total_balloons = 16)
  (h_eq : total_balloons = joan_balloons + jessica_balloons + sally_balloons) : 
  sally_balloons = 5 :=
by
  sorry

end sally_balloon_count_l1763_176327


namespace sad_girls_count_l1763_176373

-- Statement of the problem in Lean 4
theorem sad_girls_count :
  ∀ (total_children happy_children sad_children neither_happy_nor_sad children boys girls happy_boys boys_neither_happy_nor_sad : ℕ),
    total_children = 60 →
    happy_children = 30 →
    sad_children = 10 →
    neither_happy_nor_sad = 20 →
    children = total_children →
    boys = 19 →
    girls = total_children - boys →
    happy_boys = 6 →
    boys_neither_happy_nor_sad = 7 →
    girls = 41 →
    sad_children = 10 →
    (sad_children = 6 + (total_children - boys - girls - neither_happy_nor_sad - happy_children)) → 
    ∃ sad_girls, sad_girls = 4 := by
  sorry

end sad_girls_count_l1763_176373


namespace number_of_slices_l1763_176340

theorem number_of_slices 
  (pepperoni ham sausage total_meat pieces_per_slice : ℕ)
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : total_meat = pepperoni + ham + sausage)
  (h5 : pieces_per_slice = 22) :
  total_meat / pieces_per_slice = 6 :=
by
  sorry

end number_of_slices_l1763_176340


namespace LCM_14_21_l1763_176317

theorem LCM_14_21 : Nat.lcm 14 21 = 42 := 
by
  sorry

end LCM_14_21_l1763_176317


namespace no_solution_for_x_l1763_176362

theorem no_solution_for_x (a : ℝ) (h : a ≤ 8) : ¬ ∃ x : ℝ, |x - 5| + |x + 3| < a :=
by
  sorry

end no_solution_for_x_l1763_176362


namespace sum_digits_of_three_digit_numbers_l1763_176333

theorem sum_digits_of_three_digit_numbers (a c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hc : 1 ≤ c ∧ c < 10) 
  (h1 : (300 + 10 * a + 7) + 414 = 700 + 10 * c + 1)
  (h2 : ∃ k : ℤ, 700 + 10 * c + 1 = 11 * k) :
  a + c = 14 :=
by
  sorry

end sum_digits_of_three_digit_numbers_l1763_176333


namespace find_r_squared_l1763_176357

noncomputable def parabola_intersect_circle_radius_squared : Prop :=
  ∀ (x y : ℝ), y = (x - 1)^2 ∧ x - 3 = (y + 2)^2 → (x - 3/2)^2 + (y + 3/2)^2 = 1/2

theorem find_r_squared : parabola_intersect_circle_radius_squared :=
sorry

end find_r_squared_l1763_176357


namespace ball_reaches_20_feet_at_1_75_seconds_l1763_176387

noncomputable def ball_height (t : ℝ) : ℝ :=
  60 - 9 * t - 8 * t ^ 2

theorem ball_reaches_20_feet_at_1_75_seconds :
  ∃ t : ℝ, ball_height t = 20 ∧ t = 1.75 ∧ t ≥ 0 :=
by {
  sorry
}

end ball_reaches_20_feet_at_1_75_seconds_l1763_176387


namespace graveling_cost_correct_l1763_176377

-- Define the dimensions of the rectangular lawn
def lawn_length : ℕ := 80 -- in meters
def lawn_breadth : ℕ := 50 -- in meters

-- Define the width of each road
def road_width : ℕ := 10 -- in meters

-- Define the cost per square meter for graveling the roads
def cost_per_sq_m : ℕ := 3 -- in Rs. per sq meter

-- Define the area of the road parallel to the length of the lawn
def area_road_parallel_length : ℕ := lawn_length * road_width

-- Define the effective length of the road parallel to the breadth of the lawn
def effective_road_parallel_breadth_length : ℕ := lawn_breadth - road_width

-- Define the area of the road parallel to the breadth of the lawn
def area_road_parallel_breadth : ℕ := effective_road_parallel_breadth_length * road_width

-- Define the total area to be graveled
def total_area_to_be_graveled : ℕ := area_road_parallel_length + area_road_parallel_breadth

-- Define the total cost of graveling
def total_graveling_cost : ℕ := total_area_to_be_graveled * cost_per_sq_m

-- Theorem: The total cost of graveling the two roads is Rs. 3600
theorem graveling_cost_correct : total_graveling_cost = 3600 := 
by
  unfold total_graveling_cost total_area_to_be_graveled area_road_parallel_length area_road_parallel_breadth effective_road_parallel_breadth_length lawn_length lawn_breadth road_width cost_per_sq_m
  exact rfl

end graveling_cost_correct_l1763_176377


namespace mean_and_sum_l1763_176396

-- Define the sum of five numbers to be 1/3
def sum_of_five_numbers : ℚ := 1 / 3

-- Define the mean of these five numbers
def mean_of_five_numbers : ℚ := sum_of_five_numbers / 5

-- State the theorem
theorem mean_and_sum (h : sum_of_five_numbers = 1 / 3) :
  mean_of_five_numbers = 1 / 15 ∧ (mean_of_five_numbers + sum_of_five_numbers = 2 / 5) :=
by
  sorry

end mean_and_sum_l1763_176396


namespace imaginary_part_of_z_l1763_176350

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 :=
sorry

end imaginary_part_of_z_l1763_176350


namespace upper_side_length_trapezoid_l1763_176368

theorem upper_side_length_trapezoid
  (L U : ℝ) 
  (h : ℝ := 8) 
  (A : ℝ := 72) 
  (cond1 : U = L - 6)
  (cond2 : 1/2 * (L + U) * h = A) :
  U = 6 := 
by 
  sorry

end upper_side_length_trapezoid_l1763_176368


namespace jeffery_fish_count_l1763_176325

variable (J R Y : ℕ)

theorem jeffery_fish_count :
  (R = 3 * J) → (Y = 2 * R) → (J + R + Y = 100) → (Y = 60) :=
by
  intros hR hY hTotal
  have h1 : R = 3 * J := hR
  have h2 : Y = 2 * R := hY
  rw [h1, h2] at hTotal
  sorry

end jeffery_fish_count_l1763_176325


namespace points_on_same_line_l1763_176394

theorem points_on_same_line (p : ℝ) :
  (∃ m : ℝ, m = ( -3.5 - 0.5 ) / ( 3 - (-1)) ∧ ∀ x y : ℝ, 
    (x = -1 ∧ y = 0.5) ∨ (x = 3 ∧ y = -3.5) ∨ (x = 7 ∧ y = p) → y = m * x + (0.5 - m * (-1))) →
    p = -7.5 :=
by
  sorry

end points_on_same_line_l1763_176394


namespace identity_element_is_neg4_l1763_176324

def op (a b : ℝ) := a + b + 4

def is_identity (e : ℝ) := ∀ a : ℝ, op e a = a

theorem identity_element_is_neg4 : ∃ e : ℝ, is_identity e ∧ e = -4 :=
by
  use -4
  sorry

end identity_element_is_neg4_l1763_176324


namespace find_x_l1763_176366

variable (x : ℕ)

def f (x : ℕ) : ℕ := 2 * x + 5
def g (y : ℕ) : ℕ := 3 * y

theorem find_x (h : g (f x) = 123) : x = 18 :=
by {
  sorry
}

end find_x_l1763_176366


namespace distribution_schemes_l1763_176355

theorem distribution_schemes 
    (total_professors : ℕ)
    (high_schools : Finset ℕ) 
    (A : ℕ) 
    (B : ℕ) 
    (C : ℕ)
    (D : ℕ)
    (cond1 : total_professors = 6) 
    (cond2 : A = 1)
    (cond3 : B ≥ 1)
    (cond4 : C ≥ 1)
    (D' := (total_professors - A - B - C)) 
    (cond5 : D' ≥ 1) : 
    ∃ N : ℕ, N = 900 := by
  sorry

end distribution_schemes_l1763_176355


namespace ram_krish_task_completion_l1763_176332

theorem ram_krish_task_completion
  (ram_days : ℝ)
  (krish_efficiency_factor : ℝ)
  (task_time : ℝ) 
  (H1 : krish_efficiency_factor = 2)
  (H2 : ram_days = 27) 
  (H3 : task_time = 9) :
  (1 / task_time) = (1 / ram_days + 1 / (ram_days / krish_efficiency_factor)) := 
sorry

end ram_krish_task_completion_l1763_176332


namespace car_count_is_150_l1763_176312

variable (B C K : ℕ)  -- Define the variables representing buses, cars, and bikes

/-- Given conditions: The ratio of buses to cars to bikes is 3:7:10,
    there are 90 fewer buses than cars, and 140 fewer buses than bikes. -/
def conditions : Prop :=
  (C = (7 * B / 3)) ∧ (K = (10 * B / 3)) ∧ (C = B + 90) ∧ (K = B + 140)

theorem car_count_is_150 (h : conditions B C K) : C = 150 :=
by
  sorry

end car_count_is_150_l1763_176312


namespace area_of_square_BDEF_l1763_176342

noncomputable def right_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
∃ (AB BC AC : ℝ), AB = 15 ∧ BC = 20 ∧ AC = Real.sqrt (AB^2 + BC^2)

noncomputable def is_square (B D E F : Type*) [MetricSpace B] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
∃ (BD DE EF FB : ℝ), BD = DE ∧ DE = EF ∧ EF = FB

noncomputable def height_of_triangle (E H M : Type*) [MetricSpace E] [MetricSpace H] [MetricSpace M] : Prop :=
∃ (EH : ℝ), EH = 2

theorem area_of_square_BDEF (A B C D E F H M N : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F]
  [MetricSpace H] [MetricSpace M] [MetricSpace N]
  (H1 : right_triangle A B C)
  (H2 : is_square B D E F)
  (H3 : height_of_triangle E H M) :
  ∃ (area : ℝ), area = 100 :=
by
  sorry

end area_of_square_BDEF_l1763_176342


namespace not_basic_logical_structure_l1763_176389

def basic_structures : Set String := {"Sequential structure", "Conditional structure", "Loop structure"}

theorem not_basic_logical_structure : "Operational structure" ∉ basic_structures := by
  sorry

end not_basic_logical_structure_l1763_176389


namespace sum_of_ratios_is_3_or_neg3_l1763_176352

theorem sum_of_ratios_is_3_or_neg3 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a / b + b / c + c / a : ℚ).den = 1 ) 
  (h5 : (b / a + c / b + a / c : ℚ).den = 1) :
  (a / b + b / c + c / a = 3 ∨ a / b + b / c + c / a = -3) ∧ 
  (b / a + c / b + a / c = 3 ∨ b / a + c / b + a / c = -3) := 
sorry

end sum_of_ratios_is_3_or_neg3_l1763_176352


namespace solve_for_a_l1763_176323

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end solve_for_a_l1763_176323


namespace total_sum_of_ages_l1763_176378

theorem total_sum_of_ages (Y : ℕ) (interval : ℕ) (age1 age2 age3 age4 age5 : ℕ)
  (h1 : Y = 2) 
  (h2 : interval = 8) 
  (h3 : age1 = Y) 
  (h4 : age2 = Y + interval) 
  (h5 : age3 = Y + 2 * interval) 
  (h6 : age4 = Y + 3 * interval) 
  (h7 : age5 = Y + 4 * interval) : 
  age1 + age2 + age3 + age4 + age5 = 90 := 
by
  sorry

end total_sum_of_ages_l1763_176378


namespace overlap_difference_l1763_176305

namespace GeometryBiology

noncomputable def total_students : ℕ := 350
noncomputable def geometry_students : ℕ := 210
noncomputable def biology_students : ℕ := 175

theorem overlap_difference : 
    let max_overlap := min geometry_students biology_students;
    let min_overlap := geometry_students + biology_students - total_students;
    max_overlap - min_overlap = 140 := 
by
  sorry

end GeometryBiology

end overlap_difference_l1763_176305


namespace CarlaDailyItems_l1763_176321

theorem CarlaDailyItems (leaves bugs days : ℕ) 
  (h_leaves : leaves = 30) 
  (h_bugs : bugs = 20) 
  (h_days : days = 10) : 
  (leaves + bugs) / days = 5 := 
by 
  sorry

end CarlaDailyItems_l1763_176321


namespace number_of_valid_pairs_l1763_176306

theorem number_of_valid_pairs :
  ∃ (n : ℕ), n = 4950 ∧ ∀ (x y : ℕ), 
  1 ≤ x ∧ x < y ∧ y ≤ 200 ∧ 
  (Complex.I ^ x + Complex.I ^ y).im = 0 → n = 4950 :=
sorry

end number_of_valid_pairs_l1763_176306


namespace find_k_l1763_176381

theorem find_k (k : ℝ) : (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 :=
by
  sorry

end find_k_l1763_176381


namespace garrison_men_initial_l1763_176309

theorem garrison_men_initial (M : ℕ) (P : ℕ):
  (P = M * 40) →
  (P / 2 = (M + 2000) * 10) →
  M = 2000 :=
by
  intros h1 h2
  sorry

end garrison_men_initial_l1763_176309


namespace product_of_decimals_l1763_176372

def x : ℝ := 0.8
def y : ℝ := 0.12

theorem product_of_decimals : x * y = 0.096 :=
by
  sorry

end product_of_decimals_l1763_176372


namespace problem_l1763_176395

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : odd_function f
axiom f_property : ∀ x : ℝ, f (x + 2) = -f x
axiom f_at_1 : f 1 = 8

theorem problem : f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end problem_l1763_176395


namespace no_finite_set_A_exists_l1763_176375

theorem no_finite_set_A_exists (A : Set ℕ) (h : Finite A ∧ ∀ a ∈ A, 2 * a ∈ A ∨ a / 3 ∈ A) : False :=
sorry

end no_finite_set_A_exists_l1763_176375


namespace fraction_conversion_integer_l1763_176390

theorem fraction_conversion_integer (x : ℝ) :
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1 →
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 :=
by sorry

end fraction_conversion_integer_l1763_176390


namespace smallest_expression_l1763_176348

theorem smallest_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (y / x = 1 / 2) ∧ (y / x < x + y) ∧ (y / x < x * y) ∧ (y / x < x - y) ∧ (y / x < x / y) :=
by
  -- The proof is to be filled by the user
  sorry

end smallest_expression_l1763_176348


namespace find_a_l1763_176338

-- Defining the curve y and its derivative y'
def y (x : ℝ) (a : ℝ) : ℝ := x^4 + a * x^2 + 1
def y' (x : ℝ) (a : ℝ) : ℝ := 4 * x^3 + 2 * a * x

theorem find_a (a : ℝ) : 
  y' (-1) a = 8 -> a = -6 := 
by
  -- proof here
  sorry

end find_a_l1763_176338


namespace fraction_addition_l1763_176361

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end fraction_addition_l1763_176361


namespace find_y_l1763_176376

open Real

structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

def parallel (v₁ v₂ : Vec3) : Prop := ∃ s : ℝ, v₁ = ⟨s * v₂.x, s * v₂.y, s * v₂.z⟩

def orthogonal (v₁ v₂ : Vec3) : Prop := (v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z) = 0

noncomputable def correct_y (x y : Vec3) : Vec3 :=
  ⟨(8 : ℝ) - 2 * (2 : ℝ), (-4 : ℝ) - 2 * (2 : ℝ), (2 : ℝ) - 2 * (2 : ℝ)⟩

theorem find_y :
  ∀ (x y : Vec3),
    (x.x + y.x = 8) ∧ (x.y + y.y = -4) ∧ (x.z + y.z = 2) →
    (parallel x ⟨2, 2, 2⟩) →
    (orthogonal y ⟨1, -1, 0⟩) →
    y = ⟨4, -8, -2⟩ :=
by
  intros x y Hxy Hparallel Horthogonal
  sorry

end find_y_l1763_176376


namespace find_prime_squares_l1763_176382

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end find_prime_squares_l1763_176382


namespace proposition_does_not_hold_at_2_l1763_176330

variable (P : ℕ+ → Prop)
open Nat

theorem proposition_does_not_hold_at_2
  (h₁ : ¬ P 3)
  (h₂ : ∀ k : ℕ+, P k → P (k + 1)) :
  ¬ P 2 :=
by
  sorry

end proposition_does_not_hold_at_2_l1763_176330


namespace graph_equiv_l1763_176383

theorem graph_equiv {x y : ℝ} :
  (x^3 - 2 * x^2 * y + x * y^2 - 2 * y^3 = 0) ↔ (x = 2 * y) :=
sorry

end graph_equiv_l1763_176383


namespace janelle_total_marbles_l1763_176334

def initial_green_marbles := 26
def bags_of_blue_marbles := 12
def marbles_per_bag := 15
def gift_red_marbles := 7
def gift_green_marbles := 9
def gift_blue_marbles := 12
def gift_red_marbles_given := 3
def returned_blue_marbles := 8

theorem janelle_total_marbles :
  let total_green := initial_green_marbles - gift_green_marbles
  let total_blue := (bags_of_blue_marbles * marbles_per_bag) - gift_blue_marbles + returned_blue_marbles
  let total_red := gift_red_marbles - gift_red_marbles_given
  total_green + total_blue + total_red = 197 :=
by
  sorry

end janelle_total_marbles_l1763_176334


namespace rectangle_area_l1763_176384

theorem rectangle_area (sqr_area : ℕ) (rect_width rect_length : ℕ) (h1 : sqr_area = 25)
    (h2 : rect_width = Int.sqrt sqr_area) (h3 : rect_length = 2 * rect_width) :
    rect_width * rect_length = 50 := by
  sorry

end rectangle_area_l1763_176384


namespace current_population_correct_l1763_176370

def initial_population : ℕ := 4079
def percentage_died : ℕ := 5
def percentage_left : ℕ := 15

def calculate_current_population (initial_population : ℕ) (percentage_died : ℕ) (percentage_left : ℕ) : ℕ :=
  let died := (initial_population * percentage_died) / 100
  let remaining_after_bombardment := initial_population - died
  let left := (remaining_after_bombardment * percentage_left) / 100
  remaining_after_bombardment - left

theorem current_population_correct : calculate_current_population initial_population percentage_died percentage_left = 3295 :=
  by
  unfold calculate_current_population
  sorry

end current_population_correct_l1763_176370


namespace minimum_value_of_eccentricity_sum_l1763_176311

variable {a b m n c : ℝ} (ha : a > b) (hb : b > 0) (hm : m > 0) (hn : n > 0)
variable {e1 e2 : ℝ}

theorem minimum_value_of_eccentricity_sum 
  (h_equiv : a^2 + m^2 = 2 * c^2) 
  (e1_def : e1 = c / a) 
  (e2_def : e2 = c / m) : 
  (2 * e1^2 + (e2^2) / 2) = (9 / 4) :=
sorry

end minimum_value_of_eccentricity_sum_l1763_176311


namespace polynomial_coefficients_l1763_176354

theorem polynomial_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (2 * x - 1) * ((x + 1) ^ 7) = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + 
  (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 + (a 7) * x^7 + (a 8) * x^8) →
  (a 0 = -1) ∧
  (a 0 + a 2 + a 4 + a 6 + a 8 = 64) ∧
  (a 1 + 2 * (a 2) + 3 * (a 3) + 4 * (a 4) + 5 * (a 5) + 6 * (a 6) + 7 * (a 7) + 8 * (a 8) = 704) := by
  sorry

end polynomial_coefficients_l1763_176354


namespace subset_A_has_only_one_element_l1763_176310

theorem subset_A_has_only_one_element (m : ℝ) :
  (∀ x y, (mx^2 + 2*x + 1 = 0) → (mx*y^2 + 2*y + 1 = 0) → x = y) →
  (m = 0 ∨ m = 1) :=
by
  sorry

end subset_A_has_only_one_element_l1763_176310


namespace total_fencing_l1763_176304

open Real

def playground_side_length : ℝ := 27
def garden_length : ℝ := 12
def garden_width : ℝ := 9
def flower_bed_radius : ℝ := 5
def sandpit_side1 : ℝ := 7
def sandpit_side2 : ℝ := 10
def sandpit_side3 : ℝ := 13

theorem total_fencing : 
    4 * playground_side_length + 
    2 * (garden_length + garden_width) + 
    2 * Real.pi * flower_bed_radius + 
    (sandpit_side1 + sandpit_side2 + sandpit_side3) = 211.42 := 
    by sorry

end total_fencing_l1763_176304


namespace cost_price_computer_table_l1763_176358

noncomputable def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem cost_price_computer_table (SP : ℝ) (CP : ℝ) (h : SP = 7967) (h2 : SP = 1.24 * CP) : 
  approx_eq CP 6424 0.01 :=
by
  sorry

end cost_price_computer_table_l1763_176358


namespace sum_of_seven_consecutive_integers_l1763_176301

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l1763_176301


namespace cubic_root_conditions_l1763_176351

-- Define the cubic polynomial
def cubic (a b : ℝ) (x : ℝ) : ℝ := x^3 + a * x + b

-- Define a predicate for the cubic equation having exactly one real root
def has_one_real_root (a b : ℝ) : Prop :=
  ∀ y : ℝ, cubic a b y = 0 → ∃! x : ℝ, cubic a b x = 0

-- Theorem statement
theorem cubic_root_conditions (a b : ℝ) :
  (a = -3 ∧ b = -3) ∨ (a = -3 ∧ b > 2) ∨ (a = 0 ∧ b = 2) → has_one_real_root a b :=
sorry

end cubic_root_conditions_l1763_176351


namespace time_per_window_l1763_176336

-- Definitions of the given conditions
def total_windows : ℕ := 10
def installed_windows : ℕ := 6
def remaining_windows := total_windows - installed_windows
def total_hours : ℕ := 20
def hours_per_window := total_hours / remaining_windows

-- The theorem we need to prove
theorem time_per_window : hours_per_window = 5 := by
  -- This is where the proof would go
  sorry

end time_per_window_l1763_176336


namespace cos_double_angle_l1763_176343

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 5) : Real.cos (2 * α) = 23 / 25 :=
sorry

end cos_double_angle_l1763_176343


namespace calculate_division_l1763_176391

theorem calculate_division :
  (- (3 / 4) - 5 / 9 + 7 / 12) / (- 1 / 36) = 26 := by
  sorry

end calculate_division_l1763_176391


namespace convert_110110001_to_base4_l1763_176363

def binary_to_base4_conversion (b : ℕ) : ℕ :=
  -- assuming b is the binary representation of the number to be converted
  1 * 4^4 + 3 * 4^3 + 2 * 4^2 + 0 * 4^1 + 1 * 4^0

theorem convert_110110001_to_base4 : binary_to_base4_conversion 110110001 = 13201 :=
  sorry

end convert_110110001_to_base4_l1763_176363


namespace total_distance_of_journey_l1763_176364

variables (x v : ℝ)
variable (d : ℝ := 600)  -- d is the total distance given by the solution to be 600 miles

-- Define the conditions stated in the problem
def condition_1 := (x = 10 * v)  -- x = 10 * v (from first part of the solution)
def condition_2 := (3 * v * d - 90 * v = -28.5 * 3 * v)  -- 2nd condition translated from second part

theorem total_distance_of_journey : 
  ∀ (x v : ℝ), condition_1 x v ∧ condition_2 x v -> x = d :=
sorry

end total_distance_of_journey_l1763_176364


namespace extra_days_per_grade_below_b_l1763_176367

theorem extra_days_per_grade_below_b :
  ∀ (total_days lying_days grades_below_B : ℕ), 
  total_days = 26 → lying_days = 14 → grades_below_B = 4 → 
  (total_days - lying_days) / grades_below_B = 3 :=
by
  -- conditions and steps of the proof will be here
  sorry

end extra_days_per_grade_below_b_l1763_176367


namespace total_money_received_l1763_176388

-- Define the given prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def adult_tickets_sold : ℕ := 90
def child_tickets_sold : ℕ := 40

-- Define the theorem to prove the total amount received
theorem total_money_received :
  (adult_ticket_price * adult_tickets_sold + child_ticket_price * child_tickets_sold) = 1240 :=
by
  -- Proof goes here
  sorry

end total_money_received_l1763_176388


namespace sum_of_digits_joeys_age_l1763_176360

-- Given conditions
variables (C : ℕ) (J : ℕ := C + 2) (Z : ℕ := 1)

-- Define the condition that the sum of Joey's and Chloe's ages will be an integral multiple of Zoe's age.
def sum_is_multiple_of_zoe (n : ℕ) : Prop :=
  ∃ k : ℕ, (J + C) = k * Z

-- Define the problem of finding the sum of digits the first time Joey's age alone is a multiple of Zoe's age.
def sum_of_digits_first_multiple (J Z : ℕ) : ℕ :=
  (J / 10) + (J % 10)

-- The theorem we need to prove
theorem sum_of_digits_joeys_age : (sum_of_digits_first_multiple J Z = 1) :=
sorry

end sum_of_digits_joeys_age_l1763_176360


namespace find_f_f_2_l1763_176379

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_f_2 :
  f (f 2) = 14 :=
by
sorry

end find_f_f_2_l1763_176379


namespace find_numbers_l1763_176397

theorem find_numbers (N : ℕ) (a b : ℕ) :
  N = 5 * a →
  N = 7 * b →
  N = 35 ∨ N = 70 ∨ N = 105 :=
by
  sorry

end find_numbers_l1763_176397


namespace equation_of_parallel_plane_l1763_176302

theorem equation_of_parallel_plane {A B C D : ℤ} (hA : A = 3) (hB : B = -2) (hC : C = 4) (hD : D = -16)
    (point : ℝ × ℝ × ℝ) (pass_through : point = (2, -3, 1)) (parallel_plane : A * 2 + B * (-3) + C * 1 + D = 0)
    (gcd_condition : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) :
    A * 2 + B * (-3) + C + D = 0 ∧ A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by
  sorry

end equation_of_parallel_plane_l1763_176302


namespace largest_divisor_of_expression_l1763_176346

theorem largest_divisor_of_expression (n : ℤ) : ∃ k : ℤ, k = 6 ∧ (n^3 - n + 15) % k = 0 := 
by
  use 6
  sorry

end largest_divisor_of_expression_l1763_176346


namespace trigonometric_quadrant_l1763_176300

theorem trigonometric_quadrant (θ : ℝ) (h1 : Real.sin θ > Real.cos θ) (h2 : Real.sin θ * Real.cos θ < 0) : 
  (θ > π / 2) ∧ (θ < π) :=
by
  sorry

end trigonometric_quadrant_l1763_176300


namespace evaluate_expression_l1763_176374

theorem evaluate_expression :
  2 ^ (0 ^ (1 ^ 9)) + ((2 ^ 0) ^ 1) ^ 9 = 2 := 
sorry

end evaluate_expression_l1763_176374


namespace second_set_parallel_lines_l1763_176392

theorem second_set_parallel_lines (n : ℕ) (h : 7 * (n - 1) = 784) : n = 113 := 
by
  sorry

end second_set_parallel_lines_l1763_176392


namespace mutually_exclusive_necessary_for_complementary_l1763_176365

variables {Ω : Type} -- Define the sample space type
variables (A1 A2 : Ω → Prop) -- Define the events as predicates over the sample space

-- Define mutually exclusive events
def mutually_exclusive (A1 A2 : Ω → Prop) : Prop :=
∀ ω, A1 ω → ¬ A2 ω

-- Define complementary events
def complementary (A1 A2 : Ω → Prop) : Prop :=
∀ ω, (A1 ω ↔ ¬ A2 ω)

-- The proof problem: Statement 1 is a necessary but not sufficient condition for Statement 2
theorem mutually_exclusive_necessary_for_complementary (A1 A2 : Ω → Prop) :
  (mutually_exclusive A1 A2) → (complementary A1 A2) → (mutually_exclusive A1 A2) ∧ ¬ (complementary A1 A2 → mutually_exclusive A1 A2) :=
sorry

end mutually_exclusive_necessary_for_complementary_l1763_176365


namespace largest_real_solution_sum_l1763_176339

theorem largest_real_solution_sum (d e f : ℕ) (x : ℝ) (h : d = 13 ∧ e = 61 ∧ f = 0) : 
  (∃ d e f : ℕ, d + e + f = 74) ↔ 
  (n : ℝ) * n = (x - d)^2 ∧ 
  (∀ x : ℝ, 
    (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 13 * x - 6 → 
    n = x) :=
sorry

end largest_real_solution_sum_l1763_176339


namespace find_X_l1763_176337

theorem find_X : ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ X = 0.3 :=
by
  sorry

end find_X_l1763_176337


namespace find_integer_pairs_l1763_176347

theorem find_integer_pairs :
  {p : ℤ × ℤ | p.1 * (p.1 + 1) * (p.1 + 7) * (p.1 + 8) = p.2^2} =
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (0, 0), (-8, 0), (-4, -12), (-4, 12), (-1, 0), (-7, 0)} :=
sorry

end find_integer_pairs_l1763_176347


namespace bobs_fruit_drink_cost_l1763_176307

theorem bobs_fruit_drink_cost
  (cost_soda : ℕ)
  (cost_hamburger : ℕ)
  (cost_sandwiches : ℕ)
  (bob_total_spent same_amount : ℕ)
  (andy_spent_eq : same_amount = cost_soda + 2 * cost_hamburger)
  (andy_bob_spent_eq : same_amount = bob_total_spent)
  (bob_sandwich_cost_eq : cost_sandwiches = 3)
  (andy_spent_eq_total : cost_soda = 1)
  (andy_burger_cost : cost_hamburger = 2)
  : bob_total_spent - cost_sandwiches = 2 :=
by
  sorry

end bobs_fruit_drink_cost_l1763_176307


namespace average_weight_estimation_exclude_friend_l1763_176399

theorem average_weight_estimation_exclude_friend
    (w : ℝ)
    (H1 : 62.4 < w ∧ w < 72.1)
    (H2 : 60.3 < w ∧ w < 70.6)
    (H3 : w ≤ 65.9)
    (H4 : 63.7 < w ∧ w < 66.3)
    (H5 : 75.0 ≤ w ∧ w ≤ 78.5) :
    False ∧ ((63.7 < w ∧ w ≤ 65.9) → (w = 64.8)) :=
by
  sorry

end average_weight_estimation_exclude_friend_l1763_176399


namespace sum_first_19_terms_l1763_176341

variable {α : Type} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + n * d

def sum_of_arithmetic_sequence (a d : α) (n : ℕ) : α := (n : α) / 2 * (2 * a + (n - 1) * d)

theorem sum_first_19_terms (a d : α) 
  (h1 : ∀ n, arithmetic_sequence a d (2 + n) + arithmetic_sequence a d (16 + n) = 10)
  (S19 : α) :
  sum_of_arithmetic_sequence a d 19 = 95 := by
  sorry

end sum_first_19_terms_l1763_176341


namespace problem_part1_problem_part2_l1763_176380

noncomputable def quadratic_roots_conditions (x1 x2 m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1)

noncomputable def existence_of_m (x1 x2 : ℝ) (m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1) ∧ ((x1 - 1) * (x2 - 1) = 6 / (m - 5))

theorem problem_part1 : 
  ∃ x2 m, quadratic_roots_conditions 1 x2 m :=
sorry

theorem problem_part2 :
  ∃ m, ∃ x2, existence_of_m 1 x2 m ∧ m ≤ 5 :=
sorry

end problem_part1_problem_part2_l1763_176380


namespace eq_inf_solutions_l1763_176359

theorem eq_inf_solutions (a b : ℝ) : 
    (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -(4 / 3) * a := by
  sorry

end eq_inf_solutions_l1763_176359


namespace trajectory_of_moving_circle_l1763_176369

def circle1 (x y : ℝ) := (x + 4) ^ 2 + y ^ 2 = 2
def circle2 (x y : ℝ) := (x - 4) ^ 2 + y ^ 2 = 2

theorem trajectory_of_moving_circle (x y : ℝ) : 
  (x = 0) ∨ (x ^ 2 / 2 - y ^ 2 / 14 = 1) := 
  sorry

end trajectory_of_moving_circle_l1763_176369


namespace marcia_project_hours_l1763_176353

theorem marcia_project_hours (minutes_spent : ℕ) (minutes_per_hour : ℕ) 
  (h1 : minutes_spent = 300) 
  (h2 : minutes_per_hour = 60) : 
  (minutes_spent / minutes_per_hour) = 5 :=
by
  sorry

end marcia_project_hours_l1763_176353


namespace smallest_m_for_divisibility_l1763_176318

theorem smallest_m_for_divisibility : 
  ∃ (m : ℕ), 2^1990 ∣ 1989^m - 1 ∧ m = 2^1988 := 
sorry

end smallest_m_for_divisibility_l1763_176318


namespace tray_contains_40_brownies_l1763_176326

-- Definitions based on conditions
def tray_length : ℝ := 24
def tray_width : ℝ := 15
def brownie_length : ℝ := 3
def brownie_width : ℝ := 3

-- The mathematical statement to prove
theorem tray_contains_40_brownies :
  (tray_length * tray_width) / (brownie_length * brownie_width) = 40 :=
by
  sorry

end tray_contains_40_brownies_l1763_176326


namespace sqrt_expression_identity_l1763_176349

theorem sqrt_expression_identity :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := 
by
  sorry

end sqrt_expression_identity_l1763_176349


namespace coffee_machine_price_l1763_176371

noncomputable def original_machine_price : ℝ :=
  let coffees_prior_cost_per_day := 2 * 4
  let new_coffees_cost_per_day := 3
  let daily_savings := coffees_prior_cost_per_day - new_coffees_cost_per_day
  let total_savings := 36 * daily_savings
  let discounted_price := total_savings
  let discount := 20
  discounted_price + discount

theorem coffee_machine_price
  (coffees_prior_cost_per_day : ℝ := 2 * 4)
  (new_coffees_cost_per_day : ℝ := 3)
  (daily_savings : ℝ := coffees_prior_cost_per_day - new_coffees_cost_per_day)
  (total_savings : ℝ := 36 * daily_savings)
  (discounted_price : ℝ := total_savings)
  (discount : ℝ := 20) :
  original_machine_price = 200 :=
by
  sorry

end coffee_machine_price_l1763_176371


namespace scientific_notation_of_1500_l1763_176320

theorem scientific_notation_of_1500 :
  (1500 : ℝ) = 1.5 * 10^3 :=
sorry

end scientific_notation_of_1500_l1763_176320


namespace number_of_integer_pairs_l1763_176303

theorem number_of_integer_pairs (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_ineq : m^2 + m * n < 30) :
  ∃ k : ℕ, k = 48 :=
sorry

end number_of_integer_pairs_l1763_176303


namespace complement_U_A_l1763_176319

-- Definitions of U and A based on problem conditions
def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 2}

-- Definition of the complement in Lean
def complement (A B : Set ℤ) : Set ℤ := {x | x ∈ A ∧ x ∉ B}

-- The main statement to be proved
theorem complement_U_A :
  complement U A = {1, 3} :=
sorry

end complement_U_A_l1763_176319


namespace distance_between_foci_of_ellipse_l1763_176335

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ),
  -- Condition: axes are parallel to the coordinate axes (implicitly given by tangency points).
  a = 3 ∧
  b = 2 ∧
  c = Real.sqrt (a^2 - b^2) ∧
  2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_ellipse_l1763_176335


namespace total_cost_of_crayons_l1763_176308

-- Definition of the initial conditions
def usual_price : ℝ := 2.5
def discount_rate : ℝ := 0.15
def packs_initial : ℕ := 4
def packs_to_buy : ℕ := 2

-- Calculate the discounted price for one pack
noncomputable def discounted_price : ℝ :=
  usual_price - (usual_price * discount_rate)

-- Calculate the total cost of packs after purchase and validate it
theorem total_cost_of_crayons :
  (packs_initial * usual_price) + (packs_to_buy * discounted_price) = 14.25 :=
by
  sorry

end total_cost_of_crayons_l1763_176308


namespace max_area_ABC_l1763_176315

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end max_area_ABC_l1763_176315


namespace unique_B_squared_l1763_176385

theorem unique_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) : 
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B2 = B * B :=
sorry

end unique_B_squared_l1763_176385


namespace nat_numbers_equal_if_divisible_l1763_176313

theorem nat_numbers_equal_if_divisible
  (a b : ℕ)
  (h : ∀ n : ℕ, ∃ m : ℕ, n ≠ m → (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end nat_numbers_equal_if_divisible_l1763_176313


namespace given_condition_required_solution_l1763_176393

-- Define the polynomial f.
noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

-- Given condition
theorem given_condition (x : ℝ) : f (x^2 + 2) = x^4 + 5 * x^2 := by sorry

-- Proving the required equivalence
theorem required_solution (x : ℝ) : f (x^2 - 2) = x^4 - 3 * x^2 - 4 := by sorry

end given_condition_required_solution_l1763_176393


namespace flashlight_distance_difference_l1763_176386

/--
Veronica's flashlight can be seen from 1000 feet. Freddie's flashlight can be seen from a distance
three times that of Veronica's flashlight. Velma's flashlight can be seen from a distance 2000 feet
less than 5 times Freddie's flashlight distance. We want to prove that Velma's flashlight can be seen 
12000 feet farther than Veronica's flashlight.
-/
theorem flashlight_distance_difference :
  let v_d := 1000
  let f_d := 3 * v_d
  let V_d := 5 * f_d - 2000
  V_d - v_d = 12000 := by
    sorry

end flashlight_distance_difference_l1763_176386


namespace distinct_fib_sum_2017_l1763_176328

-- Define the Fibonacci sequence as given.
def fib : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => (fib (n+1)) + (fib n)

-- Define the predicate for representing a number as a sum of distinct Fibonacci numbers.
def can_be_written_as_sum_of_distinct_fibs (n : ℕ) : Prop :=
  ∃ s : Finset ℕ, (s.sum fib = n) ∧ (∀ (i j : ℕ), i ≠ j → i ∉ s → j ∉ s)

theorem distinct_fib_sum_2017 : ∃! s : Finset ℕ, s.sum fib = 2017 ∧ (∀ (i j : ℕ), i ≠ j → i ≠ j → i ∉ s → j ∉ s) :=
sorry

end distinct_fib_sum_2017_l1763_176328
